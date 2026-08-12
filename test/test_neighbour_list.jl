using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm, dot
using Random

# ---------------------------------------------------------------------------
# NeighbourListKA — explicit neighbour list
# (docs/gpu-migration-plan.md items 16-17, "Explicit neighbour list").
#
# Reachable via SystemInteraction's public `neighbour_list=true` kwarg
# (requires `ka=true`) or the internal `mode` escape hatch
# (Grasph.NeighbourListKA()) — both are exercised below. Runs its
# @kernel-based build+consume kernels on KA.CPU() here (no GPU needed —
# backend dispatch is purely by array type, see KAKernels.jl), giving a
# cheap, deterministic Tier-1 correctness oracle exactly like test_ka_cpu.jl
# does for OnesidedKA; see test_gpu_cuda.jl for the real-CUDA counterpart.
#
# Scope: self-interaction and all three coupled write shapes (WritesA/
# WritesB/WritesBoth), 2D and 3D.
# ---------------------------------------------------------------------------

function _nlist_random_fluid(rng, n, ndims; L=1.0)
    ps = FluidParticleSystem("fluid", n, ndims, 1.0, 10.0; source_v = zeros(ndims))
    for i in 1:n
        ps.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i] = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
    end
    ps.rho .= 1000.0 .+ 20 .* (rand(rng, n) .- 0.5)
    ps.p   .= 100.0 .* rand(rng, n)
    fill!(ps.dvdt, zero(SVector{ndims,Float64}))
    ps.drhodt .= 0.0
    return ps
end

function _nlist_random_boundary(rng, n, ndims; L=1.0)
    inner = BasicParticleSystem("bnd", n, ndims, 1.0, 10.0)
    for i in 1:n
        inner.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
    end
    inner.rho .= 1000.0
    fill!(inner.v, zero(SVector{ndims,Float64}))
    return inner
end

_nlist_sortbufs(ps) = (Vector{Int}(undef, ps.n), Vector{UInt64}(undef, ps.n), Grasph._make_sort_scratch(ps))

# Same tight-rtol reasoning as test_ka_cpu.jl's _kacpu_assert_*: meant to
# catch real bugs, not just pass.
function _nlist_assert_dvdt_close(dvdt_ref, dvdt_nl; rtol=1e-12)
    scale = max(maximum(norm.(dvdt_ref)), 1.0)
    @test maximum(norm.(dvdt_ref .- dvdt_nl)) < rtol * scale
end

function _nlist_assert_drhodt_close(drhodt_ref, drhodt_nl; rtol=1e-12)
    scale = max(maximum(abs.(drhodt_ref)), 1.0)
    @test maximum(abs.(drhodt_ref .- drhodt_nl)) < rtol * scale
end

# A dambreak-shaped fixture, mirroring test_verlet_skin.jl's _verlet_dambreak
# exactly (same regular-grid-not-random-cloud reasoning — see that file's
# header comment), except `mode` replaces `ka` so it can select
# NeighbourListKA instead of OnesidedKA.
function _nlist_dambreak(nfx; verlet_skin=0.0, mode=nothing, speed=(0.0, 0.0))
    dx = 0.5
    h  = 1.2 * dx
    rho0 = 1000.0
    c  = 10.0 * sqrt(2.0 * 9.81 * 25.0)

    nfy = nfx
    nbx = nfx + 8
    nby = nfx + 8

    n_fluid = nfx * nfy
    mass = rho0 * dx * dx
    fluid = FluidParticleSystem("fluid", n_fluid, 2, mass, c;
                                 source_v = [0.0, -9.81], state_updater = TaitEOSUpdater(rho0))
    k = 1
    for i in 0:nfx-1, j in 0:nfy-1
        fluid.x[k] = SVector((i + 0.5) * dx, (j + 3.5) * dx)
        k += 1
    end
    fill!(fluid.v, SVector(speed...))
    fluid.rho .= rho0
    update_state!(fluid, 1)

    n_b = 2 * (nbx + nby) + 4
    boundary = BasicParticleSystem("boundary", n_b, 2, mass, c)
    k = 1
    for i in -1:nbx
        boundary.x[k] = SVector((i + 0.5) * dx, -0.5 * dx); k += 1
    end
    for i in -1:nbx
        boundary.x[k] = SVector((i + 0.5) * dx, nby * dx + 0.5 * dx); k += 1
    end
    for j in 0:nby-1
        boundary.x[k] = SVector(-0.5 * dx, (j + 0.5) * dx); k += 1
    end
    for j in 0:nby-1
        boundary.x[k] = SVector(nbx * dx + 0.5 * dx, (j + 0.5) * dx); k += 1
    end
    boundary.rho .= rho0
    fill!(boundary.v, zero(SVector{2,Float64}))

    kernel = CubicSplineKernel(h; ndims=2)
    static_boundary = StaticBoundarySystem(boundary, dx)
    fi  = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true, mode=mode)
    fbi = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid, static_boundary; onesided=true, mode=mode)

    integ = LeapFrogTimeIntegrator([fluid, boundary], [fi, fbi]; verlet_skin=verlet_skin)
    return integ, fluid, boundary
end

# Compare by particle id, tight relative tolerance — same reasoning as
# test_verlet_skin.jl's _assert_close (index alignment isn't guaranteed to
# match between runs that re-sort a different number of times).
function _nlist_assert_close(fluid_a, fluid_b; rtol=1e-9, atol=1e-10)
    perm_a = sortperm(fluid_a.id)
    perm_b = sortperm(fluid_b.id)
    @test fluid_a.id[perm_a] == fluid_b.id[perm_b]
    dx_max = maximum(norm.(fluid_a.x[perm_a] .- fluid_b.x[perm_b]))
    dv_max = maximum(norm.(fluid_a.v[perm_a] .- fluid_b.v[perm_b]))
    scale  = maximum(norm.(fluid_a.x[perm_a]))
    @test dx_max <= atol + rtol * scale
    @test dv_max <= atol + rtol * max(scale, 1.0)
end

# Synthetic WritesB-only test pfn, distinct name from test_onesided_sweep.jl's
# own _ReverseOnlyTestPfn (both files are `include`d by runtests.jl, so
# redefining the same struct name would clash). Same formula/shape as that
# file's precedent — see its "Phase 2: reverse-sweep infrastructure" comment
# for the full rationale; here it's used to exercise NeighbourListKA's
# reverse build+consume path against the already-tested OnesidedKA reverse
# pass as oracle, not against a brute-force reference.
@inline _nl_test_pfn_contribution(ps_a, ps_b, i, j, dx, gx, w) =
    (dvdt = (ps_b.mass * w / ps_a.rho[i]) * gx, drhodt = ps_b.mass * dot(gx, ps_a.v[i]))
@inline _nl_test_pfn_zero(ps_a, i) = (dvdt = zero(eltype(ps_a.dvdt)), drhodt = zero(eltype(ps_a.drhodt)))

struct _NLReverseOnlyTestPfn end
Grasph._onesided_shape(::_NLReverseOnlyTestPfn, ps_a, ps_b) = Grasph.WritesB()
@inline Grasph.pfn_contribution(::_NLReverseOnlyTestPfn, ps_a, ps_b, i::Int, j::Int, dx::SVector, gx::SVector, w) =
    _nl_test_pfn_contribution(ps_a, ps_b, i, j, dx, gx, w)
@inline Grasph._onesided_zero_coupled(::_NLReverseOnlyTestPfn, ps_a, ps_b, i) = _nl_test_pfn_zero(ps_a, i)

@testset "NeighbourListKA" begin

    @testset "self sweep 2D: onesided ka=true (KA.CPU()) vs NeighbourListKA (KA.CPU())" begin
        rng = MersenneTwister(201)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n, L) in ((1, 1.0), (2, 1.0), (50, 1.0), (400, 1.0), (400, 0.3), (1200, 1.0))
            ps_ref = _nlist_random_fluid(rng, n, 2; L=L)
            ps_nl  = deepcopy(ps_ref)
            si_ref = SystemInteraction(kernel, pfn, ps_ref; onesided=true, ka=true)
            si_nl  = SystemInteraction(kernel, pfn, ps_nl; mode=Grasph.NeighbourListKA())
            for (ps, si) in ((ps_ref, si_ref), (ps_nl, si_nl))
                pb, kb, sc = _nlist_sortbufs(ps)
                sort_particles!(ps, cutoff, pb, kb, sc)
                create_grid!(si)
                Grasph._maybe_build_neighbour_list!(si)
                sweep!(si)
            end
            _nlist_assert_dvdt_close(ps_ref.dvdt, ps_nl.dvdt)
            _nlist_assert_drhodt_close(ps_ref.drhodt, ps_nl.drhodt)
        end
    end

    @testset "coupled sweep 2D (WritesA, fluid+StaticBoundarySystem): onesided ka=true vs NeighbourListKA" begin
        rng = MersenneTwister(202)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n, nb, L) in ((1, 5, 1.0), (50, 30, 1.0), (400, 100, 1.0), (400, 100, 0.3))
            fluid_ref = _nlist_random_fluid(rng, n, 2; L=L)
            bnd_ref   = _nlist_random_boundary(rng, nb, 2; L=L)
            fluid_nl  = deepcopy(fluid_ref)
            bnd_nl    = deepcopy(bnd_ref)
            sb_ref = StaticBoundarySystem(bnd_ref, 0.05)
            sb_nl  = StaticBoundarySystem(bnd_nl, 0.05)
            si_ref = SystemInteraction(kernel, pfn, fluid_ref, sb_ref; onesided=true, ka=true)
            si_nl  = SystemInteraction(kernel, pfn, fluid_nl, sb_nl; mode=Grasph.NeighbourListKA())
            for (fluid, si) in ((fluid_ref, si_ref), (fluid_nl, si_nl))
                pb, kb, sc = _nlist_sortbufs(fluid)
                sort_particles!(fluid, cutoff, pb, kb, sc)
                create_grid!(si)
                Grasph._maybe_build_neighbour_list!(si)
                sweep!(si)
            end
            _nlist_assert_dvdt_close(fluid_ref.dvdt, fluid_nl.dvdt)
            _nlist_assert_drhodt_close(fluid_ref.drhodt, fluid_nl.drhodt)
        end
    end

    @testset "self sweep 3D: onesided ka=true (KA.CPU()) vs NeighbourListKA (KA.CPU())" begin
        rng = MersenneTwister(211)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=3)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n, L) in ((1, 1.0), (2, 1.0), (50, 1.0), (300, 1.0), (300, 0.3))
            ps_ref = _nlist_random_fluid(rng, n, 3; L=L)
            ps_nl  = deepcopy(ps_ref)
            si_ref = SystemInteraction(kernel, pfn, ps_ref; onesided=true, ka=true)
            si_nl  = SystemInteraction(kernel, pfn, ps_nl; onesided=true, ka=true, neighbour_list=true)
            for (ps, si) in ((ps_ref, si_ref), (ps_nl, si_nl))
                pb, kb, sc = _nlist_sortbufs(ps)
                sort_particles!(ps, cutoff, pb, kb, sc)
                create_grid!(si)
                Grasph._maybe_build_neighbour_list!(si)
                sweep!(si)
            end
            _nlist_assert_dvdt_close(ps_ref.dvdt, ps_nl.dvdt)
            _nlist_assert_drhodt_close(ps_ref.drhodt, ps_nl.drhodt)
        end
    end

    @testset "coupled sweep 3D (WritesA, fluid+StaticBoundarySystem): onesided ka=true vs NeighbourListKA" begin
        rng = MersenneTwister(212)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=3)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n, nb, L) in ((1, 5, 1.0), (50, 30, 1.0), (300, 80, 1.0), (300, 80, 0.3))
            fluid_ref = _nlist_random_fluid(rng, n, 3; L=L)
            bnd_ref   = _nlist_random_boundary(rng, nb, 3; L=L)
            fluid_nl  = deepcopy(fluid_ref)
            bnd_nl    = deepcopy(bnd_ref)
            sb_ref = StaticBoundarySystem(bnd_ref, 0.05)
            sb_nl  = StaticBoundarySystem(bnd_nl, 0.05)
            si_ref = SystemInteraction(kernel, pfn, fluid_ref, sb_ref; onesided=true, ka=true)
            si_nl  = SystemInteraction(kernel, pfn, fluid_nl, sb_nl; onesided=true, ka=true, neighbour_list=true)
            for (fluid, si) in ((fluid_ref, si_ref), (fluid_nl, si_nl))
                pb, kb, sc = _nlist_sortbufs(fluid)
                sort_particles!(fluid, cutoff, pb, kb, sc)
                create_grid!(si)
                Grasph._maybe_build_neighbour_list!(si)
                sweep!(si)
            end
            _nlist_assert_dvdt_close(fluid_ref.dvdt, fluid_nl.dvdt)
            _nlist_assert_drhodt_close(fluid_ref.drhodt, fluid_nl.drhodt)
        end
    end

    @testset "coupled reverse sweep (WritesB): onesided ka=true (reverse) vs NeighbourListKA (reverse)" begin
        rng = MersenneTwister(213)
        pfn = _NLReverseOnlyTestPfn()
        for ndims in (2, 3)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=ndims)
            cutoff = kernel.interaction_length
            ps_a_ref = _nlist_random_fluid(rng, 220, ndims; L=1.0)
            ps_b_ref = _nlist_random_fluid(rng, 160, ndims; L=1.0)
            ps_a_nl  = deepcopy(ps_a_ref)
            ps_b_nl  = deepcopy(ps_b_ref)

            si_ref = SystemInteraction(kernel, pfn, ps_a_ref, ps_b_ref; onesided=true, ka=true)
            si_nl  = SystemInteraction(kernel, pfn, ps_a_nl, ps_b_nl; onesided=true, ka=true, neighbour_list=true)
            for (ps_a, ps_b, si) in ((ps_a_ref, ps_b_ref, si_ref), (ps_a_nl, ps_b_nl, si_nl))
                pa, ka_, sa = _nlist_sortbufs(ps_a)
                sort_particles!(ps_a, cutoff, pa, ka_, sa)
                pb, kb, sb = _nlist_sortbufs(ps_b)
                sort_particles!(ps_b, cutoff, pb, kb, sb)
                create_grid!(si)
                Grasph._maybe_build_neighbour_list!(si)
                sweep!(si)
            end
            _nlist_assert_dvdt_close(ps_b_ref.dvdt, ps_b_nl.dvdt)
            _nlist_assert_drhodt_close(ps_b_ref.drhodt, ps_b_nl.drhodt)
            # A pure WritesB() pfn must leave system_a completely untouched.
            @test all(==(zero(SVector{ndims,Float64})), ps_a_nl.dvdt)
            @test all(==(0.0), ps_a_nl.drhodt)
        end
    end

    @testset "coupled sweep (WritesBoth, FluidPfn fluid-fluid): onesided ka=true vs NeighbourListKA" begin
        rng = MersenneTwister(214)
        for ndims in (2, 3)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=ndims)
            cutoff = kernel.interaction_length
            pfn = FluidPfn(0.03, 0.0, h)  # fluid-fluid coupling is WritesBoth
            ps_a_ref = _nlist_random_fluid(rng, 180, ndims; L=1.0)
            ps_b_ref = _nlist_random_fluid(rng, 150, ndims; L=1.0)
            ps_a_nl  = deepcopy(ps_a_ref)
            ps_b_nl  = deepcopy(ps_b_ref)

            si_ref = SystemInteraction(kernel, pfn, ps_a_ref, ps_b_ref; onesided=true, ka=true)
            si_nl  = SystemInteraction(kernel, pfn, ps_a_nl, ps_b_nl; onesided=true, ka=true, neighbour_list=true)
            for (ps_a, ps_b, si) in ((ps_a_ref, ps_b_ref, si_ref), (ps_a_nl, ps_b_nl, si_nl))
                pa, ka_, sa = _nlist_sortbufs(ps_a)
                sort_particles!(ps_a, cutoff, pa, ka_, sa)
                pb, kb, sb = _nlist_sortbufs(ps_b)
                sort_particles!(ps_b, cutoff, pb, kb, sb)
                create_grid!(si)
                Grasph._maybe_build_neighbour_list!(si)
                sweep!(si)
            end
            _nlist_assert_dvdt_close(ps_a_ref.dvdt, ps_a_nl.dvdt)
            _nlist_assert_drhodt_close(ps_a_ref.drhodt, ps_a_nl.drhodt)
            _nlist_assert_dvdt_close(ps_b_ref.dvdt, ps_b_nl.dvdt)
            _nlist_assert_drhodt_close(ps_b_ref.drhodt, ps_b_nl.drhodt)
        end
    end

    @testset "public API: neighbour_list=true kwarg" begin
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        pfn = FluidPfn(0.03, 0.0, h)
        fluid = _nlist_random_fluid(MersenneTwister(215), 10, 2)

        si = SystemInteraction(kernel, pfn, fluid; onesided=true, ka=true, neighbour_list=true)
        @test Grasph._exec_mode(si) isa Grasph.NeighbourListKA

        @test_throws ArgumentError SystemInteraction(kernel, pfn, fluid; neighbour_list=true)
        @test_throws ArgumentError SystemInteraction(kernel, pfn, fluid; neighbour_list=true, onesided=true)
    end

    @testset "LeapFrog: NeighbourListKA (skin > 0) matches onesided CPU (skin = 0), tame motion" begin
        integ0, fluid0, _ = _nlist_dambreak(16; verlet_skin=0.0)
        time_integrate!(integ0, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        integ1, fluid1, _ = _nlist_dambreak(16; verlet_skin=0.1, mode=Grasph.NeighbourListKA())
        time_integrate!(integ1, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        _nlist_assert_close(fluid0, fluid1)
    end

    @testset "LeapFrog: NeighbourListKA (skin > 0) matches onesided CPU (skin = 0), fast motion" begin
        integ0, fluid0, _ = _nlist_dambreak(14; verlet_skin=0.0, speed=(3.0, 1.5))
        time_integrate!(integ0, 60, 10^9, 10^9, 0.15, nothing; print_timer=false)

        integ1, fluid1, _ = _nlist_dambreak(14; verlet_skin=0.08, mode=Grasph.NeighbourListKA(), speed=(3.0, 1.5))
        time_integrate!(integ1, 60, 10^9, 10^9, 0.15, nothing; print_timer=false)

        _nlist_assert_close(fluid0, fluid1)
    end

end
