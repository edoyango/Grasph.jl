using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm
using Random

# ---------------------------------------------------------------------------
# Tier 1 — CPU (Polyester) vs KernelAbstractions CPU() backend equivalence.
#
# Runs everywhere, no GPU needed. `ka=true` forces the KA `@kernel`-based
# sweep to run via KernelAbstractions' CPU task scheduler regardless of the
# underlying array type — unlike sort/grid (dispatched purely on
# `KA.get_backend(array)`, so their GPU-only code paths are unreachable
# without a real GPU backend; see test_gpu_cuda.jl for those). This makes the
# sweep kernels — the most complex, highest-risk code in this port — testable
# without a GPU at all.
#
# The KA sweep is a line-for-line, order-preserving transcription of the
# Polyester one-sided sweep (`@batch for i in 1:n` -> `@index(Global,
# Linear)`), so in principle it's bit-identical. In practice the 2D self/
# coupled sweeps and the cell-boundary-adjacent case DO come out bit-exact
# (asserted with `==` below), but the 3D self sweep measurably does not:
# KernelAbstractions' CPU task scheduler and Polyester's `@batch` generate
# different low-level code for the extra loop nesting level 3D adds (`for
# dx_cell, for dy_cell` vs 2D's single `for dx_cell`), which changes
# floating-point reassociation even though both run on the same CPU with
# identical source arithmetic. Measured difference: ~1-2 ulps (~2.6e-16
# relative). A tight relative-tolerance comparison is used everywhere a
# sweep result is checked, so a real logic bug (which showed up during
# development as relative differences >>1e-6, never ulp-scale) still fails
# loudly while this benign reassociation noise doesn't.
# ---------------------------------------------------------------------------

function _kacpu_random_fluid(rng, n, ndims; L=1.0)
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

function _kacpu_random_boundary(rng, n, ndims; L=1.0)
    inner = BasicParticleSystem("bnd", n, ndims, 1.0, 10.0)
    for i in 1:n
        inner.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
    end
    inner.rho .= 1000.0
    fill!(inner.v, zero(SVector{ndims,Float64}))
    return inner
end

function _kacpu_random_elastoplastic(rng, n, ndims; L=1.0, ns=(ndims == 2 ? 4 : 6))
    ps = ElastoPlasticParticleSystem("wall", n, ndims, ns, 1.0, 10.0; source_v = zeros(ndims))
    for i in 1:n
        ps.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i] = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
    end
    # Constructor already zero-fills p/stress/strain*/dvdt/drhodt; FluidSolidPfn
    # never reads the solid's own p/stress, only its v/rho/mass/c.
    ps.rho .= 2400.0 .+ 20 .* (rand(rng, n) .- 0.5)
    return ps
end

_kacpu_sortbufs(ps) = (Vector{Int}(undef, ps.n), Vector{UInt64}(undef, ps.n), Grasph._make_sort_scratch(ps))

# rtol is deliberately tight (~1e4x headroom over the measured ~2.6e-16
# worst case) — this is meant to catch real bugs, not just pass.
function _kacpu_assert_dvdt_close(dvdt_ref, dvdt_ka; rtol=1e-12)
    scale = max(maximum(norm.(dvdt_ref)), 1.0)
    @test maximum(norm.(dvdt_ref .- dvdt_ka)) < rtol * scale
end

function _kacpu_assert_drhodt_close(drhodt_ref, drhodt_ka; rtol=1e-12)
    scale = max(maximum(abs.(drhodt_ref)), 1.0)
    @test maximum(abs.(drhodt_ref .- drhodt_ka)) < rtol * scale
end

@testset "KA CPU() backend equivalence" begin

    @testset "self sweep 2D: onesided (Polyester) vs ka=true (KA.CPU())" begin
        rng = MersenneTwister(101)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n, L) in ((1, 1.0), (2, 1.0), (50, 1.0), (400, 1.0), (400, 0.3), (1200, 1.0))
            ps_ref = _kacpu_random_fluid(rng, n, 2; L=L)
            ps_ka  = deepcopy(ps_ref)
            si_ref = SystemInteraction(kernel, pfn, ps_ref; onesided=true)
            si_ka  = SystemInteraction(kernel, pfn, ps_ka; onesided=true, ka=true)
            for (ps, si) in ((ps_ref, si_ref), (ps_ka, si_ka))
                pb, kb, sc = _kacpu_sortbufs(ps)
                sort_particles!(ps, cutoff, pb, kb, sc)
                create_grid!(si)
                sweep!(si)
            end
            _kacpu_assert_dvdt_close(ps_ref.dvdt, ps_ka.dvdt)
            _kacpu_assert_drhodt_close(ps_ref.drhodt, ps_ka.drhodt)
        end
    end

    @testset "self sweep 3D: onesided vs ka=true" begin
        rng = MersenneTwister(102)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=3)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n, L) in ((60, 1.0), (350, 1.0), (350, 0.3))
            ps_ref = _kacpu_random_fluid(rng, n, 3; L=L)
            ps_ka  = deepcopy(ps_ref)
            si_ref = SystemInteraction(kernel, pfn, ps_ref; onesided=true)
            si_ka  = SystemInteraction(kernel, pfn, ps_ka; onesided=true, ka=true)
            for (ps, si) in ((ps_ref, si_ref), (ps_ka, si_ka))
                pb, kb, sc = _kacpu_sortbufs(ps)
                sort_particles!(ps, cutoff, pb, kb, sc)
                create_grid!(si)
                sweep!(si)
            end
            _kacpu_assert_dvdt_close(ps_ref.dvdt, ps_ka.dvdt)
            _kacpu_assert_drhodt_close(ps_ref.drhodt, ps_ka.drhodt)
        end
    end

    @testset "self sweep never pairs a particle with itself (ka=true)" begin
        # Mirrors test_onesided_sweep.jl's equivalent check for onesided=true:
        # a single isolated particle must not divide by zero against itself.
        # No pairs are found, so this is a pure sentinel-preservation check —
        # no floating-point computation happens, so `==` is exact here.
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        ps = _kacpu_random_fluid(MersenneTwister(2), 1, 2)
        ps.x[1] = SVector(0.5, 0.5)
        sentinel_dvdt, sentinel_drho = SVector(1.23, -4.56), 7.89
        ps.dvdt[1]   = sentinel_dvdt
        ps.drhodt[1] = sentinel_drho
        si = SystemInteraction(kernel, pfn, ps; onesided=true, ka=true)
        pb, kb, sc = _kacpu_sortbufs(ps)
        sort_particles!(ps, cutoff, pb, kb, sc)
        create_grid!(si)
        sweep!(si)
        @test ps.dvdt[1]   == sentinel_dvdt
        @test ps.drhodt[1] == sentinel_drho
    end

    @testset "coupled sweep (fluid<->StaticBoundarySystem): onesided vs ka=true" begin
        rng = MersenneTwister(103)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        for (n_fluid, n_bnd, L) in ((300, 200, 1.0), (300, 200, 0.3))
            fluid_ref = _kacpu_random_fluid(rng, n_fluid, 2; L=L)
            fluid_ka  = deepcopy(fluid_ref)
            bnd       = _kacpu_random_boundary(rng, n_bnd, 2; L=L)
            static_bnd = StaticBoundarySystem(bnd, 0.03)
            pbb, kbb, scb = _kacpu_sortbufs(bnd)
            sort_particles!(bnd, cutoff, pbb, kbb, scb)
            si_ref = SystemInteraction(kernel, pfn, fluid_ref, static_bnd; onesided=true)
            si_ka  = SystemInteraction(kernel, pfn, fluid_ka, static_bnd; onesided=true, ka=true)
            for (ps, si) in ((fluid_ref, si_ref), (fluid_ka, si_ka))
                pb, kb, sc = _kacpu_sortbufs(ps)
                sort_particles!(ps, cutoff, pb, kb, sc)
                create_grid!(si)
                sweep!(si)
            end
            _kacpu_assert_dvdt_close(fluid_ref.dvdt, fluid_ka.dvdt)
        end
    end

    @testset "coupled sweep 3D: onesided vs ka=true" begin
        rng = MersenneTwister(104)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=3)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        fluid_ref = _kacpu_random_fluid(rng, 250, 3; L=1.0)
        fluid_ka  = deepcopy(fluid_ref)
        bnd       = _kacpu_random_boundary(rng, 200, 3; L=1.0)
        static_bnd = StaticBoundarySystem(bnd, 0.03)
        pbb, kbb, scb = _kacpu_sortbufs(bnd)
        sort_particles!(bnd, cutoff, pbb, kbb, scb)
        si_ref = SystemInteraction(kernel, pfn, fluid_ref, static_bnd; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, fluid_ka, static_bnd; onesided=true, ka=true)
        for (ps, si) in ((fluid_ref, si_ref), (fluid_ka, si_ka))
            pb, kb, sc = _kacpu_sortbufs(ps)
            sort_particles!(ps, cutoff, pb, kb, sc)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(fluid_ref.dvdt, fluid_ka.dvdt)
    end

    @testset "cell-boundary-adjacent positions: onesided vs ka=true" begin
        # Positions snapped onto/either side of cell boundaries so pairs
        # straddle every neighbour offset the full-stencil scan must cover.
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        ps_ref = FluidParticleSystem("fluid", 9, 2, 1.0, 10.0; source_v = zeros(2))
        ps_ka  = FluidParticleSystem("fluid", 9, 2, 1.0, 10.0; source_v = zeros(2))
        k = 1
        for gi in -1:1, gj in -1:1
            x = SVector((gi + 0.5) * cutoff * 0.99, (gj + 0.5) * cutoff * 0.99)
            ps_ref.x[k] = x; ps_ka.x[k] = x
            k += 1
        end
        for ps in (ps_ref, ps_ka)
            fill!(ps.v, zero(SVector{2,Float64}))
            ps.rho .= 1000.0; ps.p .= 50.0
            fill!(ps.dvdt, zero(SVector{2,Float64})); ps.drhodt .= 0.0
        end
        si_ref = SystemInteraction(kernel, pfn, ps_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, ps_ka; onesided=true, ka=true)
        for (ps, si) in ((ps_ref, si_ref), (ps_ka, si_ka))
            pb, kb, sc = _kacpu_sortbufs(ps)
            sort_particles!(ps, cutoff, pb, kb, sc)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(ps_ref.dvdt, ps_ka.dvdt)
        _kacpu_assert_drhodt_close(ps_ref.drhodt, ps_ka.drhodt)
    end

    @testset "state update (TaitEOSUpdater) via update_state!, KA.CPU() path reachable" begin
        # _update_state!'s KA-kernel branch is dispatched purely on array
        # backend (unlike the sweep's ExecMode-driven ka=true), so on a
        # Vector-backed system this always takes the Polyester path — there
        # is no separate CPU code path to compare here. This test instead
        # confirms the state-update kernel and its device_view wiring compile
        # and produce correct output when explicitly run via KA.CPU(); the
        # CPU-vs-CUDA comparison lives in test_gpu_cuda.jl.
        rng = MersenneTwister(105)
        fluid = FluidParticleSystem("fluid", 80, 2, 1.0, 10.0; state_updater=TaitEOSUpdater(1000.0))
        fluid.rho .= 1000.0 .+ 10 .* randn(rng, 80)
        ref = deepcopy(fluid)
        update_state!(ref, 1)   # Polyester (::KA.CPU) path — the reference

        backend = Grasph.KA.CPU()
        ka_target = deepcopy(fluid)
        Grasph._update_state_kernel!(backend, Grasph._KA_WORKGROUP)(
            Grasph.device_view(ka_target), TaitEOSUpdater(1000.0), 0.0; ndrange = ka_target.n)
        Grasph.KA.synchronize(backend)

        @test ref.p == ka_target.p
    end

    # -----------------------------------------------------------------------
    # Reverse/WritesBoth sweep KA equivalence (gpu-migration-plan.md "Next
    # steps" item 6). `_sweep_coupled_ka_reverse!`/`_sweep_coupled_ka_dispatch!`
    # (KAKernels.jl) are the KA twin of `_sweep_coupled_onesided_reverse!`/
    # `_sweep_coupled_onesided_dispatch!` (Interaction.jl, Phase C). Reuses
    # `_MutualTestPfn`/`_ReverseOnlyTestPfn` (test-only, WritesBoth/WritesB)
    # from test_onesided_sweep.jl, plus three real production pfns whose
    # coupled system types are both device_view-ready: `FluidPfn` fluid-fluid
    # (WritesBoth, bubble.jl/bubble2.jl/bubble3.jl's shape), `FluidSolidPfn`
    # (WritesBoth, DambreakWall.jl's shape — asymmetric physics, needs two
    # narrowly-typed DeviceSystem{T,ND,Kind} methods rather than one; see
    # PairwiseFunctors.jl), and `InterpolateFieldFn` onto a virtual target
    # (WritesB, Trapdoor.jl/EP_ColumnCollapse2.jl's shape) —
    # `AbstractVirtualParticleSystem` got its own device_view in item 5.
    # `NeighborCountFn`/`InterpolateFieldFn`-onto-probe are WritesB pfns too
    # but target `ProbeParticleSystem`, which lacks a device_view (see item
    # 5's notes) — so `ka=true` isn't reachable for them regardless of
    # this sweep-infrastructure change; that's items 7-9's gap, not this
    # one's.
    # -----------------------------------------------------------------------

    @testset "coupled reverse sweep (WritesBoth, test pfn): onesided vs ka=true" begin
        rng = MersenneTwister(107)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = _MutualTestPfn()
        for (n_a, n_b, L) in ((220, 170, 1.0), (220, 170, 0.3))
            a_ref = _kacpu_random_fluid(rng, n_a, 2; L=L)
            b_ref = _kacpu_random_fluid(rng, n_b, 2; L=L)
            a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)
            si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
            si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
            for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
                pa, ka_, sa = _kacpu_sortbufs(a)
                sort_particles!(a, cutoff, pa, ka_, sa)
                pb, kb, sb = _kacpu_sortbufs(b)
                sort_particles!(b, cutoff, pb, kb, sb)
                create_grid!(si)
                sweep!(si)
            end
            _kacpu_assert_dvdt_close(a_ref.dvdt, a_ka.dvdt)
            _kacpu_assert_drhodt_close(a_ref.drhodt, a_ka.drhodt)
            _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
            _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
        end
    end

    @testset "coupled reverse sweep (WritesB, test pfn): onesided vs ka=true" begin
        rng = MersenneTwister(108)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = _ReverseOnlyTestPfn()
        a_ref = _kacpu_random_fluid(rng, 250, 2; L=1.0)
        b_ref = _kacpu_random_fluid(rng, 180, 2; L=1.0)
        a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)
        si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
        for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
            pa, ka_, sa = _kacpu_sortbufs(a)
            sort_particles!(a, cutoff, pa, ka_, sa)
            pb, kb, sb = _kacpu_sortbufs(b)
            sort_particles!(b, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
        _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
        # A pure WritesB() pfn must leave system_a untouched on both paths.
        @test all(==(zero(SVector{2,Float64})), a_ref.dvdt)
        @test all(==(zero(SVector{2,Float64})), a_ka.dvdt)
    end

    @testset "reverse sweep (WritesB) cell-boundary-adjacent positions: onesided vs ka=true" begin
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = _ReverseOnlyTestPfn()

        function _mkpair(shift)
            ps_a = FluidParticleSystem("fluid_a", 9, 2, 1.0, 10.0; source_v = zeros(2))
            ps_b = FluidParticleSystem("fluid_b", 9, 2, 1.0, 10.0; source_v = zeros(2))
            k = 1
            for gi in -1:1, gj in -1:1
                x = SVector((gi + 0.5) * cutoff * 0.99, (gj + 0.5) * cutoff * 0.99)
                ps_a.x[k] = x
                ps_b.x[k] = x + shift
                k += 1
            end
            for ps in (ps_a, ps_b)
                fill!(ps.v, zero(SVector{2,Float64}))
                ps.rho .= 1000.0; ps.p .= 50.0
                fill!(ps.dvdt, zero(SVector{2,Float64})); ps.drhodt .= 0.0
            end
            return ps_a, ps_b
        end

        shift = SVector(0.01 * cutoff, -0.01 * cutoff)
        a_ref, b_ref = _mkpair(shift)
        a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)

        si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
        for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
            pa, ka_, sa = _kacpu_sortbufs(a)
            sort_particles!(a, cutoff, pa, ka_, sa)
            pb, kb, sb = _kacpu_sortbufs(b)
            sort_particles!(b, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
        _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
        @test all(==(zero(SVector{2,Float64})), a_ka.dvdt)
        @test all(==(0.0), a_ka.drhodt)
    end

    @testset "coupled reverse sweep (FluidPfn fluid-fluid, WritesBoth): onesided vs ka=true" begin
        # bubble.jl/bubble2.jl/bubble3.jl's real two-phase coupling shape.
        # FluidPfn's fluid-fluid pfn_contribution/_onesided_zero_coupled
        # methods (PairwiseFunctors.jl) are typed on the CONCRETE
        # FluidParticleSystem{T,ND} on *both* sides, to disambiguate them
        # from FluidPfn's other coupled methods (which all key off a
        # specific wrapper type — StaticBoundarySystem, DynamicBoundarySystem,
        # Union{Ghost,Virtual} — on ps_b instead). device_view used to erase
        # that concrete identity entirely (every "bare" system type collapsed
        # to the same generic DeviceSystem), which MethodError'd here rather
        # than silently computing with the wrong dispatch — this was a known,
        # pinned-down gap (see the previous version of this test and item 6's
        # writeup in docs/gpu-migration-plan.md). Fixed by giving DeviceSystem
        # a phantom `Kind` type parameter (DeviceViews.jl) recording which
        # concrete host struct produced the view, and adding a
        # DeviceSystem{T,ND,FluidParticleSystem} twin of this method
        # (PairwiseFunctors.jl) — narrowly typed the same way the host method
        # is, so an unrelated device-viewed pairing still MethodErrors.
        rng = MersenneTwister(109)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        a_ref = _kacpu_random_fluid(rng, 220, 2; L=1.0)
        b_ref = _kacpu_random_fluid(rng, 170, 2; L=1.0)
        a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)
        si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
        for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
            pa, ka_, sa = _kacpu_sortbufs(a)
            sort_particles!(a, cutoff, pa, ka_, sa)
            pb, kb, sb = _kacpu_sortbufs(b)
            sort_particles!(b, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(a_ref.dvdt, a_ka.dvdt)
        _kacpu_assert_drhodt_close(a_ref.drhodt, a_ka.drhodt)
        _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
        _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
    end

    @testset "coupled reverse sweep (FluidPfn fluid-fluid, WritesBoth) 3D: onesided vs ka=true" begin
        # 3D sibling of the 2D test above, mirroring the file's established
        # self/coupled-forward-sweep pattern (every 2D KA-equivalence test in
        # this file has a 3D counterpart, since the extra loop-nesting level
        # 3D adds changes KA.CPU()'s codegen relative to Polyester's — see the
        # header comment at the top of this file). Exercises
        # DeviceSystem{T,3,FluidParticleSystem} dispatch and the 3D
        # reverse-sweep kernel's neighbour-stencil loop together.
        rng = MersenneTwister(113)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=3)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)
        a_ref = _kacpu_random_fluid(rng, 180, 3; L=1.0)
        b_ref = _kacpu_random_fluid(rng, 140, 3; L=1.0)
        a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)
        si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
        for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
            pa, ka_, sa = _kacpu_sortbufs(a)
            sort_particles!(a, cutoff, pa, ka_, sa)
            pb, kb, sb = _kacpu_sortbufs(b)
            sort_particles!(b, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(a_ref.dvdt, a_ka.dvdt)
        _kacpu_assert_drhodt_close(a_ref.drhodt, a_ka.drhodt)
        _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
        _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
    end

    @testset "coupled reverse sweep (FluidSolidPfn, WritesBoth): onesided vs ka=true" begin
        # DambreakWall.jl's fluid/wall coupling shape (fluid=ps_a,
        # solid=ps_b). Unlike FluidPfn's fluid-fluid case, FluidSolidPfn's
        # physics is NOT symmetric under relabeling — the fluid's own
        # pressure must be used for both sides regardless of which slot it's
        # in — so it needs two distinct DeviceSystem{T,ND,Kind}-typed
        # `pfn_contribution` methods (see PairwiseFunctors.jl), one per
        # physical assignment. A WritesBoth interaction exercises both: the
        # forward pass hits the fluid-as-ps_a method, the reverse pass hits
        # the solid-as-ps_a method. Same Kind mechanism as the FluidPfn
        # fluid-fluid fix above.
        rng = MersenneTwister(114)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidSolidPfn(0.03, 0.0, h)
        a_ref = _kacpu_random_fluid(rng, 220, 2; L=1.0)
        b_ref = _kacpu_random_elastoplastic(rng, 170, 2; L=1.0)
        a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)
        si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
        for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
            pa, ka_, sa = _kacpu_sortbufs(a)
            sort_particles!(a, cutoff, pa, ka_, sa)
            pb, kb, sb = _kacpu_sortbufs(b)
            sort_particles!(b, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(a_ref.dvdt, a_ka.dvdt)
        _kacpu_assert_drhodt_close(a_ref.drhodt, a_ka.drhodt)
        _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
        _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
    end

    @testset "coupled reverse sweep (FluidSolidPfn, WritesBoth) 3D: onesided vs ka=true" begin
        rng = MersenneTwister(115)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=3)
        cutoff = kernel.interaction_length
        pfn = FluidSolidPfn(0.03, 0.0, h)
        a_ref = _kacpu_random_fluid(rng, 180, 3; L=1.0)
        b_ref = _kacpu_random_elastoplastic(rng, 140, 3; L=1.0)
        a_ka, b_ka = deepcopy(a_ref), deepcopy(b_ref)
        si_ref = SystemInteraction(kernel, pfn, a_ref, b_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, a_ka, b_ka; onesided=true, ka=true)
        for (a, b, si) in ((a_ref, b_ref, si_ref), (a_ka, b_ka, si_ka))
            pa, ka_, sa = _kacpu_sortbufs(a)
            sort_particles!(a, cutoff, pa, ka_, sa)
            pb, kb, sb = _kacpu_sortbufs(b)
            sort_particles!(b, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        _kacpu_assert_dvdt_close(a_ref.dvdt, a_ka.dvdt)
        _kacpu_assert_drhodt_close(a_ref.drhodt, a_ka.drhodt)
        _kacpu_assert_dvdt_close(b_ref.dvdt, b_ka.dvdt)
        _kacpu_assert_drhodt_close(b_ref.drhodt, b_ka.drhodt)
    end

    @testset "FluidSolidPfn ka=true: solid-side contribution uses fluid's pressure, not solid's own" begin
        # Regression guard for the asymmetric-physics hazard the two
        # narrowly-typed methods exist to prevent: if the solid-as-ps_a
        # device_view method ever regressed to reading ps_a.p (the solid's
        # own pressure) instead of ps_b.p (the fluid's), this test would
        # catch it, since the swap-antisymmetry check alone (comparing a
        # call against itself) cannot.
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        pfn = FluidSolidPfn(0.03, 0.0, h)
        fluid = _kacpu_random_fluid(MersenneTwister(116), 3, 2; L=0.05)
        solid = _kacpu_random_elastoplastic(MersenneTwister(117), 3, 2; L=0.05)
        dx, gx, w = SVector(0.01, 0.0), SVector(1.0, 0.0), 0.5

        dv_fluid = Grasph.device_view(fluid)
        c1 = pfn_contribution(pfn, Grasph.device_view(solid), dv_fluid, 1, 1, dx, gx, w)

        solid.p .= 999.0   # must be ignored — FluidSolidPfn never reads the solid's own p
        c2 = pfn_contribution(pfn, Grasph.device_view(solid), dv_fluid, 1, 1, dx, gx, w)

        @test c1.dvdt == c2.dvdt
        @test c1.drhodt == c2.drhodt
    end

    @testset "device_view Kind parameter (FluidSolidPfn): mismatched pairing still MethodErrors" begin
        # Same regression guard as the FluidPfn fluid-fluid one below, for
        # FluidSolidPfn's two Kind-typed methods: neither should match a
        # fluid-fluid pairing (that's FluidPfn's job) or a fluid-bare-system
        # pairing (that's not a physical fluid-solid coupling).
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        pfn = FluidSolidPfn(0.03, 0.0, h)
        fluid  = _kacpu_random_fluid(MersenneTwister(118), 3, 2; L=0.05)
        fluid2 = _kacpu_random_fluid(MersenneTwister(119), 3, 2; L=0.05)
        bnd    = _kacpu_random_boundary(MersenneTwister(120), 3, 2; L=0.05)
        dx, gx, w = SVector(0.01, 0.0), SVector(1.0, 0.0), 0.5
        @test_throws MethodError pfn_contribution(pfn, Grasph.device_view(fluid), Grasph.device_view(fluid2), 1, 1, dx, gx, w)
        @test_throws MethodError pfn_contribution(pfn, Grasph.device_view(fluid), Grasph.device_view(bnd), 1, 1, dx, gx, w)
        @test_throws MethodError pfn_contribution(pfn, Grasph.device_view(bnd), Grasph.device_view(fluid), 1, 1, dx, gx, w)
    end

    @testset "device_view Kind parameter: mismatched pairing still MethodErrors" begin
        # Regression guard for the fix above: DeviceSystem{T,ND,Kind} must
        # still throw MethodError for a pairing FluidPfn fluid-fluid was
        # never meant to accept, not silently compute with the wrong
        # dispatch. A device-viewed bare BasicParticleSystem (the same
        # concrete type StaticBoundarySystem wraps as its `inner` field
        # elsewhere in this file, but used here directly/unwrapped as a
        # bare-system stand-in) paired with a device-viewed
        # FluidParticleSystem must not match the new
        # DeviceSystem{T,ND,FluidParticleSystem} method.
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        pfn = FluidPfn(0.03, 0.0, h)
        fluid = _kacpu_random_fluid(MersenneTwister(111), 3, 2; L=0.05)
        bnd   = _kacpu_random_boundary(MersenneTwister(112), 3, 2; L=0.05)
        dx, gx, w = SVector(0.01, 0.0), SVector(1.0, 0.0), 0.5
        @test_throws MethodError pfn_contribution(pfn, Grasph.device_view(fluid), Grasph.device_view(bnd), 1, 1, dx, gx, w)
        @test_throws MethodError pfn_contribution(pfn, Grasph.device_view(bnd), Grasph.device_view(fluid), 1, 1, dx, gx, w)
    end

    @testset "coupled reverse sweep (InterpolateFieldFn, WritesB, virtual target): onesided vs ka=true" begin
        # Trapdoor.jl/EP_ColumnCollapse2.jl's shape: real source -> virtual
        # target. Virtual is device_view-ready since item 5
        # (DeviceVirtualSystem); this is the first KA test to exercise it.
        rng = MersenneTwister(110)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
        src = _kacpu_random_fluid(rng, 220, 2; L=1.0)
        virt_ref = _as_virtual_fluid(rng, 170, 2; L=1.0)
        _zero_interp_target!(virt_ref, (:v, :rho))
        virt_ka = deepcopy(virt_ref)
        si_ref = SystemInteraction(kernel, pfn, src, virt_ref; onesided=true)
        si_ka  = SystemInteraction(kernel, pfn, src, virt_ka; onesided=true, ka=true)
        pa, ka_, sa = _kacpu_sortbufs(src)
        sort_particles!(src, cutoff, pa, ka_, sa)
        for (v, si) in ((virt_ref, si_ref), (virt_ka, si_ka))
            pb, kb, sb = _kacpu_sortbufs(v)
            sort_particles!(v, cutoff, pb, kb, sb)
            create_grid!(si)
            sweep!(si)
        end
        @test _elemdiff(virt_ref.v, virt_ka.v) < 1e-9 * _elemscale(virt_ref.v)
        @test maximum(abs.(virt_ref.rho .- virt_ka.rho)) < 1e-9 * max(maximum(abs.(virt_ref.rho)), 1.0)
        @test maximum(abs.(virt_ref.w_sum .- virt_ka.w_sum)) < 1e-9 * max(maximum(abs.(virt_ref.w_sum)), 1.0)
    end

    @testset "coupled sweep (fluid<->GhostParticleSystem, item 7): onesided vs ka=true" begin
        # GhostParticleSystem's dispatch surface (pfn_contribution methods
        # narrowly typed on AbstractGhostParticleSystem) predates item 7, but
        # `device_view(ghost)` did not — before item 7, `ka=true` on any
        # ghost-coupled interaction was an unconditional MethodError. This is
        # the first KA test to exercise it. Uses a REAL self-referencing
        # ghost (ghost.source === fluid, mirroring bubble3.jl's
        # boundary_ghost) via test_onesided_sweep.jl's
        # _xsph_ghost_fluid/_xsph_ghost_setup! fixtures — two independent
        # fluid+ghost pairs (deepcopy before either is touched further), one
        # run through the Polyester onesided sweep, one through ka=true.
        rng = MersenneTwister(113)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        pfn = FluidPfn(0.03, 0.0, h)
        for (nx, ny, dx, boundary_cutoff, L) in ((6, 6, h, 3h, 6h), (10, 10, h, 3h, 40h))
            fluid_ref = _xsph_ghost_fluid(rng, nx, ny, dx)
            fluid_ka  = deepcopy(fluid_ref)
            ghost_ref = _xsph_ghost_setup!(fluid_ref, kernel.interaction_length, boundary_cutoff, L, L)
            ghost_ka  = _xsph_ghost_setup!(fluid_ka,  kernel.interaction_length, boundary_cutoff, L, L)
            @test ghost_ref.n == ghost_ka.n   # sanity: identical starting state -> identical ghost count

            si_ref = SystemInteraction(kernel, pfn, fluid_ref, ghost_ref; onesided=true)
            si_ka  = SystemInteraction(kernel, pfn, fluid_ka,  ghost_ka;  onesided=true, ka=true)
            create_grid!(si_ref); sweep!(si_ref)
            create_grid!(si_ka);  sweep!(si_ka)

            _kacpu_assert_dvdt_close(fluid_ref.dvdt, fluid_ka.dvdt)
            _kacpu_assert_drhodt_close(fluid_ref.drhodt, fluid_ka.drhodt)
        end
    end

    @testset "device_view is a faithful proxy (self and coupled)" begin
        # Running the existing Polyester one-sided sweep against a
        # device_view-wrapped host system must reproduce the un-wrapped
        # result exactly — proves the view carries every field a pfn reads.
        # Reference-value checks only (no recomputation here), so exact.
        rng = MersenneTwister(106)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        cutoff = kernel.interaction_length
        pfn = FluidPfn(0.03, 0.0, h)

        ps = _kacpu_random_fluid(rng, 200, 2; L=1.0)
        si = SystemInteraction(kernel, pfn, ps; onesided=true)
        pb, kb, sc = _kacpu_sortbufs(ps)
        sort_particles!(ps, cutoff, pb, kb, sc)
        create_grid!(si)
        sweep!(si)

        dv = Grasph.device_view(ps)
        @test dv.x == ps.x && dv.v == ps.v && dv.rho == ps.rho && dv.p == ps.p
        @test dv.mass == ps.mass && dv.c == ps.c
        @test dv isa Grasph.AbstractParticleSystem{Float64,2}
    end

end
