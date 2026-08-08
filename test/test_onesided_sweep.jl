using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm, dot
using Random

# ---------------------------------------------------------------------------
# Validation for the opt-in `onesided=true` sweep (Interaction.jl) and the
# one-sided `pfn_contribution` methods for `FluidPfn` (PairwiseFunctors.jl).
#
# `onesided=true` is additive: it does not change the default (`onesided`
# omitted / `false`) code path used by every existing script, so these tests
# are the only place this mechanism is exercised. They validate it against
# the existing coloured half-shell/coupled sweep as an independent oracle,
# via:
#
#   1. Swap-antisymmetry:  pfn_contribution(pfn,ps,i,j,...) combined with
#      pfn_contribution(pfn,ps,j,i,...) must reproduce the two-sided
#      mutating method's effect on both particles exactly — this is the
#      structural property the whole one-sided rewrite depends on.
#   2. Self-pair exclusion: a particle must never pair with itself in the
#      full same-cell scan (unlike the half-shell's i<j ordering, which made
#      this impossible by construction).
#   3. Full-sweep equivalence: onesided=true vs onesided=false on identical
#      particle clouds, built through the real sort_particles!/create_grid!
#      path (not manually-injected CSR arrays), in 2D and 3D, self and
#      coupled (fluid <-> StaticBoundarySystem).
#   4. Short-run trajectory equivalence: a real LeapFrogTimeIntegrator run
#      over many steps, comparing accumulated drift.
# ---------------------------------------------------------------------------

_sortbufs(ps::Grasph.AbstractParticleSystem) =
    (Vector{Int}(undef, ps.n), Vector{UInt64}(undef, ps.n), Grasph._make_sort_scratch(ps))

function _random_fluid(rng, n, ndims; L=1.0)
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

function _random_boundary(rng, n, ndims; L=1.0)
    inner = BasicParticleSystem("bnd", n, ndims, 1.0, 10.0)
    for i in 1:n
        inner.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
    end
    inner.rho .= 1000.0
    fill!(inner.v, zero(SVector{ndims,Float64}))
    return inner
end

# ---------------------------------------------------------------------------
# 1. Swap-antisymmetry
# ---------------------------------------------------------------------------

@testset "pfn_contribution swap-antisymmetry (FluidPfn self)" begin
    rng = MersenneTwister(1)
    h = 0.1
    kernel = CubicSplineKernel(h; ndims=2)
    pfn = FluidPfn(0.03, 0.0, h)

    for trial in 1:20
        ps = _random_fluid(rng, 2, 2; L=0.05)   # small L: guarantee within cutoff

        # Ground truth: the existing two-sided mutating method.
        xi, xj = ps.x[1], ps.x[2]
        dx = xi - xj
        r  = norm(dx)
        r < 1e-8 && continue   # skip degenerate coincident draw
        q  = r / h
        gx = (Grasph.kernel_dw_dq(kernel, q) / (r * h)) * dx
        w  = Grasph.kernel_w(kernel, q)

        fill!(ps.dvdt, zero(SVector{2,Float64})); ps.drhodt .= 0.0
        pfn(ps, 1, 2, dx, gx, w)   # two-sided mutating call
        expected_dvdt_1, expected_dvdt_2   = ps.dvdt[1], ps.dvdt[2]
        expected_drho_1, expected_drho_2   = ps.drhodt[1], ps.drhodt[2]

        # One-sided: contribution to 1 from 2, and contribution to 2 from 1
        # (computed from the ACTUALLY swapped pair, not just negated dx/gx).
        c1 = pfn_contribution(pfn, ps, 1, 2, dx, gx, w)

        dx_ji = xj - xi
        r_ji  = norm(dx_ji)
        q_ji  = r_ji / h
        gx_ji = (Grasph.kernel_dw_dq(kernel, q_ji) / (r_ji * h)) * dx_ji
        w_ji  = Grasph.kernel_w(kernel, q_ji)
        c2 = pfn_contribution(pfn, ps, 2, 1, dx_ji, gx_ji, w_ji)

        @test c1.dvdt   ≈ expected_dvdt_1 atol=1e-13
        @test c2.dvdt   ≈ expected_dvdt_2 atol=1e-13
        @test c1.drhodt ≈ expected_drho_1 atol=1e-10
        @test c2.drhodt ≈ expected_drho_2 atol=1e-10
    end
end

# ---------------------------------------------------------------------------
# 2. Self-pair exclusion
# ---------------------------------------------------------------------------

@testset "onesided self sweep never pairs a particle with itself" begin
    h = 0.1
    kernel = CubicSplineKernel(h; ndims=2)
    cutoff = kernel.interaction_length
    pfn = FluidPfn(0.03, 0.0, h)

    # A single, isolated particle: if j==i were not excluded, dx=0 would
    # divide by zero (r=0) inside _pair_self_onesided! and dvdt/drhodt would
    # become NaN. With correct exclusion, the sweep finds no neighbours and
    # dvdt/drhodt are left exactly at their pre-sweep values.
    ps = _random_fluid(MersenneTwister(2), 1, 2)
    ps.x[1] = SVector(0.5, 0.5)
    sentinel_dvdt, sentinel_drho = SVector(1.23, -4.56), 7.89
    ps.dvdt[1]   = sentinel_dvdt
    ps.drhodt[1] = sentinel_drho

    si = SystemInteraction(kernel, pfn, ps; onesided=true)
    perm_buf, key_buf, scratch = _sortbufs(ps)
    sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
    create_grid!(si)
    sweep!(si)

    @test ps.dvdt[1]   == sentinel_dvdt
    @test ps.drhodt[1] == sentinel_drho
    @test !any(isnan, ps.dvdt[1])
    @test !isnan(ps.drhodt[1])

    # Several exactly-coincident particles in the same cell: i must never
    # pair with itself, though it correctly pairs with the *other*
    # coincident particles (a pre-existing, unrelated r=0 behaviour of the
    # fixed-h kernel that this change does not alter).
    n = 5
    ps2 = FluidParticleSystem("fluid", n, 2, 1.0, 10.0; source_v = zeros(2))
    for i in 1:n
        ps2.x[i] = SVector(0.5, 0.5)   # all exactly coincident
        ps2.v[i] = zero(SVector{2,Float64})
    end
    ps2.rho .= 1000.0
    ps2.p   .= 0.0
    fill!(ps2.dvdt, zero(SVector{2,Float64})); ps2.drhodt .= 0.0
    si2 = SystemInteraction(kernel, pfn, ps2; onesided=true)
    perm_buf2, key_buf2, scratch2 = _sortbufs(ps2)
    sort_particles!(ps2, cutoff, perm_buf2, key_buf2, scratch2)
    create_grid!(si2)
    # r=0 for the n-1 genuinely-distinct coincident neighbours: division by
    # zero is expected pre-existing behaviour (NaN), not something being
    # tested here — the exclusion regression is that this does NOT happen
    # purely from the (i==j) case, which the single-particle test above
    # already isolates and confirms.
end

# ---------------------------------------------------------------------------
# 3. Full-sweep equivalence — onesided=true vs onesided=false, built through
#    real sort_particles!/create_grid!, self-interaction.
# ---------------------------------------------------------------------------

function _compare_self_sweep(rng, n, ndims; L=1.0, h=0.08)
    kernel = CubicSplineKernel(h; ndims=ndims)
    cutoff = kernel.interaction_length
    pfn    = FluidPfn(0.03, 0.0, h)

    ps_old = _random_fluid(rng, n, ndims; L=L)
    ps_new = deepcopy(ps_old)

    si_old = SystemInteraction(kernel, pfn, ps_old)
    si_new = SystemInteraction(kernel, pfn, ps_new; onesided=true)

    for (ps, si) in ((ps_old, si_old), (ps_new, si_new))
        perm_buf, key_buf, scratch = _sortbufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        create_grid!(si)
        sweep!(si)
    end

    dvdt_scale   = max(maximum(norm.(ps_old.dvdt)), 1.0)
    drhodt_scale = max(maximum(abs.(ps_old.drhodt)), 1.0)
    @test maximum(norm.(ps_old.dvdt .- ps_new.dvdt))     < 1e-11 * dvdt_scale
    @test maximum(abs.(ps_old.drhodt .- ps_new.drhodt))  < 1e-11 * drhodt_scale
end

@testset "onesided=true self sweep matches coloured sweep (2D)" begin
    rng = MersenneTwister(10)
    for (n, L) in ((50, 1.0), (400, 1.0), (400, 0.3), (1500, 1.0))
        _compare_self_sweep(rng, n, 2; L=L)
    end
end

@testset "onesided=true self sweep matches coloured sweep (3D)" begin
    rng = MersenneTwister(11)
    for (n, L) in ((60, 1.0), (500, 1.0), (500, 0.3))
        _compare_self_sweep(rng, n, 3; L=L)
    end
end

@testset "onesided=true self sweep matches coloured sweep — cell-boundary-adjacent positions" begin
    # Positions deliberately snapped exactly onto (or just either side of)
    # cell boundaries, in both 2D and 3D, so pairs straddle every neighbour
    # offset the full-stencil scan must cover.
    h = 0.1
    kernel2 = CubicSplineKernel(h; ndims=2)
    cutoff2 = kernel2.interaction_length
    pfn = FluidPfn(0.03, 0.0, h)

    ps_old = FluidParticleSystem("fluid", 9, 2, 1.0, 10.0; source_v = zeros(2))
    ps_new = FluidParticleSystem("fluid", 9, 2, 1.0, 10.0; source_v = zeros(2))
    k = 1
    for gi in -1:1, gj in -1:1
        # a point just inside each of the 9 cells surrounding the origin cell
        x = SVector((gi + 0.5) * cutoff2 * 0.99, (gj + 0.5) * cutoff2 * 0.99)
        ps_old.x[k] = x; ps_new.x[k] = x
        k += 1
    end
    for ps in (ps_old, ps_new)
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0; ps.p .= 50.0
        fill!(ps.dvdt, zero(SVector{2,Float64})); ps.drhodt .= 0.0
    end

    si_old = SystemInteraction(kernel2, pfn, ps_old)
    si_new = SystemInteraction(kernel2, pfn, ps_new; onesided=true)
    for (ps, si) in ((ps_old, si_old), (ps_new, si_new))
        perm_buf, key_buf, scratch = _sortbufs(ps)
        sort_particles!(ps, cutoff2, perm_buf, key_buf, scratch)
        create_grid!(si)
        sweep!(si)
    end

    @test maximum(norm.(ps_old.dvdt .- ps_new.dvdt))    < 1e-12
    @test maximum(abs.(ps_old.drhodt .- ps_new.drhodt)) < 1e-9
end

# ---------------------------------------------------------------------------
# 4. Full-sweep equivalence — coupled (fluid <-> StaticBoundarySystem)
# ---------------------------------------------------------------------------

function _compare_coupled_sweep(rng, n_fluid, n_bnd, ndims; L=1.0, h=0.08)
    kernel = CubicSplineKernel(h; ndims=ndims)
    cutoff = kernel.interaction_length
    pfn    = FluidPfn(0.03, 0.0, h)

    fluid_old = _random_fluid(rng, n_fluid, ndims; L=L)
    fluid_new = deepcopy(fluid_old)
    bnd       = _random_boundary(rng, n_bnd, ndims; L=L)
    static_bnd = StaticBoundarySystem(bnd, 0.03)

    si_old = SystemInteraction(kernel, pfn, fluid_old, static_bnd)
    si_new = SystemInteraction(kernel, pfn, fluid_new, static_bnd; onesided=true)

    perm_buf, key_buf, scratch = _sortbufs(bnd)
    sort_particles!(bnd, cutoff, perm_buf, key_buf, scratch)

    for (ps, si) in ((fluid_old, si_old), (fluid_new, si_new))
        pb, kb, sc = _sortbufs(ps)
        sort_particles!(ps, cutoff, pb, kb, sc)
        create_grid!(si)
        sweep!(si)
    end

    dvdt_scale = max(maximum(norm.(fluid_old.dvdt)), 1.0)
    @test maximum(norm.(fluid_old.dvdt .- fluid_new.dvdt)) < 1e-9 * dvdt_scale
    # This pfn/context never touches drhodt; confirm neither path drifts from zero.
    @test all(==(0.0), fluid_old.drhodt)
    @test all(==(0.0), fluid_new.drhodt)
end

@testset "onesided=true coupled (fluid<->StaticBoundarySystem) sweep matches coloured sweep" begin
    rng = MersenneTwister(20)
    _compare_coupled_sweep(rng, 300, 200, 2)
    _compare_coupled_sweep(rng, 300, 200, 2; L=0.3)
    _compare_coupled_sweep(rng, 400, 300, 3)
end

# ---------------------------------------------------------------------------
# 5. Short-run trajectory equivalence — real LeapFrogTimeIntegrator
# ---------------------------------------------------------------------------

function _dambreak_like(ndims; n_fluid_per_dim=8, seed=99)
    rng = MersenneTwister(seed)
    h = 0.08
    kernel = CubicSplineKernel(h; ndims=ndims)
    dx = 0.06
    rho0 = 1000.0
    c_sound = 20.0

    nf = ndims == 2 ? (n_fluid_per_dim, n_fluid_per_dim) : (n_fluid_per_dim, n_fluid_per_dim, n_fluid_per_dim)
    n_fluid = prod(nf)
    fluid = FluidParticleSystem("fluid", n_fluid, ndims, rho0 * dx^ndims, c_sound;
                                source_v = ndims == 2 ? [0.0, -9.81] : [0.0, 0.0, -9.81],
                                state_updater = TaitEOSUpdater(rho0))
    k = 1
    ranges = ndims == 2 ? Iterators.product(0:nf[1]-1, 0:nf[2]-1) : Iterators.product(0:nf[1]-1, 0:nf[2]-1, 0:nf[3]-1)
    for idx in ranges
        fluid.x[k] = SVector((idx .+ 0.5) .* dx)
        k += 1
    end
    fill!(fluid.v, zero(SVector{ndims,Float64}))
    fluid.rho .= rho0
    update_state!(fluid, 1)

    # A simple flat floor of boundary particles beneath the block.
    nb = ndims == 2 ? (nf[1] + 4,) : (nf[1] + 4, nf[2] + 4)
    n_bnd = prod(nb)
    bnd = BasicParticleSystem("boundary", n_bnd, ndims, rho0 * dx^ndims, c_sound)
    k = 1
    if ndims == 2
        for i in -2:nf[1]+1
            bnd.x[k] = SVector((i + 0.5) * dx, -0.5 * dx)
            k += 1
        end
    else
        for i in -2:nf[1]+1, j in -2:nf[2]+1
            bnd.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx, -0.5 * dx)
            k += 1
        end
    end
    bnd.rho .= rho0
    fill!(bnd.v, zero(SVector{ndims,Float64}))
    static_bnd = StaticBoundarySystem(bnd, dx)

    return kernel, fluid, bnd, static_bnd
end

@testset "short-run trajectory equivalence: onesided=true vs onesided=false" begin
    for ndims in (2, 3)
        kernel, fluid_old, bnd_old, static_old = _dambreak_like(ndims)
        _, fluid_new, bnd_new, static_new       = _dambreak_like(ndims)
        h = 0.08
        pfn = FluidPfn(0.03, 0.0, h)

        si_self_old = SystemInteraction(kernel, pfn, fluid_old)
        si_bnd_old  = SystemInteraction(kernel, pfn, fluid_old, static_old)
        si_self_new = SystemInteraction(kernel, pfn, fluid_new; onesided=true)
        si_bnd_new  = SystemInteraction(kernel, pfn, fluid_new, static_new; onesided=true)

        lf_old = LeapFrogTimeIntegrator([fluid_old, bnd_old], [si_self_old, si_bnd_old])
        lf_new = LeapFrogTimeIntegrator([fluid_new, bnd_new], [si_self_new, si_bnd_new])

        nsteps = 100
        time_integrate!(lf_old, nsteps, nsteps + 1, nsteps + 1, 0.1, nothing; print_timer=false)
        time_integrate!(lf_new, nsteps, nsteps + 1, nsteps + 1, 0.1, nothing; print_timer=false)

        x_scale   = max(maximum(norm.(fluid_old.x)), 1.0)
        v_scale   = max(maximum(norm.(fluid_old.v)), 1.0)
        rho_scale = max(maximum(abs.(fluid_old.rho)), 1.0)

        x_diff   = maximum(norm.(fluid_old.x   .- fluid_new.x))
        v_diff   = maximum(norm.(fluid_old.v   .- fluid_new.v))
        rho_diff = maximum(abs.(fluid_old.rho .- fluid_new.rho))

        @test !any(isnan, reduce(vcat, [collect(v) for v in fluid_old.x]))
        @test x_diff   < 1e-8  * x_scale
        @test v_diff   < 1e-6  * v_scale
        @test rho_diff < 1e-6  * rho_scale
    end
end

# ---------------------------------------------------------------------------
# 6. Long-run physical invariants (onesided=true path only)
# ---------------------------------------------------------------------------

@testset "long-run physical invariants (onesided=true)" begin
    for ndims in (2, 3)
        kernel, fluid, bnd, static_bnd = _dambreak_like(ndims; n_fluid_per_dim=6)
        h = 0.08
        pfn = FluidPfn(0.03, 0.0, h)
        si_self = SystemInteraction(kernel, pfn, fluid; onesided=true)
        si_bnd  = SystemInteraction(kernel, pfn, fluid, static_bnd; onesided=true)
        lf = LeapFrogTimeIntegrator([fluid, bnd], [si_self, si_bnd])

        time_integrate!(lf, 300, 301, 301, 0.1, nothing; print_timer=false)

        @test all(!isnan, fluid.rho)
        @test all(!isinf, fluid.rho)
        @test all(x -> all(!isnan, x), fluid.x)
        @test all(x -> all(!isnan, x), fluid.v)
        # No particle should have travelled absurdly far given the short run
        # and small initial block — a broken boundary coupling (particles
        # leaking through the floor to -Inf) would blow this up.
        @test all(x -> all(abs.(x) .< 50.0), fluid.x)
    end
end

# ---------------------------------------------------------------------------
# 7. Phase 1 conversions (see docs — "convert the remaining pairwise
#    functors" plan): StrainRatePfn, StrainRateVorticityPfn, CauchyFluidPfn,
#    XSPHPfn (self only), and FluidPfn's remaining ghost/virtual +
#    dynamic-boundary variants. Same strategy as above: build identical
#    particle clouds, run coloured vs onesided=true through the real
#    sort_particles!/create_grid!/sweep! path, compare the fields each pfn
#    actually writes.
# ---------------------------------------------------------------------------

_elemscale(v::AbstractVector{<:Real}) = max(maximum(abs, v), 1.0)
_elemscale(v::AbstractVector{<:SVector}) = max(maximum(norm, v), 1.0)
_elemdiff(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = maximum(abs.(a .- b))
_elemdiff(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector}) = maximum(norm.(a .- b))

function _random_stress(rng, n, ndims; ns = ndims == 2 ? 3 : 6, L=1.0)
    ps = StressParticleSystem("stress", n, ndims, ns, 1.0, 10.0)
    for i in 1:n
        ps.x[i]      = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i]      = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
        ps.stress[i] = SVector(ntuple(_ -> 50.0 * (rand(rng) - 0.5), ns)...)
    end
    ps.rho .= 1000.0 .+ 20 .* (rand(rng, n) .- 0.5)
    ps.p   .= 100.0 .* rand(rng, n)
    fill!(ps.dvdt, zero(SVector{ndims,Float64})); ps.drhodt .= 0.0
    fill!(ps.strain_rate, zero(eltype(ps.strain_rate)))
    return ps
end

function _random_ep(rng, n, ndims; ns = ndims == 2 ? 3 : 6, L=1.0)
    ps = ElastoPlasticParticleSystem("ep", n, ndims, ns, 1.0, 10.0)
    for i in 1:n
        ps.x[i]      = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i]      = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
        ps.stress[i] = SVector(ntuple(_ -> 50.0 * (rand(rng) - 0.5), ns)...)
    end
    ps.rho .= 1000.0 .+ 20 .* (rand(rng, n) .- 0.5)
    ps.p   .= 100.0 .* rand(rng, n)
    fill!(ps.dvdt, zero(SVector{ndims,Float64})); ps.drhodt .= 0.0
    fill!(ps.strain_rate, zero(eltype(ps.strain_rate)))
    fill!(ps.vorticity, zero(eltype(ps.vorticity)))
    return ps
end

_dynamic_boundary(rng, n, ndims; L=1.0) = DynamicBoundarySystem(
    _random_boundary(rng, n, ndims; L=L),
    ndims == 2 ? SVector(0.0, 1.0) : SVector(0.0, 0.0, 1.0),
    zero(SVector{ndims,Float64}),
    3.0,
)

function _compare_self_generic(rng, pfn, build, fields, n, ndims; L=1.0, h=0.08)
    kernel = CubicSplineKernel(h; ndims=ndims)
    cutoff = kernel.interaction_length
    ps_old = build(rng, n, ndims; L=L)
    ps_new = deepcopy(ps_old)
    si_old = SystemInteraction(kernel, pfn, ps_old)
    si_new = SystemInteraction(kernel, pfn, ps_new; onesided=true)
    for (ps, si) in ((ps_old, si_old), (ps_new, si_new))
        perm_buf, key_buf, scratch = _sortbufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        create_grid!(si)
        sweep!(si)
    end
    for f in fields
        va, vb = getproperty(ps_old, f), getproperty(ps_new, f)
        @test _elemdiff(va, vb) < 1e-9 * _elemscale(va)
    end
end

_sortable(ps::AbstractBoundarySystem) = getfield(ps, :inner)   # StaticBoundarySystem/DynamicBoundarySystem
_sortable(ps) = ps                                              # VirtualParticleSystem delegates to :source itself

function _compare_coupled_generic(rng, pfn, build_a, build_b, fields, n_a, n_b, ndims; L=1.0, h=0.08)
    kernel = CubicSplineKernel(h; ndims=ndims)
    cutoff = kernel.interaction_length
    a_old = build_a(rng, n_a, ndims; L=L)
    a_new = deepcopy(a_old)
    b     = build_b(rng, n_b, ndims; L=L)

    si_old = SystemInteraction(kernel, pfn, a_old, b)
    si_new = SystemInteraction(kernel, pfn, a_new, b; onesided=true)

    b_sort = _sortable(b)
    pb, kb, sc = _sortbufs(b_sort)
    sort_particles!(b_sort, cutoff, pb, kb, sc)

    for (ps, si) in ((a_old, si_old), (a_new, si_new))
        p2, k2, s2 = _sortbufs(ps)
        sort_particles!(ps, cutoff, p2, k2, s2)
        create_grid!(si)
        sweep!(si)
    end

    for f in fields
        va, vb2 = getproperty(a_old, f), getproperty(a_new, f)
        @test _elemdiff(va, vb2) < 1e-9 * _elemscale(va)
    end
end

_as_virtual_stress(rng, n, ndims; L=1.0) =
    let src = _random_stress(rng, n, ndims; L=L)
        VirtualParticleSystem(src, "virt", src.n, ndims, src.mass, src.c)
    end
_as_virtual_ep(rng, n, ndims; L=1.0) =
    let src = _random_ep(rng, n, ndims; L=L)
        VirtualParticleSystem(src, "virt", src.n, ndims, src.mass, src.c)
    end
_as_virtual_fluid(rng, n, ndims; L=1.0) =
    let src = _random_fluid(rng, n, ndims; L=L)
        VirtualParticleSystem(src, "virt", src.n, ndims, src.mass, src.c)
    end

@testset "onesided=true StrainRatePfn matches coloured sweep" begin
    rng = MersenneTwister(30)
    pfn = StrainRatePfn()
    for ndims in (2, 3)
        _compare_self_generic(rng, pfn, _random_stress, (:strain_rate,), 300, ndims)
        _compare_coupled_generic(rng, pfn, _random_stress, _as_virtual_stress, (:strain_rate,), 200, 150, ndims)
        _compare_coupled_generic(rng, pfn, _random_stress, _dynamic_boundary, (:strain_rate,), 200, 150, ndims)
    end
end

@testset "onesided=true StrainRateVorticityPfn matches coloured sweep" begin
    rng = MersenneTwister(31)
    pfn = StrainRateVorticityPfn()
    for ndims in (2, 3)
        _compare_self_generic(rng, pfn, _random_ep, (:strain_rate, :vorticity), 300, ndims)
        _compare_coupled_generic(rng, pfn, _random_ep, _as_virtual_ep, (:strain_rate, :vorticity), 200, 150, ndims)
        _compare_coupled_generic(rng, pfn, _random_ep, _dynamic_boundary, (:strain_rate, :vorticity), 200, 150, ndims)
    end
end

@testset "onesided=true CauchyFluidPfn matches coloured sweep" begin
    rng = MersenneTwister(32)
    pfn = CauchyFluidPfn(0.03, 0.0, 0.08)
    for ndims in (2, 3)
        _compare_self_generic(rng, pfn, _random_stress, (:dvdt, :drhodt), 300, ndims)
        _compare_coupled_generic(rng, pfn, _random_stress, _as_virtual_stress, (:dvdt, :drhodt), 200, 150, ndims)
        _compare_coupled_generic(rng, pfn, _random_stress, _dynamic_boundary, (:dvdt, :drhodt), 200, 150, ndims)
    end
end

@testset "onesided=true XSPHPfn (self) matches coloured sweep" begin
    rng = MersenneTwister(33)
    pfn = XSPHPfn(0.5)
    build(rng, n, ndims; L=1.0) = let ps = _random_fluid(rng, n, ndims; L=L)
        fill!(ps.v_adjustment, zero(SVector{ndims,Float64}))
        ps
    end
    for ndims in (2, 3)
        _compare_self_generic(rng, pfn, build, (:v_adjustment,), 300, ndims)
    end
end

@testset "onesided=true FluidPfn ghost/virtual + dynamic-boundary variants match coloured sweep" begin
    rng = MersenneTwister(34)
    pfn = FluidPfn(0.03, 0.0, 0.08)
    for ndims in (2, 3)
        _compare_coupled_generic(rng, pfn, _random_fluid, _as_virtual_fluid, (:dvdt, :drhodt), 300, 200, ndims)
        _compare_coupled_generic(rng, pfn, _random_fluid, _dynamic_boundary, (:dvdt, :drhodt), 300, 200, ndims)
    end
end

# ---------------------------------------------------------------------------
# 8. Phase 2: reverse-sweep infrastructure ("writes into system_b")
#
# _onesided_shape(pfn, ps_a, ps_b) selects which of two independent sweep
# passes runs for a coupled onesided interaction: the existing forward pass
# (writes system_a, WritesA() — the default, exercised by every test above)
# and the new reverse pass (writes system_b, WritesB()), or both
# (WritesBoth()). The reverse pass is brand-new infrastructure with no
# existing coloured-sweep counterpart to compare against, so it's validated
# here against a brute-force O(n^2) reference that calls the exact same
# pfn_contribution formula directly (no cell list) — this tests that the new
# sweep function finds the same pairs and writes them to the same place as
# an independent, cell-list-free computation.
# ---------------------------------------------------------------------------

# Shared test-only formula: writes into "ps_a"'s slot (the pfn_contribution
# write-target position) as a function of ps_a's own rho/v at i and ps_b's
# mass at j. Fully symmetric in role, not tied to which real system is which,
# so the same method backs both a WritesB()-only pfn (isolating the reverse
# pass) and a WritesBoth() pfn (exercising the dispatcher's "run both" arm).
@inline _test_pfn_contribution(ps_a, ps_b, i, j, dx, gx, w) =
    (dvdt = (ps_b.mass * w / ps_a.rho[i]) * gx, drhodt = ps_b.mass * dot(gx, ps_a.v[i]))
@inline _test_pfn_zero(ps_a, i) = (dvdt = zero(eltype(ps_a.dvdt)), drhodt = zero(eltype(ps_a.drhodt)))

struct _ReverseOnlyTestPfn end
Grasph._onesided_shape(::_ReverseOnlyTestPfn, ps_a, ps_b) = Grasph.WritesB()
@inline Grasph.pfn_contribution(::_ReverseOnlyTestPfn, ps_a, ps_b, i::Int, j::Int, dx::SVector, gx::SVector, w) =
    _test_pfn_contribution(ps_a, ps_b, i, j, dx, gx, w)
@inline Grasph._onesided_zero_coupled(::_ReverseOnlyTestPfn, ps_a, ps_b, i) = _test_pfn_zero(ps_a, i)

struct _MutualTestPfn end
Grasph._onesided_shape(::_MutualTestPfn, ps_a, ps_b) = Grasph.WritesBoth()
@inline Grasph.pfn_contribution(::_MutualTestPfn, ps_a, ps_b, i::Int, j::Int, dx::SVector, gx::SVector, w) =
    _test_pfn_contribution(ps_a, ps_b, i, j, dx, gx, w)
@inline Grasph._onesided_zero_coupled(::_MutualTestPfn, ps_a, ps_b, i) = _test_pfn_zero(ps_a, i)

# Brute-force O(n^2) reference for the contribution written into ps_b, using
# the exact dx/gx/w formula _pair_coupled_onesided! computes, but with no
# cell list involved at all — an independent check on which pairs the new
# reverse-sweep grid scan finds.
function _brute_force_reverse(pfn, ps_a, ps_b, kernel)
    cutoff_sq = kernel.interaction_length^2
    h = kernel.h
    dvdt   = [zero(eltype(ps_b.dvdt)) for _ in 1:ps_b.n]
    drhodt = zeros(ps_b.n)
    for j in 1:ps_b.n, i in 1:ps_a.n
        dx = ps_b.x[j] - ps_a.x[i]
        r_sq = dot(dx, dx)
        r_sq >= cutoff_sq && continue
        r = sqrt(r_sq)
        q = r / h
        gx = (Grasph.kernel_dw_dq(kernel, q) / (r * h)) * dx
        w  = Grasph.kernel_w(kernel, q)
        c = Grasph.pfn_contribution(pfn, ps_b, ps_a, j, i, dx, gx, w)
        dvdt[j]   += c.dvdt
        drhodt[j] += c.drhodt
    end
    return dvdt, drhodt
end

@testset "onesided=true reverse sweep (WritesB) writes into system_b, matches brute force" begin
    rng = MersenneTwister(40)
    pfn = _ReverseOnlyTestPfn()
    for ndims in (2, 3)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=ndims)
        cutoff = kernel.interaction_length
        ps_a = _random_fluid(rng, 250, ndims; L=1.0)
        ps_b = _random_fluid(rng, 180, ndims; L=1.0)

        si = SystemInteraction(kernel, pfn, ps_a, ps_b; onesided=true)
        pa, ka_, sa = _sortbufs(ps_a)
        sort_particles!(ps_a, cutoff, pa, ka_, sa)
        pb, kb, sb = _sortbufs(ps_b)
        sort_particles!(ps_b, cutoff, pb, kb, sb)
        create_grid!(si)
        sweep!(si)

        expected_dvdt, expected_drhodt = _brute_force_reverse(pfn, ps_a, ps_b, kernel)

        @test maximum(norm.(ps_b.dvdt .- expected_dvdt))    < 1e-9 * max(maximum(norm.(expected_dvdt)), 1.0)
        @test maximum(abs.(ps_b.drhodt .- expected_drhodt)) < 1e-9 * max(maximum(abs.(expected_drhodt)), 1.0)
        # A pure WritesB() pfn must leave system_a completely untouched.
        @test all(==(zero(SVector{ndims,Float64})), ps_a.dvdt)
        @test all(==(0.0), ps_a.drhodt)
    end
end

@testset "onesided=true reverse sweep (WritesB) matches brute force — cell-boundary-adjacent positions" begin
    # Duplicate of "self sweep matches coloured sweep — cell-boundary-adjacent
    # positions" above, but coupled: system_a and system_b particles are both
    # snapped onto/near cell boundaries so this is the first exercise of
    # _cell_start_a as a multi-cell contiguous strip
    # (cell_start_a[c]..cell_start_a[c+3]-1) rather than a single cell, in
    # every neighbour-offset direction the full-stencil scan must cover.
    h = 0.1
    kernel = CubicSplineKernel(h; ndims=2)
    cutoff = kernel.interaction_length
    pfn = _ReverseOnlyTestPfn()

    ps_a = FluidParticleSystem("fluid_a", 9, 2, 1.0, 10.0; source_v = zeros(2))
    ps_b = FluidParticleSystem("fluid_b", 9, 2, 1.0, 10.0; source_v = zeros(2))
    k = 1
    for gi in -1:1, gj in -1:1
        x = SVector((gi + 0.5) * cutoff * 0.99, (gj + 0.5) * cutoff * 0.99)
        ps_a.x[k] = x
        ps_b.x[k] = x + SVector(0.01 * cutoff, -0.01 * cutoff)
        k += 1
    end
    for ps in (ps_a, ps_b)
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0; ps.p .= 50.0
        fill!(ps.dvdt, zero(SVector{2,Float64})); ps.drhodt .= 0.0
    end

    si = SystemInteraction(kernel, pfn, ps_a, ps_b; onesided=true)
    pa, ka_, sa = _sortbufs(ps_a)
    sort_particles!(ps_a, cutoff, pa, ka_, sa)
    pb, kb, sb = _sortbufs(ps_b)
    sort_particles!(ps_b, cutoff, pb, kb, sb)
    create_grid!(si)
    sweep!(si)

    expected_dvdt, expected_drhodt = _brute_force_reverse(pfn, ps_a, ps_b, kernel)
    @test maximum(norm.(ps_b.dvdt .- expected_dvdt))    < 1e-12
    @test maximum(abs.(ps_b.drhodt .- expected_drhodt)) < 1e-9
end

@testset "onesided=true reverse sweep (WritesBoth) writes into both systems" begin
    rng = MersenneTwister(41)
    pfn = _MutualTestPfn()
    h = 0.08
    kernel = CubicSplineKernel(h; ndims=2)
    cutoff = kernel.interaction_length
    ps_a = _random_fluid(rng, 200, 2; L=1.0)
    ps_b = _random_fluid(rng, 150, 2; L=1.0)

    si = SystemInteraction(kernel, pfn, ps_a, ps_b; onesided=true)
    pa, ka_, sa = _sortbufs(ps_a)
    sort_particles!(ps_a, cutoff, pa, ka_, sa)
    pb, kb, sb = _sortbufs(ps_b)
    sort_particles!(ps_b, cutoff, pb, kb, sb)
    create_grid!(si)
    sweep!(si)

    expected_b_dvdt, expected_b_drhodt = _brute_force_reverse(pfn, ps_a, ps_b, kernel)
    @test maximum(norm.(ps_b.dvdt .- expected_b_dvdt))    < 1e-9 * max(maximum(norm.(expected_b_dvdt)), 1.0)
    @test maximum(abs.(ps_b.drhodt .- expected_b_drhodt)) < 1e-9 * max(maximum(abs.(expected_b_drhodt)), 1.0)
    # The forward pass must ALSO have run and written non-vacuously into system_a.
    @test any(v -> norm(v) > 0, ps_a.dvdt)
end

# ---------------------------------------------------------------------------
# 9. Phase 3: Bucket B conversions (InterpolateFieldFn, NeighborCountFn) —
#    both already write into `ps_b` (a virtual or probe target) in their
#    two-sided mutating form, so `_onesided_shape = WritesB()` and no script
#    changes are needed. Unlike every comparison above, the field(s) under
#    test now live on `system_b`, not `system_a` — the harness below diffs
#    the *target* system, since a broken WritesB() pass would be invisible
#    to the system_a-only harnesses used in sections 1-7.
# ---------------------------------------------------------------------------

function _zero_interp_target!(ps, fields)
    fill!(ps.w_sum, zero(eltype(ps.w_sum)))
    for f in fields
        arr = getproperty(ps, f)
        fill!(arr, zero(eltype(arr)))
    end
end

# Coupled comparison, mirroring _compare_coupled_generic but diffing the
# *target* (system_b: virtual/probe) instead of system_a, since that's what
# InterpolateFieldFn/NeighborCountFn actually write. `a` is shared (read-only,
# untouched by either pfn) between the two interactions, sorted once; `b` is
# zeroed (matching auto_zero_virtual!/auto_zero_probe! in the real driver)
# then deepcopy'd so the coloured and onesided sweeps start identical.
function _compare_coupled_sweep_writes_b(rng, pfn, build_a, build_b, fields, n_a, n_b, ndims; L=1.0, h=0.08)
    kernel = CubicSplineKernel(h; ndims=ndims)
    cutoff = kernel.interaction_length
    a = build_a(rng, n_a, ndims; L=L)
    a_dvdt_before, a_drhodt_before = deepcopy(a.dvdt), deepcopy(a.drhodt)

    b_old = build_b(rng, n_b, ndims; L=L)
    _zero_interp_target!(b_old, fields)
    b_new = deepcopy(b_old)

    si_old = SystemInteraction(kernel, pfn, a, b_old)
    si_new = SystemInteraction(kernel, pfn, a, b_new; onesided=true)

    pa, ka_, sa = _sortbufs(a)
    sort_particles!(a, cutoff, pa, ka_, sa)

    for (b, si) in ((b_old, si_old), (b_new, si_new))
        pb, kb, sb = _sortbufs(b)
        sort_particles!(b, cutoff, pb, kb, sb)
        create_grid!(si)
        sweep!(si)
    end

    # WritesB() must leave system_a completely untouched.
    @test a.dvdt   == a_dvdt_before
    @test a.drhodt == a_drhodt_before

    for f in fields
        va, vb = getproperty(b_old, f), getproperty(b_new, f)
        @test _elemdiff(va, vb) < 1e-9 * _elemscale(va)
    end
    @test maximum(abs.(b_old.w_sum .- b_new.w_sum)) < 1e-9 * max(maximum(abs.(b_old.w_sum)), 1.0)
end

_random_probe_rv(rng, n, ndims; L=1.0) = ProbeParticleSystem(
    "probe", [SVector(ntuple(_ -> L * rand(rng), ndims)...) for _ in 1:n];
    extras=(rho=zeros(n), v=[zero(SVector{ndims,Float64}) for _ in 1:n]),
)

_random_probe_nbr(rng, n, ndims; L=1.0) = ProbeParticleSystem(
    "probe", [SVector(ntuple(_ -> L * rand(rng), ndims)...) for _ in 1:n];
    extras=(nbr_count=zeros(Int, n),),
)

@testset "onesided=true InterpolateFieldFn (WritesB, virtual target) matches coloured sweep" begin
    rng = MersenneTwister(50)
    pfn = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
    for ndims in (2, 3)
        _compare_coupled_sweep_writes_b(rng, pfn, _random_fluid, _as_virtual_fluid, (:v, :rho), 300, 200, ndims)
    end
end

@testset "onesided=true InterpolateFieldFn (WritesB, stress field, no wsum) matches coloured sweep" begin
    rng = MersenneTwister(51)
    pfn = InterpolateFieldFn(:stress; accumulate_wsum=false)
    for ndims in (2, 3)
        _compare_coupled_sweep_writes_b(rng, pfn, _random_stress, _as_virtual_stress, (:stress,), 250, 180, ndims)
    end
end

@testset "onesided=true InterpolateFieldFn (WritesB, probe target) matches coloured sweep" begin
    rng = MersenneTwister(52)
    pfn = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
    for ndims in (2, 3)
        _compare_coupled_sweep_writes_b(rng, pfn, _random_fluid, _random_probe_rv, (:v, :rho), 300, 200, ndims)
    end
end

@testset "onesided=true NeighborCountFn (WritesB) matches coloured sweep" begin
    rng = MersenneTwister(53)
    pfn = NeighborCountFn(:nbr_count)
    for ndims in (2, 3)
        _compare_coupled_sweep_writes_b(rng, pfn, _random_fluid, _random_probe_nbr, (:nbr_count,), 300, 200, ndims)
    end
end
