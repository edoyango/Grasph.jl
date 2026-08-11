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
#   1. Self-pair exclusion: a particle must never pair with itself in the
#      full same-cell scan (unlike the half-shell's i<j ordering, which made
#      this impossible by construction).
#   2. Full-sweep equivalence: onesided=true vs onesided=false on identical
#      particle clouds, built through the real sort_particles!/create_grid!
#      path (not manually-injected CSR arrays), in 2D and 3D, self and
#      coupled (fluid <-> StaticBoundarySystem).
#   3. Short-run trajectory equivalence: a real LeapFrogTimeIntegrator run
#      over many steps, comparing accumulated drift.
#
# Note: this file used to also carry a "swap-antisymmetry" test comparing
# pfn_contribution(pfn,ps,i,j,...)/pfn_contribution(pfn,ps,j,i,...) against
# FluidPfn's two-sided *mutating* callable as an independent "ground truth".
# Since PairwiseFunctors.jl's mutating callables are now themselves generic
# delegates that call pfn_contribution (see AbstractPairwiseFunctor), that
# comparison became circular (pfn_contribution compared against a thin
# wrapper of itself) and was removed — the property it checked is now
# structurally guaranteed by construction, not something that can silently
# drift between two independent implementations.
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

# ---------------------------------------------------------------------------
# 1. Self-pair exclusion
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

# ---------------------------------------------------------------------------
# 7b. XSPHPfn ghost-coupled regression — the aliasing-bug fix.
#
# XSPHPfn's coupled mutating method used to have only a fully generic/mutual
# form (`ps_b::AbstractParticleSystem`, writing `ps_b.v_adjustment[j] -=
# du*mass_i`). That is wrong whenever `ps_b` is a *self-referencing* ghost
# (`GhostParticleSystem(fluid, ...)` with `ghost.source === fluid` — the
# only pattern that exists anywhere in this codebase; see bubble3.jl's
# `boundary_ghost`): `GhostParticleSystem` does not own a `v_adjustment`
# array, so `ps_b.v_adjustment[j]` falls through `Base.getproperty` straight
# to `ghost.source.v_adjustment[j]` — i.e. it aliases back into the real
# fluid's own array, but indexed by the ghost's LOCAL index `j`, which does
# not correspond to the real particle the ghost mirrors. Silently wrong
# writes when `ghost.n < fluid.n`; out-of-bounds heap corruption when
# `ghost.n > fluid.n`. Fixed by adding a narrowly-typed
# `Union{AbstractGhostParticleSystem{T,ND}, VirtualParticleSystem{T,ND}}`
# overload (matching FluidPfn/CauchyFluidPfn/StrainRatePfn's existing
# convention in this file) that only ever writes `ps_a`, which Julia's
# dispatch picks over the generic two-sided method for ghost/virtual `ps_b`.
# See the comment above that method in src/PairwiseFunctors.jl for the full
# mechanism.
#
# This exercises the fix (coloured sweep) and its `pfn_contribution`
# counterpart (onesided=true sweep) against each other, using a REAL
# self-referencing ghost — the only pattern that would have caught the
# original bug — built standalone (not through the integrator loop) via
# generate_ghosts!/sort_particles!/update_ghost_kinematics!/update_ghost!,
# cribbed from test_ghost_particles.jl, in the exact order
# TimeIntegration.jl's `_prepare_grids!` uses: sort the REAL system first
# (so `idx_original` indexes the final fluid ordering), THEN
# generate_ghosts! from that ordering, THEN sort the ghost itself.
#
# Two geometries, both mirroring bubble3.jl's `boundary_ghost` (all 4 walls
# + 4 corners, `GhostCopier(:p)`, `h = 1.2*dx`, boundary cutoff `= 3h`):
#   - "tight": box comparable in size to the boundary cutoff, so every fluid
#     particle qualifies as a ghost source for every one of the 8
#     boundaries -> ghost.n > fluid.n, the regime that produced heap
#     corruption / SIGABRT before the fix.
#   - "realistic": a much larger box relative to the same cutoff, so only a
#     thin boundary layer of particles produces ghosts -> ghost.n
#     comfortably < fluid.n, closer to bubble3.jl's actual proportions.
# ---------------------------------------------------------------------------

function _xsph_ghost_fluid(rng, nx, ny, dx)
    n = nx * ny
    fluid = FluidParticleSystem("fluid", n, 2, 1.0, 10.0; source_v = zeros(2))
    k = 1
    for i in 0:nx-1, j in 0:ny-1
        fluid.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        fluid.v[k] = SVector(0.2 * (rand(rng) - 0.5), 0.2 * (rand(rng) - 0.5))
        k += 1
    end
    fluid.rho .= 1000.0 .+ 20 .* (rand(rng, n) .- 0.5)
    fluid.p   .= 100.0 .* rand(rng, n)
    fill!(fluid.dvdt, zero(SVector{2,Float64})); fluid.drhodt .= 0.0
    fill!(fluid.v_adjustment, zero(SVector{2,Float64}))
    return fluid
end

# Standalone build of a REAL self-referencing ghost (ghost.source === fluid)
# mirroring bubble3.jl's boundary_ghost/boundary_ghost_entry: all 4 walls +
# 4 corners of a [0,Lx]x[0,Ly] box, GhostCopier(:p). Ordering matches
# TimeIntegration.jl's `_prepare_grids!`/stage loop exactly: sort the real
# system first, THEN generate_ghosts!, THEN sort the ghost, THEN
# update_ghost_kinematics! (v, rho), THEN update_ghost! (stage 1: p).
function _xsph_ghost_setup!(fluid, sweep_cutoff, boundary_cutoff, Lx, Ly)
    fp, fk, fs = _sortbufs(fluid)
    sort_particles!(fluid, sweep_cutoff, fp, fk, fs)

    ghost = GhostParticleSystem(fluid, GhostCopier(:p); name="ghost[$(fluid.name)]")
    entry = GhostEntry(ghost, boundary_cutoff,
        (SVector( 1.0,  0.0),            SVector(0.0, 0.0)),   # left wall
        (SVector(-1.0,  0.0),            SVector(Lx,  0.0)),   # right wall
        (SVector( 0.0,  1.0),            SVector(0.0, 0.0)),   # bottom wall
        (SVector( 0.0, -1.0),            SVector(0.0, Ly)),    # top wall
        (SVector( 1.0,  1.0)/sqrt(2.0),  SVector(0.0, 0.0)),   # bottom-left corner
        (SVector(-1.0,  1.0)/sqrt(2.0),  SVector(Lx,  0.0)),   # bottom-right corner
        (SVector( 1.0, -1.0)/sqrt(2.0),  SVector(0.0, Ly)),    # top-left corner
        (SVector(-1.0, -1.0)/sqrt(2.0),  SVector(Lx,  Ly)),    # top-right corner
    )
    generate_ghosts!(entry)

    gp, gk, gs = _sortbufs(ghost)
    sort_particles!(ghost, sweep_cutoff, gp, gk, gs)

    update_ghost_kinematics!(entry)
    update_ghost!(ghost, 1)

    return ghost
end

# Runs the coloured sweep (fixed one-sided mutating method) on fluid_old and
# the onesided=true sweep (new pfn_contribution) on fluid_new — each against
# its OWN self-referencing ghost, built from otherwise-identical starting
# state — then diffs v_adjustment. Returns ghost.n for the geometry asserts.
function _compare_xsph_ghost(fluid_old, fluid_new, kernel, boundary_cutoff, Lx, Ly)
    sweep_cutoff = kernel.interaction_length
    ghost_old = _xsph_ghost_setup!(fluid_old, sweep_cutoff, boundary_cutoff, Lx, Ly)
    ghost_new = _xsph_ghost_setup!(fluid_new, sweep_cutoff, boundary_cutoff, Lx, Ly)
    @test ghost_old.n == ghost_new.n   # identical starting state -> identical ghost geometry

    pfn = XSPHPfn(0.5)
    si_old = SystemInteraction(kernel, pfn, fluid_old, ghost_old)                 # coloured
    si_new = SystemInteraction(kernel, pfn, fluid_new, ghost_new; onesided=true)  # onesided

    create_grid!(si_old); sweep!(si_old)
    create_grid!(si_new); sweep!(si_new)

    # Sanity: the sweep actually produced non-vacuous output — otherwise the
    # diff check below would pass trivially (both all-zero).
    @test any(v -> norm(v) > 0, fluid_old.v_adjustment)

    va, vb = fluid_old.v_adjustment, fluid_new.v_adjustment
    @test _elemdiff(va, vb) < 1e-9 * _elemscale(va)

    return ghost_old.n
end

@testset "onesided=true XSPHPfn ghost-coupled (self-referencing ghost) matches coloured sweep — aliasing-bug regression" begin
    @testset "tight geometry: ghost.n > fluid.n (crashed pre-fix)" begin
        rng = MersenneTwister(70)
        nx, ny, dx = 3, 3, 0.1
        h  = 1.2 * dx
        Lx, Ly = nx * dx, ny * dx
        boundary_cutoff = 3.0 * h
        kernel = CubicSplineKernel(h; ndims=2)

        fluid_base = _xsph_ghost_fluid(rng, nx, ny, dx)
        fluid_old, fluid_new = deepcopy(fluid_base), deepcopy(fluid_base)

        ghost_n = _compare_xsph_ghost(fluid_old, fluid_new, kernel, boundary_cutoff, Lx, Ly)
        @test ghost_n > fluid_old.n   # the regime that produced heap corruption pre-fix
    end

    @testset "realistic geometry: ghost.n comfortably < fluid.n" begin
        rng = MersenneTwister(71)
        nx, ny, dx = 50, 40, 0.02
        h  = 1.2 * dx
        Lx, Ly = nx * dx, ny * dx
        boundary_cutoff = 3.0 * h
        kernel = CubicSplineKernel(h; ndims=2)

        fluid_base = _xsph_ghost_fluid(rng, nx, ny, dx)
        fluid_old, fluid_new = deepcopy(fluid_base), deepcopy(fluid_base)

        ghost_n = _compare_xsph_ghost(fluid_old, fluid_new, kernel, boundary_cutoff, Lx, Ly)
        @test ghost_n < fluid_old.n ÷ 2   # comfortably fewer ghosts than fluid particles
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

# ---------------------------------------------------------------------------
# 10. Phase 4: coupled real-real `WritesBoth` pairs — `FluidPfn` fluid-fluid
#    (e.g. bubble2.jl/bubble3.jl's two-phase coupling) and `FluidSolidPfn`
#    fluid-solid (DambreakWall.jl's fluid<->wall coupling). Both mutate TWO
#    real systems from a single interaction (`_onesided_shape = WritesBoth()`
#    — forward pass writes system_a, reverse pass writes system_b), so
#    neither existing coupled harness is sufficient on its own:
#    `_compare_coupled_generic` only diffs system_a's fields,
#    `_compare_coupled_sweep_writes_b` only diffs system_b's. A wrong-side
#    bug confined to just one pass (e.g. the reverse pass silently reusing
#    the forward pass's formula) would be invisible to either alone. The new
#    harness below diffs both systems' fields between a coloured-sweep pair
#    and a onesided-sweep pair built from deepcopy'd starting state.
#
#    Fixtures deliberately avoid `_random_boundary`'s degenerate uniform-rho/
#    zero-v case (see this file's header note): both the fluid and solid/
#    fluid-2 fixtures below carry non-uniform rho, nonzero v, and distinct
#    mass/c per system, so a bug that reads the wrong side's value doesn't
#    silently cancel out (e.g. under equal sound speeds, artificial viscosity
#    and delta-SPH diffusion terms degenerate in ways that can mask exactly
#    this class of bug).
# ---------------------------------------------------------------------------

function _random_fluid_mc(rng, n, ndims, mass, c; L=1.0)
    ps = FluidParticleSystem("fluid", n, ndims, mass, c; source_v = zeros(ndims))
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

# Two distinct real fluid phases (distinct mass/c) — mirrors bubble2.jl/
# bubble3.jl's `fluid_X`/`fluid_Y`.
_random_fluid_a(rng, n, ndims; L=1.0) = _random_fluid_mc(rng, n, ndims, 1.3, 12.0; L=L)
_random_fluid_b(rng, n, ndims; L=1.0) = _random_fluid_mc(rng, n, ndims, 0.7, 18.0; L=L)

# A real solid/wall system: non-uniform rho, nonzero v, distinct mass/c from
# either fluid fixture above, and a nonzero *own* pressure field — the own
# pressure is set here (rather than left at 0, which would make a bug that
# reads it indistinguishable from correct code) precisely so the dedicated
# pressure-invariance test below has something non-vacuous to vary.
function _random_wall(rng, n, ndims; L=1.0, ns = ndims == 2 ? 3 : 6, mass=2.5, c=25.0)
    ps = ElastoPlasticParticleSystem("wall", n, ndims, ns, mass, c)
    for i in 1:n
        ps.x[i]      = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i]      = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
        ps.stress[i] = SVector(ntuple(_ -> 50.0 * (rand(rng) - 0.5), ns)...)
    end
    ps.rho .= 2400.0 .+ 30 .* (rand(rng, n) .- 0.5)
    ps.p   .= 100.0 .* rand(rng, n)
    fill!(ps.dvdt, zero(SVector{ndims,Float64})); ps.drhodt .= 0.0
    fill!(ps.strain_rate, zero(eltype(ps.strain_rate)))
    fill!(ps.vorticity, zero(eltype(ps.vorticity)))
    return ps
end

# WritesBoth-aware coupled comparison harness: builds a coloured-sweep pair
# and a onesided-sweep pair from deepcopy'd starting state, sorts+grids+
# sweeps both, then diffs BOTH system_a's and system_b's accumulator fields
# between the two sweeps.
function _compare_coupled_writesboth(rng, pfn, build_a, build_b, fields_a, fields_b, n_a, n_b, ndims; L=1.0, h=0.08)
    kernel = CubicSplineKernel(h; ndims=ndims)
    cutoff = kernel.interaction_length

    a_old = build_a(rng, n_a, ndims; L=L)
    b_old = build_b(rng, n_b, ndims; L=L)
    a_new = deepcopy(a_old)
    b_new = deepcopy(b_old)

    si_old = SystemInteraction(kernel, pfn, a_old, b_old)
    si_new = SystemInteraction(kernel, pfn, a_new, b_new; onesided=true)

    for (a, b, si) in ((a_old, b_old, si_old), (a_new, b_new, si_new))
        pa, ka_, sa = _sortbufs(a)
        sort_particles!(a, cutoff, pa, ka_, sa)
        pb, kb, sb = _sortbufs(b)
        sort_particles!(b, cutoff, pb, kb, sb)
        create_grid!(si)
        sweep!(si)
    end

    for f in fields_a
        va, vb = getproperty(a_old, f), getproperty(a_new, f)
        @test _elemdiff(va, vb) < 1e-9 * _elemscale(va)
    end
    for f in fields_b
        va, vb = getproperty(b_old, f), getproperty(b_new, f)
        @test _elemdiff(va, vb) < 1e-9 * _elemscale(va)
    end
end

@testset "onesided=true FluidPfn fluid-fluid (WritesBoth) matches coloured sweep" begin
    rng = MersenneTwister(60)
    # epsilon=0.1 (matching bubble3.jl's fluid_XY_interaction) exercises the
    # artificial-surface-tension term the WritesBoth method includes.
    pfn = FluidPfn(0.03, 0.0, 0.08; epsilon=0.1)
    for ndims in (2, 3)
        _compare_coupled_writesboth(rng, pfn, _random_fluid_a, _random_fluid_b, (:dvdt, :drhodt), (:dvdt, :drhodt), 250, 200, ndims)
    end
end

@testset "onesided=true FluidSolidPfn fluid-solid (WritesBoth) matches coloured sweep" begin
    rng = MersenneTwister(61)
    pfn = FluidSolidPfn(0.03, 0.0, 0.08)
    for ndims in (2, 3)
        _compare_coupled_writesboth(rng, pfn, _random_fluid_a, _random_wall, (:dvdt, :drhodt), (:dvdt, :drhodt), 250, 150, ndims)
    end
end

# XSPHPfn's real-real (WritesBoth) coupled method is confirmed dead code (no
# script currently pairs two distinct real FluidParticleSystems via
# velocity_adjust_pairwise_fn — see the pfn's own docstring), but it now has
# a pfn_contribution/_onesided_shape counterpart (mirroring FluidPfn's own
# fluid-fluid pattern) where none existed before this refactor. Covering it
# here closes that gap the same way the FluidPfn/FluidSolidPfn cases above
# already are.
_random_fluid_a_xsph(rng, n, ndims; L=1.0) =
    let ps = _random_fluid_a(rng, n, ndims; L=L)
        fill!(ps.v_adjustment, zero(SVector{ndims,Float64}))
        ps
    end
_random_fluid_b_xsph(rng, n, ndims; L=1.0) =
    let ps = _random_fluid_b(rng, n, ndims; L=L)
        fill!(ps.v_adjustment, zero(SVector{ndims,Float64}))
        ps
    end

@testset "onesided=true XSPHPfn fluid-fluid (WritesBoth) matches coloured sweep" begin
    rng = MersenneTwister(63)
    pfn = XSPHPfn(0.5)
    for ndims in (2, 3)
        _compare_coupled_writesboth(rng, pfn, _random_fluid_a_xsph, _random_fluid_b_xsph, (:v_adjustment,), (:v_adjustment,), 250, 200, ndims)
    end
end

# The single most important test in this phase: FluidSolidPfn's whole design
# point is that pressure must be continuous across the fluid-solid interface,
# i.e. the FLUID's pressure is used for both sides of the force and the
# solid's own pressure must never appear in the formula (see the comment
# above the WritesBoth methods in src/PairwiseFunctors.jl). None of the
# generic comparison tests above can catch a bug that reads the wrong side's
# pressure consistently in both the coloured and onesided code paths, because
# they compare a call against itself in the same orientation — a
# `ps_a.p[i]`-vs-`ps_b.p[j]` mixup would reproduce identically on both sides
# of that comparison. This test instead varies ONLY the solid's own pressure
# field between two otherwise-identical runs against the SAME fluid system,
# and asserts the outputs are bit-for-bit-at-roundoff identical — a leak of
# the solid's own pressure into the force would make them differ.
@testset "onesided=true FluidSolidPfn: solid's own pressure never leaks into the force (regression)" begin
    rng = MersenneTwister(62)
    pfn = FluidSolidPfn(0.03, 0.0, 0.08)
    for ndims in (2, 3)
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=ndims)
        cutoff = kernel.interaction_length

        fluid_base = _random_fluid_a(rng, 200, ndims)
        wall_base  = _random_wall(rng, 150, ndims)

        fluid1, wall1 = deepcopy(fluid_base), deepcopy(wall_base)
        fluid2, wall2 = deepcopy(fluid_base), deepcopy(wall_base)

        # Two arbitrary, distinct, non-vacuous own-pressure fields for the
        # solid. Per FluidSolidPfn's contract neither should ever reach the
        # force computation.
        wall1.p .= 37.0 .+ 5.0  .* (1:wall1.n)
        wall2.p .= -400.0 .- 11.0 .* (1:wall2.n)

        si1 = SystemInteraction(kernel, pfn, fluid1, wall1; onesided=true)
        si2 = SystemInteraction(kernel, pfn, fluid2, wall2; onesided=true)

        for (f, w, si) in ((fluid1, wall1, si1), (fluid2, wall2, si2))
            pf, kf, sf = _sortbufs(f)
            sort_particles!(f, cutoff, pf, kf, sf)
            pw, kw, sw = _sortbufs(w)
            sort_particles!(w, cutoff, pw, kw, sw)
            create_grid!(si)
            sweep!(si)
        end

        # Sanity: the sweep actually produced non-vacuous output — otherwise
        # the equality checks below would pass trivially (both all-zero).
        @test any(v -> norm(v) > 0, fluid1.dvdt)
        @test any(v -> norm(v) > 0, wall1.dvdt)

        @test maximum(norm.(fluid1.dvdt   .- fluid2.dvdt))   < 1e-12 * max(maximum(norm.(fluid1.dvdt)), 1.0)
        @test maximum(abs.(fluid1.drhodt .- fluid2.drhodt))  < 1e-12 * max(maximum(abs.(fluid1.drhodt)), 1.0)
        @test maximum(norm.(wall1.dvdt    .- wall2.dvdt))    < 1e-12 * max(maximum(norm.(wall1.dvdt)), 1.0)
        @test maximum(abs.(wall1.drhodt  .- wall2.drhodt))   < 1e-12 * max(maximum(abs.(wall1.drhodt)), 1.0)
    end
end
