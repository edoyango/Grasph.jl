using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Reduced-scale, in-context validation of `onesided=true` against the default
# coloured sweep for EP_ColumnCollapse2.jl's interaction shape ("Cluster C"):
#
#   - fluid = ElastoPlasticParticleSystem(ND=2, NS=4 plane-strain Voigt),
#             Drucker-Prager elastoplastic soil (ElastoPlasticStressUpdater),
#             gravity in -y. 4-stage pfn tuple / state_updater layout:
#               stage 1: nothing                                    / (interp on coupled)
#               stage 2: ZeroFieldUpdater(:strain_rate,:vorticity)  / StrainRateVorticityPfn
#               stage 3: ElastoPlasticStressUpdater                 / (interp on coupled)
#               stage 4: nothing                                    / CauchyFluidPfn
#   - TWO distinct VirtualParticleSystem boundaries, NO ghosts at all:
#       - bottom_virt: no-slip floor,   v_mult = [-1,-1]
#       - left_virt:   free-slip wall,  v_mult = [-1, 1]
#     each backed by its own StressParticleSystem(ND=2, NS=4) source, coupled
#     to `fluid` via InterpolateFieldFn (WritesB — accumulates INTO the
#     virtual) at stages 1/3 and VirtualNormUpdater at stages 2/4.
#   - LeapFrogTimeIntegrator([fluid], [fluid_self, fluid_bottom, fluid_left];
#       virtual_systems=(bottom_virt, left_virt)).
#
# This is the pure "virtual-boundary" counterpart to the soil/ghost harness:
# no ghosts anywhere, and — unlike a harness that only diffs the real system —
# this one also confirms the virtual systems' OWN accumulated state (v, rho,
# stress) matches between the coloured and onesided runs, since
# InterpolateFieldFn's WritesB pass writes into the virtual, not into `fluid`;
# a broken WritesB path would be entirely invisible to a fluid-only diff.
#
# Mirrors the `_dambreak_like` / "short-run trajectory equivalence" / "long-run
# physical invariants" shape in test_onesided_sweep.jl (sections 5-6).
# ---------------------------------------------------------------------------

_elemscale(v::AbstractVector{<:Real})    = max(maximum(abs, v), 1.0)
_elemscale(v::AbstractVector{<:SVector}) = max(maximum(norm, v), 1.0)
_elemdiff(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})     = maximum(abs.(a .- b))
_elemdiff(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector}) = maximum(norm.(a .- b))
_anynan(v::AbstractVector{<:Real})    = any(isnan, v)
_anynan(v::AbstractVector{<:SVector}) = any(x -> any(isnan, x), v)

# Deterministic regular-grid builder — no RNG needed, exactly mirroring
# EP_ColumnCollapse2.jl's own deterministic particle placement (scaled down:
# nfx x nfy = 15x10 -> 150 fluid particles, ~a few dozen virtual particles per
# wall, vs. the real script's 100x50 fluid grid). Called independently for
# each of the "old" (coloured) and "new" (onesided) runs so they start from
# bit-identical, non-aliased state.
function _ep_column_like(; nfx=15, nfy=10, nbx=10)
    dx_spacing          = 0.002
    h_sph               = 1.2 * dx_spacing
    rho0                = 1850.0
    soil_friction_angle = 19.8 * π / 180.0

    E        = 0.84e6
    nu       = 0.3
    psi      = 0.0
    cohesion = 0.0
    c_sound  = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2nu)))

    n_fluid    = nfx * nfy
    fluid_mass = rho0 * dx_spacing^2

    fluid = ElastoPlasticParticleSystem(
        "fluid", n_fluid, 2, 4, fluid_mass, c_sound;
        source_v = [0.0, -9.81],
        state_updater = (
            nothing,                                                          # stage 1: no update
            ZeroFieldUpdater(:strain_rate, :vorticity),                       # stage 2: zero before strain sweep
            ElastoPlasticStressUpdater(E, nu, soil_friction_angle, psi, cohesion),
            nothing,                                                          # stage 4: no update
        ),
    )
    let k = 1
        for i in 0:nfx-1, j in 0:nfy-1
            fluid.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
            k += 1
        end
    end
    fill!(fluid.v, zero(SVector{2,Float64}))
    fluid.rho .= rho0
    update_state!(fluid, 3)   # initialize stress via ElastoPlasticStressUpdater (stage 3), as the real script does

    # -----------------------------------------------------------------------
    # Bottom virtual boundary — no-slip (v_mult = [-1,-1]).
    # -----------------------------------------------------------------------
    n_bottom      = 3 * (nbx + 3)
    bottom_source = StressParticleSystem("bottom_boundary", n_bottom, 2, 4, fluid_mass, c_sound)
    let k = 1
        for i in 1:nbx+3, j in 1:3
            bottom_source.x[k] = SVector((i - 3.5) * dx_spacing, -(j - 0.5) * dx_spacing)
            k += 1
        end
    end
    bottom_source.rho .= rho0
    fill!(bottom_source.v, zero(SVector{2,Float64}))

    bottom_virt = VirtualParticleSystem(
        bottom_source, "bottom_virt", n_bottom, 2, fluid_mass, c_sound;
        zero_fields   = (:v, :rho, :stress),
        state_updater = (
            nothing,                                                          # stage 1: no update
            VirtualNormUpdater(SVector(-1.0, -1.0), :v, :rho),                # stage 2: normalize v+rho, flip both
            nothing,                                                          # stage 3: no update
            VirtualNormUpdater(SVector(-1.0, -1.0), :stress),                 # stage 4: normalize stress
        ),
    )

    # -----------------------------------------------------------------------
    # Left virtual boundary — free-slip (v_mult = [-1,1], negate x only).
    # -----------------------------------------------------------------------
    n_left      = 3 * nfy
    left_source = StressParticleSystem("left_boundary", n_left, 2, 4, fluid_mass, c_sound)
    let k = 1
        for j in 0:nfy-1, col in 1:3
            left_source.x[k] = SVector(-(col - 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
            k += 1
        end
    end
    left_source.rho .= rho0
    fill!(left_source.v, zero(SVector{2,Float64}))

    left_virt = VirtualParticleSystem(
        left_source, "left_virt", n_left, 2, fluid_mass, c_sound;
        zero_fields   = (:v, :rho, :stress),
        state_updater = (
            nothing,
            VirtualNormUpdater(SVector(-1.0, 1.0), :v, :rho),                 # stage 2: free-slip (negate x only)
            nothing,
            VirtualNormUpdater(SVector(-1.0, 1.0), :stress),                  # stage 4: normalize stress
        ),
    )

    kernel = CubicSplineKernel(h_sph; ndims=2)
    return kernel, fluid, bottom_virt, left_virt
end

# Builds the real script's 4-pfn-stage interaction triple (fluid self +
# fluid<->bottom_virt + fluid<->left_virt) for a given (fluid, bottom_virt,
# left_virt) instance, sharing one set of (stateless) pfn instances across
# both onesided settings — exactly as EP_ColumnCollapse2.jl reuses `sr_pfn`/
# `kin_pfn`/`interp_vel`/`interp_str` across all three of its interactions.
function _ep_column_interactions(kernel, fluid, bottom_virt, left_virt; onesided::Bool)
    art_visc_alpha, art_visc_beta = 0.1, 0.1
    sr_pfn     = StrainRateVorticityPfn()
    kin_pfn    = CauchyFluidPfn(art_visc_alpha, art_visc_beta, kernel.h)
    interp_vel = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
    interp_str = InterpolateFieldFn(:stress; accumulate_wsum=false)

    fluid_self   = SystemInteraction(kernel, (nothing, sr_pfn, nothing, kin_pfn), fluid; onesided=onesided)
    fluid_bottom = SystemInteraction(kernel, (interp_vel, sr_pfn, interp_str, kin_pfn), fluid, bottom_virt; onesided=onesided)
    fluid_left   = SystemInteraction(kernel, (interp_vel, sr_pfn, interp_str, kin_pfn), fluid, left_virt; onesided=onesided)

    return fluid_self, fluid_bottom, fluid_left
end

@testset "short-run trajectory equivalence: onesided=true vs onesided=false" begin
    kernel, fluid_old, bottom_virt_old, left_virt_old = _ep_column_like()
    _,      fluid_new, bottom_virt_new, left_virt_new = _ep_column_like()

    fluid_self_old, fluid_bottom_old, fluid_left_old = _ep_column_interactions(
        kernel, fluid_old, bottom_virt_old, left_virt_old; onesided=false)
    fluid_self_new, fluid_bottom_new, fluid_left_new = _ep_column_interactions(
        kernel, fluid_new, bottom_virt_new, left_virt_new; onesided=true)

    lf_old = LeapFrogTimeIntegrator([fluid_old], [fluid_self_old, fluid_bottom_old, fluid_left_old];
                                    virtual_systems=(bottom_virt_old, left_virt_old))
    lf_new = LeapFrogTimeIntegrator([fluid_new], [fluid_self_new, fluid_bottom_new, fluid_left_new];
                                    virtual_systems=(bottom_virt_new, left_virt_new))

    # dt = 0.1, matching the other onesided-integration harnesses in this test
    # suite (test_onesided_sweep.jl's `_dambreak_like`, test_onesided_integration_soil3d.jl).
    # This is far above what the real script's own CFL=0.1 policy (dt = CFL*h/c
    # ~= 1e-5 for this stiff a soil) would give — deliberately so: the physically
    # "correct" dt makes every field's magnitude stay well under the `_elemscale`
    # floor of 1.0 after only 75-250 steps (e.g. max|v| ~ 1e-6), which would turn
    # the "relative" tolerance checks below into much-looser absolute ones. dt=0.1
    # was verified separately (not NaN/Inf through 250 steps at this reduced
    # particle count) to give O(1)-and-above field magnitudes so the tolerances
    # below are genuinely relative checks, not floor-dominated ones.
    dt = 0.1
    nsteps = 75
    time_integrate!(lf_old, nsteps, nsteps + 1, nsteps + 1, dt, nothing; print_timer=false)
    time_integrate!(lf_new, nsteps, nsteps + 1, nsteps + 1, dt, nothing; print_timer=false)

    # No NaNs anywhere, in either run, on the real system or either virtual.
    for ps in (fluid_old, fluid_new)
        @test !_anynan(ps.x)
        @test !_anynan(ps.v)
        @test !_anynan(ps.rho)
        @test !_anynan(ps.stress)
        @test !_anynan(ps.strain_rate)
        @test !_anynan(ps.vorticity)
        @test !_anynan(ps.strain)
        @test !_anynan(ps.strain_p)
    end
    for vps in (bottom_virt_old, bottom_virt_new, left_virt_old, left_virt_new)
        @test !_anynan(vps.v)
        @test !_anynan(vps.rho)
        @test !_anynan(vps.stress)
    end

    # Real (fluid) system: position tightest, velocity/rho looser, the
    # higher-derivative elastoplastic-stress-model quantities (stress,
    # strain_rate, vorticity, strain, strain_p — all driven through repeated
    # per-stage kernel sums and the yield-surface correction) loosest.
    @test _elemdiff(fluid_old.x, fluid_new.x)                     < 1e-8 * _elemscale(fluid_old.x)
    @test _elemdiff(fluid_old.v, fluid_new.v)                     < 1e-6 * _elemscale(fluid_old.v)
    @test _elemdiff(fluid_old.rho, fluid_new.rho)                 < 1e-6 * _elemscale(fluid_old.rho)
    @test _elemdiff(fluid_old.stress, fluid_new.stress)           < 1e-6 * _elemscale(fluid_old.stress)
    @test _elemdiff(fluid_old.strain_rate, fluid_new.strain_rate) < 1e-6 * _elemscale(fluid_old.strain_rate)
    @test _elemdiff(fluid_old.vorticity, fluid_new.vorticity)     < 1e-6 * _elemscale(fluid_old.vorticity)
    @test _elemdiff(fluid_old.strain, fluid_new.strain)           < 1e-6 * _elemscale(fluid_old.strain)
    @test _elemdiff(fluid_old.strain_p, fluid_new.strain_p)       < 1e-6 * _elemscale(fluid_old.strain_p)

    # Virtual boundaries' own accumulated state — this is what exercises
    # InterpolateFieldFn's WritesB pass; a bug there would be invisible above.
    for (vold, vnew, label) in (
        (bottom_virt_old, bottom_virt_new, "bottom_virt"),
        (left_virt_old,   left_virt_new,   "left_virt"),
    )
        @test _elemdiff(vold.v, vnew.v)         < 1e-6 * _elemscale(vold.v)
        @test _elemdiff(vold.rho, vnew.rho)     < 1e-6 * _elemscale(vold.rho)
        @test _elemdiff(vold.stress, vnew.stress) < 1e-6 * _elemscale(vold.stress)
    end
end

@testset "long-run physical invariants (onesided=true)" begin
    kernel, fluid, bottom_virt, left_virt = _ep_column_like()

    fluid_self, fluid_bottom, fluid_left = _ep_column_interactions(
        kernel, fluid, bottom_virt, left_virt; onesided=true)

    lf = LeapFrogTimeIntegrator([fluid], [fluid_self, fluid_bottom, fluid_left];
                                virtual_systems=(bottom_virt, left_virt))

    dt = 0.1   # matches the short-run testset above
    nsteps = 200   # ~2.7x the short-run step count
    time_integrate!(lf, nsteps, nsteps + 1, nsteps + 1, dt, nothing; print_timer=false)

    @test !_anynan(fluid.x)
    @test !_anynan(fluid.v)
    @test !_anynan(fluid.rho)
    @test !_anynan(fluid.stress)
    @test !_anynan(fluid.strain_rate)
    @test !_anynan(fluid.vorticity)
    @test all(!isinf, fluid.rho)
    @test all(x -> all(!isinf, x), fluid.x)
    @test all(x -> all(!isinf, x), fluid.v)

    # No particle should have travelled absurdly far given the short run and
    # small initial block (~0.03m x 0.02m) — a broken boundary coupling (a
    # particle leaking through the floor or past the left wall) would blow
    # this up.
    @test all(x -> all(abs.(x) .< 5.0), fluid.x)
end
