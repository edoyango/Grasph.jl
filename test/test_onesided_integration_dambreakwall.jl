using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Cluster F: onesided=true vs coloured, real-integrator equivalence test for
# DambreakWall.jl's interaction shape.
#
# DambreakWall.jl is the codebase's canonical `FluidSolidPfn` `WritesBoth()`
# mutual-coupling case (a fluid dambreak impacting an elastic concrete wall)
# -- source comments in src/PairwiseFunctors.jl explicitly cite this script
# as the motivating real-world case for the `WritesBoth()` mutual shape, so
# this harness matters more than most. It also covers:
#   - `FluidPfn` self-interaction (fluid)
#   - `CauchyFluidPfn` self-interaction (wall, stress-driven)
#   - FOUR separate `DynamicBoundarySystem` couplings (floor/left/right/top)
#     against the fluid, plus a fifth (floor) against the wall
#   - `StrainRateVorticityPfn` -> `HookeLawStressUpdater` -> `CauchyFluidPfn`
#     2-stage pipeline on the elastic wall
# No ghosts, no virtuals, no probes in this script.
#
# Running this through a real `LeapFrogTimeIntegrator`/`time_integrate!` loop
# (rather than the standalone per-pfn harnesses in test_onesided_sweep.jl)
# exercises the full multi-stage loop and every one of these pfn/system-type
# combinations together, in the same wiring the real script uses.
# ---------------------------------------------------------------------------

# Deterministic regular layout (mirrors DambreakWall.jl exactly: no
# randomness anywhere in the real script's setup -- positions come from
# regular grid scans, velocities start at zero, density starts uniform at
# rho0, wall stress starts at zero). Called twice independently below (not
# deepcopy'd) so the "old"/coloured and "new"/onesided runs start from
# bit-identical, non-aliased state.
#
# Unlike a naive uniform-at-rest dambreak setup, this does NOT stay
# degenerate (fluid/wall never producing non-trivial pairwise forces): the
# fluid block's bottom-right corner is placed within kernel cutoff of BOTH
# the floor boundary AND the wall simultaneously. The `DynamicBoundarySystem`
# coupling's velocity-mirroring formula depends on each particle's geometric
# distance to the boundary plane, so once gravity gives every particle a
# uniform velocity kick (step 1), the floor coupling immediately produces a
# spatially-*non*-uniform correction for the fluid particles nearest it
# (step 2) -- breaking the initial symmetry and feeding a genuinely
# spatially-varying density (hence pressure, via Tait EOS) into exactly the
# corner of the fluid block that sits against the wall. That is what makes
# `FluidSolidPfn`'s mutual coupling non-trivially exercised within this
# harness's short step budget, without resorting to injected randomness.
function _dambreakwall_like()
    dx             = 0.5
    h_sph          = 1.2 * dx
    rho0_water     = 1000.0
    c_sound        = 10.0 * sqrt(2.0 * 9.81 * 25.0)   # same physical constant as the real script
    art_visc_alpha = 0.02
    art_visc_beta  = 0.0
    gravity        = SVector(0.0, -9.81)

    rho0_wall = 2400.0
    E_wall    = 5.0e8
    nu_wall   = 0.2
    c_wall    = sqrt(E_wall * (1 - nu_wall) / (rho0_wall * (1 + nu_wall) * (1 - 2*nu_wall)))

    kernel = CubicSplineKernel(h_sph; ndims=2)

    # --- Fluid (water column) -- reduced from DambreakWall.jl's 50x50=2500 ---
    nfx, nfy   = 10, 15
    n_fluid    = nfx * nfy
    fluid_mass = rho0_water * dx^2

    fluid = FluidParticleSystem(
        "fluid", n_fluid, 2, fluid_mass, c_sound;
        source_v      = gravity,
        state_updater = (nothing, TaitEOSUpdater(rho0_water)),
    )
    let k = 1
        for j in 0:nfy-1, i in 0:nfx-1
            fluid.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
            k += 1
        end
    end
    fill!(fluid.v, zero(SVector{2,Float64}))
    fluid.rho .= rho0_water
    update_state!(fluid, 2)   # initialise pressure (== 0 at rho0, matching the script)

    # --- Concrete wall (elastic solid) -- reduced from 3x20=60 ---------------
    n_wall_x, n_wall_y = 3, 8
    n_wall    = n_wall_x * n_wall_y
    x_wall    = 5.0   # small gap (0.5) from the fluid's right edge (x=4.75):
                       # within kernel cutoff (1.2) from the start, so
                       # FluidSolidPfn is geometrically active immediately.
    wall_mass = rho0_wall * dx^2

    wall = ElastoPlasticParticleSystem(
        "wall", n_wall, 2, 4, wall_mass, c_wall;
        source_v      = gravity,
        state_updater = (
            ZeroFieldUpdater(:strain_rate, :vorticity),
            HookeLawStressUpdater(E_wall, nu_wall),
        ),
    )
    let k = 1
        for j in 0:n_wall_y-1, i in 0:n_wall_x-1
            wall.x[k] = SVector(x_wall + (i + 0.5)*dx, (j + 0.5)*dx)
            k += 1
        end
    end
    fill!(wall.v, zero(SVector{2,Float64}))
    wall.rho .= rho0_wall
    fill!(wall.stress, zero(SVector{4,Float64}))
    wall.p .= 0.0

    # --- Boundary particles (3-layer DynamicBoundarySystem) ------------------
    # Reduced from DambreakWall.jl's 156/80/80/156-per-layer sheets; domain
    # kept just large enough to contain the fluid+wall block with a small
    # margin, so floor/left activate immediately (matching the script's own
    # near-floor/near-left proximity) while right/top stay inactive for the
    # duration of this short run -- exactly as in the real script, where
    # those two only ever activate after the wave has travelled the length
    # of the tank.
    n_bnd_layers = 3
    x_right = 9.0
    y_top   = 10.0

    n_floor_x = 24   # -1.25 : 0.5 : 10.25
    n_floor   = n_floor_x * n_bnd_layers
    floor_inner = BasicParticleSystem("floor", n_floor, 2, fluid_mass, c_sound)
    let k = 1
        for layer in 0:n_bnd_layers-1, ix in 0:n_floor_x-1
            floor_inner.x[k] = SVector(-1.25 + ix * dx, -(layer + 0.5) * dx)
            k += 1
        end
    end
    floor_inner.rho .= rho0_water
    fill!(floor_inner.v, zero(SVector{2,Float64}))
    floor_dyn = DynamicBoundarySystem(floor_inner, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

    n_left_y = 20   # 0.25 : 0.5 : 9.75
    n_left   = n_left_y * n_bnd_layers
    left_inner = BasicParticleSystem("left", n_left, 2, fluid_mass, c_sound)
    let k = 1
        for layer in 0:n_bnd_layers-1, iy in 0:n_left_y-1
            left_inner.x[k] = SVector(-(layer + 0.5) * dx, (iy + 0.5) * dx)
            k += 1
        end
    end
    left_inner.rho .= rho0_water
    fill!(left_inner.v, zero(SVector{2,Float64}))
    left_dyn = DynamicBoundarySystem(left_inner, SVector(1.0, 0.0), SVector(0.0, 0.0), 3.0)

    n_right_y = 20   # 0.25 : 0.5 : 9.75
    n_right   = n_right_y * n_bnd_layers
    right_inner = BasicParticleSystem("right", n_right, 2, fluid_mass, c_sound)
    let k = 1
        for layer in 0:n_bnd_layers-1, iy in 0:n_right_y-1
            right_inner.x[k] = SVector(x_right + (layer + 0.5) * dx, (iy + 0.5) * dx)
            k += 1
        end
    end
    right_inner.rho .= rho0_water
    fill!(right_inner.v, zero(SVector{2,Float64}))
    right_dyn = DynamicBoundarySystem(right_inner, SVector(-1.0, 0.0), SVector(x_right, 0.0), 3.0)

    n_top_x = 24   # -1.25 : 0.5 : 10.25
    n_top   = n_top_x * n_bnd_layers
    top_inner = BasicParticleSystem("top", n_top, 2, fluid_mass, c_sound)
    let k = 1
        for layer in 0:n_bnd_layers-1, ix in 0:n_top_x-1
            top_inner.x[k] = SVector(-1.25 + ix * dx, y_top + (layer + 0.5) * dx)
            k += 1
        end
    end
    top_inner.rho .= rho0_water
    fill!(top_inner.v, zero(SVector{2,Float64}))
    top_dyn = DynamicBoundarySystem(top_inner, SVector(0.0, -1.0), SVector(0.0, y_top), 3.0)

    return (; kernel, h_sph, art_visc_alpha, art_visc_beta,
              fluid, wall,
              floor_inner, left_inner, right_inner, top_inner,
              floor_dyn, left_dyn, right_dyn, top_dyn)
end

# Builds the 8 SystemInteractions exactly matching DambreakWall.jl's wiring,
# and the LeapFrogTimeIntegrator wrapping them, for one independently-built
# system set.
function _dambreakwall_integrator(s; onesided::Bool)
    fluid_pfn       = FluidPfn(s.art_visc_alpha, s.art_visc_beta, s.h_sph)
    cauchy_pfn      = CauchyFluidPfn(s.art_visc_alpha, s.art_visc_beta, s.h_sph)
    fluid_solid_pfn = FluidSolidPfn(s.art_visc_alpha, s.art_visc_beta, s.h_sph)
    sr_pfn          = StrainRateVorticityPfn()

    int_fluid_self  = SystemInteraction(s.kernel, (nothing, fluid_pfn), s.fluid; onesided)
    int_fluid_floor = SystemInteraction(s.kernel, (nothing, fluid_pfn), s.fluid, s.floor_dyn; onesided)
    int_fluid_left  = SystemInteraction(s.kernel, (nothing, fluid_pfn), s.fluid, s.left_dyn; onesided)
    int_fluid_right = SystemInteraction(s.kernel, (nothing, fluid_pfn), s.fluid, s.right_dyn; onesided)
    int_fluid_top   = SystemInteraction(s.kernel, (nothing, fluid_pfn), s.fluid, s.top_dyn; onesided)
    int_wall_self   = SystemInteraction(s.kernel, (sr_pfn, cauchy_pfn), s.wall; onesided)
    int_wall_floor  = SystemInteraction(s.kernel, (sr_pfn, cauchy_pfn), s.wall, s.floor_dyn; onesided)
    int_fluid_wall  = SystemInteraction(s.kernel, (nothing, fluid_solid_pfn), s.fluid, s.wall; onesided)

    return LeapFrogTimeIntegrator(
        [s.fluid, s.wall, s.floor_inner, s.left_inner, s.right_inner, s.top_inner],
        [int_fluid_self, int_wall_self, int_fluid_wall, int_fluid_floor, int_fluid_left, int_fluid_right, int_fluid_top, int_wall_floor],
    )
end

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_elemscale(v::AbstractVector{<:Real})    = max(maximum(abs, v), 1.0)
_elemscale(v::AbstractVector{<:SVector}) = max(maximum(norm, v), 1.0)
_elemdiff(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})       = maximum(abs.(a .- b))
_elemdiff(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector}) = maximum(norm.(a .- b))

_has_nan(v::AbstractVector{<:Real})    = any(isnan, v)
_has_nan(v::AbstractVector{<:SVector}) = any(x -> any(isnan, x), v)
_has_inf(v::AbstractVector{<:Real})    = any(isinf, v)
_has_inf(v::AbstractVector{<:SVector}) = any(x -> any(isinf, x), v)

# ---------------------------------------------------------------------------
# 1. Short-run trajectory equivalence -- onesided=true vs onesided=false,
#    through a real LeapFrogTimeIntegrator, all 8 interactions.
# ---------------------------------------------------------------------------

@testset "short-run trajectory equivalence: onesided=true vs onesided=false (dambreak-wall-like)" begin
    s_old = _dambreakwall_like()
    s_new = _dambreakwall_like()

    integ_old = _dambreakwall_integrator(s_old; onesided=false)
    integ_new = _dambreakwall_integrator(s_new; onesided=true)

    nsteps = 80
    time_integrate!(integ_old, nsteps, nsteps + 1, nsteps + 1, 0.05, nothing; print_timer=false)
    time_integrate!(integ_new, nsteps, nsteps + 1, nsteps + 1, 0.05, nothing; print_timer=false)

    fluid_old, fluid_new = s_old.fluid, s_new.fluid
    wall_old,  wall_new  = s_old.wall,  s_new.wall

    @test !_has_nan(fluid_old.x); @test !_has_nan(fluid_new.x)
    @test !_has_nan(fluid_old.v); @test !_has_nan(fluid_new.v)
    @test !_has_nan(fluid_old.rho); @test !_has_nan(fluid_new.rho)
    @test !_has_nan(wall_old.x); @test !_has_nan(wall_new.x)
    @test !_has_nan(wall_old.v); @test !_has_nan(wall_new.v)
    @test !_has_nan(wall_old.stress); @test !_has_nan(wall_new.stress)

    # Sanity: this harness is only meaningful if the fluid<->wall coupling
    # (FluidSolidPfn, WritesBoth) actually produced non-trivial forces --
    # otherwise a trivially-degenerate 0==0 comparison would pass for the
    # wrong reason. See the geometry/cascade note on `_dambreakwall_like`.
    @test maximum(norm, wall_old.dvdt) > 1e-8
    @test maximum(x -> abs(x[4]), wall_old.stress) > 1e-8   # shear component nonzero

    # Position/velocity/density -- fluid.
    @test _elemdiff(fluid_old.x, fluid_new.x)     < 1e-8 * _elemscale(fluid_old.x)
    @test _elemdiff(fluid_old.v, fluid_new.v)     < 1e-6 * _elemscale(fluid_old.v)
    @test _elemdiff(fluid_old.rho, fluid_new.rho) < 1e-6 * _elemscale(fluid_old.rho)
    # Final-step raw pfn output -- fluid (highest-value check: exercises
    # FluidPfn self + all active DynamicBoundarySystem couplings +
    # FluidSolidPfn's fluid-side write together).
    @test _elemdiff(fluid_old.dvdt, fluid_new.dvdt)     < 1e-6 * _elemscale(fluid_old.dvdt)
    @test _elemdiff(fluid_old.drhodt, fluid_new.drhodt) < 1e-6 * _elemscale(fluid_old.drhodt)

    # Position/velocity -- wall.
    @test _elemdiff(wall_old.x, wall_new.x) < 1e-8 * _elemscale(wall_old.x)
    @test _elemdiff(wall_old.v, wall_new.v) < 1e-6 * _elemscale(wall_old.v)
    # Final-step raw pfn output -- wall (highest-value check: exercises
    # CauchyFluidPfn self + wall-floor + FluidSolidPfn's wall-side write
    # together, plus the stress it feeds off of).
    @test _elemdiff(wall_old.dvdt, wall_new.dvdt)     < 1e-6 * _elemscale(wall_old.dvdt)
    @test _elemdiff(wall_old.drhodt, wall_new.drhodt) < 1e-6 * _elemscale(wall_old.drhodt)
    @test _elemdiff(wall_old.stress, wall_new.stress) < 1e-6 * _elemscale(wall_old.stress)
    @test _elemdiff(wall_old.strain_rate, wall_new.strain_rate) < 1e-6 * _elemscale(wall_old.strain_rate)
end

# ---------------------------------------------------------------------------
# 2. Long-run physical invariants (onesided=true path only)
# ---------------------------------------------------------------------------

@testset "long-run physical invariants (onesided=true, dambreak-wall-like)" begin
    s = _dambreakwall_like()
    integ = _dambreakwall_integrator(s; onesided=true)

    time_integrate!(integ, 200, 201, 201, 0.05, nothing; print_timer=false)

    fluid, wall = s.fluid, s.wall

    @test !_has_nan(fluid.rho); @test !_has_inf(fluid.rho)
    @test !_has_nan(wall.rho);  @test !_has_inf(wall.rho)
    @test !_has_nan(fluid.x);   @test !_has_nan(fluid.v)
    @test !_has_nan(wall.x);    @test !_has_nan(wall.v)
    @test !_has_nan(wall.stress); @test !_has_inf(wall.stress)

    # No particle should have travelled absurdly far given the short run and
    # small initial block -- a broken boundary/mutual-coupling (particles
    # leaking through the floor/wall, or the wall flying apart) would blow
    # this up.
    @test all(x -> all(abs.(x) .< 50.0), fluid.x)
    @test all(x -> all(abs.(x) .< 50.0), wall.x)
end
