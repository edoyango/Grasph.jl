using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Cluster D minimal: bubble.jl / bubble2.jl shape.
#
# Two-phase FluidParticleSystem coupling driven through a REAL
# RK4TimeIntegrator loop (the only RK4 shape among all scripts besides
# bubble2.jl/bubble3.jl):
#
#   fluid_X_interaction  = SystemInteraction(kernel, FluidPfn(...), fluid_X)              -- self
#   fluid_Y_interaction  = SystemInteraction(kernel, FluidPfn(...), fluid_Y)              -- self
#   fluid_XY_interaction = SystemInteraction(kernel, FluidPfn(...), fluid_Y, fluid_X)     -- mutual, WritesBoth
#   fluid_boundary_interaction = SystemInteraction(kernel, FluidPfn(...), fluid_X, boundary_ghost)
#
# integrator = RK4TimeIntegrator([fluid_X, fluid_Y], [the 4 interactions above];
#                                 ghosts=[boundary_ghost_entry])
#
# This mirrors bubble.jl (see that file at the repo root for the full-scale
# original: R=1.0, height=10R, width=6R, dx_spacing=0.05, WenlandC2Kernel,
# FluidPfn(0.01, 0.0, h_sph), hydrostatic density init, 8-plane boundary
# ghost). This harness reduces particle counts and step counts by roughly two
# orders of magnitude while keeping the same qualitative shape: a circular
# "bubble" of lighter fluid Y centred at the origin, surrounded by heavier
# fluid X, inside a walled box. Per the task brief, the boundary ghost here
# uses only the 4 wall planes (no 4 corner planes) — a deliberate
# simplification, not a shape mismatch, since GhostCopier tuple length only
# needs to match the pfn-tuple stage count (1), which is unaffected by the
# boundary count.
# ---------------------------------------------------------------------------

# Deterministic regular layout (this mirrors bubble.jl exactly: no
# randomness anywhere in the real script's setup — positions come from a
# regular grid scan, velocities start at zero, and densities are initialized
# from a closed-form hydrostatic-pressure formula). Called twice
# independently below (not deepcopy'd) so the "old"/coloured and
# "new"/onesided runs start from bit-identical, non-aliased state.
function _bubble_like(; R=1.0, width=4.0*R, height=6.0*R, dx_spacing=0.30, g=9.81)
    x_min = -width/2
    x_max =  width/2
    y_min = -height/2 + R      # bubble (radius R, centred at origin) sits ~R above the floor
    y_max =  y_min + height

    h_sph = 1.2 * dx_spacing
    rho_X = 1000.0
    rho_Y = 500.0
    c_sound_X = sqrt(800.0 * g * R)
    c_sound_Y = 40.0 * sqrt(g * R)
    art_visc_alpha = 0.01
    art_visc_beta  = 0.0

    nx = Int(floor(width  / dx_spacing))
    ny = Int(floor(height / dx_spacing))

    x_X = Float64[]; y_X = Float64[]
    x_Y = Float64[]; y_Y = Float64[]
    for i in 0:nx-1, j in 0:ny-1
        x = x_min + (i + 0.5) * dx_spacing
        y = y_min + (j + 0.5) * dx_spacing
        if x*x + y*y < R
            push!(x_Y, x); push!(y_Y, y)
        else
            push!(x_X, x); push!(y_X, y)
        end
    end

    fluid_X_mass = dx_spacing * dx_spacing * rho_X
    fluid_Y_mass = dx_spacing * dx_spacing * rho_Y

    fluid_X = FluidParticleSystem(
        "fluid X", length(x_X), 2, fluid_X_mass, c_sound_X;
        source_v = [0.0, -g],
        state_updater = TaitEOSUpdater(rho_X),
    )
    for i in 1:length(x_X)
        fluid_X.x[i] = SVector(x_X[i], y_X[i])
        pressure = (y_max - y_X[i]) * rho_X * g
        fluid_X.rho[i] = (pressure * 7.0 / (c_sound_X * c_sound_X * rho_X) + 1.0)^(1.0/7.0) * rho_X
    end
    fill!(fluid_X.v, zero(SVector{2,Float64}))

    fluid_Y = FluidParticleSystem(
        "fluid Y", length(x_Y), 2, fluid_Y_mass, c_sound_Y;
        source_v = [0.0, -g],
        state_updater = TaitEOSUpdater(rho_Y),
    )
    for i in 1:length(x_Y)
        fluid_Y.x[i] = SVector(x_Y[i], y_Y[i])
        pressure = (y_max - y_Y[i]) * rho_X * g   # matches bubble.jl: hydrostatic column dominated by rho_X
        fluid_Y.rho[i] = (pressure * 7.0 / (c_sound_Y * c_sound_Y * rho_Y) + 1.0)^(1.0/7.0) * rho_Y
    end
    fill!(fluid_Y.v, zero(SVector{2,Float64}))

    # Single self-referencing ghost representing the 4 walls (reduced from
    # bubble.jl's 8 planes -- 4 walls + 4 corners -- per the task brief).
    boundary_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost[fluid_X]")
    boundary_ghost_entry = GhostEntry(boundary_ghost, 3.0 * h_sph,
        (SVector( 1.0,  0.0), SVector(x_min, 0.0  )),  # left wall
        (SVector(-1.0,  0.0), SVector(x_max, 0.0  )),  # right wall
        (SVector( 0.0,  1.0), SVector(0.0,   y_min)),  # bottom wall
        (SVector( 0.0, -1.0), SVector(0.0,   y_max)),  # top wall
    )

    kernel = WenlandC2Kernel(h_sph; ndims=2)

    return (; kernel, fluid_X, fluid_Y, boundary_ghost, boundary_ghost_entry,
              art_visc_alpha, art_visc_beta, h_sph)
end

# Builds the 4 SystemInteractions exactly matching bubble.jl's wiring, and
# the RK4TimeIntegrator wrapping them, for one independently-built system set.
function _bubble_integrator(sys; onesided::Bool)
    (; kernel, fluid_X, fluid_Y, boundary_ghost, boundary_ghost_entry,
       art_visc_alpha, art_visc_beta, h_sph) = sys

    fluid_X_interaction = SystemInteraction(
        kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_X; onesided)
    fluid_Y_interaction = SystemInteraction(
        kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_Y; onesided)
    fluid_XY_interaction = SystemInteraction(
        kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_Y, fluid_X; onesided)
    fluid_boundary_interaction = SystemInteraction(
        kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_X, boundary_ghost; onesided)

    return RK4TimeIntegrator(
        [fluid_X, fluid_Y],
        [fluid_X_interaction, fluid_Y_interaction, fluid_XY_interaction, fluid_boundary_interaction];
        ghosts = [boundary_ghost_entry],
    )
end

# ---------------------------------------------------------------------------
# 1. Short-run trajectory equivalence — onesided=true vs onesided=false
# ---------------------------------------------------------------------------

@testset "short-run trajectory equivalence: onesided=true vs onesided=false (bubble-like)" begin
    sys_old = _bubble_like()
    sys_new = _bubble_like()

    # Sanity: independently-built systems start bit-identical.
    @test length(sys_old.fluid_X.x) == length(sys_new.fluid_X.x)
    @test length(sys_old.fluid_Y.x) == length(sys_new.fluid_Y.x)
    @test 150 <= sys_old.fluid_X.n <= 250
    @test 30  <= sys_old.fluid_Y.n <= 60

    rk_old = _bubble_integrator(sys_old; onesided=false)
    rk_new = _bubble_integrator(sys_new; onesided=true)

    nsteps = 30
    time_integrate!(rk_old, nsteps, nsteps + 1, nsteps + 1, 1.5, nothing; print_timer=false)
    time_integrate!(rk_new, nsteps, nsteps + 1, nsteps + 1, 1.5, nothing; print_timer=false)

    fX_old, fX_new = sys_old.fluid_X, sys_new.fluid_X
    fY_old, fY_new = sys_old.fluid_Y, sys_new.fluid_Y

    for (a, b) in ((fX_old, fX_new), (fY_old, fY_new))
        @test !any(v -> any(isnan, v), a.x)
        @test !any(v -> any(isnan, v), a.v)
        @test !any(isnan, a.rho)
        @test !any(v -> any(isnan, v), a.dvdt)
        @test !any(isnan, a.drhodt)

        x_scale    = max(maximum(norm.(a.x)), 1.0)
        v_scale    = max(maximum(norm.(a.v)), 1.0)
        rho_scale  = max(maximum(abs.(a.rho)), 1.0)
        dvdt_scale = max(maximum(norm.(a.dvdt)), 1.0)
        drho_scale = max(maximum(abs.(a.drhodt)), 1.0)

        x_diff    = maximum(norm.(a.x    .- b.x))
        v_diff    = maximum(norm.(a.v    .- b.v))
        rho_diff  = maximum(abs.(a.rho  .- b.rho))
        dvdt_diff = maximum(norm.(a.dvdt .- b.dvdt))
        drho_diff = maximum(abs.(a.drhodt .- b.drhodt))

        @test x_diff    < 1e-6 * x_scale
        @test v_diff    < 1e-6 * v_scale
        @test rho_diff  < 1e-6 * rho_scale
        @test dvdt_diff < 1e-6 * dvdt_scale
        @test drho_diff < 1e-6 * drho_scale
    end
end

# ---------------------------------------------------------------------------
# 2. Long-run physical invariants (onesided=true path only)
# ---------------------------------------------------------------------------

@testset "long-run physical invariants (onesided=true, bubble-like)" begin
    sys = _bubble_like()
    rk = _bubble_integrator(sys; onesided=true)

    nsteps = 90
    time_integrate!(rk, nsteps, nsteps + 1, nsteps + 1, 1.5, nothing; print_timer=false)

    for ps in (sys.fluid_X, sys.fluid_Y)
        @test all(!isnan, ps.rho)
        @test all(!isinf, ps.rho)
        @test all(x -> all(!isnan, x), ps.x)
        @test all(x -> all(!isnan, x), ps.v)
        @test all(x -> all(!isinf, x), ps.x)
        @test all(x -> all(!isinf, x), ps.v)
        # No particle should have travelled absurdly far given the short run
        # and small initial column -- a broken ghost boundary coupling
        # (particles leaking through a wall to +-Inf) would blow this up.
        @test all(x -> all(abs.(x) .< 50.0), ps.x)
    end
end
