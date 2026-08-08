using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm
using Random

# ---------------------------------------------------------------------------
# Cluster D extension: bubble3.jl-shaped in-context integration test for the
# opt-in one-sided sweep (`onesided=true`).
#
# Companion to test_onesided_integration_bubble.jl (the base two-phase
# fluid_X/fluid_Y/ghost shape shared with bubble.jl/bubble2.jl/bubble3.jl) --
# this file focuses on what bubble3.jl adds on top of that shape (see
# bubble3.jl directly for the real values this mirrors):
#
#   - velocity_adjust_pairwise_fn=XSPHPfn(0.5) on 3 of the 4 interactions
#     (fluid_X self, fluid_Y self, and fluid_boundary -- the ghost-coupled
#     one).
#   - FluidPfn(...; epsilon=0.1) artificial surface tension on the
#     fluid_X<->fluid_Y coupling (fluid_XY_interaction; that interaction
#     never sets velocity_adjust_pairwise_fn in any of the 3 bubble scripts).
#
# This is the first full-integrator-loop (RK4TimeIntegrator, real
# time_integrate! loop -- real ghost generation/kinematics-update/copier-
# update each step, real per-step XSPH correction pass) exercise of the
# XSPHPfn ghost-coupling aliasing-bug fix: `boundary_ghost =
# GhostParticleSystem(fluid_X, GhostCopier(:p))` is SELF-REFERENCING
# (`boundary_ghost.source === fluid_X`). See src/PairwiseFunctors.jl's
# comment above XSPHPfn's ghost/virtual-coupled methods for the full
# mechanism of the bug this fixed. Previous testing (test_onesided_sweep.jl
# section 7b) exercised the fix via a single standalone sweep built by hand;
# this harness exercises it inside the real multi-stage RK4 loop, with the
# ghost regenerated/kinematics-updated/copier-updated fresh every step
# exactly as TimeIntegration.jl does it, and with the real
# `_xsph_correction!` pass (subtract old v_adjustment from v, zero it, sweep
# the vadjust_pfn fresh, add the new v_adjustment back into v) running on
# top -- driven automatically by `time_integrate!`, nothing called by hand.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Builder: bubble3.jl's setup, reduced scale.
#
# Domain geometry (width/height/x_min/y_min) is kept at bubble3.jl's actual
# proportions (R_domain=1 => 6x10 box) but the bubble radius used for the
# fluid_X/fluid_Y split is deliberately DECOUPLED from that domain scale
# (R_bubble=2.2*R_domain, vs bubble3.jl's R_bubble==R_domain==1.0): at
# bubble3.jl's true ~5% bubble/box area ratio, hitting both the guided
# fluid_X~150-250 and fluid_Y~30-60 particle counts simultaneously isn't
# possible from dx_spacing alone (the ratio is fixed by geometry, so shrinking
# dx to hit one count overshoots or undershoots the other). Enlarging just the
# bubble preserves the qualitative shape this test needs -- a bubble embedded
# in a bigger surrounding fluid, with a boundary ghost wrapping the whole box
# that is a meaningful (not degenerate/tiny) fraction of fluid_X's extent --
# while giving both fluid_Y and the ghost-coupled fluid_X a non-trivial
# particle count to actually exercise the interactions under test.
#
# Called independently (not deepcopy'd) by each of the two comparison runs so
# they start from bit-identical state without any aliasing risk.
# ---------------------------------------------------------------------------

function _bubble3_like()
    R_domain   = 1.0
    R_bubble   = 2.2 * R_domain    # decoupled from domain scale -- see note above
    height     = 10.0 * R_domain
    width      = 6.0  * R_domain
    y_min      = -2.0 * R_domain
    y_max      = y_min + height
    x_min      = -3.0 * R_domain
    x_max      = x_min + width
    dx_spacing = 0.46              # reduced-scale analogue of bubble3.jl's 0.04
    h_sph      = 1.2 * dx_spacing
    rho_X      = 1000.0
    rho_Y      = 1.0
    g          = 9.81
    c_sound_X  = sqrt(800.0*g*R_domain)
    c_sound_Y  = 400.0*sqrt(g*R_domain)
    art_visc_alpha = 0.01
    art_visc_beta  = 0.0

    nx = Int(floor(width/dx_spacing))
    ny = Int(floor(height/dx_spacing))

    x_X = Float64[]; y_X = Float64[]
    x_Y = Float64[]; y_Y = Float64[]
    for i in 0:nx-1, j in 0:ny-1
        x = x_min + (i+0.5)*dx_spacing
        y = y_min + (j+0.5)*dx_spacing
        # particles within radius belong to bubble (Y), else the heavier fluid (X)
        if x*x + y*y < R_bubble
            push!(x_Y, x)
            push!(y_Y, y)
        else
            push!(x_X, x)
            push!(y_X, y)
        end
    end

    fluid_X_mass = dx_spacing*dx_spacing*rho_X
    fluid_Y_mass = dx_spacing*dx_spacing*rho_Y

    fluid_X = FluidParticleSystem(
        "fluid X", length(x_X), 2, fluid_X_mass, c_sound_X;
        source_v = [0.0, -g],
        state_updater = TaitEOSUpdater(rho_X),
    )
    for i in 1:length(x_X)
        fluid_X.x[i] = SVector(x_X[i], y_X[i])
        pressure = (y_min + height - y_X[i]) * rho_X * g
        fluid_X.rho[i] = (pressure*7.0/(c_sound_X*c_sound_X*rho_X) + 1.0)^(1.0/7.0)*rho_X
    end
    fill!(fluid_X.v, zero(SVector{2,Float64}))

    fluid_Y = FluidParticleSystem(
        "fluid Y", length(x_Y), 2, fluid_Y_mass, c_sound_Y;
        source_v = [0.0, -g],
        state_updater = TaitEOSUpdater(rho_Y, 1.4),
    )
    for i in 1:length(x_Y)
        fluid_Y.x[i] = SVector(x_Y[i], y_Y[i])
        pressure = (y_min + height - y_Y[i]) * rho_X * g
        fluid_Y.rho[i] = (pressure*1.4/(c_sound_Y*c_sound_Y*rho_Y) + 1.0)^(1.0/1.4)*rho_Y
    end
    fill!(fluid_Y.v, zero(SVector{2,Float64}))

    kernel = WenlandC2Kernel(h_sph; ndims=2)

    return (; kernel, fluid_X, fluid_Y, h_sph,
              x_min, x_max, y_min, y_max,
              art_visc_alpha, art_visc_beta)
end

# ---------------------------------------------------------------------------
# Build the 4 interactions + self-referencing ghost + RK4TimeIntegrator
# exactly matching bubble3.jl's wiring, from a bundle returned by
# `_bubble3_like`, with `onesided` threaded onto every SystemInteraction.
# ---------------------------------------------------------------------------

function _bubble3_integrator(b; onesided::Bool)
    kernel  = b.kernel
    fluid_X = b.fluid_X
    fluid_Y = b.fluid_Y
    h_sph   = b.h_sph

    # Single ghost system representing all 4 walls and 4 corner boundaries,
    # self-referencing its source (boundary_ghost.source === fluid_X) --
    # exactly bubble3.jl's boundary_ghost.
    boundary_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost[fluid_X]")

    boundary_ghost_entry = GhostEntry(boundary_ghost, 3.0 * h_sph,
        (SVector( 1.0,  0.0),            SVector(b.x_min, 0.0)),      # left wall
        (SVector(-1.0,  0.0),            SVector(b.x_max, 0.0)),      # right wall
        (SVector( 0.0,  1.0),            SVector(0.0,   b.y_min)),    # bottom wall
        (SVector( 0.0, -1.0),            SVector(0.0,   b.y_max)),    # top wall
        (SVector( 1.0,  1.0)/sqrt(2.0),  SVector(b.x_min, b.y_min)),  # bottom-left corner
        (SVector(-1.0,  1.0)/sqrt(2.0),  SVector(b.x_max, b.y_min)),  # bottom-right corner
        (SVector( 1.0, -1.0)/sqrt(2.0),  SVector(b.x_min, b.y_max)),  # top-left corner
        (SVector(-1.0, -1.0)/sqrt(2.0),  SVector(b.x_max, b.y_max)),  # top-right corner
    )

    fluid_X_interaction = SystemInteraction(
        kernel,
        FluidPfn(b.art_visc_alpha, b.art_visc_beta, h_sph; sigma=1),
        fluid_X;
        velocity_adjust_pairwise_fn=XSPHPfn(0.5),
        onesided=onesided,
    )

    fluid_Y_interaction = SystemInteraction(
        kernel,
        FluidPfn(b.art_visc_alpha, b.art_visc_beta, h_sph; sigma=1),
        fluid_Y;
        velocity_adjust_pairwise_fn=XSPHPfn(0.5),
        onesided=onesided,
    )

    # No velocity_adjust_pairwise_fn here -- confirmed via grep that
    # fluid_XY_interaction never sets one in any of the 3 bubble scripts.
    fluid_XY_interaction = SystemInteraction(
        kernel,
        FluidPfn(b.art_visc_alpha, b.art_visc_beta, h_sph; sigma=1, epsilon=0.1),
        fluid_Y,   # useful to make Y the first system as the iteration space is smaller
        fluid_X;
        onesided=onesided,
    )

    fluid_boundary_interaction = SystemInteraction(
        kernel,
        FluidPfn(b.art_visc_alpha, b.art_visc_beta, h_sph; sigma=1),
        fluid_X,
        boundary_ghost;
        velocity_adjust_pairwise_fn=XSPHPfn(0.5),
        onesided=onesided,
    )

    integrator = RK4TimeIntegrator(
        [fluid_X, fluid_Y],
        [fluid_X_interaction, fluid_Y_interaction, fluid_XY_interaction, fluid_boundary_interaction];
        ghosts = [boundary_ghost_entry],
    )

    return integrator
end

const _BUBBLE3_CFL = 1.5   # matches bubble3.jl's Stage(... , 1.5, "run")

# ---------------------------------------------------------------------------
# Short-run trajectory equivalence: onesided=true vs onesided=false, driven
# through a real RK4TimeIntegrator loop (ghosts generated/kinematics-updated/
# copier-updated automatically every step, XSPH correction applied
# automatically every step).
# ---------------------------------------------------------------------------

@testset "bubble3-like short-run trajectory equivalence: onesided=true vs onesided=false" begin
    b_old = _bubble3_like()
    b_new = _bubble3_like()

    # Sanity on the reduced-scale geometry itself before trusting the diffs below.
    @test b_old.fluid_X.n == b_new.fluid_X.n
    @test b_old.fluid_Y.n == b_new.fluid_Y.n
    @test 150 <= b_old.fluid_X.n <= 260
    @test 25  <= b_old.fluid_Y.n <= 65

    integ_old = _bubble3_integrator(b_old; onesided=false)
    integ_new = _bubble3_integrator(b_new; onesided=true)

    nsteps = 30
    time_integrate!(integ_old, nsteps, nsteps + 1, nsteps + 1, _BUBBLE3_CFL, nothing; print_timer=false)
    time_integrate!(integ_new, nsteps, nsteps + 1, nsteps + 1, _BUBBLE3_CFL, nothing; print_timer=false)

    for (fold, fnew, label) in ((b_old.fluid_X, b_new.fluid_X, "fluid_X"),
                                 (b_old.fluid_Y, b_new.fluid_Y, "fluid_Y"))
        @test !any(x -> any(isnan, x), fold.x)
        @test !any(x -> any(isnan, x), fnew.x)
        @test !any(x -> any(isnan, x), fold.v)
        @test !any(x -> any(isnan, x), fnew.v)
        @test !any(isnan, fold.rho)
        @test !any(isnan, fnew.rho)

        x_scale      = max(maximum(norm.(fold.x)), 1.0)
        v_scale      = max(maximum(norm.(fold.v)), 1.0)
        rho_scale    = max(maximum(abs.(fold.rho)), 1.0)
        dvdt_scale   = max(maximum(norm.(fold.dvdt)), 1.0)
        drhodt_scale = max(maximum(abs.(fold.drhodt)), 1.0)
        vadj_scale   = max(maximum(norm.(fold.v_adjustment)), 1.0)

        x_diff    = maximum(norm.(fold.x .- fnew.x))
        v_diff    = maximum(norm.(fold.v .- fnew.v))
        rho_diff  = maximum(abs.(fold.rho .- fnew.rho))
        dvdt_diff   = maximum(norm.(fold.dvdt .- fnew.dvdt))
        drhodt_diff = maximum(abs.(fold.drhodt .- fnew.drhodt))
        vadj_diff   = maximum(norm.(fold.v_adjustment .- fnew.v_adjustment))

        # Positions tightest (integrated once, least chaotic); dvdt/drhodt/
        # v_adjustment (higher-derivative, and v_adjustment in particular is
        # exactly the field the XSPHPfn ghost-coupling aliasing fix writes)
        # looser, matching the established template's tolerance ordering.
        @test x_diff      < 1e-8 * x_scale
        @test v_diff       < 1e-6 * v_scale
        @test rho_diff     < 1e-6 * rho_scale
        @test dvdt_diff    < 1e-6 * dvdt_scale
        @test drhodt_diff  < 1e-6 * drhodt_scale
        @test vadj_diff    < 1e-6 * vadj_scale
    end
end

# ---------------------------------------------------------------------------
# Long-run physical invariants (onesided=true path only).
# ---------------------------------------------------------------------------

@testset "bubble3-like long-run physical invariants (onesided=true)" begin
    b = _bubble3_like()
    integrator = _bubble3_integrator(b; onesided=true)

    nsteps = 80   # ~2.7x the short-run step count
    time_integrate!(integrator, nsteps, nsteps + 1, nsteps + 1, _BUBBLE3_CFL, nothing; print_timer=false)

    for f in (b.fluid_X, b.fluid_Y)
        @test all(!isnan, f.rho)
        @test all(!isinf, f.rho)
        @test all(x -> all(!isnan, x), f.x)
        @test all(x -> all(!isnan, x), f.v)
        @test all(x -> all(!isnan, x), f.v_adjustment)
        # No particle should have travelled absurdly far given the short run
        # and small initial block -- a broken ghost/XSPH coupling (e.g. the
        # aliasing bug this harness targets, or a particle leaking through a
        # boundary) would blow this up well past domain scale (~10).
        @test all(x -> all(abs.(x) .< 50.0), f.x)
    end
end
