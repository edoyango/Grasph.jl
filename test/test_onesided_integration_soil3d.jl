using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Reduced-scale, in-context validation of `onesided=true` against the default
# coloured sweep for GranularColumnCollapse3D.jl's interaction shape:
#
#   - fluid  = StressParticleSystem(ND=3, NS=6 Voigt xx,yy,zz,xy,xz,yz),
#              Mohr-Coulomb (ViscoPlasticMCStressUpdater), gravity in -z.
#   - pfns   = (StrainRatePfn(), CauchyFluidPfn(alpha,beta,h)) — a 2-tuple
#              reused, unmodified, across all three interactions below (as
#              in the real script).
#   - bottom boundary = BasicParticleSystem wrapped as DynamicBoundarySystem,
#              +z outward normal (floor).
#   - side walls = a single self-referencing GhostParticleSystem
#              (GhostParticleSystem(fluid, nothing, GhostCopier(:stress)))
#              covering 2 wall planes (left x=0, front y=0 — the real script
#              also has a back wall + 2 corners; reduced here per the harness
#              scale guidance, while still exercising the ghost×soil coupling
#              in 3D across more than one plane).
#
# This is the only genuinely-3D interaction shape among the 11 scripts, so it
# is kept as a small cube rather than degenerating to 2D. Mirrors the
# `_dambreak_like` / "short-run trajectory equivalence" / "long-run physical
# invariants" shape in test_onesided_sweep.jl (sections 5-6).
# ---------------------------------------------------------------------------

# Deterministic regular-grid builder — no RNG needed, exactly mirroring
# GranularColumnCollapse3D.jl's own deterministic particle placement. Called
# independently for each of the "old" (coloured) and "new" (onesided) runs so
# they start from bit-identical, non-aliased state.
function _soil3d_like(; nfx=10, nfy=3, nfz=5, nbx=25)
    dx_spacing          = 0.002
    h_sph               = 1.2 * dx_spacing
    rho0                = 1850.0
    c_sound             = 20.0
    art_visc_alpha      = 0.1
    art_visc_beta       = 0.1
    soil_friction_angle = 19.8 * π / 180.0

    n_fluid    = nfx * nfy * nfz
    fluid_mass = rho0 * dx_spacing^3

    fluid = StressParticleSystem(
        "soil", n_fluid, 3, 6, fluid_mass, c_sound;
        source_v = [0.0, 0.0, -9.81],
        state_updater = (
            ZeroFieldUpdater(:strain_rate),
            ViscoPlasticMCStressUpdater(LinearEOSUpdater(rho0), soil_friction_angle, 0.0),
        ),
    )
    let k = 1
        for i in 0:nfx-1, j in 0:nfy-1, m in 0:nfz-1
            fluid.x[k] = SVector((i + 0.5) * dx_spacing,
                                  (j + 0.5) * dx_spacing,
                                  (m + 0.5) * dx_spacing)
            k += 1
        end
    end
    fill!(fluid.v, zero(SVector{3,Float64}))
    fluid.rho .= rho0
    update_state!(fluid)

    # Bottom boundary: flat sheet, 3 layers below z = 0, wider in x/y than
    # the soil block (mirrors the real script's proportions).
    nby = nfy
    n_bottom = (nbx + 3) * (nby + 6) * 3
    bottom_boundary = BasicParticleSystem(
        "bottom_boundary", n_bottom, 3, rho0 * dx_spacing^3, c_sound,
    )
    let k = 1
        for i in 1:nbx+3, j in 1:nby+6, m in 1:3
            bottom_boundary.x[k] = SVector(
                (i - 3.5)  * dx_spacing,
                (j - 2.5)  * dx_spacing,
                -(m - 0.5) * dx_spacing,
            )
            k += 1
        end
    end
    bottom_boundary.rho .= rho0
    fill!(bottom_boundary.v, zero(SVector{3,Float64}))

    kernel = CubicSplineKernel(h_sph; ndims=3)
    dynamic_bottom = DynamicBoundarySystem(
        bottom_boundary, SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 0.0), 3.0,
    )

    sr_pfn         = StrainRatePfn()
    kinematics_pfn = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)
    pfns           = (sr_pfn, kinematics_pfn)

    return kernel, fluid, bottom_boundary, dynamic_bottom, pfns, h_sph
end

# Self-referencing ghost covering 2 side-wall planes (left x=0, front y=0),
# correct stage-2 timing (nothing at stage 1, GhostCopier(:stress) at stage 2
# — StrainRatePfn doesn't need a stress copy, CauchyFluidPfn does), exactly
# as in the real script.
function _soil3d_ghost(fluid, h_sph)
    boundary_ghost = GhostParticleSystem(fluid, nothing, GhostCopier(:stress);
                                          name = "ghost[$(fluid.name)]")
    boundary_ghost_entry = GhostEntry(boundary_ghost, 3.0 * h_sph,
        (SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)),   # left wall (x=0)
        (SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0)),   # front wall (y=0)
    )
    return boundary_ghost, boundary_ghost_entry
end

# Builds one full integrator (systems + all 3 interactions + ghost), matching
# GranularColumnCollapse3D.jl's wiring exactly, with `onesided` applied
# uniformly to every interaction.
function _build_soil3d_integrator(onesided::Bool; kwargs...)
    kernel, fluid, bottom_boundary, dynamic_bottom, pfns, h_sph = _soil3d_like(; kwargs...)
    boundary_ghost, boundary_ghost_entry = _soil3d_ghost(fluid, h_sph)

    fluid_interaction          = SystemInteraction(kernel, pfns, fluid; onesided=onesided)
    fluid_bottom_interaction   = SystemInteraction(kernel, pfns, fluid, dynamic_bottom; onesided=onesided)
    fluid_boundary_interaction = SystemInteraction(kernel, pfns, fluid, boundary_ghost; onesided=onesided)

    integrator = LeapFrogTimeIntegrator(
        [fluid, bottom_boundary],
        [fluid_interaction, fluid_bottom_interaction, fluid_boundary_interaction];
        ghosts = (boundary_ghost_entry,),
    )
    return integrator, fluid, bottom_boundary
end

# ---------------------------------------------------------------------------
# Short-run trajectory equivalence: onesided=true vs onesided=false
# ---------------------------------------------------------------------------

@testset "short-run trajectory equivalence: onesided=true vs onesided=false (3D soil column)" begin
    integrator_old, fluid_old, bottom_old = _build_soil3d_integrator(false)
    integrator_new, fluid_new, bottom_new = _build_soil3d_integrator(true)

    @test fluid_old.n == fluid_new.n
    @test all(fluid_old.x .== fluid_new.x)   # bit-identical starting state

    nsteps = 40
    time_integrate!(integrator_old, nsteps, nsteps + 1, nsteps + 1, 0.1, nothing; print_timer=false)
    time_integrate!(integrator_new, nsteps, nsteps + 1, nsteps + 1, 0.1, nothing; print_timer=false)

    x_scale     = max(maximum(norm.(fluid_old.x)), 1.0)
    v_scale     = max(maximum(norm.(fluid_old.v)), 1.0)
    rho_scale   = max(maximum(abs.(fluid_old.rho)), 1.0)
    stress_scale = max(maximum(norm.(fluid_old.stress)), 1.0)

    x_diff      = maximum(norm.(fluid_old.x .- fluid_new.x))
    v_diff      = maximum(norm.(fluid_old.v .- fluid_new.v))
    rho_diff    = maximum(abs.(fluid_old.rho .- fluid_new.rho))
    stress_diff = maximum(norm.(fluid_old.stress .- fluid_new.stress))

    @test !any(isnan, reduce(vcat, [collect(v) for v in fluid_old.x]))
    @test !any(isnan, reduce(vcat, [collect(v) for v in fluid_new.x]))
    @test !any(isnan, fluid_old.rho) && !any(isnan, fluid_new.rho)

    @test x_diff      < 1e-8 * x_scale
    @test v_diff      < 1e-6 * v_scale
    @test rho_diff    < 1e-6 * rho_scale
    @test stress_diff < 1e-5 * stress_scale
end

# ---------------------------------------------------------------------------
# Long-run physical invariants (onesided=true path only)
# ---------------------------------------------------------------------------

@testset "long-run physical invariants (onesided=true, 3D soil column)" begin
    integrator, fluid, bottom_boundary = _build_soil3d_integrator(true)

    time_integrate!(integrator, 120, 121, 121, 0.1, nothing; print_timer=false)

    @test all(!isnan, fluid.rho)
    @test all(!isinf, fluid.rho)
    @test all(x -> all(!isnan, x), fluid.x)
    @test all(x -> all(!isnan, x), fluid.v)
    @test all(x -> all(!isnan, x), fluid.stress)
    # No particle should have travelled absurdly far given the short run and
    # small initial block — a broken ghost/boundary coupling (particles
    # leaking through a wall/floor to -Inf) would blow this up.
    @test all(x -> all(abs.(x) .< 50.0), fluid.x)
end
