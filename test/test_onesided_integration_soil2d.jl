using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Integration-shaped equivalence test for the opt-in `onesided=true` sweep
# (see test_onesided_sweep.jl for the pairwise-level protocol validation).
#
# This file does NOT test any new pfn_contribution code — every pfn used
# below has already been converted to support onesided=true. Instead it
# proves the one-sided path is correct *in context*: it builds the actual
# 2D elastoplastic/granular soil interaction shape used by
# GranularColumnCollapse.jl and EP_ColumnCollapse.jl, drives it through a
# real multi-stage LeapFrogTimeIntegrator loop once under the default
# coloured half-shell sweep and once under onesided=true, and confirms they
# agree — not just in isolated pairwise comparisons.
#
# Both scripts share the same shape: 2D soil self-interaction + soil<->floor
# (DynamicBoundarySystem) + soil<->left-wall mirror (a self-referencing
# GhostParticleSystem), driven by a 2-stage pfn tuple. They differ only in
# stress model / pfn / ghost-copier stage wiring:
#
#   Group A — GranularColumnCollapse.jl shape: StressParticleSystem (NS=3),
#     Mohr-Coulomb viscoplastic stress update, pfns = (StrainRatePfn(),
#     CauchyFluidPfn(...; delta=0.1)). Left ghost is built with
#     GhostParticleSystem(fluid, nothing, GhostCopier(:stress)) — two
#     updater slots for a 2-stage interaction, so the copier fires at
#     stage 2, immediately before CauchyFluidPfn (stage 2) needs :stress.
#     This is the "correctly timed" wiring.
#
#   Group B — EP_ColumnCollapse.jl shape: ElastoPlasticParticleSystem
#     (NS=4), Drucker-Prager elastoplastic stress update, pfns =
#     (StrainRateVorticityPfn(), CauchyFluidPfn(...)). Left ghost is built
#     with GhostParticleSystem(fluid, GhostCopier(:stress)) — only ONE
#     updater slot for a 2-stage interaction, so the copier fires at
#     stage 1 instead of stage 2: a pre-existing, probably-unintended
#     off-by-one quirk in the real script (see `GhostParticleSystem`'s
#     `updaters` stage dispatch in src/GhostParticles.jl:
#     `_run_ghost_stage!` invokes the k-th updater at stage k, so a
#     1-tuple only ever fires at stage 1). This harness REPLICATES that
#     quirk faithfully rather than "fixing" it — the point is to prove
#     onesided=true reproduces the coloured sweep's actual behaviour,
#     bug-for-bug, not to validate (or correct) the script's physics.
#
# Scale: reduced from the scripts' 100x50 (5000-particle) soil grids to a
# 15x10 (150-particle) grid, with the boundary/ghost geometry shrunk
# proportionally, and step counts cut from tens of thousands down to
# double/triple digits, so the whole file runs in seconds.
# ---------------------------------------------------------------------------

const _S2D_CFL          = 0.1
const _S2D_SHORT_NSTEPS = 80
const _S2D_LONG_NSTEPS  = 200

_s2d_allfinite(v::AbstractVector{<:Real})    = all(isfinite, v)
_s2d_allfinite(v::AbstractVector{<:SVector}) = all(x -> all(isfinite, x), v)

_s2d_scale(v::AbstractVector{<:Real})    = max(maximum(abs, v), 1.0)
_s2d_scale(v::AbstractVector{<:SVector}) = max(maximum(norm, v), 1.0)

_s2d_diff(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})       = maximum(abs.(a .- b))
_s2d_diff(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector}) = maximum(norm.(a .- b))

# ---------------------------------------------------------------------------
# Group A — GranularColumnCollapse.jl shape
# ---------------------------------------------------------------------------

# Deterministic regular-grid builder mirroring GranularColumnCollapse.jl's
# particle setup, at reduced scale. Called independently (not deepcopy'd)
# for the "old"/coloured and "new"/onesided runs so they start from
# bit-identical state without any aliasing risk.
function _granular_column_like(; nfx=15, nfy=10, nbx=40)
    dx_spacing = 0.002
    h_sph      = 1.2 * dx_spacing
    rho0       = 1850.0
    E          = 0.84e6
    nu         = 0.3
    c_sound    = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2 * nu)))
    art_visc_alpha      = 0.1
    art_visc_beta       = 0.1
    soil_friction_angle = 19.8 * π / 180.0

    n_fluid    = nfx * nfy
    fluid_mass = rho0 * dx_spacing * dx_spacing

    fluid = StressParticleSystem(
        "fluid", n_fluid, 2, 3, fluid_mass, c_sound;
        source_v = [0.0, -9.81],
        state_updater = (
            ZeroFieldUpdater(:strain_rate),
            ViscoPlasticMCStressUpdater(LinearEOSUpdater(rho0), soil_friction_angle, 0.0),
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
    update_state!(fluid)

    bottom_boundary = BasicParticleSystem(
        "bottom_boundary", 3 * (nbx + 3), 2, fluid_mass, c_sound,
    )
    for i in 1:nbx+3, j in 1:3
        bottom_boundary.x[(i-1)*3+j] = SVector((i - 3.5) * dx_spacing, -(j - 0.5) * dx_spacing)
    end
    bottom_boundary.rho .= rho0
    fill!(bottom_boundary.v, zero(SVector{2,Float64}))

    return (h_sph=h_sph, art_visc_alpha=art_visc_alpha, art_visc_beta=art_visc_beta,
            fluid=fluid, bottom_boundary=bottom_boundary)
end

# Builds the real interaction shape (self, self<->floor, self<->left-wall
# ghost) and wraps it in a LeapFrogTimeIntegrator, exactly mirroring
# GranularColumnCollapse.jl's wiring (2-arg GhostParticleSystem updater list
# -> copier fires at stage 2).
function _granular_column_integrator(p; onesided::Bool)
    kernel = CubicSplineKernel(p.h_sph; ndims=2)
    fluid, bottom_boundary = p.fluid, p.bottom_boundary

    dynamic_bottom = DynamicBoundarySystem(bottom_boundary, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

    left_ghost       = GhostParticleSystem(fluid, nothing, GhostCopier(:stress))
    left_ghost_entry = GhostEntry(left_ghost, 3.0 * p.h_sph, (SVector(1.0, 0.0), SVector(0.0, 0.0)))

    sr_pfn         = StrainRatePfn()
    kinematics_pfn = CauchyFluidPfn(p.art_visc_alpha, p.art_visc_beta, p.h_sph; delta=0.1)

    fluid_interaction = SystemInteraction(kernel, (sr_pfn, kinematics_pfn), fluid; onesided=onesided)
    fluid_bottom_boundary_interaction = SystemInteraction(
        kernel, (sr_pfn, kinematics_pfn), fluid, dynamic_bottom; onesided=onesided)
    fluid_left_boundary_interaction = SystemInteraction(
        kernel, (sr_pfn, kinematics_pfn), fluid, left_ghost; onesided=onesided)

    return LeapFrogTimeIntegrator(
        [fluid, bottom_boundary],
        [fluid_interaction, fluid_bottom_boundary_interaction, fluid_left_boundary_interaction];
        ghosts = (left_ghost_entry,),
    )
end

@testset "Group A (GranularColumnCollapse.jl shape)" begin
    @testset "short-run trajectory equivalence: onesided=true vs onesided=false" begin
        p_old = _granular_column_like()
        p_new = _granular_column_like()

        integrator_old = _granular_column_integrator(p_old; onesided=false)
        integrator_new = _granular_column_integrator(p_new; onesided=true)

        time_integrate!(integrator_old, _S2D_SHORT_NSTEPS, _S2D_SHORT_NSTEPS + 1, _S2D_SHORT_NSTEPS + 1,
                         _S2D_CFL, nothing; print_timer=false)
        time_integrate!(integrator_new, _S2D_SHORT_NSTEPS, _S2D_SHORT_NSTEPS + 1, _S2D_SHORT_NSTEPS + 1,
                         _S2D_CFL, nothing; print_timer=false)

        fluid_old, fluid_new = p_old.fluid, p_new.fluid

        for f in (:x, :v, :rho, :p, :stress, :strain_rate)
            va, vb = getproperty(fluid_old, f), getproperty(fluid_new, f)
            @test _s2d_allfinite(va)
            @test _s2d_allfinite(vb)
        end

        @test _s2d_diff(fluid_old.x, fluid_new.x)                     < 1e-8 * _s2d_scale(fluid_old.x)
        @test _s2d_diff(fluid_old.v, fluid_new.v)                     < 1e-6 * _s2d_scale(fluid_old.v)
        @test _s2d_diff(fluid_old.rho, fluid_new.rho)                 < 1e-6 * _s2d_scale(fluid_old.rho)
        @test _s2d_diff(fluid_old.p, fluid_new.p)                     < 1e-6 * _s2d_scale(fluid_old.p)
        @test _s2d_diff(fluid_old.stress, fluid_new.stress)           < 1e-5 * _s2d_scale(fluid_old.stress)
        @test _s2d_diff(fluid_old.strain_rate, fluid_new.strain_rate) < 1e-5 * _s2d_scale(fluid_old.strain_rate)
    end

    @testset "long-run physical invariants (onesided=true)" begin
        p = _granular_column_like()
        integrator = _granular_column_integrator(p; onesided=true)

        time_integrate!(integrator, _S2D_LONG_NSTEPS, _S2D_LONG_NSTEPS + 1, _S2D_LONG_NSTEPS + 1,
                         _S2D_CFL, nothing; print_timer=false)

        fluid = p.fluid
        @test _s2d_allfinite(fluid.x)
        @test _s2d_allfinite(fluid.v)
        @test _s2d_allfinite(fluid.rho)
        @test _s2d_allfinite(fluid.p)
        @test _s2d_allfinite(fluid.stress)
        @test _s2d_allfinite(fluid.strain_rate)
        @test all(x -> all(abs.(x) .< 50.0), fluid.x)
    end
end

# ---------------------------------------------------------------------------
# Group B — EP_ColumnCollapse.jl shape
# ---------------------------------------------------------------------------

function _ep_column_like(; nfx=15, nfy=10, nbx=40)
    dx_spacing = 0.002
    h_sph      = 1.2 * dx_spacing
    rho0       = 1850.0
    E          = 0.84e6
    nu         = 0.3
    c_sound    = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2 * nu)))
    art_visc_alpha      = 0.1
    art_visc_beta       = 0.1
    soil_friction_angle = 19.8 * π / 180.0
    psi      = 0.0
    cohesion = 0.0

    n_fluid    = nfx * nfy
    fluid_mass = rho0 * dx_spacing * dx_spacing

    fluid = ElastoPlasticParticleSystem(
        "fluid", n_fluid, 2, 4, fluid_mass, c_sound;
        source_v = [0.0, -9.81],
        state_updater = (
            ZeroFieldUpdater(:strain_rate, :vorticity),
            ElastoPlasticStressUpdater(E, nu, soil_friction_angle, psi, cohesion),
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
    update_state!(fluid)

    bottom_boundary = BasicParticleSystem(
        "bottom_boundary", 3 * (nbx + 3), 2, fluid_mass, c_sound,
    )
    for i in 1:nbx+3, j in 1:3
        bottom_boundary.x[(i-1)*3+j] = SVector((i - 3.5) * dx_spacing, -(j - 0.5) * dx_spacing)
    end
    bottom_boundary.rho .= rho0
    fill!(bottom_boundary.v, zero(SVector{2,Float64}))

    return (h_sph=h_sph, art_visc_alpha=art_visc_alpha, art_visc_beta=art_visc_beta,
            fluid=fluid, bottom_boundary=bottom_boundary)
end

# Mirrors EP_ColumnCollapse.jl's wiring exactly, including its 1-arg
# GhostParticleSystem updater list -> the copier fires at stage 1, not
# stage 2 (a pre-existing off-by-one quirk in the real script; replicated
# faithfully here, not corrected).
function _ep_column_integrator(p; onesided::Bool)
    kernel = CubicSplineKernel(p.h_sph; ndims=2)
    fluid, bottom_boundary = p.fluid, p.bottom_boundary

    dynamic_bottom = DynamicBoundarySystem(bottom_boundary, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

    left_ghost       = GhostParticleSystem(fluid, GhostCopier(:stress))
    left_ghost_entry = GhostEntry(left_ghost, 3.0 * p.h_sph, (SVector(1.0, 0.0), SVector(0.0, 0.0)))

    sr_vor_pfn     = StrainRateVorticityPfn()
    kinematics_pfn = CauchyFluidPfn(p.art_visc_alpha, p.art_visc_beta, p.h_sph)

    fluid_interaction = SystemInteraction(kernel, (sr_vor_pfn, kinematics_pfn), fluid; onesided=onesided)
    fluid_bottom_boundary_interaction = SystemInteraction(
        kernel, (sr_vor_pfn, kinematics_pfn), fluid, dynamic_bottom; onesided=onesided)
    fluid_left_boundary_interaction = SystemInteraction(
        kernel, (sr_vor_pfn, kinematics_pfn), fluid, left_ghost; onesided=onesided)

    return LeapFrogTimeIntegrator(
        [fluid, bottom_boundary],
        [fluid_interaction, fluid_bottom_boundary_interaction, fluid_left_boundary_interaction];
        ghosts = (left_ghost_entry,),
    )
end

@testset "Group B (EP_ColumnCollapse.jl shape)" begin
    @testset "short-run trajectory equivalence: onesided=true vs onesided=false" begin
        p_old = _ep_column_like()
        p_new = _ep_column_like()

        integrator_old = _ep_column_integrator(p_old; onesided=false)
        integrator_new = _ep_column_integrator(p_new; onesided=true)

        time_integrate!(integrator_old, _S2D_SHORT_NSTEPS, _S2D_SHORT_NSTEPS + 1, _S2D_SHORT_NSTEPS + 1,
                         _S2D_CFL, nothing; print_timer=false)
        time_integrate!(integrator_new, _S2D_SHORT_NSTEPS, _S2D_SHORT_NSTEPS + 1, _S2D_SHORT_NSTEPS + 1,
                         _S2D_CFL, nothing; print_timer=false)

        fluid_old, fluid_new = p_old.fluid, p_new.fluid

        for f in (:x, :v, :rho, :stress, :vorticity, :strain, :strain_p)
            va, vb = getproperty(fluid_old, f), getproperty(fluid_new, f)
            @test _s2d_allfinite(va)
            @test _s2d_allfinite(vb)
        end

        @test _s2d_diff(fluid_old.x, fluid_new.x)             < 1e-8 * _s2d_scale(fluid_old.x)
        @test _s2d_diff(fluid_old.v, fluid_new.v)             < 1e-6 * _s2d_scale(fluid_old.v)
        @test _s2d_diff(fluid_old.rho, fluid_new.rho)         < 1e-6 * _s2d_scale(fluid_old.rho)
        @test _s2d_diff(fluid_old.stress, fluid_new.stress)   < 1e-5 * _s2d_scale(fluid_old.stress)
        @test _s2d_diff(fluid_old.vorticity, fluid_new.vorticity) < 1e-5 * _s2d_scale(fluid_old.vorticity)
        @test _s2d_diff(fluid_old.strain, fluid_new.strain)   < 1e-5 * _s2d_scale(fluid_old.strain)
        @test _s2d_diff(fluid_old.strain_p, fluid_new.strain_p) < 1e-5 * _s2d_scale(fluid_old.strain_p)
    end

    @testset "long-run physical invariants (onesided=true)" begin
        p = _ep_column_like()
        integrator = _ep_column_integrator(p; onesided=true)

        time_integrate!(integrator, _S2D_LONG_NSTEPS, _S2D_LONG_NSTEPS + 1, _S2D_LONG_NSTEPS + 1,
                         _S2D_CFL, nothing; print_timer=false)

        fluid = p.fluid
        @test _s2d_allfinite(fluid.x)
        @test _s2d_allfinite(fluid.v)
        @test _s2d_allfinite(fluid.rho)
        @test _s2d_allfinite(fluid.stress)
        @test _s2d_allfinite(fluid.vorticity)
        @test _s2d_allfinite(fluid.strain)
        @test _s2d_allfinite(fluid.strain_p)
        @test all(x -> all(abs.(x) .< 50.0), fluid.x)
    end
end
