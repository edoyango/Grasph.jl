# GranularColumnCollapse3D.jl — 3D SPH simulation of a granular column collapse.
#
# Three-dimensional adaptation of GranularColumnCollapse.jl.
#
# Soil block: 0.2 m (x) × 0.04 m (y) × 0.1 m (z), sitting at the
# bottom-left corner of a 0.8 m (x) × 0.04 m (y) box.
# Vertical coordinate is Z.
#
# Boundary treatment:
#   - Bottom (z = 0):     flat sheet of BasicParticleSystem particles
#                          wrapped as DynamicBoundarySystem
#   - Left   (x = 0):     ghost mirror (normal +x)
#   - Front  (y = 0):     ghost mirror (normal +y)
#   - Back   (y = Y_max): ghost mirror (normal -y)
#
# Stress in 6-component Voigt notation (xx, yy, zz, xy, xz, yz).
# Pairwise functors:
#   StrainRatePfn  — NS=6 uses strain_rate_tensor(dv::SVector{3}, gx::SVector{3})
#   CauchyFluidPfn — NS=6 uses cauchy_stress_force(SVector{6}, SVector{6}, gx::SVector{3})
# Both paths are fully implemented in PairwisePhysics.jl.
#
# Run from the Grasph.jl directory:
#   julia --project=. GranularColumnCollapse3D.jl

using Grasph
using StaticArrays
using Printf

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx_spacing          = 0.002              # initial particle spacing (5 mm)
const h_sph               = 1.2 * dx_spacing   # kernel smoothing length
const rho0                = 1850.0             # reference density (kg/m³)
const c_sound             = 20.0              # artificial speed of sound
const art_visc_alpha      = 0.1
const art_visc_beta       = 0.1
const soil_friction_angle = 19.8 * π / 180.0  # radians

# Soil block (particles)
const nfx = Int(floor(0.2  / dx_spacing))   # x: 0.2 m  → 40
const nfy = Int(floor(0.05 / dx_spacing))   # y: 0.04 m → 10   (thin slab)
const nfz = Int(floor(0.1  / dx_spacing))   # z: 0.1 m  → 20

# Bottom boundary extent
const nbx = Int(floor(0.8 / dx_spacing))    # x: 0.8 m  → 160
const nby = nfy                              # same y-extent as soil

# ---------------------------------------------------------------------------
# Particle setup — soil (nfx × nfy × nfz block in the lower-left corner)
# ---------------------------------------------------------------------------

n_fluid    = nfx * nfy * nfz
fluid_mass = rho0 * dx_spacing^3

# StressParticleSystem with NS=6 (3D Voigt: xx, yy, zz, xy, xz, yz)
fluid = StressParticleSystem(
    "soil",
    n_fluid,
    3,                              # ND = 3
    6,                              # NS = 6
    fluid_mass,
    c_sound;
    source_v = [0.0, 0.0, -9.81],  # gravity in -z
    state_updater = (
        ZeroFieldUpdater(:strain_rate),
        ViscoPlasticMCStressUpdater(LinearEOSUpdater(rho0), soil_friction_angle, 0.0),
    )
)

add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)
add_print_field!(fluid, :stress)

let k = 1
    for i in 0:nfx-1, j in 0:nfy-1, m in 0:nfz-1
        fluid.x[k] = SVector((i + 0.5)*dx_spacing,
                              (j + 0.5)*dx_spacing,
                              (m + 0.5)*dx_spacing)
        k += 1
    end
end
fill!(fluid.v, zero(SVector{3,Float64}))
fluid.rho .= rho0

update_state!(fluid)   # initialise stress from initial density

# ---------------------------------------------------------------------------
# Particle setup — bottom boundary (flat sheet, 3 layers below z = 0)
# ---------------------------------------------------------------------------

n_bottom = (nbx + 3) * (nby + 6) * 3 # (bottomx + west edge) * (bottomy + north edge + south edge) * nlayer

bottom_boundary = BasicParticleSystem(
    "bottom_boundary", n_bottom, 3, rho0 * dx_spacing^3, c_sound;
)

let k = 1
    for i in 1:nbx+3
        for j in 1:nby + 6
            for m in 1:3
                bottom_boundary.x[k] = SVector(
                    (i - 3.5)  * dx_spacing,   # x: same offset as 2D
                    (j - 2.5)  * dx_spacing,   # y: spans soil depth
                    -(m - 0.5) * dx_spacing,   # z: below floor
                )
                k += 1
            end
        end
    end
end
bottom_boundary.rho .= rho0
fill!(bottom_boundary.v, zero(SVector{3,Float64}))

# ---------------------------------------------------------------------------
# Ghost particles — left, front, back walls
# ---------------------------------------------------------------------------

const Y_max = nfy * dx_spacing

left_ghost  = GhostParticleSystem(fluid, nothing, GhostCopier(:stress); name="ghost_left")
front_ghost = GhostParticleSystem(fluid, nothing, GhostCopier(:stress); name="ghost_front")
back_ghost  = GhostParticleSystem(fluid, nothing, GhostCopier(:stress); name="ghost_back")
north_west_ghost = GhostParticleSystem(fluid, nothing, GhostCopier(:stress); name="ghost_northwest")
south_west_ghost = GhostParticleSystem(fluid, nothing, GhostCopier(:stress); name="ghost_southwest")

# Normal points into the domain; point lies on the wall plane.
left_ghost_entry  = GhostEntry(left_ghost,
                               SVector(1.0, 0.0,  0.0),
                               SVector(0.0, 0.0,  0.0), 3.0*h_sph)
front_ghost_entry = GhostEntry(front_ghost,
                               SVector(0.0, 1.0,  0.0),
                               SVector(0.0, 0.0,  0.0), 3.0*h_sph)
back_ghost_entry  = GhostEntry(back_ghost,
                               SVector(0.0, -1.0, 0.0),
                               SVector(0.0, Y_max, 0.0), 3.0*h_sph)
north_west_ghost_entry = GhostEntry(north_west_ghost,
                                    SVector(1.0, -1.0, 0.0)/sqrt(2),
                                    SVector(0.0, Y_max, 0.0), 3.0*h_sph)
south_west_ghost_entry = GhostEntry(south_west_ghost,
                                    SVector(1.0, 1.0, 0.0)/sqrt(2), 
                                    SVector(0.0, 0.0, 0.0), 3.0*h_sph)

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

kernel = CubicSplineKernel(h_sph; ndims=3)

dynamic_bottom = DynamicBoundarySystem(
    bottom_boundary,
    SVector(0.0, 0.0, 1.0),   # boundary normal: +z (floor pointing up)
    SVector(0.0, 0.0, 0.0),   # point on floor
    3.0,                        # velocity ratio cap (beta)
)

sr_pfn         = StrainRatePfn()
kinematics_pfn = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)
pfns           = (sr_pfn, kinematics_pfn)

fluid_interaction = SystemInteraction(kernel, pfns, fluid)

fluid_bottom_interaction = SystemInteraction(kernel, pfns, fluid, dynamic_bottom)

fluid_left_interaction  = SystemInteraction(kernel, pfns, fluid, left_ghost)
fluid_front_interaction = SystemInteraction(kernel, pfns, fluid, front_ghost)
fluid_back_interaction  = SystemInteraction(kernel, pfns, fluid, back_ghost)
fluid_northwest_interaction = SystemInteraction(kernel, pfns, fluid, north_west_ghost)
fluid_southwest_interaction  = SystemInteraction(kernel, pfns, fluid, south_west_ghost)

integrator = LeapFrogTimeIntegrator(
    [fluid, bottom_boundary],
    [fluid_interaction,
     fluid_bottom_interaction,
     fluid_left_interaction,
     fluid_front_interaction,
     fluid_back_interaction,
     fluid_northwest_interaction,
     fluid_southwest_interaction];
    ghosts = (left_ghost_entry, front_ghost_entry, back_ghost_entry, north_west_ghost_entry, south_west_ghost_entry),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@printf("n_soil = %d  |  n_bottom = %d  |  mass = %.4g\n",
        n_fluid, n_bottom, fluid_mass)

run_driver!(
    integrator,
    50000,                        # number of timesteps
    1000,                          # print frequency
    1000,                          # save frequency
    0.1,                          # CFL coefficient
    "gcc-3d-output/sph"
)
