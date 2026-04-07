# GranularColumnCollapse.jl — SPH simulation of a granular column collapse.
#
# Based on the "bursting damn" configuration, except without the ramp.
# Monaghan, J.J. (1994) Simulating Free Surface Flows with SPH. Journal of Computational Physics, 110, 399-406.
# http://dx.doi.org/10.1006/jcph.1994.1034
#
# This initializes a 25m x 25m 2d square of SPH particles representing
# a block of water. The water sits in the bottom-left corner of a 75m long, 40m tall box.
# The water starts from rest is left to flow freely in the box - affected only by gravity
# and repulsive forces from the boundary.
#
# Run from the Grasph.jl directory:
#   julia --project=. GranularColumnCollapse.jl

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx_spacing     = 0.002             # initial spacing between particles
const h_sph          = 1.2 * dx_spacing  # kernel smoothing length
const rho0           = 1850.0            # initial/reference density of particles
const c_sound        = 20.0              # speed of sound (artificial)
const art_visc_alpha = 0.1               # artificial viscosity alpha coefficient
const art_visc_beta  = 0.1               # "                  " beta coefficient
const soil_friction_angle = 19.8*π/180.0 # soil friction angle in radians

const nfx = Int(floor(0.2 / dx_spacing)) # no. of soil particles in the x direction (= 100)
const nfy = Int(floor(0.1 / dx_spacing)) # "                          " y direction (= 50)
const nbx = Int(floor(0.8 / dx_spacing)) # no. of boundary particles in the x direction (= 400)
const nby = nfy                          # "                              " y direction (= 50)

# ---------------------------------------------------------------------------
# Particle setup — fluid (nfx × nfy grid in the lower-left quadrant)
# ---------------------------------------------------------------------------

n_fluid    = nfx * nfy # total number of fluid particles
fluid_mass = rho0 * dx_spacing * dx_spacing # the mass of each particle

# initialize particles (StressParticleSystem: includes p, stress, strain_rate)
fluid = StressParticleSystem(
    "fluid",                 # name of the system - used to identify the the system when printing/saving
    n_fluid,                 # number of particles in the system
    2,                       # number of spatial dimensions
    3,                       # number of Voigt stress components (2D: xx, yy, xy)
    fluid_mass,              # each of the particles' mass
    c_sound;                 # material speed of sound
    source_v = [0.0, -9.81], # a "source term" for velocity i.e. gravity
    state_updater = (
        ZeroFieldUpdater(:strain_rate),
        ViscoPlasticMCStressUpdater(LinearEOSUpdater(rho0), soil_friction_angle, 0.0),
    )
)

# register velocity and density for printing during simulation
add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)
add_print_field!(fluid, :stress)

# setup particles' positions
let k = 1
    for i in 0:nfx-1, j in 0:nfy-1
        fluid.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
        k += 1
    end
end
# setup initial velocity and density
fill!(fluid.v, zero(SVector{2,Float64}))
fluid.rho .= rho0

update_state!(fluid) # initialize stress

# ---------------------------------------------------------------------------
# Particle setup — boundary (bottom wall and left ghost mirror)
# ---------------------------------------------------------------------------

bottom_boundary = BasicParticleSystem(
    "bottom_boundary", 3*(nbx+3), 2, rho0 * dx_spacing * dx_spacing, c_sound;
)
for i in 1:nbx+3
    for j in 1:3
        bottom_boundary.x[(i-1)*3+j] = SVector((i-3.5)*dx_spacing, -(j-0.5)*dx_spacing)
    end
end
bottom_boundary.rho .= rho0
fill!(bottom_boundary.v, zero(SVector{2,Float64}))

# Setup ghost particles for the left wall (mirroring across x=0)
left_ghost  = GhostParticleSystem(fluid, nothing, GhostCopier(:stress))
# normal pointing into the domain (right), point on the wall (x=0)
left_ghost_entry = GhostEntry(left_ghost, SVector(1.0, 0.0), SVector(0.0, 0.0), 3.0 * h_sph)

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

# initialize kernel
kernel = CubicSplineKernel(h_sph; ndims=2)

dynamic_bottom = DynamicBoundarySystem(bottom_boundary, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

sr_pfn = StrainRatePfn()
kinematics_pfn = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)

fluid_interaction = SystemInteraction(
    kernel,                # the kernel to be used in this interaction
    (sr_pfn, kinematics_pfn),
    fluid                 # the particles in the interaction
)

fluid_bottom_boundary_interaction = SystemInteraction(
    kernel,
    (sr_pfn, kinematics_pfn),
    fluid,
    dynamic_bottom
)

fluid_left_boundary_interaction = SystemInteraction(
    kernel,
    (sr_pfn, kinematics_pfn),
    fluid,
    left_ghost
)

integrator = LeapFrogTimeIntegrator(
    [fluid, bottom_boundary],
    [fluid_interaction, fluid_bottom_boundary_interaction, fluid_left_boundary_interaction];
    ghosts = (left_ghost_entry,)
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_fluid = $n_fluid  |  mass = $fluid_mass")

run_driver!(
    integrator,
    50000,                # number of timesteps to run for
    500,                  # frequency to print summary stats to terminal
    500,                  # frequency to save particle data to disk
    0.1,                  # the CFL coefficient
    "gcc-output/sph"  # the output path prefix
)
