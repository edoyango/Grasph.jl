# dambreak.jl — SPH simulation of a dam break.
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
#   julia --project=. dambreak.jl

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx_spacing     = 0.5                             # initial spacing between particles
const h_sph          = 1.2 * dx_spacing                # kernel smoothing length
const rho0           = 1000.0                          # initial/reference density of particles
const c_sound        = 10.0 * sqrt(2.0 * 9.81 * 25.0)  # speed of sound (artificial)
const art_visc_alpha = 0.01                            # artificial viscosity alpha coefficient
const art_visc_beta  = 0.0                             # "                  " beta coefficient

const nfx = Int(floor(25.0 / dx_spacing))              # no. of fluid particles in the x direction (= 50)
const nfy = nfx                                        # "                           " y direction (= 50)
const nbx = Int(floor(75.0 / dx_spacing))              # no. of boundary particles in the x direction (= 150)
const nby = Int(floor(40.0 / dx_spacing))              # "                              " y direction (= 80)

# ---------------------------------------------------------------------------
# Particle setup — fluid (nfx × nfy grid in the lower-left quadrant)
# ---------------------------------------------------------------------------

n_fluid    = nfx * nfy # total number of fluid particles
fluid_mass = rho0 * dx_spacing * dx_spacing # the mass of each particle

# initialize particles
fluid = FluidParticleSystem(
    "fluid",                 # name of the system - used to identify the the system when printing/saving
    n_fluid,                 # number of particles in the system
    2,                       # number of spatial dimensions
    fluid_mass,              # each of the particles' mass
    c_sound;                 # material speed of sound
    source_v = [0.0, -9.81], # a "source term" for velocity i.e. gravity
    state_updater = TaitEOSUpdater(rho0)
)

# register velocity and density for printing during simulation
add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)

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

update_state!(fluid, 1) # initialize pressure

# ---------------------------------------------------------------------------
# Particle setup — boundary (single layer of wall particles)
# ---------------------------------------------------------------------------

n_boundary = 2*(nbx + nby) + 4            # = 464

boundary = BasicParticleSystem(
    "boundary", n_boundary, 2, rho0 * dx_spacing * dx_spacing, c_sound;
)

let k = 1
    # Bottom wall: x in [-0.5, 75.5], y = -0.5*dx
    for i in -1:nbx
        boundary.x[k] = SVector((i + 0.5) * dx_spacing, -0.5 * dx_spacing)
        k += 1
    end
    # Top wall: x in [-0.5, 75.5], y = 40 + 0.5*dx
    for i in -1:nbx
        boundary.x[k] = SVector((i + 0.5) * dx_spacing, 40.0 + 0.5 * dx_spacing)
        k += 1
    end
    # Left wall: x = -0.5*dx, y in [0.5, 39.5]
    for j in 0:nby-1
        boundary.x[k] = SVector(-0.5 * dx_spacing, (j + 0.5) * dx_spacing)
        k += 1
    end
    # Right wall: x = 75 + 0.5*dx, y in [0.5, 39.5]
    for j in 0:nby-1
        boundary.x[k] = SVector(75.0 + 0.5 * dx_spacing, (j + 0.5) * dx_spacing)
        k += 1
    end
end

boundary.rho .= rho0
fill!(boundary.v, zero(SVector{2,Float64}))

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

# initialize kernel
kernel = CubicSplineKernel(h_sph; ndims=2)

static_boundary = StaticBoundarySystem(boundary, dx_spacing)

fluid_interaction = SystemInteraction(
    kernel,                # the kernel to be used in this interaction
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid                 # the particles in the interaction
)

fluid_boundary_interaction = SystemInteraction(
    kernel,                             # the kernel to be used in this interaction
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid,                              # the particles corresponding to particle "i"
    static_boundary                    # the particles corresponding to particle "j"
)

integrator = LeapFrogTimeIntegrator(
    # vector of particle systems involved in the simulation.
    # the boundary system could be omitted, as their positions and properties don't get
    # updated, but they're included here so their data is written and can be visualised.
    [fluid, boundary], 
    [fluid_interaction, fluid_boundary_interaction],
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_fluid = $n_fluid  |  n_boundary = $n_boundary  |  mass = $fluid_mass")

run_driver!(
    integrator, 
    100000,                # number of timesteps to run for
    1000,                  # frequency to print summary stats to terminal
    1000,                  # frequency to save particle data to disk
    0.05,                  # the CFL coefficient
    "dambreak-output/sph"  # the output path prefix
)
