# bubble.jl — SPH multi-phase simulation of a lighter fluid (bubble) interacting with a surrounding
#   heavier fluid.
#
# Based on the "bubble" configuration in Figure 1.
# Colagrossi, A., Landrini, M. (2003) Numerical simulation of interfacial flows by smoothed particle hydrodynamics. Journal of Computational Physics, 191, 448-475.
# https://doi.org/10.1016/S0021-9991(03)00324-3
#
# This initializes a 6m x 10m column of "heavy" fluid with a bubble of lighter fluid located at 2m
# from the bottom. The column is positions such that the bubble centre is at the origin. The bubble
# fluid's density is half that of the heavier fluid. Both fluids use the Tait equation of state to
# link their density to pressure. Simulation setup follows the paper, except the basic Cubic Bspline
# kernel and Leap Frog time integartion are used instead of the modified gaussian kernel, and RK4
# integration. Additionally, no background pressure is applied. Instead, particles' density are
# initialized using a hydrostatic pressure condition to mitigate the initial pressure waves.
#
# Run from the Grasph.jl directory:
#   julia --project=. bubble.jl

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const R              = 1.0               # radius of bubble
const height         = 10.0*R            # height of fluid column
const width          = 6.0*R             # width of fluid column
const y_min          = -2.0*R            # puts centre of circle at origin
const y_max          = y_min + height
const x_min          = -3.0*R            # "                             "
const x_max          = x_min + width
const dx_spacing     = 0.05              # initial spacing between particles
const h_sph          = 1.2 * dx_spacing  # kernel smoothing length
const rho_X          = 1000.0            # reference density of heavier fluid particles
const rho_Y          = 500.0             # "                  " lighter "             "
const g              = 9.81
const c_sound_X = sqrt(800.0*g*R)        # speed of sound for X particles
const c_sound_Y = 40.0*sqrt(g*R)         # "                " Y particles - 
                                         #   chosen so that rho0*c^2/gamma (tait equation) is equal 
                                         #   in both fluix X and Y.
const art_visc_alpha = 0.01              # artificial viscosity alpha coefficient
const art_visc_beta  = 0.0               # "                  " beta coefficient

const nx = Int(floor(width/dx_spacing))  # number of particles in x direction
const ny = Int(floor(height/dx_spacing)) # "                    " y "       "

# --------------------------------------------------------------------------
# Particle positions setup
# --------------------------------------------------------------------------

x_X = Float64[] # x positions for fluid X
y_X = Float64[] # y "                   "
x_Y = Float64[] # x positions for fluid Y
y_Y = Float64[] # y "                   "
for i in 0:nx-1 , j in 0:ny-1
    x = x_min + (i+0.5)*dx_spacing
    y = y_min + (j+0.5)*dx_spacing
    # particles within radius belong to bubble (Y), else belong to heavier fluid (X)
    if x*x + y*y < R
        push!(x_Y, x)
        push!(y_Y, y)
    else
        push!(x_X, x)
        push!(y_X, y)
    end
end

# ---------------------------------------------------------------------------
# Particle setup — heavy fluix (X)
# ---------------------------------------------------------------------------

fluid_X_mass = dx_spacing*dx_spacing*rho_X

fluid_Y_mass = dx_spacing*dx_spacing*rho_Y

# initialize particles
fluid_X = FluidParticleSystem(
    "fluid X",                 # name of the system - used to identify the the system when printing/saving
    length(x_X),               # number of particles in the system
    2,                         # number of spatial dimensions
    fluid_X_mass,              # each of the particles' mass
    c_sound_X;                 # material speed of sound
    source_v = [0.0, -g],   # a "source term" for velocity i.e. gravity
    state_updater = TaitEOSUpdater(rho_X)
)

# register velocity and density for printing during simulation
add_print_field!(fluid_X, :v)
add_print_field!(fluid_X, :rho)

# copy particle positions into particle system
# initialize density based on hydrostatic condition
for i in 1:length(x_X)
    fluid_X.x[i] = SVector(x_X[i], y_X[i])
    pressure = (y_min + height - y_X[i])*rho_X*g
    fluid_X.rho[i] = (pressure*7.0/(c_sound_X*c_sound_X*rho_X) + 1.0)^(1.0/7.0)*rho_X
end
fill!(fluid_X.v, zero(SVector{2,Float64}))

# repeat for bubble particles
fluid_Y = FluidParticleSystem(
    "fluid Y",
    length(x_Y),
    2,
    fluid_Y_mass,
    c_sound_Y;
    source_v = [0.0, -g],
    state_updater = TaitEOSUpdater(rho_Y)
)

add_print_field!(fluid_Y, :v)
add_print_field!(fluid_Y, :rho)

for i in 1:length(x_Y)
    fluid_Y.x[i] = SVector(x_Y[i], y_Y[i])
    pressure = (y_min + height - y_Y[i])*rho_X*g
    fluid_Y.rho[i] = (pressure*7.0/(c_sound_Y*c_sound_Y*rho_Y) + 1.0)^(1.0/7.0)*rho_Y
end
fill!(fluid_Y.v, zero(SVector{2,Float64}))

# ---------------------------------------------------------------------------
# Particle setup — ghost boundaries
# ---------------------------------------------------------------------------

left_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost left[fluid_X]")
right_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost right[fluid_X]")
bottom_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost bottom[fluid_X]")
top_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost top[fluid_X]")
bottom_left_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost bottom left[fluid_X]")
bottom_right_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost bottom right[fluid_X]")
top_left_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost top left[fluid_X]")
top_right_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost top right[fluid_X]")

left_ghost_entry = GhostEntry(left_ghost, SVector(1.0, 0.0), SVector(x_min, 0.0), 3.0 * h_sph)
right_ghost_entry = GhostEntry(right_ghost, SVector(-1.0, 0.0), SVector(x_max, 0.0), 3.0 * h_sph)
bottom_ghost_entry = GhostEntry(bottom_ghost, SVector(0.0, 1.0), SVector(0.0, y_min), 3.0 * h_sph)
top_ghost_entry = GhostEntry(top_ghost, SVector(0.0, -1.0), SVector(0.0, y_max), 3.0 * h_sph)
bottom_left_ghost_entry = GhostEntry(bottom_left_ghost, SVector(1.0, 1.0)/sqrt(2.0), SVector(x_min, y_min), 3.0 * h_sph)
bottom_right_ghost_entry = GhostEntry(bottom_right_ghost, SVector(-1.0, 1.0)/sqrt(2.0), SVector(x_max, y_min), 3.0 * h_sph)
top_left_ghost_entry = GhostEntry(top_left_ghost, SVector(1.0, -1.0)/sqrt(2.0), SVector(x_min, y_max), 3.0 * h_sph)
top_right_ghost_entry = GhostEntry(top_right_ghost, SVector(-1.0, -1.0)/sqrt(2.0), SVector(x_max, y_max), 3.0 * h_sph)

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

# initialize kernel
kernel = CubicSplineKernel(h_sph; ndims=2)

# heavy fluid particles interacting with themselves
fluid_X_interaction = SystemInteraction(
    kernel,                # the kernel to be used in this interaction
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X                 # the particles in the interaction
)

# bubble fluid particles interacting with themselves
fluid_Y_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_Y
)

# bubble/heavy fluid particles interacting with eachother
fluid_XY_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_Y, # useful to make Y the first system as the iteration space is smaller
    fluid_X,
)

# heavy fluid particles interacting with boundary
# should technically also add an interaction of fluid_Y with boundary, 
# but we're not running the simulation long enough for the bubble to touch
# the top wall.
fluid_left_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    left_ghost
)
fluid_right_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    right_ghost
)
fluid_bottom_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    bottom_ghost
)
fluid_top_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    top_ghost
)
fluid_bottom_left_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    bottom_left_ghost
)
fluid_bottom_right_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    bottom_right_ghost
)
fluid_top_left_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    top_left_ghost
)
fluid_top_right_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid_X,
    top_right_ghost
)

integrator = LeapFrogTimeIntegrator(
    # vector of particle systems involved in the simulation.
    # the boundary system could be omitted, as their positions and properties don't get
    # updated, but they're included here so their data is written and can be visualised.
    [fluid_X, fluid_Y], 
    [fluid_X_interaction, fluid_Y_interaction, fluid_XY_interaction, fluid_left_interaction, fluid_right_interaction, fluid_top_interaction, fluid_bottom_interaction, fluid_bottom_left_interaction, fluid_bottom_right_interaction, fluid_top_left_interaction, fluid_top_right_interaction];
    ghosts = [left_ghost_entry, right_ghost_entry, top_ghost_entry, bottom_ghost_entry, bottom_left_ghost_entry, bottom_right_ghost_entry, top_left_ghost_entry, top_right_ghost_entry]
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run_driver!(
    integrator, 
    150000,               # number of timesteps to run for
    500,                  # frequency to print summary stats to terminal
    500,                  # frequency to save particle data to disk
    0.05,                 # the CFL coefficient
    "bubble-output/sph"  # the output path prefix
)
