# bubble3.jl — Similar setup to bubble2.jl but with XSPH corrections and artificial surface tension.
#
# In addition to bubble2.jl, XSPH correction and artificial surface tension. XSPH correction follows
# equation 22, but the artificial surface tension deviates from the paper and instead follows
# equation 17 in Hammani et al. (2020). The description in the original work wasn't well described.
# Another deviation was to use the Wenland C2 kernel (also in line with Hammani et al.).
# Colagrossi, A., Landrini, M. (2003) Numerical simulation of interfacial flows by smoothed particle hydrodynamics. Journal of Computational Physics, 191, 448-475.
# https://doi.org/10.1016/S0021-9991(03)00324-3
# I. Hammani, S. Marrone, A. Colagrossi, G. Oger, D. Le Touzé, Detailed study on the extension of the δ-SPH model to multi-phase flow. Computer Methods in Applied Mechanics and Engineering, 368, 113189.
# https://doi.org/10.1016/j.cma.2020.113189 
#
# Run from the Grasph.jl directory:
#   julia --project=. bubble3.jl

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const R              = 1.0               # radius of bubble
const height         = 10.0*R            # height of fluid column
const width          = 6.0*R             # width of fluid column
const y_min          = -2.0*R            # puts centre of circle at origin
const y_max          = y_min + height    # maximum extent in y
const x_min          = -3.0*R            # puts centre of circle at origin
const x_max          = x_min + width     # maximum extent in x
const dx_spacing     = 0.04              # initial spacing between particles
const h_sph          = 1.2 * dx_spacing  # kernel smoothing length
const rho_X          = 1000.0            # reference density of heavier fluid particles
const rho_Y          = 1.0               # "                  " lighter "             "
const g              = 9.81              # gravity
const c_sound_X = sqrt(800.0*g*R)        # speed of sound for X particles
const c_sound_Y = 400.0*sqrt(g*R)        # "                " Y particles - 
                                         #   chosen so that rho0*c^2/gamma (tait equation) is equal 
                                         #   in both fluix X and Y. i.e. found by solving for
                                         #   rho_X*c_sound_X^2/γ_X = rho_Y*c_sound_X^2/γ_Y
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
    state_updater = TaitEOSUpdater(rho_Y, 1.4)
)

add_print_field!(fluid_Y, :v)
add_print_field!(fluid_Y, :rho)

for i in 1:length(x_Y)
    fluid_Y.x[i] = SVector(x_Y[i], y_Y[i])
    pressure = (y_min + height - y_Y[i])*rho_X*g
    fluid_Y.rho[i] = (pressure*1.4/(c_sound_Y*c_sound_Y*rho_Y) + 1.0)^(1.0/1.4)*rho_Y
end
fill!(fluid_Y.v, zero(SVector{2,Float64}))

# ---------------------------------------------------------------------------
# Particle setup — ghost boundaries
# ---------------------------------------------------------------------------

# Single ghost system representing all 4 walls and 4 corner boundaries.
boundary_ghost = GhostParticleSystem(fluid_X, GhostCopier(:p); name="ghost[fluid_X]")

boundary_ghost_entry = GhostEntry(boundary_ghost, 3.0 * h_sph,
    (SVector( 1.0,  0.0),            SVector(x_min, 0.0  )),  # left wall
    (SVector(-1.0,  0.0),            SVector(x_max, 0.0  )),  # right wall
    (SVector( 0.0,  1.0),            SVector(0.0,   y_min)),  # bottom wall
    (SVector( 0.0, -1.0),            SVector(0.0,   y_max)),  # top wall
    (SVector( 1.0,  1.0)/sqrt(2.0),  SVector(x_min, y_min)),  # bottom-left corner
    (SVector(-1.0,  1.0)/sqrt(2.0),  SVector(x_max, y_min)),  # bottom-right corner
    (SVector( 1.0, -1.0)/sqrt(2.0),  SVector(x_min, y_max)),  # top-left corner
    (SVector(-1.0, -1.0)/sqrt(2.0),  SVector(x_max, y_max)),  # top-right corner
)

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

# initialize kernel
kernel = WenlandC2Kernel(h_sph; ndims=2)

# heavy fluid particles interacting with themselves
fluid_X_interaction = SystemInteraction(
    kernel,                # the kernel to be used in this interaction
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph; sigma=1),
    fluid_X;                # the particles in the interaction
    velocity_adjust_pairwise_fn=XSPHPfn(0.5)
)

# bubble fluid particles interacting with themselves
fluid_Y_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph; sigma=1),
    fluid_Y;
    velocity_adjust_pairwise_fn=XSPHPfn(0.5)
)

# bubble/heavy fluid particles interacting with eachother
fluid_XY_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph; sigma=1, epsilon=0.1),
    fluid_Y, # useful to make Y the first system as the iteration space is smaller
    fluid_X,
)

# heavy fluid particles interacting with boundary.
# should technically also add an interaction of fluid_Y with boundary,
# but we're not running the simulation long enough for the bubble to touch
# the top wall.
fluid_boundary_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph; sigma=1),
    fluid_X,
    boundary_ghost;
    velocity_adjust_pairwise_fn=XSPHPfn(0.5)
)

integrator = RK4TimeIntegrator(
    # vector of particle systems involved in the simulation.
    # the boundary system could be omitted, as their positions and properties don't get
    # updated, but they're included here so their data is written and can be visualised.
    [fluid_X, fluid_Y],
    [fluid_X_interaction, fluid_Y_interaction, fluid_XY_interaction, fluid_boundary_interaction];
    ghosts = [boundary_ghost_entry]
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run_driver!(
    [Stage(integrator, 60000, 1.5, "run")],
    200,
    200,
    "bubble3-output/sph",
)
