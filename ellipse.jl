# ellipse.jl — SPH simulation of a rotating fluid in a unit circle.
#
# Based on "THE EVOLUTION OF AN ELLIPTICAL DROP" configuration.
# Monaghan, J.J. (1994) Simulating Free Surface Flows with SPH. Journal of Computational Physics, 110, 399-406.
# http://dx.doi.org/10.1006/jcph.1994.1034
#
# This initializes SPH particles representing fluid arranged on a square lattive in
# the shape of a circle with radius 1m - centered at the origin. The water starts 
# with velocity v_x = -100x, v_y = 100y and continues unaffacted by external action.
#
# Run from the Grasph.jl directory:
#   julia --project=. ellipse.jl

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx              = 0.04      # initial spacing between particles
const h_sph           = 1.2 * dx  # kernel smoothing length
const rho0            = 1000.0    # initial/reference density of particles
const c_sound         = 1400.0    # speed of sound (artificial)
const art_visc_alpha  = 0.01      # artificial viscosity alpha coefficient
const art_visc_beta   = 0.0       # "                  " beta coefficient

# ---------------------------------------------------------------------------
# Particle positions — regular grid clipped to the unit circle
# ---------------------------------------------------------------------------

# helper function to generate particles' positions
function _circle_particles(dx)
    n = Int(round(2.0 / dx))
    xs = Float64[]
    ys = Float64[]
    for i in 0:n-1, j in 0:n-1
        x = -1.0 + (i + 0.5) * dx
        y = -1.0 + (j + 0.5) * dx
        x*x + y*y < 1.0 && (push!(xs, x); push!(ys, y))
    end
    xs, ys
end

# these will be used to initialize the particle system later
xs, ys = _circle_particles(dx)
n_particles = length(xs)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Each particle represents an equal share of total fluid mass.
# Unit-circle area = π, so total mass = π * rho0.
particle_mass = π * rho0 / n_particles # the mass of each particle

# initialize particles
particles = FluidParticleSystem(
    "fluid",       # name of the system - used to identify the the system when printing/saving
    n_particles,   # number of particles in the system
    2,             # number of spatial dimensions
    particle_mass, # each of the particles' mass
    c_sound;       # material speed of sound
    state_updater = TaitEOSUpdater(rho0)
)
# register velocity and density for printing during simulation
add_print_field!(particles, :v)
add_print_field!(particles, :rho)

# setup particles' positions, velocity, and density
for k in 1:n_particles
    particles.x[k]   = SVector(xs[k], ys[k])
    particles.v[k]   = SVector(-100.0 * xs[k], 100.0 * ys[k])
    particles.rho[k] = rho0
end

update_state!(particles)


# initialize kernel
kernel = CubicSplineKernel(h_sph; ndims=2)
si = SystemInteraction(
    kernel,                   # the kernel to be used in this interaction
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    particles                 # the particles in the interaction
)                             # since only one system is supplied, the interaction describes
                              # the interaction between particles within the system

# initialize the time integrator with the system and interaction
# supplying the system ensures that the particles properties and
# positions evolve with time.
integrator = LeapFrogTimeIntegrator(particles, si)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_particles = $n_particles  |  mass = $(particle_mass)")

run_driver!(
    [Stage(integrator, 5000, 0.05, "run")],
    10,
    10,
    "ellipse-output/sph",
)
