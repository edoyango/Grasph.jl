# EP_ColumnCollapse.jl — Elasto-plastic SPH simulation of a granular column collapse.
#
# Based on GranularColumnCollapse.jl but using ElastoPlasticParticleSystem
# and ElastoPlasticStressUpdater.

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx_spacing     = 0.002             # initial spacing between particles
const h_sph          = 1.2 * dx_spacing  # kernel smoothing length
const rho0           = 1850.0            # initial/reference density of particles
const art_visc_alpha = 0.1               # artificial viscosity alpha coefficient
const art_visc_beta  = 0.1               # "                  " beta coefficient
const soil_friction_angle = 19.8*π/180.0 # soil friction angle in radians

# Elasto-plastic parameters
const E        = 0.84e6                  # Young's modulus
const nu       = 0.3                     # Poisson's ratio
const psi      = 0.0                     # Dilation angle
const cohesion = 0.0                     # Cohesion
const c_sound  = sqrt(E*(1-nu)/(rho0*(1+nu)*(1-2*nu))) # speed of sound in solid
const dt       = 0.1*h_sph/c_sound       # Fixed timestep for stress update

const nfx = Int(floor(0.2 / dx_spacing)) # no. of soil particles in the x direction (= 100)
const nfy = Int(floor(0.1 / dx_spacing)) # "                          " y direction (= 50)
const nbx = Int(floor(0.5 / dx_spacing)) # no. of boundary particles in the x direction (= 400)
const nby = nfy                          # "                              " y direction (= 50)

# ---------------------------------------------------------------------------
# Particle setup — soil (nfx × nfy grid in the lower-left quadrant)
# ---------------------------------------------------------------------------

n_fluid    = nfx * nfy # total number of fluid particles
fluid_mass = rho0 * dx_spacing * dx_spacing # the mass of each particle

# initialize particles (ElastoPlasticParticleSystem: includes stress, strain, vorticity, etc.)
fluid = ElastoPlasticParticleSystem(
    "fluid",                 # name of the system
    n_fluid,                 # number of particles in the system
    2,                       # number of spatial dimensions
    4,                       # number of Voigt stress components (2D plane strain: xx, yy, zz, xy)
    fluid_mass,              # each of the particles' mass
    c_sound;                 # material speed of sound
    source_v = [0.0, -9.81], # gravity
    state_updater = (
        ZeroFieldUpdater(:strain_rate, :vorticity),
        ElastoPlasticStressUpdater(E, nu, soil_friction_angle, psi, cohesion, dt),
    )
)

# register fields for printing
add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)
add_print_field!(fluid, :stress)
add_print_field!(fluid, :vorticity)
add_print_field!(fluid, :strain)
add_print_field!(fluid, :strain_p)

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
left_ghost  = GhostParticleSystem(fluid, GhostCopier(:stress))
left_ghost_entry = GhostEntry(left_ghost, 3.0 * h_sph, (SVector(1.0, 0.0), SVector(0.0, 0.0)))

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

kernel = CubicSplineKernel(h_sph; ndims=2)
dynamic_bottom = DynamicBoundarySystem(bottom_boundary, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

# Use the new combined functor
sr_vor_pfn = StrainRateVorticityPfn()
kinematics_pfn = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)

fluid_interaction = SystemInteraction(
    kernel,
    (sr_vor_pfn, kinematics_pfn),
    fluid
)

fluid_bottom_boundary_interaction = SystemInteraction(
    kernel,
    (sr_vor_pfn, kinematics_pfn),
    fluid,
    dynamic_bottom
)

fluid_left_boundary_interaction = SystemInteraction(
    kernel,
    (sr_vor_pfn, kinematics_pfn),
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
    "ep-gcc-output/sph"   # the output path prefix
)
