# dambreak_3d.jl — 3D SPH simulation of a dam break.
#
# Based on the Monaghan (1994) configuration, extended to 3D.
# The fluid is a 25m x 12.5m x 25m block in a 75m x 12.5m x 40m box.
# Vertical coordinate is Z.

using Grasph
using StaticArrays
using Printf

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx_spacing     = 0.5                             # initial spacing between particles
const h_sph          = 1.2 * dx_spacing                # kernel smoothing length
const rho0           = 1000.0                          # initial/reference density of particles
const c_sound        = 10.0 * sqrt(2.0 * 9.81 * 25.0)  # speed of sound (artificial)
const art_visc_alpha = 0.005                           # artificial viscosity alpha coefficient
const art_visc_beta  = 0.0                             # "                  " beta coefficient

# Fluid dimensions (number of particles)
const nfx = Int(floor(25.0 / dx_spacing))              # length (x): 25m -> 50
const nfy = 12                                         # thickness (y): 25 particles -> 12.5m
const nfz = Int(floor(25.0 / dx_spacing))              # height (z): 25m -> 50

# Box dimensions (number of particles)
const nbx = Int(floor(75.0 / dx_spacing))              # length (x): 75m -> 150
const nby = nfy                                        # thickness (y): same as fluid -> 25
const nbz = Int(floor(40.0 / dx_spacing))              # height (z): 40m -> 80

# ---------------------------------------------------------------------------
# Particle setup — fluid
# ---------------------------------------------------------------------------

n_fluid    = nfx * nfy * nfz
fluid_mass = rho0 * dx_spacing^3

# initialize particles
fluid = FluidParticleSystem(
    "fluid",
    n_fluid,
    3,                       # ND = 3
    fluid_mass,
    c_sound;
    source_v = [0.0, 0.0, -9.81], # Gravity in Z
    state_updater = TaitEOSUpdater(rho0)
)

add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)

# setup particles' positions
let k = 1
    for i in 0:nfx-1, j in 0:nfy-1, m in 0:nfz-1
        fluid.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing, (m + 0.5) * dx_spacing)
        k += 1
    end
end
fill!(fluid.v, zero(SVector{3,Float64}))
fluid.rho .= rho0

update_state!(fluid, 1)

# ---------------------------------------------------------------------------
# Particle setup — boundary (6-faced box)
# ---------------------------------------------------------------------------

# Calculate particles per face to allocate accurately
# We use the same indexing logic as 2D to ensure coverage
n_bottom = (nbx + 2) * (nby + 2)
n_top    = (nbx + 2) * (nby + 2)
n_left   = nby * nbz
n_right  = nby * nbz
n_front  = nbx * nbz
n_back   = nbx * nbz

n_boundary = n_bottom + n_top + n_left + n_right + n_front + n_back

boundary = BasicParticleSystem(
    "boundary", n_boundary, 3, rho0 * dx_spacing^3, c_sound;
)

let k = 1
    # 1. Bottom floor (z = -0.5*dx)
    for i in -1:nbx, j in -1:nby
        boundary.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing, -0.5 * dx_spacing)
        k += 1
    end
    # 2. Ceiling (z = 40 + 0.5*dx)
    for i in -1:nbx, j in -1:nby
        boundary.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing, 40.0 + 0.5 * dx_spacing)
        k += 1
    end
    # 3. Left wall (x = -0.5*dx)
    for j in 0:nby-1, m in 0:nbz-1
        boundary.x[k] = SVector(-0.5 * dx_spacing, (j + 0.5) * dx_spacing, (m + 0.5) * dx_spacing)
        k += 1
    end
    # 4. Right wall (x = 75 + 0.5*dx)
    for j in 0:nby-1, m in 0:nbz-1
        boundary.x[k] = SVector(75.0 + 0.5 * dx_spacing, (j + 0.5) * dx_spacing, (m + 0.5) * dx_spacing)
        k += 1
    end
    # 5. Front wall (y = -0.5*dx)
    for i in 0:nbx-1, m in 0:nbz-1
        boundary.x[k] = SVector((i + 0.5) * dx_spacing, -0.5 * dx_spacing, (m + 0.5) * dx_spacing)
        k += 1
    end
    # 6. Back wall (y = 12.5 + 0.5*dx)
    for i in 0:nbx-1, m in 0:nbz-1
        boundary.x[k] = SVector((i + 0.5) * dx_spacing, 6.0 + 0.5 * dx_spacing, (m + 0.5) * dx_spacing)
        k += 1
    end
end

boundary.rho .= rho0
fill!(boundary.v, zero(SVector{3,Float64}))

# ---------------------------------------------------------------------------
# Interactions and integrator
# ---------------------------------------------------------------------------

kernel = CubicSplineKernel(h_sph; ndims=3)
static_boundary = StaticBoundarySystem(boundary, dx_spacing)

fluid_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid
)

fluid_boundary_interaction = SystemInteraction(
    kernel,
    FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
    fluid,
    static_boundary
)

integrator = LeapFrogTimeIntegrator(
    [fluid, boundary], 
    [fluid_interaction, fluid_boundary_interaction],
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@printf("n_fluid = %d  |  n_boundary = %d  |  mass = %.4g\n", n_fluid, n_boundary, fluid_mass)

run_driver!(
    [Stage(integrator, 1000, 0.05, "run")],
    1000,
    1000,
    "dambreak-3d-output/sph",
)
