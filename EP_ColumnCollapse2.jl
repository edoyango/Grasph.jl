# EP_ColumnCollapse2.jl — EP column collapse using VirtualParticleSystem boundaries.
#
# Replaces the DynamicBoundarySystem bottom and GhostParticleSystem left wall
# with two VirtualParticleSystems that accumulate fluid fields via SPH
# interpolation and apply velocity boundary conditions via VirtualNormUpdater.
#
# Stage layout (4 stages):
#   1. InterpolateFieldFn(:v, :rho)  → virtuals accumulate v, rho, w_sum
#   2. VirtualNormUpdater (v, rho)   → normalize + apply v_mult on v
#      ZeroFieldUpdater(:strain_rate, :vorticity) for fluid
#      StrainRateVorticityPfn sweep
#   3. ElastoPlasticStressUpdater for fluid
#      InterpolateFieldFn(:stress; accumulate_wsum=false) → virtuals accumulate stress
#   4. VirtualNormUpdater (:stress)  → normalize stress
#      CauchyFluidPfn sweep

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx_spacing          = 0.002
const h_sph               = 1.2 * dx_spacing
const rho0                = 1850.0
const art_visc_alpha      = 0.1
const art_visc_beta       = 0.1
const soil_friction_angle = 19.8 * π / 180.0

const E        = 0.84e6
const nu       = 0.3
const psi      = 0.0
const cohesion = 0.0
const c_sound  = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2nu)))
const dt_ep    = 0.1 * h_sph / c_sound

const nfx = Int(floor(0.2 / dx_spacing))
const nfy = Int(floor(0.1 / dx_spacing))
const nbx = Int(floor(0.5 / dx_spacing))

# ---------------------------------------------------------------------------
# Fluid
# ---------------------------------------------------------------------------

n_fluid    = nfx * nfy
fluid_mass = rho0 * dx_spacing * dx_spacing

fluid = ElastoPlasticParticleSystem(
    "fluid", n_fluid, 2, 4, fluid_mass, c_sound;
    source_v    = [0.0, -9.81],
    state_updater = (
        nothing,                                                         # stage 1: no update
        ZeroFieldUpdater(:strain_rate, :vorticity),                      # stage 2: zero before strain sweep
        ElastoPlasticStressUpdater(E, nu, soil_friction_angle, psi, cohesion, dt_ep),  # stage 3: update stress
        nothing,                                                         # stage 4: no update
    ),
)

add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)
add_print_field!(fluid, :stress)
add_print_field!(fluid, :vorticity)
add_print_field!(fluid, :strain)
add_print_field!(fluid, :strain_p)

let k = 1
    for i in 0:nfx-1, j in 0:nfy-1
        fluid.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
        k += 1
    end
end
fill!(fluid.v, zero(SVector{2,Float64}))
fluid.rho .= rho0
update_state!(fluid, 3)  # initialize stress via ElastoPlasticStressUpdater (stage 3)

# ---------------------------------------------------------------------------
# Bottom virtual boundary — no-slip (v_mult = [-1, -1])
# ---------------------------------------------------------------------------

n_bottom     = 3 * (nbx + 3)
bottom_source = StressParticleSystem(
    "bottom_boundary", n_bottom, 2, 4, fluid_mass, c_sound,
)
let k = 1
    for i in 1:nbx+3, j in 1:3
        bottom_source.x[k] = SVector((i - 3.5) * dx_spacing, -(j - 0.5) * dx_spacing)
        k += 1
    end
end
bottom_source.rho .= rho0
fill!(bottom_source.v, zero(SVector{2,Float64}))

bottom_virt = VirtualParticleSystem(
    bottom_source, "bottom_virt", n_bottom, 2, fluid_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    state_updater = (
        nothing,                                                          # stage 1: no update
        VirtualNormUpdater(SVector(-1.0, -1.0), :v, :rho),               # stage 2: normalize v+rho, flip both
        nothing,                                                          # stage 3: no update
        VirtualNormUpdater(SVector(-1.0, -1.0), :stress),                 # stage 4: normalize stress
    ),
)

# ---------------------------------------------------------------------------
# Left virtual boundary — free-slip (v_mult = [-1, 1], negate x only)
# ---------------------------------------------------------------------------

n_left      = 3 * nfy
left_source = StressParticleSystem(
    "left_boundary", n_left, 2, 4, fluid_mass, c_sound,
)
let k = 1
    for j in 0:nfy-1, col in 1:3
        left_source.x[k] = SVector(-(col - 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
        k += 1
    end
end
left_source.rho .= rho0
fill!(left_source.v, zero(SVector{2,Float64}))

left_virt = VirtualParticleSystem(
    left_source, "left_virt", n_left, 2, fluid_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    state_updater = (
        nothing,
        VirtualNormUpdater(SVector(-1.0, 1.0), :v, :rho),                # stage 2: free-slip (negate x)
        nothing,
        VirtualNormUpdater(SVector(-1.0, 1.0), :stress),                  # stage 4: normalize stress
    ),
)

# ---------------------------------------------------------------------------
# Interactions (4 pfns each)
# ---------------------------------------------------------------------------

kernel     = CubicSplineKernel(h_sph; ndims=2)
sr_pfn     = StrainRateVorticityPfn()
kin_pfn    = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)
interp_vel = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
interp_str = InterpolateFieldFn(:stress; accumulate_wsum=false)

# fluid ↔ fluid
fluid_self = SystemInteraction(kernel, (nothing, sr_pfn, nothing, kin_pfn), fluid)

# fluid ↔ bottom_virt: interpolate (stages 1,3) + strain/kinematics (stages 2,4)
fluid_bottom = SystemInteraction(kernel, (interp_vel, sr_pfn, interp_str, kin_pfn), fluid, bottom_virt)

# fluid ↔ left_virt: interpolate (stages 1,3) + strain/kinematics (stages 2,4)
fluid_left = SystemInteraction(kernel, (interp_vel, sr_pfn, interp_str, kin_pfn), fluid, left_virt)

# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

integrator = LeapFrogTimeIntegrator(
    [fluid],
    [fluid_self, fluid_bottom, fluid_left];
    virtual_systems = (bottom_virt, left_virt),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_fluid = $n_fluid  |  mass = $fluid_mass")

run_driver!(
    integrator,
    50000,
    500,
    500,
    0.1,
    "ep-gcc-output2/sph"
)
