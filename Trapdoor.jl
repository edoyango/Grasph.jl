# Trapdoor.jl — 2D trapdoor problem using VirtualParticleSystem boundaries.
#
# Geometry (all dimensions in metres):
#   Soil:      264 × 100 particles, dx = 0.05 m  (13.2 m wide × 5.0 m deep)
#   Left/right walls: GhostParticleSystem (free-slip, stress copied per stage)
#   Static bottom (left + right of trapdoor): VirtualParticleSystem, 0.5 m thick
#   Trapdoor (centre 2.0 m, 40 columns):      VirtualParticleSystem, 0.5 m thick
#                                              prescribed_v = (0, 0.05) m/s upward
#
# Stage layout (4 stages):
#   1. InterpolateFieldFn(:v, :rho)          → virtuals accumulate v, rho, w_sum
#      GhostCopier(:stress)                  → ghosts get current stress
#   2. VirtualNormUpdater(:v, :rho)          → normalize + v_mult + prescribed_v
#      ZeroFieldUpdater(:strain_rate, :vorticity) for fluid
#      StrainRateVorticityPfn sweep
#   3. ElastoPlasticStressUpdater for fluid
#      InterpolateFieldFn(:stress)           → virtuals accumulate stress
#      GhostCopier(:stress)                  → ghosts get updated stress
#   4. VirtualNormUpdater(:stress)           → normalize stress
#      CauchyFluidPfn sweep

using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx             = 0.05
const h_sph          = 1.2 * dx
const rho0           = 1600.0
const art_visc_alpha = 0.1
const art_visc_beta  = 0.1

const E        = 10.0e6
const nu       = 0.33
const phi      = 39.0 * π / 180.0
const psi      = 19.0 * π / 180.0
const cohesion = 0.0

const c_sound = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2*nu)))
const dt_ep   = 0.1 * h_sph / c_sound

const nx       = 264   # fluid columns (13.2 m)
const ny       = 100   # fluid rows    ( 5.0 m)
const n_layers = 10    # source layers below y = 0 (0.5 m thick)

# Trapdoor: 2.0 m wide = 40 columns, centred in the domain.
const ntd_x   = 40
const nleft_x = (nx - ntd_x) ÷ 2   # 112 columns to the left of trapdoor

# ---------------------------------------------------------------------------
# Fluid
# ---------------------------------------------------------------------------

n_fluid    = nx * ny
fluid_mass = rho0 * dx * dx

fluid = ElastoPlasticParticleSystem(
    "fluid", n_fluid, 2, 4, fluid_mass, c_sound;
    source_v    = [0.0, -9.81],
    state_updater = (
        nothing,
        ZeroFieldUpdater(:strain_rate, :vorticity),
        ElastoPlasticStressUpdater(E, nu, phi, psi, cohesion, dt_ep),
        nothing,
    ),
)

add_print_field!(fluid, :v)
add_print_field!(fluid, :rho)
add_print_field!(fluid, :stress)
add_print_field!(fluid, :vorticity)
add_print_field!(fluid, :strain)
add_print_field!(fluid, :strain_p)

let k = 1
    for i in 0:nx-1, j in 0:ny-1
        fluid.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        k += 1
    end
end
fill!(fluid.v, zero(SVector{2,Float64}))
fluid.rho .= rho0
update_state!(fluid, 3)

# ---------------------------------------------------------------------------
# Static bottom boundary (left + right flanks, outside trapdoor)
# ---------------------------------------------------------------------------

n_bottom = (nx - ntd_x) * n_layers

bottom_source = StressParticleSystem(
    "bottom_source", n_bottom, 2, 4, fluid_mass, c_sound,
)
let k = 1
    for j in 1:n_layers
        for i in 0:nleft_x-1                      # left flank
            bottom_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
            k += 1
        end
        for i in nleft_x+ntd_x:nx-1               # right flank
            bottom_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
            k += 1
        end
    end
end
bottom_source.rho .= rho0
fill!(bottom_source.v, zero(SVector{2,Float64}))

bottom_virt = VirtualParticleSystem(
    bottom_source, "bottom_virt", n_bottom, 2, fluid_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    state_updater = (
        nothing,
        VirtualNormUpdater(SVector(-1.0, -1.0), :v, :rho),
        nothing,
        VirtualNormUpdater(SVector(-1.0, -1.0), :stress),
    ),
)

# ---------------------------------------------------------------------------
# Trapdoor boundary (centre 2.0 m, prescribed_v = (0, 0.05) m/s)
# ---------------------------------------------------------------------------

n_trapdoor = ntd_x * n_layers

trapdoor_source = StressParticleSystem(
    "trapdoor_source", n_trapdoor, 2, 4, fluid_mass, c_sound,
)
let k = 1
    for j in 1:n_layers
        for i in nleft_x:nleft_x+ntd_x-1
            trapdoor_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
            k += 1
        end
    end
end
trapdoor_source.rho .= rho0
fill!(trapdoor_source.v, zero(SVector{2,Float64}))

trapdoor_virt = VirtualParticleSystem(
    trapdoor_source, "trapdoor_virt", n_trapdoor, 2, fluid_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    prescribed_v  = SVector(0.0, 0.05),
    state_updater = (
        nothing,
        VirtualNormUpdater(SVector(-1.0, -1.0), :v, :rho),
        nothing,
        VirtualNormUpdater(SVector(-1.0, -1.0), :stress),
    ),
)

# ---------------------------------------------------------------------------
# Left + right ghost walls (free-slip: x component negated by generate_ghosts!)
# ---------------------------------------------------------------------------

walls_ghost = GhostParticleSystem(fluid,
    GhostCopier(:stress),   # stage 1: copy stress ready for stage-2 strain sweep
    nothing,
    GhostCopier(:stress),   # stage 3: copy updated stress ready for stage-4 kin sweep
    nothing,
)

walls_entry = GhostEntry(
    walls_ghost, 3.0 * h_sph,
    (SVector(1.0,  0.0), SVector(0.0,         0.0)),   # left  wall at x = 0
    (SVector(-1.0, 0.0), SVector(Float64(nx * dx), 0.0)),  # right wall at x = nx*dx
)

# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

kernel     = CubicSplineKernel(h_sph; ndims=2)
sr_pfn     = StrainRateVorticityPfn()
kin_pfn    = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)
interp_vel = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
interp_str = InterpolateFieldFn(:stress; accumulate_wsum=false)

fluid_self     = SystemInteraction(kernel, (nothing, sr_pfn, nothing, kin_pfn), fluid)
fluid_bottom   = SystemInteraction(kernel, (interp_vel, sr_pfn, interp_str, kin_pfn), fluid, bottom_virt)
fluid_trapdoor = SystemInteraction(kernel, (interp_vel, sr_pfn, interp_str, kin_pfn), fluid, trapdoor_virt)
fluid_walls    = SystemInteraction(kernel, (nothing, sr_pfn, nothing, kin_pfn), fluid, walls_ghost)

# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

integrator = LeapFrogTimeIntegrator(
    [fluid],
    [fluid_self, fluid_bottom, fluid_trapdoor, fluid_walls];
    ghosts          = (walls_entry,),
    virtual_systems = (bottom_virt, trapdoor_virt),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_fluid=$n_fluid  n_bottom=$n_bottom  n_trapdoor=$n_trapdoor")
println("c_sound=$(round(c_sound; digits=2)) m/s  dt_ep=$(round(dt_ep; sigdigits=4)) s")

run_driver!(
    integrator,
    50000,
    1000,
    1000,
    0.1,
    "trapdoor-output/sph"
)
