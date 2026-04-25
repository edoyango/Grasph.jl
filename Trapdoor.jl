# Trapdoor.jl — 2D trapdoor problem using VirtualParticleSystem boundaries.
#
# Geometry (all dimensions in metres):
#   Soil:      264 × 100 particles, dx = 0.05 m  (13.2 m wide × 5.0 m deep)
#   Left/right walls: GhostParticleSystem (free-slip, stress copied per stage)
#   Static bottom (left + right of trapdoor): VirtualParticleSystem, 0.5 m thick
#   Trapdoor (centre 2.0 m, 40 columns):      VirtualParticleSystem, 0.5 m thick
#
# Two-phase run:
#   Phase 1 — gravity settling: trapdoor static (prescribed_v = 0)
#   Phase 2 — trapdoor motion:  prescribed_v = (0, 0.005) m/s upward
#
# Stage layout (4 stages):
#   1. InterpolateFieldFn(:v, :rho)          → virtuals accumulate v, rho, w_sum
#   2. VirtualNormUpdater(:v, :rho)          → normalize rho
#      ZeroFieldUpdater(:strain_rate, :vorticity) for soil
#      StrainRateVorticityPfn sweep
#   3. ElastoPlasticStressUpdater for soil
#      InterpolateFieldFn(:stress)           → virtuals accumulate stress
#      GhostCopier(:stress)                  → ghosts get updated stress
#   4. VirtualNormUpdater(:stress)           → normalize stress
#      CauchyFluidPfn sweep

using Grasph
using StaticArrays
using HDF5

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx             = 0.05
const h_sph          = 1.2 * dx
const rho0           = 1600.0
const art_visc_alpha = 0.1
const art_visc_beta  = 0.0

const E        = 10.0e6
const nu       = 0.33
const phi      = 39.0 * π / 180.0
const psi      = 19.0 * π / 180.0
const cohesion = 0.0

const c_sound = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2*nu)))

const nx       = 264   # soil columns (13.2 m)
const ny       = 100   # soil rows    ( 5.0 m)
const n_layers = 10    # source layers below y = 0 (0.5 m thick)

# Trapdoor: 2.0 m wide = 40 columns, centred in the domain.
const ntd_x   = 40
const nleft_x = (nx - ntd_x) ÷ 2   # 112 columns to the left of trapdoor
const trapdoor_vel = -0.005

# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

n_soil    = nx * ny
soil_mass = rho0 * dx * dx

soil = ElastoPlasticParticleSystem(
    "soil", n_soil, 2, 4, soil_mass, c_sound;
    source_v    = [0.0, -9.81],
    state_updater = (
        nothing,
        ZeroFieldUpdater(:strain_rate, :vorticity),
        ElastoPlasticStressUpdater(E, nu, phi, psi, cohesion),
        nothing,
    ),
)

add_print_field!(soil, :v)
add_print_field!(soil, :rho)
add_print_field!(soil, :stress)
add_print_field!(soil, :strain)

let k = 1
    for i in 0:nx-1, j in 0:ny-1
        soil.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        k += 1
    end
end
fill!(soil.v, zero(SVector{2,Float64}))
soil.rho .= rho0
update_state!(soil, 3)

# ---------------------------------------------------------------------------
# Static bottom boundary (left + right flanks, outside trapdoor)
# ---------------------------------------------------------------------------

n_bottom = (nx - ntd_x + 10) * n_layers

bottom_source = StressParticleSystem(
    "bottom_source", n_bottom, 2, 4, soil_mass, c_sound,
)
let k = 1
    for j in 1:n_layers
        for i in -5:nleft_x-1                      # left flank
            bottom_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
            k += 1
        end
        for i in nleft_x+ntd_x:nx+5-1               # right flank
            bottom_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
            k += 1
        end
    end
end
bottom_source.rho .= rho0
fill!(bottom_source.v, zero(SVector{2,Float64}))

_trapdoor_updater = (
    nothing,
    (VirtualNormUpdater(SVector(0.0, 0.0), :rho), PrescribedVelocityUpdater()),
    nothing,
    VirtualNormUpdater(SVector(0.0, 0.0), :stress),
)

bottom_virt = VirtualParticleSystem(
    bottom_source, "bottom_virt", n_bottom, 2, soil_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    state_updater = _trapdoor_updater,
)

# ---------------------------------------------------------------------------
# Trapdoor boundary (centre 2.0 m)
# Two VirtualParticleSystems share the same source:
#   trapdoor_static_virt — prescribed_v = (0, 0)      phase 1: gravity settling
#   trapdoor_moving_virt — prescribed_v = (0, 0.005)  phase 2: trapdoor motion
# ---------------------------------------------------------------------------

n_trapdoor = ntd_x * n_layers

trapdoor_source = StressParticleSystem(
    "trapdoor_source", n_trapdoor, 2, 4, soil_mass, c_sound,
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

trapdoor_static_virt = VirtualParticleSystem(
    trapdoor_source, "trapdoor_static_virt", n_trapdoor, 2, soil_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    state_updater = _trapdoor_updater,
)

trapdoor_moving_virt = VirtualParticleSystem(
    trapdoor_source, "trapdoor_moving_virt", n_trapdoor, 2, soil_mass, c_sound;
    zero_fields   = (:v, :rho, :stress),
    prescribed_v  = SVector(0.0, trapdoor_vel),
    state_updater = _trapdoor_updater,
)

# ---------------------------------------------------------------------------
# Left + right ghost walls (free-slip: x component negated by generate_ghosts!)
# ---------------------------------------------------------------------------

walls_ghost = GhostParticleSystem(soil,
    nothing,
    nothing,
    GhostCopier(:stress => HouseholderReflect()),   # stage 3: copy updated stress ready for virtual stress accum and kin sweep
    nothing,
)

walls_entry = GhostEntry(
    walls_ghost, 3.0 * h_sph,
    (SVector(1.0,  0.0), SVector(0.0, 0.0)),   # left  wall at x = 0
    (SVector(-1.0, 0.0), SVector(Float64(nx * dx), 0.0)),  # right wall at x = nx*dx
)

# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

kernel     = CubicSplineKernel(h_sph; ndims=2)
sr_pfn     = StrainRateVorticityPfn()
kin_pfn    = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)
interp_rho = InterpolateFieldFn(:rho; accumulate_wsum=true)
interp_str = InterpolateFieldFn(:stress; accumulate_wsum=false)

soil_self            = SystemInteraction(kernel, (nothing,    sr_pfn,  nothing,    kin_pfn), soil)
soil_bottom          = SystemInteraction(kernel, (interp_rho, sr_pfn,  interp_str, kin_pfn), soil, bottom_virt)
ghost_bottom         = SystemInteraction(kernel, (interp_rho, nothing, interp_str, nothing), walls_ghost, bottom_virt)
soil_trapdoor_static = SystemInteraction(kernel, (interp_rho, sr_pfn,  interp_str, kin_pfn), soil, trapdoor_static_virt)
soil_trapdoor_moving = SystemInteraction(kernel, (interp_rho, sr_pfn,  interp_str, kin_pfn), soil, trapdoor_moving_virt)
soil_walls           = SystemInteraction(kernel, (nothing,    sr_pfn,  nothing,    kin_pfn), soil, walls_ghost)

# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

integrator_static = LeapFrogTimeIntegrator(
    [soil],
    [soil_self, soil_bottom, ghost_bottom, soil_trapdoor_static, soil_walls];
    ghosts          = (walls_entry,),
    virtual_systems = (bottom_virt, trapdoor_static_virt),
    Γ               = 0.002,
)

integrator_moving = LeapFrogTimeIntegrator(
    [soil],
    [soil_self, soil_bottom, ghost_bottom, soil_trapdoor_moving, soil_walls];
    ghosts          = (walls_entry,),
    virtual_systems = (bottom_virt, trapdoor_moving_virt),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_soil=$n_soil  n_bottom=$n_bottom  n_trapdoor=$n_trapdoor")
println("c_sound=$(round(c_sound; digits=2)) m/s")

stages = [
    Stage(integrator_static, 20000,  0.1, "damping"),
    Stage(integrator_moving, 500000, 0.1, "moving"),
]

run_driver!(
    stages,
    2000,  # print_interval_step
    2000,  # save_interval_step
    "trapdoor-output/sph";
    interactive = false,
)
