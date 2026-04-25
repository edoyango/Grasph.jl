# DambreakWall.jl — 2D dambreak impacting an elastic concrete wall.
#
# Geometry (all dimensions in metres, dx = 0.5 m):
#   Water column:  25 m wide × 25 m tall (50 × 50 = 2500 particles)
#                  Initial position: x ∈ [0, 25], y ∈ [0, 25]
#   Concrete wall: x ∈ [60, 61.5]  (3 particles wide)
#                  y ∈ [0, 10]     (20 particles tall)
#                  60 particles total
#
# Boundaries (3-layer DynamicBoundarySystem):
#   Bottom: y < 0,  x ∈ [-1.5, 76.5] — normal (0,1),  point (0,0)
#   Left:   x < 0,  y ∈ [0,   40]    — normal (1,0),  point (0,0)
#   Right:  x > 75, y ∈ [0,   40]    — normal (-1,0), point (75,0)
#   Top:    y > 40, x ∈ [-1.5, 76.5] — normal (0,-1), point (0,40)
#
# Stage layout (2 stages):
#   Stage 1 — accumulate:
#     • ZeroFieldUpdater(:strain_rate, :vorticity) for wall
#     • StrainRateVorticityPfn for wall self-interaction
#   Stage 2 — forces:
#     • TaitEOSUpdater (fluid pressure from density)
#     • HookeLawStressUpdater (elastic stress update for wall)
#     • FluidPfn (fluid self + fluid–floor + fluid–left via DynamicBoundarySystem)
#     • CauchyFluidPfn (wall self)
#     • FluidSolidPfn (fluid–wall coupling, uses fluid pressure for solid side)

using Grasph
using StaticArrays
using HDF5

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx             = 0.5
const h_sph          = 1.2 * dx
const rho0_water     = 1000.0
const c_sound        = 10.0 * sqrt(2.0 * 9.81 * 25.0)   # ≈ 222 m/s
const art_visc_alpha = 0.02
const art_visc_beta  = 0.0
const gravity        = SVector(0.0, -9.81)

const rho0_wall  = 2400.0
const E_wall     = 5.0e8      # Young's modulus (Pa)
const nu_wall    = 0.2
const c_wall     = sqrt(E_wall * (1 - nu_wall) / (rho0_wall * (1 + nu_wall) * (1 - 2*nu_wall)))

const CFL_num    = 0.05

# ---------------------------------------------------------------------------
# Fluid (water column)
# ---------------------------------------------------------------------------

const nfx = 50   # 25 m / 0.5 m
const nfy = 50
const n_fluid    = nfx * nfy
const fluid_mass = rho0_water * dx^2

fluid = FluidParticleSystem(
    "fluid", n_fluid, 2, fluid_mass, c_sound;
    source_v    = gravity,
    state_updater = (nothing, TaitEOSUpdater(rho0_water)),
)

let k = 1
    for j in 0:nfy-1, i in 0:nfx-1
        fluid.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        k += 1
    end
end
fill!(fluid.v, zero(SVector{2,Float64}))
fluid.rho .= rho0_water
update_state!(fluid, 2)   # initialise pressure

# ---------------------------------------------------------------------------
# Concrete wall (elastic solid)
# ---------------------------------------------------------------------------

const n_wall_x    = 3    # 1.5 m thick
const n_wall_y    = 20   # 10 m tall
const n_wall      = n_wall_x * n_wall_y
const x_wall      = 60.0
const wall_mass   = rho0_wall * dx^2

wall = ElastoPlasticParticleSystem(
    "wall", n_wall, 2, 4, wall_mass, c_wall;
    source_v    = gravity,
    state_updater = (
        ZeroFieldUpdater(:strain_rate, :vorticity),
        HookeLawStressUpdater(E_wall, nu_wall),
    ),
)

let k = 1
    for j in 0:n_wall_y-1, i in 0:n_wall_x-1
        wall.x[k] = SVector(x_wall + (i + 0.5)*dx, (j + 0.5)*dx)
        k += 1
    end
end
fill!(wall.v, zero(SVector{2,Float64}))
wall.rho    .= rho0_wall
fill!(wall.stress, zero(SVector{4,Float64}))
wall.p      .= 0.0

# ---------------------------------------------------------------------------
# Boundary particles (3 layers each, DynamicBoundarySystem)
# ---------------------------------------------------------------------------

const n_bnd_layers = 3
const x_right      = 75.0
const y_top        = 40.0

# Floor  (y < 0): x ∈ [-1.25, 76.25]  →  156 particles/layer
const n_floor_x = 156   # -1.25 : 0.5 : 76.25
const n_floor   = n_floor_x * n_bnd_layers

floor_inner = BasicParticleSystem("floor", n_floor, 2, fluid_mass, c_sound)
let k = 1
    for layer in 0:n_bnd_layers-1, ix in 0:n_floor_x-1
        floor_inner.x[k] = SVector(-1.25 + ix * dx, -(layer + 0.5) * dx)
        k += 1
    end
end
floor_inner.rho .= rho0_water
fill!(floor_inner.v, zero(SVector{2,Float64}))
floor_dyn = DynamicBoundarySystem(floor_inner, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

# Left   (x < 0): y ∈ [0.25, 39.75]  →  80 particles/layer
const n_left_y = 80   # 0.25 : 0.5 : 39.75
const n_left   = n_left_y * n_bnd_layers

left_inner = BasicParticleSystem("left", n_left, 2, fluid_mass, c_sound)
let k = 1
    for layer in 0:n_bnd_layers-1, iy in 0:n_left_y-1
        left_inner.x[k] = SVector(-(layer + 0.5) * dx, (iy + 0.5) * dx)
        k += 1
    end
end
left_inner.rho .= rho0_water
fill!(left_inner.v, zero(SVector{2,Float64}))
left_dyn = DynamicBoundarySystem(left_inner, SVector(1.0, 0.0), SVector(0.0, 0.0), 3.0)

# Right  (x > 75): y ∈ [0.25, 39.75]  →  80 particles/layer
const n_right_y = 80   # 0.25 : 0.5 : 39.75
const n_right   = n_right_y * n_bnd_layers

right_inner = BasicParticleSystem("right", n_right, 2, fluid_mass, c_sound)
let k = 1
    for layer in 0:n_bnd_layers-1, iy in 0:n_right_y-1
        right_inner.x[k] = SVector(x_right + (layer + 0.5) * dx, (iy + 0.5) * dx)
        k += 1
    end
end
right_inner.rho .= rho0_water
fill!(right_inner.v, zero(SVector{2,Float64}))
right_dyn = DynamicBoundarySystem(right_inner, SVector(-1.0, 0.0), SVector(x_right, 0.0), 3.0)

# Top    (y > 40): x ∈ [-1.25, 76.25]  →  156 particles/layer
const n_top_x = 156   # -1.25 : 0.5 : 76.25
const n_top   = n_top_x * n_bnd_layers

top_inner = BasicParticleSystem("top", n_top, 2, fluid_mass, c_sound)
let k = 1
    for layer in 0:n_bnd_layers-1, ix in 0:n_top_x-1
        top_inner.x[k] = SVector(-1.25 + ix * dx, y_top + (layer + 0.5) * dx)
        k += 1
    end
end
top_inner.rho .= rho0_water
fill!(top_inner.v, zero(SVector{2,Float64}))
top_dyn = DynamicBoundarySystem(top_inner, SVector(0.0, -1.0), SVector(0.0, y_top), 3.0)

# ---------------------------------------------------------------------------
# Kernel and interactions
# ---------------------------------------------------------------------------

kernel = CubicSplineKernel(h_sph; ndims=2)

fluid_pfn       = FluidPfn(art_visc_alpha, art_visc_beta, h_sph)
cauchy_pfn      = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)
fluid_solid_pfn = FluidSolidPfn(art_visc_alpha, art_visc_beta, h_sph)
sr_pfn          = StrainRateVorticityPfn()

# (stage-1 pfn, stage-2 pfn) for each interaction
int_fluid_self  = SystemInteraction(kernel, (nothing, fluid_pfn), fluid)
int_fluid_floor = SystemInteraction(kernel, (nothing, fluid_pfn), fluid, floor_dyn)
int_fluid_left  = SystemInteraction(kernel, (nothing, fluid_pfn), fluid, left_dyn)
int_fluid_right = SystemInteraction(kernel, (nothing, fluid_pfn), fluid, right_dyn)
int_fluid_top   = SystemInteraction(kernel, (nothing, fluid_pfn), fluid, top_dyn)
int_wall_self   = SystemInteraction(kernel, (sr_pfn, cauchy_pfn), wall)
int_wall_floor  = SystemInteraction(kernel, (sr_pfn, cauchy_pfn), wall, floor_dyn)
int_fluid_wall  = SystemInteraction(kernel, (nothing, fluid_solid_pfn), fluid, wall)

# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

integrator = LeapFrogTimeIntegrator(
    [fluid, wall, floor_inner, left_inner, right_inner, top_inner],
    [int_fluid_self, int_wall_self, int_fluid_wall, int_fluid_floor, int_fluid_left, int_fluid_right, int_fluid_top, int_wall_floor],
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

println("n_fluid=$(n_fluid)  n_wall=$(n_wall)  n_floor=$(n_floor)  n_left=$(n_left)  n_right=$(n_right)  n_top=$(n_top)")
println("c_sound=$(round(c_sound; digits=2)) m/s  c_wall=$(round(c_wall; digits=2)) m/s")

stages = [
    Stage(integrator, 1000000, CFL_num, "run"),
]

run_driver!(
    stages,
    1000,
    1000,
    "dambreak-wall-output/sph";
    interactive = false,
)
