# CantileverBeam.jl — 2D cantilevered elastic beam under gravity.
#
# Geometry (all dimensions in metres, dx = 0.05 m):
#   Beam:  2.0 m × 0.4 m, 40 × 8 = 320 particles
#          x ∈ [0, 2.0], y ∈ [0, 0.4]
#
# Fixed end boundary (x = 0):
#   DynamicBoundarySystem, 3 layers, normal (1, 0), point (0, 0)
#   particles at x ∈ [-0.15, -0.05], y ∈ [-0.025, 0.425]
#
# Stage layout (2 stages):
#   Stage 1 — strain accumulation:
#     • ZeroFieldUpdater(:strain_rate, :vorticity) for beam
#     • StrainRateVorticityPfn sweep (beam self + beam–fix)
#   Stage 2 — stress update + dynamics:
#     • HookeLawStressUpdater for beam
#     • CauchyFluidPfn sweep (beam self + beam–fix)
#
# Euler-Bernoulli analytical tip deflection (small deflection):
#   δ = q L⁴ / (8 E I) = ρg H L⁴ × 12 / (8 E H³) = 3 ρg L⁴ / (2 E H²)

using Grasph
using StaticArrays
using HDF5

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

const dx             = 0.02
const h_sph          = 1.2 * dx
const rho0           = 1000.0
const E              = 1.0e9
const nu             = 0.3
const c_beam         = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2*nu)))
const art_visc_alpha = 0.1
const art_visc_beta  = 0.0
const gravity        = SVector(0.0, -9.81)

const CFL_num        = 0.1

# Beam geometry
const n_beam_x = 250    # 2.0 m / 0.05 m
const n_beam_y =  20    # 0.4 m / 0.05 m
const L        = n_beam_x * dx   # 2.0 m
const H        = n_beam_y * dx   # 0.4 m

# ---------------------------------------------------------------------------
# Beam (linear-elastic solid)
# ---------------------------------------------------------------------------

n_beam    = n_beam_x * n_beam_y
beam_mass = rho0 * dx^2

beam = ElastoPlasticParticleSystem(
    "beam", n_beam, 2, 4, beam_mass, c_beam;
    source_v    = gravity,
    state_updater = (
        ZeroFieldUpdater(:strain_rate, :vorticity),
        HookeLawStressUpdater(E, nu),
    ),
)

add_print_field!(beam, :v)
add_print_field!(beam, :stress)
add_print_field!(beam, :strain)

let k = 1
    for j in 0:n_beam_y-1, i in 0:n_beam_x-1
        beam.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        k += 1
    end
end
fill!(beam.v,      zero(SVector{2,Float64}))
fill!(beam.stress, zero(SVector{4,Float64}))
beam.rho .= rho0
beam.p   .= 0.0

# ---------------------------------------------------------------------------
# Fixed-end boundary (DynamicBoundarySystem at x = 0)
# ---------------------------------------------------------------------------

const n_bnd_layers = 3
const n_fix        = n_bnd_layers * (n_beam_y + 2*n_bnd_layers)   # 1-row margin above/below beam

fix_inner = BasicParticleSystem("fix", n_fix, 2, beam_mass, c_beam)
let k = 1
    for layer in 1:n_bnd_layers, iy in -n_bnd_layers:n_beam_y+n_bnd_layers-1
        fix_inner.x[k] = SVector(-(layer - 0.5) * dx, (iy + 0.5) * dx)
        k += 1
    end
end
fix_inner.rho .= rho0
fill!(fix_inner.v, zero(SVector{2,Float64}))

# normal (1,0) points right (into the beam domain); point on the plane x = 0
fix_dyn = DynamicBoundarySystem(fix_inner, SVector(1.0, 0.0), SVector(0.0, 0.0), 3.0)

# ---------------------------------------------------------------------------
# Kernel and pairwise functors
# ---------------------------------------------------------------------------

kernel     = CubicSplineKernel(h_sph; ndims=2)
sr_pfn     = StrainRateVorticityPfn()
cauchy_pfn = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)

# ---------------------------------------------------------------------------
# Neighbor-count probe (mirrors all beam particle positions)
# ---------------------------------------------------------------------------

beam_probe = ProbeParticleSystem(
    "beam_probe", beam;
    extras = (nbr_count = zeros(Int, n_beam),),
)

# ---------------------------------------------------------------------------
# Backend selection
#
# Defaults to CPU (Vector-backed, coloured sweep) so the script is unchanged
# in normal use. Set GRASPH_BACKEND=cuda to run GPU-resident via
# KernelAbstractions.jl: adapts every system to CuArray and switches every
# interaction to the one-sided KA sweep (the only sweep implemented as a KA
# kernel — see docs/gpu-migration-plan.md). Requires CUDA.jl in the active
# environment (it is not a hard dependency of Grasph itself).
# ---------------------------------------------------------------------------

const GRASPH_BACKEND = get(ENV, "GRASPH_BACKEND", "cpu")
const ka_mode = GRASPH_BACKEND == "cuda"

if ka_mode
    using CUDA
    using Adapt
    # beam_probe is self-referencing (beam_probe.mirror_target === beam);
    # adapt the probe as one unit and pull the canonical GPU-resident beam
    # back out of it (see ProbeParticleSystem's docstring) — adapting beam
    # separately would create two independent, non-aliased GPU copies.
    beam_probe = adapt(CUDABackend(), beam_probe)
    beam       = getfield(beam_probe, :mirror_target)
    fix_inner  = adapt(CUDABackend(), fix_inner)
    fix_dyn    = DynamicBoundarySystem(fix_inner, SVector(1.0, 0.0), SVector(0.0, 0.0), 3.0)
end

# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

beam_self = SystemInteraction(kernel, (sr_pfn, cauchy_pfn), beam; onesided = ka_mode, ka = ka_mode)
beam_fix  = SystemInteraction(kernel, (sr_pfn, cauchy_pfn), beam, fix_dyn; onesided = ka_mode, ka = ka_mode)
probe_nbr = SystemInteraction(kernel, NeighborCountFn(:nbr_count), beam, beam_probe; onesided = ka_mode, ka = ka_mode)

# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

integrator = LeapFrogTimeIntegrator(
    [beam, fix_inner],
    [beam_self, beam_fix];
    probes             = (beam_probe,),
    probe_interactions = (probe_nbr,),
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

δ_eb = 3 * rho0 * 9.81 * L^4 / (2 * E * H^2)
println("n_beam=$n_beam  n_fix=$n_fix  c_beam=$(round(c_beam; digits=2)) m/s  |  backend=$GRASPH_BACKEND")
println("Euler-Bernoulli tip deflection ≈ $(round(δ_eb; digits=4)) m")

stages = [
    Stage(integrator, 2000000, CFL_num, "run"),
]

run_driver!(
    stages,
    2000,
    2000,
    "cantilever-output/sph";
    interactive = false,
)
