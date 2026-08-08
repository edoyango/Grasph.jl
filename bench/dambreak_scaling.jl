# bench/dambreak_scaling.jl — CPU-vs-GPU crossover benchmark for dambreak.jl's
# shape.
#
# Scales the DOMAIN, not `dx`: the smoothing length `h` is baked into
# `CubicSplineKernel`'s type parameter, so varying resolution would
# recompile the entire sweep/integrator specialisation at every size point
# and would also change `dt` (= CFL*h/c), confounding "bigger problem" with
# "different timestep". Instead each size point keeps `dx`/`h` fixed and
# scales the fluid block + boundary box together, preserving dambreak.jl's
# aspect ratio and ~18-neighbour count so ns/particle/step is comparable
# across sizes.
#
# Reports THREE columns: CPU-coloured (today's production default for every
# script), CPU-onesided (the same full-stencil algorithm the GPU runs, on
# CPU), and GPU. The middle column is not optional — without it, a GPU
# "speedup" over the coloured sweep can't be told apart from an artifact of
# comparing against a half-shell algorithm that does half the pair
# evaluations per step.
#
# Prediction, from measurements recorded in docs/gpu-migration-plan.md: this
# GPU (RTX 4060 Laptop, sm_89) has NO Float64 compute advantage over the
# 16-core CPU (0.138 vs 0.143-0.311 TFLOP/s) and pays a flat ~8.3us/kernel-
# launch overhead. At dambreak.jl's actual scale (2,500 particles, nfx=50)
# the GPU should LOSE outright — the floor from ~20-25 launches/step is
# roughly 200-650us/step regardless of particle count. If there's a
# crossover on this hardware at all, launch-count amortisation (not FP64
# throughput) is what drives it, and it should sit well above dambreak's own
# scale. Real speedup validation belongs on datacenter (A100/H100-class)
# hardware — this benchmark's job is to locate the crossover on whatever
# machine it's run on, not to prove GPU is faster.
#
# Usage:
#   julia --project bench/dambreak_scaling.jl
#   julia --project bench/dambreak_scaling.jl --sizes 50,100,200 --budget 5e6
# (CUDA must be in the active environment; falls back to CPU-only columns
# with a warning if CUDA.functional() is false.)

using Grasph
using StaticArrays
using Printf
using Dates

const HAVE_CUDA = try
    @eval using CUDA, Adapt
    CUDA.functional()
catch err
    @warn "CUDA not available for this benchmark; GPU column will be skipped" exception=err
    false
end

# nfx values: n_fluid = nfx^2. Default set is small enough to finish in a few
# minutes including JIT; pass --sizes to extend toward the ~1M-particle range
# the migration plan's benchmark step calls for (nfx up to ~1000).
const DEFAULT_SIZES = [50, 100, 200, 320, 450]

# Total pair-evaluations-ish budget controlling how many steps are timed per
# size point (bigger problems get fewer steps) — keeps each point to roughly
# the same wall-clock share regardless of size. Override with --budget.
const DEFAULT_BUDGET = 2e7

function _parse_args(args)
    sizes = DEFAULT_SIZES
    budget = DEFAULT_BUDGET
    i = 1
    while i <= length(args)
        if args[i] == "--sizes" && i < length(args)
            sizes = parse.(Int, split(args[i+1], ","))
            i += 2
        elseif args[i] == "--budget" && i < length(args)
            budget = parse(Float64, args[i+1])
            i += 2
        else
            i += 1
        end
    end
    return sizes, budget
end

function build(nfx; onesided=false, ka=false, backend=nothing)
    dx_spacing = 0.5
    h_sph = 1.2 * dx_spacing
    rho0 = 1000.0
    c_sound = 10.0 * sqrt(2.0 * 9.81 * 25.0)
    art_visc_alpha = 0.01
    art_visc_beta = 0.0

    nfy = nfx
    # Preserves dambreak.jl's 75:25 / 40:25 box-to-block aspect ratio.
    box_w = 3.0 * nfx * dx_spacing
    box_h = 1.6 * nfx * dx_spacing
    nbx = max(Int(floor(box_w / dx_spacing)), nfx + 4)
    nby = max(Int(floor(box_h / dx_spacing)), nfx + 4)

    n_fluid = nfx * nfy
    fluid_mass = rho0 * dx_spacing * dx_spacing

    fluid = FluidParticleSystem("fluid", n_fluid, 2, fluid_mass, c_sound;
                                source_v = [0.0, -9.81], state_updater = TaitEOSUpdater(rho0))
    let k = 1
        for i in 0:nfx-1, j in 0:nfy-1
            fluid.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
            k += 1
        end
    end
    fill!(fluid.v, zero(SVector{2,Float64}))
    fluid.rho .= rho0
    update_state!(fluid, 1)

    n_boundary = 2 * (nbx + nby) + 4
    boundary = BasicParticleSystem("boundary", n_boundary, 2, fluid_mass, c_sound)
    let k = 1
        for i in -1:nbx
            boundary.x[k] = SVector((i + 0.5) * dx_spacing, -0.5 * dx_spacing); k += 1
        end
        for i in -1:nbx
            boundary.x[k] = SVector((i + 0.5) * dx_spacing, nby * dx_spacing + 0.5 * dx_spacing); k += 1
        end
        for j in 0:nby-1
            boundary.x[k] = SVector(-0.5 * dx_spacing, (j + 0.5) * dx_spacing); k += 1
        end
        for j in 0:nby-1
            boundary.x[k] = SVector(nbx * dx_spacing + 0.5 * dx_spacing, (j + 0.5) * dx_spacing); k += 1
        end
    end
    boundary.rho .= rho0
    fill!(boundary.v, zero(SVector{2,Float64}))

    kernel = CubicSplineKernel(h_sph; ndims=2)

    if backend !== nothing
        fluid    = adapt(backend, fluid)
        boundary = adapt(backend, boundary)
    end

    static_boundary = StaticBoundarySystem(boundary, dx_spacing)
    fi = SystemInteraction(kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid;
                           onesided = onesided, ka = ka)
    fbi = SystemInteraction(kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid, static_boundary;
                            onesided = onesided, ka = ka)
    integrator = LeapFrogTimeIntegrator([fluid, boundary], [fi, fbi])
    return integrator, n_fluid, n_boundary
end

# Timed via wall clock around the real time_integrate! entry point — not
# synthetic per-kernel timing — with an explicit warmup pass first (h being a
# type parameter means the first call compiles the whole sweep/integrator
# specialisation; the migration plan flags this explicitly). GPU callers must
# synchronize before AND after the timed region, or this measures the launch
# queue instead of actual execution.
function us_per_step(integrator, nsteps; sync = nothing)
    time_integrate!(integrator, max(2, nsteps ÷ 10), 10^9, 10^9, 0.05, nothing; print_timer=false)
    sync !== nothing && sync()
    t0 = time_ns()
    time_integrate!(integrator, nsteps, 10^9, 10^9, 0.05, nothing; print_timer=false)
    sync !== nothing && sync()
    t1 = time_ns()
    return (t1 - t0) / 1e3 / nsteps
end

function main()
    sizes, budget = _parse_args(ARGS)
    println("=== dambreak.jl scaling benchmark — ", Dates.now(), " ===")
    println("CUDA available: ", HAVE_CUDA)
    println()
    @printf("%6s %10s %10s %8s | %12s %12s %12s | %9s %9s\n",
            "nfx", "n_fluid", "n_bnd", "steps", "cpu_col us", "cpu_1s us", "gpu us",
            "gpu/col", "gpu/1s")

    rows = NamedTuple[]
    for nfx in sizes
        n_total_est = nfx * nfx
        nsteps = clamp(round(Int, budget / max(n_total_est, 1)), 5, 300)

        integ_col, n_fluid, n_bnd = build(nfx; onesided=false, ka=false)
        t_col = us_per_step(integ_col, nsteps)

        integ_1s, _, _ = build(nfx; onesided=true, ka=false)
        t_1s = us_per_step(integ_1s, nsteps)

        t_gpu = NaN
        if HAVE_CUDA
            integ_gpu, _, _ = build(nfx; onesided=true, ka=true, backend=CUDABackend())
            t_gpu = us_per_step(integ_gpu, nsteps; sync = () -> CUDA.synchronize())
        end

        @printf("%6d %10d %10d %8d | %12.1f %12.1f %12.1f | %9.3f %9.3f\n",
                nfx, n_fluid, n_bnd, nsteps, t_col, t_1s, t_gpu, t_gpu / t_col, t_gpu / t_1s)
        push!(rows, (; nfx, n_fluid, n_bnd, t_col, t_1s, t_gpu))
    end

    println()
    if HAVE_CUDA
        crossed = findfirst(r -> r.t_gpu < r.t_col, rows)
        if crossed === nothing
            println("GPU did not beat the CPU coloured sweep at any tested size (up to n_fluid = ",
                    rows[end].n_fluid, "). Extend --sizes to look further, or see the hardware notes")
            println("in docs/gpu-migration-plan.md — this GPU has no measured FP64 throughput edge over")
            println("the CPU, so any win here would come from launch-count amortisation at larger n.")
        else
            println("GPU became faster than the CPU coloured sweep at n_fluid ≈ ", rows[crossed].n_fluid,
                    " (nfx = ", rows[crossed].nfx, ").")
        end
    else
        println("CUDA not available on this machine — CPU-coloured vs CPU-onesided columns only.")
    end

    d, n = dirname(@__DIR__), "bench-output"
    outdir = joinpath(d, n)
    mkpath(outdir)
    outpath = joinpath(outdir, "dambreak_scaling_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).csv")
    open(outpath, "w") do io
        println(io, "nfx,n_fluid,n_bnd,cpu_coloured_us,cpu_onesided_us,gpu_us")
        for r in rows
            println(io, "$(r.nfx),$(r.n_fluid),$(r.n_bnd),$(r.t_col),$(r.t_1s),$(r.t_gpu)")
        end
    end
    println("\nWrote ", outpath)
end

main()
