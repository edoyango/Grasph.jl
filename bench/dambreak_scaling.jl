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
# Reports SIX columns: CPU-coloured (today's production default for every
# script), CPU-onesided (the same full-stencil algorithm the GPU runs, on
# CPU), GPU-onesided (today's production GPU sweep, ka=true), GPU-coloured
# (ColouredKA — see docs/gpu-migration-plan.md's coloured-GPU benchmarking
# spike, Backend.jl/KAKernels.jl), GPU-onesided-with-skin (ka=true plus
# verlet_skin > 0 — docs/gpu-migration-plan.md deferred item 1: skips the
# per-step sort + grid rebuild while no particle has moved far enough to
# invalidate the current, skin-padded cell list), and GPU-neighbour-list
# (NeighbourListKA — docs/gpu-migration-plan.md's explicit-neighbour-list
# benchmarking spike: caches an explicit, over-inclusive candidate pair list
# at each of the same skin-gated rebuilds gpu_1s_skin already uses, and has
# the per-step sweep consume that flat list instead of re-deriving candidates
# from the cell grid every step). The cpu_1s column is not optional — without
# it, a GPU "speedup" over the coloured sweep can't be told apart from an
# artifact of comparing against a half-shell algorithm that does half the
# pair evaluations per step. gpu_col answers a different question: does
# porting that same half-work half-shell algorithm to GPU (one kernel launch
# per colour, 6x/2D-self more launches than the single onesided-KA launch)
# still win once launch overhead is paid 6x, or does onesided-KA's
# single-launch/double-arithmetic shape win on this hardware? gpu_1s_skin
# answers yet another question: given item 11 already showed this GPU is
# launch-count-dominated rather than compute-bound, does removing the
# sort+grid launches entirely on most steps (this benchmark's fluid block is
# falling under gravity from rest, so displacement per step is tiny relative
# to the skin margin) recover meaningfully more than gpu_1s already does?
# gpu_nlist answers a further question on top of that: given the same
# skin-gated rebuild cadence gpu_1s_skin already pays for, does ALSO removing
# the cell-stencil-walk compute on every step (not just the sort+grid
# launches) recover anything more — on hardware item 2/11 already established
# is launch-count-bound, not compute-bound?
#
# A SEVENTH column, cpu_nlist, runs NeighbourListKA on the CPU array backend
# (KA.CPU(), no CUDA involved — the same kernels already used for gpu_nlist
# are backend-generic) to ask a CPU-side version of the same question: does
# skipping the cell-stencil walk help even without a GPU's redundant-
# per-thread-scalar-math cost profile? NOTE: this is still the *one-sided*
# candidate-list shape (_pair_self_onesided!/_pair_coupled_onesided!, full
# neighbourhood per particle) — NOT cpu_col's half-shell, Newton's-third-law
# two-sided algorithm. A persistent-pairs version of the *coloured* sweep
# would need colour-grouped pair caching (colour-partitioning is what makes
# ColouredCPU's parallel two-sided writes race-free without atomics) and
# isn't built; cpu_nlist only answers whether cell-walk avoidance helps
# CPU at all, using the one-sided algorithm already on hand.
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

function build(nfx; onesided=false, ka=false, mode=nothing, backend=nothing, verlet_skin_frac=0.0)
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
                           onesided = onesided, ka = ka, mode = mode)
    fbi = SystemInteraction(kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid, static_boundary;
                            onesided = onesided, ka = ka, mode = mode)
    verlet_skin = verlet_skin_frac * kernel.interaction_length
    integrator = LeapFrogTimeIntegrator([fluid, boundary], [fi, fbi]; verlet_skin = verlet_skin)
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
    println("Julia threads: ", Threads.nthreads(), " (Sys.CPU_THREADS = ", Sys.CPU_THREADS, ")")
    println("CPU: ", Sys.cpu_info()[1].model)
    println("CUDA available: ", HAVE_CUDA)
    if HAVE_CUDA
        println("GPU: ", CUDA.name(CUDA.device()), "  (CUDA.jl v", pkgversion(CUDA), ")")
    end
    println()
    @printf("%6s %10s %10s %8s | %12s %12s %12s %12s %12s %12s %12s | %9s %9s %9s %9s %9s %9s %9s\n",
            "nfx", "n_fluid", "n_bnd", "steps", "cpu_col us", "cpu_1s us", "cpu_nlist us", "gpu_1s us", "gpu_col us", "gpu_1s+skin us", "gpu_nlist us",
            "1s/col", "col/col", "col/1s", "skin/1s", "nlist/skin", "cnl/col", "gnl/cnl")

    rows = NamedTuple[]
    for nfx in sizes
        n_total_est = nfx * nfx
        nsteps = clamp(round(Int, budget / max(n_total_est, 1)), 5, 300)

        integ_col, n_fluid, n_bnd = build(nfx; onesided=false, ka=false)
        t_col = us_per_step(integ_col, nsteps)

        integ_1s, _, _ = build(nfx; onesided=true, ka=false)
        t_1s = us_per_step(integ_1s, nsteps)

        integ_cpu_nlist, _, _ = build(nfx; mode=Grasph.NeighbourListKA(), verlet_skin_frac=0.2)
        t_cpu_nlist = us_per_step(integ_cpu_nlist, nsteps)

        t_gpu = NaN
        t_gpu_col = NaN
        t_gpu_skin = NaN
        t_gpu_nlist = NaN
        if HAVE_CUDA
            integ_gpu, _, _ = build(nfx; onesided=true, ka=true, backend=CUDABackend())
            t_gpu = us_per_step(integ_gpu, nsteps; sync = () -> CUDA.synchronize())

            integ_gpu_col, _, _ = build(nfx; mode=Grasph.ColouredKA(), backend=CUDABackend())
            t_gpu_col = us_per_step(integ_gpu_col, nsteps; sync = () -> CUDA.synchronize())

            integ_gpu_skin, _, _ = build(nfx; onesided=true, ka=true, backend=CUDABackend(), verlet_skin_frac=0.2)
            t_gpu_skin = us_per_step(integ_gpu_skin, nsteps; sync = () -> CUDA.synchronize())

            integ_gpu_nlist, _, _ = build(nfx; mode=Grasph.NeighbourListKA(), backend=CUDABackend(), verlet_skin_frac=0.2)
            t_gpu_nlist = us_per_step(integ_gpu_nlist, nsteps; sync = () -> CUDA.synchronize())
        end

        @printf("%6d %10d %10d %8d | %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f | %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n",
                nfx, n_fluid, n_bnd, nsteps, t_col, t_1s, t_cpu_nlist, t_gpu, t_gpu_col, t_gpu_skin, t_gpu_nlist,
                t_gpu / t_1s, t_gpu_col / t_col, t_gpu_col / t_gpu, t_gpu_skin / t_gpu, t_gpu_nlist / t_gpu_skin,
                t_cpu_nlist / t_col, t_gpu_nlist / t_cpu_nlist)
        push!(rows, (; nfx, n_fluid, n_bnd, t_col, t_1s, t_cpu_nlist, t_gpu, t_gpu_col, t_gpu_skin, t_gpu_nlist))
    end

    println()
    cpu_nlist_speedup = [r.t_col / r.t_cpu_nlist for r in rows]
    best_k = argmax(cpu_nlist_speedup)
    println("CPU-onesided-with-persistent-pairs (NeighbourListKA on KA.CPU(), verlet_skin_frac=0.2) vs",
            " CPU-coloured: ", round(minimum(cpu_nlist_speedup); digits=3), "x-",
            round(maximum(cpu_nlist_speedup); digits=3), "x across tested sizes (best at n_fluid ≈ ",
            rows[best_k].n_fluid, "). NOTE: cpu_nlist is the one-sided candidate shape, not cpu_col's",
            " half-shell two-sided algorithm — see this file's header comment.")

    if HAVE_CUDA
        crossed = findfirst(r -> r.t_gpu < r.t_col, rows)
        if crossed === nothing
            println("GPU-onesided did not beat the CPU coloured sweep at any tested size (up to n_fluid = ",
                    rows[end].n_fluid, "). Extend --sizes to look further, or see the hardware notes")
            println("in docs/gpu-migration-plan.md — this GPU has no measured FP64 throughput edge over")
            println("the CPU, so any win here would come from launch-count amortisation at larger n.")
        else
            println("GPU-onesided became faster than the CPU coloured sweep at n_fluid ≈ ", rows[crossed].n_fluid,
                    " (nfx = ", rows[crossed].nfx, ").")
        end

        crossed_col_vs_col = findfirst(r -> r.t_gpu_col < r.t_col, rows)
        if crossed_col_vs_col === nothing
            println("GPU-coloured (ColouredKA) did not beat the CPU coloured sweep at any tested size.")
        else
            println("GPU-coloured (ColouredKA) became faster than the CPU coloured sweep at n_fluid ≈ ",
                    rows[crossed_col_vs_col].n_fluid, " (nfx = ", rows[crossed_col_vs_col].nfx, ").")
        end

        crossed_col_vs_1s = findfirst(r -> r.t_gpu_col < r.t_gpu, rows)
        if crossed_col_vs_1s === nothing
            println("GPU-coloured (ColouredKA) did not beat GPU-onesided at any tested size — the extra",
                    " 6x/2D-self kernel launches never pay for themselves on this hardware.")
        else
            println("GPU-coloured (ColouredKA) became faster than GPU-onesided at n_fluid ≈ ",
                    rows[crossed_col_vs_1s].n_fluid, " (nfx = ", rows[crossed_col_vs_1s].nfx, ").")
        end

        skin_speedup = [r.t_gpu / r.t_gpu_skin for r in rows]
        best_i = argmax(skin_speedup)
        println("GPU-onesided-with-skin (verlet_skin_frac=0.2) vs plain GPU-onesided: ",
                round(minimum(skin_speedup); digits=3), "x-", round(maximum(skin_speedup); digits=3),
                "x across tested sizes (best at n_fluid ≈ ", rows[best_i].n_fluid, ").")

        nlist_speedup = [r.t_gpu_skin / r.t_gpu_nlist for r in rows]
        best_j = argmax(nlist_speedup)
        println("GPU-neighbour-list (NeighbourListKA, same verlet_skin_frac=0.2) vs GPU-onesided-with-skin: ",
                round(minimum(nlist_speedup); digits=3), "x-", round(maximum(nlist_speedup); digits=3),
                "x across tested sizes (best at n_fluid ≈ ", rows[best_j].n_fluid, ").")
    else
        println("CUDA not available on this machine — CPU-coloured vs CPU-onesided columns only.")
    end

    d, n = dirname(@__DIR__), "bench-output"
    outdir = joinpath(d, n)
    mkpath(outdir)
    outpath = joinpath(outdir, "dambreak_scaling_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).csv")
    open(outpath, "w") do io
        println(io, "nfx,n_fluid,n_bnd,cpu_coloured_us,cpu_onesided_us,cpu_neighbour_list_us,gpu_onesided_us,gpu_coloured_us,gpu_onesided_skin_us,gpu_neighbour_list_us")
        for r in rows
            println(io, "$(r.nfx),$(r.n_fluid),$(r.n_bnd),$(r.t_col),$(r.t_1s),$(r.t_cpu_nlist),$(r.t_gpu),$(r.t_gpu_col),$(r.t_gpu_skin),$(r.t_gpu_nlist)")
        end
    end
    println("\nWrote ", outpath)
end

main()
