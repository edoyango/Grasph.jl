# bench/gpu_microbench.jl — the explanatory variables behind
# bench/dambreak_scaling.jl's item-11/12 results, measured directly instead
# of inferred from step timings.
#
# Items 11 and 12 (docs/gpu-migration-plan.md) both rest on two numbers
# measured ad hoc on the RTX 4060 Laptop and never committed as a
# reproducible script: a ~8.3µs/kernel-launch overhead floor, and 0.138
# TFLOP/s GPU FP64 throughput vs. 0.143-0.311 TFLOP/s on that machine's CPU
# (i.e. no FP64 compute advantage at all — the 4060's only real edge was
# memory bandwidth, 216 GB/s). Re-running dambreak_scaling.jl on new
# hardware without re-measuring these tells you THAT the crossovers moved,
# not WHY. This script measures the four numbers that explain it:
#
#   - launch_us    kernel-launch overhead, GPU only (near-zero-work kernel,
#                  many reps, host-timed around a synchronize)
#   - gpu_tflops   device FP64 FMA throughput (compute-bound: many FMAs per
#                  thread, so launch/memory cost is negligible in the total)
#   - cpu_tflops   host FP64 FMA throughput at whatever thread count this
#                  process was started with (Threads.nthreads()) — run this
#                  script at -t 1 and -t 26 the same way dambreak_scaling.jl
#                  is run twice, for the same reason: Julia's thread count is
#                  fixed at process start, not a runtime switch
#   - gpu_gb_s     device memory bandwidth (STREAM-triad-style kernel)
#
# Usage:
#   julia --project=<merged-env> -t 1  bench/gpu_microbench.jl
#   julia --project=<merged-env> -t 26 bench/gpu_microbench.jl
# (CUDA must be in the active environment; GPU rows are skipped with a
# warning if CUDA.functional() is false, same convention as
# dambreak_scaling.jl.)

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using Printf
using Dates

const HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch err
    @warn "CUDA not available for this benchmark; GPU rows will be skipped" exception=err
    false
end

# ndrange=1: one thread does (nearly) nothing. Isolates fixed per-launch
# overhead from any compute or memory cost.
@kernel function _noop_kernel!(x)
    i = @index(Global, Linear)
    @inbounds x[i] += 1
end

function launch_overhead_us(backend; nreps=2000)
    x = KA.zeros(backend, Float64, 1)
    kernel = _noop_kernel!(backend, 1)
    for _ in 1:20
        kernel(x; ndrange=1)
    end
    KA.synchronize(backend)
    t0 = time_ns()
    for _ in 1:nreps
        kernel(x; ndrange=1)
    end
    KA.synchronize(backend)
    t1 = time_ns()
    return (t1 - t0) / 1e3 / nreps
end

# Compute-bound: n_iters chained FMAs per thread (each depends on the last,
# so nothing pipelines away) with a runtime-derived starting value (defeats
# constant folding of the whole chain into a closed form).
@kernel function _fma_kernel!(out, @Const(n_iters))
    i = @index(Global, Linear)
    a = 1.0 + 1f-7 * Float64(i % 7)
    b = 1.0 - 1f-7 * Float64(i % 11)
    c = 1.0 + 1f-9 * Float64(i)
    @inbounds for _ in 1:n_iters
        c = muladd(a, c, b)
    end
    @inbounds out[i] = c
end

function gpu_fma_tflops(backend; n=2^20, n_iters=20_000, nreps=5)
    out = KA.zeros(backend, Float64, n)
    kernel = _fma_kernel!(backend, 256)
    kernel(out, n_iters; ndrange=n)
    KA.synchronize(backend)
    t0 = time_ns()
    for _ in 1:nreps
        kernel(out, n_iters; ndrange=n)
    end
    KA.synchronize(backend)
    t1 = time_ns()
    elapsed = (t1 - t0) / 1e9 / nreps
    flops = Float64(n) * n_iters * 2  # 1 FMA = 2 FLOPs
    return flops / elapsed / 1e12
end

# Same kernel body, host-threaded — the CPU side of the same comparison.
# Unlike the GPU kernel, a single CPU thread has no other warps/threads to
# hide FMA latency behind, so one accumulator per thread would measure FMA
# *latency* (~1-2 GFLOP/s, pipeline-depth-bound), not throughput. Eight
# independent accumulator chains per thread give the core's out-of-order
# engine (and, via @fastmath, the auto-vectorizer) independent work to
# pipeline, so this instead measures throughput the way the GPU kernel does.
function cpu_fma_tflops(; n=2^16, n_iters=20_000)
    out = Vector{Float64}(undef, n)
    _run() = Threads.@threads for i in 1:n
        a = 1.0 + 1f-7 * Float64(i % 7)
        b = 1.0 - 1f-7 * Float64(i % 11)
        c1 = 1.0 + 1f-9 * Float64(i)
        c2, c3, c4 = c1 + 1.0, c1 + 2.0, c1 + 3.0
        c5, c6, c7, c8 = c1 + 4.0, c1 + 5.0, c1 + 6.0, c1 + 7.0
        @inbounds @fastmath for _ in 1:n_iters
            c1 = muladd(a, c1, b)
            c2 = muladd(a, c2, b)
            c3 = muladd(a, c3, b)
            c4 = muladd(a, c4, b)
            c5 = muladd(a, c5, b)
            c6 = muladd(a, c6, b)
            c7 = muladd(a, c7, b)
            c8 = muladd(a, c8, b)
        end
        @inbounds out[i] = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    end
    _run()
    t0 = time_ns()
    _run()
    t1 = time_ns()
    elapsed = (t1 - t0) / 1e9
    flops = Float64(n) * n_iters * 8 * 2  # 8 independent chains, 1 FMA = 2 FLOPs
    return flops / elapsed / 1e12
end

# STREAM-triad: a[i] = b[i] + scalar*c[i]. Memory-bound by construction (2
# FLOPs per 3 loads/stores of 8 bytes each), so this measures bandwidth, not
# compute.
@kernel function _triad_kernel!(a, @Const(b), @Const(c), @Const(scalar))
    i = @index(Global, Linear)
    @inbounds a[i] = b[i] + scalar * c[i]
end

function gpu_bandwidth_gb_s(backend; n=2^27, nreps=20)
    a = KA.allocate(backend, Float64, n)
    b = KA.allocate(backend, Float64, n)
    c = KA.allocate(backend, Float64, n)
    fill!(b, 1.0)
    fill!(c, 2.0)
    kernel = _triad_kernel!(backend, 256)
    kernel(a, b, c, 3.0; ndrange=n)
    KA.synchronize(backend)
    t0 = time_ns()
    for _ in 1:nreps
        kernel(a, b, c, 3.0; ndrange=n)
    end
    KA.synchronize(backend)
    t1 = time_ns()
    elapsed = (t1 - t0) / 1e9 / nreps
    bytes = 3.0 * n * 8  # 2 reads + 1 write per element
    return bytes / elapsed / 1e9
end

function main()
    println("=== GPU/CPU microbenchmark — ", Dates.now(), " ===")
    println("Julia threads: ", Threads.nthreads(), " (Sys.CPU_THREADS = ", Sys.CPU_THREADS, ")")
    println("CPU: ", Sys.cpu_info()[1].model)
    println("CUDA available: ", HAVE_CUDA)
    if HAVE_CUDA
        println("GPU: ", CUDA.name(CUDA.device()), "  (CUDA.jl v", pkgversion(CUDA), ")")
    end
    println()

    print("Measuring host CPU FP64 FMA throughput (nthreads=", Threads.nthreads(), ")... ")
    cpu_tflops = cpu_fma_tflops()
    @printf("%.4f TFLOP/s\n", cpu_tflops)

    launch_us = NaN
    gpu_tflops = NaN
    gpu_gb_s = NaN
    if HAVE_CUDA
        backend = CUDABackend()
        print("Measuring GPU kernel-launch overhead... ")
        launch_us = launch_overhead_us(backend)
        @printf("%.3f us/launch\n", launch_us)

        print("Measuring GPU FP64 FMA throughput... ")
        gpu_tflops = gpu_fma_tflops(backend)
        @printf("%.4f TFLOP/s\n", gpu_tflops)

        print("Measuring GPU memory bandwidth (STREAM triad)... ")
        gpu_gb_s = gpu_bandwidth_gb_s(backend)
        @printf("%.1f GB/s\n", gpu_gb_s)
    else
        println("CUDA not available — GPU rows skipped.")
    end

    println()
    println("Summary (compare against docs/gpu-migration-plan.md item 2's RTX 4060")
    println("Laptop figures: ~8.3 us/launch, 0.138 TFLOP/s GPU, 0.143-0.311 TFLOP/s CPU,")
    println("216 GB/s):")
    @printf("  %-12s %10s\n", "nthreads", string(Threads.nthreads()))
    @printf("  %-12s %10.4f TFLOP/s\n", "cpu_tflops", cpu_tflops)
    if HAVE_CUDA
        @printf("  %-12s %10.3f us\n", "launch_us", launch_us)
        @printf("  %-12s %10.4f TFLOP/s\n", "gpu_tflops", gpu_tflops)
        @printf("  %-12s %10.1f GB/s\n", "gpu_gb_s", gpu_gb_s)
    end

    d, n = dirname(@__DIR__), "bench-output"
    outdir = joinpath(d, n)
    mkpath(outdir)
    outpath = joinpath(outdir, "gpu_microbench_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS")).csv")
    open(outpath, "w") do io
        println(io, "nthreads,cpu_tflops,launch_us,gpu_tflops,gpu_gb_s")
        println(io, "$(Threads.nthreads()),$(cpu_tflops),$(launch_us),$(gpu_tflops),$(gpu_gb_s)")
    end
    println("\nWrote ", outpath)
end

main()
