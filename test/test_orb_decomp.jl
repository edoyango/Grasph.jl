# test_orb_decomp.jl – ORB decomposition MPI test suite
#
# Matches the Fortran test_suite.f90 scenarios exactly.
#
# Run:
#   mpiexec -n 2 julia --project=. test/test_orb_decomp.jl
#   mpiexec -n 4 julia --project=. test/test_orb_decomp.jl
#   mpiexec -n 8 julia --project=. test/test_orb_decomp.jl
#
# Or via MPI.mpiexec from the package root:
#   julia --project=. -e '
#     using MPI
#     MPI.mpiexec() do mpi
#       run(`$mpi -n 4 julia --project=. test/test_orb_decomp.jl`)
#     end'

using MPI
using Test
using StaticArrays
using Grasph

MPI.Init()
const COMM   = MPI.COMM_WORLD
const RANK   = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

# ---------------------------------------------------------------------------
# Particle placement convention (matches Fortran test_suite.f90)
#
# bin_size = 1.0.  Bin k contains one particle at k+0.1 for k < N-1,
# and at N-0.1 for the last bin.  This gives domain_lo = 0.1, ng = N,
# and domain_hi = N + 0.1 (before snapping).  After snapping:
#   domain_hi = domain_lo + ng * bin_size = 0.1 + N.
# A subdomain spanning bins [a..b] has
#   local_lo = 0.1 + a,   local_hi = 1.1 + b.
# ---------------------------------------------------------------------------
const DL = 0.1   # domain_lo after Allreduce

function _particle_pos(k::Int, N::Int)
    k < N - 1 ? k + 0.1 : N - 0.1
end

# Build a Vector{SVector{ndim,Float64}} for a uniform grid of N^ndim
# particles, assigning a contiguous slice of flat indices to this rank.
function _uniform_grid(N::Int, ::Val{ND}) where {ND}
    ntotal  = N^ND
    nlocal  = ntotal ÷ NPROCS
    start_p = RANK * nlocal          # 0-based flat index of first local particle
    x = Vector{SVector{ND,Float64}}(undef, nlocal)
    for p in 0:nlocal-1
        flat = start_p + p
        coords = MVector{ND,Float64}(undef)
        for d in 1:ND
            idx       = flat % N          # bin index along dim d (ix varies fastest)
            flat      = flat ÷ N
            coords[d] = _particle_pos(idx, N)
        end
        x[p+1] = SVector(coords)
    end
    return x
end

# Gather all local_lo / local_hi to rank 0 for volume check.
function _check_volume(lo, hi, domain_lo, domain_hi, label::String)
    ndim = length(lo)
    all_lo = MPI.Gather(Vector{Float64}(lo), 0, COMM)
    all_hi = MPI.Gather(Vector{Float64}(hi), 0, COMM)

    if RANK == 0
        global_vol = prod(domain_hi .- domain_lo)
        total_vol  = 0.0
        for r in 0:NPROCS-1
            rvol = prod(all_hi[r*ndim+1:(r+1)*ndim] .- all_lo[r*ndim+1:(r+1)*ndim])
            @test rvol > 0  # no zero-volume subdomain
            total_vol += rvol
        end
        @test total_vol ≈ global_vol  rtol=1e-6
    end
end

# ---------------------------------------------------------------------------
# uniform_1d  – N=4 bins, each rank holds N/nprocs consecutive particles
# ---------------------------------------------------------------------------
function test_uniform_1d()
    N  = 4
    N > NPROCS || (RANK == 0 && println("SKIP uniform_1d: N=$N < nprocs"); return)
    x   = _uniform_grid(N, Val(1))
    cfg = ORBConfig{1}(1.0)
    lo, hi = orb_decompose(cfg, x, COMM)

    if NPROCS == 2
        exp_lo = RANK == 0 ? [DL]     : [DL+2]
        exp_hi = RANK == 0 ? [DL+2]   : [DL+N]
    elseif NPROCS == 4
        exp_lo = [DL + RANK]
        exp_hi = [DL + RANK + 1]
    end

    @test lo ≈ exp_lo  atol=1e-10
    @test hi ≈ exp_hi  atol=1e-10
    _check_volume(lo, hi, [DL], [DL+N], "uniform_1d")
    RANK == 0 && println("uniform_1d     done")
end

# ---------------------------------------------------------------------------
# uniform_2d  – 4×4 grid
# ---------------------------------------------------------------------------
function test_uniform_2d()
    N   = 4
    x   = _uniform_grid(N, Val(2))
    cfg = ORBConfig{2}(1.0)
    lo, hi = orb_decompose(cfg, x, COMM)

    if NPROCS == 2
        exp_lo = RANK == 0 ? [DL,   DL] : [DL+2, DL]
        exp_hi = RANK == 0 ? [DL+2, DL+N] : [DL+N, DL+N]
    elseif NPROCS == 4
        # root x-cut bin 1; each half y-cut bin 1
        exp_lo = [[DL,   DL  ], [DL,   DL+2], [DL+2, DL  ], [DL+2, DL+2]][RANK+1]
        exp_hi = [[DL+2, DL+2], [DL+2, DL+N], [DL+N, DL+2], [DL+N, DL+N]][RANK+1]
    elseif NPROCS == 8
        # x-half → ranks 0..3 (xb=0) and 4..7 (xb=2)
        # within x-half: y-half → lo (0,1,4,5 → yb=0) and hi (2,3,6,7 → yb=2)
        # within y-half: x-bin → even rank in group gets x=0, odd gets x=1
        xb = (RANK ÷ 4) * 2 + RANK % 2
        yb = (RANK % 4 ÷ 2) * 2
        exp_lo = [DL+xb,   DL+yb  ]
        exp_hi = [DL+xb+1, DL+yb+2]
    end

    @test lo ≈ exp_lo  atol=1e-10
    @test hi ≈ exp_hi  atol=1e-10
    _check_volume(lo, hi, [DL, DL], [DL+N, DL+N], "uniform_2d")
    RANK == 0 && println("uniform_2d     done")
end

# ---------------------------------------------------------------------------
# uniform_3d  – 4×4×4 grid
# ---------------------------------------------------------------------------
function test_uniform_3d()
    N   = 4
    x   = _uniform_grid(N, Val(3))
    cfg = ORBConfig{3}(1.0)
    lo, hi = orb_decompose(cfg, x, COMM)

    if NPROCS == 2
        exp_lo = RANK == 0 ? [DL,   DL, DL] : [DL+2, DL, DL]
        exp_hi = RANK == 0 ? [DL+2, DL+N, DL+N] : [DL+N, DL+N, DL+N]
    elseif NPROCS == 4
        exp_lo = [[DL,   DL,   DL], [DL,   DL+2, DL],
                  [DL+2, DL,   DL], [DL+2, DL+2, DL]][RANK+1]
        exp_hi = [[DL+2, DL+2, DL+N], [DL+2, DL+N, DL+N],
                  [DL+N, DL+2, DL+N], [DL+N, DL+N, DL+N]][RANK+1]
    elseif NPROCS == 8
        # x-cut → y-cut → z-cut, each splitting the 4-bin axis at bin 1
        xb = (RANK ÷ 4) * 2
        yb = (RANK % 4 ÷ 2) * 2
        zb = (RANK % 2) * 2
        exp_lo = [DL+xb,   DL+yb,   DL+zb  ]
        exp_hi = [DL+xb+2, DL+yb+2, DL+zb+2]
    end

    @test lo ≈ exp_lo  atol=1e-10
    @test hi ≈ exp_hi  atol=1e-10
    _check_volume(lo, hi, [DL, DL, DL], [DL+N, DL+N, DL+N], "uniform_3d")
    RANK == 0 && println("uniform_3d     done")
end

# ---------------------------------------------------------------------------
# slope_1d  – N=20 bins, bin k has weight 20-k, n=4 ranks only
#
# Analytical cuts:
#   root [0..19] total=210: cum ≥ 105 at k=5 → cut bin 5
#     left  [0..5]  total=105: cum ≥ 53 at k=2 → cut bin 2
#       rank 0: [0..2]   lo=0.1, hi=3.1
#       rank 1: [3..5]   lo=3.1, hi=6.1
#     right [6..19] total=105: cum ≥ 53 at k=4 (from bin 6) → cut bin 10
#       rank 2: [6..10]  lo=6.1, hi=11.1
#       rank 3: [11..19] lo=11.1, hi=20.1
# ---------------------------------------------------------------------------
function test_slope_1d()
    NPROCS == 4 || return
    N    = 20
    k_lo = 5 * RANK
    k_hi = k_lo + 4

    # Count particles for this rank's 5 bins: bin k has weight (20-k)
    npart = sum(20 - k for k in k_lo:k_hi)
    x = Vector{SVector{1,Float64}}(undef, npart)
    p = 1
    for k in k_lo:k_hi
        pos = k < N - 1 ? k + 0.1 : N - 0.1
        for _ in 1:(20-k)
            x[p] = SVector(pos)
            p += 1
        end
    end

    cfg = ORBConfig{1}(1.0)
    lo, hi = orb_decompose(cfg, x, COMM)

    exp_lo = [0.1, 3.1, 6.1, 11.1][RANK+1]
    exp_hi = [3.1, 6.1, 11.1, 20.1][RANK+1]

    @test lo[1] ≈ exp_lo  atol=1e-10
    @test hi[1] ≈ exp_hi  atol=1e-10
    _check_volume(lo, hi, [0.1], [20.1], "slope_1d")
    RANK == 0 && println("slope_1d       done")
end

# ---------------------------------------------------------------------------
# predistributed_1d  – each rank holds exactly 1 particle in its own bin
# ---------------------------------------------------------------------------
function test_predistributed_1d()
    NPROCS == 4 || return
    pos_map = [0.1, 1.1, 2.1, 3.9]
    x   = [SVector(pos_map[RANK+1])]
    cfg = ORBConfig{1}(1.0)
    lo, hi = orb_decompose(cfg, x, COMM)

    @test lo[1] ≈ DL + RANK      atol=1e-10
    @test hi[1] ≈ DL + RANK + 1  atol=1e-10
    _check_volume(lo, hi, [DL], [DL+4], "predistrib_1d")
    RANK == 0 && println("predistrib_1d  done")
end

# ---------------------------------------------------------------------------
# predistributed_2d  – each rank holds its 2×2 quadrant of a 4×4 grid
# ---------------------------------------------------------------------------
function test_predistributed_2d()
    NPROCS == 4 || return
    xs_map = [[0.1, 1.1], [0.1, 1.1], [2.1, 3.9], [2.1, 3.9]]
    ys_map = [[0.1, 1.1], [2.1, 3.9], [0.1, 1.1], [2.1, 3.9]]
    xs = xs_map[RANK+1];  ys = ys_map[RANK+1]

    x = Vector{SVector{2,Float64}}(undef, 4)
    p = 1
    for iy in 1:2, ix in 1:2
        x[p] = SVector(xs[ix], ys[iy]);  p += 1
    end

    cfg = ORBConfig{2}(1.0)
    lo, hi = orb_decompose(cfg, x, COMM)

    exp_lo = [[0.1, 0.1], [0.1, 2.1], [2.1, 0.1], [2.1, 2.1]][RANK+1]
    exp_hi = [[2.1, 2.1], [2.1, 4.1], [4.1, 2.1], [4.1, 4.1]][RANK+1]

    @test lo ≈ exp_lo  atol=1e-10
    @test hi ≈ exp_hi  atol=1e-10
    _check_volume(lo, hi, [DL, DL], [DL+4, DL+4], "predistrib_2d")
    RANK == 0 && println("predistrib_2d  done")
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
RANK == 0 && println("=== ORB test suite  (nprocs = $NPROCS) ===")

nfail_local = Ref(0)

# Wrap @testset so we can count failures across ranks
function _run(f, label)
    try
        f()
    catch e
        println("ERROR in $label on rank $RANK: $e")
        nfail_local[] += 1
    end
end

if NPROCS in (2, 4, 8)
    _run(test_uniform_1d, "uniform_1d")
    _run(test_uniform_2d, "uniform_2d")
    _run(test_uniform_3d, "uniform_3d")
end
if NPROCS == 4
    _run(test_slope_1d,           "slope_1d")
    _run(test_predistributed_1d,  "predistrib_1d")
    _run(test_predistributed_2d,  "predistrib_2d")
end
if !(NPROCS in (2, 4, 8))
    RANK == 0 && println("No tests defined for nprocs=$NPROCS. Run with -n 2, 4, or 8.")
end

nfail_global = MPI.Allreduce(nfail_local[], MPI.SUM, COMM)
if RANK == 0
    println()
    nfail_global == 0 ? println("All tests PASSED.") : println("$nfail_global check(s) FAILED.")
end

MPI.Finalize()
