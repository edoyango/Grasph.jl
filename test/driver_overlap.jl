# driver_overlap.jl – ORB driver with deliberately overlapping per-rank particle regions.
#
# Ranks are arranged on a 3-D integer grid of dimensions nx × ny × nz
# (computed as the nearest-integer cube root factorisation of nprocs).
# Each rank r maps to grid cell (ix, iy, iz) and generates nlocal particles
# uniformly inside a rectangular prism at (ix*step_x, iy*step_y, iz*step_z).
# The prisms are wider than the grid spacing in every dimension, so
# neighbouring prisms always overlap.
#
# Prism geometry base values (at nlocal = 1); dimensions scale with
# nlocal^(1/3) so that particle density stays constant as problem size grows.
#
#   dim   base_step   base_width   overlap
#   x       0.75        0.80        0.05
#   y       0.55        0.60        0.05
#   z       0.45        0.50        0.05
#
# Particle positions are generated with a rank-seeded RNG, so the result is
# fully reproducible for any fixed (nprocs, nlocal) pair.
#
# Run:
#   mpiexec -n <N> julia --project=. test/driver_overlap.jl <nparticles_per_rank>

using MPI
using Random
using StaticArrays
using Printf
using Grasph

MPI.Init()
const COMM   = MPI.COMM_WORLD
const RANK   = MPI.Comm_rank(COMM)
const NPROCS = MPI.Comm_size(COMM)

# ---------------------------------------------------------------------------
# Prism geometry base values (nlocal = 1)
# ---------------------------------------------------------------------------
const BASE_STEP_X  = 0.75
const BASE_STEP_Y  = 0.55
const BASE_STEP_Z  = 0.45
const BASE_WIDTH_X = 0.80
const BASE_WIDTH_Y = 0.60
const BASE_WIDTH_Z = 0.50

# ---------------------------------------------------------------------------
# Parse nlocal from argv
# ---------------------------------------------------------------------------
if length(ARGS) < 1
    RANK == 0 && println("usage: driver_overlap.jl <nparticles_per_rank>")
    MPI.Finalize()
    exit(1)
end
nlocal = parse(Int, ARGS[1])

# ---------------------------------------------------------------------------
# Scale prism geometry with nlocal^(1/3) so particle density stays constant.
# ---------------------------------------------------------------------------
scale   = nlocal ^ (1.0 / 3.0)
step_x  = BASE_STEP_X * scale;  width_x = BASE_WIDTH_X * scale
step_y  = BASE_STEP_Y * scale;  width_y = BASE_WIDTH_Y * scale
step_z  = BASE_STEP_Z * scale;  width_z = BASE_WIDTH_Z * scale

# ---------------------------------------------------------------------------
# Lay ranks out on a 3-D grid (nx × ny × nz).
# nx ≈ ny ≈ nz ≈ nprocs^(1/3); nz rounded up so nx*ny*nz >= nprocs.
# ---------------------------------------------------------------------------
nx = max(1, round(Int, NPROCS ^ (1.0/3.0)))
ny = max(1, round(Int, sqrt(NPROCS / nx)))
nz = cld(NPROCS, nx * ny)

ix = RANK % nx
iy = (RANK ÷ nx) % ny
iz = RANK ÷ (nx * ny)

# ---------------------------------------------------------------------------
# bin_size: ~30 particles per bin across the global domain.
# ---------------------------------------------------------------------------
ntotal     = NPROCS * nlocal
domain_vol = ((nx-1)*step_x + width_x) *
             ((ny-1)*step_y + width_y) *
             ((nz-1)*step_z + width_z)
bin_size   = (domain_vol * 30.0 / ntotal) ^ (1.0/3.0)

# ---------------------------------------------------------------------------
# Prism for this rank.
# ---------------------------------------------------------------------------
prism_lo = SVector(ix*step_x, iy*step_y, iz*step_z)
prism_hi = prism_lo .+ SVector(width_x, width_y, width_z)

# ---------------------------------------------------------------------------
# Generate nlocal particles uniformly inside this rank's prism.
# Seed is deterministic and unique per rank.
# ---------------------------------------------------------------------------
rng_seed = RANK * 1337 + 42
rng = MersenneTwister(rng_seed)

x = Vector{SVector{3,Float64}}(undef, nlocal)
for p in 1:nlocal
    r = rand(rng, SVector{3,Float64})
    x[p] = prism_lo .+ r .* (prism_hi .- prism_lo)
end

# ---------------------------------------------------------------------------
# Print setup summary (rank 0), then per-rank prism bounds in rank order.
# ---------------------------------------------------------------------------
if RANK == 0
    @printf "Overlap driver: %d ranks, %d particles/rank\n" NPROCS nlocal
    @printf "Rank grid: %d x %d x %d (x/y/z)\n" nx ny nz
    @printf "Prism scale = nlocal^(1/3) = %.4f  (step / width / overlap per dimension):\n" scale
    @printf "    x:  %7.3f / %7.3f / %7.3f\n" step_x width_x (width_x - step_x)
    @printf "    y:  %7.3f / %7.3f / %7.3f\n" step_y width_y (width_y - step_y)
    @printf "    z:  %7.3f / %7.3f / %7.3f\n" step_z width_z (width_z - step_z)
    @printf "bin_size = %.5f  (~30 particles/bin for %d total particles)\n" bin_size ntotal
    println("\nPer-rank input prisms:")
end

MPI.Barrier(COMM)
for p in 0:NPROCS-1
    if RANK == p
        lo = prism_lo;  hi = prism_hi
        @printf "  rank %3d (%d,%d,%d):  lo=[%8.3f %8.3f %8.3f]  hi=[%8.3f %8.3f %8.3f]\n" RANK ix iy iz lo[1] lo[2] lo[3] hi[1] hi[2] hi[3]
        flush(stdout)
    end
    MPI.Barrier(COMM)
end

# ---------------------------------------------------------------------------
# ORB decomposition.
# ---------------------------------------------------------------------------
cfg = ORBConfig{3, true}(bin_size)
local_lo, local_hi = orb_decompose(cfg, x, COMM)

# ---------------------------------------------------------------------------
# Print assigned subdomains in rank order.
# ---------------------------------------------------------------------------
MPI.Barrier(COMM)
RANK == 0 && println("\nAssigned ORB subdomains:")

MPI.Barrier(COMM)
for p in 0:NPROCS-1
    if RANK == p
        @printf "  rank %3d:  lo=[%8.3f %8.3f %8.3f]  hi=[%8.3f %8.3f %8.3f]\n" RANK local_lo[1] local_lo[2] local_lo[3] local_hi[1] local_hi[2] local_hi[3]
        flush(stdout)
    end
    MPI.Barrier(COMM)
end

# ---------------------------------------------------------------------------
# Volume check: union of ORB subdomains must tile the global bounding box.
# ---------------------------------------------------------------------------
all_lo = MPI.Gather(Vector{Float64}(local_lo), 0, COMM)
all_hi = MPI.Gather(Vector{Float64}(local_hi), 0, COMM)

if RANK == 0
    let
        domain_lo_orb = SVector{3,Float64}(minimum(all_lo[d:3:end]) for d in 1:3)
        domain_hi_orb = SVector{3,Float64}(maximum(all_hi[d:3:end]) for d in 1:3)
        global_vol    = prod(domain_hi_orb .- domain_lo_orb)
        total_vol     = 0.0
        for r in 0:NPROCS-1
            lo_r = all_lo[r*3+1 : r*3+3]
            hi_r = all_hi[r*3+1 : r*3+3]
            rvol = prod(hi_r .- lo_r)
            rvol <= 0.0 && @printf "FAIL: rank %d has zero-volume subdomain!\n" r
            total_vol += rvol
        end
        @printf "\nVolume check  –  sum of subdomain vols: %12.6f   global: %12.6f\n" total_vol global_vol
        if abs(total_vol - global_vol) / global_vol < 1e-6
            println("PASS: volumes match.")
        else
            println("FAIL: volume mismatch.")
        end
    end
end

MPI.Finalize()
