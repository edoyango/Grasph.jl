export ORBConfig, orb_decompose

using MPI
using Printf
using StaticArrays

# ---------------------------------------------------------------------------
# Orthogonal Recursive Bisection (ORB) domain decomposition.
#
# Partitions a domain among MPI ranks using a balanced binary tree.
# All ranks traverse the same DFS stack (deterministic via broadcast), so
# no communicator splitting or duplication is needed.
#
# Usage:
#   cfg = ORBConfig{2}(bin_size)           # timers off (default)
#   cfg = ORBConfig{2, true}(bin_size)     # timers on
#   lo, hi = orb_decompose(cfg, x, comm)
#   # lo, hi are SVector{2,Float64} bounds for this rank's subdomain
# ---------------------------------------------------------------------------

struct ORBConfig{ND, T<:AbstractFloat, TIMERS}
    bin_size :: T
    function ORBConfig{ND, T, TIMERS}(args...) where {ND, T, TIMERS}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{ND, T, TIMERS}(args...)
    end
end
# ORBConfig{2}(h)       – timers off, T inferred from h
# ORBConfig{2, true}(h) – timers on,  T inferred from h  (backward-compatible)
ORBConfig{ND}(bin_size::T)        where {ND, T<:AbstractFloat} = ORBConfig{ND, T, false}(bin_size)
ORBConfig{ND, TIMERS}(bin_size::T) where {ND, TIMERS, T<:AbstractFloat} = ORBConfig{ND, T, TIMERS}(bin_size)

struct _StackEntry{ND}
    lo   :: SVector{ND,Int}   # lower bin index per dim (0-based)
    hi   :: SVector{ND,Int}   # upper bin index per dim (0-based)
    g_lo :: Int               # first rank in this node's group
    g_hi :: Int               # last  rank in this node's group
    function _StackEntry{ND}(args...) where {ND}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{ND}(args...)
    end
end

# ---------------------------------------------------------------------------
# Build a flat (column-major / dim-1-varies-fastest) histogram from particle
# positions.  Only allocates for the local bounding-box of this rank's
# particles; lbin_lo and lng describe the offset and size of that box.
# ---------------------------------------------------------------------------
function _build_local_hist!(hist::Vector{Int},
                             x::Vector{SVector{ND,T}},
                             ng::SVector{ND,Int},
                             domain_lo::SVector{ND,T},
                             bin_size::T,
                             lbin_lo::SVector{ND,Int},
                             lng::SVector{ND,Int}) where {ND, T<:AbstractFloat}
    nlocal  = length(x)
    strides = MVector{ND,Int}(undef)
    strides[1] = 1
    for d in 2:ND
        strides[d] = strides[d-1] * lng[d-1]
    end
    for i in 1:nlocal
        flat = 0
        for d in 1:ND
            b    = floor(Int, (x[i][d] - domain_lo[d]) / bin_size)
            b    = clamp(b, 0, ng[d] - 1)
            flat += (b - lbin_lo[d]) * strides[d]
        end
        hist[flat + 1] += 1   # 1-based
    end
end

# ---------------------------------------------------------------------------
# Project the node-local portion of the histogram onto one axis.
# proj[k+1] accumulates all bins at node-local axis position k.
# lo_clip / hi_clip restrict summed dimensions to a bounding box.
# proj_off shifts the axis index: proj[k - proj_off + 1] for axis bin k.
# ---------------------------------------------------------------------------
function _project_histogram!(proj::Vector{Int},
                              hist::Vector{Int},
                              lbin_lo::SVector{ND,Int},
                              lng::SVector{ND,Int},
                              lo_bin::SVector{ND,Int},
                              node_bins::SVector{ND,Int},
                              axis::Int,
                              lo_clip::SVector{ND,Int},
                              hi_clip::SVector{ND,Int},
                              proj_off::Int = 0) where {ND}
    strides = MVector{ND,Int}(undef)
    strides[1] = 1
    for d in 2:ND
        strides[d] = strides[d-1] * lng[d-1]
    end

    eff_lo = MVector{ND,Int}(lo_clip)
    eff_hi = MVector{ND,Int}(hi_clip)

    # Intersect with local histogram extent
    for d in 1:ND
        eff_lo[d] = max(eff_lo[d], lbin_lo[d] - lo_bin[d])
        eff_hi[d] = min(eff_hi[d], lbin_lo[d] + lng[d] - 1 - lo_bin[d])
    end

    for d in 1:ND
        eff_lo[d] > eff_hi[d] && return
    end

    # Mixed-radix iteration over clipped sub-region
    lm = MVector{ND,Int}(eff_lo)
    while true
        flat = 0
        for d in 1:ND
            flat += (lo_bin[d] + lm[d] - lbin_lo[d]) * strides[d]
        end
        proj[lm[axis] - proj_off + 1] += hist[flat + 1]

        d = 1
        while d <= ND
            lm[d] += 1
            lm[d] <= eff_hi[d] && break
            lm[d] = eff_lo[d]
            d += 1
        end
        d > ND && break
    end
end

# ---------------------------------------------------------------------------
# Timer stats output (rank 0 only).  Called only when TIMERS=true.
# tmr_local: vector of NTMR elapsed seconds on this rank.
# ---------------------------------------------------------------------------
const _TMR_NAMES = [
    "domain bounds  (Allreduce)",
    "histogram build            ",
    "DFS loop total             ",
    "  bb projection  (local)   ",
    "  bb MPI_Reduce            ",
    "  bb axis calc   (rank 0)  ",
    "  bb MPI_Bcast             ",
    "  1d projection  (local)   ",
    "  1d MPI_Reduce            ",
    "  1d median calc (rank 0)  ",
    "  1d MPI_Bcast             ",
]
const _NTMR = length(_TMR_NAMES)

function _print_timer_stats(tmr_local::Vector{Float64}, nprocs::Int, rank::Int,
                             comm::MPI.Comm)
    flat = MPI.Gather(tmr_local, 0, comm)
    rank != 0 && return

    @printf "[ORB timers] orb_decompose (ms) – stats over %d ranks:\n" nprocs
    @printf "  %-28s %10s %10s %10s %10s %10s\n" "timer" "min" "max" "median" "mean" "stdev"

    for ti in 1:_NTMR
        col = [flat[r * _NTMR + ti] for r in 0:nprocs-1]   # seconds
        mn  = minimum(col)
        mx  = maximum(col)
        μ   = sum(col) / nprocs
        σ   = sqrt(sum((v - μ)^2 for v in col) / nprocs)
        sort!(col)
        med = nprocs % 2 == 0 ?
              (col[nprocs÷2] + col[nprocs÷2 + 1]) * 0.5 :
              col[nprocs÷2 + 1]
        @printf "  %-28s %10.4f %10.4f %10.4f %10.4f %10.4f\n" _TMR_NAMES[ti] (mn*1e3) (mx*1e3) (med*1e3) (μ*1e3) (σ*1e3)
    end
end

# ---------------------------------------------------------------------------
# Run ORB decomposition.
#
# x        – Vector{SVector{ND,Float64}} of particle positions on this rank
# comm     – MPI communicator (default: MPI.COMM_WORLD)
#
# Returns (local_lo, local_hi) as SVector{ND,Float64}.
# ---------------------------------------------------------------------------
function orb_decompose(cfg::ORBConfig{ND, T, TIMERS}, x::Vector{SVector{ND,T}},
                       comm::MPI.Comm = MPI.COMM_WORLD) where {ND, T<:AbstractFloat, TIMERS}
    bin_size = cfg.bin_size
    rank     = MPI.Comm_rank(comm)
    nprocs   = MPI.Comm_size(comm)
    nlocal   = length(x)

    if TIMERS
        tmr_domain          = 0.0
        tmr_hist            = 0.0
        tmr_bb_proj         = 0.0
        tmr_bb_mpi_reduce   = 0.0
        tmr_bb_calc         = 0.0
        tmr_bb_bcast        = 0.0
        tmr_proj_compute    = 0.0
        tmr_proj_mpi_reduce = 0.0
        tmr_proj_calc       = 0.0
        tmr_proj_bcast      = 0.0
        t0 = MPI.Wtime()
    end

    # -----------------------------------------------------------------------
    # Determine global domain bounds
    # -----------------------------------------------------------------------
    if nlocal > 0
        local_lo = SVector{ND,T}(minimum(p[d] for p in x) for d in 1:ND)
        local_hi = SVector{ND,T}(maximum(p[d] for p in x) for d in 1:ND)
    else
        local_lo = SVector{ND,T}(fill( typemax(T), ND))
        local_hi = SVector{ND,T}(fill(-typemax(T), ND))
    end

    domain_lo = SVector{ND,T}(MPI.Allreduce(Vector(local_lo), MPI.MIN, comm))
    domain_hi = SVector{ND,T}(MPI.Allreduce(Vector(local_hi), MPI.MAX, comm))

    ng        = SVector{ND,Int}(ceil(Int, (domain_hi[d] - domain_lo[d]) / bin_size) for d in 1:ND)
    domain_hi = SVector{ND,T}(domain_lo[d] + ng[d] * bin_size for d in 1:ND)

    if TIMERS
        tmr_domain = MPI.Wtime() - t0
        t0 = MPI.Wtime()
    end

    # -----------------------------------------------------------------------
    # Build flat column-major histogram for local particles
    # -----------------------------------------------------------------------
    if nlocal > 0
        lbin_lo = SVector{ND,Int}(max(0, floor(Int, (minimum(p[d] for p in x) - domain_lo[d]) / bin_size)) for d in 1:ND)
        lbin_hi = SVector{ND,Int}(min(ng[d]-1, floor(Int, (maximum(p[d] for p in x) - domain_lo[d]) / bin_size)) for d in 1:ND)
        lng     = SVector{ND,Int}(lbin_hi[d] - lbin_lo[d] + 1 for d in 1:ND)
    else
        lbin_lo = zero(SVector{ND,Int})
        lng     = ones(SVector{ND,Int})
    end

    nhist      = prod(lng)
    local_hist = zeros(Int, nhist)
    nlocal > 0 && _build_local_hist!(local_hist, x, ng, domain_lo, bin_size, lbin_lo, lng)

    if TIMERS
        tmr_hist = MPI.Wtime() - t0
        t0 = MPI.Wtime()   # DFS loop start
    end

    # -----------------------------------------------------------------------
    # DFS traversal (deterministic on every rank via broadcast)
    # -----------------------------------------------------------------------
    local_lo_out = MVector{ND,T}(undef)
    local_hi_out = MVector{ND,T}(undef)

    root  = _StackEntry{ND}(zero(SVector{ND,Int}), ng .- 1, 0, nprocs - 1)
    stack = _StackEntry{ND}[root]

    local_bb_lo  = MVector{ND,Int}(undef)
    local_bb_hi  = MVector{ND,Int}(undef)
    global_bb_lo = MVector{ND,Int}(undef)
    global_bb_hi = MVector{ND,Int}(undef)

    while !isempty(stack)
        entry = pop!(stack)
        lo    = entry.lo
        hi    = entry.hi
        g_lo  = entry.g_lo
        g_hi  = entry.g_hi

        # Leaf: assign subdomain to the single owning rank
        if g_lo == g_hi
            if rank == g_lo
                for d in 1:ND
                    local_lo_out[d] = domain_lo[d] + T(lo[d]) * bin_size
                    local_hi_out[d] = domain_lo[d] + T(hi[d] + 1) * bin_size
                end
            end
            continue
        end

        node_bins = hi .- lo .+ 1
        g_mid     = g_lo + (g_hi - g_lo) ÷ 2
        n_left    = g_mid - g_lo + 1
        n_total   = g_hi  - g_lo + 1

        # --- Step 1: pick bisection axis via bounding box -------------------
        if TIMERS; t1 = MPI.Wtime(); end

        lo_clip = MVector{ND,Int}(zero(SVector{ND,Int}))
        hi_clip = MVector{ND,Int}(node_bins .- 1)

        for d in 1:ND
            nb         = node_bins[d]
            bb_clip_lo = max(0,    lbin_lo[d] - lo[d])
            bb_clip_hi = min(nb-1, lbin_lo[d] + lng[d] - 1 - lo[d])
            if bb_clip_lo > bb_clip_hi
                local_bb_lo[d] =  typemax(Int)
                local_bb_hi[d] = -typemax(Int)
                lo_clip[d] = 1;  hi_clip[d] = 0
                continue
            end
            clipped_nb = bb_clip_hi - bb_clip_lo + 1
            lproj_d    = zeros(Int, clipped_nb)
            lo_clip_d  = setindex(zero(SVector{ND,Int}), bb_clip_lo, d)
            hi_clip_d  = setindex(SVector{ND,Int}(node_bins .- 1), bb_clip_hi, d)
            _project_histogram!(lproj_d, local_hist, lbin_lo, lng,
                                 lo, node_bins, d, lo_clip_d, hi_clip_d, bb_clip_lo)
            first_occ = findfirst(>(0), lproj_d)
            last_occ  = findlast(>(0),  lproj_d)
            if first_occ !== nothing
                local_bb_lo[d] = bb_clip_lo + first_occ - 1
                local_bb_hi[d] = bb_clip_lo + last_occ  - 1
                lo_clip[d] = local_bb_lo[d]
                hi_clip[d] = local_bb_hi[d]
            else
                local_bb_lo[d] =  typemax(Int)
                local_bb_hi[d] = -typemax(Int)
                lo_clip[d] = 1;  hi_clip[d] = 0
            end
        end

        if TIMERS; tmr_bb_proj += MPI.Wtime() - t1; t1 = MPI.Wtime(); end

        tmp_lo = MPI.Reduce(Vector(local_bb_lo), MPI.MIN, 0, comm)
        tmp_hi = MPI.Reduce(Vector(local_bb_hi), MPI.MAX, 0, comm)
        rank == 0 && (global_bb_lo .= tmp_lo; global_bb_hi .= tmp_hi)

        if TIMERS; tmr_bb_mpi_reduce += MPI.Wtime() - t1; t1 = MPI.Wtime(); end

        bisect_axis = 1
        if rank == 0
            best_extent = -1
            for d in 1:ND
                if global_bb_hi[d] >= global_bb_lo[d]
                    ext = global_bb_hi[d] - global_bb_lo[d]
                    if ext > best_extent
                        best_extent = ext
                        bisect_axis = d
                    end
                end
            end
        end

        if TIMERS; tmr_bb_calc += MPI.Wtime() - t1; t1 = MPI.Wtime(); end

        bisect_axis = MPI.bcast(bisect_axis, 0, comm)

        if TIMERS; tmr_bb_bcast += MPI.Wtime() - t1; end

        # --- Step 2: find the median bin along bisect_axis ------------------
        nb = node_bins[bisect_axis]
        nb < 2 && error("[ORB] axis $bisect_axis has only $nb bin(s) but node [g_lo=$g_lo, g_hi=$g_hi] must be split")

        if TIMERS; t1 = MPI.Wtime(); end

        lp = zeros(Int, nb)
        lc = MVector{ND,Int}(lo_clip);  lc[bisect_axis] = 0
        hc = MVector{ND,Int}(hi_clip);  hc[bisect_axis] = nb - 1
        _project_histogram!(lp, local_hist, lbin_lo, lng,
                             lo, node_bins, bisect_axis, SVector(lc), SVector(hc))

        if TIMERS; tmr_proj_compute += MPI.Wtime() - t1; t1 = MPI.Wtime(); end

        gp = MPI.Reduce(lp, MPI.SUM, 0, comm)

        if TIMERS; tmr_proj_mpi_reduce += MPI.Wtime() - t1; t1 = MPI.Wtime(); end

        bisect_bin = lo[bisect_axis] + (nb - 1) ÷ 2   # balanced fallback
        if rank == 0
            total = Int64(sum(gp))
            cum   = Int64(0)
            for i in 1:nb
                cum += Int64(gp[i])
                if cum * n_total >= total * n_left
                    bisect_bin = lo[bisect_axis] + i - 1
                    break
                end
            end
            bisect_bin = clamp(bisect_bin, lo[bisect_axis], hi[bisect_axis] - 1)
        end

        if TIMERS; tmr_proj_calc += MPI.Wtime() - t1; t1 = MPI.Wtime(); end

        bisect_bin = MPI.bcast(bisect_bin, 0, comm)

        if TIMERS; tmr_proj_bcast += MPI.Wtime() - t1; end

        # --- Push children (right first so left is processed first) ---------
        lo_right = setindex(lo, bisect_bin + 1, bisect_axis)
        hi_left  = setindex(hi, bisect_bin,     bisect_axis)
        push!(stack, _StackEntry{ND}(lo_right, hi,      g_mid + 1, g_hi))
        push!(stack, _StackEntry{ND}(lo,       hi_left, g_lo,      g_mid))
    end

    if TIMERS
        tmr_dfs_total = MPI.Wtime() - t0
        tmr_local = Float64[tmr_domain, tmr_hist, tmr_dfs_total,
                            tmr_bb_proj, tmr_bb_mpi_reduce, tmr_bb_calc, tmr_bb_bcast,
                            tmr_proj_compute, tmr_proj_mpi_reduce, tmr_proj_calc, tmr_proj_bcast]
        _print_timer_stats(tmr_local, nprocs, rank, comm)
    end

    return SVector(local_lo_out), SVector(local_hi_out)
end
