export SystemInteraction, is_coupled, create_grid!, sweep!

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

"""
    SystemInteraction{T,ND,KT,SA,SB}

Manages pairwise interactions between one or two particle systems using a
cell list (spatial hash grid).

- **Self-interaction** (`system_b = nothing`): half-shell Newton's-3rd-law sweep.
- **Coupled interaction** (`system_b` is a `ParticleSystem`): full 3×3 neighbour
  sweep of all `system_b` particles within each `system_a` particle's
  neighbourhood.

The cell list uses a CSR (Compressed Sparse Row) layout: `_cell_start[c]` is the
index of the first particle in cell `c` (0 if empty); `_cell_count[c]` is the
number of particles in cell `c`.  This requires that particles are sorted by cell
before `create_grid!` is called (handled by `sort_particles!` in the time loop).
"""
struct SystemInteraction{T<:AbstractFloat, ND, KT<:AbstractKernel{T,ND}, SA<:AbstractParticleSystem{T,ND}, SB<:Union{Nothing,AbstractParticleSystem{T,ND}}, PFNS<:Tuple}
    kernel::KT
    pfns::PFNS
    system_a::SA
    system_b::SB
    _mingridx::MVector{ND,T}            # grid origin in each dimension (ndims,)
    _ngridx::MVector{ND,Int}            # number of cells in each dimension (ndims,)
    _cell_start::Vector{Int}            # first system_b particle in cell c (ncells,); 0 if empty
    _cell_count::Vector{Int}            # number of system_b particles in cell c (ncells,)
    _cell_start_a::Vector{Int}          # first system_a particle in cell c (ncells,); 0 if empty (coupled only)
    _cell_count_a::Vector{Int}          # number of system_a particles in cell c (ncells,) (coupled only)
    _mingridx_a::MVector{ND,T}          # min position of system_a particles per dim (ndims,)
    _maxgridx_a::MVector{ND,T}          # max position of system_a particles per dim (ndims,)
    _cell_size::T
    function SystemInteraction{T, ND, KT, SA, SB, PFNS}(args...) where {T, ND, KT, SA, SB, PFNS}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, KT, SA, SB, PFNS}(args...)
    end
end

@inline function Base.getproperty(ps::SystemInteraction{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(ps, s)
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    SystemInteraction(kernel, pairwise_fn, system_a[, system_b]) -> SystemInteraction

Construct a `SystemInteraction`.

- Omit `system_b` (or pass `nothing`) for a self-interaction sweep.
- Pass `system_b` for a coupled (e.g. fluid–boundary) sweep.

Raises `ArgumentError` if `system_b.ndims ≠ system_a.ndims`.
"""
function SystemInteraction(
    kernel::AbstractKernel,
    pairwise_fn,
    system_a::AbstractParticleSystem{T, ND},
    system_b::Union{Nothing, AbstractParticleSystem} = nothing
) where {T<:AbstractFloat, ND}
    nd = Int(ND)
    kernel.ndims == nd || throw(ArgumentError(
        "kernel.ndims ($(kernel.ndims)) must match system_a.ndims ($nd)"))
    system_b !== nothing && system_b.ndims != nd && throw(ArgumentError(
        "system_b.ndims ($(system_b.ndims)) must match system_a.ndims ($nd)"))
    pfns = pairwise_fn isa Tuple ? pairwise_fn : (pairwise_fn,)
    _check_functors_eltype(pfns, T, "pairwise functor")
    SystemInteraction{
        T, nd, typeof(kernel),
        typeof(system_a), typeof(system_b), typeof(pfns)
    }(
        kernel,
        pfns,
        system_a,
        system_b,
        MVector{nd,T}(undef),
        MVector{nd,Int}(undef),
        Vector{Int}(),
        Vector{Int}(),
        Vector{Int}(),
        Vector{Int}(),
        MVector{nd,T}(undef),
        MVector{nd,T}(undef),
        T(kernel.interaction_length),
    )
end

# ---------------------------------------------------------------------------
# Coupling predicate
# ---------------------------------------------------------------------------

"""
    is_coupled(si::SystemInteraction) -> Bool

Return `true` when `si` has a secondary `system_b`.
"""
is_coupled(si::SystemInteraction) = si.system_b !== nothing

# ---------------------------------------------------------------------------
# Cell list construction
# ---------------------------------------------------------------------------

"""
    create_grid!(si::SystemInteraction)

Build the CSR cell list used by `sweep!`.

**Requires pre-sorted particles**: `sort_particles!` must be called on both
`system_a` and `system_b` (if present) before this function, so that each
cell's particles occupy a contiguous index range.

Updates `si._mingridx`, `si._ngridx`, `si._cell_start`, and `si._cell_count`.

The grid origin is extended by 2h beyond all particle positions so that
neighbour-cell accesses in the sweep never go out of bounds.
"""
function create_grid!(si::SystemInteraction{T}) where {T}
    cutoff = T(si.kernel.interaction_length)
    _create_grid_impl!(si, si.system_b, cutoff)
    nothing
end

# --- self-interaction ---

function _create_grid_impl!(si::SystemInteraction{T}, ::Nothing, cutoff::T) where {T}
    xa = si.system_a.x
    mn, mx = reduce(xa; init=(xa[1], xa[1])) do (mn, mx), x
        (min.(mn, x), max.(mx, x))
    end
    # Snap mingridx down to the nearest multiple of cutoff below the padded minimum.
    # Since cutoff is identical for all interactions, every snapped origin is an integer
    # multiple of cutoff from 0, so cell boundaries align across all grids in the
    # simulation — a prerequisite for consistent particle sorting by cell index.
    mingridx = map(v -> floor(v / cutoff) * cutoff, mn .- 2*cutoff)
    maxgridx = mx .+ 2*cutoff
    _setup_cell_arrays!(si, mingridx, maxgridx, cutoff)
    _populate_cells_self!(si, cutoff)
end

# --- coupled interaction ---

function _create_grid_impl!(si::SystemInteraction{T}, system_b, cutoff::T) where {T}
    xa = si.system_a.x
    xb = system_b.x
    mn_a, mx_a = reduce(xa; init=(xa[1], xa[1])) do (mn, mx), x
        (min.(mn, x), max.(mx, x))
    end
    mn, mx = reduce(xb; init=(mn_a, mx_a)) do (mn, mx), x
        (min.(mn, x), max.(mx, x))
    end
    mingridx = map(v -> floor(v / cutoff) * cutoff, mn .- 2*cutoff)
    maxgridx = mx .+ 2*cutoff
    _setup_cell_arrays!(si, mingridx, maxgridx, cutoff; coupled=true)
    si._mingridx_a .= mn_a
    si._maxgridx_a .= mx_a
    _populate_cells_a!(si, cutoff)
    _populate_cells_b!(si, cutoff)
end

# --- shared setup ---

function _setup_cell_arrays!(si::SystemInteraction{T, ND}, mingridx::SVector{ND, T}, maxgridx::SVector{ND, T}, cutoff::T; coupled::Bool=false) where {T, ND}
    si._mingridx .= mingridx
    @. si._ngridx = Int(floor((maxgridx - mingridx) / cutoff)) + Int(1)
    ncells = Int(prod(si._ngridx))
    resize!(si._cell_start, ncells); fill!(si._cell_start, 0)
    resize!(si._cell_count, ncells); fill!(si._cell_count, 0)
    if coupled
        resize!(si._cell_start_a, ncells); fill!(si._cell_start_a, 0)
        resize!(si._cell_count_a, ncells); fill!(si._cell_count_a, 0)
    end
end

# Compute the 1-indexed flat cell for particle i in a Vector{SVector} x.
# Mimics the Python row-major loop:  for d in ndims-1 downto 0: flat += cell_d * stride
@inline function _cell_1idx(x::SVector{ND,T}, mingridx::AbstractVector{T}, dcell::T, ngridx::AbstractVector{Int}, ::Val{ND}) where {ND,T}
    flat   = 0
    stride = 1
    for d in ND:-1:1
        cell_d = floor(Int, (x[d] - mingridx[d]) / dcell)
        flat  += cell_d * stride
        stride *= ngridx[d]
    end
    return flat + 1          # convert to 1-indexed
end

# Sequential scan over pre-sorted particles to build the CSR cell arrays.
# Assumes x is sorted so that all particles in the same cell are contiguous.
function _populate_cells_sorted!(cell_start::Vector{Int}, cell_count::Vector{Int},
                                  x, mingridx, cutoff, ngridx, vnd)
    prev_cell = 0
    @inbounds for i in eachindex(x)
        cell = _cell_1idx(x[i], mingridx, cutoff, ngridx, vnd)
        if cell != prev_cell
            cell_start[cell] = i
            prev_cell = cell
        end
        cell_count[cell] += 1
    end
end

function _populate_cells_self!(si::SystemInteraction{T,ND}, cutoff::T) where {T,ND}
    _populate_cells_sorted!(si._cell_start, si._cell_count,
                            si.system_a.x, si._mingridx, cutoff, si._ngridx, Val{ND}())
end

function _populate_cells_a!(si::SystemInteraction{T,ND}, cutoff::T) where {T,ND}
    _populate_cells_sorted!(si._cell_start_a, si._cell_count_a,
                            si.system_a.x, si._mingridx, cutoff, si._ngridx, Val{ND}())
end

function _populate_cells_b!(si::SystemInteraction{T,ND}, cutoff::T) where {T,ND}
    _populate_cells_sorted!(si._cell_start, si._cell_count,
                            si.system_b.x, si._mingridx, cutoff, si._ngridx, Val{ND}())
end

# ---------------------------------------------------------------------------
# Pairwise sweep
# ---------------------------------------------------------------------------

"""
    sweep!(si::SystemInteraction)

Run the pairwise sweep over all particle pairs within the interaction cutoff,
using the cell list built by the most recent `create_grid!` call.

The time integrator calls `create_grid!(si)` then `sweep!(si)` each step.

Pairwise function signature (self and coupled):

    pairwise_fn(i, j, dx::SVector, gx::SVector, w)

`dx = xi - xj`; `gx` is the kernel gradient vector; `w` is the kernel value.

The pfn is a closure that captures the particle arrays it needs upfront. Use a
factory so the captured locals are loop-invariant from the compiler's perspective:

    function make_pfn(ps)                       # self-interaction
        v = ps.v; dvdt = ps.dvdt; ...
        @inline @Base.propagate_inbounds function(i, j, dx, gx, w)
            ...
        end
    end
    si = SystemInteraction(kernel, make_pfn(ps), ps)

    function make_pfn(ps_a, ps_b)               # coupled interaction
        v_a = ps_a.v; dvdt_a = ps_a.dvdt; v_b = ps_b.v; ...
        @inline @Base.propagate_inbounds function(i, j, dx, gx, w)
            ...
        end
    end
    si = SystemInteraction(kernel, make_pfn(ps_a, ps_b), ps_a, ps_b)

Indices `i`, `j` are 1-based (Julia convention).
"""
sweep!(si::SystemInteraction, stage::Int) = _sweep_pfns!(si, si.system_b, si.pfns, stage)
sweep!(si::SystemInteraction) = sweep!(si, 1)

# Type-stable tuple walk: peel off one pfn at a time so `first(pfns)` is always
# a concrete type, avoiding the Union{A,B} that si.pfns[stage::Int] would produce.
_sweep_pfns!(si, system_b, ::Tuple{}, stage) = nothing
@inline function _sweep_pfns!(si, system_b, pfns::Tuple, stage)
    if stage == 1
        _sweep_dispatch!(si, system_b, first(pfns))
    else
        _sweep_pfns!(si, system_b, Base.tail(pfns), stage - 1)
    end
end

_sweep_dispatch!(si, ::Nothing, pfn) = _sweep_self!(si, pfn)
_sweep_dispatch!(si, system_b, pfn) = _sweep_coupled!(si, system_b, pfn)

# --- self-interaction half-shell, 6-colour sweep ---
#
# HALF-SHELL STENCIL
# ==================
# For each cell C (marked with *), we process particle pairs that involve C
# and one of the 4 "forward" neighbours (marked F).  The remaining 4
# neighbours (marked .) are handled when those cells take their turn as C,
# so every pair is counted exactly once.
#
#   cj →
#   F F F      The cell above-left (-1,+1) is included so that the full
#   . * F      interaction radius is covered: a particle near the +x edge of
#   . . .      C can reach a particle in the (+1,-1) cell (lower-right), which
#              is the mirror of (-1,+1) seen from that neighbour.
#
# Forward offsets (Δci, Δcj):  (+1,0), (-1,+1), (0,+1), (+1,+1)
#
# Same-cell pairs are handled with nested for loops (jj > ii) so each
# (ii,jj) pair is visited once.
#
# COLOURING FOR CONFLICT-FREE PARALLEL EXECUTION
# ===============================================
# apply_pair accumulates into BOTH particles i and j.  Two cells A and B
# conflict if any particle in A's write-set overlaps B's write-set.
#
# Cell (ci,cj) processes pairs with itself and its 4 forward neighbours.
# The forward stencil only goes to Δcj ∈ {0, +1} — never to Δcj=-1,
# because backward neighbours (Δcj=-1) are owned by those cells themselves.
# So the write-set is an asymmetric 3×2 block, not a symmetric 3×3:
#
#   Δci ∈ {-1, 0, +1},  Δcj ∈ {0, +1}
#
# If the stencil were symmetric (Δcj ∈ {-1,0,+1}), the write-set height
# would be 3, conflicts would span |Δcj| ≤ 2, and we would need stride-3
# in cj → 9 colours total.  The half-shell halves the cj stride to 2.
#
# Two cells (ci,cj) and (ci',cj') conflict when:
#   |ci - ci'| ≤ 2  AND  |cj - cj'| ≤ 1
#
# Choosing stride-3 in ci and stride-2 in cj guarantees no two active cells
# within a colour are closer than that, so they are write-conflict-free:
#
#   colour = (ci-1)%3 * 2 + (cj-1)%2,   colours 0..5
#
# Visualised on a 6×4 grid (letter = colour, active cells per colour shown):
#
#   cj:   1  2  3  4
#   ci 1: A  B  A  B
#   ci 2: C  D  C  D
#   ci 3: E  F  E  F
#   ci 4: A  B  A  B
#   ci 5: C  D  C  D
#   ci 6: E  F  E  F
#
# Within a single colour all active cells are separated by ≥3 in ci and ≥2
# in cj — safely beyond the conflict radius — so Threads.@threads over the
# ci loop is race-free.

function _sweep_self!(si::SystemInteraction, ::Nothing)
end

# 2D Specialisation: 6-colour sweep
function _sweep_self!(si::SystemInteraction{T,2}, pfn::PFN) where {T,PFN}
    ps_a       = si.system_a
    kernel     = si.kernel
    h          = T(kernel.h)
    cutoff_sq  = si._cell_size * si._cell_size
    cell_start = si._cell_start::Vector{Int}
    cell_count = si._cell_count::Vector{Int}
    nc_y       = Int(si._ngridx[2])
    nc_x       = Int(si._ngridx[1])
    vnd        = Val{2}()

    # Forward neighbour offsets (2D): (+1,0), (-1,+1), (0,+1), (+1,+1)
    forward_dflat = (nc_y, -nc_y+1, 1, nc_y+1)

    for colour in 0:5
        ci_start = colour ÷ 2 + 1
        cj_start = colour % 2 + 1
        nci = length(ci_start:3:nc_x)
        ncj = length(cj_start:2:nc_y)
        @inbounds @batch for idx in 1:nci*ncj
            i_ci, i_cj = divrem(idx - 1, ncj)
            ci   = ci_start + i_ci * 3
            cj   = cj_start + i_cj * 2
            flat = (ci - 1) * nc_y + cj
            s    = cell_start[flat]
            cnt  = cell_count[flat]

            # Same-cell half-shell (ii < jj, so each pair visited once)
            for ii in s:s+cnt-2
                for jj in ii+1:s+cnt-1
                    _pair_self!(pfn, ps_a, ii, jj, kernel, h, cutoff_sq, vnd)
                end
            end

            # Cross-cell forward neighbours — skip empty cells to avoid
            # computing out-of-bounds nflat indices for boundary cells.
            if cnt > 0
                for k in 1:4
                    nflat = flat + forward_dflat[k]
                    ns   = cell_start[nflat]
                    ncnt = cell_count[nflat]
                    for ii in s:s+cnt-1
                        for jj in ns:ns+ncnt-1
                            _pair_self!(pfn, ps_a, ii, jj, kernel, h, cutoff_sq, vnd)
                        end
                    end
                end
            end
        end
    end
end

# 3D Specialisation: 18-colour sweep
#
# HALF-SHELL STENCIL (3D)
# =======================
# 13 forward neighbours — all (Δci,Δcj,Δck) lexicographically positive:
#   (0,0,+1), (0,+1,{-1,0,+1}), (+1,{-1,0,+1},{-1,0,+1})
#
# WRITE-SET EXTENT AND REQUIRED STRIDES
# ======================================
# ci write-set: {0,+1}     → conflict range |Δci| ≤ 1 → stride ≥ 2
# cj write-set: {-1,0,+1}  → conflict range |Δcj| ≤ 2 → stride ≥ 3
# ck write-set: {-1,0,+1}  → conflict range |Δck| ≤ 2 → stride ≥ 3
#
# 2 × 3 × 3 = 18 colours total.
#   colour = (ci-1)%2 * 9 + (cj-1)%3 * 3 + (ck-1)%3
function _sweep_self!(si::SystemInteraction{T,3}, pfn::PFN) where {T,PFN}
    ps_a       = si.system_a
    kernel     = si.kernel
    h          = T(kernel.h)
    cutoff_sq  = si._cell_size * si._cell_size
    cell_start = si._cell_start::Vector{Int}
    cell_count = si._cell_count::Vector{Int}
    nc_z       = Int(si._ngridx[3])
    nc_y       = Int(si._ngridx[2])
    nc_x       = Int(si._ngridx[1])
    ncyz       = nc_y * nc_z
    vnd        = Val{3}()

    # Forward neighbour offsets (3D, 13 neighbors)
    forward_dflat = (
        1,                                    # ( 0,  0, +1)
        nc_z-1, nc_z, nc_z+1,                 # ( 0, +1, -1), ( 0, +1, 0), ( 0, +1, +1)
        ncyz-nc_z-1, ncyz-nc_z, ncyz-nc_z+1,  # (+1, -1, -1), (+1, -1, 0), (+1, -1, +1)
        ncyz-1,      ncyz,      ncyz+1,       # (+1,  0, -1), (+1,  0, 0), (+1,  0, +1)
        ncyz+nc_z-1, ncyz+nc_z, ncyz+nc_z+1   # (+1, +1, -1), (+1, +1, 0), (+1, +1, +1)
    )

    for colour in 0:17
        ci_start = (colour ÷ 9) + 1
        cj_start = ((colour % 9) ÷ 3) + 1
        ck_start = (colour % 3) + 1
        nci  = length(ci_start:2:nc_x)
        ncj  = length(cj_start:3:nc_y)
        nck  = length(ck_start:3:nc_z)
        ncjk = ncj * nck
        @inbounds @batch for idx in 1:nci*ncjk
            i_ci, rem_jk = divrem(idx - 1, ncjk)
            i_cj, i_ck   = divrem(rem_jk, nck)
            ci   = ci_start + i_ci * 2
            cj   = cj_start + i_cj * 3
            ck   = ck_start + i_ck * 3
            flat = (ci - 1) * ncyz + (cj - 1) * nc_z + ck
            s    = cell_start[flat]
            cnt  = cell_count[flat]

            # Same-cell half-shell (ii < jj)
            for ii in s:s+cnt-2
                for jj in ii+1:s+cnt-1
                    _pair_self!(pfn, ps_a, ii, jj, kernel, h, cutoff_sq, vnd)
                end
            end

            # Forward neighbors — skip empty cells to avoid out-of-bounds nflat.
            if cnt > 0
                for k in 1:13
                    nflat = flat + forward_dflat[k]
                    ns   = cell_start[nflat]
                    ncnt = cell_count[nflat]
                    for ii in s:s+cnt-1
                        for jj in ns:ns+ncnt-1
                            _pair_self!(pfn, ps_a, ii, jj, kernel, h, cutoff_sq, vnd)
                        end
                    end
                end
            end
        end
    end
end

@inline function _pair_self!(pfn, ps_a, i::Int, j::Int, kernel::AbstractKernel{T,ND}, h::T, cutoff_sq::T, ::Val{ND}) where {T,ND}
    @inbounds begin
        xa = ps_a.x
        dx   = xa[i] - xa[j]
        r_sq = dot(dx, dx)
        if r_sq < cutoff_sq
            r          = sqrt(r_sq)
            q          = r / h
            grad_coeff = kernel_dw_dq(kernel, q) / (r * h)
            w          = kernel_w(kernel, q)
            gx         = grad_coeff * dx
            pfn(ps_a, i, j, dx, gx, w)
        end
    end
end

# --- coupled interaction: 9-colour cell sweep ---
#
# COLOURING FOR COUPLED SWEEP
# ===========================
# pfn may write to both system_a particle i and system_b particle j.
# Processing cell C_a touches:
#   - system_a particles in C_a                (Δci,Δcj = 0)
#   - system_b particles in the full 3×3 block (Δci,Δcj ∈ {-1,0,+1})
#
# Two cells conflict when their 3×3 write-regions overlap:
#   |ci - ci'| ≤ 2  AND  |cj - cj'| ≤ 2
#
# The neighbourhood is symmetric (unlike the half-shell), so both axes need
# stride-3 → 9 colours (vs 6 for the self-sweep):
#
#   colour = (ci-1)%3 * 3 + (cj-1)%3,   colours 0..8
#
# The loop iterates over cells of system_a using the _cell_start_a/_cell_count_a
# CSR arrays built by create_grid!.

function _sweep_coupled!(si::SystemInteraction, system_b, ::Nothing)
end

# 2D Specialisation: 9-colour sweep
function _sweep_coupled!(si::SystemInteraction{T,2}, system_b, pfn::PFN) where {T,PFN}
    ps_a          = si.system_a
    kernel        = si.kernel
    h             = T(kernel.h)
    cutoff_sq     = si._cell_size * si._cell_size
    cell_start    = si._cell_start::Vector{Int}
    cell_count    = si._cell_count::Vector{Int}
    cell_start_a  = si._cell_start_a::Vector{Int}
    cell_count_a  = si._cell_count_a::Vector{Int}
    nc_y          = Int(si._ngridx[2])
    nc_x          = Int(si._ngridx[1])
    vnd           = Val{2}()
    cutoff        = si._cell_size
    mingridx      = si._mingridx
    ci_lo = floor(Int, (si._mingridx_a[1] - mingridx[1]) / cutoff) + 1
    ci_hi = floor(Int, (si._maxgridx_a[1] - mingridx[1]) / cutoff) + 1
    cj_lo = floor(Int, (si._mingridx_a[2] - mingridx[2]) / cutoff) + 1
    cj_hi = floor(Int, (si._maxgridx_a[2] - mingridx[2]) / cutoff) + 1

    for colour in 0:8
        ci_start = colour ÷ 3 + ci_lo
        cj_start = colour % 3 + cj_lo
        nci = length(ci_start:3:ci_hi)
        ncj = length(cj_start:3:cj_hi)
        @inbounds @batch for idx in 1:nci*ncj
            i_ci, i_cj = divrem(idx - 1, ncj)
            ci   = ci_start + i_ci * 3
            cj   = cj_start + i_cj * 3
            flat = (ci - 1) * nc_y + cj
            sa   = cell_start_a[flat]
            cna  = cell_count_a[flat]
            for ii in sa:sa+cna-1
                for dxcell in -1:1
                    for dycell in -1:1
                        nidx = flat + dxcell * nc_y + dycell
                        sb   = cell_start[nidx]
                        cnb  = cell_count[nidx]
                        for jj in sb:sb+cnb-1
                            _pair_coupled!(pfn, ps_a, system_b, ii, jj, kernel, h, cutoff_sq, vnd)
                        end
                    end
                end
            end
        end
    end
end

# 3D Specialisation: 27-colour sweep
function _sweep_coupled!(si::SystemInteraction{T,3}, system_b, pfn::PFN) where {T,PFN}
    ps_a          = si.system_a
    kernel        = si.kernel
    h             = T(kernel.h)
    cutoff_sq     = si._cell_size * si._cell_size
    cell_start    = si._cell_start::Vector{Int}
    cell_count    = si._cell_count::Vector{Int}
    cell_start_a  = si._cell_start_a::Vector{Int}
    cell_count_a  = si._cell_count_a::Vector{Int}
    nc_z          = Int(si._ngridx[3])
    nc_y          = Int(si._ngridx[2])
    nc_x          = Int(si._ngridx[1])
    ncyz          = nc_y * nc_z
    vnd           = Val{3}()
    cutoff        = si._cell_size
    mingridx      = si._mingridx
    ci_lo = floor(Int, (si._mingridx_a[1] - mingridx[1]) / cutoff) + 1
    ci_hi = floor(Int, (si._maxgridx_a[1] - mingridx[1]) / cutoff) + 1
    cj_lo = floor(Int, (si._mingridx_a[2] - mingridx[2]) / cutoff) + 1
    cj_hi = floor(Int, (si._maxgridx_a[2] - mingridx[2]) / cutoff) + 1
    ck_lo = floor(Int, (si._mingridx_a[3] - mingridx[3]) / cutoff) + 1
    ck_hi = floor(Int, (si._maxgridx_a[3] - mingridx[3]) / cutoff) + 1

    for colour in 0:26
        ci_start = (colour ÷ 9) + ci_lo
        cj_start = ((colour % 9) ÷ 3) + cj_lo
        ck_start = (colour % 3) + ck_lo
        nci  = length(ci_start:3:ci_hi)
        ncj  = length(cj_start:3:cj_hi)
        nck  = length(ck_start:3:ck_hi)
        ncjk = ncj * nck
        @inbounds @batch for idx in 1:nci*ncjk
            i_ci, rem_jk = divrem(idx - 1, ncjk)
            i_cj, i_ck   = divrem(rem_jk, nck)
            ci   = ci_start + i_ci * 3
            cj   = cj_start + i_cj * 3
            ck   = ck_start + i_ck * 3
            flat = (ci - 1) * ncyz + (cj - 1) * nc_z + ck
            sa   = cell_start_a[flat]
            cna  = cell_count_a[flat]
            for ii in sa:sa+cna-1
                for dxcell in -1:1
                    for dycell in -1:1
                        for dzcell in -1:1
                            nidx = flat + dxcell * ncyz + dycell * nc_z + dzcell
                            sb   = cell_start[nidx]
                            cnb  = cell_count[nidx]
                            for jj in sb:sb+cnb-1
                                _pair_coupled!(pfn, ps_a, system_b, ii, jj, kernel, h, cutoff_sq, vnd)
                            end
                        end
                    end
                end
            end
        end
    end
end

@inline function _pair_coupled!(pfn, ps_a, ps_b, i::Int, j::Int, kernel::AbstractKernel{T,ND}, h::T, cutoff_sq::T, ::Val{ND}) where {T,ND}
    @inbounds begin
        xa = ps_a.x
        xb = ps_b.x
        dx   = xa[i] - xb[j]
        r_sq = dot(dx, dx)
        if r_sq < cutoff_sq
            r          = sqrt(r_sq)
            q          = r / h
            grad_coeff = kernel_dw_dq(kernel, q) / (r * h)
            w          = kernel_w(kernel, q)
            gx         = grad_coeff * dx
            pfn(ps_a, ps_b, i, j, dx, gx, w)
        end
    end
end
