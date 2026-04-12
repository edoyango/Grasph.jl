export SystemInteraction, is_coupled, create_grid!, sweep!, adjust_v!

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
struct SystemInteraction{T<:AbstractFloat, ND, KT<:AbstractKernel{T,ND}, SA<:AbstractParticleSystem{T,ND}, SB<:Union{Nothing,AbstractParticleSystem{T,ND}}, PFNS<:Tuple, VPFN}
    kernel::KT
    pfns::PFNS
    vadjust_pfn::VPFN
    system_a::SA
    system_b::SB
    _mingridx::MVector{ND,T}            # grid origin in each dimension (ndims,)
    _ngridx::MVector{ND,Int}            # number of cells in each dimension (ndims,)
    _cell_start::Vector{Int}            # CSR prefix-sum: cell_start[c]..cell_start[c+1]-1 is the range of system_b particles in cell c; length = ncells+1
    _cell_start_a::Vector{Int}          # same for system_a (coupled only); length = ncells+1
    _mingridx_a::MVector{ND,T}          # min position of system_a particles per dim (ndims,)
    _maxgridx_a::MVector{ND,T}          # max position of system_a particles per dim (ndims,)
    _cell_size::T
    function SystemInteraction{T, ND, KT, SA, SB, PFNS, VPFN}(args...) where {T, ND, KT, SA, SB, PFNS, VPFN}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, KT, SA, SB, PFNS, VPFN}(args...)
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
    system_b::Union{Nothing, AbstractParticleSystem} = nothing;
    velocity_adjust_pairwise_fn = nothing
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
        typeof(system_a), typeof(system_b), typeof(pfns), typeof(velocity_adjust_pairwise_fn)
    }(
        kernel,
        pfns,
        velocity_adjust_pairwise_fn,
        system_a,
        system_b,
        MVector{nd,T}(undef),
        MVector{nd,Int}(undef),
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
    mn = xa[1]; mx = xa[1]
    @inbounds for i in 2:length(xa)
        xi = xa[i]
        mn = min.(mn, xi)
        mx = max.(mx, xi)
    end
    # Snap mingridx down to the nearest multiple of cutoff below the padded minimum.
    # Since cutoff is identical for all interactions, every snapped origin is an integer
    # multiple of cutoff from 0, so cell boundaries align across all grids in the
    # simulation — a prerequisite for consistent particle sorting by cell index.
    mingridx = _snap_to_grid(mn .- 2*cutoff, cutoff)
    maxgridx = mx .+ 2*cutoff
    _setup_cell_arrays!(si, mingridx, maxgridx, cutoff)
    _populate_cells_self!(si, cutoff)
end

# --- coupled interaction ---

function _create_grid_impl!(si::SystemInteraction{T}, system_b, cutoff::T) where {T}
    xa = si.system_a.x
    xb = system_b.x
    mn_a = xa[1]; mx_a = xa[1]
    @inbounds for i in 2:length(xa)
        xi = xa[i]
        mn_a = min.(mn_a, xi)
        mx_a = max.(mx_a, xi)
    end
    mn = mn_a; mx = mx_a
    @inbounds for i in eachindex(xb)
        xi = xb[i]
        mn = min.(mn, xi)
        mx = max.(mx, xi)
    end
    mingridx = _snap_to_grid(mn .- 2*cutoff, cutoff)
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
    # Length ncells+1: cell_start[c]..cell_start[c+1]-1 is the particle range for cell c.
    resize!(si._cell_start, ncells + 1); fill!(si._cell_start, 0)
    if coupled
        resize!(si._cell_start_a, ncells + 1); fill!(si._cell_start_a, 0)
    end
end

# Snap each element of `raw` down to the nearest grid-aligned multiple of cutoff.
# Named @inline function avoids the heap-allocated closure that `map(v -> ..., raw)`
# would create when `cutoff` is a captured local variable.
@inline function _snap_to_grid(raw::SVector{ND,T}, cutoff::T) where {ND,T}
    SVector(ntuple(d -> floor(raw[d] / cutoff) * cutoff, Val(ND)))
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

# Sequential scan over pre-sorted particles to build the CSR prefix-sum array.
# Assumes x is sorted so that all particles in the same cell are contiguous.
# cell_start must have length ncells+1 and be zero-initialised before the call.
# After the call:
#   cell_start[c] .. cell_start[c+1]-1  is the (inclusive) range of particles in cell c.
#   Empty cells satisfy cell_start[c] == cell_start[c+1].
function _populate_cells_sorted!(cell_start::Vector{Int},
                                  x, mingridx, cutoff, ngridx, vnd)
    ncells = length(cell_start) - 1
    n      = length(x)
    # Forward pass: record the first particle index for each non-empty cell.
    prev_cell = 0
    @inbounds for i in eachindex(x)
        cell = _cell_1idx(x[i], mingridx, cutoff, ngridx, vnd)
        if cell != prev_cell
            cell_start[cell] = i
            prev_cell = cell
        end
    end
    # Sentinel: one past the last particle index.
    cell_start[ncells + 1] = n + 1
    # Backward pass: propagate non-zero starts rightward so that empty cells
    # satisfy cell_start[c] == cell_start[c+1] (empty range).
    @inbounds for c in ncells:-1:1
        if cell_start[c] == 0
            cell_start[c] = cell_start[c + 1]
        end
    end
end

function _populate_cells_self!(si::SystemInteraction{T,ND}, cutoff::T) where {T,ND}
    _populate_cells_sorted!(si._cell_start,
                            si.system_a.x, si._mingridx, cutoff, si._ngridx, Val{ND}())
end

function _populate_cells_a!(si::SystemInteraction{T,ND}, cutoff::T) where {T,ND}
    _populate_cells_sorted!(si._cell_start_a,
                            si.system_a.x, si._mingridx, cutoff, si._ngridx, Val{ND}())
end

function _populate_cells_b!(si::SystemInteraction{T,ND}, cutoff::T) where {T,ND}
    _populate_cells_sorted!(si._cell_start,
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

adjust_v!(si::SystemInteraction) = _sweep_pfns!(si, si.system_b, (si.vadjust_pfn,), 1)

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

function _sweep_self!(si::SystemInteraction{T,2}, ::Nothing) where {T}
end
function _sweep_self!(si::SystemInteraction{T,3}, ::Nothing) where {T}
end

# 2D Specialisation: 6-colour sweep
#
# SORTED-PARTICLE RANGE TRICK
# ===========================
# Particles are sorted by flat cell index (cell_x slowest, cell_y fastest).
# Therefore particles in cells with consecutive flat indices occupy contiguous
# slices of the particle array.  A single CSR range cell_start[c]..cell_start[c+k]-1
# spans k consecutive flat cells with no gaps, so we can cover multiple
# neighbouring cells with one range instead of k separate inner loops.
#
# This is exploited in two places per active cell (cell_x, cell_y):
#
#   1. Same-cell + (0,+1) neighbour — one range:
#      cell_start[cell_idx]   .. cell_start[cell_idx+2]-1
#      covers (cell_x, cell_y) and (cell_x, cell_y+1).
#      Starting particle_j at particle_i+1 gives the same-cell half-shell
#      (each unordered pair visited once); indices beyond pend_same
#      automatically reach (cell_x, cell_y+1) particles.
#
#   2. (+1,*) strip — one range:
#      cell_start[cell_idx + n_cells_y - 1] .. cell_start[cell_idx + n_cells_y + 2]-1
#      The base offset (n_cells_y - 1) lands on flat cell (cell_x+1, cell_y-1);
#      reading +3 entries from cell_start spans the full strip
#      (cell_x+1, cell_y-1), (cell_x+1, cell_y), (cell_x+1, cell_y+1).
#      This handles the three forward neighbours (+1,-1), (+1,0), (+1,+1)
#      in a single particle loop.
function _sweep_self!(si::SystemInteraction{T,2}, pfn::PFN) where {T,PFN}
    ps_a       = si.system_a
    kernel     = si.kernel
    h          = T(kernel.h)
    cutoff_sq  = si._cell_size * si._cell_size
    cell_start = si._cell_start::Vector{Int}
    n_cells_y  = Int(si._ngridx[2])
    n_cells_x  = Int(si._ngridx[1])
    val_ndims  = Val{2}()

    for colour in 0:5
        cell_x_begin = colour ÷ 2 + 2
        cell_y_begin = colour % 2 + 2
        n_active_x = length(cell_x_begin:3:n_cells_x-1)
        n_active_y = length(cell_y_begin:2:n_cells_y-1)
        @inbounds @batch for flat_idx in 1:n_active_x*n_active_y
            step_x, step_y = divrem(flat_idx - 1, n_active_y)
            cell_x   = cell_x_begin + step_x * 3
            cell_y   = cell_y_begin + step_y * 2
            cell_idx = (cell_x - 1) * n_cells_y + cell_y

            # Particles in (cell_x, cell_y) and (cell_x, cell_y+1) are contiguous;
            # a single range covers both cells.
            pstart    = cell_start[cell_idx]
            pend_same = cell_start[cell_idx + 1]  # end of (cell_x, cell_y)
            pend_next = cell_start[cell_idx + 2]  # end of (cell_x, cell_y+1)

            # Flat cell (cell_x+1, cell_y-1) = cell_idx + n_cells_y - 1.
            # Reading +3 entries spans the strip (cell_x+1, cell_y-1..cell_y+1).
            neighbour_cell_idx = cell_idx + n_cells_y - 1
            neighbour_pstart   = cell_start[neighbour_cell_idx]
            neighbour_pend     = cell_start[neighbour_cell_idx + 3]

            for particle_i in pstart:pend_same-1
                # Same-cell half-shell (particle_i < particle_j avoids double-counting),
                # then (0,+1) neighbour — both covered by the single range pstart..pend_next-1.
                for particle_j in particle_i+1:pend_next-1
                    _pair_self!(pfn, ps_a, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
                end
                # Forward neighbours (+1,-1), (+1,0), (+1,+1) as one contiguous strip.
                for particle_j in neighbour_pstart:neighbour_pend-1
                    _pair_self!(pfn, ps_a, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
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
# 3D Specialisation: 18-colour sweep
#
# SORTED-PARTICLE RANGE TRICK (3D)
# =================================
# Particles are sorted with cell_z varying fastest (cell_x slowest).
# Consecutive flat cell indices therefore differ only in cell_z, so cells
# that are adjacent in z occupy contiguous particle-array slices.
#
# This is exploited in two places per active cell (cell_x, cell_y, cell_z):
#
#   1. Same-cell + (0,0,+1) neighbour — one range:
#      pstart..pend_next-1  covers (cell_x,cell_y,cell_z) and (cell_x,cell_y,cell_z+1).
#      Starting particle_j at particle_i+1 gives the same-cell half-shell;
#      indices from pend onward reach (cell_x,cell_y,cell_z+1) automatically.
#
#   2. Each forward-offset group — one range per offset:
#      Each entry in forward_offsets points to the flat index of the cell
#      at z-1 relative to the neighbour row/plane.  Reading +3 entries from
#      cell_start spans the three consecutive z-cells (z-1, z, z+1) in that
#      row, covering three forward neighbours with a single particle loop.
#      The 4 offsets × 3 z-cells = 12 pairs, plus (0,0,+1) = 13 total.
function _sweep_self!(si::SystemInteraction{T,3}, pfn::PFN) where {T,PFN}
    ps_a       = si.system_a
    kernel     = si.kernel
    h          = T(kernel.h)
    cutoff_sq  = si._cell_size * si._cell_size
    cell_start = si._cell_start::Vector{Int}
    n_cells_z  = Int(si._ngridx[3])
    n_cells_y  = Int(si._ngridx[2])
    n_cells_x  = Int(si._ngridx[1])
    n_cells_yz = n_cells_y * n_cells_z
    val_ndims  = Val{3}()

    # Each offset lands on the (dz = -1) cell of a 3-cell z-strip; reading
    # +3 entries from cell_start covers dz ∈ {-1, 0, +1} in one range.
    forward_offsets = (
        n_cells_z-1,             # ( 0, +1, -1..+1)
        n_cells_yz-n_cells_z-1, # (+1, -1, -1..+1)
        n_cells_yz-1,            # (+1,  0, -1..+1)
        n_cells_yz+n_cells_z-1, # (+1, +1, -1..+1)
    )

    for colour in 0:17
        cell_x_begin = (colour ÷ 9) + 2
        cell_y_begin = ((colour % 9) ÷ 3) + 2
        cell_z_begin = (colour % 3) + 2
        n_active_x  = length(cell_x_begin:2:n_cells_x-1)
        n_active_y  = length(cell_y_begin:3:n_cells_y-1)
        n_active_z  = length(cell_z_begin:3:n_cells_z-1)
        n_active_yz = n_active_y * n_active_z
        @inbounds @batch for flat_idx in 1:n_active_x*n_active_yz
            step_x, remaining_yz = divrem(flat_idx - 1, n_active_yz)
            step_y, step_z       = divrem(remaining_yz, n_active_z)
            cell_x   = cell_x_begin + step_x * 2
            cell_y   = cell_y_begin + step_y * 3
            cell_z   = cell_z_begin + step_z * 3
            cell_idx  = (cell_x - 1) * n_cells_yz + (cell_y - 1) * n_cells_z + cell_z

            # Particles in (cell_x, cell_y, cell_z) and (cell_x, cell_y, cell_z+1)
            # are contiguous; a single range covers both cells.
            pstart    = cell_start[cell_idx]
            pend      = cell_start[cell_idx + 1]  # end of (cell_x, cell_y, cell_z)
            pend_next = cell_start[cell_idx + 2]  # end of (cell_x, cell_y, cell_z+1)

            # Same-cell half-shell (particle_i < particle_j), then (0,0,+1) neighbour —
            # both covered by one range because the two cells are contiguous.
            for particle_i in pstart:pend-1
                for particle_j in particle_i+1:pend_next-1
                    _pair_self!(pfn, ps_a, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
                end
            end

            # Remaining 12 forward neighbours in 4 z-strips of 3 cells each.
            for dir_idx in 1:4
                neighbour_cell_idx = cell_idx + forward_offsets[dir_idx]
                neighbour_pstart   = cell_start[neighbour_cell_idx]
                neighbour_pend     = cell_start[neighbour_cell_idx + 3]
                for particle_i in pstart:pend-1
                    for particle_j in neighbour_pstart:neighbour_pend-1
                        _pair_self!(pfn, ps_a, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
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

function _sweep_coupled!(si::SystemInteraction{T,2}, system_b, ::Nothing) where {T}
end
function _sweep_coupled!(si::SystemInteraction{T,3}, system_b, ::Nothing) where {T}
end

# 2D Specialisation: 9-colour sweep
#
# SORTED-PARTICLE RANGE TRICK (coupled, 2D)
# =========================================
# system_b particles are sorted by flat cell index (cell_y fastest).
# For each system_a cell (cell_x, cell_y) we must search the full 3×3 block
# of system_b cells: dx_cell ∈ {-1,0,+1}, dy ∈ {-1,0,+1}.
#
# Instead of 3 separate inner loops over dy, we use the contiguity of
# consecutive y-cells: the base offset (dx_cell * n_cells_y - 1) lands on
# flat cell (cell_x+dx_cell, cell_y-1), and reading +3 entries from
# cell_start spans dy ∈ {-1, 0, +1} in one contiguous particle range.
# This reduces 9 range lookups to 3 (one per dx_cell value).
#
# cell_x_min/max: cell index bounds of system_a in the system_b grid,
# used to skip the (many) grid cells that contain no system_a particles.
function _sweep_coupled!(si::SystemInteraction{T,2}, system_b, pfn::PFN) where {T,PFN}
    ps_a          = si.system_a
    kernel        = si.kernel
    h             = T(kernel.h)
    cutoff_sq     = si._cell_size * si._cell_size
    cell_start    = si._cell_start::Vector{Int}
    cell_start_a  = si._cell_start_a::Vector{Int}
    n_cells_y     = Int(si._ngridx[2])
    n_cells_x     = Int(si._ngridx[1])
    val_ndims     = Val{2}()
    cutoff        = si._cell_size
    mingridx      = si._mingridx
    # Cell index range (in the system_b grid) that system_a particles occupy.
    cell_x_min = floor(Int, (si._mingridx_a[1] - mingridx[1]) / cutoff) + 1
    cell_x_max = floor(Int, (si._maxgridx_a[1] - mingridx[1]) / cutoff) + 1
    cell_y_min = floor(Int, (si._mingridx_a[2] - mingridx[2]) / cutoff) + 1
    cell_y_max = floor(Int, (si._maxgridx_a[2] - mingridx[2]) / cutoff) + 1

    for colour in 0:8
        cell_x_begin = colour ÷ 3 + cell_x_min
        cell_y_begin = colour % 3 + cell_y_min
        n_active_x = length(cell_x_begin:3:cell_x_max)
        n_active_y = length(cell_y_begin:3:cell_y_max)
        @inbounds @batch for flat_idx in 1:n_active_x*n_active_y
            step_x, step_y = divrem(flat_idx - 1, n_active_y)
            cell_x   = cell_x_begin + step_x * 3
            cell_y   = cell_y_begin + step_y * 3
            cell_idx = (cell_x - 1) * n_cells_y + cell_y
            sys_a_pstart = cell_start_a[cell_idx]
            sys_a_pend   = cell_start_a[cell_idx + 1]
            for particle_i in sys_a_pstart:sys_a_pend-1
                # For each x-offset, read a 3-cell y-strip of system_b particles
                # in one contiguous range: base offset lands on (dx_cell, -1) in y,
                # and +3 spans dy ∈ {-1, 0, +1}.
                for dx_cell in -1:1
                    neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
                    sys_b_pstart = cell_start[neighbour_cell_idx]
                    sys_b_pend   = cell_start[neighbour_cell_idx + 3]
                    for particle_j in sys_b_pstart:sys_b_pend-1
                        _pair_coupled!(pfn, ps_a, system_b, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
                    end
                end
            end
        end
    end
end

# 3D Specialisation: 27-colour sweep
#
# SORTED-PARTICLE RANGE TRICK (coupled, 3D)
# =========================================
# system_b particles are sorted with cell_z varying fastest.
# For each system_a cell we must search the full 3×3×3 block of system_b
# cells: dx_cell ∈ {-1,0,+1}, dy_cell ∈ {-1,0,+1}, dz ∈ {-1,0,+1}.
#
# The z-dimension is handled implicitly: the base offset
# (dx_cell * n_cells_yz + dy_cell * n_cells_z - 1) lands on
# flat cell (cell_x+dx, cell_y+dy, cell_z-1), and reading +3 entries from
# cell_start spans dz ∈ {-1, 0, +1} in one contiguous particle range.
# This reduces 27 range lookups to 9 (one per (dx_cell, dy_cell) pair).
function _sweep_coupled!(si::SystemInteraction{T,3}, system_b, pfn::PFN) where {T,PFN}
    ps_a          = si.system_a
    kernel        = si.kernel
    h             = T(kernel.h)
    cutoff_sq     = si._cell_size * si._cell_size
    cell_start    = si._cell_start::Vector{Int}
    cell_start_a  = si._cell_start_a::Vector{Int}
    n_cells_z     = Int(si._ngridx[3])
    n_cells_y     = Int(si._ngridx[2])
    n_cells_x     = Int(si._ngridx[1])
    n_cells_yz    = n_cells_y * n_cells_z
    val_ndims     = Val{3}()
    cutoff        = si._cell_size
    mingridx      = si._mingridx
    # Cell index range (in the system_b grid) that system_a particles occupy.
    cell_x_min = floor(Int, (si._mingridx_a[1] - mingridx[1]) / cutoff) + 1
    cell_x_max = floor(Int, (si._maxgridx_a[1] - mingridx[1]) / cutoff) + 1
    cell_y_min = floor(Int, (si._mingridx_a[2] - mingridx[2]) / cutoff) + 1
    cell_y_max = floor(Int, (si._maxgridx_a[2] - mingridx[2]) / cutoff) + 1
    cell_z_min = floor(Int, (si._mingridx_a[3] - mingridx[3]) / cutoff) + 1
    cell_z_max = floor(Int, (si._maxgridx_a[3] - mingridx[3]) / cutoff) + 1

    for colour in 0:26
        cell_x_begin = (colour ÷ 9) + cell_x_min
        cell_y_begin = ((colour % 9) ÷ 3) + cell_y_min
        cell_z_begin = (colour % 3) + cell_z_min
        n_active_x  = length(cell_x_begin:3:cell_x_max)
        n_active_y  = length(cell_y_begin:3:cell_y_max)
        n_active_z  = length(cell_z_begin:3:cell_z_max)
        n_active_yz = n_active_y * n_active_z
        @inbounds @batch for flat_idx in 1:n_active_x*n_active_yz
            step_x, remaining_yz = divrem(flat_idx - 1, n_active_yz)
            step_y, step_z       = divrem(remaining_yz, n_active_z)
            cell_x   = cell_x_begin + step_x * 3
            cell_y   = cell_y_begin + step_y * 3
            cell_z   = cell_z_begin + step_z * 3
            cell_idx = (cell_x - 1) * n_cells_yz + (cell_y - 1) * n_cells_z + cell_z
            sys_a_pstart = cell_start_a[cell_idx]
            sys_a_pend   = cell_start_a[cell_idx + 1]
            for particle_i in sys_a_pstart:sys_a_pend-1
                # Iterate over the 3×3 (dx, dy) neighbourhood; for each pair the
                # base offset lands on dz = -1, and +3 spans the full z-strip.
                for dx_cell in -1:1
                    for dy_cell in -1:1
                        neighbour_cell_idx = cell_idx + dx_cell * n_cells_yz + dy_cell * n_cells_z - 1
                        sys_b_pstart = cell_start[neighbour_cell_idx]
                        sys_b_pend   = cell_start[neighbour_cell_idx + 3]
                        for particle_j in sys_b_pstart:sys_b_pend-1
                            _pair_coupled!(pfn, ps_a, system_b, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
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
