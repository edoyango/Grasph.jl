export sort_particles!

# ---------------------------------------------------------------------------
# Per-particle arrays — explicit per concrete type (Option B)
#
# Each method returns a tuple of every per-particle mutable Vector in the
# system, in a consistent order.  The tuple is used both to compute the sort
# permutation (from x, which is always first) and to apply it to all arrays.
#
# To support a new particle system type, add a method here and list every
# per-particle field (excluding scalars such as mass, c, and metadata such as
# name, n, pairs).
# ---------------------------------------------------------------------------

_particle_arrays(ps::BasicParticleSystem) =
    (ps.x, ps.v, ps.v_adjustment, ps.rho, ps.dvdt, ps.drhodt)

_particle_arrays(ps::FluidParticleSystem) =
    (ps.x, ps.v, ps.v_adjustment, ps.rho, ps.dvdt, ps.drhodt, ps.p)

_particle_arrays(ps::StressParticleSystem) =
    (ps.x, ps.v, ps.rho, ps.dvdt, ps.drhodt, ps.p, ps.stress, ps.strain_rate)

_particle_arrays(ps::ElastoPlasticParticleSystem) =
    (ps.x, ps.v, ps.rho, ps.dvdt, ps.drhodt, ps.p,
     ps.stress, ps.strain_rate, ps.vorticity, ps.strain, ps.strain_p)

# Ghost: first-class fields + idx_original (must move with the ghost) + extras.
# x is still first so sort_particles! can read it as the sort key.
function _particle_arrays(ps::GhostParticleSystem)
    (getfield(ps, :x),
     getfield(ps, :v),
     getfield(ps, :rho),
     getfield(ps, :idx_original),
     getfield(ps, :idx_boundary),
     values(getfield(ps, :extras))...)
end

# ---------------------------------------------------------------------------
# Scratch buffer construction
# ---------------------------------------------------------------------------

"""
    _make_sort_scratch(ps) -> Tuple of Vectors

Allocate one scratch vector per array in `_particle_arrays(ps)`, matching
element type and current length.  Called once before the time loop for real
particle systems (fixed size).
"""
_make_sort_scratch(ps::AbstractParticleSystem) =
    map(similar, _particle_arrays(ps))

"""
    _make_empty_sort_scratch(ps) -> Tuple of Vectors

Same as `_make_sort_scratch` but with length 0.  Used for ghost systems
whose particle count changes each step; the vectors are grown on demand
inside `sort_particles!`.
"""
_make_empty_sort_scratch(ps::AbstractParticleSystem) =
    map(arr -> similar(arr, 0), _particle_arrays(ps))

# ---------------------------------------------------------------------------
# Lexicographic key comparison  (ND = 1, 2, 3)
#
# SVector does not implement isless by default, so we define explicit
# short-circuit comparisons for each dimensionality.  These are inlined into
# sortperm! via the lt= keyword, giving branch-free performance.
# ---------------------------------------------------------------------------

@inline _lt_key(a::SVector{1,Int}, b::SVector{1,Int}) = a[1] < b[1]

@inline _lt_key(a::SVector{2,Int}, b::SVector{2,Int}) =
    a[1] != b[1] ? a[1] < b[1] : a[2] < b[2]

@inline _lt_key(a::SVector{3,Int}, b::SVector{3,Int}) =
    a[1] != b[1] ? a[1] < b[1] : (a[2] != b[2] ? a[2] < b[2] : a[3] < b[3])

# ---------------------------------------------------------------------------
# In-place permutation
# ---------------------------------------------------------------------------

# Gather arr[perm[i]] into scratch, then write back.  The two-pass approach
# avoids aliasing: we never read from a position we have already overwritten.
@inline function _apply_perm!(arr::AbstractVector, perm::AbstractVector{Int},
                               scratch::AbstractVector, n::Int)
    @inbounds for i in 1:n
        scratch[i] = arr[perm[i]]
    end
    @inbounds for i in 1:n
        arr[i] = scratch[i]
    end
end

# Walk a heterogeneous tuple of (arr, scratch) pairs, peeling one element at
# a time so each call to _apply_perm! sees a concrete array type.
_apply_perms!(::Tuple{}, ::Tuple{}, ::AbstractVector{Int}, ::Int) = nothing
@inline function _apply_perms!(arrs::Tuple, scratches::Tuple,
                                perm::AbstractVector{Int}, n::Int)
    _apply_perm!(first(arrs), perm, first(scratches), n)
    _apply_perms!(Base.tail(arrs), Base.tail(scratches), perm, n)
end

# Grow each scratch vector to at least n elements (only triggers for ghost
# systems whose count increases between steps).
_resize_scratches!(::Tuple{}, ::Int) = nothing
@inline function _resize_scratches!(scratches::Tuple, n::Int)
    length(first(scratches)) < n && resize!(first(scratches), n)
    _resize_scratches!(Base.tail(scratches), n)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Allocation-free position → integer cell-key conversion.
# Using a named @inline function instead of a `map(v -> ..., xi)` closure
# ensures the captured `cutoff` is never heap-boxed.
@inline function _pos_to_key(xi::SVector{ND,T}, cutoff::T) where {ND,T}
    SVector(ntuple(d -> floor(Int, xi[d] / cutoff), Val(ND)))
end

@inline _sortperm_by_key!(::AbstractParticleSystem, perm_view, key_view) = 
    sortperm!(perm_view, key_view; lt=_lt_key, alg=InsertionSort)

@inline _sortperm_by_key!(::AbstractGhostParticleSystem, perm_view, key_view) =
    sortperm!(perm_view, key_view; lt=_lt_key)

"""
    sort_particles!(ps, cutoff, perm_buf, key_buf, scratch_arrays)

Re-order every per-particle array in `ps` so particles are sorted by their
cell coordinate `(floor(x[1]/cutoff), floor(x[2]/cutoff), ...)`,
lexicographically (first dimension slowest, last dimension fastest — matching
the row-major flat index used by `_cell_1idx`).

**Why this improves performance**: after sorting, particles in the same cell
are contiguous in memory.  The sweep traverses cells sequentially, so the
particle data it reads is already in cache.  For GPU offloading the sorted
layout gives coalesced memory access.

**Sort key**: `floor(x[d] / cutoff)` uses the infinite aligned cell lattice
and requires no `mingridx`.  Because every interaction grid origin is snapped
to a multiple of `cutoff` (see `create_grid!`), this key is consistent across
all interactions — any two grids assign particles to the same relative cell
positions.

**Shared work buffers**: `perm_buf` and `key_buf` are resized on demand and
reused across all calls within a timestep (real and ghost systems).
`scratch_arrays` is a tuple of pre-allocated vectors matching
`_particle_arrays(ps)` element-for-element; each vector is resized on demand
(growth only occurs for ghost systems whose count varies each step).
"""
function sort_particles!(ps::AbstractParticleSystem{T,ND}, cutoff::T,
                          perm_buf::Vector{Int},
                          key_buf::Vector{SVector{ND,Int}},
                          scratch_arrays::Tuple) where {T,ND}
    n = ps.n
    n <= 1 && return

    # Grow shared work buffers if this system is larger than previous ones.
    length(perm_buf) < n && resize!(perm_buf, n)
    length(key_buf)  < n && resize!(key_buf,  n)
    _resize_scratches!(scratch_arrays, n)

    # Compute integer cell-coordinate keys from particle positions.
    arrs = _particle_arrays(ps)
    x    = first(arrs)       # x is always the first array
    @inbounds for i in 1:n
        key_buf[i] = _pos_to_key(x[i], cutoff)
    end

    # Fast path: if keys are already non-decreasing, no reordering needed.
    # Avoids both sortperm! and _apply_perms! on steps where particles haven't
    # crossed cell boundaries — common after the first few timesteps.
    already_sorted = true
    @inbounds for i in 1:n-1
        if _lt_key(key_buf[i+1], key_buf[i])
            already_sorted = false
            break
        end
    end
    already_sorted && return

    # Compute the sorting permutation in-place (no allocation).
    perm_view = view(perm_buf, 1:n)
    key_view  = view(key_buf,  1:n)
    _sortperm_by_key!(ps, perm_view, key_view) #sortperm!(perm_view, key_view; lt=_lt_key, alg=InsertionSort)

    # Apply permutation to every per-particle array.
    _apply_perms!(arrs, scratch_arrays, perm_view, n)
end
