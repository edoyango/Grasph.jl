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
    (ps.x, ps.id, ps.v, ps.v_adjustment, ps.rho, ps.dvdt, ps.drhodt)

_particle_arrays(ps::FluidParticleSystem) =
    (ps.x, ps.id, ps.v, ps.v_adjustment, ps.rho, ps.dvdt, ps.drhodt, ps.p)

_particle_arrays(ps::StressParticleSystem) =
    (ps.x, ps.id, ps.v, ps.v_adjustment, ps.rho, ps.dvdt, ps.drhodt, ps.p, ps.stress, ps.strain_rate)

_particle_arrays(ps::ElastoPlasticParticleSystem) =
    (ps.x, ps.id, ps.v, ps.v_adjustment, ps.rho, ps.dvdt, ps.drhodt, ps.p,
     ps.stress, ps.strain_rate, ps.vorticity, ps.strain, ps.strain_p)

# Virtual: delegate to source — w_sum is auto-zeroed before each sweep so order doesn't matter.
_particle_arrays(vps::VirtualParticleSystem) = _particle_arrays(getfield(vps, :source))

# Ghost: first-class fields + boundary metadata (must move with the ghost) + extras.
# x is still first so sort_particles! can read it as the sort key.
function _particle_arrays(ps::GhostParticleSystem)
    (getfield(ps, :x),
     getfield(ps, :v),
     getfield(ps, :rho),
     getfield(ps, :idx_original),
     getfield(ps, :idx_boundary),
     getfield(ps, :normals),
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
# Packed 64-bit cell key  (ND = 1, 2, 3)
#
# Each dimension's integer cell coordinate is bias-shifted to a non-negative
# value and packed into a fixed-width bitfield of a single UInt64, most
# significant dimension first. Comparing two packed keys with plain `<` is
# then bit-for-bit equivalent to the lexicographic (first dim slowest, last
# dim fastest) ordering the old SVector+custom-comparator key produced —
# this is a representation change only, with zero effect on sort output.
#
# The point of packing into a native integer (rather than an SVector) is
# that UInt64 is exactly what GPU radix/bucket sorts operate on; an SVector
# key with a custom `lt=` comparator is CPU-only. `_KEY_BITS_PER_DIM = 21`
# gives each dimension a representable cell-coordinate range of
# [-2^20, 2^20 - 1] (about ±1.05M cells from the origin) — enormous headroom
# for any domain/cutoff combination this code targets — while 3 × 21 = 63
# bits fits in a UInt64 with one bit to spare.
# ---------------------------------------------------------------------------

const _KEY_BITS_PER_DIM = 21
const _KEY_BIAS          = Int(1) << (_KEY_BITS_PER_DIM - 1)          # 2^20
const _KEY_FIELD_MASK    = (UInt64(1) << _KEY_BITS_PER_DIM) - one(UInt64)

# Bias-shift a single integer cell coordinate into a packable UInt64 field.
# Throws if the coordinate falls outside the representable range rather than
# silently wrapping/truncating.
@inline function _cellcoord_to_field(c::Int)
    biased = c + _KEY_BIAS
    (biased < 0 || biased > Int(_KEY_FIELD_MASK)) && _key_range_error(c)
    return biased % UInt64
end

@noinline function _key_range_error(c::Int)
    throw(ArgumentError(
        "cell coordinate $c is out of the packed sort key's representable " *
        "range [$(-_KEY_BIAS), $(_KEY_BIAS - 1)]; the domain is too large " *
        "(or the cutoff too small) for a 64-bit packed key"))
end

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

# Backend-dispatched. On KA.CPU(), walk the heterogeneous tuple of (arr,
# scratch) pairs one element at a time via _apply_perm! (unchanged, 2
# scalar-loop launches per array). On any other backend, walk the WHOLE
# tuple inside one kernel per pass (gather, then copy-back) — 2 launches for
# the entire system instead of 2 per array, since scalar-loop `_apply_perm!`
# would be illegal on a GPU array.
_apply_perms!(arrs::Tuple, scratches::Tuple, perm::AbstractVector{Int}, n::Int) =
    _apply_perms!(KA.get_backend(perm), arrs, scratches, perm, n)

_apply_perms!(::KA.CPU, arrs::Tuple, scratches::Tuple, perm::AbstractVector{Int}, n::Int) =
    _apply_perms_cpu!(arrs, scratches, perm, n)

_apply_perms_cpu!(::Tuple{}, ::Tuple{}, ::AbstractVector{Int}, ::Int) = nothing
@inline function _apply_perms_cpu!(arrs::Tuple, scratches::Tuple,
                                    perm::AbstractVector{Int}, n::Int)
    _apply_perm!(first(arrs), perm, first(scratches), n)
    _apply_perms_cpu!(Base.tail(arrs), Base.tail(scratches), perm, n)
end

function _apply_perms!(backend::KA.Backend, arrs::Tuple, scratches::Tuple,
                        perm::AbstractVector{Int}, n::Int)
    n == 0 && return nothing
    _gather_perm_kernel!(backend, _KA_WORKGROUP)(scratches, arrs, perm; ndrange = n)
    KA.synchronize(backend)
    _copyback_kernel!(backend, _KA_WORKGROUP)(arrs, scratches; ndrange = n)
    KA.synchronize(backend)
    return nothing
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

# Allocation-free position → packed UInt64 cell-key conversion. One explicit
# method per ND (matching the old _lt_key's per-ND dispatch) rather than a
# generic `for d in 1:ND` loop, so the bit-packing is fully unrolled with no
# risk of the compiler boxing a mutated loop-carried accumulator.
@inline function _pos_to_key(xi::SVector{1,T}, cutoff::T) where {T}
    _cellcoord_to_field(floor(Int, xi[1] / cutoff))
end

@inline function _pos_to_key(xi::SVector{2,T}, cutoff::T) where {T}
    cx = _cellcoord_to_field(floor(Int, xi[1] / cutoff))
    cy = _cellcoord_to_field(floor(Int, xi[2] / cutoff))
    (cx << _KEY_BITS_PER_DIM) | cy
end

@inline function _pos_to_key(xi::SVector{3,T}, cutoff::T) where {T}
    cx = _cellcoord_to_field(floor(Int, xi[1] / cutoff))
    cy = _cellcoord_to_field(floor(Int, xi[2] / cutoff))
    cz = _cellcoord_to_field(floor(Int, xi[3] / cutoff))
    (cx << (2 * _KEY_BITS_PER_DIM)) | (cy << _KEY_BITS_PER_DIM) | cz
end

# GPU-only variant: _cellcoord_to_field's range check throws a formatted
# ArgumentError on failure, and GPUCompiler rejects any reachable kernel code
# path that needs dynamic string construction — confirmed empirically, not a
# theoretical concern (compiling _pos_to_key inside a @kernel function fails
# with "unsupported call to a lazy-initialized function" from the `string(c)`
# inside the error path, even though that branch is never taken for valid
# input). The check exists to catch a domain/cutoff combination needing more
# than ~1M cells per dimension — unreachable for any problem size this
# migration targets. The device path skips it; an out-of-range coordinate
# would silently wrap instead of throwing, on GPU only.
@inline _cellcoord_to_field_gpu(c::Int) = (c + _KEY_BIAS) % UInt64

@inline function _pos_to_key_gpu(xi::SVector{1,T}, cutoff::T) where {T}
    _cellcoord_to_field_gpu(floor(Int, xi[1] / cutoff))
end

@inline function _pos_to_key_gpu(xi::SVector{2,T}, cutoff::T) where {T}
    cx = _cellcoord_to_field_gpu(floor(Int, xi[1] / cutoff))
    cy = _cellcoord_to_field_gpu(floor(Int, xi[2] / cutoff))
    (cx << _KEY_BITS_PER_DIM) | cy
end

@inline function _pos_to_key_gpu(xi::SVector{3,T}, cutoff::T) where {T}
    cx = _cellcoord_to_field_gpu(floor(Int, xi[1] / cutoff))
    cy = _cellcoord_to_field_gpu(floor(Int, xi[2] / cutoff))
    cz = _cellcoord_to_field_gpu(floor(Int, xi[3] / cutoff))
    (cx << (2 * _KEY_BITS_PER_DIM)) | (cy << _KEY_BITS_PER_DIM) | cz
end

# UInt64 has native `<`, so no custom comparator is needed any more — this is
# what makes the key GPU-radix-sortable, unlike the old SVector+lt= key.
#
# InsertionSort is a CPU-only algorithm choice (fast on the near-sorted input
# sort_particles! typically sees after the first few timesteps); dropped on
# non-CPU backends, where CUDA.jl's sortperm! is the only option and is
# verified stable (bit-identical permutation to Base's stable sort on ties).
# Ghost systems stay on Base's default sortperm! unconditionally — ghosts are
# not GPU-resident in this migration (see docs/gpu-migration-plan.md).
@inline _sortperm_by_key!(::AbstractParticleSystem, perm_view, key_view) =
    _sortperm_by_key_backend!(KA.get_backend(perm_view), perm_view, key_view)

@inline _sortperm_by_key!(::AbstractGhostParticleSystem, perm_view, key_view) =
    sortperm!(perm_view, key_view)

@inline _sortperm_by_key_backend!(::KA.CPU, perm_view, key_view) =
    sortperm!(perm_view, key_view; alg=InsertionSort)

@inline _sortperm_by_key_backend!(::KA.Backend, perm_view, key_view) =
    sortperm!(perm_view, key_view)

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
                          perm_buf::AbstractVector{Int},
                          key_buf::AbstractVector{UInt64},
                          scratch_arrays::Tuple) where {T,ND}
    n = ps.n
    n <= 1 && return

    # Grow shared work buffers if this system is larger than previous ones.
    length(perm_buf) < n && resize!(perm_buf, n)
    length(key_buf)  < n && resize!(key_buf,  n)
    _resize_scratches!(scratch_arrays, n)

    # Compute packed cell-coordinate keys from particle positions.
    arrs = _particle_arrays(ps)
    x    = first(arrs)       # x is always the first array
    _compute_keys!(key_buf, x, cutoff, n)

    # Fast path: if keys are already non-decreasing, no reordering needed.
    # Avoids both sortperm! and _apply_perms! on steps where particles haven't
    # crossed cell boundaries — common after the first few timesteps. CPU-only:
    # it's a serial scan, and on GPU the D2H sync it would need to branch on
    # costs more than the sort it might save (see docs/gpu-migration-plan.md).
    _maybe_return_if_sorted!(KA.get_backend(key_buf), key_buf, n) && return

    # Compute the sorting permutation in-place (no allocation).
    perm_view = view(perm_buf, 1:n)
    key_view  = view(key_buf,  1:n)
    _sortperm_by_key!(ps, perm_view, key_view)

    # Apply permutation to every per-particle array.
    _apply_perms!(arrs, scratch_arrays, perm_view, n)
end

@inline _compute_keys!(key_buf, x, cutoff, n) = _compute_keys!(KA.get_backend(key_buf), key_buf, x, cutoff, n)

@inline function _compute_keys!(::KA.CPU, key_buf, x, cutoff, n)
    @inbounds for i in 1:n
        key_buf[i] = _pos_to_key(x[i], cutoff)
    end
end

function _compute_keys!(backend::KA.Backend, key_buf, x, cutoff, n)
    n == 0 && return nothing
    _pos_to_key_kernel!(backend, _KA_WORKGROUP)(key_buf, x, cutoff; ndrange = n)
    KA.synchronize(backend)
    return nothing
end

function _maybe_return_if_sorted!(::KA.CPU, key_buf, n)
    @inbounds for i in 1:n-1
        key_buf[i+1] < key_buf[i] && return false
    end
    return true
end

@inline _maybe_return_if_sorted!(::KA.Backend, key_buf, n) = false
