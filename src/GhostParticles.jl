export AbstractGhostParticleSystem,
       GhostParticleSystem, GhostCopier, HouseholderReflect,
       GhostBoundary, GhostEntry,
       generate_ghosts!, update_ghost!, update_ghost_kinematics!, write_h5

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

abstract type AbstractGhostParticleSystem{T<:AbstractFloat, ND} <: AbstractParticleSystem{T,ND} end

# ---------------------------------------------------------------------------
# GhostCopier — callable per-stage field copier
# ---------------------------------------------------------------------------

abstract type AbstractGhostUpdater end

"""
    HouseholderReflect()

Marker type selecting symmetric-tensor reflection by Householder transform
`σ' = H σ H` with `H = I − 2 n̂ n̂ᵀ`, where `n̂` is the per-ghost inward
normal.  Valid for Voigt stress vectors of length 3, 4, or 6; the dimensions
of `n̂` and the Voigt length must be consistent.
"""
struct HouseholderReflect end

"""
    GhostCopier(:field1, :field2, …)
    GhostCopier(:field1 => HouseholderReflect(), :field2, …)

A callable ghost updater that copies the named fields from a ghost's source
particle system into its owned `extras` arrays when called with a ghost.

Each entry is either a bare `Symbol` (straight copy) or a
`Symbol => HouseholderReflect()` pair which applies a full Householder
reflection of the symmetric Voigt tensor against the ghost's cached normal:

    GhostCopier(:stress => HouseholderReflect())

Pass one or more `GhostCopier`s to `GhostParticleSystem` to declare which
fields should be owned and how they should be refreshed per stage:

    ghost = GhostParticleSystem(ps,
                GhostCopier(:p, :stress),   # stage 1: copy p + stress
                GhostCopier(:p))            # stage 2: copy p only

Calling `update_ghost!(ghost, stage)` invokes the stage-th copier.

Density (:rho) is core kinematics and is updated every step automatically;
you generally do not need to include it in a copier.
"""
struct GhostCopier{fields, MODES} <: AbstractGhostUpdater end

function GhostCopier(entries...)
    flds = ntuple(i -> entries[i] isa Symbol ? entries[i] : first(entries[i]), length(entries))
    mds  = ntuple(i -> entries[i] isa Symbol ? nothing    : last(entries[i]),  length(entries))
    GhostCopier{flds, mds}()
end

_updater_fields(::GhostCopier{fields, MODES}) where {fields, MODES} = fields
_updater_fields(::AbstractGhostUpdater) = ()   # fallback for custom updater types
_updater_fields(::Nothing) = ()

# Straight copy — mode nothing leaves the value untouched.
@inline _apply_mode(val, ::Nothing, n̂) = val

# 2D Voigt [σ_xx, σ_yy, σ_xy] with 2D normal.
@inline function _apply_mode(σ::SVector{3,T}, ::HouseholderReflect, n̂::SVector{2,T}) where {T}
    nx, ny = n̂
    tx = σ[1]*nx + σ[3]*ny
    ty = σ[3]*nx + σ[2]*ny
    s  = tx*nx + ty*ny
    SVector{3,T}(
        σ[1] - 4*tx*nx + 4*s*nx*nx,
        σ[2] - 4*ty*ny + 4*s*ny*ny,
        σ[3] - 2*(tx*ny + nx*ty) + 4*s*nx*ny,
    )
end

# 2D Voigt [σ_xx, σ_yy, σ_zz, σ_xy] with 2D normal — σ_zz is invariant.
@inline function _apply_mode(σ::SVector{4,T}, ::HouseholderReflect, n̂::SVector{2,T}) where {T}
    nx, ny = n̂
    tx = σ[1]*nx + σ[4]*ny
    ty = σ[4]*nx + σ[2]*ny
    s  = tx*nx + ty*ny
    SVector{4,T}(
        σ[1] - 4*tx*nx + 4*s*nx*nx,
        σ[2] - 4*ty*ny + 4*s*ny*ny,
        σ[3],
        σ[4] - 2*(tx*ny + nx*ty) + 4*s*nx*ny,
    )
end

# 3D Voigt [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz] with 3D normal.
@inline function _apply_mode(σ::SVector{6,T}, ::HouseholderReflect, n̂::SVector{3,T}) where {T}
    nx, ny, nz = n̂
    tx = σ[1]*nx + σ[4]*ny + σ[5]*nz
    ty = σ[4]*nx + σ[2]*ny + σ[6]*nz
    tz = σ[5]*nx + σ[6]*ny + σ[3]*nz
    s  = tx*nx + ty*ny + tz*nz
    SVector{6,T}(
        σ[1] - 4*tx*nx + 4*s*nx*nx,
        σ[2] - 4*ty*ny + 4*s*ny*ny,
        σ[3] - 4*tz*nz + 4*s*nz*nz,
        σ[4] - 2*(tx*ny + nx*ty) + 4*s*nx*ny,
        σ[5] - 2*(tx*nz + nx*tz) + 4*s*nx*nz,
        σ[6] - 2*(ty*nz + ny*tz) + 4*s*ny*nz,
    )
end

# CPU path: scalar loop bounded explicitly by `n` (the logical ghost count),
# not `eachindex(arr)` — `arr`/`idx`/`normals` are capacity-preallocated on
# the GPU path (see generate_ghosts! below) and can be longer than `n`, with
# stale data beyond it. On CPU this is currently a no-op difference (CPU
# generate_ghosts! always resizes exactly to the count, so capacity == n
# there), kept explicit for both backends to share one invariant.
_copy_fields_cpu!(ghost, idx, normals, n, ::Tuple{}, ::Tuple{}) = nothing
@inline function _copy_fields_cpu!(ghost, idx, normals, n, fields::Tuple, modes::Tuple)
    fname   = first(fields)
    src_arr = getproperty(getfield(ghost, :source), fname)
    arr     = getproperty(getfield(ghost, :extras), fname)
    mode    = first(modes)
    @inbounds for k in 1:n
        arr[k] = _apply_mode(src_arr[idx[k]], mode, normals[k])
    end
    _copy_fields_cpu!(ghost, idx, normals, n, Base.tail(fields), Base.tail(modes))
end

# GPU path: one kernel launch per field (fields are heterogeneously typed —
# SVector Voigt tensors, scalars — so they can't share one fused kernel the
# way _gather_tuple!/_copyback_tuple! do for homogeneous sort scratch).
_copy_fields_ka!(backend, src, idx, normals, ghost, ::Tuple{}, ::Tuple{}, n) = nothing
function _copy_fields_ka!(backend, src, idx, normals, ghost, fields::Tuple, modes::Tuple, n)
    fname   = first(fields)
    src_arr = getproperty(src, fname)
    arr     = getproperty(getfield(ghost, :extras), fname)
    mode    = first(modes)
    _ghost_copy_field_kernel!(backend, _KA_WORKGROUP)(arr, src_arr, idx, normals, mode; ndrange = n)
    KA.synchronize(backend)
    _copy_fields_ka!(backend, src, idx, normals, ghost, Base.tail(fields), Base.tail(modes), n)
end

function (gc::GhostCopier{fields, MODES})(ghost::AbstractGhostParticleSystem) where {fields, MODES}
    # Backend decided from :x, consistent with generate_ghosts!/
    # update_ghost_kinematics! — all three assume a ghost's owned arrays
    # share one backend (true by construction/adapt, see GhostParticleSystem's
    # docstring), so deciding from any one of them is equivalent; :x is used
    # everywhere else specifically so this stays uniform.
    _run_copier!(KA.get_backend(getfield(ghost, :x)), ghost, fields, MODES)
    return nothing
end

function _run_copier!(::KA.CPU, ghost, fields, modes)
    _copy_fields_cpu!(ghost, getfield(ghost, :idx_original), getfield(ghost, :normals), ghost.n, fields, modes)
end

function _run_copier!(backend::KA.Backend, ghost, fields, modes)
    n = ghost.n
    n == 0 && return nothing
    _copy_fields_ka!(backend, getfield(ghost, :source), getfield(ghost, :idx_original),
                      getfield(ghost, :normals), ghost, fields, modes, n)
end

# ---------------------------------------------------------------------------
# extras allocation helpers
# ---------------------------------------------------------------------------

# Collect the ordered union of all field names across a tuple of updaters.
function _all_extras_fields(updaters::Tuple)
    seen = Symbol[]
    for upd in updaters
        for f in _updater_fields(upd)
            f in seen || push!(seen, f)
        end
    end
    return Tuple(seen)
end

function _build_extras(ps, fields::Tuple)
    arrays = map(f -> similar(getproperty(ps, f), 0), fields)
    return NamedTuple{fields}(arrays)
end

# ---------------------------------------------------------------------------
# GhostParticleSystem
# ---------------------------------------------------------------------------

"""
    GhostParticleSystem{T, ND, PS, ET, UPD, VA, SA, IA}

A ghost particle system whose owned physics arrays and per-stage update
behaviour are fully specified by the `GhostCopier`s supplied at construction.

    # No field copying:
    ghost = GhostParticleSystem(ps)

`extras` is a `NamedTuple` containing one `Vector` per field in the union of
all copier field lists. `x`, `v`, and `rho` are first-class fields updated
every step. Scalar source fields (`mass`, `c`, …) are forwarded directly.

`idx_original[k]` maps ghost particle k to its source particle index.
`idx_boundary[k]` maps ghost particle k to the boundary that generated it,
indexing into the `boundaries` tuple of the associated `GhostEntry`.
`normals[k]` caches the inward-pointing unit normal of that boundary for
fast per-ghost reflection operations.

The per-particle array fields are generic over container type (`VA` for
`SVector`-valued fields, `SA` for scalar fields, `IA` for the index-mapping
fields), defaulting to `Vector`/`Vector`/`Vector{Int}` via the constructor
below, mirroring `BasicParticleSystem`'s convention (see `Particles.jl`).
This is what lets `Adapt.adapt(CuArray, ghost)` produce a device-resident
ghost system.

Ghost count varies every step (`generate_ghosts!` re-derives it from which
source particles currently qualify), so — unlike every other particle-system
type — array *length* is not necessarily the logical particle count: `count`
holds the logical count (`n`), while the owned arrays' length is a
capacity that only ever grows (see `generate_ghosts!`'s GPU path in this
file, which cannot cheaply `resize!` a `CuArray` every step the way the CPU
path resizes a `Vector`). `count` is a `Base.RefValue` so it can be updated
in place without needing `GhostParticleSystem` itself to be mutable — the
same trick real systems don't need since their `n` truly never changes.
"""
struct GhostParticleSystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}, ET<:NamedTuple, UPD<:Tuple,
                            VA<:AbstractVector{SVector{ND,T}}, SA<:AbstractVector{T}, IA<:AbstractVector{Int}} <: AbstractGhostParticleSystem{T, ND}
    name::String
    count::Base.RefValue{Int}         # logical ghost count (<= capacity == length(x) etc.)
    x::VA                             # reflected positions — owned
    v::VA                             # reflected velocities — owned
    rho::SA                           # mirrored density — owned
    idx_original::IA                  # ghost k → source particle index
    idx_boundary::IA                  # ghost k → boundary index in GhostEntry
    normals::VA                       # ghost k → inward unit normal (cached)
    source::PS
    extras::ET                        # owned copies: (p=…, stress=…, …)
    updaters::UPD                     # per-stage GhostCopier or nothing instances
    function GhostParticleSystem{T, ND, PS, ET, UPD, VA, SA, IA}(args...) where {T, ND, PS, ET, UPD, VA, SA, IA}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PS, ET, UPD, VA, SA, IA}(args...)
    end
end

"""
    GhostParticleSystem(ps, copiers…; name=nothing) -> GhostParticleSystem

Allocate a ghost system backed by `ps`.

Each positional argument after `ps` is a `GhostCopier` (or `nothing`) defining
which fields to copy at each corresponding sweep stage.

Always builds `Vector`-backed owned arrays, regardless of `ps`'s own array
type (matching `VirtualParticleSystem`'s identical convention) — to get a
GPU-resident ghost, construct everything CPU-first as usual, then
`adapt(CUDABackend(), ge::GhostEntry)` the whole entry as one call (this
cascades into the ghost's owned arrays, its wrapped `source`, and
`GhostEntry`'s own `_flags` scratch together). Calling this constructor
directly with an already-GPU-resident `ps` produces an internally
inconsistent, mixed-backend object instead (owned arrays `Vector`, `source`
`CuArray`) — `generate_ghosts!` would then dispatch to the CPU path based on
the owned arrays' backend and immediately hit an illegal scalar read from
the GPU-resident source.

For a *self-referencing* ghost (`ghost.source === ps`, the only pattern this
codebase's scripts use — see `docs/gpu-migration-plan.md`), note that
`adapt` does not preserve object identity: if a driver script separately
adapts its own copy of `ps` (e.g. for a `SystemInteraction`) and separately
adapts a `GhostEntry` wrapping the same `ps`, the two results are
independent GPU copies, not aliases — mutating one will not update the
other. Use `getfield(adapted_ghost, :source)` as the canonical GPU-resident
reference for the source system throughout the rest of the driver, rather
than adapting `ps` a second time (see `test/test_gpu_cuda.jl`'s ghost tests
for the pattern in practice).
"""
function GhostParticleSystem(
    ps::AbstractParticleSystem{T,ND},
    updaters::Union{Nothing, AbstractGhostUpdater}...;
    name::Union{Nothing,AbstractString} = nothing,
) where {T,ND}
    ps isa AbstractGhostParticleSystem && throw(ArgumentError(
        "GhostParticleSystem cannot wrap another AbstractGhostParticleSystem as its source: " *
        "a ghost's particle count changes every generate_ghosts! call, which would invalidate " *
        "the wrapping GhostEntry's _flags scratch buffer (fixed size NB * source.n, allocated " *
        "once at GhostEntry construction under the assumption that source.n never changes)."))
    n      = ps.n
    gname  = name === nothing ? "ghost($(ps.name))" : String(name)
    fields = _all_extras_fields(updaters)
    extras = _build_extras(ps, fields)
    x            = Vector{SVector{ND,T}}(undef, n)
    v            = Vector{SVector{ND,T}}(undef, n)
    rho          = Vector{T}(undef, n)
    idx_original = Vector{Int}(undef, n)
    idx_boundary = Vector{Int}(undef, n)
    normals      = Vector{SVector{ND,T}}(undef, n)
    GhostParticleSystem{T, ND, typeof(ps), typeof(extras), typeof(updaters),
                         typeof(x), typeof(rho), typeof(idx_original)}(
        gname, Ref(n), x, v, rho, idx_original, idx_boundary, normals,
        ps, extras, updaters,
    )
end

# ---------------------------------------------------------------------------
# getproperty override
# ---------------------------------------------------------------------------

@inline function Base.getproperty(g::GhostParticleSystem{T,ND,PS,ET,UPD}, s::Symbol) where {T,ND,PS,ET,UPD}
    s === :ndims && return ND
    s === :n     && return getfield(g, :count)[]
    s in (:name, :x, :v, :rho, :idx_original, :idx_boundary, :normals, :source, :extras, :updaters) && return getfield(g, s)

    # Owned copies (p, stress, …) — contiguous, cache-friendly
    s in fieldnames(ET) && return getproperty(getfield(g, :extras), s)

    # Scalars (mass, c, …) forwarded directly from source
    return getproperty(getfield(g, :source), s)
end

# ---------------------------------------------------------------------------
# Adapt.jl support
#
# `count` is host-only bookkeeping (an Int box, not per-particle data) and is
# rebuilt as a fresh `Ref` with the same value rather than shared — mirrors
# how `name`/`updaters` are carried over unadapted elsewhere in this codebase,
# except `count` genuinely needs a *copy* (not a shared reference) so mutating
# the adapted system's logical count can never alias back into the original.
# ---------------------------------------------------------------------------

function Adapt.adapt_structure(to, ghost::GhostParticleSystem{T,ND,PS,ET,UPD}) where {T,ND,PS,ET,UPD}
    new_source       = Adapt.adapt(to, getfield(ghost, :source))
    new_x            = Adapt.adapt(to, getfield(ghost, :x))
    new_v            = Adapt.adapt(to, getfield(ghost, :v))
    new_rho          = Adapt.adapt(to, getfield(ghost, :rho))
    new_idx_original = Adapt.adapt(to, getfield(ghost, :idx_original))
    new_idx_boundary = Adapt.adapt(to, getfield(ghost, :idx_boundary))
    new_normals      = Adapt.adapt(to, getfield(ghost, :normals))
    new_extras       = map(arr -> Adapt.adapt(to, arr), getfield(ghost, :extras))
    GhostParticleSystem{T, ND, typeof(new_source), typeof(new_extras), UPD,
                         typeof(new_x), typeof(new_rho), typeof(new_idx_original)}(
        getfield(ghost, :name), Ref(getfield(ghost, :count)[]),
        new_x, new_v, new_rho, new_idx_original, new_idx_boundary, new_normals,
        new_source, new_extras, getfield(ghost, :updaters),
    )
end

# ---------------------------------------------------------------------------
# GhostBoundary
# ---------------------------------------------------------------------------

"""
    GhostBoundary{ND, T}

A plane boundary defined by an inward-pointing unit `normal` and a `point`
on the plane.  Used as elements of the `boundaries` tuple in a `GhostEntry`.
"""
struct GhostBoundary{ND, T<:AbstractFloat}
    normal::SVector{ND,T}
    point::SVector{ND,T}
    function GhostBoundary{ND,T}(normal, point) where {ND,T}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{ND,T}(normal, point)
    end
end

# ---------------------------------------------------------------------------
# update_ghost!
# ---------------------------------------------------------------------------

_run_ghost_stage!(ghost, ::Tuple{}, stage) = nothing
@inline function _run_ghost_stage!(ghost, updaters::Tuple, stage)
    if stage == 1
        fn = first(updaters)
        fn !== nothing && fn(ghost)
    else
        _run_ghost_stage!(ghost, Base.tail(updaters), stage - 1)
    end
end

"""
    update_ghost!(ghost::GhostParticleSystem, stage::Int)

Run the `stage`-th `GhostCopier`, copying its fields from source into
`extras`.  Does nothing if no copier is registered for that stage.
"""
update_ghost!(ghost::GhostParticleSystem, stage::Int) =
    _run_ghost_stage!(ghost, getfield(ghost, :updaters), stage)

# ---------------------------------------------------------------------------
# GhostEntry
# ---------------------------------------------------------------------------

"""
    GhostEntry{GPS, ND, T, NB}

Bundles a ghost particle system with `NB` boundary planes and a shared
`cutoff` distance.  Each boundary is a `GhostBoundary{ND,T}` holding an
inward-pointing unit `normal` and a `point` on the plane.

Construct with:

    entry = GhostEntry(ghost, cutoff,
                       (normal1, point1),
                       (normal2, point2), …)

where each `(normal, point)` pair describes one boundary.
`ghost.idx_boundary[k]` gives the index into `boundaries` for ghost particle k.
"""
struct GhostEntry{GPS<:AbstractGhostParticleSystem, ND, T<:AbstractFloat, NB, FA<:AbstractVector{Int}}
    ghost::GPS
    boundaries::NTuple{NB, GhostBoundary{ND,T}}
    cutoff::T
    _flags::FA   # scratch, length NB * source.n; GPU generate_ghosts! only, see below
    function GhostEntry{GPS, ND, T, NB, FA}(args...) where {GPS, ND, T, NB, FA}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        NB isa Int || throw(ArgumentError("NB must be an Int, got $(typeof(NB))"))
        new{GPS, ND, T, NB, FA}(args...)
    end
end

"""
    GhostEntry(ghost, cutoff, (normal1, point1), (normal2, point2), …) -> GhostEntry

Construct a `GhostEntry` with one or more boundary planes.  Each boundary is
specified as a `(normal, point)` 2-tuple.
"""
function GhostEntry(
    ghost::AbstractGhostParticleSystem{T,ND},
    cutoff::Real,
    boundary_pairs...,
) where {T,ND}
    isempty(boundary_pairs) && throw(ArgumentError("at least one boundary (normal, point) pair is required"))
    boundaries = map(boundary_pairs) do pair
        normal, point = pair
        GhostBoundary{ND,T}(SVector{ND,T}(normal), SVector{ND,T}(point))
    end
    NB = length(boundaries)
    source = getfield(ghost, :source)
    flags  = similar(source.x, Int, NB * source.n)
    GhostEntry{typeof(ghost), ND, T, NB, typeof(flags)}(ghost, boundaries, T(cutoff), flags)
end

# `_flags` is rebuilt to match the adapted ghost's backend — its length (`NB *
# source.n`) never changes across an adapt, only which array type it is.
function Adapt.adapt_structure(to, ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    new_ghost = Adapt.adapt(to, getfield(ge, :ghost))
    new_flags = Adapt.adapt(to, getfield(ge, :_flags))
    GhostEntry{typeof(new_ghost), ND, T, NB, typeof(new_flags)}(
        new_ghost, getfield(ge, :boundaries), getfield(ge, :cutoff), new_flags,
    )
end

# ---------------------------------------------------------------------------
# generate_ghosts! — GhostEntry form (multi-boundary)
# ---------------------------------------------------------------------------

# Shared by both the CPU driver below and _ghost_flag_kernel!/
# _ghost_scatter_kernel! (KAKernels.jl) — written once so the two backends
# can never silently drift apart on which particles qualify as ghosts.
# Returns `(qualifies, da)`; `da` (signed distance to the boundary plane) is
# reused by the caller to compute the reflected position without recomputing
# the dot product.
@inline function _ghost_qualifies(bnd, xi, cutoff)
    da = dot(xi - bnd.point, bnd.normal)
    return (abs(da) <= cutoff) && (da > zero(da)), da
end

"""
    generate_ghosts!(ge::GhostEntry)

Populate `ge.ghost` from its source by reflecting qualifying real particles
across every boundary plane in `ge`.  Each ghost's `idx_boundary` field
records which boundary (1-based index into `ge.boundaries`) generated it.

On CPU, resizes every owned ghost array to exactly the qualifying count each
call (cheap for `Vector`). On a GPU backend, owned arrays instead grow to a
*capacity* that only ever increases — a per-step `resize!` is a full
reallocation+copy on a `CuArray`, the wrong strategy for a count that
fluctuates every step — while `ghost.n` reports the exact logical count
regardless of backend (see `GhostParticleSystem`'s docstring).
"""
generate_ghosts!(ge::GhostEntry) = _generate_ghosts!(KA.get_backend(getfield(getfield(ge, :ghost), :x)), ge)

function _generate_ghosts!(::KA.CPU, ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    ghost      = ge.ghost
    boundaries = ge.boundaries
    cutoff     = ge.cutoff
    ps         = getfield(ghost, :source)

    # First pass: count qualifying particles across all boundaries
    total = 0
    for b in boundaries
        @inbounds for i in 1:ps.n
            qualifies, _ = _ghost_qualifies(b, ps.x[i], cutoff)
            qualifies && (total += 1)
        end
    end

    # Resize all owned arrays to the total count
    resize!(getfield(ghost, :x),            total)
    resize!(getfield(ghost, :v),            total)
    resize!(getfield(ghost, :rho),          total)
    resize!(getfield(ghost, :idx_original), total)
    resize!(getfield(ghost, :idx_boundary), total)
    resize!(getfield(ghost, :normals),      total)
    for arr in getfield(ghost, :extras)
        resize!(arr, total)
    end

    # Second pass: populate positions, index mappings, and cached normals
    x        = getfield(ghost, :x)
    idx_orig = getfield(ghost, :idx_original)
    idx_bnd  = getfield(ghost, :idx_boundary)
    normals  = getfield(ghost, :normals)
    cursor   = 0
    for (b_idx, b) in enumerate(boundaries)
        @inbounds for i in 1:ps.n
            qualifies, da = _ghost_qualifies(b, ps.x[i], cutoff)
            if qualifies
                cursor          += 1
                x[cursor]        = ps.x[i] - (2 * da) * b.normal
                idx_orig[cursor] = i
                idx_bnd[cursor]  = b_idx
                normals[cursor]  = b.normal
            end
        end
    end

    getfield(ghost, :count)[] = total
    return ghost
end

# GPU path: flag + exclusive-scan + compaction, replacing the CPU count-then-
# cursor passes. The (boundary, particle) pair space is flattened to a single
# linear index (boundary-major, particle-minor — matching the CPU version's
# iteration order, so ghost k gets the same source particle/boundary on both
# backends): `flags[lin]` is set to 0/1 by _ghost_flag_kernel!, then
# `cumsum!` turns it into an inclusive prefix sum in place — for a qualifying
# pair, that value IS its final 1-based destination index (exactly that many
# qualifying pairs, including itself, precede or equal it). No atomics
# needed, unlike the cell-histogram CSR build in Interaction.jl: there,
# multiple particles can land in the same cell (a real many-to-one
# histogram); here each linear index is its own always-unique flag, a plain
# stream compaction. `ge._flags` has fixed length `NB * source.n` — the
# source's particle count never changes over a run, only how many qualify —
# so it's allocated once, at GhostEntry construction, and reused forever.
function _generate_ghosts!(backend::KA.GPU, ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    ghost       = ge.ghost
    boundaries  = ge.boundaries
    cutoff      = ge.cutoff
    ps          = getfield(ghost, :source)
    flags       = getfield(ge, :_flags)
    total_pairs = NB * ps.n

    if total_pairs == 0
        getfield(ghost, :count)[] = 0
        return ghost
    end

    _ghost_flag_kernel!(backend, _KA_WORKGROUP)(flags, ps.x, boundaries, cutoff, ps.n; ndrange = total_pairs)
    KA.synchronize(backend)
    cumsum!(flags, flags)

    total = Int(Array(view(flags, total_pairs:total_pairs))[1])

    # Grow any owned array (including extras) that's currently smaller than
    # `total` — NOT gated on comparing `total` against just `length(ghost.x)`:
    # extras arrays start at length 0 (_build_extras) regardless of what
    # capacity x/v/rho/etc. start at, so they can need growing on a step
    # where x/v/rho/etc. already have enough room. _resize_scratches! grows
    # each array in the tuple independently and is a no-op for any array
    # already >= total, so this is cheap on the (common) steps where nothing
    # needs to grow.
    _resize_scratches!(_particle_arrays(ghost), total)

    if total > 0
        _ghost_scatter_kernel!(backend, _KA_WORKGROUP)(
            getfield(ghost, :x), getfield(ghost, :idx_original), getfield(ghost, :idx_boundary), getfield(ghost, :normals),
            flags, ps.x, boundaries, cutoff, ps.n; ndrange = total_pairs)
        KA.synchronize(backend)
    end

    getfield(ghost, :count)[] = total
    return ghost
end

# ---------------------------------------------------------------------------
# update_ghost_kinematics! — GhostEntry form (multi-boundary)
# ---------------------------------------------------------------------------

"""
    update_ghost_kinematics!(ge::GhostEntry)

Reflect source velocities and mirror source densities into `ge.ghost`,
using each ghost's `idx_boundary` to select the correct boundary normal.
"""
update_ghost_kinematics!(ge::GhostEntry) = _update_ghost_kinematics!(KA.get_backend(getfield(getfield(ge, :ghost), :x)), ge)

function _update_ghost_kinematics!(::KA.CPU, ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    ghost      = ge.ghost
    ps         = getfield(ghost, :source)
    n          = ghost.n
    idx_orig   = getfield(ghost, :idx_original)
    normals    = getfield(ghost, :normals)
    v_ghost    = getfield(ghost, :v)
    rho_ghost  = getfield(ghost, :rho)
    v_real     = ps.v
    rho_real   = ps.rho

    @inbounds for k in 1:n
        normal       = normals[k]
        v_r          = v_real[idx_orig[k]]
        v_ghost[k]   = v_r - 2 * dot(v_r, normal) * normal
        rho_ghost[k] = rho_real[idx_orig[k]]
    end
end

function _update_ghost_kinematics!(backend::KA.GPU, ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    ghost = ge.ghost
    ps    = getfield(ghost, :source)
    n     = ghost.n
    n == 0 && return nothing
    _ghost_kinematics_kernel!(backend, _KA_WORKGROUP)(
        getfield(ghost, :v), getfield(ghost, :rho),
        getfield(ghost, :idx_original), getfield(ghost, :normals),
        ps.v, ps.rho; ndrange = n)
    KA.synchronize(backend)
    return nothing
end

update_ghost!(ge::GhostEntry, stage::Int) = update_ghost!(ge.ghost, stage)

# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

"""
    write_h5(ghost::GhostParticleSystem, group)

Write active ghost particles to an HDF5 group.
"""
function write_h5(ghost::GhostParticleSystem{T,ND,PS,ET,UPD}, group::Union{HDF5.File, HDF5.Group}) where {T,ND,PS,ET,UPD}
    n = ghost.n
    HDF5.attrs(group)["n"]     = n
    HDF5.attrs(group)["ndims"] = ghost.ndims
    HDF5.attrs(group)["mass"]  = ghost.mass

    # Owned arrays are capacity-preallocated (see generate_ghosts!) and can be
    # longer than n, with stale data beyond it — every array is explicitly
    # sliced to 1:n rather than written in full.
    if n > 0
        group["x"]            = reinterpret(reshape, T, view(getfield(ghost, :x), 1:n))
        group["v"]            = reinterpret(reshape, T, view(getfield(ghost, :v), 1:n))
        group["rho"]          = view(getfield(ghost, :rho), 1:n)
        group["idx_original"] = view(getfield(ghost, :idx_original), 1:n)
        group["idx_boundary"] = view(getfield(ghost, :idx_boundary), 1:n)

        # Genericly save all extra fields
        for fname in fieldnames(ET)
            arr = view(getproperty(getfield(ghost, :extras), fname), 1:n)
            if eltype(arr) <: SVector
                group[string(fname)] = reinterpret(reshape, T, arr)
            else
                group[string(fname)] = arr
            end
        end
    end
end
