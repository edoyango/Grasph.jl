export AbstractGhostParticleSystem,
       GhostParticleSystem, GhostCopier,
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
    GhostCopier(:field1, :field2, …)
    GhostCopier(:field1 => signs1, :field2, …)

A callable ghost updater that copies the named fields from a ghost's source
particle system into its owned `extras` arrays when called with a ghost.

Each entry is either a bare `Symbol` (straight copy) or a `Symbol => SVector`
pair whose `SVector` is applied element-wise to the copied value.  This allows
individual stress components to be sign-flipped for free-slip walls:

    GhostCopier(:stress => SVector(1.0, 1.0, -1.0, -1.0))

Pass one or more `GhostCopier`s to `GhostParticleSystem` to declare which
fields should be owned and how they should be refreshed per stage:

    ghost = GhostParticleSystem(ps,
                GhostCopier(:p, :stress),   # stage 1: copy p + stress
                GhostCopier(:p))            # stage 2: copy p only

Calling `update_ghost!(ghost, stage)` invokes the stage-th copier.

Density (:rho) is core kinematics and is updated every step automatically;
you generally do not need to include it in a copier.
"""
struct GhostCopier{fields, SIGNS} <: AbstractGhostUpdater end

function GhostCopier(entries...)
    flds = ntuple(i -> entries[i] isa Symbol ? entries[i] : first(entries[i]), length(entries))
    sgns = ntuple(i -> entries[i] isa Symbol ? nothing    : last(entries[i]),  length(entries))
    GhostCopier{flds, sgns}()
end

_updater_fields(::GhostCopier{fields, SIGNS}) where {fields, SIGNS} = fields
_updater_fields(::AbstractGhostUpdater) = ()   # fallback for custom updater types
_updater_fields(::Nothing) = ()

@inline _apply_signs(val, ::Nothing) = val
@inline _apply_signs(val, signs)     = val .* signs

_copy_fields!(ghost, idx, ::Tuple{}, ::Tuple{}) = nothing
@inline function _copy_fields!(ghost, idx, fields::Tuple, signs::Tuple)
    fname   = first(fields)
    src_arr = getproperty(getfield(ghost, :source), fname)
    arr     = getproperty(getfield(ghost, :extras), fname)
    sign    = first(signs)
    @inbounds for k in eachindex(arr)
        arr[k] = _apply_signs(src_arr[idx[k]], sign)
    end
    _copy_fields!(ghost, idx, Base.tail(fields), Base.tail(signs))
end

function (::GhostCopier{fields, SIGNS})(ghost::AbstractGhostParticleSystem) where {fields, SIGNS}
    _copy_fields!(ghost, getfield(ghost, :idx_original), fields, SIGNS)
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
    GhostParticleSystem{T, ND, PS, ET, UPD}

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
"""
struct GhostParticleSystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}, ET<:NamedTuple, UPD<:Tuple} <: AbstractGhostParticleSystem{T, ND}
    name::String
    x::Vector{SVector{ND,T}}    # reflected positions — owned
    v::Vector{SVector{ND,T}}    # reflected velocities — owned
    rho::Vector{T}              # mirrored density — owned
    idx_original::Vector{Int}   # ghost k → source particle index
    idx_boundary::Vector{Int}   # ghost k → boundary index in GhostEntry
    source::PS
    extras::ET                  # owned copies: (p=…, stress=…, …)
    updaters::UPD               # per-stage GhostCopier or nothing instances
    function GhostParticleSystem{T, ND, PS, ET, UPD}(args...) where {T, ND, PS, ET, UPD}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PS, ET, UPD}(args...)
    end
end

"""
    GhostParticleSystem(ps, copiers…; name=nothing) -> GhostParticleSystem

Allocate a ghost system backed by `ps`.

Each positional argument after `ps` is a `GhostCopier` (or `nothing`) defining
which fields to copy at each corresponding sweep stage.
"""
function GhostParticleSystem(
    ps::AbstractParticleSystem{T,ND},
    updaters::Union{Nothing, AbstractGhostUpdater}...;
    name::Union{Nothing,AbstractString} = nothing,
) where {T,ND}
    n      = ps.n
    gname  = name === nothing ? "ghost($(ps.name))" : String(name)
    fields = _all_extras_fields(updaters)
    extras = _build_extras(ps, fields)
    GhostParticleSystem{T, ND, typeof(ps), typeof(extras), typeof(updaters)}(
        gname,
        Vector{SVector{ND,T}}(undef, n),
        Vector{SVector{ND,T}}(undef, n),
        Vector{T}(undef, n),
        Vector{Int}(undef, n),   # idx_original
        Vector{Int}(undef, n),   # idx_boundary
        ps,
        extras,
        updaters,
    )
end

# ---------------------------------------------------------------------------
# getproperty override
# ---------------------------------------------------------------------------

@inline function Base.getproperty(g::GhostParticleSystem{T,ND,PS,ET,UPD}, s::Symbol) where {T,ND,PS,ET,UPD}
    s === :ndims && return ND
    s === :n     && return length(getfield(g, :x))
    s in (:name, :x, :v, :rho, :idx_original, :idx_boundary, :source, :extras, :updaters) && return getfield(g, s)

    # Owned copies (p, stress, …) — contiguous, cache-friendly
    s in fieldnames(ET) && return getproperty(getfield(g, :extras), s)

    # Scalars (mass, c, …) forwarded directly from source
    return getproperty(getfield(g, :source), s)
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
struct GhostEntry{GPS<:AbstractGhostParticleSystem, ND, T<:AbstractFloat, NB}
    ghost::GPS
    boundaries::NTuple{NB, GhostBoundary{ND,T}}
    cutoff::T
    function GhostEntry{GPS, ND, T, NB}(args...) where {GPS, ND, T, NB}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        NB isa Int || throw(ArgumentError("NB must be an Int, got $(typeof(NB))"))
        new{GPS, ND, T, NB}(args...)
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
    GhostEntry{typeof(ghost), ND, T, NB}(ghost, boundaries, T(cutoff))
end

# ---------------------------------------------------------------------------
# generate_ghosts! — GhostEntry form (multi-boundary)
# ---------------------------------------------------------------------------

"""
    generate_ghosts!(ge::GhostEntry)

Populate `ge.ghost` from its source by reflecting qualifying real particles
across every boundary plane in `ge`.  Each ghost's `idx_boundary` field
records which boundary (1-based index into `ge.boundaries`) generated it.

Resizes all owned ghost arrays to the total qualifying count.
"""
function generate_ghosts!(ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    ghost      = ge.ghost
    boundaries = ge.boundaries
    cutoff     = ge.cutoff
    ps         = getfield(ghost, :source)

    # First pass: count qualifying particles across all boundaries
    total = 0
    for b in boundaries
        @inbounds for i in 1:ps.n
            da = dot(ps.x[i] - b.point, b.normal)
            if abs(da) <= cutoff && da > zero(T)
                total += 1
            end
        end
    end

    # Resize all owned arrays to the total count
    resize!(getfield(ghost, :x),            total)
    resize!(getfield(ghost, :v),            total)
    resize!(getfield(ghost, :rho),          total)
    resize!(getfield(ghost, :idx_original), total)
    resize!(getfield(ghost, :idx_boundary), total)
    for arr in getfield(ghost, :extras)
        resize!(arr, total)
    end

    # Second pass: populate positions and index mappings
    x        = getfield(ghost, :x)
    idx_orig = getfield(ghost, :idx_original)
    idx_bnd  = getfield(ghost, :idx_boundary)
    cursor   = 0
    for (b_idx, b) in enumerate(boundaries)
        @inbounds for i in 1:ps.n
            da = dot(ps.x[i] - b.point, b.normal)
            if abs(da) <= cutoff && da > zero(T)
                cursor          += 1
                x[cursor]        = ps.x[i] - (2 * da) * b.normal
                idx_orig[cursor] = i
                idx_bnd[cursor]  = b_idx
            end
        end
    end

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
function update_ghost_kinematics!(ge::GhostEntry{GPS,ND,T,NB}) where {GPS,ND,T,NB}
    ghost      = ge.ghost
    boundaries = ge.boundaries
    ps         = getfield(ghost, :source)
    n          = ghost.n
    idx_orig   = getfield(ghost, :idx_original)
    idx_bnd    = getfield(ghost, :idx_boundary)
    v_ghost    = getfield(ghost, :v)
    rho_ghost  = getfield(ghost, :rho)
    v_real     = ps.v
    rho_real   = ps.rho

    @inbounds for k in 1:n
        normal       = boundaries[idx_bnd[k]].normal
        v_r          = v_real[idx_orig[k]]
        v_ghost[k]   = v_r - 2 * dot(v_r, normal) * normal
        rho_ghost[k] = rho_real[idx_orig[k]]
    end
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

    if n > 0
        group["x"]            = reinterpret(reshape, T, ghost.x)
        group["v"]            = reinterpret(reshape, T, ghost.v)
        group["rho"]          = ghost.rho
        group["idx_original"] = getfield(ghost, :idx_original)
        group["idx_boundary"] = getfield(ghost, :idx_boundary)

        # Genericly save all extra fields
        for fname in fieldnames(ET)
            arr = getproperty(ghost.extras, fname)
            if eltype(arr) <: SVector
                group[string(fname)] = reinterpret(reshape, T, arr)
            else
                group[string(fname)] = arr
            end
        end
    end
end
