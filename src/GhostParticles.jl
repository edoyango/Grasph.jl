export AbstractGhostParticleSystem,
       GhostParticleSystem, GhostCopier,
       GhostEntry, generate_ghosts!, update_ghost!, update_ghost_kinematics!, write_h5

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

A callable ghost updater that copies the named fields from a ghost's source
particle system into its owned `extras` arrays when called with a ghost.

Pass one or more `GhostCopier`s to `GhostParticleSystem` to declare which
fields should be owned and how they should be refreshed per stage:

    ghost = GhostParticleSystem(ps,
                GhostCopier(:p, :stress),   # stage 1: copy p + stress
                GhostCopier(:p))            # stage 2: copy p only

Calling `update_ghost!(ghost, stage)` invokes the stage-th copier.

Density (:rho) is core kinematics and is updated every step automatically;
you generally do not need to include it in a copier.
"""
struct GhostCopier{fields} <: AbstractGhostUpdater end

GhostCopier(fields::Symbol...) = GhostCopier{fields}()

_updater_fields(::GhostCopier{fields}) where {fields} = fields
_updater_fields(::AbstractGhostUpdater) = ()   # fallback for custom updater types
_updater_fields(::Nothing) = ()

function (::GhostCopier{fields})(ghost::AbstractGhostParticleSystem) where {fields}
    idx    = getfield(ghost, :idx_original)
    src    = getfield(ghost, :source)
    extras = getfield(ghost, :extras)
    for fname in fields
        src_arr = getproperty(src, fname)
        arr     = getproperty(extras, fname)
        @inbounds for k in eachindex(arr)
            arr[k] = src_arr[idx[k]]
        end
    end
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
"""
struct GhostParticleSystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}, ET<:NamedTuple, UPD<:Tuple} <: AbstractGhostParticleSystem{T, ND}
    name::String
    x::Vector{SVector{ND,T}}    # reflected positions — owned
    v::Vector{SVector{ND,T}}    # reflected velocities — owned
    rho::Vector{T}              # mirrored density — owned
    idx_original::Vector{Int}   # ghost k → source particle index
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
    gname  = name === nothing ? "ghost[$(ps.name)]" : String(name)
    fields = _all_extras_fields(updaters)
    extras = _build_extras(ps, fields)
    GhostParticleSystem{T, ND, typeof(ps), typeof(extras), typeof(updaters)}(
        gname,
        Vector{SVector{ND,T}}(undef, n),
        Vector{SVector{ND,T}}(undef, n),
        Vector{T}(undef, n),
        Vector{Int}(undef, n),
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
    s in (:name, :x, :v, :rho, :idx_original, :source, :extras, :updaters) && return getfield(g, s)

    # Owned copies (p, stress, …) — contiguous, cache-friendly
    s in fieldnames(ET) && return getproperty(getfield(g, :extras), s)

    # Scalars (mass, c, …) forwarded directly from source
    return getproperty(getfield(g, :source), s)
end

# ---------------------------------------------------------------------------
# generate_ghosts!
# ---------------------------------------------------------------------------

"""
    generate_ghosts!(ghost::GhostParticleSystem, normal, point, cutoff)

Populate `ghost` from `ghost.source` by reflecting qualifying real particles
across the plane defined by `normal` (unit vector) and `point`.

Resizes all owned arrays to the qualifying count. Call `update_ghost_kinematics!`
and `update_ghost!` to populate physics values.
"""
function generate_ghosts!(
    ghost::GhostParticleSystem{T,ND},
    normal::SVector{ND,T},
    point::SVector{ND,T},
    cutoff::T,
) where {T,ND}
    ps = getfield(ghost, :source)

    # First pass: count qualifying particles
    k = 0
    @inbounds for i in 1:ps.n
        da = dot(ps.x[i] - point, normal)
        if abs(da) <= cutoff && da > zero(T)
            k += 1
        end
    end

    # Resize all owned arrays to exact count
    resize!(getfield(ghost, :x),            k)
    resize!(getfield(ghost, :v),            k)
    resize!(getfield(ghost, :rho),          k)
    resize!(getfield(ghost, :idx_original), k)
    for arr in getfield(ghost, :extras)
        resize!(arr, k)
    end

    # Second pass: populate positions and index mapping
    x   = getfield(ghost, :x)
    idx = getfield(ghost, :idx_original)
    cursor = 0
    @inbounds for i in 1:ps.n
        da = dot(ps.x[i] - point, normal)
        if abs(da) <= cutoff && da > zero(T)
            cursor += 1
            x[cursor]   = ps.x[i] - (2 * da) * normal
            idx[cursor] = i
        end
    end

    return ghost
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
# update_ghost_kinematics!
# ---------------------------------------------------------------------------

"""
    update_ghost_kinematics!(ghost::GhostParticleSystem, normal)

Update ghost velocities by reflecting source velocities across the boundary
normal, and mirror source densities.
"""
function update_ghost_kinematics!(
    ghost::GhostParticleSystem{T,ND},
    normal::SVector{ND,T},
) where {T,ND}
    ps        = getfield(ghost, :source)
    n         = ghost.n
    idx       = getfield(ghost, :idx_original)
    v_ghost   = getfield(ghost, :v)
    v_real    = ps.v
    rho_ghost = getfield(ghost, :rho)
    rho_real  = ps.rho

    @inbounds for k in 1:n
        v_r          = v_real[idx[k]]
        v_ghost[k]   = v_r - 2 * dot(v_r, normal) * normal
        rho_ghost[k] = rho_real[idx[k]]
    end
end

# ---------------------------------------------------------------------------
# GhostEntry
# ---------------------------------------------------------------------------

"""
    GhostEntry{GPS, ND, T}

Bundles a ghost particle system with the boundary plane geometry
(`normal`, `point`, `cutoff`) needed to regenerate it each timestep.
"""
struct GhostEntry{GPS<:AbstractGhostParticleSystem, ND, T<:AbstractFloat}
    ghost::GPS
    normal::SVector{ND,T}
    point::SVector{ND,T}
    cutoff::T
    function GhostEntry{GPS, ND, T}(args...) where {GPS, ND, T}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{GPS, ND, T}(args...)
    end
end

"""
    GhostEntry(ghost, normal, point, cutoff) -> GhostEntry
"""
function GhostEntry(
    ghost::AbstractGhostParticleSystem{T,ND},
    normal,
    point,
    cutoff,
) where {T,ND}
    GhostEntry{typeof(ghost), ND, T}(
        ghost,
        SVector{ND,T}(normal),
        SVector{ND,T}(point),
        T(cutoff),
    )
end

generate_ghosts!(ge::GhostEntry) =
    generate_ghosts!(ge.ghost, ge.normal, ge.point, ge.cutoff)

update_ghost_kinematics!(ge::GhostEntry) =
    update_ghost_kinematics!(ge.ghost, ge.normal)

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
