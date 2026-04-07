export AbstractGhostParticleSystem,
       GhostParticleSystem,
       GhostEntry, generate_ghosts!, update_ghost!, update_ghost_kinematics!, write_h5

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

abstract type AbstractGhostParticleSystem{T<:AbstractFloat, ND} <: AbstractParticleSystem{T,ND} end

# ---------------------------------------------------------------------------
# GhostParticleSystem
# ---------------------------------------------------------------------------

"""
    GhostParticleSystem{T, ND, PS} <: AbstractGhostParticleSystem{T, ND}

A generic ghost particle system that provides live indexed views into its 
source system. Physics fields (rho, p, stress, etc.) are never copied;
only positions (x) and velocities (v) are owned and updated.
"""
struct GhostParticleSystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}} <: AbstractGhostParticleSystem{T, ND}
    name::String
    x::Vector{SVector{ND,T}}    # reflected positions — owned
    v::Vector{SVector{ND,T}}    # reflected velocities — owned
    idx_original::Vector{Int}   # ghost k → source particle idx_original[k]
    source::PS
    function GhostParticleSystem{T, ND, PS}(args...) where {T, ND, PS}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PS}(args...)
    end
end

"""
    GhostParticleSystem(ps::AbstractParticleSystem{T,ND}; name=nothing) -> GhostParticleSystem

Allocate a ghost system with initial capacity for all `ps.n` particles.
Pass `name` to override the auto-generated `"ghost[<source_name>]"` name,
which is required when multiple ghost systems share the same source.
"""
function GhostParticleSystem(ps::AbstractParticleSystem{T,ND}; name::Union{Nothing,AbstractString}=nothing) where {T,ND}
    n = ps.n
    gname = name === nothing ? "ghost[$(ps.name)]" : String(name)
    GhostParticleSystem{T, ND, typeof(ps)}(
        gname,
        Vector{SVector{ND,T}}(undef, n),
        Vector{SVector{ND,T}}(undef, n),
        Vector{Int}(undef, n),
        ps,
    )
end

# ---------------------------------------------------------------------------
# getproperty override — live indexed views
# ---------------------------------------------------------------------------

 @inline function Base.getproperty(g::GhostParticleSystem{T,ND,PS}, s::Symbol) where {T,ND,PS}
    s === :ndims && return ND
    s === :n     && return length(getfield(g, :x))
    s in (:name, :x, :v, :idx_original, :source) && return getfield(g, s)
    
    # Forward all other fields to source as indexed views or direct scalars
    val = getproperty(getfield(g, :source), s)
    if val isa AbstractVector
        return view(val, getfield(g, :idx_original))
    end
    return val   # scalars (mass, c, …) forwarded directly
end

# ---------------------------------------------------------------------------
# generate_ghosts!
# ---------------------------------------------------------------------------

"""
    generate_ghosts!(ghost::GhostParticleSystem, normal, point, cutoff)

Populate `ghost` from `ghost.source` by reflecting qualifying real particles
across the plane defined by `normal` (unit vector) and `point`.

Only initializes positions (`x`) and the mapping (`idx_original`). 
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

    # Single resize to exact count
    resize!(getfield(ghost, :x),   k)
    resize!(getfield(ghost, :v),   k)
    resize!(getfield(ghost, :idx_original), k)

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
# update_ghost_kinematics!
# ---------------------------------------------------------------------------

"""
    update_ghost_kinematics!(ghost::GhostParticleSystem, normal)

Update ghost particle velocities by reflecting them across the boundary normal.
Density is now a live view, so it is not updated here.
"""
function update_ghost_kinematics!(
    ghost::GhostParticleSystem{T,ND},
    normal::SVector{ND,T},
) where {T,ND}
    ps  = getfield(ghost, :source)
    n   = ghost.n
    idx = getfield(ghost, :idx_original)
    v_ghost = getfield(ghost, :v)
    v_real  = ps.v

    @inbounds for k in 1:n
        v_r        = v_real[idx[k]]
        v_ghost[k] = v_r - 2 * dot(v_r, normal) * normal
    end
end

# ---------------------------------------------------------------------------
# update_ghost! — now a no-op
# ---------------------------------------------------------------------------

"""
    update_ghost!(ghost::GhostParticleSystem)

No-op. All physics fields are live views.
"""
update_ghost!(ghost::GhostParticleSystem) = nothing

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

update_ghost!(ge::GhostEntry) = update_ghost!(ge.ghost)

# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

"""
    write_h5(ghost::GhostParticleSystem, group)

Write active ghost particles to an HDF5 group.
"""
function write_h5(ghost::GhostParticleSystem{T,ND,PS}, group::Union{HDF5.File, HDF5.Group}) where {T,ND,PS}
    n = ghost.n
    HDF5.attrs(group)["n"]     = n
    HDF5.attrs(group)["ndims"] = ghost.ndims
    HDF5.attrs(group)["mass"]  = ghost.mass

    if n > 0
        group["x"]   = reinterpret(reshape, T, ghost.x)
        group["v"]   = reinterpret(reshape, T, ghost.v)
        group["rho"] = collect(ghost.rho)
        group["idx_original"] = getfield(ghost, :idx_original)
        if hasfield(PS, :p)
            group["p"] = collect(ghost.p)
        end
        if hasfield(PS, :stress)
            group["stress"] = reinterpret(reshape, T, collect(ghost.stress))
        end
    end
end
