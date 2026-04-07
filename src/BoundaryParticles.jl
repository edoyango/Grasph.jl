export AbstractBoundarySystem, StaticBoundarySystem, DynamicBoundarySystem

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

abstract type AbstractBoundarySystem{T<:AbstractFloat, ND} <: AbstractParticleSystem{T,ND} end

# ---------------------------------------------------------------------------
# getproperty override — shared by all boundary wrapper types
# ---------------------------------------------------------------------------

@inline function Base.getproperty(bs::AbstractBoundarySystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    s === :n     && return getfield(bs, :inner).n
    # Expose boundary-specific fields directly
    hasfield(typeof(bs), s) && return getfield(bs, s)
    # Forward everything else to the inner system
    return getproperty(getfield(bs, :inner), s)
end

# ---------------------------------------------------------------------------
# StaticBoundarySystem — LJ repulsion boundary
# ---------------------------------------------------------------------------

"""
    StaticBoundarySystem{T, ND, PS} <: AbstractBoundarySystem{T, ND}

Wraps a particle system as a static Lennard-Jones boundary. The wrapped
system's fields (`x`, `v`, `rho`, `mass`, etc.) are forwarded transparently.

Use as `system_b` in a `SystemInteraction` to dispatch `FluidPfn` or
`CauchyFluidPfn` to the LJ + artificial viscosity boundary method.

The wrapped `inner` system should still go in the integrator's `systems`
list for time stepping and HDF5 output.
"""
struct StaticBoundarySystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}} <: AbstractBoundarySystem{T, ND}
    inner::PS
    lj_cutoff::T
    function StaticBoundarySystem{T, ND, PS}(inner::PS, lj_cutoff::T) where {T, ND, PS}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PS}(inner, lj_cutoff)
    end
end

function StaticBoundarySystem(inner::AbstractParticleSystem{T,ND}, lj_cutoff::Real) where {T,ND}
    StaticBoundarySystem{T, ND, typeof(inner)}(inner, T(lj_cutoff))
end

# ---------------------------------------------------------------------------
# DynamicBoundarySystem — distance-ratio velocity derivation
# ---------------------------------------------------------------------------

"""
    DynamicBoundarySystem{T, ND, PS} <: AbstractBoundarySystem{T, ND}

Wraps a particle system as a dynamic boundary where per-pair velocity is
derived from the distance ratio to a boundary plane.

The boundary plane is defined by `boundary_normal` (unit outward normal) and
`boundary_point` (a point on the plane). `boundary_beta` caps the velocity
ratio.

Use as `system_b` in a `SystemInteraction` to dispatch `StrainRatePfn`,
`FluidPfn`, or `CauchyFluidPfn` to the dynamic boundary method.
"""
struct DynamicBoundarySystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}} <: AbstractBoundarySystem{T, ND}
    inner::PS
    boundary_normal::SVector{ND,T}
    boundary_point::SVector{ND,T}
    boundary_beta::T
    function DynamicBoundarySystem{T, ND, PS}(inner::PS, normal::SVector{ND,T}, point::SVector{ND,T}, beta::T) where {T, ND, PS}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PS}(inner, normal, point, beta)
    end
end

function DynamicBoundarySystem(inner::AbstractParticleSystem{T,ND}, normal, point, beta::Real) where {T,ND}
    DynamicBoundarySystem{T, ND, typeof(inner)}(
        inner,
        SVector{ND,T}(normal),
        SVector{ND,T}(point),
        T(beta),
    )
end
