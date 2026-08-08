export device_view

# ---------------------------------------------------------------------------
# DeviceSystem — an isbits-able kernel-argument view of a particle system.
#
# `adapt(CuArray, ps)` alone is NOT enough to pass a system to a
# `@kernel` function: FluidParticleSystem/BasicParticleSystem carry
# `name::String` and `_print_fields::Vector{Symbol}`, which `adapt_structure`
# deliberately passes through unadapted (they're host-only bookkeeping, not
# per-particle data). GPUCompiler rejects non-isbits kernel arguments
# unconditionally, regardless of whether the field is actually read inside the
# kernel — verified empirically: passing an adapted FluidParticleSystem to a
# trivial kernel fails GPU compilation; passing a NamedTuple-backed view
# carrying only the arrays succeeds.
#
# device_view(ps) builds that view. It subtypes AbstractParticleSystem{T,ND}
# with the SAME T/ND as ps, so every existing pfn_contribution/state-updater
# method (which dispatches on AbstractParticleSystem{T,ND} or a concrete
# system type) works against it unchanged — no changes needed anywhere in
# PairwiseFunctors.jl or StateUpdaters.jl.
# ---------------------------------------------------------------------------

struct DeviceSystem{T, ND, NT<:NamedTuple} <: AbstractParticleSystem{T, ND}
    _f::NT
end

@inline function Base.getproperty(ds::DeviceSystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(getfield(ds, :_f), s)
end

# Lets cudaconvert rewrite CuArray -> CuDeviceArray inside the NamedTuple.
# Adapt already recurses into NamedTuple/Tuple; only the struct-level wrapping
# needs a method here.
function Adapt.adapt_structure(to, ds::DeviceSystem{T,ND}) where {T,ND}
    nt = Adapt.adapt(to, getfield(ds, :_f))
    DeviceSystem{T, ND, typeof(nt)}(nt)
end

# One method per system type, listing exactly the fields pairwise functors and
# state updaters read via `ps.field` (dot/getproperty access, never `getfield`
# directly — that's what makes this view transparent to them). Only the
# systems exercised by the dambreak.jl vertical slice are covered; extend this
# list alongside device_view's other consumers as more systems get ported.
@inline _device_fields(ps::BasicParticleSystem) =
    (n = ps.n, x = ps.x, v = ps.v, v_adjustment = ps.v_adjustment, rho = ps.rho,
     dvdt = ps.dvdt, drhodt = ps.drhodt, mass = ps.mass, c = ps.c,
     source_v = ps.source_v, source_rho = ps.source_rho)

@inline _device_fields(ps::FluidParticleSystem) =
    (n = ps.n, x = ps.x, v = ps.v, v_adjustment = ps.v_adjustment, rho = ps.rho,
     dvdt = ps.dvdt, drhodt = ps.drhodt, p = ps.p, mass = ps.mass, c = ps.c,
     source_v = ps.source_v, source_rho = ps.source_rho)

@inline function device_view(ps::AbstractParticleSystem{T,ND}) where {T,ND}
    nt = _device_fields(ps)
    return DeviceSystem{T, ND, typeof(nt)}(nt)
end

# Boundary wrappers rebuild around the viewed inner system, preserving the
# StaticBoundarySystem{T,ND,PS} type that FluidPfn's coupled pfn_contribution
# method dispatches on.
@inline device_view(bs::StaticBoundarySystem) =
    StaticBoundarySystem(device_view(getfield(bs, :inner)), getfield(bs, :lj_cutoff))
