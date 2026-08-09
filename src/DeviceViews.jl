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
#
# `Kind` records *which concrete host struct* produced the view (e.g.
# `FluidParticleSystem`, the bare UnionAll — not a field, so it costs nothing
# at runtime and doesn't affect isbits-ness). Every "bare" system type
# (Basic/Fluid/Stress/ElastoPlastic) collapses to this same generic
# DeviceSystem otherwise, which is fine for the common case (most
# pfn_contribution methods are typed loosely enough not to care), but a pfn
# whose pfn_contribution/_onesided_zero_coupled is narrowly typed on a
# concrete host struct on BOTH sides of a coupled interaction — to
# disambiguate it from that same pfn's OTHER coupled methods, which key off a
# specific wrapper type on ps_b instead — needs its `ka=true` twin written
# against `DeviceSystem{T,ND,ThatConcreteType}` rather than plain
# `DeviceSystem{T,ND}`. See FluidPfn's fluid-fluid section in
# PairwiseFunctors.jl for the pattern.
# ---------------------------------------------------------------------------

struct DeviceSystem{T, ND, Kind, NT<:NamedTuple} <: AbstractParticleSystem{T, ND}
    _f::NT
end

@inline function Base.getproperty(ds::DeviceSystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(getfield(ds, :_f), s)
end

# Lets cudaconvert rewrite CuArray -> CuDeviceArray inside the NamedTuple.
# Adapt already recurses into NamedTuple/Tuple; only the struct-level wrapping
# needs a method here.
function Adapt.adapt_structure(to, ds::DeviceSystem{T,ND,Kind}) where {T,ND,Kind}
    nt = Adapt.adapt(to, getfield(ds, :_f))
    DeviceSystem{T, ND, Kind, typeof(nt)}(nt)
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

@inline _device_fields(ps::StressParticleSystem) =
    (n = ps.n, x = ps.x, v = ps.v, v_adjustment = ps.v_adjustment, rho = ps.rho,
     dvdt = ps.dvdt, drhodt = ps.drhodt, p = ps.p, stress = ps.stress, strain_rate = ps.strain_rate,
     mass = ps.mass, c = ps.c, source_v = ps.source_v, source_rho = ps.source_rho)

@inline _device_fields(ps::ElastoPlasticParticleSystem) =
    (n = ps.n, x = ps.x, v = ps.v, v_adjustment = ps.v_adjustment, rho = ps.rho,
     dvdt = ps.dvdt, drhodt = ps.drhodt, p = ps.p, stress = ps.stress, strain_rate = ps.strain_rate,
     vorticity = ps.vorticity, strain = ps.strain, strain_p = ps.strain_p,
     mass = ps.mass, c = ps.c, source_v = ps.source_v, source_rho = ps.source_rho)

@inline function device_view(ps::AbstractParticleSystem{T,ND}) where {T,ND}
    nt = _device_fields(ps)
    Kind = Base.typename(typeof(ps)).wrapper
    return DeviceSystem{T, ND, Kind, typeof(nt)}(nt)
end

# Boundary wrappers rebuild around the viewed inner system, preserving the
# StaticBoundarySystem{T,ND,PS} type that FluidPfn's coupled pfn_contribution
# method dispatches on.
@inline device_view(bs::StaticBoundarySystem) =
    StaticBoundarySystem(device_view(getfield(bs, :inner)), getfield(bs, :lj_cutoff))

@inline device_view(bs::DynamicBoundarySystem) =
    DynamicBoundarySystem(device_view(getfield(bs, :inner)), getfield(bs, :boundary_normal),
                           getfield(bs, :boundary_point), getfield(bs, :boundary_beta))

# ---------------------------------------------------------------------------
# VirtualParticleSystem — owns a non-isbits `name::String` directly (unlike
# Static/DynamicBoundarySystem, which own no host-only fields at all), so its
# own concrete type can't be rebuilt around a device-viewed inner system the
# way the boundary wrappers are. Flatten into a dedicated isbits device view
# that subtypes AbstractVirtualParticleSystem (see Particles.jl), so every
# pfn_contribution method written against that abstract type still dispatches
# without modification.
# ---------------------------------------------------------------------------

struct DeviceVirtualSystem{T, ND, NT<:NamedTuple} <: AbstractVirtualParticleSystem{T, ND}
    _f::NT
end

@inline function Base.getproperty(ds::DeviceVirtualSystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(getfield(ds, :_f), s)
end

function Adapt.adapt_structure(to, ds::DeviceVirtualSystem{T,ND}) where {T,ND}
    nt = Adapt.adapt(to, getfield(ds, :_f))
    DeviceVirtualSystem{T, ND, typeof(nt)}(nt)
end

# Merges the wrapped source's device fields with Virtual's own w_sum/
# prescribed_v — `vps.mass`/`vps.c`/etc. resolve through VirtualParticleSystem's
# real getproperty forwarding here, at view-construction time, not inside the
# kernel, so the flattened NamedTuple carries concrete values regardless of
# that forwarding.
@inline _device_fields(vps::VirtualParticleSystem) =
    merge(_device_fields(getfield(vps, :source)),
          (w_sum = getfield(vps, :w_sum), prescribed_v = getfield(vps, :prescribed_v)))

@inline function device_view(vps::VirtualParticleSystem{T,ND}) where {T,ND}
    nt = _device_fields(vps)
    return DeviceVirtualSystem{T, ND, typeof(nt)}(nt)
end
