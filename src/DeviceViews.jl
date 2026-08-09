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

# ---------------------------------------------------------------------------
# GhostParticleSystem — same shape as DeviceVirtualSystem above: subtypes
# AbstractGhostParticleSystem so every pfn_contribution/_onesided_zero_coupled
# method already narrowly typed on that abstraction (ghosts are always a
# read-only ps_b, never a write target — see item 8's Kind-parameter note for
# why narrow typing matters here too) dispatches into the device view
# unmodified.
#
# Only the fields pfn_contribution/the sweep kernel actually read off a
# ghost-as-ps_b are captured: `x`/`v`/`rho` (owned, first-class),
# `mass`/`c` (scalars, forwarded from source), plus whatever `extras` fields
# (`p`, `stress`, …) the specific coupling reads. `idx_original`/
# `idx_boundary`/`normals` are pure ghost-generation bookkeeping — consumed
# by generate_ghosts!/update_ghost_kinematics!/GhostCopier, never by a pfn —
# so they're deliberately left out of the device view.
#
# `n` is captured as a concrete Int at view-construction time (ghost.n reads
# the `count` Ref then), same as every other _device_fields method — the
# kernel that launches with this view uses it only for `ndrange`/loop bounds,
# never mutates it, so a stale snapshot is exactly what's wanted.
# ---------------------------------------------------------------------------

struct DeviceGhostSystem{T, ND, NT<:NamedTuple} <: AbstractGhostParticleSystem{T, ND}
    _f::NT
end

@inline function Base.getproperty(ds::DeviceGhostSystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(getfield(ds, :_f), s)
end

function Adapt.adapt_structure(to, ds::DeviceGhostSystem{T,ND}) where {T,ND}
    nt = Adapt.adapt(to, getfield(ds, :_f))
    DeviceGhostSystem{T, ND, typeof(nt)}(nt)
end

@inline _device_fields(ghost::GhostParticleSystem) =
    merge((n = ghost.n, x = getfield(ghost, :x), v = getfield(ghost, :v), rho = getfield(ghost, :rho),
           mass = ghost.mass, c = ghost.c),
          getfield(ghost, :extras))

@inline function device_view(ghost::GhostParticleSystem{T,ND}) where {T,ND}
    nt = _device_fields(ghost)
    return DeviceGhostSystem{T, ND, typeof(nt)}(nt)
end

# ---------------------------------------------------------------------------
# ProbeParticleSystem — same shape as DeviceVirtualSystem/DeviceGhostSystem
# above: subtypes AbstractProbeParticleSystem so every pfn_contribution/
# _onesided_zero_coupled/_onesided_shape method already narrowly typed on that
# abstraction (InterpolateFieldFn/NeighborCountFn's WritesB() methods — a
# probe is always a write target, never a read-only neighbour) dispatches
# into the device view unmodified.
#
# `id`/`mirror_target`/`_print_fields` are pure host bookkeeping — sort-order
# identity, the (possibly non-isbits) mirror source, print-field names — never
# read by a pfn or state updater via `ps.field`, so they're left out, same
# reasoning as ghost's idx_original/idx_boundary/normals. `mass`/`c`/`rho` are
# deliberately absent too: every probe-target pfn method reads the *neighbour*
# system's mass/rho (`ps_b.mass`, `ps_b.rho[j]`), never the probe's own — a
# probe has no physical mass or density to begin with.
# ---------------------------------------------------------------------------

struct DeviceProbeSystem{T, ND, NT<:NamedTuple} <: AbstractProbeParticleSystem{T, ND}
    _f::NT
end

@inline function Base.getproperty(ds::DeviceProbeSystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(getfield(ds, :_f), s)
end

function Adapt.adapt_structure(to, ds::DeviceProbeSystem{T,ND}) where {T,ND}
    nt = Adapt.adapt(to, getfield(ds, :_f))
    DeviceProbeSystem{T, ND, typeof(nt)}(nt)
end

@inline _device_fields(probe::ProbeParticleSystem) =
    merge((n = probe.n, x = getfield(probe, :x), w_sum = getfield(probe, :w_sum),
           prescribed_v = getfield(probe, :prescribed_v)),
          getfield(probe, :extras))

@inline function device_view(probe::ProbeParticleSystem{T,ND}) where {T,ND}
    nt = _device_fields(probe)
    return DeviceProbeSystem{T, ND, typeof(nt)}(nt)
end
