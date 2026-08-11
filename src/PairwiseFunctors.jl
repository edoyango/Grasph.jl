export FluidPfn, StrainRatePfn, StrainRateVorticityPfn, CauchyFluidPfn, XSPHPfn, InterpolateFieldFn, FluidSolidPfn, NeighborCountFn

# ---------------------------------------------------------------------------
# Premade pairwise interaction functors
#
# Ready-to-use callable structs built from the primitives in PairwisePhysics.jl.
# Construct and pass directly to SystemInteraction:
#
#   si = SystemInteraction(kernel, FluidPfn(alpha, beta, h), ps)
# ---------------------------------------------------------------------------

"""
    AbstractPairwiseFunctor

Common supertype for every pairwise functor (pfn) struct in this file. A pfn
only ever needs to author its physics once — as `pfn_contribution` methods
for any one-sided (`WritesA`/`WritesB`) case, or as `pfn_contribution_pair`
methods for any case that writes both sides of a pair (every self-interaction,
plus any coupled case with `_onesided_shape(...) = WritesBoth()`) — and gets
a working `ColouredCPU`/`ColouredKA` mutating callable for free, via the two
generic delegate methods defined right after `_onesided_writeback_coupled!`
below. `pfn_contribution_pair` exists (rather than deriving every "both
sides" case from two separate `pfn_contribution` calls) so the shared
per-pair arithmetic — viscosity, pressure combination, density-diffusion
terms — is computed exactly once and reused for both sides, matching what a
hand-written two-sided mutating method would do; `pfn_contribution` itself is
then trivially `first ∘ pfn_contribution_pair` wherever a pair exists (see
the two generic derivations below), so nothing needs to author both.
"""
abstract type AbstractPairwiseFunctor end

"""
    StrainRatePfn()

Pairwise functor that accumulates the symmetric strain rate tensor onto both
particles in Voigt notation.
"""
struct StrainRatePfn <: AbstractPairwiseFunctor end

# --- One-sided `pfn_contribution`/`pfn_contribution_pair` methods ---

@inline @Base.propagate_inbounds function pfn_contribution_pair(f::StrainRatePfn, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps.strain_rate))
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = ps.v[j] - ps.v[i]

    sr = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j),), (strain_rate = sr * (mass / rho_i),)
end

@inline _onesided_zero_self(::StrainRatePfn, ps::AbstractParticleSystem{T,ND}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps.strain_rate)),)

# Coupled generic (one-sided) — ghosts and virtual systems. Narrowly typed
# (not `::AbstractParticleSystem`) for the same reason as FluidPfn's
# equivalent method: every actual call site (grep-verified) targets a ghost
# or virtual system_b, never a second genuinely-real dynamic system. Writes
# only ps_a, so this stays a plain (non-paired) `pfn_contribution` — no
# shared-computation-reuse question arises when there's only one side.
@inline @Base.propagate_inbounds function pfn_contribution(f::StrainRatePfn, ps_a::AbstractParticleSystem{T,ND}, ps_b::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    rho_j = ps_b.rho[j]
    mass  = ps_b.mass
    dv    = ps_b.v[j] - ps_a.v[i]

    sr = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j),)
end

@inline _onesided_zero_coupled(::StrainRatePfn, ps_a::AbstractParticleSystem{T,ND}, ::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps_a.strain_rate)),)

# Coupled dynamic boundary — derives velocity from distance ratio
@inline @Base.propagate_inbounds function pfn_contribution(f::StrainRatePfn, ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    da    = dot(ps_a.x[i] - ps_b.boundary_point, ps_b.boundary_normal)
    db    = dot(ps_b.x[j] - ps_b.boundary_point, ps_b.boundary_normal)
    vi    = ps_a.v[i]
    vj    = -min(ps_b.boundary_beta, abs(db/da)) * vi
    rho_j = ps_a.rho[i]
    mass  = ps_a.mass
    dv    = vj - vi

    sr = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j),)
end

@inline _onesided_zero_coupled(::StrainRatePfn, ps_a::AbstractParticleSystem{T,ND}, ::DynamicBoundarySystem{T,ND}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps_a.strain_rate)),)

"""
    StrainRateVorticityPfn()

Pairwise functor that accumulates both the symmetric strain rate tensor AND
the spin tensor (vorticity) onto the particles.
"""
struct StrainRateVorticityPfn <: AbstractPairwiseFunctor end

# --- One-sided `pfn_contribution`/`pfn_contribution_pair` methods ---

@inline @Base.propagate_inbounds function pfn_contribution_pair(f::StrainRateVorticityPfn, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps.strain_rate))
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = ps.v[j] - ps.v[i]

    sr  = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)
    vor = vorticity_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j), vorticity = vor * (mass / rho_j)),
           (strain_rate = sr * (mass / rho_i), vorticity = vor * (mass / rho_i))
end

@inline _onesided_zero_self(::StrainRateVorticityPfn, ps::AbstractParticleSystem{T,ND}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps.strain_rate)), vorticity = zero(eltype(ps.vorticity)))

# Coupled generic (one-sided) — ghosts and virtual systems (see StrainRatePfn's
# equivalent method for why this is narrowly typed).
@inline @Base.propagate_inbounds function pfn_contribution(f::StrainRateVorticityPfn, ps_a::AbstractParticleSystem{T,ND}, ps_b::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    rho_j = ps_b.rho[j]
    mass  = ps_b.mass
    dv    = ps_b.v[j] - ps_a.v[i]

    sr  = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)
    vor = vorticity_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j), vorticity = vor * (mass / rho_j))
end

@inline _onesided_zero_coupled(::StrainRateVorticityPfn, ps_a::AbstractParticleSystem{T,ND}, ::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps_a.strain_rate)), vorticity = zero(eltype(ps_a.vorticity)))

# Coupled dynamic boundary
@inline @Base.propagate_inbounds function pfn_contribution(f::StrainRateVorticityPfn, ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    da    = dot(ps_a.x[i] - ps_b.boundary_point, ps_b.boundary_normal)
    db    = dot(ps_b.x[j] - ps_b.boundary_point, ps_b.boundary_normal)
    vi    = ps_a.v[i]
    vj    = -min(ps_b.boundary_beta, abs(db/da)) * vi
    rho_j = ps_a.rho[i]
    mass  = ps_a.mass
    dv    = vj - vi

    sr  = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)
    vor = vorticity_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j), vorticity = vor * (mass / rho_j))
end

@inline _onesided_zero_coupled(::StrainRateVorticityPfn, ps_a::AbstractParticleSystem{T,ND}, ::DynamicBoundarySystem{T,ND}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps_a.strain_rate)), vorticity = zero(eltype(ps_a.vorticity)))

"""
    FluidPfn{S, D, E, T}

Pairwise functor for weakly-compressible SPH fluid interaction.

`D` is `Nothing` (no density diffusion) or `T` (δ-SPH density diffusion with
that coefficient). Pass `delta=<value>` to the constructor to enable it.
"""
struct FluidPfn{S, D, E, T<:AbstractFloat} <: AbstractPairwiseFunctor
    art_visc_alpha::T
    art_visc_beta::T
    h::T
    delta::D
    epsilon::E
end
function FluidPfn(alpha, beta, h; sigma=2, delta=nothing, epsilon=nothing)
    a, b, c = promote(float(alpha), float(beta), float(h))
    T = typeof(a)
    d = delta === nothing ? nothing : T(delta)
    e = epsilon === nothing ? nothing : T(epsilon)
    FluidPfn{sigma, typeof(d), typeof(e), T}(a, b, c, d, e)
end

# ---------------------------------------------------------------------------
# One-sided `pfn_contribution`/`pfn_contribution_pair` methods (see
# Interaction.jl for the protocol and the `onesided=true` sweep that calls
# these).
#
# `pfn_contribution` returns "the contribution to i from j" as a NamedTuple.
# `pfn_contribution_pair` returns "the contribution to i" AND "the
# contribution to j" as a 2-tuple of NamedTuples, computing whatever shared
# per-pair terms both sides need exactly once — it exists for every
# self-interaction (always both-sided) and every `WritesBoth`-shaped coupled
# case; `pfn_contribution` is then just `first` of that pair (see the two
# generic derivations below `_onesided_writeback_coupled!`), so a `WritesBoth`
# pfn only ever authors the pair version, never both.
#
# `_onesided_zero_self`/`_onesided_zero_coupled` (pfn-specific: they must
# construct a zero value of whatever type/shape that pfn's `pfn_contribution`
# returns) and `_onesided_writeback_self!`/`_onesided_writeback_coupled!`
# (fully generic below — every writeback is just `ps.field[i] += acc.field`
# per field, so a single NamedTuple-dispatched method covers all pfns) round
# out the per-pfn hooks the sweep needs.
# ---------------------------------------------------------------------------

@inline _onesided_writeback_fields!(::Tuple{}, ps, i, acc) = nothing
@inline @Base.propagate_inbounds function _onesided_writeback_fields!(names::Tuple, ps, i, acc)
    fname = first(names)
    getproperty(ps, fname)[i] += getproperty(acc, fname)
    _onesided_writeback_fields!(Base.tail(names), ps, i, acc)
end

@inline @Base.propagate_inbounds _onesided_writeback_self!(pfn, ps, i, acc::NamedTuple{names}) where {names} =
    _onesided_writeback_fields!(names, ps, i, acc)

@inline @Base.propagate_inbounds _onesided_writeback_coupled!(pfn, ps_a, ps_b, i, acc::NamedTuple{names}) where {names} =
    _onesided_writeback_fields!(names, ps_a, i, acc)

# `pfn_contribution` is trivially derived from `pfn_contribution_pair`
# wherever a pair exists — generic over every `AbstractPairwiseFunctor`, so
# no pfn needs to author both. Untyped on `ps`/`ps_a`/`ps_b`: any pfn with
# its own narrowly-typed `pfn_contribution` method (every `WritesA`/`WritesB`-
# only case in this file) is strictly more specific and is picked instead —
# these two only ever apply where a `pfn_contribution_pair` method exists and
# nothing more specific does (every self case; the `WritesBoth` coupled
# cases below).
@inline @Base.propagate_inbounds pfn_contribution(f::AbstractPairwiseFunctor, ps, i::Int, j::Int, dx, gx, w) =
    first(pfn_contribution_pair(f, ps, i, j, dx, gx, w))
@inline @Base.propagate_inbounds pfn_contribution(f::AbstractPairwiseFunctor, ps_a, ps_b, i::Int, j::Int, dx, gx, w) =
    first(pfn_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w))

# ---------------------------------------------------------------------------
# Generic ColouredCPU/ColouredKA callables — every pfn's physics is authored
# exactly once (`pfn_contribution`/`pfn_contribution_pair` above); these two
# methods give EVERY AbstractPairwiseFunctor subtype a working two-sided
# mutating callable for free, by delegating into that same authored physics
# instead of a second, independently hand-written implementation.
#
# Self case: always symmetric (a single system) — matches how the onesided
# self-sweep already achieves symmetry "for free" via full-neighbourhood
# traversal, with no shape dispatch at all (Interaction.jl's self dispatch
# has no `_onesided_shape` lookup, unlike the coupled case). One
# `pfn_contribution_pair` call computes both sides' writes from shared terms
# computed once, matching a hand-written mutating method's cost.
#
# Coupled case: dispatches on the SAME `_onesided_shape` trait the onesided
# sweep already uses (`WritesA`/`WritesB`/`WritesBoth`, defined in
# Interaction.jl) — no new trait needed. `WritesA`/`WritesB` need only one
# side, so they call `pfn_contribution` directly (nothing to share). The
# `-dx, -gx` swap used by the `WritesB` branch is exactly what the onesided
# sweep's own reverse pass relies on (`dx = xb[j] - xa[i]` when roles are
# swapped) — applied here at the single-pair level using the already-computed
# `dx`/`gx` (exact IEEE-754 negation, no re-derivation). `WritesBoth` calls
# `pfn_contribution_pair` once, same reasoning as the self case.
# ---------------------------------------------------------------------------

@inline @Base.propagate_inbounds function (f::AbstractPairwiseFunctor)(ps, i::Int, j::Int, dx, gx, w)
    acc_i, acc_j = pfn_contribution_pair(f, ps, i, j, dx, gx, w)
    _onesided_writeback_self!(f, ps, i, acc_i)
    _onesided_writeback_self!(f, ps, j, acc_j)
    nothing
end

@inline @Base.propagate_inbounds function (f::AbstractPairwiseFunctor)(ps_a, ps_b, i::Int, j::Int, dx, gx, w)
    _apply_coupled_shape!(_onesided_shape(f, ps_a, ps_b), f, ps_a, ps_b, i, j, dx, gx, w)
    nothing
end

@inline @Base.propagate_inbounds _apply_coupled_shape!(::WritesA, f, ps_a, ps_b, i, j, dx, gx, w) =
    _onesided_writeback_coupled!(f, ps_a, ps_b, i, pfn_contribution(f, ps_a, ps_b, i, j, dx, gx, w))
@inline @Base.propagate_inbounds _apply_coupled_shape!(::WritesB, f, ps_a, ps_b, i, j, dx, gx, w) =
    _onesided_writeback_coupled!(f, ps_b, ps_a, j, pfn_contribution(f, ps_b, ps_a, j, i, -dx, -gx, w))
@inline @Base.propagate_inbounds function _apply_coupled_shape!(::WritesBoth, f, ps_a, ps_b, i, j, dx, gx, w)
    acc_a, acc_b = pfn_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)
    _onesided_writeback_coupled!(f, ps_a, ps_b, i, acc_a)
    _onesided_writeback_coupled!(f, ps_b, ps_a, j, acc_b)
    nothing
end

@inline @Base.propagate_inbounds function pfn_contribution_pair(f::FluidPfn{S,D,E,T}, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T}
    vi, vj       = ps.v[i], ps.v[j]
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    p_i, p_j     = ps.p[i], ps.p[j]
    mass         = ps.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps.c, ps.c)
    dh     = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    dv_tmp = mass * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps.c, ps.c, f.h, f.h, gx, f.delta)
    drho_i = mass * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    drho_j = mass * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)

    return (dvdt = dv_tmp, drhodt = drho_i), (dvdt = -dv_tmp, drhodt = drho_j)
end

@inline _onesided_zero_self(::FluidPfn{S,D,E,T}, ::AbstractParticleSystem{T,ND}, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Coupled static boundary — already one-sided in its two-sided form (a static
# boundary has no dynamics), so this is a direct transcription: return instead
# of mutate.
@inline @Base.propagate_inbounds function pfn_contribution(f::FluidPfn{S,D,E,T}, ps_a::AbstractParticleSystem{T,ND}, ps_b::StaticBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    rf  = lennard_jones(dx, ps_b.lj_cutoff, ps_a.c, 12, 6)
    return (dvdt = -mass_j * piv * gx + rf * dx,)
end

@inline _onesided_zero_coupled(::FluidPfn{S,D,E,T}, ::AbstractParticleSystem{T,ND}, ::StaticBoundarySystem, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}),)

# Coupled generic (one-sided) — ghosts and virtual systems. Narrowly typed
# (not `::AbstractParticleSystem`) on purpose: once a reverse-pass sweep
# exists (see Interaction.jl), a bare `(Abstract,Abstract)` method here would
# silently absorb any mistyped reverse call instead of throwing MethodError.
@inline @Base.propagate_inbounds function pfn_contribution(f::FluidPfn{S,D,E,T}, ps_a::AbstractParticleSystem{T,ND}, ps_b::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    dh  = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    dv_tmp = mass_j * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    drho = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::FluidPfn{S,D,E,T}, ::AbstractParticleSystem{T,ND}, ::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Coupled dynamic boundary (derives velocity, pressure-based)
@inline @Base.propagate_inbounds function pfn_contribution(f::FluidPfn{S,D,E,T}, ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    da = dot(ps_a.x[i] - ps_b.boundary_point, ps_b.boundary_normal)
    db = dot(ps_b.x[j] - ps_b.boundary_point, ps_b.boundary_normal)

    vi       = ps_a.v[i]
    vj       = -min(ps_b.boundary_beta, abs(db/da)) * vi
    rho_i    = ps_a.rho[i]
    rho_j    = rho_i
    p_i      = ps_a.p[i]
    p_j      = p_i
    mass     = ps_a.mass
    dv       = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    dh  = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    dv_tmp = mass * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    drho = mass * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::FluidPfn{S,D,E,T}, ::AbstractParticleSystem{T,ND}, ::DynamicBoundarySystem{T,ND}, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Coupled real-real fluid-fluid (mutual, WritesBoth) — two distinct
# FluidParticleSystem instances (e.g. bubble.jl/bubble2.jl/bubble3.jl's two
# phases) interact symmetrically under relabeling: each side's contribution
# depends only on its own p/rho/mass plus the other side's p/rho/mass, so the
# sweep can call this one method for both the forward pass (system_a in the
# ps_a slot) and the reverse pass (system_b in the ps_a slot) — see the
# WritesBoth dispatcher in Interaction.jl. Includes the
# artificial-surface-tension term (`ast`) that the self/ghost variants omit.
_onesided_shape(::FluidPfn, ::FluidParticleSystem, ::FluidParticleSystem) = WritesBoth()

# Shared by the host (FluidParticleSystem) and device_view (DeviceSystem)
# methods below — both need the exact same formula, only the dispatch type
# on ps_a/ps_b differs. ps_a/ps_b are left untyped here on purpose: Julia
# specializes per call site, so this costs nothing at runtime, and it's what
# lets a single body serve both the host-typed and device-viewed entry points
# without duplicating the arithmetic. Computes the shared viscosity/pressure/
# diffusion terms once and derives both sides from them, matching what a
# hand-written mutating method would do.
@inline @Base.propagate_inbounds function _fluidpfn_fluidfluid_contribution_pair(f::FluidPfn{S,D,E,T}, ps_a, ps_b, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    ast    = artificial_surface_tension_coeff(f.epsilon, p_i, p_j, rho_i, rho_j)
    dv_tmp = (dh + ast - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    drho_a = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    drho_b = mass_i * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)

    return (dvdt = mass_j * dv_tmp, drhodt = drho_a), (dvdt = -mass_i * dv_tmp, drhodt = drho_b)
end

@inline @Base.propagate_inbounds pfn_contribution_pair(f::FluidPfn{S,D,E,T}, ps_a::FluidParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat} =
    _fluidpfn_fluidfluid_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::FluidPfn{S,D,E,T}, ::FluidParticleSystem{T,ND}, ::FluidParticleSystem{T,ND}, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# ka=true twin: device_view(::FluidParticleSystem) erases concrete-type
# identity (every "bare" system type collapses to the same generic
# DeviceSystem), so this must be typed on DeviceSystem{T,ND,FluidParticleSystem}
# specifically (see DeviceViews.jl's Kind parameter) rather than the loose
# AbstractParticleSystem{T,ND} every other coupled FluidPfn method uses — a
# looser type here would also match a device-viewed BasicParticleSystem/
# StressParticleSystem/ElastoPlasticParticleSystem paired with a device-viewed
# fluid, silently computing fluid-fluid physics for a pairing that was never
# meant to interact that way, instead of throwing MethodError.
@inline @Base.propagate_inbounds pfn_contribution_pair(f::FluidPfn{S,D,E,T}, ps_a::DeviceSystem{T,ND,FluidParticleSystem}, ps_b::DeviceSystem{T,ND,FluidParticleSystem}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat} =
    _fluidpfn_fluidfluid_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::FluidPfn{S,D,E,T}, ::DeviceSystem{T,ND,FluidParticleSystem}, ::DeviceSystem{T,ND,FluidParticleSystem}, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

"""
    CauchyFluidPfn{D, T}

Pairwise functor for SPH fluid self-interaction driven by a Cauchy stress tensor.

`D` is `Nothing` (no density diffusion) or `T` (δ-SPH density diffusion with
that coefficient). Pass `delta=<value>` to the constructor to enable it.
"""
struct CauchyFluidPfn{D, T<:AbstractFloat} <: AbstractPairwiseFunctor
    art_visc_alpha::T
    art_visc_beta::T
    h::T
    delta::D
end
function CauchyFluidPfn(alpha, beta, h; delta=nothing)
    a, b, c = promote(float(alpha), float(beta), float(h))
    T = typeof(a)
    d = delta === nothing ? nothing : T(delta)
    CauchyFluidPfn{typeof(d), T}(a, b, c, d)
end

# --- One-sided `pfn_contribution`/`pfn_contribution_pair` methods ---
# A genuinely independent real-real coupling (two distinct dynamic systems,
# as opposed to the ghost/virtual/boundary cases below) has no converted
# `pfn_contribution_pair` and is unsupported under EITHER sweep mode: grep of
# the 13 experiment scripts confirms every coupled CauchyFluidPfn call site
# targets a DynamicBoundarySystem, VirtualParticleSystem, or ghost system, so
# this was already dead code. Unlike FluidPfn's fluid-fluid case, there is no
# single obviously-correct narrow real-system type pair to anchor a WritesBoth
# override to here (FluidPfn's `FluidParticleSystem,FluidParticleSystem` is
# grounded in real usage; nothing analogous exists for CauchyFluidPfn), and a
# loosely-typed `AbstractParticleSystem,AbstractParticleSystem` override would
# incorrectly also match — and silently steal dispatch from — the ghost/
# virtual/boundary cases below. Left unimplemented rather than guessed.

@inline @Base.propagate_inbounds function pfn_contribution_pair(f::CauchyFluidPfn{D,T}, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    vi, vj             = ps.v[i], ps.v[j]
    rho_i, rho_j       = ps.rho[i], ps.rho[j]
    stress_i, stress_j = ps.stress[i], ps.stress[j]
    mass               = ps.mass
    dv                 = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps.c, ps.c)
    h_vec  = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    dv_tmp = mass * (h_vec - piv * gx)

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps.c, ps.c, f.h, f.h, gx, f.delta)
    drho_i = mass * (dr + psi / rho_j)
    drho_j = mass * (dr - psi / rho_i)

    return (dvdt = dv_tmp, drhodt = drho_i), (dvdt = -dv_tmp, drhodt = drho_j)
end

@inline _onesided_zero_self(::CauchyFluidPfn{D,T}, ::AbstractParticleSystem{T,ND}, i) where {D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Coupled generic (one-sided) — virtual or ghost ps_b. Narrowly typed on
# purpose (not `::AbstractParticleSystem`) — see the note on FluidPfn's
# equivalent method above for why.
@inline @Base.propagate_inbounds function pfn_contribution(f::CauchyFluidPfn{D,T}, ps_a::AbstractParticleSystem{T,ND}, ps_b::Union{AbstractVirtualParticleSystem{T,ND}, AbstractGhostParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    vi, vj             = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j       = ps_a.rho[i], ps_b.rho[j]
    stress_i, stress_j = ps_a.stress[i], ps_b.stress[j]
    mass_j             = ps_b.mass
    dv                 = vi - vj

    piv   = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    h_vec = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    dv_tmp = mass_j * (h_vec - piv * gx)

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    drho = mass_j * (dr + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::CauchyFluidPfn{D,T}, ::AbstractParticleSystem{T,ND}, ::Union{AbstractVirtualParticleSystem{T,ND}, AbstractGhostParticleSystem{T,ND}}, i) where {D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Coupled dynamic boundary (derives velocity, stress-based)
@inline @Base.propagate_inbounds function pfn_contribution(f::CauchyFluidPfn{D,T}, ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    da = dot(ps_a.x[i] - ps_b.boundary_point, ps_b.boundary_normal)
    db = dot(ps_b.x[j] - ps_b.boundary_point, ps_b.boundary_normal)

    vi       = ps_a.v[i]
    vj       = -min(ps_b.boundary_beta, abs(db/da)) * vi
    rho_i    = ps_a.rho[i]
    rho_j    = rho_i
    stress_i = ps_a.stress[i]
    stress_j = stress_i
    mass     = ps_a.mass
    dv       = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    h_vec  = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    dv_tmp = mass * (h_vec - piv * gx)

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    drho = mass * (dr + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::CauchyFluidPfn{D,T}, ::AbstractParticleSystem{T,ND}, ::DynamicBoundarySystem{T,ND}, i) where {D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

"""
    XSPHPfn{T}

XSPH velocity adjustment: nudges each particle's velocity a fraction
`epsilon` of the way toward the local mass-weighted mean of its neighbours'
velocities. Grep-verified real use, both in bubble3.jl:
  - Self-coupled (`fluid_X_interaction`, `fluid_Y_interaction`): the
    single-system method right below.
  - Ghost-coupled (`fluid_boundary_interaction`, system_b=`boundary_ghost`):
    the narrowly-typed one-sided method further down. See its comment for
    why that method exists at all — it fixes a pre-existing bug.
"""
struct XSPHPfn{T<:AbstractFloat} <: AbstractPairwiseFunctor
    epsilon::T
end

# --- One-sided `pfn_contribution`/`pfn_contribution_pair` methods ---

@inline @Base.propagate_inbounds function pfn_contribution_pair(f::XSPHPfn{T}, ps::AbstractParticleSystem, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj       = ps.v[i], ps.v[j]
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = vi - vj
    epsilon      = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    return (v_adjustment = du * mass,), (v_adjustment = -du * mass,)
end

@inline _onesided_zero_self(::XSPHPfn{T}, ps::AbstractParticleSystem{T,ND}, i) where {T,ND} =
    (v_adjustment = zero(SVector{ND,T}),)

# Coupled generic (one-sided) — ghosts and virtual systems. Fixes a
# pre-existing bug: every real ghost in this codebase self-references its
# source (`GhostParticleSystem(fluid_X, ...)` — the ghost's `source` IS
# `fluid_X`; see bubble3.jl's `boundary_ghost`), and `GhostParticleSystem`
# doesn't own a `v_adjustment` array, so under a fully generic two-sided
# treatment `ps_b.v_adjustment[j] -= du*mass_i` would fall through
# `getproperty` straight to `ghost.source.v_adjustment[j]` — aliasing back
# into the real system's own array, but indexed by the ghost's LOCAL index
# j, which does not correspond to the real particle the ghost mirrors. That
# was silently wrong (writes landing on unrelated particles) whenever
# ghost.n < fluid.n, and out-of-bounds (heap corruption/SIGABRT) whenever
# ghost.n > fluid.n — hit by bubble3.jl's `fluid_boundary_interaction`
# (velocity_adjust_pairwise_fn=XSPHPfn(0.5), system_b=boundary_ghost).
#
# Narrowly typed (not `::AbstractParticleSystem`, matching
# FluidPfn/CauchyFluidPfn/StrainRatePfn's equivalent methods in this file) so
# this only ever writes ps_a — never ps_b — matching every other ghost-coupled
# pfn's convention here: ghosts/virtuals aren't independently integrated, so a
# ps_b write is never meaningful for them regardless of aliasing; there is no
# correct index to write to even in principle. `_onesided_shape` is left at
# its default `WritesA()` since this only ever writes ps_a.
@inline @Base.propagate_inbounds function pfn_contribution(f::XSPHPfn{T}, ps_a::AbstractParticleSystem, ps_b::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_j       = ps_b.mass
    dv           = vi - vj
    epsilon      = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    return (v_adjustment = du * mass_j,)
end

@inline _onesided_zero_coupled(::XSPHPfn{T}, ps_a::AbstractParticleSystem{T,ND}, ::Union{AbstractGhostParticleSystem{T,ND},AbstractVirtualParticleSystem{T,ND}}, i) where {T,ND} =
    (v_adjustment = zero(SVector{ND,T}),)

# Coupled real-real (mutual, WritesBoth) — kept for a genuinely independent
# real-real coupling (e.g. two distinct FluidParticleSystem instances
# interacting mutually, mirroring FluidPfn's own fluid-fluid case). Grep of
# bubble.jl/bubble2.jl/bubble3.jl confirms no experiment script currently
# passes XSPHPfn as `velocity_adjust_pairwise_fn` for such a pairing
# (bubble3.jl's `fluid_XY_interaction`, the only mutual two-real-fluid
# interaction that exists, never sets `velocity_adjust_pairwise_fn`) — but
# this is still correct and is the right one to fall back to if one ever does.
_onesided_shape(::XSPHPfn, ::FluidParticleSystem, ::FluidParticleSystem) = WritesBoth()

@inline @Base.propagate_inbounds function _xsphpfn_fluidfluid_contribution_pair(f::XSPHPfn{T}, ps_a, ps_b, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv           = vi - vj
    epsilon      = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    return (v_adjustment = du * mass_j,), (v_adjustment = -du * mass_i,)
end

@inline @Base.propagate_inbounds pfn_contribution_pair(f::XSPHPfn{T}, ps_a::FluidParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat} =
    _xsphpfn_fluidfluid_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::XSPHPfn{T}, ::FluidParticleSystem{T,ND}, ::FluidParticleSystem{T,ND}, i) where {T,ND} =
    (v_adjustment = zero(SVector{ND,T}),)

# ka=true twin — same reasoning as FluidPfn's fluid-fluid twin above
# (device_view erases concrete-type identity on bare system types).
@inline @Base.propagate_inbounds pfn_contribution_pair(f::XSPHPfn{T}, ps_a::DeviceSystem{T,ND,FluidParticleSystem}, ps_b::DeviceSystem{T,ND,FluidParticleSystem}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat} =
    _xsphpfn_fluidfluid_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::XSPHPfn{T}, ::DeviceSystem{T,ND,FluidParticleSystem}, ::DeviceSystem{T,ND,FluidParticleSystem}, i) where {T,ND} =
    (v_adjustment = zero(SVector{ND,T}),)

"""
    InterpolateFieldFn(:field1, :field2, …)

Pairwise functor that accumulates the standard SPH field interpolation

    f_j += (m_i / ρ_i) * f_i * W_ij

into the virtual particle system `ps_b` from real particles `ps_a`.
Use with a coupled interaction where `ps_a` is the real (source) system
and `ps_b` is the VirtualParticleSystem being filled.

Zero the target fields before the sweep to obtain the SPH estimate.
"""
struct InterpolateFieldFn{fields, ACC_WSUM} <: AbstractPairwiseFunctor
    InterpolateFieldFn(fields::Symbol...; accumulate_wsum::Bool=true) =
        new{fields, accumulate_wsum}()
end

# --- One-sided `pfn_contribution` method ---
#
# No self-interaction and no WritesBoth case exists for this pfn (it only
# ever copies data from a real source into a virtual/probe target), so it
# never needs `pfn_contribution_pair` — a plain `pfn_contribution` is already
# the single authored implementation.
#
# Every script call site couples a real source system as system_a against a
# virtual or probe target as system_b (Trapdoor.jl, EP_ColumnCollapse2.jl):
# `_onesided_shape = WritesB()` for that shape. The reverse sweep's call
# convention puts the write target (the virtual/probe system_b) in the
# "ps_a" position and the read-only neighbour (real system_a) in "ps_b" —
# same as every other coupled pfn_contribution method — so this reads
# exactly like accumulating "the neighbour's field, weighted", written back
# into `ps_a[i]` (the target) by the generic writeback below. The same method
# also directly covers a virtual/probe-as-ps_a, real-as-ps_b call (the
# default `WritesA()` shape) with identical semantics — the target is
# whichever side is virtual/probe, regardless of which slot it's in.
#
# `fields` is a runtime-opaque, compile-time-known tuple of symbols (part of
# the type), so contributions are built the same way the plain field-copy
# helpers below mutate them: recursively over the tuple, unrolled at compile
# time.

_interp_values(::Tuple{}, ps, j, kw) = ()
@inline @Base.propagate_inbounds function _interp_values(fields::Tuple, ps, j, kw)
    fname = first(fields)
    return (kw * getproperty(ps, fname)[j], _interp_values(Base.tail(fields), ps, j, kw)...)
end

_interp_zeros(::Tuple{}, ps) = ()
@inline @Base.propagate_inbounds function _interp_zeros(fields::Tuple, ps)
    fname = first(fields)
    return (zero(eltype(getproperty(ps, fname))), _interp_zeros(Base.tail(fields), ps)...)
end

_onesided_shape(::InterpolateFieldFn, ::AbstractParticleSystem, ::Union{AbstractVirtualParticleSystem,AbstractProbeParticleSystem}) = WritesB()

@inline @Base.propagate_inbounds function pfn_contribution(::InterpolateFieldFn{fields, ACC_WSUM}, ps_a::Union{AbstractVirtualParticleSystem{T,ND},AbstractProbeParticleSystem{T,ND}}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    kw = w * (ps_b.mass / ps_b.rho[j])
    vals = _interp_values(fields, ps_b, j, kw)
    if ACC_WSUM
        return NamedTuple{(fields..., :w_sum)}((vals..., kw))
    else
        return NamedTuple{fields}(vals)
    end
end

@inline function _onesided_zero_coupled(::InterpolateFieldFn{fields, ACC_WSUM}, ps_a::Union{AbstractVirtualParticleSystem{T,ND},AbstractProbeParticleSystem{T,ND}}, ::AbstractParticleSystem{T,ND}, i) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    zeros_ = _interp_zeros(fields, ps_a)
    if ACC_WSUM
        return NamedTuple{(fields..., :w_sum)}((zeros_..., zero(T)))
    else
        return NamedTuple{fields}(zeros_)
    end
end

"""
    NeighborCountFn(field::Symbol)

Pairwise functor that increments `probe.field[j]` by 1 for every source
particle `i` that falls within the kernel support of probe particle `j`.

Use with a coupled interaction where `system_a` is the real (source) system
and `system_b` is a `ProbeParticleSystem`.  Zero the target array before the
sweep (handled automatically by `auto_zero_probe!` in the driver).
"""
struct NeighborCountFn{field} <: AbstractPairwiseFunctor end
NeighborCountFn(field::Symbol) = NeighborCountFn{field}()

# --- One-sided `pfn_contribution` method ---
#
# No self-interaction and no WritesBoth case (same reasoning as
# InterpolateFieldFn above) — never needs `pfn_contribution_pair`.
#
# Only ever used as (system_a=real, system_b=probe), so this is WritesB():
# the reverse sweep's call convention puts the write target (probe) in the
# "ps_a" position, matching that orientation.

_onesided_shape(::NeighborCountFn, ::AbstractParticleSystem, ::AbstractProbeParticleSystem) = WritesB()

@inline @Base.propagate_inbounds function pfn_contribution(::NeighborCountFn{field}, probe::AbstractProbeParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {field,ND,T<:AbstractFloat}
    return NamedTuple{(field,)}((one(eltype(getproperty(probe, field))),))
end

@inline _onesided_zero_coupled(::NeighborCountFn{field}, probe::AbstractProbeParticleSystem, ::AbstractParticleSystem, i) where {field} =
    NamedTuple{(field,)}((zero(eltype(getproperty(probe, field))),))

"""
    FluidSolidPfn{S, D, T}

Coupled pairwise functor for weakly-compressible fluid interacting with a
linear-elastic solid.

Identical to `FluidPfn` except that the fluid pressure `p_i` is used for
**both** sides of the pressure force, ensuring a continuous pressure field
across the fluid-solid interface:

    dvdt contribution ∝ (p_i/ρ_i² + p_i/ρ_j²) ∇W

Artificial viscosity and the continuity equation use each particle's own
density and sound speed. Only the coupled real-real (fluid, solid) pairing
is supported; the fluid must be `system_a` and the solid `system_b` (or vice
versa — both orderings are supported, see the `WritesBoth` methods below).
"""
struct FluidSolidPfn{S, D, T<:AbstractFloat} <: AbstractPairwiseFunctor
    art_visc_alpha::T
    art_visc_beta::T
    h::T
    delta::D
end

function FluidSolidPfn(alpha, beta, h; sigma=2, delta=nothing)
    a, b, c = promote(float(alpha), float(beta), float(h))
    T = typeof(a)
    d = delta === nothing ? nothing : T(delta)
    FluidSolidPfn{sigma, typeof(d), T}(a, b, c, d)
end

# ---------------------------------------------------------------------------
# One-sided `pfn_contribution_pair` methods — coupled real-real fluid-solid
# (mutual, WritesBoth; used by DambreakWall.jl's fluid/wall interaction). No
# self-interaction exists for this pfn.
#
# This physics is NOT symmetric under relabeling: the pressure term must
# always use the FLUID's own pressure for both sides (that is the entire
# point of FluidSolidPfn — a continuous pressure field across the interface),
# never the solid's own pressure. A single generic
# (AbstractParticleSystem, AbstractParticleSystem) method keyed off "ps_a's
# own p" — the pattern that works for FluidPfn — would silently use the
# solid's pressure whenever the reverse sweep puts the solid in the ps_a
# slot. So two narrowly-typed methods instead, one per physical assignment of
# the fluid/solid roles to the (ps_a, ps_b) slots, each explicitly reading
# the fluid's pressure from wherever the fluid actually is. No generic
# fallback is provided on purpose: a missing or mistyped call must throw
# MethodError rather than silently compute with the wrong side's pressure.
# Each host-typed method has a device_view-typed (`DeviceSystem{T,ND,Kind}`)
# twin below it for the same reason FluidPfn's fluid-fluid method needed one
# (see DeviceViews.jl's Kind parameter) — here it also happens to be what
# disambiguates the two asymmetric directions from each other once
# device_view is involved, not just from unrelated pairings.
# ---------------------------------------------------------------------------

_onesided_shape(::FluidSolidPfn, ::FluidParticleSystem, ::ElastoPlasticParticleSystem) = WritesBoth()
_onesided_shape(::FluidSolidPfn, ::ElastoPlasticParticleSystem, ::FluidParticleSystem) = WritesBoth()

# Shared by the host-typed and device_view-typed methods below — same
# reasoning as FluidPfn's fluid-fluid twin above: ps_a/ps_b left untyped so
# Julia specializes per call site, letting one body serve both entry points.
# Fluid is the target (ps_a): p_i is the fluid's own pressure, read directly.
@inline @Base.propagate_inbounds function _fluidsolidpfn_fluid_a_contribution_pair(f::FluidSolidPfn{S,D,T}, ps_a, ps_b, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i          = ps_a.p[i]       # fluid pressure used for both sides
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_i, rho_i, rho_j, Val(S))
    dv_tmp = (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    drho_a = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    drho_b = mass_i * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)

    return (dvdt = mass_j * dv_tmp, drhodt = drho_a), (dvdt = -mass_i * dv_tmp, drhodt = drho_b)
end

@inline @Base.propagate_inbounds pfn_contribution_pair(f::FluidSolidPfn{S,D,T}, ps_a::FluidParticleSystem{T,ND}, ps_b::ElastoPlasticParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat} =
    _fluidsolidpfn_fluid_a_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::FluidSolidPfn{S,D,T}, ::FluidParticleSystem{T,ND}, ::ElastoPlasticParticleSystem{T,ND}, i) where {S,D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# ka=true twin: device_view erases concrete-type identity on bare system
# types (DeviceViews.jl), so — exactly like FluidPfn's fluid-fluid twin —
# this must be typed on the specific DeviceSystem{T,ND,Kind} pairing rather
# than loose AbstractParticleSystem{T,ND}, so a device-viewed pairing this
# pfn was never meant to accept (e.g. two fluids, or a fluid paired with a
# device-viewed BasicParticleSystem) still throws MethodError instead of
# silently computing with the wrong side's pressure.
@inline @Base.propagate_inbounds pfn_contribution_pair(f::FluidSolidPfn{S,D,T}, ps_a::DeviceSystem{T,ND,FluidParticleSystem}, ps_b::DeviceSystem{T,ND,ElastoPlasticParticleSystem}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat} =
    _fluidsolidpfn_fluid_a_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::FluidSolidPfn{S,D,T}, ::DeviceSystem{T,ND,FluidParticleSystem}, ::DeviceSystem{T,ND,ElastoPlasticParticleSystem}, i) where {S,D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Solid is the target (ps_a): the fluid is now in ps_b, so its pressure must
# be read from ps_b.p[j] — the solid's own pressure (ps_a.p[i]) must never
# appear in this formula.
@inline @Base.propagate_inbounds function _fluidsolidpfn_solid_a_contribution_pair(f::FluidSolidPfn{S,D,T}, ps_a, ps_b, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_fluid      = ps_b.p[j]       # fluid pressure used for both sides
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_fluid, p_fluid, rho_i, rho_j, Val(S))
    dv_tmp = (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    drho_a = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    drho_b = mass_i * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)

    return (dvdt = mass_j * dv_tmp, drhodt = drho_a), (dvdt = -mass_i * dv_tmp, drhodt = drho_b)
end

@inline @Base.propagate_inbounds pfn_contribution_pair(f::FluidSolidPfn{S,D,T}, ps_a::ElastoPlasticParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat} =
    _fluidsolidpfn_solid_a_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::FluidSolidPfn{S,D,T}, ::ElastoPlasticParticleSystem{T,ND}, ::FluidParticleSystem{T,ND}, i) where {S,D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# ka=true twin — same reasoning as the fluid-as-ps_a twin above.
@inline @Base.propagate_inbounds pfn_contribution_pair(f::FluidSolidPfn{S,D,T}, ps_a::DeviceSystem{T,ND,ElastoPlasticParticleSystem}, ps_b::DeviceSystem{T,ND,FluidParticleSystem}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat} =
    _fluidsolidpfn_solid_a_contribution_pair(f, ps_a, ps_b, i, j, dx, gx, w)

@inline _onesided_zero_coupled(::FluidSolidPfn{S,D,T}, ::DeviceSystem{T,ND,ElastoPlasticParticleSystem}, ::DeviceSystem{T,ND,FluidParticleSystem}, i) where {S,D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))
