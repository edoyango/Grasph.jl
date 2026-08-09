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
    StrainRatePfn()

Pairwise functor that accumulates the symmetric strain rate tensor onto both
particles in Voigt notation.
"""
struct StrainRatePfn end

@inline @Base.propagate_inbounds function (f::StrainRatePfn)(ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps.strain_rate))
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = ps.v[j] - ps.v[i]
    
    if N == 4
        sr = strain_rate_tensor(dv, gx, Val{4})
    else
        sr = strain_rate_tensor(dv, gx)
    end
    
    ps.strain_rate[i] += sr * (mass / rho_j)
    ps.strain_rate[j] += sr * (mass / rho_i)
end

# Coupled generic (one-sided) — covers ghosts and any real system_b
@inline @Base.propagate_inbounds function (f::StrainRatePfn)(ps_a::AbstractParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    rho_j = ps_b.rho[j]
    mass  = ps_b.mass
    dv    = ps_b.v[j] - ps_a.v[i]

    if N == 4
        sr = strain_rate_tensor(dv, gx, Val{4})
    else
        sr = strain_rate_tensor(dv, gx)
    end

    ps_a.strain_rate[i] += sr * (mass / rho_j)
end

# Coupled dynamic boundary — derives velocity from distance ratio
@inline @Base.propagate_inbounds function (f::StrainRatePfn)(ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    da    = dot(ps_a.x[i] - ps_b.boundary_point, ps_b.boundary_normal)
    db    = dot(ps_b.x[j] - ps_b.boundary_point, ps_b.boundary_normal)
    vi    = ps_a.v[i]
    vj    = -min(ps_b.boundary_beta, abs(db/da)) * vi
    rho_j = ps_a.rho[i]
    mass  = ps_a.mass
    dv    = vj - vi

    if N == 4
        sr = strain_rate_tensor(dv, gx, Val{4})
    else
        sr = strain_rate_tensor(dv, gx)
    end

    ps_a.strain_rate[i] += sr * (mass / rho_j)
end

# --- One-sided `pfn_contribution` methods ---

@inline @Base.propagate_inbounds function pfn_contribution(f::StrainRatePfn, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps.strain_rate))
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = ps.v[j] - ps.v[i]

    sr = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j),)
end

@inline _onesided_zero_self(::StrainRatePfn, ps::AbstractParticleSystem{T,ND}, i) where {T,ND} =
    (strain_rate = zero(eltype(ps.strain_rate)),)

# Coupled generic (one-sided) — ghosts and virtual systems. Narrowly typed
# (not `::AbstractParticleSystem`) for the same reason as FluidPfn's
# equivalent method: every actual call site (grep-verified) targets a ghost
# or virtual system_b, never a second genuinely-real dynamic system.
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
struct StrainRateVorticityPfn end

@inline @Base.propagate_inbounds function (f::StrainRateVorticityPfn)(ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps.strain_rate))
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = ps.v[j] - ps.v[i]
    
    sr  = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)
    vor = vorticity_tensor(dv, gx)
    
    ps.strain_rate[i] += sr * (mass / rho_j)
    ps.strain_rate[j] += sr * (mass / rho_i)

    ps.vorticity[i] += vor * (mass / rho_j)
    ps.vorticity[j] += vor * (mass / rho_i)
end

# Coupled generic (one-sided)
@inline @Base.propagate_inbounds function (f::StrainRateVorticityPfn)(ps_a::AbstractParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps_a.strain_rate))
    rho_j = ps_b.rho[j]
    mass  = ps_b.mass
    dv    = ps_b.v[j] - ps_a.v[i]

    sr  = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)
    vor = vorticity_tensor(dv, gx)

    ps_a.strain_rate[i] += sr * (mass / rho_j)
    ps_a.vorticity[i]   += vor * (mass / rho_j)
end

# Coupled dynamic boundary
@inline @Base.propagate_inbounds function (f::StrainRateVorticityPfn)(ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
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

    ps_a.strain_rate[i] += sr * (mass / rho_j)
    ps_a.vorticity[i]   += vor * (mass / rho_j)
end

# --- One-sided `pfn_contribution` methods ---

@inline @Base.propagate_inbounds function pfn_contribution(f::StrainRateVorticityPfn, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    N = length(eltype(ps.strain_rate))
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = ps.v[j] - ps.v[i]

    sr  = N == 4 ? strain_rate_tensor(dv, gx, Val{4}) : strain_rate_tensor(dv, gx)
    vor = vorticity_tensor(dv, gx)

    return (strain_rate = sr * (mass / rho_j), vorticity = vor * (mass / rho_j))
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
struct FluidPfn{S, D, E, T<:AbstractFloat}
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

@inline @Base.propagate_inbounds function (f::FluidPfn{S,D,E,T})(ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T}
    vi, vj       = ps.v[i], ps.v[j]
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    p_i, p_j     = ps.p[i], ps.p[j]
    mass         = ps.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps.c, ps.c)
    dh     = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    dv_tmp = mass * (dh - piv) * gx

    ps.dvdt[i] += dv_tmp
    ps.dvdt[j] -= dv_tmp

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps.c, ps.c, f.h, f.h, gx, f.delta)
    ps.drhodt[i] += mass * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    ps.drhodt[j] += mass * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)
end

@inline @Base.propagate_inbounds function (f::FluidPfn{S,D,E,T})(ps_a::FluidParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    ast    = artificial_surface_tension_coeff(f.epsilon, p_i, p_j, rho_i, rho_j)
    dv_tmp = (dh + ast - piv) * gx

    ps_a.dvdt[i] += mass_j*dv_tmp
    ps_b.dvdt[j] -= mass_i*dv_tmp

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    ps_b.drhodt[j] += mass_i * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)
end

# Coupled generic (one-sided, pressure-based) — covers ghosts
@inline @Base.propagate_inbounds function (f::FluidPfn{S,D,E,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    dh  = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    ps_a.dvdt[i] += mass_j * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
end

# Coupled static boundary (LJ + artificial viscosity)
@inline @Base.propagate_inbounds function (f::FluidPfn{S,D,E,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::StaticBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    rf  = lennard_jones(dx, ps_b.lj_cutoff, ps_a.c, 12, 6)
    ps_a.dvdt[i] += -mass_j * piv * gx + rf * dx
end

# ---------------------------------------------------------------------------
# One-sided `pfn_contribution` methods (see Interaction.jl for the protocol
# and the `onesided=true` sweep that calls these).
#
# Each method below returns "the contribution to i from j" as a NamedTuple,
# instead of mutating ps.dvdt[i]/ps.dvdt[j] in place. It is line-for-line the
# same arithmetic as the corresponding two-sided method above, restricted to
# the i-side terms — no new physics. Newton's third law is recovered by the
# sweep itself calling this same function with (i,j) swapped; see the
# swap-antisymmetry argument in Interaction.jl's "One-sided, particle-parallel
# sweep" section.
#
# `_onesided_zero_self`/`_onesided_zero_coupled` (pfn-specific: they must
# construct a zero value of whatever type/shape that pfn's `pfn_contribution`
# returns) and `_onesided_writeback_self!`/`_onesided_writeback_coupled!`
# (fully generic below — every existing hand-written one was just
# `ps.field[i] += acc.field` per field, so a single NamedTuple-dispatched
# method covers all pfns) round out the per-pfn hooks the sweep needs.
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

@inline @Base.propagate_inbounds function pfn_contribution(f::FluidPfn{S,D,E,T}, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T}
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
    drho = mass * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
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
# WritesBoth dispatcher in Interaction.jl. Line-for-line the ps_a-side terms
# of the two-real-system mutating method above, including the
# artificial-surface-tension term (`ast`) that the self/ghost variants omit.
_onesided_shape(::FluidPfn, ::FluidParticleSystem, ::FluidParticleSystem) = WritesBoth()

@inline @Base.propagate_inbounds function pfn_contribution(f::FluidPfn{S,D,E,T}, ps_a::FluidParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    ast    = artificial_surface_tension_coeff(f.epsilon, p_i, p_j, rho_i, rho_j)
    dv_tmp = mass_j * (dh + ast - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    drho = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::FluidPfn{S,D,E,T}, ::FluidParticleSystem{T,ND}, ::FluidParticleSystem{T,ND}, i) where {S,D,E,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Below: the original two-sided mutating method (kept for the coloured
# sweep's default path).
@inline @Base.propagate_inbounds function (f::FluidPfn{S,D,E,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,E,ND,T<:AbstractFloat}
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
    ps_a.dvdt[i] += mass * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
end

"""
    CauchyFluidPfn{D, T}

Pairwise functor for SPH fluid self-interaction driven by a Cauchy stress tensor.

`D` is `Nothing` (no density diffusion) or `T` (δ-SPH density diffusion with
that coefficient). Pass `delta=<value>` to the constructor to enable it.
"""
struct CauchyFluidPfn{D, T<:AbstractFloat}
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

@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{D,T})(ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    vi, vj             = ps.v[i], ps.v[j]
    rho_i, rho_j       = ps.rho[i], ps.rho[j]
    stress_i, stress_j = ps.stress[i], ps.stress[j]
    mass               = ps.mass
    dv                 = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps.c, ps.c)
    h_vec  = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    dv_tmp = mass * (h_vec - piv * gx)

    ps.dvdt[i] += dv_tmp
    ps.dvdt[j] -= dv_tmp

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps.c, ps.c, f.h, f.h, gx, f.delta)
    ps.drhodt[i] += mass * (dr + psi / rho_j)
    ps.drhodt[j] += mass * (dr - psi / rho_i)
end

# Coupled boundary (one-sided) — virtual or ghost ps_b
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{D,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::Union{VirtualParticleSystem{T,ND}, AbstractGhostParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    vi, vj             = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j       = ps_a.rho[i], ps_b.rho[j]
    stress_i, stress_j = ps_a.stress[i], ps_b.stress[j]
    mass_j             = ps_b.mass
    dv                 = vi - vj

    piv   = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    h_vec = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    ps_a.dvdt[i] += mass_j * (h_vec - piv * gx)

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass_j * (dr + psi / rho_j)
end

# Coupled general (two-sided) — both real particle systems
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{D,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    vi, vj             = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j       = ps_a.rho[i], ps_b.rho[j]
    stress_i, stress_j = ps_a.stress[i], ps_b.stress[j]
    mass_i, mass_j     = ps_a.mass, ps_b.mass
    dv                 = vi - vj

    piv   = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    h_vec = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    ps_a.dvdt[i] += mass_j * (h_vec - piv * gx)
    ps_b.dvdt[j] -= mass_i * (h_vec - piv * gx)

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass_j * (dr + psi / rho_j)
    ps_b.drhodt[j] += mass_i * (dr - psi / rho_i)
end

# Coupled static boundary (LJ + artificial viscosity)
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{D,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::StaticBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    rf  = lennard_jones(dx, ps_b.lj_cutoff, ps_a.c, 12, 6)
    ps_a.dvdt[i] += -mass_j * piv * gx + rf * dx
end

# Coupled dynamic boundary (derives velocity, stress-based)
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{D,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
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
    ps_a.dvdt[i] += dv_tmp

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass * (dr + psi / rho_j)
end

# --- One-sided `pfn_contribution` methods ---
# The general two-real-system method above is not converted: it is unused by
# any of the 13 experiment scripts (every coupled CauchyFluidPfn call site
# targets a DynamicBoundarySystem, VirtualParticleSystem, or ghost system —
# verified by grep), so it stays coloured-sweep-only and untouched.

@inline @Base.propagate_inbounds function pfn_contribution(f::CauchyFluidPfn{D,T}, ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {D,ND,T<:AbstractFloat}
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
    drho = mass * (dr + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
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
struct XSPHPfn{T<:AbstractFloat}
    epsilon::T
end

@inline @Base.propagate_inbounds function (f::XSPHPfn{T})(ps::AbstractParticleSystem, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj             = ps.v[i], ps.v[j]
    rho_i, rho_j       = ps.rho[i], ps.rho[j]
    mass               = ps.mass
    dv                 = vi - vj
    epsilon            = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    ps.v_adjustment[i] += du*mass
    ps.v_adjustment[j] -= du*mass

end

# Coupled generic (two-sided, fully mutual) — kept for a genuinely
# independent real-real coupling (e.g. two distinct FluidParticleSystem
# instances interacting mutually). Grep of bubble.jl/bubble2.jl/bubble3.jl
# confirms no experiment script currently passes XSPHPfn as
# `velocity_adjust_pairwise_fn` for such a pairing (bubble3.jl's
# `fluid_XY_interaction`, the only mutual two-real-fluid interaction that
# exists, never sets `velocity_adjust_pairwise_fn`) — but the method is
# still correct and is the right one to fall back to if one ever does.
#
# It must NOT be used for ghost/virtual system_b — see the narrowly-typed
# method below, which Julia's dispatch picks instead for that case, for why.
@inline @Base.propagate_inbounds function (f::XSPHPfn{T})(ps_a::AbstractParticleSystem, ps_b::AbstractParticleSystem, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj             = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j       = ps_a.rho[i], ps_b.rho[j]
    mass_i, mass_j     = ps_a.mass, ps_b.mass
    dv                 = vi - vj
    epsilon            = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    ps_a.v_adjustment[i] += du*mass_j
    ps_b.v_adjustment[j] -= du*mass_i

end

# Coupled generic (one-sided) — ghosts and virtual systems. Fixes a
# pre-existing bug: every real ghost in this codebase self-references its
# source (`GhostParticleSystem(fluid_X, ...)` — the ghost's `source` IS
# `fluid_X`; see bubble3.jl's `boundary_ghost`), and `GhostParticleSystem`
# doesn't own a `v_adjustment` array, so under the fully generic two-sided
# method above `ps_b.v_adjustment[j] -= du*mass_i` falls through
# `getproperty` straight to `ghost.source.v_adjustment[j]` — aliasing back
# into the real system's own array, but indexed by the ghost's LOCAL index
# j, which does not correspond to the real particle the ghost mirrors. That
# was silently wrong (writes landing on unrelated particles) whenever
# ghost.n < fluid.n, and out-of-bounds (heap corruption/SIGABRT) whenever
# ghost.n > fluid.n — hit by bubble3.jl's `fluid_boundary_interaction`
# (velocity_adjust_pairwise_fn=XSPHPfn(0.5), system_b=boundary_ghost).
#
# Narrowly typed (not `::AbstractParticleSystem`, matching
# FluidPfn/CauchyFluidPfn/StrainRatePfn's equivalent methods in this file)
# so Julia picks this one-sided method over the generic two-sided one for
# ghost/virtual system_b. It only ever writes ps_a — never ps_b — matching
# every other ghost-coupled pfn's convention here: ghosts/virtuals aren't
# independently integrated, so a ps_b write is never meaningful for them
# regardless of aliasing; there is no correct index to write to even in
# principle.
@inline @Base.propagate_inbounds function (f::XSPHPfn{T})(ps_a::AbstractParticleSystem, ps_b::Union{AbstractGhostParticleSystem{T,ND},VirtualParticleSystem{T,ND}}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj             = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j       = ps_a.rho[i], ps_b.rho[j]
    mass_j             = ps_b.mass
    dv                 = vi - vj
    epsilon            = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    ps_a.v_adjustment[i] += du*mass_j

end

# --- One-sided `pfn_contribution` methods ---

@inline @Base.propagate_inbounds function pfn_contribution(f::XSPHPfn{T}, ps::AbstractParticleSystem, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND, T<:AbstractFloat}
    vi, vj       = ps.v[i], ps.v[j]
    rho_i, rho_j = ps.rho[i], ps.rho[j]
    mass         = ps.mass
    dv           = vi - vj
    epsilon      = f.epsilon

    du = xsph_veladjust(epsilon, dv, rho_i, rho_j, w)

    return (v_adjustment = du * mass,)
end

@inline _onesided_zero_self(::XSPHPfn{T}, ps::AbstractParticleSystem{T,ND}, i) where {T,ND} =
    (v_adjustment = zero(SVector{ND,T}),)

# Coupled generic (one-sided) — ghosts and virtual systems. Mirrors the
# mutating method of the same signature above (see its comment for the
# aliasing bug this narrow typing avoids); `_onesided_shape` is left at its
# default `WritesA()` since this, too, only ever writes ps_a.
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

"""
    InterpolateFieldFn(:field1, :field2, …)

Pairwise functor that accumulates the standard SPH field interpolation

    f_j += (m_i / ρ_i) * f_i * W_ij

into the virtual particle system `ps_b` from real particles `ps_a`.
Use with a coupled interaction where `ps_a` is the real (source) system
and `ps_b` is the VirtualParticleSystem being filled.

Zero the target fields before the sweep to obtain the SPH estimate.
"""
struct InterpolateFieldFn{fields, ACC_WSUM}
    InterpolateFieldFn(fields::Symbol...; accumulate_wsum::Bool=true) =
        new{fields, accumulate_wsum}()
end

_interp_fields_ab!(::Tuple{}, ps_a, ps_b, i, j, kw) = nothing
@inline @Base.propagate_inbounds function _interp_fields_ab!(fields::Tuple, ps_a, ps_b, i, j, kw)
    fname = first(fields)
    getproperty(ps_b, fname)[j] += kw * getproperty(ps_a, fname)[i]
    _interp_fields_ab!(Base.tail(fields), ps_a, ps_b, i, j, kw)
end

_interp_fields_ba!(::Tuple{}, ps_a, ps_b, i, j, kw) = nothing
@inline @Base.propagate_inbounds function _interp_fields_ba!(fields::Tuple, ps_a, ps_b, i, j, kw)
    fname = first(fields)
    getproperty(ps_a, fname)[i] += kw * getproperty(ps_b, fname)[j]
    _interp_fields_ba!(Base.tail(fields), ps_a, ps_b, i, j, kw)
end

@inline @Base.propagate_inbounds function (::InterpolateFieldFn{fields, ACC_WSUM})(ps_a::AbstractParticleSystem{T,ND}, ps_b::VirtualParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    kw = w * (ps_a.mass / ps_a.rho[i])
    _interp_fields_ab!(fields, ps_a, ps_b, i, j, kw)
    ACC_WSUM && (ps_b.w_sum[j] += kw)
end

@inline @Base.propagate_inbounds function (::InterpolateFieldFn{fields, ACC_WSUM})(ps_a::VirtualParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    kw = w * (ps_b.mass / ps_b.rho[j])
    _interp_fields_ba!(fields, ps_a, ps_b, i, j, kw)
    ACC_WSUM && (ps_a.w_sum[i] += kw)
end

@inline @Base.propagate_inbounds function (::InterpolateFieldFn{fields, ACC_WSUM})(ps_a::AbstractParticleSystem{T,ND}, ps_b::ProbeParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    kw = w * (ps_a.mass / ps_a.rho[i])
    _interp_fields_ab!(fields, ps_a, ps_b, i, j, kw)
    ACC_WSUM && (ps_b.w_sum[j] += kw)
end

# --- One-sided `pfn_contribution` method ---
#
# Every script call site couples a real source system as system_a against a
# virtual or probe target as system_b (Trapdoor.jl, EP_ColumnCollapse2.jl):
# `_onesided_shape = WritesB()` for that shape. The reverse sweep's call
# convention puts the write target (the virtual/probe system_b) in the
# "ps_a" position and the read-only neighbour (real system_a) in "ps_b" —
# same as every other coupled pfn_contribution method — so this reads
# exactly like the mutating method above with the roles already swapped:
# `kw` uses the *neighbour's* mass/rho (ps_b here), and the interpolated
# value is `kw * neighbour.field[j]`, written back into `ps_a[i]` (the
# target) by the generic writeback below.
#
# `fields` is a runtime-opaque, compile-time-known tuple of symbols (part of
# the type), so contributions are built the same way `_interp_fields_ab!`
# mutates them: recursively over the tuple, unrolled at compile time.

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

_onesided_shape(::InterpolateFieldFn, ::AbstractParticleSystem, ::Union{AbstractVirtualParticleSystem,ProbeParticleSystem}) = WritesB()

@inline @Base.propagate_inbounds function pfn_contribution(::InterpolateFieldFn{fields, ACC_WSUM}, ps_a::Union{AbstractVirtualParticleSystem{T,ND},ProbeParticleSystem{T,ND}}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    kw = w * (ps_b.mass / ps_b.rho[j])
    vals = _interp_values(fields, ps_b, j, kw)
    if ACC_WSUM
        return NamedTuple{(fields..., :w_sum)}((vals..., kw))
    else
        return NamedTuple{fields}(vals)
    end
end

@inline function _onesided_zero_coupled(::InterpolateFieldFn{fields, ACC_WSUM}, ps_a::Union{AbstractVirtualParticleSystem{T,ND},ProbeParticleSystem{T,ND}}, ::AbstractParticleSystem{T,ND}, i) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
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
struct NeighborCountFn{field} end
NeighborCountFn(field::Symbol) = NeighborCountFn{field}()

@inline @Base.propagate_inbounds function (::NeighborCountFn{field})(
    ps_a::AbstractParticleSystem, probe::ProbeParticleSystem,
    i::Int, j::Int, dx, gx, w,
) where {field}
    getproperty(probe, field)[j] += 1
end

# --- One-sided `pfn_contribution` method ---
#
# Only ever used as (system_a=real, system_b=probe), so this is WritesB():
# the reverse sweep's call convention puts the write target (probe) in the
# "ps_a" position, matching the mutating method above with roles swapped.

_onesided_shape(::NeighborCountFn, ::AbstractParticleSystem, ::ProbeParticleSystem) = WritesB()

@inline @Base.propagate_inbounds function pfn_contribution(::NeighborCountFn{field}, probe::ProbeParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {field,ND,T<:AbstractFloat}
    return NamedTuple{(field,)}((one(eltype(getproperty(probe, field))),))
end

@inline _onesided_zero_coupled(::NeighborCountFn{field}, probe::ProbeParticleSystem, ::AbstractParticleSystem, i) where {field} =
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
density and sound speed.  Only the two-sided
`(ps_a::AbstractParticleSystem, ps_b::AbstractParticleSystem)` dispatch is
provided; `ps_a` should be the fluid and `ps_b` the solid.
"""
struct FluidSolidPfn{S, D, T<:AbstractFloat}
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

@inline @Base.propagate_inbounds function (f::FluidSolidPfn{S,D,T})(
    ps_a::AbstractParticleSystem{T,ND},
    ps_b::AbstractParticleSystem{T,ND},
    i::Int, j::Int,
    dx::SVector{ND,T}, gx::SVector{ND,T}, w::T,
) where {S,D,ND,T<:AbstractFloat}
    vi, vj         = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j   = ps_a.rho[i], ps_b.rho[j]
    p_i            = ps_a.p[i]       # fluid pressure used for both sides
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv             = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_i, rho_i, rho_j, Val(S))
    dv_tmp = (dh - piv) * gx

    ps_a.dvdt[i] += mass_j * dv_tmp
    ps_b.dvdt[j] -= mass_i * dv_tmp

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    ps_a.drhodt[i] += mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)
    ps_b.drhodt[j] += mass_i * (dr * continuity_density_coeff(rho_j, rho_i, Val(S)) - psi / rho_i)
end

# ---------------------------------------------------------------------------
# One-sided `pfn_contribution` methods — coupled real-real fluid-solid
# (mutual, WritesBoth; used by DambreakWall.jl's fluid/wall interaction).
#
# Unlike FluidPfn's fluid-fluid case above, this physics is NOT symmetric
# under relabeling: the pressure term must always use the FLUID's own
# pressure for both sides (that is the entire point of FluidSolidPfn — a
# continuous pressure field across the interface), never the solid's own
# pressure. A single generic (AbstractParticleSystem, AbstractParticleSystem)
# method keyed off "ps_a's own p" — the pattern that works for FluidPfn —
# would silently use the solid's pressure whenever the reverse sweep puts the
# solid in the ps_a slot. So two narrowly-typed methods instead, one per
# physical assignment of the fluid/solid roles to the (ps_a, ps_b) slots,
# each explicitly reading the fluid's pressure from wherever the fluid
# actually is. No generic fallback is provided on purpose: a missing or
# mistyped call must throw MethodError rather than silently compute with the
# wrong side's pressure.
# ---------------------------------------------------------------------------

_onesided_shape(::FluidSolidPfn, ::FluidParticleSystem, ::ElastoPlasticParticleSystem) = WritesBoth()
_onesided_shape(::FluidSolidPfn, ::ElastoPlasticParticleSystem, ::FluidParticleSystem) = WritesBoth()

# Fluid is the target (ps_a): p_i is the fluid's own pressure, read directly.
@inline @Base.propagate_inbounds function pfn_contribution(f::FluidSolidPfn{S,D,T}, ps_a::FluidParticleSystem{T,ND}, ps_b::ElastoPlasticParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i          = ps_a.p[i]       # fluid pressure used for both sides
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_i, rho_i, rho_j, Val(S))
    dv_tmp = mass_j * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    drho = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::FluidSolidPfn{S,D,T}, ::FluidParticleSystem{T,ND}, ::ElastoPlasticParticleSystem{T,ND}, i) where {S,D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))

# Solid is the target (ps_a): the fluid is now in ps_b, so its pressure must
# be read from ps_b.p[j] — the solid's own pressure (ps_a.p[i]) must never
# appear in this formula.
@inline @Base.propagate_inbounds function pfn_contribution(f::FluidSolidPfn{S,D,T}, ps_a::ElastoPlasticParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,D,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_fluid      = ps_b.p[j]       # fluid pressure used for both sides
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_fluid, p_fluid, rho_i, rho_j, Val(S))
    dv_tmp = mass_j * (dh - piv) * gx

    dr  = continuity_rate(dv, gx)
    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_b.c, f.h, f.h, gx, f.delta)
    drho = mass_j * (dr * continuity_density_coeff(rho_i, rho_j, Val(S)) + psi / rho_j)

    return (dvdt = dv_tmp, drhodt = drho)
end

@inline _onesided_zero_coupled(::FluidSolidPfn{S,D,T}, ::ElastoPlasticParticleSystem{T,ND}, ::FluidParticleSystem{T,ND}, i) where {S,D,ND,T} =
    (dvdt = zero(SVector{ND,T}), drhodt = zero(T))