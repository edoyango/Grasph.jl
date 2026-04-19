export EOSUpdater, TaitEOSUpdater, LinearEOSUpdater, ViscoPlasticMCStressUpdater,
       ZeroFieldUpdater, ElastoPlasticStressUpdater, VirtualNormUpdater

# ---------------------------------------------------------------------------
# EOS state updater functors
#
# Callable structs (`(u::UpdaterType)(ps, i)`) that encapsulate physics parameters.
# Calling `u(ps, i)` updates particle `i` in-place.
# Backend-agnostic: work unchanged on CPU and GPU.
#
# Usage:
#
#   u  = TaitEOSUpdater(rho0)
#   ps = FluidParticleSystem(..., state_updater=u)
# ---------------------------------------------------------------------------

abstract type StateUpdater end

"""
    ZeroFieldUpdater{Syms}()

Functor that resets one or more fields to zero for each particle `i`.
Use as a stage-1 state updater to zero derivative or accumulator fields
before the interaction sweep.

    ZeroFieldUpdater(:strain_rate)
    ZeroFieldUpdater(:strain_rate, :vorticity)
"""
struct ZeroFieldUpdater{Syms} <: StateUpdater end

ZeroFieldUpdater(fields::Symbol...) = ZeroFieldUpdater{fields}()

@inline @Base.propagate_inbounds (u::ZeroFieldUpdater{Syms})(ps, i::Int) where {Syms} =
    _zero_fields!(ps, i, Syms)

@inline _zero_fields!(ps, i, ::Tuple{}) = nothing
@inline @Base.propagate_inbounds function _zero_fields!(ps, i, syms::Tuple)
    arr = getproperty(ps, first(syms))
    arr[i] = zero(eltype(arr))
    _zero_fields!(ps, i, Base.tail(syms))
end

"""
    VirtualNormUpdater(v_mult::SVector{ND,T}, :field1, :field2, …)

State updater for `VirtualParticleSystem`. On each call:

1. Divides every listed field by `ps.w_sum[i]` (SPH normalisation).
2. Multiplies `ps.v[i]` component-wise by `v_mult` (boundary condition).

`v_mult` encodes the velocity boundary condition:
- `SVector(-1,-1,-1)` — fully fixed (no-slip, all components negated)
- `SVector(1,1,-1)`   — free-slip with wall normal along z (negate z only)

Fields for particles with zero `w_sum` are set to zero.
"""
struct VirtualNormUpdater{Syms, ND, T<:AbstractFloat} <: StateUpdater
    v_mult::SVector{ND,T}
end

VirtualNormUpdater(v_mult::SVector{ND,T}, fields::Symbol...) where {ND,T<:AbstractFloat} =
    VirtualNormUpdater{fields, ND, T}(v_mult)

@inline @Base.propagate_inbounds function (u::VirtualNormUpdater{Syms,ND,T})(ps, i::Int) where {Syms,ND,T}
    _normalize_fields!(ps, i, ps.w_sum[i], u.v_mult, getfield(ps, :prescribed_v), Syms)
end

@inline _normalize_fields!(ps, i, w, v_mult, prescribed_v, ::Tuple{}) = nothing
@inline @Base.propagate_inbounds function _normalize_fields!(ps, i, w, v_mult, prescribed_v, syms::Tuple)
    arr = getproperty(ps, first(syms))
    val = iszero(w) ? zero(eltype(arr)) : arr[i] / w
    arr[i] = first(syms) === :v ? val .* v_mult .+ prescribed_v : val
    _normalize_fields!(ps, i, w, v_mult, prescribed_v, Base.tail(syms))
end

abstract type EOSUpdater <: StateUpdater end

"""
    TaitEOSUpdater{T} <: EOSUpdater

Functor that applies the Tait equation of state to particle `i`:

    p[i] = p0 * ((rho[i] / rho0)^7 - 1) + p_b

Construct with p0, rho0 and p_b, then pass to `state_updater` in `FluidParticleSystem` constructor:

    u  = TaitEOSUpdater(p0, rho0, p_b)
"""
struct TaitEOSUpdater{T} <: EOSUpdater
    rho0::T
    gamma::T
end
TaitEOSUpdater(rho0::T; gamma::T = T(7)) where {T<:AbstractFloat} = TaitEOSUpdater{T}(rho0, gamma)

@inline @Base.propagate_inbounds function (u::TaitEOSUpdater{T})(ps::AbstractParticleSystem{T}, i::Int) where {T}
    p = ps.p ; rho = ps.rho
    rho0 = T(u.rho0) ; c = ps.c ; gamma = u.gamma
    p[i] = rho0 * c * c / gamma * ((rho[i] / rho0)^gamma - one(T))
end

"""
    LinearEOSUpdater{T} <: EOSUpdater

Functor that applies a linear (acoustic) equation of state to particle `i`:

    p[i] = c^2 * (rho[i] - rho0)

Construct with rho0, then pass to `state_updater` in `FluidParticleSystem` constructor:

    u  = LinearEOSUpdater(rho0)
"""
struct LinearEOSUpdater{T} <: EOSUpdater
    rho0::T
end
LinearEOSUpdater(rho0::T) where {T<:AbstractFloat} = LinearEOSUpdater{T}(rho0)

@inline @Base.propagate_inbounds function (u::LinearEOSUpdater)(ps::AbstractParticleSystem{T}, i::Int) where {T}
    p = ps.p ; rho = ps.rho
    rho0 = T(u.rho0) ; c = ps.c
    p[i] = c * c * (rho[i] - rho0)
end

abstract type StressUpdater <: StateUpdater end

struct ViscoPlasticMCStressUpdater{T, EOS} <: StressUpdater
    eos::EOS
    friction_angle::T
    cohesion::T
end

function ViscoPlasticMCStressUpdater(eos, friction_angle::Real, cohesion::Real)
    T = float(promote_type(typeof(friction_angle), typeof(cohesion)))
    ViscoPlasticMCStressUpdater{T, typeof(eos)}(eos, T(friction_angle), T(cohesion))
end

@inline @Base.propagate_inbounds function (u::ViscoPlasticMCStressUpdater)(ps::AbstractParticleSystem{T}, i::Int) where {T}
    u.eos(ps, i)
    stress = ps.stress ; p = ps.p ; strain_rate = ps.strain_rate
    friction_angle = T(u.friction_angle) ; cohesion = T(u.cohesion)
    nelem = length(eltype(stress))
    pval  = p[i]
    zT    = zero(pval)
    if nelem == 3 # x, y, xy
        mag_strain_rate = strain_rate[i][1]^2 + strain_rate[i][2]^2 + 2*strain_rate[i][3]^2
    elseif nelem == 4 # x, y, z, xy
        mag_strain_rate = strain_rate[i][1]^2 + strain_rate[i][2]^2 + 2*strain_rate[i][4]^2
    elseif nelem == 6 # x, y, z, xy, xz, yz
        mag_strain_rate = strain_rate[i][1]^2 + strain_rate[i][2]^2 + strain_rate[i][3]^2 + 2*(strain_rate[i][4]^2 + strain_rate[i][5]^2 + strain_rate[i][6]^2)
    end
    mag_strain_rate = max(sqrt(mag_strain_rate), eps(typeof(mag_strain_rate)))
    stress[i] = T(2.0)*(cohesion + pval*tan(friction_angle))*strain_rate[i]/mag_strain_rate
    
    # subtract pressure and then emulate stress surface apex at σ_11 = σ_22 = σ_33 = 0
    if nelem == 3
        stress[i] -= SVector(pval, pval, zT)
        if stress[i][1] + stress[i][2] > 0.0
            stress[i] = zero(eltype(stress))
        end
    elseif nelem == 4
        stress[i] -= SVector(pval, pval, pval, zT)
        if stress[i][1] + stress[i][2] + stress[i][3] > 0.0
            stress[i] = zero(eltype(stress))
        end
    elseif nelem == 6
        stress[i] -= SVector(pval, pval, pval, zT, zT, zT)
        if stress[i][1] + stress[i][2] + stress[i][3] > 0.0
            stress[i] = zero(eltype(stress))
        end
    end
end

"""
    ElastoPlasticStressUpdater{T} <: StressUpdater

Functor for an elasto-plastic stress update with Drucker-Prager yield and
non-associative plastic flow (dilation angle ψ ≠ friction angle φ).

Implements a Jaumann-corrected elastic predictor / plastic corrector scheme
in Voigt notation with NS=4 components (plane-strain: xx, yy, zz, xy).

    upd = ElastoPlasticStressUpdater(E, nu, phi, psi, cohesion, dt)
"""
struct ElastoPlasticStressUpdater{T} <: StressUpdater
    E::T
    nu::T
    phi::T
    psi::T
    cohesion::T
    dt::T
end

function ElastoPlasticStressUpdater(E::Real, nu::Real, phi::Real, psi::Real,
                                    cohesion::Real, dt::Real)
    T = float(promote_type(typeof(E), typeof(nu), typeof(phi),
                           typeof(psi), typeof(cohesion), typeof(dt)))
    ElastoPlasticStressUpdater{T}(T(E), T(nu), T(phi), T(psi), T(cohesion), T(dt))
end

@inline @Base.propagate_inbounds function (u::ElastoPlasticStressUpdater)(
        ps::AbstractParticleSystem{T}, i::Int) where {T}

    @assert length(eltype(ps.stress)) == 4 "ElastoPlasticStressUpdater requires NS=4 (plane-strain Voigt). Got NS=$(length(eltype(ps.stress)))."

    E        = T(u.E)
    ν        = T(u.nu)
    φ        = T(u.phi)
    ψ        = T(u.psi)
    cohesion = T(u.cohesion)
    dt       = T(u.dt)

    # Step 1 — Read current state
    σ  = ps.stress[i]       # SVector{4,T}: [xx, yy, zz, xy]
    ε̇  = ps.strain_rate[i]  # SVector{4,T}
    ω  = ps.vorticity[i]    # T (scalar, 2D: ω_xy component)

    σ_xx = σ[1];  σ_yy = σ[2];  σ_zz = σ[3];  σ_xy = σ[4]
    ε̇_xx = ε̇[1];  ε̇_yy = ε̇[2];  ε̇_zz = ε̇[3];  ε̇_xy = ε̇[4]

    # Step 2 — Strain increment
    dε_xx = ε̇_xx * dt
    dε_yy = ε̇_yy * dt
    dε_zz = ε̇_zz * dt
    dε_xy = ε̇_xy * dt

    # Step 3 — Elastic stress increment (DE * deps, analytical — no matrix)
    D0 = E / ((one(T) + ν) * (one(T) - 2*ν))

    dσ_xx = D0 * ((one(T) - ν)*dε_xx + ν*dε_yy + ν*dε_zz)
    dσ_yy = D0 * (ν*dε_xx + (one(T) - ν)*dε_yy + ν*dε_zz)
    dσ_zz = D0 * (ν*dε_xx + ν*dε_yy + (one(T) - ν)*dε_zz)
    dσ_xy = D0 * (one(T) - 2*ν) * dε_xy

    # Step 4 — Jaumann rate correction (uses old stress σ_n, not trial)
    ω_dt     = ω * dt
    dσ_xx -= 2*σ_xy * ω_dt
    dσ_yy += 2*σ_xy * ω_dt
    dσ_xy += (σ_xx - σ_yy) * ω_dt
    # dσ_zz unchanged — no out-of-plane vorticity in 2D plane strain

    # Step 5 — Trial stress
    σ_t_xx = σ_xx + dσ_xx
    σ_t_yy = σ_yy + dσ_yy
    σ_t_zz = σ_zz + dσ_zz
    σ_t_xy = σ_xy + dσ_xy

    # Step 6 — Drucker-Prager yield function
    sqrt3 = sqrt(T(3))
    α_φ = 2*sin(φ) / (sqrt3 * (3 - sin(φ)))
    k_c = 6*cohesion*cos(φ) / (sqrt3 * (3 - sin(φ)))

    I1 = σ_t_xx + σ_t_yy + σ_t_zz
    p3 = I1 / 3

    s_xx = σ_t_xx - p3
    s_yy = σ_t_yy - p3
    s_zz = σ_t_zz - p3
    s_xy = σ_t_xy

    J2   = T(0.5) * (s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2)
    sqJ2 = sqrt(J2)

    f = α_φ * I1 + sqJ2 - k_c

    # Step 7 — Plastic return (only if f > 0)
    dε_p_xx = zero(T)
    dε_p_yy = zero(T)
    dε_p_zz = zero(T)
    dε_p_xy = zero(T)

    if f > zero(T)
        α_ψ = 2*sin(ψ) / (sqrt3 * (3 - sin(ψ)))

        inv2sqJ2 = inv(2 * sqJ2)

        dfdσ_xx = α_φ + s_xx * inv2sqJ2
        dfdσ_yy = α_φ + s_yy * inv2sqJ2
        dfdσ_zz = α_φ + s_zz * inv2sqJ2
        dfdσ_xy =       s_xy * inv2sqJ2

        dgdσ_xx = α_ψ + s_xx * inv2sqJ2
        dgdσ_yy = α_ψ + s_yy * inv2sqJ2
        dgdσ_zz = α_ψ + s_zz * inv2sqJ2
        dgdσ_xy =       s_xy * inv2sqJ2

        # DE * dg/dσ (analytical)
        DE_dgdσ_xx = D0 * ((one(T) - ν)*dgdσ_xx + ν*dgdσ_yy + ν*dgdσ_zz)
        DE_dgdσ_yy = D0 * (ν*dgdσ_xx + (one(T) - ν)*dgdσ_yy + ν*dgdσ_zz)
        DE_dgdσ_zz = D0 * (ν*dgdσ_xx + ν*dgdσ_yy + (one(T) - ν)*dgdσ_zz)
        DE_dgdσ_xy = D0 * (one(T) - 2*ν) * dgdσ_xy

        # Denominator: df' * DE * D2 * dg (factor of 2 on shear = D2 Voigt convention)
        denom = dfdσ_xx*DE_dgdσ_xx + dfdσ_yy*DE_dgdσ_yy + dfdσ_zz*DE_dgdσ_zz + 2*dfdσ_xy*DE_dgdσ_xy

        dλ = f / denom

        dε_p_xx = dλ * dgdσ_xx
        dε_p_yy = dλ * dgdσ_yy
        dε_p_zz = dλ * dgdσ_zz
        dε_p_xy = dλ * dgdσ_xy

        # Stress correction: dσ -= DE * deps_p (analytical)
        dσ_xx -= D0 * ((one(T) - ν)*dε_p_xx + ν*dε_p_yy + ν*dε_p_zz)
        dσ_yy -= D0 * (ν*dε_p_xx + (one(T) - ν)*dε_p_yy + ν*dε_p_zz)
        dσ_zz -= D0 * (ν*dε_p_xx + ν*dε_p_yy + (one(T) - ν)*dε_p_zz)
        dσ_xy -= D0 * (one(T) - 2*ν) * dε_p_xy
    end

    # Step 8 — Write back
    ps.stress[i]   = SVector(σ_xx + dσ_xx,
                             σ_yy + dσ_yy,
                             σ_zz + dσ_zz,
                             σ_xy + dσ_xy)
    ps.strain[i]   += SVector(dε_xx, dε_yy, dε_zz, dε_xy)
    ps.strain_p[i] += SVector(dε_p_xx, dε_p_yy, dε_p_zz, dε_p_xy)
end
