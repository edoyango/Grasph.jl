export EOSUpdater, TaitEOSUpdater, LinearEOSUpdater, ViscoPlasticMCStressUpdater,
       ZeroFieldUpdater, ElastoPlasticStressUpdater

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
    p_b::T
end
TaitEOSUpdater(rho0::T, p_b::T = T(0.0)) where {T<:AbstractFloat} = TaitEOSUpdater{T}(rho0, p_b)

@inline @Base.propagate_inbounds function (u::TaitEOSUpdater)(ps::AbstractParticleSystem{T}, i::Int) where {T}
    p = ps.p ; rho = ps.rho
    rho0 = T(u.rho0) ; c = ps.c ; p_b = u.p_b
    p[i] = rho0 * c * c / T(7) * ((rho[i] / rho0)^7 - one(T)) + p_b
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
    stress[i] = (cohesion + pval*tan(friction_angle))*strain_rate[i]/mag_strain_rate
    if nelem == 3
        stress[i] -= SVector(pval, pval, zT)
    elseif nelem == 4
        stress[i] -= SVector(pval, pval, pval, zT)
    elseif nelem == 6
        stress[i] -= SVector(pval, pval, pval, zT, zT, zT)
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
    nu       = T(u.nu)
    phi      = T(u.phi)
    psi      = T(u.psi)
    cohesion = T(u.cohesion)
    dt       = T(u.dt)

    # Step 1 — Read current state
    σ  = ps.stress[i]       # SVector{4,T}: [xx, yy, zz, xy]
    ε̇  = ps.strain_rate[i]  # SVector{4,T}
    ω  = ps.vorticity[i]    # T (scalar, 2D: ω_xy component)

    σ_xx = σ[1];  σ_yy = σ[2];  σ_zz = σ[3];  σ_xy = σ[4]
    ε̇_xx = ε̇[1];  ε̇_yy = ε̇[2];  ε̇_zz = ε̇[3];  ε̇_xy = ε̇[4]

    # Step 2 — Strain increment
    d_xx = ε̇_xx * dt
    d_yy = ε̇_yy * dt
    d_zz = ε̇_zz * dt
    d_xy = ε̇_xy * dt

    # Step 3 — Elastic stress increment (DE * deps, analytical — no matrix)
    D0 = E / ((one(T) + nu) * (one(T) - 2*nu))

    dsig_xx = D0 * ((one(T) - nu)*d_xx + nu*d_yy + nu*d_zz)
    dsig_yy = D0 * (nu*d_xx + (one(T) - nu)*d_yy + nu*d_zz)
    dsig_zz = D0 * (nu*d_xx + nu*d_yy + (one(T) - nu)*d_zz)
    dsig_xy = D0 * (one(T) - 2*nu) * d_xy

    # Step 4 — Jaumann rate correction (uses old stress σ_n, not trial)
    ω_dt     = ω * dt
    dsig_xx -= 2*σ_xy * ω_dt
    dsig_yy += 2*σ_xy * ω_dt
    dsig_xy += (σ_xx - σ_yy) * ω_dt
    # dsig_zz unchanged — no out-of-plane vorticity in 2D plane strain

    # Step 5 — Trial stress
    t_xx = σ_xx + dsig_xx
    t_yy = σ_yy + dsig_yy
    t_zz = σ_zz + dsig_zz
    t_xy = σ_xy + dsig_xy

    # Step 6 — Drucker-Prager yield function
    sqrt3 = sqrt(T(3))
    α_φ = 2*sin(phi) / (sqrt3 * (3 - sin(phi)))
    k_c = 6*cohesion*cos(phi) / (sqrt3 * (3 - sin(phi)))

    I1 = t_xx + t_yy + t_zz
    p3 = I1 / 3

    s_xx = t_xx - p3
    s_yy = t_yy - p3
    s_zz = t_zz - p3
    s_xy = t_xy

    J2   = T(0.5) * (s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2)
    sqJ2 = sqrt(max(J2, zero(T)))

    f = α_φ * I1 + sqJ2 - k_c

    # Step 7 — Plastic return (only if f > 0)
    deps_p_xx = zero(T)
    deps_p_yy = zero(T)
    deps_p_zz = zero(T)
    deps_p_xy = zero(T)

    if f > zero(T)
        α_ψ = 2*sin(psi) / (sqrt3 * (3 - sin(psi)))

        inv2sqJ2 = inv(2 * sqJ2)

        df_xx = α_φ + s_xx * inv2sqJ2
        df_yy = α_φ + s_yy * inv2sqJ2
        df_zz = α_φ + s_zz * inv2sqJ2
        df_xy =        s_xy * inv2sqJ2

        dg_xx = α_ψ + s_xx * inv2sqJ2
        dg_yy = α_ψ + s_yy * inv2sqJ2
        dg_zz = α_ψ + s_zz * inv2sqJ2
        dg_xy =        s_xy * inv2sqJ2

        # DE * dg/dσ (analytical)
        DEg_xx = D0 * ((one(T) - nu)*dg_xx + nu*dg_yy + nu*dg_zz)
        DEg_yy = D0 * (nu*dg_xx + (one(T) - nu)*dg_yy + nu*dg_zz)
        DEg_zz = D0 * (nu*dg_xx + nu*dg_yy + (one(T) - nu)*dg_zz)
        DEg_xy = D0 * (one(T) - 2*nu) * dg_xy

        # Denominator: df' * DE * D2 * dg (factor of 2 on shear = D2 Voigt convention)
        denom = df_xx*DEg_xx + df_yy*DEg_yy + df_zz*DEg_zz + 2*df_xy*DEg_xy

        dlambda = f / denom

        deps_p_xx = dlambda * dg_xx
        deps_p_yy = dlambda * dg_yy
        deps_p_zz = dlambda * dg_zz
        deps_p_xy = dlambda * dg_xy

        # Stress correction: dsig -= DE * deps_p (analytical)
        dsig_xx -= D0 * ((one(T) - nu)*deps_p_xx + nu*deps_p_yy + nu*deps_p_zz)
        dsig_yy -= D0 * (nu*deps_p_xx + (one(T) - nu)*deps_p_yy + nu*deps_p_zz)
        dsig_zz -= D0 * (nu*deps_p_xx + nu*deps_p_yy + (one(T) - nu)*deps_p_zz)
        dsig_xy -= D0 * (one(T) - 2*nu) * deps_p_xy
    end

    # Step 8 — Write back
    ps.stress[i]   = SVector(σ_xx + dsig_xx,
                              σ_yy + dsig_yy,
                              σ_zz + dsig_zz,
                              σ_xy + dsig_xy)
    ps.strain[i]   += SVector(d_xx, d_yy, d_zz, d_xy)
    ps.strain_p[i] += SVector(deps_p_xx, deps_p_yy, deps_p_zz, deps_p_xy)
end
