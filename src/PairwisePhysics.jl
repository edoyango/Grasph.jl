export artificial_viscosity, pressure_force_coeff, continuity_rate, lennard_jones,
       strain_rate_tensor, vorticity_tensor, cauchy_stress_force, diffusion_density

using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Pairwise SPH physics primitives
#
# Composable `@inline` building blocks for pairwise force functions.
# All functions take SVector arguments so they work in any dimension.
#
# Correspondence with grasph/pairwise_physics.py (numba):
#   artificial_viscosity  ← artificial_viscosity_monaghan1994
#   pressure_force_coeff  ← isotropic_pressure_force
#   continuity_rate       ← continuity_density
#   lennard_jones         ← lennard_jones_repulsive_force
# ---------------------------------------------------------------------------

"""
    artificial_viscosity(dx, dv, h, rho_i, rho_j, alpha, beta, c) -> piv

Monaghan (1994) artificial viscosity coefficient.

`dx = xi - xj` and `dv = vi - vj` are SVectors.
Returns the scalar `piv` to subtract from the pressure gradient coefficient.
"""
@inline function artificial_viscosity(dx::SVector{ND,T}, dv::SVector{ND,T}, h::T, rho_i::T, rho_j::T, alpha::T, beta::T, c_i::T, c_j::T) where {ND,T<:AbstractFloat}
    vr  = min(dot(dx, dv), zero(T))
    muv = h * vr / (dot(dx, dx) + h * h * T(0.01))
    mc = 0.5*(c_i + c_j)
    return (beta * muv - alpha * mc) * muv / (T(0.5) * rho_i * rho_j)
end

"""
    pressure_force_coeff(p_i, p_j, rho_i, rho_j) -> dh

Symmetric isotropic pressure gradient coefficient.

Returns `-(p_i/ρ_i² + p_j/ρ_j²)`, the scalar that multiplies the kernel
gradient vector to give the pressure acceleration contribution.
"""
@inline function pressure_force_coeff(p_i::T, p_j::T, rho_i::T, rho_j::T) where {T<:AbstractFloat}
    return -(p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j))
end

"""
    continuity_rate(dv, gx) -> drho

SPH continuity equation density rate term for a single pair.

Returns `dot(dv, gx)` where `dv = vi - vj` and `gx` is the kernel gradient
vector. Multiply by mass to get the contribution to `drhodt`.
"""
@inline continuity_rate(dv::SVector{ND,T}, gx::SVector{ND,T}) where {ND,T<:AbstractFloat} = dot(dv, gx)

"""
    cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx) -> SVector

Cauchy stress divergence acceleration coefficient for a single particle pair.

Computes `h = (σ_i/ρ_i² + σ_j/ρ_j²)·∇W` where stress is given in Voigt
notation and `gx` is the kernel gradient vector. Multiply by the neighbour
mass to get the acceleration contribution:

    h = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    dvdt[i] += mass_j * h
    dvdt[j] -= mass_i * h

Voigt layouts:
- `SVector{3}`: 2D `(σxx, σyy, σxy)`
- `SVector{4}`: 2D plane-strain `(σxx, σyy, σzz, σxy)` — σzz does not contribute to force
- `SVector{6}`: 3D `(σxx, σyy, σzz, σxy, σxz, σyz)`
"""
@inline function cauchy_stress_force(si::SVector{3,T}, sj::SVector{3,T}, rho_i::T, rho_j::T, gx::SVector{2,T}) where {T<:AbstractFloat}
    ci = one(T) / (rho_i * rho_i)
    cj = one(T) / (rho_j * rho_j)
    SVector(
        (si[1]*gx[1] + si[3]*gx[2]) * ci + (sj[1]*gx[1] + sj[3]*gx[2]) * cj,
        (si[3]*gx[1] + si[2]*gx[2]) * ci + (sj[3]*gx[1] + sj[2]*gx[2]) * cj,
    )
end

@inline function cauchy_stress_force(si::SVector{4,T}, sj::SVector{4,T}, rho_i::T, rho_j::T, gx::SVector{2,T}) where {T<:AbstractFloat}
    ci = one(T) / (rho_i * rho_i)
    cj = one(T) / (rho_j * rho_j)
    SVector(
        (si[1]*gx[1] + si[4]*gx[2]) * ci + (sj[1]*gx[1] + sj[4]*gx[2]) * cj,
        (si[4]*gx[1] + si[2]*gx[2]) * ci + (sj[4]*gx[1] + sj[2]*gx[2]) * cj,
    )
end

@inline function cauchy_stress_force(si::SVector{6,T}, sj::SVector{6,T}, rho_i::T, rho_j::T, gx::SVector{3,T}) where {T<:AbstractFloat}
    ci = one(T) / (rho_i * rho_i)
    cj = one(T) / (rho_j * rho_j)
    SVector(
        (si[1]*gx[1] + si[4]*gx[2] + si[5]*gx[3]) * ci + (sj[1]*gx[1] + sj[4]*gx[2] + sj[5]*gx[3]) * cj,
        (si[4]*gx[1] + si[2]*gx[2] + si[6]*gx[3]) * ci + (sj[4]*gx[1] + sj[2]*gx[2] + sj[6]*gx[3]) * cj,
        (si[5]*gx[1] + si[6]*gx[2] + si[3]*gx[3]) * ci + (sj[5]*gx[1] + sj[6]*gx[2] + sj[3]*gx[3]) * cj,
    )
end

"""
    strain_rate_tensor(dv, gx) -> SVector{3}
    strain_rate_tensor(dv, gx, Val{4}) -> SVector{4}
    strain_rate_tensor(dv::SVector{3}, gx::SVector{3}) -> SVector{6}

Symmetric strain rate tensor contribution for a single particle pair in Voigt
notation. `dv = vj - vi` and `gx` is the kernel gradient vector.

Multiply by `mass / rho` to get the contribution to `strain_rate[i]` or
`strain_rate[j]`.

Output layout by variant:
- 2D (default): `(εxx, εyy, εxy)`
- 2D plane-strain `Val{4}`: `(εxx, εyy, 0, εxy)` — εzz pinned to zero
- 3D: `(εxx, εyy, εzz, εxy, εxz, εyz)`
"""
@inline function strain_rate_tensor(dv::SVector{2,T}, gx::SVector{2,T}) where {T<:AbstractFloat}
    SVector(
        dv[1]*gx[1],
        dv[2]*gx[2],
        T(0.5)*(dv[1]*gx[2] + dv[2]*gx[1]),
    )
end

@inline function strain_rate_tensor(dv::SVector{2,T}, gx::SVector{2,T}, ::Type{Val{4}}) where {T<:AbstractFloat}
    SVector(
        dv[1]*gx[1],
        dv[2]*gx[2],
        zero(T),
        T(0.5)*(dv[1]*gx[2] + dv[2]*gx[1]),
    )
end

@inline function strain_rate_tensor(dv::SVector{3,T}, gx::SVector{3,T}) where {T<:AbstractFloat}
    SVector(
        dv[1]*gx[1],
        dv[2]*gx[2],
        dv[3]*gx[3],
        T(0.5)*(dv[1]*gx[2] + dv[2]*gx[1]),
        T(0.5)*(dv[1]*gx[3] + dv[3]*gx[1]),
        T(0.5)*(dv[2]*gx[3] + dv[3]*gx[2]),
    )
end

"""
    vorticity_tensor(dv, gx)

Spin tensor (vorticity) components for a single particle pair. `dv = vj - vi`
and `gx` is the kernel gradient vector.

Multiply by `mass / rho` to get the contribution to `vorticity[i]` or
`vorticity[j]`.

Output variant:
- 2D: Returns scalar W12 = 0.5 * (dv_x * gx_y - dv_y * gx_x)
- 3D: Returns SVector{3} (W12, W13, W23)
"""
@inline function vorticity_tensor(dv::SVector{2,T}, gx::SVector{2,T}) where {T<:AbstractFloat}
    return T(0.5) * (dv[1]*gx[2] - dv[2]*gx[1])
end

@inline function vorticity_tensor(dv::SVector{3,T}, gx::SVector{3,T}) where {T<:AbstractFloat}
    return SVector{3,T}(
        T(0.5) * (dv[1]*gx[2] - dv[2]*gx[1]),
        T(0.5) * (dv[1]*gx[3] - dv[3]*gx[1]),
        T(0.5) * (dv[2]*gx[3] - dv[3]*gx[2])
    )
end

"""
    diffusion_density(dx, rho_i, rho_j, c_i, c_j, h_i, h_j, gx, delta=0.1) -> psi

Molteni & Colagrossi (2009) δ-SPH density diffusion term for a single pair.

`dx = xi - xj` and `gx` is the kernel gradient vector. Returns the scalar
diffusion flux `psi`. Apply as:

    drhodt[i] += mass_j / rho_j * psi
    drhodt[j] -= mass_i / rho_i * psi
"""
@inline function diffusion_density(dx::SVector{ND,T}, rho_i::T, rho_j::T, c::T, h::T, gx::SVector{ND,T}, delta::T=0.1) where {ND,T<:AbstractFloat}
    rr  = dot(dx, dx)
    return T(2) * T(delta) * c * h * (rho_i - rho_j) * dot(dx, gx) / rr
end

@inline function diffusion_density(dx::SVector{ND,T}, rho_i::T, rho_j::T, c_i::T, c_j::T, h_i::T, h_j::T, gx::SVector{ND,T}, delta::T=0.1) where {ND,T<:AbstractFloat}
    rr = dot(dx, dx)
    c  = T(0.5) * (c_i + c_j)
    h  = T(0.5) * (h_i + h_j)
    return T(2) * T(delta) * c * h * (rho_i - rho_j) * dot(dx, gx) / rr
end

"""
    lennard_jones(dx, cutoff, c, p1) -> f

Repulsive Lennard-Jones boundary force coefficient (Monaghan & Kos 1999).

`dx = xi - xj` is the displacement SVector; `p1` is the repulsive exponent
(typically 4). Returns scalar `f` such that the boundary force on particle i
is `f * dx`.
"""
@inline function lennard_jones(dx::SVector{ND,T}, cutoff::T, c::T, p1::Int, p2::Int) where {ND,T<:AbstractFloat}
    rr = dot(dx, dx)
    r  = sqrt(rr)
    f  = (r < cutoff ? one(T) : zero(T)) * ((cutoff / r)^p1 - (cutoff / r)^p2) / rr
    return T(0.01) * c * c * f
end
