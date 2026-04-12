export FluidPfn, MultiPhaseFluidPfn, StrainRatePfn, StrainRateVorticityPfn, CauchyFluidPfn, XSPHPfn

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

"""
    FluidPfn{S, T}

Pairwise functor for weakly-compressible SPH fluid interaction.
"""
struct FluidPfn{S, T<:AbstractFloat}
    art_visc_alpha::T
    art_visc_beta::T
    h::T
end
function FluidPfn(alpha, beta, h; sigma=2)
    a, b, c = promote(float(alpha), float(beta), float(h))
    FluidPfn{sigma, typeof(a)}(a, b, c)
end

@inline @Base.propagate_inbounds function (f::FluidPfn{S,T})(ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,ND,T}
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

    dr = continuity_rate(dv, gx)
    ps.drhodt[i] += mass * dr * continuity_density_coeff(rho_i, rho_j, Val(S))
    ps.drhodt[j] += mass * dr * continuity_density_coeff(rho_j, rho_i, Val(S))
end

@inline @Base.propagate_inbounds function (f::FluidPfn{S,T})(ps_a::FluidParticleSystem{T,ND}, ps_b::FluidParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_i, mass_j = ps_a.mass, ps_b.mass
    dv           = vi - vj

    piv    = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    dh     = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    dv_tmp = (dh - piv) * gx

    ps_a.dvdt[i] += mass_j*dv_tmp
    ps_b.dvdt[j] -= mass_i*dv_tmp

    dr = continuity_rate(dv, gx)
    ps_a.drhodt[i] += mass_j * dr * continuity_density_coeff(rho_i, rho_j, Val(S))
    ps_b.drhodt[j] += mass_i * dr * continuity_density_coeff(rho_j, rho_i, Val(S))
end

# Coupled generic (one-sided, pressure-based) — covers ghosts
@inline @Base.propagate_inbounds function (f::FluidPfn{S,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    p_i, p_j     = ps_a.p[i], ps_b.p[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    dh  = pressure_force_coeff(p_i, p_j, rho_i, rho_j, Val(S))
    ps_a.dvdt[i] += mass_j * (dh - piv) * gx

    dr = continuity_rate(dv, gx)
    ps_a.drhodt[i] += mass_j * dr * continuity_density_coeff(rho_i, rho_j, Val(S))
end

# Coupled static boundary (LJ + artificial viscosity)
@inline @Base.propagate_inbounds function (f::FluidPfn{S,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::StaticBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    rf  = lennard_jones(dx, ps_b.lj_cutoff, ps_a.c, 12, 6)
    ps_a.dvdt[i] += -mass_j * piv * gx + rf * dx
end

# Coupled dynamic boundary (derives velocity, pressure-based)
@inline @Base.propagate_inbounds function (f::FluidPfn{S,T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {S,ND,T<:AbstractFloat}
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

    dr = continuity_rate(dv, gx)
    ps_a.drhodt[i] += mass * dr * continuity_density_coeff(rho_i, rho_j, Val(S))
end

"""
    CauchyFluidPfn{T}

Pairwise functor for SPH fluid self-interaction driven by a Cauchy stress tensor.
"""
struct CauchyFluidPfn{T<:AbstractFloat}
    art_visc_alpha::T
    art_visc_beta::T
    h::T
end
function CauchyFluidPfn(alpha, beta, h)
    a, b, c = promote(float(alpha), float(beta), float(h))
    CauchyFluidPfn{typeof(a)}(a, b, c)
end

@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{T})(ps::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
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

    psi = diffusion_density(dx, rho_i, rho_j, ps.c, ps.c, f.h, f.h, gx)
    dr = continuity_rate(dv, gx)
    ps.drhodt[i] += mass * (dr + psi/rho_j)
    ps.drhodt[j] += mass * (dr - psi/rho_i)
end

# Coupled generic (one-sided, stress-based) — covers ghosts
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    vi, vj             = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j       = ps_a.rho[i], ps_b.rho[j]
    stress_i, stress_j = ps_a.stress[i], ps_b.stress[j]
    mass_j             = ps_b.mass
    dv                 = vi - vj

    piv   = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_b.c)
    h_vec = cauchy_stress_force(stress_i, stress_j, rho_i, rho_j, gx)
    ps_a.dvdt[i] += mass_j * (h_vec - piv * gx)

    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx)
    dr  = continuity_rate(dv, gx)
    ps_a.drhodt[i] += mass_j * (dr + psi / rho_j)
end

# Coupled static boundary (LJ + artificial viscosity)
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::StaticBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
    vi, vj       = ps_a.v[i], ps_b.v[j]
    rho_i, rho_j = ps_a.rho[i], ps_b.rho[j]
    mass_j       = ps_b.mass
    dv           = vi - vj

    piv = artificial_viscosity(dx, dv, f.h, rho_i, rho_j, f.art_visc_alpha, f.art_visc_beta, ps_a.c, ps_a.c)
    rf  = lennard_jones(dx, ps_b.lj_cutoff, ps_a.c, 12, 6)
    ps_a.dvdt[i] += -mass_j * piv * gx + rf * dx
end

# Coupled dynamic boundary (derives velocity, stress-based)
@inline @Base.propagate_inbounds function (f::CauchyFluidPfn{T})(ps_a::AbstractParticleSystem{T,ND}, ps_b::DynamicBoundarySystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {ND,T<:AbstractFloat}
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

    psi = diffusion_density(dx, rho_i, rho_j, ps_a.c, ps_a.c, f.h, f.h, gx)
    dr  = continuity_rate(dv, gx)
    ps_a.drhodt[i] += mass * (dr + psi / rho_j)
end

"""
    XSPHPfn{T}
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