export FluidPfn, MultiPhaseFluidPfn, StrainRatePfn, StrainRateVorticityPfn, CauchyFluidPfn, XSPHPfn, InterpolateFieldFn

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

# Coupled dynamic boundary (derives velocity, pressure-based)
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

@inline @Base.propagate_inbounds function (::InterpolateFieldFn{fields, ACC_WSUM})(ps_a::AbstractParticleSystem{T,ND}, ps_b::VirtualParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    rho_i  = ps_a.rho[i]
    mass_i = ps_a.mass
    kernel_weight = w * (mass_i / rho_i)
    for fname in fields
        getproperty(ps_b, fname)[j] += kernel_weight * getproperty(ps_a, fname)[i]
    end
    ACC_WSUM && (ps_b.w_sum[j] += kernel_weight)
end

@inline @Base.propagate_inbounds function (::InterpolateFieldFn{fields, ACC_WSUM})(ps_a::VirtualParticleSystem{T,ND}, ps_b::AbstractParticleSystem{T,ND}, i::Int, j::Int, dx::SVector{ND,T}, gx::SVector{ND,T}, w::T) where {fields, ACC_WSUM, ND, T<:AbstractFloat}
    rho_j  = ps_b.rho[j]
    mass_j = ps_b.mass
    kernel_weight = w * (mass_j / rho_j)
    for fname in fields
        getproperty(ps_a, fname)[i] += kernel_weight * getproperty(ps_b, fname)[j]
    end
    ACC_WSUM && (ps_a.w_sum[i] += kernel_weight)
end