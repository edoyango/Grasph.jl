using Test
using Grasph
using StaticArrays

@testset "ElastoPlasticStressUpdater" begin

    # Material parameters
    E        = 1.0e6
    nu       = 0.3
    phi      = 30.0 * π / 180.0
    psi      = 0.0
    cohesion = 1.0e6 # Increased to keep tests elastic when desired
    dt       = 0.001

    # Setup updater
    upd = ElastoPlasticStressUpdater(E, nu, phi, psi, cohesion, dt)

    @testset "Elastic increment (Plane Strain 4-elem)" begin
        ps = ElastoPlasticParticleSystem("test", 1, 2, 4, 1.0, 100.0)
        
        ps.strain_rate[1] = SVector(0.1, 0.0, 0.0, 0.0) 
        ps.vorticity[1]   = 0.0
        ps.stress[1]      = zero(SVector{4, Float64})
        
        upd(ps, 1)
        
        D0 = E / ((1 + nu) * (1 - 2*nu))
        deps_xx = 0.1 * dt
        
        expected_xx = D0 * (1 - nu) * deps_xx
        expected_yy = D0 * nu * deps_xx
        expected_zz = D0 * nu * deps_xx
        
        @test ps.stress[1][1] ≈ expected_xx
        @test ps.stress[1][2] ≈ expected_yy
        @test ps.stress[1][3] ≈ expected_zz
        @test ps.stress[1][4] == 0.0
        
        @test ps.strain[1] ≈ ps.strain_rate[1] * dt
        @test all(iszero, ps.strain_p[1])
    end

    @testset "Jaumann Rate Correction (Rotation)" begin
        ps = ElastoPlasticParticleSystem("test", 1, 2, 4, 1.0, 100.0)
        
        ps.stress[1]      = SVector(1.0e5, 0.0, 0.0, 0.0)
        ps.strain_rate[1] = zero(SVector{4, Float64})
        ps.vorticity[1]   = 10.0 
        
        upd(ps, 1)
        
        # Expected rotation:
        # Δσ_xy = (σ_xx - σ_yy) * ω_xy * dt
        # Δσ_xy = (1.0e5 - 0) * 10 * 0.001 = 1000.0
        @test ps.stress[1][4] ≈ 1000.0
        @test ps.stress[1][1] ≈ 1.0e5
    end

    @testset "Plastic Return Mapping (Drucker-Prager)" begin
        # Set cohesion very low to ensure yielding
        upd_plastic = ElastoPlasticStressUpdater(E, nu, phi, psi, 10.0, dt)
        ps = ElastoPlasticParticleSystem("test", 1, 2, 4, 1.0, 100.0)
        
        # Pure shear loading
        ps.strain_rate[1] = SVector(0.0, 0.0, 0.0, 100.0) 
        ps.stress[1]      = zero(SVector{4, Float64})
        
        upd_plastic(ps, 1)
        
        alpha_phi = 2 * sin(phi) / (sqrt(3) * (3 - sin(phi)))
        k_c       = 6 * 10.0 * cos(phi) / (sqrt(3) * (3 - sin(phi)))
        
        sig = ps.stress[1]
        I1 = sig[1] + sig[2] + sig[3]
        s_xx = sig[1] - I1/3
        s_yy = sig[2] - I1/3
        s_zz = sig[3] - I1/3
        s_xy = sig[4]
        
        J2 = 0.5 * (s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2)
        
        f_final = alpha_phi * I1 + sqrt(J2) - k_c
        @test f_final ≈ 0.0 atol=1e-7
        @test !iszero(ps.strain_p[1])
    end
end
