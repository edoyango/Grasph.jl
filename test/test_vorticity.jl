using Test
using Grasph
using StaticArrays
using LinearAlgebra

@testset "Vorticity Physics and Functors" begin

    @testset "vorticity_tensor 2D" begin
        # dv = (dvx, dvy), gx = (gx, gy)
        # vor = 0.5 * (dvx*gy - dvy*gx)
        
        # 1. Pure rotation: v = (y, -x) => grad v = [0 1; -1 0]
        # dx = (1, 0), dv = grad v * dx = (0, -1)
        # gx = (1, 0)
        dv = SVector(0.0, -1.0)
        gx = SVector(1.0, 0.0)
        # vor = 0.5 * (0*0 - (-1)*1) = 0.5
        @test vorticity_tensor(dv, gx) ≈ 0.5
        
        # 2. Opposite gx
        @test vorticity_tensor(dv, -gx) ≈ -0.5
        
        # 3. Swap components
        dv2 = SVector(1.0, 0.0)
        gx2 = SVector(0.0, 1.0)
        # vor = 0.5 * (1*1 - 0*0) = 0.5
        @test vorticity_tensor(dv2, gx2) ≈ 0.5
    end

    @testset "vorticity_tensor 3D" begin
        # returns SVector{3} (W12, W13, W23)
        # Wij = 0.5 * (dvi*gxj - dvj*gxi)
        
        dv = SVector(1.0, 2.0, 3.0)
        gx = SVector(0.1, 0.2, 0.3)
        
        vor = vorticity_tensor(dv, gx)
        @test vor isa SVector{3, Float64}
        
        @test vor[1] ≈ 0.5 * (dv[1]*gx[2] - dv[2]*gx[1]) # W12
        @test vor[2] ≈ 0.5 * (dv[1]*gx[3] - dv[3]*gx[1]) # W13
        @test vor[3] ≈ 0.5 * (dv[2]*gx[3] - dv[3]*gx[2]) # W23
    end

    @testset "StrainRateVorticityPfn integration" begin
        # Verify that the functor accumulates into both fields
        n = 2; ndims = 2; ns = 4; rho0 = 1000.0; c = 10.0; h = 0.1
        mass = rho0 * h * h
        
        ps = ElastoPlasticParticleSystem("test", n, ndims, ns, mass, c)
        
        # Set up a simple interaction
        ps.x[1] = SVector(0.0, 0.0)
        ps.x[2] = SVector(0.05, 0.0)
        ps.v[1] = SVector(0.0, 0.0)
        ps.v[2] = SVector(0.0, 1.0) # dv = (0, 1)
        ps.rho .= rho0
        
        fill!(ps.strain_rate, zero(SVector{ns, Float64}))
        fill!(ps.vorticity, 0.0)
        
        kernel = CubicSplineKernel(h; ndims=ndims)
        pfn = StrainRateVorticityPfn()
        si = SystemInteraction(kernel, pfn, ps)
        
        # We need to manually trigger the interaction or use sweep!
        # For simplicity in unit test, call the functor directly
        dx = ps.x[1] - ps.x[2]
        r = norm(dx)
        gx = kernel_dw_dq(kernel, r/h) * (dx / (r*h))
        w = kernel_w(kernel, r/h)
        
        pfn(ps, 1, 2, dx, gx, w)
        
        @test !all(iszero, ps.strain_rate[1])
        @test !all(iszero, ps.strain_rate[2])
        @test ps.vorticity[1] != 0.0
        @test ps.vorticity[2] != 0.0
        
        # Check consistency with manual calculation
        dv = ps.v[2] - ps.v[1]
        expected_vor = vorticity_tensor(dv, gx) * (mass / rho0)
        @test ps.vorticity[1] ≈ expected_vor
    end

end
