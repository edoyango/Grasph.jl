using Test
using Grasph

@testset "CubicSplineKernel" begin

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @testset "h and interaction_length" begin
        k = CubicSplineKernel(0.1; ndims=2)
        @test k.h ≈ 0.1
        @test k.interaction_length ≈ 0.2    # k=2 → 2h
    end

    @testset "norm_coeff 1D" begin
        k = CubicSplineKernel(1.0; ndims=1)
        @test k.norm_coeff ≈ 2 / 3
    end

    @testset "norm_coeff 2D" begin
        k = CubicSplineKernel(1.0; ndims=2)
        @test k.norm_coeff ≈ 10 / (7*π)
    end

    @testset "norm_coeff 3D" begin
        k = CubicSplineKernel(1.0; ndims=3)
        @test k.norm_coeff ≈ 1 / π
    end

    @testset "dtype propagation Float32" begin
        k = CubicSplineKernel(Float32(0.1); ndims=2)
        @test k.h              isa Float32
        @test k.norm_coeff     isa Float32
        @test k.interaction_length isa Float32
    end

    @testset "dtype propagation default Float64" begin
        k = CubicSplineKernel(0.1; ndims=2)
        @test k.h              isa Float64
        @test k.norm_coeff     isa Float64
        @test k.interaction_length isa Float64
    end

    # ------------------------------------------------------------------
    # kernel_w shape
    # ------------------------------------------------------------------

    @testset "kernel_w q=0" begin
        # unnormalised shape: 0.25*(2-0)^3 - (1-0)^3 = 1.0
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_w(k, 0.0) ≈ k.norm_coeff * 1.0
    end

    @testset "kernel_w q=1" begin
        # unnormalised shape: 0.25*(2-1)^3 - (1-1)^3 = 0.25
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_w(k, 1.0) ≈ k.norm_coeff * 0.25
    end

    @testset "kernel_w q=2 equals 0" begin
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_w(k, 2.0) ≈ 0.0
    end

    @testset "kernel_w compact support: q > 2 returns 0" begin
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_w(k, 2.5) ≈ 0.0
        @test kernel_w(k, 10.0) ≈ 0.0
    end

    @testset "kernel_w non-negative everywhere" begin
        k = CubicSplineKernel(1.0; ndims=3)
        for q in 0.0:0.1:3.0
            @test kernel_w(k, q) >= 0.0
        end
    end

    # ------------------------------------------------------------------
    # kernel_dw_dq shape
    # ------------------------------------------------------------------

    @testset "kernel_dw_dq q=0 equals 0 (peak → zero derivative)" begin
        # unnormalised: -3*(0.25*(2)^2 - (1)^2) = -3*(1 - 1) = 0
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_dw_dq(k, 0.0) ≈ 0.0
    end

    @testset "kernel_dw_dq q=2 equals 0 (compact support)" begin
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_dw_dq(k, 2.0) ≈ 0.0
    end

    @testset "kernel_dw_dq q > 2 equals 0" begin
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_dw_dq(k, 3.0) ≈ 0.0
    end

    @testset "kernel_dw_dq negative for q in (0, 2) (decreasing)" begin
        k = CubicSplineKernel(1.0; ndims=3)
        @test kernel_dw_dq(k, 0.5) < 0.0
        @test kernel_dw_dq(k, 1.0) < 0.0
        @test kernel_dw_dq(k, 1.5) < 0.0
    end

end
