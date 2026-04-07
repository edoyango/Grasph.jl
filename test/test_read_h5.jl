using Test
using Grasph
using HDF5
using StaticArrays
using LinearAlgebra

@testset "read_h5!" begin
    @testset "write and read back BasicParticleSystem" begin
        mktempdir() do dir
            # Write a system
            ps1 = BasicParticleSystem("test1", 3, 2, 1.5, 2.0)
            ps1.x .= [SVector(0.0, 1.0), SVector(2.0, 3.0), SVector(4.0, 5.0)]
            ps1.v .= [SVector(0.1, 0.2), SVector(0.3, 0.4), SVector(0.5, 0.6)]
            ps1.rho .= [1.0, 2.0, 3.0]
            
            path = joinpath(dir, "out.h5")
            write_h5(ps1, path)
            
            # Read into a new system
            ps2 = BasicParticleSystem("test2", 3, 2, 1.0, 1.0)
            read_h5!(ps2, path)
            
            @test ps2.x ≈ ps1.x
            @test ps2.v ≈ ps1.v
            @test ps2.rho ≈ ps1.rho
            # Ensure metadata isn't overwritten because it's static
            @test ps2.mass == 1.0
            @test ps2.c == 1.0

            # Test the path constructor
            ps_new = BasicParticleSystem(path, "test1")
            @test ps_new.n == 3
            @test ps_new.ndims == 2
            @test ps_new.mass == 1.5
            @test ps_new.c == 2.0
            @test ps_new.x ≈ ps1.x
            @test ps_new.v ≈ ps1.v
            @test ps_new.rho ≈ ps1.rho
        end
    end
    
    @testset "write and read back FluidParticleSystem" begin
        mktempdir() do dir
            ps1 = FluidParticleSystem("fluid1", 2, 2, 0.5, 10.0; source_v=[0.0, -9.81])
            ps1.x .= [SVector(0.1, 0.2), SVector(0.3, 0.4)]
            ps1.v .= [SVector(0.5, 0.6), SVector(0.7, 0.8)]
            ps1.rho .= [1000.0, 1000.0]
            ps1.p .= [1.0, 2.0]
            
            path = joinpath(dir, "out.h5")
            write_h5(ps1, path)
            
            ps_new = FluidParticleSystem(path, "fluid1"; source_v=[0.0, -9.81])
            @test ps_new.n == 2
            @test ps_new.mass == 0.5
            @test ps_new.c == 10.0
            @test ps_new.x ≈ ps1.x
            @test ps_new.v ≈ ps1.v
            @test ps_new.rho ≈ ps1.rho
            @test ps_new.p ≈ ps1.p
            @test ps_new.source_v == SVector(0.0, -9.81)
        end
    end

    @testset "write and read back StressParticleSystem" begin
        mktempdir() do dir
            ps1 = StressParticleSystem("stress1", 2, 2, 3, 1.2, 5.0)
            ps1.x .= [SVector(0.1, 0.2), SVector(0.3, 0.4)]
            ps1.v .= [SVector(0.5, 0.6), SVector(0.7, 0.8)]
            ps1.rho .= [1800.0, 1800.0]
            ps1.stress .= [SVector(1.0, 2.0, 3.0), SVector(4.0, 5.0, 6.0)]
            ps1.strain_rate .= [SVector(0.1, 0.2, 0.3), SVector(0.4, 0.5, 0.6)]
            
            path = joinpath(dir, "out.h5")
            write_h5(ps1, path)
            
            ps_new = StressParticleSystem(path, "stress1")
            @test ps_new.n == 2
            @test eltype(ps_new.stress) == SVector{3, Float64}
            @test ps_new.mass == 1.2
            @test ps_new.c == 5.0
            @test ps_new.x ≈ ps1.x
            @test ps_new.v ≈ ps1.v
            @test ps_new.rho ≈ ps1.rho
            @test ps_new.stress ≈ ps1.stress
            @test ps_new.strain_rate ≈ ps1.strain_rate
        end
    end

    @testset "read_h5! catches mismatch errors" begin
        mktempdir() do dir
            ps1 = BasicParticleSystem("test1", 3, 2, 1.5, 2.0)
            path = joinpath(dir, "out.h5")
            write_h5(ps1, path)
            
            ps_wrong_n = BasicParticleSystem("wrong_n", 4, 2, 1.0, 1.0)
            @test_throws ArgumentError read_h5!(ps_wrong_n, path)
            
            ps_wrong_ndims = BasicParticleSystem("wrong_ndims", 3, 3, 1.0, 1.0)
            @test_throws ArgumentError read_h5!(ps_wrong_ndims, path)
        end
    end
end
