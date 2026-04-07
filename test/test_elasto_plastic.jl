using Test
using Grasph
using StaticArrays

@testset "ElastoPlasticParticleSystem" begin

    @testset "Construction and field shapes (2D)" begin
        n = 10
        ndims = 2
        ns = 3
        mass = 1.5
        c = 20.0
        ps = ElastoPlasticParticleSystem("test2d", n, ndims, ns, mass, c)

        @test ps.name == "test2d"
        @test ps.n == n
        @test ps.ndims == ndims
        @test ps.mass == mass
        @test ps.c == c

        @test length(ps.stress) == n
        @test eltype(ps.stress) == SVector{ns, Float64}
        
        @test length(ps.strain_rate) == n
        @test eltype(ps.strain_rate) == SVector{ns, Float64}

        # In 2D, vorticity is a scalar (T)
        @test length(ps.vorticity) == n
        @test eltype(ps.vorticity) == Float64

        @test length(ps.strain) == n
        @test eltype(ps.strain) == SVector{ns, Float64}

        @test length(ps.strain_p) == n
        @test eltype(ps.strain_p) == SVector{ns, Float64}
    end

    @testset "Construction and field shapes (3D)" begin
        n = 5
        ndims = 3
        ns = 6
        ps = ElastoPlasticParticleSystem("test3d", n, ndims, ns, 1.0, 1.0)

        # In 3D, vorticity is an axial vector (SVector{3, T})
        @test length(ps.vorticity) == n
        @test eltype(ps.vorticity) == SVector{3, Float64}
        
        @test eltype(ps.stress) == SVector{6, Float64}
    end

    @testset "ns validation" begin
        @test_throws ArgumentError ElastoPlasticParticleSystem("err", 5, 2, 2, 1.0, 1.0)
        @test_throws ArgumentError ElastoPlasticParticleSystem("err", 5, 2, 5, 1.0, 1.0)
        @test ElastoPlasticParticleSystem("ok", 5, 2, 3, 1.0, 1.0) isa ElastoPlasticParticleSystem
        @test ElastoPlasticParticleSystem("ok", 5, 2, 4, 1.0, 1.0) isa ElastoPlasticParticleSystem
        @test ElastoPlasticParticleSystem("ok", 5, 3, 6, 1.0, 1.0) isa ElastoPlasticParticleSystem
    end

    @testset "dtype propagation" begin
        ps32 = ElastoPlasticParticleSystem("f32", 5, 2, 3, 1.0, 1.0; dtype=Float32)
        @test eltype(ps32.rho)         === Float32
        @test eltype(ps32.stress)      === SVector{3, Float32}
        @test eltype(ps32.vorticity)   === Float32
        @test eltype(ps32.strain)      === SVector{3, Float32}
        @test eltype(ps32.strain_p)    === SVector{3, Float32}
        @test ps32.mass isa Float32
    end

    @testset "vorticity dtype propagation (3D)" begin
        ps32 = ElastoPlasticParticleSystem("f32_3d", 5, 3, 6, 1.0, 1.0; dtype=Float32)
        @test eltype(ps32.vorticity) === SVector{3, Float32}
    end

    @testset "Initial values are zero" begin
        ps = ElastoPlasticParticleSystem("test", 5, 2, 3, 1.0, 1.0)
        @test all(iszero, ps.stress)
        @test all(iszero, ps.strain_rate)
        @test all(iszero, ps.vorticity)
        @test all(iszero, ps.strain)
        @test all(iszero, ps.strain_p)
    end

end
