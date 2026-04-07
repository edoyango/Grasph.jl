using Test
using Grasph
using StaticArrays

@testset "StaticBoundarySystem construction and forwarding" begin
    ps = BasicParticleSystem("wall", 4, 2, 1.0, 10.0)
    ps.rho .= 1000.0
    fill!(ps.v, zero(SVector{2,Float64}))
    ps.x[1] = SVector(0.5, -0.25)

    sbs = StaticBoundarySystem(ps, 0.5)

    @test sbs.n == 4
    @test sbs.ndims == 2
    @test sbs.mass == 1.0
    @test sbs.c == 10.0
    @test sbs.lj_cutoff == 0.5
    @test sbs.x[1] == SVector(0.5, -0.25)
    @test sbs.rho[1] == 1000.0
    @test sbs.inner === ps
end

@testset "DynamicBoundarySystem construction and forwarding" begin
    ps = BasicParticleSystem("wall", 4, 2, 1.0, 10.0)
    ps.rho .= 1000.0
    fill!(ps.v, zero(SVector{2,Float64}))

    dbs = DynamicBoundarySystem(ps, SVector(0.0, 1.0), SVector(0.0, 0.0), 3.0)

    @test dbs.n == 4
    @test dbs.ndims == 2
    @test dbs.mass == 1.0
    @test dbs.boundary_normal == SVector(0.0, 1.0)
    @test dbs.boundary_point == SVector(0.0, 0.0)
    @test dbs.boundary_beta == 3.0
    @test dbs.x === ps.x
    @test dbs.inner === ps
end
