using Test
using Grasph
using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _make_granular(; n=8, rho0=1850.0, c=20.0, h=0.1)
    ps = StressParticleSystem("soil", n, 2, 3, rho0 * h * h, c)
    fill!(ps.v, zero(SVector{2,Float64}))
    ps.rho .= rho0
    ps
end

function _make_fluid(; n=4, rho0=1000.0, c=10.0, h=0.1)
    ps = FluidParticleSystem("fluid", n, 2, rho0 * h * h, c)
    fill!(ps.v, zero(SVector{2,Float64}))
    fill!(ps.x, SVector(0.5, 0.5))
    ps.rho .= rho0
    ps
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

 @testset "GhostParticleSystem constructor (stress source)" begin
    ps    = _make_granular(n=4)
    ghost = GhostParticleSystem(ps)
    @test ghost.ndims       == 2
    @test ghost.mass        == ps.mass
    @test ghost.n           == 4
    @test length(ghost.x)   == 4
    @test length(ghost.v)   == 4
end

 @testset "GhostParticleSystem constructor (fluid source)" begin
    ps    = _make_fluid(n=4)
    ghost = GhostParticleSystem(ps)
    @test ghost.ndims       == 2
    @test ghost.mass        == ps.mass
    @test ghost.n           == 4
    @test length(ghost.x)   == 4
    @test length(ghost.v)   == 4
end

# ---------------------------------------------------------------------------
# update_ghost! — live views
# ---------------------------------------------------------------------------

 @testset "ghost.stress is updated by update_ghost!" begin
    ps    = _make_granular(n=4)
    ps.x[1] = SVector(0.1, 0.05)   # within cutoff → ghost
    ps.x[2] = SVector(0.2, 0.05)   # within cutoff → ghost
    ps.x[3] = SVector(0.3, 0.5)    # outside cutoff → no ghost
    ps.x[4] = SVector(0.4, -0.1)   # exterior side → no ghost
    ps.stress[1] = SVector(1.0, 2.0, 3.0)
    ps.stress[2] = SVector(4.0, 5.0, 6.0)
    ps.rho .= 1850.0

    ghost = GhostParticleSystem(ps, GhostCopier(:rho, :stress))
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    generate_ghosts!(entry)
    update_ghost!(ghost, 1)

    @test ghost.n == 2
    for k in 1:ghost.n
        orig = ghost.idx_original[k]
        @test ghost.stress[k] == ps.stress[orig]
    end

    # Mutate source — ghost does NOT see it until update_ghost! is called
    ps.stress[1] = SVector(9.0, 9.0, 9.0)
    update_ghost!(ghost, 1)
    for k in 1:ghost.n
        orig = ghost.idx_original[k]
        @test ghost.stress[k] == ps.stress[orig]
    end
end

 @testset "ghost.p is updated by update_ghost!" begin
    ps = _make_fluid(n=4)
    ps.x .= [SVector(0.1*i, 0.05) for i in 1:4]
    ps.p .= [100.0, 200.0, 300.0, 400.0]
    ps.rho .= 1000.0

    ghost = GhostParticleSystem(ps, GhostCopier(:rho, :p))
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    generate_ghosts!(entry)
    update_ghost!(ghost, 1)

    for k in 1:ghost.n
        orig = ghost.idx_original[k]
        @test ghost.p[k] == ps.p[orig]
    end

    # Mutate source — ghost does NOT see it until update_ghost! is called
    ps.p .= [999.0, 888.0, 777.0, 666.0]
    update_ghost!(ghost, 1)
    for k in 1:ghost.n
        orig = ghost.idx_original[k]
        @test ghost.p[k] == ps.p[orig]
    end
end

# ---------------------------------------------------------------------------
# generate_ghosts! geometry
# ---------------------------------------------------------------------------

@testset "generate_ghosts! mirrors positions correctly" begin
    normal = SVector(0.0, 1.0)
    point  = SVector(0.0, 0.0)
    cutoff = 0.15

    ps = _make_granular(n=3)
    ps.x[1] = SVector(0.3, 0.05)    # interior, within cutoff → ghost
    ps.x[2] = SVector(0.3, 0.20)    # interior, outside cutoff → no ghost
    ps.x[3] = SVector(0.3, -0.05)   # exterior → no ghost
    ps.v[1] = SVector(1.0, -0.5)
    ps.rho .= 1850.0

    ghost = GhostParticleSystem(ps, GhostCopier(:rho))
    entry = GhostEntry(ghost, cutoff, (normal, point))
    generate_ghosts!(entry)
    update_ghost_kinematics!(entry)
    update_ghost!(ghost, 1)

    @test ghost.n                == 1
    @test length(ghost.x)        == 1
    @test ghost.x[1]             ≈ SVector(0.3, -0.05)
    @test ghost.v[1]             ≈ ps.v[1] - 2 * dot(ps.v[1], normal) * normal
    @test ghost.rho[1]           == ps.rho[1]
    @test ghost.idx_original[1]  == 1
end

@testset "generate_ghosts! with no qualifying particles" begin
    ps = _make_granular(n=3)
    ps.x[1] = SVector(0.0, -0.1)
    ps.x[2] = SVector(0.0, -0.2)
    ps.x[3] = SVector(0.0,  0.5)   # da=0.5 > cutoff=0.15
    ghost = GhostParticleSystem(ps)
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    generate_ghosts!(entry)
    @test ghost.n == 0
    @test length(ghost.x)  == 0
end

@testset "generate_ghosts! with all particles qualifying" begin
    n  = 5
    ps = _make_granular(n=n)
    for i in 1:n
        ps.x[i] = SVector(Float64(i)*0.01, 0.05)
    end
    ps.rho .= 1850.0
    ghost = GhostParticleSystem(ps)
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    generate_ghosts!(entry)
    @test ghost.n == n
    @test length(ghost.x)  == n
    for k in 1:n
        @test ghost.x[k][2] ≈ -0.05
    end
end

@testset "generate_ghosts! repeated calls resize correctly" begin
    ps    = _make_granular(n=4)
    ghost = GhostParticleSystem(ps)
    normal = SVector(0.0, 1.0)
    point  = SVector(0.0, 0.0)
    entry  = GhostEntry(ghost, 0.15, (normal, point))

    # First call: 2 particles inside
    ps.x[1] = SVector(0.0, 0.05); ps.x[2] = SVector(0.0, 0.08)
    ps.x[3] = SVector(0.0, -0.1); ps.x[4] = SVector(0.0, 0.5)
    ps.rho .= 1850.0
    generate_ghosts!(entry)
    @test ghost.n == 2

    # Second call: fewer qualify — arrays must shrink
    ps.x[2] = SVector(0.0, 0.5)
    generate_ghosts!(entry)
    @test ghost.n == 1
    @test length(ghost.x)  == 1

    # Third call: more qualify again — arrays must grow back
    ps.x[2] = SVector(0.0, 0.08)
    ps.x[3] = SVector(0.0, 0.06)
    generate_ghosts!(entry)
    @test ghost.n == 3
    @test length(ghost.x)  == 3
end

# ---------------------------------------------------------------------------
# Integration: sweep with GhostParticleSystem as system_b
# ---------------------------------------------------------------------------

@testset "StrainRatePfn accepts GhostParticleSystem" begin
    # Two real particles near y=0 with different velocities:
    #   p1: (0, 0.04) v=(0.5, 0) → Ghost G1: (0, -0.04) v=(0.5, 0)
    #   p2: (0, 0.03) v=(0.0, 0) → Ghost G2: (0, -0.03) v=(0.0, 0)
    # p1 ↔ G2: dist=0.07 < h=0.1, dv≠0 → non-zero sr[1]
    # p2 ↔ G1: dist=0.07 < h=0.1, dv≠0 → non-zero sr[2]
    rho0 = 1850.0; h = 0.1
    ps = _make_granular(n=2)
    ps.x[1] = SVector(0.0, 0.04)
    ps.x[2] = SVector(0.0, 0.03)
    ps.v[1] = SVector(0.5, 0.0)
    ps.v[2] = SVector(0.0, 0.0)
    ps.rho .= rho0

    ghost = GhostParticleSystem(ps, GhostCopier(:rho))
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    generate_ghosts!(entry)
    update_ghost_kinematics!(entry)
    update_ghost!(ghost, 1)
    @test ghost.n == 2

    k   = CubicSplineKernel(h; ndims=2)
    pfn = StrainRatePfn()
    si  = SystemInteraction(k, pfn, ps, ghost)

    fill!(ps.strain_rate, zero(SVector{3,Float64}))
    create_grid!(si)
    sweep!(si)

    @test !iszero(ps.strain_rate[1])
    @test !iszero(ps.strain_rate[2])
end

# ---------------------------------------------------------------------------
# Integration: GhostEntry + LeapFrogTimeIntegrator mirrors stress each stage
# ---------------------------------------------------------------------------

@testset "GhostEntry construction — single boundary" begin
    ps    = _make_granular(n=4)
    ps.x .= [SVector(0.1*i, 0.05) for i in 1:4]
    ps.rho .= 1850.0
    ghost = GhostParticleSystem(ps)
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    @test length(entry.boundaries) == 1
    @test entry.boundaries[1].normal == SVector(0.0, 1.0)
    @test entry.boundaries[1].point  == SVector(0.0, 0.0)
    @test entry.cutoff  ≈ 0.15
    @test entry.ghost   === ghost
end

@testset "GhostEntry construction — multiple boundaries" begin
    ps    = _make_granular(n=4)
    ps.x .= [SVector(0.1*i, 0.05) for i in 1:4]
    ps.rho .= 1850.0
    ghost = GhostParticleSystem(ps)
    entry = GhostEntry(ghost, 0.15,
                       (SVector(1.0, 0.0), SVector(0.0, 0.0)),
                       (SVector(0.0, 1.0), SVector(0.0, 0.0)))
    @test length(entry.boundaries) == 2
    @test entry.boundaries[1].normal == SVector(1.0, 0.0)
    @test entry.boundaries[2].normal == SVector(0.0, 1.0)
    @test entry.cutoff ≈ 0.15
end

@testset "GhostEntry generate_ghosts! sets idx_boundary correctly" begin
    ps = _make_granular(n=4)
    # Particles near y=0 (boundary 1) and near x=0 (boundary 2)
    ps.x[1] = SVector(0.05, 0.5)   # near x=0 only
    ps.x[2] = SVector(0.5,  0.05)  # near y=0 only
    ps.x[3] = SVector(0.5,  0.5)   # neither
    ps.x[4] = SVector(0.05, 0.05)  # near both
    ps.rho .= 1850.0
    ghost = GhostParticleSystem(ps)
    entry = GhostEntry(ghost, 0.15,
                       (SVector(1.0, 0.0), SVector(0.0, 0.0)),  # boundary 1: x=0 wall
                       (SVector(0.0, 1.0), SVector(0.0, 0.0)))  # boundary 2: y=0 wall
    generate_ghosts!(entry)
    # All ghosts from boundary 1 have idx_boundary == 1, boundary 2 → 2
    idx_bnd = ghost.idx_boundary
    @test all(b -> b in (1, 2), idx_bnd)
    # Ghosts generated from x-wall have reflected x-coordinate (negative x)
    for k in 1:ghost.n
        if idx_bnd[k] == 1
            @test ghost.x[k][1] < 0
        else
            @test ghost.x[k][2] < 0
        end
    end
end

@testset "integrator calls generate_ghosts! and update_ghost! automatically" begin
    rho0 = 1850.0; h = 0.1
    ps = _make_granular(n=2)
    ps.x[1] = SVector(0.0, 0.04)
    ps.x[2] = SVector(0.0, 0.03)
    ps.v[1] = SVector(0.5, 0.0)
    ps.v[2] = SVector(0.0, 0.0)
    ps.rho .= rho0
    ps.stress[1] = SVector(1.0, 2.0, 3.0)
    ps.stress[2] = SVector(4.0, 5.0, 6.0)

    # Set up ghost — stage 1 copier keeps rho and stress current before each sweep
    ghost = GhostParticleSystem(ps, GhostCopier(:rho, :stress))
    entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))

    k = CubicSplineKernel(h; ndims=2)
    si = SystemInteraction(k, StrainRatePfn(), ps, ghost)

    integrator = LeapFrogTimeIntegrator([ps], [si]; ghosts=[entry])

    # Run 1 step — integrator should regenerate and mirror ghosts automatically
    time_integrate!(integrator, 1, 2, 2, 0.1, nothing)

    # After the step, ghost should have been populated and stress mirrored (via live view)
    @test ghost.n == 2
    for k in 1:ghost.n
        orig = ghost.idx_original[k]
        @test ghost.stress[k] == ps.stress[orig]
    end
end
