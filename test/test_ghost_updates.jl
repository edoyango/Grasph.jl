using Test
using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _make_test_system(; n=4, rho0=1000.0, c=10.0)
    ps = BasicParticleSystem("test", n, 2, 1.0, c)
    ps.rho .= rho0
    fill!(ps.v, zero(SVector{2,Float64}))
    fill!(ps.x, SVector(0.0, 0.0))
    ps
end

# ---------------------------------------------------------------------------
# Tests for update_ghost_kinematics!
# ---------------------------------------------------------------------------

@testset "update_ghost_kinematics! basic mirroring" begin
    rho0 = 1200.0
    ps = FluidParticleSystem("test", 2, 2, 1.0, 10.0)
    ps.rho .= rho0

    ps.x[1] = SVector(0.01, 0.01)
    ps.v[1] = SVector(1.0, 2.0)
    ps.rho[1] = rho0 + 10.0
    ps.x[2] = SVector(1.0, 1.0)  # outside cutoff
    ps.v[2] = SVector(0.0, 0.0)
    ps.rho[2] = rho0

    ghost = GhostParticleSystem(ps, GhostCopier(:rho))

    normal = SVector(1.0, 0.0) # Vertical wall at x=0, normal points right
    point  = SVector(0.0, 0.0)
    entry  = GhostEntry(ghost, 0.1, (normal, point))

    # 1. Generate ghosts (positions only)
    generate_ghosts!(entry)
    @test ghost.n == 1
    @test ghost.x[1] ≈ SVector(-0.01, 0.01)

    # 2. Update kinematics and copy physics
    update_ghost_kinematics!(entry)
    update_ghost!(ghost, 1)

    # Velocity should be reflected: vx -> -vx, vy -> vy
    @test ghost.v[1] ≈ SVector(-1.0, 2.0)
    # Density copied from source
    @test ghost.rho[1] == ps.rho[1]
end

# ---------------------------------------------------------------------------
# Tests for CauchyFluidPfn and FluidPfn Ghost Awareness
# ---------------------------------------------------------------------------

@testset "CauchyFluidPfn ghost interaction" begin
    h = 0.1
    ps = StressParticleSystem("test", 1, 2, 3, 1.0, 1000.0)
    ps.rho .= 1000.0
    fill!(ps.v, zero(SVector{2,Float64}))
    fill!(ps.dvdt, zero(SVector{2,Float64}))
    ps.drhodt .= 0.0

    ps.x[1] = SVector(0.02, 0.05)
    ps.stress[1] = SVector(-100.0, -100.0, 0.0) # Pressure -100

    ghost = GhostParticleSystem(ps, GhostCopier(:rho, :stress))

    normal = SVector(1.0, 0.0)
    point  = SVector(0.0, 0.0)
    entry  = GhostEntry(ghost, 0.1, (normal, point))

    generate_ghosts!(entry)
    update_ghost_kinematics!(entry)
    update_ghost!(ghost, 1)

    # Interaction: should use Cauchy interaction because :stress is present
    kernel = CubicSplineKernel(h; ndims=2)
    pfn = CauchyFluidPfn(0.1, 0.1, h)
    si  = SystemInteraction(kernel, pfn, ps, ghost)

    fill!(ps.dvdt, zero(SVector{2,Float64}))
    fill!(ps.drhodt, 0.0)

    create_grid!(si)
    sweep!(si)

    # Since it's a mirror, pressure is the same on both sides.
    # Symmetry should mean force is primarily in the normal direction.
    @test ps.dvdt[1][1] > 0 # Repelled from wall (positive x)
    @test ps.dvdt[1][2] ≈ 0 atol=1e-10
end

@testset "FluidPfn ghost interaction" begin
    h = 0.1
    ps = FluidParticleSystem("test", 1, 2, 1.0, 1000.0)
    ps.rho .= 1000.0
    fill!(ps.v, zero(SVector{2,Float64}))
    fill!(ps.dvdt, zero(SVector{2,Float64}))
    ps.drhodt .= 0.0
    ps.p .= 100.0

    ps.x[1] = SVector(0.02, 0.05)

    ghost = GhostParticleSystem(ps, GhostCopier(:rho, :p))

    normal = SVector(1.0, 0.0)
    point  = SVector(0.0, 0.0)
    entry  = GhostEntry(ghost, 0.1, (normal, point))

    generate_ghosts!(entry)
    update_ghost_kinematics!(entry)
    update_ghost!(ghost, 1)

    # Interaction: should use Pressure interaction because :p is present (no :stress)
    kernel = CubicSplineKernel(h; ndims=2)
    pfn = FluidPfn(0.1, 0.1, h)
    si  = SystemInteraction(kernel, pfn, ps, ghost)

    fill!(ps.dvdt, zero(SVector{2,Float64}))

    create_grid!(si)
    sweep!(si)

    @test ps.dvdt[1][1] > 0 # Repelled from wall
end

# ---------------------------------------------------------------------------
# Tests for LeapFrogTimeIntegrator stage separation
# ---------------------------------------------------------------------------

@testset "LeapFrogTimeIntegrator stage separation with ghosts" begin
    rho0 = 1000.0; h = 0.1
    ps = FluidParticleSystem("test", 1, 2, 1.0, 1000.0)
    ps.rho .= rho0
    fill!(ps.v, zero(SVector{2,Float64}))
    fill!(ps.dvdt, zero(SVector{2,Float64}))
    ps.drhodt .= 0.0
    ps.x[1] = SVector(0.02, 0.05)

    ghost = GhostParticleSystem(ps, GhostCopier(:rho))
    entry = GhostEntry(ghost, 0.1, (SVector(1.0, 0.0), SVector(0.0, 0.0)))

    # Custom Pfn to check ghost state during sweep
    struct MockPfn end
    function (f::MockPfn)(ps, ghost, i, j, dx, gx, w)
        # Verify kinematics are updated
        @test ghost.rho[j] == 1000.0
        # ps.v[1] is the source particle's velocity (ps.v[i] in this case)
        @test ghost.v[j][1] ≈ -ps.v[i][1] # reflected
    end

    kernel = CubicSplineKernel(h; ndims=2)
    si = SystemInteraction(kernel, MockPfn(), ps, ghost)

    integrator = LeapFrogTimeIntegrator([ps], [si]; ghosts=[entry])

    # Run 1 step
    time_integrate!(integrator, 1, 2, 2, 0.1, nothing)

    @test ghost.n == 1
end
