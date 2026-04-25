using Test
using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _stress_src(; n=4, nstress=4, rho0=1850.0, c=20.0)
    ps = StressParticleSystem("src", n, 2, nstress, rho0 * 0.0025, c)
    ps.rho .= rho0
    fill!(ps.v, zero(SVector{2,Float64}))
    for i in 1:n
        ps.x[i] = SVector(i * 0.02, 0.05)
    end
    ps
end

# Floor boundary at y=0 with inward normal (0,1).
_floor_entry(ghost; cutoff=0.15) =
    GhostEntry(ghost, cutoff, (SVector(0.0, 1.0), SVector(0.0, 0.0)))

# ---------------------------------------------------------------------------
# Construction: type-parameter inspection
# ---------------------------------------------------------------------------

@testset "GhostCopier construction — type parameters" begin

    @testset "bare symbol produces nothing mode" begin
        gc = GhostCopier(:stress)
        fields, modes = typeof(gc).parameters
        @test fields == (:stress,)
        @test modes  == (nothing,)
    end

    @testset "symbol => HouseholderReflect stores marker in type parameter" begin
        gc = GhostCopier(:stress => HouseholderReflect())
        fields, modes = typeof(gc).parameters
        @test fields   == (:stress,)
        @test modes[1] === HouseholderReflect()
    end

    @testset "multiple bare symbols — all modes are nothing" begin
        gc = GhostCopier(:rho, :stress)
        fields, modes = typeof(gc).parameters
        @test fields == (:rho, :stress)
        @test all(m -> m === nothing, modes)
    end

    @testset "mixed entries — field order and mode pairing preserved" begin
        gc = GhostCopier(:rho, :stress => HouseholderReflect())
        fields, modes = typeof(gc).parameters
        @test fields   == (:rho, :stress)
        @test modes[1] === nothing
        @test modes[2] === HouseholderReflect()
    end

    @testset "single bare symbol is a GhostCopier subtype" begin
        gc = GhostCopier(:stress)
        @test gc isa GhostCopier
    end

end

# ---------------------------------------------------------------------------
# Extras allocation
# ---------------------------------------------------------------------------

@testset "GhostCopier extras allocation" begin

    @testset "single copier field appears in extras" begin
        ps = _stress_src()
        g  = GhostParticleSystem(ps, GhostCopier(:stress))
        @test :stress in fieldnames(typeof(g.extras))
    end

    @testset "unmentioned fields are not in extras" begin
        ps = _stress_src()
        g  = GhostParticleSystem(ps, GhostCopier(:stress))
        @test :rho ∉ fieldnames(typeof(g.extras))
        @test :v   ∉ fieldnames(typeof(g.extras))
    end

    @testset "duplicate field across two copiers is allocated once" begin
        ps    = _stress_src()
        g     = GhostParticleSystem(ps, GhostCopier(:stress), GhostCopier(:stress))
        names = fieldnames(typeof(g.extras))
        @test length(names) == 1
        @test :stress in names
    end

    @testset "no copier — extras is empty" begin
        ps = _stress_src()
        g  = GhostParticleSystem(ps)
        @test isempty(fieldnames(typeof(g.extras)))
    end

    @testset "nothing updaters — extras is empty" begin
        ps = _stress_src()
        g  = GhostParticleSystem(ps, nothing, nothing)
        @test isempty(fieldnames(typeof(g.extras)))
    end

    @testset "extras arrays resize with ghost count after generate_ghosts!" begin
        ps = _stress_src(n=3)
        ps.x[1] = SVector(0.1, 0.04)   # qualifies
        ps.x[2] = SVector(0.2, 0.06)   # qualifies
        ps.x[3] = SVector(0.3, 0.50)   # outside cutoff
        g     = GhostParticleSystem(ps, GhostCopier(:stress))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        @test length(g.extras.stress) == g.n == 2
    end

end

# ---------------------------------------------------------------------------
# Straight copy (no signs)
# ---------------------------------------------------------------------------

@testset "GhostCopier straight copy" begin

    @testset "update_ghost! copies source values exactly" begin
        ps = _stress_src(n=3)
        ps.x[1] = SVector(0.1, 0.04)
        ps.x[2] = SVector(0.2, 0.06)
        ps.x[3] = SVector(0.3, 0.50)   # outside cutoff
        ps.stress[1] = SVector(10.0, 20.0, 30.0, 40.0)
        ps.stress[2] = SVector(-5.0, -5.0,  0.0,  2.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.n == 2
        for k in 1:g.n
            @test g.stress[k] == ps.stress[g.idx_original[k]]
        end
    end

    @testset "source mutation does not propagate until update_ghost! is re-called" begin
        ps = _stress_src(n=2)
        ps.x[1] = SVector(0.1, 0.04)
        ps.x[2] = SVector(0.2, 0.06)
        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0)
        ps.stress[2] = SVector(5.0, 6.0, 7.0, 8.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        snapshot = g.stress[1]
        ps.stress[g.idx_original[1]] = SVector(99.0, 99.0, 99.0, 99.0)
        @test g.stress[1] == snapshot   # not updated yet

        update_ghost!(g, 1)
        @test g.stress[1] == SVector(99.0, 99.0, 99.0, 99.0)
    end

    @testset "stress copied for all qualifying particles" begin
        n  = 5
        ps = _stress_src(n=n)
        for i in 1:n
            ps.x[i] = SVector(i * 0.02, 0.04)
            ps.stress[i] = SVector(Float64(i), Float64(i*2), 0.0, 0.0)
        end
        g     = GhostParticleSystem(ps, GhostCopier(:stress))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.n == n
        for k in 1:g.n
            @test g.stress[k] == ps.stress[g.idx_original[k]]
        end
    end

end

# ---------------------------------------------------------------------------
# Householder reflection
# ---------------------------------------------------------------------------

@testset "GhostCopier Householder reflection" begin

    @testset "4-comp Voigt on floor normal (0,1) flips only σ_xy" begin
        ps = _stress_src(n=2)
        ps.x[1] = SVector(0.1, 0.04)
        ps.x[2] = SVector(0.2, 0.06)
        ps.stress[1] = SVector(10.0, 20.0, 30.0,  5.0)
        ps.stress[2] = SVector(-5.0, -5.0,  0.0, -3.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        expected = SVector(1.0, 1.0, 1.0, -1.0)
        for k in 1:g.n
            src = ps.stress[g.idx_original[k]]
            @test g.stress[k] ≈ src .* expected
        end
    end

    @testset "free-slip wall pattern — normal stresses preserved, shear negated" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)
        ps.stress[1] = SVector(-100.0, -50.0, 10.0, 7.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.stress[1][1] ≈  ps.stress[1][1]
        @test g.stress[1][2] ≈  ps.stress[1][2]
        @test g.stress[1][3] ≈  ps.stress[1][3]
        @test g.stress[1][4] ≈ -ps.stress[1][4]
    end

    @testset "3-comp Voigt on floor normal (0,1) flips only σ_xy" begin
        ps = _stress_src(n=1, nstress=3)
        ps.x[1] = SVector(0.1, 0.04)
        ps.stress[1] = SVector(10.0, -5.0, 3.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.stress[1] ≈ SVector(10.0, -5.0, -3.0)
    end

    @testset "4-comp reflection with 45° normal swaps σ_xx ↔ σ_yy, leaves σ_zz and σ_xy" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.05, 0.05)
        ps.stress[1] = SVector(10.0, 20.0, 30.0, 7.0)

        g      = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        n̂      = SVector(1.0/√2, 1.0/√2)
        entry  = GhostEntry(g, 0.2, (n̂, SVector(0.0, 0.0)))
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.stress[1] ≈ SVector(20.0, 10.0, 30.0, 7.0)
    end

    @testset "6-comp Voigt 3D floor normal (0,0,1) flips σ_xz and σ_yz only" begin
        ps = StressParticleSystem("src3d", 1, 3, 6, 1.0, 20.0)
        ps.rho[1]    = 1.0
        ps.v[1]      = zero(SVector{3,Float64})
        ps.x[1]      = SVector(0.05, 0.05, 0.05)
        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        entry = GhostEntry(g, 0.2, (SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 0.0)))
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.stress[1] ≈ SVector(1.0, 2.0, 3.0, 4.0, -5.0, -6.0)
    end

    @testset "per-ghost normals come from idx_boundary (multi-boundary entry)" begin
        # Two boundaries: a floor (0,1) and a left wall (1,0). Place two particles
        # so that exactly one ghost is generated from each. The shear flip direction
        # should match whichever normal generated the ghost.
        ps = _stress_src(n=2)
        ps.x[1] = SVector(0.04, 0.20)   # qualifies for left wall (x=0)
        ps.x[2] = SVector(0.20, 0.04)   # qualifies for floor    (y=0)
        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0)
        ps.stress[2] = SVector(5.0, 6.0, 7.0, 8.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        entry = GhostEntry(g, 0.15,
                           (SVector(1.0, 0.0), SVector(0.0, 0.0)),   # left wall: normal (1,0) flips σ_xy
                           (SVector(0.0, 1.0), SVector(0.0, 0.0)))   # floor:     normal (0,1) flips σ_xy
        generate_ghosts!(entry)
        update_ghost!(g, 1)

        @test g.n == 2
        for k in 1:g.n
            src = ps.stress[g.idx_original[k]]
            @test g.stress[k] ≈ SVector(src[1], src[2], src[3], -src[4])
        end
    end

    @testset "reflection recomputed on each update_ghost! call" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)
        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress => HouseholderReflect()))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        update_ghost!(g, 1)
        @test g.stress[1] ≈ SVector(1.0, 2.0, 3.0, -4.0)

        ps.stress[1] = SVector(10.0, 20.0, 30.0, 5.0)
        update_ghost!(g, 1)
        @test g.stress[1] ≈ SVector(10.0, 20.0, 30.0, -5.0)
    end

end

# ---------------------------------------------------------------------------
# Multi-stage dispatch
# ---------------------------------------------------------------------------

@testset "GhostCopier multi-stage dispatch" begin

    @testset "stage-1 copier fires; nothing at stage 2 does not update" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)
        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress), nothing)
        entry = _floor_entry(g)
        generate_ghosts!(entry)

        fill!(g.extras.stress, zero(SVector{4,Float64}))
        update_ghost!(g, 1)
        @test g.stress[1] == ps.stress[1]

        # Mutate source; stage-2 nothing should not propagate the change
        ps.stress[1] = SVector(99.0, 99.0, 99.0, 99.0)
        update_ghost!(g, 2)
        @test g.stress[1] == SVector(1.0, 2.0, 3.0, 4.0)
    end

    @testset "nothing at stage 1 does not copy; stage-2 copier fires" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)
        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0)

        g     = GhostParticleSystem(ps, nothing, GhostCopier(:stress))
        entry = _floor_entry(g)
        generate_ghosts!(entry)

        fill!(g.extras.stress, zero(SVector{4,Float64}))
        update_ghost!(g, 1)
        @test g.stress[1] == zero(SVector{4,Float64})

        update_ghost!(g, 2)
        @test g.stress[1] == ps.stress[1]
    end

    @testset "stage index beyond updater count is a no-op" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)
        ps.stress[1] = SVector(5.0, 6.0, 7.0, 8.0)

        g     = GhostParticleSystem(ps, GhostCopier(:stress))
        entry = _floor_entry(g)
        generate_ghosts!(entry)
        fill!(g.extras.stress, zero(SVector{4,Float64}))

        update_ghost!(g, 2)   # only 1 updater registered
        @test g.stress[1] == zero(SVector{4,Float64})
    end

    @testset "stage-1 plain copy then stage-2 Householder reflect are independent" begin
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)

        g = GhostParticleSystem(ps,
            GhostCopier(:stress),                            # stage 1: plain
            GhostCopier(:stress => HouseholderReflect()),    # stage 2: reflect
        )
        entry = _floor_entry(g)
        generate_ghosts!(entry)

        ps.stress[1] = SVector(10.0, 20.0, 5.0, -3.0)

        update_ghost!(g, 1)
        @test g.stress[1] == ps.stress[1]

        update_ghost!(g, 2)
        # Floor normal (0,1): only σ_xy flips.
        @test g.stress[1] ≈ SVector(10.0, 20.0, 5.0, 3.0)
    end

    @testset "four-stage layout mirrors Trapdoor pattern" begin
        # Stage 1: reflect stress
        # Stage 2: nothing
        # Stage 3: reflect stress (after EP update)
        # Stage 4: nothing
        ps = _stress_src(n=1)
        ps.x[1] = SVector(0.1, 0.04)

        g = GhostParticleSystem(ps,
            GhostCopier(:stress => HouseholderReflect()),
            nothing,
            GhostCopier(:stress => HouseholderReflect()),
            nothing,
        )
        entry = _floor_entry(g)
        generate_ghosts!(entry)

        ps.stress[1] = SVector(1.0, 2.0, 3.0, 4.0)
        update_ghost!(g, 1)
        @test g.stress[1] ≈ SVector(1.0, 2.0, 3.0, -4.0)

        # Stage 2 + 4 are nothing — do not update
        ps.stress[1] = SVector(9.0, 9.0, 9.0, 9.0)
        prev = g.stress[1]
        update_ghost!(g, 2)
        @test g.stress[1] == prev
        update_ghost!(g, 4)
        @test g.stress[1] == prev

        # Stage 3 fires with new source values
        update_ghost!(g, 3)
        @test g.stress[1] ≈ SVector(9.0, 9.0, 9.0, -9.0)
    end

end

# ---------------------------------------------------------------------------
# GhostEntry delegation
# ---------------------------------------------------------------------------

@testset "update_ghost!(entry, stage) delegates to ghost" begin
    ps = _stress_src(n=1)
    ps.x[1] = SVector(0.1, 0.04)
    ps.stress[1] = SVector(3.0, 4.0, 5.0, 6.0)

    g     = GhostParticleSystem(ps, GhostCopier(:stress))
    entry = _floor_entry(g)
    generate_ghosts!(entry)
    fill!(g.extras.stress, zero(SVector{4,Float64}))

    update_ghost!(entry, 1)
    @test g.stress[1] == ps.stress[1]
end

@testset "update_ghost!(entry, stage) respects nothing at that stage" begin
    ps = _stress_src(n=1)
    ps.x[1] = SVector(0.1, 0.04)
    ps.stress[1] = SVector(3.0, 4.0, 5.0, 6.0)

    g     = GhostParticleSystem(ps, nothing, GhostCopier(:stress))
    entry = _floor_entry(g)
    generate_ghosts!(entry)
    fill!(g.extras.stress, zero(SVector{4,Float64}))

    update_ghost!(entry, 1)
    @test g.stress[1] == zero(SVector{4,Float64})

    update_ghost!(entry, 2)
    @test g.stress[1] == ps.stress[1]
end
