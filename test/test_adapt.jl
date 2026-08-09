using Test
using Grasph
using StaticArrays
using Adapt
using CUDA

# ---------------------------------------------------------------------------
# Phase B1 — array-type-generic particle systems.
#
# These tests validate two things:
#   1. Default construction is still `Vector`-backed (zero behaviour change
#      for every existing script, which never passes an array-type override).
#   2. `Adapt.adapt(Array, ps)` round-trips every field exactly, and the
#      rebuilt struct is still fully functional (not just field-equal) —
#      exercised by running it through a real `sort_particles!`/`create_grid!`/
#      `sweep!`/`time_integrate!` path, since none of this repo's CI has
#      access to a CUDA device to test the `CuArray` path directly.
# ---------------------------------------------------------------------------

@testset "Adapt.jl support (Phase B1)" begin

    @testset "Default construction stays Vector-backed" begin
        basic = BasicParticleSystem("basic", 4, 2, 1.0, 2.0)
        @test basic.x isa Vector{SVector{2,Float64}}
        @test basic.rho isa Vector{Float64}
        @test getfield(basic, :id) isa Vector{Int}

        fluid = FluidParticleSystem("fluid", 4, 2, 1.0, 2.0)
        @test fluid.x isa Vector{SVector{2,Float64}}
        @test fluid.p isa Vector{Float64}

        stress = StressParticleSystem("stress", 4, 2, 3, 1.0, 2.0)
        @test stress.stress isa Vector{SVector{3,Float64}}

        ep = ElastoPlasticParticleSystem("ep", 4, 2, 3, 1.0, 2.0)
        @test ep.vorticity isa Vector{Float64}          # ndims==2 -> scalar vorticity
        ep3 = ElastoPlasticParticleSystem("ep3", 4, 3, 6, 1.0, 2.0)
        @test ep3.vorticity isa Vector{SVector{3,Float64}}

        vps = VirtualParticleSystem(basic, "virt", 4, 2, 1.0, 2.0; zero_fields=(:w_sum,))
        @test getfield(vps, :w_sum) isa Vector{Float64}

        sb = StaticBoundarySystem(basic, 0.5)
        @test sb.x isa Vector{SVector{2,Float64}}
        db = DynamicBoundarySystem(basic, [1.0, 0.0], [0.0, 0.0], 1.0)
        @test db.x isa Vector{SVector{2,Float64}}
    end

    @testset "Adapt round-trip preserves fields exactly — BasicParticleSystem" begin
        ps = BasicParticleSystem("basic", 6, 2, 1.5, 3.0; source_v=[0.0, -9.81])
        ps.x .= [SVector(Float64(i), Float64(2i)) for i in 1:6]
        ps.v .= [SVector(0.1 * i, -0.1 * i) for i in 1:6]
        ps.rho .= 1000.0 .+ (1:6)
        add_print_field!(ps, :v)

        ps2 = adapt(Array, ps)
        @test typeof(ps2) == typeof(ps)
        @test ps2.name == ps.name && ps2.n == ps.n && ps2.ndims == ps.ndims
        @test ps2.mass == ps.mass && ps2.c == ps.c
        @test ps2.source_v == ps.source_v && ps2.source_rho == ps.source_rho
        @test ps2.x == ps.x && ps2.v == ps.v && ps2.rho == ps.rho
        @test ps2.dvdt == ps.dvdt && ps2.drhodt == ps.drhodt
        @test getfield(ps2, :id) == getfield(ps, :id)
        @test getfield(ps2, :_print_fields) == getfield(ps, :_print_fields)
    end

    @testset "Adapt round-trip preserves fields exactly — FluidParticleSystem" begin
        ps = FluidParticleSystem("fluid", 6, 2, 1.5, 3.0; state_updater=TaitEOSUpdater(1000.0))
        ps.x .= [SVector(Float64(i), 0.0) for i in 1:6]
        ps.rho .= 1000.0
        update_state!(ps, 1)

        ps2 = adapt(Array, ps)
        @test typeof(ps2) == typeof(ps)
        @test ps2.p == ps.p
        @test length(getfield(ps2, :state_updater)) == length(getfield(ps, :state_updater))
    end

    @testset "Adapt round-trip preserves fields exactly — StressParticleSystem" begin
        ps = StressParticleSystem("stress", 5, 2, 3, 1.0, 2.0)
        ps.stress .= [SVector(Float64(i), 0.0, 0.0) for i in 1:5]
        ps2 = adapt(Array, ps)
        @test typeof(ps2) == typeof(ps)
        @test ps2.stress == ps.stress && ps2.strain_rate == ps.strain_rate
    end

    @testset "Adapt round-trip preserves fields exactly — ElastoPlasticParticleSystem (2D + 3D)" begin
        ps2d = ElastoPlasticParticleSystem("ep2d", 5, 2, 3, 1.0, 2.0)
        ps2d.vorticity .= collect(1.0:5.0)
        r2d = adapt(Array, ps2d)
        @test typeof(r2d) == typeof(ps2d)
        @test r2d.vorticity == ps2d.vorticity
        @test r2d.strain == ps2d.strain && r2d.strain_p == ps2d.strain_p

        ps3d = ElastoPlasticParticleSystem("ep3d", 5, 3, 6, 1.0, 2.0)
        ps3d.vorticity .= [SVector(Float64(i), 0.0, 0.0) for i in 1:5]
        r3d = adapt(Array, ps3d)
        @test typeof(r3d) == typeof(ps3d)
        @test r3d.vorticity == ps3d.vorticity
    end

    @testset "Adapt round-trip preserves fields exactly — VirtualParticleSystem" begin
        basic = BasicParticleSystem("basic", 5, 2, 1.0, 2.0)
        basic.x .= [SVector(Float64(i), 0.0) for i in 1:5]
        vps = VirtualParticleSystem(basic, "virt", 5, 2, 1.0, 2.0; zero_fields=(:w_sum,))
        getfield(vps, :w_sum) .= collect(1.0:5.0)

        vps2 = adapt(Array, vps)
        @test typeof(vps2) == typeof(vps)
        @test getfield(vps2, :w_sum) == getfield(vps, :w_sum)
        @test getfield(vps2, :source).x == getfield(vps, :source).x

        # zero_fields (ZF type param) must survive adapt — auto_zero_virtual!
        # should still clear w_sum on the adapted copy.
        Grasph.auto_zero_virtual!(vps2)
        @test all(iszero, getfield(vps2, :w_sum))
    end

    @testset "Adapt round-trip preserves fields exactly — ProbeParticleSystem (item 9)" begin
        positions = [SVector(Float64(i) * 0.1, 0.0) for i in 1:5]
        probe = ProbeParticleSystem("probe", positions;
            extras=(cnt=zeros(Int, 5), stress=[zero(SVector{3,Float64}) for _ in 1:5]))
        getfield(probe, :w_sum) .= collect(1.0:5.0)
        probe.cnt .= [1, 2, 3, 4, 5]

        probe2 = adapt(Array, probe)
        @test typeof(probe2) == typeof(probe)
        @test probe2.x == probe.x
        @test getfield(probe2, :id) == getfield(probe, :id)
        @test getfield(probe2, :w_sum) == getfield(probe, :w_sum)
        @test probe2.cnt == probe.cnt
        @test probe2.stress == probe.stress
        @test getfield(probe2, :prescribed_v) == getfield(probe, :prescribed_v)
        @test getfield(probe2, :mirror_target) === nothing

        # mirror_target (a real particle system) round-trips too. adapt(Array, ·)
        # on an already-Array-backed object is a no-op at the array level (see
        # the GhostParticleSystem testset below), so this does NOT assert
        # independence from the original — only a real CuArray round-trip
        # (below, CUDA.functional()-guarded) forces an actual copy.
        src = BasicParticleSystem("src", 5, 2, 1.0, 2.0)
        src.x .= positions
        probe_m = ProbeParticleSystem("probe_m", src; extras=(cnt=zeros(Int, 5),))
        probe_m2 = adapt(Array, probe_m)
        @test typeof(probe_m2) == typeof(probe_m)
        @test getfield(probe_m2, :mirror_target).x == getfield(probe_m, :mirror_target).x
    end

    @testset "Adapt round-trip preserves fields exactly — GhostParticleSystem/GhostEntry (item 7)" begin
        fluid = FluidParticleSystem("fluid", 6, 2, 1.0, 10.0; source_v=zeros(2))
        fluid.x .= [SVector(0.1 * i, 0.05) for i in 1:6]
        fill!(fluid.v, zero(SVector{2,Float64}))   # was left undef, making the ghost.v equality check below flaky (occasional NaN garbage, which never equals itself)
        fluid.rho .= 1000.0
        fluid.p   .= collect(1.0:6.0)

        ghost = GhostParticleSystem(fluid, GhostCopier(:p); name="ghost[fluid]")
        entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
        generate_ghosts!(entry)
        update_ghost_kinematics!(entry)
        update_ghost!(entry, 1)
        n = ghost.n
        @test n > 0   # sanity: the fixture actually produced ghosts

        entry2 = adapt(Array, entry)
        ghost2 = entry2.ghost
        @test typeof(ghost2) == typeof(ghost)
        @test ghost2.n == ghost.n
        @test getfield(ghost2, :x)[1:n]            == getfield(ghost, :x)[1:n]
        @test getfield(ghost2, :v)[1:n]            == getfield(ghost, :v)[1:n]
        @test getfield(ghost2, :rho)[1:n]          == getfield(ghost, :rho)[1:n]
        @test getfield(ghost2, :idx_original)[1:n] == getfield(ghost, :idx_original)[1:n]
        @test getfield(ghost2, :idx_boundary)[1:n] == getfield(ghost, :idx_boundary)[1:n]
        @test getfield(ghost2, :normals)[1:n]      == getfield(ghost, :normals)[1:n]
        @test ghost2.p[1:n] == ghost.p[1:n]
        @test getfield(ghost2, :source).x == getfield(ghost, :source).x

        # count is a Ref, not shared — mutating the adapted copy's logical
        # count must not alias back into the original.
        getfield(ghost2, :count)[] = -1
        @test ghost.n == n

        # _flags survives with the same length (NB * source.n, fixed for the run).
        @test length(getfield(entry2, :_flags)) == length(getfield(entry, :_flags))

        # Re-running generate_ghosts! on the adapted (still-Array-backed) copy
        # must still work — adapt(Array, ·) on a CPU-already system is meant
        # to be a functional no-op, not just a field-preserving one.
        fluid2 = getfield(ghost2, :source)
        fluid2.x .= [SVector(0.1 * i, 0.03) for i in 1:6]
        generate_ghosts!(entry2)
        @test ghost2.n > 0
    end

    @testset "Adapt round-trip preserves fields exactly — boundary wrappers" begin
        basic = BasicParticleSystem("basic", 4, 2, 1.0, 2.0)
        basic.x .= [SVector(Float64(i), 0.0) for i in 1:4]

        sb = StaticBoundarySystem(basic, 0.5)
        sb2 = adapt(Array, sb)
        @test typeof(sb2) == typeof(sb)
        @test sb2.x == sb.x && sb2.lj_cutoff == sb.lj_cutoff

        db = DynamicBoundarySystem(basic, [1.0, 0.0], [0.0, 0.0], 1.0)
        db2 = adapt(Array, db)
        @test typeof(db2) == typeof(db)
        @test db2.boundary_normal == db.boundary_normal
        @test db2.boundary_point == db.boundary_point
        @test db2.boundary_beta == db.boundary_beta
    end

    @testset "Adapted system is fully functional (sort + grid + sweep + time_integrate!)" begin
        # Small fluid-over-static-boundary setup, mirroring dambreak.jl's shape.
        n_fluid = 25
        h = 0.6
        fluid = FluidParticleSystem("fluid", n_fluid, 2, 1.0, 20.0;
                                     source_v=[0.0, -9.81], state_updater=TaitEOSUpdater(1000.0))
        let k = 1
            for i in 0:4, j in 0:4
                fluid.x[k] = SVector(0.5 * i, 0.5 * j)
                k += 1
            end
        end
        fill!(fluid.v, zero(SVector{2,Float64}))
        fluid.rho .= 1000.0
        update_state!(fluid, 1)

        boundary = BasicParticleSystem("boundary", 12, 2, 1.0, 20.0)
        let k = 1
            for i in 0:11
                boundary.x[k] = SVector(0.5 * i, -0.5)
                k += 1
            end
        end
        boundary.rho .= 1000.0
        fill!(boundary.v, zero(SVector{2,Float64}))

        # Round-trip both through Adapt before wiring the integrator.
        fluid_a    = adapt(Array, fluid)
        boundary_a = adapt(Array, boundary)

        kernel = CubicSplineKernel(h; ndims=2)
        static_boundary = StaticBoundarySystem(boundary_a, 0.5)
        fluid_int = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid_a)
        fluid_boundary_int = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid_a, static_boundary)

        integrator = LeapFrogTimeIntegrator([fluid_a, boundary_a], [fluid_int, fluid_boundary_int])
        time_integrate!(integrator, 10, 1000, 1000, 0.1, nothing; print_timer=false)

        @test all(isfinite, reinterpret(Float64, fluid_a.x))
        @test all(isfinite, fluid_a.rho)
        @test all(r -> r > 0, fluid_a.rho)
    end

    # -----------------------------------------------------------------------
    # Real CuArray round-trips. Everything above only exercises adapt(Array,
    # ps), the trivial identity path, because the machine Phase B1 was built
    # on had no GPU. Now that one exists (see docs/gpu-migration-plan.md),
    # round-trip through an actual device array — CUDA.functional()-guarded
    # so this stays skippable on CPU-only machines.
    # -----------------------------------------------------------------------

    if CUDA.functional()

        @testset "Adapt round-trip through CuArray" begin
            basic = BasicParticleSystem("basic", 6, 2, 1.5, 3.0; source_v=[0.0, -9.81])
            basic.x .= [SVector(Float64(i), Float64(2i)) for i in 1:6]
            basic.v .= [SVector(0.1 * i, -0.1 * i) for i in 1:6]
            basic.rho .= 1000.0 .+ (1:6)
            add_print_field!(basic, :v)

            fluid = FluidParticleSystem("fluid", 6, 2, 1.5, 3.0; state_updater=TaitEOSUpdater(1000.0))
            fluid.x .= [SVector(Float64(i), 0.0) for i in 1:6]
            fluid.rho .= 1000.0
            update_state!(fluid, 1)

            stress = StressParticleSystem("stress", 5, 2, 3, 1.0, 2.0)
            stress.stress .= [SVector(Float64(i), 0.0, 0.0) for i in 1:5]

            ep2d = ElastoPlasticParticleSystem("ep2d", 5, 2, 3, 1.0, 2.0)
            ep2d.vorticity .= collect(1.0:5.0)
            ep3d = ElastoPlasticParticleSystem("ep3d", 5, 3, 6, 1.0, 2.0)
            ep3d.vorticity .= [SVector(Float64(i), 0.0, 0.0) for i in 1:5]

            vps_source = BasicParticleSystem("basic2", 5, 2, 1.0, 2.0)
            vps = VirtualParticleSystem(vps_source, "virt", 5, 2, 1.0, 2.0; zero_fields=(:w_sum,))
            getfield(vps, :w_sum) .= collect(1.0:5.0)

            probe = ProbeParticleSystem("probe",
                [SVector(Float64(i) * 0.1, 0.0) for i in 1:5];
                extras=(cnt=zeros(Int, 5),))
            getfield(probe, :w_sum) .= collect(1.0:5.0)

            sb = StaticBoundarySystem(basic, 0.5)
            db = DynamicBoundarySystem(basic, [1.0, 0.0], [0.0, 0.0], 1.0)

            ghost_src = FluidParticleSystem("ghost_src", 6, 2, 1.0, 10.0; source_v=zeros(2))
            ghost_src.x .= [SVector(0.1 * i, 0.05) for i in 1:6]
            ghost_src.rho .= 1000.0
            ghost_src.p   .= collect(1.0:6.0)
            ghost = GhostParticleSystem(ghost_src, GhostCopier(:p); name="ghost[ghost_src]")
            ghost_entry = GhostEntry(ghost, 0.15, (SVector(0.0, 1.0), SVector(0.0, 0.0)))
            generate_ghosts!(ghost_entry)   # populate before adapting, like a real driver script would

            for ps in (basic, fluid, stress, ep2d, ep3d, vps, probe, sb, db, ghost_entry)
                d = adapt(CuArray, ps)
                r = adapt(Array, d)
                @test typeof(r) == typeof(ps)
            end

            d_probe = adapt(CuArray, probe)
            @test getfield(d_probe, :x) isa CuArray{SVector{2,Float64}}
            @test getfield(d_probe, :id) isa CuArray{Int}
            @test getfield(d_probe, :w_sum) isa CuArray{Float64}
            @test d_probe.cnt isa CuArray{Int}
            r_probe = adapt(Array, d_probe)
            @test r_probe.x == probe.x
            @test getfield(r_probe, :w_sum) == getfield(probe, :w_sum)
            @test r_probe.cnt == probe.cnt

            # Field-exact round-trips and device-storage-type assertions for
            # the three systems reachable in the dambreak.jl vertical slice.
            for (ps, arrtype) in ((basic, SVector{2,Float64}), (fluid, SVector{2,Float64}))
                d = adapt(CuArray, ps)
                @test d.x isa CuArray{arrtype}
                @test getfield(d, :id) isa CuArray{Int}
                r = adapt(Array, d)
                @test r.x == ps.x && r.v == ps.v && r.rho == ps.rho
                @test getfield(r, :id) == getfield(ps, :id)
                @test r.mass == ps.mass && r.c == ps.c
                # _print_fields is intentionally passed through unadapted
                # (host-only bookkeeping) — the SAME vector, not a copy.
                @test getfield(d, :_print_fields) === getfield(ps, :_print_fields)
            end

            d_sb = adapt(CuArray, sb)
            @test getfield(d_sb, :inner).x isa CuArray{SVector{2,Float64}}
            r_sb = adapt(Array, d_sb)
            @test r_sb.x == sb.x && r_sb.lj_cutoff == sb.lj_cutoff

            # GhostParticleSystem/GhostEntry (item 7): device-storage-type
            # assertions, field-exact round-trip, AND a real generate_ghosts!
            # call on the GPU-resident copy (not just a round-trip) — the
            # part item 5 explicitly deferred (ghost hardcoded Vector then).
            n_ghost = ghost.n
            d_entry = adapt(CuArray, ghost_entry)
            d_ghost = d_entry.ghost
            @test getfield(d_ghost, :x) isa CuArray{SVector{2,Float64}}
            @test getfield(d_ghost, :idx_original) isa CuArray{Int}
            @test getfield(d_entry, :_flags) isa CuArray{Int}
            @test d_ghost.n == n_ghost
            r_entry = adapt(Array, d_entry)
            @test r_entry.ghost.n == n_ghost
            @test getfield(r_entry.ghost, :x)[1:n_ghost] == getfield(ghost, :x)[1:n_ghost]

            # Move the GPU source and regenerate — proves the adapted entry
            # is a live, working GhostEntry, not just a value-preserving copy.
            d_source = getfield(d_ghost, :source)
            copyto!(getfield(d_source, :x), CuArray([SVector(0.1 * i, 0.03) for i in 1:6]))
            generate_ghosts!(d_entry)
            @test d_ghost.n > 0

            # Functional GPU twin of "Adapted system is fully functional"
            # above — this is Tier 3 in miniature, and where a broken
            # struct-to-device wiring actually shows up.
            n_fluid = 25
            h = 0.6
            fluid2 = FluidParticleSystem("fluid", n_fluid, 2, 1.0, 20.0;
                                         source_v=[0.0, -9.81], state_updater=TaitEOSUpdater(1000.0))
            let k = 1
                for i in 0:4, j in 0:4
                    fluid2.x[k] = SVector(0.5 * i, 0.5 * j)
                    k += 1
                end
            end
            fill!(fluid2.v, zero(SVector{2,Float64}))
            fluid2.rho .= 1000.0
            update_state!(fluid2, 1)

            boundary2 = BasicParticleSystem("boundary", 12, 2, 1.0, 20.0)
            let k = 1
                for i in 0:11
                    boundary2.x[k] = SVector(0.5 * i, -0.5)
                    k += 1
                end
            end
            boundary2.rho .= 1000.0
            fill!(boundary2.v, zero(SVector{2,Float64}))

            fluid2_g    = adapt(CuArray, fluid2)
            boundary2_g = adapt(CuArray, boundary2)

            kernel = CubicSplineKernel(h; ndims=2)
            static_boundary_g = StaticBoundarySystem(boundary2_g, 0.5)
            fluid_int_g = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid2_g; onesided=true, ka=true)
            fluid_boundary_int_g = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid2_g, static_boundary_g;
                                                     onesided=true, ka=true)

            integrator_g = LeapFrogTimeIntegrator([fluid2_g, boundary2_g], [fluid_int_g, fluid_boundary_int_g])
            CUDA.allowscalar(false)
            time_integrate!(integrator_g, 10, 1000, 1000, 0.1, nothing; print_timer=false)

            @test all(isfinite, Array(reinterpret(Float64, fluid2_g.x)))
            @test all(isfinite, Array(fluid2_g.rho))
            @test all(r -> r > 0, Array(fluid2_g.rho))
        end

    end

end
