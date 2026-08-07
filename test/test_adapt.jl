using Test
using Grasph
using StaticArrays
using Adapt

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

end
