using Test
using Grasph
using HDF5
using StaticArrays

@testset "AbstractParticleSystem" begin

    # ------------------------------------------------------------------
    # BasicParticleSystem construction and field shapes
    # ------------------------------------------------------------------

    @testset "BasicParticleSystem default construction" begin
        ps = BasicParticleSystem("test", 10, 3, 1.0, 2.0)
        @test ps.name   == "test"
        @test ps.n      == 10
        @test ps.ndims  == 3
        @test ps.mass   == 1.0
        @test ps.c      == 2.0
        @test length(ps.x)     == 10
        @test eltype(ps.x)     == SVector{3,Float64}
        @test length(ps.v)     == 10
        @test eltype(ps.v)     == SVector{3,Float64}
        @test length(ps.dvdt)  == 10
        @test eltype(ps.dvdt)  == SVector{3,Float64}
        @test size(ps.rho)    == (10,)
        @test size(ps.drhodt) == (10,)
    end

    @testset "FluidParticleSystem has pressure field" begin
        ps = FluidParticleSystem("fluid", 5, 2, 1.0, 2.0)
        @test hasfield(typeof(ps), :p)
        @test length(ps.p) == 5
        @test eltype(ps.p) === Float64
    end

    @testset "StressParticleSystem has stress and strain_rate fields" begin
        ps = StressParticleSystem("granular", 5, 2, 3, 1.0, 2.0)
        @test hasfield(typeof(ps), :p)
        @test hasfield(typeof(ps), :stress)
        @test hasfield(typeof(ps), :strain_rate)
        @test length(ps.stress) == 5
        @test eltype(ps.stress) == SVector{3,Float64}
        @test length(ps.strain_rate) == 5
    end

    @testset "StressParticleSystem ns validation" begin
        @test_throws ArgumentError StressParticleSystem("s", 5, 2, 2, 1.0, 1.0)
        @test_throws ArgumentError StressParticleSystem("s", 5, 2, 5, 1.0, 1.0)
        @test StressParticleSystem("s", 5, 2, 3, 1.0, 1.0) isa StressParticleSystem
        @test StressParticleSystem("s", 5, 2, 4, 1.0, 1.0) isa StressParticleSystem
        @test StressParticleSystem("s", 5, 2, 6, 1.0, 1.0) isa StressParticleSystem
    end

    @testset "dtype propagation — BasicParticleSystem" begin
        ps32 = BasicParticleSystem("f32", 5, 2, 1.0, 2.0; dtype=Float32)
        @test eltype(ps32.x)      === SVector{2,Float32}
        @test eltype(ps32.v)      === SVector{2,Float32}
        @test eltype(ps32.dvdt)   === SVector{2,Float32}
        @test eltype(ps32.rho)    === Float32
        @test eltype(ps32.drhodt) === Float32
        @test ps32.mass isa Float32
        @test ps32.c    isa Float32
    end

    @testset "dtype propagation — FluidParticleSystem" begin
        ps32 = FluidParticleSystem("f32", 5, 2, 1.0, 2.0; dtype=Float32)
        @test eltype(ps32.x)      === SVector{2,Float32}
        @test eltype(ps32.dvdt)   === SVector{2,Float32}
        @test eltype(ps32.rho)    === Float32
        @test eltype(ps32.p)      === Float32
        @test ps32.mass isa Float32
        @test ps32.c    isa Float32
    end

    @testset "dtype propagation — StressParticleSystem" begin
        ps32 = StressParticleSystem("f32", 5, 2, 3, 1.0, 2.0; dtype=Float32)
        @test eltype(ps32.x)           === SVector{2,Float32}
        @test eltype(ps32.dvdt)        === SVector{2,Float32}
        @test eltype(ps32.rho)         === Float32
        @test eltype(ps32.p)           === Float32
        @test eltype(ps32.stress)      === SVector{3,Float32}
        @test eltype(ps32.strain_rate) === SVector{3,Float32}
        @test ps32.mass isa Float32
        @test ps32.c    isa Float32
    end

    @testset "dvdt and drhodt initialised to zero" begin
        ps = BasicParticleSystem("test", 4, 2, 1.0, 1.0)
        @test all(iszero, ps.dvdt)
        @test all(iszero, ps.drhodt)
    end

    # ------------------------------------------------------------------
    # Source terms
    # ------------------------------------------------------------------

    @testset "default source_v is zero" begin
        ps = BasicParticleSystem("test", 5, 2, 1.0, 1.0)
        @test ps.source_v  == zeros(2)
        @test ps.source_rho == 0.0
    end

    @testset "custom source_v (gravity)" begin
        ps = BasicParticleSystem("test", 5, 2, 1.0, 1.0; source_v=[0.0, -9.81])
        @test ps.source_v ≈ [0.0, -9.81]
    end

    @testset "custom source_rho" begin
        ps = BasicParticleSystem("test", 5, 2, 1.0, 1.0; source_rho=3.14)
        @test ps.source_rho ≈ 3.14
    end

    @testset "source_v element type follows dtype" begin
        ps = BasicParticleSystem("test", 3, 2, 1.0, 1.0;
                            dtype=Float32, source_v=Float32[0.0, -9.81])
        @test eltype(ps.source_v) === Float32
    end

    # ------------------------------------------------------------------
    # State updater
    # ------------------------------------------------------------------

    @testset "nothing state updater leaves user field unchanged" begin
        ps = FluidParticleSystem("test", 4, 2, 1.0, 1.0)
        ps.p .= 99.0
        update_state!(ps)
        @test all(==(99.0), ps.p)
    end

    @testset "custom state updater p = rho + c" begin
        ps = FluidParticleSystem("test", 10, 3, 1.0, 10.0;
            state_updater = (ps, i, dt) -> ps.p[i] = ps.rho[i] + ps.c)
        ps.rho .= 0.0:9.0
        update_state!(ps)
        for i in 1:10
            @test ps.p[i] ≈ (i - 1) + 10.0
        end
    end

    @testset "virtual prescribed velocity is not accumulated when v is auto-zeroed" begin
        ps = BasicParticleSystem("source", 2, 2, 1.0, 1.0)
        v_prescribed = SVector(0.0, -0.005)
        vps = VirtualParticleSystem(ps, "virtual", 2, 2, 1.0, 1.0;
            zero_fields   = (:v,),
            prescribed_v  = v_prescribed,
            state_updater = (nothing, (PrescribedVelocityUpdater(),)),
        )

        for _ in 1:3
            Grasph.auto_zero_virtual!(vps)
            update_state!(vps, 2, 0.1)
            @test all(==(v_prescribed), ps.v)
        end
    end

    # ------------------------------------------------------------------
    # Constructor error handling
    # ------------------------------------------------------------------

    @testset "n ≤ 0 raises" begin
        @test_throws ArgumentError BasicParticleSystem("test", 0,  2, 1.0, 1.0)
        @test_throws ArgumentError BasicParticleSystem("test", -1, 2, 1.0, 1.0)
    end

    @testset "ndims ≤ 0 raises" begin
        @test_throws ArgumentError BasicParticleSystem("test", 10, 0,  1.0, 1.0)
        @test_throws ArgumentError BasicParticleSystem("test", 10, -1, 1.0, 1.0)
    end

    @testset "source_v wrong length raises" begin
        @test_throws ArgumentError BasicParticleSystem("test", 10, 2, 1.0, 1.0;
                                                   source_v=[1.0, 2.0, 3.0])
    end

    # ------------------------------------------------------------------
    # State updater float-type checking
    # ------------------------------------------------------------------

    @testset "state updater matching float type does not raise" begin
        u = TaitEOSUpdater(Float64(1000.0))
        @test_nowarn FluidParticleSystem("test", 5, 2, 1.0, 10.0; state_updater=u)
    end

    @testset "state updater mismatched float type raises" begin
        u = TaitEOSUpdater(Float32(1000.0))
        @test_throws ArgumentError FluidParticleSystem("test", 5, 2, 1.0, 10.0; state_updater=u)
    end

    @testset "state updater mismatch error mentions updater type and both float types" begin
        u = TaitEOSUpdater(Float32(1000.0))
        try
            FluidParticleSystem("test", 5, 2, 1.0, 10.0; state_updater=u)
            @test false
        catch e
            @test e isa ArgumentError
            @test occursin("TaitEOSUpdater", e.msg)
            @test occursin("Float32", e.msg)
            @test occursin("Float64", e.msg)
        end
    end

    @testset "unparameterized updater (ZeroFieldUpdater) does not raise" begin
        u = ZeroFieldUpdater(:rho)
        @test_nowarn BasicParticleSystem("test", 5, 2, 1.0, 1.0; state_updater=u)
    end

    @testset "tuple of updaters — mismatch in second raises" begin
        u1 = ZeroFieldUpdater(:rho)
        u2 = LinearEOSUpdater(Float32(1000.0))
        @test_throws ArgumentError FluidParticleSystem("test", 5, 2, 1.0, 10.0;
                                                       state_updater=(u1, u2))
    end

    @testset "state updater mismatch also caught for dtype=Float32 system with Float64 updater" begin
        u = TaitEOSUpdater(Float64(1000.0))
        @test_throws ArgumentError FluidParticleSystem("test", 5, 2, 1.0, 10.0;
                                                       dtype=Float32, state_updater=u)
    end

    # ------------------------------------------------------------------
    # Print-field management
    # ------------------------------------------------------------------

    @testset "no print fields by default" begin
        ps = BasicParticleSystem("test", 2, 2, 1.0, 1.0)
        buf = IOBuffer()
        print_summary(ps, buf)
        @test isempty(String(take!(buf)))
    end

    @testset "add_print_field! unknown field raises" begin
        ps = BasicParticleSystem("test", 2, 2, 1.0, 1.0)
        @test_throws ArgumentError add_print_field!(ps, :nonexistent)
    end

    @testset "remove_print_field! not-in-list raises" begin
        ps = BasicParticleSystem("test", 2, 2, 1.0, 1.0)
        @test_throws ArgumentError remove_print_field!(ps, :v)
    end

    @testset "add then remove print field" begin
        ps = BasicParticleSystem("test", 2, 2, 1.0, 1.0)
        add_print_field!(ps, :v)
        @test :v ∈ ps._print_fields
        remove_print_field!(ps, :v)
        @test :v ∉ ps._print_fields
    end

    # ------------------------------------------------------------------
    # print_summary content
    # ------------------------------------------------------------------

    @testset "print_summary scalar field stats" begin
        ps = BasicParticleSystem("test", 4, 1, 1.0, 1.0)
        ps.rho .= [1.0, 2.0, 3.0, 4.0]
        add_print_field!(ps, :rho)
        buf = IOBuffer()
        print_summary(ps, buf)
        out = String(take!(buf))
        @test occursin("|", out)          # markdown table
        @test occursin("rho", out)
        @test occursin("1", out)          # min value
        @test occursin("2.5", out)        # mean
        @test occursin("4", out)          # max value
        @test occursin("i_min", out)
        @test occursin("i_max", out)
    end

    @testset "print_summary vector field mag and components" begin
        ps = BasicParticleSystem("test", 3, 2, 1.0, 1.0)
        # particle 1: (3,4) → mag=5; particle 2: (0,0) → mag=0; particle 3: (1,0) → mag=1
        ps.v .= [SVector(3.0, 4.0), SVector(0.0, 0.0), SVector(1.0, 0.0)]
        add_print_field!(ps, :v)
        buf = IOBuffer()
        print_summary(ps, buf)
        out = String(take!(buf))
        @test occursin("|mag|", out)
        @test occursin("[1]", out)
        @test occursin("[2]", out)
        @test occursin("5", out)   # max magnitude
        @test occursin("0", out)   # min magnitude
    end

    @testset "print_summary not printed field absent" begin
        ps = BasicParticleSystem("test", 2, 2, 1.0, 1.0)
        fill!(ps.v, zero(SVector{2,Float64})); ps.rho .= 1.0
        # neither field added → empty output
        buf = IOBuffer()
        print_summary(ps, buf)
        out = String(take!(buf))
        @test !occursin("v", out)
        @test !occursin("rho", out)
    end

    @testset "remove_print_field! stops printing" begin
        ps = BasicParticleSystem("test", 2, 2, 1.0, 1.0)
        fill!(ps.v, zero(SVector{2,Float64}))
        add_print_field!(ps, :v)
        remove_print_field!(ps, :v)
        buf = IOBuffer()
        print_summary(ps, buf)
        out = String(take!(buf))
        @test !occursin("v", out)
    end

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    @testset "write_h5 metadata attributes (BasicParticleSystem)" begin
        mktempdir() do dir
            ps = BasicParticleSystem("test", 3, 2, 1.5, 2.0)
            ps.x .= [SVector(0.0, 1.0), SVector(2.0, 3.0), SVector(4.0, 5.0)]
            fill!(ps.v, zero(SVector{2,Float64}))
            ps.rho .= [1.0, 2.0, 3.0]

            path = joinpath(dir, "out.h5")
            write_h5(ps, path)

            h5open(path, "r") do f
                @test HDF5.attrs(f)["n"][]     == 3
                @test HDF5.attrs(f)["ndims"][] == 2
                @test HDF5.attrs(f)["mass"][]  ≈ 1.5
                @test HDF5.attrs(f)["c"][]     ≈ 2.0
            end
        end
    end

    @testset "write_h5 array datasets (BasicParticleSystem)" begin
        mktempdir() do dir
            ps = BasicParticleSystem("test", 3, 2, 1.0, 1.0)
            ps.x .= [SVector(0.0, 1.0), SVector(2.0, 3.0), SVector(4.0, 5.0)]
            fill!(ps.v, zero(SVector{2,Float64}))
            ps.rho .= [1.0, 2.0, 3.0]

            path = joinpath(dir, "out.h5")
            write_h5(ps, path)

            h5open(path, "r") do f
                @test f["x"][]   ≈ reinterpret(reshape, Float64, ps.x)
                @test f["v"][]   ≈ reinterpret(reshape, Float64, ps.v)
                @test f["rho"][] ≈ ps.rho
            end
        end
    end

    @testset "write_h5 to existing group" begin
        mktempdir() do dir
            ps = BasicParticleSystem("fluid", 2, 2, 1.0, 1.0)
            ps.x .= [SVector(1.0, 3.0), SVector(2.0, 4.0)]

            path = joinpath(dir, "out.h5")
            h5open(path, "w") do f
                write_h5(ps, create_group(f, "fluid"))
            end

            h5open(path, "r") do f
                @test HDF5.attrs(f["fluid"])["n"][] == 2
                @test f["fluid/x"][] ≈ reinterpret(reshape, Float64, ps.x)
            end
        end
    end

    @testset "write_h5 FluidParticleSystem includes p dataset" begin
        mktempdir() do dir
            ps = FluidParticleSystem("test", 2, 2, 1.0, 1.0)
            ps.p .= [3.14, 2.72]

            path = joinpath(dir, "out.h5")
            write_h5(ps, path)

            h5open(path, "r") do f
                @test f["p"][] ≈ ps.p
            end
        end
    end

end
