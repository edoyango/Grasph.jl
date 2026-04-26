using Test
using Grasph
using HDF5
using StaticArrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _make_line_fluid(; n=8, dx=0.1, rho0=1000.0, c=100.0)
    h = 1.2 * dx
    ps = FluidParticleSystem("fluid", n, 2, rho0*dx^2, c;
                             state_updater=TaitEOSUpdater(rho0))
    for i in 1:n
        ps.x[i] = SVector((i-1)*dx, 0.0)
    end
    ps.rho .= rho0
    ps.p   .= 0.0
    fill!(ps.v, zero(SVector{2,Float64}))
    kernel = CubicSplineKernel(h; ndims=2)
    ps, kernel, h
end

function _make_crossing_probe_case()
    h = 0.25
    rho0 = 1000.0
    ps = FluidParticleSystem("fluid", 2, 2, rho0, 1.0;
                             state_updater=TaitEOSUpdater(rho0))
    ps.x[1] = SVector(0.0, 0.0)
    ps.x[2] = SVector(1.0, 0.0)
    ps.rho .= rho0
    ps.p   .= 0.0
    ps.v[1] = SVector(20.0, 0.0)
    ps.v[2] = SVector(-20.0, 0.0)

    kernel = CubicSplineKernel(h; ndims=2)
    probe = ProbeParticleSystem("probe", ps; extras=(cnt=zeros(Int, ps.n),))
    si_self = SystemInteraction(kernel, nothing, ps)
    si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
    lf = LeapFrogTimeIntegrator([ps], [si_self];
                                probes=(probe,), probe_interactions=(si_probe,))
    return lf, ps, probe
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

@testset "ProbeParticleSystem construction" begin

    @testset "explicit positions — scalar extras" begin
        positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:5]
        probe = ProbeParticleSystem("p", positions; extras=(nbr=zeros(Int,5),))
        @test probe.name   == "p"
        @test probe.n      == 5
        @test probe.ndims  == 2
        @test eltype(probe.x) == SVector{2,Float64}
        @test probe.x ≈ positions
        @test probe.id == 1:5
        @test probe.w_sum == zeros(5)
        @test probe.nbr == zeros(Int,5)
    end

    @testset "explicit positions — SVector extras" begin
        positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:3]
        probe = ProbeParticleSystem("p", positions;
            extras=(s = [zero(SVector{3,Float64}) for _ in 1:3],))
        @test eltype(probe.s) == SVector{3,Float64}
        @test length(probe.s) == 3
    end

    @testset "mirror_target constructor copies source positions" begin
        ps, _, _ = _make_line_fluid(; n=6)
        probe = ProbeParticleSystem("probe_m", ps; extras=(cnt=zeros(Int,6),))
        @test probe.n == ps.n
        @test probe.x ≈ ps.x
        @test probe.mirror_target === ps
    end

    @testset "dtype kwarg sets element type" begin
        positions = [SVector(Float32(i)*0.1f0, 0.0f0) for i in 1:4]
        probe = ProbeParticleSystem("p32", positions; dtype=Float32)
        @test eltype(probe.x) == SVector{2,Float32}
        @test eltype(probe.w_sum) == Float32
    end

    @testset "extras length mismatch raises ArgumentError" begin
        positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:5]
        @test_throws ArgumentError ProbeParticleSystem("p", positions;
            extras=(cnt=zeros(Int, 3),))
    end

    @testset "empty positions raises ArgumentError" begin
        @test_throws ArgumentError ProbeParticleSystem("p", SVector{2,Float64}[])
    end

end

# ---------------------------------------------------------------------------
# getproperty
# ---------------------------------------------------------------------------

@testset "ProbeParticleSystem getproperty" begin

    positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:4]
    probe = ProbeParticleSystem("p", positions;
        extras=(cnt=zeros(Int,4), v=[zero(SVector{2,Float64}) for _ in 1:4]))

    @test probe.ndims == 2
    @test probe.n     == 4
    @test probe.name  == "p"

    @testset "extras fields accessible via getproperty" begin
        @test probe.cnt === probe.extras.cnt
        @test probe.v   === probe.extras.v
    end

    @testset "mirror_target forwarding" begin
        ps, _, _ = _make_line_fluid(; n=4)
        probe_m = ProbeParticleSystem("pm", ps; extras=(cnt=zeros(Int,4),))
        # mass, c, rho forwarded to mirror_target
        @test probe_m.mass ≈ ps.mass
        @test probe_m.c    ≈ ps.c
    end

    @testset "no mirror_target: accessing forwarded field errors" begin
        probe_nm = ProbeParticleSystem("no_m", positions;
            extras=(cnt=zeros(Int,4),))
        @test_throws Exception probe_nm.mass
    end

end

# ---------------------------------------------------------------------------
# auto_zero_probe!
# ---------------------------------------------------------------------------

@testset "auto_zero_probe!" begin

    positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:4]
    probe = ProbeParticleSystem("p", positions;
        extras=(cnt=ones(Int,4), stress=[SVector(1.0,2.0,3.0) for _ in 1:4]))

    probe.w_sum .= 99.0
    Grasph.auto_zero_probe!(probe)

    @test all(iszero, probe.w_sum)
    @test all(iszero, probe.cnt)
    @test all(iszero, probe.stress)

end

# ---------------------------------------------------------------------------
# _particle_arrays
# ---------------------------------------------------------------------------

@testset "_particle_arrays for ProbeParticleSystem" begin

    positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:3]
    probe = ProbeParticleSystem("p", positions;
        extras=(a=zeros(Float64,3), b=zeros(Float64,3)))

    arrs = Grasph._particle_arrays(probe)
    # x, id, w_sum, a, b
    @test length(arrs) == 5
    @test arrs[1] === getfield(probe, :x)
    @test arrs[2] === getfield(probe, :id)
    @test arrs[3] === getfield(probe, :w_sum)

end

# ---------------------------------------------------------------------------
# _sort_probe_by_id!
# ---------------------------------------------------------------------------

@testset "_sort_probe_by_id!" begin

    positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:4]
    probe = ProbeParticleSystem("p", positions; extras=(cnt=zeros(Int,4),))

    # Manually permute so ids are [3,1,4,2] and x is correspondingly shuffled
    getfield(probe, :x)  .= [SVector(0.3,0.0), SVector(0.1,0.0),
                              SVector(0.4,0.0), SVector(0.2,0.0)]
    getfield(probe, :id) .= [3, 1, 4, 2]
    probe.cnt            .= [30, 10, 40, 20]

    perm_buf = Vector{Int}(undef, 4)
    scratch  = Grasph._make_sort_scratch(probe)
    Grasph._sort_probe_by_id!(probe, perm_buf, scratch)

    @test probe.id  == [1, 2, 3, 4]
    @test probe.cnt == [10, 20, 30, 40]
    @test probe.x   ≈ [SVector(0.1,0.0), SVector(0.2,0.0),
                        SVector(0.3,0.0), SVector(0.4,0.0)]

end

# ---------------------------------------------------------------------------
# HDF5 write / read roundtrip
# ---------------------------------------------------------------------------

@testset "ProbeParticleSystem HDF5 roundtrip" begin

    mktempdir() do dir
        positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:5]
        probe = ProbeParticleSystem("p", positions;
            extras=(cnt=Int[10,20,30,40,50],
                    v=[SVector(Float64(i)*0.01, 0.0) for i in 1:5]))
        probe.w_sum .= [1.0,2.0,3.0,4.0,5.0]

        path = joinpath(dir, "probe.h5")
        h5open(path, "w") do f
            write_h5(probe, create_group(f, "p"))
        end

        # Read back into a fresh probe
        probe2 = ProbeParticleSystem("p", positions;
            extras=(cnt=zeros(Int,5), v=[zero(SVector{2,Float64}) for _ in 1:5]))
        h5open(path, "r") do f
            read_h5!(probe2, f["p"])
        end

        @test probe2.x   ≈ probe.x
        @test probe2.cnt == probe.cnt
        @test probe2.v   ≈ probe.v
        @test probe2.id  == 1:5   # id always reset to 1:n on read
    end

    @testset "read resets id to 1:n" begin
        mktempdir() do dir
            positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:3]
            probe = ProbeParticleSystem("p", positions; extras=(cnt=zeros(Int,3),))
            # Simulate a mid-run state: ids are permuted
            getfield(probe, :id) .= [3, 1, 2]
            path = joinpath(dir, "p.h5")
            h5open(path, "w") do f
                write_h5(probe, create_group(f, "p"))
            end
            probe2 = ProbeParticleSystem("p", positions; extras=(cnt=zeros(Int,3),))
            h5open(path, "r") do f
                read_h5!(probe2, f["p"])
            end
            @test probe2.id == [1, 2, 3]
        end
    end

    @testset "read_h5! raises on n mismatch" begin
        mktempdir() do dir
            positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:5]
            probe = ProbeParticleSystem("p", positions; extras=(cnt=zeros(Int,5),))
            path = joinpath(dir, "p.h5")
            h5open(path, "w") do f
                write_h5(probe, create_group(f, "p"))
            end
            probe2 = ProbeParticleSystem("p", [SVector(0.1,0.0), SVector(0.2,0.0)];
                extras=(cnt=zeros(Int,2),))
            h5open(path, "r") do f
                @test_throws ArgumentError read_h5!(probe2, f["p"])
            end
        end
    end

end

# ---------------------------------------------------------------------------
# NeighborCountFn
# ---------------------------------------------------------------------------

@testset "NeighborCountFn" begin

    @testset "counts correct neighbors on a line" begin
        # 8 particles on a line, spacing dx, kernel h = 1.2 dx
        # edge particles (i=1,8) have fewer neighbors than interior ones
        ps, kernel, h = _make_line_fluid(; n=8, dx=0.1)
        n = ps.n
        probe = ProbeParticleSystem("probe", ps; extras=(nbr=zeros(Int,n),))
        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:nbr), ps, probe)

        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))

        mktempdir() do dir
            time_integrate!(lf, 1, 1, 1, 0.1, joinpath(dir,"out"); print_timer=false)
            h5open(joinpath(dir,"out_1.h5"), "r") do f
                nbr = f["probe"]["nbr"][]
                # Exterior probes have fewer neighbors than interior ones
                @test nbr[1] < nbr[4]
                @test nbr[8] < nbr[4]
                # Interior probes should have maximum count
                @test maximum(nbr) == nbr[4] || maximum(nbr) == nbr[5]
            end
        end
    end

    @testset "symmetry: probe at particle matches self-neighbor count" begin
        # 5 evenly spaced particles; probe mirrors them.
        # After one save cadence, probe.nbr[i] = number of source particles
        # within h of position x[i].
        ps, kernel, h = _make_line_fluid(; n=5, dx=0.1)
        probe = ProbeParticleSystem("probe", ps; extras=(nbr=zeros(Int,5),))
        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:nbr), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))
        mktempdir() do dir
            time_integrate!(lf, 1, 1, 1, 0.1, joinpath(dir,"out"); print_timer=false)
            h5open(joinpath(dir,"out_1.h5"), "r") do f
                nbr = f["probe"]["nbr"][]
                # Symmetric pattern for a symmetric domain
                @test nbr[1] == nbr[5]
                @test nbr[2] == nbr[4]
                @test nbr[3] >= nbr[2]
            end
        end
    end

end

# ---------------------------------------------------------------------------
# InterpolateFieldFn dispatch for ProbeParticleSystem
# ---------------------------------------------------------------------------

@testset "InterpolateFieldFn → ProbeParticleSystem" begin

    @testset "constant field is recovered at probe positions" begin
        # 8 particles all with v = (1.0, 0.5); probes placed among them
        # should interpolate the same value (within SPH accuracy)
        ps, kernel, h = _make_line_fluid(; n=8, dx=0.1)
        fill!(ps.v, SVector(1.0, 0.5))

        n_p = 3
        probe = ProbeParticleSystem("probe",
            [SVector(0.35, 0.0), SVector(0.45, 0.0), SVector(0.55, 0.0)];
            extras=(v=[zero(SVector{2,Float64}) for _ in 1:n_p],),
            state_updater=VirtualNormUpdater(SVector(1.0,1.0), :v))

        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, InterpolateFieldFn(:v), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))
        mktempdir() do dir
            time_integrate!(lf, 1, 1, 1, 0.1, joinpath(dir,"out"); print_timer=false)
            h5open(joinpath(dir,"out_1.h5"), "r") do f
                v_data = f["probe"]["v"][]
                # v[1,:] = x-component, v[2,:] = y-component
                @test all(abs.(v_data[1,:] .- 1.0) .< 0.01)
                @test all(abs.(v_data[2,:] .- 0.5) .< 0.01)
            end
        end
    end

    @testset "scalar extras field roundtrip" begin
        ps, kernel, h = _make_line_fluid(; n=6, dx=0.1)
        ps.rho .= 1234.0

        probe = ProbeParticleSystem("probe",
            [SVector(0.2,0.0), SVector(0.3,0.0)];
            extras=(rho=zeros(2),),
            state_updater=VirtualNormUpdater(SVector(1.0,1.0), :rho))

        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, InterpolateFieldFn(:rho), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))
        mktempdir() do dir
            time_integrate!(lf, 1, 1, 1, 0.1, joinpath(dir,"out"); print_timer=false)
            h5open(joinpath(dir,"out_1.h5"), "r") do f
                rho_data = f["probe"]["rho"][]
                @test all(abs.(rho_data .- 1234.0) .< 10.0)
            end
        end
    end

end

# ---------------------------------------------------------------------------
# LeapFrogTimeIntegrator wiring
# ---------------------------------------------------------------------------

@testset "LeapFrogTimeIntegrator probe wiring" begin

    @testset "constructor accepts probes/probe_interactions kwargs" begin
        ps, kernel, h = _make_line_fluid(; n=4)
        probe = ProbeParticleSystem("probe", ps; extras=(cnt=zeros(Int,4),))
        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))
        @test length(lf.probes)             == 1
        @test lf.probes[1]                  === probe
        @test length(lf.probe_interactions) == 1
        @test lf.probe_interactions[1]      === si_probe
    end

    @testset "no probes: empty tuples by default" begin
        ps, kernel, h = _make_line_fluid(; n=4)
        si_self = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        lf = LeapFrogTimeIntegrator([ps], [si_self])
        @test lf.probes             === ()
        @test lf.probe_interactions === ()
    end

    @testset "prescribed_v advances probe positions every step" begin
        ps, kernel, h = _make_line_fluid(; n=4)
        v_probe = SVector(1.0, 0.0)
        probe = ProbeParticleSystem("probe", [SVector(0.0, 0.0)];
            extras=(cnt=zeros(Int,1),), prescribed_v=v_probe)
        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))

        dt = 0.1 * h / lf.c
        x0 = probe.x[1]
        # Run 3 steps without saving (no HDF5 I/O needed)
        time_integrate!(lf, 3, 100, 100, 0.1, nothing; print_timer=false)
        @test probe.x[1] ≈ x0 + 3 * dt * v_probe  atol=1e-12
    end

    @testset "mirror_target: positions copied at save time" begin
        ps, kernel, h = _make_line_fluid(; n=4)
        # Give source particles a nonzero velocity so they move
        fill!(ps.v, SVector(0.5, 0.0))

        probe = ProbeParticleSystem("probe", ps; extras=(cnt=zeros(Int,4),))
        x0 = copy(probe.x)   # initial probe positions = initial source positions

        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))

        mktempdir() do dir
            time_integrate!(lf, 5, 100, 5, 0.1, joinpath(dir,"out"); print_timer=false)
            # After 5 steps, source particles have moved; probe should mirror them
            # (positions overwritten with mirror_target.x during _measure_probes!)
            # After the save, probe is sorted by id, so check id-order is consistent
            h5open(joinpath(dir,"out_5.h5"), "r") do f
                id = f["probe"]["id"][]
                @test id == 1:ps.n
            end
        end
    end

end

# ---------------------------------------------------------------------------
# Id-stable ordering
# ---------------------------------------------------------------------------

@testset "id-stable HDF5 ordering" begin

    @testset "id is always 1:n in saved files" begin
        ps, kernel, h = _make_line_fluid(; n=6, dx=0.1)
        fill!(ps.v, SVector(0.3, 0.0))  # particles move, causing re-sorts

        probe = ProbeParticleSystem("probe", ps; extras=(cnt=zeros(Int,6),))
        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))
        mktempdir() do dir
            time_integrate!(lf, 4, 100, 2, 0.1, joinpath(dir,"out"); print_timer=false)
            for step in [2, 4]
                h5open(joinpath(dir,"out_$(step).h5"), "r") do f
                    id = f["probe"]["id"][]
                    @test id == 1:6
                end
            end
        end
    end

    @testset "probe row k always maps to original probe k across saves" begin
        # Place 5 probes at distinct positions; check that each save has the
        # same row-to-position mapping (probe 1 always x=0.1, etc.)
        dx = 0.1; h = 1.2*dx; rho0 = 1000.0; c = 100.0
        n = 10
        ps = FluidParticleSystem("fluid", n, 2, rho0*dx^2, c;
                                 state_updater=TaitEOSUpdater(rho0))
        for i in 1:n; ps.x[i] = SVector((i-1)*dx, 0.0); end
        ps.rho .= rho0; ps.p .= 0.0
        fill!(ps.v, SVector(0.05, 0.0))

        n_p = 5
        orig_pos = [SVector(i*0.15, 0.0) for i in 1:n_p]
        probe = ProbeParticleSystem("probe", orig_pos; extras=(cnt=zeros(Int,n_p),))

        kernel = CubicSplineKernel(h; ndims=2)
        si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
        si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
        lf = LeapFrogTimeIntegrator([ps], [si_self];
                                    probes=(probe,), probe_interactions=(si_probe,))

        mktempdir() do dir
            time_integrate!(lf, 6, 100, 2, 0.1, joinpath(dir,"out"); print_timer=false)
            xs_by_save = []
            for step in [2, 4, 6]
                h5open(joinpath(dir,"out_$(step).h5"), "r") do f
                    push!(xs_by_save, f["probe"]["x"][]')
                end
            end
            # x[:,1] (first probe) should be the same position at every save
            # (probe has no prescribed_v and no mirror_target, so x shouldn't change)
            for (i, xs) in enumerate(xs_by_save[2:end])
                @test xs_by_save[1] ≈ xs   atol=1e-10
            end
        end
    end

end

# ---------------------------------------------------------------------------
# Restart: probe state is saved and reloaded correctly
# ---------------------------------------------------------------------------

@testset "probe restart roundtrip" begin

    ps, kernel, h = _make_line_fluid(; n=6, dx=0.1)

    probe = ProbeParticleSystem("probe", ps;
        extras=(cnt=zeros(Int,6), v_acc=[zero(SVector{2,Float64}) for _ in 1:6]))
    si_self  = SystemInteraction(kernel, FluidPfn(0.1,0.0,h), ps)
    si_probe = SystemInteraction(kernel, NeighborCountFn(:cnt), ps, probe)
    lf = LeapFrogTimeIntegrator([ps], [si_self];
                                probes=(probe,), probe_interactions=(si_probe,))

    mktempdir() do dir
        run_driver!([Stage(lf, 4, 0.1, "run")], 1, 2,
                    joinpath(dir,"out"); interactive=false)

        h5open(joinpath(dir,"out_4.h5"), "r") do f
            @test haskey(f, "probe")
            g = f["probe"]
            @test Int(HDF5.attrs(g)["n"][]) == 6
            @test Int(HDF5.attrs(g)["ndims"][]) == 2
            @test g["id"][] == 1:6
        end
    end

end

# ---------------------------------------------------------------------------
# Active regressions for known probe bugs
# ---------------------------------------------------------------------------

@testset "probe regression bugs" begin

    @testset "probe measurement uses freshly sorted source positions" begin
        lf, _, _ = _make_crossing_probe_case()

        mktempdir() do dir
            time_integrate!(lf, 1, 100, 1, 1.0, joinpath(dir, "out"); print_timer=false)
            h5open(joinpath(dir, "out_1.h5"), "r") do f
                @test f["probe"]["cnt"][] == [1, 1]
            end
        end
    end

    @testset "mirrored probes follow original source particle identity" begin
        lf, _, _ = _make_crossing_probe_case()

        mktempdir() do dir
            time_integrate!(lf, 2, 100, 2, 1.0, joinpath(dir, "out"); print_timer=false)
            h5open(joinpath(dir, "out_2.h5"), "r") do f
                probe_x = f["probe"]["x"][]
                @test probe_x[1, :] ≈ [10.0, -9.0]
            end
        end
    end

    @testset "ProbeParticleSystem HDF5 path roundtrip" begin
        mktempdir() do dir
            positions = [SVector(Float64(i)*0.1, 0.0) for i in 1:4]
            probe = ProbeParticleSystem("p", positions;
                extras=(cnt=Int[10,20,30,40],
                        v=[SVector(Float64(i)*0.01, -Float64(i)*0.02) for i in 1:4]))
            getfield(probe, :id) .= [4, 1, 3, 2]

            path = joinpath(dir, "probe_path.h5")
            path_error = nothing
            try
                write_h5(probe, path)
            catch err
                path_error = err
            end
            @test path_error === nothing
            path_error === nothing || return

            probe2 = ProbeParticleSystem("p", positions;
                extras=(cnt=zeros(Int,4), v=[zero(SVector{2,Float64}) for _ in 1:4]))
            read_h5!(probe2, path)

            @test probe2.x   ≈ probe.x
            @test probe2.cnt == probe.cnt
            @test probe2.v   ≈ probe.v
            @test probe2.id  == 1:4
        end
    end

end
