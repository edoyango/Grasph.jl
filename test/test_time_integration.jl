using Test
using Grasph
using HDF5
using StaticArrays
using TimerOutputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_noop(args...) = nothing

function _make_lf(; n=4, c=10.0, h=0.1)
    ps = BasicParticleSystem("fluid", n, 2, 1.0, c)
    vals = collect(range(0.2, 0.8, length=n*2))
    mat  = reshape(vals, 2, n)
    ps.x .= [SVector(mat[1, i], mat[2, i]) for i in 1:n]
    fill!(ps.v, zero(SVector{2,Float64}))
    ps.rho .= 1000.0
    k  = CubicSplineKernel(h; ndims=2)
    si = SystemInteraction(k, _noop, ps)
    lf = LeapFrogTimeIntegrator(ps, si)
    lf, ps, si
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

@testset "LeapFrogTimeIntegrator constructor" begin

    @testset "single system and interaction accepted" begin
        lf, ps, si = _make_lf()
        @test lf.systems[1]      === ps
        @test lf.interactions[1] === si
    end

    @testset "c is max over all systems" begin
        ps1 = BasicParticleSystem("a", 2, 2, 1.0, 3.0)
        ps2 = BasicParticleSystem("b", 2, 2, 1.0, 7.0)
        fill!(ps1.x, SVector(0.3, 0.3)); fill!(ps2.x, SVector(0.3, 0.3))
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps1)
        lf = LeapFrogTimeIntegrator([ps1, ps2], [si])
        @test lf.c ≈ 7.0
    end

    @testset "vector of systems and interactions" begin
        ps1 = BasicParticleSystem("a", 2, 2, 1.0, 1.0)
        ps2 = BasicParticleSystem("b", 2, 2, 1.0, 1.0)
        fill!(ps1.x, SVector(0.3, 0.3)); fill!(ps2.x, SVector(0.3, 0.3))
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps1)
        lf = LeapFrogTimeIntegrator([ps1, ps2], [si])
        @test length(lf.systems)      == 2
        @test length(lf.interactions) == 1
    end

    @testset "empty systems raises ArgumentError" begin
        ps = BasicParticleSystem("fluid", 2, 2, 1.0, 1.0)
        fill!(ps.x, SVector(0.3, 0.3))
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps)
        @test_throws ArgumentError LeapFrogTimeIntegrator([], [si])
    end

    @testset "empty interactions raises ArgumentError" begin
        ps = BasicParticleSystem("fluid", 2, 2, 1.0, 1.0)
        fill!(ps.x, SVector(0.3, 0.3))
        @test_throws ArgumentError LeapFrogTimeIntegrator([ps], [])
    end

end

# ---------------------------------------------------------------------------
# time_integrate! — kinematics
# ---------------------------------------------------------------------------

@testset "time_integrate! kinematics" begin

    @testset "constant velocity advances x correctly" begin
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0)
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))
        ps.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps)
        lf = LeapFrogTimeIntegrator(ps, si)

        dt = 0.5 * 0.1 / 10.0   # CFL=0.5, h=0.1, c=10.0
        time_integrate!(lf, 3, 100, 100, 0.5, nothing)

        @test ps.x[1][1] ≈ 3 * dt  atol=1e-10
        @test ps.x[1][2] ≈ 0.0     atol=1e-10
    end

    @testset "zero velocity, no force: position unchanged" begin
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0)
        ps.x[1] = SVector(0.3, 0.4)
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps)
        lf = LeapFrogTimeIntegrator(ps, si)
        x0 = copy(ps.x)

        time_integrate!(lf, 5, 100, 100, 0.5, nothing)

        @test ps.x ≈ x0  atol=1e-12
    end

    @testset "zero drhodt source: rho unchanged with noop sweep" begin
        ps = BasicParticleSystem("fluid", 2, 2, 1.0, 10.0)
        ps.x .= [SVector(0.2, 0.2), SVector(0.4, 0.4)]
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= [800.0, 1200.0]
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps)
        lf = LeapFrogTimeIntegrator(ps, si)
        rho0 = copy(ps.rho)

        time_integrate!(lf, 3, 100, 100, 0.5, nothing)

        @test ps.rho ≈ rho0  atol=1e-12
    end

    @testset "source_v drives acceleration" begin
        # Particle at rest with downward source_v; v and x should become negative in dim 2
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0; source_v=[0.0, -1.0])
        ps.x[1] = SVector(0.5, 0.5)
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps)
        lf = LeapFrogTimeIntegrator(ps, si)

        time_integrate!(lf, 4, 100, 100, 0.5, nothing)

        @test ps.v[1][2] < 0
        @test ps.x[1][2] < 0.5
        @test ps.v[1][1] ≈ 0.0  atol=1e-12
    end

    @testset "Float32 integration preserves eltype throughout" begin
        ps = BasicParticleSystem("fluid", 4, 2, 1.0, 10.0; dtype=Float32)
        vals = collect(range(Float32(0.2), Float32(0.8); length=8))
        mat  = reshape(vals, 2, 4)
        ps.x   .= [SVector(mat[1,i], mat[2,i]) for i in 1:4]
        fill!(ps.v, zero(SVector{2,Float32}))
        ps.rho .= Float32(1000.0)
        k  = CubicSplineKernel(Float32(0.1); ndims=2)
        si = SystemInteraction(k, _noop, ps)
        lf = LeapFrogTimeIntegrator(ps, si)

        @test lf.c isa Float32
        @test lf.h isa Float32

        time_integrate!(lf, 3, 100, 100, 0.5, nothing)

        @test eltype(ps.x)      === SVector{2,Float32}
        @test eltype(ps.v)      === SVector{2,Float32}
        @test eltype(ps.dvdt)   === SVector{2,Float32}
        @test eltype(ps.rho)    === Float32
        @test eltype(ps.drhodt) === Float32
    end

    @testset "dvdt accumulator reset between steps" begin
        # Inject a one-shot force on the first sweep call only;
        # subsequent steps should not feel it
        ps = BasicParticleSystem("fluid", 2, 2, 1.0, 10.0)
        ps.x .= [SVector(0.2, 0.2), SVector(0.4, 0.4)]
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0

        injected = Ref(false)
        dvdt = ps.dvdt
        function one_shot_pfn(ps, i, j, dx, gx, w)
            if !injected[]
                dvdt[1] = Base.setindex(dvdt[1], dvdt[1][1] + 1000.0, 1)
                dvdt[2] = Base.setindex(dvdt[2], dvdt[2][1] + 1000.0, 1)
                injected[] = true
            end
        end

        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, one_shot_pfn, ps)
        lf = LeapFrogTimeIntegrator(ps, si)

        time_integrate!(lf, 2, 100, 100, 0.5, nothing)

        # After step 2, dvdt should have been reset (source_v = [0,0])
        # so the step-1 impulse does not persist into step 2 as a baseline
        @test ps.dvdt[1][1] ≈ 0.0  atol=1e-10
        @test ps.dvdt[2][1] ≈ 0.0  atol=1e-10
    end

end

# ---------------------------------------------------------------------------
# time_integrate! — state updater and callbacks
# ---------------------------------------------------------------------------

@testset "time_integrate! state updater" begin

    @testset "update_state! called once per system per step" begin
        call_count = Ref(0)
        ps = FluidParticleSystem("fluid", 1, 2, 1.0, 10.0;
            state_updater = (ps, i, dt) -> (call_count[] += 1; ps.p[i] = 0.0)
        )
        fill!(ps.x, SVector(0.5, 0.5))
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps)
        lf = LeapFrogTimeIntegrator(ps, si)

        n = 4
        time_integrate!(lf, n, 100, 100, 0.5, nothing)

        @test call_count[] == n
    end

end

# ---------------------------------------------------------------------------
# time_integrate! — multi-stage (pfns tuple) validation
# ---------------------------------------------------------------------------

@testset "time_integrate! stages validation" begin

    @testset "mismatched pfns lengths across interactions raises AssertionError" begin
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0)
        fill!(ps.x, SVector(0.3, 0.3))
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        k = CubicSplineKernel(0.1; ndims=2)
        si1 = SystemInteraction(k, (_noop, _noop), ps)   # 2 stages
        si2 = SystemInteraction(k, _noop,           ps)  # 1 stage
        lf  = LeapFrogTimeIntegrator(ps, [si1, si2])
        @test_throws AssertionError time_integrate!(lf, 1, 100, 100, 0.5, nothing)
    end

    @testset "mismatched state updater count emits warning" begin
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0; state_updater=_noop)
        fill!(ps.x, SVector(0.3, 0.3))
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        # SI has 2 stages; ps has only 1 state updater → should warn
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, (_noop, _noop), ps)
        lf = LeapFrogTimeIntegrator(ps, si)
        @test_logs (:warn, r"stages 2 and later will skip") time_integrate!(lf, 1, 100, 100, 0.5, nothing)
    end

    @testset "matching state updater count emits no warning" begin
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0; state_updater=(_noop, _noop))
        fill!(ps.x, SVector(0.3, 0.3))
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, (_noop, _noop), ps)
        lf = LeapFrogTimeIntegrator(ps, si)
        @test_logs time_integrate!(lf, 1, 100, 100, 0.5, nothing)
    end

    @testset "each stage calls its own state updater" begin
        counts = [Ref(0), Ref(0)]
        ps = BasicParticleSystem("fluid", 1, 2, 1.0, 10.0;
            state_updater = (
                (ps, i, dt) -> counts[1][] += 1,
                (ps, i, dt) -> counts[2][] += 1,
            )
        )
        fill!(ps.x, SVector(0.3, 0.3))
        fill!(ps.v, zero(SVector{2,Float64}))
        ps.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, (_noop, _noop), ps)
        lf = LeapFrogTimeIntegrator(ps, si)

        n = 3
        time_integrate!(lf, n, 100, 100, 0.5, nothing)

        @test counts[1][] == n
        @test counts[2][] == n
    end

end

# ---------------------------------------------------------------------------
# time_integrate! — HDF5 output
# ---------------------------------------------------------------------------

@testset "time_integrate! HDF5 output" begin

    @testset "files created at correct save intervals" begin
        lf, ps, _ = _make_lf()
        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            time_integrate!(lf, 4, 100, 2, 0.5, prefix)
            @test  isfile("$(prefix)_2.h5")
            @test  isfile("$(prefix)_4.h5")
            @test !isfile("$(prefix)_1.h5")
            @test !isfile("$(prefix)_3.h5")
        end
    end

    @testset "step numbers are zero-padded to num_timesteps width" begin
        lf, ps, _ = _make_lf()
        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            # num_timesteps=10 → width=2 → "01","02",...
            time_integrate!(lf, 10, 100, 5, 0.5, prefix)
            @test  isfile("$(prefix)_05.h5")
            @test  isfile("$(prefix)_10.h5")
            @test !isfile("$(prefix)_5.h5")
        end
    end

    @testset "HDF5 file contains one group per system" begin
        ps1 = BasicParticleSystem("fluid",    2, 2, 1.0, 10.0)
        ps2 = BasicParticleSystem("boundary", 2, 2, 1.0,  5.0)
        ps1.x .= [SVector(0.2, 0.4), SVector(0.2, 0.4)]; fill!(ps1.v, zero(SVector{2,Float64})); ps1.rho .= 1000.0
        ps2.x .= [SVector(0.1, 0.9), SVector(0.5, 0.5)]; fill!(ps2.v, zero(SVector{2,Float64})); ps2.rho .= 1000.0
        k  = CubicSplineKernel(0.1; ndims=2)
        si = SystemInteraction(k, _noop, ps1)
        lf = LeapFrogTimeIntegrator([ps1, ps2], [si])

        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            time_integrate!(lf, 2, 100, 2, 0.5, prefix)
            h5open("$(prefix)_2.h5", "r") do f
                @test haskey(f, "fluid")
                @test haskey(f, "boundary")
            end
        end
    end

    @testset "nothing output_prefix skips saving" begin
        lf, _, _ = _make_lf()
        # Should complete without error and create no files
        @test begin
            time_integrate!(lf, 2, 100, 1, 0.5, nothing)
            true
        end
    end

    @testset "output_prefix with subdirectory creates parent dirs" begin
        lf, _, _ = _make_lf()
        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "subdir", "run")
            time_integrate!(lf, 2, 100, 2, 0.5, prefix)
            @test isfile("$(prefix)_2.h5")
        end
    end
end

@testset "time_integrate! timing" begin
    lf, ps, si = _make_lf()
    to = time_integrate!(lf, 2, 100, 100, 0.5, nothing; print_timer=false)
    @test to isa TimerOutput
    @test length(to.inner_timers) > 0
end

# ---------------------------------------------------------------------------
# run_driver! — CLI overrides
# ---------------------------------------------------------------------------

function _with_args(f, argv::Vector{String})
    orig = copy(ARGS)
    empty!(ARGS)
    append!(ARGS, argv)
    try
        f()
    finally
        empty!(ARGS)
        append!(ARGS, orig)
    end
end

@testset "run_driver! CLI overrides (per-stage flags)" begin

    @testset "Per-stage --<label>-steps overrides stage length" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))

        _with_args(["-n", "--main-steps", "3"]) do
            dt = 0.5 * 0.1 / 10.0
            run_driver!([Stage(lf, 10, 0.5, "main")], 100, 100, nothing)
            @test ps.x[1][1] ≈ 3 * dt atol=1e-10
            @test ps.x[1][2] ≈ 0.0    atol=1e-10
        end
    end

    @testset "Per-stage --<label>-cfl overrides CFL" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))

        _with_args(["-n", "--main-steps", "1", "--main-cfl", "0.25"]) do
            dt = 0.25 * 0.1 / 10.0
            run_driver!([Stage(lf, 10, 0.5, "main")], 100, 100, nothing)
            @test ps.x[1][1] ≈ dt atol=1e-10
        end
    end

    @testset "--save-freq / --print-freq remain global" begin
        lf, ps, _ = _make_lf()
        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            _with_args(["--non-interactive", "--main-steps", "4", "--save-freq", "2", "--print-freq", "1"]) do
                run_driver!([Stage(lf, 10, 0.5, "main")], 100, 100, prefix)
                @test  isfile("$(prefix)_2.h5")
                @test  isfile("$(prefix)_4.h5")
                @test !isfile("$(prefix)_1.h5")
                @test !isfile("$(prefix)_3.h5")
            end
        end
    end

    @testset "--output-prefix=none disables saving" begin
        lf, _, _ = _make_lf()
        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            _with_args(["-n", "--main-steps", "2", "--output-prefix", "none"]) do
                run_driver!([Stage(lf, 2, 0.5, "main")], 1, 1, prefix)
                @test isempty(readdir(tmpdir))
            end
        end
    end
end

@testset "run_driver! multi-stage behaviour" begin

    @testset "Two stages run sequentially with shared step counter" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))
        dt = 0.5 * 0.1 / 10.0
        run_driver!(
            [Stage(lf, 2, 0.5, "damping"), Stage(lf, 3, 0.5, "run")],
            100, 100, nothing; interactive=false,
        )
        @test ps.x[1][1] ≈ 5 * dt atol=1e-10
    end

    @testset "nothing entries are skipped (no integrator required)" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))
        dt = 0.5 * 0.1 / 10.0
        run_driver!([nothing, Stage(lf, 4, 0.5, "run")], 100, 100, nothing; interactive=false)
        @test ps.x[1][1] ≈ 4 * dt atol=1e-10
    end

    @testset "--<label>-steps 0 skips that stage" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))
        dt = 0.5 * 0.1 / 10.0
        _with_args(["-n", "--damping-steps", "0"]) do
            run_driver!(
                [Stage(lf, 5, 0.5, "damping"), Stage(lf, 2, 0.5, "run")],
                100, 100, nothing,
            )
            @test ps.x[1][1] ≈ 2 * dt atol=1e-10
        end
    end

    @testset "Duplicate stage labels are rejected" begin
        lf, _, _ = _make_lf()
        @test_throws ArgumentError run_driver!(
            [Stage(lf, 1, 0.5, "main"), Stage(lf, 1, 0.5, "main")],
            100, 100, nothing; interactive=false,
        )
    end

    @testset "All-nothing stages raises ArgumentError" begin
        @test_throws ArgumentError run_driver!([nothing], 100, 100, nothing; interactive=false)
    end
end

@testset "run_driver! HDF5 step attr + restart" begin

    @testset "HDF5 output has step attribute" begin
        lf, _, _ = _make_lf()
        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            run_driver!(
                [Stage(lf, 3, 0.5, "main")], 3, 3, prefix; interactive=false,
            )
            path = "$(prefix)_3.h5"
            @test isfile(path)
            h5open(path, "r") do f
                @test Int(HDF5.attrs(f)["step"][]) == 3
                @test HDF5.attrs(f)["sim_time"][] isa Float64
            end
        end
    end

    @testset "Restart resumes from checkpointed step and particle state" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))
        dt = 0.5 * 0.1 / 10.0

        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")

            # First run: 2-step "damping" stage writes a checkpoint at step 2.
            run_driver!([Stage(lf, 2, 0.5, "damping"), Stage(lf, 0, 0.5, "run")],
                        2, 2, prefix; interactive=false)
            ckpt = "$(prefix)_2.h5"
            @test isfile(ckpt)
            x_after_damping = ps.x[1][1]
            @test x_after_damping ≈ 2 * dt atol=1e-10

            # Reset state; fresh driver should restore from checkpoint and only run "run".
            lf2, ps2, _ = _make_lf()
            fill!(ps2.x, zero(SVector{2,Float64}))
            fill!(ps2.v, SVector(1.0, 0.0))

            run_driver!(
                [Stage(lf2, 2, 0.5, "damping"), Stage(lf2, 3, 0.5, "run")],
                100, 100, nothing;
                interactive = false,
                restart     = ckpt,
            )
            # damping (2 steps) is skipped because checkpoint is at step 2; run adds 3 more.
            @test ps2.x[1][1] ≈ x_after_damping + 3 * dt atol=1e-10
        end
    end

    @testset "--restart CLI flag is honoured" begin
        lf, ps, _ = _make_lf()
        fill!(ps.x, zero(SVector{2,Float64}))
        fill!(ps.v, SVector(1.0, 0.0))
        dt = 0.5 * 0.1 / 10.0

        mktempdir() do tmpdir
            prefix = joinpath(tmpdir, "out")
            run_driver!([Stage(lf, 2, 0.5, "main")], 2, 2, prefix; interactive=false)
            ckpt = "$(prefix)_2.h5"

            lf2, ps2, _ = _make_lf()
            fill!(ps2.x, zero(SVector{2,Float64}))
            fill!(ps2.v, SVector(1.0, 0.0))
            _with_args(["-n", "--restart", ckpt]) do
                run_driver!([Stage(lf2, 5, 0.5, "main")], 100, 100, nothing)
            end
            # main is 5 steps cumulative; checkpoint at step 2 ⇒ 3 more steps run.
            @test ps2.x[1][1] ≈ 2 * dt + 3 * dt atol=1e-10
        end
    end
end
