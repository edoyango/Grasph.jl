using Test
using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# CPU regression guard. The risk this migration poses to the other 12
# experiment scripts (all on the default coloured sweep, never touching the
# KA/GPU path) isn't the new GPU code itself — it's type instability from
# threading new type parameters (CSA, MODE) through SystemInteraction. An
# @allocated check catches that precisely, where a wall-clock timing
# assertion would just be flaky.
# ---------------------------------------------------------------------------

@testset "CPU legacy path is unaffected by the GPU migration" begin

    @testset "default SystemInteraction still selects the coloured sweep" begin
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        fluid = FluidParticleSystem("fluid", 20, 2, 1.0, 10.0)
        si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid)
        @test Grasph._exec_mode(si) isa Grasph.ColouredCPU
        @test si._cell_start isa Vector{Int}
    end

    @testset "onesided=true (no ka) still selects the Polyester one-sided path, not KA" begin
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        fluid = FluidParticleSystem("fluid", 20, 2, 1.0, 10.0)
        si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
        @test Grasph._exec_mode(si) isa Grasph.OnesidedCPU
    end

    @testset "ka=true without onesided=true is rejected" begin
        h = 0.1
        kernel = CubicSplineKernel(h; ndims=2)
        fluid = FluidParticleSystem("fluid", 20, 2, 1.0, 10.0)
        @test_throws ArgumentError SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; ka=true)
    end

    @testset "legacy coloured-sweep dambreak-shaped run has flat per-step allocation" begin
        h = 0.08
        kernel = CubicSplineKernel(h; ndims=2)
        dx = 0.06
        rho0 = 1000.0
        nfx = 8
        n_fluid = nfx * nfx
        fluid = FluidParticleSystem("fluid", n_fluid, 2, rho0 * dx^2, 20.0;
                                    source_v = [0.0, -9.81], state_updater = TaitEOSUpdater(rho0))
        k = 1
        for i in 0:nfx-1, j in 0:nfx-1
            fluid.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
            k += 1
        end
        fill!(fluid.v, zero(SVector{2,Float64}))
        fluid.rho .= rho0
        update_state!(fluid, 1)

        nb = nfx + 4
        bnd = BasicParticleSystem("boundary", nb, 2, rho0 * dx^2, 20.0)
        k = 1
        for i in -2:nfx+1
            bnd.x[k] = SVector((i + 0.5) * dx, -0.5 * dx)
            k += 1
        end
        bnd.rho .= rho0
        fill!(bnd.v, zero(SVector{2,Float64}))

        si_self = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid)
        si_bnd  = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid, StaticBoundarySystem(bnd, dx))
        integrator = LeapFrogTimeIntegrator([fluid, bnd], [si_self, si_bnd])

        # Warm up (JIT) before measuring.
        time_integrate!(integrator, 3, 1000, 1000, 0.1, nothing; print_timer=false)
        alloc_per_10 = @allocated time_integrate!(integrator, 10, 1000, 1000, 0.1, nothing; print_timer=false)
        # Generous ceiling: TimerOutputs' own bookkeeping allocates a little
        # already (pre-existing, not this migration's doing) — the point is
        # "flat and small", not literally zero. A real regression here (e.g.
        # a captured type parameter breaking inference) would show up as
        # tens of MB, not a few KB.
        @test alloc_per_10 < 200_000
    end

end
