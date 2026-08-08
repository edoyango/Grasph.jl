using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm
using CUDA
using Adapt

# ---------------------------------------------------------------------------
# Tier 3 — reduced-scale end-to-end dambreak-shaped parity, CPU vs CUDA.
#
# Mirrors dambreak.jl's actual construction (FluidParticleSystem +
# BasicParticleSystem wall, FluidPfn, TaitEOSUpdater, LeapFrogTimeIntegrator)
# at a small particle count, driving `time_integrate!` directly rather than
# `run_driver!` — the latter defaults to `interactive=true` and blocks on
# `readline`.
#
# Elementwise comparison only up to a modest step count: SPH is chaotic, and
# the existing onesided-sweep equivalence tests already show ~1e3-1e5
# amplification of a ~1e-11-scale perturbation over 100 steps. A 1e-16-scale
# CPU/GPU seed difference would grow past any tolerance well before 1000
# steps regardless of correctness, so a long-run gate uses physical
# invariants instead of elementwise equality.
# ---------------------------------------------------------------------------

const CUDA_OK_D = CUDA.functional()

function _t3_build(nfx; backend = nothing)
    dx_spacing = 0.5
    h_sph = 1.2 * dx_spacing
    rho0 = 1000.0
    c_sound = 10.0 * sqrt(2.0 * 9.81 * 25.0)
    art_visc_alpha = 0.01
    art_visc_beta = 0.0

    nfy = nfx
    nbx = nfx + 4
    nby = 2 * nfx

    n_fluid = nfx * nfy
    fluid_mass = rho0 * dx_spacing * dx_spacing

    fluid = FluidParticleSystem("fluid", n_fluid, 2, fluid_mass, c_sound;
                                source_v = [0.0, -9.81], state_updater = TaitEOSUpdater(rho0))
    let k = 1
        for i in 0:nfx-1, j in 0:nfy-1
            fluid.x[k] = SVector((i + 0.5) * dx_spacing, (j + 0.5) * dx_spacing)
            k += 1
        end
    end
    fill!(fluid.v, zero(SVector{2,Float64}))
    fluid.rho .= rho0
    update_state!(fluid, 1)

    n_boundary = 2 * (nbx + nby) + 4
    boundary = BasicParticleSystem("boundary", n_boundary, 2, fluid_mass, c_sound)
    let k = 1
        for i in -1:nbx
            boundary.x[k] = SVector((i + 0.5) * dx_spacing, -0.5 * dx_spacing); k += 1
        end
        for i in -1:nbx
            boundary.x[k] = SVector((i + 0.5) * dx_spacing, nby * dx_spacing + 0.5 * dx_spacing); k += 1
        end
        for j in 0:nby-1
            boundary.x[k] = SVector(-0.5 * dx_spacing, (j + 0.5) * dx_spacing); k += 1
        end
        for j in 0:nby-1
            boundary.x[k] = SVector(nbx * dx_spacing + 0.5 * dx_spacing, (j + 0.5) * dx_spacing); k += 1
        end
    end
    boundary.rho .= rho0
    fill!(boundary.v, zero(SVector{2,Float64}))

    kernel = CubicSplineKernel(h_sph; ndims=2)

    ka_mode = backend !== nothing
    if ka_mode
        fluid    = adapt(backend, fluid)
        boundary = adapt(backend, boundary)
    end

    static_boundary = StaticBoundarySystem(boundary, dx_spacing)
    fluid_interaction = SystemInteraction(kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid;
                                          onesided = ka_mode, ka = ka_mode)
    fluid_boundary_interaction = SystemInteraction(kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph),
                                                   fluid, static_boundary; onesided = ka_mode, ka = ka_mode)
    integrator = LeapFrogTimeIntegrator([fluid, boundary], [fluid_interaction, fluid_boundary_interaction])
    return integrator, fluid, boundary, nbx, nby, dx_spacing
end

if !CUDA_OK_D

    @testset "dambreak-shaped end-to-end parity (CUDA)" begin
        @info "CUDA.functional() == false — Tier 3 (dambreak parity) tests skipped" CUDA.functional()
        @test_skip "CUDA not functional on this machine"
    end

else

    @testset "dambreak-shaped end-to-end parity (CUDA)" begin

        @testset "trajectory match at 50 steps" begin
            CUDA.allowscalar(false)
            nfx = 6
            integ_cpu, fluid_cpu, bnd_cpu = _t3_build(nfx)
            integ_gpu, fluid_gpu, bnd_gpu = _t3_build(nfx; backend = CUDABackend())

            # Warmup (h is baked into CubicSplineKernel's type, so the first
            # call compiles the whole sweep/integrator specialisation) —
            # excluded from the compared run.
            time_integrate!(integ_cpu, 1, 1000, 1000, 0.05, nothing; print_timer=false)
            time_integrate!(integ_gpu, 1, 1000, 1000, 0.05, nothing; print_timer=false)
            integ_cpu, fluid_cpu, bnd_cpu = _t3_build(nfx)
            integ_gpu, fluid_gpu, bnd_gpu = _t3_build(nfx; backend = CUDABackend())

            nsteps = 50
            time_integrate!(integ_cpu, nsteps, nsteps + 1, nsteps + 1, 0.05, nothing; print_timer=false)
            time_integrate!(integ_gpu, nsteps, nsteps + 1, nsteps + 1, 0.05, nothing; print_timer=false)

            fluid_gpu_h = adapt(Array, fluid_gpu)
            oc = sortperm(getfield(fluid_cpu, :id))
            og = sortperm(getfield(fluid_gpu_h, :id))

            x_scale   = max(maximum(norm.(fluid_cpu.x)), 1.0)
            v_scale   = max(maximum(norm.(fluid_cpu.v)), 1.0)
            rho_scale = max(maximum(abs.(fluid_cpu.rho)), 1.0)

            x_diff   = maximum(norm.(fluid_cpu.x[oc]   .- fluid_gpu_h.x[og]))
            v_diff   = maximum(norm.(fluid_cpu.v[oc]   .- fluid_gpu_h.v[og]))
            rho_diff = maximum(abs.(fluid_cpu.rho[oc] .- fluid_gpu_h.rho[og]))

            @test all(!isnan, reduce(vcat, [collect(v) for v in fluid_cpu.x]))
            @test x_diff   < 1e-9 * x_scale
            @test v_diff   < 1e-7 * v_scale
            @test rho_diff < 1e-7 * rho_scale
        end

        @testset "long-run physical invariants (300 steps, CPU vs GPU independently)" begin
            CUDA.allowscalar(false)
            nfx = 5
            nsteps = 300

            integ_cpu, fluid_cpu, bnd_cpu, nbx, nby, dx = _t3_build(nfx)
            time_integrate!(integ_cpu, nsteps, nsteps + 1, nsteps + 1, 0.05, nothing; print_timer=false)

            integ_gpu, fluid_gpu, bnd_gpu, _, _, _ = _t3_build(nfx; backend = CUDABackend())
            time_integrate!(integ_gpu, nsteps, nsteps + 1, nsteps + 1, 0.05, nothing; print_timer=false)
            fluid_gpu_h = adapt(Array, fluid_gpu)

            xmax = nbx * dx + dx
            ymax = nby * dx + dx
            for f in (fluid_cpu, fluid_gpu_h)
                @test all(!isnan, f.rho) && all(!isinf, f.rho)
                @test all(x -> all(!isnan, x), f.x)
                @test all(x -> all(!isnan, x), f.v)
                @test sort(getfield(f, :id)) == 1:(nfx*nfx)
                @test all(x -> -dx <= x[1] <= xmax && -dx <= x[2] <= ymax, f.x)
            end
        end

    end

end
