using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Cluster E — ellipse.jl (the "elliptical drop" configuration).
#
# This is the simplest interaction shape in the whole script survey: a single
# FluidParticleSystem with a *self*-interaction only (FluidPfn), no boundary,
# no ghosts, no virtual particles, no probes. Particles are laid out on a
# regular grid clipped to a disc and given a prescribed straining initial
# velocity field (v = (-100x, 100y)); LeapFrogTimeIntegrator drives it.
#
# This test proves the onesided=true sweep reproduces the default coloured
# half-shell sweep in-context, through a real multi-step LeapFrogTimeIntegrator
# run, mirroring the shape of ellipse.jl exactly but at reduced scale (see
# test/test_onesided_sweep.jl sections 5-6 for the template this follows).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared builder — deterministic regular grid clipped to the unit disc,
# exactly mirroring ellipse.jl's `_circle_particles` + setup, just at a
# coarser dx so the reduced-scale harness runs in seconds. Called twice
# independently (not deepcopy'd) so the two runs start from bit-identical,
# non-aliased state.
# ---------------------------------------------------------------------------

function _ellipse_like(; dx=0.13)
    h_sph          = 1.2 * dx
    rho0           = 1000.0
    c_sound        = 1400.0
    art_visc_alpha = 0.01
    art_visc_beta  = 0.0

    n = Int(round(2.0 / dx))
    xs = Float64[]
    ys = Float64[]
    for i in 0:n-1, j in 0:n-1
        x = -1.0 + (i + 0.5) * dx
        y = -1.0 + (j + 0.5) * dx
        x * x + y * y < 1.0 && (push!(xs, x); push!(ys, y))
    end
    n_particles = length(xs)

    particle_mass = π * rho0 / n_particles
    particles = FluidParticleSystem(
        "fluid", n_particles, 2, particle_mass, c_sound;
        state_updater = TaitEOSUpdater(rho0)
    )
    for k in 1:n_particles
        particles.x[k]   = SVector(xs[k], ys[k])
        particles.v[k]   = SVector(-100.0 * xs[k], 100.0 * ys[k])
        particles.rho[k] = rho0
    end
    update_state!(particles)

    kernel = CubicSplineKernel(h_sph; ndims=2)
    pfn    = FluidPfn(art_visc_alpha, art_visc_beta, h_sph)

    return kernel, pfn, particles
end

# ---------------------------------------------------------------------------
# 5. Short-run trajectory equivalence — onesided=true vs onesided=false
# ---------------------------------------------------------------------------

@testset "short-run trajectory equivalence: onesided=true vs onesided=false (ellipse)" begin
    kernel_old, pfn_old, particles_old = _ellipse_like()
    kernel_new, pfn_new, particles_new = _ellipse_like()

    si_old = SystemInteraction(kernel_old, pfn_old, particles_old)
    si_new = SystemInteraction(kernel_new, pfn_new, particles_new; onesided=true)

    lf_old = LeapFrogTimeIntegrator(particles_old, si_old)
    lf_new = LeapFrogTimeIntegrator(particles_new, si_new)

    nsteps = 50
    CFL    = 0.05
    time_integrate!(lf_old, nsteps, nsteps + 1, nsteps + 1, CFL, nothing; print_timer=false)
    time_integrate!(lf_new, nsteps, nsteps + 1, nsteps + 1, CFL, nothing; print_timer=false)

    x_scale   = max(maximum(norm.(particles_old.x)), 1.0)
    v_scale   = max(maximum(norm.(particles_old.v)), 1.0)
    rho_scale = max(maximum(abs.(particles_old.rho)), 1.0)

    x_diff   = maximum(norm.(particles_old.x   .- particles_new.x))
    v_diff   = maximum(norm.(particles_old.v   .- particles_new.v))
    rho_diff = maximum(abs.(particles_old.rho .- particles_new.rho))

    @test !any(isnan, reduce(vcat, [collect(v) for v in particles_old.x]))
    @test !any(isnan, reduce(vcat, [collect(v) for v in particles_new.x]))
    @test x_diff   < 1e-8 * x_scale
    @test v_diff   < 1e-6 * v_scale
    @test rho_diff < 1e-6 * rho_scale
end

# ---------------------------------------------------------------------------
# 6. Long-run physical invariants (onesided=true path only)
# ---------------------------------------------------------------------------

@testset "long-run physical invariants (onesided=true, ellipse)" begin
    kernel, pfn, particles = _ellipse_like()
    si = SystemInteraction(kernel, pfn, particles; onesided=true)
    lf = LeapFrogTimeIntegrator(particles, si)

    nsteps = 140
    CFL    = 0.05
    time_integrate!(lf, nsteps, nsteps + 1, nsteps + 1, CFL, nothing; print_timer=false)

    @test all(!isnan, particles.rho)
    @test all(!isinf, particles.rho)
    @test all(x -> all(!isnan, x), particles.x)
    @test all(x -> all(!isnan, x), particles.v)
    # The drop starts confined to the unit disc with no boundary; a broken
    # one-sided sweep (e.g. missing/duplicated pair contributions) would
    # show up as unphysical blow-up well beyond this loose bound.
    @test all(x -> all(abs.(x) .< 10.0), particles.x)
end
