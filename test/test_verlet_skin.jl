using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# verlet_skin: persistent/cached grid + rebuild-cadence gate
# (docs/gpu-migration-plan.md, deferred item 1)
#
# `verlet_skin > 0` opts LeapFrogTimeIntegrator/RK4TimeIntegrator into
# skipping the per-step sort + grid rebuild while no tracked particle has
# moved far enough to invalidate the current (skin-padded) cell list. This
# file checks:
#   - the ArgumentError guards (onesided-only, no ghosts/virtual, skin bound)
#   - that a skin > 0 run reproduces a skin = 0 (rebuild-every-step) run,
#     both under tame motion (rebuild only ever triggers once, at step 1)
#     and under fast motion (many rebuild-trigger events over the run)
#   - the probe-triggered cache-invalidation path
# ---------------------------------------------------------------------------

# A dambreak-shaped fixture (regular fluid block + box boundary), matching
# bench/dambreak_scaling.jl's geometry — deliberately NOT a random cloud:
# the earlier onesided-vs-coloured CPU investigation (see
# docs/gpu-migration-plan.md) found random/near-overlapping clouds amplify
# ordinary floating-point reordering into large-looking (but spurious) diffs.
# A regular, well-conditioned layout keeps this test sensitive to genuine
# bugs without that false-positive risk.
#
# `speed` gives every fluid particle a uniform initial velocity (0 for the
# "tame" case; nonzero to force the particles to cross several cells over
# the run, exercising multiple rebuild-trigger events rather than just the
# unconditional step-1 rebuild).
function _verlet_dambreak(nfx; verlet_skin=0.0, ka=false, speed=(0.0, 0.0), integrator=:leapfrog)
    dx = 0.5
    h  = 1.2 * dx
    rho0 = 1000.0
    c  = 10.0 * sqrt(2.0 * 9.81 * 25.0)

    nfy = nfx
    nbx = nfx + 8
    nby = nfx + 8

    n_fluid = nfx * nfy
    mass = rho0 * dx * dx
    fluid = FluidParticleSystem("fluid", n_fluid, 2, mass, c;
                                 source_v = [0.0, -9.81], state_updater = TaitEOSUpdater(rho0))
    k = 1
    for i in 0:nfx-1, j in 0:nfy-1
        fluid.x[k] = SVector((i + 0.5) * dx, (j + 3.5) * dx)
        k += 1
    end
    fill!(fluid.v, SVector(speed...))
    fluid.rho .= rho0
    update_state!(fluid, 1)

    n_b = 2 * (nbx + nby) + 4
    boundary = BasicParticleSystem("boundary", n_b, 2, mass, c)
    k = 1
    for i in -1:nbx
        boundary.x[k] = SVector((i + 0.5) * dx, -0.5 * dx); k += 1
    end
    for i in -1:nbx
        boundary.x[k] = SVector((i + 0.5) * dx, nby * dx + 0.5 * dx); k += 1
    end
    for j in 0:nby-1
        boundary.x[k] = SVector(-0.5 * dx, (j + 0.5) * dx); k += 1
    end
    for j in 0:nby-1
        boundary.x[k] = SVector(nbx * dx + 0.5 * dx, (j + 0.5) * dx); k += 1
    end
    boundary.rho .= rho0
    fill!(boundary.v, zero(SVector{2,Float64}))

    kernel = CubicSplineKernel(h; ndims=2)
    static_boundary = StaticBoundarySystem(boundary, dx)
    fi  = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true, ka=ka)
    fbi = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid, static_boundary; onesided=true, ka=ka)

    integ = if integrator == :leapfrog
        LeapFrogTimeIntegrator([fluid, boundary], [fi, fbi]; verlet_skin=verlet_skin)
    else
        RK4TimeIntegrator([fluid, boundary], [fi, fbi]; verlet_skin=verlet_skin)
    end
    return integ, fluid, boundary
end

# Compare two fluid systems' x/v by particle id (index-position alignment is
# NOT guaranteed to match between a verlet_skin > 0 run — fewer re-sorts — and
# a verlet_skin = 0 run — a re-sort every step — even when the physics is
# identical), then assert a tight relative tolerance rather than exact `==`:
# the two runs generally use differently-pitched grids (padded vs unpadded
# cell width), which can reorder floating-point summation even when the set
# of interacting pairs is identical — see this file's header comment.
function _assert_close(fluid_a, fluid_b; rtol=1e-9, atol=1e-10)
    perm_a = sortperm(fluid_a.id)
    perm_b = sortperm(fluid_b.id)
    @test fluid_a.id[perm_a] == fluid_b.id[perm_b]
    dx_max = maximum(norm.(fluid_a.x[perm_a] .- fluid_b.x[perm_b]))
    dv_max = maximum(norm.(fluid_a.v[perm_a] .- fluid_b.v[perm_b]))
    scale  = maximum(norm.(fluid_a.x[perm_a]))
    @test dx_max <= atol + rtol * scale
    @test dv_max <= atol + rtol * max(scale, 1.0)
end

@testset "verlet_skin" begin

    @testset "constructor guards" begin
        dx = 0.5; h = 1.2 * dx; c = 10.0; mass = 1.0
        fluid = FluidParticleSystem("fluid", 4, 2, mass, c;
                                     source_v = [0.0, -9.81], state_updater = TaitEOSUpdater(1000.0))
        fluid.x .= [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
        fluid.rho .= 1000.0
        kernel = CubicSplineKernel(h; ndims=2)

        @testset "coloured interaction rejected" begin
            si_coloured = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid)  # default: coloured
            @test_throws ArgumentError LeapFrogTimeIntegrator([fluid], [si_coloured]; verlet_skin=0.1)
            @test_throws ArgumentError RK4TimeIntegrator([fluid], [si_coloured]; verlet_skin=0.1)
        end

        @testset "onesided interaction accepted" begin
            si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
            lf = LeapFrogTimeIntegrator([fluid], [si]; verlet_skin=0.1)
            @test lf.verlet_skin ≈ 0.1
        end

        @testset "skin == 0 is always accepted regardless of mode" begin
            si_coloured = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid)
            lf = LeapFrogTimeIntegrator([fluid], [si_coloured]; verlet_skin=0.0)
            @test lf.verlet_skin == 0.0
        end

        @testset "ghosts rejected" begin
            si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
            ghost = GhostParticleSystem(fluid, GhostCopier(:p))
            entry = GhostEntry(ghost, 3h, (SVector(1.0, 0.0), SVector(0.0, 0.0)))
            @test_throws ArgumentError LeapFrogTimeIntegrator([fluid], [si]; ghosts=entry, verlet_skin=0.1)
        end

        @testset "virtual_systems rejected" begin
            si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
            vps = VirtualParticleSystem(fluid, "virt", fluid.n, 2, mass, c)
            @test_throws ArgumentError LeapFrogTimeIntegrator([fluid], [si]; virtual_systems=vps, verlet_skin=0.1)
        end

        @testset "negative skin rejected" begin
            si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
            @test_throws ArgumentError LeapFrogTimeIntegrator([fluid], [si]; verlet_skin=-0.1)
        end

        @testset "skin >= 2*cutoff rejected" begin
            si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
            cutoff = kernel.interaction_length
            @test_throws ArgumentError LeapFrogTimeIntegrator([fluid], [si]; verlet_skin=2*cutoff)
            @test_throws ArgumentError LeapFrogTimeIntegrator([fluid], [si]; verlet_skin=10*cutoff)
        end
    end

    @testset "create_grid! skin defaults to 0 (unchanged behaviour)" begin
        dx = 0.5; h = 1.2 * dx; c = 10.0; mass = 1.0
        fluid = FluidParticleSystem("fluid", 4, 2, mass, c; source_v=[0.0,-9.81], state_updater=TaitEOSUpdater(1000.0))
        fluid.x .= [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)]
        fluid.rho .= 1000.0
        kernel = CubicSplineKernel(h; ndims=2)
        si = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
        create_grid!(si)
        @test si._grid_cutoff[] == si._cell_size
        create_grid!(si, 0.2)
        @test si._grid_cutoff[] ≈ si._cell_size + 0.2
        create_grid!(si)   # skin omitted again -> back to unpadded
        @test si._grid_cutoff[] == si._cell_size
    end

    @testset "LeapFrog: skin > 0 matches skin = 0 (tame motion, self+coupled onesided CPU)" begin
        integ0, fluid0, _ = _verlet_dambreak(16; verlet_skin=0.0)
        time_integrate!(integ0, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        integ1, fluid1, _ = _verlet_dambreak(16; verlet_skin=0.1)
        time_integrate!(integ1, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        _assert_close(fluid0, fluid1)
    end

    @testset "LeapFrog: skin > 0 matches skin = 0 (fast motion, many rebuild-trigger events)" begin
        integ0, fluid0, _ = _verlet_dambreak(14; verlet_skin=0.0, speed=(3.0, 1.5))
        time_integrate!(integ0, 60, 10^9, 10^9, 0.15, nothing; print_timer=false)

        integ1, fluid1, _ = _verlet_dambreak(14; verlet_skin=0.08, speed=(3.0, 1.5))
        time_integrate!(integ1, 60, 10^9, 10^9, 0.15, nothing; print_timer=false)

        _assert_close(fluid0, fluid1)
    end

    @testset "LeapFrog: skin > 0 matches skin = 0 on KA.CPU() (ka=true)" begin
        integ0, fluid0, _ = _verlet_dambreak(14; verlet_skin=0.0, ka=true, speed=(2.0, -1.0))
        time_integrate!(integ0, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        integ1, fluid1, _ = _verlet_dambreak(14; verlet_skin=0.1, ka=true, speed=(2.0, -1.0))
        time_integrate!(integ1, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        _assert_close(fluid0, fluid1)
    end

    @testset "RK4: skin > 0 matches skin = 0" begin
        integ0, fluid0, _ = _verlet_dambreak(14; verlet_skin=0.0, speed=(2.0, 1.0), integrator=:rk4)
        time_integrate!(integ0, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        integ1, fluid1, _ = _verlet_dambreak(14; verlet_skin=0.08, speed=(2.0, 1.0), integrator=:rk4)
        time_integrate!(integ1, 40, 10^9, 10^9, 0.15, nothing; print_timer=false)

        _assert_close(fluid0, fluid1)
    end

    @testset "probes force a rebuild on the step after they're measured" begin
        # A probe's source system gets re-sorted by _measure_probes! at save
        # cadence, independent of the main gate — this must not desync the
        # cached run from a skin=0 reference. Save every 5 steps so the probe
        # re-sort interacts with the cache multiple times over the run.
        dx = 0.5; h = 1.2*dx; c = 10.0*sqrt(2.0*9.81*25.0)
        function build_with_probe(; verlet_skin)
            nfx = 12
            mass = 1000.0*dx*dx
            fluid = FluidParticleSystem("fluid", nfx*nfx, 2, mass, c;
                                         source_v=[0.0,-9.81], state_updater=TaitEOSUpdater(1000.0))
            k = 1
            for i in 0:nfx-1, j in 0:nfx-1
                fluid.x[k] = SVector((i+0.5)*dx, (j+3.5)*dx); k += 1
            end
            fill!(fluid.v, SVector(1.5, 0.5))
            fluid.rho .= 1000.0
            update_state!(fluid, 1)

            nb = nfx + 8
            n_b = 2*(nb+nb)+4
            boundary = BasicParticleSystem("boundary", n_b, 2, mass, c)
            k = 1
            for i in -1:nb
                boundary.x[k] = SVector((i+0.5)*dx, -0.5*dx); k += 1
            end
            for i in -1:nb
                boundary.x[k] = SVector((i+0.5)*dx, nb*dx+0.5*dx); k += 1
            end
            for j in 0:nb-1
                boundary.x[k] = SVector(-0.5*dx, (j+0.5)*dx); k += 1
            end
            for j in 0:nb-1
                boundary.x[k] = SVector(nb*dx+0.5*dx, (j+0.5)*dx); k += 1
            end
            boundary.rho .= 1000.0
            fill!(boundary.v, zero(SVector{2,Float64}))

            kernel = CubicSplineKernel(h; ndims=2)
            static_boundary = StaticBoundarySystem(boundary, dx)
            fi  = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid; onesided=true)
            fbi = SystemInteraction(kernel, FluidPfn(0.01, 0.0, h), fluid, static_boundary; onesided=true)

            probe = ProbeParticleSystem("probe", fluid; extras=(cnt=zeros(Int, fluid.n),))
            pint  = SystemInteraction(kernel, NeighborCountFn(:cnt), fluid, probe; onesided=true)

            integ = LeapFrogTimeIntegrator([fluid, boundary], [fi, fbi];
                                            probes=probe, probe_interactions=pint,
                                            verlet_skin=verlet_skin)
            return integ, fluid
        end

        mktempdir() do tmpdir
            integ0, fluid0 = build_with_probe(; verlet_skin=0.0)
            time_integrate!(integ0, 20, 10^9, 5, 0.15, joinpath(tmpdir, "skin0"); print_timer=false)

            integ1, fluid1 = build_with_probe(; verlet_skin=0.08)
            time_integrate!(integ1, 20, 10^9, 5, 0.15, joinpath(tmpdir, "skin1"); print_timer=false)

            _assert_close(fluid0, fluid1)
        end
    end

end
