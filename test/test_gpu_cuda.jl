using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm
using Random
using CUDA
using Adapt

# ---------------------------------------------------------------------------
# Tier 2 — CUDA backend equivalence. CUDA.functional()-guarded so the suite
# still passes on a CPU-only machine (CUDA.jl itself always loads; only
# `functional()` depends on an actual driver being present).
#
# Unlike the sweep kernels (Tier 1, KA.CPU()-testable via ka=true regardless
# of array backend), sort/grid dispatch purely on `KA.get_backend(array)` —
# their new histogram/atomic-CSR and gather/copyback kernel code paths are
# only reachable with a real non-CPU backend, so this is the only tier that
# exercises them.
#
# Tolerances: integer/exactly-associative quantities (cell keys, sort
# permutation → cell_start, grid extents) are asserted `==` — min/max and
# integer counting have no rounding error, so any mismatch is a real bug, not
# noise. Float quantities (dvdt, drhodt, p) use a relative tolerance because
# the NVPTX backend contracts `mul`+`add` into `fma` (the host LLVM may not),
# and TaitEOSUpdater's `(rho/rho0)^gamma` dispatches to a different `pow`
# implementation than glibc's — both confirmed by measurement during
# development to differ at ~1-3 ulp, never more.
# ---------------------------------------------------------------------------

const CUDA_OK = CUDA.functional()

function _gpucuda_random_fluid(rng, n, ndims; L=1.0)
    ps = FluidParticleSystem("fluid", n, ndims, 1.0, 10.0; source_v = zeros(ndims))
    for i in 1:n
        ps.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i] = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
    end
    ps.rho .= 1000.0 .+ 20 .* (rand(rng, n) .- 0.5)
    ps.p   .= 100.0 .* rand(rng, n)
    fill!(ps.dvdt, zero(SVector{ndims,Float64}))
    ps.drhodt .= 0.0
    return ps
end

function _gpucuda_random_elastoplastic(rng, n, ndims; L=1.0, ns=(ndims == 2 ? 4 : 6))
    ps = ElastoPlasticParticleSystem("wall", n, ndims, ns, 1.0, 10.0; source_v = zeros(ndims))
    for i in 1:n
        ps.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
        ps.v[i] = SVector(ntuple(_ -> 0.2 * (rand(rng) - 0.5), ndims)...)
    end
    # Constructor already zero-fills p/stress/strain*/dvdt/drhodt; FluidSolidPfn
    # never reads the solid's own p/stress, only its v/rho/mass/c.
    ps.rho .= 2400.0 .+ 20 .* (rand(rng, n) .- 0.5)
    return ps
end

function _gpucuda_random_boundary(rng, n, ndims; L=1.0)
    inner = BasicParticleSystem("bnd", n, ndims, 1.0, 10.0)
    for i in 1:n
        inner.x[i] = SVector(ntuple(_ -> L * rand(rng), ndims)...)
    end
    inner.rho .= 1000.0
    fill!(inner.v, zero(SVector{ndims,Float64}))
    return inner
end

_gpucuda_sortbufs_cpu(ps) = (Vector{Int}(undef, ps.n), Vector{UInt64}(undef, ps.n), Grasph._make_sort_scratch(ps))
_gpucuda_sortbufs_gpu(ps) = (CuArray{Int}(undef, ps.n), CuArray{UInt64}(undef, ps.n), Grasph._make_sort_scratch(ps))

# Reorder-by-id comparison: CUDA.jl's sortperm! is verified stable and
# produces the identical permutation to a stable CPU sort on identical keys,
# but we compare via `id` anyway rather than assume exact ordering matches —
# robust regardless of that stability guarantee holding exactly.
_byid(ps) = sortperm(getfield(ps, :id))

if !CUDA_OK

    @testset "GPU (CUDA backend)" begin
        @info "CUDA.functional() == false — Tier 2 (CUDA backend) tests skipped" CUDA.functional()
        @test_skip "CUDA not functional on this machine"
    end

else

    @testset "GPU (CUDA backend)" begin

        @testset "device_view is isbits after cudaconvert" begin
            fluid = FluidParticleSystem("fluid", 10, 2, 1.0, 10.0)
            fluid_gpu = adapt(CUDABackend(), fluid)
            dv = Grasph.device_view(fluid_gpu)
            @test isbitstype(typeof(cudaconvert(dv)))

            boundary = BasicParticleSystem("bnd", 10, 2, 1.0, 10.0)
            static_bnd = StaticBoundarySystem(adapt(CUDABackend(), boundary), 0.03)
            dvb = Grasph.device_view(static_bnd)
            @test isbitstype(typeof(cudaconvert(dvb)))
        end

        @testset "full pipeline (sort+grid+sweep), self: CPU oracle vs CUDA" begin
            CUDA.allowscalar(false)
            rng = MersenneTwister(201)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length
            pfn = FluidPfn(0.03, 0.0, h)

            n = 2000
            ps_cpu = _gpucuda_random_fluid(rng, n, 2; L=3.0)
            ps_gpu = adapt(CUDABackend(), deepcopy(ps_cpu))

            si_cpu = SystemInteraction(kernel, pfn, ps_cpu; onesided=true)
            pb, kb, sc = _gpucuda_sortbufs_cpu(ps_cpu)
            sort_particles!(ps_cpu, cutoff, pb, kb, sc)
            create_grid!(si_cpu)
            sweep!(si_cpu)

            si_gpu = SystemInteraction(kernel, pfn, ps_gpu; onesided=true, ka=true)
            pbg, kbg, scg = _gpucuda_sortbufs_gpu(ps_gpu)
            sort_particles!(ps_gpu, cutoff, pbg, kbg, scg)
            create_grid!(si_gpu)
            sweep!(si_gpu)

            @test Array(si_gpu._cell_start) == si_cpu._cell_start
            @test Array(si_gpu._mingridx) == Vector(si_cpu._mingridx)
            @test Array(si_gpu._ngridx) == Vector(si_cpu._ngridx)

            ps_gpu_h = adapt(Array, ps_gpu)
            @test sort(getfield(ps_gpu_h, :id)) == 1:n
            oc, og = _byid(ps_cpu), _byid(ps_gpu_h)
            dvdt_scale = max(maximum(norm.(ps_cpu.dvdt)), 1.0)
            drhodt_scale = max(maximum(abs.(ps_cpu.drhodt)), 1.0)
            @test maximum(norm.(ps_cpu.dvdt[oc] .- ps_gpu_h.dvdt[og])) < 1e-10 * dvdt_scale
            @test maximum(abs.(ps_cpu.drhodt[oc] .- ps_gpu_h.drhodt[og])) < 1e-10 * drhodt_scale
        end

        @testset "full pipeline (sort+grid+sweep), coupled: CPU oracle vs CUDA" begin
            CUDA.allowscalar(false)
            rng = MersenneTwister(202)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length
            pfn = FluidPfn(0.03, 0.0, h)

            n_fluid, n_bnd = 1500, 600
            fluid_cpu = _gpucuda_random_fluid(rng, n_fluid, 2; L=3.0)
            bnd_cpu   = _gpucuda_random_boundary(rng, n_bnd, 2; L=3.0)
            static_bnd_cpu = StaticBoundarySystem(bnd_cpu, 0.03)

            fluid_gpu = adapt(CUDABackend(), deepcopy(fluid_cpu))
            bnd_gpu   = adapt(CUDABackend(), deepcopy(bnd_cpu))
            static_bnd_gpu = StaticBoundarySystem(bnd_gpu, 0.03)

            pbb, kbb, scb = _gpucuda_sortbufs_cpu(bnd_cpu)
            sort_particles!(bnd_cpu, cutoff, pbb, kbb, scb)
            pb2, kb2, sc2 = _gpucuda_sortbufs_cpu(fluid_cpu)
            sort_particles!(fluid_cpu, cutoff, pb2, kb2, sc2)
            si_cpu = SystemInteraction(kernel, pfn, fluid_cpu, static_bnd_cpu; onesided=true)
            create_grid!(si_cpu)
            sweep!(si_cpu)

            pbbg, kbbg, scbg = _gpucuda_sortbufs_gpu(bnd_gpu)
            sort_particles!(bnd_gpu, cutoff, pbbg, kbbg, scbg)
            pb2g, kb2g, sc2g = _gpucuda_sortbufs_gpu(fluid_gpu)
            sort_particles!(fluid_gpu, cutoff, pb2g, kb2g, sc2g)
            si_gpu = SystemInteraction(kernel, pfn, fluid_gpu, static_bnd_gpu; onesided=true, ka=true)
            create_grid!(si_gpu)
            sweep!(si_gpu)

            @test Array(si_gpu._cell_start) == si_cpu._cell_start

            fluid_gpu_h = adapt(Array, fluid_gpu)
            oc, og = _byid(fluid_cpu), _byid(fluid_gpu_h)
            dvdt_scale = max(maximum(norm.(fluid_cpu.dvdt)), 1.0)
            @test maximum(norm.(fluid_cpu.dvdt[oc] .- fluid_gpu_h.dvdt[og])) < 1e-10 * dvdt_scale
        end

        @testset "full pipeline (sort+grid+sweep), coupled reverse/WritesBoth: CPU oracle vs CUDA" begin
            # gpu-migration-plan.md "Next steps" item 6: KA kernel twin for
            # the reverse/WritesBoth sweep (_sweep_coupled_ka_reverse!/
            # _sweep_coupled_ka_dispatch!, KAKernels.jl). Uses the same
            # test-only, generically-dispatched _MutualTestPfn/
            # _ReverseOnlyTestPfn as test_onesided_sweep.jl's CPU-side
            # reverse-sweep tests — in scope here via runtests.jl's shared
            # top-level @testset (same reuse convention test_device_views.jl
            # documents) — to validate the sweep infrastructure itself,
            # independent of any one pfn's dispatch typing. Plus the real
            # production FluidPfn fluid-fluid method (bubble.jl/bubble2.jl/
            # bubble3.jl's shape): this used to be unreachable under ka=true
            # (device_view erased which concrete system type produced a
            # DeviceSystem, and FluidPfn's fluid-fluid method needs that
            # identity on both sides — see test_ka_cpu.jl's regression test),
            # fixed by giving DeviceSystem a phantom Kind type parameter
            # (DeviceViews.jl) plus a DeviceSystem{T,ND,FluidParticleSystem}
            # twin of the method (PairwiseFunctors.jl). This is the first
            # real-hardware exercise of that fix.
            CUDA.allowscalar(false)
            rng = MersenneTwister(205)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length

            for pfn in (_ReverseOnlyTestPfn(), _MutualTestPfn(), FluidPfn(0.03, 0.0, h))
                a_cpu = _gpucuda_random_fluid(rng, 800, 2; L=2.0)
                b_cpu = _gpucuda_random_fluid(rng, 600, 2; L=2.0)
                a_gpu = adapt(CUDABackend(), deepcopy(a_cpu))
                b_gpu = adapt(CUDABackend(), deepcopy(b_cpu))

                si_cpu = SystemInteraction(kernel, pfn, a_cpu, b_cpu; onesided=true)
                pa, ka_, sa = _gpucuda_sortbufs_cpu(a_cpu)
                sort_particles!(a_cpu, cutoff, pa, ka_, sa)
                pb, kb, sb = _gpucuda_sortbufs_cpu(b_cpu)
                sort_particles!(b_cpu, cutoff, pb, kb, sb)
                create_grid!(si_cpu)
                sweep!(si_cpu)

                si_gpu = SystemInteraction(kernel, pfn, a_gpu, b_gpu; onesided=true, ka=true)
                pag, kag, sag = _gpucuda_sortbufs_gpu(a_gpu)
                sort_particles!(a_gpu, cutoff, pag, kag, sag)
                pbg, kbg, sbg = _gpucuda_sortbufs_gpu(b_gpu)
                sort_particles!(b_gpu, cutoff, pbg, kbg, sbg)
                create_grid!(si_gpu)
                sweep!(si_gpu)

                a_gpu_h = adapt(Array, a_gpu)
                b_gpu_h = adapt(Array, b_gpu)
                oa, oag = _byid(a_cpu), _byid(a_gpu_h)
                ob, obg = _byid(b_cpu), _byid(b_gpu_h)

                a_dvdt_scale = max(maximum(norm.(a_cpu.dvdt)), 1.0)
                b_dvdt_scale = max(maximum(norm.(b_cpu.dvdt)), 1.0)
                @test maximum(norm.(a_cpu.dvdt[oa] .- a_gpu_h.dvdt[oag])) < 1e-10 * a_dvdt_scale
                @test maximum(norm.(b_cpu.dvdt[ob] .- b_gpu_h.dvdt[obg])) < 1e-10 * b_dvdt_scale
                a_drho_scale = max(maximum(abs.(a_cpu.drhodt)), 1.0)
                b_drho_scale = max(maximum(abs.(b_cpu.drhodt)), 1.0)
                @test maximum(abs.(a_cpu.drhodt[oa] .- a_gpu_h.drhodt[oag])) < 1e-10 * a_drho_scale
                @test maximum(abs.(b_cpu.drhodt[ob] .- b_gpu_h.drhodt[obg])) < 1e-10 * b_drho_scale

                if pfn isa _ReverseOnlyTestPfn
                    # A pure WritesB() pfn must leave system_a untouched on the GPU path too.
                    @test all(==(zero(SVector{2,Float64})), a_gpu_h.dvdt)
                    @test all(==(0.0), a_gpu_h.drhodt)
                end
            end
        end

        @testset "full pipeline (sort+grid+sweep), FluidSolidPfn WritesBoth (fluid-solid): CPU oracle vs CUDA" begin
            # DambreakWall.jl's fluid/wall coupling shape. Same Kind-based
            # fix as the FluidPfn fluid-fluid case above, but FluidSolidPfn's
            # physics is asymmetric (fluid's own pressure used on both
            # sides), so it needed two distinct DeviceSystem{T,ND,Kind}
            # methods rather than one shared one — see PairwiseFunctors.jl.
            # A separate testset from the loop above since the fluid/solid
            # pairing needs two different system types, not two fluids.
            CUDA.allowscalar(false)
            rng = MersenneTwister(207)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length
            pfn = FluidSolidPfn(0.03, 0.0, h)

            a_cpu = _gpucuda_random_fluid(rng, 800, 2; L=2.0)
            b_cpu = _gpucuda_random_elastoplastic(rng, 600, 2; L=2.0)
            a_gpu = adapt(CUDABackend(), deepcopy(a_cpu))
            b_gpu = adapt(CUDABackend(), deepcopy(b_cpu))

            si_cpu = SystemInteraction(kernel, pfn, a_cpu, b_cpu; onesided=true)
            pa, ka_, sa = _gpucuda_sortbufs_cpu(a_cpu)
            sort_particles!(a_cpu, cutoff, pa, ka_, sa)
            pb, kb, sb = _gpucuda_sortbufs_cpu(b_cpu)
            sort_particles!(b_cpu, cutoff, pb, kb, sb)
            create_grid!(si_cpu)
            sweep!(si_cpu)

            si_gpu = SystemInteraction(kernel, pfn, a_gpu, b_gpu; onesided=true, ka=true)
            pag, kag, sag = _gpucuda_sortbufs_gpu(a_gpu)
            sort_particles!(a_gpu, cutoff, pag, kag, sag)
            pbg, kbg, sbg = _gpucuda_sortbufs_gpu(b_gpu)
            sort_particles!(b_gpu, cutoff, pbg, kbg, sbg)
            create_grid!(si_gpu)
            sweep!(si_gpu)

            a_gpu_h = adapt(Array, a_gpu)
            b_gpu_h = adapt(Array, b_gpu)
            oa, oag = _byid(a_cpu), _byid(a_gpu_h)
            ob, obg = _byid(b_cpu), _byid(b_gpu_h)

            a_dvdt_scale = max(maximum(norm.(a_cpu.dvdt)), 1.0)
            b_dvdt_scale = max(maximum(norm.(b_cpu.dvdt)), 1.0)
            @test maximum(norm.(a_cpu.dvdt[oa] .- a_gpu_h.dvdt[oag])) < 1e-10 * a_dvdt_scale
            @test maximum(norm.(b_cpu.dvdt[ob] .- b_gpu_h.dvdt[obg])) < 1e-10 * b_dvdt_scale
            a_drho_scale = max(maximum(abs.(a_cpu.drhodt)), 1.0)
            b_drho_scale = max(maximum(abs.(b_cpu.drhodt)), 1.0)
            @test maximum(abs.(a_cpu.drhodt[oa] .- a_gpu_h.drhodt[oag])) < 1e-10 * a_drho_scale
            @test maximum(abs.(b_cpu.drhodt[ob] .- b_gpu_h.drhodt[obg])) < 1e-10 * b_drho_scale
        end

        @testset "full pipeline (sort+grid+sweep), InterpolateFieldFn WritesB virtual target: CPU oracle vs CUDA" begin
            # Virtual is device_view-ready since item 5 (DeviceVirtualSystem)
            # — this is the first CUDA-hardware exercise of that path.
            CUDA.allowscalar(false)
            rng = MersenneTwister(206)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length
            pfn = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)

            src_cpu  = _gpucuda_random_fluid(rng, 800, 2; L=2.0)
            virt_src = _gpucuda_random_fluid(rng, 600, 2; L=2.0)
            virt_cpu = VirtualParticleSystem(virt_src, "virt", virt_src.n, 2, virt_src.mass, virt_src.c)
            _zero_interp_target!(virt_cpu, (:v, :rho))

            src_gpu  = adapt(CUDABackend(), deepcopy(src_cpu))
            virt_gpu = adapt(CUDABackend(), deepcopy(virt_cpu))

            si_cpu = SystemInteraction(kernel, pfn, src_cpu, virt_cpu; onesided=true)
            pa, ka_, sa = _gpucuda_sortbufs_cpu(src_cpu)
            sort_particles!(src_cpu, cutoff, pa, ka_, sa)
            pb, kb, sb = _gpucuda_sortbufs_cpu(virt_cpu)
            sort_particles!(virt_cpu, cutoff, pb, kb, sb)
            create_grid!(si_cpu)
            sweep!(si_cpu)

            si_gpu = SystemInteraction(kernel, pfn, src_gpu, virt_gpu; onesided=true, ka=true)
            pag, kag, sag = _gpucuda_sortbufs_gpu(src_gpu)
            sort_particles!(src_gpu, cutoff, pag, kag, sag)
            pbg, kbg, sbg = _gpucuda_sortbufs_gpu(virt_gpu)
            sort_particles!(virt_gpu, cutoff, pbg, kbg, sbg)
            create_grid!(si_gpu)
            sweep!(si_gpu)

            virt_gpu_h = adapt(Array, virt_gpu)
            # Not _byid(): that reads `id` via raw getfield, which works on a
            # bare FluidParticleSystem but VirtualParticleSystem doesn't own
            # an `id` field directly — it forwards to its wrapped `source`
            # via getproperty (Particles.jl), so getfield here throws
            # FieldError. Use getproperty (`.id`) instead.
            ov, ovg = sortperm(virt_cpu.id), sortperm(virt_gpu_h.id)
            v_scale = max(maximum(norm.(virt_cpu.v)), 1.0)
            rho_scale = max(maximum(abs.(virt_cpu.rho)), 1.0)
            @test maximum(norm.(virt_cpu.v[ov] .- virt_gpu_h.v[ovg])) < 1e-10 * v_scale
            @test maximum(abs.(virt_cpu.rho[ov] .- virt_gpu_h.rho[ovg])) < 1e-10 * rho_scale
            @test maximum(abs.(virt_cpu.w_sum[ov] .- virt_gpu_h.w_sum[ovg])) < 1e-10 * max(maximum(abs.(virt_cpu.w_sum)), 1.0)
        end

        @testset "state update (TaitEOSUpdater): CPU vs CUDA" begin
            CUDA.allowscalar(false)
            rng = MersenneTwister(203)
            fluid_cpu = FluidParticleSystem("fluid", 500, 2, 1.0, 10.0; state_updater=TaitEOSUpdater(1000.0))
            fluid_cpu.rho .= 1000.0 .+ 10 .* randn(rng, 500)
            fluid_gpu = adapt(CUDABackend(), deepcopy(fluid_cpu))

            update_state!(fluid_cpu, 1)
            update_state!(fluid_gpu, 1)

            p_gpu_h = Array(fluid_gpu.p)
            p_scale = max(maximum(abs, fluid_cpu.p), 1.0)
            @test maximum(abs.(fluid_cpu.p .- p_gpu_h)) < 1e-12 * p_scale
        end

        @testset "no scalar indexing anywhere in a real time_integrate! step path" begin
            # The decisive test: wraps several full steps (sort, grid, sweep,
            # state update, integrator axpy, print, save) in
            # CUDA.allowscalar(false) and requires no exception. Catches every
            # scalar-indexing landmine documented in the migration plan in
            # one shot, rather than one test per landmine.
            rng = MersenneTwister(204)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            dx = 0.06
            rho0 = 1000.0
            c_sound = 20.0

            nf = (8, 8)
            n_fluid = prod(nf)
            fluid = FluidParticleSystem("fluid", n_fluid, 2, rho0 * dx^2, c_sound;
                                        source_v = [0.0, -9.81],
                                        state_updater = TaitEOSUpdater(rho0))
            add_print_field!(fluid, :v)
            add_print_field!(fluid, :rho)
            k = 1
            for idx in Iterators.product(0:nf[1]-1, 0:nf[2]-1)
                fluid.x[k] = SVector((idx .+ 0.5) .* dx)
                k += 1
            end
            fill!(fluid.v, zero(SVector{2,Float64}))
            fluid.rho .= rho0
            update_state!(fluid, 1)

            nb = nf[1] + 4
            bnd = BasicParticleSystem("boundary", nb, 2, rho0 * dx^2, c_sound)
            k = 1
            for i in -2:nf[1]+1
                bnd.x[k] = SVector((i + 0.5) * dx, -0.5 * dx)
                k += 1
            end
            bnd.rho .= rho0
            fill!(bnd.v, zero(SVector{2,Float64}))

            fluid_gpu = adapt(CUDABackend(), fluid)
            bnd_gpu   = adapt(CUDABackend(), bnd)
            static_bnd_gpu = StaticBoundarySystem(bnd_gpu, dx)

            si_self = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid_gpu; onesided=true, ka=true)
            si_bnd  = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid_gpu, static_bnd_gpu; onesided=true, ka=true)
            integrator = LeapFrogTimeIntegrator([fluid_gpu, bnd_gpu], [si_self, si_bnd])

            mktempdir() do dir
                prefix = joinpath(dir, "gpu_scalar_check")
                CUDA.allowscalar(false)
                # An uncaught scalar-indexing error here fails this testset
                # directly (Test.jl reports it as an error), which is exactly
                # the assertion this test makes — no explicit try/catch needed.
                time_integrate!(integrator, 5, 2, 2, 0.1, prefix; print_timer=false)
            end
            @test true   # reached only if the run above completed without throwing

            @test all(isfinite, Array(fluid_gpu.rho))
            @test all(x -> all(isfinite, x), Array(fluid_gpu.x))
        end

    end

end
