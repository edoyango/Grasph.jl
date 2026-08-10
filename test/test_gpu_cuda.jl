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

            ghost_src = FluidParticleSystem("ghost_src", 10, 2, 1.0, 10.0)
            ghost = GhostParticleSystem(ghost_src, GhostCopier(:p); name="ghost")
            entry = GhostEntry(ghost, 0.1, (SVector(1.0, 0.0), SVector(0.0, 0.0)))
            generate_ghosts!(entry)
            ghost_gpu = adapt(CUDABackend(), entry).ghost
            dvg = Grasph.device_view(ghost_gpu)
            @test isbitstype(typeof(cudaconvert(dvg)))

            # Regression (Trapdoor.jl's two-virtuals-share-one-source shape,
            # item 10): VirtualParticleSystem's keyword constructor used to
            # hardcode `w_sum = zeros(T, n)` regardless of the source's own
            # array type. Building a *second* virtual directly around an
            # already GPU-resident source (the correct idiom for keeping two
            # virtual systems aliased to the same live physical state — see
            # VirtualParticleSystem's docstring) silently produced a
            # mixed-backend struct (source on CuArray, w_sum on Vector) that
            # is not isbits, so `cudaconvert` succeeds (it only converts
            # CuArrays it finds) but the *kernel launch* itself fails to
            # compile — a much worse failure mode than a caught type error,
            # since it only surfaces the first time that struct actually
            # reaches a `@kernel` call. Fixed by deriving w_sum's array type
            # from the source via `similar`, mirroring how RK4's sort buffers
            # were already fixed to do the same thing (item 9).
            virt_src = FluidParticleSystem("virt_src", 10, 2, 1.0, 10.0)
            virt1 = VirtualParticleSystem(virt_src, "virt1", 10, 2, 1.0, 10.0)
            virt1_gpu = adapt(CUDABackend(), virt1)
            shared_src_gpu = getfield(virt1_gpu, :source)
            virt2_gpu = VirtualParticleSystem(shared_src_gpu, "virt2", 10, 2, 1.0, 10.0)
            @test getfield(virt2_gpu, :w_sum) isa CuArray
            dvv = Grasph.device_view(virt2_gpu)
            @test isbitstype(typeof(cudaconvert(dvv)))
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

        @testset "full pipeline (sort+grid+sweep), ColouredKA self+coupled: CPU-coloured oracle vs CUDA" begin
            # ColouredKA (docs/gpu-migration-plan.md's coloured-GPU
            # benchmarking spike, Backend.jl) reuses the *two-sided mutating*
            # pfn contract via `mode=ColouredKA()` (not `onesided=true`), one
            # kernel launch per colour (KAKernels.jl). Unlike OnesidedKA vs.
            # ColouredCPU (different pair-visitation order — bounded only by
            # FMA-contraction-level rtol below), ColouredKA visits pairs in
            # the identical colour-by-colour, once-per-pair order as
            # ColouredCPU, so a real bug here would show up as more than
            # noise, not as amplified chaos.
            CUDA.allowscalar(false)
            rng = MersenneTwister(203)
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
            si_self_cpu = SystemInteraction(kernel, pfn, fluid_cpu; mode=Grasph.ColouredCPU())
            si_cpl_cpu  = SystemInteraction(kernel, pfn, fluid_cpu, static_bnd_cpu; mode=Grasph.ColouredCPU())
            create_grid!(si_self_cpu); create_grid!(si_cpl_cpu)
            sweep!(si_self_cpu); sweep!(si_cpl_cpu)

            pbbg, kbbg, scbg = _gpucuda_sortbufs_gpu(bnd_gpu)
            sort_particles!(bnd_gpu, cutoff, pbbg, kbbg, scbg)
            pb2g, kb2g, sc2g = _gpucuda_sortbufs_gpu(fluid_gpu)
            sort_particles!(fluid_gpu, cutoff, pb2g, kb2g, sc2g)
            si_self_gpu = SystemInteraction(kernel, pfn, fluid_gpu; mode=Grasph.ColouredKA())
            si_cpl_gpu  = SystemInteraction(kernel, pfn, fluid_gpu, static_bnd_gpu; mode=Grasph.ColouredKA())
            create_grid!(si_self_gpu); create_grid!(si_cpl_gpu)
            sweep!(si_self_gpu); sweep!(si_cpl_gpu)

            @test Array(si_self_gpu._cell_start) == si_self_cpu._cell_start

            fluid_gpu_h = adapt(Array, fluid_gpu)
            oc, og = _byid(fluid_cpu), _byid(fluid_gpu_h)
            dvdt_scale = max(maximum(norm.(fluid_cpu.dvdt)), 1.0)
            drhodt_scale = max(maximum(abs.(fluid_cpu.drhodt)), 1.0)
            @test maximum(norm.(fluid_cpu.dvdt[oc] .- fluid_gpu_h.dvdt[og])) < 1e-10 * dvdt_scale
            @test maximum(abs.(fluid_cpu.drhodt[oc] .- fluid_gpu_h.drhodt[og])) < 1e-10 * drhodt_scale
        end

        @testset "full pipeline (sort+grid+sweep), verlet_skin > 0 (padded grid), self+coupled onesided: CPU oracle vs CUDA" begin
            # docs/gpu-migration-plan.md deferred item 1: create_grid!(si, skin)
            # widens the cell-partitioning pitch (si._grid_cutoff) while
            # si._cell_size (the physical cutoff feeding cutoff_sq) stays
            # unchanged — see Interaction.jl/KAKernels.jl. The KA onesided
            # kernels now take two scalar cutoff arguments instead of one
            # (grid pitch for _cell_1idx, physical cutoff_sq for the pairwise
            # filter); this is the only tier that exercises that arg split on
            # a real GPU launch, not just KA.CPU().
            CUDA.allowscalar(false)
            rng = MersenneTwister(211)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length
            skin = 0.3 * cutoff
            pfn = FluidPfn(0.03, 0.0, h)

            n_fluid, n_bnd = 1500, 600
            fluid_cpu = _gpucuda_random_fluid(rng, n_fluid, 2; L=3.0)
            bnd_cpu   = _gpucuda_random_boundary(rng, n_bnd, 2; L=3.0)
            static_bnd_cpu = StaticBoundarySystem(bnd_cpu, 0.03)

            fluid_gpu = adapt(CUDABackend(), deepcopy(fluid_cpu))
            bnd_gpu   = adapt(CUDABackend(), deepcopy(bnd_cpu))
            static_bnd_gpu = StaticBoundarySystem(bnd_gpu, 0.03)

            # Sort with the same padded cutoff time_integrate! would use
            # (sort_cutoff = 2h + verlet_skin), so the CSR build's "already
            # sorted per this cell assignment" invariant holds under the
            # padded grid too — see TimeIntegration.jl's verlet_skin gate.
            pbb, kbb, scb = _gpucuda_sortbufs_cpu(bnd_cpu)
            sort_particles!(bnd_cpu, cutoff + skin, pbb, kbb, scb)
            pb2, kb2, sc2 = _gpucuda_sortbufs_cpu(fluid_cpu)
            sort_particles!(fluid_cpu, cutoff + skin, pb2, kb2, sc2)
            si_self_cpu = SystemInteraction(kernel, pfn, fluid_cpu; onesided=true)
            si_cpl_cpu  = SystemInteraction(kernel, pfn, fluid_cpu, static_bnd_cpu; onesided=true)
            create_grid!(si_self_cpu, skin); create_grid!(si_cpl_cpu, skin)
            sweep!(si_self_cpu); sweep!(si_cpl_cpu)

            pbbg, kbbg, scbg = _gpucuda_sortbufs_gpu(bnd_gpu)
            sort_particles!(bnd_gpu, cutoff + skin, pbbg, kbbg, scbg)
            pb2g, kb2g, sc2g = _gpucuda_sortbufs_gpu(fluid_gpu)
            sort_particles!(fluid_gpu, cutoff + skin, pb2g, kb2g, sc2g)
            si_self_gpu = SystemInteraction(kernel, pfn, fluid_gpu; onesided=true, ka=true)
            si_cpl_gpu  = SystemInteraction(kernel, pfn, fluid_gpu, static_bnd_gpu; onesided=true, ka=true)
            create_grid!(si_self_gpu, skin); create_grid!(si_cpl_gpu, skin)
            sweep!(si_self_gpu); sweep!(si_cpl_gpu)

            @test si_self_gpu._grid_cutoff[] ≈ cutoff + skin
            @test si_self_cpu._grid_cutoff[] ≈ cutoff + skin
            @test Array(si_self_gpu._cell_start) == si_self_cpu._cell_start
            @test Array(si_self_gpu._mingridx) == Vector(si_self_cpu._mingridx)
            @test Array(si_self_gpu._ngridx) == Vector(si_self_cpu._ngridx)
            @test Array(si_cpl_gpu._cell_start) == si_cpl_cpu._cell_start

            fluid_gpu_h = adapt(Array, fluid_gpu)
            oc, og = _byid(fluid_cpu), _byid(fluid_gpu_h)
            dvdt_scale = max(maximum(norm.(fluid_cpu.dvdt)), 1.0)
            drhodt_scale = max(maximum(abs.(fluid_cpu.drhodt)), 1.0)
            @test maximum(norm.(fluid_cpu.dvdt[oc] .- fluid_gpu_h.dvdt[og])) < 1e-10 * dvdt_scale
            @test maximum(abs.(fluid_cpu.drhodt[oc] .- fluid_gpu_h.drhodt[og])) < 1e-10 * drhodt_scale
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

        @testset "generate_ghosts!/update_ghost_kinematics!/GhostCopier GPU vs CPU oracle (item 7)" begin
            # The core new machinery item 7 adds: generate_ghosts!'s GPU path
            # (flag + cumsum exclusive-scan + compaction into
            # capacity-preallocated buffers, GhostParticles.jl/KAKernels.jl)
            # replacing the CPU count-then-cursor algorithm, which cannot
            # port to a CuArray directly (a per-step resize! to the exact
            # count is a full reallocation every step). Also exercises the
            # GPU kernel twins of update_ghost_kinematics! and GhostCopier's
            # field-copy, which are new scalar-loop-on-GPU hazards item 7
            # introduces (both would violate CUDA.allowscalar(false) if left
            # as the original CPU-only loops).
            CUDA.allowscalar(false)
            rng = MersenneTwister(208)

            boundaries = (
                (SVector(1.0, 0.0), SVector(0.0, 0.0)),
                (SVector(-1.0, 0.0), SVector(0.4, 0.0)),
                (SVector(0.0, 1.0), SVector(0.0, 0.0)),
                (SVector(0.0, -1.0), SVector(0.0, 0.4)),
            )
            _build_ghost(fluid) = begin
                ghost = GhostParticleSystem(fluid, GhostCopier(:p); name="ghost[$(fluid.name)]")
                GhostEntry(ghost, 0.15, boundaries...)
            end

            fluid_cpu = _gpucuda_random_fluid(rng, 500, 2; L=0.4)
            entry_cpu = _build_ghost(fluid_cpu)

            entry_gpu = adapt(CUDABackend(), entry_cpu)
            ghost_gpu = entry_gpu.ghost
            fluid_gpu = getfield(ghost_gpu, :source)
            ghost_cpu = entry_cpu.ghost

            function _compare_ghosts()
                n = ghost_cpu.n
                @test ghost_gpu.n == n
                cap = length(getfield(ghost_gpu, :x))
                @test n <= cap
                x_h   = Array(view(getfield(ghost_gpu, :x), 1:n))
                io_h  = Array(view(getfield(ghost_gpu, :idx_original), 1:n))
                ib_h  = Array(view(getfield(ghost_gpu, :idx_boundary), 1:n))
                nrm_h = Array(view(getfield(ghost_gpu, :normals), 1:n))
                @test x_h   == getfield(ghost_cpu, :x)[1:n]
                @test io_h  == getfield(ghost_cpu, :idx_original)[1:n]
                @test ib_h  == getfield(ghost_cpu, :idx_boundary)[1:n]
                @test nrm_h == getfield(ghost_cpu, :normals)[1:n]
                return cap
            end

            generate_ghosts!(entry_cpu)
            generate_ghosts!(entry_gpu)
            _compare_ghosts()

            update_ghost_kinematics!(entry_cpu)
            update_ghost_kinematics!(entry_gpu)
            update_ghost!(entry_cpu, 1)
            update_ghost!(entry_gpu, 1)
            n0 = ghost_cpu.n
            v_h   = Array(view(getfield(ghost_gpu, :v), 1:n0))
            rho_h = Array(view(getfield(ghost_gpu, :rho), 1:n0))
            p_h   = Array(view(ghost_gpu.p, 1:n0))
            @test maximum(norm.(v_h .- getfield(ghost_cpu, :v)[1:n0])) < 1e-10
            @test maximum(abs.(rho_h .- getfield(ghost_cpu, :rho)[1:n0])) < 1e-10
            @test maximum(abs.(p_h .- ghost_cpu.p[1:n0])) < 1e-10

            # Grow: pull particles toward every wall at once -> more ghosts
            # qualify -> capacity must grow (both the owned arrays AND the
            # :p extras array, which starts at length 0 independently of
            # x/v/rho/etc. — the exact bug the capacity-growth fix here
            # covers).
            for i in 1:fluid_cpu.n
                fluid_cpu.x[i] = SVector(0.2, 0.2) .+ (fluid_cpu.x[i] .- SVector(0.2, 0.2)) .* 0.3
            end
            copyto!(getfield(fluid_gpu, :x), CuArray(fluid_cpu.x))
            generate_ghosts!(entry_cpu)
            generate_ghosts!(entry_gpu)
            cap_grown = _compare_ghosts()
            @test cap_grown >= ghost_cpu.n

            # Shrink: push particles toward the interior, away from every
            # wall -> fewer ghosts qualify, but capacity (grow-only) must NOT
            # shrink -> ghost.n < capacity from here on, the exact
            # capacity-vs-logical-count regime this item's _bbox/
            # _populate_cells_sorted! explicit-`n` fix (Interaction.jl)
            # exists for (an earlier version of this fix, keyed only off
            # length(ghost.x), silently missed the :p extras array's smaller
            # starting capacity here and corrupted GPU memory).
            for i in 1:fluid_cpu.n
                fluid_cpu.x[i] = SVector(0.2, 0.2) .+ (fluid_cpu.x[i] .- SVector(0.2, 0.2)) .* 20.0 .+ SVector(0.15, 0.0)
            end
            copyto!(getfield(fluid_gpu, :x), CuArray(fluid_cpu.x))
            generate_ghosts!(entry_cpu)
            generate_ghosts!(entry_gpu)
            cap_after_shrink = _compare_ghosts()
            @test cap_after_shrink == cap_grown          # capacity never shrinks
            @test ghost_cpu.n < cap_after_shrink          # genuinely in the capacity > n regime
        end

        @testset "GhostCopier HouseholderReflect mode: GPU vs CPU oracle (item 7)" begin
            # _ghost_copy_field_kernel! is launched identically regardless of
            # `mode` (GhostParticles.jl's _copy_fields_ka!), so `mode = nothing`
            # (every other ghost GPU test above) and `mode =
            # HouseholderReflect()` are two distinct kernel specializations —
            # the reflection arithmetic runs entirely inside the kernel body
            # via _apply_mode, unlike the flag/scatter kernels' simpler
            # dot-product predicate. This is the first real-hardware exercise
            # of that specialization.
            CUDA.allowscalar(false)
            rng = MersenneTwister(211)
            ep_cpu = _gpucuda_random_elastoplastic(rng, 60, 2; L=0.4)
            ep_cpu.stress .= [SVector(1.0 + 0.1i, 2.0 - 0.05i, 0.3, 0.7 + 0.02i) for i in 1:60]

            ghost_cpu = GhostParticleSystem(ep_cpu, GhostCopier(:stress => HouseholderReflect()); name="ghost[ep]")
            entry_cpu = GhostEntry(ghost_cpu, 0.15, (SVector(1.0, 0.0), SVector(0.0, 0.0)))
            entry_gpu = adapt(CUDABackend(), entry_cpu)
            ghost_gpu = entry_gpu.ghost

            generate_ghosts!(entry_cpu)
            generate_ghosts!(entry_gpu)
            n = ghost_cpu.n
            @test n > 0
            @test ghost_gpu.n == n

            update_ghost!(entry_cpu, 1)
            update_ghost!(entry_gpu, 1)

            stress_cpu = ghost_cpu.stress[1:n]
            stress_gpu_h = Array(view(ghost_gpu.stress, 1:n))
            @test maximum(norm.(stress_cpu .- stress_gpu_h)) < 1e-10
        end

        @testset "generate_ghosts! 3D: GPU vs CPU oracle (item 7)" begin
            # Every other ghost GPU test above is 2D-only; the flag/scatter/
            # kinematics kernels are dimension-generic via SVector{ND,T}, but
            # ND=3 kernel compilation had never actually run on hardware.
            CUDA.allowscalar(false)
            rng = MersenneTwister(212)
            fluid_cpu = _gpucuda_random_fluid(rng, 400, 3; L=0.4)
            ghost_cpu = GhostParticleSystem(fluid_cpu, GhostCopier(:p); name="ghost3d[fluid]")
            entry_cpu = GhostEntry(ghost_cpu, 0.1,
                (SVector(1.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)),
                (SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 0.0)),
                (SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, 0.0)),
            )
            entry_gpu = adapt(CUDABackend(), entry_cpu)
            ghost_gpu = entry_gpu.ghost

            generate_ghosts!(entry_cpu)
            generate_ghosts!(entry_gpu)
            n = ghost_cpu.n
            @test n > 0
            @test ghost_gpu.n == n
            @test Array(view(getfield(ghost_gpu, :x), 1:n)) == getfield(ghost_cpu, :x)[1:n]
            @test Array(view(getfield(ghost_gpu, :idx_original), 1:n)) == getfield(ghost_cpu, :idx_original)[1:n]

            update_ghost_kinematics!(entry_cpu)
            update_ghost_kinematics!(entry_gpu)
            update_ghost!(entry_cpu, 1)
            update_ghost!(entry_gpu, 1)
            @test maximum(norm.(Array(view(getfield(ghost_gpu, :v), 1:n)) .- getfield(ghost_cpu, :v)[1:n])) < 1e-10
            @test maximum(abs.(Array(view(ghost_gpu.p, 1:n)) .- ghost_cpu.p[1:n])) < 1e-10
        end

        @testset "generate_ghosts! NB=8 (bubble3.jl's real wall+corner shape): GPU vs CPU oracle (item 7)" begin
            # Every other ghost GPU test above uses NB<=4; the actual
            # production shape (bubble3.jl's boundary_ghost, mirrored by
            # test_onesided_sweep.jl's _xsph_ghost_fluid/_xsph_ghost_setup!)
            # uses 4 walls + 4 corners = NB=8. The flattened (boundary,
            # particle) linear-index arithmetic in _ghost_flag_kernel!/
            # _ghost_scatter_kernel! is NB-generic, but this is the first
            # real-hardware exercise at the actual NB this codebase uses.
            CUDA.allowscalar(false)
            rng = MersenneTwister(213)
            fluid_cpu = _xsph_ghost_fluid(rng, 8, 8, 0.05)
            Lx = Ly = 0.4
            boundary_cutoff = 3 * 0.05
            ghost_cpu = GhostParticleSystem(fluid_cpu, GhostCopier(:p); name="ghost8[fluid]")
            entry_cpu = GhostEntry(ghost_cpu, boundary_cutoff,
                (SVector( 1.0,  0.0),            SVector(0.0, 0.0)),
                (SVector(-1.0,  0.0),            SVector(Lx,  0.0)),
                (SVector( 0.0,  1.0),            SVector(0.0, 0.0)),
                (SVector( 0.0, -1.0),            SVector(0.0, Ly)),
                (SVector( 1.0,  1.0)/sqrt(2.0),  SVector(0.0, 0.0)),
                (SVector(-1.0,  1.0)/sqrt(2.0),  SVector(Lx,  0.0)),
                (SVector( 1.0, -1.0)/sqrt(2.0),  SVector(0.0, Ly)),
                (SVector(-1.0, -1.0)/sqrt(2.0),  SVector(Lx,  Ly)),
            )
            entry_gpu = adapt(CUDABackend(), entry_cpu)
            ghost_gpu = entry_gpu.ghost

            generate_ghosts!(entry_cpu)
            generate_ghosts!(entry_gpu)
            n = ghost_cpu.n
            @test n > 0
            @test ghost_gpu.n == n
            # x uses a relative tolerance, not ==: the 4 corner boundaries'
            # diagonal normals (SVector(±1,±1)/sqrt(2)) involve genuine
            # floating-point products where NVPTX's FMA contraction can
            # differ from the host by 1 ulp — the same well-documented
            # CPU/GPU float noise this codebase tolerances everywhere else
            # (the 4 axis-aligned wall normals above have no such risk,
            # which is why every other ghost GPU test above safely uses ==).
            # idx_original/idx_boundary stay ==: integer, no rounding.
            x_cpu = getfield(ghost_cpu, :x)[1:n]
            x_gpu_h = Array(view(getfield(ghost_gpu, :x), 1:n))
            @test maximum(norm.(x_cpu .- x_gpu_h)) < 1e-12 * max(maximum(norm.(x_cpu)), 1.0)
            @test Array(view(getfield(ghost_gpu, :idx_original), 1:n)) == getfield(ghost_cpu, :idx_original)[1:n]
            @test Array(view(getfield(ghost_gpu, :idx_boundary), 1:n)) == getfield(ghost_cpu, :idx_boundary)[1:n]
            @test extrema(Array(view(getfield(ghost_gpu, :idx_boundary), 1:n))) == extrema(getfield(ghost_cpu, :idx_boundary)[1:n])
        end

        @testset "full pipeline (sort+grid+sweep), fluid<->GhostParticleSystem (item 7): CPU oracle vs CUDA" begin
            # End-to-end version of the testset above: sort + grid + sweep
            # with a real ghost, deliberately run AFTER the ghost count has
            # shrunk below a previously-grown capacity — the scenario that
            # would silently pull stale capacity-slot data into the CSR grid
            # if _bbox/_populate_cells_sorted! (Interaction.jl) still derived
            # their particle count from `length(x)` instead of an explicit
            # `n`. FluidPfn's ghost-coupled pfn_contribution method
            # (PairwiseFunctors.jl) predates item 7; only device_view(ghost)
            # is new, making `ka=true` reachable here for the first time.
            CUDA.allowscalar(false)
            rng = MersenneTwister(209)
            h = 0.04
            kernel = CubicSplineKernel(h; ndims=2)
            cutoff = kernel.interaction_length
            pfn = FluidPfn(0.03, 0.0, h)

            boundaries = (
                (SVector(1.0, 0.0), SVector(0.0, 0.0)),
                (SVector(-1.0, 0.0), SVector(0.4, 0.0)),
                (SVector(0.0, 1.0), SVector(0.0, 0.0)),
                (SVector(0.0, -1.0), SVector(0.0, 0.4)),
            )
            _build_ghost2(fluid) = begin
                ghost = GhostParticleSystem(fluid, GhostCopier(:p); name="ghost[$(fluid.name)]")
                GhostEntry(ghost, 3h, boundaries...)
            end

            fluid_cpu = _gpucuda_random_fluid(rng, 500, 2; L=0.4)
            entry_cpu = _build_ghost2(fluid_cpu)
            entry_gpu = adapt(CUDABackend(), entry_cpu)
            ghost_gpu = entry_gpu.ghost
            fluid_gpu = getfield(ghost_gpu, :source)
            ghost_cpu = entry_cpu.ghost

            # Grow then shrink (same recipe as above) so both sides enter the
            # sweep with ghost.n < capacity on the GPU.
            for i in 1:fluid_cpu.n
                fluid_cpu.x[i] = SVector(0.2, 0.2) .+ (fluid_cpu.x[i] .- SVector(0.2, 0.2)) .* 0.3
            end
            copyto!(getfield(fluid_gpu, :x), CuArray(fluid_cpu.x))
            generate_ghosts!(entry_cpu); generate_ghosts!(entry_gpu)
            grown_cap = length(getfield(ghost_gpu, :x))

            for i in 1:fluid_cpu.n
                fluid_cpu.x[i] = SVector(0.2, 0.2) .+ (fluid_cpu.x[i] .- SVector(0.2, 0.2)) .* 20.0 .+ SVector(0.15, 0.0)
            end
            copyto!(getfield(fluid_gpu, :x), CuArray(fluid_cpu.x))

            si_cpu = SystemInteraction(kernel, pfn, fluid_cpu, ghost_cpu; onesided=true)
            pb, kb, sc = _gpucuda_sortbufs_cpu(fluid_cpu)
            sort_particles!(fluid_cpu, cutoff, pb, kb, sc)
            generate_ghosts!(entry_cpu)
            gpb, gkb, gsc = _gpucuda_sortbufs_cpu(ghost_cpu)
            sort_particles!(ghost_cpu, cutoff, gpb, gkb, gsc)
            update_ghost_kinematics!(entry_cpu)
            update_ghost!(entry_cpu, 1)
            create_grid!(si_cpu)
            sweep!(si_cpu)

            si_gpu = SystemInteraction(kernel, pfn, fluid_gpu, ghost_gpu; onesided=true, ka=true)
            pbg, kbg, scg = _gpucuda_sortbufs_gpu(fluid_gpu)
            sort_particles!(fluid_gpu, cutoff, pbg, kbg, scg)
            generate_ghosts!(entry_gpu)
            @test length(getfield(ghost_gpu, :x)) == grown_cap   # still in the capacity > n regime
            @test ghost_gpu.n < grown_cap
            gpbg, gkbg, gscg = _gpucuda_sortbufs_gpu(ghost_gpu)
            sort_particles!(ghost_gpu, cutoff, gpbg, gkbg, gscg)
            update_ghost_kinematics!(entry_gpu)
            update_ghost!(entry_gpu, 1)
            create_grid!(si_gpu)
            sweep!(si_gpu)

            fluid_gpu_h = adapt(Array, fluid_gpu)
            oc, og = _byid(fluid_cpu), _byid(fluid_gpu_h)
            dvdt_scale = max(maximum(norm.(fluid_cpu.dvdt)), 1.0)
            drhodt_scale = max(maximum(abs.(fluid_cpu.drhodt)), 1.0)
            @test maximum(norm.(fluid_cpu.dvdt[oc] .- fluid_gpu_h.dvdt[og])) < 1e-9 * dvdt_scale
            @test maximum(abs.(fluid_cpu.drhodt[oc] .- fluid_gpu_h.drhodt[og])) < 1e-9 * drhodt_scale
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

        @testset "RK4TimeIntegrator on real CUDA hardware (item 9 buffer-type fix)" begin
            # bubble.jl/bubble2.jl/bubble3.jl's shape: RK4TimeIntegrator plus a
            # self-referencing ghost, reusing the _bubble_like/_bubble_integrator
            # fixtures from test_onesided_integration_bubble.jl (included
            # earlier in runtests.jl, same top-level @testset scope).
            #
            # Before item 9, RK4's time_integrate! hardcoded Vector for its
            # sort_perm_buf/sort_key_buf scratch buffers regardless of the
            # systems' actual array type — LeapFrog's own loop already derived
            # them via `similar(first(sys).x, ...)`, RK4's didn't. On a real
            # CUDA-resident RK4 run this mismatched CPU-buffer/GPU-array pair
            # would hit scalar indexing inside sort_particles! the first time
            # it ran. This can only be exercised on a real non-CPU backend —
            # a KA.CPU() test can't distinguish `Vector` from `similar(x,...)`
            # when `x` is itself a `Vector`.
            CUDA.allowscalar(false)
            sys_cpu = _bubble_like()
            rk_cpu  = _bubble_integrator(sys_cpu; onesided=true)

            sys_gpu = _bubble_like()
            fluid_Y_gpu = adapt(CUDABackend(), sys_gpu.fluid_Y)
            # boundary_ghost is self-referencing (boundary_ghost.source ===
            # fluid_X); adapt the GhostEntry as one unit and pull the
            # canonical GPU-resident fluid_X back out of it (see
            # GhostParticleSystem's docstring) rather than adapting fluid_X
            # a second time.
            ghost_entry_gpu    = adapt(CUDABackend(), sys_gpu.boundary_ghost_entry)
            boundary_ghost_gpu = ghost_entry_gpu.ghost
            fluid_X_gpu        = getfield(boundary_ghost_gpu, :source)

            kernel = sys_gpu.kernel
            art_visc_alpha, art_visc_beta, h_sph = sys_gpu.art_visc_alpha, sys_gpu.art_visc_beta, sys_gpu.h_sph
            fluid_X_interaction = SystemInteraction(
                kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_X_gpu; onesided=true, ka=true)
            fluid_Y_interaction = SystemInteraction(
                kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_Y_gpu; onesided=true, ka=true)
            fluid_XY_interaction = SystemInteraction(
                kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_Y_gpu, fluid_X_gpu; onesided=true, ka=true)
            fluid_boundary_interaction = SystemInteraction(
                kernel, FluidPfn(art_visc_alpha, art_visc_beta, h_sph), fluid_X_gpu, boundary_ghost_gpu; onesided=true, ka=true)

            rk_gpu = RK4TimeIntegrator(
                [fluid_X_gpu, fluid_Y_gpu],
                [fluid_X_interaction, fluid_Y_interaction, fluid_XY_interaction, fluid_boundary_interaction];
                ghosts = [ghost_entry_gpu],
            )

            nsteps = 8
            time_integrate!(rk_cpu, nsteps, nsteps + 1, nsteps + 1, 1.5, nothing; print_timer=false)
            time_integrate!(rk_gpu, nsteps, nsteps + 1, nsteps + 1, 1.5, nothing; print_timer=false)

            fX_gpu_h = adapt(Array, fluid_X_gpu)
            fY_gpu_h = adapt(Array, fluid_Y_gpu)
            for (a, b) in ((sys_cpu.fluid_X, fX_gpu_h), (sys_cpu.fluid_Y, fY_gpu_h))
                oa, ob = _byid(a), _byid(b)
                @test !any(v -> any(isnan, v), b.x)
                @test !any(isnan, b.rho)
                x_scale   = max(maximum(norm.(a.x)), 1.0)
                rho_scale = max(maximum(abs.(a.rho)), 1.0)
                @test maximum(norm.(a.x[oa] .- b.x[ob])) < 1e-6 * x_scale
                @test maximum(abs.(a.rho[oa] .- b.rho[ob])) < 1e-6 * rho_scale
            end
        end

        @testset "VirtualParticleSystem position advance (nonzero prescribed_v) on CUDA (item 9)" begin
            # Trapdoor.jl's trapdoor_moving_virt shape: a VirtualParticleSystem
            # with nonzero prescribed_v, driven through a real
            # LeapFrogTimeIntegrator loop. Before item 9,
            # _update_virtual_positions! was a raw scalar `vps.x[i] += pv*dt`
            # loop — an unconditional CUDA.allowscalar(false) violation the
            # instant prescribed_v != 0 on a GPU-resident system, regardless
            # of whether the virtual system couples into any interaction.
            CUDA.allowscalar(false)
            rng = MersenneTwister(220)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)
            pv = SVector(0.3, -0.2)

            fluid_cpu    = _gpucuda_random_fluid(rng, 200, 2; L=1.0)
            virt_src_cpu = _gpucuda_random_boundary(rng, 60, 2; L=1.0)
            x0 = copy(virt_src_cpu.x)   # snapshot before any sort permutes it
            virt_cpu = VirtualParticleSystem(virt_src_cpu, "virt", virt_src_cpu.n, 2, 1.0, 10.0;
                zero_fields=(:w_sum,), prescribed_v=pv)

            fluid_gpu = adapt(CUDABackend(), deepcopy(fluid_cpu))
            virt_gpu  = adapt(CUDABackend(), deepcopy(virt_cpu))

            si_cpu = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid_cpu; onesided=true)
            si_gpu = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid_gpu; onesided=true, ka=true)

            lf_cpu = LeapFrogTimeIntegrator([fluid_cpu], [si_cpu]; virtual_systems=(virt_cpu,))
            lf_gpu = LeapFrogTimeIntegrator([fluid_gpu], [si_gpu]; virtual_systems=(virt_gpu,))

            nsteps = 5
            time_integrate!(lf_cpu, nsteps, nsteps + 1, nsteps + 1, 0.1, nothing; print_timer=false)
            time_integrate!(lf_gpu, nsteps, nsteps + 1, nsteps + 1, 0.1, nothing; print_timer=false)

            dt = 0.1 * h / lf_cpu.c
            expected_disp = nsteps * dt * pv

            src_cpu = getfield(virt_cpu, :source)
            disp_cpu = src_cpu.x .- x0[getfield(src_cpu, :id)]
            @test maximum(norm.(disp_cpu .- Ref(expected_disp))) < 1e-10

            src_gpu_h = adapt(Array, getfield(virt_gpu, :source))
            @test all(isfinite, reinterpret(Float64, src_gpu_h.x))
            disp_gpu = src_gpu_h.x .- x0[getfield(src_gpu_h, :id)]
            @test maximum(norm.(disp_gpu .- Ref(expected_disp))) < 1e-8
        end

        @testset "ProbeParticleSystem measurement (mirror_target + NeighborCountFn) on CUDA (item 9)" begin
            # CantileverBeam.jl's shape: a probe mirroring a real system's
            # positions each measurement, summed via NeighborCountFn. Before
            # item 9, ProbeParticleSystem hardcoded Vector for x/id/w_sum/
            # extras (no Adapt.adapt_structure at all) and _measure_probes!'s
            # mirror step was a raw scalar `x[src_id[i]] = src_x[i]` loop —
            # either alone would make this combination unreachable under
            # ka=true.
            CUDA.allowscalar(false)
            rng = MersenneTwister(221)
            h = 0.08
            kernel = CubicSplineKernel(h; ndims=2)

            fluid_cpu = _gpucuda_random_fluid(rng, 200, 2; L=1.0)
            probe_cpu = ProbeParticleSystem("probe", fluid_cpu; extras=(nbr=zeros(Int, fluid_cpu.n),))

            # probe_cpu is self-referencing (probe_cpu.mirror_target ===
            # fluid_cpu); adapt the probe as one unit and pull the canonical
            # GPU-resident fluid back out of it (see ProbeParticleSystem's
            # docstring) rather than adapting fluid_cpu a second time.
            probe_gpu = adapt(CUDABackend(), deepcopy(probe_cpu))
            fluid_gpu = getfield(probe_gpu, :mirror_target)

            si_cpu = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid_cpu; onesided=true)
            si_gpu = SystemInteraction(kernel, FluidPfn(0.03, 0.0, h), fluid_gpu; onesided=true, ka=true)
            pi_cpu = SystemInteraction(kernel, NeighborCountFn(:nbr), fluid_cpu, probe_cpu; onesided=true)
            pi_gpu = SystemInteraction(kernel, NeighborCountFn(:nbr), fluid_gpu, probe_gpu; onesided=true, ka=true)

            lf_cpu = LeapFrogTimeIntegrator([fluid_cpu], [si_cpu]; probes=(probe_cpu,), probe_interactions=(pi_cpu,))
            lf_gpu = LeapFrogTimeIntegrator([fluid_gpu], [si_gpu]; probes=(probe_gpu,), probe_interactions=(pi_gpu,))

            mktempdir() do dir
                time_integrate!(lf_cpu, 3, 4, 3, 0.1, joinpath(dir, "cpu"); print_timer=false)
                time_integrate!(lf_gpu, 3, 4, 3, 0.1, joinpath(dir, "gpu"); print_timer=false)
            end

            nbr_cpu = probe_cpu.nbr
            nbr_gpu = Array(probe_gpu.nbr)
            @test any(!=(0), nbr_cpu)   # sanity: neighbor counting actually happened
            @test nbr_cpu == nbr_gpu
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
