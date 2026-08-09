using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm
using Random

# ---------------------------------------------------------------------------
# DeviceViews.jl coverage for gpu-migration-plan.md "Next steps" item 5:
# extending device_view/_device_fields to StressParticleSystem,
# ElastoPlasticParticleSystem, DynamicBoundarySystem, and
# VirtualParticleSystem.
#
# ProbeParticleSystem/GhostParticleSystem are deliberately NOT covered here
# — both hardcode `Vector` for their per-particle arrays (not the
# array-type-generic parameter every other system uses), which blocks Adapt
# entirely regardless of device_view; that is a separate, deeper problem
# already tracked as its own item in docs/gpu-migration-plan.md.
#
# VirtualParticleSystem is a concrete struct that owns a non-isbits `name`
# field directly, so (unlike Static/DynamicBoundarySystem) its own type can't
# be rebuilt around a device-viewed inner system. Its device_view flattens
# into a new `DeviceVirtualSystem`, which only dispatches into the existing
# pfn_contribution methods (all written against `VirtualParticleSystem{T,ND}`)
# because those signatures were widened to the new `AbstractVirtualParticleSystem`
# common supertype (see Particles.jl/PairwiseFunctors.jl). The tests below
# exist specifically to catch a dispatch regression from that widening — both
# a missing-method failure (device_view falls through to nothing) and an
# over-broad one (some unrelated real system starts matching a ghost/virtual-
# coupled method it shouldn't).
#
# Reuses fixtures from test_onesided_sweep.jl (_random_fluid, _random_boundary,
# _random_stress, _random_ep, _dynamic_boundary, _as_virtual_stress,
# _as_virtual_ep, _as_virtual_fluid) — included first in runtests.jl, same
# top-level @testset scope.
# ---------------------------------------------------------------------------

_dv_pairkernel(h; ndims=2) = CubicSplineKernel(h; ndims=ndims)

# dx/gx/w for a specific (i,j) pair, exactly as the sweep would compute them.
function _dv_pair_dx_gx_w(kernel, h, xi, xj)
    dx = xi - xj
    r  = norm(dx)
    q  = r / h
    gx = (Grasph.kernel_dw_dq(kernel, q) / (r * h)) * dx
    w  = Grasph.kernel_w(kernel, q)
    return dx, gx, w
end

@testset "device_view is a faithful proxy (Stress/ElastoPlastic/DynamicBoundary/Virtual)" begin
    rng = MersenneTwister(500)

    stress = _random_stress(rng, 50, 2)
    dvs = Grasph.device_view(stress)
    for f in (:x, :v, :v_adjustment, :rho, :dvdt, :drhodt, :p, :stress, :strain_rate,
              :mass, :c, :source_v, :source_rho)
        @test getproperty(dvs, f) == getproperty(stress, f)
    end
    @test dvs isa Grasph.AbstractParticleSystem{Float64,2}

    ep = _random_ep(rng, 50, 2)
    dve = Grasph.device_view(ep)
    for f in (:x, :v, :v_adjustment, :rho, :dvdt, :drhodt, :p, :stress, :strain_rate,
              :vorticity, :strain, :strain_p, :mass, :c, :source_v, :source_rho)
        @test getproperty(dve, f) == getproperty(ep, f)
    end
    @test dve isa Grasph.AbstractParticleSystem{Float64,2}

    bnd = _dynamic_boundary(rng, 40, 2)
    dvb = Grasph.device_view(bnd)
    @test dvb isa DynamicBoundarySystem
    @test dvb.x == bnd.x
    @test dvb.boundary_normal == bnd.boundary_normal
    @test dvb.boundary_point  == bnd.boundary_point
    @test dvb.boundary_beta   == bnd.boundary_beta
    # Structural check, not just value equality: the wrapped inner system must
    # actually have been replaced by a device view (a device_view that forgot
    # to recurse into :inner would pass every check above unchanged, since the
    # underlying arrays — and thus every value comparison — are identical
    # either way; only inner's *type* reveals the difference).
    @test getfield(dvb, :inner) isa Grasph.DeviceSystem

    virt = _as_virtual_fluid(rng, 60, 2)
    dvv = Grasph.device_view(virt)
    @test dvv isa Grasph.AbstractVirtualParticleSystem{Float64,2}
    for f in (:x, :v, :rho, :p, :mass, :c, :w_sum, :prescribed_v)
        @test getproperty(dvv, f) == getproperty(virt, f)
    end
    # Fields that must NOT survive into the device view — this is the whole
    # reason device_view exists (see DeviceViews.jl's header comment).
    inner_names = keys(getfield(dvv, :_f))
    @test :name           ∉ inner_names
    @test :state_updater  ∉ inner_names
    @test :source         ∉ inner_names   # flattened away, not carried as a sub-object
end

@testset "pfn_contribution: device_view(ps_b) matches host ps_b exactly" begin
    rng = MersenneTwister(501)
    h = 0.1
    kernel = _dv_pairkernel(h)

    @testset "StrainRatePfn, ghost/virtual-coupled" begin
        pfn = StrainRatePfn()
        for (build_a, build_b) in ((_random_stress, _as_virtual_stress), (_random_ep, _as_virtual_ep))
            ps_a = build_a(rng, 2, 2; L=0.05)
            ps_b = build_b(rng, 2, 2; L=0.05)
            dx, gx, w = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], ps_b.x[2])
            c_host   = pfn_contribution(pfn, ps_a, ps_b, 1, 2, dx, gx, w)
            c_device = pfn_contribution(pfn, ps_a, Grasph.device_view(ps_b), 1, 2, dx, gx, w)
            @test c_host == c_device
        end
    end

    @testset "StrainRateVorticityPfn, ghost/virtual-coupled" begin
        pfn = StrainRateVorticityPfn()
        ps_a = _random_ep(rng, 2, 2; L=0.05)
        virt = _as_virtual_ep(rng, 2, 2; L=0.05)
        dx, gx, w = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], virt.x[2])
        c_host   = pfn_contribution(pfn, ps_a, virt, 1, 2, dx, gx, w)
        c_device = pfn_contribution(pfn, ps_a, Grasph.device_view(virt), 1, 2, dx, gx, w)
        @test c_host == c_device
    end

    @testset "InterpolateFieldFn, Virtual-in-ps_a (device_view of the WRITE target)" begin
        # The one dispatch shape among the pfns touched by this widening where
        # the virtual system sits in ps_a (the write target), not ps_b — every
        # other pfn above has it in ps_b. Exercised separately since it's a
        # structurally distinct method (PairwiseFunctors.jl's InterpolateFieldFn
        # pfn_contribution), not just a relabelling of the same one.
        pfn = InterpolateFieldFn(:v, :rho; accumulate_wsum=true)
        virt  = _as_virtual_fluid(rng, 2, 2; L=0.05)
        fluid = _random_fluid(rng, 2, 2; L=0.05)
        dx, gx, w = _dv_pair_dx_gx_w(kernel, h, virt.x[1], fluid.x[2])
        c_host   = pfn_contribution(pfn, virt, fluid, 1, 2, dx, gx, w)
        c_device = pfn_contribution(pfn, Grasph.device_view(virt), fluid, 1, 2, dx, gx, w)
        @test c_host == c_device
    end

    @testset "FluidPfn, ghost/virtual-coupled and dynamic-boundary-coupled" begin
        pfn = FluidPfn(0.03, 0.0, h)
        ps_a = _random_fluid(rng, 2, 2; L=0.05)

        virt = _as_virtual_fluid(rng, 2, 2; L=0.05)
        dx, gx, w = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], virt.x[2])
        @test pfn_contribution(pfn, ps_a, virt, 1, 2, dx, gx, w) ==
              pfn_contribution(pfn, ps_a, Grasph.device_view(virt), 1, 2, dx, gx, w)

        bnd = _dynamic_boundary(rng, 2, 2; L=0.05)
        dx2, gx2, w2 = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], bnd.x[2])
        @test pfn_contribution(pfn, ps_a, bnd, 1, 2, dx2, gx2, w2) ==
              pfn_contribution(pfn, ps_a, Grasph.device_view(bnd), 1, 2, dx2, gx2, w2)
    end

    @testset "CauchyFluidPfn, ghost/virtual-coupled and dynamic-boundary-coupled" begin
        pfn = CauchyFluidPfn(0.03, 0.0, h)
        ps_a = _random_stress(rng, 2, 2; L=0.05)

        virt = _as_virtual_stress(rng, 2, 2; L=0.05)
        dx, gx, w = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], virt.x[2])
        @test pfn_contribution(pfn, ps_a, virt, 1, 2, dx, gx, w) ==
              pfn_contribution(pfn, ps_a, Grasph.device_view(virt), 1, 2, dx, gx, w)

        bnd = _dynamic_boundary(rng, 2, 2; L=0.05)
        dx2, gx2, w2 = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], bnd.x[2])
        @test pfn_contribution(pfn, ps_a, bnd, 1, 2, dx2, gx2, w2) ==
              pfn_contribution(pfn, ps_a, Grasph.device_view(bnd), 1, 2, dx2, gx2, w2)
    end

    @testset "XSPHPfn, ghost/virtual-coupled" begin
        pfn = XSPHPfn(0.5)
        ps_a = _random_fluid(rng, 2, 2; L=0.05)
        virt = _as_virtual_fluid(rng, 2, 2; L=0.05)
        dx, gx, w = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], virt.x[2])
        @test pfn_contribution(pfn, ps_a, virt, 1, 2, dx, gx, w) ==
              pfn_contribution(pfn, ps_a, Grasph.device_view(virt), 1, 2, dx, gx, w)
    end
end

@testset "state updaters read prescribed_v through getproperty, not raw getfield" begin
    # Regression test: VirtualNormUpdater/PrescribedVelocityUpdater used to
    # read prescribed_v via getfield(ps, :prescribed_v), which works on the
    # real VirtualParticleSystem (prescribed_v is a genuine struct field
    # there) but bypasses DeviceVirtualSystem's getproperty entirely — raw
    # getfield never consults getproperty overrides — so it threw FieldError
    # the moment either updater ran against a device_view'd virtual system
    # (i.e. the first time a VirtualParticleSystem's arrays are GPU-resident,
    # which is exactly what this device_view extension newly makes reachable).
    rng = MersenneTwister(503)
    build_virt() = let src = _random_fluid(rng, 5, 2; L=0.05)
        fill!(src.v, zero(SVector{2,Float64}))
        VirtualParticleSystem(src, "virt", src.n, 2, src.mass, src.c;
                              prescribed_v = SVector(1.5, -2.5))
    end

    virt_host = build_virt()
    virt_dev  = deepcopy(virt_host)
    dvv = Grasph.device_view(virt_dev)

    u1 = VirtualNormUpdater(SVector(1.0, 1.0), :v)
    for v in (virt_host, dvv)
        v.w_sum[1] = 2.0
        v.v[1] = SVector(4.0, 6.0)
    end
    u1(virt_host, 1)
    u1(dvv, 1)   # must not throw FieldError; only reachable if getproperty (not raw getfield) is used
    @test virt_host.v[1] == virt_dev.v[1]

    u2 = PrescribedVelocityUpdater()
    for v in (virt_host, dvv)
        v.v[1] = SVector(0.1, 0.2)
    end
    u2(virt_host, 1)
    u2(dvv, 1)
    @test virt_host.v[1] == virt_dev.v[1]
end

@testset "narrow typing wasn't loosened by the AbstractVirtualParticleSystem widening" begin
    rng = MersenneTwister(502)
    h = 0.1
    kernel = _dv_pairkernel(h)
    pfn = FluidPfn(0.03, 0.0, h)
    ps_a = _random_fluid(rng, 2, 2; L=0.05)
    ps_b_bare = _random_boundary(rng, 2, 2; L=0.05)   # bare BasicParticleSystem, no wrapper

    dx, gx, w = _dv_pair_dx_gx_w(kernel, h, ps_a.x[1], ps_b_bare.x[1])
    @test_throws MethodError pfn_contribution(pfn, ps_a, ps_b_bare, 1, 1, dx, gx, w)
    @test_throws MethodError pfn_contribution(pfn, ps_a, Grasph.device_view(ps_b_bare), 1, 1, dx, gx, w)
end
