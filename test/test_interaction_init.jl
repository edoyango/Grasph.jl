using Test
using Grasph

# Helpers used across all interaction init tests
_make_ps(; n=10, ndims=2) = BasicParticleSystem("test", n, ndims, 1.0, 1.0)
_make_kernel(; h=0.1, ndims=2) = CubicSplineKernel(h; ndims=ndims)
_dummy_pairwise(args...) = nothing

@testset "SystemInteraction initialisation" begin

    # ------------------------------------------------------------------
    # Basic field assignment
    # ------------------------------------------------------------------

    @testset "self interaction stores system_a and nothing for system_b" begin
        ps = _make_ps()
        k  = _make_kernel()
        si = SystemInteraction(k, _dummy_pairwise, ps)
        @test si.system_a === ps
        @test si.system_b === nothing
        @test si.ndims    == ps.ndims
        @test !is_coupled(si)
    end

    @testset "coupled interaction stores both systems" begin
        ps_a = _make_ps()
        ps_b = _make_ps()
        k    = _make_kernel()
        si   = SystemInteraction(k, _dummy_pairwise, ps_a, ps_b)
        @test si.system_a === ps_a
        @test si.system_b === ps_b
        @test si.ndims    == ps_a.ndims
        @test is_coupled(si)
    end

    @testset "kernel and pairwise_fn are stored by reference" begin
        ps = _make_ps()
        k  = _make_kernel()
        si = SystemInteraction(k, _dummy_pairwise, ps)
        @test si.kernel        === k
        @test si.pfns[1]       === _dummy_pairwise
    end

    # ------------------------------------------------------------------
    # Internal working-array shapes
    # ------------------------------------------------------------------

    @testset "self interaction internal arrays have correct shapes" begin
        n, nd = 15, 2
        ps = _make_ps(n=n, ndims=nd)
        si = SystemInteraction(_make_kernel(), _dummy_pairwise, ps)
        @test length(si._mingridx) == nd
        @test length(si._ngridx)   == nd
    end

    @testset "grid arrays are empty before create_grid!" begin
        ps = _make_ps()
        si = SystemInteraction(_make_kernel(), _dummy_pairwise, ps)
        @test isempty(si._cell_start)
        @test isempty(si._cell_count)
    end

    # ------------------------------------------------------------------
    # Cell size
    # ------------------------------------------------------------------

    @testset "cell_size equals kernel interaction_length" begin
        k  = _make_kernel(h=0.15)
        ps = _make_ps()
        si = SystemInteraction(k, _dummy_pairwise, ps)
        @test si._cell_size ≈ k.interaction_length
    end

    @testset "cell_size is 2h for CubicSplineKernel" begin
        h  = 0.05
        k  = _make_kernel(h=h)
        ps = _make_ps()
        si = SystemInteraction(k, _dummy_pairwise, ps)
        @test si._cell_size ≈ 2*h
    end

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @testset "mismatched ndims raises ArgumentError" begin
        ps_a = _make_ps(ndims=2)
        ps_b = _make_ps(ndims=3)
        k    = _make_kernel()
        @test_throws ArgumentError SystemInteraction(k, _dummy_pairwise, ps_a, ps_b)
    end

    @testset "kernel ndims mismatch raises ArgumentError" begin
        ps = _make_ps(ndims=2)
        k  = _make_kernel(ndims=3)
        @test_throws ArgumentError SystemInteraction(k, _dummy_pairwise, ps)
    end

    @testset "mismatched ndims error message" begin
        ps_a = _make_ps(ndims=2)
        ps_b = _make_ps(ndims=3)
        k    = _make_kernel()
        try
            SystemInteraction(k, _dummy_pairwise, ps_a, ps_b)
            @test false  # should not reach here
        catch e
            @test e isa ArgumentError
            @test occursin("system_b", e.msg)
        end
    end

    @testset "matching ndims does not raise" begin
        ps_a = _make_ps(ndims=3)
        ps_b = _make_ps(ndims=3)
        k    = _make_kernel(ndims=3)
        si   = SystemInteraction(k, _dummy_pairwise, ps_a, ps_b)
        @test si.ndims == 3
    end

    # ------------------------------------------------------------------
    # Pairwise functor float-type checking
    # ------------------------------------------------------------------

    @testset "pfn with matching float type does not raise" begin
        ps = _make_ps()  # Float64
        k  = _make_kernel()
        pfn = FluidPfn(0.1, 0.0, 0.1)  # infers Float64
        @test pfn isa FluidPfn{Float64}
        si = SystemInteraction(k, pfn, ps)
        @test si isa SystemInteraction
    end

    @testset "pfn with mismatched float type raises ArgumentError" begin
        ps  = _make_ps()  # Float64
        k   = _make_kernel()
        pfn = FluidPfn(Float32(0.1), Float32(0.0), Float32(0.1))
        @test pfn isa FluidPfn{Float32}
        @test_throws ArgumentError SystemInteraction(k, pfn, ps)
    end

    @testset "pfn mismatch error mentions functor type and both float types" begin
        ps  = _make_ps()
        k   = _make_kernel()
        pfn = FluidPfn(Float32(0.1), Float32(0.0), Float32(0.1))
        try
            SystemInteraction(k, pfn, ps)
            @test false
        catch e
            @test e isa ArgumentError
            @test occursin("FluidPfn", e.msg)
            @test occursin("Float32", e.msg)
            @test occursin("Float64", e.msg)
        end
    end

    @testset "unparameterized pfn (StrainRatePfn) does not raise" begin
        ps = BasicParticleSystem("test", 5, 2, 1.0, 1.0)
        k  = _make_kernel()
        @test_nowarn SystemInteraction(k, StrainRatePfn(), ps)
    end

    @testset "tuple of pfns — mismatch in second pfn raises" begin
        ps   = _make_ps()
        k    = _make_kernel()
        pfn1 = _dummy_pairwise
        pfn2 = FluidPfn(Float32(0.1), Float32(0.0), Float32(0.1))
        @test_throws ArgumentError SystemInteraction(k, (pfn1, pfn2), ps)
    end

end
