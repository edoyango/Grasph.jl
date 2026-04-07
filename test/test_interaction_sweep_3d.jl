using Test
using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Counting pairwise functions (3D versions)
# ---------------------------------------------------------------------------

function make_counting_pfn_3d()
    (ps, i, j, dx, gx, w) -> begin
        ps.dvdt[i]   += SVector(1.0, 0.0, 0.0)
        ps.dvdt[j]   += SVector(1.0, 0.0, 0.0)
        ps.drhodt[i] += 1.0
        ps.drhodt[j] += 1.0
    end
end

function make_counting_pfn_coupled_3d()
    (ps_a, ps_b, i, j, dx, gx, w) -> begin
        ps_a.dvdt[i]   += SVector(1.0, 0.0, 0.0)
        ps_b.dvdt[j]   += SVector(1.0, 0.0, 0.0)
        ps_a.drhodt[i] += 1.0
        ps_b.drhodt[j] += 1.0
    end
end

# ---------------------------------------------------------------------------
# Helpers: inject a manually-constructed cell list into si (3D)
# ---------------------------------------------------------------------------

function _inject_self_grid_3d!(si, n; ngrid=(4, 4, 4))
    ncells = prod(ngrid)
    nc_y, nc_z = ngrid[2], ngrid[3]
    # Place in cell (2,2,2)
    cell = (2 - 1) * (nc_y * nc_z) + (2 - 1) * nc_z + 2
    si._ngridx .= Int[ngrid[1], ngrid[2], ngrid[3]]
    resize!(si._cell_head, ncells); si._cell_head .= 0
    resize!(si._cell_next, n);      si._cell_next .= 0
    for i in 1:n
        si._cell_next[i]   = si._cell_head[cell]
        si._cell_head[cell] = Int(i)
    end
end

function _inject_coupled_grid_3d!(si, n_a, n_b; ngrid=(5, 5, 5))
    ncells = prod(ngrid)
    nc_y, nc_z = ngrid[2], ngrid[3]
    # Place in center cell (3,3,3)
    center = (3 - 1) * (nc_y * nc_z) + (3 - 1) * nc_z + 3
    si._ngridx .= Int[ngrid[1], ngrid[2], ngrid[3]]
    resize!(si._cell_head,   ncells); si._cell_head   .= 0
    resize!(si._cell_next,   n_b);    si._cell_next   .= 0
    resize!(si._cell_head_a, ncells); si._cell_head_a .= 0
    resize!(si._cell_next_a, n_a);    si._cell_next_a .= 0
    cutoff = si._cell_size
    si._mingridx   .= 0.0
    si._mingridx_a .= 2 * cutoff
    si._maxgridx_a .= 2 * cutoff
    for i in 1:n_b
        si._cell_next[i]       = si._cell_head[center]
        si._cell_head[center]  = Int(i)
    end
    for i in 1:n_a
        si._cell_next_a[i]       = si._cell_head_a[center]
        si._cell_head_a[center]  = Int(i)
    end
end

_make_ps_3d(; n=2)   = BasicParticleSystem("test", n, 3, 1.0, 1.0)
_make_k_3d(; h=0.1)  = CubicSplineKernel(h; ndims=3)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "3D interaction sweep" begin

    @testset "self sweep 3D — manual grid" begin
        n  = 2
        k  = _make_k_3d()
        ps = _make_ps_3d(n=n)
        si = SystemInteraction(k, make_counting_pfn_3d(), ps)

        ps.x[1] = SVector(0.05, 0.05, 0.05)
        ps.x[2] = SVector(0.10, 0.10, 0.10)   # dist ≈ 0.086 < 0.2
        fill!(ps.dvdt, zero(SVector{3,Float64}));  ps.drhodt .= 0.0

        _inject_self_grid_3d!(si, n)
        sweep!(si)

        @test ps.dvdt[1][1] == 1.0
        @test ps.dvdt[2][1] == 1.0
        @test ps.drhodt[1]  == 1.0
        @test ps.drhodt[2]  == 1.0
    end

    @testset "coupled sweep 3D — manual grid" begin
        n  = 1
        k  = _make_k_3d()
        ps_a = _make_ps_3d(n=n)
        ps_b = _make_ps_3d(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled_3d(), ps_a, ps_b)

        ps_a.x[1] = SVector(0.05, 0.05, 0.05)
        ps_b.x[1] = SVector(0.10, 0.10, 0.10)
        fill!(ps_a.dvdt, zero(SVector{3,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{3,Float64}));  ps_b.drhodt .= 0.0

        _inject_coupled_grid_3d!(si, n, n)
        sweep!(si)

        @test ps_a.dvdt[1][1] == 1.0
        @test ps_b.dvdt[1][1] == 1.0
        @test ps_a.drhodt[1]  == 1.0
        @test ps_b.drhodt[1]  == 1.0
    end

    @testset "self sweep 3D — full pipeline" begin
        n  = 3
        k  = _make_k_3d()
        ps = _make_ps_3d(n=n)
        si = SystemInteraction(k, make_counting_pfn_3d(), ps)

        ps.x[1] = SVector(0.05, 0.05, 0.05)
        ps.x[2] = SVector(0.10, 0.10, 0.10)
        ps.x[3] = SVector(0.50, 0.50, 0.50) # Far away
        fill!(ps.dvdt, zero(SVector{3,Float64}));  ps.drhodt .= 0.0

        create_grid!(si)
        sweep!(si)

        @test getindex.(ps.dvdt, 1) ≈ [1.0, 1.0, 0.0]
        @test ps.drhodt ≈ [1.0, 1.0, 0.0]
    end

    @testset "coupled sweep 3D — full pipeline" begin
        n  = 2
        k  = _make_k_3d()
        ps_a = _make_ps_3d(n=n)
        ps_b = _make_ps_3d(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled_3d(), ps_a, ps_b)

        ps_a.x[1] = SVector(0.05, 0.05, 0.05)
        ps_a.x[2] = SVector(0.50, 0.50, 0.50)
        ps_b.x[1] = SVector(0.06, 0.06, 0.06)
        ps_b.x[2] = SVector(0.51, 0.51, 0.51)
        fill!(ps_a.dvdt, zero(SVector{3,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{3,Float64}));  ps_b.drhodt .= 0.0

        create_grid!(si)
        sweep!(si)

        # (a1, b1) interact, (a2, b2) interact. 
        @test getindex.(ps_a.dvdt, 1) ≈ [1.0, 1.0]
        @test getindex.(ps_b.dvdt, 1) ≈ [1.0, 1.0]
    end

end
