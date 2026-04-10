using Test
using Grasph
using StaticArrays

# ---------------------------------------------------------------------------
# Counting pairwise functions (mirrors Python conftest.py)
#
# The sweep calls pairwise_fn with 5 args: (i, j, dx, gx, w).
# All particle state is captured via closure (factory pattern).
# These factory helpers return a closure that increments counters so tests
# can assert exactly which pairs were found.
# ---------------------------------------------------------------------------

function make_counting_pfn()
    (ps, i, j, dx, gx, w) -> begin
        ps.dvdt[i]   += SVector(1.0, 0.0)
        ps.dvdt[j]   += SVector(1.0, 0.0)
        ps.drhodt[i] += 1.0
        ps.drhodt[j] += 1.0
    end
end

function make_counting_pfn_coupled()
    (ps_a, ps_b, i, j, dx, gx, w) -> begin
        ps_a.dvdt[i]   += SVector(1.0, 0.0)
        ps_b.dvdt[j]   += SVector(1.0, 0.0)
        ps_a.drhodt[i] += 1.0
        ps_b.drhodt[j] += 1.0
    end
end

# ---------------------------------------------------------------------------
# Helpers: inject a manually-constructed cell list into si
# ---------------------------------------------------------------------------

# Place all n system_a particles in cell (2,2) of a (4,4) grid.
# Cell (ci=2, cj=2) has buffer cells in every direction, so all 4
# forward-neighbour offsets (+1,0), (-1,+1), (0,+1), (+1,+1) stay
# within [1, ncells].  Cell 1 (the corner) would make (-1,+1) compute
# flat - nc_y + 1 = -1, which is an out-of-bounds access.
function _inject_self_grid!(si, n; ngrid=(4, 4))
    ncells = prod(ngrid)
    nc_y   = ngrid[2]
    cell   = (2 - 1) * nc_y + 2   # ci=2, cj=2 → flat index
    si._ngridx .= [ngrid[1], ngrid[2]]
    # Prefix-sum format (length ncells+1):
    #   cells 1..cell all point to particle 1 (empty cells before + the non-empty cell)
    #   cells cell+1..ncells+1 all point to n+1 (sentinel, empty cells after)
    resize!(si._cell_start, ncells + 1)
    fill!(view(si._cell_start, 1:cell),          1)
    fill!(view(si._cell_start, cell+1:ncells+1), n + 1)
end

# Place all n_a system_a particles and all n_b system_b particles in the
# centre cell of a (5×5) grid (cell 13, 1-indexed, ci=3,cj=3).
# Using a 5×5 grid ensures the centre cell has valid neighbours in all
# directions for the full 3×3 neighbourhood lookup, and also that it
# falls on colour 0 of the 9-colour scheme (ci%3==0, cj%3==0 → start=3,3).
function _inject_coupled_grid!(si, n_a, n_b; ngrid=(5, 5))
    ncells = prod(ngrid)
    nc_y   = ngrid[2]
    center = (3 - 1) * nc_y + 3               # ci=3, cj=3
    si._ngridx .= [ngrid[1], ngrid[2]]
    cutoff = si._cell_size
    si._mingridx   .= 0.0
    si._mingridx_a .= 2 * cutoff   # lower bound of cell ci=cj=3: (3-1)*cutoff
    si._maxgridx_a .= 2 * cutoff
    # system_b in cell `center`: prefix-sum arrays of length ncells+1
    resize!(si._cell_start,   ncells + 1)
    fill!(view(si._cell_start,   1:center),          1)
    fill!(view(si._cell_start,   center+1:ncells+1), n_b + 1)
    # system_a in cell `center`
    resize!(si._cell_start_a, ncells + 1)
    fill!(view(si._cell_start_a, 1:center),          1)
    fill!(view(si._cell_start_a, center+1:ncells+1), n_a + 1)
end

_make_ps(; n=2, ndims=2)   = BasicParticleSystem("test", n, ndims, 1.0, 1.0)
_make_k(; h=0.1, ndims=2)  = CubicSplineKernel(h; ndims=ndims)

# ---------------------------------------------------------------------------
# Self-interaction sweep — manual grid
# ---------------------------------------------------------------------------

@testset "self sweep — manual grid" begin

    @testset "2 particles within cutoff → exactly one pair" begin
        n  = 2
        k  = _make_k()
        ps = _make_ps(n=n)
        si = SystemInteraction(k, make_counting_pfn(), ps)

        ps.x[1] = SVector(0.05, 0.05)
        ps.x[2] = SVector(0.10, 0.10)   # distance ≈ 0.071 < cutoff 0.2
        fill!(ps.dvdt, zero(SVector{2,Float64}));  ps.drhodt .= 0.0

        _inject_self_grid!(si, n)
        sweep!(si)

        @test ps.dvdt[1][1] == 1.0
        @test ps.dvdt[2][1] == 1.0
        @test ps.drhodt[1]  == 1.0
        @test ps.drhodt[2]  == 1.0
    end

    @testset "3 particles — only nearby pair survives distance cutoff" begin
        # d(1,2) ≈ 0.071 < 0.2 → found
        # d(1,3) ≈ 0.283 > 0.2 → filtered
        # d(2,3) ≈ 0.212 > 0.2 → filtered
        n  = 3
        k  = _make_k()
        ps = _make_ps(n=n)
        si = SystemInteraction(k, make_counting_pfn(), ps)

        ps.x[1] = SVector(0.05, 0.05)
        ps.x[2] = SVector(0.10, 0.10)
        ps.x[3] = SVector(0.25, 0.25)
        fill!(ps.dvdt, zero(SVector{2,Float64}));  ps.drhodt .= 0.0

        _inject_self_grid!(si, n)
        sweep!(si)

        @test getindex.(ps.dvdt, 1)  == [1.0, 1.0, 0.0]
        @test ps.drhodt      == [1.0, 1.0, 0.0]
    end

    @testset "8 particles all within cutoff → 7 pairs each" begin
        # C(8,2) = 28 pairs; each particle in exactly 7.
        n  = 8
        k  = _make_k()
        ps = _make_ps(n=n)
        si = SystemInteraction(k, make_counting_pfn(), ps)

        rng = (i -> mod(i * 6364136223846793005 + 1442695040888963407, 2^32))  # LCG
        vals = [(mod(rng(i), 100) / 2000.0) for i in 1:2n]
        ps.x .= [SVector(vals[2i-1], vals[2i]) for i in 1:n]
        fill!(ps.dvdt, zero(SVector{2,Float64}));  ps.drhodt .= 0.0

        _inject_self_grid!(si, n)
        sweep!(si)

        @test all(==(7.0), getindex.(ps.dvdt, 1))
        @test all(==(7.0), ps.drhodt)
    end

    @testset "out-of-cutoff pair is never called" begin
        # Two particles far apart — no pair should be found.
        n  = 2
        k  = _make_k()
        ps = _make_ps(n=n)
        si = SystemInteraction(k, make_counting_pfn(), ps)

        ps.x[1] = SVector(0.0,  0.0)
        ps.x[2] = SVector(1.0,  1.0)   # distance ≈ 1.41 >> cutoff 0.2
        fill!(ps.dvdt, zero(SVector{2,Float64}));  ps.drhodt .= 0.0

        _inject_self_grid!(si, n)
        sweep!(si)

        @test all(iszero, ps.dvdt)
        @test all(iszero, ps.drhodt)
    end

end

# ---------------------------------------------------------------------------
# Self-interaction sweep — full pipeline (create_grid! + sweep!)
# ---------------------------------------------------------------------------

@testset "self sweep — full pipeline" begin

    @testset "2 particles within cutoff" begin
        n  = 2
        k  = _make_k()
        ps = _make_ps(n=n)
        si = SystemInteraction(k, make_counting_pfn(), ps)

        ps.x[1] = SVector(0.05, 0.05)
        ps.x[2] = SVector(0.10, 0.10)
        fill!(ps.dvdt, zero(SVector{2,Float64}));  ps.drhodt .= 0.0

        create_grid!(si)
        sweep!(si)

        @test ps.dvdt[1][1] == 1.0
        @test ps.dvdt[2][1] == 1.0
        @test ps.drhodt[1]  == 1.0
        @test ps.drhodt[2]  == 1.0
    end

    @testset "3 particles — distance cutoff filter" begin
        n  = 3
        k  = _make_k()
        ps = _make_ps(n=n)
        si = SystemInteraction(k, make_counting_pfn(), ps)

        ps.x[1] = SVector(0.05, 0.05)
        ps.x[2] = SVector(0.10, 0.10)
        ps.x[3] = SVector(0.25, 0.25)
        fill!(ps.dvdt, zero(SVector{2,Float64}));  ps.drhodt .= 0.0

        create_grid!(si)
        sweep!(si)

        @test getindex.(ps.dvdt, 1)  == [1.0, 1.0, 0.0]
        @test ps.drhodt      == [1.0, 1.0, 0.0]
    end

end

# ---------------------------------------------------------------------------
# Coupled sweep — manual grid
# ---------------------------------------------------------------------------

@testset "coupled sweep — manual grid" begin

    @testset "1 particle each, within cutoff → one pair" begin
        n  = 1
        k  = _make_k()
        ps_a = _make_ps(n=n)
        ps_b = _make_ps(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

        ps_a.x[1] = SVector(0.05, 0.05)
        ps_b.x[1] = SVector(0.10, 0.10)   # d ≈ 0.071 < 0.2
        fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

        _inject_coupled_grid!(si, n, n)
        sweep!(si)

        @test ps_a.dvdt[1][1] == 1.0
        @test ps_b.dvdt[1][1] == 1.0
        @test ps_a.drhodt[1]  == 1.0
        @test ps_b.drhodt[1]  == 1.0
    end

    @testset "3 particles each — cluster + lone pair" begin
        # a0,a1 cluster near b0,b1; a2 pairs only with b2.
        # Pairs within cutoff: (a1,b1),(a1,b2),(a2,b1),(a2,b2),(a3,b3).
        # Expected per particle: a=[2,2,1], b=[2,2,1].
        n  = 3
        k  = _make_k()
        ps_a = _make_ps(n=n)
        ps_b = _make_ps(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

        ps_a.x .= [SVector(0.05, 0.05), SVector(0.10, 0.10), SVector(0.50, 0.50)]
        ps_b.x .= [SVector(0.06, 0.06), SVector(0.11, 0.11), SVector(0.55, 0.55)]
        fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

        _inject_coupled_grid!(si, n, n)
        sweep!(si)

        @test getindex.(ps_a.dvdt, 1)  == [2.0, 2.0, 1.0]
        @test ps_a.drhodt      == [2.0, 2.0, 1.0]
        @test getindex.(ps_b.dvdt, 1)  == [2.0, 2.0, 1.0]
        @test ps_b.drhodt      == [2.0, 2.0, 1.0]
    end

    @testset "8 particles each — all pairs within cutoff → 8 each" begin
        n  = 8
        k  = _make_k()
        ps_a = _make_ps(n=n)
        ps_b = _make_ps(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

        rng = (i -> mod(i * 6364136223846793005 + 1442695040888963407, 2^32))
        vals_a = [(mod(rng(i),    100) / 2000.0) for i in 1:2n]
        vals_b = [(mod(rng(i+2n), 100) / 2000.0) for i in 1:2n]
        ps_a.x .= [SVector(vals_a[2i-1], vals_a[2i]) for i in 1:n]
        ps_b.x .= [SVector(vals_b[2i-1], vals_b[2i]) for i in 1:n]
        fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

        _inject_coupled_grid!(si, n, n)
        sweep!(si)

        @test all(==(Float64(n)), getindex.(ps_a.dvdt, 1))
        @test all(==(Float64(n)), ps_a.drhodt)
        @test all(==(Float64(n)), getindex.(ps_b.dvdt, 1))
        @test all(==(Float64(n)), ps_b.drhodt)
    end

    @testset "out-of-cutoff pair is never called (coupled)" begin
        n  = 1
        k  = _make_k()
        ps_a = _make_ps(n=n)
        ps_b = _make_ps(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

        ps_a.x[1] = SVector(0.0, 0.0)
        ps_b.x[1] = SVector(1.0, 1.0)   # far apart
        fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

        _inject_coupled_grid!(si, n, n)
        sweep!(si)

        @test all(iszero, ps_a.dvdt)
        @test all(iszero, ps_b.dvdt)
    end

end

# ---------------------------------------------------------------------------
# Coupled sweep — full pipeline (create_grid! + sweep!)
# ---------------------------------------------------------------------------

@testset "coupled sweep — full pipeline" begin

    @testset "1 particle each, within cutoff" begin
        n  = 1
        k  = _make_k()
        ps_a = _make_ps(n=n)
        ps_b = _make_ps(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

        ps_a.x[1] = SVector(0.05, 0.05)
        ps_b.x[1] = SVector(0.10, 0.10)
        fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

        create_grid!(si)
        sweep!(si)

        @test ps_a.dvdt[1][1] == 1.0
        @test ps_b.dvdt[1][1] == 1.0
        @test ps_a.drhodt[1]  == 1.0
        @test ps_b.drhodt[1]  == 1.0
    end

    @testset "3 particles each — cluster + lone pair" begin
        n  = 3
        k  = _make_k()
        ps_a = _make_ps(n=n)
        ps_b = _make_ps(n=n)
        si   = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

        ps_a.x .= [SVector(0.05, 0.05), SVector(0.10, 0.10), SVector(0.50, 0.50)]
        ps_b.x .= [SVector(0.06, 0.06), SVector(0.11, 0.11), SVector(0.55, 0.55)]
        fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
        fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

        create_grid!(si)
        sweep!(si)

        @test getindex.(ps_a.dvdt, 1)  == [2.0, 2.0, 1.0]
        @test ps_a.drhodt      == [2.0, 2.0, 1.0]
        @test getindex.(ps_b.dvdt, 1)  == [2.0, 2.0, 1.0]
        @test ps_b.drhodt      == [2.0, 2.0, 1.0]
    end

end

# ---------------------------------------------------------------------------
# Accumulation across two interactions (self + coupled)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# pfn field access — demonstrate that pfns can read/write arbitrary fields
# ---------------------------------------------------------------------------

@testset "pfn can access and mutate a captured array" begin
    # A closure accumulates 1 per pair per endpoint into a separately allocated array.
    n  = 2
    k  = _make_k()
    ps = _make_ps(n=n)
    ps.x[1] = SVector(0.05, 0.05)
    ps.x[2] = SVector(0.10, 0.10)   # d ≈ 0.071 < cutoff 0.2

    denergy = zeros(Float64, n)
    si = SystemInteraction(k, (ps, i, j, dx, gx, w) -> (denergy[i] += 1.0; denergy[j] += 1.0), ps)

    _inject_self_grid!(si, n)
    sweep!(si)

    @test denergy[1] == 1.0
    @test denergy[2] == 1.0
end

@testset "coupled pfn reads ps_b field, writes only to ps_a" begin
    # Closure reads ps_b.rho[j] and accumulates into ps_a.drhodt[i]; ps_b untouched.
    n    = 1
    k    = _make_k()
    ps_a = _make_ps(n=n)
    ps_b = _make_ps(n=n)
    ps_a.x[1] = SVector(0.05, 0.05)
    ps_b.x[1] = SVector(0.10, 0.10)   # within cutoff
    ps_a.rho .= 500.0;  ps_b.rho .= 750.0
    ps_a.drhodt .= 0.0; ps_b.drhodt .= 0.0
    fill!(ps_a.dvdt, zero(SVector{2,Float64}))
    fill!(ps_b.dvdt, zero(SVector{2,Float64}))

    si = SystemInteraction(k, (ps_a, ps_b, i, j, dx, gx, w) -> (ps_a.drhodt[i] += ps_b.rho[j]), ps_a, ps_b)

    _inject_coupled_grid!(si, n, n)
    sweep!(si)

    @test ps_a.drhodt[1] == 750.0   # accumulated ps_b.rho[j]
    @test all(iszero, ps_b.drhodt)  # ps_b untouched
    @test all(iszero, ps_b.dvdt)
end

# ---------------------------------------------------------------------------

@testset "two interactions accumulate independently" begin
    # system_a (2 particles) has a self-interaction and couples with
    # system_b (1 particle).  All 3 are within cutoff of each other.
    #
    # Self sweep (a only):
    #   pair (a1,a2) → dvdt_a[1,1] += 1, dvdt_a[2,1] += 1
    #
    # Coupled sweep (a × b):
    #   pair (a1,b1) → dvdt_a[1,1] += 1, dvdt_b[1,1] += 1
    #   pair (a2,b1) → dvdt_a[2,1] += 1, dvdt_b[1,1] += 1
    #
    # Totals: dvdt_a[:,1] = drhodt_a = [2,2]; dvdt_b[:,1] = drhodt_b = [2]
    n_a, n_b = 2, 1
    k  = _make_k()
    ps_a = _make_ps(n=n_a)
    ps_b = _make_ps(n=n_b)
    si_self    = SystemInteraction(k, make_counting_pfn(), ps_a)
    si_coupled = SystemInteraction(k, make_counting_pfn_coupled(), ps_a, ps_b)

    ps_a.x[1] = SVector(0.05, 0.05)
    ps_a.x[2] = SVector(0.10, 0.10)
    ps_b.x[1] = SVector(0.07, 0.07)
    fill!(ps_a.dvdt, zero(SVector{2,Float64}));  ps_a.drhodt .= 0.0
    fill!(ps_b.dvdt, zero(SVector{2,Float64}));  ps_b.drhodt .= 0.0

    _inject_self_grid!(si_self, n_a)
    sweep!(si_self)

    _inject_coupled_grid!(si_coupled, n_a, n_b)
    sweep!(si_coupled)

    @test getindex.(ps_a.dvdt, 1)  == [2.0, 2.0]
    @test ps_a.drhodt      == [2.0, 2.0]
    @test getindex.(ps_b.dvdt, 1)  == [2.0]
    @test ps_b.drhodt      == [2.0]
end
