using Test
using Grasph
using StaticArrays

_make_ps(; n=10, ndims=2) = BasicParticleSystem("test", n, ndims, 1.0, 1.0)
_make_k(; h=0.1, ndims=2)  = CubicSplineKernel(h; ndims=ndims)
_noop(args...)              = nothing

# Count particles in a single cell.
function _cell_count(si, cell)
    si._cell_count[cell]
end

# Count all particles across all cells.
function _total_count(si)
    sum(si._cell_count)
end

# Build a particle-index → cell-index mapping from the system_b (or self) CSR arrays.
function _particle_to_cell(si)
    result = Dict{Int,Int}()
    for c in eachindex(si._cell_start)
        cnt = si._cell_count[c]
        cnt == 0 && continue
        s = si._cell_start[c]
        for j in s:s+cnt-1
            result[j] = c
        end
    end
    result
end

# Build a particle-index → cell-index mapping from the system_a CSR arrays.
function _particle_to_cell_a(si)
    result = Dict{Int,Int}()
    for c in eachindex(si._cell_start_a)
        cnt = si._cell_count_a[c]
        cnt == 0 && continue
        s = si._cell_start_a[c]
        for j in s:s+cnt-1
            result[j] = c
        end
    end
    result
end

@testset "create_grid! self-interaction" begin

    @testset "total particle count in grid equals n" begin
        n, nd = 6, 2
        k  = _make_k()
        ps = _make_ps(n=n, ndims=nd)
        si = SystemInteraction(k, _noop, ps)

        ps.x[1:3] = [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps.x[4:6] = [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        @test _total_count(si) == n
    end

    @testset "mingridx is snapped to a multiple of cutoff at or below min - 2h" begin
        n, nd = 6, 2
        k  = _make_k(h=0.1)
        ps = _make_ps(n=n, ndims=nd)
        si = SystemInteraction(k, _noop, ps)

        ps.x[1:3] = [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps.x[4:6] = [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        cutoff = k.interaction_length
        raw_min = reduce((a,b)->min.(a,b), ps.x) .- 2*cutoff
        # Snapped origin must provide at least the same padding as the raw minimum.
        @test all(si._mingridx .<= raw_min .+ 1e-10)
        # Snapped origin must be a multiple of cutoff (alignment invariant).
        @test all(v -> isapprox(mod(v, cutoff), 0; atol=1e-9) ||
                       isapprox(mod(v, cutoff), cutoff; atol=1e-9),
                  si._mingridx)
    end

    @testset "co-located particles share a cell, two clusters in different cells" begin
        n, nd = 6, 2
        k  = _make_k()
        ps = _make_ps(n=n, ndims=nd)
        si = SystemInteraction(k, _noop, ps)

        # particles 1-3 in one tight cluster, 4-6 in another
        ps.x[1:3] = [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps.x[4:6] = [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        cells = _particle_to_cell(si)
        @test cells[1] == cells[2] == cells[3]
        @test cells[4] == cells[5] == cells[6]
        @test cells[1] != cells[4]
        @test _cell_count(si, cells[1]) == 3
        @test _cell_count(si, cells[4]) == 3
    end

    @testset "grid arrays are populated after create_grid!" begin
        ps = _make_ps()
        si = SystemInteraction(_make_k(), _noop, ps)
        fill!(ps.x, zero(SVector{2,Float64}))
        create_grid!(si)
        @test !isempty(si._cell_start)
        @test !isempty(si._cell_count)
    end

    @testset "repeated create_grid! gives consistent results" begin
        n, nd = 4, 2
        k  = _make_k()
        ps = _make_ps(n=n, ndims=nd)
        si = SystemInteraction(k, _noop, ps)
        ps.x .= [SVector(0.1, 0.1), SVector(0.2, 0.2), SVector(0.3, 0.3), SVector(0.4, 0.4)]

        create_grid!(si)
        count_first = _total_count(si)
        cells_first = _particle_to_cell(si)

        create_grid!(si)
        @test _total_count(si) == count_first
        @test _particle_to_cell(si) == cells_first
    end

end

@testset "create_grid! coupled interaction" begin

    @testset "system_b count placed in grid, system_a count absent" begin
        n_a, n_b, nd = 3, 3, 2
        k    = _make_k()
        ps_a = _make_ps(n=n_a, ndims=nd)
        ps_b = _make_ps(n=n_b, ndims=nd)
        si   = SystemInteraction(k, _noop, ps_a, ps_b)

        # system_a cluster far from system_b cluster
        ps_a.x .= [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps_b.x .= [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        # grid counts system_b particles only
        @test _total_count(si) == n_b
    end

    @testset "system_a particles all assigned to valid cells" begin
        n_a, n_b, nd = 3, 3, 2
        k    = _make_k()
        ps_a = _make_ps(n=n_a, ndims=nd)
        ps_b = _make_ps(n=n_b, ndims=nd)
        si   = SystemInteraction(k, _noop, ps_a, ps_b)

        ps_a.x .= [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps_b.x .= [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        ncells = Int(prod(si._ngridx))
        cells_a = _particle_to_cell_a(si)
        @test length(cells_a) == n_a
        @test all(1 .<= values(cells_a) .<= ncells)
    end

    @testset "mingridx covers both systems and is snapped to a multiple of cutoff" begin
        n_a, n_b, nd = 3, 3, 2
        k    = _make_k()
        ps_a = _make_ps(n=n_a, ndims=nd)
        ps_b = _make_ps(n=n_b, ndims=nd)
        si   = SystemInteraction(k, _noop, ps_a, ps_b)

        ps_a.x .= [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps_b.x .= [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        cutoff = k.interaction_length
        combined_min = min.(reduce((a,b)->min.(a,b), ps_a.x),
                            reduce((a,b)->min.(a,b), ps_b.x))
        @test all(si._mingridx .<= combined_min .- 2*cutoff .+ 1e-10)
        @test all(v -> isapprox(mod(v, cutoff), 0; atol=1e-9) ||
                       isapprox(mod(v, cutoff), cutoff; atol=1e-9),
                  si._mingridx)
    end

    @testset "system_a particles in one cell, system_b particles in different cell" begin
        n_a, n_b, nd = 3, 3, 2
        k    = _make_k()
        ps_a = _make_ps(n=n_a, ndims=nd)
        ps_b = _make_ps(n=n_b, ndims=nd)
        si   = SystemInteraction(k, _noop, ps_a, ps_b)

        ps_a.x .= [SVector(0.05, 0.05), SVector(0.06, 0.08), SVector(0.07, 0.06)]
        ps_b.x .= [SVector(0.35, 0.05), SVector(0.36, 0.08), SVector(0.37, 0.06)]

        create_grid!(si)

        cells_a = _particle_to_cell_a(si)
        # all system_a in one cell
        @test cells_a[1] == cells_a[2] == cells_a[3]
        # system_a's cell has no system_b particles (they are far apart)
        @test _cell_count(si, cells_a[1]) == 0
        # exactly one cell contains all system_b particles
        occupied = findall(c -> _cell_count(si, c) > 0, eachindex(si._cell_start))
        @test length(occupied) == 1
        @test _cell_count(si, occupied[1]) == n_b
        @test occupied[1] != cells_a[1]
    end

    @testset "colocation: system_b particles land in same cell as system_a" begin
        n_a, n_b, nd = 2, 2, 2
        k    = _make_k()
        ps_a = _make_ps(n=n_a, ndims=nd)
        ps_b = _make_ps(n=n_b, ndims=nd)
        si   = SystemInteraction(k, _noop, ps_a, ps_b)

        ps_a.x .= [SVector(0.051, 0.051), SVector(0.06, 0.07)]
        ps_b.x .= [SVector(0.05, 0.06), SVector(0.07, 0.05)]   # same cell as system_a

        create_grid!(si)

        cells_a = _particle_to_cell_a(si)
        @test cells_a[1] == cells_a[2]                    # system_a in one cell
        @test _cell_count(si, cells_a[1]) == n_b          # all system_b in that cell
    end

    @testset "mingridx driven by system_b when it extends further" begin
        n_a, n_b, nd = 2, 2, 2
        k    = _make_k()
        ps_a = _make_ps(n=n_a, ndims=nd)
        ps_b = _make_ps(n=n_b, ndims=nd)
        si   = SystemInteraction(k, _noop, ps_a, ps_b)

        ps_a.x .= [SVector(0.5, 0.5), SVector(0.6, 0.5)]
        ps_b.x .= [SVector(0.1, 0.1), SVector(0.2, 0.2)]   # further down-left than system_a

        create_grid!(si)

        cutoff = k.interaction_length
        combined_min = min.(reduce((a,b)->min.(a,b), ps_a.x),
                            reduce((a,b)->min.(a,b), ps_b.x))
        # system_b drives the minimum; snapped origin must be ≤ the padded minimum.
        @test all(si._mingridx .<= combined_min .- 2*cutoff .+ 1e-10)
        @test all(v -> isapprox(mod(v, cutoff), 0; atol=1e-9) ||
                       isapprox(mod(v, cutoff), cutoff; atol=1e-9),
                  si._mingridx)
        # system_a's minimum must also lie inside the grid.
        ps_a_min = reduce((a,b)->min.(a,b), ps_a.x)
        @test all(si._mingridx .<= ps_a_min)
    end

end

@testset "grid alignment across interactions" begin

    @testset "two self-interactions share cell boundary lattice" begin
        k    = _make_k(h=0.1)
        ps_a = _make_ps(n=4, ndims=2)
        ps_b = _make_ps(n=4, ndims=2)
        si_a = SystemInteraction(k, _noop, ps_a)
        si_b = SystemInteraction(k, _noop, ps_b)

        # Place systems at different positions so they compute different raw origins.
        ps_a.x .= [SVector(0.05, 0.05), SVector(0.07, 0.07),
                   SVector(0.09, 0.05), SVector(0.06, 0.09)]
        ps_b.x .= [SVector(0.53, 0.53), SVector(0.55, 0.57),
                   SVector(0.57, 0.55), SVector(0.54, 0.56)]

        create_grid!(si_a)
        create_grid!(si_b)

        cutoff = k.interaction_length
        # Both mingridx vectors must be multiples of cutoff — same lattice.
        for si in (si_a, si_b), v in si._mingridx
            @test isapprox(mod(v, cutoff), 0; atol=1e-9) ||
                  isapprox(mod(v, cutoff), cutoff; atol=1e-9)
        end
        # The difference of the two origins must also be an integer multiple of cutoff,
        # confirming their cell boundary lattices coincide.
        Δ = si_a._mingridx .- si_b._mingridx
        @test all(v -> isapprox(mod(v, cutoff), 0; atol=1e-9) ||
                       isapprox(mod(v, cutoff), cutoff; atol=1e-9),
                  Δ)
    end

end
