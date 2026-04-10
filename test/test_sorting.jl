using Test
using Grasph
using StaticArrays

_make_ps(; n=8, ndims=2) = BasicParticleSystem("test", n, ndims, 1.0, 1.0)
_make_k(; h=0.1, ndims=2)  = CubicSplineKernel(h; ndims=ndims)
_noop(args...) = nothing

# Allocate the shared work buffers that time_integrate! would normally manage.
function _make_sort_bufs(ps::Grasph.AbstractParticleSystem{T,ND}) where {T,ND}
    perm_buf    = Vector{Int}(undef, ps.n)
    key_buf     = Vector{SVector{ND,Int}}(undef, ps.n)
    scratch     = Grasph._make_sort_scratch(ps)
    return perm_buf, key_buf, scratch
end

function _make_sort_bufs_empty(ps::Grasph.AbstractParticleSystem{T,ND}) where {T,ND}
    perm_buf    = Vector{Int}(undef, 0)
    key_buf     = Vector{SVector{ND,Int}}(undef, 0)
    scratch     = Grasph._make_empty_sort_scratch(ps)
    return perm_buf, key_buf, scratch
end

# ---------------------------------------------------------------------------
# _lt_key
# ---------------------------------------------------------------------------

@testset "_lt_key lexicographic comparison" begin
    @test  Grasph._lt_key(SVector(1),    SVector(2))
    @test !Grasph._lt_key(SVector(2),    SVector(1))
    @test !Grasph._lt_key(SVector(1),    SVector(1))

    @test  Grasph._lt_key(SVector(1, 0), SVector(2, 0))
    @test  Grasph._lt_key(SVector(1, 5), SVector(2, 0))   # first coord wins
    @test  Grasph._lt_key(SVector(1, 0), SVector(1, 1))   # tie-break on second
    @test !Grasph._lt_key(SVector(1, 1), SVector(1, 0))
    @test !Grasph._lt_key(SVector(1, 1), SVector(1, 1))

    @test  Grasph._lt_key(SVector(0, 0, 1), SVector(0, 1, 0))
    @test  Grasph._lt_key(SVector(0, 0, 0), SVector(0, 0, 1))
    @test !Grasph._lt_key(SVector(0, 1, 0), SVector(0, 0, 1))
end

# ---------------------------------------------------------------------------
# sort_particles! — 2D BasicParticleSystem
# ---------------------------------------------------------------------------

@testset "sort_particles! 2D" begin

    @testset "particles sorted in row-major cell order" begin
        n, nd = 4, 2
        k  = _make_k(h=0.1)
        ps = _make_ps(n=n, ndims=nd)
        cutoff = k.interaction_length

        ps.x[1] = SVector(0.35, 0.05)   # cell (1, 0)
        ps.x[2] = SVector(0.05, 0.15)   # cell (0, 0)
        ps.x[3] = SVector(0.25, 0.05)   # cell (1, 0)
        ps.x[4] = SVector(0.05, 0.05)   # cell (0, 0)
        # Tag so we can verify rho moved with x after the sort
        ps.rho .= [1.0, 2.0, 3.0, 4.0]
        orig_x = Dict(1.0 => ps.x[1], 2.0 => ps.x[2], 3.0 => ps.x[3], 4.0 => ps.x[4])

        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)

        # Keys must be non-decreasing after sort.
        for i in 1:n-1
            ki = map(v -> floor(Int, v / cutoff), ps.x[i])
            kj = map(v -> floor(Int, v / cutoff), ps.x[i+1])
            @test !Grasph._lt_key(kj, ki)
        end
        # rho must have moved with the same permutation as x.
        for i in 1:n
            @test ps.x[i] == orig_x[ps.rho[i]]
        end
    end

    @testset "all arrays move together (x and rho track the same permutation)" begin
        n, nd = 6, 2
        ps = _make_ps(n=n, ndims=nd)
        cutoff = 0.2

        # Positions in a 3×2 grid of cells, deliberately out of order.
        positions = [SVector(0.45, 0.05), SVector(0.05, 0.25), SVector(0.25, 0.05),
                     SVector(0.05, 0.05), SVector(0.45, 0.25), SVector(0.25, 0.25)]
        ps.x .= positions
        for i in 1:n; ps.rho[i] = Float64(i); end   # tag with original index
        orig_x = Dict(Float64(i) => positions[i] for i in 1:n)

        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)

        # After sort: key[i] ≤ key[i+1]
        for i in 1:n-1
            ki = map(v -> floor(Int, v / cutoff), ps.x[i])
            kj = map(v -> floor(Int, v / cutoff), ps.x[i+1])
            @test !Grasph._lt_key(kj, ki)
        end
        # x and rho must correspond to the same original particle.
        for i in 1:n
            @test ps.x[i] ≈ orig_x[ps.rho[i]]
        end
    end

    @testset "single particle — no-op" begin
        ps = _make_ps(n=1, ndims=2)
        ps.x[1] = SVector(0.5, 0.5)
        ps.rho[1] = 42.0
        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        sort_particles!(ps, 0.2, perm_buf, key_buf, scratch)
        @test ps.x[1] == SVector(0.5, 0.5)
        @test ps.rho[1] == 42.0
    end

    @testset "already-sorted input is unchanged" begin
        n, nd = 4, 2
        ps = _make_ps(n=n, ndims=nd)
        cutoff = 0.2
        # Positions already in sorted order by cell
        ps.x[1] = SVector(0.05, 0.05)
        ps.x[2] = SVector(0.05, 0.25)
        ps.x[3] = SVector(0.25, 0.05)
        ps.x[4] = SVector(0.45, 0.35)
        expected_x = copy(ps.x)

        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        @test ps.x == expected_x
    end

    @testset "already-sorted input takes the fast path (zero allocations)" begin
        n, nd = 6, 2
        ps = _make_ps(n=n, ndims=nd)
        cutoff = 0.2
        # Positions already in sorted cell order
        ps.x[1] = SVector(0.05, 0.05)
        ps.x[2] = SVector(0.05, 0.25)
        ps.x[3] = SVector(0.25, 0.05)
        ps.x[4] = SVector(0.25, 0.25)
        ps.x[5] = SVector(0.45, 0.05)
        ps.x[6] = SVector(0.45, 0.25)

        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        # Warm up to avoid first-call JIT overhead
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        allocs = @allocated sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        @test allocs == 0
    end

    @testset "sort is idempotent (second call leaves order unchanged)" begin
        n, nd = 6, 2
        ps = _make_ps(n=n, ndims=nd)
        cutoff = 0.2
        ps.x[1] = SVector(0.45, 0.05); ps.x[2] = SVector(0.05, 0.25)
        ps.x[3] = SVector(0.25, 0.05); ps.x[4] = SVector(0.05, 0.05)
        ps.x[5] = SVector(0.45, 0.25); ps.x[6] = SVector(0.25, 0.25)

        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        x_after_first = copy(ps.x)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)
        @test ps.x == x_after_first
    end

    @testset "negative coordinates sort correctly" begin
        n, nd = 3, 2
        ps = _make_ps(n=n, ndims=nd)
        cutoff = 0.2
        ps.x[1] = SVector( 0.05,  0.05)
        ps.x[2] = SVector(-0.15, -0.15)
        ps.x[3] = SVector(-0.35,  0.05)
        for i in 1:n; ps.rho[i] = Float64(i); end

        perm_buf, key_buf, scratch = _make_sort_bufs(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)

        for i in 1:n-1
            ki = map(v -> floor(Int, v / cutoff), ps.x[i])
            kj = map(v -> floor(Int, v / cutoff), ps.x[i+1])
            @test !Grasph._lt_key(kj, ki)
        end
    end
end

# ---------------------------------------------------------------------------
# sort_particles! — 3D BasicParticleSystem
# ---------------------------------------------------------------------------

@testset "sort_particles! 3D" begin

    @testset "3D sort order is row-major (first dim slowest)" begin
        n, nd = 4, 3
        ps = BasicParticleSystem("test3d", n, nd, 1.0, 1.0)
        cutoff = 0.2
        # Deliberately out of order
        ps.x[1] = SVector(0.25, 0.05, 0.05)   # key (1,0,0)
        ps.x[2] = SVector(0.05, 0.25, 0.05)   # key (0,1,0)
        ps.x[3] = SVector(0.05, 0.05, 0.25)   # key (0,0,1)
        ps.x[4] = SVector(0.05, 0.05, 0.05)   # key (0,0,0)
        for i in 1:n; ps.rho[i] = Float64(i); end

        perm_buf = Vector{Int}(undef, n)
        key_buf  = Vector{SVector{3,Int}}(undef, n)
        scratch  = Grasph._make_sort_scratch(ps)
        sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)

        # Expected order: (0,0,0), (0,0,1), (0,1,0), (1,0,0)
        keys = [map(v -> floor(Int, v / cutoff), ps.x[i]) for i in 1:n]
        @test keys[1] == SVector(0, 0, 0)
        @test keys[2] == SVector(0, 0, 1)
        @test keys[3] == SVector(0, 1, 0)
        @test keys[4] == SVector(1, 0, 0)
        # rho moved with x
        orig_x = Dict(1.0 => SVector(0.25,0.05,0.05), 2.0 => SVector(0.05,0.25,0.05),
                      3.0 => SVector(0.05,0.05,0.25),  4.0 => SVector(0.05,0.05,0.05))
        for i in 1:n
            @test ps.x[i] ≈ orig_x[ps.rho[i]]
        end
    end
end

# ---------------------------------------------------------------------------
# sort_particles! — GhostParticleSystem (variable-size, empty scratch)
# ---------------------------------------------------------------------------

@testset "sort_particles! GhostParticleSystem" begin

    @testset "ghost sort with empty scratch buffers (resize on demand)" begin
        k    = _make_k(h=0.1)
        ps   = _make_ps(n=8, ndims=2)
        ghost = GhostParticleSystem(ps)

        # Set up source particles near a boundary
        for i in 1:8
            ps.x[i] = SVector(Float64(i) * 0.05, 0.02)
        end
        ps.rho .= 1000.0

        normal = SVector(0.0, 1.0)
        point  = SVector(0.0, 0.0)
        ge = GhostEntry(ghost, normal, point, k.interaction_length)
        generate_ghosts!(ge)

        n_ghost = ghost.n
        @test n_ghost > 0

        perm_buf, key_buf, scratch = _make_sort_bufs_empty(ghost)
        cutoff = k.interaction_length

        # Should not error (buffers resize automatically)
        sort_particles!(ghost, cutoff, perm_buf, key_buf, scratch)
        @test length(perm_buf) >= n_ghost
        @test length(key_buf)  >= n_ghost

        # Result is sorted
        for i in 1:n_ghost-1
            ki = map(v -> floor(Int, v / cutoff), ghost.x[i])
            kj = map(v -> floor(Int, v / cutoff), ghost.x[i+1])
            @test !Grasph._lt_key(kj, ki)
        end
    end

    @testset "idx_original moves with ghost positions" begin
        k    = _make_k(h=0.1)
        ps   = _make_ps(n=6, ndims=2)
        ghost = GhostParticleSystem(ps)

        for i in 1:6
            ps.x[i] = SVector(Float64(i) * 0.04, 0.02)
        end
        ps.rho .= 1000.0

        normal = SVector(0.0, 1.0)
        point  = SVector(0.0, 0.0)
        ge = GhostEntry(ghost, normal, point, k.interaction_length)
        generate_ghosts!(ge)

        # Record which original index each ghost position comes from before sort
        pre_sort_pairs = [(copy(ghost.x[i]), ghost.idx_original[i]) for i in 1:ghost.n]

        perm_buf, key_buf, scratch = _make_sort_bufs_empty(ghost)
        sort_particles!(ghost, k.interaction_length, perm_buf, key_buf, scratch)

        # After sort: (x, idx_original) pairs must still match
        post_sort_pairs = Set([(ghost.x[i], ghost.idx_original[i]) for i in 1:ghost.n])
        @test Set(pre_sort_pairs) == post_sort_pairs
    end
end

# ---------------------------------------------------------------------------
# Consistency with grid alignment
# ---------------------------------------------------------------------------

@testset "sort order consistent with grid cell index" begin
    # After sorting, particles in the same grid cell should be contiguous.
    n, nd = 12, 2
    k  = _make_k(h=0.1)
    ps = _make_ps(n=n, ndims=nd)
    cutoff = k.interaction_length
    si = SystemInteraction(k, _noop, ps)

    # Three clusters of 4 particles each
    ps.x[1:4]  .= [SVector(0.05+i*0.01, 0.05) for i in 0:3]
    ps.x[5:8]  .= [SVector(0.45+i*0.01, 0.05) for i in 0:3]
    ps.x[9:12] .= [SVector(0.05+i*0.01, 0.45) for i in 0:3]

    perm_buf, key_buf, scratch = _make_sort_bufs(ps)
    sort_particles!(ps, cutoff, perm_buf, key_buf, scratch)

    create_grid!(si)

    # All particles in each cell's CSR range must actually map to that cell.
    for c in eachindex(si._cell_start)
        cnt = si._cell_count[c]
        cnt == 0 && continue
        s = si._cell_start[c]
        for j in s:s+cnt-1
            @test Grasph._cell_1idx(ps.x[j], si._mingridx, si._cell_size, si._ngridx, Val{2}()) == c
        end
    end
end
