# ---------------------------------------------------------------------------
# KernelAbstractions.jl kernels — one source, dispatched onto KA.CPU() or any
# GPU backend (CUDABackend, etc.) by the array type of the particle systems
# involved. Each kernel here is a mechanical transcription of an existing
# Polyester @batch loop; see the call site (Particles.jl/Interaction.jl) for
# the CPU original it replaces on non-CPU backends.
#
# Every launch below is immediately followed by KA.synchronize — this trades
# some performance (an extra host sync per launch) for a simple, obviously
# correct first pass. Removing synchronization to chain kernels without a
# host round-trip is a deliberate follow-up once correctness is established
# (see docs/gpu-migration-plan.md's benchmark step).
# ---------------------------------------------------------------------------

const _KA_WORKGROUP = 256

@kernel function _update_state_kernel!(ps, fn, dt)
    i = @index(Global, Linear)
    @inbounds fn(ps, i, dt)
end

# ---------------------------------------------------------------------------
# Cell-histogram kernels for the non-CPU branch of _populate_cells_sorted!
# (Interaction.jl). One atomic increment per particle; Atomix works
# identically on KA.CPU() and CUDABackend().
# ---------------------------------------------------------------------------

@kernel function _cell_histogram_kernel_2d!(counts, @Const(x), mingridx, cutoff, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        c = _cell_1idx(x[i], mingridx, cutoff, ngridx, Val{2}())
        Atomix.@atomic counts[c + 1] += 1
    end
end

@kernel function _cell_histogram_kernel_3d!(counts, @Const(x), mingridx, cutoff, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        c = _cell_1idx(x[i], mingridx, cutoff, ngridx, Val{3}())
        Atomix.@atomic counts[c + 1] += 1
    end
end

# ---------------------------------------------------------------------------
# Sort kernels for the non-CPU branch of sort_particles! (Sorting.jl).
# ---------------------------------------------------------------------------

@kernel function _pos_to_key_kernel!(key_buf, @Const(x), cutoff)
    i = @index(Global, Linear)
    @inbounds key_buf[i] = _pos_to_key_gpu(x[i], cutoff)
end

# Fused permutation apply: the whole heterogeneous tuple of per-particle
# arrays is walked per-thread via the same Base.tail recursion _apply_perm!
# uses on CPU, so the entire system reorders in 2 kernel launches (gather,
# then copy-back) instead of 2 per array.
@inline _gather_tuple!(::Tuple{}, ::Tuple{}, i, pi) = nothing
@inline function _gather_tuple!(dsts::Tuple, srcs::Tuple, i, pi)
    @inbounds first(dsts)[i] = first(srcs)[pi]
    _gather_tuple!(Base.tail(dsts), Base.tail(srcs), i, pi)
end

@kernel function _gather_perm_kernel!(scratches, arrs, @Const(perm))
    i = @index(Global, Linear)
    @inbounds pi = perm[i]
    _gather_tuple!(scratches, arrs, i, pi)
end

@inline _copyback_tuple!(::Tuple{}, ::Tuple{}, i) = nothing
@inline function _copyback_tuple!(dsts::Tuple, srcs::Tuple, i)
    @inbounds first(dsts)[i] = first(srcs)[i]
    _copyback_tuple!(Base.tail(dsts), Base.tail(srcs), i)
end

@kernel function _copyback_kernel!(arrs, scratches)
    i = @index(Global, Linear)
    _copyback_tuple!(arrs, scratches, i)
end

# ---------------------------------------------------------------------------
# One-sided, particle-parallel sweep kernels — line-for-line transcriptions of
# _sweep_self_onesided!/_sweep_coupled_onesided! (Interaction.jl), with
# `@batch for i in 1:n` replaced by `@index(Global, Linear)`. The loop nesting
# and iteration order within each thread are preserved exactly, so a thread's
# accumulation is bit-identical to the corresponding Polyester thread's.
#
# mingridx/ngridx must be snapshotted from SystemInteraction's mutable MVector
# fields to immutable SVectors at the launch site — MVector is not isbits and
# cannot cross the kernel boundary. `ps`/`ps_a`/`ps_b` must be `device_view`s,
# not the real particle-system structs (see DeviceViews.jl for why).
# ---------------------------------------------------------------------------

# --- self-interaction ---

@kernel function _sweep_self_onesided_kernel_2d!(ps, pfn, kernel, h, cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_y = ngridx[2]
        val_ndims = Val{2}()
        cell_idx  = _cell_1idx(ps.x[i], mingridx, cutoff, ngridx, val_ndims)
        acc = _onesided_zero_self(pfn, ps, i)
        for dx_cell in -1:1
            neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
            pstart = cell_start[neighbour_cell_idx]
            pend   = cell_start[neighbour_cell_idx + 3]
            for j in pstart:pend-1
                j == i && continue
                acc = _pair_self_onesided!(pfn, ps, i, j, kernel, h, cutoff_sq, val_ndims, acc)
            end
        end
        _onesided_writeback_self!(pfn, ps, i, acc)
    end
end

@kernel function _sweep_self_onesided_kernel_3d!(ps, pfn, kernel, h, cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_z  = ngridx[3]
        n_cells_y  = ngridx[2]
        n_cells_yz = n_cells_y * n_cells_z
        val_ndims  = Val{3}()
        cell_idx   = _cell_1idx(ps.x[i], mingridx, cutoff, ngridx, val_ndims)
        acc = _onesided_zero_self(pfn, ps, i)
        for dx_cell in -1:1
            for dy_cell in -1:1
                neighbour_cell_idx = cell_idx + dx_cell * n_cells_yz + dy_cell * n_cells_z - 1
                pstart = cell_start[neighbour_cell_idx]
                pend   = cell_start[neighbour_cell_idx + 3]
                for j in pstart:pend-1
                    j == i && continue
                    acc = _pair_self_onesided!(pfn, ps, i, j, kernel, h, cutoff_sq, val_ndims, acc)
                end
            end
        end
        _onesided_writeback_self!(pfn, ps, i, acc)
    end
end

_sweep_self_ka!(si::SystemInteraction{T,2}, ::Nothing) where {T} = nothing
_sweep_self_ka!(si::SystemInteraction{T,3}, ::Nothing) where {T} = nothing

function _sweep_self_ka!(si::SystemInteraction{T,2}, pfn::PFN) where {T,PFN}
    ps      = si.system_a
    backend = KA.get_backend(ps.x)
    _sweep_self_onesided_kernel_2d!(backend, _KA_WORKGROUP)(
        device_view(ps), pfn, si.kernel, T(si.kernel.h), si._grid_cutoff[], si._cell_size * si._cell_size,
        si._cell_start, SVector(si._mingridx), SVector(si._ngridx);
        ndrange = ps.n)
    KA.synchronize(backend)
    return nothing
end

function _sweep_self_ka!(si::SystemInteraction{T,3}, pfn::PFN) where {T,PFN}
    ps      = si.system_a
    backend = KA.get_backend(ps.x)
    _sweep_self_onesided_kernel_3d!(backend, _KA_WORKGROUP)(
        device_view(ps), pfn, si.kernel, T(si.kernel.h), si._grid_cutoff[], si._cell_size * si._cell_size,
        si._cell_start, SVector(si._mingridx), SVector(si._ngridx);
        ndrange = ps.n)
    KA.synchronize(backend)
    return nothing
end

# --- coupled interaction ---
#
# No `j == i` guard: ps_a and ps_b are distinct systems (distinct backing
# arrays), so index collision between i and j has no meaning here — matches
# _sweep_coupled_onesided!'s comment in Interaction.jl.

@kernel function _sweep_coupled_onesided_kernel_2d!(ps_a, ps_b, pfn, kernel, h, cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_y = ngridx[2]
        val_ndims = Val{2}()
        cell_idx  = _cell_1idx(ps_a.x[i], mingridx, cutoff, ngridx, val_ndims)
        acc = _onesided_zero_coupled(pfn, ps_a, ps_b, i)
        for dx_cell in -1:1
            neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
            pstart = cell_start[neighbour_cell_idx]
            pend   = cell_start[neighbour_cell_idx + 3]
            for j in pstart:pend-1
                acc = _pair_coupled_onesided!(pfn, ps_a, ps_b, i, j, kernel, h, cutoff_sq, val_ndims, acc)
            end
        end
        _onesided_writeback_coupled!(pfn, ps_a, ps_b, i, acc)
    end
end

@kernel function _sweep_coupled_onesided_kernel_3d!(ps_a, ps_b, pfn, kernel, h, cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_z  = ngridx[3]
        n_cells_y  = ngridx[2]
        n_cells_yz = n_cells_y * n_cells_z
        val_ndims  = Val{3}()
        cell_idx   = _cell_1idx(ps_a.x[i], mingridx, cutoff, ngridx, val_ndims)
        acc = _onesided_zero_coupled(pfn, ps_a, ps_b, i)
        for dx_cell in -1:1
            for dy_cell in -1:1
                neighbour_cell_idx = cell_idx + dx_cell * n_cells_yz + dy_cell * n_cells_z - 1
                pstart = cell_start[neighbour_cell_idx]
                pend   = cell_start[neighbour_cell_idx + 3]
                for j in pstart:pend-1
                    acc = _pair_coupled_onesided!(pfn, ps_a, ps_b, i, j, kernel, h, cutoff_sq, val_ndims, acc)
                end
            end
        end
        _onesided_writeback_coupled!(pfn, ps_a, ps_b, i, acc)
    end
end

_sweep_coupled_ka!(si::SystemInteraction{T,2}, ps_b, ::Nothing) where {T} = nothing
_sweep_coupled_ka!(si::SystemInteraction{T,3}, ps_b, ::Nothing) where {T} = nothing

function _sweep_coupled_ka!(si::SystemInteraction{T,2}, ps_b, pfn::PFN) where {T,PFN}
    ps_a    = si.system_a
    backend = KA.get_backend(ps_a.x)
    _sweep_coupled_onesided_kernel_2d!(backend, _KA_WORKGROUP)(
        device_view(ps_a), device_view(ps_b), pfn, si.kernel, T(si.kernel.h), si._grid_cutoff[], si._cell_size * si._cell_size,
        si._cell_start, SVector(si._mingridx), SVector(si._ngridx);
        ndrange = ps_a.n)
    KA.synchronize(backend)
    return nothing
end

function _sweep_coupled_ka!(si::SystemInteraction{T,3}, ps_b, pfn::PFN) where {T,PFN}
    ps_a    = si.system_a
    backend = KA.get_backend(ps_a.x)
    _sweep_coupled_onesided_kernel_3d!(backend, _KA_WORKGROUP)(
        device_view(ps_a), device_view(ps_b), pfn, si.kernel, T(si.kernel.h), si._grid_cutoff[], si._cell_size * si._cell_size,
        si._cell_start, SVector(si._mingridx), SVector(si._ngridx);
        ndrange = ps_a.n)
    KA.synchronize(backend)
    return nothing
end

# --- coupled interaction, reverse pass (writes into system_b) ---
#
# KA twin of _sweep_coupled_onesided_reverse! (Interaction.jl). Reuses the
# *same* _sweep_coupled_onesided_kernel_2d!/_3d! kernels the forward pass
# uses above — the kernel body only ever refers to its arguments as "ps_a"
# (self/write-target, iterated over) and "ps_b" (read-only neighbour), so
# launching it with system_b/system_a's device views swapped and
# si._cell_start_a (system_a's cell list, already built unconditionally for
# every coupled interaction — see _create_grid_impl!) in place of
# si._cell_start reproduces the reverse pass exactly, with no new kernel
# needed. Mirrors the CPU reverse sweep's own role-swap comment in
# Interaction.jl. ndrange/backend come from ps_b (the iteration domain and
# write target here), matching the forward launcher's convention of driving
# both from the "self" system.

_sweep_coupled_ka_reverse!(si::SystemInteraction{T,2}, ps_b, ::Nothing) where {T} = nothing
_sweep_coupled_ka_reverse!(si::SystemInteraction{T,3}, ps_b, ::Nothing) where {T} = nothing

function _sweep_coupled_ka_reverse!(si::SystemInteraction{T,2}, ps_b, pfn::PFN) where {T,PFN}
    ps_a    = si.system_a
    backend = KA.get_backend(ps_b.x)
    _sweep_coupled_onesided_kernel_2d!(backend, _KA_WORKGROUP)(
        device_view(ps_b), device_view(ps_a), pfn, si.kernel, T(si.kernel.h), si._grid_cutoff[], si._cell_size * si._cell_size,
        si._cell_start_a, SVector(si._mingridx), SVector(si._ngridx);
        ndrange = ps_b.n)
    KA.synchronize(backend)
    return nothing
end

function _sweep_coupled_ka_reverse!(si::SystemInteraction{T,3}, ps_b, pfn::PFN) where {T,PFN}
    ps_a    = si.system_a
    backend = KA.get_backend(ps_b.x)
    _sweep_coupled_onesided_kernel_3d!(backend, _KA_WORKGROUP)(
        device_view(ps_b), device_view(ps_a), pfn, si.kernel, T(si.kernel.h), si._grid_cutoff[], si._cell_size * si._cell_size,
        si._cell_start_a, SVector(si._mingridx), SVector(si._ngridx);
        ndrange = ps_b.n)
    KA.synchronize(backend)
    return nothing
end

# --- write-direction dispatch ---
#
# KA twin of _sweep_coupled_onesided_dispatch! (Interaction.jl, where
# WritesA/WritesB/WritesBoth and _onesided_shape are defined).

_sweep_coupled_ka_dispatch!(::WritesA, si, system_b, pfn) = _sweep_coupled_ka!(si, system_b, pfn)
_sweep_coupled_ka_dispatch!(::WritesB, si, system_b, pfn) = _sweep_coupled_ka_reverse!(si, system_b, pfn)
function _sweep_coupled_ka_dispatch!(::WritesBoth, si, system_b, pfn)
    _sweep_coupled_ka!(si, system_b, pfn)
    _sweep_coupled_ka_reverse!(si, system_b, pfn)
    nothing
end

# ---------------------------------------------------------------------------
# Coloured, two-sided, GPU sweep kernels (ColouredKA — internal benchmarking
# spike, see docs/gpu-migration-plan.md and Backend.jl's ColouredKA comment).
#
# One kernel LAUNCH PER COLOUR, mirroring _sweep_self!/_sweep_coupled!'s host
# colour loop (Interaction.jl) exactly, with `@batch for flat_idx in
# 1:n_active` replaced by `@index(Global, Linear)`. Each kernel body reuses
# _pair_self!/_pair_coupled! (Interaction.jl) verbatim — those are already
# plain, backend-agnostic @inline functions (just array indexing + kernel
# math), so the pair-evaluation logic is bit-identical to the CPU coloured
# sweep, not merely equivalent. The two-sided *mutating* pfn contract
# (`pfn(ps_a, i, j, dx, gx, w)` / `pfn(ps_a, ps_b, i, j, dx, gx, w)`) is used
# directly — no pfn_contribution needed, unlike OnesidedKA.
#
# Safety argument for plain (non-atomic) `+=`/`-=` writes: identical to the
# CPU coloured sweep's — active cells within one colour are separated by
# ≥2/≥3 cells (see the colouring comments above _sweep_self!/_sweep_coupled!),
# which is exactly the write-set-disjointness argument that makes concurrent
# writes safe, independent of whether the concurrent execution substrate is
# Polyester threads or GPU threads. What GPU adds is that colour N+1 must not
# begin until colour N's writes are globally visible — enforced below by
# `KA.synchronize` between every colour's launch (the same conservative
# per-launch-synchronize style used throughout this file).
#
# 2D only (dambreak.jl's shape) — see the plan's scope note; 3D is
# unimplemented on purpose, not an oversight.
# ---------------------------------------------------------------------------

@kernel function _sweep_self_coloured_ka_kernel_2d!(ps, pfn, kernel, h, cutoff, @Const(cell_start), n_cells_y, cell_x_begin, cell_y_begin, n_active_y)
    flat_idx = @index(Global, Linear)
    @inbounds begin
        cutoff_sq = cutoff * cutoff
        val_ndims = Val{2}()
        step_x, step_y = divrem(flat_idx - 1, n_active_y)
        cell_x   = cell_x_begin + step_x * 2
        cell_y   = cell_y_begin + step_y * 3
        cell_idx = (cell_x - 1) * n_cells_y + cell_y

        pstart    = cell_start[cell_idx]
        pend_same = cell_start[cell_idx + 1]
        pend_next = cell_start[cell_idx + 2]

        neighbour_cell_idx = cell_idx + n_cells_y - 1
        neighbour_pstart = cell_start[neighbour_cell_idx]
        neighbour_pend   = cell_start[neighbour_cell_idx + 3]

        for particle_i in pstart:pend_same-1
            for particle_j in particle_i+1:pend_next-1
                _pair_self!(pfn, ps, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
            end
            for particle_j in neighbour_pstart:neighbour_pend-1
                _pair_self!(pfn, ps, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
            end
        end
    end
end

_sweep_self_coloured_ka!(si::SystemInteraction{T,2}, ::Nothing) where {T} = nothing
_sweep_self_coloured_ka!(si::SystemInteraction{T,3}, ::Nothing) where {T} = nothing

function _sweep_self_coloured_ka!(si::SystemInteraction{T,2}, pfn::PFN) where {T,PFN}
    ps         = si.system_a
    backend    = KA.get_backend(ps.x)
    cell_start = si._cell_start
    n_cells_y  = Int(si._ngridx[2])
    n_cells_x  = Int(si._ngridx[1])
    dv         = device_view(ps)
    for colour in 0:5
        cell_x_begin = colour ÷ 3 + 2
        cell_y_begin = colour % 3 + 2
        n_active_x = length(cell_x_begin:2:n_cells_x-1)
        n_active_y = length(cell_y_begin:3:n_cells_y-1)
        n_active_x * n_active_y == 0 && continue
        _sweep_self_coloured_ka_kernel_2d!(backend, _KA_WORKGROUP)(
            dv, pfn, si.kernel, T(si.kernel.h), si._cell_size,
            cell_start, n_cells_y, cell_x_begin, cell_y_begin, n_active_y;
            ndrange = n_active_x * n_active_y)
        KA.synchronize(backend)
    end
    return nothing
end

@kernel function _sweep_coupled_coloured_ka_kernel_2d!(ps_a, ps_b, pfn, kernel, h, cutoff, @Const(cell_start), @Const(cell_start_a), n_cells_y, cell_x_begin, cell_y_begin, n_active_y)
    flat_idx = @index(Global, Linear)
    @inbounds begin
        cutoff_sq = cutoff * cutoff
        val_ndims = Val{2}()
        step_x, step_y = divrem(flat_idx - 1, n_active_y)
        cell_x   = cell_x_begin + step_x * 3
        cell_y   = cell_y_begin + step_y * 3
        cell_idx = (cell_x - 1) * n_cells_y + cell_y

        sys_a_pstart = cell_start_a[cell_idx]
        sys_a_pend   = cell_start_a[cell_idx + 1]
        for particle_i in sys_a_pstart:sys_a_pend-1
            for dx_cell in -1:1
                neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
                sys_b_pstart = cell_start[neighbour_cell_idx]
                sys_b_pend   = cell_start[neighbour_cell_idx + 3]
                for particle_j in sys_b_pstart:sys_b_pend-1
                    _pair_coupled!(pfn, ps_a, ps_b, particle_i, particle_j, kernel, h, cutoff_sq, val_ndims)
                end
            end
        end
    end
end

_sweep_coupled_coloured_ka!(si::SystemInteraction{T,2}, system_b, ::Nothing) where {T} = nothing
_sweep_coupled_coloured_ka!(si::SystemInteraction{T,3}, system_b, ::Nothing) where {T} = nothing

function _sweep_coupled_coloured_ka!(si::SystemInteraction{T,2}, system_b, pfn::PFN) where {T,PFN}
    ps_a         = si.system_a
    backend      = KA.get_backend(ps_a.x)
    cell_start   = si._cell_start
    cell_start_a = si._cell_start_a
    n_cells_y    = Int(si._ngridx[2])
    cutoff       = si._cell_size
    mingridx     = si._mingridx
    cell_x_min = floor(Int, (si._mingridx_a[1] - mingridx[1]) / cutoff) + 1
    cell_x_max = floor(Int, (si._maxgridx_a[1] - mingridx[1]) / cutoff) + 1
    cell_y_min = floor(Int, (si._mingridx_a[2] - mingridx[2]) / cutoff) + 1
    cell_y_max = floor(Int, (si._maxgridx_a[2] - mingridx[2]) / cutoff) + 1
    dv_a = device_view(ps_a)
    dv_b = device_view(system_b)
    for colour in 0:8
        cell_x_begin = colour ÷ 3 + cell_x_min
        cell_y_begin = colour % 3 + cell_y_min
        n_active_x = length(cell_x_begin:3:cell_x_max)
        n_active_y = length(cell_y_begin:3:cell_y_max)
        n_active_x * n_active_y == 0 && continue
        _sweep_coupled_coloured_ka_kernel_2d!(backend, _KA_WORKGROUP)(
            dv_a, dv_b, pfn, si.kernel, T(si.kernel.h), si._cell_size,
            cell_start, cell_start_a, n_cells_y, cell_x_begin, cell_y_begin, n_active_y;
            ndrange = n_active_x * n_active_y)
        KA.synchronize(backend)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Explicit neighbour list — internal benchmarking spike (NeighbourListKA, see
# docs/gpu-migration-plan.md and Backend.jl's NeighbourListKA comment).
#
# Build: two passes per interaction (count, then scatter), each a per-thread
# re-walk of the *same* 9-cell stencil the onesided sweep kernels above
# already use, but filtered at the padded _grid_cutoff radius (an
# over-inclusive superset, valid until the next Verlet-skin rebuild) instead
# of the tight physical cutoff, and with the "call pfn_contribution" step
# replaced by "count a candidate" / "write a candidate index". No atomics
# anywhere: each thread owns exactly one output slot in the count pass (its
# own `nbr_count[i]`) and exactly one exclusive destination range in the
# scatter pass (`nbr_start[i]:nbr_start[i+1]-1`, fixed by the host-side
# `cumsum!` between the two passes) — the CSR/ragged-output analogue of
# GhostParticleSystem's flat flag+cumsum!+scatter GPU pattern below, adapted
# by adding the count pass a flat 1-to-1 compaction doesn't need.
#
# Consume: one flat loop over the cached `nbr_start[i]:nbr_start[i+1]-1`
# range per thread instead of a cell-stencil walk, calling the *same*
# _pair_self_onesided!/_pair_coupled_onesided! (Interaction.jl) the onesided
# kernels above call — these already apply the tight, unpadded
# `r_sq < cutoff_sq` filter internally, so consuming an over-inclusive
# candidate list changes nothing about the physics, only how candidates are
# found. Zero pfn/physics duplication.
#
# Self + one coupled (WritesA) shape only, 2D only — see the plan this
# implements (docs/gpu-migration-plan.md) for why this is scoped as a spike,
# not a production feature.
# ---------------------------------------------------------------------------

@kernel function _nbr_count_self_kernel_2d!(nbr_count, @Const(x), cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_y = ngridx[2]
        val_ndims = Val{2}()
        cell_idx  = _cell_1idx(x[i], mingridx, cutoff, ngridx, val_ndims)
        count = 0
        for dx_cell in -1:1
            neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
            pstart = cell_start[neighbour_cell_idx]
            pend   = cell_start[neighbour_cell_idx + 3]
            for j in pstart:pend-1
                j == i && continue
                dxv = x[i] - x[j]
                dot(dxv, dxv) < cutoff_sq && (count += 1)
            end
        end
        nbr_count[i] = count
    end
end

@kernel function _nbr_scatter_self_kernel_2d!(nbr_idx, @Const(nbr_start), @Const(x), cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_y = ngridx[2]
        val_ndims = Val{2}()
        cell_idx  = _cell_1idx(x[i], mingridx, cutoff, ngridx, val_ndims)
        dest = nbr_start[i]
        for dx_cell in -1:1
            neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
            pstart = cell_start[neighbour_cell_idx]
            pend   = cell_start[neighbour_cell_idx + 3]
            for j in pstart:pend-1
                j == i && continue
                dxv = x[i] - x[j]
                if dot(dxv, dxv) < cutoff_sq
                    nbr_idx[dest] = j
                    dest += 1
                end
            end
        end
    end
end

# No `j == i` guard: ps_a and ps_b are distinct systems, matching the
# coupled onesided kernels' own comment above.
@kernel function _nbr_count_coupled_kernel_2d!(nbr_count, @Const(xa), @Const(xb), cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_y = ngridx[2]
        val_ndims = Val{2}()
        cell_idx  = _cell_1idx(xa[i], mingridx, cutoff, ngridx, val_ndims)
        count = 0
        for dx_cell in -1:1
            neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
            pstart = cell_start[neighbour_cell_idx]
            pend   = cell_start[neighbour_cell_idx + 3]
            for j in pstart:pend-1
                dxv = xa[i] - xb[j]
                dot(dxv, dxv) < cutoff_sq && (count += 1)
            end
        end
        nbr_count[i] = count
    end
end

@kernel function _nbr_scatter_coupled_kernel_2d!(nbr_idx, @Const(nbr_start), @Const(xa), @Const(xb), cutoff, cutoff_sq, @Const(cell_start), mingridx, ngridx)
    i = @index(Global, Linear)
    @inbounds begin
        n_cells_y = ngridx[2]
        val_ndims = Val{2}()
        cell_idx  = _cell_1idx(xa[i], mingridx, cutoff, ngridx, val_ndims)
        dest = nbr_start[i]
        for dx_cell in -1:1
            neighbour_cell_idx = cell_idx + dx_cell * n_cells_y - 1
            pstart = cell_start[neighbour_cell_idx]
            pend   = cell_start[neighbour_cell_idx + 3]
            for j in pstart:pend-1
                dxv = xa[i] - xb[j]
                if dot(dxv, dxv) < cutoff_sq
                    nbr_idx[dest] = j
                    dest += 1
                end
            end
        end
    end
end

# --- build launchers ---
#
# Both self and coupled share si._nbr_start/si._nbr_idx (one SystemInteraction
# is exclusively self or exclusively coupled, never both — the same duality
# _cell_start already has). nbr_start[1] and the +=1 shift are written via
# broadcasting views (not scalar indexing) so this stays allowscalar(false)-safe
# on CuArray. total is pulled back to host the same way GhostParticles.jl's GPU
# path pulls its own compaction total (`Array(view(...))[1]`).

function _build_neighbour_list_self!(si::SystemInteraction{T,2}) where {T}
    ps = si.system_a
    n  = ps.n
    nbr_start = si._nbr_start
    resize!(nbr_start, n + 1)
    if n == 0
        fill!(nbr_start, 1)
        return nothing
    end
    backend   = KA.get_backend(ps.x)
    cutoff    = si._grid_cutoff[]
    cutoff_sq = cutoff * cutoff
    mingridx  = SVector(si._mingridx)
    ngridx    = SVector(si._ngridx)

    _nbr_count_self_kernel_2d!(backend, _KA_WORKGROUP)(
        view(nbr_start, 2:n+1), ps.x, cutoff, cutoff_sq, si._cell_start, mingridx, ngridx;
        ndrange = n)
    KA.synchronize(backend)
    fill!(view(nbr_start, 1:1), 1)
    cumsum!(view(nbr_start, 2:n+1), view(nbr_start, 2:n+1))
    view(nbr_start, 2:n+1) .+= 1

    total = Int(Array(view(nbr_start, n+1:n+1))[1]) - 1
    _resize_scratches!((si._nbr_idx,), total)
    if total > 0
        _nbr_scatter_self_kernel_2d!(backend, _KA_WORKGROUP)(
            si._nbr_idx, nbr_start, ps.x, cutoff, cutoff_sq, si._cell_start, mingridx, ngridx;
            ndrange = n)
        KA.synchronize(backend)
    end
    return nothing
end

function _build_neighbour_list_coupled!(si::SystemInteraction{T,2}, ps_b) where {T}
    ps_a = si.system_a
    n    = ps_a.n
    nbr_start = si._nbr_start
    resize!(nbr_start, n + 1)
    if n == 0
        fill!(nbr_start, 1)
        return nothing
    end
    backend   = KA.get_backend(ps_a.x)
    cutoff    = si._grid_cutoff[]
    cutoff_sq = cutoff * cutoff
    mingridx  = SVector(si._mingridx)
    ngridx    = SVector(si._ngridx)

    _nbr_count_coupled_kernel_2d!(backend, _KA_WORKGROUP)(
        view(nbr_start, 2:n+1), ps_a.x, ps_b.x, cutoff, cutoff_sq, si._cell_start, mingridx, ngridx;
        ndrange = n)
    KA.synchronize(backend)
    fill!(view(nbr_start, 1:1), 1)
    cumsum!(view(nbr_start, 2:n+1), view(nbr_start, 2:n+1))
    view(nbr_start, 2:n+1) .+= 1

    total = Int(Array(view(nbr_start, n+1:n+1))[1]) - 1
    _resize_scratches!((si._nbr_idx,), total)
    if total > 0
        _nbr_scatter_coupled_kernel_2d!(backend, _KA_WORKGROUP)(
            si._nbr_idx, nbr_start, ps_a.x, ps_b.x, cutoff, cutoff_sq, si._cell_start, mingridx, ngridx;
            ndrange = n)
        KA.synchronize(backend)
    end
    return nothing
end

# --- consume: sweep kernels ---

@kernel function _sweep_self_nlist_kernel_2d!(ps, pfn, kernel, h, cutoff_sq, @Const(nbr_start), @Const(nbr_idx))
    i = @index(Global, Linear)
    @inbounds begin
        val_ndims = Val{2}()
        acc = _onesided_zero_self(pfn, ps, i)
        pstart = nbr_start[i]
        pend   = nbr_start[i+1] - 1
        for k in pstart:pend
            j = nbr_idx[k]
            acc = _pair_self_onesided!(pfn, ps, i, j, kernel, h, cutoff_sq, val_ndims, acc)
        end
        _onesided_writeback_self!(pfn, ps, i, acc)
    end
end

@kernel function _sweep_coupled_nlist_kernel_2d!(ps_a, ps_b, pfn, kernel, h, cutoff_sq, @Const(nbr_start), @Const(nbr_idx))
    i = @index(Global, Linear)
    @inbounds begin
        val_ndims = Val{2}()
        acc = _onesided_zero_coupled(pfn, ps_a, ps_b, i)
        pstart = nbr_start[i]
        pend   = nbr_start[i+1] - 1
        for k in pstart:pend
            j = nbr_idx[k]
            acc = _pair_coupled_onesided!(pfn, ps_a, ps_b, i, j, kernel, h, cutoff_sq, val_ndims, acc)
        end
        _onesided_writeback_coupled!(pfn, ps_a, ps_b, i, acc)
    end
end

_sweep_self_nlist!(si::SystemInteraction{T,2}, ::Nothing) where {T} = nothing

function _sweep_self_nlist!(si::SystemInteraction{T,2}, pfn::PFN) where {T,PFN}
    ps      = si.system_a
    backend = KA.get_backend(ps.x)
    _sweep_self_nlist_kernel_2d!(backend, _KA_WORKGROUP)(
        device_view(ps), pfn, si.kernel, T(si.kernel.h), si._cell_size * si._cell_size,
        si._nbr_start, si._nbr_idx;
        ndrange = ps.n)
    KA.synchronize(backend)
    return nothing
end

_sweep_coupled_nlist!(si::SystemInteraction{T,2}, ps_b, ::Nothing) where {T} = nothing

function _sweep_coupled_nlist!(si::SystemInteraction{T,2}, ps_b, pfn::PFN) where {T,PFN}
    ps_a    = si.system_a
    backend = KA.get_backend(ps_a.x)
    _sweep_coupled_nlist_kernel_2d!(backend, _KA_WORKGROUP)(
        device_view(ps_a), device_view(ps_b), pfn, si.kernel, T(si.kernel.h), si._cell_size * si._cell_size,
        si._nbr_start, si._nbr_idx;
        ndrange = ps_a.n)
    KA.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# Ghost kernels — GPU (item 7). GhostParticleSystem's owned arrays are
# capacity-preallocated (see GhostParticles.jl's generate_ghosts! GPU path),
# so every kernel below is bounded by an explicit `n`/`ndrange` argument
# rather than the arrays' own `length`, exactly like every non-CPU kernel
# elsewhere in this file.
# ---------------------------------------------------------------------------

# --- generate_ghosts!: flag + compaction ---
#
# The (boundary, particle) pair space is flattened to one linear index,
# boundary-major then particle-minor (`lin = (b-1)*ps_n + i`), matching the
# CPU version's nested-loop order exactly — so a ghost's k-th slot maps to
# the same source particle/boundary on both backends. `boundaries` is a
# small NTuple of isbits GhostBoundary structs passed by value (not a device
# array) — cheap to broadcast to every thread, and `boundaries[b]` with a
# runtime `b` compiles to ordinary unrolled-select code since NB is a
# compile-time constant (part of the tuple's type). The qualification
# predicate itself (`_ghost_qualifies`, GhostParticles.jl) is shared with the
# CPU generate_ghosts! path so the two backends can never silently drift
# apart on which particles qualify as ghosts.

@kernel function _ghost_flag_kernel!(flags, @Const(x), boundaries, cutoff, ps_n)
    lin = @index(Global, Linear)
    @inbounds begin
        b = (lin - 1) ÷ ps_n + 1
        i = (lin - 1) % ps_n + 1
        qualifies, _ = _ghost_qualifies(boundaries[b], x[i], cutoff)
        flags[lin] = qualifies ? 1 : 0
    end
end

# `offsets` holds the inclusive prefix sum of the same predicate (computed by
# the caller via `cumsum!` on the flags this kernel's sibling produced) — for
# a qualifying pair, `offsets[lin]` IS its final 1-based destination index.
@kernel function _ghost_scatter_kernel!(dst_x, dst_idx_orig, dst_idx_bnd, dst_normals,
                                         @Const(offsets), @Const(x), boundaries, cutoff, ps_n)
    lin = @index(Global, Linear)
    @inbounds begin
        b = (lin - 1) ÷ ps_n + 1
        i = (lin - 1) % ps_n + 1
        bnd = boundaries[b]
        qualifies, da = _ghost_qualifies(bnd, x[i], cutoff)
        if qualifies
            dst               = offsets[lin]
            dst_x[dst]        = x[i] - (2 * da) * bnd.normal
            dst_idx_orig[dst] = i
            dst_idx_bnd[dst]  = b
            dst_normals[dst]  = bnd.normal
        end
    end
end

# --- update_ghost_kinematics!: one thread per ghost ---

@kernel function _ghost_kinematics_kernel!(v_ghost, rho_ghost, @Const(idx_orig), @Const(normals),
                                            @Const(v_real), @Const(rho_real))
    k = @index(Global, Linear)
    @inbounds begin
        normal       = normals[k]
        io           = idx_orig[k]
        v_r          = v_real[io]
        v_ghost[k]   = v_r - 2 * dot(v_r, normal) * normal
        rho_ghost[k] = rho_real[io]
    end
end

# --- GhostCopier field copy: one thread per ghost, one launch per field ---

@kernel function _ghost_copy_field_kernel!(arr, @Const(src_arr), @Const(idx), @Const(normals), mode)
    k = @index(Global, Linear)
    @inbounds arr[k] = _apply_mode(src_arr[idx[k]], mode, normals[k])
end

# ---------------------------------------------------------------------------
# _measure_probes!'s mirror step (TimeIntegration.jl): overwrite a probe's
# positions from its mirror_target's current (post-sort) positions, keyed by
# the target's own stable id. One thread per source particle i, scattering
# into dst_x[src_id[i]] — src_id is always a permutation of 1:n (the
# sort-tracking invariant every system maintains), so every thread writes a
# distinct destination slot and no atomics are needed, same reasoning as
# _ghost_scatter_kernel! above.
# ---------------------------------------------------------------------------

@kernel function _probe_mirror_kernel!(dst_x, @Const(src_x), @Const(src_id))
    i = @index(Global, Linear)
    @inbounds dst_x[src_id[i]] = src_x[i]
end
