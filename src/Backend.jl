# ---------------------------------------------------------------------------
# Execution-mode singletons for SystemInteraction sweep dispatch.
#
# ColouredCPU is today's default (Polyester @batch, two-sided pfns).
# OnesidedCPU is the existing onesided=true path (Polyester @batch, one-sided
# pfns). OnesidedKA runs the same one-sided pfns through KernelAbstractions.jl
# kernels, on CPU() or any GPU backend (CUDABackend, etc.) depending on the
# array type of the systems involved. ColouredKA is an internal benchmarking
# spike (see docs/gpu-migration-plan.md): it ports the ColouredCPU
# colour-partitioned half-shell sweep to KernelAbstractions.jl, launching one
# kernel per colour with the same two-sided *mutating* pfn contract as
# ColouredCPU (not pfn_contribution) — answering whether halving the
# arithmetic (as ColouredCPU does over OnesidedCPU) is still a net win once
# it costs 6-27x more kernel launches on a GPU. Not wired into any script;
# reachable only via SystemInteraction's internal `mode` override kwarg.
# NeighbourListKA is a second internal benchmarking spike (see
# docs/gpu-migration-plan.md, "Explicit neighbour list"): it caches an
# explicit, over-inclusive candidate pair list at each Verlet-skin rebuild
# and has the per-step sweep walk that flat list instead of re-deriving
# candidates from the cell grid every step — answering whether removing
# cell-traversal compute (not launches) is worth anything on hardware whose
# cost is dominated by launch count, not compute. Self + one coupled
# (WritesA) shape only, 2D only; not wired into any script; reachable only
# via SystemInteraction's internal `mode` override kwarg.
# ---------------------------------------------------------------------------

abstract type ExecMode end
struct ColouredCPU      <: ExecMode end
struct OnesidedCPU      <: ExecMode end
struct OnesidedKA       <: ExecMode end
struct ColouredKA       <: ExecMode end
struct NeighbourListKA  <: ExecMode end

# ---------------------------------------------------------------------------
# Host copy-back for paths that cannot run on a GPU-resident system: HDF5's C
# API and the sequential scalar-reduction stats in print_summary both need
# plain host arrays. Identity on CPU().
# ---------------------------------------------------------------------------

@inline _to_host(ps) = _to_host(KA.get_backend(ps.x), ps)
@inline _to_host(::KA.CPU, ps) = ps
@inline _to_host(::KA.Backend, ps) = Adapt.adapt(Array, ps)
