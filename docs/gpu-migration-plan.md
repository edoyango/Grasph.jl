# GraSPH.jl → GPU (CUDA.jl) migration: status and plan

_Last updated: 2026-08-08. Branch: `onesided-sweep-gpu-prep` (off `main`). Written
because this work is moving from a CPU-only dev machine to one with an NVIDIA
GPU — everything needed to pick the work back up should be in this file._

## Why this migration, and why not an octree

The original ask was to get GraSPH.jl (a Julia SPH code) running on GPUs via
CUDA.jl. The starting hypothesis was that the cell-linked-list (CLL) neighbour
search was GPU-hostile and should be replaced with an octree. That hypothesis
was evaluated and rejected: a uniform grid sized to the kernel cutoff is what
production GPU SPH codes (DualSPHysics, GPUSPH, SPlisHSPlasH) already use,
because constant `h` makes one cell size optimal — no traversal, no pointer
chasing, O(1) neighbour-cell lookup. Octrees earn their cost for long-range
forces (Barnes-Hut gravity) or a genuinely wide dynamic range in smoothing
length; this codebase assumes constant `h` (variable-h is stashed and out of
scope), so neither applies.

| | Uniform grid (kept) | Octree |
|---|---|---|
| Build cost | O(n) counting/radix sort, embarrassingly parallel | O(n log n), recursive, harder to parallelise |
| Neighbour query for fixed-radius SPH | O(1): exactly 9/27 known cells | O(log n) traversal, warp divergence |
| Memory access on GPU | Coalesced (sorted-array slices) | Pointer-chasing unless flattened to LBVH |
| Fit for constant `h` | Optimal | Pure overhead |
| GraSPH's actual situation | Constant `h` | Not applicable |

The real GPU blockers, found by reading the code rather than assuming:

1. **Symmetric pairwise functors** — every `pfn` in `src/PairwiseFunctors.jl` mutated
   *both* `ps.dvdt[i] += t` and `ps.dvdt[j] -= t`. That two-sided write is the sole
   reason a 6/18/9/27-colour graph-colouring scheme existed (to make Polyester
   `@batch` writes race-free).
2. **Comparison-based sort** — `sortperm!` with a custom comparator over
   `SVector{ND,Int}` keys; no GPU radix/bucket sort operates on that.
3. **Sequential CSR build** — `_populate_cells_sorted!` is an inherently serial
   forward-scan + backward-fill.
4. **Host-only structs** — particle-system structs hard-coded `Vector{...}`
   field types; no way to put a `CuArray` in them.

This plan fixes all four while keeping the uniform grid, adopting only the
packed-integer-key idea (not the octree itself) from the original proposal.

## Decisions locked in during the original planning interview

| Question | Decision |
|---|---|
| Octree vs. grid | **Keep uniform grid.** |
| GPU target | Datacenter NVIDIA (A100/H100-class). **Float64 stays.** |
| Kernel layer | **KernelAbstractions.jl** — one source, CUDA backend for the cluster, CPU backend for laptop dev (Apple GPUs have no FP64, so Metal isn't viable regardless). |
| Time-loop residency | Full GPU residency is the eventual goal; **scope is `dambreak.jl` + `dambreak_3d.jl` only** for now. Ghosts, virtual systems, probes, RK4, stress/elasto-plastic are deferred. |
| Pfn contract | **One-sided, register-accumulation.** Thread/particle owns `i`, scans the full 9/27-cell stencil, `pfn` *returns* its contribution, accumulated locally and written once. |
| Coupled/mutual pfns | `is_mutual(pfn)` trait, default `false`; not exercised yet — dambreak's fluid↔`StaticBoundarySystem` coupling is already one-sided (boundary has no dynamics). |
| Neighbour list | On-the-fly 27-cell scan for now; explicit neighbour list is a later optimisation. |
| Sequencing | **Two phases.** Phase A = CPU-only refactor, fully validated. Phase B = KernelAbstractions GPU backend on top of the already-correct Phase A code. |
| Acceptance gate | **Layered**: exact-pair equivalence (~1e-13), short-run trajectory equivalence (~1e-10 over 100–1000 steps), long-run physical invariants. |
| Sort key | **Pack cell coords into `UInt64`**, order-preserving vs. the old key. Morton/Z-order deferred. |
| Old coloured sweep | Kept as a reference oracle (see "deviation" note below — currently still the *production default*, not yet retired). |

## What's actually been built so far

### Phase A — CPU refactor (done, committed)

Two commits already on this branch:

- **`12ac526`** — Pack cell sort key into `UInt64` instead of `SVector{ND,Int}`
  (`src/Sorting.jl`, `test/test_sorting.jl`). Order-preserving, bit-identical
  permutation to the old key — isolates the one-sided rewrite as the *only*
  source of floating-point difference for the equivalence tests below.
- **`16fd88a`** — Add opt-in one-sided particle-parallel sweep
  (`src/Interaction.jl`, `src/PairwiseFunctors.jl`, `src/TimeIntegration.jl`,
  `test/test_onesided_sweep.jl`, `Project.toml`). Adds a new generic function
  `pfn_contribution(pfn, ps, i, j, dx, gx, w) -> NamedTuple` and a
  particle-parallel, full-stencil, register-accumulation sweep
  (`_sweep_self_onesided!`/`_sweep_coupled_onesided!`), gated behind a new
  `onesided::Bool=false` keyword on `SystemInteraction`. Verified to ~1e-13
  against the existing coloured sweep across single-pair, full-sweep 2D/3D
  (self and coupled), 100-step trajectory, and 300-step long-run-invariant
  tests.

  **Deviation from the original plan text**: the plan said to *delete* the
  coloured sweep from `src/` and move it to `test/` as a permanent oracle.
  That was reconsidered mid-implementation: converting every pfn and deleting
  colouring outright would break 12 of the 16 experiment scripts (only
  `FluidPfn`'s self and fluid↔`StaticBoundarySystem` methods were converted,
  matching what `dambreak.jl`/`dambreak_3d.jl` actually use). The user chose
  the additive path — `onesided` defaults to `false`, so every existing script
  is completely unaffected, and only interactions that explicitly opt in use
  the new sweep. The coloured sweep is therefore **still the production
  default**, not yet retired to `test/`.

### Phase B1 — array-type parameterization (done, not yet committed as of this doc's writing)

Uncommitted on this branch at doc-writing time (see "Next steps" for what to
do with it):

- All 5 particle-system structs (`BasicParticleSystem`, `FluidParticleSystem`,
  `StressParticleSystem`, `ElastoPlasticParticleSystem`, `VirtualParticleSystem`)
  now carry their per-particle array fields as type parameters (`VA` for
  `SVector`-valued fields, `SA` for scalar fields, `IA` for `id`, plus `NSA`/`VTA`
  for the stress-system-specific fields) instead of hardcoded `Vector{...}`,
  defaulting to `Vector` via the existing keyword constructors — zero behaviour
  change for every existing script.
- `Adapt.jl` added as a dependency; `Adapt.adapt_structure` methods added for
  all 5 systems plus `StaticBoundarySystem`/`DynamicBoundarySystem` (the
  latter two needed no struct changes — they already just wrap `inner`).
- New test file `test/test_adapt.jl` (42 tests): default-Vector checks, exact
  field round-trips for all 5 systems + both boundary wrappers, and a full
  `sort → grid → sweep → time_integrate!` run on an adapted system to catch
  wiring bugs field-equality alone would miss.
- Full suite: 834/834 passing (792 pre-existing + 42 new). Reduced-scale
  smoke runs of both `dambreak.jl` and `dambreak_3d.jl` shapes complete
  cleanly end-to-end.

  **Scope cut**: `SystemInteraction`'s own `_cell_start`/`_cell_start_a` CSR
  arrays were deliberately *not* made generic here. They're pure per-step
  scratch space rebuilt every timestep by the CPU sweep code; genericizing
  them now would mean threading a new type parameter through ~9 function
  signatures in `Interaction.jl` for no immediate benefit, since nothing can
  exercise a non-`Vector` cell_start until B2's kernels exist to populate/read
  it on GPU. That work is bundled into B2 instead.

  **Important caveat, confirmed by inspection but not empirically tested (no
  GPU in the dev environment this was built in)**: `adapt(CuArray, ps)` should
  succeed by construction (Adapt.jl + CUDA.jl's storage adaptor do the
  `Vector → CuArray` conversion per field; only the struct-level recursion
  needed to be written here, and that's what got tested via the trivial
  `adapt(Array, ps)` path). But **no computation would run yet** — every
  hot-path loop is still either a scalar `for i in 1:n` loop (illegal on a
  `CuArray` — CUDA.jl disallows scalar indexing by default) or uses
  Polyester's `@batch`, which schedules CPU threads only and has no GPU
  dispatch path at all. Concretely, `sort_particles!`,
  `_populate_cells_sorted!`, every sweep variant, `update_state!`, and the
  integrator's `_axpy_ip!`/`_axpy_oop!` all fall in this category. This is
  exactly Phase B2's job.

## Decision made about B2's shape (not yet built)

When B1 finished, two follow-up questions were asked and answered:

1. **Scope for what's next**: land B1 (struct parameterization) and checkpoint
   before B2, rather than doing all of Phase B in one pass. *(This is that
   checkpoint — B1 is done, B2 has not been started.)*
2. **Should B2 unify or stay opt-in?**: **Unify — replace the Polyester
   `@batch` CPU loops with `KernelAbstractions.@kernel` functions everywhere**,
   one code path for both `CPU()` and `CUDABackend()`, matching the original
   plan's literal "unify, delete colouring" language. This is a deliberate
   contrast with how Phase A's one-sided sweep was done (additive/opt-in) —
   the user chose the unify path for B2 specifically, accepting that it
   changes the default execution engine for all 16 experiment scripts at
   once, in exchange for a single code path long-term.

## Next steps, in priority order

1. **Commit the B1 work** sitting uncommitted on this branch (struct
   parameterization + Adapt support + tests) — see this doc's own commit for
   how that was sequenced.
2. **Add `KernelAbstractions.jl` as a dependency**, and `CUDA.jl` (optionally
   `AcceleratedKernels.jl` for scan/sort primitives) as a **package
   extension** (`[weakdeps]`/`[extensions]`) so the base package still
   installs and runs CPU-only when CUDA.jl isn't loaded.
3. **B2 — rewrite the hot-path loops as KA kernels**, replacing the Polyester
   versions outright (per the unify decision above):
   - Sort: pack keys on-device, sort via CUDA.jl's/`AcceleratedKernels.jl`'s
     array sort to get a permutation, then a parallel "detect boundary" kernel
     (`key[i] != key[i-1] ⇒ cell_start[cell(key[i])] = i`) plus a scan-fill for
     empty cells. On `CPU()` backend the same kernel source runs
     sequentially/threaded via KA — this is the laptop-testable path.
   - Reorder: elementwise `arr_new[i] = arr[perm[i]]` — trivial KA kernel.
   - Sweep: the one-sided, full-stencil, particle-per-thread kernel from Phase
     A — already GPU-shaped (no atomics, no colours, one write per thread);
     this is mostly a mechanical port of `_sweep_self_onesided!`/
     `_sweep_coupled_onesided!` into `@kernel` form. **Also make `onesided=true`
     the only mode** (drop the `false` branch and the coloured sweep from
     `src/`, moving it to `test/` as the reference oracle — this is where the
     Phase A deviation above finally gets resolved).
   - State updaters: already a `1 thread : 1 particle` elementwise loop —
     ports with no logic change.
   - Integrator axpy (`_axpy_ip!`/`_axpy_oop!`/`_zero_field`): likely don't
     need hand-written kernels at all — plain broadcast (`@. q += a*dqdt`)
     already dispatches correctly on both `Vector` and `CuArray`.
   - `SystemInteraction`'s `_cell_start`/`_cell_start_a`/`_mingridx`/`_ngridx`
     genericization (deferred from B1) happens here, alongside the kernels
     that consume them. `MVector`-based `_mingridx`/`_ngridx` are tiny and
     `isbits`-incompatible (mutable) — convert to immutable `SVector`/`Tuple`
     when captured by a kernel.
4. **Host-only paths**: `print_summary`/`_scalar_stats` and `write_h5` need an
   explicit `Array(...)`/`adapt(Array, ps)` copy-back before running today's
   CPU logic — both already run at a low, existing cadence, so this adds no
   new per-step cost.
5. **Validate on the GPU machine** — this is the actual point of moving
   machines. Concretely:
   - `Pkg.test()` should still pass 834/834 (nothing here should regress CPU).
   - New tests: CPU-vs-KA-CPU-backend equivalence (near-ulp, laptop-runnable,
     should go in default `test/runtests.jl`), then CUDA-backend equivalence
     (`if CUDA.functional()`-guarded, cluster-only), an `Adapt` round-trip
     test that's finally real (`adapt(Array, adapt(CuArray, ps)) == ps`), and
     an end-to-end reduced-step `dambreak.jl`/`dambreak_3d.jl` parity run,
     CPU vs. GPU, at the tier-7 tolerance (~1e-10) from the original test
     plan.
   - **Measure, don't assume, the actual payoff**: at current problem sizes
     (2,500–30k particles), a datacenter GPU run may be kernel-launch-latency-
     bound rather than throughput-bound. Time `dambreak.jl`/`dambreak_3d.jl`
     end-to-end, CPU vs. GPU, before investing further. If it's a wash, the
     next lever is a Verlet-skin rebuild cadence (rebuild the grid every K
     steps instead of every step) — not built, flagged as a follow-up.

## Explicitly deferred (not started, not part of the current scope)

- Converting the remaining pfns (`StrainRatePfn`, `StrainRateVorticityPfn`,
  `CauchyFluidPfn`, `XSPHPfn`, `InterpolateFieldFn`, `NeighborCountFn`,
  `FluidSolidPfn`) to the one-sided `pfn_contribution` protocol.
  `FluidSolidPfn` and any two-real-system coupling need the `is_mutual` trait
  design (sweep runs a second pass over `system_b`'s particles), which was
  scoped but never implemented.
- Ghost particles, virtual particle systems, probes, RK4 integrator,
  stress/elasto-plastic systems — none have one-sided pfn/sweep support yet.
  Ghosts are used by 8 of 16 experiment scripts (including
  `GranularColumnCollapse3D.jl`, which also has a pre-existing, unrelated
  broken `run_driver!` call signature, found but not fixed this round). Ghosts
  need `generate_ghosts!`'s two-pass count-then-cursor logic rewritten for
  GPU compatibility (flag + exclusive-scan + compaction into
  capacity-preallocated buffers, since per-step `resize!` doesn't work on
  GPU).
- Extending `onesided=true`/GPU support to the other 14 experiment scripts.
- Morton/Z-order sort keys (packed lexicographic `UInt64` key shipped
  instead).
- Explicit neighbour list (on-the-fly 27-cell scan is current approach).
- Float32/mixed precision (Float64 retained per the GPU-target decision).
- Multi-GPU + MPI/ORB integration.

## Practical notes for picking this back up

- Branch: `onesided-sweep-gpu-prep`, based on `main` at commit `37e9a5a`.
  Nothing on this branch has been pushed to any remote.
- Run the full suite with `julia --project -e 'using Pkg; Pkg.test()'` —
  should show `834/834` before any B2 changes begin.
- `test/test_onesided_sweep.jl` and `test/test_adapt.jl` are the two new test
  files from this work; both are already wired into `test/runtests.jl`.
- No CUDA-specific code exists anywhere in this repo yet — `CUDA.jl` is not a
  dependency (not even a weakdep). Adding it as a package extension is the
  first concrete step of B2, not something already in place.
