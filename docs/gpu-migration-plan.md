# GraSPH.jl → GPU (CUDA.jl) migration: status and plan

_Last updated: 2026-08-10. Branch: `onesided-sweep-gpu-prep` (off `main`). Written
because this work is moving from a CPU-only dev machine to one with an NVIDIA
GPU — everything needed to pick the work back up should be in this file._

_2026-08-09 update: Phase C (below) landed since the previous update —_
_"Next steps" item 4 from the 2026-08-08 revision ("converting the remaining_
_pfns to `pfn_contribution`") is now done, CPU-side only. `ka=true`/GPU_
_support for any of it was explicitly kept out of scope; that's the bulk of_
_what's left, and the priority list at the bottom is rewritten around it._

_2026-08-09 update #2: item 5 (`device_view` for Stress/ElastoPlastic/_
_DynamicBoundary/Virtual systems) is also done now — see the item itself,_
_below, for what shipped, what was scoped out, and a real bug an adversarial_
_review caught and got fixed along the way._

_2026-08-09 update #3: item 6 (KA kernel twin for the reverse/`WritesBoth`_
_sweep) is done — see the item itself, below. This landed back on the RTX_
_4060 Laptop GPU machine from Phase B2 (confirmed via `nvidia-smi`/_
_`CUDA.functional()`, both true this session — Phase C and item 5 were done_
_CPU-only in between), so this is the first item validated against real CUDA_
_hardware since Phase B2's own crossover benchmark. Testing it found a real,_
_previously-unreachable dispatch gap in `FluidPfn`'s fluid-fluid method,_
_fixed as a direct follow-up — see update #4._

_2026-08-09 update #4: the `FluidPfn` fluid-fluid dispatch gap from update #3_
_is fixed — `DeviceSystem` (`src/DeviceViews.jl`) gained a phantom `Kind`_
_type parameter so `device_view` no longer erases which concrete system_
_produced a view. Chosen via an independent 3-way design panel, landed after_
_an adversarial review pass. `FluidSolidPfn`'s identical, separately-tracked_
_gap (item 8) is not fixed by this but now has a two-line-signature pattern_
_to follow. See item 6's rewritten note, below, for the full story._

_2026-08-09 update #5: `FluidSolidPfn`'s gap (item 8, `DambreakWall.jl`'s_
_fluid/wall coupling) is also fixed now, same session, reusing the `Kind`_
_mechanism from update #4 — two `DeviceSystem{T,ND,Kind}` methods instead of_
_one, since this pfn's physics is asymmetric (the fluid's own pressure_
_drives both sides' pressure term, unlike `FluidPfn` fluid-fluid's symmetric_
_case). See item 8's rewritten note, below (commit `0fac396`). This update_
_also corrects a stale suite count this doc previously cited for item 6:_
_"1475/1475" predated that item's own last test addition (the adversarial_
_review's 3D-coverage-gap fix); the correct figure, already used in that_
_fix's commit message, was 1479/1479. It's now 1496/1496 after this item._

_2026-08-09 update #6: item 7 ("Ghosts on GPU," previously the last_
_completely-unstarted item and this doc's own "hardest remaining piece") is_
_done — see the item itself, below (commit `eda4234`). `GhostParticleSystem`_
_gained array-type-generic owned fields, a capacity-vs-logical-count split_
_for the GPU backend, `Adapt.jl`/`device_view` support, and real GPU kernels_
_for `generate_ghosts!`/`update_ghost_kinematics!`/`GhostCopier`. A real bug_
_(a capacity-growth check that missed the `extras` arrays' independent_
_starting size) was found and fixed via real-CUDA-hardware testing, and a_
_load-bearing fix outside `GhostParticles.jl` itself was needed too: grid_
_building used to derive its particle count from `length(x)`, silently wrong_
_once a ghost's capacity can exceed its count — see the item's own note for_
_the full story, including what a 3-reviewer adversarial pass found and_
_closed. Suite: 1600/1600 (up from 1496)._

_2026-08-09 update #7: item 8's remaining scope (actually wiring_
_`onesided=true`/`ka=true` into the 11 non-dambreak scripts) is partly done —_
_5 of the 11 (`ellipse.jl`, `DambreakWall.jl`, `GranularColumnCollapse.jl`,_
_`GranularColumnCollapse3D.jl`, `EP_ColumnCollapse.jl`) now have the_
_`GRASPH_BACKEND` switch, verified on real CUDA hardware. The other 6_
_(`bubble.jl`/`bubble2.jl`/`bubble3.jl`, `EP_ColumnCollapse2.jl`,_
_`Trapdoor.jl`, `CantileverBeam.jl`) turn out to be blocked by item 9, not_
_item 8 — they use `RK4TimeIntegrator`, `VirtualParticleSystem`, or_
_`ProbeParticleSystem`, none of which have GPU-resident orchestration yet,_
_confirmed directly this session rather than assumed. See item 8's own note_
_for the full breakdown. No new tests; suite stays 1600/1600._

_2026-08-09 update #8: item 9 (Virtual/probe/RK4 GPU-orchestration gaps) is_
_done — see the item itself, below. All three gaps turned out to be small,_
_targeted fixes once traced to source, not the "no GPU sweep path at all"_
_scope the item's original wording suggested — RK4's was a 2-line buffer-type_
_fix, Virtual's was one scalar loop, and `ProbeParticleSystem` got the same_
_array-type-generic + `device_view` treatment items 5/7 already gave_
_Virtual/Ghost. A real bug (a broken `Ref`-less broadcast in the new_
_`_axpy_const_ip!` primitive) was caught only by real CUDA hardware — no_
_`KA.CPU()` test can distinguish this class of bug, since position-advance_
_backend-dispatches on the array's own type, not on `ka=true`. The 6 scripts_
_item 8 found blocked by this item are now unblocked at the infrastructure_
_level but still not wired — that remains a follow-up to item 8. Suite:_
_1651/1651 (up from 1600)._

_2026-08-10 update #9: item 10 (wiring the last 6 experiment scripts) is_
_done — see the item itself, below (commit `149c238`). All 13 experiment_
_scripts now support `GRASPH_BACKEND=cuda`. Two aliasing idioms already_
_established by items 7-9 (adapt a self-referencing wrapper as one unit,_
_pull the canonical GPU-resident source back out via `getfield`) covered_
_most of it; `Trapdoor.jl`'s two virtual systems sharing one physical source_
_across a two-stage run needed a third idiom (rebuild the second wrapper_
_directly around the first's already-adapted source) that surfaced a real,_
_previously-unreachable bug — `VirtualParticleSystem`'s keyword constructor_
_hardcoded `Vector` for `w_sum` regardless of the source's own array type,_
_the same buffer-type bug class item 9 already fixed once for RK4. Fixed,_
_with a regression test. `CantileverBeam.jl` also needed two pre-existing,_
_unrelated broken calls (documented but never fixed back in Phase C) fixed_
_to actually run. Suite: 1653/1653 (up from 1651)._

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
  colouring outright would break 11 of the repo's 13 experiment scripts (only
  `FluidPfn`'s self and fluid↔`StaticBoundarySystem` methods were converted,
  matching what `dambreak.jl`/`dambreak_3d.jl` actually use). The user chose
  the additive path — `onesided` defaults to `false`, so every existing script
  is completely unaffected, and only interactions that explicitly opt in use
  the new sweep. The coloured sweep is therefore **still the production
  default**, not yet retired to `test/`.

### Phase B1 — array-type parameterization (done, committed as `71b4160`)

_Correction: this section originally described B1 as uncommitted. It was in
fact committed (`71b4160`, alongside this doc's own commit `e1d2991`) by the
time this doc was first read on the GPU machine — so the "commit the B1 work"
item that used to head the next-steps list below was already done and B2
picked up directly at "add KernelAbstractions.jl."_

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
   changes the default execution engine for all of the repo's 13 experiment
   scripts at once, in exchange for a single code path long-term.

   **Revised when B2 was actually built** (see "Phase B2" section below):
   unifying immediately turned out to contradict the still-deferred pfn
   conversion (deleting the coloured sweep before every pfn has a
   `pfn_contribution` method would break 11 of 13 scripts today), so the
   coloured sweep stays in `src/` as the default and `onesided=true`/`ka=true`
   remain opt-in, same as Phase A. Full unification is still the eventual
   goal; it happens once pfn conversion (still not started) is done.

## Phase B2 — KernelAbstractions GPU backend (done, this session)

_Added 2026-08-08, on the machine this doc's "picking this back up" section
pointed to. This is that machine: RTX 4060 Laptop GPU (sm_89, 8 GB), Julia
1.12.5, CUDA.jl 6.2.1 (resolved to 5.8.5 alongside the rest of the
dependency graph — see "environment notes" below), KernelAbstractions.jl
0.9.42, 16 CPU cores._

**Hardware reality check, before anything else**: this is a consumer Ada
Lovelace card, not the datacenter A100/H100 the "GPU target" decision above
assumed. Measured on this machine: Float64 FMA throughput is 0.138 TFLOP/s on
the GPU vs. 0.143–0.311 TFLOP/s on the 16-core CPU (**no FP64 compute
advantage**; FP32 is 35× faster than FP64 on this card). The GPU's only real
edge is bandwidth (216 GB/s vs. ~50–80 GB/s) and it pays a flat ~8.3 µs per
kernel launch regardless of size. Conclusion, refined by the actual
crossover benchmark (see "Next steps" below, now done): at `dambreak.jl`'s
own production scale (2,500 particles) this machine is indeed the
**correctness/parity target**, not a performance win — the GPU is 3.6×
slower there. But it is *not* uniformly performance-irrelevant: the
benchmark found a real, bandwidth/launch-amortization-driven crossover
around 40,000 particles, past which the GPU wins by a growing margin. So the
"measure, don't assume" posture holds at the scale actually shipped, but
doesn't generalize to "this GPU is never worth it" — that would have been
the wrong conclusion to draw from the small-scale-only smoke test.

_(Update, item 13: this whole paragraph is laptop-specific and does not hold
on server hardware. On a real datacenter GPU — A100-PCIE-40GB — there is a
large FP64 compute edge over the host CPU, ~38× the laptop GPU's own
measured throughput, on top of the bandwidth edge. See item 13 for the
numbers and for how items 11/12's conclusions do, and don't, change as a
result.)_

**Scope**: a vertical slice — `dambreak.jl` (2D) fully GPU-resident
end-to-end (sort → grid → sweep → state update → integrator), gated behind
`GRASPH_BACKEND=cuda` (default `cpu`, zero behaviour change otherwise).
`dambreak_3d.jl` got the same wiring in a follow-up pass (see below) — the
2D/3D histogram, sort-key, and sweep kernels were written side by side from
the start and only needed the script-level switch and a validation pass. The
other 12 experiment scripts and full pfn conversion are still out of scope —
see "Explicitly deferred" below, which is mostly unchanged.

**Key design decision that revised the original B2 text**: the coloured
sweep **stays in `src/`** and remains every script's default. The original
"make `onesided=true` the only mode, drop the coloured sweep" plan
contradicted its own deferral of pfn conversion — only `FluidPfn` self and
`FluidPfn`↔`StaticBoundarySystem` have `pfn_contribution`, so deleting the
coloured sweep would have broken 11 of 13 scripts immediately. It retires
once every pfn is converted; that conversion work never started.

### What got built

- **`device_view(ps)`** (`src/DeviceViews.jl`, new) — the load-bearing piece.
  `adapt(CuArray, ps)` alone is *not* sufficient to pass a system into a
  `@kernel` function: `name::String`/`_print_fields::Vector{Symbol}` make the
  real structs non-`isbits`, and GPUCompiler rejects non-isbits kernel
  arguments unconditionally, whether or not the field is used — verified by
  reproducing the failure directly. `device_view` builds a `NamedTuple`-backed
  proxy carrying only the arrays a kernel needs, subtyping
  `AbstractParticleSystem{T,ND}` so every existing `pfn_contribution`/state-
  updater method works against it with **zero changes to `PairwiseFunctors.jl`
  or `StateUpdaters.jl`**.
- **`ExecMode`** (`src/Backend.jl`) — `SystemInteraction`'s old `ONESIDED::Bool`
  type parameter became `MODE<:ExecMode` (`ColouredCPU`/`OnesidedCPU`/
  `OnesidedKA`), plus a new `CSA<:AbstractVector{Int}` parameter so
  `_cell_start`/`_cell_start_a` follow `system_a`'s array type
  (`similar(system_a.x, Int, 0)`). Public API unchanged: `onesided::Bool` kwarg
  still works as before, plus a new `ka::Bool` kwarg (requires `onesided=true`).
- **Sweep kernels** (`src/KAKernels.jl`) — line-for-line transcriptions of
  `_sweep_self_onesided!`/`_sweep_coupled_onesided!`, `@batch for i in 1:n`
  replaced by `@index(Global, Linear)`, loop nesting and iteration order
  preserved exactly so a KA-`CPU()` thread's accumulation is bit-identical to
  the Polyester original's. `_mingridx`/`_ngridx` (mutable `MVector`, not
  isbits) are snapshotted to `SVector` at each launch site — no signature
  changes to `_cell_1idx` were needed, since it already took `AbstractVector`.
- **GPU grid build** (`src/Interaction.jl`) — the bounding-box scalar loop
  became a `mapreduce` with an explicit `init` (an unseeded `minimum` errors
  on `SVector` — no `typemax(SVector)` method — and this form works
  identically on `Vector`/`CuArray`, bit-for-bit, since min/max have no
  rounding error). The sequential forward-scan + backward-fill CSR build
  (`_populate_cells_sorted!`) got a second, backend-dispatched implementation
  for non-CPU backends: an `Atomix`-based histogram + in-place `cumsum!` +
  `+1`, verified to produce an **integer-identical** `cell_start` to the
  serial version (both compute `cell_start[c] = 1 + Σ_{k<c} count[k]`). The
  CPU path is untouched.
- **GPU sort** (`src/Sorting.jl`) — key generation and the permutation-apply
  gather/copy-back got KA-kernel GPU branches (CPU paths untouched, including
  the `InsertionSort`-on-near-sorted-data fast path). The already-sorted
  early-exit is CPU-only — it's a serial scan, and on GPU the D2H sync it'd
  need to branch on costs more than the sort it might save. **Verified
  CUDA.jl's `sortperm!` is stable and produces a bit-identical permutation to
  Base's stable sort** (checked against 20k keys with heavy duplication), so
  no custom counting/radix sort or extra dependency (`AcceleratedKernels.jl`)
  was needed — contrary to what was assumed going in.
- **Host-copy boundaries** — `print_summary`/`write_h5` call sites in
  `_maybe_print!`/`_maybe_save!` wrapped in a new `_to_host(ps)` (identity on
  `CPU()`, `adapt(Array, ps)` otherwise); both already run at a low, existing
  cadence.
- **`dambreak.jl`** gained a `GRASPH_BACKEND` env-var switch (default `cpu`,
  unchanged behaviour) that adapts `fluid`/`boundary` to `CUDABackend()` and
  passes `onesided=ka_mode, ka=ka_mode` to both interactions.
- **`_xsph_correction!`** is now skipped entirely when no interaction has a
  velocity-adjust pfn (`dambreak.jl` has none) — it was previously 6 wasted
  kernel launches of pure no-op every step.

**A real GPU-compilation blocker found and fixed**: `_pos_to_key`'s range
check throws a formatted `ArgumentError` on an out-of-range cell coordinate.
GPUCompiler rejects *any reachable* kernel code path needing dynamic string
construction, even a branch that's never taken for valid input — confirmed by
reproducing the exact failure ("unsupported call to a lazy-initialized
function", from `string(c)` inside the error path). Fixed with a GPU-only,
non-throwing `_pos_to_key_gpu`/`_cellcoord_to_field_gpu` used solely inside the
key-generation kernel; the CPU path (and its `@test_throws` test) is
untouched. This is exactly the risk a prior design pass flagged as needing
isolated testing before building on top of it — worth remembering for any
similar `@noinline`-throw helper considered for a future kernel.

**Validation performed** (manual scripts during development; formal
`test/` additions are the next step — see below): the sweep kernels and
state-update kernel were checked bit-exact against the Polyester originals on
KA's `CPU()` backend, then re-checked on real CUDA hardware — self-sweep,
coupled-sweep (fluid↔`StaticBoundarySystem`), and the state-update kernel all
agreed with the CPU oracle to ~1–3×10⁻¹⁶ relative (pure ulp noise from NVPTX's
FMA contraction and device `pow` vs. glibc's). The **full** GPU pipeline —
`sort_particles!` → `create_grid!` → `sweep!`, all CUDA-resident — was then
validated end-to-end: `cell_start`, `mingridx`, `ngridx` came back
integer-identical to the CPU build, and dvdt matched to ~2×10⁻¹⁶ relative
after reordering both results by particle `id`. Finally, `dambreak.jl` itself
was run with `GRASPH_BACKEND=cuda` through the real `run_driver!`/CLI path
(not a hand-rolled script) for a few steps: velocities matched the expected
`g·t` free-fall exactly, densities stayed at `rho0`, no NaNs. `Pkg.test()`
stayed at 834/834 throughout every step of this work.

## Phase C — pairwise-functor conversion to `pfn_contribution` (done, CPU-only)

_Landed 2026-08-09, on the CPU-only dev machine (no GPU work in this phase —_
_see "what this does NOT do" below). This is "Next steps" item 4 from the_
_previous revision of this doc, plus a bug fix and a verification pass that_
_weren't originally scoped but turned out to be necessary along the way._

**Why this was next**: the coloured sweep can only retire once every pfn
actually used by the 13 experiment scripts has a `pfn_contribution` method.
Before this phase, only `FluidPfn`'s self-interaction and its
`StaticBoundarySystem`-coupled method were converted — the minimum
`dambreak.jl`/`dambreak_3d.jl` needed. Everything else
(`StrainRatePfn`, `StrainRateVorticityPfn`, `CauchyFluidPfn`, `XSPHPfn`,
`InterpolateFieldFn`, `NeighborCountFn`, `FluidSolidPfn`, plus `FluidPfn`'s
remaining variants) was still coloured-sweep-only, which is what kept the
other 11 scripts permanently off the one-sided path.

**Scope note, unchanged from Phase A/B2's own framing**: this is CPU
(`onesided=true`) only. `ka=true`/GPU support for any pfn converted here was
explicitly deferred — see "What this does NOT do" below and the revised
"Next steps" list.

### What got built, in commit order (`bbf6283` → `d548e06`)

- **Bucket A** (`bbf6283`) — mechanical one-sided transcriptions for pfns
  that were already one-sided in their mutating form (only ever wrote
  `ps_a`'s fields): `StrainRatePfn`/`StrainRateVorticityPfn` (self +
  ghost/virtual-coupled + `DynamicBoundarySystem`-coupled),
  `CauchyFluidPfn` (same three shapes, general two-real-system method left
  unconverted and narrowed defensively — confirmed dead via grep),
  `XSPHPfn` (self only at this point), `FluidPfn`'s remaining two one-sided
  variants. Also generalized `_onesided_writeback_self!`/
  `_onesided_writeback_coupled!` to one NamedTuple-name-dispatched method
  for every pfn, replacing per-pfn hand-written versions.
- **Reverse-sweep infrastructure** (`0c29f6d`, `301a886`) — new sweep pass
  mirroring `_sweep_coupled_onesided!`'s shape but iterating `system_b`
  instead of `system_a`, scanning `_cell_start_a` (already built
  unconditionally for every coupled interaction) for `system_a` neighbours,
  calling `pfn_contribution(pfn, system_b, system_a, j, i, -dx, -gx, w)` —
  i.e. **the "ps_a" argument slot is always the write target and "ps_b" is
  always the read-only neighbour, regardless of which physical system fills
  which slot or which pass is running.** Added the `_onesided_shape(pfn,
  ps_a, ps_b)` trait (`WritesA()` default / `WritesB()` / `WritesBoth()`)
  that picks which pass(es) run. This is the one genuinely new piece of
  sweep machinery in Phase C — everything else is either a mechanical
  transcription or a direct consumer of this.
- **Bucket B** (`5f46096`) — `InterpolateFieldFn`/`NeighborCountFn` →
  `WritesB()`. Both already wrote into `ps_b` in their mutating form, so
  this was a direct application of the reverse-sweep infra with zero script
  changes.
- **Bucket C** (`90bbd44`) — the genuinely mutual pfns, `WritesBoth()`:
  `FluidPfn` fluid-fluid (two distinct real `FluidParticleSystem` instances,
  `bubble.jl`/`bubble2.jl`/`bubble3.jl`) got **one** `pfn_contribution`
  method serving both pass directions, since the physics is symmetric under
  relabeling. `FluidSolidPfn` (`DambreakWall.jl`) got **two** distinct,
  narrowly-typed methods (fluid-as-target / solid-as-target) with no
  generic fallback, because its physics is deliberately *not* symmetric —
  the fluid's own pressure drives the pressure-force term on both sides, to
  keep pressure continuous across the interface. Verified via mutation
  testing (injecting the wrong-side-pressure bug and confirming a dedicated
  regression test catches it).
- **`XSPHPfn` ghost-coupling bug fix** (`91d9016`) — not originally scoped;
  found by an adversarial review while converting `XSPHPfn`'s coupled form.
  Every real `GhostParticleSystem` in this codebase self-references its
  source (`GhostParticleSystem(fluid_X, ...)`), and `GhostParticleSystem`
  doesn't own a `v_adjustment` array, so `XSPHPfn`'s old, fully-symmetric
  mutating method's `ps_b.v_adjustment[j] -= ...` write fell through
  `getproperty` straight into the real system's own array — aliased, and
  indexed by the ghost's *local* index rather than the particle it mirrors.
  Silently wrong whenever `ghost.n < fluid.n`; heap-corrupting
  (SIGABRT/SIGSEGV, reproduced directly) whenever `ghost.n > fluid.n` — a
  regime `bubble3.jl`'s actual `fluid_boundary_interaction` can hit. This
  predates Phase C entirely; it was just never exercised as a "coloured
  sweep is the oracle" comparison before. Fixed by adding a narrowly-typed
  one-sided mutating method (matching the `FluidPfn`/`CauchyFluidPfn`/
  `StrainRatePfn` ghost/virtual precedent) that only ever writes `ps_a`.
- **Integration harnesses** (`d548e06`) — the pairwise-comparison tests
  above only ever proved single-pair or single-sweep equivalence. This adds
  one reduced-scale standalone test per distinct interaction shape across
  all 11 non-dambreak scripts, each running a *real* multi-stage
  `LeapFrogTimeIntegrator`/`RK4TimeIntegrator` loop (with ghosts/virtuals/
  probes wired up exactly as the real script does) once coloured and once
  `onesided=true`, checked for both short-run trajectory equivalence and
  long-run stability. Also confirmed `CantileverBeam.jl` is broken as
  committed (invalid `CubicSplineKernel`/`SystemInteraction` calls,
  unrelated to this work, not fixed) — its harness reconstructs the
  intended shape with corrected calls instead.

Full suite: **1371/1371** (up from 935 before Phase C; +100 pfn-conversion
tests, +8 for the `XSPHPfn` fix, +311 for the integration harnesses, +17
from other test additions along the way).

### What this does NOT do

- **No `ka=true`/GPU support for any pfn converted in Phase C.** Every
  `pfn_contribution` method added here runs on CPU (`onesided=true`) only.
  `device_view`/`DeviceViews.jl` still only covers `BasicParticleSystem`/
  `FluidParticleSystem` (the two types `dambreak.jl` needs) — none of
  `StressParticleSystem`, `ElastoPlasticParticleSystem`,
  `VirtualParticleSystem`, `ProbeParticleSystem`, `DynamicBoundarySystem`,
  or ghost systems have an isbits GPU-kernel-safe proxy.
- **No KA kernel twin for the reverse/`WritesBoth` sweep.** The new
  `_sweep_coupled_onesided_reverse!` pass and the `WritesB()`/`WritesBoth()`
  dispatch only exist as Polyester CPU code in `Interaction.jl`; nothing in
  `KAKernels.jl` mirrors it yet.
- **The other 12 scripts are still on the coloured sweep by default.**
  Phase C proved `onesided=true` is *correct* for their interaction shapes
  (that's what the integration harnesses are for); it didn't flip any
  script's actual default, and none of them gained a `GRASPH_BACKEND`
  switch. `dambreak.jl`/`dambreak_3d.jl` remain the only two scripts wired
  to `ka=true` at all.
- **Ghosts, virtual systems, probes, and RK4 still have no GPU story.**
  Phase C's reduced-scale harnesses run them on CPU inside a real integrator
  loop, which proves onesided-sweep correctness but says nothing about
  GPU-residency — `generate_ghosts!`'s serial count-then-cursor algorithm in
  particular doesn't port by simple translation (see "Explicitly deferred"
  below, unchanged on this point).

The coloured sweep has **not** been retired from `src/` — that was always
staged for after every pfn was converted, and Phase C is that "every pfn"
milestone reached (for actual usage; a few narrowly-typed defensive
fallbacks exist for combinations grep confirmed are unused, e.g.
`CauchyFluidPfn`'s general two-real-system method). Retiring it is now
unblocked on the CPU side but wasn't done in this phase — it still defaults
to coloured for all 13 scripts, and no script's behavior changed.

### Environment notes for the next machine move

- A fresh checkout needs `Pkg.Registry.update()` before `Pkg.instantiate()`/
  `Pkg.test()` — the committed `Manifest.toml` pins `julia_version = "1.12.5"`
  (this doc previously miscited this as `1.12.6` — corrected, item 13)
  and an `Adapt` version the local registry cache didn't have yet, producing
  an "Unsatisfiable requirements" error that looks like a real dependency
  conflict but is just a stale registry.
- `CUDA` is a **test-only** dependency (`test/Project.toml`, not a root
  `[weakdeps]`/`[extensions]` entry — nothing in `src/` needed extension-only
  code; `get_backend`, `Adapt` storage rules, `sortperm!`, `cumsum!`, and
  `Atomix`'s CUDA support are all provided directly by CUDA.jl). Don't pin a
  `CUDA` version in `test/Project.toml`: CUDA 6.2's `CUDATools` pulls a
  `PrettyTables` version incompatible with this package's own `PrettyTables =
  "2"` compat bound. Leaving it unpinned lets the resolver settle on CUDA
  5.8.5, which works fine and has no such conflict.
- **Compute nodes with no outbound internet** (confirmed on NCI Gadi, item
  13's H200 run): do all `Pkg.Registry.update()`/`instantiate`/`Pkg.add`
  work on a login node first, sharing the same NFS-mounted `~/.julia` depot
  the compute node will see — nothing further needs the network once
  packages/artifacts are cached there. Separately: a bare `ssh host "julia
  --project=... script.jl"` runs in the SSH login shell's default directory
  (`$HOME`), not wherever an interactive session last `cd`'d — use
  fully-absolute `--project=`/script paths for any one-shot remote command,
  or the failure looks like a broken environment when it's actually just the
  wrong `pwd`.
- **CUDA 13 dropped Volta (`sm_70`) `ptxas` support** (confirmed on NCI
  Gadi's V100 nodes, item 13's V100 run): a shared `~/.julia` depot that
  already resolved `CUDA_Runtime_jll` to 13.x while setting up a newer GPU
  (Ampere/Hopper) will fail on Volta with `ptxas fatal: Value 'sm_70' is
  not defined for option 'gpu-name'` the first time a KA kernel compiles.
  Fix on a login node (needs internet to fetch the older toolkit):
  `julia --project=<env> -e 'using CUDA; CUDA.set_runtime_version!(v"12.6")'`
  — 12.6 still supports Ampere/Hopper, so pinning it doesn't regress an
  environment also shared with A100/H200 work. The pin lives in that
  environment's `LocalPreferences.toml`, not the depot globally.

## Next steps, in priority order

1. **Formal test suite additions** (manual validation above needs to become
   real, checked-in tests):
   - Tier 1, `test/test_ka_cpu.jl`: CPU-vs-KA-`CPU()` equivalence, exact
     equality (order-preserving port ⇒ bit-identical), runs everywhere.
   - Tier 2, `test/test_gpu_cuda.jl`: CUDA-backend equivalence,
     `CUDA.functional()`-guarded; integer quantities at `==`, floats at a
     measured-ulp `rtol`. Plus `CUDA.allowscalar(false)` over a few real steps
     as a blanket landmine-catcher, and an `isbitstype(cudaconvert(device_view(...)))`
     check that fails loudly instead of as a wall of compiler output.
   - Tier 3, `test/test_gpu_dambreak.jl`: reduced-scale end-to-end parity via
     `time_integrate!` directly (**not** `run_driver!` — it blocks on
     `readline` by default). Elementwise comparison only up to ~100 steps;
     SPH's chaotic amplification makes a 1000-step elementwise gate
     permanently flaky — use physical invariants there instead.
   - Extend `test/test_adapt.jl` with real `adapt(Array, adapt(CuArray, ps))
     == ps` round-trips (currently only exercises the trivial `Array→Array`
     identity path) and a device-storage-type assertion (a broken
     `adapt_structure` that silently no-ops would otherwise still pass a
     round-trip check).
   - A CPU-regression guard (`@allocated` stays flat on the legacy/coloured
     path) — the actual risk to the other 12 scripts is type instability from
     the new `SystemInteraction` parameters, which an allocation check catches
     precisely where a wall-clock timing test would just be flaky.
2. ~~**The crossover benchmark**~~ — **done**, run at default scale
   (`nfx` = 50, 100, 200, 320, 450 → `n_fluid` = 2,500 to 202,500; via a
   throwaway merged Grasph+CUDA environment, the same way as the 3D smoke
   test above):

   | nfx | n_fluid | cpu_col µs/step | cpu_1s µs/step | gpu µs/step | gpu/col | gpu/1s |
   |---|---|---|---|---|---|---|
   | 50  | 2,500   | 398    | 542    | 1,445  | 3.63 | 2.67 |
   | 100 | 10,000  | 1,692  | 2,907  | 1,899  | 1.12 | 0.65 |
   | 200 | 40,000  | 5,810  | 8,794  | 4,518  | 0.78 | 0.51 |
   | 320 | 102,400 | 16,028 | 24,920 | 9,240  | 0.58 | 0.37 |
   | 450 | 202,500 | 26,008 | 55,105 | 16,980 | 0.65 | 0.31 |

   **There is a real crossover, and it sits close to dambreak's own scale.**
   The GPU loses badly at `dambreak.jl`'s actual production size (2,500
   particles — 3.6× slower than CPU-coloured, matching the original
   small-scale prediction), but flips to a **win** by `n_fluid ≈ 40,000`
   (`nfx = 200`) and the margin grows with size — 1.5–3.2× faster than both
   CPU columns by 202,500 particles. This confirms the plan's fallback
   hypothesis: with no FP64 throughput edge on this card, the crossover is
   driven by amortizing the ~8.3 µs/launch overhead over more work per
   launch, not raw compute — and it lands well within a range this class of
   simulation could plausibly reach (dam breaks at a few hundred thousand
   particles are not unusual), not just in some purely theoretical regime.
   The `bench-output/*.csv` this run produced is gitignored, not committed;
   re-run `julia --project bench/dambreak_scaling.jl` (in a CUDA-capable
   merged environment) to reproduce, or pass `--sizes`/`--budget` to push
   toward the ~1M-particle range for a finer picture of where the win keeps
   growing.
3. ~~**3D** (`dambreak_3d.jl`)~~ — **done.** Got the same `GRASPH_BACKEND`
   switch as `dambreak.jl`; the 2D/3D histogram, sort-key, and sweep kernels
   already existed and needed no changes. This was also the first time the
   3D CUDA kernels ran on real hardware (Tier 2, `test_gpu_cuda.jl`, only
   ever exercised 2D) — validated via a live `run_driver!`/CLI smoke run
   (`GRASPH_BACKEND=cuda`, 30k fluid + 30k boundary particles, free-fall
   velocities matching `g·t` exactly, `CUDA.allowscalar(false)`-clean) and a
   new Tier 3 testset in `test/test_gpu_dambreak.jl`
   (`_t3_build_3d`/"dambreak_3d-shaped end-to-end parity") mirroring the
   existing 2D one: 20-step trajectory match at the same tolerances
   (`1e-9`/`1e-7`/`1e-7`), plus 150-step physical invariants. Suite is now
   935/935 (up from 921).
   - Note for the next environment setup: `CUDA` is test-only, so running a
     GPU-mode script directly (not via `Pkg.test()`) needs a merged
     environment — stacking `JULIA_LOAD_PATH="@:test:@stdlib"` does **not**
     work, because root's `PrettyTables = "2"` compat and `test/`'s
     standalone resolve (unconstrained by that compat) land on different
     `PrettyTables` versions, and only one can be loaded per process
     (`CUDATools` needs the newer one, `Grasph` needs the older one —
     manifests as `UndefVarError: TextHighlighter not defined`). The fix is
     to build one throwaway environment the same way `Pkg.test()` does
     internally: `Pkg.activate(path)`, `Pkg.develop(path="/path/to/Grasph.jl")`,
     then `Pkg.add` the test-only deps (CUDA, Adapt, KernelAbstractions, plus
     whatever the script itself `using`s directly, e.g. StaticArrays/Printf)
     — letting the resolver settle everything at once avoids the conflict
     (lands on CUDA 5.8.5, same as `Pkg.test()`'s own resolve).
4. ~~**Converting the remaining pfns to `pfn_contribution`**~~ — **done,
   CPU-only** (Phase C above, `bbf6283`..`d548e06`). Every pfn actually used
   by the 13 experiment scripts now has a `pfn_contribution` method and an
   `_onesided_shape` (the `is_mutual` trait mentioned in earlier revisions
   of this doc, actually built as a three-way `WritesA()`/`WritesB()`/
   `WritesBoth()` trait rather than a boolean — `FluidSolidPfn`'s two-method
   asymmetric-physics case is why: a boolean can't distinguish "mutual" from
   "which side's pressure wins"). The coloured sweep itself has **not** been
   retired — see "What this does NOT do" in Phase C. **What's actually next
   is GPU work, not more pfn conversion:**
5. ~~**Extend `device_view`/`DeviceViews.jl`**~~ — **done, for 4 of the 6
   types originally listed** (commit `e57b787`): `StressParticleSystem`,
   `ElastoPlasticParticleSystem`, and `DynamicBoundarySystem` were fully
   mechanical, following the `BasicParticleSystem`/`FluidParticleSystem`/
   `StaticBoundarySystem` pattern already built. `VirtualParticleSystem` was
   not: unlike the boundary wrappers (which own no host-only fields at all),
   it owns a non-isbits `name::String` directly, so its own concrete type
   can't be rebuilt around a device-viewed inner system the way
   `StaticBoundarySystem`/`DynamicBoundarySystem` are. Fixed by introducing
   an `AbstractVirtualParticleSystem` supertype (mirroring the existing
   `AbstractGhostParticleSystem`/`GhostParticleSystem` precedent) and a new
   isbits `DeviceVirtualSystem`, then widening *only* the one-sided-protocol
   method signatures (`pfn_contribution`/`_onesided_zero_coupled`/
   `_onesided_shape`) that pattern-matched on `VirtualParticleSystem{T,ND}`
   to dispatch on the abstract type instead — the legacy coloured-sweep
   methods are untouched, since that sweep is CPU-only forever and never
   constructs a `device_view`.

   `ProbeParticleSystem`/ghost systems (`GhostParticleSystem`) were **not**
   covered here and were deliberately dropped from this item's scope: both
   hardcoded `Vector` for their per-particle arrays (`x`, `id`, `w_sum`/`v`/
   `rho`, etc.) rather than the array-type-generic parameter every other
   system type uses, which blocks `Adapt.jl` entirely regardless of
   `device_view` — a deeper struct change than "add a device_view method",
   and ghosts specifically were already covered by item 7 below (their
   `resize!`-based generation was the harder problem anyway, not the missing
   proxy). **`GhostParticleSystem`'s side of this is now done — see item 7's
   own entry, done in a later session; the sentence above describes its state
   only up to that point.** `ProbeParticleSystem` is still hardcoded-`Vector`;
   revisit its array-type genericity whenever probes get their own GPU story
   (item 9).

   An adversarial review of this change caught a real, reproducible bug it
   exposed rather than introduced: `VirtualNormUpdater`/
   `PrescribedVelocityUpdater` (`StateUpdaters.jl`) read `prescribed_v` via
   raw `getfield(ps, :prescribed_v)` instead of `ps.prescribed_v` — harmless
   on the real struct (a genuine field there) but bypassing
   `DeviceVirtualSystem`'s `getproperty` entirely, since `getfield` never
   consults `getproperty` overrides. Would have thrown `FieldError` the first
   time either updater ran against a device-viewed virtual system, i.e. the
   first time a `VirtualParticleSystem` is actually GPU-resident — exactly
   the capability this item adds. Fixed both call sites; the regression test
   was confirmed to fail-then-pass via mutation testing (revert the fix →
   the predicted `FieldError` reproduces exactly → restore it → green).

   One gap surfaced but deliberately left unfixed here, since it's
   pre-existing and belongs with item 8, not this item: `FluidSolidPfn`'s two
   `pfn_contribution` methods are typed on the concrete
   `FluidParticleSystem{T,ND}`/`ElastoPlasticParticleSystem{T,ND}` pair, not
   on any device-view-compatible abstraction — so `device_view(fluid)`/
   `device_view(wall)` (both become an unrelated `DeviceSystem`) can't
   dispatch into them yet. `ElastoPlasticParticleSystem`'s own device view is
   otherwise field-complete; this only matters once `DambreakWall.jl`
   (`FluidSolidPfn`'s one real call site) actually tries `ka=true`.
   Suite: 1433/1433 (up from 1371).
6. ~~**Write the KA kernel twin for the reverse/`WritesBoth` sweep.**~~ —
   **done** (commit `0f18e35`). `_sweep_coupled_ka_reverse!`/
   `_sweep_coupled_ka_dispatch!` (`src/KAKernels.jl`) mirror
   `_sweep_coupled_onesided_reverse!`/`_sweep_coupled_onesided_dispatch!`
   (`Interaction.jl`, Phase C). No new `@kernel` function was needed: the
   existing `_sweep_coupled_onesided_kernel_2d!`/`_3d!` kernels only ever
   refer to their arguments as "ps_a" (self/write-target, iterated over) and
   "ps_b" (read-only neighbour), so launching the *same* kernel with
   `system_b`/`system_a`'s device views swapped and `si._cell_start_a` in
   place of `si._cell_start` reproduces the reverse pass exactly — the same
   role-swap trick Phase C's CPU reverse sweep already used, one level
   further. `_sweep_mode!(::OnesidedKA, ...)` now dispatches on
   `_onesided_shape` the same way `OnesidedCPU` already did, so `WritesB()`/
   `WritesBoth()` pfns run the right pass(es) under `ka=true` too. Verified
   both via `KA.CPU()` (bit-reproducible, no GPU needed) and, for the first
   time since Phase B2's crossover benchmark, on real CUDA hardware (RTX
   4060 Laptop GPU) — this machine had a functional GPU again this session,
   unlike Phase C/item 5.

   ~~**A real, previously-unreachable dispatch gap found and left unfixed,
   scoped together with the existing `FluidSolidPfn` gap**~~ — **also fixed,
   same session, as a direct follow-up.** `FluidPfn`'s fluid-fluid
   `pfn_contribution`/`_onesided_zero_coupled` methods (`PairwiseFunctors.jl`)
   are typed on the *concrete* `FluidParticleSystem{T,ND}` on **both**
   sides — needed to disambiguate them from `FluidPfn`'s other coupled
   methods, which all key off a specific wrapper type (`StaticBoundarySystem`,
   `DynamicBoundarySystem`, `Union{Ghost,Virtual}`) on `ps_b` instead.
   `device_view` used to erase that concrete identity entirely (a
   device-viewed `FluidParticleSystem` and a device-viewed
   `BasicParticleSystem`/`StressParticleSystem` produced the exact same
   `DeviceSystem` type, indistinguishable to the dispatcher), so `ka=true`
   was never reachable for `FluidPfn` fluid-fluid — not a regression from
   this item, but a latent gap this item's tests were what first exercised
   (nothing called `FluidPfn` fluid-fluid under `ka=true` before, forward or
   reverse). `MethodError`d loudly rather than computing with the wrong
   dispatch, pinned down by a regression test before the fix.

   **The fix**: a 3-way independent design panel (phantom type parameter on
   `DeviceSystem` vs. a dedicated struct per source type mirroring
   `DeviceVirtualSystem` vs. a host-resolved tag threaded through the
   kernel) converged on the first option, verified against a standalone
   Julia repro before being trusted. `DeviceSystem` gained a phantom `Kind`
   type parameter (`src/DeviceViews.jl`), set at `device_view` construction
   time via `Base.typename(typeof(ps)).wrapper` (the bare, unparameterized
   type of whatever concrete struct produced the view — costs nothing at
   runtime, doesn't affect `isbits`-ness). `FluidPfn`'s fluid-fluid method
   got a `DeviceSystem{T,ND,FluidParticleSystem}`-typed twin
   (`PairwiseFunctors.jl`) sharing one extracted helper with the host-typed
   method, so the physics formula isn't duplicated. An unrelated
   device-viewed pairing (e.g. a device-viewed `BasicParticleSystem` next to
   a device-viewed `FluidParticleSystem`) still `MethodError`s — verified
   directly, not just asserted, both by a dedicated regression test and by
   an adversarial review pass specifically hunting for a way the new method
   could accidentally widen to match something it shouldn't (none found).
   `test/test_ka_cpu.jl`'s gap-pinning test was rewritten into a real
   `onesided` vs `ka=true` equivalence check (2D and 3D), plus
   `test/test_gpu_cuda.jl` gained a real `FluidPfn` fluid-fluid entry
   alongside the test-only pfns already validating the reverse-sweep
   infrastructure, both passing on real CUDA hardware.
   `InterpolateFieldFn`'s `WritesB()` method onto a `VirtualParticleSystem`
   target had no such problem to begin with (item 5 gave
   `AbstractVirtualParticleSystem` its own device view) and is confirmed
   working under `ka=true` on real CUDA hardware too — see
   `test/test_gpu_cuda.jl`.

   `FluidSolidPfn`'s identical gap (`DambreakWall.jl`) was **not** fixed by
   this at the time — the `Kind` mechanism was left as the pattern to reuse,
   just two more method signatures typed on
   `DeviceSystem{T,ND,FluidParticleSystem}`/
   `DeviceSystem{T,ND,ElastoPlasticParticleSystem}` (mirroring both slot
   orders), no new struct or abstract type needed — and was fixed exactly
   that way as a same-session follow-up; see item 8's note.

   Suite: 1479/1479 (up from 1433 before this item; 1466 after the reverse-sweep
   kernel twin alone, 1475 after also fixing the `FluidPfn` fluid-fluid gap, 1479
   after the adversarial review's 3D-coverage-gap finding added a 3D sibling
   test for that fix). This corrects a stale "1475/1475" this doc previously
   cited here and in Practical Notes, which predated that last test addition —
   see update #5.
7. ~~**Ghosts on GPU** — the hardest remaining piece, and gates 7 of the 13
   scripts. `generate_ghosts!`'s two-pass count-then-cursor logic doesn't
   port by direct translation; needs a GPU-compatible rewrite (flag +
   exclusive-scan + compaction into capacity-preallocated buffers, since
   per-step `resize!` — while it does work on `CuVector` — isn't the right
   growth strategy for a count that changes every step).~~ — **done**
   (`src/GhostParticles.jl`, `src/KAKernels.jl`, `src/DeviceViews.jl`,
   `src/Interaction.jl`, `src/Sorting.jl`).

   **Struct change first**: `GhostParticleSystem`'s six owned arrays (`x`,
   `v`, `rho`, `idx_original`, `idx_boundary`, `normals`) went from hardcoded
   `Vector` to the same `VA`/`SA`/`IA` array-type-generic parameterization
   every other system uses (item 5 had explicitly deferred this — see the
   correction just above). A new `count::Base.RefValue{Int}` field was added
   because ghosts are the *one* particle-system type whose logical count
   isn't `length(x)`: on a GPU backend, owned arrays now grow to a
   **capacity** that only ever increases (never shrinks) — `_resize_scratches!`
   (already existed, `Sorting.jl`, used unchanged) grows each array
   independently, no-op if already big enough — while `ghost.n` reads
   `count[]` for the exact logical count regardless of backend. On CPU the
   original exact-`resize!`-every-step behaviour is unchanged (capacity == n
   always there), so this is a genuine zero-behaviour-change for the 7
   ghost-using scripts, none of which have opted into `ka=true` yet.

   **GPU `generate_ghosts!`**: flag + inclusive-cumsum-scan + compaction, per
   the plan above. The `(boundary, particle)` pair space flattens to one
   linear index (boundary-major, particle-minor, matching the CPU nested-loop
   order exactly); a flag kernel (`_ghost_flag_kernel!`) marks which pairs
   qualify, `cumsum!` turns that into each qualifying pair's final 1-based
   destination index in place (no atomics needed — unlike the cell-histogram
   CSR build, this is a genuine 1-to-1 stream compaction, not a many-to-one
   histogram), then a scatter kernel (`_ghost_scatter_kernel!`) writes
   directly into the (now sufficiently large) owned arrays. `GhostEntry`
   gained a `_flags::FA` scratch field for this — fixed length `NB *
   source.n`, allocated once at construction (a ghost's *source* particle
   count never changes over a run, only how many currently qualify).
   `update_ghost_kinematics!` and `GhostCopier`'s per-field copy (previously
   CPU-only scalar loops — harmless before this item since a ghost's arrays
   could never be anything but `Vector`) also gained real GPU kernel twins
   (`_ghost_kinematics_kernel!`, `_ghost_copy_field_kernel!`), backend-
   dispatched the same way as everywhere else in this codebase
   (`KA.get_backend(...)` → `::KA.CPU`/`::KA.GPU` methods).

   **`device_view`/`Adapt.jl`**: `DeviceGhostSystem` mirrors item 5's
   `DeviceVirtualSystem` pattern exactly — subtypes `AbstractGhostParticleSystem`
   so every ghost-coupled `pfn_contribution` method (already narrowly typed on
   that abstraction) dispatches into it unmodified, flattening only the
   fields a pfn actually reads (`x`, `v`, `rho`, `mass`, `c`, plus `extras`)
   — `idx_original`/`idx_boundary`/`normals` are pure ghost-generation
   bookkeeping, never read by a pfn, so they're left out. `Adapt.adapt_structure`
   for `GhostParticleSystem`/`GhostEntry` follows the same recursive-rebuild
   pattern as `VirtualParticleSystem`.

   **A load-bearing correctness fix outside `GhostParticles.jl` itself**: grid
   building (`_bbox`/`_populate_cells_sorted!`, `Interaction.jl`) used to
   derive its particle count from `length(x)` — true for every system type
   *until this item*, since a GPU-resident ghost's owned arrays can now be
   longer than its logical count. Both functions were changed to take an
   explicit `n` parameter instead (a no-op for every non-ghost system, since
   `length(x) == n` still holds for those); without this, stale data in a
   ghost's unused capacity slots would silently leak into the CSR cell grid
   as phantom particles.

   **A real bug found and fixed during development** (not from the
   adversarial review below — from testing on real CUDA hardware): the first
   version of the GPU capacity-growth check compared the new count only
   against `length(getfield(ghost, :x))`. `extras` arrays (`p`, `stress`, …)
   start at length 0 independently of whatever capacity `x`/`v`/etc. start
   at (`_build_extras`), so a step where `x` already had enough room but
   `extras` didn't skipped growing `extras` entirely — the next
   `GhostCopier` GPU kernel launch wrote out of bounds into a length-0 array,
   reproduced directly as a CUDA "illegal memory access". Fixed by growing
   every owned array (including `extras`) unconditionally via
   `_resize_scratches!(_particle_arrays(ghost), total)` rather than gating on
   one array's length.

   **Adversarial review** (3 independent reviewers — capacity/count-invariant
   correctness, GPU dispatch/kernel safety, test-coverage/doc-accuracy) found
   no other instance of that bug class, but did surface real, actionable
   gaps, all acted on:
   - A latent, unenforced assumption: `GhostEntry._flags`'s fixed size (`NB *
     source.n`) silently breaks if `source` were itself a `GhostParticleSystem`
     (the one type whose `n` isn't constant) — nothing in the codebase does
     this today, but nothing stopped it either. Fixed by rejecting it at
     construction (`GhostParticleSystem`'s constructor now throws
     `ArgumentError` if `ps isa AbstractGhostParticleSystem`), converting a
     hypothetical silent GPU out-of-bounds write into a loud, immediate error.
   - `GhostCopier`'s callable decided its backend from `ghost.idx_original`
     while `generate_ghosts!`/`update_ghost_kinematics!` decided from
     `ghost.x` — harmless today (construction/adapt always keep a ghost's
     owned arrays backend-consistent) but needlessly inconsistent; unified to
     `ghost.x` everywhere.
   - Test-coverage gaps: `HouseholderReflect` mode (vs. the no-op `nothing`
     mode every other ghost GPU test used) had never run through the GPU
     field-copy kernel; no ghost GPU test used 3D; no ghost GPU test used the
     real `NB=8` (4 walls + 4 corners) shape `bubble3.jl` actually needs;
     `write_h5(ghost, ...)` — itself rewritten by this item to explicitly
     slice every array to `1:n` — had zero test coverage before or after.
     All four added (`test/test_gpu_cuda.jl`, `test/test_ghost_particles.jl`).
     The `NB=8` test's diagonal corner normals (`SVector(±1,±1)/sqrt(2)`)
     surfaced the same ~1 ulp CPU/GPU FMA-contraction noise this codebase
     already tolerances everywhere else for float comparisons — not a bug,
     just the first ghost test to hit non-axis-aligned reflection arithmetic;
     switched that one assertion from `==` to a relative-tolerance check.
   - Two findings judged *not* bugs introduced by this item, documented
     instead of "fixed": `GhostParticleSystem`'s convenience constructor
     always builds `Vector` arrays regardless of `ps`'s own array type
     (identical to `VirtualParticleSystem`'s pre-existing convention,
     unchanged by this item) — constructing directly from an
     already-GPU-resident source gives a mixed-backend object; the fix is to
     build CPU-first as usual, then `adapt(CUDABackend(), ge::GhostEntry)`
     the whole entry as one call. And `Adapt.adapt_structure` doesn't
     preserve object identity across separate `adapt()` calls (inherent to
     how every wrapper type in this codebase — Virtual, boundary, now Ghost —
     rebuilds itself from adapted fields), which matters specifically for
     ghosts because they're always self-referencing
     (`ghost.source === fluid`): a driver that separately adapts its own
     `fluid` and a `GhostEntry` wrapping the same `fluid` ends up with two
     independent GPU copies, not aliases. Both documented directly in
     `GhostParticleSystem`'s docstring for whoever picks up item 8's
     ghost-script wiring next.

   Verified via `KA.CPU()` (backend-dispatch and kernel logic, no GPU needed)
   and real CUDA hardware (RTX 4060 Laptop) — the flag/scatter/kinematics/
   copy-field kernels, capacity growth *and* the capacity-stays-above-a-
   shrunk-count regime (deliberately constructed in
   `test/test_gpu_cuda.jl` by growing then shrinking the ghost count before
   sweeping, to exercise the `_bbox`/`_populate_cells_sorted!` fix above under
   the exact condition it exists for), a full fluid↔ghost `onesided=true` vs
   `ka=true` sweep-equivalence test (`test/test_ka_cpu.jl`, `KA.CPU()`) and a
   full sort+grid+sweep pipeline test against real CUDA
   (`test/test_gpu_cuda.jl`), plus `Adapt.jl` round-trips
   (`test/test_adapt.jl`) and `device_view` dispatch equivalence
   (`test/test_device_views.jl`). Suite: 1600/1600 (up from 1496 after item
   8).

   **What's still not done**: none of the 7 ghost-using scripts
   (`EP_ColumnCollapse.jl`, `GranularColumnCollapse.jl`,
   `GranularColumnCollapse3D.jl`, `Trapdoor.jl`, `bubble.jl`, `bubble2.jl`,
   `bubble3.jl`) have been touched — every pfn and every piece of
   infrastructure they need now has a working `ka=true` path, but wiring
   `onesided=true`/`ka=true` into their actual `SystemInteraction`/
   `GhostEntry` calls is still item 8's job (see item 8's own remaining-scope
   note below, now further unblocked).
8. **Wire `onesided=true`/`ka=true` into the other 11 scripts** (12 minus
   `dambreak_3d.jl`, done separately as item 3), one at a
   time, mirroring `dambreak.jl`'s `GRASPH_BACKEND` switch — now unblocked
   by items 5-7 plus Phase C's integration harnesses (`test/
   test_onesided_integration_*.jl`), which give a per-shape correctness
   oracle to validate each script's GPU wiring against before trusting it.
   ~~`DambreakWall.jl` specifically also needs `FluidSolidPfn`'s two
   `pfn_contribution` methods widened off the concrete `FluidParticleSystem`/
   `ElastoPlasticParticleSystem` pair to something `device_view` can
   dispatch into before it can try `ka=true` at all~~ — **done** (commit
   `0fac396`), same session, as a direct follow-up to item 6's `FluidPfn`
   fix. Unlike `FluidPfn` fluid-fluid, `FluidSolidPfn`'s physics is
   asymmetric under relabeling — the fluid's own pressure drives both
   sides' pressure term, never the solid's (that's the entire point of the
   functor: a continuous pressure field across the fluid-solid interface) —
   so a single shared `DeviceSystem`-typed method, the pattern that worked
   for `FluidPfn`, would silently use the wrong side's pressure whenever the
   reverse sweep put the solid in the `ps_a` slot. Two distinct methods
   instead, exactly mirroring the two host-typed CPU methods that already
   existed (from Phase C):
   `pfn_contribution(f, ps_a::DeviceSystem{T,ND,FluidParticleSystem},
   ps_b::DeviceSystem{T,ND,ElastoPlasticParticleSystem}, ...)` and its mirror
   with the slots swapped, each calling a shared, untyped helper function
   together with its host-typed counterpart so the arithmetic isn't
   duplicated (`PairwiseFunctors.jl`) — no new struct or abstract type
   needed, exactly as this item's earlier note predicted. Verified via
   `KA.CPU()` (2D and 3D onesided-vs-`ka=true` equivalence, a
   mismatched-pairing `MethodError` regression test covering both new
   methods) and real CUDA hardware (`test/test_ka_cpu.jl`,
   `test/test_gpu_cuda.jl`), plus a dedicated regression test asserting the
   solid-as-target method reads the fluid's pressure from `ps_b.p` and never
   touches `ps_a.p` — the specific failure mode the two-method split exists
   to prevent, which a swap-antisymmetry check alone can't catch (it only
   compares a call against itself in one orientation). An adversarial review
   pass (dispatch-safety/correctness/test-coverage, 3 independent reviewers)
   found no confirmed issues. Suite: 1496/1496 (up from 1479 after item 6).

   ~~This item's remaining scope — actually wiring `onesided=true`/`ka=true`
   into the 12 non-dambreak scripts' `SystemInteraction` calls — is
   unstarted~~ — **5 of the remaining 11 done** (item 7, above, closed the
   last pfn-/infrastructure-side gap for ghost-using scripts, which is what
   unblocked this). `ellipse.jl`, `DambreakWall.jl`,
   `GranularColumnCollapse.jl`, `GranularColumnCollapse3D.jl`, and
   `EP_ColumnCollapse.jl` now carry the same `GRASPH_BACKEND` switch as
   `dambreak.jl`/`dambreak_3d.jl`: defaults to CPU/coloured, and
   `GRASPH_BACKEND=cuda` adapts every system to `CuArray` and passes
   `onesided=ka_mode, ka=ka_mode` to every `SystemInteraction`. These 5 were
   chosen specifically because they use nothing but `LeapFrogTimeIntegrator`
   plus system types that already have a working `ka=true` path end to end
   (`FluidParticleSystem`, `BasicParticleSystem`/`DynamicBoundarySystem`,
   `StressParticleSystem`, `ElastoPlasticParticleSystem`,
   `GhostParticleSystem`) — no `VirtualParticleSystem`, no
   `ProbeParticleSystem`, no `RK4TimeIntegrator`.

   For the 3 ghost-using scripts in this batch (`GranularColumnCollapse.jl`,
   `GranularColumnCollapse3D.jl`, `EP_ColumnCollapse.jl`), the adapt call
   follows `GhostParticleSystem`'s own docstring exactly (see item 7): the
   `GhostEntry` is adapted as one unit and the canonical GPU-resident source
   is pulled back out via `getfield(ghost, :source)` rather than adapting the
   source fluid a second time — e.g. `left_ghost_entry =
   adapt(CUDABackend(), left_ghost_entry); left_ghost = left_ghost_entry.ghost;
   fluid = getfield(left_ghost, :source)`. `DambreakWall.jl` needed every one
   of its 6 systems (fluid, wall, and 4 independently-normaled
   `DynamicBoundarySystem`-wrapped walls) adapted and all 8 interactions
   switched — mechanically the largest script in this batch, but the same
   pattern throughout since none of its systems are self-referencing.

   **Verified per script** by running each one directly, once unmodified
   (coloured sweep) and once with `GRASPH_BACKEND=cuda`, on real CUDA
   hardware (RTX 4060 Laptop) via the merged-throwaway-environment technique
   (see "Environment notes" above) — `run_driver!`'s scripts aren't part of
   `Pkg.test()`'s own dependency resolution, so this needed
   `Pkg.add(["CUDA","Adapt","KernelAbstractions","HDF5","Printf"])` in a
   throwaway environment same as always. Every script ran cleanly for
   several thousand steps (some far more — `ellipse.jl`, 1976 particles,
   ran to full completion, 5000/5000 steps, in both modes) with no
   `MethodError`, no `scalar indexing disabled` violation, and no `NaN`.
   Where directly comparable (deterministic setup, matching step number),
   printed field summaries agreed with the CPU run to float noise consistent
   with the CPU/GPU FMA-contraction difference this codebase already
   tolerances elsewhere — e.g. `GranularColumnCollapse.jl` step 500 `v`
   magnitude range: CPU `[0.000143611, 0.048455]`, GPU
   `[0.000140276, 0.0484567]`.

   No new automated `ka=true` integration tests were added for these 5
   scripts specifically, deliberately: the pfn-level `ka=true` sweep
   computation they depend on is already exhaustively tested (items 6 and 8
   itself, both `KA.CPU()` and real CUDA), and
   `test_onesided_integration_{dambreakwall,ellipse,soil2d,soil3d}.jl`
   (Phase C) already prove `onesided=true` (CPU) matches the coloured sweep
   for exactly these shapes — so by transitivity the only risk this item
   could actually introduce was mechanical (right systems threaded to the
   right interactions, adapt ordering, ghost-aliasing), which direct
   execution checks for as directly as a new fixture would. Revisit if that
   judgment call turns out wrong (a real regression slips through unnoticed).

   **Still not wired, and blocked by item 9 below, not by anything in this
   item**: `bubble.jl`/`bubble2.jl`/`bubble3.jl` (all `RK4TimeIntegrator`),
   `EP_ColumnCollapse2.jl` (`VirtualParticleSystem` boundaries),
   `Trapdoor.jl` (`VirtualParticleSystem` *and* `ProbeParticleSystem`), and
   `CantileverBeam.jl` (`ProbeParticleSystem`). Confirmed directly rather
   than assumed: `_update_virtual_positions!` (`TimeIntegration.jl`) is a
   raw `@inbounds for i in 1:vps.n; vps.x[i] += pv*dt; end` scalar loop with
   no backend dispatch at all, which would violate `CUDA.allowscalar(false)`
   the instant a virtual system's arrays are `CuArray`s — regardless of
   whether `prescribed_v` happens to be zero, since the loop still executes;
   `ProbeParticleSystem` is still hardcoded-`Vector` (item 5), so it has no
   `Adapt.adapt_structure` at all; `RK4TimeIntegrator`'s own multi-stage
   bookkeeping was already named directly in item 9. Also deliberately not
   given `onesided=true` alone (without `ka=true`) as a partial step: item
   2's crossover benchmark shows the one-sided CPU sweep is *slower* than
   the coloured sweep at every scale measured (2,500-202,500 particles, up
   to ~2.1× slower) — it exists purely as the GPU-compatible sweep shape,
   not a CPU-side optimization, so shipping it alone on these 6 scripts
   would be a pure regression with no offsetting benefit until item 9 makes
   `ka=true` reachable for them too.
9. ~~Virtual particle systems, probes, and the RK4 integrator have no GPU
   sweep path at all yet, independent of pfn support — `VirtualParticleSystem`
   position/state advance, `_measure_probes!`, and RK4's multi-stage
   bookkeeping are all still CPU-`for`-loop or Polyester code. Confirmed
   directly (not just asserted) while scoping item 8's script-wiring pass:
   6 of the remaining 11 scripts (`bubble.jl`/`bubble2.jl`/`bubble3.jl`,
   `EP_ColumnCollapse2.jl`, `Trapdoor.jl`, `CantileverBeam.jl`) are blocked
   here rather than by any pfn or `device_view` gap — see item 8's own note
   for exactly which gap blocks which script.~~ — **done** (commit
   `9182bd8`). All three named gaps turned out to be much narrower than "no
   GPU sweep path at all," once traced to source:

   - **RK4's own gap** (`src/TimeIntegration.jl`): its `time_integrate!`
     hardcoded `sort_perm_buf = Vector{Int}(undef, ...)`/`sort_key_buf =
     Vector{UInt64}(undef, ...)` regardless of the systems' actual array
     type — `LeapFrogTimeIntegrator`'s own loop already derived them via
     `similar(first(sys).x, Int, sort_max_n)`, RK4's just never got the same
     treatment. A 2-line fix (mirror LeapFrog exactly). This is invisible to
     any `KA.CPU()` test — `Vector` and `similar(x,...)` are the same type
     when `x` is itself a `Vector` — so it could only be caught (and only
     matters) on a real non-CPU backend.
   - **`VirtualParticleSystem`'s gap**: `_update_virtual_positions!` was a
     raw `@inbounds for i in 1:vps.n; vps.x[i] += pv * dt; end` loop. Fixed
     with a new backend-dispatched primitive, `_axpy_const_ip!` (`src/Utils.jl`,
     same shape as the existing `_axpy_ip!`/`_axpy_oop!`): a `@batch` loop on
     `KA.CPU()`, `q .+= Ref(a * c)` elsewhere. `_advance_probe_positions!`
     (the equivalent scalar loop for probes) got the same fix. **A real bug
     here, caught only by real CUDA hardware**: the first version of the
     non-CPU branch wrote `@. q += a * $c`, using `$c` inside `@.` to try to
     stop `c` (a constant `SVector`) from being dot-broadcast — that doesn't
     work, since an `SVector` is itself a genuine `AbstractArray` and
     broadcasts shape-checked against `q` regardless, throwing
     `DimensionMismatch` the instant `length(q) != length(c)`. No `KA.CPU()`
     test could have caught this either: position-advance backend-dispatches
     on the array's own type (`KA.get_backend(vps.x)`), not on whether
     `ka=true` was set on some unrelated `SystemInteraction` — so this branch
     is only reachable with a real `CuArray`. Fixed by wrapping in `Ref(...)`
     instead, the standard idiom for broadcasting a constant struct across an
     array.
   - **`ProbeParticleSystem`'s gap** (the largest piece — this is the system
     item 5 explicitly deferred): it hardcoded `Vector` for `x`/`id`/`w_sum`,
     which blocked `Adapt.jl` entirely regardless of `device_view`, same as
     `VirtualParticleSystem`/`GhostParticleSystem` before items 5/7. Given
     the exact same array-type-generic treatment (`VA`/`SA`/`IA` type
     params, a fully-generic positional constructor mirroring
     `BasicParticleSystem`'s, `Adapt.adapt_structure`), plus a new
     `AbstractProbeParticleSystem` supertype and `DeviceProbeSystem`
     (`src/DeviceViews.jl`) mirroring `DeviceVirtualSystem`/`DeviceGhostSystem`
     exactly — needed because `InterpolateFieldFn`/`NeighborCountFn`'s
     one-sided `pfn_contribution`/`_onesided_zero_coupled`/`_onesided_shape`
     methods were typed on the concrete `ProbeParticleSystem{T,ND}`, which a
     separate flattened device-view struct can't match without the
     abstraction; widened to `AbstractProbeParticleSystem{T,ND}`, mirroring
     item 5's identical `AbstractVirtualParticleSystem` widening precisely.
     Like every other system in this codebase, the keyword constructors
     always build plain `Vector`s regardless of any argument's own array
     type — documented directly in `ProbeParticleSystem`'s docstring
     (mirroring `GhostParticleSystem`'s self-referencing-source caveat) since
     it's a real trap: `mirror_target`'s array type doesn't propagate to
     `id`/`w_sum`, so constructing a probe directly from an already-adapted
     GPU system gives a broken mixed-backend object — the correct idiom is
     build CPU-side, then `adapt(CUDABackend(), probe)` as one unit and pull
     `mirror_target` back out via `getfield`, same as ghosts.
   - **`_measure_probes!`'s mirror step** (`x[src_id[i]] = src_x[i]` for
     `i in 1:probe.n`) is a scatter, not an elementwise op — no broadcast
     expresses it. Given a dedicated KA kernel (`_probe_mirror_kernel!`,
     `src/KAKernels.jl`) rather than a `Base` fancy-indexing scatter
     (`dst_x[src_id] = src_x`, which would work for CUDA specifically but
     isn't guaranteed across arbitrary `KernelAbstractions.jl` backends) —
     same reasoning `_apply_perms!`'s existing gather/copyback kernels
     already use. `src_id` is always a permutation of `1:n` (every system's
     sort-tracking invariant), so every thread writes a distinct destination
     slot and no atomics are needed, identical to `_ghost_scatter_kernel!`'s
     stream-compaction case.

   **Verified**: `KA.CPU()` equivalence (`test/test_ka_cpu.jl`, extending the
   existing `InterpolateFieldFn`/`NeighborCountFn` WritesB reverse-sweep
   pattern to a probe target) plus real CUDA hardware
   (`test/test_gpu_cuda.jl`) covering exactly the three fixes above: an
   `RK4TimeIntegrator` run reusing `test_onesided_integration_bubble.jl`'s
   `_bubble_like`/`_bubble_integrator` fixtures (`bubble.jl`'s real shape —
   RK4 plus a self-referencing ghost) against a CPU oracle; a
   `VirtualParticleSystem` with nonzero `prescribed_v` (Trapdoor.jl's
   `trapdoor_moving_virt` shape) checked against the exact expected constant
   translation, id-gather-based so it's permutation-safe across re-sorts; and
   a `ProbeParticleSystem` with a self-referencing `mirror_target` plus
   `NeighborCountFn` (CantileverBeam.jl's shape), CPU vs GPU neighbour counts
   compared exactly after a real `time_integrate!` run. All three exist
   specifically because they can't be validated any other way — none of
   these bugs are reachable through `KA.CPU()` alone. `test/test_adapt.jl`
   and `test/test_device_views.jl` got the same round-trip/device_view-proxy/
   narrow-typing treatment items 5 and 7 gave Virtual and Ghost.

   A pre-existing, unrelated flaky test was found and fixed along the way
   (not introduced by this item): `test_adapt.jl`'s Ghost/GhostEntry
   adapt-round-trip fixture never initialised `fluid.v` before use, leaving
   it `undef` — `update_ghost_kinematics!` propagates that garbage into
   `ghost.v`, and this run happened to land on a NaN bit pattern, failing the
   `==` comparison (`NaN != NaN`, even printed identically on both sides).
   Fixed with a `fill!`.

   Suite: 1651/1651 (up from 1600 after item 8).

   ~~**What's still not done**: the 6 scripts item 8 identified as blocked
   here (`bubble.jl`/`bubble2.jl`/`bubble3.jl`, `EP_ColumnCollapse2.jl`,
   `Trapdoor.jl`, `CantileverBeam.jl`) are now unblocked at the
   infrastructure level — every piece they need has a working `ka=true` path
   — but none of them have actually been wired with the `GRASPH_BACKEND`
   switch yet. That's a follow-up to item 8's script-wiring pass, not part of
   this item's scope, mirroring how item 7 unblocked item 8's ghost-using
   scripts without wiring them itself.~~ — **done, item 10, below.**
10. **Wire `GRASPH_BACKEND` into the last 6 experiment scripts** — **done**
   (commit `149c238`). `bubble.jl`, `bubble2.jl`, `bubble3.jl`,
   `EP_ColumnCollapse2.jl`, `Trapdoor.jl`, and `CantileverBeam.jl` now carry
   the same `GRASPH_BACKEND` switch as the other 7 scripts — all 13
   experiment scripts support `GRASPH_BACKEND=cuda` now. Purely script-level
   wiring; item 9 had already closed every infrastructure gap these needed.

   Most of it followed the two aliasing idioms items 7-9 already
   established for a system that self-references another (adapt the wrapper
   as one unit, pull the canonical GPU-resident source back out via
   `getfield`): the ghost in `bubble.jl`/`bubble2.jl`/`bubble3.jl`/
   `Trapdoor.jl` (self-referencing its fluid/soil source), and the probe in
   `CantileverBeam.jl` (`beam_probe.mirror_target === beam`) — the latter
   also required reordering the script slightly, constructing the probe
   *before* the interactions that need `beam` rather than after, since the
   correct adapted `beam` only exists once pulled back out of the adapted
   probe.

   `Trapdoor.jl` needed a third idiom: `trapdoor_static_virt` and
   `trapdoor_moving_virt` both wrap the *same* `trapdoor_source`, so the two
   run phases (static settling, then moving) share live position/stress
   state across the stage boundary — adapting both independently would
   silently give each phase its own disconnected copy of the trapdoor. The
   fix is to adapt one, then rebuild the other directly around the same
   already-adapted source (`VirtualParticleSystem(trapdoor_source_gpu, ...)`).

   **That reconstruction pattern surfaced a real, previously-unreachable
   bug**: `VirtualParticleSystem`'s keyword constructor
   (`src/Particles.jl`) hardcoded `w_sum = zeros(T, n)` regardless of the
   source's own array type. Every prior GPU-resident virtual system reached
   `w_sum`'s correct type via `Adapt.adapt_structure` adapting `source` and
   `w_sum` together — this was the first time anything constructed a
   *fresh* `VirtualParticleSystem` directly around an already GPU-resident
   source. The result was a mixed-backend struct (source on `CuArray`,
   `w_sum` on `Vector`) that isn't `isbits`, so `device_view` builds fine
   and `cudaconvert` even succeeds (it only converts the `CuArray`s it
   finds), but the actual `@kernel` launch fails to *compile* the first
   time that struct reaches one — a GPU-compilation error, not a caught
   type mismatch or a soft runtime one, and it only surfaces the first time
   the affected interaction actually sweeps (`Trapdoor.jl`'s moving-phase
   stage, past the point where the static phase alone would have caught
   it). Fixed by deriving `w_sum`'s array type from the source via
   `similar` (`w_sum = fill!(similar(ps.x, T, n), zero(T))`), the same
   buffer-type-follows-the-system idiom RK4's sort scratch buffers were
   already fixed to use in item 9. Zero behaviour change for the CPU case
   (`similar` on a `Vector` gives a `Vector`, `fill!`ed to the same zeros
   `zeros(T,n)` already produced).

   `CantileverBeam.jl` also needed two pre-existing, unrelated broken calls
   fixed before it would even run on CPU: `CubicSplineKernel(; ndims=2)`
   (missing its required positional `h`) and `SystemInteraction(...; h=h_sph)`
   (`h` isn't a keyword `SystemInteraction` accepts at all). Phase C's own
   integration harness had already found and documented both — "confirmed
   `CantileverBeam.jl` is broken as committed... its harness reconstructs
   the intended shape with corrected calls instead" — but left the script
   itself unfixed, since fixing it wasn't in that item's scope. This item
   needed the actual script runnable to verify its GPU wiring, so it applied
   the same corrected calls the harness already used.

   **Verified per script** on real CUDA hardware (RTX 4060 Laptop), via the
   merged-throwaway-environment technique (see "Environment notes" above):
   every script ran cleanly for several thousand steps (`bubble.jl`'s 6000
   steps to full completion) with no `MethodError`, no `scalar indexing
   disabled` violation, and no `NaN`. `Trapdoor.jl` was additionally run at
   full production particle scale (26,400 soil / 2,340 bottom / 400
   trapdoor particles — only the two stages' step counts were reduced, to
   300 each) specifically to cross the static-to-moving phase boundary,
   confirming the shared-source reconstruction survives a real phase
   transition end to end, not just the static phase alone.

   New regression test (`test/test_gpu_cuda.jl`, "device_view is isbits
   after cudaconvert"): builds a second `VirtualParticleSystem` directly
   around a first virtual's already-adapted source (mirroring `Trapdoor.jl`'s
   shape exactly) and asserts `w_sum isa CuArray` plus
   `isbitstype(typeof(cudaconvert(device_view(...))))` — confirmed to fail
   with the pre-fix code (`w_sum` comes back `Vector`) and pass with the fix,
   via direct mutation testing (temporarily reverted the fix, re-ran, saw the
   predicted failure, restored it, saw green). This bug class is invisible to
   any `KA.CPU()` test, same reasoning as item 9's `_axpy_const_ip!` bug —
   `similar`/`zeros` are identical when `ps.x` is itself a `Vector`, so it
   only manifests with a real non-CPU backend.

   Suite: 1653/1653 (up from 1651 after item 9).

11. **Coloured, two-sided GPU sweep — benchmarking spike (`ColouredKA`)** —
    **done.** Items 8/10 finished wiring `onesided=true`/`ka=true` into every
    script; item 2's crossover benchmark separately established that
    one-sided GPU (`OnesidedKA`) beats CPU-coloured from `n_fluid ≈ 40,000`
    on this hardware. Since the CPU comparison run earlier this session
    (13 scripts, `onesided=true` vs. `onesided=false`, both on CPU) found
    CPU-coloured consistently 1.1-2.4× faster per sweep than CPU-onesided —
    expected, since coloured visits each pair once (half-shell,
    Newton's-third-law reuse) vs. onesided's full-neighbour-list-per-particle
    traversal (~2× the pairwise evaluations) — the natural question was
    whether porting *that same* half-work colouring scheme to GPU (one
    `@kernel` launch per colour, reusing the original two-sided *mutating*
    pfn contract, not `pfn_contribution` — so zero pfn-conversion work,
    unlike `OnesidedKA`) would also beat `OnesidedKA` on GPU, given this
    GPU's already-established launch-overhead-bound (not compute-bound)
    profile (item 2's ~8.3µs/launch finding).

    Built as a 4th `ExecMode`, `ColouredKA` (`src/Backend.jl`), reachable
    only via `SystemInteraction`'s new internal `mode::Union{Nothing,
    ExecMode}=nothing` override kwarg (`src/Interaction.jl`) — deliberately
    not a new public `onesided`/`ka` boolean combination, since this is a
    benchmarking spike, not a proposed new default. The kernels
    (`src/KAKernels.jl`, `_sweep_self_coloured_ka_kernel_2d!`/
    `_sweep_coupled_coloured_ka_kernel_2d!`, 2D only — `dambreak.jl`'s shape,
    matching what `bench/dambreak_scaling.jl` benchmarks) are a direct port
    of the existing `ColouredCPU` colour loops (`_sweep_self!`/
    `_sweep_coupled!`, `Interaction.jl`), one launch per colour instead of
    one `@batch` pass per colour, `KA.synchronize` between every colour
    (consecutive colours have overlapping write-sets by design; colours
    within one launch don't, by the same cell-separation argument
    `ColouredCPU`'s own comments already prove). Each kernel body reuses
    `_pair_self!`/`_pair_coupled!` verbatim — already plain, backend-agnostic
    `@inline` functions — so pair evaluation is bit-identical to
    `ColouredCPU`'s, not merely equivalent; confirmed exactly on real CUDA
    hardware (RTX 4060 Laptop): `max|Δdvdt| = max|Δdrhodt| = 0.0` (exact,
    not just within tolerance) against a `ColouredCPU` oracle at both 1,600
    and 40,000 particles, using `bench/dambreak_scaling.jl`'s own geometry.
    (An earlier ad hoc test geometry produced a spurious `NaN` — traced to a
    self-inflicted exact-position collision between a fluid particle and a
    boundary particle, a `0/0` kernel-gradient singularity at `r=0` that
    would hit *any* sweep mode identically, not a `ColouredKA`-specific bug;
    not present in any real script or in `bench/dambreak_scaling.jl`'s
    actual geometry.) Formalised as a new `test_gpu_cuda.jl` testset ("full
    pipeline (sort+grid+sweep), ColouredKA self+coupled: CPU-coloured oracle
    vs CUDA") using the file's existing random-fixture/`_byid`/sort-buffer
    helpers; 3/3 passing on real CUDA. Suite: 1656/1656 (up from 1653).

    `bench/dambreak_scaling.jl` extended with a 4th column (`gpu_col`),
    reusing the existing `cpu_col`/`cpu_1s`/`gpu` (`OnesidedKA`) machinery —
    run at the same default sizes as item 2's original table (RTX 4060
    Laptop, same hardware):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | 1s/col | col/col | col/1s |
    |---|---|---|---|---|---|---|---|---|
    | 50  | 2,500   | 431.2   | 540.3   | 1332.1  | 6163.9  | 2.466 | 14.295 | 4.627 |
    | 100 | 10,000  | 1361.9  | 2154.2  | 1659.2  | 10772.3 | 0.770 | 7.910  | 6.492 |
    | 200 | 40,000  | 6860.8  | 9240.8  | 4524.5  | 11013.9 | 0.490 | 1.605  | 2.434 |
    | 320 | 102,400 | 14565.9 | 27762.7 | 9200.6  | 11055.3 | 0.331 | 0.759  | 1.202 |
    | 450 | 202,500 | 34395.5 | 45683.8 | 16967.1 | 11394.0 | 0.371 | 0.331  | 0.672 |

    **The trade does not pay off at any size this migration actually
    targets, and only barely starts to at the very top of the tested
    range.** `gpu_col` is dramatically slower than `gpu_1s` at small/medium
    scale (4.6× slower at dambreak.jl's own production size, 2,500
    particles) and is nearly *flat* from `n_fluid = 10,000` through `202,500`
    (10,772 → 11,394µs, barely moving across a 20× size increase) — the
    signature of a cost dominated entirely by kernel-launch count, not
    compute. `ColouredKA` issues 6 launches for the self interaction (2D) +
    9 for the coupled interaction (2D) = 15 per `sweep!` call, vs. 1 each
    (2 total) for `OnesidedKA` — a 7.5× multiplier on the same
    ~8.3µs/launch floor item 2 already measured, which alone accounts for
    most of the flat ~11ms plateau. `gpu_col` only overtakes `gpu_1s` at the
    largest tested size (202,500 particles, 1.49× faster) — right at the
    edge of the tested range, well past `OnesidedKA`'s own crossover against
    CPU-coloured (`n_fluid ≈ 40,000`, item 2). Not investigated further
    (extending `--sizes` toward the ~1M-particle range to see whether the
    margin keeps growing past 202,500, the way item 2's `gpu_1s` margin did)
    — out of scope per this item's plan, since the result already answers
    the question the plan asked: on this hardware, at the particle counts
    this migration's actual scripts run at, `OnesidedKA` remains the right
    choice, and `ColouredKA` is not worth wiring into any script.

    **Not wired into any script, and not intended to be** — this was
    explicitly scoped as a benchmarking spike (see the plan this item
    executed). `ColouredKA` stays reachable only via the internal `mode`
    override, 2D/`FluidPfn`-self/`StaticBoundarySystem`-coupled only; no
    ghosts, virtual particles, probes, 3D, or `RK4TimeIntegrator` support
    was built, and none is planned unless a future, much-larger-scale result
    changes this conclusion.

12. **Persistent/cached grid + Verlet-skin rebuild cadence** — **done.**
    First part of deferred item 1 below (fused sort gather/copyback
    `Ref`-swap stays deferred — see "Not built" below). Item 11 confirmed
    launch count, not compute, is what this hardware pays for; every
    timestep before this item unconditionally paid for a full re-sort +
    grid rebuild (`_sort_all_systems!` + `_prepare_grids!`,
    `TimeIntegration.jl`) regardless of whether any particle had moved far
    enough to invalidate the previous step's cell list. This item makes
    that rebuild skippable.

    **The correctness trap that shaped the design**: `si._cell_size`
    (`SystemInteraction`, fixed at construction to `kernel.interaction_length`)
    was doing double duty everywhere in the onesided sweep functions —
    `cutoff_sq = si._cell_size^2` (the physical SPH interaction radius, fed
    to `_pair_self_onesided!`/`_pair_coupled_onesided!`'s `r_sq < cutoff_sq`
    filter) *and* `cutoff = si._cell_size` (the grid cell pitch, fed to
    `_cell_1idx` for cell-index lookups). Naively widening cell size so the
    grid tolerates a few steps of drift — the standard Verlet-skin trick —
    would, with that conflation, have silently widened the *physical*
    interaction radius too: a correctness bug, not a perf change. Fixed by
    adding a second, mutable field, `_grid_cutoff::Base.RefValue{T}`
    (`Interaction.jl`), set by `create_grid!(si, skin)` to
    `si._cell_size + skin` and read by every grid-index call site; `_cell_size`
    itself, and every `cutoff_sq` derived from it, is completely untouched —
    `skin = 0` (the default) makes `_grid_cutoff[] == _cell_size`, so nothing
    about existing behaviour changes unless `verlet_skin > 0` is explicitly
    requested. The same split was threaded through the onesided KA kernels
    (`KAKernels.jl`): each kernel now takes two scalar cutoff arguments
    instead of one (grid pitch for `_cell_1idx`, `cutoff_sq` for the pairwise
    filter) rather than deriving both from a single reused value.

    **Scope, opt-in and deliberately narrow** — a new `verlet_skin::T = 0`
    kwarg on `LeapFrogTimeIntegrator`/`RK4TimeIntegrator` (`TimeIntegration.jl`),
    validated by `_validate_verlet_skin`:
    - `verlet_skin == 0` (default) reproduces today's rebuild-every-step
      behaviour exactly — zero risk to any of the 13 scripts.
    - `verlet_skin > 0` requires every interaction to be `onesided=true`
      (`OnesidedCPU`/`OnesidedKA`) — `ArgumentError` otherwise. The coloured
      sweep's own `cutoff`/`cutoff_sq` split (same conflation, different
      call sites, e.g. `_sweep_coupled!`'s `cell_x_min`/`cell_x_max`
      derivation) was **not** fixed, since nothing exercises it under skin —
      matches item 11's own precedent of touching only the code path the
      GPU launch-count story is actually about.
    - `verlet_skin > 0` requires empty `ghosts`/`virtual_systems` —
      `ArgumentError` otherwise. Ghosts are fully regenerated from live
      boundary positions every step regardless of skin
      (`generate_ghosts!`); tracking their staleness is a separate,
      unbuilt problem.
    - `verlet_skin` must be `< 2 * min(interaction cutoff)` — the existing
      `2*cutoff` bounding-box padding in `_create_grid_impl!` is what
      guarantees a drifting particle can't walk off the edge of a stale
      grid into an out-of-bounds cell index; skin has to stay inside that
      headroom.

    **Mechanism** (`time_integrate!`, both loops): one reference-position
    buffer per tracked system (every system in `sys`, plus each
    interaction's `system_b`), snapshotted right after every rebuild. Each
    subsequent step computes `max_disp` (a `maximum(norm, ...)` reduction
    per tracked system, reusing a preallocated scratch buffer — no
    per-step GPU allocation) and skips `_sort_all_systems!` +
    `_prepare_grids!` entirely whenever `2 * max_disp <= verlet_skin` — the
    standard Verlet-list bound. `_maybe_save!` now returns whether it
    measured probes this step (`_measure_probes!` re-sorts a probe source
    system independently of the main gate, at save cadence); the caller
    forces a full rebuild on the following step when it does, since that
    re-sort invalidates the "same array index = same particle" assumption
    the cached reference buffers depend on.

    **Why a cached run reproduces an always-rebuild run exactly, not just
    approximately**: `sort_particles!`'s permutation is a *stable* sort —
    ties (particles still in the same cell) preserve whatever order they
    already had. A particle's cell assignment under the padded grid can
    only change by crossing a padded-cell boundary, which is exactly the
    event `2*max_disp <= verlet_skin` guarantees hasn't happened. So
    skipping the re-sort reproduces what re-running it *would* have
    produced (a no-op, same order) bit-for-bit, and skipping `create_grid!`
    likewise reproduces identical `_mingridx`/`_ngridx`/`_cell_start` —
    confirmed empirically (see Validation below): every equivalence test
    compares a `verlet_skin > 0` run against a `verlet_skin = 0` run and
    finds exact or near-machine-epsilon agreement, not merely
    tolerance-level agreement.

    **Validation**: `test/test_verlet_skin.jl` (new, 27 tests) — all four
    `ArgumentError` guards; `create_grid!(si, skin)`'s default-vs-explicit
    `_grid_cutoff` behaviour; `LeapFrogTimeIntegrator`/`RK4TimeIntegrator`
    equivalence (`verlet_skin > 0` vs `= 0`, compared by particle `id` since
    the two runs re-sort a different number of times and can end up with
    different index-to-particle mappings) under both tame motion (rebuild
    only ever triggers once, at step 1) and fast motion (an imposed initial
    velocity forces many rebuild-trigger events over the run, exercising the
    skip/rebuild state machine's transitions, not just its initial branch);
    the same equivalence check on `KA.CPU()` (`ka=true`); and the
    probe-triggered forced-rebuild path across a save boundary. Every
    equivalence check passed at the tight tolerance used
    (`rtol=1e-9`), most at exact or near-machine-epsilon agreement, matching
    the argument above. `test/test_gpu_cuda.jl` gained a new real-CUDA
    testset ("verlet_skin > 0 (padded grid), self+coupled onesided: CPU
    oracle vs CUDA") exercising the two-cutoff KA kernel argument split on
    actual hardware, 8/8 passing (`_grid_cutoff`, `_cell_start`,
    `_mingridx`/`_ngridx`, `dvdt`/`drhodt` all matching a CPU oracle).
    Full suite: **1691/1691** (up from 1656 after item 11).

    **Benchmark** (`bench/dambreak_scaling.jl`, extended with a 5th column,
    `gpu_1s_skin` — `OnesidedKA` + `verlet_skin = 0.2 * cutoff` — run at the
    same default sizes, same hardware, RTX 4060 Laptop):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | skin/1s |
    |---|---|---|---|---|---|---|---|
    | 50  | 2,500   | 329.5   | 556.5   | 1385.4  | 6129.5  | 890.4   | **0.643** |
    | 100 | 10,000  | 1814.8  | 2921.1  | 1998.9  | 10952.4 | 1527.6  | **0.764** |
    | 200 | 40,000  | 6689.1  | 10604.6 | 4525.1  | 12292.6 | 4337.8  | 0.959 |
    | 320 | 102,400 | 13334.9 | 28426.9 | 9042.1  | 10941.5 | 9261.2  | 1.024 |
    | 450 | 202,500 | 29141.5 | 55450.5 | 17083.8 | 11514.8 | 17154.6 | 1.004 |

    (`skin/1s` = `gpu_1s+skin / gpu_1s`; below 1.0 means skin caching won.)

    **A partial, genuinely useful win — unlike item 11's fully negative
    result, this one lands right where it matters most**: at dambreak.jl's
    actual production scale (`n_fluid = 2,500`, `nfx = 50`), skin caching
    cuts `OnesidedKA`'s per-step cost by **1.56×** (1385→890µs), and still
    by 1.31× at 10,000 particles — this benchmark's fluid block falls from
    rest under gravity, so per-step displacement is tiny relative to a
    `0.2*cutoff` skin margin and almost every step after the first skips the
    rebuild entirely. The win shrinks as `n_fluid` grows (0.96× at 40,000,
    roughly break-even from 100,000 up) because the fixed rebuild cost being
    saved becomes a smaller fraction of a total that now scales with `n`,
    while the displacement check itself (2 reduction kernels × 2 tracked
    systems = 4 launches/step) is a fixed cost paid on *every* step,
    skipped or not. It does **not** close the remaining gap to CPU-coloured
    at dambreak's own scale (`gpu_1s+skin` at 890µs is still ~2.7× slower
    than `cpu_col`'s 330µs) — this item narrows that gap, it doesn't erase
    it. The crossover against CPU-coloured (item 2's `n_fluid ≈ 40,000` for
    plain `OnesidedKA`) isn't materially moved by this item, since skin's
    benefit is concentrated below that scale, not near it.

    **Not built** (see "Explicitly deferred" below, which this item leaves
    otherwise unchanged): fusing the sort's gather+copyback into a single
    `Ref`-swap. Investigated during planning and descoped — it would require
    every per-particle array field across the whole particle-system type
    hierarchy to become a swappable container instead of a plain array
    field, touching every `ps.x`/`ps.v`/... read site in the codebase, for a
    benefit (removing 1 of 2 already-batched permutation-apply kernel
    launches) that shrinks further now that most steps skip the sort
    entirely. Ghost/virtual-aware staleness tracking, and any change to the
    coloured sweep's grid-pitch handling, are also explicitly out of scope —
    see the constructor guards above.

13. **Server hardware validation (A100 + V100 + H200) — done.** `docs/server-
    handoff-2026-08-10.md` handed off one open question: items 11 and 12
    were measured only on the RTX 4060 Laptop, and both conclusions rest on
    that GPU having no FP64 compute edge over its own CPU and a
    ~8.3µs/launch overhead floor. Does that story hold on server hardware
    with a real compute/bandwidth edge and far more VRAM headroom? The
    handoff was originally picked up independently, same day, on two
    different machines (A100, H200) — a third machine (V100) was added in a
    follow-up session on request, extending coverage to three GPU
    generations. Results below are merged from all three rather than kept
    as separate docs, so they can be compared directly instead of read
    apart.

    _**A100 run**: `mlerp-monash-node05` (Slurm job 158888, `BigCats`
    partition): NVIDIA **A100-PCIE-40GB** (sm_80, ECC on, persistence mode
    on), 26 allocated Intel Xeon (Icelake) cores of a 52-core node, Julia
    1.12.6, CUDA.jl 5.8.5. Compute and login access were the same node, with
    outbound internet, so environment setup needed no special handling._

    _**H200 run**: NCI Gadi, PBS job `175894287` (`gpuhopper-exec` queue):
    NVIDIA **H200** (140GB VRAM, Hopper/sm_90), 12 allocated CPUs, 64GB RAM,
    driver 580.173.02. Gadi's compute nodes have no outbound internet, so
    `Pkg.Registry.update()`/`instantiate`/the merged benchmark env were all
    built on the login node first, sharing the same NFS-mounted `~/.julia`
    depot the compute node (`gadi-gpu-h200-0017`) then read with no further
    network access needed — see "Environment notes" above for the gotcha
    this surfaced (SSH one-shot commands landing in the wrong directory)._

    _**V100 run**: NCI Gadi, PBS job `175894283` (`gpuvolta` queue): NVIDIA
    **Tesla V100-SXM2-32GB** (Volta/sm_70), 12 allocated CPUs (Intel Xeon
    Platinum 8268 @ 2.90GHz), 64GB RAM, driver 580.173.02. Same login/
    compute split as the H200 run, same shared NFS `~/.julia` depot. One
    additional wrinkle this surfaced: that depot's `benchenv` had already
    resolved `CUDA_Runtime_jll` to 13.3 while setting up for the H200, and
    **CUDA 13 dropped `ptxas` support for Volta (`sm_70`)** —
    `dambreak_scaling.jl` failed immediately with `ptxas fatal: Value
    'sm_70' is not defined for option 'gpu-name'` on first kernel launch.
    Fixed on the login node with `CUDA.set_runtime_version!(v"12.6")`
    against `benchenv` (writes a `LocalPreferences.toml`, downloads the
    older toolkit — needs internet, hence login node, not compute node);
    12.6 still supports Ampere/Hopper, so this doesn't regress the A100/H200
    paths sharing the same depot. See "Environment notes for the next
    machine move" below._

    **Test suite**, reconfirmed before trusting any timing on either machine:

    | where | GPU functional? | result |
    |---|---|---|
    | A100 node (compute+login combined) | yes | **1691 passed, 0 failed** |
    | Gadi login node (`gadi-login-06`) | no | 1517 passed, 3 broken, 0 failed |
    | Gadi H200 compute node | yes | **1691 passed, 0 failed** |

    The login-node shortfall is just GPU-gated tests reporting `broken`
    rather than running at all without a functional device — both real-GPU
    runs match the handoff's expected 1691/0 exactly. This item changes no
    source code, only `bench/` and this doc.

    **The microbenchmark numbers item 2 quoted for the laptop were never
    turned into a script or committed.** `bench/gpu_microbench.jl` (new this
    item, A100 run only — the H200 run used step-timing alone, the same
    methodology as item 2's original crossover benchmark) fixes that,
    measuring launch overhead, FP64 FMA throughput (device and host), and
    device bandwidth directly instead of inferring them from step timings:

    | | RTX 4060 Laptop (item 2) | A100 (this item) | ratio |
    |---|---|---|---|
    | launch overhead | ~8.3 µs | 38-40 µs | 4.6-4.8× **higher** |
    | GPU FP64 FMA | 0.138 TFLOP/s | 5.24-5.25 TFLOP/s | ~38× |
    | GPU memory bandwidth | 216 GB/s | ~1400 GB/s | ~6.5× |
    | CPU FP64 FMA (multi-core) | 0.143-0.311 TFLOP/s (16-core, thread count unrecorded) | 0.1331 TFLOP/s (26-core) | roughly comparable |

    (`launch_us`: near-zero-work KA kernel, `ndrange=1`, 2000 synchronized
    reps, same `@kernel`/launch path the real sweep kernels use — not a raw
    CUDA-API lower bound. `gpu_tflops`/`cpu_tflops`: chained-FMA kernel, 8
    independent accumulators per thread so a single CPU thread measures
    pipelined throughput rather than FMA *latency* — the GPU kernel needed no
    such fix, since thousands of concurrent threads already hide one
    another's latency the same way. `gpu_gb_s`: STREAM-triad-style kernel. No
    equivalent H200 numbers exist yet — running this script there is
    unstarted follow-up work, not done.)

    **The laptop's defining fact — no FP64 compute edge over its own CPU —
    does not hold on either server GPU.** The A100 has both the bandwidth
    edge the laptop also had (~6.5×) and a large compute edge it never had
    (~38× the laptop GPU's own throughput); H200's own class of hardware is
    stronger again on both axes (below). **Launch overhead moved the
    "wrong" way for the hypothesis that server hardware would fix
    `ColouredKA`'s crossover** — the A100's measured overhead is *higher*
    than the laptop's, not lower, though the two figures were never
    guaranteed to be methodologically comparable (item 2's number came from
    an ad hoc script that was never committed, so its exact methodology
    can't be checked). Whatever the cause, both effects push the same way:
    more launches are, if anything, more expensive on server hardware, and
    halving arithmetic is worth even less when compute is already this
    cheap — which is exactly what all three server GPUs' `col/1s` numbers
    below show.

    **Thread count matters more than the laptop numbers can settle.** Nothing
    in this repo sets `Threads.nthreads()` (Julia defaults to 1), so on the
    A100 `bench/dambreak_scaling.jl` was run twice — `-t 1` and `-t 26` (that
    job's full Slurm CPU allocation) — to bracket whichever thread count the
    laptop numbers actually used, which the original doc does not record
    (the H200 run used its full 12-core PBS allocation throughout, not
    split this way). The CPU-FMA microbenchmark shows the honest cost of not
    knowing: 0.0101 TFLOP/s at 1 thread vs. 0.1331 TFLOP/s at 26 (13.2× from
    26× the threads, ~51% parallel efficiency) — but the real
    `cpu_col`/`cpu_1s` sweep columns below scale far worse at their own
    largest size (`n_fluid = 202,500`: 3.74×/4.85× from the same 26×, 14-19%
    efficiency), because the SPH sweep is memory-bandwidth-bound with a
    scattered cell-list access pattern, not the embarrassingly-parallel
    compute-bound loop the microbenchmark measures.

    **Four machines, one tie point** (`nfx = 450`, `n_fluid = 202,500` —
    the laptop's own ceiling, and a size every run tested):

    | machine | cores | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | RTX 4060 Laptop | unrecorded | 29,141.5 | 55,450.5 | 17,083.8 | 11,514.8 | 17,154.6 | 0.672 | 1.004 |
    | A100 | 26 | 16,080.9 | 21,328.5 | 3,518.6  | 6,414.7  | 1,325.6  | 1.823 | 0.377 |
    | V100 | 12 | 15,429.0 | 21,414.3 | 2,857.9  | 7,855.7  | 1,324.0  | 2.749 | 0.463 |
    | H200 | 12 | 30,522.9 | 51,981.1 | 1,576.6  | 5,040.5  | 856.2    | 3.197 | 0.543 |

    `gpu_1s` falls from the laptop's 17,084µs to three much closer server-GPU
    figures — 3,519µs (A100), 2,858µs (V100), 1,577µs (H200) — and `col/1s`
    rises from the laptop's 0.672 through 1.823 (A100) and 2.749 (V100) to
    3.197 (H200). The `col/1s` climb is monotonic across all three server
    GPUs and is the single cleanest confirmation that `ColouredKA`'s problem
    is structural, not laptop-specific: every step up in raw GPU power makes
    the 15-launches-vs-2 gap matter *more*, not less. `gpu_1s` itself is
    *not* quite monotonic with a naive "GPU generation" ordering — the V100
    (Volta, 2017) edges out the newer A100 here (2,858 vs 3,519µs), even
    though A100 has substantially more memory bandwidth and SMs on paper.
    That gap (~23%) sits right at the ~15-25% run-to-run noise band
    `us_per_step`'s single-warmup-then-timed-pass design already produces
    (see below), so it reads as most likely noise rather than a genuine
    "V100 beats A100" result — but it means `gpu_1s` alone isn't as clean a
    monotonic story as `col/1s` is. `skin/1s` stays solidly below 1.0 on all
    three server GPUs (0.377 A100, 0.463 V100, 0.543 H200) — a clear
    improvement on the laptop's 1.004 breakeven at this exact size, though
    (same noise caveat) the exact values shouldn't be over-read past ~±20%.

    **A100 default sizes** (`nfx` = 50/100/200/320/450 → `n_fluid` = 2,500 to
    202,500 — the laptop's full tested range), Run A (`-t 1`):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 50  | 2,500   | 668.4    | 1,195.6  | 1,416.5 | 4,386.0 | 746.3   | 3.096 | 0.527 |
    | 100 | 10,000  | 2,753.2  | 4,713.9  | 1,625.8 | 5,790.3 | 591.7   | 3.561 | 0.364 |
    | 200 | 40,000  | 13,284.9 | 23,084.0 | 1,792.6 | 6,327.9 | 645.8   | 3.530 | 0.360 |
    | 320 | 102,400 | 32,782.1 | 55,955.4 | 2,214.4 | 6,162.2 | 860.8   | 2.783 | 0.389 |
    | 450 | 202,500 | 60,180.9 | 103,402.1| 3,434.6 | 6,287.1 | 1,262.3 | 1.830 | 0.368 |

    Run B (`-t 26`, this job's full CPU allocation — the realistic server
    baseline):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 50  | 2,500   | 380.8    | 370.4    | 1,639.5 | 4,577.9 | 810.1   | 2.792 | 0.494 |
    | 100 | 10,000  | 1,024.3  | 1,150.3  | 1,608.6 | 5,735.9 | 675.0   | 3.566 | 0.420 |
    | 200 | 40,000  | 4,431.2  | 5,825.5  | 1,877.9 | 5,878.0 | 760.4   | 3.130 | 0.405 |
    | 320 | 102,400 | 10,898.1 | 14,047.4 | 2,315.1 | 6,103.2 | 906.1   | 2.636 | 0.391 |
    | 450 | 202,500 | 16,080.9 | 21,328.5 | 3,518.6 | 6,414.7 | 1,325.6 | 1.823 | 0.377 |

    GPU columns agree between Run A and Run B to within ~15% at every size
    (expected — host thread count shouldn't affect device work; the residual
    is timing noise from `us_per_step`'s single warmup-pass-then-timed-pass
    design, with no repeated-trial averaging). `col/1s` and `skin/1s` tell
    the same story in both runs regardless of that noise, which is the point
    of running both.

    **A100 extended sizes** (`nfx` = 450/640/900/1273/1800 → `n_fluid` =
    202,500 to 3,240,000 — up to 16× past the laptop's ceiling, `-t 26`,
    `--budget 5e7` to keep step counts off their floor at the top end; a
    scale-safety review of index widths, grid memory, and KA launch sizing
    preceded this run and found no correctness risk this far out — indices
    are all `Int`/`UInt64` with 2-3 orders of magnitude of headroom, and grid
    memory stays linear in particle count, ~54 MB of cell-list arrays at the
    top size):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 450  | 202,500   | 17,553.8  | 23,242.8  | 2,920.4  | 6,576.7  | 1,654.5  | 2.252 | 0.567 |
    | 640  | 409,600   | 30,086.2  | 40,977.1  | 4,749.4  | 7,511.2  | 2,028.8  | 1.582 | 0.427 |
    | 900  | 810,000   | 55,140.5  | 67,782.0  | 7,566.6  | 10,760.1 | 3,276.9  | 1.422 | 0.433 |
    | 1273 | 1,620,529 | 93,862.4  | 119,222.9 | 14,582.4 | 20,637.9 | 6,134.8  | 1.415 | 0.421 |
    | 1800 | 3,240,000 | 198,821.6 | 225,910.7 | 29,843.8 | 39,757.9 | 12,324.0 | 1.332 | 0.413 |

    (`nfx = 450` repeats as a tie point against Run B, at a higher step-count
    budget: `col/1s` 1.823 → 2.252, `skin/1s` 0.377 → 0.567 — a bigger swing
    than the Run A/B cross-check, from timing 247 steps here vs. 99 there.
    The qualitative conclusions below are unaffected by this noise; read the
    precise ratios as ±20-25%, not exact — the same caveat applies to the H200
    extended table's own `nfx = 450` repeat just below, ~10% between its two
    runs by the same mechanism.)

    **H200 default sizes** (same five sizes as the A100/laptop tables, 12
    allocated cores throughout):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 50  | 2,500   | 328.7   | 553.1   | 821.8  | 3,851.8 | 544.0 | 4.687 | 0.662 |
    | 100 | 10,000  | 1,379.8 | 2,346.2 | 830.1  | 4,935.4 | 473.3 | 5.945 | 0.570 |
    | 200 | 40,000  | 7,136.5 | 12,158.2| 920.1  | 5,024.6 | 504.4 | 5.461 | 0.548 |
    | 320 | 102,400 | 17,479.7| 30,748.1| 1,126.0| 5,182.7 | 638.5 | 4.603 | 0.567 |
    | 450 | 202,500 | 30,522.9| 51,981.1| 1,576.6| 5,040.5 | 856.2 | 3.197 | 0.543 |

    **H200 extended sizes** (`--sizes 450,600,800,1000,1300` — up to 8.3×
    past the laptop's ceiling):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 450  | 202,500   | 31,566.4  | 52,110.5  | 1,730.5 | 5,064.6  | 1,010.1 | 2.927 | 0.584 |
    | 600  | 360,000   | 49,863.4  | 89,441.2  | 2,487.5 | 5,501.7  | 1,332.0 | 2.212 | 0.535 |
    | 800  | 640,000   | 86,942.8  | 150,575.8 | 4,101.7 | 6,538.0  | 1,682.2 | 1.594 | 0.410 |
    | 1000 | 1,000,000 | 137,584.6 | 236,786.0 | 5,231.7 | 7,818.8  | 2,537.9 | 1.494 | 0.485 |
    | 1300 | 1,690,000 | 229,661.9 | 405,445.5 | 8,221.9 | 11,548.8 | 3,910.0 | 1.405 | 0.476 |

    **V100 default sizes** (same five sizes, 12 allocated cores, CUDA
    runtime pinned to 12.6 — see the environment note above):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 50  | 2,500   | 200.8    | 233.8    | 1,172.8 | 4,458.7 | 517.4  | 3.802 | 0.441 |
    | 100 | 10,000  | 716.1    | 995.5    | 1,358.0 | 5,263.0 | 755.4  | 3.876 | 0.556 |
    | 200 | 40,000  | 3,884.7  | 5,806.4  | 1,592.5 | 5,484.9 | 584.0  | 3.444 | 0.367 |
    | 320 | 102,400 | 8,564.5  | 12,705.5 | 2,089.2 | 6,462.4 | 919.6  | 3.093 | 0.440 |
    | 450 | 202,500 | 15,429.0 | 21,414.3 | 2,857.9 | 7,855.7 | 1,324.0| 2.749 | 0.463 |

    **V100 extended sizes** (`--sizes 450,600,800,1000,1300`, matching the
    H200 run exactly for direct comparison; up to 1,690,000 particles with
    no CUDA out-of-memory issues on this card's 32GB — comfortably inside
    the same linear-in-`n`, ~54MB-at-the-top-end grid memory footprint the
    A100 extended run's scale-safety review already established):

    | nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
    |---|---|---|---|---|---|---|---|---|
    | 450  | 202,500   | 14,629.4  | 20,937.7  | 3,094.2  | 7,828.5  | 1,379.5 | 2.530 | 0.446 |
    | 600  | 360,000   | 24,309.2  | 32,024.7  | 4,878.0  | 10,643.6 | 2,305.8 | 2.182 | 0.473 |
    | 800  | 640,000   | 39,077.2  | 51,040.8  | 8,703.8  | 18,120.6 | 3,961.7 | 2.082 | 0.455 |
    | 1000 | 1,000,000 | 69,185.3  | 83,617.7  | 11,986.8 | 29,421.1 | 5,603.7 | 2.454 | 0.467 |
    | 1300 | 1,690,000 | 116,528.6 | 147,080.4 | 24,132.4 | 53,693.8 | 9,363.2 | 2.225 | 0.388 |

    **Item 11's crossover does not reappear on any of the three server
    GPUs — it gets pushed off the entire tested range on all of them.** The
    laptop crossed `col/1s < 1.0` at its largest size, `n_fluid = 202,500`
    (`col/1s = 0.672`). On the A100, `col/1s` falls monotonically across the
    extended run's five points as `n_fluid` grows — 2.252 → 1.582 → 1.422 →
    1.415 → 1.332 — but is still comfortably above 1.0 at 3,240,000
    particles. On the V100, `col/1s` stays in a tighter 2.08-2.53 band
    across the same extended range without as clean a monotonic trend
    (2.530 → 2.182 → 2.082 → 2.454 → 2.225 — likely the same run-to-run
    noise flagged elsewhere in this item), but never comes close to 1.0
    either. On the H200 the same pattern holds even more strongly at small
    sizes (4.687 → 5.945 → 5.461 → 4.603 → 3.197 by `n_fluid = 202,500`,
    i.e. `ColouredKA` loses by up to ~5.9× rather than the A100's ~3.6×) and
    settles at 1.405 by 1,690,000 particles, still above 1.0. This directly
    answers the handoff's question on all three machines: **no, the
    crossover does not move to a smaller `n_fluid` on server hardware; if
    anything `ColouredKA` is further from paying off than on the laptop, and
    further still on H200, the fastest of the three server GPUs tested** —
    exactly the monotonic trend the four-way tie-point table above shows
    directly for `col/1s`. The
    likely reason isn't just launch overhead (which item 2's mechanism alone
    would predict shrinking in relative terms as compute gets cheaper) — a
    scale-safety review that preceded the A100 extended run found
    `ColouredKA`'s coupled-interaction colours launch one thread per *cell*
    in the fluid bounding box regardless of whether a boundary particle is
    actually nearby (`src/KAKernels.jl`, coupled colour loop), so it carries
    a fixed per-cell tax that `OnesidedKA`'s per-particle traversal doesn't
    pay — consistent with all three machines' ratios flattening out well
    above 1.0 instead of continuing to fall toward it. **`ColouredKA`
    remains correctly out of scope for every script** — this item does not
    change item 11's "not wired into any script" conclusion; all three
    independent runs reinforce it.

    **Item 12's skin caching does not taper off on any of the three
    machines — it stays a clear win across the entire tested range.** The
    laptop's `skin/1s` crossed *above* 1.0 (stopped helping) around
    `n_fluid = 100,000` and sat at breakeven (1.004) by 202,500. On the A100
    it never leaves the 0.36-0.57 band across all three A100 runs, all the
    way to 3,240,000 particles. On the V100 it stays in a similar 0.39-0.47
    band across the same extended range. On the H200 it stays in a
    near-identical 0.41-0.66 band all the way to 1,690,000. All three server
    GPUs show a consistent ~1.5-2.8× speedup at every tested size, not just
    the small ones the laptop's own benefit was concentrated in. This also
    directly answers the handoff's question, in the direction items 2/11's
    own reasoning would predict: removing launches is worth *more*, not
    less, when the hardware's per-launch floor is a bigger share of an
    otherwise-cheaper step — true on all three server GPUs, independently
    measured.

    **The CPU-vs-GPU crossover (item 2) moved on every configuration tested,
    but for two different reasons that happen to land on similar numbers.**
    At `-t 1` on the A100, `gpu_1s` beats `cpu_col` from `n_fluid ≈ 10,000` —
    far earlier than the laptop's 40,000, because a single Icelake core is a
    weak baseline, not because the A100 itself is exceptional. At `-t 26` on
    the A100, the realistic server number, the crossover lands at `n_fluid ≈
    40,000` — the same point item 2 found on the laptop; whether that is a
    real coincidence or an artifact of the laptop measurement having used a
    similar effective thread count isn't knowable from the original doc,
    which never recorded `Threads.nthreads()`. On the V100 (12 cores
    throughout, the job's full allocation), the crossover also lands at
    `n_fluid ≈ 40,000` — matching the laptop and the A100's `-t 26` run, but
    for neither of those runs' specific reason: the V100's CPU baseline (a
    normal 12-core allocation, not artificially starved like the A100 `-t
    1` case) and its GPU (`gpu_1s` 1,592.5µs at this size — faster than a
    laptop GPU, nowhere near H200's edge) both land in an unremarkable
    middle ground that happens to reproduce the same number two different
    extremes landed on elsewhere. On the H200 (12 cores throughout), the
    crossover also lands at `n_fluid ≈ 10,000` — but this
    time because the GPU itself is dramatically faster (`gpu_1s` at
    `n_fluid=202,500` is 1,576.6µs vs. the A100 `-t 26` run's 3,518.6µs),
    against a CPU baseline in the same ballpark as the laptop's, not because
    the CPU side is weak. Three different mechanisms landing on two
    crossover values — a reminder that `col/1s`/`skin/1s` (GPU-vs-GPU
    ratios, immune to CPU thread count entirely) are the more reliable
    numbers to compare across machines than any CPU-involving crossover
    point.

    **Not built or changed**: no script's default changed, `ColouredKA`
    remains reachable only via the internal `mode` override (item 11's scope
    guards are untouched), and no source-level GPU code was added — this item
    is measurement and documentation only. `bench/dambreak_scaling.jl` gained
    a provenance header (thread count, CPU model, GPU name, CUDA.jl version)
    so future runs are self-describing; its CSV schema is unchanged. Running
    `bench/gpu_microbench.jl` on the H200 or V100 (no equivalent numbers
    exist yet for either) is natural, unstarted follow-up work.
    `bench-output/*.csv`/`*.log` from all three sessions are gitignored, not
    committed — re-run the commands above (merged environment, same as item
    2) to reproduce any machine's numbers. The H200 run was originally
    written up as a separate
    freestanding doc (`docs/h200-benchmark-results-2026-08-10.md`); its full
    content is preserved here and the standalone file was removed to keep one
    canonical record instead of two differently-shaped ones — see git history
    for the original if the raw, unmerged version is ever needed.

14. **Real-workload GPU validation (`bubble3.jl`, H200 + V100) — done.**
    Items 11-13 all measure `dambreak_scaling.jl`'s synthetic single/coupled-
    interaction shape. Does the GPU speedup hold on an actual production
    script — `bubble3.jl` (two-phase bubble-rise, XSPH + artificial surface
    tension, 4 `SystemInteraction`s including a ghost boundary, 37,500
    particles: 35,524 fluid-X + 1,976 fluid-Y), run end-to-end rather than
    step-timed in isolation? Run on both PBS job `175894287` (H200) and job
    `175894283` (V100), same NCI Gadi allocations as item 13, no source
    changes needed — `bubble3.jl` already supports `GRASPH_BACKEND=cuda`
    (see its own header) and `run_driver!`'s `--run-steps`/
    `--non-interactive`/`--output-prefix none` CLI flags (`src/Driver.jl`)
    already covered everything this needed.

    **Method**: wall-clock the whole `julia ... bubble3.jl --run-steps N
    --non-interactive --output-prefix none` process (shell `time`, around
    each `ssh <compute-node>` invocation) at two step counts per backend
    (`N=200`, `N=3000`) and fit a line — the slope is the steady-state
    per-step cost once every kernel/method has compiled at least once (all
    of them do, within the first ~100 steps, confirmed via a smoke-test
    run); the intercept lumps together everything that's a one-time cost
    regardless of step count: Julia startup, package precompilation, and —
    for the GPU backend specifically — every distinct KA kernel's
    first-launch PTX/SASS compile. This is deliberately different from
    `dambreak_scaling.jl`'s `us_per_step` (single warmup pass then a single
    timed pass, no fixed-cost/steady-state separation) because a
    60,000-step production run's fixed cost is negligible, but a 200-step
    smoke test's isn't — conflating the two would make either the GPU look
    artificially slow (short run swamped by warm-up) or hide a real
    per-step difference (implicitly assuming the fixed cost scales with
    `N`, which it doesn't).

    | | 12 cores (H200 job) | H200 GPU | 12 cores (V100 job) | V100 GPU |
    |---|---|---|---|---|
    | Steady-state cost/step | 7.5 ms | 3.3 ms | 8.4 ms | 4.2 ms |
    | Fixed warm-up (intercept) | ~13.3 s | ~44.5 s | ~20.3 s | ~53.1 s |
    | Wall-clock, 200 steps | 14.8 s | 45.2 s | 22.0 s | 54.0 s |
    | Wall-clock, 3000 steps | 35.7 s | 54.4 s | 45.6 s | 65.7 s |
    | Break-even step count | ~7,400 | | ~7,730 | |
    | Projected @ 60,000 steps (script's actual default) | ~463 s | ~242 s | ~527 s | ~305 s |

    (The two jobs' CPU steady-state costs differ — 7.5ms vs 8.4ms/step —
    because they land on different physical node types, `gpuhopper-exec`'s
    vs `gpuvolta`'s CPU allocation, not because of anything GPU-related;
    item 13's own `dambreak_scaling.jl` provenance header independently
    confirms the V100 job's CPU as an Intel Xeon Platinum 8268.)

    **GPU wins end-to-end at the script's real 60,000-step default on both
    cards** — 1.9× on H200, 1.73× on V100 — despite paying a warm-up cost
    3-4× larger than the CPU backend's own Julia-startup-only fixed cost.
    Below the ~7,400-7,730-step break-even, plain CPU actually finishes
    *sooner* in wall-clock terms purely because of that warm-up tax — a
    result `dambreak_scaling.jl`'s step-timing methodology structurally
    cannot see, since it explicitly discards a warmup pass before timing.
    Both cards' per-step GPU-vs-CPU ratio here (2.0-2.3×) is far smaller
    than `dambreak_scaling.jl`'s own `gpu_1s`-vs-`cpu_col` ratio at a
    comparable particle count (item 13's tie point: ~19.4× on H200, ~5.4×
    on V100) — expected, since `bubble3.jl` spends much of its per-step
    budget on ghost regeneration, XSPH, and sorting across 4 interactions,
    work that doesn't offload to GPU as cleanly as a single bare
    fluid-fluid sweep.

    **Not built or changed**: no source change, benchmark-only, same as item
    13. No CSV/committed script artifact — the wall-clock numbers came from
    `time` around each `ssh` invocation using the existing driver CLI, not a
    new benchmark script; the raw `*.log`/`*.time` files live under a
    scratch directory outside the repo, not `bench-output/`.

## Explicitly deferred (not started, not part of the current scope)

- ~~**GPU (`ka=true`) support for any pfn converted in Phase C**~~ — items
  5-9 above are all done now; nothing left deferred here.
- ~~Virtual particle systems, probes, RK4 integrator... none have GPU-resident
  *sweep-orchestration* support yet~~ — **done, item 9.** Ghost particles
  (item 7), Virtual/probe/RK4 orchestration (item 9), and
  `StressParticleSystem`/`ElastoPlasticParticleSystem`'s `device_view`/`Adapt`
  support (item 5) are all in place now; nothing in this category remains
  unstarted.
- ~~Wiring `GRASPH_BACKEND` into the 6 experiment scripts item 9 unblocked~~
  — **done, item 10.** All 13 experiment scripts now support
  `GRASPH_BACKEND=cuda`; nothing left deferred in this category.
- ~~Persistent/cached grid + Verlet-skin rebuild cadence~~ — **done, item
  12.** Opt-in (`verlet_skin` kwarg, default `0`), onesided-only, no
  ghosts/virtual — see item 12 for the scope guards and why. Fusing the
  sort's gather+copyback into a single `Ref`-swap (the third part of this
  originally-bundled bullet) stays deferred — investigated and explicitly
  descoped in item 12 ("Not built"), not merely unstarted.
- Morton/Z-order sort keys (packed lexicographic `UInt64` key shipped
  instead).
- Explicit neighbour list (on-the-fly 27-cell scan is current approach).
- Float32/mixed precision (Float64 retained per the GPU-target decision —
  though note the hardware section above: this machine gets zero benefit
  from that decision either way).
- Multi-GPU + MPI/ORB integration.

## Practical notes for picking this back up

- Branch: `onesided-sweep-gpu-prep`, based on `main` at commit `37e9a5a`, now
  32 commits ahead of it (`12ac526`..`149c238`). Nothing on this branch has
  been pushed to any remote.
- Run the full suite with `julia --project -e 'using Pkg; Pkg.test()'` —
  should show `1691/1691` (834 through Phase B1, up to 935 after Phase B2's
  3D work, up to 1371 after Phase C, up to 1433 after item 5's `device_view`
  extension, up to 1466 after item 6's reverse-sweep KA kernel twin, up to
  1479 after also fixing the `FluidPfn` fluid-fluid `ka=true` dispatch gap
  (this doc previously miscited this figure as 1475 — see update #5), up to
  1496 after also fixing `FluidSolidPfn`'s identical gap (item 8), up to
  1600 after item 7 (`GhostParticleSystem` GPU residency plus its
  adversarial-review test additions), up to 1651 after item 9
  (`ProbeParticleSystem`'s `device_view`/`Adapt` extension plus real-CUDA
  tests for the RK4/Virtual/Probe fixes), up to 1653 after item 10 (the
  `VirtualParticleSystem` `w_sum` buffer-type regression test), up to 1656
  after item 11 (the `ColouredKA` self+coupled real-CUDA testset), up to 1691
  after item 12 (`test/test_verlet_skin.jl`'s 27 new tests plus one new
  real-CUDA testset in `test_gpu_cuda.jl`).
  `Pkg.test()` resolves its own CUDA
  from `test/Project.toml` and picks up real hardware automatically when
  present (confirmed again this item) — no merged-throwaway-environment
  workaround needed for the test suite itself, only for running a top-level
  driver script directly (see "Environment notes" above).
- `test/test_onesided_sweep.jl` (Phase A/C, ~1100 lines by now — one section
  per pfn/shape, including the `XSPHPfn` ghost-aliasing regression test) and
  `test/test_adapt.jl` (Phase B1) are the two long-running test files that
  keep growing with each phase. Phase C also added nine new standalone files,
  `test/test_onesided_integration_{soil2d,soil3d,virtual,trapdoor,bubble,
  bubble3,ellipse,dambreakwall,cantilever}.jl` — one per interaction shape
  across the 11 non-dambreak scripts, all wired into `test/runtests.jl`.
  Item 5 added `test/test_device_views.jl` (proxy-correctness, dispatch
  equivalence between host and `device_view`'d systems, and the
  `VirtualNormUpdater`/`PrescribedVelocityUpdater` `getfield`-bypass
  regression test), also wired in.
  Phase B2's `device_view`/KA-kernel work has its own three tiers
  (`test_ka_cpu.jl`, `test_gpu_cuda.jl`, `test_gpu_dambreak.jl`) — see next
  steps item 1 for what's still ad hoc there vs. checked in. Item 6 extended
  `test_ka_cpu.jl` (reverse/`WritesBoth` sweep, `KA.CPU()`-vs-Polyester
  equivalence, 2D and 3D, plus a `DeviceSystem` `Kind`-mismatch regression
  test) and `test_gpu_cuda.jl` (the same shapes, including a real `FluidPfn`
  fluid-fluid entry, against real `CUDABackend()`). Item 8's `FluidSolidPfn`
  fix added the same shape of tests again (2D/3D equivalence, mismatched-
  pairing regression, real-CUDA entry), plus a fluid-vs-solid-pressure
  regression test specific to that pfn's asymmetric physics. Item 7 extended
  `test_adapt.jl`/`test_device_views.jl`/`test_ka_cpu.jl`/`test_gpu_cuda.jl`
  with ghost-specific sections (adapt round-trip including a live
  post-adapt `generate_ghosts!` call, `device_view` proxy/dispatch
  equivalence, a full `KA.CPU()` sweep-equivalence test, and real-CUDA tests
  for `generate_ghosts!`/`update_ghost_kinematics!`/`GhostCopier` — including
  `HouseholderReflect` mode, 3D, the real `NB=8` wall+corner shape, and a
  deliberately-constructed capacity-above-a-shrunk-count regime — plus a full
  sort+grid+sweep pipeline test), and added two new CPU-only regression tests
  to `test/test_ghost_particles.jl` (the nested-ghost constructor guard, and
  `write_h5(ghost, ...)`'s new `1:n`-slicing behaviour, which had zero prior
  coverage before this item).
  Item 9 extended `test/test_adapt.jl`/`test/test_device_views.jl` with
  `ProbeParticleSystem` sections mirroring Virtual's exactly (adapt
  round-trip including the `mirror_target` aliasing caveat, `device_view`
  proxy/dispatch equivalence, a `getproperty`-vs-`getfield` state-updater
  regression test, narrow-typing checks), extended `test/test_ka_cpu.jl` with
  a probe-target `InterpolateFieldFn`/`NeighborCountFn` reverse-sweep
  equivalence pair, and added three new real-CUDA testsets to
  `test/test_gpu_cuda.jl` (RK4+ghost via the bubble-like fixtures, Virtual
  with nonzero `prescribed_v`, Probe with a self-referencing `mirror_target`)
  — the only tier that could actually catch this item's real bug (the
  `Ref`-less `_axpy_const_ip!` broadcast), since it's invisible on `KA.CPU()`.
  Item 10 added one more regression case to the same "device_view is isbits
  after cudaconvert" testset in `test/test_gpu_cuda.jl` (a second
  `VirtualParticleSystem` built directly around a first one's already-adapted
  source), for the same reason: the `w_sum` buffer-type bug it caught is only
  reachable on a real (or `cudaconvert`-checked) non-CPU backend.
- `CUDA.jl` is now confirmed to install and run correctly on a real GPU from
  this repo's dependency graph (see environment notes above) — the earlier
  "no GPU in the dev environment this was built in" caveat throughout Phase
  A/B1 no longer applies. Phase C and item 5 were both done on a CPU-only
  machine again, though (`device_view` correctness is fully checkable via
  `KA.CPU()` and plain field/dispatch comparisons — no CUDA hardware needed
  until a kernel actually launches on `CUDABackend()`). Item 6 landed back on
  a machine with a functional GPU (RTX 4060 Laptop, sm_89, 8 GB — the same
  card Phase B2 used), confirmed via `nvidia-smi` and `CUDA.functional()`,
  and `Pkg.test()` picked it up automatically (`test/Project.toml`'s CUDA
  dependency resolves and runs for real, no merged-throwaway-environment
  workaround needed this time) — so item 6 is validated on real hardware,
  not just `KA.CPU()`, including the two new `test_gpu_cuda.jl` testsets.
