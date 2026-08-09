# GraSPH.jl → GPU (CUDA.jl) migration: status and plan

_Last updated: 2026-08-09. Branch: `onesided-sweep-gpu-prep` (off `main`). Written
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
_previously-unreachable dispatch gap in `FluidPfn`'s fluid-fluid method — see_
_the item for what it is and why it's deliberately left unfixed here._

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
  `Pkg.test()` — the committed `Manifest.toml` pins `julia_version = "1.12.6"`
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

   `ProbeParticleSystem`/ghost systems (`GhostParticleSystem`) are **not**
   covered and were deliberately dropped from this item's scope: both
   hardcode `Vector` for their per-particle arrays (`x`, `id`, `w_sum`/`v`/
   `rho`, etc.) rather than the array-type-generic parameter every other
   system type uses, which blocks `Adapt.jl` entirely regardless of
   `device_view` — a deeper struct change than "add a device_view method",
   and ghosts specifically are already covered by item 7 below (their
   `resize!`-based generation is the harder problem anyway, not the missing
   proxy). Revisit `ProbeParticleSystem`'s array-type genericity whenever
   probes get their own GPU story (item 9).

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
   unlike Phase C/item 5. Suite: 1466/1466 (up from 1433).

   **A real, previously-unreachable dispatch gap found and left unfixed,
   scoped together with the existing `FluidSolidPfn` gap (item 8's note)**:
   `FluidPfn`'s fluid-fluid `pfn_contribution`/`_onesided_zero_coupled`
   methods (`PairwiseFunctors.jl`) are typed on the *concrete*
   `FluidParticleSystem{T,ND}` on **both** sides — needed to disambiguate
   them from `FluidPfn`'s other coupled methods, which all key off a
   specific wrapper type (`StaticBoundarySystem`, `DynamicBoundarySystem`,
   `Union{Ghost,Virtual}`) on `ps_b` instead. `device_view` erases that
   concrete identity: a device-viewed `FluidParticleSystem` and a
   device-viewed `BasicParticleSystem`/`StressParticleSystem` are the exact
   same `DeviceSystem` type, indistinguishable to the dispatcher. This means
   `ka=true` was *never* reachable for `FluidPfn` fluid-fluid — not a
   regression from this item, but a latent gap this item's tests are what
   first exercised it (nothing called `FluidPfn` fluid-fluid under `ka=true`
   before, forward or reverse). `MethodError`s loudly rather than computing
   with the wrong dispatch — verified directly and pinned down by a
   regression test (`test/test_ka_cpu.jl`, "ka=true not yet reachable (known
   gap)"). Left unfixed here on purpose: safely widening it needs
   `device_view` to preserve *which* concrete system a view came from
   (exactly what `FluidSolidPfn`'s already-documented gap needs too — see
   item 8), which is a shared prerequisite for both, not a one-off patch.
   `InterpolateFieldFn`'s `WritesB()` method onto a `VirtualParticleSystem`
   target has no such problem (item 5 gave `AbstractVirtualParticleSystem`
   its own device view) and is confirmed working under `ka=true` on real
   CUDA hardware — see `test/test_gpu_cuda.jl`.
7. **Ghosts on GPU** — the hardest remaining piece, and gates 7 of the 13
   scripts. `generate_ghosts!`'s two-pass count-then-cursor logic doesn't
   port by direct translation; needs a GPU-compatible rewrite (flag +
   exclusive-scan + compaction into capacity-preallocated buffers, since
   per-step `resize!` — while it does work on `CuVector` — isn't the right
   growth strategy for a count that changes every step). Unchanged from
   every earlier revision of this doc; still not started.
8. **Wire `onesided=true`/`ka=true` into the other 12 scripts**, one at a
   time, mirroring `dambreak.jl`'s `GRASPH_BACKEND` switch — now unblocked
   by items 5-7 plus Phase C's integration harnesses (`test/
   test_onesided_integration_*.jl`), which give a per-shape correctness
   oracle to validate each script's GPU wiring against before trusting it.
   `DambreakWall.jl` specifically also needs `FluidSolidPfn`'s two
   `pfn_contribution` methods widened off the concrete `FluidParticleSystem`/
   `ElastoPlasticParticleSystem` pair to something `device_view` can
   dispatch into (see item 5's note) before it can try `ka=true` at all —
   and the `bubble*.jl` scripts need the identical fix for `FluidPfn`'s
   fluid-fluid method (see item 6's note) for the same reason. Both need
   `device_view` to start preserving *which* concrete system a view came
   from (today it doesn't: `FluidParticleSystem`, `BasicParticleSystem`,
   `StressParticleSystem`, and `ElastoPlasticParticleSystem` all erase to the
   same `DeviceSystem` type) — worth solving once, shared by both, rather
   than as two separate patches.
9. Virtual particle systems, probes, and the RK4 integrator have no GPU
   sweep path at all yet, independent of pfn support — `VirtualParticleSystem`
   position/state advance, `_measure_probes!`, and RK4's multi-stage
   bookkeeping are all still CPU-`for`-loop or Polyester code.

## Explicitly deferred (not started, not part of the current scope)

- **GPU (`ka=true`) support for any pfn converted in Phase C** — see items
  5-9 just above; this is now the actual next-steps list, not a deferred
  afterthought, but it's still true that none of it has started.
- Ghost particles, virtual particle systems, probes, RK4 integrator,
  stress/elasto-plastic systems — none have GPU-resident sweep support yet
  (Phase C gave them CPU one-sided support and proved it correct in-context;
  GPU residency is a separate, unstarted piece — see item 7 above for why
  ghosts specifically are the hard part).
- Extending `onesided=true`/`ka=true` support to the other 12 experiment
  scripts (all still on the coloured sweep by default; see item 8 above).
- Persistent/cached grid, Verlet-skin rebuild cadence, and fusing the sort's
  gather+copyback into a single `Ref`-swap — all flagged during Phase B2 as
  follow-ups once the crossover benchmark shows whether launch count is
  actually the bottleneck at dambreak's scale; not built.
- Morton/Z-order sort keys (packed lexicographic `UInt64` key shipped
  instead).
- Explicit neighbour list (on-the-fly 27-cell scan is current approach).
- Float32/mixed precision (Float64 retained per the GPU-target decision —
  though note the hardware section above: this machine gets zero benefit
  from that decision either way).
- Multi-GPU + MPI/ORB integration.

## Practical notes for picking this back up

- Branch: `onesided-sweep-gpu-prep`, based on `main` at commit `37e9a5a`, now
  20 commits ahead of it (`12ac526`..`0f18e35`). Nothing on this branch has
  been pushed to any remote.
- Run the full suite with `julia --project -e 'using Pkg; Pkg.test()'` —
  should show `1466/1466` (834 through Phase B1, up to 935 after Phase B2's
  3D work, up to 1371 after Phase C, up to 1433 after item 5's `device_view`
  extension, up to 1466 after item 6's reverse-sweep KA kernel twin).
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
  equivalence, plus the `FluidPfn` fluid-fluid known-gap regression test) and
  `test_gpu_cuda.jl` (the same shapes against real `CUDABackend()`).
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
