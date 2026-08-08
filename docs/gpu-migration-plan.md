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
kernel launch regardless of size. Conclusion, confirmed by later benchmarking
below: this machine is the **correctness/parity target**, not a performance
target — exactly the posture the "measure, don't assume" note above
recommended.

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
2. **The crossover benchmark** (`bench/dambreak_scaling.jl`) — scale the
   *domain*, not `dx` (smoothing length `h` is a `CubicSplineKernel` type
   parameter, so varying resolution recompiles the whole sweep per size
   point and changes `dt` too). Report CPU-coloured, CPU-onesided, *and* GPU
   per-step time — without the CPU-onesided column a GPU "speedup" can't be
   told apart from an artifact of comparing against a half-shell algorithm
   that does half the pair evaluations.
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
4. Converting the remaining pfns to `pfn_contribution`, and only then
   retiring the coloured sweep — unchanged from before, still not started.

## Explicitly deferred (not started, not part of the current scope)

- Converting the remaining pfns (`StrainRatePfn`, `StrainRateVorticityPfn`,
  `CauchyFluidPfn`, `XSPHPfn`, `InterpolateFieldFn`, `NeighborCountFn`,
  `FluidSolidPfn`) to the one-sided `pfn_contribution` protocol.
  `FluidSolidPfn` and any two-real-system coupling need the `is_mutual` trait
  design (sweep runs a second pass over `system_b`'s particles), which was
  scoped but never implemented.
- Ghost particles, virtual particle systems, probes, RK4 integrator,
  stress/elasto-plastic systems — none have one-sided pfn/sweep support yet.
  Ghosts are used by 7 of the repo's 13 experiment scripts. Ghosts need
  `generate_ghosts!`'s two-pass count-then-cursor logic rewritten for GPU
  compatibility (flag + exclusive-scan + compaction into
  capacity-preallocated buffers, since per-step `resize!`, while it does work
  on `CuVector`, isn't the right growth strategy for a count that changes
  every step).
- Extending `onesided=true`/GPU support to the other 12 experiment scripts
  (all still on the coloured sweep, unaffected by this session's work).
- Persistent/cached grid, Verlet-skin rebuild cadence, and fusing the sort's
  gather+copyback into a single `Ref`-swap — all flagged during this session
  as follow-ups once the crossover benchmark shows whether launch count is
  actually the bottleneck at dambreak's scale; not built.
- Morton/Z-order sort keys (packed lexicographic `UInt64` key shipped
  instead).
- Explicit neighbour list (on-the-fly 27-cell scan is current approach).
- Float32/mixed precision (Float64 retained per the GPU-target decision —
  though note the hardware section above: this machine gets zero benefit
  from that decision either way).
- Multi-GPU + MPI/ORB integration.

## Practical notes for picking this back up

- Branch: `onesided-sweep-gpu-prep`, based on `main` at commit `37e9a5a`.
  Nothing on this branch has been pushed to any remote.
- Run the full suite with `julia --project -e 'using Pkg; Pkg.test()'` —
  should show `834/834` (plus whatever the Tier 1/2/3 additions above bring
  it to once they land).
- `test/test_onesided_sweep.jl` and `test/test_adapt.jl` are the two test
  files from Phase A/B1; both are wired into `test/runtests.jl`. This
  session's `device_view`/KA-kernel work has been validated with ad hoc
  scripts, not yet with checked-in tests — that's next steps item 1 above.
- `CUDA.jl` is now confirmed to install and run correctly on a real GPU from
  this repo's dependency graph (see environment notes above) — the earlier
  "no GPU in the dev environment this was built in" caveat throughout Phase
  A/B1 no longer applies.
