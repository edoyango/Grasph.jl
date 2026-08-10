# Handoff: moving to a server machine for GPU benchmarking

Written 2026-08-10, branch `onesided-sweep-gpu-prep`, commit `069bcfa`
("Add ColouredKA benchmarking spike and Verlet-skin rebuild caching (items
11-12)"). Everything described below is committed — working tree was clean
at handoff time.

## Why you're moving machines

All GPU work so far (items 1-12 of `docs/gpu-migration-plan.md`) was
measured on a single laptop GPU: **RTX 4060 Laptop (8GB, sm_89)**. That
GPU has **no measured FP64 compute advantage over this machine's CPU** and a
flat **~8.3µs/kernel-launch overhead floor** (item 2's crossover benchmark) —
which is *why* items 11 and 12 exist and turned out the way they did:

- **Item 11 (`ColouredKA`)** ported the CPU half-shell colour-partitioned
  sweep (each pair visited once, ~2× less arithmetic than the one-sided
  full-neighbour sweep) to a GPU kernel-per-colour scheme, to see if halving
  compute beats paying more launches. On the laptop GPU it lost badly at
  every scale that matters (4.6× slower than one-sided at dambreak.jl's own
  production size) because launch count, not arithmetic, is what this GPU
  is paying for. It only broke even around 202,500 particles — past the
  edge of anything these scripts actually run.
- **Item 12 (Verlet-skin rebuild caching)** followed directly from that
  finding: if launches are the bottleneck, remove launches. It skips the
  per-step sort+grid-rebuild when no particle has moved far enough to
  invalidate the cell list. Genuine win at dambreak.jl's scale (1.56×), but
  it tapers to break-even past ~100k particles and never fully closes the
  gap to CPU-coloured.

**Both of those conclusions are laptop-GPU-specific.** A server GPU
(different SM count, different memory bandwidth, likely a much lower
launch-overhead floor relative to its compute throughput, and almost
certainly more VRAM headroom for the larger particle counts this laptop
couldn't reach) could plausibly flip the item-11 crossover point, change
where item-12's skin caching stops paying off, or both. That's the
open question this handoff hands you: **re-run the same two benchmarks on
server hardware and see whether the "launch-count-bound, not compute-bound"
story still holds.**

## What to actually run

`bench/dambreak_scaling.jl` already reports both comparisons in one script —
you do not need to write anything new. Its 5 timing columns are:

| column | meaning |
|---|---|
| `cpu_coloured_us` | CPU, `ColouredCPU` (today's 13-script default) |
| `cpu_onesided_us` | CPU, `OnesidedCPU` (full-neighbour, no half-shell reuse) |
| `gpu_onesided_us` | GPU, `OnesidedKA` — **1-sided**, today's production GPU path |
| `gpu_coloured_us` | GPU, `ColouredKA` — **2-sided/half-shell**, item 11's spike, kernel-per-colour |
| `gpu_onesided_skin_us` | GPU, `OnesidedKA` + `verlet_skin = 0.2 * cutoff` — item 12's rebuild caching |

The `gpu_onesided_us` vs `gpu_coloured_us` pair *is* the "2-sided vs
1-sided GPU performance" comparison you asked about. Run:

```bash
julia --project=<merged-env> bench/dambreak_scaling.jl
```

(see environment setup below for why `--project` can't just be the repo
root). Default sizes match item 2's original table (`nfx` 50/100/200/320/450
→ `n_fluid` 2,500/10,000/40,000/102,400/202,500). Pass `--sizes` for a custom
list, e.g. to push past 202,500 and see whether `ColouredKA`'s margin over
`OnesidedKA` keeps growing on server-class VRAM — that direction was
explicitly left uninvestigated on the laptop (see item 11's writeup, "not
investigated further").

It writes a CSV to `bench-output/dambreak_scaling_<timestamp>.csv` and
prints a summary table with the ratio columns already computed
(`gpu_col/gpu_1s`, `skin/1s`, etc.) — that's what you want to compare
against the laptop numbers below.

## Baseline numbers to compare against (RTX 4060 Laptop)

Item 11 (`ColouredKA` vs `OnesidedKA`, both GPU):

| nfx | n_fluid | gpu_1s µs | gpu_col µs | col/1s |
|---|---|---|---|---|
| 50  | 2,500   | 1332.1  | 6163.9  | 4.627 |
| 100 | 10,000  | 1659.2  | 10772.3 | 6.492 |
| 200 | 40,000  | 4524.5  | 11013.9 | 2.434 |
| 320 | 102,400 | 9200.6  | 11055.3 | 1.202 |
| 450 | 202,500 | 16967.1 | 11394.0 | 0.672 (only crossover point tested) |

Item 12 (`OnesidedKA` + 0.2×cutoff skin vs plain `OnesidedKA`):

| nfx | n_fluid | gpu_1s µs | gpu_1s+skin µs | skin/1s |
|---|---|---|---|---|
| 50  | 2,500   | 1385.4  | 890.4   | 0.643 |
| 100 | 10,000  | 1998.9  | 1527.6  | 0.764 |
| 200 | 40,000  | 4525.1  | 4337.8  | 0.959 |
| 320 | 102,400 | 9042.1  | 9261.2  | 1.024 |
| 450 | 202,500 | 17083.8 | 17154.6 | 1.004 |

Full writeups with the reasoning behind each (why `ColouredKA` loses, why
skin caching's benefit shrinks with scale) are in
`docs/gpu-migration-plan.md`, items 11 (line 1204) and 12 (line 1294).
Item 2 (further up the same doc) has the original CPU-vs-GPU crossover
benchmark and the ~8.3µs/launch floor measurement these two builds on.

Things worth watching for on server hardware:

- **Does the item-11 crossover move?** If the server GPU's launch overhead
  is proportionally smaller relative to its compute throughput, `col/1s`
  could drop below 1.0 at a much smaller `n_fluid` than 202,500 — which
  would mean `ColouredKA` is worth reconsidering as an actual script
  option, not just a benchmarking spike.
- **Does item-12's `skin/1s` stay below 1.0 further out?** On the laptop it
  crosses 1.0 (stops helping) around `n_fluid = 100,000`. A GPU with a
  proportionally higher fixed launch cost would push that crossover
  further right; a GPU where per-step cost is more compute-dominated would
  pull it left (or make the whole feature not worth it there).
- **CPU-coloured baseline will also shift** — `cpu_coloured_us`/
  `cpu_onesided_us` depend on server core count (`Polyester @batch`
  parallelism), so the CPU-vs-GPU crossover from item 2 needs re-measuring
  too, not just assumed constant while only the GPU columns move.

## Environment setup on the server

The root `Project.toml` deliberately does **not** depend on CUDA.jl — it's
a `test/Project.toml`-only dependency (see
`docs/gpu-migration-plan.md`'s "Environment notes for the next machine
move", line 509, for why: no `src/` code needs CUDA-specific extension
code, and pinning a CUDA version there causes a `PrettyTables` conflict, so
`test/Project.toml` leaves the CUDA version unpinned on purpose — don't
add a pin). That means:

- `Pkg.test()` resolves CUDA automatically via `test/Project.toml` — no
  extra setup needed for the test suite itself. Do this first as a sanity
  check that the port still passes on the new hardware:
  ```bash
  julia --project -e 'using Pkg; Pkg.Registry.update(); Pkg.instantiate(); Pkg.test()'
  ```
  (`Pkg.Registry.update()` first — a fresh checkout's committed
  `Manifest.toml` pins a `julia_version`/`Adapt` combination a stale
  registry cache won't satisfy, which looks like a real dependency
  conflict but isn't.) Expect **1691 passed, 0 failed** if the port and
  hardware are both healthy — that's the current count after items 11-12.
  It'll run long (~5+ minutes); background it if running interactively.

- **`bench/dambreak_scaling.jl` is a plain script, not a test**, so it
  needs its own environment with `Grasph` (this checkout) + `CUDA` both
  available as regular deps, which the root `Project.toml` doesn't give
  you. Build a throwaway merged environment once:
  ```julia
  using Pkg
  Pkg.activate("/path/to/some/scratch/dir/benchenv")
  Pkg.develop(path="/path/to/Grasph.jl")   # this checkout, dev-mode
  Pkg.add(["CUDA", "Adapt", "KernelAbstractions", "StaticArrays", "Printf", "Dates"])
  ```
  then run the benchmark against it:
  ```bash
  julia --project=/path/to/some/scratch/dir/benchenv bench/dambreak_scaling.jl
  ```

- Confirm the GPU is visible first (`nvidia-smi`) and check
  `CUDA.functional()` inside that merged env before running anything long —
  the benchmark script itself checks `HAVE_CUDA` and will silently print
  `NaN` in the GPU columns instead of erroring if CUDA isn't functional,
  which is easy to miss in a long scrollback.

## If you want to poke at `ColouredKA` directly

It's not wired into any script (deliberate — see item 11's "Not wired into
any script" note). To exercise it outside the benchmark script, pass the
internal `mode` override to `SystemInteraction` instead of `onesided`/`ka`:

```julia
SystemInteraction(kernel, pfn, system_a, system_b; mode = Grasph.ColouredKA())
```

2D only, `FluidPfn`-self / `StaticBoundarySystem`-coupled only — no ghosts,
virtual particles, probes, 3D, or `RK4TimeIntegrator` support was built for
it (see item 11's final paragraph for the full list of what's intentionally
missing). If server numbers make it look worth generalizing, that
generalization work hasn't started.

## Everything else that's relevant

`docs/gpu-migration-plan.md` is the living doc for this whole migration —
items 1-12 done, remaining backlog under "Explicitly deferred" (line 1441):
Morton/Z-order sort keys, an explicit neighbour list, Float32/mixed
precision, multi-GPU/MPI/ORB. None of those were touched this session and
none depend on what you're about to measure — they're independent future
work, not blocked on this handoff.
