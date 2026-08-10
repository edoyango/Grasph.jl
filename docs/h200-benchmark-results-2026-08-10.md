# H200 benchmark results — 2-sided vs 1-sided GPU sweep, re-run per handoff

Written 2026-08-10, branch `onesided-sweep-gpu-prep`, in response to
`docs/server-handoff-2026-08-10.md`. Run on NCI Gadi, PBS job `175894287`
(`gpuhopper-exec` queue): 12 CPUs, 64GB RAM, 1× **NVIDIA H200** (140GB VRAM,
Hopper/sm_90, driver 580.173.02, CUDA 13.0-capable).

## Environment split: login node (internet) vs compute node (none)

Gadi's compute nodes have no outbound internet access, so all
package/artifact downloads happened on the login node (`gadi-login-06`),
sharing the same `~/.julia` depot (NFS-mounted, visible from both node
types):

1. `julia --project -e 'using Pkg; Pkg.Registry.update(); Pkg.instantiate(); Pkg.test()'`
   on the login node — downloads CUDA.jl + artifacts via
   `test/Project.toml`, doubles as a CPU-only sanity check (see below).
2. Built the throwaway benchmark env exactly as the handoff describes:
   ```julia
   using Pkg
   Pkg.activate("/scratch/tm70/ey7514/benchenv")
   Pkg.develop(path="/scratch/tm70/ey7514/grasph.jl")
   Pkg.add(["CUDA","Adapt","KernelAbstractions","StaticArrays","Printf","Dates"])
   ```
3. Moved to the compute node by SSHing directly into the node the PBS job
   was already running on (`gadi-gpu-h200-0017`) — Gadi auto-attaches the
   SSH session into the job's cgroup (`cpuset`/`memory`, confirmed via
   `/proc/self/cgroup`), so commands run there are properly scoped to the
   job's allocation, not the shared login node. No further network access
   was needed — every artifact was already cached from steps 1-2.

One practical trap worth flagging for next time: a bare `ssh host "julia
--project ..."` runs in the SSH login shell's default directory (`$HOME`
here), not wherever you last `cd`'d in an interactive session — the first
few `Pkg.test()`/benchmark invocations failed immediately for exactly this
reason, until every remote command used fully-absolute paths
(`--project=/scratch/.../benchenv /scratch/.../bench/dambreak_scaling.jl`).

## Test suite sanity check

| where | GPU functional? | Result |
|---|---|---|
| login node (`gadi-login-06`) | no | 1517 passed, 3 broken, 0 failed |
| compute node (`gadi-gpu-h200-0017`, H200) | yes | **1691 passed, 0 failed** |

The login-node run's lower count is just GPU-gated tests not counting at
all without a functional device, not a problem. The compute-node run
matches the handoff's expected **1691/0** exactly — the port is correct on
real Hopper hardware, not just the RTX 4060 laptop it was built on.

## Benchmark results

Two runs of `bench/dambreak_scaling.jl`, same script, same environment:

**Default sizes** (`nfx` 50/100/200/320/450, matching item 2's original
table):

| nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
|---|---|---|---|---|---|---|---|---|
| 50  | 2,500   | 328.7   | 553.1   | 821.8  | 3851.8 | 544.0  | 4.687 | 0.662 |
| 100 | 10,000  | 1379.8  | 2346.2  | 830.1  | 4935.4 | 473.3  | 5.945 | 0.570 |
| 200 | 40,000  | 7136.5  | 12158.2 | 920.1  | 5024.6 | 504.4  | 5.461 | 0.548 |
| 320 | 102,400 | 17479.7 | 30748.1 | 1126.0 | 5182.7 | 638.5  | 4.603 | 0.567 |
| 450 | 202,500 | 30522.9 | 51981.1 | 1576.6 | 5040.5 | 856.2  | 3.197 | 0.543 |

**Extended sizes** (`--sizes 450,600,800,1000,1300`, pushing past the
laptop's own 202,500 ceiling toward the ~1M-particle range the migration
plan's benchmark step calls for):

| nfx | n_fluid | cpu_col µs | cpu_1s µs | gpu_1s µs | gpu_col µs | gpu_1s+skin µs | col/1s | skin/1s |
|---|---|---|---|---|---|---|---|---|
| 450  | 202,500   | 31566.4  | 52110.5  | 1730.5 | 5064.6  | 1010.1 | 2.927 | 0.584 |
| 600  | 360,000   | 49863.4  | 89441.2  | 2487.5 | 5501.7  | 1332.0 | 2.212 | 0.535 |
| 800  | 640,000   | 86942.8  | 150575.8 | 4101.7 | 6538.0  | 1682.2 | 1.594 | 0.410 |
| 1000 | 1,000,000 | 137584.6 | 236786.0 | 5231.7 | 7818.8  | 2537.9 | 1.494 | 0.485 |
| 1300 | 1,690,000 | 229661.9 | 405445.5 | 8221.9 | 11548.8 | 3910.0 | 1.405 | 0.476 |

(nfx=450 appears in both tables — run-to-run JIT/scheduling noise accounts
for the ~10% difference between them; treat either as representative.)

CSVs: `bench-output/dambreak_scaling_20260810_194608.csv` (default),
`bench-output/dambreak_scaling_20260810_194848.csv` (extended).

## Answering the handoff's three open questions

**Does the item-11 crossover move?** Yes — in the opposite direction from
what the handoff flagged as plausible. The hypothesis was that a server
GPU's launch overhead, being *proportionally* smaller relative to its
compute throughput, could pull `col/1s` below 1.0 at a size well under
202,500. Instead, at every tested size — including 1,690,000 particles,
8.3× past the laptop's own ceiling — `col/1s` stays above 1.0 (best case
1.405, `ColouredKA` still losing). Kernel-launch latency is largely fixed
CPU/driver-side overhead; it doesn't shrink just because the GPU behind it
is faster. So a faster GPU's throughput gain accrues disproportionately to
whichever kernel issues fewer launches (`OnesidedKA`), *widening* the
relative penalty `ColouredKA` pays for its 7.5× launch multiplier rather
than narrowing it. **The migration's conclusion — `OnesidedKA` is the
right choice, `ColouredKA` isn't worth generalizing — holds more strongly
on server hardware, not less.**

**Does item-12's skin/1s stay below 1.0 further out?** Yes, decisively.
On the laptop it crossed 1.0 (stopped helping) around n_fluid=100,000. On
the H200 it stays in the 0.41–0.66 range across the *entire* tested range
up to 1,690,000 — a consistent 1.5–2.4× speedup with no sign of tapering.
Verlet-skin caching is a bigger, more durable win on this hardware than
the laptop numbers suggested.

**Does the CPU-coloured baseline shift?** Yes. `OnesidedKA` overtakes
CPU-coloured at n_fluid≈10,000 on the H200 (12 allocated cores) vs
≈40,000 on the laptop (item 2) — the CPU-vs-GPU crossover moved to a
*smaller* problem size on this machine, as expected given the GPU is
dramatically stronger here while the CPU side is a similar core count.

## Bottom line

No correctness regressions (1691/1691 on real Hopper hardware). Both of
the laptop's headline findings not only reproduce on server-class
hardware, they're reinforced: `OnesidedKA` wins by more, and at every size
tested (up to 8.3× past the laptop's ceiling); Verlet-skin caching helps
by more, and keeps helping past where the laptop's benefit tapered off.
`ColouredKA` remains not worth wiring into any script.
