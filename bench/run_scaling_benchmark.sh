#!/usr/bin/env bash
# bench/run_scaling_benchmark.sh — reproduce bench/dambreak_scaling.jl on any
# CUDA-capable machine, with an optional automatic before/after comparison
# against another git ref.
#
# Usage:
#   bench/run_scaling_benchmark.sh                        # run the current checkout only
#   bench/run_scaling_benchmark.sh --before main           # A/B vs `main`
#   bench/run_scaling_benchmark.sh --before HEAD~1 --sizes 50,100,200 --budget 5e6
#
# `--before <ref>` is meant for a ref right before the change you want to
# evaluate (e.g. the commit before NeighbourListKA was added) — anything
# recognized by `git checkout` works. The current checkout is always
# restored afterward, even on failure.
#
# Why this script exists: bench/dambreak_scaling.jl needs CUDA/Adapt/
# KernelAbstractions in its environment, but the root Project.toml
# deliberately doesn't depend on them (only test/Project.toml does — see
# docs/gpu-migration-plan.md, "Environment notes for the next machine
# move"). Running the script directly needs a throwaway *merged*
# environment built via Pkg.develop + Pkg.add, exactly like Pkg.test() does
# internally; this wraps that setup so it doesn't need to be retyped by
# hand on every new machine, and reuses the environment across runs (both
# "before" and "after" resolve against the same env — only the source under
# Pkg.develop changes when the git ref switches).
#
# Safety: never discards uncommitted work. If --before is given and the
# tree is dirty, changes are `git stash`-ed before switching refs and
# restored unconditionally on exit (success or failure) via a trap.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${GRASPH_BENCH_ENV:-$REPO_ROOT/.benchenv}"
BEFORE_REF=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --before)
            BEFORE_REF="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

cd "$REPO_ROOT"

if [[ ! -f "$ENV_DIR/Project.toml" ]]; then
    echo "=== Building throwaway merged benchmark environment at $ENV_DIR ==="
    julia -e '
        using Pkg
        Pkg.activate(ARGS[1])
        Pkg.develop(path=ARGS[2])
        Pkg.add(["CUDA", "Adapt", "KernelAbstractions", "StaticArrays", "Printf", "Dates"])
    ' "$ENV_DIR" "$REPO_ROOT"
else
    echo "=== Reusing existing benchmark environment at $ENV_DIR ==="
fi

OUT_DIR="$REPO_ROOT/bench-output"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

run_scaling() {
    local logfile="$1"
    shift
    julia --project="$ENV_DIR" "$REPO_ROOT/bench/dambreak_scaling.jl" "$@" 2>&1 | tee "$logfile"
}

if [[ -z "$BEFORE_REF" ]]; then
    LOGFILE="$OUT_DIR/scaling_${STAMP}.log"
    echo "=== Running dambreak_scaling.jl — log: $LOGFILE ==="
    run_scaling "$LOGFILE" "${EXTRA_ARGS[@]}"
    exit 0
fi

ORIG_REF="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
STASHED=0

cleanup() {
    git checkout --quiet "$ORIG_REF"
    if [[ "$STASHED" -eq 1 ]]; then
        git stash pop --quiet
    fi
}
trap cleanup EXIT

if [[ -n "$(git status --porcelain)" ]]; then
    echo "=== Stashing uncommitted changes before switching to $BEFORE_REF ==="
    git stash push -u --quiet -m "run_scaling_benchmark.sh auto-stash"
    STASHED=1
fi

BEFORE_LOG="$OUT_DIR/scaling_before_${STAMP}.log"
AFTER_LOG="$OUT_DIR/scaling_after_${STAMP}.log"

echo "=== Checking out $BEFORE_REF for the 'before' run — log: $BEFORE_LOG ==="
git checkout --quiet "$BEFORE_REF"
run_scaling "$BEFORE_LOG" "${EXTRA_ARGS[@]}"

echo "=== Restoring $ORIG_REF for the 'after' run — log: $AFTER_LOG ==="
git checkout --quiet "$ORIG_REF"
if [[ "$STASHED" -eq 1 ]]; then
    git stash pop --quiet
    STASHED=0
fi
run_scaling "$AFTER_LOG" "${EXTRA_ARGS[@]}"

echo
echo "=== Done ==="
echo "before ($BEFORE_REF): $BEFORE_LOG"
echo "after  ($ORIG_REF):   $AFTER_LOG"
echo "Diff with: diff -u '$BEFORE_LOG' '$AFTER_LOG'"
