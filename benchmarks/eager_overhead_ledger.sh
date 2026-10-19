#!/bin/sh
# Reproduces benchmarks/history/eager-overhead-ledger-2026-09-21.md (#1313).
# usage: benchmarks/eager_overhead_ledger.sh OUT_DIR
# Run on a quiet machine: the script refuses to start while another cargo,
# rustc, or julia process is running.
set -eu
out=${1:?usage: eager_overhead_ledger.sh OUT_DIR}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out"
busy=$(ps -Ao comm | grep -E '(^|/)(cargo|rustc|julia)$' || true)
[ -z "$busy" ] || { echo "machine busy: $busy" >&2; exit 1; }

CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
    cargo bench --no-run --profile release --manifest-path "$root/Cargo.toml" --locked --offline \
    -p symtn --bench eager_overhead_ledger --message-format=json > "$out/build.jsonl"
bin=$(python3 -c 'import json,sys; print(next(r["executable"] for r in map(json.loads, sys.stdin) if r.get("executable") and r.get("target", {}).get("name") == "eager_overhead_ledger"))' < "$out/build.jsonl")

# A second one-thread and TensorKit pass measures run-to-run spread.
for run in one default one-run2; do
    LEDGER_THREADS=${run%-run2} "$bin" > "$out/symtn-$run.csv"
done
for run in tensorkit tensorkit-run2; do
    julia -t 1 --project="$root/benchmarks/tensorkit_benchmark" \
        "$root/benchmarks/tensorkit_eager_overhead.jl" > "$out/$run.csv"
done

# Phase samples: every operation and symmetry, f64, the rank-4 and the
# smallest rank-2 case, one thread; plus default threads for the rank-4 case.
for symmetry in U1 fZ2xU1 SU2; do
    for case in r2_s8_d2 r4_s3_d4; do
        for op in compose contract permute repartition qr_compact restrict_leg scale add norm; do
            LEDGER_SAMPLE="$out/$symmetry-$case-$op.one.sample" "$bin" $symmetry f64 $case $op
            if [ $case = r4_s3_d4 ]; then
                LEDGER_THREADS=default \
                    LEDGER_SAMPLE="$out/$symmetry-$case-$op.default.sample" "$bin" $symmetry f64 $case $op
            fi
        done
    done
done
