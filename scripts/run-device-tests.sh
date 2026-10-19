#!/usr/bin/env bash
# Runs every CUDA device test of the workspace on the GPU(s) the caller selects.
# Usage: CUDA_VISIBLE_DEVICES=<ordinal> scripts/run-device-tests.sh [--two-gpu] [--skip-known-failing] [--out DIR]
# See docs/backend_policy.md ("Device test runner") for the evidence rules.
set -u
cd "$(dirname "$0")/.." || exit 1

two_gpu=0 skip_known=0 out=
while [ $# -gt 0 ]; do
  case $1 in
    --two-gpu) two_gpu=1 ;;
    --skip-known-failing) skip_known=1 ;;
    --out) shift; out=${1:?--out needs a directory} ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# The script never picks a GPU: the caller must name the free one(s).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to the free GPU (two, comma-separated, for --two-gpu)" >&2
  exit 2
fi
if [ "$two_gpu" = 1 ]; then
  case $CUDA_VISIBLE_DEVICES in *,*) ;; *) echo "--two-gpu needs two devices in CUDA_VISIBLE_DEVICES" >&2; exit 2 ;; esac
fi

# Ignored tests that are not CUDA device tests (benchmarks, measurements,
# exhaustive CPU sweeps, external tools, a recorded CPU xfail). libtest --skip
# matches substrings; each name below is a full test function name.
non_device=(
  checked_generic_public_transform_measurement
  dynamic_tree_release_measurement direct_core_release_measurement signed_core_release_measurement
  eager_c1_c2_per_member_baseline signed_eager_per_member_baseline
  su2_recoupling_matrix_transforms_scale_componentwise
  public_copy_c_batch_measurement public_host_batch_cold_warm_measurement
  public_host_swapped_core_release_measurement public_signed_direct_release_measurement
  public_dynamic_source_twist_release_measurement ignored_isolated_probe member_batch_measurement
  axioms_su2_exhaustive axioms_fz2_u1_su2_triple_exhaustive itebd_heisenberg_short_run_release
  runs_installed_cotengra_when_requested measure_checked_generic_transform_phases
)
# Need device 1 as well as device 0; run only with --two-gpu.
two_gpu_tests=(
  typed_cuda_into_rejects_a_foreign_device_destination_untouched
  a_spectrum_from_another_device_is_rejected_by_every_consumer
)
# Fail on main; run unless --skip-known-failing, so the failure stays visible.
known_failing=()

skips=("${non_device[@]}")
[ "$two_gpu" = 1 ] || skips+=("${two_gpu_tests[@]}")
[ "$skip_known" = 1 ] && skips+=(${known_failing[@]+"${known_failing[@]}"})
skip_args=()
for s in "${skips[@]}"; do skip_args+=(--skip "$s"); done

[ -n "$out" ] || out=$(mktemp -d)
mkdir -p "$out"
echo "git sha: $(git rev-parse HEAD)$(git diff --quiet HEAD -- || echo ' (dirty)')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,driver_version --format=csv
echo "logs: $out"
echo "skipped by name: ${skips[*]}"

F=(--locked --no-default-features --features "cuda,cpu-faer")
rc=0
# Whole workspace: the ignored device tests plus the non-ignored tests.
cargo test "${F[@]}" --workspace --lib --bins --tests --no-fail-fast -- \
  --include-ignored --test-threads=1 "${skip_args[@]}" >"$out/tests.log" 2>&1 || rc=1
# Doctests separately: --include-ignored would force `ignore` doc fences to run.
cargo test "${F[@]}" --workspace --doc --no-fail-fast >"$out/doc.log" 2>&1 || rc=1

grep -h '^test ' "$out/tests.log" "$out/doc.log" >"$out/test-lines.log"
for r in ok FAILED ignored; do
  echo "$r: $(grep -c "\.\.\. $r\$" "$out/test-lines.log")"
done
grep '\.\.\. FAILED$' "$out/test-lines.log"
grep -hE '^error(\[|:)' "$out/tests.log" "$out/doc.log" | head
[ "$two_gpu" = 1 ] || echo "NOT RUN: two-GPU tests (${two_gpu_tests[*]})"
[ "$skip_known" = 1 ] && echo "NOT RUN: known failing (${known_failing[*]-})"
if [ "$rc" = 0 ]; then echo "RESULT: PASS"; else echo "RESULT: FAIL (see $out)"; fi
exit $rc
