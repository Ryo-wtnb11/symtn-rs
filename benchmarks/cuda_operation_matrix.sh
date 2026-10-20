#!/usr/bin/env bash
# Revision-pinned device performance baseline of the existing CUDA path.
#
# Mirrors benchmarks/operation_matrix.sh: capture the environment that pins the
# measurement, build the release bench, run it, emit the header lines and the
# CSV on stdout. The bench owns the protocol (fresh Runtime per row, fixture
# before the timer, cold/warm-up/warm, correctness after the timers).
#
# Usage: benchmarks/cuda_operation_matrix.sh [--blocks A,B] [--degeneracy A,B]
#                                            [--iterations N] [--warmup N]
#                                            [--device N]
set -euo pipefail
cd "$(dirname "$0")/.."

RAYON_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
: "${CUDA_OP_MATRIX_CARGO_FEATURES:=cuda,cpu-faer}"
export RAYON_NUM_THREADS OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        printf 'sha256sum or shasum is required\n' >&2
        return 1
    fi
}

if [[ ! -f Cargo.lock ]]; then
    printf 'Cargo.lock is required for a revision-pinned device baseline\n' >&2
    exit 1
fi
lock_sha256="$(sha256_file Cargo.lock)"

# A measurement host usually runs from a source copy without the git metadata,
# so the pinned revision may be supplied explicitly instead.
if symtn_sha="$(git rev-parse HEAD 2>/dev/null)"; then
    symtn_dirty=false
    if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
        symtn_dirty=true
    fi
else
    symtn_sha="${SYMTN_SHA:-unknown}"
    symtn_dirty="${SYMTN_DIRTY:-unknown}"
fi
export SYMTN_AUTHORITY="$symtn_sha dirty=$symtn_dirty"

# Resolved device-stack versions, read from the pinned lock rather than from
# whatever happens to be newest on crates.io.
TENFERRO_AUTHORITY="$(python3 - Cargo.lock <<'PY'
import re
import sys

wanted = {
    "tenferro-gpu",
    "tenferro-linalg",
    "tenferro-tensor",
    "cudarc",
    "t4a-cubecl-cuda",
}
records = []
for block in open(sys.argv[1]).read().split("[[package]]"):
    name = re.search(r'^name = "(.*)"$', block, re.M)
    version = re.search(r'^version = "(.*)"$', block, re.M)
    if not name or not version or name.group(1) not in wanted:
        continue
    checksum = re.search(r'^checksum = "(.*)"$', block, re.M)
    records.append(
        f"{name.group(1)}@{version.group(1)}"
        + (f" checksum={checksum.group(1)}" if checksum else " checksum=path")
    )
if not records:
    raise SystemExit("Cargo.lock resolves no device-stack packages")
print("; ".join(sorted(records)))
PY
)"
export TENFERRO_AUTHORITY

printf '# host_os=%s host_release=%s host_arch=%s\n' "$(uname -s)" "$(uname -r)" "$(uname -m)"
if [[ -r /proc/cpuinfo ]]; then
    printf '# cpu=%s\n' \
        "$(awk -F: '/model name/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' /proc/cpuinfo)"
else
    printf '# cpu=%s\n' "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf unknown)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
        | while IFS= read -r line; do printf '# gpu=%s\n' "$line"; done
else
    printf '# gpu=nvidia-smi unavailable\n'
fi
if command -v nvcc >/dev/null 2>&1; then
    printf '# nvcc=%s\n' "$(nvcc --version | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/[[:space:]]*$//')"
else
    printf '# nvcc=unavailable\n'
fi
printf '# rustc=%s cargo=%s\n' "$(rustc --version)" "$(cargo --version)"
printf '# symtn_sha=%s dirty=%s lock=Cargo.lock lock_sha256=%s\n' \
    "$symtn_sha" "$symtn_dirty" "$lock_sha256"
printf '# cargo_package=symtn-network cargo_features=%s profile=release locked=true offline=true\n' \
    "$CUDA_OP_MATRIX_CARGO_FEATURES"
printf '# cuda_paths TENFERRO_CUTENSOR_PATH=%s CUDA_PATH=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "${TENFERRO_CUTENSOR_PATH:-unset}" "${CUDA_PATH:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
# cuTENSOR exposes its version only through cutensorGetVersion at run time, so
# the pinned version is read from the resolved soname of the library that
# TENFERRO_CUTENSOR_PATH selects.
cutensor_resolved=unresolved
if [[ -n "${TENFERRO_CUTENSOR_PATH:-}" && -e "${TENFERRO_CUTENSOR_PATH}" ]]; then
    cutensor_resolved="$(readlink -f "$TENFERRO_CUTENSOR_PATH" 2>/dev/null || printf '%s' "$TENFERRO_CUTENSOR_PATH")"
fi
printf '# cutensor_resolved=%s cutensor_version=%s\n' \
    "$cutensor_resolved" "$(basename "$cutensor_resolved" | sed -n 's/^libcutensor\.so\.//p')"
printf '# threads=RAYON_NUM_THREADS:%s OPENBLAS_NUM_THREADS:%s OMP_NUM_THREADS:%s MKL_NUM_THREADS:%s\n' \
    "$RAYON_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS"

cargo bench --no-run --profile release --locked --offline -p symtn-network --bench cuda_operation_matrix \
    --no-default-features --features "$CUDA_OP_MATRIX_CARGO_FEATURES" >&2

cargo bench --profile release --locked --offline -p symtn-network --bench cuda_operation_matrix --quiet \
    --no-default-features --features "$CUDA_OP_MATRIX_CARGO_FEATURES" -- "$@"

if [[ "$(sha256_file Cargo.lock)" != "$lock_sha256" ]]; then
    printf 'Cargo.lock changed during the device baseline run\n' >&2
    exit 1
fi
