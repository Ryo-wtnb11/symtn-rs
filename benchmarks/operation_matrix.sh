#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RAYON_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
: "${CARGO_TARGET_DIR:=target/operation-matrix}"
: "${OP_MATRIX_GEMM_BACKEND:=faer}"
: "${OP_MATRIX_CARGO_FEATURES:=cpu-faer,cgc-gen}"
export RAYON_NUM_THREADS OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS CARGO_TARGET_DIR OP_MATRIX_GEMM_BACKEND

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

symtn_sha="$(git rev-parse HEAD)"
symtn_dirty=false
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    symtn_dirty=true
fi
export SYMTN_AUTHORITY="$symtn_sha dirty=$symtn_dirty"

sample_dir="$(mktemp -d)"
metadata_file="$(mktemp)"
build_file="$(mktemp)"
cleanup() {
    rm -rf "$sample_dir"
    rm -f "$metadata_file" "$build_file"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -f Cargo.lock ]]; then
    printf 'Cargo.lock is required; install the reviewed benchmark lock explicitly:\n' >&2
    printf '  cp benchmarks/cpu_shape_reuse.Cargo.lock Cargo.lock\n' >&2
    exit 1
fi
lock_sha256="$(sha256_file Cargo.lock)"
cargo metadata --locked --offline --format-version 1 \
    --manifest-path symtn/Cargo.toml --no-default-features \
    --features "$OP_MATRIX_CARGO_FEATURES" >"$metadata_file"
cargo bench --no-run --profile release --locked --offline -p symtn --bench operation_matrix \
    --no-default-features --features "$OP_MATRIX_CARGO_FEATURES" \
    --message-format=json-render-diagnostics >"$build_file"
resolved_authorities="$(python3 - "$metadata_file" "$build_file" <<'PY'
import json
import pathlib
import sys

data = json.load(open(sys.argv[1]))
packages = {package["id"]: package for package in data["packages"]}
# The symbol layer was published as `racah` and is `fusion-symbols` from
# 0.3.0 on; historical benchmark locks still resolve `racah`.
SYMBOL_PACKAGES = {"racah", "fusion-symbols"}
built = {}
for line in open(sys.argv[2]):
    record = json.loads(line)
    if record.get("reason") != "compiler-artifact":
        continue
    package_id = record["package_id"]
    package = packages[package_id]
    if package["name"].startswith("tenferro-") or package["name"] in SYMBOL_PACKAGES:
        built.setdefault(package_id, set()).add(tuple(sorted(record["features"])))
if not built:
    raise SystemExit("Cargo build reported no Tenferro or symbol-layer compiler artifacts for symtn")
tenferro_records = []
racah_records = []
for package_id, feature_sets in sorted(
    built.items(), key=lambda item: (packages[item[0]]["name"], packages[item[0]]["version"])
):
    package = packages[package_id]
    source = package.get("source")
    if not source:
        raise SystemExit(f"resolved package has no source: {package['name']}")
    vcs_path = pathlib.Path(package["manifest_path"]).parent / ".cargo_vcs_info.json"
    vcs = "unavailable"
    if vcs_path.is_file():
        vcs = json.load(open(vcs_path)).get("git", {}).get("sha1")
        if not vcs:
            raise SystemExit(f"invalid VCS metadata: {vcs_path}")
    for feature_set in sorted(feature_sets):
        features = ",".join(feature_set) or "none"
        record = (
            f"{package['name']}@{package['version']} source={source} "
            f"features=[{features}] vcs={vcs}"
        )
        (racah_records if package["name"] in SYMBOL_PACKAGES else tenferro_records).append(record)
if not tenferro_records:
    raise SystemExit("Cargo build reported no Tenferro compiler artifacts for symtn")
if not racah_records:
    raise SystemExit("Cargo build reported no fusion-symbols/racah compiler artifact for symtn")
print("tenferro=" + "; ".join(tenferro_records))
print("racah=" + "; ".join(racah_records))
PY
)"
TENFERRO_AUTHORITY="$(printf '%s\n' "$resolved_authorities" | sed -n 's/^tenferro=//p')"
RACAH_AUTHORITY="$(printf '%s\n' "$resolved_authorities" | sed -n 's/^racah=//p')"
export TENFERRO_AUTHORITY

cpu_name="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
if [[ -z "$cpu_name" ]] && command -v system_profiler >/dev/null 2>&1; then
    cpu_name="$(system_profiler SPHardwareDataType | awk -F': ' '/Chip:/ { print $2; exit }')"
fi
if [[ -z "$cpu_name" && -r /proc/cpuinfo ]]; then
    cpu_name="$(awk -F: '/model name/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' /proc/cpuinfo)"
fi
: "${cpu_name:=unknown}"
printf '# host_os=%s host_release=%s host_arch=%s cpu=%s\n' \
    "$(uname -s)" "$(uname -r)" "$(uname -m)" "$cpu_name"
printf '# rustc=%s cargo=%s\n' "$(rustc --version)" "$(cargo --version)"
printf '# symtn_sha=%s dirty=%s lock=Cargo.lock lock_sha256=%s\n' \
    "$symtn_sha" "$symtn_dirty" "$lock_sha256"
printf '# cargo_package=symtn cargo_features=%s gemm_backend=%s locked=true offline=true\n' \
    "$OP_MATRIX_CARGO_FEATURES" "$OP_MATRIX_GEMM_BACKEND"
printf '# racah_authority=%s\n' "$RACAH_AUTHORITY"

for sample in 0 1 2; do
    cargo bench --profile release --locked --offline -p symtn --bench operation_matrix --quiet \
        --no-default-features --features "$OP_MATRIX_CARGO_FEATURES" >"$sample_dir/$sample"
done
if [[ "$(sha256_file Cargo.lock)" != "$lock_sha256" ]]; then
    printf 'Cargo.lock changed during the operation-matrix run\n' >&2
    exit 1
fi

# The executable reports raw one-process samples.  This wrapper keeps all raw
# rows and appends complete median-time rows, so counters always belong to the
# selected child rather than to a synthetic aggregate.
awk 'NR==FNR { if ($0 ~ /^#|^symmetry,/) print; next } { print "# raw_sample=" sample "," $0 }' \
    "$sample_dir/0" /dev/null
for sample in 0 1 2; do
    awk -v sample="$sample" '!/^#|^symmetry,/ { print "# raw_sample=" sample "," $0 }' "$sample_dir/$sample"
done
printf '# median_rows=fresh process samples; warm is per-child batch mean\n'
awk -F, '
    /^#|^symmetry,/ { next }
    { key=$1 SUBSEP $2 SUBSEP $3 SUBSEP $4; n[key]++; row[key,n[key]]=$0; t[key,n[key]]=$6 }
    END {
        for (key in n) {
            a=1; b=2; c=3;
            if (t[key,a] > t[key,b]) { x=a; a=b; b=x }
            if (t[key,b] > t[key,c]) { x=b; b=c; c=x }
            if (t[key,a] > t[key,b]) { x=a; a=b; b=x }
            print row[key,b]
        }
    }' "$sample_dir/0" "$sample_dir/1" "$sample_dir/2"
