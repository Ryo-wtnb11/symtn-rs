#!/usr/bin/env bash
# Public API snapshot of every published symtn library crate.
#
#   scripts/api-snapshot.sh --write   regenerate api/*.txt
#   scripts/api-snapshot.sh --check   regenerate into a temporary directory and
#                                     diff against api/; exit 1 on any difference
#
# Needs `cargo-public-api` and the pinned nightly below (rustdoc JSON); the
# rendering of rustdoc JSON changes between nightlies, so CI installs the same
# dated toolchain (`.github/workflows/ci.yml`, `lint`). Override with
# API_SNAPSHOT_TOOLCHAIN only to try a newer nightly, then re-pin both. The
# feature combinations are fixed below and documented in api/README.md; change
# both together. A snapshot is review evidence, not a proof of behaviour.
set -euo pipefail
cd "$(dirname "$0")/.."

toolchain="${API_SNAPSHOT_TOOLCHAIN:-nightly-2026-09-19}"

mode="${1:-}"
case "$mode" in
--write | --check) ;;
*)
    echo "usage: $0 --write | --check" >&2
    exit 2
    ;;
esac

# Every workspace library crate that is published (`publish = false`
# packages are excluded). symtn-macros is a proc-macro crate, for which rustdoc
# emits no JSON; its `contract!` macro is re-exported by `symtn-network` and appears there.
# Format: <crate> <file-suffix> <features ("-" for the default set)>.
# The second row of a crate adds every additive feature that widens its public
# surface (default features stay on); `symtn/ledger-roots` (a dependency
# switch with no symtn item) is excluded.
COMBOS=(
    "symtn-sectors default -"
    "symtn-sectors all cgc-gen,testing"
    "symtn-core default -"
    "symtn-core all cgc-gen,testing"
    "symtn-dense default -"
    "symtn-dense all cuda"
    "symtn-operations default -"
    "symtn-operations all cuda,testing"
    "symtn-tensors default -"
    "symtn-tensors all cuda,cgc-gen,testing"
    "symtn-matrixalgebra default -"
    "symtn-matrixalgebra all cgc-gen"
    "symtn default -"
    "symtn all cuda,cgc-gen,krylov"
    "symtn-network default -"
    "symtn-network all cuda,cgc-gen,opt-path,cotengra-python"
)

if [ "$mode" = --write ]; then
    out_dir=api
else
    out_dir=$(mktemp -d)
    trap 'rm -rf "$out_dir"' EXIT
fi
mkdir -p "$out_dir"

for combo in "${COMBOS[@]}"; do
    read -r crate suffix features <<<"$combo"
    args=(-p "$crate" -ss)
    if [ "$features" != - ]; then
        args+=(--features "$features")
    fi
    if [ "$suffix" = default ]; then
        file="$out_dir/$crate.txt"
    else
        file="$out_dir/$crate+$suffix.txt"
    fi
    echo "api-snapshot: $crate ($suffix)" >&2
    # `#[doc(hidden)]` items are reachable public API (crate seams included),
    # so they are snapshotted too; cargo-public-api omits them unless rustdoc
    # documents them.
    RUSTUP_TOOLCHAIN="$toolchain" RUSTDOCFLAGS="-Zunstable-options --document-hidden-items" \
        cargo public-api "${args[@]}" >"$file"
done

# A type named by a public symtn signature, or by a public member of a type
# symtn re-exports, must itself be exported by symtn; otherwise a caller has to
# depend on an internal crate to name it. Ratchet: the names below already
# violate this. Remove a name once it is exported or no longer named.
UNEXPORTED_ALLOWED="
    AdjointBlockMap BlockStructure BoundDynamicFusionMapSpace
    CheckedGenericFactorPlanError CoefficientAlgebra ContractDestinationInit
    CudaDenseContext DenseBlockScalar DenseRecouplingScalar
    FactorMode FactorSpaceAuthority FusionProductSpace
    FusionTreeGroupRef FusionTreeHomSpace FusionTreePairRef FusionTreeRef
    LazyAdjointSpace PhysicalConversionError PlanarContractOrders
    RigidCoefficientAlgebra SectorSpectrum TensorTraceAxisSpec
    TensorTraceFusionStructure Token TreeTransformOperationView
    TreeTransformStructure TruncationDecision UnitLegInsertion
"
python3 - "$out_dir" "$UNEXPORTED_ALLOWED" <<'PY' || exit 1
import re, sys, pathlib
d = pathlib.Path(sys.argv[1]); allowed = set(sys.argv[2].split())
PATH = re.compile(r"\bsymtn_[a-z_]+(?:::[a-z_0-9]+)*::[A-Z][A-Za-z0-9_]*")
status = 0
for suffix in ("", "+all"):
    facade = (d / f"symtn{suffix}.txt").read_text().splitlines()
    exported = {l.rsplit("::", 1)[1] for l in facade if l.startswith("pub use ")}
    deps = [l for f in sorted(d.glob(f"symtn-*{suffix}.txt"))
            if suffix or "+" not in f.name for l in f.read_text().splitlines()]
    todo = {p for l in facade if not l.startswith("pub use ") for p in PATH.findall(l)}
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        name = path.rsplit("::", 1)[1]
        if name not in exported:
            continue
        member = re.compile(r"^pub (?:(?:const |unsafe )?fn )?symtn_[a-z_]+(?:::[a-z_0-9]+)*::" + name + r"(?:<[^:]*>)?::")
        todo |= {p for l in deps if member.match(l) for p in PATH.findall(l)}
    for name in sorted({p.rsplit("::", 1)[1] for p in seen} - exported - allowed):
        print(f"api-snapshot: symtn{suffix} names {name} in a public signature but does not export it", file=sys.stderr)
        status = 1
sys.exit(status)
PY

if [ "$mode" = --check ]; then
    status=0
    for file in "$out_dir"/*.txt; do
        name=$(basename "$file")
        if [ ! -f "api/$name" ]; then
            echo "api-snapshot: api/$name is missing (run $0 --write)" >&2
            status=1
            continue
        fi
        if ! diff -u "api/$name" "$file"; then
            status=1
        fi
    done
    for file in api/*.txt; do
        name=$(basename "$file")
        if [ ! -f "$out_dir/$name" ]; then
            echo "api-snapshot: api/$name is stale (no such crate/feature combination)" >&2
            status=1
        fi
    done
    if [ "$status" -ne 0 ]; then
        echo "api-snapshot: the public API differs from api/; review the diff and run $0 --write" >&2
        exit 1
    fi
    echo "api-snapshot: ok" >&2
fi
