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

# A type named by a public symtn signature, or by the declaration or a public
# member of a type symtn re-exports, must itself be exported by symtn;
# otherwise a caller has to depend on an internal crate to name it. Ratchet:
# the names below were already unnameable at d72c610c (the check's base) and
# still are. Remove a name once it is exported or no longer named.
UNEXPORTED_ALLOWED="
    AdjointBlockMap BlockStructure BoundDynamicFusionMapSpace
    CheckedGenericFactorPlanError CoefficientAlgebra
    ContractDestinationInit CudaScalarOperands DenseBlockScalar
    DenseRecouplingScalar FactorMode FactorOutput FactorSpaceAuthority
    FusionProductSpace FusionTreeGroupRef FusionTreeHomSpace
    FusionTreePairRef FusionTreeRef LazyAdjointSpace
    PhysicalConversionError PlanarContractOrders Provider
    RigidCoefficientAlgebra SectorSpectrum TensorTraceAxisSpec
    TensorTraceFusionStructure TreeTransformOperationView
    TreeTransformStructure TruncationDecision UnitLegInsertion ZeroBytes
"
python3 - "$out_dir" "$UNEXPORTED_ALLOWED" <<'PY' || exit 1
import re, sys, pathlib
d = pathlib.Path(sys.argv[1]); allowed = set(sys.argv[2].split())
# The first capitalised segment of an internal-crate path is the type it names.
# Sealing items are unnameable on purpose, but sealing alone does not stop
# calls, so each one is audited and listed by full path: a marker trait with no
# methods, an unconstructible token, or a trait whose every method takes one.
SEALED = {
    "symtn_dense::cuda_adapter::cuda_scalar_sealed::Sealed",  # no methods
    "symtn_sectors::product_rule::sealed::Token",  # unnameable, so unconstructible
    "symtn_matrixalgebra::factorize::scalar::sealed::DenseIo",  # token-gated
}
PATH = re.compile(r"\bsymtn_[a-z_]+(?:::[a-z_0-9]+)*::([A-Z][A-Za-z0-9_]*)")
ATTRS = re.compile(r"^(?:#\[[^\]]*\] )*")
def names(line):
    return {m.group(1) for m in PATH.finditer(line) if m.group(0) not in SEALED}
def unexported(facade, deps):
    exported = {l.rsplit("::", 1)[1] for l in facade if l.startswith("pub use ")}
    deps = [ATTRS.sub("", l) for l in deps]
    # Roots: every public symtn signature and every re-exported type.
    todo = set(exported).union(*(names(l) for l in facade if not l.startswith("pub use ")))
    seen = set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        if name not in exported:
            continue
        # The declaration (tuple fields and supertraits included) and every member.
        item = re.compile(r"^pub (?:[a-z]+ )*symtn_[a-z_]+(?:::[a-z_0-9]+)*::" + name + r"\b")
        todo |= set().union(*(names(l) for l in deps if item.match(l)))
    return seen - exported

# Self-test: a tuple field of a re-exported type, a type reached only through
# a re-export, and a sealing trait that has not been audited.
assert unexported(
    ["pub use symtn::expert::Storage"],
    ["pub struct symtn_ops::cuda::Storage<D>(pub symtn_dense::buf::Buffer, _)"],
) == {"Buffer"}
assert unexported(
    ["pub use symtn::sector::Rule"],
    ["pub fn symtn_sectors::Rule::by_uuid(&str) -> core::result::Result<Self, symtn_sectors::RuleError>",
     "pub trait symtn_sectors::Rule: symtn_sectors::seal::Sealed"],
) == {"RuleError", "Sealed"}

status = 0
for suffix in ("", "+all"):
    facade = (d / f"symtn{suffix}.txt").read_text().splitlines()
    deps = [l for f in sorted(d.glob(f"symtn-*{suffix}.txt"))
            if suffix or "+" not in f.name for l in f.read_text().splitlines()]
    for name in sorted(unexported(facade, deps) - allowed):
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
