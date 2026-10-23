# Structured-complexity policy

Every operation must preserve the asymptotic FLOP and working-set advantage
of each storage form it accepts. Densifying a compact operand or introducing
an avoidable intermediate that adds a size factor is a defect even when the
values are correct. Different constant factors are allowed; a different
asymptotic order is not an implementation detail.

symtn retains compact diagonal and lazy-adjoint representations behind one
`TensorMap` API. Structural routes belong at the layer that can still see the
representation. A numerically supported fallback whose structured complexity
is unproved must be documented as a gap. External implementations supply
comparison evidence, not the definition of this contract.

## Diagonal storage

Let `d` be the per-sector bond degeneracy and `n` the other operand's
open-leg size, so that operand has `d·n` entries per block.

| Geometry | Contract and current route |
| --- | --- |
| `compose`, including `U*S*Vh` | Block scaling: `O(d·n)` work, `O(d)` diagonal payload; no `O(d²)` diagonal materialization |
| Single-axis composition-equivalent `contract` / network step, in either operand order | Bond scaling followed by the required permutation: the same bounds |
| Other accepted diagonal contraction geometries | Dense fallback; general structured order remains unproved |

The other operand and output require their own `O(d·n)` storage. A diagonal
against a diagonal can multiply spectra directly in `O(d)` work/storage.
Densifying first would instead use `O(d²)` diagonal storage and
`O(d²·n)` multiplication work.

The fast path covers the documented composition-equivalent leg orientations,
not every single-axis contraction. Use `compose` where it expresses the
operation. Exact geometry bounds are in
[`TensorMap::contract`](../symtn/src/typed/contract_ops.rs); the matching
value and destination-space tests are in
[diagonal tests](../symtn/tests/typed_facade/diagonal.rs).

## Review checklist

1. State the size variables and required FLOP/working-set order for each
   accepted representation and geometry.
2. Check every fallback for densification or avoidable materialization.
3. Put the structural route at its owning layer. If the required bound cannot
   be met, record the gap and order-correct interim guidance explicitly.

Wall-clock improvements do not prove these bounds. Measure constant factors
separately under the [repository rules](../REPOSITORY_RULES.md).
