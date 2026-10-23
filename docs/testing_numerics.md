# Numerical comparisons in tests

A test asserts bit equality only when the promised result is exact:

- **Data movement:** copy, permute, scatter, restrict/embed, conjugation,
  sign application, and results left untouched after an error.
- **Determinism:** replaying the same plan or cache entry, or the same
  schedule with the same inputs, gives identical results.
- **Combinatorial output:** block and tree identity, signs, coefficient and
  truncation selection, and values that are exact by construction (for
  example small-integer or dyadic fixtures whose every partial sum is
  representable, so every summation order gives the same bits).

Any other floating-point result may be computed in more than one valid order
(GEMM, reductions, recoupling sums, factorizations, scaled accumulation). Such
a test compares against an **independent oracle** — a dense expansion, a hand
or closed-form value, a reference fixture, a reconstruction residual such as
`‖A − U S Vᴴ‖`, or orthogonality — within a dtype tolerance. When two paths
agreeing is itself the contract (for example full and compact factorizations
of the same input), the test compares them within the tolerance and says so in
its name or comment.

Comparing against the bits of a previous implementation is not a contract. It
freezes one accumulation order and is weaker evidence than an oracle.

## Tolerance

Use [`tests/support/numerics.rs`](../tests/support/numerics.rs), included with
`#[path = "<relative path>/tests/support/numerics.rs"] mod numerics;` from a
module that has `Complex32` and `Complex64` in scope. It applies

    K * sqrt(terms) * eps(dtype) * max(1, scale),    K = 32,

as implemented by the shared helper:

- `eps` is `f32::EPSILON` for `f32`/`Complex32` and `f64::EPSILON` for
  `f64`/`Complex64`; complex values are compared by the modulus of the error;
- `terms` is an upper bound on the number of floating terms that reach one
  compared entry, including every recoupled fusion-tree term, not only the
  GEMM length. For a symmetric contraction it is the contracted inner
  dimension of the coupled block (inner fusion trees times degeneracies)
  times the number of source trees recoupled into one output entry. When
  that count is impractical, `len(lhs) * len(rhs)` is always valid, because
  an entry is bilinear in the operands; a linear operation (permute, trace,
  transform) is bounded by the source length. State the bound at the call
  site;
- `scale` is the largest oracle magnitude.

Factorizations and other results whose error grows with conditioning pass an
explicit conditioning factor (`*_scaled`) and state where it comes from. Do not
loosen a bound to make a test pass; a failure under the rule is a finding.
