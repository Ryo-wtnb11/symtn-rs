# Krylov methods on tensors

Feature `krylov` implements `krylov::Vector` for supported Host
`TensorMap<R, D>` with `D = f32, f64, Complex32, Complex64`. Import
`symtn::krylov` to use the exact trait version symtn implements.

The vector adapter maps `similar` to `zeros_like`, copying and `axpby` to
`axpby_into`, and scalar products/norms to `inner` / `norm(2.0)`. They use
quantum-dimension weights, matching the full tensor's Frobenius product.
`axpby` with zero destination coefficient overwrites without reading it.
Only `similar` allocates a payload.

For `eigsolve`, `exponentiate`, `linsolve` and Arnoldi solvers, a linear map
is a closure
`FnMut(&TensorMap<R, D>, &mut TensorMap<R, D>) -> Result<(), FacadeError<R>>`.
`svdsolve` also needs an adjoint and uses a caller-defined `AdjointMap`.
See the [adapter and example](../symtn/src/krylov.rs) and
[vector tests](../symtn/tests/krylov_vector.rs); solver theory belongs in the
[krylov documentation](https://docs.rs/krylov).

## Boundaries

- Host only; CUDA tensors do not implement `Vector`.
- `similar` rejects compact diagonals and lazy adjoints with
  `Unsupported { operation: "Vector::similar", alternative: Materialize }`.
  Materialize the initial vector explicitly. They remain valid copy/axpby
  sources.
- Operation errors become `krylov::Error::Vector`; krylov checks nonfinite
  values.
- The final binary must link LAPACK for krylov's small dense problems, via a
  `blas-*` feature or an application-linked provider. Default `cpu-faer`
  links none; a missing provider is a link error.
