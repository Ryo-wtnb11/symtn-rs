# Krylov methods on tensors (feature `krylov`)

With the `krylov` feature, Host `TensorMap<R, D>` (`D` = `f64`, `f32`,
`Complex64`, `Complex32`) implements `krylov::Vector`, and the krylov crate
is re-exported as `symtn::krylov`. Use that path: a second krylov version in
the dependency graph has its own `Vector` trait, which symtn does not
implement. For `eigsolve`, `exponentiate`, `linsolve` and the Arnoldi
solvers a linear map is a closure
`FnMut(&TensorMap<R, D>, &mut TensorMap<R, D>) -> Result<(), FacadeError<R>>`;
`contract_into` writes `H·x` into `y` without a temporary. `svdsolve` also
applies the adjoint, so its map is a user type implementing
`krylov::AdjointMap`.

Pinned references: TensorKit 0.17.1 (git-tree-sha1
`cd7e7b7d1137c08ba8b17e393b79fb9c858d610f`), KrylovKit 0.10.4, VectorInterface
0.6.1, QSpace qspace-v4-pub `dd2cc7e1`.

## Correspondence

| krylov | symtn | TensorKit (`src/tensors/vectorinterface.jl`, `linalg.jl`) |
|---|---|---|
| `similar` | `zeros_like` | `zerovector` (`zerovector!(similar(t))`) |
| `copy_from(x)` | `x.axpby_into(self, 1, 0)` | `scale!(ty, tx, 1)` |
| `axpby(a, x, b)` | `x.axpby_into(self, a, b)` | `add!(ty, tx, α, β)` |
| `inner(y)` | `inner` | `inner`: `Σ_c dim(c)·inner(block(tx,c), block(ty,c))` |
| `norm` | `norm(2.0)` | `norm`: `(Σ_c dim(c)·‖b_c‖²)^(1/2)` |

- `inner` and `norm` carry the quantum-dimension weight, so they are the
  Frobenius inner product and norm of the full tensor (real quantum dimension
  for complex-symbol rules, #2281). Unique-fusion rules have `dim(c) = 1`.
- `axpby` with `b = 0` is a strong zero: it does not read `self`, as
  VectorInterface's `scale!` for `β = Zero()`/`false` and krylov's contract
  require.
- All five work in place on the reduced-block payload; only `similar`
  allocates one payload, as `zerovector` does.

KrylovKit calls nothing else on a vector: by grep of `src/` in 0.10.4,
`norm`, `add!!`, `inner`, `scale!!`, `scale` and `zerovector` (and a few
`add`, `scale!`, `zerovector!!`, `add!`). Its basis operations
(`orthonormal.jl:project!!`, `unproject!!`) are one `inner` or `add!!` per
basis vector, as krylov's default `VecBasis`; the per-step work is
`O(k·N)` for Krylov dimension `k` and `N` stored entries in both.

**QSpace** (`Source/eigs2SiteQS.cc:RunEigs`) flattens the reduced blocks into
one plain vector (`map2Vec`), runs ARPACK with the plain Euclidean product
(`bmat = 'I'`), and maps back per matrix-vector product; `QSpace::scalarProd`
(`Source/QSpace.hh`) is the unweighted sum over matching blocks. TensorKit's
weighted product takes precedence. For a map that is block diagonal by
coupled sector and Hermitian on each block (every effective Hamiltonian),
both products make it self-adjoint, so the eigenvalues and `exp(tH)·x` agree;
only the Krylov iterates and error estimates differ. symtn's in-place
operations avoid QSpace's per-product flatten and unflatten copies.

## Boundaries

- Host storage only: CUDA tensors do not implement `Vector`. Every device
  `inner`/`norm` is a device-to-host sync, so a device implementation belongs
  with a fused basis projection (#1139).
- `similar` of a lazy adjoint or a compact diagonal is
  `Error::InvalidArgument`, naming `materialize()`: neither is an
  `axpby_into` destination, and densifying silently would hide a copy. Both
  are accepted as sources of `copy_from` and `axpby`.
- Operation errors (space mismatch, a destination shared with a clone, …)
  arrive as `krylov::Error::Vector`. NaN and infinite values are not errors
  in symtn's operations; krylov's own checks report them.
- krylov declares LAPACK symbols for its small dense problems. The final
  binary must link a provider: one of symtn's `blas-*` features, or one the
  application links. The default `cpu-faer` links none; a missing provider is
  a link error, never a fallback.
