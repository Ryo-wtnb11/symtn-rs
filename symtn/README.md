# symtn

`symtn` is the public facade crate of symtn; import it as `use symtn::…`.
Ordinary applications import `Runtime`, `RuntimeBuilder`, `GradedSpace`,
`TensorMap`, `LinalgBackend`, and `Truncation` from `symtn::typed`, and the
built-in symmetries such as `U1FusionRule` from `symtn::sector`. Every public
item has exactly one path.

Start with the primitive examples below. The main
`TensorMap` operations are:

| Task | Methods |
| --- | --- |
| Construct | `from_subblock_fn`, `zeros`, `isomorphism`, `rand_with_seed` |
| Reorder and orient legs | `permute`, `braid`, `repartition`, `adjoint` |
| Combine | `compose`, `contract` with `ContractSpec` |
| Reduce and inspect | `trace_pairs`, `inner`, `norm`, `blocks`, `subblocks` |
| Factorize | `qr_compact`, `svd_compact`, `eigh_full` |

For index notation and network planning, add `symtn-network` and run its
[contract example](../symtn-network/examples/contract.rs). Mathematical conventions
are in [tensor-map mathematics](src/mathematics.md).

The default host provider is Tenferro's resolved compiled default: BLAS when
its CPU build enables `cpu-blas`, otherwise `cpu-faer`; see the
[backend policy](../docs/backend_policy.md) and [`Cargo.toml`](Cargo.toml) for
alternatives. `cuda` and `cgc-gen` add
the typed CUDA surface and `LieGroupFusionRule`, respectively. The latter uses
fusion-symbols A/B/C/D and G2 categories, their products, tori and U(n) with checked
Generic Host execution, Dynkin-label-then-charge labels and explicit `GroupId`
global forms. `SU2FusionRule`
retains its compact multiplicity-free and physical-basis capabilities.
Contraction-order planners and their configuration (`opt-path`, `cotengra-python`) belong to `symtn-network`.

The `krylov` feature makes Host `TensorMap` a vector of the
[krylov](https://docs.rs/krylov) crate (re-exported as `symtn::krylov`), so
`eigsolve`, `exponentiate` and `linsolve` run directly on tensors with
closures as linear maps (`svdsolve` takes a user type implementing
`AdjointMap`); the [krylov correspondence](../docs/krylov.md)
records the TensorKit and QSpace counterparts. krylov needs a LAPACK provider
in the final binary: enable a `blas-*` feature or link one yourself (the
default `cpu-faer` links none).

## Primitive examples

- [spaces_and_tensors](examples/spaces_and_tensors.rs): U(1) and SU(2) spaces, dual legs, construction, and blocks.
- [permute_braid_twist](examples/permute_braid_twist.rs): fermionic exchange and twist signs.
- [factorize_and_truncate](examples/factorize_and_truncate.rs): SVD reconstruction and spectrum-based bond restriction.
- [eigh_and_matrix_functions](examples/eigh_and_matrix_functions.rs): Hermitian eigendecomposition, exponential, and a spectral square root.
- [trace_inner_norm](examples/trace_inner_norm.rs): trace, inner product, and norm.
- [dense_roundtrip](examples/dense_roundtrip.rs): physical dense expansion and projection.
- [cuda_transfer](examples/cuda_transfer.rs): explicit Host/CUDA roundtrip (requires `cuda` and a device).

Run a Host example with `cargo run -p symtn --example NAME`.
