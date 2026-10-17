# symtn

`symtn` is the public facade crate of symtn; import it as `use symtn::…`.
Ordinary applications import `Runtime`, `RuntimeBuilder`, `GradedSpace`,
`TensorMap`, `LinalgBackend`, and `Truncation` from `symtn::typed`, and the
built-in symmetries such as `U1FusionRule` from `symtn::sector`. Every public
item has exactly one path.

Start with the [crate tutorial](src/tutorial.md#quick-start). The main
`TensorMap` operations are:

| Task | Methods |
| --- | --- |
| Construct | `from_subblock_fn`, `zeros`, `isomorphism`, `rand_with_seed` |
| Reorder and orient legs | `permute`, `braid`, `repartition`, `adjoint` |
| Combine | `compose`, `contract` with `ContractSpec` |
| Reduce and inspect | `trace_pairs`, `inner`, `norm`, `blocks`, `subblocks` |
| Factorize | `qr_compact`, `svd_compact`, `eigh_full` |

For index notation and network planning, add `symtn-network` and run its
[quickstart](../symtn-network/examples/quickstart.rs). Mathematical conventions
are in [tensor-map mathematics](src/mathematics.md).

The default host provider is Tenferro's resolved compiled default: BLAS when
its CPU build enables `cpu-blas`, otherwise `cpu-faer`; see the
[backend policy](../docs/backend_policy.md) and [`Cargo.toml`](Cargo.toml) for
alternatives. `cuda` and `cgc-gen` add
the typed CUDA surface and SUN providers. Contraction-order planners and their
configuration (`opt-path`, `cotengra-python`) belong to `symtn-network`.

The `krylov` feature makes Host `TensorMap` a vector of the
[krylov](https://docs.rs/krylov) crate (re-exported as `symtn::krylov`), so
`eigsolve`, `exponentiate` and `linsolve` run directly on tensors with
closures as linear maps (`svdsolve` takes a user type implementing
`AdjointMap`); the [krylov correspondence](../docs/krylov.md)
records the TensorKit and QSpace counterparts. krylov needs a LAPACK provider
in the final binary: enable a `blas-*` feature or link one yourself (the
default `cpu-faer` links none).
