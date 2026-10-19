# symtn

[![CI](https://github.com/Ryo-wtnb11/symtn-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ryo-wtnb11/symtn-rs/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/symtn-rs/badges/badge.json)](https://github.com/Ryo-wtnb11/symtn-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/symtn-rs/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/symtn-rs)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Symmetric tensors in Rust, inspired by TensorKit.jl.

symtn follows TensorKit's semantics for tensor maps on graded spaces — sectors,
fusion trees, reduced blocks and their conventions — and provides only the
basic operations on them: construct spaces and tensors, permute, braid and
transpose legs, contract and trace, factorize, and move data between Host and
CUDA storage. Algorithms such as DMRG, TDVP or PEPS are not part of symtn;
applications write them on top of these operations.

Unlike TensorKit, symtn keeps execution explicit: each public operation is one
mathematical operation, transfers between Host and CUDA are calls the
application makes, and destination (`_into`) forms expose allocation. Its
reduced-block execution also borrows techniques from QSpace.
Differences in detail are listed in [docs/](docs/tensorkit-differences.md).

## Status and installation

symtn is under active development; the public API is not yet stable. Ordinary
applications use the `symtn` crate. Add
`symtn-network` for labeled tensor networks and network planning. The crates
are not yet published on crates.io; depend on the repository:

```toml
[dependencies]
symtn = { git = "https://github.com/Ryo-wtnb11/symtn-rs" }
symtn-network = { git = "https://github.com/Ryo-wtnb11/symtn-rs" }
```

## Example

This example creates two U(1)-symmetric maps and composes them. A `GradedSpace`
describes one leg by its charge sectors and degeneracies. A `TensorMap` is a map
`codomain <- domain`; only symmetry-allowed reduced blocks are stored.

```rust
use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let runtime = Runtime::builder().build()?;
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )?;
    let a: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 1)?;
    let b: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 2)?;
    let c = a.compose(&b)?;
    println!("squared norm: {}", c.inner(&c)?);
    Ok(())
}
```

## Documentation

- [symtn crate guide](symtn/README.md): the main `TensorMap` operations and
  the [primitive examples](symtn/README.md#primitive-examples).
- [`symtn-network/examples/contract.rs`](symtn-network/examples/contract.rs):
  a labeled contraction with `contract!`
  (`cargo run -p symtn-network --example contract`).
- [Tensor-map mathematics](symtn/src/mathematics.md): conventions and current
  scope.
- [Differences from TensorKit](docs/tensorkit-differences.md).
- [Backend policy](docs/backend_policy.md): Host and CUDA capabilities and
  provider selection.
- [Documentation index](docs/README.md): design, provider interface, and
  revision-pinned evidence.
- For development: [design](docs/design.md) and
  [coding agent rules](AGENTS.md).

## Acknowledgments

symtn builds on [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) and [QSpace](https://bitbucket.org/qspace4u/). We thank their authors and contributors.

See [provenance and references](symtn/references.md) for detailed source correspondences.

## Citation

**If you use symtn in research, please cite TensorKit and QSpace.** This requests scholarly credit for the work on which
symtn builds; it does not add a condition to the software license.

- **TensorKit:** Lukas Devos and Jutho Haegeman, *TensorKit.jl: A Julia package
  for large-scale tensor computations, with a hint of category theory* (2025),
  [doi:10.48550/arXiv.2508.10076](https://doi.org/10.48550/arXiv.2508.10076).
  See TensorKit's [official citation metadata](https://github.com/QuantumKitHub/TensorKit.jl/blob/main/CITATION.cff)
  for its preferred citation and software record.
- **QSpace:** Andreas Weichselbaum, *QSpace — An open-source tensor library for
  Abelian and non-Abelian symmetries*, SciPost Physics Codebases **40** (2024),
  [doi:10.21468/SciPostPhysCodeb.40](https://doi.org/10.21468/SciPostPhysCodeb.40).
  When using QSpace directly, also cite the release used, following the
  [QSpace publication's citation guidance](https://scipost.org/SciPostPhysCodeb.40).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
