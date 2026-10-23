# symtn

[![CI](https://github.com/Ryo-wtnb11/symtn-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ryo-wtnb11/symtn-rs/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/symtn-rs/badges/badge.json)](https://github.com/Ryo-wtnb11/symtn-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/symtn-rs/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/symtn-rs)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Symmetric tensors in Rust, inspired by TensorKit.jl.

symtn provides graded spaces, tensor maps, transformations, contraction and
factorization. Applications compose these primitives into algorithms such as
DMRG, TDVP and PEPS.

Intentional differences include runtime tensor rank, explicit Host/CUDA
transfers, destination methods, and operation-specific prepared plans. Dense
kernels and resources belong to Tenferro; reduced-block execution also draws
on QSpace. See [differences from TensorKit](docs/tensorkit-differences.md).

## Status and installation

symtn is under active development; the public API is not yet stable. Ordinary
applications use the `symtn` crate. Add
`symtn-network` for labeled tensor networks and network planning. To depend
on the repository:

```toml
[dependencies]
symtn = { git = "https://github.com/Ryo-wtnb11/symtn-rs" }
symtn-network = { git = "https://github.com/Ryo-wtnb11/symtn-rs" }
```

## Documentation

- [Crate guide and runnable examples](symtn/README.md#primitive-examples).
- [Labeled contraction example](symtn-network/examples/contract.rs).
- [Design and contracts](docs/README.md): ownership, providers, storage,
  backends and numerical testing.
- [Contributor rules](AGENTS.md).

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
