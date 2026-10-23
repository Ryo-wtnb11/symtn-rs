# Differences from TensorKit

TensorKit is the reference for mathematical conventions and public behavior.
[Source correspondences](../symtn/references.md) support individual methods;
this page records intentional differences.

## Scope

- **Tensor primitives.** Applications compose operations into DMRG, TDVP,
  PEPS and other algorithms; symtn supplies no application schedule.
- **Separate symbol authority.** fusion-symbols owns elementary categorical
  data, including product composition. symtn keeps provider types, codecs and admission.
  See [layering](layering.md).
- **Separate dense backend.** Tenferro owns dense kernels, factorizations,
  devices and execution resources; symtn owns categorical structure and
  layout decisions.

## Public API shape

- **Runtime rank.** Rank, sectors and dimensions are values of
  `TensorMap<R, D, S>`, not rank type parameters. This accommodates dynamic
  tensor networks without a second static-rank pipeline.
- **Composable operations.** Truncation is factorization, spectrum selection
  (`find_truncated`), then `restrict_leg`. SVD-based null-space selection uses
  `svd_full`, `extend_spectrum`, `Truncation::below` and `restrict_leg`.
  `left_null` / `right_null` remain shape-based.
- **Runtime ownership.** Tensors carry a `Runtime` chosen at construction.
  The application chooses dense providers and thread counts. Destination
  methods and operation-specific plan/workspace pairs expose reuse costs;
  eager methods remain primary.
- **Nested products.** Provider order and association remain in the Rust type
  and nested `ProductSector` label. TensorKit flattens product association.
- **Narrow index macro.** `contract!` wraps `Network::contract`; traces,
  hyperedges, sums and scalar factors are compile errors.

## Execution and errors

- **Explicit placement.** `to_cuda` / `to_host` move payloads. Supported device
  methods execute on device; unsupported combinations fail through bounds or
  typed errors. See [backend policy](backend_policy.md).
- **Checked failures.** Fallible methods return `Result`. `qr_compact`,
  `svd_compact` and `eigh_full` reject nonfinite inputs with an `InvalidArgument`
  operation error before factorization on supported storage.
- **Distinct `flip` legs.** Repeated legs are `InvalidArgument` before provider
  queries. `twist` applies one factor per occurrence.

## Conventions

- **Storage versus observable order.** Blocks use `SectorId` order. Truncation
  ties and identity/isometry bases use the semantic ordering described in
  [sector compatibility](sector_id_compatibility.md), including its SU(N)
  inner-line exception.
- **AnyonWiki gauge.** `TableRule` serves table values unconverted. Fibonacci
  matches TensorKit; Ising differs by the vertex gauge `u^{σσ}_ψ = −1` and
  object order (`1, ψ, σ` versus `I, σ, ψ`). TensorKit is a
  gauge-equivalence oracle for those tables.
- **Table certificates.** `TableRule` forwards fusion-symbols' braiding style
  and trivial-associator certificate. Anyonic tables require explicit braids
  or crossing-free `transpose`, `compose` and `planar_contract` where symmetric
  braiding is required; see
  [provider boundaries](provider_interface.md#current-typed-api-boundaries).
