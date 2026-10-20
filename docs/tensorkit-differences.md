# Differences from TensorKit

symtn takes TensorKit.jl as the reference for mathematical conventions and
public behavior ([design](design.md)). This page lists where symtn
intentionally differs and why. Source correspondences for individual methods
are in [provenance and references](../symtn/references.md); the supported
provider, scalar and placement combinations are in the
[current scope](../symtn/src/mathematics.md#current-scope).

## Scope

- **Only basic tensor operations.** symtn provides spaces, tensors and
  operations on them; algorithms such as DMRG, TDVP or PEPS stay in the
  application. Reason: the same small operations should be usable in code
  written by a person or a coding agent, without the library prescribing an
  MPS, PEPS, VMC or device-transfer schedule ([design](design.md)).
- **Symmetry data in a separate crate.** The symbols of the abelian groups
  (Z2, Z_N, U(1)), fermion parity, CU(1), SU(2), SU(N) and the AnyonWiki tables
  come from [fusion-symbols](https://github.com/Ryo-wtnb11/fusion-symbols)
  rather than a symtn-internal sector package. Reason: the provider owns
  categorical labels and coefficients, and two authorities for one set of
  symbols can drift ([design](design.md), [layering](layering.md), #2280).
  symtn keeps label types, `SectorId` codecs, styles and products of rules
  (`ProductFusionRule`), whose symbols it composes from the factors'.
- **Dense kernels in Tenferro.** Dense matrix kernels, factorizations and
  backend resources come from [Tenferro](https://github.com/tensor4all/tenferro-rs).
  Reason: symtn owns categorical semantics and layout decisions; Tenferro owns
  dense-provider resources ([backend policy](backend_policy.md)).

## Public API shape

- **Runtime rank.** `TensorMap<R, D, S>` carries rank, sectors and leg
  dimensions as runtime values, not as type parameters like TensorKit's
  `TensorMap{T, S, N₁, N₂}`. Reason: symtn targets dynamic-rank tensor
  networks ([backend policy](backend_policy.md)) and uses QSpace's runtime-rank
  technique ([design](design.md)); the former static-rank layer had no
  production caller and kept a second copy of the transform, add and trace
  pipelines (#2222).
- **One mathematical operation per method.** There is no `svd_trunc`,
  `eigh_trunc` or `eig_trunc`, and no `left_orth`/`right_orth` aliases.
  Truncation is `svd_compact`/`eigh_full`/`eig_full`, then
  `GradedSpace::find_truncated`, then `restrict_leg`. Reason: the public API
  offers small primitives with visible cost; each compound was exactly this
  composition with no cheaper body (#1534, #1535).
- **Rank-revealing null space is a composition.** `left_null`/`right_null` are
  shape-based, as TensorKit's default. TensorKit's
  `left_null(t; alg = :svd, trunc)` is written as `svd_full`,
  `GradedSpace::extend_spectrum`, `Truncation::below` and `restrict_leg`.
  Reason: maintainer decision to add general selection primitives instead of a
  dedicated SVD null-space API (#1984, #2110).
- **Explicit `Runtime`.** Every tensor is built against a `Runtime` that owns
  the dense provider choice, the dense thread pool and the attached device.
  Reason: thread pools and other process-wide policy are set by the
  application, not by an intermediate library ([layering](layering.md),
  [backend policy](backend_policy.md)).
- **Prepared plans and destinations.** Besides the eager methods, symtn has
  destination forms (`permute_into`, `contract_into`, …) and operation-specific
  plan/workspace pairs (`ContractPlan`, `ComposePlan`, `EighFullPlan`).
  Reason: repeated work is an explicit opt-in whose allocation is visible; the
  eager method stays the primary API ([design](design.md#reuse-and-placement)).
- **`contract!` is narrower than `@tensor`.** It is a front-end over
  `symtn_network::Network::contract`; traces, hyperedges, sums and scalar
  factors are compile errors. Reason: the macro adds no semantics beyond
  `Network` (#2029, #2271).

## Execution and errors

- **Explicit Host/CUDA placement.** Tensors move between Host and CUDA only
  through `to_cuda()` and `to_host()`. A device operation either runs on the
  device or returns an unsupported-capability error; it never runs through
  Host storage silently. Reason: the caller chooses transfers between
  operations ([design](design.md#reuse-and-placement)).
- **Nonfinite factorization input is an error.** `qr_compact`, `svd_compact`
  and `eigh_full` reject NaN or infinite input with `InvalidArgument` on every
  storage, before any numerical work. Reason: on CUDA the same input otherwise
  gave NaN factors or a solver-dependent error, so one documented contract
  must hold for every storage (#1986, #2038).
- **Errors instead of exceptions.** Fallible operations return a typed
  `Result` error and no output tensor. Reason: Rust's checked errors
  ([design](design.md)).

- **Distinct legs for `flip`.** Repeated legs return `InvalidArgument` before
  provider queries. Reason: TensorKit 0.17.1 inconsistently treats repeated
  indices in its space and fusion-tree helpers, so acceptance depends on
  multiplicity and sectors. symtn rejects all duplicates explicitly (#2290).
  `twist` still applies one factor per occurrence, matching TensorKit.

## Conventions

- **Storage order of sectors.** Blocks are stored in `SectorId` order, which
  for U(1) is not TensorKit's order; results that depend on sector order
  (cross-sector ties in truncation) use TensorKit's order. Reason: storage
  order is internal; observable order follows TensorKit
  ([sector IDs](sector_id_compatibility.md)).
- **AnyonWiki tables in their own gauge.** `TableRule` values are the AnyonWiki
  database's, unconverted. Fibonacci matches TensorKit exactly; Ising differs
  by the vertex gauge `u^{σσ}_ψ = −1` and by object order (`1, ψ, σ` against
  TensorKit's `I, σ, ψ`). Reason: maintainer decision (2026-10-17) that
  AnyonWiki is the authority for table categories; values are served
  unconverted, and TensorKit is used only as a gauge-equivalence oracle
  ([fusion-symbols#134](https://github.com/Ryo-wtnb11/fusion-symbols/issues/134),
  #2277).
- **Braided tables are `Anyonic`.** A `TableRule` with a symmetric braiding is
  still reported as `Anyonic`, so general-axes `permute`, `contract` and `tr`
  are rejected where TensorKit would accept them. Reason: the tables carry no
  symmetry flag, and deciding symmetry from rounded floats would be a tolerance
  heuristic (`symtn-sectors/src/table_rule.rs`, `TableRule`).
