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
- **Destination grammar.** `_into` methods take one `Dest` value
  (`overwrite`, `accumulate`, `scaled(out, α, β)`), tenferro's call grammar,
  in place of TensorKit's trailing `(tdst, …, α, β)`; `scale!` is
  `scale_in_place`.
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
- **Compiled traces are cached.** A fusion-aware trace over canonical
  structures keeps its compiled terms in the completed-transformer cache, so
  a warm trace is one lookup in every fusion style. TensorKit's
  `_trace_permute!` looks up `fsbraid` per fusion block on every call and
  recomposes Unique traces per tree (`NoCache`). Without the entry, a warm
  SU(2) trace with 979 blocks read 552 fusion groups per call (81 µs against
  10 µs), and a U(1) one with 146 blocks recomposed every block (68 µs
  against 4 µs); see
  [the benchmark record](../benchmarks/history/compiled-trace-cache-2026-10-28.md).

## Conventions

- **Storage versus observable order.** Blocks use `SectorId` order, and the
  trees inside a block follow TensorKit's `fusiontrees` nesting with external
  sectors in that order and inner lines in the provider's channel order.
  Truncation
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

### Checked Fermionic and planar contraction

On Host storage, checked Generic contraction follows TensorKit
`blas_contract!`: ordinary contraction admits symmetric braiding and applies
provider-owned θ to dual contracted legs, while `planar_contract` uses
`transpose`, `compose`, and an optional output transpose, with no θ or R
queries. Checked contraction requires `CheckedGenericPivotal<Scalar = f64>`;
`TensorScalar` payloads (f32, f64, Complex32 or Complex64) are supported. Complex categorical coefficients
remain unsupported on this checked lane, including `TableRule` products.

The checked planner replans every eager call, warm or cold. For `B_t`
tree-pair blocks and `k_d` dual contracted core-right legs, transform scales
require `B_t k_d` θ queries through the captured provider function pointer,
one vector allocation of at most `B_t` entries, an `O(B_t log B_t)` worst-case
sort, and validation over Multi-block destination layouts. The shared MF loop
uses a statically dispatched θ closure. No loop allocation or additional
payload pass is needed for the twist. Unlike TensorKit's separate `twist!`
pass over the copied operand, θ is fused into source-transform destination
scales. The uniform core case queries `Σ_c T_c k_d` twists for `T_c` RHS row
trees and folds one alpha per coupled sector into GEMM, avoiding a copy.
GEMM work stays `Σ_c m_c n_c k_c`.

The checked Fermionic identity-source shortcut currently builds a transformer
for both identity operands: warm calls look up each completed transformer;
cold calls build it in `O(blocks)`. Multiplicity increases `B_t` but leaves
per-block θ cost unchanged. No θ cache or negligible-overhead claim is made;
these costs have not been benchmarked. Planar contraction has no θ work and
uses two source transposes (each over its recoupling terms), sector GEMMs,
and an optional output transpose, as in TensorKit `planarcontract!`.
