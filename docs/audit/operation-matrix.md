# Operation matrix

Audited at: `eb99cc405bc57c24c9755d4a9c30b2fcc5aeec2b`

> Historical evidence only; not current capability authority. Later outcomes
> are listed in the [audit index](README.md).

This is a historical capability inventory for #938 at the revision above. It
must not be read as current-main evidence without being rerun. The pinned
source and executable tests are authoritative for that revision; an export, a
satisfiable trait bound, or an old benchmark is not proof.

## Status

- **PROVED**: the canonical public path has a current executable semantic test.
- **MEASURED-GAP**: the path works and a current measurement proves a remaining
  production cost.
- **UNSUPPORTED**: the public capability boundary does not expose the path.
- **INTENTIONAL-DIFFERENCE**: the path is supported with a deliberately
  different public or numerical contract.
- **NEEDS-PROOF**: the path is reachable, but the exact provider/operation cell
  lacks an independent public-API oracle.

`PROVED` is deliberately fixture-scoped. It does not claim exhaustive ranks,
sector distributions, products, devices, or checked providers.

## Provider families on default Host storage

All cells refer to `TensorMap<R, D, Vec<D>>`. The ZN column is the built-in
`ZNFusionRule::new(3)` fixture. The two product columns name the exact fixtures
exercised by the suite; none of these columns proves arbitrary moduli, products,
or recursively nested `ProductFusionRule` values.

| Operation family | U(1) | Z2 | ZN(3) | CU(1) | fZ2 | SU(2) | fZ2 x U(1) | (fZ2 x U(1)) x SU(2) | checked Generic seam [1] | Fibonacci |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Space/tensor construction and labelled block readback | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED |
| Physical dense expansion and symmetric projection [2] | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| `adjoint` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `permute`, `transpose`, `repartition` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED |
| `braid`, `twist` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `flip` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Unit insertion/removal | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `catdomain`, `catcodomain`, `absorb` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `otimes`, arbitrary/ordered `contract`, `compose` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED |
| `add`, `scale`, `norm`, `trace_pairs`, `tr` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `inner` [3] | [MEASURED-GAP](https://github.com/Ryo-wtnb11/TeNeT/issues/875) | PROVED | PROVED | PROVED | PROVED | [MEASURED-GAP](https://github.com/Ryo-wtnb11/TeNeT/issues/875) | PROVED | PROVED | PROVED | UNSUPPORTED |
| Compact QR; `left_orth`, `right_orth`; SVD values | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Compact/full/truncated SVD; full QR; compact/full LQ | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Left/right numerical null spaces [4] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | INTENTIONAL-DIFFERENCE | UNSUPPORTED |
| Left/right polar factors | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| EIGH full/truncated/values | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| EIG full/truncated/values | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `inv`, `exp`, `solve` | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| `pinv` [5] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | INTENTIONAL-DIFFERENCE | UNSUPPORTED |
| Compact `map_diagonal` | PROVED | PROVED | NEEDS-PROOF | NEEDS-PROOF | NEEDS-PROOF | PROVED | NEEDS-PROOF | NEEDS-PROOF | PROVED | UNSUPPORTED |
| Network ordinary planning, contraction/permute replay [6] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Network intra-operand trace [6] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |
| Network payload-destination reuse [6] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | [INTENTIONAL-DIFFERENCE](https://github.com/Ryo-wtnb11/TeNeT/issues/1005) | UNSUPPORTED |
| v1 Host snapshot (`f64`/`Complex64`; admitted dense, compact, and lazy forms) [7] | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | UNSUPPORTED |

Notes:

1. “Checked Generic seam” means provider-neutral dispatch proved with synthetic
   failure-injection providers and exact SU(3) `[1,1]` / SU(4) `[1,0,1]`
   fixtures, for `f64` and `Complex64` where applicable. SU(N) is evidence, not
   a named-group implementation branch. B/C/D provider work is excluded and
   remains in [#989](https://github.com/Ryo-wtnb11/TeNeT/issues/989).
2. Host physical expansion/projection is proved for built-in U(1) and SU(2) multiplicity-free providers by `tenet/tests/physical_dense.rs`, including real/complex SU(2) round trips, an independent TensorKit SU(2) oracle, and U(1) basis-order projection. Checked Generic, products, devices, and Python export remain unsupported; the Python/JAX surface is deferred in [#1086](https://github.com/Ryo-wtnb11/TeNeT/issues/1086).
3. Only the U(1)/SU(2) Host `inner` production gap has a current measurement;
   [#875](https://github.com/Ryo-wtnb11/TeNeT/issues/875) owns it. The
   values-only risk in [#880](https://github.com/Ryo-wtnb11/TeNeT/issues/880)
   is not evidence for another `MEASURED-GAP` cell.
4. TeNeT publishes numerical null spaces using its existing numerical-rank
   policy. This is not a claim of a canonical basis or a TensorKit/QSpace gauge.
5. TeNeT `pinv` is a tolerance-defined numerical pseudoinverse using one strict
   global `rcond * sigma_max`; TensorKit applies its tolerance block-locally.
   The Moore-Penrose identities for the original input hold when the cutoff
   removes only numerical-null directions. Discarding an arbitrary nonzero mode
   instead gives the pseudoinverse of the thresholded effective-rank tensor.
6. For an expression with a repeated label, the macro calls the ordinary
   checked trace during call-local lowering, before it looks up the reduced
   contraction/permutation plan. The trace is recomputed on each call; only the
   reduced plan and its workspace are reused. The macro asks for pivotal
   provider data only for these trace-bearing expressions. Host workspaces in
   both modes reuse compatible intermediate payload buffers through
   `contract_into`, and park them in the plan-cache pool between calls. The
   final
   tensor leaves the workspace in both modes, and neither mode accepts a
   caller-owned payload destination. That public destination boundary remains
   an [intentional difference](https://github.com/Ryo-wtnb11/TeNeT/issues/1005)
   from TensorKit's mutating `@tensor C[...] = ...` form.
   Lazy/conjugated checked inputs reject at the same ordinary-operation seam and
   are not silently rerouted through a weaker fallback.
7. Persistence is provider-neutral; reconstruction dispatches through the
   provider's admission mode. For this row, the MF cells combine the U(1)/SU(2)
   persistence fixtures with the existing MF provider-admission proofs. They do
   not claim that TeNeT ships or tests an application codec for every provider
   column. The caller supplies a stable provider key, a resolver returning the
   exact provider `Arc`, and semantic sector-label encoding through
   `TypedPersistenceCodec`; TeNeT does not ship a fixed provider registry. Tests
   also cover checked-Generic vertex multiplicity, all four scalar types, and every
   admitted representation kind. Compact adjoints normalize to owned compact
   tensors in both modes; a legacy checked-Generic lazy-adjoint-over-compact
   record decodes to that owned compact tensor (#1449).
   Direct device snapshots and storage types other than Host `Vec<D>` are
   unsupported; use explicit `to_host()` before encoding a device tensor.

Fibonacci has partial canonical typed Host coverage: construction/readback,
the tested complex transforms, tensor products, and composition are admitted.
Full snapshot, decomposition, network, planar, and device coverage remains
unsupported unless a row above has an executable fixture. Slicing is separately
tracked in [#6](https://github.com/Ryo-wtnb11/TeNeT/issues/6). No public
trivial/dense provider exists.

## Storage and device matrix

| Capability | Host MF `Vec<D>` | Host checked Generic `Vec<D>` | other Host-readable `S` | CUDA f64 MF | CUDA f64 checked Generic | CUDA c64 | CUDA f32/c32 MF [10] |
|---|---|---|---|---|---|---|
| Metadata, provider ownership, handle clone | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED | PROVED |
| Borrowed `dense_data() -> Result<&[D], Error>` | PROVED | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Physical expansion/projection [2] | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Explicit Host/device transfer | PROVED | PROVED | [NEEDS-PROOF](https://github.com/Ryo-wtnb11/TeNeT/issues/3) | PROVED | PROVED | PROVED | PROVED |
| Lazy adjoint | PROVED | PROVED | [NEEDS-PROOF](https://github.com/Ryo-wtnb11/TeNeT/issues/3) | PROVED | UNSUPPORTED | PROVED | PROVED |
| Permute/braid/recoupling | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | PROVED |
| `twist`/`twist_inverse` [11] | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | [NEEDS-PROOF](https://github.com/Ryo-wtnb11/TeNeT/issues/1336) |
| Canonical contraction/compose | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | PROVED |
| General-axes `contract` [12] | PROVED | [NEEDS-PROOF](https://github.com/Ryo-wtnb11/TeNeT/issues/3) | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | PROVED |
| Arithmetic/reductions | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | PROVED [10] |
| `trace_pairs` [13] | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | PROVED |
| SVD/EIGH [9] | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED | PROVED [11] |
| QR | PROVED | PROVED | UNSUPPORTED | PROVED | UNSUPPORTED | PROVED [11] | PROVED [11] |
| EIG/null/polar/solve/matrix functions | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |
| Network ordinary replay [8] | PROVED | PROVED | INTENTIONAL-DIFFERENCE | PROVED | UNSUPPORTED | PROVED | PROVED |
| v1 typed snapshot (`f64`/`Complex64`) [7] | PROVED | PROVED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED | UNSUPPORTED |

[12] Device `contract`/`contract_ordered` with arbitrary contracted and output
axes replay the Host `DynamicTree` artifact — source tree transforms, the
fully-direct core GEMMs, the output transform — with the device executors
(G2c-1a, [#1345](https://github.com/Ryo-wtnb11/TeNeT/issues/1345)); the
TensorKit `mul!` form keeps the storage-direct core route. Gated by
`tenet/tests/typed_cuda_contract.rs` (device == Host at every device dtype,
owned and lazy-adjoint, and device == TensorKit's `blas_contract!` sequence and
the physical-basis contraction, both oracles pinned against the Host by the
ungated `typed_contract_host_oracle.rs`) and by the forced-orientation replays
of `tenet-tensors/src/contract/storage_contract_tests.rs`. The fermionic
supertrace twist of the core-right operand is folded into that operand's
source transform as per-destination-block descriptor scales (G2c-2,
[#1347](https://github.com/Ryo-wtnb11/TeNeT/issues/1347)), including the
canonical form whose twist varies within one coupled sector, which now leaves
the storage-direct core for `DynamicTree` on device; gated against TensorKit's
`blas_contract!` with the twist on the B role and on the A role (fZ2 x U(1),
fZ2 (x) SU(2), all four device dtypes, lazy adjoints, both orientations forced
at artifact level) and by the TensorKit-valued FZ2 loops as explicit device
`contract` calls. Device `contract_into` takes the same
resolution for arbitrary axes and twists and computes
`dst = alpha * contract + beta * dst` with no reset: `alpha` and `beta` ride the
core GEMMs' epilogue or the output transform's `Axpby(beta)` mode, and exactly
the plan's inactive blocks become `beta * dst` (G2c-1b,
[#1346](https://github.com/Ryo-wtnb11/TeNeT/issues/1346);
[#1550](https://github.com/Ryo-wtnb11/TeNeT/issues/1550) removed the
`alpha = 1` restriction and added `beta`); gated into NaN-poisoned
destinations against the Host and the same oracles, with a warm call
transferring and allocating nothing. Device `tensor!` networks run every
compiled schedule — general steps, result and final permutations, open
outputs, dual contracted legs — through these device operations
([#1348](https://github.com/Ryo-wtnb11/TeNeT/issues/1348)), intra-operand
traces included (see [13]). Behaviour change: an anyonic device
`contract` is now `UnsupportedTensorContractScope` even in canonical form,
as on Host (it was accepted before). The Host checked-Generic cell is
`NEEDS-PROOF` for a named gap: `tensorcontract_owned_checked_generic_in_context`
accepts arbitrary axes for owned operands (lazy adjoints are
`InvalidArgument`), but no fixture gates a non-canonical checked-Generic
contraction against an independent oracle. Host and device
`contract_into` reject anyonic providers with `contract`'s error
right after the Runtime check, and the Host compact-spectrum `trace_pairs` arm
rejects non-symmetric braiding with the dense trace's error, as TensorKit
`blas_contract!` / `trace_permute!` do
([#1355](https://github.com/Ryo-wtnb11/TeNeT/issues/1355)).
Since [#1372](https://github.com/Ryo-wtnb11/TeNeT/issues/1372) every ordinary
contraction entry — Host and device `contract`, `contract_ordered`,
`contract_into` (formerly `contract_overwrite_into` and its
`contract_ordered_overwrite_into` alias), the checked-Generic `contract`, and Host and device `tensor!`
contraction steps — admits only symmetric braiding (Bosonic, Fermionic) and
rejects `Anyonic` and `NoBraiding` with one shared
`UnsupportedTensorContractScope` error (`NON_SYMMETRIC_CONTRACTION_UNSUPPORTED`),
canonical axes included, as TensorKit 0.17.1 `blas_contract!` does before any
layout test. Behaviour change: a canonical `NoBraiding` contraction was
accepted before. `compose` (TensorKit `mul!`) does not route through this
boundary; planar contraction for every braiding style is the separate
`planar_contract` / `planar_contract_into` (TensorKit `@planar` /
`planarcontract!`, #2284), multiplicity-free Host only. The checked-Generic
engine itself still restricts contraction and `compose` to Bosonic braiding.

[13] Device `trace_pairs` replays the Host trace structure — its valid
tree-pair terms, their coefficients (recoupling row, `dim(c)/dim(a_1)`, the
fermionic supertrace twist) and every stride — as one contraction per term:
the source block read through one merged diagonal axis per traced pair
(extent `t_k`, stride `s_lhs + s_rhs`; pairs never merged with each other)
against the context-owned ones template, accumulated with `beta = 1` into the
zero-initialised output, so repeated destinations sum (G2c-4,
[#1349](https://github.com/Ryo-wtnb11/TeNeT/issues/1349)). A warm call
transfers only the #740 output initialisation and, while its distinct term
signatures fit the transform executor's plan-cache budget (the bound is
raised by that count under the same rule), misses no cuTENSOR plan.
Gated by `tenet/tests/typed_cuda_trace.rs` (device == Host at every device
dtype, owned and lazy-adjoint, U(1)/SU(2)/U(1)xSU(2)/fZ2xU(1)/fZ2(x)SU(2);
device == the physical-basis diagonal sum and the identity contraction, both
pinned against the Host by the ungated `typed_trace_host_oracle.rs`; device ==
the hand-valued fZ2 supertrace) and by the unmerged-index oracle of
`tenet-dense/tests/cuda_region_trace.rs`. The `tensor!` trace pre-step runs
on device with this `trace_pairs` (G2c-5,
[#1350](https://github.com/Ryo-wtnb11/TeNeT/issues/1350)): every trace of the
expression is validated and compiled on the Host before the first executes,
so a trace rejection on any operand leaves the device untouched; a
contracted-leg mismatch between the reduced operands is decided from their
spaces before any trace runs (#1371); the reduced network then takes the
device network path. Each traced operand costs its own
#740 output upload on every call (trace outputs are call-local, as on Host,
not retained workspace). Gated by the trace gates of
`tenet-network/tests/typed_cuda_network.rs` (device == Host `tensor!`, and the
physical-basis einsum for U(1)/SU(2)).

[10] The `CUDA f32/c32 MF` column is the single-precision device payload of
[#1336](https://github.com/Ryo-wtnb11/TeNeT/issues/1336) (leaf C2), gated by
`tenet/tests/typed_cuda_single_precision.rs` and the two `single_precision_*`
gates of `tenet-network/tests/typed_cuda_network.rs`. Every cell is one generic
body instantiated for all four device dtypes against the host result *of the
same dtype*.

The reductions accumulate in `f64` like the host at every payload dtype:
Tenferro 0.7.1 exposes no widening dot or reduction, so a single-precision
payload is widened on the device by one Tenferro `cast` per distinct operand
before the per-sector reduction, and the cross-sector combine runs in
`WideScalar::Wide`; this is documented on `weighted_inner_cuda`. `norm` (and
therefore `normalize`) has done so since
[#1344](https://github.com/Ryo-wtnb11/TeNeT/issues/1344), pinned by
`device_norm_and_normalize_match_the_host_where_a_payload_sum_would_overflow_or_underflow`;
`inner`/`dot` since [#1383](https://github.com/Ryo-wtnb11/TeNeT/issues/1383),
pinned by `device_inner_matches_the_host_where_single_precision_products_overflow`
(a cancelling sum whose products overflow `f32`). Cost: one extra device
allocation per distinct operand, pinned by
`a_warm_single_precision_reduction_costs_one_extra_device_allocation_per_operand`.

[11] The device factorizations are open for single precision since leaf C4
([#1341](https://github.com/Ryo-wtnb11/TeNeT/issues/1341)): `svd_compact` and
`eigh_full` hang off the sealed marker `CudaFactorizationPayload`, now
implemented for all four device payloads. `qr_compact` joined them in
[#1271](https://github.com/Ryo-wtnb11/TeNeT/issues/1271): the backend's
positive-diagonal gauge runs a `triu` kernel whose complex zero constant
Tenferro 0.5.0 could not compile (tenferro-rs#1833); 0.6.0 compiles it, and
the complex device QR is verified against Host `qr_compact` in
`tenet/tests/typed_cuda_single_precision_factorizations.rs` (A100 record
`benchmarks/history/cuda-complex-qr-2026-09-21.md`).

Single-precision device factorization evidence:
`tenet/tests/typed_cuda_single_precision_factorizations.rs` and
`benchmarks/history/cuda-typed-single-precision-factorizations-2026-09-21.md`.
The oracle is the host factorization of the same tensor at the same payload
dtype, read through gauge-independent identities (reconstruction, isometry, the
spectrum, the positive-diagonal gauge) — the device SVD and `eigh` keep the raw
cuSOLVER gauge and their factor payloads are never compared pointwise.

[11] `flip` stays `UnsupportedOnDevice`. The device `twist` is not a tree
transform: the per-block ribbon-twist factor is built on the Host from the
block structure and rides the contraction descriptor's own scale, so no factor
table is uploaded. That is sound only because the factor is a real sign for
every provider the device impl admits (`Scalar = f64`) and never zero; the
value domain is pinned without a device in `typed_transform_host_side.rs`. A
compact (diagonal) device payload — the Host's compact-spectrum arm — is an
explicit `UnsupportedOnDevice` boundary, and flat elements belonging to no
block are zero where the Host copies them through, the convention device
structural results have carried since
[#1322](https://github.com/Ryo-wtnb11/TeNeT/issues/1322). NaN and real
infinities propagate as on Host; an infinite *complex* entry becomes NaN in
both components, because the device always multiplies where Host bit-copies a
factor-1 block (the #1301 deviation). The body is one generic over
`CudaPayload`, so it is instantiated for the single-precision payloads too and
`±1` is exact there, but no single-precision twist fixture exists yet — hence
`NEEDS-PROOF` rather than `PROVED` in that column.

[9] The CUDA cells cover the *compact/full* factorizations (`svd_compact`,
`eigh_full`, and `qr_compact` in its own row). The truncated variants
`svd_trunc`/`eigh_trunc` are an explicit `UnsupportedOnDevice` boundary
([#1297](https://github.com/Ryo-wtnb11/TeNeT/issues/1297)); the device result
is composed on the host as described below the table.

[8] Device network replay leases a workspace from the same per-plan pool,
quarantine and byte-budget machinery as Host, keyed by `(provider, dtype,
storage)`, and reuses slots, producers, the input snapshot and the payload
destinations of every intermediate step through the device
`contract_into` and `permute_into` with `beta = 0`. Since
[#1348](https://github.com/Ryo-wtnb11/TeNeT/issues/1348) the schedule is not
restricted: device admission (placement, compact operands, non-symmetric braiding)
is decided from the operand count and metadata before the plan-cache lookup
or a workspace lease. Since
[#1371](https://github.com/Ryo-wtnb11/TeNeT/issues/1371) the same holds for
the storage-generic metadata rejections of a Host or device `tensor!` network
(non-symmetric braiding that contracts, then a contracted-leg space or duality
mismatch after any trace pre-step): a rejection counts no hit or miss,
installs no static alias and leases no workspace, on the miss, topology-hit
and alias-hit paths
(`host_metadata_rejections_leave_the_plan_cache_untouched_on_every_path` and
`device_metadata_rejections_leave_the_plan_cache_and_device_untouched_on_every_path`
in `tenet-network/src/plancache.rs`). The preflight a warm hit runs allocates
nothing (`the_warm_preflight_allocates_nothing` in
`tenet-network/tests/preflight_allocations.rs`). Device execution is gated in
`tenet-network/tests/typed_cuda_network.rs` against the Host `tensor!` run
(U(1), SU(2), U(1)×SU(2), fZ2×U(1), fZ2⊠SU(2)) and the physical-basis dense
expansion (U(1), SU(2)); a warm general network transfers and allocates only
its returned output and misses no cuTENSOR plan. The final schedule slot leaves the workspace and so
still allocates and uploads a fresh returning output
([#740](https://github.com/Ryo-wtnb11/TeNeT/issues/740)). A reused device
destination is not reset: the contraction zeroes only the blocks no GEMM
writes, from the context's zero source
([#1346](https://github.com/Ryo-wtnb11/TeNeT/issues/1346)).

The two storage `NEEDS-PROOF` cells describe future storage implementations,
not provider conformance. Host/device checked-Generic parity belongs to
[#3](https://github.com/Ryo-wtnb11/TeNeT/issues/3). CUDA `PROVED` means a
real-device test exists and is ignored without CUDA; default CI is not claimed
to run it. The CUDA c64 column is the `Complex64` device payload of
[#1268](https://github.com/Ryo-wtnb11/TeNeT/issues/1268). `svd_compact` and
`eigh_full` carry both device payloads: EIGH admits a block only when it equals
its conjugate transpose, and `u`/`vh` follow the Host largest-pivot gauge,
applied on the device ([#1552](https://github.com/Ryo-wtnb11/TeNeT/issues/1552)). `svd_trunc` and `eigh_trunc`
are deliberately *not* device operations
([#1297](https://github.com/Ryo-wtnb11/TeNeT/issues/1297)): they return
`UnsupportedOnDevice` before any device work, because the truncation decision
is global over quantum-dimension-weighted spectra and belongs on the host. The
device result is composed from `svd_compact`/`eigh_full`, `to_host`,
`diagview`, `GradedSpace::find_truncated` and
`restrict_leg`.
`qr_compact` carries both payloads in the Host positive-diagonal gauge
(`R_jj` real and non-negative, phase 1 kept at zero;
[#1271](https://github.com/Ryo-wtnb11/TeNeT/issues/1271)).
Full/values factorizations, `eig`, and matrix functions stay
device-unsupported for both payloads.
Release/feature topology remains [#129](https://github.com/Ryo-wtnb11/TeNeT/issues/129).

Standalone checked-Generic compact construction is `PROVED` by
[#1004](https://github.com/Ryo-wtnb11/TeNeT/issues/1004). Compact diagonal
factors produced by EIGH/EIG remain `PROVED`; checked SVD currently publishes
its diagonal factor densely.

## Current call-path authority

| Public family | Admission/planning | Execution/publication |
|---|---|---|
| Construction | `TypedTensorRootDispatch` selected by `R::Mode` | Complete bound layout before payload allocation; publish one owned body |
| Transform/index operations | `TypedTensorTransformDispatch`, twist/flip dispatch, checked tree context | Stage fallible provider work, then replay to an owned output; identity cases may share storage |
| Product/contraction/trace | Mode dispatch and bound output-space derivation | Ordinary MF or checked lower seam; destination forms preserve caller ownership where exposed |
| Reductions | Mode-dispatched qdim weights | Scalar result only; no tensor/result cache |
| Decompositions/matrix functions | Factor-space admission and sector matricization | Request-local dense calls; publish all factors only after successful staging |
| Left/right solve | Receiver-owned output HomSpace and exact receiver provider `Arc` | One sector solve per populated block; direct/lazy work remains operation-local |
| Network | `Network::plan` plus runtime plan cache | Each step calls ordinary typed contraction/permutation; bounded workspace reuse |
| CUDA | MF-only typed device impl, payload `f64`/`Complex64` | Explicit transfer/device kernels; conjugation is a GEMM operand flag; unsupported scopes reject before publication |

Key source anchors at the pinned revision:

- `tenet/src/typed.rs`: `TypedTensor*Dispatch`, `TypedAdjointView`,
  `materialized_tensor_uncached`, checked solve/null/polar/EIGH/EIG dispatch.
- `tenet-matrixalgebra/src/factorize.rs`:
  `*_dyn_checked_generic`, checked factor-space builders, sector
  matricizations, truncation and numerical-rank gates.
- `tenet-network/src/network.rs`: `Network`, `PlannedNetwork`, ordinary replay.
- `tenet-network/src/plancache.rs`: bounded plan/workspace ownership and stats.

All checked-provider admissions use the tensor's exact provider `Arc`.
Fallible output admission and dense work are staged before a result is returned.
Provider-owned append-only catalog warming is not rolled back, but a failed
operation publishes no TeNeT output tensor or partial factor tuple.

## Lazy materialization

| Mechanism | Owner/trigger | Contract |
|---|---|---|
| Parent-backed lazy view | `TypedAdjointView` from `adjoint()` | Metadata plus canonical parent; no payload copy |
| Explicit logical payload | `materialize()` | Fresh owned dense tensor; the only public materialization |
| Operation-local logical payload | `materialized_tensor_uncached()`, compact densification | Documented part of the operation's cost; released with the operation, never retained by the receiver (#1548) |

Orientation-aware transforms, contractions and algebraic adjoint redirects use
the parent where their operation law proves it. QR/LQ, EIGH/EIG logical
orientation, `exp`, solves, selected cat/absorb conversions and network
rejection checks may use or encounter the operation-local seam. A cold receiver
cache proves only nonpublication; it does not prove zero peak-copy cost. The
ownership/cache audit is #783.

## Executable evidence

- `tenet/tests/typed_serialization.rs`: deterministic v1 Host snapshots,
  exact provider restoration, semantic fusion-tree keys, scalar bit patterns,
  dense/compact/lazy representation boundaries, decode limits, and malformed
  input ordering for multiplicity-free and checked-Generic modes.
- `tenet/tests/mf_structural_conformance.rs`: exact ZN(3), CU(1), SU(2),
  fermionic and exact-product structural, pivotal, contraction and reduction
  laws, including nontrivial CU(1) exchange phase and quantum-dimension oracles.
- `tenet/tests/mf_linalg_conformance.rs`: rectangular and square
  factorization, spectral and matrix-function laws for the remaining provider
  fixtures in #1002, with exact provider-`Arc` preservation.
- `tenet-network/tests/mf_provider_conformance.rs`: explicit and macro network
  execution, static trace and workspace reuse for the exact provider fixtures.
- `tenet/tests/checked_generic_facade.rs`: synthetic failure staging plus
  SU(3)/SU(4) full-key `f64`/`Complex64` oracles for reductions,
  decompositions, null/polar, EIGH/EIG, solve and matrix functions.
- `tenet/tests/checked_generic_twist.rs` and
  `tenet/tests/checked_generic_absorb.rs`: checked pivotal/index/cat/absorb
  layout, exact-Arc, lazy and failure contracts.
- `tenet-network/tests/checked_generic_network.rs`: explicit/greedy/macro
  replay, provider authority, cache/workspace recovery, trace and
  lazy/conjugated rejection boundaries.
- `tenet/tests/semantic_suite.rs` and `tenet/tests/typed_facade.rs`: U(1), Z2,
  fZ2, SU(2), CU(1), external finite-provider and exact product fixtures.
- `tenet/tests/inner_norm_allocations.rs` and
  `tenet/tests/permute_overwrite_allocations.rs`: reduction and destination
  allocation contracts.
- `tenet/tests/typed_cuda_transfer.rs` and
  `tenet-network/tests/typed_cuda_network.rs`: real-device MF transfer,
  operation and canonical-network gates, for both device payload dtypes.
- `tenet/tests/typed_cuda_transform.rs` and
  `tenet/tests/typed_cuda_transform_contracts.rs`: real-device
  `permute`/`braid`/`transpose`/`transpose_axes`/`repartition` against the Host
  answer for U(1), SU(2), fZ2, fZ2xU(1) and fZ2xSU(2), both device payload
  dtypes, plus the transfer, allocation, statistics and rejection contracts.
  The same two files carry the device
  `permute`/`transpose`/`transpose_axes`/`repartition` `*_into`
  gates: device == Host for every caller scale including `0` and `-0.0`, a
  NaN-poisoned destination, the Host precondition order with the Host's error
  text, the warm 0 H2D / 0 D2H / 0 device-allocation contract and exact-layout
  admission. Their device-free half — the dense physical-basis oracle pinned
  against the Host, the device-less Runtime state, and the Host
  `*_into` precondition order and wording the device mirrors — is
  `tenet/tests/typed_transform_host_side.rs`, which is not feature gated.
- `tenet/tests/beta_destinations.rs` (#1550): every linear `*_into`
  (`contract`, `permute`, `braid`, `transpose`, `repartition`, `trace_pairs`,
  `axpby`) against its eager operation combined through `axpby`, for
  `beta` in {0, 1, general}, NaN destinations under `beta = 0`, unreached
  contraction sectors left bit-identical under `beta = 1`, U(1)/SU(2)/fZ2xU(1)
  with dual legs, f64/c64/f32/c32 and lazy-adjoint inputs; its ignored CUDA
  test checks the device against the Host for the same cases.
- `tenet/tests/typed_cuda_twist.rs`: real-device `twist`/`twist_inverse`
  against the Host answer for fZ2, fZ2xU(1) and fZ2xSU(2), single-axis,
  multi-axis, repeated, dual and lazy-adjoint operands, both device payload
  dtypes, the exact `twist_inverse` round trip, the bosonic and all-legs
  identity short circuits and an empty space. Its transfer, allocation and
  rejection contracts are in `typed_cuda_transform_contracts.rs`; its
  device-free half — the `±1` factor domain the descriptor-scale lowering
  rests on — is in `typed_transform_host_side.rs`.
- `tenet/tests/physical_dense.rs`: U(1)/SU(2) Host physical expansion and
  projection, real and complex SU(2) round trips, and an independent
  TensorKit SU(2) coefficient oracle.

TensorKit's `matrixalgebrakit.jl` wrappers and MatrixAlgebraKit's factorization
interfaces are the API/law comparison, not proof of TeNeT execution. QSpace is
used only where a representation/coefficient oracle exists; no QSpace gauge or
provider-wide performance conclusion is inferred from SU(3)/SU(4) fixtures.

## Findings and routing

1. Checked Generic now has the ordinary Host path through network replay and
   static intra-operand trace. Network execution still returns a new tensor;
   caller-owned payload destinations remain an intentional difference recorded
   in [#1005](https://github.com/Ryo-wtnb11/TeNeT/issues/1005). Standalone compact
   construction is [#1004](https://github.com/Ryo-wtnb11/TeNeT/issues/1004).
   Lazy/conjugated contraction still rejects without fallback, and checked CUDA
   remains with [#3](https://github.com/Ryo-wtnb11/TeNeT/issues/3).
2. Exact fixture tests now cover the remaining multiplicity-free evidence gaps
   for U(1), Z2, built-in ZN(3), CU(1), fZ2, SU(2), and the two named product
   fixtures. This does not prove arbitrary moduli or arbitrary products.
3. B/C/D are not closure criteria for #938 and remain isolated in
   [#989](https://github.com/Ryo-wtnb11/TeNeT/issues/989).
4. The only current measured provider-operation gap in this inventory is U(1)/SU(2)
   Host `inner`, [#875](https://github.com/Ryo-wtnb11/TeNeT/issues/875).
5. Fibonacci, storage/device scope, physical dense conversion, slicing and
   release topology remain with their narrow owners listed above. The obsolete
   CUDA doctest finding is removed: current compile-fail examples use checked
   Generic bounds rather than U(1).

After the checked Generic provider route is complete, performance should be
reviewed as a separate measured phase. This matrix does not manufacture a
performance gap from call-path inspection alone.
