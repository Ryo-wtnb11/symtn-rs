# Typed single-precision device tensors — leaf C2 (#1336)

Device evidence for admitting `f32` and `Complex32` to `tenet`'s device payload
marker `CudaPayload`, and for closing the device factorization family behind
the new sealed marker `CudaFactorizationPayload`.

Audit note [historical audit](https://github.com/Ryo-wtnb11/symtn-rs/blob/8c54ce960d0dd5ea000c75ff1893624e3da42420/docs/audit/issue-1336-single-precision-device-payload.md); survey
`reviews/gpu-phase-20260920/single-precision-survey.md` (row C2); adapter leaf
`benchmarks/history/cuda-scalar-single-precision-2026-09-21.md` (#1326, C1);
probe `benchmarks/history/cuda-single-precision-probe-2026-09-20.md` (#1303,
C0).

No dependency change.

## Environment

| | |
|---|---|
| TeNeT base | `origin/main` `ab47d16c` (#1339), branch `c2-cuda-payload-single-precision` |
| Host | `qg1`, `/data2/ryo-w/gpu-phase/c2`, private target `/data2/ryo-w/gpu-phase/c2-target` (removed after the run) |
| GPU | NVIDIA A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES=0` (all eight GPUs held no compute app of any user before and after the run; no Rust build of ours was active) |
| CUDA / cuTENSOR | 12.6 (`/usr/local/cuda-12.6`) / `libcutensor.so.2.5.0` via `TENFERRO_CUTENSOR_PATH` |
| Tenferro | 0.5.0 (`tenferro-tensor`, `tenferro-gpu`, `tenferro-linalg`), unchanged |
| Build | `dev`, `--no-default-features --features cuda,cpu-faer`, `--test-threads=1` |
| Toolchain | `cargo 1.96.0 (30a34c682 2026-05-25)` |

No timing claim is made: `dev` profile, correctness and counter contracts only.

## What opened and what stayed closed

One production line opens the family — `impl CudaPayload for f32 / Complex32`.
Everything in the base device family was already generic over `CudaPayload`.
The per-dtype plumbing (execution lanes, device scalar operands, the context
zero template, the network workspace pools) was already per dtype from #1315
and #1326, and was verified rather than assumed; see the audit note.

The device factorizations were bounded `CudaPayload + FactorizationScalar`,
both halves of which now hold for single precision. They are closed by the
sealed marker `CudaFactorizationPayload` (`f64`/`Complex64` only), pinned by
six `compile_fail` doctests each with a compiling twin. That is leaf C4.

## Reduction accumulation contract

Within a coupled sector the device sums in the payload dtype, inside the
backend GEMM: `dot_general_read_into_accum`
(`tenferro-gpu-0.5.0/src/cubecl/gemm.rs:726`) dispatches on the single dtype
shared by both operands and the destination, `accum_erased` (`:743`) reads all
three as one `T`, and the cuTENSOR compute descriptor is fixed per dtype with
no caller control (`CUTENSOR_COMPUTE_DESC_32F` at `gemm.rs:101`, `..._64F` at
`:134`). Tenferro 0.5.0 therefore offers no widening reduction, and a widened
device sum would cost a second pass — so the device reduction algorithm is
unchanged by this leaf.

Across coupled sectors the host half now accumulates in `WideScalar::Wide` and
narrows once, like every host reduction; `norm` takes its root from the wide
accumulator. `Wide = Self` for the double-precision pair, so those results are
unchanged down to the emitted arithmetic.

The observable consequence is the overflow boundary: a `norm` finite on the
host can be `inf` on the device at single precision. It is reported as `inf`,
the convention `f64` overflow already has, and it is asserted rather than
tolerated.

## Device results

Four phases in one run on the tree rebased onto `ab47d16c`.

### 1. The new single-precision suite

`tenet/tests/typed_cuda_single_precision.rs`, `--ignored --test-threads=1`:

```
test a_warm_single_precision_overwrite_into_transfers_nothing ... ok
test device_arithmetic_matches_the_host_at_every_payload ... ok
test device_contract_and_compose_match_the_host_at_every_payload ... ok
test device_fermionic_signs_are_exact_at_every_payload ... ok
test device_reductions_match_the_host_within_the_device_accumulation_bound ... ok
test device_single_precision_norm_can_overflow_where_the_host_stays_finite ... ok
test device_transfer_round_trips_bit_exactly_at_every_payload ... ok
test device_transforms_match_the_host_at_every_payload ... ok
test single_precision_costs_the_same_device_calls_and_half_the_bytes ... ok

test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 3.53s
```

The three filtered-out tests were the device-free gates below, which do not
carry `#[ignore]` and run in the non-ignored phase; one of the three was
deleted on review afterwards (see phase 3). No `#[ignore]` gate changed after
the run.

Suites instantiated, each as one generic body over all four device dtypes with
the host result of the same dtype as oracle:

| Suite | Families | Providers |
|---|---|---|
| transfer | round trip (bit exact), lazy-adjoint round trip, structure preservation | U(1), SU(2), fZ2 ⊠ SU(2) |
| arithmetic | `scale`, `add`, `zeros_like`, `normalize`, the lazy-adjoint fold, the lazy/owned rejection | U(1), SU(2) |
| reductions | `norm`, `inner`, `dot`, conjugate linearity in the first argument | U(1), SU(2) |
| contraction | `contract`, `compose`, lazy-adjoint `compose`, `contract_overwrite_into` (incl. idempotence) | U(1), SU(2), fZ2 ⊠ SU(2) |
| fermionic signs | the hand-computed `-6`/`+6` fixture, **bit exact** at every dtype | fZ2 |
| transforms | `permute`, `braid`, `repartition`, `transpose`, round trip, and the four `*_overwrite_into` forms of #1339 at `alpha ∈ {1, -2.5+0.5i, 0, -0}` over poisoned destinations | U(1), SU(2), fZ2 ⊠ SU(2) |
| cost | equal device calls, halved bytes; warm `*_overwrite_into` transfers and allocates nothing | U(1) |

### 2. Device network chains

`tenet-network/tests/typed_cuda_network.rs`, `--ignored --test-threads=1`:

```
test a_warm_single_precision_chain_costs_the_same_calls_and_half_the_bytes ... ok
test single_precision_device_chains_match_the_host_and_reuse_their_destinations ... ok
```

with the ten pre-existing device network gates unchanged and still passing:

```
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.29s
```

### 3. Non-ignored tests and doctests with the `cuda` feature

All non-ignored tests of `tenet-rs`, `tenet-network`, `tenet-operations` and
`tenet-dense` with the `cuda` feature set — the device-free gates of this leaf
live here, so ordinary `cargo test --features cuda` runs them:

```
exit_nonignored=0
```

40 test binaries, **415 passed, 0 failed**. The device-free gates this leaf
adds are `every_base_family_payload_is_a_device_payload` and
`a_device_less_runtime_rejects_every_payload_the_same_way`. (A third,
`device_storage_is_distinct_per_payload`, ran in this phase and was deleted
afterwards on review: comparing `type_name::<CudaStorage<D>>()` across dtypes
cannot fail, so it asserted nothing. Its removal is the only test change after
the device run, and it removes a test rather than a gate.)

Doctests of `tenet-rs` with the `cuda` feature, which is where every
`compile_fail` pin and its compiling twin lives:

```
test result: ok. 90 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 9.18s
exit_doc=0
```

### 4. Full workspace device suite

`cargo test --workspace --lib --tests --no-default-features --features
cuda,cpu-faer --no-fail-fast -- --ignored --skip
measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip
cross_library --test-threads=1`:

```
exit_suite=0
```

117 test binaries, **174 passed, 0 failed, 0 ignored**; no `FAILED` line in the
run log. (The same phase was 155 on the C1 tree at `4442fef8`; the difference
is this leaf's 11 device gates and #1339's.)

`nvidia-smi --query-compute-apps` was empty before and after the run, and the
private target directory was removed afterwards.

## Contracts measured

Both are relative, same-process comparisons: the single-precision run of a
fixture against the double-precision run of the *same* fixture in the same test
binary. No absolute platform constant is asserted.

**Typed contraction fixture** (one U(1) rank-2 pair: upload both operands,
contract, download), measured as
`(h2d_calls, h2d_bytes, d2h_calls, d2h_bytes, device_allocs, gemm_calls)` after
both lanes are warm:

| Comparison | single precision | double precision |
|---|---|---|
| `f32` against `f64` | `(3, 108, 1, 36, 3, 3)` | `(3, 216, 1, 72, 3, 3)` |
| `Complex32` against `Complex64` | `(3, 216, 1, 72, 3, 3)` | `(3, 432, 1, 144, 3, 3)` |

Every call count is equal and every byte count is exactly halved. The
`Complex32` row equals the `f64` row byte for byte, which is the element-size
identity the admission predicts (8 bytes either way).

**Warm `tensor!` device chain** (three U(1) tensors, steady state), measured as
`(h2d_calls, h2d_bytes, d2h_calls, device_allocs, copy_calls, gemm_calls)`:

| | value |
|---|---|
| `f32` | `(1, 80, 0, 1, 1, 4)` |
| `f64` | `(1, 160, 0, 1, 1, 4)` |

The retained-destination contract is unchanged by the payload: one upload (the
returned output's own zeros, #740), one allocation, one D2D reset copy from the
per-dtype zero template, no download, four GEMMs — and half the bytes.

**Warm `*_overwrite_into`** (#1339) at `f64`, `f32` and `Complex32`: a warm
replay transfers nothing and allocates nothing at every dtype, the device
tree-transform state does not grow, and the zero-scale route uploads at most
one *element* of the payload's own zero template — `size_of::<D>()` bytes,
once per context, so single precision pays half of what double precision pays
there too.

## Residuals

* **C3** — warm-up per dtype: nothing added; the warmed libraries are per
  backend instance and the C0 probe measured no per-dtype stall.
* **C4** — device factorizations at single precision: closed here behind
  `CudaFactorizationPayload`, with the `compile_fail` pins as the handover.
* `tenet-network/examples/cuda_operation_matrix.rs` gains no single-precision
  rows: its `HarnessScalar` is bounded on the device factorization family, so
  that is not the mechanical change the leaf allowed. Its bound was
  re-expressed as `CudaFactorizationPayload`; the pinned baseline CSVs are
  untouched.
