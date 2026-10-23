# Device factorizations at single precision — leaf C4 (#1341)

Device evidence for admitting `f32` and `Complex32` to `tenet`'s device
factorization marker `CudaFactorizationPayload` (`svd_compact`, `eigh_full`)
and `f32` to the new device QR marker `CudaQrPayload`, and for keeping both
complex payloads out of device QR as a compile-time boundary.

Audit note [historical audit](https://github.com/Ryo-wtnb11/symtn-rs/blob/8c54ce960d0dd5ea000c75ff1893624e3da42420/docs/audit/issue-1341-single-precision-device-factorizations.md);
predecessors [historical audit](https://github.com/Ryo-wtnb11/symtn-rs/blob/8c54ce960d0dd5ea000c75ff1893624e3da42420/docs/audit/issue-1336-single-precision-device-payload.md) (C2),
`benchmarks/history/cuda-scalar-single-precision-2026-09-21.md` (#1326, C1),
[historical audit](https://github.com/Ryo-wtnb11/symtn-rs/blob/8c54ce960d0dd5ea000c75ff1893624e3da42420/docs/audit/issue-1324-single-precision-factorizations.md) (host).

No dependency change.

## Environment

| | |
|---|---|
| TeNeT base | `origin/main` `91d15f97` (#1340), branch `c4-device-factorizations-single` |
| Host | `qg1`, `/data2/ryo-w/gpu-phase/c4`, private target `/data2/ryo-w/gpu-phase/c4-target` (removed after the run) |
| GPU | NVIDIA A100-SXM4-40GB, `CUDA_VISIBLE_DEVICES=0` (all eight GPUs held no compute app of any user before and after the run; no Rust build of ours was active) |
| CUDA / cuTENSOR | 12.6 (`/usr/local/cuda-12.6`) / `libcutensor.so.2.5.0` via `TENFERRO_CUTENSOR_PATH` |
| Tenferro | 0.5.0 (`tenferro-tensor`, `tenferro-gpu`, `tenferro-linalg`), unchanged |
| Build | `dev`, `--no-default-features --features cuda,cpu-faer`, `--test-threads=1` |
| Toolchain | `cargo 1.96.0 (30a34c682 2026-05-25)` |

No timing claim is made: `dev` profile, correctness and counter contracts only.

## What opened and what stayed closed

Three production sites (see the audit note for the table):

* `impl CudaFactorizationPayload for f32 / Complex32` — device `svd_compact`
  and `eigh_full` open at single precision.
* A new marker `CudaQrPayload` (`f64`, `f32`), a **projection** of the adapter
  capability constant `CudaScalar::DEVICE_CONSTANT_KERNELS` held equal to it by
  a `const` assertion. The constant stays the single authority; a drift fails
  the build with a message naming the repair.
* Device `qr_compact` generalised from its concrete `f64` impl block to
  `D: CudaQrPayload` — two dtype-generic lines in the body
  (`vec![D::ZERO; ..]`, `cuda_qr_region::<D>`) and the block's bound.

Complex device QR stays a compile-time boundary. The defect is the complex
zero constant of the positive-diagonal gauge's `triu` kernel
(tenferro-rs#1833 / #1271), not the precision, so it excludes `Complex32` and
`Complex64` and admits both real payloads.

## Device summary lines

Targeted run (`c4-run1.log`), in order: the new factorization suite
(non-ignored, then `--ignored`), the base single-precision suite, the device
network suite, and the `cuda` doctests.

```
test result: ok. 0 passed; 0 failed; 9 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.10s
test result: ok. 3 passed; 0 failed; 11 ignored; 0 measured; 0 filtered out; finished in 0.03s
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 4.10s
test result: ok. 0 passed; 0 failed; 12 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.09s
test result: ok. 92 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 9.07s
```

The doctest line is the `compile_fail` evidence: 92 passed, 0 failed, so every
pin that must still fail to compile does, and every twin this leaf turned into
a compiling example compiles.

Full device suite (`c4-run2.log`),
`cargo test --workspace --lib --tests --no-default-features --features cuda,cpu-faer --no-fail-fast -- --ignored --skip measure_checked_generic_transform_phases --skip axioms_ --skip itebd_ --skip cross_library --test-threads=1`:

118 test-result lines, aggregated:

```
passed 185 failed 0 ignored 0
```

Every `--ignored` device gate in the workspace, at this branch, on this A100.

## Contracts measured

Relative and same-process: the single-precision run of a fixture is compared
against the double-precision run of the same fixture in the same test binary.
No absolute platform constant appears.

`device_factorizations_cost_the_same_calls_and_half_the_bytes`, one warm
`svd_compact` plus one warm `eigh_full` over a U(1) `[2,3,2]` endomorphism,
as `(h2d_calls, h2d_bytes, d2h_calls, d2h_bytes, device_allocs, copy_calls,
gemm_calls)`:

| Pair | narrow | wide |
|---|---|---|
| `f32` / `f64` | `(11, 420, 15, 92, 20, 6, 7)` | `(11, 840, 15, 184, 20, 6, 7)` |
| `Complex32` / `Complex64` | `(11, 828, 15, 92, 20, 6, 7)` | `(11, 1656, 15, 184, 20, 6, 7)` |

Every call count is equal across the pair and both byte counts are exactly
halved. The D2H bytes are the spectra, which are the payload's *real* lane, so
they halve for the complex pair too while the H2D payload bytes do not halve
between `f32` and `Complex32` — that difference is the real/complex split, not
the precision one, and is why both pairs are measured.

`a_warm_single_precision_chain_costs_the_same_calls_and_half_the_bytes`
(`tenet-network`), warm `tensor!` three-tensor chain, as `(h2d_calls,
h2d_bytes, d2h_calls, device_allocs, copy_calls, gemm_calls)`:

| Pair | narrow | wide |
|---|---|---|
| `f32` / `f64` | `(1, 80, 0, 1, 1, 4)` | `(1, 160, 0, 1, 1, 4)` |
| `Complex32` / `Complex64` | `(1, 160, 0, 1, 1, 4)` | `(1, 320, 0, 1, 1, 4)` |

The one warm H2D call is the per-dtype zero template; it is one element wide,
so its bytes halve with the element size and its call count does not move.

## Correctness evidence

`tenet/tests/typed_cuda_single_precision_factorizations.rs`, nine gates, each
one generic body instantiated over the dtypes its marker admits, with the
double-precision instantiation as the control:

* `device_svd_compact_matches_the_host_at_every_payload` — structure, spectrum,
  descending non-negative diagonal, `u` isometry, `vh` coisometry,
  reconstruction, source unchanged bit for bit; square, tall and wide fixtures
  over U(1), SU(2) and `fZ2 x U(1) x SU(2)`.
* `device_qr_compact_matches_the_host_at_every_real_payload` — the same, plus a
  pointwise factor comparison, which QR alone permits because its gauge is
  fixed on device.
* `device_qr_returns_the_positive_diagonal_gauge_at_every_real_payload` —
  `R_jj > 0` exactly, imaginary part within the fixture's own tolerance.
* `device_eigh_full_matches_the_host_at_every_payload` — over a
  positive-definite **and** an indefinite fixture, the second being what
  separates the `|lambda|`-descending contract from cuSOLVER's own ascending
  order.
* `device_eigh_admits_a_nearly_hermitian_single_precision_block` — a relative
  anti-Hermitian residual near `6e-8`: admitted at `f32` and `Complex32`,
  rejected at `f64` on the same numbers. That asymmetry is the C1 fix (the rule
  is `64 * eps(real(D))`, not `64 * eps(f64)`) observed from the typed layer.
  A gross perturbation is rejected at every payload, naming the requirement.
* `device_factorization_rejections_do_not_depend_on_the_payload` — a
  lazy-adjoint receiver is rejected by `svd_compact`/`eigh_full` as an operand,
  the truncated entry points report the missing *capability* even on that
  receiver, and every device counter is unchanged across all of it.
* `device_factorizations_handle_blocks_at_unaligned_offsets_at_every_payload` —
  #1320 at 4- and 8-byte elements; the degeneracy pairs put the second sector's
  block at element offsets 9, 25 and 9, which are unaligned for all four
  payload sizes at once, with `(4, 2)` and `(2, 2)` as controls.
* `the_device_truncation_composition_matches_host_svd_trunc_at_every_payload` —
  the #1297 recipe at every payload: kept bond space exactly, kept spectrum and
  discarded weight within tolerance, truncated reconstruction. Budgets land in
  gaps of the fixture spectrum, so no `rank` boundary falls on a tie.
* `device_factorizations_cost_the_same_calls_and_half_the_bytes` — above.

Oracle and tolerance: the host factorization of the same tensor at the same
payload dtype, read through gauge-independent identities;
`K * sqrt(n) * eps(real(D)) * scale * kappa` with `kappa` measured from the
double-precision twin of the same fixture. See the audit note.

## Carried device-test follow-ups closed here

From the #1336 review:

* `device_single_precision_normalize_of_an_overflowed_norm_is_all_zero` —
  characterises a known limitation (Ryo-wtnb11/TeNeT#1344), not an intended
  contract: `inner(a, a)` reports `+inf` at `f32` and `normalize` returns an
  all-zero tensor with no error, with the `f64` twin of the same fixture as
  the control.
* `a_zero_scale_overwrite_into_clears_a_nan_poisoned_destination` — `alpha = 0`
  and `-0` over a `NaN`-poisoned destination, at all four payloads. The finite
  poison of the existing loop cannot tell an overwrite from a
  scale-and-accumulate; `NaN` can.
* The reduction gates now assert the bound their documentation states.
  `assert_close_absolutely` drops the `(1 + |expected|)` factor #1336 left in
  place for want of a device run; validated here.
* `tenet-network/tests/typed_cuda_network.rs` gained an SU(2) single-precision
  chain and a `Complex32` warm-cost twin, and its chain tolerances are derived
  from the payload epsilon instead of being written as constants.
* The `DevicePayload` harness trait moved to `tenet/tests/common/mod.rs`, so
  the base and factorization device suites share one definition.
* `tenet-network/examples/cuda_operation_matrix.rs` gained `f32` and
  `Complex32` rows behind a new `--precision double|single|all` flag,
  defaulting to `double`. The base/factorization harness split #1336 expected
  turned out to be unnecessary — it existed only because `f32` was not a
  `CudaFactorizationPayload`. Smoke run only; the pinned baseline CSVs are
  untouched.

  **The flag is a resource limit, not a preference.** The first smoke run
  emitted all four lanes in one process and died 1900 rows in, inside the last
  provider, at `Runtime::builder().cuda(..).build()`:

  ```
  thread 'main' panicked at tenet-network/examples/cuda_operation_matrix.rs:607:14:
  CUDA Runtime for the measured row: Operation(Dense(Backend { backend: Cuda,
    op: "cuda_matmul", message: "cuda_cutensor: extension cuda failed:
    cuTENSOR call cutensorCreate returned status 14" }))
  ```

  The harness protocol builds a **fresh `Runtime` per row**, so each row
  creates a CUDA context and a cuTENSOR handle, and roughly tripling the row
  count crosses a limit. The failure is cumulative, not dtype-specific: it
  struck an `f64` row set after every single-precision row of the three
  preceding providers had passed. Emitting the lanes in separate invocations
  keeps each process inside the limit; `--precision single` then completes with
  `EXIT=0`, 1234 lines, every provider and family, and zero `not-ok` verdicts.

  The underlying per-`Runtime` device-resource retention is a pre-existing
  property of this harness that single precision merely made visible. It is
  **not** fixed here — that is a `Runtime`/backend lifetime question, not a
  test-fixture one — and it is left as a residual below.

## Residuals

**The operation-matrix harness retains device resources per row.** A fresh
`Runtime` per row is the protocol, and the resources each one takes are not
fully returned when it drops, so a long enough single process fails at
`cutensorCreate`. The `--precision` flag keeps every invocation short enough,
which is a workaround and is labelled as one. Whether `Runtime` — or the
tenferro CUDA backend behind it — should release its cuTENSOR handle on drop
belongs to its own issue, Ryo-wtnb11/TeNeT#1343; no production caller builds
one `Runtime` per operation, so the library's own device path is unaffected.

**Carried to a later device leaf** (independent review P2-2, P2-3): a nonzero
64-element-aligned `f32` factor offset control (e.g. `d0 = 8`) for the #1320
test, and an `f32`-vs-`f64` cost contract for device `qr_compact`. Neither
needed a code change here, and no device run backs this follow-up commit.

See the audit note. In short: a compact-diagonal *device* receiver is
unreachable through the public API (`to_cuda` densifies), so only the
lazy-adjoint half of that rejection order is testable on device; complex device
QR waits on tenferro-rs#1833 / #1271 in a released Tenferro; persistence and
the host advanced-linalg family stay closed for single precision.
