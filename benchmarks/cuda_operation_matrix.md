# CUDA operation matrix

Revision-pinned device performance baseline of the existing CUDA path
(issue #1273, roadmap #1139). It is a validation fixture: it records what the
current device path costs per call, it is never a CI timing gate, and no
measured value feeds a dispatch decision.

Runner: `benchmarks/cuda_operation_matrix.sh`
Executable: `symtn-network/benches/cuda_operation_matrix.rs`
(feature `cuda`; without it the binary prints an unsupported message and exits
with status 2).

The example lives in `symtn-network`, not `symtn`, because the canonical
network row needs `Network`, and `symtn-network` depends on `symtn` rather
than the other way round.

## Protocol

Mirrors `benchmarks/operation_matrix.md`:

- A fresh `Runtime`, serial `CpuEngine` and `CudaEngine` per row.
- The Host fixture is built and uploaded **before** any timer starts.
- `cold`: the genuine first call of the measured operation on that engine,
  one iteration. Nothing calls the operation before this: `bench` hands the
  first call's output back to the caller, which is what the correctness check
  consumes.
  Fixture setup warms the engine's tenferro backend libraries (cuTENSOR,
  cuSOLVER/cuBLAS handles, `CudaDenseContext::warm_up`), so `cold` does not
  contain that one-time handle initialization. What a `cold`
  cell carries beyond the warm cost is CubeCL's NVRTC compile of each kernel
  family it first touches, the cuTENSOR plan for its shape, and the call's
  device execution scratch (tree-transform coefficient uploads and
  workspaces), which is invocation-local and so is also paid by every `warm`
  call. The dated record
  `cuda-baseline-2026-09-20.md` predates the warm-up and includes the
  initialization in every `cold` cell.
- `first_after_setup`: used instead of `cold` for the two transfer rows, whose
  measured operation (`to_cuda`) also runs while the fixture is built. This
  follows the Host harness, which labels a phase that way whenever a preflight
  call is unavoidable.
- `warm`: `--warmup` unmeasured calls (default 3), then `--iterations`
  measured calls (default 20). Every column is the **per-iteration median**;
  `ns_min` is the minimum elapsed time over the same iterations.
- The first call's result is returned rather than dropped inside the timer, and
  warm results are dropped after the clock stops, so a row reports the cost of
  producing the result, not of releasing it.
- The correctness check runs **after both timed phases**, on the first call's
  output, and its verdict is reported in the `check` column of every row of
  that operation.
- Each row is also run against the same fixture on Host (`target=host`) in the
  same process, first call then warm, so a device row that is slower than Host
  is visible as such. The Host arm's first-call output is the oracle, so no
  extra Host call happens outside a timer either.

### Sampling

One process per invocation. The Host `operation_matrix.sh` wrapper takes a
three-process median; this runner does not, so a timing is this process's
per-iteration median with no between-process variance estimate.

### Engine construction

Every row opens a fresh `CudaEngine` and runs `CudaDenseContext::warm_up` on it
during fixture setup, so backend library initialization (cuTENSOR, cuSOLVER,
cuBLAS handles) stays out of the `cold` cells. The header line
`# engine_open_ns first=... second=...` reports one `CudaEngine::open` plus
`warm_up` each, as single observations rather than medians. `first` is the
process's first CUDA engine and also pays CUDA context creation; `second` is a
second fresh engine in the same process. The line exists so that the
initialization the `cold` cells do not carry stays visible.

### Synchronization and the completion barrier

No device synchronization is reachable through symtn's public API, and
tenferro's `with_cubecl` success path does not synchronize. A timed region
therefore measures submission unless the operation itself downloads (`norm`,
`inner`, the factorizations' spectra).

The completion barrier that runs after every timed call is `norm()` on one
fixture input. It is **not** a one-scalar operation: it submits one GEMM per
coupled block and downloads the per-block partials. Its downloads are recorded
in the `barrier_d2h_*` columns; its uploads and GEMMs are not recorded at all,
and none of its traffic enters the phase columns. It is used because no
genuinely one-scalar device operation exists on the public surface, and a
download on the same stream still orders after the work the timed call
submitted.

### Peak device memory

Absent. Neither symtn nor tenferro exposes a device allocation high-water
mark, and an `nvidia-smi --query-gpu=memory.used` sample would observe the
whole device rather than this process's buffers. `device_allocs` counts
buffers, not bytes retained.

## Rows

| Dimension | Values |
|---|---|
| provider | `U1`, `fZ2`, `SU2`, `U1xfZ2` (fermion parity x U(1) product) |
| dtype | `f64`, `c64` (`qr_compact` rows are real-payload only, the pinned row set; device QR admits every payload, #1271) |
| family | `many-small`, `few-large` |
| operation | `to_cuda`, `to_host`, `contract_direct`, `contract_lazy_adjoint_lhs`, `compose`, `scale`, `add_owned`, `add_lazy_fold`, `norm`, `inner`, `svd_compact`, `svd_trunc_composition`, `eigh_full`, `qr_compact`, `network_chain3` |
| target | `cuda`, `host` |
| phase | `cold` (or `first_after_setup` for the transfer rows), `warm`, `skipped` |

All fixtures are rank-2 endomorphisms `[V; V]`, which is what every existing
device test uses and what the device contract/compose route accepts (whole
codomain against whole domain, canonical order, identity output order).
`eigh_full` uses a separately built Hermitian fixture; `svd_trunc_composition`
uses `Truncation::rank(degeneracy)`; `network_chain3` is the canonical three-tensor
`[p; s] = a[p; q] * b[q; r] * c[r; s]` chain, run with `Network::contract`.

The truncated SVD is not an operation of its own (#1534), so
`svd_trunc_composition` measures the composition in both arms. The `cuda` arm runs device
`svd_compact`, a D2H of all three factors, then the Host `diagview` /
`find_truncated` / `restrict_leg` chain; the `host` arm
runs the same chain on Host `svd_compact`. The two arms are therefore not the
same work, and rows named `svd_trunc_rank` in the pinned baselines under
`benchmarks/history/` measure the removed fused device path, not this one.

A row whose device probe returns an error is printed once per target with
`phase=skipped` and the error text in `check`, so an unsupported boundary stays
visible in the table.

## Fixture parameters

Explicit CLI inputs, never used for dispatch. Both take one value per family,
`many-small,few-large`:

| Flag | Default | Meaning |
|---|---|---|
| `--blocks` | `64,4` | sectors per leg, i.e. coupled blocks of the endomorphism |
| `--degeneracy` | `4,64` | degeneracy per sector, i.e. each block is `d x d` |
| `--iterations` | `20` | measured warm calls |
| `--warmup` | `3` | unmeasured calls before the warm phase |
| `--device` | `0` | CUDA device ordinal |

So `many-small` is 64 blocks of `4 x 4` and `few-large` is 4 blocks of
`64 x 64`. Fermion parity has exactly two irreps, so its realized block count
is 2 in both families; that is a provider capability, not a decision made here.
The `blocks` column records the requested value.

## Columns

| Column | Meaning |
|---|---|
| `provider,dtype,family,blocks,degeneracy,operation,target,phase` | row identity |
| `iterations` | calls behind this row (1 for the first call, `--iterations` for `warm`) |
| `ns_median`, `ns_min` | per-iteration elapsed nanoseconds |
| `alloc_calls`, `alloc_bytes` | caller-thread Rust allocation calls and requested bytes during the timed region; excludes worker threads, frees, and device memory |
| `h2d_calls`, `h2d_bytes` | host-to-device transfers at the CUDA dense boundary |
| `d2h_calls`, `d2h_bytes` | device-to-host transfers at the same boundary |
| `device_allocs` | device buffers created or taken ownership of: uploads plus tenferro tensors wrapped as `CudaDenseStorage`; excludes tenferro-internal solver workspaces and the intermediate tensors of the Hermiticity check |
| `gemm_calls` | `dot_general` submissions |
| `solver_calls` | cuSOLVER region calls (SVD, QR, EIGH) |
| `copy_calls` | `cuda_copy_region_into` calls that move data |
| `barrier_d2h_calls`, `barrier_d2h_bytes` | the completion barrier's own downloads, outside the timed region |
| `check` | `ok:<oracle>`, `MISMATCH:<oracle>`, or the skip reason |

The device columns come from `symtn::expert::cuda_transfer_stats()`, an
always-compiled, `Relaxed`, backend-local counter set in
`symtn-dense/src/cuda_adapter/mod.rs`. They are observability only: nothing in the
library reads them back, so no execution decision depends on them.

## Oracles

| Operation | Check |
|---|---|
| `to_cuda` / `to_host` | roundtrip payload is bit-equal to the Host source |
| `contract_*`, `compose`, `scale`, `add_*`, `network_chain3` | downloaded device result equals the Host result to `1e-9` relative |
| `norm`, `inner` | device scalar equals the Host scalar to `1e-9` relative |
| `svd_compact`, `eigh_full`, `qr_compact` | the device factors recompose on device to the source (gauge-independent) |
| `svd_trunc_composition` | kept spectrum per coupled sector and discarded weight match the Host composition |

## Absent references

Neither TensorKit nor QSpace has a matched CUDA fixture for these rows at this
revision, so there is no cross-library device comparison to record.
