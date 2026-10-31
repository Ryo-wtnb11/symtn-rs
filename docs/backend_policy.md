# Backend selection policy

symtn owns categorical semantics and layout decisions. Tenferro owns dense
kernels, factorization providers and execution resources; Strided owns
applicable strided movement. Reuse those boundaries before adding a trait,
selection knob or retained execution state. Preserve the
[structured-complexity contract](complexity_parity_policy.md).

## Selection

`symtn-dense` owns `DenseExecutor`; `symtn-operations` owns
`TreeTransformBackend`; `symtn-tensors` owns `TensorContractBackend` and
`HostTensorContractBackend`. Operators describe spaces, axes, conjugation and
output order; backend adapters choose dense kernels. Do not put raw BLAS calls or
threshold-selected numerical kernels in operator code.

The CPU provider is a compile-time choice: faer by default, BLAS/LAPACK
when `cpu-blas` or a `blas-*` feature is enabled (OpenBLAS, MKL or
Accelerate as the linked implementation). It is not a runtime or
per-operation switch; see [`symtn/Cargo.toml`](../symtn/Cargo.toml).
With feature `cuda`, a device operation runs on the caller's `CudaEngine`
(`CudaEngine::open(CudaDeviceId::from_ordinal(n))`), passed like any other
engine; a `Runtime` is not tied to a device.

On the native provider a Host GEMM batch that is neither one job nor one
affine run, with or without transposed or adjoint operands, runs on symtn's
block GEMM kernel: one `tcontract::Plan` per distinct block geometry (scalar
type, extents, effective strides, conjugation). The plans live in the
process-global structure cache `StructureCacheKind::GemmPlans`, shared by
every call, `Runtime`, engine and prepared workspace: a warm call, eager or
prepared, of any member count, looks each geometry up once and prepares no
plan. The call's staging (O(J) job records, the geometry index, plan
handles, costs and the lane table) belongs to the dense executor that runs
it: an eager call's executor lives for the call, while a prepared
`ContractPlan`/`ComposePlan` workspace keeps its executor, so a warm replay
on a serial engine allocates nothing; that staging keeps its high-water
capacity and is part of `ContractWorkspace::retained_bytes`. A pooled
engine's multi-lane fork still hands its lanes their jobs per call. Offsets, payloads, `alpha`/`beta` and the engine's width and
policy are execute-time inputs, not part of the key. Each entry is charged
its inline size, key and entry overhead, 16 KiB, and 8 bytes per
`3m + 3n + 2k + 4` (a Packed plan's scatter vectors); with a budget of zero,
or for an entry the budget rejects, every call prepares its plans as before
with the same results. This holds for the faer provider and the four payload
scalars only; BLAS and CUDA builds prepare as before, and single-job or
affine batches use Tenferro's binary-plan cache, outside this cache.

Backend choice does not change mathematical semantics or numerical tolerance
policy. Capabilities may differ and must be explicit. New dense capabilities
belong behind the dependency adapter, with independent correctness tests;
performance evidence is separate from correctness gates.

## Storage and capability bounds

Host is the broadest path. CUDA uses `TensorMap<R, D, CudaStorage<D>>` and
explicit `to_cuda` / `to_host` transfers. A missing device implementation is
a compile-time boundary or a typed unsupported error, never a silent Host
fallback.

Device operations require multiplicity-free providers with `f64` categorical
coefficients and the operation's payload bounds (`f32`, `f64`, `Complex32`,
`Complex64` where implemented). Checked Generic and complex-symbol table
operations are Host-only. Uploading a table payload does not enable device
operations. Per-method bounds remain authoritative; see the
[provider interface](provider_interface.md#current-typed-api-boundaries).

## Parallel execution

Every numerical operation takes the caller's engine
(`engine: &mut dyn Engine`). A `CpuEngine::serial()` runs on the calling
thread; `CpuEngine::new(pool)` runs on a caller-owned `Pool` of threads. The
engine owns the threads and the dense provider's resources an operation uses:
its thread team, its buffer pool and its provider scratch. symtn stores no
engine, pool or thread setting, and an engine that is broken by an earlier
panic is refused with a typed error instead of a serial fallback. To run
operations concurrently, hold one engine per thread and share the tensors and
the `Runtime`.

Host parallelism is the engine's own decision. Dense kernels split work by
Tenferro's batch policy on the engine, an engine-owned setting symtn never
overrides. Under the default policy (`CpuBatchStrategy::Auto`) the native
provider's block GEMM kernel (recoupling, block-wise contraction) takes its
thread count from the engine's cost rule over the batch's estimated serial
time (Tenferro's own per-flop model), and its lane count from the engine's
batch decision. The lanes share all of those threads: a batch of a few large
blocks gives each lane several threads, and a batch of many small blocks runs
one thread per lane. Blocks are placed largest first on the least-loaded lane; prepared
`ContractPlan`/`ComposePlan` workspaces of any member count run their grouped
core batches the same way. The native provider's compact SVD over a
tensor's sector blocks runs on the same lanes: each block's cost is the
provider's SVD work (`m n min(m, n)`, four times that for complex scalars)
under the engine's work model, and each block's factorization gets its faer
width from the engine's cost rule within its lane's share, so a small block
takes no width the engine would not grant it. Every block runs up to the
first one refused before the engine is entered (an unsupported dtype or a
malformed view); the error reported is the lowest-indexed block's, the one the per-matrix SVD would
report, and nothing is returned on error. Each block allocates its own
provider workspace. QR and the other factorizations, and every BLAS
factorization, still run block after block on the whole engine.
Callers can still give each GEMM the whole pool with
`CpuEngine::new(pool).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Inner))`
(or `CpuEngine::set_batch_policy`; `CpuBatchPolicy` and `CpuBatchStrategy`
are in `tenferro_cpu`), or force lanes with `Outer` or `Split`. A BLAS provider under `Auto` instead runs the batch's
vendor GEMMs one after another on the calling thread; any threads the vendor
library starts are outside the engine's lanes. Forcing `Inner` (or another
lane strategy) on a BLAS provider runs those GEMMs on the native kernels, not
the vendor library. A backend-free Host region
(tree-transform replay, plan compile, strided moves) takes the engine's
placement token; a replay forks over the engine's width only when the
engine's cost rule (threadteam's `pays`, with strided's copy estimate for the
destination length against the pool's dispatch cost) admits the parallel
section. A serial engine never forks. Results are bit-identical for one
configuration (provider, build, engine width, inputs); no promise is made
across widths or providers. BLAS-internal threads remain separate.

[`Runtime`](../symtn/src/runtime.rs) holds prepared structural state only.
An eager operation builds its execution scratch (tree-transform and
contraction lanes, a factorization's dense executor) for the call and drops
it on return, as TensorKit allocates its temporaries per call; a prepared
plan's workspace keeps that scratch for its replays. Structural and plan
caches take short lookup and publication locks.

A CUDA operation borrows the caller's `CudaEngine` for its duration and
stores nothing of it. The engine owns its stream, its allocation domain and
its vendor handles (including Tenferro's fixed-size cuTENSOR plan cache);
symtn takes no device lock and sets no process-wide CubeCL configuration.
Work on one engine runs in its stream's enqueue order. A `CudaEngine` is
`Send` but not `Sync`: to submit from several threads, hold one engine per
thread. A buffer belongs to the engine that allocated it, and every device
operation refuses an operand of another engine (checked by allocation domain,
not device ordinal, so a second engine on the same device is refused too)
before any transfer or submission; moving data between engines is an explicit
`to_host` / `to_cuda`. A non-CUDA engine given to a device operation, or a
CUDA engine given to a Host-only entry, is a typed refusal.

Opening a `CudaEngine` is expensive: tenferro-gpu compiles its device kernels
with NVRTC per engine, seconds per kernel module (about 8 s for a whole module
on an A100). Hold one long-lived engine per device and thread rather than
opening one per task. Setting `TENFERRO_CUDA_ARTIFACT_CACHE=<dir>` persists
the compiled kernels in `<dir>`, so later engines and processes read them back
in milliseconds; unset, the compiled kernels live only in memory.

Device execution scratch — a tree transform's uploaded coefficient vectors and
pack/scatter workspaces, a contraction's operand scratch, the region
primitive's scalar templates — lives for one eager operation, `*_into`
included. A prepared plan's workspace keeps its own device buffers and the
scalar templates of its fills, lending the templates to each call, so a warm
replay uploads nothing. A call on another engine drops that state — scratch,
templates and the workspace's retained outputs alike — before any work,
rather than submits or keeps it, so a workspace can follow a caller who
switches engines. A network's steps are eager operations and keep no device
scratch between calls.

## Device validation

Tree-transform admission checks structure, workspace arithmetic and exact
lengths before placement/context/capability checks or device work.
`OperationError::{StructureMismatch, ElementCountMismatch, ElementCountOverflow}`
preserve the corresponding Host error categories. Executor-only failures
(placement, context, aliasing, regions, workspace/readiness) map to
`InvalidArgument`. See
[`opaque_admission.rs`](../symtn-operations/src/opaque_admission.rs).

`CudaSpectrum` records the engine (allocation domain) whose solver produced
it. Every consumer checks it, and any destination, before submission, allocation or empty-input
return. This is a resource guard, not a categorical convention ported from
TensorKit, TensorOperations or QSpace. Spectra can remain device-resident;
collecting sector
spectra with `cuda_download_spectra` uses at most one payload transfer and a
host synchronization. See
[`factorization.rs`](../symtn-dense/src/cuda_adapter/factorization.rs).

## Device test runner

CI has no GPU. On a host with a free device, run
[`scripts/run-device-tests.sh`](../scripts/run-device-tests.sh):

```sh
CUDA_VISIBLE_DEVICES=<free ordinal> scripts/run-device-tests.sh [--two-gpu] [--out DIR]
```

The caller selects unoccupied devices. The script runs workspace tests with
`--locked --no-default-features --features cuda,cpu-faer`, including ignored
device tests serially, then doctests. Its `non_device` list excludes ignored
CPU measurements, exhaustive sweeps and external-tool tests; add new ignored
non-device tests there.

Without `--two-gpu`, the foreign-device destination and spectrum tests are
reported `NOT RUN`. With it, select two free devices. Other skipped failures
must be disclosed, not treated as passing coverage. Logs include the commit,
GPU/driver, individual test results, counts and `RESULT: PASS/FAIL`; failure
returns nonzero.

Validate CUDA execution changes with this script and record the tested
revision and evidence in the CUDA tracking issue. A failing or unexplained `NOT RUN` entry is not passing
evidence. Scaling claims additionally need pinned provider, build, thread
count, workload and raw measurements; wall-clock timing is not a CI gate.
