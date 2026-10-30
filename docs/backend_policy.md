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

| `Runtime::builder()` option | Effect |
| --- | --- |
| `with_dense_executor(...)` | Injected factorization executor |
| `cuda(device)` (feature `cuda`) | Attach the explicit CUDA device |

The CPU provider is a compile-time choice, following Tenferro's mutually
exclusive `native` and `blas` providers: faer (`cpu-faer`, the default) or the
linked BLAS/LAPACK (`cpu-blas` or a `blas-*` feature selecting OpenBLAS, MKL or
Accelerate). There is no runtime provider switch. See
[`symtn/Cargo.toml`](../symtn/Cargo.toml).

On faer builds, the coupled-block contraction of a heterogeneous batch runs on
symtn's own block kernel, one `cpueinsum::GroupedPlan` per batch on the
engine's execution token; it rejects overlapping destinations and
out-of-range jobs (the lowest job index first) before any write. BLAS builds
run the same jobs in order through the BLAS dot.

Eager calls prepare that grouped plan per batch: two top-level allocations
in cpueinsum 0.9, plus any child heap allocations (a Packed `Complex64`
child's boxes and scatters). A one-member Host `ContractPlan`/`ComposePlan`
workspace keeps the plan of each grouped core submission instead, so a warm
one-member call prepares no compatible grouped core plan: it is prepared on
the workspace's first call, reused while the ordered jobs and dtype match (payloads, buffers and view offsets
are checked on every call, not keyed), and released when the workspace runs
another member count or plan. Its charge (16 KiB plus a few words per job
extent for each job, plus the job list) is part of
`ContractWorkspace::retained_bytes`. This holds for the faer provider and the
four payload scalars only; BLAS and CUDA builds, member counts above one,
transform stages and single-job or affine batches prepare as before; the
latter use Tenferro's binary-plan cache, whose misses this guarantee does
not cover.

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

[`Runtime`](../symtn/src/runtime.rs) leases execution contexts and mintable
dense executors from pools. Each idle pool retains at most
`max(available_parallelism, 2)` resources (fallback 4); a resource that unwinds
is not returned. Checkout, return, structural caches and plan caches have
synchronization costs.

The built-in CPU executors share one `SharedCpuContext` per runtime.
`dense_threads` sizes its pool; unset uses available parallelism, 1 runs on
the calling thread, and 0 is an error. Host dense, tree replay, plan compile
and strided work use that pool: dense calls through the engine, and the other
regions by forking the engine's placement token (symtn runs no scheduler of
its own). `recoupling_threads` defaults to 1 and caps
recoupling/compile parallelism within it. CPU runtime construction does not
configure Rayon's global pool or read thread-count environment variables.
BLAS-internal threads remain separate. An injected executor owns its
configuration and uses the runtime state lock for factorization.

CUDA leases hold a process-wide lock per device ordinal plus the runtime's
CUDA-context mutex and Tenferro's resource locks. Enqueue serializes across
runtimes on that device. Host work does not take the CUDA lock. A host sync
under the lease (`to_host`, scalar or spectrum downloads) blocks other
runtimes' submissions to that device too.

Opening the first CUDA context fixes CubeCL's process-global
`streaming.max_streams` to 1. This orders writes to reused destinations and
scratch across leases; an enqueue lock alone cannot order buffers whose bind
preceded a later write. GPU work from different threads shares one stream
per device. This is not a multi-stream or overlap guarantee.

Process-wide effects in
[`single_cubecl_stream`](../symtn-dense/src/cuda_adapter/context.rs):

- It reads CubeCL's file/environment configuration and overrides the stream
  count. If CubeCL already loaded a count other than 1, device construction
  returns `DenseError::Unsupported`.
- The setting survives a failed device open and affects other CubeCL clients,
  including wgpu. A later `CubeClRuntimeConfig::set` panics.
- An ordering residual remains in Tenferro's cached `typed_device_ptr` path:
  an unflushed CubeCL write can be overtaken by a vendor call, even on one
  thread. symtn's empty-contraction scale/fill of an owned destination can
  reach it. See [tenferro-rs#1868](https://github.com/tensor4all/tenferro-rs/issues/1868).

## Device validation

Tree-transform admission checks structure, workspace arithmetic and exact
lengths before placement/context/capability checks or device work.
`OperationError::{StructureMismatch, ElementCountMismatch, ElementCountOverflow}`
preserve the corresponding Host error categories. Executor-only failures
(placement, context, aliasing, regions, workspace/readiness) map to
`InvalidArgument`. See
[`opaque_admission.rs`](../symtn-operations/src/opaque_admission.rs).

`CudaSpectrum` records its solver context's device ordinal. Every consumer
checks it, and any destination, before submission, allocation or empty-input
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
