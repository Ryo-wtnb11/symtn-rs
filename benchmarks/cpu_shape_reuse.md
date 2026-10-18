# CPU network shape-reuse benchmark protocol

Issue: #1141 (parent #1140). Baseline authority: symtn
`4ea8348dad1cf2dbc3907a89c1734aae29881580`.

`microbench_cpu_shape_reuse` measures one deterministic two-step, three-operand
matrix chain. The first contraction destination remains eligible for reuse by
`NetworkExecutionWorkspace`. The same plan and fixtures run under either
`PlannedNetwork::execute` with either a fresh or a reused caller-owned
workspace. Before warmup and timing, every fixture is executed
once in the selected mode and checked by direct column-major indexing of every
reduced block and two nested summations. Timed repetitions are not checked
individually; production regressions own the full replay-history contract.

The fixtures cover a dense-equivalent one-charge U(1) layout, U(1) few-large
and many-small block layouts, and fZ2 x U(1), each with `f64` and genuinely
complex `Complex64` values. Dynamic rows cycle continuously through growing
and shrinking degeneracies or equal-total-dimension sector redistribution,
disappearance and appearance. `cold_structure_input_plan_first_execute`
includes graded-space construction, inputs, label-order planning and first
execution. Runtime and provider construction occur immediately before that
scope and are excluded. Warm rows exclude fixture and plan construction and
change input values on every call.

There is no no-symmetry provider in the public typed `TensorMap` API in this
checkout. The one-charge U(1) case is dense-equivalent (one allowed reduced
block), but it is not presented as a separate no-symmetry implementation. The
lower-level core `TensorMap<Trivial>` and `DenseExecutor` paths are outside this
typed-network workspace comparison.

The global allocator reports allocation/reallocation calls and requested bytes
only inside the target scope. Its always-on live-byte counter reports absolute
requested live bytes at scope entry, peak, output publication and after the
returned output is dropped. These absolute samples are intentionally not
baseline-subtracted: freeing allocations created before the scope therefore
cannot underflow or hide the reported active peak. The allocator adds atomic
instrumentation overhead. Each successful allocation or reallocation samples
the measurement gate once and uses that decision consistently for its peak and
request counters. Elapsed time therefore characterizes this harness rather
than an instrumentation-free latency floor.

Within a batched row, assignment retains the preceding returned output until
the next output has been constructed. The reported live-byte peak can therefore
include two returned outputs; it remains identical between the fresh and reuse
protocols and is not interpreted as private workspace capacity.

The explicit workspace's private retained capacity is not publicly observable.
`live_idle_bytes` is process-wide and may include runtime or provider caches;
it is not attributed to `NetworkExecutionWorkspace`. The macro plan-cache
`retained_workspace_bytes` and `peak_retained_workspace_bytes` describe idle
macro-cache retention, not the active allocator peak of this explicit plan, so
they are not reported. Copy/pack calls and bytes are `NA`: there is no public
direct counter, and allocator traffic is not used as a proxy. The harness
asserts the selected `(A, B)` then `(C, AB)` plan. Its execution enters
`PlannedNetwork::execute` and the multiplicity-free
`HostNetworkModeDispatch::contract_step`; those source facts alone do not prove
whether a backend call packed or copied data.

The benchmark uses the checked-in evidence lock copied to root before the run
(records before #2273, including this lock, use `tenet-<x>` / `TENET_*`; they
are now `symtn-<x>` / `SYMTN_*`. To reproduce the recorded run, check out its
baseline revision):

```sh
cp benchmarks/cpu_shape_reuse.Cargo.lock Cargo.lock
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
CARGO_TARGET_DIR=../tenet/target cargo run --locked --release \
  -p symtn-network --example microbench_cpu_shape_reuse --no-default-features \
  --features cpu-faer > /absolute/path/to/raw.csv
```

`SYMTN_CPU_SHAPE_SAMPLES` controls the number of isolated child processes per
case/workspace pair (default 5). The `median` row is the complete raw row whose
`ns_per_iter` is the sample median; raw rows remain in the same CSV.
`SYMTN_CPU_SHAPE_WORKSPACE_FILTER=fresh` or `reuse` runs only that workspace
mode, which permits valid baseline fresh evidence when reuse correctness fails.

Record the exact symtn SHA, lock SHA-256, command, `rustc -Vv`, CPU/OS details,
and the raw CSV with the reported result. Run the committed
baseline harness and final benchmark commit with the same executable settings.
Do not report a speedup or non-regression unless the two commits differ in
production code relevant to the measured path and the measurements support
that claim.
