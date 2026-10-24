# Operation matrix

`operation_matrix.sh` launches one dependency-free Rust executable. Each row
uses a fresh `Runtime` and constructs its fixture before timing. Owned rows run
one cold call followed by warm-up and repeated calls in that same process.
The wrapper records the OS, architecture, CPU name, Rust compiler, Cargo,
full symtn SHA and dirty state, lock SHA-256, selected package/features/backend,
and the built Tenferro and Racah package versions, Cargo sources, active
features, and available registry VCS revisions before the CSV. TensorKit
records the Julia kernel/machine/CPU report alongside its pinned package and
BLAS authorities.
Here `cold` means that the fresh `Runtime` tree-transform store is empty;
process-global interned structure may already exist.
Destination rows require an exact output from the owned operation; they report
`first_after_setup` and `warm_after_setup`, never a false cold sample.
Validation assertions run after the timer.

The executable covers U1 and SU2 controls for owned and actual
caller-owned-destination forms of `permute`, planar `transpose`,
`repartition`, owned partial trace (direct and lazy-adjoint input), and
arbitrary-axis `contract`. Owned `scale`/`add` and scalar `norm`/`inner` rows
cover both direct tensors and pre-built lazy adjoints.
The contraction rows distinguish canonical input order, contracted-input
swap, and contracted-input plus output swap. An owned `compose` row checks
that its result equals the canonical contraction before reporting it.
With the default `cgc-gen` feature it also runs exact checked-Generic
SU3 `[1,1]` and SU4 `[1,0,1]` fixtures for public owned transforms, reductions,
compose, and contract, plus compact QR for those fixtures and a one-sector SU3
vacuum control. Compact QR coverage is limited to these checked Lie-group owned rows;
no trivial or multiplicity-free row is implied. Transform/compose/contract
checks are public-call self-consistency checks comparing provider authority, spaces, block layout,
full fusion-tree keys, and payload; scale/add also check elementary payload
laws, norm/inner check scalar identities, and compact QR reconstructs the input
outside timing. On this base those fixtures
print an explicit trace exclusion because this exact `LieGroupFusionRule` lacks the
`SectorCodec` bound required by checked trace dispatch, and a destination
exclusion because it lacks the required multiplicity-free dispatch bounds.
The checked-Generic API exists; these operations are neither emulated nor
replaced for the exact SU(N) fixtures.
The `qr_compact_generic_layout` selector adds one lower-level public owned-QR
control for an infallible Generic provider. It compares the same two-sector
rank-(1,1) tensor in two padded expert layouts: canonical key order and the
same blocks in reverse order. Both inputs therefore use staged matrix assembly;
only the reversed order requires output scatter. Its two sector matrices have
shapes `d x 2d` and `2d x d`, where `d` is `OP_MATRIX_DEGENERACY`. Fixture
construction, keyed source-coordinate checks, factor-coordinate comparison,
and direct `Q * R` reconstruction are outside the timers. Because that
preflight calls QR, its first timed phase is labeled `first_after_setup`; the
measurement is a matched production-build layout control, not a cold-start
claim.
Runtime tree-transform counters are reported as cold and warm deltas. The same
snapshots report process-global fusion-layout and complete-HomSpace cache
deltas, with charged bytes before and after each phase. The fusion-layout cache
does not expose a hit counter, so only its available miss, eviction, bypass,
entry, and charged-byte fields are printed.

The destination rows call the stable public `permute_into`,
`transpose_into`, `repartition_into`, and `contract_into` APIs with
`beta = 0` (they were the `*_overwrite_into` forms before #1550). Exact-layout admission is attached to the
Runtime tree-transform entry, but it has no public activity counter, so the
`exact_layout_admission` column is literal `NA`. The 38-column CSV schema omits
the old erased-only destination preparation/comparison fields. Caller-thread
Rust allocation calls and requested bytes are measured directly for each
phase; divide warm totals by `iterations` for a per-call value. They exclude
worker-thread and native-BLAS allocation and do not represent frees, live
bytes, or peak memory. Scratch, provider queries, dense kernel calls, and
transfers remain `NA` rather than being inferred from elapsed time.

```sh
OP_MATRIX_MIN_MS=20 benchmarks/operation_matrix.sh
```

Install the reviewed benchmark lock explicitly in a fresh checkout, then run
the wrapper:

```sh
cp benchmarks/cpu_shape_reuse.Cargo.lock Cargo.lock
OP_MATRIX_MIN_MS=20 benchmarks/operation_matrix.sh
```

The runner requires that root lock. Before timing, it builds the exact selected
release example and takes Tenferro features from Cargo's compiler-artifact
records. Metadata supplies package source and manifest provenance. The build
and all three samples use `--locked --offline`; fetch the locked dependencies
before disconnecting if they are not already in the local Cargo cache.

`OP_MATRIX_DEGENERACY` selects the common per-sector degeneracy (default 8).
`OP_MATRIX_GEMM_BACKEND` is `faer` by default. A macOS BLAS control uses the
same Apple Accelerate provider as the pinned TensorKit environment.
`OP_MATRIX_OPERATION` can select one exact operation name for focused profiling.
`OP_MATRIX_FORM` similarly selects `owned` or `destination`.
`OP_MATRIX_PROFILE_PAUSE_MS` pauses after that row's warm phase so an external
profiler can inspect the live process; it is outside every reported timer.
`OP_MATRIX_CACHE=disabled` sets the process-global completed-transformer and
composed-coefficient budgets to zero (`symtn::cache::configure_budgets`), so
every transformer and coefficient-group build is offered and counted as a
rejection: the disabled-control rows are not fully cold. The setting is
process-global, not per Runtime. The default is `enabled`.
The executable emits raw per-process samples. `operation_matrix.sh` launches
three fresh processes, preserves every raw row as a `# raw_sample` record, and
then appends the complete median-time row for each CSV key; its warm value is
the mean within that selected child batch, not a cross-child mean.

```sh
OP_MATRIX_OPERATION=contract_input_swap \
OP_MATRIX_FORM=destination \
OP_MATRIX_PROFILE_PAUSE_MS=30000 \
OP_MATRIX_MIN_MS=0 benchmarks/operation_matrix.sh
```

The matched Generic compact-QR layout control is:

```sh
OP_MATRIX_OPERATION=qr_compact_generic_layout \
OP_MATRIX_FORM=owned \
OP_MATRIX_MIN_MS=100 \
benchmarks/operation_matrix.sh
```

## Oriented uniform-run diagnostic

`OP_MATRIX_OPERATION=oriented_uniform_run` is an explicit-only #1179
benchmark. It has two scopes. `DenseAdapter` calls the public
`DenseExecutor::matmul_batch_axpby_with_ops_into` seam into caller-owned
destination storage; the destination remains allocated across calls. `U1Public`
calls public owned `compose` and `contract`; every first, warm, and shape-cycle
call copies the first payload value and length, observes them through
`black_box`, and drops the complete owned output before its measured span ends.
The copied marker is compared with the precomputed oracle after timing.

All jobs, buffers, run partitions, tensors, expected payloads, and changing-shape
fixtures are constructed before timing. A separate executor or `Runtime` runs a
full literal nested-sum preflight first and is dropped before the measured
fixture is built. Measured fixtures retain only copied expected first-value and
length markers; their full expected payloads are dropped before timing.
Adapter preflight uses nonzero complex alpha and beta and
checks gaps as well as active destinations. Timed calls use beta zero. Public
U1 expected tensors are built directly from parent coordinates and distinct
coupled-sector labels; they do not use the measured compose, contract, or
lazy-adjoint materialization as their oracle.

The adapter sweep contains:

- `many_small`: `L=32`, `(m,k,n)=(4,3,5)`, with affine gaps;
- `few_large`: `L=4`, `(64,48,56)`, with bases `(3,5,7)` and strides
  `(3080,2696,3592)`;
- `minimum_run`: `L=2`, `(4,3,5)`, for the structural oriented-run minimum;
- `singleton`: `L=1`, `(64,48,56)`;
- `heterogeneous`: eight alternating `(4,3,5)` and `(5,4,3)` jobs whose
  maximal runs all have length one.

The two uniform workloads measure f64 `II`/`TI` and complex64 `II`/`AI`/`AA`.
The minimum-run control measures f64 `TI` and complex64 `AI`; singleton and
heterogeneous controls measure complex64 `AI`. Shape-cycle rows retain one
executor while cycling the fully preconstructed geometries
`[(4,3,5),(5,4,6),(3,6,4)]` at `L=32` or
`[(64,48,56),(56,40,64),(72,56,48)]` at `L=4`. All three geometries are
warmed once before timing, and each timing check completes a whole three-case
cycle. This does not claim that the backend retains three compiled plans.

The public U1 layer uses 32 sectors of degeneracy 4 (`many_small`) or four
sectors of degeneracy 32 (`few_large`). For f64 and genuinely complex
complex64 it measures direct compose, lazy-lhs-adjoint compose, and the same
lazy lhs through `contract` with `ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] }`. Shape-cycle compose rows
use degeneracies `4,5,3` or `32,33,31` on one `Runtime`. These public rows make
no backend submission-count claim; correctness seam counts belong to #1179's
tests.

Adapter rows preserve the 38-column CSV schema but print `NA` for Runtime,
tree/cache, provider, exact-admission, scratch, GEMM-call, and transfer fields,
because that public adapter has no such counters. Caller-thread Rust allocation
calls/bytes and elapsed time are measured. Native/provider allocation, frees,
live or peak bytes, copy/pack traffic, and isolated kernel time remain
unavailable. Reduced backend submissions therefore do not establish an elapsed
improvement.

Use the reviewed lock and run three fresh child processes at each position in
the unconditional `A -> B -> B -> A` order. Keep all raw samples; do not choose
favorable repeats:

```sh
cp benchmarks/cpu_shape_reuse.Cargo.lock Cargo.lock
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
OP_MATRIX_OPERATION=oriented_uniform_run \
OP_MATRIX_MIN_MS=200 \
OP_MATRIX_GEMM_BACKEND=faer \
OP_MATRIX_CARGO_FEATURES=cpu-faer,cgc-gen \
CARGO_TARGET_DIR=/Users/ryowatanabe/Research/codes/MyTensorNetworks/libraries/tenet/target \
benchmarks/operation_matrix.sh
```

Execute that command once in baseline A, twice in candidate B, then once again
in baseline A, with separate output files for all four wrapper invocations.

The checked-Generic compact input-lowering group runs public compact QR, SVD,
and LQ on the same source geometry. It covers `f64` and genuinely complex
`Complex64` payloads, and compares a canonical contiguous layout with a padded,
reversed-block fallback. `few-large` uses two sectors with alternating matrix
shapes `d x 2d` and `2d x d`; `many-small` uses labels 0 through 15 with the
same alternating geometry and `max(1,d/16)` in place of `d`. Fixture
construction, provider binding, literal
reconstruction (`Q*R`, `U*S*Vh`, `L*Q`), QR/LQ/SVD orthogonality, singular-value
ordering, and source-immutability checks are outside timing. Every invocation
owns fresh returned factors; warm-loop temporaries are dropped per iteration,
while the first returned value is retained until its sample has been printed.
That ownership and teardown scope is identical in baseline and candidate. The
first phase is therefore `first_after_setup`, not a process-cold claim.

```sh
OP_MATRIX_OPERATION=checked_compact_input \
OP_MATRIX_FORM=owned \
OP_MATRIX_DEGENERACY=32 \
OP_MATRIX_MIN_MS=100 \
benchmarks/operation_matrix.sh
```

`eig_source_geometry` is an explicit-only rank-2 measurement group for MF U(1)
and checked-Generic full EIG. It covers `f64` and genuinely complex
`Complex64`, few-large `G=2,d=D`, and many-small
`G=16,d=max(1,D/16)`. Checked inputs use canonical and padded reversed layouts;
both layouts exercise the EIG geometry path. Compact QR over separate inputs
with identical geometry and scalar types is the unchanged control.

Fixed and alternating `d`/`d+1` inputs are preconstructed. Full preflight
checks the known distinct spectrum, finite nonzero eigenvector columns,
scale-invariant `A V = V Lambda`, provider/block identity, QR reconstruction,
and source preservation outside timing. Every timed operation drops its owned
result inside the closure, including the first row. Because preflight may warm
Runtime and process-global metadata, rows say `first_after_preflight`; they do
not claim cold cache admission. Caller-thread Rust allocation calls and
requested bytes are available. Native and worker allocations, frees, live or
peak bytes, exact source-copy bytes, and backend materialization are not.
MF rows observe the Runtime that executes EIG. Checked EIG and QR call
`DefaultDenseExecutor` directly, so their printed Runtime cache counters belong
to a separate reporting Runtime and are `NA` for execution interpretation.

```sh
OP_MATRIX_OPERATION=eig_source_geometry \
OP_MATRIX_FORM=owned \
OP_MATRIX_DEGENERACY=32 \
OP_MATRIX_MIN_MS=100 \
OP_MATRIX_CARGO_FEATURES=cpu-faer,cgc-gen \
benchmarks/operation_matrix.sh
```

`eig_output_placement` is an explicit-only multi-tree measurement group for
one-sided factor publication. Checked-Generic full EIG is the affected rank-4
route. Each side has two sector legs under the XOR fixture rule: few-large uses
`G=2,T=2,d=D`, while many-small uses
`G=16,T=16,d=max(1,D/16)`. The corresponding canonical inputs have 8 and 4,096
source blocks. Reversed blocks with one-element gaps exercise the existing
fallback. Checked compact QR over the same geometry is the unchanged paired
publication control. Existing rank-2 MF EIG rows are included as affected
`T=1` evidence.

Fixed and alternating `d`/`d+1` inputs are preconstructed. Outside timing,
preflight verifies every literal full key and source value, the known distinct
spectrum, finite nonzero eigenvector columns, scale-invariant
`A V = V Lambda`, QR reconstruction and orthogonality, provider identity and
source preservation. Setup rows separately include construction, checked
binding and literal fill. Every operation row black-boxes and drops its
complete owned result inside the timed closure. The first row is
`first_after_preflight`, not cold admission.

Caller-thread Rust allocation calls and requested bytes are available. Native
and worker allocations, frees, live or peak bytes, exact placement comparisons,
exact copied bytes and isolated solver time are not. Checked EIG and QR use
`DefaultDenseExecutor` directly, so printed Runtime cache counters are not
execution-owned and are `NA` for interpretation. The group supplies no timing
evidence for full SVD, EIGH or null operations and no universal speed claim.

```sh
OP_MATRIX_OPERATION=eig_output_placement \
OP_MATRIX_FORM=owned \
OP_MATRIX_DEGENERACY=32 \
OP_MATRIX_MIN_MS=100 \
OP_MATRIX_CARGO_FEATURES=cpu-faer,cgc-gen \
benchmarks/operation_matrix.sh
```

`full_qr_lowering` is an explicit-only diagnostic for the original-input full
QR/LQ lowering. It fixes global `D=32`: few-large uses `G=2,d=32`, and
many-small uses `G=16,d=2`. The sweep covers homogeneous square `d x d`, wide
`d x 2d`, and tall `2d x d` sectors, `f64` and genuinely complex
`Complex64`, and canonical and padded reversed layouts. QR's direct-input
cases are square/wide and its unchanged completion control is tall; LQ's
direct-input cases are square/tall and its completion control is wide.

Fixture construction has separate `fixture_first` and `fixture_repeat` rows.
Factor rows measure fixed preconstructed inputs and alternating preconstructed
`d`/`d+1` inputs, with at least 100 ms per repeated phase. Independent
preflight checks exact factor shapes, literal reconstruction, square-Q
unitarity, nonnegative real diagonal gauge, provider identity, and source
immutability. Returned owned factors are passed through `black_box` and dropped
inside every timed call. The fixed D32 sweep is a bounded comparison; it does
not establish general rank or symmetry speed. Pure solver time, peak memory,
and retained idle storage remain unavailable in this runner.

These checked rows construct `DefaultDenseExecutor` directly. The prescribed
comparison builds only `cpu-faer,cgc-gen` and uses the wrapper's
`RAYON_NUM_THREADS=1` and BLAS thread limits. `OP_MATRIX_GEMM_BACKEND` configures
tensor-operation GEMM and does not select the factorization provider.

```sh
OP_MATRIX_OPERATION=full_qr_lowering \
OP_MATRIX_FORM=owned \
OP_MATRIX_DEGENERACY=32 \
OP_MATRIX_MIN_MS=100 \
OP_MATRIX_CARGO_FEATURES=cpu-faer,cgc-gen \
benchmarks/operation_matrix.sh
```

The grouped output separates three scopes. `checked_compact_input_fixture`
times construction and payload initialization of both canonical and fallback
fixtures (`fixture_first`/`fixture_repeat`). The ordinary QR/SVD/LQ rows keep one
fixed preconstructed input. Their `_shape_alternating` controls alternate
preconstructed `d` and `d+1` inputs with one dense executor per numerical
family; fixture construction is excluded, and literal preflight has already
initialized both inputs' region metadata. Fresh factor results are dropped
inside every alternating timed closure. These labels describe setup and shape
reuse only; none claims an isolated process-cold cache.

The benchmark-local XOR provider has a fixed 16-label domain; labels 0 and 1
retain the previous fixture behavior. The many-small geometry is `G=16`, with
16 row trees and 16 column trees (`T=32` total side trees under the issue's
convention), and 16 source blocks. It is only a finite cardinality/dataflow
control and makes no physical non-Abelian claim. Existing generated SU(N) QR rows
supply a physical many-tree supplement. Rank-4, multiplicity, and broader G/T
correctness sweeps, plus
direct borrow/pack probes, belong to tests because the public benchmark API
exposes no input-copy counter. Reported allocation totals remain caller-thread
requested calls/bytes; peak, live, native, and worker-thread memory are
unavailable.

The Apple Accelerate control is:

```sh
OP_MATRIX_GEMM_BACKEND=blas \
OP_MATRIX_CARGO_FEATURES=blas-accelerate \
benchmarks/operation_matrix.sh

julia --project=benchmarks/tensorkit_benchmark \
    benchmarks/tensorkit_microbench.jl 8 300
```

The TensorKit script records its exact revision, Julia and AppleAccelerate
versions, BLAS configuration, first-call timing, warm timing, and Julia's
per-call allocated bytes. It uses the same rank-3 fixture and axis placement
for symtn's `permute`, planar `transpose`, and `repartition` rows, and the same
rank-4 fixture and middle trace pair for the direct/lazy-adjoint trace rows.
The same rank-4 values feed matched direct/lazy-adjoint `scale`, `add`, `norm`,
and `inner` rows; both implementations validate the same norm identities
outside the timer.
It also reports TensorKit's real caller-destination trace form;
the canonical symtn typed facade has no matching destination method. Its
first-call row can include JIT compilation and may observe
process-global TensorKit caches warmed by earlier rows; it is not directly
comparable to symtn's fresh-`Runtime` cold row. Only matching warm rows under
the recorded one-thread BLAS configuration are timing controls.

The remaining #9 rows (ordered contract, compact SVD, compact diagonal,
the multiplicity-free and other compact QR forms, and other lazy-adjoint consumers) are
not substituted with other operations. Add each only with its real public form
and available counters. This diagnostic harness remains outside required CI;
semantic coverage belongs in the existing user API tests.
