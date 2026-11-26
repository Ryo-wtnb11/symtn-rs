# Backend selection policy

symtn owns categorical semantics and layout decisions. Tenferro owns dense
kernels, factorization providers and execution resources; Strided owns
applicable strided movement. Reuse those boundaries before adding a trait,
selection knob or retained execution state. Preserve the
[structured-complexity contract](complexity_parity_policy.md).

## Selection

`symtn-dense` owns `DenseExecutor`; `symtn-operations` owns
`TreeTransformBackend`; `symtn-tensors` owns `TensorContractBackend`.
Operators describe spaces, axes, conjugation and output order; backend adapters
choose dense kernels. Do not put raw BLAS calls or threshold-selected numerical
kernels in operator code.

The CPU provider is Tenferro's faer-backed stack (`cpu-faer`); Tenferro
links no vendor BLAS/LAPACK provider. It is not a runtime or per-operation
switch; see [`symtn/Cargo.toml`](../symtn/Cargo.toml).
With feature `cuda`, a device operation runs on the caller's `CudaEngine`
(`CudaEngine::open(CudaDeviceId::from_ordinal(n))`), passed like any other
engine; a `Runtime` is not tied to a device.

On the native provider an initialized Host GEMM batch that is neither one job
nor one affine run, with or without transposed or adjoint operands, runs on symtn's
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
the plan's own retained bytes as tcontract reports them, plus its key and
the entry overhead; with a budget of zero,
or for an entry the budget rejects, every call prepares its plans as before
with the same results. For initialized destinations this holds for the faer
provider and the four payload scalars only; CUDA builds prepare as
before, and initialized single-job or affine batches use Tenferro's binary-plan
cache, outside this cache.

The dense executor also has a hidden fresh entry that writes a batch into
uninitialized destination memory, for owned Host outputs. Every batch on the
faer provider, including one job and gapless affine runs, uses the block
kernel's geometry-plan cache and tcontract's `execute_uninit_slices` entry.
Identical geometries share one plan per scalar type. A single job stages its
plan handle and destination on the stack; larger batches reuse the executor's
staging. The same lane driver distributes independent blocks across lanes,
with each lane's share available for the provider's inner threads. No symtn
caller-side output fill precedes the GEMM: the computing workers first-write
their destinations. tcontract's faer fresh entry uses an overwrite-only
uninitialized kernel, which skips its initialization pass on x86_64 with AVX2
or AVX-512. Other architectures may initialize inside the provider. The trait's
default implementation for custom executors still zero-fills each block through
the carrier before its initialized overwrite call.

An owned eager Host contraction (`contract`, `planar_contract`, `compose`,
in both modes) allocates its output without zeroing it when the route's core
GEMMs write the destination (the canonical core, or a `DynamicTree` route
without an output transform). Each GEMM call receives only its own
destination ranges through the fresh entry above; the blocks no GEMM writes
(inactive blocks and padding) are written `+0.0` by symtn. The output is published only after every element has been written. A route
with an output transform (`copyC`, a `DynamicTree` output) runs its core into
scratch; the transform then writes every destination element once into an
unzeroed payload, inactive blocks as `+0.0`, when the destination proves
exact overwrite coverage: its blocks, inactive ones included, reach every
payload element exactly once (canonical layouts, including coupled sectors
whose several codomain trees interleave their blocks' columns), and the
coupled-sector regions tile the payload. Without that proof (for example a
layout with padding) the transform overwrites an allocator-zeroed payload.

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

The prepared `EighFullPlan` solves each coupled sector of all members with
one Tenferro `eigh` call. Tenferro launches cuSOLVER `XsyevBatched` for it
only on cuSOLVER 11.7.1 or newer; on an older library it loops one `syevd`
per member, and exposes no way to require or detect the batched routine.
A real batch accepts that fallback. A complex batch of more than one member
is refused with `DenseError::Unsupported` before device work; a single
complex member takes the per-matrix solver, as eager `eigh_full` does.

## Parallel execution

Every numerical operation takes the caller's execution context
(`cx: &mut (impl ExecContext + ?Sized)`), which is the caller's engine
(`&mut CpuEngine`, `&mut dyn Engine`). A `CpuEngine::serial()` runs on the calling
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
provider's grouped block GEMMs (recoupling, block-wise contraction) and its
block factorizations over a tensor's sector blocks (compact and full SVD,
values-only SVD, the pseudo-inverse's SVD, compact QR and LQ, full QR and LQ, the QR
null bases of `left_null` and `right_null`, Hermitian and general
eigendecomposition full and values-only, the polar decompositions, the
Hermitian `exp` and the inverse) are placed by one routine and run by one lane runner (the factorizations inside the
engine's guarded linalg entry, the GEMMs inside its contraction entry).
Each block is one provider job whose cost is the provider's work estimate:
tcontract's per-flop rate over the block's `2 m k n` flops (four times that
for complex scalars) for a GEMM, tlinalg's work estimate under the engine's
work model for a factorization (SVD `m n min(m, n)`, QR `m n k - k^3/3` with
`k = min(m, n)`, LQ that of its transpose's QR, a tall full QR or null basis
tlinalg's Householder factor plus its reflector application, EIGH and EIG,
full and values-only, `n^3`; a polar decomposition or Hermitian `exp` adds its products;
an inverse its LU factorization and solve against `I_n`, `n^3/3 + n^3`).
The call takes its thread count from the engine's cost rule over the batch's
estimated serial time, and its lane count is at most the engine's batch
decision, so the work model's fan-out limits are never exceeded. Each lane
reserves a nonzero share of those threads, the shares summing to at most
the thread count, and each block runs on a context of exactly its lane's
share; a thread no lane reserves stays idle for the call. A batch of a few
large blocks gives each lane several threads, and a batch of many small
blocks runs one thread per lane. The baseline placement is equal shares over
`L0 = min(threads, blocks, decision lanes)` lanes, blocks largest first on
the lane of least accumulated serial time; a single lane takes every block
before any ordering. When every block's provider width is modeled (every
GEMM; a factorization whose every input is one matrix), the placement also scores
an equal-share placement for every lane count from `L0` down to one and a
two large-block reservations under one model, and keeps the lowest score;
ties keep the baseline, then the equal shares, then the shared reservation.
A reservation finds the least target time at which the blocks above it,
each on the narrowest lane that brings it under the target, fit in the
thread count; it gives each such block that lane, makes the remaining
threads one-thread lanes, places the other blocks largest first on the lane
where they finish earliest, then widens the reserved lanes with the threads
of unused one-thread lanes while some stage's provider width still grows.
The shared reservation may place the other blocks on a reserved lane; the
exclusive one keeps each reserved block alone on its lane, and its target
must also leave one-thread lanes enough for the other blocks' serial time.
Each is discarded when more than `L0` of its lanes hold blocks, and scored
as placed, since a block below the target can still finish above it. The
model charges a block its serial time where the provider would run it on
one thread in that lane, otherwise its time over the provider's granted
width plus the engine's dispatch cost at that width. A composite block (a
polar decomposition or a Hermitian `exp`) is charged per sequential stage:
its factorization (SVD or eigh) and each product, each its own serial time
(the factorization's under the engine's work model, a product's at
tcontract's GEMM rate) over its own provider's width plus its own dispatch
cost, summed; its serial time is the stages' sum. On a one-thread context
the driver prices every block by its kind's cost alone, since the placement
is then one sequential lane. The granted width is an estimate composed from the provider's
public decisions: for a GEMM the plan's own two decisions (its `width_for`
narrowing, then a Faer plan's one-item batch decision with the token's team
capability, or a Packed plan's balanced-lanes decision, an optimistic upper
bound on its grid); for a factorization tlinalg's decision for one item
under `Auto`, the engine's cost rule over the total work's time capped at
the width the provider's payback work (`estimate_payback_work`) pays for,
and one thread on a token without a team. The model has no sublinear
speedup, so the choice is a scheduling estimate, not a promise of a faster
run. The search queries the provider width of each group of blocks with the
same estimated time and the same grant inputs (for GEMM, the plan's
algorithm and the work; for a factorization, its total and payback work;
for a composite block, its op, scalar, extents and whether its plans
resolved, which fix every stage) once, at the full thread count; only the groups the provider widens there
are queried again, once per distinct share width during the sweep, since a
block the provider keeps serial at the full count stays serial at every
narrower share; a lane count whose proved lower bound on its score is not
below the best score so far is not placed. A
batch holding an input whose batch product is not one (including zero) keeps
the baseline as well: tlinalg runs such an input through its multi-item
route, which its public queries do not expose. Placement is integer
arithmetic. Each searched placement is kept in the process-global structure
cache `StructureCacheKind::LaneAllocations`, keyed by every input it is a
function of (the kernel's kind, GEMM or factorization, which fixes the
grant; the engine's budget, work model, team and dispatch cost at every
width, the mean work, and each block's serial time and grant inputs), so a
batch already searched, on any executor including an eager call's fresh one,
rebuilds its lanes without a query or a sort. Each entry is charged its
inline key and entry, the entry overhead, and its per-block, per-width and
per-lane slices; an entry above the largest admissible size is refused
before its key is copied, and with a budget of zero every call searches,
with the same placement. A forced batch policy (`Inner`, `Outer`,
`Split`, `Sequential`) reserves its batch decision's threads, the blocks
placed by the same largest-first baseline over its lanes and each block run
on its lane's whole share (`Inner`: every block on the whole engine). Prepared
`ContractPlan`/`ComposePlan` workspaces of any member count run their
grouped core batches the same way. Every block runs up to the first one refused before the engine is entered
(an unsupported dtype or a malformed view), and keeps its own outcome; the
outcomes, then that refusal, reach the operation in block order, where the
operation applies its error precedence, and nothing is returned on error.
Compact SVD checks each delivered sector's output, gauge and spectrum before
staging the next, so the lowest-indexed failing sector's error is reported;
the operation's own linear-algebra scope's refusal takes precedence over that error.
Full EIGH, polar and exp also report the lowest failing sector.
The solver-first destination routes (full SVD, QR/LQ, null bases and inverse)
instead let a later solver failure override an earlier malformed output.
Two costs differ from a block-by-block loop
and are accepted: blocks after a failing one are still solved, and a
provider panic in such a later block unwinds and leaves the engine broken
where the loop would have returned the earlier error. Compact QR, compact
LQ, full QR and LQ, the `left_null` and `right_null` bases, compact and
full SVD, full EIGH, the left and right polar decompositions, the Hermitian `exp`
and the inverse run through
`DenseExecutor::factorize_batch_into`: the
operation allocates each output once (the final packed factor for a
multiplicity-free or Checked Generic canonical output; a Checked Generic
publication whose factor keys are not in source order instead scatters one
packed transient per call by tree identity, about twice the factor payload on
that route) and one packed real spectrum, and
each block writes its factors straight into its slice through tlinalg's
initialized-destination entries; SVD and EIGH write their real spectra
straight into the destination, with no staging. Each lane allocates one
provider workspace and one typed scratch (the provider-shaped factors of an
adjoint placement or of LQ) per call, sized from that lane's blocks at its
own width before its first block, and drops them when the lane finishes;
nothing is kept across calls. Compact LQ of `X` reads
each block's transpose in place (a strided view, no adjoint copy), solves
`X^T = Q' R'` into the lane scratch and writes `L = R'^T`, `Q = Q'^T` into
the destination in one plain transpose pass; LQ of a lazy adjoint `A^H`
solves the parent blocks `A` by QR placed as the adjoint problem
(`l = R^H`, `q = Q^H`). The positive-diagonal and phase gauges run in place
on the delivered destination, and each sector's public `f64` spectrum is
still its own allocation. The positive-diagonal LQ is unique only when a
sector's leading `k x k` triangular factor is nonsingular; otherwise the
native transpose route, the default's adjoint route and the lazy parent's
QR return valid but possibly different factors. Native full QR of an
`m x n` block with `m <= n` is its compact QR; with `m > n`, as the
`left_null` basis, the input is copied once into the destination `R` (the
null basis: into the lane scratch), factored in place by tlinalg's compact
Householder kernel, and its reflectors applied straight onto `I_m` (onto
`[0; I_{m-n}]`, so only the final `m x (m - n)` basis is written):
`O(m n^2 + m^2 n)` against the `O(m^2 (n + m))` economy QR of `[X | I]`
that every other executor still runs through its
own `qr_into` (as does the full SVD's `[U1 | I]` completion on an
executor without a native full SVD). Full LQ and the `right_null` basis are the same full QR and null
basis batches on each block's transpose `X^T`, read in place (a strided
view, no adjoint copy): `X^T = Q' R'` gives `L = R'^T`, `Q = Q'^T`, and the
null basis `N'` of `X^T` gives `N = N'^T` (`X N^H = (N'^H X^T)^T = 0`), so
each runs at its provider's full QR cost above. A lazy adjoint `A^H`'s full
LQ reads its parent blocks `A` untransposed and takes `L = R'^H`,
`Q = Q'^H`. The positive-diagonal gauge runs on `(Q', R')`; `Q'` is then
transposed in place in its final slice, while `R'` and `N'` are solved into
one packed scratch of the output's size per call and transposed once into
the final buffer after the batch. Each tall block allocates the compact
factor's own workspace once (tlinalg has no initialized-coefficient
destination form; the lane lends the reused coefficient `Vec`). Columns past
`min(m, n)` of the full `Q`, and the null basis, are any orthonormal basis
of the complement and may differ by provider. Native full SVD of an `m x n`
block is tlinalg's `svd_into` with its full flag straight into the
destination `U` (`m x m`), `S` and `Vh` (`n x n`), as tenferro's
`svd_full` runs it with no post-step (a zero extent gives identity `U` and
`Vh`); a lazy adjoint's parent blocks are solved into the lane scratch
(`m^2 + n^2` elements of `T` per item) and placed as the adjoint problem
(`u = Vh^H`, `vh = U^H`), so no adjoint input is staged. The phase gauge
then runs in place on the delivered destination. Other executors solve each
sector with `svd_full_owned` of a compact copy when they support it, and
otherwise solve the compact SVD with their `svd_into` and complete its `U`
and `V` from `[U1 | I]` with `qr_into`; the columns of `U` and rows of `Vh` past `min(m, n)` may differ
by provider. A native polar job of an
`m x n` block (MatrixAlgebraKit's `PolarViaSVD`, `k = min(m, n)`) solves the
compact SVD into its lane's typed scratch (`U` and `Vh`, `(m + n) k`
elements of `T` per item) and a separate real spectrum scratch (`S`, `k`
elements of `Real(T)` per item), both allocated per call and dropped with the
lane, so a polar lane owns two scratch buffers; it writes `W = U Vh` into its
destination, scales `Vh` (left) or `U` (right) by `sqrt(S)` in that
scratch, writes `P = X^H X` (or `X X^H`) and makes it exactly Hermitian; both
products run on the lane's context with geometry plans resolved before the
lanes. The `P` product is a full GEMM, twice the multiply-adds of the
`herk`/`syrk` MatrixAlgebraKit calls. `P` is unique; `W` is unique only for a
full-rank block and otherwise may differ by provider or lane width. A native
Hermitian `exp` job of an `n x n` block (MatrixAlgebraKit's
`MatrixFunctionViaEigh`) solves the eigendecomposition into its lane's typed
scratch (`V`, `n^2` elements of `T` per item) and a separate real spectrum
scratch (`n` elements of `Real(T)` per item), both allocated per call and
dropped with the lane (two scratch owners per lane); it checks the
eigenvalues are finite (a nonfinite one is that block's error and skips the
rest), scales `V`'s columns by `exp(lambda / 2)` in that scratch, writes
`exp = Y Y^H` of the scaled `Y` into its destination and makes it exactly
Hermitian (`copytri`); the product runs on the lane's context with its plan
resolved before the lanes. It is a full GEMM, twice the multiply-adds of the
`syrk` MatrixAlgebraKit's real branch calls (its complex branch is a general
GEMM too). The half exponent keeps a representable result finite where
`exp(lambda)` alone overflows. A native inverse job of an `n x n` block
writes `I_n` into its destination and solves `X Y = I` there in place with
tlinalg's partial-pivot LU `solve_into` (no separate right-hand side, no
typed scratch; only the provider workspace per lane), the call tenferro's
per-matrix `solve_into` makes from a copied right-hand side, with the same
singularity check (an exactly zero pivot) and errors. This is TensorKit's
`inv` per block (`one!` on the output block, then `ldiv!(lu(b), ·)`, LAPACK
`getrf` + `getrs`), `O(n^3)`.
The left solve `A \ B` submits every nonempty coupled sector as one
`DenseExecutor::solve_batch_into` batch. On the native provider each system
in the valid compact rank-two subset of tenferro's direct route (one
floating dtype, `A` `[n, n]`, `B` and `X` `[n, k]` with `n, k > 0`, `X`
strided `[1, n]`; every sector the solve builds) runs on the lanes as
tlinalg's partial-pivot LU `solve_into` with `B` passed as its right-hand
side, which the provider copies into `X` only after that system's
factorization and singularity check succeed, so the copy runs on the lane
and a singular system's `X` is left as it was; one provider workspace per
lane, no typed scratch. `A` and `B` are read through their own strides and
offsets. Every other system (a vector right-hand side, a padded `X`, other
ranks, zero extents, malformed or mixed-dtype inputs) is solved by the
executor's per-matrix `solve_into` on the caller, never inside a lane, so
its result is that entry's. The error is the lowest original index's: the
per-matrix systems below the first lane system run first and stop at their
first error; a wrong engine, a refused engine entry or a refused lane fork
is the outcome of that first lane system; the per-matrix systems after it
run only below the lowest failing lane system. A provider panic in a lane
system unwinds after the lanes join and leaves the engine broken, even
where a lower lane system already failed. This is TensorKit's `\` per
block (`copy!(b, block(t1, c) \ block(t2, c))`, LU and a solve, `O(n^3 +
n^2 k)`), with the solution written straight into the output block.
Values-only SVD and EIGH
and the general eigendecomposition (values only or with vectors) keep one
provider workspace and fresh outputs per block (tlinalg has no initialized
destination form for them); the full one normalizes each block's
eigenvector columns in place with Tenferro's exported
`normalize_eig_vectors`, the gauge and failure its per-matrix `eig` reports. The
pseudo-inverse's SVD keeps the owned batch, one provider workspace and
fresh factors per block, which it holds until its global cutoff is known and
then reads in place in its reconstruction GEMMs: a destination needs the
whole transient zero-initialized first, a serial pass the owned outputs do
not pay and that a caller which never scatters or publishes the factors
does not recover (TensorKit's `pinv` also solves each block into its own
temporaries). Every other executor
reaches `factorize_batch_into` through its default: the owned
`factorize_batch` with each delivered output copied into its destination
(for LQ, each block's adjoint written into one scratch sized for the
largest block and factored with QR; for full SVD, the per-matrix route
above). The polar decompositions have no owned
form (`factorize_batch` refuses them); the default instead runs, per block,
one `svd` and two `dot_general_into` calls writing the destinations, the
`sqrt(S)` scaling applied in place to the owned factor, and checks the block
before the next one is solved; the Hermitian `exp` likewise runs one `eigh`,
the eigenvalue check and one `dot_general_into` per block, and the inverse
one `solve_into` per block against one identity per scalar sized for the
largest block, and `solve_batch_into` one `solve_into` per system in order,
stopping at the first error.
The other dense problems still run block after block on the whole engine.
Callers can still give each GEMM the whole pool with
`CpuEngine::new(pool).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Inner))`
(or `CpuEngine::set_batch_policy`; `CpuBatchPolicy` and `CpuBatchStrategy`
are in `tenferro_cpu`), or force lanes with `Outer` or `Split`. A backend-free Host region (plan compile, strided
moves) takes the engine's placement token; a serial engine never forks.
An entry whose arithmetic needs no token (vector arithmetic, the
physical-basis conversions, the up-front engine check of an owned
contraction, and a stage with nothing to move: a contraction destination
without inactive blocks to initialize, a transform without destination
scales) only admits the engine, with the Host region's refusals, and takes
no token from it.

Tree-transform movement prices the jobs it executes: a destination hull run
for Singles or inactive fills, a pack column, a scatter group run, or a member
range. Every executed role contributes its setup, run and element costs:

```text
t_i = a[c] + r[c](W) * runs_i + e[c](W) * elements_i
planned leaf time = min(T_planned, k * D(s) + max(T_planned/s, X_planned/B(s, W)))
tau_leaf = T_serial + planned leaf time
T(q) = D(q) * [q > 1] + max(max_leaf(tau_leaf), X/B(p, W))
q* = argmin T(q), with ties to smaller q
```

Here p is the caller token's slot budget, q its number of leaves, s a leaf's
actual slot share, and k its number of planned roles. The same expression
prices q=1 on all p slots. Concurrent leaves share the aggregate bandwidth
at p slots for every candidate, including inline execution. The planned-leaf expression aggregates each leaf's planned work before
comparing serial and parallel execution. The candidates are
`{1,2,3,4,6,8,12,16,24,32,48,64,p}` intersected with `[1,min(p,J)]`, where J
is the number of runnable jobs.

W bounds source and destination address extents, and X counts structural
logical reads and writes, including destination reads for accumulation.
Runtime zero or identity elisions can make this count conservative. Structural
classes distinguish contiguous or strided inner traversal, and inactive fill
from real maps. Copy and non-identity real scale share a class; runtime alpha
and destination scales never alter the compiled classes. Rows are indexed by
structural class only; unmeasured fill rows use the same-stride real-map row.
An unlisted `HostClass` uses the checked-in AMD EPYC 9V45 CI rows, the same
fallback as `X86_64Avx512`; these seeds are not validated predictions for that
host. Reference curves interpolate in log W and remain flat outside their knots.
Beyond the last online knot, an online curve keeps the reference curve's shape,
scaled to meet that knot.

Planned roles are exactly those retaining the compiled Single or scatter plan;
pack columns, inactive fills and fused member ranges execute serial walks.
Credit also requires an overwriting writer whose adapter replays plans, and
the caller gate's Copy admission on that leaf's token share. A phase compiles,
per executed job, its smallest planned role size and its largest planned role,
in a segment tree over the jobs, and a leaf reads its range's two extremes in
O(log J). Admission uses the same cheapest-width `pays` predicate as the
execution gate and is monotone in a role's element count: when the leaf's
slots admit its smallest role, every planned role of the leaf is credited;
otherwise, when they admit its largest, that role alone is credited and the
admitted roles between the two are priced serially (an upper bound). No
runtime pass over individual roles is needed. This policy can withhold inner
fan-out credit, but never grants it to a rejected role or accumulation.

The plan-group chunk decision adds the dense provider's existing serial GEMM
estimate to each group's pack and scatter costs. Its q consecutive GEMMs are
submitted together, with serial barriers between chunks, so the critical-path
estimate is the sum of each chunk's maximum group cost. Every chunk also pays
its three engine entries (pack, GEMM batch, scatter) at the engine's measured
entry cost (none when unmeasured), and every chunk of more than one group pays
D at its group count for its batch fork; the chooser's D(q) is the first
chunk's, which always holds q groups. That estimate conservatively withholds
inner movement credit; actual pack and scatter phases choose their own leaves,
and their forks are priced by their own decisions. It does not predict the dense provider's placement or
runtime. Only movement elements feed B(q,W); dense time never changes that roof.

The schedule compiles per-class setup, run and element totals, planned totals,
logical traffic and cumulative components over executed jobs. Model-only counts
and envelopes saturate without adding compilation failures; subtracting a
saturated prefix stays conservative. Fixed phases retain cumulative job
components, and pack columns share one compiled prefix in column order.
Scatter ownership runs depend on the chosen group chunk. On a decision-key
miss, the workspace orders each selected chunk's groups, builds its ownership
runs, reusable movement prefix and planned-extremes tree, and combines its pack/scatter summaries
before evaluating widths. These aggregates depend on the runtime group width;
the plan-only counts, extents, planned extremes and pack/job prefixes remain
compiled with the structure. Each modeled leaf reads two prefix entries in
O(1) and its planned extremes in O(log J). There are no per-candidate
allocations.
A cold pack fill is divided by the same recursive leaf counts as execution.

With I role intervals, B blocks, F total axis metadata, and G plan groups,
compilation takes O(I log I + B + F) time and O(I+G) retained metadata.
For J_s Single jobs, J_z inactive jobs and C pack columns, the new prefix
payload is `size_of::<Job>() * (J_s + J_z + C)` bytes (224 bytes per entry:
two arrays of four `(items,runs,elements)` u64 triples and the job's own
32-byte planned extremes, the segment tree's leaves), plus
`size_of::<Planned>() * J` bytes of internal tree nodes for each phase of J >= 2
jobs, reserved once at its exact size so compilation adds a constant number of
allocations per phase. Actual capacities are charged, together with every
Summary and its inline planned totals.
Each per-dtype workspace in the engine's retained context keeps decisions for
at most 16 compiled structures. Entries are ordered by most recent use (LRU);
a missing structure reuses a dead Weak identity first, then an unused slot,
then the least recently used live entry. A changed decision key replaces that
structure's entry. Replacement reuses the width, dispatch and chunk vector
capacities, including each chunk's order, run and prefix buffers. Spare chunks
stay allocated for reuse. The existing `symtn::scratch::trim` drops the retained
context and all these decisions. The key includes the compiled structure's Weak
identity, slot budget, constants generation,
full batch strategy, dtype size, destination-read and plan-replay flags,
destination scratch length/capacity, and member count. Token team authority
and, under Auto, the exact dispatch costs at widths 2 through p complete the
key: these also determine `gate_keys.admits`, together with the compiled plan bound.
Gate admission therefore participates in memoization rather than being
assumed constant across engines. Forced strategies, including their parameters,
are part of the key; `forced_width` still resolves before model evaluation.
Forced policies retain no dispatch-cost row because their widths do not use it.
When a cold replay needs a packed-destination fill, the same decision pass
also prices that pack without the fill. A successful replay installs the warm
pack width and records the resulting scratch length/capacity, so the second
identical replay evaluates no model. A failed replay does not advance the key.

For each entry with P phases the width payload is `P * size_of::<usize>()`
bytes (8P on a 64-bit target): three direct/group decisions plus two per selected chunk, or
two member decisions. Its allocation charge uses the vector's actual capacity.
The Auto dispatch key adds `capacity * size_of::<Duration>()` bytes. Retained
chunk order/run tables charge `capacity * size_of::<ChunkMovement>()` for the
outer vector, plus each group's `capacity * size_of::<usize>()` and each run
list's `capacity * size_of::<u32>()`, scatter prefix's
`capacity * size_of::<Job>()` and its planned-extremes tree's
`capacity * size_of::<Planned>()`. Each chunk's inline size includes one additional
`usize` for its warm pack width. Temporary aggregate summaries are dropped
after deciding. The store additionally charges
`entry_capacity * size_of::<WidthDecision>()`; every initialized entry and
spare chunk contributes all of its vector capacities. Each initialized entry
also conservatively charges the Weak marker's two-usize Arc control block,
even while the structure shares it. All these charges are included in `HostTreeTransformWorkspace::retained_bytes`.
Gate keys retain direct budget indices
(`size_of::<usize>() * capacity`) plus their existing distinct cost/scan slots;
the existing O(p) preparation pass makes each admission lookup O(1).
On a key hit each phase reads its width in O(1), without model evaluation,
summary construction, sorting or an engine-extension lookup. The operation's
structure lookup checks at most 16 Weak identities; its exact dispatch-key
comparison costs O(p) under Auto and O(1) under forced policies. On a miss,
a fixed chooser costs O(sum_candidate q), with constant-size class arithmetic per leaf and O(log q)
recursion depth. The plan-group decision visits G groups per candidate to
price actual batch barriers, and each chunk combines t group summaries once
in O(t). One effective width returns before cost preparation or element-count
reads.

Single-tensor one-slot, Sequential and Inner replays bypass the movement model
and calibration, using the serial item and plan-group sequence; Inner keeps
the whole token for inline kernels. Wider Auto replays retain independently
priced pack and scatter phases even when group chunk width is one.
A q-way split divides the whole token into nonzero leaf shares summing to p
and differing by at most one. Job boundaries and slot remainders follow the
executor's recursive `parallel_split`, rather than flat equal-sized job chunks.
Each destination ownership run (a maximal address-ordered stretch of
transitively overlapping hulls) stays with one writer; width cannot exceed
the number of runs. A leaf stops at its first error; the lowest failing job
and the first failing phase or GEMM determine replay's error.

Movement constants are retained separately in the caller CPU engine's bounded
extension store and fetched once per operation, before preparing all its
movement decisions. Reference entry creation and completed online updates
receive fresh process-wide generations, so a workspace cannot reuse widths
from a different engine entry or an evicted entry's replacement.
Online measurement is attempted once per retained entry,
through the real replay writer using shared buffers, within a 10 ms budget
including fixture construction. The deadline is checked before every fixture.
Failed or unfinished entries keep the next source and
are not retried before eviction or `scratch::trim`. Each row uses, in order,
its online measurement, the checked-in reference for threadteam's HostClass,
or the slowest available row with the same inner stride. An online row must be
physical (a >= 0, r >= 0, e > 0). A negative fitted setup or per-run term means
that term is below the measurement's resolution, not that the host differs: it
is fixed at zero and each one-run sample's total is attributed to the remaining
terms (their least-squares fit on the same samples). Only a row that is still
non-physical (e <= 0), has non-finite samples, or hit the deadline before its
samples falls back. The reference table
records the CI CPU, date and run provenance; that run's fit-acceptance flag was
false, so these rows are measured seeds, not validated phase predictions.
D(q) uses the engine entry's online empty-fork measurement when available,
otherwise threadteam's `Exec::dispatch_cost`, including the pool creator's
overrides. The chooser's candidate widths below p, and p, are measured through
the running engine's `limit(q)` prefix after warm-up, taking the median of five
batches of eight empty forks, before the movement fixtures and inside their
shared 10 ms budget. Samples above 20 times the table value are rejected; there
is no lower bound. When at least one width is accepted, every width from 2 to p
is completed from the accepted anchors: linearly between them, and outside them
the table scaled to the nearest anchor; the row is then made non-decreasing by
its running maximum, so measured and tabulated widths never mix into a dip.
Without an accepted width every width keeps the table. The same deadline also
times one empty guarded engine entry (median of five batches of eight, outside
any region), which prices plan-group chunks. These measurements affect only symtn's
movement model; the pool's dispatch table and dense provider remain unchanged.
Bandwidth uses the same element costs as the serial model:
E(W) = sum_c(elements_c * e[c](W)), b1(W) = X/E(W), and
B(q, W) = min(q*b1(W), B_sat(q, W)). Setup and run costs remain in S, rather
than being charged again in E. Logical destination reads, fused member ranges
and cold fills contribute their existing element counts and traffic.

The terminal value of each measured element curve defines its memory plateau;
its knee is the first knot after the curve's minimum that reaches that value.
Online knots and the scaled reference tail both participate. B_sat is infinity
below the smaller of the largest knee of the phase's active classes and the
measured bandwidth fixture's working set: a mixed phase uses the memory cap
only after every active curve has reached its plateau, unless a measured
fixture already bounds its bandwidth. The knees are
computed once per retained constants entry. Compiled counts supply the phase's
class weights directly, including member and cold-fill element counts, without
extra cost preparations or allocations.

Bandwidth fixtures are at most the contiguous curve's knee in size and use
exactly the same q-slot prefix when measuring q, regardless of the caller's
budget. Each ladder knot is attempted once per retained entry. Measurements
below the fixture's single-thread memory rate or below the previous accepted
knot are rejected before linear interpolation in q. Missing widths use the
nearest accepted endpoint outside the measured range; no accepted measurement
means infinity. A phase also rejects a prefix below either its own b1(memory W)
or b1(W), using the nearest accepted endpoint at or above both rates, or
infinity if none exists. The current-W check covers noise after the knee.
The row-calibration buffers occupy 8 MiB; they are released before bandwidth
buffers are allocated lazily, requesting combined payload capacity at most the
contiguous knee plus 16*(p+1) bytes. Their destination pages are first written by workers.
Before allocating those buffers, admission bounds their size by the remaining
deadline using the observed small-buffer allocation, source first-touch and
teardown rate, the serial element curve, and every remaining ladder dispatch.
The fixture is the memory knee when that bound accommodates it, and otherwise
the largest size the bound admits; below the largest row-calibration working
set (4 MiB) the measurement is skipped and the cap stays unknown. A bandwidth
measured at a working set bounds the bandwidth of every larger one, so the cap
applies to phases whose W reaches the smaller of the knee and the measured
fixture's working set.
The deadline is checked again before each fixture and after each sample;
unfinished measurements leave the cap unknown. Source selection never changes
the formula or chooser.

A fixed-stride member batch admits all member ranges before writing, resolves
destination scales once per role, and moves its members around one dense
submission (plan group outer, member inner). For N members, let A be the sum
of setup costs per role and P the per-member sum of run and element costs.
Its executed q ranges have critical path A+ceil(N/q)*P because their fused
walks are serial-only. This expression feeds the same chooser, including q=1
and the common aggregate bandwidth floor. A cold packed-destination fill is
also priced by its executed contiguous leaf ranges, and first-written by the
pack workers; a warm destination needs no fill. Growth within retained capacity
fills only the missing tail. Forced policies delegate to Tenferro's batch
decision: its lane count is clamped to the caller's budget and runnable jobs;
item threads stay inside each leaf token.
An executor that cannot honor a forced
batch policy refuses a
tree-transform replay, owned overwrite or member batch under a forced
`Inner`, `Outer` or `Split` policy with a typed `Unsupported` before any
coefficient preparation, write or executor call; `Auto` and `Sequential` run. Accumulating updates
(`beta` neither zero nor absent), and blocks whose normalized layout exceeds
Strided's inline fused rank of eight, are never written through a Strided
plan. The cold compile of a tree transform fans its staged groups out over
the engine's width in contiguous batches, one per one-thread lane, and
reports the first error in source order. symtn's own data movement (a tree
replay's direct writes, packs and scatters) writes the same bits at every
engine width. Results are bit-identical for one
configuration (provider, build, engine width, batch policy and work model,
inputs), whether a block placement is searched or taken from the cache. No promise is made across widths or providers: a provider GEMM may
reduce in another order at another width (faer's matrix-vector route splits
the contraction over its threads), so results agree within rounding.

[`Runtime`](../symtn/src/runtime.rs) holds prepared structural state only.
Serial Host transforms, transform staging and factorizations build execution
scratch for the call and drop it on return; a prepared plan's workspace keeps
its scratch for its replays. Eager Host contractions
(`contract`, `contract_into`, `compose`, the planar contractions and every
network step) and multithreaded Host transform replays use one shared context
entry of the caller's `CpuEngine` extension store, kept between calls under that
store's entry and byte limits (64 MiB by default) and released by
[`symtn::scratch::trim`](../symtn/src/scratch.rs); a ledger that does not
fit is dropped, so that workload allocates its scratch per call. Movement
widths live in that context's replay workspace. Movement constants use
another typed entry under the same bounded store and trim
lifecycle; retaining either payload never replaces the other. Structural
and plan caches take short lookup and publication locks.

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
