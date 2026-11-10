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
with the same results. For initialized destinations this holds for the faer
provider and the four payload scalars only; BLAS and CUDA builds prepare as
before, and single-job or affine batches use Tenferro's binary-plan cache,
outside this cache.

The dense executor also has a hidden fresh entry that writes a batch into
uninitialized destination memory, for owned outputs. It keeps the
initialized route's choices. A single job or a gapless affine run inside one
destination segment is zero-filled and then takes its initialized strided
call. An affine run with gaps between its outputs goes grouped. On the faer
provider a grouped fresh batch runs the same lanes, geometry-plan cache and
executor staging as an initialized one, through tcontract's fresh entry.
That entry still zero-fills each output block on its lane before the product
(tcontract's faer strategy initializes a fresh destination itself), so the
fill moves from one serial pass into the lanes rather than disappearing. On a
BLAS provider under `Auto` or `Sequential` with nonzero `alpha`, a grouped
fresh batch is zero-filled and makes its initialized call: the same grouped
plan and whole-group vendor/native decision, the same per-job engine cache
slots, and one extra store over the destination. When its jobs lie in
several destination segments, an identity batch makes one initialized call
per segment (so the vendor/native decision is per segment) and an op-bearing
batch runs each job as the initialized serial route does. A forced
`Outer`/`Inner`/`Split` policy, or `alpha == 0`, runs a grouped fresh BLAS
batch on the native lanes instead (cpueinsum's grouped native alternative
has no fresh form); those calls use the geometry-plan cache and the
executor's staging above, which a BLAS executor then also keeps.

An owned eager Host contraction (`contract`, `planar_contract`, `compose`,
in both modes) allocates its output without zeroing it when the route's core
GEMMs write the destination (the canonical core, or a `DynamicTree` route
without an output transform). Each GEMM call receives only its own
destination ranges through the fresh entry above; the blocks no direct GEMM
writes (inactive blocks, padding, and the blocks a packed group scatters
into) are written `+0.0` by symtn, and scatter groups then overwrite theirs.
The output is published only after every element has been written. A route
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
large-block reservation under one model, and keeps the lowest score; ties
keep the baseline, then the equal shares. The reservation finds the least
target time at which the blocks above it, each on the narrowest lane that
brings it under the target, fit in the thread count; it gives each such
block that lane, makes the remaining threads one-thread lanes, places the
other blocks largest first on the lane where they finish earliest, then
widens the reserved lanes with the threads of unused one-thread lanes while
the provider's width still grows. It is discarded when more than `L0` of its
lanes hold blocks, and it is scored as placed, since a block below the
target can still finish above it. The model charges a block its serial time
where the provider would run it on one thread in that lane, otherwise its
time over the provider's granted width plus the engine's dispatch cost at
that width. The granted width is an estimate composed from the provider's
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
algorithm and the work; for a factorization, its total and payback work)
once, at the full thread count; only the groups the provider widens there
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
same per-sector checks as a block-by-block loop run on each, so the error
reported is the lowest-indexed sector's, the one the per-matrix loop would
report, and nothing is returned on error. Two costs differ from that loop
and are accepted: blocks after a failing one are still solved, and a
provider panic in such a later block unwinds and leaves the engine broken
where the loop would have returned the earlier error. Compact QR, compact
LQ, full QR and LQ, the `left_null` and `right_null` bases, compact and
full SVD, full EIGH, the left and right polar decompositions, the Hermitian `exp`
and the inverse run through
`DenseExecutor::factorize_batch_into`: the
operation allocates each output once (the final packed factor for a
direct-region or Checked Generic canonical output, otherwise one packed
transient per call that is scattered by tree identity afterwards, about
twice the factor payload on that route) and one packed real spectrum, and
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
that every other executor, and the BLAS provider, still runs through its
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
temporaries). Every other executor, and the BLAS provider,
reach `factorize_batch_into` through its default: the owned
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
The other dense problems, and every
BLAS factorization, still run block after block on the whole engine.
Callers can still give each GEMM the whole pool with
`CpuEngine::new(pool).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Inner))`
(or `CpuEngine::set_batch_policy`; `CpuBatchPolicy` and `CpuBatchStrategy`
are in `tenferro_cpu`), or force lanes with `Outer` or `Split`. A BLAS provider under `Auto` instead runs the batch's
vendor GEMMs one after another on the calling thread; any threads the vendor
library starts are outside the engine's lanes. Forcing `Inner` (or another
lane strategy) on a BLAS provider runs those GEMMs on the native kernels, not
the vendor library. A backend-free Host region (plan compile, strided
moves) takes the engine's placement token; a serial engine never forks.

A tree-transform replay follows TensorKit's threaded kernels and main's
replay. One Host region fixes its width `t`: under `Auto` the engine's
width, at most the compiled runnable parts (the Singles when their
destinations are slice-disjoint, else one, or the pack columns, whichever is
more), when Strided's copy estimate of the whole destination pays for that
section, and one otherwise; under a forced policy its batch decision
(`lanes * item_threads`, one for `Sequential`), at most the runnable parts,
on the caller's whole token. Destinations are grouped into hull runs:
maximal address-ordered stretches whose bounding slices overlap
transitively, so distinct runs occupy disjoint address ranges and one
writer owns each run. In that region the inactive blocks are split over
their runs when there are two or more and the same copy rule admits more
than one worker for that many runs, otherwise written inline on the
caller's token in destination address order; then the Singles: at `t > 1`
with slice-disjoint destinations split over `t` workers, when they are not
slice-disjoint but form two or more runs split over those runs unless the
caller's gate would admit the largest Single with a retained Strided plan
on the caller's token (then inline, where that plan fans the move out), otherwise
inline: in block order into a caller-provided destination at `t = 1`
(main's one-worker replay), in address order into an owned output or at
`t > 1`. A split part of
`n` workers and `k` items gives `ceil(k (n / 2) / n)` items and `n / 2`
workers to its left half and forks its token in two (the odd slot to the
left); a part of one worker or one item is a leaf, which writes its items
whole on its token. The Multi groups then replay in chunks of `t`
consecutive groups: each chunk's pack is split the same way over its
columns, its jobs go to the executor as one grouped GEMM, and its non-empty
groups, ordered by destination, are scattered split by group when every
group's columns and every pair of groups are slice-disjoint, over their hull
runs when they form two or more and the caller's gate would not admit the
chunk's largest planned column on the caller's token, and inline on the
caller's token otherwise. At `t = 1` each group is packed,
multiplied and scattered in plan order. Every inline move is offered the
caller's whole token (one slot under a forced `Sequential`), so Strided's
own rule decides whether one large block's move fans out; an overwrite of a
block whose role retains a Strided plan writes through that plan when
Strided's rule (`execution::parallel_width` for a copy of its elements)
gives it more than one worker. One leaf gate per call on the caller, and
one per split leaf, answers that rule exactly, scanning the token's
dispatch costs once and searching widths only for a role at or above the
smallest paying size; the split leaves of one call share that smallest
size per cost of their cheapest width, which the caller prepares for every
leaf budget before the first split, so each such cost is searched at most
once per call among the leaves, and only when a leaf needs it (the
caller's own gate may search once more, as before). A leaf stops at its first failure, a split runs both
halves and returns the left failure first, and the first failing phase or
GEMM stops the replay. The packed destination of the chunks is sized once
for the largest chunk; when it must grow it is zero-initialized inside the
first chunk's pack, each split part its own range.
A fixed-stride member batch (several members of one transform)
admits every member's ranges before any write and resolves each
destination scale once per role, then moves its members (the member axis
one more strided dimension of each move) around one dense submission of
every member's GEMMs, ordered plan job outer, member inner, and returns
the first failure in that order. Each move phase takes the width the
replay rule admits over the members (under `Auto` the engine's width when
Strided's copy estimate of the whole batch destination pays for it, under
a forced policy its batch decision): at one every member moves on the
caller's token, above it the members split into at most that many
contiguous ranges, each moving its members as one move per role. An executor that cannot honor a forced
batch policy (the BLAS provider and every custom executor) refuses a
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
BLAS-internal threads remain separate.

[`Runtime`](../symtn/src/runtime.rs) holds prepared structural state only.
An eager operation builds its execution scratch (tree-transform pack
scratch, contraction lanes, a factorization's dense executor) for the call and drops
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
