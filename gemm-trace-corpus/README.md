# symtn grouped-GEMM trace corpus

Grouped GEMM calls that symtn's default dense executor receives during a real
calculation, for tensor4all/cpueinsum-benchmark#18.

## Files

The digest below is for the capture host (Apple M4 Max, arm64, 4 threads). The threaded SVD can truncate differently on other hardware, so regenerate the corpus and its digest on the host that runs the timing campaign instead of reusing this one.

| file | content | sha256 |
|---|---|---|
| `itebd-heff-b705d809-t4.jsonl.gz` | **main corpus**: iTEBD + H_eff on iTEBD-converged bonds | `3f48a98a7208dc6a3bf6482f567b4e64571333e50724b6f7a209d36336114ed9` |
| `summarize.py` | per-workload summary table below | - |

## Provenance

- symtn fork branch `trace/gemm-trace` of Ryo-wtnb11/symtn-rs, commit
  `b705d809973f3d73118f399d368c35af143078e0`, branched from upstream main
  `a2825c6cdc16ad5bbc2bef9109edb8d3435de763`. The branch diff adds only the
  capture hook (`symtn-dense/src/gemm_trace.rs`, three record calls at the
  entries of `DefaultDenseExecutor`'s batch methods in
  `symtn-dense/src/tenferro_adapter.rs`, feature `gemm-trace`, off by
  default) and the driver `symtn-network/examples/gemm_trace_capture.rs`.
- Provider `cpu-faer` (default features), engine width 4
  (`CpuEngine` with a 4-thread pool), release build, Apple M4 Max (arm64),
  rustc 1.100.0. Calls are recorded at executor entry, before dispatch, so
  the job lists do not depend on the provider; they can depend on the engine
  width (lane splitting), hence the `threads` field.

## Workload

Spin-1/2 antiferromagnetic Heisenberg chain, U(1) (twice S_z) and SU(2)
(twice spin, multiplicity-free), iTEBD in Vidal form (the update of
`symtn-network/tests/itebd_smoke.rs`), SVD truncation
`Truncation::rank(chi) & relative_cutoff(1e-12)`, chi = 64, 256, 1024
states (SU(2) rank counts states; the 1024 run settles at 935 states / 273
multiplets because of the cutoff).

1. Untraced warm-up: 200 AB+BA update pairs at dt = 0.1, then 100 at
   dt = 0.01. Energy per bond afterwards: U(1) -0.442236, SU(2) -0.442467
   (exact 1/4 - ln 2 = -0.443147; the residual is finite imaginary time).
2. Traced: 2 steady-state update pairs (4 bond updates) at dt = 0.01, steps
   `itebd.theta` (6-tensor network: Lambda, Gamma, Lambda, Gamma, Lambda,
   gate), `itebd.svd` (SVD, truncation, restriction, normalization),
   `itebd.pinv`, `itebd.gamma_left`, `itebd.gamma_right`. Only
   `itebd.theta` issues grouped GEMMs; the others use diagonal or
   factorization paths and record nothing.
3. Traced once, f64 and c64: the two-site DMRG matvec H_eff * theta on the
   converged bond space v (from `lb`), theta: v (x) p (x) p <- v, MPO
   virtual space U(1) {0:3, +-2:1}, SU(2) {0:2, 1:1}; steps `heff.L_theta`,
   `heff.W1`, `heff.W2`, `heff.R`. These tensors are random, but the
   bond space is the one iTEBD produced, so the block structure (and the job
   lists) are those of a DMRG matvec at that bond.

Deviations from #18: no finite DMRG sweep (symtn has no DMRG example; H_eff
on the iTEBD bond stands in for the Lanczos matvec); c64 only for H_eff (the
iTEBD job lists would be identical up to dtype).

## Format `symtn-gemm-trace/1`

gzip-compressed JSON lines. Line 1 is a provenance record (`kind:
"provenance"`, `format`, `symtn_commit`, `symtn_base`, `provider`,
`engine_threads`, `workload`, `job_fields`, `layout`). Every other line
(`kind: "call"`) is one distinct grouped call
`C_j = alpha * op0(A_j) * op1(B_j) + beta * C_j` of one
(`model`, `chi_states`, `chi_multiplets`, `scalar`, `step`):

- `multiplicity`: how many identical calls (all fields equal) this record
  stands for within that group; records are in first-seen order;
- `dtype` (`F64`/`C64`), `ops` (`N`/`T`/`C` per operand),
  `alpha`, `beta` (`[re, im]`; `null` = fresh, never-read destination),
  `threads` (engine width), `runs` (the caller's strided-run partition,
  symtn's `strided_batch_runs`; empty for fresh calls);
- `jobs`: `[a_offset, b_offset, c_offset, m, n, k, lda, ldb, ldc]`,
  column-major, offsets relative to each operand buffer (the
  tensor4all/tcontract-rs#209 descriptor).

## Regenerate

```sh
git -C symtn-rs fetch <fork> trace/gemm-trace
git -C symtn-rs worktree add ../wt-gemm-trace b705d809973f3d73118f399d368c35af143078e0
cd ../wt-gemm-trace
cargo build --release -p symtn-network --features gemm-trace --example gemm_trace_capture
./target/release/examples/gemm_trace_capture itebd-heff-b705d809-t4.jsonl \
    b705d809973f3d73118f399d368c35af143078e0 a2825c6cdc16ad5bbc2bef9109edb8d3435de763 4
gzip -9 itebd-heff-b705d809-t4.jsonl
python3 summarize.py itebd-heff-b705d809-t4.jsonl.gz
```

Runtime about 40 s. The SVD runs on 4 threads, so a rerun on other hardware
can truncate a degenerate multiplet differently and shift a sector size; the
digest above is for this host.

## Summary (404 calls, 824 jobs, 231 distinct records)

Counts weight records by multiplicity; `mkn` = m*k*n per job; "work-wtd med"
is the job size at the median of total work; "largest-job share" is the
fraction of a call's work in its largest job.

| model | chi states (multiplets) | scalar | steps | calls | distinct | jobs/call min/med/max | m | n | k | mkn min/med/max | work-wtd med mkn | largest-job share med | distinct geometries/call med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| su2_heisenberg | 64 (22) | Complex64 | heff | 43 | 28 | 1/1/3 | 1-242 | 2-704 | 1-44 | 100/495/15972 | 5130 | 1.00 | 1 |
| su2_heisenberg | 64 (22) | f64 | heff | 43 | 28 | 1/1/3 | 1-242 | 2-704 | 1-44 | 100/495/15972 | 5130 | 1.00 | 1 |
| su2_heisenberg | 64 (22) | f64 | itebd | 32 | 9 | 1/1/3 | 1-308 | 2-11 | 1-18 | 4/616/6534 | 3584 | 1.00 | 1 |
| su2_heisenberg | 256 (80) | Complex64 | heff | 64 | 41 | 1/1/4 | 1-2592 | 2-9152 | 1-148 | 64/4384/596736 | 596736 | 1.00 | 1 |
| su2_heisenberg | 256 (80) | f64 | heff | 64 | 41 | 1/1/4 | 1-2592 | 2-9152 | 1-148 | 64/4384/596736 | 596736 | 1.00 | 1 |
| su2_heisenberg | 256 (80) | f64 | itebd | 40 | 11 | 1/1/4 | 1-3772 | 1-36 | 1-63 | 4/5400/254016 | 177408 | 1.00 | 1 |
| su2_heisenberg | 935 (273) | Complex64 | heff | 25 | 17 | 2/4/5 | 1-27378 | 1-108480 | 1-486 | 4/28080/20982078 | 20982078 | 0.47 | 3 |
| su2_heisenberg | 935 (273) | f64 | heff | 25 | 17 | 2/4/5 | 1-27378 | 1-108480 | 1-486 | 4/28080/20982078 | 20982078 | 0.47 | 3 |
| su2_heisenberg | 935 (273) | f64 | itebd | 20 | 6 | 1/4/5 | 1-41782 | 1-117 | 1-203 | 4/69525/8764119 | 6921600 | 0.77 | 4 |
| u1_heisenberg | 64 | Complex64 | heff | 4 | 4 | 4/4.5/5 | 1-98 | 5-1910 | 1-98 | 1340/30560/163856 | 70800 | 0.50 | 2.5 |
| u1_heisenberg | 64 | f64 | heff | 4 | 4 | 4/4.5/5 | 1-98 | 5-1910 | 1-98 | 1340/30560/163856 | 70800 | 0.50 | 2.5 |
| u1_heisenberg | 64 | f64 | itebd | 8 | 3 | 2/3.5/6 | 14-980 | 2-22 | 1-40 | 140/2940/66880 | 53200 | 0.50 | 2 |
| u1_heisenberg | 256 | Complex64 | heff | 4 | 4 | 4/5.5/7 | 1-366 | 3-27958 | 1-366 | 2970/359640/8523408 | 3878160 | 0.49 | 3 |
| u1_heisenberg | 256 | f64 | heff | 4 | 4 | 4/5.5/7 | 1-366 | 3-27958 | 1-366 | 2970/359640/8523408 | 3878160 | 0.49 | 3 |
| u1_heisenberg | 256 | f64 | itebd | 8 | 3 | 2/4.5/8 | 13-14336 | 1-82 | 1-148 | 39/43008/3446624 | 2774112 | 0.50 | 2.5 |
| u1_heisenberg | 1024 | Complex64 | heff | 4 | 4 | 4/6.5/9 | 1-1348 | 2-410230 | 1-1348 | 1680/6.56368e+06/417712848 | 221754022 | 0.47 | 3.5 |
| u1_heisenberg | 1024 | f64 | heff | 4 | 4 | 4/6.5/9 | 1-1348 | 2-410230 | 1-1348 | 1680/6.56368e+06/417712848 | 221754022 | 0.47 | 3.5 |
| u1_heisenberg | 1024 | f64 | itebd | 8 | 3 | 2/5/9 | 28-213086 | 2-294 | 1-566 | 392/2.0306e+06/175389816 | 151527256 | 0.48 | 2.5 |

Highlights: U(1) issues one call per contraction step with 2-9 jobs whose
sizes span two to five orders of magnitude (chi = 1024 H_eff: 1680 to 4.2e8
mkn, half of each call's work in its largest job). SU(2) issues many calls
with one job each at chi <= 256 (one call per fusion-tree pair) and 2-5 jobs at
chi = 1024; per-call largest-job share is 1.00 below chi = 1024. Shapes are
strongly rectangular: H_eff n reaches 4.1e5 against m, k <= 1348.
