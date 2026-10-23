# Issue 1682: diagonal full SVD

## Reproduction

Baseline: `8dd296d54d6a0eb87c1dae26ac0c1021c27a8fc7`.
Production implementation: `b2b82413e971f88c0852d2022f4eef0b5c122192`.
Harness: `../issue_1682_svd_full.rs`, SHA256
`5a55e8a2cd11289a010b3fd1d31040fb40cfb99ee37831b2a0c222423f60553e`.
Both builds use Cargo.lock SHA256
`cc03b84d11f9bacff29a6dcdab1259a2349b76f9ece45a60153f421ebc5d74ae`.

Apple M4 Max (12 performance + 4 efficiency cores), macOS 15.5;
aarch64-apple-darwin rustc 1.96.0 `ac68faa20`, LLVM 22.1.2. Release/default
features, explicit Faer, dense threads 1. OMP/OPENBLAS/VECLIB/MKL/RAYON thread
environment variables are all 1. No CPU affinity is asserted.

Copy the harness into `tenet/examples/issue_1682_svd_full.rs` in each checkout.
Build sequentially with a compatible shared target, saving each executable:

```sh
cargo clean --release -p tenet-matrixalgebra -p tenet-rs
cargo build --release -p tenet-rs --example issue_1682_svd_full
```

Scoped package cleaning prevents shared-target cross-worktree freshness from
reusing the wrong executable. Saved hashes must differ; `--probe` verifies
baseline dense S versus candidate compact S, with dense U/Vh and matching
factor bonds in both. The compact probe requires `dense_data()` rejection as
well as `diagview()` success: dense diagonal S also permits `diagview()`. An
initial probe using only `diagview()` was rejected before any timing.
No measurement with identical binaries is accepted.

Five pairs alternate B/C, C/B, B/C, C/B, B/C. Each process measures 96 cases:
U1, SU2, U1 × fermion parity; f32/f64/c32/c64; (sectors, degeneracy, dual) =
(1,8,false), (4,8,true), (1,64,false), (4,64,true); compact diagonal and dense
storage. Dense controls materialize the same diagonal-valued square inputs
before timing. This is not a sweep over arbitrary dense matrix shapes.

Setup is separate. Full public `svd_full` calls include publication and output
destruction. `first_ns` is the first call of a case, not an independent cold
runtime: previous cases share the runtime. Five warmups precede one allocation
sample (calls and requested bytes), followed by eleven samples of twenty calls.
Each CSV records the median sample. Allocation counting is disabled for timing;
the allocator's atomic enable check remains. No dense-kernel-only claim is made.

`summary.csv` reports the median of five process medians, their minimum and
maximum, and allocations. `../issue_1682_summarize.py` regenerates it from the
raw paired CSVs. Timing is evidence, never a CI pass/fail gate.

## Result

All ten CSVs are complete: 96 cases per run. The original shell continued
through the session restart; no partial run or duplicate sample is included.
The 48 compact-diagonal cases
improve 4.47–91.47× (geometric mean of per-case speedups 16.75×). Allocation
calls fall by 32–262 per call, and requested bytes by 14,280–4,990,008.
For the 48 dense controls, median speedup ranges 0.986–1.022× (geometric mean
1.003×); allocation calls and bytes match exactly. No dense case has a
non-overlapping slowdown range; one small dense case has a non-overlapping
improvement range. These measurements support
the admitted compact route on this machine; they do not establish general
speedup or bitwise agreement across providers.

## Correctness

`red.log` records the baseline failure of the direct materialization assertion.
`gates.txt` records the final checks. Independent hand matrices verify descending
order, equal-magnitude source-index ordering, permutation U, right phase in Vh,
and zero phase +1 for all four scalars. Tests cover dual exact full-factor
spaces, reconstruction/isometry, multiple/empty/SU2/fermionic sectors,
subnormals, nonfinite and unrepresentable fallback, serialization, and compact
readback/materialization. Direct materialization and solver counters are zero.
Dense/lazy/checked Generic/CUDA retain their existing routes; no CUDA runtime
performance claim is made. Reference correspondence is in
[historical reference correspondence](https://github.com/Ryo-wtnb11/symtn-rs/blob/8c54ce960d0dd5ea000c75ff1893624e3da42420/docs/audit/issue-1682-compact-full-svd.md).
