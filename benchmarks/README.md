# Benchmarks

Benchmark harnesses are executable measurement tools. Recorded reports are
revision-specific evidence, not current performance claims.

Records in `history/`, `issue_*_results/`, and the dated `issue_*.md` files
before #2273 use `tenet-<x>` / `TENET_*`; they are now `symtn-<x>` / `SYMTN_*`.

## Current harnesses

- `symtn/benches/vector_ops.rs`: single-thread Host vector operations, slice
  oracles and 1/4/8/16-lane dot candidates for U(1), SU(2), and Fibonacci;
  f64, Complex64, and f32 for U(1)/SU(2), Complex64 only for Fibonacci,
  with allocation calls/bytes. Run
  `cargo bench -p symtn --bench vector_ops --offline -- U1` (optional substring
  filter). Each selected row runs for at least 0.2 seconds after warmup.

- `symtn-network/benches/cache_scaling.rs`: warm-hit and cold-miss
  throughput of the structure caches and the plan cache under 1–96 scoped
  threads (raw `StructureCache` lookups and admissions; `permute`,
  `contract`, `Network::contract` on U(1)/SU(2)), with per-window CPU time
  and cache-statistics deltas. Run
  `cargo bench -p symtn-network --bench cache_scaling -- [filter ...]`;
  thread counts and windows are set by `CACHE_SCALING_*` (see the file).
- `operation_matrix.sh` and `operation_matrix.md`: Host operation matrix,
  environment capture, and comparison protocol.
- `eager_overhead_ledger.sh` (`symtn/benches/eager_overhead_ledger.rs`,
  `tensorkit_eager_overhead.jl`, `eager_overhead_phases.py`): per-call time,
  allocations, thread layouts, and sampled phase attribution of the eager
  primitives on small many-block tensors against TensorKit.
- [`issue_1684_null.rs`](issue_1684_null.rs): paired Host compact-diagonal
  left/right null-space calls with dense and near-cutoff controls; raw results
  and method are in [`issue_1684_results/`](issue_1684_results/README.md).

Run a harness at the revision being evaluated and keep its correctness checks
enabled. A historical result does not become current because its script still
exists.

## Historical results

- [Structure-cache thread scaling](history/cache-scaling-2026-10-28.md)
- [July 2026 contraction microbenchmark](history/contraction-microbench-2026-07-11.md)
- [Eager per-call overhead ledger](history/eager-overhead-ledger-2026-09-21.md)
- [One CPU session per compact factorization](history/factorization-session-2026-09-21.md)
- [Generic-path performance review](issue_1007_current_main.md)
- [Checked Generic values input borrowing](history/checked-generic-values-borrow-2026-09-13.md)
- [Warm-cache audit](issue_118_warm_cache.md)
- [Allocation evidence](issue_124_allocation_evidence.md)
- [Fusion-layout cache benchmark](issue_245_layout_cache.md)

The issue-specific shell script `issue_118_audit.sh` is retained only to
reproduce its linked historical report and external-checkout setup.
