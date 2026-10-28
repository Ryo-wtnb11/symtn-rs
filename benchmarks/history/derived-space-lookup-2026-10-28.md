# One complete-structure lookup per warm derived space

Date 2026-10-28. Host: Apple M4 Max, macOS 15.5, rustc 1.99.1.

- **Before:** `origin/main` `0b349bd3`.
- **After:** the derived output space looks up its complete
  (`DegeneracyStructure`) entry first and consults the sector structure only
  on a miss (on `0b349bd3`).
- **Build:** each revision's own `Cargo.lock` (identical, sha256 prefix
  `e9a0aac2dd4196bf`), `--release`, `CARGO_PROFILE_RELEASE_DEBUG=0`, default
  features, one scratch example per tree. Binary sha256 prefixes: before
  `083412db0b7dd40b`, after `fc099b203a96018b`.
- **Method:** rank-4 `[l, l] ← [l, l]` operands, `l` with `s` sectors (U(1)
  charges centred on 0, SU(2) spins `0..s/2`) and degeneracies `2 + i % 3`.
  `contract` is `TensorMap::contract` with a `[l] ← [l]` matrix on one leg;
  `svd` is `svd_compact(&[0, 1], &[2, 3])`. Runtime `dense_threads(1)`,
  `recoupling_threads(1)`, `RAYON_NUM_THREADS=1`. Per case: 20 warm-up calls,
  then the median of 7 windows (2000/200 contract and 500/20 svd calls for
  s3/s6). Order before, after, after, before, twice (ABBA ×2). No other
  benchmark ran meanwhile.
- **The timed "after" binary is not the final diff.** It also staged checked
  permutes complete-first, while the final change keeps permutes
  sector-first. The timings below are observational only and are not tied to
  the final binary.
- The evidence for the change is its lookup count, asserted by
  `symtn/tests/derived_space_lookups.rs` on the final diff. The lookup columns
  below are the `symtn::cache::stats()` hits + misses of one warm call.

## Results (µs per call; before / after is the ratio of the medians)

| op | sym | case | before (4 runs) | after (4 runs) | before / after | sector lookups/call | degeneracy lookups/call |
|---|---|---|---|---|---:|---|---|
| contract | U1 | s3 | 5.27 5.15 5.08 5.08 | 5.02 5.10 5.05 4.98 | 1.016 | 1 → 0 | 3 → 3 |
| contract | U1 | s6 | 18.33 18.62 17.53 17.95 | 17.43 17.16 17.39 18.55 | 1.042 | 1 → 0 | 3 → 3 |
| contract | SU2 | s3 | 20.65 20.65 20.38 20.41 | 20.41 20.50 19.96 20.00 | 1.016 | 1 → 0 | 3 → 3 |
| contract | SU2 | s6 | 394.89 404.81 396.74 401.80 | 402.06 403.38 400.71 401.59 | 0.994 | 1 → 0 | 3 → 3 |
| svd | U1 | s3 | 137.10 137.16 137.38 137.72 | 137.74 136.79 137.32 136.60 | 1.002 | 3 → 0 | 3 → 3 |
| svd | U1 | s6 | 988.67 1000.83 987.87 998.97 | 996.03 994.98 993.87 997.20 | 0.998 | 3 → 0 | 3 → 3 |
| svd | SU2 | s3 | 366.45 367.04 364.98 364.57 | 367.64 367.97 366.77 367.93 | 0.994 | 3 → 0 | 3 → 3 |
| svd | SU2 | s6 | 9714.00 9689.79 9722.96 9662.45 | 9647.17 9695.44 9584.38 9671.27 | 1.004 | 3 → 0 | 3 → 3 |

Every warm sector-structure lookup of a derived space is gone and the
degeneracy lookups are unchanged. The time differences are within run-to-run
spread except the small U(1) contract cases (about 2–4%); one lookup is small
next to the dense kernels of these shapes.
