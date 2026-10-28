# Warm fusion-aware trace from the completed-transformer cache

Date 2026-10-28. Host: Apple M4 Max, macOS 15.5, rustc 1.99.1.

- **Before:** `origin/main` `59c82d54`.
- **After:** the compiled-trace entry of the completed-transformer cache
  (branch `lookup-granularity`, on `59c82d54`).
- **Build:** one Cargo lock (the repository's), both revisions built
  `--release` (`debug = 0`) as path dependencies of one scratch probe, with
  `symtn/cgc-gen`.
- **Method:** a warm `TensorMap::trace_pairs(&[(0, 2)])` on
  `[l0, l1] ← [l0, l3]`, `l_k` with `s` sectors (U(1) charges centred on 0,
  SU(2) spins `0..s/2`) and degeneracies `2 + (k + i) % 3`; Runtime
  `dense_threads(1)`, `recoupling_threads(1)`. Per case: 20 warm-up calls,
  then the median of 7 windows of 200 calls. Order before, after, after,
  before, twice (ABBA ×2). No other benchmark ran meanwhile.
- Timings are observations only; the change rests on its lookup count
  (`symtn/tests/trace_compiled_cache.rs`).

## Structural change

Before, every warm trace re-lowered its terms and re-assembled its
descriptor: Simple and checked-Generic fusion looked up every source fusion
group in the composed-coefficient cache (G lookups per call, 552 for SU(2)
`s = 6`); Unique fusion recomposed one phase, split and term per source
block. Now the compiled trace over canonical structures is one
completed-transformer entry, and a warm call is one lookup that rebinds it.

## Results (µs per call, median of the window medians)

| case | B | before (4 runs) | after (4 runs) | before / after |
|---|---|---|---|---|
| U(1) s3 | 19 | 10.84 10.97 10.84 10.88 | 1.62 1.64 1.63 1.67 | 6.6 |
| U(1) s6 | 146 | 68.70 68.47 68.02 68.14 | 3.87 4.09 3.84 3.97 | 17.4 |
| SU(2) s3 | 46 | 7.58 7.79 7.65 7.58 | 2.68 2.66 2.64 2.65 | 2.9 |
| SU(2) s6 | 979 | 81.72 83.10 81.49 76.51 | 10.28 10.07 9.74 9.94 | 8.1 |

The remaining warm cost is the destination space (one sector and one
degeneracy lookup), the axis admission and the dense replay.
