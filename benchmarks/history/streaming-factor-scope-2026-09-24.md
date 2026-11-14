# One pool hop per call for the streaming factorization loops (#1389)

Date 2026-09-24. Host: Apple M4 Max (16 cores), macOS 15.5, rustc 1.96.0.

- **Before:** `origin/main` `7110daac`.
- **After:** branch `streaming-factor-scope`.
- **Build:** each revision into its own target directory with
  `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --release -p
  tenet-rs --example eager_overhead_ledger`, the ledger carrying this change's
  `lq_compact`, `left_null` and `eigh_full` rows in both builds. Binaries
  (sha256 prefix): before `e290c83a607c519b`, after `1a8a1b6545bb75a8`.
- **Method:** E1 with one op filter per run (`qr_compact` as control,
  `lq_compact`, `left_null`, `eigh_full`), all five cases, three symmetries,
  f64 and c64, `LEDGER_THREADS=one` and `default`, `nice -n 19`, two passes
  ordered before, after, before, after, each run started only when no cargo
  or rustc process was active. Medians are the mean of the two passes. Raw
  pass-1 rows: `streaming-factor-scope-2026-09-24-{one,default}-{before,after}.csv`.
- Timings are observations only.

## Change

The streaming per-block factorization loops (compact LQ direct and
matricization, the Generic and checked-Generic LQ, `eigh_full` direct and
matricization, checked-Generic eigh, the multiplicity-free and Generic SVD
matricization fallbacks, the checked-Generic SVD and polar stages, the four
null-space loops, and the checked pinv) now run inside one
`DenseExecutor::with_linalg_scope`. On `DefaultDenseExecutor` that is one
Tenferro 0.6.0 `CpuBackend::with_execution_scope`, whose sessions skip the
per-session permit and pool install. Each loop still holds one block's input
and output at a time.

## Admissions

`cpu_session_stats().admissions` (new) counts permit acquisitions.
`tenet-matrixalgebra/tests/factorization_session_scope.rs` asserts one
admission per call on every listed site at one and at default threads, while
`sessions_opened` still counts one session per dense factorization (2 to 5 on
the fixtures). With the scope disabled the same test reports admissions equal
to sessions (negative control, run by hand).

## Bitwise results and peak working set

[The original digest program](https://github.com/Ryo-wtnb11/symtn-rs/blob/3282c3a8f6451553f4ee0bf171f3293c0a05a15b/benchmarks/history/streaming-factor-scope-2026-09-24-digest.rs) runs 58 site/fixture rows
(matrixalgebra-level U(1), Generic toy and checked-Generic toy fixtures on
direct and matricization layouts, and facade U(1), SU(2), fZ2×U(1) `lq`,
`left_null`, `right_null`, `eigh` rows; f64 and c64) and prints a digest of
every output bit, the process-wide live-heap peak above the call's entry,
and the session count. Digests are identical between the revisions at one
thread and at default threads (`-digest-{before,after}-{1,default}.txt`).
Session counts are identical. At one thread the peak is unchanged or +48 B
(the scope's shared permit `Arc`) on every row. At default threads the peak is
the same except where Tenferro's pool workers allocate concurrently, which
varies run to run in both revisions (for example `c64 u1 lq direct`: 35 to
57 kB in repeated runs of either binary).

## E1, one thread

Every changed op pays one extra allocation of 48 B per call (the permit
`Arc`); `qr_compact` is unchanged. Median ratios after/before span 0.91 to 1.01 over all 180 changed-op rows.

| op | symmetry | dtype | case | before calls / B | after calls / B | before med µs | after med µs | ratio |
|---|---|---|---|---:|---:|---:|---:|---:|
| eigh_full | SU2 | c64 | r2_s8_d2 | 156 / 114098 | 157 / 114146 | 27.94 | 26.38 | 0.944 |
| eigh_full | SU2 | c64 | r4_s3_d4 | 253 / 1974001 | 254 / 1974049 | 727.52 | 716.46 | 0.985 |
| eigh_full | SU2 | f64 | r2_s8_d2 | 140 / 111074 | 141 / 111122 | 23.19 | 22.02 | 0.950 |
| eigh_full | SU2 | f64 | r4_s3_d4 | 243 / 743134 | 244 / 743182 | 465.31 | 455.62 | 0.979 |
| eigh_full | U1 | c64 | r2_s8_d2 | 156 / 114098 | 157 / 114146 | 27.31 | 26.15 | 0.957 |
| eigh_full | U1 | c64 | r4_s3_d4 | 201 / 681953 | 202 / 682001 | 276.33 | 275.85 | 0.998 |
| eigh_full | U1 | f64 | r2_s8_d2 | 140 / 111074 | 141 / 111122 | 23.50 | 22.00 | 0.936 |
| eigh_full | U1 | f64 | r4_s3_d4 | 191 / 324538 | 192 / 324586 | 196.90 | 196.65 | 0.999 |
| eigh_full | fZ2xU1 | c64 | r2_s8_d2 | 156 / 114098 | 157 / 114146 | 28.44 | 27.27 | 0.959 |
| eigh_full | fZ2xU1 | c64 | r4_s3_d4 | 201 / 681953 | 202 / 682001 | 278.21 | 277.21 | 0.996 |
| eigh_full | fZ2xU1 | f64 | r2_s8_d2 | 140 / 111074 | 141 / 111122 | 23.65 | 22.27 | 0.942 |
| eigh_full | fZ2xU1 | f64 | r4_s3_d4 | 191 / 324538 | 192 / 324586 | 197.29 | 196.00 | 0.993 |
| left_null | SU2 | c64 | r2_s8_d2 | 237 / 207300 | 238 / 207348 | 39.52 | 37.08 | 0.938 |
| left_null | SU2 | c64 | r4_s3_d4 | 347 / 3679640 | 348 / 3679688 | 1512.56 | 1510.04 | 0.998 |
| left_null | SU2 | f64 | r2_s8_d2 | 221 / 203652 | 222 / 203700 | 34.21 | 32.65 | 0.954 |
| left_null | SU2 | f64 | r4_s3_d4 | 334 / 1390271 | 335 / 1390319 | 1007.00 | 996.71 | 0.990 |
| left_null | U1 | c64 | r2_s8_d2 | 237 / 207300 | 238 / 207348 | 39.29 | 37.25 | 0.948 |
| left_null | U1 | c64 | r4_s3_d4 | 210 / 1217718 | 211 / 1217766 | 540.71 | 539.31 | 0.997 |
| left_null | U1 | f64 | r2_s8_d2 | 221 / 203652 | 222 / 203700 | 34.35 | 32.58 | 0.948 |
| left_null | U1 | f64 | r4_s3_d4 | 199 / 584767 | 200 / 584815 | 386.60 | 384.10 | 0.994 |
| left_null | fZ2xU1 | c64 | r2_s8_d2 | 237 / 207300 | 238 / 207348 | 40.15 | 38.31 | 0.954 |
| left_null | fZ2xU1 | c64 | r4_s3_d4 | 210 / 1217718 | 211 / 1217766 | 541.48 | 540.65 | 0.998 |
| left_null | fZ2xU1 | f64 | r2_s8_d2 | 221 / 203652 | 222 / 203700 | 34.77 | 32.98 | 0.948 |
| left_null | fZ2xU1 | f64 | r4_s3_d4 | 199 / 584767 | 200 / 584815 | 382.19 | 380.60 | 0.996 |
| lq_compact | SU2 | c64 | r2_s8_d2 | 125 / 108944 | 126 / 108992 | 24.52 | 22.54 | 0.919 |
| lq_compact | SU2 | c64 | r4_s3_d4 | 281 / 2815327 | 282 / 2815375 | 459.08 | 458.19 | 0.998 |
| lq_compact | SU2 | f64 | r2_s8_d2 | 125 / 107888 | 126 / 107936 | 23.73 | 21.50 | 0.906 |
| lq_compact | SU2 | f64 | r4_s3_d4 | 278 / 1073489 | 279 / 1073537 | 205.27 | 207.81 | 1.012 |
| lq_compact | U1 | c64 | r2_s8_d2 | 125 / 108944 | 126 / 108992 | 24.06 | 22.77 | 0.946 |
| lq_compact | U1 | c64 | r4_s3_d4 | 189 / 895135 | 190 / 895183 | 164.15 | 163.29 | 0.995 |
| lq_compact | U1 | f64 | r2_s8_d2 | 125 / 107888 | 126 / 107936 | 23.92 | 22.48 | 0.940 |
| lq_compact | U1 | f64 | r4_s3_d4 | 188 / 442153 | 189 / 442201 | 90.00 | 90.42 | 1.005 |
| lq_compact | fZ2xU1 | c64 | r2_s8_d2 | 125 / 108944 | 126 / 108992 | 24.58 | 23.08 | 0.939 |
| lq_compact | fZ2xU1 | c64 | r4_s3_d4 | 189 / 895135 | 190 / 895183 | 165.25 | 163.35 | 0.989 |
| lq_compact | fZ2xU1 | f64 | r2_s8_d2 | 125 / 107888 | 126 / 107936 | 24.00 | 22.21 | 0.925 |
| lq_compact | fZ2xU1 | f64 | r4_s3_d4 | 188 / 442153 | 189 / 442201 | 90.54 | 90.60 | 1.001 |
| qr_compact | SU2 | c64 | r2_s8_d2 | 129 / 112080 | 129 / 112080 | 22.42 | 23.31 | 1.040 |
| qr_compact | SU2 | c64 | r4_s3_d4 | 288 / 2752655 | 288 / 2752655 | 432.50 | 433.23 | 1.002 |
| qr_compact | SU2 | f64 | r2_s8_d2 | 129 / 111056 | 129 / 111056 | 22.15 | 22.02 | 0.994 |
| qr_compact | SU2 | f64 | r4_s3_d4 | 285 / 1043585 | 285 / 1043585 | 194.62 | 197.40 | 1.014 |
| qr_compact | U1 | c64 | r2_s8_d2 | 129 / 112080 | 129 / 112080 | 22.58 | 23.08 | 1.022 |
| qr_compact | U1 | c64 | r4_s3_d4 | 196 / 861135 | 196 / 861135 | 153.00 | 150.79 | 0.986 |
| qr_compact | U1 | f64 | r2_s8_d2 | 129 / 111056 | 129 / 111056 | 22.06 | 22.29 | 1.010 |
| qr_compact | U1 | f64 | r4_s3_d4 | 195 / 426585 | 195 / 426585 | 83.90 | 84.71 | 1.010 |
| qr_compact | fZ2xU1 | c64 | r2_s8_d2 | 129 / 112080 | 129 / 112080 | 22.96 | 23.12 | 1.007 |
| qr_compact | fZ2xU1 | c64 | r4_s3_d4 | 196 / 861135 | 196 / 861135 | 151.67 | 151.85 | 1.001 |
| qr_compact | fZ2xU1 | f64 | r2_s8_d2 | 129 / 111056 | 129 / 111056 | 22.58 | 22.40 | 0.992 |
| qr_compact | fZ2xU1 | f64 | r4_s3_d4 | 195 / 426585 | 195 / 426585 | 84.46 | 85.44 | 1.012 |
## E1, default threads

At default threads the per-block pool hops are gone: the changed ops' medians
fall 2 to 73 % (ratios 0.27 to 0.98 over 180 rows), most on many-sector small blocks (`r2_s8_d2`). The
allocation columns here are **not comparable**: the ledger counts allocations
on the calling thread only, and the scope body now runs on a Tenferro pool
worker, so its allocations leave the count. The one-thread table is the
allocation evidence.

| op | symmetry | dtype | case | before calls / B | after calls / B | before med µs | after med µs | ratio |
|---|---|---|---|---:|---:|---:|---:|---:|
| eigh_full | SU2 | c64 | r2_s8_d2 | 68 / 59570 | 28 / 3298 | 130.19 | 40.12 | 0.308 |
| eigh_full | SU2 | c64 | r4_s3_d4 | 134 / 255436 | 109 / 28764 | 1019.83 | 926.62 | 0.909 |
| eigh_full | SU2 | f64 | r2_s8_d2 | 68 / 59170 | 28 / 3154 | 127.35 | 34.25 | 0.269 |
| eigh_full | SU2 | f64 | r4_s3_d4 | 134 / 158924 | 109 / 26460 | 714.17 | 634.98 | 0.889 |
| eigh_full | U1 | c64 | r2_s8_d2 | 68 / 59570 | 28 / 3298 | 130.10 | 39.96 | 0.307 |
| eigh_full | U1 | c64 | r4_s3_d4 | 130 / 141298 | 105 / 26498 | 339.67 | 288.71 | 0.850 |
| eigh_full | U1 | f64 | r2_s8_d2 | 68 / 59170 | 28 / 3154 | 128.75 | 34.65 | 0.269 |
| eigh_full | U1 | f64 | r4_s3_d4 | 131 / 102370 | 105 / 24962 | 262.77 | 210.23 | 0.800 |
| eigh_full | fZ2xU1 | c64 | r2_s8_d2 | 68 / 59570 | 28 / 3298 | 129.75 | 40.77 | 0.314 |
| eigh_full | fZ2xU1 | c64 | r4_s3_d4 | 130 / 141298 | 105 / 26498 | 336.48 | 292.71 | 0.870 |
| eigh_full | fZ2xU1 | f64 | r2_s8_d2 | 68 / 59170 | 28 / 3154 | 128.52 | 34.17 | 0.266 |
| eigh_full | fZ2xU1 | f64 | r4_s3_d4 | 130 / 100850 | 105 / 24962 | 259.06 | 211.85 | 0.818 |
| left_null | SU2 | c64 | r2_s8_d2 | 125 / 125444 | 86 / 14196 | 134.33 | 54.52 | 0.406 |
| left_null | SU2 | c64 | r4_s3_d4 | 149 / 273044 | 125 / 201820 | 1992.02 | 1913.54 | 0.961 |
| left_null | SU2 | f64 | r2_s8_d2 | 125 / 125188 | 86 / 13940 | 132.19 | 47.17 | 0.357 |
| left_null | SU2 | f64 | r4_s3_d4 | 149 / 178836 | 125 / 107612 | 1446.92 | 1362.10 | 0.941 |
| left_null | U1 | c64 | r2_s8_d2 | 125 / 125444 | 86 / 14196 | 134.23 | 54.21 | 0.404 |
| left_null | U1 | c64 | r4_s3_d4 | 108 / 159564 | 84 / 88980 | 598.19 | 546.69 | 0.914 |
| left_null | U1 | f64 | r2_s8_d2 | 125 / 125188 | 86 / 13940 | 132.25 | 47.71 | 0.361 |
| left_null | U1 | f64 | r4_s3_d4 | 109 / 122172 | 84 / 50068 | 450.02 | 394.94 | 0.878 |
| left_null | fZ2xU1 | c64 | r2_s8_d2 | 125 / 125444 | 86 / 14196 | 134.77 | 54.75 | 0.406 |
| left_null | fZ2xU1 | c64 | r4_s3_d4 | 108 / 159564 | 84 / 88980 | 604.50 | 551.58 | 0.912 |
| left_null | fZ2xU1 | f64 | r2_s8_d2 | 125 / 125188 | 86 / 13940 | 131.58 | 47.94 | 0.364 |
| left_null | fZ2xU1 | f64 | r4_s3_d4 | 109 / 122172 | 84 / 50068 | 439.98 | 394.15 | 0.896 |
| lq_compact | SU2 | c64 | r2_s8_d2 | 37 / 58384 | 14 / 2880 | 128.60 | 35.33 | 0.275 |
| lq_compact | SU2 | c64 | r4_s3_d4 | 110 / 499612 | 96 / 464940 | 1002.42 | 898.87 | 0.897 |
| lq_compact | SU2 | f64 | r2_s8_d2 | 37 / 57840 | 14 / 2336 | 127.19 | 34.56 | 0.272 |
| lq_compact | SU2 | f64 | r4_s3_d4 | 110 / 278428 | 96 / 243756 | 731.23 | 619.75 | 0.848 |
| lq_compact | U1 | c64 | r2_s8_d2 | 37 / 58384 | 14 / 2880 | 127.85 | 35.35 | 0.277 |
| lq_compact | U1 | c64 | r4_s3_d4 | 106 / 249170 | 92 / 214498 | 221.48 | 172.02 | 0.777 |
| lq_compact | U1 | f64 | r2_s8_d2 | 37 / 57840 | 14 / 2336 | 127.79 | 34.77 | 0.272 |
| lq_compact | U1 | f64 | r4_s3_d4 | 106 / 152914 | 92 / 118242 | 149.79 | 101.94 | 0.681 |
| lq_compact | fZ2xU1 | c64 | r2_s8_d2 | 37 / 58384 | 14 / 2880 | 127.73 | 35.81 | 0.280 |
| lq_compact | fZ2xU1 | c64 | r4_s3_d4 | 106 / 249170 | 92 / 214498 | 221.25 | 171.87 | 0.777 |
| lq_compact | fZ2xU1 | f64 | r2_s8_d2 | 37 / 57840 | 14 / 2336 | 127.46 | 34.88 | 0.274 |
| lq_compact | fZ2xU1 | f64 | r4_s3_d4 | 106 / 152914 | 92 / 118242 | 151.50 | 102.15 | 0.674 |
| qr_compact | SU2 | c64 | r2_s8_d2 | 41 / 61520 | 41 / 61520 | 34.56 | 34.58 | 1.001 |
| qr_compact | SU2 | c64 | r4_s3_d4 | 117 / 436940 | 117 / 436940 | 868.10 | 866.50 | 0.998 |
| qr_compact | SU2 | f64 | r2_s8_d2 | 41 / 61008 | 41 / 61008 | 33.08 | 33.10 | 1.001 |
| qr_compact | SU2 | f64 | r4_s3_d4 | 117 / 248524 | 117 / 248524 | 600.81 | 594.94 | 0.990 |
| qr_compact | U1 | c64 | r2_s8_d2 | 41 / 61520 | 41 / 61520 | 34.58 | 33.94 | 0.981 |
| qr_compact | U1 | c64 | r4_s3_d4 | 113 / 215170 | 113 / 215170 | 167.04 | 166.12 | 0.995 |
| qr_compact | U1 | f64 | r2_s8_d2 | 41 / 61008 | 41 / 61008 | 33.12 | 32.73 | 0.988 |
| qr_compact | U1 | f64 | r4_s3_d4 | 113 / 137346 | 113 / 137346 | 100.31 | 100.92 | 1.006 |
| qr_compact | fZ2xU1 | c64 | r2_s8_d2 | 41 / 61520 | 41 / 61520 | 35.54 | 34.77 | 0.978 |
| qr_compact | fZ2xU1 | c64 | r4_s3_d4 | 113 / 215170 | 113 / 215170 | 168.54 | 167.15 | 0.992 |
| qr_compact | fZ2xU1 | f64 | r2_s8_d2 | 41 / 61008 | 41 / 61008 | 34.10 | 33.58 | 0.985 |
| qr_compact | fZ2xU1 | f64 | r4_s3_d4 | 113 / 137346 | 113 / 137346 | 100.83 | 100.98 | 1.001 |
## Disclosed cost

The scope holds the process-wide execution permit for the whole call, including
TeNeT's own between-block gauge, adjoint copies, scatter and validation. Under
default threads another thread's Tenferro call now waits for that work as well
as for the kernels. The body also runs on a Tenferro pool worker (on Linux
even at one thread, where Tenferro's managed engine keeps a worker pool), so
thread-local state read inside it belongs to that worker; the factorization
layer's test probes are therefore moved into and out of the body by
`in_linalg_scope`.

Retrieve the benchmark source for the recorded revision with:

```sh
git show 3282c3a8f6451553f4ee0bf171f3293c0a05a15b:benchmarks/history/streaming-factor-scope-2026-09-24-digest.rs
```
