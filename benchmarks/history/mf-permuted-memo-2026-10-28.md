# Multiplicity-free permuted-destination memo (before/after)

Date 2026-10-28. Base `origin/main` `e3a37e91` against the change that lets
multiplicity-free permutes (direct and lazy-adjoint operands) read and publish
the permuted-destination memo of the source's `DegeneracyStructure` entry,
which only checked Generic used before. This is the next leaf named at the end
of [`cache-hit-striping-2026-10-28.md`](cache-hit-striping-2026-10-28.md).

The memo's first four keys per entry live in set-once slots, which are read
with one acquire load each. A hit looks up the destination's entry with a
borrowed key, and the hit returns that entry's canonical HomSpace.

**Setup:**
- **Host:** Apple M4 Max (12 performance + 4 efficiency cores), macOS 15.5.
- **Build:** rustc 1.99.1, `cargo bench --locked -p symtn-network --bench
  cache_scaling --no-run`, `CARGO_PROFILE_RELEASE_DEBUG=0`, default features.
  `Cargo.lock` sha256 prefix `eaec5d66265e35ab`. Binary sha256 prefixes: base
  `4bc9e4ff276402c7`, change `eb55060069157cec`.
- **Method:** the probe's warm rows, 500 ms windows, `RAYON_NUM_THREADS=1`.
  - Single thread: every warm row, with passes ordered A B B A A B B A. The
    table shows the median of the four passes per variant.
  - Threads: `permute U1 s3 own warm` at 1, 4, 8, 12, 16 and 32 threads,
    passes A B B A.
  - Lookups per op are `(hits + misses) / ops` of the row's own counters.
  - Mac timings are informative only. They are not a scaling verdict for the
    96-core lg nodes.

## Single thread (ops/s, median of 4 passes)

| scenario | sym | case | runtime | base | change | change/base | sector lookups/op | degeneracy lookups/op |
|---|---|---|---|---:|---:|---:|---|---|
| permute | U1 | s3 | own | 8.16e+05 | 1.20e+06 | 1.47 | 1 → 0 | 1 → 1 |
| permute | U1 | s3 | shared | 8.09e+05 | 1.20e+06 | 1.48 | 1 → 0 | 1 → 1 |
| permute | U1 | s6 | own | 2.31e+05 | 2.61e+05 | 1.13 | 1 → 0 | 1 → 1 |
| permute | U1 | s6 | shared | 2.31e+05 | 2.60e+05 | 1.12 | 1 → 0 | 1 → 1 |
| permute | SU2 | s3 | own | 9.29e+04 | 1.00e+05 | 1.08 | 1 → 0 | 1 → 1 |
| permute | SU2 | s3 | shared | 9.22e+04 | 1.02e+05 | 1.10 | 1 → 0 | 1 → 1 |
| permute | SU2 | s6 | own | 3.51e+03 | 3.48e+03 | 0.99 | 1 → 0 | 1 → 1 |
| permute | SU2 | s6 | shared | 3.51e+03 | 3.47e+03 | 0.99 | 1 → 0 | 1 → 1 |
| contract | U1 | s3 | own | 2.12e+05 | 2.55e+05 | 1.20 | 3 → 1 | 3 → 3 |
| contract | U1 | s3 | shared | 2.10e+05 | 2.51e+05 | 1.19 | 3 → 1 | 3 → 3 |
| contract | U1 | s6 | own | 1.14e+05 | 1.26e+05 | 1.11 | 3 → 1 | 3 → 3 |
| contract | U1 | s6 | shared | 1.14e+05 | 1.26e+05 | 1.10 | 3 → 1 | 3 → 3 |
| contract | SU2 | s3 | own | 6.19e+04 | 6.90e+04 | 1.11 | 3 → 1 | 3 → 3 |
| contract | SU2 | s3 | shared | 6.20e+04 | 6.83e+04 | 1.10 | 3 → 1 | 3 → 3 |
| contract | SU2 | s6 | own | 3.44e+03 | 3.46e+03 | 1.01 | 3 → 1 | 3 → 3 |
| contract | SU2 | s6 | shared | 3.45e+03 | 3.46e+03 | 1.00 | 3 → 1 | 3 → 3 |
| network | U1 | s3 | own | 1.65e+05 | 1.84e+05 | 1.12 | 4 → 2 | 4 → 4 |
| network | U1 | s3 | shared | 1.65e+05 | 1.87e+05 | 1.13 | 4 → 2 | 4 → 4 |
| network | U1 | s6 | own | 7.54e+04 | 8.03e+04 | 1.06 | 4 → 2 | 4 → 4 |
| network | U1 | s6 | shared | 7.51e+04 | 7.96e+04 | 1.06 | 4 → 2 | 4 → 4 |
| network | SU2 | s3 | own | 3.90e+04 | 4.04e+04 | 1.04 | 4 → 2 | 4 → 4 |
| network | SU2 | s3 | shared | 3.91e+04 | 4.03e+04 | 1.03 | 4 → 2 | 4 → 4 |
| network | SU2 | s6 | own | 1.77e+03 | 1.75e+03 | 0.99 | 4 → 2 | 4 → 4 |
| network | SU2 | s6 | shared | 1.78e+03 | 1.77e+03 | 0.99 | 4 → 2 | 4 → 4 |

The degeneracy lookup that remains on a hit is the residency check of the
destination's own entry. The Generic memo makes the same check.

## Threads: `permute U1 s3 own warm` (ops/s)

| threads | base (2 passes) | change (2 passes) | change/base (medians) |
|---:|---|---|---:|
| 1 | 8.22e+05, 7.88e+05 | 1.18e+06, 1.19e+06 | 1.47 |
| 4 | 1.32e+06, 1.44e+06 | 2.31e+06, 1.74e+06 | 1.47 |
| 8 | 1.92e+06, 1.81e+06 | 2.51e+06, 2.47e+06 | 1.34 |
| 12 | 2.03e+06, 2.05e+06 | 2.25e+06, 2.32e+06 | 1.12 |
| 16 | 2.06e+06, 1.96e+06 | 2.20e+06, 2.10e+06 | 1.07 |
| 32 | 2.13e+06, 2.10e+06 | 2.27e+06, 2.23e+06 | 1.06 |

## Findings

1. **Single thread: the gain matches the profile.** A warm U1 s3 permute is
   47–48% faster. The probe predicted 30–40%, from the 42% share of
   `stage_transform`. A warm permute does no sector lookup, and a warm
   contraction or network does two fewer, because each transformed operand
   also reads the memo. Rows dominated by dense work (s6, SU2 s6) do not
   change.
2. **Threads: the read path must write nothing shared.** A first version read
   the memo through the entry's `RwLock` map. It also cloned the memoized id
   and an owned cache key. It was 16–27% slower than base from 12 threads up.

   Two attribution builds of that version, ops/s at 1/8/16/32 threads:
   - **Lookup without using the answer:** 7.8e5/1.54e6/1.39e6/1.34e6. Base is
     8.2e5/1.87e6/2.0e6/2.05e6.
   - **Entry upgrade only:** 7.7e5/1.71e6/1.81e6/1.89e6.

   The lock read was the main cost. The fix:
   - **Slots:** the first four keys sit in set-once slots, read with an
     acquire load each.
   - **Borrowed key:** the destination lookup uses a borrowed key.
   - **No extra clones:** the hit returns the resident entry's canonical
     HomSpace, so it touches no reference count beyond the cache lookup's.

   With these, the change is at or above base at every thread count. The
   `Weak::upgrade` of the source entry remains. Avoiding it would need a
   strong link from the structure to the memo, and that link would keep
   serving the memo after `clear`.
3. **Slot count.** Logging every publication across the `symtn` and
   `symtn-tensors` test suites gave the following keys per entry:

   | Keys per entry | Share of memoizing entries |
   |---|---:|
   | 1 | 63% |
   | 2 | 14% |
   | at most 4 | 91% |
   | at most 8 | 96% |

   The tail comes from exhaustive permutation tests. Every `cache_scaling`
   entry, cold and warm, holds one key. Four slots cover the common case at
   half the fixed inline size of eight, and further keys overflow to the map.
