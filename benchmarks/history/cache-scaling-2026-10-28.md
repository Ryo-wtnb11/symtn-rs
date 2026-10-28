# Structure-cache thread scaling (probe)

Date 2026-10-28. Revision `origin/main` `5c51b1bd` plus the probe harness
`symtn-network/benches/cache_scaling.rs`; no production code changed.

- **Host:** lg cluster `CPUcluster010`, PBS `workq`, `select=1:ncpus=96`,
  `place=excl`; 2 × Intel Xeon Platinum 9242 (96 cores, 1 thread per core,
  4 NUMA nodes); `CLK_TCK=100`.
- **Build:** rustc/cargo 1.96.0, `cargo bench --locked -p symtn-network
  --bench cache_scaling --no-run`, `CARGO_PROFILE_RELEASE_DEBUG=0`, default
  features (`cpu-faer`). Binary sha256 prefix `24a4f1ac970e7444`, `Cargo.lock`
  sha256 prefix `9a6679ce2e67ec65`. Feature unification gives `quick_cache`
  0.7.0 the features `custom-hasher` and `sharded-lock` (requested by
  fusion-symbols 0.9.0, not by symtn-core), so its shard lock is crossbeam's
  `ShardedLock`.
- **Threads:** every worker is a `std::thread::scope` thread with its own
  operation; every Runtime is `dense_threads(1).recoupling_threads(1)`;
  `RAYON_NUM_THREADS=1`. Requested counts 1, 2, 4, 8, 16, 32, 48, 96;
  `active_threads` (workers that completed an operation) equals the request
  in every row. `os_threads` is the request + 1 (the timing thread) except
  in ten 1-thread `cold_disjoint` rows, each sampled just after a 96-thread
  row, where 7–50 exiting threads of that row were still listed; their one
  worker ran alone.
- **Whole run (`/usr/bin/time -v`, run 2):** 1:50:42 wall, 2058 % CPU,
  2.39 GB maximum RSS.
- **Method:** warm rows loop for 500 ms on equal cache keys; cold rows clear
  every structure cache (and the plan cache) and then run 64 operations per
  thread (`cold_disjoint`: thread-private degeneracies; `cold_shared`: the same
  64 spaces in every thread). `cpu_s` is the process user+system time of the
  window (`/proc/self/stat`), CPU% = 100 · `cpu_s`/`wall_s` (96 cores = 9600%).
- **Raw results:**
  - `cache-scaling-2026-10-28-run2.csv`: the complete matrix, one cold
    window per row (`CACHE_SCALING_COLD_ROUNDS=1`), under `/usr/bin/time -v`
    (`cache-scaling-2026-10-28-run2-time.txt`).
  - `cache-scaling-2026-10-28-run1-partial.csv`: an earlier pass with three
    cold windows per row, stopped during SU(2) `s6` cold rows because those
    rows exceeded the walltime budget; its completed rows agree with run 2.
  - `cache-scaling-2026-10-28-stacks-*.txt.gz`: two `eu-stack` samples each of
    `contract U1 s3`, `permute SU2 s3` and `permute U1 s3` warm at 96 threads
    (12 s windows).
- Timings are observations; the conclusions below rest on the structural
  causes the stack samples and CPU% identify.

## Caches

| cache | owner | concurrency | key | bound | stats | reset | hot callers |
|---|---|---|---|---|---|---|---|
| `SectorStructure` | `symtn-core` `fusion_space/layout_cache.rs:sector_structure_cache` | `StructureCache`: one `quick_cache::sync::Cache` shard (`ShardedLock` via feature unification), a process-global `last` front behind a `std::sync::RwLock`, a `publication` `RwLock`, unpadded `AtomicU64` counters. Single-flight on `get_or_try_build`; none on `publish` | `FusionTreeHomSpaceCacheKey` (rule identity, generic flag, HomSpace sector content) | 64 MiB charged bytes; largest entry 97 % of it | `StructureCacheInfo`: entries, charged bytes, budget, max entry, hits, misses, admissions, evictions, rejections | `symtn::cache::clear` (reset lock + odd epoch), `configure_budgets` | `hom_space.rs:FusionTreeHomSpace::prepare_fusion_tree_layout_with` (get), `hom_space.rs:generic_sector_layout` (get_or_try_build), `layout.rs:PreparedFusionTreeLayout::commit_layout` (publish) |
| `DegeneracyStructure` | `symtn-core` `layout_cache.rs:degeneracy_structure_cache` | as above, `publish` only; each entry adds a `RwLock<FxHashMap>` permuted-destination memo and charges lazily via `add_charge` (shard write) | `CompleteHomSpaceStructureCacheKey` | as above | as above | as above | `complete_hom_space_structure_cached` from `hom_space.rs` (5 sites), `block_structure/structure.rs` (2), `layout.rs` (2); `admit_complete_hom_space_structure` |
| `CompletedTreeTransformer` | `symtn-tensors` `tree_transform/cache.rs:completed_transformers` | as above, `publish` only (no caller parks on another's build, #2014) | `CompletedTransformerKey` (rule, mode, coefficient type, operation, orientation, basis order, conjugation, destination/source content ids) | as above | as above | registered with the core registry | `tree_transform/cache.rs:lookup`, `resolve`, `TreeTransformPlanning::resolve_tree_pair` |
| `TreeTransformCoefficients` | `symtn-tensors` `tree_transform/cache/coefficients.rs:coefficient_groups` | as above, `publish` only, batched after commit | `CoefficientGroupKey` (group context, external-sector key, source tree pairs, precomputed hash); Unique fusion never enters | as above | as above | registered | `coefficients.rs` group `get`, `PendingCoefficientGroups::publish` |
| plan cache | `symtn-network` `plancache/mod.rs:with_plan_cache` | per Runtime, inside the Runtime's `Mutex<ExtensionSlot>`; `lru::LruCache` | `NetworkTopology` (labels, conj, splits, output, optimizer) | entry capacity + workspace byte budget | `PlanCacheStats` | `clear_plan_cache` | `Network::contract` |

Shared state on the same paths that is not a cache: `CORE_RESET_EPOCH`
(SeqCst load per admission), `BLOCK_STRUCTURE_CONTENT_ID` (one `fetch_add`
per new structure), the Runtime `context_pool` `Mutex` (per Runtime), and
tenferro-cpu's process-global CPU-set `ResourceArbiter`, which every
symtn Runtime enters with the whole process CPU set (finding 2).

## Results (run 2)

Speedup is aggregate throughput over the 1-thread throughput of the same row;
CPU% is at the largest thread count the row reached.

| scenario | sym | case | blocks | runtime | phase | ops/s @1 | ×2 | ×4 | ×8 | ×16 | ×32 | ×48 | ×96 | CPU% @96 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_front | raw | k4096 | 0 | own | warm | 3.4e+07 | 0.17 | 0.09 | 0.09 | 0.09 | 0.08 | 0.10 | 0.10 | 8477 |
| raw_shard | raw | k4096 | 0 | own | warm | 2.3e+07 | 0.05 | 0.07 | 0.11 | 0.12 | 0.14 | 0.15 | 0.16 | 9232 |
| raw_spread | raw | k4096 | 0 | own | warm | 2.24e+07 | 0.29 | 0.25 | 0.13 | 0.12 | 0.12 | 0.13 | 0.13 | 7848 |
| raw_publish | raw | fresh | 0 | own | cold | 1.23e+06 | 1.45 | 1.61 | 1.75 | 1.91 | 2.05 | 2.19 | 2.32 | 9023 |
| permute | U1 | s3 | 19 | own | warm | 3.7e+05 | 0.47 | 0.63 | 0.76 | 0.65 | 0.52 | 0.52 | 0.46 | 9550 |
| permute | U1 | s3 | 19 | own | cold_disjoint | 1.52e+04 | 2.27 | 3.13 | 5.13 | 6.81 | 7.24 | 7.90 | 7.58 | 7696 |
| permute | U1 | s3 | 19 | own | cold_shared | 2.01e+04 | 1.96 | 2.74 | 3.99 | 6.45 | 9.20 | 10.68 | 10.03 | 9256 |
| permute | U1 | s3 | 19 | shared | warm | 3.49e+05 | 1.31 | 0.91 | 1.15 | 1.35 | 1.45 | 1.48 | 1.69 | 9317 |
| permute | U1 | s3 | 19 | shared | cold_disjoint | 1.83e+04 | 1.40 | 2.83 | 4.86 | 6.26 | 6.29 | 6.55 | 6.46 | 8442 |
| permute | U1 | s3 | 19 | shared | cold_shared | 1.97e+04 | 1.02 | 2.34 | 4.28 | 6.67 | 10.48 | 12.41 | 12.97 | 8945 |
| permute | U1 | s6 | 146 | own | warm | 9.15e+04 | 1.61 | 2.36 | 4.49 | 6.30 | 7.59 | 8.51 | 9.13 | 9562 |
| permute | U1 | s6 | 146 | own | cold_disjoint | 3.34e+03 | 2.21 | 4.14 | 6.23 | 13.48 | 21.44 | 26.18 | 26.10 | 6989 |
| permute | U1 | s6 | 146 | own | cold_shared | 3.64e+03 | 2.11 | 3.82 | 6.44 | 8.09 | 16.00 | 22.08 | 11.42 | 9464 |
| permute | U1 | s6 | 146 | shared | warm | 8.95e+04 | 1.79 | 2.22 | 3.24 | 4.59 | 5.34 | 5.34 | 5.25 | 9254 |
| permute | U1 | s6 | 146 | shared | cold_disjoint | 3.52e+03 | 2.12 | 4.02 | 6.18 | 11.64 | 19.10 | 24.90 | 23.90 | 7598 |
| permute | U1 | s6 | 146 | shared | cold_shared | 3.49e+03 | 2.23 | 3.28 | 6.04 | 11.76 | 17.06 | 27.57 | 40.98 | 9347 |
| contract | U1 | s3 | 19 | own | warm | 8.99e+04 | 1.52 | 1.24 | 0.67 | 0.33 | 0.18 | 0.12 | 0.06 | 2351 |
| contract | U1 | s3 | 19 | own | cold_disjoint | 1.13e+04 | 1.95 | 3.35 | 5.31 | 2.76 | 1.37 | 0.99 | 0.54 | 2070 |
| contract | U1 | s3 | 19 | own | cold_shared | 1.31e+04 | 1.88 | 2.48 | 4.40 | 2.08 | 1.16 | 0.80 | 0.43 | 2663 |
| contract | U1 | s3 | 19 | shared | warm | 8.53e+04 | 1.48 | 1.05 | 0.90 | 0.34 | 0.19 | 0.12 | 0.07 | 2532 |
| contract | U1 | s3 | 19 | shared | cold_disjoint | 1.16e+04 | 1.88 | 2.66 | 4.82 | 2.54 | 1.35 | 0.94 | 0.52 | 2097 |
| contract | U1 | s3 | 19 | shared | cold_shared | 1.3e+04 | 1.90 | 2.36 | 3.92 | 2.05 | 1.16 | 0.79 | 0.43 | 2430 |
| contract | U1 | s6 | 146 | own | warm | 4.62e+04 | 1.69 | 1.74 | 1.52 | 0.58 | 0.33 | 0.22 | 0.12 | 2013 |
| contract | U1 | s6 | 146 | own | cold_disjoint | 2.51e+03 | 2.31 | 4.13 | 6.33 | 7.13 | 8.06 | 8.36 | 2.14 | 2190 |
| contract | U1 | s6 | 146 | own | cold_shared | 3.06e+03 | 1.96 | 3.30 | 5.69 | 11.62 | 5.00 | 3.24 | 1.81 | 2287 |
| contract | U1 | s6 | 146 | shared | warm | 4.61e+04 | 1.64 | 1.74 | 1.40 | 0.56 | 0.32 | 0.21 | 0.12 | 1982 |
| contract | U1 | s6 | 146 | shared | cold_disjoint | 2.58e+03 | 2.24 | 3.71 | 4.57 | 7.35 | 8.29 | 5.58 | 2.19 | 2237 |
| contract | U1 | s6 | 146 | shared | cold_shared | 3.18e+03 | 1.91 | 3.11 | 5.90 | 9.40 | 4.62 | 3.31 | 1.71 | 2231 |
| network | U1 | s3 | 19 | own | warm | 7.13e+04 | 1.57 | 1.24 | 1.40 | 0.44 | 0.23 | 0.15 | 0.08 | 2333 |
| network | U1 | s3 | 19 | own | cold_disjoint | 6.52e+03 | 2.04 | 3.68 | 5.09 | 7.75 | 2.63 | 1.84 | 0.94 | 2418 |
| network | U1 | s3 | 19 | own | cold_shared | 7e+03 | 2.14 | 2.79 | 4.82 | 4.61 | 2.23 | 1.49 | 0.80 | 2334 |
| network | U1 | s3 | 19 | shared | warm | 7.11e+04 | 0.70 | 1.08 | 1.17 | 0.41 | 0.22 | 0.15 | 0.08 | 2153 |
| network | U1 | s3 | 19 | shared | cold_disjoint | 6.74e+03 | 1.91 | 3.33 | 4.87 | 6.66 | 2.49 | 1.64 | 0.91 | 2211 |
| network | U1 | s3 | 19 | shared | cold_shared | 7.32e+03 | 1.98 | 2.81 | 4.54 | 4.03 | 2.11 | 1.45 | 0.77 | 2192 |
| network | U1 | s6 | 146 | own | warm | 3.51e+04 | 0.80 | 1.85 | 2.60 | 0.85 | 0.46 | 0.32 | 0.16 | 2028 |
| network | U1 | s6 | 146 | own | cold_disjoint | 1.48e+03 | 2.02 | 4.07 | 6.51 | 6.42 | 7.78 | 5.46 | 5.56 | 8890 |
| network | U1 | s6 | 146 | own | cold_shared | 1.6e+03 | 2.06 | 3.61 | 6.71 | 11.70 | 11.36 | 6.66 | 3.41 | 1949 |
| network | U1 | s6 | 146 | shared | warm | 3.56e+04 | 1.55 | 1.68 | 2.31 | 0.80 | 0.44 | 0.30 | 0.16 | 1996 |
| network | U1 | s6 | 146 | shared | cold_disjoint | 1.5e+03 | 2.00 | 3.18 | 5.71 | 6.85 | 7.77 | 8.78 | 5.16 | 8746 |
| network | U1 | s6 | 146 | shared | cold_shared | 1.63e+03 | 2.03 | 3.57 | 6.47 | 11.72 | 10.88 | 6.54 | 3.38 | 1941 |
| permute | SU2 | s3 | 46 | own | warm | 3.52e+04 | 0.98 | 0.44 | 0.17 | 0.10 | 0.06 | 0.04 | 0.02 | 2305 |
| permute | SU2 | s3 | 46 | own | cold_disjoint | 5.72e+03 | 1.80 | 1.81 | 1.08 | 0.61 | 0.37 | 0.25 | 0.13 | 2404 |
| permute | SU2 | s3 | 46 | own | cold_shared | 6.55e+03 | 1.32 | 2.43 | 1.97 | 1.08 | 0.38 | 0.22 | 0.11 | 2376 |
| permute | SU2 | s3 | 46 | shared | warm | 3.52e+04 | 0.83 | 0.73 | 0.36 | 0.21 | 0.07 | 0.04 | 0.02 | 2481 |
| permute | SU2 | s3 | 46 | shared | cold_disjoint | 5.24e+03 | 1.84 | 2.89 | 3.11 | 1.32 | 0.48 | 0.27 | 0.14 | 2473 |
| permute | SU2 | s3 | 46 | shared | cold_shared | 6.5e+03 | 1.29 | 2.40 | 1.95 | 1.05 | 0.37 | 0.22 | 0.12 | 2459 |
| permute | SU2 | s6 | 979 | own | warm | 1.36e+03 | 0.71 | 0.58 | 0.28 | 0.17 | 0.06 | 0.03 | 0.02 | 2293 |
| permute | SU2 | s6 | 979 | own | cold_disjoint | 324 | 1.63 | 2.03 | 0.98 | 0.61 | 0.23 | 0.15 | 0.07 | 2386 |
| permute | SU2 | s6 | 979 | own | cold_shared | 327 | 1.24 | 1.89 | 0.74 | 0.44 | 0.21 | 0.13 | 0.07 | 2425 |
| permute | SU2 | s6 | 979 | shared | warm | 1.35e+03 | 0.73 | 0.58 | 0.28 | 0.17 | 0.05 | 0.03 | 0.02 | 2358 |
| permute | SU2 | s6 | 979 | shared | cold_disjoint | 319 | 1.72 | 2.16 | 0.90 | 0.45 | 0.22 | 0.14 | 0.07 | 2480 |
| permute | SU2 | s6 | 979 | shared | cold_shared | 318 | 1.17 | 1.87 | 1.23 | 0.58 | 0.21 | 0.14 | 0.07 | 2526 |
| contract | SU2 | s3 | 46 | own | warm | 2.25e+04 | 1.53 | 1.03 | 0.51 | 0.29 | 0.11 | 0.06 | 0.03 | 2522 |
| contract | SU2 | s3 | 46 | own | cold_disjoint | 4.22e+03 | 1.99 | 2.98 | 3.45 | 1.40 | 0.55 | 0.31 | 0.16 | 2549 |
| contract | SU2 | s3 | 46 | own | cold_shared | 4.99e+03 | 1.38 | 1.57 | 2.18 | 1.19 | 0.47 | 0.26 | 0.13 | 2299 |
| contract | SU2 | s3 | 46 | shared | warm | 2.26e+04 | 1.46 | 0.99 | 0.51 | 0.29 | 0.10 | 0.05 | 0.03 | 2371 |
| contract | SU2 | s3 | 46 | shared | cold_disjoint | 3.87e+03 | 2.20 | 2.96 | 3.78 | 1.52 | 0.57 | 0.33 | 0.17 | 2414 |
| contract | SU2 | s3 | 46 | shared | cold_shared | 4.99e+03 | 1.34 | 1.57 | 2.14 | 1.18 | 0.43 | 0.27 | 0.13 | 2343 |
| contract | SU2 | s6 | 979 | own | warm | 1.28e+03 | 0.77 | 0.61 | 0.29 | 0.18 | 0.06 | 0.03 | 0.02 | 2289 |
| contract | SU2 | s6 | 979 | own | cold_disjoint | 292 | 1.64 | 2.36 | 0.80 | 0.57 | 0.23 | 0.14 | 0.07 | 2281 |
| contract | SU2 | s6 | 979 | own | cold_shared | 296 | 1.19 | 2.32 | 0.80 | 0.55 | 0.22 | 0.14 | 0.07 | 2402 |
| contract | SU2 | s6 | 979 | shared | warm | 1.27e+03 | 0.77 | 0.60 | 0.30 | 0.17 | 0.05 | 0.04 | 0.02 | 2285 |
| contract | SU2 | s6 | 979 | shared | cold_disjoint | 295 | 1.66 | 2.66 | 0.96 | 0.59 | 0.22 | 0.14 | 0.07 | 2419 |
| contract | SU2 | s6 | 979 | shared | cold_shared | 296 | 1.20 | 1.76 | 1.24 | 0.65 | 0.24 | 0.15 | 0.08 | 2447 |
| network | SU2 | s3 | 46 | own | warm | 1.47e+04 | 1.08 | 0.81 | 0.41 | 0.24 | 0.08 | 0.05 | 0.02 | 2220 |
| network | SU2 | s3 | 46 | own | cold_disjoint | 2.48e+03 | 2.00 | 2.76 | 2.63 | 1.29 | 0.48 | 0.27 | 0.15 | 2421 |
| network | SU2 | s3 | 46 | own | cold_shared | 2.79e+03 | 1.31 | 2.21 | 1.88 | 1.18 | 0.42 | 0.24 | 0.13 | 2394 |
| network | SU2 | s3 | 46 | shared | warm | 1.42e+04 | 1.08 | 0.83 | 0.42 | 0.25 | 0.09 | 0.06 | 0.03 | 2502 |
| network | SU2 | s3 | 46 | shared | cold_disjoint | 2.5e+03 | 1.83 | 1.61 | 1.46 | 0.66 | 0.40 | 0.27 | 0.14 | 2462 |
| network | SU2 | s3 | 46 | shared | cold_shared | 2.8e+03 | 1.30 | 2.07 | 1.85 | 1.15 | 0.41 | 0.25 | 0.13 | 2380 |
| network | SU2 | s6 | 979 | own | warm | 651 | 0.73 | 0.59 | 0.29 | 0.17 | 0.05 | 0.03 | 0.02 | 2410 |
| network | SU2 | s6 | 979 | own | cold_disjoint | 155 | 1.57 | 1.63 | 0.97 | 0.42 | 0.22 | 0.14 | 0.07 | 2483 |
| network | SU2 | s6 | 979 | own | cold_shared | 155 | 1.17 | 1.63 | 0.85 | 0.61 | 0.21 | 0.14 | 0.07 | 2468 |
| network | SU2 | s6 | 979 | shared | warm | 657 | 0.72 | 0.59 | 0.29 | 0.17 | 0.06 | 0.04 | 0.02 | 2501 |
| network | SU2 | s6 | 979 | shared | cold_disjoint | 156 | 1.40 | 1.73 | 0.70 | 0.46 | 0.22 | 0.14 | 0.07 | 2507 |
| network | SU2 | s6 | 979 | shared | cold_shared | 155 | 1.20 | 1.61 | 1.03 | 0.58 | 0.21 | 0.14 | 0.07 | 2470 |

## Findings

1. **Warm hits do not scale, and the cause is symtn's wrapper.** A raw
   `StructureCache::get` runs at 34 M/s on one thread and at 2.5–3 M/s
   aggregate from 2 to 96 threads, with CPU% ≈ 100 · threads (busy, not
   parked). The plateau is one contended cache line: every hit takes the
   process-global `last` front's `std::sync::RwLock` read (two atomic
   read-modify-writes on one word) and `fetch_add`s the unpadded `hits`
   counter. `raw_spread` (distinct values, front missed) collapses the same
   way, so the value `Arc` refcount is not the cause; the quick_cache shard
   read is per-thread sharded. In operations, U(1) `s3` permute (four cache
   lookups per call, little dense work) is slower at 96 threads than at one
   (0.46×) at 95 % CPU per core; the largest stack bucket of its workers is
   `StructureCache::get`/`contains`. Larger dense work per call hides it
   (U(1) `s6` permute, 9×).
2. **Every operation that reaches the tenferro CPU GEMM is serialized by
   the CPU configuration symtn's Runtime chooses, not by a symtn cache.**
   `contract`, `network` and SU(2) `permute` peak at 2–8 threads and fall to
   0.02–0.16× at 96 threads with mostly 20–25 % CPU per core (threads
   parked); 95/96, 60/62, 96/96 and 55/55 sampled workers wait in
   `tenferro_cpu::arbiter::ResourceArbiter::acquire_request_recovering` under
   `CpuBackend::acquire_execution_permit` (`grouped_gemm_cached` ←
   `symtn_dense::DefaultDenseExecutor::matmul_batch_axpby_into`).
   tenferro-cpu 0.7.1's arbiter admits concurrent executions whose CPU
   requests do not overlap (`arbiter.rs:ResourceRequest::conflicts_with`).
   Every symtn Runtime builds its GEMM and factorization backends with
   `CpuBackend::from_context` (`symtn/src/runtime/lanes.rs:make_transform_ops`,
   `symtn/src/runtime.rs:mint_dense` →
   `symtn-dense/src/tenferro_adapter.rs:DefaultDenseExecutor::with_shared_context`).
   Under faer that resolves to `CpuExecutionMode::Compatibility`, whose
   permit requests the whole process-allowed CPU set even for a one-thread
   context (`backend.rs:CpuBackend::compatibility`,
   `acquire_execution_permit`). Under BLAS it resolves to
   `ProviderDefaultExclusive`, which conflicts with everything. So the
   permits of independent Runtimes always overlap, and at most one tenferro
   execution runs at a time in the process. That is tenferro's documented
   oversubscription guard for unpinned contexts. tenferro does offer
   non-conflicting configurations: cooperative `ExternalCpuDomain`s with
   disjoint declared CPU sets, or `ExternalCpuDomain::new_caller_managed`,
   which bypasses the global arbiter. symtn's Runtime API cannot select either
   for the GEMM lanes. Until it can, these rows cannot show structure-cache
   effects.
3. **Cold misses build more than once.** `DegeneracyStructure`,
   `CompletedTreeTransformer` and `TreeTransformCoefficients` admit through
   `publish`, so concurrent misses of one key all build and the first
   admission wins; a lost race is counted nowhere in `StructureCacheInfo`.
   `cold_shared` costs as much CPU per operation as `cold_disjoint` (U(1)
   `s3` permute at 96 threads: 0.33–0.46 vs 0.65–0.67 ms; U(1) `s6`:
   0.84–2.28 vs 0.80–0.82 ms) instead of dropping by the thread count, and
   CPU per operation grows 6–12× from 1 to 96 threads in both, so cold
   builds also contend on the shared words of finding 1.
4. **Admission is serialized but not limiting.** Raw admissions of fresh
   keys plateau at 2.8 M/s (2.3× at 96 threads): each takes the publication
   read lock, the single shard's write lock (all eight `ShardedLock`
   stripes) and the `last` write lock. Cold operations admit at most tens of
   thousands of entries per second, two orders below that.
5. **Plan cache:** per Runtime, under the Runtime's extension `Mutex`.
   `runtime=shared` and `runtime=own` network rows do not differ, because
   finding 2 dominates both.

## Mapping to fusion-symbols 0.9.0 techniques (`src/cache.rs`)

| finding | technique | status |
|---|---|---|
| 1, hit counter | per-tier hit counters over 16 `#[repr(align(128))]` stripes picked by a round-robin thread-local index (`PaddedCounter`, `hit_stripe`); std only | missing in `StructureCache` |
| 1, `last` front | fusion-symbols has no shared front; a hit is the quick_cache read alone | the shared `RwLock` front is symtn-only and is the contended word |
| 1, shard lock | quick_cache `sharded-lock` (`ShardedLock` read stripes) | in effect only through feature unification; symtn-core does not request it |
| 3 | single-flight through quick_cache placeholder guards; a waiter that joins an in-flight load counts as a hit in `TierStats` | only `SectorStructure` (`get_or_try_build`); the other three publish without it, and a lost race is counted nowhere |
| 4 | one shard per tier, byte-weighted | same design; already fine |
| bounds | byte budgets, evictions, rejections | already fine |
| 2 | — | not a cache: symtn's `from_context` (Compatibility) backends request the whole CPU set from tenferro-cpu's arbiter |
