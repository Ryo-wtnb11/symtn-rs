# Structure-cache hit striping (before/after)

Date 2026-10-28. Base `origin/main` `9dcf91e7` against the change for
finding 1 of [`cache-scaling-2026-10-28.md`](cache-scaling-2026-10-28.md):

- the hit/miss/admission/eviction/rejection counters are striped over 16
  `#[repr(align(128))]` stripes, as fusion-symbols 0.9.1 stripes its hit
  counters (`src/cache.rs:HIT_STRIPES`);
- the process-global `last` front is removed;
- the hot-queue share goes from 0.97 to 0.5, so an entry admitted into a full
  cache is no longer evicted by its own admission (see finding 2);
- symtn-core requests `quick_cache`'s `sharded-lock` itself.

**Setup:**
- **Host:** lg `CPUcluster008`, PBS `workq`, `select=1:ncpus=96`,
  `place=excl`; 2 × Xeon Platinum 9242 (96 cores, 1 thread per core, 4 NUMA
  nodes).
- **Build:** rustc/cargo 1.96.0, `cargo bench --locked --offline -p
  symtn-network --bench cache_scaling --no-run`, `CARGO_PROFILE_RELEASE_DEBUG=0`,
  default features. One `Cargo.lock` for both, sha256 prefix
  `780812718e28ea89`. Binary sha256 prefixes: base `8b7c0fbf6ad3fc74`, final
  `a4b885c6e52fc5cd`.
- **Method:** the probe's `raw` rows and its `permute U1 warm` rows (s3 and
  s6, own and shared Runtime), at 1, 2, 4, 8, 16, 32, 48 and 96 threads,
  500 ms windows, `RAYON_NUM_THREADS=1`.
  - Four passes in one job, ordered base/final, final/base, final/base,
    base/final.
  - The table shows the median of the four passes.
  - `active_threads` equalled the requested count in every row.
  - CPU% @96 is 100 · `cpu_s`/`wall_s` of the 96-thread window
    (9600% = all cores).
  - Whole invocations (`/usr/bin/time -v`) used 2379–2554% CPU, 16.2–16.4 s
    each.
- **Raw rows:** `cache-hit-striping-2026-10-28.csv`, with `variant` and `pass`
  columns.

## Results (ops/s, median of 4 passes)

| scenario | sym | case | runtime | variant | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 96 | CPU% @96 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_front | raw | k4096 | own | base | 3.59e+07 | 6.18e+06 | 3.47e+06 | 3.09e+06 | 3.31e+06 | 3.18e+06 | 3.45e+06 | 3.68e+06 | 8500 |
| raw_front | raw | k4096 | own | final | 3.07e+07 | 1.77e+07 | 1.25e+07 | 1.06e+07 | 8.06e+06 | 7.03e+06 | 7.57e+06 | 7.51e+06 | 8483 |
| raw_shard | raw | k4096 | own | base | 2.32e+07 | 2.62e+06 | 2.44e+06 | 2.74e+06 | 3e+06 | 3.1e+06 | 3.21e+06 | 3.05e+06 | 8292 |
| raw_shard | raw | k4096 | own | final | 3.09e+07 | 1.54e+07 | 8.87e+06 | 7.94e+06 | 7.38e+06 | 7.62e+06 | 7.54e+06 | 7.33e+06 | 8954 |
| raw_spread | raw | k4096 | own | base | 2.15e+07 | 6.61e+06 | 4.24e+06 | 3.36e+06 | 3.45e+06 | 3.05e+06 | 3.1e+06 | 2.92e+06 | 8344 |
| raw_spread | raw | k4096 | own | final | 2.89e+07 | 5.44e+07 | 6.97e+07 | 9.68e+07 | 8.38e+07 | 2.98e+07 | 2.61e+07 | 2.37e+07 | 8089 |
| raw_publish | raw | fresh | own | base | 1.23e+06 | 1.6e+06 | 1.81e+06 | 1.99e+06 | 2.19e+06 | 2.4e+06 | 2.54e+06 | 2.72e+06 | 8690 |
| raw_publish | raw | fresh | own | final | 1.25e+06 | 1.66e+06 | 2.11e+06 | 2.3e+06 | 2.49e+06 | 2.68e+06 | 2.85e+06 | 3.01e+06 | 8423 |
| permute | U1 | s3 | own | base | 3.49e+05 | 4.5e+05 | 3.26e+05 | 4.27e+05 | 5.11e+05 | 5.99e+05 | 6.47e+05 | 7.05e+05 | 9538 |
| permute | U1 | s3 | own | final | 3.4e+05 | 3.6e+05 | 3.24e+05 | 4.4e+05 | 5.32e+05 | 6.06e+05 | 6.36e+05 | 6.88e+05 | 9551 |
| permute | U1 | s3 | shared | base | 3.49e+05 | 3.54e+05 | 3.1e+05 | 3.96e+05 | 4.72e+05 | 5.12e+05 | 5.22e+05 | 5.15e+05 | 9282 |
| permute | U1 | s3 | shared | final | 3.39e+05 | 3.5e+05 | 3.07e+05 | 3.99e+05 | 4.77e+05 | 5.16e+05 | 5.32e+05 | 5.12e+05 | 9317 |
| permute | U1 | s6 | own | base | 8.06e+04 | 1.33e+05 | 1.94e+05 | 3.01e+05 | 4.01e+05 | 4.71e+05 | 5.07e+05 | 5.02e+05 | 9558 |
| permute | U1 | s6 | own | final | 8.68e+04 | 1.61e+05 | 2.32e+05 | 3.62e+05 | 4.87e+05 | 5.98e+05 | 6.52e+05 | 6.97e+05 | 9560 |
| permute | U1 | s6 | shared | base | 8.07e+04 | 1.3e+05 | 1.85e+05 | 2.87e+05 | 3.73e+05 | 4.22e+05 | 4.37e+05 | 4.46e+05 | 9238 |
| permute | U1 | s6 | shared | final | 8.6e+04 | 1.46e+05 | 2.08e+05 | 3.23e+05 | 4.25e+05 | 5.11e+05 | 5.29e+05 | 5.23e+05 | 9301 |

## Findings

1. **Warm hits scale.**
   - Distinct keys (`raw_spread`): 54–97 M/s aggregate at 2–16 threads and
     24–30 M/s at 32–96. The base stays at 3 M/s or below from 2 threads on.
   - One shared key (`raw_front`, `raw_shard`): 7–18 M/s against 2.4–6 M/s.
     Every hit still clones the one value `Arc`, so its reference count is a
     shared word, and that caps this row.
   - The quick_cache shard read is per-thread striped.
2. **Admission under byte pressure (`HOT_ALLOCATION`).**
   - With quick_cache's default hot share of 0.97, a new key enters the cold
     queue unreferenced once the hot queue is full. The over-budget sweep
     (`shard.rs:advance_cold`) then evicted it immediately whenever its
     weight exceeded the 3% cold share.
   - A re-admitted ghost key enters hot unreferenced and lost the CLOCK sweep
     to referenced hot entries, so every repeat rebuilt the entry.
   - The old shared front hid this, because it served the just-evicted entry
     while a caller still held it.
   - With a hot share of 0.5, every admissible entry fits the cold queue. The
     new entry is the youngest, so the sweep removes older unreferenced
     entries first.
   - `symtn-core` `cache::tests::an_entry_admitted_into_a_full_cache_is_served_on_its_next_repeat`
     builds 4 times on base and once after. The `symtn-tensors` serial suite
     drops from 28 s to 13 s on the Mac, because the rank-6 warm-planning
     test no longer rebuilds its transformer.
   - The cost: the largest admissible entry is half the budget instead of
     97%. The default per-cache budget therefore rises from 64 MiB to
     128 MiB, so the largest admissible entry stays at 64 MiB
     (67,108,864 B). That still admits the U(1) `V^6 <- V^6` structures of
     30–46 MB. The runs above used the 64 MiB default, which none of these
     rows comes near.
   - The rows above show the effect: U(1) `s6` permute is +17–39% at 32–96
     threads and +8% at 1 thread.
3. **Single-thread warm hits.** On lg, at 1 thread, final against base:
   - `raw_front` 3.07e7 vs 3.59e7 (−14%; base served it through the front);
   - `raw_shard` +33%;
   - U(1) `s3` permute −3%; U(1) `s6` +8%.

   On the Mac (16 cores, release, three alternating runs):
   - `raw_front` −9%; U(1) `s3` permute 7.9e5 vs 8.9e5 (−11%); `s6` −3%;
     contract flat.

   The remaining cost of a hit is comparing the sector signature: the
   caller's HomSpace `Arc` differs from the key's, so the `Arc::ptr_eq`
   fast path in `FusionTreeHomSpaceCacheKey::eq` misses and
   `product_space_signature_eq` walks every sector of every leg.
4. **A stored HomSpace hash was tried and left out.**
   - The variant stored the sector-signature and content hashes in
     `FusionTreeHomSpaceContent`. It recovered 4% of `s3` on the Mac and
     none on lg (1 thread: 3.23e5 against 3.41e5 without it).
   - In two jobs it collapsed U(1) `s3` own-Runtime permute at 8–96 threads,
     to 1.5–1.9e5 at 96. A 15 s `eu-stack` at 96 threads put 155 of 194
     worker samples in the inner copy loop of
     `symtn_operations::kernel_adapter::fused_layout::apply_fused_pair_slices`;
     base workers were in cache lookup and planning.
   - The same rows with glibc's tcache disabled
     (`GLIBC_TUNABLES=glibc.malloc.tcache_count=0`), as ops/s at 96 threads:

     | variant | default | tcache off |
     |---|---:|---:|
     | base | 5.3–5.7e5 | 6.0e5 |
     | no front, hot 0.97 | 7.2–7.4e5 | 3.5–4.0e5 |
     | no front, hot 0.5 (final) | 7.1–7.5e5 | 3.9–4.3e5 |
     | final + stored hash | 1.5–1.6e5 | 3.7e5 |

     The variants exchange ranks when only the allocator changes. So at this
     scale the `s3` own-Runtime row depends on heap placement, not only on
     the cache path. Which placement causes it is not established.
   - Because the stored hash gave no measured single-thread gain on lg and
     coincides with the collapse, it is not part of this change.

5. **The half share must survive budget changes.**
   - quick_cache 0.7.0 `shard.rs:set_capacity` does not keep the configured
     share.
     - From capacity 0 it resets the hot target to its default of 0.97.
     - From any other capacity it carries over the old target/capacity
       ratio, which is 1.0 after a capacity of 1.
   - Disabling a cache with a budget of 0 or 1 and re-enabling it therefore
     brought back the rebuild on every repeat: 4 builds instead of 1.
   - `StructureCache` now keeps its own budget. quick_cache only ever gets an
     even capacity of at least 2: the budget rounded down to even, so half of
     it is exact and carries over exactly.
   - A budget below 2 clears the cache, counting the cleared entries as
     evictions, and rejects every admission. quick_cache keeps its last
     capacity meanwhile.
   - Rebuilding the quick_cache instance on each budget change was rejected.
     The instance would then sit behind a lock or a swappable pointer read on
     every hit, and std has no lock-free swap.

## References

| reference | revision and path | hit path | front | counters |
|---|---|---|---|---|
| TensorKit | cfaa073 `src/auxiliary/caches.jl:@cached` (`GLOBAL_CACHES` :1-11, `LRU{Any,Any}` :162); `src/spaces/structure.jl:39,74` | LRUCache 1.6.2 `src/LRUCache.jl:get!` :158-183 takes one `SpinLock` per cache on every hit; `Hashed(W, sectorhash, sectorequal)` recomputes the sector-only hash on every lookup | none | plain `hits`/`misses` under that lock |
| QSpace | dd2cc7e `Source/clebsch.hh:CStore` :3709, `RStore` :4606, `gStore` :4861 | caches only Clebsch-Gordan and recoupling symbol data (`std::map` plus disk); blocks travel with each QSpace object, so it has no structure cache | none | — |
| fusion-symbols | 0.9.1 `src/cache.rs:HIT_STRIPES`, `PaddedCounter`, `hit_stripe`, `tier_stats` | one quick_cache read | none | 16 `#[repr(align(128))]` hit stripes, round-robin thread slot, summed on read |
| quick_cache | 0.7.0 `shard.rs:non_resident_to_placeholder`, `replace_placeholder`, `advance_cold`, `advance_hot`, `set_capacity`; `options.rs` (`DEFAULT_HOT_ALLOCATION` 0.97, `DEFAULT_GHOST_ALLOCATION` 0.5) | sharded read lock (`sharded-lock`); CLOCK-Pro with hot/cold/ghost queues; a new key enters cold unreferenced; an entry above the hot target is dropped | — | — |

symtn now matches TensorKit's and fusion-symbols' hit semantics (a pure memo,
no front). A hit takes a sharded read lock and writes only its own counter
stripe; TensorKit's hit is serialized by the cache lock. The counters are
striped as in fusion-symbols, extended to all five. The hot share of 0.5 and
its preservation across budget changes are symtn's own: neither reference
weighs entries by bytes.

## Follow-up

**Intern the sector signature.** Give each distinct HomSpace sector
signature one canonical `Arc`, looked up when a HomSpace is constructed.
`FusionTreeHomSpaceCacheKey` equality then becomes `Arc::ptr_eq`, and its
hash becomes the pointer or a hash stored with the canonical entry.

- This removes the `product_space_signature_eq` walk that dominates a warm
  hit at one thread (finding 3).
- It needs its own bounded, clearable intern table, and that table needs its
  own leaf. It is not implemented here.
