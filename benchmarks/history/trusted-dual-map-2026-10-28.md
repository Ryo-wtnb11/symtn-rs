# Dual map trusted under its filling rule identity (before/after)

Date 2026-10-28. Base `origin/main` `b5ba8d18` against the change for #2422:
a leg's shared dual map records the `RuleIdentity` of the rule that filled
it, and a later dual under an equal identity returns the map without asking
the provider for any sector's dual. Any other identity re-queries every
sector, as before.

TensorKit derives a permuted HomSpace with `select(W, p)`
(`spaces/homspace.jl:240-244` in 0.17.1), and the dual of a space is a flag
flip that shares `dims` (`spaces/gradedspace.jl:112`). Before this change,
symtn queried `rule.dual` once per sector of every dualized leg on every
derivation, to check the stored map. For SU(2), that query goes through the
generic Lie adapter and costs about 27 ns. After the change, a derivation
under the filling identity is O(rank), as in TensorKit.

**Setup:**
- **Host:** Apple M4 Max (12 performance + 4 efficiency cores), macOS 15.5.
- **Toolchain:** rustc 1.99.1.
- **Lock file:** `Cargo.lock` sha256 prefix `e9a0aac2dd4196bf`. Default
  features, `CARGO_PROFILE_RELEASE_DEBUG=0`.
- **Derivation probe:** a standalone release binary that depends on the
  tree's `symtn-core` by path; it uses public API only (shown below).
  - Fixture: `(l0 l1 ← l2 l3)` permuted to `([1, 2], [3, 0])`, which
    dualizes two legs.
  - Legs: U(1) charges centred on 0, or SU(2) twice-spins `0..s`; every
    degeneracy is 2.
  - 400,000 iterations per row after warming.
  - `derive` is `permute` plus drop.
  - `derive + complete lookup` adds
    `coupled_subblock_structure_from_leg_degeneracies`, a warm cache hit.
- **Operation rows:** the `cache_scaling` bench (`cargo bench --locked -p
  symtn-network --bench cache_scaling --no-run`), filters
  `contract|network warm own`, `CACHE_SCALING_THREADS=1`,
  `RAYON_NUM_THREADS=1`, 500 ms windows. Binary sha256 prefixes: base
  `2ce1c03676c669cb`, change `7c1773c0d2d64aeb`.
- **Order:** A B B A A B B A (A = base). The tables show the median of four
  passes per variant.
- **Raw rows:** `trusted-dual-map-2026-10-28-derivation.csv` and
  `trusted-dual-map-2026-10-28-ops.csv`.

## Derivation (ns/op, single thread)

| symmetry | case | metric | base | change | change/base |
|---|---|---|---:|---:|---:|
| U1 | s3 | derive | 117.4 | 108.8 | 0.93 |
| U1 | s3 | derive + complete lookup | 184.8 | 169.4 | 0.92 |
| U1 | s6 | derive | 129.2 | 112.7 | 0.87 |
| U1 | s6 | derive + complete lookup | 208.4 | 187.2 | 0.90 |
| SU2 | s3 | derive | 314.0 | 113.3 | 0.36 |
| SU2 | s3 | derive + complete lookup | 381.4 | 170.8 | 0.45 |
| SU2 | s6 | derive | 487.0 | 110.8 | 0.23 |
| SU2 | s6 | derive + complete lookup | 564.7 | 192.9 | 0.34 |

After the change, the derivation cost no longer depends on the symmetry or
on the number of sectors. What remains is the content allocation, the leg
`Arc` clones, one identity read and the drop.

## Warm owned operations (ops/s, single thread)

| scenario | symmetry | case | base | change | change/base |
|---|---|---|---:|---:|---:|
| contract | U1 | s3 | 2.60e5 | 2.62e5 | 1.007 |
| contract | U1 | s6 | 1.30e5 | 1.31e5 | 1.005 |
| contract | SU2 | s3 | 7.07e4 | 7.19e4 | 1.016 |
| contract | SU2 | s6 | 3.54e3 | 3.51e3 | 0.994 |
| network | U1 | s3 | 1.89e5 | 1.91e5 | 1.007 |
| network | U1 | s6 | 8.20e4 | 8.14e4 | 0.991 |
| network | SU2 | s3 | 4.20e4 | 4.17e4 | 0.994 |
| network | SU2 | s6 | 1.81e3 | 1.80e3 | 0.995 |

A warm owned contraction derives its output HomSpace once, outside the
permuted-destination memo.
- SU(2) `s3` gains 1.6%: about 0.24 µs of 14.1 µs, which is in line with
  the derivation saving.
- The other rows are within ±1%, the pass-to-pass spread of this host. Dense
  work dominates the `s6` rows, and the network rows' remaining derivations
  are covered by the memo.

A cold first derivation still fills each leg's map with one query per
sector, so cold rows are not expected to change.

## Probe source

```rust
fn run<R: MultiplicityFreeFusionRule>(name: &str, rule: &R, leg: &dyn Fn() -> SectorLeg) {
    let h = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let (cod, dom) = (&[1usize, 2][..], &[3usize, 0][..]);
    let _ = h.permute(rule, cod, dom).unwrap()
        .coupled_subblock_structure_from_leg_degeneracies(rule).unwrap();
    let t = Instant::now();
    for _ in 0..N {
        drop(black_box(h.permute(rule, cod, dom).unwrap()));
    }
    // ... `derive_ns` = elapsed / N
    let t = Instant::now();
    for _ in 0..N {
        let d = h.permute(rule, cod, dom).unwrap();
        black_box(d.coupled_subblock_structure_from_leg_degeneracies(rule).unwrap());
    }
    // ... `derive_plus_complete_lookup_ns` = elapsed / N
}
```
