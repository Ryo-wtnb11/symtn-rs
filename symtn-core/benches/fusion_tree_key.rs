//! `FusionTreeKey` identity hash/eq canary for a multiplicity-free
//! provider. Vertex labels remain part of exact identity; the benchmark uses
//! the categorical label one required by `Z2FusionRule`. No timing assertion
//! is used because shared CI runners are too noisy for a pass/fail latency
//! gate. The gate is this bench compiling and running plus the `size_of`
//! canary in `src/tests.rs`.

use std::hash::{Hash, Hasher};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustc_hash::FxHasher;
use symtn_core::{FusionTreeKey, MultiplicityIndex, SectorId, Z2FusionRule};

fn mult_free_key() -> FusionTreeKey {
    FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [SectorId::new(0); 4],
        SectorId::new(0),
        [false; 4],
        [SectorId::new(0); 2],
        [MultiplicityIndex::ONE; 3],
    )
    .expect("the benchmark fixture is a valid Z2 fusion tree")
}

fn bench_hash_eq(c: &mut Criterion) {
    let a = mult_free_key();
    let b = mult_free_key();
    c.bench_function("fusion_tree_key_hash_mult_free", |bencher| {
        bencher.iter(|| {
            let mut state = FxHasher::default();
            black_box(&a).hash(&mut state);
            black_box(state.finish())
        })
    });
    c.bench_function("fusion_tree_key_eq_mult_free", |bencher| {
        bencher.iter(|| black_box(&a) == black_box(&b))
    });
}

criterion_group!(benches, bench_hash_eq);
criterion_main!(benches);
