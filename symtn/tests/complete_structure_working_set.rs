//! The complete-structure cache holds the working set of an E1-style op
//! ledger and of a sweep-like loop, so warm iterations never evict (#1365).
//! One test per process keeps the global cache statistics isolated.

use symtn::cache::{StructureCacheInfo, StructureCacheKind};

fn structure_cache_info(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

/// Activity between two snapshots: (hits, misses, admissions, evictions, bypasses).
fn delta(before: StructureCacheInfo, after: StructureCacheInfo) -> (u64, u64, u64, u64, u64) {
    (
        after.hits() - before.hits(),
        after.misses() - before.misses(),
        after.admissions() - before.admissions(),
        after.evictions() - before.evictions(),
        after.rejections() - before.rejections(),
    )
}

/// Runs `call` cold once, then three warm times, each warm call exactly
/// `hits` hits and no other activity.
fn assert_warm(label: &str, hits: u64, mut call: impl FnMut()) {
    call();
    for _ in 0..3 {
        let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        call();
        let after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        assert_eq!(delta(before, after), (hits, 0, 0, 0, 0), "{label}");
    }
}

#[test]
fn warm_working_sets_never_evict() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let u1 = |q: i32| U1Irrep::new(q);

    // E1-style ledger: several ops back to back without a reset, as the E1
    // rows run, so their structures share the cache.
    let leg = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        (-1..=1).map(|q| (u1(q), 2)),
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1).unwrap();
    let square =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 2).unwrap();
    let matrix = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 3).unwrap();
    assert_warm("permute", 1, || {
        a.permute(&[1, 2], &[3, 0]).unwrap();
    });
    assert_warm("compose", 1, || {
        a.compose(&square).unwrap();
    });
    assert_warm("contract", 3, || {
        a.contract(
            &matrix,
            &ContractSpec {
                lhs: &[0],
                rhs: &[1],
                codomain: &[0, 1, 2],
                domain: &[3],
            },
        )
        .unwrap();
    });
    assert_warm("qr_compact", 2, || {
        a.qr_compact(&[0, 1], &[2, 3]).unwrap();
    });
    assert_warm("svd_compact", 3, || {
        a.svd_compact(&[0, 1], &[2, 3]).unwrap();
    });

    // Sweep-like loop: an open chain `[vL, p] <- [vR]` with a distinct space
    // on every bond, contracted pairwise and repartitioned left to right.
    // One sweep touches more structures than the old 5-entry cap held.
    let physical =
        GradedSpace::try_new(std::sync::Arc::new(U1FusionRule), [(u1(-1), 1), (u1(1), 1)]).unwrap();
    let bonds: Vec<_> = (0..9)
        .map(|b| {
            GradedSpace::try_new(
                std::sync::Arc::new(U1FusionRule),
                [(u1(-1), b + 1), (u1(0), b + 2), (u1(1), b + 1)],
            )
            .unwrap()
        })
        .collect();
    let sites: Vec<_> = (0..8)
        .map(|i| {
            TensorMap::<_, f64>::rand_with_seed(
                &runtime,
                [&bonds[i], &physical],
                [&bonds[i + 1]],
                10 + i as u64,
            )
            .unwrap()
        })
        .collect();
    let sweep = || {
        for pair in sites.windows(2) {
            let theta = pair[0]
                .contract(
                    &pair[1],
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap();
            theta.repartition(2).unwrap();
        }
    };
    sweep();
    let cold = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert!(cold.entries() > 5, "sweep live set {}", cold.entries());
    for _ in 0..3 {
        let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        sweep();
        let after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        assert_eq!(delta(before, after), (7 * 3, 0, 0, 0, 0));
    }
}
