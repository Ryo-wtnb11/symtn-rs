//! The composed-coefficient cache (cache 4 of #2014; #1554 before it).
//!
//! A truncation changes block dimensions but usually not which sectors are
//! present. Such a degeneracy-only change misses the completed-transformer
//! cache (its key holds the exact layout) but must not recompose any source
//! group's recoupling coefficients, and the result must be bit-identical to
//! a cold Runtime's. A sector change must still compose the new groups.
//! Unique fusion is `NoCache`, as in TensorKit: it never enters the cache,
//! and its degeneracy-only rebuild must still equal a cold Runtime's.

use std::sync::Arc;
use symtn::cache::{StructureCacheInfo, StructureCacheKind};
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn cache(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}

fn completed() -> StructureCacheInfo {
    cache(StructureCacheKind::CompletedTreeTransformer)
}

fn groups() -> StructureCacheInfo {
    cache(StructureCacheKind::TreeTransformCoefficients)
}

/// The caches are process-global: a test that counts their activity clears
/// them first and must not race another test publishing the same keys.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|value| value.to_bits()).collect()
}

/// Rank (2,2) tensor on `[leg, leg*] <- [leg, leg*]`, so every operation also
/// moves a dual leg.
macro_rules! tensor {
    ($runtime:expr, $leg:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        TensorMap::<_, f64>::rand_with_seed($runtime, [leg, &dual], [leg, &dual], 17).unwrap()
    }};
}

/// Runs permute, braid and transpose on three spaces over one rule: `a` warms
/// the caches, `b` has the same sectors with other degeneracies, and `c`
/// adds a sector. The warm result must be bit-identical to a cold Runtime's.
/// `$cached` says whether the rule's fusion enters the coefficient cache.
macro_rules! check_rule {
    ($label:literal, $cached:expr, $a:expr, $b:expr, $c:expr $(,)?) => {{
        let (cached, a, b, c): (bool, _, _, _) = ($cached, $a, $b, $c);
        let operations: [(&str, &dyn Fn(&TensorMap<_, f64>) -> TensorMap<_, f64>); 4] = [
            ("permute", &|t| t.permute(&[2, 0], &[3, 1]).unwrap()),
            ("braid", &|t| {
                t.braid(&[1, 3], &[0, 2], &[0, 1, 2, 3]).unwrap()
            }),
            ("transpose", &|t| t.transpose(&[1, 3], &[0, 2]).unwrap()),
            // A lazy adjoint source reads its parent's storage through the
            // adjoint orientation.
            ("adjoint permute", &|t| {
                t.adjoint().unwrap().permute(&[2, 0], &[3, 1]).unwrap()
            }),
        ];
        for (name, operation) in operations {
            let what = format!("{} {name}", $label);
            symtn::cache::clear();
            let warm = Runtime::builder().dense_threads(1).build().unwrap();
            let _ = operation(&tensor!(&warm, &a));
            let groups_before = groups();
            let structures_before = completed();
            assert_eq!(groups_before.misses() > 0, cached, "{what}: cold groups");

            let degeneracy_only = operation(&tensor!(&warm, &b));
            let after = groups();
            let structures = completed();
            assert!(
                structures.misses() > structures_before.misses(),
                "{what}: a new layout must miss the completed-transformer cache"
            );
            assert_eq!(
                after.misses(),
                groups_before.misses(),
                "{what}: a degeneracy-only change recomposed a coefficient group"
            );
            assert_eq!(after.hits() > groups_before.hits(), cached, "{what}: hits");

            // Clear every cache so the cold Runtime composes from scratch.
            symtn::cache::clear();
            let cold = Runtime::builder().dense_threads(1).build().unwrap();
            let expected = operation(&tensor!(&cold, &b));
            assert_eq!(groups().misses() > 0, cached, "{what}: cold rebuild");
            assert_eq!(degeneracy_only.codomain(), expected.codomain(), "{what}");
            assert_eq!(degeneracy_only.domain(), expected.domain(), "{what}");
            assert_eq!(
                bits(degeneracy_only.materialize().unwrap().dense_data().unwrap()),
                bits(expected.materialize().unwrap().dense_data().unwrap()),
                "{what}: warm result differs from a cold Runtime"
            );

            let before_sector_change = groups();
            let _ = operation(&tensor!(&warm, &c));
            assert_eq!(
                groups().misses() > before_sector_change.misses(),
                cached,
                "{what}: a sector change must compose its new groups"
            );
            if !cached {
                assert_eq!(
                    groups().entries(),
                    0,
                    "{what}: Unique fusion must not publish a composed-coefficient group"
                );
            }
        }
    }};
}

#[test]
fn u1_degeneracy_change_rebuilds_without_the_coefficient_cache() {
    let _serial = serial();
    let leg = |sectors: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            sectors.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    check_rule!(
        "U(1)",
        false,
        leg(&[(-1, 2), (0, 1), (1, 2)]),
        leg(&[(-1, 3), (0, 2), (1, 1)]),
        leg(&[(-1, 2), (0, 1), (1, 2), (2, 1)]),
    );
}

#[test]
fn su2_degeneracy_change_reuses_the_composed_coefficients() {
    let _serial = serial();
    let leg = |sectors: &[(usize, usize)]| {
        GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            sectors
                .iter()
                .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
        )
        .unwrap()
    };
    check_rule!(
        "SU(2)",
        true,
        leg(&[(0, 2), (1, 2), (2, 1)]),
        leg(&[(0, 1), (1, 3), (2, 2)]),
        leg(&[(0, 2), (1, 2), (2, 1), (3, 1)]),
    );
}

#[test]
fn fermion_u1_degeneracy_change_rebuilds_without_the_coefficient_cache() {
    let _serial = serial();
    let leg = |sectors: &[(i32, usize)]| {
        let rule = ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::new(
            FermionParityFusionRule,
            U1FusionRule,
        );
        GradedSpace::try_new(
            Arc::new(rule),
            sectors.iter().map(|&(q, d)| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                (ProductSector::new(parity, U1Irrep::new(q)), d)
            }),
        )
        .unwrap()
    };
    check_rule!(
        "fZ2xU(1)",
        false,
        leg(&[(-1, 2), (0, 1), (1, 2)]),
        leg(&[(-1, 1), (0, 3), (1, 2)]),
        leg(&[(-1, 2), (0, 1), (1, 2), (2, 1)]),
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_su3_degeneracy_change_reuses_the_composed_coefficients() {
    let _serial = serial();
    use symtn::sector::SUNFusionRule;

    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = |sectors: &[([i64; 2], usize)]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            sectors.iter().map(|&(labels, d)| (labels.to_vec(), d)),
        )
        .unwrap()
    };
    check_rule!(
        "SU(3)",
        true,
        leg(&[([1, 0], 2), ([1, 1], 1)]),
        leg(&[([1, 0], 1), ([1, 1], 2)]),
        leg(&[([0, 0], 1), ([1, 0], 2), ([1, 1], 1)]),
    );
}
