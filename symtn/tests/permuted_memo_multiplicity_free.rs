//! A warm multiplicity-free permute takes its destination from the source's
//! permuted memo, without deriving it (no sector-structure lookup), and is
//! bit-identical to the cold derivation; a cleared cache serves no memo.

use symtn::cache::StructureCacheKind;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, ZNFusionRule};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn sector_lookups() -> u64 {
    let info = symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::SectorStructure)
        .unwrap();
    info.hits() + info.misses()
}

macro_rules! assert_memoized {
    ($label:literal, $runtime:expr, $leg:expr) => {{
        let (label, runtime, leg) = ($label, $runtime, $leg);
        let dual = leg.try_dual().unwrap();
        // Dual legs on both sides; each permutation also moves the
        // codomain/domain split.
        let splits: [(&[usize], &[usize]); 2] = [(&[1, 2], &[3, 0, 4]), (&[4, 0, 3, 1], &[2])];
        for (codomain, domain) in splits {
            let permute = |a: &TensorMap<_, f64>| {
                let t = a.permute(codomain, domain).unwrap();
                let bits: Vec<u64> = t
                    .dense_data()
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect();
                (bits, t.codomain(), t.domain())
            };
            let source = || {
                TensorMap::<_, f64>::rand_with_seed(runtime, [&leg, &dual, &leg], [&dual, &leg], 7)
                    .unwrap()
            };
            symtn::cache::clear();
            let a = source();
            let cold = permute(&a);
            let before = sector_lookups();
            let warm = permute(&a);
            assert_eq!(
                sector_lookups(),
                before,
                "{label} {codomain:?}/{domain:?}: warm permute derived its destination"
            );
            assert_eq!(warm, cold, "{label}: memoized result differs");

            symtn::cache::clear();
            let a = source();
            let before = sector_lookups();
            let recold = permute(&a);
            assert!(
                sector_lookups() > before,
                "{label}: a cleared cache served the memo"
            );
            assert_eq!(recold, cold, "{label}: rederived result differs");
        }
    }};
}

#[test]
fn warm_multiplicity_free_permute_reuses_the_memoized_destination() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        (-1..=1).map(|q| (U1Irrep::new(q), 2)),
    )
    .unwrap();
    assert_memoized!("U1", &runtime, u1);
    let z3 = ZNFusionRule::new(3).unwrap();
    let z3_leg = GradedSpace::try_new(
        std::sync::Arc::new(z3.clone()),
        (0..3).map(|q| (z3.irrep(q), q as usize + 1)),
    )
    .unwrap();
    assert_memoized!("Z3", &runtime, z3_leg);
    let su2 = GradedSpace::try_new(
        std::sync::Arc::new(SU2FusionRule),
        (0..3).map(|twice| (SU2Irrep::from_twice_spin(twice), 2)),
    )
    .unwrap();
    assert_memoized!("SU2", &runtime, su2);
}
