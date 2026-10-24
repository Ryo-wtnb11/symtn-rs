#![cfg(feature = "cgc-gen")]

use std::sync::Arc;
use symtn::sector::{LieGroupFusionRule, ProductFusionRule, ProductSector, U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

#[test]
fn product_braid_moves_unequal_degeneracies_and_multiplicity_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(
        ProductFusionRule::<_, _>::try_new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            U1FusionRule,
        )
        .unwrap(),
    );
    let label = ProductSector::new(vec![1, 1], U1Irrep::new(0));
    let legs =
        [2, 3, 2].map(|n| GradedSpace::try_new(Arc::clone(&rule), [(label.clone(), n)]).unwrap());
    let value = |mu: usize, i: &[usize]| (100 * mu + 10 * i[0] + 3 * i[1] + i[2]) as f64;
    let source: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&legs[0], &legs[1]],
        [&legs[2]],
        |trees, indices| value(trees.codomain_vertices()[0].get(), indices),
    )
    .unwrap();
    let braided = source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap();
    assert_eq!(source.subblock_count(), 2);
    assert_eq!(braided.subblock_count(), 2);
    // TensorKitSectors Rsymbol(SU3Irrep(1,1), SU3Irrep(1,1), SU3Irrep(1,1)),
    // exported by explicit row/column indices; U1 contributes 1.
    let r = [
        [-2.0 / 7.0, 3.0 * 5.0_f64.sqrt() / 7.0],
        [3.0 * 5.0_f64.sqrt() / 7.0, 2.0 / 7.0],
    ];
    assert_eq!(braided.codomain()[0].degeneracies(), &[3]);
    assert_eq!(braided.codomain()[1].degeneracies(), &[2]);
    let mut saw_mixing = false;
    for (trees, block) in braided.subblocks().unwrap() {
        let row = trees.codomain_vertices()[0].get() - 1;
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..2 {
                    let expected =
                        r[row][0] * value(1, &[j, i, k]) + r[row][1] * value(2, &[j, i, k]);
                    assert!((*block.get(&[i, j, k]).unwrap() - expected).abs() < 1e-10);
                    saw_mixing |= (expected - value(row + 1, &[j, i, k])).abs() > 1.0;
                }
            }
        }
    }
    assert!(
        saw_mixing,
        "negative control: an axis-only permutation is wrong"
    );
    let back = braided.braid(&[1, 0], &[2], &[1, 0, 2]).unwrap();
    let bent_back = source.repartition(1).unwrap().repartition(2).unwrap();
    for result in [&back, &bent_back] {
        assert_eq!(result.codomain(), source.codomain());
        assert_eq!(result.domain(), source.domain());
        assert_eq!(
            result.dense_data().unwrap().len(),
            source.dense_data().unwrap().len()
        );
        for (&actual, &expected) in result
            .dense_data()
            .unwrap()
            .iter()
            .zip(source.dense_data().unwrap())
        {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}
