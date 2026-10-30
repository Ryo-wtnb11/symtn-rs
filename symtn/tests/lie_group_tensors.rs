//! Public-facade Lie admission and Host transform round trips.
//!
//! The categorical oracle is the snake/rigidity identity used by TensorKit's
//! permute/repartition production path. Different outer-multiplicity vertices
//! carry different payloads, so losing or exchanging an axis cannot pass.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;
use symtn::typed::Engine;

use fusion_symbols::lie::{GroupId, LieError};
use num_complex::Complex64;
use symtn::sector::{LieGroupFusionRule, LieGroupFusionRuleError};
use symtn::typed::{
    CheckedGenericStructureError, GenericTensorError, GradedSpace, Runtime, TensorMap, TensorScalar,
};

use crate::numerics;

#[test]
fn graded_space_preserves_typed_global_form_rejections() {
    // Independent literal center-character matrix, mirrored at the public
    // construction boundary rather than inferred from GroupId::admits.
    let cases: Vec<(GroupId, Vec<i64>, Vec<Vec<i64>>)> = vec![
        (GroupId::su(3).unwrap(), vec![1, 0], vec![]),
        (
            GroupId::su_quotient(4, 2).unwrap(),
            vec![0, 1, 0],
            vec![vec![1, 0, 0]],
        ),
        (GroupId::psu(3).unwrap(), vec![1, 1], vec![vec![1, 0]]),
        (GroupId::sp(4).unwrap(), vec![1, 0], vec![]),
        (GroupId::psp(4).unwrap(), vec![0, 1], vec![vec![1, 0]]),
        (GroupId::spin(5).unwrap(), vec![0, 1], vec![]),
        (GroupId::so(5).unwrap(), vec![1, 0], vec![vec![0, 1]]),
        (GroupId::pso(5).unwrap(), vec![1, 0], vec![vec![0, 1]]),
        (GroupId::so(3).unwrap(), vec![2], vec![vec![1]]),
        (GroupId::psp(2).unwrap(), vec![2], vec![vec![1]]),
        (GroupId::psu(2).unwrap(), vec![2], vec![vec![1]]),
        (
            GroupId::so(6).unwrap(),
            vec![1, 0, 0],
            vec![vec![0, 1, 0], vec![0, 0, 1]],
        ),
        (
            GroupId::pso(6).unwrap(),
            vec![0, 1, 1],
            vec![vec![1, 0, 0], vec![0, 1, 0]],
        ),
        (
            GroupId::so(8).unwrap(),
            vec![1, 0, 0, 0],
            vec![vec![0, 0, 1, 0], vec![0, 0, 0, 1]],
        ),
        (
            GroupId::pso(8).unwrap(),
            vec![0, 1, 0, 0],
            vec![vec![1, 0, 0, 0], vec![0, 0, 1, 0], vec![0, 0, 0, 1]],
        ),
        (
            GroupId::half_spin_plus(8).unwrap(),
            vec![0, 0, 0, 1],
            vec![vec![0, 0, 1, 0], vec![1, 0, 0, 0]],
        ),
        (
            GroupId::half_spin_minus(8).unwrap(),
            vec![0, 0, 1, 0],
            vec![vec![0, 0, 0, 1], vec![1, 0, 0, 0]],
        ),
        (
            GroupId::so(12).unwrap(),
            vec![1, 0, 0, 0, 0, 0],
            vec![vec![0, 0, 0, 0, 1, 0], vec![0, 0, 0, 0, 0, 1]],
        ),
        (
            GroupId::pso(12).unwrap(),
            vec![0, 1, 0, 0, 0, 0],
            vec![
                vec![1, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 1, 0],
                vec![0, 0, 0, 0, 0, 1],
            ],
        ),
        (
            GroupId::half_spin_plus(12).unwrap(),
            vec![0, 0, 0, 0, 0, 1],
            vec![vec![0, 0, 0, 0, 1, 0], vec![1, 0, 0, 0, 0, 0]],
        ),
        (
            GroupId::half_spin_minus(12).unwrap(),
            vec![0, 0, 0, 0, 1, 0],
            vec![vec![0, 0, 0, 0, 0, 1], vec![1, 0, 0, 0, 0, 0]],
        ),
    ];
    for (group, accepted, rejected) in cases {
        let provider = Arc::new(LieGroupFusionRule::new(group.clone()).unwrap());
        GradedSpace::try_new(Arc::clone(&provider), [(accepted.clone(), 2)]).unwrap();
        assert!(matches!(
            GradedSpace::try_new(Arc::clone(&provider), [(vec![], 2)]),
            Err(GenericTensorError::Structure(
                CheckedGenericStructureError::Provider(LieGroupFusionRuleError::LabelLength { .. })
            ))
        ));
        let mut negative = accepted;
        negative[0] = -1;
        assert!(matches!(
            GradedSpace::try_new(Arc::clone(&provider), [(negative, 2)]),
            Err(GenericTensorError::Structure(
                CheckedGenericStructureError::Provider(LieGroupFusionRuleError::NegativeLabel {
                    index: 0,
                    value: -1
                })
            ))
        ));
        for label in rejected {
            assert!(
                matches!(
                    GradedSpace::try_new(Arc::clone(&provider), [(label, 2)]),
                    Err(GenericTensorError::Structure(
                        CheckedGenericStructureError::Provider(LieGroupFusionRuleError::Provider(
                            LieError::NotAdmissible
                        ))
                    ))
                ),
                "{group:?}"
            );
        }
    }
}

fn host_round_trips<D>(
    engine: &mut dyn Engine,
    group: GroupId,
    label: &[i64],
    coupled: &[i64],
    multiplicity: usize,
    value: fn(usize) -> D,
) where
    D: TensorScalar + numerics::Numeric,
{
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(LieGroupFusionRule::new(group).unwrap());
    let left = GradedSpace::try_new(Arc::clone(&provider), [(label.to_vec(), 2)]).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&provider), [(label.to_vec(), 3)]).unwrap();
    let output = GradedSpace::try_new(provider, [(coupled.to_vec(), 2)]).unwrap();
    let source: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&left, &right],
        [&output],
        |trees, indices| {
            value(
                1 + 100 * trees.codomain_vertices()[0].get()
                    + indices[0]
                    + 3 * indices[1]
                    + 11 * indices[2],
            )
        },
    )
    .unwrap();
    assert_eq!(source.subblock_count(), multiplicity);
    assert_eq!(source.dense_data().unwrap().len(), 12 * multiplicity);
    let swapped = source.permute(engine, &[1, 0], &[2]).unwrap();
    let restored = swapped.permute(engine, &[1, 0], &[2]).unwrap();
    assert_eq!(restored.codomain(), source.codomain());
    assert_eq!(restored.domain(), source.domain());
    // Both operations are linear. Every output entry receives at most all
    // source entries per operation; source_len^2 bounds the two compositions.
    let terms = source.dense_data().unwrap().len().pow(2);
    numerics::assert_nonzero_slices_close(
        "permute then inverse",
        restored.dense_data().unwrap(),
        source.dense_data().unwrap(),
        terms,
    );
    let bent = source.repartition(engine, 1).unwrap();
    let restored = bent.repartition(engine, 2).unwrap();
    assert_eq!(restored.codomain(), source.codomain());
    assert_eq!(restored.domain(), source.domain());
    numerics::assert_nonzero_slices_close(
        "repartition then return",
        restored.dense_data().unwrap(),
        source.dense_data().unwrap(),
        terms,
    );
}

#[test]
fn host_real_and_complex_lie_tensors_preserve_fs_multiplicity_and_duality() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for (group, label, coupled, multiplicity) in [
        // Quaternionic Sp fundamental: FS=-1, bosonic twist=+1.
        (GroupId::sp(4).unwrap(), vec![1, 0], vec![0, 0], 1),
        // The two 16x16 -> 10 vertices carry distinct payloads.
        (GroupId::sp(4).unwrap(), vec![1, 1], vec![2, 0], 2),
        // D3 chiral 4 is non-self-dual, and 4x4 -> 6 is allowed.
        (GroupId::spin(6).unwrap(), vec![0, 0, 1], vec![1, 0, 0], 1),
    ] {
        host_round_trips(engine, group.clone(), &label, &coupled, multiplicity, |i| {
            i as f64 / 8.0
        });
        host_round_trips(engine, group, &label, &coupled, multiplicity, |i| {
            Complex64::new(i as f64 / 8.0, (i + 1) as f64 / 16.0)
        });
    }
}
