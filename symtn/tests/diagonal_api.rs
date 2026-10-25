use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Eigh;
use symtn::typed::{Complex64, GradedSpace, SectorSpectrum, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

#[test]
fn typed_diagonal_preserves_canonical_positions_and_dual_leg() {
    let runtime = host_runtime();
    let bond = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let values: Vec<SectorSpectrum<U1Irrep, Complex64>> = bond
        .sectors()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(position, &sector)| SectorSpectrum {
            sector,
            values: (0..bond.degeneracies()[position])
                .map(|index| Complex64::new((10 * position + index) as f64, index as f64))
                .collect(),
        })
        .collect();
    let tensor =
        TensorMap::<U1FusionRule, Complex64>::diagonal(&runtime, &bond, values.clone()).unwrap();
    assert_eq!(tensor.codomain()[0], bond);
    assert_eq!(tensor.domain()[0], bond);
    assert!(symtn::expert::is_diagonal(&tensor, 0.0).unwrap());
    let readback: Vec<SectorSpectrum<U1Irrep, Complex64>> =
        symtn::expert::diagonal_spectrum(&tensor).unwrap().unwrap();
    assert_eq!(readback, values);
    let singular_values: Vec<SectorSpectrum<U1Irrep>> = tensor.svd_vals(&[0], &[1]).unwrap();
    assert_eq!(singular_values.len(), values.len());
}

#[test]
fn typed_real_c64_eigenvalue_readback_stays_compact() {
    let runtime = host_runtime();
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let source =
        TensorMap::<U1FusionRule, Complex64>::from_subblock_fn(&runtime, [&v], [&v], |_, index| {
            Complex64::new(
                if index[0] == index[1] {
                    (index[0] + 1) as f64
                } else {
                    0.0
                },
                0.0,
            )
        })
        .unwrap();
    let Eigh { d: diagonal, .. } = source.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert_eq!(
        symtn::expert::diagonal_spectrum(&diagonal)
            .unwrap()
            .unwrap()[0]
            .values,
        [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    );
}

#[test]
fn typed_diagonal_canonicalizes_labels_and_dense_predicate_handles_nonfinite_offdiag() {
    let runtime = host_runtime();
    let rule = Arc::new(SU2FusionRule);
    let bond = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (SU2Irrep::from_twice_spin(2), 1),
            (SU2Irrep::from_twice_spin(0), 2),
        ],
    )
    .unwrap();
    let tensor = TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(2),
                values: vec![3.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, 2.0],
            },
        ],
    )
    .unwrap();
    assert_eq!(
        symtn::expert::diagonal_spectrum(&tensor).unwrap().unwrap()[0].sector,
        SU2Irrep::from_twice_spin(0)
    );
    assert!(symtn::expert::is_diagonal(&tensor, 0.0).unwrap());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(0),
            values: vec![1.0, 2.0],
        }],
    )
    .is_err());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(2),
                values: vec![3.0],
            },
        ],
    )
    .is_err());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(4),
                values: vec![3.0],
            },
        ],
    )
    .is_err());
    assert!(TensorMap::<SU2FusionRule, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![3.0, 4.0],
            },
        ],
    )
    .is_err());

    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let dense =
        TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&v], [&v], |_, index| {
            if index == [0, 1] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(!symtn::expert::is_diagonal(&dense, 1.0).unwrap());
    let finite =
        TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&v], [&v], |_, index| {
            if index[0] == index[1] {
                4.0
            } else if index == [0, 1] {
                0.25
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(!symtn::expert::is_diagonal(&finite, 0.05).unwrap());
    assert!(symtn::expert::is_diagonal(&finite, 0.1).unwrap());
    let exact = TensorMap::<U1FusionRule, f64>::isomorphism(&runtime, [&v], [&v]).unwrap();
    assert!(symtn::expert::is_diagonal(&exact, 0.0).unwrap());
    let inf = TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&v], [&v], |_, index| {
        if index == [1, 0] {
            f64::INFINITY
        } else {
            0.0
        }
    })
    .unwrap();
    assert!(!symtn::expert::is_diagonal(&inf, 1.0).unwrap());
    assert!(symtn::expert::is_diagonal(&dense, -1.0).is_err());
    let vector = TensorMap::<U1FusionRule, f64>::zeros(&runtime, [&v], []).unwrap();
    assert!(symtn::expert::is_diagonal(&vector, f64::NAN).is_err());
}
