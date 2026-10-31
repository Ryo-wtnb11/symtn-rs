//! `map_diagonal` cells only multiplicity-free categories express: a
//! non-self-dual U(1) bond and SU(2)'s physical expansion. The contracts both
//! admission modes share are in `admission_modes/matrix_fn.rs`.

use super::*;

fn bits_f64(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn map_diagonal_matches_hand_built_diagonals_on_a_dual_u1_bond() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the result is the compact diagonal whose values are `f` of the
    // stored values, sector by sector, on the receiver's own (dual) bond.
    // Oracle: diagonals built from hand-computed literals.
    let runtime = runtime();
    let bond = GradedSpace::try_new(
        Arc::new(symtn::sector::U1FusionRule),
        [
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
            (symtn::sector::U1Irrep::new(-2), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    assert!(bond.is_dual());
    let q = symtn::sector::U1Irrep::new;
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (q(0), &[4.0, 9.0][..]),
            (q(-1), &[0.25][..]),
            (q(2), &[16.0, 1.0][..]),
        ]),
    )
    .unwrap();

    let root = source.map_diagonal(engine, f64::sqrt).unwrap();
    let expected: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (q(0), &[2.0, 3.0][..]),
            (q(-1), &[0.5][..]),
            (q(2), &[4.0, 1.0][..]),
        ]),
    )
    .unwrap();
    assert_eq!(root.codomain(), vec![bond.clone()]);
    assert_eq!(root.domain(), vec![bond.clone()]);
    assert_eq!(
        root.diagview(engine).unwrap(),
        expected.diagview(engine).unwrap()
    );
    assert_eq!(
        bits_f64(root.materialize(engine).unwrap().dense_data().unwrap()),
        bits_f64(expected.materialize(engine).unwrap().dense_data().unwrap())
    );

    let reciprocal = source.map_diagonal(engine, |value| 1.0 / value).unwrap();
    let expected: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (q(0), &[0.25, 1.0 / 9.0][..]),
            (q(-1), &[4.0][..]),
            (q(2), &[0.0625, 1.0][..]),
        ]),
    )
    .unwrap();
    assert_eq!(
        bits_f64(
            reciprocal
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        ),
        bits_f64(expected.materialize(engine).unwrap().dense_data().unwrap())
    );

    // Complex payload: `f` decides the branch; `Complex64::sqrt` is principal.
    let source: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (
                q(0),
                &[Complex64::new(-1.0, 0.0), Complex64::new(4.0, 0.0)][..],
            ),
            (q(-1), &[Complex64::new(0.0, 2.0)][..]),
            (
                q(2),
                &[Complex64::new(9.0, 0.0), Complex64::new(-4.0, 0.0)][..],
            ),
        ]),
    )
    .unwrap();
    let root = source.map_diagonal(engine, |value| value.sqrt()).unwrap();
    let expected = [
        Complex64::new(0.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 2.0),
    ];
    let actual: Vec<_> = root
        .diagview(engine)
        .unwrap()
        .into_iter()
        .flat_map(|entry| entry.values)
        .collect();
    let sorted = |mut values: Vec<Complex64>| {
        values.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
        values
    };
    let (actual, expected) = (sorted(actual), sorted(expected.to_vec()));
    for (value, expected) in actual.iter().zip(&expected) {
        assert!((value - expected).norm() < 1e-15, "{value} != {expected}");
    }
}

#[test]
fn map_diagonal_on_su2_repeats_each_value_over_its_carrier_dimension() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `f` acts on the reduced values once per degeneracy index, and the
    // physical-basis expansion repeats each result `2j + 1` times on the
    // diagonal. Oracle: the hand-listed physical diagonal.
    let runtime = runtime();
    let j = SU2Irrep::from_twice_spin;
    let bond =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 1), (j(2), 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (j(0), &[4.0, 9.0][..]),
            (j(1), &[0.25][..]),
            (j(2), &[16.0, 1.0][..]),
        ]),
    )
    .unwrap();
    let root = source.map_diagonal(engine, f64::sqrt).unwrap();
    let expected: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (j(0), &[2.0, 3.0][..]),
            (j(1), &[0.5][..]),
            (j(2), &[4.0, 1.0][..]),
        ]),
    )
    .unwrap();
    assert_eq!(
        root.diagview(engine).unwrap(),
        expected.diagview(engine).unwrap()
    );
    assert_eq!(
        bits_f64(root.materialize(engine).unwrap().dense_data().unwrap()),
        bits_f64(expected.materialize(engine).unwrap().dense_data().unwrap())
    );

    let physical = root.to_physical_dense(engine).unwrap();
    assert_eq!(physical.shape, vec![10, 10]);
    let mut diagonal = Vec::new();
    for row in 0..10 {
        for col in 0..10 {
            let value = physical.data[row + 10 * col];
            if row == col {
                diagonal.push(value);
            } else {
                assert_eq!(value, 0.0, "off-diagonal ({row}, {col})");
            }
        }
    }
    diagonal.sort_by(f64::total_cmp);
    let mut expected = vec![2.0, 3.0, 0.5, 0.5, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0];
    expected.sort_by(f64::total_cmp);
    for (value, expected) in diagonal.iter().zip(&expected) {
        assert!((value - expected).abs() < 1e-14, "{value} != {expected}");
    }
}
