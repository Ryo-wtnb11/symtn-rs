//! User-level decomposition contract not already owned by the typed
//! factorization suites.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, Svd, TensorMap, Truncation};

use crate::numerics;

/// A truncated bond may drop a whole coupled sector, but recomposition must
/// restore the source layout with a zero block so ordinary typed operations
/// remain compatible. The positive-only charge set also pins the historical
/// non-dualization-closed U(1) case.
#[test]
fn truncated_svd_restores_dropped_sector_in_non_dual_closed_space() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();

    // Spectra: charge 0 -> {4, 3}, charge 1 -> {1}, charge 2 -> {2}.
    let tensor = TensorMap::<_, f64>::from_subblock_fn(
        engine,
        &runtime,
        [&space],
        [&space],
        |trees, indices| {
            if *trees.coupled() == U1Irrep::new(0) {
                match indices {
                    [0, 0] => 4.0,
                    [1, 1] => 3.0,
                    _ => 0.0,
                }
            } else if *trees.coupled() == U1Irrep::new(1) {
                1.0
            } else {
                2.0
            }
        },
    )
    .unwrap();

    // Reconstruction is backward stable: within the tolerance rule, one term
    // per entry of the largest (2 x 2) block.
    let Svd { u, s, vh } = tensor.svd_compact(engine, &[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "u s vh",
        u.compose(engine, &s)
            .unwrap()
            .compose(engine, &vh)
            .unwrap()
            .dense_data()
            .unwrap(),
        tensor.dense_data().unwrap(),
        4,
    );

    // Truncation is the composition `find_truncated` + `restrict_*`.
    let found = s.domain()[0]
        .find_truncated(&s.diagview(engine).unwrap(), &Truncation::rank(3))
        .unwrap();
    let u = u
        .restrict_leg(engine, &[(u.codomain_rank(), &found.selection)])
        .unwrap();
    let s = s
        .restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let vh = vh.restrict_leg(engine, &[(0, &found.selection)]).unwrap();
    let kept: Vec<_> = s
        .diagview(engine)
        .unwrap()
        .iter()
        .map(|entry| (entry.sector, entry.values.len()))
        .collect();
    assert_eq!(kept, [(U1Irrep::new(0), 2), (U1Irrep::new(2), 1)]);
    assert!((found.error - 1.0).abs() < 1.0e-12);

    let recomposed = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    numerics::assert_slices_close(
        "truncated u s vh",
        recomposed.dense_data().unwrap(),
        &[4.0, 0.0, 0.0, 3.0, 0.0, 2.0],
        4,
    );
    let error = tensor
        .axpby(engine, 1.0, &recomposed, -1.0)
        .unwrap()
        .norm(engine, 2.0)
        .unwrap();
    assert!((error - found.error).abs() < 1.0e-12);
}

#[test]
fn right_solve_composes_adjoints_around_left_solve_for_real_and_complex_u1() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&space], [&space], |trees, _| {
            if *trees.coupled() == U1Irrep::new(0) {
                6.0
            } else {
                10.0
            }
        })
        .unwrap();
    let rhs =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&space], [&space], |trees, _| {
            if *trees.coupled() == U1Irrep::new(0) {
                2.0
            } else {
                5.0
            }
        })
        .unwrap();
    let lhs_adjoint = &lhs.adjoint(engine).unwrap();
    let solved = rhs
        .adjoint(engine)
        .unwrap()
        .solve(engine, &[0], &[1], lhs_adjoint, &[0], &[1])
        .unwrap()
        .adjoint(engine)
        .unwrap();
    assert_eq!(
        solved.compose(engine, &rhs).unwrap().dense_data().unwrap(),
        lhs.dense_data().unwrap()
    );

    let lhs_c = lhs.convert::<Complex64>(engine).unwrap();
    let rhs_c = rhs.convert::<Complex64>(engine).unwrap();
    let lhs_c_adjoint = &lhs_c.adjoint(engine).unwrap();
    let solved_c = rhs_c
        .adjoint(engine)
        .unwrap()
        .solve(engine, &[0], &[1], lhs_c_adjoint, &[0], &[1])
        .unwrap()
        .adjoint(engine)
        .unwrap();
    assert!(solved_c
        .compose(engine, &rhs_c)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(lhs_c.dense_data().unwrap())
        .all(|(a, b)| (*a - *b).norm() < 1.0e-12));
}
