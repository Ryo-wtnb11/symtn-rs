use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{Error, GradedSpace, Runtime, SectorSpectrum, TensorMap};

#[test]
fn solve_roles_equal_explicit_permutation_and_reconstruct_rhs() {
    // Isolated: the completed-transformer counters are process-global.
    if run_isolated_or_return(
        "SYMTN_SOLVE_LEG_ROLES_ISOLATED",
        "solve_roles_equal_explicit_permutation_and_reconstruct_rhs",
    ) {
        return;
    }
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let a: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |_, ij| {
            if ij[0] == ij[1] && ij[2] == ij[3] {
                2.0
            } else {
                0.0
            }
        })
        .unwrap();
    let b: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |_, ij| {
            (1 + ij[0] + 2 * ij[1] + 3 * ij[2] + 4 * ij[3]) as f64
        })
        .unwrap();
    let a_rows = &[0, 2];
    let a_cols = &[1, 3];
    let b_rows = &[1, 3];
    let b_cols = &[0, 2];
    let actual = a.solve(a_rows, a_cols, &b, b_rows, b_cols).unwrap();
    let pa = a.permute(a_rows, a_cols).unwrap();
    let pb = b.permute(b_rows, b_cols).unwrap();
    let expected = pa.solve(&[0, 1], &[2, 3], &pb, &[0, 1], &[2, 3]).unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.codomain(), pa.domain());
    assert_eq!(actual.domain(), pb.domain());
    let rebuilt = pa.compose(&actual).unwrap();
    for (&x, &y) in rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(pb.dense_data().unwrap())
    {
        assert!((x - y).abs() < 1e-11);
    }

    let ac = a.convert::<Complex64>();
    let bc = b
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.5))
        .unwrap();
    let actual = ac.solve(a_rows, a_cols, &bc, b_rows, b_cols).unwrap();
    let pa = ac.permute(a_rows, a_cols).unwrap();
    let pb = bc.permute(b_rows, b_cols).unwrap();
    let expected = pa.solve(&[0, 1], &[2, 3], &pb, &[0, 1], &[2, 3]).unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    let rebuilt = pa.compose(&actual).unwrap();
    for (&x, &y) in rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(pb.dense_data().unwrap())
    {
        assert!((x - y).norm() < 1e-11);
    }

    assert!(
        matches!(a.solve(&[0, 0], &[1, 3], &b, b_rows, b_cols), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 4, .. }))
    );
    assert!(
        matches!(a.solve(a_rows, a_cols, &b, &[0, 0], &[1, 3]), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 4, .. }))
    );

    let before = completed_transformers();
    assert!(matches!(
        a.solve(a_rows, a_cols, b.adjoint_view(), b_rows, b_cols),
        Err(Error::Unsupported {
            operation: "solve",
            ..
        })
    ));
    let after = completed_transformers();
    assert_eq!(
        before.hits() + before.misses(),
        after.hits() + after.misses()
    );

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_b: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&other_runtime, [&leg, &leg], [&leg, &leg], 8).unwrap();
    assert!(matches!(
        a.solve(a_rows, a_cols, other_b.adjoint_view(), b_rows, b_cols),
        Err(Error::RuntimeMismatch)
    ));
}

#[test]
fn compact_divisor_keeps_the_identity_path() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let divisor: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![2.0, 4.0],
        }],
    )
    .unwrap();
    let rhs = divisor.scale(2.0).unwrap();
    let solution = divisor.solve(&[0], &[1], &rhs, &[0], &[1]).unwrap();
    assert!(solution.map_diagonal(|x| x).is_ok());
    assert_eq!(
        solution.materialize().unwrap().dense_data().unwrap(),
        [2.0, 0.0, 0.0, 2.0]
    );
}

#[test]
fn nonabelian_and_fermionic_dual_roles_match_composition() {
    macro_rules! check {
        ($left:expr, $right:expr) => {{
            let runtime = Runtime::builder().dense_threads(1).build().unwrap();
            let (left, right) = ($left, $right);
            let a: TensorMap<_, f64> =
                TensorMap::isomorphism(&runtime, [&left, &right], [&left, &right]).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&runtime, [&left, &right], [&left, &right], 19).unwrap();
            let rows = &[1, 0];
            let cols = &[3, 2];
            let actual = a.solve(rows, cols, &b, rows, cols).unwrap();
            let pa = a.permute(rows, cols).unwrap();
            let pb = b.permute(rows, cols).unwrap();
            let expected = pa.solve(&[0, 1], &[2, 3], &pb, &[0, 1], &[2, 3]).unwrap();
            assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
            let rebuilt = pa.compose(&actual).unwrap();
            let delta = rebuilt.axpby(1.0, &pb, -1.0).unwrap().norm(2.0).unwrap();
            assert!(delta < 1e-10, "residual {delta}");
        }};
    }

    let su2 = Arc::new(SU2FusionRule);
    let v = GradedSpace::try_new(Arc::clone(&su2), [(SU2Irrep::from_twice_spin(1), 2)]).unwrap();
    let w = GradedSpace::try_new(su2, [(SU2Irrep::from_twice_spin(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    check!(v, w);

    let fermion = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let sector = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
    let v = GradedSpace::try_new(Arc::clone(&fermion), [(sector, 2)]).unwrap();
    let w = GradedSpace::try_new(fermion, [(sector, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    check!(v, w);
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}

/// Re-executes exactly one global-cache probe in a fresh process. This binary
/// also contains ordinary tests that mutate the same caches without a shared
/// lock, so test-thread serialization inside one probe is insufficient.
fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
    if std::env::var_os(isolated_env).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_path, "--include-ignored"])
        .env(isolated_env, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
        "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    true
}
