//! The eigh Hermiticity admission at its default and an explicit tolerance
//! (#1987), through every multiplicity-free Host entry: dense and compact
//! diagonal `eigh_vals`/`eigh_full`, and the prepared [`EighFullPlan`]. The
//! checked Generic counterpart is in `checked_generic_facade/eig.rs`.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    BatchError, EighFullPlan, GradedSpace, HermitianTol, MemberFault, Runtime, SectorSpectrum,
    StackedTensorMap, TensorMap,
};

fn leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap()
}

/// `[[1, δ], [0, 2]]`: Hermitian to relative tolerance `tol` exactly when
/// `|δ|/√2 ≤ tol·√(5 + δ²)` (independent hand calculation).
/// The tests use the small-δ threshold `tol·√10` with a 3% margin.
fn dense(runtime: &Runtime, delta: f64) -> TensorMap<U1FusionRule, f64> {
    let leg = leg();
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |_, index| match index {
        [0, 0] => 1.0,
        [1, 0] => delta,
        [1, 1] => 2.0,
        _ => 0.0,
    })
    .unwrap()
}

/// `diag(1 + iδ, 2)`: Hermitian to `tol` when `|δ| ≤ tol·√(5 + δ²)`.
/// The tests use the small-δ threshold `tol·√5` with a 3% margin.
fn diagonal(runtime: &Runtime, delta: f64) -> TensorMap<U1FusionRule, Complex64> {
    TensorMap::diagonal(
        runtime,
        &leg(),
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(1.0, delta), Complex64::new(2.0, 0.0)],
        }],
    )
    .unwrap()
}

fn admits<D>(t: &TensorMap<U1FusionRule, D>, tol: HermitianTol) -> bool
where
    D: symtn::typed::FactorizationScalar,
{
    let vals = t.eigh_vals(&[0], &[1], tol);
    let full = t.eigh_full(&[0], &[1], tol);
    assert_eq!(vals.is_ok(), full.is_ok());
    if let Err(error) = vals {
        assert!(
            matches!(
                &error,
                symtn::typed::Error::Operation(operation) if matches!(
                    **operation,
                    symtn::typed::OperationError::InvalidArgument {
                        message: "eigh requires Hermitian coupled-sector blocks"
                    }
                )
            ),
            "{error:?}"
        );
        return false;
    }
    true
}

#[test]
fn the_default_admits_up_to_eps_three_quarters_relative() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let tol = f64::EPSILON.powf(0.75);
    for (factor, admitted) in [(0.97, true), (1.03, false)] {
        let dense = dense(&runtime, factor * tol * 10.0_f64.sqrt());
        assert_eq!(admits(&dense, HermitianTol::DEFAULT), admitted);
        let diagonal = diagonal(&runtime, factor * tol * 5.0_f64.sqrt());
        assert_eq!(admits(&diagonal, HermitianTol::DEFAULT), admitted);
    }
}

#[test]
fn an_explicit_tolerance_replaces_the_default() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for tol in [64.0 * f64::EPSILON, 1.0e-3] {
        let explicit = HermitianTol::relative(tol).unwrap();
        for (factor, admitted) in [(0.97, true), (1.03, false)] {
            let dense = dense(&runtime, factor * tol * 10.0_f64.sqrt());
            assert_eq!(admits(&dense, explicit), admitted, "dense {tol:e} {factor}");
            let diagonal = diagonal(&runtime, factor * tol * 5.0_f64.sqrt());
            assert_eq!(
                admits(&diagonal, explicit),
                admitted,
                "diagonal {tol:e} {factor}"
            );
        }
    }
}

#[test]
fn a_prepared_plan_fixes_its_tolerance() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let tol = 1.0e-3;
    let near = dense(&runtime, 0.5 * tol * 10.0_f64.sqrt());
    let stack = StackedTensorMap::pack(&[near]).unwrap();
    let explicit =
        EighFullPlan::new(&stack, &[0], &[1], HermitianTol::relative(tol).unwrap()).unwrap();
    let mut workspace = explicit.workspace().unwrap();
    assert!(explicit.execute(&stack, &mut workspace).is_ok());
    let default = EighFullPlan::new(&stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = default.workspace().unwrap();
    assert!(matches!(
        default.execute(&stack, &mut workspace),
        Err(BatchError::MemberRejected { members }) if members == [(0, MemberFault::NotHermitian)]
    ));
}

/// exp's spectral-route predicate (64·eps) is looser than eigh's default
/// admission in single precision (`eps^(3/4) < 64·eps` for f32), so a block
/// in between takes the spectral route and must still be admitted there.
#[test]
fn single_precision_exp_admits_what_its_route_predicate_sends_to_eigh() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = leg();
    // Relative residual 7e-6: above eps(f32)^(3/4) = 6.4e-6, below
    // 64·eps(f32) = 7.6e-6.
    let delta = (7.0e-6 * 10.0_f64.sqrt()) as f32;
    let t: TensorMap<_, f32> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| match index {
            [0, 0] => 1.0,
            [1, 0] => delta,
            [1, 1] => 2.0,
            _ => 0.0,
        })
        .unwrap();
    assert!(t.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).is_err());
    let exp = t.exp(&[0], &[1]).unwrap();
    assert!(exp
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
}
