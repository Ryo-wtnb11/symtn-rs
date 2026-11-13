//! Host gates of `EighFullPlan`.
//!
//! Per member the plan must equal Host eager `eigh_full` (gauge-fixed, so
//! the eigenvectors themselves are compared) and the dense Jacobi oracle of
//! `prepared/eigh.rs`, over U(1), SU(2) and fZ2xU(1) with several coupled
//! sectors and degeneracies above one. Complex members (c32, c64) are also
//! checked against a hand-made `U diag(λ) Uᴴ` per coupled sector: its
//! eigenvalues and cluster projectors, reconstruction, unitarity and the
//! `gaugefix!` rule. A batch fails as a whole and names every failing member.

use crate::backend;

use crate::numerics;
use crate::prepared;

use crate::report;
use report::Report;

use std::fmt::Debug;
use symtn::typed::HermitianTol;

use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{
    BatchError, Eigh, EighFullPlan, EighFullWorkspace, GradedSpace, MemberFault, Runtime,
    SignatureField, StackedTensorMap, TensorMap,
};
use symtn::typed::{Error, OperationError};

use prepared::eigh::{
    check_member, degenerate_entry, has_degenerate_group, has_plus_minus_tie, hermitian_members,
    plus_minus_entry, single_leg,
};
use prepared::{fz2u1_legs, members, su2_legs, u1_legs};

use crate::fixtures;
use crate::truncation_oracle::Matrix;

use fixtures::{codomain_axes, domain_axes};

const MEMBER_COUNTS: [usize; 3] = [1, 2, 7];

/// Runs the plan on `inputs` and checks every member against Host eager
/// and the oracle. At `B = 1` the factors must equal eager bit for bit
/// (IEEE `==`); at any `B` the Host plan runs the eager code per member,
/// so its gauge-fixed `v` equals eager's within the numerics rule.
fn check_batch<R>(label: &str, runtime: &Runtime, inputs: &[TensorMap<R, f64>])
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _ = runtime;
    let stack = StackedTensorMap::pack(inputs).unwrap();
    let plan = EighFullPlan::new(
        engine,
        &stack,
        &codomain_axes(&inputs[0]),
        &domain_axes(&inputs[0]),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let output = plan.execute(engine, &stack, &mut ws).unwrap();
    assert_eq!(
        (output.d.len(), output.v.len()),
        (inputs.len(), inputs.len())
    );
    assert_eq!(output.spectra.len(), inputs.len());
    for (member, input) in inputs.iter().enumerate() {
        let what = format!("{label} member {member}/{}", inputs.len());
        let Eigh {
            d: eager_d,
            v: eager_v,
        } = input
            .eigh_full(
                engine,
                &codomain_axes(input),
                &domain_axes(input),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        let d = output.d.member(engine, member).unwrap();
        let v = output.v.member(engine, member).unwrap();
        assert!(
            d.structure_signature() == eager_d.structure_signature(),
            "{what}: d space"
        );
        assert!(
            v.structure_signature() == eager_v.structure_signature(),
            "{what}: v space"
        );
        if inputs.len() == 1 {
            assert!(
                d.dense_data().unwrap()
                    == eager_d.materialize(engine).unwrap().dense_data().unwrap(),
                "{what}: d bit-identical"
            );
            assert!(
                v.dense_data().unwrap() == eager_v.dense_data().unwrap(),
                "{what}: v bit-identical"
            );
        }
        let terms = input.dense_data().unwrap().len();
        numerics::assert_nonzero_slices_close(
            &format!("{what}: gauge-fixed v"),
            v.dense_data().unwrap(),
            eager_v.dense_data().unwrap(),
            terms,
        );
        numerics::assert_nonzero_slices_close(
            &format!("{what}: d"),
            d.dense_data().unwrap(),
            eager_d.materialize(engine).unwrap().dense_data().unwrap(),
            terms,
        );
        // The plan's spectra come from the factorization with vectors, eager
        // `eigh_vals` from the values-only driver call; a LAPACK provider may
        // round those two routines differently, so the sectors match exactly
        // and the values to the numerics rule.
        let eager_spectra = input
            .eigh_vals(
                engine,
                &codomain_axes(input),
                &domain_axes(input),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        assert_eq!(
            output.spectra[member].len(),
            eager_spectra.len(),
            "{what}: spectra sectors"
        );
        for (got, want) in output.spectra[member].iter().zip(&eager_spectra) {
            assert!(got.sector == want.sector, "{what}: spectra sector order");
            numerics::assert_slices_close(
                &format!("{what}: spectra"),
                &got.values,
                &want.values,
                terms,
            );
        }
        // On faer the values-only and full drivers agree bitwise.
        assert!(
            !backend::faer_is_default_backend() || output.spectra[member] == eager_spectra,
            "{what}: spectra bitwise"
        );
        check_member(&what, input, &d, &v, (&eager_d, &eager_v), f64::EPSILON);
    }
}

fn equivalence<R>(label: &str, leg: GradedSpace<R>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
{
    let runtime = Runtime::builder().build().unwrap();
    for count in MEMBER_COUNTS {
        let inputs = hermitian_members(&runtime, &[&leg, &leg], count, 3);
        assert!(inputs[0].subblock_count() > 3, "{label}: several blocks");
        check_batch(&format!("{label} B={count}"), &runtime, &inputs);
    }
}

#[test]
fn host_members_equal_host_eager_and_the_dense_oracle() {
    equivalence("u1", u1_legs().0);
    equivalence("su2", su2_legs().0);
    equivalence("fz2u1", fz2u1_legs().0);
}

#[test]
fn plan_supports_independent_workspaces_changing_b_and_rejects_foreign_workspace() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let inputs = hermitian_members(&runtime, &[&leg], 4, 3);
    let four = StackedTensorMap::pack(&inputs).unwrap();
    let two = StackedTensorMap::pack(&inputs[..2]).unwrap();
    let plan = EighFullPlan::new(engine, &four, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut first = plan.workspace().unwrap();
    let mut second = plan.workspace().unwrap();

    assert_eq!(plan.execute(engine, &four, &mut first).unwrap().d.len(), 4);
    let first_bytes = first.retained_bytes();
    assert_eq!(plan.execute(engine, &two, &mut second).unwrap().d.len(), 2);
    assert!(second.retained_bytes() > 0);
    assert_eq!(plan.execute(engine, &two, &mut first).unwrap().d.len(), 2);
    assert_ne!(
        first.retained_bytes(),
        first_bytes,
        "workspace capacity follows B"
    );

    let foreign = EighFullPlan::new(engine, &four, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut foreign_workspace = foreign.workspace().unwrap();
    foreign
        .execute(engine, &four, &mut foreign_workspace)
        .unwrap();
    assert!(matches!(
        plan.execute(engine, &four, &mut foreign_workspace),
        Err(symtn::typed::BatchError::Operation(symtn::typed::Error::InvalidArgument(reason)))
            if reason.contains("another plan")
    ));
    assert!(
        foreign_workspace.take_output().is_some(),
        "identity rejection leaves another plan's workspace untouched"
    );
}

#[test]
fn plan_workspace_failure_hides_output_and_recovers() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let good = hermitian_members(&runtime, &[&leg], 3, 11);
    let mut bad = good.clone();
    bad[1] = members::<_, f64>(&runtime, &[&leg], &[&leg], 1, 12).remove(0);
    let good_stack = StackedTensorMap::pack(&good).unwrap();
    let bad_stack = StackedTensorMap::pack(&bad).unwrap();
    let plan = EighFullPlan::new(engine, &good_stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();

    plan.execute(engine, &good_stack, &mut workspace).unwrap();
    assert!(matches!(
        plan.execute(engine, &bad_stack, &mut workspace).map(|_| ()),
        Err(BatchError::MemberRejected { .. })
    ));
    assert!(workspace.take_output().is_none());

    let output = plan.execute(engine, &good_stack, &mut workspace).unwrap();
    for (member, input) in good.iter().enumerate() {
        let Eigh { d, v } = input
            .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert!(
            output
                .d
                .member(engine, member)
                .unwrap()
                .dense_data()
                .unwrap()
                == d.materialize(engine).unwrap().dense_data().unwrap()
        );
        assert!(
            output
                .v
                .member(engine, member)
                .unwrap()
                .dense_data()
                .unwrap()
                == v.dense_data().unwrap()
        );
    }
}

#[test]
fn plus_minus_lambda_and_degenerate_groups_compare_by_value() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_legs().0;
    let j = SU2Irrep::from_twice_spin;
    let su2 =
        GradedSpace::try_new(std::sync::Arc::new(SU2FusionRule), [(j(0), 3), (j(1), 4)]).unwrap();
    for count in [1, 5] {
        let pm = single_leg(&runtime, &leg, count, plus_minus_entry);
        let Eigh { d, .. } = pm[0]
            .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert!(has_plus_minus_tie(&d), "the ±λ fixture has |λ| ties");
        check_batch(&format!("u1 ±λ B={count}"), &runtime, &pm);
        let degenerate = single_leg(&runtime, &su2, count, degenerate_entry);
        let Eigh { d, .. } = degenerate[0]
            .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert!(
            has_degenerate_group(&d),
            "the degenerate fixture has a group"
        );
        check_batch(&format!("su2 degenerate B={count}"), &runtime, &degenerate);
    }
}

/// Eager's verdict: `Some(fault)` when eager `eigh_full` rejects the member.
fn eager_fault<R>(input: &TensorMap<R, f64>) -> Option<MemberFault>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let error = input
        .eigh_full(
            engine,
            &codomain_axes(input),
            &domain_axes(input),
            HermitianTol::DEFAULT,
        )
        .err()?;
    let Error::Operation(error) = error else {
        panic!("unexpected eager error {error:?}")
    };
    Some(match *error {
        OperationError::InvalidArgument { .. } => MemberFault::NotHermitian,
        OperationError::Dense(symtn_dense::DenseError::NumericalFailure { .. }) => {
            MemberFault::NonFiniteEigenvalue
        }
        other => panic!("unexpected eager error {other:?}"),
    })
}

fn rejected_members(result: Result<(), BatchError>) -> Vec<(usize, MemberFault)> {
    match result {
        Ok(()) => Vec::new(),
        Err(BatchError::MemberRejected { members }) => members,
        Err(other) => panic!("unexpected batch error {other:?}"),
    }
}

/// Members spanning ±1000 binary orders of magnitude, Hermitian and not.
pub fn mixed_scale_members<R>(runtime: &Runtime, leg: &GradedSpace<R>) -> Vec<TensorMap<R, f64>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let h = hermitian_members(runtime, &[leg, leg], 1, 5).remove(0);
    let x = members::<R, f64>(runtime, &[leg, leg], &[leg, leg], 1, 9).remove(0);
    // `2^-30` relative skew rejects (`> 64 eps`); `2^-60` is admitted.
    let rejected = h.axpby(engine, 1.0, &x, 2f64.powi(-30)).unwrap();
    let admitted = h.axpby(engine, 1.0, &x, 2f64.powi(-60)).unwrap();
    let mut out = Vec::new();
    for exponent in [0, -1000, 1000, -1060] {
        let s = 2f64.powi(exponent);
        out.push(h.scale(engine, s).unwrap());
        out.push(rejected.scale(engine, s).unwrap());
        out.push(admitted.scale(engine, s).unwrap());
    }
    out
}

#[test]
fn mixed_scale_admission_verdicts_equal_eager_per_member() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let inputs = mixed_scale_members(&runtime, &leg);
    let expected: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter_map(|(member, input)| eager_fault(input).map(|fault| (member, fault)))
        .collect();
    assert!(
        expected.len() >= 3,
        "the fixture rejects some members: {expected:?}"
    );
    assert!(expected.len() < inputs.len(), "and admits others");
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0, 1], &[2, 3], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    let got = rejected_members(plan.execute(engine, &stack, &mut ws).map(|_| ()));
    assert_eq!(got, expected);
}

#[test]
fn every_non_hermitian_or_non_finite_member_is_named() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = su2_legs();
    let mut inputs = hermitian_members(&runtime, &[&leg, &leg], 6, 1);
    let x = members::<_, f64>(&runtime, &[&leg, &leg], &[&leg, &leg], 1, 4).remove(0);
    inputs[1] = x.clone();
    inputs[4] = x;
    inputs[5] = inputs[5].scale(engine, f64::NAN).unwrap();
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0, 1], &[2, 3], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    let got = rejected_members(plan.execute(engine, &stack, &mut ws).map(|_| ()));
    let fault = MemberFault::NotHermitian;
    assert_eq!(got, vec![(1, fault), (4, fault), (5, fault)]);
    assert!(
        ws.take_output().is_none(),
        "no output after a rejected batch"
    );
}

#[test]
fn a_non_finite_eigenvalue_rejects_its_members() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    // `M [[1, 1], [1, 1]]`-like sector matrices with `M` near the largest
    // double: finite and Hermitian, but `2M` overflows.
    let huge = f64::MAX / 1.5;
    let mut inputs = single_leg(&runtime, &leg, 4, |_, _, _| 1.0);
    for member in [0, 2] {
        inputs[member] = inputs[member].scale(engine, huge).unwrap();
    }
    let expected: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter_map(|(member, input)| eager_fault(input).map(|fault| (member, fault)))
        .collect();
    assert_eq!(
        expected,
        vec![
            (0, MemberFault::NonFiniteEigenvalue),
            (2, MemberFault::NonFiniteEigenvalue)
        ],
        "eager rejects the overflowing members"
    );
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    assert_eq!(
        rejected_members(plan.execute(engine, &stack, &mut ws).map(|_| ())),
        expected
    );
}

#[test]
fn other_signatures_and_non_endomorphisms_are_typed_errors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let inputs = hermitian_members(&runtime, &[&v], 2, 1);
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    let other = StackedTensorMap::pack(&hermitian_members(&runtime, &[&w], 2, 1)).unwrap();
    assert!(matches!(
        plan.execute(engine, &other, &mut ws).map(|_| ()),
        Err(BatchError::Operation(Error::BatchSignatureMismatch {
            member: None,
            field: SignatureField::HomSpace
        }))
    ));
    let rectangular = members::<_, f64>(&runtime, &[&v], &[&w], 1, 1);
    assert!(EighFullPlan::new(
        engine,
        &StackedTensorMap::pack(&rectangular).unwrap(),
        &[0],
        &[1],
        HermitianTol::DEFAULT
    )
    .err()
    .unwrap()
    .report()
    .contains("eigh requires an endomorphism"));
}

#[test]
fn warm_calls_reuse_the_outputs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = fz2u1_legs();
    let inputs = hermitian_members(&runtime, &[&leg, &leg], 3, 2);
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0, 1], &[2, 3], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    let first = plan
        .execute(engine, &stack, &mut ws)
        .unwrap()
        .v
        .member(engine, 2)
        .unwrap();
    let retained = ws.retained_bytes();
    assert!(retained > 0);
    let second = plan
        .execute(engine, &stack, &mut ws)
        .unwrap()
        .v
        .member(engine, 2)
        .unwrap();
    assert!(
        first.dense_data().unwrap() == second.dense_data().unwrap(),
        "deterministic replay"
    );
    assert_eq!(ws.retained_bytes(), retained, "flat on a warm call");
    let (d, v) = ws.take_output().unwrap();
    assert_eq!((d.len(), v.len()), (3, 3));
    assert!(ws.retained_bytes() < retained);
}

/// Every member of `output` equals Host eager bit for bit (the Host plan
/// runs the eager per-member code), so `d` is exactly zero off its diagonal.
fn assert_equals_eager<R>(
    what: &str,
    plan: &EighFullPlan<R, f64>,
    ws: &mut EighFullWorkspace<R, f64>,
    inputs: &[TensorMap<R, f64>],
    stack: &StackedTensorMap<R, f64>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let output = plan.execute(engine, stack, ws).unwrap();
    for (member, input) in inputs.iter().enumerate() {
        let Eigh { d, v } = input
            .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert!(
            output
                .d
                .member(engine, member)
                .unwrap()
                .dense_data()
                .unwrap()
                == d.materialize(engine).unwrap().dense_data().unwrap(),
            "{what}: d {member}"
        );
        assert!(
            output
                .v
                .member(engine, member)
                .unwrap()
                .dense_data()
                .unwrap()
                == v.dense_data().unwrap(),
            "{what}: v {member}"
        );
    }
}

#[test]
fn a_failed_batch_leaves_no_observable_output_and_the_next_call_is_whole() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: after a non-finite or non-Hermitian rejection `take_output` is
    // `None`; the reused buffers of the next successful call hold exactly
    // eager's factors, `d` zero off its diagonal; the same after
    // `take_output` and after a change of `B`.
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let good = single_leg(&runtime, &leg, 4, |member, i, j| {
        1.0 + (i + j + member) as f64 / 4.0 + f64::from(u8::from(i == j))
    });
    let mut overflowing = good.clone();
    overflowing[2] = overflowing[2].scale(engine, f64::MAX / 1.5).unwrap();
    let mut skewed = good.clone();
    skewed[1] = members::<_, f64>(&runtime, &[&leg], &[&leg], 1, 3).remove(0);
    let stack = |inputs: &[TensorMap<_, f64>]| StackedTensorMap::pack(inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack(&good), &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    assert_equals_eager("first", &plan, &mut ws, &good, &stack(&good));

    let failed = plan
        .execute(engine, &stack(&overflowing), &mut ws)
        .map(|_| ());
    assert!(matches!(failed, Err(BatchError::MemberRejected { .. })));
    assert!(
        ws.take_output().is_none(),
        "no output after a non-finite rejection"
    );
    assert_equals_eager("after non-finite", &plan, &mut ws, &good, &stack(&good));

    let failed = plan.execute(engine, &stack(&skewed), &mut ws).map(|_| ());
    assert!(matches!(failed, Err(BatchError::MemberRejected { .. })));
    assert!(
        ws.take_output().is_none(),
        "no output after a non-Hermitian rejection"
    );
    assert_equals_eager("after non-Hermitian", &plan, &mut ws, &good, &stack(&good));

    assert!(ws.take_output().is_some());
    assert_equals_eager("after take_output", &plan, &mut ws, &good, &stack(&good));
    assert_equals_eager("B change", &plan, &mut ws, &good[..2], &stack(&good[..2]));
}

/// Bit-identity pin: through one workspace, at `B = 1, 2, 17, 1`
/// (buffers reallocated on every change of `B`), every member's `d`, `v` and
/// spectra equal Host eager `eigh_full` of that member exactly, over
/// U(1), SU(2) and fZ2xU(1).
fn bitwise_across_member_counts<R>(label: &str, leg: GradedSpace<R>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let all = hermitian_members(&runtime, &[&leg, &leg], 17, 21);
    let stack = StackedTensorMap::pack(&all[..1]).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0, 1], &[2, 3], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    for count in [1, 2, 17, 1] {
        let inputs = &all[..count];
        let output = plan
            .execute(engine, &StackedTensorMap::pack(inputs).unwrap(), &mut ws)
            .unwrap();
        for (member, input) in inputs.iter().enumerate() {
            let what = format!("{label} B={count} member {member}");
            let Eigh { d, v } = input
                .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
                .unwrap();
            let d = d.materialize(engine).unwrap();
            let got_d = output.d.member(engine, member).unwrap();
            assert!(
                got_d.dense_data().unwrap() == d.dense_data().unwrap(),
                "{what}: d"
            );
            assert!(
                output
                    .v
                    .member(engine, member)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    == v.dense_data().unwrap(),
                "{what}: v"
            );
            // The spectra are `d`'s diagonal: equal nonzero bit patterns as
            // multisets (`d` has no other nonzero entry), each ascending.
            let mut spectra: Vec<u64> = output.spectra[member]
                .iter()
                .inspect(|entry| {
                    assert!(entry.values.is_sorted(), "{what}: ascending");
                })
                .flat_map(|entry| entry.values.iter())
                .filter(|value| **value != 0.0)
                .map(|value| value.to_bits())
                .collect();
            let mut diagonal: Vec<u64> = d
                .dense_data()
                .unwrap()
                .iter()
                .filter(|value| **value != 0.0)
                .map(|value| value.to_bits())
                .collect();
            spectra.sort_unstable();
            diagonal.sort_unstable();
            assert!(spectra == diagonal, "{what}: spectra");
        }
    }
}

#[test]
fn members_equal_eager_bitwise_across_member_counts() {
    bitwise_across_member_counts("u1", u1_legs().0);
    bitwise_across_member_counts("su2", su2_legs().0);
    bitwise_across_member_counts("fz2u1", fz2u1_legs().0);
}

/// Complex members: the bounds of a payload the Host plan factors.
trait ComplexPayload: crate::common::Payload + symtn::typed::FactorizationScalar {}
impl ComplexPayload for Complex32 {}
impl ComplexPayload for Complex64 {}

/// One coupled sector of a hand-made Hermitian `H = U diag(λ) Uᴴ`: `λ`
/// ascending and the unitary `U` whose column `k` is `λ_k`'s eigenvector.
struct SectorOracle<S> {
    sector: S,
    values: Vec<f64>,
    vectors: Matrix,
}

/// An `n x n` unitary with complex phases and unequal entry magnitudes: two
/// sweeps of complex Givens rotations `[[c, -s̄], [s, c]]` on adjacent
/// columns of the identity, angles drawn from `seed`.
fn givens_unitary(n: usize, seed: usize) -> Matrix {
    let mut u = Matrix::zeros(n, n);
    for i in 0..n {
        u.data[i + i * n] = Complex64::ONE;
    }
    let frac = |x: f64| x - x.floor();
    for sweep in 0..2 {
        for p in 0..n.saturating_sub(1) {
            let k = (seed * 31 + sweep * 17 + p * 7 + 1) as f64;
            let theta = 0.3 + 0.9 * frac(k * 0.618_033_988_75);
            let phi = std::f64::consts::TAU * frac(k * 0.414_213_562_37);
            let (c, s) = (theta.cos(), Complex64::from_polar(theta.sin(), phi));
            for r in 0..n {
                let (a, b) = (u.data[r + p * n], u.data[r + (p + 1) * n]);
                u.data[r + p * n] = a * c + b * s;
                u.data[r + (p + 1) * n] = b * c - a * s.conj();
            }
        }
    }
    u
}

/// The spectrum of coupled sector order `n` for `member`: exactly repeated
/// triples (even members) or triples clustered `2^-30` apart (odd members)
/// in the lower half, distinct values above, scaled by `2^-40`, `1` or
/// `2^40` so members of one batch differ by up to `2^80`.
fn spectrum(member: usize, salt: usize, n: usize) -> Vec<f64> {
    let scale = 2f64.powi(40 * ((member + salt) % 3) as i32 - 40);
    let cluster = if member % 2 == 1 { 2f64.powi(-30) } else { 0.0 };
    (0..n)
        .map(|k| {
            let group = if k < n / 2 { k / 3 } else { k };
            scale * (-1.0 + 0.4 * group as f64 + (k % 3) as f64 * cluster + 0.1 * salt as f64)
        })
        .collect()
}

/// `count` endomorphisms `leg ← leg` whose coupled sector of order `n` is
/// `U diag(spectrum) Uᴴ`, Hermitian by construction (the lower triangle is
/// the conjugate of the upper, the diagonal real), with the oracle of every
/// member.
#[allow(clippy::type_complexity)]
fn spectral_members<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    count: usize,
    salt: usize,
) -> (
    Vec<TensorMap<R, D>>,
    Vec<Vec<SectorOracle<<R as SectorCodec>::Sector>>>,
)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    // Each coupled sector's order, read from the block indices.
    let mut orders: Vec<(<R as SectorCodec>::Sector, usize)> = Vec::new();
    TensorMap::<R, D>::from_subblock_fn(engine, runtime, [leg], [leg], |trees, index| {
        match orders.iter_mut().find(|(s, _)| s == trees.coupled()) {
            Some((_, n)) => *n = (*n).max(index[0] + 1),
            None => orders.push((trees.coupled().clone(), index[0] + 1)),
        }
        D::entry(0.0, 0.0)
    })
    .unwrap();
    let mut tensors = Vec::new();
    let mut oracles = Vec::new();
    for member in 0..count {
        let oracle: Vec<_> = orders
            .iter()
            .enumerate()
            .map(|(position, (sector, n))| SectorOracle {
                sector: sector.clone(),
                values: spectrum(member, salt, *n),
                vectors: givens_unitary(*n, 97 * member + 13 * position + 5 * salt),
            })
            .collect();
        let h = |o: &SectorOracle<_>, i: usize, j: usize| -> Complex64 {
            let u = &o.vectors;
            (0..u.rows)
                .map(|k| u.at(i, k) * o.values[k] * u.at(j, k).conj())
                .sum()
        };
        let tensor = TensorMap::from_subblock_fn(engine, runtime, [leg], [leg], |trees, index| {
            let o = oracle
                .iter()
                .find(|o| &o.sector == trees.coupled())
                .unwrap();
            let (i, j) = (index[0], index[1]);
            let z = match i.cmp(&j) {
                std::cmp::Ordering::Less => h(o, i, j),
                std::cmp::Ordering::Equal => Complex64::new(h(o, i, i).re, 0.0),
                std::cmp::Ordering::Greater => h(o, j, i).conj(),
            };
            D::entry(z.re, z.im)
        })
        .unwrap();
        tensors.push(tensor);
        oracles.push(oracle);
    }
    (tensors, oracles)
}

/// `count` complex Hermitian `X + Xᴴ` over `codomain ← codomain`: several
/// fusion trees per coupled sector.
fn complex_hermitian_members<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    count: usize,
    salt: usize,
) -> Vec<TensorMap<R, D>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let one = D::entry(1.0, 0.0);
    members::<R, D>(runtime, codomain, codomain, count, salt)
        .into_iter()
        .map(|x| {
            let adjoint = x.adjoint(engine).unwrap();
            x.axpby(engine, one, &adjoint, one).unwrap()
        })
        .collect()
}

fn bits<D: crate::common::Payload>(data: &[D]) -> Vec<(u64, u64)> {
    data.iter()
        .map(|z| {
            let (re, im) = z.parts();
            (re.to_bits(), im.to_bits())
        })
        .collect()
}

/// One complex member's factors `(d, v)` of `input`:
///
/// - bit for bit Host eager `eigh_full`;
/// - per coupled sector, `d` ascending, `v d vᴴ = t` and `vᴴ v = 1`;
/// - MatrixAlgebraKit's `gaugefix!`: the largest-magnitude entry of each
///   column of `v` is real and positive;
/// - with an oracle: the eigenvalues equal `λ`, and the spectral projector
///   `Σ v_k v_kᴴ` of every eigenvalue cluster equals `Σ U_k U_kᴴ`.
///
/// Tolerances are `64 n eps` of the sector's largest `|λ|`, the projectors
/// further divided by the relative gap to the next cluster (Davis–Kahan).
fn check_complex_member<R, D>(
    what: &str,
    input: &TensorMap<R, D>,
    d: &TensorMap<R, D>,
    v: &TensorMap<R, D>,
    oracle: Option<&[SectorOracle<<R as SectorCodec>::Sector>]>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq + Debug,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let Eigh {
        d: eager_d,
        v: eager_v,
    } = input
        .eigh_full(
            engine,
            &codomain_axes(input),
            &domain_axes(input),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    assert!(
        bits(d.dense_data().unwrap())
            == bits(eager_d.materialize(engine).unwrap().dense_data().unwrap()),
        "{what}: d bit-identical to eager"
    );
    assert!(
        bits(v.dense_data().unwrap()) == bits(eager_v.dense_data().unwrap()),
        "{what}: v bit-identical to eager"
    );

    let eps = <D as crate::numerics::Numeric>::EPS;
    let t = sector_matrices!(engine, input);
    let dm = sector_matrices!(engine, d);
    let vm = sector_matrices!(engine, v);
    assert_eq!(t.len(), vm.len(), "{what}: sectors");
    for (sector, a) in &t {
        let what = format!("{what} sector {sector:?}");
        let n = a.rows;
        let find = |list: &[(_, Matrix)]| {
            list.iter()
                .find(|(s, _)| s == sector)
                .map(|(_, m)| m.clone())
                .unwrap()
        };
        let (dd, vv) = (find(&dm), find(&vm));
        assert_eq!((vv.rows, vv.cols), (n, n), "{what}: v shape");
        let values: Vec<f64> = (0..n).map(|k| dd.at(k, k).re).collect();
        assert!(values.is_sorted(), "{what}: ascending {values:?}");
        let scale = values.iter().fold(f64::MIN_POSITIVE, |m, x| m.max(x.abs()));
        let bound = 64.0 * n as f64 * eps;
        for i in 0..n {
            for j in 0..n {
                let rebuilt: Complex64 = (0..n)
                    .map(|k| vv.at(i, k) * values[k] * vv.at(j, k).conj())
                    .sum();
                assert!(
                    (rebuilt - a.at(i, j)).norm() <= bound * scale,
                    "{what}: (v d vᴴ)[{i},{j}] = {rebuilt} vs {}",
                    a.at(i, j)
                );
                let gram: Complex64 = (0..n).map(|k| vv.at(k, i).conj() * vv.at(k, j)).sum();
                let unit = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                assert!(
                    (gram - unit).norm() <= bound,
                    "{what}: (vᴴv)[{i},{j}] = {gram}"
                );
            }
        }
        for k in 0..n {
            let pivot = (0..n)
                .map(|i| vv.at(i, k))
                .max_by(|x, y| x.norm().total_cmp(&y.norm()))
                .unwrap();
            assert!(
                pivot.re > 0.0 && pivot.im.abs() <= 4.0 * eps * pivot.re,
                "{what}: column {k} pivot {pivot} is not real positive"
            );
        }
        let Some(oracle) = oracle else { continue };
        let o = oracle.iter().find(|o| &o.sector == sector).unwrap();
        for (got, want) in values.iter().zip(&o.values) {
            assert!(
                (got - want).abs() <= bound * scale,
                "{what}: λ {got} vs oracle {want}"
            );
        }
        // Clusters: maximal runs of oracle values within `1e-6` relative.
        let mut clusters: Vec<std::ops::Range<usize>> = Vec::new();
        for k in 0..n {
            match clusters.last_mut() {
                Some(c) if o.values[k] - o.values[k - 1] <= 1e-6 * scale => c.end = k + 1,
                _ => clusters.push(k..k + 1),
            }
        }
        let gap = clusters
            .windows(2)
            .map(|pair| o.values[pair[1].start] - o.values[pair[0].end - 1])
            .fold(scale, f64::min);
        for cluster in &clusters {
            for i in 0..n {
                for j in 0..n {
                    let got: Complex64 = cluster
                        .clone()
                        .map(|k| vv.at(i, k) * vv.at(j, k).conj())
                        .sum();
                    let want: Complex64 = cluster
                        .clone()
                        .map(|k| o.vectors.at(i, k) * o.vectors.at(j, k).conj())
                        .sum();
                    assert!(
                        (got - want).norm() <= bound * scale / gap,
                        "{what}: projector {cluster:?} [{i},{j}] {got} vs {want}"
                    );
                }
            }
        }
    }
}

/// Through one workspace, at `B = 1, 2, 17` and for two different inputs at
/// each `B`, every member of the plan's output passes
/// [`check_complex_member`].
fn complex_case<R, D>(label: &str, leg: GradedSpace<R>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq + Debug,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (first, _) = spectral_members::<R, D>(&runtime, &leg, 1, 0);
    let plan = EighFullPlan::new(
        engine,
        &StackedTensorMap::pack(&first).unwrap(),
        &[0],
        &[1],
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    for count in [1, 2, 17] {
        for salt in [0, 1] {
            let (inputs, oracles) = spectral_members::<R, D>(&runtime, &leg, count, salt);
            let stack = StackedTensorMap::pack(&inputs).unwrap();
            let output = plan.execute(engine, &stack, &mut ws).unwrap();
            for (member, input) in inputs.iter().enumerate() {
                check_complex_member(
                    &format!("{label} {} B={count} input {salt} member {member}", D::NAME),
                    input,
                    &output.d.member(engine, member).unwrap(),
                    &output.v.member(engine, member).unwrap(),
                    Some(&oracles[member]),
                );
            }
        }
    }
}

#[test]
fn complex_members_equal_host_eager_and_the_spectral_oracle() {
    // Coupled-sector orders on both sides of 32; a dual non-self-dual U(1)
    // leg and a non-Abelian SU(2) leg.
    let u1 = crate::legs::u1_leg(&[(-1, 4), (0, 31), (1, 33), (2, 48)])
        .try_dual()
        .unwrap();
    let su2 = crate::legs::su2_leg(&[(0, 5), (1, 33), (2, 3)]);
    complex_case::<_, Complex64>("dual u1", u1.clone());
    complex_case::<_, Complex32>("dual u1", u1);
    complex_case::<_, Complex64>("su2", su2.clone());
    complex_case::<_, Complex32>("su2", su2);
}

fn complex_multi_tree_case<R, D>(label: &str, codomain: &[&GradedSpace<R>])
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq + Debug,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let inputs = complex_hermitian_members::<R, D>(&runtime, codomain, 17, 3);
    assert!(inputs[0].subblock_count() > 3, "{label}: several blocks");
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let rows = codomain_axes(&inputs[0]);
    let cols = domain_axes(&inputs[0]);
    let plan = EighFullPlan::new(engine, &stack, &rows, &cols, HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    let output = plan.execute(engine, &stack, &mut ws).unwrap();
    for (member, input) in inputs.iter().enumerate() {
        check_complex_member(
            &format!("{label} {} member {member}", D::NAME),
            input,
            &output.d.member(engine, member).unwrap(),
            &output.v.member(engine, member).unwrap(),
            None,
        );
    }
}

#[test]
fn complex_members_with_several_trees_per_sector_equal_host_eager() {
    let (su2, _) = su2_legs();
    let (fz2u1, _) = fz2u1_legs();
    let fz2u1_dual = fz2u1.try_dual().unwrap();
    complex_multi_tree_case::<_, Complex64>("su2", &[&su2, &su2]);
    complex_multi_tree_case::<_, Complex32>("su2", &[&su2, &su2]);
    complex_multi_tree_case::<_, Complex64>("fz2u1", &[&fz2u1, &fz2u1_dual]);
    complex_multi_tree_case::<_, Complex32>("fz2u1", &[&fz2u1, &fz2u1_dual]);
}

fn complex_symmetric_member_is_rejected<D: ComplexPayload>() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let (mut inputs, _) = spectral_members::<_, D>(&runtime, &leg, 17, 0);
    // `Sᵀ = S` with imaginary off-diagonal entries: equal to its transpose,
    // not to its conjugate transpose.
    inputs[1] = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, index| {
        let (lo, hi) = (index[0].min(index[1]), index[0].max(index[1]));
        D::entry(1.0 + lo as f64, (hi - lo) as f64 / 2.0)
    })
    .unwrap();
    let stack = StackedTensorMap::pack(&inputs).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut ws = plan.workspace().unwrap();
    assert_eq!(
        rejected_members(plan.execute(engine, &stack, &mut ws).map(|_| ())),
        vec![(1, MemberFault::NotHermitian)],
        "{}",
        D::NAME
    );
    assert!(ws.take_output().is_none(), "{}: no output", D::NAME);
}

#[test]
fn a_complex_symmetric_non_hermitian_member_is_rejected() {
    complex_symmetric_member_is_rejected::<Complex64>();
    complex_symmetric_member_is_rejected::<Complex32>();
}
