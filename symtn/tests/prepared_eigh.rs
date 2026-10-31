//! Host gates of `EighFullPlan`.
//!
//! Per member the plan must equal Host eager `eigh_full` (gauge-fixed, so
//! the eigenvectors themselves are compared) and the dense Jacobi oracle of
//! `prepared/eigh.rs`, over U(1), SU(2) and fZ2xU(1) with several coupled
//! sectors and degeneracies above one. A batch fails as a whole and names
//! every failing member.

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
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0), runtime);
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
fn complex_payloads_and_other_signatures_are_typed_errors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let complex = members::<_, Complex64>(&runtime, &[&v], &[&v], 2, 1);
    let stack = StackedTensorMap::pack(&complex).unwrap();
    assert!(matches!(
        EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT),
        Err(Error::Operation(error)) if matches!(
            *error,
            OperationError::UnsupportedTensorContractScope { message }
                if message.contains("real payloads")
        )
    ));

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
