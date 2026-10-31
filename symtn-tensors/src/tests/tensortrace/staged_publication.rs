use super::*;
use crate::probe;
use crate::tensortrace::tensortrace_checked_generic_in;
use std::cell::Cell;
use std::rc::Rc;
use symtn_core::{clear_structure_caches, structure_cache_info, StructureCacheKind};

fn isolated(name: &str) -> bool {
    if std::env::var_os("SYMTN_STAGED_TRACE_CHILD").is_some() {
        return true;
    }
    let name = format!("tests::tensortrace::staged_publication::{name}");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--nocapture"])
        .env("SYMTN_STAGED_TRACE_CHILD", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed; 0 failed;")
    );
    false
}

fn source() -> BoundDynamicFusionMapSpace<CheckedTraceToy> {
    let leg = || SectorLeg::new([(SectorId::new(0), 2)], false);
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedTraceToy::new(symtn_core::BraidingStyleKind::Bosonic)),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        ),
    )
    .unwrap()
}

fn stage(
    src: &BoundDynamicFusionMapSpace<CheckedTraceToy>,
) -> crate::TensorTraceStage<'_, CheckedTraceToy, crate::PreparedCheckedGenericDynamicSpace> {
    crate::tensortrace_stage_checked_generic(src, TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]), 1)
        .unwrap()
}

/// The complete-layout owner's admission state; sector-cache admission after
/// valid enumeration is allowed and deliberately not observed.
fn owner() -> (u64, usize, u64) {
    let info = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    (info.admissions(), info.entries(), info.charged_bytes())
}

/// Every caller resets the owner before staging, so nonpublication is exact.
fn assert_unpublished() {
    assert_eq!(owner(), (0, 0, 0));
}

#[test]
fn raw_execution_failure_does_not_publish() {
    if !isolated("raw_execution_failure_does_not_publish") {
        return;
    }
    let src = source();
    clear_structure_caches();
    let dst = stage(&src);
    let payloads = Rc::new(Cell::new(0));
    let executions = Rc::new(Cell::new(0));
    let seen_payloads = Rc::clone(&payloads);
    let seen_executions = Rc::clone(&executions);
    TRACE_TEST_TWIST_CALLS.set(0);
    probe::set_trace_raw_execution_hook(Box::new(move || {
        assert_eq!(seen_payloads.get(), 1);
        assert!(TRACE_TEST_TWIST_CALLS.get() > 0);
        seen_executions.set(seen_executions.get() + 1);
        Err(OperationError::StridedKernel {
            message: "injected shared raw failure".into(),
        })
    }));
    let error = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |_| None,
        || {
            payloads.set(payloads.get() + 1);
            vec![1.0; 16]
        },
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap_err();
    assert!(
        matches!(error, crate::CheckedGenericPlanError::Operation(OperationError::StridedKernel { message }) if message == "injected shared raw failure")
    );
    assert_eq!(payloads.get(), 1);
    assert_eq!(executions.get(), 1);
    assert_unpublished();
    // Nor are its trace permutation groups (#2072).
    assert_eq!(crate::probe::take_trace_column_activity().publications, 0);
}

#[test]
fn pivotal_failure_precedes_payload_and_raw_execution() {
    if !isolated("pivotal_failure_precedes_payload_and_raw_execution") {
        return;
    }
    let src = source();
    clear_structure_caches();
    let dst = stage(&src);
    TRACE_TEST_FAIL_TWIST.set(true);
    probe::set_trace_raw_execution_hook(Box::new(|| panic!("raw execution after pivotal failure")));
    let error = tensortrace_checked_generic_in::<_, f64, Vec<f64>>(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |_| None,
        || panic!("payload after pivotal failure"),
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap_err();
    assert!(
        matches!(error, crate::CheckedGenericPlanError::Provider(error) if error.to_string() == "trace pivotal failure")
    );
    probe::take_trace_raw_execution_hook();
    TRACE_TEST_FAIL_TWIST.set(false);
    assert_unpublished();
    assert_eq!(crate::probe::take_trace_column_activity().publications, 0);
}

#[test]
fn destination_error_precedes_pivotal_and_payload() {
    if !isolated("destination_error_precedes_pivotal_and_payload") {
        return;
    }
    let src = source();
    clear_structure_caches();
    // A stage whose source is another binding is compiled as against a
    // committed destination, so its `dst` comparison runs.
    let other = src.clone();
    let wrong_dst = crate::TensorTraceStage {
        destination: src
            .prepare_final_homspace_generic_with_checked(
                src.provider(),
                FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
            )
            .unwrap(),
        src: &other,
        axes: TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
    };
    TRACE_TEST_FAIL_TWIST.set(true);
    TRACE_TEST_TWIST_CALLS.set(0);
    let error = tensortrace_checked_generic_in::<_, f64, Vec<f64>>(
        strided_kernel::Exec::serial(),
        wrong_dst,
        &src,
        |_| None,
        || panic!("payload after destination failure"),
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::CheckedGenericPlanError::Operation(OperationError::StructureMismatch {
            tensor: "dst"
        })
    ));
    assert_eq!(TRACE_TEST_TWIST_CALLS.get(), 0);
    TRACE_TEST_FAIL_TWIST.set(false);
    assert_unpublished();
}

#[test]
fn success_uses_independent_partial_trace_and_commits_after_execution() {
    if !isolated("success_uses_independent_partial_trace_and_commits_after_execution") {
        return;
    }
    let src = source();
    clear_structure_caches();
    let dst = stage(&src);
    let payload = (0..16)
        .map(|i| num_complex::Complex64::new(i as f64 + 1.0, 2.0 * i as f64 - 3.0))
        .collect::<Vec<_>>();
    let expected = (0..4)
        .map(|i| {
            let (row, col) = (i % 2, i / 2);
            (0..2)
                .map(|k| payload[row + 2 * k + 4 * col + 8 * k])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let executions = Rc::new(Cell::new(0));
    let seen = Rc::clone(&executions);
    probe::set_trace_raw_execution_hook(Box::new(move || {
        assert_unpublished();
        seen.set(seen.get() + 1);
        Ok(())
    }));
    let calls = Cell::new(0);
    let (out, data) = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |_| panic!("read asked for a non-scalar destination"),
        || {
            calls.set(calls.get() + 1);
            &payload[..]
        },
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        num_complex::Complex64::new(1.0, 0.0),
    )
    .unwrap();
    assert_eq!(data, expected);
    assert_eq!(calls.get(), 1);
    assert_eq!(executions.get(), 1);
    assert!(Arc::ptr_eq(src.provider_arc(), out.provider_arc()));
    let (admissions, entries, bytes) = owner();
    assert_eq!((admissions, entries), (1, 1));
    assert!(bytes > 0);
    // Its trace permutation groups are published after the commit (#2072).
    let activity = crate::probe::take_trace_column_activity();
    assert!(activity.publications > 0);
    assert_eq!(activity.publications, activity.misses);
}

#[test]
fn admission_runs_before_payload_and_again_after_execution() {
    if !isolated("admission_runs_before_payload_and_again_after_execution") {
        return;
    }
    let src = source();
    clear_structure_caches();
    let dst = stage(&src);
    TRACE_TEST_WRONG_STYLE.set(true);
    let early = tensortrace_checked_generic_in::<_, f64, Vec<f64>>(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |_| None,
        || panic!("payload before admission"),
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap_err();
    TRACE_TEST_WRONG_STYLE.set(false);
    assert_unpublished();
    let dst = stage(&src);
    let executions = Rc::new(Cell::new(0));
    let seen = Rc::clone(&executions);
    probe::set_trace_raw_execution_hook(Box::new(move || {
        seen.set(1);
        Ok(())
    }));
    let late = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |_| None,
        || {
            TRACE_TEST_WRONG_STYLE.set(true);
            vec![1.0; 16]
        },
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap_err();
    TRACE_TEST_WRONG_STYLE.set(false);
    let crate::CheckedGenericPlanError::Operation(early) = early else {
        panic!("wrong early error")
    };
    let crate::CheckedGenericPlanError::Operation(late) = late else {
        panic!("wrong late error")
    };
    assert_eq!(early, late);
    assert_eq!(executions.get(), 1);
    assert_unpublished();
}

#[test]
fn warm_failure_preserves_winner_and_reset_stale_success_stays_uncached() {
    if !isolated("warm_failure_preserves_winner_and_reset_stale_success_stays_uncached") {
        return;
    }
    let src = source();
    clear_structure_caches();
    let (winner, _) = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        stage(&src),
        &src,
        |_| None,
        || vec![1.0; 16],
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap();
    let before = owner();
    assert_eq!(before.1, 1);
    probe::set_trace_raw_execution_hook(Box::new(|| {
        Err(OperationError::StridedKernel {
            message: "warm failure".into(),
        })
    }));
    let failed = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        stage(&src),
        &src,
        |_| None,
        || vec![1.0; 16],
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    );
    assert!(failed.is_err());
    assert_eq!(owner(), before);
    let (again, _) = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        stage(&src),
        &src,
        |_| None,
        || vec![1.0; 16],
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap();
    assert!(Arc::ptr_eq(
        winner.space().structure(),
        again.space().structure()
    ));
    assert_eq!(owner(), before);
    let staged = stage(&src);
    let stale = staged.destination.structure().content_key();
    clear_structure_caches();
    let (out, data) = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        staged,
        &src,
        |_| None,
        || vec![1.0; 16],
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap();
    assert_eq!(data, vec![2.0; 4]);
    assert!(Arc::ptr_eq(&stale, &out.space().structure().content_key()));
    // A retained pre-reset hit stays usable but is not republished.
    assert_unpublished();

    drop((winner, again, out));
    clear_structure_caches();
    let staged_miss = stage(&src);
    clear_structure_caches();
    let (_, data) = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        staged_miss,
        &src,
        |_| None,
        || vec![1.0; 16],
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        1.0,
    )
    .unwrap();
    assert_eq!(data, vec![2.0; 4]);
    // A miss staged before reset succeeds uncached under the new epoch.
    assert_unpublished();
}

fn bond_source() -> BoundDynamicFusionMapSpace<CheckedTraceToy> {
    let leg = || SectorLeg::new([(SectorId::new(0), 2)], false);
    BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedTraceToy::new(symtn_core::BraidingStyleKind::Bosonic)),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap()
}

#[test]
fn read_answers_a_scalar_destination_in_the_same_transaction() {
    // What (#1866): a scalar destination may be answered from the compiled
    // terms. The answer is scaled by `alpha`, neither the payload nor the
    // execution runs, and the destination still commits. A conjugated source
    // is not offered to `read`.
    if !isolated("read_answers_a_scalar_destination_in_the_same_transaction") {
        return;
    }
    let src = bond_source();
    clear_structure_caches();
    let axes = TensorTraceAxisSpec::new(&[], &[0], &[1]);
    let dst = crate::tensortrace_stage_checked_generic(&src, axes, 0).unwrap();
    probe::set_trace_raw_execution_hook(Box::new(|| panic!("executed after a read")));
    let terms = Cell::new(0);
    let (out, data) = tensortrace_checked_generic_in::<_, f64, Vec<f64>>(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |structure| {
            terms.set(structure.terms().len());
            Some(3.0)
        },
        || panic!("payload read after a read"),
        axes,
        2.0,
    )
    .unwrap();
    assert!(probe::take_trace_raw_execution_hook().is_some());
    assert_eq!(terms.get(), 1);
    assert_eq!(data, vec![6.0]);
    assert_eq!(out.space().nout() + out.space().nin(), 0);
    assert_eq!(owner().1, 1, "the scalar destination commits");

    let conjugated = TensorTraceAxisSpec::new_with_conjugation(&[], &[0], &[1], true);
    let dst = crate::tensortrace_stage_checked_generic(&src, conjugated, 0).unwrap();
    let (_, data) = tensortrace_checked_generic_in(
        strided_kernel::Exec::serial(),
        dst,
        &src,
        |_| panic!("read asked for a conjugated source"),
        || vec![1.0, 0.0, 0.0, 1.0],
        conjugated,
        2.0,
    )
    .unwrap();
    assert_eq!(data, vec![4.0]);
}
