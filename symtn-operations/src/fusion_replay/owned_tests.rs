//! `execute_owned`: an owned output is published iff every element was
//! written by its owner. The buffers start genuinely uninitialized, so
//! under Miri any read of an unwritten element, or a published `Vec` with
//! one, fails the test.

use super::gemm::job_pieces;
use super::tests::{four_block_plan, scalar_group, NaiveGemm, FOUR_BLOCK_LHS, FOUR_BLOCK_RHS};
use super::*;
use crate::kernel_adapter::observed::ObservedKernels;
use crate::owned_overwrite_buffer::write_zeros;
use symtn_core::BlockSpec;

/// How [`ScriptedGemm`] breaks the fresh-GEMM contract; it computes jobs
/// with [`NaiveGemm`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum Script {
    /// Every job reported with its own output.
    Honest,
    /// The last job is never reported.
    OmitLast,
    /// The first job is written and reported, then the call fails before
    /// writing the others.
    ErrorAfterFirst,
    /// The first job is written and reported, then the call panics before
    /// writing the others.
    PanicAfterFirst,
    /// The first job is reported with a leaked slice of the right length.
    Foreign,
    /// The second job's output is reported under the first job's index.
    Duplicate,
    /// The first job is reported with its output minus one element.
    Short,
}

struct ScriptedGemm(Script);

impl Rank2Gemm<f64> for ScriptedGemm {
    fn matmul_rank2(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [f64],
        lhs: &[f64],
        rhs: &[f64],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<(), OperationError> {
        NaiveGemm.matmul_rank2(engine, dst, lhs, rhs, rows, contracted, cols, alpha, beta)
    }

    fn matmul_rank2_batch_fresh<'a>(
        &mut self,
        engine: &mut dyn Engine,
        segments: &mut [(usize, &'a mut [MaybeUninit<f64>])],
        lhs: &[f64],
        rhs: &[f64],
        jobs: &[Rank2GemmBatchJob],
        _ops: [MatrixOp; 2],
        alpha: f64,
        written: &mut dyn FnMut(usize, &'a mut [f64]),
    ) -> Result<(), OperationError> {
        let mut outputs = Vec::new();
        for (job, piece) in jobs.iter().zip(job_pieces(segments, jobs)?) {
            let output = write_zeros(piece);
            NaiveGemm.matmul_rank2(
                engine,
                output,
                &lhs[job.lhs_offset..],
                &rhs[job.rhs_offset..],
                job.rows,
                job.contracted,
                job.cols,
                alpha,
                0.0,
            )?;
            outputs.push(output);
            // Fail with the later jobs' pieces never written.
            if matches!(self.0, Script::ErrorAfterFirst | Script::PanicAfterFirst) {
                written(0, outputs.pop().unwrap());
                if self.0 == Script::PanicAfterFirst {
                    panic!("injected fresh GEMM panic");
                }
                return Err(OperationError::StridedKernel {
                    message: "injected fresh GEMM failure".into(),
                });
            }
        }
        let mut outputs = outputs.into_iter().enumerate();
        match self.0 {
            Script::Honest => outputs.for_each(|(index, output)| written(index, output)),
            Script::OmitLast => outputs
                .take(jobs.len() - 1)
                .for_each(|(index, output)| written(index, output)),
            Script::ErrorAfterFirst | Script::PanicAfterFirst => {
                unreachable!("the job loop returned")
            }
            Script::Foreign => {
                let len = outputs.next().unwrap().1.len();
                written(0, Box::leak(vec![0.0; len].into_boxed_slice()));
                outputs.for_each(|(index, output)| written(index, output));
            }
            Script::Duplicate => {
                let (_, first) = outputs.next().unwrap();
                let (_, second) = outputs.next().unwrap();
                written(0, first);
                written(0, second);
            }
            Script::Short => {
                let (index, output) = outputs.next().unwrap();
                let len = output.len();
                written(index, &mut output[..len - 1]);
                outputs.for_each(|(index, output)| written(index, output));
            }
        }
        Ok(())
    }
}

/// Two 2x2 direct jobs in one call: the first job's short report leaves
/// one element uncovered; scalar jobs would make that report empty.
fn two_matrix_plan() -> (Arc<BlockStructure>, FusionBlockContractPlan) {
    let structure =
        Arc::new(BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap());
    let group = |offset| {
        let mut group = scalar_group(0, true, 1.0);
        group.rows = 2;
        group.cols = 2;
        group.direct_offset = Some(offset);
        group.block_indices = vec![offset / 4];
        group.subblocks[0].block = FusionStridedBlockLayout {
            shape: vec![2, 2],
            strides: vec![1, 2],
            offset: offset as isize,
        };
        group.subblocks[0].matrix_strides = vec![1, 2];
        group
    };
    let groups = [0, 4]
        .map(|offset| {
            FusionBlockContractGroupPlan::new(group(offset), group(offset), group(offset)).unwrap()
        })
        .to_vec();
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        groups,
    )
    .unwrap();
    (structure, plan)
}

fn run_owned<G: Rank2Gemm<f64>>(
    plan: &FusionBlockContractPlan,
    structure: &Arc<BlockStructure>,
    gemm: &mut G,
    workspace: &mut FusionBlockContractWorkspace<f64>,
    data: &[f64],
    alpha: f64,
) -> Result<Vec<f64>, OperationError> {
    plan.execute_owned(
        &mut symtn_dense::CpuEngine::serial(),
        &mut ObservedKernels::recording(),
        gemm,
        workspace,
        structure,
        structure,
        data,
        structure,
        data,
        alpha,
    )
}

#[test]
fn owned_replay_writes_every_element_from_uninitialized_storage() {
    // What: direct, packed-operand and scatter-destination groups produce
    // `alpha * lhs * rhs` on active blocks and `+0.0` on inactive ones,
    // through the default fresh GEMM.
    for irregular_first in [false, true] {
        let (structure, plan) = four_block_plan(irregular_first);
        let output = plan
            .execute_owned(
                &mut symtn_dense::CpuEngine::serial(),
                &mut ObservedKernels::recording(),
                &mut NaiveGemm,
                &mut FusionBlockContractWorkspace::default(),
                &structure,
                &structure,
                &FOUR_BLOCK_LHS,
                &structure,
                &FOUR_BLOCK_RHS,
                2.0,
            )
            .unwrap();
        assert_eq!(output[..2], [42.0, 110.0], "{irregular_first}");
        assert!(output[2..].iter().all(|v| v.to_bits() == 0));
    }
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![
            FusionBlockContractGroupPlan::new(
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
                scalar_group(0, false, -1.0),
            )
            .unwrap(),
            FusionBlockContractGroupPlan::new(
                scalar_group(1, true, 1.0),
                scalar_group(1, true, 1.0),
                scalar_group(1, true, 1.0),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let output = plan
        .execute_owned(
            &mut symtn_dense::CpuEngine::serial(),
            &mut ObservedKernels::recording(),
            &mut NaiveGemm,
            &mut FusionBlockContractWorkspace::default(),
            &structure,
            &structure,
            &[3.0, 5.0],
            &structure,
            &[7.0, 11.0],
            1.0,
        )
        .unwrap();
    assert_eq!(output, [-21.0, 55.0]);
}

#[test]
fn zero_alpha_owned_replay_is_zero_with_nonfinite_operands() {
    // What: at `alpha = 0` the GEMM outputs are assigned `+0.0` without
    // forming `0 * Inf`.
    let (structure, plan) = four_block_plan(false);
    let output = run_owned(
        &plan,
        &structure,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &[f64::INFINITY, f64::NAN, 1.0, 1.0],
        0.0,
    )
    .unwrap();
    assert!(output.iter().all(|v| v.to_bits() == 0), "{output:?}");
}

#[test]
fn zero_alpha_owned_replay_refuses_a_broken_engine() {
    // What: the zero-alpha shortcut runs no GEMM, yet the caller's engine
    // is still admitted before any write.
    let mut engine = symtn_dense::CpuEngine::serial();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.with_linalg_pool(|_, _| -> Result<(), _> { panic!("break the engine") })
    }));
    assert!(panicked.is_err() && engine.is_broken());
    let (structure, plan) = four_block_plan(false);
    let error = plan
        .execute_owned(
            &mut engine,
            &mut ObservedKernels::recording(),
            &mut NaiveGemm,
            &mut FusionBlockContractWorkspace::default(),
            &structure,
            &structure,
            &FOUR_BLOCK_LHS,
            &structure,
            &FOUR_BLOCK_RHS,
            0.0,
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            OperationError::Dense(symtn_dense::DenseError::Backend { .. })
        ),
        "{error:?}"
    );
}

#[test]
fn honest_fresh_reports_publish_the_output() {
    // What: the scripted backend's honest mode is accepted, so each refusal
    // below is caused by its own contract break.
    let (structure, plan) = two_matrix_plan();
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let output = run_owned(
        &plan,
        &structure,
        &mut ScriptedGemm(Script::Honest),
        &mut FusionBlockContractWorkspace::default(),
        &data,
        1.0,
    )
    .unwrap();
    assert_eq!(output, [7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0]);
}

#[test]
fn a_missing_report_is_refused() {
    let (structure, plan) = two_matrix_plan();
    let error = run_owned(
        &plan,
        &structure,
        &mut ScriptedGemm(Script::OmitLast),
        &mut FusionBlockContractWorkspace::default(),
        &[1.0; 8],
        1.0,
    )
    .unwrap_err();
    assert!(matches!(error, OperationError::InvalidArgument { .. }));
}

#[test]
fn an_error_after_partial_completion_is_refused() {
    let (structure, plan) = two_matrix_plan();
    let error = run_owned(
        &plan,
        &structure,
        &mut ScriptedGemm(Script::ErrorAfterFirst),
        &mut FusionBlockContractWorkspace::default(),
        &[1.0; 8],
        1.0,
    )
    .unwrap_err();
    assert!(matches!(error, OperationError::StridedKernel { .. }));
}

#[test]
fn a_panic_after_partial_completion_publishes_nothing() {
    let (structure, plan) = two_matrix_plan();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run_owned(
            &plan,
            &structure,
            &mut ScriptedGemm(Script::PanicAfterFirst),
            &mut FusionBlockContractWorkspace::default(),
            &[1.0; 8],
            1.0,
        );
    }));
    assert!(unwound.is_err());
}

#[test]
fn forged_reports_are_refused() {
    // What: a leaked foreign slice, a second report under one job index
    // and a report of the wrong length each refuse publication.
    let (structure, plan) = two_matrix_plan();
    for script in [Script::Foreign, Script::Duplicate, Script::Short] {
        let error = run_owned(
            &plan,
            &structure,
            &mut ScriptedGemm(script),
            &mut FusionBlockContractWorkspace::default(),
            &[1.0; 8],
            1.0,
        )
        .unwrap_err();
        assert!(
            matches!(error, OperationError::InvalidArgument { .. }),
            "{script:?}: {error:?}"
        );
    }
}

#[test]
fn a_workspace_reused_after_a_refusal_publishes_the_next_replay() {
    // What: report state left by a refused call does not leak into the
    // next call on the same workspace.
    let (structure, plan) = two_matrix_plan();
    let mut workspace = FusionBlockContractWorkspace::default();
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for script in [Script::OmitLast, Script::Duplicate] {
        run_owned(
            &plan,
            &structure,
            &mut ScriptedGemm(script),
            &mut workspace,
            &data,
            1.0,
        )
        .unwrap_err();
    }
    let output = run_owned(
        &plan,
        &structure,
        &mut ScriptedGemm(Script::Honest),
        &mut workspace,
        &data,
        1.0,
    )
    .unwrap();
    assert_eq!(output, [7.0, 10.0, 15.0, 22.0, 67.0, 78.0, 91.0, 106.0]);
}

#[test]
fn padded_plan_keeps_initialized_replay_and_zeroes_padding_when_owned() {
    // What: a scalar block at offset 1 with `required_len = 2` stays a valid
    // plan; initialized replay leaves index 0 untouched and owned replay
    // writes it `+0.0`.
    let padded = Arc::new(
        BlockStructure::from_blocks_with_rank(
            1,
            vec![BlockSpec::new(vec![1], vec![1], 1).unwrap()],
        )
        .unwrap(),
    );
    let source = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
    let mut dst = scalar_group(0, true, 1.0);
    dst.direct_offset = Some(1);
    dst.subblocks[0].block.offset = 1;
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&padded),
        Arc::clone(&source),
        Arc::clone(&source),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
            dst,
        )
        .unwrap()],
    )
    .unwrap();
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut initialized = vec![f64::NAN; 2];
    plan.execute_raw(
        engine,
        &mut ObservedKernels::recording(),
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &padded,
        &mut initialized,
        &source,
        &[3.0],
        &source,
        &[7.0],
        1.0,
        0.0,
    )
    .unwrap();
    assert!(initialized[0].is_nan());
    assert_eq!(initialized[1], 21.0);
    let owned: Vec<f64> = plan
        .execute_owned(
            engine,
            &mut ObservedKernels::recording(),
            &mut NaiveGemm,
            &mut FusionBlockContractWorkspace::default(),
            &padded,
            &source,
            &[3.0],
            &source,
            &[7.0],
            1.0,
        )
        .unwrap();
    assert_eq!(owned[0].to_bits(), 0);
    assert_eq!(owned[1], 21.0);
}
