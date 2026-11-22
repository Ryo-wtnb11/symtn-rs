use super::*;
use crate::kernel_adapter::observed::ObservedKernels;
use std::ops::{Add, Mul};

use num_complex::Complex64;
use num_traits::Zero;
use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, ZNFusionRule, ZNIrrep};

fn direct_group(rows: usize, cols: usize, offset: Option<usize>) -> FusionBlockMatrixGroup {
    FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows,
        cols,
        needs_clear: false,
        direct_offset: offset,
        block_indices: Vec::new(),
        subblocks: Vec::new(),
    }
}

fn matrix_group(rows: usize, cols: usize, direct: bool) -> FusionBlockMatrixGroup {
    FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows,
        cols,
        needs_clear: false,
        direct_offset: direct.then_some(0),
        block_indices: vec![0],
        subblocks: vec![FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![rows, cols],
                strides: vec![1, rows as isize],
                offset: 0,
            },
            matrix_offset: 0,
            matrix_strides: vec![1, rows as isize],
            coefficient: 1.0,
        }],
    }
}

fn group_plan(
    dims: (usize, usize, usize),
    offsets: (usize, usize, usize),
) -> FusionBlockContractGroupPlan {
    let (rows, contracted, cols) = dims;
    let (lhs_offset, rhs_offset, dst_offset) = offsets;
    FusionBlockContractGroupPlan::new(
        direct_group(rows, contracted, Some(lhs_offset)),
        direct_group(contracted, cols, Some(rhs_offset)),
        direct_group(rows, cols, Some(dst_offset)),
    )
    .unwrap()
}

pub(super) fn scalar_group(
    block_index: usize,
    direct: bool,
    coefficient: f64,
) -> FusionBlockMatrixGroup {
    FusionBlockMatrixGroup {
        coupled: SectorId::new(block_index),
        rows: 1,
        cols: 1,
        needs_clear: false,
        direct_offset: direct.then_some(block_index),
        block_indices: vec![block_index],
        subblocks: vec![FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1],
                strides: vec![1],
                offset: block_index as isize,
            },
            matrix_offset: 0,
            matrix_strides: vec![1],
            coefficient,
        }],
    }
}

#[derive(Default)]
pub(super) struct NaiveGemm;

impl<T> Rank2Gemm<T> for NaiveGemm
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    fn matmul_rank2(
        &mut self,
        _engine: &mut dyn Engine,
        dst: &mut [T],
        lhs: &[T],
        rhs: &[T],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError> {
        for col in 0..cols {
            for row in 0..rows {
                let mut value = T::zero();
                for inner in 0..contracted {
                    value = value + lhs[row + inner * rows] * rhs[inner + col * contracted];
                }
                let index = row + col * rows;
                // BLAS semantics: `beta == 0` never reads the destination.
                dst[index] = if beta.is_zero() {
                    alpha * value
                } else {
                    alpha * value + beta * dst[index]
                };
            }
        }
        Ok(())
    }
}

#[test]
fn coefficient_free_calls_infer_f64() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
    let _ = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        Vec::new(),
    );
    let _ = FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops(
        &structure,
        1,
        &structure,
        1,
        &structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
    );
    let _ = FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_and_alpha(
        &structure,
        1,
        &structure,
        1,
        &structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
        |_| Ok(Default::default()),
    );
}

#[test]
fn mixed_active_and_inactive_blocks_apply_beta_once() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
    let inactive = vec![FusionScaleBlockLayout {
        block: FusionStridedBlockLayout {
            shape: vec![1],
            strides: vec![1],
            offset: 1,
        },
    }];
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        inactive,
        vec![FusionBlockContractGroupPlan::new(
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap()],
    )
    .unwrap();
    let mut dst = vec![7.0, 11.0];
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &structure,
        &mut dst,
        &structure,
        &[3.0, 0.0],
        &structure,
        &[5.0, 0.0],
        2.0,
        -3.0,
    )
    .unwrap();
    assert_eq!(dst, [9.0, -33.0]);
}

#[test]
fn malformed_subblock_bounds_are_rejected_during_compile() {
    let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
    let mut lhs = scalar_group(0, false, 1.0);
    lhs.subblocks[0].matrix_offset = 1;
    let error = FusionBlockContractPlan::<f64>::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            lhs,
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap()],
    )
    .unwrap_err();
    assert!(matches!(error, OperationError::OffsetOverflow { .. }));
}

#[test]
fn direct_and_clear_flags_must_follow_subblock_layouts() {
    let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
    let mut invalid_direct = scalar_group(0, true, 2.0);
    invalid_direct.direct_offset = Some(0);
    let direct_error = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            invalid_direct,
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap()],
    )
    .unwrap_err();
    assert!(matches!(
        direct_error,
        OperationError::StructureMismatch { tensor: "lhs" }
    ));

    let mut invalid_clear = scalar_group(0, false, 1.0);
    invalid_clear.needs_clear = true;
    let clear_error = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            invalid_clear,
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap()],
    )
    .unwrap_err();
    assert!(matches!(
        clear_error,
        OperationError::StructureMismatch { tensor: "lhs" }
    ));
}

#[test]
fn packed_matrix_layouts_must_be_injective() {
    let lhs_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2]]).unwrap());
    let scalar_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
    let mut lhs = matrix_group(2, 1, false);
    lhs.subblocks[0].block.shape = vec![2];
    lhs.subblocks[0].block.strides = vec![1];
    lhs.subblocks[0].matrix_strides[0] = 0;
    lhs.subblocks[0].matrix_strides.truncate(1);
    let mut dst = matrix_group(2, 1, false);
    dst.subblocks[0].block.shape = vec![2];
    dst.subblocks[0].block.strides = vec![1];
    dst.subblocks[0].matrix_strides = vec![1];
    let error = FusionBlockContractPlan::from_parts(
        Arc::clone(&lhs_structure),
        Arc::clone(&lhs_structure),
        Arc::clone(&scalar_structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(lhs, scalar_group(0, false, 1.0), dst).unwrap()],
    )
    .unwrap_err();
    assert!(
        matches!(error, OperationError::StructureMismatch { tensor: "lhs" }),
        "{error:?}"
    );
}

#[test]
fn packed_matrix_geometry_distinguishes_overlap_gap_and_empty_blocks() {
    let subblock = |matrix_offset, extent| FusionSubblockMatrixLayout {
        block: FusionStridedBlockLayout {
            shape: vec![extent],
            strides: vec![1],
            offset: 0,
        },
        matrix_offset,
        matrix_strides: vec![1],
        coefficient: 1.0,
    };
    let group = |rows, subblocks: Vec<FusionSubblockMatrixLayout>| FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows,
        cols: 1,
        needs_clear: true,
        direct_offset: None,
        block_indices: (0..subblocks.len()).collect(),
        subblocks,
    };

    // What: partial rectangle overlap is invalid even when element counts
    // equal the matrix size.
    assert!(
        matrix_layouts_cover_exactly(&group(4, vec![subblock(0, 2), subblock(1, 2)]), 4, 1)
            .is_err()
    );
    // What: a disjoint incomplete grid is valid but does not cover the matrix.
    assert_eq!(
        matrix_layouts_cover_exactly(&group(5, vec![subblock(0, 2), subblock(3, 2)]), 5, 1),
        Ok(false)
    );
    // What: distinct empty tree blocks do not alias any matrix element.
    assert_eq!(
        matrix_layouts_cover_exactly(
            &group(2, vec![subblock(0, 0), subblock(0, 0), subblock(0, 2)]),
            2,
            1
        ),
        Ok(true)
    );
}

#[test]
fn destination_storage_alias_is_rejected_during_compile() {
    use symtn_core::{BlockKey, BlockSpec};

    let block =
        |sector| BlockSpec::with_key(BlockKey::ordinal(sector), vec![1], vec![1], 0).unwrap();
    let aliased =
        Arc::new(BlockStructure::from_blocks_with_rank(1, vec![block(0), block(1)]).unwrap());
    let error = FusionBlockContractPlan::<f64>::from_parts(
        Arc::clone(&aliased),
        Arc::clone(&aliased),
        Arc::clone(&aliased),
        Vec::new(),
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::InvalidArgument {
            message: "fusion block contraction destination layouts overlap"
        }
    ));
}

#[test]
fn inactive_destination_layouts_are_the_exact_active_complement() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
    let active = FusionBlockContractGroupPlan::new(
        scalar_group(0, true, 1.0),
        scalar_group(0, true, 1.0),
        scalar_group(0, true, 1.0),
    )
    .unwrap();
    let missing = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![active.clone()],
    )
    .unwrap_err();
    assert!(matches!(
        missing,
        OperationError::StructureMismatch {
            tensor: "inactive dst blocks"
        }
    ));

    let active_layout = FusionScaleBlockLayout {
        block: scalar_group(0, true, 1.0).subblocks[0].block.clone(),
    };
    let overlap = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        vec![active_layout],
        vec![active],
    )
    .unwrap_err();
    assert!(matches!(
        overlap,
        OperationError::StructureMismatch {
            tensor: "inactive dst blocks"
        }
    ));
}

#[test]
fn duplicate_destination_group_is_rejected() {
    let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
    let group = FusionBlockContractGroupPlan::new(
        scalar_group(0, true, 1.0),
        scalar_group(0, true, 1.0),
        scalar_group(0, true, 1.0),
    )
    .unwrap();
    let error = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![group.clone(), group],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::DuplicateTransformDestination { dst_block: 0 }
    ));
}

#[test]
fn direct_batch_lowers_disjoint_groups() {
    let groups = vec![
        group_plan((2, 3, 4), (0, 0, 0)),
        group_plan((3, 2, 5), (6, 12, 8)),
    ];
    let batch = compile_group_execution(&groups).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].dst_offset, 0);
    assert_eq!(batch[0].rows * batch[0].cols, 8);
    assert_eq!(batch[1].dst_offset, 8);
    assert_eq!(batch[1].contracted, 2);
}

#[test]
fn direct_batch_orders_same_shape_groups_as_strided_runs() {
    let groups = [2usize, 0, 4, 1, 3]
        .into_iter()
        .map(|block| {
            let base = block * 4;
            group_plan((2, 2, 2), (base, base, base))
        })
        .collect::<Vec<_>>();
    let batch = compile_group_execution(&groups).unwrap();
    let offsets = batch
        .iter()
        .map(|job| (job.dst_offset, job.lhs_offset, job.rhs_offset))
        .collect::<Vec<_>>();
    assert_eq!(
        offsets,
        vec![(0, 0, 0), (4, 4, 4), (8, 8, 8), (12, 12, 12), (16, 16, 16)]
    );
}

#[test]
fn direct_batch_plan_bakes_run_partition() {
    // Five same-shape groups fold into one length-5 constant-stride run;
    // the plan stores that partition so the backend routes it
    // without recomputing. Storage matches the shared partition helper.
    let groups = (0usize..5)
        .map(|block| {
            FusionBlockContractGroupPlan::new(
                scalar_group(block, true, 1.0),
                scalar_group(block, true, 1.0),
                scalar_group(block, true, 1.0),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, (0..5).map(|_| vec![1])).unwrap());
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        groups,
    )
    .unwrap();
    assert_eq!(plan.direct_batch_runs(), &[5]);
    assert_eq!(
        plan.direct_batch_runs(),
        strided_batch_runs(plan.direct_batch())
    );
    assert_eq!(
        plan.direct_batch_runs().iter().sum::<usize>(),
        plan.direct_batch().len()
    );
}

#[test]
fn direct_batch_rejects_overlapping_destination_ranges() {
    // First dst covers [0, 8); second starts at 7.
    let groups = vec![
        group_plan((2, 3, 4), (0, 0, 0)),
        group_plan((3, 2, 5), (6, 12, 7)),
    ];
    let error = compile_group_execution(&groups).unwrap_err();
    assert!(matches!(error, OperationError::Internal { .. }));
}

#[test]
fn direct_batch_ignores_empty_destination_ranges() {
    let groups = vec![
        group_plan((2, 1, 1), (0, 0, 0)),
        group_plan((0, 1, 1), (0, 0, 1)),
    ];
    let direct = compile_group_execution(&groups).unwrap();
    assert_eq!(direct.len(), 2);
}

#[test]
fn canonical_direct_plan_rejects_equal_dimension_tree_basis_mismatch() {
    let rule = ZNFusionRule::new(2).unwrap();
    let leg = || {
        SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 1),
                (ZNIrrep::new(1, 2).unwrap(), 1),
            ],
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let shapes = vec![vec![1; 4]; keys.len()];
    let canonical = homspace
        .coupled_subblock_structure(&rule, 2, shapes.clone())
        .unwrap();
    let canonical = Arc::clone(&canonical);
    let mut reordered = keys.iter().cloned().zip(shapes).collect::<Vec<_>>();
    let mut start = 0usize;
    while start < reordered.len() {
        let coupled = reordered[start].0.codomain_tree().coupled();
        let end = start
            + reordered[start..]
                .iter()
                .take_while(|(key, _)| key.codomain_tree().coupled() == coupled)
                .count();
        reordered[start..end].reverse();
        start = end;
    }
    let reordered =
        Arc::new(BlockStructure::coupled_sector_matrix_with_keys(&rule, 2, 4, reordered).unwrap());

    let plan = FusionBlockContractPlan::<f64>::try_from_canonical_coupled_regions_with_ops(
        &canonical,
        2,
        &canonical,
        2,
        &reordered,
        2,
        MatrixOp::Identity,
        MatrixOp::Identity,
    )
    .unwrap();

    // What: aggregate sector dimensions cannot substitute for exact
    // contracted and output fusion-tree bases at the safe plan boundary.
    assert!(plan.is_none());

    let oriented_plan =
        FusionBlockContractPlan::<f64>::try_from_canonical_coupled_regions_with_ops(
            &canonical,
            2,
            &reordered,
            2,
            &canonical,
            2,
            MatrixOp::Adjoint,
            MatrixOp::Identity,
        )
        .unwrap();

    // What: swapping the physical row/column view for an adjoint cannot
    // turn a reordered physical fusion-tree basis into a canonical match.
    assert!(oriented_plan.is_none());
}

#[test]
fn a_group_matrix_not_in_place_is_unsupported() {
    let plan = FusionBlockContractGroupPlan::new(
        direct_group(2, 3, None),
        direct_group(3, 4, Some(0)),
        direct_group(2, 4, Some(0)),
    )
    .unwrap();
    assert!(matches!(
        compile_group_execution(&[plan]).unwrap_err(),
        OperationError::Internal {
            message: NON_COUPLED_OPERAND_MESSAGE
        }
    ));
}

struct RejectingStorageGemm;

impl StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>> for RejectingStorageGemm {
    fn matmul_range_into(
        &mut self,
        _dst: &mut Vec<f64>,
        _dst_offset: usize,
        _lhs: &Vec<f64>,
        _lhs_offset: usize,
        _rhs: &Vec<f64>,
        _rhs_offset: usize,
        _rows: usize,
        _contracted: usize,
        _cols: usize,
    ) -> Result<(), OperationError> {
        panic!("op-bearing storage replay must reject before GEMM")
    }
}

#[test]
fn op_bearing_plan_rejects_unsupported_storage_gemm_before_mutation() {
    let structure = Arc::new(BlockStructure::trivial(&[2, 2]).unwrap());
    let plan = FusionBlockContractPlan::from_parts_with_ops(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            matrix_group(2, 2, true),
            matrix_group(2, 2, true),
            matrix_group(2, 2, true),
        )
        .unwrap()],
        MatrixOp::Adjoint,
        MatrixOp::Identity,
    )
    .unwrap();
    let lhs = vec![0.0; 4];
    let rhs = vec![0.0; 4];
    let mut dst = vec![0.0; 4];

    let error = plan
        .execute_direct_on_storage_prezeroed(&mut RejectingStorageGemm, &mut dst, &lhs, &rhs)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Internal {
            message: "storage GEMM backend does not implement scaled replay"
        }
    ));
    assert_eq!(dst, [0.0; 4]);
}

#[derive(Default)]
struct CountingBatchGemm {
    rank2_calls: usize,
    batch_calls: usize,
    last_batch_len: usize,
}

impl Rank2Gemm<f64> for CountingBatchGemm {
    fn matmul_rank2(
        &mut self,
        _engine: &mut dyn Engine,
        _dst: &mut [f64],
        _lhs: &[f64],
        _rhs: &[f64],
        _rows: usize,
        _contracted: usize,
        _cols: usize,
        _alpha: f64,
        _beta: f64,
    ) -> Result<(), OperationError> {
        self.rank2_calls += 1;
        Ok(())
    }

    fn matmul_rank2_batch(
        &mut self,
        _engine: &mut dyn Engine,
        _dst: &mut [f64],
        _lhs: &[f64],
        _rhs: &[f64],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        _alpha: f64,
        _beta: f64,
    ) -> Result<(), OperationError> {
        debug_assert_eq!(runs.iter().sum::<usize>(), jobs.len());
        self.batch_calls += 1;
        self.last_batch_len = jobs.len();
        Ok(())
    }
}

#[test]
fn direct_replay_uses_one_batched_gemm_call() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let leg = || {
        SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 1),
                (ZNIrrep::new(1, 2).unwrap(), 1),
            ],
            false,
        )
    };
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, [vec![1, 1], vec![1, 1]])
    .unwrap();
    let structure = Arc::clone(&space);
    let plan = FusionBlockContractPlan::<f64>::try_from_canonical_coupled_regions_with_ops(
        &structure,
        1,
        &structure,
        1,
        &structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
    )
    .unwrap()
    .unwrap();
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut gemm = CountingBatchGemm::default();
    let mut workspace = FusionBlockContractWorkspace::default();
    let lhs = vec![0.0; 2];
    let rhs = vec![0.0; 2];
    let mut dst = vec![0.0; 2];

    plan.execute_raw(
        engine,
        &mut kernels,
        &mut gemm,
        &mut workspace,
        &structure,
        &mut dst,
        &structure,
        &lhs,
        &structure,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(gemm.batch_calls, 1);
    assert_eq!(gemm.rank2_calls, 0);
    assert_eq!(gemm.last_batch_len, 2);
}

#[derive(Default)]
struct RecordingComplexStorageGemm {
    calls: Vec<(usize, Complex64)>,
    axpby_calls: Vec<(usize, Complex64, Complex64)>,
}

impl StorageGemm<Complex64, Vec<Complex64>, Vec<Complex64>, Vec<Complex64>>
    for RecordingComplexStorageGemm
{
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
    }

    fn matmul_range_into(
        &mut self,
        dst: &mut Vec<Complex64>,
        dst_offset: usize,
        lhs: &Vec<Complex64>,
        lhs_offset: usize,
        rhs: &Vec<Complex64>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        self.matmul_range_with_ops_scaled_into(
            dst,
            dst_offset,
            lhs,
            lhs_offset,
            rhs,
            rhs_offset,
            rows,
            contracted,
            cols,
            MatrixOp::Identity,
            MatrixOp::Identity,
            Complex64::new(1.0, 0.0),
        )
    }

    fn matmul_range_with_ops_scaled_into(
        &mut self,
        dst: &mut Vec<Complex64>,
        dst_offset: usize,
        lhs: &Vec<Complex64>,
        lhs_offset: usize,
        rhs: &Vec<Complex64>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: Complex64,
    ) -> Result<(), OperationError> {
        assert_eq!(lhs_op, MatrixOp::Identity);
        assert_eq!(rhs_op, MatrixOp::Identity);
        self.calls.push((dst_offset, alpha));
        for col in 0..cols {
            for row in 0..rows {
                dst[dst_offset + row + rows * col] = alpha
                    * (0..contracted)
                        .map(|inner| {
                            lhs[lhs_offset + row + rows * inner]
                                * rhs[rhs_offset + inner + contracted * col]
                        })
                        .sum::<Complex64>();
            }
        }
        Ok(())
    }

    fn matmul_range_axpby_with_ops_into(
        &mut self,
        dst: &mut Vec<Complex64>,
        dst_offset: usize,
        lhs: &Vec<Complex64>,
        lhs_offset: usize,
        rhs: &Vec<Complex64>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: Complex64,
        beta: Complex64,
    ) -> Result<(), OperationError> {
        assert_eq!(lhs_op, MatrixOp::Identity);
        assert_eq!(rhs_op, MatrixOp::Identity);
        self.axpby_calls.push((dst_offset, alpha, beta));
        for col in 0..cols {
            for row in 0..rows {
                let product = (0..contracted)
                    .map(|inner| {
                        lhs[lhs_offset + row + rows * inner]
                            * rhs[rhs_offset + inner + contracted * col]
                    })
                    .sum::<Complex64>();
                let slot = &mut dst[dst_offset + row + rows * col];
                // BLAS: beta = 0 does not read the destination.
                *slot = if beta == Complex64::ZERO {
                    alpha * product
                } else {
                    alpha * product + beta * *slot
                };
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct UnitOnlyComplexStorageGemm {
    unit_calls: usize,
}

impl StorageGemm<Complex64, Vec<Complex64>, Vec<Complex64>, Vec<Complex64>>
    for UnitOnlyComplexStorageGemm
{
    fn matmul_range_into(
        &mut self,
        _dst: &mut Vec<Complex64>,
        _dst_offset: usize,
        _lhs: &Vec<Complex64>,
        _lhs_offset: usize,
        _rhs: &Vec<Complex64>,
        _rhs_offset: usize,
        _rows: usize,
        _contracted: usize,
        _cols: usize,
    ) -> Result<(), OperationError> {
        self.unit_calls += 1;
        Ok(())
    }
}

#[test]
fn scaled_storage_jobs_keep_canonical_coefficients_aligned_and_convert_payload() {
    let leg = || {
        SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 2),
                (ZNIrrep::new(1, 2).unwrap(), 1),
            ],
            false,
        )
    };
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, [vec![2, 2], vec![1, 1]])
    .unwrap();
    let structure = Arc::clone(&space);
    let plan = FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_and_alpha(
        &structure,
        1,
        &structure,
        1,
        &structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
        |coupled| {
            Ok(if coupled == SectorId::new(0) {
                1.0
            } else {
                -1.0
            })
        },
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.direct_batch_alpha, [1.0, -1.0]);
    assert!(plan.require_identity_direct_replay().is_err());
    plan.require_identity_signed_direct_replay().unwrap();
    let non_sign = FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_and_alpha(
        &structure,
        1,
        &structure,
        1,
        &structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
        |coupled| {
            Ok(if coupled == SectorId::new(0) {
                1.0
            } else {
                2.0
            })
        },
    )
    .unwrap()
    .unwrap();
    assert!(matches!(
        non_sign.require_identity_signed_direct_replay(),
        Err(OperationError::Internal {
            message: "signed stacked replay requires exact unit-magnitude coefficients"
        })
    ));

    let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0]
        .into_iter()
        .map(|value| Complex64::new(value, 0.0))
        .collect();
    let rhs = vec![1.0, 0.0, 0.0, 1.0, 2.0]
        .into_iter()
        .map(|value| Complex64::new(value, 0.0))
        .collect();
    let mut dst = vec![Complex64::ZERO; 5];
    let mut gemm = RecordingComplexStorageGemm::default();
    plan.execute_direct_on_storage_prezeroed(&mut gemm, &mut dst, &lhs, &rhs)
        .unwrap();

    assert_eq!(
        gemm.calls,
        [
            (0, Complex64::new(1.0, 0.0)),
            (4, Complex64::new(-1.0, 0.0))
        ]
    );
    assert_eq!(
        dst,
        [1.0, 2.0, 3.0, 4.0, -10.0].map(|value| Complex64::new(value, 0.0))
    );

    let unit_first_space = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 1),
                (ZNIrrep::new(1, 2).unwrap(), 1),
            ],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 1),
                (ZNIrrep::new(1, 2).unwrap(), 1),
            ],
            false,
        )]),
    )
    .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, [vec![1, 1], vec![1, 1]])
    .unwrap();
    let unit_first_structure = Arc::clone(&unit_first_space);
    let unit_first_plan =
        FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_and_alpha(
            &unit_first_structure,
            1,
            &unit_first_structure,
            1,
            &unit_first_structure,
            1,
            MatrixOp::Identity,
            MatrixOp::Identity,
            |coupled| {
                Ok(if coupled == SectorId::new(0) {
                    1.0
                } else {
                    -1.0
                })
            },
        )
        .unwrap()
        .unwrap();
    let mut rejected = vec![Complex64::ZERO; 2];
    let source = vec![Complex64::new(1.0, 0.0); 2];
    let mut unit_only = UnitOnlyComplexStorageGemm::default();
    let error = unit_first_plan
        .execute_direct_on_storage_prezeroed(&mut unit_only, &mut rejected, &source, &source)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Internal {
            message: "storage GEMM backend does not implement scaled replay"
        }
    ));
    assert_eq!(unit_only.unit_calls, 0);
    assert_eq!(rejected, vec![Complex64::ZERO; 2]);
}

/// Fails on the `fail_at`-th (zero-based) GEMM job, after the earlier
/// jobs have written their destination blocks.
struct FailingAtJob {
    calls: usize,
    fail_at: usize,
}

impl Rank2Gemm<f64> for FailingAtJob {
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
        let call = self.calls;
        self.calls += 1;
        if call == self.fail_at {
            return Err(OperationError::StridedKernel {
                message: "injected GEMM failure".into(),
            });
        }
        NaiveGemm.matmul_rank2(engine, dst, lhs, rhs, rows, contracted, cols, alpha, beta)
    }
}

/// Four scalar blocks: blocks 0 and 1 are active and blocks 2 and 3 are the
/// inactive complement.
pub(super) fn four_block_plan() -> (Arc<BlockStructure>, FusionBlockContractPlan) {
    let structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1]]).unwrap(),
    );
    let inactive = [2usize, 3]
        .into_iter()
        .map(|block| FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1],
                strides: vec![1],
                offset: block as isize,
            },
        })
        .collect();
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        inactive,
        vec![
            FusionBlockContractGroupPlan::new(
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
                scalar_group(0, true, 1.0),
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
    (structure, plan)
}

pub(super) const FOUR_BLOCK_LHS: [f64; 4] = [3.0, 5.0, 0.0, 0.0];
pub(super) const FOUR_BLOCK_RHS: [f64; 4] = [7.0, 11.0, 0.0, 0.0];

fn touched_inactive(offsets: &[isize]) -> Vec<isize> {
    offsets.iter().copied().filter(|&o| o >= 2).collect()
}

#[test]
fn zeroed_replay_never_touches_inactive_blocks() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: on an all-zero destination `Zeroed` runs no scale/copy kernel
    // over the inactive blocks; the active blocks are written by their
    // job with `beta = 0` and the inactive ones keep their zeros.
    let (structure, plan) = four_block_plan();
    let mut kernels = ObservedKernels::recording();
    let mut dst = vec![0.0; 4];
    plan.execute_raw_zeroed(
        engine,
        &mut kernels,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &structure,
        &mut dst,
        &structure,
        &FOUR_BLOCK_LHS,
        &structure,
        &FOUR_BLOCK_RHS,
        2.0,
    )
    .unwrap();
    assert_eq!(dst, [42.0, 110.0, 0.0, 0.0]);
    assert!(kernels.scale_offsets().is_empty());
    assert!(
        touched_inactive(&kernels.copy_offsets()).is_empty(),
        "{:?}",
        kernels.copy_offsets()
    );
    assert!(dst[2..].iter().all(|v| v.to_bits() == 0));
}

#[test]
fn axpby_zero_assigns_inactive_blocks_without_reading_them() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: `Axpby(0)` is a strong zero: NaN, Inf and -0.0 in the inactive
    // blocks become exactly +0.0 through one copy per block, never a
    // multiply that would keep NaN.
    let (structure, plan) = four_block_plan();
    let mut kernels = ObservedKernels::recording();
    let mut dst = vec![f64::NAN, f64::INFINITY, f64::NAN, -0.0];
    plan.execute_raw(
        engine,
        &mut kernels,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &structure,
        &mut dst,
        &structure,
        &FOUR_BLOCK_LHS,
        &structure,
        &FOUR_BLOCK_RHS,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(dst[..2], [21.0, 55.0]);
    assert_eq!(dst[2].to_bits(), 0);
    assert_eq!(dst[3].to_bits(), 0);
    assert!(kernels.scale_offsets().is_empty());
    assert_eq!(touched_inactive(&kernels.copy_offsets()), [2, 3]);
}

#[test]
fn axpby_one_leaves_inactive_blocks_untouched() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (structure, plan) = four_block_plan();
    let mut kernels = ObservedKernels::recording();
    let mut dst = vec![1.0, 2.0, 3.0, f64::NAN];
    plan.execute_raw(
        engine,
        &mut kernels,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &structure,
        &mut dst,
        &structure,
        &FOUR_BLOCK_LHS,
        &structure,
        &FOUR_BLOCK_RHS,
        1.0,
        1.0,
    )
    .unwrap();
    assert_eq!(dst[..3], [22.0, 57.0, 3.0]);
    assert!(dst[3].is_nan());
    assert!(kernels.scale_offsets().is_empty());
    assert!(kernels.copy_offsets().is_empty());
}

#[test]
fn axpby_general_beta_scales_inactive_blocks_exactly_once() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (structure, plan) = four_block_plan();
    let mut kernels = ObservedKernels::recording();
    let mut dst = vec![1.0, 2.0, 3.0, 4.0];
    plan.execute_raw(
        engine,
        &mut kernels,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &structure,
        &mut dst,
        &structure,
        &FOUR_BLOCK_LHS,
        &structure,
        &FOUR_BLOCK_RHS,
        1.0,
        0.5,
    )
    .unwrap();
    assert_eq!(dst, [21.5, 56.0, 1.5, 2.0]);
    assert_eq!(kernels.scale_offsets(), [2, 3]);
    assert!(kernels.copy_offsets().is_empty());
}

#[test]
fn gemm_failure_after_the_first_job_surfaces_without_panicking() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an error on a later job returns `Err`; the destination is a
    // valid buffer throughout (born zero, first job written), so nothing
    // uninitialised can be observed and the owner drops it.
    let (structure, plan) = four_block_plan();
    let mut kernels = ObservedKernels::recording();
    let mut gemm = FailingAtJob {
        calls: 0,
        fail_at: 1,
    };
    let mut dst = vec![0.0; 4];
    let error = plan
        .execute_raw_zeroed(
            engine,
            &mut kernels,
            &mut gemm,
            &mut FusionBlockContractWorkspace::default(),
            &structure,
            &mut dst,
            &structure,
            &FOUR_BLOCK_LHS,
            &structure,
            &FOUR_BLOCK_RHS,
            1.0,
        )
        .unwrap_err();
    assert!(matches!(error, OperationError::StridedKernel { .. }));
    assert_eq!(gemm.calls, 2);
    assert_eq!(dst, [21.0, 0.0, 0.0, 0.0]);
    assert!(kernels.scale_offsets().is_empty());
    assert!(kernels.copy_offsets().is_empty());
}

#[test]
fn canonical_plan_zeroed_replay_skips_the_inactive_coupled_range() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a canonical coupled-layout plan whose contracted leg carries
    // only the even sector leaves the odd destination range untouched
    // under `Zeroed` and assigns it under `Axpby(0)`.
    let both = || {
        SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 2),
                (ZNIrrep::new(1, 2).unwrap(), 3),
            ],
            false,
        )
    };
    let even = || SectorLeg::new([(ZNIrrep::new(0, 2).unwrap(), 2)], false);
    // `shapes` holds one `[rows, cols]` per coupled tree pair.
    let space = |codomain: SectorLeg, domain: SectorLeg, shapes: Vec<Vec<usize>>| {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([codomain]),
            FusionProductSpace::new([domain]),
        )
        .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, shapes)
        .unwrap()
    };
    let dst = space(both(), both(), vec![vec![2, 2], vec![3, 3]]);
    let lhs = space(both(), even(), vec![vec![2, 2]]);
    let rhs = space(even(), both(), vec![vec![2, 2]]);
    let dst_structure = Arc::clone(&dst);
    let lhs_structure = Arc::clone(&lhs);
    let rhs_structure = Arc::clone(&rhs);
    let plan = FusionBlockContractPlan::<f64>::try_from_canonical_coupled_regions_with_ops(
        &dst_structure,
        1,
        &lhs_structure,
        1,
        &rhs_structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.inactive_dst_scale_blocks.len(), 1);
    assert_eq!(plan.direct_batch().len(), 1);
    let required = dst_structure.required_len().unwrap();
    assert_eq!(required, 2 * 2 + 3 * 3);
    // What: the accessor names exactly the odd 3x3 block no job writes, as
    // one contiguous range — the set a retained device destination must
    // zero.
    assert_eq!(
        plan.inactive_destination_regions(),
        [FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: vec![9],
                strides: vec![1],
                offset: 4,
            },
        }]
    );
    let lhs_data = vec![1.0; lhs_structure.required_len().unwrap()];
    let rhs_data = vec![2.0; rhs_structure.required_len().unwrap()];

    let mut kernels = ObservedKernels::recording();
    let mut zeroed = vec![0.0; required];
    plan.execute_raw_zeroed(
        engine,
        &mut kernels,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &dst_structure,
        &mut zeroed,
        &lhs_structure,
        &lhs_data,
        &rhs_structure,
        &rhs_data,
        1.0,
    )
    .unwrap();
    assert!(kernels.scale_offsets().is_empty());
    assert!(kernels.copy_offsets().is_empty());

    let mut kernels = ObservedKernels::recording();
    let mut assigned = vec![f64::NAN; required];
    plan.execute_raw(
        engine,
        &mut kernels,
        &mut NaiveGemm,
        &mut FusionBlockContractWorkspace::default(),
        &dst_structure,
        &mut assigned,
        &lhs_structure,
        &lhs_data,
        &rhs_structure,
        &rhs_data,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(kernels.copy_offsets().len(), 1);
    assert!(kernels.scale_offsets().is_empty());
    assert_eq!(zeroed, assigned);
    assert!(zeroed.iter().all(|v| v.to_bits() == 0 || *v == 4.0));
    assert_eq!(zeroed.iter().filter(|v| **v == 4.0).count(), 4);
}

/// Canonical Z2 plan with job coefficients `+1` (even block, offset 0) and
/// `-1` (odd block, offset 4), and operands whose products are
/// `[1, 2, 3, 4]` and `[10]`.
fn signed_z2_storage_plan() -> (FusionBlockContractPlan<f64>, Vec<Complex64>, Vec<Complex64>) {
    let leg = || {
        SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), 2),
                (ZNIrrep::new(1, 2).unwrap(), 1),
            ],
            false,
        )
    };
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, [vec![2, 2], vec![1, 1]])
    .unwrap();
    let structure = Arc::clone(&space);
    let plan = FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_and_alpha(
        &structure,
        1,
        &structure,
        1,
        &structure,
        1,
        MatrixOp::Identity,
        MatrixOp::Identity,
        |coupled| {
            Ok(if coupled == SectorId::new(0) {
                1.0
            } else {
                -1.0
            })
        },
    )
    .unwrap()
    .unwrap();
    let complex = |values: [f64; 5]| values.map(|value| Complex64::new(value, 0.0)).to_vec();
    (
        plan,
        complex([1.0, 2.0, 3.0, 4.0, 5.0]),
        complex([1.0, 0.0, 0.0, 1.0, 2.0]),
    )
}

#[test]
fn storage_axpby_jobs_carry_alpha_times_job_coefficient_and_beta() {
    // What: each job's GEMM gets `alpha * job_alpha` and the caller's beta in
    // its epilogue, for beta = 0 (a NaN destination is not read), 1 and a
    // general complex value; the oracle is the hand product `[1, 2, 3, 4]`
    // and `-[10]`.
    let (plan, lhs, rhs) = signed_z2_storage_plan();
    let products = [1.0, 2.0, 3.0, 4.0, -10.0].map(|value| Complex64::new(value, 0.0));
    let alpha = Complex64::new(0.5, -1.0);
    let prior = [0.25, -1.5, 2.0, 0.75, 3.0].map(|value| Complex64::new(value, 0.5));
    for beta in [
        Complex64::ZERO,
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.5, 0.25),
    ] {
        let mut dst = if beta == Complex64::ZERO {
            vec![Complex64::new(f64::NAN, f64::NAN); 5]
        } else {
            prior.to_vec()
        };
        let mut gemm = RecordingComplexStorageGemm::default();
        plan.execute_direct_on_storage_axpby(&mut gemm, &mut dst, &lhs, &rhs, alpha, beta)
            .unwrap();
        assert!(gemm.calls.is_empty(), "the unscaled entry must not run");
        assert_eq!(
            gemm.axpby_calls,
            [(0, alpha, beta), (4, -alpha, beta)],
            "beta = {beta}"
        );
        for (index, (&got, &product)) in dst.iter().zip(&products).enumerate() {
            let want = if beta == Complex64::ZERO {
                alpha * product
            } else {
                alpha * product + beta * prior[index]
            };
            assert!(
                (got - want).norm() <= 1e-14,
                "beta = {beta}, {index}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn a_storage_gemm_without_the_axpby_entry_rejects_before_writing() {
    // What: the trait's default beta-accumulating entry is a typed capability
    // error, raised by the first job, so the destination is untouched.
    let (plan, lhs, rhs) = signed_z2_storage_plan();
    let mut dst = vec![Complex64::new(7.0, 0.0); 5];
    let mut gemm = UnitOnlyComplexStorageGemm::default();
    let error = plan
        .execute_direct_on_storage_axpby(
            &mut gemm,
            &mut dst,
            &lhs,
            &rhs,
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
        )
        .unwrap_err();
    assert!(
        matches!(error, OperationError::Internal { .. }),
        "{error:?}"
    );
    assert_eq!(gemm.unit_calls, 0);
    assert_eq!(dst, vec![Complex64::new(7.0, 0.0); 5]);
}
