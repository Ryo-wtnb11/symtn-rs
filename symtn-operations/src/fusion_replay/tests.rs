use super::*;
use crate::kernel_adapter::observed::ObservedKernels;
use std::ops::{Add, Mul};

use num_complex::Complex64;
use num_traits::Zero;
use symtn_core::{
    FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeFusionSymbols, SectorCodec, SectorLeg,
    TableRule, TableSector, Z2FusionRule, Z2Irrep,
};

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

fn scalar_group(block_index: usize, direct: bool, coefficient: f64) -> FusionBlockMatrixGroup {
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
struct NaiveGemm;

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
fn fibonacci_complex_r_symbol_survives_fusion_block_contraction() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = TableRule::fibonacci();
    let tau = provider.encode_sector(&TableSector::new(2)).unwrap();
    let vacuum = provider.encode_sector(&TableSector::new(1)).unwrap();
    let r_tau_tau_vacuum = provider.r_symbol_scalar(tau, tau, vacuum);
    assert!(
        r_tau_tau_vacuum.im.abs() > 0.5,
        "the oracle must exercise a genuinely non-real R-symbol"
    );

    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
    let group = |direct: bool, coefficient: Complex64| FusionBlockMatrixGroup {
        coupled: vacuum,
        rows: 1,
        cols: 1,
        needs_clear: false,
        direct_offset: direct.then_some(0),
        block_indices: vec![0],
        subblocks: vec![FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1],
                strides: vec![1],
                offset: 0,
            },
            matrix_offset: 0,
            matrix_strides: vec![1],
            coefficient,
        }],
    };
    let contract_group = FusionBlockContractGroupPlan::new(
        group(false, r_tau_tau_vacuum),
        group(true, Complex64::new(1.0, 0.0)),
        group(true, Complex64::new(1.0, 0.0)),
    )
    .unwrap();
    let plan = FusionBlockContractPlan::from_parts_generic(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![contract_group],
    )
    .unwrap();

    let lhs = [Complex64::new(2.0, 0.0)];
    let rhs = [Complex64::new(3.0, 0.0)];
    let mut dst = [Complex64::new(0.0, 0.0)];
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut gemm = NaiveGemm;
    let mut workspace = FusionBlockContractWorkspace::<Complex64>::default();
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
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    // The non-dyadic R-symbol makes the product order-dependent in its
    // last bit, so it is compared to the hand value under the tolerance
    // rule; three factors (R, lhs, rhs) reach the entry.
    crate::test_numerics::numerics::assert_close(
        "R(tau, tau; 1) * 2 * 3",
        dst[0],
        r_tau_tau_vacuum * Complex64::new(6.0, 0.0),
        3,
    );
    assert_ne!(dst[0].im, 0.0, "discarding the R-symbol phase must fail");
}

fn all_classes_apply_alpha_beta_and_coefficients<T>()
where
    T: DenseBlockScalar
        + RecouplingCoefficientAction<f64>
        + From<f64>
        + Add<Output = T>
        + Mul<Output = T>
        + std::fmt::Debug
        + PartialEq,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, (0..8).map(|_| vec![1])).unwrap());
    let mut groups = Vec::new();
    for class in 0usize..8 {
        let pack_lhs = class & usize::from(FusionGroupExecutionClass::PACK_LHS) != 0;
        let pack_rhs = class & usize::from(FusionGroupExecutionClass::PACK_RHS) != 0;
        let scatter_dst = class & usize::from(FusionGroupExecutionClass::SCATTER_DST) != 0;
        groups.push(
            FusionBlockContractGroupPlan::new(
                scalar_group(class, !pack_lhs, if pack_lhs { 2.0 } else { 1.0 }),
                scalar_group(class, !pack_rhs, if pack_rhs { 3.0 } else { 1.0 }),
                scalar_group(class, !scatter_dst, if scatter_dst { 5.0 } else { 1.0 }),
            )
            .unwrap(),
        );
    }
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        groups,
    )
    .unwrap();
    assert!(!plan.is_fully_direct());
    assert_eq!(plan.direct_batch.len(), 1);
    assert_eq!(plan.irregular.len(), 7);
    assert_eq!(plan.max_irregular_scratch_len, 3);

    let lhs = (0..8)
        .map(|index| T::from(index as f64 + 1.0))
        .collect::<Vec<_>>();
    let rhs = (0..8)
        .map(|index| T::from(index as f64 + 2.0))
        .collect::<Vec<_>>();
    let initial = (0..8)
        .map(|index| T::from(index as f64 + 7.0))
        .collect::<Vec<_>>();
    let alpha = T::from(1.5);
    let beta = T::from(-0.25);
    let mut expected = Vec::new();
    for class in 0usize..8 {
        let lhs_coefficient = if class & usize::from(FusionGroupExecutionClass::PACK_LHS) != 0 {
            T::from(2.0)
        } else {
            T::from(1.0)
        };
        let rhs_coefficient = if class & usize::from(FusionGroupExecutionClass::PACK_RHS) != 0 {
            T::from(3.0)
        } else {
            T::from(1.0)
        };
        let dst_coefficient = if class & usize::from(FusionGroupExecutionClass::SCATTER_DST) != 0 {
            T::from(5.0)
        } else {
            T::from(1.0)
        };
        expected.push(
            dst_coefficient * alpha * lhs_coefficient * lhs[class] * rhs_coefficient * rhs[class]
                + beta * initial[class],
        );
    }

    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut gemm = NaiveGemm;
    let mut workspace = FusionBlockContractWorkspace::<T>::default();
    let mut dst = initial.clone();
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
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(dst, expected);
    assert_eq!(workspace.scratch.len(), 3);

    workspace.scratch.fill(T::from(99.0));
    dst.clone_from(&initial);
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
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(dst, expected);

    // A zero alpha (either sign) leaves exactly TensorKit's `scale(C,
    // beta)` in every class, over Inf/NaN operands, a NaN destination and
    // a NaN-filled scratch: no pack, GEMM or product is formed.
    let nan = T::from(f64::NAN);
    let poisoned = |values: &[T], bad: T| {
        let mut values = values.to_vec();
        values[0] = bad;
        values[5] = bad;
        values
    };
    let (bad_lhs, bad_rhs) = (poisoned(&lhs, T::from(f64::INFINITY)), poisoned(&rhs, nan));
    let dirty = poisoned(&initial, nan);
    let same = |got: &[T], want: &[T]| {
        assert_eq!(got.len(), want.len());
        for (&got, &want) in got.iter().zip(want) {
            #[allow(clippy::eq_op)]
            let both_nan = got != got && want != want;
            assert!(got == want || both_nan, "{got:?} != {want:?}");
        }
    };
    for zero in [0.0, -0.0] {
        for beta in [0.0, 1.0, 2.0] {
            let want: Vec<T> = dirty
                .iter()
                .map(|&value| match beta {
                    0.0 => T::zero(),
                    1.0 => value,
                    _ => T::from(beta) * value,
                })
                .collect();
            workspace.scratch.fill(nan);
            dst.clone_from(&dirty);
            plan.execute_raw(
                engine,
                &mut kernels,
                &mut gemm,
                &mut workspace,
                &structure,
                &mut dst,
                &structure,
                &bad_lhs,
                &structure,
                &bad_rhs,
                T::from(zero),
                T::from(beta),
            )
            .unwrap();
            same(&dst, &want);
        }
    }

    // The dirty workspace left by the zero-alpha runs replays a non-zero
    // alpha exactly as a fresh one does.
    dst.clone_from(&initial);
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
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(dst, expected);
}

#[test]
fn all_eight_group_classes_replay_f64() {
    all_classes_apply_alpha_beta_and_coefficients::<f64>();
}

#[test]
fn all_eight_group_classes_replay_c64() {
    all_classes_apply_alpha_beta_and_coefficients::<Complex64>();
}

#[test]
fn sparse_pack_clears_stale_group_scratch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let lhs_structure = Arc::new(BlockStructure::trivial(&[1, 1]).unwrap());
    let rhs_structure = Arc::new(BlockStructure::trivial(&[2, 1]).unwrap());
    let dst_structure = Arc::new(BlockStructure::trivial(&[2, 1]).unwrap());
    let lhs = FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows: 2,
        cols: 2,
        needs_clear: true,
        direct_offset: None,
        block_indices: vec![0],
        subblocks: vec![FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1, 1],
                strides: vec![1, 1],
                offset: 0,
            },
            matrix_offset: 0,
            matrix_strides: vec![1, 2],
            coefficient: 1.0,
        }],
    };
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&dst_structure),
        Arc::clone(&lhs_structure),
        Arc::clone(&rhs_structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            lhs,
            matrix_group(2, 1, true),
            matrix_group(2, 1, true),
        )
        .unwrap()],
    )
    .unwrap();
    let mut workspace = FusionBlockContractWorkspace::<f64>::default();
    workspace.scratch.resize_filled(4, 41.0);
    let mut dst = vec![7.0, 11.0];
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut NaiveGemm,
        &mut workspace,
        &dst_structure,
        &mut dst,
        &lhs_structure,
        &[3.0],
        &rhs_structure,
        &[5.0, 13.0],
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(dst, [15.0, 0.0]);
    assert_eq!(workspace.scratch.len(), 4);
}

struct ComplexOpGemm;

impl Rank2Gemm<Complex64> for ComplexOpGemm {
    fn matmul_rank2(
        &mut self,
        _engine: &mut dyn Engine,
        dst: &mut [Complex64],
        lhs: &[Complex64],
        rhs: &[Complex64],
        _rows: usize,
        _contracted: usize,
        _cols: usize,
        alpha: Complex64,
        beta: Complex64,
    ) -> Result<(), OperationError> {
        dst[0] = alpha * lhs[0] * rhs[0] + beta * dst[0];
        Ok(())
    }

    fn matmul_rank2_batch_with_ops(
        &mut self,
        _engine: &mut dyn Engine,
        dst: &mut [Complex64],
        lhs: &[Complex64],
        rhs: &[Complex64],
        jobs: &[Rank2GemmBatchJob],
        _runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: Complex64,
        beta: Complex64,
    ) -> Result<(), OperationError> {
        for job in jobs {
            let lhs_value = match lhs_op {
                MatrixOp::Adjoint => lhs[job.lhs_offset].conj(),
                MatrixOp::Identity | MatrixOp::Transpose => lhs[job.lhs_offset],
            };
            let rhs_value = match rhs_op {
                MatrixOp::Adjoint => rhs[job.rhs_offset].conj(),
                MatrixOp::Identity | MatrixOp::Transpose => rhs[job.rhs_offset],
            };
            let dst_value = &mut dst[job.dst_offset];
            *dst_value = alpha * lhs_value * rhs_value + beta * *dst_value;
        }
        Ok(())
    }
}

#[test]
fn packed_adjoint_is_conjugated_only_by_gemm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let structure = Arc::new(BlockStructure::trivial(&[1, 1]).unwrap());
    let plan = FusionBlockContractPlan::from_parts_with_ops(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            matrix_group(1, 1, false),
            matrix_group(1, 1, true),
            matrix_group(1, 1, true),
        )
        .unwrap()],
        MatrixOp::Adjoint,
        MatrixOp::Identity,
    )
    .unwrap();
    let lhs = [Complex64::new(2.0, 3.0)];
    let rhs = [Complex64::new(5.0, -7.0)];
    let mut dst = [Complex64::new(11.0, 13.0)];
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut ComplexOpGemm,
        &mut FusionBlockContractWorkspace::default(),
        &structure,
        &mut dst,
        &structure,
        &lhs,
        &structure,
        &rhs,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    assert_eq!(dst[0], lhs[0].conj() * rhs[0]);
}

#[test]
fn asymmetric_packed_adjoint_validates_physical_matrix_shape() {
    let dst_structure = Arc::new(BlockStructure::trivial(&[2, 1]).unwrap());
    let lhs_structure = Arc::new(BlockStructure::trivial(&[3, 2]).unwrap());
    let rhs_structure = Arc::new(BlockStructure::trivial(&[3, 1]).unwrap());
    let mut physical_lhs = matrix_group(3, 2, false);
    physical_lhs.rows = 2;
    physical_lhs.cols = 3;
    let plan = FusionBlockContractPlan::from_parts_with_ops(
        dst_structure,
        lhs_structure,
        rhs_structure,
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            physical_lhs,
            matrix_group(3, 1, false),
            matrix_group(2, 1, false),
        )
        .unwrap()],
        MatrixOp::Adjoint,
        MatrixOp::Identity,
    )
    .unwrap();
    assert!(!plan.is_fully_direct());
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
            scalar_group(0, false, 2.0),
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
    assert_eq!(dst, [39.0, -33.0]);
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
    // What: a disjoint incomplete grid is valid but requires scratch clear.
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
fn mixed_plan_rejects_storage_direct_before_mutation() {
    let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![FusionBlockContractGroupPlan::new(
            scalar_group(0, false, 1.0),
            scalar_group(0, true, 1.0),
            scalar_group(0, true, 1.0),
        )
        .unwrap()],
    )
    .unwrap();
    let lhs = vec![2.0];
    let rhs = vec![3.0];
    let mut dst = vec![5.0];
    let error = plan
        .execute_direct_on_storage_prezeroed(&mut RejectingStorageGemm, &mut dst, &lhs, &rhs)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope {
            message: NON_COUPLED_OPERAND_MESSAGE
        }
    ));
    assert_eq!(dst, [5.0]);
}

#[test]
fn direct_batch_lowers_disjoint_groups() {
    let groups = vec![
        group_plan((2, 3, 4), (0, 0, 0)),
        group_plan((3, 2, 5), (6, 12, 8)),
    ];
    let (batch, irregular, _) = compile_group_execution(&groups).unwrap();
    assert!(irregular.is_empty());
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
    let (batch, irregular, _) = compile_group_execution(&groups).unwrap();
    assert!(irregular.is_empty());
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
fn non_direct_plan_has_empty_run_partition() {
    let plan = FusionBlockContractGroupPlan::new(
        matrix_group(2, 3, false),
        matrix_group(3, 4, true),
        matrix_group(2, 4, true),
    )
    .unwrap();
    let dst_structure = Arc::new(BlockStructure::trivial(&[2, 4]).unwrap());
    let lhs_structure = Arc::new(BlockStructure::trivial(&[2, 3]).unwrap());
    let rhs_structure = Arc::new(BlockStructure::trivial(&[3, 4]).unwrap());
    let plan = FusionBlockContractPlan::from_parts(
        dst_structure,
        lhs_structure,
        rhs_structure,
        Vec::new(),
        vec![plan],
    )
    .unwrap();
    assert!(!plan.is_fully_direct());
    assert!(plan.direct_batch_runs().is_empty());
}

#[test]
fn direct_batch_rejects_overlapping_destination_ranges() {
    // First dst covers [0, 8); second starts at 7.
    let groups = vec![
        group_plan((2, 3, 4), (0, 0, 0)),
        group_plan((3, 2, 5), (6, 12, 7)),
    ];
    let error = compile_group_execution(&groups).unwrap_err();
    assert!(matches!(
        error,
        OperationError::UnsupportedTensorContractScope { .. }
    ));
}

#[test]
fn direct_batch_ignores_empty_destination_ranges() {
    let groups = vec![
        group_plan((2, 1, 1), (0, 0, 0)),
        group_plan((0, 1, 1), (0, 0, 1)),
    ];
    let (direct, irregular, _) = compile_group_execution(&groups).unwrap();
    assert_eq!(direct.len(), 2);
    assert!(irregular.is_empty());
}

#[test]
fn canonical_direct_plan_rejects_equal_dimension_tree_basis_mismatch() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
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
fn non_direct_group_compiles_one_irregular_execution() {
    let plan = FusionBlockContractGroupPlan::new(
        direct_group(2, 3, None),
        direct_group(3, 4, Some(0)),
        direct_group(2, 4, Some(0)),
    )
    .unwrap();
    let (direct, irregular, _) = compile_group_execution(&[plan]).unwrap();
    assert!(direct.is_empty());
    assert_eq!(irregular.len(), 1);
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
        OperationError::UnsupportedTensorContractScope {
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
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure(&Z2FusionRule, 1, [vec![1, 1], vec![1, 1]])
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
    let mut workspace = FusionBlockContractWorkspace::<f64>::default();
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
    assert_eq!(workspace.scratch.len(), 0);
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
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)], false);
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure(&Z2FusionRule, 1, [vec![2, 2], vec![1, 1]])
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
        Err(OperationError::UnsupportedTensorContractScope {
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
            [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
            false,
        )]),
    )
    .coupled_subblock_structure(&Z2FusionRule, 1, [vec![1, 1], vec![1, 1]])
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
        OperationError::UnsupportedTensorContractScope {
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

/// Four scalar blocks: blocks 0 and 1 are active (block 0 irregular when
/// `irregular_first`, both direct otherwise) and blocks 2 and 3 are the
/// inactive complement.
fn four_block_plan(irregular_first: bool) -> (Arc<BlockStructure>, FusionBlockContractPlan) {
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
                scalar_group(0, !irregular_first, 1.0),
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

const FOUR_BLOCK_LHS: [f64; 4] = [3.0, 5.0, 0.0, 0.0];
const FOUR_BLOCK_RHS: [f64; 4] = [7.0, 11.0, 0.0, 0.0];

fn touched_inactive(offsets: &[isize]) -> Vec<isize> {
    offsets.iter().copied().filter(|&o| o >= 2).collect()
}

#[test]
fn zeroed_replay_never_touches_inactive_blocks() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: on an all-zero destination `Zeroed` runs no scale/copy kernel
    // over the inactive blocks; the active blocks are written by their
    // job with `beta = 0` and the inactive ones keep their zeros.
    for irregular_first in [false, true] {
        let (structure, plan) = four_block_plan(irregular_first);
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
        assert!(kernels.scale_offsets().is_empty(), "{irregular_first}");
        assert!(
            touched_inactive(&kernels.copy_offsets()).is_empty(),
            "irregular_first={irregular_first}: {:?}",
            kernels.copy_offsets()
        );
        assert!(dst[2..].iter().all(|v| v.to_bits() == 0));
    }
}

#[test]
fn axpby_zero_assigns_inactive_blocks_without_reading_them() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: `Axpby(0)` is a strong zero: NaN, Inf and -0.0 in the inactive
    // blocks become exactly +0.0 through one copy per block, never a
    // multiply that would keep NaN.
    for irregular_first in [false, true] {
        let (structure, plan) = four_block_plan(irregular_first);
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
}

#[test]
fn axpby_one_leaves_inactive_blocks_untouched() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (structure, plan) = four_block_plan(false);
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
    let (structure, plan) = four_block_plan(false);
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
    let (structure, plan) = four_block_plan(false);
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
    let both = || SectorLeg::new([(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)], false);
    let even = || SectorLeg::new([(Z2Irrep::EVEN, 2)], false);
    // `shapes` holds one `[rows, cols]` per coupled tree pair.
    let space = |codomain: SectorLeg, domain: SectorLeg, shapes: Vec<Vec<usize>>| {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([codomain]),
            FusionProductSpace::new([domain]),
        )
        .coupled_subblock_structure(&Z2FusionRule, 1, shapes)
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

/// Payload components that separate a componentwise real scale from a
/// promoted complex multiply: `0 * inf` and `0 * NaN` pollute the other
/// component, and `x - 0 * y` can flip the sign of a zero.
const SPECIAL_COMPONENTS: [(f64, f64); 12] = [
    (f64::INFINITY, 1.0),
    (-0.0, -0.0),
    (f64::from_bits(0x7ff8_0000_0000_1407), 2.5),
    (1.5, f64::NEG_INFINITY),
    (-0.0, 3.0),
    (f64::from_bits(1), -0.0),
    (0.0, -2.0),
    (-4.0, f64::from_bits(0xfff8_0000_0000_0042)),
    (0.75, -0.0),
    (f64::NEG_INFINITY, -0.0),
    (-1.25, 0.5),
    (2.0, f64::INFINITY),
];

/// Three `2 x 2` storage blocks packed into one `2 x 6` column-major
/// matrix, each with its own real structural coefficient (`0` included:
/// `0 * (inf + 1i)` is `(NaN, 0)` componentwise, `(NaN, NaN)` promoted).
fn three_block_group<C: Copy>(coefficients: [C; 3]) -> FusionBlockMatrixGroup<C> {
    FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows: 2,
        cols: 6,
        needs_clear: false,
        direct_offset: None,
        block_indices: vec![0, 1, 2],
        subblocks: coefficients
            .iter()
            .enumerate()
            .map(|(block, &coefficient)| FusionSubblockMatrixLayout {
                // Storage blocks are transposed against the matrix so the
                // pack is a genuine strided move.
                block: FusionStridedBlockLayout {
                    shape: vec![2, 2],
                    strides: vec![2, 1],
                    offset: 4 * block as isize,
                },
                matrix_offset: 4 * block as isize,
                matrix_strides: vec![1, 2],
                coefficient,
            })
            .collect(),
    }
}

fn component_bits(values: &[Complex64]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|v| (v.re.to_bits(), v.im.to_bits()))
        .collect()
}

fn pair_bits(re: &[f64], im: &[f64]) -> Vec<(u64, u64)> {
    let _exec = strided_kernel::Exec::serial();
    re.iter()
        .zip(im)
        .map(|(re, im)| (re.to_bits(), im.to_bits()))
        .collect()
}

/// What: pack and scatter apply a real structural coefficient to a complex
/// block componentwise, i.e. exactly as it acts on the two real component
/// blocks, for coefficients `1`, `-0.5` and `0`, and for scatter `beta`
/// in `{0, 1, 0.5}`. The coefficient-0 block is VectorInterface's
/// `scale(strided_kernel::Exec::serial(), strided_kernel::Exec::serial(), x, 0) = zero(strided_kernel::Exec::serial(), x) * 0`, an exact zero that wipes NaN and inf;
/// the real component runs give the same zeros.
#[test]
fn pack_and_scatter_scale_real_coefficients_componentwise() {
    let mut exec = strided_kernel::Exec::serial();
    let group = three_block_group([1.0, -0.5, 0.0]);
    let data: Vec<Complex64> = SPECIAL_COMPONENTS
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let re: Vec<f64> = data.iter().map(|v| v.re).collect();
    let im: Vec<f64> = data.iter().map(|v| v.im).collect();
    let mut kernels = crate::StridedHostKernelAdapter::default();

    let mut packed = vec![Complex64::zero(); 12];
    pack_group(exec.reborrow(), &mut kernels, &group, &data, &mut packed).unwrap();
    let mut packed_re = vec![0.0; 12];
    let mut packed_im = vec![0.0; 12];
    pack_group(exec.reborrow(), &mut kernels, &group, &re, &mut packed_re).unwrap();
    pack_group(exec.reborrow(), &mut kernels, &group, &im, &mut packed_im).unwrap();
    assert_eq!(component_bits(&packed), pair_bits(&packed_re, &packed_im));
    // The coefficient-0 block of the `(inf, 1)`-bearing payload is the
    // sharpest case; pin it independently of the oracle as well:
    // TensorKit's `scale(strided_kernel::Exec::serial(), complex(Inf, 1.0), 0.0)` is `0.0 + 0.0im`.
    let zero_block = &packed[8..];
    assert!(SPECIAL_COMPONENTS[8..]
        .iter()
        .any(|&(re, im)| !re.is_finite() || !im.is_finite()));
    assert_eq!(
        component_bits(zero_block),
        vec![(0.0f64.to_bits(), 0.0f64.to_bits()); 4]
    );

    // Why a finite, nonzero destination for the general beta: beta is a
    // payload-typed scalar, so `(0.5 + 0i) * dst` is a complex multiply by
    // design; only the packed side carries the non-finite values there.
    let finite: Vec<Complex64> = (0..12)
        .map(|k| Complex64::new(k as f64 * 0.37 - 2.1, 1.3 - k as f64 * 0.29))
        .collect();
    for beta in [0.0, 1.0, 0.5] {
        let initial = if beta == 0.5 { &finite } else { &data };
        let mut scattered = initial.clone();
        scatter_group(
            exec.reborrow(),
            &mut kernels,
            &group,
            &mut scattered,
            &packed,
            Complex64::new(beta, 0.0),
        )
        .unwrap();
        let mut scattered_re: Vec<f64> = initial.iter().map(|v| v.re).collect();
        let mut scattered_im: Vec<f64> = initial.iter().map(|v| v.im).collect();
        scatter_group(
            exec.reborrow(),
            &mut kernels,
            &group,
            &mut scattered_re,
            &packed_re,
            beta,
        )
        .unwrap();
        scatter_group(
            exec.reborrow(),
            &mut kernels,
            &group,
            &mut scattered_im,
            &packed_im,
            beta,
        )
        .unwrap();
        assert_eq!(
            component_bits(&scattered),
            pair_bits(&scattered_re, &scattered_im),
            "beta = {beta}"
        );
    }
}

/// What: a complex (anyonic) coefficient keeps the full complex multiply,
/// except that an exact `1 + 0i` does not multiply at all.
#[test]
fn pack_keeps_complex_coefficients_complex_and_skips_one() {
    let coefficients = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.6, 0.8),
        Complex64::new(-0.5, 0.25),
    ];
    let group = three_block_group(coefficients);
    let data: Vec<Complex64> = SPECIAL_COMPONENTS
        .iter()
        .map(|&(re, im)| Complex64::new(re, im))
        .collect();
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut packed = vec![Complex64::zero(); 12];
    pack_group(
        strided_kernel::Exec::serial(),
        &mut kernels,
        &group,
        &data,
        &mut packed,
    )
    .unwrap();
    for (block, coefficient) in coefficients.into_iter().enumerate() {
        for row in 0..2 {
            for col in 0..2 {
                let value = data[4 * block + 2 * row + col];
                let want = if block == 0 {
                    value
                } else {
                    value * coefficient
                };
                let got = packed[4 * block + row + 2 * col];
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (want.re.to_bits(), want.im.to_bits()),
                    "block {block} ({row}, {col})"
                );
            }
        }
    }
}

/// What: for finite payloads the componentwise scale that pack applies is
/// bit for bit the promoted complex product `(strided_kernel::Exec::serial(), c + 0i)(x + yi)` whenever
/// both component products are nonzero, and differs only in the sign of a
/// zero product, where the componentwise result is the real oracle's.
#[test]
fn componentwise_pack_matches_promoted_product_on_finite_values() {
    let mut exec = strided_kernel::Exec::serial();
    let coefficients = [-0.5, 3.0, 0.1];
    let group = three_block_group(coefficients);
    let finite: Vec<Complex64> = (0..12)
        .map(|k| {
            let k = k as f64;
            Complex64::new((k - 5.5) * 0.37, (7.0 - k) * 1.3 + 0.01)
        })
        .collect();
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut packed = vec![Complex64::zero(); 12];
    pack_group(exec.reborrow(), &mut kernels, &group, &finite, &mut packed).unwrap();
    for (block, &coefficient) in coefficients.iter().enumerate() {
        for row in 0..2 {
            for col in 0..2 {
                let value = finite[4 * block + 2 * row + col];
                let promoted = Complex64::new(coefficient, 0.0) * value;
                let got = packed[4 * block + row + 2 * col];
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (promoted.re.to_bits(), promoted.im.to_bits())
                );
            }
        }
    }

    // `-1 * (+0 - 3i)`: componentwise `(-0, 3)`; promoted
    // `(-1)(+0) - (0)(-3) = -0 - (strided_kernel::Exec::serial(), -0) = +0`.
    let group = three_block_group([-1.0, -1.0, -1.0]);
    let data = vec![Complex64::new(0.0, -3.0); 12];
    let mut packed = vec![Complex64::zero(); 12];
    pack_group(exec.reborrow(), &mut kernels, &group, &data, &mut packed).unwrap();
    let promoted = Complex64::new(-1.0, 0.0) * data[0];
    assert_eq!(promoted.re.to_bits(), 0.0f64.to_bits());
    assert!(packed
        .iter()
        .all(|v| v.re.to_bits() == (-0.0f64).to_bits() && v.im == 3.0));
}

/// Canonical Z2 plan with job coefficients `+1` (even block, offset 0) and
/// `-1` (odd block, offset 4), and operands whose products are
/// `[1, 2, 3, 4]` and `[10]`.
fn signed_z2_storage_plan() -> (FusionBlockContractPlan<f64>, Vec<Complex64>, Vec<Complex64>) {
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)], false);
    let space = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure(&Z2FusionRule, 1, [vec![2, 2], vec![1, 1]])
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
        matches!(error, OperationError::UnsupportedTensorContractScope { .. }),
        "{error:?}"
    );
    assert_eq!(gemm.unit_calls, 0);
    assert_eq!(dst, vec![Complex64::new(7.0, 0.0); 5]);
}
