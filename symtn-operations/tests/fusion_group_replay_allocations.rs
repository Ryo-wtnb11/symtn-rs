use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{BlockStructure, SectorId};
use symtn_operations::{
    FusionBlockContractGroupPlan, FusionBlockContractPlan, FusionBlockContractWorkspace,
    FusionBlockMatrixGroup, FusionStridedBlockLayout, FusionSubblockMatrixLayout, OperationError,
    Rank2Gemm, StridedHostKernelAdapter,
};

use crate::counting_alloc;

struct ScalarGemm;

impl Rank2Gemm<f64> for ScalarGemm {
    fn matmul_rank2(
        &mut self,
        _engine: &mut dyn Engine,
        dst: &mut [f64],
        lhs: &[f64],
        rhs: &[f64],
        _rows: usize,
        _contracted: usize,
        _cols: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<(), OperationError> {
        dst[0] = alpha * lhs[0] * rhs[0] + beta * dst[0];
        Ok(())
    }
}

fn scalar_group() -> FusionBlockMatrixGroup {
    FusionBlockMatrixGroup {
        coupled: SectorId::new(0),
        rows: 1,
        cols: 1,
        needs_clear: false,
        direct_offset: None,
        block_indices: vec![0],
        subblocks: vec![FusionSubblockMatrixLayout {
            block: FusionStridedBlockLayout {
                shape: vec![1],
                strides: vec![1],
                offset: 0,
            },
            matrix_offset: 0,
            matrix_strides: vec![1],
            coefficient: 1.0,
        }],
    }
}

#[test]
fn warmed_irregular_group_replay_allocates_nothing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let structure = Arc::new(BlockStructure::trivial(&[1]).unwrap());
    let plan = FusionBlockContractPlan::from_parts(
        Arc::clone(&structure),
        Arc::clone(&structure),
        Arc::clone(&structure),
        Vec::new(),
        vec![
            FusionBlockContractGroupPlan::new(scalar_group(), scalar_group(), scalar_group())
                .unwrap(),
        ],
    )
    .unwrap();
    let mut kernels = StridedHostKernelAdapter::default();
    let mut gemm = ScalarGemm;
    let mut workspace = FusionBlockContractWorkspace::default();
    let lhs = [2.0];
    let rhs = [3.0];
    let mut dst = [5.0];

    for _ in 0..2 {
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
    }
    counting_alloc::start();
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
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 0);
    assert_eq!(dst, [6.0]);
}

/// A fresh adapter per eager call, as degeneracy restriction and scatter
/// build it, must not allocate its normalization or traversal scratch up to
/// rank 8: the per-call cost is then independent of the call count.
#[test]
fn a_fresh_adapter_checked_block_copy_allocates_nothing_up_to_rank_eight() {
    let mut exec = strided_kernel::Exec::serial();
    let _serial = crate::counting_alloc::serial();
    let shape = [2usize; 8];
    let mut dst_strides = [0isize; 8];
    let mut src_strides = [0isize; 8];
    let (mut dst_stride, mut src_stride) = (1isize, 1isize);
    for axis in 0..8 {
        dst_strides[axis] = dst_stride;
        src_strides[axis] = src_stride;
        dst_stride *= 2;
        // A window of a parent three wide on every axis: nothing fuses.
        src_stride *= 3;
    }
    let src: Vec<f64> = (0..3usize.pow(8)).map(|value| value as f64).collect();
    let mut dst = vec![0.0; 1 << 8];

    counting_alloc::start();
    for (alpha, beta) in [(1.0, 0.0), (1.0, 1.0), (2.0, 0.5)] {
        StridedHostKernelAdapter::default()
            .tensoradd_strided_checked(
                exec.reborrow(),
                &mut dst,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                1,
                false,
                alpha,
                beta,
            )
            .unwrap();
    }
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 0);
}
