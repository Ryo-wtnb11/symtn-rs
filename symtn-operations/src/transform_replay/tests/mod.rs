use super::*;
use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
use std::time::Duration;
use symtn_core::{BlockKey, BlockSpec};
use symtn_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite,
};

mod compile_validation;
mod destination_scales;
mod overwrite;
mod profile;
mod threaded;

/// Test-only host operand: a block structure and the storage it describes.
struct TestTensor {
    structure: Arc<BlockStructure>,
    data: Vec<f64>,
}

impl TestTensor {
    fn new(data: Vec<f64>, structure: BlockStructure) -> Self {
        assert_eq!(structure.required_len().unwrap(), data.len());
        Self {
            structure: Arc::new(structure),
            data,
        }
    }

    fn structure(&self) -> &Arc<BlockStructure> {
        &self.structure
    }

    fn data(&self) -> &[f64] {
        &self.data
    }

    fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

fn fixture() -> (TestTensor, TestTensor, TreeTransformStructure<f64>) {
    let src = TestTensor::new(
        vec![3.0],
        BlockStructure::packed_column_major(1, [vec![1]]).unwrap(),
    );
    let dst = TestTensor::new(
        vec![10.0, 20.0],
        BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap(),
    );
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::single(0, 0, 2.0)],
    )
    .unwrap();
    (dst, src, structure)
}

fn custom_structure(blocks: Vec<BlockSpec>) -> BlockStructure {
    BlockStructure::from_blocks_with_rank(1, blocks).unwrap()
}

fn block(sector: usize, shape: usize, stride: usize, offset: usize) -> BlockSpec {
    BlockSpec::with_key(BlockKey::ordinal(sector), vec![shape], vec![stride], offset).unwrap()
}

fn identity_multi_fixture() -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let src = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
    let dst =
        Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1]]).unwrap());
    let replay = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap();
    (dst, src, replay)
}

#[test]
fn tree_transform_replay_dispatches_through_kernel_adapter() {
    use crate::kernel_adapter::observed::{KernelCall, ObservedKernels};

    // Source columns sit in a different block order from the spec, and the
    // destination blocks are reversed, so pack and scatter must follow the
    // spec's block indices rather than storage order.
    let src =
        Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2], vec![2]]).unwrap());
    let dst = Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
    let structure = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[TreeTransformBlockSpec::multi(
            vec![1, 0],
            vec![0, 2, 1],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut dst_data = [0.0; 4];
    let mut kernels = ObservedKernels::recording();

    tree_transform_structure_with_strided_kernel_raw(
        strided_kernel::Exec::serial(),
        &mut kernels,
        &mut TreeTransformWorkspace::default(),
        &structure,
        &dst,
        &src,
        &mut dst_data,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst_data, [7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(kernels.copy_offsets().len(), 3);
    assert_eq!(kernels.count(KernelCall::Recoupling), 1);
    assert_eq!(kernels.count(KernelCall::Axpby), 2);
    assert_eq!(kernels.count(KernelCall::Add), 0);
    assert!(kernels.scale_offsets().is_empty());
}
