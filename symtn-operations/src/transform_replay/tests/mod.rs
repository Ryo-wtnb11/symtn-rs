use super::*;
use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
use std::time::Duration;
use symtn_core::{BlockKey, BlockSpec};
use symtn_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite,
};

#[cfg(feature = "tenferro")]
mod chunk;
mod compile_validation;
#[cfg(feature = "tenferro")]
mod counters;
mod destination_scales;
#[cfg(feature = "tenferro")]
mod members;
mod overwrite;
mod profile;
#[cfg(feature = "tenferro")]
mod recording;
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

/// SU(2) `W ⊗ P ← W ⊗ P`, `W` the spins 0, ½, 1, 3/2, 2 at degeneracy two
/// and `P` spin ½ at degeneracy one, in the canonical layout: fifteen
/// `[2, 1, 2, 1]` blocks in coupled sectors 0, ½, …, 5/2 of 4, 16, 16, 16, 4
/// and 4 elements, at the offsets below. Each sector matrix has 2 (outer
/// sectors) or 4 rows; a block `(j1, j2)` of sector `c` occupies its
/// codomain tree's two rows and domain tree's two columns, so blocks sharing
/// a domain tree interleave.
///
/// The transform maps the structure onto itself: the seven blocks whose
/// external spins fix one coupled sector are Singles, and the two blocks of
/// each equal-spin pair `(j, j)`, `j ≥ ½`, form a Multi group. Its Singles
/// overlap in one place (`(0, 0)` and `(1, 0)` of sector ½), and each Multi
/// group's two blocks lie in different sectors with other groups' blocks
/// between them in address order.
mod su2_interleaved {
    use super::*;
    use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SU2FusionRule, SU2Irrep, SectorLeg};

    /// `(destination offset, coefficient)` of each Single.
    pub(crate) const SINGLES: [(usize, f64); 7] = [
        (4, 2.0),
        (6, -1.0),
        (12, 0.5),
        (22, 3.0),
        (28, -2.0),
        (38, 1.5),
        (44, -0.5),
    ];
    /// Block offsets of each Multi group and its row-major
    /// destination-by-source matrix.
    pub(crate) const MULTIS: [([usize; 2], [f64; 4]); 4] = [
        ([0, 20], [1.0, 2.0, 3.0, 4.0]),
        ([14, 36], [-1.0, 2.0, 0.5, 1.0]),
        ([30, 52], [2.0, -3.0, 1.0, 1.0]),
        ([46, 56], [0.5, 0.25, -1.0, 3.0]),
    ];

    /// Physical addresses of the block at `offset`, in its column-major
    /// element order.
    pub(crate) fn addresses(offset: usize) -> [usize; 4] {
        let rows = if (4..52).contains(&offset) { 4 } else { 2 };
        [offset, offset + 1, offset + rows, offset + rows + 1]
    }

    pub(crate) fn fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
        let w = || {
            SectorLeg::new(
                (0..=4).map(|twice_spin| (SU2Irrep::from_twice_spin(twice_spin), 2)),
                false,
            )
        };
        let p = || SectorLeg::new([(SU2Irrep::from_twice_spin(1), 1)], false);
        let structure = FusionTreeHomSpace::new(
            FusionProductSpace::new([w(), p()]),
            FusionProductSpace::new([w(), p()]),
        )
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
        assert_eq!(structure.required_len().unwrap(), 60);
        assert_eq!(structure.block_count(), 15);
        let block_at = |offset: usize| {
            let index = (0..15)
                .find(|&index| structure.block(index).unwrap().offset() == offset)
                .unwrap();
            let block = structure.block(index).unwrap();
            assert_eq!(block.shape(), [2, 1, 2, 1]);
            let [first, second, third, _] = addresses(offset);
            assert_eq!(
                (block.strides()[0], block.strides()[2]),
                (second - first, third - first)
            );
            index
        };
        let mut specs = SINGLES
            .iter()
            .map(|&(offset, coefficient)| {
                TreeTransformBlockSpec::single(block_at(offset), block_at(offset), coefficient)
            })
            .collect::<Vec<_>>();
        specs.extend(MULTIS.iter().map(|(offsets, matrix)| {
            let blocks = offsets.map(block_at).to_vec();
            TreeTransformBlockSpec::multi(blocks.clone(), blocks, matrix.to_vec())
        }));
        let transform =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        (structure, transform)
    }

    /// `alpha` times the fixture's transform of `source`, by block.
    pub(crate) fn oracle(source: &[f64], alpha: f64) -> Vec<f64> {
        let mut expected = vec![f64::NAN; 60];
        for (offset, coefficient) in SINGLES {
            for at in addresses(offset) {
                expected[at] = alpha * coefficient * source[at];
            }
        }
        for (offsets, matrix) in MULTIS {
            let [left, right] = offsets.map(addresses);
            for element in 0..4 {
                let (a, b) = (source[left[element]], source[right[element]]);
                expected[left[element]] = alpha * (matrix[0] * a + matrix[1] * b);
                expected[right[element]] = alpha * (matrix[2] * a + matrix[3] * b);
            }
        }
        expected
    }
}
pub(super) use su2_interleaved::{fixture as su2_interleaved_fixture, oracle as su2_oracle};

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
