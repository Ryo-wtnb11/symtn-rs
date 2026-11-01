use super::*;

use symtn_core::Placement;
use symtn_core::{
    BlockStructure, BraidingStyleKind, FusionProductSpace, FusionStyleKind, SectorLeg, SectorVec,
    TensorStorage, Z2FusionRule,
};

use crate::probe::{self, Event};
use crate::{DenseTreeTransformOperations, TensorContractWorkspace};

/// Storage with no host-slice access: compiling the storage-direct path
/// against this type proves the seam has no host contract.
#[derive(Debug)]
struct OpaqueStorage<T> {
    cells: Vec<T>,
}

impl<T> TensorStorage<T> for OpaqueStorage<T> {
    fn len(&self) -> usize {
        self.cells.len()
    }

    fn placement(&self) -> Placement {
        Placement::Host
    }
}

struct NaiveOpaqueGemm;

impl StorageGemm<f64, OpaqueStorage<f64>, OpaqueStorage<f64>, OpaqueStorage<f64>>
    for NaiveOpaqueGemm
{
    fn matmul_range_into(
        &mut self,
        dst: &mut OpaqueStorage<f64>,
        dst_offset: usize,
        lhs: &OpaqueStorage<f64>,
        lhs_offset: usize,
        rhs: &OpaqueStorage<f64>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = 0.0;
                for inner in 0..contracted {
                    sum += lhs.cells[lhs_offset + row + rows * inner]
                        * rhs.cells[rhs_offset + inner + contracted * col];
                }
                dst.cells[dst_offset + row + rows * col] = sum;
            }
        }
        Ok(())
    }
}

#[test]
fn canonical_region_join_scales_missing_sector_without_layout_compile() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let outer = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let inner = || SectorLeg::new([(SectorId::new(0), 1)], false);
    let space = |codomain: SectorLeg, domain: SectorLeg| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([codomain]),
                FusionProductSpace::new([domain]),
            ),
        )
        .unwrap()
    };
    let lhs = space(outer(), inner());
    let rhs = space(inner(), outer());
    let dst = space(outer(), outer());

    let layout_rec = probe::record();
    let plan = compile_fusion_block_contract_plan(
        &rule,
        &dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(layout_rec.get(Event::FusionLayoutCompile), 0);

    let mut output = vec![11.0, 7.0];
    let mut dense = DenseTreeTransformOperations::default();
    let mut dense_workspace = TensorContractWorkspace::default();
    let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut dense, &mut dense_workspace),
        &mut fusion_workspace,
        dst.structure(),
        &mut output,
        lhs.structure(),
        &[3.0],
        rhs.structure(),
        &[5.0],
        2.0,
        3.0,
    )
    .unwrap();

    // What: the matched sector applies alpha and beta, while the absent
    // inner sector applies beta exactly once across its contiguous range.
    assert_eq!(output, vec![63.0, 21.0]);
}

#[derive(Clone, Copy)]
struct LayoutToyGenericRule;

impl symtn_core::testing::Provider for LayoutToyGenericRule {}

impl FusionRule for LayoutToyGenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, 0) => [SectorId::new(0)].into_iter().collect(),
            (1, 1) => [SectorId::new(1)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

#[test]
fn generic_multiplicity_grid_uses_canonical_region_gemm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = LayoutToyGenericRule;
    let leg = || SectorLeg::new([(SectorId::new(1), 1)], false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = || {
        let homspace = homspace();
        let key_count = homspace.fusion_tree_keys_generic(&rule).unwrap().len();
        assert_eq!(key_count, 4);
        DynamicFusionMapSpace::from_final_homspace_generic(&rule, homspace).unwrap()
    };
    let lhs = space();
    let rhs = space();
    let dst = space();

    let layout_rec = probe::record();
    // The checked core rung of `lhs[2, 3] . rhs[0, 1]` (core form).
    let plan = FusionBlockContractPlan::<f64>::try_from_canonical_coupled_regions_with_ops_generic(
        dst.structure(),
        dst.nout(),
        lhs.structure(),
        lhs.nout(),
        rhs.structure(),
        rhs.nout(),
        MatrixOp::Identity,
        MatrixOp::Identity,
    )
    .unwrap()
    .expect("the multiplicity grid is canonically aligned");
    assert_eq!(layout_rec.get(Event::FusionLayoutCompile), 0);

    let mut output = vec![0.0; 4];
    let mut dense = DenseTreeTransformOperations::default();
    let mut dense_workspace = TensorContractWorkspace::default();
    let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut dense, &mut dense_workspace),
        &mut fusion_workspace,
        dst.structure(),
        &mut output,
        lhs.structure(),
        &[1.0, 2.0, 3.0, 4.0],
        rhs.structure(),
        &[5.0, 6.0, 7.0, 8.0],
        1.0,
        0.0,
    )
    .unwrap();

    // What: outer-multiplicity vertices remain matrix rows/columns on the
    // direct canonical route.
    assert_eq!(output, vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn storage_direct_replay_runs_without_host_slice_contract() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let fusion_space = || {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    let space = fusion_space();
    let plan = compile_fusion_block_contract_plan(
        &rule,
        &space.clone(),
        &space.clone(),
        &space.clone(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();

    let lhs = OpaqueStorage {
        cells: vec![2.0, 3.0],
    };
    let rhs = OpaqueStorage {
        cells: vec![5.0, 7.0],
    };
    let mut dst = OpaqueStorage {
        cells: vec![0.0, 0.0],
    };
    plan.execute_direct_on_storage_prezeroed(&mut NaiveOpaqueGemm, &mut dst, &lhs, &rhs)
        .unwrap();

    // Two 1x1 sector matrices: dst = lhs * rhs per sector.
    assert_eq!(dst.cells, vec![10.0, 21.0]);
}

#[test]
fn storage_direct_replay_matches_host_execute_raw() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let leg =
        |dims: usize| SectorLeg::new([(SectorId::new(0), dims), (SectorId::new(1), dims)], false);
    let fusion_space = |dims: usize| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(dims)]),
                FusionProductSpace::new([leg(dims)]),
            ),
        )
        .unwrap()
    };
    let space = fusion_space(2);
    let len = space.required_len().unwrap();
    let lhs_data: Vec<f64> = (0..len).map(|i| 0.5 * i as f64 - 1.0).collect();
    let rhs_data: Vec<f64> = (0..len).map(|i| 1.5 - 0.25 * i as f64).collect();
    let layout_rec = probe::record();
    let plan = compile_fusion_block_contract_plan(
        &rule,
        &space.clone(),
        &space.clone(),
        &space.clone(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();
    // What: canonical tensor-owned regions bypass the per-tree layout compiler.
    assert_eq!(layout_rec.get(Event::FusionLayoutCompile), 0);

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TensorContractWorkspace::default();
    let mut expected = vec![0.0; len];
    let structure = std::sync::Arc::clone(space.structure());
    let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut workspace),
        &mut fusion_workspace,
        &structure,
        &mut expected,
        &structure,
        &lhs_data,
        &structure,
        &rhs_data,
        1.0,
        0.0,
    )
    .unwrap();

    let mut direct = vec![0.0; len];
    let mut gemm = HostStorageGemm::new(engine, &mut backend, &mut workspace);
    plan.execute_direct_on_storage_prezeroed(
        &mut gemm,
        &mut direct,
        &lhs_data.clone(),
        &rhs_data.clone(),
    )
    .unwrap();

    assert_eq!(direct, expected);
}

fn z2_adjoint_mapping_spaces() -> (DynamicFusionMapSpace, DynamicFusionMapSpace) {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let storage = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let logical = storage
        .validate_rule(&rule)
        .and_then(|()| storage.adjoint_view())
        .unwrap();
    (logical, storage)
}

/// The DynamicTree scorer charges a borrowable identity adjoint
/// nothing, as TensorKit `cfaa073e` `contract_memcost` does through
/// `has_shared_permute(::AdjointTensorMap)`. `A = P'` (8 elements) contracts
/// its domain `(2, 3)` with `U`'s codomain in reverse order `(1, 0)`, `U`
/// (6 elements). TensorKit's candidates: m1 keeps `A`'s order (`A` free,
/// `U` permuted: cost |U| = 6), m2 keeps `U`'s (`A` permuted: cost |P| = 8),
/// the reversed orders also permute the output; m1 wins. The same holds for
/// `P` itself read directly (`P[2, 3]·U[1, 0]`).
#[test]
fn an_identity_side_is_free_in_selection_over_canonical_layouts() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = std::sync::Arc::new(Z2FusionRule);
    let unit = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&rule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([unit(), unit()]),
            FusionProductSpace::new([unit(), unit()]),
        ),
    )
    .unwrap();
    let rhs = crate::BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&rule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([unit(), unit()]),
            FusionProductSpace::new([SectorLeg::new(
                [(SectorId::new(0), 1), (SectorId::new(1), 2)],
                false,
            )]),
        ),
    )
    .unwrap();
    assert_eq!(canonical.space().required_len().unwrap(), 8);
    assert_eq!(rhs.space().required_len().unwrap(), 6);
    let (lhs_axes, rhs_axes) = ([2, 3], [1, 0]);
    for adjoint in [true, false] {
        let lhs_logical = if adjoint {
            canonical.adjoint_view().unwrap()
        } else {
            canonical.clone()
        };
        let dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
            &lhs_logical,
            &rhs,
            &lhs_axes,
            &rhs_axes,
            symtn_operations::OutputAxisOrder::identity(),
            2,
        )
        .unwrap();
        let axes = symtn_operations::TensorContractSpec::new_with_conjugation(
            &lhs_axes,
            &rhs_axes,
            symtn_operations::OutputAxisOrder::identity(),
            adjoint,
            false,
        );
        let lhs = if adjoint {
            crate::FusionOperand::adjoint(canonical.space())
        } else {
            crate::FusionOperand::direct(canonical.space())
        };
        let mut context =
            crate::TensorContractFusionExecutionContext::<f64, crate::RuleIdentity>::default();
        let resolution = context
            .compile_storage_contract_resolution(
                engine,
                &dst,
                lhs,
                crate::FusionOperand::direct(rhs.space()),
                axes,
            )
            .unwrap();
        let super::super::resolution::ContractRoute::DynamicTree(artifact) = &resolution.route
        else {
            panic!("no candidate is zero-copy, so the route is DynamicTree");
        };
        // What: the scorer takes m1, and the artifact reads `P` in place while
        // copying `U`.
        assert_eq!(
            artifact.borrowed_sources(),
            (true, false),
            "adjoint {adjoint}"
        );
    }
}

#[test]
fn adjoint_operand_without_canonical_regions_uses_exact_fallback() {
    let rule = Z2FusionRule;
    let (logical, storage) = z2_adjoint_mapping_spaces();
    let dst = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        logical,
        std::sync::Arc::new(rule),
    )
    .unwrap();
    let fusion_operand_projection_prepares_rec = probe::record();
    let layout_rec = probe::record();
    let resolution = crate::plan_compose::<crate::HostEagerExecutor, _>(
        &dst,
        crate::FusionOperand::adjoint(&storage),
        crate::FusionOperand::adjoint(&storage),
    )
    .unwrap();

    // What: a transposed logical structure without canonical coupled
    // regions prepares the exact projection and retains the tree-mapped
    // implementation: the irregular Host core.
    let (core, swapped) = resolution.direct_core().unwrap();
    assert!(!swapped && !core.is_fully_direct());
    assert_eq!(
        fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
        2
    );
    assert!(layout_rec.get(Event::FusionLayoutCompile) > 0);
}

#[test]
fn rank22_adjoint_reentry_keeps_matrix_orientation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let layout = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let space = layout;
    let dst = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        space.clone(),
        std::sync::Arc::new(rule),
    )
    .unwrap();
    let resolution = crate::plan_compose::<crate::HostEagerExecutor, _>(
        &dst,
        crate::FusionOperand::adjoint(&space),
        crate::FusionOperand::adjoint(&space),
    )
    .unwrap();
    let (plan, swapped) = resolution.direct_core().unwrap();
    assert!(!swapped);

    let len = space.required_len().unwrap();
    let lhs_data = (0..len)
        .map(|index| (index % 17) as f64 - 8.0)
        .collect::<Vec<_>>();
    let rhs_data = (0..len)
        .map(|index| (index % 13) as f64 - 6.0)
        .collect::<Vec<_>>();
    let mut actual = vec![0.0; len];
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TensorContractWorkspace::default();
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut workspace),
        &mut FusionBlockContractWorkspace::default(),
        space.structure(),
        &mut actual,
        space.structure(),
        &lhs_data,
        space.structure(),
        &rhs_data,
        1.0,
        0.0,
    )
    .unwrap();

    let mut expected = vec![0.0; len];
    for region in space
        .structure()
        .coupled_sector_regions(space.nout())
        .unwrap()
        .unwrap()
        .iter()
    {
        let start = region.range().start;
        let rows = region.rows();
        let cols = region.cols();
        for col in 0..rows {
            for row in 0..cols {
                expected[start + row + cols * col] = (0..rows)
                    .map(|contracted| {
                        lhs_data[start + contracted + rows * row]
                            * rhs_data[start + col + rows * contracted]
                    })
                    .sum();
            }
        }
    }

    // What: post-projection core re-entry retains both lazy-adjoint
    // matrix operations instead of replaying parent storage as identity.
    assert_eq!(actual, expected);
}

/// GPU vertical: the same core direct replay executed on CUDA
/// storage must reproduce the host result bit-for-bit (same GEMM
/// ordering, overwrite semantics). Requires a CUDA device; run with
/// `cargo test --features cuda -- --ignored`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn storage_direct_replay_on_cuda_matches_host() {
    use symtn_dense::CudaDenseContext;
    use symtn_operations::cuda::{CudaStorage, CudaStorageGemm};

    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 3), (SectorId::new(1), 3)], false);
    let space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let len = space.required_len().unwrap();
    let lhs_data: Vec<f64> = (0..len).map(|i| 0.5 * i as f64 - 1.0).collect();
    let rhs_data: Vec<f64> = (0..len).map(|i| 1.5 - 0.25 * i as f64).collect();
    let plan = compile_fusion_block_contract_plan(
        &rule,
        &space.clone(),
        &space.clone(),
        &space.clone(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TensorContractWorkspace::default();
    let mut expected = vec![0.0; len];
    let mut cpu = symtn_dense::CpuEngine::serial();
    let mut gemm = HostStorageGemm::new(&mut cpu, &mut backend, &mut workspace);
    plan.execute_direct_on_storage_prezeroed(&mut gemm, &mut expected, &lhs_data, &rhs_data)
        .unwrap();

    let mut cuda =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut cuda).unwrap();
    let lhs_dev = CudaStorage::upload(&ctx, &lhs_data).unwrap();
    let rhs_dev = CudaStorage::upload(&ctx, &rhs_data).unwrap();
    let mut dst_dev = CudaStorage::upload(&ctx, &vec![0.0; len]).unwrap();
    plan.execute_direct_on_storage_prezeroed(
        &mut CudaStorageGemm::new(&mut ctx),
        &mut dst_dev,
        &lhs_dev,
        &rhs_dev,
    )
    .unwrap();
    let result = dst_dev.download(&ctx).unwrap();

    assert_eq!(result, expected);
}

/// Entries of one block in column-major order over its axes, which is the
/// column-major matrix whose rows fuse any leading split of the axes.
fn block_matrix(structure: &BlockStructure, index: usize, data: &[f64]) -> Vec<f64> {
    let block = structure.block(index).unwrap();
    let shape = block.shape();
    let len = shape.iter().product::<usize>();
    (0..len)
        .map(|mut linear| {
            let mut at = block.offset();
            for (&extent, &stride) in shape.iter().zip(block.strides()) {
                at += (linear % extent) * stride;
                linear /= extent;
            }
            data[at]
        })
        .collect()
}

/// Hand oracle of the core GEMM: `C[X, Z] = Σ_Y A[X, Y] B[Y, Z]` over tree
/// identity, read from block keys and degeneracy layouts only.
fn core_oracle(
    dst: &DynamicFusionMapSpace,
    lhs: (&Arc<BlockStructure>, &[f64]),
    rhs: (&Arc<BlockStructure>, &[f64]),
    nout: usize,
) -> Vec<f64> {
    let dims = |structure: &BlockStructure, index: usize, split: usize| {
        let shape = structure.block(index).unwrap().shape().to_vec();
        (
            shape[..split].iter().product::<usize>(),
            shape[split..].iter().product::<usize>(),
        )
    };
    let pair = |structure: &BlockStructure, index: usize| match structure
        .block(index)
        .unwrap()
        .key()
        .clone()
    {
        BlockKey::FusionTree(pair) => pair,
        other => panic!("{other:?}"),
    };
    let dst_structure = dst.structure();
    let mut out = vec![0.0; dst.required_len().unwrap()];
    for d in 0..dst_structure.block_count() {
        let key = pair(dst_structure, d);
        let (rows, cols) = dims(dst_structure, d, nout);
        let mut value = vec![0.0; rows * cols];
        for l in 0..lhs.0.block_count() {
            let lkey = pair(lhs.0, l);
            if lkey.codomain_tree() != key.codomain_tree() {
                continue;
            }
            for r in 0..rhs.0.block_count() {
                let rkey = pair(rhs.0, r);
                if rkey.domain_tree() != key.domain_tree()
                    || rkey.codomain_tree() != lkey.domain_tree()
                {
                    continue;
                }
                let (_, inner) = dims(lhs.0, l, nout);
                let a = block_matrix(lhs.0, l, lhs.1);
                let b = block_matrix(rhs.0, r, rhs.1);
                for col in 0..cols {
                    for row in 0..rows {
                        for k in 0..inner {
                            value[row + rows * col] += a[row + rows * k] * b[k + inner * col];
                        }
                    }
                }
            }
        }
        let block = dst_structure.block(d).unwrap();
        let shape = block.shape();
        for (mut linear, entry) in value.into_iter().enumerate() {
            let mut at = block.offset();
            for (&extent, &stride) in shape.iter().zip(block.strides()) {
                at += (linear % extent) * stride;
                linear /= extent;
            }
            out[at] = entry;
        }
    }
    out
}

fn run_core_plan(
    plan: &FusionBlockContractPlan,
    dst: &Arc<BlockStructure>,
    init: &[f64],
    lhs: (&Arc<BlockStructure>, &[f64]),
    rhs: (&Arc<BlockStructure>, &[f64]),
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut output = init.to_vec();
    let mut dense = DenseTreeTransformOperations::default();
    let mut dense_workspace = TensorContractWorkspace::default();
    let mut fusion_workspace = FusionBlockContractWorkspace::<f64>::default();
    plan.execute_raw(
        engine,
        &mut crate::StridedHostKernelAdapter::default(),
        &mut BackendRank2Gemm::<_, _, f64>::new(&mut dense, &mut dense_workspace),
        &mut fusion_workspace,
        dst,
        &mut output,
        lhs.0,
        lhs.1,
        rhs.0,
        rhs.1,
        alpha,
        beta,
    )
    .unwrap();
    output
}

/// What: the checked Generic core plan of canonical staged
/// structures compiles from coupled regions (no per-subblock layout) and is
/// bitwise identical to the per-subblock builder; a non-canonical operand
/// tiling falls back to that builder. Outer multiplicity (`1 ⊗ 1 → 1` twice),
/// dual and non-dual legs, degeneracies 2 and 3, two active coupled sectors,
/// and an inactive destination sector are covered. A zero-extent tree cannot
/// be built: `SectorLeg` drops zero-degeneracy sectors.
#[test]
fn checked_core_plan_uses_coupled_regions_bitwise_equal_to_general_builder() {
    use crate::tests::GenericMultiplicityRule;
    let rule = GenericMultiplicityRule;
    let full = |dual| SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 3)], dual);
    let vacuum = |dual| SectorLeg::new([(SectorId::new(0), 2)], dual);
    let space = |codomain: [SectorLeg; 2], domain: [SectorLeg; 2]| {
        DynamicFusionMapSpace::from_final_homspace_generic(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new(codomain),
                FusionProductSpace::new(domain),
            ),
        )
        .unwrap()
    };
    let values = |space: &DynamicFusionMapSpace, seed: f64| {
        (0..space.required_len().unwrap())
            .map(|index| ((index * 7 + 3) % 13) as f64 * 0.25 - 1.5 + seed)
            .collect::<Vec<_>>()
    };
    // The checked core plan of `lhs[2, 3] . rhs[0, 1]` (core form): the
    // coupled-region plan, else the general per-subblock builder.
    let checked_core = |dst: &Arc<symtn_core::BlockStructure>,
                        lhs: &Arc<symtn_core::BlockStructure>,
                        rhs: &Arc<symtn_core::BlockStructure>| {
        FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_generic(
            dst,
            2,
            lhs,
            2,
            rhs,
            2,
            MatrixOp::Identity,
            MatrixOp::Identity,
        )
        .unwrap()
        .unwrap_or_else(|| {
            compile_checked_generic_core_plan_general(dst, 2, lhs, 2, rhs, 2).unwrap()
        })
    };

    let dst = space([full(false), full(true)], [full(false), full(false)]);
    let cases = [
        (
            "multiplicity",
            space([full(false), full(true)], [full(true), full(false)]),
            space([full(true), full(false)], [full(false), full(false)]),
        ),
        (
            "inactive coupled 1",
            space([full(false), full(true)], [vacuum(true), vacuum(false)]),
            space([vacuum(true), vacuum(false)], [full(false), full(false)]),
        ),
    ];
    for (name, lhs, rhs) in &cases {
        let (lhs_data, rhs_data) = (values(lhs, 0.0), values(rhs, 0.5));
        let init = values(&dst, 2.0);
        let layout_rec = probe::record();
        let region = checked_core(dst.structure(), lhs.structure(), rhs.structure());
        // What: canonical staged structures never build per-subblock layouts.
        assert_eq!(layout_rec.get(Event::FusionLayoutCompile), 0, "{name}");
        let general = compile_checked_generic_core_plan_general(
            dst.structure(),
            2,
            lhs.structure(),
            2,
            rhs.structure(),
            2,
        )
        .unwrap();
        let operands = (
            (lhs.structure(), lhs_data.as_slice()),
            (rhs.structure(), rhs_data.as_slice()),
        );
        for (alpha, beta) in [(1.0, 0.0), (0.75, -1.25)] {
            let got = run_core_plan(
                &region,
                dst.structure(),
                &init,
                operands.0,
                operands.1,
                alpha,
                beta,
            );
            let want = run_core_plan(
                &general,
                dst.structure(),
                &init,
                operands.0,
                operands.1,
                alpha,
                beta,
            );
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name} alpha={alpha} beta={beta}"
            );
        }
        let oracle = core_oracle(&dst, operands.0, operands.1, 2);
        let got = run_core_plan(
            &region,
            dst.structure(),
            &init,
            operands.0,
            operands.1,
            1.0,
            0.0,
        );
        for (got, want) in got.iter().zip(&oracle) {
            assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "{name}");
        }
        assert!(
            oracle.iter().any(|v| v.abs() > 0.5),
            "{name}: nonzero oracle"
        );
    }

    // Negative: an operand packed block by block is not a coupled-sector
    // tiling, so the region route declines and the general builder re-bases.
    let (lhs, rhs) = (&cases[0].1, &cases[0].2);
    let packed = Arc::new(
        crate::tests::packed_fixture_structure(
            4,
            (0..lhs.structure().block_count()).map(|index| {
                let block = lhs.structure().block(index).unwrap();
                (block.key().clone(), block.shape().to_vec())
            }),
        )
        .unwrap(),
    );
    let mut packed_data = vec![0.0; lhs.required_len().unwrap()];
    let canonical_data = values(lhs, 0.0);
    for index in 0..packed.block_count() {
        let source = block_matrix(lhs.structure(), index, &canonical_data);
        let block = packed.block(index).unwrap();
        for (mut linear, value) in source.into_iter().enumerate() {
            let mut at = block.offset();
            for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
                at += (linear % extent) * stride;
                linear /= extent;
            }
            packed_data[at] = value;
        }
    }
    let rhs_data = values(rhs, 0.5);
    let init = values(&dst, 2.0);
    let layout_rec = probe::record();
    let fallback = checked_core(dst.structure(), &packed, rhs.structure());
    assert!(layout_rec.get(Event::FusionLayoutCompile) > 0);
    let got = run_core_plan(
        &fallback,
        dst.structure(),
        &init,
        (&packed, packed_data.as_slice()),
        (rhs.structure(), rhs_data.as_slice()),
        1.0,
        0.0,
    );
    let oracle = core_oracle(
        &dst,
        (&packed, packed_data.as_slice()),
        (rhs.structure(), rhs_data.as_slice()),
        2,
    );
    for (got, want) in got.iter().zip(&oracle) {
        assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()));
    }
}
