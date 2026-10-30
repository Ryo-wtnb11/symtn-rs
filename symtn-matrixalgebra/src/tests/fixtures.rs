use super::*;

/// Eager blockwise adjoint of an unbound test operand through the bound
/// dynamic entry.
pub(super) fn adjoint<R, D>(
    rule: &R,
    tensor: &TestTensor<D>,
) -> Result<TestTensor<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + Clone,
    D: FactorScalar,
{
    let bound = BoundTensorMap::try_new(Arc::new(rule.clone()), tensor.clone())?;
    let (space, data) = symtn_tensors::adjoint_bound_dyn(bound.space(), bound.data())?;
    TestTensor::from_dyn(data, space.space().clone())
        .map_err(OperationError::from_core_preserving_context)
}

pub(super) fn bound_tensor<R, D>(provider: Arc<R>, tensor: &TestTensor<D>) -> BoundTensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Clone,
{
    BoundTensorMap::try_new(provider, tensor.clone()).unwrap()
}

pub(super) fn assert_svd_blocks_match(lhs: &TestTensor<f64>, rhs: &TestTensor<f64>) {
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    assert_eq!(lhs_structure.block_count(), rhs_structure.block_count());
    for index in 0..lhs_structure.block_count() {
        let lhs_block = lhs_structure.block(index).unwrap();
        let rhs_block = rhs_structure.block(index).unwrap();
        assert_eq!(lhs_block.key(), rhs_block.key());
        assert_eq!(lhs_block.shape(), rhs_block.shape());
        let shape = lhs_block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let lhs_position = lhs_block.offset()
                + multi_index
                    .iter()
                    .zip(lhs_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let rhs_position = rhs_block.offset()
                + multi_index
                    .iter()
                    .zip(rhs_block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let lhs_value = lhs.data()[lhs_position];
            let rhs_value = rhs.data()[rhs_position];
            assert!(
                (lhs_value - rhs_value).abs() < 1e-10,
                "block {index} element {multi_index:?}: {lhs_value} != {rhs_value}"
            );
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
}

pub(super) fn assert_factor_layout_matches_per_tree_shapes<R>(
    actual: &BoundDynamicFusionMapSpace<R>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    // What: canonical factor construction has the block layout of the
    // ordinary final-HomSpace root, with every tree's shape equal to an
    // independent per-tree computation from the final legs.
    let provider = Arc::clone(actual.provider_arc());
    let homspace = actual.space().homspace().clone();
    let shapes = homspace
        .fusion_tree_keys(provider.as_ref())
        .iter()
        .map(|key| {
            let shape = homspace
                .codomain()
                .legs()
                .iter()
                .zip(key.codomain_tree().uncoupled())
                .chain(
                    homspace
                        .domain()
                        .legs()
                        .iter()
                        .zip(key.domain_tree().uncoupled()),
                )
                .map(|(leg, &sector)| {
                    leg.degeneracy(sector)
                        .expect("factor tree sector must belong to its final leg")
                })
                .collect::<Vec<_>>();
            (BlockKey::FusionTree(key.clone()), shape)
        })
        .collect::<Vec<_>>();
    let canonical =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(provider, homspace)
            .unwrap();
    let actual_space = actual.space();
    let canonical_space = canonical.space();
    assert_eq!(actual_space.nout(), canonical_space.nout());
    assert_eq!(actual_space.nin(), canonical_space.nin());
    assert_eq!(
        actual_space.required_len().unwrap(),
        canonical_space.required_len().unwrap()
    );
    assert_eq!(actual_space.structure().block_count(), shapes.len());
    assert_eq!(
        actual_space.structure().block_count(),
        canonical_space.structure().block_count()
    );
    for index in 0..actual_space.structure().block_count() {
        let actual_block = actual_space.structure().block(index).unwrap();
        let canonical_block = canonical_space.structure().block(index).unwrap();
        let (_, expected_shape) = shapes
            .iter()
            .find(|(key, _)| key == actual_block.key())
            .expect("factor block key must be a fusion tree of its final HomSpace");
        assert_eq!(actual_block.shape(), expected_shape.as_slice());
        assert_eq!(actual_block.key(), canonical_block.key());
        assert_eq!(actual_block.shape(), canonical_block.shape());
        assert_eq!(actual_block.strides(), canonical_block.strides());
        assert_eq!(actual_block.offset(), canonical_block.offset());
    }
}

pub(super) fn assert_compact_factors_reconstruct_input<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
    left: &BoundDynFactor<R, D>,
    diagonal: Option<&BoundDynFactor<R, D>>,
    right: &BoundDynFactor<R, D>,
) where
    D: FactorScalar,
{
    let source_structure = input.space().space().structure();
    let left_structure = left.space().space().structure();
    let right_structure = right.space().space().structure();
    for source_index in 0..source_structure.block_count() {
        let source = source_structure.block(source_index).unwrap();
        let BlockKey::FusionTree(source_key) = source.key() else {
            panic!("factorization fixture must use fusion-tree blocks")
        };
        let left_block = (0..left_structure.block_count())
            .map(|index| left_structure.block(index).unwrap())
            .find(|block| {
                block
                    .key()
                    .as_fusion_tree_pair()
                    .is_some_and(|key| key.codomain_tree() == source_key.codomain_tree())
            })
            .unwrap();
        let right_block = (0..right_structure.block_count())
            .map(|index| right_structure.block(index).unwrap())
            .find(|block| {
                block
                    .key()
                    .as_fusion_tree_pair()
                    .is_some_and(|key| key.domain_tree() == source_key.domain_tree())
            })
            .unwrap();
        let rows = source.shape()[..input.space().space().nout()]
            .iter()
            .product::<usize>();
        let cols = source.shape()[input.space().space().nout()..]
            .iter()
            .product::<usize>();
        let rank = left_block.shape()[left_block.shape().len() - 1];
        assert_eq!(right_block.shape()[0], rank);
        let diagonal_block = diagonal.map(|factor| {
            let structure = factor.space().space().structure();
            (0..structure.block_count())
                .map(|index| structure.block(index).unwrap())
                .find(|block| {
                    block
                        .key()
                        .as_fusion_tree_pair()
                        .is_some_and(|key| key.coupled() == source_key.coupled())
                })
                .unwrap()
        });
        for column in 0..cols {
            for row in 0..rows {
                let mut reconstructed = D::zero();
                for bond in 0..rank {
                    let (left_offset, _) = flattened_block_value(
                        left.data(),
                        left_block,
                        0..left_block.shape().len() - 1,
                        row,
                    );
                    let left_value = left.data()
                        [left_offset + bond * left_block.strides()[left_block.shape().len() - 1]];
                    let (right_offset, _) = flattened_block_value(
                        right.data(),
                        right_block,
                        1..right_block.shape().len(),
                        column,
                    );
                    let right_value = right.data()[right_offset + bond * right_block.strides()[0]];
                    let scale = diagonal_block.map_or_else(D::one, |block| {
                        diagonal.unwrap().data()
                            [block.offset() + bond * block.strides()[0] + bond * block.strides()[1]]
                    });
                    reconstructed = reconstructed + left_value * scale * right_value;
                }
                let (_, expected) = flattened_block_value(
                    input.data(),
                    source,
                    0..source.shape().len(),
                    row + rows * column,
                );
                assert!(
                    (reconstructed.widen_complex() - expected.widen_complex()).norm() < 1.0e-10
                );
            }
        }
    }
}

pub(super) fn f64_qr_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; rows * cols];
    dense
        .qr(
            engine,
            DenseRead::F64(
                symtn_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
            ),
        )
        .unwrap()
}

pub(super) fn c64_qr_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); rows * cols];
    dense
        .qr(
            engine,
            DenseRead::C64(
                symtn_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
            ),
        )
        .unwrap()
}

pub(super) fn f64_eigh_outputs(order: usize) -> Vec<DenseTensor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; order * order];
    let shape = [order, order];
    let strides = [1, order];
    dense
        .eigh(
            engine,
            DenseRead::F64(symtn_dense::DenseView::new(&data, &shape, &strides, 0).unwrap()),
        )
        .unwrap()
}

pub(super) fn c64_eigh_outputs(order: usize) -> Vec<DenseTensor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); order * order];
    let shape = [order, order];
    let strides = [1, order];
    dense
        .eigh(
            engine,
            DenseRead::C64(symtn_dense::DenseView::new(&data, &shape, &strides, 0).unwrap()),
        )
        .unwrap()
}

pub(super) fn rectangular_svd_tensor(rows: usize, cols: usize) -> TestTensor<f64> {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, cols)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    TestTensor::from_vec_with_fusion_space(
        (0..rows * cols)
            .map(|index| ((index * 11 + 2) % 19) as f64 - 7.0)
            .collect(),
        space,
    )
    .unwrap()
}

pub(super) fn mixed_rectangular_tensor(
    even_shape: (usize, usize),
    odd_shape: (usize, usize),
) -> TestTensor<f64> {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(even, even_shape.0), (odd, odd_shape.0)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(even, even_shape.1), (odd, odd_shape.1)],
            false,
        )]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    TestTensor::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| ((index * 7 + 3) % 17) as f64 - 6.0)
            .collect(),
        space,
    )
    .unwrap()
}

pub(super) fn transposed_rectangular_tensor(
    tensor: &TestTensor<f64>,
    rows: usize,
    cols: usize,
) -> TestTensor<f64> {
    let mut data = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            data[col + cols * row] = tensor.data()[row + rows * col];
        }
    }
    TestTensor::from_vec_with_fusion_space(
        data,
        rectangular_svd_tensor(cols, rows).fusion_space().clone(),
    )
    .unwrap()
}

pub(super) fn mixed_rectangular_c32_tensor() -> TestTensor<Complex32> {
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 5), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 3), (odd, 4)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    TestTensor::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| {
                Complex32::new(
                    ((index * 7 + 2) % 17) as f32 - 6.0,
                    ((index * 5 + 3) % 13) as f32 * 0.25 - 1.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap()
}

pub(super) fn tsvd_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TestTensor<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap();
    let len = space.required_len().unwrap();
    TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 11 + 5) % 29) as f64 * 0.25 - 3.0)
            .collect(),
        space,
    )
    .unwrap()
}

/// The eager fusion contraction `dst = alpha * contract(lhs, rhs) + beta *
/// dst` of fixtures through the context's bound dynamic entry.
#[allow(clippy::too_many_arguments)]
pub(super) fn context_contract_into<R, D>(
    context: &mut TensorContractFusionExecutionContext<D, R::Key>,
    rule: &R,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + MultiplicityFreeFusionRule
        + TreeTransformRuleCacheKey
        + Clone,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    D: symtn_tensors::DenseRecouplingScalar + symtn_tensors::RecouplingCoefficientAction<f64>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    fn bind<R: MultiplicityFreeFusionRule>(
        provider: &Arc<R>,
        space: &DynamicFusionMapSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError> {
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space.clone(), Arc::clone(provider))
    }
    let provider = Arc::new(rule.clone());
    let dst_space = bind(&provider, dst.fusion_space())?;
    let lhs_space = bind(&provider, lhs.fusion_space())?;
    let rhs_space = bind(&provider, rhs.fusion_space())?;
    assert!(
        !axes.lhs_conjugate() && !axes.rhs_conjugate(),
        "fixture operands are direct"
    );
    let output_axes: Vec<usize> = match axes.output_permutation() {
        OutputAxisOrder::Axes(output) => output.to_vec(),
        OutputAxisOrder::Identity => (0..dst_space.space().rank()).collect(),
    };
    context.tensorcontract_planned_into(
        engine,
        &dst_space,
        dst.data_mut(),
        (FusionOperand::direct(lhs_space.space()), lhs.data()),
        (FusionOperand::direct(rhs_space.space()), rhs.data()),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        &output_axes,
        alpha,
        symtn_tensors::ContractDestinationInit::Axpby(beta),
    )
}

pub(super) fn contract_pair<R>(
    rule: &R,
    template: &TestTensor<f64>,
    left: &TestTensor<f64>,
    right: &TestTensor<f64>,
) -> TestTensor<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Clone,
{
    let mut reconstructed = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; template.data().len()],
        template.fusion_space().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    context_contract_into(
        &mut context,
        rule,
        &mut reconstructed,
        left,
        right,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();
    reconstructed
}

pub(super) fn hermitian_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TestTensor<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap();
    // Symmetric under swapping the (codomain tree, row indices) and
    // (domain tree, column indices) labels, so every coupled sector matrix is
    // symmetric (real Hermitian).
    let side_label = |tree: &FusionTreeKey, indices: &[usize]| -> u64 {
        let mut label = 17u64;
        for &sector in tree.uncoupled() {
            label = label.wrapping_mul(31).wrapping_add(sector.id() as u64 + 1);
        }
        for &index in indices {
            label = label.wrapping_mul(37).wrapping_add(index as u64 + 1);
        }
        label
    };
    TestTensor::<f64>::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        let row = side_label(tree.codomain_tree(), &indices[..2]);
        let col = side_label(tree.domain_tree(), &indices[2..]);
        let (low, high) = if row <= col { (row, col) } else { (col, row) };
        let hash = low
            .wrapping_mul(6364136223846793005)
            .wrapping_add(high.wrapping_mul(1442695040888963407));
        ((hash >> 33) % 19) as f64 * 0.5 - 4.0
    })
    .unwrap()
}

pub(super) fn assert_eigen_equation<R>(
    rule: &R,
    tensor: &TestTensor<f64>,
    v: &TestTensor<f64>,
    d: &TestTensor<f64>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Clone,
{
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    // t . V
    let mut tv = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; v.data().len()],
        v.fusion_space().clone(),
    )
    .unwrap();
    context_contract_into(
        &mut context,
        rule,
        &mut tv,
        tensor,
        v,
        TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2])),
        1.0,
        0.0,
    )
    .unwrap();
    // V . D
    let mut vd = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; v.data().len()],
        v.fusion_space().clone(),
    )
    .unwrap();
    context_contract_into(
        &mut context,
        rule,
        &mut vd,
        v,
        d,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
        1.0,
        0.0,
    )
    .unwrap();

    for (index, (lhs, rhs)) in tv.data().iter().zip(vd.data()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "eigen equation violated at raw position {index}: {lhs} != {rhs}"
        );
    }
}

pub(super) fn dense_sector_matrices(
    tensor_nout: usize,
    t: &TestTensor<f64>,
) -> Vec<(SectorId, usize, usize, Vec<f64>)> {
    // Matricize per coupled sector (rows = codomain trees x degeneracy,
    // cols = domain trees x degeneracy) for dense checks in tests.
    struct SectorAccumulator {
        sector: SectorId,
        rows: usize,
        cols: usize,
        row_trees: Vec<(FusionTreeKey, usize)>,
        col_trees: Vec<(FusionTreeKey, usize)>,
        entries: Vec<(usize, usize, f64)>,
    }
    let structure = std::sync::Arc::clone(t.structure());
    let mut sectors: Vec<SectorAccumulator> = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key.codomain_tree().coupled();
        let entry = match sectors.iter_mut().find(|entry| entry.sector == sector) {
            Some(entry) => entry,
            None => {
                sectors.push(SectorAccumulator {
                    sector,
                    rows: 0,
                    cols: 0,
                    row_trees: Vec::new(),
                    col_trees: Vec::new(),
                    entries: Vec::new(),
                });
                sectors.last_mut().unwrap()
            }
        };
        let shape = block.shape().to_vec();
        let row_dim: usize = shape[..tensor_nout].iter().product();
        let col_dim: usize = shape[tensor_nout..].iter().product();
        let row_offset = match entry
            .row_trees
            .iter()
            .find(|(tree, _)| tree == key.codomain_tree())
        {
            Some((_, offset)) => *offset,
            None => {
                let offset = entry.rows;
                entry.row_trees.push((key.codomain_tree().clone(), offset));
                entry.rows += row_dim;
                offset
            }
        };
        let col_offset = match entry
            .col_trees
            .iter()
            .find(|(tree, _)| tree == key.domain_tree())
        {
            Some((_, offset)) => *offset,
            None => {
                let offset = entry.cols;
                entry.col_trees.push((key.domain_tree().clone(), offset));
                entry.cols += col_dim;
                offset
            }
        };
        let strides = block.strides().to_vec();
        let offset = block.offset();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..shape.iter().product::<usize>() {
            let position = offset
                + indices
                    .iter()
                    .zip(&strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            let mut row = 0;
            let mut stride = 1;
            for axis in 0..tensor_nout {
                row += indices[axis] * stride;
                stride *= shape[axis];
            }
            let mut col = 0;
            let mut col_stride = 1;
            for axis in tensor_nout..shape.len() {
                col += indices[axis] * col_stride;
                col_stride *= shape[axis];
            }
            entry
                .entries
                .push((row_offset + row, col_offset + col, t.data()[position]));
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    sectors
        .into_iter()
        .map(|entry| {
            let mut matrix = vec![0.0; entry.rows * entry.cols];
            for (row, col, value) in entry.entries {
                matrix[row + entry.rows * col] = value;
            }
            (entry.sector, entry.rows, entry.cols, matrix)
        })
        .collect()
}

pub(super) fn assert_orthonormal_columns(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for left in 0..*cols {
            for right in 0..*cols {
                let mut dot = 0.0;
                for row in 0..*rows {
                    dot += matrix[row + rows * left] * matrix[row + rows * right];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-9,
                    "sector {sector:?}: column dot ({left},{right}) = {dot}"
                );
            }
        }
    }
}

pub(super) fn one_sector_matrix<D: Clone>(data: Vec<D>) -> TestTensor<D> {
    one_sector_rectangular_matrix(data, 2, 2)
}

pub(super) fn assert_eigh_preflight<D: FactorScalar + std::fmt::Debug>(
    tensor: &TestTensor<D>,
    accepted: bool,
) {
    assert_eigh_preflight_at(tensor, accepted, HermitianTol::DEFAULT);
}

pub(super) fn assert_eigh_preflight_at<D: FactorScalar + std::fmt::Debug>(
    tensor: &TestTensor<D>,
    accepted: bool,
    hermitian_tol: HermitianTol,
) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();
    let error = eigh_full_dyn(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), tensor).dynamic(),
        hermitian_tol,
    )
    .map(drop)
    .unwrap_err();

    if accepted {
        assert!(matches!(error, OperationError::Dense(_)));
        assert_eq!(dense.counts().of(EIGH_ENTRIES), 1);
    } else {
        assert!(
            matches!(error, OperationError::InvalidArgument { message } if message.contains("Hermitian"))
        );
        assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
    }
}

pub(super) fn one_sector_rectangular_matrix<D: Clone>(
    data: Vec<D>,
    rows: usize,
    cols: usize,
) -> TestTensor<D> {
    let rule = Z2FusionRule;
    let codomain = SectorLeg::new([(SectorId::new(0), rows)], false);
    let domain = SectorLeg::new([(SectorId::new(0), cols)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    TestTensor::from_vec_with_fusion_space(data, space).unwrap()
}

/// Path agreement of two spectra: sectors and kept counts exact, values under
/// the workspace rule with the sector's value count as `terms`.
pub(super) fn assert_spectra_agree(what: &str, lhs: &[SectorSpectrum], rhs: &[SectorSpectrum]) {
    assert_eq!(lhs.len(), rhs.len(), "{what}");
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector, "{what}");
        numerics::assert_slices_close(what, &lhs.values, &rhs.values, lhs.values.len());
    }
}

pub(super) fn assert_real_spectra_close(lhs: &[SectorSpectrum], rhs: &[SectorSpectrum]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector);
        assert_eq!(lhs.values.len(), rhs.values.len());
        for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
            assert!((lhs - rhs).abs() <= 1e-10, "{lhs} vs {rhs}");
        }
    }
}

/// `source`'s space over the caller-chosen block layout `structure`.
pub(super) fn relayout<R, D>(
    rule: &R,
    source: &TestTensor<D>,
    structure: BlockStructure,
) -> DynamicFusionMapSpace
where
    R: FusionRule,
{
    DynamicFusionMapSpace::try_from_layout(source.fusion_space().homspace().clone(), structure)
        .unwrap()
        .try_bind_rule(rule)
        .unwrap()
}

pub(super) fn padded_copy<R, D>(rule: &R, source: &TestTensor<D>) -> TestTensor<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let mut offset = 1usize;
    let mut blocks = Vec::with_capacity(source.structure().block_count());
    for index in 0..source.structure().block_count() {
        let block = source.structure().block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let structure =
        BlockStructure::from_blocks_with_rank(source.fusion_space().rank(), blocks).unwrap();
    let padded_space = relayout(rule, source, structure);
    TestTensor::from_block_fn_with_fusion_space(padded_space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        let position = block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>();
        block.data()[position]
    })
    .unwrap()
}

pub(super) fn reversed_complete_grid_copy<R, D>(rule: &R, source: &TestTensor<D>) -> TestTensor<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let mut offset = 0usize;
    let mut blocks = Vec::with_capacity(source.structure().block_count());
    for index in (0..source.structure().block_count()).rev() {
        let block = source.structure().block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>();
    }
    let structure =
        BlockStructure::from_blocks_with_rank(source.fusion_space().rank(), blocks).unwrap();
    let reordered_space = relayout(rule, source, structure);
    TestTensor::from_block_fn_with_fusion_space(reordered_space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        let position = block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>();
        block.data()[position]
    })
    .unwrap()
}

pub(super) fn reversed_coupled_tree_basis_copy<R, D>(
    rule: &R,
    source: &TestTensor<D>,
) -> TestTensor<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let blocks = (0..source.structure().block_count())
        .rev()
        .map(|index| {
            let block = source.structure().block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!("source has fusion-tree blocks")
            };
            (key.clone(), block.shape().to_vec())
        })
        .collect();
    let source_space = source.fusion_space();
    let structure = BlockStructure::coupled_sector_matrix_with_keys(
        rule,
        source_space.nout(),
        source_space.rank(),
        blocks,
    )
    .unwrap();
    let space = relayout(rule, source, structure);
    TestTensor::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        block.data()[block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>()]
    })
    .unwrap()
}

pub(super) fn assert_identity_matrices(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    assert!(!matrices.is_empty());
    for (sector, rows, cols, matrix) in matrices {
        assert_eq!(rows, cols, "identity block must be square in {sector:?}");
        for col in 0..*cols {
            for row in 0..*rows {
                let expected = if row == col { 1.0 } else { 0.0 };
                let value = matrix[row + rows * col];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "sector {sector:?} ({row},{col}): {value}"
                );
            }
        }
    }
}

pub(super) fn default_context() -> TensorContractFusionExecutionContext<f64, RuleIdentity> {
    TensorContractFusionExecutionContext::<f64, RuleIdentity>::default()
}

pub(super) fn u1_cross_space_map<D: FactorScalar>(
    codomain: &[(i32, usize)],
    domain: &[(i32, usize)],
) -> TestTensor<D> {
    let codomain_leg = SectorLeg::new(
        codomain
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
        false,
    );
    let domain_leg = SectorLeg::new(
        domain
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain_leg.clone()]),
        FusionProductSpace::new([domain_leg.clone()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&U1FusionRule, homspace).unwrap();
    TestTensor::from_block_fn_with_fusion_space(space, D::zero(), |_, indices| {
        if indices[0] == indices[1] {
            D::one()
        } else {
            D::zero()
        }
    })
    .unwrap()
}

pub(super) fn u1_block_endomorphism<D>(blocks: &[(i32, usize, Vec<D>)]) -> TestTensor<D>
where
    D: Copy + Zero,
{
    let blocks = blocks
        .iter()
        .map(|(charge, dimension, data)| {
            (U1Irrep::new(*charge).sector_id(), *dimension, data.clone())
        })
        .collect::<Vec<_>>();
    block_endomorphism(&U1FusionRule, &blocks)
}

/// `1 <- 1` endomorphism with one fusion tree per coupled sector, on any rule.
pub(super) fn block_endomorphism<R, D>(
    rule: &R,
    blocks: &[(SectorId, usize, Vec<D>)],
) -> TestTensor<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Copy + Zero,
{
    for (_, dimension, data) in blocks {
        assert_eq!(data.len(), dimension * dimension);
    }
    let sectors = blocks
        .iter()
        .map(|(sector, dimension, _)| (*sector, *dimension))
        .collect::<Vec<_>>();
    let leg = SectorLeg::new(sectors.iter().copied(), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap();
    TestTensor::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let coupled = tree.codomain_tree().coupled();
        let (_, dimension, data) = blocks
            .iter()
            .find(|(sector, _, _)| *sector == coupled)
            .unwrap();
        data[indices[0] + dimension * indices[1]]
    })
    .unwrap()
}

pub(super) fn scalar_u1_block<D: Copy>(tensor: &TestTensor<D>, charge: i32) -> D {
    scalar_block(tensor, U1Irrep::new(charge).sector_id())
}

pub(super) fn scalar_block<D: Copy>(tensor: &TestTensor<D>, sector: SectorId) -> D {
    let structure = tensor.structure();
    let block = (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| {
            let BlockKey::FusionTree(key) = block.key() else {
                return false;
            };
            key.codomain_tree().coupled() == sector
        })
        .unwrap();
    assert_eq!(block.shape(), &[1, 1]);
    tensor.data()[block.offset()]
}

pub(super) fn assert_identity_sector_matrices(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        assert_eq!(rows, cols, "sector {sector:?}: expected square factor");
        for col in 0..*cols {
            for row in 0..*rows {
                let expected = if row == col { 1.0 } else { 0.0 };
                let value = matrix[row + rows * col];
                assert!(
                    (value - expected).abs() < 1e-9,
                    "sector {sector:?}: entry ({row},{col}) = {value}"
                );
            }
        }
    }
}

/// The `V = U1Space(0=>3, 1=>2)` oracle fill, in 0-based degeneracy indices.
pub(super) fn exp_oracle_fill(charge: i32, row: usize, column: usize, scale: f64) -> f64 {
    scale * (0.5 + 0.25 * row as f64 - 0.75 * column as f64 + 0.125 * charge as f64)
}

fn exp_oracle_fill_imaginary(row: usize, column: usize, scale: f64) -> f64 {
    scale * (0.125 * row as f64 + 0.375 * column as f64 - 0.25)
}

pub(super) fn exp_oracle_block<D: FactorScalar>(charge: i32, order: usize, scale: f64) -> Vec<D> {
    let mut data = vec![D::zero(); order * order];
    for column in 0..order {
        for row in 0..order {
            let real = exp_oracle_fill(charge, row, column, scale);
            let imaginary = if D::epsilon() == f64::EPSILON && size_of::<D>() == size_of::<f64>() {
                0.0
            } else {
                exp_oracle_fill_imaginary(row, column, scale)
            };
            data[row + order * column] = D::from_complex64(Complex64::new(real, imaginary));
        }
    }
    data
}

pub(super) fn exp_oracle_tensor<D: FactorScalar>(scale: f64) -> TestTensor<D> {
    u1_block_endomorphism(&[
        (0, 3, exp_oracle_block::<D>(0, 3, scale)),
        (1, 2, exp_oracle_block::<D>(1, 2, scale)),
    ])
}

/// Copy of `source` whose coupled sectors stack rows by ascending codomain
/// tree and columns by descending domain tree: the same operator, but row `i`
/// and column `i` of a multi-tree sector name different tree states.
pub(super) fn mis_stacked_endomorphism_copy<R, D>(rule: &R, source: &TestTensor<D>) -> TestTensor<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let mut blocks: Vec<(symtn_core::FusionTreePairKey, Vec<usize>)> =
        (0..source.structure().block_count())
            .map(|index| {
                let block = source.structure().block(index).unwrap();
                let BlockKey::FusionTree(key) = block.key() else {
                    unreachable!("source has fusion-tree blocks")
                };
                (key.clone(), block.shape().to_vec())
            })
            .collect();
    blocks.sort_by(|(a, _), (b, _)| {
        a.codomain_tree()
            .cmp(b.codomain_tree())
            .then(b.domain_tree().cmp(a.domain_tree()))
    });
    let structure = BlockStructure::coupled_sector_matrix_with_keys(rule, 2, 4, blocks).unwrap();
    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert!(
        regions
            .iter()
            .any(|region| region.row_trees() != region.col_trees()),
        "the fixture must mis-stack at least one coupled sector"
    );
    let space = relayout(rule, source, structure);
    TestTensor::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let block = source.block_by_key(key).unwrap();
        block.data()[block.offset()
            + indices
                .iter()
                .zip(block.strides())
                .map(|(&index, &stride)| index * stride)
                .sum::<usize>()]
    })
    .unwrap()
}

pub(super) fn assert_stacking_refusal<T: fmt::Debug>(
    result: Result<T, OperationError>,
    operation: &str,
) {
    match result {
        Err(OperationError::UnsupportedTensorContractScope { message })
            if message.starts_with(operation) && message.contains("stacking") => {}
        other => panic!("{operation}: expected a stacking refusal, got {other:?}"),
    }
}
