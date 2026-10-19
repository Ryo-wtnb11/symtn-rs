use super::*;

#[test]
fn tensorcontract_block_spec_new_supports_complex_coefficients() {
    let spec = TensorContractBlockSpec::<num_complex::Complex64>::new(2, 3, 5);
    assert_eq!(spec.coefficient(), num_complex::Complex64::new(1.0, 0.0));
}
use std::sync::Arc;

#[test]
fn tensorcontract_structure_replays_custom_host_storage_without_vec_fixing() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![3, 2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs = TestTensor::from_vec(vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space).unwrap();
    let mut dst = TestTensor::from_vec(vec![1.0_f64; 4], dst_space).unwrap();
    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TensorContractWorkspace::default();

    replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
}

#[test]
fn tensorcontract_default_host_api_accepts_custom_host_storage() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![3, 2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs = TestTensor::from_vec(vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space).unwrap();
    let mut dst = TestTensor::from_vec(vec![1.0_f64; 4], dst_space).unwrap();

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
}

#[test]
fn tensorcontract_structure_precomputes_core_dense_descriptor() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![3, 2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::<f64>::from_vec(vec![1.0; 6], lhs_space).unwrap();
    let rhs = TestTensor::<f64>::from_vec(vec![1.0; 6], rhs_space).unwrap();
    let dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();

    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();

    assert_eq!(structure.dst_rank(), 2);
    assert_eq!(structure.lhs_rank(), 2);
    assert_eq!(structure.rhs_rank(), 2);
    assert_eq!(structure.lhs_contracting_axes(), &[1]);
    assert_eq!(structure.rhs_contracting_axes(), &[0]);
    assert_eq!(structure.output_axes(), &[0, 1]);
    assert_eq!(structure.terms().len(), 1);
    assert_eq!(structure.terms()[0].key(), &BlockKey::trivial());
    assert_eq!(structure.terms()[0].dst_block(), 0);
    assert_eq!(structure.terms()[0].lhs_block(), 0);
    assert_eq!(structure.terms()[0].rhs_block(), 0);
}

#[test]
fn dense_contract_uses_dense_backend_for_matmul_and_alpha_beta() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![3, 2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs =
        TestTensor::<f64>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], rhs_space).unwrap();
    let mut dst = TestTensor::<f64>::filled(1.0, dst_space).unwrap();

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[155.0, 203.0, 209.0, 275.0]);
}

#[test]
fn tensorcontract_dense_route_uses_tensorkit_forward_tie_order() {
    let lhs_space = vec![5, 7, 3, 11];
    let rhs_space = vec![13, 3, 17, 5];
    let dst_space = vec![7, 11, 13, 17];
    let lhs =
        TestTensor::<f64>::from_vec(vec![1.0; lhs_space.iter().product()], lhs_space).unwrap();
    let rhs =
        TestTensor::<f64>::from_vec(vec![1.0; rhs_space.iter().product()], rhs_space).unwrap();
    let dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();

    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[2, 0], &[1, 3]),
    )
    .unwrap();

    assert_eq!(
        structure.dense_route_kind(),
        TensorContractDenseRouteKind::ForwardSortLhsContractingAxes
    );
    assert_eq!(structure.lhs_contracting_axes(), &[2, 0]);
    assert_eq!(structure.rhs_contracting_axes(), &[1, 3]);
    let (lhs_route_axes, rhs_route_axes) = structure.dense_route_contracting_axes();
    assert_eq!(lhs_route_axes, &[0, 2]);
    assert_eq!(rhs_route_axes, &[3, 1]);
}

#[test]
fn tensorcontract_dense_route_sorted_by_rhs_matches_independent_oracle() {
    let lhs_shape = [3usize, 2, 2, 3];
    let lhs_strides = [10usize, 1, 5, 2];
    let rhs_shape = [2usize, 2, 2, 3];
    let rhs_strides = [5usize, 1, 10, 2];
    let dst_shape = [2usize, 2, 2, 3];
    let lhs_structure = dense_block_structure(&lhs_shape, &lhs_strides);
    let rhs_structure = dense_block_structure(&rhs_shape, &rhs_strides);
    let dst_structure = BlockStructure::trivial(&dst_shape).unwrap();
    let lhs_space = [&lhs_shape[..], &[][..]].concat();
    let rhs_space = [&rhs_shape[..], &[][..]].concat();
    let dst_space = [&dst_shape[..], &[][..]].concat();
    let lhs_data = (0..strided_storage_len(&lhs_shape, &lhs_strides))
        .map(|index| 1.0 + index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..strided_storage_len(&rhs_shape, &rhs_strides))
        .map(|index| -3.0 + 0.5 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = (0..dst_shape.iter().product::<usize>())
        .map(|index| 0.25 * index as f64 - 1.0)
        .collect::<Vec<_>>();
    let lhs =
        TestTensor::<f64>::from_vec_with_structure(lhs_data.clone(), lhs_space, lhs_structure)
            .unwrap();
    let rhs =
        TestTensor::<f64>::from_vec_with_structure(rhs_data.clone(), rhs_space, rhs_structure)
            .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(initial_dst.clone(), dst_space, dst_structure)
            .unwrap();
    let axes = TensorContractSpec::new(&[2, 0], &[1, 3], OutputAxisOrder::from_axes(&[2, 0, 3, 1]));
    let alpha = -1.25;
    let beta = 0.5;

    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        axes,
    )
    .unwrap();
    assert_eq!(
        structure.dense_route_kind(),
        TensorContractDenseRouteKind::ForwardSortRhsContractingAxes
    );
    let (lhs_route_axes, rhs_route_axes) = structure.dense_route_contracting_axes();
    assert_eq!(lhs_route_axes, &[2, 0]);
    assert_eq!(rhs_route_axes, &[1, 3]);

    dense_contract_into(&mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    let expected = rank4_contract_oracle(
        &lhs_data,
        &lhs_shape,
        &lhs_strides,
        &rhs_data,
        &rhs_shape,
        &rhs_strides,
        &initial_dst,
        &dst_shape,
        &[2, 0, 3, 1],
        alpha,
        beta,
    );
    for (&actual, &expected) in dst.data().iter().zip(&expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn tensorcontract_dense_route_reverse_ba_matches_independent_oracle() {
    let lhs_shape = [3usize, 2, 4];
    let lhs_strides = [1usize, 12, 3];
    let rhs_shape = [2usize, 5, 4, 7];
    let rhs_strides = [140usize, 1, 35, 5];
    let dst_shape = [3usize, 5, 7];
    let dst_strides = [35usize, 1, 5];
    let lhs_structure = dense_block_structure(&lhs_shape, &lhs_strides);
    let rhs_structure = dense_block_structure(&rhs_shape, &rhs_strides);
    let dst_structure = dense_block_structure(&dst_shape, &dst_strides);
    let lhs_space = [&lhs_shape[..], &[][..]].concat();
    let rhs_space = [&rhs_shape[..], &[][..]].concat();
    let dst_space = [&dst_shape[..], &[][..]].concat();
    let lhs_data = (0..strided_storage_len(&lhs_shape, &lhs_strides))
        .map(|index| 0.25 + index as f64)
        .collect::<Vec<_>>();
    let rhs_data = (0..strided_storage_len(&rhs_shape, &rhs_strides))
        .map(|index| -2.0 + 0.75 * index as f64)
        .collect::<Vec<_>>();
    let initial_dst = (0..strided_storage_len(&dst_shape, &dst_strides))
        .map(|index| 0.125 * index as f64 + 1.0)
        .collect::<Vec<_>>();
    let lhs =
        TestTensor::<f64>::from_vec_with_structure(lhs_data.clone(), lhs_space, lhs_structure)
            .unwrap();
    let rhs =
        TestTensor::<f64>::from_vec_with_structure(rhs_data.clone(), rhs_space, rhs_structure)
            .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(initial_dst.clone(), dst_space, dst_structure)
            .unwrap();
    let axes = TensorContractSpec::new(&[1, 2], &[0, 2], OutputAxisOrder::from_axes(&[0, 1, 2]));
    let alpha = 0.75;
    let beta = -0.25;

    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        axes,
    )
    .unwrap();
    assert_eq!(
        structure.dense_route_kind(),
        TensorContractDenseRouteKind::ReverseSortLhsContractingAxes
    );
    let (lhs_route_axes, rhs_route_axes) = structure.dense_route_contracting_axes();
    assert_eq!(lhs_route_axes, &[1, 2]);
    assert_eq!(rhs_route_axes, &[0, 2]);

    dense_contract_into(&mut dst, &lhs, &rhs, axes, alpha, beta).unwrap();
    let expected = rank3_by_rank4_contract_oracle(
        &lhs_data,
        &lhs_shape,
        &lhs_strides,
        &rhs_data,
        &rhs_shape,
        &rhs_strides,
        &initial_dst,
        &dst_shape,
        &dst_strides,
        &[0, 1, 2],
        alpha,
        beta,
    );
    for (&actual, &expected) in dst.data().iter().zip(&expected) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "actual {actual} expected {expected}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn rank3_by_rank4_contract_oracle(
    lhs_data: &[f64],
    lhs_shape: &[usize; 3],
    lhs_strides: &[usize; 3],
    rhs_data: &[f64],
    rhs_shape: &[usize; 4],
    rhs_strides: &[usize; 4],
    initial_dst: &[f64],
    dst_shape: &[usize; 3],
    dst_strides: &[usize; 3],
    output_axes: &[usize; 3],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    assert_eq!(lhs_shape[1], rhs_shape[0]);
    assert_eq!(lhs_shape[2], rhs_shape[2]);
    let mut out = initial_dst.to_vec();
    for d2 in 0..dst_shape[2] {
        for d1 in 0..dst_shape[1] {
            for d0 in 0..dst_shape[0] {
                let dst_coords = [d0, d1, d2];
                let mut core = [0usize; 3];
                for (dst_axis, &core_axis) in output_axes.iter().enumerate() {
                    core[core_axis] = dst_coords[dst_axis];
                }
                let mut sum = 0.0;
                for k1 in 0..lhs_shape[2] {
                    for k0 in 0..lhs_shape[1] {
                        let lhs_coords = [core[0], k0, k1];
                        let rhs_coords = [k0, core[1], k1, core[2]];
                        sum += lhs_data[strided_offset3(&lhs_coords, lhs_strides)]
                            * rhs_data[strided_offset(&rhs_coords, rhs_strides)];
                    }
                }
                let dst_index = strided_offset3(&dst_coords, dst_strides);
                out[dst_index] = beta * initial_dst[dst_index] + alpha * sum;
            }
        }
    }
    out
}

fn dense_block_structure(shape: &[usize], strides: &[usize]) -> BlockStructure {
    BlockStructure::from_blocks_with_rank(
        shape.len(),
        vec![
            BlockSpec::with_key(BlockKey::trivial(), shape.to_vec(), strides.to_vec(), 0).unwrap(),
        ],
    )
    .unwrap()
}

fn strided_storage_len(shape: &[usize], strides: &[usize]) -> usize {
    1 + shape
        .iter()
        .zip(strides)
        .map(|(&dim, &stride)| dim.saturating_sub(1) * stride)
        .sum::<usize>()
}

#[allow(clippy::too_many_arguments)]
fn rank4_contract_oracle(
    lhs_data: &[f64],
    lhs_shape: &[usize; 4],
    lhs_strides: &[usize; 4],
    rhs_data: &[f64],
    rhs_shape: &[usize; 4],
    rhs_strides: &[usize; 4],
    initial_dst: &[f64],
    dst_shape: &[usize; 4],
    output_axes: &[usize; 4],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    assert_eq!(lhs_shape[2], rhs_shape[1]);
    assert_eq!(lhs_shape[0], rhs_shape[3]);
    let mut out = vec![0.0; initial_dst.len()];
    for d3 in 0..dst_shape[3] {
        for d2 in 0..dst_shape[2] {
            for d1 in 0..dst_shape[1] {
                for d0 in 0..dst_shape[0] {
                    let dst_coords = [d0, d1, d2, d3];
                    let mut core = [0usize; 4];
                    for (dst_axis, &core_axis) in output_axes.iter().enumerate() {
                        core[core_axis] = dst_coords[dst_axis];
                    }
                    let mut sum = 0.0;
                    for c1 in 0..lhs_shape[0] {
                        for c0 in 0..lhs_shape[2] {
                            let lhs_coords = [c1, core[0], c0, core[1]];
                            let rhs_coords = [core[2], c0, core[3], c1];
                            sum += lhs_data[strided_offset(&lhs_coords, lhs_strides)]
                                * rhs_data[strided_offset(&rhs_coords, rhs_strides)];
                        }
                    }
                    let dst_index =
                        (((d3 * dst_shape[2] + d2) * dst_shape[1] + d1) * dst_shape[0]) + d0;
                    out[dst_index] = beta * initial_dst[dst_index] + alpha * sum;
                }
            }
        }
    }
    out
}

fn strided_offset(coords: &[usize; 4], strides: &[usize; 4]) -> usize {
    coords
        .iter()
        .zip(strides)
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

fn strided_offset3(coords: &[usize; 3], strides: &[usize; 3]) -> usize {
    coords
        .iter()
        .zip(strides)
        .map(|(&coord, &stride)| coord * stride)
        .sum()
}

#[derive(Default)]
struct MatmulOnlyDenseExecutor {
    matmul_into_calls: usize,
}

impl DenseExecutor for MatmulOnlyDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("rank-2 matmul boundary does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("rank-2 matmul boundary does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("rank-2 matmul boundary does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("rank-2 fusion-block matmul must call matmul_into directly")
    }

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.matmul_into_calls += 1;
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => panic!("test only covers f64 core fusion block matmul"),
        };
        assert_eq!(lhs.shape(), &[2, 3]);
        assert_eq!(lhs.strides(), &[1, 2]);
        assert_eq!(rhs.shape(), &[3, 2]);
        assert_eq!(rhs.strides(), &[1, 3]);
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.strides(), &[1, 2]);
        output
            .data_mut()
            .copy_from_slice(&[76.0, 100.0, 103.0, 136.0]);
        Ok(())
    }
}

#[test]
fn tensorcontract_backend_rank2_matmul_uses_matmul_boundary_not_descriptor_replay() {
    let mut backend = DenseTreeTransformOperations::new(MatmulOnlyDenseExecutor::default());
    let mut workspace = TensorContractWorkspace::<f64>::default();
    let lhs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut dst = vec![0.0; 4];

    <DenseTreeTransformOperations<MatmulOnlyDenseExecutor> as TensorContractBackend<
        f64,
        f64,
    >>::matmul_rank2_into_raw(&mut backend, &mut workspace, &mut dst, &lhs, &rhs, 2, 3, 2)
    .unwrap();

    assert_eq!(backend.dense().matmul_into_calls, 1);
    assert_eq!(dst, vec![76.0, 100.0, 103.0, 136.0]);
}

#[test]
fn tensorcontract_structure_honors_output_permutation_with_workspace_scatter() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![4, 3];
    let dst_space = vec![4, 2];
    let lhs = TestTensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], lhs_space).unwrap();
    let rhs = TestTensor::<f64>::from_vec((7..=18).map(|value| value as f64).collect(), rhs_space)
        .unwrap();
    let mut dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();
    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::new(&[1], &[1], OutputAxisOrder::from_axes(&[1, 0])),
    )
    .unwrap();
    let mut backend =
        DenseTreeTransformOperations::new(ContractLayoutCheckingDenseExecutor::default());
    let mut workspace = TensorContractWorkspace::default();

    replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(backend.dense().dot_general_into_calls, 1);
    assert_eq!(
        dst.data(),
        &[115.0, 124.0, 133.0, 142.0, 148.0, 160.0, 172.0, 184.0]
    );
    assert_eq!(workspace.output_len(), 8);
}

#[test]
fn tensorcontract_with_conjugation_matches_dense_reference_with_output_permutation() {
    let lhs_space = vec![2, 2];
    let rhs_space = vec![2, 2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::<Complex64>::from_vec(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(4.0, -3.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(8.0, 2.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::filled(Complex64::new(0.0, 0.0), dst_space).unwrap();

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new_with_conjugation(
            &[1],
            &[0],
            OutputAxisOrder::from_axes(&[1, 0]),
            true,
            false,
        ),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -16.0),
            Complex64::new(31.0, -21.0),
            Complex64::new(33.0, 23.0),
            Complex64::new(44.0, 31.0),
        ]
    );

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new_with_conjugation(
            &[1],
            &[0],
            OutputAxisOrder::from_axes(&[1, 0]),
            false,
            true,
        ),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, 16.0),
            Complex64::new(31.0, 21.0),
            Complex64::new(33.0, -23.0),
            Complex64::new(44.0, -31.0),
        ]
    );
}

#[test]
fn tensorcontract_workspace_reuse_overwrites_dense_route_output() {
    let lhs_space = vec![2, 2];
    let rhs_space = vec![2, 2];
    let dst_space = vec![2, 2];
    let lhs_a = TestTensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], lhs_space.clone()).unwrap();
    let rhs_a = TestTensor::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], rhs_space.clone()).unwrap();
    let lhs_b = TestTensor::<f64>::from_vec(vec![2.0, 0.0, 0.0, 3.0], lhs_space).unwrap();
    let rhs_b = TestTensor::<f64>::from_vec(vec![11.0, 13.0, 17.0, 19.0], rhs_space).unwrap();
    let mut dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();
    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs_a.structure(),
        rhs_a.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TensorContractWorkspace::default();

    replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs_a,
        &rhs_a,
        1.0,
        0.0,
    )
    .unwrap();
    replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs_b,
        &rhs_b,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[22.0, 39.0, 34.0, 57.0]);
    assert_eq!(workspace.output_len(), 4);
}

#[test]
fn tensorcontract_workspace_reuse_clears_conjugating_route_output() {
    let lhs_space = vec![2, 2];
    let rhs_space = vec![2, 2];
    let dst_space = vec![2, 2];
    let lhs_a = TestTensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], lhs_space.clone()).unwrap();
    let rhs_a = TestTensor::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0], rhs_space.clone()).unwrap();
    let lhs_b = TestTensor::<f64>::from_vec(vec![2.0, 0.0, 0.0, 3.0], lhs_space).unwrap();
    let rhs_b = TestTensor::<f64>::from_vec(vec![11.0, 13.0, 17.0, 19.0], rhs_space).unwrap();
    let mut dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();
    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs_a.structure(),
        rhs_a.structure(),
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, false),
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TensorContractWorkspace::default();

    replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs_a,
        &rhs_a,
        1.0,
        0.0,
    )
    .unwrap();
    replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs_b,
        &rhs_b,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[22.0, 39.0, 34.0, 57.0]);
    assert_eq!(workspace.output_len(), 4);
}

#[test]
fn tensorcontract_dense_backend_covers_all_gemm_dtypes() {
    assert_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 27.0_f32);
    assert_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 27.0_f64);
    assert_tensorcontract_scalar_dtype(
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(5.0, 0.0),
        Complex32::new(27.0, 0.0),
    );
    assert_tensorcontract_scalar_dtype(
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(5.0, 0.0),
        Complex64::new(27.0, 0.0),
    );
}

#[test]
fn tensorcontract_with_no_contracted_axes_is_the_outer_product() {
    let lhs_space = vec![2];
    let rhs_space = vec![3];
    let dst_space = vec![2, 3];
    let lhs = TestTensor::<f64>::from_vec(vec![2.0, 3.0], lhs_space).unwrap();
    let rhs = TestTensor::<f64>::from_vec(vec![5.0, 7.0, 11.0], rhs_space).unwrap();
    let mut dst = TestTensor::<f64>::filled(1.0, dst_space).unwrap();

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[], &[], OutputAxisOrder::identity()),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[23.0, 33.0, 31.0, 45.0, 47.0, 69.0]);
}

#[test]
fn tensorcontract_outer_product_conjugates_lhs() {
    let lhs_space = vec![2];
    let rhs_space = vec![2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::<Complex64>::from_vec(
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -3.0)],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec(
        vec![Complex64::new(4.0, 2.0), Complex64::new(5.0, -1.0)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::filled(Complex64::new(0.0, 0.0), dst_space).unwrap();

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new_with_conjugation(
            &[],
            &[],
            OutputAxisOrder::identity(),
            true,
            false,
        ),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(6.0, -2.0),
            Complex64::new(2.0, 16.0),
            Complex64::new(4.0, -6.0),
            Complex64::new(13.0, 13.0),
        ]
    );
}

#[test]
fn tensorcontract_weighted_terms_support_all_gemm_dtypes() {
    assert_weighted_tensorcontract_scalar_dtype(2.0_f32, 3.0_f32, 5.0_f32, 21.0_f32);
    assert_weighted_tensorcontract_scalar_dtype(2.0_f64, 3.0_f64, 5.0_f64, 21.0_f64);
    assert_weighted_tensorcontract_scalar_dtype(
        Complex32::new(2.0, 1.0),
        Complex32::new(3.0, -1.0),
        Complex32::new(5.0, 2.0),
        Complex32::new(22.0, 7.0),
    );
    assert_weighted_tensorcontract_scalar_dtype(
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(5.0, 2.0),
        Complex64::new(22.0, 7.0),
    );
}

#[test]
fn tensorcontract_structure_rejects_invalid_axes_at_compile_time() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![3, 2];
    let dst_space = vec![2, 2];
    let lhs = TestTensor::<f64>::filled(1.0, lhs_space).unwrap();
    let rhs = TestTensor::<f64>::filled(1.0, rhs_space).unwrap();
    let dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();

    let duplicate = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1, 1], &[0, 1]),
    )
    .unwrap_err();
    assert_eq!(
        duplicate,
        OperationError::InvalidAxisSet {
            tensor: "lhs",
            axes: vec![1, 1],
            rank: 2,
        }
    );

    let count = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0, 1]),
    )
    .unwrap_err();
    assert_eq!(
        count,
        OperationError::ContractAxisCountMismatch { lhs: 1, rhs: 2 }
    );
}

#[test]
fn tensorcontract_structure_rejects_dimension_and_output_mismatch_at_compile_time() {
    let lhs_space = vec![2, 3];
    let rhs_space = vec![4, 2];
    let dst_space = vec![2, 4];
    let lhs = TestTensor::<f64>::filled(1.0, lhs_space).unwrap();
    let rhs = TestTensor::<f64>::filled(1.0, rhs_space).unwrap();
    let dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();

    let contracted_dim = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[1]),
    )
    .unwrap_err();
    assert_eq!(
        contracted_dim,
        OperationError::ShapeMismatch {
            dst: vec![3],
            src: vec![2],
        }
    );

    let wrong_dst_space = vec![4, 2];
    let wrong_dst = TestTensor::<f64>::filled(0.0, wrong_dst_space).unwrap();
    let valid_rhs_space = vec![3, 2];
    let valid_rhs = TestTensor::<f64>::filled(1.0, valid_rhs_space).unwrap();
    let output = TensorContractStructure::compile_structures(
        wrong_dst.structure(),
        lhs.structure(),
        valid_rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap_err();
    assert_eq!(
        output,
        OperationError::ShapeMismatch {
            dst: vec![4, 2],
            src: vec![2, 2],
        }
    );
}

#[test]
fn tensorcontract_structure_rejects_incompatible_replay_structure_before_dense_execution() {
    let compile_lhs_space = vec![2, 3];
    let compile_rhs_space = vec![3, 2];
    let compile_dst_space = vec![2, 2];
    let compile_lhs = TestTensor::<f64>::filled(1.0, compile_lhs_space).unwrap();
    let compile_rhs = TestTensor::<f64>::filled(1.0, compile_rhs_space).unwrap();
    let compile_dst = TestTensor::<f64>::filled(0.0, compile_dst_space).unwrap();
    let structure = TensorContractStructure::compile_structures(
        compile_dst.structure(),
        compile_lhs.structure(),
        compile_rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();

    let lhs_space = vec![4, 3];
    let rhs_space = vec![3, 2];
    let dst_space = vec![4, 2];
    let lhs = TestTensor::<f64>::filled(1.0, lhs_space).unwrap();
    let rhs = TestTensor::<f64>::filled(1.0, rhs_space).unwrap();
    let mut dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();
    let mut backend = DenseTreeTransformOperations::new(PanicDenseExecutor);
    let mut workspace = TensorContractWorkspace::default();

    let err = replay_contract_structure_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap_err();

    assert_eq!(err, OperationError::StructureMismatch { tensor: "dst" });
}

#[test]
fn tensorcontract_structure_rejects_multiblock_until_block_sparse_enumeration_exists() {
    let dense = BlockStructure::trivial(&[2, 2]).unwrap();
    let multiblock = BlockStructure::packed_column_major(2, [vec![1, 2], vec![1, 2]]).unwrap();

    let err = TensorContractStructure::compile_structures(
        &dense,
        &multiblock,
        &dense,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: "block-sparse contraction enumeration is not implemented yet",
        }
    );
}

#[test]
fn tensorcontract_structure_replays_explicit_block_terms_and_applies_beta_once() {
    let lhs_space = vec![2, 2];
    let rhs_space = vec![2, 2];
    let dst_space = vec![1, 1];
    let lhs_structure = packed_fixture_structure(
        2,
        [
            (BlockKey::opaque([10]), vec![1, 2]),
            (BlockKey::opaque([20]), vec![1, 2]),
        ],
    )
    .unwrap();
    let rhs_structure = packed_fixture_structure(
        2,
        [
            (BlockKey::opaque([30]), vec![2, 1]),
            (BlockKey::opaque([40]), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(BlockKey::opaque([99]), vec![1, 1])]).unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        lhs_space,
        lhs_structure,
    )
    .unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_structure(
        vec![5.0, 6.0, 7.0, 8.0],
        rhs_space,
        rhs_structure,
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![10.0], dst_space, dst_structure).unwrap();
    let structure = TensorContractStructure::compile_structures_with_block_specs(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &[
            TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5),
            TensorContractBlockSpec::with_coefficient(0, 1, 1, 2.0),
        ],
    )
    .unwrap();

    replay_contract_structure_into(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(structure.terms().len(), 2);
    assert_eq!(dst.data(), &[259.0]);
}

fn issue311_contract_structures() -> (BlockStructure, BlockStructure, BlockStructure) {
    let lhs = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::column_major_with_key(BlockKey::ordinal(10), vec![1, 2], 0).unwrap(),
            BlockSpec::column_major_with_key(BlockKey::ordinal(20), vec![1, 2], 2).unwrap(),
        ],
    )
    .unwrap();
    let rhs = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::column_major_with_key(BlockKey::ordinal(30), vec![2, 1], 0).unwrap(),
            BlockSpec::column_major_with_key(BlockKey::ordinal(40), vec![2, 1], 2).unwrap(),
        ],
    )
    .unwrap();
    let dst = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::column_major_with_key(BlockKey::ordinal(50), vec![1, 1], 0).unwrap(),
            BlockSpec::column_major_with_key(BlockKey::ordinal(60), vec![1, 1], 2).unwrap(),
        ],
    )
    .unwrap();
    (dst, lhs, rhs)
}

fn issue311_contract_specs() -> [TensorContractBlockSpec; 2] {
    [
        TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5),
        TensorContractBlockSpec::with_coefficient(0, 1, 1, 2.0),
    ]
}

#[test]
fn prepared_structure_scales_inactive_destination_once_for_each_beta() {
    let (dst_structure, lhs_structure, rhs_structure) = issue311_contract_structures();
    let structure = TensorContractStructure::compile_structures_with_block_specs(
        &dst_structure,
        &lhs_structure,
        &rhs_structure,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &issue311_contract_specs(),
    )
    .unwrap();
    let dst_structure = Arc::new(dst_structure);
    let lhs_structure = Arc::new(lhs_structure);
    let rhs_structure = Arc::new(rhs_structure);
    let lhs = [1.0, 2.0, 3.0, 4.0];
    let rhs = [5.0, 6.0, 7.0, 8.0];

    for beta in [0.0, 0.5, 1.0] {
        for _ in 0..2 {
            let mut dst = [10.0, -777.0, 20.0];
            crate::contract::tensorcontract_structure_with_dense_executor_raw(
                &mut symtn_dense::DefaultDenseExecutor::new(),
                &mut TensorContractWorkspace::default(),
                &structure,
                &dst_structure,
                &lhs_structure,
                &rhs_structure,
                &mut dst,
                &lhs,
                &rhs,
                2.0,
                beta,
            )
            .unwrap();

            assert_eq!(dst, [229.0 + beta * 10.0, -777.0, beta * 20.0]);
        }
    }
}

#[test]
fn inactive_scaling_waits_for_complete_raw_replay_validation() {
    let (dst_structure, lhs_structure, rhs_structure) = issue311_contract_structures();
    let structure = TensorContractStructure::compile_structures_with_block_specs(
        &dst_structure,
        &lhs_structure,
        &rhs_structure,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &issue311_contract_specs(),
    )
    .unwrap();
    let dst_structure = Arc::new(dst_structure);
    let lhs_structure = Arc::new(lhs_structure);
    let rhs_structure = Arc::new(rhs_structure);
    let mut dst = [10.0, -777.0, 20.0];

    let error = crate::contract::tensorcontract_structure_with_dense_executor_raw(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &dst_structure,
        &lhs_structure,
        &rhs_structure,
        &mut dst,
        &[1.0, 2.0, 3.0],
        &[5.0, 6.0, 7.0, 8.0],
        2.0,
        0.5,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::ElementCountMismatch {
            expected: 4,
            actual: 3,
        }
    );
    assert_eq!(dst, [10.0, -777.0, 20.0]);
}

#[test]
fn complex_structure_scales_inactive_destination_without_norm_projection() {
    let (dst_structure, lhs_structure, rhs_structure) = issue311_contract_structures();
    let structure = TensorContractStructure::compile_structures_with_block_specs(
        &dst_structure,
        &lhs_structure,
        &rhs_structure,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &issue311_contract_specs(),
    )
    .unwrap();
    let dst_structure = Arc::new(dst_structure);
    let lhs_structure = Arc::new(lhs_structure);
    let rhs_structure = Arc::new(rhs_structure);
    let lhs = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(0.5, 4.0),
    ];
    let rhs = [
        Complex64::new(2.0, -1.0),
        Complex64::new(1.0, 3.0),
        Complex64::new(-1.0, 2.0),
        Complex64::new(4.0, -0.5),
    ];
    let initial = [
        Complex64::new(10.0, -3.0),
        Complex64::new(-777.0, 99.0),
        Complex64::new(20.0, 5.0),
    ];
    let alpha = Complex64::new(0.75, -0.25);
    let beta = Complex64::new(0.5, 0.25);
    let mut dst = initial;

    crate::contract::tensorcontract_structure_with_dense_executor_raw(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &dst_structure,
        &lhs_structure,
        &rhs_structure,
        &mut dst,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();

    let expected_active = beta * initial[0]
        + alpha * Complex64::new(0.5, 0.0) * (lhs[0] * rhs[0] + lhs[1] * rhs[1])
        + alpha * Complex64::new(2.0, 0.0) * (lhs[2] * rhs[2] + lhs[3] * rhs[3]);
    let expected_inactive = beta * initial[2];
    assert!((dst[0].re - expected_active.re).abs() < 1.0e-12);
    assert!((dst[0].im - expected_active.im).abs() < 1.0e-12);
    assert_eq!(dst[1], initial[1]);
    assert!((dst[2].re - expected_inactive.re).abs() < 1.0e-12);
    assert!((dst[2].im - expected_inactive.im).abs() < 1.0e-12);
}

fn contract_alias_fixture(
    dst_structure: &BlockStructure,
    terms: &[TensorContractBlockSpec],
) -> Result<TensorContractStructure, OperationError> {
    let lhs = BlockStructure::packed_column_major(2, [vec![1, 1], vec![1, 1]]).unwrap();
    let rhs = BlockStructure::packed_column_major(2, [vec![1, 1], vec![1, 1]]).unwrap();
    TensorContractStructure::compile_structures_with_block_specs(
        dst_structure,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        terms,
    )
}

#[test]
fn tensorcontract_structure_rejects_all_destination_alias_classes() {
    let block = |key, offset| {
        BlockSpec::with_key(BlockKey::ordinal(key), vec![1, 1], vec![1, 1], offset).unwrap()
    };
    let active_inactive =
        BlockStructure::from_blocks_with_rank(2, vec![block(0, 0), block(1, 0)]).unwrap();
    let inactive_inactive =
        BlockStructure::from_blocks_with_rank(2, vec![block(0, 0), block(1, 1), block(2, 1)])
            .unwrap();
    let active_active =
        BlockStructure::from_blocks_with_rank(2, vec![block(0, 0), block(1, 0)]).unwrap();
    let cases = [
        (active_inactive, vec![TensorContractBlockSpec::new(0, 0, 0)]),
        (
            inactive_inactive,
            vec![TensorContractBlockSpec::new(0, 0, 0)],
        ),
        (
            active_active,
            vec![
                TensorContractBlockSpec::new(0, 0, 0),
                TensorContractBlockSpec::new(1, 1, 1),
            ],
        ),
    ];

    for (dst, terms) in cases {
        assert_eq!(
            contract_alias_fixture(&dst, &terms).unwrap_err(),
            OperationError::InvalidArgument {
                message: "tensor contract destination layouts overlap",
            }
        );
    }
}

#[test]
fn tensorcontract_structure_rejects_zero_stride_self_overlap() {
    let dst = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![1, 1], vec![1, 1], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 1], vec![0, 1], 1).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(
        contract_alias_fixture(&dst, &[TensorContractBlockSpec::new(0, 0, 0)]).unwrap_err(),
        OperationError::InvalidArgument {
            message: "tensor contract destination layouts overlap",
        }
    );
}

#[test]
fn tensorcontract_structure_accepts_interleaved_disjoint_destinations() {
    let lhs = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::column_major_with_key(BlockKey::ordinal(0), vec![2, 1], 0).unwrap()],
    )
    .unwrap();
    let rhs = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::column_major_with_key(BlockKey::ordinal(0), vec![1, 1], 0).unwrap()],
    )
    .unwrap();
    let dst = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 1], vec![2, 1], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 1], vec![2, 1], 1).unwrap(),
        ],
    )
    .unwrap();
    let structure = TensorContractStructure::compile_structures_with_block_specs(
        &dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &[TensorContractBlockSpec::new(0, 0, 0)],
    )
    .unwrap();
    let dst = Arc::new(dst);
    let lhs = Arc::new(lhs);
    let rhs = Arc::new(rhs);
    let mut data = [7.0, 11.0, 13.0, 17.0];

    crate::contract::tensorcontract_structure_with_dense_executor_raw(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &dst,
        &lhs,
        &rhs,
        &mut data,
        &[2.0, 3.0],
        &[5.0],
        1.0,
        0.5,
    )
    .unwrap();

    assert_eq!(data, [13.5, 5.5, 21.5, 8.5]);
}

#[test]
fn tensorcontract_structure_rejects_invalid_explicit_block_term_at_compile_time() {
    let dense = BlockStructure::trivial(&[1, 1]).unwrap();

    let err = TensorContractStructure::compile_structures_with_block_specs(
        &dense,
        &dense,
        &dense,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &[TensorContractBlockSpec::new(0, 1, 0)],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::BlockIndexOutOfBounds {
            tensor: "lhs",
            index: 1,
            count: 1,
        }
    );
}
