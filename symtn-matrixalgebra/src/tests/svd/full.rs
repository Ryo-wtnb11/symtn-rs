use super::*;

#[test]
fn full_svd_late_error_preserves_input_and_publishes_no_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the adjoint-oriented full-SVD engine finishes every sector before
    // allocating any returned factor.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = ScriptedExecutor::<FailSecondSvd>::default();

    let buffer_builds_rec = crate::probe::record();
    let result = svd_full_adjoint_dyn(engine, &mut dense, &bound.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    assert_eq!(buffer_builds_rec.factor_buffer_builds(), (0, 0));
}

#[test]
fn full_svd_publishes_owned_factors_without_scatter_in_both_orientations() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the production full SVD (direct and adjoint engines) admits U and
    // Vh layouts that the staged per-sector factors already occupy, so both
    // factors are transferred instead of zero-filled and scattered.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let one_sided_rec = crate::probe::record();
    let direct = svd_full_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    let probe = one_sided_rec.one_sided_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (2, 0)
    );
    assert!(probe.appended_elements > 0);

    let one_sided_rec = crate::probe::record();
    let adjoint = svd_full_adjoint_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    let probe = one_sided_rec.one_sided_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (2, 0)
    );
    assert_eq!(
        direct.u().space().space().required_len().unwrap(),
        adjoint.u().space().space().required_len().unwrap()
    );
}

#[test]
fn full_svd_adjoint_builds_only_the_final_factor_buffers() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = one_sector_rectangular_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], 2, 3);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let buffer_builds_rec = crate::probe::record();
    let output = svd_full_adjoint_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();

    assert_eq!(buffer_builds_rec.factor_buffer_builds(), (1, 1));
    assert_eq!(output.u().space().space().required_len().unwrap(), 9);
    assert_eq!(output.s().space().space().required_len().unwrap(), 6);
    assert_eq!(output.vh().space().space().required_len().unwrap(), 4);
}

#[test]
fn full_svd_adjoint_completion_reconstructs_rectangular_input() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = one_sector_rectangular_matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], 2, 3);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = ScriptedExecutor::<SvdCallSpy>::default();
    let output = svd_full_adjoint_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    assert_eq!(dense.counts().svd, 1);
    let u = output.u().data();
    let s = output.s().data();
    let vh = output.vh().data();
    for col in 0..2 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|middle| {
                    (0..2)
                        .map(|inner| {
                            u[row + 3 * middle] * s[middle + 3 * inner] * vh[inner + 2 * col]
                        })
                        .sum::<f64>()
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[col + 2 * row]).abs() < 1e-10);
        }
    }
}

#[test]
fn svd_full_rejects_a_different_provider_before_dense_execution() {
    let tensor = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);

    let _backend = ScriptedExecutor::new(RejectExecutorCalls);
    let error = match BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor) {
        Ok(_) => panic!("mismatched provider must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn svd_full_gives_square_unitaries_and_reconstructs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let full = svd_full(
        engine,
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let matrices = dense_sector_matrices(2, &full.u);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full U must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    // U . S has U's codomain and S's (column) bond as domain; build its space
    // from the contraction homspace and per-tree shapes.
    let us_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        full.u.fusion_space().homspace(),
        full.s.fusion_space().homspace(),
        &[2],
        &[0],
        &[0, 1, 2],
        2,
    )
    .unwrap();
    let us_space = DynamicFusionMapSpace::from_final_homspace(&rule, us_hom).unwrap();
    let mut us = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; us_space.required_len().unwrap()],
        us_space,
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context_contract_into(
        &mut context,
        &rule,
        &mut us,
        &full.u,
        &full.s,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
        1.0,
        0.0,
    )
    .unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &us, &full.vh);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn svd_full_uses_native_owned_full_svd_without_orthonormal_completion() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = rectangular_svd_tensor(2, 3);
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn(engine, &mut dense, &input.as_ref().dynamic()).unwrap();

    assert_eq!(dense.counts().svd_full, 1);
    assert!(!full.singular_values().is_empty());
}

fn assert_native_full_svd_uses_builtin_owned_dtype<D: FactorScalar>() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let source = mixed_rectangular_c32_tensor();
    let tensor = TestTensor::<D>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| D::from_complex64(Complex64::new(value.re as f64, value.im as f64)))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn(engine, &mut dense, &input.as_ref().dynamic()).unwrap();

    assert_eq!(dense.counts().svd_full, 2);
    assert_eq!(full.singular_values().len(), 2);
    assert!(full
        .singular_values()
        .iter()
        .all(|entry| !entry.values.is_empty()));
}

#[test]
fn native_full_svd_uses_owned_inputs_for_every_builtin_dtype() {
    assert_native_full_svd_uses_builtin_owned_dtype::<f32>();
    assert_native_full_svd_uses_builtin_owned_dtype::<f64>();
    assert_native_full_svd_uses_builtin_owned_dtype::<Complex32>();
    assert_native_full_svd_uses_builtin_owned_dtype::<Complex64>();
}

#[test]
fn native_full_svd_reconstructs_complex_mixed_rectangular_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let source = mixed_rectangular_c32_tensor();
    let tensor = TestTensor::<Complex64>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn(engine, &mut dense, &input.as_ref().dynamic()).unwrap();
    assert_eq!(dense.counts().svd_full, 2);

    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = full
        .u()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let s_regions = full
        .s()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = full
        .vh()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let s_region = s_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let rows = input_region.rows();
        let cols = input_region.cols();
        assert_eq!(u_region.rows(), rows);
        assert_eq!(u_region.cols(), rows);
        assert_eq!(s_region.rows(), rows);
        assert_eq!(s_region.cols(), cols);
        assert_eq!(vh_region.rows(), cols);
        assert_eq!(vh_region.cols(), cols);

        for column in 0..rows {
            for row in 0..rows {
                let gram = (0..rows)
                    .map(|inner| {
                        full.u().data()[u_region.range().start + inner + rows * row].conj()
                            * full.u().data()[u_region.range().start + inner + rows * column]
                    })
                    .sum::<Complex64>();
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::zero()
                };
                assert!((gram - expected).norm() < 1.0e-10);
            }
        }
        for column in 0..cols {
            for row in 0..cols {
                let gram = (0..cols)
                    .map(|inner| {
                        full.vh().data()[vh_region.range().start + row + cols * inner]
                            * full.vh().data()[vh_region.range().start + column + cols * inner]
                                .conj()
                    })
                    .sum::<Complex64>();
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::zero()
                };
                assert!((gram - expected).norm() < 1.0e-10);
            }
        }

        let mut us = vec![Complex64::zero(); rows * cols];
        for col in 0..cols {
            for inner in 0..rows {
                for row in 0..rows {
                    us[row + rows * col] += full.u().data()
                        [u_region.range().start + row + rows * inner]
                        * full.s().data()[s_region.range().start + inner + rows * col];
                }
            }
        }
        for col in 0..cols {
            for row in 0..rows {
                let actual = (0..cols)
                    .map(|inner| {
                        us[row + rows * inner]
                            * full.vh().data()[vh_region.range().start + inner + cols * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((actual - expected).norm() < 1.0e-10);
            }
        }
    }
}

/// The single coupled sector of a one-sector fixture, widened to `Complex64`
/// so one oracle serves every supported dtype.
fn single_sector_block<R, D>(factor: &BoundDynFactor<R, D>) -> (usize, usize, Vec<Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let space = factor.space().space();
    let regions = space
        .structure()
        .coupled_sector_regions(space.nout())
        .unwrap()
        .unwrap();
    let mut regions = regions.iter();
    let region = regions.next().expect("the oracle fixture has one sector");
    assert!(
        regions.next().is_none(),
        "the oracle fixture has one sector"
    );
    (
        region.rows(),
        region.cols(),
        factor.data()[region.range()]
            .iter()
            .copied()
            .map(FactorScalar::widen_complex)
            .collect(),
    )
}

/// Independent full-SVD oracle: reconstruction, unitarity of both square
/// factors, and the Frobenius identity for the spectrum. Never factor entries
/// — the basis spanning the null space is not unique, so it differs between
/// the direct provider call and the `[U1 | I]` completion fallback.
fn assert_native_full_svd_oracle<D: FactorScalar>(tol: f64) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for (rows, cols) in [(3usize, 3usize), (4, 2), (2, 4)] {
        for kind in ["generic", "rank-deficient", "repeated"] {
            let entry = |index: usize| {
                let (row, col) = (index % rows, index / rows);
                match kind {
                    // every column is a fixed multiple of the first: rank 1.
                    "rank-deficient" => {
                        let value = (((row * 7 + 3) % 13) as f64 - 6.0) * (col as f64 + 1.0);
                        Complex64::new(value, 0.5 * value)
                    }
                    // sigma repeated min(rows, cols) times.
                    "repeated" => {
                        let value = if row == col { 2.0 } else { 0.0 };
                        Complex64::new(value, 0.5 * value)
                    }
                    _ => Complex64::new(
                        ((index * 11 + 2) % 19) as f64 - 7.0,
                        ((index * 5 + 1) % 17) as f64 - 8.0,
                    ),
                }
            };
            let data = (0..rows * cols)
                .map(|index| D::from_complex64(entry(index)))
                .collect::<Vec<_>>();
            let expected = data
                .iter()
                .copied()
                .map(FactorScalar::widen_complex)
                .collect::<Vec<_>>();
            let norm = expected
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt();
            let tensor = one_sector_rectangular_matrix(data, rows, cols);
            let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
            let Some(mut dense) = NativeFullSvdSpy::new() else {
                return;
            };

            let full = svd_full_dyn(engine, &mut dense, &input.as_ref().dynamic()).unwrap();

            let context = format!("{kind} {rows}x{cols}");
            assert_eq!(
                dense.counts().svd_full,
                1,
                "{context}: direct provider call"
            );
            let (u_rows, u_cols, u) = single_sector_block(full.u());
            let (s_rows, s_cols, s) = single_sector_block(full.s());
            let (vh_rows, vh_cols, vh) = single_sector_block(full.vh());
            assert_eq!((u_rows, u_cols), (rows, rows), "{context}: U shape");
            assert_eq!((s_rows, s_cols), (rows, cols), "{context}: S shape");
            assert_eq!((vh_rows, vh_cols), (cols, cols), "{context}: Vh shape");

            for left in 0..rows {
                for right in 0..rows {
                    let gram = (0..rows)
                        .map(|inner| u[inner + rows * left].conj() * u[inner + rows * right])
                        .sum::<Complex64>();
                    let unit = f64::from(u8::from(left == right));
                    assert!(
                        (gram - Complex64::new(unit, 0.0)).norm() <= tol,
                        "{context}: U column dot ({left},{right}) = {gram}"
                    );
                }
            }
            for upper in 0..cols {
                for lower in 0..cols {
                    let gram = (0..cols)
                        .map(|inner| vh[upper + cols * inner] * vh[lower + cols * inner].conj())
                        .sum::<Complex64>();
                    let unit = f64::from(u8::from(upper == lower));
                    assert!(
                        (gram - Complex64::new(unit, 0.0)).norm() <= tol,
                        "{context}: Vh row dot ({upper},{lower}) = {gram}"
                    );
                }
            }

            let rank = rows.min(cols);
            for col in 0..cols {
                for row in 0..rows {
                    let actual = (0..rank)
                        .map(|inner| {
                            u[row + rows * inner] * s[inner + rows * inner] * vh[inner + cols * col]
                        })
                        .sum::<Complex64>();
                    assert!(
                        (actual - expected[row + rows * col]).norm() <= tol * norm,
                        "{context}: reconstruction ({row},{col})"
                    );
                }
            }

            let spectra = full.singular_values();
            assert_eq!(spectra.len(), 1, "{context}: one sector spectrum");
            let values = &spectra[0].values;
            assert_eq!(values.len(), rank, "{context}: spectrum length");
            for (index, value) in values.iter().enumerate() {
                assert!(*value >= 0.0, "{context}: sigma {index} is negative");
                assert!(
                    index == 0 || values[index - 1] >= *value,
                    "{context}: sigma {index} breaks descending order"
                );
                assert!(
                    (s[index + rows * index] - Complex64::new(*value, 0.0)).norm() <= tol * norm,
                    "{context}: S diagonal {index} disagrees with the spectrum"
                );
            }
            // Independent of the factors: the Frobenius norm is the 2-norm of
            // the spectrum, so no singular value may be missing or spurious.
            let spectrum_norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!(
                (spectrum_norm - norm).abs() <= tol * norm.max(1.0),
                "{context}: ||sigma||_2 = {spectrum_norm} but ||A||_F = {norm}"
            );
            match kind {
                "rank-deficient" => {
                    assert!(values[0] > tol, "{context}: the rank-1 sigma vanished");
                    assert!(
                        values[1..].iter().all(|value| *value <= tol * norm),
                        "{context}: a rank-deficient sigma is nonzero"
                    );
                }
                "repeated" => assert!(
                    (values[0] - values[rank - 1]).abs() <= tol * norm,
                    "{context}: repeated singular values disagree"
                ),
                _ => assert!(values[rank - 1] > tol, "{context}: unexpected rank loss"),
            }
        }
    }
}

#[test]
fn native_full_svd_oracle_holds_for_every_builtin_dtype_shape_and_rank() {
    assert_native_full_svd_oracle::<f32>(1.0e-4);
    assert_native_full_svd_oracle::<f64>(1.0e-10);
    assert_native_full_svd_oracle::<Complex32>(1.0e-4);
    assert_native_full_svd_oracle::<Complex64>(1.0e-10);
}

#[test]
fn native_full_svd_late_failure_does_not_publish_or_retry_compatibility() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = mixed_rectangular_c32_tensor();
    let before = tensor.data().to_vec();
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let Some(mut dense) = FailSecondOwnedFullSvd::new() else {
        return;
    };

    let buffer_builds_rec = crate::probe::record();
    let result = svd_full_dyn(engine, &mut dense, &input.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(dense.counts().svd_full, 2);
    assert_eq!(input.data(), before);
    assert_eq!(buffer_builds_rec.factor_buffer_builds(), (0, 0));
}

#[test]
fn native_full_svd_reconstructs_complex_square_and_padded_inputs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let square = one_sector_rectangular_matrix(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(-3.0, 1.0),
            Complex64::new(0.5, -1.5),
            Complex64::new(2.0, 0.25),
        ],
        2,
        2,
    );
    for tensor in [&square, &padded_copy(&Z2FusionRule, &square)] {
        let input = bound_tensor(Arc::new(Z2FusionRule), tensor);
        let Some(mut dense) = NativeFullSvdSpy::new() else {
            return;
        };

        let full = svd_full_dyn(engine, &mut dense, &input.as_ref().dynamic()).unwrap();

        assert_eq!(dense.counts().svd_full, 1);
        assert_compact_factors_reconstruct_input(
            &input.as_ref().dynamic(),
            full.u(),
            Some(full.s()),
            full.vh(),
        );
    }
}

#[test]
fn rectangular_full_svd_has_square_outer_factors_and_reconstructs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: full SVD returns U(m,m), S(m,n), Vh(n,n) and recomposes tall and wide inputs.
    let rule = Z2FusionRule;
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    for (rows, cols) in [(2, 3), (3, 2)] {
        let matrix = one_sector_rectangular_matrix(
            (0..rows * cols)
                .map(|index| ((index * 5 + 1) % 11) as f64 - 4.0)
                .collect(),
            rows,
            cols,
        );
        let input = bound_tensor(Arc::new(rule), &matrix);
        let full = svd_full(engine, &mut dense, &input.as_ref()).unwrap();
        assert_factor_layout_matches_per_tree_shapes(full.u.space());
        assert_factor_layout_matches_per_tree_shapes(full.s.space());
        assert_factor_layout_matches_per_tree_shapes(full.vh.space());
        assert_eq!(full.u.structure().block(0).unwrap().shape(), &[rows, rows]);
        assert_eq!(full.s.structure().block(0).unwrap().shape(), &[rows, cols]);
        assert_eq!(full.vh.structure().block(0).unwrap().shape(), &[cols, cols]);

        let mut us = vec![0.0; rows * cols];
        for col in 0..cols {
            for inner in 0..rows {
                for row in 0..rows {
                    us[row + rows * col] +=
                        full.u.data()[row + rows * inner] * full.s.data()[inner + rows * col];
                }
            }
        }
        let mut reconstructed = vec![0.0; rows * cols];
        for col in 0..cols {
            for inner in 0..cols {
                for row in 0..rows {
                    reconstructed[row + rows * col] +=
                        us[row + rows * inner] * full.vh.data()[inner + cols * col];
                }
            }
        }
        for (actual, expected) in reconstructed.iter().zip(matrix.data()) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
    }
}

#[test]
fn svd_full_gauge_fixes_extra_vh_rows_without_changing_product() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![c(0.0, -2.0), c(0.25, 0.5), c(1.0, -1.0), c(-3.0, 0.0)];
    let mut vh = vec![
        c(1.0, 0.5),
        c(-0.25, 0.75),
        c(1.0, -1.0),
        c(0.5, -0.5),
        c(2.0, 0.0),
        c(-0.5, 0.25),
        c(-1.0, 0.75),
        c(0.0, -1.5),
        c(0.25, 0.0),
    ];
    let sigma = [1.5, 0.7];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 6];
        for col in 0..3 {
            for row in 0..2 {
                for k in 0..2 {
                    out[row + 2 * col] += u[row + 2 * k] * sigma[k] * vh[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_full_gauge(&mut u, 2, 2, &mut vh, 3, 3);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "U pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "U pivot {pivot} negative");
    }
    let extra_pivot = vh[2]; // row 2, col 0 (row + 3 * col)
    assert!(
        extra_pivot.im.abs() < 1e-14,
        "Vh pivot {extra_pivot} not real"
    );
    assert!(extra_pivot.re >= 0.0, "Vh pivot {extra_pivot} negative");
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}
