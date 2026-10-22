use super::*;

fn assert_checked_compact_svd<D>(complex: bool)
where
    D: FactorScalar,
{
    let kept = 2;
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let Svd { u, s, vh } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.counts().svd_vals, 0);
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));

    let expected_spectra = [
        (SectorId::new(0), [4.0, 1.0]),
        (SectorId::new(1), [3.0, 2.0]),
    ];
    let mut residual_squared = 0.0;
    for (sector, expected) in expected_spectra {
        // The checked compact S is dense: read its diagonal.
        let s_block = (0..s.space().space().structure().block_count())
            .map(|index| s.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.codomain_tree().coupled() == sector
                )
            })
            .unwrap();
        let spectrum = (0..s_block.shape()[0])
            .map(|k| {
                s.data()[s_block.offset() + k * (s_block.strides()[0] + s_block.strides()[1])]
                    .widen_complex()
                    .re
            })
            .collect::<Vec<f64>>();
        assert_eq!(spectrum.len(), kept);
        for (&actual, &expected) in spectrum.iter().zip(&expected[..kept]) {
            assert!((actual - expected).abs() < 1.0e-10);
        }

        let u_block = (0..u.space().space().structure().block_count())
            .map(|index| u.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.codomain_tree().coupled() == sector
                )
            })
            .unwrap();
        let vh_block = (0..vh.space().space().structure().block_count())
            .map(|index| vh.space().space().structure().block(index).unwrap())
            .find(|block| {
                matches!(
                    block.key(),
                    BlockKey::FusionTree(key) if key.domain_tree().coupled() == sector
                )
            })
            .unwrap();
        let (rows, cols, matrix) = checked_svd_matrix(sector, complex);
        assert_eq!(u_block.shape(), [rows, kept]);
        assert_eq!(vh_block.shape(), [kept, cols]);

        let u_value = |row: usize, col: usize| {
            u.data()[u_block.offset() + row * u_block.strides()[0] + col * u_block.strides()[1]]
                .widen_complex()
        };
        let vh_value = |row: usize, col: usize| {
            vh.data()[vh_block.offset() + row * vh_block.strides()[0] + col * vh_block.strides()[1]]
                .widen_complex()
        };
        for left in 0..kept {
            for right in 0..kept {
                let u_inner = (0..rows)
                    .map(|row| u_value(row, left).conj() * u_value(row, right))
                    .sum::<Complex64>();
                let vh_inner = (0..cols)
                    .map(|col| vh_value(left, col) * vh_value(right, col).conj())
                    .sum::<Complex64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((u_inner - expected).norm() < 1.0e-10);
                assert!((vh_inner - expected).norm() < 1.0e-10);
            }
        }
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..kept)
                    .map(|bond| u_value(row, bond) * spectrum[bond] * vh_value(bond, col))
                    .sum::<Complex64>();
                residual_squared += if sector == SectorId::new(1) {
                    (1.0 + 2.0_f64.sqrt()) * (reconstructed - matrix[row + rows * col]).norm_sqr()
                } else {
                    (reconstructed - matrix[row + rows * col]).norm_sqr()
                };
            }
        }
    }
    assert!(residual_squared.sqrt() < 1.0e-10);
}

#[test]
fn checked_generic_svd_compact_uses_each_real_compact_decomposition_once() {
    assert_checked_compact_svd::<f64>(false);
}

#[test]
fn checked_generic_svd_compact_uses_each_complex_compact_decomposition_once() {
    assert_checked_compact_svd::<Complex64>(true);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_compact_empty_input_skips_dense_execution() {
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let data = Vec::<f64>::new();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let Svd { u, s, vh } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 0);
    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.counts().svd_vals, 0);
    assert!(s.data().is_empty());
    assert!(u.data().is_empty());
    assert!(vh.data().is_empty());
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));
}

#[test]
fn checked_generic_svd_compact_dense_failure_precedes_output_provider_admission() {
    let (provider, space, data) = checked_svd_truncation_input::<f64>(false);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = ScriptedExecutor::<FailAfterObservingSvdInput>::default();
    let result = svd_compact_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::Dense(DenseError::Backend { op: "svd_into", .. })
        ))
    ));
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(dense.observed.len(), 1);
    assert_eq!(input.data(), before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_keeps_the_provider_and_emits_the_unmatched_coupled_sector() {
    let rule = FactorGenericRule;
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 1), (vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), homspace).unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let checked_provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &checked_provider);
    checked_provider.calls.set(0);
    // A non-bond space is misuse for diagonal storage, refused before any
    // provider query (MAK `@assert m == n && isdiag(A)`).
    assert!(matches!(
        svd_compact_from_source::<CheckedGenericAdmissionMode, _, _, _, f64>(
            &mut symtn_dense::DefaultDenseExecutor::new(),
            FactorSource::Diagonal {
                space: &checked,
                spectrum: &[],
            },
        ),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::SpaceMismatch { message }
        )) if message.contains("one-leg endomorphism")
    ));
    assert_eq!(checked_provider.calls.get(), 0);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 1);
    assert!(Arc::ptr_eq(
        full.u().space().provider_arc(),
        &checked_provider
    ));
    assert!(Arc::ptr_eq(
        full.s().space().provider_arc(),
        &checked_provider
    ));
    assert!(Arc::ptr_eq(
        full.vh().space().provider_arc(),
        &checked_provider
    ));
    let structure = full.u().space().space().structure();
    assert!((0..structure.block_count()).any(|index| {
        matches!(
            structure.block(index).unwrap().key(),
            BlockKey::FusionTree(key) if key.codomain_tree().coupled() == vacuum
        )
    }));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_failure_publishes_no_factors() {
    let (source, data) = generic_factorization_input();
    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 2,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &failing);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let result = svd_full_dyn_checked_generic(&mut dense, &input);
    assert!(matches!(
        result,
        Err(crate::CheckedGenericFactorPlanError::Provider(_))
    ));
}

fn assert_checked_compact_svd_live_stage_owners<D: crate::factorize::FactorScalar>() {
    let (_, space, data) = checked_svd_truncation_input::<D>(true);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = ScriptedExecutor::<RejectSvdInto>::default();
    let svd_stage_rec = crate::probe::record();

    svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let stage = svd_stage_rec.checked_svd_stages();

    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.counts().svd, 2);
    assert_eq!(stage.len(), 2);
    assert_eq!(dense.output_ptrs, stage);
    assert!(stage.iter().all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
}

#[test]
fn checked_generic_compact_svd_keeps_live_stage_owners_for_every_dtype() {
    assert_checked_compact_svd_live_stage_owners::<f64>();
    assert_checked_compact_svd_live_stage_owners::<f32>();
    assert_checked_compact_svd_live_stage_owners::<Complex32>();
    assert_checked_compact_svd_live_stage_owners::<Complex64>();
}

#[test]
fn checked_compact_svd_zero_rank_skips_backend_and_post_gauge_stage() {
    let mut dense = ScriptedExecutor::<RejectSvdInto>::default();
    let svd_stage_rec = crate::probe::record();

    for (rows, cols) in [(0, 3), (3, 0)] {
        assert_eq!(
            crate::factorize::compact_svd_numerical_stage_lengths_for_test(
                &mut dense,
                &[] as &[f64],
                rows,
                cols,
            )
            .unwrap(),
            (0, 0, 0)
        );
    }
    assert_eq!(dense.counts().svd, 0);
    assert_eq!(dense.counts().svd_into, 0);
    assert!(svd_stage_rec.checked_svd_stages().is_empty());
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_completes_unmatched_columns_and_disjoint_space() {
    let rule = FactorGenericRule;
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1), (vacuum, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), homspace).unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: FactorGenericRule.rule_identity(),
        }),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    let structure = full.vh().space().space().structure();
    assert!((0..structure.block_count()).any(|index| {
        matches!(
            structure.block(index).unwrap().key(),
            BlockKey::FusionTree(key) if key.domain_tree().coupled() == vacuum
        )
    }));

    let disjoint = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(rule), disjoint).unwrap();
    let checked = BoundDynamicFusionMapSpace::bind_generic(
        source.space().clone(),
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: FactorGenericRule.rule_identity(),
        }),
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert!(full.singular_values().is_empty());
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn assert_checked_full_svd_builder_failure(fail_at: usize) {
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let result = svd_full_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(crate::CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
            if call == fail_at
    ));
    assert_eq!(input.data(), before);
    assert!(Arc::ptr_eq(input.space().provider_arc(), &provider));
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    assert_eq!(dense.counts().qr, 2);
}

/// The checked full-SVD provider calls of `generic_factorization_input`,
/// each step counted on its own by a fresh probe: the two multiplicity-aware
/// dimension preflights, one U output-layout enumeration and one Vh
/// enumeration. Also returns the calls one full SVD makes, and those of an
/// S enumeration.
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn full_svd_provider_steps() -> ([usize; 3], usize, usize) {
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let output = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();
    let observed = provider.calls.get();
    assert!(Arc::ptr_eq(output.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(output.vh().space().provider_arc(), &provider));

    let probe_calls = |run: &dyn Fn(&LateGenericSpy)| {
        let probe = LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: RuleIdentity::new_unique::<LateGenericSpy>(),
        };
        run(&probe);
        probe.calls.get()
    };
    let homspace = source.space().homspace();
    let preflight = probe_calls(&|probe| {
        coupled_sector_block_dimensions_generic_checked(homspace.codomain(), probe).unwrap();
        coupled_sector_block_dimensions_generic_checked(homspace.domain(), probe).unwrap();
    });
    let enumeration = |factor: &BoundDynFactor<LateGenericSpy, f64>| {
        let homspace = factor.space().space().homspace().clone();
        probe_calls(&|probe| {
            homspace
                .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(probe)
                .unwrap();
        })
    };
    (
        [preflight, enumeration(output.u()), enumeration(output.vh())],
        observed,
        enumeration(output.s()),
    )
}

#[test]
fn checked_generic_full_svd_u_builder_failure_preserves_provider_context() {
    // What: the first and last post-dense checked-provider calls of U-space
    // construction propagate their exact provider error.
    let ([preflight, u, _], ..) = full_svd_provider_steps();
    assert!(u > 0);
    assert_checked_full_svd_builder_failure(preflight + 1);
    assert_checked_full_svd_builder_failure(preflight + u);
}

#[test]
fn checked_generic_full_svd_vh_builder_failure_preserves_provider_context() {
    // What: Vh-space construction propagates its exact provider error without
    // publishing U, from its first call to its last.
    let ([preflight, u, vh], ..) = full_svd_provider_steps();
    assert!(vh > 0);
    assert_checked_full_svd_builder_failure(preflight + u + 1);
    assert_checked_full_svd_builder_failure(preflight + u + vh);
}

#[test]
fn checked_generic_full_svd_enumerates_each_output_layout_once() {
    // What: the provider sees the dimension preflights plus exactly one
    // layout enumeration per one-sided output; the diagonal S publication
    // adds none.
    let (steps, observed, s_enumeration) = full_svd_provider_steps();
    assert_eq!(observed, steps.iter().sum::<usize>(), "{steps:?}");
    assert_eq!(s_enumeration, 0);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_native_full_svd_stages_before_unchanged_provider_admission() {
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.counts().svd_full, 2);
    let (steps, ..) = full_svd_provider_steps();
    assert_eq!(provider.calls.get(), steps.iter().sum::<usize>());
    assert!(Arc::ptr_eq(full.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.s().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.vh().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_full_svd_local_shape_error_precedes_provider_query() {
    // What: checked tensor admission reports the local storage mismatch before
    // any provider-backed factorization work can run.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let error = match BoundDynamicTensorRef::try_new(&checked, &data[..data.len() - 1]) {
        Ok(_) => panic!("short storage must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        OperationError::Core(CoreError::DimensionMismatch { .. })
    ));
    assert_eq!(provider.calls.get(), 0);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_native_full_svd_reconstructs_complex_interleaved_square_trees() {
    let (canonical, _, data) = generic_values_endomorphism_input();
    let (interleaved_space, interleaved_data) =
        interleaved_generic_endomorphism_input(&canonical, &data);
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let space = BoundDynamicFusionMapSpace::bind_generic(
        interleaved_space.space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &interleaved_data).unwrap();
    let Some(mut dense) = NativeFullSvdSpy::new() else {
        return;
    };

    let full = svd_full_dyn_checked_generic(&mut dense, &input).unwrap();

    assert_eq!(dense.counts().svd_full, 2);
    assert_compact_factors_reconstruct_input(&input, full.u(), Some(full.s()), full.vh());
    assert!(Arc::ptr_eq(full.u().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.s().space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(full.vh().space().provider_arc(), &provider));
}

fn assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm<D: crate::factorize::FactorScalar>() {
    let (base, data) = generic_factorization_input();
    let data = data.into_iter().map(D::from_real).collect::<Vec<_>>();
    let (provider, source) = bind_checked_only(&base);
    let input = BoundDynamicTensorRef::try_new(&source, &data).unwrap();
    let output = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            source.space().homspace().codomain().clone(),
        ),
    )
    .unwrap();
    let mut dense = ScriptedExecutor::<RejectSvdInto>::default();

    let result = pinv_direct_into_dyn(&mut dense, &input, output, 0.0).unwrap();

    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.counts().svd, 2);
    assert_eq!(dense.output_ptrs.len(), 2);
    assert_eq!(dense.gemm_ptrs.len(), 2);
    assert_eq!(
        dense.gemm_ptrs,
        dense
            .output_ptrs
            .iter()
            .map(|&(u, vt)| (vt, u))
            .collect::<Vec<_>>(),
    );
    let regions = source
        .space()
        .structure()
        .coupled_sector_regions(source.space().nout())
        .unwrap()
        .unwrap();
    assert_eq!(dense.gemm_views.len(), regions.len());
    for ((lhs, rhs, lhs_conj, rhs_conj), region) in dense.gemm_views.iter().zip(regions.iter()) {
        let rank = region.rows().min(region.cols());
        assert_eq!(lhs.shape, [region.cols(), rank]);
        assert_eq!(lhs.strides, [rank, 1]);
        assert_eq!(lhs.offset, 0);
        assert_eq!(rhs.shape, [rank, region.rows()]);
        assert_eq!(rhs.strides, [region.rows(), 1]);
        assert_eq!(rhs.offset, 0);
        assert!(*lhs_conj && *rhs_conj);
    }
    assert!(std::ptr::eq(
        result.space().provider_arc().as_ref(),
        provider.as_ref()
    ));
}

#[test]
fn checked_pinv_uses_owned_svd_outputs_at_final_gemm() {
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<f32>();
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<f64>();
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<Complex32>();
    assert_checked_pinv_uses_owned_svd_outputs_at_final_gemm::<Complex64>();
}

struct FailSingleLegFold {
    rule: FactorGenericRule,
    fail_at: usize,
    single_leg_folds: Cell<usize>,
    /// Its own identity: it fails where `rule` does not, so no structure
    /// cache may answer for it from `rule`'s entries.
    identity: RuleIdentity,
}

impl SpyTwin for FailSingleLegFold {
    fn twin(&self) -> Self {
        Self {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            single_leg_folds: Cell::new(0),
            identity: self.identity.clone(),
        }
    }
}

impl symtn_core::testing::Provider for FailSingleLegFold {}

impl FusionRule for FailSingleLegFold {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.rule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.rule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.rule.nsymbol(left, right, coupled)
    }
}

impl CheckedGenericFusion for FailSingleLegFold {
    type Error = LateGenericError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(self.rule.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.rule.fusion_channels(left, right))
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }
    fn try_coupled_sector_fold(
        &self,
        effective: &[SectorId],
    ) -> Result<CoupledSectorFold, Self::Error> {
        if effective.len() == 1 {
            let call = self.single_leg_folds.get() + 1;
            self.single_leg_folds.set(call);
            if call == self.fail_at {
                return Err(LateGenericError(call));
            }
        }
        Ok(InfallibleGeneric::new(&self.rule)
            .try_coupled_sector_fold(effective)
            .unwrap())
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for FailSingleLegFold {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.dim_scalar(sector))
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.sqrt_dim_scalar(sector))
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.inv_sqrt_dim_scalar(sector))
    }
    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Ok(self.rule.frobenius_schur_phase_scalar(sector))
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Ok(self.rule.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        coupled: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(self.rule.r_symbol_generic(a, b, coupled))
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_compact_preserves_a_diagonal_s_fold_failure() {
    // What: a provider error while laying out checked compact SVD's dense `S`
    // is returned unchanged as `Provider(e)`, after both dense SVDs and with
    // the input untouched. U and Vh each enumerate their layout once before
    // S, so S's first one-leg fold follows theirs, each counted on its own.
    let (source, data) = generic_factorization_input();
    let first_s_fold = {
        let provider = Arc::new(FailSingleLegFold {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            single_leg_folds: Cell::new(0),
            identity: RuleIdentity::new_unique::<FailSingleLegFold>(),
        });
        let checked = bind_to_spy(&source, &provider);
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let mut dense = ScriptedExecutor::<CountingDense>::default();
        let Svd { u, vh, .. } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        let folds = |homspace: &FusionTreeHomSpace| {
            let probe = FailSingleLegFold {
                rule: FactorGenericRule,
                fail_at: usize::MAX,
                single_leg_folds: Cell::new(0),
                identity: RuleIdentity::new_unique::<FailSingleLegFold>(),
            };
            homspace
                .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&probe)
                .unwrap();
            probe.single_leg_folds.get()
        };
        folds(u.space().space().homspace()) + folds(vh.space().space().homspace()) + 1
    };
    let failing_provider = Arc::new(FailSingleLegFold {
        rule: FactorGenericRule,
        fail_at: first_s_fold,
        single_leg_folds: Cell::new(0),
        identity: RuleIdentity::new_unique::<FailSingleLegFold>(),
    });
    let failing_space = bind_to_spy(&source, &failing_provider);
    let input = BoundDynamicTensorRef::try_new(&failing_space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let result = svd_compact_dyn_checked_generic(&mut dense, &input);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
            if call == first_s_fold
    ));
    assert_eq!(failing_provider.single_leg_folds.get(), first_s_fold);
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.counts().svd_vals, 0);
    assert_eq!(input.data(), before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_svd_compact_enumerates_each_factor_layout_once() {
    // What: checked compact SVD queries the provider for exactly one
    // enumeration of each of the U, S and Vh spaces.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let Svd { u, s, vh } = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let compact_calls = provider.calls.get();
    assert!(s.data().len() > 1);
    assert_eq!(
        compact_calls,
        checked_enumeration_calls(&u)
            + checked_enumeration_calls(&s)
            + checked_enumeration_calls(&vh)
    );
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn compact_diagonal_on_an_expert_bond_layout_is_normalized_not_refused() {
    // What: a compact diagonal whose bond space has an expert (padded) layout
    // is re-derived in canonical layout before the direct route, so its
    // values, eigenvalues and null space equal those on the canonical bond,
    // and its compact QR factors stay on the given expert space.
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let leg = SectorLeg::new([(vacuum, 1), (x, 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let canonical = bind_to_spy(&source, &provider);
    let source_structure = source.space().structure();
    let mut offset = 1usize;
    let mut blocks = Vec::new();
    for index in 0..source_structure.block_count() {
        let block = source_structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(2, blocks).unwrap();
    let layout =
        DynamicFusionMapSpace::try_from_layout(canonical.space().homspace().clone(), structure)
            .unwrap()
            .try_bind_rule(provider.as_ref())
            .unwrap();
    let expert =
        BoundDynamicFusionMapSpace::bind_generic(layout.clone(), Arc::clone(&provider)).unwrap();
    assert!(!matches!(
        expert.space().structure().coupled_sector_regions(1),
        Ok(Some(regions)) if regions.iter().all(|region| region.has_aligned_diagonal())
    ));
    let spectrum = [
        SectorSpectrum {
            sector: vacuum,
            values: vec![-2.0],
        },
        SectorSpectrum {
            sector: x,
            values: vec![0.0, 3.0],
        },
    ];
    let diagonal = |space| FactorSource::Diagonal {
        space,
        spectrum: &spectrum,
    };
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    assert_eq!(
        svd_vals_checked_generic(&mut dense, diagonal(&expert)).unwrap(),
        svd_vals_checked_generic(&mut dense, diagonal(&canonical)).unwrap()
    );
    assert_eq!(
        eig_vals_checked_generic(&mut dense, diagonal(&expert)).unwrap(),
        eig_vals_checked_generic(&mut dense, diagonal(&canonical)).unwrap()
    );
    let expert_null = left_null_checked_generic(&mut dense, diagonal(&expert)).unwrap();
    let canonical_null = left_null_checked_generic(&mut dense, diagonal(&canonical)).unwrap();
    assert_eq!(expert_null.data(), canonical_null.data());
    assert_eq!(
        expert_null.space().space().structure(),
        canonical_null.space().space().structure()
    );
    let Qr { q, r } = qr_compact_checked_generic(&mut dense, diagonal(&expert)).unwrap();
    for factor in [q, r] {
        let FactorOutput::Diagonal { space, .. } = factor else {
            panic!("a compact diagonal's QR factors stay compact");
        };
        assert_eq!(space.space().structure(), expert.space().structure());
    }
    // #1994: the elementwise matrix functions admit the expert layout through
    // the same structural admission (no dense fallback), and a spectrum
    // factor's bond `fuse(V)` is the canonical layout, not the expert one.
    admit_compact_diagonal::<CheckedGenericAdmissionMode, _, _>(&expert, &spectrum).unwrap();
    let bond = |space| {
        spectrum_bond::<CheckedGenericAdmissionMode, _, _>(space, &spectrum)
            .unwrap()
            .space()
            .structure()
            .clone()
    };
    assert_eq!(bond(&expert), bond(&canonical));
    assert_ne!(bond(&expert), *expert.space().structure());
    let mut missing = spectrum.to_vec();
    missing.pop();
    assert!(matches!(
        admit_compact_diagonal::<CheckedGenericAdmissionMode, _, _>(&expert, &missing),
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::InvalidArgument { message }
        )) if message.contains("sector count")
    ));
}

/// `space` re-laid with one padding element before every block: an expert
/// layout of the same hom space whose regions are not aligned diagonals.
fn padded_layout(space: &DynamicFusionMapSpace) -> DynamicFusionMapSpace {
    let structure = space.structure();
    let (mut offset, mut blocks) = (1usize, Vec::new());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>() + 1;
    }
    DynamicFusionMapSpace::try_from_layout(
        space.homspace().clone(),
        BlockStructure::from_blocks_with_rank(2, blocks).unwrap(),
    )
    .unwrap()
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn elementwise_compact_admission_allocates_nothing_and_queries_no_provider() {
    // What (#1994): the admission of inv/pinv/exp/solve is a
    // structural check on the bond's blocks. Warm, it allocates nothing and
    // queries no provider, on the canonical and on an expert layout, in both
    // fusion modes; misuse is still refused.
    use super::super::null_space::counting_alloc;

    let leg = SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 3)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let spectrum = [
        SectorSpectrum {
            sector: SectorId::new(0),
            values: vec![1.0, 2.0],
        },
        SectorSpectrum {
            sector: SectorId::new(1),
            values: vec![3.0, 4.0, 5.0],
        },
    ];
    let mut short = spectrum.to_vec();
    short[1].values.pop();

    let mf = Arc::new(U1FusionRule);
    let mf_canonical = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&mf),
        homspace.clone(),
    )
    .unwrap();
    let mf_expert = BoundDynamicFusionMapSpace::bind_multiplicity_free_checked(
        padded_layout(mf_canonical.space())
            .try_bind_rule(mf.as_ref())
            .unwrap(),
        Arc::clone(&mf),
    )
    .unwrap();

    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked_canonical = bind_to_spy(&source, &provider);
    let checked_expert = BoundDynamicFusionMapSpace::bind_generic(
        padded_layout(checked_canonical.space())
            .try_bind_rule(provider.as_ref())
            .unwrap(),
        Arc::clone(&provider),
    )
    .unwrap();

    let mut report = Vec::new();
    for (name, space) in [("mf canonical", &mf_canonical), ("mf expert", &mf_expert)] {
        let admit = || {
            admit_compact_diagonal::<symtn_core::MultiplicityFreeAdmissionMode, _, _>(
                space, &spectrum,
            )
        };
        admit().unwrap();
        let ((), allocs) = counting_alloc::measure(|| admit().unwrap());
        report.push((name, allocs.calls, allocs.bytes, 0));
        assert!(
            matches!(admit_compact_diagonal::<symtn_core::MultiplicityFreeAdmissionMode, _, _>(space, &short), Err(OperationError::InvalidArgument { message }) if message.contains("length"))
        );
    }
    for (name, space) in [
        ("checked canonical", &checked_canonical),
        ("checked expert", &checked_expert),
    ] {
        let admit =
            || admit_compact_diagonal::<CheckedGenericAdmissionMode, _, _>(space, &spectrum);
        admit().unwrap();
        provider.calls.set(0);
        let ((), allocs) = counting_alloc::measure(|| admit().unwrap());
        report.push((name, allocs.calls, allocs.bytes, provider.calls.get()));
        assert!(matches!(
            admit_compact_diagonal::<CheckedGenericAdmissionMode, _, _>(space, &short),
            Err(CheckedGenericFactorPlanError::Operation(
                OperationError::InvalidArgument { message }
            )) if message.contains("length")
        ));
    }
    eprintln!("admission (calls, bytes, provider queries): {report:?}");
    for (name, calls, bytes, queries) in report {
        assert_eq!((calls, bytes, queries), (0, 0, 0), "{name}");
    }
}

#[test]
fn elementwise_compact_admission_takes_unsorted_spectra_through_the_region_admission() {
    // What (#1994): compact QR/LQ/polar keep the input space and emit
    // their spectra in region order. On an aligned expert layout whose
    // blocks are not in sector order that order is not sector order, so the
    // elementwise admission must accept an unsorted spectrum exactly as the
    // region admission does, and still refuse misuse.
    let mf = Arc::new(U1FusionRule);
    let leg = SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 3)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let canonical = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&mf),
        homspace.clone(),
    )
    .unwrap();
    // The same blocks, contiguous and aligned, stored in reverse sector order.
    let structure = canonical.space().structure();
    let mut offset = 0usize;
    let mut blocks = Vec::new();
    for index in (0..structure.block_count()).rev() {
        let block = structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offset += block.shape().iter().product::<usize>();
    }
    let reversed = BoundDynamicFusionMapSpace::bind_multiplicity_free_checked(
        DynamicFusionMapSpace::try_from_layout(
            homspace,
            BlockStructure::from_blocks_with_rank(2, blocks).unwrap(),
        )
        .unwrap()
        .try_bind_rule(mf.as_ref())
        .unwrap(),
        Arc::clone(&mf),
    )
    .unwrap();
    let regions = reversed
        .space()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    assert!(regions.iter().all(|region| region.has_aligned_diagonal()));

    let sorted = [
        SectorSpectrum {
            sector: SectorId::new(0),
            values: vec![1.0, -2.0],
        },
        SectorSpectrum {
            sector: SectorId::new(1),
            values: vec![3.0, 4.0, -5.0],
        },
    ];
    let unsorted = [sorted[1].clone(), sorted[0].clone()];
    type Mf = symtn_core::MultiplicityFreeAdmissionMode;
    for spectrum in [&sorted[..], &unsorted[..]] {
        admit_compact_diagonal::<Mf, _, _>(&reversed, spectrum).unwrap();
        admit_compact_diagonal::<Mf, _, _>(&canonical, spectrum).unwrap();
    }
    // Region order is what QR emits; on this layout it is the reverse of
    // sector order, and that spectrum is admitted.
    let Qr { q, .. } = qr_compact_from_source::<Mf, _, _, _, _>(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        FactorSource::Diagonal {
            space: &reversed,
            spectrum: &sorted,
        },
    )
    .unwrap();
    let FactorOutput::Diagonal { space, values } = q else {
        panic!("compact QR of a diagonal stays compact");
    };
    let emitted: Vec<_> = values.iter().map(|entry| entry.sector).collect();
    let region_order: Vec<_> = regions.iter().map(|region| region.coupled()).collect();
    assert_eq!(emitted, region_order);
    assert_eq!(emitted, [SectorId::new(1), SectorId::new(0)]);
    admit_compact_diagonal::<Mf, _, _>(&space, &values).unwrap();

    let mut short = unsorted.to_vec();
    short[0].values.pop();
    let mut duplicate = unsorted.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    for (malformed, reason) in [
        (&short[..], "length"),
        (&duplicate[..], "twice"),
        (&unsorted[..1], "sector count"),
    ] {
        assert!(matches!(
            admit_compact_diagonal::<Mf, _, _>(&reversed, malformed),
            Err(OperationError::InvalidArgument { message }) if message.contains(reason)
        ));
    }
}
