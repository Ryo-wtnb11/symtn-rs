use super::*;

/// Hermitian and general payloads of one coupled-sector tiling, written per
/// region so the Hermitian one is Hermitian in the tree basis.
fn region_payloads(
    regions: &[symtn_core::CoupledSectorRegion],
    len: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let entry = |salt: usize| {
        let value = |seed: usize| ((seed * 2_654_435_761) % 1009) as f64 / 1009.0 - 0.5;
        Complex64::new(value(salt), value(salt + 7919))
    };
    let mut general = vec![Complex64::zero(); len];
    let mut hermitian = general.clone();
    for region in regions {
        let (rows, start) = (region.rows(), region.range().start);
        for column in 0..region.cols() {
            for row in 0..rows {
                general[start + row + rows * column] = entry(start + row + rows * column);
            }
        }
        if rows == region.cols() {
            for column in 0..rows {
                for row in 0..rows {
                    hermitian[start + row + rows * column] = general[start + row + rows * column]
                        + general[start + column + rows * row].conj();
                }
            }
        }
    }
    (hermitian, general)
}

// #1494: the leg-degeneracy (facade) builder lists multi-leg Generic trees in
// its own order, which a `FusionTreeKey`-Ord admission would refuse, packing
// the whole payload before QR/SVD/LQ and the value-only spectra. A coupled-sector tiling is the same matricization the
// pack would build, so no pack may happen.
#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_facade_layouts_take_the_direct_region_path() {
    let x = SectorId::new(1);
    let v = SectorLeg::new([(SectorId::new(0), 1), (x, 2)], false);
    let w = SectorLeg::new([(x, 1)], false);
    let product =
        |legs: &[&SectorLeg]| FusionProductSpace::new(legs.iter().map(|&leg| leg.clone()));
    let cases = [
        (product(&[&v, &v]), product(&[&w])),
        (product(&[&w]), product(&[&v, &v])),
        (product(&[&v, &w]), product(&[&v, &w])),
        (product(&[&v, &v, &w]), product(&[&v])),
    ];
    let mut unordered_fixtures = 0;
    for (codomain, domain) in cases {
        let endomorphism = codomain == domain;
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: FactorGenericRule.rule_identity(),
        });
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            FusionTreeHomSpace::new(codomain, domain),
        )
        .unwrap();
        let regions = space
            .space()
            .structure()
            .coupled_sector_regions(space.space().nout())
            .unwrap()
            .expect("the facade builder lays out a coupled-sector tiling");
        let ord_ordered = regions.iter().all(|region| {
            [region.row_trees(), region.col_trees()]
                .iter()
                .all(|trees| trees.windows(2).all(|pair| pair[0].tree() < pair[1].tree()))
        });
        unordered_fixtures += usize::from(!ord_ordered);
        let (hermitian, general) = region_payloads(&regions, space.space().required_len().unwrap());
        let input = BoundDynamicTensorRef::try_new(&space, &general).unwrap();
        let mut dense = symtn_dense::DefaultDenseExecutor::new();

        let generic_pair_rec = crate::probe::record();
        let compact_qr_copy_rec = crate::probe::record();
        let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &qr.q, None, &qr.r);
        assert_eq!(compact_qr_copy_rec.compact_qr_copy().input_pack_bytes, 0);

        let compact_svd_copy_rec = crate::probe::record();
        let svd = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &svd.u, Some(&svd.s), &svd.vh);
        assert_eq!(compact_svd_copy_rec.compact_svd_copy().input_pack_bytes, 0);

        let compact_lq_copy_rec = crate::probe::record();
        let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
        assert_compact_factors_reconstruct_input(&input, &lq.l, None, &lq.q);
        assert_eq!(compact_lq_copy_rec.compact_lq_copy().input_pack_bytes, 0);

        // The fresh bond-space builder enumerates the preserved side's trees
        // in the facade input's order, so the per-call route proof publishes
        // every factor in place: no pack in, no scatter out.
        let publication = generic_pair_rec.generic_pair_publication();
        assert_eq!(
            (
                publication.canonical_publications,
                publication.fallback_publications
            ),
            (3, 0),
            "{publication:?}"
        );
        let values_fallbacks_rec = crate::probe::record();
        svd_vals_dyn_checked_generic(&mut dense, &input).unwrap();
        if endomorphism {
            let hermitian_input = BoundDynamicTensorRef::try_new(&space, &hermitian).unwrap();
            eigh_vals_dyn_checked_generic(&mut dense, &hermitian_input).unwrap();
            eig_vals_dyn_checked_generic(&mut dense, &input).unwrap();
        }
        assert_eq!(
            values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
            0
        );
    }
    assert!(
        unordered_fixtures > 0,
        "the fixtures must include the facade order that Ord admission packed"
    );
}

// Tree order is not admission, but it is still proven where it matters:
// sector `x` of this tiling lists its rows as (t0, t1) and its columns as
// (t1, t0). The direct QR/SVD/LQ outputs must still reconstruct the input
// (the fresh bond spaces are reached by tree identity, never by position),
// and eigenvalues must refuse the mis-stacked endomorphism instead of
// returning the spectrum of a column-permuted block (which the packed path
// silently did, since it stacks rows and columns by first appearance too).
#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_mis_stacked_tiling_scatters_factors_and_refuses_eigenvalues() {
    let (canonical, hermitian, general) = generic_values_endomorphism_input();
    let structure = canonical.space().structure();
    let matrix = structure
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap()
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap()
        .clone();
    let (t0, t1) = (matrix.row_trees()[0].tree(), matrix.row_trees()[1].tree());
    let key_of = |index: usize| {
        let block = structure.block(index).unwrap();
        (
            block.key().as_fusion_tree_pair().unwrap().clone(),
            block.shape().to_vec(),
        )
    };
    let find = |row: &FusionTreeKey, col: &FusionTreeKey| {
        (0..structure.block_count())
            .find(|&index| {
                let key = key_of(index).0;
                key.codomain_tree() == row && key.domain_tree() == col
            })
            .unwrap()
    };
    let scalar = (0..structure.block_count())
        .find(|&index| key_of(index).0.coupled() == SectorId::new(0))
        .unwrap();
    let order = [
        scalar,
        find(t0, t1),
        find(t0, t0),
        find(t1, t0),
        find(t1, t1),
    ];
    let blocks = order.iter().map(|&index| key_of(index)).collect();
    let mis_stacked =
        BlockStructure::coupled_sector_matrix_with_keys(&FactorGenericRule, 2, 4, blocks).unwrap();
    let region = mis_stacked
        .coupled_sector_regions(2)
        .unwrap()
        .expect("the reordered layout is still a coupled-sector tiling")
        .iter()
        .find(|region| region.coupled() == SectorId::new(1))
        .unwrap()
        .clone();
    assert_ne!(region.row_trees(), region.col_trees());

    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let relayout = |source: &[Complex64]| {
        let typed_space = DynamicFusionMapSpace::try_from_layout(
            canonical.space().homspace().clone(),
            mis_stacked.clone(),
        )
        .unwrap()
        .try_bind_rule(&FactorGenericRule)
        .unwrap();
        let tensor = TestTensor::<Complex64>::from_block_fn_with_fusion_space(
            typed_space,
            Complex64::zero(),
            |key, indices| {
                let block = structure
                    .block(structure.find_block_index_by_key(key).unwrap())
                    .unwrap();
                source[block.offset()
                    + indices
                        .iter()
                        .zip(block.strides())
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>()]
            },
        )
        .unwrap();
        (tensor.fusion_space().clone(), tensor.data().to_vec())
    };
    let (dynamic, general_data) = relayout(&general);
    let (_, hermitian_data) = relayout(&hermitian);
    let space = BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(&provider)).unwrap();
    let input = BoundDynamicTensorRef::try_new(&space, &general_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_qr_copy_rec = crate::probe::record();
    let generic_pair_rec = crate::probe::record();
    let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &qr.q, None, &qr.r);
    assert_eq!(compact_qr_copy_rec.compact_qr_copy().input_pack_bytes, 0);
    assert_eq!(
        generic_pair_rec
            .generic_pair_publication()
            .canonical_publications,
        0
    );
    let svd = svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &svd.u, Some(&svd.s), &svd.vh);
    let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &lq.l, None, &lq.q);

    let hermitian_input = BoundDynamicTensorRef::try_new(&space, &hermitian_data).unwrap();
    for (error, operation) in [
        (
            eigh_vals_dyn_checked_generic(&mut dense, &hermitian_input).unwrap_err(),
            "eigh_vals ",
        ),
        (
            eig_vals_dyn_checked_generic(&mut dense, &input).unwrap_err(),
            "eig_vals ",
        ),
    ] {
        assert!(
            matches!(
                error,
                CheckedGenericFactorPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope { message }
                ) if message.starts_with(operation)
            ),
            "{error:?}"
        );
    }
}

fn assert_multiplicity_free_eigen_ops_refuse_mis_stacking<R>(rule: R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = RuleIdentity>
        + Clone
        + fmt::Debug
        + 'static,
{
    let provider = Arc::new(rule.clone());
    let general = mis_stacked_endomorphism_copy(&rule, &tsvd_test_tensor(&rule, sectors));
    let hermitian = mis_stacked_endomorphism_copy(&rule, &hermitian_test_tensor(&rule, sectors));
    let general_bound = bound_tensor(Arc::clone(&provider), &general);
    let hermitian_bound = bound_tensor(Arc::clone(&provider), &hermitian);
    let (general, hermitian) = (general_bound.as_ref(), hermitian_bound.as_ref());
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    assert_stacking_refusal(eig_vals(&mut dense, &general), "eig_vals ");
    assert_stacking_refusal(eig_full(&mut dense, &general), "eig_full ");
    assert_stacking_refusal(eigh_vals(&mut dense, &hermitian), "eigh_vals ");
    assert_stacking_refusal(eigh_full(&mut dense, &hermitian), "eigh_full ");
    assert_stacking_refusal(exp(&mut dense, &general), "exp ");
    assert_stacking_refusal(exp(&mut dense, &hermitian), "exp ");
    assert_stacking_refusal(
        exp_direct_into_dyn(&mut dense, &general.dynamic(), general.space().clone()),
        "exp ",
    );

    // The lazy adjoint is a noncanonical layout, so these reach the packed
    // matricization branches; the adjoint of a mis-stacked block stays
    // mis-stacked.
    let general_adjoint = general_bound.space().adjoint_view().unwrap();
    let general_adjoint =
        BoundDynamicTensorRef::try_new(&general_adjoint, general_bound.data()).unwrap();
    let hermitian_adjoint = hermitian_bound.space().adjoint_view().unwrap();
    let hermitian_adjoint =
        BoundDynamicTensorRef::try_new(&hermitian_adjoint, hermitian_bound.data()).unwrap();
    assert_stacking_refusal(eig_vals_dyn(&mut dense, &general_adjoint), "eig_vals ");
    assert_stacking_refusal(eig_full_dyn(&mut dense, &general_adjoint), "eig_full ");
    assert_stacking_refusal(
        eigh_vals_dyn(&mut dense, &hermitian_adjoint, HermitianTol::DEFAULT),
        "eigh_vals ",
    );
    assert_stacking_refusal(
        eigh_full_dyn(&mut dense, &hermitian_adjoint, HermitianTol::DEFAULT),
        "eigh_full ",
    );
    assert_stacking_refusal(exp_into_mf(&mut dense, &general_adjoint), "exp ");
}

#[test]
fn multiplicity_free_eigen_ops_refuse_mis_stacked_z2_endomorphisms() {
    // What: every multiplicity-free spectral op refuses a tiling whose row and
    // column tree stackings differ, instead of returning the spectrum (or the
    // function) of a column-permuted block.
    assert_multiplicity_free_eigen_ops_refuse_mis_stacking(
        Z2FusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
}

#[test]
fn multiplicity_free_eigen_ops_refuse_mis_stacked_u1_endomorphisms() {
    assert_multiplicity_free_eigen_ops_refuse_mis_stacking(
        U1FusionRule,
        &[-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id()),
    );
}

#[test]
fn multiplicity_free_eigen_ops_refuse_mis_stacked_su2_endomorphisms() {
    assert_multiplicity_free_eigen_ops_refuse_mis_stacking(
        SU2FusionRule,
        &[0, 1].map(|twice| SU2Irrep::from_twice_spin(twice).sector_id()),
    );
}

#[test]
fn multiplicity_free_eigen_ops_accept_consistently_reordered_tree_stacking() {
    // What: the guard is stacking identity, not canonical order — reversing
    // rows and columns together is the same operator in a permuted basis and
    // keeps the facade spectra.
    let rule = U1FusionRule;
    let sectors = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
    let provider = Arc::new(rule);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let general = tsvd_test_tensor(&rule, &sectors);
    let hermitian = hermitian_test_tensor(&rule, &sectors);
    let reordered_general = reversed_coupled_tree_basis_copy(&rule, &general);
    let reordered_hermitian = reversed_coupled_tree_basis_copy(&rule, &hermitian);

    let eig = |dense: &mut symtn_dense::DefaultDenseExecutor, tensor| {
        eig_vals(dense, &bound_tensor_ref!(Arc::clone(&provider), tensor)).unwrap()
    };
    let eigh = |dense: &mut symtn_dense::DefaultDenseExecutor, tensor| {
        eigh_vals(dense, &bound_tensor_ref!(Arc::clone(&provider), tensor)).unwrap()
    };
    for (left, right) in eig(&mut dense, &general)
        .iter()
        .zip(&eig(&mut dense, &reordered_general))
    {
        assert_eq!(left.sector, right.sector);
        for (a, b) in left.values.iter().zip(&right.values) {
            assert!((a - b).norm() <= 1e-10 * a.norm().max(1.0), "{a} vs {b}");
        }
    }
    for (left, right) in eigh(&mut dense, &hermitian)
        .iter()
        .zip(&eigh(&mut dense, &reordered_hermitian))
    {
        assert_eq!(left.sector, right.sector);
        for (a, b) in left.values.iter().zip(&right.values) {
            assert!((a - b).abs() <= 1e-10 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
    eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &reordered_hermitian),
    )
    .unwrap();
    eig_full(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &reordered_general),
    )
    .unwrap();
}

/// Both factors come from the fresh builder of one homspace, so equal
/// layouts compare position by position.
fn assert_factor_matches_facade<R, D>(
    what: &str,
    expected: &BoundDynFactor<R, D>,
    actual: &BoundDynFactor<R, D>,
) where
    D: FactorScalar,
{
    let (left, right) = (
        expected.space().space().structure(),
        actual.space().space().structure(),
    );
    assert_eq!(left.block_count(), right.block_count(), "{what}");
    for index in 0..left.block_count() {
        let (a, b) = (left.block(index).unwrap(), right.block(index).unwrap());
        assert_eq!(a.key(), b.key(), "{what}");
        assert_eq!(a.offset(), b.offset(), "{what}");
        assert_eq!(a.shape(), b.shape(), "{what}");
    }
    let scale = expected
        .data()
        .iter()
        .fold(1.0f64, |max, &value| max.max(value.widen_complex().norm()));
    assert_eq!(expected.data().len(), actual.data().len(), "{what}");
    for (position, (&a, &b)) in expected.data().iter().zip(actual.data()).enumerate() {
        let (a, b) = (a.widen_complex(), b.widen_complex());
        assert!(
            (a - b).norm() <= 1e-10 * scale,
            "{what} at {position}: {a} vs {b}"
        );
    }
}

/// Dense eig leaves each eigenvector's scale free, so compare every bond
/// column of a sector up to one complex factor fixed at its largest entry.
fn assert_eig_columns_match_up_to_scale<R>(
    expected: &BoundDynFactor<R, Complex64>,
    actual: &BoundDynFactor<R, Complex64>,
) {
    let structure = expected.space().space().structure();
    let actual_structure = actual.space().space().structure();
    let mut columns = std::collections::BTreeMap::<(SectorId, usize), Vec<usize>>::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let other = actual_structure.block(index).unwrap();
        assert_eq!(block.key(), other.key());
        assert_eq!(block.offset(), other.offset());
        let BlockKey::FusionTree(key) = block.key() else {
            unreachable!("fusion-tree blocks")
        };
        let shape = block.shape();
        let extent = shape.iter().product::<usize>();
        for flat in 0..extent {
            let mut rest = flat;
            let mut position = block.offset();
            let mut bond = 0;
            for (axis, (&dim, &stride)) in shape.iter().zip(block.strides()).enumerate() {
                let index = rest % dim;
                rest /= dim;
                position += index * stride;
                if axis + 1 == shape.len() {
                    bond = index;
                }
            }
            columns
                .entry((key.coupled(), bond))
                .or_default()
                .push(position);
        }
    }
    for (column, positions) in columns {
        let pivot = *positions
            .iter()
            .max_by(|&&a, &&b| {
                expected.data()[a]
                    .norm()
                    .total_cmp(&expected.data()[b].norm())
            })
            .unwrap();
        let ratio = actual.data()[pivot] / expected.data()[pivot];
        for position in positions {
            let (a, b) = (expected.data()[position] * ratio, actual.data()[position]);
            assert!(
                (a - b).norm() <= 1e-10 * ratio.norm().max(1.0),
                "eig_full column {column:?} at {position}: {a} vs {b}"
            );
        }
    }
}

/// [`hermitian_test_tensor`]'s layout with unsymmetrized hashed entries, so
/// every coupled-sector matrix is full rank with a simple spectrum and its
/// gauge-fixed factors are unique.
fn generic_spectrum_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TestTensor<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let template = hermitian_test_tensor(rule, sectors);
    let space = template.fusion_space().clone();
    TestTensor::<f64>::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let mut hash = 0x9e37_79b9_7f4a_7c15u64;
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        for &sector in tree
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(tree.domain_tree().uncoupled())
        {
            hash = (hash ^ (sector.id() as u64 + 1)).wrapping_mul(0x100_0000_01b3);
        }
        for &index in indices {
            hash = (hash ^ (index as u64 + 7)).wrapping_mul(0x100_0000_01b3);
        }
        ((hash >> 40) % 1009) as f64 / 100.0 - 5.0
    })
    .unwrap()
}

/// What: a consistently reordered compact input (every coupled sector's rows
/// and columns stacked in reverse tree order) is the same tensor, so its
/// factors reconstruct it by tree identity and its SVD factors equal the
/// facade's gauge-fixed ones.
fn assert_reordered_compact_factor_publication<R, D>(provider: Arc<R>, facade: &TestTensor<D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let reordered = reversed_coupled_tree_basis_copy(provider.as_ref(), facade);
    let facade = bound_tensor(Arc::clone(&provider), facade);
    let reordered = bound_tensor(provider, &reordered);
    assert!(reordered
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let (facade, reordered) = (facade.as_ref(), reordered.as_ref());
    let (facade, input) = (facade.dynamic(), reordered.dynamic());
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let expected = svd_compact_dyn(&mut dense, &facade).unwrap();
    let actual = svd_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, actual.u(), Some(actual.s()), actual.vh());
    assert_factor_matches_facade("svd_compact U", expected.u(), actual.u());
    assert_factor_matches_facade("svd_compact Vh", expected.vh(), actual.vh());

    // Why reconstruction only: QR of a column-permuted matrix is a different
    // factorization, so QR/LQ factors are not facade-comparable here.
    let actual = qr_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.q, None, &actual.r);

    let actual = lq_compact_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.l, None, &actual.q);

    // Full factorizations publish through the one-sided tree-identity
    // scatter; their completion columns are not unique, so only
    // reconstruction is checked.
    let actual = svd_full_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, actual.u(), Some(actual.s()), actual.vh());
    let actual = qr_full_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.q, None, &actual.r);
    let actual = lq_full_dyn(&mut dense, &input).unwrap();
    assert_compact_factors_reconstruct_input(&input, &actual.l, None, &actual.q);
}

/// What: eigenvectors of a consistently reordered Hermitian or general
/// endomorphism map rows by tree identity: `A V = V Λ` against the facade
/// operator, `V D Vᴴ = A` over the reordered input, and the facade's
/// gauge-fixed eigenvectors, polar factors and eig vectors.
fn assert_reordered_eigen_factor_publication<R>(rule: R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = RuleIdentity>
        + Clone,
{
    let provider = Arc::new(rule.clone());
    let hermitian = hermitian_test_tensor(&rule, sectors);
    let general = generic_spectrum_test_tensor(&rule, sectors);
    let reordered_hermitian = reversed_coupled_tree_basis_copy(&rule, &hermitian);
    let reordered_general = reversed_coupled_tree_basis_copy(&rule, &general);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &reordered_hermitian),
    )
    .unwrap();
    assert_eigen_equation(&rule, &hermitian, &eigh.v, &eigh.d);

    let input = bound_tensor(Arc::clone(&provider), &reordered_hermitian);
    let input = input.as_ref();
    let input = input.dynamic();
    let eigh = eigh_full_dyn(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
    let vh = crate::factorize::adjoint_bound_factor(eigh.v()).unwrap();
    let mut vd = eigh.v().clone();
    let vd_space = vd.space().space().clone();
    scale_axis_by_spectrum(&vd_space, vd.data_mut(), None, eigh.eigenvalues()).unwrap();
    assert_compact_factors_reconstruct_input(&input, &vd, None, &vh);

    let facade = bound_tensor(Arc::clone(&provider), &general);
    let facade = facade.as_ref();
    let facade = facade.dynamic();
    let reordered = bound_tensor(Arc::clone(&provider), &reordered_general);
    let reordered = reordered.as_ref();
    let reordered = reordered.dynamic();
    let expected = eig_full_dyn(&mut dense, &facade).unwrap();
    let actual = eig_full_dyn(&mut dense, &reordered).unwrap();
    assert_eig_columns_match_up_to_scale(expected.v(), actual.v());

    let expected = left_polar_dyn(&mut dense, &facade).unwrap();
    let actual = left_polar_dyn(&mut dense, &reordered).unwrap();
    assert_factor_matches_facade("left_polar W", &expected.w, &actual.w);
    assert_factor_matches_facade("left_polar P", &expected.p, &actual.p);

    assert_reordered_compact_factor_publication(Arc::clone(&provider), &general);
    let complex = TestTensor::<Complex64>::from_vec_with_fusion_space(
        general
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, ((index * 7 + 3) % 13) as f64 * 0.2 - 1.0))
            .collect(),
        general.fusion_space().clone(),
    )
    .unwrap();
    assert_reordered_compact_factor_publication(provider, &complex);
}

#[test]
fn z2_factor_publication_maps_consistently_reordered_trees_by_identity() {
    assert_reordered_eigen_factor_publication(Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
}

#[test]
fn u1_factor_publication_maps_consistently_reordered_trees_by_identity() {
    assert_reordered_eigen_factor_publication(
        U1FusionRule,
        &[-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id()),
    );
}

#[test]
fn su2_factor_publication_maps_consistently_reordered_trees_by_identity() {
    assert_reordered_eigen_factor_publication(
        SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
}
