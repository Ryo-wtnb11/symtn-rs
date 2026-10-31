use super::*;

#[cfg(feature = "cgc-gen")]
fn multiply_2x2<D: symtn::typed::TensorScalar>(
    left: [[D; 2]; 2],
    right: [[D; 2]; 2],
) -> [[D; 2]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| left[row][0] * right[0][col] + left[row][1] * right[1][col])
    })
}

#[test]
fn checked_generic_polar_lazy_redirects_to_the_opposite_parent_operation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: lazy adjoints reuse the parent's owned P, return owned W-adjoints,
    // and report the operation requested on the receiver.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            [
                [Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.75)],
                [Complex64::new(-0.25, 0.5), Complex64::new(3.0, -0.25)],
            ][ij[0]][ij[1]]
        })
        .unwrap();
    let lazy = source.adjoint(engine).unwrap();
    let RightPolar {
        p: parent_p,
        wh: parent_w,
    } = source.right_polar(engine, &[0], &[1]).unwrap();
    let LeftPolar {
        w: actual_w,
        p: actual_p,
    } = lazy.left_polar(engine, &[0], &[1]).unwrap();
    assert_same_checked_generic_layout_and_close(
        &actual_w,
        &parent_w.adjoint(engine).unwrap(),
        |a, b| (a - b).norm(),
    );
    assert_same_checked_generic_layout_and_close(&actual_p, &parent_p, |a, b| (a - b).norm());
    let LeftPolar {
        w: parent_w,
        p: parent_p,
    } = source.left_polar(engine, &[0], &[1]).unwrap();
    let RightPolar {
        p: actual_p,
        wh: actual_w,
    } = lazy.right_polar(engine, &[0], &[1]).unwrap();
    assert_same_checked_generic_layout_and_close(&actual_p, &parent_p, |a, b| (a - b).norm());
    assert_same_checked_generic_layout_and_close(
        &actual_w,
        &parent_w.adjoint(engine).unwrap(),
        |a, b| (a - b).norm(),
    );
    assert!(std::ptr::eq(actual_w.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::runtime_identity(actual_w.runtime()).matches(source.runtime())
    );

    let tall = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let tall_source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&tall], [&narrow], |_, _| 1.0).unwrap();
    let error = tall_source
        .adjoint(engine)
        .unwrap()
        .left_polar(engine, &[0], &[1])
        .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(error))
            if matches!(
                error,
                symtn::typed::OperationError::InvalidArgument { message }
                    if message == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));
}

#[test]
fn checked_generic_polar_completes_rank_deficient_and_zero_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: compact-full SVD completion returns a full isometry/coisometry;
    // P remains Hermitian PSD for rank-deficient and zero matrices.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for zero in [false, true] {
        let source: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
                if zero {
                    Complex64::new(0.0, 0.0)
                } else {
                    let row = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -0.5)];
                    let col = [Complex64::new(0.5, -1.0), Complex64::new(-1.5, 2.0)];
                    row[ij[0]] * col[ij[1]].conj()
                }
            })
            .unwrap();
        for left in [true, false] {
            let (w, p) = if left {
                let LeftPolar { w, p } = source.left_polar(engine, &[0], &[1]).unwrap();
                (w, p)
            } else {
                let RightPolar { p, wh } = source.right_polar(engine, &[0], &[1]).unwrap();
                (wh, p)
            };
            let rebuilt = if left {
                w.compose(engine, &p).unwrap()
            } else {
                p.compose(engine, &w).unwrap()
            };
            for (&actual, &expected) in rebuilt
                .dense_data()
                .unwrap()
                .iter()
                .zip(source.dense_data().unwrap())
            {
                assert!((actual - expected).norm() < 1e-9);
            }
            for col in 0..2 {
                for row in 0..2 {
                    let gram = (0..2).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                        if left {
                            sum + w.dense_data().unwrap()[inner + 2 * row].conj()
                                * w.dense_data().unwrap()[inner + 2 * col]
                        } else {
                            sum + w.dense_data().unwrap()[row + 2 * inner]
                                * w.dense_data().unwrap()[col + 2 * inner].conj()
                        }
                    });
                    assert!((gram - Complex64::new(f64::from(row == col), 0.0)).norm() < 1e-10);
                    assert!(
                        (p.dense_data().unwrap()[row + 2 * col]
                            - p.dense_data().unwrap()[col + 2 * row].conj())
                        .norm()
                            < 1e-10
                    );
                }
            }
            assert!(p
                .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap()
                .iter()
                .flat_map(|entry| &entry.values)
                .all(|&value| value >= -1e-10));
        }
    }
}

#[test]
fn checked_generic_null_spaces_cover_shape_zero_disjoint_and_side_only_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: Generic null spaces are shape-based (MatrixAlgebraKit
    // `qr_null!`/`lq_null!`): square sectors are empty whatever their rank,
    // a tall sector keeps `rows - cols` orthonormal directions annihilating
    // the input even when rank-deficient, zero/side-only directions are kept
    // in full, and the provider authority is retained.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for small in [0.0, f64::EPSILON, 1.0] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&x], [&x], |_, index| match index {
                [0, 0] => 1.0,
                [1, 1] => small,
                _ => 0.0,
            })
            .unwrap();
        for null in [
            source.left_null(engine, &[0], &[1]).unwrap(),
            source.right_null(engine, &[0], &[1]).unwrap(),
        ] {
            assert!(std::ptr::eq(null.provider(), provider.as_ref()));
            assert!(null.dense_data().unwrap().is_empty());
        }
    }

    let x3 = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    // Rank one: an outer product `a b^T`, so both columns are multiples of
    // `a`. Checked Generic composition reads owned tensors, so adjoints are
    // materialized through `axpby`.
    let owned = |engine: &mut dyn Engine, t: &TensorMap<_, Complex64>| {
        let lazy = t.adjoint(engine).unwrap();
        lazy.axpby(
            engine,
            Complex64::new(1.0, 0.0),
            &lazy,
            Complex64::new(0.0, 0.0),
        )
        .unwrap()
    };
    let tall: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x3], [&x], |_, index| {
            Complex64::new((index[0] + 1) as f64, 0.5) * Complex64::new(1.0, index[1] as f64)
        })
        .unwrap();
    let left = tall.left_null(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(left.provider(), provider.as_ref()));
    assert_eq!(left.dense_data().unwrap().len(), 3);
    let left_adjoint = owned(engine, &left);
    assert!(
        left_adjoint
            .compose(engine, &tall)
            .unwrap()
            .norm(engine, 2.0)
            .unwrap()
            < 1e-12
    );
    let gram = left_adjoint.compose(engine, &left).unwrap();
    assert!((gram.dense_data().unwrap()[0] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    assert!(tall
        .right_null(engine, &[0], &[1])
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    let wide = owned(engine, &tall);
    let right = wide.right_null(engine, &[0], &[1]).unwrap();
    assert_eq!(right.dense_data().unwrap().len(), 3);
    let operand = &owned(engine, &right);
    assert!(
        wide.compose(engine, operand)
            .unwrap()
            .norm(engine, 2.0)
            .unwrap()
            < 1e-12
    );

    let zero: TensorMap<_, Complex64> = TensorMap::zeros(engine, &runtime, [&x], [&x]).unwrap();
    assert!(zero
        .left_null(engine, &[0], &[1])
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(zero
        .right_null(engine, &[0], &[1])
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    let vacuum = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 3)]).unwrap();
    let disjoint: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&x], [&vacuum]).unwrap();
    let left = disjoint.left_null(engine, &[0], &[1]).unwrap();
    let right = disjoint.right_null(engine, &[0], &[1]).unwrap();
    assert_eq!(left.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.dense_data().unwrap().len(), 9);
    for column in 0..3 {
        for row in 0..3 {
            assert_eq!(
                right.dense_data().unwrap()[row + 3 * column],
                f64::from(row == column)
            );
        }
    }

    let shared =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let codomain_side: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&shared], [&unit], |trees, index| {
            f64::from(trees.coupled() == &Label::Vacuum && index[0] == index[1])
        })
        .unwrap();
    assert_eq!(
        codomain_side
            .left_null(engine, &[0], &[1])
            .unwrap()
            .dense_data()
            .unwrap()
            .len(),
        4
    );
    assert!(codomain_side
        .right_null(engine, &[0], &[1])
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_null_spaces<D>(
    engine: &mut dyn Engine,
    n: usize,
    label: Vec<i64>,
    vertex_weights: [D; 2],
    close: impl Fn(D, D) -> f64,
) where
    D: symtn::typed::FactorizationScalar
        + symtn::typed::SpectrumMagnitude
        + fmt::Debug
        + PartialEq
        + numerics::Numeric,
{
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    // `V ⊗ V <- V` with `V` the adjoint irrep: its coupled sector `V` has
    // outer multiplicity 2, so that block is `(2 * 2 * 2) x 2`, its rows
    // mixing both vertices. Both columns are equal, so the block has rank
    // one; the shape-based null space ignores that.
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, index| {
            vertex_weights[trees.codomain_vertices()[0].get() - 1]
                * D::from_real((1 + index[0] + 2 * index[1]) as f64)
        })
        .unwrap();

    let left = source.left_null(engine, &[0, 1], &[2]).unwrap();
    let right = source.right_null(engine, &[0, 1], &[2]).unwrap();
    assert!(std::ptr::eq(left.provider(), provider.as_ref()));
    assert!(std::ptr::eq(right.provider(), provider.as_ref()));
    assert!(right.dense_data().unwrap().is_empty());
    assert_eq!(left.domain()[0].degeneracy(&label).unwrap(), 2 * 2 * 2 - 2);

    // `axpby` of two lazy adjoints stays lazy (A10); materialize instead.
    let left_adjoint = left.adjoint(engine).unwrap().materialize(engine).unwrap();
    for value in left_adjoint
        .compose(engine, &source)
        .unwrap()
        .dense_data()
        .unwrap()
    {
        assert!(close(*value, D::from_real(0.0)) < 1e-9);
    }
    let gram = left_adjoint.compose(engine, &left).unwrap();
    for block_index in 0..gram.subblock_count() {
        let block = gram.subblock(block_index).unwrap();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                let expected = D::from_real(f64::from(row == column));
                let actual = gram.dense_data().unwrap()
                    [block.offset() + row * block.strides()[0] + column * block.strides()[1]];
                assert!(close(actual, expected) < 1e-9);
            }
        }
    }

    // The rank-revealing null space by composition (TensorKit
    // `left_null(t; alg = :svd)`, default `rtol`): the rank-one multiplicity
    // block keeps `8 - 1` directions, every codomain-only sector all of its
    // states, as the shape-based basis does.
    let symtn::typed::Svd { u, s, .. } = source.svd_full(engine, &[0, 1], &[2]).unwrap();
    let bond = u.domain()[0].clone();
    let spectrum = bond.extend_spectrum(&s.diagview(engine).unwrap()).unwrap();
    let below = symtn::typed::Truncation::below(0.0, f64::EPSILON.powf(2.0 / 3.0)).unwrap();
    let found = bond.find_truncated(&spectrum, &below).unwrap();
    let revealed = u.restrict_leg(engine, &[(2, &found.selection)]).unwrap();
    assert!(std::ptr::eq(revealed.provider(), provider.as_ref()));
    assert_eq!(
        revealed.domain()[0].degeneracy(&label).unwrap(),
        2 * 2 * 2 - 1
    );
    for sector in left.domain()[0].sectors().unwrap() {
        if sector != label {
            assert_eq!(
                revealed.domain()[0].degeneracy(&sector).unwrap(),
                left.domain()[0].degeneracy(&sector).unwrap()
            );
        }
    }
    let revealed_adjoint = revealed.adjoint(engine).unwrap();
    let revealed_adjoint = revealed_adjoint
        .axpby(
            engine,
            D::from_real(1.0),
            &revealed_adjoint,
            D::from_real(0.0),
        )
        .unwrap();
    for value in revealed_adjoint
        .compose(engine, &source)
        .unwrap()
        .dense_data()
        .unwrap()
    {
        assert!(close(*value, D::from_real(0.0)) < 1e-9);
    }
    let gram = revealed_adjoint.compose(engine, &revealed).unwrap();
    for block_index in 0..gram.subblock_count() {
        let block = gram.subblock(block_index).unwrap();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                let expected = D::from_real(f64::from(row == column));
                let actual = gram.dense_data().unwrap()
                    [block.offset() + row * block.strides()[0] + column * block.strides()[1]];
                assert!(close(actual, expected) < 1e-9);
            }
        }
    }

    let lazy = source.adjoint(engine).unwrap();
    let lazy_right = lazy.right_null(engine, &[0], &[1, 2]).unwrap();
    assert!(std::ptr::eq(lazy_right.provider(), provider.as_ref()));
    // Path agreement is the contract: the lazy route is defined as the
    // adjoint of the parent's left null space, gauge included.
    numerics::assert_slices_close(
        "right_null of the lazy adjoint",
        lazy_right.dense_data().unwrap(),
        left_adjoint.dense_data().unwrap(),
        endomorphism_terms(source.dense_data().unwrap().len()),
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_null_spaces_resolve_cross_mu_for_both_dtypes() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: SU(3)/SU(4) multiplicity blocks keep `rows - cols` orthonormal
    // directions annihilating a rank-deficient input whose rows mix both
    // vertices, and the composed rank-revealing null space adds the rank
    // deficit, for real and complex payloads.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_null_spaces::<f64>(
            engine,
            n,
            label.clone(),
            [1.0, 2.0],
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_null_spaces::<Complex64>(
            engine,
            n,
            label,
            [Complex64::new(1.0, 1.0), Complex64::new(2.0, -0.5)],
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_polar_qh<D>(
    engine: &mut dyn Engine,
    n: usize,
    label: Vec<i64>,
    q: [[D; 2]; 2],
    h: [[D; 2]; 2],
    close: impl Fn(D, D) -> f64 + Copy,
) where
    D: symtn::typed::FactorizationScalar + fmt::Debug + PartialEq,
{
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let build = |engine: &mut dyn Engine, matrix: [[D; 2]; 2], fallback: D| {
        let cross_sector = label.clone();
        TensorMap::from_subblock_fn(
            engine,
            &runtime,
            [&leg, &leg],
            [&leg, &leg],
            move |trees, index| {
                let row = index[0] + 2 * index[1];
                let col = index[2] + 2 * index[3];
                if row != col {
                    return D::from_real(0.0);
                }
                if trees.coupled() == &cross_sector {
                    matrix[trees.codomain_vertices()[0].get() - 1]
                        [trees.domain_vertices()[0].get() - 1]
                } else if trees.codomain_vertices() == trees.domain_vertices() {
                    fallback
                } else {
                    D::from_real(0.0)
                }
            },
        )
        .unwrap()
    };
    let expected_q = build(engine, q, D::from_real(1.0));
    let expected_h = build(engine, h, D::from_real(2.0));
    let mut saw_cross_mu = false;
    for (source_matrix, left) in [(multiply_2x2(q, h), true), (multiply_2x2(h, q), false)] {
        let source = build(engine, source_matrix, D::from_real(2.0));
        saw_cross_mu |= (0..source.subblock_count()).any(|index| {
            let trees = source.subblock_fusion_trees(index).unwrap();
            trees.coupled() == &label
                && trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
        });
        let (actual_q, actual_h) = if left {
            let LeftPolar { w, p } = source.left_polar(engine, &[0, 1], &[2, 3]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = source.right_polar(engine, &[0, 1], &[2, 3]).unwrap();
            (wh, p)
        };
        assert!(std::ptr::eq(actual_q.provider(), provider.as_ref()));
        assert!(std::ptr::eq(actual_h.provider(), provider.as_ref()));
        assert_eq!(actual_q.codomain(), source.codomain());
        assert_eq!(actual_q.domain(), source.domain());
        assert_same_checked_generic_layout_and_close(&actual_q, &expected_q, close);
        assert_same_checked_generic_layout_and_close(&actual_h, &expected_h, close);
        let rebuilt = if left {
            actual_q.compose(engine, &actual_h).unwrap()
        } else {
            actual_h.compose(engine, &actual_q).unwrap()
        };
        assert_same_checked_generic_layout_and_close(&rebuilt, &source, close);
    }
    assert!(
        saw_cross_mu,
        "SU(N) polar fixture must carry a cross-mu full key"
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_polar_cross_mu_qh_oracles_for_both_dtypes() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_polar_qh::<f64>(
            engine,
            n,
            label.clone(),
            [[0.0, -1.0], [1.0, 0.0]],
            [[2.0, 0.5], [0.5, 3.0]],
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_polar_qh::<Complex64>(
            engine,
            n,
            label,
            [
                [Complex64::new(s, 0.0), Complex64::new(0.0, s)],
                [Complex64::new(0.0, s), Complex64::new(s, 0.0)],
            ],
            [
                [Complex64::new(2.0, 0.0), Complex64::new(0.25, 0.5)],
                [Complex64::new(0.25, -0.5), Complex64::new(3.0, 0.0)],
            ],
            |actual, expected| (actual - expected).norm(),
        );
    }
}

/// A compact diagonal's polar factors live on the input space for either
/// orientation (TensorKit: `W = space(t)`, `P = domain <- domain`): no
/// provider query is made, so a failing or style-changing provider cannot
/// fail them.
#[test]
fn checked_compact_diagonal_polar_publishes_on_the_input_space_without_provider_queries() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    for dual in [false, true] {
        for fault in ["fail_algebra", "invalid_style"] {
            let provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
            let mut bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
            if dual {
                bond = bond.try_dual().unwrap();
            }
            let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
                engine,
                &runtime,
                &bond,
                [SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0, -1.0],
                }],
            )
            .unwrap();
            match fault {
                "fail_algebra" => provider.fail_algebra.store(true, Ordering::Relaxed),
                _ => provider.invalid_style.store(true, Ordering::Relaxed),
            }
            reset_provider_queries(&provider);
            let LeftPolar { w, p } = diagonal.left_polar(engine, &[0], &[1]).unwrap();
            let RightPolar { p: right_p, wh } = diagonal.right_polar(engine, &[0], &[1]).unwrap();
            assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
            for factor in [&w, &p, &right_p, &wh] {
                assert_eq!(factor.codomain(), diagonal.codomain());
                assert_eq!(factor.domain(), diagonal.domain());
            }
        }
    }
}
