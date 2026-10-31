//! Polar decomposition tests.

use super::*;

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_polar_covers_scalar_maps() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a rank-zero tensor map is a one-by-one vacuum-sector matrix for
    // both left and right polar decomposition.
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    let homspace = FusionTreeHomSpace::new(empty.clone(), empty);
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();
    let data: [f64; 1] = [-3.0];
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let LeftPolar {
        w: left_w,
        p: left_p,
    } = left_polar_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    let RightPolar {
        p: right_p,
        wh: right_w,
    } = right_polar_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    assert!(Arc::ptr_eq(left_w.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(left_p.space().provider_arc(), &provider));
    assert!((left_w.data()[0].abs() - 1.0).abs() < 1.0e-12);
    assert!((right_w.data()[0].abs() - 1.0).abs() < 1.0e-12);
    assert!((left_p.data()[0] - 3.0).abs() < 1.0e-12);
    assert!((right_p.data()[0] - 3.0).abs() < 1.0e-12);
    assert!((left_w.data()[0] * left_p.data()[0] - data[0]).abs() < 1.0e-12);
    assert!((right_p.data()[0] * right_w.data()[0] - data[0]).abs() < 1.0e-12);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_polar_and_pinv_land_a_tiled_source_whose_tree_order_differs_from_fresh_outputs()
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: admission accepts any coupled-sector tiling, so a tiled source
    // whose multiplicity trees are stacked in reverse of the fresh P / pinv
    // output order lands in the fresh outputs by tree identity: every factor
    // equals the one of the fresh-layout source, never an answer in
    // mismatched coordinates.
    let (base, base_data) = generic_factorization_input();
    let base_structure = base.space().structure();
    let blocks = (0..base_structure.block_count())
        .rev()
        .map(|index| {
            let block = base_structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!("generic input has fusion-tree blocks")
            };
            (key.clone(), block.shape().to_vec())
        })
        .collect();
    let structure =
        BlockStructure::coupled_sector_matrix_with_keys(&FactorGenericRule, 2, 4, blocks).unwrap();
    let typed_space =
        DynamicFusionMapSpace::try_from_layout(base.space().homspace().clone(), structure)
            .unwrap()
            .try_bind_rule(&FactorGenericRule)
            .unwrap();
    let tensor =
        TestTensor::<f64>::from_block_fn_with_fusion_space(typed_space, 0.0, |key, indices| {
            let block = base_structure
                .block(base_structure.find_block_index_by_key(key).unwrap())
                .unwrap();
            base_data[block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()]
        })
        .unwrap();
    // The space was admitted for `FactorGenericRule`, so this
    // never-failing spy answers under that identity.
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let source = BoundDynamicFusionMapSpace::bind_generic(
        tensor.fusion_space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let input = BoundDynamicTensorRef::try_new(&source, tensor.data()).unwrap();

    let fresh = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        base.space().homspace().clone(),
    )
    .unwrap();
    let tiled = source
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .expect("the reordered source is still a coupled-sector tiling");
    let fresh_regions = fresh
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    assert!(tiled
        .iter()
        .zip(fresh_regions.iter())
        .any(|(source, fresh)| source.col_trees() != fresh.col_trees()));

    // The same values on the spy, in the fresh layout.
    assert_eq!(fresh.space().structure(), base.space().structure());
    let base_input = BoundDynamicTensorRef::try_new(&fresh, &base_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let expected = left_polar_dyn_checked_generic(engine, &mut dense, &base_input).unwrap();
    let actual = left_polar_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    assert_blocks_close_by_key("checked W", &expected.w, &actual.w);
    assert_blocks_close_by_key("checked P", &expected.p, &actual.p);

    let swapped = || {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            FusionTreeHomSpace::new(
                source.space().homspace().domain().clone(),
                source.space().homspace().codomain().clone(),
            ),
        )
        .unwrap()
    };
    let expected = pinv_direct_into_dyn(engine, &mut dense, &base_input, swapped(), 0.0).unwrap();
    let actual = pinv_direct_into_dyn(engine, &mut dense, &input, swapped(), 0.0).unwrap();
    assert_blocks_close_by_key("checked pinv", &expected, &actual);
}

/// Every block of `expected` and `actual`, matched by fusion-tree key and
/// compared element by element, whatever either layout's block order.
fn assert_blocks_close_by_key<R, D: FactorScalar>(
    what: &str,
    expected: &BoundDynFactor<R, D>,
    actual: &BoundDynFactor<R, D>,
) {
    let (left, right) = (
        expected.space().space().structure(),
        actual.space().space().structure(),
    );
    assert_eq!(left.block_count(), right.block_count(), "{what}");
    for index in 0..left.block_count() {
        let a = left.block(index).unwrap();
        let b = right
            .block(right.find_block_index_by_key(a.key()).unwrap())
            .unwrap();
        assert_eq!(a.shape(), b.shape(), "{what}");
        let elements = a.shape().iter().product::<usize>();
        for linear in 0..elements {
            let (mut rest, mut x, mut y) = (linear, a.offset(), b.offset());
            for (axis, &extent) in a.shape().iter().enumerate() {
                x += (rest % extent) * a.strides()[axis];
                y += (rest % extent) * b.strides()[axis];
                rest /= extent;
            }
            let (x, y) = (
                expected.data()[x].widen_complex(),
                actual.data()[y].widen_complex(),
            );
            assert!(
                (x - y).norm() <= 1e-12,
                "{what} {index}/{linear}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn polar_rejects_wrong_rectangular_direction_before_dense_execution() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: invalid left/right directions are rejected before any sector SVD starts.
    let rule = Z2FusionRule;
    for (operation, rows, cols) in [("left_polar", 2, 3), ("right_polar", 3, 2)] {
        let tensor = rectangular_svd_tensor(rows, cols);
        let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
        let result = if operation == "left_polar" {
            left_polar(
                engine,
                &mut dense,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .map(drop)
        } else {
            right_polar(
                engine,
                &mut dense,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .map(drop)
        };

        assert!(matches!(
            result,
            Err(OperationError::InvalidArgument { message })
                if message.contains(operation)
                    && message.contains("coupled-sector")
        ));
    }
}

fn assert_polar_direction_error_before_dense(tensor: &TestTensor<f64>, left: bool) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let before = tensor.data().to_vec();
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let input = bound_tensor(Arc::new(U1FusionRule), tensor);
    let error = if left {
        left_polar(engine, &mut dense, &input.as_ref()).unwrap_err()
    } else {
        right_polar(engine, &mut dense, &input.as_ref()).unwrap_err()
    };
    let operation = if left { "left_polar" } else { "right_polar" };
    assert!(matches!(
        error,
        OperationError::InvalidArgument { message }
            if message.contains(operation) && message.contains("coupled-sector")
    ));
    assert_eq!(tensor.data(), before);
}

fn assert_valid_unmatched_left_polar(tensor: &TestTensor<f64>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let input = bound_tensor(Arc::clone(&provider), tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let LeftPolar {
        w: isometry,
        p: positive,
    } = left_polar(engine, &mut dense, &input.as_ref()).unwrap();

    assert!(Arc::ptr_eq(isometry.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(positive.space().provider_arc(), &provider));
    let reconstructed = crate::compose::compose(
        engine,
        &mut context,
        provider.as_ref(),
        &isometry,
        &positive,
    )
    .unwrap();
    assert_svd_blocks_match(tensor, &reconstructed);
    let adjoint = adjoint(provider.as_ref(), &isometry).unwrap();
    let gram =
        crate::compose::compose(engine, &mut context, provider.as_ref(), &adjoint, &isometry)
            .unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &gram));
    assert_identity_matrices(&dense_sector_matrices(1, &positive));
}

fn assert_valid_unmatched_right_polar(tensor: &TestTensor<f64>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let input = bound_tensor(Arc::clone(&provider), tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let RightPolar {
        p: positive,
        wh: isometry,
    } = right_polar(engine, &mut dense, &input.as_ref()).unwrap();

    assert!(Arc::ptr_eq(positive.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(isometry.space().provider_arc(), &provider));
    let reconstructed = crate::compose::compose(
        engine,
        &mut context,
        provider.as_ref(),
        &positive,
        &isometry,
    )
    .unwrap();
    assert_svd_blocks_match(tensor, &reconstructed);
    let adjoint = adjoint(provider.as_ref(), &isometry).unwrap();
    let gram =
        crate::compose::compose(engine, &mut context, provider.as_ref(), &isometry, &adjoint)
            .unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &gram));
    assert_identity_matrices(&dense_sector_matrices(1, &positive));
}

#[test]
fn polar_complete_dimension_preflight_handles_unmatched_and_disjoint_support() {
    // What: side-only sectors participate as rows x 0 or 0 x columns, so only
    // the direction whose isometry law is structurally possible is accepted.
    let codomain_only = u1_cross_space_map::<f64>(&[(0, 2), (1, 3)], &[(0, 2)]);
    assert_valid_unmatched_left_polar(&codomain_only);
    assert_polar_direction_error_before_dense(&codomain_only, false);

    let domain_only = u1_cross_space_map::<f64>(&[(0, 2)], &[(0, 2), (1, 3)]);
    assert_valid_unmatched_right_polar(&domain_only);
    assert_polar_direction_error_before_dense(&domain_only, true);

    let disjoint = u1_cross_space_map::<f64>(&[(1, 2)], &[(0, 3)]);
    assert_polar_direction_error_before_dense(&disjoint, true);
    assert_polar_direction_error_before_dense(&disjoint, false);
}

#[test]
fn polar_complete_dimension_preflight_handles_empty_sides_and_empty_products() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an empty smaller side remains a valid vacuous isometry, while an
    // empty larger side is rejected; rank-zero products still carry vacuum.
    let empty_codomain = u1_cross_space_map::<f64>(&[], &[(0, 2)]);
    assert_polar_direction_error_before_dense(&empty_codomain, true);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    right_polar(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &empty_codomain),
    )
    .unwrap();

    let empty_domain = u1_cross_space_map::<f64>(&[(0, 2)], &[]);
    assert_polar_direction_error_before_dense(&empty_domain, false);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    left_polar(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &empty_domain),
    )
    .unwrap();

    let empty = u1_cross_space_map::<f64>(&[], &[]);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let input = bound_tensor(Arc::new(U1FusionRule), &empty);
    left_polar(engine, &mut dense, &input.as_ref()).unwrap();
    right_polar(engine, &mut dense, &input.as_ref()).unwrap();

    let rule = U1FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let scalar = TestTensor::<f64>::from_vec_with_fusion_space(vec![2.0], space).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let input = bound_tensor(Arc::new(rule), &scalar);
    // `P = (sqrt(S) Vh)^H (sqrt(S) Vh)` (MAK `PolarViaSVD`): `sqrt(2)^2`
    // rounds, so `P` is `2` to one ulp.
    let LeftPolar { w, p } = left_polar(engine, &mut dense, &input.as_ref()).unwrap();
    assert_eq!(w.data(), &[1.0]);
    numerics::assert_slices_close("left P", p.data(), &[2.0], 1);
    let RightPolar { p, wh: w } = right_polar(engine, &mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("right P", p.data(), &[2.0], 1);
    assert_eq!(w.data(), &[1.0]);
}

#[test]
fn polar_second_sector_failure_leaves_the_source_unchanged() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a later dense failure publishes no factors and cannot mutate the
    // borrowed source, in either direction. Sectors stream, so the first
    // sector's products run before the second SVD fails.
    let canonical = u1_cross_space_map::<f64>(&[(0, 2), (1, 2)], &[(0, 2), (1, 2)]);
    let tensor = padded_copy(&U1FusionRule, &canonical);
    let before = tensor.data().to_vec();
    let input = bound_tensor(Arc::new(U1FusionRule), &tensor);
    for left in [true, false] {
        let mut dense = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
        dense.script.fail(
            &[Op::Svd, Op::SvdInto],
            Some(2),
            "svd_into",
            "injected second-sector failure",
        );
        let compact_svd_copy_rec = crate::probe::record();
        let input_pack_bytes_rec = crate::probe::record();
        let result = if left {
            left_polar(engine, &mut dense, &input.as_ref()).map(drop)
        } else {
            right_polar(engine, &mut dense, &input.as_ref()).map(drop)
        };
        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
        assert_eq!(tensor.data(), before);
        // The padded input is packed once; no SVD factor is laid out.
        assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
        assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
    }
}

#[test]
fn polar_valid_direct_and_fallback_layouts_agree() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: valid fallback matricizations preserve the direct polar factors in both directions.
    let rule = Z2FusionRule;
    for (operation, source_rows, source_cols) in [("left_polar", 2, 3), ("right_polar", 3, 2)] {
        let source = rectangular_svd_tensor(source_rows, source_cols);
        let transposed = transposed_rectangular_tensor(&source, source_rows, source_cols);
        let source_bound = bound_tensor(Arc::new(rule), &source);
        let direct_bound = bound_tensor(Arc::new(rule), &transposed);
        let fallback_space = source_bound.space().adjoint_view().unwrap();
        let fallback_input =
            BoundDynamicTensorRef::try_new(&fallback_space, source_bound.data()).unwrap();
        assert!(
            crate::factorize::compact_factor_plan_for_test(direct_bound.space())
                .unwrap()
                .is_some()
        );
        assert!(
            crate::factorize::compact_factor_plan_for_test(&fallback_space)
                .unwrap()
                .is_none()
        );
        let mut direct_dense = symtn_dense::DefaultDenseExecutor::new();
        let mut fallback_dense = symtn_dense::DefaultDenseExecutor::new();
        let compact_svd_copy_rec = crate::probe::record();
        let input_pack_bytes_rec = crate::probe::record();

        let (direct_first, direct_second, fallback_first, fallback_second) =
            if operation == "left_polar" {
                let LeftPolar {
                    w: direct_first,
                    p: direct_second,
                } = left_polar(engine, &mut direct_dense, &direct_bound.as_ref()).unwrap();
                let LeftPolar {
                    w: fallback_first,
                    p: fallback_second,
                } = left_polar_dyn(engine, &mut fallback_dense, &fallback_input).unwrap();
                (
                    direct_first.data().to_vec(),
                    direct_second.data().to_vec(),
                    fallback_first.data().to_vec(),
                    fallback_second.data().to_vec(),
                )
            } else {
                let RightPolar {
                    p: direct_first,
                    wh: direct_second,
                } = right_polar(engine, &mut direct_dense, &direct_bound.as_ref()).unwrap();
                let RightPolar {
                    p: fallback_first,
                    wh: fallback_second,
                } = right_polar_dyn(engine, &mut fallback_dense, &fallback_input).unwrap();
                (
                    direct_first.data().to_vec(),
                    direct_second.data().to_vec(),
                    fallback_first.data().to_vec(),
                    fallback_second.data().to_vec(),
                )
            };

        assert_eq!(direct_first.len(), fallback_first.len());
        assert_eq!(direct_second.len(), fallback_second.len());
        for (direct, fallback) in direct_first.iter().zip(&fallback_first) {
            assert!((direct - fallback).abs() < 1e-10);
        }
        for (direct, fallback) in direct_second.iter().zip(&fallback_second) {
            assert!((direct - fallback).abs() < 1e-10);
        }
        // The padded input is packed once; no SVD factor is laid out.
        assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
        assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
    }
}

/// An SU(2) map with two legs of spins {0, 1/2} on each side, degeneracy 2
/// on the long side and 1 on the short side: `tall` has the long side as
/// codomain, so every coupled-sector matrix has at least as many rows as
/// columns; otherwise the sides are swapped. Spins 0 and 1/2 each stack two
/// trees on both sides, so `P` has several trees per sector too.
fn su2_polar_input<D: FactorScalar>(
    tall: bool,
    value: fn(f64, f64) -> D,
) -> BoundTensorMap<SU2FusionRule, D> {
    let rule = SU2FusionRule;
    let side = |degeneracy| {
        let leg = || {
            SectorLeg::new(
                [0, 1].map(|twice| (SU2Irrep::from_twice_spin(twice).sector_id(), degeneracy)),
                false,
            )
        };
        FusionProductSpace::new([leg(), leg()])
    };
    let (long, short) = (side(2), side(1));
    let homspace = if tall {
        FusionTreeHomSpace::new(long, short)
    } else {
        FusionTreeHomSpace::new(short, long)
    };
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let data = (0..space.required_len().unwrap())
        .map(|index| {
            value(
                ((index * 11 + 5) % 29) as f64 * 0.25 - 3.0,
                ((index * 7 + 3) % 17) as f64 * 0.25 - 2.0,
            )
        })
        .collect();
    let tensor = TestTensor::<D>::from_vec_with_fusion_space(data, space).unwrap();
    bound_tensor(Arc::new(rule), &tensor)
}

/// Column-major dense matrix of one coupled sector, widened to `Complex64`.
struct SectorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<Complex64>,
}

impl SectorMatrix {
    fn identity(n: usize) -> Self {
        let mut data = vec![Complex64::zero(); n * n];
        (0..n).for_each(|i| data[i + n * i] = Complex64::new(1.0, 0.0));
        Self {
            rows: n,
            cols: n,
            data,
        }
    }

    fn adjoint(&self) -> Self {
        let mut data = vec![Complex64::zero(); self.data.len()];
        for col in 0..self.cols {
            for row in 0..self.rows {
                data[col + self.cols * row] = self.data[row + self.rows * col].conj();
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    fn mul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows);
        let mut data = vec![Complex64::zero(); self.rows * rhs.cols];
        for col in 0..rhs.cols {
            for k in 0..self.cols {
                for row in 0..self.rows {
                    data[row + self.rows * col] +=
                        self.data[row + self.rows * k] * rhs.data[k + rhs.rows * col];
                }
            }
        }
        Self {
            rows: self.rows,
            cols: rhs.cols,
            data,
        }
    }

    fn assert_close(&self, expected: &Self, what: &str) {
        assert_eq!(
            (self.rows, self.cols),
            (expected.rows, expected.cols),
            "{what}"
        );
        for (index, (x, y)) in self.data.iter().zip(&expected.data).enumerate() {
            assert!((x - y).norm() < 1e-10, "{what} [{index}]: {x} vs {y}");
        }
    }

    /// Positive definite iff every pivot of the Hermitian `L D Lᴴ`
    /// elimination is positive (a singular `P` would give a zero pivot).
    fn assert_positive_definite(&self, what: &str) {
        let n = self.rows;
        let mut m = self.data.clone();
        for k in 0..n {
            let pivot = m[k + n * k];
            assert!(
                pivot.re > 1e-8 && pivot.im.abs() < 1e-10,
                "{what}: pivot {k} = {pivot}"
            );
            for col in k + 1..n {
                for row in k + 1..n {
                    let update = m[row + n * k] * m[k + n * col] / pivot;
                    m[row + n * col] -= update;
                }
            }
        }
    }
}

/// Every stored coupled-sector matrix of `(space, data)`.
fn sector_matrices<R, D: FactorScalar>(
    space: &BoundDynamicFusionMapSpace<R>,
    data: &[D],
) -> Vec<(SectorId, SectorMatrix)> {
    let space = space.space();
    let regions = space
        .structure()
        .coupled_sector_regions(space.nout())
        .unwrap()
        .expect("fresh layouts are coupled-sector tilings");
    regions
        .iter()
        .map(|region| {
            let data = data[region.range()]
                .iter()
                .map(|value| value.widen_complex())
                .collect();
            let (rows, cols) = (region.rows(), region.cols());
            (region.coupled(), SectorMatrix { rows, cols, data })
        })
        .collect()
}

/// Checks the polar laws sector by sector against `source`: a left polar
/// `A = W P` with `Wᴴ W = 1`, or a right polar `A = P Wh` with
/// `Wh Whᴴ = 1`; in both `P` is Hermitian positive definite (the inputs
/// have full rank). Returns how many sectors were checked.
fn assert_polar_laws<D: FactorScalar>(
    source: &BoundTensorMap<SU2FusionRule, D>,
    w: &BoundDynFactor<SU2FusionRule, D>,
    p: &BoundDynFactor<SU2FusionRule, D>,
    left: bool,
) -> usize {
    let a = sector_matrices(source.space(), source.data());
    let p = sector_matrices(p.space(), p.data());
    let mut checked = 0;
    for (sector, w) in sector_matrices(w.space(), w.data()) {
        let find = |matrices: &[(SectorId, SectorMatrix)]| {
            matrices.iter().position(|(other, _)| *other == sector)
        };
        let a = &a[find(&a).expect("every W sector is stored in A")].1;
        let p = &p[find(&p).expect("every W sector has a P block")].1;
        let what = format!("{sector:?}");
        if left {
            w.mul(p).assert_close(a, &format!("W P = A in {what}"));
            w.adjoint()
                .mul(&w)
                .assert_close(&SectorMatrix::identity(w.cols), &format!("Wᴴ W in {what}"));
        } else {
            p.mul(&w).assert_close(a, &format!("P Wh = A in {what}"));
            w.mul(&w.adjoint()).assert_close(
                &SectorMatrix::identity(w.rows),
                &format!("Wh Whᴴ in {what}"),
            );
        }
        p.assert_close(&p.adjoint(), &format!("P = Pᴴ in {what}"));
        p.assert_positive_definite(&format!("P in {what}"));
        checked += 1;
    }
    checked
}

fn check_polar_laws_on_su2<D: FactorScalar>(value: fn(f64, f64) -> D) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    type Mf = symtn_core::MultiplicityFreeAdmissionMode;
    let dense_factor = |output| match output {
        FactorOutput::Dense(factor) => factor,
        FactorOutput::Diagonal { .. } => panic!("a dense source has dense polar factors"),
    };
    fn parent<D>(
        bound: &BoundTensorMap<SU2FusionRule, D>,
    ) -> BoundDynamicTensorRef<'_, SU2FusionRule, D> {
        BoundDynamicTensorRef::try_new(bound.space(), bound.data()).unwrap()
    }
    let source = |bound| FactorSource::Dense(parent(bound));
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let (tall, wide) = (su2_polar_input(true, value), su2_polar_input(false, value));

    let LeftPolar { w, p } =
        left_polar_from_source::<Mf, _, _, _, _>(engine, &mut dense, source(&tall)).unwrap();
    assert_eq!(
        assert_polar_laws(&tall, &dense_factor(w), &dense_factor(p), true),
        3
    );
    let RightPolar { p, wh } =
        right_polar_from_source::<Mf, _, _, _, _>(engine, &mut dense, source(&wide)).unwrap();
    assert_eq!(
        assert_polar_laws(&wide, &dense_factor(wh), &dense_factor(p), false),
        3
    );

    // The lazy-adjoint seams factor the dense parent in the opposite
    // direction: the view `wideᴴ` is tall, so its left polar is the
    // parent's right polar, and `tallᴴ`'s right polar is the parent's left.
    let RightPolar { p, wh } =
        left_polar_adjoint_from_parent::<Mf, _, _, _, _>(engine, &mut dense, &parent(&wide))
            .unwrap();
    assert_eq!(assert_polar_laws(&wide, &wh, &p, false), 3);
    let LeftPolar { w, p } =
        right_polar_adjoint_from_parent::<Mf, _, _, _, _>(engine, &mut dense, &parent(&tall))
            .unwrap();
    assert_eq!(assert_polar_laws(&tall, &w, &p, true), 3);

    // A view whose shape forbids the requested direction is refused under
    // the view's operation name, not the parent's.
    for (left, parent_bound) in [(true, &tall), (false, &wide)] {
        let error = if left {
            left_polar_adjoint_from_parent::<Mf, _, _, _, _>(
                engine,
                &mut dense,
                &parent(parent_bound),
            )
            .map(drop)
        } else {
            right_polar_adjoint_from_parent::<Mf, _, _, _, _>(
                engine,
                &mut dense,
                &parent(parent_bound),
            )
            .map(drop)
        }
        .unwrap_err();
        let operation = if left { "left_polar" } else { "right_polar" };
        assert!(matches!(
            error,
            OperationError::InvalidArgument { message } if message.starts_with(operation)
        ));
    }
}

#[test]
fn polar_factors_are_isometric_and_hermitian_positive_on_both_sides() {
    // What: on non-Abelian SU(2) with multi-tree sectors on both sides,
    // both polar directions and both lazy-adjoint seams return
    // `A = W P` / `A = P Wh` with an isometric `W` and a Hermitian positive
    // definite `P`, for real and complex data.
    check_polar_laws_on_su2::<f64>(|re, _| re);
    check_polar_laws_on_su2(Complex64::new);
}

#[test]
fn polar_validates_every_sector_before_direct_or_fallback_svd_execution() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a later invalid sector prevents SVD of an earlier valid sector on both layouts.
    let rule = Z2FusionRule;
    let direct = mixed_rectangular_tensor((4, 2), (1, 3));
    let direct_bound = bound_tensor(Arc::new(rule), &direct);
    assert!(
        crate::factorize::compact_factor_plan_for_test(direct_bound.space())
            .unwrap()
            .is_some()
    );
    let mut dense = ScriptedExecutor::<SvdCallSpy>::default();
    let direct_error = left_polar(engine, &mut dense, &direct_bound.as_ref()).unwrap_err();
    assert!(matches!(
        direct_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.counts().svd, 0);

    let fallback_source = mixed_rectangular_tensor((2, 4), (3, 1));
    let fallback_bound = bound_tensor(Arc::new(rule), &fallback_source);
    let fallback_space = fallback_bound.space().adjoint_view().unwrap();
    assert!(
        crate::factorize::compact_factor_plan_for_test(&fallback_space)
            .unwrap()
            .is_none()
    );
    let fallback_input =
        BoundDynamicTensorRef::try_new(&fallback_space, fallback_bound.data()).unwrap();
    let mut dense = ScriptedExecutor::<SvdCallSpy>::default();
    let fallback_error = left_polar_dyn(engine, &mut dense, &fallback_input).unwrap_err();
    assert!(matches!(
        fallback_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.counts().svd, 0);
}
