//! A symmetric braiding that certifies the trivial associator gauge but has
//! non-symmetric R (#2178): permute, braid and contract must apply the Artin
//! value `R(source-left, source-right)` for every inverted source pair.

use super::*;

/// Z3×Z3, trivial F, `R(a, b) = ω^{a₁b₂ − a₂b₁}` with `ω = e^{2πi/3}`;
/// sector id `a₁ + 3·a₂`.
#[derive(Clone, Copy, Debug)]
struct AlternatingZ3SquaredRule;

impl AlternatingZ3SquaredRule {
    fn components(sector: SectorId) -> (usize, usize) {
        assert!(sector.id() < 9, "Z3×Z3 fixture sector must be in 0..9");
        (sector.id() % 3, sector.id() / 3)
    }

    fn id(first: usize, second: usize) -> usize {
        first % 3 + 3 * (second % 3)
    }
}

fn omega_power(exponent: usize) -> Complex64 {
    Complex64::from_polar(
        1.0,
        2.0 * std::f64::consts::PI * (exponent % 3) as f64 / 3.0,
    )
}

impl FusionRule for AlternatingZ3SquaredRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (a1, a2) = Self::components(sector);
        SectorId::new(Self::id(3 - a1, 3 - a2))
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (a1, a2) = Self::components(left);
        let (b1, b2) = Self::components(right);
        vec![SectorId::new(Self::id(a1 + b1, a2 + b2))].into()
    }
}

impl MultiplicityFreeFusionRule for AlternatingZ3SquaredRule {}

impl MultiplicityFreeFusionSymbols for AlternatingZ3SquaredRule {
    type Scalar = Complex64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        let (a1, a2) = Self::components(left);
        let (b1, b2) = Self::components(right);
        omega_power(a1 * b2 + 2 * a2 * b1)
    }
}

impl MultiplicityFreeRigidSymbols for AlternatingZ3SquaredRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    /// `θ_a = R(a, a) = ω^0`.
    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
}

// v = (1,2), w = (1,1), x = (2,1). Hand values of the exponent of ω:
// e(v,w) = 1·1 − 2·1 ≡ 2, e(v,x) = 1·1 − 2·2 ≡ 0, e(w,x) = 1·1 − 1·2 ≡ 2.
const V: usize = 1 + 3 * 2;
const W: usize = 1 + 3;
const X: usize = 2 + 3;
/// v ⊗ w = (2, 0); v ⊗ w ⊗ x = (1, 1) = w.
const VW: usize = 2;
const VWX: usize = W;

fn single_block<const NOUT: usize, const NIN: usize>(
    codomain: [usize; NOUT],
    domain: [usize; NIN],
    value: Complex64,
) -> TestTensor<Complex64> {
    let rule = AlternatingZ3SquaredRule;
    let space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::from_sector_ids(codomain.map(|s| (s, 1)), domain.map(|s| (s, 1))),
    )
    .unwrap();
    TestTensor::from_vec_with_fusion_space(vec![value], space).unwrap()
}

fn assert_single_value(tensor: &TestTensor<Complex64>, expected: Complex64) {
    assert_eq!(tensor.data().len(), 1);
    let actual = tensor.data()[0];
    assert!(
        (actual - expected).norm() < 1.0e-13,
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn alternating_bicharacter_permute_and_braid_apply_source_order_r_symbol() {
    let rule = AlternatingZ3SquaredRule;
    let value = Complex64::new(2.0, -0.5);

    // What: (v, w) → (w, v) carries R(v, w) = ω², by permute and by braid in
    // both directions (symmetric braiding: levels are irrelevant).
    let operations = [
        TreeTransformOperation::permute([1, 0], [2]),
        TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
        TreeTransformOperation::braid([1, 0], [2], [1, 0], [2]),
    ];
    for operation in operations {
        let src = single_block([V, W], [VW], value);
        let mut dst = single_block([W, V], [VW], Complex64::zero());
        eager_tree_transform_into(
            &rule,
            operation,
            &mut dst,
            &src,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
        assert_single_value(&dst, omega_power(2) * value);
    }

    // What: rank-3 reversal (v, w, x) → (x, w, v) inverts all three source
    // pairs: ω^{2 + 0 + 2} = ω, again for permute and both braid directions.
    let operations = [
        TreeTransformOperation::permute([2, 1, 0], [3]),
        TreeTransformOperation::braid([2, 1, 0], [3], [0, 1, 2], [3]),
        TreeTransformOperation::braid([2, 1, 0], [3], [2, 1, 0], [3]),
    ];
    for operation in operations {
        let src = single_block([V, W, X], [VWX], value);
        let mut dst = single_block([X, W, V], [VWX], Complex64::zero());
        eager_tree_transform_into(
            &rule,
            operation,
            &mut dst,
            &src,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
        assert_single_value(&dst, omega_power(1) * value);
    }
}

#[test]
fn alternating_bicharacter_contract_into_swapped_output_applies_source_order_r_symbol() {
    // What: contracting A: v ⊗ w ← c with B: c ← c into the output order
    // (w, v) ← c equals ω² · A·B, the braid of the source pair (v, w).
    let rule = AlternatingZ3SquaredRule;
    let lhs = single_block([V, W], [VW], Complex64::new(1.5, 0.25));
    let rhs = single_block([VW], [VW], Complex64::new(-0.75, 2.0));
    let mut dst = single_block([W, V], [VW], Complex64::zero());
    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        symtn_operations::TensorContractSpec::new(
            &[2],
            &[0],
            symtn_operations::OutputAxisOrder::Axes(&[1, 0, 2]),
        ),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    assert_single_value(
        &dst,
        omega_power(2) * Complex64::new(1.5, 0.25) * Complex64::new(-0.75, 2.0),
    );
}
