//! TensorKit scaling parity for non-finite data (#1438).
//!
//! Oracle: outputs observed in Julia 1.11 with TensorKit main f87ca7fe
//! (version 0.17.1), TensorOperations 5.6.2 and VectorInterface 0.6.0, on
//! `V ⊗ V ← V` tensors whose
//! data cycles through `±Inf`, NaN, `-0.0` and finite values:
//!
//! - `permute!(tdst, tsrc, ((1, 3), (2,)), zero(T), T(β))` equals
//!   `VectorInterface.scale.(tdst, T(β))` elementwise for U(1), SU(2) and
//!   fZ₂, `Float64` and `ComplexF64`, `β ∈ {1, 0, 2}`: zeros for `β = 0`
//!   and `tdst * T(β)` otherwise (NaN kept);
//! - `trace_permute!(tdst, tsrc, ((1,), ()), ((2,), (3,)), zero(T), T(β))`
//!   equals `scale.(tdst, T(β))` for the same rules, dtypes and `β ∈ {1, 0}`;
//! - `trace_permute!(d, A, ((), ()), ((1,), (2,)), T(α), 0.0)` for `A ∈ V ← V`
//!   with a single sector of degeneracy 2 and `1e308` (or `1e308 + 1e308im`)
//!   on the diagonal: U(1) charge 1, `α = 0.5` → `1.0e308`; SU(2) spin ½,
//!   `α = 0.25` (quantum dimension 2) → `1.0000000000000002e308`; fZ₂ odd,
//!   `α = 0.5` (twist −1) → `-1.0e308`, with the same value in the imaginary
//!   part for `ComplexF64`. Scaling the traced sum instead overflows to
//!   `±Inf` (or NaN for the complex products).
//!
//! The destination fixtures carry no `-0.0`, so a zero contribution cannot
//! change the expected value through the sign of a zero.

use super::*;
use std::sync::Arc;

trait Payload: Copy + std::fmt::Debug + std::ops::Mul<Output = Self> {
    fn from_parts(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
    fn from_real(value: f64) -> Self {
        Self::from_parts(value, 0.0)
    }
}

impl Payload for f64 {
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

impl Payload for Complex64 {
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

const SOURCE: [(f64, f64); 6] = [
    (f64::INFINITY, 1.0),
    (-0.0, -0.0),
    (f64::NAN, 2.5),
    (1.5, f64::NEG_INFINITY),
    (-1.25, 0.5),
    (2.0, -0.0),
];
const DESTINATION: [(f64, f64); 4] = [(1.0, -2.0), (-4.0, 0.5), (f64::NAN, 1.0), (3.0, -7.0)];

fn cycle<D: Payload>(values: &[(f64, f64)], len: usize) -> Vec<D> {
    (0..len)
        .map(|index| {
            let (re, im) = values[index % values.len()];
            D::from_parts(re, im)
        })
        .collect()
}

/// VectorInterface's `scale(x, β)` for the `β ∈ {0, 1, 2}` used here, with
/// `β` in the payload type as symtn takes it (`T(β)` in Julia, so a complex
/// payload is multiplied by `β + 0im`), and an exact `β = 1` as TensorKit's
/// `One()`, which symtn's exact-one convention (#1398) stands for.
fn tensorkit_scale<D: Payload>(value: D, beta: f64) -> (f64, f64) {
    if beta == 0.0 {
        (0.0, 0.0)
    } else if beta == 1.0 {
        value.parts()
    } else {
        (value * D::from_real(beta)).parts()
    }
}

/// Equal values, with NaN equal to NaN and `0.0 == -0.0`.
fn assert_same<D: Payload>(what: &str, got: &[D], want: &[(f64, f64)]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let same = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
    for (index, (value, &(re, im))) in got.iter().zip(want).enumerate() {
        let (got_re, got_im) = value.parts();
        assert!(
            same(got_re, re) && same(got_im, im),
            "{what}: element {index} is {value:?}, TensorKit gives ({re}, {im})"
        );
    }
}

fn legs<R>(provider: &Arc<R>, sectors: &[SectorId]) -> BoundDynamicFusionMapSpace<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 2)), false);
    homspace_space(
        provider,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
}

fn homspace_space<R>(
    provider: &Arc<R>,
    homspace: FusionTreeHomSpace,
) -> BoundDynamicFusionMapSpace<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        homspace,
    )
    .unwrap()
}

/// `permute!` and `trace_permute!` with `α = 0` over a non-finite source.
fn check_zero_alpha<R, D>(label: &str, provider: Arc<R>, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    D: Payload + DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let src = legs(&provider, sectors);
    let src_data: Vec<D> = cycle(&SOURCE, src.space().required_len().unwrap());
    let alpha = D::from_real(0.0);

    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let dst = src.transformed_multiplicity_free(&operation).unwrap();
    let dst_len = dst.space().required_len().unwrap();
    let initial: Vec<D> = cycle(&DESTINATION, dst_len);
    let mut context = TreeTransformExecutionContext::<D, R::Key, f64>::default();
    for beta in [1.0, 0.0, 2.0] {
        let mut got = initial.clone();
        context
            .tree_transform_dyn_into(
                provider.as_ref(),
                operation.clone(),
                dst.space().structure(),
                src.space().structure(),
                &mut got,
                &src_data,
                alpha,
                D::from_real(beta),
            )
            .unwrap();
        let want: Vec<_> = initial
            .iter()
            .map(|&value| tensorkit_scale(value, beta))
            .collect();
        assert_same(&format!("{label} permute! β = {beta}"), &got, &want);
    }
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 2)), false);
    let trace_dst = homspace_space(
        &provider,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([]),
        ),
    );
    let axes = TensorTraceAxisSpec::new(&[0], &[1], &[2]);
    let trace_len = trace_dst.space().required_len().unwrap();
    let initial: Vec<D> = cycle(&DESTINATION, trace_len);
    for beta in [1.0, 0.0] {
        let mut got = initial.clone();
        tensortrace_fusion_dyn_into(
            &trace_dst,
            &mut got,
            &src,
            &src_data,
            axes,
            alpha,
            D::from_real(beta),
        )
        .unwrap();
        let want: Vec<_> = initial
            .iter()
            .map(|&value| tensorkit_scale(value, beta))
            .collect();
        assert_same(&format!("{label} trace_permute! β = {beta}"), &got, &want);
    }
    let owned = tensortrace_fusion_dyn_owned(&trace_dst, &src, &src_data, axes, alpha).unwrap();
    assert_same(
        &format!("{label} owned trace"),
        &owned,
        &vec![(0.0, 0.0); trace_len],
    );
}

#[test]
fn zero_alpha_transform_and_trace_match_tensorkit() {
    let u1 = [0, 1, -1].map(|charge| U1Irrep::new(charge).sector_id());
    let su2 = [0, 1, 2].map(|twice| SU2Irrep::from_twice_spin(twice).sector_id());
    let fz2 = [SectorId::new(0), SectorId::new(1)];
    check_zero_alpha::<_, f64>("U(1) f64", Arc::new(U1FusionRule), &u1);
    check_zero_alpha::<_, Complex64>("U(1) c64", Arc::new(U1FusionRule), &u1);
    check_zero_alpha::<_, f64>("SU(2) f64", Arc::new(SU2FusionRule), &su2);
    check_zero_alpha::<_, Complex64>("SU(2) c64", Arc::new(SU2FusionRule), &su2);
    check_zero_alpha::<_, f64>("fZ2 f64", Arc::new(FermionParityFusionRule), &fz2);
    check_zero_alpha::<_, Complex64>("fZ2 c64", Arc::new(FermionParityFusionRule), &fz2);
}

/// The full trace of `A ∈ V ← V` with `x` on the diagonal of its one block.
fn check_trace_overflow<R, D>(
    label: &str,
    provider: Arc<R>,
    sector: SectorId,
    alpha: f64,
    x: D,
    want: D,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: Payload
        + DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + crate::test_numerics::numerics::Numeric,
{
    let leg = || SectorLeg::new([(sector, 2)], false);
    let src = homspace_space(
        &provider,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    );
    let zero = D::from_real(0.0);
    let src_data = vec![x, zero, zero, x];
    let dst = homspace_space(
        &provider,
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
    );
    let axes = TensorTraceAxisSpec::new(&[], &[0], &[1]);
    let alpha = D::from_real(alpha);
    let mut got = vec![D::from_real(7.0)];
    tensortrace_fusion_dyn_into(&dst, &mut got, &src, &src_data, axes, alpha, zero).unwrap();
    // Two traced elements reach the one output entry.
    crate::test_numerics::numerics::assert_close(label, got[0], want, 2);
    let owned = tensortrace_fusion_dyn_owned(&dst, &src, &src_data, axes, alpha).unwrap();
    crate::test_numerics::numerics::assert_close(label, owned[0], want, 2);
}

#[test]
fn trace_scales_each_element_before_the_sum_as_tensorkit() {
    let big = Complex64::new(1e308, 1e308);
    let u1 = U1Irrep::new(1).sector_id();
    let spin_half = SU2Irrep::from_twice_spin(1).sector_id();
    let odd = SectorId::new(1);
    let su2 = 1.000_000_000_000_000_2e308;
    check_trace_overflow("U(1) f64", Arc::new(U1FusionRule), u1, 0.5, 1e308, 1e308);
    check_trace_overflow("U(1) c64", Arc::new(U1FusionRule), u1, 0.5, big, big);
    check_trace_overflow(
        "SU(2) f64",
        Arc::new(SU2FusionRule),
        spin_half,
        0.25,
        1e308,
        su2,
    );
    check_trace_overflow(
        "SU(2) c64",
        Arc::new(SU2FusionRule),
        spin_half,
        0.25,
        big,
        Complex64::new(su2, su2),
    );
    check_trace_overflow(
        "fZ2 f64",
        Arc::new(FermionParityFusionRule),
        odd,
        0.5,
        1e308,
        -1e308,
    );
    check_trace_overflow(
        "fZ2 c64",
        Arc::new(FermionParityFusionRule),
        odd,
        0.5,
        big,
        -big,
    );
}

/// `W ← V` over `sectors` with degeneracy 2 on every leg.
fn matrix_space<R>(
    provider: &Arc<R>,
    sectors: &[SectorId],
    dual: bool,
) -> BoundDynamicFusionMapSpace<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let leg = |dual| SectorLeg::new(sectors.iter().map(|&sector| (sector, 2)), dual);
    homspace_space(
        provider,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(dual)]),
            FusionProductSpace::new([leg(false)]),
        ),
    )
}

/// `tensorcontract!(C, A, pA, conjA, B, ((1,), (2,)), false, pAB, α, β)`
/// with `α = 0` over a non-finite `A`, and with `β = 0` over a NaN-bearing
/// `C`. TensorKit (observed in the environment of the module docs) gives
/// `scale.(C, β)` elementwise for `α = 0` — `A·B` is never formed, as BLAS
/// `gemm` skips it at `α = 0` — and for `β = 0` the same values as over an
/// all-zero `C`, since `C` is never read. Both hold for U(1), SU(2) and fZ₂,
/// `Float64` and `ComplexF64`, for `mul!` and for three contractions, one per
/// symtn route:
///
/// - direct, `A ∈ V ⊗ V ← V`, `B ∈ V ← V`, `pA = ((1, 2), (3,))` (`Core`);
/// - recoupled, `B ∈ V' ← V`, `pA = ((1, 3), (2,))` (`DynamicTree`);
/// - conjugated, `A, B ∈ V ← V`, `pA = ((2,), (1,))`, `conjA = true`
///   (`Core` over the lazy adjoint `A†`).
fn check_zero_scale_contraction<R, D>(label: &str, provider: Arc<R>, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    D: Payload + DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    const FINITE: [(f64, f64); 4] = [(0.5, 0.25), (-1.0, 2.0), (0.75, -0.5), (1.5, 1.0)];
    let square = || matrix_space(&provider, sectors, false);
    let cases = [
        // (route, lhs, rhs, contracted lhs axis, conjA, Core)
        ("direct", legs(&provider, sectors), square(), 2, false, true),
        (
            "recoupled",
            legs(&provider, sectors),
            matrix_space(&provider, sectors, true),
            1,
            false,
            false,
        ),
        ("conjugated", square(), square(), 0, true, true),
    ];
    for (route, lhs, rhs, contracted, conjugate, expected_route) in cases {
        // `A†` has `A`'s space here (no dual legs), so the contraction of `A`'s
        // domain leg gives the conjugated destination too.
        let dst_axis = if conjugate { 1 } else { contracted };
        let dst =
            BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lhs, &rhs, &[dst_axis], &[0])
                .unwrap();
        let lhs_len = lhs.space().required_len().unwrap();
        let rhs_data: Vec<D> = cycle(&FINITE, rhs.space().required_len().unwrap());
        let dst_len = dst.space().required_len().unwrap();
        let initial: Vec<D> = cycle(&DESTINATION, dst_len);
        // A set flag names `A`'s lazy adjoint, whose legs the axes index.
        let lhs_axes = [dst_axis];
        let axes = TensorContractSpec::with_default_output_order_and_conjugation(
            &lhs_axes,
            &[0],
            conjugate,
            false,
        );
        let mut context = TensorContractFusionExecutionContext::<D, R::Key>::default();
        let mut contract = |dst_data: &mut [D], lhs_data: &[D], alpha: f64, beta: f64| {
            context
                .tensorcontract_prelowered_into(
                    &dst,
                    dst_data,
                    if conjugate {
                        crate::FusionOperand::adjoint(lhs.space())
                    } else {
                        crate::FusionOperand::direct(lhs.space())
                    },
                    lhs_data,
                    crate::FusionOperand::direct(rhs.space()),
                    &rhs_data,
                    axes,
                    D::from_real(alpha),
                    crate::ContractDestinationInit::Axpby(D::from_real(beta)),
                )
                .unwrap();
            assert_eq!(
                context.last_resolution_is_core(),
                expected_route,
                "{label} {route}: route"
            );
        };

        let non_finite: Vec<D> = cycle(&SOURCE, lhs_len);
        for beta in [1.0, 2.0, 0.0] {
            let mut got = initial.clone();
            contract(&mut got, &non_finite, 0.0, beta);
            let want: Vec<_> = initial
                .iter()
                .map(|&value| tensorkit_scale(value, beta))
                .collect();
            assert_same(&format!("{label} {route} α = 0, β = {beta}"), &got, &want);
        }

        let finite: Vec<D> = cycle(&FINITE, lhs_len);
        for alpha in [1.0, 0.5] {
            let mut over_zero = vec![D::from_real(0.0); dst_len];
            contract(&mut over_zero, &finite, alpha, 0.0);
            assert!(
                over_zero.iter().any(|value| value.parts() != (0.0, 0.0)),
                "{label} {route}: the β = 0 fixture must produce a nonzero product"
            );
            let mut over_nan = initial.clone();
            contract(&mut over_nan, &finite, alpha, 0.0);
            let want: Vec<_> = over_zero.iter().map(|value| value.parts()).collect();
            assert_same(
                &format!("{label} {route} α = {alpha}, β = 0 over NaN"),
                &over_nan,
                &want,
            );
        }
    }
}

#[test]
fn zero_scale_contraction_matches_tensorkit() {
    let u1 = [0, 1, -1].map(|charge| U1Irrep::new(charge).sector_id());
    let su2 = [0, 1, 2].map(|twice| SU2Irrep::from_twice_spin(twice).sector_id());
    let fz2 = [SectorId::new(0), SectorId::new(1)];
    check_zero_scale_contraction::<_, f64>("U(1) f64", Arc::new(U1FusionRule), &u1);
    check_zero_scale_contraction::<_, Complex64>("U(1) c64", Arc::new(U1FusionRule), &u1);
    check_zero_scale_contraction::<_, f64>("SU(2) f64", Arc::new(SU2FusionRule), &su2);
    check_zero_scale_contraction::<_, Complex64>("SU(2) c64", Arc::new(SU2FusionRule), &su2);
    check_zero_scale_contraction::<_, f64>("fZ2 f64", Arc::new(FermionParityFusionRule), &fz2);
    check_zero_scale_contraction::<_, Complex64>(
        "fZ2 c64",
        Arc::new(FermionParityFusionRule),
        &fz2,
    );
}
