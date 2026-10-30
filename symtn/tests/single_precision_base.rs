//! Base-family oracle for the single-precision payloads (#1315).
//!
//! Oracle, tolerance and fixtures come from `single_precision_oracle`, the
//! module this suite shares with the factorization suite (#1324); its rationale
//! is documented there.
//!
//! Both dtypes are driven from one `Runtime`. That is deliberate: the
//! single-precision lanes, their contract workspaces and their coefficient
//! scratch are created on first use next to the double-precision ones, and a
//! `f32` tensor must never read the `f64` lane's converted recoupling matrix
//! for the same structure identity.

use crate::single_precision_oracle;

use num_complex::{Complex32, Complex64};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, Direction, Side};
use symtn::typed::{LegSelection, SectorSpectrum, TensorMap};

use single_precision_oracle::{
    assert_payloads_agree, assert_scalars_agree, draw_parts, fermion_su2_leg, one as wide_of_one,
    runtime, u1_leg, Parts,
};

/// Everything below is generated once per admitted single-precision dtype.
macro_rules! base_suite {
    ($suite:ident, $narrow:ty, $wide:ty) => {
        mod $suite {
            use super::*;

            #[test]
            fn arithmetic_and_reductions_match_the_widened_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
                let runtime = runtime();
                let leg = u1_leg();
                let (a, wa) = twin!(engine, &runtime, $narrow, $wide, [&leg, &leg], [&leg], 9_001);
                let (b, wb) = twin!(engine, &runtime, $narrow, $wide, [&leg, &leg], [&leg], 9_002);
                let terms = wa.dense_data().unwrap().len();

                assert_payloads_agree("input", a.dense_data().unwrap(), wa.dense_data().unwrap(), 1);

                let scale = <$narrow as Parts>::parts(0.5, -0.25);
                let wide_scale = <$wide as Parts>::parts(0.5, -0.25);
                assert_payloads_agree(
                    "scale",
                    a.scale(engine, scale).unwrap().dense_data().unwrap(),
                    wa.scale(engine, wide_scale).unwrap().dense_data().unwrap(),
                    1,
                );
                assert_payloads_agree(
                    "add",
                    a.axpby(engine, scale, &b, wide_of_one::<$narrow>()).unwrap().dense_data().unwrap(),
                    wa.axpby(engine, wide_scale, &wb, wide_of_one::<$wide>())
                        .unwrap()
                        .dense_data().unwrap(),
                    2,
                );
                assert_payloads_agree(
                    "zeros_like",
                    a.zeros_like(engine).unwrap().dense_data().unwrap(),
                    wa.zeros_like(engine).unwrap().dense_data().unwrap(),
                    1,
                );

                assert_scalars_agree(
                    "norm",
                    Complex64::new(a.norm(engine, 2.0).unwrap(), 0.0),
                    Complex64::new(wa.norm(engine, 2.0).unwrap(), 0.0),
                    terms,
                );
                assert_scalars_agree(
                    "norm(Inf)",
                    Complex64::new(a.norm(engine, f64::INFINITY).unwrap(), 0.0),
                    Complex64::new(wa.norm(engine, f64::INFINITY).unwrap(), 0.0),
                    1,
                );
                assert_scalars_agree(
                    "norm(3)",
                    Complex64::new(a.norm(engine, 3.0).unwrap(), 0.0),
                    Complex64::new(wa.norm(engine, 3.0).unwrap(), 0.0),
                    terms,
                );
                assert_scalars_agree(
                    "inner",
                    a.inner(engine, &b).unwrap().wide(),
                    wa.inner(engine, &wb).unwrap().wide(),
                    terms,
                );
let a_adjoint = &a.adjoint(engine).unwrap();

                let square = a.compose(engine, a_adjoint).unwrap();
                let wa_adjoint = &wa.adjoint(engine).unwrap();
                let wide_square = wa.compose(engine, wa_adjoint).unwrap();
                assert_scalars_agree(
                    "tr",
                    square.tr(engine).unwrap().wide(),
                    wide_square.tr(engine).unwrap().wide(),
                    terms,
                );
            }

            #[test]
            fn structural_transforms_match_the_widened_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
                let runtime = runtime();
                let leg = fermion_su2_leg();
                let (a, wa) = twin!(engine, &runtime, $narrow, $wide, [&leg, &leg], [&leg, &leg], 9_101);
                let terms = wa.dense_data().unwrap().len();

                for (what, got, expected) in [
                    (
                        "permute",
                        a.permute(engine, &[1, 2], &[3, 0]).unwrap(),
                        wa.permute(engine, &[1, 2], &[3, 0]).unwrap(),
                    ),
                    (
                        "braid",
                        a.braid(engine, &[1, 2], &[3, 0], &[0, 1, 2, 3]).unwrap(),
                        wa.braid(engine, &[1, 2], &[3, 0], &[0, 1, 2, 3]).unwrap(),
                    ),
                    (
                        "transpose",
                        a.transpose(engine, &[3, 2], &[1, 0]).unwrap(),
                        wa.transpose(engine, &[3, 2], &[1, 0]).unwrap(),
                    ),
                    (
                        "repartition",
                        a.repartition(engine, 1).unwrap(),
                        wa.repartition(engine, 1).unwrap(),
                    ),
                    ("twist", a.twist(engine, &[0, 2], Direction::Forward).unwrap(), wa.twist(engine, &[0, 2], Direction::Forward).unwrap()),
                    (
                        "inverse twist",
                        a.twist(engine, &[1], Direction::Inverse).unwrap(),
                        wa.twist(engine, &[1], Direction::Inverse).unwrap(),
                    ),
                    ("flip", a.flip(engine, &[0], Direction::Forward).unwrap(), wa.flip(engine, &[0], Direction::Forward).unwrap()),
                    (
                        "adjoint (lazy)",
                        a.adjoint(engine).unwrap(),
                        wa.adjoint(engine).unwrap(),
                    ),
                    (
                        "adjoint materialized",
                        a.adjoint(engine).unwrap().permute(engine, &[0, 1], &[2, 3]).unwrap(),
                        wa.adjoint(engine).unwrap().permute(engine, &[0, 1], &[2, 3]).unwrap(),
                    ),
                    (
                        "trace_pairs",
                        a.trace_pairs(engine, &[(0, 3)]).unwrap(),
                        wa.trace_pairs(engine, &[(0, 3)]).unwrap(),
                    ),
                ] {
                    assert_payloads_agree(what, got.materialize(engine).unwrap().dense_data().unwrap(), expected.materialize(engine).unwrap().dense_data().unwrap(), terms);
                    assert_eq!(
                        got.subblock_count(),
                        expected.subblock_count(),
                        "{what}: block structure diverged between the twins"
                    );
                }

                // Negative control: the fixture is not invariant under the
                // transforms above, so agreement is evidence and not an
                // accident of a payload the operations leave alone.
                let moved = a.permute(engine, &[1, 2], &[3, 0]).unwrap();
                let largest = wa
                    .dense_data().unwrap()
                    .iter()
                    .zip(moved.dense_data().unwrap())
                    .map(|(&expected, &got)| (got.wide() - expected.wide()).norm())
                    .fold(0.0f64, f64::max);
                assert!(
                    largest > 1e-3,
                    "the SU(2)/fermionic fixture must be moved by permute, largest change {largest:e}"
                );
                assert_ne!(
                    a.twist(engine, &[0, 2], Direction::Forward).unwrap().dense_data().unwrap(),
                    a.dense_data().unwrap(),
                    "the fZ2 twist must change the payload of an odd-sector fixture"
                );

                // Round trip: the inverse permutation restores the input to
                // within the module tolerance. The recoupling of a
                // permutation and its inverse is the identity matrix, but the
                // two transforms are still executed, so the payload makes a
                // round trip through the dense kernel rather than being
                // returned untouched.
                let restored = a.permute(engine, &[1, 2], &[3, 0]).unwrap();
                assert_payloads_agree(
                    "permute round trip",
                    restored.permute(engine, &[3, 0], &[1, 2]).unwrap().dense_data().unwrap(),
                    a.dense_data().unwrap(),
                    terms,
                );
            }

            #[test]
            fn products_and_contractions_match_the_widened_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
                let runtime = runtime();
                let leg = fermion_su2_leg();
                let (a, wa) = twin!(engine, &runtime, $narrow, $wide, [&leg], [&leg], 9_201);
                let (b, wb) = twin!(engine, &runtime, $narrow, $wide, [&leg], [&leg], 9_202);
                let terms = wa.dense_data().unwrap().len();

                for (what, got, expected) in [
                    (
                        "contract",
                        a.contract(engine, &b, &ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] }).unwrap(),
                        wa.contract(engine, &wb, &ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] }).unwrap(),
                    ),
                    ("compose", a.compose(engine, &b).unwrap(), wa.compose(engine, &wb).unwrap()),
                    ("otimes", a.otimes(engine, &b).unwrap(), wa.otimes(engine, &wb).unwrap()),
                    (
                        "cat domain",
                        a.cat(engine, &b, Side::Domain).unwrap(),
                        wa.cat(engine, &wb, Side::Domain).unwrap(),
                    ),
                    (
                        "cat codomain",
                        a.cat(engine, &b, Side::Codomain).unwrap(),
                        wa.cat(engine, &wb, Side::Codomain).unwrap(),
                    ),
                ] {
                    assert_payloads_agree(what, got.dense_data().unwrap(), expected.dense_data().unwrap(), terms);
                }
            }

            #[test]
            fn rank_five_multi_block_contraction_matches_the_widened_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
                let runtime = runtime();
                let leg = u1_leg();
                let (a, wa) = twin!(engine,
                    &runtime,
                    $narrow,
                    $wide,
                    [&leg, &leg, &leg],
                    [&leg, &leg],
                    9_301
                );
                assert_eq!(a.rank(), 5);
                assert!(a.subblock_count() > 1, "the fixture must be multi-block");
                let (b, wb) = twin!(engine, &runtime, $narrow, $wide, [&leg, &leg], [&leg], 9_302);
                let terms = wa.dense_data().unwrap().len().max(wb.dense_data().unwrap().len());

                assert_payloads_agree(
                    "rank-5 contract",
                    a.contract(engine, &b, &ContractSpec { lhs: &[3, 4], rhs: &[0, 1], codomain: &[0, 1, 2], domain: &[3] })
                        .unwrap()
                        .dense_data().unwrap(),
                    wa.contract(engine, &wb, &ContractSpec { lhs: &[3, 4], rhs: &[0, 1], codomain: &[0, 1, 2], domain: &[3] })
                        .unwrap()
                        .dense_data().unwrap(),
                    terms,
                );
                assert_payloads_agree(
                    "rank-5 permute",
                    a.permute(engine, &[4, 0, 2], &[1, 3]).unwrap().dense_data().unwrap(),
                    wa.permute(engine, &[4, 0, 2], &[1, 3]).unwrap().dense_data().unwrap(),
                    terms,
                );
            }

            #[test]
            fn leg_restriction_and_diagonal_views_match_the_widened_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
                let runtime = runtime();
                let leg = u1_leg();
                let (a, wa) = twin!(engine, &runtime, $narrow, $wide, [&leg], [&leg], 9_401);
                let terms = wa.dense_data().unwrap().len();

                let selection =
                    LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..2), (U1Irrep::new(1), 0..1)])
                        .unwrap();
                let restricted = a.restrict_leg(engine, &[(1, &selection)]).unwrap();
                let wide_restricted = wa.restrict_leg(engine, &[(1, &selection)]).unwrap();
                assert_payloads_agree(
                    "restrict_leg",
                    restricted.dense_data().unwrap(),
                    wide_restricted.dense_data().unwrap(),
                    1,
                );
                assert_payloads_agree(
                    "embed_leg",
                    restricted.embed_leg(engine, 1, &selection).unwrap().dense_data().unwrap(),
                    wide_restricted.embed_leg(engine, 1, &selection).unwrap().dense_data().unwrap(),
                    1,
                );

                // Compact diagonal storage: an operand that never materializes
                // its off-diagonal zeros.
                // Compact diagonal storage: an operand that never
                // materializes its off-diagonal zeros.
                let compact: TensorMap<U1FusionRule, $narrow> =
                    TensorMap::diagonal(engine, &runtime, &leg, diagonal_spectra(9_402)).unwrap();
                let wide_compact: TensorMap<U1FusionRule, $wide> =
                    TensorMap::diagonal(engine, &runtime, &leg, diagonal_spectra(9_402)).unwrap();

                for (got, expected) in compact
                    .diagview(engine)
                    .unwrap()
                    .iter()
                    .zip(wide_compact.diagview(engine).unwrap().iter())
                {
                    assert_eq!(got.sector, expected.sector);
                    assert_payloads_agree("diagview", &got.values, &expected.values, 1);
                }
                assert_scalars_agree(
                    "compact norm",
                    Complex64::new(compact.norm(engine, 2.0).unwrap(), 0.0),
                    Complex64::new(wide_compact.norm(engine, 2.0).unwrap(), 0.0),
                    terms,
                );
                assert_scalars_agree(
                    "compact tr",
                    compact.tr(engine).unwrap().wide(),
                    wide_compact.tr(engine).unwrap().wide(),
                    terms,
                );
                assert_payloads_agree(
                    "compact restrict_leg",
                    compact.restrict_leg(engine, &[(0, &selection), (1, &selection)]).unwrap().materialize(engine).unwrap().dense_data().unwrap(),
                    wide_compact.restrict_leg(engine, &[(0, &selection), (1, &selection)]).unwrap().materialize(engine).unwrap().dense_data().unwrap(),
                    1,
                );
                assert_payloads_agree(
                    "compact compose",
                    compact.compose(engine, &a).unwrap().dense_data().unwrap(),
                    wide_compact.compose(engine, &wa).unwrap().dense_data().unwrap(),
                    terms,
                );
            }

            /// The recoupling matrix of one structure identity is converted
            /// into the lane's own workspace scratch. Running both precisions
            /// against the same structure on one runtime, interleaved and
            /// repeated, fails if either lane reads the other's converted
            /// scratch.
            #[test]
            fn single_and_double_lanes_do_not_share_coefficient_scratch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
                let runtime = runtime();
                let leg = fermion_su2_leg();
                let (a, wa) = twin!(engine, &runtime, $narrow, $wide, [&leg, &leg], [&leg, &leg], 9_501);
                let terms = wa.dense_data().unwrap().len();
                for _ in 0..3 {
                    let narrow = a.permute(engine, &[2, 0], &[1, 3]).unwrap();
                    let wide = wa.permute(engine, &[2, 0], &[1, 3]).unwrap();
                    assert_payloads_agree(
                        "interleaved permute",
                        narrow.dense_data().unwrap(),
                        wide.dense_data().unwrap(),
                        terms,
                    );
                }
            }
        }
    };
}

/// The compact diagonal fixture on [`u1_leg`], at either precision.
fn diagonal_spectra<D: Parts>(mut state: u64) -> [SectorSpectrum<U1Irrep, D>; 3] {
    [
        (U1Irrep::new(-1), 2),
        (U1Irrep::new(0), 3),
        (U1Irrep::new(1), 2),
    ]
    .map(|(sector, degeneracy)| SectorSpectrum {
        sector,
        values: (0..degeneracy).map(|_| draw_parts(&mut state)).collect(),
    })
}

base_suite!(f32_payload, f32, f64);
base_suite!(complex32_payload, Complex32, Complex64);

/// Checked-Generic providers: outer multiplicity, real structural
/// coefficients, and a recoupling matrix per vertex pattern.
///
/// SU(3) with the adjoint irrep produces fusion trees that differ only by
/// their multiplicity vertex, which the multiplicity-free fixtures above
/// cannot reach.
#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use std::sync::Arc;

    use super::*;
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::GradedSpace;

    macro_rules! checked_generic_suite {
        ($suite:ident, $narrow:ty, $wide:ty) => {
            mod $suite {
                use super::*;

                #[test]
                fn su3_base_operations_match_the_widened_oracle() {
                    let engine = &mut symtn::typed::CpuEngine::serial();
                    let runtime = runtime();
                    let provider = Arc::new(
                        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
                            .unwrap(),
                    );
                    let adjoint = vec![2i64, 2];
                    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)])
                        .unwrap();
                    let (a, wa) = twin!(
                        engine,
                        &runtime,
                        $narrow,
                        $wide,
                        [&leg, &leg],
                        [&leg, &leg],
                        9_601
                    );
                    assert!(
                        (0..a.subblock_count()).any(|index| a
                            .subblock_fusion_trees(index)
                            .unwrap()
                            .codomain_vertices()
                            .iter()
                            .any(|vertex| vertex.get() > 1)),
                        "the fixture must carry a Generic multiplicity vertex"
                    );
                    let terms = wa.dense_data().unwrap().len();

                    for (what, got, expected) in [
                        (
                            "permute",
                            a.permute(engine, &[2, 0], &[1, 3]).unwrap(),
                            wa.permute(engine, &[2, 0], &[1, 3]).unwrap(),
                        ),
                        (
                            "repartition",
                            a.repartition(engine, 3).unwrap(),
                            wa.repartition(engine, 3).unwrap(),
                        ),
                        (
                            "adjoint",
                            a.adjoint(engine).unwrap(),
                            wa.adjoint(engine).unwrap(),
                        ),
                        (
                            "trace_pairs",
                            a.trace_pairs(engine, &[(0, 3)]).unwrap(),
                            wa.trace_pairs(engine, &[(0, 3)]).unwrap(),
                        ),
                    ] {
                        assert_payloads_agree(
                            what,
                            got.materialize(engine).unwrap().dense_data().unwrap(),
                            expected.materialize(engine).unwrap().dense_data().unwrap(),
                            terms,
                        );
                    }

                    assert_scalars_agree(
                        "norm",
                        Complex64::new(a.norm(engine, 2.0).unwrap(), 0.0),
                        Complex64::new(wa.norm(engine, 2.0).unwrap(), 0.0),
                        terms,
                    );
                    assert_scalars_agree(
                        "inner",
                        a.inner(engine, &a).unwrap().wide(),
                        wa.inner(engine, &wa).unwrap().wide(),
                        terms,
                    );
                }
            }
        };
    }

    checked_generic_suite!(f32_payload, f32, f64);
    checked_generic_suite!(complex32_payload, Complex32, Complex64);
}
