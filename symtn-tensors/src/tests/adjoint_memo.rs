use super::*;
use crate::probe::{self, Event};
use num_complex::Complex64;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;
use symtn_core::{
    CheckedFusionAlgebra, FermionParityFusionRule, ProductFusionRule, ProductSectorCodec,
    SU2FusionRule, SU2Irrep, SectorId, TensorKitProductCodec, U1FusionRule, U1Irrep,
};

use crate::{BoundDynamicFusionMapSpace, OutputAxisOrder};

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn u1_legs() -> Vec<(SectorId, usize)> {
    [(-1, 2), (0, 3), (1, 4)]
        .into_iter()
        .map(|(charge, deg)| (U1Irrep::new(charge).sector_id(), deg))
        .collect()
}

fn su2_legs() -> Vec<(SectorId, usize)> {
    [(0, 2), (1, 3), (2, 2)]
        .into_iter()
        .map(|(twice, deg)| (SU2Irrep::from_twice_spin(twice).sector_id(), deg))
        .collect()
}

fn fz2_u1_legs() -> Vec<(SectorId, usize)> {
    [((-1, 1), 2), ((0, 0), 3), ((1, 1), 4)]
        .into_iter()
        .map(|((charge, parity), deg)| {
            let sector = TensorKitProductCodec::try_encode(
                SectorId::new(parity),
                U1Irrep::new(charge).sector_id(),
            )
            .unwrap();
            (sector, deg)
        })
        .collect()
}

fn bound<R>(rule: &Arc<R>, legs: &[(SectorId, usize)]) -> BoundDynamicFusionMapSpace<R>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
{
    let leg = || SectorLeg::new(legs.iter().copied(), false);
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(rule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap()
}

trait Sample: Copy + PartialEq + Debug + Zero + One {
    fn sample(index: usize, seed: f64) -> Self;
}

impl Sample for f64 {
    fn sample(index: usize, seed: f64) -> Self {
        seed + index as f64 * 0.25
    }
}

impl Sample for Complex64 {
    fn sample(index: usize, seed: f64) -> Self {
        Complex64::new(seed + index as f64 * 0.25, 1.0 - (index % 3) as f64)
    }
}

/// `conj(lhs)` contracted on its parent codomain axis with `rhs`'s codomain,
/// i.e. `lhs† ∘ rhs`, with `lhs` as a lazy-adjoint operand.
fn conjugated_contract<R, D>(
    context: &mut TensorContractFusionExecutionContext<D, R::Key>,
    dst: &BoundDynamicFusionMapSpace<R>,
    lhs: &BoundDynamicFusionMapSpace<R>,
    lhs_data: &[D],
    rhs: &BoundDynamicFusionMapSpace<R>,
    rhs_data: &[D],
) -> Vec<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    R::Key: 'static + Clone + Eq + Hash + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + Sample,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut out = vec![D::zero(); dst.space().required_len().unwrap()];
    context
        .tensorcontract_prelowered_into(
            engine,
            dst,
            &mut out,
            crate::FusionOperand::adjoint(lhs.space()),
            lhs_data,
            crate::FusionOperand::direct(rhs.space()),
            rhs_data,
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                OutputAxisOrder::identity(),
                true,
                false,
            ),
            D::one(),
            crate::ContractDestinationInit::Axpby(D::zero()),
        )
        .unwrap();
    out
}

fn assert_conjugated_contract_builds_no_adjoint<R, D>(rule: R, legs: &[(SectorId, usize)])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + TreeTransformRuleCacheKey,
    R::Key: 'static + Clone + Eq + Hash + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + Sample,
{
    let rule = Arc::new(rule);
    let lhs = bound(&rule, legs);
    let rhs = bound(&rule, legs);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &bound(&rule, legs).adjoint_view().unwrap(),
        &rhs,
        &[1],
        &[0],
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let len = lhs.space().required_len().unwrap();
    let lhs_data: Vec<D> = (0..len).map(|i| D::sample(i, 1.0)).collect();
    let rhs_data: Vec<D> = (0..len).map(|i| D::sample(i, -2.0)).collect();
    assert!(lhs.space().structure().block_count() >= 2);

    // Previous behaviour: a never-adjointed space under a context that keeps
    // nothing runs the unmemoized builder.
    let mut cold = TensorContractFusionExecutionContext::<D, R::Key>::default();
    let expected = conjugated_contract(
        &mut cold,
        &dst,
        &bound(&rule, legs),
        &lhs_data,
        &rhs,
        &rhs_data,
    );

    let mut context = TensorContractFusionExecutionContext::<D, R::Key>::default();
    let adjoint_view_build_count_rec = probe::record();
    for _ in 0..5 {
        let actual = conjugated_contract(&mut context, &dst, &lhs, &lhs_data, &rhs, &rhs_data);
        assert_eq!(actual, expected);
    }
    // What: a conjugation flag is the lazy adjoint of the owned space, which
    // the planner reads in place: no adjoint structure is built.
    assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);

    let memoized = lhs.space().adjoint_view().unwrap();
    let rebuilt = crate::lowering::adjoint_block_structure_view(
        lhs.space().nout(),
        lhs.space().nin(),
        lhs.space().structure(),
    )
    .unwrap();
    assert_eq!(memoized.structure().as_ref(), &rebuilt);
    assert_ne!(memoized.structure().content_id(), rebuilt.content_id());
    assert_eq!(
        memoized.adjoint_view().unwrap(),
        *lhs.space(),
        "the adjoint of the memoized view is the source again"
    );
}

#[test]
fn conjugated_dyn_contract_builds_no_adjoint_structure() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fz2_u1 = || Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    assert_conjugated_contract_builds_no_adjoint::<_, f64>(U1FusionRule, &u1_legs());
    assert_conjugated_contract_builds_no_adjoint::<_, Complex64>(U1FusionRule, &u1_legs());
    assert_conjugated_contract_builds_no_adjoint::<_, f64>(SU2FusionRule, &su2_legs());
    assert_conjugated_contract_builds_no_adjoint::<_, Complex64>(SU2FusionRule, &su2_legs());
    assert_conjugated_contract_builds_no_adjoint::<_, f64>(fz2_u1(), &fz2_u1_legs());
    assert_conjugated_contract_builds_no_adjoint::<_, Complex64>(fz2_u1(), &fz2_u1_legs());
}

#[test]
fn adjoint_memo_leaves_space_equality_and_layout_hash_unchanged() {
    let rule = Arc::new(U1FusionRule);
    let filled = bound(&rule, &u1_legs());
    // A separate construction can cross an intern reset and mint a different layout id.
    let empty = filled.rebind_validated(&filled.validated_layout()).unwrap();
    let hash = |space: &BoundDynamicFusionMapSpace<U1FusionRule>| {
        rustc_hash::FxBuildHasher.hash_one(space.validated_layout())
    };
    let empty_hash = hash(&empty);
    let adjoint = filled.space().adjoint_view().unwrap();

    assert_eq!(filled.space(), empty.space());
    assert_eq!(filled.validated_layout(), empty.validated_layout());
    assert_eq!(hash(&filled), empty_hash);
    // A clone of a filled space shares the memo and still compares equal.
    let cloned = filled.space().clone();
    assert!(Arc::ptr_eq(
        cloned.adjoint_view().unwrap().structure(),
        adjoint.structure()
    ));
    assert_eq!(&cloned, empty.space());
}
