use super::*;
use std::cell::{Cell, RefCell};
use symtn_core::{
    generic_braid_tree_pair_checked, validate_generic_fusion_tree_pair_checked, BraidingStyleKind,
    FusionStyleKind, LieGroupFusionRule, LieGroupFusionRuleError, SectorVec,
};

#[derive(Debug)]
enum TwoKeyError {
    SourceALater,
    SourceBEarlier,
    Delegated(LieGroupFusionRuleError),
}
impl std::fmt::Display for TwoKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceALater => f.write_str("source A later R query"),
            Self::SourceBEarlier => f.write_str("source B earlier R query"),
            Self::Delegated(error) => std::fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for TwoKeyError {}

// Immutable failure policy: only these two admitted R keys fail. Logging does
// not affect answers, and ordinary Lie-group structural/other symbol answers remain.
struct TwoKeyFailures {
    inner: LieGroupFusionRule,
    vacuum: SectorId,
    adjoint: SectorId,
    r_calls: RefCell<Vec<[SectorId; 3]>>,
    f_calls: Cell<usize>,
}
impl symtn_core::testing::Provider for TwoKeyFailures {}

impl CheckedGenericFusion for TwoKeyFailures {
    type Error = TwoKeyError;
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.vacuum
    }
    fn try_dual(&self, a: SectorId) -> Result<SectorId, Self::Error> {
        self.inner.try_dual(a).map_err(TwoKeyError::Delegated)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        self.inner
            .try_fusion_channels(a, b)
            .map_err(TwoKeyError::Delegated)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.inner
            .try_fusion_channels_in_table(a, b)
            .map_err(TwoKeyError::Delegated)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.inner
            .try_nsymbol(a, b, c)
            .map_err(TwoKeyError::Delegated)
    }
}
impl CheckedGenericRigidSymbols for TwoKeyFailures {
    type Scalar = f64;
    fn try_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner.try_dim_scalar(a).map_err(TwoKeyError::Delegated)
    }
    fn try_sqrt_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner
            .try_sqrt_dim_scalar(a)
            .map_err(TwoKeyError::Delegated)
    }
    fn try_inv_sqrt_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner
            .try_inv_sqrt_dim_scalar(a)
            .map_err(TwoKeyError::Delegated)
    }
    fn try_frobenius_schur_phase_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner
            .try_frobenius_schur_phase_scalar(a)
            .map_err(TwoKeyError::Delegated)
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.f_calls.set(self.f_calls.get() + 1);
        self.inner
            .try_f_symbol_generic(a, b, c, d, e, f)
            .map_err(TwoKeyError::Delegated)
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.r_calls.borrow_mut().push([a, b, c]);
        if [a, b, c] == [self.vacuum, self.adjoint, self.adjoint] {
            return Err(TwoKeyError::SourceALater);
        }
        if [a, b, c] == [self.adjoint, self.adjoint, self.adjoint] {
            return Err(TwoKeyError::SourceBEarlier);
        }
        self.inner
            .try_r_symbol_generic(a, b, c)
            .map_err(TwoKeyError::Delegated)
    }
}

struct OrderFixture {
    provider: TwoKeyFailures,
    vacuum: SectorId,
    adjoint: SectorId,
    source_a: FusionTreePairKey,
    ordered: Vec<FusionTreePairKey>,
    homspace: FusionTreeHomSpace,
}

/// The actual checked SU(3) four-adjoint HomSpace to the vacuum (eight full
/// basis trees), with source A (`[vac, adj]` inner lines) placed before source
/// B (`[adj, adj]`), all vertices one.
fn order_fixture() -> OrderFixture {
    let inner = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let v = inner.encode_irrep(&[0, 0]).unwrap();
    let a = inner.encode_irrep(&[1, 1]).unwrap();
    assert_eq!(inner.try_nsymbol(a, a, v).unwrap(), 1);
    assert_eq!(inner.try_nsymbol(a, a, a).unwrap(), 2);
    assert_eq!(inner.try_nsymbol(v, a, a).unwrap(), 1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new((0..4).map(|_| SectorLeg::new([(a, 1)], false))),
        FusionProductSpace::new([]),
    );
    let canonical = homspace
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&inner)
        .unwrap();
    // Select full keys from the actual checked Lie-group layout, including the
    // one-based multiplicity vertices; do not invent an admitted tree.
    let keys = (0..canonical.block_count())
        .map(|i| {
            let BlockKey::FusionTree(key) = canonical.block(i).unwrap().key() else {
                panic!("fusion key expected")
            };
            key.clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(keys.len(), 8);
    let select = |innerlines: &[SectorId]| {
        keys.iter()
            .find(|key| {
                key.codomain_tree().innerlines() == innerlines
                    && key
                        .codomain_tree()
                        .vertices()
                        .iter()
                        .all(|vertex| vertex.get() == 1)
            })
            .expect("actual checked Lie-group layout must contain the proposed source")
            .clone()
    };
    let source_a = select(&[v, a]);
    let source_b = select(&[a, a]);
    for key in [&source_a, &source_b] {
        assert_eq!(key.codomain_tree().uncoupled(), &[a; 4]);
        assert_eq!(key.codomain_tree().coupled(), v);
        assert_eq!(key.codomain_tree().is_dual(), &[false; 4]);
        assert_eq!(key.codomain_tree().vertices().len(), 3);
        assert!(key.domain_tree().uncoupled().is_empty());
        assert_eq!(key.domain_tree().coupled(), v);
    }
    let mut ordered = vec![source_a.clone(), source_b.clone()];
    ordered.extend(
        keys.iter()
            .filter(|key| **key != source_a && **key != source_b)
            .cloned(),
    );
    assert_eq!(ordered.len(), keys.len());
    let provider = TwoKeyFailures {
        inner,
        vacuum: v,
        adjoint: a,
        r_calls: RefCell::new(Vec::new()),
        f_calls: Cell::new(0),
    };
    for key in &ordered {
        validate_generic_fusion_tree_pair_checked(&provider, key).unwrap();
    }
    assert!(provider.r_calls.borrow().is_empty());
    OrderFixture {
        provider,
        vacuum: v,
        adjoint: a,
        source_a,
        ordered,
        homspace,
    }
}

/// What: the production checked compiler reports the approved step-major
/// first provider error (#1962): the first whole-basis Artin step visits
/// source A (succeeds) then source B (fails), before source A's later step.
/// TensorKit fsbraid(block) applies each Artin move to the whole current
/// basis:
/// https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/fusiontrees/braiding_manipulations.jl#L302-L339
#[test]
fn checked_group_compiler_reports_step_major_first_provider_error() {
    let fixture = order_fixture();
    let (provider, v, a) = (&fixture.provider, fixture.vacuum, fixture.adjoint);
    let source = packed_fixture_structure(
        4,
        fixture.ordered.iter().cloned().map(|key| (key, vec![1; 4])),
    )
    .unwrap();
    assert_eq!(source.fusion_tree_group_slice().len(), 1);
    assert_eq!(
        source.fusion_tree_group_slice()[0].block_indices()[..2],
        [0, 1]
    );

    // Source-major contrast: source A alone completes its whole schedule
    // (its later step queries R(vac, adj, adj), the other failing key).
    let first = generic_braid_tree_pair_checked(
        &fixture.provider.inner,
        &fixture.source_a,
        &[1, 0, 2, 3],
        &[],
        &[0, 1, 2, 3],
        &[],
    )
    .unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].0, fixture.source_a);
    assert!(first[0].1.abs() > 0.5);

    // [1,2,0,3] has forward Artin steps [0,1]; cold, then repeated.
    for _attempt in 0..2 {
        provider.r_calls.borrow_mut().clear();
        provider.f_calls.set(0);
        let error = crate::build_checked_generic_tree_pair_transform_group_plan(
            provider,
            TreeTransformOperation::braid([1, 2, 0, 3], [], [0, 1, 2, 3], []),
            &source,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                CheckedGenericPlanError::Provider(TwoKeyError::SourceBEarlier)
            ),
            "{error:?}"
        );
        assert_eq!(*provider.r_calls.borrow(), vec![[a, a, v], [a, a, a]]);
        assert_eq!(provider.f_calls.get(), 0);
    }
}

/// What: the cached checked path reports the same first provider error as
/// the standalone compiler on the same source, cold and repeated, and a
/// failed build leaves the completed-transformer and composed-coefficient
/// caches untouched.
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_runtime_failure_order_is_stable_and_publishes_nothing() {
    let fixture = order_fixture();
    let homspace = fixture.homspace.clone();
    let provider = Arc::new(fixture.provider);
    let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let operation = TreeTransformOperation::braid([1, 2, 0, 3], [], [0, 1, 2, 3], []);

    provider.r_calls.borrow_mut().clear();
    let standalone = crate::build_checked_generic_tree_pair_transform_group_plan(
        provider.as_ref(),
        operation.clone(),
        source.space().structure(),
    )
    .unwrap_err();
    let standalone_r = provider.r_calls.borrow().clone();
    // The canonical HomSpace order also visits a `[vac, adj]`-inner tree
    // before an `[adj, adj]`-inner tree at the first Artin step.
    let (v, a) = (provider.vacuum, provider.adjoint);
    assert!(
        matches!(
            standalone,
            CheckedGenericPlanError::Provider(TwoKeyError::SourceBEarlier)
        ),
        "{standalone:?}"
    );
    assert_eq!(standalone_r, vec![[a, a, v], [a, a, a]]);
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    owner_activity();
    crate::probe::take_coefficient_group_activity();
    for _attempt in 0..2 {
        provider.r_calls.borrow_mut().clear();
        provider.f_calls.set(0);
        let error = context
            .tree_transform_owned_checked_generic_in(&source, None, &data, &operation, 1.0)
            .unwrap_err();
        assert_eq!(format!("{error:?}"), format!("{standalone:?}"));
        assert_eq!(*provider.r_calls.borrow(), standalone_r);
        assert_eq!(provider.f_calls.get(), 0);
        assert_owner_untouched(owner_activity());
        // The failing group was looked up and missed both times: nothing was
        // staged for publication.
        let groups = crate::probe::take_coefficient_group_activity();
        assert_eq!((groups.hits, groups.publications), (0, 0));
    }
}
