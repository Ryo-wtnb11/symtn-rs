use super::*;
use crate::probe::{self, Event};
use crate::tests::GenericMultiplicityRule;
use crate::BoundDynamicTensorRef;
use std::cell::Cell;
use symtn_core::{
    clear_structure_caches, structure_cache_info, BlockSpec, BraidingStyleKind, CoupledSectorFold,
    FermionParityFusionRule, FusionAlgebraError, FusionProductSpace, FusionTreePairKey,
    Fz2SectorLayout, InfallibleGeneric, PackedProductCodec, ProductFusionRule, ProductSectorCodec,
    ProductSectorLayout, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorVec,
    StructureCacheKind, Su2SectorLayout, U1FusionRule, U1Irrep, U1SectorLayout, Z2FusionRule,
    Z2Irrep,
};

type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

fn lowered_product_rule() -> TripleRule {
    TripleRule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

fn matrix_space() -> DynamicFusionMapSpace {
    let rule = Z2FusionRule;
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, [vec![1, 1]]).unwrap()
}

fn z2_matrix_space() -> DynamicFusionMapSpace {
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    DynamicFusionMapSpace::from_degeneracy_shapes(
        &Z2FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
        [vec![1, 1], vec![1, 1]],
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FailAtCallError(usize);

impl fmt::Display for FailAtCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "checked Generic query {} failed", self.0)
    }
}

impl std::error::Error for FailAtCallError {}

/// A Generic rule (both plain and checked) that counts every query and can
/// fail the `fail_at`-th checked one. Not `contract::checked_generic`'s
/// `CheckedGenericSpy`, which records per-query events and injects malformed
/// symbol shapes; the two exercise different contracts.
struct FailAtCallRule {
    identity: RuleIdentity,
    style: Cell<FusionStyleKind>,
    calls: Cell<usize>,
    identity_calls: Cell<usize>,
    fail_at: Option<usize>,
    reset_on_query: Cell<bool>,
}

impl FailAtCallRule {
    /// A rule answering under `identity`, so that one test's instances share
    /// it and no other test's structure-cache entries can answer for them.
    fn with_identity(identity: RuleIdentity) -> Self {
        let mut rule = Self::new();
        rule.identity = identity;
        rule
    }

    fn new() -> Self {
        Self {
            identity: RuleIdentity::of_type::<Self>(),
            style: Cell::new(FusionStyleKind::Generic),
            calls: Cell::new(0),
            identity_calls: Cell::new(0),
            fail_at: None,
            reset_on_query: Cell::new(false),
        }
    }

    fn hit(&self) -> Result<(), FailAtCallError> {
        if self.reset_on_query.replace(false) {
            clear_structure_caches();
        }
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            Err(FailAtCallError(call))
        } else {
            Ok(())
        }
    }
}

impl FusionRule for FailAtCallRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity_calls.set(self.identity_calls.get() + 1);
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.style.get()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.calls.set(self.calls.get() + 1);
        sector
    }

    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        self.calls.set(self.calls.get() + 1);
        [SectorId::new(0)].into_iter().collect()
    }

    fn nsymbol(&self, _: SectorId, _: SectorId, _: SectorId) -> usize {
        self.calls.set(self.calls.get() + 1);
        2
    }
}

impl CheckedGenericFusion for FailAtCallRule {
    type Error = FailAtCallError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity_calls.set(self.identity_calls.get() + 1);
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.style.get()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.hit()?;
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        let _ = (left, right);
        self.hit()?;
        Ok([SectorId::new(0)].into_iter().collect())
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        let _ = (left, right);
        self.hit()?;
        Ok([SectorId::new(0)].into_iter().collect())
    }

    fn try_coupled_sector_fold(
        &self,
        effective: &[SectorId],
    ) -> Result<CoupledSectorFold, Self::Error> {
        let _ = effective;
        self.hit()?;
        Ok(CoupledSectorFold::complete(vec![SectorId::new(0)]))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        let _ = (left, right, coupled);
        self.hit()?;
        Ok(2)
    }
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // Bound ownership requires Arc; this local equality checker borrows a non-Sync rule.
fn checked_generic_equal_identity_checker_commits_under_source_arc() {
    let rule = GenericMultiplicityRule;
    let charge = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(charge, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(charge, 1)], false)]),
    );
    let source_provider = Arc::new(InfallibleGeneric::new(&rule));
    let checker_provider = Arc::new(InfallibleGeneric::new(&rule));
    assert!(!Arc::ptr_eq(&source_provider, &checker_provider));
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&source_provider),
        homspace.clone(),
    )
    .unwrap();

    let prepared = source
        .prepare_final_homspace_generic_with_checked(checker_provider.as_ref(), homspace)
        .unwrap();
    let output = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap();

    assert!(Arc::ptr_eq(output.provider_arc(), &source_provider));
    assert!(!Arc::ptr_eq(output.provider_arc(), &checker_provider));
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; these local guard spies use Cell counters.
fn checked_generic_bound_guards_preserve_typed_errors_before_commit() {
    let homspace = || FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], [(0, 1)]);
    // A fresh identity for this test's rules. Why not `of_type`: a
    // successful walk may answer later ones from the sector-structure cache
    // (#2030), so the failing checker gets a HomSpace no walk under this
    // identity has published.
    let identity = RuleIdentity::new_unique::<FailAtCallRule>();
    let failing_homspace =
        || FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1), (0, 1)], [(0, 1)]);

    let wrong_root_style = Arc::new(FailAtCallRule::with_identity(identity.clone()));
    wrong_root_style.style.set(FusionStyleKind::Unique);
    let error = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&wrong_root_style),
        homspace(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(wrong_root_style.calls.get(), 0);

    let mut failing_root = FailAtCallRule::with_identity(identity.clone());
    failing_root.fail_at = Some(1);
    let failing_root = Arc::new(failing_root);
    let error = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&failing_root),
        homspace(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(FailAtCallError(1))
    ));

    let source_provider = Arc::new(FailAtCallRule::with_identity(identity.clone()));
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&source_provider),
        homspace(),
    )
    .unwrap();

    let mut wrong_identity = FailAtCallRule::with_identity(identity.clone());
    wrong_identity.identity = RuleIdentity::of_type::<Z2FusionRule>();
    let error = source
        .prepare_final_homspace_generic_with_checked(&wrong_identity, homspace())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert_eq!(wrong_identity.calls.get(), 0);

    let wrong_checker_style = FailAtCallRule::with_identity(identity.clone());
    wrong_checker_style.style.set(FusionStyleKind::Unique);
    let error = source
        .prepare_final_homspace_generic_with_checked(&wrong_checker_style, homspace())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(wrong_checker_style.calls.get(), 0);

    let mut failing_checker = FailAtCallRule::with_identity(identity.clone());
    failing_checker.fail_at = Some(1);
    let error = source
        .prepare_final_homspace_generic_with_checked(&failing_checker, failing_homspace())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(FailAtCallError(1))
    ));

    let checker = FailAtCallRule::with_identity(identity.clone());
    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    let unchecked = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FailAtCallRule::with_identity(identity.clone())),
        homspace(),
    )
    .unwrap();
    let error = unchecked
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap_err();
    assert!(matches!(error, OperationError::StructureMismatch { .. }));

    let mut prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    prepared.identity = RuleIdentity::of_type::<Z2FusionRule>();
    let error = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));

    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    // Contract-violating adversarial mutation proves the defensive final style guard.
    source_provider.style.set(FusionStyleKind::Unique);
    let error = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    source_provider.style.set(FusionStyleKind::Generic);

    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    let output = source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap();
    assert!(Arc::ptr_eq(output.provider_arc(), &source_provider));
    assert!(std::ptr::eq(output.provider(), source_provider.as_ref()));
    assert_eq!(output.space().homspace(), &homspace());
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the single-threaded spy uses Cell counters.
fn checked_generic_bound_prepare_commit_builds_identity_once() {
    let homspace = || FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], [(0, 1)]);
    let identity = RuleIdentity::new_unique::<FailAtCallRule>();
    let source_provider = Arc::new(FailAtCallRule::with_identity(identity.clone()));
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&source_provider),
        homspace(),
    )
    .unwrap();
    let checker = FailAtCallRule::with_identity(identity);
    source_provider.identity_calls.set(0);

    let prepared = source
        .prepare_final_homspace_generic_with_checked(&checker, homspace())
        .unwrap();
    let checker_reads = checker.identity_calls.get();
    source
        .commit_final_homspace_generic_bound_checked(prepared)
        .unwrap();

    assert_eq!(checker_reads, 1);
    assert_eq!(checker.identity_calls.get(), 1);
    assert_eq!(source_provider.identity_calls.get(), 0);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the single-threaded spy uses Cell counters.
fn checked_generic_preparation_rejects_unchecked_binding_before_checker_algebra_queries() {
    let source_provider = Arc::new(FailAtCallRule::new());
    let checker = FailAtCallRule::new();
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(source_provider, homspace.clone())
            .unwrap();

    let error = match source.prepare_final_homspace_generic_with_checked(&checker, homspace) {
        Ok(_) => panic!("unchecked binding was accepted as checked authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        CheckedGenericStructureError::Core(CoreError::MalformedFusionTree { .. })
    ));
    assert_eq!(checker.calls.get(), 0);
    // The seam reads the checker identity once before the closure rejects (#2046).
    assert_eq!(checker.identity_calls.get(), 1);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the spy and rule are single-threaded test values.
fn checked_generic_prepared_structure_constructor_matches_root_constructor() {
    // What: committing a structure the caller already enumerated yields
    // the same checked Generic root as enumerating inside the constructor
    // (blocks, required length, HomSpace, provider Arc, checked binding),
    // and the commit itself issues no provider query.
    let rule = GenericMultiplicityRule;
    let charge = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(charge, 2)], false),
            SectorLeg::new([(charge, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(charge, 1)], false),
            SectorLeg::new([(charge, 3)], false),
        ]),
    );
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let root = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )
    .unwrap();
    let prepared = homspace
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(provider.as_ref())
        .unwrap();
    let committed = BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
        prepared,
    )
    .unwrap();
    let rows = |space: &DynamicFusionMapSpace| {
        let structure = space.structure();
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(root.space().structure().block_count(), 5);
    assert_eq!(rows(root.space()), rows(committed.space()));
    assert_eq!(
        root.space().required_len().unwrap(),
        committed.space().required_len().unwrap()
    );
    assert_eq!(root.space().homspace(), committed.space().homspace());
    assert_eq!(root.space().nout(), committed.space().nout());
    assert_eq!(root.space().nin(), committed.space().nin());
    assert!(Arc::ptr_eq(committed.provider_arc(), &provider));
    // Only a `CheckedGeneric` binding stages a checked final HomSpace.
    committed
        .prepare_final_homspace_generic_with_checked(provider.as_ref(), homspace)
        .unwrap();

    // A fresh identity, so the walk counted here is not answered by another
    // test's published layout.
    let spy = Arc::new(FailAtCallRule::with_identity(RuleIdentity::new_unique::<
        FailAtCallRule,
    >()));
    let vacuum_hom = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
    let prepared = vacuum_hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(spy.as_ref())
        .unwrap();
    let enumerated = spy.calls.get();
    assert!(enumerated > 0);
    BoundDynamicFusionMapSpace::from_prepared_final_homspace_generic_checked(
        Arc::clone(&spy),
        vacuum_hom,
        prepared,
    )
    .unwrap();
    assert_eq!(spy.calls.get(), enumerated);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; the isolated spy uses Cell counters.
fn checked_generic_bound_space_commits_the_staged_layout_without_reenumeration() {
    const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_BOUND_SPACE_ISOLATED";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "contract::dynamic_space::bound_invariant_tests::checked_generic_bound_space_commits_the_staged_layout_without_reenumeration",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        return;
    }

    clear_structure_caches();
    let provider = Arc::new(FailAtCallRule::new());
    let source_hom = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::clone(&provider), source_hom)
            .unwrap();
    provider.calls.set(0);
    let final_hom = || FusionTreeHomSpace::from_sector_ids([(0, 2), (0, 3)], [(0, 5)]);
    let snapshots = || {
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        )
    };

    for (identity, style) in [
        (
            RuleIdentity::of_type::<Z2FusionRule>(),
            FusionStyleKind::Generic,
        ),
        (
            FusionRule::rule_identity(provider.as_ref()),
            FusionStyleKind::Unique,
        ),
    ] {
        let mut spy = FailAtCallRule::new();
        spy.identity = identity;
        spy.style.set(style);
        let before = snapshots();
        let error = source
            .prepare_final_homspace_generic_checked(&spy, final_hom())
            .err()
            .unwrap();
        assert!(matches!(
            error,
            CheckedGenericStructureError::Core(
                CoreError::FusionRuleMismatch { .. } | CoreError::UnsupportedFusionStyle { .. }
            )
        ));
        assert_eq!(spy.calls.get(), 0);
        assert_eq!(provider.calls.get(), 0);
        assert_eq!(snapshots(), before);
    }

    let before_prepare = snapshots();
    let spy = FailAtCallRule::new();
    let prepared = source
        .prepare_final_homspace_generic_checked(&spy, final_hom())
        .unwrap();
    let prepare_calls = spy.calls.get();
    assert!(prepare_calls > 0);
    assert_eq!(provider.calls.get(), 0);
    let preview = prepared.structure().clone();
    let required_len = prepared.required_len();
    // The successful walk publishes its layout to the sector-structure cache
    // (#2030); no block structure is admitted before the commit.
    let after_prepare = snapshots();
    assert_eq!(
        after_prepare.0.admissions(),
        before_prepare.0.admissions() + 1
    );
    assert_eq!(after_prepare.1, before_prepare.1);

    let before_commit = snapshots();
    let committed = source
        .commit_final_homspace_generic_checked(prepared)
        .unwrap();
    assert_eq!(spy.calls.get(), prepare_calls);
    assert_eq!(provider.calls.get(), 0);
    assert!(Arc::ptr_eq(committed.provider_arc(), &provider));
    assert_eq!(committed.space().structure().as_ref(), &preview);
    assert_eq!(committed.space().required_len().unwrap(), required_len);
    assert_eq!(committed.space().homspace(), &final_hom());
    assert!(matches!(
        committed.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure).entries(),
        before_commit.1.entries() + 1
    );

    // Both walks below are cold: each must query the provider in full.
    clear_structure_caches();
    let complete = FailAtCallRule::new();
    let complete_calls = {
        let _staged = source
            .prepare_final_homspace_generic_checked(&complete, final_hom())
            .unwrap();
        complete.calls.get()
    };
    let mut failing = FailAtCallRule::new();
    failing.fail_at = Some(complete_calls);
    clear_structure_caches();
    let before_failure = snapshots();
    let error = source
        .prepare_final_homspace_generic_checked(&failing, final_hom())
        .err()
        .unwrap();
    assert!(matches!(
        error,
        CheckedGenericStructureError::Provider(FailAtCallError(call))
            if call == complete_calls
    ));
    assert_eq!(failing.calls.get(), complete_calls);
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(snapshots(), before_failure);

    let spy = FailAtCallRule::new();
    let prepared = source
        .prepare_final_homspace_generic_checked(&spy, final_hom())
        .unwrap();
    provider.style.set(FusionStyleKind::Unique);
    let before_style_rejected_commit = snapshots();
    let error = source
        .commit_final_homspace_generic_checked(prepared)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(snapshots(), before_style_rejected_commit);
    provider.style.set(FusionStyleKind::Generic);

    let spy = FailAtCallRule::new();
    let mut corrupted = source
        .prepare_final_homspace_generic_checked(&spy, final_hom())
        .unwrap();
    corrupted.identity = RuleIdentity::of_type::<Z2FusionRule>();
    let before_rejected_commit = snapshots();
    let error = source
        .commit_final_homspace_generic_checked(corrupted)
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert_eq!(snapshots(), before_rejected_commit);
}

#[test]
fn layout_constructor_checks_rank_and_storage_but_not_admission() {
    // What: the rank-free expert layout constructor rejects a structure of
    // the wrong rank and aliased storage footprints, and returns an unbound
    // space over an accepted layout.
    let complete = z2_matrix_space();
    let homspace = complete.homspace().clone();
    let rank_one = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    assert_eq!(
        DynamicFusionMapSpace::try_from_layout(homspace.clone(), rank_one),
        Err(CoreError::StructureRankMismatch {
            expected: 2,
            actual: 1,
        })
    );

    let keys = homspace.fusion_tree_keys(&Z2FusionRule);
    let aliased = BlockStructure::from_blocks_with_rank(
        2,
        keys.iter()
            .map(|key| BlockSpec::with_key(key.clone().into(), vec![1, 1], vec![1, 1], 0))
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        DynamicFusionMapSpace::try_from_layout(homspace.clone(), aliased),
        Err(CoreError::OverlappingBlockStorage {
            first_block: 0,
            second_block: 1,
            offset: 0,
        })
    );

    let raw = DynamicFusionMapSpace::try_from_layout(
        homspace,
        BlockStructure::clone(complete.structure()),
    )
    .unwrap();
    assert_eq!(raw.admission(), &FusionSpaceAdmission::Unbound);
    assert_eq!(raw.structure(), complete.structure());
    assert_eq!((raw.nout(), raw.nin()), (1, 1));
}

#[test]
fn expert_sparse_layout_is_a_subset_with_structural_zeros() {
    // What: filtering an expert layout drops completeness, while an omitted
    // valid tree remains an accepted structural zero.
    let complete = z2_matrix_space();
    let keys = complete.homspace().fusion_tree_keys(&Z2FusionRule);
    assert!(matches!(
        complete.admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    let first = complete.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let raw =
        DynamicFusionMapSpace::try_from_layout(complete.homspace().clone(), structure).unwrap();
    assert_eq!(raw.admission(), &FusionSpaceAdmission::Unbound);
    assert!(matches!(
        raw.validate_rule(&Z2FusionRule),
        Err(OperationError::Core(CoreError::MissingFusionRuleIdentity))
    ));

    let subset = raw.try_bind_rule(&Z2FusionRule).unwrap();
    assert!(matches!(
        subset.admission(),
        FusionSpaceAdmission::Subset(_)
    ));
    assert!(subset
        .structure()
        .find_block_index_by_fusion_tree_pair(&keys[0])
        .is_some());
    assert!(subset
        .structure()
        .find_block_index_by_fusion_tree_pair(&keys[1])
        .is_none());
    assert_eq!(
        subset.adjoint_view().unwrap().admission(),
        subset.admission()
    );
}

#[test]
fn sparse_subset_operands_keep_only_present_canonical_keys() {
    let complete = z2_matrix_space();
    let first = complete.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = DynamicFusionMapSpace::try_from_layout(complete.homspace().clone(), structure)
        .unwrap()
        .try_bind_rule(&Z2FusionRule)
        .unwrap();
    let storage = subset;
    let direct = FusionOperand::direct(&storage)
        .prepare(&Z2FusionRule, encoded_layout_primer::<Z2FusionRule>)
        .unwrap();
    let adjoint = FusionOperand::adjoint(&storage)
        .prepare(&Z2FusionRule, encoded_layout_primer::<Z2FusionRule>)
        .unwrap();

    // What: Direct retains the parent-owned order, while Adjoint retains
    // only parent-present keys in canonical logical order.
    assert_eq!(direct.logical_block_count(), 1);
    assert_eq!(adjoint.logical_block_count(), 1);
    assert_eq!(direct.storage_index(0).unwrap(), 0);
    assert_eq!(adjoint.storage_index(0).unwrap(), 0);
    assert!(direct.is_direct());
    assert_eq!(adjoint.adjoint_storage_indices().unwrap(), [0].as_slice());
}

#[test]
fn complete_and_subset_spaces_are_mathematically_equal() {
    // What: proof strength survives adjoint views without changing
    // mathematical equality.
    let complete = z2_matrix_space();
    let subset = DynamicFusionMapSpace::try_from_layout(
        complete.homspace().clone(),
        BlockStructure::clone(complete.structure()),
    )
    .unwrap()
    .try_bind_rule(&Z2FusionRule)
    .unwrap();
    assert_eq!(complete, subset);
    assert!(matches!(
        complete.admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(matches!(
        subset.admission(),
        FusionSpaceAdmission::Subset(_)
    ));
    assert!(matches!(
        complete.adjoint_view().unwrap().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(matches!(
        subset.adjoint_view().unwrap().admission(),
        FusionSpaceAdmission::Subset(_)
    ));

    let bound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(subset, Arc::new(Z2FusionRule)).unwrap();
    assert!(matches!(
        bound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
}

#[test]
fn bind_revalidates_complete_without_replacing_layout() {
    // What: an unchecked Complete stamp gains checked algebra proof, commits
    // one lowered capability, and retains the caller's exact block layout.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let scratch_structure_builds_rec = probe::record();
    let raw = z2_matrix_space();
    let structure = Arc::clone(raw.structure());

    let bound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(Z2FusionRule))
            .unwrap();

    assert!(matches!(
        bound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(Arc::ptr_eq(&structure, &bound.space.subblock_structure));
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );

    let rule = lowered_product_rule();
    let pair = Fz2U1Codec::encode(Z2Irrep::ODD.sector_id(), U1Irrep::new(2).sector_id());
    let sector = TripleCodec::encode(pair, SU2Irrep::from_twice_spin(1).sector_id());
    let layout = DynamicFusionMapSpace::from_degeneracy_shapes(
        &rule,
        FusionTreeHomSpace::from_sectors([(sector, 1)], [(sector, 1)]),
        [vec![1, 1]],
    )
    .unwrap();
    let raw = layout;
    let structure = Arc::clone(raw.structure());
    let scratch_structure_builds_rec = probe::record();

    let bound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(rule)).unwrap();

    assert!(matches!(
        bound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert!(Arc::ptr_eq(&structure, &bound.space.subblock_structure));
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn lowered_bind_failure_publishes_no_layout_or_admission() {
    // What: finite U1 nonclosure revalidates matching unchecked Subset and
    // Complete stamps, returns the exact algebra failure, and commits no state.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let max = U1Irrep::new(i32::MAX).sector_id();
    let one = U1Irrep::new(1).sector_id();
    let vacuum = U1Irrep::new(0).sector_id();
    let pair = FusionTreePairKey::try_pair_from_sector_ids(
        [max.id(), one.id()],
        [],
        vacuum.id(),
        [false; 2],
        [],
        [],
        [],
        [1],
        [],
    )
    .unwrap();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(max, 1)], false),
            SectorLeg::new([(one, 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let structure =
        crate::tests::packed_fixture_structure(2, [(BlockKey::FusionTree(pair), vec![1, 1])])
            .unwrap()
            .into_shared();
    let base = DynamicFusionMapSpace {
        nout: 2,
        nin: 0,
        homspace: Arc::new(homspace),
        subblock_structure: structure,
        admission: FusionSpaceAdmission::Unbound,
        adjoint: OnceLock::new(),
    };

    clear_structure_caches();
    let scratch_structure_builds_rec = probe::record();
    let mut mismatched = base.clone();
    mismatched.admission = FusionSpaceAdmission::Subset(Z2FusionRule.rule_identity());
    assert!(matches!(
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(
            mismatched,
            Arc::new(U1FusionRule),
        ),
        Err(OperationError::Core(CoreError::FusionRuleMismatch { .. }))
    ));
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );

    for admission in [
        FusionSpaceAdmission::Subset(U1FusionRule.rule_identity()),
        FusionSpaceAdmission::Complete(U1FusionRule.rule_identity()),
    ] {
        clear_structure_caches();
        let scratch_structure_builds_rec = probe::record();
        let mut raw = base.clone();
        raw.admission = admission.clone();
        let original = raw.clone();
        let error =
            BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(U1FusionRule))
                .unwrap_err();
        assert!(matches!(
            error,
            OperationError::FusionAlgebra(cause)
                if *cause == FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1,
                }
        ));
        assert_eq!(original.admission(), &admission);
        assert_eq!(
            scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
            0
        );
    }
}

#[test]
fn lowered_bind_preserves_omitted_product_codec_failure() {
    // What: an invalid packed-product sector omitted by a valid Subset is
    // reported exactly when complete-grid preparation reaches it, without publication.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = lowered_product_rule();
    let valid = rule.vacuum();
    let invalid = SectorId::new(usize::MAX);
    let expected = TripleCodec::decode_checked(invalid).unwrap_err();
    let pair = FusionTreePairKey::try_pair_from_sector_ids(
        [valid.id()],
        [valid.id()],
        valid.id(),
        [false],
        [false],
        [],
        [],
        [],
        [],
    )
    .unwrap();
    let leg = || SectorLeg::new([(valid, 1), (invalid, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let structure =
        crate::tests::packed_fixture_structure(2, [(BlockKey::FusionTree(pair), vec![1, 1])])
            .unwrap()
            .into_shared();
    let raw = DynamicFusionMapSpace {
        nout: 1,
        nin: 1,
        homspace: Arc::new(homspace),
        subblock_structure: structure,
        admission: FusionSpaceAdmission::Subset(rule.rule_identity()),
        adjoint: OnceLock::new(),
    };
    clear_structure_caches();
    let scratch_structure_builds_rec = probe::record();

    let error = BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(rule))
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::ProductCodec(expected)))
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn lowered_subset_to_complete_validates_grid_before_commit() {
    // What: a checked sparse subset remains a subset when complete-grid
    // validation fails, and the prepared lowered layout is not published.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let complete = z2_matrix_space();
    let first = complete.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = DynamicFusionMapSpace::try_from_layout(complete.homspace().clone(), structure)
        .unwrap()
        .try_bind_rule(&Z2FusionRule)
        .unwrap();
    let raw = subset;
    let original = raw.clone();
    let scratch_structure_builds_rec = probe::record();

    assert!(matches!(
        BoundDynamicFusionMapSpace::bind_multiplicity_free_lowered(raw, Arc::new(Z2FusionRule),),
        Err(OperationError::Core(CoreError::BlockCountMismatch { .. }))
    ));
    assert!(matches!(
        original.admission(),
        FusionSpaceAdmission::Subset(_)
    ));
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn derived_and_validated_paths_reject_subset_proofs() {
    // What: internal derived construction and validated-layout rebinding
    // require an executable Complete proof in release builds.
    let complete = z2_matrix_space();
    let subset = DynamicFusionMapSpace::try_from_layout(
        complete.homspace().clone(),
        BlockStructure::clone(complete.structure()),
    )
    .unwrap()
    .try_bind_rule(&Z2FusionRule)
    .unwrap();
    let provider = Arc::new(Z2FusionRule);
    let expected = OperationError::StructureMismatch {
        tensor: "complete fusion layout",
    };

    assert_eq!(
        BoundDynamicFusionMapSpace::from_derived(Arc::clone(&provider), subset.clone())
            .unwrap_err(),
        expected
    );
    let authority = BoundDynamicFusionMapSpace::from_derived(provider, complete).unwrap();
    assert_eq!(
        authority
            .rebind_validated(&ValidatedDynamicFusionLayout(subset))
            .unwrap_err(),
        expected
    );
}

#[test]
fn bound_space_rejects_incoherent_axis_split() {
    // What: bound admission and the public borrowed tensor proof both reject
    // a dynamic rank that disagrees with its HomSpace.
    let raw = DynamicFusionMapSpace {
        nout: 0,
        nin: 1,
        adjoint: OnceLock::new(),
        ..matrix_space()
    };

    let corrupted = BoundDynamicFusionMapSpace {
        space: raw.clone(),
        provider: Arc::new(Z2FusionRule),
        layout_build: LayoutBuildCapability::encoded(),
    };
    assert!(matches!(
        BoundDynamicTensorRef::<_, f64>::try_new(&corrupted, &[]),
        Err(OperationError::Core(
            CoreError::StructureRankMismatch { .. }
        ))
    ));

    let error = BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(Z2FusionRule))
        .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionSpaceSplitMismatch { .. })
    ));
}

#[test]
fn bound_space_rejects_incoherent_structure_rank() {
    // What: bound admission and the public borrowed tensor proof both reject
    // storage whose rank differs from its HomSpace.
    let raw = matrix_space();
    let block = raw.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::with_key(block.key().clone(), vec![1], vec![1], 0).unwrap()],
    )
    .unwrap();
    let raw = DynamicFusionMapSpace {
        subblock_structure: Arc::new(structure),
        adjoint: OnceLock::new(),
        ..raw
    };

    let corrupted = BoundDynamicFusionMapSpace {
        space: raw.clone(),
        provider: Arc::new(Z2FusionRule),
        layout_build: LayoutBuildCapability::encoded(),
    };
    assert!(matches!(
        BoundDynamicTensorRef::<_, f64>::try_new(&corrupted, &[]),
        Err(OperationError::Core(
            CoreError::StructureRankMismatch { .. }
        ))
    ));

    let error = BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(Z2FusionRule))
        .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::StructureRankMismatch { .. })
    ));
}

#[test]
fn direct_bound_builders_keep_coherent_split_and_rank() {
    // What: direct multiplicity-free and Generic builders satisfy bound
    // invariants, and final-HomSpace Generic layout exactly matches the
    // caller-shape expert layout for a genuine outer multiplicity.
    let provider = Arc::new(Z2FusionRule);
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let multiplicity_free = BoundDynamicFusionMapSpace::from_degeneracy_shapes(
        Arc::clone(&provider),
        homspace.clone(),
        [vec![1, 1]],
    )
    .unwrap();
    let generic_provider = Arc::new(GenericMultiplicityRule);
    let charge = SectorId::new(1);
    let generic_homspace = FusionTreeHomSpace::from_sector_ids(
        [(charge.id(), 2), (charge.id(), 3)],
        [(charge.id(), 4)],
    );
    let generic_key_count = generic_homspace
        .fusion_tree_keys_generic(generic_provider.as_ref())
        .unwrap()
        .len();
    assert_eq!(generic_key_count, 2);
    let expert = BoundDynamicFusionMapSpace::from_degeneracy_shapes_generic(
        Arc::clone(&generic_provider),
        generic_homspace.clone(),
        vec![vec![2, 3, 4]; generic_key_count],
    )
    .unwrap();
    let generic =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(generic_provider, generic_homspace)
            .unwrap();

    assert_eq!(multiplicity_free.space().nout(), 1);
    assert_eq!(generic.space().nin(), 1);
    assert_eq!(multiplicity_free.space().structure().rank(), 2);
    assert_eq!(generic.space().structure().rank(), 3);
    assert_eq!(generic.space().structure(), expert.space().structure());
    assert_eq!(
        generic.space().required_len().unwrap(),
        expert.space().required_len().unwrap()
    );
    for index in 0..expert.space().structure().block_count() {
        let actual = generic.space().structure().block(index).unwrap();
        let expected = expert.space().structure().block(index).unwrap();
        assert_eq!(actual.key(), expected.key());
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.strides(), expected.strides());
        assert_eq!(actual.offset(), expected.offset());
    }
}

#[test]
fn generic_completeness_requires_every_vertex_label() {
    // What: N(1,1,1)=2 requires both vertex labels 1 and 2 in a Complete grid.
    let provider = Arc::new(GenericMultiplicityRule);
    let homspace = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1)]);
    let keys = homspace
        .fusion_tree_keys_generic(provider.as_ref())
        .unwrap();
    assert_eq!(
        keys.iter()
            .map(|key| key.codomain_vertices()[0].get())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let complete = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
        provider.as_ref(),
        homspace,
        [vec![1, 1, 1], vec![1, 1, 1]],
    )
    .unwrap();
    assert_eq!(complete.structure().block_count(), 2);
    assert!(matches!(
        complete.admission(),
        FusionSpaceAdmission::Complete(_)
    ));

    let first = complete.structure().block(0).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        complete.rank(),
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = DynamicFusionMapSpace {
        subblock_structure: Arc::new(structure),
        admission: FusionSpaceAdmission::Subset(provider.rule_identity()),
        adjoint: OnceLock::new(),
        ..complete
    };

    let error = BoundDynamicFusionMapSpace::bind_generic(subset, provider).unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::BlockCountMismatch {
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn generic_adjoint_rejects_multiplicity_free_before_final_layout_build() {
    // What: a Generic adjoint rejects a multiplicity-free provider before
    // constructing its target layout.
    let source =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(matrix_space(), Arc::new(Z2FusionRule))
            .unwrap();
    let final_result_layout_builds_rec = probe::record();

    let error = crate::adjoint::adjoint_bound_space_dyn_generic(&source).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        0
    );
}

#[test]
fn memoized_adjoint_view_reads_the_current_admission() {
    // What: admission is reassigned after construction when a layout is
    // promoted, so the adjoint view reads it at call time; only the
    // admission-independent hom space and structure are memoized.
    let complete = z2_matrix_space();
    let first = complete.adjoint_view().unwrap();
    let mut unbound = complete.clone();
    unbound.admission = FusionSpaceAdmission::Unbound;
    let second = unbound.adjoint_view().unwrap();
    assert_eq!(second.admission(), &FusionSpaceAdmission::Unbound);
    assert_eq!(first.admission(), complete.admission());
    assert!(Arc::ptr_eq(second.structure(), first.structure()));
    assert!(Arc::ptr_eq(second.homspace_arc(), first.homspace_arc()));
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; this isolated spy uses Cell counters.
fn checked_adjoint_provider_derivation_cannot_republish_after_reset() {
    if crate::test_support::run_isolated_or_return(
        "SYMTN_CHECKED_ADJOINT_PRODUCER_RESET", "contract::dynamic_space::bound_invariant_tests::checked_adjoint_provider_derivation_cannot_republish_after_reset",
    ) { return; }
    let provider = Arc::new(FailAtCallRule::new());
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new((0..2).map(|_| SectorLeg::new([(SectorId::new(0), 2)], true))),
        FusionProductSpace::new((0..2).map(|_| SectorLeg::new([(SectorId::new(0), 3)], true))),
    );
    let source =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(Arc::clone(&provider), hom)
            .unwrap();
    clear_structure_caches();
    provider.reset_on_query.set(true);
    let stale = crate::adjoint_bound_space_dyn_generic_checked(&source).unwrap();
    assert!(!provider.reset_on_query.get());
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure).entries(),
        0
    );
    let fresh = crate::adjoint_bound_space_dyn_generic_checked(&source).unwrap();
    assert_eq!(stale.space().structure(), fresh.space().structure());
    assert_ne!(
        stale.space().structure().content_id(),
        fresh.space().structure().content_id()
    );
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure).entries(),
        1
    );
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The bound API requires Arc; this isolated spy uses Cell counters.
fn checked_adjoint_of_a_subset_source_reports_the_first_missing_target_key() {
    // What (#1942): only a Complete source skips the adjoint key check; a
    // Subset source still enumerates the adjoint trees and reports the first
    // target-ordered key whose swapped source key is absent.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let provider = Arc::new(FailAtCallRule::new());
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new((0..2).map(|_| SectorLeg::new([(SectorId::new(0), 2)], true))),
        FusionProductSpace::new((0..2).map(|_| SectorLeg::new([(SectorId::new(0), 3)], true))),
    );
    let complete =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(Arc::clone(&provider), hom)
            .unwrap();
    let full = complete.space().structure();
    assert!(full.block_count() > 2);
    let kept = full.block(full.block_count() - 1).unwrap();
    let structure = BlockStructure::from_blocks_with_rank(
        full.rank(),
        vec![BlockSpec::with_key(
            kept.key().clone(),
            kept.shape().to_vec(),
            kept.strides().to_vec(),
            kept.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let subset = BoundDynamicFusionMapSpace {
        space: DynamicFusionMapSpace {
            subblock_structure: structure.into_shared(),
            admission: FusionSpaceAdmission::Subset(provider.identity.clone()),
            adjoint: OnceLock::new(),
            ..complete.space().clone()
        },
        provider: Arc::clone(&provider),
        layout_build: complete.layout_build,
    };

    let homspace = complete.space().homspace();
    let expected = FusionTreeHomSpace::new(homspace.domain().clone(), homspace.codomain().clone())
        .fusion_tree_keys_generic_checked(provider.as_ref())
        .unwrap()
        .into_iter()
        .map(|key| {
            BlockKey::FusionTree(FusionTreePairKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ))
        })
        .find(|key| {
            subset
                .space()
                .structure()
                .find_block_index_by_key(key)
                .is_none()
        })
        .unwrap();

    assert!(crate::adjoint_bound_space_dyn_generic_checked(&complete).is_ok());
    match crate::adjoint_bound_space_dyn_generic_checked(&subset) {
        Err(crate::CheckedGenericPlanError::Operation(OperationError::MissingBlockKey { key })) => {
            assert_eq!(*key, expected);
        }
        other => panic!("expected the first missing target key, got {other:?}"),
    }
}
