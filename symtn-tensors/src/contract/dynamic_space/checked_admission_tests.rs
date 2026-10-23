use super::*;
use crate::test_support::CACHE_TEST_LOCK;
use symtn_core::{
    clear_structure_caches, structure_cache_info, BraidingStyleKind, FusionAlgebraError,
    FusionProductSpace, FusionStyleKind, MultiplicityFreeFusionSymbols, RuleIdentity,
    SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorVec, StructureCacheKind,
};

#[derive(Clone, Copy)]
enum CheckedFailStage {
    Dual,
    Channels,
    Nsymbol,
}

// External abelian (Unique) provider with XOR fusion on sectors {0, 1}.
// It certifies only CheckedFusionAlgebra + MultiplicityFreeRigidSymbols,
// never the sealed built-in lowered codec, so it exercises the checked
// admission path as a genuine downstream provider would. An optional stage
// flag makes exactly one checked primitive fail.
#[derive(Clone, Copy)]
struct ExternalUniqueRule {
    fail: Option<CheckedFailStage>,
}

impl ExternalUniqueRule {
    fn new() -> Self {
        Self { fail: None }
    }
}

impl symtn_core::testing::Provider for ExternalUniqueRule {}

impl FusionRule for ExternalUniqueRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
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
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(left.id() ^ right.id())).collect()
    }
}

impl MultiplicityFreeFusionRule for ExternalUniqueRule {}

impl MultiplicityFreeFusionSymbols for ExternalUniqueRule {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for ExternalUniqueRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

impl CheckedFusionAlgebra for ExternalUniqueRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if matches!(self.fail, Some(CheckedFailStage::Dual)) {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        if matches!(self.fail, Some(CheckedFailStage::Channels)) {
            return Err(FusionAlgebraError::FusionNotRepresentable { left, right });
        }
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        if matches!(self.fail, Some(CheckedFailStage::Nsymbol)) {
            return Err(FusionAlgebraError::MultiplicityOverflow {
                left,
                right,
                coupled,
            });
        }
        Ok(self.nsymbol(left, right, coupled))
    }
}

// External Simple (multiplicity-free non-abelian) provider that reuses the
// SU(2) fusion mathematics but is a distinct type without the sealed
// lowered codec, so a genuine Simple external provider drives the checked
// path. Delegation keeps the categorical semantics honest.
#[derive(Clone, Copy)]
struct ExternalSimpleRule;

impl symtn_core::testing::Provider for ExternalSimpleRule {}

impl FusionRule for ExternalSimpleRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        SU2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for ExternalSimpleRule {}

impl MultiplicityFreeFusionSymbols for ExternalSimpleRule {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        l: SectorId,
        m: SectorId,
        r: SectorId,
        c: SectorId,
        lc: SectorId,
        rc: SectorId,
    ) -> f64 {
        SU2FusionRule.f_symbol_scalar(l, m, r, c, lc, rc)
    }
    fn r_symbol_scalar(&self, l: SectorId, r: SectorId, c: SectorId) -> f64 {
        SU2FusionRule.r_symbol_scalar(l, r, c)
    }
}

impl MultiplicityFreeRigidSymbols for ExternalSimpleRule {
    fn dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.dim_scalar(s)
    }
    fn inv_dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.inv_dim_scalar(s)
    }
    fn sqrt_dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.sqrt_dim_scalar(s)
    }
    fn inv_sqrt_dim_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.inv_sqrt_dim_scalar(s)
    }
    fn twist_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.twist_scalar(s)
    }
    fn frobenius_schur_phase_scalar(&self, s: SectorId) -> f64 {
        SU2FusionRule.frobenius_schur_phase_scalar(s)
    }
}

impl CheckedFusionAlgebra for ExternalSimpleRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(self.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(self.nsymbol(left, right, coupled))
    }
}

/// Degeneracies of sectors 0 and 1 on the two codomain legs of
/// [`unique_matrix_homspace`]; its domain legs are their duals in the same
/// order, so the domain of one copy contracts with the codomain of another.
const UNIQUE_LEGS: [[usize; 2]; 2] = [[2, 3], [1, 4]];

fn unique_matrix_homspace() -> FusionTreeHomSpace {
    // Rank-4 XOR space whose codomain and domain each couple to {0, 1}.
    // The domain legs are dual so every fixture built from this space also
    // covers the dual-leg duality flags in the checked layout.
    let leg = |[even, odd]: [usize; 2], dual| {
        SectorLeg::new([(SectorId::new(0), even), (SectorId::new(1), odd)], dual)
    };
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(UNIQUE_LEGS[0], false), leg(UNIQUE_LEGS[1], false)]),
        FusionProductSpace::new([leg(UNIQUE_LEGS[0], true), leg(UNIQUE_LEGS[1], true)]),
    )
}

/// One block: codomain sectors and dual flags, domain sectors and dual
/// flags, coupled sector, shape.
type BlockRow = (
    Vec<usize>,
    Vec<bool>,
    Vec<usize>,
    Vec<bool>,
    usize,
    Vec<usize>,
);

fn block_rows(space: &DynamicFusionMapSpace) -> Vec<BlockRow> {
    let ids = |sectors: &[SectorId]| sectors.iter().map(|s| s.id()).collect::<Vec<_>>();
    let mut rows = (0..space.structure().block_count())
        .map(|index| {
            let block = space.structure().block(index).unwrap();
            let key = block.key().as_fusion_tree_pair().unwrap();
            (
                ids(key.codomain_uncoupled()),
                key.codomain_is_dual().to_vec(),
                ids(key.domain_uncoupled()),
                key.domain_is_dual().to_vec(),
                key.coupled().id(),
                block.shape().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

/// Hand oracle for the XOR rule, where every sector is self-dual and a
/// rank-2 tree has no inner line: one block per sector assignment whose
/// codomain and domain XORs agree (that XOR is the coupled sector), shaped
/// by the leg degeneracies.
fn xor_rows(codomain: &[([usize; 2], bool)], domain: &[([usize; 2], bool)]) -> Vec<BlockRow> {
    let legs = codomain.len() + domain.len();
    let mut rows = Vec::new();
    for mask in 0..1usize << legs {
        let sectors = (0..legs).map(|leg| (mask >> leg) & 1).collect::<Vec<_>>();
        let (cod, dom) = sectors.split_at(codomain.len());
        let coupled = cod.iter().fold(0, |acc, s| acc ^ s);
        if dom.iter().fold(0, |acc, s| acc ^ s) != coupled {
            continue;
        }
        let shape = codomain
            .iter()
            .chain(domain)
            .zip(&sectors)
            .map(|((degeneracy, _), &sector)| degeneracy[sector])
            .collect();
        rows.push((
            cod.to_vec(),
            codomain.iter().map(|&(_, dual)| dual).collect(),
            dom.to_vec(),
            domain.iter().map(|&(_, dual)| dual).collect(),
            coupled,
            shape,
        ));
    }
    rows.sort();
    rows
}

/// Packed storage of a complete grid: the tree blocks tile each coupled
/// matrix, so the length is the sum of the block volumes.
fn packed_len(rows: &[BlockRow]) -> usize {
    rows.iter().map(|row| row.5.iter().product::<usize>()).sum()
}

fn simple_matrix_homspace() -> FusionTreeHomSpace {
    // SU(2) spin-1/2 x spin-1/2 on each side couples to spin 0 and spin 1.
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let leg = |dual| SectorLeg::new([(half, 2)], dual);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(false)]),
        FusionProductSpace::new([leg(false), leg(false)]),
    )
}

#[test]
fn checked_final_root_builds_the_hand_enumerated_multi_block_layout() {
    // What: external Unique and Simple providers build, through the checked
    // root, a complete runtime-rank layout whose blocks are exactly the
    // hand-enumerated fusion-tree pairs with their leg degeneracies.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    clear_structure_caches();
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(ExternalUniqueRule::new()),
        unique_matrix_homspace(),
    )
    .unwrap();
    let [first, second] = UNIQUE_LEGS;
    let expected = xor_rows(
        &[(first, false), (second, false)],
        &[(first, true), (second, true)],
    );
    assert_eq!(expected.len(), 8);
    assert!(matches!(
        checked.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));
    assert_eq!(block_rows(checked.space()), expected);
    assert_eq!(
        checked.space().required_len().unwrap(),
        packed_len(&expected)
    );

    clear_structure_caches();
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(ExternalSimpleRule),
        simple_matrix_homspace(),
    )
    .unwrap();
    // spin-1/2 x spin-1/2 = spin 0 + spin 1 on both sides, one tree each.
    let half = SU2Irrep::from_twice_spin(1).sector_id().id();
    let row = |twice_spin| {
        (
            vec![half, half],
            vec![false, false],
            vec![half, half],
            vec![false, false],
            SU2Irrep::from_twice_spin(twice_spin).sector_id().id(),
            vec![2, 2, 2, 2],
        )
    };
    let mut expected = vec![row(0), row(2)];
    expected.sort();
    assert_eq!(block_rows(checked.space()), expected);
    assert_eq!(checked.space().required_len().unwrap(), 32);
}

#[test]
fn separately_allocated_equal_identity_providers_build_equivalent_structures() {
    // What: two independent provider allocations with the same RuleIdentity
    // admit content-identical checked layouts.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let first = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(ExternalUniqueRule::new()),
        unique_matrix_homspace(),
    )
    .unwrap();
    let second = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(ExternalUniqueRule::new()),
        unique_matrix_homspace(),
    )
    .unwrap();
    assert!(!Arc::ptr_eq(first.provider_arc(), second.provider_arc()));
    assert_eq!(first.space(), second.space());
}

#[test]
fn checked_final_root_stage_failures_publish_no_layout_or_admission() {
    // Isolated like the neighboring checked_bind_failure test (#649/#650):
    // this asserts an absolute process-global cache snapshot, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked encoded constructions.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_CHECKED_FINAL_ROOT_STAGE_FAILURES_ISOLATED",
        "contract::dynamic_space::checked_admission_tests::checked_final_root_stage_failures_publish_no_layout_or_admission",
    ) {
        return;
    }
    // What: an injected failure in dual, channels, or nsymbol returns the
    // exact typed error and leaves both process-global cache snapshots and
    // scratch publication observations unchanged.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    for (stage, expected) in [
        (
            CheckedFailStage::Dual,
            FusionAlgebraError::InvalidSector {
                sector: SectorId::new(0),
            },
        ),
        (
            CheckedFailStage::Channels,
            FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(0),
                right: SectorId::new(0),
            },
        ),
        (
            CheckedFailStage::Nsymbol,
            FusionAlgebraError::MultiplicityOverflow {
                left: SectorId::new(0),
                right: SectorId::new(0),
                coupled: SectorId::new(0),
            },
        ),
    ] {
        symtn_core::clear_structure_caches();
        clear_structure_caches();
        let before = (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        );

        let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            Arc::new(ExternalUniqueRule { fail: Some(stage) }),
            unique_matrix_homspace(),
        )
        .unwrap_err();

        assert_eq!(error, OperationError::FusionAlgebra(Box::new(expected)));
        assert_eq!(
            (
                structure_cache_info(StructureCacheKind::SectorStructure),
                structure_cache_info(StructureCacheKind::DegeneracyStructure),
            ),
            before
        );
    }
}

#[test]
fn checked_bound_space_permute_and_contract_build_hand_enumerated_layouts() {
    // What: the reachable Permute and Contract dispatcher arms of a checked
    // capability build the hand-enumerated output layouts. The Select and
    // OutwardLeg arms have no public BoundDynamicFusionMapSpace caller that
    // routes through the stored capability (their capability methods are
    // dead code outside the encoded and lowered helpers), so they are not
    // reachable to smoke-test here.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();

    let checked = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(ExternalUniqueRule::new()),
        unique_matrix_homspace(),
    )
    .unwrap();
    let [first, second] = UNIQUE_LEGS;

    // Swapping the two codomain legs and the two domain legs swaps their
    // degeneracies; distinct per-leg degeneracies make the swap visible.
    let permuted = checked
        .transformed_multiplicity_free(&TreeTransformOperation::permute([1, 0], [3, 2]))
        .unwrap();
    let expected = xor_rows(
        &[(second, false), (first, false)],
        &[(second, true), (first, true)],
    );
    assert_eq!(block_rows(permuted.space()), expected);
    assert_eq!(
        permuted.space().required_len().unwrap(),
        packed_len(&expected)
    );

    // The outer product keeps every lhs axis in the codomain and every rhs
    // axis in the domain. A leg crossing the split is dualized (TensorKit
    // `permute`): the lhs dual domain legs become plain codomain legs and
    // the rhs plain codomain legs become dual domain legs.
    let outer =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&checked, &checked, &[], &[])
            .unwrap();
    let plain = [(first, false), (second, false)];
    let dual = [(first, true), (second, true)];
    let expected = xor_rows(&[plain, plain].concat(), &[dual, dual].concat());
    assert_eq!(expected.len(), 128);
    assert_eq!(block_rows(outer.space()), expected);
    assert_eq!(outer.space().required_len().unwrap(), packed_len(&expected));
}

#[test]
fn checked_bind_failure_preserves_subset_admission_and_caches() {
    const ISOLATED: &str = "SYMTN_CHECKED_BIND_FAILURE_ISOLATED";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "contract::dynamic_space::checked_admission_tests::checked_bind_failure_preserves_subset_admission_and_caches",
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

    // What: a checked bind fails transactionally at every checked stage —
    // structural-proof algebra (channels, nsymbol) and the prepare leg-dual
    // guard — keeping the caller's Subset admission and both cache snapshots
    // unchanged.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    for stage in [
        CheckedFailStage::Dual,
        CheckedFailStage::Channels,
        CheckedFailStage::Nsymbol,
    ] {
        symtn_core::clear_structure_caches();
        clear_structure_caches();
        let good = DynamicFusionMapSpace::from_final_homspace(
            &ExternalUniqueRule::new(),
            unique_matrix_homspace(),
        )
        .unwrap();
        let mut subset = good.clone();
        subset.admission = FusionSpaceAdmission::Subset(ExternalUniqueRule::new().rule_identity());
        let original = subset.clone();

        let before = (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        );

        let error = BoundDynamicFusionMapSpace::bind_multiplicity_free_checked(
            subset,
            Arc::new(ExternalUniqueRule { fail: Some(stage) }),
        )
        .unwrap_err();

        assert!(matches!(error, OperationError::FusionAlgebra(_)));
        assert!(matches!(
            original.admission(),
            FusionSpaceAdmission::Subset(_)
        ));
        assert_eq!(
            (
                structure_cache_info(StructureCacheKind::SectorStructure),
                structure_cache_info(StructureCacheKind::DegeneracyStructure),
            ),
            before
        );
    }
}
