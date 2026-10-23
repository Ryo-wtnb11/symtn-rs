use super::*;

pub(super) use smallvec::smallvec;

pub(super) use std::hash::{Hash, Hasher};

/// Fixture layout: subblocks packed contiguously in key order. Not a product
/// layout (the only one is the coupled sector matrix); fixtures use it to
/// exercise the arbitrary-strided-view contract of [`BlockStructure`].
pub(super) fn packed_fixture_structure<I, K>(
    rank: usize,
    blocks: I,
) -> Result<BlockStructure, CoreError>
where
    I: IntoIterator<Item = (K, Vec<usize>)>,
    K: Into<BlockKey>,
{
    let mut keys = Vec::new();
    let mut shapes = Vec::new();
    for (key, shape) in blocks {
        keys.push(key.into());
        shapes.push(shape);
    }
    BlockStructure::from_parts(
        SectorStructure::from_keys(rank, keys)?,
        DegeneracyStructure::packed_column_major(rank, shapes)?,
    )
}

pub(super) fn u1(charge: i32) -> SectorId {
    U1Irrep::new(charge).sector_id()
}

pub(super) fn excluded_u1_id() -> SectorId {
    SectorId::new(u32::MAX as usize)
}

pub(super) fn z2_even() -> SectorId {
    Z2Irrep::EVEN.sector_id()
}

pub(super) fn z2_odd() -> SectorId {
    Z2Irrep::ODD.sector_id()
}

pub(super) fn su2(twice_spin: usize) -> SectorId {
    SU2Irrep::from_twice_spin(twice_spin).sector_id()
}

#[derive(Clone, Copy, Debug)]
pub(super) struct IsomorphismMultiplicityRule;

impl symtn_sectors::testing::Provider for IsomorphismMultiplicityRule {}

impl FusionRule for IsomorphismMultiplicityRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, sector) | (sector, 0) => smallvec![SectorId::new(sector)],
            (1, 1) => smallvec![SectorId::new(2)],
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 2) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Z4PointedRule;

impl symtn_sectors::testing::Provider for Z4PointedRule {}

impl FusionRule for Z4PointedRule {
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

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((4 - sector.id() % 4) % 4)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        smallvec![SectorId::new((left.id() + right.id()) % 4)]
    }
}

impl MultiplicityFreeFusionRule for Z4PointedRule {}

#[derive(Clone, Copy, Debug)]
pub(super) struct PlanarZ2Rule;

impl symtn_sectors::testing::Provider for PlanarZ2Rule {}

impl FusionRule for PlanarZ2Rule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::NoBraiding
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        smallvec![SectorId::new((left.id() + right.id()) % 2)]
    }
}

impl MultiplicityFreeFusionRule for PlanarZ2Rule {}

impl MultiplicityFreeFusionSymbols for PlanarZ2Rule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for PlanarZ2Rule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct IdentitySymbolPanicRule;

impl symtn_sectors::testing::Provider for IdentitySymbolPanicRule {}

impl FusionRule for IdentitySymbolPanicRule {
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
        smallvec![SectorId::new(left.id() ^ right.id())]
    }
}

impl MultiplicityFreeFusionRule for IdentitySymbolPanicRule {}

impl MultiplicityFreeFusionSymbols for IdentitySymbolPanicRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        panic!("identity braid evaluated an F symbol")
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        panic!("identity braid evaluated an R symbol")
    }
}

impl MultiplicityFreeRigidSymbols for IdentitySymbolPanicRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Debug, Default)]
pub(super) struct SplitOnlyCountingRule {
    pub(super) n_calls: std::sync::atomic::AtomicUsize,
    pub(super) f_calls: std::sync::atomic::AtomicUsize,
    pub(super) r_calls: std::sync::atomic::AtomicUsize,
}

impl symtn_sectors::testing::Provider for SplitOnlyCountingRule {}

impl FusionRule for SplitOnlyCountingRule {
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
        smallvec![SectorId::new(left.id() ^ right.id())]
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.n_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        usize::from(left.id() ^ right.id() == coupled.id())
    }
}

impl MultiplicityFreeFusionRule for SplitOnlyCountingRule {}

impl MultiplicityFreeFusionSymbols for SplitOnlyCountingRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        self.f_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        self.r_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for SplitOnlyCountingRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

pub(super) fn split_only_tree_pair_route_composition_oracle<R>(
    rule: &R,
    source: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    // What: freeze the pre-shortcut composition as an independent oracle:
    // repartition to all-codomain, apply the identity tree braid, then
    // repartition back to the requested split.
    let total_rank =
        source.codomain_tree().uncoupled().len() + source.domain_tree().uncoupled().len();
    let identity = (0..total_rank).collect::<Vec<_>>();
    let levels = identity.clone();
    let all_codomain = multiplicity_free_repartition_tree_pair(rule, source, total_rank)?;
    let braided = compose_terms(all_codomain, |key| {
        multiplicity_free_braid_tree(rule, key.codomain_tree(), &identity, &levels).map(|terms| {
            terms
                .into_iter()
                .map(|(tree, coefficient)| {
                    (
                        FusionTreePairKey::pair(tree, key.domain_tree().clone()),
                        coefficient,
                    )
                })
                .collect::<Vec<_>>()
        })
    })?;
    multiplicity_free_repartition_terms(rule, braided, target_codomain_rank)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AsymmetricAnyonicRule;

impl symtn_sectors::testing::Provider for AsymmetricAnyonicRule {}

impl FusionRule for AsymmetricAnyonicRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(3)],
            (3, 1) | (1, 3) => smallvec![SectorId::new(2)],
            (3, 2) | (2, 3) => smallvec![SectorId::new(1)],
            _ => smallvec![SectorId::new((left.id() + right.id()) % 4)],
        }
    }
}

impl MultiplicityFreeFusionRule for AsymmetricAnyonicRule {}

impl MultiplicityFreeFusionSymbols for AsymmetricAnyonicRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        11.0
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        match (left.id(), right.id()) {
            (1, 2) => 5.0,
            (2, 1) => 7.0,
            (3, 2) => 13.0,
            (2, 3) => 17.0,
            (1, 3) => 19.0,
            (3, 1) => 23.0,
            _ => 1.0,
        }
    }
}

impl MultiplicityFreeRigidSymbols for AsymmetricAnyonicRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

pub(super) fn sector_ids(sectors: &[SectorId]) -> Vec<usize> {
    sectors.iter().map(|sector| sector.id()).collect()
}

#[derive(Debug, Default)]
pub(super) struct CheckedTreeProbe {
    pub(super) channel_calls: AtomicUsize,
    pub(super) nsymbol_calls: AtomicUsize,
    pub(super) unchecked_nsymbol_calls: AtomicUsize,
}

impl symtn_sectors::testing::Provider for CheckedTreeProbe {}

impl FusionRule for CheckedTreeProbe {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }

    fn nsymbol(&self, _left: SectorId, _right: SectorId, _coupled: SectorId) -> usize {
        self.unchecked_nsymbol_calls.fetch_add(1, Ordering::Relaxed);
        1
    }
}

impl CheckedFusionAlgebra for CheckedTreeProbe {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.channel_calls.fetch_add(1, Ordering::Relaxed);
        if left == SectorId::new(9) {
            Err(FusionAlgebraError::FusionNotRepresentable { left, right })
        } else {
            Ok(self.fusion_channels(left, right))
        }
    }

    fn try_nsymbol(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.nsymbol_calls.fetch_add(1, Ordering::Relaxed);
        Ok(1)
    }
}

pub(super) struct FibonacciFAdmissibilityProbe {
    calls: std::sync::Mutex<Vec<[SectorId; 6]>>,
    complex_f_phase: bool,
}

impl FibonacciFAdmissibilityProbe {
    pub(super) const SENTINEL: Complex64 = Complex64::new(97.0, -31.0);

    pub(super) fn new() -> Self {
        Self {
            calls: std::sync::Mutex::new(Vec::new()),
            complex_f_phase: false,
        }
    }

    pub(super) fn with_complex_f_phase() -> Self {
        Self {
            calls: std::sync::Mutex::new(Vec::new()),
            complex_f_phase: true,
        }
    }

    pub(super) fn take_calls(&self) -> Vec<[SectorId; 6]> {
        std::mem::take(
            &mut *self
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

impl symtn_sectors::testing::Provider for FibonacciFAdmissibilityProbe {}

impl FusionRule for FibonacciFAdmissibilityProbe {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        TableRule::fibonacci().fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        TableRule::fibonacci().braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        TableRule::fibonacci().vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        TableRule::fibonacci().dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        TableRule::fibonacci().fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        TableRule::fibonacci().nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for FibonacciFAdmissibilityProbe {}

impl MultiplicityFreeFusionSymbols for FibonacciFAdmissibilityProbe {
    type Scalar = Complex64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let call = [left, middle, right, coupled, left_coupled, right_coupled];
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(call);
        let admissible = TableRule::fibonacci().nsymbol(left, middle, left_coupled) != 0
            && TableRule::fibonacci().nsymbol(left_coupled, right, coupled) != 0
            && TableRule::fibonacci().nsymbol(middle, right, right_coupled) != 0
            && TableRule::fibonacci().nsymbol(left, right_coupled, coupled) != 0;
        if admissible {
            let value = TableRule::fibonacci().f_symbol_scalar(
                left,
                middle,
                right,
                coupled,
                left_coupled,
                right_coupled,
            );
            if self.complex_f_phase {
                value * Complex64::new(0.6, 0.8)
            } else {
                value
            }
        } else {
            Self::SENTINEL
        }
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        TableRule::fibonacci().r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for FibonacciFAdmissibilityProbe {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        TableRule::fibonacci().dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        TableRule::fibonacci().inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        TableRule::fibonacci().sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        TableRule::fibonacci().inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        TableRule::fibonacci().twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        TableRule::fibonacci().frobenius_schur_phase_scalar(sector)
    }
}

pub(super) fn assert_fibonacci_f_calls_are_admissible(calls: &[[SectorId; 6]]) {
    for &[left, middle, right, coupled, left_coupled, right_coupled] in calls {
        assert_ne!(
            TableRule::fibonacci().nsymbol(left, middle, left_coupled),
            0
        );
        assert_ne!(
            TableRule::fibonacci().nsymbol(left_coupled, right, coupled),
            0
        );
        assert_ne!(
            TableRule::fibonacci().nsymbol(middle, right, right_coupled),
            0
        );
        assert_ne!(
            TableRule::fibonacci().nsymbol(left, right_coupled, coupled),
            0
        );
    }
}

pub(super) fn tree_pair_group_fixture(
    codomain: &[usize],
    domain: &[usize],
    coupled: usize,
    codomain_dual: &[bool],
    domain_dual: &[bool],
) -> FusionTreePairKey {
    let codomain_vertices = vec![1; codomain.len().saturating_sub(1)];
    let domain_vertices = vec![1; domain.len().saturating_sub(1)];
    FusionTreePairKey::try_pair_from_sector_ids(
        codomain.iter().copied(),
        domain.iter().copied(),
        coupled,
        codomain_dual.iter().copied(),
        domain_dual.iter().copied(),
        [],
        [],
        codomain_vertices,
        domain_vertices,
    )
    .unwrap()
}

pub(super) fn assert_mixed_tree_pair_block_group_is_rejected<R>(
    rule: &R,
    keys: &[FusionTreePairKey],
    expected: CoreError,
) where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + std::fmt::Debug,
{
    // What: identity and non-identity braid entry paths reject before
    // returning a shared coefficient matrix or evaluating rigid symbols.
    assert_eq!(
        multiplicity_free_braid_tree_pair_block(rule, keys, &[0, 1], &[2], &[0, 1], &[2],)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_braid_tree_pair_block(rule, keys, &[1, 0], &[2], &[0, 1], &[2],)
            .unwrap_err(),
        expected
    );

    // What: symmetric permutation validates the same block invariant on
    // both its identity shortcut and general braid delegation.
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(rule, keys, &[0, 1], &[2]).unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(rule, keys, &[1, 0], &[2]).unwrap_err(),
        expected
    );

    // What: planar transpose validates before either its identity return or
    // cyclic repartition path can consume symbols.
    assert_eq!(
        multiplicity_free_transpose_tree_pair_block(rule, keys, &[0, 1], &[2]).unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_transpose_tree_pair_block(rule, keys, &[1, 2], &[0]).unwrap_err(),
        expected
    );
}

pub(super) fn materialized_leg_tuple_oracle(space: &FusionProductSpace) -> Vec<Vec<FusionTreeLeg>> {
    fn visit(
        legs: &[SectorLeg],
        remaining: usize,
        current: &mut [FusionTreeLeg],
        out: &mut Vec<Vec<FusionTreeLeg>>,
    ) {
        if remaining == 0 {
            out.push(current.to_vec());
            return;
        }
        let index = remaining - 1;
        for &sector in legs[index].sectors() {
            current[index] = FusionTreeLeg::new(sector, legs[index].is_dual());
            visit(legs, remaining - 1, current, out);
        }
    }

    let mut out = Vec::new();
    let mut current = vec![FusionTreeLeg::new(SectorId::new(0), false); space.len()];
    visit(space.legs(), space.len(), &mut current, &mut out);
    out
}

pub(super) fn reference_select<R: FusionRule>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    codomain_axes: &[usize],
    domain_axes: &[usize],
) -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new(
            codomain_axes
                .iter()
                .map(|&axis| homspace.external_axis_leg(rule, axis)),
        ),
        FusionProductSpace::new(
            domain_axes
                .iter()
                .map(|&axis| homspace.external_axis_leg(rule, axis).dual(rule)),
        ),
    )
}

fn reference_tensorcontract_homspace<R: FusionRule>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    nout: usize,
) -> FusionTreeHomSpace {
    let lhs_open = (0..lhs.rank())
        .filter(|axis| !lhs_axes.contains(axis))
        .collect::<Vec<_>>();
    let rhs_open = (0..rhs.rank())
        .filter(|axis| !rhs_axes.contains(axis))
        .collect::<Vec<_>>();
    let lhs = reference_select(rule, lhs, &lhs_open, lhs_axes);
    let rhs = reference_select(rule, rhs, rhs_axes, &rhs_open);
    let composed = FusionTreeHomSpace::compose(rule, &lhs, &rhs).unwrap();
    reference_select(rule, &composed, &output_axes[..nout], &output_axes[nout..])
}

pub(super) fn assert_direct_contract_matches_reference<R: CheckedFusionAlgebra>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    nout: usize,
) {
    let expected =
        reference_tensorcontract_homspace(rule, lhs, rhs, lhs_axes, rhs_axes, output_axes, nout);
    let actual = FusionTreeHomSpace::tensorcontract_homspace(
        rule,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_axes,
        nout,
    )
    .unwrap();
    assert_eq!(actual, expected);
    let checked = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
        rule,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_axes,
        nout,
    )
    .unwrap();
    assert_eq!(checked, actual);

    // What: the destination check proves equality from the legs alone,
    // for both algebra contracts, and a different destination is
    // rejected with the materializing comparison's answer.
    let recording = crate::probe::record();
    let matches = |expected: &FusionTreeHomSpace| {
        let unchecked = FusionTreeHomSpace::tensorcontract_homspace_matches(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            nout,
            expected,
        )
        .unwrap();
        let checked = FusionTreeHomSpace::try_tensorcontract_homspace_matches_checked(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            nout,
            expected,
        )
        .unwrap();
        assert_eq!(unchecked, checked);
        unchecked
    };
    assert!(matches(&actual));
    assert_eq!(
        recording.get(crate::probe::Event::DescriptorMaterialization),
        0
    );
    let flipped_leg = |leg: &SectorLeg| SectorLeg::new(leg.iter(), !leg.is_dual());
    let mut codomain = actual.codomain().legs().to_vec();
    let mut domain = actual.domain().legs().to_vec();
    if let Some(leg) = codomain.first_mut() {
        *leg = flipped_leg(leg);
    } else if let Some(leg) = domain.first_mut() {
        *leg = flipped_leg(leg);
    }
    let other = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain),
        FusionProductSpace::new(domain),
    );
    if other != actual {
        assert!(!matches(&other));
    }
}

pub(super) fn reference_leg_degeneracy_structure<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) -> Arc<BlockStructure>
where
    R: MultiplicityFreeFusionRule,
{
    let keys = homspace.fusion_tree_keys(rule);
    let blocks = keys
        .iter()
        .map(|key| {
            (
                key.clone(),
                homspace.degeneracy_shape_for_key(key).unwrap().to_vec(),
            )
        })
        .collect();
    BlockStructure::coupled_sector_matrix_with_keys(
        rule,
        homspace.codomain().len(),
        homspace.rank(),
        blocks,
    )
    .unwrap()
    .into_shared()
}

pub(super) fn singleton_rank_hom(sector: SectorId, rank: usize) -> FusionTreeHomSpace {
    let side =
        |invert_dual| {
            FusionProductSpace::new((0..rank).map(|axis| {
                SectorLeg::new([(sector, axis % 3 + 1)], (axis % 2 == 0) ^ invert_dual)
            }))
        };
    FusionTreeHomSpace::new(side(false), side(true))
}

pub(super) fn compact_operator_cohort_fixture<R>(
    rule: &R,
    external: SectorId,
    coupled: SectorId,
) -> Vec<FusionTreePairKey>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
{
    let codomain: [SectorLeg; 8] = std::array::from_fn(|_| SectorLeg::new([(external, 1)], false));
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain),
        FusionProductSpace::new([SectorLeg::new([(coupled, 1)], false)]),
    );
    let keys = hom.fusion_tree_keys(rule).to_vec();
    assert!(
        keys.len() >= 16,
        "fixture must expose every requested source cohort"
    );
    keys
}

pub(super) trait TransposeOracleScalar {
    fn oracle_distance(&self, other: &Self) -> f64;
    fn oracle_magnitude(&self) -> f64;
}

impl TransposeOracleScalar for f64 {
    fn oracle_distance(&self, other: &Self) -> f64 {
        (self - other).abs()
    }

    fn oracle_magnitude(&self) -> f64 {
        self.abs()
    }
}

impl TransposeOracleScalar for Complex64 {
    fn oracle_distance(&self, other: &Self) -> f64 {
        (self - other).norm()
    }

    fn oracle_magnitude(&self) -> f64 {
        self.norm()
    }
}

pub(super) fn assert_compact_transpose_matches_full_key_oracle<R>(
    rule: &R,
    sources: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    expect_compact_boundary: bool,
) where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + std::fmt::Debug
        + TransposeOracleScalar,
{
    let recording = crate::probe::record();
    let compact = multiplicity_free_transpose_tree_pair_block(
        rule,
        sources,
        codomain_permutation,
        domain_permutation,
    )
    .unwrap();
    let scatters = recording.get(crate::probe::Event::CompactBlockScatter);
    let dimensions = (
        recording.get(crate::probe::Event::CompactDestinationRows),
        recording.get(crate::probe::Event::CompactSourceColumns),
        recording.get(crate::probe::Event::CompactCoefficientSlots),
        recording.get(crate::probe::Event::CompactCoefficientBytes),
    );
    let group = validate_tree_pair_block_group_for_rule(rule, sources)
        .unwrap()
        .expect("oracle cohort is nonempty");
    let prepared = PreparedTreePairOperation::prepare_transpose(
        group.codomain_rank,
        group.domain_rank,
        codomain_permutation,
        domain_permutation,
    )
    .unwrap();
    let ordered =
        multiplicity_free_transpose_tree_pair_block_ordered_validated(group, &prepared).unwrap();
    let full_key = multiplicity_free_transpose_tree_pair_block_full_key_oracle(
        rule,
        sources,
        codomain_permutation,
        domain_permutation,
    )
    .unwrap();

    let mut expected_destinations = Vec::new();
    for source_rows in &compact {
        for (destination, _) in source_rows {
            if !expected_destinations.contains(destination) {
                expected_destinations.push(destination.clone());
            }
        }
    }
    assert_eq!(ordered.destinations(), expected_destinations);
    assert_eq!(ordered.source_count(), sources.len());

    let mut ordered_coefficients =
        vec![None; ordered.destinations().len().saturating_mul(sources.len())];
    match ordered.storage() {
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        } => {
            assert_eq!(destination_rows.len(), sources.len());
            assert_eq!(coefficients.len(), sources.len());
            for (source, (&destination_row, coefficient)) in
                destination_rows.iter().zip(coefficients).enumerate()
            {
                ordered_coefficients[destination_row * sources.len() + source] =
                    Some(coefficient.clone());
            }
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients) => {
            assert_eq!(coefficients.len(), ordered_coefficients.len());
            ordered_coefficients.clone_from_slice(coefficients);
        }
    }
    for destination_row in 0..ordered.destinations().len() {
        assert!(
            ordered_coefficients
                [destination_row * sources.len()..(destination_row + 1) * sources.len()]
                .iter()
                .any(Option::is_some),
            "ordered maps omit structurally empty destination rows"
        );
    }
    for (source, source_rows) in compact.iter().enumerate() {
        for (destination_row, destination) in ordered.destinations().iter().enumerate() {
            let expected = source_rows
                .iter()
                .find(|(candidate, _)| candidate == destination)
                .map(|(_, coefficient)| coefficient);
            let actual = ordered_coefficients[destination_row * sources.len() + source].as_ref();
            assert_eq!(actual.is_some(), expected.is_some());
            if let (Some(actual), Some(expected)) = (actual, expected) {
                assert!(
                    actual.oracle_distance(expected)
                        <= 1.0e-12 * (1.0 + expected.oracle_magnitude()),
                    "ordered coefficient mismatch {expected:?} vs {actual:?}"
                );
            }
        }
    }

    assert_eq!(compact.len(), full_key.len());
    for (compact_rows, full_key_rows) in compact.iter().zip(&full_key) {
        // What: compact execution reaches every full-key destination label of
        // each source with the same coefficient. Row order within a source is
        // unspecified, so rows are matched by key.
        assert_eq!(compact_rows.len(), full_key_rows.len());
        for (key, expected) in full_key_rows {
            let actual = compact_rows
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, coefficient)| coefficient)
                .unwrap_or_else(|| panic!("compact rows omit destination {key:?}"));
            assert!(
                actual.oracle_distance(expected) <= 1.0e-12 * (1.0 + expected.oracle_magnitude()),
                "coefficient mismatch {expected:?} vs {actual:?}"
            );
        }
    }

    if expect_compact_boundary {
        // What: a nonidentity compact transform scatters exactly one block.
        assert_eq!(scatters, 1);
        let (destination_rows, source_columns, coefficient_slots, coefficient_bytes) = dimensions;
        let destinations = full_key
            .iter()
            .flatten()
            .map(|(key, _)| key)
            .collect::<std::collections::BTreeSet<_>>();
        // What: the dense coefficient matrix is exactly the canonical
        // reachable block basis by the caller's source columns.
        assert_eq!(destination_rows, destinations.len());
        assert_eq!(source_columns, sources.len());
        assert_eq!(coefficient_slots, destination_rows * source_columns);
        assert_eq!(
            coefficient_bytes,
            coefficient_slots * std::mem::size_of::<Option<R::Scalar>>()
        );
    } else {
        assert_eq!(scatters, 0);
    }
}

// --- Outer-multiplicity (Generic fusion) wiring rule -----------------
//
// `ToyOmRule` is purely synthetic (following `AsymmetricAnyonicRule`
// above as a template): sector 1 ("a") fuses with itself to sector 3
// ("c") via two independent channels, i.e. N(a,a,c) = 2. It exists only
// to exercise the `GenericFusionSymbols` wiring and the provider-owned
// `FusionStyleKind::Generic` gate. Pentagon/hexagon are NOT required to
// hold: this is wiring validation only, not a physical anyon model. The
// recoupling engine (recouple wrapper, `UnsupportedFusionStyle` guards)
// does not consume this rule; `UnitaryToyOmRule` below covers recoupling.
#[derive(Clone, Copy, Debug)]
pub(super) struct ToyOmRule;

impl ToyOmRule {
    pub(super) const VACUUM: usize = 0;
    pub(super) const A: usize = 1;
    pub(super) const C: usize = 3;
}

impl symtn_sectors::testing::Provider for ToyOmRule {}

impl FusionRule for ToyOmRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(Self::VACUUM)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
            (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
            (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
            _ => smallvec![SectorId::new(Self::VACUUM)],
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        // The one artificial outer multiplicity this toy rule carries:
        // a (x) a -> c has two independent fusion channels (N=2). Every
        // other triple falls back to the multiplicity-free default (0
        // or 1, from whether `coupled` is a fusion channel of
        // `left (x) right`) — this is the override the design doc calls
        // for instead of trying to encode multiplicity through repeated
        // `fusion_channels` entries.
        if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for ToyOmRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let n_mu = self.nsymbol(a, b, e);
        let n_nu = self.nsymbol(e, c, d);
        let n_kappa = self.nsymbol(b, c, f);
        let n_lambda = self.nsymbol(a, f, d);
        let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
        if ids == (Self::A, Self::A, Self::VACUUM, Self::C, Self::C, Self::A) {
            // mu and lambda both range over the N(a,a,c)=2 channel here
            // (nu = N(c,0,c) = 1, kappa = N(a,0,a) = 1 are trivial), so
            // this F-block is genuinely a 2x2 matrix. Filled with a
            // pi/4 rotation — an actual orthogonal matrix, not just
            // shape-correct filler — so that F's (mu, lambda) block is
            // orthogonal and a braid * inverse == identity
            // self-consistency check has something real to check.
            let s = std::f64::consts::FRAC_1_SQRT_2;
            GenericFArray::new(vec![s, -s, s, s], (n_mu, n_nu, n_kappa, n_lambda))
        } else {
            let total = n_mu * n_nu * n_kappa * n_lambda;
            GenericFArray::new(vec![1.0; total], (n_mu, n_nu, n_kappa, n_lambda))
        }
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let rows = self.nsymbol(a, b, c);
        let cols = self.nsymbol(b, a, c);
        if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
            GenericRMatrix::new(vec![1.0, 0.0, 0.0, 1.0], rows, cols)
        } else {
            GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
        }
    }
}

// --- Generic-fusion Artin braid (braid × inverse == identity) ----------
//
// `UnitaryToyOmRule` is a separate rule from `ToyOmRule`, with the same fusion
// structure (a⊗a→c has N=2, everything else N≤1, mixing an OM vertex with
// multiplicity-1 vertices), but with UNITARY F/R blocks so that TensorKit's
// inverse-braid identity actually holds:
//
//   * R(a,a,c) is a genuine 2×2 rotation (unitary), so the braid mixes the
//     two outer-multiplicity channels non-trivially — the round-trip test
//     is real, not vacuous.
//   * F(a,a,a,a,c,c) — the only F block the rank-3 braid touches — is the
//     2×2 IDENTITY. This is deliberate: the toy rule is not
//     hexagon-consistent, and for the index>1 braid the round-trip
//     coefficient works out to `Rθᵀ·M·Rθ·M` (M = that F block); with
//     nontrivial rotations Rθ, M that equals I *iff* M = I. A hexagon-
//     consistent model would let a nontrivial F cancel, but this rule only needs
//     the wiring + inverse-adjoint handling exercised, which the nontrivial
//     R already does. (Since both braided legs are the sector `a`,
//     R(a,b,c)=R(b,a,c), so no hexagon relation between the two R's is
//     needed for index==0 either.)
//   * F(a,a,0,c,c,a) is a genuine π/4 rotation, kept only so the unitarity
//     assertion test has a non-identity unitary F block to check. It is not
//     on any braid path here.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnitaryToyOmRule;

impl UnitaryToyOmRule {
    pub(super) const VACUUM: usize = 0;
    pub(super) const A: usize = 1;
    pub(super) const C: usize = 3;
    // R(a,a,c) rotation angle. Any nonzero angle whose sin/cos are both
    // nonzero makes the braid genuinely spread over both OM channels.
    const R_THETA: f64 = std::f64::consts::PI / 5.0;
}

impl symtn_sectors::testing::Provider for UnitaryToyOmRule {}

impl FusionRule for UnitaryToyOmRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        // Bosonic => has_braiding() (needed to pass the NoBraiding guard);
        // the actual crossings are governed by the (non-symmetric) R blocks.
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(Self::VACUUM)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
            (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
            (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
            _ => smallvec![SectorId::new(Self::VACUUM)],
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for UnitaryToyOmRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let n_mu = self.nsymbol(a, b, e);
        let n_nu = self.nsymbol(e, c, d);
        let n_kappa = self.nsymbol(b, c, f);
        let n_lambda = self.nsymbol(a, f, d);
        let shape = (n_mu, n_nu, n_kappa, n_lambda);
        let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
        if ids == (Self::A, Self::A, Self::VACUUM, Self::C, Self::C, Self::A) {
            // (mu, lambda) 2×2 block (nu = kappa = 0): a real π/4 rotation.
            // Kept only for the unitarity assertion — not on a braid path.
            let s = std::f64::consts::FRAC_1_SQRT_2;
            GenericFArray::new(vec![s, -s, s, s], shape)
        } else if ids == (Self::A, Self::A, Self::A, Self::A, Self::C, Self::C) {
            // The one F block the rank-3 braid reads: shape (2,1,2,1), a
            // 2×2 in (mu, kappa). IDENTITY (see the module comment for why
            // it must be I, not a rotation). Row-major over
            // (mu, nu, kappa, lambda): flat idx = mu*2 + kappa.
            GenericFArray::new(vec![1.0, 0.0, 0.0, 1.0], shape)
        } else {
            // Defensive default: identity on the leading diagonal of the
            // flattened ((mu,nu) × (kappa,lambda)) matrix. For 1×1 blocks
            // this is [1.0] (unitary); for any larger square block it is a
            // genuine unitary. No such block is reached by these tests, but
            // an all-ones fill would be silently non-unitary if one were.
            let rows = n_mu * n_nu;
            let cols = n_kappa * n_lambda;
            let mut data = vec![0.0; rows * cols];
            for r in 0..rows {
                if r < cols {
                    data[r * cols + r] = 1.0;
                }
            }
            GenericFArray::new(data, shape)
        }
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let rows = self.nsymbol(a, b, c);
        let cols = self.nsymbol(b, a, c);
        if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
            // 2×2 rotation Rθ = [[cosθ, -sinθ], [sinθ, cosθ]] (unitary),
            // row-major. This is the block that makes the braid non-trivial.
            let (s, c_) = Self::R_THETA.sin_cos();
            GenericRMatrix::new(vec![c_, -s, s, c_], rows, cols)
        } else {
            // Every other block is 1×1 with modulus-1 entry (unitary).
            GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
        }
    }
}

impl CheckedGenericFusion for UnitaryToyOmRule {
    type Error = std::convert::Infallible;
    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(self)
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(self)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(sector)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        Ok(self.fusion_channels(a, b))
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.fusion_channels(a, b))
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        Ok(self.nsymbol(a, b, c))
    }
}

impl CheckedGenericRigidSymbols for UnitaryToyOmRule {
    type Scalar = f64;
    fn try_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn try_inv_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn try_frobenius_schur_phase_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        Ok(1.0)
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
        Ok(self.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        Ok(self.r_symbol_generic(a, b, c))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ArtinSpyError {
    F,
    R,
}

impl std::fmt::Display for ArtinSpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ArtinSpyError {}

pub(super) struct ArtinSpy {
    pub(super) inner: UnitaryToyOmRule,
    pub(super) f_calls: std::cell::Cell<usize>,
    pub(super) r_calls: std::cell::Cell<usize>,
    pub(super) rigid_calls: std::cell::Cell<usize>,
    pub(super) fail_f: Option<usize>,
    pub(super) fail_r: Option<usize>,
    pub(super) bad_f: bool,
    pub(super) bad_r: bool,
}

impl ArtinSpy {
    pub(super) fn new() -> Self {
        Self {
            inner: UnitaryToyOmRule,
            f_calls: std::cell::Cell::new(0),
            r_calls: std::cell::Cell::new(0),
            rigid_calls: std::cell::Cell::new(0),
            fail_f: None,
            fail_r: None,
            bad_f: false,
            bad_r: false,
        }
    }
    fn trip(
        counter: &std::cell::Cell<usize>,
        fail: Option<usize>,
        error: ArtinSpyError,
    ) -> Result<(), ArtinSpyError> {
        let n = counter.get() + 1;
        counter.set(n);
        if fail == Some(n) {
            Err(error)
        } else {
            Ok(())
        }
    }
}

impl symtn_sectors::testing::Provider for ArtinSpy {}

impl CheckedGenericFusion for ArtinSpy {
    type Error = ArtinSpyError;
    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(&self.inner)
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(&self.inner)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(&self.inner)
    }
    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(&self.inner)
    }
    fn try_dual(&self, s: SectorId) -> Result<SectorId, Self::Error> {
        Ok(s)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        Ok(self.inner.fusion_channels(a, b))
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.inner.fusion_channels(a, b))
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        Ok(self.inner.nsymbol(a, b, c))
    }
}

impl CheckedGenericRigidSymbols for ArtinSpy {
    type Scalar = f64;
    fn try_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.rigid_calls.set(self.rigid_calls.get() + 1);
        Ok(1.0)
    }
    fn try_inv_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.rigid_calls.set(self.rigid_calls.get() + 1);
        Ok(1.0)
    }
    fn try_frobenius_schur_phase_scalar(&self, _: SectorId) -> Result<f64, Self::Error> {
        self.rigid_calls.set(self.rigid_calls.get() + 1);
        Ok(1.0)
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
        Self::trip(&self.f_calls, self.fail_f, ArtinSpyError::F)?;
        let x = self.inner.f_symbol_generic(a, b, c, d, e, f);
        if self.bad_f {
            Ok(GenericFArray::new(
                x.data().to_vec(),
                (1, 1, x.data().len(), 1),
            ))
        } else {
            Ok(x)
        }
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        Self::trip(&self.r_calls, self.fail_r, ArtinSpyError::R)?;
        let x = self.inner.r_symbol_generic(a, b, c);
        if self.bad_r {
            Ok(GenericRMatrix::new(x.data().to_vec(), 1, x.data().len()))
        } else {
            Ok(x)
        }
    }
}

// Rank-2 tree [a, a] -> c with a single OM vertex label `vertex`.
pub(super) fn unitary_rank2_tree(vertex: usize) -> FusionTreeKey {
    let a = SectorId::new(UnitaryToyOmRule::A);
    let c = SectorId::new(UnitaryToyOmRule::C);
    FusionTreeKey::new(
        [a, a],
        c,
        [false, false],
        [],
        [MultiplicityIndex::new(vertex).expect("test multiplicity label is one-based")],
    )
}

// Rank-3 tree [a, a, a] -> a: fuse a⊗a->c (OM vertex `vertex1`, N=2), then
// c⊗a->a (vertex2, forced label 1). Innerline [c]. Mixes an OM vertex with
// a multiplicity-1 vertex.
pub(super) fn unitary_rank3_tree(vertex1: usize) -> FusionTreeKey {
    let a = SectorId::new(UnitaryToyOmRule::A);
    let c = SectorId::new(UnitaryToyOmRule::C);
    FusionTreeKey::new(
        [a, a, a],
        a,
        [false, false, false],
        [c],
        [
            MultiplicityIndex::new(vertex1).expect("test multiplicity label is one-based"),
            MultiplicityIndex::ONE,
        ],
    )
}

// ======================== Generic bend / repartition ======================
//
// Oracle & gate rule: the REAL A4Irrep(3) outer-multiplicity sub-block.
// A4Irrep(3) is self-dual (dual(3)=3), 3⊗3 = {0,1,2,3} with N(3,3,3)=2 (the
// only outer multiplicity) AND 3⊗3 ∋ vacuum — so it is genuinely rigid and
// can bend. (Contrast the braid-only `UnitaryToyOmRule`: its sector `a` has
// NO proper dual — a⊗a ∌ vacuum — so every B-symbol there is degenerate and
// bending is undefined; extending it would change its fusion structure, so a
// separate rigid rule is used.)
//
// Constants computed out-of-band from TensorKit v0.16.2 + TensorKitSectors
// v0.3.6 (git-tree-sha1 334a0ed5a0a0088a2b6fe7a39f78dda928038d85), by
// TensorKit's OWN Bsymbol / Asymbol / bendright, independent of this port.
//
// DISCRIMINATING POWER (honest note): for A4Irrep(3), Bsymbol(3,3,3) AND
// Asymbol(3,3,3) are BOTH the 2×2 identity (asserted below) — the A4 3-irrep
// bend has no off-diagonal channel mixing. So this oracle does NOT catch a
// μ↔ν transpose in the B-matrix indexing. It DOES pin the tree surgery, the
// √dim(c)·(1/√dim(a)) coefficient (√3 vs 1 across the varied innerline in the
// rank-3 table — a sqrt/invsqrt swap changes these), μ = last-codomain-vertex
// selection, ν → domain vertex-label storage, and the F→B derivation. A
// μ↔ν-discriminating BEND oracle needs a non-diagonal Bsymbol; the SU(3)
// rule below supplies it.
#[derive(Clone, Copy, Debug)]
pub(super) struct A4BendRule;

impl symtn_sectors::testing::Provider for A4BendRule {}

impl FusionRule for A4BendRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        // Unused: bend/repartition are planar (no braiding). Bosonic keeps
        // the rule well-formed.
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        // A4: 0 and 3 self-dual; the two nontrivial 1-dim irreps 1,2 are
        // each other's dual (verified in Julia). Only dual(3)=3 is exercised
        // by the bend, but the full map is correct.
        match sector.id() {
            1 => SectorId::new(2),
            2 => SectorId::new(1),
            _ => sector,
        }
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (3, 3) => smallvec![
                SectorId::new(0),
                SectorId::new(1),
                SectorId::new(2),
                SectorId::new(3)
            ],
            (3, _) | (_, 3) => smallvec![SectorId::new(3)],
            // {1,2}⊗{1,2}: never touched by the bend trees; defensive stub.
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for A4BendRule {
    type Scalar = f64;
    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let inv = 1.0 / 3.0_f64.sqrt(); // 1/√dim(3)
        let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
        match ids {
            // F(3,3,3,3,3,0): the block Bsymbol(3,3,3) reshapes, shape
            // (N(3,3,3),N(3,3,3),1,1)=(2,2,1,1). (μ,ν)=(1/√3)·I, row-major.
            (3, 3, 3, 3, 3, 0) => GenericFArray::new(vec![inv, 0.0, 0.0, inv], (2, 2, 1, 1)),
            // F(x,3,3,x,3,0), x∈{0,1,2}: Bsymbol(x,3,3), shape (1,1,1,1)=[1].
            (0, 3, 3, 0, 3, 0) | (1, 3, 3, 1, 3, 0) | (2, 3, 3, 2, 3, 0) => {
                GenericFArray::new(vec![1.0], (1, 1, 1, 1))
            }
            // F(3,3,3,3,x,0), x∈{0,1,2}: Bsymbol(3,3,x) (the return bend when
            // the intermediate coupled sector is x), (1,1,1,1)=[1/3].
            (3, 3, 3, 3, 0, 0) | (3, 3, 3, 3, 1, 0) | (3, 3, 3, 3, 2, 0) => {
                GenericFArray::new(vec![1.0 / 3.0], (1, 1, 1, 1))
            }
            // F(3,3,3,3,0,3): the block Asymbol(3,3,3) reshapes, shape
            // (1,1,N(3,3,3),N(3,3,3))=(1,1,2,2). (κ,λ)=(1/√3)·I, row-major.
            (3, 3, 3, 3, 0, 3) => GenericFArray::new(vec![inv, 0.0, 0.0, inv], (1, 1, 2, 2)),
            _ => panic!("A4BendRule: unmodelled F{ids:?}"),
        }
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        // Unused: planar duality moves never braid. Trivial 1×1 stub.
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

impl GenericRigidSymbols for A4BendRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 3 {
            3.0
        } else {
            1.0
        }
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 3 {
            3.0_f64.sqrt()
        } else {
            1.0
        }
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 3 {
            1.0 / 3.0_f64.sqrt()
        } else {
            1.0
        }
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        // All A4 sectors reached here have FS phase +1 (verified in Julia).
        1.0
    }
}

impl CheckedGenericFusion for A4BendRule {
    type Error = std::convert::Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(self)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(self)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(self)
    }

    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(self)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(FusionRule::dual(self, sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(FusionRule::fusion_channels(self, left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(FusionRule::fusion_channels(self, left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(FusionRule::nsymbol(self, left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for A4BendRule {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(GenericRigidSymbols::dim_scalar(self, sector))
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(GenericRigidSymbols::sqrt_dim_scalar(self, sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(GenericRigidSymbols::inv_sqrt_dim_scalar(self, sector))
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Ok(GenericRigidSymbols::frobenius_schur_phase_scalar(
            self, sector,
        ))
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Ok(GenericFusionSymbols::f_symbol_generic(
            self, a, b, c, d, e, f,
        ))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(GenericFusionSymbols::r_symbol_generic(self, a, b, c))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RigidSpyError {
    Dual,
    N,
    F,
    A,
    B,
    Sqrt,
    InvSqrt,
    Fs,
}

impl std::fmt::Display for RigidSpyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RigidSpyError {}

pub(super) struct CheckedA4Spy {
    pub(super) dual_calls: std::cell::Cell<usize>,
    pub(super) n_calls: std::cell::Cell<usize>,
    pub(super) f_calls: std::cell::Cell<usize>,
    pub(super) a_calls: std::cell::Cell<usize>,
    pub(super) b_calls: std::cell::Cell<usize>,
    pub(super) sqrt_calls: std::cell::Cell<usize>,
    pub(super) inv_sqrt_calls: std::cell::Cell<usize>,
    pub(super) fs_calls: std::cell::Cell<usize>,
    pub(super) fail_dual: Option<usize>,
    pub(super) fail_n: Option<usize>,
    pub(super) fail_f: Option<usize>,
    pub(super) fail_a: Option<usize>,
    pub(super) fail_b: Option<usize>,
    pub(super) fail_sqrt: Option<usize>,
    pub(super) fail_inv_sqrt: Option<usize>,
    pub(super) fail_fs: Option<usize>,
    pub(super) bad_b: bool,
    pub(super) bad_a: bool,
    pub(super) non_diagonal_b: bool,
}

impl CheckedA4Spy {
    pub(super) fn new() -> Self {
        Self {
            dual_calls: std::cell::Cell::new(0),
            n_calls: std::cell::Cell::new(0),
            f_calls: std::cell::Cell::new(0),
            a_calls: std::cell::Cell::new(0),
            b_calls: std::cell::Cell::new(0),
            sqrt_calls: std::cell::Cell::new(0),
            inv_sqrt_calls: std::cell::Cell::new(0),
            fs_calls: std::cell::Cell::new(0),
            fail_dual: None,
            fail_n: None,
            fail_f: None,
            fail_a: None,
            fail_b: None,
            fail_sqrt: None,
            fail_inv_sqrt: None,
            fail_fs: None,
            bad_b: false,
            bad_a: false,
            non_diagonal_b: false,
        }
    }

    fn trip(
        counter: &std::cell::Cell<usize>,
        fail: Option<usize>,
        error: RigidSpyError,
    ) -> Result<(), RigidSpyError> {
        let call = counter.get() + 1;
        counter.set(call);
        if fail == Some(call) {
            Err(error)
        } else {
            Ok(())
        }
    }
}

impl symtn_sectors::testing::Provider for CheckedA4Spy {}

impl CheckedGenericFusion for CheckedA4Spy {
    type Error = RigidSpyError;

    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(&A4BendRule)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(&A4BendRule)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Self::trip(&self.dual_calls, self.fail_dual, RigidSpyError::Dual)?;
        Ok(FusionRule::dual(&A4BendRule, sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(FusionRule::fusion_channels(&A4BendRule, left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Self::trip(&self.n_calls, self.fail_n, RigidSpyError::N)?;
        Ok(FusionRule::nsymbol(&A4BendRule, left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for CheckedA4Spy {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Ok(GenericRigidSymbols::dim_scalar(&A4BendRule, sector))
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Self::trip(&self.sqrt_calls, self.fail_sqrt, RigidSpyError::Sqrt)?;
        Ok(GenericRigidSymbols::sqrt_dim_scalar(&A4BendRule, sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        Self::trip(
            &self.inv_sqrt_calls,
            self.fail_inv_sqrt,
            RigidSpyError::InvSqrt,
        )?;
        Ok(GenericRigidSymbols::inv_sqrt_dim_scalar(
            &A4BendRule,
            sector,
        ))
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        Self::trip(&self.fs_calls, self.fail_fs, RigidSpyError::Fs)?;
        Ok(GenericRigidSymbols::frobenius_schur_phase_scalar(
            &A4BendRule,
            sector,
        ))
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<Self::Scalar>, Self::Error> {
        Self::trip(&self.f_calls, self.fail_f, RigidSpyError::F)?;
        Ok(GenericFusionSymbols::f_symbol_generic(
            &A4BendRule,
            a,
            b,
            c,
            d,
            e,
            f,
        ))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Ok(GenericFusionSymbols::r_symbol_generic(&A4BendRule, a, b, c))
    }

    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Self::trip(&self.a_calls, self.fail_a, RigidSpyError::A)?;
        let symbol =
            CheckedGenericRigidSymbols::try_a_symbol_generic(&A4BendRule, a, b, c).unwrap();
        if self.bad_a {
            Ok(GenericRMatrix::new(symbol.data().to_vec(), 1, 4))
        } else {
            Ok(symbol)
        }
    }

    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        Self::trip(&self.b_calls, self.fail_b, RigidSpyError::B)?;
        let symbol =
            CheckedGenericRigidSymbols::try_b_symbol_generic(&A4BendRule, a, b, c).unwrap();
        if self.bad_b {
            Ok(GenericRMatrix::new(symbol.data().to_vec(), 1, 4))
        } else if self.non_diagonal_b {
            Ok(GenericRMatrix::new(vec![0.3, 0.7, 0.9, 0.1], 2, 2))
        } else {
            Ok(symbol)
        }
    }
}

pub(super) fn a4_three() -> SectorId {
    SectorId::new(3)
}

// cod [3,3]->3 (vertex μ), dom [3]->3.
pub(super) fn a4_pair_rank2(mu: usize) -> FusionTreePairKey {
    let t = a4_three();
    let cod = FusionTreeKey::new(
        [t, t],
        t,
        [false, false],
        [],
        [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
    );
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    FusionTreePairKey::pair(cod, dom)
}

pub(super) fn a4_dual_pair_rank2(mu: usize) -> FusionTreePairKey {
    let t = a4_three();
    let cod = FusionTreeKey::new(
        [t, t],
        t,
        [false, true],
        [],
        [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
    );
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    FusionTreePairKey::pair(cod, dom)
}

pub(super) fn assert_rigid_provider_error(
    error: CheckedGenericSymbolError<RigidSpyError>,
    expected: RigidSpyError,
) {
    assert!(std::error::Error::source(&error).is_some());
    assert!(matches!(
        error,
        CheckedGenericSymbolError::Provider(actual) if actual == expected
    ));
}

// ============ REFUTE: μ↔ν / κ↔λ transpose discriminator ============
//
// The A4 oracle CANNOT catch a B-matrix (or A-matrix) index transpose: for
// A4Irrep(3) both Bsymbol and Asymbol are the 2×2 IDENTITY, which is its own
// transpose. This synthetic rule closes that gap with a *deliberately
// non-symmetric* B and A block (B[0,1]≠B[1,0], A[0,1]≠A[1,0]).
//
// The reshape-collapse is transpose-free in Julia (verified out-of-band:
// `reshape(F,(N1,N2))[μ,ν]==F[μ,ν,1,1]` for trailing singletons and
// `[κ,λ]==F[1,1,κ,λ]` for leading singletons), so the CORRECT reading is
//   B[μ,ν] = √dim(a)·√dim(b)·invsqrtdim(c) · F(a,b,dual(b),a,c,unit)[μ,ν,0,0]
//   A[κ,λ] = √dim(a)·√dim(b)·invsqrtdim(c) · conj(κ_a·F(dual(a),a,b,b,unit,c)[0,0,κ,λ]).
// A μ↔ν (or κ↔λ) swap in the impl would read F[ν,μ,0,0] / F[0,0,λ,κ] and
// produce the TRANSPOSE — which THIS test detects and the A4 oracle does not.
#[derive(Clone, Copy, Debug)]
pub(super) struct TransposeProbeRule;

// Sector 1 is self-dual with dim 4 (so √dim=2, exercising the coeff factor);
// 1⊗1 = {0 (rigidity), 1 (with N=2)}. Only the (1,1,1) block is non-trivial.
impl symtn_sectors::testing::Provider for TransposeProbeRule {}

impl FusionRule for TransposeProbeRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        sector // 0 and 1 both self-dual
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2 // the single outer multiplicity
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

// The raw F data, in TK [μ,ν,κ,λ] semantic order, row-major — the SAME bytes
// the from-scratch oracle below reads directly.
const TP_FB: [f64; 4] = [0.3, 0.7, 0.9, 0.1];

// F(1,1,1,1,1,0)[μ,ν] block, non-symmetric
pub(super) const TP_FA: [f64; 4] = [0.2, 0.5, 0.6, 0.4];

// F(1,1,1,1,0,1)[κ,λ] block, non-symmetric
impl GenericFusionSymbols for TransposeProbeRule {
    type Scalar = f64;
    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        match (a.id(), b.id(), c.id(), d.id(), e.id(), f.id()) {
            // B block: shape (N(1,1,1),N(1,1,1),1,1) = (2,2,1,1).
            (1, 1, 1, 1, 1, 0) => GenericFArray::new(TP_FB.to_vec(), (2, 2, 1, 1)),
            // A block: shape (1,1,N(1,1,1),N(1,1,1)) = (1,1,2,2).
            (1, 1, 1, 1, 0, 1) => GenericFArray::new(TP_FA.to_vec(), (1, 1, 2, 2)),
            other => panic!("TransposeProbeRule: unmodelled F{other:?}"),
        }
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

impl GenericRigidSymbols for TransposeProbeRule {
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 1 {
            2.0
        } else {
            1.0
        } // dim(1)=4
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 1 {
            0.5
        } else {
            1.0
        }
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

// Independent from-scratch TK evaluation of the reshape formula — explicit
// index loops, NO call into the core A/B derivation.
pub(super) fn tp_expected_b() -> [[f64; 2]; 2] {
    let factor = 2.0 * 2.0 * 0.5; // √dim(1)·√dim(1)·invsqrtdim(1) = 2
    let mut b = [[0.0; 2]; 2];
    for mu in 0..2 {
        for nu in 0..2 {
            b[mu][nu] = factor * TP_FB[mu * 2 + nu]; // F[μ,ν,0,0]
        }
    }
    b
}

// ===================== Generic fold / multi_Fmove ========================
//
// ORACLE PROVENANCE. All numeric constants below are TensorKit's own values
// for `A4Irrep(3)`, extracted by running the mirrored source:
//   TensorKit.jl @ git cfaa073 (v0.17.0), TensorKitSectors v0.3.9
//   (Fsymbol values identical to v0.3.6, the version the `A4BendRule`
//   constants cite — the A4 category data is version-stable).
// The `multi_Fmove` tables come from `TensorKit.multi_Fmove(f)` and the
// `foldright` tables from `TensorKit.foldright(FusionTreeBlock)`, both called
// directly on A4 fusion trees. F-symbol arrays are transcribed ROW-MAJOR
// over the TK axis order (μ, ν, κ, λ) — Julia's `vec()` is column-major, so
// the transcription applies `permutedims(F,(4,3,2,1))` first. (The
// `A4BendRule` blocks are all transpose-symmetric, so they never exposed this;
// the non-trivial F(3,3,3,3,3,3) block below does.)

// Full A4Irrep(3) fusion rule with the COMPLETE F(3,3,3,3,e,f) table —
// `A4BendRule` only models the handful of bend/A-symbol tuples, which is
// insufficient for multi_Fmove/associator (they consult every (e,f)).
#[derive(Clone, Copy, Debug)]
pub(super) struct A4FoldRule;

impl symtn_sectors::testing::Provider for A4FoldRule {}

impl FusionRule for A4FoldRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        match sector.id() {
            1 => SectorId::new(2),
            2 => SectorId::new(1),
            _ => sector,
        }
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (3, 3) => smallvec![
                SectorId::new(0),
                SectorId::new(1),
                SectorId::new(2),
                SectorId::new(3)
            ],
            (3, _) | (_, 3) => smallvec![SectorId::new(3)],
            (1, 1) => smallvec![SectorId::new(2)],
            (2, 2) => smallvec![SectorId::new(1)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(0)],
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (3, 3, 3) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for A4FoldRule {
    type Scalar = f64;
    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let s = 1.0 / 3.0_f64.sqrt(); // 1/√3
        let m = -1.0 / (2.0 * 3.0_f64.sqrt()); // -1/(2√3)
        let hs = 3.0_f64.sqrt() / 2.0; // √3/2
        let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
        // The only non-trivial family: F(3,3,3,3,e,f). ROW-MAJOR (μ,ν,κ,λ).
        match ids {
            (3, 3, 3, 3, 0, 0)
            | (3, 3, 3, 3, 0, 1)
            | (3, 3, 3, 3, 0, 2)
            | (3, 3, 3, 3, 1, 0)
            | (3, 3, 3, 3, 1, 1)
            | (3, 3, 3, 3, 1, 2)
            | (3, 3, 3, 3, 2, 0)
            | (3, 3, 3, 3, 2, 1)
            | (3, 3, 3, 3, 2, 2) => GenericFArray::new(vec![1.0 / 3.0], (1, 1, 1, 1)),
            // A-symbol reshape F(3,3,b,b,0,3) for b∈{1,2}: TK gives [1] (1×1).
            (3, 3, 1, 1, 0, 3) | (3, 3, 2, 2, 0, 3) => GenericFArray::new(vec![1.0], (1, 1, 1, 1)),
            (3, 3, 3, 3, 0, 3) => GenericFArray::new(vec![s, 0.0, 0.0, s], (1, 1, 2, 2)),
            (3, 3, 3, 3, 1, 3) => GenericFArray::new(vec![m, -0.5, 0.5, m], (1, 1, 2, 2)),
            (3, 3, 3, 3, 2, 3) => GenericFArray::new(vec![m, 0.5, -0.5, m], (1, 1, 2, 2)),
            (3, 3, 3, 3, 3, 0) => GenericFArray::new(vec![s, 0.0, 0.0, s], (2, 2, 1, 1)),
            (3, 3, 3, 3, 3, 1) => GenericFArray::new(vec![m, 0.5, -0.5, m], (2, 2, 1, 1)),
            (3, 3, 3, 3, 3, 2) => GenericFArray::new(vec![m, -0.5, 0.5, m], (2, 2, 1, 1)),
            (3, 3, 3, 3, 3, 3) => GenericFArray::new(
                vec![
                    0.5, 0.0, 0.0, -0.5, 0.0, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.0, -0.5, 0.0,
                    0.0, 0.5,
                ],
                (2, 2, 2, 2),
            ),
            // The 9 non-trivial off-family blocks the CYCLE bends reach
            // (√3/2 = 0.8660254038). TK row-major (μ,ν,κ,λ) values.
            (1, 3, 3, 3, 3, 3) => GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (1, 2, 2, 1)),
            (2, 3, 3, 3, 3, 3) => GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (1, 2, 2, 1)),
            (3, 1, 3, 3, 3, 3) => GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (1, 2, 1, 2)),
            (3, 2, 3, 3, 3, 3) => GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (1, 2, 1, 2)),
            (3, 3, 1, 3, 3, 3) => GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (2, 1, 1, 2)),
            (3, 3, 2, 3, 3, 3) => GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (2, 1, 1, 2)),
            (3, 3, 3, 0, 3, 3) => GenericFArray::new(vec![1.0, 0.0, 0.0, 1.0], (2, 1, 2, 1)),
            (3, 3, 3, 1, 3, 3) => GenericFArray::new(vec![-0.5, hs, -hs, -0.5], (2, 1, 2, 1)),
            (3, 3, 3, 2, 3, 3) => GenericFArray::new(vec![-0.5, -hs, hs, -0.5], (2, 1, 2, 1)),
            // Everything else valid in A4 is a singleton block equal to [1]:
            // any F with a vacuum a/b/c leg, and the residual all-singleton
            // triples (e.g. F(3,1,3,0,3,3)). Shape-aware so a genuinely
            // unmodelled MULTI-dim block still panics instead of silently
            // returning a wrong scalar.
            _ => {
                let shape = (
                    self.nsymbol(a, b, e),
                    self.nsymbol(e, c, d),
                    self.nsymbol(b, c, f),
                    self.nsymbol(a, f, d),
                );
                if shape == (1, 1, 1, 1) {
                    GenericFArray::new(vec![1.0], shape)
                } else {
                    panic!("A4FoldRule: unmodelled non-singleton F{ids:?} shape={shape:?}");
                }
            }
        }
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

impl GenericRigidSymbols for A4FoldRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 3 {
            3.0
        } else {
            1.0
        }
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 3 {
            3.0_f64.sqrt()
        } else {
            1.0
        }
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector.id() == 3 {
            1.0 / 3.0_f64.sqrt()
        } else {
            1.0
        }
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

pub(super) fn a4f_rank3(inner: usize, v1: usize, v2: usize) -> FusionTreeKey {
    let t = SectorId::new(3);
    FusionTreeKey::new(
        [t, t, t],
        t,
        [false, false, false],
        [SectorId::new(inner)],
        [
            MultiplicityIndex::new(v1).expect("test multiplicity label is one-based"),
            MultiplicityIndex::new(v2).expect("test multiplicity label is one-based"),
        ],
    )
}

// ============================ Complex conj path ============================
//
// The complex path of bend and fold (`CategoricalScalar for Complex64`, the
// fold's `coeff₂.conj()`, and the A derivation's inner `conj`) is checked
// numerically here: a synthetic
// Complex64 Generic rule whose A-move / B-move are a genuinely complex 2×2
// UNITARY U (non-Hermitian, non-real). Self-dual sector 1, N(1,1,1)=2,
// dim=1 (all coeff factors = 1). U = (1/√2)[[1, i],[i, 1]].
//
// From the A derivation: A[κ,λ] = conj(κ_a · F(1,1,1,1,0,1)[0,0,κ,λ]) with
// κ_a=1, so setting the F block to conj(U) gives A = U. Likewise B = U from
// the F(1,1,1,1,1,0) block. A wrong conj (missing/extra) or a μ↔ν transpose
// flips the sign of the imaginary parts and fails both the direct check and
// the round-trip (which needs U U† = I).
#[derive(Clone, Copy, Debug)]
pub(super) struct ComplexUnitaryRule;

impl symtn_sectors::testing::Provider for ComplexUnitaryRule {}

impl FusionRule for ComplexUnitaryRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        sector // 0 and 1 self-dual
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

pub(super) fn cx(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// U = (1/√2)[[1, i],[i, 1]], row-major.
pub(super) fn cx_u() -> [Complex64; 4] {
    let r = 1.0 / 2.0_f64.sqrt();
    [cx(r, 0.0), cx(0.0, r), cx(0.0, r), cx(r, 0.0)]
}

impl GenericFusionSymbols for ComplexUnitaryRule {
    type Scalar = Complex64;
    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let u = cx_u();
        match (a.id(), b.id(), c.id(), d.id(), e.id(), f.id()) {
            // A block: A = conj(U) reshaped => set F = conj(U). shape (1,1,2,2).
            (1, 1, 1, 1, 0, 1) => GenericFArray::new(
                vec![u[0].conj(), u[1].conj(), u[2].conj(), u[3].conj()],
                (1, 1, 2, 2),
            ),
            // B block: B = U reshaped. shape (2,2,1,1).
            (1, 1, 1, 1, 1, 0) => GenericFArray::new(vec![u[0], u[1], u[2], u[3]], (2, 2, 1, 1)),
            (aa, bb, cc, _, _, _) if aa == 0 || bb == 0 || cc == 0 => {
                GenericFArray::new(vec![cx(1.0, 0.0)], (1, 1, 1, 1))
            }
            other => panic!("ComplexUnitaryRule: unmodelled F{other:?}"),
        }
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![cx(1.0, 0.0)], 1, 1)
    }
}

impl GenericRigidSymbols for ComplexUnitaryRule {
    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        cx(1.0, 0.0)
    }
    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        cx(1.0, 0.0)
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        cx(1.0, 0.0)
    }
}

// ================= Real non-diagonal SU(3) B-symbol ======================
//
// The A4 bend oracle cannot discriminate a μ↔ν B-matrix transpose:
// A4Irrep(3)'s Bsymbol is I₂ (its own transpose), so a *non-diagonal
// Bsymbol from a real category* (SU(3)) is used here. Extracted
// from SUNRepresentations.jl v0.4.0 + TensorKitSectors v0.3.9:
//   Bsymbol((4,2,0),(3,1,0),(3,1,0)) = [[-1/(2√2), -√(7/8)], [√(7/8), -1/(2√2)]]
//   Bsymbol((3,1,0),(3,2,0),(4,2,0)) = its inverse (real orthogonal, so the
//   transpose): [[-1/(2√2), √(7/8)], [-√(7/8), -1/(2√2)]]
// (B_fwd · B_ret = I₂, verified in Julia). These are GENUINELY non-diagonal
// and non-symmetric, so they discriminate the μ↔ν indexing in the real bend,
// not just the F→B reshape (which TransposeProbeRule already pins).
//
// dim((4,2,0))=27, dim((3,1,0))=dim((3,2,0))=15, all FS phases +1.
// Only the two bend F-blocks are transcribed, as B/(√d_a·√d_b/√d_c) (the full
// SU(3) F-table is not), so the bend surgery, coeff₀ = √dim(c)/√dim(a), μ→ν row
// distribution and round-trip are all exercised against real categorical B.
#[derive(Clone, Copy, Debug)]
pub(super) struct Su3BendRule;

// ids: 1 = (4,2,0) self-dual, 2 = (3,1,0), 3 = (3,2,0) = dual((3,1,0)).
impl symtn_sectors::testing::Provider for Su3BendRule {}

impl FusionRule for Su3BendRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        match sector.id() {
            2 => SectorId::new(3),
            3 => SectorId::new(2),
            _ => sector, // 0, 1 self-dual
        }
    }
    // Honest N(a,b,c)=2 for the bent triples, and the vacuum channel of
    // (3,1,0)⊗(3,2,0) that sizes the bend F-blocks.
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(2)],
            (2, 3) | (3, 2) => smallvec![SectorId::new(0), SectorId::new(1)],
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        match (left.id(), right.id(), coupled.id()) {
            (1, 2, 2) | (2, 3, 1) => 2,
            _ => usize::from(self.fusion_channels(left, right).contains(&coupled)),
        }
    }
}

impl GenericFusionSymbols for Su3BendRule {
    type Scalar = f64;
    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        // Bend block F(a, b, b̄, a, c, 1): B = √d_a·√d_b/√d_c · F[μ,ν,0,0].
        assert_eq!(
            (c, d, f),
            (self.dual(b), a, self.vacuum()),
            "Su3BendRule: unmodelled F"
        );
        let factor =
            self.sqrt_dim_scalar(a) * self.sqrt_dim_scalar(b) * self.inv_sqrt_dim_scalar(e);
        let b_matrix = su3_bend_b(a, b, e);
        GenericFArray::new(b_matrix.iter().map(|x| x / factor).collect(), (2, 2, 1, 1))
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

impl GenericRigidSymbols for Su3BendRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        match sector.id() {
            1 => 27.0,
            2 | 3 => 15.0,
            _ => 1.0,
        }
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        match sector.id() {
            1 => 27.0_f64.sqrt(),
            2 | 3 => 15.0_f64.sqrt(),
            _ => 1.0,
        }
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        1.0 / self.sqrt_dim_scalar(sector)
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

fn su3_bend_b(a: SectorId, b: SectorId, c: SectorId) -> [f64; 4] {
    let e = -1.0 / (2.0 * 2.0_f64.sqrt()); // -1/(2√2) = -0.35355339
    let g = (7.0_f64 / 8.0).sqrt(); //  √(7/8)  =  0.93541435
    match (a.id(), b.id(), c.id()) {
        (1, 2, 2) => [e, -g, g, e], // B_fwd
        (2, 3, 1) => [e, g, -g, e], // B_ret
        other => panic!("Su3BendRule: unmodelled B{other:?}"),
    }
}

// ==================================================================
// Generic tree-pair permute / braid / transpose composers.
// These are thin structural mirrors of the multiplicity-free tree-pair
// functions, chaining the independently tested primitives
// (generic_braid_tree, generic_repartition_tree_pair, generic_cycle_*).
// The tests below prove the COMPOSITION adds no math: each composer
// equals the hand-chained primitives it is built from.
// ==================================================================

pub(super) use std::collections::HashMap;

pub(super) fn map_terms(terms: Vec<(FusionTreePairKey, f64)>) -> HashMap<FusionTreePairKey, f64> {
    let mut map = HashMap::new();
    for (key, coeff) in terms {
        *map.entry(key).or_insert(0.0) += coeff;
    }
    map
}

pub(super) fn assert_term_maps_eq(
    got: &HashMap<FusionTreePairKey, f64>,
    want: &HashMap<FusionTreePairKey, f64>,
    label: &str,
) {
    let mut keys: std::collections::HashSet<&FusionTreePairKey> = got.keys().collect();
    keys.extend(want.keys());
    for key in keys {
        let g = got.get(key).copied().unwrap_or(0.0);
        let w = want.get(key).copied().unwrap_or(0.0);
        assert!((g - w).abs() < 1e-10, "{label}: coeff {g} != {w}");
    }
}

pub(super) fn u1_leg(charge: i32, deg: usize, dual: bool) -> SectorLeg {
    SectorLeg::new([(u1(charge), deg)], dual)
}

// Fixture for the cached-hash contracts (#1207): keys that differ from a
// base tree in exactly one identity field each (uncoupled, coupled, dual
// flag, inner line, vertex), plus equal copies built with shared and with
// fresh backings.
pub(super) fn key_hash_fixture() -> (Vec<FusionTreeKey>, FusionTreeKey, FusionTreeKey) {
    let base = FusionTreeKey::try_from_sector_ids([3, 3, 3], 3, [false, true, false], [1], [1, 2])
        .unwrap();
    let distinct = vec![
        base.clone(),
        FusionTreeKey::try_from_sector_ids([3, 3, 4], 3, [false, true, false], [1], [1, 2])
            .unwrap(),
        FusionTreeKey::try_from_sector_ids([3, 3, 3], 4, [false, true, false], [1], [1, 2])
            .unwrap(),
        FusionTreeKey::try_from_sector_ids([3, 3, 3], 3, [false, true, true], [1], [1, 2]).unwrap(),
        FusionTreeKey::try_from_sector_ids([3, 3, 3], 3, [false, true, false], [2], [1, 2])
            .unwrap(),
        FusionTreeKey::try_from_sector_ids([3, 3, 3], 3, [false, true, false], [1], [2, 2])
            .unwrap(),
    ];
    // The same tree on a later row of a shared arena.
    let mut builder = TreeArenaBuilder::default();
    builder.push_view(distinct[1].as_view());
    let row = builder.push_view(base.as_view());
    let shared = builder.freeze().key(row);
    let fresh = FusionTreeKey::new(
        base.uncoupled().iter().copied(),
        base.coupled(),
        base.is_dual().iter().copied(),
        base.innerlines().iter().copied(),
        base.vertices().iter().copied(),
    );
    (distinct, shared, fresh)
}

pub(super) fn fx_hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Admits `canonical` through the construction witness with no element
/// visit, then checks the same geometry without the witness through the
/// exact per-element enumeration, which must agree and must actually
/// enumerate (interleaved subblocks of one coupled-sector matrix).
pub(super) fn assert_canonical_storage_admitted_without_enumeration(canonical: &BlockStructure) {
    assert!(canonical.storage_tiling_proven());
    // Independent of `record_storage_tiling`: every offset of the payload
    // is reached by exactly one (block, element), the fact the
    // uninitialized-output path relies on.
    let mut hits = vec![0usize; canonical.required_len().unwrap()];
    for index in 0..canonical.block_count() {
        let block = canonical.block(index).unwrap();
        let count = block.shape().iter().product::<usize>();
        for linear in 0..count {
            let mut rest = linear;
            let mut offset = block.offset();
            for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
                offset += (rest % extent) * stride;
                rest /= extent;
            }
            hits[offset] += 1;
        }
    }
    assert!(
        hits.iter().all(|&hit| hit == 1),
        "offset histogram {hits:?}"
    );
    let recording = crate::probe::record();
    validate_block_storage_injective(canonical).unwrap();
    assert_eq!(recording.get(crate::probe::Event::ExactStorageFallback), 0);

    let unwitnessed = PreparedBlockStructure::from_parts(
        canonical.sector_structure().clone(),
        canonical.degeneracy_structure().clone(),
    )
    .unwrap();
    assert!(!unwitnessed.structure().storage_tiling_proven());
    let recording = crate::probe::record();
    validate_block_storage_injective(unwitnessed.structure()).unwrap();
    assert!(recording.get(crate::probe::Event::ExactStorageFallback) > 0);
}
