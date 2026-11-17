use super::*;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::hash::Hash;
use symtn_core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, FusionStyleKind, LieGroupFusionRule,
    LieGroupFusionRuleError, SectorVec,
};

/// Every provider query of a build, by kind and arguments.
#[derive(Default)]
struct Log {
    n: Vec<[SectorId; 3]>,
    ch: Vec<[SectorId; 2]>,
    f: Vec<[SectorId; 6]>,
    r: Vec<[SectorId; 3]>,
    a: Vec<[SectorId; 3]>,
    b: Vec<[SectorId; 3]>,
}

#[derive(Debug)]
enum SpyError {
    FailingN,
    Delegated(LieGroupFusionRuleError),
}
impl std::fmt::Display for SpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SpyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FailingN => None,
            Self::Delegated(error) => Some(error),
        }
    }
}

/// Checked SU(3) that logs each query it answers and fails one `N` triple.
struct Spy {
    inner: LieGroupFusionRule,
    log: RefCell<Log>,
    failing_n: Cell<Option<[SectorId; 3]>>,
}
impl symtn_core::testing::Provider for Spy {}
impl CheckedGenericFusion for Spy {
    type Error = SpyError;
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(&self.inner)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(&self.inner)
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }
    fn try_dual(&self, a: SectorId) -> Result<SectorId, Self::Error> {
        self.inner.try_dual(a).map_err(SpyError::Delegated)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        self.log.borrow_mut().ch.push([a, b]);
        self.inner
            .try_fusion_channels(a, b)
            .map_err(SpyError::Delegated)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.log.borrow_mut().ch.push([a, b]);
        self.inner
            .try_fusion_channels_in_table(a, b)
            .map_err(SpyError::Delegated)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.log.borrow_mut().n.push([a, b, c]);
        if self.failing_n.get() == Some([a, b, c]) {
            return Err(SpyError::FailingN);
        }
        self.inner.try_nsymbol(a, b, c).map_err(SpyError::Delegated)
    }
}
impl CheckedGenericRigidSymbols for Spy {
    type Scalar = f64;
    fn try_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner.try_dim_scalar(a).map_err(SpyError::Delegated)
    }
    fn try_sqrt_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner
            .try_sqrt_dim_scalar(a)
            .map_err(SpyError::Delegated)
    }
    fn try_inv_sqrt_dim_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner
            .try_inv_sqrt_dim_scalar(a)
            .map_err(SpyError::Delegated)
    }
    fn try_frobenius_schur_phase_scalar(&self, a: SectorId) -> Result<f64, Self::Error> {
        self.inner
            .try_frobenius_schur_phase_scalar(a)
            .map_err(SpyError::Delegated)
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
        self.log.borrow_mut().f.push([a, b, c, d, e, f]);
        self.inner
            .try_f_symbol_generic(a, b, c, d, e, f)
            .map_err(SpyError::Delegated)
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.log.borrow_mut().r.push([a, b, c]);
        self.inner
            .try_r_symbol_generic(a, b, c)
            .map_err(SpyError::Delegated)
    }
    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.log.borrow_mut().a.push([a, b, c]);
        self.inner
            .try_a_symbol_generic(a, b, c)
            .map_err(SpyError::Delegated)
    }
    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.log.borrow_mut().b.push([a, b, c]);
        self.inner
            .try_b_symbol_generic(a, b, c)
            .map_err(SpyError::Delegated)
    }
}

/// SU(3) with legs `3 ⊕ 8`, rank 2 → 2: several fusion-tree groups, and
/// `8 ⊗ 8` has multiplicity-two vertices, so rows of one group differ only
/// in vertex labels and reuse every symbol.
fn fixture() -> (Spy, Arc<BlockStructure>) {
    let inner = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
        fusion_symbols::lie::GroupId::su(3).unwrap(),
    )
    .unwrap();
    let fundamental = inner.encode_irrep_with_charges(&[1, 0], &[]).unwrap();
    let adjoint = inner.encode_irrep_with_charges(&[1, 1], &[]).unwrap();
    let spy = Spy {
        inner,
        log: RefCell::default(),
        failing_n: Cell::new(None),
    };
    let leg = || SectorLeg::new([(fundamental, 1), (adjoint, 1)], false);
    let structure = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    )
    .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&spy)
    .unwrap();
    assert!(structure.fusion_tree_group_slice().len() > 1);
    (spy, structure)
}

/// The `N` and channel queries of source admission, which the build entry
/// runs first and production runs once per structure: they precede the
/// build's own queries in the log.
fn admission_prefix(spy: &Spy, structure: &BlockStructure) -> (usize, usize) {
    *spy.log.borrow_mut() = Log::default();
    symtn_core::CheckedGenericAdmittedFusionTreeBlockStructure::try_new(spy, structure).unwrap();
    let log = spy.log.borrow();
    assert!(log.f.is_empty() && log.r.is_empty());
    (log.n.len(), log.ch.len())
}

fn all_distinct<T: Hash + Eq>(queries: &[T]) -> bool {
    queries.iter().collect::<HashSet<_>>().len() == queries.len()
}

fn distinct<T: Hash + Eq + Clone>(queries: &[T]) -> HashSet<T> {
    queries.iter().cloned().collect()
}

/// What: a cold checked Generic build sends each distinct provider query once,
/// however many basis rows read it, and keeps nothing for the next build.
/// The permute runs bends and Artin moves, the transpose folds (tree
/// enumeration, A symbols).
#[test]
fn checked_generic_build_queries_each_provider_argument_once() {
    let (spy, structure) = fixture();
    let (admission_n, admission_ch) = admission_prefix(&spy, &structure);
    for operation in [
        TreeTransformOperation::permute([2, 3], [1, 0]),
        TreeTransformOperation::transpose([1, 3], [0, 2]),
    ] {
        let mut first = None;
        for _build in 0..2 {
            *spy.log.borrow_mut() = Log::default();
            crate::build_checked_generic_tree_pair_transform_group_plan(
                &spy,
                operation.clone(),
                &structure,
            )
            .unwrap();
            let log = spy.log.borrow();
            let (n, ch) = (&log.n[admission_n..], &log.ch[admission_ch..]);
            assert!(all_distinct(n), "{operation:?}");
            assert!(all_distinct(ch), "{operation:?}");
            assert!(all_distinct(&log.f), "{operation:?}");
            assert!(all_distinct(&log.r), "{operation:?}");
            assert!(all_distinct(&log.a), "{operation:?}");
            assert!(all_distinct(&log.b), "{operation:?}");
            assert!(!n.is_empty() && !log.f.is_empty() && !log.b.is_empty());
            let sets = (
                distinct(n),
                distinct(ch),
                distinct(&log.f),
                distinct(&log.r),
                distinct(&log.a),
                distinct(&log.b),
            );
            match &first {
                None => first = Some(sets),
                Some(first) => assert!(*first == sets, "{operation:?}: the next build re-queries"),
            }
        }
    }
}

/// What: a provider error inside a build is reported, not kept as an answer:
/// failing an `N` triple that only the build queries fails the build with
/// that provider error.
#[test]
fn checked_generic_build_reports_a_provider_error_it_meets() {
    let (spy, structure) = fixture();
    let (admission_n, _) = admission_prefix(&spy, &structure);
    let admitted = spy.log.borrow().n.iter().copied().collect::<HashSet<_>>();
    let operation = TreeTransformOperation::permute([2, 3], [1, 0]);
    *spy.log.borrow_mut() = Log::default();
    crate::build_checked_generic_tree_pair_transform_group_plan(
        &spy,
        operation.clone(),
        &structure,
    )
    .unwrap();
    let build_only = spy.log.borrow().n[admission_n..]
        .iter()
        .copied()
        .find(|triple| !admitted.contains(triple))
        .expect("the build queries an N triple admission does not");
    spy.failing_n.set(Some(build_only));
    let error =
        crate::build_checked_generic_tree_pair_transform_group_plan(&spy, operation, &structure)
            .unwrap_err();
    assert!(
        matches!(error, CheckedGenericPlanError::Provider(SpyError::FailingN)),
        "{error:?}"
    );
}
