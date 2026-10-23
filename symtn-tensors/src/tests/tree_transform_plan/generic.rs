use super::*;
use crate::probe::{self, Event};

// ======================================================================
// Generic-fusion (outer-multiplicity) plan compile.
// A minimal Generic rule with one self-dual outer-multiplicity sector (1):
// N(1,1,1)=2 makes the rule genuinely Generic, while the tree pairs used
// below couple [1,1] to the vacuum (N(1,1,0)=1), so the rank-2 codomain
// braid is a pure 1×1 R-symbol — no bends, no F-moves. This exercises the
// `build_generic_tree_pair_transform_group_plan` wiring (style guard, group
// iteration, shared assembly, core-row dispatch) without an external symbol
// table; the recoupling math itself is proven in symtn-core's Generic recoupling tests.
// ======================================================================

#[derive(Clone, Copy)]
pub(super) struct ToyGenericRule {
    pub(super) style: FusionStyleKind,
}

impl symtn_core::testing::Provider for ToyGenericRule {}

impl FusionRule for ToyGenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.style
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
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => [SectorId::new(0)].into_iter().collect(),
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

impl GenericFusionSymbols for ToyGenericRule {
    type Scalar = f64;
    fn f_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
        _d: SectorId,
        _e: SectorId,
        _f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        // Unreached for the rank-2 vacuum-coupled pair (pure R braid).
        GenericFArray::new(vec![1.0], (1, 1, 1, 1))
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        if c == SectorId::new(1) {
            GenericRMatrix::new(vec![0.0, 2.0, 3.0, 0.0], 2, 2)
        } else {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
}

impl GenericRigidSymbols for ToyGenericRule {
    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy)]
pub(super) struct DenseGenericRule;

impl symtn_core::testing::Provider for DenseGenericRule {}

impl FusionRule for DenseGenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
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
        sector
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => [SectorId::new(x)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => [SectorId::new(0)].into_iter().collect(),
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

impl GenericFusionSymbols for DenseGenericRule {
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
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let cols = shape.2 * shape.3;
        let mut data = vec![0.0; rows * cols];
        for index in 0..rows.min(cols) {
            data[index * cols + index] = 1.0;
        }
        GenericFArray::new(data, shape)
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        if c == SectorId::new(1) {
            GenericRMatrix::new(vec![2.0, 5.0, 7.0, 11.0], 2, 2)
        } else {
            GenericRMatrix::new(vec![1.0], 1, 1)
        }
    }
}

impl GenericRigidSymbols for DenseGenericRule {
    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CheckedPlanCall {
    Dual,
    N,
    Dim,
    SqrtDim,
    InvSqrtDim,
    FrobeniusSchur,
    F,
    R,
}

impl CheckedPlanCall {
    pub(super) const COUNT: usize = 8;

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MalformedCheckedSymbol {
    F,
    R,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CheckedPlanSpyError(pub(super) CheckedPlanCall);

impl std::fmt::Display for CheckedPlanSpyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "injected {:?} failure", self.0)
    }
}

impl std::error::Error for CheckedPlanSpyError {}

pub(super) struct CheckedPlanSpy<'a, R> {
    rule: &'a R,
    pub(super) identity: std::cell::RefCell<Option<symtn_core::RuleIdentity>>,
    pub(super) fusion_style: std::cell::Cell<Option<FusionStyleKind>>,
    pub(super) braiding_style: std::cell::Cell<Option<BraidingStyleKind>>,
    pub(super) fail: std::cell::Cell<Option<(CheckedPlanCall, usize)>>,
    pub(super) malformed: std::cell::Cell<Option<MalformedCheckedSymbol>>,
    /// Switches the reported fusion style at this call: a provider whose
    /// capability changes after the group builds, before the commit.
    pub(super) restyle_at: std::cell::Cell<Option<(CheckedPlanCall, FusionStyleKind)>>,
    pub(super) calls: std::cell::Cell<[usize; CheckedPlanCall::COUNT]>,
}

impl<'a, R> CheckedPlanSpy<'a, R> {
    pub(super) fn new(rule: &'a R) -> Self {
        Self {
            rule,
            identity: std::cell::RefCell::new(None),
            fusion_style: std::cell::Cell::new(None),
            braiding_style: std::cell::Cell::new(None),
            fail: std::cell::Cell::new(None),
            malformed: std::cell::Cell::new(None),
            restyle_at: std::cell::Cell::new(None),
            calls: std::cell::Cell::new([0; CheckedPlanCall::COUNT]),
        }
    }

    fn trip(&self, call: CheckedPlanCall) -> Result<(), CheckedPlanSpyError> {
        let mut calls = self.calls.get();
        calls[call.index()] += 1;
        self.calls.set(calls);
        if let Some((at, style)) = self.restyle_at.get() {
            if at == call {
                self.fusion_style.set(Some(style));
            }
        }
        if self.fail.get() == Some((call, calls[call.index()])) {
            Err(CheckedPlanSpyError(call))
        } else {
            Ok(())
        }
    }

    pub(super) fn call_count(&self, call: CheckedPlanCall) -> usize {
        self.calls.get()[call.index()]
    }
}

/// The rule identity of a [`CheckedPlanSpy`] wrapping `R`.
struct CheckedPlanSpyIdentity<R>(std::marker::PhantomData<R>);

impl<R: FusionRule + 'static> symtn_core::testing::Provider for CheckedPlanSpy<'_, R> {}

impl<R: FusionRule + 'static> CheckedGenericFusion for CheckedPlanSpy<'_, R> {
    type Error = CheckedPlanSpyError;

    // Why not the wrapped rule's identity: composed coefficients are cached
    // per identity, and a sibling test's publication under the shared rule's
    // identity would skip this spy's provider queries and failures. Why one
    // identity per wrapped rule type: spies of different rules must not
    // alias each other's entries either.
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        self.identity
            .borrow()
            .clone()
            .unwrap_or_else(symtn_core::RuleIdentity::of_type::<CheckedPlanSpyIdentity<R>>)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.fusion_style
            .get()
            .unwrap_or_else(|| self.rule.fusion_style())
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.braiding_style
            .get()
            .unwrap_or_else(|| self.rule.braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.trip(CheckedPlanCall::Dual)?;
        Ok(self.rule.dual(sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.rule.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(self.rule.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.trip(CheckedPlanCall::N)?;
        Ok(self.rule.nsymbol(left, right, coupled))
    }
}

impl<R> CheckedGenericRigidSymbols for CheckedPlanSpy<'_, R>
where
    R: GenericRigidSymbols,
    R::Scalar: symtn_core::CategoricalScalar + Send + Sync,
{
    type Scalar = R::Scalar;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.trip(CheckedPlanCall::Dim)?;
        Ok(self.rule.dim_scalar(sector))
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.trip(CheckedPlanCall::SqrtDim)?;
        Ok(self.rule.sqrt_dim_scalar(sector))
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.trip(CheckedPlanCall::InvSqrtDim)?;
        Ok(self.rule.inv_sqrt_dim_scalar(sector))
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        self.trip(CheckedPlanCall::FrobeniusSchur)?;
        Ok(self.rule.frobenius_schur_phase_scalar(sector))
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
        self.trip(CheckedPlanCall::F)?;
        let symbol = self.rule.f_symbol_generic(a, b, c, d, e, f);
        if self.malformed.get() == Some(MalformedCheckedSymbol::F) {
            Ok(GenericFArray::new(
                symbol.data().to_vec(),
                (1, 1, symbol.data().len(), 1),
            ))
        } else {
            Ok(symbol)
        }
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        self.trip(CheckedPlanCall::R)?;
        let symbol = self.rule.r_symbol_generic(a, b, c);
        if self.malformed.get() == Some(MalformedCheckedSymbol::R) {
            Ok(GenericRMatrix::new(
                symbol.data().to_vec(),
                1,
                symbol.data().len(),
            ))
        } else {
            Ok(symbol)
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[derive(Clone, Copy, Debug)]
enum MeasurementProviderCall {
    Channels,
    Dual,
    N,
    SqrtDim,
    InvSqrtDim,
    FrobeniusSchur,
    F,
    R,
    Dim,
}

#[cfg(feature = "cgc-gen")]
impl MeasurementProviderCall {
    const COUNT: usize = 9;

    fn index(self) -> usize {
        self as usize
    }
}

#[cfg(feature = "cgc-gen")]
struct MeasurementProvider<P> {
    inner: P,
    calls: std::cell::Cell<[usize; MeasurementProviderCall::COUNT]>,
}

#[cfg(feature = "cgc-gen")]
impl<P> MeasurementProvider<P> {
    fn new(inner: P) -> Self {
        Self {
            inner,
            calls: std::cell::Cell::new([0; MeasurementProviderCall::COUNT]),
        }
    }

    fn hit(&self, call: MeasurementProviderCall) {
        let mut calls = self.calls.get();
        calls[call.index()] += 1;
        self.calls.set(calls);
    }

    fn reset_calls(&self) {
        self.calls.set([0; MeasurementProviderCall::COUNT]);
    }
}

#[cfg(feature = "cgc-gen")]
impl<P: CheckedGenericFusion> symtn_core::testing::Provider for MeasurementProvider<P> {}

#[cfg(feature = "cgc-gen")]
impl<P: CheckedGenericFusion> CheckedGenericFusion for MeasurementProvider<P> {
    type Error = P::Error;

    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        self.inner.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.inner.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.hit(MeasurementProviderCall::Dual);
        self.inner.try_dual(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.hit(MeasurementProviderCall::Channels);
        self.inner.try_fusion_channels(left, right)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.hit(MeasurementProviderCall::Channels);
        self.inner.try_fusion_channels_in_table(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.hit(MeasurementProviderCall::N);
        self.inner.try_nsymbol(left, right, coupled)
    }
}

#[cfg(feature = "cgc-gen")]
impl<P: CheckedGenericRigidSymbols> CheckedGenericRigidSymbols for MeasurementProvider<P> {
    type Scalar = P::Scalar;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.hit(MeasurementProviderCall::Dim);
        self.inner.try_dim_scalar(sector)
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.hit(MeasurementProviderCall::SqrtDim);
        self.inner.try_sqrt_dim_scalar(sector)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<Self::Scalar, Self::Error> {
        self.hit(MeasurementProviderCall::InvSqrtDim);
        self.inner.try_inv_sqrt_dim_scalar(sector)
    }

    fn try_frobenius_schur_phase_scalar(
        &self,
        sector: SectorId,
    ) -> Result<Self::Scalar, Self::Error> {
        self.hit(MeasurementProviderCall::FrobeniusSchur);
        self.inner.try_frobenius_schur_phase_scalar(sector)
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
        self.hit(MeasurementProviderCall::F);
        self.inner.try_f_symbol_generic(a, b, c, d, e, f)
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<Self::Scalar>, Self::Error> {
        self.hit(MeasurementProviderCall::R);
        self.inner.try_r_symbol_generic(a, b, c)
    }
}

pub(super) struct SynchronizedCheckedGeneric {
    rule: DenseGenericRule,
    pub(super) calls: std::sync::Mutex<usize>,
}

impl SynchronizedCheckedGeneric {
    pub(super) fn new() -> Self {
        Self {
            rule: DenseGenericRule,
            calls: std::sync::Mutex::new(0),
        }
    }

    fn hit(&self) {
        *self.calls.lock().unwrap() += 1;
    }
}

impl symtn_core::testing::Provider for SynchronizedCheckedGeneric {}

impl FusionRule for SynchronizedCheckedGeneric {
    fn rule_identity(&self) -> RuleIdentity {
        self.rule.rule_identity()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.rule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.rule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.rule.nsymbol(left, right, coupled)
    }
}

impl CheckedGenericFusion for SynchronizedCheckedGeneric {
    type Error = std::convert::Infallible;

    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        self.rule.rule_identity()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.rule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.rule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        self.rule.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.hit();
        Ok(self.rule.dual(sector))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.hit();
        Ok(self.rule.fusion_channels(left, right))
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.hit();
        Ok(self.rule.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.hit();
        Ok(self.rule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for SynchronizedCheckedGeneric {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.hit();
        Ok(self.rule.dim_scalar(sector))
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.hit();
        Ok(self.rule.sqrt_dim_scalar(sector))
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.hit();
        Ok(self.rule.inv_sqrt_dim_scalar(sector))
    }
    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.hit();
        Ok(self.rule.frobenius_schur_phase_scalar(sector))
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
        self.hit();
        Ok(self.rule.f_symbol_generic(a, b, c, d, e, f))
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.hit();
        Ok(self.rule.r_symbol_generic(a, b, c))
    }
}

#[derive(Clone, Copy)]
pub(super) struct AnyonicGenericRule;

impl symtn_core::testing::Provider for AnyonicGenericRule {}

impl FusionRule for AnyonicGenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }
    fn vacuum(&self) -> SectorId {
        DenseGenericRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        DenseGenericRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        DenseGenericRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        DenseGenericRule.nsymbol(left, right, coupled)
    }
}

impl GenericFusionSymbols for AnyonicGenericRule {
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
        DenseGenericRule.f_symbol_generic(a, b, c, d, e, f)
    }
    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        DenseGenericRule.r_symbol_generic(a, b, c)
    }
}

impl GenericRigidSymbols for AnyonicGenericRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.dim_scalar(sector)
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.sqrt_dim_scalar(sector)
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.inv_sqrt_dim_scalar(sector)
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.frobenius_schur_phase_scalar(sector)
    }
}

pub(super) fn vacuum_rank2_src_pair_for_rule<R: FusionRule>(rule: &R) -> FusionTreePairKey {
    let cod = FusionTreeKey::try_new_for_rule(
        rule,
        [SectorId::new(1), SectorId::new(1)],
        SectorId::new(0),
        [false, false],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    let dom = FusionTreeKey::try_new_for_rule(rule, [], SectorId::new(0), [], [], []).unwrap();
    FusionTreePairKey::pair(cod, dom)
}

pub(super) fn vacuum_rank2_toy_src_pair() -> FusionTreePairKey {
    // cod [1,1]->0 (vacuum-coupled, N(1,1,0)=1, single vertex label 1), dom []->0.
    vacuum_rank2_src_pair_for_rule(&ToyGenericRule {
        style: FusionStyleKind::Generic,
    })
}

pub(super) fn dense_generic_source_pairs(rule: &DenseGenericRule) -> [FusionTreePairKey; 2] {
    [MultiplicityIndex::ONE, MultiplicityIndex::new(2).unwrap()].map(|vertex| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                rule,
                [SectorId::new(1), SectorId::new(1)],
                SectorId::new(1),
                [false, false],
                [],
                [vertex],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                rule,
                [SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [],
            )
            .unwrap(),
        )
    })
}

pub(super) fn dense_generic_rank3_pairs(rule: &DenseGenericRule) -> [FusionTreePairKey; 2] {
    let charge = SectorId::new(1);
    let vacuum = SectorId::new(0);
    [MultiplicityIndex::ONE, MultiplicityIndex::new(2).unwrap()].map(|vertex| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                rule,
                [charge, charge, charge],
                vacuum,
                [false, false, false],
                [charge],
                [vertex, MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(rule, [], vacuum, [], [], []).unwrap(),
        )
    })
}

pub(super) fn dense_generic_dual_source_pair(rule: &DenseGenericRule) -> FusionTreePairKey {
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            rule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(1),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            rule,
            [SectorId::new(1)],
            SectorId::new(1),
            [false],
            [],
            [],
        )
        .unwrap(),
    )
}

pub(super) fn dense_generic_dynamic_space() -> crate::contract::DynamicFusionMapSpace {
    let sector = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(
            [2usize, 3].map(|degeneracy| SectorLeg::new([(sector, degeneracy)], false)),
        ),
        FusionProductSpace::new([SectorLeg::new([(sector, 5)], false)]),
    );
    crate::contract::DynamicFusionMapSpace::from_final_homspace_generic(&DenseGenericRule, homspace)
        .unwrap()
}

pub(super) fn literal_dense_generic_adjoint_braid(
    parent_data: &[Complex64],
    alpha: Complex64,
) -> Vec<Complex64> {
    assert_eq!(parent_data.len(), 60);
    let coefficients = [[2.0, 5.0], [7.0, 11.0]];
    let mut expected = vec![Complex64::new(0.0, 0.0); 60];
    for destination_vertex in 0..2 {
        for source_axis_0 in 0..2 {
            for source_axis_1 in 0..3 {
                for source_axis_2 in 0..5 {
                    let mut value = Complex64::new(0.0, 0.0);
                    for (source_vertex, source_coefficients) in coefficients.iter().enumerate() {
                        let parent_position = source_vertex * 30
                            + source_axis_2
                            + 5 * source_axis_0
                            + 10 * source_axis_1;
                        value += parent_data[parent_position].conj()
                            * source_coefficients[destination_vertex];
                    }
                    let destination_position = destination_vertex * 6
                        + source_axis_1
                        + 3 * source_axis_0
                        + 12 * source_axis_2;
                    expected[destination_position] = alpha * value;
                }
            }
        }
    }
    expected
}

// The generic plan compile reproduces the core `generic_permute_tree_pair`
// rows exactly (the assembly adds no math), and its runtime style gate rejects
// a rule that reports a multiplicity-free style — the symmetric sibling of the
// mult-free builders' `UnsupportedFusionStyle` guards.
#[test]
fn build_generic_tree_pair_plan_matches_core_rows_and_guards_style() {
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let src_pair = vacuum_rank2_toy_src_pair();
    let src_key = BlockKey::from(src_pair.clone());
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    // The plan's single group spec must reproduce the per-source core rows for
    // each operation kind (exercises every `transformed_generic_tree_pair_rows`
    // arm). cod [1,1]->0 with an empty domain stays a pure 1×1 R braid, so the
    // codomain permute/braid and the cyclic transpose all resolve numerically.
    let assert_plan_matches =
        |operation: TreeTransformOperation, core_rows: Vec<(FusionTreePairKey, f64)>| {
            assert_eq!(core_rows.len(), 1);
            let (core_dst, core_coeff) = &core_rows[0];
            let plan =
                build_generic_tree_pair_transform_group_plan(&rule, operation, &src_structure)
                    .unwrap();
            assert_eq!(plan.specs().len(), 1);
            let spec = &plan.specs()[0];
            assert_eq!(spec.src_keys(), std::slice::from_ref(&src_pair));
            assert_eq!(spec.dst_keys(), std::slice::from_ref(core_dst));
            assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
            assert!((spec.recoupling_coefficients_dst_src()[0] - core_coeff).abs() < 1e-12);
        };

    assert_plan_matches(
        TreeTransformOperation::permute([1, 0], []),
        generic_permute_tree_pair(&rule, &src_pair, &[1, 0], &[]).unwrap(),
    );
    assert_plan_matches(
        TreeTransformOperation::braid([1, 0], [], [0, 1], []),
        generic_braid_tree_pair(&rule, &src_pair, &[1, 0], &[], &[0, 1], &[]).unwrap(),
    );
    assert_plan_matches(
        TreeTransformOperation::braid([0, 1], [], [29, 7], []),
        vec![(src_pair.clone(), 1.0)],
    );
    assert_plan_matches(
        TreeTransformOperation::transpose([1, 0], []),
        generic_transpose_tree_pair(&rule, &src_pair, &[1, 0], &[]).unwrap(),
    );

    // Style guard: a multiplicity-free style is rejected before any compile.
    let mf = ToyGenericRule {
        style: FusionStyleKind::Simple,
    };
    let err = build_generic_tree_pair_transform_group_plan(
        &mf,
        TreeTransformOperation::permute([1, 0], []),
        &src_structure,
    )
    .unwrap_err();
    assert!(matches!(err, OperationError::UnsupportedFusionStyle { .. }));
}

#[test]
fn generic_multiplicity_monomial_rows_compile_and_execute_as_direct_singles() {
    // What: a GenericFusion R matrix whose core rows are structurally
    // singleton and destination-injective uses the same direct replay contract.
    let rule = ToyGenericRule {
        style: FusionStyleKind::Generic,
    };
    let pairs = [MultiplicityIndex::ONE, MultiplicityIndex::new(2).unwrap()].map(|vertex| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [SectorId::new(1), SectorId::new(1)],
                SectorId::new(1),
                [false, false],
                [],
                [vertex],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &rule,
                [SectorId::new(1)],
                SectorId::new(1),
                [false],
                [],
                [],
            )
            .unwrap(),
        )
    });
    let keys = pairs.clone().map(BlockKey::from);
    let structure =
        packed_fixture_structure(3, keys.iter().cloned().map(|key| (key, vec![1usize; 3])))
            .unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);
    let core_rows = pairs
        .iter()
        .map(|pair| generic_braid_tree_pair(&rule, pair, &[1, 0], &[2], &[0, 1], &[2]).unwrap())
        .collect::<Vec<_>>();
    assert!(core_rows.iter().all(|row| row.len() == 1));
    assert_ne!(core_rows[0][0].0, core_rows[1][0].0);

    let tree_pair_lowering_calls_rec = probe::record();
    let plan = build_generic_tree_pair_transform_group_plan(&rule, operation, &structure).unwrap();
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        structure.fusion_tree_groups().len()
    );
    assert_eq!(plan.specs().len(), 2);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[2.0]);
    assert_eq!(plan.specs()[1].recoupling_coefficients_dst_src(), &[3.0]);
    let compiled = plan.compile_structures(&structure, &structure).unwrap();
    assert!(!compiled.has_pack_gemm_scatter_blocks());

    let space = vec![1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![5.0, 7.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![11.0, 13.0], space, structure).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[75.0, 59.0]);
    assert_eq!(
        (workspace.source_len(), workspace.destination_len()),
        (0, 0)
    );
}

// `DenseGenericRule` whose R(1,1,1) is 1x1 although N(1,1,1) = 2: an
// inconsistent user provider of the open `GenericFusionSymbols` trait.
#[derive(Clone, Copy)]
struct TruncatedRGenericRule;

impl symtn_core::testing::Provider for TruncatedRGenericRule {}

impl FusionRule for TruncatedRGenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        DenseGenericRule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        DenseGenericRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        DenseGenericRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        DenseGenericRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        DenseGenericRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        DenseGenericRule.nsymbol(left, right, coupled)
    }
}

impl GenericFusionSymbols for TruncatedRGenericRule {
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
        DenseGenericRule.f_symbol_generic(a, b, c, d, e, f)
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

impl GenericRigidSymbols for TruncatedRGenericRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.dim_scalar(sector)
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.sqrt_dim_scalar(sector)
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.inv_sqrt_dim_scalar(sector)
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        DenseGenericRule.frobenius_schur_phase_scalar(sector)
    }
}

#[test]
fn generic_braid_plan_rejects_r_symbol_shape_mismatch() {
    let pairs = dense_generic_source_pairs(&DenseGenericRule);
    let structure = packed_fixture_structure(
        3,
        pairs
            .iter()
            .cloned()
            .map(BlockKey::from)
            .map(|key| (key, vec![1usize; 3])),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);

    // What: the tensor-level Generic braid plan reports a malformed provider
    // symbol as a typed error instead of panicking.
    let error =
        build_generic_tree_pair_transform_group_plan(&TruncatedRGenericRule, operation, &structure)
            .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "Generic symbol shape mismatch",
        })
    );
}

#[test]
fn generic_dense_block_plan_matches_per_source_oracle_matrix() {
    let rule = DenseGenericRule;
    let pairs = dense_generic_source_pairs(&rule);
    let structure = packed_fixture_structure(
        3,
        pairs
            .iter()
            .cloned()
            .map(BlockKey::from)
            .map(|key| (key, vec![1usize; 3])),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);

    let oracle = pairs
        .iter()
        .map(|pair| generic_braid_tree_pair(&rule, pair, &[1, 0], &[2], &[0, 1], &[2]).unwrap())
        .collect::<Vec<_>>();
    assert!(oracle.iter().all(|rows| rows.len() == 2));

    let mut expected_destinations = Vec::<FusionTreePairKey>::new();
    for rows in &oracle {
        for (destination, _) in rows {
            if !expected_destinations
                .iter()
                .any(|existing| existing == destination)
            {
                expected_destinations.push(destination.clone());
            }
        }
    }
    let mut expected_coefficients = vec![0.0; expected_destinations.len() * pairs.len()];
    for (source, rows) in oracle.iter().enumerate() {
        for (destination, coefficient) in rows {
            let row = expected_destinations
                .iter()
                .position(|existing| existing == destination)
                .unwrap();
            expected_coefficients[row * pairs.len() + source] = *coefficient;
        }
    }

    let plan = build_generic_tree_pair_transform_group_plan(&rule, operation, &structure).unwrap();
    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), pairs.as_slice());
    assert_eq!(spec.dst_keys(), expected_destinations.as_slice());
    assert_eq!(
        spec.recoupling_coefficients_dst_src(),
        expected_coefficients.as_slice()
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn racah_generated_sun_adjoint_checked_bound_round_trips() {
    use symtn_sectors::SUNFusionRule;

    fn repartition(source_nout: usize, target_nout: usize) -> TreeTransformOperation {
        let mut axes = (0..source_nout)
            .chain((source_nout..3).rev())
            .collect::<Vec<_>>();
        axes[target_nout..].reverse();
        let (codomain, domain) = axes.split_at(target_nout);
        TreeTransformOperation::transpose(codomain.iter().copied(), domain.iter().copied())
    }

    fn snapshot(
        rule: &SUNFusionRule,
        space: &crate::BoundDynamicFusionMapSpace<SUNFusionRule>,
    ) -> Vec<(Vec<Vec<i64>>, Vec<MultiplicityIndex>)> {
        let structure = space.space().structure();
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                let BlockKey::FusionTree(pair) = block.key() else {
                    unreachable!()
                };
                let mut labels = Vec::new();
                let mut vertices = Vec::new();
                for tree in [pair.codomain_tree(), pair.domain_tree()] {
                    labels.extend(
                        tree.uncoupled()
                            .iter()
                            .chain(tree.innerlines())
                            .chain(std::iter::once(&tree.coupled()))
                            .map(|&sector| rule.decode_dynkin(sector).unwrap()),
                    );
                    vertices.extend_from_slice(tree.vertices());
                }
                (labels, vertices)
            })
            .collect()
    }

    for (n, adjoint_labels) in [(3, &[1, 1][..]), (4, &[1, 0, 1][..])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let adjoint = provider.encode_dynkin(adjoint_labels).unwrap();
        assert_eq!(provider.try_nsymbol(adjoint, adjoint, adjoint).unwrap(), 2);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new(
                [1usize, 1].map(|degeneracy| SectorLeg::new([(adjoint, degeneracy)], false)),
            ),
            FusionProductSpace::new([SectorLeg::new([(adjoint, 1)], false)]),
        );
        let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            homspace,
        )
        .unwrap();
        assert!(Arc::ptr_eq(source.clone().provider_arc(), &provider));
        let source_snapshot = snapshot(provider.as_ref(), &source);
        let source_data = (0..source.space().required_len().unwrap())
            .map(|index| index as f64 + 1.0)
            .collect::<Vec<_>>();

        let mut operations = vec![(
            TreeTransformOperation::permute([1, 0], [2]),
            TreeTransformOperation::permute([1, 0], [2]),
        )];
        operations.extend((0..=3).map(|target| (repartition(2, target), repartition(target, 2))));
        for (forward, backward) in operations {
            let (moved, moved_data) =
                crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
                    .tree_transform_owned_checked_generic_in(
                        &source,
                        None,
                        &source_data,
                        &forward,
                        1.0,
                    )
                    .unwrap();
            assert!(Arc::ptr_eq(moved.provider_arc(), &provider));
            assert_eq!(moved_data.len(), moved.space().required_len().unwrap());
            let (round_trip, round_trip_data) = crate::TreeTransformExecutionContext::<
                f64,
                RuleIdentity,
                f64,
            >::default()
            .tree_transform_owned_checked_generic_in(&moved, None, &moved_data, &backward, 1.0)
            .unwrap();
            assert_eq!(round_trip.space(), source.space());
            assert_eq!(snapshot(provider.as_ref(), &round_trip), source_snapshot);
            assert!(Arc::ptr_eq(round_trip.provider_arc(), &provider));
            assert_eq!(round_trip_data.len(), source_data.len());
            for (actual, expected) in round_trip_data.iter().zip(&source_data) {
                assert!((actual - expected).abs() <= 1e-10);
            }
        }
    }
}

/// #2233: a checked SU(3) source admits its trees once. A warm transform
/// whose source outlived its structure-cache entry derives its destination
/// again, yet asks the provider no admission query (channels or N).
#[cfg(feature = "cgc-gen")]
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn warm_checked_generic_transform_of_an_evicted_source_makes_no_admission_query() {
    use symtn_sectors::SUNFusionRule;

    let inner = SUNFusionRule::new(3).unwrap();
    let adjoint = inner.encode_dynkin(&[1, 1]).unwrap();
    let fundamental = inner.encode_dynkin(&[1, 0]).unwrap();
    let provider = Arc::new(MeasurementProvider::new(inner));
    let leg = || SectorLeg::new([(fundamental, 1), (adjoint, 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let operation = TreeTransformOperation::permute([0, 2, 3], [1, 4]);
    let transform = || {
        crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
            .tree_transform_owned_checked_generic_in(&source, None, &data, &operation, 1.0)
            .unwrap()
    };
    drop(transform());
    symtn_core::clear_structure_caches();
    drop(transform());
    // Why the fewest of three calls: a concurrent test's clear may make one
    // call rebuild its destination layout; re-admission queries every call.
    let admission_queries = (0..3)
        .map(|_| {
            provider.reset_calls();
            drop(transform());
            let calls = provider.calls.get();
            calls[MeasurementProviderCall::Channels.index()]
                + calls[MeasurementProviderCall::N.index()]
        })
        .min();
    assert_eq!(admission_queries, Some(0));
}

#[cfg(feature = "cgc-gen")]
fn measured_provider_phase<T>(
    provider: &MeasurementProvider<symtn_sectors::SUNFusionRule>,
    case: &str,
    phase: &str,
    run: impl FnOnce() -> T,
) -> T {
    provider.reset_calls();
    let started = std::time::Instant::now();
    let output = run();
    println!(
        "case={case} phase={phase} ns={} calls={:?}",
        started.elapsed().as_nanos(),
        provider.calls.get()
    );
    output
}

#[cfg(feature = "cgc-gen")]
#[allow(clippy::arc_with_non_send_sync)]
fn measure_checked_generic_transform_case(
    n: usize,
    adjoint_labels: &[i64],
    case: &str,
    operation: TreeTransformOperation,
) {
    use symtn_sectors::SUNFusionRule;

    fusion_symbols::cache::reset();
    let inner = SUNFusionRule::new(n).unwrap();
    let adjoint = inner.encode_dynkin(adjoint_labels).unwrap();
    let provider = Arc::new(MeasurementProvider::new(inner));
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(
            [1usize, 1].map(|degeneracy| SectorLeg::new([(adjoint, degeneracy)], false)),
        ),
        FusionProductSpace::new([SectorLeg::new([(adjoint, 1)], false)]),
    );
    // Warm only Racah's product enumeration before measuring symtn's residual
    // lifecycle. Coefficient generation remains cold for the separately
    // reported first plan build below.
    let _product_warm_source =
        crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            homspace.clone(),
        )
        .unwrap();
    provider.reset_calls();

    let source = measured_provider_phase(provider.as_ref(), case, "source_admission", || {
        crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            homspace,
        )
        .unwrap()
    });
    let source_data = (0..source.space().required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();

    let _identity = measured_provider_phase(provider.as_ref(), case, "owned_preflight", || {
        let identity = source
            .space()
            .validate_transformed_generic_checked_identity(&CheckedGenericFusion::rule_identity(
                provider.as_ref(),
            ))
            .unwrap();
        crate::tree_transform::validate_checked_generic_tree_pair_plan_preflight(
            provider.as_ref(),
            &identity,
            &operation,
            source.space().structure(),
        )
        .unwrap();
        identity
    });
    let prepared = measured_provider_phase(provider.as_ref(), case, "destination_layout", || {
        source
            .space()
            .prepare_transformed_generic_checked(provider.as_ref(), &operation)
            .unwrap()
    });
    let plan = measured_provider_phase(
        provider.as_ref(),
        case,
        "plan_build_including_repeated_preflight_first",
        || {
            build_checked_generic_tree_pair_transform_group_plan(
                provider.as_ref(),
                operation.clone(),
                source.space().structure(),
            )
            .unwrap()
        },
    );
    let mut warm_plan_ns = Vec::with_capacity(7);
    let mut warm_plan_calls = None;
    for _ in 0..7 {
        provider.reset_calls();
        let started = std::time::Instant::now();
        let warm_plan = build_checked_generic_tree_pair_transform_group_plan(
            provider.as_ref(),
            operation.clone(),
            source.space().structure(),
        )
        .unwrap();
        std::hint::black_box(warm_plan);
        warm_plan_ns.push(started.elapsed().as_nanos());
        let calls = provider.calls.get();
        if let Some(expected) = warm_plan_calls {
            assert_eq!(calls, expected);
        } else {
            warm_plan_calls = Some(calls);
        }
    }
    warm_plan_ns.sort_unstable();
    println!(
        "case={case} phase=plan_build_including_repeated_preflight_repeat samples_ns={warm_plan_ns:?} median_ns={} calls={:?}",
        warm_plan_ns[warm_plan_ns.len() / 2],
        warm_plan_calls.unwrap()
    );
    let replay = measured_provider_phase(provider.as_ref(), case, "structure_compile", || {
        plan.compile_structures(prepared.structure(), source.space().structure())
            .unwrap()
    });
    println!(
        "case={case} structure_retained_payload_bytes={} excludes_dependent_layouts=true excludes_runtime_entry=true",
        replay.charged_payload_bytes()
    );

    let destination_structure = Arc::new(prepared.structure().clone());
    let mut destination_data = vec![0.0; prepared.required_len()];
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::<f64>::default();
    backend
        .tree_transform_structure_into_raw(
            &mut workspace,
            &replay,
            &destination_structure,
            source.space().structure(),
            &mut destination_data,
            &source_data,
            1.0,
            0.0,
        )
        .unwrap();
    destination_data.fill(0.0);
    let mut replay_profile = TreeTransformReplayProfile::default();
    measured_provider_phase(provider.as_ref(), case, "replay_preallocated", || {
        backend
            .tree_transform_structure_into_raw_profiled(
                &mut workspace,
                &replay,
                &destination_structure,
                source.space().structure(),
                &mut destination_data,
                &source_data,
                1.0,
                0.0,
                &mut replay_profile,
            )
            .unwrap()
    });
    println!(
        "case={case} replay_profile_ns={} workspace_source_len_after_warmup={} workspace_destination_len_after_warmup={} workspace_logical_bytes_after_warmup={}",
        replay_profile.total.as_nanos(),
        workspace.source_len(),
        workspace.destination_len(),
        (workspace.source_len() + workspace.destination_len()) * core::mem::size_of::<f64>()
    );

    let committed = measured_provider_phase(provider.as_ref(), case, "commit", || {
        source
            .commit_final_homspace_generic_bound_checked(prepared)
            .unwrap()
    });
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    crate::probe::take_completed_transformer_activity();
    let (cold_space, cold_data) =
        measured_provider_phase(provider.as_ref(), case, "runtime_cold_seed", || {
            context
                .tree_transform_owned_checked_generic_in(
                    &source,
                    None,
                    &source_data,
                    &operation,
                    1.0,
                )
                .unwrap()
        });
    let cold_info = crate::probe::take_completed_transformer_activity();
    let mut samples_ns = Vec::with_capacity(7);
    let mut warm_calls = None;
    let mut warm_output = None;
    for _ in 0..7 {
        provider.reset_calls();
        let started = std::time::Instant::now();
        let output = context
            .tree_transform_owned_checked_generic_in(&source, None, &source_data, &operation, 1.0)
            .unwrap();
        samples_ns.push(started.elapsed().as_nanos());
        let calls = provider.calls.get();
        assert!(warm_calls.is_none_or(|expected| expected == calls));
        warm_calls = Some(calls);
        warm_output = Some(output);
    }
    samples_ns.sort_unstable();
    let warm_info = crate::probe::take_completed_transformer_activity();
    println!(
        "case={case} phase=runtime_warm_hit samples_ns={samples_ns:?} median_ns={} calls={:?} cold={cold_info:?} warm={warm_info:?}",
        samples_ns[samples_ns.len() / 2], warm_calls.unwrap()
    );
    let (warm_space, warm_data) = warm_output.unwrap();
    assert_eq!(cold_info.builds + cold_info.hits, 1);
    assert_eq!((warm_info.hits, warm_info.builds), (7, 0));
    assert_eq!(committed.space(), cold_space.space());
    assert_eq!(cold_space.space(), warm_space.space());
    for (actual, expected) in destination_data.iter().zip(&cold_data) {
        assert!((actual - expected).abs() <= 1e-12);
    }
    for (actual, expected) in cold_data.iter().zip(&warm_data) {
        assert!((actual - expected).abs() <= 1e-12);
    }
    assert!(Arc::ptr_eq(warm_space.provider_arc(), &provider));
}

/// Measurement only: direct private-phase timings are intentionally kept in
/// the unit-test crate rather than exposed as production APIs. Allocation
/// counts for these private phases are unavailable under `forbid(unsafe_code)`;
/// the companion integration measurement covers public owned calls with its
/// existing counting allocator.
#[cfg(feature = "cgc-gen")]
#[test]
#[ignore = "benchmark: run via benchmarks.yml"]
fn measure_checked_generic_transform_phases() {
    println!("call_order=channels,dual,n,sqrt_dim,inv_sqrt_dim,frobenius_schur,f,r,dim");
    println!("spy_instrumented_phase_timings_auxiliary_only=true");
    for (n, adjoint_labels) in [(3, &[1, 1][..]), (4, &[1, 0, 1][..])] {
        for (operation_name, operation) in [
            ("permute", TreeTransformOperation::permute([1, 0], [2])),
            (
                "braid",
                TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
            ),
            (
                "repartition",
                TreeTransformOperation::transpose([0], [2, 1]),
            ),
        ] {
            let case = format!("su{n}_{operation_name}");
            measure_checked_generic_transform_case(n, adjoint_labels, &case, operation);
        }
    }
}

// Multiplicity-free styles cannot enter the generic path: the compile-time
// guarantee is trait disjointness (`GenericRigidSymbols` vs
// `MultiplicityFreeRigidSymbols`); this is its runtime sibling — the generic
// builder rejects a rule that reports a multiplicity-free style, mirroring the
// multiplicity-free builders' `UnsupportedFusionStyle` guards.
#[test]
fn generic_builder_rejects_multiplicity_free_style() {
    let mf = ToyGenericRule {
        style: FusionStyleKind::Simple,
    };
    let structure = packed_fixture_structure(
        2,
        [(BlockKey::from(vacuum_rank2_toy_src_pair()), vec![1, 1])],
    )
    .unwrap();
    let err = build_generic_tree_pair_transform_group_plan(
        &mf,
        TreeTransformOperation::permute([1, 0], []),
        &structure,
    )
    .unwrap_err();
    assert!(matches!(err, OperationError::UnsupportedFusionStyle { .. }));
}
