//! A checked-generic SU(3) provider that fails a chosen decode, encode,
//! dual or symbol query, shared by the serial network tests and the crate's
//! unit tests.
#![allow(dead_code)]

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use symtn::sector::SectorId;
use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, LieGroupFusionRule,
    RuleIdentity, SectorVec, TypedSectorAdmission,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InjectedError {
    Decode,
    Encode,
    Dual,
    Symbol,
    Provider,
}

impl fmt::Display for InjectedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "injected {self:?} failure")
    }
}

impl std::error::Error for InjectedError {}

pub(crate) struct InjectedGeneric {
    pub(crate) inner: LieGroupFusionRule,
    pub(crate) fail_encode: AtomicBool,
    pub(crate) fail_decode_at: AtomicUsize,
    pub(crate) decode_calls: AtomicUsize,
    pub(crate) fail_dual: AtomicBool,
    pub(crate) dual_calls: AtomicUsize,
    pub(crate) fail_symbol_at: AtomicUsize,
    pub(crate) symbol_calls: AtomicUsize,
    /// Reports a Fermionic braiding in place of the inner rule's.
    pub(crate) fermionic: AtomicBool,
}

impl InjectedGeneric {
    pub(crate) fn new() -> Self {
        Self {
            inner: symtn_core::testing::su_lie_rule(3),
            fail_encode: AtomicBool::new(false),
            fail_decode_at: AtomicUsize::new(0),
            decode_calls: AtomicUsize::new(0),
            fail_dual: AtomicBool::new(false),
            dual_calls: AtomicUsize::new(0),
            fail_symbol_at: AtomicUsize::new(0),
            symbol_calls: AtomicUsize::new(0),
            fermionic: AtomicBool::new(false),
        }
    }

    pub(crate) fn reset_symbols(&self) {
        self.fail_symbol_at.store(0, Ordering::SeqCst);
        self.symbol_calls.store(0, Ordering::SeqCst);
    }

    pub(crate) fn arm_symbol(&self, ordinal: usize) {
        self.symbol_calls.store(0, Ordering::SeqCst);
        self.fail_symbol_at.store(ordinal, Ordering::SeqCst);
    }

    pub(crate) fn symbol(&self) -> Result<(), InjectedError> {
        let ordinal = self.symbol_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_symbol_at.load(Ordering::SeqCst) == ordinal {
            Err(InjectedError::Symbol)
        } else {
            Ok(())
        }
    }

    pub(crate) fn reset_decodes(&self) {
        self.fail_decode_at.store(0, Ordering::SeqCst);
        self.decode_calls.store(0, Ordering::SeqCst);
    }

    pub(crate) fn arm_decode(&self, ordinal: usize) {
        self.decode_calls.store(0, Ordering::SeqCst);
        self.fail_decode_at.store(ordinal, Ordering::SeqCst);
    }
}

impl symtn_core::testing::Provider for InjectedGeneric {}

impl CheckedGenericFusion for InjectedGeneric {
    type Error = InjectedError;

    fn rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(&self.inner)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(&self.inner)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        if self.fermionic.load(Ordering::SeqCst) {
            BraidingStyleKind::Fermionic
        } else {
            CheckedGenericFusion::braiding_style(&self.inner)
        }
    }

    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.dual_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_dual.load(Ordering::SeqCst) {
            Err(InjectedError::Dual)
        } else {
            CheckedGenericFusion::try_dual(&self.inner, sector).map_err(|_| InjectedError::Provider)
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_fusion_channels(&self.inner, left, right)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_fusion_channels_in_table(&self.inner, left, right)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_nsymbol(&self.inner, left, right, coupled)
            .map_err(|_| InjectedError::Provider)
    }
}

impl CheckedGenericRigidSymbols for InjectedGeneric {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_sqrt_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
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
        self.symbol()?;
        CheckedGenericRigidSymbols::try_f_symbol_generic(&self.inner, a, b, c, d, e, f)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_r_symbol_generic(&self.inner, a, b, c)
            .map_err(|_| InjectedError::Provider)
    }
}

impl CheckedGenericPivotal for InjectedGeneric {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericPivotal::try_twist_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }
}

impl TypedSectorAdmission for InjectedGeneric {
    type Sector = symtn_core::LieIrrep;
    type Error = InjectedError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        if self.fail_encode.load(Ordering::SeqCst) {
            return Err(InjectedError::Encode);
        }
        TypedSectorAdmission::try_encode_label(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        let ordinal = self.decode_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_decode_at.load(Ordering::SeqCst) == ordinal {
            return Err(InjectedError::Decode);
        }
        TypedSectorAdmission::try_decode_label(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.try_dual(sector)
    }
}
