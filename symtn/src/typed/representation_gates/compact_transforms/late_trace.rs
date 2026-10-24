use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use symtn_core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, LieGroupFusionRule,
    LieGroupFusionRuleError, RuleIdentity, SectorId, SectorVec, TypedSectorAdmission,
};

// Negative fixture: real SU(3) structural answers, with a typed error only at
// the pivotal query. A distinct identity prevents hits under the real provider.
struct RejectingTraceTwist {
    inner: LieGroupFusionRule,
    twist_calls: AtomicUsize,
}

impl symtn_core::testing::Provider for RejectingTraceTwist {}

impl CheckedGenericFusion for RejectingTraceTwist {
    type Error = LieGroupFusionRuleError;
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
        self.inner.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.inner.try_dual(sector)
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        self.inner.try_fusion_channels(a, b)
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.inner.try_fusion_channels_in_table(a, b)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        self.inner.try_nsymbol(a, b, c)
    }
}

impl CheckedGenericRigidSymbols for RejectingTraceTwist {
    type Scalar = f64;
    fn try_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.inner.try_dim_scalar(sector)
    }
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.inner.try_sqrt_dim_scalar(sector)
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.inner.try_inv_sqrt_dim_scalar(sector)
    }
    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
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
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.inner.try_f_symbol_generic(a, b, c, d, e, f)
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.inner.try_r_symbol_generic(a, b, c)
    }
}

impl CheckedGenericPivotal for RejectingTraceTwist {
    fn try_twist_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
        self.twist_calls.fetch_add(1, Ordering::Relaxed);
        // Sentinel capability is unavailable only in this synthetic fixture.
        Err(LieGroupFusionRuleError::Provider(
            fusion_symbols::lie::LieError::Unsupported {
                requires: "synthetic trace twist",
            },
        ))
    }
}

impl TypedSectorAdmission for RejectingTraceTwist {
    type Sector = Vec<i64>;
    type Error = LieGroupFusionRuleError;
    type Mode = CheckedGenericAdmissionMode;
    fn typed_rule_identity(&self) -> RuleIdentity {
        self.rule_identity()
    }
    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.inner.encode_dynkin(sector)
    }
    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.inner.decode_dynkin(sector)
    }
    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.inner.try_dual(sector)
    }
}

fn assert_compact_late_rejection<D>()
where
    D: TensorScalar + symtn_tensors::RecouplingCoefficientAction<f64>,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(RejectingTraceTwist {
        inner: LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        twist_calls: AtomicUsize::new(0),
    });
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![D::from_real(1.0), D::from_real(2.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![D::from_real(3.0), D::from_real(4.0), D::from_real(5.0)],
            },
        ],
    )
    .unwrap();
    assert_eq!(provider.twist_calls.load(Ordering::Relaxed), 0);
    let densified = probe::record();
    let result = diagonal.trace_pairs(&[(0, 1)]);
    assert!(matches!(
        result,
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            LieGroupFusionRuleError::Provider(fusion_symbols::lie::LieError::Unsupported {
                requires: "synthetic trace twist"
            })
        )))
    ));
    assert_eq!(provider.twist_calls.load(Ordering::Relaxed), 1);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert!(matches!(
        owned(&diagonal).data.as_ref(),
        TypedData::Diagonal(_)
    ));
}

#[test]
fn checked_compact_late_pivotal_failure_never_materializes_real_or_complex_payload() {
    assert_compact_late_rejection::<f64>();
    assert_compact_late_rejection::<Complex64>();
}
