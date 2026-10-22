//! Capability classification: styles, associator certificate, unit markers,
//! admission mode, and coefficient scalar.
//!
//! Already pinned elsewhere: table styles and the semion's uncertified
//! associator gauge (`tests/table_rule.rs`
//! `styles_follow_the_braiding_and_the_exact_fusion_table`); SU(3) styles
//! (`src/sun.rs` `su3_octet_self_fusion_has_dynkin_ordered_channels_and_multiplicities`).

use num_complex::Complex64;
use symtn_sectors::*;

use BraidingStyleKind::{Anyonic, Bosonic, Fermionic, NoBraiding};
use FusionStyleKind::{Simple, Unique};

fn styles<R: FusionRule>(rule: &R) -> (FusionStyleKind, BraidingStyleKind) {
    (rule.fusion_style(), rule.braiding_style())
}

#[test]
fn multiplicity_free_rule_styles() {
    let rep_s3 = TableRule::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c").unwrap();
    assert_eq!(styles(&Z2FusionRule), (Unique, Bosonic));
    assert_eq!(styles(&ZNFusionRule::new(7).unwrap()), (Unique, Bosonic));
    assert_eq!(styles(&FermionParityFusionRule), (Unique, Fermionic));
    assert_eq!(styles(&U1FusionRule), (Unique, Bosonic));
    assert_eq!(styles(&CU1FusionRule), (Simple, Bosonic));
    assert_eq!(styles(&SU2FusionRule), (Simple, Bosonic));
    assert_eq!(styles(&TableRule::fibonacci()), (Simple, Anyonic));
    assert_eq!(styles(&rep_s3), (Simple, NoBraiding));

    assert_eq!(
        styles(&Z2FusionRule.product(U1FusionRule)),
        (Unique, Bosonic)
    );
    assert_eq!(
        styles(&FermionParityFusionRule.product(SU2FusionRule)),
        (Simple, Fermionic)
    );
    assert_eq!(
        styles(&TableRule::fibonacci().product(FermionParityFusionRule)),
        (Simple, Anyonic)
    );
    assert_eq!(styles(&rep_s3.product(Z2FusionRule)), (Simple, NoBraiding));
    assert_eq!(
        styles(
            &U1FusionRule
                .product(FermionParityFusionRule)
                .product(Z2FusionRule)
        ),
        (Unique, Fermionic)
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_reports_generic_bosonic_for_every_rank_including_two() {
    for n in [2, 3, 4] {
        let rule = SUNFusionRule::new(n).unwrap();
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic, "SU({n})");
        assert_eq!(rule.braiding_style(), Bosonic, "SU({n})");
        assert_eq!(rule.vacuum(), SectorId::new(0));
    }
}

#[test]
fn trivial_associator_gauge_is_certified_only_by_the_abelian_group_rules() {
    assert!(Z2FusionRule.has_trivial_associator_gauge());
    assert!(ZNFusionRule::new(7).unwrap().has_trivial_associator_gauge());
    assert!(FermionParityFusionRule.has_trivial_associator_gauge());
    assert!(U1FusionRule.has_trivial_associator_gauge());
    assert!(!CU1FusionRule.has_trivial_associator_gauge());
    assert!(!SU2FusionRule.has_trivial_associator_gauge());
    assert!(!TableRule::fibonacci().has_trivial_associator_gauge());
    // Products: conjunction of the factors.
    assert!(Z2FusionRule
        .product(U1FusionRule)
        .has_trivial_associator_gauge());
    assert!(!Z2FusionRule
        .product(SU2FusionRule)
        .has_trivial_associator_gauge());
    assert!(!CU1FusionRule
        .product(U1FusionRule)
        .has_trivial_associator_gauge());
}

fn canonical_unit<R: CanonicalUnitFusionRule>(_: &R) {}
fn multiplicity_free<R: MultiplicityFreeRigidSymbols>(_: &R) {}
fn mf_admission<R: TypedSectorAdmission<Mode = MultiplicityFreeAdmissionMode>>(_: &R) {}
fn scalar<S, R: MultiplicityFreeFusionSymbols<Scalar = S>>(_: &R) {}
fn physical<R: PhysicalFusionBasis>(_: &R) {}

/// Compile-time capability matrix of the multiplicity-free providers.
#[test]
fn multiplicity_free_providers_carry_their_markers() {
    macro_rules! mf {
        ($scalar:ty: $($rule:expr),+ $(,)?) => {$(
            let rule = $rule;
            canonical_unit(&rule);
            multiplicity_free(&rule);
            mf_admission(&rule);
            scalar::<$scalar, _>(&rule);
        )+};
    }
    mf!(f64:
        Z2FusionRule,
        ZNFusionRule::new(3).unwrap(),
        FermionParityFusionRule,
        U1FusionRule,
        CU1FusionRule,
        SU2FusionRule,
        Z2FusionRule.product(SU2FusionRule),
        U1FusionRule.product(Z2FusionRule).product(SU2FusionRule),
    );
    mf!(Complex64:
        TableRule::fibonacci(),
        TableRule::fibonacci().product(Z2FusionRule),
        Z2FusionRule.product(TableRule::ising()),
    );
    physical(&U1FusionRule);
    physical(&SU2FusionRule);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_carries_the_checked_generic_markers_for_every_rank() {
    fn checked_canonical_unit<R: CheckedCanonicalUnitFusionRule>(_: &R) {}
    fn generic_admission<R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>>(_: &R) {}
    fn checked_pivotal<R: CheckedGenericPivotal + CheckedGenericRigidSymbols<Scalar = f64>>(_: &R) {
    }
    for n in [2, 3] {
        let rule = SUNFusionRule::new(n).unwrap();
        checked_canonical_unit(&rule);
        generic_admission(&rule);
        checked_pivotal(&rule);
    }
}
