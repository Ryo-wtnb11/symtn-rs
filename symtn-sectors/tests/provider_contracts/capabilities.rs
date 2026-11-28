//! Capability classification: styles, associator certificate, unit markers,
//! admission mode, and coefficient scalar.
//!
//! Already pinned elsewhere: table styles and the semion's uncertified
//! associator gauge (`tests/table_categories.rs`
//! `styles_follow_the_braiding_and_the_exact_fusion_table`); SU(3) styles
//! (`src/lie.rs` `su3_octet_self_fusion_has_dynkin_ordered_channels_and_multiplicities`).

use crate::su2::su2_rule;
use crate::tables::*;

use num_complex::Complex64;
use symtn_sectors::*;

use BraidingStyleKind::{Anyonic, Bosonic, Fermionic, NoBraiding};
use FusionStyleKind::{Simple, Unique};

fn styles<R: FusionRule>(rule: &R) -> (FusionStyleKind, BraidingStyleKind) {
    (rule.fusion_style(), rule.braiding_style())
}

#[test]
fn multiplicity_free_rule_styles() {
    let rep_s3 = table_by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c");
    assert_eq!(styles(&ZNFusionRule::new(2).unwrap()), (Unique, Bosonic));
    assert_eq!(styles(&ZNFusionRule::new(7).unwrap()), (Unique, Bosonic));
    assert_eq!(styles(&FermionParityFusionRule::new()), (Unique, Fermionic));
    assert_eq!(styles(&U1FusionRule), (Unique, Bosonic));
    assert_eq!(styles(&CU1FusionRule::new()), (Simple, Bosonic));
    assert_eq!(styles(&su2_rule()), (Simple, Bosonic));
    assert_eq!(styles(&table_fibonacci()), (Simple, Anyonic));
    assert_eq!(styles(&rep_s3), (Simple, NoBraiding));

    assert_eq!(
        styles(
            &symtn_sectors::ProductFusionRule::<_, _>::try_new(
                ZNFusionRule::new(2).unwrap(),
                U1FusionRule
            )
            .unwrap()
        ),
        (Unique, Bosonic)
    );
    assert_eq!(
        styles(
            &symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule::new(),
                su2_rule()
            )
            .unwrap()
        ),
        (Simple, Fermionic)
    );
    assert_eq!(
        styles(
            &symtn_sectors::ProductFusionRule::<_, _>::try_new(
                table_fibonacci(),
                FermionParityFusionRule::new()
            )
            .unwrap()
        ),
        (Simple, Anyonic)
    );
    assert_eq!(
        styles(
            &symtn_sectors::ProductFusionRule::<_, _>::try_new(
                rep_s3,
                ZNFusionRule::new(2).unwrap()
            )
            .unwrap()
        ),
        (Simple, NoBraiding)
    );
    assert_eq!(
        styles(
            &symtn_sectors::ProductFusionRule::<_, _>::try_new(
                symtn_sectors::ProductFusionRule::<_, _>::try_new(
                    U1FusionRule,
                    FermionParityFusionRule::new()
                )
                .unwrap(),
                ZNFusionRule::new(2).unwrap()
            )
            .unwrap()
        ),
        (Unique, Fermionic)
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn lie_group_reports_generic_bosonic_for_every_rank_including_two() {
    for n in [2, 3, 4] {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            fusion_symbols::lie::GroupId::su(n).unwrap(),
        )
        .unwrap();
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic, "SU({n})");
        assert_eq!(rule.braiding_style(), Bosonic, "SU({n})");
        assert_eq!(rule.vacuum(), SectorId::new(0));
    }
}

/// Multiplicity-free admission reports fusion-symbols' own style.
#[test]
fn multiplicity_free_lie_rule_styles() {
    use fusion_symbols::lie::GroupId;
    let mf = |group| LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(group).unwrap();
    assert_eq!(styles(&mf(GroupId::su(2).unwrap())), (Simple, Bosonic));
    assert_eq!(styles(&mf(GroupId::so(3).unwrap())), (Simple, Bosonic));
    assert_eq!(styles(&mf(GroupId::u(1).unwrap())), (Unique, Bosonic));
    assert_eq!(styles(&mf(GroupId::torus(2).unwrap())), (Unique, Bosonic));
    assert_eq!(styles(&mf(GroupId::u(2).unwrap())), (Simple, Bosonic));
    assert_eq!(styles(&mf(GroupId::spin(4).unwrap())), (Simple, Bosonic));
}

/// A group whose fusion has multiplicity has no multiplicity-free rule.
#[cfg(feature = "cgc-gen")]
#[test]
fn multiplicity_free_lie_rule_refuses_generic_fusion() {
    use fusion_symbols::{category::FusionStyle, lie::GroupId};
    for group in [GroupId::su(3).unwrap(), GroupId::so(5).unwrap()] {
        assert!(matches!(
            LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(group.clone()),
            Err(LieGroupFusionRuleError::IncompatibleFusionStyle {
                actual: FusionStyle::Generic
            })
        ));
        assert!(LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(group).is_ok());
    }
}

#[test]
fn trivial_associator_gauge_is_certified_only_by_the_abelian_group_rules() {
    assert!(ZNFusionRule::new(2).unwrap().has_trivial_associator_gauge());
    assert!(ZNFusionRule::new(7).unwrap().has_trivial_associator_gauge());
    assert!(FermionParityFusionRule::new().has_trivial_associator_gauge());
    assert!(U1FusionRule.has_trivial_associator_gauge());
    assert!(!CU1FusionRule::new().has_trivial_associator_gauge());
    assert!(!su2_rule().has_trivial_associator_gauge());
    assert!(!table_fibonacci().has_trivial_associator_gauge());
    // Products: conjunction of the factors.
    assert!(symtn_sectors::ProductFusionRule::<_, _>::try_new(
        ZNFusionRule::new(2).unwrap(),
        U1FusionRule
    )
    .unwrap()
    .has_trivial_associator_gauge());
    assert!(!symtn_sectors::ProductFusionRule::<_, _>::try_new(
        ZNFusionRule::new(2).unwrap(),
        su2_rule()
    )
    .unwrap()
    .has_trivial_associator_gauge());
    assert!(
        !symtn_sectors::ProductFusionRule::<_, _>::try_new(CU1FusionRule::new(), U1FusionRule)
            .unwrap()
            .has_trivial_associator_gauge()
    );
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
        ZNFusionRule::new(2).unwrap(),
        ZNFusionRule::new(3).unwrap(),
        FermionParityFusionRule::new(),
        U1FusionRule,
        CU1FusionRule::new(),
        su2_rule(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(fusion_symbols::lie::GroupId::u(2).unwrap()).unwrap()).unwrap(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(ZNFusionRule::new(2).unwrap(), su2_rule()).unwrap(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, ZNFusionRule::new(2).unwrap()).unwrap(), su2_rule()).unwrap(),
    );
    mf!(Complex64:
        table_fibonacci(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(table_fibonacci(), ZNFusionRule::new(2).unwrap()).unwrap(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(ZNFusionRule::new(2).unwrap(), table_ising()).unwrap(),
    );
    physical(&U1FusionRule);
    physical(&su2_rule());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn lie_group_carries_the_checked_generic_markers_for_every_rank() {
    fn checked_canonical_unit<R: CheckedCanonicalUnitFusionRule>(_: &R) {}
    fn generic_admission<R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>>(_: &R) {}
    fn checked_pivotal<R: CheckedGenericPivotal + CheckedGenericRigidSymbols<Scalar = f64>>(_: &R) {
    }
    for n in [2, 3] {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            fusion_symbols::lie::GroupId::su(n).unwrap(),
        )
        .unwrap();
        checked_canonical_unit(&rule);
        generic_admission(&rule);
        checked_pivotal(&rule);
    }
}
