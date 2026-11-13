//! `RuleIdentity` construction and the equalities that must (not) hold.
//!
//! Already pinned elsewhere:
//! - Lie identity follows `GroupId` equality (SU(3) = SU(3)/Z1 != PSU(3)):
//!   `src/lie.rs` `validation_and_identity_are_typed_and_stable`; rank
//!   distinction: `stored_identity_is_shared_and_distinguishes_rank`.
//! - Every shipped table entry distinct, stable, and unequal to Z2 / Z_2:
//!   `tests/table_rule.rs` `every_shipped_entry_constructs_with_its_own_identity`.
//! - Product identity keeps child order, codec and stateful children:
//!   `tests/rule_identity.rs`.

use symtn_sectors::*;

use super::table_identity;

#[test]
fn stateless_rules_are_identified_by_their_type() {
    assert_eq!(
        U1FusionRule.rule_identity(),
        RuleIdentity::of_type::<U1FusionRule>()
    );
    assert_eq!(
        SU2FusionRule.rule_identity(),
        RuleIdentity::of_type::<SU2FusionRule>()
    );
}

#[test]
fn enumerated_rules_are_identified_by_their_category_value() {
    let z2 = ZNFusionRule::new(2).unwrap();
    let fz2 = FermionParityFusionRule::new();
    let cu1 = CU1FusionRule::new();
    assert_eq!(
        z2.rule_identity(),
        ZNFusionRule::new(2).unwrap().rule_identity()
    );
    assert_ne!(
        z2.rule_identity(),
        ZNFusionRule::new(3).unwrap().rule_identity()
    );
    assert_eq!(
        fz2.rule_identity(),
        FermionParityFusionRule::new().rule_identity()
    );
    assert_eq!(cu1.rule_identity(), CU1FusionRule::new().rule_identity());
    // Equal groups under different categories stay distinct.
    assert_ne!(z2.rule_identity(), fz2.rule_identity());
    assert_ne!(z2.rule_identity(), U1FusionRule.rule_identity());
}

#[test]
fn table_identities_follow_the_table() {
    assert_ne!(
        TableRule::fibonacci().rule_identity(),
        TableRule::ising().rule_identity()
    );
    assert_eq!(
        TableRule::fibonacci().rule_identity(),
        TableRule::fibonacci().rule_identity()
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn lie_group_su2_and_compact_su2_are_different_rules() {
    let sun2 = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap();
    assert_ne!(sun2.rule_identity(), SU2FusionRule.rule_identity());
    assert_eq!(
        sun2.rule_identity(),
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap())
            .unwrap()
            .rule_identity()
    );
    assert_ne!(
        sun2.rule_identity(),
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
            .unwrap()
            .rule_identity()
    );
    assert_eq!(sun2.typed_rule_identity(), sun2.rule_identity());
}

#[test]
fn product_identity_composes_factor_identities_under_the_codec() {
    let z2 = ZNFusionRule::new(2).unwrap();
    let z2_u1 =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(z2.clone(), U1FusionRule).unwrap();
    // The composition itself has no byte form, so this pins the
    // construction rule (codec tag over ordered children), not a serialized
    // value.
    assert_eq!(
        z2_u1.rule_identity(),
        RuleIdentity::compose_with_codec::<TensorKitProductCodec>(
            z2.rule_identity(),
            RuleIdentity::of_type::<U1FusionRule>()
        )
    );
    assert_eq!(z2_u1.typed_rule_identity(), z2_u1.rule_identity());
    assert_eq!(
        z2_u1.rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            ZNFusionRule::new(2).unwrap(),
            U1FusionRule
        )
        .unwrap()
        .rule_identity()
    );

    // Factor order.
    assert_ne!(
        z2_u1.rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            U1FusionRule,
            ZNFusionRule::new(2).unwrap()
        )
        .unwrap()
        .rule_identity()
    );
    // Factor rule, not only factor group.
    assert_ne!(
        z2_u1.rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            FermionParityFusionRule::new(),
            U1FusionRule
        )
        .unwrap()
        .rule_identity()
    );
    // Codec.
    let packed = ProductFusionRule::<
        ZNFusionRule,
        U1FusionRule,
        PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
    >::try_new(ZNFusionRule::new(2).unwrap(), U1FusionRule)
    .unwrap();
    assert_ne!(z2_u1.rule_identity(), packed.rule_identity());
    // Association.
    assert_ne!(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                ZNFusionRule::new(2).unwrap(),
                U1FusionRule
            )
            .unwrap(),
            SU2FusionRule
        )
        .unwrap()
        .rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            ZNFusionRule::new(2).unwrap(),
            symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap()
        )
        .unwrap()
        .rule_identity()
    );
    // Stateful factors.
    assert_ne!(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            ZNFusionRule::new(3).unwrap(),
            ZNFusionRule::new(2).unwrap()
        )
        .unwrap()
        .rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            ZNFusionRule::new(4).unwrap(),
            ZNFusionRule::new(2).unwrap()
        )
        .unwrap()
        .rule_identity()
    );
}

#[test]
fn table_identity_is_the_uuid_under_the_table_tag() {
    assert_eq!(
        TableRule::fibonacci().rule_identity(),
        table_identity(&fusion_symbols::table::TableCategory::fibonacci())
    );
    assert_eq!(
        TableRule::ising().rule_identity(),
        table_identity(&fusion_symbols::table::TableCategory::ising())
    );
}

/// The identity holds its group weakly, as a `CategoryRule` identity holds its
/// category: sharing lasts while a clone of the rule is alive.
#[cfg(feature = "cgc-gen")]
#[test]
fn lie_group_identity_shares_only_while_a_rule_is_alive() {
    let su3 = || LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let rule = su3();
    let clone = rule.clone();
    let identity = rule.rule_identity();
    drop(rule);
    assert_eq!(identity, su3().rule_identity());
    drop(clone);
    assert_ne!(identity, su3().rule_identity());
    assert_eq!(identity.clone(), identity);
}
