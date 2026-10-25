//! `RuleIdentity` construction and the equalities that must (not) hold.
//!
//! Already pinned elsewhere:
//! - SU(2) canonical bytes: `src/su2.rs` `identity_and_label_boundary_are_unchanged`.
//! - SU(N) rank distinction and shared allocation: `src/lie.rs`
//!   `stored_identity_is_shared_and_distinguishes_rank` (the public
//!   `rule_identity()` value itself is pinned below).
//! - Every shipped table entry distinct, stable, and unequal to Z2 / Z_2:
//!   `tests/table_rule.rs` `every_shipped_entry_constructs_with_its_own_identity`.
//! - Product identity keeps child order, codec and stateful children:
//!   `tests/rule_identity.rs`.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use symtn_sectors::*;

use super::table_identity;

#[test]
fn stateless_rules_are_identified_by_their_type() {
    assert_eq!(
        Z2FusionRule.rule_identity(),
        RuleIdentity::of_type::<Z2FusionRule>()
    );
    assert_eq!(
        FermionParityFusionRule.rule_identity(),
        RuleIdentity::of_type::<FermionParityFusionRule>()
    );
    assert_eq!(
        U1FusionRule.rule_identity(),
        RuleIdentity::of_type::<U1FusionRule>()
    );
    assert_eq!(
        CU1FusionRule.rule_identity(),
        RuleIdentity::of_type::<CU1FusionRule>()
    );
}

#[test]
fn zn_identity_is_the_little_endian_modulus_under_its_type() {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    5u32.hash(&mut hasher);
    assert_eq!(
        ZNFusionRule::new(5).unwrap().rule_identity(),
        RuleIdentity::from_canonical_bytes::<ZNFusionRule>(
            hasher.finish(),
            Arc::from([5u8, 0, 0, 0].as_slice())
        )
    );
}

#[test]
fn equal_groups_under_different_rules_stay_distinct() {
    let z2 = Z2FusionRule.rule_identity();
    let fz2 = FermionParityFusionRule.rule_identity();
    let zn2 = ZNFusionRule::new(2).unwrap().rule_identity();
    assert_ne!(z2, fz2);
    assert_ne!(z2, zn2);
    assert_ne!(fz2, zn2);
    assert_eq!(zn2, ZNFusionRule::new(2).unwrap().rule_identity());
    assert_ne!(zn2, ZNFusionRule::new(3).unwrap().rule_identity());
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
    let z2_u1 =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(Z2FusionRule, U1FusionRule).unwrap();
    // The factor identities are the independent type-tag literals; the
    // composition itself has no byte form, so this pins the construction
    // rule (codec tag over ordered children), not a serialized value.
    assert_eq!(
        z2_u1.rule_identity(),
        RuleIdentity::compose_with_codec::<TensorKitProductCodec>(
            RuleIdentity::of_type::<Z2FusionRule>(),
            RuleIdentity::of_type::<U1FusionRule>()
        )
    );
    assert_eq!(z2_u1.typed_rule_identity(), z2_u1.rule_identity());
    assert_eq!(
        z2_u1.rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(Z2FusionRule, U1FusionRule)
            .unwrap()
            .rule_identity()
    );

    // Factor order.
    assert_ne!(
        z2_u1.rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, Z2FusionRule)
            .unwrap()
            .rule_identity()
    );
    // Factor rule, not only factor group.
    assert_ne!(
        z2_u1.rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap()
            .rule_identity()
    );
    // Codec.
    let packed = ProductFusionRule::<
        Z2FusionRule,
        U1FusionRule,
        PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
    >::try_new(Z2FusionRule, U1FusionRule)
    .unwrap();
    assert_ne!(z2_u1.rule_identity(), packed.rule_identity());
    // Association.
    assert_ne!(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(Z2FusionRule, U1FusionRule).unwrap(),
            SU2FusionRule
        )
        .unwrap()
        .rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            Z2FusionRule,
            symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap()
        )
        .unwrap()
        .rule_identity()
    );
    // Stateful factors.
    assert_ne!(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            ZNFusionRule::new(3).unwrap(),
            Z2FusionRule
        )
        .unwrap()
        .rule_identity(),
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            ZNFusionRule::new(4).unwrap(),
            Z2FusionRule
        )
        .unwrap()
        .rule_identity()
    );
}

#[test]
fn table_identity_is_the_fingerprint_under_the_table_tag() {
    assert_eq!(
        TableRule::fibonacci().rule_identity(),
        table_identity(&fusion_symbols::table::TableCategory::fibonacci())
    );
    assert_eq!(
        TableRule::ising().rule_identity(),
        table_identity(&fusion_symbols::table::TableCategory::ising())
    );
}

/// SU(n) identity oracle: explicit A_(n-1), simply-connected metadata,
/// 64-bit graded Dynkin codec conventions and literal provider fingerprints.
#[cfg(all(feature = "cgc-gen", target_pointer_width = "64"))]
fn lie_group_identity(n: usize, fingerprint: &[u8]) -> RuleIdentity {
    let mut bytes = b"symtn:lie:dynkin:graded-total-then-lex:v1".to_vec();
    bytes.extend_from_slice(&64u32.to_le_bytes());
    bytes.push(0); // A root system.
    bytes.extend_from_slice(&((n - 1) as u64).to_le_bytes());
    bytes.push(0); // Simply connected.
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(b":generic:bosonic:rank-one-max=u32::MAX");
    bytes.extend_from_slice(
        b":rigid=f64-finite-exact-dim:f-axes=mu-nu-kappa-lambda:r-axes=mu-nu:pivotal=f-sign-0000",
    );
    bytes.extend_from_slice(fingerprint);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (
        0x4c49_455f_434f_4445_u64,
        fusion_symbols::lie::GroupId::su(n).unwrap(),
    )
        .hash(&mut hasher);
    RuleIdentity::from_canonical_bytes::<LieGroupFusionRule>(hasher.finish(), Arc::from(bytes))
}

#[cfg(all(feature = "cgc-gen", target_pointer_width = "64"))]
#[test]
fn lie_group_public_identity_has_explicit_canonical_group_metadata() {
    let a1 = b"racah:su2-exact:model=bigrational-round-once:3j=condon-shortley:cg=condon-shortley:6j=racah-single-sum:f=tks-su2irrep:r=tks-su2irrep:fs=tks-su2irrep:epoch=1";
    let higher_a = b"racah:sun-gt:ref=sunrep-0.4:basis=gt-order:gauge=qrpos-cref:descent=ladder-lstsq:tol=sunrep-tol-tier:epoch=1";
    assert_eq!(
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap())
            .unwrap()
            .rule_identity(),
        lie_group_identity(2, a1)
    );
    assert_eq!(
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
            .unwrap()
            .rule_identity(),
        lie_group_identity(3, higher_a)
    );
    assert_ne!(
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
            .unwrap()
            .rule_identity(),
        lie_group_identity(3, a1)
    );
}
