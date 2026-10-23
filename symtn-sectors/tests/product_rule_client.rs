//! Downstream view of the product API: public bounds, nesting, and execution.
#![cfg(feature = "cgc-gen")]
use symtn_sectors::{CategoryProvider, ProductAdmission, ProductFusionRule};
use symtn_sectors::{CheckedGenericFusion, FusionRule, SUNFusionRule, U1FusionRule};

fn channels_mf<T: FusionRule>(rule: &T) -> usize {
    rule.fusion_channels(rule.vacuum(), rule.vacuum()).len()
}
fn channels_cg<T: CheckedGenericFusion>(rule: &T) -> usize {
    rule.try_fusion_channels(rule.vacuum(), rule.vacuum())
        .map(|c| c.len())
        .unwrap_or(0)
}
// A downstream generic over any product repeats only public bounds.
fn generic_product<L, R>(rule: &ProductFusionRule<L, R>) -> usize
where
    L: CategoryProvider,
    R: CategoryProvider,
    (L::Mode, R::Mode): ProductAdmission<L::Error, R::Error>,
    ProductFusionRule<L, R>: CheckedGenericFusion,
{
    channels_cg(rule)
}

#[test]
fn product_type_checks() {
    let u1u1 = ProductFusionRule::<_, _>::try_new(U1FusionRule, U1FusionRule).unwrap();
    channels_mf(&u1u1);
    let nested = ProductFusionRule::<_, _>::try_new(u1u1, SUNFusionRule::new(3).unwrap()).unwrap();
    channels_cg(&nested);
    generic_product(&nested);
}

#[test]
fn product_runs_through_fs_product() {
    use symtn_sectors::ProductFusionRuleError;
    let rule =
        ProductFusionRule::<_, _>::try_new(U1FusionRule, SUNFusionRule::new(3).unwrap()).unwrap();
    let vacuum = rule.vacuum();
    assert_eq!(
        rule.try_fusion_channels(vacuum, vacuum).unwrap().as_slice(),
        &[vacuum]
    );
    assert_eq!(rule.try_nsymbol(vacuum, vacuum, vacuum).unwrap(), 1);
    // An extreme id reports a typed failure instead of panicking.
    let bad = symtn_sectors::SectorId::new(usize::MAX);
    let error = rule.try_dual(bad).unwrap_err();
    assert!(
        matches!(
            error,
            ProductFusionRuleError::Right(_) | ProductFusionRuleError::Codec(_)
        ),
        "{error:?}"
    );
}

#[test]
fn product_orders_products_as_tensorkit() {
    use symtn_sectors::{ProductSector, TypedSectorAdmission, U1Irrep};
    let rule = ProductFusionRule::<_, _>::try_new(U1FusionRule, U1FusionRule).unwrap();
    let id = |a, b| {
        rule.try_encode_label(&ProductSector::new(U1Irrep::new(a), U1Irrep::new(b)))
            .unwrap()
    };
    // TensorKit U1 order: 0, +1, -1, ...; the product compares factor by factor.
    assert!(
        FusionRule::sector_order_key(&rule, id(1, 0))
            < FusionRule::sector_order_key(&rule, id(-1, 0))
    );
    assert!(
        FusionRule::sector_order_key(&rule, id(0, 0))
            < FusionRule::sector_order_key(&rule, id(1, 0))
    );
}
