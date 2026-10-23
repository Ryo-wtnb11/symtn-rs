use std::collections::BTreeSet;

use symtn_sectors::{
    product_sector, BraidingStyleKind, CU1FusionRule, CU1Irrep, CheckedFusionAlgebra,
    FermionParityFusionRule, FusionAlgebraError, FusionRule, FusionStyleKind, Fz2SectorLayout,
    PackedProductCodec, ProductFusionRule, ProductSector, ProductSectorCodecError, RuleIdentity,
    SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SectorVec, TableRule, TableSector,
    U1FusionRule, U1Irrep, U1SectorLayout, Z2FusionRule, Z2Irrep, CU1_MAX_TWICE_CHARGE,
    SU2_MAX_DOUBLED_SPIN,
};

#[test]
fn product_sector_is_usable_as_a_btree_key() {
    // What: the label order is the left component first, the right component
    // only as a tie-break, and equal labels deduplicate in ordered containers.
    let labels: BTreeSet<ProductSector<U1Irrep, Z2Irrep>> = [
        product_sector(U1Irrep::new(1), Z2Irrep::ODD),
        product_sector(U1Irrep::new(0), Z2Irrep::ODD),
        product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
        product_sector(U1Irrep::new(1), Z2Irrep::EVEN),
        product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        labels.into_iter().collect::<Vec<_>>(),
        vec![
            product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
            product_sector(U1Irrep::new(0), Z2Irrep::ODD),
            product_sector(U1Irrep::new(1), Z2Irrep::EVEN),
            product_sector(U1Irrep::new(1), Z2Irrep::ODD),
        ]
    );
}

// ---------------------------------------------------------------------------
// `SectorCodec`: the typed facade's label <-> `SectorId` boundary.
// ---------------------------------------------------------------------------

/// Asserts all three codec laws over `labels`: label round trip, id round
/// trip, and injectivity of the encoding.
fn assert_round_trip<Rule>(rule: &Rule, labels: impl IntoIterator<Item = Rule::Sector>)
where
    Rule: SectorCodec,
{
    let mut ids = BTreeSet::new();
    let mut count = 0;
    for label in labels {
        count += 1;
        let id = rule.encode_sector(&label).expect("label is representable");
        let decoded = SectorCodec::decode_sector(rule, id).expect("id is representable");
        assert_eq!(decoded, label);
        assert_eq!(
            rule.encode_sector(&decoded)
                .expect("label is representable"),
            id
        );
        ids.insert(id);
    }
    assert_eq!(ids.len(), count, "distinct labels aliased onto one id");
}

/// Asserts the decode-totality law: `vacuum` decodes, and so does every id
/// the provider generates from decodable ids through duals and fusion, closed
/// under `rounds` fusion rounds starting from `seeds`.
///
/// Why a closure rather than a fixed sample: the law is about ids the engine
/// itself produces, so the only faithful test generates them the way the
/// engine does.
fn assert_decode_total<Rule>(
    rule: &Rule,
    seeds: impl IntoIterator<Item = Rule::Sector>,
    rounds: usize,
) where
    Rule: SectorCodec + CheckedFusionAlgebra,
{
    let mut reachable: BTreeSet<SectorId> = seeds
        .into_iter()
        .map(|label| rule.encode_sector(&label).expect("seed label encodes"))
        .collect();
    reachable.insert(rule.vacuum());

    // A boundary seed's dual or fusion is often genuinely unrepresentable
    // (U(1) charge overflow, SU(2) past the maximum spin). That is the encode
    // side of the contract, which the law explicitly permits, so such an id is
    // never generated and never has to decode. Only what the rule does hand
    // back is held to decode totality.
    let mut generated = 0usize;
    for _ in 0..rounds {
        let mut next = reachable.clone();
        for &left in &reachable {
            if let Ok(dual) = rule.try_dual_sector(left) {
                generated += 1;
                next.insert(dual);
            }
            for &right in &reachable {
                if let Ok(channels) = rule.try_fusion_channels(left, right) {
                    generated += channels.len();
                    next.extend(channels);
                }
            }
        }
        reachable = next;
    }
    assert!(generated > 0, "the rule generated no ids at all");

    // Why not assert the closure grew: a finite closed algebra (Z2, fZ2)
    // reaches its whole sector set from the vacuum, which is the law fully
    // discharged, not a vacuous test.
    for id in reachable {
        assert!(
            SectorCodec::decode_sector(rule, id).is_ok(),
            "id {id:?} is reachable from the provider's own algebra but does not decode"
        );
    }
}

#[test]
fn builtin_codecs_decode_every_id_their_own_algebra_reaches() {
    // What: each built-in codec's decode domain covers the vacuum and the
    // dual/fusion closure of its decodable ids, so the typed facade can label
    // any block the engine builds.
    // Seeds sit both in the interior and hard against each codec's advertised
    // representable maximum, so a decode domain truncated anywhere below that
    // maximum fails here instead of slipping past an easy sample.
    assert_decode_total(
        &U1FusionRule,
        [-2, 3, i32::MIN + 1, i32::MAX - 1].map(U1Irrep::new),
        2,
    );
    assert_decode_total(&Z2FusionRule, [Z2Irrep::ODD], 2);
    assert_decode_total(&FermionParityFusionRule, [Z2Irrep::ODD], 2);
    assert_decode_total(&TableRule::fibonacci(), [TableSector::new(2)], 2);
    assert_decode_total(&TableRule::ising(), [TableSector::new(3)], 2);
    assert_decode_total(
        &SU2FusionRule,
        [1, 2, SU2_MAX_DOUBLED_SPIN - 1, SU2_MAX_DOUBLED_SPIN].map(SU2Irrep::from_twice_spin),
        2,
    );
    assert_decode_total(
        &CU1FusionRule,
        [
            CU1Irrep::VACUUM,
            CU1Irrep::PSEUDOSCALAR,
            CU1Irrep::from_twice_charge(1),
            CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE),
        ],
        2,
    );
    assert_decode_total(
        &U1Fz2Rule::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
        [
            product_sector(U1Irrep::new(1), Z2Irrep::ODD),
            product_sector(U1Irrep::new(-2), Z2Irrep::EVEN),
            product_sector(U1Irrep::new(i32::MAX - 1), Z2Irrep::ODD),
            product_sector(U1Irrep::new(i32::MIN + 1), Z2Irrep::EVEN),
        ],
        2,
    );
    // Why no `TinyRule` here: it certifies only `FusionRule` + `SectorCodec`,
    // and the closure must be generated through the checked primitives. The
    // external-provider side of this law is covered by the typed facade tests,
    // whose providers do certify `CheckedFusionAlgebra`.
}

#[test]
fn cu1_codec_and_checked_overflow_preserve_the_tensor_kit_boundary() {
    let rule = CU1FusionRule;
    assert_round_trip(
        &rule,
        [
            CU1Irrep::VACUUM,
            CU1Irrep::PSEUDOSCALAR,
            CU1Irrep::from_twice_charge(1),
            CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE),
        ],
    );
    assert!(CU1Irrep::try_from_twice_charge(0).is_none());
    assert!(CU1Irrep::try_from_twice_charge(u32::MAX).is_none());
    let edge: SectorId = CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE).into();
    assert_eq!(
        rule.try_fusion_channels(edge, edge),
        Err(FusionAlgebraError::FusionNotRepresentable {
            left: edge,
            right: edge
        })
    );
    #[cfg(target_pointer_width = "64")]
    assert!(matches!(
        rule.decode_sector(SectorId::new(u32::MAX as usize + 1)),
        Err(FusionAlgebraError::InvalidSector { .. })
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn cu1_checked_validation_preserves_the_shared_error_precedence() {
    // What: checked fusion validates left then right; checked nsymbol first
    // validates the requested coupled sector, then left then right. This
    // mirrors the SU(2) checked-provider contract and makes diagnostics
    // deterministic when more than one input is invalid.
    let rule = CU1FusionRule;
    let invalid = SectorId::new(u32::MAX as usize + 1);
    let later_invalid = SectorId::new(u32::MAX as usize + 2);
    let op: SectorId = CU1Irrep::VACUUM.into();
    assert_eq!(
        rule.try_fusion_channels(invalid, later_invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
    assert_eq!(
        rule.try_nsymbol(invalid, op, later_invalid),
        Err(FusionAlgebraError::InvalidSector {
            sector: later_invalid
        })
    );
    assert_eq!(
        rule.try_nsymbol(op, invalid, op),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
}

#[test]
fn u1_codec_round_trips_every_boundary_charge() {
    // What: the label domain keeps every i32 charge except the one whose dual
    // has no i32 representation.
    assert_round_trip(
        &U1FusionRule,
        [-7, -1, 0, 1, 9, i32::MIN + 1, i32::MAX].map(U1Irrep::new),
    );
}

// Only a 64-bit target has ids above the u32 zigzag domain to reject.
#[cfg(target_pointer_width = "64")]
#[test]
fn u1_codec_rejects_ids_outside_the_zigzag_domain() {
    // What: the decode side reports the offending id, not a wrapped charge.
    let sector = SectorId::new(1usize << 32);
    assert_eq!(
        SectorCodec::decode_sector(&U1FusionRule, sector),
        Err(FusionAlgebraError::InvalidSector { sector })
    );
}

#[test]
fn parity_codecs_round_trip_and_reject_non_parity_ids() {
    // What: both Z2-labelled rules share the parity label domain and both
    // reject an id above it.
    assert_round_trip(&Z2FusionRule, [Z2Irrep::EVEN, Z2Irrep::ODD]);
    assert_round_trip(&FermionParityFusionRule, [Z2Irrep::EVEN, Z2Irrep::ODD]);

    let sector = SectorId::new(2);
    assert_eq!(
        SectorCodec::decode_sector(&Z2FusionRule, sector),
        Err(FusionAlgebraError::InvalidSector { sector })
    );
    assert_eq!(
        SectorCodec::decode_sector(&FermionParityFusionRule, sector),
        Err(FusionAlgebraError::InvalidSector { sector })
    );
}

#[test]
fn su2_codec_round_trips_up_to_the_supported_doubled_spin() {
    // What: the label domain ends exactly at the provider's maximum 2j.
    assert_round_trip(
        &SU2FusionRule,
        [0, 1, 2, 5, SU2_MAX_DOUBLED_SPIN].map(SU2Irrep::from_twice_spin),
    );

    let sector = SectorId::new(SU2_MAX_DOUBLED_SPIN + 1);
    assert_eq!(
        SectorCodec::decode_sector(&SU2FusionRule, sector),
        Err(FusionAlgebraError::InvalidSector { sector })
    );
}

type U1Fz2Codec = PackedProductCodec<U1SectorLayout, Fz2SectorLayout>;
type U1Fz2Rule = ProductFusionRule<U1FusionRule, FermionParityFusionRule, U1Fz2Codec>;

#[test]
fn product_codec_round_trips_component_labels() {
    // What: the product label is decoded back component-wise, so the pair
    // survives the packed id, not just the id itself.
    let rule = U1Fz2Rule::try_new(U1FusionRule, FermionParityFusionRule).unwrap();
    assert_round_trip(
        &rule,
        [
            product_sector(U1Irrep::new(0), Z2Irrep::EVEN),
            product_sector(U1Irrep::new(-5), Z2Irrep::ODD),
            product_sector(U1Irrep::new(i32::MAX), Z2Irrep::ODD),
        ],
    );
}

// The 33-bit U(1) x fZ2 layout only fits, and only has spare high bits, on a
// 64-bit target.
#[cfg(target_pointer_width = "64")]
#[test]
fn product_codec_reports_the_packing_failure_it_hit() {
    // What: the packed layout's own diagnosis survives to the caller instead
    // of being flattened into a generic label rejection.
    let rule = U1Fz2Rule::try_new(U1FusionRule, FermionParityFusionRule).unwrap();
    let sector = SectorId::new(1usize << 33);
    assert_eq!(
        SectorCodec::decode_sector(&rule, sector),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: 33,
            }
        ))
    );
}

/// Bounded external provider: three labels, only the first three ids.
#[derive(Clone, Copy, Debug, Default)]
struct TinyRule;

impl symtn_sectors::testing::Provider for TinyRule {}

impl FusionRule for TinyRule {
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
        core::iter::once(SectorId::new((left.id() + right.id()) % 3)).collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TinyLabel(u8);

impl SectorCodec for TinyRule {
    type Sector = TinyLabel;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if value.0 < 3 {
            Ok(SectorId::new(usize::from(value.0)))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: format!("Z3 charge {}", value.0),
            })
        }
    }

    fn decode_sector(&self, sector: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        u8::try_from(sector.id())
            .ok()
            .filter(|&charge| charge < 3)
            .map(TinyLabel)
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }
}

#[test]
fn an_external_provider_implements_the_codec_without_builtin_machinery() {
    // What: `SectorCodec` is implementable outside symtn from the public
    // vocabulary alone, and both failure directions are expressible.
    assert_round_trip(&TinyRule, [TinyLabel(0), TinyLabel(1), TinyLabel(2)]);

    let rejected = TinyRule.encode_sector(&TinyLabel(7)).unwrap_err();
    assert_eq!(
        rejected,
        FusionAlgebraError::UnrepresentableSectorLabel {
            rule: RuleIdentity::of_type::<TinyRule>(),
            label: "Z3 charge 7".to_owned(),
        }
    );
    assert_eq!(
        rejected.to_string(),
        "sector label Z3 charge 7 is not representable by its fusion rule"
    );
    let sector = SectorId::new(3);
    assert_eq!(
        SectorCodec::decode_sector(&TinyRule, sector),
        Err(FusionAlgebraError::InvalidSector { sector })
    );
}
