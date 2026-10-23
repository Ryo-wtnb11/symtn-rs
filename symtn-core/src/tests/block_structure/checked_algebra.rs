use super::*;

#[test]
fn checked_u1_rejects_the_unlabelled_id_and_preserves_valid_boundaries() {
    // What: the excluded zigzag ID is typed, while boundary sums that
    // remain representable are identical to the expert infallible path.
    let rule = U1FusionRule;
    assert_eq!(
        rule.try_dual_sector(excluded_u1_id()),
        Err(FusionAlgebraError::InvalidSector {
            sector: excluded_u1_id()
        })
    );
    for (left, right) in [(i32::MAX, 1), (i32::MIN + 1, -1)] {
        assert_eq!(
            rule.try_fusion_channels(u1(left), u1(right)),
            Err(FusionAlgebraError::U1FusionOverflow { left, right })
        );
    }
    for (left, right, expected) in [
        (i32::MAX, 0, i32::MAX),
        (i32::MIN + 1, 0, i32::MIN + 1),
        (i32::MAX, i32::MIN + 1, 0),
    ] {
        let checked = rule.try_fusion_channels(u1(left), u1(right)).unwrap();
        assert_eq!(checked.as_slice(), &[u1(expected)]);
        assert_eq!(checked, rule.fusion_channels(u1(left), u1(right)));
        assert_eq!(
            rule.try_nsymbol(u1(left), u1(right), u1(expected)),
            Ok(rule.nsymbol(u1(left), u1(right), u1(expected)))
        );
    }
    assert_eq!(
        rule.try_dual_sector(u1(i32::MAX)).unwrap(),
        rule.dual(u1(i32::MAX))
    );
}

#[test]
fn checked_su2_distinguishes_invalid_inputs_from_unrepresentable_fusion() {
    // What: valid SU2 inputs whose output exceeds the supported algebra
    // report closure failure, while the exact boundary matches the hot path.
    let rule = SU2FusionRule;
    let boundary = rule.try_fusion_channels(su2(127), su2(127)).unwrap();
    assert_eq!(boundary, rule.fusion_channels(su2(127), su2(127)));
    assert_eq!(
        rule.try_fusion_channels(su2(128), su2(127)),
        Err(FusionAlgebraError::FusionNotRepresentable {
            left: su2(128),
            right: su2(127),
        })
    );
    assert_eq!(
        rule.try_fusion_channels(SectorId::new(255), su2(0)),
        Err(FusionAlgebraError::InvalidSector {
            sector: SectorId::new(255),
        })
    );
}

#[test]
fn checked_fibonacci_matches_valid_operations_and_rejects_unknown_sectors() {
    // What: Fibonacci's checked companion preserves every valid operation
    // and rejects IDs outside the two-sector algebra with the exact input.
    let rule = TableRule::fibonacci();
    let vacuum = SectorId::new(0);
    let tau = SectorId::new(1);
    assert_eq!(rule.try_dual_sector(tau), Ok(rule.dual(tau)));
    assert_eq!(
        rule.try_fusion_channels(tau, tau),
        Ok(rule.fusion_channels(tau, tau))
    );
    for coupled in [vacuum, tau] {
        assert_eq!(
            rule.try_nsymbol(tau, tau, coupled),
            Ok(rule.nsymbol(tau, tau, coupled))
        );
    }
    let invalid = SectorId::new(2);
    assert_eq!(
        rule.try_dual_sector(invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
    assert_eq!(
        rule.try_fusion_channels(tau, invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
    assert_eq!(
        rule.try_nsymbol(tau, tau, invalid),
        Err(FusionAlgebraError::InvalidSector { sector: invalid })
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn checked_products_preserve_child_u1_errors_and_distinguish_codec_errors() {
    // What: recursive products retain the exact U1 closure cause, while
    // malformed packed IDs remain a distinct codec failure.
    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type TripleLayout = ProductSectorLayout<Fz2U1Layout, Su2SectorLayout>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

    let pair = Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let pair_min = pair
        .try_encode_component_ids(z2_odd(), excluded_u1_id())
        .unwrap();
    assert_eq!(
        pair.try_dual_sector(pair_min),
        Err(FusionAlgebraError::InvalidSector {
            sector: excluded_u1_id()
        })
    );
    let pair_max = pair
        .try_encode_component_ids(z2_even(), u1(i32::MAX))
        .unwrap();
    let pair_one = pair.try_encode_component_ids(z2_odd(), u1(1)).unwrap();
    assert_eq!(
        pair.try_fusion_channels(pair_max, pair_one),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        })
    );

    let triple = TripleRule::try_new(pair, SU2FusionRule).unwrap();
    let triple_min = triple.try_encode_component_ids(pair_min, su2(1)).unwrap();
    assert_eq!(
        triple.try_dual_sector(triple_min),
        Err(FusionAlgebraError::InvalidSector {
            sector: excluded_u1_id()
        })
    );
    let invalid = SectorId::new(1usize << TripleLayout::BITS);
    assert!(matches!(
        triple.try_dual_sector(invalid),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::InvalidHighBits { .. }
        ))
    ));
}

#[test]
fn checked_fusion_algebra_is_object_safe_and_matches_closed_builtins() {
    // What: callers can use checked algebra through one provider object,
    // and closed built-ins retain their infallible results exactly.
    let checked: &dyn CheckedFusionAlgebra = &U1FusionRule;
    assert_eq!(checked.try_dual_sector(u1(7)), Ok(u1(-7)));
    for rule in [
        &Z2FusionRule as &dyn CheckedFusionAlgebra,
        &FermionParityFusionRule,
        &SU2FusionRule,
    ] {
        let left = rule.vacuum();
        let right = rule.vacuum();
        assert_eq!(rule.try_dual_sector(left), Ok(rule.dual(left)));
        assert_eq!(
            rule.try_fusion_channels(left, right),
            Ok(rule.fusion_channels(left, right))
        );
        assert_eq!(
            rule.try_nsymbol(left, right, rule.vacuum()),
            Ok(rule.nsymbol(left, right, rule.vacuum()))
        );
    }
}

#[test]
fn u1_trivial_a_b_symbols_accept_lowest_charge_valid_triples() {
    // What: trivial U1 rigidity symbols remain exactly one at the lowest
    // representable charge.
    let rule = U1FusionRule;
    assert_eq!(
        rule.a_symbol_scalar(u1(i32::MIN + 1), u1(0), u1(i32::MIN + 1)),
        1.0
    );
    assert_eq!(
        rule.b_symbol_scalar(u1(0), u1(i32::MIN + 1), u1(i32::MIN + 1)),
        1.0
    );
}
