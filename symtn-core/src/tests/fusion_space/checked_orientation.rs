use super::*;

#[test]
fn canonical_coupled_storage_is_admitted_by_witness_across_symmetries() {
    // What: symtn-built coupled-sector destinations are admitted as
    // injective without visiting elements for Abelian, non-Abelian, and
    // fermionic product sectors with dual legs and nontrivial
    // degeneracies, while the exact check agrees on the same geometry.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(-1), 2), (u1(0), 1), (u1(1), 3)], false),
            SectorLeg::new([(u1(-1), 2), (u1(1), 2)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 2), (u1(1), 1), (u1(-1), 2)], false),
            SectorLeg::new([(u1(1), 3), (u1(-1), 1)], true),
        ]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap(),
    );

    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(su2(1), 2), (su2(3), 1)], false),
            SectorLeg::new([(su2(0), 1), (su2(2), 2)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(su2(1), 3), (su2(2), 1)], true),
            SectorLeg::new([(su2(0), 2), (su2(1), 2)], false),
        ]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap(),
    );

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = Fz2U1::new(FermionParityFusionRule, U1FusionRule);
    let even0 = rule.encode_component_ids(z2_even(), u1(0));
    let odd_p = rule.encode_component_ids(z2_odd(), u1(1));
    let odd_m = rule.encode_component_ids(z2_odd(), u1(-1));
    let leg = |dual| SectorLeg::new([(even0, 2), (odd_p, 1), (odd_m, 3)], dual);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap(),
    );

    let (s0, s1) = (SectorId::new(0), SectorId::new(1));
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(s0, 2), (s1, 1)], false),
            SectorLeg::new([(s0, 1), (s1, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(s0, 2), (s1, 2)], false),
            SectorLeg::new([(s0, 3), (s1, 1)], false),
        ]),
    );
    assert_canonical_storage_admitted_without_enumeration(
        &hom.coupled_subblock_structure_from_leg_degeneracies_generic(&IsomorphismMultiplicityRule)
            .unwrap(),
    );
}

#[test]
fn canonical_and_factor_bridge_storage_skip_exact_enumeration() {
    // What: hom-space-generated coupled storage and its typed factor-style
    // shared bridge are admitted by structural proof, not element scans.
    let rule = SU2FusionRule;
    let homspace = FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);
    let recording = crate::probe::record();
    let canonical = homspace
        .clone()
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();
    assert_eq!(recording.get(crate::probe::Event::ExactStorageFallback), 0);

    // The expert layout constructor skips the exact check exactly when the
    // structure compiles to coupled-sector regions.
    let recording = crate::probe::record();
    assert!(canonical.coupled_sector_regions(1).unwrap().is_some());
    homspace
        .validate_subblock_structure_subset(&rule, &canonical)
        .unwrap();
    assert_eq!(recording.get(crate::probe::Event::ExactStorageFallback), 0);
}

#[test]
fn checked_select_and_permute_reject_a_malformed_id_when_orientation_needs_it() {
    // What: moving an excluded raw U1 ID across the HomSpace boundary
    // reports it, while same-side selection and identity permutation do not
    // inspect sectors they do not dualize.
    let min_leg = SectorLeg::new([(excluded_u1_id(), 2)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg]),
        FusionProductSpace::new([]),
    );
    let rule = DualCountingRule::new(U1FusionRule);

    let same_side = hom.try_select_checked(&rule, &[0], &[]).unwrap();
    assert_eq!(same_side, hom);
    assert_eq!(rule.dual_calls(), 0);
    let identity = hom.try_permute_checked(&rule, &[0], &[]).unwrap();
    assert_eq!(identity, hom);
    assert_eq!(rule.dual_calls(), 0);

    assert_eq!(
        hom.try_select_checked(&rule, &[], &[0]),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 1);

    #[cfg(target_pointer_width = "64")]
    {
        type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
        let product_rule = Rule::new(U1FusionRule, Z2FusionRule);
        let product_min = TensorKitProductCodec::encode(excluded_u1_id(), z2_odd());
        let product_hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(product_min, 1)], false)]),
            FusionProductSpace::new([]),
        );
        assert_eq!(
            product_hom.try_permute_checked(&product_rule, &[], &[0]),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::InvalidSector {
                    sector: excluded_u1_id()
                }
            )))
        );
    }
}

#[test]
fn checked_orientation_validates_axes_before_sector_ids() {
    // What: malformed axis requests retain Core validation precedence even
    // when the first legal orientation mapping would inspect an invalid ID.
    let min_leg = SectorLeg::new([(excluded_u1_id(), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg]),
        FusionProductSpace::new([]),
    );
    let rule = DualCountingRule::new(U1FusionRule);
    assert_eq!(
        hom.try_select_checked(&rule, &[], &[1]),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::InvalidPermutation {
                permutation: vec![1],
                rank: 1,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);
    assert_eq!(
        hom.try_permute_checked(&rule, &[], &[]),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::InvalidPermutation {
                permutation: vec![],
                rank: 1,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);

    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &hom,
            &scalar,
            &[],
            &[],
            &[1],
            0,
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::InvalidPermutation {
                permutation: vec![1],
                rank: 1,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);
}

#[test]
fn checked_tensorcontract_separates_pair_validation_and_output_orientation_failures() {
    // What: structural pair mismatches retain Core precedence without
    // inspecting sectors, while an excluded raw ID moved to the output
    // domain reports the exact algebra failure.
    let min_leg = || SectorLeg::new([(excluded_u1_id(), 1)], false);
    let scalar =
        || FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg()]),
        FusionProductSpace::new([]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([min_leg()]),
        FusionProductSpace::new([]),
    );
    let rule = DualCountingRule::new(U1FusionRule);
    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &lhs,
            &rhs,
            &[0],
            &[0],
            &[],
            0,
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::ContractedLegDualityMismatch {
                lhs_axis: 0,
                rhs_axis: 0,
                lhs_is_dual: false,
                rhs_is_dual: false,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);

    let rhs_count_mismatch = FusionTreeHomSpace::new(
        FusionProductSpace::new([]),
        FusionProductSpace::new([SectorLeg::new([(excluded_u1_id(), 1), (u1(0), 1)], false)]),
    );
    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &lhs,
            &rhs_count_mismatch,
            &[0],
            &[0],
            &[],
            0,
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::DimensionMismatch {
                expected: 1,
                actual: 2,
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 0);

    assert_eq!(
        FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            &rule,
            &lhs,
            &scalar(),
            &[],
            &[],
            &[0],
            0,
        ),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::InvalidSector {
                sector: excluded_u1_id()
            }
        )))
    );
    assert_eq!(rule.dual_calls(), 1);
}

#[test]
fn checked_homspace_derivation_keeps_semantic_identity_lazy() {
    // What: successful checked metadata derivation does not publish or
    // resolve a process-local HomSpace identity until a caller asks for it.
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-2), 1), (u1(1), 2)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg]),
        FusionProductSpace::new([]),
    );
    let selected = hom.try_select_checked(&rule, &[1, 0], &[]).unwrap();
    let permuted = hom.try_permute_checked(&rule, &[1, 0], &[]).unwrap();
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let contracted = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
        &rule,
        &hom,
        &scalar,
        &[],
        &[],
        &[1, 0],
        2,
    )
    .unwrap();
    for derived in [&hom, &scalar, &selected, &permuted, &contracted] {
        assert!(derived.existing_id().is_none());
    }
    assert_eq!(selected, permuted);
    assert_eq!(contracted, permuted);
}

#[test]
fn unit_homspace_geometry_keeps_tensor_kit_seams_and_stored_duality() {
    // What: left/right placement, including the seam, preserves stored
    // domain duality while its external-axis view remains complemented.
    let rule = U1FusionRule;
    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(-3), 1)], false)]),
    );
    for dual in [false, true] {
        for (left, position, codomain_duals, domain_duals) in [
            (true, 0, vec![dual, false], vec![false]),
            (true, 1, vec![false], vec![dual, false]),
            (true, 2, vec![false], vec![false, dual]),
            (false, 0, vec![dual, false], vec![false]),
            (false, 1, vec![false, dual], vec![false]),
            (false, 2, vec![false], vec![false, dual]),
        ] {
            let inserted = if left {
                base.insert_left_unit(&rule, position, dual)
            } else {
                base.insert_right_unit(&rule, position, dual)
            }
            .unwrap();
            assert_eq!(
                inserted
                    .codomain()
                    .legs()
                    .iter()
                    .map(SectorLeg::is_dual)
                    .collect::<Vec<_>>(),
                codomain_duals
            );
            assert_eq!(
                inserted
                    .domain()
                    .legs()
                    .iter()
                    .map(SectorLeg::is_dual)
                    .collect::<Vec<_>>(),
                domain_duals
            );
            let axis = if left || position != 1 { position } else { 1 };
            assert_eq!(
                inserted.external_axis_is_dual(axis),
                Some(if axis < inserted.codomain().len() {
                    dual
                } else {
                    !dual
                })
            );
            assert_eq!(inserted.remove_unit(&rule, axis).unwrap(), base);
        }
    }
    assert!(base.remove_unit(&rule, 0).is_err());

    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    assert_eq!(
        scalar
            .insert_left_unit(&rule, 0, true)
            .unwrap()
            .domain()
            .len(),
        1
    );
    assert_eq!(
        scalar
            .insert_right_unit(&rule, 0, true)
            .unwrap()
            .codomain()
            .len(),
        1
    );
}

#[derive(Clone, Copy)]
enum CheckedFailStage {
    Dual,
    Channels,
    Nsymbol,
}

// Trivial external multiplicity-free algebra (every product fuses to the
// vacuum) that can be told to fail exactly one checked primitive, so the
// checked enumeration transaction can be probed one fallible stage at a time.
#[derive(Clone, Copy)]
struct CheckedFailRule {
    fail: CheckedFailStage,
}

impl symtn_sectors::testing::Provider for CheckedFailRule {}

impl FusionRule for CheckedFailRule {
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
    fn fusion_channels(&self, _left: SectorId, _right: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl MultiplicityFreeFusionRule for CheckedFailRule {}

impl CheckedFusionAlgebra for CheckedFailRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if matches!(self.fail, CheckedFailStage::Dual) {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        if matches!(self.fail, CheckedFailStage::Channels) {
            return Err(FusionAlgebraError::FusionNotRepresentable { left, right });
        }
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        if matches!(self.fail, CheckedFailStage::Nsymbol) {
            return Err(FusionAlgebraError::MultiplicityOverflow {
                left,
                right,
                coupled,
            });
        }
        Ok(self.nsymbol(left, right, coupled))
    }
}

#[test]
fn checked_prepare_surfaces_each_stage_failure_without_publishing() {
    // What: an injected failure in dual, channels, or nsymbol returns the
    // exact typed error and leaves the layout cache statistics untouched
    // (no layout id issued, no admission).
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], []);

    for (stage, expected) in [
        (
            CheckedFailStage::Dual,
            FusionAlgebraError::InvalidSector {
                sector: SectorId::new(0),
            },
        ),
        (
            CheckedFailStage::Channels,
            FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(0),
                right: SectorId::new(1),
            },
        ),
        (
            CheckedFailStage::Nsymbol,
            FusionAlgebraError::MultiplicityOverflow {
                left: SectorId::new(0),
                right: SectorId::new(1),
                coupled: SectorId::new(0),
            },
        ),
    ] {
        clear_structure_caches();
        let recording = crate::probe::record();
        let error = homspace
            .prepare_fusion_tree_layout_checked(&CheckedFailRule { fail: stage })
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(
            (
                recording.get(crate::probe::Event::LayoutBuild),
                recording.get(crate::probe::Event::LayoutAdmission)
            ),
            (0, 0)
        );
    }
}
