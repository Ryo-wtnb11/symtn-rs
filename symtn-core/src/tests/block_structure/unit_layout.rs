use super::*;

#[derive(Clone, Copy, Debug)]
struct UnitLayoutGenericRule;

impl symtn_sectors::testing::Provider for UnitLayoutGenericRule {}

impl FusionRule for UnitLayoutGenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, 0) => [SectorId::new(0)].into_iter().collect(),
            (0, 1) | (1, 0) => [SectorId::new(1)].into_iter().collect(),
            (1, 1) => [SectorId::new(0)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        match (left.id(), right.id(), coupled.id()) {
            (1, 1, 0) => 2,
            (0, a, b) | (a, 0, b) if a == b && a < 2 => 1,
            _ => 0,
        }
    }
}

impl CanonicalUnitFusionRule for UnitLayoutGenericRule {}

impl CheckedFusionAlgebra for UnitLayoutGenericRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        (sector.id() < 2)
            .then_some(sector)
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.try_dual_sector(left)?;
        self.try_dual_sector(right)?;
        Ok(self.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.try_dual_sector(left)?;
        self.try_dual_sector(right)?;
        self.try_dual_sector(coupled)?;
        Ok(self.nsymbol(left, right, coupled))
    }
}

#[test]
fn unit_layout_correspondence_preserves_generic_tree_and_storage_order() {
    // What: canonical-unit insertion preserves every nonunit tree label and
    // every existing layout coordinate, including Generic multiplicity.
    let rule = UnitLayoutGenericRule;
    let one = SectorId::new(1);
    let vacuum = rule.vacuum();
    let two = MultiplicityIndex::new(2).unwrap();
    let small_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(vacuum, 2), (one, 2)], false),
            SectorLeg::new([(vacuum, 3), (one, 3)], false),
            SectorLeg::new([(vacuum, 4), (one, 4)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 5), (one, 5)], false)]),
    );
    for (position, shape, strides) in [
        (0, vec![1, 2, 3, 4, 5], vec![99, 1, 2, 6, 24]),
        (1, vec![2, 1, 3, 4, 5], vec![1, 99, 2, 6, 24]),
        (2, vec![2, 3, 1, 4, 5], vec![1, 2, 99, 6, 24]),
        (3, vec![2, 3, 4, 1, 5], vec![1, 2, 6, 99, 24]),
    ] {
        let large_homspace = small_homspace
            .insert_right_unit(&rule, position, true)
            .unwrap();
        let mut small_specs = Vec::new();
        let mut large_specs = Vec::new();
        for (offset, (sector, vertex)) in [(0, (one, two)), (120, (vacuum, MultiplicityIndex::ONE))]
        {
            let small_key = FusionTreePairKey::pair(
                FusionTreeKey::new(
                    [sector, sector, sector],
                    sector,
                    [false, false, false],
                    [vacuum],
                    [vertex, MultiplicityIndex::ONE],
                ),
                FusionTreeKey::new([sector], sector, [false], [], []),
            );
            let (uncoupled, duals, innerlines, vertices) = match position {
                0 => (
                    vec![vacuum, sector, sector, sector],
                    vec![true, false, false, false],
                    vec![sector, vacuum],
                    vec![MultiplicityIndex::ONE, vertex, MultiplicityIndex::ONE],
                ),
                1 => (
                    vec![sector, vacuum, sector, sector],
                    vec![false, true, false, false],
                    vec![sector, vacuum],
                    vec![MultiplicityIndex::ONE, vertex, MultiplicityIndex::ONE],
                ),
                2 => (
                    vec![sector, sector, vacuum, sector],
                    vec![false, false, true, false],
                    vec![vacuum, vacuum],
                    vec![vertex, MultiplicityIndex::ONE, MultiplicityIndex::ONE],
                ),
                3 => (
                    vec![sector, sector, sector, vacuum],
                    vec![false, false, false, true],
                    vec![vacuum, sector],
                    vec![vertex, MultiplicityIndex::ONE, MultiplicityIndex::ONE],
                ),
                _ => unreachable!(),
            };
            let large_key = FusionTreePairKey::pair(
                FusionTreeKey::new(uncoupled, sector, duals, innerlines, vertices),
                FusionTreeKey::new([sector], sector, [false], [], []),
            );
            small_specs.push(
                BlockSpec::with_key(
                    BlockKey::FusionTree(small_key),
                    vec![2, 3, 4, 5],
                    vec![1, 2, 6, 24],
                    offset,
                )
                .unwrap(),
            );
            large_specs.push(
                BlockSpec::with_key(
                    BlockKey::FusionTree(large_key),
                    shape.clone(),
                    strides.clone(),
                    offset,
                )
                .unwrap(),
            );
        }
        let small = BlockStructure::from_blocks(small_specs).unwrap();
        let large = BlockStructure::from_blocks(large_specs).unwrap();
        let insertion = UnitLegInsertion::Right {
            position,
            dual: true,
        };
        validate_unit_layout_correspondence(
            &rule,
            (&small_homspace, &small),
            (&large_homspace, &large),
            insertion,
        )
        .unwrap();
        validate_unit_layout_correspondence_checked(
            &rule,
            (&small_homspace, &small),
            (&large_homspace, &large),
            insertion,
        )
        .unwrap();
        assert_eq!(
            large_homspace.remove_unit(&rule, position).unwrap(),
            small_homspace
        );
        assert_eq!(
            validate_unit_layout_correspondence(
                &rule,
                (&small_homspace, &small),
                (&large_homspace, &large),
                UnitLegInsertion::Right {
                    position,
                    dual: false
                },
            ),
            Err(CoreError::UnitLayoutCorrespondence)
        );
        let mut swapped = vec![large.block(1).unwrap(), large.block(0).unwrap()];
        let swapped = BlockStructure::from_blocks(
            swapped
                .drain(..)
                .map(|block| {
                    BlockSpec::with_key(
                        block.key().clone(),
                        block.shape().to_vec(),
                        block.strides().to_vec(),
                        block.offset(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(
            validate_unit_layout_correspondence(
                &rule,
                (&small_homspace, &small),
                (&large_homspace, &swapped),
                insertion
            ),
            Err(CoreError::UnitLayoutCorrespondence)
        );
    }
}

#[test]
fn unit_layout_validator_handles_local_rank_zero_and_one() {
    // What: tree rank is local to the modified side, including the two
    // TensorKit seams where total HomSpace rank does not decide the case.
    let rule = U1FusionRule;
    let vacuum = rule.vacuum();
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let scalar_structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new([], vacuum, [], [], []),
            FusionTreeKey::new([], vacuum, [], [], []),
        )),
        vec![],
        vec![],
        0,
    )
    .unwrap()])
    .unwrap();
    for (insertion, larger_homspace, key) in [
        (
            UnitLegInsertion::Left {
                position: 0,
                dual: true,
            },
            scalar.insert_left_unit(&rule, 0, true).unwrap(),
            FusionTreePairKey::pair(
                FusionTreeKey::new([], vacuum, [], [], []),
                FusionTreeKey::new([vacuum], vacuum, [true], [], []),
            ),
        ),
        (
            UnitLegInsertion::Right {
                position: 0,
                dual: true,
            },
            scalar.insert_right_unit(&rule, 0, true).unwrap(),
            FusionTreePairKey::pair(
                FusionTreeKey::new([vacuum], vacuum, [true], [], []),
                FusionTreeKey::new([], vacuum, [], [], []),
            ),
        ),
    ] {
        let larger = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::FusionTree(key),
            vec![1],
            vec![7],
            0,
        )
        .unwrap()])
        .unwrap();
        validate_unit_layout_correspondence(
            &rule,
            (&scalar, &scalar_structure),
            (&larger_homspace, &larger),
            insertion,
        )
        .unwrap();
    }

    let rank_one = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([]),
    );
    let small = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new([vacuum], vacuum, [false], [], []),
            FusionTreeKey::new([], vacuum, [], [], []),
        )),
        vec![1],
        vec![1],
        0,
    )
    .unwrap()])
    .unwrap();
    for (position, insertion) in [
        (
            0,
            UnitLegInsertion::Right {
                position: 0,
                dual: true,
            },
        ),
        (
            1,
            UnitLegInsertion::Right {
                position: 1,
                dual: true,
            },
        ),
    ] {
        let larger_homspace = rank_one.insert_right_unit(&rule, position, true).unwrap();
        let mut duals = vec![false];
        duals.insert(position, true);
        let mut strides = vec![1];
        strides.insert(position, 9);
        let large = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(
                FusionTreeKey::new(
                    [vacuum, vacuum],
                    vacuum,
                    duals,
                    [],
                    [MultiplicityIndex::ONE],
                ),
                FusionTreeKey::new([], vacuum, [], [], []),
            )),
            vec![1, 1],
            strides,
            0,
        )
        .unwrap()])
        .unwrap();
        validate_unit_layout_correspondence(
            &rule,
            (&rank_one, &small),
            (&larger_homspace, &large),
            insertion,
        )
        .unwrap();
    }

    let left_larger = rank_one.insert_left_unit(&rule, 0, true).unwrap();
    let left_large = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new(
                [vacuum, vacuum],
                vacuum,
                [true, false],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new([], vacuum, [], [], []),
        )),
        vec![1, 1],
        vec![9, 1],
        0,
    )
    .unwrap()])
    .unwrap();
    validate_unit_layout_correspondence(
        &rule,
        (&rank_one, &small),
        (&left_larger, &left_large),
        UnitLegInsertion::Left {
            position: 0,
            dual: true,
        },
    )
    .unwrap();

    let domain_small_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 1)], false)]),
    );
    let domain_small = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new([vacuum], vacuum, [false], [], []),
            FusionTreeKey::new([vacuum], vacuum, [false], [], []),
        )),
        vec![1, 1],
        vec![1, 1],
        0,
    )
    .unwrap()])
    .unwrap();
    let domain_larger = domain_small_homspace
        .insert_right_unit(&rule, 2, true)
        .unwrap();
    let domain_large = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new([vacuum], vacuum, [false], [], []),
            FusionTreeKey::new(
                [vacuum, vacuum],
                vacuum,
                [false, true],
                [],
                [MultiplicityIndex::ONE],
            ),
        )),
        vec![1, 1, 1],
        vec![1, 1, 9],
        0,
    )
    .unwrap()])
    .unwrap();
    validate_unit_layout_correspondence(
        &rule,
        (&domain_small_homspace, &domain_small),
        (&domain_larger, &domain_large),
        UnitLegInsertion::Right {
            position: 2,
            dual: true,
        },
    )
    .unwrap();
}

#[test]
fn checked_unit_layout_validation_reports_algebra_before_correspondence() {
    // What: checked finite algebra rejects an invalid supplied sector before
    // inspecting the deliberately mismatched unit descriptor.
    let rule = UnitLayoutGenericRule;
    let invalid = SectorId::new(9);
    let vacuum = rule.vacuum();
    let small_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(invalid, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(invalid, 1)], false)]),
    );
    let larger_homspace = small_homspace.insert_right_unit(&rule, 0, true).unwrap();
    let small = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new([invalid], invalid, [false], [], []),
            FusionTreeKey::new([invalid], invalid, [false], [], []),
        )),
        vec![1, 1],
        vec![1, 1],
        0,
    )
    .unwrap()])
    .unwrap();
    let large = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(
            FusionTreeKey::new(
                [vacuum, invalid],
                invalid,
                [true, false],
                [],
                [MultiplicityIndex::ONE],
            ),
            FusionTreeKey::new([invalid], invalid, [false], [], []),
        )),
        vec![1, 1, 1],
        vec![9, 1, 1],
        0,
    )
    .unwrap()])
    .unwrap();
    assert!(matches!(
        validate_unit_layout_correspondence_checked(
            &rule,
            (&small_homspace, &small),
            (&larger_homspace, &large),
            UnitLegInsertion::Right { position: 0, dual: false },
        ),
        Err(CheckedFusionSpaceError::FusionAlgebra(error))
            if *error == FusionAlgebraError::InvalidSector { sector: invalid }
    ));
}
