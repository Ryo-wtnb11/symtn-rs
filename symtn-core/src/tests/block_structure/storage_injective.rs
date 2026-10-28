use super::*;

/// The storage half of expert-layout admission (the rank-free
/// `DynamicFusionMapSpace::try_from_layout` runs it for non-canonical layouts).
fn expert_layout_storage(rank: usize, blocks: Vec<BlockSpec>) -> Result<(), CoreError> {
    // Storage-only fixtures use opaque ordinals deliberately: categorical
    // key and shape admission is covered independently by subset validation.
    validate_block_storage_injective(&BlockStructure::from_blocks_with_rank(rank, blocks).unwrap())
}

#[test]
fn expert_fusion_space_rejects_self_overlapping_storage() {
    // What: an owning symmetric space cannot assign two logical elements of
    // one block to the same physical element.
    for (block, offset) in [
        (
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 1], 0).unwrap(),
            1,
        ),
        (
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 1], vec![0, 1], 0).unwrap(),
            0,
        ),
    ] {
        assert_eq!(
            expert_layout_storage(2, vec![block]),
            Err(CoreError::OverlappingBlockStorage {
                first_block: 0,
                second_block: 0,
                offset,
            })
        );
    }
}

#[test]
fn expert_fusion_space_rejects_cross_block_storage_aliases() {
    // What: distinct logical symmetric blocks cannot own the same physical
    // destination element, and diagnostics preserve caller block order.
    let blocks = vec![
        BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![2], 0).unwrap(),
        BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 1).unwrap(),
    ];
    assert_eq!(
        expert_layout_storage(1, blocks),
        Err(CoreError::OverlappingBlockStorage {
            first_block: 0,
            second_block: 1,
            offset: 2,
        })
    );
}

#[test]
fn expert_fusion_space_accepts_exact_non_overlapping_strided_storage() {
    // What: expert admission retains arbitrary legal layouts, including
    // layouts that a conservative sorted-span proof cannot establish.
    let rank_two_cases = [
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![3, 2], vec![2, 3], 0).unwrap()],
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 3], vec![3, 1], 0).unwrap()],
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![1, 2], vec![0, 1], 0).unwrap()],
    ];
    let recording = crate::probe::record();
    for blocks in rank_two_cases {
        expert_layout_storage(2, blocks).unwrap();
    }
    assert!(recording.get(crate::probe::Event::ExactStorageFallback) > 0);

    let rank_one_cases = [
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![4], vec![2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![4], vec![2], 1).unwrap(),
        ],
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 4).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
        ],
        vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![0], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![1], vec![1], 0).unwrap(),
        ],
    ];
    for blocks in rank_one_cases {
        expert_layout_storage(1, blocks).unwrap();
    }

    expert_layout_storage(
        0,
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![], vec![], 0).unwrap()],
    )
    .unwrap();
    expert_layout_storage(
        1,
        vec![BlockSpec::with_key(BlockKey::ordinal(0), vec![0], vec![1], 0).unwrap()],
    )
    .unwrap();
}

#[test]
fn general_block_structure_retains_aliasing_read_view_contract() {
    // What: storage ownership admission belongs to the expert fusion space;
    // general block metadata remains usable for intentionally aliased views.
    let structure = BlockStructure::from_blocks(vec![
        BlockSpec::with_key(BlockKey::ordinal(0), vec![2], vec![1], 0).unwrap(),
        BlockSpec::with_key(BlockKey::ordinal(1), vec![2], vec![1], 0).unwrap(),
    ])
    .unwrap();
    assert_eq!(structure.required_len().unwrap(), 2);
}

fn assert_expert_storage_admission_for_rule<R>(rule: &R, sector: SectorId)
where
    R: MultiplicityFreeFusionRule,
{
    let homspace = FusionTreeHomSpace::from_sectors([(sector, 2)], [(sector, 3)]);
    let key = homspace.fusion_tree_keys(rule)[0].clone();
    let structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(key),
        vec![2, 3],
        vec![1, 4],
        2,
    )
    .unwrap()])
    .unwrap();
    validate_block_storage_injective(&structure).unwrap();
    homspace
        .validate_subblock_structure_subset(rule, &structure)
        .unwrap();
}

#[test]
fn expert_storage_admission_is_symmetry_independent() {
    // What: the same exact custom-layout admission and categorical binding
    // succeeds for U(1), SU(2), fZ2, and their nested product.
    assert_expert_storage_admission_for_rule(&U1FusionRule, u1(0));
    assert_expert_storage_admission_for_rule(&SU2FusionRule, su2(0));
    assert_expert_storage_admission_for_rule(&FermionParityFusionRule, z2_even());

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let pair_vacuum = pair.encode_component_ids(z2_even(), u1(0));
    let triple = Triple::try_new(pair, SU2FusionRule).unwrap();
    let vacuum = triple.encode_component_ids(pair_vacuum, su2(0));
    assert_expert_storage_admission_for_rule(&triple, vacuum);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_coupled_storage_is_admitted_by_witness() {
    // What: checked-Generic SU(3) destinations with outer multiplicity use
    // the same coupled-sector builder, so both the staged preview and the
    // committed structure carry the witness.
    let rule = symtn_sectors::testing::su_lie_rule(3);
    let fundamental = rule.encode_irrep(&[1, 0]).unwrap();
    let adjoint = rule.encode_irrep(&[1, 1]).unwrap();
    let trivial = rule.encode_irrep(&[0, 0]).unwrap();
    let leg = |dual| SectorLeg::new([(fundamental, 3), (adjoint, 2), (trivial, 2)], dual);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    );
    let prepared = hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&rule)
        .unwrap();
    assert_canonical_storage_admitted_without_enumeration(prepared.structure());
    assert_canonical_storage_admitted_without_enumeration(&prepared.commit());
}

#[test]
fn external_aliased_storage_keeps_exact_rejection() {
    // What: layouts without the construction witness keep the exact check
    // and its error, even when they interleave like coupled subblocks.
    let aliased = BlockStructure::from_blocks(vec![
        BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 3], 0).unwrap(),
        BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 2], vec![1, 3], 1).unwrap(),
    ])
    .unwrap();
    assert!(!aliased.storage_tiling_proven());
    let recording = crate::probe::record();
    assert_eq!(
        validate_block_storage_injective(&aliased),
        Err(CoreError::OverlappingBlockStorage {
            first_block: 0,
            second_block: 1,
            offset: 1,
        })
    );
    assert!(recording.get(crate::probe::Event::ExactStorageFallback) > 0);
}
