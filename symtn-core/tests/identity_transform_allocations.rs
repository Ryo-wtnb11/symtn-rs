use std::hint::black_box;

use symtn_core::testing::{
    multiplicity_free_braid_tree_block, multiplicity_free_permute_tree_pair_block,
    unique_permute_tree,
};
use symtn_core::{
    multiplicity_free_permute_tree_pair_block_indexed, BlockKey, BlockSpec, BlockStructure,
    BraidingStyleKind, FermionParityFusionRule, FusionProductSpace, FusionRule, FusionStyleKind,
    FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey, FusionTreePairOrientation,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    MultiplicityIndex, PreparedTreePairOperation, RuleIdentity, SU2FusionRule, SU2Irrep, SectorId,
    SectorLeg, SectorStructure, SectorVec, Z2FusionRule, Z2Irrep,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[derive(Clone, Copy)]
struct AllocationAnyonicRule;

impl FusionRule for AllocationAnyonicRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SectorVec::from_slice(&[SectorId::new((left.id() + right.id()) % 32)])
    }
}

impl MultiplicityFreeFusionRule for AllocationAnyonicRule {}

impl MultiplicityFreeFusionSymbols for AllocationAnyonicRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for AllocationAnyonicRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[test]
fn constructed_sector_structure_borrows_fusion_groups_without_allocation() {
    let first = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2],
        [3],
        4,
        [false, true],
        [true],
        [5],
        [],
        [1, 1],
        [1],
    )
    .unwrap();
    let second = FusionTreePairKey::try_pair_from_sector_ids(
        [7],
        [8, 9],
        10,
        [true],
        [false, true],
        [],
        [11],
        [1],
        [1, 1],
    )
    .unwrap();
    let same_group_as_first = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2],
        [3],
        12,
        [false, true],
        [true],
        [13],
        [],
        [1, 1],
        [1],
    )
    .unwrap();
    let structure = SectorStructure::from_keys(
        3,
        [first, second, same_group_as_first]
            .into_iter()
            .map(BlockKey::from),
    )
    .unwrap();
    assert_eq!(
        structure.fusion_tree_group_slice()[0].block_indices(),
        &[0, 2]
    );
    assert_eq!(structure.fusion_tree_group_slice()[1].block_indices(), &[1]);

    let (_, allocations) = measured_allocations(|| {
        let groups = black_box(&structure).fusion_tree_group_slice();
        black_box(groups.as_ptr());
        black_box(groups[0].group_key());
        black_box(groups[0].block_indices());
    });

    // What: repeated group lookup borrows construction-time metadata and does
    // not rebuild group keys, indices, or a membership hash table.
    assert_eq!(allocations, 0);
}

#[test]
fn unique_identity_permute_does_not_allocate() {
    // What: the identity Unique permutation returns its inline tree key
    // without constructing permutation-level scratch.
    let tree = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [Z2Irrep::ODD.sector_id()],
        Z2Irrep::ODD.sector_id(),
        [false],
        [],
        [],
    )
    .unwrap();
    let _ = unique_permute_tree(&Z2FusionRule, &tree, &[0]).unwrap();

    let (transformed, allocations) =
        measured_allocations(|| unique_permute_tree(&Z2FusionRule, &tree, &[0]).unwrap());

    assert_eq!(transformed.0, tree);
    assert_eq!(transformed.1, 1.0);
    assert_eq!(allocations, 0);
}

#[test]
fn large_tree_pair_syntax_validation_does_not_allocate() {
    let codomain_rank = 10;
    let domain_rank = 9;
    let codomain = (0..codomain_rank).collect::<Vec<_>>();
    let domain = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    let run = || {
        PreparedTreePairOperation::validate_permute_syntax(
            codomain_rank,
            domain_rank,
            &codomain,
            &domain,
        )
        .unwrap();
        PreparedTreePairOperation::validate_braid_syntax(
            codomain_rank,
            domain_rank,
            &codomain,
            &domain,
            &codomain_levels,
            &domain_levels,
        )
        .unwrap();
        PreparedTreePairOperation::validate_transpose_syntax(
            codomain_rank,
            domain_rank,
            &codomain,
            &domain,
        )
        .unwrap();
    };
    run();

    let (_, allocations) = measured_allocations(|| {
        let _: () = run();
        black_box(())
    });

    // What: rank-19 syntax preflight, beyond any rank-8 inline scratch,
    // remains allocation-free on successful validation.
    assert_eq!(allocations, 0);
}

#[test]
fn axis_validation_is_allocation_free_through_rank_128() {
    let permutation = (0..128).rev().collect::<Vec<_>>();
    let subset = (0..128).step_by(3).collect::<Vec<_>>();

    let (open, allocations) = measured_allocations(|| {
        symtn_core::axes::validate_permutation(&permutation, 128).unwrap();
        let mask = symtn_core::axes::validate_axis_subset(&subset, 128).unwrap();
        black_box(mask.complement().count())
    });

    // What: the shared validator replaces per-call `vec![false; rank]` and
    // 8/16-entry inline scratch; one inline bitset covers rank <= 128.
    assert_eq!(open, 128 - subset.len());
    assert_eq!(allocations, 0);
}

#[test]
fn unique_rank_129_preparation_borrows_prebuilt_operation_metadata() {
    let rank = 129;
    let permutation = (0..rank).rev().collect::<Vec<_>>();
    let raw_axis_positions = (0..rank).rev().collect::<Vec<_>>();
    let run = || {
        PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
            &Z2FusionRule,
            rank,
            0,
            &permutation,
            &[],
            &raw_axis_positions,
        )
        .unwrap()
    };
    run();

    let (_, calls, bytes) = measured_allocation_stats(|| black_box(run()));

    // What: once runtime-rank operation metadata exists, UniqueFusion
    // preparation creates no retained permutation or Artin-step allocation.
    assert_eq!((calls, bytes), (0, 0));
}

#[test]
fn block_identity_permute_allocates_only_owned_output() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [0],
        0,
        [false, false],
        [false],
        [],
        [],
        [1],
        [],
    )
    .unwrap();
    let sources = [source.clone()];
    let _ =
        multiplicity_free_permute_tree_pair_block(&Z2FusionRule, &sources, &[0, 1], &[2]).unwrap();

    let (transformed, allocations) = measured_allocations(|| {
        multiplicity_free_permute_tree_pair_block(&Z2FusionRule, &sources, &[0, 1], &[2]).unwrap()
    });

    // What: identity block permutation allocates only the intentional owned
    // outer result and its one owned row, with no level-vector temporaries.
    assert_eq!(transformed, vec![vec![(source, 1.0)]]);
    assert_eq!(allocations, 2);
}

/// Per-source `(destination, coefficient)` rows of an indexed result.
fn columns(
    block: &symtn_core::BlockSourceColumns<FusionTreePairKey, f64>,
) -> Vec<Vec<(FusionTreePairKey, f64)>> {
    (0..block.source_count())
        .map(|source| {
            block
                .column(source)
                .iter()
                .map(|(row, coefficient)| (block.destinations()[*row].clone(), *coefficient))
                .collect()
        })
        .collect()
}

#[test]
fn indexed_adjoint_identity_allocates_only_owned_output() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [0],
        0,
        [false, false],
        [false],
        [],
        [],
        [1],
        [],
    )
    .unwrap();
    let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        source.clone().into(),
        vec![1; 3],
        0,
    )
    .unwrap()])
    .unwrap();
    let indices = structure.fusion_tree_group_slice()[0].block_indices();
    let run = || {
        multiplicity_free_permute_tree_pair_block_indexed(
            &Z2FusionRule,
            &structure,
            indices,
            FusionTreePairOrientation::Adjoint,
            &[0],
            &[1, 2],
        )
        .unwrap()
    };
    let _ = run();

    let (transformed, allocations) = measured_allocations(|| black_box(run()));
    let logical_source =
        FusionTreePairKey::pair(source.domain_tree().clone(), source.codomain_tree().clone());

    // What: oriented indexed preparation borrows parent key, group, shape, and
    // stride metadata; only the owned columns (destinations, column starts,
    // entries) allocate.
    assert_eq!(columns(&transformed), vec![vec![(logical_source, 1.0)]]);
    assert!(allocations <= 3, "{allocations}");
}

#[test]
fn indexed_adjoint_simple_group_allocates_only_owned_rows() {
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let leg = || SectorLeg::new([(half, 1)], false);
    // `½^⊗3 ← ½^⊗3`: five tree pairs in one Simple group, so a per-source
    // allocation would exceed the bound below.
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new([leg(), leg(), leg()]),
    );
    let sources = homspace.fusion_tree_keys(&SU2FusionRule);
    assert_eq!(sources.len(), 5);
    let structure = BlockStructure::from_blocks(
        sources
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, key)| {
                BlockSpec::column_major_with_key(key.into(), vec![1; 6], index).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let indices = structure.fusion_tree_group_slice()[0].block_indices();
    assert_eq!(indices, (0..sources.len()).collect::<Vec<_>>());
    let run = || {
        multiplicity_free_permute_tree_pair_block_indexed(
            &SU2FusionRule,
            &structure,
            indices,
            FusionTreePairOrientation::Adjoint,
            &[0, 1, 2],
            &[3, 4, 5],
        )
        .unwrap()
    };
    let _ = run();

    let (transformed, allocations) = measured_allocations(|| black_box(run()));
    let expected = sources
        .iter()
        .map(|source| {
            vec![(
                FusionTreePairKey::pair(
                    source.domain_tree().clone(),
                    source.codomain_tree().clone(),
                ),
                1.0,
            )]
        })
        .collect::<Vec<_>>();

    // What: a Simple cohort borrows its parent group and pair frames; the
    // owned columns (three slices) are the only allocations, independent of
    // the source count (the per-source rows form allocated 1 + 5).
    assert_eq!(columns(&transformed), expected);
    assert!(
        allocations <= 3 && allocations < sources.len(),
        "{allocations}"
    );
}

#[test]
fn compact_block_warm_allocations_do_not_restore_per_source_scratch() {
    let rule = SU2FusionRule;
    let spin_one = SU2Irrep::from_twice_spin(2).sector_id();
    let codomain: [SectorLeg; 8] = std::array::from_fn(|_| SectorLeg::new([(spin_one, 1)], false));
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain),
        FusionProductSpace::new([SectorLeg::new([(spin_one, 1)], false)]),
    );
    let keys = hom.fusion_tree_keys(&rule);
    let sources = &keys[..16];
    let codomain_permutation = [7usize, 6, 5, 4, 3, 2, 1, 0];
    let domain_permutation = [8usize];
    let _ = multiplicity_free_permute_tree_pair_block(
        &rule,
        sources,
        &codomain_permutation,
        &domain_permutation,
    )
    .unwrap();

    let (output, allocations) = measured_allocations(|| {
        black_box(
            multiplicity_free_permute_tree_pair_block(
                &rule,
                sources,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap(),
        )
    });

    // What: the fixed 16-source non-Abelian warm transform retains one compact
    // block workspace. Frozen identity adds one final innerline owner per 232
    // destination rows, two external owners per 28 Artin steps, and four bend
    // vertex owners, but no per-source scratch.
    assert_eq!(output.len(), sources.len());
    const COMPACT_WORKSPACE_BUDGET: usize = 3072;
    const FROZEN_OWNER_ALLOWANCE: usize = 232 + 2 * 28 + 4;
    assert!(
        allocations <= COMPACT_WORKSPACE_BUDGET + FROZEN_OWNER_ALLOWANCE,
        "compact warm block allocated {allocations} times"
    );
}

#[test]
fn shared_frame_decode_does_not_allocate_per_source_above_inline_rank() {
    for (rank, max_per_extra_source) in [(9usize, 4usize), (16, 8)] {
        let even = Z2Irrep::EVEN.sector_id();
        let source = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            std::iter::repeat_n(even, rank),
            even,
            std::iter::repeat_n(false, rank),
            std::iter::repeat_n(even, rank.saturating_sub(2)),
            std::iter::repeat_n(MultiplicityIndex::ONE, rank.saturating_sub(1)),
        )
        .unwrap();
        let single = [source.clone()];
        let cohort = std::iter::repeat_n(source, 16).collect::<Vec<_>>();
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(0, 1);
        let levels = (0..rank).collect::<Vec<_>>();

        let run = |sources: &[FusionTreeKey]| {
            multiplicity_free_braid_tree_block(&Z2FusionRule, sources, &permutation, &levels)
                .unwrap()
        };
        black_box(run(&single));
        black_box(run(&cohort));
        let (_, single_allocations) = measured_allocations(|| black_box(run(&single)));
        let (_, cohort_allocations) = measured_allocations(|| black_box(run(&cohort)));

        // What: duplicate high-rank sources do not exceed the previous
        // per-source owned-output/scratch slope. Frozen frame sharing may lower
        // the count, but rank-dependent reconstruction may not raise it.
        assert!(
            cohort_allocations - single_allocations
                <= max_per_extra_source * (cohort.len() - single.len()),
            "rank-{rank} shared-frame allocation slope changed"
        );
    }
}

#[test]
fn tree_pair_shared_frames_do_not_allocate_per_source_above_inline_rank() {
    for (rank, max_per_extra_source) in [(9usize, 4usize), (16, 8)] {
        let even = Z2Irrep::EVEN.sector_id();
        let codomain = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            std::iter::repeat_n(even, rank),
            even,
            std::iter::repeat_n(false, rank),
            std::iter::repeat_n(even, rank.saturating_sub(2)),
            std::iter::repeat_n(MultiplicityIndex::ONE, rank.saturating_sub(1)),
        )
        .unwrap();
        let domain = FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            std::iter::empty(),
            even,
            std::iter::empty(),
            std::iter::empty(),
            std::iter::empty(),
        )
        .unwrap();
        let source = FusionTreePairKey::pair(codomain, domain);
        let single = [source.clone()];
        let cohort = std::iter::repeat_n(source, 16).collect::<Vec<_>>();
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(0, 1);

        let run = |sources: &[FusionTreePairKey]| {
            multiplicity_free_permute_tree_pair_block(&Z2FusionRule, sources, &permutation, &[])
                .unwrap()
        };
        black_box(run(&single));
        black_box(run(&cohort));
        let (_, single_allocations) = measured_allocations(|| black_box(run(&single)));
        let (_, cohort_allocations) = measured_allocations(|| black_box(run(&cohort)));

        // What: codomain/domain frame matching does not exceed the previous
        // high-rank per-source owned-output/scratch slope. Sharing may improve
        // the count without invalidating this non-regression contract.
        assert!(
            cohort_allocations - single_allocations
                <= max_per_extra_source * (cohort.len() - single.len()),
            "rank-{rank} tree-pair shared-frame allocation slope changed"
        );
    }
}

fn measured_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    let (output, allocations, _) = measured_allocation_stats(operation);
    (output, allocations)
}

/// Allocating calls (`alloc`, `alloc_zeroed`, `realloc`) and their requested
/// bytes on this thread while `operation` runs.
fn measured_allocation_stats<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    let (output, allocs) = counting_alloc::measure(operation);
    (output, allocs.calls as usize, allocs.bytes as usize)
}

#[test]
fn prepared_nonidentity_unique_operations_have_zero_prepare_and_stable_execute_allocations() {
    let odd = Z2Irrep::ODD.sector_id();
    let even = Z2Irrep::EVEN.sector_id();
    let tree = || {
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [odd, odd],
            even,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let source = FusionTreePairKey::pair(tree(), tree());
    let single = [source.clone()];
    let cohort = std::iter::repeat_n(source, 16).collect::<Vec<_>>();
    let permute = PreparedTreePairOperation::prepare_permute(
        &FermionParityFusionRule,
        2,
        2,
        &[3, 0],
        &[1, 2],
    )
    .unwrap();
    let braid = PreparedTreePairOperation::prepare_braid(
        &FermionParityFusionRule,
        2,
        2,
        &[3, 0],
        &[1, 2],
        &[7, 2],
        &[11, 5],
    )
    .unwrap();
    let transpose = PreparedTreePairOperation::prepare_transpose(2, 2, &[1, 3], &[0, 2]).unwrap();

    let preparations: [fn() -> PreparedTreePairOperation<'static>; 3] = [
        || {
            PreparedTreePairOperation::prepare_permute(
                &FermionParityFusionRule,
                2,
                2,
                &[3, 0],
                &[1, 2],
            )
            .unwrap()
        },
        || {
            PreparedTreePairOperation::prepare_braid(
                &FermionParityFusionRule,
                2,
                2,
                &[3, 0],
                &[1, 2],
                &[7, 2],
                &[11, 5],
            )
            .unwrap()
        },
        || PreparedTreePairOperation::prepare_transpose(2, 2, &[1, 3], &[0, 2]).unwrap(),
    ];
    for prepare in preparations {
        let (_, calls, bytes) = measured_allocation_stats(prepare);
        // What: valid rank<=8 operation validation and lowering fit entirely
        // in the prepared representation's inline storage.
        assert_eq!((calls, bytes), (0, 0));
    }

    for prepared in [&permute, &braid, &transpose] {
        let run = |sources: &[FusionTreePairKey]| {
            for source in sources {
                black_box(
                    prepared
                        .execute_unique_rigid(&FermionParityFusionRule, source)
                        .unwrap(),
                );
            }
        };
        run(&single);
        run(&cohort);
        let (_, single_calls, single_bytes) = measured_allocation_stats(|| run(&single));
        let (_, repeated_calls, repeated_bytes) = measured_allocation_stats(|| run(&single));
        let (_, cohort_calls, cohort_bytes) = measured_allocation_stats(|| run(&cohort));

        // What: repeated nonidentity replay has a stable allocation contract,
        // and a source cohort pays exactly the measured per-source cost.
        assert_eq!(
            (repeated_calls, repeated_bytes),
            (single_calls, single_bytes)
        );
        assert_eq!(cohort_calls, single_calls * cohort.len());
        assert_eq!(cohort_bytes, single_bytes * cohort.len());
    }
}

#[test]
fn borrowed_unique_execution_allocations_do_not_scale_with_artin_schedule() {
    let rule = AllocationAnyonicRule;
    let rank = 9;
    let source_tree = FusionTreeKey::try_new_for_rule(
        &rule,
        std::iter::repeat_n(SectorId::new(1), rank - 1).chain([SectorId::new(24)]),
        rule.vacuum(),
        std::iter::repeat_n(false, rank),
        (2..rank).map(SectorId::new),
        std::iter::repeat_n(MultiplicityIndex::ONE, rank - 1),
    )
    .unwrap();
    let domain = FusionTreeKey::try_new_for_rule(&rule, [], rule.vacuum(), [], [], []).unwrap();
    let source = FusionTreePairKey::pair(source_tree, domain);
    let levels = (0..rank).collect::<Vec<_>>();
    let short_permutation = [1]
        .into_iter()
        .chain([0])
        .chain(2..rank)
        .collect::<Vec<_>>();
    let short_positions = short_permutation.iter().enumerate().fold(
        vec![0; rank],
        |mut positions, (position, &axis)| {
            positions[axis] = position;
            positions
        },
    );
    let long_permutation = (0..rank).rev().collect::<Vec<_>>();
    let long_positions = long_permutation.iter().enumerate().fold(
        vec![0; rank],
        |mut positions, (position, &axis)| {
            positions[axis] = position;
            positions
        },
    );
    let short = PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
        &rule,
        rank,
        0,
        &short_permutation,
        &[],
        &levels,
        &[],
        &short_positions,
    )
    .unwrap();
    let long = PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
        &rule,
        rank,
        0,
        &long_permutation,
        &[],
        &levels,
        &[],
        &long_positions,
    )
    .unwrap();
    let run = |prepared: &PreparedTreePairOperation<'_>| {
        black_box(prepared.execute_multiplicity_free(&rule, &source).unwrap())
    };

    run(&short);
    run(&long);
    let (_, short_calls) = measured_allocations(|| run(&short));
    let (_, long_calls) = measured_allocations(|| run(&long));

    // What: UniqueFusion replay allocations are bounded by the one frozen
    // output arena (its sector, duality and vertex buffers and its `Arc`)
    // and the term vectors, not by the number of Artin steps.
    const MUTATED_FIELD_OWNER_BUDGET: usize = 8;
    assert!(short_calls <= MUTATED_FIELD_OWNER_BUDGET);
    assert!(long_calls <= MUTATED_FIELD_OWNER_BUDGET);
}

#[test]
fn direct_simple_preparation_does_not_freeze_compiler_owners() {
    let prepare = || {
        PreparedTreePairOperation::prepare_braid(&SU2FusionRule, 2, 0, &[1, 0], &[], &[0, 1], &[])
            .unwrap()
    };

    let _ = prepare();
    let (_, allocations) = measured_allocations(|| black_box(prepare()));

    // What: direct SimpleFusion preparation keeps its ephemeral lowering;
    // compiler-owned runtime storage is introduced only after this API returns.
    assert_eq!(allocations, 0);
}

#[test]
fn coupled_matricization_builder_allocates_per_map_growth_not_per_block() {
    let leg = SectorLeg::new(
        (0..5).map(|spin| (SU2Irrep::from_twice_spin(spin).sector_id(), spin + 1)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let structure = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
    let blocks = (0..structure.block_count())
        .map(|index| {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                unreachable!("coupled structures carry fusion-tree keys")
            };
            let (rows, cols) = block.shape().split_at(2);
            (key, rows.iter().product(), cols.iter().product())
        })
        .collect::<Vec<(_, usize, usize)>>();
    assert!(blocks.len() >= 256);

    let (_, allocations) = measured_allocations(|| {
        let mut layout = symtn_core::CoupledMatricizationBuilder::new();
        for &(key, rows, cols) in &blocks {
            black_box(layout.place(key, rows, cols).unwrap());
        }
        black_box(layout)
    });

    // What: the shared coupled-sector placement keeps only its sector and
    // tree maps, so placing every block of a large SU(2) grid costs map
    // growth (logarithmic in the entry count), never an allocation per block.
    assert!(
        allocations <= 32,
        "{allocations} allocations for {} blocks",
        blocks.len()
    );
}
