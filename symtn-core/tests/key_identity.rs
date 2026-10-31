//! Block-key identity contract (#2192).
//!
//! What: `Debug`, `Ord`, `Eq` and the cache hasher's `Hash` of
//! `FusionTreeKey`, `FusionTreePairKey`, `FusionTreeGroupKey` and `BlockKey`
//! depend only on the key's identity fields, never on which arena stores
//! them, and the structure lookups find exactly the content-equal block.
//!
//! The hash formulas are written here from the public accessors; the few
//! literal hashes pin the in-process `rustc_hash::FxBuildHasher` values (the
//! hasher of every structure cache), so a reordered field or a changed hasher
//! is caught even when both sides of a formula would move together. The hashes
//! are never persisted. Slice hashing feeds native-endian `usize` bytes, so the
//! literals hold on 64-bit little-endian targets only.
#![cfg(all(target_pointer_width = "64", target_endian = "little"))]

use std::cmp::Ordering;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

use smallvec::SmallVec;
use symtn_core::{
    BlockKey, BlockStructure, BraidingStyleKind, CoreError, FusionProductSpace, FusionRule,
    FusionStyleKind, FusionTreeGroupKey, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey,
    FusionTreeRef, RuleIdentity, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorStructure,
    SectorVec, StructureCacheEquivalent, TableRule, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
};

fn fx<T: Hash + ?Sized>(value: &T) -> u64 {
    rustc_hash::FxBuildHasher.hash_one(value)
}

/// The stored tree hash: FxHasher over uncoupled, coupled, is_dual,
/// innerlines, vertices, in that order.
fn stored_tree_hash(tree: &FusionTreeKey) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    tree.uncoupled().hash(&mut hasher);
    tree.coupled().hash(&mut hasher);
    tree.is_dual().hash(&mut hasher);
    tree.innerlines().hash(&mut hasher);
    tree.vertices().hash(&mut hasher);
    hasher.finish()
}

/// `Hash` of a tree writes its stored hash as one `u64`.
fn tree_fx(tree: &FusionTreeKey) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hasher.write_u64(stored_tree_hash(tree));
    hasher.finish()
}

/// `Hash` of a pair writes the codomain then the domain stored hash.
fn pair_fx(pair: &FusionTreePairKey) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    hasher.write_u64(stored_tree_hash(pair.codomain_tree()));
    hasher.write_u64(stored_tree_hash(pair.domain_tree()));
    hasher.finish()
}

/// `Hash` of a group hashes its four slices, interleaved like its `Ord`.
fn group_fx(group: &FusionTreeGroupKey) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    group.codomain_uncoupled().hash(&mut hasher);
    group.domain_uncoupled().hash(&mut hasher);
    group.codomain_is_dual().hash(&mut hasher);
    group.domain_is_dual().hash(&mut hasher);
    hasher.finish()
}

/// A raw tree through the public, unvalidated pair constructor.
fn tree(
    uncoupled: &[usize],
    coupled: usize,
    is_dual: &[bool],
    innerlines: &[usize],
    vertices: &[usize],
) -> Result<FusionTreeKey, CoreError> {
    FusionTreePairKey::try_pair_from_sector_ids(
        uncoupled.iter().copied(),
        [],
        coupled,
        is_dual.iter().copied(),
        [],
        innerlines.iter().copied(),
        [],
        vertices.iter().copied(),
        [],
    )
    .map(|pair| pair.codomain_tree().clone())
}

fn ok_tree(
    uncoupled: &[usize],
    coupled: usize,
    is_dual: &[bool],
    innerlines: &[usize],
    vertices: &[usize],
) -> FusionTreeKey {
    tree(uncoupled, coupled, is_dual, innerlines, vertices).unwrap()
}

/// Representative trees: ranks 0, 1, 3, 5 and 16, sector ids above
/// `u32::MAX`, and field lengths that do not match the rank.
fn sample_trees() -> Vec<FusionTreeKey> {
    let rank16: Vec<usize> = (0..16).map(|i| i % 3).collect();
    let duals16: Vec<bool> = (0..16).map(|i| i % 2 == 0).collect();
    let inner14: Vec<usize> = (0..14).map(|i| (i * 2) % 3).collect();
    let vertices15 = vec![1; 15];
    vec![
        ok_tree(&[], 0, &[], &[], &[]),
        ok_tree(&[], 2, &[], &[], &[]),
        ok_tree(&[], usize::MAX, &[], &[], &[]),
        ok_tree(&[0], 2, &[false], &[], &[]),
        ok_tree(&[0], 2, &[true], &[], &[]),
        ok_tree(&[1, 2, 1], 2, &[true, true, true], &[2], &[1, 3]),
        ok_tree(&[1, 2, 1], 2, &[true, true, true], &[2], &[1, 2]),
        ok_tree(&[1, 2, 1], 2, &[true, true, false], &[2], &[1, 3]),
        ok_tree(&[1, 2, 2], 1, &[true, true, true], &[0], &[1, 1]),
        ok_tree(
            &[1 << 33, u32::MAX as usize + 1, usize::MAX],
            usize::MAX,
            &[false, true, false],
            &[usize::MAX - 1],
            &[2, 1],
        ),
        ok_tree(
            &[1, 0, 1, 2, 0],
            1,
            &[false, true, false, true, false],
            &[0, 2, 0],
            &[2, 1, 3, 2],
        ),
        ok_tree(&rank16, 2, &duals16, &inner14, &vertices15),
        // Malformed lengths are stored as given.
        ok_tree(&[], 7, &[true, false], &[1, 2, 3], &[4]),
        ok_tree(&[1, 2], 0, &[true], &[], &[1, 1, 1]),
        // The same tree built again from a separate arena.
        ok_tree(&[1, 2, 1], 2, &[true, true, true], &[2], &[1, 3]),
    ]
}

fn sample_pairs(trees: &[FusionTreeKey]) -> Vec<FusionTreePairKey> {
    let mut pairs = Vec::new();
    for (index, codomain) in trees.iter().enumerate() {
        let domain = &trees[(index * 5 + 3) % trees.len()];
        pairs.push(FusionTreePairKey::pair(codomain.clone(), domain.clone()));
        pairs.push(FusionTreePairKey::pair(domain.clone(), codomain.clone()));
    }
    pairs
}

fn sample_groups(pairs: &[FusionTreePairKey]) -> Vec<FusionTreeGroupKey> {
    let mut groups: Vec<_> = pairs.iter().map(FusionTreePairKey::group_key).collect();
    groups.push(FusionTreeGroupKey::new([], [], [], []));
    groups.push(FusionTreeGroupKey::from_sector_ids(
        [1, 2],
        [3],
        [true, false],
        [false],
    ));
    groups.push(FusionTreeGroupKey::from_sector_ids(
        [1, 2],
        [3],
        [true],
        [false, true, true],
    ));
    groups.push(FusionTreeGroupKey::new(
        [SectorId::new(usize::MAX)],
        [SectorId::new(1 << 40)],
        [false],
        [true],
    ));
    groups
}

fn sample_block_keys(pairs: &[FusionTreePairKey]) -> Vec<BlockKey> {
    let mut keys = vec![
        BlockKey::trivial(),
        BlockKey::opaque([]),
        BlockKey::opaque([0]),
        BlockKey::opaque([1, 2, 3]),
        BlockKey::opaque([u64::MAX]),
        BlockKey::ordinal(5),
        BlockKey::opaque([0, 0]),
        BlockKey::trivial(),
    ];
    keys.extend(pairs.iter().step_by(3).cloned().map(BlockKey::from));
    keys
}

/// `Eq` agrees with `Ord`, `Ord` is antisymmetric and transitive, and equal
/// keys hash equally.
fn check_consistency<T: Ord + Eq + Hash>(items: &[T]) {
    for left in items {
        for right in items {
            let ordering = left.cmp(right);
            assert_eq!(left == right, ordering == Ordering::Equal);
            assert_eq!(right.cmp(left), ordering.reverse());
            if left == right {
                assert_eq!(fx(left), fx(right));
            }
            for third in items {
                if ordering != Ordering::Greater && right.cmp(third) != Ordering::Greater {
                    assert_ne!(left.cmp(third), Ordering::Greater);
                }
            }
        }
    }
}

#[test]
fn key_hashes_follow_the_field_formula() {
    let trees = sample_trees();
    for tree in &trees {
        assert_eq!(fx(tree), tree_fx(tree), "{tree:?}");
    }
    let pairs = sample_pairs(&trees);
    for pair in &pairs {
        assert_eq!(fx(pair), pair_fx(pair), "{pair:?}");
    }
    for group in &sample_groups(&pairs) {
        assert_eq!(fx(group), group_fx(group), "{group:?}");
    }
}

#[test]
fn debug_prints_the_identity_fields() {
    assert_eq!(
        format!("{:?}", ok_tree(&[1, 2], 0, &[true], &[], &[1, 1, 1])),
        "FusionTreeKey { uncoupled: [SectorId(1), SectorId(2)], coupled: SectorId(0), \
         is_dual: [true], innerlines: [], vertices: [MultiplicityIndex(1), \
         MultiplicityIndex(1), MultiplicityIndex(1)] }"
    );
    assert_eq!(
        format!(
            "{:?}",
            FusionTreeGroupKey::new(
                [SectorId::new(usize::MAX)],
                [SectorId::new(1 << 40)],
                [false],
                [true],
            )
        ),
        "FusionTreeGroupKey { codomain_uncoupled: [SectorId(18446744073709551615)], \
         domain_uncoupled: [SectorId(1099511627776)], codomain_is_dual: [false], \
         domain_is_dual: [true] }"
    );
    assert_eq!(
        format!(
            "{:?}",
            BlockKey::from(FusionTreePairKey::pair(
                ok_tree(&[1], 2, &[false], &[], &[]),
                ok_tree(&[1], 1, &[false], &[], &[]),
            ))
        ),
        "FusionTree(FusionTreePairKey { codomain_tree: FusionTreeKey { uncoupled: \
         [SectorId(1)], coupled: SectorId(2), is_dual: [false], innerlines: [], vertices: [] }, \
         domain_tree: FusionTreeKey { uncoupled: [SectorId(1)], coupled: SectorId(1), \
         is_dual: [false], innerlines: [], vertices: [] } })"
    );
    assert_eq!(
        format!("{:?}", BlockKey::opaque([1, 2, 3])),
        "Opaque(OpaqueBlockKey { words: [1, 2, 3] })"
    );
}

#[test]
fn tree_order_breaks_ties_field_by_field() {
    // Uncoupled, then coupled, is_dual, innerlines, vertices; each field
    // compares lexicographically, so a shorter prefix sorts first.
    let base = ok_tree(&[1, 2, 1], 2, &[true, true, false], &[2], &[1, 3]);
    let ascending = [
        ok_tree(&[1, 2], 9, &[true, true], &[], &[9]),
        ok_tree(&[1, 2, 1], 1, &[true, true, true], &[9], &[9, 9]),
        ok_tree(&[1, 2, 1], 2, &[false, true, true], &[9], &[9, 9]),
        ok_tree(&[1, 2, 1], 2, &[true, true, false], &[1], &[9, 9]),
        ok_tree(&[1, 2, 1], 2, &[true, true, false], &[2], &[1, 2]),
        base.clone(),
        ok_tree(&[1, 2, 1], 2, &[true, true, false], &[2], &[1, 3, 1]),
        ok_tree(&[1, 2, 1], 2, &[true, true, false], &[3], &[1, 1]),
        ok_tree(&[1, 2, 1], 2, &[true, true, true], &[0], &[1, 1]),
        ok_tree(&[1, 2, 1], 3, &[false, false, false], &[0], &[1, 1]),
        ok_tree(&[1, 2, 2], 0, &[false, false, false], &[0], &[1, 1]),
    ];
    for window in ascending.windows(2) {
        assert_eq!(
            window[0].cmp(&window[1]),
            Ordering::Less,
            "{:?} < {:?}",
            window[0],
            window[1]
        );
    }
    // An equal tree from a separate arena compares equal.
    let again = ok_tree(&[1, 2, 1], 2, &[true, true, false], &[2], &[1, 3]);
    assert!(!base.shares_arena(&again));
    assert_eq!(base.cmp(&again), Ordering::Equal);
}

#[test]
fn composite_keys_order_by_their_parts() {
    let small = ok_tree(&[0], 1, &[false], &[], &[]);
    let large = ok_tree(&[1], 0, &[false], &[], &[]);
    // A pair orders by its codomain tree, then its domain tree.
    let pair = |c: &FusionTreeKey, d: &FusionTreeKey| FusionTreePairKey::pair(c.clone(), d.clone());
    assert!(pair(&small, &large) < pair(&large, &small));
    assert!(pair(&small, &small) < pair(&small, &large));
    // A group orders codomain sectors, domain sectors, codomain duals, domain
    // duals: a smaller domain sector wins over larger codomain duals.
    let lhs = FusionTreeGroupKey::from_sector_ids([1], [0], [true], [true]);
    let rhs = FusionTreeGroupKey::from_sector_ids([1], [1], [false], [false]);
    assert!(lhs < rhs);
    let lhs = FusionTreeGroupKey::from_sector_ids([1], [1], [false], [true]);
    assert!(lhs < FusionTreeGroupKey::from_sector_ids([1], [1], [true], [false]));
    // Block keys: Dense, then opaque words lexicographically, then fusion trees.
    let ascending = [
        BlockKey::trivial(),
        BlockKey::opaque([]),
        BlockKey::opaque([0]),
        BlockKey::opaque([0, 0]),
        BlockKey::opaque([1, 2, 3]),
        BlockKey::ordinal(5),
        BlockKey::opaque([u64::MAX]),
        BlockKey::from(pair(&small, &small)),
    ];
    for window in ascending.windows(2) {
        assert!(window[0] < window[1], "{:?} < {:?}", window[0], window[1]);
    }
}

#[test]
fn construction_keeps_fields_and_rejects_zero_vertices() {
    assert!(matches!(
        tree(&[0, 1, 2], 0, &[false; 3], &[1], &[1, 0]),
        Err(CoreError::InvalidMultiplicityIndex { value: 0 })
    ));
    // Unique (U(1)) trees whose inner lines are not the partial charge sums
    // are distinct raw keys: inner lines are stored, never recomputed.
    let id = |charge: i32| U1Irrep::new(charge).sector_id().id();
    let uncoupled: Vec<usize> = [1, -2, 3, 0, -1].into_iter().map(id).collect();
    let duals = [false, true, false, true, false];
    let build = |inner: [i32; 3]| {
        let inner: Vec<usize> = inner.into_iter().map(id).collect();
        ok_tree(&uncoupled, id(1), &duals, &inner, &[1; 4])
    };
    let canonical = build([-1, 2, 2]);
    assert_eq!(canonical, build([-1, 2, 2]));
    for other in [build([-1, 2, 3]), build([5, -5, 0])] {
        assert_ne!(canonical, other);
        assert_ne!(canonical.innerlines(), other.innerlines());
    }
}

#[test]
fn equality_order_and_hash_are_consistent_across_arenas() {
    let trees = sample_trees();
    let pairs = sample_pairs(&trees);
    let groups = sample_groups(&pairs);
    let block_keys = sample_block_keys(&pairs);
    check_consistency(&trees);
    check_consistency(&pairs);
    check_consistency(&groups);
    check_consistency(&block_keys);
}

/// The borrowed views agree with the owned keys: same hash bits, same order,
/// and `Equivalent` exactly where `Eq` holds.
#[test]
fn borrowed_views_match_owned_keys() {
    let trees = sample_trees();
    let pairs = sample_pairs(&trees);
    let groups = sample_groups(&pairs);
    for tree in &trees {
        let rebuilt = FusionTreeRef::new(
            tree.uncoupled(),
            tree.coupled(),
            tree.is_dual(),
            tree.innerlines(),
            tree.vertices(),
        );
        assert_eq!(rebuilt, tree.as_view());
        assert_eq!(fx(&rebuilt), fx(tree));
        assert!(rebuilt.equivalent(tree));
    }
    for left in &trees {
        for right in &trees {
            assert_eq!(left.as_view().cmp(&right.as_view()), left.cmp(right));
            assert_eq!(left.as_view().equivalent(right), left == right);
        }
    }
    for left in &pairs {
        let view = left.as_view();
        assert_eq!(fx(&view), fx(left));
        assert_eq!(view.group(), left.group_key().as_view());
        assert_eq!(fx(&view.group()), fx(&left.group_key()));
        for right in &pairs {
            assert_eq!(view.cmp(&right.as_view()), left.cmp(right));
            assert_eq!(view.equivalent(right), left == right);
        }
    }
    for left in &groups {
        assert_eq!(fx(&left.as_view()), fx(left));
        for right in &groups {
            assert_eq!(left.as_view().cmp(&right.as_view()), left.cmp(right));
            assert_eq!(left.as_view().equivalent(right), left == right);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MultiplicityTwoRule;

impl symtn_core::testing::Provider for MultiplicityTwoRule {}

impl FusionRule for MultiplicityTwoRule {
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
            (0, sector) | (sector, 0) => SmallVec::from_slice(&[SectorId::new(sector)]),
            (1, 1) => SmallVec::from_slice(&[SectorId::new(2)]),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 2) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

type Leg = (Vec<(SectorId, usize)>, bool);

fn hom(codomain: &[Leg], domain: &[Leg]) -> FusionTreeHomSpace {
    let side = |legs: &[Leg]| {
        FusionProductSpace::new(
            legs.iter()
                .map(|(pairs, dual)| SectorLeg::new(pairs.iter().copied(), *dual)),
        )
    };
    FusionTreeHomSpace::new(side(codomain), side(domain))
}

fn real_structures() -> Vec<(&'static str, Arc<BlockStructure>)> {
    let u1 = |charge: i32| U1Irrep::new(charge).sector_id();
    let su2 = |twice: usize| SU2Irrep::from_twice_spin(twice).sector_id();
    let z2 = vec![(Z2Irrep::EVEN.into(), 1), (Z2Irrep::ODD.into(), 2)];
    let u1_leg: Vec<_> = (-1..=1).map(|charge| (u1(charge), 2)).collect();
    let su2_leg: Vec<_> = (0..=2).map(|twice| (su2(twice), 1)).collect();
    let fib = vec![(SectorId::new(0), 1), (SectorId::new(1), 2)];
    let mult = vec![(SectorId::new(0), 1), (SectorId::new(1), 2)];
    let leg = |sectors: &Vec<(SectorId, usize)>, dual: bool| (sectors.clone(), dual);
    vec![
        (
            "u1-r4",
            hom(
                &[leg(&u1_leg, false), leg(&u1_leg, true)],
                &[leg(&u1_leg, false), leg(&u1_leg, false)],
            )
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap(),
        ),
        (
            "u1-r3-empty-domain",
            hom(
                &[leg(&u1_leg, false), leg(&u1_leg, false), leg(&u1_leg, true)],
                &[],
            )
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap(),
        ),
        (
            "z2-r5",
            hom(
                &[leg(&z2, false), leg(&z2, true), leg(&z2, false)],
                &[leg(&z2, false), leg(&z2, true)],
            )
            .coupled_subblock_structure_from_leg_degeneracies(&Z2FusionRule)
            .unwrap(),
        ),
        // Identical codomain and domain legs: swapped pairs built from the
        // structure's own trees name real blocks on the other side.
        (
            "su2-r4-symmetric",
            hom(
                &[leg(&su2_leg, false), leg(&su2_leg, false)],
                &[leg(&su2_leg, false), leg(&su2_leg, false)],
            )
            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap(),
        ),
        (
            "fibonacci-r4",
            hom(
                &[leg(&fib, false), leg(&fib, false), leg(&fib, true)],
                &[leg(&fib, false)],
            )
            .coupled_subblock_structure_from_leg_degeneracies(&TableRule::fibonacci())
            .unwrap(),
        ),
        (
            "generic-multiplicity-r3",
            hom(
                &[leg(&mult, false), leg(&mult, false)],
                &[leg(&mult, false)],
            )
            .coupled_subblock_structure_from_leg_degeneracies_generic(&MultiplicityTwoRule)
            .unwrap(),
        ),
    ]
}

/// The same pair rebuilt into a fresh arena.
fn rebuild_pair(pair: &FusionTreePairKey) -> FusionTreePairKey {
    let rebuild = |tree: &FusionTreeKey| {
        let ids = |sectors: &[SectorId]| sectors.iter().map(|s| s.id()).collect::<Vec<_>>();
        let vertices: Vec<usize> = tree.vertices().iter().map(|v| v.get()).collect();
        ok_tree(
            &ids(tree.uncoupled()),
            tree.coupled().id(),
            tree.is_dual(),
            &ids(tree.innerlines()),
            &vertices,
        )
    };
    FusionTreePairKey::pair(rebuild(pair.codomain_tree()), rebuild(pair.domain_tree()))
}

fn swap(pair: &FusionTreePairKey) -> FusionTreePairKey {
    FusionTreePairKey::pair(pair.domain_tree().clone(), pair.codomain_tree().clone())
}

#[test]
fn structure_lookups_find_exactly_the_content_equal_block() {
    let structures = real_structures();
    let foreign_pairs: Vec<FusionTreePairKey> = structures
        .iter()
        .flat_map(|(_, structure)| {
            let sectors = structure.sector_structure();
            (0..structure.block_count()).map(move |index| {
                sectors
                    .key(index)
                    .unwrap()
                    .as_fusion_tree_pair()
                    .unwrap()
                    .clone()
            })
        })
        .chain(sample_pairs(&sample_trees()))
        .collect();
    for (label, structure) in &structures {
        let sectors = structure.sector_structure();
        let own: Vec<FusionTreePairKey> = (0..structure.block_count())
            .map(|index| {
                sectors
                    .key(index)
                    .unwrap()
                    .as_fusion_tree_pair()
                    .unwrap()
                    .clone()
            })
            .collect();
        let expected = |key: &FusionTreePairKey| own.iter().position(|block| block == key);
        let expected_adjoint =
            |key: &FusionTreePairKey| own.iter().position(|block| *block == swap(key));

        // Stored keys are distinct and `sorted_indices` lists them in `Ord` order.
        let sorted = sectors.sorted_indices();
        let mut permutation = sorted.to_vec();
        permutation.sort_unstable();
        assert_eq!(permutation, (0..own.len()).collect::<Vec<_>>(), "{label}");
        for window in sorted.windows(2) {
            assert!(own[window[0]] < own[window[1]], "{label}");
        }

        // Each block sits in the one group of its external sectors.
        let mut grouped = vec![0usize; own.len()];
        for group in structure.fusion_tree_group_slice() {
            for &index in group.block_indices() {
                assert_eq!(own[index].group_key(), *group.group_key(), "{label}");
                grouped[index] += 1;
            }
        }
        assert!(grouped.iter().all(|&count| count == 1), "{label}");

        for (index, key) in own.iter().enumerate() {
            let rebuilt = rebuild_pair(key);
            assert!(!rebuilt.codomain_tree().shares_arena(key.codomain_tree()));
            for probe in [key, &rebuilt] {
                assert_eq!(
                    sectors.find_fusion_tree_pair_index(probe),
                    Some(index),
                    "{label}"
                );
                assert_eq!(
                    structure.find_block_index_by_fusion_tree_pair(probe),
                    Some(index)
                );
                assert_eq!(
                    sectors.find_index(&BlockKey::from(probe.clone())),
                    Some(index)
                );
                assert_eq!(
                    sectors.find_adjoint_fusion_tree_pair_index(&swap(probe)),
                    Some(index)
                );
                let swapped = swap(probe);
                assert_eq!(
                    sectors.find_fusion_tree_pair_index(&swapped),
                    expected(&swapped),
                    "{label}"
                );
                assert_eq!(
                    sectors.find_adjoint_fusion_tree_pair_index(probe),
                    expected_adjoint(probe)
                );
            }
        }
        for key in &foreign_pairs {
            assert_eq!(
                sectors.find_fusion_tree_pair_index(key),
                expected(key),
                "{label} {key:?}"
            );
            assert_eq!(
                sectors.find_adjoint_fusion_tree_pair_index(key),
                expected_adjoint(key)
            );
            assert_eq!(
                sectors.find_index(&BlockKey::from(key.clone())),
                expected(key)
            );
        }
        for key in [
            BlockKey::trivial(),
            BlockKey::opaque([0]),
            BlockKey::ordinal(1),
        ] {
            assert_eq!(sectors.find_index(&key), None, "{label}");
        }
    }
}

// Block order recorded by the former key corpus at e5023d98 and unchanged
// through 85d16e6d. Block indices are observable (block access, lookups), so
// a change of this order is a contract change, not a refactor.
#[test]
fn structure_block_order_matches_recorded_order() {
    let structures = real_structures();
    let uncoupled = |label: &str| -> Vec<Vec<usize>> {
        let (_, structure) = structures.iter().find(|(name, _)| *name == label).unwrap();
        let sectors = structure.sector_structure();
        (0..structure.block_count())
            .map(|index| {
                let pair = sectors.key(index).unwrap().as_fusion_tree_pair().unwrap();
                pair.codomain_tree()
                    .uncoupled()
                    .iter()
                    .map(|s| s.id())
                    .collect()
            })
            .collect()
    };
    assert_eq!(
        uncoupled("u1-r3-empty-domain"),
        [
            [0, 0, 0],
            [2, 1, 0],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2],
            [0, 1, 2]
        ]
    );
    assert_eq!(
        uncoupled("generic-multiplicity-r3"),
        [[0, 0], [1, 0], [0, 1]]
    );
}

#[test]
fn keyed_structures_find_their_own_keys_only() {
    let opaque_keys = [
        BlockKey::opaque([1, 2, 3]),
        BlockKey::ordinal(5),
        BlockKey::opaque([0]),
        BlockKey::opaque([u64::MAX]),
        BlockKey::opaque([]),
        BlockKey::opaque([9, 9]),
    ];
    let opaque = SectorStructure::from_keys(2, opaque_keys.clone()).unwrap();
    // `Ord` order: [] < [0] < [1, 2, 3] < [5] < [9, 9] < [u64::MAX].
    assert_eq!(opaque.sorted_indices(), [4, 2, 0, 1, 5, 3]);
    let probes = sample_block_keys(&sample_pairs(&sample_trees()));
    for key in &probes {
        let expected = opaque_keys.iter().position(|own| own == key);
        assert_eq!(opaque.find_index(key), expected, "{key:?}");
    }
    let dense = SectorStructure::dense(2);
    let empty = SectorStructure::empty(2);
    assert_eq!(dense.sorted_indices(), [0]);
    assert!(empty.sorted_indices().is_empty());
    for key in &probes {
        let expected = (*key == BlockKey::trivial()).then_some(0);
        assert_eq!(dense.find_index(key), expected, "{key:?}");
        assert_eq!(empty.find_index(key), None);
    }
    for pair in sample_pairs(&sample_trees()) {
        assert_eq!(opaque.find_fusion_tree_pair_index(&pair), None);
        assert_eq!(dense.find_fusion_tree_pair_index(&pair), None);
    }
}
