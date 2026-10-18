//! Golden corpus for the block-key identity contract (#2192).
//!
//! What: `Debug`, `Ord` and the cache hasher's `Hash` of `FusionTreeKey`,
//! `FusionTreePairKey`, `FusionTreeGroupKey` and `BlockKey`, plus the block
//! order of real structures, equal the values recorded in
//! `tests/golden/key_corpus.txt`. That file was computed from the `Arc`-slice
//! representation at `53ddf679`, so it is an oracle independent of the arena
//! representation.
//!
//! Regenerate only for an intentional contract change:
//! `SYMTN_BLESS_KEY_CORPUS=1 cargo test -p symtn-core --test key_golden_corpus`.
//!
//! The hashes are those of `rustc_hash::FxBuildHasher`, the hasher of every
//! structure cache. Slice hashing feeds native-endian `usize` bytes and the
//! sector ids above `u32::MAX` print differently on narrower targets, so the
//! corpus is pinned to 64-bit little-endian targets (macOS aarch64 and Linux
//! x86_64 CI).
#![cfg(all(target_pointer_width = "64", target_endian = "little"))]

use std::cmp::Ordering;
use std::fmt::Write as _;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;

use smallvec::SmallVec;
use symtn_core::{
    BlockKey, BlockStructure, BraidingStyleKind, CoreError, FusionProductSpace, FusionRule,
    FusionStyleKind, FusionTreeGroupKey, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey,
    FusionTreeRef, RuleIdentity, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorStructure,
    SectorVec, StructureCacheEquivalent, TableRule, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
};

const FIXTURE: &str = "tests/golden/key_corpus.txt";
const MAX_RANK: usize = 16;

fn fx<T: Hash + ?Sized>(value: &T) -> u64 {
    rustc_hash::FxBuildHasher.hash_one(value)
}

/// The stored tree hash, respecified from the public accessors: FxHasher over
/// uncoupled, coupled, is_dual, innerlines, vertices, in that order.
fn stored_tree_hash(tree: &FusionTreeKey) -> u64 {
    let mut hasher = rustc_hash::FxHasher::default();
    tree.uncoupled().hash(&mut hasher);
    tree.coupled().hash(&mut hasher);
    tree.is_dual().hash(&mut hasher);
    tree.innerlines().hash(&mut hasher);
    tree.vertices().hash(&mut hasher);
    hasher.finish()
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn sectors(&mut self, len: usize, bound: usize) -> Vec<usize> {
        (0..len).map(|_| self.below(bound)).collect()
    }

    fn duals(&mut self, len: usize) -> Vec<bool> {
        (0..len).map(|_| self.next() & 1 == 1).collect()
    }

    fn vertices(&mut self, len: usize, max: usize) -> Vec<usize> {
        (0..len).map(|_| 1 + self.below(max)).collect()
    }
}

#[derive(Clone)]
struct RawTree {
    uncoupled: Vec<usize>,
    coupled: usize,
    is_dual: Vec<bool>,
    innerlines: Vec<usize>,
    vertices: Vec<usize>,
}

impl RawTree {
    fn canonical(rng: &mut SplitMix, rank: usize, bound: usize, max_vertex: usize) -> Self {
        Self {
            uncoupled: rng.sectors(rank, bound),
            coupled: rng.below(bound),
            is_dual: rng.duals(rank),
            innerlines: rng.sectors(rank.saturating_sub(2), bound),
            vertices: rng.vertices(rank.saturating_sub(1), max_vertex),
        }
    }

    /// Raw construction through the public, unvalidated pair constructor.
    fn build(&self) -> Result<FusionTreeKey, String> {
        FusionTreePairKey::try_pair_from_sector_ids(
            self.uncoupled.iter().copied(),
            [],
            self.coupled,
            self.is_dual.iter().copied(),
            [],
            self.innerlines.iter().copied(),
            [],
            self.vertices.iter().copied(),
            [],
        )
        .map(|pair| pair.codomain_tree().clone())
        .map_err(|error| format!("{error:?}"))
    }
}

fn tree_corpus(errors: &mut Vec<String>) -> Vec<FusionTreeKey> {
    let mut rng = SplitMix(0x2192_11a0);
    let mut raws = Vec::new();
    let huge = [
        1usize << 33,
        u32::MAX as usize + 1,
        usize::MAX,
        usize::MAX - 1,
    ];
    for rank in 0..=MAX_RANK {
        let base = RawTree::canonical(&mut rng, rank, 3, 3);
        raws.push(base.clone());
        // Single-field perturbations exercise each Ord tie-break in turn.
        let mut coupled = base.clone();
        coupled.coupled += 1;
        raws.push(coupled);
        if rank > 0 {
            let mut dual = base.clone();
            dual.is_dual[rank - 1] ^= true;
            raws.push(dual);
            let mut last = base.clone();
            last.uncoupled[rank - 1] += 1;
            raws.push(last);
        }
        if rank > 2 {
            let mut inner = base.clone();
            inner.innerlines[rank - 3] += 1;
            raws.push(inner);
        }
        if rank > 1 {
            let mut vertex = base.clone();
            vertex.vertices[rank - 2] += 1;
            raws.push(vertex);
        }
        // An equal key built again, and further canonical-shape variants.
        raws.push(base);
        raws.push(RawTree::canonical(&mut rng, rank, 5, 1));
        raws.push(RawTree::canonical(&mut rng, rank, 2, 4));
        let mut wide = RawTree::canonical(&mut rng, rank, 2, 2);
        for (slot, sector) in wide.uncoupled.iter_mut().zip(huge.iter().cycle()) {
            *slot = *sector;
        }
        wide.coupled = usize::MAX;
        raws.push(wide);
    }
    // Malformed lengths: every field length is independent of the rank.
    for rank in [0usize, 1, 2, 3, 5, 9, 16] {
        let base = RawTree::canonical(&mut rng, rank, 3, 2);
        let mut more_duals = base.clone();
        more_duals.is_dual.push(true);
        raws.push(more_duals);
        let mut fewer_duals = base.clone();
        fewer_duals.is_dual.pop();
        raws.push(fewer_duals);
        let mut more_inner = base.clone();
        more_inner.innerlines = rng.sectors(rank + 1, 3);
        raws.push(more_inner);
        let mut no_inner = base.clone();
        no_inner.innerlines.clear();
        raws.push(no_inner);
        let mut more_vertices = base.clone();
        more_vertices.vertices = rng.vertices(rank + 2, 2);
        raws.push(more_vertices);
        let mut no_vertices = base.clone();
        no_vertices.vertices.clear();
        raws.push(no_vertices);
    }
    raws.push(RawTree {
        uncoupled: Vec::new(),
        coupled: 7,
        is_dual: vec![true, false],
        innerlines: vec![1, 2, 3],
        vertices: vec![4],
    });
    // Unique (U(1)) trees whose inner lines are not the partial charge sums.
    let charges = [1, -2, 3, 0, -1];
    let id = |charge: i32| U1Irrep::new(charge).sector_id().id();
    let uncoupled: Vec<usize> = charges.iter().copied().map(id).collect();
    let canonical_inner: Vec<usize> = [-1, 2, 2].into_iter().map(id).collect();
    for innerlines in [
        canonical_inner.clone(),
        [-1, 2, 3].into_iter().map(id).collect(),
        [5, -5, 0].into_iter().map(id).collect(),
        canonical_inner,
    ] {
        raws.push(RawTree {
            uncoupled: uncoupled.clone(),
            coupled: id(1),
            is_dual: vec![false, true, false, true, false],
            innerlines,
            vertices: vec![1; 4],
        });
    }
    // Vertices are one-based; zero is rejected at construction.
    let mut zero_vertex = RawTree::canonical(&mut rng, 3, 3, 2);
    zero_vertex.vertices[1] = 0;
    raws.push(zero_vertex);

    let mut trees = Vec::new();
    for raw in &raws {
        match raw.build() {
            Ok(tree) => trees.push(tree),
            Err(error) => errors.push(error),
        }
    }
    trees
}

fn pair_corpus(trees: &[FusionTreeKey]) -> Vec<FusionTreePairKey> {
    let mut rng = SplitMix(0x2192_11a1);
    let mut pairs = Vec::new();
    for index in 0..trees.len() {
        let other = rng.below(trees.len());
        pairs.push(FusionTreePairKey::pair(
            trees[index].clone(),
            trees[other].clone(),
        ));
    }
    // Swapped sides and a repeated pair.
    for index in (0..trees.len()).step_by(7) {
        let other = (index * 5 + 3) % trees.len();
        pairs.push(FusionTreePairKey::pair(
            trees[other].clone(),
            trees[index].clone(),
        ));
        pairs.push(FusionTreePairKey::pair(
            trees[index].clone(),
            trees[other].clone(),
        ));
    }
    pairs
}

fn group_corpus(pairs: &[FusionTreePairKey]) -> Vec<FusionTreeGroupKey> {
    let mut groups: Vec<_> = pairs.iter().map(FusionTreePairKey::group_key).collect();
    groups.push(FusionTreeGroupKey::new([], [], [], []));
    groups.push(FusionTreeGroupKey::from_sector_ids(
        [1, 2],
        [3],
        [true, false],
        [false],
    ));
    // Malformed: dual lengths independent of the sector lengths.
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

fn block_key_corpus(pairs: &[FusionTreePairKey]) -> Vec<BlockKey> {
    let mut keys = vec![
        BlockKey::trivial(),
        BlockKey::opaque([]),
        BlockKey::opaque([0]),
        BlockKey::opaque([1, 2, 3]),
        BlockKey::opaque([u64::MAX]),
        BlockKey::ordinal(5),
        BlockKey::opaque([0, 0]),
    ];
    keys.extend(pairs.iter().step_by(3).cloned().map(BlockKey::from));
    keys.push(BlockKey::trivial());
    keys
}

/// Stable-sorted permutation, and `<` / `=` between sorted neighbours: the
/// total preorder exactly.
fn order_lines<T: Ord + Eq>(out: &mut String, tag: &str, items: &[T]) {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&left, &right| items[left].cmp(&items[right]));
    let mut line = format!("{tag}-order");
    for (position, &index) in order.iter().enumerate() {
        if position > 0 {
            let previous = &items[order[position - 1]];
            let relation = match previous.cmp(&items[index]) {
                Ordering::Less => '<',
                Ordering::Equal => '=',
                Ordering::Greater => unreachable!("sorted"),
            };
            line.push(' ');
            line.push(relation);
        }
        write!(line, " {index}").unwrap();
    }
    writeln!(out, "{line}").unwrap();
}

fn check_consistency<T: Ord + Eq + Hash>(items: &[T]) {
    for left in items {
        for right in items {
            let ordering = left.cmp(right);
            assert_eq!(left == right, ordering == Ordering::Equal);
            assert_eq!(right.cmp(left), ordering.reverse());
            if left == right {
                assert_eq!(fx(left), fx(right));
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MultiplicityTwoRule;

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
    let z2 = [(Z2Irrep::EVEN.into(), 1), (Z2Irrep::ODD.into(), 2)];
    let u1_leg: Vec<_> = (-1..=1).map(|charge| (u1(charge), 2)).collect();
    let su2_leg: Vec<_> = (0..=2).map(|twice| (su2(twice), 1)).collect();
    let su2_sym: Vec<_> = (0..=2).map(|twice| (su2(twice), 2)).collect();
    let u1_sym: Vec<_> = (-1..=1).map(|charge| (u1(charge), 1)).collect();
    let fib = vec![(SectorId::new(0), 1), (SectorId::new(1), 2)];
    let mult = vec![(SectorId::new(0), 1), (SectorId::new(1), 2)];
    let unpack = |structure: Result<Arc<BlockStructure>, CoreError>| structure.unwrap();
    vec![
        (
            "u1-r4",
            unpack(
                hom(
                    &[(u1_leg.clone(), false), (u1_leg.clone(), true)],
                    &[(u1_leg.clone(), false), (u1_leg.clone(), false)],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule),
            ),
        ),
        (
            "u1-r3-empty-domain",
            unpack(
                hom(
                    &[
                        (u1_leg.clone(), false),
                        (u1_leg.clone(), false),
                        (u1_leg, true),
                    ],
                    &[],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule),
            ),
        ),
        (
            "z2-r5",
            unpack(
                hom(
                    &[
                        (z2.to_vec(), false),
                        (z2.to_vec(), true),
                        (z2.to_vec(), false),
                    ],
                    &[(z2.to_vec(), false), (z2.to_vec(), true)],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&Z2FusionRule),
            ),
        ),
        (
            "su2-r4",
            unpack(
                hom(
                    &[(su2_leg.clone(), false), (su2_leg.clone(), true)],
                    &[(su2_leg.clone(), false), (su2_leg, false)],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule),
            ),
        ),
        // Identical codomain and domain legs: swapped pairs built from the
        // structure's own trees name real blocks on the other side (#2192).
        (
            "su2-r4-symmetric",
            unpack(
                hom(
                    &[(su2_sym.clone(), false), (su2_sym.clone(), false)],
                    &[(su2_sym.clone(), false), (su2_sym, false)],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule),
            ),
        ),
        (
            "u1-r4-symmetric",
            unpack(
                hom(
                    &[(u1_sym.clone(), false), (u1_sym.clone(), true)],
                    &[(u1_sym.clone(), false), (u1_sym, true)],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule),
            ),
        ),
        (
            "fibonacci-r4",
            unpack(
                hom(
                    &[
                        (fib.clone(), false),
                        (fib.clone(), false),
                        (fib.clone(), true),
                    ],
                    &[(fib, false)],
                )
                .coupled_subblock_structure_from_leg_degeneracies(&TableRule::fibonacci()),
            ),
        ),
        (
            "generic-multiplicity-r3",
            unpack(
                hom(
                    &[(mult.clone(), false), (mult.clone(), false)],
                    &[(mult, false)],
                )
                .coupled_subblock_structure_from_leg_degeneracies_generic(&MultiplicityTwoRule),
            ),
        ),
    ]
}

fn corpus() -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# #2192 key golden corpus, computed at symtn e5023d98 (Arc-slice keys)."
    )
    .unwrap();
    writeln!(
        out,
        "# fx = rustc_hash::FxBuildHasher.hash_one(key); stored = the FusionTreeKey cached hash."
    )
    .unwrap();

    let mut errors = Vec::new();
    let trees = tree_corpus(&mut errors);
    for error in &errors {
        writeln!(out, "tree-error {error}").unwrap();
    }
    for (index, tree) in trees.iter().enumerate() {
        let stored = stored_tree_hash(tree);
        let mut hasher = rustc_hash::FxHasher::default();
        hasher.write_u64(stored);
        assert_eq!(fx(tree), hasher.finish(), "Hash writes the stored hash");
        writeln!(
            out,
            "tree {index} fx={:016x} stored={stored:016x} {tree:?}",
            fx(tree)
        )
        .unwrap();
    }
    order_lines(&mut out, "tree", &trees);

    let pairs = pair_corpus(&trees);
    for (index, pair) in pairs.iter().enumerate() {
        // Pair and BlockKey Debug derive from the pinned tree Debug; every
        // tenth line keeps the composition pinned.
        if index % 10 == 0 {
            writeln!(out, "pair {index} fx={:016x} {pair:?}", fx(pair)).unwrap();
        } else {
            writeln!(out, "pair {index} fx={:016x}", fx(pair)).unwrap();
        }
    }
    order_lines(&mut out, "pair", &pairs);

    let groups = group_corpus(&pairs);
    for (index, group) in groups.iter().enumerate() {
        writeln!(out, "group {index} fx={:016x} {group:?}", fx(group)).unwrap();
    }
    order_lines(&mut out, "group", &groups);

    let block_keys = block_key_corpus(&pairs);
    for (index, key) in block_keys.iter().enumerate() {
        if index % 10 == 0 {
            writeln!(out, "blockkey {index} fx={:016x} {key:?}", fx(key)).unwrap();
        } else {
            writeln!(out, "blockkey {index} fx={:016x}", fx(key)).unwrap();
        }
    }
    order_lines(&mut out, "blockkey", &block_keys);

    // Real structures: stored block order, sorted lookup order, groups, and
    // lookups of content-equal keys built outside the structure.
    let structures = real_structures();
    let mut merged = Vec::new();
    for (label, structure) in &structures {
        let sectors = structure.sector_structure();
        writeln!(out, "structure {label} blocks={}", structure.block_count()).unwrap();
        for index in 0..structure.block_count() {
            let key = sectors.key(index).unwrap();
            writeln!(out, "  block {index} fx={:016x} {key:?}", fx(key)).unwrap();
            merged.push(key.clone());
        }
        writeln!(out, "  sorted {:?}", sectors.sorted_indices()).unwrap();
        for group in structure.fusion_tree_group_slice() {
            writeln!(
                out,
                "  group fx={:016x} {:?} -> {:?}",
                fx(group.group_key()),
                group.group_key(),
                group.block_indices()
            )
            .unwrap();
        }
        let own: Vec<&FusionTreePairKey> = (0..structure.block_count())
            .map(|index| sectors.key(index).unwrap().as_fusion_tree_pair().unwrap())
            .collect();
        let rebuilt: Vec<FusionTreePairKey> = own.iter().map(|pair| rebuild_pair(pair)).collect();
        for (own, rebuilt) in own.iter().zip(&rebuilt) {
            assert_eq!(*own, rebuilt);
        }
        let swap = |pair: &FusionTreePairKey| {
            FusionTreePairKey::pair(pair.domain_tree().clone(), pair.codomain_tree().clone())
        };
        let swapped_own: Vec<_> = own.iter().map(|pair| swap(pair)).collect();
        let swapped_rebuilt: Vec<_> = rebuilt.iter().map(swap).collect();
        let pair_lookup = |keys: &mut dyn Iterator<Item = &FusionTreePairKey>| {
            indices(keys.map(|key| sectors.find_fusion_tree_pair_index(key)))
        };
        let adjoint_lookup = |keys: &mut dyn Iterator<Item = &FusionTreePairKey>| {
            indices(keys.map(|key| sectors.find_adjoint_fusion_tree_pair_index(key)))
        };
        writeln!(out, "  lookup {}", pair_lookup(&mut rebuilt.iter())).unwrap();
        writeln!(
            out,
            "  lookup-own {}",
            pair_lookup(&mut own.iter().copied())
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-own-blockkey {}",
            indices(
                (0..structure.block_count())
                    .map(|index| sectors.find_index(sectors.key(index).unwrap()))
            )
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-own-structure {}",
            indices(
                own.iter()
                    .map(|key| structure.find_block_index_by_fusion_tree_pair(key))
            )
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-swapped-own {}",
            pair_lookup(&mut swapped_own.iter())
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-swapped-rebuilt {}",
            pair_lookup(&mut swapped_rebuilt.iter())
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-adjoint-own {}",
            adjoint_lookup(&mut own.iter().copied())
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-adjoint-swapped-own {}",
            adjoint_lookup(&mut swapped_own.iter())
        )
        .unwrap();
        writeln!(
            out,
            "  lookup-adjoint-swapped-rebuilt {}",
            adjoint_lookup(&mut swapped_rebuilt.iter())
        )
        .unwrap();
        // Keys owned by every other structure: misses and cross hits.
        for (other_label, other) in &structures {
            if other_label == label {
                continue;
            }
            let other_sectors = other.sector_structure();
            let foreign: Vec<_> = (0..other.block_count())
                .map(|index| other_sectors.key(index).unwrap())
                .collect();
            writeln!(
                out,
                "  lookup-foreign {other_label} pair {} adjoint {} blockkey {}",
                pair_lookup(&mut foreign.iter().map(|key| key.as_fusion_tree_pair().unwrap())),
                adjoint_lookup(&mut foreign.iter().map(|key| key.as_fusion_tree_pair().unwrap())),
                indices(foreign.iter().map(|key| sectors.find_index(key)))
            )
            .unwrap();
        }
        writeln!(
            out,
            "  lookup-corpus pair {} adjoint {} blockkey {}",
            pair_lookup(&mut pairs.iter()),
            adjoint_lookup(&mut pairs.iter()),
            indices(block_keys.iter().map(|key| sectors.find_index(key)))
        )
        .unwrap();
    }
    // `find_index` over the BlockKey corpus in Dense and Opaque structures.
    let keyed = [
        ("dense", SectorStructure::dense(2)),
        (
            "opaque",
            SectorStructure::from_keys(
                2,
                [
                    BlockKey::opaque([1, 2, 3]),
                    BlockKey::ordinal(5),
                    BlockKey::opaque([0]),
                    BlockKey::opaque([u64::MAX]),
                    BlockKey::opaque([]),
                    BlockKey::opaque([9, 9]),
                ],
            )
            .unwrap(),
        ),
        ("empty", SectorStructure::empty(2)),
    ];
    for (label, sectors) in &keyed {
        writeln!(
            out,
            "keyed-structure {label} sorted {:?} lookup-corpus blockkey {} pair {}",
            sectors.sorted_indices(),
            indices(block_keys.iter().map(|key| sectors.find_index(key))),
            indices(
                pairs
                    .iter()
                    .map(|key| sectors.find_fusion_tree_pair_index(key))
            )
        )
        .unwrap();
    }
    // Cross-structure order: keys from different structures and raw keys.
    merged.extend(
        pairs
            .iter()
            .take(12)
            .cloned()
            .map(BlockKey::from)
            .chain(block_keys.iter().take(7).cloned()),
    );
    order_lines(&mut out, "cross", &merged);

    check_consistency(&trees);
    check_consistency(&pairs);
    check_consistency(&groups);
    check_consistency(&block_keys);
    check_consistency(&merged);
    check_views(&trees, &pairs, &groups);
    out
}

/// The borrowed views agree with the owned keys: same hash bits, same order,
/// and `Equivalent` exactly where `Eq` holds.
fn check_views(
    trees: &[FusionTreeKey],
    pairs: &[FusionTreePairKey],
    groups: &[FusionTreeGroupKey],
) {
    for tree in trees {
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
    for left in trees {
        for right in trees {
            assert_eq!(left.as_view().cmp(&right.as_view()), left.cmp(right));
            assert_eq!(left.as_view().equivalent(right), left == right);
        }
    }
    for left in pairs {
        let view = left.as_view();
        assert_eq!(fx(&view), fx(left));
        assert_eq!(view.group(), left.group_key().as_view());
        assert_eq!(fx(&view.group()), fx(&left.group_key()));
        for right in pairs.iter().step_by(5) {
            assert_eq!(view.cmp(&right.as_view()), left.cmp(right));
            assert_eq!(view.equivalent(right), left == right);
        }
    }
    for left in groups {
        assert_eq!(fx(&left.as_view()), fx(left));
        for right in groups {
            assert_eq!(left.as_view().cmp(&right.as_view()), left.cmp(right));
            assert_eq!(left.as_view().equivalent(right), left == right);
        }
    }
}

/// Block indices, `-` for a miss.
fn indices(found: impl Iterator<Item = Option<usize>>) -> String {
    found
        .map(|index| index.map_or_else(|| "-".to_owned(), |index| index.to_string()))
        .collect::<Vec<_>>()
        .join(" ")
}

fn rebuild_pair(pair: &FusionTreePairKey) -> FusionTreePairKey {
    let ids = |sectors: &[SectorId]| sectors.iter().map(|s| s.id()).collect::<Vec<_>>();
    let vertices =
        |tree: &FusionTreeKey| tree.vertices().iter().map(|v| v.get()).collect::<Vec<_>>();
    let (codomain, domain) = (pair.codomain_tree(), pair.domain_tree());
    let mut rebuilt = FusionTreePairKey::try_pair_from_sector_ids(
        ids(codomain.uncoupled()),
        ids(domain.uncoupled()),
        codomain.coupled().id(),
        codomain.is_dual().iter().copied(),
        domain.is_dual().iter().copied(),
        ids(codomain.innerlines()),
        ids(domain.innerlines()),
        vertices(codomain),
        vertices(domain),
    )
    .unwrap();
    if domain.coupled() != codomain.coupled() {
        rebuilt = FusionTreePairKey::pair(rebuilt.codomain_tree().clone(), domain.clone());
    }
    rebuilt
}

#[test]
fn key_identity_matches_golden_corpus() {
    let live = corpus();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    if std::env::var_os("SYMTN_BLESS_KEY_CORPUS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &live).unwrap();
        return;
    }
    let golden = std::fs::read_to_string(&path).unwrap();
    for (line, (want, got)) in golden.lines().zip(live.lines()).enumerate() {
        assert_eq!(got, want, "corpus line {} differs", line + 1);
    }
    assert_eq!(
        golden.lines().count(),
        live.lines().count(),
        "corpus line count differs"
    );
}
