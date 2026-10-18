use super::*;
use core::num::NonZeroUsize;

/// A contiguous run in one of an arena's typed buffers.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct Run {
    start: usize,
    len: usize,
}

impl Run {
    #[inline]
    fn of<T>(self, buffer: &[T]) -> &[T] {
        &buffer[self.start..self.start + self.len]
    }
}

/// One tree: its external runs (uncoupled sectors, duality flags, vertex
/// labels), which the trees of one external tuple share, its own `coupled ‖
/// innerlines` run, and its stored hash.
///
/// Every field has its own length, so a malformed raw key is stored exactly
/// as given.
#[derive(Clone, Copy, Debug)]
struct TreeRow {
    uncoupled: Run,
    duals: Run,
    vertices: Run,
    /// `coupled` at `start`, then the inner lines; never empty.
    lines: Run,
    hash: u64,
}

type TreeRows = SmallVec<[TreeRow; 2]>;

/// Immutable storage of every fusion tree one holder owns (#2192).
///
/// Why typed buffers and not one word buffer: symtn-core forbids `unsafe`,
/// so the `&[bool]` and `&[MultiplicityIndex]` accessors need their own
/// typed storage; one arena therefore costs at most 5 allocations (3
/// buffers, the row table and the `Arc`), empty buffers none, and the row
/// table none for a lone tree.
///
/// Why shared external runs: the trees of one external tuple (an
/// enumeration's, a structure side's, a materialized batch's) differ only in
/// their coupled sector and inner lines (and, for Generic fusion, their
/// vertex labels), so a row points at one stored copy of each equal
/// external run. The bytes per tree are then the row and its own lines, as
/// the base's one externals `Arc` per tuple plus one inner-line `Arc` per
/// tree, instead of the whole rank.
///
/// Single-arena ownership: every holder of keys owns the arena those keys point into. A
/// `SectorStructure` holds handles only into its own arena, a published
/// composed-coefficient entry only into its entry arena.
pub(crate) struct TreeArena {
    // Why a `Vec`: an enumeration's builder grows its per-tree sector buffer,
    // and boxing it would pay a shrinking reallocation, a second full copy,
    // to drop slack the transient arena frees moments later. The external
    // buffers are shared and small.
    sectors: Vec<SectorId>,
    vertices: Box<[MultiplicityIndex]>,
    duals: Box<[bool]>,
    rows: TreeRows,
    /// Set by [`rehome_canonical_with`] only: rows `..n` are the codomain-side
    /// trees and rows `n..` the domain-side trees, each side strictly
    /// increasing by tree `Ord`, then stored hash. A row's side rank is then
    /// its index (minus `n` on the domain side). Every tree pair has a
    /// codomain tree, so a canonical arena has at least one.
    codomain_rows: Option<NonZeroUsize>,
}

impl fmt::Debug for TreeArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TreeArena")
            .field("trees", &self.len())
            .field("codomain_rows", &self.codomain_rows)
            .finish()
    }
}

impl TreeArena {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub(crate) fn codomain_rows(&self) -> Option<usize> {
        self.codomain_rows.map(NonZeroUsize::get)
    }

    #[inline]
    pub(crate) fn uncoupled(&self, index: usize) -> &[SectorId] {
        self.rows[index].uncoupled.of(&self.sectors)
    }

    #[inline]
    pub(crate) fn coupled(&self, index: usize) -> SectorId {
        self.sectors[self.rows[index].lines.start]
    }

    #[inline]
    pub(crate) fn is_dual(&self, index: usize) -> &[bool] {
        self.rows[index].duals.of(&self.duals)
    }

    #[inline]
    pub(crate) fn innerlines(&self, index: usize) -> &[SectorId] {
        &self.rows[index].lines.of(&self.sectors)[1..]
    }

    #[inline]
    pub(crate) fn vertices(&self, index: usize) -> &[MultiplicityIndex] {
        self.rows[index].vertices.of(&self.vertices)
    }

    #[inline]
    pub(crate) fn hash(&self, index: usize) -> u64 {
        self.rows[index].hash
    }

    #[inline]
    pub(crate) fn view(&self, index: usize) -> FusionTreeRef<'_> {
        row_view(
            &self.rows[index],
            &self.sectors,
            &self.duals,
            &self.vertices,
        )
    }

    /// The handle of row `index`.
    #[inline]
    pub(crate) fn key(self: &Arc<Self>, index: usize) -> FusionTreeKey {
        debug_assert!(index < self.len());
        FusionTreeKey {
            arena: Arc::clone(self),
            index,
        }
    }

    /// Stored sector, vertex and duality entries, for the sharing tests.
    #[cfg(test)]
    pub(crate) fn stored_entries(&self) -> (usize, usize, usize) {
        (self.sectors.len(), self.vertices.len(), self.duals.len())
    }

    /// Heap bytes this arena retains: its `Arc` allocation and its buffers.
    pub(crate) fn heap_bytes(&self) -> usize {
        (2 * std::mem::size_of::<usize>())
            .saturating_add(std::mem::size_of::<Self>())
            .saturating_add(self.sectors.capacity() * std::mem::size_of::<SectorId>())
            .saturating_add(std::mem::size_of_val(&*self.vertices))
            .saturating_add(std::mem::size_of_val(&*self.duals))
            .saturating_add(if self.rows.spilled() {
                self.rows.capacity() * std::mem::size_of::<TreeRow>()
            } else {
                0
            })
    }
}

#[inline]
fn row_view<'a>(
    row: &TreeRow,
    sectors: &'a [SectorId],
    duals: &'a [bool],
    vertices: &'a [MultiplicityIndex],
) -> FusionTreeRef<'a> {
    let lines = row.lines.of(sectors);
    FusionTreeRef::with_hash(
        row.uncoupled.of(sectors),
        lines[0],
        row.duals.of(duals),
        &lines[1..],
        row.vertices.of(vertices),
        row.hash,
    )
}

/// Point the tree pairs of `items` at one canonical arena of their own trees
/// (codomain rows, then domain rows, each side in tree `Ord`, then stored
/// hash) and return it.
///
/// Trees equal as keys (`Ord` equal and the same stored hash, which is
/// exactly `Eq`) share one row per side; sides are never deduplicated
/// against each other. Cost: the arena (at most 5 allocations) plus one
/// sort buffer of `2·items` tagged entries.
pub(crate) fn rehome_canonical_with<T>(
    items: &mut [T],
    pair: impl Fn(&T) -> &FusionTreePairKey,
    pair_mut: impl Fn(&mut T) -> &mut FusionTreePairKey,
) -> Arc<TreeArena> {
    // (domain side, item, row): sorted by side, then tree, then input order.
    let count = items.len();
    let tree = |side: bool, item: usize| {
        let pair = pair(&items[item]);
        if side {
            pair.domain_tree.as_view()
        } else {
            pair.codomain_tree.as_view()
        }
    };
    let mut order = (0..2 * count)
        .map(|entry| (entry >= count, entry % count, 0usize))
        .collect::<Vec<_>>();
    order.sort_unstable_by(|left, right| {
        left.0.cmp(&right.0).then_with(|| {
            let (a, b) = (tree(left.0, left.1), tree(right.0, right.1));
            a.cmp(&b)
                .then_with(|| a.stored_hash().cmp(&b.stored_hash()))
                .then_with(|| left.1.cmp(&right.1))
        })
    });
    // Equal keys are adjacent (`Eq` is `Ord`-equal with an equal hash):
    // mark each entry that starts a new row, in the row slot.
    for position in 0..order.len() {
        let (side, item, _) = order[position];
        let starts = position == 0 || {
            let (previous_side, previous_item, _) = order[position - 1];
            previous_side != side || tree(previous_side, previous_item) != tree(side, item)
        };
        order[position].2 = usize::from(starts);
    }
    let mut builder = TreeArenaBuilder::for_views(
        order
            .iter()
            .filter(|entry| entry.2 == 1)
            .map(|&(side, item, _)| tree(side, item)),
    );
    let mut n_codomain = 0;
    for entry in order.iter_mut() {
        let (side, item, starts) = *entry;
        if starts == 1 {
            builder.push_view(tree(side, item));
            if !side {
                n_codomain = builder.len();
            }
        }
        entry.2 = builder.len() - 1;
    }
    let mut arena = builder.into_arena();
    arena.codomain_rows = NonZeroUsize::new(n_codomain);
    let arena = Arc::new(arena);
    for (side, item, row) in order {
        let pair = pair_mut(&mut items[item]);
        let slot = if side {
            &mut pair.domain_tree
        } else {
            &mut pair.codomain_tree
        };
        *slot = arena.key(row);
    }
    arena
}

/// Point `pairs` at one canonical arena of their own trees, which every
/// `SectorStructure` built from exactly these keys then reuses.
pub(crate) fn rehome_canonical(pairs: &mut [FusionTreePairKey]) {
    rehome_canonical_with(pairs, |pair| pair, |pair| pair);
}

/// Which typed buffer a shared external run lives in.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum Shared {
    Uncoupled,
    Duals,
    Vertices,
}

/// Content hash of a run, tagged by its buffer → its first stored copy.
/// Why the stored run keeps its length: runs of different lengths can share
/// a 64-bit hash, so a match must compare length, then content.
type SharedIndex = FxHashMap<(Shared, u64), Run>;

fn content_hash<T: Hash>(kind: Shared, run: &[T]) -> (Shared, u64) {
    use std::hash::Hasher;
    let mut hasher = rustc_hash::FxHasher::default();
    run.hash(&mut hasher);
    (kind, hasher.finish())
}

/// Keep the run just appended at `start..` once: reuse an equal stored run
/// (found by content hash, then compared), else record this one.
fn share_appended<T: Hash + PartialEq>(
    buffer: &mut Vec<T>,
    start: usize,
    kind: Shared,
    index: &mut SharedIndex,
) -> Run {
    let len = buffer.len() - start;
    if len == 0 {
        return Run::default();
    }
    let run = Run { start, len };
    match index.entry(content_hash(kind, &buffer[start..])) {
        Entry::Occupied(stored) => {
            let stored = *stored.get();
            if stored.len == len && stored.of(buffer) == &buffer[start..] {
                buffer.truncate(start);
                return stored;
            }
            // A hash collision: keep this copy unshared.
        }
        Entry::Vacant(slot) => {
            slot.insert(run);
        }
    }
    run
}

/// Store `run` once: reuse an equal stored run, else append it. Looking up
/// before appending keeps an exactly sized buffer from growing.
fn share_slice<T: Hash + PartialEq + Clone>(
    buffer: &mut Vec<T>,
    run: &[T],
    kind: Shared,
    index: &mut SharedIndex,
) -> Run {
    if run.is_empty() {
        return Run::default();
    }
    let new = Run {
        start: buffer.len(),
        len: run.len(),
    };
    match index.entry(content_hash(kind, run)) {
        Entry::Occupied(stored) => {
            let stored = *stored.get();
            if stored.len == run.len() && stored.of(buffer) == run {
                return stored;
            }
            // A hash collision: keep this copy unshared.
        }
        Entry::Vacant(slot) => {
            slot.insert(new);
        }
    }
    buffer.extend_from_slice(run);
    new
}

/// The three external runs of a row.
#[derive(Clone, Copy)]
struct ExternalRuns {
    uncoupled: Run,
    duals: Run,
    vertices: Run,
}

/// Rows pushed in order, frozen once into a [`TreeArena`].
///
/// Two ways to share external runs, neither with a per-row allocation:
/// - [`Self::push_lines_with_externals_of`] gives a new tree the external
///   runs of a row already pushed (an enumeration's or a frame's trees of
///   one tuple);
/// - a sharing builder ([`Self::sharing`], [`Self::for_views`]) looks every
///   run up by content hash, so equal runs are stored once whatever the row
///   order (canonical, entry and Generic arenas).
#[derive(Default)]
pub(crate) struct TreeArenaBuilder {
    sectors: Vec<SectorId>,
    vertices: Vec<MultiplicityIndex>,
    duals: Vec<bool>,
    rows: TreeRows,
    shared: Option<SharedIndex>,
}

impl TreeArenaBuilder {
    /// A builder sized exactly for `trees`; from three trees on a sharing
    /// one, sized by a counting pass, with equal external runs stored once.
    pub(crate) fn for_views<'a>(trees: impl Iterator<Item = FusionTreeRef<'a>> + Clone) -> Self {
        // One or two trees gain nothing from a lookup table: store them as
        // given, in exact buffers.
        if trees.clone().nth(2).is_none() {
            let (mut rows, mut sectors, mut vertices, mut duals) = (0, 0, 0, 0);
            for tree in trees {
                rows += 1;
                sectors += tree.uncoupled().len() + 1 + tree.innerlines().len();
                vertices += tree.vertices().len();
                duals += tree.is_dual().len();
            }
            return Self::exact(rows, sectors, vertices, duals);
        }
        // Size the run table once: runs that differ from the previous tree's
        // bound the distinct runs from above (exactly, when equal runs are
        // adjacent, as in sorted and per-tuple orders).
        let mut bound = 0;
        let mut previous: Option<FusionTreeRef<'_>> = None;
        for tree in trees.clone() {
            bound += [
                previous.is_none_or(|p| p.uncoupled() != tree.uncoupled())
                    && !tree.uncoupled().is_empty(),
                previous.is_none_or(|p| p.is_dual() != tree.is_dual())
                    && !tree.is_dual().is_empty(),
                previous.is_none_or(|p| p.vertices() != tree.vertices())
                    && !tree.vertices().is_empty(),
            ]
            .into_iter()
            .filter(|&new| new)
            .count();
            previous = Some(tree);
        }
        let mut index = SharedIndex::default();
        index.reserve(bound);
        let (mut rows, mut sectors, mut vertices, mut duals) = (0, 0, 0, 0);
        for tree in trees {
            rows += 1;
            sectors += 1 + tree.innerlines().len();
            let mut first_copy = |key: (Shared, u64), len: usize, total: &mut usize| {
                // `contains_key` first: `insert` reserves before it probes,
                // so re-inserting a present run could rehash the table.
                if len > 0 && !index.contains_key(&key) {
                    index.insert(key, Run::default());
                    *total += len;
                }
            };
            first_copy(
                content_hash(Shared::Uncoupled, tree.uncoupled()),
                tree.uncoupled().len(),
                &mut sectors,
            );
            first_copy(
                content_hash(Shared::Duals, tree.is_dual()),
                tree.is_dual().len(),
                &mut duals,
            );
            first_copy(
                content_hash(Shared::Vertices, tree.vertices()),
                tree.vertices().len(),
                &mut vertices,
            );
        }
        index.clear();
        let mut builder = Self::exact(rows, sectors, vertices, duals);
        builder.shared = Some(index);
        builder
    }

    /// A builder that looks every external run up by content; for arenas
    /// whose rows interleave tuples or vary in vertex labels.
    pub(crate) fn sharing() -> Self {
        Self {
            shared: Some(SharedIndex::default()),
            ..Self::default()
        }
    }

    /// A builder with exact capacity for `rows` trees holding the given
    /// field totals, so freezing it moves the buffers without a reallocation.
    pub(crate) fn exact(rows: usize, sectors: usize, vertices: usize, duals: usize) -> Self {
        Self {
            sectors: Vec::with_capacity(sectors),
            vertices: Vec::with_capacity(vertices),
            duals: Vec::with_capacity(duals),
            rows: TreeRows::with_capacity(rows),
            shared: None,
        }
    }

    /// A one-tree builder over buffers already in arena layout (`sectors` is
    /// `uncoupled ‖ coupled ‖ innerlines`), so freezing it allocates only the
    /// `Arc`.
    pub(crate) fn single(
        sectors: Vec<SectorId>,
        len_uncoupled: usize,
        is_dual: Vec<bool>,
        vertices: Vec<MultiplicityIndex>,
    ) -> Self {
        let mut row = TreeRow {
            uncoupled: Run {
                start: 0,
                len: len_uncoupled,
            },
            duals: Run {
                start: 0,
                len: is_dual.len(),
            },
            vertices: Run {
                start: 0,
                len: vertices.len(),
            },
            lines: Run {
                start: len_uncoupled,
                len: sectors.len() - len_uncoupled,
            },
            hash: 0,
        };
        row.hash = hash_row(&row, &sectors, &is_dual, &vertices);
        let mut rows = TreeRows::new();
        rows.push(row);
        Self {
            sectors,
            vertices,
            duals: is_dual,
            rows,
            shared: None,
        }
    }

    /// Reserve exactly for `rows` more trees of the given per-tree widths.
    pub(crate) fn reserve_exact(
        &mut self,
        rows: usize,
        sectors_per_row: usize,
        vertices_per_row: usize,
        duals_per_row: usize,
    ) {
        self.rows.reserve_exact(rows);
        self.sectors.reserve_exact(rows * sectors_per_row);
        self.vertices.reserve_exact(rows * vertices_per_row);
        self.duals.reserve_exact(rows * duals_per_row);
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Row `row` as pushed, with its stored hash.
    #[inline]
    pub(crate) fn view(&self, row: usize) -> FusionTreeRef<'_> {
        row_view(&self.rows[row], &self.sectors, &self.duals, &self.vertices)
    }

    /// Drop every row and keep the buffers, so a reused builder stops
    /// allocating once it has held its widest batch.
    pub(crate) fn clear(&mut self) {
        self.sectors.clear();
        self.vertices.clear();
        self.duals.clear();
        self.rows.clear();
        if let Some(index) = self.shared.as_mut() {
            index.clear();
        }
    }

    /// Append one tree from raw parts and hash it over the stored typed
    /// slices, exactly as [`fusion_tree_key_hash`] does for any key.
    pub(crate) fn push_parts<U, D, I, V>(
        &mut self,
        uncoupled: U,
        coupled: SectorId,
        is_dual: D,
        innerlines: I,
        vertices: V,
    ) -> usize
    where
        U: IntoIterator<Item = SectorId>,
        D: IntoIterator<Item = bool>,
        I: IntoIterator<Item = SectorId>,
        V: IntoIterator<Item = MultiplicityIndex>,
    {
        let (u, d, v) = (self.sectors.len(), self.duals.len(), self.vertices.len());
        self.sectors.extend(uncoupled);
        self.duals.extend(is_dual);
        self.vertices.extend(vertices);
        let externals = match self.shared.as_mut() {
            Some(index) => ExternalRuns {
                uncoupled: share_appended(&mut self.sectors, u, Shared::Uncoupled, index),
                duals: share_appended(&mut self.duals, d, Shared::Duals, index),
                vertices: share_appended(&mut self.vertices, v, Shared::Vertices, index),
            },
            None => ExternalRuns {
                uncoupled: Run {
                    start: u,
                    len: self.sectors.len() - u,
                },
                duals: Run {
                    start: d,
                    len: self.duals.len() - d,
                },
                vertices: Run {
                    start: v,
                    len: self.vertices.len() - v,
                },
            },
        };
        self.push_lines(externals, coupled, innerlines, None)
    }

    /// Copy one tree, keeping its stored hash; returns its row.
    pub(crate) fn push_view(&mut self, tree: FusionTreeRef<'_>) -> usize {
        let externals = match self.shared.as_mut() {
            Some(index) => ExternalRuns {
                uncoupled: share_slice(
                    &mut self.sectors,
                    tree.uncoupled(),
                    Shared::Uncoupled,
                    index,
                ),
                duals: share_slice(&mut self.duals, tree.is_dual(), Shared::Duals, index),
                vertices: share_slice(&mut self.vertices, tree.vertices(), Shared::Vertices, index),
            },
            None => {
                let (u, d, v) = (self.sectors.len(), self.duals.len(), self.vertices.len());
                self.sectors.extend_from_slice(tree.uncoupled());
                self.duals.extend_from_slice(tree.is_dual());
                self.vertices.extend_from_slice(tree.vertices());
                ExternalRuns {
                    uncoupled: Run {
                        start: u,
                        len: tree.uncoupled().len(),
                    },
                    duals: Run {
                        start: d,
                        len: tree.is_dual().len(),
                    },
                    vertices: Run {
                        start: v,
                        len: tree.vertices().len(),
                    },
                }
            }
        };
        self.push_lines(
            externals,
            tree.coupled(),
            tree.innerlines().iter().copied(),
            Some(tree.stored_hash()),
        )
    }

    /// Append a tree with the external runs of row `row` and its own coupled
    /// sector and inner lines.
    pub(crate) fn push_lines_with_externals_of<I>(
        &mut self,
        row: usize,
        coupled: SectorId,
        innerlines: I,
    ) -> usize
    where
        I: IntoIterator<Item = SectorId>,
    {
        let row = &self.rows[row];
        let externals = ExternalRuns {
            uncoupled: row.uncoupled,
            duals: row.duals,
            vertices: row.vertices,
        };
        self.push_lines(externals, coupled, innerlines, None)
    }

    fn push_lines<I>(
        &mut self,
        externals: ExternalRuns,
        coupled: SectorId,
        innerlines: I,
        hash: Option<u64>,
    ) -> usize
    where
        I: IntoIterator<Item = SectorId>,
    {
        let start = self.sectors.len();
        self.sectors.push(coupled);
        self.sectors.extend(innerlines);
        let mut row = TreeRow {
            uncoupled: externals.uncoupled,
            duals: externals.duals,
            vertices: externals.vertices,
            lines: Run {
                start,
                len: self.sectors.len() - start,
            },
            hash: 0,
        };
        row.hash =
            hash.unwrap_or_else(|| hash_row(&row, &self.sectors, &self.duals, &self.vertices));
        self.rows.push(row);
        self.rows.len() - 1
    }

    fn into_arena(self) -> TreeArena {
        TreeArena {
            sectors: self.sectors,
            vertices: self.vertices.into_boxed_slice(),
            duals: self.duals.into_boxed_slice(),
            rows: self.rows,
            codomain_rows: None,
        }
    }

    /// Freeze, moving the buffers.
    pub(crate) fn freeze(self) -> Arc<TreeArena> {
        Arc::new(self.into_arena())
    }
}

/// The key hash of `row`, by [`fusion_tree_key_hash`] over its stored slices.
fn hash_row(
    row: &TreeRow,
    sectors: &[SectorId],
    duals: &[bool],
    vertices: &[MultiplicityIndex],
) -> u64 {
    let lines = row.lines.of(sectors);
    fusion_tree_key_hash(
        row.uncoupled.of(sectors),
        lines[0],
        row.duals.of(duals),
        &lines[1..],
        row.vertices.of(vertices),
    )
}

/// A builder that stores each distinct tree once, found by stored hash and
/// then content; for holders whose trees repeat (a composed-coefficient
/// entry's key and value share their source trees).
///
/// Trees are recorded first and stored at [`Self::rehome`], so the entry
/// arena is sized exactly for its distinct trees and shared runs: no growth
/// and no copy. The record buffers are reused by later holders.
#[doc(hidden)]
#[derive(Default)]
pub struct DedupTreeArenaBuilder {
    /// Each distinct recorded tree, in row order.
    distinct: Vec<FusionTreeKey>,
    rows_by_hash: FxHashMap<u64, usize>,
    /// Row per recorded tree, for [`Self::rehome`].
    inserted: Vec<usize>,
}

impl DedupTreeArenaBuilder {
    fn insert(&mut self, tree: &FusionTreeKey) {
        let hash = tree.arena.hash(tree.index);
        let new_row = self.distinct.len();
        let row = match self.rows_by_hash.get(&hash) {
            Some(&row) if self.distinct[row] == *tree => row,
            Some(_) => {
                // A hash collision: fall back to a scan, which only another
                // collision can make long.
                (0..self.distinct.len())
                    .find(|&row| self.distinct[row] == *tree)
                    .unwrap_or(new_row)
            }
            None => {
                self.rows_by_hash.insert(hash, new_row);
                new_row
            }
        };
        if row == new_row {
            self.distinct.push(tree.clone());
        }
        self.inserted.push(row);
    }

    /// Reserve the record for `pairs` more pairs and `groups` more group
    /// keys.
    pub fn reserve<'p, 'g>(
        &mut self,
        pairs: impl Iterator<Item = &'p FusionTreePairKey>,
        groups: impl Iterator<Item = &'g FusionTreeGroupKey>,
    ) {
        let trees = 2 * (pairs.count() + groups.count());
        self.inserted.reserve(trees);
        self.distinct.reserve(trees);
        self.rows_by_hash.reserve(trees);
    }

    /// Record one pair's trees for the next [`Self::rehome`].
    pub fn insert_pair(&mut self, pair: &FusionTreePairKey) {
        self.insert(&pair.codomain_tree);
        self.insert(&pair.domain_tree);
    }

    /// Record one group key's trees for the next [`Self::rehome`].
    pub fn insert_group(&mut self, group: &FusionTreeGroupKey) {
        self.insert(&group.codomain);
        self.insert(&group.domain);
    }

    /// Store the recorded trees in one exact arena and hand out a re-homer
    /// that yields the new keys in insertion order.
    pub fn rehome(&mut self) -> TreeRehomer {
        let mut builder =
            TreeArenaBuilder::for_views(self.distinct.iter().map(FusionTreeKey::as_view));
        for tree in &self.distinct {
            builder.push_view(tree.as_view());
        }
        let arena = builder.freeze();
        self.distinct.clear();
        self.rows_by_hash.clear();
        TreeRehomer {
            arena,
            rows: std::mem::take(&mut self.inserted),
            next: 0,
        }
    }

    /// Return a finished re-homer's row buffer for reuse.
    pub fn recycle(&mut self, rehomer: TreeRehomer) {
        let mut rows = rehomer.rows;
        rows.clear();
        self.inserted = rows;
    }
}

/// Keys of one frozen dedup arena, handed out in insertion order: call
/// [`Self::pair`] and [`Self::group`] in the order the pairs and groups were
/// recorded. Asking for more trees than were recorded panics.
#[doc(hidden)]
pub struct TreeRehomer {
    arena: Arc<TreeArena>,
    rows: Vec<usize>,
    next: usize,
}

impl TreeRehomer {
    fn next_tree(&mut self) -> FusionTreeKey {
        let row = self.rows[self.next];
        self.next += 1;
        self.arena.key(row)
    }

    /// The re-homed form of the next inserted pair.
    pub fn pair(&mut self) -> FusionTreePairKey {
        let codomain_tree = self.next_tree();
        let domain_tree = self.next_tree();
        FusionTreePairKey {
            codomain_tree,
            domain_tree,
        }
    }

    /// The re-homed form of the next inserted group key.
    pub fn group(&mut self) -> FusionTreeGroupKey {
        let codomain = self.next_tree();
        let domain = self.next_tree();
        FusionTreeGroupKey { codomain, domain }
    }
}
