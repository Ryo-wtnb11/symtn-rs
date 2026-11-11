//! Caller-owned scratch for one-tree-at-a-time fusion-tree paths (#2192).
//!
//! TensorKit's `FusionTree` is isbits, so its Unique `permute` and `split`
//! (`_trace_permute!(::UniqueFusion)`, `src/tensors/tensoroperations.jl:242`)
//! build every intermediate tree on the stack. A symtn key owns an arena, so
//! building one per intermediate tree cost several allocations per term. These
//! scratches hold the trees a path builds and reads back without keeping
//! them; the caller owns one per call and reuses it for every tree, so once it
//! has held its widest tree the path allocates nothing per tree.

use super::*;

/// Reusable rows for trees a path builds only to read back as
/// [`FusionTreeRef`]s: the halves of a split, a destination to look up.
#[doc(hidden)]
#[derive(Default)]
pub struct TreeScratch {
    rows: TreeArenaBuilder,
}

impl TreeScratch {
    /// A scratch sized for splitting both trees of a pair of total `rank`.
    pub fn with_rank(rank: usize) -> Self {
        let (sectors, vertices, duals) = tree_widths(rank);
        Self {
            rows: TreeArenaBuilder::exact(4, 2 * sectors, 2 * vertices, 2 * duals),
        }
    }

    /// Drop every row, keeping the buffers.
    #[inline]
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    /// Row `row`, hashed as a key holding the same tree.
    #[inline]
    pub fn view(&self, row: usize) -> FusionTreeRef<'_> {
        self.rows.view(row)
    }

    #[inline]
    pub(crate) fn builder(&mut self) -> &mut TreeArenaBuilder {
        &mut self.rows
    }

    /// Rows `first` and `second` as keys of one arena.
    pub(crate) fn into_keys(self, first: usize, second: usize) -> (FusionTreeKey, FusionTreeKey) {
        let arena = self.rows.freeze();
        (arena.key(first), arena.key(second))
    }
}

/// The working tree pair of one Unique-fusion tree-pair operation, edited in
/// place by every bend and braid step
/// ([`PreparedTreePairOperation::execute_unique_rigid_into`]).
#[doc(hidden)]
#[derive(Default)]
pub struct UniqueTreePairScratch {
    pub(crate) codomain: UnhashedFusionTree,
    pub(crate) domain: UnhashedFusionTree,
    /// The output buffer of a direct braid rebuild, swapped in afterwards.
    pub(crate) spare: UnhashedFusionTree,
}

impl UniqueTreePairScratch {
    /// A scratch sized for tree pairs of total `rank`: no growth while it
    /// repartitions every leg into one tree.
    pub fn with_rank(rank: usize) -> Self {
        let tree = || UnhashedFusionTree::with_widths(tree_widths(rank));
        Self {
            codomain: tree(),
            domain: tree(),
            spare: tree(),
        }
    }

    /// Grow every tree to [`Self::with_rank`]'s widths for `rank`, keeping
    /// wider buffers: a caller-owned scratch reused across compiles allocates
    /// only when a compile is wider than every earlier one.
    pub fn reserve_rank(&mut self, rank: usize) {
        let widths = tree_widths(rank);
        for tree in [&mut self.codomain, &mut self.domain, &mut self.spare] {
            tree.reserve_widths(widths);
        }
    }

    /// Bytes of the three trees' buffer capacity.
    pub fn retained_bytes(&self) -> usize {
        [&self.codomain, &self.domain, &self.spare]
            .into_iter()
            .map(UnhashedFusionTree::retained_bytes)
            .fold(0, usize::saturating_add)
    }

    /// The pair the last execution left, as a key holding it would compare.
    pub fn pair_view(&self) -> FusionTreePairRef<'_> {
        FusionTreePairRef::new(self.codomain.view(), self.domain.view())
    }

    pub(crate) fn load(&mut self, pair: &FusionTreePairKey) {
        self.codomain.load(pair.codomain_tree().as_view());
        self.domain.load(pair.domain_tree().as_view());
    }

    /// The pair as a key: both trees in one exactly sized arena.
    pub(crate) fn materialize(&self) -> FusionTreePairKey {
        let (codomain, domain) = (self.codomain.widths(), self.domain.widths());
        let mut builder = TreeArenaBuilder::exact(
            2,
            codomain.0 + domain.0,
            codomain.1 + domain.1,
            codomain.2 + domain.2,
        );
        let codomain = self.codomain.push_into(&mut builder);
        let domain = self.domain.push_into(&mut builder);
        let arena = builder.freeze();
        FusionTreePairKey::pair(arena.key(codomain), arena.key(domain))
    }
}

/// The widest buffers a tree on `rank` legs, or a split half of one, needs:
/// sectors (uncoupled, coupled and inner lines, plus a split's leading
/// line), vertices, duality flags.
fn tree_widths(rank: usize) -> (usize, usize, usize) {
    (2 * rank + 1, rank, rank + 1)
}
