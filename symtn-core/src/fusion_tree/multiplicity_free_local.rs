use super::*;

/// A fusion tree edited in place, without a stored hash: the working state
/// of one-tree-at-a-time Unique moves (#2192).
///
/// The buffers are in arena layout, so [`Self::freeze`] moves them into a
/// one-row arena (allocating only the `Arc`), and a scratch that
/// [`Self::load`]s tree after tree reuses them: once it has held its widest
/// tree it no longer allocates.
#[derive(Default)]
pub(crate) struct UnhashedFusionTree {
    /// `uncoupled ‖ coupled ‖ innerlines`, the arena's sector layout.
    sectors: Vec<SectorId>,
    len_uncoupled: usize,
    is_dual: Vec<bool>,
    vertices: Vec<MultiplicityIndex>,
}

impl From<FusionTreeKey> for UnhashedFusionTree {
    fn from(key: FusionTreeKey) -> Self {
        let mut tree = Self::default();
        tree.load(key.as_view());
        tree
    }
}

impl UnhashedFusionTree {
    pub(crate) fn with_widths((sectors, vertices, duals): (usize, usize, usize)) -> Self {
        Self {
            sectors: Vec::with_capacity(sectors),
            len_uncoupled: 0,
            is_dual: Vec::with_capacity(duals),
            vertices: Vec::with_capacity(vertices),
        }
    }

    /// Grow each buffer to at least [`Self::with_widths`]' capacity.
    pub(crate) fn reserve_widths(&mut self, (sectors, vertices, duals): (usize, usize, usize)) {
        self.sectors
            .reserve(sectors.saturating_sub(self.sectors.len()));
        self.is_dual
            .reserve(duals.saturating_sub(self.is_dual.len()));
        self.vertices
            .reserve(vertices.saturating_sub(self.vertices.len()));
    }

    /// Overwrite this tree with `tree`, keeping the buffers.
    pub(crate) fn load(&mut self, tree: FusionTreeRef<'_>) {
        self.sectors.clear();
        self.sectors
            .reserve(tree.uncoupled().len() + 1 + tree.innerlines().len());
        self.sectors.extend_from_slice(tree.uncoupled());
        self.sectors.push(tree.coupled());
        self.sectors.extend_from_slice(tree.innerlines());
        self.len_uncoupled = tree.uncoupled().len();
        self.is_dual.clear();
        self.is_dual.extend_from_slice(tree.is_dual());
        self.vertices.clear();
        self.vertices.extend_from_slice(tree.vertices());
    }

    pub(crate) fn vertex_at(&self, position: usize) -> Option<MultiplicityIndex> {
        self.vertices.get(position).copied()
    }

    #[inline]
    pub(crate) fn uncoupled(&self) -> &[SectorId] {
        &self.sectors[..self.len_uncoupled]
    }

    #[inline]
    pub(crate) fn is_dual(&self) -> &[bool] {
        &self.is_dual
    }

    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.len_uncoupled
    }

    /// This tree as a key would compare and hash it.
    pub(crate) fn view(&self) -> FusionTreeRef<'_> {
        FusionTreeRef::new(
            self.uncoupled(),
            self.coupled(),
            &self.is_dual,
            self.innerlines(),
            &self.vertices,
        )
    }

    /// The multiplicity-free projection's vertex gate
    /// (`project_multiplicity_free_tree_pair`).
    pub(crate) fn check_multiplicity_free_vertices(&self) -> Result<(), CoreError> {
        if self
            .vertices
            .iter()
            .any(|&vertex| vertex != MultiplicityIndex::ONE)
        {
            return Err(CoreError::MalformedFusionTree {
                message: "multiplicity-free projection requires vertex label one",
            });
        }
        Ok(())
    }

    /// Every vertex label one, as a multiplicity-free frame materializes it.
    pub(crate) fn reset_multiplicity_free_vertices(&mut self) {
        self.vertices.clear();
        self.vertices
            .resize(self.len_uncoupled.saturating_sub(1), MultiplicityIndex::ONE);
    }

    /// The codomain side of a `bendright`: drop the bent last leg and, above
    /// rank two, the last inner line; the tree now couples to `coupled`
    /// (`duality_manipulations.jl:40-51`, as `MultiplicityFreeBend::local`
    /// and `output_frame` write it for a frame).
    pub(crate) fn shrink_after_bend(&mut self, coupled: SectorId) {
        let rank = self.len_uncoupled;
        let kept_innerlines = if rank > 2 {
            self.innerlines().len() - 1
        } else {
            0
        };
        self.sectors.remove(rank - 1);
        self.len_uncoupled = rank - 1;
        self.sectors[rank - 1] = coupled;
        self.sectors.truncate(rank + kept_innerlines);
        self.is_dual.truncate(rank - 1);
    }

    /// The domain side of a `bendright`: append the bent leg's dual and
    /// `innerline` (the old coupled sector, when the domain had more than one
    /// leg); the tree now couples to `coupled`.
    pub(crate) fn grow_after_bend(
        &mut self,
        leg: SectorId,
        leg_is_dual: bool,
        coupled: SectorId,
        innerline: Option<SectorId>,
    ) {
        let rank = self.len_uncoupled;
        self.sectors.insert(rank, leg);
        self.len_uncoupled = rank + 1;
        self.sectors[rank + 1] = coupled;
        self.sectors.extend(innerline);
        self.is_dual.push(leg_is_dual);
    }

    /// Overwrite this tree with the canonical Unique tree of `uncoupled`
    /// (every inner line the one channel of its left fusion), which must
    /// couple to `coupled`.
    pub(crate) fn rebuild_unique_standard<R>(
        &mut self,
        rule: &R,
        uncoupled: impl IntoIterator<Item = SectorId>,
        coupled: SectorId,
        is_dual: impl IntoIterator<Item = bool>,
    ) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        self.fill_externals(uncoupled, is_dual)?;
        if self.len_uncoupled < 2 {
            return Err(CoreError::MalformedFusionTree {
                message: "direct unique braid rebuild requires at least two sectors",
            });
        }
        self.fold_unique_lines(rule, coupled)
    }

    /// [`Self::rebuild_unique_standard`] at every rank, admitting rank zero
    /// and one as the tree enumeration does: rank zero couples to the
    /// vacuum, rank one to its only leg.
    pub(crate) fn rebuild_unique_tree<R>(
        &mut self,
        rule: &R,
        uncoupled: impl IntoIterator<Item = SectorId>,
        coupled: SectorId,
        is_dual: impl IntoIterator<Item = bool>,
    ) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        self.fill_externals(uncoupled, is_dual)?;
        if self.len_uncoupled >= 2 {
            return self.fold_unique_lines(rule, coupled);
        }
        let admitted = match self.sectors.first() {
            None => coupled == rule.vacuum(),
            Some(&only) => only == coupled,
        };
        if !admitted {
            return Err(CoreError::FusionChannelCount {
                left: coupled,
                right: coupled,
                count: 0,
            });
        }
        self.sectors.push(coupled);
        self.reset_multiplicity_free_vertices();
        Ok(())
    }

    fn fill_externals(
        &mut self,
        uncoupled: impl IntoIterator<Item = SectorId>,
        is_dual: impl IntoIterator<Item = bool>,
    ) -> Result<(), CoreError> {
        self.sectors.clear();
        self.sectors.extend(uncoupled);
        self.len_uncoupled = self.sectors.len();
        self.is_dual.clear();
        self.is_dual.extend(is_dual);
        if self.len_uncoupled != self.is_dual.len() {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree sectors and duality flags must have matching length",
            });
        }
        Ok(())
    }

    /// Append `coupled` and the inner lines of the left fusion of at least
    /// two uncoupled legs, which must end at `coupled`.
    fn fold_unique_lines<R>(&mut self, rule: &R, coupled: SectorId) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        self.sectors.push(coupled);
        let rank = self.len_uncoupled;
        let mut running = self.sectors[0];
        for offset in 0..rank - 1 {
            let next = only_fusion_channel(rule, running, self.sectors[offset + 1])?;
            if offset + 2 == rank {
                if next != coupled {
                    return Err(CoreError::FusionChannelCount {
                        left: coupled,
                        right: coupled,
                        count: 0,
                    });
                }
            } else {
                self.sectors.push(next);
            }
            running = next;
        }
        self.reset_multiplicity_free_vertices();
        Ok(())
    }

    /// Append this tree to `builder`; returns its row.
    pub(crate) fn push_into(&self, builder: &mut TreeArenaBuilder) -> usize {
        builder.push_parts(
            self.uncoupled().iter().copied(),
            self.coupled(),
            self.is_dual.iter().copied(),
            self.innerlines().iter().copied(),
            self.vertices.iter().copied(),
        )
    }

    /// The arena widths of this tree: sectors, vertices, duality flags.
    pub(crate) fn widths(&self) -> (usize, usize, usize) {
        (self.sectors.len(), self.vertices.len(), self.is_dual.len())
    }

    pub(crate) fn freeze(self) -> FusionTreeKey {
        TreeArenaBuilder::single(
            self.sectors,
            self.len_uncoupled,
            self.is_dual,
            self.vertices,
        )
        .freeze()
        .key(0)
    }
}

impl TreeView for UnhashedFusionTree {
    #[inline]
    fn coupled(&self) -> SectorId {
        self.sectors[self.len_uncoupled]
    }
    #[inline]
    fn innerlines(&self) -> &[SectorId] {
        &self.sectors[self.len_uncoupled + 1..]
    }
    fn vertex(&self, position: usize) -> Option<MultiplicityIndex> {
        self.vertex_at(position)
    }
}

#[cfg(test)]
impl MultiplicityFreeTreeData for UnhashedFusionTree {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        UnhashedFusionTree::uncoupled(self)
    }
}

pub(crate) fn apply_unique_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    index: usize,
    inverse: bool,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let kernel = UniqueK(rule);
    let site = ArtinSite::new(&kernel, tree.uncoupled(), index, inverse)?;
    let mut out = InPlaceArtinEdit {
        innerline_count: tree.innerlines().len(),
        vertex_count: tree.vertices.len(),
        index,
        edit: None,
    };
    artin_surgery(&kernel, &site, tree, &mut out)?;
    let (innerline, vertices, coefficient) = out.edit.ok_or(CoreError::MalformedFusionTree {
        message: "a unique-fusion braid emits exactly one tree",
    })?;
    if let Some((position, sector)) = innerline {
        tree.sectors[tree.len_uncoupled + 1 + position] = sector;
    }
    if vertices == ArtinVertices::SwapUnit {
        tree.vertices.swap(index - 1, index);
    }
    tree.sectors[..tree.len_uncoupled].swap(index, index + 1);
    tree.is_dual.swap(index, index + 1);
    Ok(coefficient)
}

/// The one output of an in-place unique-fusion swap, validated against the
/// tree's shape while the surgery still reads it and applied afterwards.
struct InPlaceArtinEdit<S> {
    innerline_count: usize,
    vertex_count: usize,
    index: usize,
    edit: Option<(InnerlineEdit, ArtinVertices, S)>,
}

type InnerlineEdit = Option<(usize, SectorId)>;

impl<S> ArtinWriter<S, CoreError> for InPlaceArtinEdit<S> {
    type Slot = (InnerlineEdit, ArtinVertices);

    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, CoreError> {
        if innerline.is_some_and(|(position, _)| position >= self.innerline_count) {
            return Err(CoreError::MalformedFusionTree {
                message: artin_innerline_message(vertices),
            });
        }
        if vertices == ArtinVertices::SwapUnit && self.vertex_count <= self.index {
            return Err(CoreError::MalformedFusionTree {
                message: "unit braid past the first adjacent pair requires adjacent vertices",
            });
        }
        Ok((innerline, vertices))
    }

    fn finish(
        &mut self,
        (innerline, vertices): Self::Slot,
        coefficient: S,
    ) -> Result<(), CoreError> {
        self.edit = Some((innerline, vertices, coefficient));
        Ok(())
    }
}

/// First-pair multiplicity-free Artin coefficient: TensorKit
/// `artin_braid(f::FusionTree, i; inv)` at `i == 1`
/// (`braiding_manipulations.jl:56`), `inv ? conj(R(b, a, c)) : R(a, b, c)`.
pub(super) fn mf_artin_first_coefficient<R>(
    rule: &R,
    left: SectorId,
    right: SectorId,
    coupled: SectorId,
    inverse: bool,
) -> R::Scalar
where
    R: MultiplicityFreeFusionSymbols,
{
    if inverse {
        rule.r_symbol_scalar(right, left, coupled).conj()
    } else {
        rule.r_symbol_scalar(left, right, coupled)
    }
}

/// The one multiplicity-free Artin coefficient for a braid past the first
/// pair, over the inner-extended lines `[a, b, c, d, e, c′]`: TensorKit
/// `artin_braid(f::FusionTree, i; inv)` (`braiding_manipulations.jl:70-73`),
/// `inv ? conj(R(d, c, e)·F(d, a, b, e, c′, c))·R(d, a, c′)
///      : R(c, d, e)·conj(F(d, a, b, e, c′, c)·R(a, d, c′))`.
pub(super) fn mf_artin_coefficient<R>(
    rule: &R,
    [a, b, c, d, e, c_prime]: [SectorId; 6],
    inverse: bool,
) -> R::Scalar
where
    R: MultiplicityFreeFusionSymbols,
{
    let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
    if inverse {
        let left = rule.r_symbol_scalar(d, c, e);
        let right = rule.r_symbol_scalar(d, a, c_prime);
        (left * f_symbol).conj() * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * (f_symbol * right).conj()
    }
}

#[cfg(test)]
pub(super) trait MultiplicityFreeTreeData: TreeView {
    fn uncoupled(&self) -> &[SectorId];
}

#[cfg(test)]
impl MultiplicityFreeTreeData for FusionTreeKey {
    #[inline]
    fn uncoupled(&self) -> &[SectorId] {
        self.uncoupled()
    }
}

/// The externals of a multiplicity-free tree: uncoupled sectors and duality
/// flags (every vertex is label one).
///
/// Why two forms: a frame split off a key borrows that key's arena (one
/// refcount), while a frame a move derives owns two exact slices; a one-row
/// arena would cost a third allocation and its row table for fields no frame
/// reads.
#[derive(Clone)]
pub(crate) enum MultiplicityFreeTreeFrame {
    Tree(FusionTreeKey),
    Owned {
        uncoupled: Box<[SectorId]>,
        is_dual: Box<[bool]>,
    },
}

impl PartialEq for MultiplicityFreeTreeFrame {
    fn eq(&self, other: &Self) -> bool {
        self.uncoupled() == other.uncoupled() && self.is_dual() == other.is_dual()
    }
}

impl Eq for MultiplicityFreeTreeFrame {}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct MultiplicityFreeTreeLocal {
    pub(super) coupled: SectorId,
    pub(super) innerlines: SectorVec,
}

pub(super) type MultiplicityFreeTreeLocalTerms<S> = Vec<(MultiplicityFreeTreeLocal, S)>;
pub(super) type MultiplicityFreeFoldInverseCache<S> =
    FxHashMap<(SectorId, MultiplicityFreeTreeLocal), MultiplicityFreeTreeLocalTerms<S>>;

mod multiplicity_free_projection {
    use super::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Trees<'a> {
        keys: &'a [FusionTreeKey],
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Pairs<'a> {
        source: PairSource<'a>,
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Tree<'a> {
        key: &'a FusionTreeKey,
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Pair<'a> {
        codomain: &'a FusionTreeKey,
        domain: &'a FusionTreeKey,
    }

    #[derive(Clone, Copy)]
    enum PairSource<'a> {
        Slice(&'a [FusionTreePairKey]),
        Structure {
            structure: &'a BlockStructure,
            indices: &'a [usize],
            orientation: FusionTreePairOrientation,
        },
    }

    pub(crate) struct TreeBatch<'rule, R> {
        rule: &'rule R,
        keys: SmallVec<[FusionTreeKey; 8]>,
    }

    pub(crate) struct PairBatch<'rule, R> {
        rule: &'rule R,
        keys: SmallVec<[FusionTreePairKey; 8]>,
    }

    impl<'rule, R> TreeBatch<'rule, R>
    where
        R: FusionRule,
    {
        pub(crate) fn from_locally_validated<'structure, I>(
            proof: &LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>,
            indices: I,
        ) -> Result<Self, CoreError>
        where
            I: IntoIterator<Item = usize>,
        {
            let keys = indices
                .into_iter()
                .map(|index| {
                    proof
                        .required_fusion_tree_pair_key(index)
                        .map(|key| key.codomain_tree().clone())
                })
                .collect::<Result<SmallVec<_>, _>>()?;
            Ok(Self {
                rule: proof.rule,
                keys,
            })
        }

        pub(crate) fn parts(&self) -> (&R, &[FusionTreeKey]) {
            (self.rule, &self.keys)
        }
    }

    impl<'rule, R> PairBatch<'rule, R>
    where
        R: FusionRule,
    {
        pub(crate) fn from_locally_validated<'structure, I>(
            proof: &LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>,
            indices: I,
        ) -> Result<Self, CoreError>
        where
            I: IntoIterator<Item = usize>,
        {
            let keys = indices
                .into_iter()
                .map(|index| proof.required_fusion_tree_pair_key(index).cloned())
                .collect::<Result<SmallVec<_>, _>>()?;
            Ok(Self {
                rule: proof.rule,
                keys,
            })
        }

        pub(crate) fn parts(&self) -> (&R, &[FusionTreePairKey]) {
            (self.rule, &self.keys)
        }
    }

    impl<'a> Trees<'a> {
        pub(crate) fn checked<R>(rule: &R, trees: &'a [FusionTreeKey]) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            for tree in trees {
                Self::check_vertices(tree)?;
            }
            Ok(Self { keys: trees })
        }

        pub(crate) fn from_validated<R>(
            rule: &R,
            trees: &'a [FusionTreeKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            // Why not rescan vertices: categorical validation already proved
            // every multiplicity-free N-symbol bound, hence label ONE. The
            // `_proven` callers receive keys from a LocallyValidated block.
            validate_multiplicity_free_execution_style(rule)?;
            Ok(Self { keys: trees })
        }

        pub(crate) fn tree_at(&self, index: usize) -> Option<Tree<'a>> {
            self.keys.get(index).map(|key| Tree { key })
        }

        fn check_vertices(tree: &FusionTreeKey) -> Result<(), CoreError> {
            check_vertices(tree)
        }
    }

    impl<'a> Pairs<'a> {
        pub(crate) fn checked<R>(
            rule: &R,
            pairs: &'a [FusionTreePairKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            for pair in pairs {
                check_vertices(pair.codomain_tree())?;
                check_vertices(pair.domain_tree())?;
            }
            Ok(Self {
                source: PairSource::Slice(pairs),
            })
        }

        pub(crate) fn from_validated<R>(
            rule: &R,
            pairs: &'a [FusionTreePairKey],
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            // Pair validation proves the same vertex invariant for both trees
            // in the single categorical validation pass.
            validate_multiplicity_free_execution_style(rule)?;
            Ok(Self {
                source: PairSource::Slice(pairs),
            })
        }

        pub(crate) fn checked_structure<R>(
            rule: &R,
            structure: &'a BlockStructure,
            indices: &'a [usize],
            orientation: FusionTreePairOrientation,
        ) -> Result<Self, CoreError>
        where
            R: FusionRule,
        {
            validate_multiplicity_free_execution_style(rule)?;
            let projection = Self {
                source: PairSource::Structure {
                    structure,
                    indices,
                    orientation,
                },
            };
            for (index, &block_index) in indices.iter().enumerate() {
                let block = structure.block(block_index)?;
                if !matches!(block.key(), BlockKey::FusionTree(_)) {
                    return Err(CoreError::ExpectedFusionTreePairKey {
                        actual: block.key().kind(),
                    });
                }
                let pair = projection
                    .pair_at(index)
                    .expect("validated block index and key kind produce a pair");
                let codomain = validate_fusion_tree_for_rule(rule, pair.codomain().key())?;
                let domain = validate_fusion_tree_for_rule(rule, pair.domain().key())?;
                if codomain.key.coupled() != domain.key.coupled() {
                    return Err(CoreError::MalformedFusionTree {
                        message: "fusion tree pair requires matching coupled sectors",
                    });
                }
                check_vertices(pair.codomain().key())?;
                check_vertices(pair.domain().key())?;
            }
            Ok(projection)
        }

        pub(crate) fn has_strictly_ordered_structure_indices(&self) -> bool {
            match self.source {
                // SectorStructure rejects duplicate keys; swapping whole trees
                // preserves that uniqueness for the adjoint projection.
                PairSource::Structure { indices, .. } => {
                    indices.windows(2).all(|pair| pair[0] < pair[1])
                }
                PairSource::Slice(_) => false,
            }
        }

        pub(crate) fn len(&self) -> usize {
            match self.source {
                PairSource::Slice(keys) => keys.len(),
                PairSource::Structure { indices, .. } => indices.len(),
            }
        }

        pub(crate) fn pair_at(&self, index: usize) -> Option<Pair<'a>> {
            match self.source {
                PairSource::Slice(keys) => keys.get(index).map(|key| Pair {
                    codomain: key.codomain_tree(),
                    domain: key.domain_tree(),
                }),
                PairSource::Structure {
                    structure,
                    indices,
                    orientation,
                } => {
                    let block = structure.block(*indices.get(index)?).ok()?;
                    let BlockKey::FusionTree(key) = block.key() else {
                        return None;
                    };
                    let (codomain, domain) = match orientation {
                        FusionTreePairOrientation::Direct => {
                            (key.codomain_tree(), key.domain_tree())
                        }
                        FusionTreePairOrientation::Adjoint => {
                            (key.domain_tree(), key.codomain_tree())
                        }
                    };
                    Some(Pair { codomain, domain })
                }
            }
        }
    }

    fn check_vertices(tree: &FusionTreeKey) -> Result<(), CoreError> {
        if tree
            .vertices()
            .iter()
            .any(|&vertex| vertex != MultiplicityIndex::ONE)
        {
            return Err(CoreError::MalformedFusionTree {
                message: "multiplicity-free projection requires vertex label one",
            });
        }
        Ok(())
    }

    impl<'a> Tree<'a> {
        pub(crate) fn key(self) -> &'a FusionTreeKey {
            self.key
        }
    }

    impl<'a> Pair<'a> {
        pub(crate) fn materialize(self) -> FusionTreePairKey {
            FusionTreePairKey::pair(self.codomain.clone(), self.domain.clone())
        }

        pub(crate) fn codomain(self) -> Tree<'a> {
            Tree { key: self.codomain }
        }

        pub(crate) fn domain(self) -> Tree<'a> {
            Tree { key: self.domain }
        }
    }
}

pub(crate) use multiplicity_free_projection::{
    Pair as ValidatedMultiplicityFreeTreePair, PairBatch as ValidatedMultiplicityFreePairBatch,
    Pairs as MultiplicityFreePairProjection, Tree as ValidatedMultiplicityFreeTree,
    TreeBatch as ValidatedMultiplicityFreeTreeBatch, Trees as MultiplicityFreeTreeProjection,
};

type MultiplicityFreeArtinTerms<S> = SmallVec<[(MultiplicityFreeTreeLocal, S); 2]>;

impl MultiplicityFreeTreeLocal {
    pub(super) fn from_proven(tree: ValidatedMultiplicityFreeTree<'_>) -> Self {
        let tree = tree.key();
        Self {
            coupled: tree.coupled(),
            innerlines: tree.innerlines().iter().copied().collect(),
        }
    }
}

impl MultiplicityFreeTreeFrame {
    /// A frame over new externals: one lone arena.
    pub(super) fn from_externals<U, D>(uncoupled: U, is_dual: D) -> Self
    where
        U: IntoIterator<Item = SectorId>,
        D: IntoIterator<Item = bool>,
    {
        Self::Owned {
            uncoupled: uncoupled.into_iter().collect(),
            is_dual: is_dual.into_iter().collect(),
        }
    }

    #[inline]
    pub(super) fn uncoupled(&self) -> &[SectorId] {
        match self {
            Self::Tree(tree) => tree.uncoupled(),
            Self::Owned { uncoupled, .. } => uncoupled,
        }
    }

    #[inline]
    pub(super) fn is_dual(&self) -> &[bool] {
        match self {
            Self::Tree(tree) => tree.is_dual(),
            Self::Owned { is_dual, .. } => is_dual,
        }
    }

    fn from_tree(tree: &FusionTreeKey) -> Self {
        Self::Tree(tree.clone())
    }

    pub(super) fn split(
        tree: ValidatedMultiplicityFreeTree<'_>,
    ) -> (Self, MultiplicityFreeTreeLocal) {
        let key = tree.key();
        (
            Self::from_tree(key),
            MultiplicityFreeTreeLocal::from_proven(tree),
        )
    }

    pub(super) fn matches_tree(&self, tree: &FusionTreeKey) -> bool {
        // Why not split and compare frames: every source shares these slices,
        // while collecting rank > 8 frames would allocate once per source.
        self.uncoupled() == tree.uncoupled() && self.is_dual() == tree.is_dual()
    }

    pub(super) fn materialize(&self, local: MultiplicityFreeTreeLocal) -> FusionTreeKey {
        let mut builder = TreeArenaBuilder::exact(
            1,
            self.uncoupled().len() + 1 + local.innerlines.len(),
            self.uncoupled().len().saturating_sub(1),
            self.is_dual().len(),
        );
        self.push_materialized(&mut builder, &local);
        builder.freeze().key(0)
    }

    /// Every tree of `locals` over these externals, in one arena; tree `i`
    /// is row `i`.
    pub(super) fn materialize_all<'a>(
        &self,
        locals: impl ExactSizeIterator<Item = &'a MultiplicityFreeTreeLocal> + Clone,
    ) -> Arc<TreeArena> {
        let (uncoupled, is_dual) = (self.uncoupled(), self.is_dual());
        let count = locals.len();
        let inner = locals
            .clone()
            .map(|local| local.innerlines.len())
            .sum::<usize>();
        // Every tree shares the first one's externals.
        let shared = count.min(1);
        let mut builder = TreeArenaBuilder::exact(
            count,
            count + inner + shared * uncoupled.len(),
            shared * uncoupled.len().saturating_sub(1),
            shared * is_dual.len(),
        );
        for (index, local) in locals.enumerate() {
            if index == 0 {
                self.push_materialized(&mut builder, local);
            } else {
                builder.push_lines_with_externals_of(
                    0,
                    local.coupled,
                    local.innerlines.iter().copied(),
                );
            }
        }
        builder.freeze()
    }

    /// Push the tree of `local` over these externals.
    pub(super) fn push_materialized(
        &self,
        builder: &mut TreeArenaBuilder,
        local: &MultiplicityFreeTreeLocal,
    ) -> usize {
        let (uncoupled, is_dual) = (self.uncoupled(), self.is_dual());
        let vertices = uncoupled.len().saturating_sub(1);
        builder.push_parts(
            uncoupled.iter().copied(),
            local.coupled,
            is_dual.iter().copied(),
            local.innerlines.iter().copied(),
            std::iter::repeat_n(MultiplicityIndex::ONE, vertices),
        )
    }
}

pub(crate) fn project_multiplicity_free_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<(MultiplicityFreeTreeFrame, MultiplicityFreeTreeLocal), CoreError>
where
    R: FusionRule,
{
    let projection = MultiplicityFreeTreeProjection::checked(rule, std::slice::from_ref(tree))?;
    Ok(MultiplicityFreeTreeFrame::split(
        projection
            .tree_at(0)
            .expect("single-tree projection contains index zero"),
    ))
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct MultiplicityFreeTreePairFrame {
    pub(super) codomain: MultiplicityFreeTreeFrame,
    pub(super) domain: MultiplicityFreeTreeFrame,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct MultiplicityFreeTreePairLocal {
    pub(super) codomain: MultiplicityFreeTreeLocal,
    pub(super) domain: MultiplicityFreeTreeLocal,
}

impl MultiplicityFreeTreePairLocal {
    pub(super) fn from_proven(tree_pair: ValidatedMultiplicityFreeTreePair<'_>) -> Self {
        Self {
            codomain: MultiplicityFreeTreeLocal::from_proven(tree_pair.codomain()),
            domain: MultiplicityFreeTreeLocal::from_proven(tree_pair.domain()),
        }
    }
}

/// Pair `index` of an arena from [`MultiplicityFreeTreePairFrame::materialize_all`].
#[inline]
pub(super) fn materialized_pair(arena: &Arc<TreeArena>, index: usize) -> FusionTreePairKey {
    FusionTreePairKey::pair(arena.key(2 * index), arena.key(2 * index + 1))
}

impl MultiplicityFreeTreePairFrame {
    pub(super) fn split(
        tree_pair: ValidatedMultiplicityFreeTreePair<'_>,
    ) -> (Self, MultiplicityFreeTreePairLocal) {
        let codomain = MultiplicityFreeTreeFrame::from_tree(tree_pair.codomain().key());
        let domain = MultiplicityFreeTreeFrame::from_tree(tree_pair.domain().key());
        (
            Self { codomain, domain },
            MultiplicityFreeTreePairLocal::from_proven(tree_pair),
        )
    }

    pub(super) fn matches_tree_pair_ref(
        &self,
        tree_pair: ValidatedMultiplicityFreeTreePair<'_>,
    ) -> bool {
        self.codomain.matches_tree(tree_pair.codomain().key())
            && self.domain.matches_tree(tree_pair.domain().key())
    }

    /// Every pair of `locals` over these externals, all trees in one arena;
    /// see [`materialized_pair`].
    pub(super) fn materialize_all<'a>(
        &self,
        locals: impl ExactSizeIterator<Item = &'a MultiplicityFreeTreePairLocal> + Clone,
    ) -> Arc<TreeArena> {
        let (codomain, domain) = (self.codomain.uncoupled(), self.domain.uncoupled());
        let count = locals.len();
        let inner = locals
            .clone()
            .map(|local| local.codomain.innerlines.len() + local.domain.innerlines.len())
            .sum::<usize>();
        // Every pair shares the first pair's externals per side.
        let shared = count.min(1);
        let mut builder = TreeArenaBuilder::exact(
            2 * count,
            2 * count + inner + shared * (codomain.len() + domain.len()),
            shared * (codomain.len().saturating_sub(1) + domain.len().saturating_sub(1)),
            shared * (self.codomain.is_dual().len() + self.domain.is_dual().len()),
        );
        for (index, local) in locals.enumerate() {
            if index == 0 {
                self.codomain
                    .push_materialized(&mut builder, &local.codomain);
                self.domain.push_materialized(&mut builder, &local.domain);
            } else {
                for (row, side) in [(0, &local.codomain), (1, &local.domain)] {
                    builder.push_lines_with_externals_of(
                        row,
                        side.coupled,
                        side.innerlines.iter().copied(),
                    );
                }
            }
        }
        builder.freeze()
    }

    /// Both trees in one arena.
    pub(super) fn materialize(&self, local: MultiplicityFreeTreePairLocal) -> FusionTreePairKey {
        let (codomain, domain) = (self.codomain.uncoupled(), self.domain.uncoupled());
        let mut builder = TreeArenaBuilder::exact(
            2,
            codomain.len()
                + domain.len()
                + 2
                + local.codomain.innerlines.len()
                + local.domain.innerlines.len(),
            codomain.len().saturating_sub(1) + domain.len().saturating_sub(1),
            self.codomain.is_dual().len() + self.domain.is_dual().len(),
        );
        let codomain = self
            .codomain
            .push_materialized(&mut builder, &local.codomain);
        let domain = self.domain.push_materialized(&mut builder, &local.domain);
        let arena = builder.freeze();
        FusionTreePairKey::pair(arena.key(codomain), arena.key(domain))
    }
}

pub(super) fn project_multiplicity_free_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(MultiplicityFreeTreePairFrame, MultiplicityFreeTreePairLocal), CoreError>
where
    R: FusionRule,
{
    let projection =
        MultiplicityFreePairProjection::checked(rule, std::slice::from_ref(tree_pair))?;
    Ok(MultiplicityFreeTreePairFrame::split(
        projection
            .pair_at(0)
            .expect("single-pair projection contains index zero"),
    ))
}

pub(super) struct PreparedMultiplicityFreeArtin {
    pub(super) output_frame: MultiplicityFreeTreeFrame,
    site: ArtinSite,
}

pub(super) fn prepare_multiplicity_free_artin<R>(
    rule: &R,
    frame: &MultiplicityFreeTreeFrame,
    index: usize,
    inverse: bool,
) -> Result<PreparedMultiplicityFreeArtin, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
{
    let site = ArtinSite::new(&SimpleK(rule), frame.uncoupled(), index, inverse)?;
    let swapped = |position: usize| {
        if position == index {
            index + 1
        } else if position == index + 1 {
            index
        } else {
            position
        }
    };
    let (uncoupled, is_dual) = (frame.uncoupled(), frame.is_dual());
    let output_frame = MultiplicityFreeTreeFrame::from_externals(
        (0..uncoupled.len()).map(|position| uncoupled[swapped(position)]),
        (0..is_dual.len()).map(|position| is_dual[swapped(position)]),
    );
    Ok(PreparedMultiplicityFreeArtin { output_frame, site })
}

impl PreparedMultiplicityFreeArtin {
    // External frame data stays out of this kernel: the block runner must enumerate
    // locals from this prepared frame rather than rebuilding full tree keys.
    #[inline]
    pub(super) fn apply<R>(
        &self,
        rule: &R,
        tree: &MultiplicityFreeTreeLocal,
    ) -> Result<MultiplicityFreeArtinTerms<R::Scalar>, CoreError>
    where
        R: MultiplicityFreeFusionSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
    {
        let mut terms = SmallVec::new();
        self.apply_into(rule, tree, &mut terms)?;
        Ok(terms)
    }

    /// [`Self::apply`] appending to a caller-owned buffer, so a block move
    /// reuses one buffer for every basis row.
    #[inline]
    pub(super) fn apply_into<R, T>(
        &self,
        rule: &R,
        tree: &MultiplicityFreeTreeLocal,
        terms: &mut T,
    ) -> Result<(), CoreError>
    where
        R: MultiplicityFreeFusionSymbols,
        R::Scalar: Clone + Mul<Output = R::Scalar>,
        T: Extend<(MultiplicityFreeTreeLocal, R::Scalar)>,
    {
        artin_surgery(
            &SimpleK(rule),
            &self.site,
            tree,
            &mut LocalArtinTerms { tree, terms },
        )
    }
}

/// Multiplicity-free Artin outputs as compact locals.
struct LocalArtinTerms<'t, T> {
    tree: &'t MultiplicityFreeTreeLocal,
    terms: &'t mut T,
}

impl<S, T: Extend<(MultiplicityFreeTreeLocal, S)>> ArtinWriter<S, CoreError>
    for LocalArtinTerms<'_, T>
{
    type Slot = MultiplicityFreeTreeLocal;

    #[inline(always)]
    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, CoreError> {
        let mut innerlines: SectorVec = self.tree.innerlines.iter().copied().collect();
        if let Some((position, sector)) = innerline {
            *innerlines
                .get_mut(position)
                .ok_or(CoreError::MalformedFusionTree {
                    message: artin_innerline_message(vertices),
                })? = sector;
        }
        Ok(MultiplicityFreeTreeLocal {
            coupled: self.tree.coupled,
            innerlines,
        })
    }

    #[inline(always)]
    fn finish(&mut self, local: Self::Slot, coefficient: S) -> Result<(), CoreError> {
        self.terms.extend(std::iter::once((local, coefficient)));
        Ok(())
    }
}

#[expect(
    clippy::type_complexity,
    reason = "the SmallVec inline capacity is part of this local braid allocation contract"
)]
pub(crate) fn multiplicity_free_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<SmallVec<[(FusionTreeKey, R::Scalar); 2]>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    // Why not keep a second full-key formula for block transforms: the compact
    // block basis must reuse exactly the same F/R ordering and error gates.
    let (frame, local) = project_multiplicity_free_tree(rule, tree)?;
    let prepared = prepare_multiplicity_free_artin(rule, &frame, index, inverse)?;
    let terms = prepared.apply(rule, &local)?;
    Ok(terms
        .into_iter()
        .map(|(local, coefficient)| (prepared.output_frame.materialize(local), coefficient))
        .collect())
}
