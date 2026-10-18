use super::*;

/// External-sector identity of a fusion-tree pair: the uncoupled sectors and
/// duality flags of both trees.
///
/// Held as the pair's two tree handles, of which only `uncoupled` and
/// `is_dual` are ever read, so [`FusionTreePairKey::group_key`] is two
/// refcount bumps across arenas too. Eq, Ord, Hash and Debug are written by
/// hand to match what deriving them over four slices would give: order
/// interleaved (codomain sectors, domain sectors, codomain duals, domain
/// duals) and the structural slice hash, not a stored one.
#[derive(Clone)]
pub struct FusionTreeGroupKey {
    pub(crate) codomain: FusionTreeKey,
    pub(crate) domain: FusionTreeKey,
}

impl PartialEq for FusionTreeGroupKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}

impl Eq for FusionTreeGroupKey {}

impl Ord for FusionTreeGroupKey {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl PartialOrd for FusionTreeGroupKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for FusionTreeGroupKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_view().hash(state);
    }
}

impl fmt::Debug for FusionTreeGroupKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusionTreeGroupKey")
            .field("codomain_uncoupled", &self.codomain_uncoupled())
            .field("domain_uncoupled", &self.domain_uncoupled())
            .field("codomain_is_dual", &self.codomain_is_dual())
            .field("domain_is_dual", &self.domain_is_dual())
            .finish()
    }
}

impl FusionTreeGroupKey {
    /// The codomain tree handle, whose arena its holder charges (single-arena ownership).
    #[doc(hidden)]
    #[inline]
    pub fn codomain_tree(&self) -> &FusionTreeKey {
        &self.codomain
    }

    /// The domain tree handle.
    #[doc(hidden)]
    #[inline]
    pub fn domain_tree(&self) -> &FusionTreeKey {
        &self.domain
    }

    pub fn new<Codomain, Domain, CodomainDual, DomainDual>(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
    ) -> Self
    where
        Codomain: IntoIterator<Item = SectorId>,
        Domain: IntoIterator<Item = SectorId>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
    {
        // The coupled sector, inner lines and vertices of these two rows are
        // never read through a group key.
        let (codomain_uncoupled, domain_uncoupled) =
            (codomain_uncoupled.into_iter(), domain_uncoupled.into_iter());
        let (codomain_is_dual, domain_is_dual) =
            (codomain_is_dual.into_iter(), domain_is_dual.into_iter());
        let mut builder = TreeArenaBuilder::exact(
            2,
            codomain_uncoupled.size_hint().0 + domain_uncoupled.size_hint().0 + 2,
            0,
            codomain_is_dual.size_hint().0 + domain_is_dual.size_hint().0,
        );
        builder.push_parts(
            codomain_uncoupled,
            SectorId::new(0),
            codomain_is_dual,
            [],
            [],
        );
        builder.push_parts(domain_uncoupled, SectorId::new(0), domain_is_dual, [], []);
        let arena = builder.freeze();
        Self {
            codomain: arena.key(0),
            domain: arena.key(1),
        }
    }

    pub fn from_sector_ids<Codomain, Domain, CodomainDual, DomainDual>(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
    {
        Self::new(
            codomain_uncoupled.into_iter().map(SectorId::new),
            domain_uncoupled.into_iter().map(SectorId::new),
            codomain_is_dual,
            domain_is_dual,
        )
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &[SectorId] {
        self.codomain.uncoupled()
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &[SectorId] {
        self.domain.uncoupled()
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &[bool] {
        self.codomain.is_dual()
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &[bool] {
        self.domain.is_dual()
    }
}

/// Identity of one fusion tree: a handle into an immutable tree arena
/// shared by every tree of its holder (#2192).
///
/// Eq, Ord, Hash and Debug depend only on the five identity fields (and the
/// hash stored with them), never on which arena holds them, so equal trees
/// from different structures compare, order and hash identically. A clone
/// is one refcount increment.
///
/// Retention: a key keeps its whole arena alive. A key taken from a block
/// structure keeps that structure's tree arena (all of its distinct trees)
/// resident for as long as the key lives.
// Why the hash is stored: keys are built once per structure and looked up
// many times per publication, so an `O(K)` hash per lookup (TensorKit's
// `hash(::FusionTree)` per `Dict` access) is avoidable constant work. It is
// a pure function of the five identity fields, is excluded from `Ord` and
// `Debug`, and is never persisted.
#[derive(Clone)]
pub struct FusionTreeKey {
    pub(crate) arena: Arc<TreeArena>,
    pub(crate) index: usize,
}

#[inline]
pub(crate) fn fusion_tree_key_hash(
    uncoupled: &[SectorId],
    coupled: SectorId,
    is_dual: &[bool],
    innerlines: &[SectorId],
    vertices: &[MultiplicityIndex],
) -> u64 {
    use std::hash::Hasher;
    let mut hasher = rustc_hash::FxHasher::default();
    uncoupled.hash(&mut hasher);
    coupled.hash(&mut hasher);
    is_dual.hash(&mut hasher);
    innerlines.hash(&mut hasher);
    vertices.hash(&mut hasher);
    hasher.finish()
}

impl Hash for FusionTreeKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.arena.hash(self.index));
    }
}

// Why the impls go through the view: one comparison authority for owned
// keys and borrowed lookups, so `Equivalent` agrees with `Eq` by construction.
// Arenas are immutable, so one row of one arena is equal to itself.
impl PartialEq for FusionTreeKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (self.index == other.index && Arc::ptr_eq(&self.arena, &other.arena))
            || self.as_view() == other.as_view()
    }
}

impl Eq for FusionTreeKey {}

impl Ord for FusionTreeKey {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl PartialOrd for FusionTreeKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for FusionTreeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_view().fmt_named("FusionTreeKey", f)
    }
}

impl FusionTreeKey {
    /// Heap bytes of the tree arena this key keeps alive. Under single-arena
    /// ownership (#2192) every key of one holder points into the holder's own arena,
    /// so the holder charges this once, through any of its keys.
    #[doc(hidden)]
    #[inline]
    pub fn arena_retained_bytes(&self) -> usize {
        self.arena.heap_bytes()
    }

    /// Whether both handles point into one arena.
    #[doc(hidden)]
    #[inline]
    pub fn shares_arena(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.arena, &other.arena)
    }

    #[cfg(test)]
    pub(crate) fn cached_hash_for_test(&self) -> u64 {
        self.arena.hash(self.index)
    }

    /// Test-only hook that overrides the cached hash so collision handling can
    /// be exercised deterministically; it never changes the identity fields.
    #[cfg(test)]
    pub(crate) fn with_cached_hash_for_test(self, hash: u64) -> Self {
        let view = self.as_view();
        let mut builder = TreeArenaBuilder::for_views(std::iter::once(view));
        builder.push_view(FusionTreeRef::with_hash(
            view.uncoupled(),
            view.coupled(),
            view.is_dual(),
            view.innerlines(),
            view.vertices(),
            hash,
        ));
        builder.freeze().key(0)
    }

    /// Construct and validate a categorical fusion tree for `rule`.
    ///
    /// # Provider-domain precondition
    ///
    /// Every input [`SectorId`] and every channel generated by an exercised
    /// algebra query must be representable by `rule`. This validates
    /// categorical structure, not finite-provider closure, and has the same
    /// limitation documented on [`Self::validate_for_rule`].
    pub fn try_new_for_rule<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: SectorId,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = MultiplicityIndex>,
    {
        let tree = Self::new(uncoupled, coupled, is_dual, innerlines, vertices);
        tree.validate_for_rule(rule)?;
        Ok(tree)
    }

    /// Construct and validate a fusion tree through checked finite algebra.
    pub fn try_new_for_rule_checked<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: SectorId,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = MultiplicityIndex>,
    {
        let tree = Self::new(uncoupled, coupled, is_dual, innerlines, vertices);
        tree.validate_for_rule_checked(rule)?;
        Ok(tree)
    }

    // Why not store a fusion style here: the bound rule is the authority for
    // categorical capabilities; duplicating it in identity can become stale.
    /// A lone key: one arena of one row, at most 5 allocations.
    pub(crate) fn new<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: SectorId,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Self
    where
        Uncoupled: IntoIterator<Item = SectorId>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = SectorId>,
        Vertices: IntoIterator<Item = MultiplicityIndex>,
    {
        let (uncoupled, is_dual, innerlines, vertices) = (
            uncoupled.into_iter(),
            is_dual.into_iter(),
            innerlines.into_iter(),
            vertices.into_iter(),
        );
        let mut builder = TreeArenaBuilder::exact(
            1,
            uncoupled.size_hint().0 + 1 + innerlines.size_hint().0,
            vertices.size_hint().0,
            is_dual.size_hint().0,
        );
        builder.push_parts(uncoupled, coupled, is_dual, innerlines, vertices);
        builder.freeze().key(0)
    }

    pub(crate) fn try_from_sector_ids<Uncoupled, Dual, Innerlines, Vertices>(
        uncoupled: Uncoupled,
        coupled: usize,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        Ok(Self::new(
            uncoupled.into_iter().map(SectorId::new),
            SectorId::new(coupled),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices
                .into_iter()
                .map(MultiplicityIndex::try_from)
                .collect::<Result<MultiplicityVec, _>>()?,
        ))
    }

    /// Construct and validate a categorical fusion tree from numeric labels.
    ///
    /// `uncoupled`, `coupled`, and `innerlines` contain provider-local sector
    /// IDs. `vertices` instead contains one-based outer-multiplicity labels;
    /// a zero vertex returns [`CoreError::InvalidMultiplicityIndex`].
    ///
    /// # Provider-domain precondition
    ///
    /// Every input sector ID and every channel generated by an exercised
    /// algebra query must be representable by `rule`. This method checks
    /// categorical structure, not finite-provider closure, and has the same
    /// limitation documented on [`Self::validate_for_rule`].
    pub fn try_from_sector_ids_for_rule<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: usize,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        Self::try_new_for_rule(
            rule,
            uncoupled.into_iter().map(SectorId::new),
            SectorId::new(coupled),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices
                .into_iter()
                .map(MultiplicityIndex::try_from)
                .collect::<Result<MultiplicityVec, _>>()?,
        )
    }

    /// Construct and checked-validate a fusion tree from numeric labels.
    ///
    /// Sector values are provider-local IDs. Vertex values are one-based
    /// outer-multiplicity labels.
    pub fn try_from_sector_ids_for_rule_checked<R, Uncoupled, Dual, Innerlines, Vertices>(
        rule: &R,
        uncoupled: Uncoupled,
        coupled: usize,
        is_dual: Dual,
        innerlines: Innerlines,
        vertices: Vertices,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
        Uncoupled: IntoIterator<Item = usize>,
        Dual: IntoIterator<Item = bool>,
        Innerlines: IntoIterator<Item = usize>,
        Vertices: IntoIterator<Item = usize>,
    {
        let vertices = vertices
            .into_iter()
            .map(MultiplicityIndex::try_from)
            .collect::<Result<MultiplicityVec, _>>()?;
        Self::try_new_for_rule_checked(
            rule,
            uncoupled.into_iter().map(SectorId::new),
            SectorId::new(coupled),
            is_dual,
            innerlines.into_iter().map(SectorId::new),
            vertices,
        )
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        self.arena.uncoupled(self.index)
    }

    #[inline]
    pub fn coupled(&self) -> SectorId {
        self.arena.coupled(self.index)
    }

    #[inline]
    pub fn is_dual(&self) -> &[bool] {
        self.arena.is_dual(self.index)
    }

    #[inline]
    pub fn innerlines(&self) -> &[SectorId] {
        self.arena.innerlines(self.index)
    }

    #[inline]
    pub fn vertices(&self) -> &[MultiplicityIndex] {
        self.arena.vertices(self.index)
    }

    /// Validate this raw key as a categorical fusion tree for `rule`.
    ///
    /// Ruleless construction remains available for exact categorical
    /// reconstruction, deserialization, and expert import. Call this boundary
    /// before categorical execution; application routing labels are represented
    /// by [`OpaqueBlockKey`], not raw fusion trees.
    ///
    /// # Provider-domain precondition
    ///
    /// The infallible [`FusionRule`] contract assumes every [`SectorId`]
    /// belongs to the provider domain, and every channel generated by an
    /// exercised algebra query is representable. Arbitrary out-of-domain
    /// labels, provider nonclosure, and coherence are not checked here; a
    /// finite-table provider may panic when queried outside that precondition.
    pub fn validate_for_rule<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        validate_fusion_tree_for_rule(rule, self)?;
        Ok(())
    }

    /// Validate this raw key through checked finite algebra.
    pub fn validate_for_rule_checked<R>(&self, rule: &R) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_fusion_tree_for_rule_checked(rule, self)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ValidatedFusionTree<'a, R> {
    pub(crate) rule: &'a R,
    pub(crate) key: &'a FusionTreeKey,
}

pub(super) fn validate_fusion_tree_for_rule<'a, R>(
    rule: &'a R,
    tree: &'a FusionTreeKey,
) -> Result<ValidatedFusionTree<'a, R>, CoreError>
where
    R: FusionRule,
{
    validate_fusion_tree_view_for_rule(rule, tree.as_view())?;
    Ok(ValidatedFusionTree { rule, key: tree })
}

/// [`validate_fusion_tree_for_rule`] of a borrowed tree, for scratch trees
/// that have no key.
pub(super) fn validate_fusion_tree_view_for_rule<R>(
    rule: &R,
    tree: FusionTreeRef<'_>,
) -> Result<(), CoreError>
where
    R: FusionRule,
{
    validate_fusion_tree_structure(rule, tree)?;
    validate_fusion_tree_vertices(tree, |left, right, coupled| {
        Ok(rule.nsymbol(left, right, coupled))
    })
}

pub(crate) fn validate_fusion_tree_for_rule_checked<'a, R>(
    rule: &'a R,
    tree: &'a FusionTreeKey,
) -> Result<ValidatedFusionTree<'a, R>, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    ShapeValidatedFusionTree::try_new(tree)?.validate_for_rule_checked(rule)
}

pub(crate) fn validate_fusion_tree_for_rule_checked_after_shape<'tree, R>(
    rule: &'tree R,
    tree: &'tree FusionTreeKey,
) -> Result<ValidatedFusionTree<'tree, R>, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    ShapeValidatedFusionTree { tree }.validate_for_rule_checked(rule)
}

#[derive(Clone, Copy)]
pub(crate) struct ShapeValidatedFusionTree<'tree> {
    tree: &'tree FusionTreeKey,
}

impl<'tree> ShapeValidatedFusionTree<'tree> {
    pub(crate) fn try_new(tree: &'tree FusionTreeKey) -> Result<Self, CoreError> {
        validate_fusion_tree_key_shape(tree.as_view())?;
        Ok(Self { tree })
    }

    fn validate_for_rule_checked<R>(
        self,
        rule: &'tree R,
    ) -> Result<ValidatedFusionTree<'tree, R>, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_fusion_tree_checked_after_shape(
            self.tree.as_view(),
            rule.vacuum(),
            |left, right| {
                rule.try_fusion_channels(left, right)
                    .map(drop)
                    .map_err(CheckedFusionSpaceError::from)
            },
            |left, right, coupled| {
                rule.try_nsymbol(left, right, coupled)
                    .map_err(CheckedFusionSpaceError::from)
            },
        )?;
        Ok(ValidatedFusionTree {
            rule,
            key: self.tree,
        })
    }
}

fn validate_fusion_tree_structure<R>(rule: &R, tree: FusionTreeRef<'_>) -> Result<(), CoreError>
where
    R: FusionRule,
{
    validate_fusion_tree_key_shape(tree)?;
    validate_fusion_tree_structure_after_shape(rule.vacuum(), tree)
}

/// Checked admission of one shape-valid fusion tree, shared by every fusion
/// style: unit-sector structure, the rank-1 `N(c, 1, c)` provider-domain probe,
/// and every vertex.
pub(super) fn validate_fusion_tree_checked_after_shape<E>(
    tree: FusionTreeRef<'_>,
    vacuum: SectorId,
    mut fusion_channels: impl FnMut(SectorId, SectorId) -> Result<(), E>,
    mut nsymbol: impl FnMut(SectorId, SectorId, SectorId) -> Result<usize, E>,
) -> Result<(), E>
where
    E: From<CoreError>,
{
    validate_fusion_tree_structure_after_shape(vacuum, tree)?;
    if tree.uncoupled().len() == 1 && nsymbol(tree.coupled(), vacuum, tree.coupled())? == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "rank-1 fusion tree sector is absent from unit fusion",
        }
        .into());
    }
    validate_fusion_tree_vertices(tree, |left, right, coupled| {
        fusion_channels(left, right)?;
        nsymbol(left, right, coupled)
    })
}

fn validate_fusion_tree_structure_after_shape(
    vacuum: SectorId,
    tree: FusionTreeRef<'_>,
) -> Result<(), CoreError> {
    match tree.uncoupled().len() {
        0 if tree.coupled() != vacuum => Err(CoreError::MalformedFusionTree {
            message: "rank-0 fusion tree coupled sector must equal the vacuum",
        }),
        1 if Some(tree.coupled()) != tree.uncoupled().first().copied() => {
            Err(CoreError::MalformedFusionTree {
                message: "rank-1 fusion tree coupled sector must equal its uncoupled sector",
            })
        }
        _ => Ok(()),
    }
}

fn validate_fusion_tree_vertices<E>(
    tree: FusionTreeRef<'_>,
    mut multiplicity: impl FnMut(SectorId, SectorId, SectorId) -> Result<usize, E>,
) -> Result<(), E>
where
    E: From<CoreError>,
{
    let rank = tree.uncoupled().len();
    for vertex_index in 0..rank.saturating_sub(1) {
        let left = if vertex_index == 0 {
            tree.uncoupled()[0]
        } else {
            tree.innerlines()[vertex_index - 1]
        };
        let right = tree.uncoupled()[vertex_index + 1];
        let coupled = if vertex_index + 2 == rank {
            tree.coupled()
        } else {
            tree.innerlines()[vertex_index]
        };
        let multiplicity = multiplicity(left, right, coupled)?;
        if multiplicity == 0 {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
            .into());
        }
        if tree.vertices()[vertex_index].get() > multiplicity {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
            .into());
        }
    }
    Ok(())
}

impl FusionTreePairKey {
    /// Validate both trees and their shared coupled sector.
    ///
    /// # Provider-domain precondition
    ///
    /// Both trees follow [`FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
    pub fn validate_for_rule<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        validate_fusion_tree_pair_for_rule(rule, self)?;
        Ok(())
    }

    /// Validate both trees and their shared coupled sector through checked
    /// finite algebra.
    pub fn validate_for_rule_checked<R>(&self, rule: &R) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_fusion_tree_pair_for_rule_checked(rule, self)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ValidatedFusionTreePair<'a, R> {
    pub(crate) rule: &'a R,
    pub(crate) key: &'a FusionTreePairKey,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FusionTreePairOrientation {
    Direct,
    Adjoint,
}

pub(super) fn validate_fusion_tree_pair_for_rule<'a, R>(
    rule: &'a R,
    tree_pair: &'a FusionTreePairKey,
) -> Result<ValidatedFusionTreePair<'a, R>, CoreError>
where
    R: FusionRule,
{
    let codomain = validate_fusion_tree_for_rule(rule, tree_pair.codomain_tree())?;
    let domain = validate_fusion_tree_for_rule(rule, tree_pair.domain_tree())?;
    validate_fusion_tree_pair_coupled(codomain.key, domain.key)?;
    Ok(ValidatedFusionTreePair {
        rule,
        key: tree_pair,
    })
}

fn validate_fusion_tree_pair_for_rule_checked<'a, R>(
    rule: &'a R,
    tree_pair: &'a FusionTreePairKey,
) -> Result<ValidatedFusionTreePair<'a, R>, CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    let codomain = validate_fusion_tree_for_rule_checked(rule, tree_pair.codomain_tree())?;
    let domain = validate_fusion_tree_for_rule_checked(rule, tree_pair.domain_tree())?;
    validate_fusion_tree_pair_coupled(codomain.key, domain.key)?;
    Ok(ValidatedFusionTreePair {
        rule,
        key: tree_pair,
    })
}

pub(crate) fn validate_fusion_tree_pair_coupled(
    codomain: &FusionTreeKey,
    domain: &FusionTreeKey,
) -> Result<(), CoreError> {
    if codomain.coupled() == domain.coupled() {
        Ok(())
    } else {
        Err(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    }
}

pub(crate) fn validate_fusion_tree_key_shape(tree: FusionTreeRef<'_>) -> Result<(), CoreError> {
    let rank = tree.uncoupled().len();
    if tree.is_dual().len() != rank {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    let expected_innerlines = rank.saturating_sub(2);
    if tree.innerlines().len() != expected_innerlines {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of innerlines",
        });
    }
    let expected_vertices = rank.saturating_sub(1);
    if tree.vertices().len() != expected_vertices {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        });
    }
    Ok(())
}
