use super::*;

/// Categorical identity of one codomain/domain fusion-tree basis pair.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FusionTreePairKey {
    pub(crate) codomain_tree: FusionTreeKey,
    pub(crate) domain_tree: FusionTreeKey,
}

impl FusionTreePairKey {
    /// Combine two raw tree identities without categorical validation.
    ///
    /// This constructor does not check provider membership, individual tree
    /// admissibility, or equality of the coupled sectors. Call
    /// [`Self::validate_for_rule`] before categorical use.
    pub fn pair(codomain_tree: FusionTreeKey, domain_tree: FusionTreeKey) -> Self {
        Self {
            codomain_tree,
            domain_tree,
        }
    }

    /// Construct a raw tree pair from numeric labels.
    ///
    /// Sector arrays and `coupled` contain provider-local sector IDs. The two
    /// vertex arrays instead contain one-based outer-multiplicity labels; a
    /// zero vertex returns [`CoreError::InvalidMultiplicityIndex`].
    ///
    /// # Provider-domain precondition
    ///
    /// Every sector ID must already name a sector in the intended provider. This
    /// constructor cannot check provider membership; call
    /// [`Self::validate_for_rule`] before categorical use. Providers with a
    /// finite table may otherwise panic through their infallible
    /// [`FusionRule`] methods.
    #[expect(
        clippy::too_many_arguments,
        reason = "the raw constructor mirrors two FusionTreeKey descriptions without duplicating that domain model"
    )]
    pub fn try_pair_from_sector_ids<
        Codomain,
        Domain,
        CodomainDual,
        DomainDual,
        CodomainInner,
        DomainInner,
        CodomainVertices,
        DomainVertices,
    >(
        codomain_uncoupled: Codomain,
        domain_uncoupled: Domain,
        coupled: usize,
        codomain_is_dual: CodomainDual,
        domain_is_dual: DomainDual,
        codomain_innerlines: CodomainInner,
        domain_innerlines: DomainInner,
        codomain_vertices: CodomainVertices,
        domain_vertices: DomainVertices,
    ) -> Result<Self, CoreError>
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainDual: IntoIterator<Item = bool>,
        DomainDual: IntoIterator<Item = bool>,
        CodomainInner: IntoIterator<Item = usize>,
        DomainInner: IntoIterator<Item = usize>,
        CodomainVertices: IntoIterator<Item = usize>,
        DomainVertices: IntoIterator<Item = usize>,
    {
        Ok(Self::pair(
            FusionTreeKey::try_from_sector_ids(
                codomain_uncoupled,
                coupled,
                codomain_is_dual,
                codomain_innerlines,
                codomain_vertices,
            )?,
            FusionTreeKey::try_from_sector_ids(
                domain_uncoupled,
                coupled,
                domain_is_dual,
                domain_innerlines,
                domain_vertices,
            )?,
        ))
    }

    #[inline]
    pub fn codomain_tree(&self) -> &FusionTreeKey {
        &self.codomain_tree
    }

    #[inline]
    pub fn domain_tree(&self) -> &FusionTreeKey {
        &self.domain_tree
    }

    #[inline]
    pub fn uncoupled(&self) -> &[SectorId] {
        self.codomain_tree.uncoupled()
    }

    #[inline]
    pub fn codomain_uncoupled(&self) -> &[SectorId] {
        self.codomain_tree.uncoupled()
    }

    #[inline]
    pub fn domain_uncoupled(&self) -> &[SectorId] {
        self.domain_tree.uncoupled()
    }

    #[inline]
    pub fn coupled(&self) -> SectorId {
        self.codomain_tree.coupled()
    }

    #[inline]
    pub fn vertices(&self) -> &[MultiplicityIndex] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn codomain_vertices(&self) -> &[MultiplicityIndex] {
        self.codomain_tree.vertices()
    }

    #[inline]
    pub fn domain_vertices(&self) -> &[MultiplicityIndex] {
        self.domain_tree.vertices()
    }

    #[inline]
    pub fn codomain_innerlines(&self) -> &[SectorId] {
        self.codomain_tree.innerlines()
    }

    #[inline]
    pub fn domain_innerlines(&self) -> &[SectorId] {
        self.domain_tree.innerlines()
    }

    #[inline]
    pub fn codomain_is_dual(&self) -> &[bool] {
        self.codomain_tree.is_dual()
    }

    #[inline]
    pub fn domain_is_dual(&self) -> &[bool] {
        self.domain_tree.is_dual()
    }

    pub fn external_sectors<R>(&self, rule: &R) -> Vec<SectorId>
    where
        R: FusionRule,
    {
        let mut sectors = Vec::with_capacity(
            self.codomain_tree.uncoupled().len() + self.domain_tree.uncoupled().len(),
        );
        sectors.extend(self.codomain_tree.uncoupled().iter().copied());
        sectors.extend(
            self.domain_tree
                .uncoupled()
                .iter()
                .copied()
                .map(|sector| rule.dual(sector)),
        );
        sectors
    }

    #[cfg(test)]
    pub(crate) fn external_is_dual(&self) -> Vec<bool> {
        let mut is_dual = Vec::with_capacity(
            self.codomain_tree.is_dual().len() + self.domain_tree.is_dual().len(),
        );
        is_dual.extend(self.codomain_tree.is_dual().iter().copied());
        is_dual.extend(self.domain_tree.is_dual().iter().copied());
        is_dual
    }

    /// The pair's external-sector group: two refcount bumps, no allocation.
    pub fn group_key(&self) -> FusionTreeGroupKey {
        FusionTreeGroupKey {
            codomain: self.codomain_tree.clone(),
            domain: self.domain_tree.clone(),
        }
    }
}

/// Application-defined block identity with no categorical interpretation.
///
/// The number of words is independent of tensor rank. The inline capacity is
/// a storage detail and never selects an execution path.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct OpaqueBlockKey {
    pub(super) words: SmallVec<[u64; 2]>,
}

impl OpaqueBlockKey {
    /// Construct an opaque key from owned words.
    pub fn new(words: Vec<u64>) -> Self {
        Self {
            words: words.into_iter().collect(),
        }
    }

    /// Construct an opaque key from any sequence of words.
    pub(crate) fn from_words<I>(words: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        Self {
            words: words.into_iter().collect(),
        }
    }

    /// Construct the conventional one-word key for an ordinal block index.
    pub fn ordinal(index: u64) -> Self {
        Self::from_words([index])
    }

    /// Return the uninterpreted words that identify this block.
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    fn compact_id(&self) -> Option<usize> {
        let [word] = self.words.as_slice() else {
            return None;
        };
        usize::try_from(*word).ok()
    }
}

/// Structural namespace occupied by every key in one [`SectorStructure`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlockKeyKind {
    Dense,
    Opaque,
    FusionTree,
}

impl fmt::Display for BlockKeyKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense => formatter.write_str("dense"),
            Self::Opaque => formatter.write_str("opaque"),
            Self::FusionTree => formatter.write_str("fusion-tree"),
        }
    }
}

/// Identity of a stored tensor block.
///
/// A [`SectorStructure`] contains either the anonymous dense key, only opaque
/// keys, or only categorical fusion-tree pairs.
// Why-not box the categorical variant: every owned block key would pay a heap
// allocation and indirection on the production lookup path. Its size is
// intentional because the complete categorical identity is stored inline.
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BlockKey {
    Dense,
    Opaque(OpaqueBlockKey),
    FusionTree(FusionTreePairKey),
}

impl BlockKey {
    /// Return the anonymous key used by a single dense block.
    pub fn trivial() -> Self {
        Self::Dense
    }

    /// Construct an application-defined opaque key.
    pub fn opaque<I>(words: I) -> Self
    where
        I: IntoIterator<Item = u64>,
    {
        Self::Opaque(OpaqueBlockKey::from_words(words))
    }

    /// Construct the conventional one-word opaque key for a block ordinal.
    pub fn ordinal(index: usize) -> Self {
        Self::Opaque(OpaqueBlockKey::ordinal(supported_usize_to_u64(index)))
    }

    /// Borrow the categorical tree pair, if this key names fusion-tree data.
    #[inline]
    pub fn as_fusion_tree_pair(&self) -> Option<&FusionTreePairKey> {
        match self {
            Self::FusionTree(key) => Some(key),
            _ => None,
        }
    }

    /// Return this key's structural namespace.
    #[inline]
    pub fn kind(&self) -> BlockKeyKind {
        match self {
            Self::Dense => BlockKeyKind::Dense,
            Self::Opaque(_) => BlockKeyKind::Opaque,
            Self::FusionTree(_) => BlockKeyKind::FusionTree,
        }
    }

    pub(super) fn compact_id(&self) -> Option<usize> {
        match self {
            Self::Dense => Some(0),
            Self::Opaque(key) => key.compact_id(),
            Self::FusionTree(_) => None,
        }
    }

    pub(crate) fn fusion_tree_group_key(&self) -> Option<FusionTreeGroupKey> {
        match self {
            Self::Dense => None,
            Self::Opaque(_) => None,
            Self::FusionTree(tree) => Some(tree.group_key()),
        }
    }
}

#[cfg(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
))]
const fn supported_usize_to_u64(value: usize) -> u64 {
    value as u64
}

#[cfg(not(any(
    target_pointer_width = "16",
    target_pointer_width = "32",
    target_pointer_width = "64"
)))]
compile_error!("OpaqueBlockKey requires a target whose usize fits in u64");

impl From<OpaqueBlockKey> for BlockKey {
    fn from(value: OpaqueBlockKey) -> Self {
        Self::Opaque(value)
    }
}

impl From<FusionTreePairKey> for BlockKey {
    fn from(value: FusionTreePairKey) -> Self {
        Self::FusionTree(value)
    }
}
