use super::*;

use core::num::NonZeroUsize;

/// One-based outer-multiplicity label at a fusion vertex.
///
/// Unlike [`SectorId`], this value selects a basis vector within a fixed
/// fusion channel. Zero is not a categorical basis label.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MultiplicityIndex(NonZeroUsize);

impl MultiplicityIndex {
    /// The unique vertex label of a multiplicity-free fusion channel.
    pub const ONE: Self = Self(NonZeroUsize::MIN);

    /// Constructs a one-based fusion-vertex basis label.
    ///
    /// Returns `None` for zero. Numeric import boundaries that need a typed
    /// error can use [`TryFrom<usize>`], which reports
    /// [`CoreError::InvalidMultiplicityIndex`].
    #[inline]
    pub const fn new(value: usize) -> Option<Self> {
        match NonZeroUsize::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the one-based fusion-vertex basis label.
    #[inline]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for MultiplicityIndex {
    type Error = CoreError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(value).ok_or(CoreError::InvalidMultiplicityIndex { value })
    }
}

impl From<MultiplicityIndex> for usize {
    fn from(value: MultiplicityIndex) -> Self {
        value.get()
    }
}

struct SectorLegData {
    sectors: SectorVec,
    /// Per-sector degeneracy, parallel to `sectors`. The leg is the single
    /// source of truth for the sector -> degeneracy map of one tensor axis
    /// (TensorKit `GradedSpace` parity: the space stores the complete map
    /// independent of which fusion trees are populated).
    degeneracies: DimVec,
    /// The dual of this map, filled by the first dual query and shared by
    /// every leg dualized from this storage afterwards (TensorKit
    /// `dual(V)` shares `V.dims`).
    ///
    /// Why lazy and not at construction: `SectorLeg::new` has no rule, and a
    /// `SectorId`'s dual is rule-relative. Why trusted only under the filling
    /// rule's identity: equal [`RuleIdentity`] values promise equal
    /// operations, the same contract every structure cache keys on, so a
    /// query under that identity reuses the map without asking the provider
    /// (TensorKit's `dual(V)` is a flag flip, `gradedspace.jl:112`). Two
    /// rules of one Rust type can still dualize one id differently, so any
    /// other identity re-queries every sector, and a query whose rule
    /// disagrees with `images` builds its own storage and leaves this map
    /// untouched.
    ///
    /// Why boxed exact-size slices and not inline vectors: every leg storage
    /// carries this field, most are never dualized, and the few allocations
    /// of the first dual query happen once per storage on the cold path.
    dual: OnceLock<Box<DualSectorMap>>,
}

pub(crate) struct DualSectorMap {
    /// `dual(sectors[i])` under the rule that filled this map.
    pub(crate) images: Box<[SectorId]>,
    /// The sorted dual map, `None` when it equals the source map.
    pub(crate) moved: Option<MovedSectorMap>,
    /// The identity of the rule that filled this map.
    pub(crate) filled_by: RuleIdentity,
}

pub(crate) struct MovedSectorMap {
    pub(crate) sectors: Box<[SectorId]>,
    pub(crate) degeneracies: Box<[usize]>,
    /// `dual(sectors[j])` under the filling rule: the source sector that
    /// `sectors[j]` is the dual of.
    pub(crate) images: Box<[SectorId]>,
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectorLegConstructionError {
    DuplicateSector { sector: SectorId },
}

impl fmt::Display for SectorLegConstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSector { sector } => {
                write!(formatter, "sector {sector:?} appears multiple times")
            }
        }
    }
}

impl std::error::Error for SectorLegConstructionError {}

#[derive(Clone)]
pub struct SectorLeg {
    data: Arc<SectorLegData>,
    is_dual: bool,
    /// Whether this leg's map is `data.dual`'s moved map rather than
    /// `data`'s own. Invariant: set only after `data.dual` holds a moved map;
    /// a `OnceLock` is never cleared.
    moved: bool,
}

impl PartialEq for SectorLeg {
    fn eq(&self, other: &Self) -> bool {
        self.is_dual == other.is_dual
            && ((Arc::ptr_eq(&self.data, &other.data) && self.moved == other.moved)
                || self.map() == other.map())
    }
}

impl Eq for SectorLeg {}

impl Hash for SectorLeg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Same fields in the same order as a derived hash would use, so hashes
        // depend on content only, never on which storage a leg shares.
        let (sectors, degeneracies) = self.map();
        sectors.hash(state);
        degeneracies.hash(state);
        self.is_dual.hash(state);
    }
}

impl fmt::Debug for SectorLeg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Data<'a>(&'a [SectorId], &'a [usize]);
        impl fmt::Debug for Data<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct("SectorLegData")
                    .field("sectors", &self.0)
                    .field("degeneracies", &self.1)
                    .finish()
            }
        }
        let (sectors, degeneracies) = self.map();
        formatter
            .debug_struct("SectorLeg")
            .field("data", &Data(sectors, degeneracies))
            .field("is_dual", &self.is_dual)
            .finish()
    }
}

impl SectorLeg {
    fn from_data(sectors: SectorVec, degeneracies: DimVec, is_dual: bool) -> Self {
        Self {
            data: Arc::new(SectorLegData {
                sectors,
                degeneracies,
                dual: OnceLock::new(),
            }),
            is_dual,
            moved: false,
        }
    }

    #[inline]
    fn map(&self) -> (&[SectorId], &[usize]) {
        if self.moved {
            let moved = self
                .data
                .dual
                .get()
                .and_then(|dual| dual.moved.as_ref())
                .expect("a moved leg's storage holds its moved dual map");
            (&moved.sectors, &moved.degeneracies)
        } else {
            (&self.data.sectors, &self.data.degeneracies)
        }
    }

    /// The stored `dual(sector)` of each of this leg's sectors, when a dual
    /// query has filled the shared map.
    fn stored_images(&self) -> Option<&[SectorId]> {
        let dual = self.data.dual.get()?;
        if self.moved {
            dual.moved.as_ref().map(|moved| &*moved.images)
        } else {
            Some(&dual.images)
        }
    }

    /// The dual leg over the same storage; `data.dual` must be filled.
    fn flipped(&self) -> Self {
        let moves = self
            .data
            .dual
            .get()
            .is_some_and(|dual| dual.moved.is_some());
        Self {
            data: Arc::clone(&self.data),
            is_dual: !self.is_dual,
            moved: moves && !self.moved,
        }
    }

    /// Conservative retained bytes for this leg's Arc-backed sector and
    /// degeneracy metadata, excluding the inline `SectorLeg` pointer shell.
    ///
    /// The dual map is charged up front at its exact worst case, a moved map:
    /// `size_of::<DualSectorMap>() + n * (3 * size_of::<SectorId>() +
    /// size_of::<usize>())` for `n` sectors (images, sorted sectors,
    /// back-images, degeneracies). A dual query that fills it later therefore
    /// never grows a charge a cache has already admitted.
    ///
    /// `size_of::<DualSectorMap>()` includes the inline [`RuleIdentity`] of
    /// the filling rule. The map holds a strong clone of that identity: one
    /// identity per leg storage, which can outlive the rule while a leg (for
    /// example one inside a cached HomSpace) is alive. Any heap behind it
    /// (content bytes, a product or value node) is shared with the rule's own
    /// identity and is not charged here, because the charge is fixed before
    /// the first dual query, and a dual query must never grow a charge a
    /// cache has already admitted. In-tree identities retain a few tens of
    /// bytes; a provider's content bytes are unbounded by design, as they are
    /// for cache keys.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        let dual_map = self
            .data
            .sectors
            .len()
            .saturating_mul(3 * std::mem::size_of::<SectorId>() + std::mem::size_of::<usize>());
        std::mem::size_of::<SectorLegData>()
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(spilled_smallvec_heap_bytes(&self.data.sectors))
            .saturating_add(spilled_smallvec_heap_bytes(&self.data.degeneracies))
            .saturating_add(std::mem::size_of::<DualSectorMap>())
            .saturating_add(dual_map)
    }

    /// Builds one external leg from `(sector, degeneracy)` pairs.
    ///
    /// Zero-degeneracy sectors are absent from the resulting leg. Remaining
    /// pairs are stored sorted by sector id.
    ///
    /// # Panics
    ///
    /// Panics when a sector appears more than once. Use [`Self::try_new`] to
    /// handle this error.
    ///
    /// # Examples
    ///
    /// ```
    /// use symtn_core::{SectorLeg, Z2Irrep};
    ///
    /// let leg = SectorLeg::new([(Z2Irrep::ODD, 3), (Z2Irrep::EVEN, 2)], false);
    /// assert_eq!(leg.sectors().len(), 2);
    /// assert_eq!(leg.degeneracies(), &[2, 3]);
    /// assert!(!leg.is_dual());
    ///
    /// let dual_leg = SectorLeg::new([(Z2Irrep::ODD, 1)], true);
    /// assert!(dual_leg.is_dual());
    /// ```
    pub fn new<Pairs, Sector>(pairs: Pairs, is_dual: bool) -> Self
    where
        Pairs: IntoIterator<Item = (Sector, usize)>,
        Sector: Into<SectorId>,
    {
        Self::try_new(pairs, is_dual).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Fallible counterpart of [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`SectorLegConstructionError::DuplicateSector`] when any sector
    /// is declared more than once, including zero-degeneracy declarations.
    pub fn try_new<Pairs, Sector>(
        pairs: Pairs,
        is_dual: bool,
    ) -> Result<Self, SectorLegConstructionError>
    where
        Pairs: IntoIterator<Item = (Sector, usize)>,
        Sector: Into<SectorId>,
    {
        let pairs = pairs
            .into_iter()
            .map(|(sector, degeneracy)| (sector.into(), degeneracy))
            .collect::<SmallVec<[(SectorId, usize); 8]>>();
        Self::build(pairs, is_dual, None)
    }

    /// Shared body of [`Self::try_new`] and the dual constructors.
    ///
    /// `share` is the leg the result may share its sector data with when the
    /// two carry the same sector -> degeneracy map: the legs then differ only
    /// in their dual flag, which is the case TensorKit's `dual(V)` covers by
    /// reusing the sector dictionary outright.
    fn build(
        mut pairs: SmallVec<[(SectorId, usize); 8]>,
        is_dual: bool,
        share: Option<&Self>,
    ) -> Result<Self, SectorLegConstructionError> {
        pairs.sort_unstable_by_key(|&(sector, _)| sector);
        // Why not discard zeros first: duplicate validity must not depend on
        // input order or storage representation; symtn follows TensorKit's
        // strict tuple invariant, not its SectorDict zero-first corner.
        for window in pairs.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(SectorLegConstructionError::DuplicateSector {
                    sector: window[0].0,
                });
            }
        }
        // Why compare before building: a leg whose map is unchanged by the
        // rule's dual differs from its dual only in the flag, so the dual can
        // borrow the same `Arc` and allocate nothing.
        if let Some(share) = share {
            if share.has_sorted_pairs(&pairs) {
                return Ok(Self {
                    data: Arc::clone(&share.data),
                    is_dual,
                    moved: share.moved,
                });
            }
        }
        let nonzero_count = pairs
            .iter()
            .filter(|(_, degeneracy)| *degeneracy != 0)
            .count();
        let mut sectors = SectorVec::new();
        let mut degeneracies = DimVec::new();
        sectors.reserve(nonzero_count);
        degeneracies.reserve(nonzero_count);
        for (sector, degeneracy) in pairs {
            if degeneracy == 0 {
                continue;
            }
            sectors.push(sector);
            degeneracies.push(degeneracy);
        }
        Ok(Self::from_data(sectors, degeneracies, is_dual))
    }

    pub fn from_sector_id(sector: usize, degeneracy: usize) -> Self {
        Self::new([(SectorId::new(sector), degeneracy)], false)
    }

    #[inline]
    pub fn sectors(&self) -> &[SectorId] {
        self.map().0
    }

    /// Per-sector degeneracies, parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        self.map().1
    }

    /// Degeneracy of `sector` on this leg, `None` when the sector is not
    /// part of the leg.
    pub fn degeneracy(&self, sector: SectorId) -> Option<usize> {
        let (sectors, degeneracies) = self.map();
        sectors
            .binary_search(&sector)
            .ok()
            .map(|index| degeneracies[index])
    }

    /// `(sector, degeneracy)` pairs in sorted sector order.
    pub fn iter(&self) -> impl Iterator<Item = (SectorId, usize)> + '_ {
        let (sectors, degeneracies) = self.map();
        sectors.iter().copied().zip(degeneracies.iter().copied())
    }

    /// The dual leg: every sector is replaced by its dual (degeneracies
    /// carried along) and the dual flag is flipped.
    ///
    /// After the first query on a leg's storage, the dual shares that
    /// storage and allocates nothing; a later query by a rule with the same
    /// [`FusionRule::rule_identity`] also asks the rule for no sector's dual.
    pub fn dual<R>(&self, rule: &R) -> Self
    where
        R: FusionRule,
    {
        self.dual_under(rule, &rule.rule_identity())
    }

    /// [`Self::dual`] under `identity`, which must be `rule`'s identity;
    /// callers that derive several legs read it once.
    ///
    /// Why not assert that here: reading the identity is a provider query,
    /// and debug builds would then ask it on every call.
    pub(crate) fn dual_under<R>(&self, rule: &R, identity: &RuleIdentity) -> Self
    where
        R: FusionRule,
    {
        self.dual_by(
            identity,
            |sector| Ok(rule.dual(sector)),
            |sector| SectorLegConstructionError::DuplicateSector { sector },
        )
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Shared body of the dual constructors. Under the identity that filled
    /// this storage's dual map, the map answers without any `dual_of` query.
    /// Otherwise `dual_of` is queried once per sector in sector order, so the
    /// first rule error is the one returned; `not_injective` reports the
    /// smallest sector two sectors dualize onto.
    fn dual_by<E>(
        &self,
        identity: &RuleIdentity,
        mut dual_of: impl FnMut(SectorId) -> Result<SectorId, E>,
        not_injective: impl FnOnce(SectorId) -> E,
    ) -> Result<Self, E> {
        if self
            .data
            .dual
            .get()
            .is_some_and(|map| map.filled_by == *identity)
        {
            return Ok(self.flipped());
        }
        let (sectors, degeneracies) = self.map();
        if let Some(images) = self.stored_images() {
            for (index, (&sector, &image)) in sectors.iter().zip(images).enumerate() {
                let dual = dual_of(sector)?;
                if dual != image {
                    // A rule that disagrees with the filling rule: build the
                    // dual as a fresh leg and leave the shared map untouched.
                    let mut pairs =
                        SmallVec::<[(SectorId, usize); 8]>::with_capacity(sectors.len());
                    pairs.extend(
                        images[..index]
                            .iter()
                            .copied()
                            .zip(degeneracies.iter().copied()),
                    );
                    pairs.push((dual, degeneracies[index]));
                    for (&sector, &degeneracy) in
                        sectors[index + 1..].iter().zip(&degeneracies[index + 1..])
                    {
                        pairs.push((dual_of(sector)?, degeneracy));
                    }
                    return Self::build(pairs, !self.is_dual, Some(self)).map_err(
                        |SectorLegConstructionError::DuplicateSector { sector }| {
                            not_injective(sector)
                        },
                    );
                }
            }
            return Ok(self.flipped());
        }

        // First dual query on this storage. Only an unmoved leg gets here: a
        // moved leg exists only after its storage's dual map is filled.
        let mut triples =
            SmallVec::<[(SectorId, usize, SectorId); 8]>::with_capacity(sectors.len());
        for (&sector, &degeneracy) in sectors.iter().zip(degeneracies) {
            triples.push((dual_of(sector)?, degeneracy, sector));
        }
        let images = triples.iter().map(|&(dual, _, _)| dual).collect();
        triples.sort_unstable_by_key(|&(dual, _, _)| dual);
        for window in triples.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(not_injective(window[0].0));
            }
        }
        let unchanged = triples
            .iter()
            .map(|&(dual, degeneracy, _)| (dual, degeneracy))
            .eq(self.iter());
        let moved = (!unchanged).then(|| MovedSectorMap {
            sectors: triples.iter().map(|&(dual, _, _)| dual).collect(),
            degeneracies: triples
                .iter()
                .map(|&(_, degeneracy, _)| degeneracy)
                .collect(),
            images: triples.iter().map(|&(_, _, source)| source).collect(),
        });
        Ok(self.install_dual_map(DualSectorMap {
            images,
            moved,
            filled_by: identity.clone(),
        }))
    }

    /// The dual leg once `map`, computed from this leg's own sectors, is
    /// offered to the shared storage.
    pub(crate) fn install_dual_map(&self, map: DualSectorMap) -> Self {
        match self.data.dual.set(Box::new(map)) {
            Ok(()) => self.flipped(),
            // Another thread filled the map first; share it only when its
            // rule agrees with this one.
            Err(map) if self.stored_images() == Some(&*map.images) => self.flipped(),
            Err(map) => match map.moved {
                None => Self {
                    data: Arc::clone(&self.data),
                    is_dual: !self.is_dual,
                    moved: false,
                },
                Some(moved) => Self::from_data(
                    SectorVec::from_vec(moved.sectors.into_vec()),
                    DimVec::from_vec(moved.degeneracies.into_vec()),
                    !self.is_dual,
                ),
            },
        }
    }

    /// Whether `pairs`, sorted by sector and free of zero degeneracies, is
    /// this leg's own sector -> degeneracy map.
    fn has_sorted_pairs(&self, pairs: &[(SectorId, usize)]) -> bool {
        pairs.len() == self.sectors().len()
            && pairs
                .iter()
                .zip(self.iter())
                .all(|(&pair, own)| pair == own)
    }

    /// Whether `other` borrows this leg's sector and degeneracy storage.
    #[doc(hidden)]
    pub fn shares_sector_data_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// The dual leg, returning a finite-algebra error when any sector has no
    /// representable dual ([`FusionAlgebraError`] from the rule) or when the
    /// rule's dual collapses two of this leg's sectors onto one
    /// ([`FusionAlgebraError::DualNotInjective`]).
    pub fn try_dual<R>(&self, rule: &R) -> Result<Self, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        self.try_dual_under(rule, &rule.rule_identity())
    }

    /// [`Self::try_dual`] under `identity`, which must be `rule`'s identity.
    pub(crate) fn try_dual_under<R>(
        &self,
        rule: &R,
        identity: &RuleIdentity,
    ) -> Result<Self, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        // Why not mutate a cloned leg as sectors succeed: a later failure must
        // leave no partially dualized value available to callers.
        //
        // Why an algebra error and not the panicking constructor: a rule whose
        // dual collapses two sectors onto one id breaks the rule's rigidity
        // structure; it is not a caller mistake.
        self.dual_by(
            identity,
            |sector| rule.try_dual_sector(sector),
            |dual| FusionAlgebraError::DualNotInjective { dual },
        )
    }

    /// Checked Generic sibling of [`Self::try_dual`] that preserves the
    /// provider's concrete error without requiring `CheckedFusionAlgebra`.
    #[doc(hidden)]
    pub fn try_dual_generic<R>(
        &self,
        rule: &R,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        self.try_dual_generic_under(rule, &rule.rule_identity())
    }

    /// [`Self::try_dual_generic`] under `identity`, which must be `rule`'s
    /// identity.
    pub(crate) fn try_dual_generic_under<R>(
        &self,
        rule: &R,
        identity: &RuleIdentity,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        self.dual_by(
            identity,
            |sector| {
                rule.try_dual(sector)
                    .map_err(CheckedGenericStructureError::Provider)
            },
            |_| {
                CoreError::MalformedFusionTree {
                    message: "checked Generic dual is not injective on one tensor leg",
                }
                .into()
            },
        )
    }

    #[inline]
    pub const fn is_dual(&self) -> bool {
        self.is_dual
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct FusionTreeLeg {
    sector: SectorId,
    is_dual: bool,
}

impl FusionTreeLeg {
    pub(crate) const fn new(sector: SectorId, is_dual: bool) -> Self {
        Self { sector, is_dual }
    }

    pub(crate) const fn sector(self) -> SectorId {
        self.sector
    }

    pub(crate) const fn is_dual(self) -> bool {
        self.is_dual
    }
}
