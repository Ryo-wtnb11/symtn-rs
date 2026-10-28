use super::*;
use std::sync::RwLock;

use crate::block_structure::may_publish_since;
use crate::cache::{StructureCache, StructureCacheKind};

/// Semantic identity of one multiplicity-free fusion-tree layout: the rule
/// plus the HomSpace sector signature.
///
/// Why the shared content rather than an owned per-leg signature: a warm
/// lookup must construct this key without allocating, so it borrows the
/// caller's content the way [`CompleteHomSpaceStructureCacheKey`] does.
///
/// Why not a derived `Eq`/`Hash` over that content: degeneracies are not part
/// of the layout identity, and deriving would split one layout into one entry
/// per degeneracy assignment. Equality and hashing therefore cover only the
/// per-leg sectors and duality, exactly as the previous owned signature did.
///
/// `generic` separates the vertex-resolved Generic enumeration from the
/// multiplicity-free one of a rule identity.
#[derive(Clone, Debug)]
pub(crate) struct FusionTreeHomSpaceCacheKey {
    pub(super) rule: RuleIdentity,
    pub(super) generic: bool,
    pub(super) homspace: Arc<FusionTreeHomSpaceContent>,
}

impl PartialEq for FusionTreeHomSpaceCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule
            && self.generic == other.generic
            && (Arc::ptr_eq(&self.homspace, &other.homspace)
                || (product_space_signature_eq(&self.homspace.codomain, &other.homspace.codomain)
                    && product_space_signature_eq(&self.homspace.domain, &other.homspace.domain)))
    }
}

impl Eq for FusionTreeHomSpaceCacheKey {}

impl std::hash::Hash for FusionTreeHomSpaceCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.generic.hash(state);
        hash_product_space_signature(&self.homspace.codomain, state);
        hash_product_space_signature(&self.homspace.domain, state);
    }
}

pub(super) fn product_space_signature_eq(
    left: &FusionProductSpace,
    right: &FusionProductSpace,
) -> bool {
    left.legs().len() == right.legs().len()
        && left.legs().iter().zip(right.legs()).all(|(left, right)| {
            left.is_dual() == right.is_dual() && left.sectors() == right.sectors()
        })
}

fn hash_product_space_signature<H: std::hash::Hasher>(space: &FusionProductSpace, state: &mut H) {
    space.legs().len().hash(state);
    for leg in space.legs() {
        leg.sectors().hash(state);
        leg.is_dual().hash(state);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CompleteFusionMode {
    MultiplicityFree,
    Generic,
}

/// Semantic identity of one complete block layout.
#[derive(Clone, Debug)]
pub(crate) struct CompleteHomSpaceStructureCacheKey {
    pub(crate) rule: RuleIdentity,
    pub(crate) mode: CompleteFusionMode,
    pub(crate) homspace: Arc<FusionTreeHomSpaceContent>,
}

impl CompleteHomSpaceStructureCacheKey {
    pub(crate) fn new<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule: rule.rule_identity(),
            mode: CompleteFusionMode::MultiplicityFree,
            homspace: Arc::clone(&homspace.content),
        }
    }

    pub(crate) fn generic(rule: RuleIdentity, homspace: &FusionTreeHomSpace) -> Self {
        Self {
            rule,
            mode: CompleteFusionMode::Generic,
            homspace: Arc::clone(&homspace.content),
        }
    }
}

/// A [`CompleteHomSpaceStructureCacheKey`] lookup that borrows its fields,
/// so a warm lookup touches no shared reference count. Hashes as the owned
/// key does.
pub(crate) struct CompleteHomSpaceStructureKeyRef<'a> {
    pub(crate) rule: &'a RuleIdentity,
    pub(crate) mode: CompleteFusionMode,
    pub(crate) homspace: &'a FusionTreeHomSpaceContent,
}

impl std::hash::Hash for CompleteHomSpaceStructureKeyRef<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.mode.hash(state);
        self.homspace.hash(state);
    }
}

impl quick_cache::Equivalent<CompleteHomSpaceStructureCacheKey>
    for CompleteHomSpaceStructureKeyRef<'_>
{
    fn equivalent(&self, key: &CompleteHomSpaceStructureCacheKey) -> bool {
        self.rule == &key.rule && self.mode == key.mode && self.homspace == &*key.homspace
    }
}

impl PartialEq for CompleteHomSpaceStructureCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule && self.mode == other.mode && self.homspace == other.homspace
    }
}

impl Eq for CompleteHomSpaceStructureCacheKey {}

impl std::hash::Hash for CompleteHomSpaceStructureCacheKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.mode.hash(state);
        self.homspace.hash(state);
    }
}

impl FusionTreeHomSpaceCacheKey {
    pub(crate) fn new<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Self
    where
        R: MultiplicityFreeFusionRule,
    {
        Self {
            rule: rule.rule_identity(),
            generic: false,
            homspace: Arc::clone(&homspace.content),
        }
    }

    pub(crate) fn generic(rule: RuleIdentity, homspace: &FusionTreeHomSpace) -> Self {
        Self {
            rule,
            generic: true,
            homspace: Arc::clone(&homspace.content),
        }
    }
}

/// The budgets of the caches these replace (#1993).
pub(crate) const SECTOR_STRUCTURE_CACHE_BYTE_BUDGET: u64 =
    crate::cache::DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET;
pub(crate) const DEGENERACY_STRUCTURE_CACHE_BYTE_BUDGET: u64 =
    crate::cache::DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET;

/// Fusion-tree keys and the shared sector structure per HomSpace sector
/// signature (TensorKit `sectorstructure`, `structure.jl:41` @cfaa073):
/// degeneracies are not part of the key, so a degeneracy-only change hits.
pub(crate) fn sector_structure_cache(
) -> &'static StructureCache<FusionTreeHomSpaceCacheKey, FusionTreeHomSpaceLayout> {
    static CACHE: OnceLock<StructureCache<FusionTreeHomSpaceCacheKey, FusionTreeHomSpaceLayout>> =
        OnceLock::new();
    CACHE.get_or_init(|| {
        StructureCache::new(
            StructureCacheKind::SectorStructure,
            SECTOR_STRUCTURE_CACHE_BYTE_BUDGET,
        )
    })
}

pub(crate) fn charged_fusion_tree_layout_bytes(
    identity: &FusionTreeHomSpaceCacheKey,
    layout: &FusionTreeHomSpaceLayoutData,
) -> u64 {
    // The key retains the caller's HomSpace content instead of an owned
    // signature copy, so the charge covers that shared content.
    let key_bytes = std::mem::size_of::<FusionTreeHomSpaceCacheKey>()
        .saturating_add(identity.homspace.charged_retained_bytes())
        .saturating_add(identity.rule.charged_retained_bytes());
    // Single-arena ownership (#2192): the layout keys point into one canonical arena, which
    // the sector structure reuses. Charge it here only when that structure
    // does not, so it is charged exactly once per entry.
    let arena = layout.keys.first().map(|key| &key.codomain_tree().arena);
    debug_assert!(layout.keys.iter().all(|key| arena.is_some_and(|arena| {
        Arc::ptr_eq(arena, &key.codomain_tree().arena)
            && Arc::ptr_eq(arena, &key.domain_tree().arena)
    })));
    let sector_arena = layout
        .sector
        .as_ref()
        .ok()
        .and_then(|sector| sector.trees.as_ref());
    let arena_bytes = match arena {
        Some(arena) if !sector_arena.is_some_and(|sector| Arc::ptr_eq(sector, arena)) => {
            arena.heap_bytes()
        }
        _ => 0,
    };
    let tree_bytes = layout
        .keys
        .len()
        .saturating_mul(std::mem::size_of::<FusionTreePairKey>())
        .saturating_add(arena_bytes);
    let sector_bytes = layout
        .sectors
        .capacity()
        .saturating_mul(std::mem::size_of::<FusionTreeCoupledSectorLayout>());
    key_bytes
        .saturating_add(std::mem::size_of::<FusionTreeHomSpaceLayout>())
        .saturating_add(tree_bytes)
        .saturating_add(sector_bytes)
        .saturating_add(std::mem::size_of::<SectorStructure>())
        .saturating_add(
            layout
                .sector
                .as_ref()
                .map_or(0, |sector| sector.charged_heap_bytes()),
        )
        // Map node and the retained Arc control allocations.
        .saturating_add(10 * std::mem::size_of::<usize>()) as u64
}

/// One complete block structure: the content strongly, the canonical wrapper
/// weakly. TensorKit's complete dimensions/strides/offsets boundary is
/// [degeneracystructure](https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/spaces/structure.jl#L114-L198).
/// [QSpace's QIDX/DATA/CGR ownership](https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/QSpace.hh#lines-124:260)
/// keeps runtime structural records with the tensor; it has no corresponding
/// global interner. Bounded Arc/Weak canonical reuse and reset epochs are the
/// Rust adaptation of that complete-geometry boundary.
///
/// Why strong: the wrapper's coupled-sector region memo is immutable data
/// derived from this key alone (part of the complete structure, as in
/// TensorKit's `degeneracystructure`). A weak link let it die between calls,
/// so every warm call through a derived space rebuilt the wrapper and
/// recompiled its regions. The memo is charged here when it materializes.
///
/// `permuted` is the destination memo (#2154): TensorKit derives a permuted
/// HomSpace as `select(W, p)` with a flag-flip `dual` (`homspace.jl:239-243`,
/// `gradedspace.jl:112` @cfaa073), free of provider queries. symtn stores
/// dualized sectors explicitly, so a derivation asks the provider's `dual`;
/// this keeps its answer, a pure function of this entry's key and the axes.
pub(crate) struct DegeneracyStructureEntry {
    key: CompleteHomSpaceStructureCacheKey,
    homspace: HomSpaceId,
    structure: Arc<BlockStructure>,
    /// Canonical destinations of this HomSpace or, for a multiplicity-free
    /// entry, of its adjoint, keyed by `[nout, codomain axes.., domain
    /// axes..]` with the adjoint marked in `nout`'s top bit, and read only
    /// under this entry's own (rule, mode, HomSpace). Published only after a
    /// committed transform; charged to this entry.
    ///
    /// The first [`PERMUTED_SLOTS`] keys fill set-once slots in order, read
    /// with one acquire load each. Why not only the map: every thread of a
    /// warm loop reads the same hot entry, and a lock read writes the lock's
    /// shared state, which stopped warm permutes from scaling past 8 threads.
    /// Only keys beyond the slots go to the map.
    permuted_slots: [OnceLock<(Box<[usize]>, HomSpaceId)>; PERMUTED_SLOTS],
    permuted: RwLock<FxHashMap<Box<[usize]>, HomSpaceId>>,
}

/// Lock-free memo slots per entry. Why four: across the symtn and
/// symtn-tensors test suites, 91% of entries that memoize anything hold at
/// most four distinct (orientation, axes) keys (1: 63%, 2: 14%); every entry
/// of the `cache_scaling` probe holds one. Eight would cover 96% at twice
/// the fixed inline size every entry carries; the rest overflow to the map.
pub(crate) const PERMUTED_SLOTS: usize = 4;

impl DegeneracyStructureEntry {
    fn new(key: &CompleteHomSpaceStructureCacheKey, structure: Arc<BlockStructure>) -> Self {
        Self {
            key: key.clone(),
            homspace: HomSpaceId::from_content(Arc::clone(&key.homspace)),
            structure,
            permuted_slots: std::array::from_fn(|_| OnceLock::new()),
            permuted: RwLock::default(),
        }
    }

    /// Whether this is the `mode` entry of exactly `(rule, source)`.
    pub(crate) fn is_entry_of(
        &self,
        mode: CompleteFusionMode,
        rule: &RuleIdentity,
        source: &FusionTreeHomSpace,
    ) -> bool {
        self.key.mode == mode && &self.key.rule == rule && self.key.homspace == source.content
    }

    /// `read` of the memoized destination under `axes`, borrowed from its
    /// slot; the caller has checked [`Self::is_entry_of`] its source.
    pub(crate) fn permuted<T>(
        &self,
        axes: &[usize],
        read: impl FnOnce(&HomSpaceId) -> T,
    ) -> Option<T> {
        for slot in &self.permuted_slots {
            match slot.get() {
                Some((key, destination)) if **key == *axes => return Some(read(destination)),
                Some(_) => {}
                // Slots fill in order and the map only after all of them.
                None => return None,
            }
        }
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::PermutedMemoMapRead);
        let destination = self
            .permuted
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(axes)
            .cloned()?;
        Some(read(&destination))
    }

    /// Inserts one derived destination unless present, and charges it while
    /// this entry is resident; an entry no longer resident keeps nothing in
    /// its map, and its slots die with it.
    /// The charge inherits [`StructureCache::add_charge`]'s contract: the
    /// cache may sit above its budget until its next insert evicts.
    fn publish_permuted(self: &Arc<Self>, axes: &[usize], destination: HomSpaceId) {
        if self.permuted(axes, |_| ()).is_some() {
            return;
        }
        // The boxed key, and the destination content with its `Arc` header:
        // double-counted when its own entry retains it too.
        let item = std::mem::size_of_val(axes)
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(destination.content.charged_retained_bytes());
        let mut value = (Box::from(axes), destination);
        for slot in &self.permuted_slots {
            match slot.set(value) {
                // Why no removal on a refused charge: a set slot is
                // permanent, and an entry no longer resident dies with its
                // last holder, slot included. A concurrent publication of
                // the same key may take a second slot; both answer equally.
                Ok(()) => {
                    if degeneracy_structure_cache().add_charge(&self.key, self, item as u64) {
                        #[cfg(test)]
                        crate::probe::hit(crate::probe::Event::PermutedMemoPublication);
                    }
                    return;
                }
                Err(refused) => value = refused,
            }
        }
        let (_, destination) = value;
        let bytes = {
            let mut map = self
                .permuted
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if map.contains_key(axes) {
                return;
            }
            let table = permuted_table_bytes(map.capacity());
            map.insert(Box::from(axes), destination);
            permuted_table_bytes(map.capacity())
                .saturating_sub(table)
                .saturating_add(item)
        };
        if degeneracy_structure_cache().add_charge(&self.key, self, bytes as u64) {
            #[cfg(test)]
            crate::probe::hit(crate::probe::Event::PermutedMemoPublication);
        } else {
            self.permuted
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(axes);
        }
    }

    /// The canonical wrapper.
    pub(crate) fn structure(&self) -> Arc<BlockStructure> {
        Arc::clone(&self.structure)
    }

    /// Charges a region memo of the canonical wrapper that just materialized.
    pub(crate) fn charge_regions(self: &Arc<Self>, bytes: u64) {
        degeneracy_structure_cache().add_charge(&self.key, self, bytes);
    }

    pub(crate) fn homspace_id(&self) -> HomSpaceId {
        self.homspace.clone()
    }

    /// The canonical HomSpace of this entry.
    pub(crate) fn canonical_homspace(&self) -> FusionTreeHomSpace {
        FusionTreeHomSpace::from_canonical_id(self.homspace_id())
    }
}

/// Complete block structure per HomSpace, degeneracies included (TensorKit
/// `degeneracystructure`, `structure.jl:114` @cfaa073). A miss reuses the
/// sector structure of [`sector_structure_cache`] and builds only the
/// degeneracy part.
pub(crate) fn degeneracy_structure_cache(
) -> &'static StructureCache<CompleteHomSpaceStructureCacheKey, DegeneracyStructureEntry> {
    static CACHE: OnceLock<
        StructureCache<CompleteHomSpaceStructureCacheKey, DegeneracyStructureEntry>,
    > = OnceLock::new();
    CACHE.get_or_init(|| {
        StructureCache::new(
            StructureCacheKind::DegeneracyStructure,
            DEGENERACY_STRUCTURE_CACHE_BYTE_BUDGET,
        )
    })
}

/// The cached complete structure of `key`: one shard read and the entry's
/// wrapper.
pub(crate) fn complete_hom_space_structure_cached<Q>(
    key: &Q,
) -> Option<Arc<DegeneracyStructureEntry>>
where
    Q: std::hash::Hash + quick_cache::Equivalent<CompleteHomSpaceStructureCacheKey> + ?Sized,
{
    degeneracy_structure_cache().get(key)
}

/// Upper bound of the memo table's retained bytes at `capacity` items:
/// hashbrown keeps at most `2·capacity` buckets (load factor 7/8, or one
/// spare below eight buckets), each a slot plus one control byte, and one
/// trailing control group of at most 16 bytes.
fn permuted_table_bytes(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let slot = std::mem::size_of::<(Box<[usize]>, HomSpaceId)>() + 1;
    capacity.saturating_mul(2 * slot).saturating_add(16)
}

/// What staging learned about the destination memo of one permutation.
#[doc(hidden)]
pub enum Permuted {
    /// The memo answered: no derivation ran.
    Memo,
    /// The destination was derived; publish it after the caller's commit.
    Derived(PermutedMemoTicket),
}

/// Publishes one derived destination into its source's entry after the
/// caller's fallible commit (#2063).
#[doc(hidden)]
pub struct PermutedMemoTicket {
    pub(super) entry: Option<Arc<DegeneracyStructureEntry>>,
    pub(super) axes: SmallVec<[usize; 9]>,
    pub(super) epoch: usize,
}

impl PermutedMemoTicket {
    /// Inserts `committed`, the destination HomSpace the commit returned,
    /// iff the source had a resident entry, `committed_structure` is
    /// canonical and no reset ran since staging began.
    pub fn publish(self, committed: &FusionTreeHomSpace, committed_structure: &BlockStructure) {
        let Some(entry) = self.entry else {
            return;
        };
        if committed_structure.is_canonical() && may_publish_since(self.epoch) {
            entry.publish_permuted(&self.axes, committed.id());
        }
    }
}

/// Admits a structure whose build began at reset epoch `epoch`. A build that
/// straddles a reset returns its result unpublished (the reset contract at
/// [`crate::clear_structure_caches`]). Resident content is marked canonical.
pub(crate) fn admit_complete_hom_space_structure(
    key: CompleteHomSpaceStructureCacheKey,
    structure: Arc<BlockStructure>,
    epoch: usize,
) -> (Arc<DegeneracyStructureEntry>, Arc<BlockStructure>) {
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::CompleteHomSpaceMiss);
    let entry = Arc::new(DegeneracyStructureEntry::new(&key, Arc::clone(&structure)));
    // Link before measuring: a memo that materializes after the snapshot
    // charges itself through the link instead.
    structure.link_region_owner(&entry);
    let charged_bytes = charged_complete_hom_space_structure_bytes(&key, &structure.content_key())
        .saturating_add(structure.materialized_region_bytes());
    let (published, resident) =
        degeneracy_structure_cache().publish(&key, entry, charged_bytes, epoch);
    // Only real residency makes content canonical: an oversize, reset-refused
    // or zero-budget build gets a fresh id on every call, and keys over it
    // must not take completed-transformer slots (#2014).
    if resident {
        published.structure.content_key().mark_canonical();
    }
    let canonical = published.structure();
    (published, canonical)
}

pub(crate) fn charged_complete_hom_space_structure_bytes(
    key: &CompleteHomSpaceStructureCacheKey,
    content: &BlockStructureContent,
) -> u64 {
    std::mem::size_of::<CompleteHomSpaceStructureCacheKey>()
        .saturating_add(std::mem::size_of::<DegeneracyStructureEntry>())
        .saturating_add(key.rule.charged_retained_bytes())
        .saturating_add(key.homspace.charged_retained_bytes())
        .saturating_add(content.charged_retained_bytes())
        // Map node and the retained Arc control allocations.
        .saturating_add(10 * std::mem::size_of::<usize>())
        // The owned wrapper and its region state; the region memo itself is
        // charged by the admission snapshot and as it materializes.
        .saturating_add(BlockStructure::wrapper_retained_bytes(content.rank())) as u64
}

pub(crate) fn reset_structure_caches() {
    degeneracy_structure_cache().clear();
    sector_structure_cache().clear();
}
