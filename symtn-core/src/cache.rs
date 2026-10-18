//! symtn's structure caches: process-global, byte-bounded, one
//! wrapper over `quick_cache` for every one of them, and one registry of all
//! of them.
//!
//! They mirror TensorKit's `@cached` functions (cfaa073,
//! `src/auxiliary/caches.jl`): each memoizes a pure function of its key, so a
//! hit is interchangeable with a rebuild, and only the application clears or
//! sizes them (as fusion-symbols does for its symbol caches). The registry is
//! TensorKit's `GLOBAL_CACHES` (`caches.jl:1-11`, `push!` at `:160-165`):
//! [`structure_cache_infos`], [`set_structure_cache_byte_budget`] and
//! [`clear_structure_caches`] cover every kind, including the
//! completed-transformer and composed-coefficient caches that `symtn-tensors`
//! owns and registers on first use ([`register_structure_cache`]).
//!
//! Why `quick_cache` and not one `RwLock<LruCache>` per cache: a warm hit is
//! one shard read with a borrowed key, eviction weighs entries by their
//! charged bytes, and [`StructureCache::get_or_try_build`] makes concurrent
//! misses of one key build once. ([`StructureCache::publish`] admits a value
//! built outside the cache, for transactions that stage a build before
//! publishing it; racing builders of one key there all build, and the first
//! admission wins.)

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use quick_cache::sync::{Cache, EntryAction, GuardResult};
use quick_cache::{Equivalent, Lifecycle, OptionsBuilder, Weighter};

use crate::block_structure::may_publish_since;

/// The structure caches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum StructureCacheKind {
    /// Fusion-tree keys and sector structure per HomSpace sector signature,
    /// degeneracies excluded (TensorKit `sectorstructure`).
    SectorStructure,
    /// Complete block structure per HomSpace, degeneracies included
    /// (TensorKit `degeneracystructure`).
    DegeneracyStructure,
    /// Completed tree transformers (TensorKit `treetransposer` /
    /// `treebraider`), shared by every Runtime of the process.
    CompletedTreeTransformer,
    /// Composed fusion-tree transformation coefficients per source
    /// fusion-tree group, degeneracies excluded (TensorKit `fsbraid` /
    /// `fstranspose`), shared by every Runtime of the process.
    TreeTransformCoefficients,
}

impl StructureCacheKind {
    /// Every structure cache, in a fixed order.
    pub const ALL: [Self; 4] = [
        Self::SectorStructure,
        Self::DegeneracyStructure,
        Self::CompletedTreeTransformer,
        Self::TreeTransformCoefficients,
    ];
}

/// A snapshot of one structure cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructureCacheInfo {
    kind: StructureCacheKind,
    entries: usize,
    charged_bytes: u64,
    byte_budget: u64,
    max_entry_bytes: u64,
    hits: u64,
    misses: u64,
    admissions: u64,
    evictions: u64,
    rejections: u64,
}

impl StructureCacheInfo {
    pub fn kind(self) -> StructureCacheKind {
        self.kind
    }
    pub fn entries(self) -> usize {
        self.entries
    }
    /// Bytes charged for the resident entries: a conservative accounting of
    /// what each entry retains, not allocator-observed memory.
    pub fn charged_bytes(self) -> u64 {
        self.charged_bytes
    }
    pub fn byte_budget(self) -> u64 {
        self.byte_budget
    }
    /// The largest entry the cache admits: quick_cache's hot-queue share of
    /// one shard's budget.
    pub fn max_entry_bytes(self) -> u64 {
        self.max_entry_bytes
    }
    pub fn hits(self) -> u64 {
        self.hits
    }
    /// Completed builds offered for admission: admitted, rejected as
    /// oversize, or refused because a reset ran during the build. A build
    /// that loses a publication race to an equal key is not counted.
    pub fn misses(self) -> u64 {
        self.misses
    }
    pub fn admissions(self) -> u64 {
        self.admissions
    }
    pub fn evictions(self) -> u64 {
        self.evictions
    }
    /// Builds larger than [`Self::max_entry_bytes`], returned uncached.
    pub fn rejections(self) -> u64 {
        self.rejections
    }
}

struct Charged<V: ?Sized> {
    value: Arc<V>,
    bytes: u64,
}

// Why manual: a derive would bound `V: Clone`.
impl<V: ?Sized> Clone for Charged<V> {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            bytes: self.bytes,
        }
    }
}

/// quick_cache's default hot-queue share, set explicitly because the
/// largest admissible entry depends on it (`StructureCache::max_entry_bytes`).
const HOT_ALLOCATION: f64 = 0.97;

#[derive(Clone)]
struct ChargeWeighter;

impl<K, V: ?Sized> Weighter<K, Charged<V>> for ChargeWeighter {
    fn weight(&self, _: &K, charged: &Charged<V>) -> u64 {
        charged.bytes
    }
}

#[derive(Clone)]
struct CountEvictions(Arc<AtomicU64>);

impl<K, V> Lifecycle<K, V> for CountEvictions {
    /// Evicted items, dropped when the request ends, outside the shard lock
    /// (as quick_cache's default lifecycle does).
    type RequestState = Vec<(K, V)>;

    fn on_evict(&self, evicted: &mut Self::RequestState, key: K, value: V) {
        self.0.fetch_add(1, Ordering::Relaxed);
        evicted.push((key, value));
    }
}

/// One structure cache: values are `Arc<V>`, weighed by the bytes the caller
/// charges for them. Cross-crate only for the completed-transformer and
/// composed-coefficient caches, whose owners live in `symtn-tensors`.
#[doc(hidden)]
pub struct StructureCache<K, V: ?Sized> {
    kind: StructureCacheKind,
    entries: Cache<K, Charged<V>, ChargeWeighter, rustc_hash::FxBuildHasher, CountEvictions>,
    /// Read-held while an admission checks the reset epoch and inserts;
    /// write-held by [`Self::clear`]. A build that straddles a reset is
    /// therefore either wiped by the clear or refused by the epoch check.
    publication: RwLock<()>,
    /// The last admitted entry: a repeated hit compares keys (an `Arc`
    /// pointer first) instead of hashing one. The former layout cache kept
    /// the same front for the same reason. Why weak: the front must not keep
    /// an entry alive once the budget evicted it, so it retains no charged
    /// bytes; an evicted entry no caller holds simply misses here.
    last: RwLock<Option<(K, Weak<V>)>>,
    hits: AtomicU64,
    misses: AtomicU64,
    admissions: AtomicU64,
    evictions: Arc<AtomicU64>,
    rejections: AtomicU64,
}

impl<K, V> StructureCache<K, V>
where
    K: Eq + Hash + Clone,
    V: ?Sized,
{
    /// One shard: entries are heavy-tailed, and the largest must fit it
    /// (the U(1) `V^6 <- V^6` structures charge 30-46 MB, so more shards would
    /// force a larger budget). A hit takes only the shard's read lock, so one
    /// shard adds no contention.
    pub fn new(kind: StructureCacheKind, byte_budget: u64) -> Self {
        let evictions = Arc::new(AtomicU64::new(0));
        let options = OptionsBuilder::new()
            .shards(1)
            .hot_allocation(HOT_ALLOCATION)
            .weight_capacity(byte_budget)
            .estimated_items_capacity(1024)
            .build()
            .expect("structure cache options are valid");
        Self {
            kind,
            entries: Cache::with_options(
                options,
                ChargeWeighter,
                rustc_hash::FxBuildHasher,
                CountEvictions(Arc::clone(&evictions)),
            ),
            publication: RwLock::new(()),
            last: RwLock::new(None),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            admissions: AtomicU64::new(0),
            evictions,
            rejections: AtomicU64::new(0),
        }
    }

    /// A warm hit: one shard read, no allocation.
    pub fn get<Q>(&self, key: &Q) -> Option<Arc<V>>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        if let Some(value) = self.last_hit(key) {
            return Some(value);
        }
        let found = self.entries.get(key)?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(found.value)
    }

    fn last_hit<Q>(&self, key: &Q) -> Option<Arc<V>>
    where
        Q: Equivalent<K> + ?Sized,
    {
        let last = self
            .last
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (_, value) = last.as_ref().filter(|(last, _)| key.equivalent(last))?;
        let value = value.upgrade()?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(value)
    }

    /// Whether `key` is resident; counts no hit.
    pub(crate) fn contains<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let is_last = self
            .last
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|(last, value)| key.equivalent(last) && value.strong_count() > 0);
        is_last || self.entries.contains_key(key)
    }

    /// Publishes a value built since reset epoch `epoch`, or returns the
    /// entry another caller published first. Returned uncached when it is
    /// larger than one shard or its build straddled a reset. The flag says
    /// whether the returned value is resident: admitted now, or the earlier
    /// winner.
    ///
    /// The placeholder guard is held only across the epoch check and one
    /// insert: no caller code runs under it, so a concurrent caller of the
    /// same key never waits on a build.
    pub fn publish(&self, key: &K, value: Arc<V>, bytes: u64, epoch: usize) -> (Arc<V>, bool) {
        match self.entries.get_value_or_guard(key, None) {
            GuardResult::Value(existing) => (existing.value, true),
            GuardResult::Guard(guard) => self.admit(key, value, bytes, epoch, |charged| {
                let _ = guard.insert(charged);
            }),
            GuardResult::Timeout => unreachable!("a lookup without a timeout never times out"),
        }
    }

    /// Adds `bytes` to the charge of `key` while `value` is its resident
    /// entry: a lazily derived part of an entry joins the budget when it
    /// materializes. Never waits; a key being admitted is left alone, and an
    /// absent key is skipped, since no resident entry retains the part.
    ///
    /// Side effects of quick_cache's `entry`, all accepted:
    /// - the touch counts as a use of the entry for eviction order;
    /// - a weight increase does not evict by itself, so the cache may sit
    ///   above its budget by `bytes` until the next insert evicts;
    /// - between the residency check and `entry` the key may be evicted, and
    ///   then a placeholder is created and dropped, which forgets that key's
    ///   ghost (recently evicted) record. Rare and harmless: no value is
    ///   charged or retained.
    ///
    /// Returns whether it charged: `value` was the resident entry of `key`.
    pub(crate) fn add_charge(&self, key: &K, value: &Arc<V>, bytes: u64) -> bool {
        if !self.entries.contains_key(key) {
            return false;
        }
        matches!(
            self.entries
                .entry(key, Some(std::time::Duration::ZERO), |_, charged| {
                    let resident = Arc::ptr_eq(&charged.value, value);
                    if resident {
                        charged.bytes = charged.bytes.saturating_add(bytes);
                    }
                    EntryAction::Retain(resident)
                }),
            quick_cache::sync::EntryResult::Retained(true)
        )
    }

    /// The entry for `key`, built by `build` on a miss. Concurrent misses of
    /// one key build once; a failed build is not cached.
    pub(crate) fn get_or_try_build<E>(
        &self,
        key: &K,
        epoch: usize,
        build: impl FnOnce() -> Result<(Arc<V>, u64), E>,
    ) -> Result<Arc<V>, E> {
        if let Some(value) = self.last_hit(key) {
            return Ok(value);
        }
        match self.entries.get_value_or_guard(key, None) {
            GuardResult::Value(existing) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Ok(existing.value)
            }
            GuardResult::Guard(guard) => {
                let (value, bytes) = build()?;
                Ok(self
                    .admit(key, value, bytes, epoch, |charged| {
                        let _ = guard.insert(charged);
                    })
                    .0)
            }
            GuardResult::Timeout => unreachable!("a lookup without a timeout never times out"),
        }
    }

    fn admit(
        &self,
        key: &K,
        value: Arc<V>,
        bytes: u64,
        epoch: usize,
        insert: impl FnOnce(Charged<V>),
    ) -> (Arc<V>, bool) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        // quick_cache drops an item heavier than its shard without a trace;
        // the rejection is counted here instead.
        if bytes > self.max_entry_bytes() {
            self.rejections.fetch_add(1, Ordering::Relaxed);
            return (value, false);
        }
        let _publication = self
            .publication
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !may_publish_since(epoch) {
            return (value, false);
        }
        insert(Charged {
            value: Arc::clone(&value),
            bytes,
        });
        *self
            .last
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some((key.clone(), Arc::downgrade(&value)));
        self.admissions.fetch_add(1, Ordering::Relaxed);
        (value, true)
    }

    pub(crate) fn clear(&self) {
        let _publication = self
            .publication
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.entries.clear();
        self.forget_last();
        for counter in [&self.hits, &self.misses, &self.admissions, &self.rejections] {
            counter.store(0, Ordering::Relaxed);
        }
        self.evictions.store(0, Ordering::Relaxed);
    }

    pub(crate) fn set_byte_budget(&self, bytes: u64) {
        // Exclusive, as in `clear`: an admission in flight finishes before
        // the budget changes, so it cannot republish the front afterwards.
        let _publication = self
            .publication
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.entries.set_capacity(bytes);
        self.forget_last();
    }

    /// quick_cache admits an entry only up to the hot-queue target of its
    /// shard (`shard.rs` `replace_placeholder`), not the whole shard; this is
    /// that target, computed as quick_cache computes it.
    fn max_entry_bytes(&self) -> u64 {
        let shard = self.entries.shard_capacity();
        ((shard as f64 * HOT_ALLOCATION) as u64).clamp(shard.min(1), shard)
    }

    fn forget_last(&self) {
        *self
            .last
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    pub fn info(&self) -> StructureCacheInfo {
        StructureCacheInfo {
            kind: self.kind,
            entries: self.entries.len(),
            charged_bytes: self.entries.weight(),
            byte_budget: self.entries.capacity(),
            max_entry_bytes: self.max_entry_bytes(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            admissions: self.admissions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            rejections: self.rejections.load(Ordering::Relaxed),
        }
    }
}

/// The control surface of one registered cache, for a kind whose owner
/// lives outside `symtn-core`.
#[doc(hidden)]
pub trait ErasedStructureCacheControl: Sync {
    fn info(&self) -> StructureCacheInfo;
    fn clear(&self);
    fn set_byte_budget(&self, bytes: u64);
}

/// Process default of every structure cache's byte budget.
pub(crate) const DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET: u64 = 64 * 1024 * 1024;

/// The slot of an externally owned kind: its owner registers on first use.
/// A budget configured earlier is held here and handed to the owner at
/// registration.
struct ExternalSlot {
    control: OnceLock<&'static dyn ErasedStructureCacheControl>,
    pending_budget: Mutex<Option<u64>>,
}

impl ExternalSlot {
    const fn new() -> Self {
        Self {
            control: OnceLock::new(),
            pending_budget: Mutex::new(None),
        }
    }
}

static COMPLETED_TREE_TRANSFORMER: ExternalSlot = ExternalSlot::new();
static TREE_TRANSFORM_COEFFICIENTS: ExternalSlot = ExternalSlot::new();

fn external_slot(kind: StructureCacheKind) -> Option<&'static ExternalSlot> {
    match kind {
        StructureCacheKind::CompletedTreeTransformer => Some(&COMPLETED_TREE_TRANSFORMER),
        StructureCacheKind::TreeTransformCoefficients => Some(&TREE_TRANSFORM_COEFFICIENTS),
        StructureCacheKind::SectorStructure | StructureCacheKind::DegeneracyStructure => None,
    }
}

/// Why a registration was refused.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureCacheRegistrationError {
    /// `symtn-core` owns this kind itself.
    OwnedByCore,
    /// Another owner registered this kind first.
    AlreadyRegistered,
}

/// Registers the owner of an externally owned kind and returns the byte
/// budget it must start with. Called once, from the owner's lazy
/// initializer; the control may not call back into the owner's initializer.
///
/// Refusal is an error, not a panic. Why not a sealed token instead: any
/// token constructor this crate exposes to `symtn-tensors` is public to
/// every crate, so the slot cannot be reserved by type. A refused owner
/// must run without admitting anything, since the registry cannot clear
/// it.
#[doc(hidden)]
pub fn register_structure_cache(
    kind: StructureCacheKind,
    control: &'static dyn ErasedStructureCacheControl,
) -> Result<u64, StructureCacheRegistrationError> {
    let slot = external_slot(kind).ok_or(StructureCacheRegistrationError::OwnedByCore)?;
    let pending = slot
        .pending_budget
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    slot.control
        .set(control)
        .map_err(|_| StructureCacheRegistrationError::AlreadyRegistered)?;
    Ok(pending.unwrap_or(DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET))
}

/// A snapshot of one structure cache. A registered kind not yet in use
/// reports an empty cache with its pending budget.
#[doc(hidden)]
pub fn structure_cache_info(kind: StructureCacheKind) -> StructureCacheInfo {
    let slot = match kind {
        StructureCacheKind::SectorStructure => {
            return crate::fusion_space::sector_structure_cache().info()
        }
        StructureCacheKind::DegeneracyStructure => {
            return crate::fusion_space::degeneracy_structure_cache().info()
        }
        StructureCacheKind::CompletedTreeTransformer => &COMPLETED_TREE_TRANSFORMER,
        StructureCacheKind::TreeTransformCoefficients => &TREE_TRANSFORM_COEFFICIENTS,
    };
    if let Some(control) = slot.control.get() {
        return control.info();
    }
    let budget = slot
        .pending_budget
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .unwrap_or(DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET);
    StructureCacheInfo {
        kind,
        entries: 0,
        charged_bytes: 0,
        byte_budget: budget,
        // One shard, as every structure cache.
        max_entry_bytes: ((budget as f64 * HOT_ALLOCATION) as u64).clamp(budget.min(1), budget),
        hits: 0,
        misses: 0,
        admissions: 0,
        evictions: 0,
        rejections: 0,
    }
}

/// Snapshots of every structure cache, in [`StructureCacheKind::ALL`] order.
#[doc(hidden)]
pub fn structure_cache_infos() -> Vec<StructureCacheInfo> {
    StructureCacheKind::ALL
        .into_iter()
        .map(structure_cache_info)
        .collect()
}

/// Sets one cache's byte budget; entries beyond it are evicted. Owned by the
/// application: library code never calls it. A registered kind not yet in
/// use starts with this budget.
#[doc(hidden)]
pub fn set_structure_cache_byte_budget(kind: StructureCacheKind, bytes: u64) {
    let slot = match kind {
        StructureCacheKind::SectorStructure => {
            return crate::fusion_space::sector_structure_cache().set_byte_budget(bytes)
        }
        StructureCacheKind::DegeneracyStructure => {
            return crate::fusion_space::degeneracy_structure_cache().set_byte_budget(bytes)
        }
        StructureCacheKind::CompletedTreeTransformer => &COMPLETED_TREE_TRANSFORMER,
        StructureCacheKind::TreeTransformCoefficients => &TREE_TRANSFORM_COEFFICIENTS,
    };
    let control = {
        let mut pending = slot
            .pending_budget
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match slot.control.get() {
            Some(control) => *control,
            None => {
                *pending = Some(bytes);
                return;
            }
        }
    };
    // Outside the slot lock: the owner's initializer takes it.
    control.set_byte_budget(bytes);
}

/// Clears the registered kinds; the caller holds the reset lock and has
/// opened the odd epoch.
pub(crate) fn clear_registered_structure_caches() {
    for slot in [&COMPLETED_TREE_TRANSFORMER, &TREE_TRANSFORM_COEFFICIENTS] {
        if let Some(control) = slot.control.get() {
            control.clear();
        }
    }
}

impl<K, V> ErasedStructureCacheControl for StructureCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync,
    V: ?Sized + Send + Sync,
{
    fn info(&self) -> StructureCacheInfo {
        StructureCache::info(self)
    }

    fn clear(&self) {
        StructureCache::clear(self)
    }

    fn set_byte_budget(&self, bytes: u64) {
        StructureCache::set_byte_budget(self, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_structure::core_reset_epoch;

    /// Tests that publish at the live reset epoch must exclude every
    /// clear-all (it opens an odd epoch that refuses their publications).
    fn epoch_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn cache(byte_budget: u64) -> StructureCache<u32, u32> {
        StructureCache::new(StructureCacheKind::SectorStructure, byte_budget)
    }

    #[test]
    fn a_miss_builds_once_and_a_failed_build_is_not_cached() {
        let _guard = epoch_lock();
        let cache = cache(1 << 20);
        let epoch = core_reset_epoch();
        let failed = cache.get_or_try_build(&1, epoch, || Err::<(Arc<u32>, u64), _>("no"));
        assert_eq!(failed, Err("no"));
        assert!(cache.get(&1).is_none());
        let built = cache
            .get_or_try_build(&1, epoch, || Ok::<_, ()>((Arc::new(7), 64)))
            .unwrap();
        let again = cache
            .get_or_try_build(&1, epoch, || -> Result<_, ()> { panic!("rebuilt") })
            .unwrap();
        // What: the second call hits the published Arc; the error left no trace.
        assert!(Arc::ptr_eq(&built, &again));
        let info = cache.info();
        assert_eq!((info.hits(), info.misses(), info.admissions()), (1, 1, 1));
    }

    #[test]
    fn publish_keeps_the_first_value_of_a_key() {
        let _guard = epoch_lock();
        let cache = cache(1 << 20);
        let epoch = core_reset_epoch();
        let (first, admitted) = cache.publish(&3, Arc::new(1), 64, epoch);
        let (second, resident) = cache.publish(&3, Arc::new(2), 64, epoch);
        assert!(Arc::ptr_eq(&first, &second));
        // What: both report residency; the second caller got the winner.
        assert!(admitted && resident);
        assert_eq!(*cache.get(&3).unwrap(), 1);
    }

    #[test]
    fn an_entry_over_one_shard_is_rejected_and_counted() {
        let _guard = epoch_lock();
        let cache = cache(1 << 20);
        let max_entry = cache.info().max_entry_bytes();
        // The hot-queue share of the single shard, as quick_cache admits.
        assert_eq!(max_entry, ((1u64 << 20) as f64 * HOT_ALLOCATION) as u64);
        let (value, resident) = cache.publish(&5, Arc::new(9), max_entry + 1, core_reset_epoch());
        // What: the caller still gets its value; the cache keeps nothing and
        // says so (quick_cache alone would drop it silently).
        assert_eq!(*value, 9);
        assert!(!resident);
        assert!(cache.get(&5).is_none());
        let info = cache.info();
        assert_eq!((info.rejections(), info.admissions()), (1, 0));
    }

    #[test]
    fn a_build_from_a_stale_epoch_is_not_published() {
        let _guard = epoch_lock();
        let cache = cache(1 << 20);
        let stale = core_reset_epoch().wrapping_add(2);
        let (value, resident) = cache.publish(&6, Arc::new(4), 64, stale);
        assert_eq!(*value, 4);
        assert!(!resident);
        assert!(cache.get(&6).is_none());
    }

    #[test]
    fn the_byte_budget_bounds_the_resident_entries() {
        let _guard = epoch_lock();
        let cache = cache(64 * 1024);
        let epoch = core_reset_epoch();
        for key in 0..1024 {
            cache.publish(&key, Arc::new(key), 1024, epoch);
        }
        let info = cache.info();
        // What: weight, not entry count, bounds retention, and the excess
        // was evicted.
        assert!(info.charged_bytes() <= info.byte_budget());
        assert!(info.evictions() > 0);
        cache.set_byte_budget(0);
        assert_eq!(cache.info().entries(), 0);
        cache.clear();
        assert_eq!(cache.info().evictions(), 0);
    }

    #[test]
    fn every_kind_is_listed_once_and_the_tensor_owned_kinds_have_their_own_slots() {
        assert_eq!(StructureCacheKind::ALL.len(), 4);
        for kind in StructureCacheKind::ALL {
            assert_eq!(
                StructureCacheKind::ALL
                    .iter()
                    .filter(|listed| **listed == kind)
                    .count(),
                1
            );
        }
        // What: the two `symtn-tensors` caches register in distinct slots;
        // the core-owned kinds refuse registration.
        let completed = external_slot(StructureCacheKind::CompletedTreeTransformer).unwrap();
        let coefficients = external_slot(StructureCacheKind::TreeTransformCoefficients).unwrap();
        assert!(!std::ptr::eq(completed, coefficients));
        assert!(external_slot(StructureCacheKind::SectorStructure).is_none());
        assert!(external_slot(StructureCacheKind::DegeneracyStructure).is_none());
        let info = structure_cache_info(StructureCacheKind::TreeTransformCoefficients);
        assert_eq!(
            (info.kind(), info.entries()),
            (StructureCacheKind::TreeTransformCoefficients, 0)
        );
    }

    /// A stand-in for the externally owned completed-transformer cache: the
    /// registry only sees its control.
    struct FakeOwner;

    static FAKE_CACHE: OnceLock<StructureCache<u32, dyn std::any::Any + Send + Sync>> =
        OnceLock::new();
    static CLEARED_IN_ODD_EPOCH: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    fn fake_cache() -> &'static StructureCache<u32, dyn std::any::Any + Send + Sync> {
        FAKE_CACHE.get_or_init(|| {
            let budget =
                register_structure_cache(StructureCacheKind::CompletedTreeTransformer, &FakeOwner)
                    .unwrap();
            StructureCache::new(StructureCacheKind::CompletedTreeTransformer, budget)
        })
    }

    impl ErasedStructureCacheControl for FakeOwner {
        fn info(&self) -> StructureCacheInfo {
            fake_cache().info()
        }
        fn clear(&self) {
            CLEARED_IN_ODD_EPOCH.store(core_reset_epoch() % 2 == 1, Ordering::SeqCst);
            fake_cache().clear()
        }
        fn set_byte_budget(&self, bytes: u64) {
            fake_cache().set_byte_budget(bytes)
        }
    }

    #[test]
    fn the_registry_holds_a_budget_until_its_owner_registers_and_clears_it_in_the_epoch_window() {
        // Isolated: the clear-all below would refuse the unlocked sibling
        // cache tests' publications, and registration is once per process.
        if crate::test_support::run_isolated_or_return(
            "SYMTN_CORE_REGISTRY_ISOLATED",
            "cache::tests::the_registry_holds_a_budget_until_its_owner_registers_and_clears_it_in_the_epoch_window",
        ) {
            return;
        }
        let _guard = crate::test_support::CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let kind = StructureCacheKind::CompletedTreeTransformer;
        // What: before its owner exists, the kind reports an empty cache
        // with the configured budget, and that budget reaches the owner.
        set_structure_cache_byte_budget(kind, 1 << 20);
        let pending = structure_cache_info(kind);
        assert_eq!((pending.entries(), pending.byte_budget()), (0, 1 << 20));
        assert_eq!(
            pending.max_entry_bytes(),
            ((1u64 << 20) as f64 * HOT_ALLOCATION) as u64
        );
        assert_eq!(fake_cache().info().byte_budget(), 1 << 20);
        assert_eq!(structure_cache_info(kind), fake_cache().info());
        assert_eq!(structure_cache_infos().len(), StructureCacheKind::ALL.len());
        // What: the slot cannot be claimed twice, nor a core-owned kind at
        // all; both refusals leave the registered owner in place.
        assert_eq!(
            register_structure_cache(kind, &FakeOwner),
            Err(StructureCacheRegistrationError::AlreadyRegistered)
        );
        assert_eq!(
            register_structure_cache(StructureCacheKind::DegeneracyStructure, &FakeOwner),
            Err(StructureCacheRegistrationError::OwnedByCore)
        );
        assert_eq!(structure_cache_info(kind), fake_cache().info());

        let value: Arc<dyn std::any::Any + Send + Sync> = Arc::new(5_u64);
        let (resident, admitted) = fake_cache().publish(&1, value, 64, core_reset_epoch());
        assert!(admitted);
        // The erased value downcasts back without allocating.
        assert_eq!(*resident.downcast::<u64>().unwrap(), 5);
        assert_eq!(structure_cache_info(kind).entries(), 1);

        // What: the one clear-all reaches the registered kind, inside the
        // odd reset epoch that refuses straddling publications.
        crate::clear_structure_caches();
        assert!(CLEARED_IN_ODD_EPOCH.load(Ordering::SeqCst));
        assert_eq!(structure_cache_info(kind).entries(), 0);

        // After registration a budget goes straight to the owner.
        set_structure_cache_byte_budget(kind, 0);
        assert_eq!(fake_cache().info().byte_budget(), 0);
        let (_, admitted) = fake_cache().publish(&2, Arc::new(1_u8), 1, core_reset_epoch());
        assert!(!admitted);
        set_structure_cache_byte_budget(kind, DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET);
    }
}
