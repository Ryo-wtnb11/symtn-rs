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
//! publishing it; racing builders of one key there all build, the first
//! admission wins, and the others count in
//! [`StructureCacheInfo::duplicate_builds`].)

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

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
    /// `treebraider`) and compiled fusion-aware traces, shared by every
    /// Runtime of the process.
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
///
/// The counters are summed over per-thread stripes without stopping the
/// threads that count: exact once they are quiescent, eventually consistent
/// while they run.
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
    duplicate_builds: u64,
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
    /// that loses a publication race to an equal key is not counted here but
    /// in [`Self::duplicate_builds`].
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
    /// Completed builds not retained because a value for the same key was
    /// already resident when they were published; the resident value was
    /// returned instead. The waste of concurrent misses of one key on a
    /// cache without single-flight. Rejections and builds that straddled a
    /// reset are not counted here. Concurrent misses that the cache resolves
    /// with single-flight (one builds, the others wait for its entry) never
    /// add to it, so a cache admitted only that way reports 0.
    pub fn duplicate_builds(self) -> u64 {
        self.duplicate_builds
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

/// The hot-queue share of each cache: half, so every admissible entry also
/// fits the cold queue.
///
/// quick_cache admits an entry only up to the hot target
/// (`StructureCache::max_entry_bytes`). Once the hot queue is full, a new key
/// enters the cold queue unreferenced, and the over-budget sweep
/// (`shard.rs:advance_cold`) evicts unreferenced cold entries oldest first.
/// The new entry is the youngest. It survives the sweep when its weight fits
/// the cold share and the hot queue is within its target, because the sweep
/// stops at the latest when cold holds only the new entry. (A re-admitted
/// ghost key enters hot past its target; until the next promotion trims hot,
/// a new cold entry can still be swept.)
///
/// Why not quick_cache's default 0.97: an entry larger than the 3% cold share
/// was evicted by its own admission, so every repeat rebuilt it. The price is
/// a largest admissible entry of half the budget instead of 97%, so the
/// default budget is 128 MiB (64 MiB per entry); a larger entry is rejected
/// and counted in [`StructureCacheInfo::rejections`].
///
/// The share must survive budget changes, and quick_cache 0.7.0's
/// `set_capacity` does not keep it on its own (`shard.rs:set_capacity`): from
/// capacity 0 it resets the hot target to its default 0.97, and otherwise it
/// carries over the old target/capacity ratio, which truncation has moved
/// off one half. [`quick_cache_capacity`] therefore hands quick_cache only
/// even capacities, whose half is exact, and never 0.
const HOT_ALLOCATION: f64 = 0.5;

/// The capacity quick_cache gets for byte budget `budget`: the largest even
/// number not above it, so the hot target is exactly half and an exact half
/// carries over to the next capacity; `None` below 2, where nothing is
/// admitted. Exact for budgets up to 2^53 bytes.
///
/// Why not rebuild the quick_cache instance on a budget change instead: the
/// instance would then sit behind a lock or a swappable pointer that every
/// hit reads, and std has no lock-free swap.
fn quick_cache_capacity(budget: u64) -> Option<u64> {
    (budget >= 2).then_some(budget & !1)
}

/// The largest entry a cache with byte budget `budget` admits: the hot
/// target of [`quick_cache_capacity`], which quick_cache enforces
/// (`shard.rs:replace_placeholder`).
fn max_entry_bytes_for(budget: u64) -> u64 {
    quick_cache_capacity(budget).map_or(0, |capacity| capacity / 2)
}

#[derive(Clone)]
struct ChargeWeighter;

impl<K, V: ?Sized> Weighter<K, Charged<V>> for ChargeWeighter {
    fn weight(&self, _: &K, charged: &Charged<V>) -> u64 {
        charged.bytes
    }
}

/// Counter stripes per cache, as fusion-symbols stripes its hit counters
/// (0.9.1 `src/cache.rs:HIT_STRIPES`): twice quick_cache's eight
/// `sharded-lock` reader stripes, so up to 16 concurrent threads count on
/// distinct cache lines while a snapshot sums only 16 lines.
const COUNTER_STRIPES: usize = 16;

/// One thread's counters, alone on its cache line pair.
#[repr(align(128))]
#[derive(Default)]
struct CounterStripe {
    hits: AtomicU64,
    misses: AtomicU64,
    admissions: AtomicU64,
    evictions: AtomicU64,
    rejections: AtomicU64,
    duplicate_builds: AtomicU64,
}

type Counters = [CounterStripe; COUNTER_STRIPES];

/// This thread's counter stripe, assigned round-robin on first use.
fn counter_stripe(counters: &Counters) -> &CounterStripe {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static STRIPE: usize = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    &counters[STRIPE.try_with(|stripe| *stripe).unwrap_or(0) % COUNTER_STRIPES]
}

/// The sum of one counter over every stripe. Relaxed and unsynchronized
/// with the counting threads: exact once they are quiescent, eventually
/// consistent while they run.
fn counter_sum(counters: &Counters, counter: impl Fn(&CounterStripe) -> &AtomicU64) -> u64 {
    counters
        .iter()
        .map(|stripe| counter(stripe).load(Ordering::Relaxed))
        .sum()
}

#[derive(Clone)]
struct CountEvictions(Arc<Counters>);

impl<K, V> Lifecycle<K, V> for CountEvictions {
    /// Evicted items, dropped when the request ends, outside the shard lock
    /// (as quick_cache's default lifecycle does).
    type RequestState = Vec<(K, V)>;

    fn on_evict(&self, evicted: &mut Self::RequestState, key: K, value: V) {
        counter_stripe(&self.0)
            .evictions
            .fetch_add(1, Ordering::Relaxed);
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
    /// The byte budget, which bounds the charged bytes. quick_cache holds
    /// [`quick_cache_capacity`] of it; written under the publication write
    /// lock.
    budget: AtomicU64,
    /// Striped so that a warm hit writes no cache line every thread writes;
    /// shared with the eviction lifecycle.
    counters: Arc<Counters>,
}

impl<K, V> StructureCache<K, V>
where
    K: Eq + Hash + Clone,
    V: ?Sized,
{
    /// One shard: entries are heavy-tailed, and the largest must fit the
    /// hot share of the one shard (the U(1) `V^6 <- V^6` structures charge
    /// 30-46 MB, within the 64 MiB hot share of the default budget; more
    /// shards would force a larger budget). A hit takes only the shard's read lock, so one
    /// shard adds no contention.
    pub fn new(kind: StructureCacheKind, byte_budget: u64) -> Self {
        let counters: Arc<Counters> = Arc::default();
        // A budget below 2 admits nothing; quick_cache still gets a real
        // capacity so its ghost records keep a size for a later budget.
        let capacity =
            quick_cache_capacity(byte_budget).unwrap_or(DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET);
        let options = OptionsBuilder::new()
            .shards(1)
            .hot_allocation(HOT_ALLOCATION)
            .weight_capacity(capacity)
            .estimated_items_capacity(1024)
            .build()
            .expect("structure cache options are valid");
        Self {
            kind,
            entries: Cache::with_options(
                options,
                ChargeWeighter,
                rustc_hash::FxBuildHasher,
                CountEvictions(Arc::clone(&counters)),
            ),
            publication: RwLock::new(()),
            budget: AtomicU64::new(byte_budget),
            counters,
        }
    }

    /// A warm hit: one shard read, no allocation.
    ///
    /// Why no front holding the last admitted entry: a shared front's lock
    /// is a word every hit writes, so hits stopped scaling past one thread.
    pub fn get<Q>(&self, key: &Q) -> Option<Arc<V>>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.probe_lookup();
        let found = self.entries.get(key)?;
        self.stripe().hits.fetch_add(1, Ordering::Relaxed);
        Some(found.value)
    }

    fn stripe(&self) -> &CounterStripe {
        counter_stripe(&self.counters)
    }

    #[inline]
    fn probe_lookup(&self) {
        #[cfg(test)]
        if self.kind == StructureCacheKind::SectorStructure {
            crate::probe::hit(crate::probe::Event::SectorStructureLookup);
        }
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
        self.probe_lookup();
        match self.entries.get_value_or_guard(key, None) {
            GuardResult::Value(existing) => {
                self.stripe()
                    .duplicate_builds
                    .fetch_add(1, Ordering::Relaxed);
                (existing.value, true)
            }
            GuardResult::Guard(guard) => self.admit(value, bytes, epoch, |charged| {
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
        self.probe_lookup();
        match self.entries.get_value_or_guard(key, None) {
            GuardResult::Value(existing) => {
                self.stripe().hits.fetch_add(1, Ordering::Relaxed);
                Ok(existing.value)
            }
            GuardResult::Guard(guard) => {
                let (value, bytes) = build()?;
                Ok(self
                    .admit(value, bytes, epoch, |charged| {
                        let _ = guard.insert(charged);
                    })
                    .0)
            }
            GuardResult::Timeout => unreachable!("a lookup without a timeout never times out"),
        }
    }

    fn admit(
        &self,
        value: Arc<V>,
        bytes: u64,
        epoch: usize,
        insert: impl FnOnce(Charged<V>),
    ) -> (Arc<V>, bool) {
        let stripe = self.stripe();
        stripe.misses.fetch_add(1, Ordering::Relaxed);
        let _publication = self
            .publication
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // quick_cache drops an item heavier than its hot target without a
        // trace; the rejection is counted here instead. Checked under the
        // lock, so a budget change cannot slip between check and insert.
        let max_entry_bytes = self.max_entry_bytes();
        if max_entry_bytes == 0 || bytes > max_entry_bytes {
            stripe.rejections.fetch_add(1, Ordering::Relaxed);
            return (value, false);
        }
        if !may_publish_since(epoch) {
            return (value, false);
        }
        insert(Charged {
            value: Arc::clone(&value),
            bytes,
        });
        stripe.admissions.fetch_add(1, Ordering::Relaxed);
        (value, true)
    }

    pub(crate) fn clear(&self) {
        let _publication = self
            .publication
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.entries.clear();
        for stripe in self.counters.iter() {
            for counter in [
                &stripe.hits,
                &stripe.misses,
                &stripe.admissions,
                &stripe.evictions,
                &stripe.rejections,
                &stripe.duplicate_builds,
            ] {
                counter.store(0, Ordering::Relaxed);
            }
        }
    }

    pub(crate) fn set_byte_budget(&self, bytes: u64) {
        let _publication = self
            .publication
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.budget.store(bytes, Ordering::Relaxed);
        match quick_cache_capacity(bytes) {
            Some(capacity) => self.entries.set_capacity(capacity),
            None => {
                // Every resident entry is over this budget: evicted, and
                // counted as quick_cache's lifecycle counts evictions.
                // quick_cache keeps its last capacity, so its hot share is
                // still exact when a later budget re-enables the cache.
                let evicted = self.entries.len() as u64;
                self.entries.clear();
                self.stripe()
                    .evictions
                    .fetch_add(evicted, Ordering::Relaxed);
            }
        }
    }

    fn max_entry_bytes(&self) -> u64 {
        max_entry_bytes_for(self.budget.load(Ordering::Relaxed))
    }

    pub fn info(&self) -> StructureCacheInfo {
        StructureCacheInfo {
            kind: self.kind,
            entries: self.entries.len(),
            charged_bytes: self.entries.weight(),
            byte_budget: self.budget.load(Ordering::Relaxed),
            max_entry_bytes: self.max_entry_bytes(),
            hits: counter_sum(&self.counters, |stripe| &stripe.hits),
            misses: counter_sum(&self.counters, |stripe| &stripe.misses),
            admissions: counter_sum(&self.counters, |stripe| &stripe.admissions),
            evictions: counter_sum(&self.counters, |stripe| &stripe.evictions),
            rejections: counter_sum(&self.counters, |stripe| &stripe.rejections),
            duplicate_builds: counter_sum(&self.counters, |stripe| &stripe.duplicate_builds),
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
pub(crate) const DEFAULT_STRUCTURE_CACHE_BYTE_BUDGET: u64 = 128 * 1024 * 1024;

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
        max_entry_bytes: max_entry_bytes_for(budget),
        hits: 0,
        misses: 0,
        admissions: 0,
        evictions: 0,
        rejections: 0,
        duplicate_builds: 0,
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
    fn hits_counted_on_every_thread_sum_exactly_once_quiescent() {
        let _guard = epoch_lock();
        let cache = cache(1 << 20);
        let epoch = core_reset_epoch();
        cache.publish(&1, Arc::new(1), 64, epoch);
        cache.publish(&2, Arc::new(2), 64, epoch);
        const THREADS: u32 = 24;
        const HITS: u32 = 1000;
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let cache = &cache;
                scope.spawn(move || {
                    for i in 0..HITS {
                        assert!(cache.get(&(1 + (t + i) % 2)).is_some());
                    }
                    assert!(cache.get(&(100 + t)).is_none());
                    cache.publish(&50, Arc::new(t), 64, epoch);
                });
            }
        });
        // What: more threads than counter stripes, so stripes are shared, and
        // still no hit is lost or double counted; a lookup miss counts nothing;
        // of the racing publishers of one key, one is admitted and every other
        // is a duplicate.
        let info = cache.info();
        assert_eq!((info.hits(), info.misses()), (u64::from(THREADS * HITS), 3));
        assert_eq!(
            (info.admissions(), info.duplicate_builds()),
            (3, u64::from(THREADS) - 1)
        );
        cache.clear();
        let info = cache.info();
        assert_eq!((info.hits(), info.duplicate_builds()), (0, 0));
    }

    #[test]
    fn a_cleared_entry_is_not_served_to_any_thread_that_still_holds_it() {
        let _guard = epoch_lock();
        let cache = cache(1 << 20);
        let epoch = core_reset_epoch();
        const THREADS: u32 = 8;
        let admitted = std::sync::Barrier::new(THREADS as usize + 1);
        let cleared = std::sync::Barrier::new(THREADS as usize + 1);
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let (cache, admitted, cleared) = (&cache, &admitted, &cleared);
                scope.spawn(move || {
                    let held = cache.publish(&t, Arc::new(t), 64, epoch).0;
                    assert!(Arc::ptr_eq(&cache.get(&t).unwrap(), &held));
                    admitted.wait();
                    cleared.wait();
                    // What: the held Arc keeps the entry alive, yet the cleared cache
                    // misses, and the next admission is what it serves.
                    assert!(cache.get(&t).is_none());
                    let fresh = cache.publish(&t, Arc::new(t + 100), 64, epoch).0;
                    assert!(Arc::ptr_eq(&cache.get(&t).unwrap(), &fresh));
                    assert_eq!(*held, t);
                });
            }
            admitted.wait();
            cache.clear();
            cleared.wait();
        });
    }

    /// Fills `cache` (budget 1 MiB) to 94% with entries in use, then repeats a
    /// new entry of a fifth of the budget with no caller holding it.
    fn assert_a_new_entry_in_a_full_cache_is_built_once(cache: &StructureCache<u32, u32>) {
        let epoch = core_reset_epoch();
        for key in 0..16 {
            cache.publish(&key, Arc::new(key), 60 << 10, epoch);
        }
        for _ in 0..3 {
            for key in 0..16 {
                assert!(cache.get(&key).is_some());
            }
        }
        let mut builds = 0;
        for _ in 0..4 {
            let value = cache
                .get_or_try_build(&100, epoch, || {
                    builds += 1;
                    Ok::<_, ()>((Arc::new(100), 200 << 10))
                })
                .unwrap();
            assert_eq!(*value, 100);
        }
        // What: built once and then hit; the budget evicted older entries
        // to fit it, and still bounds the charged bytes.
        assert_eq!(builds, 1);
        assert!(cache.get(&100).is_some());
        let info = cache.info();
        assert!(info.charged_bytes() <= info.byte_budget());
        assert!(info.evictions() > 0);
    }

    #[test]
    fn an_entry_admitted_into_a_full_cache_is_served_on_its_next_repeat() {
        let _guard = epoch_lock();
        assert_a_new_entry_in_a_full_cache_is_built_once(&cache(1 << 20));
    }

    #[test]
    fn a_cache_re_enabled_after_a_disabling_budget_keeps_new_entries() {
        let _guard = epoch_lock();
        // What: disabling a cache with a budget of 0 or 1 and enabling it
        // again keeps the half hot share, so the full-cache admission still
        // builds once (quick_cache's own `set_capacity` would restore a 97%
        // or 100% hot share here).
        for disabling in [0, 1] {
            let cache = cache(1 << 20);
            cache.set_byte_budget(disabling);
            cache.set_byte_budget(1 << 20);
            assert_a_new_entry_in_a_full_cache_is_built_once(&cache);
            let built_disabled = cache_with_budget_then(disabling, 1 << 20);
            assert_a_new_entry_in_a_full_cache_is_built_once(&built_disabled);
        }
    }

    fn cache_with_budget_then(first: u64, then: u64) -> StructureCache<u32, u32> {
        let cache = cache(first);
        cache.set_byte_budget(then);
        cache
    }

    #[test]
    fn the_reported_largest_entry_is_what_the_cache_admits_after_any_budget_change() {
        let _guard = epoch_lock();
        let epoch = core_reset_epoch();
        let changes: [&[u64]; 7] = [
            &[1 << 20],
            &[0, 1 << 20],
            &[1, 1 << 20],
            &[(1 << 20) + 1],
            &[3],
            &[1 << 20, 4097, 1 << 16],
            &[0],
        ];
        for budgets in changes {
            let cache = cache(1 << 20);
            for &budget in budgets {
                cache.set_byte_budget(budget);
            }
            let info = cache.info();
            let max = info.max_entry_bytes();
            assert_eq!(info.byte_budget(), *budgets.last().unwrap());
            // What: an entry of exactly the reported size is admitted and
            // resident; one byte more is rejected and counted.
            if max > 0 {
                let (_, resident) = cache.publish(&1, Arc::new(1), max, epoch);
                assert!(resident, "{budgets:?}");
                assert!(cache.get(&1).is_some(), "{budgets:?}");
            }
            let (_, resident) = cache.publish(&2, Arc::new(2), max + 1, epoch);
            assert!(!resident, "{budgets:?}");
            assert!(cache.get(&2).is_none(), "{budgets:?}");
            assert_eq!(cache.info().rejections(), 1, "{budgets:?}");
            assert!(cache.info().charged_bytes() <= cache.info().byte_budget());
        }
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
        // The second build was wasted: counted as a duplicate, not a miss.
        let info = cache.info();
        assert_eq!(
            (info.misses(), info.admissions(), info.duplicate_builds()),
            (1, 1, 1)
        );
        cache.clear();
        assert_eq!(cache.info().duplicate_builds(), 0);
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
        // A rejected build is not a duplicate, nor is the next one of the
        // same key: nothing was resident.
        cache.publish(&5, Arc::new(9), max_entry + 1, core_reset_epoch());
        let info = cache.info();
        assert_eq!(
            (
                info.rejections(),
                info.admissions(),
                info.duplicate_builds()
            ),
            (2, 0, 0)
        );
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
        cache.publish(&6, Arc::new(4), 64, stale);
        assert_eq!(cache.info().duplicate_builds(), 0);
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
