//! The process-global composed-coefficient cache: the
//! transform specs of one source fusion-tree group, keyed on sectors and
//! trees only.
//!
//! Reference: TensorKit `cfaa073e` `braiding_manipulations.jl:fsbraid`
//! (`:288-339`) and `duality_manipulations.jl:fstranspose` (`:526-569`)
//! memoize `dst block => U` per `FusionTreeBlock` and `p` (and `levels`),
//! with `CacheStyle = NoCache()` for `UniqueFusion` and a global LRU
//! otherwise; `GenericTreeTransformer` (`treetransformers.jl:53-114`) looks
//! each source block up on a completed-transformer miss. QSpace `d2d3d7da`
//! `X3Map::contractDegQ` (`clebsch_aux.cc:4512-4560`) caches composed CGC
//! overlaps process-globally, degeneracy-free, and skips abelian `CRef`s
//! (`:4521-4526`); symtn borrows that technique, not its store.
//!
//! A cache-3 miss whose sectors were seen before (a degeneracy-only change,
//! or another HomSpace sharing a source group) therefore makes no F/R
//! provider call.
//!
//! Trace (#2072) keeps its group here too, under its own scope
//! [`TreeTransformScope::TraceTerms`] and value form
//! ([`BlockSourceColumns`]: present entries only, so a present zero stays
//! distinct from an absent one, which the dense transform spec merges). The
//! trace of TensorKit `_trace_permute!` permutes a `FusionTreeBlock` through
//! the same cached `fsbraid`, then splits every row on every call; symtn
//! stores the rows already lowered (split, `g₁ == g₂`, channel factor,
//! #2149), since its arena-backed, provider-admitted split costs allocations
//! and queries per row where TensorKit's isbits split costs none. Unique
//! trace stays uncached, as TensorKit's `NoCache`, since a trace hit saves
//! no work over one phase per source.
//!
//! Unique fusion never enters this cache, as TensorKit's `NoCache`: its
//! cache-3 miss recomputes one phase per block on a reused scratch and
//! allocates nothing per block, so a hit would save no allocation.
//!
//! Rust deviations, each deliberate:
//! - entries are byte-weighted under the per-kind budget of `symtn::cache`,
//!   not counted; a group heavier than one shard is rebuilt per miss;
//! - values are erased (`dyn Any`, the coefficient `TypeId` in the key);
//! - the key holds a group's ordered tree subset, since an expert structure
//!   may hold part of a `FusionTreeBlock`; the builder family (`mode`,
//!   `scope`) is keyed because the cache is shared by all of them;
//! - `storage_conjugate` is not keyed: no group builder reads it, binding
//!   applies it;
//! - no single-flight, a reset epoch, and built groups are staged in a
//!   [`PendingCoefficientGroups`] that the caller publishes only after its
//!   transaction succeeded (checked Generic: after the destination commit).

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use symtn_core::{
    BlockSourceColumns, DedupTreeArenaBuilder, ErasedStructureCacheControl, FusionTreeGroupKey,
    FusionTreePairKey, FusionTreePairOrientation, RuleIdentity, StructureCache, StructureCacheInfo,
    StructureCacheKind, TreeRehomer,
};

use symtn_core::BlockStructure;
use symtn_operations::transform_plan::TreeTransformSpecGroup;
use symtn_operations::TreeTransformStructure;

use super::{
    publish_committed, publishable, CompletedTransformerKey, TransformerMode, TreeTransformScope,
    ENTRY_OVERHEAD_BYTES,
};
use crate::tree_transform::operation::TreeTransformOperation;
use crate::TreeTransformGroupBlockSpec;

#[cfg(test)]
#[path = "coefficients_tests.rs"]
mod tests;

type ErasedEntry = dyn Any + Send + Sync;

const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();

/// Everything one source group's specs read besides the group's own trees.
#[derive(Debug, Eq, Hash, PartialEq)]
struct GroupContext {
    rule: RuleIdentity,
    mode: TransformerMode,
    coefficient: TypeId,
    scope: TreeTransformScope,
    operation: TreeTransformOperation,
    orientation: FusionTreePairOrientation,
}

/// Key of one source group's composed specs: the context plus the group's
/// external sectors and duals and its ordered source tree pairs (coupled
/// sectors, inner lines, vertices, basis order). No degeneracy, offset,
/// block index or content id: no group builder reads them.
///
/// The hash is computed once; a lookup compares through [`GroupKeyRef`]
/// without cloning the tree list.
#[derive(Clone)]
pub(crate) struct CoefficientGroupKey {
    context: Arc<GroupContext>,
    group_key: FusionTreeGroupKey,
    src_keys: Arc<[FusionTreePairKey]>,
    hash: u64,
}

impl PartialEq for CoefficientGroupKey {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && (Arc::ptr_eq(&self.context, &other.context) || self.context == other.context)
            && self.group_key == other.group_key
            && self.src_keys == other.src_keys
    }
}

impl Eq for CoefficientGroupKey {}

impl Hash for CoefficientGroupKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

/// Borrowed lookup form of a [`CoefficientGroupKey`].
struct GroupKeyRef<'a> {
    context: &'a Arc<GroupContext>,
    group_key: &'a FusionTreeGroupKey,
    src_keys: &'a [&'a FusionTreePairKey],
    hash: u64,
}

impl Hash for GroupKeyRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl symtn_core::StructureCacheEquivalent<CoefficientGroupKey> for GroupKeyRef<'_> {
    fn equivalent(&self, key: &CoefficientGroupKey) -> bool {
        self.hash == key.hash
            && (Arc::ptr_eq(self.context, &key.context) || **self.context == *key.context)
            && *self.group_key == key.group_key
            && self.src_keys.len() == key.src_keys.len()
            && self
                .src_keys
                .iter()
                .zip(key.src_keys.iter())
                .all(|(lhs, rhs)| **lhs == *rhs)
    }
}

/// The cached value: one source group's specs, in builder order, shared
/// whole by a plan assembled from hits, so it copies no spec. A `Multi`
/// spec's recoupling matrix is a shared `Arc`, so a cache-3 core bound from
/// it shares those bytes; the payload slot it carries holds only a `Weak`.
pub(crate) type CoefficientGroupEntry<T> = TreeTransformSpecGroup<T>;

struct CoefficientGroupsControl;

impl ErasedStructureCacheControl for CoefficientGroupsControl {
    fn info(&self) -> StructureCacheInfo {
        coefficient_groups().info()
    }

    fn clear(&self) {
        ErasedStructureCacheControl::clear(coefficient_groups())
    }

    fn set_byte_budget(&self, bytes: u64) {
        ErasedStructureCacheControl::set_byte_budget(coefficient_groups(), bytes)
    }
}

fn coefficient_groups() -> &'static StructureCache<CoefficientGroupKey, ErasedEntry> {
    static CACHE: OnceLock<StructureCache<CoefficientGroupKey, ErasedEntry>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Refused only if another crate claimed the slot first; then nothing
        // is admitted and every group builds per miss, with equal results.
        let budget = symtn_core::register_structure_cache(
            StructureCacheKind::TreeTransformCoefficients,
            &CoefficientGroupsControl,
        )
        .unwrap_or(0);
        StructureCache::new(StructureCacheKind::TreeTransformCoefficients, budget)
    })
}

/// One source group: its external-sector key and its ordered tree pairs.
pub(crate) type SourceGroup<'a> = (&'a FusionTreeGroupKey, &'a [&'a FusionTreePairKey]);

/// One source group's cache state during a plan build.
pub(crate) enum GroupSlot<T> {
    /// The resident specs of an unchanged group.
    Hit(Arc<CoefficientGroupEntry<T>>),
    /// The group must be built; carries its key hash for staging.
    Miss(u64),
}

/// Groups built during one resolution, not yet visible to other callers.
/// Errors are never staged; the owner publishes the batch once its
/// transaction succeeded, or drops it.
#[derive(Default)]
#[must_use = "built groups are cached only when published"]
pub(crate) struct PendingCoefficientGroups {
    groups: Vec<(CoefficientGroupKey, Arc<ErasedEntry>, u64)>,
}

impl PendingCoefficientGroups {
    /// Publishes every staged group built since reset epoch `epoch`; the
    /// first resident value of a key wins.
    pub(crate) fn publish(self, epoch: usize) {
        for (key, entry, bytes) in self.groups {
            #[cfg(test)]
            crate::probe::hit(
                if matches!(key.context.scope, TreeTransformScope::TraceTerms { .. }) {
                    crate::probe::Event::TracePublication
                } else {
                    crate::probe::Event::GroupPublication
                },
            );
            let _ = coefficient_groups().publish(&key, entry, bytes, epoch);
        }
    }
}

/// The checked Generic contraction's composed coefficients and completed
/// transformers, staged across its staged-operand and output transforms and
/// published only by [`Self::flush_committed`] once the contraction's
/// commits succeeded. One instance per contraction call: holding it across
/// calls would publish what a failed call built.
#[must_use = "built groups are cached only when flushed after the commit"]
pub(crate) struct CheckedPendingCoefficients {
    pending: PendingCoefficientGroups,
    transformers: Vec<(CompletedTransformerKey, TreeTransformStructure<f64>)>,
    epoch: usize,
}

impl CheckedPendingCoefficients {
    /// Captures the reset epoch: create it before the call's first build.
    pub(crate) fn new() -> Self {
        Self {
            pending: PendingCoefficientGroups::default(),
            transformers: Vec::new(),
            epoch: symtn_core::core_reset_epoch(),
        }
    }

    pub(crate) fn stage(&mut self, built: PendingCoefficientGroups) {
        self.pending.groups.extend(built.groups);
    }

    /// Stages a completed transformer built on a cache-3 miss under its
    /// preview `key`; [`Self::flush_committed`] re-keys it.
    pub(crate) fn stage_transformer(
        &mut self,
        key: CompletedTransformerKey,
        built: &TreeTransformStructure<f64>,
    ) {
        self.transformers.push((key, built.clone()));
    }

    /// Call only after the destination commit succeeded.
    pub(crate) fn flush(self) {
        self.flush_committed(&[]);
    }

    /// Publishes the staged groups, then each staged transformer whose
    /// structures are committed. `committed` maps each preview a transformer
    /// was built on to its committed structure (the resident winner after a
    /// race). #2128's rule: the key is re-formed over the committed ids and
    /// published only when the committed structure equals the preview and
    /// every keyed structure is canonical, so two previews of one HomSpace
    /// in a call converge on one id within that call.
    pub(crate) fn flush_committed(self, committed: &[(Arc<BlockStructure>, Arc<BlockStructure>)]) {
        self.pending.publish(self.epoch);
        let committed_of = |built: &Arc<BlockStructure>| {
            committed
                .iter()
                .find(|(preview, _)| preview.content_id() == built.content_id())
                .map_or_else(|| Arc::clone(built), |(_, committed)| Arc::clone(committed))
        };
        for (key, built) in self.transformers {
            let (dst, src) = (
                committed_of(built.dst_structure()),
                committed_of(built.src_structure()),
            );
            if dst.as_ref() == built.dst_structure().as_ref()
                && src.as_ref() == built.src_structure().as_ref()
                && publishable([dst.as_ref(), src.as_ref()])
            {
                let key = CompletedTransformerKey {
                    dst: dst.content_id(),
                    src: src.content_id(),
                    ..key
                };
                publish_committed(&key, built.replay_core(), self.epoch);
            }
        }
    }
}

/// The key context of one build's lookups, hashed once; shared by the
/// transform and trace reuse handles.
struct ReuseContext {
    context: Arc<GroupContext>,
    context_hash: u64,
}

impl ReuseContext {
    fn new<T: 'static>(
        rule: RuleIdentity,
        mode: TransformerMode,
        scope: TreeTransformScope,
        operation: TreeTransformOperation,
        orientation: FusionTreePairOrientation,
    ) -> Self {
        let context = GroupContext {
            rule,
            mode,
            coefficient: TypeId::of::<T>(),
            scope,
            operation,
            orientation,
        };
        let mut hasher = rustc_hash::FxHasher::default();
        context.hash(&mut hasher);
        Self {
            context_hash: hasher.finish(),
            context: Arc::new(context),
        }
    }

    fn group_hash(&self, group_key: &FusionTreeGroupKey, src_keys: &[&FusionTreePairKey]) -> u64 {
        let mut hasher = rustc_hash::FxHasher::default();
        hasher.write_u64(self.context_hash);
        group_key.hash(&mut hasher);
        for key in src_keys {
            key.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The resident entry of one group, or the hash to stage it under.
    fn get(&self, (group_key, src_keys): SourceGroup<'_>) -> Result<Arc<ErasedEntry>, u64> {
        let hash = self.group_hash(group_key, src_keys);
        let view = GroupKeyRef {
            context: &self.context,
            group_key,
            src_keys,
            hash,
        };
        coefficient_groups().get(&view).ok_or(hash)
    }

    /// Rule O (#2192): the owned key of a freshly built group and its value
    /// point into one arena the entry owns, never into the source structure,
    /// so an entry retains and is charged only its own trees. The returned
    /// re-homer then yields `value_pairs`' new keys, in order.
    fn rehomed_key<'v>(
        &self,
        hash: u64,
        (group_key, src_keys): SourceGroup<'v>,
        arena: &mut DedupTreeArenaBuilder,
        value_pairs: impl Iterator<Item = &'v FusionTreePairKey> + Clone,
    ) -> (CoefficientGroupKey, TreeRehomer) {
        arena.reserve(
            src_keys.iter().copied().chain(value_pairs.clone()),
            std::iter::once(group_key),
        );
        arena.insert_group(group_key);
        for key in src_keys {
            arena.insert_pair(key);
        }
        for key in value_pairs {
            arena.insert_pair(key);
        }
        let mut rehomer = arena.rehome();
        let key = CoefficientGroupKey {
            context: Arc::clone(&self.context),
            group_key: rehomer.group(),
            src_keys: src_keys.iter().map(|_| rehomer.pair()).collect(),
            hash,
        };
        (key, rehomer)
    }
}

/// Per-group reuse handle of one plan build: looks source groups up and
/// stages the built ones.
pub(crate) struct CoefficientGroupReuse<T> {
    context: ReuseContext,
    built: RefCell<PendingCoefficientGroups>,
    /// Entry-arena scratch reused across this build's staged groups.
    arena: RefCell<DedupTreeArenaBuilder>,
    coefficient: PhantomData<fn() -> T>,
}

impl<T> CoefficientGroupReuse<T>
where
    T: 'static + Send + Sync,
{
    pub(crate) fn new(
        rule: RuleIdentity,
        mode: TransformerMode,
        scope: TreeTransformScope,
        operation: &TreeTransformOperation,
        orientation: FusionTreePairOrientation,
    ) -> Self {
        Self {
            context: ReuseContext::new::<T>(rule, mode, scope, operation.clone(), orientation),
            built: RefCell::new(PendingCoefficientGroups::default()),
            arena: RefCell::default(),
            coefficient: PhantomData,
        }
    }

    /// Looks every group up, in order.
    pub(crate) fn lookup(&self, groups: &[SourceGroup<'_>]) -> Vec<GroupSlot<T>> {
        groups
            .iter()
            .map(|&group| match self.context.get(group) {
                Ok(entry) => {
                    #[cfg(test)]
                    crate::probe::hit(crate::probe::Event::GroupHit);
                    GroupSlot::Hit(entry.downcast::<CoefficientGroupEntry<T>>().unwrap_or_else(
                        |_| unreachable!("the key's TypeId and scope fix the entry type"),
                    ))
                }
                Err(hash) => {
                    #[cfg(test)]
                    crate::probe::hit(crate::probe::Event::GroupMiss);
                    GroupSlot::Miss(hash)
                }
            })
            .collect()
    }

    /// Stages one freshly built group.
    pub(crate) fn stage(
        &self,
        hash: u64,
        group: SourceGroup<'_>,
        mut specs: Vec<TreeTransformGroupBlockSpec<T>>,
    ) -> Arc<CoefficientGroupEntry<T>> {
        let mut arena = self.arena.borrow_mut();
        let (key, mut rehomer) = self.context.rehomed_key(
            hash,
            group,
            &mut arena,
            specs
                .iter()
                .flat_map(|spec| spec.dst_keys().iter().chain(spec.src_keys())),
        );
        for spec in &mut specs {
            spec.rehome_keys(|_| rehomer.pair());
        }
        arena.recycle(rehomer);
        drop(arena);
        let entry = Arc::new(CoefficientGroupEntry::new(specs));
        debug_assert!(owns_one_arena(
            &key,
            entry
                .specs()
                .iter()
                .flat_map(|spec| spec.dst_keys().iter().chain(spec.src_keys()))
        ));
        let bytes =
            charged_key_bytes(&key).saturating_add(charged_entry_bytes(entry.specs())) as u64;
        self.built
            .borrow_mut()
            .groups
            .push((key, Arc::clone(&entry) as Arc<ErasedEntry>, bytes));
        entry
    }

    /// The groups staged by this build, for the owner to publish.
    pub(crate) fn into_pending(self) -> PendingCoefficientGroups {
        self.built.into_inner()
    }
}

/// One trace group's lowered terms: per source column, the
/// `(open destination tree pair, coefficient)` rows that survive the trace
/// selection, in permutation row order.
pub(crate) type TraceTerms<T> = BlockSourceColumns<FusionTreePairKey, T>;

/// Trace-term reuse of one trace compile (scope
/// [`TreeTransformScope::TraceTerms`]). A hit stands for the build's member
/// admission, composition and lowering as well: the build validated every
/// member with a provider of the same identity and mode before composing,
/// and only a whole successful call publishes.
pub(crate) struct TraceTermReuse<T> {
    context: ReuseContext,
    built: PendingCoefficientGroups,
    arena: DedupTreeArenaBuilder,
    coefficient: PhantomData<fn() -> T>,
}

impl<T> TraceTermReuse<T>
where
    T: 'static + Send + Sync,
{
    /// `operation` is the trace's `(p…, q…)` permutation; the lowering
    /// splits each permuted codomain tree at `open_codomain_rank`.
    pub(crate) fn new(
        rule: RuleIdentity,
        mode: TransformerMode,
        operation: TreeTransformOperation,
        orientation: FusionTreePairOrientation,
        open_codomain_rank: usize,
    ) -> Self {
        Self {
            context: ReuseContext::new::<T>(
                rule,
                mode,
                TreeTransformScope::TraceTerms { open_codomain_rank },
                operation,
                orientation,
            ),
            built: PendingCoefficientGroups::default(),
            arena: DedupTreeArenaBuilder::default(),
            coefficient: PhantomData,
        }
    }

    /// The resident terms of one group (storage keys, storage order), or
    /// the hash to stage its build under.
    pub(crate) fn lookup(&self, group: SourceGroup<'_>) -> Result<Arc<TraceTerms<T>>, u64> {
        match self.context.get(group) {
            Ok(entry) => {
                #[cfg(test)]
                crate::probe::hit(crate::probe::Event::TraceHit);
                Ok(entry.downcast::<TraceTerms<T>>().unwrap_or_else(|_| {
                    unreachable!("the key's TypeId and scope fix the entry type")
                }))
            }
            Err(hash) => {
                #[cfg(test)]
                crate::probe::hit(crate::probe::Event::TraceMiss);
                Err(hash)
            }
        }
    }

    /// Stages one freshly built group.
    pub(crate) fn stage(
        &mut self,
        hash: u64,
        group: SourceGroup<'_>,
        mut columns: TraceTerms<T>,
    ) -> Arc<TraceTerms<T>> {
        let (key, mut rehomer) =
            self.context
                .rehomed_key(hash, group, &mut self.arena, columns.destinations().iter());
        columns.rehome_destinations(|_| rehomer.pair());
        self.arena.recycle(rehomer);
        let columns = Arc::new(columns);
        debug_assert!(owns_one_arena(&key, columns.destinations().iter()));
        let bytes = charged_key_bytes(&key).saturating_add(charged_trace_bytes(&columns)) as u64;
        self.built
            .groups
            .push((key, Arc::clone(&columns) as Arc<ErasedEntry>, bytes));
        columns
    }

    /// The groups staged by this compile, for the owner to publish.
    pub(crate) fn into_pending(self) -> PendingCoefficientGroups {
        self.built
    }
}

/// Single-arena ownership: an entry's key and value trees all point into the one
/// arena the entry owns, the group key's.
fn owns_one_arena<'k>(
    key: &'k CoefficientGroupKey,
    value_pairs: impl IntoIterator<Item = &'k FusionTreePairKey>,
) -> bool {
    let arena = key.group_key.codomain_tree();
    arena.shares_arena(key.group_key.domain_tree())
        && key.src_keys.iter().chain(value_pairs).all(|pair| {
            arena.shares_arena(pair.codomain_tree()) && arena.shares_arena(pair.domain_tree())
        })
}

/// What one entry's key retains, fixed at staging: its own fields and the
/// entry arena, which by Rule O also holds every value tree.
fn charged_key_bytes(key: &CoefficientGroupKey) -> usize {
    core::mem::size_of::<CoefficientGroupKey>()
        .saturating_add(ENTRY_OVERHEAD_BYTES)
        // ponytail: the shared context is charged in full to every entry.
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(core::mem::size_of::<GroupContext>())
        .saturating_add(key.context.rule.charged_retained_bytes())
        .saturating_add(key.context.operation.charged_retained_bytes())
        .saturating_add(key.group_key.codomain_tree().arena_retained_bytes())
        .saturating_add(ARC_CONTROL_BYTES)
        .saturating_add(
            key.src_keys
                .len()
                .saturating_mul(core::mem::size_of::<FusionTreePairKey>()),
        )
}

/// What one transform entry's value retains.
fn charged_entry_bytes<T>(specs: &[TreeTransformGroupBlockSpec<T>]) -> usize {
    core::mem::size_of::<CoefficientGroupEntry<T>>()
        .saturating_add(
            specs
                .len()
                .saturating_mul(core::mem::size_of::<TreeTransformGroupBlockSpec<T>>()),
        )
        .saturating_add(charged_spec_bytes(specs))
}

/// What one trace entry's value retains: its `Arc`, the inline value and
/// its three exact-length slices. The destinations' trees live in the entry
/// arena, which the key charges.
fn charged_trace_bytes<T>(columns: &TraceTerms<T>) -> usize {
    ARC_CONTROL_BYTES
        .saturating_add(core::mem::size_of::<TraceTerms<T>>())
        .saturating_add(
            columns
                .destinations()
                .len()
                .saturating_mul(core::mem::size_of::<FusionTreePairKey>()),
        )
        .saturating_add((columns.source_count() + 1).saturating_mul(core::mem::size_of::<usize>()))
        .saturating_add(
            columns
                .entry_count()
                .saturating_mul(core::mem::size_of::<(usize, T)>()),
        )
}

/// Heap bytes of `specs` (excluding their inline structs and the trees in
/// the entry arena), coefficients included once.
fn charged_spec_bytes<T>(specs: &[TreeTransformGroupBlockSpec<T>]) -> usize {
    let key_bytes = core::mem::size_of::<FusionTreePairKey>();
    let mut bytes = 0usize;
    let mut previous_axes = None;
    // A `Multi` spec holds its keys and coefficients in three shared heap
    // slices (dst, src, coefficients); a `Single` spec holds them inline,
    // where the caller's `size_of` charge already counts them.
    for spec in specs {
        if spec.has_shared_slices() {
            let coefficients = spec.recoupling_coefficients_dst_src().len();
            bytes = bytes
                .saturating_add(3 * ARC_CONTROL_BYTES)
                .saturating_add(
                    (spec.dst_keys().len().saturating_add(spec.src_keys().len()))
                        .saturating_mul(key_bytes),
                )
                .saturating_add(coefficients.saturating_mul(core::mem::size_of::<T>()));
        }
        if let Some(axes) = spec.source_axes() {
            // ponytail: one build shares one axes `Arc` across its specs, so
            // only a change of pointer is charged; interleaved sharing would
            // overcount, never undercount.
            if previous_axes.replace(axes.as_ptr()) != Some(axes.as_ptr()) {
                bytes = bytes
                    .saturating_add(ARC_CONTROL_BYTES)
                    .saturating_add(axes.len().saturating_mul(core::mem::size_of::<usize>()));
            }
        }
    }
    bytes
}
