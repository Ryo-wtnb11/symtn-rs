//! The process-global completed-transformer cache, the composed-coefficient
//! cache ([`coefficients`]) a completed-transformer miss consults per source
//! group, and the multiplicity-free resolution paths.
//!
//! Reference: TensorKit `cfaa073e` `treetransformers.jl:treetransposer` /
//! `treebraider` (`@cached`, `:156-180`) memoize one completed transformer
//! per `(Vdst, Vsrc, p[, levels])` and `treetransformertype` (`:132-139`),
//! registered in `GLOBAL_CACHES` (`caches.jl:1-11, :160-165`). QSpace
//! `d2d3d7da` has no counterpart: `QSpace::Permute` (`QSpace.hh:2837-2887`)
//! permutes tensor-locally, and its global stores (`CStore gCS`, `RStore
//! gRS`, `clebsch.hh:3718, :4615`) are symbol-layer.
//!
//! Rust deviations, each deliberate:
//! - the key holds block-content ids, never reused, where TensorKit keys on
//!   HomSpace values; so only *canonical* contents (resident in the
//!   complete-HomSpace cache) are published: per-call views, staged
//!   candidates and expert layouts would get a fresh id on every call and
//!   only evict live entries ([`publishable`]);
//! - the value is a structure-free replay core, so an entry charges no
//!   structure content and a hit rebinds a by-value handle to the caller's
//!   structures without allocating;
//! - entries are byte-weighted; values are erased (`dyn Any`, with the
//!   coefficient `TypeId` in the key) in one cache for every coefficient type,
//!   which needs `Send + Sync + 'static` coefficients;
//! - no single-flight: a miss builds with no cache lock or placeholder held,
//!   then publishes transactionally. A build may run Rayon regions, and a
//!   thread waiting on a placeholder inside one could steal a job that wants
//!   the same key and park on its own placeholder. Concurrent cold misses of
//!   one key therefore build up to once each; the first resident value wins;
//! - a reset epoch: a build that straddles `symtn::cache::clear` is returned
//!   unpublished;
//! - checked Generic transforms publish only after their destination commit.

use core::ops::{Add, Mul};
use std::any::{Any, TypeId};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};

use num_traits::Zero;
use rustc_hash::FxHashMap;
use symtn_core::{
    BlockStructure, ErasedStructureCacheControl, FusionTreePairKey, FusionTreePairOrientation,
    HomSpaceId, MultiplicityFreeRigidSymbols, RuleIdentity, StructureCache, StructureCacheInfo,
    StructureCacheKind, UniqueTreePairScratch, WeakHomSpaceId,
};

#[cfg(test)]
use symtn_core::LocallyValidatedFusionTreeBlockStructure;

use crate::{OperationError, TreeTransformStructure};
use symtn_operations::{TreeTransformOperationKind, TreeTransformReplay};

use super::operation::TreeTransformOperation;
use super::plan::{
    build_multiplicity_free_tree_pair_plan_after_capability_with_threads,
    build_oriented_tree_pair_transform_group_plan_capability_validated,
    compile_unique_oriented_tree_pair_structure, compile_unique_tree_pair_structure,
    validate_multiplicity_free_tree_transform_capability,
    validate_tree_pair_namespace_before_cache,
};

mod coefficients;
#[cfg(test)]
mod owner_tests;
pub(crate) use coefficients::{
    CheckedPendingCoefficients, CoefficientGroupReuse, GroupSlot, PendingCoefficientGroups,
    SourceGroup, TraceTermReuse,
};

fn oriented_source_projection<'a>(
    logical_keys: &'a [FusionTreePairKey],
    storage_indices: &[usize],
    storage_src_structure: &BlockStructure,
) -> Result<FxHashMap<&'a FusionTreePairKey, usize>, OperationError> {
    let mut projection =
        FxHashMap::with_capacity_and_hasher(logical_keys.len(), rustc_hash::FxBuildHasher);
    // Why not track storage-index uniqueness here: FusionOperandLayout
    // already proves this projection is a bijection onto parent blocks.
    for (position, (key, &storage_index)) in logical_keys.iter().zip(storage_indices).enumerate() {
        if storage_index >= storage_src_structure.block_count() {
            return Err(OperationError::BlockIndexOutOfBounds {
                tensor: "oriented src",
                index: storage_index,
                count: storage_src_structure.block_count(),
            });
        }
        if projection.insert(key, storage_index).is_some() {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index: position,
            });
        }
    }
    Ok(projection)
}

#[allow(clippy::too_many_arguments)]
fn compile_oriented_tree_pair_structure<R, FAxis>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: &Arc<BlockStructure>,
    logical_keys: &[FusionTreePairKey],
    storage_src_structure: &Arc<BlockStructure>,
    orientation: FusionTreePairOrientation,
    basis_order: OrientedBasisOrder,
    logical_rank: usize,
    projection: &FxHashMap<&FusionTreePairKey, usize>,
    logical_to_storage_axis: FAxis,
    threads: usize,
    epoch: usize,
    unique_scratch: &mut UniqueTreePairScratch,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    FAxis: Fn(usize) -> Result<usize, OperationError>,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::OrientedTreePairCompile);
    if rule.fusion_style() == symtn_core::FusionStyleKind::Unique {
        return compile_unique_oriented_tree_pair_structure(
            rule,
            operation,
            dst_structure,
            logical_keys,
            storage_src_structure,
            orientation,
            basis_order,
            logical_rank,
            projection,
            logical_to_storage_axis,
            unique_scratch,
        );
    }
    // Why the logical keys key a group: the orientation maps each logical
    // tree pair to exactly one storage tree pair, the one the builder reads.
    let reuse = multiplicity_free_reuse(rule, TreeTransformScope::TreePair, operation, orientation);
    let plan = build_oriented_tree_pair_transform_group_plan_capability_validated(
        rule,
        operation.clone(),
        logical_keys,
        storage_src_structure,
        orientation,
        basis_order,
        logical_rank,
        projection,
        threads,
        Some(&reuse),
    )?;
    reuse.into_pending().publish(epoch);
    let source_index = |key: &FusionTreePairKey| {
        projection
            .get(key)
            .copied()
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(symtn_core::BlockKey::FusionTree(key.clone())),
            })
    };
    plan.compile_shared_structures_with_source_projection(
        Arc::clone(dst_structure),
        Arc::clone(storage_src_structure),
        logical_rank,
        source_index,
        logical_to_storage_axis,
        orientation == FusionTreePairOrientation::Adjoint,
    )
}

/// The admission and builder family of a completed transformer; mirrors the
/// complete-HomSpace cache's mode.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TransformerMode {
    MultiplicityFree,
    CheckedGeneric,
}

/// The plan family: a tree-pair recoupling or a trace's lowered terms.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum TreeTransformScope {
    TreePair,
    /// A trace's lowered terms (cache 4 only; cache 3 never keys it). The
    /// open codomain rank is keyed because the `(p…, q…)` operation alone
    /// does not fix where the lowering splits: `[a, b] ← [c, d]` permutes a
    /// 0-, 1- and 2-pair trace alike.
    TraceTerms {
        open_codomain_rank: usize,
    },
}

/// The source block order an oriented (lazy-adjoint) transform reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum OrientedBasisOrder {
    Canonical,
    Storage,
}

/// A borrowed tree-transform operation: kind, permutations and levels, as
/// [`TreeTransformOperation`]'s accessors return them. A typed caller forms
/// it over axes it already holds, so an exact-layout hit allocates nothing.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TreeTransformOperationView<'a> {
    kind: TreeTransformOperationKind,
    codomain_permutation: &'a [usize],
    domain_permutation: &'a [usize],
    codomain_levels: &'a [usize],
    domain_levels: &'a [usize],
}

impl<'a> TreeTransformOperationView<'a> {
    pub fn new(
        kind: TreeTransformOperationKind,
        codomain_permutation: &'a [usize],
        domain_permutation: &'a [usize],
        codomain_levels: &'a [usize],
        domain_levels: &'a [usize],
    ) -> Self {
        Self {
            kind,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        }
    }

    pub fn of(operation: &'a TreeTransformOperation) -> Self {
        Self::new(
            operation.kind(),
            operation.codomain_permutation(),
            operation.domain_permutation(),
            operation.codomain_levels(),
            operation.domain_levels(),
        )
    }
}

/// Key of one completed transformer: every determinant of its replay core.
///
/// Excluded, each with its reason: the recoupling thread count (the output
/// is thread-independent); backend, placement, alpha/beta and init
/// (replay-time only); sector identity beyond the structures (the content ids
/// fix it); oriented logical keys, projection and axis map (derived from the
/// parent and `basis_order`).
#[derive(Clone, Debug)]
pub(crate) struct CompletedTransformerKey {
    rule: RuleIdentity,
    mode: TransformerMode,
    coefficient: TypeId,
    operation: TreeTransformOperation,
    orientation: FusionTreePairOrientation,
    basis_order: OrientedBasisOrder,
    storage_conjugate: bool,
    logical_source: Option<usize>,
    dst: usize,
    src: usize,
}

/// What equality and hashing read, borrowed. The owned key and the borrowed
/// exact-layout probe both reduce to it, so their hashes agree by
/// construction.
#[derive(Eq, Hash, PartialEq)]
struct KeyParts<'a> {
    rule: &'a RuleIdentity,
    mode: TransformerMode,
    coefficient: TypeId,
    operation: TreeTransformOperationView<'a>,
    orientation: FusionTreePairOrientation,
    basis_order: OrientedBasisOrder,
    storage_conjugate: bool,
    logical_source: Option<usize>,
    dst: usize,
    src: usize,
}

impl CompletedTransformerKey {
    #[allow(clippy::too_many_arguments)] // Every determinant belongs in the key.
    pub(crate) fn new<T: 'static>(
        rule: RuleIdentity,
        mode: TransformerMode,
        operation: &TreeTransformOperation,
        orientation: FusionTreePairOrientation,
        basis_order: OrientedBasisOrder,
        storage_conjugate: bool,
        logical_source: Option<&BlockStructure>,
        dst: &BlockStructure,
        src: &BlockStructure,
    ) -> Self {
        Self {
            rule,
            mode,
            coefficient: TypeId::of::<T>(),
            operation: operation.clone(),
            orientation,
            basis_order,
            storage_conjugate,
            logical_source: logical_source.map(BlockStructure::content_id),
            dst: dst.content_id(),
            src: src.content_id(),
        }
    }

    /// An ordinary multiplicity-free tree-pair key.
    pub(crate) fn tree_pair<T: 'static>(
        rule: RuleIdentity,
        operation: &TreeTransformOperation,
        dst: &BlockStructure,
        src: &BlockStructure,
        storage_conjugate: bool,
    ) -> Self {
        Self::new::<T>(
            rule,
            TransformerMode::MultiplicityFree,
            operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            storage_conjugate,
            None,
            dst,
            src,
        )
    }

    fn parts(&self) -> KeyParts<'_> {
        KeyParts {
            rule: &self.rule,
            mode: self.mode,
            coefficient: self.coefficient,
            operation: TreeTransformOperationView::of(&self.operation),
            orientation: self.orientation,
            basis_order: self.basis_order,
            storage_conjugate: self.storage_conjugate,
            logical_source: self.logical_source,
            dst: self.dst,
            src: self.src,
        }
    }

    fn charged_retained_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            .saturating_add(self.rule.charged_retained_bytes())
            .saturating_add(self.operation.charged_retained_bytes())
    }
}

impl PartialEq for CompletedTransformerKey {
    fn eq(&self, other: &Self) -> bool {
        self.parts() == other.parts()
    }
}

impl Eq for CompletedTransformerKey {}

impl Hash for CompletedTransformerKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parts().hash(state);
    }
}

/// Borrowed probe of an ordinary multiplicity-free tree-pair key.
struct TreePairKeyRef<'a> {
    rule: &'a RuleIdentity,
    coefficient: TypeId,
    operation: TreeTransformOperationView<'a>,
    dst: usize,
    src: usize,
}

impl TreePairKeyRef<'_> {
    fn parts(&self) -> KeyParts<'_> {
        KeyParts {
            rule: self.rule,
            mode: TransformerMode::MultiplicityFree,
            coefficient: self.coefficient,
            operation: self.operation,
            orientation: FusionTreePairOrientation::Direct,
            basis_order: OrientedBasisOrder::Canonical,
            storage_conjugate: false,
            logical_source: None,
            dst: self.dst,
            src: self.src,
        }
    }
}

impl Hash for TreePairKeyRef<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.parts().hash(state);
    }
}

impl symtn_core::StructureCacheEquivalent<CompletedTransformerKey> for TreePairKeyRef<'_> {
    fn equivalent(&self, key: &CompletedTransformerKey) -> bool {
        self.parts() == key.parts()
    }
}

/// One identity of a typed source or destination: its canonical HomSpace,
/// weakly, and `[content id, nout, nin]`.
#[derive(Debug)]
struct LayoutIdentity {
    homspace: WeakHomSpaceId,
    layout: [usize; 3],
}

impl LayoutIdentity {
    fn matches(&self, homspace: &HomSpaceId, layout: [usize; 3]) -> bool {
        self.layout == layout && self.homspace.matches(homspace)
    }
}

/// The typed exact-layout admission memo: the typed Host `*_into` path proved
/// once that this destination is the operation's result space of this
/// source, so a repeated call with the same two spaces skips rebuilding the
/// owned operation and the result space. Set at most once; a second HomSpace
/// pair mapping to the same contents takes the full check every call.
#[derive(Debug)]
struct ExactLayoutProof {
    source: LayoutIdentity,
    destination: LayoutIdentity,
}

/// The cached value: a replay core and its exact-layout memo. The memo's
/// bytes are part of the entry from admission on, so a charge never grows.
pub(crate) struct CompletedTransformerEntry<T> {
    core: Arc<TreeTransformReplay<T>>,
    exact_layout: OnceLock<ExactLayoutProof>,
}

impl<T> CompletedTransformerEntry<T> {
    fn bind(
        &self,
        dst: &Arc<BlockStructure>,
        src: &Arc<BlockStructure>,
    ) -> TreeTransformStructure<T> {
        TreeTransformStructure::from_replay_core(
            Arc::clone(&self.core),
            Arc::clone(dst),
            Arc::clone(src),
        )
    }
}

type ErasedEntry = dyn Any + Send + Sync;

/// quick_cache node and the entry's `Arc` control, per entry.
const ENTRY_OVERHEAD_BYTES: usize = 10 * core::mem::size_of::<usize>();

struct CompletedTransformerControl;

impl ErasedStructureCacheControl for CompletedTransformerControl {
    fn info(&self) -> StructureCacheInfo {
        completed_transformers().info()
    }

    fn clear(&self) {
        ErasedStructureCacheControl::clear(completed_transformers())
    }

    fn set_byte_budget(&self, bytes: u64) {
        ErasedStructureCacheControl::set_byte_budget(completed_transformers(), bytes)
    }
}

/// The process-global completed-transformer cache, registered with
/// `symtn-core`'s structure-cache registry on first use.
fn completed_transformers() -> &'static StructureCache<CompletedTransformerKey, ErasedEntry> {
    static CACHE: OnceLock<StructureCache<CompletedTransformerKey, ErasedEntry>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Refused only if another crate claimed the slot first. Then
        // `symtn::cache::clear` cannot reach this cache, so it admits
        // nothing: every transform builds uncached, with identical results.
        let budget = symtn_core::register_structure_cache(
            StructureCacheKind::CompletedTreeTransformer,
            &CompletedTransformerControl,
        )
        .unwrap_or(0);
        StructureCache::new(StructureCacheKind::CompletedTreeTransformer, budget)
    })
}

/// Whether a key over these structures may be published: every keyed
/// content is canonical. Any other key is lookup-only.
pub(crate) fn publishable<'a>(structures: impl IntoIterator<Item = &'a BlockStructure>) -> bool {
    structures.into_iter().all(BlockStructure::is_canonical)
}

fn lookup<T, Q>(key: &Q) -> Option<Arc<CompletedTransformerEntry<T>>>
where
    T: 'static + Send + Sync,
    Q: Hash + symtn_core::StructureCacheEquivalent<CompletedTransformerKey> + ?Sized,
{
    let entry = completed_transformers().get(key)?;
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::CompletedHit);
    // The key's coefficient `TypeId` fixes the value type.
    Some(
        entry
            .downcast::<CompletedTransformerEntry<T>>()
            .unwrap_or_else(|_| unreachable!("the key's TypeId fixes the entry type")),
    )
}

/// Offers `core` for `key`, built since `epoch`; returns the resident core
/// (the earlier winner after a race) or `core` itself when it was refused.
fn publish<T>(
    key: &CompletedTransformerKey,
    core: Arc<TreeTransformReplay<T>>,
    epoch: usize,
) -> Arc<TreeTransformReplay<T>>
where
    T: 'static + Send + Sync,
{
    #[cfg(test)]
    {
        crate::probe::hit(crate::probe::Event::CompletedPublication);
        if let Some(hook) = crate::probe::take_before_publish_hook() {
            hook();
        }
    }
    let bytes = key
        .charged_retained_bytes()
        .saturating_add(core::mem::size_of::<CompletedTransformerEntry<T>>())
        .saturating_add(ENTRY_OVERHEAD_BYTES)
        .saturating_add(core.charged_bytes()) as u64;
    let entry: Arc<ErasedEntry> = Arc::new(CompletedTransformerEntry {
        core: Arc::clone(&core),
        exact_layout: OnceLock::new(),
    });
    let (resident, admitted) = completed_transformers().publish(key, entry, bytes, epoch);

    if !admitted {
        return core;
    }
    resident
        .downcast::<CompletedTransformerEntry<T>>()
        .map(|entry| Arc::clone(&entry.core))
        .unwrap_or_else(|_| unreachable!("the key's TypeId fixes the entry type"))
}

/// The completed transformer for `key`: a hit binds the cached core to
/// `dst`/`src`; a miss runs `build` with no cache lock held and publishes the
/// result when `may_publish` (every keyed content canonical).
///
/// `epoch` is captured before any admission or key formation the caller
/// performed for this request.
pub(crate) fn resolve<T, E>(
    key: CompletedTransformerKey,
    may_publish: bool,
    epoch: usize,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    build: impl FnOnce() -> Result<TreeTransformStructure<T>, E>,
) -> Result<TreeTransformStructure<T>, E>
where
    T: 'static + Send + Sync,
{
    if let Some(entry) = lookup::<T, _>(&key) {
        return Ok(entry.bind(dst, src));
    }
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::CompletedBuild);
    let built = build()?;
    if !may_publish {
        return Ok(built);
    }
    let core = publish(&key, Arc::clone(built.replay_core()), epoch);
    Ok(TreeTransformStructure::from_replay_core(
        core,
        Arc::clone(built.dst_structure()),
        Arc::clone(built.src_structure()),
    ))
}

/// The lookup half of [`resolve`], for transactions that publish only after
/// a later commit ([`publish_committed`]). A miss is the caller's build.
pub(crate) fn lookup_bound<T>(
    key: &CompletedTransformerKey,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
) -> Option<TreeTransformStructure<T>>
where
    T: 'static + Send + Sync,
{
    let found = lookup::<T, _>(key).map(|entry| entry.bind(dst, src));
    #[cfg(test)]
    if found.is_none() {
        crate::probe::hit(crate::probe::Event::CompletedBuild);
    }
    found
}

/// The publication half of [`resolve`], after the caller's commit.
pub(crate) fn publish_committed<T>(
    key: &CompletedTransformerKey,
    core: &Arc<TreeTransformReplay<T>>,
    epoch: usize,
) where
    T: 'static + Send + Sync,
{
    let _ = publish(key, Arc::clone(core), epoch);
}

/// The completed tree-pair transformer of a typed Host `*_into` request whose
/// destination was already proved to be the operation's result space of the
/// source: one borrowed-key shard read, no allocation.
#[doc(hidden)]
pub fn exact_layout_tree_pair_hit<T>(
    rule: &RuleIdentity,
    operation: TreeTransformOperationView<'_>,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    source: (&HomSpaceId, [usize; 3]),
    destination: (&HomSpaceId, [usize; 3]),
) -> Option<TreeTransformStructure<T>>
where
    T: 'static + Send + Sync,
{
    let key = TreePairKeyRef {
        rule,
        coefficient: TypeId::of::<T>(),
        operation,
        dst: dst.content_id(),
        src: src.content_id(),
    };
    let entry = lookup::<T, _>(&key)?;
    let proof = entry.exact_layout.get()?;
    (proof.source.matches(source.0, source.1)
        && proof.destination.matches(destination.0, destination.1))
    .then(|| entry.bind(dst, src))
}

/// Records the exact-layout proof on the resident tree-pair entry, if any.
/// A missing (lookup-only or evicted) entry retains no proof.
#[doc(hidden)]
pub fn admit_exact_tree_pair_layout<T>(
    rule: &RuleIdentity,
    operation: &TreeTransformOperation,
    dst: &BlockStructure,
    src: &BlockStructure,
    source: (&HomSpaceId, [usize; 3]),
    destination: (&HomSpaceId, [usize; 3]),
) -> bool
where
    T: 'static + Send + Sync,
{
    let key = TreePairKeyRef {
        rule,
        coefficient: TypeId::of::<T>(),
        operation: TreeTransformOperationView::of(operation),
        dst: dst.content_id(),
        src: src.content_id(),
    };
    let Some(entry) = lookup::<T, _>(&key) else {
        return false;
    };
    let _ = entry.exact_layout.set(ExactLayoutProof {
        source: LayoutIdentity {
            homspace: source.0.downgrade(),
            layout: source.1,
        },
        destination: LayoutIdentity {
            homspace: destination.0.downgrade(),
            layout: destination.1,
        },
    });
    true
}

/// The composed-coefficient reuse of one Simple multiplicity-free plan
/// build. Unique fusion never builds one: it is `NoCache`, as in TensorKit.
fn multiplicity_free_reuse<R, T>(
    rule: &R,
    scope: TreeTransformScope,
    operation: &TreeTransformOperation,
    orientation: FusionTreePairOrientation,
) -> CoefficientGroupReuse<T>
where
    R: symtn_core::FusionRule,
    T: 'static + Send + Sync,
{
    CoefficientGroupReuse::new(
        rule.rule_identity(),
        TransformerMode::MultiplicityFree,
        scope,
        operation,
        orientation,
    )
}

/// What a context's multiplicity-free resolution needs besides the global
/// owners (caches 3 and 4): the recoupling worker count, and the Unique
/// compile's tree-pair scratch, reused across cache-3 misses.
///
/// Why a plain field rather than a `RefCell` behind `&self`: the context
/// already plans through `&mut`, and a cell would make the planning `!Sync`.
pub(crate) struct TreeTransformPlanning {
    recoupling_threads: NonZeroUsize,
    unique_scratch: UniqueTreePairScratch,
}

impl Default for TreeTransformPlanning {
    fn default() -> Self {
        Self {
            recoupling_threads: NonZeroUsize::MIN,
            unique_scratch: UniqueTreePairScratch::default(),
        }
    }
}

impl std::fmt::Debug for TreeTransformPlanning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeTransformPlanning")
            .field("recoupling_threads", &self.recoupling_threads)
            .finish_non_exhaustive()
    }
}

impl TreeTransformPlanning {
    pub(crate) fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
        self.recoupling_threads = threads;
    }
}

impl TreeTransformPlanning {
    /// The ordinary multiplicity-free tree-pair transformer.
    ///
    /// Fusion-tree block keys in `dst` and `src` follow
    /// [`symtn_core::FusionTreeKey::validate_for_rule`]'s provider-domain
    /// precondition.
    pub(crate) fn resolve_tree_pair<R, T>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
    {
        let epoch = symtn_core::core_reset_epoch();
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, src_structure)?;
        let key = CompletedTransformerKey::tree_pair::<T>(
            rule.rule_identity(),
            operation,
            dst_structure,
            src_structure,
            storage_conjugate,
        );
        let may_publish = publishable([dst_structure.as_ref(), src_structure.as_ref()]);
        let threads = self.recoupling_threads.get();
        resolve(
            key,
            may_publish,
            epoch,
            dst_structure,
            src_structure,
            || {
                if rule.fusion_style() == symtn_core::FusionStyleKind::Unique {
                    return compile_unique_tree_pair_structure(
                        rule,
                        operation,
                        dst_structure,
                        src_structure,
                        storage_conjugate,
                        &mut self.unique_scratch,
                    );
                }
                let reuse = multiplicity_free_reuse(
                    rule,
                    TreeTransformScope::TreePair,
                    operation,
                    FusionTreePairOrientation::Direct,
                );
                // The preflight and destination proof run before any group
                // lookup.
                let plan = build_multiplicity_free_tree_pair_plan_after_capability_with_threads(
                    rule,
                    operation,
                    dst_structure,
                    src_structure,
                    threads,
                    Some(&reuse),
                )?;
                reuse.into_pending().publish(epoch);
                plan.compile_shared_structures_with_storage_conjugation(
                    Arc::clone(dst_structure),
                    Arc::clone(src_structure),
                    storage_conjugate,
                )
            },
        )
    }

    /// A lazy adjoint addressed through its parent's storage blocks. The key
    /// is the parent's storage content with the orientation and basis order,
    /// never a per-call adjoint view, so it is publishable whenever the
    /// parent is canonical.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve_tree_pair_oriented<'p, R, T, FIndices, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_keys: &[FusionTreePairKey],
        storage_indices: FIndices,
        storage_src_structure: &Arc<BlockStructure>,
        orientation: FusionTreePairOrientation,
        basis_order: OrientedBasisOrder,
        logical_rank: usize,
        logical_to_storage_axis: FAxis,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        FIndices: FnOnce() -> Result<&'p [usize], OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        self.resolve_tree_pair_oriented_with(
            rule,
            operation,
            dst_structure,
            || {
                Ok((
                    std::borrow::Cow::Borrowed(logical_keys),
                    std::borrow::Cow::Borrowed(storage_indices()?),
                ))
            },
            storage_src_structure,
            orientation,
            basis_order,
            logical_rank,
            logical_to_storage_axis,
        )
    }

    /// [`Self::resolve_tree_pair_oriented`] whose logical keys, in the keyed
    /// basis order, and their parent storage indices are derived only on a
    /// miss, so a warm call reads neither.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn resolve_tree_pair_oriented_with<'k, 'p, R, T, FProjection, FAxis>(
        &mut self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        source_projection: FProjection,
        storage_src_structure: &Arc<BlockStructure>,
        orientation: FusionTreePairOrientation,
        basis_order: OrientedBasisOrder,
        logical_rank: usize,
        logical_to_storage_axis: FAxis,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        FProjection: FnOnce() -> Result<
            (
                std::borrow::Cow<'k, [FusionTreePairKey]>,
                std::borrow::Cow<'p, [usize]>,
            ),
            OperationError,
        >,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let epoch = symtn_core::core_reset_epoch();
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        validate_tree_pair_namespace_before_cache(operation, storage_src_structure)?;
        let storage_conjugate = orientation == FusionTreePairOrientation::Adjoint;
        let threads = self.recoupling_threads.get();
        // Why lazy: only a miss reads the storage map, so a warm call derives
        // no per-block projection.
        let build = || {
            let (logical_keys, storage_indices) = source_projection()?;
            if logical_keys.len() != storage_indices.len() {
                return Err(OperationError::StructureMismatch {
                    tensor: "oriented source projection",
                });
            }
            let projection =
                oriented_source_projection(&logical_keys, &storage_indices, storage_src_structure)?;
            compile_oriented_tree_pair_structure(
                rule,
                operation,
                dst_structure,
                &logical_keys,
                storage_src_structure,
                orientation,
                basis_order,
                logical_rank,
                &projection,
                logical_to_storage_axis,
                threads,
                epoch,
                &mut self.unique_scratch,
            )
        };
        // Why only the adjoint orientation: a direct oriented key would equal
        // the ordinary tree-pair key while its plan follows the caller's
        // logical key order, and no production caller compiles it.
        if !storage_conjugate {
            return build();
        }
        // Why not key the logical keys, projection, rank, or axis map: the two
        // operand preparations derive them from the parent structure and the
        // keyed basis order: canonical enumeration for Dynamic, parent block
        // order for static views. A transform's lazy-adjoint source keys the
        // same entry: its logical space is the multiplicity-free adjoint,
        // derived canonically from the parent. The axis map reads the parent
        // split; a blockless parent compiles no spec.
        let key = CompletedTransformerKey::new::<T>(
            rule.rule_identity(),
            TransformerMode::MultiplicityFree,
            operation,
            orientation,
            basis_order,
            storage_conjugate,
            None,
            dst_structure,
            storage_src_structure,
        );
        let may_publish = publishable([dst_structure.as_ref(), storage_src_structure.as_ref()]);
        resolve(
            key,
            may_publish,
            epoch,
            dst_structure,
            storage_src_structure,
            build,
        )
    }
}

#[cfg(test)]
impl TreeTransformPlanning {
    /// Eager prelowered compile: a test oracle for storage-mapped sources,
    /// never cached.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_compile_tree_pair_prelowered<R, T, FBlock, FAxis>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        logical_src_structure: &Arc<BlockStructure>,
        storage_src_structure: &Arc<BlockStructure>,
        storage_conjugate: bool,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = T>,
        T: 'static + Copy + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        let source_proof = super::plan::validate_multiplicity_free_tree_pair_preflight(
            rule,
            operation,
            logical_src_structure,
        )?;
        let logical_source_id = logical_src_structure.content_id();
        let storage_source_id = storage_src_structure.content_id();
        let destination_id = dst_structure.content_id();
        if storage_source_id != logical_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, storage_src_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        if destination_id != logical_source_id && destination_id != storage_source_id {
            LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
                .map_err(OperationError::from_core_preserving_context)?;
        }
        let plan = super::plan::build_tree_pair_transform_group_plan_validated_with_threads(
            &source_proof,
            operation.clone(),
            self.recoupling_threads.get(),
        )?;
        plan.compile_shared_structures_with_storage_mapping(
            Arc::clone(dst_structure),
            logical_src_structure,
            Arc::clone(storage_src_structure),
            logical_to_storage_block,
            logical_to_storage_axis,
            storage_conjugate,
        )
    }
}
