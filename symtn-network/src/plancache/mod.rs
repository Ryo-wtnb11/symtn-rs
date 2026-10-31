//! Topology-keyed contraction-plan cache behind [`Network::contract`].
//!
//! The cache key is the network *topology*: per-operand label lists, conj
//! flags, codomain ranks and written `;` splits, plus the output labels and
//! the [`Optimizer`] choice. Leg dimensions are deliberately NOT part of the
//! key: a pairwise contraction order is correct for any dimensions, and
//! truncation drifts bond dimensions every sweep — an exact-dims key would
//! miss every iteration. Each entry stores the dimensions it was planned
//! under; the [`ReplanPolicy`] decides whether a dimension change forces a
//! re-plan. The default ([`ReplanPolicy::BakeOnce`]) finds the order once at
//! real dims and reuses it for any later dims — the standard "search once,
//! reuse the path regardless of rank" design (cotengra's reusable
//! `ContractionTree`, `@tensoropt`'s compile-time bake) — so the
//! (χ-dependent) order search is paid at most once per topology, not per χ.
//! Eviction is LRU; ordinary pairwise contraction routes in `symtn-tensors`
//! are resolved eagerly instead.
//!
//! Storage is per-[`Runtime`]: the configuration (set it with
//! [`configure_plan_cache`]) and the cache state both sit in the runtime's
//! opaque type-keyed extension slot, under this crate's own type. The
//! operands' runtime is resolved per call, so different runtimes never share
//! plans or counters.

use std::any::{Any, TypeId};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use symtn::typed::Engine;

use lru::LruCache;
use symtn::expert::TensorStorage;
use symtn::sector::TypedSectorAdmission;
use symtn::typed::TensorMap;
use symtn::typed::{Error, Runtime, TensorScalar};

mod config;

#[cfg(feature = "cotengra-python")]
pub use config::{
    CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod, CotengraSlicingConfig,
};
pub use config::{
    Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy, DEFAULT_PLAN_CACHE_CAPACITY,
    DEFAULT_REPLAN_DRIFT_FACTOR, DEFAULT_WORKSPACE_BUDGET_BYTES,
};

use crate::labels::TemporaryLabel;
use crate::network::{
    FacadeError, Network, NetworkExecutionWorkspace, NetworkModeDispatch, PlannedNetwork,
};
use crate::optimizer::GreedyDenseOptimizer;
use crate::plan::push_label;
use symtn::typed::__network::{self, NetworkPayloadStorage};

#[derive(Clone, PartialEq, Eq, Hash)]
struct OperandTopology {
    labels: Vec<TemporaryLabel>,
    conj: bool,
    /// Codomain rank of the operand tensor: it fixes the conj label
    /// rotation, so it is structural even though it is not a label.
    codomain_rank: usize,
    written_split: usize,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct NetworkTopology {
    operands: Vec<OperandTopology>,
    output: Vec<TemporaryLabel>,
    output_codomain_rank: Option<usize>,
    optimizer: Optimizer,
}

struct CacheEntry {
    planned: Arc<PlannedNetwork>,
    workspaces: Arc<WorkspacePools>,
    /// Flat leg dims per operand at plan time (written leg order).
    dims_snapshot: Vec<Vec<usize>>,
}

impl Drop for CacheEntry {
    fn drop(&mut self) {
        // Why: an executing CachedPlan can outlive map eviction/clear. Marking
        // its pools inactive makes those late lease returns quarantine their
        // buffers instead of repopulating an orphaned cache entry.
        self.workspaces.deactivate_all();
    }
}

#[derive(Default)]
struct WorkspacePoolCounters {
    created: AtomicU64,
    reused: AtomicU64,
    slot_grows: AtomicU64,
    idle: AtomicU64,
}

#[derive(Default)]
struct WorkspaceBudgetState {
    limit: usize,
    retained: usize,
    peak: usize,
    admissions: u64,
    rejections: u64,
    evictions: u64,
}

struct WorkspaceBudget {
    state: Mutex<WorkspaceBudgetState>,
}

// The byte budget covers dimension-dependent storage retained for reuse. Plan
// metadata, the TypeId registry, pool mutexes, and at most two idle-workspace
// shells per plan remain under the existing plan/idle count bounds; they are
// deliberately not mixed into this numerical-storage budget.

impl WorkspaceBudget {
    fn new(limit: usize) -> Self {
        Self {
            state: Mutex::new(WorkspaceBudgetState {
                limit,
                ..WorkspaceBudgetState::default()
            }),
        }
    }

    fn reserve(&self, bytes: usize) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        let Some(next) = state.retained.checked_add(bytes) else {
            state.rejections = state.rejections.saturating_add(1);
            return false;
        };
        if state.limit == 0 || next > state.limit {
            state.rejections = state.rejections.saturating_add(1);
            return false;
        }
        state.retained = next;
        state.peak = state.peak.max(next);
        state.admissions = state.admissions.saturating_add(1);
        true
    }

    fn release(&self, bytes: usize, evicted: bool) {
        let mut state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        state.retained = state
            .retained
            .checked_sub(bytes)
            .expect("workspace budget release exceeds retained bytes");
        if evicted {
            state.evictions = state.evictions.saturating_add(1);
        }
    }

    fn set_limit(&self, limit: usize) {
        self.state
            .lock()
            .expect("network workspace budget poisoned")
            .limit = limit;
    }

    fn reset_statistics(&self) {
        let mut state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        debug_assert_eq!(state.retained, 0);
        state.retained = 0;
        state.peak = 0;
        state.admissions = 0;
        state.rejections = 0;
        state.evictions = 0;
    }

    fn snapshot(&self) -> (usize, usize, u64, u64, u64) {
        let state = self
            .state
            .lock()
            .expect("network workspace budget poisoned");
        (
            state.retained,
            state.peak,
            state.admissions,
            state.rejections,
            state.evictions,
        )
    }
}

struct WorkspacePools {
    pools: Mutex<LruCache<TypeId, Arc<dyn ErasedWorkspacePool>>>,
    counters: Arc<WorkspacePoolCounters>,
    budget: Arc<WorkspaceBudget>,
    accepting: AtomicBool,
}

struct WorkspacePool<R, D, S = Vec<D>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    available: Mutex<Vec<IdleWorkspace<R, D, S>>>,
    counters: Arc<WorkspacePoolCounters>,
    budget: Arc<WorkspaceBudget>,
    registered: AtomicBool,
}

struct IdleWorkspace<R, D, S = Vec<D>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    workspace: NetworkExecutionWorkspace<R, D, S>,
    charge: usize,
}

const MAX_IDLE_WORKSPACES_PER_PLAN: usize = 2;
// Empty typed pool shells have no reuse value beyond the plan-wide number of
// buffers that can remain idle, so the same bound caps the TypeId registry.

trait ErasedWorkspacePool: Any + Send + Sync {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
    fn deactivate(&self);
}

struct WorkspaceLease<R, D, S = Vec<D>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    pool: Arc<WorkspacePool<R, D, S>>,
    workspace: Option<NetworkExecutionWorkspace<R, D, S>>,
    recyclable: bool,
}

impl Default for WorkspacePools {
    fn default() -> Self {
        Self::new(Arc::new(WorkspaceBudget::new(
            DEFAULT_WORKSPACE_BUDGET_BYTES,
        )))
    }
}

impl WorkspacePools {
    fn new(budget: Arc<WorkspaceBudget>) -> Self {
        Self {
            pools: Mutex::new(LruCache::new(lru_capacity(MAX_IDLE_WORKSPACES_PER_PLAN))),
            counters: Arc::new(WorkspacePoolCounters::default()),
            budget,
            accepting: AtomicBool::new(true),
        }
    }

    fn unpooled() -> Self {
        Self::new(Arc::new(WorkspaceBudget::new(0)))
    }

    fn deactivate_all(&self) {
        self.accepting.store(false, Ordering::SeqCst);
        self.rotate_all();
    }

    fn rotate_all(&self) {
        let mut pools = self.pools.lock().expect("network pool registry poisoned");
        while let Some((_, pool)) = pools.pop_lru() {
            pool.deactivate();
        }
    }

    /// The pool serving one `(provider, dtype, storage)` triple.
    ///
    /// Storage is part of the key: a Host and a device execution of the same
    /// topology and `(R, D)` are different workspace types, and the erased
    /// registry must never hand one to the other.
    fn pool<R, D, S>(&self) -> Arc<WorkspacePool<R, D, S>>
    where
        R: TypedSectorAdmission + Send + Sync,
        R::Mode: NetworkModeDispatch<R, D, S>,
        D: TensorScalar + Send + Sync + 'static,
        S: NetworkPayloadStorage<D> + Send + Sync,
    {
        if !self.accepting.load(Ordering::SeqCst) {
            return Arc::new(WorkspacePool {
                available: Mutex::new(Vec::new()),
                counters: Arc::clone(&self.counters),
                budget: Arc::clone(&self.budget),
                registered: AtomicBool::new(false),
            });
        }
        let key = TypeId::of::<(R, D, S)>();
        let mut pools = self.pools.lock().expect("network pool registry poisoned");
        if !self.accepting.load(Ordering::SeqCst) {
            return Arc::new(WorkspacePool {
                available: Mutex::new(Vec::new()),
                counters: Arc::clone(&self.counters),
                budget: Arc::clone(&self.budget),
                registered: AtomicBool::new(false),
            });
        }
        if let Some(pool) = pools.get(&key) {
            return Arc::clone(pool)
                .as_any_arc()
                .downcast::<WorkspacePool<R, D, S>>()
                .expect("workspace TypeId mapped to the wrong pool type");
        }
        let pool = Arc::new(WorkspacePool {
            available: Mutex::new(Vec::new()),
            counters: Arc::clone(&self.counters),
            budget: Arc::clone(&self.budget),
            registered: AtomicBool::new(true),
        });
        if let Some((_, evicted)) =
            pools.push(key, Arc::clone(&pool) as Arc<dyn ErasedWorkspacePool>)
        {
            evicted.deactivate();
        }
        pool
    }
}

impl<R, D, S> ErasedWorkspacePool for WorkspacePool<R, D, S>
where
    R: TypedSectorAdmission + Send + Sync,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar + Send + Sync + 'static,
    S: NetworkPayloadStorage<D> + Send + Sync,
{
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn deactivate(&self) {
        self.registered.store(false, Ordering::SeqCst);
        let mut available = self
            .available
            .lock()
            .expect("network workspace pool poisoned");
        let removed = available.len() as u64;
        for idle in available.drain(..) {
            self.budget.release(idle.charge, true);
        }
        self.counters.idle.fetch_sub(removed, Ordering::SeqCst);
    }
}

impl<R, D, S> WorkspacePool<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    fn lease(self: &Arc<Self>) -> WorkspaceLease<R, D, S> {
        let workspace = {
            let mut available = self
                .available
                .lock()
                .expect("network workspace pool poisoned");
            let idle = available.pop();
            if let Some(idle) = idle {
                self.budget.release(idle.charge, false);
                self.counters.idle.fetch_sub(1, Ordering::SeqCst);
                Some(idle.workspace)
            } else {
                None
            }
        };
        let workspace = match workspace {
            Some(workspace) => {
                self.counters.reused.fetch_add(1, Ordering::Relaxed);
                workspace
            }
            None => {
                self.counters.created.fetch_add(1, Ordering::Relaxed);
                NetworkExecutionWorkspace::default()
            }
        };
        WorkspaceLease {
            pool: Arc::clone(self),
            workspace: Some(workspace),
            recyclable: false,
        }
    }
}

impl<R, D, S> WorkspaceLease<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    fn workspace(&mut self) -> &mut NetworkExecutionWorkspace<R, D, S> {
        self.workspace
            .as_mut()
            .expect("workspace lease always owns a workspace")
    }

    fn commit_recycling(&mut self) {
        self.recyclable = true;
    }
}

impl<R, D, S> Drop for WorkspaceLease<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    fn drop(&mut self) {
        if std::thread::panicking() || !self.recyclable {
            self.workspace.take();
            return;
        }
        if let Some(mut workspace) = self.workspace.take() {
            workspace.clear_slots();
            <R::Mode as NetworkModeDispatch<R, D, S>>::park_workspace(&mut workspace);
            let mut available = self
                .pool
                .available
                .lock()
                .expect("network workspace pool poisoned");
            if !self.pool.registered.load(Ordering::SeqCst) {
                return;
            }
            let reserved = self
                .pool
                .counters
                .idle
                .try_update(Ordering::SeqCst, Ordering::SeqCst, |idle| {
                    (idle < MAX_IDLE_WORKSPACES_PER_PLAN as u64).then_some(idle + 1)
                })
                .is_ok();
            if reserved {
                let charge = workspace.retained_idle_bytes();
                if self.pool.budget.reserve(charge) {
                    available.push(IdleWorkspace { workspace, charge });
                } else {
                    self.pool.counters.idle.fetch_sub(1, Ordering::SeqCst);
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct CachedPlan {
    planned: Arc<PlannedNetwork>,
    workspaces: Arc<WorkspacePools>,
}

impl CachedPlan {
    /// Runs the plan for [`Network::contract`](crate::Network::contract),
    /// which decided the operand count and admission before the lookup.
    pub(crate) fn execute<R, D, S>(
        &self,
        engine: &mut dyn Engine,
        tensors: &[&TensorMap<R, D, S>],
    ) -> Result<TensorMap<R, D, S>, FacadeError<R>>
    where
        R: TypedSectorAdmission + Send + Sync,
        R::Mode: NetworkModeDispatch<R, D, S>,
        D: TensorScalar + Send + Sync + 'static,
        S: crate::network::NetworkOperandStorage<R, D>,
    {
        self.execute_leased(tensors, |planned, tensors, workspace| {
            planned.execute_admitted(engine, tensors, workspace)
        })
    }

    /// One lease lifecycle for every placement: take a workspace from this
    /// plan's `(R, D, S)` pool, run `execute`, and recycle only on success so a
    /// failed or panicking step quarantines the buffers it may have touched.
    fn execute_leased<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        execute: impl FnOnce(
            &PlannedNetwork,
            &[&TensorMap<R, D, S>],
            &mut NetworkExecutionWorkspace<R, D, S>,
        ) -> Result<TensorMap<R, D, S>, FacadeError<R>>,
    ) -> Result<TensorMap<R, D, S>, FacadeError<R>>
    where
        R: TypedSectorAdmission + Send + Sync,
        R::Mode: NetworkModeDispatch<R, D, S>,
        D: TensorScalar + Send + Sync + 'static,
        S: NetworkPayloadStorage<D> + Send + Sync,
    {
        let pool = self.workspaces.pool::<R, D, S>();
        let mut lease = pool.lease();
        let previous_capacity = lease.workspace().slot_capacity();
        let result = execute(&self.planned, tensors, lease.workspace());
        if lease.workspace().slot_capacity() > previous_capacity {
            self.workspaces
                .counters
                .slot_grows
                .fetch_add(1, Ordering::Relaxed);
        }
        if result.is_ok() {
            lease.commit_recycling();
        }
        result
    }
}

struct PlanCache {
    hits: u64,
    misses: u64,
    replans: u64,
    topology_materializations: u64,
    /// O(1) LRU (HashMap + intrusive linked list): touch-on-hit, evict-LRU-on-
    /// insert, all O(1) — the Rust analog of TensorKit's `LRUCache.jl`-backed
    /// `GlobalLRUCache`. Capacity tracks `PlanCacheConfig::capacity`, resized on
    /// insert if the configured capacity changed.
    map: LruCache<Arc<NetworkTopology>, CacheEntry>,
    /// [`Network::contract`]'s aliases, keyed by the network's precomputed
    /// topology hash combined with the optimizer's. Each alias holds its full
    /// [`NetworkTopology`], which a lookup compares against the network, the
    /// operands' codomain ranks and the optimizer before it hits: a hash
    /// collision is a miss, never another network's plan.
    network_aliases: LruCache<u64, Vec<NetworkAlias>>,
    /// Persisted contraction orders keyed by stable topology text (see
    /// [`topology_text`]), populated by [`load_plan_cache`] and grown on
    /// every fresh search. A disk hit skips the (cold) optimal-order search
    /// entirely — the plancache analog of `@tensoropt`'s compile-time bake.
    /// LRU-bounded by `PlanCacheConfig::capacity`, like `map`.
    disk: LruCache<String, crate::plan::ContractionPlan>,
    /// Whether cross-process persistence is in use. Set by [`load_plan_cache`]
    /// (the application's opt-in) and only then is [`disk`] consulted/grown.
    /// Off by default so the in-memory replan behavior is byte-identical when
    /// persistence is not used: a persisted order recorded from an early
    /// non-degenerate search must not silently replace a later drift-replan's
    /// fresh search, which the truncation basis (hence energy) depends on.
    persist: bool,
    /// One Runtime-wide owner shared by every resident typed workspace pool.
    workspace_budget: Arc<WorkspaceBudget>,
}

struct NetworkAlias {
    topology: Arc<NetworkTopology>,
    dims_snapshot: Vec<Vec<usize>>,
    planned: Weak<PlannedNetwork>,
    workspaces: Weak<WorkspacePools>,
}

impl NetworkAlias {
    fn cached(&self) -> Option<CachedPlan> {
        Some(CachedPlan {
            planned: self.planned.upgrade()?,
            workspaces: self.workspaces.upgrade()?,
        })
    }
}

/// Whether `topology` is the topology [`topology_for`] would build for
/// `network`, `tensors` and the (already topology-normalized) `optimizer`,
/// compared field by field without building it.
fn topology_matches<R, D, S>(
    topology: &NetworkTopology,
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> bool
where
    R: TypedSectorAdmission,
    S: TensorStorage<D>,
{
    topology.optimizer == *optimizer
        && topology.output == network.output
        && topology.output_codomain_rank == network.output_codomain_rank
        && topology.operands.len() == network.inputs.len()
        && topology.operands.len() == tensors.len()
        && topology
            .operands
            .iter()
            .zip(&network.inputs)
            .zip(&network.conj)
            .zip(&network.codomain_splits)
            .zip(tensors)
            .all(|((((operand, labels), &conj), &split), tensor)| {
                operand.labels == *labels
                    && operand.conj == conj
                    && operand.written_split == split
                    && operand.codomain_rank == tensor.codomain_rank()
            })
}

fn network_alias_hash(network: &Network, optimizer: &Optimizer) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    network.topology_hash.hash(&mut hasher);
    optimizer.hash(&mut hasher);
    hasher.finish()
}

/// [`Network::contract`]'s plan lookup: the network's alias, else the
/// shared topology entry, else a persisted or fresh plan.
///
/// Hits, misses, promotions and aliases record the lookup, not the
/// execution, so the caller decides every metadata rejection of its operands
/// first (#1371). A later execution failure is then a runtime fault of a plan
/// that stays valid for these operands: the lookup stays counted, and only
/// the workspace leased for that call is quarantined. Why not count a hit
/// only after a successful execution: a miss publishes the fresh plan before
/// it runs, and withdrawing a valid plan (or its hit) on a provider or device
/// fault would only cost the next call a search.
pub(crate) fn get_or_plan_network<R, D, S>(
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> Result<CachedPlan, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let Some(runtime) = tensors.first().map(|tensor| tensor.runtime()) else {
        return Err(Error::InvalidArgument(
            "network execution requires at least one operand".to_string(),
        )
        .into());
    };
    let topology_optimizer = topology_optimizer(optimizer);
    let key = network_alias_hash(network, &topology_optimizer);
    let lookup = with_plan_cache(runtime, |config, slot| -> Result<Lookup, FacadeError<R>> {
        if !config.enabled {
            return Ok(Lookup::Disabled);
        }
        let Some(cache) = existing_cache_mut(slot) else {
            return Ok(Lookup::Miss);
        };
        let Some(aliases) = cache.network_aliases.peek(&key) else {
            return Ok(Lookup::Miss);
        };
        let Some(alias) = aliases
            .iter()
            .find(|alias| topology_matches(&alias.topology, network, tensors, &topology_optimizer))
        else {
            return Ok(Lookup::Miss);
        };
        if needs_replan_tensors(config.replan, &alias.dims_snapshot, tensors)? {
            return Ok(Lookup::Miss);
        }
        let Some(cached) = alias.cached() else {
            return Ok(Lookup::Miss);
        };
        let topology = alias.topology.clone();
        cache.network_aliases.promote(&key);
        Ok(match promote_if_resident(cache, &topology, cached) {
            Some(cached) => Lookup::Hit(cached),
            None => Lookup::Miss,
        })
    })?;
    match lookup {
        Lookup::Hit(cached) => return Ok(cached),
        Lookup::Disabled => return plan_uncached(network, tensors, optimizer),
        Lookup::Miss => {}
    }
    plan_through_topology(
        runtime,
        network,
        tensors,
        optimizer,
        |cache, dims_snapshot, topology, cached, capacity| {
            if cache.network_aliases.cap() != capacity {
                cache.network_aliases.resize(capacity);
            }
            let alias = NetworkAlias {
                topology,
                dims_snapshot,
                planned: Arc::downgrade(&cached.planned),
                workspaces: Arc::downgrade(&cached.workspaces),
            };
            let aliases = cache.network_aliases.get_or_insert_mut(key, Vec::new);
            match aliases
                .iter_mut()
                .find(|existing| *existing.topology == *alias.topology)
            {
                Some(existing) => *existing = alias,
                None => aliases.push(alias),
            }
        },
    )
}

/// Clamp a configured capacity to a non-zero LRU capacity (0 would disable
/// caching, which the search-once design never wants — treat it as 1).
fn lru_capacity(capacity: usize) -> NonZeroUsize {
    NonZeroUsize::new(capacity.max(1)).expect("capacity.max(1) is non-zero")
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(DEFAULT_WORKSPACE_BUDGET_BYTES)
    }
}

impl PlanCache {
    fn new(workspace_budget_bytes: usize) -> Self {
        Self {
            hits: 0,
            misses: 0,
            replans: 0,
            topology_materializations: 0,
            map: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            network_aliases: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            disk: LruCache::new(lru_capacity(DEFAULT_PLAN_CACHE_CAPACITY)),
            persist: false,
            workspace_budget: Arc::new(WorkspaceBudget::new(workspace_budget_bytes)),
        }
    }
}

/// Serialized-plan-cache format version. Bumped whenever the cost model or an
/// optimizer's order search changes so that a stale on-disk file (which would
/// otherwise replay a now-suboptimal order and silently drift truncation) is
/// rejected on load rather than trusted, and whenever the entry or plan text
/// format changes (4: plan text writes labels escaped, plan header v2).
const PLAN_CACHE_FILE_VERSION: &str = "SYMTN_PLANCACHE 4";

/// Stable one-line text key for a network topology: optimizer, output split
/// and labels, then each operand's conj / codomain rank / written split /
/// labels. Each label is written as a quoted, escaped string (`push_label`,
/// shared with the plan text), so a label holding a separator or a line break
/// cannot make two topologies share one key: `Network` labels are arbitrary
/// strings, not identifiers.
fn topology_text(topology: &NetworkTopology) -> String {
    let mut text = format!("{:?}|", topology.optimizer);
    match topology.output_codomain_rank {
        Some(rank) => text.push_str(&rank.to_string()),
        None => text.push('-'),
    }
    text.push('|');
    for (i, label) in topology.output.iter().enumerate() {
        if i > 0 {
            text.push(',');
        }
        push_label(&mut text, label.as_str());
    }
    for operand in &topology.operands {
        text.push('|');
        text.push(if operand.conj { '1' } else { '0' });
        text.push(':');
        text.push_str(&operand.codomain_rank.to_string());
        text.push(':');
        text.push_str(&operand.written_split.to_string());
        text.push(':');
        for (i, label) in operand.labels.iter().enumerate() {
            if i > 0 {
                text.push(',');
            }
            push_label(&mut text, label.as_str());
        }
    }
    text
}

// The cache lives in the runtime's `Any + Send` extension slot; plans are
// step lists + label vectors, so this holds by construction.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<PlanCacheHome>();
};

/// What this crate keeps in the runtime's extension slot: the configuration
/// and the lazily claimed cache, under one lock acquisition (#155) so the
/// network hot path resolves enable/replan policy and the lookup together.
#[derive(Default)]
struct PlanCacheHome {
    config: PlanCacheConfig,
    cache: Option<PlanCache>,
}

/// Configuration and cache of `runtime` under the runtime's one extension
/// lock. Do not run tensor operations inside `f`.
fn with_plan_cache<T>(
    runtime: &Runtime,
    f: impl FnOnce(&PlanCacheConfig, &mut Option<PlanCache>) -> T,
) -> T {
    __network::with_extension_slot(runtime, |slot| {
        let home = slot.get_or_insert_with(PlanCacheHome::default);
        f(&home.config, &mut home.cache)
    })
}

/// Stores `config` and lets `f` adapt the cache to the change from the
/// previous configuration, atomically.
fn replace_plan_cache_config<T>(
    runtime: &Runtime,
    config: PlanCacheConfig,
    f: impl FnOnce(&PlanCacheConfig, &PlanCacheConfig, &mut Option<PlanCache>) -> T,
) -> T {
    __network::with_extension_slot(runtime, |slot| {
        let home = slot.get_or_insert_with(PlanCacheHome::default);
        let result = f(&home.config, &config, &mut home.cache);
        home.config = config;
        result
    })
}

/// Snapshot of `runtime`'s contraction-plan-cache configuration.
pub fn plan_cache_config(runtime: &Runtime) -> PlanCacheConfig {
    with_plan_cache(runtime, |config, _| config.clone())
}

/// The runtime slot's cache, claimed (created) on first use.
fn cache_mut(slot: &mut Option<PlanCache>, workspace_budget_bytes: usize) -> &mut PlanCache {
    slot.get_or_insert_with(|| PlanCache::new(workspace_budget_bytes))
}

/// Read/write access that preserves an unclaimed runtime extension slot.
/// Rejected device plans must not make an otherwise-cold runtime own an empty
/// cache merely by probing it.
fn existing_cache_mut(slot: &mut Option<PlanCache>) -> Option<&mut PlanCache> {
    slot.as_mut()
}

/// Replaces the runtime's plan-cache configuration; the one setter (the getter
/// is [`plan_cache_config`]).
///
/// Retention: compiled plans survive every transition that keeps the cache
/// enabled, so re-tuning a live runtime never forces a re-plan. Disabling
/// drops all plans; lowering `capacity` evicts least-recently-used plans to
/// fit; lowering `workspace_budget_bytes` releases idle workspace storage
/// synchronously. Raising either bound retains everything.
pub fn configure_plan_cache(runtime: &Runtime, config: PlanCacheConfig) {
    replace_plan_cache_config(runtime, config, |previous, next, slot| {
        let Some(cache) = existing_cache_mut(slot) else {
            return;
        };
        // Before the shrink rotation, so no recycle reserves against the old
        // limit after the idle storage is released.
        cache
            .workspace_budget
            .set_limit(next.workspace_budget_bytes);

        if previous.enabled && !next.enabled {
            cache.map.clear();
            cache.network_aliases.clear();
        } else {
            if next.workspace_budget_bytes < previous.workspace_budget_bytes {
                for (_, entry) in cache.map.iter() {
                    entry.workspaces.rotate_all();
                }
            }

            if next.capacity < previous.capacity {
                let capacity = lru_capacity(next.capacity);
                cache.map.resize(capacity);
                cache.network_aliases.resize(capacity);
                cache.disk.resize(capacity);
            }
        }
    });
}

/// Hit/miss/re-plan counters and the current entry count.
pub fn plan_cache_stats(runtime: &Runtime) -> PlanCacheStats {
    with_plan_cache(runtime, |config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
        let (workspaces_created, workspace_reuses, workspace_slot_grows) =
            cache
                .map
                .iter()
                .fold((0, 0, 0), |(created, reused, grows), (_, entry)| {
                    (
                        created + entry.workspaces.counters.created.load(Ordering::Relaxed),
                        reused + entry.workspaces.counters.reused.load(Ordering::Relaxed),
                        grows + entry.workspaces.counters.slot_grows.load(Ordering::Relaxed),
                    )
                });
        let idle_workspaces = cache
            .map
            .iter()
            .map(|(_, entry)| entry.workspaces.counters.idle.load(Ordering::Relaxed) as usize)
            .sum();
        let (
            retained_workspace_bytes,
            peak_retained_workspace_bytes,
            workspace_byte_admissions,
            workspace_byte_rejections,
            workspace_byte_evictions,
        ) = cache.workspace_budget.snapshot();
        PlanCacheStats {
            hits: cache.hits,
            misses: cache.misses,
            replans: cache.replans,
            entries: cache.map.len(),
            persisted_orders: cache.disk.len(),
            workspaces_created,
            workspace_reuses,
            workspace_slot_grows,
            topology_materializations: cache.topology_materializations,
            idle_workspaces,
            retained_workspace_bytes,
            peak_retained_workspace_bytes,
            workspace_byte_admissions,
            workspace_byte_rejections,
            workspace_byte_evictions,
        }
    })
}

/// Drops every cached plan and resets the counters (not the configuration).
///
/// This includes the persisted orders loaded by [`load_plan_cache`] or
/// recorded since, and it turns persistence off until the next
/// [`load_plan_cache`], so the next miss searches afresh. Files the
/// application wrote from [`save_plan_cache`] are not touched.
pub fn clear_plan_cache(runtime: &Runtime) {
    with_plan_cache(runtime, |config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
        cache.map.clear();
        cache.network_aliases.clear();
        cache.disk.clear();
        cache.persist = false;
        cache.hits = 0;
        cache.misses = 0;
        cache.replans = 0;
        cache.topology_materializations = 0;
        cache.workspace_budget.reset_statistics();
    });
}

/// Serialize the persisted contraction orders (topology text + plan) to a
/// versioned text blob for the application to write to a cache file. Restore
/// it in a later process with [`load_plan_cache`] before the first contraction
/// to skip the cold optimal-order search. The order is topology-only and thus
/// dimension-independent, so one saved file serves every χ.
pub fn save_plan_cache(runtime: &Runtime) -> String {
    with_plan_cache(runtime, |config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
        let mut text = String::from(PLAN_CACHE_FILE_VERSION);
        text.push('\n');
        // Sort at save rather than switching `disk` to an ordered map: saving
        // is cold (once per process, at shutdown/checkpoint) while `disk` is
        // read on every cache miss, so paying the sort here keeps the hot
        // lookup path's hash lookup unchanged. Without this, the iteration
        // order (recency order) would make the saved bytes vary run to run
        // for identical content, breaking reproducible builds and
        // content-addressed/git-diffed cache blobs (issue #151).
        let mut entries: Vec<(&String, &crate::plan::ContractionPlan)> =
            cache.disk.iter().collect();
        entries.sort_by_key(|(topo, _)| topo.as_str());
        for (topo, plan) in entries {
            let plan_text = plan.to_text();
            text.push_str("TOPO ");
            text.push_str(topo);
            text.push('\n');
            text.push_str(&format!("PLAN {}\n", plan_text.lines().count()));
            text.push_str(&plan_text);
            if !plan_text.ends_with('\n') {
                text.push('\n');
            }
        }
        text
    })
}

/// Restore orders saved by [`save_plan_cache`]. A blob whose version header
/// does not match this build is ignored (returns 0): a stale file would
/// replay now-suboptimal orders and silently drift truncation, so it is
/// dropped rather than trusted. Returns the number of orders parsed.
///
/// A corrupt entry is skipped, not fatal: an entry whose `PLAN` line is
/// missing or has no numeric count, whose plan has fewer lines than that
/// count before the next `TOPO ` line, or whose plan text does not parse is
/// dropped, and loading resumes at the next line starting with `TOPO `, so
/// later valid entries still load. No plan line starts with `TOPO `, whatever
/// the labels: plan text writes every label quoted and escaped.
///
/// The persisted store is bounded by `PlanCacheConfig::capacity` entries
/// with least-recently-used eviction, the same bound as the compiled plans;
/// a larger file keeps its last `capacity` orders. [`clear_plan_cache`]
/// empties it and turns persistence off.
///
/// Loading any blob with a matching header, even an empty one, turns
/// persistence on for this runtime. From then on a topology that has a
/// persisted order replays it instead of searching, on a cold miss *and* on a
/// [`ReplanPolicy::DriftFactor`] drift-replan: drift still rebuilds the plan
/// for the current dimensions, but it does not re-search the order.
pub fn load_plan_cache(runtime: &Runtime, text: &str) -> usize {
    let mut lines = text.lines();
    // An empty blob is a fresh persistence file (first run): activate
    // persistence and load nothing. A non-empty blob with a mismatched version
    // header is stale/foreign and is ignored WITHOUT activating persistence, so
    // it neither replays bad orders nor perturbs in-memory replan numerics.
    let header = lines.next();
    if header.is_some() && header != Some(PLAN_CACHE_FILE_VERSION) {
        return 0;
    }
    with_plan_cache(runtime, |config, slot| {
        let cache = cache_mut(slot, config.workspace_budget_bytes);
        // The application opted into persistence: from now on record and reuse
        // orders through the disk map (even if this file was empty).
        cache.persist = true;
        let capacity = lru_capacity(config.capacity);
        if cache.disk.cap() != capacity {
            cache.disk.resize(capacity);
        }
        // Plan lines start with the plan header, `tensor_count`, `output` or
        // `step` (escaped labels hold no line break), so the next "TOPO " line
        // is a resynchronisation point after a corrupt entry.
        let mut lines = lines.peekable();
        let mut loaded = 0;
        while let Some(topo_line) = lines.next() {
            let Some(topo) = topo_line.strip_prefix("TOPO ") else {
                continue;
            };
            let Some(count) = lines
                .peek()
                .and_then(|l| l.strip_prefix("PLAN "))
                .and_then(|n| n.trim().parse::<usize>().ok())
            else {
                continue;
            };
            lines.next();
            let mut plan_text = String::new();
            let mut framed = 0;
            while framed < count {
                let Some(l) = lines.next_if(|l| !l.starts_with("TOPO ")) else {
                    break;
                };
                plan_text.push_str(l);
                plan_text.push('\n');
                framed += 1;
            }
            if framed != count {
                continue;
            }
            if let Ok(plan) = crate::plan::ContractionPlan::from_text(&plan_text) {
                cache.disk.put(topo.to_string(), plan);
                loaded += 1;
            }
        }
        loaded
    })
}

/// A plan made while some leg was trivial (dim ≤ 1) can encode a degenerate,
/// outer-product-heavy order that fits the real state poorly (reusing it is
/// catastrophically slow — that is what [`ReplanPolicy::BakeOnce`] guards
/// against). Once planned at non-degenerate dims the order is frozen.
fn snapshot_is_degenerate(snapshot: &[Vec<usize>]) -> bool {
    snapshot.iter().flatten().any(|&d| d <= 1)
}

/// Whether a topology-matched cache entry must be re-planned given how its
/// leg dims have drifted, per the [`ReplanPolicy`].
fn needs_replan(policy: ReplanPolicy, snapshot: &[Vec<usize>], current: &[Vec<usize>]) -> bool {
    match policy {
        ReplanPolicy::AlwaysReuse => false,
        // Reuse the once-found path for any real dims (cotengra/@tensoropt
        // style); only replace a plan seeded at degenerate dims, and only
        // once the dims have actually moved off that seed.
        ReplanPolicy::BakeOnce => {
            snapshot_is_degenerate(snapshot)
                && snapshot.iter().flatten().ne(current.iter().flatten())
        }
        ReplanPolicy::DriftFactor(factor) => snapshot
            .iter()
            .flatten()
            .zip(current.iter().flatten())
            .any(|(&snap, &cur)| {
                if snap == cur {
                    return false;
                }
                if snap == 0 || cur == 0 {
                    return true;
                }
                let ratio = snap.max(cur) as f64 / snap.min(cur) as f64;
                ratio > factor
            }),
    }
}

fn needs_replan_tensors<R, D, S>(
    policy: ReplanPolicy,
    snapshot: &[Vec<usize>],
    tensors: &[&TensorMap<R, D, S>],
) -> Result<bool, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    // The per-operand rank guard runs for every policy: the snapshot is only
    // comparable leg by leg when the ranks agree, and a tensor of another rank
    // must force a replan rather than reuse a plan built for a different rank.
    for (operand, dims) in snapshot.iter().enumerate() {
        if tensors[operand].rank() != dims.len() {
            return Ok(true);
        }
    }

    // Past the rank guard the per-axis leg-dimension scan is dead work for the
    // policies whose result never depends on it: `AlwaysReuse` never replans on
    // drift, and a non-degenerate `BakeOnce` snapshot is frozen for any real
    // dims. Skipping the scan loses no error side effect: an axis lookup could
    // only fail on `axis >= rank`, which the guard above already precludes. `DriftFactor` (and a
    // degenerate `BakeOnce` seed) still need the full comparison.
    match policy {
        ReplanPolicy::AlwaysReuse => return Ok(false),
        ReplanPolicy::BakeOnce if !snapshot_is_degenerate(snapshot) => return Ok(false),
        _ => {}
    }

    let mut changed = false;
    let mut exceeds_factor = false;
    for (operand, dims) in snapshot.iter().enumerate() {
        let current_dims = <R::Mode as NetworkModeDispatch<R, D>>::leg_dims(tensors[operand])?;
        for (&snap, current) in dims.iter().zip(current_dims) {
            changed |= snap != current;
            if snap != current {
                exceeds_factor |= match policy {
                    ReplanPolicy::DriftFactor(factor) if snap != 0 && current != 0 => {
                        snap.max(current) as f64 / snap.min(current) as f64 > factor
                    }
                    ReplanPolicy::DriftFactor(_) => true,
                    _ => false,
                };
            }
        }
    }
    Ok(match policy {
        ReplanPolicy::AlwaysReuse => false,
        ReplanPolicy::BakeOnce => snapshot_is_degenerate(snapshot) && changed,
        ReplanPolicy::DriftFactor(_) => exceeds_factor,
    })
}

enum Lookup {
    /// Caching is off; execute uncached.
    Disabled,
    Hit(CachedPlan),
    Miss,
}

/// Confirms the alias still points at the resident cache entry AND promotes it
/// to most-recently-used in ONE `LruCache::get` (#155): `get` moves the entry to
/// MRU and hands it back, so the residency check (Arc identity) and the LRU
/// touch share a single `NetworkTopology` hash instead of a `peek` (residency)
/// followed by a `promote`. Counts the hit only on a confirmed match. A miss
/// here (evicted, or the topology now maps to a different plan) is harmless: the
/// caller replans, and promoting whatever currently holds the key — or nothing —
/// changes only LRU order.
fn promote_if_resident(
    cache: &mut PlanCache,
    topology: &Arc<NetworkTopology>,
    cached: CachedPlan,
) -> Option<CachedPlan> {
    let resident = match cache.map.get(topology) {
        Some(entry) => {
            Arc::ptr_eq(&entry.planned, &cached.planned)
                && Arc::ptr_eq(&entry.workspaces, &cached.workspaces)
        }
        None => false,
    };
    if resident {
        cache.hits += 1;
        Some(cached)
    } else {
        None
    }
}

fn plan_fresh<R, D, S>(
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> Result<PlannedNetwork, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    match optimizer {
        Optimizer::Greedy => network.plan(tensors, &GreedyDenseOptimizer),
        #[cfg(feature = "opt-path")]
        Optimizer::Optimal => network.plan(
            tensors,
            &crate::pathopt::OptEinsumPathOptimizer::new(crate::pathopt::PathStrategy::Optimal),
        ),
        // Dynamic programming yields the optimal order in polynomial time for
        // symtn's small networks — the `@tensoropt` analog without exhaustive
        // search cost. Upstream `dp` errors on all-dim-1 networks (the same
        // degenerate case `auto-hq` trips on), where the order is irrelevant
        // anyway, so fall back to greedy there.
        #[cfg(feature = "opt-path")]
        Optimizer::DynamicProgramming => {
            use crate::pathopt::{OptEinsumPathOptimizer, PathStrategy};
            let dynamic = OptEinsumPathOptimizer::new(PathStrategy::DynamicProgramming);
            network.plan_with_optimizer_fallbacks(tensors, &dynamic, &[&GreedyDenseOptimizer])
        }
        // Fallback chain: auto-hq -> auto -> dp
        // -> greedy. Upstream `opt-einsum-path` errors on some all-dim-1
        // networks, so each failed driver falls through to the next.
        #[cfg(feature = "opt-path")]
        Optimizer::AutoHq => {
            use crate::pathopt::{OptEinsumPathOptimizer, PathStrategy};
            let auto_hq = OptEinsumPathOptimizer::new(PathStrategy::AutoHq);
            let auto = OptEinsumPathOptimizer::new(PathStrategy::Auto);
            let dynamic = OptEinsumPathOptimizer::new(PathStrategy::DynamicProgramming);
            network.plan_with_optimizer_fallbacks(
                tensors,
                &auto_hq,
                &[&auto, &dynamic, &GreedyDenseOptimizer],
            )
        }
        #[cfg(feature = "cotengra-python")]
        Optimizer::CotengraPython(config) => network.plan(
            tensors,
            &crate::cotengra_python::CotengraPythonOptimizer::new(config.clone()),
        ),
    }
}

fn topology_optimizer(optimizer: &Optimizer) -> Optimizer {
    #[cfg(feature = "cotengra-python")]
    if let Optimizer::CotengraPython(config) = optimizer {
        let mut config = config.clone();
        // Normal cached contractions are path-only. `optimize_sliced` consumes
        // slicing explicitly and does not go through this cache, so slicing
        // policy must not fragment ordinary plan-cache entries.
        config.slicing = crate::plancache::CotengraSlicingConfig::None;
        // The timeout supervises a cold search; it cannot change the resulting
        // plan, so it is not part of topology or persisted-plan identity.
        config.timeout = None;
        return Optimizer::CotengraPython(config);
    }
    optimizer.clone()
}

fn topology_for<R, D, S>(
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> Arc<NetworkTopology>
where
    R: TypedSectorAdmission,
    S: TensorStorage<D>,
{
    Arc::new(NetworkTopology {
        operands: network
            .inputs
            .iter()
            .zip(&network.conj)
            .zip(&network.codomain_splits)
            .zip(tensors)
            .map(
                |(((labels, &conj), &written_split), tensor)| OperandTopology {
                    labels: labels.clone(),
                    conj,
                    codomain_rank: tensor.codomain_rank(),
                    written_split,
                },
            )
            .collect(),
        output: network.output.clone(),
        output_codomain_rank: network.output_codomain_rank,
        optimizer: topology_optimizer(optimizer),
    })
}

/// One uncached plan, for a Runtime whose plan cache is disabled.
fn plan_uncached<R, D, S>(
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
) -> Result<CachedPlan, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    Ok(CachedPlan {
        planned: Arc::new(plan_fresh(network, tensors, optimizer)?),
        workspaces: Arc::new(WorkspacePools::unpooled()),
    })
}

/// The topology-keyed half of a lookup that missed its caller's alias: a
/// topology hit, else a persisted or fresh plan published under the
/// topology. `install_alias` records the caller's alias to the plan it
/// returns, with the planning-time dims that alias measures drift from.
fn plan_through_topology<R, D, S>(
    runtime: &Runtime,
    network: &Network,
    tensors: &[&TensorMap<R, D, S>],
    optimizer: &Optimizer,
    mut install_alias: impl FnMut(
        &mut PlanCache,
        Vec<Vec<usize>>,
        Arc<NetworkTopology>,
        &CachedPlan,
        NonZeroUsize,
    ),
) -> Result<CachedPlan, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let dims: Vec<Vec<usize>> = tensors
        .iter()
        .map(|tensor| <R::Mode as NetworkModeDispatch<R, D>>::leg_dims(tensor))
        .collect::<Result<_, _>>()?;
    let topology = topology_for(network, tensors, optimizer);

    // A hit carries the entry's planning-time dims: the alias must measure
    // drift from them, not from this caller's dims, or each network that
    // reaches the entry through a topology hit would re-base the snapshot and
    // let drift compound to factor² across networks (#1740).
    enum Outcome {
        Hit(CachedPlan, Vec<Vec<usize>>),
        Replan,
        Miss,
    }
    let outcome = with_plan_cache(runtime, |config, slot| -> Result<Outcome, FacadeError<R>> {
        let Some(cache) = existing_cache_mut(slot) else {
            return Ok(Outcome::Miss);
        };
        // `peek` inspects without touching LRU order, so a stale entry that will
        // be replanned does not count as a use; a genuine hit is promoted to
        // most-recently-used with an O(1) `promote`.
        match cache.map.peek(&topology) {
            Some(entry) if !needs_replan(config.replan, &entry.dims_snapshot, &dims) => {
                let planned = CachedPlan {
                    planned: Arc::clone(&entry.planned),
                    workspaces: Arc::clone(&entry.workspaces),
                };
                let snapshot = entry.dims_snapshot.clone();
                cache.hits += 1;
                cache.map.promote(&topology);
                Ok(Outcome::Hit(planned, snapshot))
            }
            Some(_) => Ok(Outcome::Replan),
            None => Ok(Outcome::Miss),
        }
    })?;
    if let Outcome::Hit(planned, snapshot) = outcome {
        with_plan_cache(runtime, |config, slot| {
            install_alias(
                cache_mut(slot, config.workspace_budget_bytes),
                snapshot,
                topology,
                &planned,
                lru_capacity(config.capacity),
            )
        });
        return Ok(planned);
    }

    // With persistence in use, consult the persisted orders before paying for
    // a fresh search — on a miss AND on a drift-replan (a degenerate seed
    // reused at real dims still pays the full search otherwise). Disk plans are
    // only ever recorded from non-degenerate searches, so a disk hit wraps that
    // good order via `plan_with`, skipping the cold optimal-order search. When
    // persistence is off the disk map is never touched, keeping in-memory
    // replan numerics byte-identical.
    let topo_key = topology_text(&topology);
    let disk_plan = with_plan_cache(runtime, |_, slot| {
        existing_cache_mut(slot).and_then(|cache| {
            cache
                .persist
                .then(|| cache.disk.get(&topo_key).cloned())
                .flatten()
        })
    });
    let (planned, fresh_plan_copy) = match disk_plan {
        Some(plan) => (Arc::new(network.plan_with(tensors, plan)?), None),
        None => {
            let fresh = Arc::new(plan_fresh(network, tensors, optimizer)?);
            // Record the freshly searched order so a later process reusing
            // this cache file skips the search — but only under persistence and
            // only when searched at non-degenerate dims. A degenerate seed
            // (dim ≤ 1) yields the outer-product-heavy order `BakeOnce` exists
            // to reject; persisting it would replay that bad order on reuse.
            let plan_copy = (!snapshot_is_degenerate(&dims)).then(|| fresh.plan().clone());
            (fresh, plan_copy)
        }
    };
    let workspace_budget = with_plan_cache(runtime, |config, slot| {
        Arc::clone(&cache_mut(slot, config.workspace_budget_bytes).workspace_budget)
    });
    let candidate = CachedPlan {
        planned,
        workspaces: Arc::new(WorkspacePools::new(workspace_budget)),
    };
    let winner = with_plan_cache(
        runtime,
        |config, slot| -> Result<CachedPlan, FacadeError<R>> {
            let cache = cache_mut(slot, config.workspace_budget_bytes);
            if !config.enabled {
                candidate.workspaces.deactivate_all();
                return Ok(candidate.clone());
            }
            let capacity = lru_capacity(config.capacity);
            if cache.map.cap() != capacity {
                cache.map.resize(capacity);
            }
            let (cached, snapshot) = match cache.map.peek(&topology) {
                Some(entry) if !needs_replan(config.replan, &entry.dims_snapshot, &dims) => {
                    cache.hits += 1;
                    let cached = CachedPlan {
                        planned: Arc::clone(&entry.planned),
                        workspaces: Arc::clone(&entry.workspaces),
                    };
                    (cached, entry.dims_snapshot.clone())
                }
                _ => {
                    cache.topology_materializations += 1;
                    if cache.persist {
                        if let Some(plan_copy) = &fresh_plan_copy {
                            if cache.disk.cap() != capacity {
                                cache.disk.resize(capacity);
                            }
                            cache.disk.put(topo_key.clone(), plan_copy.clone());
                        }
                    }
                    match outcome {
                        Outcome::Replan => cache.replans += 1,
                        _ => cache.misses += 1,
                    }
                    let cached = candidate.clone();
                    cache.map.put(
                        topology.clone(),
                        CacheEntry {
                            planned: Arc::clone(&cached.planned),
                            workspaces: Arc::clone(&cached.workspaces),
                            dims_snapshot: dims.clone(),
                        },
                    );
                    (cached, dims.clone())
                }
            };
            cache.map.promote(&topology);
            install_alias(cache, snapshot, topology.clone(), &cached, capacity);
            Ok(cached)
        },
    )?;
    Ok(winner)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cotengra-python")]
    use super::topology_optimizer;
    use super::Optimizer;
    use super::{needs_replan, ReplanPolicy, WorkspaceBudget, WorkspacePools};
    use crate::network::NetworkExecutionWorkspace;
    use crate::{Network, TemporaryLabel};
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier};
    #[cfg(feature = "cotengra-python")]
    use std::time::Duration;
    use symtn::sector::{SU2FusionRule, U1FusionRule};
    use symtn::typed::Complex64;
    use symtn::typed::Engine;
    use symtn::typed::{ContractSpec, GradedSpace, TensorMap};

    #[cfg(feature = "cotengra-python")]
    #[test]
    fn cotengra_timeout_does_not_fragment_cache_identity_or_mutate_runtime_config() {
        let short =
            crate::plancache::CotengraPythonConfig::default().timeout(Duration::from_secs(1));
        let unbounded = crate::plancache::CotengraPythonConfig::default().without_timeout();
        let runtime_config = crate::plancache::PlanCacheConfig {
            optimizer: Optimizer::CotengraPython(short),
            ..Default::default()
        };

        assert_eq!(
            topology_optimizer(&runtime_config.optimizer),
            topology_optimizer(&Optimizer::CotengraPython(unbounded))
        );
        let Optimizer::CotengraPython(config) = &runtime_config.optimizer else {
            unreachable!()
        };
        assert_eq!(config.timeout, Some(Duration::from_secs(1)));
    }

    /// Plan-cache state a metadata rejection must leave as it was: every
    /// counter and workspace statistic, and the network aliases.
    #[derive(Debug, PartialEq)]
    struct CacheState {
        stats: super::PlanCacheStats,
        aliases: Option<usize>,
    }

    fn cache_state(runtime: &symtn::typed::Runtime) -> CacheState {
        CacheState {
            stats: super::plan_cache_stats(runtime),
            aliases: super::with_plan_cache(runtime, |_, slot| {
                super::existing_cache_mut(slot).map(|cache| {
                    cache
                        .network_aliases
                        .iter()
                        .map(|(_, aliases)| aliases.len())
                        .sum::<usize>()
                })
            }),
        }
    }

    type Call<'a> = &'a dyn Fn(&mut dyn Engine) -> Result<(), String>;

    /// #1371: `reject` is rejected on the plan-cache miss, topology-hit and
    /// alias-hit paths of the `Network::contract` call that `accept` publishes,
    /// leaving [`CacheState`] and `extra` (device transfers) unchanged, with
    /// one error on every path, which is returned.
    fn assert_rejected_before_the_lookup<X: PartialEq + std::fmt::Debug>(
        engine: &mut dyn Engine,
        runtime: &symtn::typed::Runtime,
        what: &str,
        accept: Call<'_>,
        reject: Call<'_>,
        extra: &dyn Fn() -> X,
    ) -> String {
        let clear_aliases = || {
            super::with_plan_cache(runtime, |_, slot| {
                super::existing_cache_mut(slot)
                    .unwrap()
                    .network_aliases
                    .clear();
            })
        };
        let rejected = |engine: &mut dyn Engine, path: &str| {
            let before = (cache_state(runtime), extra());
            let error = reject(engine).expect_err(what);
            assert_eq!((cache_state(runtime), extra()), before, "{what}: {path}");
            error
        };

        super::clear_plan_cache(runtime);
        let error = rejected(engine, "miss");
        assert_eq!(super::plan_cache_stats(runtime).entries, 0, "{what}");

        accept(engine).unwrap();
        clear_aliases();
        assert_eq!(cache_state(runtime).aliases, Some(0), "{what}: setup");
        assert_eq!(
            rejected(engine, "topology hit"),
            error,
            "{what}: topology hit"
        );
        // Non-vacuity: the same site accepted is a topology hit, which
        // installs the alias the next rejection meets.
        let hits = super::plan_cache_stats(runtime).hits;
        accept(engine).unwrap();
        assert_eq!(super::plan_cache_stats(runtime).hits, hits + 1, "{what}");
        assert_eq!(cache_state(runtime).aliases, Some(1), "{what}: alias");

        assert_eq!(rejected(engine, "alias hit"), error, "{what}: alias hit");
        let hits = super::plan_cache_stats(runtime).hits;
        accept(engine).unwrap();
        assert_eq!(super::plan_cache_stats(runtime).hits, hits + 1, "{what}");
        error
    }

    /// The two networks of the #1371 tests: `[i; k] = a[i; j] * b[j; k]`, and
    /// a contraction of two codomain legs, neither read as a dual,
    /// `[i; k] = a[i, j;] * b[j, k;]`. Each is one topology for every
    /// provider and storage, so a probe operand meets the plan a U(1) operand
    /// of the same leg dimensions published.
    fn pair<T>(engine: &mut dyn Engine, a: &T, b: &T) -> Result<(), String>
    where
        T: crate::NetworkOperand,
        T::Error: std::fmt::Display,
    {
        Network::new(
            vec![vec!["i", "j"], vec!["j", "k"]],
            vec![false, false],
            vec![1, 1],
            vec!["i", "k"],
            Some(1),
        )
        .unwrap()
        .contract(engine, &[a, b])
        .map(drop)
        .map_err(|error| error.to_string())
    }

    fn codomain_pair<T>(engine: &mut dyn Engine, a: &T, b: &T) -> Result<(), String>
    where
        T: crate::NetworkOperand,
        T::Error: std::fmt::Display,
    {
        Network::new(
            vec![vec!["i", "j"], vec!["j", "k"]],
            vec![false, false],
            vec![2, 2],
            vec!["i", "k"],
            Some(1),
        )
        .unwrap()
        .contract(engine, &[a, b])
        .map(drop)
        .map_err(|error| error.to_string())
    }

    /// The operands of the #1371 tests, every leg of dimension 2: `v` has two
    /// sectors, `w` one, and `v*` is `v`'s dual.
    struct Rejections<T, P> {
        a: T,
        b: T,
        b_sector: T,
        b_dual: T,
        probe_a: P,
        probe_b: P,
        /// `[x, x;]`, `[x*, x;]` and its two impostors for a contraction of
        /// two undualised codomain legs, `x = {0: 1, 1: 2}`.
        cc_a: T,
        cc_b: T,
        cc_degeneracy: T,
        cc_flag: T,
    }

    type ProbeMap<const ANYONIC: bool> =
        TensorMap<crate::braiding_probe::RealBraidingProbe<ANYONIC>, f64>;

    fn rejections(
        engine: &mut dyn Engine,
        runtime: &symtn::typed::Runtime,
    ) -> Rejections<TensorMap<U1FusionRule, f64>, ProbeMap<true>> {
        use crate::braiding_probe::{ProbeSector, RealBraidingProbe};
        use symtn::sector::U1Irrep;
        let provider = Arc::new(U1FusionRule);
        let v = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
        )
        .unwrap();
        let w = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
        let v_dual = v.try_dual().unwrap();
        let x = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        // The sectors of `x*` with the degeneracies swapped, and the sectors
        // and degeneracies of `x*` on a leg that is not dual.
        let x_degeneracy = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        )
        .unwrap()
        .try_dual()
        .unwrap();
        let x_flag = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 1), (U1Irrep::new(-1), 2)],
        )
        .unwrap();
        let x_dual = x.try_dual().unwrap();
        // Oracle: the dualise-and-compare the preflight replaced.
        assert!(x_degeneracy != x_dual && x_flag != x_dual);
        assert_eq!(x_degeneracy.dim().unwrap(), x_dual.dim().unwrap());
        assert_eq!(x_flag.dim().unwrap(), x_dual.dim().unwrap());
        let p =
            GradedSpace::try_new(Arc::new(RealBraidingProbe::<true>), [(ProbeSector, 2)]).unwrap();
        let u1 = |engine: &mut dyn Engine,
                  codomain: &[&GradedSpace<U1FusionRule>],
                  domain: &[&GradedSpace<U1FusionRule>],
                  seed| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(
                engine,
                runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                seed,
            )
            .unwrap()
        };
        let probe = |engine: &mut dyn Engine, rank: usize, seed| {
            TensorMap::rand_with_seed(engine, runtime, vec![&p; rank], vec![&p; rank], seed)
                .unwrap()
        };
        Rejections {
            a: u1(engine, &[&v], &[&v], 1_371_000),
            b: u1(engine, &[&v], &[&v], 1_371_001),
            b_sector: u1(engine, &[&w], &[&v], 1_371_002),
            b_dual: u1(engine, &[&v_dual], &[&v], 1_371_003),
            probe_a: probe(engine, 1, 1_371_007),
            probe_b: probe(engine, 1, 1_371_008),
            cc_a: u1(engine, &[&x, &x], &[], 1_371_010),
            cc_b: u1(engine, &[&x_dual, &x], &[], 1_371_011),
            cc_degeneracy: u1(engine, &[&x_degeneracy, &x], &[], 1_371_012),
            cc_flag: u1(engine, &[&x_flag, &x], &[], 1_371_013),
        }
    }

    /// #1274: an execution that fails after the lease quarantines the
    /// leased workspace instead of recycling buffers the failure may have
    /// disturbed, and the next call builds a fresh one. After the #1371
    /// preflight only a runtime fault fails an execution, so the fault is
    /// injected through the executor `execute_leased` already takes.
    #[test]
    fn a_failed_execution_quarantines_its_lease() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = symtn::typed::Runtime::builder().build().unwrap();
        let o = rejections(engine, &runtime);
        let tensors = [&o.a, &o.b];
        let network = Network::new(
            vec![vec!["i", "j"], vec!["j", "k"]],
            vec![false, false],
            vec![1, 1],
            vec!["i", "k"],
            Some(1),
        )
        .unwrap();
        let cached = super::get_or_plan_network(&network, &tensors, &Default::default()).unwrap();
        cached.execute(engine, &tensors).unwrap();
        let warm = super::plan_cache_stats(&runtime);
        assert_eq!(warm.idle_workspaces, 1, "a success recycles its lease");

        let failed = cached.execute_leased(&tensors, |_, _, _| {
            Err(symtn::typed::Error::InvalidArgument(
                "injected fault".to_string(),
            ))
        });
        assert!(failed.is_err());
        let after = super::plan_cache_stats(&runtime);
        assert_eq!(after.idle_workspaces, 0, "the failed lease is quarantined");

        cached.execute(engine, &tensors).unwrap();
        let rebuilt = super::plan_cache_stats(&runtime);
        assert_eq!(
            rebuilt.workspaces_created,
            after.workspaces_created + 1,
            "the quarantined workspace is replaced, not resurrected"
        );
    }

    /// #2269: a written `;` split that is not the tensor's codomain rank is
    /// rejected before the plan-cache lookup on the miss, topology-hit and
    /// alias-hit paths of the same `Network`, so the alias the accepted call
    /// installed (keyed by the network, not the tensors) is never reached.
    #[test]
    fn a_mismatched_written_split_leaves_the_plan_cache_untouched_on_every_path() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = symtn::typed::Runtime::builder().build().unwrap();
        let o = rejections(engine, &runtime);
        let none = || ();
        for (what, accept, reject, split, rank) in [
            (
                "split 1 on codomain rank 2",
                &(|engine: &mut dyn Engine| pair(engine, &o.a, &o.b)) as Call<'_>,
                &(|engine: &mut dyn Engine| pair(engine, &o.cc_a, &o.cc_b)) as Call<'_>,
                1,
                2,
            ),
            (
                "split 2 on codomain rank 1",
                &|engine: &mut dyn Engine| codomain_pair(engine, &o.cc_a, &o.cc_b),
                &|engine: &mut dyn Engine| codomain_pair(engine, &o.a, &o.b),
                2,
                1,
            ),
        ] {
            let error =
                assert_rejected_before_the_lookup(engine, &runtime, what, accept, reject, &none);
            assert_eq!(
                error,
                format!(
                    "invalid argument: operand 0 puts {split} label(s) before `;` but the tensor's codomain rank is {rank}"
                ),
                "{what}"
            );
        }
    }

    /// #1371: a Host `Network::contract` whose operands are
    /// inadmissible from metadata alone — a contracted leg of another sector
    /// structure or of the wrong duality, or a non-symmetric braiding that
    /// contracts — is rejected before the plan-cache lookup on the miss,
    /// topology-hit and alias-hit paths: counters, workspaces and network
    /// aliases stay as they were, and the error is one text on every path,
    /// the typed `contract`'s for braiding.
    #[test]
    fn host_metadata_rejections_leave_the_plan_cache_untouched_on_every_path() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = symtn::typed::Runtime::builder().build().unwrap();
        let o = rejections(engine, &runtime);
        let none = || ();
        let accept_pair = |engine: &mut dyn Engine| pair(engine, &o.a, &o.b);
        let accept_codomain = |engine: &mut dyn Engine| codomain_pair(engine, &o.cc_a, &o.cc_b);
        for (what, accept, reject, label) in [
            (
                "degeneracy",
                &accept_codomain as Call<'_>,
                &(|engine: &mut dyn Engine| codomain_pair(engine, &o.cc_a, &o.cc_degeneracy))
                    as Call<'_>,
                "j",
            ),
            (
                "undualised flag",
                &accept_codomain,
                &|engine: &mut dyn Engine| codomain_pair(engine, &o.cc_a, &o.cc_flag),
                "j",
            ),
            (
                "sector",
                &accept_pair as Call<'_>,
                &(|engine: &mut dyn Engine| pair(engine, &o.a, &o.b_sector)) as Call<'_>,
                "j",
            ),
            (
                "duality",
                &accept_pair,
                &|engine: &mut dyn Engine| pair(engine, &o.a, &o.b_dual),
                "j",
            ),
        ] {
            let error =
                assert_rejected_before_the_lookup(engine, &runtime, what, accept, reject, &none);
            assert!(
                error.starts_with(&format!(
                    "invalid argument: space mismatch for contracted label `{label}` between operand 0 leg 1 and operand 1 leg 0"
                )),
                "{what}: {error}"
            );
        }

        let contract_error = o
            .probe_a
            .contract(
                engine,
                &o.probe_b,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap_err()
            .to_string();
        let error = assert_rejected_before_the_lookup(
            engine,
            &runtime,
            "non-symmetric",
            &accept_pair,
            &|engine: &mut dyn Engine| pair(engine, &o.probe_a, &o.probe_b),
            &none,
        );
        assert_eq!(error, contract_error, "non-symmetric");
    }

    /// #1371 on the device: the same rejections, before any lease or plan
    /// lookup, with the Host's error.
    ///
    /// The transfer counters are process-wide: run with `--test-threads=1`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn device_metadata_rejections_leave_the_plan_cache_and_device_untouched_on_every_path() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let cuda = &mut symtn::typed::CudaEngine::open(symtn::typed::CudaDeviceId::from_ordinal(0))
            .unwrap();
        let runtime = symtn::typed::Runtime::builder().build().unwrap();
        let h = rejections(engine, &runtime);
        let (probe_a, probe_b) = (
            h.probe_a.to_cuda(&mut *cuda).unwrap(),
            h.probe_b.to_cuda(&mut *cuda).unwrap(),
        );
        let mut lift = |tensor: &TensorMap<U1FusionRule, f64>| tensor.to_cuda(&mut *cuda).unwrap();
        let (a, b, b_sector, b_dual) = (lift(&h.a), lift(&h.b), lift(&h.b_sector), lift(&h.b_dual));
        let (cc_a, cc_b, cc_degeneracy, cc_flag) = (
            lift(&h.cc_a),
            lift(&h.cc_b),
            lift(&h.cc_degeneracy),
            lift(&h.cc_flag),
        );
        let transfers = symtn::expert::cuda_transfer_stats;
        let accept_pair = |e: &mut dyn Engine| pair(e, &a, &b);
        let accept_codomain = |e: &mut dyn Engine| codomain_pair(e, &cc_a, &cc_b);
        // Warm the engine so lazily created state is not attributed to the
        // rejected calls.
        accept_pair(cuda).unwrap();
        for (what, accept, reject, host) in [
            (
                "degeneracy",
                &accept_codomain as Call<'_>,
                &(|e: &mut dyn Engine| codomain_pair(e, &cc_a, &cc_degeneracy)) as Call<'_>,
                codomain_pair(engine, &h.cc_a, &h.cc_degeneracy),
            ),
            (
                "undualised flag",
                &accept_codomain,
                &|e: &mut dyn Engine| codomain_pair(e, &cc_a, &cc_flag),
                codomain_pair(engine, &h.cc_a, &h.cc_flag),
            ),
            (
                "sector",
                &accept_pair,
                &(|e: &mut dyn Engine| pair(e, &a, &b_sector)) as Call<'_>,
                pair(engine, &h.a, &h.b_sector),
            ),
            (
                "duality",
                &accept_pair,
                &|e: &mut dyn Engine| pair(e, &a, &b_dual),
                pair(engine, &h.a, &h.b_dual),
            ),
            (
                "non-symmetric",
                &accept_pair,
                &|e: &mut dyn Engine| pair(e, &probe_a, &probe_b),
                pair(engine, &h.probe_a, &h.probe_b),
            ),
        ] {
            let error =
                assert_rejected_before_the_lookup(cuda, &runtime, what, accept, reject, &transfers);
            assert_eq!(Err(error), host, "{what}");
        }
    }

    /// #2022: `Network` labels are arbitrary strings, so a label holding the
    /// key's separators must not make two topologies share a persisted key.
    /// Unescaped, both of these would be `0:1:1:a,b`.
    #[test]
    fn topology_text_keeps_separator_labels_distinct() {
        let topology = |labels: &[&str]| super::NetworkTopology {
            operands: vec![super::OperandTopology {
                labels: labels.iter().copied().map(TemporaryLabel::from).collect(),
                conj: false,
                codomain_rank: 1,
                written_split: 1,
            }],
            output: Vec::new(),
            output_codomain_rank: Some(0),
            optimizer: Optimizer::default(),
        };
        let joined = super::topology_text(&topology(&["a,b"]));
        let split = super::topology_text(&topology(&["a", "b"]));
        assert_ne!(joined, split);
        let line_break = super::topology_text(&topology(&["a\nTOPO x"]));
        assert!(!line_break.contains('\n'), "{line_break}");
    }

    #[test]
    fn replan_policy_matches_dimension_drift_semantics() {
        let snapshot = vec![vec![2, 3]];
        assert!(!needs_replan(
            ReplanPolicy::AlwaysReuse,
            &snapshot,
            &[vec![2]],
        ));
        assert!(!needs_replan(
            ReplanPolicy::AlwaysReuse,
            &snapshot,
            &[vec![9, 11]],
        ));
        assert!(needs_replan(
            ReplanPolicy::DriftFactor(2.0),
            &snapshot,
            &[vec![5, 3]],
        ));
    }

    #[test]
    fn panic_quarantines_typed_workspace_lease() {
        let pools = WorkspacePools::default();
        let pool = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let pool = pool.clone();
            move || {
                let _lease = pool.lease();
                panic!("injected workspace panic");
            }
        }));
        assert!(result.is_err());
        assert!(pool.available.lock().unwrap().is_empty());
        let mut lease = pool.lease();
        lease.commit_recycling();
        drop(lease);
        assert_eq!(
            pools
                .counters
                .created
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn active_lease_return_after_typed_pool_eviction_is_not_retained() {
        let pools = WorkspacePools::default();
        let first = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let mut lease = first.lease();
        lease.commit_recycling();
        drop(pools.pool::<U1FusionRule, Complex64, Vec<Complex64>>());
        drop(pools.pool::<SU2FusionRule, f64, Vec<f64>>());
        assert!(!first.registered.load(Ordering::SeqCst));
        drop(lease);
        assert!(first.available.lock().unwrap().is_empty());
        assert_eq!(pools.counters.idle.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn active_lease_return_after_plan_clear_is_not_retained() {
        let pools = WorkspacePools::default();
        let pool = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let mut lease = pool.lease();
        lease.commit_recycling();
        pools.deactivate_all();
        drop(lease);
        assert!(pool.available.lock().unwrap().is_empty());
        assert_eq!(pools.counters.idle.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn deactivated_registry_rejects_late_typed_pool_creation() {
        let pools = WorkspacePools::default();
        pools.deactivate_all();
        let pool = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        assert!(!pool.registered.load(Ordering::SeqCst));
        let mut lease = pool.lease();
        lease.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(16));
        lease.commit_recycling();
        drop(lease);
        assert!(pool.available.lock().unwrap().is_empty());
        assert_eq!(pools.budget.snapshot(), (0, 0, 0, 0, 0));
    }

    #[test]
    fn budget_rotation_rejects_old_return_and_accepts_new_pool() {
        let budget = Arc::new(WorkspaceBudget::new(usize::MAX));
        let pools = WorkspacePools::new(Arc::clone(&budget));
        let old = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let mut stale = old.lease();
        stale.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(16));
        stale.commit_recycling();
        pools.rotate_all();
        drop(stale);
        assert!(old.available.lock().unwrap().is_empty());

        let current = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        assert!(current.registered.load(Ordering::SeqCst));
        let mut lease = current.lease();
        lease.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(16));
        lease.commit_recycling();
        drop(lease);
        assert_eq!(current.available.lock().unwrap().len(), 1);
        let (retained, _, admissions, rejections, _) = budget.snapshot();
        assert!(retained > 0);
        assert_eq!((admissions, rejections), (1, 0));
    }

    #[test]
    fn concurrent_whole_returns_race_for_one_final_budget_slot() {
        let probe = NetworkExecutionWorkspace::<U1FusionRule, f64>::with_test_slot_capacity(32);
        let charge = probe.retained_idle_bytes();
        assert!(charge > 0);
        let budget = Arc::new(WorkspaceBudget::new(charge));
        let pools = WorkspacePools::new(Arc::clone(&budget));
        let pool = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let mut first = pool.lease();
        let mut second = pool.lease();
        first.workspace = Some(probe);
        second.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(32));
        first.commit_recycling();
        second.commit_recycling();
        let barrier = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let second_barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                first_barrier.wait();
                drop(first);
            });
            scope.spawn(move || {
                second_barrier.wait();
                drop(second);
            });
        });
        assert_eq!(pool.available.lock().unwrap().len(), 1);
        let (retained, peak, admissions, rejections, _) = budget.snapshot();
        assert_eq!((retained, peak), (charge, charge));
        assert_eq!((admissions, rejections), (1, 1));
    }

    #[test]
    fn typed_pools_share_one_plan_wide_byte_owner() {
        let f64_probe = NetworkExecutionWorkspace::<U1FusionRule, f64>::with_test_slot_capacity(8);
        let complex_probe =
            NetworkExecutionWorkspace::<U1FusionRule, Complex64>::with_test_slot_capacity(8);
        let f64_charge = f64_probe.retained_idle_bytes();
        let complex_charge = complex_probe.retained_idle_bytes();
        let budget = Arc::new(WorkspaceBudget::new(f64_charge + complex_charge - 1));
        let pools = WorkspacePools::new(Arc::clone(&budget));

        let f64_pool = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let complex_pool = pools.pool::<U1FusionRule, Complex64, Vec<Complex64>>();
        let mut f64_lease = f64_pool.lease();
        let mut complex_lease = complex_pool.lease();
        f64_lease.workspace = Some(f64_probe);
        complex_lease.workspace = Some(complex_probe);
        f64_lease.commit_recycling();
        complex_lease.commit_recycling();
        drop(f64_lease);
        drop(complex_lease);

        assert_eq!(f64_pool.available.lock().unwrap().len(), 1);
        assert!(complex_pool.available.lock().unwrap().is_empty());
        assert_eq!(budget.snapshot().0, f64_charge);
        assert_eq!(budget.snapshot().2, 1);
        assert_eq!(budget.snapshot().3, 1);
    }

    #[test]
    fn lifo_small_return_keeps_one_large_bottom_workspace_charged() {
        let budget = Arc::new(WorkspaceBudget::new(usize::MAX));
        let pools = WorkspacePools::new(Arc::clone(&budget));
        let pool = pools.pool::<U1FusionRule, f64, Vec<f64>>();
        let mut first = pool.lease();
        let mut second = pool.lease();
        first.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(64));
        second.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(64));
        let large = first.workspace.as_ref().unwrap().retained_idle_bytes();
        first.commit_recycling();
        second.commit_recycling();
        drop(first);
        drop(second);

        let mut top = pool.lease();
        top.workspace = Some(NetworkExecutionWorkspace::with_test_slot_capacity(1));
        let small = top.workspace.as_ref().unwrap().retained_idle_bytes();
        top.commit_recycling();
        drop(top);
        assert_eq!(pool.available.lock().unwrap().len(), 2);
        assert_eq!(budget.snapshot().0, large + small);
    }
}
