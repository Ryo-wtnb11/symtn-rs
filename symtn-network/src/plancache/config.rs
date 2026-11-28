//! Configuration value types of the topology-keyed contraction-plan cache
//! (see the parent module): search choice, replan policy, capacity and
//! workspace budget, and the diagnostic counters.
//!
//! Set the configuration after building a [`Runtime`](symtn::typed::Runtime)
//! with [`configure_plan_cache`](super::configure_plan_cache); read it with
//! [`plan_cache_config`](super::plan_cache_config).

/// Which contraction-order search to run: a hashable value type (usable as
/// a cache-key component and as a runtime-wide default) rather than a trait
/// object. `#[non_exhaustive]` so future external searches (e.g. another
/// external-planner adapter carrying its config) slot in without a
/// breaking change.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Optimizer {
    /// Greedy pairwise search ([`GreedyDenseOptimizer`](crate::GreedyDenseOptimizer));
    /// the default.
    #[default]
    Greedy,
}

/// When to re-plan a topology-matched cache entry whose leg dimensions have
/// drifted from the snapshot it was planned under. Reusing is always
/// *correct* (a pairwise order is dimension-independent); re-planning only
/// restores *optimality*.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum ReplanPolicy {
    /// Always reuse the cached order, whatever the current dimensions.
    AlwaysReuse,
    /// Find the order once at real (non-degenerate) dims, then freeze and
    /// reuse it for any later dimensions — the standard "search once, reuse
    /// the path regardless of rank" design (cotengra's reusable
    /// `ContractionTree`, `@tensoropt`'s compile-time bake). Unlike
    /// [`AlwaysReuse`](Self::AlwaysReuse) it *does* replace a plan that was
    /// seeded at degenerate dims (some leg trivial, dim ≤ 1), whose order can
    /// be a poor outer-product-heavy fit for the real state. This is the
    /// default: re-searching a drifted topology (see
    /// [`DriftFactor`](Self::DriftFactor)) buys no measured speedup on
    /// symtn's networks while paying the (χ-dependent) search cost each time.
    #[default]
    BakeOnce,
    /// Re-plan when any leg dimension differs from the snapshot by more
    /// than this factor (as a ratio, in either direction). Chases the
    /// per-shape-optimal order; only worth it when a network's winning order
    /// genuinely flips between dimension regimes.
    ///
    /// The drift is measured from the dimensions the cached order was
    /// searched at, whichever call site reaches it. With plan
    /// persistence enabled ([`load_plan_cache`](super::load_plan_cache)), a
    /// drift-replan of a topology that has a persisted order replays that
    /// order rather than searching a new one.
    DriftFactor(f64),
}

/// Drift ratio for [`ReplanPolicy::DriftFactor`] when that (non-default)
/// policy is selected: re-plan once a leg has grown or shrunk past 2x its
/// planning-time value, on the theory that the cost balance has changed
/// qualitatively. Not the default — see [`ReplanPolicy::BakeOnce`].
pub const DEFAULT_REPLAN_DRIFT_FACTOR: f64 = 2.0;

/// Default maximum number of cached plans (per runtime).
///
/// Rationale: an entry's compiled plan is metadata only (label lists, a step
/// list and a dims snapshot — well under a kilobyte for realistic networks),
/// so 256 covers drivers that cycle through many distinct expressions (e.g.
/// every bond of a large unit cell) without plan-eviction thrash. Idle
/// numerical workspace storage is bounded separately by
/// [`DEFAULT_WORKSPACE_BUDGET_BYTES`].
pub const DEFAULT_PLAN_CACHE_CAPACITY: usize = 256;

/// Default Runtime-wide ceiling for idle network-execution workspace storage.
///
/// The budget covers reusable Host dense payload capacity and workspace-owned
/// dynamic metadata. Checked-out workspaces are request-owned and are not
/// charged. A value of zero disables idle workspace retention; it never means
/// unlimited.
pub const DEFAULT_WORKSPACE_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// Plan-cache behavior; set with [`configure_plan_cache`](super::configure_plan_cache).
#[derive(Clone, Debug)]
pub struct PlanCacheConfig {
    /// Master switch; `false` makes every network contraction plan fresh.
    pub enabled: bool,
    /// Maximum cached entries before LRU eviction.
    pub capacity: usize,
    /// Runtime-wide byte ceiling for idle network execution workspaces.
    ///
    /// Zero disables idle workspace pooling while preserving compiled-plan
    /// caching.
    pub workspace_budget_bytes: usize,
    /// When to re-plan on dimension drift.
    pub replan: ReplanPolicy,
    /// Default optimizer for network contraction (`Network::contract`).
    pub optimizer: Optimizer,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity: DEFAULT_PLAN_CACHE_CAPACITY,
            workspace_budget_bytes: DEFAULT_WORKSPACE_BUDGET_BYTES,
            replan: ReplanPolicy::default(),
            optimizer: Optimizer::default(),
        }
    }
}

/// Counters for tests and diagnostics; see
/// [`plan_cache_stats`](super::plan_cache_stats).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCacheStats {
    /// Topology hits that reused the cached order.
    pub hits: u64,
    /// Topology misses (planned and inserted fresh).
    pub misses: u64,
    /// Topology hits re-planned because dimension drift exceeded the
    /// [`ReplanPolicy`].
    pub replans: u64,
    /// Current number of cached plans.
    pub entries: usize,
    /// Current number of persisted contraction orders (see
    /// [`load_plan_cache`](super::load_plan_cache)), bounded by [`PlanCacheConfig::capacity`].
    pub persisted_orders: usize,
    /// Execution workspaces allocated because no cached lease was available.
    pub workspaces_created: u64,
    /// Execution workspace leases served from the per-plan pool.
    pub workspace_reuses: u64,
    /// Slot-table capacity growth events across cached workspaces.
    pub workspace_slot_grows: u64,
    /// Owned topology entries materialized by [`Network::contract`](crate::Network::contract)
    /// lookups that did not hit (misses and replans).
    pub topology_materializations: u64,
    /// Idle execution workspaces retained by current cached plans.
    pub idle_workspaces: usize,
    /// Current Runtime-owned idle workspace storage.
    pub retained_workspace_bytes: usize,
    /// Maximum admitted idle workspace storage since the last explicit clear.
    pub peak_retained_workspace_bytes: usize,
    /// Whole workspaces admitted to idle pooling by the byte budget.
    pub workspace_byte_admissions: u64,
    /// Returning workspaces rejected because the byte budget could not admit them.
    pub workspace_byte_rejections: u64,
    /// Already-retained workspaces released by cache/configuration eviction.
    pub workspace_byte_evictions: u64,
}
