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
    /// Exhaustive optimal search (opt_einsum `"optimal"`; small networks
    /// only). Requires the `opt-path` feature.
    #[cfg(feature = "opt-path")]
    Optimal,
    /// Dynamic-programming search (opt_einsum `"dp"`): the SAME optimal
    /// pairwise order as `Optimal` for the small networks symtn contracts,
    /// but polynomial-time instead of exhaustive `O(n!)`. This is the
    /// `@tensoropt` analog — optimal order without the branch-and-bound
    /// search cost that dominates the first (cold) contraction of each
    /// topology. Requires the `opt-path` feature.
    #[cfg(feature = "opt-path")]
    DynamicProgramming,
    /// opt_einsum `"auto-hq"`
    /// with fallback to `"auto"`, then `"dp"`, then greedy when a driver
    /// errors (upstream `opt-einsum-path` rejects some all-dim-1 networks).
    /// Near-optimal orders for the large gram / environment-body networks
    /// where plain greedy picks memory-exploding orders. Requires
    /// the `opt-path` feature.
    #[cfg(feature = "opt-path")]
    AutoHq,
    /// External Python `cotengra` path search. Requires the
    /// `cotengra-python` feature and an importable Python
    /// `cotengra` installation. This is intentionally a cold-path planner:
    /// the returned pairwise order is cached and warm execution stays in Rust.
    #[cfg(feature = "cotengra-python")]
    CotengraPython(CotengraPythonConfig),
}

/// Search family used by the Python `cotengra` backend.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CotengraPythonMethod {
    /// `cotengra` preset string `"auto"`.
    Auto,
    /// `cotengra` preset string `"auto-hq"`.
    #[default]
    AutoHq,
    /// `cotengra.GreedyOptimizer`.
    Greedy,
    /// `cotengra.OptimalOptimizer`.
    Optimal,
    /// `cotengra.RandomGreedyOptimizer`.
    RandomGreedy,
    /// `cotengra.HyperOptimizer`.
    Hyper,
}

/// Objective string passed to `cotengra` optimizers that accept `minimize`.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CotengraMinimize {
    /// Minimize estimated contraction FLOPs.
    #[default]
    Flops,
    /// Minimize largest intermediate tensor.
    Size,
    /// Minimize total tensor write volume.
    Write,
    /// Minimize cotengra's combined FLOPs/write objective.
    Combo,
    /// Minimize cotengra's memory-limit-oriented combined objective.
    Limit,
    /// Pass a custom objective name through unchanged.
    Custom(String),
}

/// Configuration for the optional Python `cotengra` planner.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CotengraPythonConfig {
    /// Which cotengra optimizer family to instantiate.
    pub method: CotengraPythonMethod,
    /// Objective for optimizers that expose `minimize`.
    pub minimize: CotengraMinimize,
    /// Trial count for `RandomGreedyOptimizer` and `HyperOptimizer`.
    pub max_repeats: usize,
    /// Optional RNG seed for randomized optimizers.
    pub seed: Option<u64>,
    /// Whether cotengra may parallelize the planner. Defaults to `false` so
    /// path search is reproducible and does not compete with symtn execution.
    pub parallel: bool,
    /// Optional slicing policy. `None` keeps this as a path-only planner.
    pub slicing: CotengraSlicingConfig,
    /// Python executable (default `python3` on `PATH`). symtn reads no
    /// environment variable to choose it; an application that wants an
    /// environment override reads it and calls [`Self::python`] or
    /// [`Self::uv_project`].
    pub python: String,
    /// Arguments inserted between the Python executable and `-c <planner>`.
    /// This supports launchers such as `uv run --project <dir> python`.
    pub python_args: Vec<String>,
    /// Maximum wall-clock time for writing the request and waiting for the
    /// planner process group. `None` explicitly disables the deadline.
    pub timeout: Option<std::time::Duration>,
}

/// Optional post-processing to ask cotengra to slice a searched tree.
///
/// This only affects explicit sliced-plan searches. Normal
/// `Optimizer::CotengraPython` contraction planning remains path-only and
/// ignores slicing, because symtn's ordinary executor cannot consume slices.
#[cfg(feature = "cotengra-python")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CotengraSlicingConfig {
    /// Do not slice.
    #[default]
    None,
    /// Call `ContractionTree.slice`.
    Slice {
        /// Target maximum intermediate size in scalar elements.
        target_size: usize,
        /// Number of repeated slicing searches.
        max_repeats: usize,
        /// Whether cotengra may slice output/open indices.
        allow_outer: bool,
    },
    /// Call `ContractionTree.slice_and_reconfigure`.
    Reconfigure {
        /// Target maximum intermediate size in scalar elements.
        target_size: usize,
        /// Minimum slicing progress before each reconfiguration round.
        step_size: usize,
        /// Number of repeated slicing searches per round.
        max_repeats: usize,
        /// Whether cotengra may slice output/open indices.
        allow_outer: bool,
        /// Use cotengra's forested subtree reconfiguration option.
        forested: bool,
    },
    /// Call `ContractionTree.slice_and_reconfigure_forest`.
    ForestReconfigure {
        /// Target maximum intermediate size in scalar elements.
        target_size: usize,
        /// Size reduction factor per forest round.
        step_size: usize,
        /// Number of candidate trees in the forest.
        num_trees: usize,
        /// Number of repeated slicing searches per candidate.
        max_repeats: usize,
        /// Whether cotengra may slice output/open indices.
        allow_outer: bool,
    },
}

#[cfg(feature = "cotengra-python")]
impl Default for CotengraPythonConfig {
    fn default() -> Self {
        Self {
            method: CotengraPythonMethod::AutoHq,
            minimize: CotengraMinimize::Flops,
            max_repeats: 128,
            seed: Some(0),
            parallel: false,
            slicing: CotengraSlicingConfig::None,
            python: "python3".to_string(),
            python_args: Vec::new(),
            timeout: Some(std::time::Duration::from_secs(300)),
        }
    }
}

#[cfg(feature = "cotengra-python")]
impl CotengraPythonConfig {
    /// Default config launched through `uv run --project <project> python`.
    pub fn with_uv_project(project: impl Into<String>) -> Self {
        Self::default().uv_project(project)
    }

    /// Launch this config through `uv run --project <project> python`.
    /// `project` is stored verbatim and resolved by `uv` against the process
    /// working directory at launch.
    pub fn uv_project(mut self, project: impl Into<String>) -> Self {
        self.python = "uv".to_string();
        self.python_args = vec![
            "run".to_string(),
            "--project".to_string(),
            project.into(),
            "python".to_string(),
        ];
        self
    }

    /// Launch this config through a specific Python executable.
    pub fn python(mut self, python: impl Into<String>) -> Self {
        self.python = python.into();
        self.python_args.clear();
        self
    }

    /// Set the planner deadline.
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Explicitly disable the planner deadline.
    pub fn without_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }
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

#[cfg(all(test, feature = "cotengra-python"))]
mod tests {
    use std::time::Duration;

    use super::CotengraPythonConfig;

    #[test]
    fn cotengra_timeout_defaults_to_five_minutes_and_without_timeout_clears_it() {
        let default = CotengraPythonConfig::default();
        let unbounded = default.clone().without_timeout();

        assert_eq!(default.timeout, Some(Duration::from_secs(300)));
        assert_ne!(default, unbounded);
    }

    #[test]
    fn uv_project_is_stored_verbatim_independent_of_cwd_and_build_location() {
        // Plan-cache identity hashes `python_args`; a build-location- or
        // CWD-dependent rewrite would make equal configs compare unequal.
        let config = CotengraPythonConfig::with_uv_project("tools/cotengra-python");
        assert_eq!(config.python, "uv");
        assert_eq!(
            config.python_args,
            ["run", "--project", "tools/cotengra-python", "python"]
        );
    }
}
