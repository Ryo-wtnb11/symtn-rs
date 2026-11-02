//! Process-global structure caches: the one public control of symtn's
//! retained structural data.
//!
//! Six caches, each a pure function of its key and each byte-bounded on its
//! own budget (128 MiB by default), shared by every
//! [`Runtime`](crate::typed::Runtime) of the process:
//!
//! - fusion-tree keys and sector structure per HomSpace sector signature,
//!   [`StructureCacheKind::SectorStructure`] (TensorKit `sectorstructure`);
//! - complete block structure per HomSpace,
//!   [`StructureCacheKind::DegeneracyStructure`] (TensorKit
//!   `degeneracystructure`);
//! - completed permute/braid/transpose transformers,
//!   [`StructureCacheKind::CompletedTreeTransformer`] (TensorKit
//!   `treetransposer` / `treebraider`), and compiled fusion-aware traces,
//!   so a warm trace is one lookup in every fusion style. TensorKit has no
//!   trace counterpart: its `_trace_permute!` looks every fusion block up in
//!   `fsbraid` on each call, and recomposes Unique traces per tree;
//! - composed fusion-tree transformation coefficients per source fusion-tree
//!   group, [`StructureCacheKind::TreeTransformCoefficients`] (TensorKit
//!   `fsbraid` / `fstranspose`). As TensorKit's `NoCache`, Unique fusion
//!   (U(1), Z2, ...) never enters it: a degeneracy-only change recomputes
//!   its one phase per block without allocating per block. A trace keeps its
//!   multiplicity-free Simple or checked Generic group permutations here as
//!   well (present entries only), read when its compiled trace misses;
//! - prepared dense GEMM plans per block geometry (scalar type, extents,
//!   effective strides, conjugation), [`StructureCacheKind::GemmPlans`], on
//!   the native CPU provider. TensorKit has no counterpart: it calls BLAS
//!   `mul!` per block, which keeps no plan;
//! - searched lane allocations of block GEMM batches per schedule input
//!   (budget, batch policy, mean work, team, dispatch costs, per-block
//!   serial times and grant inputs), [`StructureCacheKind::LaneAllocations`],
//!   on the native CPU provider, so a repeated batch on a fresh executor
//!   skips the lane search. Neither TensorKit nor QSpace has a model-based
//!   lane search or a cache of its result.
//!
//! This mirrors TensorKit's `GLOBAL_CACHES` with `empty_globalcaches!` and
//! `global_cache_info` (`caches.jl:1-11` @cfaa073).
//!
//! The composed coefficients are one process-global cache, so several
//! Runtimes working on disjoint transforms share its one budget. Its largest
//! admissible entry is [`StructureCacheInfo::max_entry_bytes`], half its
//! budget: 67,108,864 B at the default budget. Raise its budget
//! with [`configure_budgets`] when a workload needs more.
//!
//! A cache hit is interchangeable with a rebuild, so clearing or sizing these
//! changes cost, never results.

pub use symtn_core::{StructureCacheInfo, StructureCacheKind};

/// Snapshots of every structure cache, in [`StructureCacheKind::ALL`] order.
/// Each is sampled separately, not as one atomic snapshot.
///
/// [`StructureCacheInfo::duplicate_builds`] counts the builds wasted when
/// several threads miss one key at once and the first to publish wins. The
/// [`StructureCacheKind::DegeneracyStructure`],
/// [`StructureCacheKind::CompletedTreeTransformer`] and
/// [`StructureCacheKind::TreeTransformCoefficients`] caches have no
/// single-flight, because their builds run Rayon regions and a thread waiting
/// on another's build could steal work that waits on its own. A concurrent
/// miss resolved by single-flight builds once and reports no duplicate: the
/// [`StructureCacheKind::GemmPlans`] cache's builds run no parallel region,
/// so it has single-flight. The [`StructureCacheKind::LaneAllocations`]
/// cache has none: its lookup borrows the call's inputs, and of racing
/// searches of one batch all but the first published count as duplicates.
pub fn stats() -> Vec<StructureCacheInfo> {
    symtn_core::structure_cache_infos()
}

/// Clears every process-global structure cache (sector, degeneracy,
/// completed tree transformer, composed coefficients, GEMM plans, lane
/// allocations) for all
/// Runtimes.
/// Semantic data only: no device executor, scratch or fusion-symbols symbol cache is
/// touched. Builds in flight across the clear return their result
/// unpublished.
///
/// Live tensors stay valid: their structures and transformers are held by
/// value, and structure ids are never reused.
pub fn clear() {
    symtn_core::clear_structure_caches();
}

/// Sets the byte budget of each listed cache; entries beyond a lowered
/// budget are evicted. Owned by the application: library code never calls it.
/// A budget below 2 bytes disables the cache; otherwise its largest
/// admissible entry is half its budget, rounded down.
///
/// A completed transformer is published only while every block structure it
/// is keyed on is resident in, or was admitted to, the
/// [`StructureCacheKind::DegeneracyStructure`] cache: content that cache
/// rejected (oversize, a budget below 2, or a build that straddled a
/// [`clear`]) gets a fresh identity on every call, so its transformers are
/// rebuilt per call rather than cached. A `DegeneracyStructure` budget below 2
/// therefore also disables completed-transformer caching. It does not
/// disable the [`StructureCacheKind::TreeTransformCoefficients`] cache,
/// whose keys hold sectors and fusion trees only, never block contents.
pub fn configure_budgets(budgets: impl IntoIterator<Item = (StructureCacheKind, u64)>) {
    for (kind, bytes) in budgets {
        symtn_core::set_structure_cache_byte_budget(kind, bytes);
    }
}
