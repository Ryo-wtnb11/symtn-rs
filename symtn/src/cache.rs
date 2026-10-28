//! Process-global structure caches: the one public control of symtn's
//! retained structural data.
//!
//! Four caches, each a pure function of its key and each byte-bounded on its
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
//!   `treetransposer` / `treebraider`);
//! - composed fusion-tree transformation coefficients per source fusion-tree
//!   group, [`StructureCacheKind::TreeTransformCoefficients`] (TensorKit
//!   `fsbraid` / `fstranspose`). As TensorKit's `NoCache`, Unique fusion
//!   (U(1), Z2, ...) never enters it: a degeneracy-only change recomputes
//!   its one phase per block without allocating per block. A trace keeps its
//!   multiplicity-free Simple or checked Generic group permutations here as
//!   well (present entries only); Unique trace stays uncached, as in
//!   TensorKit.
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
pub fn stats() -> Vec<StructureCacheInfo> {
    symtn_core::structure_cache_infos()
}

/// Clears every process-global structure cache (sector, degeneracy,
/// completed tree transformer, composed coefficients) for all Runtimes.
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
