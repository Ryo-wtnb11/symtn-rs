use super::*;

/// Test-only synchronization for symtn-core's process-global intern tables
/// (block-structure content/arc tables, the hom-space intern table).
///
/// Why-not (the alternatives this replaces): a per-test-file `#[serial]`
/// dependency would serialize this whole (large) test module for the sake of
/// a handful of tests; making the tables test-scoped (thread-local, or wiped
/// between tests) would stop exercising the process-global design these
/// tables actually ship with, and would hide the exact "concurrent
/// reset/flood lands between two reads" bugs this suite exists to catch.
///
/// So: one process-wide `Mutex`, taken by every test that either mutates
/// shared intern-table state (`clear_structure_caches`, LRU-cap floods) or
/// asserts on it (`Arc::ptr_eq` of interned values, table lengths, content
/// ids). Poison-tolerant: a panicking test must not cascade spurious
/// failures onto every other test sharing the lock.
pub(crate) mod test_support {
    pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Re-executes exactly one test, alone, in a child process, for a test
    /// whose assertion cannot be phrased as a delta CACHE_TEST_LOCK-holding
    /// siblings cannot move: an absolute snapshot of a process-global cache
    /// counter (for example `admissions() == 1` read right after a reset).
    /// `CACHE_TEST_LOCK` only serializes the small set of tests that take
    /// it; any ordinary, unlocked construction elsewhere in the same
    /// binary's cached path can still move that same counter between this
    /// test's own reads (the same technique `checked_bind_failure_preserves_subset_admission_and_caches`
    /// in symtn-tensors uses).
    ///
    /// Call at the top of the `#[test]` fn with a name-unique env var and
    /// the test's libtest path; when it returns `true`, the child already
    /// ran the real body and the caller must return immediately.
    pub(crate) fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
        if std::env::var_os(isolated_env).is_some() {
            return false;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_path, "--include-ignored"])
            .env(isolated_env, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        true
    }
}

// Shared fixtures and helper fusion rules used across more than one of
// the per-responsibility test files below. They live in `support`
// and are glob-imported here, so every sibling test module sees them through
// its own `use super::*;`.
mod support;
use support::*;

// The one counting allocator of the allocation tests, shared with the
// integration targets. Its counters are thread-local, so concurrently
// running unit tests do not see each other's allocations.
// SAFETY: the allowed `unsafe` is only that file's `GlobalAlloc` impl, which
// forwards every call unchanged to `System` (see its own SAFETY comment).
#[path = "../../tests/support/counting_alloc.rs"]
#[allow(unsafe_code)]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod block_structure;
mod block_view;
mod checked_rank1_tree_admission;
mod error;
mod external_sector_lookup;
mod fusion_space;
mod fusion_tree;
mod generic_symbol_shape_mismatch;
mod sector;
mod storage;
mod trusted_dual_map;
