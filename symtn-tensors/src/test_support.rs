//! Test-only synchronization for symtn-tensors' process-global caches
//! (the operation-cache registry, its chained `symtn-core` intern tables,
//! and the scratch-space structure cache).
//!
//! Why-not (the alternatives this replaces):
//! - Per-test-file `#[serial]` (a crate dependency) serializes every test in
//!   a module even though only a handful touch process-global state — most
//!   of a file's tests would pay a needless throughput tax.
//! - Making the caches test-scoped (thread-local, or reset before/after each
//!   test) would stop testing what the process-global design actually does
//!   in production (one process, many callers sharing one cache) and would
//!   hide exactly the "a concurrent reset lands between two reads" bugs this
//!   suite exists to catch.
//!
//! So: one process-wide `Mutex`, taken by every test that either mutates
//! shared cache state (`symtn_core::clear_structure_caches`, LRU-cap floods) or
//! asserts on it (`Arc::ptr_eq` of cached values, intern-table lengths/ids).
//! Both species must serialize against each other, not just against their
//! own kind — a reader racing an unlocked resetter is exactly the bug class
//! this lock exists to close (see #169, #172).
//!
//! Poison-tolerant: a panicking test must not poison the mutex and cascade
//! spurious failures onto every other test sharing it, so callers use
//! `.lock().unwrap_or_else(|e| e.into_inner())` rather than `.unwrap()`.
#[cfg(test)]
pub(crate) static CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Re-executes exactly one test, alone, in a child process, when it requires
/// cache residency or an absolute snapshot of `symtn-core`'s process-global
/// cache or intern-table counters. `CACHE_TEST_LOCK` only serializes the small
/// set of tests that take it; ordinary, unlocked construction elsewhere in
/// the same binary's cached path can still evict entries or move counters
/// between this test's own reads (#649/#650 first used this for
/// `checked_bind_failure_preserves_subset_admission_and_caches`).
///
/// Call at the top of the `#[test]` fn with a name-unique env var and the
/// test's libtest path; when it returns `true`, the child already ran the
/// real body and the caller must return immediately.
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
