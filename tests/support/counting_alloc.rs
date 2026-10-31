//! The one counting allocator for allocation-budget tests and probes.
//!
//! Test and bench targets include this file with
//! `#[path = ".../tests/support/counting_alloc.rs"] mod counting_alloc;` and
//! install it with one line:
//! `#[global_allocator] static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;`
//!
//! Counters are thread-local: a measurement sees only the allocations its own
//! thread makes. Why not process-global counters: a test running concurrently
//! in the same binary allocates inside the measured window and breaks the
//! budget (#1824). Budgets that also depend on process-global caches take
//! [`serial`] as well.
//!
//! A consequence: with a multi-threaded runtime (for example
//! `transform_owned_parallel`, the CUDA tests, or the default row of the
//! `eager_overhead_ledger` bench) a measurement gates only the calling
//! thread's allocations, not those made on worker threads.

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ops::RangeInclusive;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Allocator hooks counted on the calling thread while [`measure`] or
/// [`measure_matching`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allocs {
    /// `alloc`, `alloc_zeroed` and `realloc` calls.
    pub calls: u64,
    /// Requested bytes; a `realloc` is charged its full new size, so a
    /// payload-sized buffer grown in place still shows up.
    pub bytes: u64,
    /// `alloc_zeroed` calls, counted apart so a scalar zero fill and an
    /// allocator-zeroed buffer cannot masquerade as each other.
    pub zeroed_calls: u64,
    pub zeroed_bytes: u64,
    /// `dealloc` calls plus `realloc` releases, and their bytes.
    pub frees: u64,
    pub freed_bytes: u64,
    /// Largest `allocated - freed` balance reached, never below zero.
    pub peak_live_bytes: u64,
    /// `allocated - freed` when the measurement ended; negative when it
    /// released memory allocated before it started.
    pub live_bytes: i64,
    /// Allocating calls whose size lies in the range given to
    /// [`measure_matching`], and the `alloc_zeroed` subset of them.
    pub matched_calls: u64,
    pub matched_zeroed_calls: u64,
}

impl Allocs {
    pub const ZERO: Self = Self {
        calls: 0,
        bytes: 0,
        zeroed_calls: 0,
        zeroed_bytes: 0,
        frees: 0,
        freed_bytes: 0,
        peak_live_bytes: 0,
        live_bytes: 0,
        matched_calls: 0,
        matched_zeroed_calls: 0,
    };
}

struct State {
    enabled: Cell<bool>,
    matching: Cell<(usize, usize)>,
    counts: Cell<Allocs>,
}

thread_local! {
    // Const-initialized and without `Drop`: the hooks never allocate here and
    // the state stays readable during thread teardown.
    static STATE: State = const {
        State {
            enabled: Cell::new(false),
            matching: Cell::new(NO_MATCH),
            counts: Cell::new(Allocs::ZERO),
        }
    };
}

/// One hook: the `(size, zeroed)` it requested and the size it released.
fn record(allocated: Option<(usize, bool)>, freed: Option<usize>) {
    let _ = STATE.try_with(|state| {
        if !state.enabled.get() {
            return;
        }
        let mut counts = state.counts.get();
        if let Some((size, zeroed)) = allocated {
            counts.calls += 1;
            counts.bytes += size as u64;
            let (lo, hi) = state.matching.get();
            let matched = (lo..=hi).contains(&size);
            counts.matched_calls += u64::from(matched);
            if zeroed {
                counts.zeroed_calls += 1;
                counts.zeroed_bytes += size as u64;
                counts.matched_zeroed_calls += u64::from(matched);
            }
        }
        if let Some(size) = freed {
            counts.frees += 1;
            counts.freed_bytes += size as u64;
        }
        counts.live_bytes +=
            allocated.map_or(0, |(size, _)| size) as i64 - freed.unwrap_or(0) as i64;
        counts.peak_live_bytes = counts.peak_live_bytes.max(counts.live_bytes.max(0) as u64);
        state.counts.set(counts);
    });
}

pub struct CountingAllocator;

// SAFETY: every hook forwards its arguments unchanged to `System`, and the
// bookkeeping only touches this thread's `Cell`s, which never allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(Some((layout.size(), false)), None);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(Some((layout.size(), true)), None);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        record(None, Some(layout.size()));
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() {
            record(Some((new_size, false)), Some(layout.size()));
        }
        pointer
    }
}

/// Runs `f` and returns its value with the allocations this thread made
/// meanwhile. The value is returned, not dropped, inside the window.
pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Allocs) {
    start();
    let value = f();
    (value, stop())
}

/// [`measure`] that also counts allocating calls whose size is in `sizes`
/// (`n..=n` for an exact payload size, `n..=usize::MAX` for "at least `n`").
pub fn measure_matching<T>(sizes: RangeInclusive<usize>, f: impl FnOnce() -> T) -> (T, Allocs) {
    start_matching((*sizes.start(), *sizes.end()));
    let value = f();
    (value, stop())
}

/// Starts counting this thread's allocations; [`stop`] returns them. For
/// windows that bind values a closure would scope away.
pub fn start() {
    start_matching(NO_MATCH);
}

/// An inclusive `(lo, hi)` size range that no allocation lies in.
const NO_MATCH: (usize, usize) = (1, 0);

fn start_matching(sizes: (usize, usize)) {
    STATE.with(|state| {
        state.matching.set(sizes);
        state.counts.set(Allocs::ZERO);
        state.enabled.set(true);
    });
}

/// Stops counting and returns what was counted since [`start`].
pub fn stop() -> Allocs {
    STATE.with(|state| {
        state.enabled.set(false);
        state.counts.get()
    })
}

/// The no-growth contract of a paired fixture: `large`, the same structure
/// as `small` scaled in the dimension `what` names, costs exactly as many
/// allocation calls.
#[track_caller]
pub fn assert_same_calls(what: &str, small: u64, large: u64) {
    assert_eq!(
        small, large,
        "{what}: {small} allocation calls on the small fixture, {large} on the large one"
    );
}

static SERIAL: Mutex<()> = Mutex::new(());

/// Serializes the tests of one binary whose budgets depend on process-global
/// caches (layouts, tree transforms, intern tables). Poisoning is ignored: a
/// failing budget test must not fail every later test with `PoisonError`.
pub fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}
