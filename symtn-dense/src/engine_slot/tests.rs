use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_runtime::{ExecutionEngine, ExtensionCacheKey, ExtensionCacheLimits};

use super::store::slot_charge;
use crate::CpuEngine;

thread_local! {
    static AFTER_ADMISSION: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
}

/// Runs the hook a test installed, between check-in's admission and its
/// insertion.
pub(crate) fn after_admission() {
    if let Some(hook) = AFTER_ADMISSION.with(|cell| cell.borrow_mut().take()) {
        hook();
    }
}

type Value = Vec<u8>;

fn charge(value: &Value) -> usize {
    value.capacity()
}

fn value(bytes: usize) -> Value {
    Vec::with_capacity(bytes)
}

/// The store charge of a slot holding `value` (of an empty slot for `None`).
fn charged(value: Option<&Value>) -> usize {
    slot_charge(value, charge)
}

const FOREIGN: ExtensionCacheKey = ExtensionCacheKey::new("test.foreign.v1", "entry", 0);

fn set_limits(engine: &mut CpuEngine, limits: ExtensionCacheLimits) {
    let mut store = engine.take_extension_caches();
    store.set_limits(limits);
    engine.restore_extension_caches(store);
}

fn byte_limit(bytes: usize) -> ExtensionCacheLimits {
    ExtensionCacheLimits::default().with_max_retained_bytes(NonZeroUsize::new(bytes).unwrap())
}

fn put_foreign(engine: &mut CpuEngine, bytes: usize) {
    let mut store = engine.take_extension_caches();
    store.put(FOREIGN, (), bytes);
    engine.restore_extension_caches(store);
}

fn has_foreign(engine: &mut CpuEngine) -> bool {
    let mut store = engine.take_extension_caches();
    let present = store.get::<()>(&FOREIGN).is_some();
    engine.restore_extension_caches(store);
    present
}

fn kept_len(engine: &mut CpuEngine) -> Option<usize> {
    let kept = super::checkout::<Value>(engine);
    let len = kept.as_ref().map(Vec::capacity);
    if let Some(kept) = kept {
        super::checkin(engine, kept, charge);
    }
    len
}

#[test]
fn checkout_leaves_an_empty_slot_charged_its_fixed_allocation() {
    let engine = &mut CpuEngine::serial();
    assert_eq!(super::retained_bytes(engine), 0);
    super::checkin(engine, value(64), charge);
    assert_eq!(super::retained_bytes(engine), charged(Some(&value(64))));
    assert_eq!(
        super::checkout::<Value>(engine).map(|v| v.capacity()),
        Some(64)
    );
    assert_eq!(super::retained_bytes(engine), charged(None));
    assert!(super::checkout::<Value>(engine).is_none());
    super::clear(engine);
    assert_eq!(super::retained_bytes(engine), 0);
}

#[test]
fn a_nested_checkin_is_replaced_by_the_outer_one() {
    let engine = &mut CpuEngine::serial();
    super::checkin(engine, value(8), charge);
    let outer = super::checkout::<Value>(engine).unwrap();
    // An inner call finds the slot empty, uses its own value, and checks in
    // first; the outer check-in replaces it.
    assert!(super::checkout::<Value>(engine).is_none());
    super::checkin(engine, value(32), charge);
    super::checkin(engine, outer, charge);
    assert_eq!(kept_len(engine), Some(8));
    assert_eq!(super::retained_bytes(engine), charged(Some(&value(8))));
}

#[test]
fn foreign_reporter_panic_during_checkin_keeps_the_engine_store() {
    let engine = &mut CpuEngine::serial();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let mut store = engine.take_extension_caches();
    // The store reads the report once while inserting it; the next read is
    // check-in's preflight.
    store.put_with_retained_bytes(FOREIGN, (), move |_: &()| {
        if seen.fetch_add(1, Ordering::SeqCst) == 1 {
            panic!("foreign reporter");
        }
        1
    });
    engine.restore_extension_caches(store);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::checkin(engine, value(16), charge);
    }));
    assert!(unwound.is_err());
    assert!(has_foreign(engine));
    super::checkin(engine, value(16), charge);
    assert_eq!(kept_len(engine), Some(16));
}

#[test]
fn checkin_admission_never_evicts_a_foreign_entry() {
    let candidate = charged(Some(&value(64)));

    // (a) A byte budget the foreign entry already fills.
    let engine = &mut CpuEngine::serial();
    set_limits(engine, byte_limit(candidate));
    put_foreign(engine, 1);
    super::checkin(engine, value(64), charge);
    assert!(has_foreign(engine), "a: foreign entry kept");
    assert_eq!(kept_len(engine), None, "a: candidate dropped");

    // (b) Byte room, but no entry left.
    let engine = &mut CpuEngine::serial();
    set_limits(
        engine,
        ExtensionCacheLimits::new(NonZeroUsize::MIN)
            .with_max_retained_bytes(NonZeroUsize::new(candidate * 4).unwrap()),
    );
    put_foreign(engine, 1);
    super::checkin(engine, value(64), charge);
    assert!(has_foreign(engine), "b: foreign entry kept");
    assert_eq!(kept_len(engine), None, "b: candidate dropped");

    // (c) A kept value that grows past the budget.
    let engine = &mut CpuEngine::serial();
    set_limits(engine, byte_limit(candidate + 1));
    put_foreign(engine, 1);
    super::checkin(engine, value(64), charge);
    assert_eq!(kept_len(engine), Some(64), "c: fitting value kept");
    let mut grown = super::checkout::<Value>(engine).unwrap();
    grown.reserve_exact(128);
    super::checkin(engine, grown, charge);
    assert!(has_foreign(engine), "c: foreign entry kept");
    assert_eq!(kept_len(engine), None, "c: grown value dropped");
    assert_eq!(
        super::retained_bytes(engine),
        charged(None),
        "c: empty slot"
    );
}

/// The documented limit of the guarantee: a foreign report that grows
/// between the preflight and the insertion is evicted by the store's own
/// LRU rule, not protected.
#[test]
fn a_foreign_report_changed_after_admission_is_evicted_by_the_store() {
    let engine = &mut CpuEngine::serial();
    let candidate = charged(Some(&value(64)));
    let limit = candidate + 2;
    set_limits(
        engine,
        ExtensionCacheLimits::new(NonZeroUsize::new(4).unwrap())
            .with_max_retained_bytes(NonZeroUsize::new(limit).unwrap()),
    );
    let report = Arc::new(AtomicUsize::new(1));
    let reporter = Arc::clone(&report);
    let mut store = engine.take_extension_caches();
    store.put_with_retained_bytes(FOREIGN, (), move |_: &()| reporter.load(Ordering::SeqCst));
    engine.restore_extension_caches(store);
    let bump = Arc::clone(&report);
    AFTER_ADMISSION.with(|cell| {
        *cell.borrow_mut() = Some(Box::new(move || bump.store(limit, Ordering::SeqCst)));
    });
    super::checkin(engine, value(64), charge);
    assert!(!has_foreign(engine));
    assert_eq!(kept_len(engine), Some(64));
    let mut store = engine.take_extension_caches();
    let stats = store.stats(tenferro_runtime::ExtensionCacheSelector::All);
    assert!(stats.retained_bytes <= limit && stats.entries <= 4);
    let _ = store.get::<()>(&FOREIGN);
    engine.restore_extension_caches(store);
}

/// Whole-value dropping: a value whose charge exceeds the free budget is
/// never kept, so it stays cold on every call until the limit is raised.
#[test]
fn whole_value_dropping_stays_cold_until_the_limit_is_raised() {
    let engine = &mut CpuEngine::serial();
    let (alone, both) = (value(64), value(128));
    set_limits(engine, byte_limit(charged(Some(&alone))));
    super::checkin(engine, alone, charge);
    assert_eq!(kept_len(engine), Some(64));
    for _ in 0..2 {
        let _warm = super::checkout::<Value>(engine);
        super::checkin(engine, value(128), charge);
        assert_eq!(kept_len(engine), None);
    }
    set_limits(engine, byte_limit(charged(Some(&both))));
    super::checkin(engine, both, charge);
    assert_eq!(kept_len(engine), Some(128));
}
