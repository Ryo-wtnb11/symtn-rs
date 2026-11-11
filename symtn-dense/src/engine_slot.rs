//! One value kept in the caller's CPU engine between calls: an entry of the
//! engine's extension store (`ExecutionEngine::take_extension_caches`), under
//! the store's own entry and retained-byte limits.
//!
//! Internal and unstable despite being public for `symtn`, which keeps its
//! eager Host contraction scratch here.
//!
//! The store is reachable only by taking it out of the engine and restoring
//! it. Every access runs inside one transaction that restores the store on
//! return and on unwind, so a panicking foreign byte reporter or destructor
//! loses only the value being checked in, never the engine's store.
//!
//! The slot is charged `size_of` of its own fixed allocation whenever it
//! exists, plus the held value's reported charge. A check-in is admitted, or
//! the value dropped, before anything in the store is mutated: an admitted
//! insertion evicts no foreign entry as long as foreign byte reports do not
//! change between that preflight and the insertion. The store re-reads every
//! dynamic reporter while it inserts, and offers no insertion that skips its
//! own eviction.
//!
//! Without a CPU engine, or without the `tenferro` feature, nothing is kept:
//! [`checkout`] finds nothing, [`checkin`] drops the value, [`retained_bytes`]
//! is 0 and [`clear`] does nothing.

use tenferro_tensor::Engine;

#[cfg(feature = "tenferro")]
mod store {
    use std::sync::{Mutex, PoisonError};

    use tenferro_cpu::{with_cpu_engine, CpuEngine};
    use tenferro_runtime::{
        ExecutionEngine, ExtensionCacheKey, ExtensionCacheSelector, ExtensionCacheStore,
    };
    use tenferro_tensor::Engine;

    const FAMILY: &str = "symtn.host-scratch.v1";
    const CACHE: &str = "contract";
    const KEY: ExtensionCacheKey = ExtensionCacheKey::new(FAMILY, CACHE, 0);
    const SELECTOR: ExtensionCacheSelector = ExtensionCacheSelector::Cache {
        family_id: FAMILY,
        cache_name: CACHE,
    };

    /// The entry's value. Why a `Mutex`: the store requires `Sync` values,
    /// and only the byte reporter, which gets `&Slot`, ever locks it; every
    /// other access goes through `&mut` (`Mutex::get_mut`).
    pub(super) struct Slot<T> {
        value: Mutex<Option<T>>,
    }

    pub(super) fn slot_charge<T>(value: Option<&T>, charge: fn(&T) -> usize) -> usize {
        std::mem::size_of::<Slot<T>>().saturating_add(value.map_or(0, charge))
    }

    fn held<T>(slot: &mut Slot<T>) -> &mut Option<T> {
        slot.value.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    /// The engine's store, out of the engine until drop.
    struct StoreTxn<'e> {
        cpu: &'e mut CpuEngine,
        store: Option<ExtensionCacheStore>,
    }

    impl Drop for StoreTxn<'_> {
        fn drop(&mut self) {
            if let Some(store) = self.store.take() {
                self.cpu.restore_extension_caches(store);
            }
        }
    }

    /// Runs `f` on the CPU engine's extension store; `None` on any other
    /// engine.
    pub(super) fn with_store<R>(
        engine: &mut dyn Engine,
        f: impl FnOnce(&mut ExtensionCacheStore) -> R,
    ) -> Option<R> {
        with_cpu_engine(engine, |cpu| {
            let store = cpu.take_extension_caches();
            let mut txn = StoreTxn {
                cpu,
                store: Some(store),
            };
            txn.store.as_mut().map(f)
        })
        .flatten()
    }

    pub(super) fn checkout<T: Send + 'static>(store: &mut ExtensionCacheStore) -> Option<T> {
        store
            .get_mut::<Slot<T>>(&KEY)
            .and_then(|slot| held(slot).take())
    }

    pub(super) fn checkin<T: Send + 'static>(
        store: &mut ExtensionCacheStore,
        value: T,
        charge: fn(&T) -> usize,
    ) {
        let ours_now = store
            .get_mut::<Slot<T>>(&KEY)
            .map(|slot| slot_charge(held(slot).as_ref(), charge));
        let foreign = store
            .stats(ExtensionCacheSelector::All)
            .retained_bytes
            .saturating_sub(ours_now.unwrap_or(0));
        let candidate = slot_charge(Some(&value), charge);
        let limits = store.limits();
        let bytes_fit = limits
            .max_retained_bytes()
            .is_none_or(|limit| foreign.saturating_add(candidate) <= limit.get());
        let entries_fit = ours_now.is_some() || store.len() < limits.max_entries().get();
        if !(bytes_fit && entries_fit) {
            return;
        }
        #[cfg(test)]
        super::tests::after_admission();
        match store.get_mut::<Slot<T>>(&KEY) {
            Some(slot) => *held(slot) = Some(value),
            None => store.put_with_retained_bytes(
                KEY,
                Slot {
                    value: Mutex::new(Some(value)),
                },
                move |slot: &Slot<T>| {
                    let value = slot.value.lock().unwrap_or_else(PoisonError::into_inner);
                    slot_charge(value.as_ref(), charge)
                },
            ),
        }
    }

    pub(super) fn retained_bytes(store: &ExtensionCacheStore) -> usize {
        store.stats(SELECTOR).retained_bytes
    }

    pub(super) fn clear(store: &mut ExtensionCacheStore) {
        store.clear_selected(SELECTOR);
    }
}

/// Takes the kept value out of the engine, leaving its slot empty; `None`
/// when nothing is kept.
pub fn checkout<T: Send + 'static>(engine: &mut dyn Engine) -> Option<T> {
    #[cfg(feature = "tenferro")]
    return store::with_store(engine, store::checkout::<T>).flatten();
    #[cfg(not(feature = "tenferro"))]
    {
        let _ = engine;
        None
    }
}

/// Keeps `value`, whose retained bytes are `charge(value)`, in the engine if
/// the store's limits admit it with every other entry; otherwise drops it.
/// An admitted value replaces any value kept before (the last writer wins).
pub fn checkin<T: Send + 'static>(engine: &mut dyn Engine, value: T, charge: fn(&T) -> usize) {
    #[cfg(feature = "tenferro")]
    store::with_store(engine, |s| store::checkin(s, value, charge));
    #[cfg(not(feature = "tenferro"))]
    let _ = (engine, value, charge);
}

/// The slot's charge in the engine's store: 0 when there is no slot, its
/// fixed allocation while it is empty, plus the kept value's charge.
pub fn retained_bytes(engine: &mut dyn Engine) -> usize {
    #[cfg(feature = "tenferro")]
    return store::with_store(engine, |s| store::retained_bytes(s)).unwrap_or(0);
    #[cfg(not(feature = "tenferro"))]
    {
        let _ = engine;
        0
    }
}

/// Removes the slot and drops the kept value.
pub fn clear(engine: &mut dyn Engine) {
    #[cfg(feature = "tenferro")]
    store::with_store(engine, store::clear);
    #[cfg(not(feature = "tenferro"))]
    let _ = engine;
}

#[cfg(all(test, feature = "tenferro"))]
pub(crate) mod tests;
