//! Execution scratch and movement calibration kept in the caller's CPU engine
//! between calls.
//!
//! An eager Host contraction ([`TensorMap::contract`](crate::typed::TensorMap::contract),
//! `contract_into`, `compose`, the planar contractions, and every step of a
//! `symtn-network` network) takes its execution scratch from the engine it
//! runs on and returns it afterwards: the transformed operands, the
//! permuted-output (`copyC`) temporary and core destination, the
//! tree-transform workspaces, the planning scratch and the
//! dense staging, for every payload type the engine has run. A repeated
//! contraction on a warm engine therefore allocates no operand or core
//! scratch, only its owned output.
//! Multithreaded Host transform replays use the same retained context for
//! pack buffers, converted coefficients, dense staging and movement widths.
//! Serial transforms keep their replay scratch local to the call; transform
//! staging remains local at every width.
//!
//! Movement width selection keeps its calibration result, including a failed
//! attempt, in a separate typed slot. [`trim`] releases both scratch and
//! calibration, so the next eligible movement phase calibrates again.
//!
//! This is execution workspace, not semantic cache: it changes cost, never
//! results. It is separate from the process-global structure caches of
//! [`crate::cache`].
//!
//! # Bound
//!
//! Each buffer keeps the largest size it has needed since the last [`trim`].
//! The scratch and movement calibration occupy separate entries of the engine's
//! extension store and are charged against that store's limits (by default 256
//! entries and 64 MiB in all).
//! When a contraction or multithreaded Host transform replay returns, its
//! scratch is kept only if its whole ledger
//! fits those limits next to every other entry; otherwise it is dropped, and
//! that workload stays correct and cold (it allocates its scratch on every
//! call). A warm repeated eager contraction allocates no operand or core
//! scratch **when the engine's retained ledger, including that call's
//! payload type, fits the extension store's limits**.
//!
//! Admission never evicts another owner's entry while the other entries'
//! byte reports stay unchanged during the check-in. A store entry whose
//! report grows later is governed by the store's own least-recently-used
//! eviction, which may evict this scratch.
//!
//! A known larger working set is made eligible by raising the store's byte
//! limit through tenferro's own procedure:
//!
//! ```
//! use std::num::NonZeroUsize;
//! use symtn::typed::CpuEngine;
//! use tenferro_runtime::ExecutionEngine;
//!
//! let mut engine = CpuEngine::serial();
//! let bytes = 256 << 20;
//! let mut store = engine.take_extension_caches();
//! store.set_limits(store.limits().with_max_retained_bytes(NonZeroUsize::new(bytes).unwrap()));
//! engine.restore_extension_caches(store);
//! assert_eq!(symtn::scratch::retained_bytes(&mut engine), 0);
//! ```
//!
//! # Scope
//!
//! Only the CPU engine keeps scratch; on any other engine [`retained_bytes`]
//! is 0 and [`trim`] does nothing. `CpuEngine::reset_buffer_pool` leaves these
//! entries intact; `CpuEngine::clear_runtime_caches` releases the entire
//! extension store, including scratch and calibration. Slot access moves the
//! store out and restores it before entering numerical execution; taking a
//! store already lent to another runtime violates the engine's borrow contract.

use crate::typed::Engine;

/// The bytes of scratch and movement calibration the engine keeps between calls:
/// every kept buffer
/// at capacity, the box that holds its execution context, the boxes of its
/// later payload-type lanes, calibration constants, and the fixed slots that
/// hold them. A checked-out slot reports only its fixed allocation; other
/// slots still report their kept values. It is 0 when no slot exists (never created, or
/// [`trim`]med) and on a non-CPU engine. It is not peak or in-flight memory.
///
/// `&mut` only because the engine's store is reachable solely by moving it
/// out and back.
pub fn retained_bytes(engine: &mut dyn Engine) -> usize {
    symtn_dense::engine_slot::retained_bytes(engine)
}

/// Releases retained scratch and movement calibration with their slots; the
/// next eligible contraction or transform replay is cold and the next eligible movement phase
/// calibrates again. Checked-out values are unaffected and return under the
/// store's limits. Does nothing on a non-CPU engine.
pub fn trim(engine: &mut dyn Engine) {
    symtn_dense::engine_slot::clear(engine);
}
