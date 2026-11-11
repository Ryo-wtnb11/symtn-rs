//! Execution scratch that eager Host contractions keep in the caller's CPU
//! engine between calls.
//!
//! An eager Host contraction ([`TensorMap::contract`](crate::typed::TensorMap::contract),
//! `contract_into`, `compose`, the planar contractions, and every step of a
//! `symtn-network` network) takes its execution scratch from the engine it
//! runs on and returns it afterwards: the transformed operands, the
//! permuted-output (`copyC`) temporary and core destination, the irregular
//! core payload, the tree-transform workspaces, the planning scratch and the
//! dense staging, for every payload type the engine has run. A repeated
//! contraction on a warm engine therefore allocates no operand or core
//! scratch, only its owned output.
//!
//! This is execution workspace, not semantic cache: it changes cost, never
//! results. It is separate from the process-global structure caches of
//! [`crate::cache`].
//!
//! # Bound
//!
//! Each buffer keeps the largest size it has needed since the last [`trim`].
//! The scratch is one entry of the engine's extension store and is charged
//! against that store's limits (by default 256 entries and 64 MiB in all).
//! When a contraction returns, its scratch is kept only if its whole ledger
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
//! is 0 and [`trim`] does nothing. `CpuEngine::reset_buffer_pool` and
//! `CpuEngine::clear_runtime_caches` neither count nor release this scratch
//! (tensor4all/tenferro-rs#2587), so release it with [`trim`]. Each call
//! moves the engine's store out and back, which allocates an empty
//! placeholder store each time (tensor4all/tenferro-rs#2587).
//!
//! While a tenferro runtime run has moved the engine's store out, a
//! contraction run inside it finds no scratch and its scratch is dropped
//! with that run's placeholder; neither [`retained_bytes`] nor [`trim`]
//! reaches the store during that run.

use crate::typed::Engine;

/// The bytes of scratch the engine keeps between calls: every kept buffer
/// at capacity, the box that holds its execution context, the boxes of its
/// later payload-type lanes, and the fixed slot that holds them. While a contraction has the scratch checked out, and after a
/// call that failed in planning or whose scratch did not fit, only the fixed
/// slot is reported. It is 0 when no slot exists (never created, or
/// [`trim`]med) and on a non-CPU engine. It is not peak or in-flight memory.
///
/// `&mut` only because the engine's store is reachable solely by moving it
/// out and back.
pub fn retained_bytes(engine: &mut dyn Engine) -> usize {
    symtn_dense::engine_slot::retained_bytes(engine)
}

/// Releases the scratch the engine keeps, with its slot; the next eligible
/// contraction is cold. Scratch a contraction has checked out at that moment
/// is unaffected, and returns under the store's limits. Does nothing on a
/// non-CPU engine.
pub fn trim(engine: &mut dyn Engine) {
    symtn_dense::engine_slot::clear(engine);
}
