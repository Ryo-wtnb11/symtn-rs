use super::*;
use symtn_dense::Engine;

/// The execution scratch of one operation (tree-transform and contraction
/// lanes): built for a standalone call and dropped when it returns, kept by a
/// prepared workspace for its replays, or, for an eager contraction, checked
/// out of the caller's CPU engine and checked back in ([`crate::scratch`]).
/// It holds only the lanes the operation used (the first inline, any later
/// one boxed; see [`Ctxs`]), structural state and scratch; the operation runs
/// on the caller's engine.
pub(crate) struct ContextLease(TensorExecutionContext);

/// An eager contraction's lease, as the caller's CPU engine keeps it.
///
/// Why boxed: the context is about 9.7 KB inline, and checkout, check-in and
/// every hand-off between them move the lease by value; inline, those moves
/// cost more per small contraction than the warm lease saves
/// (`reviews/symtn-2516-overhead.md`). Why only here: a standalone transform
/// builds its lease per call, where a box would be one more allocation.
pub(crate) type EngineLease = Box<ContextLease>;

// The engine's extension store keeps only `Send` values, and an engine lease
// must stay one pointer.
const _: () = assert!(std::mem::size_of::<Option<EngineLease>>() == std::mem::size_of::<usize>());
const _: fn() = || {
    fn send<T: Send>() {}
    send::<EngineLease>();
};

impl ContextLease {
    pub(crate) fn new() -> Self {
        Self(TensorExecutionContext::new())
    }

    /// The lease the caller's CPU engine keeps from an earlier eager
    /// contraction, else a fresh one (one allocation, its box). The engine
    /// keeps an empty slot meanwhile, so a nested call builds its own.
    pub(crate) fn checkout(engine: &mut dyn Engine) -> EngineLease {
        symtn_dense::engine_slot::checkout::<EngineLease>(engine)
            .unwrap_or_else(|| Box::new(Self::new()))
    }

    /// Returns this lease to the caller's CPU engine, which keeps it when its
    /// whole ledger, its box included, fits the engine's extension-store
    /// limits next to every other entry, and drops it otherwise.
    pub(crate) fn checkin(self: EngineLease, engine: &mut dyn Engine) {
        symtn_dense::engine_slot::checkin(engine, self, Self::engine_charge);
    }

    #[allow(
        clippy::borrowed_box,
        reason = "the engine slot charges the value it keeps, which is the box"
    )]
    fn engine_charge(lease: &EngineLease) -> usize {
        std::mem::size_of::<Self>().saturating_add(lease.retained_bytes())
    }

    /// Every buffer this lease retains: each built lane's ledger and the
    /// boxes of boxed lanes.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.0.retained_bytes()
    }

    /// The context for one call.
    pub(crate) fn context(&mut self) -> &mut TensorExecutionContext {
        &mut self.0
    }

    /// Bytes of call scratch the context's dense executors keep.
    pub(crate) fn dense_scratch_bytes(&self) -> usize {
        self.0.dense_scratch_bytes()
    }
}

/// The dense executor one factorization owns for its duration. In unit
/// tests it may be one the probe injected, which returns to the probe on drop.
pub(crate) struct DenseLease {
    pub(crate) executor: Option<Box<dyn symtn_dense::DenseExecutor + Send>>,
    #[cfg(test)]
    pub(super) injected: bool,
}

impl DenseLease {
    pub(crate) fn dense(&mut self) -> &mut (dyn symtn_dense::DenseExecutor + Send) {
        &mut **self
            .executor
            .as_mut()
            .expect("dense lease always owns an executor")
    }
}

#[cfg(test)]
impl Drop for DenseLease {
    fn drop(&mut self) {
        if self.injected && !std::thread::panicking() {
            if let Some(executor) = self.executor.take() {
                crate::typed::probe::return_injected_dense(executor);
            }
        }
    }
}
