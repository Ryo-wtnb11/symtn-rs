use super::Engine;

/// The engine a numerical call borrows.
///
/// Every engine is an execution context, so `&mut CpuEngine` and
/// `&mut dyn Engine` arguments both work. The context is borrowed for one
/// call: symtn does not retain the engine. An eager Host contraction leaves
/// its execution scratch in a CPU engine's extension store for the next call
/// on that engine ([`crate::scratch`]).
pub trait ExecContext {
    /// The engine the call runs on.
    fn engine(&mut self) -> &mut dyn Engine;
}

impl<E: Engine> ExecContext for E {
    fn engine(&mut self) -> &mut dyn Engine {
        self
    }
}

// Why one impl per auto-trait combination: an erased engine that keeps a
// `Send` or `Sync` bound is a distinct object type, which a call taking
// `&mut dyn Engine` accepted by unsizing; the generic carrier argument needs
// each object type to be a context itself.
macro_rules! erased_engine_context {
    ($($object:ty),+ $(,)?) => {$(
        impl ExecContext for $object {
            fn engine(&mut self) -> &mut dyn Engine {
                self
            }
        }
    )+};
}

erased_engine_context!(
    dyn Engine + '_,
    dyn Engine + Send + '_,
    dyn Engine + Sync + '_,
    dyn Engine + Send + Sync + '_,
);
