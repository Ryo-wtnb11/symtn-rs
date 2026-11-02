use super::{Engine, Error};

/// The execution options of one numerical call, beside its engine.
///
/// A plain value: every call reads the options its [`ExecContext`] supplies
/// and applies them to that call only, including the executors of a reused
/// prepared workspace. The default selects symtn's default execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExecOptions {
    lane_allocation: LaneAllocation,
}

impl ExecOptions {
    /// These options with `allocation` as the block scheduler's lane
    /// allocation (see [`LaneAllocation`]).
    pub fn with_lane_allocation(mut self, allocation: LaneAllocation) -> Self {
        self.lane_allocation = allocation;
        self
    }

    /// The selected lane allocation.
    pub(crate) fn lane_allocation(&self) -> LaneAllocation {
        self.lane_allocation
    }
}

/// How a call's grouped dense blocks (contraction and Multi recoupling GEMM
/// batches, and the searched factorization batches) are placed on the
/// engine's lanes.
///
/// The selection applies on the native provider's CPU engine under the
/// engine's `Auto` batch policy, the only configuration that searches lane
/// allocations. There, a call holding no batch to search (one block, or one
/// slot) runs as it does by default. A non-default selection is refused with
/// a typed dense `Unsupported` error before the call writes, uploads or
/// submits anything when a phase of the call that would search (a
/// contraction, a tree transform, or an SVD, Hermitian eigen or general
/// eigenvalue batch) runs on an engine or provider without the allocator (a
/// CUDA engine, a BLAS provider build) or under a forced batch policy, which
/// fixes the placement itself. Operations without such a phase (norms, linear
/// combinations, QR and LQ, solves, general eigendecompositions, the prepared
/// batched Hermitian eigensolver, transfers) ignore it.
///
/// One complete configuration (provider, engine width, batch policy and work
/// model, lane allocation and inputs) repeats its bits; results of different
/// allocations agree within rounding.
pub use symtn_dense::LaneAllocation;

/// The execution context of a numerical call: the caller's engine and the
/// call's [`ExecOptions`].
///
/// Every engine is an execution context with the default options, so
/// `&mut CpuEngine` and `&mut dyn Engine` arguments run as they always do;
/// [`Configured`] pairs an engine with other options. The context is
/// borrowed for one call: symtn retains neither the engine nor its options.
pub trait ExecContext {
    /// The engine the call runs on and the call's options.
    fn parts(&mut self) -> (&mut dyn Engine, ExecOptions);
}

impl<E: Engine> ExecContext for E {
    fn parts(&mut self) -> (&mut dyn Engine, ExecOptions) {
        (self, ExecOptions::default())
    }
}

impl ExecContext for dyn Engine + '_ {
    fn parts(&mut self) -> (&mut dyn Engine, ExecOptions) {
        (self, ExecOptions::default())
    }
}

/// An engine owned together with the [`ExecOptions`] of the calls it runs.
///
/// Pass `&mut Configured<_>` wherever an operation takes its engine; it is
/// an [`ExecContext`], not an engine, so a direct tenferro call takes
/// [`engine_mut`](Self::engine_mut) instead.
///
/// ```
/// use symtn::typed::{Configured, CpuEngine, LaneAllocation};
///
/// let mut cx = Configured::new(CpuEngine::serial())
///     .with_lane_allocation(LaneAllocation::EqualShare);
/// let engine: &mut CpuEngine = cx.engine_mut();
/// # let _ = engine;
/// ```
#[derive(Debug)]
pub struct Configured<E: ?Sized = dyn Engine> {
    options: ExecOptions,
    engine: E,
}

impl<E> Configured<E> {
    /// `engine` with the default options.
    pub fn new(engine: E) -> Self {
        Self {
            options: ExecOptions::default(),
            engine,
        }
    }

    /// This context with `allocation` as its calls' lane allocation.
    pub fn with_lane_allocation(mut self, allocation: LaneAllocation) -> Self {
        self.options = self.options.with_lane_allocation(allocation);
        self
    }
}

impl<E: ?Sized> Configured<E> {
    /// The owned engine, for calls outside symtn.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }
}

impl<E: Engine> ExecContext for Configured<E> {
    fn parts(&mut self) -> (&mut dyn Engine, ExecOptions) {
        (&mut self.engine, self.options)
    }
}

impl ExecContext for Configured<dyn Engine + '_> {
    fn parts(&mut self) -> (&mut dyn Engine, ExecOptions) {
        (&mut self.engine, self.options)
    }
}

/// One call's engine and options, borrowed: the context a call body hands
/// to the public operations it composes, so they run with the caller's
/// options rather than an engine's defaults.
pub(crate) struct Invocation<'a> {
    engine: &'a mut dyn Engine,
    options: ExecOptions,
}

impl<'a> Invocation<'a> {
    pub(crate) fn new(engine: &'a mut dyn Engine, options: ExecOptions) -> Self {
        Self { engine, options }
    }
}

impl ExecContext for Invocation<'_> {
    fn parts(&mut self) -> (&mut dyn Engine, ExecOptions) {
        (&mut *self.engine, self.options)
    }
}

impl ExecOptions {
    /// Admits these options on `engine` for a call with a phase that
    /// searches lane allocations, before the call has any effect.
    pub(crate) fn admit(self, engine: &mut dyn Engine) -> Result<(), Error> {
        symtn_dense::admit_lane_allocation(engine, self.lane_allocation)
            .map_err(|error| Error::Operation(Box::new(super::OperationError::Dense(error))))
    }
}
