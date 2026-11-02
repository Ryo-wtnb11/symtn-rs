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
/// slot) runs as it does by default. Participation is decided per phase from
/// the call's plan before it writes: a grouped GEMM batch (not one GEMM or one
/// strided run), Multi recoupling in a tree transform, and an SVD, Hermitian
/// eigen or general-eigenvalue batch of two or more blocks participate. A
/// participating phase refuses a non-default selection with a typed dense
/// `Unsupported` error before the call writes, uploads or submits anything on
/// a provider without the allocator (a BLAS build) or under a forced batch
/// policy, which fixes the placement itself; a CUDA engine, which has no
/// allocator, refuses it for every contraction, transform and compact
/// factorization. Phases without a search (one GEMM or strided run, a
/// transform without Multi recoupling, norms, linear combinations, QR and LQ,
/// solves, general eigendecompositions, the prepared batched Hermitian
/// eigensolver, transfers) ignore it.
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

// Why one impl per auto-trait combination: an erased engine that keeps a
// `Send` or `Sync` bound is a distinct object type, which a call taking
// `&mut dyn Engine` accepted by unsizing; the generic carrier argument needs
// each object type to be a context itself.
macro_rules! erased_engine_context {
    ($($object:ty),+ $(,)?) => {$(
        impl ExecContext for $object {
            fn parts(&mut self) -> (&mut dyn Engine, ExecOptions) {
                (self, ExecOptions::default())
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
    /// Admits these options on `engine` before the call has any effect: a
    /// non-default lane allocation needs a CPU engine, and, where the call's
    /// plan has a phase the allocation selects for (`participates`), the
    /// native provider under an `Auto` batch policy.
    /// `participates` is evaluated only for a non-default allocation, so the
    /// default never pays for deciding participation.
    pub(crate) fn admit(
        self,
        engine: &mut dyn Engine,
        participates: impl FnOnce() -> bool,
    ) -> Result<(), Error> {
        if self.lane_allocation == LaneAllocation::Moldable {
            return Ok(());
        }
        symtn_dense::admit_lane_allocation(engine, self.lane_allocation, participates())
            .map_err(|error| Error::Operation(Box::new(super::OperationError::Dense(error))))
    }
}
