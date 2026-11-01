//! Test probes of the facade (#2221): one thread-local ledger of [`Event`]
//! counts and [`Gauge`] peaks, read through [`record`], plus the
//! [`RoutePolicy`] that forces the branches no public geometry reaches and
//! the armed [`Fault`].
//!
//! Only [`RoutePolicy`] exists outside tests: there it is the constant
//! production policy, so a forced branch folds away and a release build
//! carries no probe state.

/// Test-only overrides of an execution route. Production code reads
/// [`RoutePolicy::current`] once per operation and never branches on
/// `cfg(test)`; outside tests every field is `false`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoutePolicy {
    /// Every eigh route (eager CUDA and the batched Host plan) takes the
    /// per-tree assembly, so the branch aligned fixtures never take runs.
    pub(crate) eigh_treewise: bool,
    /// Every compact-SVD device route is treated as non-aligned.
    pub(crate) svd_treewise: bool,
    /// Device `eigh_full` assembles through the selector GEMM even when every
    /// order is the identity.
    pub(crate) eigh_force_selector: bool,
    /// Device `eigh_full` reverses every odd-indexed sector's order, so the
    /// selector assembly runs a non-identity permutation.
    pub(crate) eigh_reverse_odd_sectors: bool,
}

impl RoutePolicy {
    /// The production policy: no override.
    pub(crate) const DEFAULT: Self = Self {
        eigh_treewise: false,
        svd_treewise: false,
        eigh_force_selector: false,
        eigh_reverse_odd_sectors: false,
    };

    /// The policy of the calling thread: the test's, or the production one.
    #[inline(always)]
    pub(crate) fn current() -> Self {
        #[cfg(test)]
        return ledger::PROBE.with(|probe| probe.borrow().routes);
        #[cfg(not(test))]
        Self::DEFAULT
    }
}

#[cfg(test)]
pub(crate) use ledger::*;

#[cfg(test)]
mod ledger {
    use std::cell::RefCell;

    use super::RoutePolicy;

    /// One observed occurrence in facade execution. Each fact has one event.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Event {
        /// A multiplicity-free contraction entered the tensor contract seam.
        ContractSeamCall,
        /// A multiplicity-free tree transform entered its seam.
        TreeTransformSeamCall,
        /// A checked tensor product published its destination.
        TensorProductCommit,
        /// A dense executor lease was taken.
        DenseLease,
        /// A Host contraction replayed a `copyC` route (contract into a
        /// temporary, then one transform into the destination).
        CopyCReplay,
        /// A lazy adjoint's payload was written out as an owned payload.
        AdjointMaterialization,
        /// A compact diagonal payload was densified for one operation.
        DiagonalMaterialization,
        /// A device reduction downloaded its partials.
        #[cfg(feature = "cuda")]
        CudaReductionDownload,
        /// Device-side partial elements of those downloads.
        #[cfg(feature = "cuda")]
        CudaReductionDevicePartials,
        /// Host-side partial elements those downloads returned.
        #[cfg(feature = "cuda")]
        CudaReductionHostPartials,
        /// A device arithmetic output was zero-initialized by upload.
        #[cfg(feature = "cuda")]
        CudaZeroUpload,
        /// A device arithmetic kernel ran on a non-empty region.
        #[cfg(feature = "cuda")]
        CudaArithmeticKernel,
        /// A device QR decomposition ran on one region.
        #[cfg(feature = "cuda")]
        CudaQrDecomposition,
        /// A whole or per-tree factor copy on a device assembly route.
        #[cfg(feature = "cuda")]
        CudaFactorCopy,
        /// A host-built assembly selector was uploaded.
        #[cfg(feature = "cuda")]
        CudaSelectorUpload,
        /// A device QR output was zero-initialized by upload.
        #[cfg(feature = "cuda")]
        CudaQrOutputUpload,
        /// A device factor assembly GEMM ran.
        #[cfg(feature = "cuda")]
        CudaAssemblyGemm,
        /// A device SVD decomposition returned on one region.
        #[cfg(feature = "cuda")]
        CudaSvdDecomposition,
        /// Singular values those decompositions returned.
        #[cfg(feature = "cuda")]
        CudaSvdSpectrumScalar,
        /// A device SVD output was zero-initialized by upload.
        #[cfg(feature = "cuda")]
        CudaSvdOutputUpload,
    }

    const EVENTS: usize = 32;
    #[cfg(not(feature = "cuda"))]
    const _: () = assert!((Event::DiagonalMaterialization as usize) < EVENTS);
    #[cfg(feature = "cuda")]
    const _: () = assert!((Event::CudaSvdOutputUpload as usize) < EVENTS);

    /// Objects alive at once, whose peak a test bounds.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Gauge {
        /// Device QR route scratch.
        #[cfg(feature = "cuda")]
        CudaQrScratch,
        /// Device SVD route scratch.
        #[cfg(feature = "cuda")]
        CudaSvdScratch,
    }

    const GAUGES: usize = 3;

    /// A failure site a test can inject.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Fault {
        /// The checked tensor product fails after its provider work, before
        /// scattering its payload.
        TensorProductScatter,
        /// Device `eigh_full` fails at one check of the named stage.
        #[cfg(feature = "cuda")]
        CudaEigh(&'static str),
    }

    pub(super) struct Probe {
        counts: [usize; EVENTS],
        live: [usize; GAUGES],
        peak: [usize; GAUGES],
        pub(super) routes: RoutePolicy,
        fault: Option<(Fault, usize)>,
        dense: Option<Box<dyn symtn_dense::DenseExecutor + Send>>,
    }

    thread_local! {
        pub(super) static PROBE: RefCell<Probe> = const {
            RefCell::new(Probe {
                counts: [0; EVENTS],
                live: [0; GAUGES],
                peak: [0; GAUGES],
                routes: RoutePolicy::DEFAULT,
                fault: None,
                dense: None,
            })
        };
    }

    /// Records one `event`.
    pub(crate) fn hit(event: Event) {
        add(event, 1);
    }

    /// Records `count` occurrences of `event`.
    pub(crate) fn add(event: Event, count: usize) {
        PROBE.with(|probe| probe.borrow_mut().counts[event as usize] += count);
    }

    /// One more `gauge` object is alive.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn enter(gauge: Gauge) {
        PROBE.with(|probe| {
            let probe = &mut *probe.borrow_mut();
            let live = &mut probe.live[gauge as usize];
            *live += 1;
            probe.peak[gauge as usize] = probe.peak[gauge as usize].max(*live);
        });
    }

    /// One `gauge` object was released.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn leave(gauge: Gauge) {
        PROBE.with(|probe| probe.borrow_mut().live[gauge as usize] -= 1);
    }

    /// What this thread recorded since [`record`] returned this guard.
    pub(crate) struct Recording {
        counts: [usize; EVENTS],
        #[cfg(feature = "cuda")]
        live: [usize; GAUGES],
    }

    /// Starts a recording. It restarts every gauge's peak at its current
    /// value, so recordings must not nest.
    pub(crate) fn record() -> Recording {
        PROBE.with(|probe| {
            let probe = &mut *probe.borrow_mut();
            probe.peak = probe.live;
            Recording {
                counts: probe.counts,
                #[cfg(feature = "cuda")]
                live: probe.live,
            }
        })
    }

    impl Recording {
        /// Occurrences of `event` since the recording started.
        pub(crate) fn get(&self, event: Event) -> usize {
            PROBE.with(|probe| probe.borrow().counts[event as usize]) - self.counts[event as usize]
        }

        /// `(live, peak)` of `gauge` above its value at the start.
        #[cfg(feature = "cuda")]
        pub(crate) fn gauge(&self, gauge: Gauge) -> (usize, usize) {
            let start = self.live[gauge as usize];
            PROBE.with(|probe| {
                let probe = probe.borrow();
                (
                    probe.live[gauge as usize] - start,
                    probe.peak[gauge as usize] - start,
                )
            })
        }
    }

    /// Runs `f` under `routes`, restoring the previous policy afterwards.
    pub(crate) fn with_routes<T>(routes: RoutePolicy, f: impl FnOnce() -> T) -> T {
        let previous =
            PROBE.with(|probe| std::mem::replace(&mut probe.borrow_mut().routes, routes));
        let value = f();
        PROBE.with(|probe| probe.borrow_mut().routes = previous);
        value
    }

    /// Arms `fault` to fire at its `ordinal`-th (one-based) [`fires`] check.
    pub(crate) fn arm(fault: Fault, ordinal: usize) {
        PROBE.with(|probe| probe.borrow_mut().fault = Some((fault, ordinal)));
    }

    /// Disarms any armed fault.
    #[cfg(feature = "cuda")]
    pub(crate) fn disarm() {
        PROBE.with(|probe| probe.borrow_mut().fault = None);
    }

    /// Whether the armed `fault` fires at this check; a fired fault disarms.
    pub(crate) fn fires(fault: Fault) -> bool {
        PROBE.with(|probe| {
            let armed = &mut probe.borrow_mut().fault;
            match *armed {
                Some((armed_fault, 1)) if armed_fault == fault => {
                    *armed = None;
                    true
                }
                Some((armed_fault, ordinal)) if armed_fault == fault => {
                    *armed = Some((fault, ordinal - 1));
                    false
                }
                _ => false,
            }
        })
    }

    /// Replaces the rhs structure the checked tensor product validates.
    /// While the returned guard lives, every factorization on this thread
    /// runs on `executor` instead of a fresh default one (the facade's only
    /// dense-executor seam is this test hook); dropping the guard restores
    /// the previous injection.
    pub(crate) fn inject_dense(
        executor: Box<dyn symtn_dense::DenseExecutor + Send>,
    ) -> DenseInjection {
        let previous = PROBE.with(|probe| probe.borrow_mut().dense.replace(executor));
        DenseInjection {
            previous: Some(previous),
        }
    }

    pub(crate) struct DenseInjection {
        previous: Option<Option<Box<dyn symtn_dense::DenseExecutor + Send>>>,
    }

    impl Drop for DenseInjection {
        fn drop(&mut self) {
            let previous = self.previous.take().flatten();
            PROBE.with(|probe| probe.borrow_mut().dense = previous);
        }
    }

    pub(crate) fn take_injected_dense() -> Option<Box<dyn symtn_dense::DenseExecutor + Send>> {
        PROBE.with(|probe| probe.borrow_mut().dense.take())
    }

    pub(crate) fn return_injected_dense(executor: Box<dyn symtn_dense::DenseExecutor + Send>) {
        PROBE.with(|probe| probe.borrow_mut().dense = Some(executor));
    }
}
