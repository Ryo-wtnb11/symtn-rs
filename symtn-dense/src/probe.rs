//! Test probes of symtn-dense (#2221): one thread-local ledger of [`Event`]
//! counts and a log of buffer identities, read through [`record`].
//!
//! The always-compiled `cuda_transfer_stats` and the Tenferro session
//! counters are public observability, not probes, and stay with their
//! owners. `CudaFullDownloadBytes` is the part of `d2h_bytes` that went
//! through `CudaDenseStorage::download`, `CudaMetadataDownloadBytes` the
//! part that went through `cuda_download`; the two are not one event.

// Without the `cuda` feature only the GEMM plan events are recorded.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::cell::RefCell;

/// One observed occurrence in the dense boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// Bytes `CudaDenseStorage::download` moved device to host.
    CudaFullDownloadBytes,
    /// Bytes `cuda_download` moved device to host (reductions, spectra).
    CudaMetadataDownloadBytes,
    /// The native block GEMM kernel looked up one distinct geometry's plan.
    GemmPlanLookup,
    /// The native block GEMM kernel prepared one geometry's `tcontract::Plan`
    /// (a geometry-plan cache miss).
    GemmPlanBuild,
}

const EVENTS: usize = Event::GemmPlanBuild as usize + 1;

#[derive(Default)]
struct Probe {
    counts: [usize; EVENTS],
    owned_full_svd_inputs: Vec<usize>,
}

std::thread_local! {
    static PROBE: RefCell<Probe> = const {
        RefCell::new(Probe { counts: [0; EVENTS], owned_full_svd_inputs: Vec::new() })
    };
}

/// Records `count` occurrences of `event`.
pub(crate) fn add(event: Event, count: usize) {
    PROBE.with(|probe| probe.borrow_mut().counts[event as usize] += count);
}

/// Records the address of the buffer `svd_full_owned` handed to the backend.
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
pub(crate) fn owned_full_svd_input(pointer: usize) {
    PROBE.with(|probe| probe.borrow_mut().owned_full_svd_inputs.push(pointer));
}

/// What this thread recorded since [`record`] returned this guard.
pub(crate) struct Recording {
    counts: [usize; EVENTS],
    inputs_start: usize,
}

/// Starts a recording.
pub(crate) fn record() -> Recording {
    PROBE.with(|probe| {
        let probe = probe.borrow();
        Recording {
            counts: probe.counts,
            inputs_start: probe.owned_full_svd_inputs.len(),
        }
    })
}

impl Recording {
    /// Occurrences of `event` since the recording started.
    pub(crate) fn get(&self, event: Event) -> usize {
        PROBE.with(|probe| probe.borrow().counts[event as usize]) - self.counts[event as usize]
    }

    /// Input buffer addresses of `svd_full_owned` since the recording started.
    #[cfg(all(
        any(feature = "cpu-faer", feature = "cpu-blas-core"),
        not(feature = "provider-inject")
    ))]
    pub(crate) fn owned_full_svd_inputs(&self) -> Vec<usize> {
        PROBE.with(|probe| probe.borrow().owned_full_svd_inputs[self.inputs_start..].to_vec())
    }
}
