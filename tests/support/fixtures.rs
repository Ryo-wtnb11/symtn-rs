//! Fixtures several symtn test targets share verbatim.
//!
//! Test targets include this file with
//! `#[path = ".../tests/support/fixtures.rs"] mod fixtures;`. Leg fixtures
//! stay local: targets that name a `u1_leg`/`su2_leg` build different spaces.

#![allow(dead_code)]

use symtn::typed::{Runtime, TensorMap};

/// A host Runtime with one dense thread, so results and allocation counts do
/// not depend on the machine's core count.
pub fn host_runtime() -> Runtime {
    Runtime::builder().build().unwrap()
}

/// The receiver's own split as leg roles: `rows = 0..nout`.
pub fn codomain_axes<R, D, S>(t: &TensorMap<R, D, S>) -> Vec<usize> {
    (0..t.codomain_rank()).collect()
}

/// The receiver's own split as leg roles: `cols = nout..rank`.
pub fn domain_axes<R, D, S>(t: &TensorMap<R, D, S>) -> Vec<usize> {
    (t.codomain_rank()..t.rank()).collect()
}

#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub use cuda::*;

#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use symtn::expert::CudaTransferStats;

    static SERIAL: Mutex<()> = Mutex::new(());

    /// Serializes the device tests of one binary, whose counters are
    /// process-wide; a failing test does not poison the rest.
    pub fn serial() -> MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Transfer and kernel counters accrued between `before` and `after`.
    pub fn delta(after: CudaTransferStats, before: CudaTransferStats) -> CudaTransferStats {
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
            gauge_ops: after.gauge_ops - before.gauge_ops,
        }
    }
}
