//! The lane-GEMM seam: a Host region that owns its own lanes runs one
//! prepared GEMM per job on a lane token, without entering the engine again.
//! See [`crate::DenseExecutor::with_lane_gemms`].

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};

#[cfg(feature = "tenferro")]
use core::num::NonZeroUsize;
#[cfg(feature = "tenferro")]
use core::time::Duration;

#[cfg(feature = "tenferro")]
use crate::DenseError;

/// One lane GEMM's geometry: `rows x k` times `k x n`, every operand packed
/// column-major with no transposition.
#[cfg(feature = "tenferro")]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneGemmShape {
    pub rows: usize,
    pub k: usize,
    pub n: usize,
}

/// One lane GEMM's packed column-major operands, its uninitialized
/// destination and the slot that receives the destination once written, for
/// one GEMM dtype. `lhs` is `rows x contracted`, `rhs` `contracted x cols`,
/// `destination` `rows x cols`, each exactly that long. `written` is set only
/// when the GEMM succeeded.
#[doc(hidden)]
pub enum LaneGemmIo<'s, 'd> {
    F32 {
        lhs: &'s [f32],
        rhs: &'s [f32],
        destination: &'d mut [MaybeUninit<f32>],
        written: &'s mut Option<&'d mut [f32]>,
    },
    F64 {
        lhs: &'s [f64],
        rhs: &'s [f64],
        destination: &'d mut [MaybeUninit<f64>],
        written: &'s mut Option<&'d mut [f64]>,
    },
    C32 {
        lhs: &'s [Complex32],
        rhs: &'s [Complex32],
        destination: &'d mut [MaybeUninit<Complex32>],
        written: &'s mut Option<&'d mut [Complex32]>,
    },
    C64 {
        lhs: &'s [Complex64],
        rhs: &'s [Complex64],
        destination: &'d mut [MaybeUninit<Complex64>],
        written: &'s mut Option<&'d mut [Complex64]>,
    },
}

/// The placement token a lane GEMM runs on: the engine team's token, the
/// same type as `strided_kernel::Exec`.
#[cfg(feature = "tenferro")]
#[doc(hidden)]
pub type LaneToken<'a> = tenferro_cpu::threadteam::Exec<'a>;

/// The prepared GEMMs of one [`crate::DenseExecutor::with_lane_gemms`] call,
/// shared by every lane of its body.
#[cfg(feature = "tenferro")]
#[doc(hidden)]
pub trait LaneGemms: Sync {
    /// `D = A B` of job `job` (`alpha = 1`, `beta = 0`), on `token`, into
    /// `io`'s destination; no destination value is read. Never enters the
    /// engine. On `Err`, `io.written` stays `None`.
    fn run<'d>(
        &self,
        token: LaneToken<'_>,
        job: usize,
        io: LaneGemmIo<'_, 'd>,
    ) -> Result<(), DenseError>;

    /// Dense's own time for `job` at its grant within `width` slots of
    /// `token`'s team: the serial estimate the block scheduler costs it by,
    /// over the width the job's plan would take, plus that width's dispatch
    /// cost. Under a forced batch policy the scheduler does not model
    /// widening, so this is the serial estimate at every width. An unknown
    /// job costs [`Duration::MAX`].
    fn time(&self, token: &mut LaneToken<'_>, job: usize, width: NonZeroUsize) -> Duration;
}

/// The body of a [`crate::DenseExecutor::with_lane_gemms`]: called at most
/// once, on the calling thread, with the entry's whole token.
#[cfg(feature = "tenferro")]
#[doc(hidden)]
pub type LaneGemmBody<'a> = dyn FnMut(LaneToken<'_>, &dyn LaneGemms) + 'a;

#[cfg(feature = "tenferro")]
pub(crate) const LANE_GEMM_OP: &str = "with_lane_gemms";
