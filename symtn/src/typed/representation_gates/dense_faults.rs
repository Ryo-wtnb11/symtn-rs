//! Dense-executor faults and spies at the facade: each test runs its dense
//! work on an injected `SpyExecutor` (`probe::inject_dense`) and states which
//! kernels a route reaches, how a kernel failure surfaces, and that a failed
//! call publishes nothing.

use super::*;

use crate as symtn;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorId, SectorVec, TypedSectorAdmission,
};
use crate::typed::__network::NetworkReuseClass;
use crate::typed::{GenericTensorError, TensorMap};

include!("../../../tests/common/checked_only_toy.rs");

mod backend_selection;
mod batched;
mod eig;
mod error_kinds;
mod null_polar;
mod pinv;
mod qr_lq;
mod svd;

/// The full and values-only SVD entries a pinv/polar call may reach.
const PINV_SVD: &[Kernel] = &[Kernel::Svd, Kernel::SvdInto, Kernel::SvdVals];

/// Counts SVD (`PINV_SVD`) and GEMM (`Kernel::GEMM`) calls; the `fail_svd`-th
/// SVD or the `fail_gemm`-th GEMM call fails.
fn pinv_spy(
    counts: &Arc<SpyCounts>,
    fail_svd: Option<usize>,
    fail_gemm: Option<usize>,
) -> SpyExecutor {
    let mut spy = SpyExecutor::counting(counts);
    if let Some(nth) = fail_svd {
        spy = spy.failing(PINV_SVD, Some(nth), "injected pinv SVD failure");
    }
    if let Some(nth) = fail_gemm {
        spy = spy.failing(Kernel::GEMM, Some(nth), "injected pinv GEMM failure");
    }
    spy
}
