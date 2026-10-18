//! Krylov methods on symmetric tensors (feature `krylov`).
//!
//! The [krylov](https://docs.rs/krylov) crate, re-exported so that an
//! application names exactly the krylov version whose [`Vector`] trait Host
//! [`TensorMap`] implements. Every krylov solver then runs directly on
//! tensors. For `eigsolve`, `exponentiate`, `linsolve` and the Arnoldi
//! solvers a linear map is any closure
//! `FnMut(&TensorMap<R, D>, &mut TensorMap<R, D>) -> Result<(), FacadeError<R>>`;
//! `svdsolve` also needs the adjoint, so its map is a type of your own
//! implementing [`AdjointMap`].
//!
//! # The vector operations
//!
//! Each is one public symtn operation, with TensorKit's `VectorInterface`
//! semantics (TensorKit 0.17.1 `src/tensors/vectorinterface.jl`):
//!
//! | `krylov::Vector` | symtn | TensorKit |
//! |---|---|---|
//! | `similar` | [`TensorMap::zeros_like`] | `zerovector` |
//! | `copy_from(x)` | `x.axpby_into(self, 1, 0)` | `scale!(ty, tx, 1)` |
//! | `axpby(a, x, b)` | `x.axpby_into(self, a, b)` | `add!(ty, tx, α, β)` |
//! | `inner(y)` | [`TensorMap::inner`] | `inner` |
//! | `norm` | [`TensorMap::norm`]`(2.0)` | `norm` |
//!
//! `inner` and `norm` are weighted by the quantum dimension of each coupled
//! sector, so they are the Frobenius inner product and norm of the full
//! tensor; `axpby` with `b = 0` overwrites without reading `self`, as krylov
//! requires. Everything works in place on the reduced-block payload: only
//! `similar` allocates (one payload). `norm` accumulates in `f64` and is
//! narrowed once for `f32`/`Complex32`.
//!
//! # Boundaries
//!
//! - Host storage only: CUDA tensors do not implement [`Vector`].
//! - `similar` of a lazy adjoint or a compact diagonal is
//!   [`crate::typed::Error::InvalidArgument`]: neither is a valid `axpby_into`
//!   destination, and densifying silently would hide a copy. Start from
//!   `x0.materialize()?`; such tensors are still accepted as the source of
//!   `copy_from`/`axpby`.
//! - Errors of the operations (space mismatch, a destination shared with a
//!   clone, …) reach the caller as `krylov::Error::Vector`; NaN or infinite
//!   values are left to krylov's own checks.
//!
//! ```
//! use std::sync::Arc;
//!
//! use symtn::krylov::{self, Lanczos, Which, Workspace};
//! use symtn::sector::{U1FusionRule, U1Irrep};
//! use symtn::typed::{GradedSpace, Runtime, TensorMap};
//!
//! let rt = Runtime::builder().build()?;
//! let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)])?;
//! // H = diag(1, 2, 3) on charge 0 and diag(1, 2) on charge 1.
//! let h = TensorMap::<_, f64>::from_subblock_fn(&rt, [&v], [&v], |_, i| {
//!     if i[0] == i[1] { (i[0] + 1) as f64 } else { 0.0 }
//! })?;
//! let x0 = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [], 1)?;
//! let mut op = |x: &TensorMap<U1FusionRule, f64>, y: &mut TensorMap<U1FusionRule, f64>| {
//!     h.compose(x)?.axpby_into(y, 1.0, 0.0)
//! };
//! let mut ws = Workspace::with_capacity(&x0, 5)?;
//! let (values, _, _) =
//!     krylov::eigsolve(&mut op, &x0, 1, Which::LargestReal, &Lanczos::new(5, 100, 1e-12), &mut ws)?;
//! assert!((values[0] - 3.0).abs() < 1e-10);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # LAPACK provider
//!
//! krylov declares the LAPACK routines of its small dense problems; the
//! final binary must link a provider. Enable one of symtn's `blas-*`
//! features (`blas-openblas`, `blas-accelerate`, `blas-mkl`) or link one
//! yourself. The default `cpu-faer` links none, and a missing provider is a
//! link error, never a fallback.

pub use ::krylov::*;

use num_complex::{Complex32, Complex64};

use crate::sector::TypedSectorAdmission;
// `Error` is not imported: it would shadow the re-exported `krylov::Error`.
use crate::typed::{
    ElementwiseExec, FacadeError, FactorScalar, ReduceExec, TensorMap, TypedAdjointSpace,
    TypedSpaceModeDispatch, TypedTensorModeDispatch,
};

// One impl per payload dtype: krylov's `Real<D>` comes from a sealed trait,
// and the `f64` norm narrows to it with a plain cast.
macro_rules! host_vector {
    ($d:ty, $real:ty) => {
        impl<R> Vector for TensorMap<R, $d>
        where
            R: TypedSectorAdmission,
            R::Mode: TypedTensorModeDispatch<R> + TypedSpaceModeDispatch<R> + TypedAdjointSpace<R>,
            Vec<$d>: ElementwiseExec<R, $d> + ReduceExec<R, $d>,
        {
            type Scalar = $d;
            type Error = FacadeError<R>;

            fn similar(&self) -> Result<Self, Self::Error> {
                if !self.is_owned_dense() {
                    return Err(crate::typed::Error::InvalidArgument(
                        "krylov needs an owned dense vector: pass `materialize()` of a lazy \
                         adjoint or compact diagonal"
                            .into(),
                    )
                    .into());
                }
                self.zeros_like()
            }

            fn copy_from(&mut self, x: &Self) -> Result<(), Self::Error> {
                x.axpby_into(self, <$d>::from_real(1.0), <$d>::from_real(0.0))
            }

            fn inner(&self, y: &Self) -> Result<$d, Self::Error> {
                TensorMap::inner(self, y)
            }

            fn norm(&self) -> Result<$real, Self::Error> {
                Ok(TensorMap::norm(self, 2.0)? as $real)
            }

            fn axpby(&mut self, a: $d, x: &Self, b: $d) -> Result<(), Self::Error> {
                x.axpby_into(self, a, b)
            }
        }
    };
}

host_vector!(f64, f64);
host_vector!(f32, f32);
host_vector!(Complex64, f64);
host_vector!(Complex32, f32);
