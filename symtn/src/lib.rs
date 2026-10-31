#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Public facade for symtn.
//!
//! # Execution model
//!
//! A [`typed::TensorMap`] is a block-sparse symmetric tensor map stored as
//! TensorKit-equivalent reduced blocks indexed by fusion trees (one coupled
//! sector per block, column-major dense storage inside). Every op is dispatched
//! through the symmetry **rule provider** owned by its
//! [`typed::GradedSpace`]s. The provider, scalar and storage are concrete type
//! parameters, while tensor rank and sector content remain runtime values. A
//! category defined outside this workspace enters as a `fusion_symbols`
//! category under [`sector::CategoryRule`] and therefore uses the same ordinary
//! tensor API as the built-in U(1), Z2, fZ2, SU(2), and product providers when
//! its admission satisfies the operation's capability bounds; those bounds,
//! the sealed provider traits generic code names, are in [`sector`].
//!
//! **Execution.** Every numerical operation takes the caller's engine as its
//! first argument, `engine: &mut dyn Engine` ([`typed::Engine`]): a
//! [`typed::CpuEngine`] (serial, or over a [`typed::Pool`] of threads) for
//! Host tensors. The caller owns the engine; symtn stores none, and the
//! dense provider (faer or a linked BLAS/LAPACK) is the one compiled in by
//! cargo feature. A [`typed::Runtime`] holds prepared structural state only:
//! the per-rule contraction/tree-transform contexts, reusable scratch, and
//! the contraction-plan cache `symtn-network` keys by network topology.
//!
//! **Parallelism.** One engine runs one operation at a time; to run
//! operations concurrently, hold one engine per thread and share the
//! tensors and the `Runtime`. Pool checkout and return and plan-cache and
//! structural-store access still synchronize briefly. A device operation
//! borrows the caller's CUDA engine, which owns its stream and allocation
//! domain; symtn takes no device lock and sets no process-wide CubeCL
//! configuration, and an operand of another engine is refused before any
//! work. No general lock-free, overlap, or outer-thread scaling guarantee is
//! made. See
//! `docs/backend_policy.md` for the ownership and synchronization model.
//!
//! # Modules
//!
//! Every public item has exactly one path:
//!
//! - [`typed`]: the tensor API — [`typed::Runtime`], [`typed::GradedSpace`],
//!   [`typed::TensorMap`], their options, results and errors;
//! - [`sector`]: the symmetry contract — the built-in fusion rules and their
//!   labels, category adapters and codecs, and the sealed capability bounds
//!   used by generic tensor code;
//! - [`expert`]: storage, block views, the dense executor seam and cache
//!   observability;
//! - `krylov` (feature `krylov`): the [krylov](https://docs.rs/krylov)
//!   crate re-exported, with Host [`typed::TensorMap`] as its vector type
//!   and the caller's [`typed::Engine`] as its execution context.
//!
//! Checked-provider failures remain structured as
//! [`typed::GenericTensorError`] variants whose nested errors retain their
//! standard [`std::error::Error::source`] chains.
//!
//! ```
//! use std::error::Error;
//! use symtn::typed::GenericTensorError;
//!
//! fn is_plan<E>(error: &GenericTensorError<E>) -> bool {
//!     matches!(error, GenericTensorError::Plan(_))
//! }
//!
//! fn source<E: Error + 'static>(error: &GenericTensorError<E>) -> Option<&(dyn Error + 'static)> {
//!     error.source()
//! }
//! ```
//!
//! # Primitive examples
//!
//! symtn supplies composable operations on symmetric tensors. A
//! [`typed::GradedSpace`] describes a leg; [`typed::TensorMap`] stores a map
//! `codomain <- domain` as reduced blocks. A shared [`typed::Runtime`] owns
//! execution resources. Host is the default; CUDA transfers are explicit.
//!
//! Run a Host example with `cargo run -p symtn --example NAME`.
//! The contraction example belongs to `symtn-network`.
//!
//! - [spaces_and_tensors](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/spaces_and_tensors.rs): U(1) and SU(2) spaces, dual legs, construction, and blocks.
//! - [permute_braid_twist](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/permute_braid_twist.rs): fermionic exchange and twist signs.
//! - [factorize_and_truncate](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/factorize_and_truncate.rs): SVD reconstruction and spectrum-based bond restriction.
//! - [eigh_and_matrix_functions](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/eigh_and_matrix_functions.rs): Hermitian eigendecomposition, exponential, and a spectral square root.
//! - [trace_inner_norm](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/trace_inner_norm.rs): trace, inner product, and norm.
//! - [dense_roundtrip](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/dense_roundtrip.rs): physical dense expansion and projection.
//! - [cuda_transfer](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn/examples/cuda_transfer.rs): explicit Host/CUDA roundtrip (requires `cuda` and a device).
//! - [contract](https://github.com/Ryo-wtnb11/symtn-rs/blob/main/symtn-network/examples/contract.rs): index notation and explicit labeled networks.

pub mod cache;
// Compiles the repository README quickstart as a doctest (never part of the
// shipped API; the README sits outside the package directory).
#[cfg(doctest)]
#[doc = include_str!("../../README.md")]
struct ReadmeDoctests;
#[cfg(test)]
mod cache_concurrency_tests;
mod error;
pub mod expert;
#[cfg(feature = "krylov")]
pub mod krylov;
mod runtime;
pub mod sector;
mod tensor_product;
#[cfg(test)]
mod test_cache;
pub mod typed;

/// Formula-first explanation of symtn's tensor-map convention, duals,
/// contractions, block layout, and weighted norms.
pub mod mathematics {
    #![doc = include_str!("mathematics.md")]
}

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use num_complex::{Complex32, Complex64};
    pub(crate) mod numerics;
}

/// Variant predicates for unit tests: a test asserts which error it got, not
/// only that there was one. `InvalidArgument` is stringly, so only its reason
/// is matched.
#[cfg(test)]
mod test_errors {
    use crate::typed::{Error, OperationError};

    pub(crate) trait ErrorKind {
        fn is_invalid_argument(&self, reason: &str) -> bool;
        fn is_destination_shared(&self) -> bool;
        /// `Error::Operation` whose payload `accept` takes.
        fn fails_in_operation(&self, accept: impl Fn(&OperationError) -> bool) -> bool;
    }

    impl<T> ErrorKind for Result<T, Error> {
        fn is_invalid_argument(&self, reason: &str) -> bool {
            matches!(self, Err(Error::InvalidArgument(text)) if text.contains(reason))
        }

        fn is_destination_shared(&self) -> bool {
            matches!(self, Err(Error::DestinationShared))
        }

        fn fails_in_operation(&self, accept: impl Fn(&OperationError) -> bool) -> bool {
            matches!(self, Err(Error::Operation(operation)) if accept(operation))
        }
    }
}
