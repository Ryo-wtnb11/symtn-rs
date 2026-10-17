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
//! fusion rule defined outside this workspace therefore uses the same ordinary
//! tensor API as the built-in U(1), Z2, fZ2, SU(2), and product providers when
//! they satisfy the required admission and operation capability bounds; the
//! traits it implements and the bounds generic code names are in [`sector`].
//!
//! A [`typed::Runtime`] owns the shared execution state: the per-rule
//! contraction/tree-transform contexts, the dense backend (selectable per
//! [`typed::LinalgBackend`]), and the
//! contraction-plan cache `symtn-network` keys by network topology.
//!
//! **Parallelism.** A `Runtime` is cheap to clone across threads. Standalone
//! operations normally lease independent per-rule contexts and, for
//! factorizations, dense executors instead of holding the Runtime's coarse state
//! mutex for the full operation. Cached network contraction uses plan-local workspace
//! pools. Pool checkout and return, plan-cache and structural-store access, and
//! dense providers may still synchronize; an injected non-mintable executor
//! serializes factorization through the Runtime state lock. Device operations
//! take only a process-wide lock per CUDA device and the runtime's own CUDA
//! context (and the device backend's own handle and plan locks), never the
//! state mutex, so Host work is not blocked by device work; device operations
//! of every Runtime on one device serialize their enqueue on that device lock,
//! which is what lets one Runtime safely read a fresh device output returned
//! by another (#1384). A host sync under a lease (`to_host`, scalar and
//! spectrum downloads) therefore stalls
//! every Runtime on that device. Opening a device also pins CubeCL to one
//! stream per device for the whole process (#1391), so every device write,
//! including `*_into` destinations and reused scratch, is ordered
//! before every later read or write from any thread; building a device
//! Runtime fails if CubeCL was already configured with more streams. GPU work
//! of different threads therefore never overlaps. The setting is
//! process-wide: it overrides a `cubecl.toml` value without notice, stays
//! fixed even if opening the device then fails, also gives every other CubeCL
//! client in the process (wgpu included) one stream, and makes a later
//! `CubeClRuntimeConfig::set` panic. Consequently no
//! general lock-free, overlap, or outer-thread scaling guarantee is made. See
//! `docs/backend_policy.md` for the ownership and synchronization model.
//!
//! # Modules
//!
//! Every public item has exactly one path:
//!
//! - [`typed`]: the tensor API — [`typed::Runtime`], [`typed::GradedSpace`],
//!   [`typed::TensorMap`], their options, results and errors;
//! - [`sector`]: the symmetry contract — the built-in fusion rules and their
//!   labels, and the traits a rule of your own implements;
//! - [`expert`]: storage, block views, the dense executor seam and cache
//!   observability;
//! - `krylov` (feature `krylov`): the [krylov](https://docs.rs/krylov)
//!   crate re-exported, with Host [`typed::TensorMap`] as its vector type.
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
#![doc = include_str!("tutorial.md")]

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
