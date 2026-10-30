//! The seam `symtn-network` drives: runtime extension state, parked
//! destinations, and the slicing and replay hooks of `Network` execution.
//!
//! Not part of the facade. Every item here exists for `symtn-network`, which
//! sits above this crate and so cannot reach `pub(crate)` items; keeping them
//! in one hidden module keeps them out of the inherent namespaces of
//! [`TensorMap`], [`Runtime`] and [`GradedSpace`].

use super::*;

pub use super::network_seam::NetworkDegeneracyRestriction;
pub use super::tensor_repr::{NetworkPayloadStorage, NetworkReuseClass, RuntimeDetachedTensorMap};
pub use crate::runtime::{ExtensionSlot, RuntimeIdentity};

/// Non-owning identity of `runtime`, for state parked outside an execution.
pub fn runtime_identity(runtime: &Runtime) -> RuntimeIdentity {
    runtime.identity()
}

/// Locked access to the runtime's extension state. Do not run tensor
/// operations inside `f`: the extension mutex is held for its duration.
pub fn with_extension_slot<T>(runtime: &Runtime, f: impl FnOnce(&mut ExtensionSlot) -> T) -> T {
    runtime.with_extension_slot(f)
}

/// The device ordinal of the caller's CUDA engine; a non-CUDA engine is a
/// typed refusal. Opens no device work.
#[cfg(feature = "cuda")]
pub fn cuda_engine_device(engine: &mut dyn Engine) -> Result<usize, Error> {
    Ok(crate::runtime::cuda_call(engine)?.device())
}

/// Parks an ordinary dense destination without its runtime and provider.
pub fn detach_runtime<R, D, S>(
    tensor: TensorMap<R, D, S>,
) -> Option<RuntimeDetachedTensorMap<D, S>> {
    tensor.detach_runtime()
}

/// Identity and retained bytes of the dense payload allocation `tensor` owns.
pub fn network_owned_payload<R, D, S>(tensor: &TensorMap<R, D, S>) -> Option<(usize, usize)>
where
    S: NetworkPayloadStorage<D>,
{
    tensor.network_owned_payload()
}

/// The representation `tensor` has after an optional adjoint.
pub fn network_reuse_class<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    adjoint: bool,
) -> NetworkReuseClass {
    tensor.network_reuse_class(adjoint)
}

/// Whether `tensor`, after an optional adjoint, has the expected legs and
/// representation; allocation-free.
pub fn network_input_metadata_matches<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    adjoint: bool,
    expected_legs: &[SectorLeg],
    expected_class: NetworkReuseClass,
) -> bool
where
    R: TypedSectorAdmission,
{
    tensor.network_input_metadata_matches(adjoint, expected_legs, expected_class)
}

/// The stored logical leg of source axis `axis`.
pub fn network_source_leg<R, D, S>(tensor: &TensorMap<R, D, S>, axis: usize) -> Option<&SectorLeg> {
    tensor.network_source_leg(axis)
}

/// Restricts tensor-local degeneracy coordinates for network slicing.
pub fn network_restrict_degeneracies<R, D>(
    tensor: &TensorMap<R, D>,
    adjoint: bool,
    restrictions: &[NetworkDegeneracyRestriction],
) -> Result<TensorMap<R, D>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    tensor.network_restrict_degeneracies(adjoint, restrictions)
}

/// Whether `tensor` stores a compact diagonal payload.
pub fn network_has_compact_payload<R, D>(tensor: &TensorMap<R, D>) -> bool
where
    D: TensorScalar,
{
    tensor.network_has_compact_payload()
}

/// A zero tensor on `tensor`'s provider over the given effective legs.
pub fn network_zeros_from_effective_legs<R, D>(
    tensor: &TensorMap<R, D>,
    codomain: &[GradedSpace<R>],
    domain: &[GradedSpace<R>],
) -> Result<TensorMap<R, D>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    tensor.network_zeros_from_effective_legs(codomain, domain)
}

/// Adds every block of `source` into the matching rectangle of `destination`.
pub fn network_scatter_add_assign<R, D>(
    engine: &mut dyn Engine,
    destination: &mut TensorMap<R, D>,
    source: &TensorMap<R, D>,
    ranges: &[Option<std::ops::Range<usize>>],
) -> Result<(), Error>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    destination.network_scatter_add_assign(engine, source, ranges)
}

/// The raw logical leg of `space`, for typed network replay admission.
pub fn network_sector_leg<R>(space: &GradedSpace<R>) -> &SectorLeg {
    space.network_sector_leg()
}
