//! Contracts every engine promises, written once and run on each: the Host
//! instantiation runs everywhere, the CUDA one (`--features cuda`) is
//! `#[ignore]`d and needs a real device. Device-only contracts (transfer
//! counts, residency, device == Host parity) stay in the `typed_cuda_*`
//! files.

use symtn::typed::{Engine, TensorMap};

/// The Host engine: upload and download are the identity.
struct Host;

impl Host {
    fn engine() -> symtn::typed::CpuEngine {
        symtn::typed::CpuEngine::serial()
    }
    fn upload<R, D>(_: &mut dyn Engine, tensor: &TensorMap<R, D>) -> TensorMap<R, D> {
        tensor.clone()
    }
    fn download<R, D>(_: &mut dyn Engine, tensor: &TensorMap<R, D>) -> TensorMap<R, D> {
        tensor.clone()
    }
}

/// The CUDA engine on device 0: operands are uploaded and results
/// downloaded through the engine that owns them.
#[cfg(feature = "cuda")]
struct Cuda;

#[cfg(feature = "cuda")]
impl Cuda {
    fn engine() -> symtn::typed::CudaEngine {
        crate::common::cuda_engine()
    }
    fn upload<R, D: symtn::typed::CudaPayload>(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D>,
    ) -> TensorMap<R, D, symtn::typed::CudaStorage<D>> {
        tensor.to_cuda(engine).unwrap()
    }
    fn download<R, D: symtn::typed::CudaPayload>(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D, symtn::typed::CudaStorage<D>>,
    ) -> TensorMap<R, D> {
        tensor.to_host(engine).unwrap()
    }
}

/// Generates `$name::host` and, under `cuda`, the ignored `$name::cuda`,
/// each running `$body` with `$backend` bound to that engine.
macro_rules! on_each_engine {
    ($(#[doc = $doc:literal])* fn $name:ident<$backend:ident>() $body:block) => {
        $(#[doc = $doc])*
        mod $name {
            use super::*;

            #[test]
            fn host() {
                type $backend = crate::engine_suites::Host;
                $body
            }

            #[cfg(feature = "cuda")]
            #[test]
            #[ignore = "requires a real CUDA device"]
            fn cuda() {
                type $backend = crate::engine_suites::Cuda;
                $body
            }
        }
    };
}

mod contract;
mod leg_roles;
mod trace;
