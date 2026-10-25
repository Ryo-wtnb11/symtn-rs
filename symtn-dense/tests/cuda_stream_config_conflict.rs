//! Separate binary: CubeCL stream configuration is process-wide and set once.
//! #1391: a device must not open over a CubeCL configuration that something
//! else loaded first with more than one stream, because the single stream is
//! what orders writes into buffers bound earlier (tensor4all/cubecl#16).
//! Its own binary: CubeCL's configuration is process-wide and set once. The
//! check precedes any device access, so this runs without a GPU.

#![cfg(feature = "cuda")]

use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use symtn_dense::{CudaDenseContext, DenseError};

#[test]
fn a_preloaded_multi_stream_configuration_is_rejected() {
    let mut config = CubeClRuntimeConfig::default();
    config.streaming.max_streams = 4;
    CubeClRuntimeConfig::set(config);
    match CudaDenseContext::new(0) {
        Err(DenseError::Unsupported { op, message }) => {
            assert_eq!(op, "cuda_context");
            assert!(message.contains("max_streams = 4"), "{message}");
        }
        Err(other) => panic!("expected the stream-count rejection, got {other:?}"),
        Ok(_) => panic!("opened a device over a multi-stream CubeCL configuration"),
    }
}
