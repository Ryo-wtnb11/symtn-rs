//! Separate binary: CubeCL stream configuration is process-wide and set once.
//! #1391: opening a device pins CubeCL to one stream when nothing has loaded
//! its configuration yet, overriding a `cubecl.toml` value, and a later open
//! accepts the already-pinned configuration. Its own binary: the
//! configuration is process-wide and set once. It asserts only on the
//! configuration, so it runs without a GPU (the open itself may fail there).

#![cfg(feature = "cuda")]

use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use symtn_dense::{CudaDenseContext, DenseError};

fn open() -> Result<(), DenseError> {
    // Without a GPU the backend may fail or panic after the configuration
    // step; either way only the configuration is under test.
    match std::panic::catch_unwind(|| CudaDenseContext::new(0).map(drop)) {
        Ok(result) => result,
        Err(_) => Ok(()),
    }
}

#[test]
fn an_unloaded_configuration_becomes_one_stream_and_stays_accepted() {
    let dir = std::env::temp_dir().join(format!("symtn-1391-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("cubecl.toml"), "[streaming]\nmax_streams = 4\n").unwrap();
    std::env::set_current_dir(&dir).unwrap();

    let first = open();
    assert_eq!(CubeClRuntimeConfig::get().streaming.max_streams, 1);
    let second = open();
    for result in [first, second] {
        assert!(
            !matches!(
                result,
                Err(DenseError::Unsupported {
                    op: "cuda_context",
                    ..
                })
            ),
            "{result:?}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
