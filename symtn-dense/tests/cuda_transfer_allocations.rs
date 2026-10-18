//! Host-allocation contract of the CUDA transfer boundary.
//!
//! The probe counts caller-thread allocations whose size is exactly the
//! payload length in bytes, which is the only allocation class a redundant
//! payload copy can belong to; every other allocation Tenferro, CubeCL and the
//! CUDA driver make is a different size and is ignored. The device is required
//! because `upload`/`download` are the units under test, so these are
//! `#[ignore]` like the rest of the device suite.

#![cfg(feature = "cuda")]

use std::sync::Mutex;

use num_complex::Complex64;
use symtn_dense::{cuda_transfer_stats, CudaDenseContext, CudaDenseStorage, CudaScalar};

/// Payload-sized host allocations a download makes on the caller thread: the
/// one vector Tenferro decodes the readback bytes into, handed straight back.
/// Measured on an A100: 1 for both f64 and `Complex64`; a second copy of that
/// vector by symtn would make it 2.
///
/// Payload-sized host allocations an owned upload makes, measured on the same
/// host: CubeCL's own staging, which symtn does not control but must not add
/// to. `Complex64` needs one more than f64 because CubeCL stages complex data
/// through an extra buffer. Pinning them absolutely is what catches a copy
/// reintroduced inside `upload_owned`; the borrowed entry point is then
/// asserted at exactly one more, which is the copy Tenferro's owned-host-tensor
/// contract requires.
const OWNED_UPLOAD_PAYLOAD_ALLOCATIONS_F64: usize = 2;
const OWNED_UPLOAD_PAYLOAD_ALLOCATIONS_C64: usize = 3;
const DOWNLOAD_PAYLOAD_ALLOCATIONS: usize = 1;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

static DEVICE: Mutex<()> = Mutex::new(());

/// Runs `f` while counting caller-thread allocations of exactly
/// `payload_bytes`. A zero target would match nothing, so it is rejected.
fn measure<T>(payload_bytes: usize, f: impl FnOnce() -> T) -> (T, usize) {
    assert_ne!(payload_bytes, 0, "the probe needs a nonzero payload size");
    let (output, allocs) = counting_alloc::measure_matching(payload_bytes..=payload_bytes, f);
    (output, allocs.matched_calls as usize)
}

/// Element count large enough to leave Tenferro's pinned small-payload
/// readback path (`PINNED_SCALAR_BYTES`) and odd enough that no incidental
/// allocation happens to share the payload size.
const ELEMENTS: usize = 4099;

fn assert_transfer_copies<D>(ctx: &CudaDenseContext, data: Vec<D>, owned_upload_allocations: usize)
where
    D: CudaScalar + PartialEq + std::fmt::Debug,
{
    let payload_bytes = std::mem::size_of_val(data.as_slice());

    let borrowed_before = cuda_transfer_stats();
    let (borrowed, borrowed_hits) = measure(payload_bytes, || {
        CudaDenseStorage::upload::<D>(ctx, &data).unwrap()
    });
    let borrowed_upload = stats_delta(borrowed_before, cuda_transfer_stats());

    let owned_source = data.clone();
    let owned_before = cuda_transfer_stats();
    let (owned, owned_hits) = measure(payload_bytes, || {
        CudaDenseStorage::upload_owned::<D>(ctx, owned_source).unwrap()
    });
    let owned_upload = stats_delta(owned_before, cuda_transfer_stats());

    assert_eq!(
        owned_hits, owned_upload_allocations,
        "an owned upload of {payload_bytes} bytes must stage exactly what CubeCL \
         stages and copy the payload no further"
    );
    assert_eq!(
        borrowed_hits,
        owned_upload_allocations + 1,
        "a borrowed upload of {payload_bytes} bytes costs exactly one host copy \
         more than an owned one"
    );
    assert_eq!(
        owned_upload, borrowed_upload,
        "device traffic must be identical for owned and borrowed uploads"
    );

    let download_before = cuda_transfer_stats();
    let (values, download_hits) = measure(payload_bytes, || owned.download::<D>(ctx).unwrap());
    let download = stats_delta(download_before, cuda_transfer_stats());

    assert_eq!(values, data, "download must return the uploaded payload");
    assert_eq!(
        download_hits, DOWNLOAD_PAYLOAD_ALLOCATIONS,
        "download must hand back Tenferro's vector instead of copying it"
    );
    assert_eq!(download.d2h_calls, 1);
    assert_eq!(download.d2h_bytes, payload_bytes as u64);

    let (round_trip, _) = measure(payload_bytes, || borrowed.download::<D>(ctx).unwrap());
    assert_eq!(round_trip, data);
}

fn stats_delta(
    before: symtn_dense::CudaTransferStats,
    after: symtn_dense::CudaTransferStats,
) -> symtn_dense::CudaTransferStats {
    symtn_dense::CudaTransferStats {
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

#[test]
#[ignore = "requires a real CUDA device"]
fn owned_f64_transfers_cost_no_redundant_host_copy() {
    let _serialized = DEVICE.lock().unwrap_or_else(|err| err.into_inner());
    let ctx = CudaDenseContext::new(0).unwrap();
    let data: Vec<f64> = (0..ELEMENTS).map(|index| index as f64).collect();
    assert_transfer_copies(&ctx, data, OWNED_UPLOAD_PAYLOAD_ALLOCATIONS_F64);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn owned_complex64_transfers_cost_no_redundant_host_copy() {
    let _serialized = DEVICE.lock().unwrap_or_else(|err| err.into_inner());
    let ctx = CudaDenseContext::new(0).unwrap();
    let data: Vec<Complex64> = (0..ELEMENTS)
        .map(|index| Complex64::new(index as f64, -(index as f64) - 0.5))
        .collect();
    assert_transfer_copies(&ctx, data, OWNED_UPLOAD_PAYLOAD_ALLOCATIONS_C64);
}
