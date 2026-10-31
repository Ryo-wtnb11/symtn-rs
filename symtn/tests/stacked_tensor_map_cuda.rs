//! `StackedTensorMap` transfers on a real device.
//!
//! `cuda_transfer_stats` is thread-local. Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test integration stacked_tensor_map_cuda:: -- --ignored`.

#![cfg(feature = "cuda")]

use num_complex::Complex64;

use symtn::expert::cuda_transfer_stats;
use symtn::typed::Runtime;
use symtn::typed::StackedTensorMap;

use crate::stacked_fixtures as fixtures;

macro_rules! device_round_trip {
    ($cuda:ident, $engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let runtime = Runtime::builder().build().unwrap();
        device_round_trip!($cuda, $engine, @dtype $label, &runtime, &a, f64);
        device_round_trip!($cuda, $engine, @dtype $label, &runtime, &a, Complex64);
    }};
    ($cuda:ident, $engine:ident, @dtype $label:expr, $runtime:expr, $a:expr, $d:ty) => {{
        let label = format!("{} {}", $label, stringify!($d));
        let members = members!($engine, $runtime, $a, $d, 3);
        let stack = StackedTensorMap::pack(&members).unwrap();
        let bytes = (stack.len() * members[0].dense_data().unwrap().len() * std::mem::size_of::<$d>()) as u64;

        let before = cuda_transfer_stats();
        let device = stack.to_cuda($cuda).unwrap();
        let uploaded = cuda_transfer_stats();
        assert_eq!(uploaded.h2d_calls - before.h2d_calls, 1, "{label}: to_cuda H2D");
        assert_eq!(uploaded.h2d_bytes - before.h2d_bytes, bytes, "{label}: H2D bytes");
        assert_eq!(uploaded.d2h_calls, before.d2h_calls, "{label}: to_cuda D2H");

        let host = device.to_host($cuda).unwrap();
        let downloaded = cuda_transfer_stats();
        assert_eq!(downloaded.d2h_calls - uploaded.d2h_calls, 1, "{label}: to_host D2H");
        assert_eq!(downloaded.h2d_calls, uploaded.h2d_calls, "{label}: to_host H2D");

        assert!(*host.signature() == *stack.signature(), "{label}");
        let device_member = members[0].to_cuda($cuda).unwrap();
        assert!(
            *device.signature() == device_member.structure_signature(),
            "{label}: device signature"
        );
        assert!(*device.signature() != *stack.signature(), "{label}: placement");
        for (index, member) in members.iter().enumerate() {
            fixtures::assert_bit_exact(host.member($engine, index).unwrap().dense_data().unwrap(), member.dense_data().unwrap(), &label);
        }
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_round_trip_is_bit_exact_with_one_transfer_each_way() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, device_round_trip, cuda);
}
