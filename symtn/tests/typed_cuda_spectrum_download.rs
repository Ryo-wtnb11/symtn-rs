//! Host/device traffic of the device diagonal factorizations, whatever the
//! number of coupled sectors.
//!
//! - Compact SVD writes each sector's singular values into `s` on the device
//!   (#1536): it downloads nothing, and uploads only the three zero-initialized
//!   factors (the only device allocation path, #740), so `s` costs no
//!   upload beyond its zero initialization.
//! - Full EIGH must read its eigenvalues on the host (non-finite check, ascending
//!   order, factor-space plan), all sectors' with one download (#1484), and
//!   uploads `d` once, already filled, together with `v`'s zeros; with
//!   cuSOLVER's ascending order no selector is uploaded (#1985).
//!
//! [`cuda_transfer_stats`] snapshots count only the calling thread.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! integration typed_cuda_spectrum_download:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Eigh, GradedSpace, Runtime, Svd, TensorMap};

use crate::fixtures;

use fixtures::{codomain_axes, domain_axes};

fn delta(before: CudaTransferStats) -> CudaTransferStats {
    let after = cuda_transfer_stats();
    CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        gauge_ops: after.gauge_ops - before.gauge_ops,
        ..CudaTransferStats::default()
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_diagonal_factors_transfer_only_what_the_host_decides_on() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let bytes = |len: usize| (len * std::mem::size_of::<f64>()) as u64;
    for charges in [1_usize, 9] {
        let half = charges as i32 / 2;
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (-half..=half).map(|charge| (U1Irrep::new(charge), 3)),
        )
        .unwrap();
        let host = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, index| {
            1.0 + ((index[0] * 3 + index[1] * 5) % 7) as f64
        })
        .unwrap();
        let device = host.to_cuda(cuda).unwrap();

        let before = cuda_transfer_stats();
        let Svd { u, s, vh } = device
            .svd_compact(cuda, &codomain_axes(&device), &domain_axes(&device))
            .unwrap();
        let svd = delta(before);
        let (u, s, vh) = (
            u.to_host(cuda).unwrap(),
            s.to_host(cuda).unwrap(),
            vh.to_host(cuda).unwrap(),
        );
        // Only the finite-input scan's one real scalar (#2038) comes back.
        assert_eq!(
            (svd.d2h_calls, svd.d2h_bytes),
            (1, bytes(1)),
            "{charges} sectors"
        );
        // Exactly the three zero uploads plus one upload of the gauge's
        // row weights (every route has 3 rows), so `s` adds nothing beyond
        // its zero initialization.
        assert_eq!(svd.h2d_calls, 4, "{charges} sectors");
        assert_eq!(
            svd.h2d_bytes,
            bytes(
                u.materialize(engine).unwrap().dense_data().unwrap().len()
                    + s.materialize(engine).unwrap().dense_data().unwrap().len()
                    + vh.materialize(engine).unwrap().dense_data().unwrap().len()
                    + 3
            ),
            "{charges} sectors"
        );
        // Per aligned real route: 13 phase ops, then a broadcast and a mul on
        // each side; the gauge never downloads (checked above).
        assert_eq!(svd.gauge_ops, 17 * charges as u64, "{charges} sectors");
        let Svd { s: expected, .. } = host
            .svd_compact(engine, &codomain_axes(&host), &domain_axes(&host))
            .unwrap();
        for (device, host) in s
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize(engine).unwrap().dense_data().unwrap())
        {
            assert!((device - host).abs() <= 1e-12 * host.abs().max(1.0));
        }

        let host_adjoint = host.adjoint(engine).unwrap();
        let hermitian = host.axpby(engine, 1.0, &host_adjoint, 1.0).unwrap();
        let device = hermitian.to_cuda(cuda).unwrap();
        let before = cuda_transfer_stats();
        let Eigh { d, v } = device
            .eigh_full(
                cuda,
                &codomain_axes(&device),
                &domain_axes(&device),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        let eigh = delta(before);
        let (d, v) = (d.to_host(cuda).unwrap(), v.to_host(cuda).unwrap());
        let eigenvalues = 3 * charges;
        // The eigenvalues must reach the host; the rest of the download is
        // the O(1)-per-sector Hermiticity verdicts.
        assert!(eigh.d2h_bytes >= bytes(eigenvalues), "{charges}: {eigh:?}");
        // `d` arrives filled in its one dense upload, beside `v`'s zeros; the
        // remainder is per-sector scalars. cuSOLVER's order is already the
        // published ascending one (#1985), so no `n_c x n_c` selector is
        // uploaded.
        let d_len = d.materialize(engine).unwrap().dense_data().unwrap().len();
        let v_len = v.materialize(engine).unwrap().dense_data().unwrap().len();
        assert!(
            eigh.h2d_bytes >= bytes(d_len + v_len),
            "{charges}: {eigh:?}"
        );
        assert!(
            eigh.h2d_bytes < bytes(2 * d_len + v_len),
            "{charges}: no selector upload: {eigh:?}"
        );
        let Eigh { d: expected, .. } = hermitian
            .eigh_full(
                engine,
                &codomain_axes(&hermitian),
                &domain_axes(&hermitian),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        for (device, host) in d
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize(engine).unwrap().dense_data().unwrap())
        {
            assert!((device - host).abs() <= 1e-10 * host.abs().max(1.0));
        }
    }
}
