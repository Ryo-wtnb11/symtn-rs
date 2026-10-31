//! The device SVD gauge phases against hand-computed phases: the
//! first largest-magnitude entry of each `U` column decides, including exact
//! magnitude ties, a maximum in the last row, and columns longer than a warp.
//!
//! Run with `cargo test -p symtn-dense --features cuda,cpu-faer --test
//! integration cuda_svd_gauge:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use num_complex::Complex64;
use symtn_dense::{
    cuda_svd_gauge_phases, cuda_transfer_stats, CudaDenseContext, CudaDenseStorage, CudaScalar,
    CudaSvdGaugeWeights,
};

/// The phases `cuda_svd_gauge_phases` computes for the column-major
/// `rows x k` matrix `u`, read off the diagonal of `right_selector`
/// (`diag(phase)`), and the `gauge_ops` the phases alone cost.
fn device_phases<D: CudaScalar + Copy>(u: Vec<D>, rows: usize, k: usize) -> (Vec<D>, u64) {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let storage = CudaDenseStorage::upload_members(&ctx, u, rows, k).unwrap();
    // More weights than rows: the prefix-view path of a shorter route.
    let weights = CudaSvdGaugeWeights::upload(&ctx, rows + 3).unwrap();
    let before = cuda_transfer_stats().gauge_ops;
    let phases = cuda_svd_gauge_phases::<D>(&mut ctx, &storage, rows, k, &weights).unwrap();
    let ops = cuda_transfer_stats().gauge_ops - before;
    let diagonal = phases
        .right_selector::<D>(&mut ctx)
        .unwrap()
        .download::<D>(&ctx)
        .unwrap();
    ((0..k).map(|j| diagonal[j + k * j]).collect(), ops)
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
#[ignore = "requires a real CUDA device"]
fn real_phases_take_the_first_of_tied_maxima() {
    // Columns: an exact tie whose first entry is negative; a maximum in the
    // last row; a positive pivot already.
    let u = vec![
        -0.5, 0.5, 0.3, //
        0.1, 0.2, -0.9, //
        0.3, -0.2, 0.1,
    ];
    let (phases, ops) = device_phases::<f64>(u, 3, 3);
    assert_eq!(phases, vec![-1.0, -1.0, 1.0]);
    assert_eq!(ops, 13);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn complex_phases_take_the_first_of_tied_maxima() {
    // An exact tie (|0.6i| == |-0.6|) with different phases: row 0 wins, so
    // the phase is i, not -1. Then a maximum in the last row.
    let u = vec![
        c(0.0, 0.6),
        c(-0.6, 0.0),
        c(0.1, 0.1), //
        c(0.1, 0.0),
        c(0.2, 0.0),
        c(0.0, -0.9),
    ];
    let (phases, ops) = device_phases::<Complex64>(u, 3, 2);
    for (actual, expected) in phases.iter().zip([c(0.0, 1.0), c(0.0, -1.0)]) {
        assert!(
            (actual - expected).norm() <= 1e-15,
            "{actual} vs {expected}"
        );
    }
    assert_eq!(ops, 13);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn phases_over_more_rows_than_a_warp() {
    // 40 rows: the column reductions take the multi-lane path. Column 0 ties
    // rows 5 and 37 (-0.7 first); column 1 peaks in the last row.
    let rows = 40;
    let column = |f: &dyn Fn(usize) -> f64| (0..rows).map(f).collect::<Vec<_>>();
    let tied = column(&|row| match row {
        5 => -0.7,
        37 => 0.7,
        _ => 0.01 * (row % 7) as f64,
    });
    let last = column(&|row| if row == rows - 1 { -0.8 } else { 0.05 });
    let real: Vec<f64> = tied.iter().chain(&last).copied().collect();
    let (phases, _) = device_phases::<f64>(real.clone(), rows, 2);
    assert_eq!(phases, vec![-1.0, -1.0]);

    // The same columns rotated by distinct phases: the pivot phase is the
    // first tied entry's (row 5), and the last row's.
    let (a, b) = (c(0.6, 0.8), c(0.0, 1.0));
    let complex: Vec<Complex64> = tied
        .iter()
        .map(|&x| a * x)
        .chain(last.iter().map(|&x| b * x))
        .collect();
    let (phases, _) = device_phases::<Complex64>(complex, rows, 2);
    for (actual, expected) in phases.iter().zip([-a, -b]) {
        assert!(
            (actual - expected).norm() <= 1e-15,
            "{actual} vs {expected}"
        );
    }
}
