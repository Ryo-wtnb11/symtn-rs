//! One spectrum download per factorization call: the spectra of many
//! SVD or EIGH regions come back with a single device-to-host transfer, and
//! bitwise equal to downloading each spectrum on its own (the per-block
//! download path), so gathering them on device only moves data.
//!
//! The deltas read the calling thread's [`cuda_transfer_stats`] counters.
//!
//! Run with `cargo test -p symtn-dense --no-default-features --features \
//! cuda,cpu-faer --test integration cuda_spectrum_download:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cuda_download_spectra, cuda_eigh_region, cuda_svd_region, cuda_transfer_stats,
    CudaDenseContext, CudaDenseStorage, CudaScalar, CudaSpectrum,
};

trait Payload: CudaScalar + Copy {
    fn make(re: f64, im: f64) -> Self;
}

impl Payload for f32 {
    fn make(re: f64, _: f64) -> Self {
        re as f32
    }
}

impl Payload for f64 {
    fn make(re: f64, _: f64) -> Self {
        re
    }
}

impl Payload for Complex32 {
    fn make(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}

impl Payload for Complex64 {
    fn make(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

/// Hermitian `n x n` blocks of distinct sizes packed column-major into one
/// buffer, `copies` times over; returns the buffer and the `(offset, n)`s.
fn packed<D: Payload>(copies: usize) -> (Vec<D>, Vec<(usize, usize)>) {
    let mut data = Vec::new();
    let mut regions = Vec::new();
    for copy in 0..copies {
        for n in [1, 3, 4, 7] {
            regions.push((data.len(), n));
            data.extend((0..n * n).map(|index| {
                let (row, col) = (index % n, index / n);
                let (low, high) = (row.min(col), row.max(col));
                let re = 1.0 + copy as f64 + 0.5 * (low % 5) as f64 + 0.25 * (high % 3) as f64;
                let im = match row.cmp(&col) {
                    std::cmp::Ordering::Equal => 0.0,
                    std::cmp::Ordering::Less => 0.5,
                    std::cmp::Ordering::Greater => -0.5,
                };
                D::make(re, im)
            }));
        }
    }
    (data, regions)
}

fn bits(spectra: &[Vec<f64>]) -> Vec<Vec<u64>> {
    spectra
        .iter()
        .map(|values| values.iter().map(|value| value.to_bits()).collect())
        .collect()
}

/// Downloads `spectra` together, then each alone, returning the batched
/// download's D2H count.
fn check<D: Payload>(
    ctx: &mut CudaDenseContext,
    spectra: &[CudaSpectrum],
    expected_lengths: &[usize],
    label: &str,
) -> u64 {
    let before = cuda_transfer_stats().d2h_calls;
    let batched = cuda_download_spectra::<D>(ctx, spectra).expect("batched download");
    let downloads = cuda_transfer_stats().d2h_calls - before;

    let lengths: Vec<_> = batched.iter().map(Vec::len).collect();
    assert_eq!(lengths, expected_lengths, "{label}: spectrum lengths");
    let alone: Vec<_> = spectra
        .iter()
        .map(|spectrum| {
            cuda_download_spectra::<D>(ctx, std::slice::from_ref(spectrum))
                .expect("single download")
                .remove(0)
        })
        .collect();
    assert_eq!(bits(&batched), bits(&alone), "{label}: bitwise values");
    downloads
}

fn case<D: Payload>(ctx: &mut CudaDenseContext, name: &str) {
    for copies in [1, 3] {
        let (data, regions) = packed::<D>(copies);
        let src = CudaDenseStorage::upload::<D>(ctx, &data).expect("upload");
        let lengths: Vec<_> = regions.iter().map(|&(_, n)| n).collect();

        let singular: Vec<_> = regions
            .iter()
            .map(|&(offset, n)| {
                cuda_svd_region::<D>(ctx, &src, offset, n, n)
                    .expect("svd")
                    .1
            })
            .collect();
        let eigen: Vec<_> = regions
            .iter()
            .map(|&(offset, n)| cuda_eigh_region::<D>(ctx, &src, offset, n).expect("eigh").0)
            .collect();

        let label = format!("{name} x{copies}");
        let svd = check::<D>(ctx, &singular, &lengths, &format!("{label} SVD"));
        let eigh = check::<D>(ctx, &eigen, &lengths, &format!("{label} EIGH"));
        assert_eq!(
            (svd, eigh),
            (1, 1),
            "{label}: one download per call, whatever the region count"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn many_spectra_download_once_and_bitwise_unchanged() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).expect("CUDA device 0");
    case::<f32>(&mut ctx, "f32");
    case::<f64>(&mut ctx, "f64");
    case::<Complex32>(&mut ctx, "Complex32");
    case::<Complex64>(&mut ctx, "Complex64");

    let before = cuda_transfer_stats().d2h_calls;
    assert!(cuda_download_spectra::<f64>(&mut ctx, &[])
        .expect("no spectra")
        .is_empty());
    assert_eq!(
        cuda_transfer_stats().d2h_calls,
        before,
        "no spectra, no downloads"
    );
}
