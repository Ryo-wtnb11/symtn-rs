//! Adapter-level device tests over every admitted [`CudaScalar`] payload
//! (`f32`, `f64`, `Complex32`, `Complex64`), issue #1326.
//!
//! What this file owns that the `f64`/`Complex64` suites do not: the paths
//! that were typed by `f64` rather than by the payload's real lane — the
//! spectrum and reduction downloads, the normalizers of the Hermitian
//! test, and its tolerance — plus complex device QR, the last
//! dtype gate (tenferro-rs#1833, lifted by #1271).
//!
//! Oracles are host computations in **double precision** over the widened
//! fixture: a single-precision device answer is compared against `f64` host
//! arithmetic on the same values, never against a second single-precision run
//! and never against a symtn descriptor. Tolerances are written in epsilons of
//! the payload's own lane, so no assertion carries a platform-dependent
//! absolute constant.
//!
//! Run with `cargo test -p symtn-dense --no-default-features --features \
//! cuda,cpu-faer --test integration cuda_scalar_dtypes:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cuda_download_spectra, cuda_eigh_region, cuda_gemm_region_with_ops_into,
    cuda_hermitian_regions, cuda_qr_region, cuda_svd_region, CudaDenseContext, CudaDenseStorage,
    CudaScalar, DenseDType, DenseError, MatrixOp,
};

/// The one-region case of [`cuda_hermitian_regions`]; test support since no
/// production caller decides a single region (#1805).
fn cuda_is_hermitian_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<bool, symtn_dense::DenseError> {
    Ok(cuda_hermitian_regions::<D>(
        ctx,
        src,
        &[(offset, n)],
        <D::Real as symtn_dense::CudaRealScalar>::EPSILON.powf(0.75),
    )?[0])
}

/// The payload dtypes under test, with just enough host arithmetic for a
/// double-precision oracle over the widened fixture.
trait ProbeScalar: CudaScalar + Copy + Debug + PartialEq {
    const NAME: &'static str;
    /// Machine epsilon of this payload's real lane, widened.
    const EPSILON: f64;
    /// The scale window over which the device Hermitian rule is required to be
    /// scale-invariant for this payload: the real lane's smallest normal
    /// magnitude and `2^(MAX_EXP - 4)`, both exact powers of two, so scaling a
    /// fixture only moves exponents and the fixture's largest element (a few
    /// times the scale) is still finite. Real and complex payloads share the
    /// window: the rule normalizes by an exact power of two, never by a
    /// complex-by-real division that squares the divisor (tenferro-rs#1922).
    const TINY: f64;
    /// `2^(MAX_EXP - 4)`; not a `const` because `powi` is not a `const fn`.
    fn huge() -> f64;

    /// The payload value nearest `value`, i.e. the fixture as the device sees
    /// it. The oracle runs on `widen(narrow(z))`, so the comparison isolates
    /// the device's error rather than the fixture's rounding.
    fn narrow(value: Complex64) -> Self;
    fn widen(self) -> Complex64;
}

impl ProbeScalar for f32 {
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;
    const TINY: f64 = f32::MIN_POSITIVE as f64;
    fn huge() -> f64 {
        f64::from(2f32.powi(124))
    }

    fn narrow(value: Complex64) -> Self {
        value.re as Self
    }

    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl ProbeScalar for f64 {
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;
    const TINY: f64 = f64::MIN_POSITIVE;
    fn huge() -> f64 {
        2f64.powi(1020)
    }

    fn narrow(value: Complex64) -> Self {
        value.re
    }

    fn widen(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl ProbeScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    const EPSILON: f64 = f32::EPSILON as f64;
    const TINY: f64 = f32::MIN_POSITIVE as f64;
    fn huge() -> f64 {
        f64::from(2f32.powi(124))
    }

    fn narrow(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }

    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl ProbeScalar for Complex64 {
    const NAME: &'static str = "Complex64";
    const EPSILON: f64 = f64::EPSILON;
    const TINY: f64 = f64::MIN_POSITIVE;
    fn huge() -> f64 {
        2f64.powi(1020)
    }

    fn narrow(value: Complex64) -> Self {
        value
    }

    fn widen(self) -> Complex64 {
        self
    }
}

fn engine() -> symtn_dense::CudaEngine {
    symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 must be available for the device suite")
}

fn upload<D: ProbeScalar>(ctx: &CudaDenseContext, data: &[D]) -> CudaDenseStorage {
    CudaDenseStorage::upload::<D>(ctx, data).expect("upload")
}

fn download<D: ProbeScalar>(ctx: &CudaDenseContext, storage: &CudaDenseStorage) -> Vec<D> {
    storage.download::<D>(ctx).expect("download")
}

/// A deterministic, non-degenerate `rows x cols` column-major fixture, as the
/// payload sees it and as the oracle sees it.
fn fixture<D: ProbeScalar>(rows: usize, cols: usize, seed: f64) -> (Vec<D>, Vec<Complex64>) {
    let payload: Vec<D> = (0..rows * cols)
        .map(|index| {
            let index = index as f64;
            D::narrow(Complex64::new(
                seed + 0.5 + index * 0.25 - (index * index) * 0.03125,
                -0.375 + index * 0.125,
            ))
        })
        .collect();
    let wide = payload.iter().map(|value| value.widen()).collect();
    (payload, wide)
}

/// `k * sqrt(n) * eps(real(D))` of the magnitude the arithmetic actually
/// passed through — the sum of `|term|` over the `n` accumulated products,
/// never the magnitude of the *result*. A cancelling sum has a small result
/// and a large error budget, and pinning it to the result instead would make
/// this suite fail on the first GPU whose contraction order or FMA fusion
/// differs from this one's.
fn tolerance<D: ProbeScalar>(terms: usize, magnitude: f64) -> f64 {
    16.0 * (terms as f64).sqrt() * D::EPSILON * magnitude
}

/// `magnitudes[i]` is the sum of the absolute values the oracle accumulated
/// for element `i`, so the bound follows the arithmetic rather than the
/// answer.
fn assert_close<D: ProbeScalar>(
    actual: &[D],
    expected: &[Complex64],
    magnitudes: &[f64],
    terms: usize,
    what: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let allowed = tolerance::<D>(terms, magnitudes[index]);
        assert!(
            (got.widen() - want).norm() <= allowed,
            "{what} ({}): element {index} got {got:?} want {want} (allowed {allowed:e})",
            D::NAME
        );
    }
}

// ---------------------------------------------------------------------------
// GEMM regions: every operand-op pair, in every payload dtype.
// ---------------------------------------------------------------------------

/// The logical `m x k` operand a [`MatrixOp`] reads out of a packed parent.
fn operand(op: MatrixOp, parent: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    // `Identity` reads the parent as `rows x cols`; the other two read it as
    // `cols x rows` and transpose (and conjugate) it into `rows x cols`.
    (0..rows * cols)
        .map(|index| {
            let (row, col) = (index % rows, index / rows);
            match op {
                MatrixOp::Identity => parent[row + col * rows],
                MatrixOp::Transpose => parent[col + row * cols],
                MatrixOp::Adjoint => parent[col + row * cols].conj(),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn gemm_case<D: ProbeScalar>(
    ctx: &mut CudaDenseContext,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    m: usize,
    k: usize,
    n: usize,
    alpha: Complex64,
    beta: Complex64,
) {
    let (lhs_payload, lhs_wide) = fixture::<D>(m, k, 0.0);
    let (rhs_payload, rhs_wide) = fixture::<D>(k, n, 1.25);
    let (dst_payload, dst_wide) = fixture::<D>(m, n, -2.0);

    // Each operand sits behind a leading pad, so the regions are sub-regions
    // of longer buffers rather than whole allocations.
    let (lhs_offset, rhs_offset, dst_offset) = (5usize, 7usize, 3usize);
    let padded = |offset: usize, payload: &[D]| -> Vec<D> {
        let mut buffer = vec![D::narrow(Complex64::new(7.0, -3.0)); offset];
        buffer.extend_from_slice(payload);
        buffer.extend(std::iter::repeat_n(D::narrow(Complex64::new(7.0, -3.0)), 8));
        buffer
    };
    let lhs = upload::<D>(ctx, &padded(lhs_offset, &lhs_payload));
    let rhs = upload::<D>(ctx, &padded(rhs_offset, &rhs_payload));
    let mut dst = upload::<D>(ctx, &padded(dst_offset, &dst_payload));

    cuda_gemm_region_with_ops_into::<D>(
        ctx,
        &mut dst,
        dst_offset,
        &lhs,
        lhs_offset,
        &rhs,
        rhs_offset,
        m,
        k,
        n,
        lhs_op,
        rhs_op,
        D::narrow(alpha),
        D::narrow(beta),
    )
    .unwrap_or_else(|err| panic!("{} GEMM {lhs_op:?}/{rhs_op:?} failed: {err}", D::NAME));

    // Double-precision oracle over the widened operands, with the payload's
    // own rounding of alpha/beta so only the device's error is left.
    let alpha = D::narrow(alpha).widen();
    let beta = D::narrow(beta).widen();
    let lhs_view = operand(lhs_op, &lhs_wide, m, k);
    let rhs_view = operand(rhs_op, &rhs_wide, k, n);
    let mut magnitudes = vec![0.0; m * n];
    let expected: Vec<Complex64> = (0..m * n)
        .map(|index| {
            let (row, col) = (index % m, index / m);
            let mut acc = Complex64::new(0.0, 0.0);
            let mut magnitude = 0.0;
            for step in 0..k {
                let term = lhs_view[row + step * m] * rhs_view[step + col * k];
                acc += term;
                magnitude += term.norm();
            }
            magnitudes[index] = alpha.norm() * magnitude + beta.norm() * dst_wide[index].norm();
            alpha * acc + beta * dst_wide[index]
        })
        .collect();

    // The pad around the destination region must be untouched.
    let downloaded = download::<D>(ctx, &dst);
    let pad = D::narrow(Complex64::new(7.0, -3.0));
    assert!(
        downloaded[..dst_offset].iter().all(|v| *v == pad)
            && downloaded[dst_offset + m * n..].iter().all(|v| *v == pad),
        "GEMM {lhs_op:?}/{rhs_op:?} {m}x{k}x{n}: wrote outside its region"
    );
    assert_close::<D>(
        &downloaded[dst_offset..dst_offset + m * n],
        &expected,
        &magnitudes,
        k + 1,
        &format!("GEMM {lhs_op:?}/{rhs_op:?} {m}x{k}x{n}"),
    );
}

fn gemm_sweep<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let ops = [MatrixOp::Identity, MatrixOp::Transpose, MatrixOp::Adjoint];
    for lhs_op in ops {
        for rhs_op in ops {
            for &(m, k, n) in &[(3usize, 4usize, 2usize), (1, 6, 1), (5, 1, 4), (4, 4, 4)] {
                gemm_case::<D>(
                    ctx,
                    lhs_op,
                    rhs_op,
                    m,
                    k,
                    n,
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                );
                gemm_case::<D>(
                    ctx,
                    lhs_op,
                    rhs_op,
                    m,
                    k,
                    n,
                    Complex64::new(-0.75, 0.5),
                    Complex64::new(1.0, 0.0),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_gemm_matches_a_double_precision_oracle_for_every_dtype_and_op_pair() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    gemm_sweep::<f32>(&mut ctx);
    gemm_sweep::<f64>(&mut ctx);
    gemm_sweep::<Complex32>(&mut ctx);
    gemm_sweep::<Complex64>(&mut ctx);
}

// ---------------------------------------------------------------------------
// Factorization regions: the laws, and the widened spectra.
// ---------------------------------------------------------------------------

fn download_spectrum<D: ProbeScalar>(
    ctx: &mut CudaDenseContext,
    spectrum: symtn_dense::CudaSpectrum,
) -> Vec<f64> {
    cuda_download_spectra::<D>(ctx, &[spectrum])
        .expect("spectrum download")
        .remove(0)
}

/// `U diag(s) Vt == A`, `U^H U == I`, `Vt Vt^H == I`, and a spectrum that is
/// non-negative, descending and downloaded as `f64` whatever the lane.
fn svd_case<D: ProbeScalar>(ctx: &mut CudaDenseContext, rows: usize, cols: usize) {
    let k = rows.min(cols);
    let (payload, wide) = fixture::<D>(rows, cols, 0.0);
    let src = upload::<D>(ctx, &payload);

    let (u, values, vt) = cuda_svd_region::<D>(ctx, &src, 0, rows, cols)
        .unwrap_or_else(|err| panic!("{} SVD {rows}x{cols}: {err}", D::NAME));
    let values = download_spectrum::<D>(ctx, values);
    let u: Vec<Complex64> = download::<D>(ctx, &u).iter().map(|v| v.widen()).collect();
    let vt: Vec<Complex64> = download::<D>(ctx, &vt).iter().map(|v| v.widen()).collect();

    assert_eq!(values.len(), k);
    for window in values.windows(2) {
        assert!(
            window[0] >= window[1] && window[1] >= 0.0,
            "{}: singular values must be non-negative and descending: {values:?}",
            D::NAME
        );
    }

    let scale = values[0].max(1.0);
    let allowed = tolerance::<D>(rows * cols, scale);
    for col in 0..cols {
        for row in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for index in 0..k {
                acc += u[row + index * rows] * values[index] * vt[index + col * k];
            }
            let residual = (acc - wide[row + col * rows]).norm();
            assert!(
                residual <= allowed,
                "{}: SVD reconstruction at ({row}, {col}) off by {residual:e} (allowed {allowed:e})",
                D::NAME
            );
        }
    }
    assert_orthonormal_columns::<D>(&u, rows, k, "SVD U");
    assert_orthonormal_rows::<D>(&vt, k, cols, "SVD Vt");
}

/// `Q R == A`, `Q^H Q == I`, and the positive-diagonal gauge: `R_jj` real and
/// non-negative.
fn qr_case<D: ProbeScalar>(ctx: &mut CudaDenseContext, rows: usize, cols: usize) {
    let k = rows.min(cols);
    let (payload, wide) = fixture::<D>(rows, cols, 0.0);
    let src = upload::<D>(ctx, &payload);

    let (q, r) = cuda_qr_region::<D>(ctx, &src, 0, rows, cols)
        .unwrap_or_else(|err| panic!("{} QR {rows}x{cols}: {err}", D::NAME));
    let q: Vec<Complex64> = download::<D>(ctx, &q).iter().map(|v| v.widen()).collect();
    let r: Vec<Complex64> = download::<D>(ctx, &r).iter().map(|v| v.widen()).collect();

    let scale = wide
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.norm()));
    let allowed = tolerance::<D>(rows * cols, scale);
    for col in 0..cols {
        for row in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for index in 0..k {
                acc += q[row + index * rows] * r[index + col * k];
            }
            let residual = (acc - wide[row + col * rows]).norm();
            assert!(
                residual <= allowed,
                "{}: QR reconstruction at ({row}, {col}) off by {residual:e}",
                D::NAME
            );
        }
    }
    assert_orthonormal_columns::<D>(&q, rows, k, "QR Q");
    for index in 0..k {
        let diagonal = r[index + index * k];
        assert!(
            diagonal.re >= -allowed && diagonal.im.abs() <= allowed,
            "{}: positive-diagonal gauge broken at {index}: {diagonal}",
            D::NAME
        );
    }
}

/// `A v_j == lambda_j v_j`, eigenvalues real and ascending, vectors
/// orthonormal.
fn eigh_case<D: ProbeScalar>(ctx: &mut CudaDenseContext, n: usize) {
    let (payload, _) = fixture::<D>(n, n, 0.0);
    // Hermitian by construction: `(A + A^H) / 2`, formed in the payload dtype
    // so the device sees an exactly Hermitian block.
    let hermitian: Vec<D> = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let upper = payload[row + col * n].widen();
            let lower = payload[col + row * n].widen();
            D::narrow((upper + lower.conj()) * 0.5)
        })
        .collect();
    let wide: Vec<Complex64> = hermitian.iter().map(|value| value.widen()).collect();
    let src = upload::<D>(ctx, &hermitian);

    assert!(
        cuda_is_hermitian_region::<D>(ctx, &src, 0, n)
            .unwrap_or_else(|err| panic!("{}: hermitian test: {err}", D::NAME)),
        "{}: an exactly Hermitian block must be admitted",
        D::NAME
    );

    let (values, vectors) = cuda_eigh_region::<D>(ctx, &src, 0, n)
        .unwrap_or_else(|err| panic!("{} EIGH {n}x{n}: {err}", D::NAME));
    let values = download_spectrum::<D>(ctx, values);
    let vectors: Vec<Complex64> = download::<D>(ctx, &vectors)
        .iter()
        .map(|v| v.widen())
        .collect();

    assert_eq!(values.len(), n);
    for window in values.windows(2) {
        assert!(
            window[0] <= window[1],
            "{}: eigenvalues must ascend: {values:?}",
            D::NAME
        );
    }

    let scale = values
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    let allowed = tolerance::<D>(n * n, scale);
    for (index, &value) in values.iter().enumerate() {
        for row in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for step in 0..n {
                acc += wide[row + step * n] * vectors[step + index * n];
            }
            let residual = (acc - vectors[row + index * n] * value).norm();
            assert!(
                residual <= allowed,
                "{}: eigenpair {index} residual at row {row} is {residual:e}",
                D::NAME
            );
        }
    }
    assert_orthonormal_columns::<D>(&vectors, n, n, "EIGH vectors");
}

fn assert_orthonormal_columns<D: ProbeScalar>(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    what: &str,
) {
    let allowed = tolerance::<D>(rows, 1.0);
    for left in 0..cols {
        for right in 0..cols {
            let mut acc = Complex64::new(0.0, 0.0);
            for row in 0..rows {
                acc += matrix[row + left * rows].conj() * matrix[row + right * rows];
            }
            let want = Complex64::new(f64::from(u8::from(left == right)), 0.0);
            assert!(
                (acc - want).norm() <= allowed,
                "{what} ({}): columns {left},{right} give {acc} (allowed {allowed:e})",
                D::NAME
            );
        }
    }
}

fn assert_orthonormal_rows<D: ProbeScalar>(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
    what: &str,
) {
    let allowed = tolerance::<D>(cols, 1.0);
    for left in 0..rows {
        for right in 0..rows {
            let mut acc = Complex64::new(0.0, 0.0);
            for col in 0..cols {
                acc += matrix[left + col * rows] * matrix[right + col * rows].conj();
            }
            let want = Complex64::new(f64::from(u8::from(left == right)), 0.0);
            assert!(
                (acc - want).norm() <= allowed,
                "{what} ({}): rows {left},{right} give {acc} (allowed {allowed:e})",
                D::NAME
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_svd_obeys_its_laws_for_every_dtype() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    for &(rows, cols) in &[(5usize, 3usize), (3, 5), (4, 4)] {
        svd_case::<f32>(&mut ctx, rows, cols);
        svd_case::<f64>(&mut ctx, rows, cols);
        svd_case::<Complex32>(&mut ctx, rows, cols);
        svd_case::<Complex64>(&mut ctx, rows, cols);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_qr_obeys_its_laws_for_every_dtype() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    for &(rows, cols) in &[(5usize, 3usize), (3, 5), (4, 4)] {
        qr_case::<f32>(&mut ctx, rows, cols);
        qr_case::<f64>(&mut ctx, rows, cols);
        qr_case::<Complex32>(&mut ctx, rows, cols);
        qr_case::<Complex64>(&mut ctx, rows, cols);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_eigh_obeys_its_laws_for_every_dtype() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    for n in [4usize, 6] {
        eigh_case::<f32>(&mut ctx, n);
        eigh_case::<f64>(&mut ctx, n);
        eigh_case::<Complex32>(&mut ctx, n);
        eigh_case::<Complex64>(&mut ctx, n);
    }
}

// ---------------------------------------------------------------------------
// The real lane itself: spectra widened to f64, and the Hermitian normalizers
// typed by the payload rather than by f64.
// ---------------------------------------------------------------------------

/// A device spectrum of a single-precision payload comes back as `F32` and is
/// widened here, so the `f64` values symtn's truncation decisions consume
/// exist for every dtype. The oracle is the known spectrum of a diagonal
/// matrix, which no factorization has to be trusted for.
fn spectrum_widening_case<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let n = 4usize;
    let diagonal = [4.0_f64, 3.0, 2.0, 0.5];
    let payload: Vec<D> = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            D::narrow(Complex64::new(
                if row == col { diagonal[row] } else { 0.0 },
                0.0,
            ))
        })
        .collect();
    let src = upload::<D>(ctx, &payload);

    let (_, values, _) = cuda_svd_region::<D>(ctx, &src, 0, n, n).expect("svd");
    let values = download_spectrum::<D>(ctx, values);
    let allowed = tolerance::<D>(n, 4.0);
    for (index, value) in values.iter().enumerate() {
        assert!(
            (value - diagonal[index]).abs() <= allowed,
            "{}: singular value {index} is {value}, expected {}",
            D::NAME,
            diagonal[index]
        );
    }

    let (eigenvalues, _) = cuda_eigh_region::<D>(ctx, &src, 0, n).expect("eigh");
    let eigenvalues = download_spectrum::<D>(ctx, eigenvalues);
    let mut ascending = diagonal;
    ascending.sort_by(f64::total_cmp);
    for (index, value) in eigenvalues.iter().enumerate() {
        assert!(
            (value - ascending[index]).abs() <= allowed,
            "{}: eigenvalue {index} is {value}, expected {}",
            D::NAME,
            ascending[index]
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_spectra_are_widened_to_f64_for_every_dtype() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    spectrum_widening_case::<f32>(&mut ctx);
    spectrum_widening_case::<f64>(&mut ctx);
    spectrum_widening_case::<Complex32>(&mut ctx);
    spectrum_widening_case::<Complex64>(&mut ctx);
}

/// The Hermitian test multiplies the block by power-of-two normalizers in the
/// payload's own dtype, and reduces in the payload's real lane. Tenferro
/// rejects an operand of another dtype, so a block this test admits is
/// evidence that every operand is typed by the payload: the whole chain would
/// error out otherwise, not merely answer differently.
/// An `n x n` block that is Hermitian by construction — element `(row, col)`
/// and `(col, row)` are built from the same pair of magnitudes with the
/// imaginary part negated — scaled by an exact power of two so no rounding is
/// introduced by the scaling itself.
fn hermitian_block<D: ProbeScalar>(n: usize, scale: f64) -> Vec<D> {
    (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let (low, high) = (row.min(col), row.max(col));
            // Bounded regardless of `n`, so a large block stays in range for
            // the single-precision lane.
            let magnitude = 1.0 + 0.5 * ((low % 7) as f64) + 0.25 * ((high % 5) as f64);
            let magnitude_imaginary = 0.5 + ((low % 3) as f64);
            let imaginary = match row.cmp(&col) {
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Less => magnitude_imaginary,
                std::cmp::Ordering::Greater => -magnitude_imaginary,
            };
            D::narrow(Complex64::new(magnitude * scale, imaginary * scale))
        })
        .collect()
}

fn hermitian_admits_an_exactly_hermitian_block<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    // A small block, and one past the 512 the survey asks for: the device rule
    // reduces over the whole region, so block size is a real variable of it and
    // a single-precision sum over 512^2 elements is where a lane-typed
    // reduction would show up if it were still `f64`-shaped.
    for n in [3usize, 512] {
        let src = upload::<D>(ctx, &hermitian_block::<D>(n, 1.0));
        assert!(
            cuda_is_hermitian_region::<D>(ctx, &src, 0, n).expect("hermitian"),
            "{}: an exactly Hermitian {n}x{n} block must be admitted",
            D::NAME
        );
    }
}

/// Scale invariance and non-finite rejection, in the payload's own lane: the
/// rule normalizes by an exact power of two near the input maximum before
/// reducing, so an exactly Hermitian block stays Hermitian and a clearly
/// asymmetric one stays rejected at both ends of the lane's normal range, for
/// real and complex payloads alike. A NaN or infinity anywhere rejects rather
/// than propagating into the decision.
fn hermitian_extremes_case<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    let n = 4usize;
    for scale in [D::TINY, D::huge()] {
        let src = upload::<D>(ctx, &hermitian_block::<D>(n, scale));
        assert!(
            cuda_is_hermitian_region::<D>(ctx, &src, 0, n).expect("hermitian"),
            "{}: a Hermitian block scaled by {scale:e} must still be admitted",
            D::NAME
        );

        let mut asymmetric = hermitian_block::<D>(n, scale);
        // One off-diagonal pair broken by a whole unit of the block's own
        // magnitude: a relative defect no tolerance may absorb.
        asymmetric[1] = D::narrow(asymmetric[1].widen() + Complex64::new(scale, 0.0));
        let src = upload::<D>(ctx, &asymmetric);
        assert!(
            !cuda_is_hermitian_region::<D>(ctx, &src, 0, n).expect("hermitian"),
            "{}: a block asymmetric by a whole unit at scale {scale:e} must be rejected",
            D::NAME
        );
    }

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut poisoned = hermitian_block::<D>(n, 1.0);
        poisoned[0] = D::narrow(Complex64::new(bad, 0.0));
        let src = upload::<D>(ctx, &poisoned);
        assert!(
            !cuda_is_hermitian_region::<D>(ctx, &src, 0, n).expect("hermitian"),
            "{}: a non-finite ({bad}) block must be rejected",
            D::NAME
        );
    }
}

/// The device Hermitian admission at the default tolerance of the payload's
/// own lane, `eps(real(D))^(3/4)` (#1987): a block whose relative residual
/// sits at half of it is Hermitian, and one at twice it is not.
fn hermitian_tolerance_case<D: ProbeScalar>(ctx: &mut CudaDenseContext) {
    // [[1, delta], [0, 1]]: the relative residual is `delta / 2` up to a
    // negligible O(delta^2), so the rule changes truth value at
    // `delta = 2 * eps^(3/4)`.
    let threshold = 2.0 * D::EPSILON.powf(0.75);
    let block = |ctx: &CudaDenseContext, of_threshold: f64| {
        let delta = of_threshold * threshold;
        upload::<D>(
            ctx,
            &[
                D::narrow(Complex64::new(1.0, 0.0)),
                D::narrow(Complex64::new(0.0, 0.0)),
                D::narrow(Complex64::new(delta, 0.0)),
                D::narrow(Complex64::new(1.0, 0.0)),
            ],
        )
    };

    let below = block(ctx, 0.5);
    assert!(
        cuda_is_hermitian_region::<D>(ctx, &below, 0, 2).expect("below"),
        "{}: half the default threshold must be admitted",
        D::NAME
    );

    let above = block(ctx, 2.0);
    assert!(
        !cuda_is_hermitian_region::<D>(ctx, &above, 0, 2).expect("above"),
        "{}: twice the default threshold must be rejected",
        D::NAME
    );

    // A block that is non-Hermitian by a whole unit is rejected in any lane.
    let asymmetric = upload::<D>(
        ctx,
        &[
            D::narrow(Complex64::new(1.0, 0.0)),
            D::narrow(Complex64::new(0.0, 0.0)),
            D::narrow(Complex64::new(1.0, 0.0)),
            D::narrow(Complex64::new(1.0, 0.0)),
        ],
    );
    assert!(
        !cuda_is_hermitian_region::<D>(ctx, &asymmetric, 0, 2).expect("asymmetric"),
        "{}: a clearly non-Hermitian block must be rejected",
        D::NAME
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_hermitian_rule_scales_with_the_payload_real_lane() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    hermitian_admits_an_exactly_hermitian_block::<f32>(&mut ctx);
    hermitian_admits_an_exactly_hermitian_block::<f64>(&mut ctx);
    hermitian_admits_an_exactly_hermitian_block::<Complex32>(&mut ctx);
    hermitian_admits_an_exactly_hermitian_block::<Complex64>(&mut ctx);

    hermitian_tolerance_case::<f32>(&mut ctx);
    hermitian_tolerance_case::<f64>(&mut ctx);
    hermitian_tolerance_case::<Complex32>(&mut ctx);
    hermitian_tolerance_case::<Complex64>(&mut ctx);

    hermitian_extremes_case::<f32>(&mut ctx);
    hermitian_extremes_case::<f64>(&mut ctx);
    hermitian_extremes_case::<Complex32>(&mut ctx);
    hermitian_extremes_case::<Complex64>(&mut ctx);

    // The lane hazard, as a device decision rather than host arithmetic: a
    // residual an `f32` block is entitled to (half its default threshold) is
    // Hermitian in the single-precision lane and not in the double-precision
    // one — the one comparison that fails if the tolerance stops following
    // `D::Real`.
    let delta = f64::from(f32::EPSILON).powf(0.75);
    let single = |ctx: &CudaDenseContext| upload::<f32>(ctx, &[1.0, 0.0, delta as f32, 1.0]);
    let double = |ctx: &CudaDenseContext| upload::<f64>(ctx, &[1.0, 0.0, delta, 1.0]);
    let single = single(&ctx);
    let double = double(&ctx);
    assert!(
        cuda_is_hermitian_region::<f32>(&mut ctx, &single, 0, 2).expect("f32"),
        "the f32 lane must admit its own rounding residual"
    );
    assert!(
        !cuda_is_hermitian_region::<f64>(&mut ctx, &double, 0, 2).expect("f64"),
        "the f64 lane must reject the same residual, which is what makes the lane the decision"
    );
}

// ---------------------------------------------------------------------------
// Capability boundary: Complex32 device QR.
// ---------------------------------------------------------------------------

/// A dtype mismatch stays a typed error for the new payloads too, rather than
/// a reinterpretation of the bytes: `f32` and `Complex32` buffers are half the
/// width of their double-precision twins.
#[test]
#[ignore = "requires a real CUDA device"]
fn every_admitted_dtype_reports_its_own_tag_and_rejects_the_others() {
    let mut engine = engine();
    let ctx = CudaDenseContext::new(&mut engine).unwrap();
    let single = upload::<f32>(&ctx, &[1.0f32, 2.0, 3.0, 4.0]);
    let single_complex = upload::<Complex32>(&ctx, &[Complex32::new(1.0, -1.0); 4]);

    assert_eq!(single.dtype(), DenseDType::F32);
    assert_eq!(single_complex.dtype(), DenseDType::C32);
    assert!(matches!(
        single.download::<f64>(&ctx),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::F64,
            actual: DenseDType::F32,
            ..
        })
    ));
    assert!(matches!(
        single_complex.download::<Complex64>(&ctx),
        Err(DenseError::DTypeMismatch {
            expected: DenseDType::C64,
            actual: DenseDType::C32,
            ..
        })
    ));
}
