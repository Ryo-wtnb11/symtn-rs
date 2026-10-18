//! Device probe for single-precision (`f32` / `Complex32`) payloads on
//! Tenferro CUDA.
//!
//! Tenferro's sources make F32/C32 coverage symmetric with F64/C64 in every
//! CUDA file symtn reaches. That is a source-level expectation: NVRTC
//! compiles a fresh kernel per dtype, cuTENSOR selects a different algorithm
//! per dtype, and TensorKit's own CUDA suite records a Float32 cuTENSOR
//! failure for the outer product (`test/cuda/tensors.jl:523`). This file turns
//! the expectation into device evidence, or records the exact rejection text
//! where Tenferro declines.
//!
//! The operation list is what the symtn-dense CUDA adapter calls, one probe
//! per call a single-precision `CudaScalar` reaches:
//!
//! | Adapter site | Tenferro call |
//! |---|---|
//! | `CudaDenseStorage::upload_owned` / `download` | `upload_tensor` / `download_tensor` |
//! | `region_view*` | `TypedTensor::backend_region_view{,_mut}` |
//! | `cuda_gemm_region_strided_into` | `dot_general_read_into_accum` (+ `lhs_conj`/`rhs_conj`, alpha/beta) |
//! | `cuda_axpby_owned` / `weighted_inner_cuda` (`symtn/src/typed.rs:11529`, `:11760`) | the same call with a 1x1 operand and an `Adjoint` row view |
//! | `cuda_is_hermitian_region` | `to_contiguous_read`, `abs`, `conj`, `div`, `sub`, `reduce_max`, `reduce_sum_squares_read` |
//! | `upload_scalar` / `download_values` | rank-0 `upload_tensor`; `download_tensor` of a real metadata tensor |
//! | `cuda_copy_region_into` | `copy_read_into` |
//! | `cuda_svd_region` / `cuda_qr_region` / `cuda_eigh_region` | `svd_read` / `qr_with_options_read(PositiveDiagonal)` / `eigh_read` under `with_backend_session` |
//! | (not reached by symtn; #1065 gate) | `solve`, `lu` |
//!
//! Oracles are independent host loops in the *same* dtype, never a symtn
//! descriptor and never the f64 device path. Pure moves (`alpha = 1`,
//! `beta = 0`/`1` with no multiply) are compared bitwise; arithmetic is
//! compared at `k * sqrt(n) * eps(real(T))`, and factorization laws at a
//! `sqrt(eps)`-scaled bound.
//!
//! Tenferro is called directly, so the probe adds no production surface and
//! isolates Tenferro's behaviour from symtn's adapter.
//! The device is the unit under test, so every test is `#[ignore]`.

#![cfg(feature = "cuda")]

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_linalg::{QrGauge, QrOptions, TensorLinalgExt, TensorReadLinalgExt};
use tenferro_tensor::{
    BackendSessionHost, ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, Tensor,
    TensorDot, TensorElementwise, TensorRead, TensorReduction, TensorScalar, TensorStructural,
    TensorWrite, TypedTensor,
};

// ---------------------------------------------------------------------------
// Probe scalar: the two dtypes under test, plus the host arithmetic the
// oracles need. Deliberately NOT `symtn_dense::CudaScalar`, so the probe does
// not depend on the adapter it checks.
// ---------------------------------------------------------------------------

trait ProbeScalar:
    TensorScalar
    + Copy
    + Debug
    + PartialEq
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    const NAME: &'static str;
    const ZERO: Self;
    const ONE: Self;
    const IS_COMPLEX: bool;
    /// `eps` of the real component type, the tolerance unit for this dtype.
    const EPS: f64;
    /// Dtype the real-valued device metadata (`abs`, `reduce_max`,
    /// `reduce_sum_squares`, SVD/EIGH spectra) is expected to carry.
    const REAL_DTYPE: DType;

    fn from_parts(re: f64, im: f64) -> Self;
    fn re(self) -> f64;
    fn conj(self) -> Self;
    fn magnitude(self) -> f64;
    fn contraction_scalar(self) -> ContractionScalar;
    fn typed(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        tensor.as_typed::<Self>()
    }
    fn typed_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
        tensor.as_typed_mut::<Self>()
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).magnitude()
    }

    /// Deterministic, non-degenerate sample values on a quarter-integer grid,
    /// so every sample is exactly representable in `f32` and a pure device
    /// move can be asserted bitwise.
    fn sample(index: usize) -> Self {
        let re = 1.0 + ((index % 23) as f64) * 0.25;
        let im = -0.5 + ((index % 17) as f64) * 0.125;
        Self::from_parts(re, im)
    }
}

impl ProbeScalar for f32 {
    const NAME: &'static str = "f32";
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;
    const EPS: f64 = f32::EPSILON as f64;
    const REAL_DTYPE: DType = DType::F32;

    fn from_parts(re: f64, _im: f64) -> Self {
        re as f32
    }

    fn re(self) -> f64 {
        self as f64
    }

    fn conj(self) -> Self {
        self
    }

    fn magnitude(self) -> f64 {
        (self as f64).abs()
    }

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::F32(self)
    }
}

impl ProbeScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    const ZERO: Self = Complex32::new(0.0, 0.0);
    const ONE: Self = Complex32::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;
    const EPS: f64 = f32::EPSILON as f64;
    const REAL_DTYPE: DType = DType::F32;

    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }

    fn re(self) -> f64 {
        self.re as f64
    }

    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    fn magnitude(self) -> f64 {
        Complex64::new(self.re as f64, self.im as f64).norm()
    }

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::C32(self)
    }
}

// ---------------------------------------------------------------------------
// Probe harness.
// ---------------------------------------------------------------------------

struct Probe {
    backend: CudaBackend,
}

impl Probe {
    fn new() -> Self {
        let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0))
            .expect("CUDA device 0 must be available for the device probe");
        Self { backend }
    }

    fn upload_flat<D: ProbeScalar>(&mut self, data: Vec<D>) -> Tensor {
        let len = data.len();
        self.upload_shaped(vec![len], data)
    }

    fn upload_shaped<D: TensorScalar>(&mut self, shape: Vec<usize>, data: Vec<D>) -> Tensor {
        let host = Tensor::from_vec_col_major(shape, data).expect("host tensor");
        upload_tensor(self.backend.runtime(), &host).expect("upload")
    }

    fn download<D: ProbeScalar>(&mut self, tensor: &Tensor) -> Vec<D> {
        assert_eq!(
            tensor.dtype(),
            D::dtype(),
            "device tensor dtype must survive the round trip"
        );
        let host = download_tensor(self.backend.runtime(), tensor).expect("download");
        D::into_typed(host)
            .expect("dtype")
            .into_host_vec()
            .expect("host vec")
    }

    /// The production `download_values` shape: a small real metadata tensor
    /// (spectrum, reduction) widened to `f64` on the host. The adapter's own
    /// helper widens the real lane to `f64`; this records that boundary.
    fn download_real(&mut self, tensor: &Tensor, what: &str) -> Vec<f64> {
        let dtype = tensor.dtype();
        let host = download_tensor(self.backend.runtime(), tensor).expect("download");
        let values = match dtype {
            DType::F32 => host
                .into_typed::<f32>()
                .and_then(TypedTensor::into_host_vec)
                .expect("host vec")
                .into_iter()
                .map(f64::from)
                .collect(),
            DType::F64 => host
                .into_typed::<f64>()
                .and_then(TypedTensor::into_host_vec)
                .expect("host vec"),
            other => panic!("{what}: unexpected metadata dtype {other:?}"),
        };
        println!("  {what}: dtype {dtype:?}, {} value(s)", values.len());
        values
    }
}

/// Flat buffer positions of a strided region, in column-major index order.
fn region_offsets(dims: &[usize], strides: &[isize], base: isize) -> Vec<usize> {
    let total: usize = dims.iter().product();
    (0..total)
        .map(|flat| {
            let mut rest = flat;
            let mut offset = base;
            for (dim, stride) in dims.iter().zip(strides) {
                offset += (rest % dim) as isize * stride;
                rest /= dim;
            }
            usize::try_from(offset).expect("nonnegative region offset")
        })
        .collect()
}

/// Strides of a column-major layout over `dims` permuted by `order`:
/// axis `order[k]` is the k-th fastest-varying axis.
fn permuted_strides(dims: &[usize], order: &[usize], leading: isize) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    let mut running = leading;
    for &axis in order {
        strides[axis] = running;
        running *= dims[axis] as isize;
    }
    strides
}

fn assert_bitwise<D: ProbeScalar>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(got, want, "{what}: element {index} ({})", D::NAME);
    }
}

/// `k * sqrt(n) * eps(real(D))`, relative to the larger of 1 and the expected
/// magnitude. `terms` is the number of accumulated products behind one output.
fn assert_close<D: ProbeScalar>(actual: &[D], expected: &[D], terms: usize, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let unit = 16.0 * (terms.max(1) as f64).sqrt() * D::EPS;
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let bound = unit * 1.0_f64.max(want.magnitude());
        assert!(
            got.distance(*want) <= bound,
            "{what}: element {index} ({}) got {got:?} want {want:?} (bound {bound:e})",
            D::NAME
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Round trip: upload / download in the payload dtype, bitwise.
// ---------------------------------------------------------------------------

fn round_trip_case<D: ProbeScalar>(probe: &mut Probe) {
    let host: Vec<D> = (0..257).map(D::sample).collect();
    let device = probe.upload_flat(host.clone());
    assert_eq!(device.dtype(), D::dtype());
    let back = probe.download::<D>(&device);
    assert_bitwise(&back, &host, "upload/download round trip");
    println!("upload/download {}: bitwise round trip ok", D::NAME);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn upload_download_round_trip_f32() {
    round_trip_case::<f32>(&mut Probe::new());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn upload_download_round_trip_c32() {
    round_trip_case::<Complex32>(&mut Probe::new());
}

// ---------------------------------------------------------------------------
// 2. Region GEMM with operand ops -- the `cuda_gemm_region_with_ops_into`
//    shape, including the `Adjoint` conjugation flag and alpha/beta.
// ---------------------------------------------------------------------------

/// `cuda_adapter::cuda_operand_view`, reproduced so the probe drives exactly
/// the strides and conjugation flag production would.
fn operand_view(op: &str, rows: usize, cols: usize) -> ([isize; 2], bool) {
    match op {
        "Identity" => ([1, rows as isize], false),
        "Transpose" => ([cols as isize, 1], false),
        "Adjoint" => ([cols as isize, 1], true),
        other => panic!("unknown operand op {other}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_case<D: ProbeScalar>(
    probe: &mut Probe,
    m: usize,
    k: usize,
    n: usize,
    lhs_op: &str,
    rhs_op: &str,
    alpha: D,
    beta: D,
) {
    let (lhs_strides, lhs_conj) = operand_view(lhs_op, m, k);
    let (rhs_strides, rhs_conj) = operand_view(rhs_op, k, n);
    // Offsets and a longer buffer, so the region really is a sub-region.
    let (lhs_base, rhs_base, dst_base) = (5isize, 7isize, 3isize);
    let lhs_host: Vec<D> = (0..(lhs_base as usize + m * k + 8))
        .map(D::sample)
        .collect();
    let rhs_host: Vec<D> = (0..(rhs_base as usize + k * n + 8))
        .map(|i| D::sample(i + 40))
        .collect();
    let dst_host: Vec<D> = (0..(dst_base as usize + m * n + 8))
        .map(|i| D::sample(i + 90))
        .collect();

    let lhs_dev = probe.upload_flat(lhs_host.clone());
    let rhs_dev = probe.upload_flat(rhs_host.clone());
    let mut dst_dev = probe.upload_flat(dst_host.clone());

    let lhs_view = D::typed(&lhs_dev)
        .expect("lhs dtype")
        .backend_region_view(vec![m, k], lhs_strides.to_vec(), lhs_base)
        .expect("lhs view");
    let rhs_view = D::typed(&rhs_dev)
        .expect("rhs dtype")
        .backend_region_view(vec![k, n], rhs_strides.to_vec(), rhs_base)
        .expect("rhs view");
    let dst_view = D::typed_mut(&mut dst_dev)
        .expect("dst dtype")
        .backend_region_view_mut(vec![m, n], vec![1, m as isize], dst_base)
        .expect("dst view");
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj,
        rhs_conj,
        alpha: alpha.contraction_scalar(),
        beta: beta.contraction_scalar(),
    };
    probe
        .backend
        .dot_general_read_into_accum(
            TensorRead::from_view(D::tensor_view(lhs_view)),
            TensorRead::from_view(D::tensor_view(rhs_view)),
            &config,
            accumulation,
            TensorWrite::from_view(D::tensor_view_mut(dst_view)),
        )
        .unwrap_or_else(|err| {
            panic!(
                "{} GEMM {lhs_op}/{rhs_op} {m}x{k}x{n} failed: {err}",
                D::NAME
            )
        });

    // Independent host oracle over the same strided index space.
    let mut expected = dst_host.clone();
    for j in 0..n {
        for i in 0..m {
            let mut acc = D::ZERO;
            for p in 0..k {
                let a_at =
                    (lhs_base + i as isize * lhs_strides[0] + p as isize * lhs_strides[1]) as usize;
                let b_at =
                    (rhs_base + p as isize * rhs_strides[0] + j as isize * rhs_strides[1]) as usize;
                let a = if lhs_conj {
                    lhs_host[a_at].conj()
                } else {
                    lhs_host[a_at]
                };
                let b = if rhs_conj {
                    rhs_host[b_at].conj()
                } else {
                    rhs_host[b_at]
                };
                acc = acc + a * b;
            }
            let at = dst_base as usize + i + j * m;
            expected[at] = alpha * acc + beta * expected[at];
        }
    }

    let actual = probe.download::<D>(&dst_dev);
    assert_close(
        &actual,
        &expected,
        k,
        &format!(
            "{} GEMM {lhs_op}/{rhs_op} {m}x{k}x{n} alpha {alpha:?} beta {beta:?}",
            D::NAME
        ),
    );
}

fn gemm_sweep<D: ProbeScalar>(probe: &mut Probe) {
    for (m, k, n) in [(3usize, 4usize, 2usize), (1, 6, 1), (5, 1, 4), (4, 4, 4)] {
        for lhs_op in ["Identity", "Transpose", "Adjoint"] {
            for rhs_op in ["Identity", "Transpose", "Adjoint"] {
                for beta in [D::ZERO, D::ONE] {
                    gemm_case::<D>(probe, m, k, n, lhs_op, rhs_op, D::ONE, beta);
                    gemm_case::<D>(
                        probe,
                        m,
                        k,
                        n,
                        lhs_op,
                        rhs_op,
                        D::from_parts(-0.75, 0.5),
                        beta,
                    );
                }
            }
        }
    }
    println!("region GEMM {}: all operand ops, alpha/beta ok", D::NAME);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_gemm_with_ops_f32() {
    gemm_sweep::<f32>(&mut Probe::new());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn region_gemm_with_ops_c32() {
    gemm_sweep::<Complex32>(&mut Probe::new());
}

// ---------------------------------------------------------------------------
// 3. Strided N-D region accumulation against a 1x1 ones operand -- the
//    #1298/#1301 form behind `cuda_axpby_owned` and the structural transforms.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn nd_region_case<D: ProbeScalar>(
    probe: &mut Probe,
    dims: &[usize],
    src_order: &[usize],
    dst_order: &[usize],
    alpha: D,
    beta: D,
    conj: bool,
) {
    let block: usize = dims.iter().product();
    let src_strides = permuted_strides(dims, src_order, 2);
    let dst_strides = permuted_strides(dims, dst_order, 3);
    let src_base = 7isize;
    let dst_base = 11isize;
    let src_host: Vec<D> = (0..(16 + 4 * block)).map(D::sample).collect();
    let dst_host: Vec<D> = (0..(16 + 6 * block)).map(|i| D::sample(i + 300)).collect();

    let src = probe.upload_flat(src_host.clone());
    let ones = probe.upload_flat(vec![D::ONE]);
    let mut dst = probe.upload_flat(dst_host.clone());

    let mut view_dims = dims.to_vec();
    view_dims.push(1);
    let mut lhs_strides = src_strides.clone();
    lhs_strides.push(1);
    let mut out_strides = dst_strides.clone();
    out_strides.push(1);

    let lhs = D::typed(&src)
        .expect("src dtype")
        .backend_region_view(view_dims.clone(), lhs_strides, src_base)
        .expect("src view");
    let rhs = D::typed(&ones)
        .expect("ones dtype")
        .backend_region_view(vec![1, 1], vec![1, 1], 0)
        .expect("ones view");
    let out = D::typed_mut(&mut dst)
        .expect("dst dtype")
        .backend_region_view_mut(view_dims, out_strides, dst_base)
        .expect("dst view");
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![dims.len()],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    probe
        .backend
        .dot_general_read_into_accum(
            TensorRead::from_view(D::tensor_view(lhs)),
            TensorRead::from_view(D::tensor_view(rhs)),
            &config,
            DotGeneralAccumulation {
                lhs_conj: conj,
                rhs_conj: false,
                alpha: alpha.contraction_scalar(),
                beta: beta.contraction_scalar(),
            },
            TensorWrite::from_view(D::tensor_view_mut(out)),
        )
        .unwrap_or_else(|err| panic!("{} rank {} accumulate failed: {err}", D::NAME, dims.len()));

    let mut expected = dst_host.clone();
    for (&from, &to) in region_offsets(dims, &src_strides, src_base)
        .iter()
        .zip(&region_offsets(dims, &dst_strides, dst_base))
    {
        let value = if conj {
            src_host[from].conj()
        } else {
            src_host[from]
        };
        expected[to] = alpha * value + beta * expected[to];
    }

    let actual = probe.download::<D>(&dst);
    let what = format!(
        "{} rank {} dims {dims:?} conj {conj} alpha {alpha:?} beta {beta:?}",
        D::NAME,
        dims.len()
    );
    if alpha == D::ONE && (beta == D::ZERO || beta == D::ONE) {
        assert_bitwise(&actual, &expected, &what);
    } else {
        assert_close(&actual, &expected, 1, &what);
    }
}

const ND_CASES: &[(&[usize], &[usize], &[usize])] = &[
    (&[2, 3, 4], &[0, 1, 2], &[2, 0, 1]),
    (&[3, 2, 2, 3], &[0, 1, 2, 3], &[1, 3, 0, 2]),
    (&[2, 2, 3, 2, 2], &[0, 1, 2, 3, 4], &[4, 2, 0, 3, 1]),
];

fn nd_region_sweep<D: ProbeScalar>(probe: &mut Probe) {
    for &(dims, src_order, dst_order) in ND_CASES {
        for conj in [false, true] {
            for beta in [D::ZERO, D::ONE] {
                nd_region_case::<D>(probe, dims, src_order, dst_order, D::ONE, beta, conj);
                nd_region_case::<D>(
                    probe,
                    dims,
                    src_order,
                    dst_order,
                    D::from_parts(-0.75, 0.25),
                    beta,
                    conj,
                );
            }
        }
    }
    println!("N-D strided region accumulation {}: ok", D::NAME);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_strided_region_accumulation_f32() {
    nd_region_sweep::<f32>(&mut Probe::new());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_strided_region_accumulation_c32() {
    nd_region_sweep::<Complex32>(&mut Probe::new());
}

// ---------------------------------------------------------------------------
// 4. Outer product (no contracted modes). Called out explicitly because
//    TensorKit's CUDA suite records a Float32 cuTENSOR failure for `otimes`
//    (`test/cuda/tensors.jl:523`).
// ---------------------------------------------------------------------------

fn outer_product_case<D: ProbeScalar>(probe: &mut Probe) {
    let lhs_dims = [2usize, 3];
    let rhs_dims = [3usize, 2];
    let out_dims = [2usize, 3, 3, 2];
    let lhs_strides = permuted_strides(&lhs_dims, &[1, 0], 2);
    let rhs_strides = permuted_strides(&rhs_dims, &[0, 1], 3);
    let out_strides = permuted_strides(&out_dims, &[2, 0, 3, 1], 2);
    let (lhs_base, rhs_base, out_base) = (5isize, 4isize, 7isize);

    let lhs_host: Vec<D> = (0..64).map(D::sample).collect();
    let rhs_host: Vec<D> = (0..64).map(|i| D::sample(i + 200)).collect();
    let out_host: Vec<D> = (0..256).map(|i| D::sample(i + 500)).collect();

    let lhs_dev = probe.upload_flat(lhs_host.clone());
    let rhs_dev = probe.upload_flat(rhs_host.clone());
    let mut out_dev = probe.upload_flat(out_host.clone());

    let alpha = D::from_parts(0.5, -0.25);
    let beta = D::ONE;
    let lhs_view = D::typed(&lhs_dev)
        .expect("lhs dtype")
        .backend_region_view(lhs_dims.to_vec(), lhs_strides.clone(), lhs_base)
        .expect("lhs view");
    let rhs_view = D::typed(&rhs_dev)
        .expect("rhs dtype")
        .backend_region_view(rhs_dims.to_vec(), rhs_strides.clone(), rhs_base)
        .expect("rhs view");
    let out_view = D::typed_mut(&mut out_dev)
        .expect("out dtype")
        .backend_region_view_mut(out_dims.to_vec(), out_strides.clone(), out_base)
        .expect("out view");
    let result = probe.backend.dot_general_read_into_accum(
        TensorRead::from_view(D::tensor_view(lhs_view)),
        TensorRead::from_view(D::tensor_view(rhs_view)),
        &DotGeneralConfig {
            lhs_contracting_dims: Vec::new(),
            rhs_contracting_dims: Vec::new(),
            lhs_batch_dims: Vec::new(),
            rhs_batch_dims: Vec::new(),
        },
        DotGeneralAccumulation {
            lhs_conj: true,
            rhs_conj: false,
            alpha: alpha.contraction_scalar(),
            beta: beta.contraction_scalar(),
        },
        TensorWrite::from_view(D::tensor_view_mut(out_view)),
    );
    if let Err(err) = result {
        panic!(
            "{} outer product (no contracted modes) FAILED with `{err}` -- this is the \
             TensorKit `test/cuda/tensors.jl:523` failure class reproduced in Tenferro",
            D::NAME
        );
    }

    let mut expected = out_host.clone();
    let lhs_positions = region_offsets(&lhs_dims, &lhs_strides, lhs_base);
    let rhs_positions = region_offsets(&rhs_dims, &rhs_strides, rhs_base);
    let out_positions = region_offsets(&out_dims, &out_strides, out_base);
    for (rhs_flat, &rhs_at) in rhs_positions.iter().enumerate() {
        for (lhs_flat, &lhs_at) in lhs_positions.iter().enumerate() {
            let at = out_positions[rhs_flat * lhs_positions.len() + lhs_flat];
            expected[at] = alpha * lhs_host[lhs_at].conj() * rhs_host[rhs_at] + beta * expected[at];
        }
    }

    let actual = probe.download::<D>(&out_dev);
    assert_close(
        &actual,
        &expected,
        1,
        &format!("{} outer product into permuted strides", D::NAME),
    );
    println!("outer product {}: ok", D::NAME);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn outer_product_f32() {
    outer_product_case::<f32>(&mut Probe::new());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn outer_product_c32() {
    outer_product_case::<Complex32>(&mut Probe::new());
}

// ---------------------------------------------------------------------------
// 5. The `cuda_is_hermitian_region` kernel chain: `to_contiguous_read`, `abs`,
//    `conj`, `div`, `sub`, `reduce_max`, `reduce_sum_squares_read`, plus the
//    scalar upload/download the chain needs. Each NVRTC kernel family here is
//    compiled per dtype, so this is the elementwise/reduction evidence.
// ---------------------------------------------------------------------------

fn hermitian_chain_case<D: ProbeScalar>(probe: &mut Probe) {
    const N: usize = 4;
    // A genuinely Hermitian block, so the chain's residual is exactly zero,
    // plus a perturbed copy so the residual path is non-degenerate too.
    let mut hermitian = vec![D::ZERO; N * N];
    for col in 0..N {
        for row in 0..N {
            let value = if row <= col {
                D::from_parts(
                    1.0 + (row * N + col) as f64 * 0.25,
                    if row == col {
                        0.0
                    } else {
                        0.125 * (col - row) as f64
                    },
                )
            } else {
                D::ZERO
            };
            hermitian[col * N + row] = value;
        }
    }
    for col in 0..N {
        for row in (col + 1)..N {
            hermitian[col * N + row] = hermitian[row * N + col].conj();
        }
    }

    let offset = 3isize;
    let mut host = vec![D::ZERO; offset as usize + N * N + 5];
    host[offset as usize..offset as usize + N * N].copy_from_slice(&hermitian);
    let src = probe.upload_flat(host.clone());

    println!("{} hermitian chain:", D::NAME);

    // `to_contiguous_read` of the strided normal view.
    let normal_view = D::typed(&src)
        .expect("dtype")
        .backend_region_view(vec![N, N], vec![1, N as isize], offset)
        .expect("normal view");
    let normal = probe
        .backend
        .to_contiguous_read(TensorRead::from_view(D::tensor_view(normal_view)))
        .unwrap_or_else(|err| panic!("{} to_contiguous_read FAILED: {err}", D::NAME));
    assert_bitwise(
        &probe.download::<D>(&normal),
        &hermitian,
        "to_contiguous_read of a packed region",
    );

    // `abs` -> `reduce_max`: the input scale.
    let input_abs = probe
        .backend
        .abs(&normal)
        .unwrap_or_else(|err| panic!("{} abs FAILED: {err}", D::NAME));
    assert_eq!(
        input_abs.dtype(),
        if D::IS_COMPLEX {
            D::REAL_DTYPE
        } else {
            D::dtype()
        },
        "abs output dtype"
    );
    let input_max = probe
        .backend
        .reduce_max(&input_abs, &[0, 1])
        .unwrap_or_else(|err| panic!("{} reduce_max FAILED: {err}", D::NAME));
    let input_scale = probe.download_real(&input_max, "reduce_max(abs)")[0];
    let expected_scale = hermitian
        .iter()
        .map(|value| value.magnitude())
        .fold(0.0_f64, f64::max);
    assert!(
        (input_scale - expected_scale).abs() <= 16.0 * D::EPS * expected_scale.max(1.0),
        "{} reduce_max(abs) got {input_scale} want {expected_scale}",
        D::NAME
    );

    // Rank-0 scalar upload in the *real* component dtype, the shape
    // `upload_scalar` takes when typed by `D::Real`: the production chain
    // divides a complex payload by a real scale, never by a complex one.
    let scale_tensor = probe.upload_shaped(Vec::new(), vec![input_scale as f32]);
    let normal_scaled = probe
        .backend
        .div(&normal, &scale_tensor)
        .unwrap_or_else(|err| panic!("{} div by a rank-0 f32 scalar FAILED: {err}", D::NAME));

    // An *f64* scalar against a single-precision payload is a dtype
    // mismatch; record what Tenferro does with it.
    let f64_scalar = probe.upload_shaped(Vec::new(), vec![input_scale]);
    match probe.backend.div(&normal, &f64_scalar) {
        Ok(_) => println!("  div({}, f64 scalar): ACCEPTED (mixed dtype)", D::NAME),
        Err(err) => println!("  div({}, f64 scalar): rejected with `{err}`", D::NAME),
    }
    if D::IS_COMPLEX {
        // A same-dtype (complex) rank-0 divisor is *not* the production shape;
        // record what it does, so the adapter does not reach for it by symmetry.
        let complex_scalar = probe.upload_shaped(Vec::new(), vec![D::from_parts(input_scale, 0.0)]);
        match probe.backend.div(&normal, &complex_scalar) {
            Ok(_) => println!("  div({}, {} scalar): accepted", D::NAME, D::NAME),
            Err(err) => println!(
                "  div({}, {} scalar): rejected with `{err}`",
                D::NAME,
                D::NAME
            ),
        }
    }

    // `conj` of the transposed view, then `sub`, `abs`, `reduce_max`.
    let transpose_view = D::typed(&src)
        .expect("dtype")
        .backend_region_view(vec![N, N], vec![N as isize, 1], offset)
        .expect("transpose view");
    let transpose = probe
        .backend
        .to_contiguous_read(TensorRead::from_view(D::tensor_view(transpose_view)))
        .unwrap_or_else(|err| panic!("{} to_contiguous_read (transpose) FAILED: {err}", D::NAME));
    let transpose = if D::IS_COMPLEX {
        probe
            .backend
            .conj(&transpose)
            .unwrap_or_else(|err| panic!("{} conj FAILED: {err}", D::NAME))
    } else {
        transpose
    };
    let transpose_scaled = probe
        .backend
        .div(&transpose, &scale_tensor)
        .unwrap_or_else(|err| panic!("{} div FAILED: {err}", D::NAME));
    let residual = probe
        .backend
        .sub(&normal_scaled, &transpose_scaled)
        .unwrap_or_else(|err| panic!("{} sub FAILED: {err}", D::NAME));
    let residual_abs = probe
        .backend
        .abs(&residual)
        .unwrap_or_else(|err| panic!("{} abs (residual) FAILED: {err}", D::NAME));
    let residual_max = probe
        .backend
        .reduce_max(&residual_abs, &[0, 1])
        .unwrap_or_else(|err| panic!("{} reduce_max (residual) FAILED: {err}", D::NAME));
    let residual_scale = probe.download_real(&residual_max, "reduce_max(abs(residual))")[0];
    // A^H == A exactly on this fixture, so the conjugate-transpose residual is
    // exactly zero -- an independent check that `conj` conjugates and does not
    // merely copy.
    assert_eq!(
        residual_scale,
        0.0,
        "{} conjugate-transpose residual of an exactly Hermitian block must be 0",
        D::NAME
    );

    // `reduce_sum_squares_read` over the real magnitudes.
    let magnitudes = if D::IS_COMPLEX {
        probe
            .backend
            .abs(&normal_scaled)
            .unwrap_or_else(|err| panic!("{} abs (magnitudes) FAILED: {err}", D::NAME))
    } else {
        normal_scaled
    };
    let sum_squares = probe
        .backend
        .reduce_sum_squares_read(TensorRead::from_tensor(&magnitudes), &[0, 1])
        .unwrap_or_else(|err| panic!("{} reduce_sum_squares_read FAILED: {err}", D::NAME));
    let got = probe.download_real(&sum_squares, "reduce_sum_squares")[0];
    let want: f64 = hermitian
        .iter()
        .map(|value| {
            let scaled = value.magnitude() / input_scale;
            scaled * scaled
        })
        .sum();
    assert!(
        (got - want).abs() <= 16.0 * ((N * N) as f64).sqrt() * D::EPS * want.max(1.0),
        "{} reduce_sum_squares got {got} want {want}",
        D::NAME
    );

    // A fixed `64 * f64::EPSILON` admission constant would be meaningless in
    // single precision; production resolves the caller's `HermitianTol` in
    // the payload's own lane (#1987).
    println!(
        "  residual_scale {residual_scale:e}, input_ss {got:e}; \
         64*f64::EPSILON = {:e}, 64*f32::EPSILON = {:e}",
        64.0 * f64::EPSILON,
        64.0 * D::EPS
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn hermitian_kernel_chain_f32() {
    hermitian_chain_case::<f32>(&mut Probe::new());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn hermitian_kernel_chain_c32() {
    hermitian_chain_case::<Complex32>(&mut Probe::new());
}

// ---------------------------------------------------------------------------
// 6. `copy_read_into` -- the factor scatter behind `cuda_copy_region_into`.
// ---------------------------------------------------------------------------

fn copy_region_case<D: ProbeScalar>(probe: &mut Probe) {
    let (rows, cols, ld, dst_offset) = (3usize, 4usize, 5usize, 7isize);
    let src_host: Vec<D> = (0..(rows * cols + 6)).map(D::sample).collect();
    let dst_host: Vec<D> = (0..(dst_offset as usize + ld * cols + 4))
        .map(|i| D::sample(i + 700))
        .collect();
    let src = probe.upload_flat(src_host.clone());
    let mut dst = probe.upload_flat(dst_host.clone());

    let src_view = D::typed(&src)
        .expect("dtype")
        .backend_region_view(vec![rows, cols], vec![1, rows as isize], 0)
        .expect("src view");
    let dst_view = D::typed_mut(&mut dst)
        .expect("dtype")
        .backend_region_view_mut(vec![rows, cols], vec![1, ld as isize], dst_offset)
        .expect("dst view");
    probe
        .backend
        .copy_read_into(
            TensorRead::from_view(D::tensor_view(src_view)),
            TensorWrite::from_view(D::tensor_view_mut(dst_view)),
        )
        .unwrap_or_else(|err| panic!("{} copy_read_into FAILED: {err}", D::NAME));

    let mut expected = dst_host.clone();
    for col in 0..cols {
        for row in 0..rows {
            expected[dst_offset as usize + row + col * ld] = src_host[row + col * rows];
        }
    }
    let actual = probe.download::<D>(&dst);
    assert_bitwise(&actual, &expected, "copy_read_into into a strided region");
    println!("copy_read_into {}: bitwise ok", D::NAME);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn copy_region_into_f32() {
    copy_region_case::<f32>(&mut Probe::new());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn copy_region_into_c32() {
    copy_region_case::<Complex32>(&mut Probe::new());
}

// ---------------------------------------------------------------------------
// 7. cuSOLVER factorizations on regions: SVD, QR (positive-diagonal gauge),
//    EIGH.
// ---------------------------------------------------------------------------

/// A well-conditioned `rows x cols` column-major fixture with distinct,
/// gap-separated singular values.
fn matrix_fixture<D: ProbeScalar>(rows: usize, cols: usize) -> Vec<D> {
    (0..rows * cols)
        .map(|flat| {
            let (row, col) = (flat % rows, flat / rows);
            let re =
                1.0 + 0.5 * row as f64 - 0.25 * col as f64 + if row == col { 3.0 } else { 0.0 };
            let im = if row == col {
                0.0
            } else {
                0.125 * (row as f64 - col as f64)
            };
            D::from_parts(re, im)
        })
        .collect()
}

/// `sqrt(eps(real(D)))`-scaled bound for factorization laws.
fn factor_tol<D: ProbeScalar>(n: usize) -> f64 {
    D::EPS.sqrt() * (n.max(1) as f64).sqrt()
}

fn svd_case<D: ProbeScalar>(probe: &mut Probe, rows: usize, cols: usize) {
    let k = rows.min(cols);
    let a = matrix_fixture::<D>(rows, cols);
    let a_dev = probe.upload_shaped(vec![rows, cols], a.clone());
    let view = D::typed(&a_dev)
        .expect("dtype")
        .backend_region_view(vec![rows, cols], vec![1, rows as isize], 0)
        .expect("view");
    let result = probe.backend.with_backend_session(|session| {
        TensorRead::from_view(D::tensor_view(view)).svd_read(session)
    });
    let (u, s, vt) = match result {
        Ok(factors) => factors,
        Err(err) => panic!("{} SVD {rows}x{cols} FAILED with `{err}`", D::NAME),
    };
    assert_eq!(u.shape(), [rows, k]);
    assert_eq!(vt.shape(), [k, cols]);
    let values = probe.download_real(&s, &format!("{} SVD singular values", D::NAME));
    assert_eq!(values.len(), k);
    for window in values.windows(2) {
        assert!(
            window[0] >= window[1] && window[1] >= 0.0,
            "{} SVD singular values must be non-negative and descending: {values:?}",
            D::NAME
        );
    }

    let u_host = probe.download::<D>(&u);
    let vt_host = probe.download::<D>(&vt);
    let tol = factor_tol::<D>(rows * cols);
    let scale = values[0].max(1.0);

    // Reconstruction: U diag(s) Vt == A.
    reconstruct_check::<D>(
        &u_host,
        &values,
        &vt_host,
        &a,
        rows,
        cols,
        k,
        tol * scale,
        "SVD",
    );

    // Orthonormal columns of U and rows of Vt.
    gram_check::<D>(&u_host, rows, k, tol, &format!("{} SVD U", D::NAME));
    let v_host: Vec<D> = (0..cols * k)
        .map(|flat| {
            let (row, col) = (flat % cols, flat / cols);
            vt_host[col + row * k].conj()
        })
        .collect();
    gram_check::<D>(&v_host, cols, k, tol, &format!("{} SVD V", D::NAME));
    println!("SVD {} {rows}x{cols}: ok", D::NAME);
}

/// `left * diag(values) * right == original`, all column-major.
#[allow(clippy::too_many_arguments)]
fn reconstruct_check<D: ProbeScalar>(
    left: &[D],
    values: &[f64],
    right: &[D],
    original: &[D],
    rows: usize,
    cols: usize,
    k: usize,
    bound: f64,
    what: &str,
) {
    for col in 0..cols {
        for row in 0..rows {
            let mut acc = D::ZERO;
            for p in 0..k {
                let scaled = left[row + p * rows] * D::from_parts(values[p], 0.0);
                acc = acc + scaled * right[p + col * k];
            }
            let want = original[row + col * rows];
            assert!(
                acc.distance(want) <= bound,
                "{what} ({}) reconstruction at ({row},{col}): got {acc:?} want {want:?} (bound {bound:e})",
                D::NAME
            );
        }
    }
}

/// `M^H M == I` for a column-major `rows x k` matrix.
fn gram_check<D: ProbeScalar>(m: &[D], rows: usize, k: usize, tol: f64, what: &str) {
    for a in 0..k {
        for b in 0..k {
            let mut acc = D::ZERO;
            for row in 0..rows {
                acc = acc + m[row + a * rows].conj() * m[row + b * rows];
            }
            let want = if a == b { D::ONE } else { D::ZERO };
            assert!(
                acc.distance(want) <= tol,
                "{what} orthonormality at ({a},{b}): got {acc:?} want {want:?} (bound {tol:e})"
            );
        }
    }
}

/// Returns the QR failure instead of panicking so the caller names the dtype
/// in its `expect`.
fn qr_case<D: ProbeScalar>(
    probe: &mut Probe,
    rows: usize,
    cols: usize,
) -> Result<(), tenferro_tensor::Error> {
    let k = rows.min(cols);
    let a = matrix_fixture::<D>(rows, cols);
    let a_dev = probe.upload_shaped(vec![rows, cols], a.clone());
    let view = D::typed(&a_dev)
        .expect("dtype")
        .backend_region_view(vec![rows, cols], vec![1, rows as isize], 0)
        .expect("view");
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let (q, r) = probe.backend.with_backend_session(|session| {
        TensorRead::from_view(D::tensor_view(view)).qr_with_options_read(options, session)
    })?;
    assert_eq!(q.shape(), [rows, k]);
    assert_eq!(r.shape(), [k, cols]);
    let q_host = probe.download::<D>(&q);
    let r_host = probe.download::<D>(&r);
    let tol = factor_tol::<D>(rows * cols);
    let scale = a.iter().map(|v| v.magnitude()).fold(1.0_f64, f64::max);

    // Q R == A.
    for col in 0..cols {
        for row in 0..rows {
            let mut acc = D::ZERO;
            for p in 0..k {
                acc = acc + q_host[row + p * rows] * r_host[p + col * k];
            }
            let want = a[row + col * rows];
            assert!(
                acc.distance(want) <= tol * scale,
                "{} QR reconstruction at ({row},{col}): got {acc:?} want {want:?}",
                D::NAME
            );
        }
    }
    gram_check::<D>(&q_host, rows, k, tol, &format!("{} QR Q", D::NAME));
    // Positive-diagonal gauge: R_jj real and non-negative.
    for j in 0..k {
        let diag = r_host[j + j * k];
        assert!(
            diag.re() >= -tol * scale && (diag.magnitude() - diag.re().abs()).abs() <= tol * scale,
            "{} QR diagonal {j} is not real non-negative: {diag:?}",
            D::NAME
        );
    }
    println!("QR {} {rows}x{cols}: ok (positive-diagonal gauge)", D::NAME);
    Ok(())
}

fn eigh_case<D: ProbeScalar>(probe: &mut Probe, n: usize) {
    // Hermitian by construction: B + B^H.
    let b = matrix_fixture::<D>(n, n);
    let mut a = vec![D::ZERO; n * n];
    for col in 0..n {
        for row in 0..n {
            a[row + col * n] = b[row + col * n] + b[col + row * n].conj();
        }
    }
    let a_dev = probe.upload_shaped(vec![n, n], a.clone());
    let view = D::typed(&a_dev)
        .expect("dtype")
        .backend_region_view(vec![n, n], vec![1, n as isize], 0)
        .expect("view");
    let result = probe.backend.with_backend_session(|session| {
        TensorRead::from_view(D::tensor_view(view)).eigh_read(session)
    });
    let (values, vectors) = match result {
        Ok(pair) => pair,
        Err(err) => panic!("{} EIGH {n}x{n} FAILED with `{err}`", D::NAME),
    };
    assert_eq!(vectors.shape(), [n, n]);
    let lambda = probe.download_real(&values, &format!("{} EIGH eigenvalues", D::NAME));
    assert_eq!(lambda.len(), n);
    for window in lambda.windows(2) {
        assert!(
            window[0] <= window[1],
            "{} EIGH eigenvalues must be ascending: {lambda:?}",
            D::NAME
        );
    }

    let v = probe.download::<D>(&vectors);
    let tol = factor_tol::<D>(n * n);
    let scale = lambda.iter().fold(1.0_f64, |m, l| m.max(l.abs()));
    gram_check::<D>(&v, n, n, tol, &format!("{} EIGH vectors", D::NAME));
    // A v_j == lambda_j v_j.
    for j in 0..n {
        for row in 0..n {
            let mut acc = D::ZERO;
            for p in 0..n {
                acc = acc + a[row + p * n] * v[p + j * n];
            }
            let want = v[row + j * n] * D::from_parts(lambda[j], 0.0);
            assert!(
                acc.distance(want) <= tol * scale,
                "{} EIGH pair {j} row {row}: got {acc:?} want {want:?}",
                D::NAME
            );
        }
    }
    println!("EIGH {} {n}x{n}: ok", D::NAME);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn svd_region_f32() {
    let mut probe = Probe::new();
    svd_case::<f32>(&mut probe, 5, 3);
    svd_case::<f32>(&mut probe, 3, 5);
    svd_case::<f32>(&mut probe, 4, 4);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn svd_region_c32() {
    let mut probe = Probe::new();
    svd_case::<Complex32>(&mut probe, 5, 3);
    svd_case::<Complex32>(&mut probe, 3, 5);
    svd_case::<Complex32>(&mut probe, 4, 4);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn qr_region_f32() {
    let mut probe = Probe::new();
    qr_case::<f32>(&mut probe, 5, 3).expect("f32 device QR");
    qr_case::<f32>(&mut probe, 4, 4).expect("f32 device QR");
}

/// Raw Tenferro device QR of a `Complex32` region: the probe requires the
/// reconstruction and gauge checks of `qr_case` (the `triu` zero must compile
/// for `cuFloatComplex`, tenferro-rs#1833, #1837); #1271 admits it in symtn.
#[test]
#[ignore = "requires a real CUDA device"]
fn qr_region_c32() {
    let mut probe = Probe::new();
    qr_case::<Complex32>(&mut probe, 5, 3).expect("Complex32 device QR");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn eigh_region_f32() {
    let mut probe = Probe::new();
    eigh_case::<f32>(&mut probe, 4);
    eigh_case::<f32>(&mut probe, 6);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn eigh_region_c32() {
    let mut probe = Probe::new();
    eigh_case::<Complex32>(&mut probe, 4);
    eigh_case::<Complex32>(&mut probe, 6);
}

// ---------------------------------------------------------------------------
// 8. `lu` / `solve`: not reached by symtn (#1065 keeps solve behind its
//    own gate); the raw Tenferro results are checked against host oracles.
// ---------------------------------------------------------------------------

/// Device `solve` must reach the host residual `A x - b`, and device `lu` must
/// reconstruct `P A = L U` with `L` lower and `U` upper triangular, all checked
/// on the host against the host fixture, never against device output alone.
///
/// `expected_p[row]` is the column of the one in row `row` of `P`, from partial
/// pivoting by hand.
fn lu_solve_case<D: ProbeScalar>(probe: &mut Probe, a: [(f64, f64); 9], expected_p: [usize; 3]) {
    let a: Vec<D> = a.iter().map(|&(re, im)| D::from_parts(re, im)).collect();
    let b: Vec<D> = vec![
        D::from_parts(1.0, 0.5),
        D::from_parts(-2.0, 0.25),
        D::from_parts(0.5, -1.0),
    ];
    let a_dev = probe.upload_shaped(vec![3, 3], a.clone());
    let b_dev = probe.upload_shaped(vec![3, 1], b.clone());

    match probe
        .backend
        .with_backend_session(|session| a_dev.solve(&b_dev, session))
    {
        Ok(x_dev) => {
            let x = probe.download::<D>(&x_dev);
            let mut worst = 0.0_f64;
            for row in 0..3 {
                let mut acc = D::ZERO;
                for col in 0..3 {
                    acc = acc + a[col * 3 + row] * x[col];
                }
                worst = worst.max(acc.distance(b[row]));
            }
            println!("{} solve: succeeded, residual {worst:.3e}", D::NAME);
            assert!(
                worst <= factor_tol::<D>(9) * 10.0,
                "{} solve residual {worst:e}",
                D::NAME
            );
        }
        Err(err) => panic!("{} solve failed: {err}", D::NAME),
    }

    match probe
        .backend
        .with_backend_session(|session| a_dev.lu(session))
    {
        Ok((p, l, u, parity)) => {
            println!(
                "{} lu: succeeded, shapes P={:?} L={:?} U={:?} parity={:?}",
                D::NAME,
                p.shape(),
                l.shape(),
                u.shape(),
                parity.shape()
            );
            for factor in [&p, &l, &u] {
                assert_eq!(factor.shape(), [3, 3], "{} lu factor shape", D::NAME);
            }
            let p = probe.download::<D>(&p);
            let l = probe.download::<D>(&l);
            let u = probe.download::<D>(&u);
            // Column-major 3x3: entry (row, col) is at `col * 3 + row`.
            let at = |m: &[D], row: usize, col: usize| m[col * 3 + row];
            for (row, &one) in expected_p.iter().enumerate() {
                for col in 0..3 {
                    let want = if col == one {
                        D::from_parts(1.0, 0.0)
                    } else {
                        D::ZERO
                    };
                    assert_eq!(at(&p, row, col), want, "{} lu P[{row},{col}]", D::NAME);
                }
            }
            let a_max = a.iter().map(|&x| x.magnitude()).fold(0.0_f64, f64::max);
            let tol = 16.0 * 3.0 * D::EPS * a_max;
            let mut worst = 0.0_f64;
            for row in 0..3 {
                for col in 0..3 {
                    let mut pa = D::ZERO;
                    let mut lu = D::ZERO;
                    for k in 0..3 {
                        pa = pa + at(&p, row, k) * at(&a, k, col);
                        lu = lu + at(&l, row, k) * at(&u, k, col);
                    }
                    worst = worst.max(pa.distance(lu));
                    if col > row {
                        assert_eq!(at(&l, row, col), D::ZERO, "{} L not lower", D::NAME);
                    }
                    if row > col {
                        assert_eq!(at(&u, row, col), D::ZERO, "{} U not upper", D::NAME);
                    }
                }
            }
            println!(
                "{} lu: P A - L U residual {worst:.3e} (bound {tol:.3e})",
                D::NAME
            );
            assert!(worst <= tol, "{} lu residual {worst:e} > {tol:e}", D::NAME);
        }
        Err(err) => panic!("{} lu failed: {err}", D::NAME),
    }
}

/// Diagonally dominant, column-major: partial pivoting keeps `P = I`.
const DOMINANT: [(f64, f64); 9] = [
    (4.0, 0.0),
    (1.0, 0.5),
    (0.0, -0.25),
    (1.0, -0.5),
    (5.0, 0.0),
    (1.0, 0.25),
    (0.0, 0.25),
    (1.0, -0.25),
    (6.0, 0.0),
];

/// Rows `[0, 8+i/4, 0]`, `[2, 1, 0]`, `[4, 0, 1-i/2]`, column-major: a zero
/// leading pivot. Step 1 swaps rows 1 and 3 (`|4|` is largest), step 2 swaps
/// rows 2 and 3 (`|8+i/4| > |1|`), so `P A = [A_3; A_1; A_2]`. That `P` is a
/// 3-cycle, not symmetric, so `P A = L U` and `A = P L U` disagree on it — the
/// diagonally dominant fixture cannot tell the two conventions apart.
const ROW_SWAP: [(f64, f64); 9] = [
    (0.0, 0.0),
    (2.0, 0.0),
    (4.0, 0.0),
    (8.0, 0.25),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, -0.5),
];

#[test]
#[ignore = "requires a real CUDA device"]
fn lu_and_solve_behaviour_f32() {
    lu_solve_case::<f32>(&mut Probe::new(), DOMINANT, [0, 1, 2]);
    lu_solve_case::<f32>(&mut Probe::new(), ROW_SWAP, [2, 0, 1]);
}

/// `Complex32` `lu`/`solve` shared the #1833 complex-constant kernel defect
/// with QR up to Tenferro 0.5.0; 0.6.0 carries the fix (#1837), so the probe
/// asserts the host oracles. symtn exposes no device `lu`/`solve`.
#[test]
#[ignore = "requires a real CUDA device"]
fn lu_and_solve_behaviour_c32() {
    lu_solve_case::<Complex32>(&mut Probe::new(), DOMINANT, [0, 1, 2]);
    lu_solve_case::<Complex32>(&mut Probe::new(), ROW_SWAP, [2, 0, 1]);
}
