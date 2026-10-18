//! Device probe for the N-D strided region accumulation symtn's basic device
//! operations (structural transforms, trace, `otimes`, `cat`, mixed
//! owned/lazy `add`) are built on.
//!
//! Tenferro's sources imply that `dot_general_read_into_accum` against a 1x1
//! ones operand moves a strided, offset, rank-N source region into a strided,
//! offset, rank-N destination region with alpha/beta and conjugation -- the
//! N-D generalization of the production `cuda_axpby_owned` pattern. This file converts that source-level claim into
//! device evidence, and records the exact rejection text where Tenferro
//! declines.
//!
//! The oracle is an independent host strided loop over the same index space;
//! nothing here is derived from a symtn descriptor. Comparison is bitwise
//! wherever the arithmetic is a pure copy, conjugation or single add
//! (`alpha = 1`), and tolerance-based only where an `alpha != 1` multiply is
//! involved.
//!
//! These tests call Tenferro directly rather than through `cuda_adapter`:
//! its `Tensor` handles are private, and the probe deliberately adds no
//! production surface. The device is the
//! unit under test, so every test is `#[ignore]` like the rest of the device
//! suite.

#![cfg(feature = "cuda")]

use std::fmt::Debug;
use std::num::NonZeroUsize;

use num_complex::Complex64;
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{
    BackendSessionHost, DotGeneralAccumulation, DotGeneralConfig, Error, Tensor, TensorDot,
    TensorRead, TensorWrite,
};

use symtn_dense::CudaScalar;

/// Payload dtypes this probe drives, with the host-side arithmetic its oracle
/// needs. `CudaScalar` already pins the device-side dtype boundary
/// (`f64` / `Complex64`); this adds only the host operations.
trait ProbeScalar:
    CudaScalar + Copy + Debug + PartialEq + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    const NAME: &'static str;

    fn from_parts(re: f64, im: f64) -> Self;
    fn conj(self) -> Self;
    /// Distance used only for the `alpha != 1` comparisons.
    fn distance(self, other: Self) -> f64;
    /// A deterministic, non-degenerate sample value for buffer index `index`.
    fn sample(index: usize) -> Self {
        let re = 1.0 + (index as f64) * 0.25;
        let im = -0.5 + (index as f64) * 0.125;
        Self::from_parts(re, im)
    }
}

impl ProbeScalar for f64 {
    const NAME: &'static str = "f64";

    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }

    fn conj(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl ProbeScalar for Complex64 {
    const NAME: &'static str = "Complex64";

    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }

    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

/// Host/device transfer counter, so each probe can state that the contraction
/// phase itself moves nothing across the boundary. The adapter's
/// `cuda_transfer_stats` counts only the `symtn-dense` seam, which this file
/// bypasses, so the transfers are counted where they are made.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Transfers {
    h2d: usize,
    d2h: usize,
}

struct Probe {
    backend: CudaBackend,
    transfers: Transfers,
}

impl Probe {
    fn new() -> Self {
        let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0))
            .expect("CUDA device 0 must be available for the device probe");
        Self {
            backend,
            transfers: Transfers::default(),
        }
    }

    fn upload<D: ProbeScalar>(&mut self, data: Vec<D>) -> Tensor {
        let len = data.len();
        let host = Tensor::from_vec_col_major(vec![len], data).expect("host tensor");
        let device = upload_tensor(self.backend.runtime(), &host).expect("upload");
        self.transfers.h2d += 1;
        device
    }

    fn download<D: ProbeScalar>(&mut self, tensor: &Tensor) -> Vec<D> {
        let host = download_tensor(self.backend.runtime(), tensor).expect("download");
        self.transfers.d2h += 1;
        D::into_typed(host)
            .expect("dtype")
            .into_host_vec()
            .expect("host vec")
    }

    /// `dst_region = alpha * [conj] src_region + beta * dst_region`, expressed
    /// as a contraction of the rank-N source region (plus a trailing unit
    /// mode) against a 1x1 ones operand. This is the primitive under test.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_region<D: ProbeScalar>(
        &mut self,
        src: &Tensor,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        ones: &Tensor,
        dst: &mut Tensor,
        dst_strides: &[isize],
        dst_offset: isize,
        alpha: D,
        beta: D,
        conj: bool,
    ) -> Result<(), Error> {
        let mut view_dims = dims.to_vec();
        view_dims.push(1);
        let mut lhs_strides = src_strides.to_vec();
        lhs_strides.push(1);
        let mut out_strides = dst_strides.to_vec();
        out_strides.push(1);
        let contracted = dims.len();

        let lhs = D::typed(src).expect("source dtype").backend_region_view(
            view_dims.clone(),
            lhs_strides,
            src_offset,
        )?;
        let rhs =
            D::typed(ones)
                .expect("ones dtype")
                .backend_region_view(vec![1, 1], vec![1, 1], 0)?;
        let out = D::typed_mut(dst)
            .expect("destination dtype")
            .backend_region_view_mut(view_dims, out_strides, dst_offset)?;

        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![contracted],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: Vec::new(),
            rhs_batch_dims: Vec::new(),
        };
        let accumulation = DotGeneralAccumulation {
            lhs_conj: conj,
            rhs_conj: false,
            alpha: alpha.contraction_scalar(),
            beta: beta.contraction_scalar(),
        };
        self.backend.dot_general_read_into_accum(
            TensorRead::from_view(D::tensor_view(lhs)),
            TensorRead::from_view(D::tensor_view(rhs)),
            &config,
            accumulation,
            TensorWrite::from_view(D::tensor_view_mut(out)),
        )
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
/// axis `order[k]` is the k-th fastest-varying axis. `order` must be a
/// permutation of `0..dims.len()`.
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

fn assert_close<D: ProbeScalar>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let scale = 1.0_f64.max(want.distance(D::from_parts(0.0, 0.0)));
        assert!(
            got.distance(*want) <= 1e-12 * scale,
            "{what}: element {index} ({}) got {got:?} want {want:?}",
            D::NAME
        );
    }
}

// ---------------------------------------------------------------------------
// Item 1: rank 3-6 permuted-stride source at an offset into a permuted-stride
// destination at an offset, several blocks inside one flat buffer.
// ---------------------------------------------------------------------------

/// One (dims, source axis order, destination axis order) case. The orders
/// differ so the destination carries a genuine axis permutation, which is the
/// part of the survey's claim that has no upstream or symtn device evidence.
const ND_CASES: &[(&[usize], &[usize], &[usize])] = &[
    (&[2, 3, 4], &[0, 1, 2], &[2, 0, 1]),
    (&[3, 2, 2, 3], &[0, 1, 2, 3], &[1, 3, 0, 2]),
    (&[2, 2, 3, 2, 2], &[0, 1, 2, 3, 4], &[4, 2, 0, 3, 1]),
    (
        &[2, 2, 2, 2, 2, 3],
        &[5, 0, 1, 2, 3, 4],
        &[0, 5, 4, 1, 2, 3],
    ),
];

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
    // Three blocks inside one flat buffer, at unrelated offsets, with a
    // leading stride > 1 so no axis is contiguous with the allocation start.
    let src_strides = permuted_strides(dims, src_order, 2);
    let dst_strides = permuted_strides(dims, dst_order, 3);
    let src_bases = [7isize, 5 + 2 * block as isize, 3 + 5 * block as isize];
    let dst_bases = [11isize, 2 + 4 * block as isize, 9 + 8 * block as isize];

    let src_len = 16 + 12 * block;
    let dst_len = 16 + 16 * block;
    let src_host: Vec<D> = (0..src_len).map(D::sample).collect();
    let dst_host: Vec<D> = (0..dst_len).map(|index| D::sample(index + 1000)).collect();

    let src = probe.upload(src_host.clone());
    let ones = probe.upload(vec![D::ONE]);
    let mut dst = probe.upload(dst_host.clone());

    let before = probe.transfers;
    for (&src_base, &dst_base) in src_bases.iter().zip(&dst_bases) {
        probe
            .accumulate_region::<D>(
                &src,
                dims,
                &src_strides,
                src_base,
                &ones,
                &mut dst,
                &dst_strides,
                dst_base,
                alpha,
                beta,
                conj,
            )
            .unwrap_or_else(|err| panic!("rank {} accumulate failed: {err}", dims.len()));
    }
    assert_eq!(
        probe.transfers, before,
        "the contraction phase must not transfer host data"
    );

    // Independent oracle: the same index space walked on the host.
    let mut expected = dst_host.clone();
    for (&src_base, &dst_base) in src_bases.iter().zip(&dst_bases) {
        let src_positions = region_offsets(dims, &src_strides, src_base);
        let dst_positions = region_offsets(dims, &dst_strides, dst_base);
        for (&from, &to) in src_positions.iter().zip(&dst_positions) {
            let value = if conj {
                src_host[from].conj()
            } else {
                src_host[from]
            };
            expected[to] = alpha * value + beta * expected[to];
        }
    }

    let actual = probe.download::<D>(&dst);
    let what = format!(
        "rank {} dims {dims:?} src_order {src_order:?} dst_order {dst_order:?} conj {conj}",
        dims.len()
    );
    if alpha == D::ONE {
        assert_bitwise(&actual, &expected, &what);
    } else {
        assert_close(&actual, &expected, &what);
    }
}

fn nd_region_sweep<D: ProbeScalar>(probe: &mut Probe) {
    for &(dims, src_order, dst_order) in ND_CASES {
        for &conj in &[false, true] {
            for &beta in &[D::ZERO, D::ONE] {
                // alpha = 1: pure copy / conj / add, compared bitwise.
                nd_region_case::<D>(probe, dims, src_order, dst_order, D::ONE, beta, conj);
                // alpha != 1, real and (for a complex payload) complex.
                nd_region_case::<D>(
                    probe,
                    dims,
                    src_order,
                    dst_order,
                    D::from_parts(-0.75, 0.0),
                    beta,
                    conj,
                );
                if D::IS_COMPLEX {
                    nd_region_case::<D>(
                        probe,
                        dims,
                        src_order,
                        dst_order,
                        D::from_parts(0.5, -1.25),
                        beta,
                        conj,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_strided_region_accumulation_matches_host_f64() {
    let mut probe = Probe::new();
    nd_region_sweep::<f64>(&mut probe);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_strided_region_accumulation_matches_host_c64() {
    let mut probe = Probe::new();
    nd_region_sweep::<Complex64>(&mut probe);
}

// ---------------------------------------------------------------------------
// Item 2: outer product (no contracting dims) into a permuted-stride region.
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
    let rhs_host: Vec<D> = (0..64).map(|index| D::sample(index + 200)).collect();
    let out_host: Vec<D> = (0..256).map(|index| D::sample(index + 2000)).collect();

    let lhs_dev = probe.upload(lhs_host.clone());
    let rhs_dev = probe.upload(rhs_host.clone());
    let mut out_dev = probe.upload(out_host.clone());

    let alpha = D::from_parts(0.5, -0.25);
    let beta = D::ONE;
    let config = DotGeneralConfig {
        lhs_contracting_dims: Vec::new(),
        rhs_contracting_dims: Vec::new(),
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: true,
        rhs_conj: false,
        alpha: alpha.contraction_scalar(),
        beta: beta.contraction_scalar(),
    };
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
    probe
        .backend
        .dot_general_read_into_accum(
            TensorRead::from_view(D::tensor_view(lhs_view)),
            TensorRead::from_view(D::tensor_view(rhs_view)),
            &config,
            accumulation,
            TensorWrite::from_view(D::tensor_view_mut(out_view)),
        )
        .expect("outer product");

    let mut expected = out_host.clone();
    let lhs_positions = region_offsets(&lhs_dims, &lhs_strides, lhs_base);
    let rhs_positions = region_offsets(&rhs_dims, &rhs_strides, rhs_base);
    let out_positions = region_offsets(&out_dims, &out_strides, out_base);
    // Output index order is (lhs free axes, then rhs free axes), i.e. the
    // column-major flat order of `out_dims`, so the two operand walks index
    // into it exactly like a column-major outer product.
    for (rhs_flat, &rhs_at) in rhs_positions.iter().enumerate() {
        for (lhs_flat, &lhs_at) in lhs_positions.iter().enumerate() {
            let out_flat = rhs_flat * lhs_positions.len() + lhs_flat;
            let at = out_positions[out_flat];
            expected[at] = alpha * lhs_host[lhs_at].conj() * rhs_host[rhs_at] + beta * expected[at];
        }
    }

    let actual = probe.download::<D>(&out_dev);
    assert_close(&actual, &expected, "outer product into permuted strides");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn outer_product_into_permuted_region_matches_host() {
    let mut probe = Probe::new();
    outer_product_case::<f64>(&mut probe);
    outer_product_case::<Complex64>(&mut probe);
}

// ---------------------------------------------------------------------------
// Item 3: diagonal read view (stride s_i + s_j) against a ones operand
// = partial trace of a block.
// ---------------------------------------------------------------------------

fn diagonal_trace_case<D: ProbeScalar>(probe: &mut Probe) {
    // Block [o0, o1, t, t] with a permuted source layout; the traced pair is
    // axes 2 and 3, merged into one axis of extent t and stride s2 + s3.
    let block_dims = [2usize, 3, 4, 4];
    let block_strides = permuted_strides(&block_dims, &[2, 0, 3, 1], 2);
    let base = 9isize;
    let traced = block_dims[2];

    let merged_dims = [block_dims[0], block_dims[1], traced];
    let merged_strides = [
        block_strides[0],
        block_strides[1],
        block_strides[2] + block_strides[3],
    ];

    let out_dims = [block_dims[0], block_dims[1]];
    let out_strides = permuted_strides(&out_dims, &[1, 0], 3);
    let out_base = 6isize;

    let src_host: Vec<D> = (0..512).map(D::sample).collect();
    let out_host: Vec<D> = (0..64).map(|index| D::sample(index + 3000)).collect();
    let src_dev = probe.upload(src_host.clone());
    let ones_dev = probe.upload(vec![D::ONE; traced]);
    let mut out_dev = probe.upload(out_host.clone());

    let alpha = D::from_parts(-1.5, 0.0);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: alpha.contraction_scalar(),
        beta: D::ONE.contraction_scalar(),
    };
    let lhs_view = D::typed(&src_dev)
        .expect("src dtype")
        .backend_region_view(merged_dims.to_vec(), merged_strides.to_vec(), base)
        .expect("diagonal view");
    let rhs_view = D::typed(&ones_dev)
        .expect("ones dtype")
        .backend_region_view(vec![traced, 1], vec![1, traced as isize], 0)
        .expect("ones view");
    let out_view = D::typed_mut(&mut out_dev)
        .expect("out dtype")
        .backend_region_view_mut(
            vec![out_dims[0], out_dims[1], 1],
            vec![out_strides[0], out_strides[1], 1],
            out_base,
        )
        .expect("out view");
    probe
        .backend
        .dot_general_read_into_accum(
            TensorRead::from_view(D::tensor_view(lhs_view)),
            TensorRead::from_view(D::tensor_view(rhs_view)),
            &config,
            accumulation,
            TensorWrite::from_view(D::tensor_view_mut(out_view)),
        )
        .expect("diagonal trace");

    // Oracle: trace over the two axes of the *unmerged* block, so the merged
    // stride itself is under test rather than assumed.
    let mut expected = out_host.clone();
    for i1 in 0..out_dims[1] {
        for i0 in 0..out_dims[0] {
            let mut sum = D::ZERO;
            for d in 0..traced {
                let at = base
                    + i0 as isize * block_strides[0]
                    + i1 as isize * block_strides[1]
                    + d as isize * block_strides[2]
                    + d as isize * block_strides[3];
                sum = sum + src_host[at as usize];
            }
            let to =
                (out_base + i0 as isize * out_strides[0] + i1 as isize * out_strides[1]) as usize;
            expected[to] = alpha * sum + expected[to];
        }
    }

    let actual = probe.download::<D>(&out_dev);
    assert_close(&actual, &expected, "diagonal-stride partial trace");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn diagonal_read_view_traces_block() {
    let mut probe = Probe::new();
    diagonal_trace_case::<f64>(&mut probe);
    diagonal_trace_case::<Complex64>(&mut probe);
}

// ---------------------------------------------------------------------------
// Item 4: rejection behaviour -- overlap/alias, zero-size blocks, high rank.
// ---------------------------------------------------------------------------

/// A source and a destination region inside the *same* buffer cannot be
/// requested at all: `backend_region_view` borrows the tensor shared and
/// `backend_region_view_mut` borrows it exclusively, so in-buffer aliasing is
/// a compile-time boundary rather than a runtime check. This test pins the
/// runtime half that remains: a mutable view whose own logical elements alias.
#[test]
#[ignore = "requires a real CUDA device"]
fn overlapping_mutable_region_is_rejected() {
    let mut probe = Probe::new();
    let mut dst = probe.upload::<f64>(vec![0.0; 64]);
    let err = f64::typed_mut(&mut dst)
        .expect("dtype")
        .backend_region_view_mut(vec![4, 4], vec![1, 0], 0)
        .expect_err("a stride-0 destination axis aliases and must be rejected");
    let text = err.to_string();
    println!("overlapping mutable destination view: {text}");
    assert!(
        text.to_lowercase().contains("overlap"),
        "unexpected rejection text: {text}"
    );

    // Out-of-bounds regions are rejected by the same constructor.
    let err = f64::typed(&dst)
        .expect("dtype")
        .backend_region_view(vec![4, 4], vec![1, 4], 56)
        .expect_err("a region running past the allocation must be rejected");
    println!("out-of-bounds read view: {err}");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn negative_stride_read_view_is_rejected() {
    let mut probe = Probe::new();
    let src = probe.upload::<f64>((0..64).map(|index| index as f64).collect());
    let ones = probe.upload::<f64>(vec![1.0]);
    let mut dst = probe.upload::<f64>(vec![0.0; 64]);
    let result = probe.accumulate_region::<f64>(
        &src,
        &[2, 3],
        &[1, -2],
        8,
        &ones,
        &mut dst,
        &[1, 2],
        0,
        1.0,
        0.0,
        false,
    );
    match result {
        Ok(()) => panic!("a negative source stride was accepted"),
        Err(err) => {
            let text = err.to_string();
            println!("negative source stride: {text}");
            assert!(
                text.contains("nonnegative"),
                "unexpected rejection text: {text}"
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn zero_extent_region_is_a_no_op_or_explicit_error() {
    let mut probe = Probe::new();
    let src = probe.upload::<f64>((0..64).map(|index| index as f64).collect());
    let ones = probe.upload::<f64>(vec![1.0]);
    let dst_host: Vec<f64> = (0..64).map(|index| -(index as f64)).collect();
    let mut dst = probe.upload::<f64>(dst_host.clone());
    let result = probe.accumulate_region::<f64>(
        &src,
        &[2, 0, 3],
        &[1, 2, 2],
        0,
        &ones,
        &mut dst,
        &[1, 2, 2],
        0,
        1.0,
        0.0,
        false,
    );
    match result {
        Ok(()) => {
            let actual = probe.download::<f64>(&dst);
            assert_bitwise(&actual, &dst_host, "zero-extent region must touch nothing");
            println!("zero-extent region: accepted, destination unchanged");
        }
        Err(err) => println!("zero-extent region: rejected with `{err}`"),
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rank_beyond_cutensor_mode_limit_is_reported() {
    let mut probe = Probe::new();
    let src = probe.upload::<f64>((0..64).map(|index| index as f64).collect());
    let ones = probe.upload::<f64>(vec![1.0]);
    let mut dst = probe.upload::<f64>(vec![0.0; 64]);

    // Ranks are padded with unit axes, so the payload stays at 8 elements and
    // only the mode count varies. The first rank that fails is the limit.
    let mut first_failure = None;
    for rank in [8usize, 16, 24, 32, 40, 48, 56, 63, 64, 65, 72] {
        let mut dims = vec![1usize; rank];
        dims[0] = 8;
        let strides = vec![1isize; rank];
        let result = probe.accumulate_region::<f64>(
            &src, &dims, &strides, 0, &ones, &mut dst, &strides, 0, 1.0, 0.0, false,
        );
        match result {
            Ok(()) => println!("rank {rank}: accepted"),
            Err(err) => {
                println!("rank {rank}: rejected with `{err}`");
                if first_failure.is_none() {
                    first_failure = Some((rank, err.to_string()));
                }
            }
        }
    }
    match &first_failure {
        Some((rank, text)) => println!("first rejected rank: {rank} (`{text}`)"),
        None => println!("no mode-count limit reached up to rank 72"),
    }
    // Every rank symtn can reach through a fusion tree is far below any
    // plausible cuTENSOR mode limit; the probe only records where it lies.
    assert!(
        first_failure.as_ref().map_or(usize::MAX, |(rank, _)| *rank) > 8,
        "rank 8 must be usable"
    );
}

// ---------------------------------------------------------------------------
// Item 5: plan-cache behaviour per distinct (shape, strides) key.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn plan_cache_reuses_one_key_and_evicts_past_the_limit() {
    let mut probe = Probe::new();
    let max_entries = probe
        .backend
        .cutensor_plan_cache_max_entries()
        .expect("plan cache max entries");
    println!("default cutensor plan cache max entries: {max_entries}");

    let src = probe.upload::<f64>((0..4096).map(|index| index as f64).collect());
    let ones = probe.upload::<f64>(vec![1.0]);
    let mut dst = probe.upload::<f64>(vec![0.0; 4096]);

    // Same signature, many replays: one entry, hits grow.
    let before = probe.backend.cutensor_plan_cache_stats().expect("stats");
    for _ in 0..16 {
        probe
            .accumulate_region::<f64>(
                &src,
                &[2, 3, 4],
                &[1, 2, 6],
                0,
                &ones,
                &mut dst,
                &[1, 2, 6],
                0,
                1.0,
                0.0,
                false,
            )
            .expect("repeat");
    }
    let repeated = probe.backend.cutensor_plan_cache_stats().expect("stats");
    println!("after 16 replays of one key: {repeated:?} (was {before:?})");
    assert!(
        repeated.hits > before.hits,
        "a repeated signature must hit the plan cache"
    );

    // Distinct signatures past the limit: entries cap, evictions appear.
    let distinct = max_entries.get() + 16;
    for index in 0..distinct {
        let extent = 2 + index;
        probe
            .accumulate_region::<f64>(
                &src,
                &[extent, 3],
                &[1, extent as isize],
                0,
                &ones,
                &mut dst,
                &[1, extent as isize],
                0,
                1.0,
                0.0,
                false,
            )
            .unwrap_or_else(|err| panic!("distinct key {index} failed: {err}"));
    }
    let thrashed = probe.backend.cutensor_plan_cache_stats().expect("stats");
    println!("after {distinct} distinct keys: {thrashed:?}");
    assert!(
        thrashed.entries <= max_entries.get(),
        "plan cache must respect its entry limit"
    );
    assert!(
        thrashed.evictions > 0,
        "exceeding the entry limit must evict"
    );

    // The cap is configurable, which is what a many-block transform needs.
    probe
        .backend
        .set_cutensor_plan_cache_max_entries(NonZeroUsize::new(256).expect("nonzero"))
        .expect("resize plan cache");
    assert_eq!(
        probe
            .backend
            .cutensor_plan_cache_max_entries()
            .expect("max entries"),
        NonZeroUsize::new(256).expect("nonzero")
    );
    println!(
        "after resize to 256: {:?}",
        probe.backend.cutensor_plan_cache_stats().expect("stats")
    );
    println!("probe transfers (uploads/downloads): {:?}", probe.transfers);
}

// ---------------------------------------------------------------------------
// Item 6: Complex64 solve / lu on CUDA -- does it hit the #1833 defect class?
// ---------------------------------------------------------------------------

fn solve_probe<D: ProbeScalar>(probe: &mut Probe) {
    // Diagonally dominant 3x3 so the system is well conditioned; column-major.
    let a: Vec<D> = vec![
        D::from_parts(4.0, 0.0),
        D::from_parts(1.0, 0.5),
        D::from_parts(0.0, -0.25),
        D::from_parts(1.0, -0.5),
        D::from_parts(5.0, 0.0),
        D::from_parts(1.0, 0.25),
        D::from_parts(0.0, 0.25),
        D::from_parts(1.0, -0.25),
        D::from_parts(6.0, 0.0),
    ];
    let b: Vec<D> = vec![
        D::from_parts(1.0, 0.5),
        D::from_parts(-2.0, 0.25),
        D::from_parts(0.5, -1.0),
    ];
    let a_host = Tensor::from_vec_col_major(vec![3usize, 3], a.clone()).expect("a host");
    let b_host = Tensor::from_vec_col_major(vec![3usize, 1], b.clone()).expect("b host");
    let a_dev = upload_tensor(probe.backend.runtime(), &a_host).expect("upload a");
    let b_dev = upload_tensor(probe.backend.runtime(), &b_host).expect("upload b");

    let solved = probe
        .backend
        .with_backend_session(|session| a_dev.solve(&b_dev, session));
    match solved {
        Ok(x_dev) => {
            let x = probe.download::<D>(&x_dev);
            // Oracle: residual of A x - b computed on the host.
            let mut worst = 0.0_f64;
            for row in 0..3 {
                let mut acc = D::ZERO;
                for col in 0..3 {
                    acc = acc + a[col * 3 + row] * x[col];
                }
                worst = worst.max(acc.distance(b[row]));
            }
            println!("{} solve: succeeded, residual {worst:.3e}", D::NAME);
            assert!(worst <= 1e-10, "{} solve residual {worst:e}", D::NAME);
        }
        Err(err) => println!("{} solve: FAILED with `{err}`", D::NAME),
    }

    let factored = probe
        .backend
        .with_backend_session(|session| a_dev.lu(session));
    match factored {
        Ok((p, l, u, parity)) => println!(
            "{} lu: succeeded, shapes P={:?} L={:?} U={:?} parity={:?}",
            D::NAME,
            p.shape(),
            l.shape(),
            u.shape(),
            parity.shape()
        ),
        Err(err) => println!("{} lu: FAILED with `{err}`", D::NAME),
    }
}

/// Records exact behaviour rather than asserting success: an `Unsupported` or
/// an NVRTC failure here is the evidence the upstream report needs, and the
/// f64 leg in the same run separates "complex-only defect" from "no CUDA
/// solve at all". Only the f64 result is asserted, because the survey already
/// treats it as a supported path.
#[test]
#[ignore = "requires a real CUDA device"]
fn complex_solve_and_lu_behaviour_is_recorded() {
    let mut probe = Probe::new();
    solve_probe::<f64>(&mut probe);
    solve_probe::<Complex64>(&mut probe);
}
