//! Device tests for `cuda_region_trace_accumulate`, the partial-trace region
//! primitive (issue #1349): a source block read through one merged
//! diagonal axis per traced pair, contracted against the context ones
//! template and accumulated into a strided destination.
//!
//! The oracle walks the *unmerged* source block — both axes of each pair at
//! the same index — so the merged stride is under test, not assumed. Every
//! test is `#[ignore]` like the rest of the device suite.

#![cfg(feature = "cuda")]

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cuda_region_trace_accumulate, cuda_transfer_stats, CudaDenseContext, CudaDenseStorage,
    CudaRegion, CudaScalar, CudaTransferStats, DenseError,
};

trait TraceScalar:
    CudaScalar + Copy + Debug + PartialEq + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    const NAME: &'static str;
    const EPSILON: f64;
    fn from_parts(re: f64, im: f64) -> Self;
    fn conj(self) -> Self;
    fn distance(self, other: Self) -> f64;
    fn sample(index: usize) -> Self {
        Self::from_parts(
            ((index * 37) % 17) as f64 / 8.0 - 1.0,
            ((index * 11) % 13) as f64 / 16.0 - 0.375,
        )
    }
}

impl TraceScalar for f32 {
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;
    fn from_parts(re: f64, _im: f64) -> Self {
        re as Self
    }
    fn conj(self) -> Self {
        self
    }
    fn distance(self, other: Self) -> f64 {
        f64::from((self - other).abs())
    }
}

impl TraceScalar for f64 {
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;
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

impl TraceScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    const EPSILON: f64 = f32::EPSILON as f64;
    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }
    fn distance(self, other: Self) -> f64 {
        f64::from((self - other).norm())
    }
}

impl TraceScalar for Complex64 {
    const NAME: &'static str = "Complex64";
    const EPSILON: f64 = f64::EPSILON;
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

fn delta<T>(body: impl FnOnce() -> T) -> (T, CudaTransferStats) {
    let before = cuda_transfer_stats();
    let value = body();
    let after = cuda_transfer_stats();
    (
        value,
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
            gauge_ops: after.gauge_ops - before.gauge_ops,
        },
    )
}

/// Strides of a column-major layout over `dims` permuted by `order`: axis
/// `order[k]` is the k-th fastest-varying axis.
fn permuted_strides(dims: &[usize], order: &[usize], leading: usize) -> Vec<usize> {
    let mut strides = vec![0usize; dims.len()];
    let mut running = leading;
    for &axis in order {
        strides[axis] = running;
        running *= dims[axis];
    }
    strides
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        visit(&index);
        let mut axis = 0;
        loop {
            if axis == shape.len() {
                return;
            }
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}

/// A block `[out..., t_0, t_0, t_1, t_1, ...]` (pair axes adjacent in the
/// logical order, scattered in memory by `order`) at `base` inside a larger
/// buffer, and a permuted destination at `out_base`.
struct Geometry {
    out: Vec<usize>,
    traced: Vec<usize>,
    block_strides: Vec<usize>,
    base: usize,
    out_strides: Vec<usize>,
    out_base: usize,
}

impl Geometry {
    fn new(out: &[usize], traced: &[usize], order: &[usize], out_order: &[usize]) -> Self {
        let mut block = out.to_vec();
        for &t in traced {
            block.extend([t, t]);
        }
        Self {
            out: out.to_vec(),
            traced: traced.to_vec(),
            block_strides: permuted_strides(&block, order, 2),
            base: 5,
            out_strides: permuted_strides(out, out_order, 3),
            out_base: 4,
        }
    }

    /// The merged read region: one axis per pair, stride `s_lhs + s_rhs`.
    fn src_region(&self) -> CudaRegion {
        let rank = self.out.len();
        let mut dims = self.out.clone();
        let mut strides = self.block_strides[..rank].to_vec();
        for (pair, &t) in self.traced.iter().enumerate() {
            dims.push(t);
            strides.push(
                self.block_strides[rank + 2 * pair] + self.block_strides[rank + 2 * pair + 1],
            );
        }
        CudaRegion::new(dims, strides, self.base).unwrap()
    }

    fn dst_region(&self) -> CudaRegion {
        CudaRegion::new(self.out.clone(), self.out_strides.clone(), self.out_base).unwrap()
    }

    fn src_len(&self) -> usize {
        let mut block = self.out.clone();
        for &t in &self.traced {
            block.extend([t, t]);
        }
        let span: usize = block
            .iter()
            .zip(&self.block_strides)
            .map(|(&d, &s)| d.saturating_sub(1) * s)
            .sum();
        self.base + span + 3
    }

    fn dst_len(&self) -> usize {
        let span: usize = self
            .out
            .iter()
            .zip(&self.out_strides)
            .map(|(&d, &s)| d.saturating_sub(1) * s)
            .sum();
        self.out_base + span + 2
    }

    /// `dst[o] += sum_k alpha * [conj] src[o, k_0, k_0, k_1, k_1, ...]`, the
    /// scale applied to each element before the sum as TensorOperations'
    /// `tensortrace!` does.
    fn oracle<D: TraceScalar>(&self, src: &[D], dst: &[D], conj: bool, alpha: D) -> Vec<D> {
        let rank = self.out.len();
        let mut expected = dst.to_vec();
        for_each_index(&self.out, |o| {
            let mut sum = D::from_parts(0.0, 0.0);
            for_each_index(&self.traced, |k| {
                let mut at = self.base;
                for (axis, &index) in o.iter().enumerate() {
                    at += index * self.block_strides[axis];
                }
                for (pair, &index) in k.iter().enumerate() {
                    at += index * self.block_strides[rank + 2 * pair];
                    at += index * self.block_strides[rank + 2 * pair + 1];
                }
                let value = src[at];
                sum = sum + alpha * if conj { value.conj() } else { value };
            });
            let mut to = self.out_base;
            for (axis, &index) in o.iter().enumerate() {
                to += index * self.out_strides[axis];
            }
            expected[to] = expected[to] + sum;
        });
        expected
    }
}

fn assert_close<D: TraceScalar>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.distance(e) <= 256.0 * D::EPSILON * (1.0 + e.distance(D::from_parts(0.0, 0.0))),
            "{what} [{}] element {index}: {a:?} vs {e:?}",
            D::NAME
        );
    }
}

fn traces_match_the_unmerged_oracle<D: TraceScalar>(ctx: &mut CudaDenseContext) {
    let geometries = [
        // One pair, a rank-2 output, both permuted.
        Geometry::new(&[2, 3], &[4], &[2, 0, 3, 1], &[1, 0]),
        // Two pairs of different extents, interleaved in memory.
        Geometry::new(&[3], &[2, 3], &[3, 0, 4, 1, 2], &[0]),
        // Three pairs and a rank-2 output.
        Geometry::new(&[2, 2], &[2, 3, 2], &[7, 1, 4, 0, 6, 2, 5, 3], &[0, 1]),
        // A full trace over two pairs: a scalar destination.
        Geometry::new(&[], &[3, 2], &[1, 3, 0, 2], &[]),
    ];
    let alphas = [
        D::from_parts(1.0, 0.0),
        D::from_parts(-1.5, 0.0),
        D::from_parts(0.5, -1.25),
    ];
    for (g, geometry) in geometries.iter().enumerate() {
        let src: Vec<D> = (0..geometry.src_len()).map(D::sample).collect();
        let dst: Vec<D> = (0..geometry.dst_len())
            .map(|i| D::sample(i + 500))
            .collect();
        let src_dev = CudaDenseStorage::upload(ctx, &src).unwrap();
        for conj in [false, true] {
            for &alpha in &alphas {
                let mut dst_dev = CudaDenseStorage::upload(ctx, &dst).unwrap();
                cuda_region_trace_accumulate::<D>(
                    ctx,
                    &src_dev,
                    &geometry.src_region(),
                    conj,
                    alpha,
                    &mut dst_dev,
                    &geometry.dst_region(),
                )
                .unwrap();
                let actual = dst_dev.download::<D>(ctx).unwrap();
                let expected = geometry.oracle(&src, &dst, conj, alpha);
                assert_close(
                    &actual,
                    &expected,
                    &format!("geometry {g} conj {conj} alpha {alpha:?}"),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_trace_matches_the_unmerged_host_oracle_at_every_dtype() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    traces_match_the_unmerged_oracle::<f64>(&mut ctx);
    traces_match_the_unmerged_oracle::<Complex64>(&mut ctx);
    traces_match_the_unmerged_oracle::<f32>(&mut ctx);
    traces_match_the_unmerged_oracle::<Complex32>(&mut ctx);
}

/// A zero scale is VectorInterface's `scale(x, 0) = 0` (#1438): it adds
/// nothing and submits nothing, even with a NaN on the traced diagonal.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_scale_adds_nothing_even_over_a_nan_diagonal() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let geometry = Geometry::new(&[2], &[3], &[0, 1, 2], &[0]);
    let mut src: Vec<f64> = (0..geometry.src_len()).map(f64::sample).collect();
    let dst: Vec<f64> = (0..geometry.dst_len())
        .map(|i| f64::sample(i + 500))
        .collect();
    let region = geometry.src_region();
    let at = geometry.base + geometry.block_strides[0] + 2 * (region.strides()[1]);
    src[at] = f64::NAN;
    let poisoned = CudaDenseStorage::upload(&ctx, &src).unwrap();
    let mut out = CudaDenseStorage::upload(&ctx, &dst).unwrap();
    let (_, counters) = delta(|| {
        cuda_region_trace_accumulate::<f64>(
            &mut ctx,
            &poisoned,
            &region,
            false,
            0.0,
            &mut out,
            &geometry.dst_region(),
        )
        .unwrap()
    });
    assert_eq!(counters, CudaTransferStats::default());
    assert_eq!(out.download::<f64>(&ctx).unwrap(), dst);
}

/// The scale reaches every element before the sum, as TensorOperations'
/// `Scaler(α)` inside `_mapreducedim!`: Julia's `tensortrace!(C, [1e308 0;
/// 0 1e308], ((), ()), ((1,), (2,)), false, 0.5, 0.0)` is `1.0e308`, where
/// scaling the traced sum overflows to `Inf`. A reserved scaled template
/// makes a later change of scale a device refill, not an upload.
#[test]
#[ignore = "requires a real CUDA device"]
fn the_scale_applies_per_element_before_the_sum() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let src_region = CudaRegion::new(vec![2], vec![3], 0).unwrap();
    let dst_region = CudaRegion::new(vec![], vec![], 0).unwrap();
    let src = CudaDenseStorage::upload(&ctx, &[1e308, 0.0, 0.0, 1e308]).unwrap();
    ctx.reserve_scaled_template::<f64>(2).unwrap();
    for (alpha, want) in [(0.5, 1e308), (-0.25, -5e307), (0.5, 1e308)] {
        let mut dst = CudaDenseStorage::upload(&ctx, &[0.0]).unwrap();
        let (_, counters) = delta(|| {
            cuda_region_trace_accumulate::<f64>(
                &mut ctx,
                &src,
                &src_region,
                false,
                alpha,
                &mut dst,
                &dst_region,
            )
            .unwrap()
        });
        assert_eq!(counters.h2d_calls, 0, "alpha {alpha}: {counters:?}");
        assert_eq!(dst.download::<f64>(&ctx).unwrap(), [want], "alpha {alpha}");
    }
    let big = Complex64::new(1e308, 1e308);
    let zero = Complex64::new(0.0, 0.0);
    let src = CudaDenseStorage::upload(&ctx, &[big, zero, zero, big]).unwrap();
    let mut dst = CudaDenseStorage::upload(&ctx, &[zero]).unwrap();
    cuda_region_trace_accumulate::<Complex64>(
        &mut ctx,
        &src,
        &src_region,
        false,
        Complex64::new(0.5, 0.0),
        &mut dst,
        &dst_region,
    )
    .unwrap();
    assert_eq!(dst.download::<Complex64>(&ctx).unwrap(), [big]);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_reserved_template_makes_every_trace_upload_free_and_rejections_do_nothing() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let geometry = Geometry::new(&[2, 3], &[4, 2], &[2, 0, 3, 1, 5, 4], &[1, 0]);
    let src: Vec<f64> = (0..geometry.src_len()).map(f64::sample).collect();
    let src_dev = CudaDenseStorage::upload(&ctx, &src).unwrap();
    let mut dst_dev = CudaDenseStorage::upload(&ctx, &vec![0.0; geometry.dst_len()]).unwrap();

    let before = ctx.scalar_operand_bytes();
    ctx.reserve_ones_template::<f64>(8).unwrap();
    assert_eq!(
        ctx.scalar_operand_bytes() - before,
        8 * std::mem::size_of::<f64>(),
        "the ones template is reported"
    );
    let (_, counters) = delta(|| {
        cuda_region_trace_accumulate::<f64>(
            &mut ctx,
            &src_dev,
            &geometry.src_region(),
            false,
            1.0,
            &mut dst_dev,
            &geometry.dst_region(),
        )
        .unwrap()
    });
    assert_eq!(counters.h2d_calls, 0, "{counters:?}");
    assert_eq!(counters.d2h_calls, 0, "{counters:?}");
    assert_eq!(counters.gemm_calls, 1, "{counters:?}");

    // Leading source extents that are not the destination's.
    let wrong = CudaRegion::new(
        vec![3, 2, 4, 2],
        geometry.src_region().strides().to_vec(),
        5,
    )
    .unwrap();
    // A source reaching past its buffer.
    let far = CudaRegion::new(
        geometry.src_region().dims().to_vec(),
        geometry.src_region().strides().to_vec(),
        geometry.src_len(),
    )
    .unwrap();
    // A destination that is not injective.
    let overlapping = CudaRegion::new(vec![2, 3], vec![1, 1], 0).unwrap();
    let (results, counters) = delta(|| {
        [
            cuda_region_trace_accumulate::<f64>(
                &mut ctx,
                &src_dev,
                &wrong,
                false,
                1.0,
                &mut dst_dev,
                &geometry.dst_region(),
            ),
            cuda_region_trace_accumulate::<f64>(
                &mut ctx,
                &src_dev,
                &far,
                false,
                1.0,
                &mut dst_dev,
                &geometry.dst_region(),
            ),
            cuda_region_trace_accumulate::<f64>(
                &mut ctx,
                &src_dev,
                &geometry.src_region(),
                false,
                1.0,
                &mut dst_dev,
                &overlapping,
            ),
            cuda_region_trace_accumulate::<Complex64>(
                &mut ctx,
                &src_dev,
                &geometry.src_region(),
                false,
                Complex64::new(1.0, 0.0),
                &mut dst_dev,
                &geometry.dst_region(),
            ),
        ]
    });
    assert!(
        matches!(results[0], Err(DenseError::ShapeMismatch { .. })),
        "{:?}",
        results[0]
    );
    assert!(
        matches!(results[1], Err(DenseError::OutOfBounds)),
        "{:?}",
        results[1]
    );
    assert!(
        matches!(results[2], Err(DenseError::Unsupported { .. })),
        "{:?}",
        results[2]
    );
    assert!(
        matches!(results[3], Err(DenseError::DTypeMismatch { .. })),
        "{:?}",
        results[3]
    );
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "rejections submit nothing"
    );

    // An empty traced extent adds nothing and submits nothing.
    let empty = CudaRegion::new(vec![2, 3, 0], vec![1, 2, 6], 0).unwrap();
    let (result, counters) = delta(|| {
        cuda_region_trace_accumulate::<f64>(
            &mut ctx,
            &src_dev,
            &empty,
            false,
            1.0,
            &mut dst_dev,
            &geometry.dst_region(),
        )
    });
    result.unwrap();
    assert_eq!(counters, CudaTransferStats::default());
}

/// A rank-8 destination, the largest write view Tenferro 0.8 serves, is
/// traced in place: the template is contracted over the traced mode and the
/// destination keeps its own rank. The fixture is a ten-leg block of extent 2
/// traced over its last pair, read as nine modes with the merged diagonal
/// stride `2^8 + 2^9 = 768`; the oracle sums the two diagonal entries of each
/// output element directly.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_rank_eight_destination_is_traced_without_a_ninth_mode() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let src_strides: Vec<usize> = (0..8).map(|axis| 1 << axis).chain([768]).collect();
    let src_region = CudaRegion::new(vec![2; 9], src_strides, 0).unwrap();
    let dst_region =
        CudaRegion::new(vec![2; 8], (0..8).map(|axis| 1 << axis).collect(), 0).unwrap();
    let src: Vec<f64> = (0..1024).map(f64::sample).collect();
    let dst: Vec<f64> = (0..256).map(|i| f64::sample(i + 2000)).collect();
    let expected: Vec<f64> = (0..256).map(|i| dst[i] + src[i] + src[i + 768]).collect();
    let source = CudaDenseStorage::upload(&ctx, &src).unwrap();
    let mut out = CudaDenseStorage::upload(&ctx, &dst).unwrap();
    cuda_region_trace_accumulate::<f64>(
        &mut ctx,
        &source,
        &src_region,
        false,
        1.0,
        &mut out,
        &dst_region,
    )
    .unwrap();
    let got = out.download::<f64>(&ctx).unwrap();
    for (index, (got, want)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON * (1.0 + want.abs()),
            "element {index}: {got} != {want}"
        );
    }
}
