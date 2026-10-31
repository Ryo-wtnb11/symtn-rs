//! Device tests for `cuda_region_axpby` / `cuda_region_zero`, the strided
//! N-D region primitive the device structural operations are built on
//!.
//!
//! The oracle in every value test is an independent host strided loop over the
//! same index space; nothing here is derived from a symtn descriptor, and the
//! adapter's own metadata helpers are never used to build an expectation.
//! An earlier probe established the same claim directly against Tenferro; this
//! file re-establishes it through the production seam and adds the cases the
//! probe did not cover: a coefficient that is a 1x1 *data* operand (so 0 and
//! Inf propagate as they do on the host), a poisoned destination, the context
//! zero template as a source, and every rejection.
//!
//! The device is the unit under test, so every test is `#[ignore]` like the
//! rest of the device suite.

#![cfg(feature = "cuda")]

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cuda_region_axpby, cuda_region_zero, cuda_transfer_stats, reset_cuda_transfer_stats,
    CudaDenseContext, CudaDenseStorage, CudaRegion, CudaRegionBeta, CudaRegionCoefficient,
    CudaScalar, DenseError,
};

/// Payload dtypes under test, with the host arithmetic their oracle needs.
trait RegionScalar:
    CudaScalar + Copy + Debug + PartialEq + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    const NAME: &'static str;
    /// Machine epsilon of this payload's real lane, widened.
    const EPSILON: f64;
    /// Relative bound for a move that is not bit-exact.
    ///
    /// The double-precision payloads keep the `1e-12` this file has always
    /// used, because a tighter bound would be this test asserting the
    /// contraction order of whichever GPU runs it rather than the semantics of
    /// the move. The single-precision ones get the *same number of epsilons*
    /// of their own lane, which is what makes the four instantiations one
    /// claim instead of four hand-picked numbers.
    const TOLERANCE: f64 = 1e-12 * (Self::EPSILON / f64::EPSILON);

    fn from_parts(re: f64, im: f64) -> Self;
    fn re(self) -> f64;
    fn im(self) -> f64;
    fn conj(self) -> Self;
    fn distance(self, other: Self) -> f64;
    /// A deterministic, non-degenerate sample value for buffer index `index`.
    fn sample(index: usize) -> Self {
        Self::from_parts(1.0 + (index as f64) * 0.25, -0.5 + (index as f64) * 0.125)
    }
}

impl RegionScalar for f32 {
    const NAME: &'static str = "f32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_parts(re: f64, _im: f64) -> Self {
        re as Self
    }

    fn re(self) -> f64 {
        f64::from(self)
    }

    fn im(self) -> f64 {
        0.0
    }

    fn conj(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        f64::from((self - other).abs())
    }
}

impl RegionScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    const EPSILON: f64 = f32::EPSILON as f64;

    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }

    fn re(self) -> f64 {
        f64::from(self.re)
    }

    fn im(self) -> f64 {
        f64::from(self.im)
    }

    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    fn distance(self, other: Self) -> f64 {
        f64::from((self - other).norm())
    }
}

impl RegionScalar for f64 {
    const NAME: &'static str = "f64";
    const EPSILON: f64 = f64::EPSILON;

    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }

    fn re(self) -> f64 {
        self
    }

    fn im(self) -> f64 {
        0.0
    }

    fn conj(self) -> Self {
        self
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl RegionScalar for Complex64 {
    const NAME: &'static str = "Complex64";
    const EPSILON: f64 = f64::EPSILON;

    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }

    fn re(self) -> f64 {
        self.re
    }

    fn im(self) -> f64 {
        self.im
    }

    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

fn engine() -> symtn_dense::CudaEngine {
    symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0))
        .expect("CUDA device 0 must be available for the device suite")
}

fn upload<D: RegionScalar>(ctx: &CudaDenseContext, data: &[D]) -> CudaDenseStorage {
    CudaDenseStorage::upload::<D>(ctx, data).expect("upload")
}

fn download<D: RegionScalar>(ctx: &CudaDenseContext, storage: &CudaDenseStorage) -> Vec<D> {
    storage.download::<D>(ctx).expect("download")
}

/// Flat buffer positions of a strided region, in column-major index order.
/// Independent of the adapter: this is the oracle's own walk.
fn region_offsets(dims: &[usize], strides: &[usize], base: usize) -> Vec<usize> {
    let total: usize = dims.iter().product();
    (0..total)
        .map(|flat| {
            let mut rest = flat;
            let mut offset = base;
            for (dim, stride) in dims.iter().zip(strides) {
                offset += (rest % dim) * stride;
                rest /= dim;
            }
            offset
        })
        .collect()
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

fn region(dims: &[usize], strides: &[usize], offset: usize) -> CudaRegion {
    CudaRegion::new(dims.to_vec(), strides.to_vec(), offset).expect("region")
}

fn assert_bitwise<D: RegionScalar>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(got, want, "{what}: element {index} ({})", D::NAME);
    }
}

fn assert_close<D: RegionScalar>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        let scale = 1.0_f64.max(want.distance(D::from_parts(0.0, 0.0)));
        assert!(
            got.distance(*want) <= D::TOLERANCE * scale,
            "{what}: element {index} ({}) got {got:?} want {want:?}",
            D::NAME
        );
    }
}

// ---------------------------------------------------------------------------
// Values: rank 3-6 permuted-stride regions at offsets, several blocks per
// buffer, conj, beta, coefficient from a data operand and from the context.
// ---------------------------------------------------------------------------

/// (dims, source axis order, destination axis order). The orders differ, so
/// the destination carries a genuine axis permutation.
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

/// One sweep case: three blocks moved out of one buffer into another, each
/// with its own coefficient read from a *shared* coefficient vector at the
/// block's own offset — exactly how a structure's uploaded coefficients are
/// consumed.
#[allow(clippy::too_many_arguments)]
fn nd_case<D: RegionScalar>(
    ctx: &mut CudaDenseContext,
    dims: &[usize],
    src_order: &[usize],
    dst_order: &[usize],
    coefficients: &[D],
    use_context_one: bool,
    beta: CudaRegionBeta<D>,
    conj: bool,
) {
    let block: usize = dims.iter().product();
    let src_strides = permuted_strides(dims, src_order, 2);
    let dst_strides = permuted_strides(dims, dst_order, 3);
    let src_bases = [7usize, 5 + 2 * block, 3 + 5 * block];
    let dst_bases = [11usize, 2 + 4 * block, 9 + 8 * block];

    let src_host: Vec<D> = (0..16 + 12 * block).map(D::sample).collect();
    let dst_host: Vec<D> = (0..16 + 16 * block)
        .map(|index| D::sample(index + 1000))
        .collect();

    let src = upload::<D>(ctx, &src_host);
    let coeff = upload::<D>(ctx, coefficients);
    let mut dst = upload::<D>(ctx, &dst_host);

    for (block_index, (&src_base, &dst_base)) in src_bases.iter().zip(&dst_bases).enumerate() {
        cuda_region_axpby::<D>(
            ctx,
            &src,
            &region(dims, &src_strides, src_base),
            conj,
            D::ONE,
            if use_context_one {
                CudaRegionCoefficient::One
            } else {
                CudaRegionCoefficient::Buffer(&coeff, block_index)
            },
            beta,
            &mut dst,
            &region(dims, &dst_strides, dst_base),
        )
        .unwrap_or_else(|err| panic!("rank {} region axpby failed: {err}", dims.len()));
    }

    // Independent oracle: the same index space walked on the host.
    let mut expected = dst_host.clone();
    for (block_index, (&src_base, &dst_base)) in src_bases.iter().zip(&dst_bases).enumerate() {
        let coefficient = if use_context_one {
            D::ONE
        } else {
            coefficients[block_index]
        };
        let from_positions = region_offsets(dims, &src_strides, src_base);
        let to_positions = region_offsets(dims, &dst_strides, dst_base);
        for (&from, &to) in from_positions.iter().zip(&to_positions) {
            let value = if conj {
                src_host[from].conj()
            } else {
                src_host[from]
            };
            let kept = match beta {
                CudaRegionBeta::Overwrite => D::ZERO,
                CudaRegionBeta::Accumulate => expected[to],
                CudaRegionBeta::Scale(beta) => beta * expected[to],
            };
            expected[to] = coefficient * value + kept;
        }
    }

    let actual = download::<D>(ctx, &dst);
    let what = format!(
        "rank {} dims {dims:?} src {src_order:?} dst {dst_order:?} conj {conj} context_one {use_context_one}",
        dims.len()
    );
    // A coefficient of 1 is a pure copy, conjugation or single add, and the
    // probe established that the device reproduces the host's element order
    // exactly there. Asserting it bitwise is what makes "finite payloads are
    // unaffected by the multiply" a tested claim rather than a tolerance.
    if (use_context_one || coefficients.iter().all(|value| *value == D::ONE))
        && !matches!(beta, CudaRegionBeta::Scale(_))
    {
        assert_bitwise(&actual, &expected, &what);
    } else {
        assert_close(&actual, &expected, &what);
    }
}

fn nd_sweep<D: RegionScalar>(ctx: &mut CudaDenseContext) {
    let unit = [D::ONE; 3];
    let real = [
        D::from_parts(-0.75, 0.0),
        D::from_parts(2.5, 0.0),
        D::from_parts(0.125, 0.0),
    ];
    let complex = [
        D::from_parts(0.5, -1.25),
        D::from_parts(-1.5, 0.75),
        D::from_parts(0.25, 2.0),
    ];
    for &(dims, src_order, dst_order) in ND_CASES {
        for &conj in &[false, true] {
            // A general beta rides the same epilogue as 0 and 1.
            for &beta in &[
                CudaRegionBeta::Overwrite,
                CudaRegionBeta::Accumulate,
                CudaRegionBeta::Scale(D::from_parts(-0.5, 0.0)),
            ] {
                nd_case::<D>(ctx, dims, src_order, dst_order, &unit, true, beta, conj);
                nd_case::<D>(ctx, dims, src_order, dst_order, &unit, false, beta, conj);
                nd_case::<D>(ctx, dims, src_order, dst_order, &real, false, beta, conj);
                if D::IS_COMPLEX {
                    nd_case::<D>(ctx, dims, src_order, dst_order, &complex, false, beta, conj);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_region_axpby_matches_a_host_strided_loop_f64() {
    nd_sweep::<f64>(&mut CudaDenseContext::new(&mut engine()).unwrap());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_region_axpby_matches_a_host_strided_loop_f32() {
    nd_sweep::<f32>(&mut CudaDenseContext::new(&mut engine()).unwrap());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_region_axpby_matches_a_host_strided_loop_c32() {
    nd_sweep::<Complex32>(&mut CudaDenseContext::new(&mut engine()).unwrap());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nd_region_axpby_matches_a_host_strided_loop_c64() {
    nd_sweep::<Complex64>(&mut CudaDenseContext::new(&mut engine()).unwrap());
}

// ---------------------------------------------------------------------------
// Non-finite behaviour: the reason the coefficient is a data operand.
// ---------------------------------------------------------------------------

fn zero_coefficient_case<D: RegionScalar>(ctx: &mut CudaDenseContext) {
    let dims = [2usize, 2];
    let strides = [1usize, 2];
    let mut src_host: Vec<D> = (0..4).map(D::sample).collect();
    src_host[2] = D::from_parts(f64::NAN, 0.0);
    let src = upload::<D>(ctx, &src_host);
    let coeff = upload::<D>(ctx, &[D::ONE, D::ZERO]);
    let mut dst = upload::<D>(ctx, &[D::ZERO; 4]);

    cuda_region_axpby::<D>(
        ctx,
        &src,
        &region(&dims, &strides, 0),
        false,
        // Coefficient 0, read from offset 1 of the coefficient vector.
        D::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 1),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &strides, 0),
    )
    .expect("zero coefficient");

    let actual = download::<D>(ctx, &dst);
    assert!(
        actual[2].re().is_nan(),
        "{}: a zero *data* coefficient must propagate NaN like the host, got {:?}",
        D::NAME,
        actual[2]
    );
    for (index, value) in actual.iter().enumerate() {
        if index != 2 {
            assert_eq!(*value, D::ZERO, "{}: element {index}", D::NAME);
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_coefficient_propagates_nan_from_the_source() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    zero_coefficient_case::<f32>(&mut ctx);
    zero_coefficient_case::<f64>(&mut ctx);
    zero_coefficient_case::<Complex32>(&mut ctx);
    zero_coefficient_case::<Complex64>(&mut ctx);
}

fn poisoned_destination_case<D: RegionScalar>(ctx: &mut CudaDenseContext) {
    let dims = [2usize, 3];
    let src_strides = [1usize, 2];
    let dst_strides = [3usize, 1];
    let src_host: Vec<D> = (0..6).map(D::sample).collect();
    let poison = D::from_parts(f64::NAN, f64::NAN);
    let src = upload::<D>(ctx, &src_host);
    let mut dst = upload::<D>(ctx, &[poison; 6]);

    cuda_region_axpby::<D>(
        ctx,
        &src,
        &region(&dims, &src_strides, 0),
        false,
        D::ONE,
        CudaRegionCoefficient::One,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &dst_strides, 0),
    )
    .expect("overwrite a poisoned destination");

    let actual = download::<D>(ctx, &dst);
    let from = region_offsets(&dims, &src_strides, 0);
    let to = region_offsets(&dims, &dst_strides, 0);
    let mut expected = [poison; 6];
    for (&from, &to) in from.iter().zip(&to) {
        expected[to] = src_host[from];
    }
    for (index, value) in actual.iter().enumerate() {
        assert_eq!(
            *value,
            expected[index],
            "{}: overwrite must be destination independent at {index}",
            D::NAME
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_is_independent_of_a_nan_poisoned_destination() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    poisoned_destination_case::<f32>(&mut ctx);
    poisoned_destination_case::<f64>(&mut ctx);
    poisoned_destination_case::<Complex32>(&mut ctx);
    poisoned_destination_case::<Complex64>(&mut ctx);
}

/// Non-finite payloads are accepted and propagated, never erased: the
/// structural coefficient is a data operand, so an infinity stays non-finite
/// through a coefficient of 1, and a real infinity keeps its value. Which
/// non-finite value a complex infinity becomes (the backend multiplies where
/// the host copies) is the dense backend's arithmetic, not asserted here.
#[test]
#[ignore = "requires a real CUDA device"]
fn non_finite_payloads_are_accepted_and_propagated() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let dims = [2usize];
    let strides = [1usize];

    let real_src = upload::<f64>(&ctx, &[f64::INFINITY, f64::NEG_INFINITY]);
    let mut real_dst = upload::<f64>(&ctx, &[0.0, 0.0]);
    cuda_region_axpby::<f64>(
        &mut ctx,
        &real_src,
        &region(&dims, &strides, 0),
        false,
        f64::ONE,
        CudaRegionCoefficient::One,
        CudaRegionBeta::Overwrite,
        &mut real_dst,
        &region(&dims, &strides, 0),
    )
    .expect("real infinity");
    let real = download::<f64>(&ctx, &real_dst);
    assert_eq!(real, vec![f64::INFINITY, f64::NEG_INFINITY]);

    let complex_src = upload::<Complex64>(
        &ctx,
        &[
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
        ],
    );
    let mut complex_dst = upload::<Complex64>(&ctx, &[Complex64::new(0.0, 0.0); 2]);
    cuda_region_axpby::<Complex64>(
        &mut ctx,
        &complex_src,
        &region(&dims, &strides, 0),
        false,
        Complex64::ONE,
        CudaRegionCoefficient::One,
        CudaRegionBeta::Overwrite,
        &mut complex_dst,
        &region(&dims, &strides, 0),
    )
    .expect("complex infinity");
    let complex = download::<Complex64>(&ctx, &complex_dst);
    for value in &complex {
        assert!(
            !(value.re.is_finite() && value.im.is_finite()),
            "a complex infinity must stay non-finite, got {value:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// The context-owned zero template as a source.
// ---------------------------------------------------------------------------

fn zero_fill_case<D: RegionScalar>(ctx: &mut CudaDenseContext) {
    let dims = [2usize, 3, 2];
    let strides = permuted_strides(&dims, &[2, 0, 1], 3);
    let base = 5usize;
    let host: Vec<D> = (0..64)
        .map(|index| D::from_parts(f64::NAN, index as f64))
        .collect();
    let mut dst = upload::<D>(ctx, &host);

    cuda_region_zero::<D>(ctx, &mut dst, &region(&dims, &strides, base)).expect("zero fill");

    let actual = download::<D>(ctx, &dst);
    let zeroed = region_offsets(&dims, &strides, base);
    let mut expected = host.clone();
    for &at in &zeroed {
        expected[at] = D::ZERO;
    }
    for (index, value) in actual.iter().enumerate() {
        if zeroed.contains(&index) {
            assert_eq!(
                *value,
                D::ZERO,
                "{}: position {index} must be zero",
                D::NAME
            );
        } else {
            assert_eq!(
                value.im(),
                expected[index].im(),
                "{}: position {index} is outside the region and must be untouched",
                D::NAME
            );
            assert!(value.re().is_nan(), "{}: position {index}", D::NAME);
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_context_zero_template_overwrites_exactly_its_region() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    zero_fill_case::<f32>(&mut ctx);
    zero_fill_case::<f64>(&mut ctx);
    zero_fill_case::<Complex32>(&mut ctx);
    zero_fill_case::<Complex64>(&mut ctx);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_zero_template_is_uploaded_once_per_dtype_and_grows_monotonically() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let mut dst = upload::<f64>(&ctx, &[1.0f64; 64]);

    // First fill of this dtype: the `1` operand and a 16-element template.
    reset_cuda_transfer_stats();
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[4, 4], &[1, 4], 0)).expect("first");
    let first = cuda_transfer_stats();
    assert_eq!(first.h2d_calls, 2, "ones plus the zero template");
    assert_eq!(first.d2h_calls, 0);

    // A shorter region is a prefix of the same template, and the `1` operand
    // is already resident: no transfer at all.
    reset_cuda_transfer_stats();
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[2, 2], &[1, 8], 16)).expect("shorter");
    assert_eq!(cuda_transfer_stats().h2d_calls, 0);

    // A longer region replaces the template exactly once.
    reset_cuda_transfer_stats();
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[8, 4], &[1, 8], 32)).expect("longer");
    assert_eq!(cuda_transfer_stats().h2d_calls, 1);
    reset_cuda_transfer_stats();
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[8, 4], &[1, 8], 32)).expect("again");
    assert_eq!(cuda_transfer_stats().h2d_calls, 0);

    // Every *other* dtype has its own pair, and taking one does not disturb
    // the others: the operands are payload-typed, so a slot shared between
    // `f32` and `f64` (or between the two complex dtypes) would hand the next
    // call a buffer of the wrong dtype. Each of the three remaining dtypes
    // therefore pays its own two uploads, and the `f64` pair stays resident
    // throughout.
    let mut single_dst = upload::<f32>(&ctx, &[1.0f32; 64]);
    reset_cuda_transfer_stats();
    cuda_region_zero::<f32>(&mut ctx, &mut single_dst, &region(&[4, 4], &[1, 4], 0)).expect("f32");
    assert_eq!(cuda_transfer_stats().h2d_calls, 2, "f32 owns its own pair");

    let mut single_complex_dst = upload::<Complex32>(&ctx, &[Complex32::new(1.0, 1.0); 64]);
    reset_cuda_transfer_stats();
    cuda_region_zero::<Complex32>(
        &mut ctx,
        &mut single_complex_dst,
        &region(&[4, 4], &[1, 4], 0),
    )
    .expect("Complex32");
    assert_eq!(
        cuda_transfer_stats().h2d_calls,
        2,
        "Complex32 owns its own pair"
    );

    let mut complex_dst = upload::<Complex64>(&ctx, &[Complex64::new(1.0, 1.0); 64]);
    reset_cuda_transfer_stats();
    cuda_region_zero::<Complex64>(&mut ctx, &mut complex_dst, &region(&[4, 4], &[1, 4], 0))
        .expect("complex");
    assert_eq!(cuda_transfer_stats().h2d_calls, 2);
    reset_cuda_transfer_stats();
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[8, 4], &[1, 8], 32)).expect("f64 again");
    assert_eq!(cuda_transfer_stats().h2d_calls, 0);
    reset_cuda_transfer_stats();

    // The pinned per-dtype byte accounting: each slot charges its own element
    // size, so the four resident pairs are not all counted as `f64`.
    let expected_bytes = (32 + 1) * std::mem::size_of::<f64>()
        + (16 + 1) * std::mem::size_of::<f32>()
        + (16 + 1) * std::mem::size_of::<Complex32>()
        + (16 + 1) * std::mem::size_of::<Complex64>();
    assert_eq!(ctx.scalar_operand_bytes(), expected_bytes);
    ctx.release_scalar_operands();
    assert_eq!(ctx.scalar_operand_bytes(), 0);
}

// ---------------------------------------------------------------------------
// Layout forms the structural operations need: a broadcast (outer-product)
// source and a diagonal source view.
// ---------------------------------------------------------------------------

/// `dst[i, j] = c * src[i]`: a stride-0 *source* axis is the outer product of
/// the source vector with ones. Stride 0 is rejected on the destination
/// (it would write one position repeatedly) and accepted on the source.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_broadcast_source_axis_forms_an_outer_product() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let rows = 3usize;
    let cols = 4usize;
    let src_host: Vec<f64> = (0..rows).map(|index| 1.0 + index as f64).collect();
    let src = upload::<f64>(&ctx, &src_host);
    let coeff = upload::<f64>(&ctx, &[0.0, -0.5]);
    let dst_host = vec![7.0f64; rows * cols];
    let mut dst = upload::<f64>(&ctx, &dst_host);

    cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&[rows, cols], &[1, 0], 0),
        false,
        f64::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 1),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&[rows, cols], &[1, rows], 0),
    )
    .expect("broadcast source");

    let actual = download::<f64>(&ctx, &dst);
    let mut expected = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            expected[col * rows + row] = -0.5 * src_host[row];
        }
    }
    assert_close(&actual, &expected, "broadcast outer product");
}

/// The diagonal of a strided square block, read as one axis of stride
/// `s_row + s_col` and written into a packed vector. This is the read view the
/// device trace is built on; the trace sums over it.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_diagonal_source_view_extracts_the_block_diagonal() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let n = 4usize;
    let block_strides = [3usize, 3 * n];
    let base = 5usize;
    let src_host: Vec<Complex64> = (0..128).map(Complex64::sample).collect();
    let src = upload::<Complex64>(&ctx, &src_host);
    let dst_host = vec![Complex64::new(9.0, 9.0); n];
    let mut dst = upload::<Complex64>(&ctx, &dst_host);

    cuda_region_axpby::<Complex64>(
        &mut ctx,
        &src,
        &region(&[n], &[block_strides[0] + block_strides[1]], base),
        true,
        Complex64::ONE,
        CudaRegionCoefficient::One,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&[n], &[1], 0),
    )
    .expect("diagonal view");

    let actual = download::<Complex64>(&ctx, &dst);
    // Oracle: the diagonal of the *unmerged* block, so the merged stride is
    // under test rather than assumed.
    let expected: Vec<Complex64> = (0..n)
        .map(|index| {
            Complex64::conj(&src_host[base + index * block_strides[0] + index * block_strides[1]])
        })
        .collect();
    assert_bitwise(&actual, &expected, "conjugated block diagonal");
}

/// A sub-block of a wider parent: the slower axis steps by the parent's
/// leading dimension, so its stride strictly *exceeds* the span of the faster
/// axis instead of being a multiple of it. This is the real destination form
/// of a transform that writes into one sector of a larger allocation, and the
/// interleaved `dims [2,2] strides [2,3]` layout below is the boundary case of
/// the host's cumulative-span rule — admitted here exactly as the host admits
/// it, and accepted by the backend.
#[test]
#[ignore = "requires a real CUDA device"]
fn gapped_and_interleaved_destinations_are_accepted() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let src_host: Vec<f64> = (0..64).map(|index| 1.0 + index as f64).collect();
    let src = upload::<f64>(&ctx, &src_host);

    // Gapped: a 2x3 block inside a parent whose leading dimension is 5.
    let dims = [2usize, 3];
    let src_strides = [1usize, 2];
    let dst_strides = [1usize, 5];
    let parent: Vec<f64> = (0..32).map(|index| -(index as f64)).collect();
    let mut dst = upload::<f64>(&ctx, &parent);
    cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&dims, &src_strides, 3),
        false,
        f64::ONE,
        CudaRegionCoefficient::One,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &dst_strides, 4),
    )
    .expect("a gapped destination sub-block");

    let actual = download::<f64>(&ctx, &dst);
    let mut expected = parent.clone();
    for (&from, &to) in region_offsets(&dims, &src_strides, 3)
        .iter()
        .zip(&region_offsets(&dims, &dst_strides, 4))
    {
        expected[to] = src_host[from];
    }
    assert_bitwise(&actual, &expected, "gapped destination sub-block");

    // Interleaved but injective: positions 0, 2, 3, 5 of the destination.
    let dims = [2usize, 2];
    let dst_strides = [2usize, 3];
    let mut dst = upload::<f64>(&ctx, &parent);
    cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&dims, &[1, 2], 0),
        false,
        f64::ONE,
        CudaRegionCoefficient::One,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &dst_strides, 1),
    )
    .expect("an interleaved-but-injective destination the host also proves");

    let actual = download::<f64>(&ctx, &dst);
    let mut expected = parent.clone();
    for (&from, &to) in region_offsets(&dims, &[1, 2], 0)
        .iter()
        .zip(&region_offsets(&dims, &dst_strides, 1))
    {
        expected[to] = src_host[from];
    }
    assert_bitwise(&actual, &expected, "interleaved destination");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_reserved_zero_template_makes_every_later_fill_transfer_free() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    assert_eq!(ctx.scalar_operand_bytes(), 0);
    let mut dst = upload::<f64>(&ctx, &[1.0f64; 64]);

    // Size once from what the caller knows, then fill in any order.
    ctx.reserve_zero_template::<f64>(32).expect("reserve");
    let reserved = ctx.scalar_operand_bytes();
    assert_eq!(reserved, 32 * std::mem::size_of::<f64>());

    reset_cuda_transfer_stats();
    for region_dims in [[8usize, 4], [2, 2], [4, 4]] {
        cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&region_dims, &[1, 8], 0))
            .expect("fill");
    }
    // Only the one-element `1` is still missing; the template is never
    // re-uploaded, where ascending fills without the reservation would have
    // paid one upload each.
    let stats = cuda_transfer_stats();
    assert_eq!(stats.h2d_calls, 1, "{stats:?}");
    assert_eq!(
        ctx.scalar_operand_bytes(),
        reserved + std::mem::size_of::<f64>()
    );

    // Releasing frees both, and the next call re-creates what it needs.
    ctx.release_scalar_operands();
    assert_eq!(ctx.scalar_operand_bytes(), 0);
    reset_cuda_transfer_stats();
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[2, 2], &[1, 8], 0))
        .expect("after release");
    assert_eq!(cuda_transfer_stats().h2d_calls, 2);
    reset_cuda_transfer_stats();
}

// ---------------------------------------------------------------------------
// Rejections and the zero-transfer contract.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn every_rejection_is_typed_and_submits_no_device_work() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let src = upload::<f64>(&ctx, &(0..64).map(|index| index as f64).collect::<Vec<_>>());
    let coeff = upload::<f64>(&ctx, &[2.0, 3.0]);
    let complex = upload::<Complex64>(&ctx, &[Complex64::new(1.0, 0.0); 64]);
    let mut dst = upload::<f64>(&ctx, &[0.0f64; 64]);
    let mut complex_dst = upload::<Complex64>(&ctx, &[Complex64::new(0.0, 0.0); 64]);
    let good = region(&[2, 3], &[1, 2], 0);

    reset_cuda_transfer_stats();

    // Source and destination extents must agree.
    let err = cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&[2, 3], &[1, 2], 0),
        false,
        f64::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 0),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&[3, 2], &[1, 3], 0),
    )
    .expect_err("dims must match");
    assert!(
        matches!(err, DenseError::ShapeMismatch { op: "cuda_region_axpby", ref expected, ref actual }
            if expected == &[3, 2] && actual == &[2, 3]),
        "{err}"
    );

    // The coefficient offset must be inside the coefficient buffer.
    let err = cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &good,
        false,
        f64::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 2),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &good,
    )
    .expect_err("coefficient offset must be in bounds");
    assert!(matches!(err, DenseError::OutOfBounds), "{err}");

    // Neither region may run past its buffer.
    for (src_region, dst_region) in [
        (region(&[2, 3], &[1, 32], 0), good.clone()),
        (good.clone(), region(&[2, 3], &[1, 32], 0)),
    ] {
        let err = cuda_region_axpby::<f64>(
            &mut ctx,
            &src,
            &src_region,
            false,
            f64::ONE,
            CudaRegionCoefficient::Buffer(&coeff, 0),
            CudaRegionBeta::Overwrite,
            &mut dst,
            &dst_region,
        )
        .expect_err("region must stay inside its buffer");
        assert!(matches!(err, DenseError::OutOfBounds), "{err}");
    }

    // A destination that writes one position twice is an explicit boundary.
    let err = cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&[2, 3], &[1, 2], 0),
        false,
        f64::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 0),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&[2, 3], &[1, 0], 0),
    )
    .expect_err("a repeated destination axis must be rejected");
    assert!(
        matches!(
            err,
            DenseError::Unsupported {
                op: "cuda_region_axpby",
                ..
            }
        ),
        "{err}"
    );
    let err = cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[2, 3], &[1, 0], 0))
        .expect_err("the fill validates its destination the same way");
    assert!(
        matches!(
            err,
            DenseError::Unsupported {
                op: "cuda_region_zero",
                ..
            }
        ),
        "{err}"
    );

    // A payload dtype mismatch is typed, never a reinterpretation.
    let err = cuda_region_axpby::<f64>(
        &mut ctx,
        &complex,
        &good,
        false,
        f64::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 0),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &good,
    )
    .expect_err("dtype must match");
    assert!(
        matches!(err, DenseError::DTypeMismatch { .. }),
        "{err}: {err:?}"
    );
    let err = cuda_region_zero::<f64>(&mut ctx, &mut complex_dst, &good)
        .expect_err("the fill checks the destination dtype too");
    assert!(matches!(err, DenseError::DTypeMismatch { .. }), "{err}");

    // Nothing above reached the device: no transfer, no allocation, no
    // submission.
    let stats = cuda_transfer_stats();
    assert_eq!(stats.h2d_calls, 0, "{stats:?}");
    assert_eq!(stats.device_allocs, 0, "{stats:?}");
    assert_eq!(stats.d2h_calls, 0, "{stats:?}");
    assert_eq!(stats.gemm_calls, 0, "{stats:?}");
    assert_eq!(stats.solver_calls, 0, "{stats:?}");
    assert_eq!(stats.copy_calls, 0, "{stats:?}");
    reset_cuda_transfer_stats();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_extent_region_is_a_no_op_without_a_submission() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let src = upload::<f64>(&ctx, &(0..16).map(|index| index as f64).collect::<Vec<_>>());
    let coeff = upload::<f64>(&ctx, &[2.0]);
    let host: Vec<f64> = (0..16).map(|index| -(index as f64)).collect();
    let mut dst = upload::<f64>(&ctx, &host);

    reset_cuda_transfer_stats();
    cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&[2, 0, 3], &[1, 2, 2], 0),
        false,
        f64::ONE,
        CudaRegionCoefficient::Buffer(&coeff, 0),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&[2, 0, 3], &[1, 2, 2], 0),
    )
    .expect("a zero-extent region is accepted");
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&[0], &[1], 0))
        .expect("a zero-extent fill is accepted");
    let stats = cuda_transfer_stats();
    assert_eq!(stats.gemm_calls, 0, "{stats:?}");
    assert_eq!(stats.h2d_calls, 0, "{stats:?}");
    assert_eq!(stats.device_allocs, 0, "{stats:?}");
    reset_cuda_transfer_stats();

    assert_bitwise(
        &download::<f64>(&ctx, &dst),
        &host,
        "a zero-extent region touches nothing",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_call_phase_moves_nothing_across_the_host_boundary() {
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let dims = [3usize, 2, 4];
    let src_strides = permuted_strides(&dims, &[0, 1, 2], 1);
    let dst_strides = permuted_strides(&dims, &[2, 0, 1], 1);
    let block: usize = dims.iter().product();
    let src = upload::<f64>(&ctx, &(0..8 * block).map(|i| i as f64).collect::<Vec<_>>());
    let coeff = upload::<f64>(&ctx, &[1.5; 8]);
    let dst_host = vec![0.0f64; 8 * block];
    let mut dst = upload::<f64>(&ctx, &dst_host);

    // Warm both context operands so the lazy per-dtype upload is not counted
    // as call-phase traffic; it is a one-off, asserted in its own test.
    cuda_region_zero::<f64>(&mut ctx, &mut dst, &region(&dims, &dst_strides, 0)).expect("warm");

    reset_cuda_transfer_stats();
    for block_index in 0..8 {
        cuda_region_axpby::<f64>(
            &mut ctx,
            &src,
            &region(&dims, &src_strides, block_index * block),
            false,
            f64::ONE,
            CudaRegionCoefficient::Buffer(&coeff, block_index),
            CudaRegionBeta::Accumulate,
            &mut dst,
            &region(&dims, &dst_strides, block_index * block),
        )
        .expect("replay block");
        cuda_region_zero::<f64>(
            &mut ctx,
            &mut dst,
            &region(&dims, &dst_strides, block_index * block),
        )
        .expect("zero block");
    }
    let stats = cuda_transfer_stats();
    assert_eq!(stats.h2d_calls, 0, "{stats:?}");
    assert_eq!(stats.h2d_bytes, 0, "{stats:?}");
    assert_eq!(stats.d2h_calls, 0, "{stats:?}");
    assert_eq!(stats.device_allocs, 0, "{stats:?}");
    assert_eq!(stats.gemm_calls, 16, "one submission per region call");
    reset_cuda_transfer_stats();
}

/// `dst = alpha * c * [conj] src` over a strided region, against a host loop.
fn scaled_move_matches_a_host_loop<D: RegionScalar>(ctx: &mut CudaDenseContext, alpha: D) {
    let dims = [2usize, 3];
    let src_strides = [3usize, 1];
    let dst_strides = [1usize, 2];
    let src_host: Vec<D> = (0..6).map(D::sample).collect();
    let coefficient = D::from_parts(-0.75, 0.5);
    let src = upload::<D>(ctx, &src_host);
    let coeff = upload::<D>(ctx, &[D::ONE, coefficient]);
    let dst_host = vec![D::from_parts(9.0, -9.0); 6];
    let mut dst = upload::<D>(ctx, &dst_host);

    cuda_region_axpby::<D>(
        ctx,
        &src,
        &region(&dims, &src_strides, 0),
        false,
        alpha,
        CudaRegionCoefficient::Buffer(&coeff, 1),
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &dst_strides, 0),
    )
    .expect("a scaled move");

    let actual = download::<D>(ctx, &dst);
    let mut expected = dst_host.clone();
    for (&from, &to) in region_offsets(&dims, &src_strides, 0)
        .iter()
        .zip(&region_offsets(&dims, &dst_strides, 0))
    {
        expected[to] = alpha * coefficient * src_host[from];
    }
    for (index, (left, right)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            left.distance(*right) <= 1e-12,
            "{}: element {index} is {left:?}, expected {right:?}",
            D::NAME
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_descriptor_scale_multiplies_the_move_in_both_dtypes() {
    // What: the caller's scale rides the contraction descriptor while the
    // structural coefficient stays a data operand, so the written values are
    // the product of both — for a real scale and for a genuinely complex one,
    // whose imaginary part a real-only descriptor would drop.
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    scaled_move_matches_a_host_loop::<f64>(&mut ctx, -2.5);
    scaled_move_matches_a_host_loop::<Complex64>(&mut ctx, Complex64::new(0.5, -1.25));
    scaled_move_matches_a_host_loop::<Complex64>(&mut ctx, Complex64::new(-2.5, 0.0));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_zero_coefficient_operand_multiplies_on_a_fresh_context() {
    // What: `CudaRegionCoefficient::Zero` is an exact zero *operand*, so the
    // source is still read and multiplied — a NaN survives it, where a
    // descriptor scale of zero (rejected) would erase it. The context
    // is fresh, so this also executes the lazy one-element template upload and
    // proves the second call needs none.
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let dims = [2usize, 2];
    let strides = [1usize, 2];
    let poisoned = upload::<f64>(&ctx, &[f64::NAN, 1.0, f64::INFINITY, 2.0]);
    let mut dst = upload::<f64>(&ctx, &[7.0; 4]);

    reset_cuda_transfer_stats();
    let before = cuda_transfer_stats();
    cuda_region_axpby::<f64>(
        &mut ctx,
        &poisoned,
        &region(&dims, &strides, 0),
        false,
        1.0,
        CudaRegionCoefficient::Zero,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &strides, 0),
    )
    .expect("a zero coefficient operand");
    let cold = cuda_transfer_stats();

    let actual = download::<f64>(&ctx, &dst);
    assert!(actual[0].is_nan(), "0 * NaN must be NaN: {actual:?}");
    assert!(actual[2].is_nan(), "0 * inf must be NaN: {actual:?}");
    assert_eq!(actual[1], 0.0, "0 * finite must be zero: {actual:?}");
    assert_eq!(actual[3], 0.0, "0 * finite must be zero: {actual:?}");
    assert!(
        cold.h2d_calls > before.h2d_calls,
        "a fresh context must create its scalar operands: {before:?} -> {cold:?}"
    );

    // Warm: the one-element template is resident, so nothing crosses again.
    let finite = upload::<f64>(&ctx, &[3.0; 4]);
    let warm_before = cuda_transfer_stats();
    cuda_region_axpby::<f64>(
        &mut ctx,
        &finite,
        &region(&dims, &strides, 0),
        false,
        1.0,
        CudaRegionCoefficient::Zero,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&dims, &strides, 0),
    )
    .expect("a warm zero coefficient operand");
    let warm = cuda_transfer_stats();
    assert_eq!(
        warm.h2d_calls, warm_before.h2d_calls,
        "a warm zero operand must upload nothing: {warm_before:?} -> {warm:?}"
    );
    assert!(download::<f64>(&ctx, &dst)
        .iter()
        .all(|value| *value == 0.0));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_descriptor_scale_and_a_rejected_zero_operand_cost_nothing() {
    // What: the zero-scale rejection and every other rejection happen before
    // any device work — including before the lazy zero-template upload the
    // accepted call would have done.
    let mut engine = engine();
    let mut ctx = CudaDenseContext::new(&mut engine).unwrap();
    let src = upload::<f64>(&ctx, &[1.0, 2.0, 3.0, 4.0]);
    let mut dst = upload::<f64>(&ctx, &[0.0; 4]);

    reset_cuda_transfer_stats();
    let before = cuda_transfer_stats();

    for alpha in [0.0_f64, -0.0] {
        let err = cuda_region_axpby::<f64>(
            &mut ctx,
            &src,
            &region(&[2, 2], &[1, 2], 0),
            false,
            alpha,
            CudaRegionCoefficient::One,
            CudaRegionBeta::Overwrite,
            &mut dst,
            &region(&[2, 2], &[1, 2], 0),
        )
        .expect_err("a zero descriptor scale must be rejected");
        assert!(
            matches!(
                err,
                DenseError::Unsupported {
                    op: "cuda_region_axpby",
                    ..
                }
            ),
            "{err}"
        );
    }

    // A zero *operand* call that fails validation must not upload the template
    // either: the extents disagree, which is checked first.
    let err = cuda_region_axpby::<f64>(
        &mut ctx,
        &src,
        &region(&[2, 2], &[1, 2], 0),
        false,
        1.0,
        CudaRegionCoefficient::Zero,
        CudaRegionBeta::Overwrite,
        &mut dst,
        &region(&[4, 1], &[1, 4], 0),
    )
    .expect_err("mismatched extents must be rejected");
    assert!(
        matches!(
            err,
            DenseError::ShapeMismatch {
                op: "cuda_region_axpby",
                ..
            }
        ),
        "{err}"
    );

    let after = cuda_transfer_stats();
    assert_eq!(
        (after.h2d_calls, after.device_allocs, after.gemm_calls),
        (before.h2d_calls, before.device_allocs, before.gemm_calls),
        "a rejection did device work: {before:?} -> {after:?}"
    );
}
