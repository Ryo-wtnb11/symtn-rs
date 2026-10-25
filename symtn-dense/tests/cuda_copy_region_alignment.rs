//! Device oracle for `cuda_copy_region_into` at arbitrary destination offsets
//! (#1320, #1376).
//!
//! `copy_read_into` folds a view's element offset into the operand pointer.
//! Advertising the allocation's 256-byte alignment for that shifted pointer
//! would let cuTENSOR select a vectorized kernel the pointer cannot satisfy,
//! and the launch fails with `cudaErrorMisalignedAddress` (observed on an A100
//! at `f64` offset 9, `2x2`). Tenferro reports the shifted address's true
//! alignment (`device_address_alignment`), and the adapter sends every offset
//! through that copy.
//!
//! The oracle therefore drives the direct copy — it asserts one copy and no
//! `dot_general` fallback — at destination byte offsets that are not a
//! multiple of 256, including odd element offsets and every power-of-two
//! sub-256-byte shift, for all four payload dtypes. Values are compared
//! **bitwise** against a host scatter over the same index space. Real payloads
//! include negative zero, infinities and subnormals, which cuTENSOR's
//! `alpha = 1` permutation carries exactly; the removed multiply-by-one
//! contraction turned a real `-0.0` into `+0.0`. Complex payloads are finite
//! with no negative-zero component, the set `x * (1, 0)` leaves bit-exact, and
//! NaN payloads are excluded because the permutation canonicalizes `f32` NaNs.
//!
//! Run with `cargo test -p symtn-dense --features cuda,cpu-faer --test \
//! integration cuda_copy_region_alignment:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cuda_copy_region_into, cuda_transfer_stats, CudaDenseContext, CudaDenseStorage, CudaScalar,
};

trait CopyScalar: CudaScalar + Copy {
    const NAME: &'static str;
    fn sample(index: usize) -> Self;
    fn sentinel(index: usize) -> Self;
    fn bits(self) -> (u64, u64);
}

/// Real payloads cycled through the source: infinities, negative zero and the
/// subnormal `tiny`, all of which IEEE multiplication by `1` preserves bit for
/// bit.
fn special(index: usize, tiny: f64) -> Option<f64> {
    match index % 7 {
        1 => Some(f64::INFINITY),
        3 => Some(-0.0),
        5 => Some(tiny),
        _ if index % 11 == 4 => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

/// Finite complex payloads with subnormal and positive-zero components but no
/// `-0.0`, the set multiplication by `(1, 0)` preserves bit for bit.
fn complex_parts(index: usize, tiny: f64) -> (f64, f64) {
    let re = if index % 7 == 5 { tiny } else { finite(index) };
    let im = match index % 5 {
        2 => 0.0,
        4 => -tiny,
        _ => -finite(index) * 0.5,
    };
    (re, im)
}

const TINY_F32: f64 = f32::MIN_POSITIVE as f64 / 8.0;
const TINY_F64: f64 = f64::MIN_POSITIVE / 8.0;

fn finite(index: usize) -> f64 {
    1.0 + index as f64 * 0.25
}

impl CopyScalar for f32 {
    const NAME: &'static str = "f32";
    fn sample(index: usize) -> Self {
        special(index, TINY_F32).unwrap_or(finite(index)) as f32
    }
    fn sentinel(index: usize) -> Self {
        (-1000.0 - index as f64) as f32
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
}

impl CopyScalar for f64 {
    const NAME: &'static str = "f64";
    fn sample(index: usize) -> Self {
        special(index, TINY_F64).unwrap_or(finite(index))
    }
    fn sentinel(index: usize) -> Self {
        -1000.0 - index as f64
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl CopyScalar for Complex32 {
    const NAME: &'static str = "Complex32";
    fn sample(index: usize) -> Self {
        let (re, im) = complex_parts(index, TINY_F32);
        Complex32::new(re as f32, im as f32)
    }
    fn sentinel(index: usize) -> Self {
        Complex32::new((-1000.0 - index as f64) as f32, (7.0 + index as f64) as f32)
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.re.to_bits()), u64::from(self.im.to_bits()))
    }
}

impl CopyScalar for Complex64 {
    const NAME: &'static str = "Complex64";
    fn sample(index: usize) -> Self {
        let (re, im) = complex_parts(index, TINY_F64);
        Complex64::new(re, im)
    }
    fn sentinel(index: usize) -> Self {
        Complex64::new(-1000.0 - index as f64, 7.0 + index as f64)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

/// One copy of a compact `rows x cols` source into `dst[offset..]` with
/// leading dimension `ld`, checked bitwise against a host scatter, and checked
/// to be exactly one direct copy with no contraction and no transfer.
fn copy_case<D: CopyScalar>(
    ctx: &mut CudaDenseContext,
    offset: usize,
    rows: usize,
    cols: usize,
    ld: usize,
) {
    let len = offset + rows + (cols - 1) * ld + 3;
    let source: Vec<D> = (0..rows * cols).map(D::sample).collect();
    let initial: Vec<D> = (0..len).map(D::sentinel).collect();
    let mut expected = initial.clone();
    for col in 0..cols {
        for row in 0..rows {
            expected[offset + row + col * ld] = source[row + col * rows];
        }
    }

    let src = CudaDenseStorage::upload::<D>(ctx, &source).expect("upload source");
    let mut dst = CudaDenseStorage::upload::<D>(ctx, &initial).expect("upload destination");

    let before = cuda_transfer_stats();
    cuda_copy_region_into::<D>(ctx, &mut dst, offset, ld, &src, rows, cols).unwrap_or_else(|err| {
        panic!(
            "{} copy at offset {offset} ({rows}x{cols}, ld {ld}) failed: {err}",
            D::NAME
        )
    });
    let after = cuda_transfer_stats();
    let actual = dst.download::<D>(ctx).unwrap_or_else(|err| {
        panic!(
            "{} download after a copy at offset {offset} ({rows}x{cols}, ld {ld}) failed: {err}",
            D::NAME
        )
    });

    let case = format!(
        "{} {rows}x{cols} copy at offset {offset} ({} bytes), ld {ld}",
        D::NAME,
        offset * std::mem::size_of::<D>()
    );
    assert_eq!(actual.len(), len, "{case}");
    for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(got.bits(), want.bits(), "{case}: element {index}");
    }
    assert_eq!(
        after.copy_calls - before.copy_calls,
        1,
        "{case}: copy calls"
    );
    assert_eq!(
        after.gemm_calls - before.gemm_calls,
        0,
        "{case}: gemm calls"
    );
    assert_eq!(after.h2d_calls, before.h2d_calls, "{case}: h2d calls");
    assert_eq!(after.d2h_calls, before.d2h_calls, "{case}: d2h calls");
    assert_eq!(
        after.device_allocs, before.device_allocs,
        "{case}: device allocations"
    );
}

/// Odd element offsets, #1320's 9 and 25, every power-of-two byte shift below
/// 256 that the element size can express, and the neighbours of the first two
/// 256-byte boundaries. Blocks range from scalars to extents cuTENSOR can
/// vectorize by 2, 4 and 8 along a contiguous or gapped leading dimension.
fn offset_sweep<D: CopyScalar>(ctx: &mut CudaDenseContext) {
    let size = std::mem::size_of::<D>();
    let per_256 = 256 / size;
    let mut offsets = vec![
        0,
        1,
        3,
        5,
        7,
        9,
        25,
        per_256 - 1,
        per_256 + 1,
        2 * per_256 - 1,
    ];
    offsets.extend(
        [4usize, 8, 16, 32, 64, 128]
            .into_iter()
            .filter(|bytes| bytes % size == 0)
            .map(|bytes| bytes / size),
    );
    offsets.push(per_256);
    offsets.sort_unstable();
    offsets.dedup();

    for &offset in &offsets {
        for &(rows, cols) in &[
            (1, 1),
            (2, 2),
            (3, 3),
            (2, 3),
            (3, 2),
            (4, 1),
            (1, 4),
            (8, 8),
            (16, 4),
            (5, 7),
        ] {
            for ld in [rows, rows + 1, rows + 2] {
                copy_case::<D>(ctx, offset, rows, cols, ld);
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn copy_region_is_a_bitwise_direct_copy_at_every_destination_offset() {
    let mut ctx =
        CudaDenseContext::new(0).expect("CUDA device 0 must be available for the device suite");
    offset_sweep::<f32>(&mut ctx);
    offset_sweep::<f64>(&mut ctx);
    offset_sweep::<Complex32>(&mut ctx);
    offset_sweep::<Complex64>(&mut ctx);
}
