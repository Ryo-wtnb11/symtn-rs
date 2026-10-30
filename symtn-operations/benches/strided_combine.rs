//! Regression microbench for issue #230: the non-contiguous strided-combine
//! recurse path in `host_scalar_kernels`. This is the regime a rank-4 SU(2)
//! d=4 contract feeds — a recoupling/permute block combine whose source is not
//! column-major-contiguous, so `raw_strided_combine_recurse` walks it element
//! by element through `checked_strided_offset` + `checked_offset_to_index`.
//!
//! No timing assertion here: shared CI runners are too noisy for a pass/fail
//! latency gate (same rationale as benches/fusion_tree_key.rs in symtn-core).
//! The gate is this bench compiling and running, plus the `size_of` canary
//! `offset_error_result_stays_small` in host_scalar_kernels.rs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symtn_operations::tensoradd_raw_strided_kernel;

// Rank-4, d=4 block: 256 elements, the SU(2) recoupling block size.
const D: usize = 4;
const RANK: usize = 4;
const LEN: usize = D * D * D * D;

fn bench_noncontiguous_combine(c: &mut Criterion) {
    let shape = [D; RANK];
    // Destination is column-major-contiguous; source reads as a transposed
    // (row-major) layout. Mismatched contiguity forces the per-element recurse
    // path instead of the fast contiguous slice copy — exactly the recoupling
    // access pattern that regressed in #230.
    let dst_strides: [isize; RANK] = [1, 4, 16, 64];
    let src_strides: [isize; RANK] = [64, 16, 4, 1];

    let mut dst = vec![0.0f64; LEN];
    let src = vec![1.0f64; LEN];
    let mut zero_strides: Vec<isize> = Vec::new();

    c.bench_function("strided_combine_noncontiguous_rank4_d4", |bencher| {
        bencher.iter(|| {
            tensoradd_raw_strided_kernel(
                strided_kernel::Exec::serial(),
                &mut zero_strides,
                black_box(&mut dst),
                black_box(&src),
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                false,
                black_box(1.0f64),
                black_box(1.0f64),
            )
            .unwrap();
        })
    });
}

/// Logical rank-8 layout `[4, 4, 4, 4, 2, 2, 2, 2]` (4096 elements) with a
/// padded column-major destination (extent `dim + 1`) and a reversed
/// column-major source, so no two logical axes fuse.
type Layout = (Vec<usize>, Vec<isize>, Vec<isize>);

fn logical_rank8() -> Layout {
    let dims = vec![4usize, 4, 4, 4, 2, 2, 2, 2];
    padded_reversed(dims)
}

fn padded_reversed(dims: Vec<usize>) -> Layout {
    let mut dst = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in &dims {
        dst.push(stride);
        stride *= dim as isize + 1;
    }
    let mut src = vec![0isize; dims.len()];
    let mut stride = 1isize;
    for axis in (0..dims.len()).rev() {
        src[axis] = stride;
        stride *= dims[axis] as isize;
    }
    (dims, dst, src)
}

/// Issue #2230 layouts: native rank 8; the same layout at raw rank 12 with
/// four unit axes; at raw rank 12 with the four extent-4 axes split into
/// contiguous `(2, 2)` pairs; and a true rank-12 layout that fuses to 12.
fn rank_gate_layouts() -> Vec<(&'static str, Layout)> {
    let (dims, dst, src) = logical_rank8();
    let mut unit = (dims.clone(), dst.clone(), src.clone());
    for position in [1usize, 4, 7, 10] {
        unit.0.insert(position, 1);
        unit.1.insert(position, 7);
        unit.2.insert(position, 5);
    }
    let mut split = (Vec::new(), Vec::new(), Vec::new());
    for axis in 0..dims.len() {
        if dims[axis] == 4 {
            split.0.extend([2, 2]);
            split.1.extend([dst[axis], 2 * dst[axis]]);
            split.2.extend([src[axis], 2 * src[axis]]);
        } else {
            split.0.push(dims[axis]);
            split.1.push(dst[axis]);
            split.2.push(src[axis]);
        }
    }
    let true12 = padded_reversed(vec![2usize; 12]);
    vec![
        ("native_r8", (dims, dst, src)),
        ("r12_unit_fuses_to_8", unit),
        ("r12_split_fuses_to_8", split),
        ("r12_fuses_to_12", true12),
    ]
}

fn bench_fused_rank_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_rank_gate");
    let mut zero_strides: Vec<isize> = Vec::new();
    for (name, (dims, dst_strides, src_strides)) in rank_gate_layouts() {
        let src_len: usize = dims.iter().product();
        let dst_len: usize = 1 + dims
            .iter()
            .zip(&dst_strides)
            .map(|(&dim, &stride)| (dim - 1) * stride as usize)
            .sum::<usize>();
        let src: Vec<f64> = (0..src_len).map(|i| i as f64 * 0.5).collect();
        let mut dst = vec![1.0f64; dst_len];
        // `axpy` reaches Strided's raw kernel when the fused rank allows it;
        // `axpby` always stays on symtn's own loop.
        for (action, alpha, beta) in [("axpy", 0.5f64, 1.0f64), ("axpby", 0.5, 0.25)] {
            group.bench_function(format!("{name}_{action}"), |bencher| {
                bencher.iter(|| {
                    tensoradd_raw_strided_kernel(
                        strided_kernel::Exec::serial(),
                        &mut zero_strides,
                        black_box(&mut dst),
                        black_box(&src),
                        &dims,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        false,
                        black_box(alpha),
                        black_box(beta),
                    )
                    .unwrap();
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_noncontiguous_combine, bench_fused_rank_gate);
criterion_main!(benches);
