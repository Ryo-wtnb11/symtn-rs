use core::ops::{Add, Mul};

use num_traits::{One, Zero};
use smallvec::SmallVec;

use crate::host_scalar_kernels::{
    combine_fused_layout, raw_rank_within_strided_limit_routed, raw_strided_action,
    validate_raw_strided_bounds,
};
use crate::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel_trusted, ConjugateValue, OperationError,
    RecouplingCoefficientAction, TransformScale,
};

mod adapter;
mod fused_layout;
#[cfg(test)]
pub(crate) mod observed;

pub use self::adapter::*;
pub use self::fused_layout::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensoradd_raw_strided_kernel;

    #[test]
    fn strided_host_adapter_add_strided_matches_axpby_semantics() {
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let mut dst = [10.0_f64, 20.0];
        let src = [2.0_f64, 3.0];

        adapter
            .add_strided_baked(
                strided_kernel::Exec::serial(),
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                3.0,
                None,
                None,
            )
            .unwrap();

        assert_eq!(dst, [34.0, 66.0]);
    }

    #[test]
    fn strided_host_adapter_recoupling_applies_u_transpose() {
        let mut adapter = StridedHostKernelAdapter::default();
        // Two source columns of two elements, two destination columns:
        // destination[:, d] = sum_s U[d, s] * source[:, s].
        let source = [1.0_f64, 2.0, 10.0, 20.0];
        let mut destination = [0.0_f64; 4];
        let coefficients = [1.0_f64, 0.5, -1.0, 2.0];

        adapter
            .recoupling_src_times_u_transpose(
                strided_kernel::Exec::serial(),
                &mut destination,
                &source,
                &coefficients,
                2,
                2,
                2,
            )
            .unwrap();

        assert_eq!(destination, [6.0, 12.0, 19.0, 38.0]);
    }

    #[test]
    fn recoupling_len_validation_rejects_mismatched_columns() {
        let err = validate_recoupling_lens(4, 3, 4, 2, 2, 2).unwrap_err();
        assert_eq!(
            err,
            OperationError::ElementCountMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    fn layout(shape: &[usize], dst: &[isize], src: &[isize]) -> FusedLayoutScratch {
        let mut layout = FusedLayoutScratch::default();
        normalize_fused_layout(shape, dst, src, &mut layout).unwrap();
        layout
    }

    /// Row-major strides for a shape (last axis fastest).
    fn row_major(shape: &[usize]) -> Vec<isize> {
        let mut strides = vec![1isize; shape.len()];
        for axis in (0..shape.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1] as isize;
        }
        strides
    }

    /// Naive odometer reference for `dst[i...] = src[i...]` over strided views.
    fn reference_copy(
        dst: &mut [f64],
        src: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
    ) {
        let total: usize = shape.iter().product();
        let mut index = vec![0usize; shape.len()];
        for _ in 0..total {
            let dst_pos: isize = index
                .iter()
                .zip(dst_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum();
            let src_pos: isize = index
                .iter()
                .zip(src_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum();
            dst[dst_pos as usize] = src[src_pos as usize];
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }

    #[test]
    fn normalized_layout_drops_extent_one_axes_and_fuses_contiguous_runs() {
        // Extent-1 axis dropped regardless of its (garbage) strides, then the
        // two remaining contiguous axes fuse into one 6-element run.
        let fused = layout(&[2, 1, 3], &[1, 999, 2], &[1, -7, 2]);
        assert_eq!(fused.dims(), &[6]);
        assert_eq!(fused.dst_strides(), &[1]);
        assert_eq!(fused.src_strides(), &[1]);
    }

    #[test]
    fn normalized_layout_orders_axes_without_fusing_mismatched_source() {
        // Axes arrive in descending destination-stride order and must be
        // reordered ascending; destination strides are contiguous (1 * 2 == 2)
        // but source strides are not (3 * 2 != 1), so the axes must NOT fuse.
        let unfused = layout(&[3, 2], &[2, 1], &[1, 3]);
        assert_eq!(unfused.dims(), &[2, 3]);
        assert_eq!(unfused.dst_strides(), &[1, 2]);
        assert_eq!(unfused.src_strides(), &[3, 1]);
    }

    #[test]
    fn normalized_layout_zero_extent_collapses_to_empty_marker() {
        let empty = layout(&[2, 0, 3], &[1, 2, 4], &[1, 2, 4]);
        assert_eq!(empty.dims(), &[0]);
    }

    #[test]
    fn normalized_layout_all_extent_one_collapses_to_scalar() {
        let scalar = layout(&[1, 1], &[5, 3], &[2, 8]);
        assert_eq!(scalar.dims(), &[1]);
        assert_eq!(scalar.dst_strides(), &[0]);
        assert_eq!(scalar.src_strides(), &[0]);
    }

    #[test]
    fn zero_extent_replaces_prior_normalization_state() {
        let mut scratch = layout(&[2, 3], &[1, 2], &[1, 2]);
        normalize_fused_layout(&[2, 0, 3], &[1, 2, 4], &[1, 2, 4], &mut scratch).unwrap();
        assert_eq!(scratch.dims(), &[0]);
        assert_eq!(scratch.dst_strides(), &[0]);
        assert_eq!(scratch.src_strides(), &[0]);
    }

    #[test]
    fn normalized_layout_reports_element_count_overflow() {
        // What: normalization and sealed baked construction report the same
        // typed overflow instead of panicking or admitting an invalid token.
        let mut scratch = FusedLayoutScratch::default();
        assert_eq!(
            normalize_fused_layout(&[usize::MAX, 2], &[1, 2], &[1, 2], &mut scratch,).unwrap_err(),
            OperationError::ElementCountOverflow
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[usize::MAX, 2], &[1, 2], &[1, 2],)
                .unwrap_err(),
            OperationError::ElementCountOverflow
        );
        assert_eq!(
            normalize_fused_layout(&[usize::MAX, 2], &[1], &[1, 2], &mut scratch).unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[usize::MAX, 2], &[1], &[1, 2],)
                .unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn fused_span_order_keeps_axis_one_fastest_with_negative_strides() {
        // What: the shared odometer preserves exact visit order and signed
        // offset arithmetic while using caller-owned runtime-rank scratch.
        let mut index = [0; 3];
        let mut visits = Vec::new();
        for_each_fused_span(
            &[3, 2, 2],
            &[1, 10, -100],
            &[2, -20, 200],
            300,
            60,
            &mut index,
            |dst, src, len, dst_stride, src_stride| {
                visits.push((dst, src, len, dst_stride, src_stride));
            },
        );
        assert_eq!(
            visits,
            [
                (300, 60, 3, 1, 2),
                (310, 40, 3, 1, 2),
                (200, 260, 3, 1, 2),
                (210, 240, 3, 1, 2),
            ]
        );
    }

    #[test]
    fn baked_fused_layout_accepts_runtime_rank_and_rejects_length_mismatches() {
        let empty_dims = [];
        let empty_strides = [];
        let dynamic_dims = [2usize; 9];
        let dynamic_strides = [1isize; 9];

        // What: only nonempty normalized slices with one stride per dimension
        // can become trusted replay tokens.
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(
                &empty_dims,
                &empty_strides,
                &empty_strides
            )
            .unwrap_err(),
            OperationError::RankMismatch {
                expected: 1,
                actual: 0,
            }
        );
        assert!(BakedFusedLayout::try_from_normalized_slices(
            &dynamic_dims,
            &dynamic_strides,
            &dynamic_strides
        )
        .is_ok());
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[0], &[1], &[0]).unwrap_err(),
            OperationError::InvalidArgument {
                message: "baked fused layout is not normalized",
            }
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&[1], &[1], &[0]).unwrap_err(),
            OperationError::InvalidArgument {
                message: "baked fused layout is not normalized",
            }
        );

        let dims = [2usize, 3];
        let short = [1isize];
        let complete = [1isize, 2];
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&dims, &short, &complete).unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            BakedFusedLayout::try_from_normalized_slices(&dims, &complete, &short).unwrap_err(),
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    fn assert_copy_matches_index_reference(
        mut exec: crate::host_pool::Exec<'_>,
        shape: &[usize],
        strides: &[isize],
    ) {
        let backing_len = shape
            .iter()
            .zip(strides)
            .map(|(&dim, &stride)| (dim - 1) * stride as usize)
            .sum::<usize>()
            + 1;
        let src = (0..backing_len)
            .map(|index| index as f64 + 0.25)
            .collect::<Vec<_>>();
        let scaled_src = src.iter().map(|value| 2.0 * value).collect::<Vec<_>>();
        let mut expected = vec![-1.0; backing_len];
        reference_copy(&mut expected, &scaled_src, shape, strides, strides);

        let normalized = layout(shape, strides, strides);
        let baked = BakedFusedLayout::try_from_normalized_slices(
            normalized.dims(),
            normalized.dst_strides(),
            normalized.src_strides(),
        )
        .unwrap();
        let mut adapter = StridedHostKernelAdapter::default();
        for baked in [None, Some(baked)] {
            let mut actual = vec![-1.0; backing_len];
            adapter
                .copy_scale_strided_baked(
                    exec.reborrow(),
                    &mut actual,
                    &src,
                    shape,
                    strides,
                    strides,
                    0,
                    0,
                    false,
                    2.0,
                    baked,
                    None,
                )
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn baked_and_eager_copies_match_the_index_reference_for_dynamic_rank() {
        // What: a rank-9 strided copy-scale writes alpha * src exactly at the
        // strided positions and nothing else, with and without a baked layout.
        assert_copy_matches_index_reference(
            strided_kernel::Exec::serial(),
            &[2, 2, 2, 2, 2, 2, 2, 2, 2],
            &[1, 3, 9, 27, 81, 243, 729, 2187, 6561],
        );
    }

    #[test]
    fn public_baked_methods_reject_raw_rank_mismatches_before_dispatch() {
        let mut exec = strided_kernel::Exec::serial();
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let mut dst = [0.0_f64; 2];
        let src = [1.0_f64; 2];
        let shape = [2usize];
        let strides = [1isize];
        let missing = [];
        let dst_error = OperationError::RankMismatch {
            expected: 1,
            actual: 0,
        };

        // What: all three public baked entry points validate both raw stride
        // ranks even without a baked token and on general-beta paths.
        assert_eq!(
            adapter
                .add_strided_baked(
                    exec.reborrow(),
                    &mut zero_strides,
                    &mut dst,
                    &src,
                    &shape,
                    &missing,
                    &strides,
                    0,
                    0,
                    false,
                    1.0,
                    2.0,
                    None,
                    None
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .add_strided_baked(
                    exec.reborrow(),
                    &mut zero_strides,
                    &mut dst,
                    &src,
                    &shape,
                    &strides,
                    &missing,
                    0,
                    0,
                    false,
                    1.0,
                    2.0,
                    None,
                    None
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .axpby_strided_baked(
                    exec.reborrow(),
                    &mut dst,
                    &src,
                    &shape,
                    &missing,
                    &strides,
                    0,
                    0,
                    1.0,
                    2.0,
                    None,
                    None
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .axpby_strided_baked(
                    exec.reborrow(),
                    &mut dst,
                    &src,
                    &shape,
                    &strides,
                    &missing,
                    0,
                    0,
                    1.0,
                    2.0,
                    None,
                    None
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .copy_scale_strided_baked(
                    exec.reborrow(),
                    &mut dst,
                    &src,
                    &shape,
                    &missing,
                    &strides,
                    0,
                    0,
                    false,
                    1.0,
                    None,
                    None
                )
                .unwrap_err(),
            dst_error
        );
        assert_eq!(
            adapter
                .copy_scale_strided_baked(
                    exec.reborrow(),
                    &mut dst,
                    &src,
                    &shape,
                    &strides,
                    &missing,
                    0,
                    0,
                    false,
                    1.0,
                    None,
                    None
                )
                .unwrap_err(),
            dst_error
        );
    }

    #[test]
    fn apply_fused_pair_copies_transposed_layout_exactly() {
        // dst[j * 2 + i] = 2 * src[i * 3 + j] over a logical [2, 3] iteration:
        // exact element placement through non-fusable permuted strides.
        let src = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = [0.0_f64; 6];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &[2, 3],
            &[1, 2],
            &[3, 1],
            0,
            0,
            |dst, value| *dst = value,
            |value| 2.0 * value,
        )
        .unwrap();
        assert_eq!(dst, [2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
    }

    #[test]
    fn apply_fused_pair_accumulates_with_offsets() {
        let src = [0.0_f64, 1.0, 2.0];
        let mut dst = [10.0_f64, 20.0, 30.0];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &[2],
            &[1],
            &[1],
            1,
            1,
            |dst, value| *dst += value,
            |value| 3.0 * value,
        )
        .unwrap();
        assert_eq!(dst, [10.0, 23.0, 36.0]);
    }

    #[test]
    fn fused_pair_dynamic_rank_matches_reference() {
        let shape = [2usize; 9];
        let src_strides = row_major(&shape);
        let dst_strides: Vec<isize> = src_strides.iter().rev().copied().collect();
        let src: Vec<f64> = (0..512).map(|value| value as f64 - 100.0).collect();

        let mut dst = vec![0.0_f64; 512];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &shape,
            &dst_strides,
            &src_strides,
            0,
            0,
            |dst, value| *dst = value,
            |value| value,
        )
        .unwrap();

        let mut dst_reference = vec![0.0_f64; 512];
        reference_copy(&mut dst_reference, &src, &shape, &dst_strides, &src_strides);
        assert_eq!(dst, dst_reference);
    }

    #[test]
    fn fused_pair_zero_extent_shape_is_a_noop() {
        let src = [1.0_f64; 4];
        let mut dst = [9.0_f64; 4];
        let mut scratch = StridedKernelScratch::default();
        fused_pair(
            &mut scratch,
            &mut dst,
            &src,
            &[2, 0],
            &[1, 2],
            &[1, 2],
            0,
            0,
            |dst, value| *dst = value,
            |value| value,
        )
        .unwrap();
        assert_eq!(dst, [9.0; 4]);
    }

    #[test]
    fn strided_host_adapter_fused_beta_branches_match_axpby_semantics() {
        let mut exec = strided_kernel::Exec::serial();
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let src = [2.0_f64, 3.0];

        // add_strided beta = 1: accumulate through the fused path.
        let mut dst = [10.0_f64, 20.0];
        adapter
            .add_strided_baked(
                exec.reborrow(),
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                1.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(dst, [14.0, 26.0]);

        // add_strided beta = 0: assign through the fused path.
        adapter
            .add_strided_baked(
                exec.reborrow(),
                &mut zero_strides,
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                2.0,
                0.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(dst, [4.0, 6.0]);

        // axpby_strided beta = 1 then beta = 0.
        let mut dst = [1.0_f64, 2.0];
        adapter
            .axpby_strided_baked(
                exec.reborrow(),
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                3.0,
                1.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(dst, [7.0, 11.0]);
        adapter
            .axpby_strided_baked(
                exec.reborrow(),
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                3.0,
                0.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(dst, [6.0, 9.0]);

        // copy_scale_strided always assigns.
        let mut dst = [99.0_f64, 99.0];
        adapter
            .copy_scale_strided_baked(
                exec.reborrow(),
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                -1.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(dst, [-2.0, -3.0]);

        // scale_strided scales in place.
        adapter
            .scale_strided(exec.reborrow(), &mut dst, &[2], &[1], 0, 2.0)
            .unwrap();
        assert_eq!(dst, [-4.0, -6.0]);
    }

    /// Rectangular windows of a larger parent block, as a degeneracy
    /// restriction or scatter reads them: `(shape, strides, offset)` per role.
    /// `(shape, destination strides, destination offset, source strides,
    /// source offset)`.
    type CheckedCase = (Vec<usize>, Vec<isize>, isize, Vec<isize>, isize);

    fn checked_cases() -> Vec<CheckedCase> {
        vec![
            // 3x4x2 window at (1, 2, 0) of a column-major 5x7x2 parent into a
            // compact destination at offset 3.
            (vec![3, 4, 2], vec![1, 3, 12], 3, vec![1, 5, 35], 1 + 2 * 5),
            // Axis order permuted (lazy-adjoint storage): destination axis 0
            // is the parent's slowest axis.
            (vec![2, 3, 4], vec![1, 2, 6], 0, vec![35, 1, 5], 2 + 5),
            // Extent-one axes, single-state restriction.
            (vec![1, 4, 1, 3], vec![1, 1, 4, 4], 1, vec![1, 5, 20, 20], 4),
            // Negative strides on the source (reversed read) and on one
            // destination axis (reversed write).
            (vec![3, 4], vec![1, 3], 0, vec![-1, -5], 40),
            (vec![3, 4], vec![-1, 3], 2, vec![5, 1], 6),
            // Rank zero and an empty block.
            (vec![], vec![], 2, vec![], 7),
            (vec![2, 0, 3], vec![1, 2, 2], 0, vec![1, 5, 5], 0),
        ]
    }

    fn assert_checked_matches_scalar_kernel<T>(
        mut exec: crate::host_pool::Exec<'_>,
        values: &[T],
        alpha_beta: &[(T, T)],
        bits: fn(T) -> u128,
    ) where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + Send
            + Sync,
    {
        let mut adapter = StridedHostKernelAdapter::default();
        for (shape, dst_strides, dst_offset, src_strides, src_offset) in checked_cases() {
            for &(alpha, beta) in alpha_beta {
                for conjugate in [false, true] {
                    let src: Vec<T> = (0..80).map(|i| values[i % values.len()]).collect();
                    let initial: Vec<T> = (0..30)
                        .map(|i| values[(i * 7 + 3) % values.len()])
                        .collect();
                    let mut expected = initial.clone();
                    tensoradd_raw_strided_kernel(
                        exec.reborrow(),
                        &mut Vec::new(),
                        &mut expected,
                        &src,
                        &shape,
                        &dst_strides,
                        &src_strides,
                        dst_offset,
                        src_offset,
                        conjugate,
                        alpha,
                        beta,
                    )
                    .unwrap();
                    let mut actual = initial.clone();
                    adapter
                        .tensoradd_strided_checked(
                            exec.reborrow(),
                            &mut actual,
                            &src,
                            &shape,
                            &dst_strides,
                            &src_strides,
                            dst_offset,
                            src_offset,
                            conjugate,
                            alpha,
                            beta,
                        )
                        .unwrap();
                    let expected: Vec<u128> = expected.into_iter().map(bits).collect();
                    let actual: Vec<u128> = actual.into_iter().map(bits).collect();
                    assert_eq!(actual, expected, "shape {shape:?} conj {conjugate}");

                    // A window one element past either end is rejected before
                    // any destination element is written.
                    if shape.iter().all(|&extent| extent > 0) {
                        let mut untouched = initial.clone();
                        let past_end = isize::try_from(src.len()).unwrap();
                        assert!(adapter
                            .tensoradd_strided_checked(
                                exec.reborrow(),
                                &mut untouched,
                                &src,
                                &shape,
                                &dst_strides,
                                &src_strides,
                                dst_offset,
                                past_end,
                                conjugate,
                                alpha,
                                beta,
                            )
                            .is_err());
                        assert!(adapter
                            .tensoradd_strided_checked(
                                exec.reborrow(),
                                &mut untouched,
                                &src,
                                &shape,
                                &dst_strides,
                                &src_strides,
                                -1,
                                src_offset,
                                conjugate,
                                alpha,
                                beta,
                            )
                            .is_err());
                        let untouched: Vec<u128> = untouched.into_iter().map(bits).collect();
                        let initial: Vec<u128> = initial.into_iter().map(bits).collect();
                        assert_eq!(untouched, initial);
                    }
                }
            }
        }
    }

    /// The checked block copy is bit-identical to the per-element scalar
    /// kernel it replaces for restriction and scatter, including signed
    /// zeros and non-finite values, where `1 * src` and `0 + src` would not be.
    #[test]
    fn checked_block_copy_is_bit_identical_to_the_scalar_kernel() {
        let mut exec = strided_kernel::Exec::serial();
        let reals = [
            1.5,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -2.25,
            7.0,
            f64::MIN_POSITIVE / 4.0,
        ];
        assert_checked_matches_scalar_kernel(
            exec.reborrow(),
            &reals,
            &[(1.0, 0.0), (1.0, 1.0), (-2.0, 0.0), (0.5, -3.0)],
            |value: f64| u128::from(value.to_bits()),
        );

        use num_complex::Complex64;
        let complexes = [
            Complex64::new(1.5, -0.0),
            Complex64::new(-0.0, -1.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
            Complex64::new(f64::NAN, 2.0),
            Complex64::new(-3.0, 4.5),
            Complex64::new(-0.0, 0.0),
        ];
        let one = Complex64::new(1.0, 0.0);
        assert_checked_matches_scalar_kernel(
            exec.reborrow(),
            &complexes,
            &[
                (one, Complex64::new(0.0, 0.0)),
                (one, one),
                (Complex64::new(0.5, -1.0), Complex64::new(0.0, 0.0)),
                (Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.5)),
            ],
            |value: Complex64| {
                (u128::from(value.re.to_bits()) << 64) | u128::from(value.im.to_bits())
            },
        );
    }

    /// What: every block-level scaling entry — the transform op used by
    /// tree-transform singles and by fusion pack/scatter, and the payload
    /// `alpha` copy/add/axpby ops — gives VectorInterface's
    /// `scale(x, 0) = zero(x) * 0` for a zero scale, so `±inf`/NaN sources
    /// contribute zero (#1438). Oracle (Julia, TensorOperations 5.6.2):
    /// `scale(complex(Inf, 1.0), 0.0)` and `scale(complex(Inf, NaN),
    /// complex(0.0, 0.0))` are `0.0 + 0.0im`, and `tensoradd!(C, A, p, false,
    /// 0, β)` leaves `scale(C, β)`: `C` for `β = 1`, `0` for `β = 0`.
    #[test]
    fn zero_scales_ignore_non_finite_sources() {
        let mut exec = strided_kernel::Exec::serial();
        use num_complex::Complex64;
        let src = [
            Complex64::new(f64::INFINITY, 1.0),
            Complex64::new(f64::NAN, f64::NEG_INFINITY),
        ];
        let initial = [Complex64::new(1.5, -2.0), Complex64::new(f64::NAN, 3.0)];
        let pairs = |values: &[Complex64]| {
            values
                .iter()
                .map(|v| {
                    let canonical = |x: f64| if x.is_nan() { f64::NAN } else { x };
                    (canonical(v.re).to_bits(), canonical(v.im).to_bits())
                })
                .collect::<Vec<_>>()
        };
        let zero = [Complex64::zero(); 2];
        let mut adapter = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        let scales: [TransformScale<Complex64, f64>; 3] = [
            TransformScale::Structural(0.0),
            TransformScale::new(Complex64::zero(), 2.0),
            TransformScale::new(Complex64::new(2.0, 1.0), 0.0),
        ];
        for scale in scales {
            for (beta, want) in [
                (None, zero),
                (Some(Complex64::zero()), zero),
                (Some(Complex64::one()), initial),
            ] {
                let mut dst = initial;
                adapter
                    .transform_strided_baked(
                        exec.reborrow(),
                        &mut zero_strides,
                        &mut dst,
                        &src,
                        &[2],
                        &[1],
                        &[1],
                        0,
                        0,
                        false,
                        scale,
                        beta,
                        None,
                        None,
                    )
                    .unwrap();
                assert_eq!(pairs(&dst), pairs(&want), "{scale:?}, beta {beta:?}");
            }
        }

        let alpha = Complex64::zero();
        let mut dst = initial;
        adapter
            .copy_scale_strided_baked(
                exec.reborrow(),
                &mut dst,
                &src,
                &[2],
                &[1],
                &[1],
                0,
                0,
                false,
                alpha,
                None,
                None,
            )
            .unwrap();
        assert_eq!(pairs(&dst), pairs(&zero));
        for beta in [Complex64::zero(), Complex64::one()] {
            let want = if beta.is_zero() { zero } else { initial };
            let mut dst = initial;
            adapter
                .add_strided_baked(
                    exec.reborrow(),
                    &mut zero_strides,
                    &mut dst,
                    &src,
                    &[2],
                    &[1],
                    &[1],
                    0,
                    0,
                    false,
                    alpha,
                    beta,
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(pairs(&dst), pairs(&want), "add, beta {beta}");
            let mut dst = initial;
            adapter
                .axpby_strided_baked(
                    exec.reborrow(),
                    &mut dst,
                    &src,
                    &[2],
                    &[1],
                    &[1],
                    0,
                    0,
                    alpha,
                    beta,
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(pairs(&dst), pairs(&want), "axpby, beta {beta}");
        }
    }
}
