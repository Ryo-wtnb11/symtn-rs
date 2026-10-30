//! The device Hermitian admission rule, as pure host arithmetic.
//!
//! Compiled (and tested) without the `cuda` feature, like `cuda_region`, so
//! CI — which merely `cargo check`s that feature — still executes the rule
//! that decides whether a device block is offered to `eigh`.

/// Decides `0.5 * ||A - A†||_F <= tolerance * ||A||_F` from the *scaled*
/// device reductions.
///
/// Both norms arrive factored as `scale * sqrt(sum_of_squares)` of a
/// copy normalized by an exact power of two near its maximum
/// ([`power_of_two_normalizer`]), which is what keeps the relative decision
/// stable when either norm would overflow or underflow if formed directly —
/// the residual is formed from the normalized input, so the input normalizer
/// cancels out of both sides and only the residual's `residual_scale`
/// (the reciprocal of its own normalizer) remains. A non-finite or negative part
/// rejects: it can only come from a non-finite block.
///
/// `relative_tolerance` is the caller's resolved eigh admission tolerance for
/// the payload under test (symtn's `HermitianTol`, whose default is
/// `eps(real(D))^(3/4)`); it is a parameter so this rule is one authority for
/// all four payload dtypes and every tolerance.
pub(crate) fn scaled_hermitian_residual_accepts(
    input_ss: f64,
    residual_scale: f64,
    residual_ss: f64,
    relative_tolerance: f64,
) -> bool {
    input_ss.is_finite()
        && input_ss >= 0.0
        && residual_scale.is_finite()
        && residual_scale >= 0.0
        && residual_ss.is_finite()
        && residual_ss >= 0.0
        && 0.5 * residual_scale * residual_ss.sqrt() <= relative_tolerance * input_ss.sqrt()
}

/// What one (member, region) learned from a downloaded stage scalar: its
/// decision, or the exact power of two that normalizes its next reduction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum StageOutcome {
    Decided(bool),
    Normalize(f64),
}

/// Stage 1 of the device rule, from the input's maximum magnitude.
///
/// A non-finite maximum rejects *before* the zero fast path: pinned
/// Tenferro's CUDA `reduce_max` propagates NaN, so an otherwise-zero matrix
/// containing NaN must not be admitted as zero. A zero matrix is Hermitian.
pub(crate) fn input_stage(input_scale: f64, max_exp: i32) -> StageOutcome {
    if !input_scale.is_finite() {
        StageOutcome::Decided(false)
    } else if input_scale == 0.0 {
        StageOutcome::Decided(true)
    } else {
        StageOutcome::Normalize(power_of_two_normalizer(input_scale, max_exp))
    }
}

/// Stage 2 of the device rule, from the normalized input's sum of squares
/// and the normalized residual's maximum magnitude. A zero residual admits
/// any finite input; a non-finite one rejects.
pub(crate) fn residual_stage(input_ss: f64, residual_scale: f64, max_exp: i32) -> StageOutcome {
    if !residual_scale.is_finite() {
        StageOutcome::Decided(false)
    } else if residual_scale == 0.0 {
        StageOutcome::Decided(input_ss.is_finite() && input_ss >= 0.0)
    } else {
        StageOutcome::Normalize(power_of_two_normalizer(residual_scale, max_exp))
    }
}

/// The exact power of two `2^-k`, `k = floor(log2(scale))`, that brings a
/// finite positive `scale` of a real lane with `max_exp` (its `MAX_EXP`)
/// into `[1, 2)`.
///
/// Why not divide by `scale` itself: Tenferro's complex division by
/// `Complex(s, 0)` forms `s * s`, which leaves the lane once `|s|` passes the
/// square root of its range (tenferro-rs#1922), while a product with
/// `(2^-k, 0)` scales each component. Multiplying by a power of two is also
/// exact wherever the product stays normal.
///
/// `k` is read from the exponent bits, never from a float `log2`. It is
/// clamped to `[-(MAX_EXP - 2), MAX_EXP - 2]` so both `2^-k` (the operand
/// the payload is multiplied by) and its reciprocal `2^k` are *normal*
/// numbers of the lane, which flush-to-zero cannot touch. The lower bound
/// acts only on a subnormal `scale`, whose clamped product still has a normal
/// maximum (`>= 2^-23` in `f32`, `>= 2^-52` in `f64`); the upper bound acts
/// on the top octave, whose clamped product is merely in `[2, 4)`.
pub(crate) fn power_of_two_normalizer(scale: f64, max_exp: i32) -> f64 {
    const MANTISSA_BITS: u32 = 52;
    const BIAS: i32 = 1023;
    let bits = scale.to_bits();
    let biased = ((bits >> MANTISSA_BITS) & 0x7ff) as i32;
    let floor_log2 = if biased == 0 {
        // An `f64` subnormal is `mantissa * 2^-1074`.
        let mantissa = bits & ((1 << MANTISSA_BITS) - 1);
        63 - mantissa.leading_zeros() as i32 - 1074
    } else {
        biased - BIAS
    };
    let k = floor_log2.clamp(-(max_exp - 2), max_exp - 2);
    f64::from_bits(((BIAS - k) as u64) << MANTISSA_BITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    // symtn's default eigh admission tolerance, `eps^(3/4)`.
    fn tolerance(epsilon: f64) -> f64 {
        epsilon.powf(0.75)
    }

    /// The single-precision tolerance hazard, as host arithmetic: a residual of a few `f32` roundings
    /// relative to the input norm is accepted by the single-precision
    /// tolerance and rejected by the double-precision one. Both are expressed
    /// in epsilons of their own lane, so nothing here is a platform-dependent
    /// absolute constant.
    #[test]
    fn a_single_precision_rounding_residual_needs_the_single_precision_lane() {
        let input_ss: f64 = 1.0;
        // Half the anti-Hermitian residual sits at 8 f32 epsilons of the input
        // norm: inside `eps(f32)^(3/4)`, and ~5e5 times outside
        // `eps(f64)^(3/4)`.
        let residual_scale = 2.0 * 8.0 * f64::from(f32::EPSILON);
        assert!(scaled_hermitian_residual_accepts(
            input_ss,
            residual_scale,
            1.0,
            tolerance(f64::from(f32::EPSILON)),
        ));
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            residual_scale,
            1.0,
            tolerance(f64::EPSILON),
        ));

        // A clearly non-Hermitian block is rejected in either lane: the same
        // residual grown past the single-precision threshold.
        let asymmetric = 2.0 * 256.0 * f64::from(f32::EPSILON);
        assert!(!scaled_hermitian_residual_accepts(
            input_ss,
            asymmetric,
            1.0,
            tolerance(f64::from(f32::EPSILON)),
        ));
    }

    /// The threshold on the half-residual is exactly
    /// `tolerance * ||A||_F` at every input scale; a non-finite residual or
    /// scale is rejected.
    #[test]
    fn the_half_residual_threshold_is_tolerance_times_the_input_norm() {
        let tolerance = tolerance(f64::EPSILON);
        for &input_ss in &[1.0_f64, 2.0, 1e-30, 1e30] {
            let threshold = 2.0 * tolerance * input_ss.sqrt();
            assert!(scaled_hermitian_residual_accepts(
                input_ss,
                threshold * 0.99,
                1.0,
                tolerance
            ));
            assert!(!scaled_hermitian_residual_accepts(
                input_ss,
                threshold * 1.01,
                1.0,
                tolerance
            ));
        }
        assert!(!scaled_hermitian_residual_accepts(
            2.0,
            f64::NAN,
            1.0,
            tolerance
        ));
        assert!(!scaled_hermitian_residual_accepts(
            2.0,
            1.0,
            f64::INFINITY,
            tolerance,
        ));
    }

    /// Every finite positive scale of either lane lands in `[1, 2)` except
    /// where the clamp is documented to act, and the multiplier and its
    /// reciprocal are always normal numbers of the lane, so narrowing and
    /// flush-to-zero leave them alone.
    #[test]
    fn the_power_of_two_normalizer_is_exact_across_each_lane() {
        fn check(scale: f64, max_exp: i32, min_positive: f64) {
            // `1 / MIN_POSITIVE = 2^(MAX_EXP - 2)`, the largest power of two
            // whose reciprocal is still normal.
            let max_power = min_positive.recip();
            let normalizer = power_of_two_normalizer(scale, max_exp);
            for value in [normalizer, normalizer.recip()] {
                assert!(
                    (min_positive..=max_power).contains(&value),
                    "{scale:e}: {value:e} is not a normal lane power of two"
                );
                assert_eq!(value.to_bits() & ((1 << 52) - 1), 0, "{scale:e}");
            }
            let scaled = scale * normalizer;
            if scale < min_positive {
                assert_eq!(normalizer, max_power, "{scale:e}");
            } else if scale >= 2.0 * max_power {
                assert!((2.0..4.0).contains(&scaled), "{scale:e}: {scaled:e}");
            } else {
                assert!((1.0..2.0).contains(&scaled), "{scale:e}: {scaled:e}");
            }
        }
        let f32_lane = |scale: f64| check(scale, f32::MAX_EXP, f64::from(f32::MIN_POSITIVE));
        for scale in [
            f32::from_bits(1),
            f32::from_bits(0x0000_1234),
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE * 1.5,
            0.75,
            1.0,
            1.999_999_9,
            2.0f32.powi(124) * 5.0,
            2.0f32.powi(127),
            f32::MAX,
        ] {
            f32_lane(f64::from(scale));
        }
        let f64_lane = |scale: f64| check(scale, f64::MAX_EXP, f64::MIN_POSITIVE);
        for scale in [
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            3.0e-300,
            1.0,
            1.0 - f64::EPSILON / 2.0,
            2f64.powi(1020) * 5.0,
            2f64.powi(1023),
            f64::MAX,
        ] {
            f64_lane(scale);
        }
        // Subnormal lane scales are clamped yet still end in the lane's normal
        // range after the multiply.
        assert!(
            f64::from(f32::from_bits(1))
                * power_of_two_normalizer(f64::from(f32::from_bits(1)), f32::MAX_EXP)
                >= 2f64.powi(-23)
        );
        assert!(
            f64::from_bits(1) * power_of_two_normalizer(f64::from_bits(1), f64::MAX_EXP)
                >= 2f64.powi(-52)
        );
    }

    /// The stage boundaries, including NaN before the zero fast path.
    #[test]
    fn the_stages_decide_nonfinite_and_zero_before_normalizing() {
        let max_exp = f64::MAX_EXP;
        for bad in [f64::NAN, f64::INFINITY] {
            assert_eq!(input_stage(bad, max_exp), StageOutcome::Decided(false));
            assert_eq!(
                residual_stage(1.0, bad, max_exp),
                StageOutcome::Decided(false)
            );
        }
        assert_eq!(input_stage(0.0, max_exp), StageOutcome::Decided(true));
        assert_eq!(
            residual_stage(4.0, 0.0, max_exp),
            StageOutcome::Decided(true)
        );
        assert_eq!(
            residual_stage(f64::NAN, 0.0, max_exp),
            StageOutcome::Decided(false)
        );
        assert_eq!(input_stage(3.0, max_exp), StageOutcome::Normalize(0.5));
        assert_eq!(
            residual_stage(1.0, 0.75, max_exp),
            StageOutcome::Normalize(2.0)
        );
    }

    /// Independent oracle: `||(A - A^T)/2||_F <= tol ||A||_F` with both norms
    /// kept as the LAPACK `lassq` pair `(scale, sum of squares)` (no
    /// power-of-two normalization), compared as a ratio of scales so a norm
    /// past the lane's range still decides; any non-finite entry rejects.
    fn half_residual_oracle(a: &[f64], n: usize, tolerance: f64) -> bool {
        fn lassq(values: impl Iterator<Item = f64>) -> (f64, f64) {
            let (mut scale, mut ssq) = (0.0_f64, 1.0_f64);
            for value in values.map(f64::abs).filter(|&value| value != 0.0) {
                if scale < value {
                    ssq = 1.0 + ssq * (scale / value).powi(2);
                    scale = value;
                } else {
                    ssq += (value / scale).powi(2);
                }
            }
            (scale, ssq)
        }
        if a.iter().any(|value| !value.is_finite()) {
            return false;
        }
        let (residual_scale, residual_ssq) =
            lassq((0..n * n).map(|index| a[index] - a[(index % n) * n + index / n]));
        if residual_scale == 0.0 {
            return true;
        }
        let (input_scale, input_ssq) = lassq(a.iter().copied());
        0.5 * (residual_scale / input_scale) * residual_ssq.sqrt() <= tolerance * input_ssq.sqrt()
    }

    /// The device pipeline's three stages as host `f64` arithmetic: a
    /// NaN-propagating maximum, the power-of-two normalizers the stages
    /// choose, and the scaled sums of squares.
    fn staged_rule(a: &[f64], n: usize, tolerance: f64) -> bool {
        let max_exp = f64::MAX_EXP;
        let max_abs = |values: &[f64]| {
            values.iter().fold(0.0_f64, |max, value| {
                if max.is_nan() || value.is_nan() {
                    f64::NAN
                } else {
                    max.max(value.abs())
                }
            })
        };
        let normalizer = match input_stage(max_abs(a), max_exp) {
            StageOutcome::Decided(decision) => return decision,
            StageOutcome::Normalize(normalizer) => normalizer,
        };
        let scaled: Vec<f64> = a.iter().map(|value| value * normalizer).collect();
        let input_ss: f64 = scaled.iter().map(|value| value * value).sum();
        let residual: Vec<f64> = (0..n * n)
            .map(|index| scaled[index] - scaled[(index % n) * n + index / n])
            .collect();
        let residual_normalizer = match residual_stage(input_ss, max_abs(&residual), max_exp) {
            StageOutcome::Decided(decision) => return decision,
            StageOutcome::Normalize(normalizer) => normalizer,
        };
        let residual_ss: f64 = residual
            .iter()
            .map(|value| (value * residual_normalizer).powi(2))
            .sum();
        scaled_hermitian_residual_accepts(
            input_ss,
            residual_normalizer.recip(),
            residual_ss,
            tolerance,
        )
    }

    /// The shared stages compose to the half-residual rule at every scale
    /// and tolerance, decided against the independent oracle.
    #[test]
    fn the_staged_rule_matches_an_independent_half_residual_oracle() {
        let symmetric = |n: usize, scale: f64| -> Vec<f64> {
            (0..n * n)
                .map(|index| {
                    let (row, col) = (index % n, index / n);
                    scale
                        * (1.0 + 0.5 * (row.min(col) % 7) as f64 + 0.25 * (row.max(col) % 5) as f64)
                })
                .collect()
        };
        let asymmetric = |n: usize, scale: f64| {
            let mut block = symmetric(n, scale);
            block[1] += scale;
            block
        };
        let default = f64::EPSILON.powf(0.75);
        let skewed = |of: f64| vec![1.0, 0.0, of * 2.0 * default, 1.0];
        let poisoned = |bad: f64| {
            let mut block = vec![0.0; 9];
            block[4] = bad;
            block
        };
        let mut blocks = vec![
            vec![0.0; 4],
            skewed(0.5),
            skewed(2.0),
            poisoned(f64::NAN),
            poisoned(f64::INFINITY),
            poisoned(f64::NEG_INFINITY),
        ];
        for scale in [
            1.0,
            f64::MIN_POSITIVE,
            2f64.powi(-500),
            2f64.powi(500),
            2f64.powi(1020),
        ] {
            for n in [1, 3, 8] {
                blocks.push(symmetric(n, scale));
                if n > 1 {
                    blocks.push(asymmetric(n, scale));
                }
            }
        }
        let mut decisions = [0usize; 2];
        for tolerance in [0.0, default, 1e-3, 0.5] {
            for block in &blocks {
                let n = (block.len() as f64).sqrt() as usize;
                let expected = half_residual_oracle(block, n, tolerance);
                assert_eq!(
                    staged_rule(block, n, tolerance),
                    expected,
                    "n {n}, tolerance {tolerance:e}, block {block:?}"
                );
                decisions[usize::from(expected)] += 1;
            }
        }
        assert!(decisions[0] > 0 && decisions[1] > 0, "{decisions:?}");
    }
}
