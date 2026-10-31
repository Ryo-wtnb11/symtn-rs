//! Batched device Hermitian admission: one call over many regions
//! makes the same per-region decisions as the per-region rule, and its
//! downloads do not grow with the region count. Member stacks run the
//! same pipeline: every (member, region) decision equals the single call on
//! that member, and downloads do not grow with the member count.
//!
//! The decisions are asserted against hand-known truth values (exactly
//! Hermitian, asymmetric by a whole unit, zero, empty, non-finite, and
//! residuals inside and outside the default `eps(real(D))^(3/4)`), at both ends
//! of each lane's normal range. The deltas read the calling thread's
//! [`cuda_transfer_stats`] counters. Every decision, at the default and at
//! explicit tolerances, is also checked against an independent host
//! half-residual Frobenius oracle ([`oracle`]).
//!
//! Run with `cargo test -p symtn-dense --no-default-features --features \
//! cuda,cpu-faer --test integration cuda_hermitian_admission:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cuda_hermitian_regions, cuda_hermitian_regions_batched, cuda_transfer_stats, CudaDenseContext,
    CudaDenseStorage, CudaRealScalar, CudaScalar, CudaTransferStats, DenseError,
};

/// The one-region case of [`cuda_hermitian_regions`] at the default
/// tolerance; test support since no production caller decides a single
/// region.
fn cuda_is_hermitian_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<bool, symtn_dense::DenseError> {
    Ok(cuda_hermitian_regions::<D>(ctx, src, &[(offset, n)], default_tolerance::<D>())?[0])
}

/// symtn's default eigh admission tolerance, `eps(real(D))^(3/4)`.
fn default_tolerance<D: CudaScalar>() -> f64 {
    <D::Real as CudaRealScalar>::EPSILON.powf(0.75)
}

/// Explicit tolerances that move decisions of the fixtures: zero admits only
/// an exactly Hermitian block, `0.5` admits the whole-unit asymmetric ones.
const EXPLICIT_TOLERANCES: [f64; 3] = [0.0, 1e-3, 0.5];

/// Independent host oracle for the stored block (`n x n`, column-major):
/// `||(A - A^H)/2||_F <= tolerance * ||A||_F`, both norms as LAPACK `lassq`
/// `(scale, sum of squares)` pairs over real and imaginary components,
/// compared as a ratio of scales so no norm overflows; any non-finite
/// component rejects. Nothing here shares code with the device rule.
fn oracle(block: &[Complex64], n: usize, tolerance: f64) -> bool {
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
    let components = |values: Vec<Complex64>| values.into_iter().flat_map(|z| [z.re, z.im]);
    if components(block.to_vec()).any(|value| !value.is_finite()) {
        return false;
    }
    let residual = (0..n * n)
        .map(|index| block[index] - block[(index % n) * n + index / n].conj())
        .collect();
    let (residual_scale, residual_ssq) = lassq(components(residual));
    if residual_scale == 0.0 {
        return true;
    }
    let (input_scale, input_ssq) = lassq(components(block.to_vec()));
    0.5 * (residual_scale / input_scale) * residual_ssq.sqrt() <= tolerance * input_ssq.sqrt()
}

/// The oracle's decisions for `regions` of the stored (narrowed) `data`.
fn oracle_decisions<D: Payload>(
    data: &[D],
    regions: &[(usize, usize)],
    tolerance: f64,
) -> Vec<bool> {
    regions
        .iter()
        .map(|&(offset, n)| {
            let block: Vec<Complex64> = data[offset..offset + n * n]
                .iter()
                .map(|&value| value.widen())
                .collect();
            oracle(&block, n, tolerance)
        })
        .collect()
}

/// Transfer counters accumulated by `run`, alongside its result.
fn counted<T>(run: impl FnOnce() -> T) -> (T, CudaTransferStats) {
    let before = cuda_transfer_stats();
    let value = run();
    let after = cuda_transfer_stats();
    let stats = CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        device_allocs: after.device_allocs - before.device_allocs,
        ..CudaTransferStats::default()
    };
    (value, stats)
}

trait Payload: CudaScalar + Copy {
    const EPSILON: f64;
    /// The lane's smallest normal magnitude and `2^(MAX_EXP - 4)`.
    const TINY: f64;
    /// Not a `const`: `powi` is not a `const fn`.
    fn huge() -> f64;
    fn narrow(value: Complex64) -> Self;
    fn widen(self) -> Complex64;
}

impl Payload for f32 {
    const EPSILON: f64 = f32::EPSILON as f64;
    const TINY: f64 = f32::MIN_POSITIVE as f64;
    fn huge() -> f64 {
        f64::from(2f32.powi(124))
    }
    fn narrow(value: Complex64) -> Self {
        value.re as f32
    }
    fn widen(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl Payload for f64 {
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

impl Payload for Complex32 {
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

impl Payload for Complex64 {
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

/// Hermitian by construction: `(row, col)` and `(col, row)` share a
/// magnitude and negate the imaginary part, scaled by an exact power of two.
fn hermitian(n: usize, scale: f64) -> Vec<Complex64> {
    (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            let (low, high) = (row.min(col), row.max(col));
            let re = 1.0 + 0.5 * ((low % 7) as f64) + 0.25 * ((high % 5) as f64);
            let im = match row.cmp(&col) {
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Less => 0.5 + (low % 3) as f64,
                std::cmp::Ordering::Greater => -(0.5 + (low % 3) as f64),
            };
            Complex64::new(re * scale, im * scale)
        })
        .collect()
}

fn asymmetric(n: usize, scale: f64) -> Vec<Complex64> {
    let mut block = hermitian(n, scale);
    block[1] += Complex64::new(scale, 0.0);
    block
}

/// `[[1, delta], [0, 1]]`, relative residual `delta / 2`, with `delta` the
/// fraction `of_threshold` of `2 * eps(real(D))^(3/4)`: admitted below 1,
/// rejected above.
fn skewed<D: Payload>(of_threshold: f64) -> Vec<Complex64> {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let delta = of_threshold * 2.0 * D::EPSILON.powf(0.75);
    vec![one, zero, Complex64::new(delta, 0.0), one]
}

fn poisoned(bad: f64) -> Vec<Complex64> {
    let mut block = hermitian(3, 1.0);
    block[0] = Complex64::new(bad, 0.0);
    block
}

/// The mixed fixture: every early return and every stage of the rule.
fn cases<D: Payload>() -> Vec<(Vec<Complex64>, bool)> {
    vec![
        (hermitian(3, 1.0), true),
        (asymmetric(4, 1.0), false),
        (vec![Complex64::new(0.0, 0.0); 4], true),
        (Vec::new(), true),
        (hermitian(4, D::huge()), true),
        (asymmetric(4, D::huge()), false),
        (hermitian(4, D::TINY), true),
        (asymmetric(4, D::TINY), false),
        (poisoned(f64::NAN), false),
        (poisoned(f64::INFINITY), false),
        (skewed::<D>(0.5), true),
        (skewed::<D>(2.0), false),
        (hermitian(33, 1.0), true),
    ]
}

/// Packs `copies` repetitions of the fixture into one buffer, returning it,
/// the `(offset, n)` regions and the expected decisions.
fn packed<D: Payload>(copies: usize) -> (Vec<D>, Vec<(usize, usize)>, Vec<bool>) {
    let mut data = Vec::new();
    let mut regions = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..copies {
        for (block, hermitian) in cases::<D>() {
            let n = (block.len() as f64).sqrt() as usize;
            regions.push((data.len(), n));
            data.extend(block.into_iter().map(D::narrow));
            expected.push(hermitian);
        }
    }
    (data, regions, expected)
}

fn case<D: Payload>(ctx: &mut CudaDenseContext, name: &str) {
    let mut downloads = Vec::new();
    for copies in [1, 3] {
        let (data, regions, expected) = packed::<D>(copies);
        let src = CudaDenseStorage::upload::<D>(ctx, &data).expect("upload");

        let before = cuda_transfer_stats().d2h_calls;
        let batched = cuda_hermitian_regions::<D>(
            ctx,
            &src,
            &regions,
            <D::Real as symtn_dense::CudaRealScalar>::EPSILON.powf(0.75),
        )
        .expect("batched");
        downloads.push(cuda_transfer_stats().d2h_calls - before);

        assert_eq!(batched, expected, "{name}: batched decisions x{copies}");
        assert_eq!(
            oracle_decisions(&data, &regions, default_tolerance::<D>()),
            expected,
            "{name}: the oracle agrees with the hand-known truth"
        );
        for tolerance in EXPLICIT_TOLERANCES {
            let (decisions, stats) = counted(|| {
                cuda_hermitian_regions::<D>(ctx, &src, &regions, tolerance).expect("explicit")
            });
            assert_eq!(
                decisions,
                oracle_decisions(&data, &regions, tolerance),
                "{name}: tolerance {tolerance:e} x{copies}"
            );
            assert!(stats.d2h_calls <= 3, "{name}: {stats:?}");
            println!("{name} single x{copies} tol {tolerance:e}: {decisions:?} {stats:?}");
        }
        for (&(offset, n), &decision) in regions.iter().zip(&batched) {
            assert_eq!(
                cuda_is_hermitian_region::<D>(ctx, &src, offset, n).expect("single"),
                decision,
                "{name}: region ({offset}, {n}) alone"
            );
        }
    }
    assert!(
        downloads[0] <= 3,
        "{name}: at most one download per stage, got {downloads:?}"
    );
    assert_eq!(
        downloads[0], downloads[1],
        "{name}: downloads must not grow with the region count"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn batched_admission_matches_the_per_region_rule_with_bounded_downloads() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).expect("CUDA device 0");
    case::<f32>(&mut ctx, "f32");
    case::<f64>(&mut ctx, "f64");
    case::<Complex32>(&mut ctx, "Complex32");
    case::<Complex64>(&mut ctx, "Complex64");

    let empty = CudaDenseStorage::upload::<f64>(&ctx, &[0.0]).expect("upload");
    let before = cuda_transfer_stats().d2h_calls;
    assert_eq!(
        cuda_hermitian_regions::<f64>(&mut ctx, &empty, &[], f64::EPSILON.powf(0.75))
            .expect("no regions"),
        Vec::<bool>::new()
    );
    assert_eq!(
        cuda_transfer_stats().d2h_calls,
        before,
        "no regions, no downloads"
    );
}

/// One member of a real stack: every region keeps its size across members,
/// while its content, scale and truth value change with `member`, so one
/// region mixes accepted and rejected members of very different magnitudes.
fn stack_member<D: Payload>(member: usize) -> Vec<Vec<Complex64>> {
    let even = member.is_multiple_of(2);
    let scale = 2f64.powi(-7 * member as i32);
    let zero = vec![Complex64::new(0.0, 0.0); 4];
    vec![
        if even {
            hermitian(3, scale)
        } else {
            asymmetric(3, scale)
        },
        if even {
            asymmetric(4, scale)
        } else {
            hermitian(4, scale)
        },
        if even { zero.clone() } else { skewed::<D>(2.0) },
        Vec::new(),
        if even {
            hermitian(4, D::huge())
        } else {
            hermitian(4, D::TINY)
        },
        if even {
            asymmetric(4, D::TINY)
        } else {
            asymmetric(4, D::huge())
        },
        if even {
            poisoned(f64::NAN)
        } else {
            hermitian(3, scale)
        },
        if even {
            hermitian(3, 1.0)
        } else {
            poisoned(f64::INFINITY)
        },
        if even { skewed::<D>(0.5) } else { zero },
        if member == 3 {
            asymmetric(33, scale)
        } else {
            hermitian(33, scale)
        },
    ]
}

/// `members` stacked [`stack_member`]s `stride` elements apart (`pad`
/// elements past the packed member), with member 0's regions.
fn stacked<D: Payload>(members: usize, pad: usize) -> (Vec<D>, Vec<(usize, usize)>, usize) {
    let mut regions = Vec::new();
    let mut len = 0;
    for block in stack_member::<D>(0) {
        let n = (block.len() as f64).sqrt() as usize;
        regions.push((len, n));
        len += block.len();
    }
    let stride = len + pad;
    let mut data = vec![D::narrow(Complex64::new(0.0, 0.0)); members * stride];
    for member in 0..members {
        for (block, &(offset, _)) in stack_member::<D>(member).into_iter().zip(&regions) {
            for (index, value) in block.into_iter().enumerate() {
                data[member * stride + offset + index] = D::narrow(value);
            }
        }
    }
    (data, regions, stride)
}

fn stack_case<D: Payload>(ctx: &mut CudaDenseContext, name: &str) {
    let mut downloads = Vec::new();
    for (members, pad) in [(1, 0), (4, 0), (4, 5)] {
        let (data, regions, stride) = stacked::<D>(members, pad);
        let src = CudaDenseStorage::upload::<D>(ctx, &data).expect("upload");
        for tolerance in std::iter::once(default_tolerance::<D>()).chain(EXPLICIT_TOLERANCES) {
            let (decisions, stats) = counted(|| {
                cuda_hermitian_regions_batched::<D>(ctx, &src, &regions, members, stride, tolerance)
                    .expect("stack")
            });
            println!("{name} stack m{members} pad{pad} tol {tolerance:e}: {decisions:?} {stats:?}");
            assert!(stats.d2h_calls <= 3, "{name}: {stats:?}");
            if tolerance == default_tolerance::<D>() && pad == 0 {
                downloads.push(stats.d2h_calls);
            }
            assert_eq!(decisions.len(), members * regions.len());
            for (member, decided) in decisions.chunks(regions.len()).enumerate() {
                let shifted: Vec<_> = regions
                    .iter()
                    .map(|&(offset, n)| (offset + member * stride, n))
                    .collect();
                assert_eq!(
                    decided,
                    oracle_decisions(&data, &shifted, tolerance),
                    "{name}: member {member} of {members}, tolerance {tolerance:e}"
                );
                assert_eq!(
                    decided,
                    cuda_hermitian_regions::<D>(ctx, &src, &shifted, tolerance).expect("single"),
                    "{name}: member {member} alone, tolerance {tolerance:e}"
                );
            }
        }
    }
    assert_eq!(
        downloads[0], downloads[1],
        "{name}: downloads must not grow with the member count"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn member_stacks_decide_every_member_as_the_single_rule_with_bounded_downloads() {
    let mut engine =
        symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0)).unwrap();
    let mut ctx = CudaDenseContext::new(&mut engine).expect("CUDA device 0");
    stack_case::<f32>(&mut ctx, "f32");
    stack_case::<f64>(&mut ctx, "f64");

    // An empty stack decides nothing, before any device work, whatever its
    // regions or stride.
    let (data, regions, stride) = stacked::<f64>(1, 0);
    let src = CudaDenseStorage::upload::<f64>(&ctx, &data).expect("upload");
    for member_stride in [stride, 0, usize::MAX] {
        let (decisions, stats) = counted(|| {
            cuda_hermitian_regions_batched::<f64>(
                &mut ctx,
                &src,
                &regions,
                0,
                member_stride,
                default_tolerance::<f64>(),
            )
            .expect("empty stack")
        });
        assert_eq!(decisions, Vec::<bool>::new());
        assert_eq!(
            stats,
            CudaTransferStats::default(),
            "stride {member_stride}"
        );
    }

    // The explicit complex boundary, before any device work.
    let (data, regions, stride) = stacked::<Complex64>(2, 0);
    let src = CudaDenseStorage::upload::<Complex64>(&ctx, &data).expect("upload");
    let (result, stats) = counted(|| {
        cuda_hermitian_regions_batched::<Complex64>(
            &mut ctx,
            &src,
            &regions,
            2,
            stride,
            default_tolerance::<Complex64>(),
        )
    });
    assert!(
        matches!(result, Err(DenseError::Unsupported { .. })),
        "{result:?}"
    );
    assert_eq!(stats, CudaTransferStats::default());
    let (data, regions, stride) = stacked::<Complex32>(1, 0);
    let src = CudaDenseStorage::upload::<Complex32>(&ctx, &data).expect("upload");
    assert!(matches!(
        cuda_hermitian_regions_batched::<Complex32>(
            &mut ctx,
            &src,
            &regions,
            1,
            stride,
            default_tolerance::<Complex32>(),
        ),
        Err(DenseError::Unsupported { .. })
    ));
}
