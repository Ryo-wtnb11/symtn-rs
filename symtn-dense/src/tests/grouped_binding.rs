//! `DefaultDenseExecutor::matmul_batch_axpby_into_bound`: the caller-owned
//! grouped binding changes how often a native grouped plan is prepared,
//! never a value or an error. Values are checked against an independent
//! column-major scalar triple loop in `Complex64`.

use super::*;

/// The four batch dtypes, seen through `Complex64` for the oracle.
trait Scalar: Copy + 'static {
    const TOL: f64;
    fn make(value: Complex64) -> Self;
    fn wide(self) -> Complex64;
    fn dense(value: Complex64) -> DenseScalar;
    fn write<'a>(data: &'a mut [Self], shape: &'a [usize], offset: usize) -> DenseWrite<'a>;
    fn read<'a>(data: &'a [Self], shape: &'a [usize], offset: usize) -> DenseRead<'a>;
}

macro_rules! scalar {
    ($ty:ty, $variant:ident, $tol:expr, |$v:ident| $make:expr, |$s:ident| $wide:expr, |$c:ident| $dense:expr) => {
        impl Scalar for $ty {
            const TOL: f64 = $tol;
            fn make($v: Complex64) -> Self {
                $make
            }
            fn wide(self) -> Complex64 {
                let $s = self;
                $wide
            }
            fn dense($c: Complex64) -> DenseScalar {
                DenseScalar::$variant($dense)
            }
            fn write<'a>(
                data: &'a mut [Self],
                shape: &'a [usize],
                offset: usize,
            ) -> DenseWrite<'a> {
                DenseWrite::$variant(DenseViewMut::new(data, shape, &[1], offset).unwrap())
            }
            fn read<'a>(data: &'a [Self], shape: &'a [usize], offset: usize) -> DenseRead<'a> {
                DenseRead::$variant(DenseView::new(data, shape, &[1], offset).unwrap())
            }
        }
    };
}
scalar!(
    f32,
    F32,
    1e-4,
    |v| v.re as f32,
    |s| Complex64::new(s as f64, 0.0),
    |c| c.re as f32
);
scalar!(f64, F64, 1e-12, |v| v.re, |s| Complex64::new(s, 0.0), |c| c
    .re);
scalar!(
    Complex32,
    C32,
    1e-4,
    |v| Complex32::new(v.re as f32, v.im as f32),
    |s| Complex64::new(s.re as f64, s.im as f64),
    |c| Complex32::new(c.re as f32, c.im as f32)
);
scalar!(Complex64, C64, 1e-12, |v| v, |s| s, |c| c);

const fn job(
    dst: usize,
    lhs: usize,
    rhs: usize,
    m: usize,
    k: usize,
    n: usize,
) -> DenseGemmBatchJob {
    DenseGemmBatchJob {
        dst_offset: dst,
        lhs_offset: lhs,
        rhs_offset: rhs,
        rows: m,
        contracted: k,
        cols: n,
    }
}

/// Three shapes, output ranges out of job order and with gaps (`0..2`,
/// `8..9`, `17..20`, `32..34` are never written).
const JOBS: [DenseGemmBatchJob; 3] = [
    job(20, 0, 0, 3, 2, 4),
    job(2, 6, 8, 2, 5, 3),
    job(9, 16, 23, 4, 1, 2),
];
const LENS: [usize; 3] = [34, 20, 25];
const RUNS: [usize; 3] = [1, 1, 1];
const SHAPES: [[usize; 1]; 3] = [[LENS[0]], [LENS[1]], [LENS[2]]];

fn values<T: Scalar>(len: usize, salt: f64) -> Vec<T> {
    (0..len)
        .map(|i| {
            let x = i as f64 + salt;
            T::make(Complex64::new((0.37 * x).sin(), (0.11 * x).cos()))
        })
        .collect()
}

/// `alpha * lhs * rhs + beta * dst` per job, column-major, by scalar loops;
/// `beta == 0` overwrites (a NaN destination does not propagate) and every
/// element no job writes keeps its value.
fn oracle<T: Scalar>(
    jobs: &[DenseGemmBatchJob],
    out: &[T],
    lhs: &[T],
    rhs: &[T],
    alpha: Complex64,
    beta: Complex64,
) -> Vec<Complex64> {
    let mut want: Vec<Complex64> = out.iter().map(|value| value.wide()).collect();
    for job in jobs {
        for j in 0..job.cols {
            for i in 0..job.rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for p in 0..job.contracted {
                    sum += lhs[job.lhs_offset + i + job.rows * p].wide()
                        * rhs[job.rhs_offset + p + job.contracted * j].wide();
                }
                let at = job.dst_offset + i + job.rows * j;
                want[at] = if beta == Complex64::new(0.0, 0.0) {
                    alpha * sum
                } else {
                    alpha * sum + beta * out[at].wide()
                };
            }
        }
    }
    want
}

fn builds(f: impl FnOnce()) -> usize {
    let recording = crate::probe::record();
    f();
    recording.get(crate::probe::Event::GroupedPlanBuild)
}

/// The grouped plans one call prepares on the native route; the BLAS route
/// never prepares one.
fn native(count: usize) -> usize {
    if cfg!(feature = "cpu-blas-core") {
        0
    } else {
        count
    }
}

/// One bound call with views starting `base` elements into fresh buffers.
#[allow(clippy::too_many_arguments)]
fn bound_call<T: Scalar>(
    executor: &mut DefaultDenseExecutor,
    binding: &mut DenseGemmBatchBinding,
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
    base: [usize; 3],
    salt: f64,
    init: Option<Complex64>,
    alpha: Complex64,
    beta: Complex64,
) -> (Result<(), DenseError>, Vec<T>, Vec<Complex64>) {
    let [out_base, lhs_base, rhs_base] = base;
    // The coefficients as the dtype represents them.
    let (alpha, beta) = (T::make(alpha).wide(), T::make(beta).wide());
    let mut out = values::<T>(out_base + LENS[0], salt);
    if let Some(fill) = init {
        out.fill(T::make(fill));
    }
    let lhs = values::<T>(lhs_base + LENS[1], salt + 100.0);
    let rhs = values::<T>(rhs_base + LENS[2], salt + 200.0);
    let fits = jobs.iter().all(|job| {
        let end = |offset: usize, a: usize, b: usize| {
            a.checked_mul(b).and_then(|n| n.checked_add(offset))
        };
        end(job.dst_offset, job.rows, job.cols).is_some_and(|e| e <= LENS[0])
            && end(job.lhs_offset, job.rows, job.contracted).is_some_and(|e| e <= LENS[1])
            && end(job.rhs_offset, job.contracted, job.cols).is_some_and(|e| e <= LENS[2])
    });
    // A descriptor outside the buffers is refused; there is nothing to predict.
    let want = if !fits {
        Vec::new()
    } else {
        let mut want: Vec<Complex64> = out[..out_base].iter().map(|v| v.wide()).collect();
        want.extend(oracle(
            jobs,
            &out[out_base..],
            &lhs[lhs_base..],
            &rhs[rhs_base..],
            alpha,
            beta,
        ));
        want
    };
    let result = executor.matmul_batch_axpby_into_bound(
        T::write(&mut out, &SHAPES[0], out_base),
        T::read(&lhs, &SHAPES[1], lhs_base),
        T::read(&rhs, &SHAPES[2], rhs_base),
        jobs,
        runs,
        T::dense(alpha),
        T::dense(beta),
        binding,
    );
    (result, out, want)
}

fn assert_matches<T: Scalar>(found: &[T], want: &[Complex64], what: &str) {
    assert_eq!(found.len(), want.len(), "{what}");
    for (index, (found, want)) in found.iter().zip(want).enumerate() {
        let found = found.wide();
        if want.re.is_nan() || want.im.is_nan() {
            assert!(
                found.re.is_nan() || found.im.is_nan(),
                "{what} [{index}]: {found} kept"
            );
            continue;
        }
        let tol = T::TOL * (1.0 + want.norm());
        assert!(
            (found - want).norm() <= tol,
            "{what} [{index}]: {found} vs {want}"
        );
    }
}

fn charge(jobs: &[DenseGemmBatchJob]) -> usize {
    std::mem::size_of_val(jobs)
        + jobs
            .iter()
            .map(|job| {
                let [m, n, k] = [job.rows, job.cols, job.contracted].map(|d| d.max(1));
                16_384 + 8 * (3 * m + 3 * n + 2 * k + 4)
            })
            .sum::<usize>()
}

fn reuse_across_values_offsets_and_coefficients<T: Scalar>(name: &str) {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let mut binding = DenseGemmBatchBinding::default();
    let calls = [
        // NaN destination, overwrite.
        (
            [0, 0, 0],
            1.0,
            Some(Complex64::new(f64::NAN, f64::NAN)),
            Complex64::new(1.5, -0.5),
            Complex64::new(0.0, 0.0),
        ),
        // Other view offsets, values and coefficients; accumulate.
        (
            [3, 1, 5],
            2.0,
            None,
            Complex64::new(-2.0, 0.25),
            Complex64::new(0.75, 0.5),
        ),
        (
            [0, 4, 2],
            3.0,
            None,
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ),
    ];
    for (call, (base, salt, init, alpha, beta)) in calls.into_iter().enumerate() {
        let mut outcome = None;
        let count = builds(|| {
            outcome = Some(bound_call::<T>(
                &mut executor,
                &mut binding,
                &JOBS,
                &RUNS,
                base,
                salt,
                init,
                alpha,
                beta,
            ));
        });
        let (result, found, want) = outcome.unwrap();
        result.unwrap();
        assert_matches(&found, &want, &format!("{name} call {call}"));
        assert_eq!(count, native(usize::from(call == 0)), "{name} call {call}");
        assert_eq!(
            binding.retained_bytes(),
            native(1) * charge(&JOBS),
            "{name} call {call}"
        );
    }
}

#[test]
fn bound_grouped_batch_prepares_once_and_matches_the_scalar_oracle() {
    // What: one binding over three calls with new buffers, view offsets,
    // values, alpha and beta prepares one grouped plan (at the first call),
    // and every call equals the scalar-loop oracle, including the NaN
    // overwrite at beta = 0, for all four dtypes.
    reuse_across_values_offsets_and_coefficients::<f32>("f32");
    reuse_across_values_offsets_and_coefficients::<f64>("f64");
    reuse_across_values_offsets_and_coefficients::<Complex32>("c32");
    reuse_across_values_offsets_and_coefficients::<Complex64>("c64");
}

#[test]
fn bound_batches_without_a_grouped_route_release_the_binding() {
    // What: empty, one-job and affine-run batches keep their routes (the
    // strided entry and its engine cache) and prepare no grouped plan. Each
    // follows a grouped call that populated the binding, and on success it
    // releases that binding: the charge returns to the empty binding's.
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let mut binding = DenseGemmBatchBinding::default();
    let one = [JOBS[0]];
    let affine = [job(0, 0, 0, 2, 3, 2), job(4, 6, 6, 2, 3, 2)];
    let (alpha, beta) = (Complex64::new(2.0, 0.0), Complex64::new(-0.5, 0.0));
    for (what, jobs, runs) in [
        ("affine run", &affine[..], &[2][..]),
        ("one job", &one[..], &[1][..]),
        ("empty", &[][..], &[][..]),
    ] {
        let (result, _, _) = bound_call::<f64>(
            &mut executor,
            &mut binding,
            &JOBS,
            &RUNS,
            [0, 0, 0],
            3.0,
            None,
            alpha,
            beta,
        );
        result.unwrap();
        assert_eq!(
            binding.retained_bytes(),
            native(1) * charge(&JOBS),
            "{what}: populated"
        );
        let mut outcome = None;
        let count = builds(|| {
            outcome = Some(bound_call::<f64>(
                &mut executor,
                &mut binding,
                jobs,
                runs,
                [1, 0, 2],
                4.0,
                None,
                alpha,
                beta,
            ));
        });
        let (result, found, want) = outcome.unwrap();
        result.unwrap();
        assert_matches(&found, &want, what);
        assert_eq!(count, 0, "{what}");
        assert_eq!(binding.retained_bytes(), 0, "{what}: released");
    }
}

#[test]
fn bound_grouped_refusals_keep_the_binding_and_the_destination() {
    // What: a short buffer, an overflowing descriptor and an overlapping
    // descriptor are refused before any write and prepare nothing that
    // replaces the binding; the next valid call reuses it. A different valid
    // descriptor or dtype replaces it, and the original jobs then replace it
    // back: matching is by the complete ordered jobs and dtype.
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let mut binding = DenseGemmBatchBinding::default();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let valid = |executor: &mut DefaultDenseExecutor, binding: &mut DenseGemmBatchBinding| {
        let mut outcome = None;
        let count = builds(|| {
            outcome = Some(bound_call::<f64>(
                executor,
                binding,
                &JOBS,
                &RUNS,
                [0, 0, 0],
                5.0,
                None,
                one,
                zero,
            ));
        });
        let (result, found, want) = outcome.unwrap();
        result.unwrap();
        assert_matches(&found, &want, "valid");
        count
    };
    assert_eq!(valid(&mut executor, &mut binding), native(1));
    let held = binding.retained_bytes();

    // A short output view: the jobs reach element 32, the view holds 31.
    let lhs = values::<f64>(LENS[1], 1.0);
    let rhs = values::<f64>(LENS[2], 2.0);
    let mut out = vec![f64::NAN; 31];
    let before = out.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let mut outcome = None;
    let count = builds(|| {
        outcome = Some(executor.matmul_batch_axpby_into_bound(
            DenseWrite::F64(DenseViewMut::new(&mut out, &[31], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &[LENS[1]], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[LENS[2]], &[1], 0).unwrap()),
            &JOBS,
            &RUNS,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
            &mut binding,
        ));
    });
    assert!(outcome.unwrap().is_err(), "short output");
    assert_eq!(out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
    assert_eq!((count, binding.retained_bytes()), (0, held), "short output");

    let overflow = [JOBS[0], job(0, 0, 0, usize::MAX, 1, 2)];
    let overlap = [JOBS[0], job(21, 6, 8, 2, 5, 3)];
    for (what, jobs) in [("overflow", &overflow), ("overlap", &overlap)] {
        let mut outcome = None;
        let count = builds(|| {
            outcome = Some(bound_call::<f64>(
                &mut executor,
                &mut binding,
                jobs,
                &[1, 1],
                [0, 0, 0],
                6.0,
                Some(Complex64::new(f64::NAN, 0.0)),
                one,
                zero,
            ));
        });
        let (result, found, _) = outcome.unwrap();
        assert!(result.is_err(), "{what}");
        assert!(found.iter().all(|v| v.is_nan()), "{what}: written");
        assert_eq!((count, binding.retained_bytes()), (0, held), "{what}");
        assert_eq!(valid(&mut executor, &mut binding), 0, "{what}: reuse");
    }

    // A different valid descriptor (the same jobs in another order).
    let reordered = [JOBS[1], JOBS[0], JOBS[2]];
    let mut outcome = None;
    let count = builds(|| {
        outcome = Some(bound_call::<f64>(
            &mut executor,
            &mut binding,
            &reordered,
            &RUNS,
            [0, 0, 0],
            7.0,
            None,
            one,
            zero,
        ));
    });
    let (result, found, want) = outcome.unwrap();
    result.unwrap();
    assert_matches(&found, &want, "reordered");
    assert_eq!(count, native(1), "reordered");
    assert_eq!(valid(&mut executor, &mut binding), native(1), "back");

    // Another dtype with the same jobs.
    let mut outcome = None;
    let count = builds(|| {
        outcome = Some(bound_call::<f32>(
            &mut executor,
            &mut binding,
            &JOBS,
            &RUNS,
            [0, 0, 0],
            8.0,
            None,
            one,
            zero,
        ));
    });
    let (result, found, want) = outcome.unwrap();
    result.unwrap();
    assert_matches(&found, &want, "f32");
    assert_eq!(count, native(1), "f32");
    assert_eq!(valid(&mut executor, &mut binding), native(1), "f64 again");
}
