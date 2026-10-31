//! The native block GEMM kernel through the public batch entries: values
//! against an independent column-major triple loop for every scalar, op
//! lowering, engine width and batch policy; one plan lookup per distinct
//! geometry; rejections before any write; the engine-derived reserved lane
//! widths; and the geometry-plan cache (a hit is the rebuilt plan, every key
//! field separates entries, a disabled or rejecting cache builds per lookup,
//! concurrent misses build once).

use super::*;
use crate::block_kernel::{gemm, schedule};
use tcontract_exec::{BatchPolicy, BatchStrategy};

trait Scalar: Copy + PartialEq + std::fmt::Debug + 'static {
    const TOL: f64;
    fn make(value: Complex64) -> Self;
    fn wide(self) -> Complex64;
    fn dense(value: Complex64) -> DenseScalar;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
}

macro_rules! scalar {
    ($ty:ty, $variant:ident, $tol:expr, |$v:ident| $make:expr, |$s:ident| $wide:expr) => {
        impl Scalar for $ty {
            const TOL: f64 = $tol;
            fn make($v: Complex64) -> Self {
                $make
            }
            fn wide(self) -> Complex64 {
                let $s = self;
                $wide
            }
            fn dense(value: Complex64) -> DenseScalar {
                DenseScalar::$variant(Self::make(value))
            }
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
        }
    };
}
scalar!(f32, F32, 1e-3, |v| v.re as f32, |s| c64(s as f64, 0.0));
scalar!(f64, F64, 1e-11, |v| v.re, |s| c64(s, 0.0));
scalar!(
    Complex32,
    C32,
    1e-3,
    |v| Complex32::new(v.re as f32, v.im as f32),
    |s| c64(s.re as f64, s.im as f64)
);
scalar!(Complex64, C64, 1e-11, |v| v, |s| s);

/// `op(X)[r, c]` of a job operand stored at `offset`: Identity is
/// column-major `rows x cols`, Transpose/Adjoint read the `cols x rows`
/// column-major storage transposed, and Adjoint conjugates.
fn element(
    data: &[Complex64],
    offset: usize,
    shape: [usize; 2],
    at: [usize; 2],
    op: MatrixOp,
) -> Complex64 {
    let [rows, cols] = shape;
    let [r, c] = at;
    match op {
        MatrixOp::Identity => data[offset + r + rows * c],
        MatrixOp::Transpose => data[offset + c + cols * r],
        MatrixOp::Adjoint => data[offset + c + cols * r].conj(),
    }
}

/// `alpha op(A) op(B) + beta D` per job by scalar loops on wide values;
/// `beta == 0` overwrites, and untouched elements keep their value.
#[allow(clippy::too_many_arguments)]
fn oracle(
    jobs: &[DenseGemmBatchJob],
    base: [usize; 3],
    out: &[Complex64],
    lhs: &[Complex64],
    rhs: &[Complex64],
    ops: [MatrixOp; 2],
    alpha: Complex64,
    beta: Complex64,
) -> Vec<Complex64> {
    let mut want = out.to_vec();
    for job in jobs {
        let (m, k, n) = (job.rows, job.contracted, job.cols);
        for j in 0..n {
            for i in 0..m {
                let sum = (0..k)
                    .map(|p| {
                        element(lhs, base[1] + job.lhs_offset, [m, k], [i, p], ops[0])
                            * element(rhs, base[2] + job.rhs_offset, [k, n], [p, j], ops[1])
                    })
                    .sum::<Complex64>();
                let at = base[0] + job.dst_offset + i + m * j;
                want[at] = if beta == c64(0.0, 0.0) {
                    alpha * sum
                } else {
                    alpha * sum + beta * out[at]
                };
            }
        }
    }
    want
}

fn value(i: usize, salt: f64) -> Complex64 {
    let x = i as f64 + salt;
    c64((0.37 * x).sin(), (0.11 * x).cos())
}

/// Runs `jobs` as one public batch call (identity entry when both ops are
/// Identity) and returns the result and the geometry plans looked up.
#[allow(clippy::too_many_arguments)]
fn run<T: Scalar>(
    engine: &mut CpuEngine,
    jobs: &[DenseGemmBatchJob],
    base: [usize; 3],
    out: &mut [T],
    lhs: &[T],
    rhs: &[T],
    ops: [MatrixOp; 2],
    alpha: Complex64,
    beta: Complex64,
) -> (Result<(), DenseError>, usize) {
    let mut executor = DefaultDenseExecutor::new();
    let shapes = [
        out.len() - base[0],
        lhs.len() - base[1],
        rhs.len() - base[2],
    ];
    let runs = strided_batch_runs(jobs);
    let recording = crate::probe::record();
    let result = executor.matmul_batch_axpby_with_ops_into(
        engine,
        T::write(DenseViewMut::new(out, &shapes[..1], &[1], base[0]).unwrap()),
        T::read(DenseView::new(lhs, &shapes[1..2], &[1], base[1]).unwrap()),
        T::read(DenseView::new(rhs, &shapes[2..], &[1], base[2]).unwrap()),
        jobs,
        &runs,
        ops[0],
        ops[1],
        T::dense(alpha),
        T::dense(beta),
    );
    (result, recording.get(crate::probe::Event::GemmPlanLookup))
}

/// Heterogeneous, non-affine jobs with repeated geometries, gaps, outputs
/// out of job order, a zero-`k` job (`beta D` only) and a zero-`m` job.
fn mixed_jobs() -> (Vec<DenseGemmBatchJob>, usize) {
    let shapes = [
        (3, 4, 2),
        (2, 5, 3),
        (3, 4, 2),
        (4, 1, 5),
        (2, 5, 3),
        (3, 0, 2),
        (0, 3, 2),
        (3, 4, 2),
    ];
    // Destinations reversed with a one-element gap.
    let mut dst = shapes.iter().map(|&(m, _, n)| m * n + 1).sum::<usize>();
    let (mut lhs, mut rhs) = (1, 2);
    let jobs = shapes
        .iter()
        .map(|&(m, k, n)| {
            dst -= m * n + 1;
            let job = batch_job((m, k, n), (dst, lhs, rhs));
            lhs += m * k + 2;
            rhs += k * n + 1;
            job
        })
        .collect();
    // Distinct (m, k, n) without the duplicates: 3x4x2, 2x5x3, 4x1x5, 3x0x2, 0x3x2.
    (jobs, 5)
}

fn check<T: Scalar>(engine: &mut CpuEngine, ops: [MatrixOp; 2], beta: Complex64, label: &str) {
    let (jobs, geometries) = mixed_jobs();
    let base = [2, 1, 3];
    let len = |f: fn(&DenseGemmBatchJob) -> usize| jobs.iter().map(f).max().unwrap() + 4;
    let out_len = base[0] + len(|j| j.dst_offset + j.rows * j.cols);
    let lhs_len = base[1] + len(|j| j.lhs_offset + j.rows * j.contracted);
    let rhs_len = base[2] + len(|j| j.rhs_offset + j.contracted * j.cols);
    let lhs: Vec<T> = (0..lhs_len).map(|i| T::make(value(i, 0.5))).collect();
    let rhs: Vec<T> = (0..rhs_len).map(|i| T::make(value(i, 7.0))).collect();
    let initial: Vec<T> = (0..out_len).map(|i| T::make(value(i, 3.0))).collect();
    let wide = |v: &[T]| v.iter().map(|x| x.wide()).collect::<Vec<_>>();
    let alpha = c64(0.75, -0.5);
    let alpha = T::make(alpha).wide();
    let beta = T::make(beta).wide();
    let want = oracle(
        &jobs,
        base,
        &wide(&initial),
        &wide(&lhs),
        &wide(&rhs),
        ops,
        alpha,
        beta,
    );
    let mut out = initial.clone();
    let (result, looked_up) = run(engine, &jobs, base, &mut out, &lhs, &rhs, ops, alpha, beta);
    result.unwrap();
    assert_eq!(
        looked_up, geometries,
        "{label}: one lookup per distinct geometry"
    );
    for (index, (found, want)) in wide(&out).iter().zip(&want).enumerate() {
        assert!(
            (found - want).norm() <= T::TOL * (1.0 + want.norm()),
            "{label}: element {index}: {found} vs {want}"
        );
    }
}

fn engines() -> Vec<(String, CpuEngine)> {
    let pool = |threads| crate::Pool::builder().threads(threads).build().unwrap();
    let mut engines = vec![("serial".to_string(), CpuEngine::serial())];
    for policy in [
        BatchPolicy::default(),
        BatchPolicy::new(BatchStrategy::Outer),
        BatchPolicy::new(BatchStrategy::Inner),
        BatchPolicy::new(BatchStrategy::Split {
            lanes: std::num::NonZeroUsize::new(2).unwrap(),
        }),
    ] {
        engines.push((
            format!("4 threads {:?}", policy.strategy),
            CpuEngine::new(pool(4)).with_batch_policy(policy),
        ));
    }
    engines
}

#[test]
fn every_scalar_op_and_policy_matches_the_dense_oracle() {
    let ops = [MatrixOp::Identity, MatrixOp::Transpose, MatrixOp::Adjoint];
    for (name, mut engine) in engines() {
        for lhs_op in ops {
            for rhs_op in ops {
                for beta in [c64(0.0, 0.0), c64(-0.25, 0.125)] {
                    let label = format!("{name} {lhs_op:?} {rhs_op:?} beta={beta}");
                    check::<f32>(&mut engine, [lhs_op, rhs_op], beta, &label);
                    check::<f64>(&mut engine, [lhs_op, rhs_op], beta, &label);
                    check::<Complex32>(&mut engine, [lhs_op, rhs_op], beta, &label);
                    check::<Complex64>(&mut engine, [lhs_op, rhs_op], beta, &label);
                }
            }
        }
    }
}

/// Review B2's decisive case: `A` stored `[1..6]` read as the transpose of a
/// column-major 3x2 matrix, times `B = [10, 20, 30]`, is `[140, 320]`; the
/// compact identity layout would read `[220, 280]`. It runs as one of two
/// disjoint, non-affine jobs with nonzero offsets and nontrivial alpha/beta.
#[test]
fn nonsquare_real_transpose_reads_the_lowered_strides() {
    for (name, mut engine) in engines() {
        let lhs = [-9.0, -9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -1.5];
        let rhs = [-9.0, 10.0, 20.0, 30.0, 2.0, 1.0];
        let jobs = [
            batch_job((2, 3, 1), (1, 1, 0)),
            batch_job((1, 2, 1), (4, 7, 3)),
        ];
        assert_eq!(strided_batch_runs(&jobs), [1, 1]);
        let mut out = [7.0, -1.0, 3.0, 11.0, -4.0, 5.0];
        let (alpha, beta) = (c64(2.0, 0.0), c64(-0.5, 0.0));
        let (result, _) = run::<f64>(
            &mut engine,
            &jobs,
            [1, 1, 1],
            &mut out,
            &lhs,
            &rhs,
            [MatrixOp::Transpose, MatrixOp::Identity],
            alpha,
            beta,
        );
        result.unwrap();
        // Job 0: [140, 320] at out[2..4]; job 1: [0.5, -1.5] . [2, 1] = -0.5 at out[5].
        let want = [
            7.0,
            -1.0,
            2.0 * 140.0 - 0.5 * 3.0,
            2.0 * 320.0 - 0.5 * 11.0,
            -4.0,
            2.0 * -0.5 - 0.5 * 5.0,
        ];
        for (found, want) in out.iter().zip(want) {
            assert!((found - want).abs() <= 1e-12, "{name}: {out:?} vs {want}");
        }
    }
}

/// A complex adjoint pair with base and job offsets and complex alpha/beta,
/// against the hand-written conjugate-transposed products.
#[test]
fn complex_adjoint_conjugates_and_transposes() {
    for (name, mut engine) in engines() {
        let a: Vec<Complex64> = (0..6)
            .map(|i| c64(1.0 + i as f64, 0.5 - i as f64))
            .collect();
        let b: Vec<Complex64> = (0..6)
            .map(|i| c64(-0.5 * i as f64, 2.0 + i as f64))
            .collect();
        let mut lhs = vec![c64(-99.0, 99.0); 3];
        lhs.extend(&a);
        lhs.extend(&a[..2]);
        let mut rhs = vec![c64(-99.0, 99.0); 2];
        rhs.extend(&b);
        rhs.extend(&b[..2]);
        // Job 0: m=2, k=3, n=2 (A stored 3x2, B stored 2x3); job 1: 1x2x1.
        let jobs = [
            batch_job((2, 3, 2), (3, 2, 1)),
            batch_job((1, 2, 1), (0, 8, 7)),
        ];
        let initial: Vec<Complex64> = (0..9).map(|i| c64(i as f64, -1.0)).collect();
        let mut out = initial.clone();
        let (alpha, beta) = (c64(0.5, -0.75), c64(0.25, 1.5));
        let (result, looked_up) = run::<Complex64>(
            &mut engine,
            &jobs,
            [1, 1, 1],
            &mut out,
            &lhs,
            &rhs,
            [MatrixOp::Adjoint, MatrixOp::Adjoint],
            alpha,
            beta,
        );
        result.unwrap();
        assert_eq!(looked_up, 2, "{name}");
        let mut want = initial.clone();
        // Job 0: (A^H)[i,p] = conj(a[p + 3 i]), (B^H)[p,j] = conj(b[j + 2 p]).
        for j in 0..2 {
            for i in 0..2 {
                let sum: Complex64 = (0..3)
                    .map(|p| a[p + 3 * i].conj() * b[j + 2 * p].conj())
                    .sum();
                let at = 1 + 3 + i + 2 * j;
                want[at] = alpha * sum + beta * initial[at];
            }
        }
        // Job 1: lhs[9..11] = a[0..2], rhs[8..10] = b[0..2] after the base offset.
        let sum = lhs[9].conj() * rhs[8].conj() + lhs[10].conj() * rhs[9].conj();
        want[1] = alpha * sum + beta * initial[1];
        for (found, want) in out.iter().zip(&want) {
            assert!(
                (found - want).norm() <= 1e-12,
                "{name}: {out:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn overlapping_destinations_are_rejected_before_any_write() {
    let mut engine = CpuEngine::new(crate::Pool::builder().threads(4).build().unwrap());
    let lhs = [1.0; 16];
    let rhs = [2.0; 16];
    // Destinations 0..4 and 3..6 overlap; job 2 is disjoint.
    let jobs = [
        batch_job((2, 2, 2), (0, 0, 0)),
        batch_job((3, 1, 1), (8, 4, 4)),
        batch_job((1, 3, 3), (3, 6, 6)),
    ];
    let mut out = [-5.0; 12];
    let (result, looked_up) = run::<f64>(
        &mut engine,
        &jobs,
        [0; 3],
        &mut out,
        &lhs,
        &rhs,
        [MatrixOp::Transpose, MatrixOp::Identity],
        c64(1.0, 0.0),
        c64(0.0, 0.0),
    );
    let error = result.unwrap_err();
    assert!(
        matches!(&error, DenseError::Backend { op: "grouped_gemm", message, .. } if message.contains("jobs 0 and 2 overlap")),
        "{error:?}"
    );
    assert_eq!(looked_up, 0, "rejected before any plan");
    assert_eq!(out, [-5.0; 12]);
}

#[test]
fn the_lowest_indexed_out_of_range_job_is_reported() {
    let mut engine = CpuEngine::new(crate::Pool::builder().threads(4).build().unwrap());
    let lhs = [1.0; 8];
    let rhs = [2.0; 8];
    // Jobs 1 and 3 read past their buffers; job 3's rhs fails, job 1's lhs.
    let jobs = [
        batch_job((1, 2, 1), (0, 0, 0)),
        batch_job((2, 4, 1), (1, 2, 0)),
        batch_job((1, 1, 1), (3, 0, 0)),
        batch_job((1, 2, 2), (4, 0, 6)),
    ];
    let mut out = [-5.0; 8];
    let (result, _) = run::<f64>(
        &mut engine,
        &jobs,
        [0; 3],
        &mut out,
        &lhs,
        &rhs,
        [MatrixOp::Adjoint, MatrixOp::Identity],
        c64(1.0, 0.0),
        c64(0.0, 0.0),
    );
    let error = result.unwrap_err();
    assert!(
        matches!(&error, DenseError::Backend { op: "grouped_gemm", message, .. } if message.starts_with("job 1 lhs")),
        "{error:?}"
    );
    assert_eq!(out, [-5.0; 8]);
}

/// The lanes and reserved widths the kernel derives from its own costs and
/// the engine's decisions: few large jobs under Auto share every slot (no
/// lane is capped at one thread), forced Split and Inner reserve their
/// decision, and a reserved width is what each job's child context gets.
#[test]
fn reserved_widths_follow_the_engine_under_auto_split_and_inner() {
    let pool = || crate::Pool::builder().threads(16).build().unwrap();
    let large = vec![batch_job((256, 256, 256), (0, 0, 0)); 5];
    let mut times = Vec::new();
    let mean = gemm::costs_into::<f64>(&large, &mut times);
    let widths = |policy: BatchPolicy| {
        let mut engine = CpuEngine::new(pool()).with_batch_policy(policy);
        engine
            .with_contraction_exec(|exec, _, _| {
                let allocation = schedule::allocate(exec, &times, mean);
                let seen: Vec<_> = (0..times.len())
                    .map(|_| std::sync::atomic::AtomicUsize::new(0))
                    .collect();
                schedule::run(
                    exec.reborrow(),
                    &allocation,
                    &mut vec![(); times.len()],
                    |j, _, child| {
                        seen[j].store(child.budget(), std::sync::atomic::Ordering::Relaxed);
                        Ok::<(), ()>(())
                    },
                )
                .unwrap();
                let per_job: Vec<usize> = seen.into_iter().map(|s| s.into_inner()).collect();
                let lanes: Vec<usize> = allocation.lanes.iter().map(|l| l.budget).collect();
                Ok((allocation.slots, lanes, per_job))
            })
            .unwrap()
    };
    let (slots, lanes, per_job) = widths(BatchPolicy::default());
    assert_eq!((slots, lanes.as_slice()), (16, &[4, 3, 3, 3, 3][..]));
    assert_eq!(per_job.iter().sum::<usize>(), 16);
    let split = BatchPolicy::new(BatchStrategy::Split {
        lanes: std::num::NonZeroUsize::new(2).unwrap(),
    });
    let (slots, lanes, per_job) = widths(split);
    assert_eq!((slots, lanes.as_slice()), (16, &[8, 8][..]));
    assert!(per_job.iter().all(|&w| w == 8));
    let (slots, lanes, per_job) = widths(BatchPolicy::new(BatchStrategy::Inner));
    assert_eq!((slots, lanes.as_slice()), (16, &[16][..]));
    assert!(per_job.iter().all(|&w| w == 16));

    // Many tiny jobs: every reserved width is one or the batch is serial.
    let tiny = vec![batch_job((2, 2, 2), (0, 0, 0)); 200];
    let mut times = Vec::new();
    let mean = gemm::costs_into::<f64>(&tiny, &mut times);
    let mut engine = CpuEngine::new(pool());
    let allocation = engine
        .with_contraction_exec(|exec, _, _| Ok(schedule::allocate(exec, &times, mean)))
        .unwrap();
    assert!(
        allocation.lanes.len() == 1 || allocation.lanes.iter().all(|l| l.budget == 1),
        "{allocation:?}"
    );
}

mod plan_cache {
    use super::*;
    use crate::block_kernel::gemm::{cached_plan, GemmPlanCache, Geometry};
    use std::sync::Arc;
    use symtn_core::{ErasedStructureCacheControl, StructureCache, StructureCacheKind};
    use tcontract::api::Scalar as PlanScalar;
    use tcontract::{Algorithm, Plan, SliceAccumulationSource};

    fn cache(budget: u64) -> GemmPlanCache {
        StructureCache::new(StructureCacheKind::GemmPlans, budget)
    }

    fn geometry(mkn: (usize, usize, usize), ops: [MatrixOp; 2]) -> Geometry {
        Geometry::new(&batch_job(mkn, (0, 0, 0)), ops).unwrap()
    }

    /// `plan` run serially on fixed inputs with nontrivial alpha and beta,
    /// widened bit for bit.
    fn execute<T: Scalar + PlanScalar + Default>(plan: &Plan<T>, at: &Geometry) -> Vec<[u64; 2]> {
        let [m, k, n] = at.mkn;
        let data = |len: usize, salt: f64| (0..len).map(|i| T::make(value(i, salt))).collect();
        let (lhs, rhs): (Vec<T>, Vec<T>) = (data(m * k, 0.5), data(k * n, 7.0));
        let mut out: Vec<T> = data(m * n, 3.0);
        plan.execute_slices_accum(
            &mut tcontract_exec::Exec::serial(),
            T::make(c64(0.75, -0.5)),
            (&lhs, 0),
            (&rhs, 0),
            T::make(c64(-0.25, 0.125)),
            SliceAccumulationSource::Output,
            (&mut out, 0),
        )
        .unwrap();
        out.iter()
            .map(|x| [x.wide().re.to_bits(), x.wide().im.to_bits()])
            .collect()
    }

    fn builds<R>(f: impl FnOnce() -> R) -> (R, usize) {
        let recording = crate::probe::record();
        let result = f();
        (result, recording.get(crate::probe::Event::GemmPlanBuild))
    }

    /// Small Faer geometries in every op lowering, and a `Complex64` one
    /// above tcontract's default Faer volume limit, which plans Packed.
    fn cases() -> Vec<Geometry> {
        let ops = [MatrixOp::Identity, MatrixOp::Transpose, MatrixOp::Adjoint];
        let mut cases = Vec::new();
        for a in ops {
            for b in ops {
                cases.push(geometry((3, 4, 2), [a, b]));
                cases.push(geometry((64, 64, 64), [a, b]));
            }
        }
        cases
    }

    fn hit_equals_rebuild<T: Scalar + PlanScalar + Default>() {
        let cache = cache(1 << 26);
        for at in cases() {
            let cold = cached_plan::<T>(&cache, at).unwrap();
            let (hit, built) = builds(|| cached_plan::<T>(&cache, at).unwrap());
            assert_eq!(built, 0, "{at:?}: a repeat is a hit");
            assert!(Arc::ptr_eq(&cold, &hit), "{at:?}");
            let rebuilt = at.plan::<T>().unwrap();
            assert_eq!(hit.report().algorithm, rebuilt.report().algorithm);
            assert_eq!(execute(&hit, &at), execute(&rebuilt, &at), "{at:?}");
        }
    }

    #[test]
    fn a_cached_plan_computes_what_a_rebuilt_plan_computes_bit_for_bit() {
        hit_equals_rebuild::<f32>();
        hit_equals_rebuild::<f64>();
        hit_equals_rebuild::<Complex32>();
        hit_equals_rebuild::<Complex64>();
    }

    #[test]
    fn every_reachable_strategy_is_covered_by_the_cases() {
        // What: the lowered GEMM always has nonempty m, n and k roles, so
        // tcontract never plans it Elementwise (all-batch); small geometries
        // plan Faer, and `Complex64` above the Faer volume limit Packed.
        let algorithm = |at: Geometry| at.plan::<Complex64>().unwrap().report().algorithm;
        for at in cases() {
            let want = if at.mkn == [64, 64, 64] {
                Algorithm::Packed
            } else {
                Algorithm::Faer
            };
            assert_eq!(algorithm(at), want, "{at:?}");
            assert_eq!(
                at.plan::<f64>().unwrap().report().algorithm,
                Algorithm::Faer
            );
        }
        for mkn in [(1, 1, 1), (3, 0, 2), (0, 3, 2), (0, 0, 0)] {
            let at = geometry(mkn, [MatrixOp::Identity; 2]);
            assert_ne!(algorithm(at), Algorithm::Elementwise, "{mkn:?}");
        }
    }

    #[test]
    fn every_key_field_separates_entries() {
        let cache = cache(1 << 24);
        let base = geometry((3, 4, 2), [MatrixOp::Identity; 2]);
        let first = cached_plan::<f64>(&cache, base).unwrap();
        let mut padded_d = base;
        padded_d.strides[2] = [1, 4];
        let variants = [
            ("m", geometry((4, 4, 2), [MatrixOp::Identity; 2])),
            ("k", geometry((3, 5, 2), [MatrixOp::Identity; 2])),
            ("n", geometry((3, 4, 3), [MatrixOp::Identity; 2])),
            (
                "A strides",
                geometry((3, 4, 2), [MatrixOp::Transpose, MatrixOp::Identity]),
            ),
            (
                "B strides",
                geometry((3, 4, 2), [MatrixOp::Identity, MatrixOp::Transpose]),
            ),
            ("D strides", padded_d),
        ];
        let charged = std::cell::Cell::new(cache.info().charged_bytes());
        let misses = std::cell::Cell::new(1);
        let check = |what: &str, lookup: &dyn Fn() -> bool| {
            // What: the changed field misses (a new plan is built), and the
            // next lookup of it hits.
            let (fresh, built) = builds(lookup);
            assert!(fresh, "{what}: a distinct entry");
            assert_eq!(built, 1, "{what}: built once");
            assert_eq!(builds(lookup).1, 0, "{what}: then hit");
            misses.set(misses.get() + 1);
            let info = cache.info();
            assert_eq!(info.misses(), misses.get(), "{what}");
            assert!(info.charged_bytes() > charged.get(), "{what}: charged");
            charged.set(info.charged_bytes());
        };
        for (what, at) in variants {
            check(what, &|| {
                !Arc::ptr_eq(&cached_plan::<f64>(&cache, at).unwrap(), &first)
            });
        }
        check("dtype", &|| cached_plan::<f32>(&cache, base).is_ok());
        // Adjoint and Transpose lower to the same strides; only the
        // conjugation flag tells the plans apart.
        let transposed = geometry((3, 4, 2), [MatrixOp::Transpose, MatrixOp::Transpose]);
        let plain = cached_plan::<Complex64>(&cache, transposed).unwrap();
        misses.set(misses.get() + 1);
        charged.set(cache.info().charged_bytes());
        for (what, ops) in [
            ("A conj", [MatrixOp::Adjoint, MatrixOp::Transpose]),
            ("B conj", [MatrixOp::Transpose, MatrixOp::Adjoint]),
        ] {
            let at = geometry((3, 4, 2), ops);
            assert_eq!(at.strides, transposed.strides, "{what}: premise");
            check(what, &|| {
                !Arc::ptr_eq(&cached_plan::<Complex64>(&cache, at).unwrap(), &plain)
            });
        }
        let info = cache.info();
        assert_eq!(
            (info.entries() as u64, info.admissions()),
            (misses.get(), misses.get())
        );
    }

    #[test]
    fn a_disabled_or_rejecting_cache_builds_every_lookup_with_the_same_result() {
        let at = geometry((64, 64, 64), [MatrixOp::Adjoint, MatrixOp::Identity]);
        let resident = cached_plan::<Complex64>(&cache(1 << 26), at).unwrap();
        let want = execute(&resident, &at);
        // Budget 0 admits nothing; 4 KiB admits at most 2 KiB, below one
        // plan's charge.
        for budget in [0, 4096] {
            let cache = cache(budget);
            for lookup in 1..=2 {
                let (plan, built) = builds(|| cached_plan::<Complex64>(&cache, at).unwrap());
                assert_eq!(built, 1, "budget {budget}: lookup {lookup} builds");
                assert_eq!(execute(&plan, &at), want, "budget {budget}");
            }
            let info = cache.info();
            assert_eq!(
                (
                    info.entries(),
                    info.misses(),
                    info.rejections(),
                    info.hits()
                ),
                (0, 2, 2, 0),
                "budget {budget}"
            );
        }
    }

    #[test]
    fn concurrent_first_lookups_of_one_geometry_build_once() {
        const THREADS: usize = 8;
        let cache = cache(1 << 26);
        let at = geometry((64, 64, 64), [MatrixOp::Identity; 2]);
        let start = std::sync::Barrier::new(THREADS);
        let results: Vec<(Arc<Plan<Complex64>>, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        builds(|| cached_plan::<Complex64>(&cache, at).unwrap())
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // What: one thread built, every thread got its plan, and the waiters
        // count as hits rather than duplicate builds.
        assert_eq!(results.iter().map(|(_, built)| built).sum::<usize>(), 1);
        assert!(results
            .iter()
            .all(|(plan, _)| Arc::ptr_eq(plan, &results[0].0)));
        let info = cache.info();
        assert_eq!(
            (info.misses(), info.hits(), info.duplicate_builds()),
            (1, THREADS as u64 - 1, 0)
        );
    }

    #[test]
    fn a_cleared_cache_rebuilds_the_same_plan() {
        let cache = cache(1 << 26);
        let at = geometry((5, 3, 4), [MatrixOp::Adjoint, MatrixOp::Transpose]);
        let before = cached_plan::<Complex32>(&cache, at).unwrap();
        ErasedStructureCacheControl::clear(&cache);
        let info = cache.info();
        assert_eq!(
            (info.entries(), info.charged_bytes(), info.misses()),
            (0, 0, 0)
        );
        let (after, built) = builds(|| cached_plan::<Complex32>(&cache, at).unwrap());
        // What: the clear dropped the entry (a held plan stays valid), and
        // the rebuild computes the same values.
        assert_eq!(built, 1);
        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(execute(&before, &at), execute(&after, &at));
        assert_eq!(cache.info().entries(), 1);
    }
}
