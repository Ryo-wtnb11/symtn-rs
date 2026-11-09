//! The native block SVD kernel through `factorize_batch`: factors against an
//! independent `Q1 diag(sigma) Q2^H` construction for every scalar, engine
//! width and batch policy (degenerate singular values compared as subspace
//! projectors); determinism for a fixed engine; errors identical to the
//! per-matrix entry, lowest input first; the provider work estimate and the
//! widths each policy reserves; and a lane panic breaking the engine.

use super::*;
use crate::block_kernel::driver::{self, cost, BlockOp, Entry, LaneMembers, OpKind};
use crate::block_kernel::factor;
use crate::lanes::{self, GrantKey, LaneTables};
use std::time::Duration;
use symtn_core::StructureCacheKind;
use tcontract_exec::{BatchPolicy, BatchStrategy, Exec, WorkModel};

pub(super) use super::numerics;

pub(super) trait Scalar: Copy + std::fmt::Debug + 'static {
    const COMPLEX: bool;
    const EPS: f64;
    fn make(value: Complex64) -> Self;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
}

macro_rules! scalar {
    ($ty:ty, $variant:ident, $complex:expr, $eps:expr, |$v:ident| $make:expr) => {
        impl Scalar for $ty {
            const COMPLEX: bool = $complex;
            const EPS: f64 = $eps;
            fn make($v: Complex64) -> Self {
                $make
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
        }
    };
}

scalar!(f32, F32, false, f32::EPSILON as f64, |v| v.re as f32);
scalar!(f64, F64, false, f64::EPSILON, |v| v.re);
scalar!(Complex32, C32, true, f32::EPSILON as f64, |v| {
    Complex32::new(v.re as f32, v.im as f32)
});
scalar!(Complex64, C64, true, f64::EPSILON, |v| v);

pub(super) fn engine(threads: usize, policy: BatchPolicy) -> CpuEngine {
    CpuEngine::new(Pool::builder().threads(threads).build().unwrap()).with_batch_policy(policy)
}

pub(super) fn policies() -> Vec<(usize, BatchPolicy)> {
    let split = |lanes| {
        BatchPolicy::new(BatchStrategy::Split {
            lanes: std::num::NonZeroUsize::new(lanes).unwrap(),
        })
    };
    vec![
        (1, BatchPolicy::default()),
        (4, BatchPolicy::default()),
        (4, BatchPolicy::new(BatchStrategy::Outer)),
        (4, BatchPolicy::new(BatchStrategy::Inner)),
        (4, BatchPolicy::new(BatchStrategy::Sequential)),
        (5, split(2)),
    ]
}

/// Column-major `n x n` Householder reflector `I - 2 v v^H / (v^H v)`, an
/// exactly unitary matrix built without any factorization.
pub(super) fn reflector(n: usize, seed: usize, complex: bool) -> Vec<Complex64> {
    let v: Vec<Complex64> = (0..n)
        .map(|i| {
            let re = ((7 * i + 3 * seed) % 11) as f64 - 4.5;
            let im = if complex {
                ((5 * i + seed) % 7) as f64 - 2.5
            } else {
                0.0
            };
            Complex64::new(re, im)
        })
        .collect();
    let norm: f64 = v.iter().map(|x| x.norm_sqr()).sum();
    let mut h = vec![Complex64::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..n {
            let identity = if i == j { 1.0 } else { 0.0 };
            h[i + n * j] = Complex64::new(identity, 0.0) - 2.0 * v[i] * v[j].conj() / norm;
        }
    }
    h
}

/// One block of the oracle: `A = Q1[:, :k] diag(sigma) Q2[:, :k]^H` with
/// `sigma` non-increasing.
struct Block {
    rows: usize,
    cols: usize,
    sigma: Vec<f64>,
    q1: Vec<Complex64>,
    q2: Vec<Complex64>,
    a: Vec<Complex64>,
}

fn block(rows: usize, cols: usize, sigma: &[f64], seed: usize, complex: bool) -> Block {
    assert_eq!(sigma.len(), rows.min(cols));
    let q1 = reflector(rows, seed, complex);
    let q2 = reflector(cols, seed + 1, complex);
    let mut a = vec![Complex64::new(0.0, 0.0); rows * cols];
    for (l, &s) in sigma.iter().enumerate() {
        for j in 0..cols {
            for i in 0..rows {
                a[i + rows * j] += q1[i + rows * l] * s * q2[j + cols * l].conj();
            }
        }
    }
    Block {
        rows,
        cols,
        sigma: sigma.to_vec(),
        q1,
        q2,
        a,
    }
}

/// Tall, wide, square, degenerate and rank-deficient blocks of unequal
/// cost, plus enough small ones that every lane gets several jobs.
fn blocks(complex: bool) -> Vec<Block> {
    let mut blocks = vec![
        block(6, 6, &[3.0, 3.0, 2.0, 1.0, 1.0, 0.5], 1, complex),
        block(9, 4, &[2.0, 2.0, 2.0, 1.0], 2, complex),
        block(4, 9, &[4.0, 1.0, 1.0, 1.0], 3, complex),
        block(5, 5, &[2.0, 1.0, 1.0, 0.0, 0.0], 4, complex),
        block(1, 1, &[1.5], 5, complex),
        block(
            12,
            10,
            &[5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.25],
            6,
            complex,
        ),
    ];
    for seed in 0..8 {
        blocks.push(block(3, 2, &[2.0, 1.0], 10 + seed, complex));
    }
    blocks
}

/// The factors as wide complex column-major payloads.
pub(super) fn wide(tensor: &DenseTensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DenseDType::F32 => tensor
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(x.into(), 0.0))
            .collect(),
        DenseDType::F64 => tensor
            .as_f64_slice()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect(),
        DenseDType::C32 => tensor
            .as_c32_slice()
            .unwrap()
            .iter()
            .map(|x| Complex64::new(x.re.into(), x.im.into()))
            .collect(),
        DenseDType::C64 => tensor.as_c64_slice().unwrap().to_vec(),
        other => panic!("unexpected factor dtype {other:?}"),
    }
}

/// `X[:, cols] X[:, cols]^H` of a column-major `rows`-row matrix.
fn projector(x: &[Complex64], rows: usize, cols: &[usize]) -> Vec<Complex64> {
    let mut p = vec![Complex64::new(0.0, 0.0); rows * rows];
    for &c in cols {
        for j in 0..rows {
            for i in 0..rows {
                p[i + rows * j] += x[i + rows * c] * x[j + rows * c].conj();
            }
        }
    }
    p
}

fn check_block<T: Scalar>(block: &Block, factors: &[DenseTensor]) {
    let (m, n) = (block.rows, block.cols);
    let k = m.min(n);
    let terms = m.max(n);
    assert_eq!(factors.len(), 3);
    assert_eq!(factors[0].shape(), [m, k]);
    assert_eq!(factors[1].shape(), [k]);
    assert_eq!(factors[2].shape(), [k, n]);
    let (u, s, vh) = (wide(&factors[0]), wide(&factors[1]), wide(&factors[2]));
    let want: Vec<Complex64> = block
        .sigma
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    let tol = |scale: f64| numerics::K * (terms as f64).sqrt() * T::EPS * scale.max(1.0);
    let smax = block.sigma[0];
    for (index, (got, want)) in s.iter().zip(&want).enumerate() {
        assert!(
            (got - want).norm() <= tol(smax),
            "singular value {index}: {got} against {want}"
        );
    }
    // Reconstruction and orthonormal factors.
    for j in 0..n {
        for i in 0..m {
            let got: Complex64 = (0..k).map(|l| u[i + m * l] * s[l] * vh[l + k * j]).sum();
            assert!((got - block.a[i + m * j]).norm() <= tol(smax) * k as f64);
        }
    }
    for a in 0..k {
        for b in 0..k {
            let identity = if a == b { 1.0 } else { 0.0 };
            let uu: Complex64 = (0..m).map(|i| u[i + m * a].conj() * u[i + m * b]).sum();
            let vv: Complex64 = (0..n).map(|j| vh[a + k * j] * vh[b + k * j].conj()).sum();
            assert!((uu - identity).norm() <= tol(1.0) * k as f64);
            assert!((vv - identity).norm() <= tol(1.0) * k as f64);
        }
    }
    // Each nonzero singular value's subspace, degenerate or not, against the
    // oracle's: the projector is gauge-free. The conditioning factor is
    // sigma_max over the gap to the neighbouring values (Wedin's bound).
    let v: Vec<Complex64> = (0..k)
        .flat_map(|l| (0..n).map(move |j| (l, j)))
        .map(|(l, j)| vh[l + k * j].conj())
        .collect();
    let mut start = 0;
    while start < k {
        let value = block.sigma[start];
        let end = start
            + block.sigma[start..]
                .iter()
                .take_while(|&&x| x == value)
                .count();
        if value > 0.0 {
            let group: Vec<usize> = (start..end).collect();
            let gap = [start.checked_sub(1), (end < k).then_some(end)]
                .into_iter()
                .flatten()
                .map(|other| (block.sigma[other] - value).abs())
                .fold(value, f64::min);
            let conditioning = smax / gap;
            assert_projectors_close::<T>(
                "left singular subspace",
                &projector(&u, m, &group),
                &projector(&block.q1, m, &group),
                terms,
                conditioning,
            );
            // `v` holds V column-major with `n` rows (only the first k columns).
            assert_projectors_close::<T>(
                "right singular subspace",
                &projector(&v, n, &group),
                &projector(&block.q2, n, &group),
                terms,
                conditioning,
            );
        }
        start = end;
    }
}

/// The workspace tolerance rule at the input scalar's precision (the
/// projectors are formed in `Complex64` from rounded factors).
#[track_caller]
fn assert_projectors_close<T: Scalar>(
    what: &str,
    got: &[Complex64],
    want: &[Complex64],
    terms: usize,
    conditioning: f64,
) {
    let bound = numerics::K * (terms as f64).sqrt() * T::EPS * conditioning;
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).norm() <= bound,
            "{what}: entry {index} is {g} against {w} (bound {bound:e})"
        );
    }
}

fn oracle_matches<T: Scalar>(real: DenseDType) {
    let blocks = blocks(T::COMPLEX);
    let data: Vec<Vec<T>> = blocks
        .iter()
        .map(|b| b.a.iter().map(|&x| T::make(x)).collect())
        .collect();
    let layouts: Vec<_> = blocks
        .iter()
        .map(|b| ([b.rows, b.cols], [1usize, b.rows]))
        .collect();
    let inputs: Vec<DenseRead<'_>> = data
        .iter()
        .zip(&layouts)
        .map(|(d, (shape, strides))| T::read(DenseView::new(d, shape, strides, 0).unwrap()))
        .collect();
    for (threads, policy) in policies() {
        let mut engine = engine(threads, policy);
        let outputs = factorize_all(
            &mut DefaultDenseExecutor::new(),
            &mut engine,
            DenseFactorization::Svd,
            &inputs,
        )
        .unwrap();
        assert_eq!(outputs.len(), blocks.len());
        for (block, factors) in blocks.iter().zip(&outputs) {
            assert_eq!(factors[1].dtype(), real, "singular values are real");
            check_block::<T>(block, factors);
        }
    }
}

/// Purpose: every scalar's compact SVD equals the constructed one (values,
/// subspaces, reconstruction, orthonormality) under every engine width and
/// policy.
#[test]
fn factors_match_the_constructed_svd_f32() {
    oracle_matches::<f32>(DenseDType::F32);
}

/// Purpose: as above, for `f64`.
#[test]
fn factors_match_the_constructed_svd_f64() {
    oracle_matches::<f64>(DenseDType::F64);
}

/// Purpose: as above, for `Complex32`, whose singular values are `f32`.
#[test]
fn factors_match_the_constructed_svd_c32() {
    oracle_matches::<Complex32>(DenseDType::F32);
}

/// Purpose: as above, for `Complex64`, whose singular values are `f64`.
#[test]
fn factors_match_the_constructed_svd_c64() {
    oracle_matches::<Complex64>(DenseDType::F64);
}

pub(super) fn col_major<'a, T>(
    data: &'a [T],
    layout: &'a ([usize; 2], [usize; 2]),
) -> DenseView<'a, T> {
    DenseView::new(data, &layout.0, &layout.1, 0).unwrap()
}

/// Every factor entry's bit pattern (real and imaginary parts): `-0.0`
/// against `0.0` or two NaN payloads are differences here.
pub(super) fn bits(tensor: &DenseTensor) -> Vec<(u64, u64)> {
    wide(tensor)
        .iter()
        .map(|x| (x.re.to_bits(), x.im.to_bits()))
        .collect()
}

/// Purpose: the determinism contract — for one engine configuration
/// (provider, build, width, policy) and input, the factors are bitwise
/// reproducible; nothing is promised across widths or policies.
#[test]
fn factors_are_bitwise_reproducible_for_a_fixed_engine() {
    let blocks = blocks(true);
    let data: Vec<Vec<Complex64>> = blocks.iter().map(|b| b.a.clone()).collect();
    let layouts: Vec<_> = blocks
        .iter()
        .map(|b| ([b.rows, b.cols], [1usize, b.rows]))
        .collect();
    let inputs: Vec<_> = data
        .iter()
        .zip(&layouts)
        .map(|(d, l)| DenseRead::C64(col_major(d, l)))
        .collect();
    for (threads, policy) in policies() {
        let mut engine = engine(threads, policy);
        let mut run = || {
            factorize_all(
                &mut DefaultDenseExecutor::new(),
                &mut engine,
                DenseFactorization::Svd,
                &inputs,
            )
            .unwrap()
            .iter()
            .map(|factors| factors.iter().map(bits).collect::<Vec<_>>())
            .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "{threads} threads, {policy:?}");
    }
}

/// The batch's error and the per-matrix entry's error on `inputs[failing]`.
fn batch_and_per_call_errors(
    engine: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
    failing: usize,
) -> (DenseError, DenseError) {
    let batch = factorize_all(
        &mut DefaultDenseExecutor::new(),
        engine,
        DenseFactorization::Svd,
        inputs,
    )
    .unwrap_err();
    let per_call = DefaultDenseExecutor::new()
        .svd(engine, inputs[failing])
        .unwrap_err();
    (batch, per_call)
}

/// Purpose: the batch reports the lowest failing input's error, equal to the
/// per-matrix entry's error for that input, whether it fails before the
/// engine (dtype), inside it (rank) or in the provider (non-finite input).
#[test]
fn lowest_failing_input_reports_the_per_matrix_error() {
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let nan = vec![1.0_f64, f64::NAN, 3.0, 4.0];
    let ints = vec![1_i32, 2, 3, 4];
    let square = ([2usize, 2], [1usize, 2]);
    let vector = ([4usize], [1usize]);
    let ok = DenseRead::F64(col_major(&good, &square));
    let non_finite = DenseRead::F64(col_major(&nan, &square));
    let integer = DenseRead::I32(DenseView::new(&ints, &square.0, &square.1, 0).unwrap());
    let rank_one = DenseRead::F64(DenseView::new(&good, &vector.0, &vector.1, 0).unwrap());
    let cases: [(&[DenseRead<'_>], usize); 4] = [
        (&[ok, non_finite, ok, integer], 1),
        (&[ok, integer, ok, non_finite], 1),
        (&[ok, rank_one, non_finite], 1),
        (&[non_finite, ok, ok, ok, ok, ok], 0),
    ];
    for (threads, policy) in policies() {
        let mut engine = engine(threads, policy);
        for (inputs, failing) in cases {
            let (batch, per_call) = batch_and_per_call_errors(&mut engine, inputs, failing);
            assert_eq!(batch, per_call, "{threads} threads, {policy:?}");
        }
        let (dtype, _) = batch_and_per_call_errors(&mut engine, cases[1].0, 1);
        assert!(matches!(
            dtype,
            DenseError::Unsupported { op: "svd_read", .. }
        ));
        let (numeric, _) = batch_and_per_call_errors(&mut engine, cases[0].0, 1);
        assert!(matches!(
            numeric,
            DenseError::Backend { op: "svd_read", .. }
        ));
    }
}

/// Purpose: the full SVD batch's validation errors (rank, dtype) are
/// tenferro's `svd_full_read` errors exactly, labelled `svd_full`, not the
/// compact SVD's.
#[test]
fn full_svd_validation_errors_match_tenferro() {
    use tenferro_linalg::TensorReadLinalgExt;
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ints = vec![1_i32, 2, 3, 4];
    let rank_one = DenseRead::F64(DenseView::new(&good, &[4], &[1], 0).unwrap());
    let integer = DenseRead::I32(DenseView::new(&ints, &[2, 2], &[1, 2], 0).unwrap());
    let mut engine = engine(1, BatchPolicy::default());
    for input in [rank_one, integer] {
        let batch = factorize_all(
            &mut DefaultDenseExecutor::new(),
            &mut engine,
            DenseFactorization::SvdFull,
            &[input],
        )
        .unwrap_err();
        let view = crate::tenferro_adapter::tenferro_view(input).unwrap();
        let reference = tenferro_tensor::TensorRead::from_view(view)
            .svd_full_read(&mut engine)
            .unwrap_err();
        assert_eq!(
            batch,
            crate::tenferro_adapter::tenferro_error("svd_full_read", reference),
            "{:?} {:?}",
            input.dtype(),
            input.shape()
        );
    }
}

/// Every outcome `factorize_batch` delivers to a sink that accepts errors,
/// with its index, and the batch's own result.
#[allow(clippy::type_complexity)]
fn recorded_svd_batch(
    engine: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
) -> (
    Vec<(usize, Result<Vec<DenseTensor>, DenseError>)>,
    Result<(), DenseError>,
) {
    let mut outcomes = Vec::new();
    let result = DefaultDenseExecutor::new().factorize_batch(
        engine,
        DenseFactorization::Svd,
        inputs,
        &mut |index, outcome| {
            outcomes.push((index, outcome));
            Ok(())
        },
    );
    (outcomes, result)
}

/// Purpose: the native batch keeps each admitted input's own outcome and
/// delivers every one in input order, errors included; a pre-entry refusal
/// is delivered as the outcome after the admitted prefix and later inputs
/// are not delivered; an engine that refuses entry is the batch's own error
/// with nothing delivered.
#[test]
fn every_admitted_outcome_and_the_refusal_reach_the_sink_in_order() {
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let nan = vec![1.0_f64, f64::NAN, 3.0, 4.0];
    let ints = vec![1_i32, 2, 3, 4];
    let square = ([2usize, 2], [1usize, 2]);
    let ok = DenseRead::F64(col_major(&good, &square));
    let non_finite = DenseRead::F64(col_major(&nan, &square));
    let integer = DenseRead::I32(DenseView::new(&ints, &square.0, &square.1, 0).unwrap());
    let rank_one = DenseRead::F64(DenseView::new(&good, &[4], &[1], 0).unwrap());
    let per_call = |engine: &mut CpuEngine, input| DefaultDenseExecutor::new().svd(engine, input);
    for (threads, policy) in policies() {
        let mut engine = engine(threads, policy);
        let (outcomes, result) = recorded_svd_batch(&mut engine, &[rank_one, ok, non_finite]);
        assert_eq!(result, Ok(()));
        let indices: Vec<_> = outcomes.iter().map(|(index, _)| *index).collect();
        assert_eq!(indices, [0, 1, 2], "{threads} threads, {policy:?}");
        assert_eq!(
            outcomes[0].1.as_ref().unwrap_err(),
            &per_call(&mut engine, rank_one).unwrap_err()
        );
        assert_eq!(outcomes[1].1.as_ref().unwrap().len(), 3);
        assert_eq!(
            outcomes[2].1.as_ref().unwrap_err(),
            &per_call(&mut engine, non_finite).unwrap_err()
        );

        let (outcomes, result) = recorded_svd_batch(&mut engine, &[ok, non_finite, integer, ok]);
        assert_eq!(result, Ok(()));
        let indices: Vec<_> = outcomes.iter().map(|(index, _)| *index).collect();
        assert_eq!(indices, [0, 1, 2], "{threads} threads, {policy:?}");
        assert!(outcomes[0].1.is_ok());
        assert_eq!(
            outcomes[2].1.as_ref().unwrap_err(),
            &per_call(&mut engine, integer).unwrap_err()
        );
    }

    let mut broken = engine(4, BatchPolicy::new(BatchStrategy::Outer));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        driver::drive(
            &mut broken,
            &PanicsAt(0),
            &mut [(); 2],
            &mut LaneTables::default(),
        )
    }));
    assert!(unwound.is_err() && broken.is_broken());
    let (outcomes, result) = recorded_svd_batch(&mut broken, &[ok, ok]);
    assert!(outcomes.is_empty());
    assert_eq!(result.unwrap_err(), per_call(&mut broken, ok).unwrap_err());
}

/// Purpose: zero-extent inputs return the per-matrix entry's empty factor
/// shapes without a provider call failing the batch.
#[test]
fn zero_extent_inputs_return_the_per_matrix_shapes() {
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let empty: Vec<f64> = Vec::new();
    let layouts = [
        ([0usize, 3], [1usize, 1]),
        ([3, 0], [1, 3]),
        ([0, 0], [1, 1]),
        ([2, 2], [1, 2]),
    ];
    let inputs: Vec<_> = layouts
        .iter()
        .map(|l| {
            let data = if l.0.contains(&0) { &empty } else { &good };
            DenseRead::F64(col_major(data, l))
        })
        .collect();
    let mut engine = engine(4, BatchPolicy::new(BatchStrategy::Outer));
    let batch = factorize_all(
        &mut DefaultDenseExecutor::new(),
        &mut engine,
        DenseFactorization::Svd,
        &inputs,
    )
    .unwrap();
    for (factors, &input) in batch.iter().zip(&inputs) {
        let want = DefaultDenseExecutor::new().svd(&mut engine, input).unwrap();
        let shapes = |f: &[DenseTensor]| f.iter().map(|t| t.shape().to_vec()).collect::<Vec<_>>();
        assert_eq!(shapes(factors), shapes(&want));
    }
}

/// Purpose: an empty input whose trailing extents multiply past `usize`
/// (`[0, 0, usize::MAX / 2, 3]`, no element) is placed and solved like any
/// empty input on a pooled `Auto` engine: its empty factors are delivered
/// and the engine stays usable.
#[test]
fn an_empty_input_with_an_unrepresentable_batch_product_solves_empty() {
    let shape = [0, 0, usize::MAX / 2, 3];
    let input = DenseRead::F64(DenseView::new(&[], &shape, &[0; 4], 0).unwrap());
    let mut engine = engine(2, BatchPolicy::default());
    let (outcomes, result) = recorded_svd_batch(&mut engine, &[input]);
    assert_eq!(result, Ok(()));
    assert_eq!(outcomes.len(), 1);
    let factors = outcomes[0].1.as_ref().unwrap();
    assert!(factors.iter().all(|factor| factor.shape().contains(&0)));
    assert!(!engine.is_broken());
}

/// The placement's lane shares and the batch decision's lane count for jobs
/// of `work` multiply-adds (every job granted one slot), computed through
/// the kernel's own cost function inside the linalg entry.
fn allocation(threads: usize, policy: BatchPolicy, work: &[f64]) -> (Vec<usize>, usize) {
    engine(threads, policy)
        .with_linalg_pool(|exec, _| {
            let model = exec.batch_policy().model;
            let times: Vec<_> = work
                .iter()
                .map(|&w| cost(OpKind::Factorization, w, &model))
                .collect();
            let mean = work.iter().sum::<f64>() / work.len() as f64;
            let mut tables = LaneTables::default();
            let grant = Some(|_, _, _| 1);
            let cache = lanes::LaneAllocationCache::new(StructureCacheKind::LaneAllocations, 0);
            let key = (OpKind::Factorization, |_| [0; 2]);
            lanes::allocate(exec, &times, key, mean, grant, &cache, &mut tables);
            let placement = tables.placement();
            let shares = (0..placement.lanes()).map(|l| placement.share(l)).collect();
            let decision = exec.batch_decision(&tcontract_exec::BatchQuery::new(work.len(), mean));
            Ok((shares, decision.lanes))
        })
        .unwrap()
}

fn budgets((shares, _): &(Vec<usize>, usize)) -> Vec<usize> {
    shares.clone()
}

/// Purpose: tlinalg's SVD work estimate, `m n min(m, n)` per item, four times
/// that for complex scalars, over every batch item.
#[test]
fn job_work_is_the_provider_svd_estimate() {
    let data = vec![0.0_f64; 2 * 3 * 4];
    let cdata = vec![Complex64::new(0.0, 0.0); 2 * 3];
    let tall = DenseRead::F64(DenseView::new(&data[..6], &[3, 2], &[1, 3], 0).unwrap());
    let wide = DenseRead::C64(DenseView::new(&cdata, &[2, 3], &[1, 2], 0).unwrap());
    let batched = DenseRead::F64(DenseView::new(&data, &[3, 2, 4], &[1, 3, 6], 0).unwrap());
    assert_eq!(factor::work::<factor::Svd>(tall), (3 * 2 * 2) as f64);
    assert_eq!(factor::work::<factor::Svd>(wide), (2 * 3 * 2 * 4) as f64);
    assert_eq!(factor::work::<factor::Svd>(batched), (3 * 2 * 2 * 4) as f64);
}

/// Purpose: each job's reserved width is the engine policy's: Auto shares
/// all slots over its lanes, forced Split/Outer reserve their decision,
/// Inner keeps the budget on one lane, Sequential one slot.
#[test]
fn reserved_widths_follow_the_engine_policy() {
    // Blocks large enough that every width pays under Auto.
    let large = [512.0_f64.powi(3); 3];
    let split = BatchPolicy::new(BatchStrategy::Split {
        lanes: std::num::NonZeroUsize::new(3).unwrap(),
    });
    assert_eq!(
        budgets(&allocation(8, BatchPolicy::default(), &large)),
        [3, 3, 2]
    );
    assert_eq!(budgets(&allocation(8, split, &large)), [2, 2, 2]);
    let outer = BatchPolicy::new(BatchStrategy::Outer);
    assert_eq!(budgets(&allocation(8, outer, &large)), [1, 1, 1]);
    let inner = BatchPolicy::new(BatchStrategy::Inner);
    assert_eq!(budgets(&allocation(8, inner, &large)), [8]);
    let sequential = BatchPolicy::new(BatchStrategy::Sequential);
    assert_eq!(budgets(&allocation(8, sequential, &large)), [1]);
}

/// Purpose: the lane decision takes the jobs' multiply-adds and the slot
/// decision their times (no double conversion): two real 48x48 SVDs at 15.6
/// multiply-adds per ns with a caller-set 1000 ns lane minimum get two lanes.
#[test]
fn lane_minimum_is_checked_against_work_not_time() {
    let data = vec![0.0_f64; 48 * 48];
    let block = DenseRead::F64(DenseView::new(&data, &[48, 48], &[1, 48], 0).unwrap());
    let model = WorkModel::default()
        .with_muladds_per_ns(15.6)
        .with_item_overhead_ns(0.0)
        .with_min_work_per_lane_ns(1000.0);
    let work = [factor::work::<factor::Svd>(block); 2];
    let (_, lanes) = allocation(8, BatchPolicy::default().with_model(model), &work);
    assert_eq!(lanes, 2);
}

/// A linalg-entry kernel whose block `self.0` panics.
pub(super) struct PanicsAt(pub(super) usize);

impl BlockOp for PanicsAt {
    type Job = ();
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = "svd_read";
    type Lane = ();

    fn lane(&self, _: &mut LaneMembers<'_, ()>, _: &Exec<'_>) {}

    fn work(&self, _: usize, _: &()) -> f64 {
        1.0
    }

    fn grant(&self, _: usize, _: &(), _: &Exec<'_>, _: usize, _: Duration) -> usize {
        1
    }

    fn grant_key(&self, _: usize, _: &()) -> GrantKey {
        [0; 2]
    }

    fn run(&self, index: usize, _: &mut (), (): &mut (), _: Exec<'_>) -> Result<(), DenseError> {
        assert_ne!(index, self.0, "injected lane panic");
        Ok(())
    }
}

/// Purpose: a panicking job unwinds after the lanes join through the guarded
/// linalg entry, which leaves the engine broken; the next batch then reports
/// the per-matrix entry's refusal, after the same pre-entry checks.
#[test]
fn lane_panic_breaks_the_engine() {
    let mut engine = engine(4, BatchPolicy::new(BatchStrategy::Outer));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        driver::drive(
            &mut engine,
            &PanicsAt(2),
            &mut [(); 4],
            &mut LaneTables::default(),
        )
    }));
    assert!(unwound.is_err());
    assert!(engine.is_broken());
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ints = vec![1_i32, 2, 3, 4];
    let square = ([2usize, 2], [1usize, 2]);
    let ok = DenseRead::F64(col_major(&good, &square));
    let integer = DenseRead::I32(DenseView::new(&ints, &square.0, &square.1, 0).unwrap());
    for inputs in [[ok, integer], [integer, ok]] {
        let (batch, per_call) = batch_and_per_call_errors(&mut engine, &inputs, 0);
        assert_eq!(batch, per_call);
    }
}
