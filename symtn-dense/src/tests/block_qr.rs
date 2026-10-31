//! The native block QR kernel through `factorize_batch`: factors against an
//! independent `Q0 R0` construction for every scalar, engine width and batch
//! policy (unique up to one unit phase per column, which the dense entry
//! leaves to the provider); the per-matrix entry's factors and errors,
//! lowest input first; and the broken-engine refusal after a lane panic.

use super::block_svd::{col_major, engine, numerics, policies, reflector, wide, PanicsAt, Scalar};
use super::*;
use crate::block_kernel::driver::{self, LaneScratch};
use tcontract_exec::{BatchPolicy, BatchStrategy};

/// One block of the oracle: `A = Q0[:, :k] R0` with `R0` upper trapezoidal.
struct Block {
    rows: usize,
    cols: usize,
    /// Whether `R0`'s leading `k x k` block is nonsingular, which makes the
    /// factorization unique up to the column phases.
    full_rank: bool,
    q0: Vec<Complex64>,
    r0: Vec<Complex64>,
    a: Vec<Complex64>,
}

/// `R0` (column-major `k x n`): the given diagonal, small upper entries
/// (so the leading block stays well conditioned), zeros below.
fn block(rows: usize, cols: usize, diagonal: &[f64], seed: usize, complex: bool) -> Block {
    let k = rows.min(cols);
    assert_eq!(diagonal.len(), k);
    let q0 = reflector(rows, seed, complex);
    let mut r0 = vec![Complex64::new(0.0, 0.0); k * cols];
    for j in 0..cols {
        for i in 0..k.min(j + 1) {
            r0[i + k * j] = if i == j {
                Complex64::new(diagonal[i], 0.0)
            } else {
                let re = ((3 * i + 5 * j + seed) % 7) as f64 / 14.0 - 0.2;
                let im = if complex {
                    ((2 * i + j + seed) % 5) as f64 / 10.0 - 0.2
                } else {
                    0.0
                };
                Complex64::new(re, im)
            };
        }
    }
    let mut a = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            a[i + rows * j] = (0..k).map(|l| q0[i + rows * l] * r0[l + k * j]).sum();
        }
    }
    Block {
        rows,
        cols,
        full_rank: diagonal.iter().all(|&d| d != 0.0),
        q0,
        r0,
        a,
    }
}

/// Tall, wide, square, rank-deficient and 1x1 blocks of unequal cost, plus
/// enough small ones that every lane gets several jobs.
fn blocks(complex: bool) -> Vec<Block> {
    let mut blocks = vec![
        block(6, 6, &[3.0, 2.5, 2.0, 1.5, 1.0, 1.0], 1, complex),
        block(9, 4, &[2.0, 2.0, 1.5, 1.0], 2, complex),
        block(4, 9, &[4.0, 1.0, 1.5, 1.0], 3, complex),
        block(5, 5, &[2.0, 1.0, 0.0, 1.0, 0.0], 4, complex),
        block(1, 1, &[1.5], 5, complex),
        block(
            12,
            10,
            &[5.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0, 1.5, 1.0, 1.0],
            6,
            complex,
        ),
    ];
    for seed in 0..8 {
        blocks.push(block(3, 2, &[2.0, 1.0], 10 + seed, complex));
    }
    blocks
}

fn check_block<T: Scalar>(block: &Block, factors: &[DenseTensor]) {
    let (m, n) = (block.rows, block.cols);
    let k = m.min(n);
    assert_eq!(factors.len(), 2);
    assert_eq!(factors[0].shape(), [m, k]);
    assert_eq!(factors[1].shape(), [k, n]);
    let (q, r) = (wide(&factors[0]), wide(&factors[1]));
    let scale = block.a.iter().map(|x| x.norm()).fold(1.0, f64::max);
    let tol = |terms: usize, scale: f64| {
        numerics::K * (terms as f64).sqrt() * T::EPS * scale.max(1.0) * k as f64
    };
    // Reconstruction, orthonormal Q, upper-trapezoidal R.
    for j in 0..n {
        for i in 0..m {
            let got: Complex64 = (0..k).map(|l| q[i + m * l] * r[l + k * j]).sum();
            assert!((got - block.a[i + m * j]).norm() <= tol(m.max(n), scale));
        }
    }
    for a in 0..k {
        for b in 0..k {
            let identity = if a == b { 1.0 } else { 0.0 };
            let qq: Complex64 = (0..m).map(|i| q[i + m * a].conj() * q[i + m * b]).sum();
            assert!((qq - identity).norm() <= tol(m, 1.0));
        }
    }
    for j in 0..n {
        for i in (j + 1)..k {
            assert_eq!(r[i + k * j], Complex64::new(0.0, 0.0), "R[{i}, {j}]");
        }
    }
    if !block.full_rank {
        return;
    }
    // Uniqueness: Q = Q0 conj(P) and R = P R0 for the unit phases
    // P = diag(R[i, i] / |R[i, i]|).
    for i in 0..k {
        let pivot = r[i + k * i];
        assert!((pivot.norm() - block.r0[i + k * i].re).abs() <= tol(m, scale));
        let phase = pivot / pivot.norm();
        for j in 0..n {
            let want = phase * block.r0[i + k * j];
            assert!((r[i + k * j] - want).norm() <= tol(m, scale), "R[{i}, {j}]");
        }
        for row in 0..m {
            let want = block.q0[row + m * i] * phase.conj();
            assert!(
                (q[row + m * i] - want).norm() <= tol(m, scale),
                "Q[{row}, {i}]"
            );
        }
    }
}

fn inputs<'a, T: Scalar>(data: &'a [Vec<T>], layouts: &'a [Layout]) -> Vec<DenseRead<'a>> {
    data.iter()
        .zip(layouts)
        .map(|(d, l)| T::read(col_major(d, l)))
        .collect()
}

/// A column-major matrix's shape and strides.
type Layout = ([usize; 2], [usize; 2]);

fn fixture<T: Scalar>(blocks: &[Block]) -> (Vec<Vec<T>>, Vec<Layout>) {
    (
        blocks
            .iter()
            .map(|b| b.a.iter().map(|&x| T::make(x)).collect())
            .collect(),
        blocks
            .iter()
            .map(|b| ([b.rows, b.cols], [1usize, b.rows]))
            .collect(),
    )
}

fn oracle_matches<T: Scalar>() {
    let blocks = blocks(T::COMPLEX);
    let (data, layouts) = fixture::<T>(&blocks);
    let inputs = inputs(&data, &layouts);
    for (threads, policy) in policies() {
        let mut engine = engine(threads, policy);
        let outputs = DefaultDenseExecutor::new()
            .factorize_batch(&mut engine, DenseFactorization::Qr, &inputs)
            .unwrap();
        assert_eq!(outputs.len(), blocks.len());
        for (block, factors) in blocks.iter().zip(&outputs) {
            check_block::<T>(block, factors);
        }
    }
}

/// Purpose: every scalar's compact QR reconstructs the constructed matrix
/// with orthonormal Q and upper-trapezoidal R, equal to the construction up
/// to column phases, under every engine width and policy.
#[test]
fn factors_match_the_constructed_qr_f32() {
    oracle_matches::<f32>();
}

/// Purpose: as above, for `f64`.
#[test]
fn factors_match_the_constructed_qr_f64() {
    oracle_matches::<f64>();
}

/// Purpose: as above, for `Complex32`.
#[test]
fn factors_match_the_constructed_qr_c32() {
    oracle_matches::<Complex32>();
}

/// Purpose: as above, for `Complex64`.
#[test]
fn factors_match_the_constructed_qr_c64() {
    oracle_matches::<Complex64>();
}

/// Purpose: the batch and the per-matrix entry factor each block the same
/// way (same shapes, factors equal within the numerical tolerance), so the
/// batch is a drop-in for per-matrix calls.
#[test]
fn factors_match_the_per_matrix_factors() {
    let blocks = blocks(true);
    let (data, layouts) = fixture::<Complex64>(&blocks);
    let inputs = inputs(&data, &layouts);
    let mut engine = engine(1, BatchPolicy::default());
    let batch = DefaultDenseExecutor::new()
        .factorize_batch(&mut engine, DenseFactorization::Qr, &inputs)
        .unwrap();
    for ((factors, &input), block) in batch.iter().zip(&inputs).zip(&blocks) {
        let want = DefaultDenseExecutor::new().qr(&mut engine, input).unwrap();
        let scale = block.a.iter().map(|x| x.norm()).fold(1.0, f64::max);
        let bound = numerics::K * (block.rows.max(block.cols) as f64).sqrt() * f64::EPSILON * scale;
        for (got, want) in factors.iter().zip(&want) {
            assert_eq!(got.shape(), want.shape());
            for (g, w) in wide(got).iter().zip(&wide(want)) {
                assert!((g - w).norm() <= bound, "{g} against {w}");
            }
        }
    }
}

/// The variant and op label of a dense error: what the batch must share with
/// the per-matrix entry (the diagnostic text is the provider's).
fn kind(error: &DenseError) -> (std::mem::Discriminant<DenseError>, Option<&'static str>) {
    let op = match error {
        DenseError::Unsupported { op, .. }
        | DenseError::Backend { op, .. }
        | DenseError::NumericalFailure { op, .. }
        | DenseError::DTypeMismatch { op, .. }
        | DenseError::ShapeMismatch { op, .. } => Some(*op),
        _ => None,
    };
    (std::mem::discriminant(error), op)
}

/// The reason text of a stringly error variant.
fn reason(error: &DenseError) -> &str {
    match error {
        DenseError::Unsupported { message, .. }
        | DenseError::Backend { message, .. }
        | DenseError::NumericalFailure { message, .. } => message,
        other => panic!("not a stringly variant: {other:?}"),
    }
}

/// The batch's error and the per-matrix entry's error on `inputs[failing]`.
fn batch_and_per_call_errors(
    engine: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
    failing: usize,
) -> (DenseError, DenseError) {
    let batch = DefaultDenseExecutor::new()
        .factorize_batch(engine, DenseFactorization::Qr, inputs)
        .unwrap_err();
    let per_call = DefaultDenseExecutor::new()
        .qr(engine, inputs[failing])
        .unwrap_err();
    (batch, per_call)
}

/// Purpose: the batch reports the lowest failing input's error, equal to the
/// per-matrix entry's error for that input, whether it fails before the
/// engine (dtype) or inside it (rank); a pre-entry refusal stops admission,
/// so a higher rank failure is never reported over it.
#[test]
fn lowest_failing_input_reports_the_per_matrix_error() {
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ints = vec![1_i32, 2, 3, 4];
    let square = ([2usize, 2], [1usize, 2]);
    let ok = DenseRead::F64(col_major(&good, &square));
    let integer = DenseRead::I32(DenseView::new(&ints, &square.0, &square.1, 0).unwrap());
    let rank_one = DenseRead::F64(DenseView::new(&good, &[4], &[1], 0).unwrap());
    let cases: [(&[DenseRead<'_>], usize); 3] = [
        (&[ok, rank_one, ok, integer], 1),
        (&[ok, integer, rank_one], 1),
        (&[rank_one, ok, ok, ok, ok, ok], 0),
    ];
    for (threads, policy) in policies() {
        let mut engine = engine(threads, policy);
        for (inputs, failing) in cases {
            let (batch, per_call) = batch_and_per_call_errors(&mut engine, inputs, failing);
            assert_eq!(
                kind(&batch),
                kind(&per_call),
                "{threads} threads, {policy:?}"
            );
        }
        let (rank, _) = batch_and_per_call_errors(&mut engine, cases[0].0, 1);
        assert!(matches!(rank, DenseError::Backend { op: "qr_read", .. }));
        assert!(reason(&rank).contains("rank"), "{rank:?}");
        let (dtype, _) = batch_and_per_call_errors(&mut engine, cases[1].0, 1);
        assert!(matches!(
            dtype,
            DenseError::Unsupported { op: "qr_read", .. }
        ));
    }
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
    let batch = DefaultDenseExecutor::new()
        .factorize_batch(&mut engine, DenseFactorization::Qr, &inputs)
        .unwrap();
    for (factors, &input) in batch.iter().zip(&inputs) {
        let want = DefaultDenseExecutor::new().qr(&mut engine, input).unwrap();
        let shapes = |f: &[DenseTensor]| f.iter().map(|t| t.shape().to_vec()).collect::<Vec<_>>();
        assert_eq!(shapes(factors), shapes(&want));
    }
}

/// Purpose: after a lane panic broke the engine, a QR batch reports the
/// per-matrix entry's refusal, after the same pre-entry checks.
#[test]
fn broken_engine_refuses_like_the_per_matrix_entry() {
    let mut engine = engine(4, BatchPolicy::new(BatchStrategy::Outer));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        driver::drive(
            &mut engine,
            &PanicsAt(1),
            &mut [(); 3],
            &mut LaneScratch::default(),
        )
    }));
    assert!(unwound.is_err());
    assert!(engine.is_broken());
    let good = vec![1.0_f64, 2.0, 3.0, 4.0];
    let ints = vec![1_i32, 2, 3, 4];
    let square = ([2usize, 2], [1usize, 2]);
    let ok = DenseRead::F64(col_major(&good, &square));
    let integer = DenseRead::I32(DenseView::new(&ints, &square.0, &square.1, 0).unwrap());
    // A supported first input reaches the engine's refusal; an unsupported
    // one is refused before the engine is entered.
    let (broken, per_call) = batch_and_per_call_errors(&mut engine, &[ok, integer], 0);
    assert_eq!(kind(&broken), kind(&per_call));
    assert!(reason(&broken).contains("broken"), "{broken:?}");
    let (dtype, per_call) = batch_and_per_call_errors(&mut engine, &[integer, ok], 0);
    assert_eq!(kind(&dtype), kind(&per_call));
    assert!(matches!(
        dtype,
        DenseError::Unsupported { op: "qr_read", .. }
    ));
    assert!(!reason(&dtype).contains("broken"), "{dtype:?}");
}
