//! The process-global geometry-plan cache behind the native block GEMM
//! kernel, through the public batch entry and the structure-cache registry:
//! it is registered, cleared and sized like every other structure cache; a
//! warm call is a hit with the cold call's bits; a disabled cache gives the
//! same bits; and the bytes it charges for an entry cover the live
//! allocations that entry keeps, for every scalar, op lowering, plan strategy
//! and small or large extents.
//!
//! Scope: the native provider (no BLAS), one serial engine, so every
//! allocation is on the measured thread.

#![cfg(not(feature = "cpu-blas-core"))]

use num_complex::{Complex32, Complex64};
use symtn_core::{
    clear_structure_caches, set_structure_cache_byte_budget, structure_cache_info,
    structure_cache_infos, StructureCacheInfo, StructureCacheKind,
};
use symtn_dense::{
    CpuEngine, DefaultDenseExecutor, DenseExecutor, DenseGemmBatchJob, DenseRead, DenseScalar,
    DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

use crate::counting_alloc;

const KIND: StructureCacheKind = StructureCacheKind::GemmPlans;

fn info() -> StructureCacheInfo {
    structure_cache_info(KIND)
}

trait Scalar: Copy + 'static {
    fn make(re: f64, im: f64) -> Self;
    fn bits(self) -> [u64; 2];
    fn dense(re: f64, im: f64) -> DenseScalar;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
}

macro_rules! scalar {
    ($ty:ty, $variant:ident, |$re:ident, $im:ident| $make:expr, |$s:ident| $bits:expr) => {
        impl Scalar for $ty {
            fn make($re: f64, $im: f64) -> Self {
                $make
            }
            fn bits(self) -> [u64; 2] {
                let $s = self;
                $bits
            }
            fn dense(re: f64, im: f64) -> DenseScalar {
                DenseScalar::$variant(Self::make(re, im))
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
scalar!(f32, F32, |re, _im| re as f32, |s| [
    u64::from(s.to_bits()),
    0
]);
scalar!(f64, F64, |re, _im| re, |s| [s.to_bits(), 0]);
scalar!(
    Complex32,
    C32,
    |re, im| Complex32::new(re as f32, im as f32),
    |s| [u64::from(s.re.to_bits()), u64::from(s.im.to_bits())]
);
scalar!(Complex64, C64, |re, im| Complex64::new(re, im), |s| [
    s.re.to_bits(),
    s.im.to_bits()
]);

/// Two disjoint jobs of one geometry, submitted as two runs so the batch
/// takes the block kernel rather than the strided entry.
struct Batch<T> {
    jobs: [DenseGemmBatchJob; 2],
    lhs: Vec<T>,
    rhs: Vec<T>,
    out_len: usize,
    ops: [MatrixOp; 2],
}

impl<T: Scalar> Batch<T> {
    fn new((m, k, n): (usize, usize, usize), ops: [MatrixOp; 2]) -> Self {
        let job = |i: usize| DenseGemmBatchJob {
            dst_offset: i * (m * n + 1),
            lhs_offset: i * (m * k + 2),
            rhs_offset: i * (k * n + 3),
            rows: m,
            contracted: k,
            cols: n,
        };
        let data = |len: usize, salt: f64| {
            (0..len)
                .map(|i| {
                    let x = i as f64 + salt;
                    T::make((0.37 * x).sin(), (0.11 * x).cos())
                })
                .collect()
        };
        Self {
            jobs: [job(0), job(1)],
            lhs: data(2 * (m * k + 2), 0.5),
            rhs: data(2 * (k * n + 3), 7.0),
            out_len: 2 * (m * n + 1),
            ops,
        }
    }

    /// One call with `beta == 0` (so repeats compute the same output), the
    /// output's bits.
    fn call(&self, executor: &mut DefaultDenseExecutor, engine: &mut CpuEngine) -> Vec<[u64; 2]> {
        let mut out = vec![T::make(f64::NAN, f64::NAN); self.out_len];
        let shapes = [[self.out_len], [self.lhs.len()], [self.rhs.len()]];
        executor
            .matmul_batch_axpby_with_ops_into(
                engine,
                T::write(DenseViewMut::new(&mut out, &shapes[0], &[1], 0).unwrap()),
                T::read(DenseView::new(&self.lhs, &shapes[1], &[1], 0).unwrap()),
                T::read(DenseView::new(&self.rhs, &shapes[2], &[1], 0).unwrap()),
                &self.jobs,
                &[1, 1],
                self.ops[0],
                self.ops[1],
                T::dense(0.75, -0.5),
                T::dense(0.0, 0.0),
            )
            .unwrap();
        out.into_iter().map(Scalar::bits).collect()
    }
}

/// Small (Faer), unit, zero-`k`, zero-`m`, and large extents (Faer for real
/// and `Complex32`; Packed for `Complex64` above the Faer volume limit).
const SHAPES: [(usize, usize, usize); 6] = [
    (3, 4, 2),
    (1, 1, 1),
    (3, 0, 2),
    (0, 3, 2),
    (64, 64, 64),
    (72, 48, 56),
];

const OPS: [MatrixOp; 3] = [MatrixOp::Identity, MatrixOp::Transpose, MatrixOp::Adjoint];

/// Heap bytes of a call's returned bits, live after a measured call.
fn returned(bits: &Vec<[u64; 2]>) -> i64 {
    (bits.capacity() * std::mem::size_of::<[u64; 2]>()) as i64
}

/// Charge against the live bytes of the entry one cold call admits, then
/// the warm hit and the disabled-cache result, for every shape and op pair.
fn check<T: Scalar>(name: &str) {
    let mut executor = DefaultDenseExecutor::new();
    let mut engine = CpuEngine::serial();
    let budget = info().byte_budget();
    for shape in SHAPES {
        for a in OPS {
            for b in OPS {
                let what = format!("{name} {shape:?} {a:?} {b:?}");
                let batch = Batch::<T>::new(shape, [a, b]);
                // Warms the executor, the engine and the cache's own storage;
                // the clear then drops only the entry.
                let want = batch.call(&mut executor, &mut engine);
                clear_structure_caches();
                let (cold, allocs) =
                    counting_alloc::measure(|| batch.call(&mut executor, &mut engine));
                let admitted = info();
                assert_eq!(cold, want, "{what}: cold");
                assert_eq!(
                    (admitted.entries(), admitted.misses()),
                    (1, 1),
                    "{what}: the two jobs share one entry"
                );

                // What: a warm call hits with the cold bits and keeps nothing.
                let (warm_bits, warm_allocs) =
                    counting_alloc::measure(|| batch.call(&mut executor, &mut engine));
                assert_eq!(warm_bits, want, "{what}: warm");
                let warm = info();
                assert_eq!((warm.misses(), warm.hits()), (1, 1), "{what}: warm hit");
                assert_eq!(
                    warm_allocs.live_bytes,
                    returned(&warm_bits),
                    "{what}: warm retention beyond the returned bits"
                );

                // What: the charge is a conservative bound of what the entry
                // keeps alive, which dropping it releases, and which the
                // cold call left live besides its returned bits.
                let ((), dropped) = counting_alloc::measure(clear_structure_caches);
                let kept = dropped.freed_bytes;
                assert!(kept > 0, "{what}: the entry keeps its plan");
                assert_eq!(
                    kept as i64,
                    allocs.live_bytes - returned(&cold),
                    "{what}: kept by the cold call"
                );
                assert!(
                    admitted.charged_bytes() >= kept,
                    "{what}: charge {} B below kept {kept} B",
                    admitted.charged_bytes(),
                );
                eprintln!(
                    "{what}: entry {kept} B, charge {} B",
                    admitted.charged_bytes()
                );

                // What: a disabled cache builds per call with the same bits.
                set_structure_cache_byte_budget(KIND, 0);
                assert_eq!(info().entries(), 0, "{what}: evicted");
                let rejected = info().rejections();
                assert_eq!(
                    batch.call(&mut executor, &mut engine),
                    want,
                    "{what}: disabled"
                );
                assert_eq!(info().rejections(), rejected + 1, "{what}: rejected");
                set_structure_cache_byte_budget(KIND, budget);
            }
        }
    }
}

#[test]
fn gemm_plan_charge_covers_live_bytes_and_hits_replay_the_cold_bits() {
    let _serial = counting_alloc::serial();
    // What: the kind is registered with the other structure caches.
    assert_eq!(structure_cache_infos().len(), StructureCacheKind::ALL.len());
    assert!(structure_cache_infos().iter().any(|i| i.kind() == KIND));
    check::<f32>("f32");
    check::<f64>("f64");
    check::<Complex32>("c32");
    check::<Complex64>("c64");
    clear_structure_caches();
    assert_eq!((info().entries(), info().charged_bytes()), (0, 0));
}
