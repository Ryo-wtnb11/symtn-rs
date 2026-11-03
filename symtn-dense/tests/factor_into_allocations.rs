//! The native destination batch allocates its provider workspace and
//! scratch per lane, not per block: with the destinations prepared outside
//! the measured window, a batch of twice the blocks at the same engine width
//! costs exactly as many allocation calls, for every scalar and every op
//! with a destination form.
//!
//! Scope: one serial engine (one lane), so every allocation is on the
//! measured thread.

#![cfg(all(
    feature = "cpu-faer",
    not(feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    CpuEngine, DefaultDenseExecutor, DenseExecutor, DenseFactorization, DenseRead, DenseView,
    DenseViewMut, DenseWrite, FactorDest, FactorPlacement,
};

use crate::counting_alloc;

trait Elem: Copy + Default + std::ops::Add<Output = Self> + 'static {
    type Real: Copy + Default + 'static;
    fn make(value: f64) -> Self;
    fn conj(self) -> Self;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn write_real(view: DenseViewMut<'_, Self::Real>) -> DenseWrite<'_>;
}

macro_rules! elem {
    ($ty:ty, $real:ty, $variant:ident, $real_variant:ident, |$v:ident| $make:expr, |$c:ident| $conj:expr) => {
        impl Elem for $ty {
            type Real = $real;
            fn make($v: f64) -> Self {
                $make
            }
            fn conj(self) -> Self {
                let $c = self;
                $conj
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn write_real(view: DenseViewMut<'_, Self::Real>) -> DenseWrite<'_> {
                DenseWrite::$real_variant(view)
            }
        }
    };
}

elem!(f32, f32, F32, F32, |v| v as f32, |c| c);
elem!(f64, f64, F64, F64, |v| v, |c| c);
elem!(
    Complex32,
    f32,
    C32,
    F32,
    |v| Complex32::new(v as f32, 0.5),
    |c| Complex32::conj(&c)
);
elem!(Complex64, f64, C64, F64, |v| Complex64::new(v, 0.5), |c| {
    Complex64::conj(&c)
});

const N: usize = 6;

/// The allocation calls of one warm native batch of `blocks` Hermitian
/// `N x N` inputs (eigh) or `N x N` inputs (QR, SVD; adjoint placement for
/// QR so its typed scratch is exercised).
fn batch_calls<T: Elem>(op: DenseFactorization, blocks: usize) -> u64 {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    let input: Vec<T> = (0..N * N)
        .map(|index| {
            let (row, col) = (index % N, index / N);
            T::make(((row * col + row + col) % 7) as f64 + if row == col { 10.0 } else { 0.0 })
        })
        .collect();
    let input = match op {
        // `X + X^H` keeps the fixture Hermitian for every scalar.
        DenseFactorization::Eigh => (0..N * N)
            .map(|index| {
                let (row, col) = (index % N, index / N);
                input[index] + input[col + N * row].conj()
            })
            .collect(),
        _ => input,
    };
    let shape = [N, N];
    let strides = [1, N];
    let vector = ([N], [1]);
    let inputs: Vec<DenseRead<'_>> = (0..blocks)
        .map(|_| T::read(DenseView::new(&input, &shape, &strides, 0).unwrap()))
        .collect();
    let mut factors: Vec<[Vec<T>; 2]> = (0..blocks)
        .map(|_| [vec![T::default(); N * N], vec![T::default(); N * N]])
        .collect();
    let mut spectra: Vec<Vec<T::Real>> = (0..blocks).map(|_| vec![T::Real::default(); N]).collect();
    let placement = match op {
        DenseFactorization::Qr => FactorPlacement::Adjoint,
        _ => FactorPlacement::AsComputed,
    };
    let mut run = |factors: &mut Vec<[Vec<T>; 2]>, spectra: &mut Vec<Vec<T::Real>>| {
        let mut dests: Vec<FactorDest<'_>> = factors
            .iter_mut()
            .zip(spectra.iter_mut())
            .map(|([a, b], s)| {
                let a = T::write(DenseViewMut::new(a, &shape, &strides, 0).unwrap());
                let b = T::write(DenseViewMut::new(b, &shape, &strides, 0).unwrap());
                let s = T::write_real(DenseViewMut::new(s, &vector.0, &vector.1, 0).unwrap());
                let outputs = match op {
                    DenseFactorization::Svd => [Some(a), Some(s), Some(b)],
                    DenseFactorization::Eigh => [Some(s), Some(a), None],
                    _ => [Some(a), Some(b), None],
                };
                FactorDest::new(outputs, placement)
            })
            .collect();
        let ((), allocs) = counting_alloc::measure(|| {
            executor
                .factorize_batch_into(engine, op, &inputs, &mut dests, &mut |_, _, outcome| {
                    outcome.map_err(|failure| failure.into_error())
                })
                .unwrap()
        });
        allocs.calls
    };
    run(&mut factors, &mut spectra);
    run(&mut factors, &mut spectra)
}

fn no_growth<T: Elem>() {
    for op in [
        DenseFactorization::Qr,
        DenseFactorization::Svd,
        DenseFactorization::Eigh,
    ] {
        counting_alloc::assert_same_calls(
            &format!("{op:?} blocks"),
            batch_calls::<T>(op, 4),
            batch_calls::<T>(op, 8),
        );
    }
}

#[test]
fn native_destination_batch_allocation_calls_do_not_grow_with_blocks() {
    let _serial = counting_alloc::serial();
    no_growth::<f32>();
    no_growth::<f64>();
    no_growth::<Complex32>();
    no_growth::<Complex64>();
}
