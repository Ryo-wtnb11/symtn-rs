//! `DenseExecutor::factorize_batch_into`: whole-batch destination
//! validation before any solve; the compatibility default's delivery,
//! failure provenance and owned dispatch; the native lanes' destination
//! writes against the native owned batch, placed, their full QR and null
//! basis laws on strided transposed reads and their full SVD laws; and the
//! destination-only
//! polar decompositions, Hermitian exponential and inverse against their
//! laws and closed forms.

use super::*;
use crate::{FactorDest, FactorFailure, FactorPlacement};

/// A scalar of the destination tests: its dtype-erased views and an exact
/// widening for bit comparisons.
trait Elem: Copy + std::fmt::Debug + 'static {
    type Real: Elem;
    fn make(value: Complex64) -> Self;
    fn to_c64(self) -> Complex64;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn slice(tensor: &DenseTensor) -> &[Self];
}

macro_rules! elem {
    ($ty:ty, $real:ty, $variant:ident, $slice:ident, |$v:ident| $make:expr, |$w:ident| $widen:expr) => {
        impl Elem for $ty {
            type Real = $real;
            fn make($v: Complex64) -> Self {
                $make
            }
            fn to_c64(self) -> Complex64 {
                let $w = self;
                $widen
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn slice(tensor: &DenseTensor) -> &[Self] {
                tensor.$slice().unwrap()
            }
        }
    };
}

elem!(f32, f32, F32, as_f32_slice, |v| v.re as f32, |w| {
    Complex64::new(w as f64, 0.0)
});
elem!(f64, f64, F64, as_f64_slice, |v| v.re, |w| Complex64::new(
    w, 0.0
));
elem!(
    Complex32,
    f32,
    C32,
    as_c32_slice,
    |v| Complex32::new(v.re as f32, v.im as f32),
    |w| Complex64::new(w.re as f64, w.im as f64)
);
elem!(Complex64, f64, C64, as_c64_slice, |v| v, |w| w);

/// Column-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut step = 1;
    shape
        .iter()
        .map(|&dim| {
            let stride = step;
            step *= dim;
            stride
        })
        .collect()
}

/// One input's owned destination buffers and their layouts.
struct Buffers<T: Elem> {
    factors: [Vec<T>; 2],
    spectrum: Vec<T::Real>,
    shapes: [Vec<usize>; 3],
    strides: [Vec<usize>; 3],
}

/// The destination shapes of `op` on an input of `shape` under `placement`.
fn dest_shapes(
    op: DenseFactorization,
    shape: &[usize],
    placement: FactorPlacement,
) -> [Vec<usize>; 3] {
    let (m, n, batch) = (shape[0], shape[1], &shape[2..]);
    let k = m.min(n);
    let with = |head: &[usize]| [head, batch].concat();
    let adjoint = placement == FactorPlacement::Adjoint;
    match op {
        DenseFactorization::Qr if adjoint => [with(&[n, k]), with(&[k, m]), vec![]],
        DenseFactorization::Qr | DenseFactorization::Lq => [with(&[m, k]), with(&[k, n]), vec![]],
        DenseFactorization::Svd if adjoint => [with(&[n, k]), with(&[k]), with(&[k, m])],
        DenseFactorization::Svd => [with(&[m, k]), with(&[k]), with(&[k, n])],
        DenseFactorization::SvdFull if adjoint => [with(&[n, n]), with(&[k]), with(&[m, m])],
        DenseFactorization::SvdFull => [with(&[m, m]), with(&[k]), with(&[n, n])],
        DenseFactorization::Eigh => [with(&[m]), with(&[m, m]), vec![]],
        DenseFactorization::QrFull => [with(&[m, m]), with(&[m, n]), vec![]],
        DenseFactorization::QrNull => [with(&[m, m - k]), vec![], vec![]],
        DenseFactorization::LeftPolar => [with(&[m, n]), with(&[n, n]), vec![]],
        DenseFactorization::RightPolar => [with(&[m, m]), with(&[m, n]), vec![]],
        DenseFactorization::ExpViaEigh | DenseFactorization::Inv => [with(&[m, m]), vec![], vec![]],
        _ => unreachable!(),
    }
}

impl<T: Elem> Buffers<T> {
    /// Sentinel-filled buffers of the shapes `shapes`, in the op's output
    /// order (the spectrum is output 1 for SVD, 0 for eigh).
    fn new(op: DenseFactorization, shapes: [Vec<usize>; 3], sentinel: f64) -> Self {
        let len = |shape: &Vec<usize>| shape.iter().product::<usize>();
        let fill = Complex64::new(sentinel, 0.0);
        let (factor_shapes, spectrum_shape): ([&Vec<usize>; 2], &Vec<usize>) = match op {
            DenseFactorization::Svd | DenseFactorization::SvdFull => {
                ([&shapes[0], &shapes[2]], &shapes[1])
            }
            DenseFactorization::Eigh => ([&shapes[1], &shapes[2]], &shapes[0]),
            _ => ([&shapes[0], &shapes[1]], &shapes[2]),
        };
        Self {
            factors: factor_shapes.map(|shape| vec![T::make(fill); len(shape)]),
            spectrum: vec![
                T::Real::make(fill);
                if matches!(
                    op,
                    DenseFactorization::Qr
                        | DenseFactorization::Lq
                        | DenseFactorization::QrFull
                        | DenseFactorization::QrNull
                        | DenseFactorization::LeftPolar
                        | DenseFactorization::RightPolar
                        | DenseFactorization::ExpViaEigh
                        | DenseFactorization::Inv
                ) {
                    0
                } else {
                    len(spectrum_shape)
                }
            ],
            strides: [
                strides(&shapes[0]),
                strides(&shapes[1]),
                strides(&shapes[2]),
            ],
            shapes,
        }
    }

    fn dest(&mut self, op: DenseFactorization, placement: FactorPlacement) -> FactorDest<'_> {
        let [a, b] = &mut self.factors;
        let view = |data, index: usize| -> DenseViewMut<'_, _> {
            DenseViewMut::new(data, &self.shapes[index], &self.strides[index], 0).unwrap()
        };
        let outputs = match op {
            DenseFactorization::Svd | DenseFactorization::SvdFull => [
                Some(T::write(view(a, 0))),
                Some(T::Real::write(
                    DenseViewMut::new(&mut self.spectrum, &self.shapes[1], &self.strides[1], 0)
                        .unwrap(),
                )),
                Some(T::write(view(b, 2))),
            ],
            DenseFactorization::Eigh => [
                Some(T::Real::write(
                    DenseViewMut::new(&mut self.spectrum, &self.shapes[0], &self.strides[0], 0)
                        .unwrap(),
                )),
                Some(T::write(view(a, 1))),
                None,
            ],
            DenseFactorization::QrNull
            | DenseFactorization::ExpViaEigh
            | DenseFactorization::Inv => [Some(T::write(view(a, 0))), None, None],
            _ => [Some(T::write(view(a, 0))), Some(T::write(view(b, 1))), None],
        };
        FactorDest::new(outputs, placement)
    }

    /// Every element, widened, in output order.
    fn widened(&self, op: DenseFactorization) -> Vec<Vec<Complex64>> {
        let factor = |data: &Vec<T>| data.iter().map(|&value| value.to_c64()).collect();
        let spectrum = self.spectrum.iter().map(|&value| value.to_c64()).collect();
        match op {
            DenseFactorization::Svd | DenseFactorization::SvdFull => {
                vec![factor(&self.factors[0]), spectrum, factor(&self.factors[1])]
            }
            DenseFactorization::Eigh => vec![spectrum, factor(&self.factors[0])],
            DenseFactorization::QrNull
            | DenseFactorization::ExpViaEigh
            | DenseFactorization::Inv => {
                vec![factor(&self.factors[0])]
            }
            _ => vec![factor(&self.factors[0]), factor(&self.factors[1])],
        }
    }
}

/// The conjugate transpose of each `rows x cols` item of `data`, written
/// out entry by entry as the oracle of the adjoint placement.
fn adjoint_items(data: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::default(); data.len()];
    for item in 0..data.len() / (rows * cols).max(1) {
        let base = item * rows * cols;
        for row in 0..rows {
            for col in 0..cols {
                out[base + col + cols * row] = data[base + row + rows * col].conj();
            }
        }
    }
    out
}

/// The owned outputs placed as `placement` places them, widened.
fn placed(
    op: DenseFactorization,
    owned: &[DenseTensor],
    placement: FactorPlacement,
) -> Vec<Vec<Complex64>> {
    let widen = |tensor: &DenseTensor| -> Vec<Complex64> {
        match tensor.dtype() {
            DenseDType::F32 => f32::slice(tensor).iter().map(|v| v.to_c64()).collect(),
            DenseDType::F64 => f64::slice(tensor).iter().map(|v| v.to_c64()).collect(),
            DenseDType::C32 => Complex32::slice(tensor)
                .iter()
                .map(|v| v.to_c64())
                .collect(),
            _ => Complex64::slice(tensor).to_vec(),
        }
    };
    let mut outputs: Vec<_> = owned.iter().map(widen).collect();
    if placement == FactorPlacement::Adjoint {
        let last = outputs.len() - 1;
        for (index, tensor) in owned.iter().enumerate() {
            if matches!(op, DenseFactorization::Svd | DenseFactorization::SvdFull) && index == 1 {
                continue;
            }
            let shape = tensor.shape();
            outputs[index] = adjoint_items(&outputs[index], shape[0], shape[1]);
        }
        outputs.swap(0, last);
    }
    outputs
}

/// Deterministic entries of an `m x n x batch` input; with `hermitian`,
/// each square item is `X + X^H`.
fn input_data<T: Elem>(shape: &[usize], hermitian: bool, seed: usize) -> Vec<T> {
    let (m, n) = (shape[0], shape[1]);
    let item = (m * n).max(1);
    let raw = |index: usize| {
        let x = ((index * 7 + seed * 13) % 17) as f64 / 17.0 - 0.4;
        let y = ((index * 5 + seed * 3) % 11) as f64 / 11.0 - 0.5;
        Complex64::new(x, y)
    };
    (0..shape.iter().product::<usize>())
        .map(|index| {
            if !hermitian {
                return T::make(raw(index));
            }
            let (base, local) = (index - index % item, index % item);
            let (row, col) = (local % m, local / m);
            T::make(raw(index) + raw(base + col + m * row).conj())
        })
        .collect()
}

#[cfg(all(feature = "cpu-faer", not(feature = "cpu-blas-core")))]
mod native {
    use super::*;
    use crate::tests::block_svd::{engine, policies};

    /// Purpose: on the native provider, each input's destinations receive
    /// exactly the outputs of the native owned batch (bit for bit, since
    /// both run the same provider item on the same lane width), and under
    /// `Adjoint` exactly their conjugate transposes in reversed factor
    /// order with the spectrum unchanged; for every scalar, op, engine width
    /// and batch policy, over unequal rectangular blocks, a two-item input
    /// and empty extents. The owned batch is checked against independent
    /// constructions in `block_svd` and `block_qr`, and owned LQ, full QR
    /// and the QR null basis by their properties on a complex input in
    /// `executor`, so a destination LQ whose transpose conjugated fails here
    /// for complex scalars.
    fn destinations_match_owned<T: Elem>() {
        let shapes: &[&[usize]] = &[&[5, 3], &[2, 4], &[3, 2, 2], &[4, 4], &[2, 0], &[3, 2, 0]];
        let square: &[&[usize]] = &[&[4, 4], &[2, 2], &[3, 3, 2], &[1, 1], &[0, 0], &[2, 2, 0]];
        let cases = [
            (DenseFactorization::Qr, FactorPlacement::AsComputed),
            (DenseFactorization::Qr, FactorPlacement::Adjoint),
            (DenseFactorization::Lq, FactorPlacement::AsComputed),
            (DenseFactorization::QrFull, FactorPlacement::AsComputed),
            (DenseFactorization::QrNull, FactorPlacement::AsComputed),
            (DenseFactorization::Svd, FactorPlacement::AsComputed),
            (DenseFactorization::Svd, FactorPlacement::Adjoint),
            (DenseFactorization::SvdFull, FactorPlacement::AsComputed),
            (DenseFactorization::SvdFull, FactorPlacement::Adjoint),
            (DenseFactorization::Eigh, FactorPlacement::AsComputed),
        ];
        for (threads, policy) in policies() {
            let mut engine = engine(threads, policy);
            for (op, placement) in cases {
                let shapes = if op == DenseFactorization::Eigh {
                    square
                } else {
                    shapes
                };
                let data: Vec<Vec<T>> = shapes
                    .iter()
                    .enumerate()
                    .map(|(seed, shape)| {
                        input_data::<T>(shape, op == DenseFactorization::Eigh, seed)
                    })
                    .collect();
                let layouts: Vec<_> = shapes.iter().map(|shape| strides(shape)).collect();
                let inputs: Vec<_> = data
                    .iter()
                    .zip(shapes)
                    .zip(&layouts)
                    .map(|((data, shape), strides)| {
                        T::read(DenseView::new(data, shape, strides, 0).unwrap())
                    })
                    .collect();
                let owned =
                    factorize_all(&mut DefaultDenseExecutor::new(), &mut engine, op, &inputs)
                        .unwrap();
                let mut buffers: Vec<Buffers<T>> = shapes
                    .iter()
                    .map(|shape| Buffers::new(op, dest_shapes(op, shape, placement), 7.0))
                    .collect();
                let mut dests: Vec<_> = buffers.iter_mut().map(|b| b.dest(op, placement)).collect();
                let mut delivered = Vec::new();
                DefaultDenseExecutor::new()
                    .factorize_batch_into(
                        &mut engine,
                        op,
                        &inputs,
                        &mut dests,
                        &mut |index, _, outcome| {
                            delivered.push((index, outcome));
                            Ok(())
                        },
                    )
                    .unwrap();
                drop(dests);
                assert_eq!(
                    delivered,
                    (0..shapes.len()).map(|i| (i, Ok(()))).collect::<Vec<_>>()
                );
                for (index, buffers) in buffers.iter().enumerate() {
                    assert_eq!(
                        buffers.widened(op),
                        placed(op, &owned[index], placement),
                        "{op:?} {placement:?} input {index}, {threads} threads, {policy:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn native_destinations_match_the_owned_batch_f32() {
        destinations_match_owned::<f32>();
    }

    #[test]
    fn native_destinations_match_the_owned_batch_f64() {
        destinations_match_owned::<f64>();
    }

    #[test]
    fn native_destinations_match_the_owned_batch_c32() {
        destinations_match_owned::<Complex32>();
    }

    #[test]
    fn native_destinations_match_the_owned_batch_c64() {
        destinations_match_owned::<Complex64>();
    }

    /// Purpose: full QR and the QR null basis of strided transposed reads
    /// (the extents and strides of a column-major block swapped, no copy),
    /// over unequal wide, tall and square blocks on separate native lanes
    /// (`Outer`, four threads): per block `Q R = X^T` with `Q` unitary and
    /// `R` upper trapezoidal, and `N` orthonormal with `N^H X^T = 0`.
    fn transposed_reads_hold_the_qr_laws<T: Elem>() {
        let tol = 1.0e-12;
        let mut engine = engine(
            4,
            tcontract_exec::BatchPolicy::new(tcontract_exec::BatchStrategy::Outer),
        );
        // Stored `X` shapes; each read is the `n x m` transpose.
        let stored: &[[usize; 2]] = &[[5, 3], [2, 4], [3, 3], [1, 4]];
        let data: Vec<Vec<T>> = stored
            .iter()
            .enumerate()
            .map(|(seed, shape)| input_data::<T>(shape, false, seed))
            .collect();
        let read_shapes: Vec<[usize; 2]> = stored.iter().map(|&[m, n]| [n, m]).collect();
        let read_strides: Vec<[usize; 2]> = stored.iter().map(|&[m, _]| [m, 1]).collect();
        let inputs: Vec<_> = data
            .iter()
            .zip(&read_shapes)
            .zip(&read_strides)
            .map(|((data, shape), strides)| {
                T::read(DenseView::new(data, shape, strides, 0).unwrap())
            })
            .collect();
        // `X^T` entry `(row, col)` is `X[col, row]`.
        let xt = |index: usize, row: usize, col: usize| {
            data[index][col + stored[index][0] * row].to_c64()
        };
        for op in [DenseFactorization::QrFull, DenseFactorization::QrNull] {
            let mut buffers: Vec<Buffers<T>> = read_shapes
                .iter()
                .map(|shape| {
                    Buffers::new(op, dest_shapes(op, shape, FactorPlacement::AsComputed), 7.0)
                })
                .collect();
            let mut dests: Vec<_> = buffers
                .iter_mut()
                .map(|b| b.dest(op, FactorPlacement::AsComputed))
                .collect();
            let mut delivered = Vec::new();
            DefaultDenseExecutor::new()
                .factorize_batch_into(
                    &mut engine,
                    op,
                    &inputs,
                    &mut dests,
                    &mut |index, _, outcome| {
                        delivered.push((index, outcome));
                        Ok(())
                    },
                )
                .unwrap();
            drop(dests);
            assert_eq!(
                delivered,
                (0..stored.len()).map(|i| (i, Ok(()))).collect::<Vec<_>>()
            );
            for (index, (buffers, &[rows, cols])) in buffers.iter().zip(&read_shapes).enumerate() {
                let outputs = buffers.widened(op);
                let gram = |a: &[Complex64], width: usize| {
                    for i in 0..width {
                        for j in 0..width {
                            let dot: Complex64 = (0..rows)
                                .map(|r| a[r + rows * i].conj() * a[r + rows * j])
                                .sum();
                            let want = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                            assert!(
                                (dot - want).norm() < tol,
                                "{op:?} block {index}: not orthonormal"
                            );
                        }
                    }
                };
                if op == DenseFactorization::QrFull {
                    let (q, r) = (&outputs[0], &outputs[1]);
                    gram(q, rows);
                    for row in 0..rows {
                        for col in 0..cols {
                            let qr: Complex64 = (0..rows)
                                .map(|j| q[row + rows * j] * r[j + rows * col])
                                .sum();
                            assert!(
                                (qr - xt(index, row, col)).norm() < tol,
                                "{op:?} block {index}: Q R != X^T"
                            );
                            if row > col {
                                assert_eq!(r[row + rows * col], Complex64::new(0.0, 0.0));
                            }
                        }
                    }
                } else {
                    let null = &outputs[0];
                    let width = rows - rows.min(cols);
                    assert_eq!(null.len(), rows * width);
                    gram(null, width);
                    for k in 0..width {
                        for col in 0..cols {
                            let dot: Complex64 = (0..rows)
                                .map(|r| null[r + rows * k].conj() * xt(index, r, col))
                                .sum();
                            assert!(dot.norm() < tol, "{op:?} block {index}: N^H X^T != 0");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn native_transposed_reads_hold_the_qr_laws_f64() {
        transposed_reads_hold_the_qr_laws::<f64>();
    }

    #[test]
    fn native_transposed_reads_hold_the_qr_laws_c64() {
        transposed_reads_hold_the_qr_laws::<Complex64>();
    }

    /// Purpose: the full SVD of unequal tall, wide and square blocks on
    /// separate native lanes (`Outer`, four threads): per block
    /// `U S Vh = X` with `U` (`m x m`) and `Vh` (`n x n`) unitary and `S`
    /// non-negative and non-increasing.
    fn full_svd_lanes_hold_the_svd_laws<T: Elem>() {
        let tol = 1.0e-12;
        let mut engine = engine(
            4,
            tcontract_exec::BatchPolicy::new(tcontract_exec::BatchStrategy::Outer),
        );
        let op = DenseFactorization::SvdFull;
        let shapes: &[[usize; 2]] = &[[5, 3], [2, 4], [3, 3], [1, 4]];
        let data: Vec<Vec<T>> = shapes
            .iter()
            .enumerate()
            .map(|(seed, shape)| input_data::<T>(shape, false, seed))
            .collect();
        let layouts: Vec<_> = shapes.iter().map(|shape| strides(shape)).collect();
        let inputs: Vec<_> = data
            .iter()
            .zip(shapes)
            .zip(&layouts)
            .map(|((data, shape), strides)| {
                T::read(DenseView::new(data, shape, strides, 0).unwrap())
            })
            .collect();
        let mut buffers: Vec<Buffers<T>> = shapes
            .iter()
            .map(|shape| Buffers::new(op, dest_shapes(op, shape, FactorPlacement::AsComputed), 7.0))
            .collect();
        let mut dests: Vec<_> = buffers
            .iter_mut()
            .map(|b| b.dest(op, FactorPlacement::AsComputed))
            .collect();
        let mut delivered = Vec::new();
        DefaultDenseExecutor::new()
            .factorize_batch_into(
                &mut engine,
                op,
                &inputs,
                &mut dests,
                &mut |index, _, outcome| {
                    delivered.push((index, outcome));
                    Ok(())
                },
            )
            .unwrap();
        drop(dests);
        assert_eq!(
            delivered,
            (0..shapes.len()).map(|i| (i, Ok(()))).collect::<Vec<_>>()
        );
        for (index, (buffers, &[m, n])) in buffers.iter().zip(shapes).enumerate() {
            let k = m.min(n);
            let outputs = buffers.widened(op);
            let (u, s, vh) = (&outputs[0], &outputs[1], &outputs[2]);
            assert_eq!(
                (u.len(), s.len(), vh.len()),
                (m * m, k, n * n),
                "block {index}: output sizes"
            );
            for i in 0..k {
                assert!(s[i].im == 0.0 && s[i].re >= 0.0, "block {index}: S sign");
                assert!(i == 0 || s[i - 1].re >= s[i].re, "block {index}: S order");
            }
            for i in 0..m {
                for j in 0..m {
                    let dot: Complex64 = (0..m).map(|r| u[r + m * i].conj() * u[r + m * j]).sum();
                    let want = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                    assert!((dot - want).norm() < tol, "block {index}: U not unitary");
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let dot: Complex64 = (0..n).map(|c| vh[i + n * c] * vh[j + n * c].conj()).sum();
                    let want = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                    assert!((dot - want).norm() < tol, "block {index}: Vh not unitary");
                }
            }
            for row in 0..m {
                for col in 0..n {
                    let usv: Complex64 = (0..k)
                        .map(|p| u[row + m * p] * s[p] * vh[p + n * col])
                        .sum();
                    let x = data[index][row + m * col].to_c64();
                    assert!((usv - x).norm() < tol, "block {index}: U S Vh != X");
                }
            }
        }
    }

    #[test]
    fn native_full_svd_lanes_hold_the_svd_laws_f64() {
        full_svd_lanes_hold_the_svd_laws::<f64>();
    }

    #[test]
    fn native_full_svd_lanes_hold_the_svd_laws_c64() {
        full_svd_lanes_hold_the_svd_laws::<Complex64>();
    }

    /// Runs one native inverse batch of the column-major square `blocks` on
    /// separate lanes (`Outer`, four threads) and returns each delivered
    /// outcome with the block's widened destination.
    fn inverse_batch<T: Elem>(
        blocks: &[Vec<T>],
        orders: &[usize],
    ) -> Vec<(usize, Result<(), FactorFailure>, Vec<Complex64>)> {
        let mut engine = engine(
            4,
            tcontract_exec::BatchPolicy::new(tcontract_exec::BatchStrategy::Outer),
        );
        let op = DenseFactorization::Inv;
        let shapes: Vec<[usize; 2]> = orders.iter().map(|&n| [n, n]).collect();
        let layouts: Vec<_> = shapes.iter().map(|shape| strides(shape)).collect();
        let inputs: Vec<_> = blocks
            .iter()
            .zip(&shapes)
            .zip(&layouts)
            .map(|((data, shape), strides)| {
                T::read(DenseView::new(data, shape, strides, 0).unwrap())
            })
            .collect();
        let mut buffers: Vec<Buffers<T>> = shapes
            .iter()
            .map(|shape| Buffers::new(op, dest_shapes(op, shape, FactorPlacement::AsComputed), 7.0))
            .collect();
        let mut dests: Vec<_> = buffers
            .iter_mut()
            .map(|b| b.dest(op, FactorPlacement::AsComputed))
            .collect();
        let mut delivered = Vec::new();
        DefaultDenseExecutor::new()
            .factorize_batch_into(
                &mut engine,
                op,
                &inputs,
                &mut dests,
                &mut |index, _, outcome| {
                    delivered.push((index, outcome));
                    Ok(())
                },
            )
            .unwrap();
        drop(dests);
        assert_eq!(delivered.len(), blocks.len());
        delivered
            .into_iter()
            .zip(&buffers)
            .map(|((index, outcome), buffers)| (index, outcome, buffers.widened(op).swap_remove(0)))
            .collect()
    }

    /// Purpose: the native inverse of unequal square blocks on separate
    /// lanes: per block `X Y = I` and `Y X = I`; and a singular block is
    /// that block's solver failure, exactly the per-matrix `solve_into`
    /// error of `X Y = I`, while the blocks around it are still solved.
    fn inverse_lanes_hold_the_inverse_laws<T: Elem>() {
        let tol = 1.0e-10;
        let orders = [4usize, 1, 3, 2];
        // Diagonally dominant, so every block is well conditioned.
        let blocks: Vec<Vec<T>> = orders
            .iter()
            .enumerate()
            .map(|(seed, &n)| {
                let mut data = input_data::<T>(&[n, n], false, seed);
                for diagonal in 0..n {
                    let value = data[diagonal * (n + 1)].to_c64();
                    data[diagonal * (n + 1)] = T::make(value + Complex64::new(3.0, 0.0));
                }
                data
            })
            .collect();
        let solved = inverse_batch(&blocks, &orders);
        for ((index, outcome, y), (x, &n)) in solved.iter().zip(blocks.iter().zip(&orders)) {
            assert_eq!(*outcome, Ok(()), "block {index}");
            assert_eq!(y.len(), n * n, "block {index}: output size");
            let x: Vec<Complex64> = x.iter().map(|value| value.to_c64()).collect();
            for row in 0..n {
                for col in 0..n {
                    let want = Complex64::new(f64::from(u8::from(row == col)), 0.0);
                    let xy: Complex64 = (0..n).map(|p| x[row + n * p] * y[p + n * col]).sum();
                    let yx: Complex64 = (0..n).map(|p| y[row + n * p] * x[p + n * col]).sum();
                    assert!((xy - want).norm() < tol, "block {index}: X Y != I");
                    assert!((yx - want).norm() < tol, "block {index}: Y X != I");
                }
            }
        }

        // `[[1, 2], [2, 4]]`: partial pivoting leaves an exactly zero pivot.
        let singular: Vec<T> = [1.0, 2.0, 2.0, 4.0]
            .map(|value| T::make(Complex64::new(value, 0.0)))
            .to_vec();
        let one = vec![T::make(Complex64::new(2.0, 0.0))];
        let identity: Vec<T> = [1.0, 0.0, 0.0, 1.0]
            .map(|value| T::make(Complex64::new(value, 0.0)))
            .to_vec();
        let mut scratch = identity.clone();
        let expected = DefaultDenseExecutor::new()
            .solve_into(
                &mut CpuEngine::serial(),
                T::read(DenseView::new(&singular, &[2, 2], &[1, 2], 0).unwrap()),
                T::read(DenseView::new(&identity, &[2, 2], &[1, 2], 0).unwrap()),
                T::write(DenseViewMut::new(&mut scratch, &[2, 2], &[1, 2], 0).unwrap()),
            )
            .unwrap_err();
        assert!(matches!(expected, DenseError::NumericalFailure { .. }));
        let outcomes: Vec<_> = inverse_batch(&[one.clone(), singular, one], &[1, 2, 1])
            .into_iter()
            .map(|(index, outcome, _)| (index, outcome))
            .collect();
        assert_eq!(
            outcomes,
            vec![
                (0, Ok(())),
                (1, Err(FactorFailure::Solver(expected))),
                (2, Ok(()))
            ]
        );
    }

    /// Purpose: the native inverse's failure of each tenferro route is the
    /// per-matrix `solve_into(A, I)` error in full: a singular rank-two
    /// input (the direct `solve_read_into` route), a batch holding a
    /// singular item, a rank-one input and a rank-two input with a zero
    /// extent (all on the `solve_read` fallback).
    #[test]
    fn native_inverse_errors_match_the_per_matrix_solve() {
        let cases: [(Vec<f64>, Vec<usize>, Vec<usize>); 4] = [
            (vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], vec![2, 2]),
            (
                vec![2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 2.0, 4.0],
                vec![2, 2, 2],
                vec![2, 2, 2],
            ),
            (vec![1.0, 2.0], vec![2], vec![2, 2]),
            (vec![], vec![2, 0], vec![2, 2]),
        ];
        for (data, shape, out_shape) in cases {
            let input_strides = strides(&shape);
            let out_strides = strides(&out_shape);
            let read = || DenseRead::F64(DenseView::new(&data, &shape, &input_strides, 0).unwrap());
            let mut identity = vec![0.0_f64; out_shape.iter().product()];
            for item in identity.as_chunks_mut::<4>().0 {
                item[0] = 1.0;
                item[3] = 1.0;
            }
            let mut out = identity.clone();
            let expected = DefaultDenseExecutor::new()
                .solve_into(
                    &mut CpuEngine::serial(),
                    read(),
                    DenseRead::F64(DenseView::new(&identity, &out_shape, &out_strides, 0).unwrap()),
                    DenseWrite::F64(
                        DenseViewMut::new(&mut out, &out_shape, &out_strides, 0).unwrap(),
                    ),
                )
                .unwrap_err();
            let mut dest_data = vec![7.0_f64; out_shape.iter().product()];
            let mut dests = vec![FactorDest::new(
                [
                    Some(DenseWrite::F64(
                        DenseViewMut::new(&mut dest_data, &out_shape, &out_strides, 0).unwrap(),
                    )),
                    None,
                    None,
                ],
                FactorPlacement::AsComputed,
            )];
            let mut delivered = Vec::new();
            DefaultDenseExecutor::new()
                .factorize_batch_into(
                    &mut engine(
                        4,
                        tcontract_exec::BatchPolicy::new(tcontract_exec::BatchStrategy::Outer),
                    ),
                    DenseFactorization::Inv,
                    &[read()],
                    &mut dests,
                    &mut |index, _, outcome| {
                        delivered.push((index, outcome));
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(
                delivered,
                vec![(0, Err(FactorFailure::Solver(expected)))],
                "{shape:?}"
            );
        }
    }

    #[test]
    fn native_inverse_lanes_hold_the_inverse_laws_f64() {
        inverse_lanes_hold_the_inverse_laws::<f64>();
    }

    #[test]
    fn native_inverse_lanes_hold_the_inverse_laws_c64() {
        inverse_lanes_hold_the_inverse_laws::<Complex64>();
    }

    /// The allocation calls of one native batch of `blocks` copies of an
    /// `8 x 8` input of `T` on four threads split into two lanes: the
    /// lanes' buffer setups (summed once per lane), every block's solve
    /// outside its dependency products (provider call included), and the
    /// products' calls summed over the blocks, observed at their own sites
    /// on the lane threads; scheduler and route metadata are outside all.
    fn owner_calls<T: Elem>(
        op: DenseFactorization,
        placement: FactorPlacement,
        blocks: usize,
    ) -> (usize, u64, Vec<Option<u64>>, u64) {
        use crate::block_kernel::factor::owners;
        let split = tcontract_exec::BatchPolicy::new(tcontract_exec::BatchStrategy::Split {
            lanes: std::num::NonZeroUsize::new(2).unwrap(),
        });
        let mut engine = engine(4, split);
        let shape = [8usize, 8];
        let hermitian = matches!(
            op,
            DenseFactorization::Eigh | DenseFactorization::ExpViaEigh
        );
        let item = input_data::<T>(&shape, hermitian, 1);
        let data: Vec<T> = (0..blocks).flat_map(|_| item.iter().copied()).collect();
        let layout = strides(&shape);
        let inputs: Vec<_> = data
            .as_chunks::<64>()
            .0
            .iter()
            .map(|chunk| T::read(DenseView::new(chunk, &shape, &layout, 0).unwrap()))
            .collect();
        let mut buffers: Vec<Buffers<T>> = (0..blocks)
            .map(|_| Buffers::new(op, dest_shapes(op, &shape, placement), 0.0))
            .collect();
        let mut dests: Vec<_> = buffers.iter_mut().map(|b| b.dest(op, placement)).collect();
        let mark = owners::mark();
        DefaultDenseExecutor::new()
            .factorize_batch_into(
                &mut engine,
                op,
                &inputs,
                &mut dests,
                &mut |_, _, outcome| outcome.map_err(FactorFailure::into_error),
            )
            .unwrap();
        let start = data.as_ptr() as usize;
        let records = owners::within(mark, start..start + std::mem::size_of_val(&data[..]));
        assert_eq!(records.len(), blocks);
        let mut lanes = std::collections::BTreeMap::new();
        for record in &records {
            lanes.insert(record.lane, record.setup.unwrap());
        }
        (
            lanes.len(),
            lanes.values().sum(),
            records.iter().map(|record| record.solve).collect(),
            records.iter().map(|record| record.products).sum(),
        )
    }

    /// Purpose: on lanes of a multi-lane engine, the destination batch's
    /// provider workspace and temporary factor and spectrum scratch are
    /// allocated per lane, never per block: no block's solve allocates
    /// (provider call included), and a batch of twice the blocks at the
    /// same width sets its lanes up with exactly as many allocation calls;
    /// for every scalar, op, and the adjoint placement. A polar or
    /// exponential job's products allocate nothing either where tcontract has its native GEMM
    /// kernels; elsewhere faer may allocate a packing buffer per product
    /// (tcontract-rs#162), which is the dependency's, not a symtn owner.
    fn owners_do_not_grow<T: Elem>() {
        let cases = [
            (DenseFactorization::Qr, FactorPlacement::AsComputed),
            (DenseFactorization::Qr, FactorPlacement::Adjoint),
            (DenseFactorization::Lq, FactorPlacement::AsComputed),
            (DenseFactorization::Svd, FactorPlacement::AsComputed),
            (DenseFactorization::Svd, FactorPlacement::Adjoint),
            (DenseFactorization::SvdFull, FactorPlacement::AsComputed),
            (DenseFactorization::SvdFull, FactorPlacement::Adjoint),
            (DenseFactorization::Eigh, FactorPlacement::AsComputed),
            (DenseFactorization::LeftPolar, FactorPlacement::AsComputed),
            (DenseFactorization::RightPolar, FactorPlacement::AsComputed),
            (DenseFactorization::ExpViaEigh, FactorPlacement::AsComputed),
            (DenseFactorization::Inv, FactorPlacement::AsComputed),
        ];
        for (op, placement) in cases {
            let (small_lanes, small_setup, small_solves, small_products) =
                owner_calls::<T>(op, placement, 4);
            let (large_lanes, large_setup, large_solves, large_products) =
                owner_calls::<T>(op, placement, 8);
            assert_eq!((small_lanes, large_lanes), (2, 2), "{op:?} {placement:?}");
            assert_eq!(small_setup, large_setup, "{op:?} {placement:?} lane setup");
            assert!(
                small_solves
                    .iter()
                    .chain(&large_solves)
                    .all(|&calls| calls == Some(0)),
                "{op:?} {placement:?} solves: {small_solves:?} {large_solves:?}"
            );
            if native_gemm_kernel() {
                assert_eq!(
                    (small_products, large_products),
                    (0, 0),
                    "{op:?} {placement:?} products"
                );
            }
        }
    }

    /// Whether tcontract's GEMM runs its native packing kernels here (x86_64
    /// with AVX-512, or AVX2 and FMA), which reuse a per-thread slab instead
    /// of allocating per product.
    fn native_gemm_kernel() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx512f")
                || std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    #[test]
    fn lane_owners_do_not_grow_with_blocks() {
        owners_do_not_grow::<f32>();
        owners_do_not_grow::<f64>();
        owners_do_not_grow::<Complex32>();
        owners_do_not_grow::<Complex64>();
    }

    /// Purpose: the native destination batch keeps the owned batch's
    /// admission: each admitted input's failure is its own `Solver`
    /// outcome, delivered in order to a sink that accepts it; a pre-entry
    /// refusal is delivered after the admitted prefix and later inputs are
    /// not; a rank refusal precedes the zero-extent shortcut; an
    /// empty batch delivers nothing; a sink error stops delivery; and an
    /// engine that refuses entry is the batch's own error, nothing
    /// delivered.
    #[test]
    fn native_destination_batch_keeps_the_owned_admission() {
        let good = [1.0_f64, 2.0, 3.0, 4.0];
        let nan = [1.0_f64, f64::NAN, 3.0, 4.0];
        let ints = [1_i32, 2, 3, 4];
        let ok = DenseRead::F64(DenseView::new(&good, &[2, 2], &[1, 2], 0).unwrap());
        let non_finite = DenseRead::F64(DenseView::new(&nan, &[2, 2], &[1, 2], 0).unwrap());
        let integer = DenseRead::I32(DenseView::new(&ints, &[2, 2], &[1, 2], 0).unwrap());
        let rank_one = DenseRead::F64(DenseView::new(&good, &[4], &[1], 0).unwrap());
        let empty_rank_one = DenseRead::F64(DenseView::new(&[] as &[f64], &[0], &[1], 0).unwrap());
        let op = DenseFactorization::Svd;
        let run = |engine: &mut CpuEngine, inputs: &[DenseRead<'_>], stop: Option<usize>| {
            let mut buffers: Vec<Buffers<f64>> = inputs
                .iter()
                .map(|_| {
                    Buffers::new(
                        op,
                        dest_shapes(op, &[2, 2], FactorPlacement::AsComputed),
                        0.0,
                    )
                })
                .collect();
            let mut dests: Vec<_> = buffers
                .iter_mut()
                .map(|b| b.dest(op, FactorPlacement::AsComputed))
                .collect();
            let mut delivered = Vec::new();
            let result = DefaultDenseExecutor::new().factorize_batch_into(
                engine,
                op,
                inputs,
                &mut dests,
                &mut |index, _, outcome| {
                    delivered.push((index, outcome));
                    if stop == Some(index) {
                        return Err(DenseError::OutOfBounds);
                    }
                    Ok(())
                },
            );
            (delivered, result)
        };
        let per_call = |engine: &mut CpuEngine, input| {
            DefaultDenseExecutor::new().svd(engine, input).unwrap_err()
        };
        for (threads, policy) in policies() {
            let mut engine = engine(threads, policy);
            let (delivered, result) = run(
                &mut engine,
                &[rank_one, ok, non_finite, empty_rank_one],
                None,
            );
            assert_eq!(result, Ok(()));
            let expected = vec![
                (
                    0,
                    Err(FactorFailure::Solver(per_call(&mut engine, rank_one))),
                ),
                (1, Ok(())),
                (
                    2,
                    Err(FactorFailure::Solver(per_call(&mut engine, non_finite))),
                ),
                (
                    3,
                    Err(FactorFailure::Solver(per_call(&mut engine, empty_rank_one))),
                ),
            ];
            assert_eq!(delivered, expected, "{threads} threads, {policy:?}");

            let (delivered, result) = run(&mut engine, &[ok, integer, ok], None);
            assert_eq!(result, Ok(()));
            assert_eq!(
                delivered,
                vec![
                    (0, Ok(())),
                    (
                        1,
                        Err(FactorFailure::Solver(per_call(&mut engine, integer)))
                    )
                ]
            );

            let (delivered, result) = run(&mut engine, &[ok, ok, ok], Some(1));
            assert_eq!(result, Err(DenseError::OutOfBounds));
            assert_eq!(delivered.len(), 2);

            assert_eq!(run(&mut engine, &[], None), (vec![], Ok(())));
        }

        let mut broken = engine(
            4,
            tcontract_exec::BatchPolicy::new(tcontract_exec::BatchStrategy::Outer),
        );
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::block_kernel::driver::drive(
                &mut broken,
                &crate::tests::block_svd::PanicsAt(0),
                &mut [(); 2],
                &mut crate::lanes::LaneTables::default(),
            )
        }));
        assert!(unwound.is_err() && broken.is_broken());
        let (delivered, result) = run(&mut broken, &[ok, ok], None);
        assert!(delivered.is_empty());
        assert_eq!(result.unwrap_err(), per_call(&mut broken, ok));
    }
}

/// An executor whose owned QR and destination QR disagree: its owned entry
/// returns the scripted outcome of each call in turn (a well-formed factor
/// pair where none is scripted), its `qr_into` always fails, and it counts
/// both.
#[derive(Default)]
struct Scripted {
    owned: std::collections::VecDeque<Option<Result<usize, DenseError>>>,
    qr_calls: usize,
    qr_into_calls: usize,
}

impl Scripted {
    /// `Some(Ok(count))` keeps the first `count` factors.
    fn new(owned: impl IntoIterator<Item = Option<Result<usize, DenseError>>>) -> Self {
        Self {
            owned: owned.into_iter().collect(),
            ..Self::default()
        }
    }
}

impl DenseExecutor for Scripted {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        DefaultDenseExecutor::new().svd(engine, input)
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.qr_calls += 1;
        let mut factors = DefaultDenseExecutor::new().qr(engine, input)?;
        match self.owned.pop_front().flatten() {
            None => Ok(factors),
            Some(Ok(count)) => {
                factors.truncate(count);
                Ok(factors)
            }
            Some(Err(error)) => Err(error),
        }
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        DefaultDenseExecutor::new().eigh(engine, input)
    }

    fn qr_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
        _: DenseWrite<'_>,
        _: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        self.qr_into_calls += 1;
        Err(DenseError::OutOfBounds)
    }

    fn dot_general_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        Err(DenseError::OutOfBounds)
    }
}

/// An executor that overrides the owned batch to deliver its first outcome
/// at index `self.0`.
struct Misdelivers(usize);

impl DenseExecutor for Misdelivers {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        DefaultDenseExecutor::new().svd(engine, input)
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        DefaultDenseExecutor::new().qr(engine, input)
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        DefaultDenseExecutor::new().eigh(engine, input)
    }

    fn factorize_batch(
        &mut self,
        engine: &mut dyn Engine,
        _: DenseFactorization,
        inputs: &[DenseRead<'_>],
        sink: &mut dyn FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), DenseError>,
    ) -> Result<(), DenseError> {
        let outcome = self.qr(engine, inputs[0]);
        sink(self.0, outcome)
    }

    fn dot_general_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        Err(DenseError::OutOfBounds)
    }
}

/// Two `3 x 2` QR inputs over `data`.
fn qr_inputs(data: &[f64]) -> [DenseRead<'_>; 2] {
    let view = || DenseRead::F64(DenseView::new(data, &[3, 2], &[1, 3], 0).unwrap());
    [view(), view()]
}

/// Purpose: a batch is validated whole before any input is solved or any
/// destination written: a valid first destination followed by an invalid
/// one (count, missing output, dtype, shape, non-compact layout, adjoint
/// eigh, LQ, full QR or null basis, an op without a destination form) is
/// refused with its typed variant, every destination keeps its sentinel,
/// and neither the native lanes nor a custom executor's owned entry runs.
#[test]
fn destinations_are_validated_whole_before_any_solve() {
    for op in [
        DenseFactorization::Qr,
        DenseFactorization::Lq,
        DenseFactorization::QrFull,
        DenseFactorization::QrNull,
        DenseFactorization::LeftPolar,
    ] {
        refuses_invalid_destinations(op);
    }
    no_destination_form();
}

/// [`destinations_are_validated_whole_before_any_solve`] for the QR-family
/// `op` (QR, LQ, full QR or the null basis) or left polar on `3 x 2`
/// inputs.
fn refuses_invalid_destinations(op: DenseFactorization) {
    let engine = &mut CpuEngine::serial();
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0];
    let inputs = qr_inputs(&data);
    let qr_shapes = || dest_shapes(op, &[3, 2], FactorPlacement::AsComputed);
    // The first output's shape, and a shape it must not have.
    let first_shape = qr_shapes()[0].clone();
    let first_len = first_shape.iter().product::<usize>();
    let first_strides = strides(&first_shape);
    let mut wrong_shape = qr_shapes();
    wrong_shape[0][1] += 1;
    let mut f32_q = vec![7.0_f32; first_len];
    type Check = fn(&DenseError) -> bool;
    let cases: [(&str, Check); 5] = [
        ("count", |e| matches!(e, DenseError::ShapeMismatch { .. })),
        ("missing output", |e| {
            matches!(e, DenseError::ShapeMismatch { .. })
        }),
        ("dtype", |e| matches!(e, DenseError::DTypeMismatch { .. })),
        ("shape", |e| matches!(e, DenseError::ShapeMismatch { .. })),
        ("layout", |e| matches!(e, DenseError::ShapeMismatch { .. })),
    ];
    for (case, is_expected) in cases {
        for native in [true, false] {
            let mut first: Buffers<f64> = Buffers::new(op, qr_shapes(), 7.0);
            let mut second: Buffers<f64> = Buffers::new(op, qr_shapes(), 7.0);
            let mut wide: Buffers<f64> = Buffers::new(op, wrong_shape.clone(), 7.0);
            let padded_strides = [1usize, first_shape[0] + 1];
            let mut padded = vec![7.0_f64; (first_shape[0] + 1) * first_shape[1]];
            let mut dests = vec![first.dest(op, FactorPlacement::AsComputed)];
            match case {
                "count" => {}
                "missing output" => {
                    let mut dest = second.dest(op, FactorPlacement::AsComputed);
                    dest.outputs_mut()[0] = None;
                    dests.push(dest);
                }
                "dtype" => {
                    let mut dest = second.dest(op, FactorPlacement::AsComputed);
                    dest.outputs_mut()[0] = Some(DenseWrite::F32(
                        DenseViewMut::new(&mut f32_q, &first_shape, &first_strides, 0).unwrap(),
                    ));
                    dests.push(dest);
                }
                "shape" => dests.push(wide.dest(op, FactorPlacement::AsComputed)),
                _ => {
                    let mut dest = second.dest(op, FactorPlacement::AsComputed);
                    dest.outputs_mut()[0] = Some(DenseWrite::F64(
                        DenseViewMut::new(&mut padded, &first_shape, &padded_strides, 0).unwrap(),
                    ));
                    dests.push(dest);
                }
            }
            let mut sink = |_: usize,
                            _: &mut FactorDest<'_>,
                            _: Result<(), FactorFailure>|
             -> Result<(), DenseError> {
                panic!("a refused batch delivers nothing")
            };
            let mut scripted = Scripted::default();
            let error = if native {
                DefaultDenseExecutor::new()
                    .factorize_batch_into(engine, op, &inputs, &mut dests, &mut sink)
            } else {
                scripted.factorize_batch_into(engine, op, &inputs, &mut dests, &mut sink)
            }
            .unwrap_err();
            drop(dests);
            assert!(is_expected(&error), "{op:?} {case}: {error:?}");
            assert_eq!(scripted.qr_calls, 0, "{op:?} {case}");
            for buffers in [&first, &second, &wide] {
                assert!(
                    buffers.factors.iter().flatten().all(|&value| value == 7.0),
                    "{case}"
                );
            }
            assert!(padded.iter().all(|&value| value == 7.0), "{case}");
            assert!(f32_q.iter().all(|&value| value == 7.0), "{case}");
        }
    }
}

/// Ops and placements without a destination form: adjoint eigh, LQ, full
/// QR, null basis, polar and the Hermitian exponential, and values-only
/// eigh.
fn no_destination_form() {
    let engine = &mut CpuEngine::serial();
    let square = [2.0_f64, 0.0, 0.0, 3.0];
    let eigh_input = [DenseRead::F64(
        DenseView::new(&square, &[2, 2], &[1, 2], 0).unwrap(),
    )];
    let eigh = DenseFactorization::Eigh;
    let mut buffers: Buffers<f64> = Buffers::new(
        eigh,
        dest_shapes(eigh, &[2, 2], FactorPlacement::AsComputed),
        7.0,
    );
    let mut dests = vec![buffers.dest(eigh, FactorPlacement::Adjoint)];
    for op in [eigh, DenseFactorization::EighVals] {
        let error = DefaultDenseExecutor::new()
            .factorize_batch_into(engine, op, &eigh_input, &mut dests, &mut |_, _, _| Ok(()))
            .unwrap_err();
        assert!(
            matches!(error, DenseError::Unsupported { .. }),
            "{op:?}: {error:?}"
        );
    }
    drop(dests);
    assert!(buffers.factors.iter().flatten().all(|&value| value == 7.0));

    for op in [
        DenseFactorization::Lq,
        DenseFactorization::QrFull,
        DenseFactorization::QrNull,
        DenseFactorization::LeftPolar,
        DenseFactorization::RightPolar,
        DenseFactorization::ExpViaEigh,
        DenseFactorization::Inv,
    ] {
        let mut buffers: Buffers<f64> = Buffers::new(
            op,
            dest_shapes(op, &[2, 2], FactorPlacement::AsComputed),
            7.0,
        );
        let mut dests = vec![buffers.dest(op, FactorPlacement::Adjoint)];
        let error = DefaultDenseExecutor::new()
            .factorize_batch_into(engine, op, &eigh_input, &mut dests, &mut |_, _, _| Ok(()))
            .unwrap_err();
        assert!(
            matches!(error, DenseError::Unsupported { .. }),
            "{op:?}: {error:?}"
        );
        drop(dests);
        assert!(buffers.factors.iter().flatten().all(|&value| value == 7.0));
    }
}

/// Purpose: LQ's compatibility default is the owned default's: each input's
/// adjoint factored by the executor's own `qr` (never `qr_into`), the
/// outcome placed into the destination as the owned batch returns it.
#[test]
fn lq_compatibility_default_places_the_owned_adjoint_qr() {
    let engine = &mut CpuEngine::serial();
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0];
    let inputs = qr_inputs(&data);
    let op = DenseFactorization::Lq;
    let owned = factorize_all(&mut Scripted::default(), engine, op, &inputs).unwrap();
    let mut executor = Scripted::default();
    let mut buffers: Vec<Buffers<f64>> = (0..2)
        .map(|_| {
            Buffers::new(
                op,
                dest_shapes(op, &[3, 2], FactorPlacement::AsComputed),
                7.0,
            )
        })
        .collect();
    let mut dests: Vec<_> = buffers
        .iter_mut()
        .map(|b| b.dest(op, FactorPlacement::AsComputed))
        .collect();
    let mut delivered = Vec::new();
    executor
        .factorize_batch_into(engine, op, &inputs, &mut dests, &mut |index, _, outcome| {
            delivered.push((index, outcome));
            Ok(())
        })
        .unwrap();
    drop(dests);
    assert_eq!(delivered, vec![(0, Ok(())), (1, Ok(()))]);
    assert_eq!((executor.qr_calls, executor.qr_into_calls), (2, 0));
    for (buffers, owned) in buffers.iter().zip(&owned) {
        assert_eq!(
            buffers.widened(op),
            placed(op, owned, FactorPlacement::AsComputed)
        );
    }
}

/// Purpose: the compatibility default reaches a custom executor only
/// through its owned entry (never its `*_into` entries), once per input
/// and in the order and stopping points its sink decides; tags an owned
/// `Err` as `Solver` and an owned `Ok` that does not fit as `Output`, even
/// when both carry the identical error; delivers to a sink that accepts
/// failures and continues; and refuses a first delivery at an out-of-range
/// index with the batch-arity error before touching any destination or
/// calling the sink.
#[test]
fn compatibility_default_keeps_the_owned_calls_and_tags_failures() {
    let engine = &mut CpuEngine::serial();
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0];
    let inputs = qr_inputs(&data);
    let op = DenseFactorization::Qr;
    let identical = arity_mismatch("qr_into", 2, 1);
    let mut executor = Scripted::new([Some(Err(identical.clone())), Some(Ok(1))]);
    let mut buffers: Vec<Buffers<f64>> = (0..2)
        .map(|_| {
            Buffers::new(
                op,
                dest_shapes(op, &[3, 2], FactorPlacement::AsComputed),
                7.0,
            )
        })
        .collect();
    let mut dests: Vec<_> = buffers
        .iter_mut()
        .map(|b| b.dest(op, FactorPlacement::AsComputed))
        .collect();
    let mut delivered = Vec::new();
    executor
        .factorize_batch_into(engine, op, &inputs, &mut dests, &mut |index, _, outcome| {
            delivered.push((index, outcome));
            Ok(())
        })
        .unwrap();
    assert_eq!(
        delivered,
        vec![
            (0, Err(FactorFailure::Solver(identical.clone()))),
            (1, Err(FactorFailure::Output(identical.clone()))),
        ]
    );
    assert_eq!((executor.qr_calls, executor.qr_into_calls), (2, 0));

    // A sink stop ends the owned loop before the next solve.
    let mut executor = Scripted::default();
    let result = executor.factorize_batch_into(engine, op, &inputs, &mut dests, &mut |_, _, _| {
        Err(DenseError::OutOfBounds)
    });
    assert_eq!(result, Err(DenseError::OutOfBounds));
    assert_eq!((executor.qr_calls, executor.qr_into_calls), (1, 0));
    drop(dests);

    // A well-formed outcome lands in its destination as the owned factors.
    let owned = DefaultDenseExecutor::new().qr(engine, inputs[0]).unwrap();
    assert_eq!(
        buffers[0].widened(op),
        placed(op, &owned, FactorPlacement::AsComputed)
    );

    let mut buffers: Vec<Buffers<f64>> = (0..2)
        .map(|_| {
            Buffers::new(
                op,
                dest_shapes(op, &[3, 2], FactorPlacement::AsComputed),
                7.0,
            )
        })
        .collect();
    let mut dests: Vec<_> = buffers
        .iter_mut()
        .map(|b| b.dest(op, FactorPlacement::AsComputed))
        .collect();
    let error = Misdelivers(5)
        .factorize_batch_into(engine, op, &inputs, &mut dests, &mut |_, _, _| {
            panic!("a misdelivered outcome reaches no sink")
        })
        .unwrap_err();
    drop(dests);
    assert_eq!(error, arity_mismatch("factorize_batch", 0, 5));
    assert!(buffers
        .iter()
        .all(|b| b.factors.iter().flatten().all(|&value| value == 7.0)));
}

/// Column-major `rows x cols` matrices widened to `Complex64`, the polar
/// oracle's arithmetic.
fn matmul(a: &[Complex64], b: &[Complex64], [rows, inner, cols]: [usize; 3]) -> Vec<Complex64> {
    let mut out = vec![Complex64::default(); rows * cols];
    for col in 0..cols {
        for k in 0..inner {
            for row in 0..rows {
                out[row + rows * col] += a[row + rows * k] * b[k + inner * col];
            }
        }
    }
    out
}

fn assert_matrix_close(actual: &[Complex64], expected: &[Complex64], tol: f64, what: &str) {
    let scale = expected.iter().fold(1.0_f64, |m, v| m.max(v.norm()));
    for (index, (x, y)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (x - y).norm() <= tol * scale,
            "{what} [{index}]: {x} vs {y}"
        );
    }
}

/// The polar laws of one item, from the definitions only: left
/// `A = W P` with `W^H W = I`, right `A = P Wh` with `Wh Wh^H = I`; `P`
/// exactly Hermitian with `P^2 = A^H A` (left) or `A A^H` (right) and
/// nonnegative Hermitian pivots.
fn assert_polar_item(
    left: bool,
    [m, n]: [usize; 2],
    a: &[Complex64],
    w: &[Complex64],
    p: &[Complex64],
    tol: f64,
    what: &str,
) {
    let order = if left { n } else { m };
    let a_h = adjoint_items(a, m, n);
    let w_h = adjoint_items(w, m, n);
    let identity: Vec<Complex64> = (0..order * order)
        .map(|i| Complex64::new(if i % (order + 1) == 0 { 1.0 } else { 0.0 }, 0.0))
        .collect();
    if left {
        assert_matrix_close(
            &matmul(w, p, [m, n, n]),
            a,
            tol,
            &format!("{what}: W P = A"),
        );
        assert_matrix_close(
            &matmul(&w_h, w, [n, m, n]),
            &identity,
            tol,
            &format!("{what}: W^H W"),
        );
        assert_matrix_close(
            &matmul(p, p, [n, n, n]),
            &matmul(&a_h, a, [n, m, n]),
            tol,
            &format!("{what}: P^2"),
        );
    } else {
        assert_matrix_close(
            &matmul(p, w, [m, m, n]),
            a,
            tol,
            &format!("{what}: P Wh = A"),
        );
        assert_matrix_close(
            &matmul(w, &w_h, [m, n, m]),
            &identity,
            tol,
            &format!("{what}: Wh Wh^H"),
        );
        assert_matrix_close(
            &matmul(p, p, [m, m, m]),
            &matmul(a, &a_h, [m, n, m]),
            tol,
            &format!("{what}: P^2"),
        );
    }
    for col in 0..order {
        for row in 0..order {
            assert_eq!(
                p[row + order * col],
                p[col + order * row].conj(),
                "{what}: P = P^H"
            );
        }
    }
    // Hermitian elimination: every pivot of a positive semidefinite `P` is
    // nonnegative.
    let mut reduced = p.to_vec();
    let scale = p.iter().fold(1.0_f64, |m, v| m.max(v.norm()));
    for k in 0..order {
        let pivot = reduced[k + order * k];
        assert!(pivot.re >= -tol * scale, "{what}: pivot {k} = {pivot}");
        if pivot.re <= tol * scale {
            continue;
        }
        for col in k + 1..order {
            for row in k + 1..order {
                let update = reduced[row + order * k] * reduced[k + order * col] / pivot;
                reduced[row + order * col] -= update;
            }
        }
    }
}

/// Each delivered input's index and outcome, in delivery order.
type Delivered = Vec<(usize, Result<(), FactorFailure>)>;

/// One destination batch of `op` over `shapes` with `data` on `engine`, each
/// outcome delivered to the returned list, the destinations returned.
fn run_batch_into<T: Elem>(
    engine: &mut dyn Engine,
    op: DenseFactorization,
    shapes: &[Vec<usize>],
    data: &[Vec<T>],
) -> (Delivered, Vec<Buffers<T>>) {
    run_batch_into_on(&mut DefaultDenseExecutor::new(), engine, op, shapes, data)
}

/// [`run_batch_into`] on `executor`.
fn run_batch_into_on<T: Elem>(
    executor: &mut dyn DenseExecutor,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    shapes: &[Vec<usize>],
    data: &[Vec<T>],
) -> (Delivered, Vec<Buffers<T>>) {
    let layouts: Vec<_> = shapes.iter().map(|shape| strides(shape)).collect();
    let inputs: Vec<_> = data
        .iter()
        .zip(shapes)
        .zip(&layouts)
        .map(|((data, shape), strides)| T::read(DenseView::new(data, shape, strides, 0).unwrap()))
        .collect();
    let mut buffers: Vec<Buffers<T>> = shapes
        .iter()
        .map(|shape| {
            let shape = if shape.len() >= 2 {
                shape.clone()
            } else {
                vec![2, 2]
            };
            Buffers::new(
                op,
                dest_shapes(op, &shape, FactorPlacement::AsComputed),
                7.0,
            )
        })
        .collect();
    let mut dests: Vec<_> = buffers
        .iter_mut()
        .map(|b| b.dest(op, FactorPlacement::AsComputed))
        .collect();
    let mut delivered = Vec::new();
    executor
        .factorize_batch_into(engine, op, &inputs, &mut dests, &mut |index, _, outcome| {
            delivered.push((index, outcome));
            Ok(())
        })
        .unwrap();
    drop(dests);
    (delivered, buffers)
}

/// Purpose: the default executor's polar destinations hold the polar laws
/// (independent of any SVD gauge) for every scalar, both directions,
/// square, rectangular and trailing-batch inputs of unequal size in one
/// batch, and the closed form of `A = R P0` (left) / `A = P0 R` (right) with
/// a rotation `R` and a positive definite `P0`; empty extents deliver
/// success and write nothing.
fn polar_laws<T: Elem>(tol: f64) {
    let engine = &mut CpuEngine::serial();
    let tall: &[&[usize]] = &[
        &[5, 3],
        &[3, 3],
        &[4, 2, 2],
        &[1, 1],
        &[3, 0],
        &[0, 0],
        &[2, 2, 0],
    ];
    for op in [
        DenseFactorization::LeftPolar,
        DenseFactorization::RightPolar,
    ] {
        let left = op == DenseFactorization::LeftPolar;
        let shapes: Vec<Vec<usize>> = tall
            .iter()
            .map(|shape| {
                let mut shape = shape.to_vec();
                if !left {
                    shape.swap(0, 1);
                }
                shape
            })
            .collect();
        let data: Vec<Vec<T>> = shapes
            .iter()
            .enumerate()
            .map(|(seed, shape)| input_data::<T>(shape, false, seed + 1))
            .collect();
        let (delivered, buffers) = run_batch_into(engine, op, &shapes, &data);
        assert_eq!(
            delivered,
            (0..shapes.len()).map(|i| (i, Ok(()))).collect::<Vec<_>>(),
            "{op:?}"
        );
        for ((shape, data), buffers) in shapes.iter().zip(&data).zip(&buffers) {
            let (m, n) = (shape[0], shape[1]);
            let items = shape[2..].iter().product::<usize>();
            let outputs = buffers.widened(op);
            let (w, p) = if left {
                (&outputs[0], &outputs[1])
            } else {
                (&outputs[1], &outputs[0])
            };
            let order = if left { n } else { m };
            let a: Vec<Complex64> = data.iter().map(|&v| v.to_c64()).collect();
            for item in 0..items {
                assert_polar_item(
                    left,
                    [m, n],
                    &a[item * m * n..][..m * n],
                    &w[item * m * n..][..m * n],
                    &p[item * order * order..][..order * order],
                    tol,
                    &format!("{op:?} {shape:?} item {item}"),
                );
            }
        }

        // R = [[0.6, -0.8], [0.8, 0.6]], P0 = [[2, 1], [1, 2]].
        let rotation = [0.6, 0.8, -0.8, 0.6].map(|v| T::make(Complex64::new(v, 0.0)));
        let positive = [2.0, 1.0, 1.0, 2.0].map(|v| T::make(Complex64::new(v, 0.0)));
        let widen = |m: &[T]| m.iter().map(|&v| v.to_c64()).collect::<Vec<_>>();
        let (r, p0) = (widen(&rotation), widen(&positive));
        let a = if left {
            matmul(&r, &p0, [2, 2, 2])
        } else {
            matmul(&p0, &r, [2, 2, 2])
        };
        let a: Vec<T> = a.into_iter().map(T::make).collect();
        let (_, buffers) = run_batch_into(engine, op, &[vec![2, 2]], &[a]);
        let outputs = buffers[0].widened(op);
        let (w, p) = if left {
            (&outputs[0], &outputs[1])
        } else {
            (&outputs[1], &outputs[0])
        };
        assert_matrix_close(w, &r, tol, &format!("{op:?} closed-form W"));
        assert_matrix_close(p, &p0, tol, &format!("{op:?} closed-form P"));
    }
}

#[test]
fn polar_destinations_hold_the_polar_laws() {
    polar_laws::<f32>(1.0e-4);
    polar_laws::<f64>(1.0e-12);
    polar_laws::<Complex32>(1.0e-4);
    polar_laws::<Complex64>(1.0e-12);
}

/// Purpose: the polar decompositions have only a destination form, and
/// each input's refusal is its own `Solver` outcome: the owned batch of
/// either executor refuses them with `Unsupported` before any solve; in a
/// destination batch a rank-one input carries the SVD's own refusal and an
/// input in the wrong direction (`m < n` for left, `n < m` for right) its
/// shape refusal even when it is empty, while the inputs around them
/// succeed.
#[test]
fn polar_refuses_without_a_destination_and_per_input() {
    let engine = &mut CpuEngine::serial();
    let good = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0];
    for op in [
        DenseFactorization::LeftPolar,
        DenseFactorization::RightPolar,
    ] {
        let left = op == DenseFactorization::LeftPolar;
        let (ok, wrong, empty_wrong) = if left {
            (vec![3, 2], vec![2, 3], vec![0, 3])
        } else {
            (vec![2, 3], vec![3, 2], vec![3, 0])
        };
        let shapes = [ok.clone(), vec![4], wrong, empty_wrong, ok];
        let data: Vec<Vec<f64>> = shapes
            .iter()
            .map(|shape| good[..shape.iter().product::<usize>()].to_vec())
            .collect();
        let rank_one = DenseRead::F64(DenseView::new(&data[1], &[4], &[1], 0).unwrap());
        let rank_error = DefaultDenseExecutor::new()
            .svd(engine, rank_one)
            .unwrap_err();
        let (delivered, _) = run_batch_into(engine, op, &shapes, &data);
        let direction = |outcome: &Result<(), FactorFailure>| {
            matches!(
                outcome,
                Err(FactorFailure::Solver(DenseError::ShapeMismatch { op, .. }))
                    if *op == if left { "left_polar" } else { "right_polar" }
            )
        };
        assert_eq!(delivered.len(), 5, "{op:?}");
        assert_eq!(delivered[0], (0, Ok(())), "{op:?}");
        assert_eq!(
            delivered[1],
            (1, Err(FactorFailure::Solver(rank_error))),
            "{op:?}"
        );
        assert!(direction(&delivered[2].1), "{op:?}: {:?}", delivered[2]);
        assert!(direction(&delivered[3].1), "{op:?}: {:?}", delivered[3]);
        assert_eq!(delivered[4], (4, Ok(())), "{op:?}");

        let input = DenseRead::F64(DenseView::new(&good, &[3, 2], &[1, 3], 0).unwrap());
        for executor in [
            &mut DefaultDenseExecutor::new() as &mut dyn DenseExecutor,
            &mut Scripted::default(),
        ] {
            let error = factorize_all(executor, engine, op, &[input]).unwrap_err();
            assert!(
                matches!(
                    error,
                    DenseError::Unsupported {
                        op: "factorize_batch",
                        ..
                    }
                ),
                "{op:?}: {error:?}"
            );
        }
    }
}

/// An executor whose SVD returns the factors of its input cast to `f32`.
struct F32Svd;

impl DenseExecutor for F32Svd {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let DenseRead::F64(view) = input else {
            unreachable!("the fixture factors f64 inputs")
        };
        let data: Vec<f32> = view.data().iter().map(|&value| value as f32).collect();
        let cast = DenseRead::F32(DenseView::new(&data, view.shape(), view.strides(), 0)?);
        DefaultDenseExecutor::new().svd(engine, cast)
    }

    fn qr(&mut self, _: &mut dyn Engine, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("polar solves no QR")
    }

    fn eigh(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("polar solves no eigh")
    }

    fn dot_general_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("a malformed SVD output stops the input before its products")
    }
}

/// Purpose: a custom executor's polar SVD factors of the wrong dtype are
/// that input's `Output` failure with the per-matrix check's own typed
/// error (the borrowed slice access of the expected scalar), for both
/// directions.
#[test]
fn polar_composite_reports_a_wrong_dtype_factor_as_the_borrowed_check() {
    let engine = &mut CpuEngine::serial();
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 7.0];
    for op in [
        DenseFactorization::LeftPolar,
        DenseFactorization::RightPolar,
    ] {
        let shape = if op == DenseFactorization::LeftPolar {
            [3, 2]
        } else {
            [2, 3]
        };
        let layout = strides(&shape);
        let inputs = [DenseRead::F64(
            DenseView::new(&data, &shape, &layout, 0).unwrap(),
        )];
        let mut buffers: Buffers<f64> = Buffers::new(
            op,
            dest_shapes(op, &shape, FactorPlacement::AsComputed),
            7.0,
        );
        let mut dests = vec![buffers.dest(op, FactorPlacement::AsComputed)];
        let mut delivered = Vec::new();
        F32Svd
            .factorize_batch_into(engine, op, &inputs, &mut dests, &mut |index, _, outcome| {
                delivered.push((index, outcome));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            delivered,
            vec![(
                0,
                Err(FactorFailure::Output(DenseError::DTypeMismatch {
                    op: "DenseTensor::as_f64_slice",
                    expected: DenseDType::F64,
                    actual: DenseDType::F32,
                }))
            )],
            "{op:?}"
        );
    }
}

/// An executor that keeps the default destination batch, so the Hermitian
/// exponential runs its composite: the default executor's `eigh` (its first
/// eigenvalue replaced by NaN with `nan_value`) and products, counted.
#[derive(Default)]
struct CompositeExp {
    nan_value: bool,
    products: usize,
}

impl DenseExecutor for CompositeExp {
    fn svd(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the exponential solves no SVD")
    }

    fn qr(&mut self, _: &mut dyn Engine, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the exponential solves no QR")
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let mut outputs = DefaultDenseExecutor::new().eigh(engine, input)?;
        if self.nan_value {
            let mut values = outputs[0].as_f64_slice()?.to_vec();
            values[0] = f64::NAN;
            outputs[0] = DenseTensor::host_col_major(outputs[0].shape().to_vec(), values)?;
        }
        Ok(outputs)
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        out: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.products += 1;
        DefaultDenseExecutor::new().dot_general_into(engine, out, lhs, rhs, config)
    }
}

/// `exp` of the Hermitian `[[a, conj(b)], [b, a]]`, column-major:
/// `e^a [[cosh |b|, conj(b)/|b| sinh |b|], [b/|b| sinh |b|, cosh |b|]]`.
fn exp_two_by_two(a: f64, b: Complex64) -> Vec<Complex64> {
    let r = b.norm();
    let (cosh, sinh) = (a.exp() * r.cosh(), a.exp() * r.sinh());
    let phase = b / r;
    vec![
        Complex64::new(cosh, 0.0),
        phase * sinh,
        phase.conj() * sinh,
        Complex64::new(cosh, 0.0),
    ]
}

/// The exact Hermitian structure of each column-major `n x n` item: the
/// lower triangle the upper one's adjoint, the diagonal real.
fn assert_exactly_hermitian(data: &[Complex64], n: usize, what: &str) {
    for item in data.chunks_exact(n * n) {
        for col in 0..n {
            for row in 0..n {
                assert_eq!(
                    item[row + n * col],
                    item[col + n * row].conj(),
                    "{what}: ({row}, {col})"
                );
            }
        }
    }
}

/// Purpose: the Hermitian exponential's destinations hold the closed form
/// on the default executor and on the per-matrix composite, for every
/// scalar: `[[a, conj(b)], [b, a]]` (hand formula above), a diagonal
/// `3 x 3`, a two-item trailing batch, `1 x 1`, exactly Hermitian; empty
/// extents deliver success.
fn exp_closed_form<T: Elem>(tol: f64) {
    let engine = &mut CpuEngine::serial();
    let widen = |values: &[T]| values.iter().map(|&v| v.to_c64()).collect::<Vec<_>>();
    let make = |values: &[Complex64]| values.iter().map(|&v| T::make(v)).collect::<Vec<_>>();
    let (a, b) = (0.25, Complex64::new(0.3, 0.4));
    let pair = make(&[Complex64::new(a, 0.0), b, b.conj(), Complex64::new(a, 0.0)]);
    // A real scalar keeps `b`'s real part only.
    let b = widen(&pair)[1];
    let real = |v: f64| Complex64::new(v, 0.0);
    let diagonal = |values: &[f64]| {
        let n = values.len();
        let mut out = vec![real(0.0); n * n];
        for (i, &v) in values.iter().enumerate() {
            out[i * (n + 1)] = real(v);
        }
        out
    };
    let exp_diagonal =
        |values: &[f64]| diagonal(&values.iter().map(|v| v.exp()).collect::<Vec<_>>());
    let cases: Vec<(Vec<usize>, Vec<T>, Vec<Complex64>)> = vec![
        (vec![2, 2], pair.clone(), exp_two_by_two(a, b)),
        (
            vec![3, 3],
            make(&diagonal(&[-1.0, 0.5, 2.0])),
            exp_diagonal(&[-1.0, 0.5, 2.0]),
        ),
        (
            vec![2, 2, 2],
            [pair.clone(), make(&diagonal(&[1.0, -2.0]))].concat(),
            [exp_two_by_two(a, b), exp_diagonal(&[1.0, -2.0])].concat(),
        ),
        (
            vec![1, 1],
            make(&[real(-0.75)]),
            vec![real((-0.75_f64).exp())],
        ),
        (vec![0, 0], vec![], vec![]),
        (vec![2, 2, 0], vec![], vec![]),
    ];
    let shapes: Vec<_> = cases.iter().map(|case| case.0.clone()).collect();
    let data: Vec<_> = cases.iter().map(|case| case.1.clone()).collect();
    let op = DenseFactorization::ExpViaEigh;
    for composite in [false, true] {
        let (delivered, buffers) = if composite {
            run_batch_into_on(&mut CompositeExp::default(), engine, op, &shapes, &data)
        } else {
            run_batch_into(engine, op, &shapes, &data)
        };
        assert_eq!(
            delivered,
            (0..shapes.len()).map(|i| (i, Ok(()))).collect::<Vec<_>>(),
            "composite {composite}"
        );
        for ((shape, _, expected), buffers) in cases.iter().zip(&buffers) {
            let what = format!("composite {composite} {shape:?}");
            let exp = &buffers.widened(op)[0];
            assert_matrix_close(&exp[..expected.len()], expected, tol, &what);
            if !expected.is_empty() {
                assert_exactly_hermitian(exp, shape[0], &what);
            }
        }
    }
}

#[test]
fn exp_destinations_hold_the_closed_form() {
    exp_closed_form::<f32>(1.0e-5);
    exp_closed_form::<f64>(1.0e-13);
    exp_closed_form::<Complex32>(1.0e-5);
    exp_closed_form::<Complex64>(1.0e-13);
}

/// Purpose: the Hermitian exponential is finite wherever the result is
/// representable, even where `exp` of an eigenvalue alone overflows:
/// `[[355, 355], [355, 355]]` has eigenvalues `0` and `710`, and every
/// entry of its exponential is `(e^710 ± 1) / 2 = e^(710 - ln 2)` to double
/// precision, on the default executor and on the composite.
#[test]
fn exp_is_finite_where_only_the_eigenvalue_exponential_overflows() {
    let engine = &mut CpuEngine::serial();
    let op = DenseFactorization::ExpViaEigh;
    let shapes = [vec![2, 2]];
    let data = [vec![355.0_f64; 4]];
    let expected = vec![Complex64::new((710.0 - std::f64::consts::LN_2).exp(), 0.0); 4];
    for composite in [false, true] {
        let (delivered, buffers) = if composite {
            run_batch_into_on(&mut CompositeExp::default(), engine, op, &shapes, &data)
        } else {
            run_batch_into(engine, op, &shapes, &data)
        };
        assert_eq!(delivered, vec![(0, Ok(()))], "composite {composite}");
        assert_matrix_close(
            &buffers[0].widened(op)[0],
            &expected,
            1.0e-12,
            &format!("composite {composite}"),
        );
    }
}

/// Purpose: the Hermitian exponential has only a destination form, and each
/// input's refusal is its own `Solver` outcome: the owned batch of either
/// executor refuses it with `Unsupported` before any solve; in a destination
/// batch a rank-one input carries the eigensolver's own refusal and a
/// non-square input its squareness refusal even when it is empty, while the
/// inputs around them succeed; an empty batch delivers nothing.
#[test]
fn exp_refuses_without_a_destination_and_per_input() {
    let engine = &mut CpuEngine::serial();
    let op = DenseFactorization::ExpViaEigh;
    let good = [1.0_f64, 2.0, 2.0, 4.0, 5.0, 7.0];
    let shapes = [vec![2, 2], vec![4], vec![2, 3], vec![0, 3], vec![2, 2]];
    let data: Vec<Vec<f64>> = shapes
        .iter()
        .map(|shape| good[..shape.iter().product::<usize>()].to_vec())
        .collect();
    let refusal = |engine: &mut dyn Engine, shape: &[usize], data: &[f64]| {
        let layout = strides(shape);
        let input = DenseRead::F64(DenseView::new(data, shape, &layout, 0).unwrap());
        DefaultDenseExecutor::new().eigh(engine, input).unwrap_err()
    };
    let expected: Vec<_> = (0..shapes.len())
        .map(|i| {
            if shapes[i] == [2, 2] {
                (i, Ok(()))
            } else {
                (
                    i,
                    Err(FactorFailure::Solver(refusal(engine, &shapes[i], &data[i]))),
                )
            }
        })
        .collect();
    for executor in [
        &mut DefaultDenseExecutor::new() as &mut dyn DenseExecutor,
        &mut CompositeExp::default(),
    ] {
        let (delivered, _) = run_batch_into_on(executor, engine, op, &shapes, &data);
        assert_eq!(delivered, expected);
        let (delivered, _) = run_batch_into_on(executor, engine, op, &[], &[] as &[Vec<f64>]);
        assert!(delivered.is_empty());
    }

    let input = DenseRead::F64(DenseView::new(&good[..4], &[2, 2], &[1, 2], 0).unwrap());
    for executor in [
        &mut DefaultDenseExecutor::new() as &mut dyn DenseExecutor,
        &mut CompositeExp::default(),
    ] {
        let error = factorize_all(executor, engine, op, &[input]).unwrap_err();
        assert!(
            matches!(
                error,
                DenseError::Unsupported {
                    op: "factorize_batch",
                    ..
                }
            ),
            "{error:?}"
        );
    }
}

/// The eigenvalue check's failure: symtn's own `NumericalFailure`.
fn eigenvalue_check_failure() -> FactorFailure {
    FactorFailure::Output(DenseError::NumericalFailure {
        backend: crate::DenseBackend::Tenferro,
        op: crate::EIGH_EIGENVALUE_CHECK,
        message: "eigenvalues must be finite".to_string(),
    })
}

/// Purpose: the composite's nonfinite eigenvalue is that input's `Output`
/// failure from symtn's eigenvalue check, reported before any product.
#[test]
fn exp_composite_tags_a_nonfinite_eigenvalue_as_an_output_failure() {
    let engine = &mut CpuEngine::serial();
    let mut executor = CompositeExp {
        nan_value: true,
        ..CompositeExp::default()
    };
    let (delivered, _) = run_batch_into_on(
        &mut executor,
        engine,
        DenseFactorization::ExpViaEigh,
        &[vec![2, 2]],
        &[vec![1.0_f64, 0.5, 0.5, 2.0]],
    );
    assert_eq!(delivered, vec![(0, Err(eigenvalue_check_failure()))]);
    assert_eq!(executor.products, 0);
}

/// Purpose: on the native provider a finite Hermitian input whose largest
/// eigenvalue overflows (`[[M, M], [M, M]]`, `M = f64::MAX / sqrt(2)`, so
/// `2 M = inf`) is that input's `Output` failure from symtn's eigenvalue
/// check, not a provider error, and the input after it still succeeds.
#[cfg(all(feature = "cpu-faer", not(feature = "cpu-blas-core")))]
#[test]
fn native_exp_tags_an_overflowing_eigenvalue_as_an_output_failure() {
    let engine = &mut CpuEngine::serial();
    let m = f64::MAX / std::f64::consts::SQRT_2;
    let (delivered, _) = run_batch_into(
        engine,
        DenseFactorization::ExpViaEigh,
        &[vec![2, 2], vec![2, 2]],
        &[vec![m; 4], vec![1.0, 0.5, 0.5, 2.0]],
    );
    assert_eq!(
        delivered,
        vec![(0, Err(eigenvalue_check_failure())), (1, Ok(()))]
    );
}
