//! TensorKit's zero-scale rule on the typed `add`/`scale` family (#1442).
//!
//! Oracle: Julia 1.11, TensorKit main f87ca7fe (version 0.17.1),
//! VectorInterface 0.6.0. For `x, y ∈ V ⊗ V ← V`
//! over U(1), SU(2) and fZ₂ legs, `Float64` and `ComplexF64`, with `Inf`,
//! NaN and finite entries, `add(y, x, α, β)`, `add!(y, x, α, β)`,
//! `scale(x, α)`, `scale!(x, α)` and `scale!(y, x, α)` equal, elementwise,
//! VectorInterface's number rule `scale(x, α) = (iszero(α) ? zero(x) : x) * α`
//! composed as `scale(y, β) + scale(x, α)`: a zero coefficient drops its
//! operand, NaN and `Inf` included. symtn's `a.axpby(alpha, &b, beta)` is
//! `alpha * a + beta * b`, so the expectation here is
//! `scale(a, alpha) + scale(b, beta)`.
//!
//! No fixture holds `-0.0`, so the sign of a zero cannot decide a comparison.

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap, TensorScalar};

const X: [(f64, f64); 5] = [
    (f64::INFINITY, 1.0),
    (f64::NAN, 2.5),
    (1.5, f64::NEG_INFINITY),
    (-1.25, 0.5),
    (2.0, 3.0),
];
const Y: [(f64, f64); 4] = [(1.0, -2.0), (-4.0, 0.5), (f64::NAN, 1.0), (3.0, -7.0)];

trait Payload: TensorScalar + std::fmt::Debug {
    fn from_parts(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
}

impl Payload for f64 {
    fn from_parts(re: f64, _: f64) -> Self {
        re
    }
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

impl Payload for Complex64 {
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

fn real<D: Payload>(value: f64) -> D {
    D::from_parts(value, 0.0)
}

fn scale<D: Payload>(value: D, factor: f64) -> D {
    if factor == 0.0 {
        real(0.0)
    } else {
        value * real(factor)
    }
}

fn assert_same<D: Payload>(what: &str, got: &[D], want: &[D]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let same = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        let ((a, b), (c, d)) = (got.parts(), want.parts());
        assert!(
            same(a, c) && same(b, d),
            "{what}: element {index} is {got:?}, TensorKit gives {want:?}"
        );
    }
}

/// A tensor on `leg ⊗ leg ← leg` whose entries cycle through `values`.
macro_rules! tensor {
    ($runtime:expr, $leg:expr, $values:expr, $d:ty) => {{
        let next = std::cell::Cell::new(0usize);
        TensorMap::<_, $d>::from_subblock_fn($runtime, [$leg, $leg], [$leg], |_, _| {
            let index = next.get();
            next.set(index + 1);
            let (re, im) = $values[index % $values.len()];
            <$d as Payload>::from_parts(re, im)
        })
        .unwrap()
    }};
}

macro_rules! check_dense {
    ($label:expr, $leg:expr, $d:ty) => {{
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = $leg;
        let x = tensor!(&runtime, &leg, X, $d);
        let y = tensor!(&runtime, &leg, Y, $d);
        let combine = |alpha: f64, beta: f64| -> Vec<$d> {
            x.dense_data()
                .unwrap()
                .iter()
                .zip(y.dense_data().unwrap())
                .map(|(&a, &b)| scale(a, alpha) + scale(b, beta))
                .collect()
        };
        for (alpha, beta) in [(0.0, 1.0), (0.0, 2.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0)] {
            let what = format!("{} alpha = {alpha}, beta = {beta}", $label);
            let (a, b) = (real::<$d>(alpha), real::<$d>(beta));
            let want = combine(alpha, beta);
            assert_same(
                &format!("{what} add"),
                x.axpby(a, &y, b).unwrap().dense_data().unwrap(),
                &want,
            );
            let mut assigned = x.materialize().unwrap();
            y.axpby_into(&mut assigned, b, a).unwrap();
            assert_same(
                &format!("{what} axpby_into"),
                assigned.dense_data().unwrap(),
                &want,
            );
            // The lazy adjoint route: `(x^H)^H` reads `x` in its logical
            // orientation.
            let lazy = x.adjoint().unwrap().adjoint().unwrap();
            assert_same(
                &format!("{what} lazy add"),
                lazy.axpby(a, &y, b).unwrap().dense_data().unwrap(),
                &want,
            );
        }
        // Non-zero coefficients keep the base arithmetic `x * alpha + y * beta`
        // bit for bit.
        let bits = |values: &[$d]| -> Vec<(u64, u64)> {
            values
                .iter()
                .map(|v| (v.parts().0.to_bits(), v.parts().1.to_bits()))
                .collect()
        };
        let (a, b) = (real::<$d>(2.0), real::<$d>(-1.5));
        let base: Vec<$d> = x
            .dense_data()
            .unwrap()
            .iter()
            .zip(y.dense_data().unwrap())
            .map(|(&u, &v)| u * a + v * b)
            .collect();
        assert_eq!(
            bits(x.axpby(a, &y, b).unwrap().dense_data().unwrap()),
            bits(&base)
        );
        let mut assigned = x.materialize().unwrap();
        y.axpby_into(&mut assigned, b, a).unwrap();
        assert_eq!(bits(assigned.dense_data().unwrap()), bits(&base));
        let base: Vec<$d> = x.dense_data().unwrap().iter().map(|&u| u * a).collect();
        assert_eq!(bits(x.scale(a).unwrap().dense_data().unwrap()), bits(&base));
        for factor in [0.0, 2.0] {
            let what = format!("{} factor = {factor}", $label);
            let want: Vec<$d> = x
                .dense_data()
                .unwrap()
                .iter()
                .map(|&a| scale(a, factor))
                .collect();
            let f = real::<$d>(factor);
            assert_same(
                &format!("{what} scale"),
                x.scale(f).unwrap().dense_data().unwrap(),
                &want,
            );
            let mut assigned = x.materialize().unwrap();
            assigned.scale_assign(f).unwrap();
            assert_same(
                &format!("{what} scale_assign"),
                assigned.dense_data().unwrap(),
                &want,
            );
        }
    }};
}

fn u1() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [0, 1, -1].map(|charge| (U1Irrep::new(charge), 2)),
    )
    .unwrap()
}

fn su2() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        std::sync::Arc::new(SU2FusionRule),
        [0, 1, 2].map(|twice| (SU2Irrep::from_twice_spin(twice), 2)),
    )
    .unwrap()
}

fn fz2() -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        std::sync::Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)],
    )
    .unwrap()
}

#[test]
fn typed_add_and_scale_drop_zero_scaled_operands_as_tensorkit() {
    check_dense!("U(1) f64", u1(), f64);
    check_dense!("U(1) c64", u1(), Complex64);
    check_dense!("SU(2) f64", su2(), f64);
    check_dense!("SU(2) c64", su2(), Complex64);
    check_dense!("fZ2 f64", fz2(), f64);
    check_dense!("fZ2 c64", fz2(), Complex64);
}

/// The compact diagonal arms (TensorKit's `DiagonalTensorMap` `add`/`scale`,
/// observed with the same rule): two spectra, and a spectrum against a dense
/// operand.
#[test]
fn compact_diagonal_add_and_scale_drop_zero_scaled_operands_as_tensorkit() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let spectra = |values: [[f64; 2]; 2]| {
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: values[0].to_vec(),
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: values[1].to_vec(),
            },
        ]
    };
    let inf = f64::INFINITY;
    let x = TensorMap::diagonal(&runtime, &leg, spectra([[inf, 1.0], [f64::NAN, -2.0]])).unwrap();
    let y = TensorMap::diagonal(&runtime, &leg, spectra([[3.0, f64::NAN], [0.5, 4.0]])).unwrap();
    let dense_y: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
            if ij[0] == ij[1] {
                f64::NAN
            } else {
                ij[0] as f64 + 1.0
            }
        })
        .unwrap();
    for (alpha, beta) in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)] {
        let what = format!("alpha = {alpha}, beta = {beta}");
        let want: Vec<f64> = x
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(y.materialize().unwrap().dense_data().unwrap())
            .map(|(&a, &b)| scale(a, alpha) + scale(b, beta))
            .collect();
        assert_same(
            &format!("{what} spectra"),
            x.axpby(alpha, &y, beta)
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            &want,
        );
        let want: Vec<f64> = x
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(dense_y.dense_data().unwrap())
            .map(|(&a, &b)| scale(a, alpha) + scale(b, beta))
            .collect();
        assert_same(
            &format!("{what} spectrum + dense"),
            x.axpby(alpha, &dense_y, beta)
                .unwrap()
                .dense_data()
                .unwrap(),
            &want,
        );
    }
    assert_same(
        "spectrum scale 0",
        x.scale(0.0)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        &vec![0.0; x.materialize().unwrap().dense_data().unwrap().len()],
    );
}
