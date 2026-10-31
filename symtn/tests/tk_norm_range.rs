//! `norm(p)` keeps a representable result when the unscaled power sum
//! would overflow or underflow.
//!
//! # Reference behaviour
//!
//! TensorKit `f87ca7f` (0.17.1), `src/tensors/linalg.jl`:
//!
//! - `norm(t::TensorMap, p)` (`:277-282`) returns `norm(t.data, p)` for
//!   `UniqueFusion`. Julia 1.11.6 `LinearAlgebra.norm2` takes `generic_norm2`
//!   below `NRM2_CUTOFF = 32` entries and `BLAS.nrm2` above (`dense.jl:9,106`);
//!   both rescale, as does `generic_normp` (`generic.jl:468,498`). So U(1) and
//!   fermion-parity norms are finite at `1e200` and nonzero at `1e-200`.
//! - Every other fusion style goes through `_norm` (`:261-275`), which adds
//!   `dim(c) * norm(b, p)^p` **unscaled** in `float(real(scalartype(t)))`: each
//!   block norm is finite, but its power overflows to `Inf` (or flushes to `0`),
//!   so TensorKit's SU(2) `norm(t)` is `Inf` at `1e200`, `0.0` at `1e-200`, and
//!   `Inf` for a `Float32` tensor at `1e30`. symtn rescales across all coupled
//!   sectors and returns the representable norm there instead; those rows use
//!   the hand oracle `norm(s * t) = s * norm(t)` with TensorKit's in-range
//!   value, and record TensorKit's own output beside it.
//!
//! QSpace `d2d3d7da` `Source/wbarray.hh:wbarray<T>::norm2` (`:2962`) is an
//! unscaled `Σ x²` and `Source/QSpace.cc:QSpace<TQ,TD>::norm2` (`:2258`) sums
//! it over blocks, so QSpace has no overflow-safe path to compare against.
//!
//! # Oracle
//!
//! The fills are dyadic, so every expected norm is the exact formula
//! `‖t‖_p = (Σ_c dim(c) Σ_ij |x_ij|^p)^(1/p)` (and `max |x_ij|` at `p = Inf`)
//! evaluated in the test by [`expected`]. TensorKit main `f87ca7f` (0.17.1) on
//! Julia 1.11.6 prints the unit-scale rows kept in
//! `formula_and_symtn_reproduce_tensorkit_at_unit_scale`, which anchor the
//! formula's conventions (quantum-dimension weight, complex modulus).

use std::sync::Arc;

use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{Complex32, Complex64};
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap};

use crate::numerics;

use numerics::Numeric;

use crate::fixtures;

use fixtures::host_runtime;

fn real_fill(indices: &[usize]) -> f64 {
    1.0 + 0.5 * indices[0] as f64 - 0.25 * indices[1] as f64
}

fn complex_fill(indices: &[usize]) -> Complex64 {
    Complex64::new(
        real_fill(indices),
        0.5 + 0.125 * indices[0] as f64 + 0.375 * indices[1] as f64,
    )
}

fn u1_space(entries: [(i32, usize); 3]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        entries.map(|(charge, deg)| (U1Irrep::new(charge), deg)),
    )
    .unwrap()
}

/// `U1Space(-1 => 2, 0 => 3, 1 => 4) ← U1Space(-1 => 5, 0 => 1, 1 => 2)`:
/// 21 entries, Julia's `generic_norm2` dispatch.
fn u1() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    (
        u1_space([(-1, 2), (0, 3), (1, 4)]),
        u1_space([(-1, 5), (0, 1), (1, 2)]),
    )
}

/// `U1Space(-1 => 6, 0 => 5, 1 => 4) ← U1Space(-1 => 5, 0 => 3, 1 => 2)`:
/// 53 entries, past `NRM2_CUTOFF`, so TensorKit reaches `BLAS.nrm2`.
fn u1_big() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    (
        u1_space([(-1, 6), (0, 5), (1, 4)]),
        u1_space([(-1, 5), (0, 3), (1, 2)]),
    )
}

fn su2_space(entries: [(usize, usize); 3]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        entries.map(|(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
    )
    .unwrap()
}

/// `SU2Space(0 => 2, 1//2 => 3, 1 => 4) ← SU2Space(0 => 5, 1//2 => 1, 1 => 2)`.
fn su2() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    (
        su2_space([(0, 2), (1, 3), (2, 4)]),
        su2_space([(0, 5), (1, 1), (2, 2)]),
    )
}

/// `Vect[FermionParity](0 => 2, 1 => 3) ← Vect[FermionParity](0 => 5, 1 => 1)`.
fn fz2() -> (
    GradedSpace<FermionParityFusionRule>,
    GradedSpace<FermionParityFusionRule>,
) {
    let rule = Arc::new(FermionParityFusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)]).unwrap(),
        GradedSpace::try_new(rule, [(Z2Irrep::EVEN, 5), (Z2Irrep::ODD, 1)]).unwrap(),
    )
}

macro_rules! tensor {
    ($engine:ident, $spaces:expr, $d:ty, $fill:expr) => {{
        let (v, w) = $spaces;
        let tensor: TensorMap<_, $d> =
            TensorMap::from_subblock_fn($engine, &host_runtime(), [&v], [&w], $fill).unwrap();
        tensor
    }};
}

/// `[norm(1), norm(2), norm(3), norm(Inf)]`.
macro_rules! norms {
    ($engine:ident, $tensor:expr) => {{
        let tensor = &$tensor;
        [
            tensor.norm($engine, 1.0).unwrap(),
            tensor.norm($engine, 2.0).unwrap(),
            tensor.norm($engine, 3.0).unwrap(),
            tensor.norm($engine, f64::INFINITY).unwrap(),
        ]
    }};
}

/// The workspace rule (`docs/testing_numerics.md`) relative to `|want|`.
///
/// Why not `numerics::assert_close`: its `max(1, scale)` floor makes the
/// bound absolute below one, so at `1e-200` it would accept `0.0`, the very
/// underflow under test. `terms` is the stored entry count, the number of
/// terms in each power sum. Non-finite and zero oracles compare exactly.
#[track_caller]
fn assert_rows<T: Numeric>(what: &str, got: [f64; 4], want: [f64; 4], terms: usize) {
    for (index, (&got, &want)) in got.iter().zip(&want).enumerate() {
        let p = ["1", "2", "3", "Inf"][index];
        if want.is_nan() {
            assert!(got.is_nan(), "{what} p={p}: {got:e}, TensorKit NaN");
        } else if want.is_infinite() || want == 0.0 {
            assert_eq!(got, want, "{what} p={p}");
        } else {
            let bound = numerics::tolerance::<T>(terms, 1.0) * want.abs();
            assert!(
                (got - want).abs() <= bound,
                "{what} p={p}: {got:.17e} against {want:.17e} (tolerance {bound:e})"
            );
        }
    }
}

fn scaled(scale: f64, row: [f64; 4]) -> [f64; 4] {
    row.map(|value| scale * value)
}

/// Coupled sectors of `V ← W` as `(dim(c), rows, columns)`, in block order.
type Blocks = [(f64, usize, usize)];

const U1: [(f64, usize, usize); 3] = [(1.0, 2, 5), (1.0, 3, 1), (1.0, 4, 2)];
const U1_BIG: [(f64, usize, usize); 3] = [(1.0, 6, 5), (1.0, 5, 3), (1.0, 4, 2)];
const FZ2: [(f64, usize, usize); 2] = [(1.0, 2, 5), (1.0, 3, 1)];
const SU2: [(f64, usize, usize); 3] = [(1.0, 2, 5), (2.0, 3, 1), (3.0, 4, 2)];

fn real_abs(indices: &[usize]) -> f64 {
    real_fill(indices).abs()
}

fn complex_abs(indices: &[usize]) -> f64 {
    complex_fill(indices).norm()
}

/// `[‖t‖_1, ‖t‖_2, ‖t‖_3, ‖t‖_Inf]` of the tensor whose block `b` holds
/// `scale(b) * fill(i, j)`, by the defining formula. The largest scale is
/// factored out first, so the formula neither overflows nor flushes to zero.
fn expected(
    blocks: &Blocks,
    scale: impl Fn(usize) -> f64,
    abs_fill: fn(&[usize]) -> f64,
) -> [f64; 4] {
    let entries: Vec<(f64, f64, f64)> = blocks
        .iter()
        .enumerate()
        .flat_map(|(block, &(dim, rows, columns))| {
            let scale = &scale;
            (0..rows).flat_map(move |i| {
                (0..columns).map(move |j| (dim, scale(block), abs_fill(&[i, j])))
            })
        })
        .collect();
    let top = entries.iter().map(|&(_, s, _)| s).fold(0.0, f64::max);
    let power = |p: f64| {
        let sum: f64 = entries
            .iter()
            .map(|&(dim, s, value)| dim * (s / top * value).powf(p))
            .sum();
        top * sum.powf(1.0 / p)
    };
    let max = entries
        .iter()
        .map(|&(_, s, value)| s * value)
        .fold(0.0, f64::max);
    [power(1.0), power(2.0), power(3.0), max]
}

#[test]
#[allow(clippy::excessive_precision)]
fn formula_and_symtn_reproduce_tensorkit_at_unit_scale() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // TensorKit f87ca7f `[norm(t, p) for p in (1, 2, 3, Inf)]`.
    let u1_big_f64 = [91.75, 13.975424859373685, 7.749999999999999, 3.5];
    let su2_f64 = [55.5, 9.656603957913982, 5.588779616741119, 2.5];
    let fz2_c64 = [
        21.152860776507886,
        6.002603601771485,
        3.9970711168729895,
        2.1830311495716224,
    ];
    assert_rows::<f64>(
        "formula u1big",
        expected(&U1_BIG, |_| 1.0, real_abs),
        u1_big_f64,
        53,
    );
    assert_rows::<f64>(
        "formula su2",
        expected(&SU2, |_| 1.0, real_abs),
        su2_f64,
        21,
    );
    assert_rows::<f64>(
        "formula fz2",
        expected(&FZ2, |_| 1.0, complex_abs),
        fz2_c64,
        13,
    );

    assert_rows::<f64>(
        "u1big",
        norms!(engine, tensor!(engine, u1_big(), f64, |_, i| real_fill(i))),
        u1_big_f64,
        53,
    );
    assert_rows::<f64>(
        "su2",
        norms!(engine, tensor!(engine, su2(), f64, |_, i| real_fill(i))),
        su2_f64,
        21,
    );
    assert_rows::<Complex64>(
        "fz2",
        norms!(
            engine,
            tensor!(engine, fz2(), Complex64, |_, i| complex_fill(i))
        ),
        fz2_c64,
        13,
    );
}

#[test]
fn unique_fusion_norms_rescale_at_1e200_and_1e_minus_200() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // TensorKit agrees here: Julia's `generic_norm2` (U1, 21 entries) and
    // OpenBLAS `nrm2` (U1_BIG, 53 entries, past NRM2_CUTOFF) both rescale.
    for s in [1e200, 1e-200] {
        let f = tensor!(engine, u1(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(
            &format!("u1 f64 x{s:e}"),
            norms!(engine, f),
            expected(&U1, |_| s, real_abs),
            21,
        );
        let c = tensor!(engine, u1(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(
            &format!("u1 c64 x{s:e}"),
            norms!(engine, c),
            expected(&U1, |_| s, complex_abs),
            21,
        );
        let f = tensor!(engine, u1_big(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(
            &format!("u1big f64 x{s:e}"),
            norms!(engine, f),
            expected(&U1_BIG, |_| s, real_abs),
            53,
        );
        let c = tensor!(engine, u1_big(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(
            &format!("u1big c64 x{s:e}"),
            norms!(engine, c),
            expected(&U1_BIG, |_| s, complex_abs),
            53,
        );
    }
}

#[test]
fn fermionic_norms_rescale_across_the_range() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for s in [1e200, 1e-200] {
        let f = tensor!(engine, fz2(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(
            &format!("fz2 f64 x{s:e}"),
            norms!(engine, f),
            expected(&FZ2, |_| s, real_abs),
            13,
        );
        let c = tensor!(engine, fz2(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(
            &format!("fz2 c64 x{s:e}"),
            norms!(engine, c),
            expected(&FZ2, |_| s, complex_abs),
            13,
        );
    }
}

#[test]
fn weighted_su2_norms_stay_representable_where_tensorkit_does_not() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // TensorKit f87ca7f prints, for p = (1, 2, 3, Inf):
    //   x1e200:  [5.55e201, Inf, Inf, 2.4999999999999998e200] (c64 alike)
    //   x1e-200: [5.550000000000001e-199, 0.0, 0.0, 2.5e-200]
    // Its p = 1 and Inf entries agree with the formula; p = 2 and 3 are its
    // unscaled `dim(c) * norm(b)^p` overflowing or flushing.
    for s in [1e200, 1e-200] {
        let f = tensor!(engine, su2(), f64, |_, i| s * real_fill(i));
        let want = expected(&SU2, |_| s, real_abs);
        assert_rows::<f64>(&format!("su2 f64 x{s:e}"), norms!(engine, f), want, 21);
        let c = tensor!(engine, su2(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(
            &format!("su2 c64 x{s:e}"),
            norms!(engine, c),
            expected(&SU2, |_| s, complex_abs),
            21,
        );
        // A lazy adjoint reads its parent's payload.
        assert_rows::<f64>(
            &format!("lazy su2 f64 x{s:e}"),
            norms!(engine, f.adjoint(engine).unwrap()),
            want,
            21,
        );
    }
}

#[test]
fn single_precision_norms_near_the_f32_range_edges() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // SU(2): TensorKit adds `dim(c) * norm(b)^2` in Float32 and prints
    // [5.55f31, Inf, Inf, 2.5f30] at 1e30 and [5.55f-29, 0, 0, 2.5f-30] at
    // 1e-30; symtn accumulates in f64, so the formula applies.
    for s in [1e30_f32, 1e-30] {
        let wide = f64::from(s);
        let f = tensor!(engine, u1(), f32, |_, i| s * real_fill(i) as f32);
        assert_rows::<f32>(
            &format!("u1 f32 x{s:e}"),
            norms!(engine, f),
            expected(&U1, |_| wide, real_abs),
            21,
        );
        let c = tensor!(engine, u1(), Complex32, |_, i| {
            let z = complex_fill(i);
            Complex32::new(z.re as f32, z.im as f32) * s
        });
        assert_rows::<Complex32>(
            &format!("u1 c32 x{s:e}"),
            norms!(engine, c),
            expected(&U1, |_| wide, complex_abs),
            21,
        );
        let f = tensor!(engine, su2(), f32, |_, i| s * real_fill(i) as f32);
        assert_rows::<f32>(
            &format!("su2 f32 x{s:e}"),
            norms!(engine, f),
            expected(&SU2, |_| wide, real_abs),
            21,
        );
    }
}

#[test]
fn mixed_magnitudes_keep_the_dominant_block() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // Only the charge +1 block (block 2) is scaled.
    let plus_one =
        |trees: &symtn::typed::BlockFusionTrees<U1Irrep>| *trees.coupled() == U1Irrep::new(1);
    for s in [1e200, 1e-200] {
        let t = tensor!(engine, u1(), f64, |trees, i| if plus_one(trees) {
            s * real_fill(i)
        } else {
            real_fill(i)
        });
        let want = expected(&U1, |block| if block == 2 { s } else { 1.0 }, real_abs);
        assert_rows::<f64>(&format!("u1 +1 block x{s:e}"), norms!(engine, t), want, 21);
    }
}

#[test]
fn zero_and_empty_payloads_have_zero_norm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let zeros: TensorMap<SU2FusionRule, Complex64> = {
        let (v, w) = su2();
        TensorMap::zeros(engine, &host_runtime(), [&v], [&w]).unwrap()
    };
    assert_rows::<f64>("zeros", norms!(engine, zeros), [0.0; 4], 21);

    // TensorKit: `U1Space(0 => 2) ← U1Space(1 => 3)` has no block and every
    // norm is 0.0.
    let rule = Arc::new(U1FusionRule);
    let only_zero = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)]).unwrap();
    let only_one = GradedSpace::try_new(rule, [(U1Irrep::new(1), 3)]).unwrap();
    let empty: TensorMap<U1FusionRule, f64> =
        TensorMap::zeros(engine, &host_runtime(), [&only_zero], [&only_one]).unwrap();
    assert_rows::<f64>("empty", norms!(engine, empty), [0.0; 4], 1);
}

#[test]
fn compact_diagonal_norms_rescale_with_quantum_dimensions() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // Hand oracle: spins 0, 1/2, 1 carry dim 1, 2, 3, so
    //   Σ dim·v   = 1·5 + 2·10 + 3·9       = 52
    //   Σ dim·v²  = 1·13 + 2·42 + 3·25     = 172
    //   Σ dim·v³  = 1·35 + 2·190 + 3·81    = 658
    //   max v     = 5
    let (bond, _) = su2();
    for s in [1e200, 1e-200] {
        let compact = TensorMap::<_, f64>::diagonal(
            engine,
            &host_runtime(),
            &bond,
            [
                (0, vec![2.0, 3.0]),
                (1, vec![1.0, 4.0, 5.0]),
                (2, vec![1.0, 2.0, 2.0, 4.0]),
            ]
            .map(|(twice_spin, values)| SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(twice_spin),
                values: values.into_iter().map(|v| s * v).collect(),
            }),
        )
        .unwrap();
        assert!(symtn::expert::diagonal_spectrum(engine, &compact)
            .unwrap()
            .is_some());
        assert_rows::<f64>(
            &format!("compact su2 x{s:e}"),
            norms!(engine, compact),
            scaled(s, [52.0, 172f64.sqrt(), 658f64.cbrt(), 5.0]),
            9,
        );
    }
}
