//! `norm(p)` keeps a representable result when the unscaled power sum
//! would overflow or underflow (#1437).
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
//! # Oracle provenance
//!
//! Constants were printed by TensorKit main `f87ca7f` (version 0.17.1) on
//! Julia 1.11.6 from tensors filled block by
//! block with the fills of `tk_norm_p.rs`, multiplied by the named scale:
//!
//! ```julia
//! t = zeros(T, V ← W); for (c, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
//!     b[i, j] = s * fill(i, j)
//! end
//! [norm(t, p) for p in (1, 2, 3, Inf)]
//! ```
//!
//! Rows are indexed `[p = 1, 2, 3, Inf]`.

use std::sync::Arc;

use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{Complex32, Complex64};
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

use numerics::Numeric;

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

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
    ($spaces:expr, $d:ty, $fill:expr) => {{
        let (v, w) = $spaces;
        let tensor: TensorMap<_, $d> =
            TensorMap::from_subblock_fn(&host_runtime(), [&v], [&w], $fill).unwrap();
        tensor
    }};
}

/// `[norm(1), norm(2), norm(3), norm(Inf)]`.
macro_rules! norms {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        [
            tensor.norm(1.0).unwrap(),
            tensor.norm(2.0).unwrap(),
            tensor.norm(3.0).unwrap(),
            tensor.norm(f64::INFINITY).unwrap(),
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

#[test]
fn unique_fusion_norms_match_tensorkit_at_1e200_and_1e_minus_200() {
    let cases: [(&str, f64, [f64; 4], [f64; 4]); 2] = [
        (
            "1e200",
            1e200,
            [
                2.5e201,
                6.204836822995429e200,
                4.0804042115227644e200,
                2.4999999999999998e200,
            ],
            [
                3.608814448804254e201,
                8.145167278822456e200,
                5.060139147568353e200,
                2.6487025125521362e200,
            ],
        ),
        (
            "1e-200",
            1e-200,
            [
                2.5000000000000002e-199,
                6.204836822995428e-200,
                4.0804042115227645e-200,
                2.5e-200,
            ],
            [
                3.6088144488042534e-199,
                8.145167278822455e-200,
                5.060139147568353e-200,
                2.648702512552136e-200,
            ],
        ),
    ];
    for (label, s, real, complex) in cases {
        let f = tensor!(u1(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(&format!("u1 f64 x{label}"), norms!(f), real, 21);
        let c = tensor!(u1(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(&format!("u1 c64 x{label}"), norms!(c), complex, 21);
    }

    // Past NRM2_CUTOFF: TensorKit's value comes from OpenBLAS `dnrm2`/`dznrm2`.
    let big: [(f64, [f64; 4], [f64; 4]); 3] = [
        (
            1.0,
            [91.75, 13.975424859373685, 7.749999999999999, 3.5],
            [
                121.05382556737885,
                17.486602014113547,
                9.417334153387264,
                3.6763602924631855,
            ],
        ),
        (
            1e200,
            [
                9.174999999999999e201,
                1.3975424859373685e201,
                7.750000000000001e200,
                3.5e200,
            ],
            [
                1.2105382556737885e202,
                1.7486602014113548e201,
                9.417334153387263e200,
                3.6763602924631856e200,
            ],
        ),
        (
            1e-200,
            [
                9.175e-199,
                1.3975424859373686e-199,
                7.749999999999999e-200,
                3.4999999999999996e-200,
            ],
            [
                1.2105382556737892e-198,
                1.7486602014113548e-199,
                9.417334153387264e-200,
                3.676360292463185e-200,
            ],
        ),
    ];
    for (s, real, complex) in big {
        let f = tensor!(u1_big(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(&format!("u1big f64 x{s:e}"), norms!(f), real, 53);
        let c = tensor!(u1_big(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(&format!("u1big c64 x{s:e}"), norms!(c), complex, 53);
    }
}

#[test]
fn fermionic_norms_match_tensorkit_across_the_range() {
    let rows: [(f64, [f64; 4], [f64; 4]); 3] = [
        (
            1.0,
            [12.0, 3.840572873934304, 2.75068853282797, 2.0],
            [
                21.152860776507886,
                6.002603601771485,
                3.9970711168729895,
                2.1830311495716224,
            ],
        ),
        (
            1e200,
            [
                1.1999999999999998e201,
                3.840572873934304e200,
                2.7506885328279704e200,
                2.0e200,
            ],
            [
                2.115286077650789e201,
                6.002603601771484e200,
                3.9970711168729895e200,
                2.1830311495716225e200,
            ],
        ),
        (
            1e-200,
            [
                1.2000000000000002e-199,
                3.840572873934304e-200,
                2.7506885328279704e-200,
                2.0e-200,
            ],
            [
                2.1152860776507893e-199,
                6.002603601771484e-200,
                3.99707111687299e-200,
                2.1830311495716225e-200,
            ],
        ),
    ];
    for (s, real, complex) in rows {
        let f = tensor!(fz2(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(&format!("fz2 f64 x{s:e}"), norms!(f), real, 13);
        let c = tensor!(fz2(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(&format!("fz2 c64 x{s:e}"), norms!(c), complex, 13);
    }
}

/// TensorKit's in-range SU(2) rows (`tk_norm_p.rs`), the base of the hand
/// oracle `norm(s * t) = s * norm(t)`.
const SU2_F64: [f64; 4] = [55.5, 9.656603957913982, 5.588779616741119, 2.5];
const SU2_C64: [f64; 4] = [
    70.83774683619109,
    11.637090486887175,
    6.51405041399105,
    2.6487025125521364,
];

#[test]
fn weighted_su2_norms_stay_representable_where_tensorkit_does_not() {
    // TensorKit f87ca7f prints, for p = (1, 2, 3, Inf):
    //   x1e200:  [5.55e201, Inf, Inf, 2.4999999999999998e200] (c64 alike)
    //   x1e-200: [5.550000000000001e-199, 0.0, 0.0, 2.5e-200]
    // Its p = 1 and Inf entries agree with the hand oracle below; p = 2 and 3
    // are its unscaled `dim(c) * norm(b)^p` overflowing or flushing.
    for s in [1e200, 1e-200] {
        let f = tensor!(su2(), f64, |_, i| s * real_fill(i));
        assert_rows::<f64>(
            &format!("su2 f64 x{s:e}"),
            norms!(f),
            scaled(s, SU2_F64),
            21,
        );
        let c = tensor!(su2(), Complex64, |_, i| complex_fill(i) * s);
        assert_rows::<Complex64>(
            &format!("su2 c64 x{s:e}"),
            norms!(c),
            scaled(s, SU2_C64),
            21,
        );
        // A lazy adjoint reads its parent's payload.
        assert_rows::<f64>(
            &format!("lazy su2 f64 x{s:e}"),
            norms!(f.adjoint().unwrap()),
            scaled(s, SU2_F64),
            21,
        );
    }
}

#[test]
fn single_precision_norms_near_the_f32_range_edges() {
    // U(1): TensorKit's own values (Float32 results, 21 entries, generic_norm2
    // accumulating in Float64).
    let u1_rows: [(f32, [f64; 4], [f64; 4]); 2] = [
        (
            1e30,
            [2.5e31, 6.204837e30, 4.0804042e30, 2.5e30],
            [3.6088148e31, 8.1451673e30, 5.060139e30, 2.6487024e30],
        ),
        (
            1e-30,
            [2.4999999e-29, 6.204837e-30, 4.0804042e-30, 2.5e-30],
            [3.6088148e-29, 8.145167e-30, 5.060139e-30, 2.6487025e-30],
        ),
    ];
    for (s, real, complex) in u1_rows {
        let f = tensor!(u1(), f32, |_, i| s * real_fill(i) as f32);
        assert_rows::<f32>(&format!("u1 f32 x{s:e}"), norms!(f), real, 21);
        let c = tensor!(u1(), Complex32, |_, i| {
            let z = complex_fill(i);
            Complex32::new(z.re as f32, z.im as f32) * s
        });
        assert_rows::<Complex32>(&format!("u1 c32 x{s:e}"), norms!(c), complex, 21);
    }

    // SU(2): TensorKit adds `dim(c) * norm(b)^2` in Float32 and prints
    // [5.55f31, Inf, Inf, 2.5f30] at 1e30 and [5.55f-29, 0, 0, 2.5f-30] at
    // 1e-30; symtn accumulates in f64, so the hand oracle applies.
    for s in [1e30_f32, 1e-30] {
        let f = tensor!(su2(), f32, |_, i| s * real_fill(i) as f32);
        assert_rows::<f32>(
            &format!("su2 f32 x{s:e}"),
            norms!(f),
            scaled(f64::from(s), SU2_F64),
            21,
        );
    }
}

#[test]
fn mixed_magnitudes_match_tensorkit() {
    let plus_one =
        |trees: &symtn::typed::BlockFusionTrees<U1Irrep>| *trees.coupled() == U1Irrep::new(1);
    let huge = tensor!(u1(), f64, |trees, i| if plus_one(trees) {
        1e200 * real_fill(i)
    } else {
        real_fill(i)
    });
    assert_rows::<f64>(
        "u1 +1 block x1e200",
        norms!(huge),
        [
            1.3e201,
            4.8733971724044814e200,
            3.6120225619516305e200,
            2.4999999999999998e200,
        ],
        21,
    );
    let tiny = tensor!(u1(), f64, |trees, i| if plus_one(trees) {
        1e-200 * real_fill(i)
    } else {
        real_fill(i)
    });
    assert_rows::<f64>(
        "u1 +1 block x1e-200",
        norms!(tiny),
        [12.0, 3.840572873934304, 2.75068853282797, 2.0],
        21,
    );
}

#[test]
fn non_finite_entries_follow_julia() {
    let at_plus_one = |value: f64, second: f64| {
        tensor!(u1(), f64, move |trees: &symtn::typed::BlockFusionTrees<
            U1Irrep,
        >,
                                 i: &[usize]| {
            match (*trees.coupled() == U1Irrep::new(1), i) {
                (true, [0, 0]) => value,
                (true, [1, 0]) => second,
                _ => real_fill(i),
            }
        })
    };
    let nan = [f64::NAN; 4];
    let inf = [f64::INFINITY; 4];
    // TensorKit: NaN -> all NaN; ±Inf -> all Inf; Inf beside NaN -> all NaN.
    assert_rows::<f64>("NaN", norms!(at_plus_one(f64::NAN, 1.5)), nan, 21);
    assert_rows::<f64>("Inf", norms!(at_plus_one(f64::INFINITY, 1.5)), inf, 21);
    assert_rows::<f64>("-Inf", norms!(at_plus_one(f64::NEG_INFINITY, 1.5)), inf, 21);
    assert_rows::<f64>(
        "Inf and NaN",
        norms!(at_plus_one(f64::INFINITY, f64::NAN)),
        nan,
        21,
    );

    // |NaN + Inf i| = hypot(NaN, Inf) = Inf, so Julia's maxabs is Inf and
    // generic_norm2 returns it: TensorKit prints all Inf on these 21 entries.
    // (Past NRM2_CUTOFF OpenBLAS `dznrm2` returns NaN for p = 2 instead; symtn
    // follows the generic_norm2 rule at every length.)
    let mixed = tensor!(u1(), Complex64, |trees, i| {
        if *trees.coupled() == U1Irrep::new(1) && i == [0, 0] {
            Complex64::new(f64::NAN, f64::INFINITY)
        } else {
            complex_fill(i)
        }
    });
    assert_rows::<Complex64>("NaN + Inf i", norms!(mixed), inf, 21);

    // Weighted SU(2): TensorKit prints all NaN and all Inf.
    let su2_with = |value: f64| {
        tensor!(su2(), f64, move |trees: &symtn::typed::BlockFusionTrees<
            SU2Irrep,
        >,
                                  i: &[usize]| {
            if *trees.coupled() == SU2Irrep::from_twice_spin(2) && i == [0, 0] {
                value
            } else {
                real_fill(i)
            }
        })
    };
    assert_rows::<f64>("su2 NaN", norms!(su2_with(f64::NAN)), nan, 21);
    assert_rows::<f64>("su2 Inf", norms!(su2_with(f64::INFINITY)), inf, 21);
}

#[test]
fn zero_and_empty_payloads_have_zero_norm() {
    let zeros: TensorMap<SU2FusionRule, Complex64> = {
        let (v, w) = su2();
        TensorMap::zeros(&host_runtime(), [&v], [&w]).unwrap()
    };
    assert_rows::<f64>("zeros", norms!(zeros), [0.0; 4], 21);

    // TensorKit: `U1Space(0 => 2) ← U1Space(1 => 3)` has no block and every
    // norm is 0.0.
    let rule = Arc::new(U1FusionRule);
    let only_zero = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)]).unwrap();
    let only_one = GradedSpace::try_new(rule, [(U1Irrep::new(1), 3)]).unwrap();
    let empty: TensorMap<U1FusionRule, f64> =
        TensorMap::zeros(&host_runtime(), [&only_zero], [&only_one]).unwrap();
    assert_rows::<f64>("empty", norms!(empty), [0.0; 4], 1);
}

#[test]
fn compact_diagonal_norms_rescale_with_quantum_dimensions() {
    // Hand oracle: spins 0, 1/2, 1 carry dim 1, 2, 3, so
    //   Σ dim·v   = 1·5 + 2·10 + 3·9       = 52
    //   Σ dim·v²  = 1·13 + 2·42 + 3·25     = 172
    //   Σ dim·v³  = 1·35 + 2·190 + 3·81    = 658
    //   max v     = 5
    let (bond, _) = su2();
    for s in [1e200, 1e-200] {
        let compact = TensorMap::<_, f64>::diagonal(
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
        assert!(symtn::expert::diagonal_spectrum(&compact)
            .unwrap()
            .is_some());
        assert_rows::<f64>(
            &format!("compact su2 x{s:e}"),
            norms!(compact),
            scaled(s, [52.0, 172f64.sqrt(), 658f64.cbrt(), 5.0]),
            9,
        );
    }
}
