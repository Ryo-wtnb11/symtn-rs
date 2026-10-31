//! Every named factorization result carries the factors its fields name:
//! composed in the documented order they reconstruct the input
//! (`u∘s∘vh`, `q∘r`, `l∘q`, `w∘p`, `p∘wh`, `t∘v = v∘d`), and the factor
//! documented as an isometry is one (`x†x = 1` or `x x† = 1` on its own
//! space). Owned and lazy-adjoint inputs, of a rank-3 `[v, v] ← [v]`
//! fixture with random data, over U(1), SU(2) and fZ2×U(1), f64 and c64.
//! A lazy-adjoint input is reconstructed against its materialized adjoint.
//!
//! The isometry and reconstruction laws hold in every valid gauge, so no
//! factor values are pinned.

use crate::fixtures;

use num_complex::Complex64;
use std::sync::Arc;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{
    Eig, Eigh, GradedSpace, HermitianTol, LeftPolar, Lq, Qr, RightPolar, Svd, TensorMap,
};

use fixtures::host_runtime;

const TOL: f64 = 1e-10;

/// `‖actual - expected‖ ≤ TOL · max(‖expected‖, 1)`.
macro_rules! assert_close {
    ($engine:ident, $what:expr, $actual:expr, $expected:expr, $one:expr) => {{
        let (actual, expected) = (&$actual, &$expected);
        let error = actual
            .axpby($engine, $one, expected, -$one)
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        let scale = expected.norm($engine, 2.0).unwrap().max(1.0);
        assert!(error <= TOL * scale, "{}: {error:e}", $what);
    }};
}

/// `x†x` is the identity on the domain of `x`.
macro_rules! assert_isometry {
    ($engine:ident, $what:expr, $runtime:expr, $x:expr, $one:expr) => {{
        let x = &$x;
        let gram = x.adjoint($engine).unwrap().compose($engine, x).unwrap();
        let identity = TensorMap::isomorphism($engine, $runtime, &x.domain(), &x.domain()).unwrap();
        assert_close!($engine, $what, gram, identity, $one);
    }};
}

/// `x x†` is the identity on the codomain of `x`.
macro_rules! assert_coisometry {
    ($engine:ident, $what:expr, $runtime:expr, $x:expr, $one:expr) => {{
        let x = &$x;
        let x_adjoint = x.adjoint($engine).unwrap();
        let gram = x.compose($engine, &x_adjoint).unwrap();
        let identity =
            TensorMap::isomorphism($engine, $runtime, &x.codomain(), &x.codomain()).unwrap();
        assert_close!($engine, $what, gram, identity, $one);
    }};
}

/// Every factorization of `tall: [v, v] ← [v]`, `wide = [v] ← [v, v]`, a
/// Hermitian endomorphism and a general endomorphism, owned and through a
/// lazy adjoint. `eig` returns complex factors, so it is checked on complex
/// input only (`$eig`), where `t∘v` stays in one dtype.
macro_rules! check_all {
    ($engine:ident, $label:expr, $runtime:expr, $v:expr, $d:ty, $one:expr $(, $eig:ident)?) => {{
        let (runtime, v, one) = ($runtime, $v, $one);
        let what = |name: &str, lazy: bool| {
            format!("{} {} {name}", $label, if lazy { "adjoint" } else { "owned" })
        };
        let tall = TensorMap::<_, $d>::rand_with_seed($engine, runtime, [v, v], [v], 2312).unwrap();
        let wide = TensorMap::<_, $d>::rand_with_seed($engine, runtime, [v], [v, v], 2313).unwrap();
        // Real symmetric blocks with a dominant diagonal: Hermitian for both
        // dtypes.
        let herm = TensorMap::<_, $d>::from_subblock_fn($engine, runtime, [v], [v], |_, ij| {
            let (i, j) = (ij[0] as f64, ij[1] as f64);
            <$d>::from(1.0 / (1.0 + i + j) + if ij[0] == ij[1] { 0.37 * (1.0 + i) } else { 0.0 })
        })
        .unwrap();
        // `(input, its materialized value, lazy)`: an owned tensor, then the
        // lazy adjoint of the opposite shape, which has the same leg counts.
        let inputs = [
            (tall.clone(), wide.clone(), herm.clone(), false),
            (
                wide.adjoint($engine).unwrap(),
                tall.adjoint($engine).unwrap(),
                herm.adjoint($engine).unwrap(),
                true,
            ),
        ];
        for (tall, wide, herm, lazy) in inputs {
            let (tall_m, wide_m) = (tall.materialize($engine).unwrap(), wide.materialize($engine).unwrap());
            let herm_m = herm.materialize($engine).unwrap();

            for (name, Svd { u, s, vh }) in [
                ("svd_compact", tall.svd_compact($engine, &[0, 1], &[2]).unwrap()),
                ("svd_full", tall.svd_full($engine, &[0, 1], &[2]).unwrap()),
            ] {
                let product = u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap();
                assert_close!($engine, what(name, lazy), product, tall_m, one);
                assert_isometry!($engine, what(name, lazy) + " u", runtime, u, one);
                assert_coisometry!($engine, what(name, lazy) + " vh", runtime, vh, one);
            }
            for (name, Qr { q, r }) in [
                ("qr_compact", tall.qr_compact($engine, &[0, 1], &[2]).unwrap()),
                ("qr_full", tall.qr_full($engine, &[0, 1], &[2]).unwrap()),
            ] {
                assert_close!($engine, what(name, lazy), q.compose($engine, &r).unwrap(), tall_m, one);
                assert_isometry!($engine, what(name, lazy) + " q", runtime, q, one);
            }
            for (name, Lq { l, q }) in [
                ("lq_compact", wide.lq_compact($engine, &[0], &[1, 2]).unwrap()),
                ("lq_full", wide.lq_full($engine, &[0], &[1, 2]).unwrap()),
            ] {
                assert_close!($engine, what(name, lazy), l.compose($engine, &q).unwrap(), wide_m, one);
                assert_coisometry!($engine, what(name, lazy) + " q", runtime, q, one);
            }
            let LeftPolar { w, p } = tall.left_polar($engine, &[0, 1], &[2]).unwrap();
            assert_close!($engine, what("left_polar", lazy), w.compose($engine, &p).unwrap(), tall_m, one);
            assert_isometry!($engine, what("left_polar", lazy) + " w", runtime, w, one);
            let RightPolar { p, wh } = wide.right_polar($engine, &[0], &[1, 2]).unwrap();
            assert_close!($engine, what("right_polar", lazy), p.compose($engine, &wh).unwrap(), wide_m, one);
            assert_coisometry!($engine, what("right_polar", lazy) + " wh", runtime, wh, one);

            let Eigh { d, v: vectors } = herm.eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT).unwrap();
            assert_close!($engine,
                what("eigh_full", lazy),
                herm_m.compose($engine, &vectors).unwrap(),
                vectors.compose($engine, &d).unwrap(),
                one
            );
            assert_isometry!($engine, what("eigh_full", lazy) + " v", runtime, vectors, one);
            $(
                let endo = TensorMap::<_, $d>::rand_with_seed($engine, runtime, [v], [v], 2314).unwrap();
                let endo = if lazy { endo.adjoint($engine).unwrap() } else { endo };
                let $eig = endo.eig_full($engine, &[0], &[1]).unwrap();
                let Eig { d, v: vectors } = $eig;
                let endo_m = endo.materialize($engine).unwrap();
                assert_close!($engine,
                    what("eig_full", lazy),
                    endo_m.compose($engine, &vectors).unwrap(),
                    vectors.compose($engine, &d).unwrap(),
                    one
                );
            )?
        }
    }};
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-1, 1), (0, 2), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(0, 2), (1, 2), (2, 1)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
    )
    .unwrap()
}

fn fz2_u1_leg() -> GradedSpace<ProductFusionRule<FermionParityFusionRule, U1FusionRule>> {
    let rule =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    GradedSpace::try_new(
        Arc::new(rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
        ],
    )
    .unwrap()
}

#[test]
fn named_factors_reconstruct_the_input_and_name_the_isometric_factor() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = host_runtime();
    let one = Complex64::new(1.0, 0.0);
    check_all!(engine, "u1 f64", &runtime, &u1_leg(), f64, 1.0);
    check_all!(engine, "u1 c64", &runtime, &u1_leg(), Complex64, one, eig);
    check_all!(engine, "su2 f64", &runtime, &su2_leg(), f64, 1.0);
    check_all!(engine, "su2 c64", &runtime, &su2_leg(), Complex64, one, eig);
    check_all!(engine, "fz2u1 f64", &runtime, &fz2_u1_leg(), f64, 1.0);
    check_all!(
        engine,
        "fz2u1 c64",
        &runtime,
        &fz2_u1_leg(),
        Complex64,
        one,
        eig
    );
}
