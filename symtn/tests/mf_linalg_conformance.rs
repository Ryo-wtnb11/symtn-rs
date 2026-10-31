//! Multiplicity-free public linear-algebra conformance (#1002).
//!
//! These are deliberately public-API tests. They cover the residual provider
//! fixtures in #1002, not every constructible `FusionRule` implementation.

include!("common/predicate_chains.rs");
include!("common/predicate_chain_coefficients.rs");

use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep, ZNFusionRule,
};
use symtn::sector::{
    MultiplicityFreeRigidSymbols, PackedProductCodec, ProductSectorLayout, SectorCodec,
    Su2SectorLayout, U1SectorLayout,
};
use symtn::typed::{Complex64, GradedSpace, TensorMap, Truncation};
use symtn::typed::{Eig, Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};

type Fz2SectorLayout = symtn::sector::Fz2SectorLayout;
type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Su2Codec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type Fz2U1Su2Rule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, Fz2U1Su2Codec>;

use crate::fixtures;

use fixtures::host_runtime;

macro_rules! assert_close {
    ($engine:ident, $actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        let error = actual
            .axpby($engine, 1.0, expected, -1.0)
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        assert!(
            error <= 1e-9 * (1.0 + expected.norm($engine, 2.0).unwrap()),
            "residual {error}"
        );
    }};
}

macro_rules! assert_complex_close {
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        assert_eq!(
            actual.dense_data().unwrap().len(),
            expected.dense_data().unwrap().len()
        );
        for (actual, expected) in actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
        {
            assert!(
                (*actual - *expected).norm() <= 1e-9 * (1.0 + expected.norm()),
                "{actual} != {expected}"
            );
        }
    }};
}

macro_rules! assert_provider {
    ($provider:expr; $($tensor:expr),+ $(,)?) => {
        $(assert!(std::ptr::eq($tensor.provider(), $provider.as_ref()));)+
    };
}

/// The rectangular rows deliberately use a rank-two diagonal embedded in a
/// larger block.  This makes both null directions and polar isometry laws
/// observable without pinning a dense-factor gauge.
macro_rules! factor_conformance {
    ($engine:ident, $name:expr, $rule:expr, $pairs:expr) => {{
        let rt = host_runtime();
        let provider = Arc::new($rule);
        let pairs = $pairs;
        let endo_space = GradedSpace::try_new(Arc::clone(&provider), pairs.clone()).unwrap();
        let tall_pairs: Vec<_> = pairs
            .iter()
            .map(|(sector, degeneracy)| (sector.clone(), degeneracy + 1))
            .collect();
        let wide_pairs: Vec<_> = pairs
            .iter()
            .map(|(sector, degeneracy)| (sector.clone(), *degeneracy))
            .collect();
        let tall_space = GradedSpace::try_new(Arc::clone(&provider), tall_pairs).unwrap();
        let wide_space = GradedSpace::try_new(Arc::clone(&provider), wide_pairs).unwrap();
        let tall: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($engine, &rt, [&tall_space], [&wide_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 2.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($engine, &rt, [&wide_space], [&tall_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 2.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();

        let Svd { u, s, vh } = tall.svd_compact($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; u, s, vh);
        assert_close!($engine, &u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap(), &tall);
        let Svd { u, s, vh } = tall.svd_full($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; u, s, vh);
        assert_close!($engine, &u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap(), &tall);
        // Truncated SVD: svd_compact -> diagview -> find_truncated -> restrict.
        let Svd { u, s, vh } = tall.svd_compact($engine, &[0], &[1]).unwrap();
        let found = s.domain()[0]
            .find_truncated(&s.diagview($engine).unwrap(), &Truncation::rank(1))
            .unwrap();
        let u = u.restrict_leg($engine, &[(u.codomain_rank(), &found.selection)]).unwrap();
        let s = s.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)]).unwrap();
        let vh = vh.restrict_leg($engine, &[(0, &found.selection)]).unwrap();
        assert_provider!(provider; u, s, vh);
        let reconstructed = u.compose($engine, &s).unwrap().compose($engine, &vh).unwrap();
        let error = reconstructed.axpby($engine, 1.0, &tall, -1.0).unwrap().norm($engine, 2.0).unwrap();
        assert!((error - found.error).abs() <= 1e-9 * (1.0 + found.error));
        assert!(found.error > 0.0, $name);
        let singular_values = tall.svd_vals($engine, &[0], &[1]).unwrap();
        assert!(singular_values
            .iter()
            .all(|entry| entry.values == [4.0, 2.0]));
        let weighted_norm_squared: f64 = singular_values
            .iter()
            .map(|entry| {
                provider.dim_scalar(
                    SectorCodec::encode_sector(provider.as_ref(), &entry.sector).unwrap(),
                )
                    * entry.values.iter().map(|value| value * value).sum::<f64>()
            })
            .sum();
        assert!((tall.norm($engine, 2.0).unwrap().powi(2) - weighted_norm_squared).abs() <= 1e-9);

        let Qr { q, r } = tall.qr_compact($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; q, r);
        assert_close!($engine, &q.compose($engine, &r).unwrap(), &tall);
        let id = TensorMap::isomorphism($engine, &rt, q.domain().iter(), q.domain().iter()).unwrap();
        assert_close!($engine, &q.adjoint($engine).unwrap().compose($engine, &q).unwrap(), &id);
        let Qr { q, r } = tall.qr_full($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; q, r);
        assert_close!($engine, &q.compose($engine, &r).unwrap(), &tall);
        let id = TensorMap::isomorphism($engine, &rt, q.domain().iter(), q.domain().iter()).unwrap();
        assert_close!($engine, &q.adjoint($engine).unwrap().compose($engine, &q).unwrap(), &id);

        let Lq { l, q } = wide.lq_compact($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; l, q);
        assert_close!($engine, &l.compose($engine, &q).unwrap(), &wide);
        let id = TensorMap::isomorphism($engine, &rt, q.codomain().iter(), q.codomain().iter()).unwrap();
        let q_adjoint = &q.adjoint($engine).unwrap();
        assert_close!($engine, &q.compose($engine, q_adjoint).unwrap(), &id);
        let Lq { l, q } = wide.lq_full($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; l, q);
        assert_close!($engine, &l.compose($engine, &q).unwrap(), &wide);
        let id = TensorMap::isomorphism($engine, &rt, q.codomain().iter(), q.codomain().iter()).unwrap();
        let q_adjoint = &q.adjoint($engine).unwrap();
        assert_close!($engine, &q.compose($engine, q_adjoint).unwrap(), &id);

        let left = tall.left_null($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; left);
        assert!(
            left.adjoint($engine)
                .unwrap()
                .compose($engine, &tall)
                .unwrap()
                .norm($engine, 2.0)
                .unwrap()
                <= 1e-9,
            $name
        );
        let left_id = TensorMap::isomorphism($engine, &rt, left.domain().iter(), left.domain().iter()).unwrap();
        assert_close!($engine, &left.adjoint($engine).unwrap().compose($engine, &left).unwrap(), &left_id);
        let right = wide.right_null($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; right);
        let right_adjoint = &right.adjoint($engine).unwrap();
        assert!(
            wide.compose($engine, right_adjoint)
                .unwrap()
                .norm($engine, 2.0)
                .unwrap()
                <= 1e-9,
            $name
        );
        let right_id = TensorMap::isomorphism($engine, &rt, right.codomain().iter(), right.codomain().iter()).unwrap();
        let right_adjoint = &right.adjoint($engine).unwrap();
        assert_close!($engine,
            &right.compose($engine, right_adjoint).unwrap(),
            &right_id
        );

        let LeftPolar { w, p } = tall.left_polar($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; w, p);
        assert_close!($engine, &w.compose($engine, &p).unwrap(), &tall);
        let id = TensorMap::isomorphism($engine, &rt, w.domain().iter(), w.domain().iter()).unwrap();
        assert_close!($engine, &w.adjoint($engine).unwrap().compose($engine, &w).unwrap(), &id);
        assert!(is_hermitian!($engine, p, 1e-10));
        assert!(p
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .flat_map(|s| &s.values)
            .all(|&x| x >= -1e-10));
        let RightPolar { p, wh: w } = wide.right_polar($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; p, w);
        assert_close!($engine, &p.compose($engine, &w).unwrap(), &wide);
        let id = TensorMap::isomorphism($engine, &rt, w.codomain().iter(), w.codomain().iter()).unwrap();
        let w_adjoint = &w.adjoint($engine).unwrap();
        assert_close!($engine, &w.compose($engine, w_adjoint).unwrap(), &id);
        assert!(is_hermitian!($engine, p, 1e-10));
        assert!(p
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .flat_map(|s| &s.values)
            .all(|&x| x >= -1e-10));

        let h: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($engine, &rt, [&endo_space], [&endo_space], |_, index| {
                [[3.0, 1.0], [1.0, 3.0]][index[0]][index[1]]
            })
            .unwrap();
        let Eigh { d, v } = h.eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT).unwrap();
        assert_provider!(provider; d, v);
        let v_adjoint = &v.adjoint($engine).unwrap();
        assert_close!($engine,
            &v.compose($engine, &d)
                .unwrap()
                .compose($engine, v_adjoint)
                .unwrap(),
            &h
        );
        assert!(h
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .all(|entry| entry.values == [2.0, 4.0]));
        let found = d.domain()[0]
            .find_truncated(&d.diagview($engine).unwrap(), &Truncation::rank(1))
            .unwrap();
        let trunc_d = d.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)]).unwrap();
        let trunc_v = v.restrict_leg($engine, &[(v.codomain_rank(), &found.selection)]).unwrap();
        assert_provider!(provider; trunc_d, trunc_v);
        let trunc_v_adjoint = &trunc_v.adjoint($engine).unwrap();
        let reconstructed = trunc_v
            .compose($engine, &trunc_d)
            .unwrap()
            .compose($engine, trunc_v_adjoint)
            .unwrap();
        let error = reconstructed.axpby($engine, 1.0, &h, -1.0).unwrap().norm($engine, 2.0).unwrap();
        assert!((error - found.error).abs() <= 1e-9 * (1.0 + found.error));
        assert!(found.error > 0.0, $name);

        let g: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($engine, &rt, [&endo_space], [&endo_space], |_, index| {
                [[3.0, 1.0], [0.0, 1.0]][index[0]][index[1]]
            })
            .unwrap();
        let Eig { d, v } = g.eig_full($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; d, v);
        let v_inv = &v.inv($engine, &[0], &[1]).unwrap();
        assert_complex_close!(
            &v.compose($engine, &d).unwrap().compose($engine, v_inv).unwrap(),
            &g.convert::<Complex64>($engine).unwrap()
        );
        assert!(g
            .eig_vals($engine, &[0], &[1])
            .unwrap()
            .iter()
            .all(|entry| entry.values == [1.0.into(), 3.0.into()]));
        let found = d.domain()[0]
            .find_truncated(&d.diagview($engine).unwrap(), &Truncation::rank(1))
            .unwrap();
        let trunc_d = d.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)]).unwrap();
        let trunc_v = v.restrict_leg($engine, &[(v.codomain_rank(), &found.selection)]).unwrap();
        assert_provider!(provider; trunc_d, trunc_v);
        assert_complex_close!(
            &g.convert::<Complex64>($engine).unwrap().compose($engine, &trunc_v).unwrap(),
            &trunc_v.compose($engine, &trunc_d).unwrap()
        );
        assert!(found.error > 0.0, $name);

        let id = TensorMap::isomorphism($engine, &rt, h.domain().iter(), h.domain().iter()).unwrap();
        let inverse = h.inv($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; inverse);
        assert_close!($engine, &h.compose($engine, &inverse).unwrap(), &id);
        let exponential = h.exp($engine, &[0], &[1]).unwrap();
        assert_provider!(provider; exponential);
        assert_close!($engine,
            &h.scale($engine, -1.0).unwrap()
                .exp($engine, &[0], &[1])
                .unwrap()
                .compose($engine, &exponential)
                .unwrap(),
            &id
        );
        let solved = h.solve($engine, &[0], &[1], &id, &[0], &[1]).unwrap();
        assert_provider!(provider; solved);
        assert_close!($engine, &solved, &inverse);
        let id_adjoint = &id.adjoint($engine).unwrap();
        let solved_right = h
            .adjoint($engine)
            .unwrap()
            .solve($engine, &[0], &[1], id_adjoint, &[0], &[1])
            .unwrap()
            .adjoint($engine)
            .unwrap();
        assert_provider!(provider; solved_right);
        assert_close!($engine, &solved_right, &inverse);
        let pseudo = tall.pinv($engine, &[0], &[1], 1e-12).unwrap();
        assert_provider!(provider; pseudo);
        assert_close!($engine,
            &tall.compose($engine, &pseudo).unwrap().compose($engine, &tall).unwrap(),
            &tall
        );
        let aa_plus = tall.compose($engine, &pseudo).unwrap();
        let a_plus_a = pseudo.compose($engine, &tall).unwrap();
        assert_close!($engine, &a_plus_a.compose($engine, &pseudo).unwrap(), &pseudo);
        assert_close!($engine, &aa_plus.adjoint($engine).unwrap(), &aa_plus);
        assert_close!($engine, &a_plus_a.adjoint($engine).unwrap(), &a_plus_a);
        let left_adjoint = &left.adjoint($engine).unwrap();
        let left_projector = left.compose($engine, left_adjoint).unwrap();
        let id = TensorMap::isomorphism($engine, &rt, tall.codomain().iter(), tall.codomain().iter()).unwrap();
        assert_close!($engine, &left_projector.axpby($engine, 1.0, &aa_plus, 1.0).unwrap(), &id);
        let diagonal: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($engine, &rt, [&endo_space], [&endo_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 9.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        assert!(diagonal.map_diagonal($engine, f64::sqrt).is_err());
        let diagonal_diagview = diagonal.diagview($engine).unwrap();
        let compact =
            TensorMap::diagonal($engine, &rt, &endo_space, diagonal_diagview).unwrap();
        let root = compact.map_diagonal($engine, f64::sqrt).unwrap();
        assert_provider!(provider; root);
        assert!(root.materialize($engine).unwrap().dense_data().unwrap().iter().any(|&value| value == 2.0));
        assert!(root.materialize($engine).unwrap().dense_data().unwrap().iter().any(|&value| value == 3.0));
        assert!(root
            .materialize($engine).unwrap().dense_data().unwrap()
            .iter()
            .all(|&value| value == 0.0 || value == 2.0 || value == 3.0));
        assert_close!($engine, &root.compose($engine, &root).unwrap(), &diagonal);
        let before = tall.dense_data().unwrap().to_vec();
        assert!(tall.pinv($engine, &[0], &[1], -1.0).is_err());
        assert_eq!(tall.dense_data().unwrap(), before.as_slice());
    }};
}

#[test]
fn public_multiplicity_free_linalg_conformance() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    factor_conformance!(
        engine,
        "U(1)",
        U1FusionRule,
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]
    );
    let z3 = ZNFusionRule::new(3).unwrap();
    let z3_pairs = [(z3.irrep(0), 2), (z3.irrep(1), 2)];
    factor_conformance!(engine, "Z3", z3, z3_pairs);
    factor_conformance!(
        engine,
        "CU1 vacuum/pseudo/charged",
        symtn::sector::CU1FusionRule,
        [
            (symtn::sector::CU1Irrep::VACUUM, 2),
            (symtn::sector::CU1Irrep::PSEUDOSCALAR, 2),
            (symtn::sector::CU1Irrep::from_twice_charge(1), 2),
        ]
    );
    factor_conformance!(
        engine,
        "fZ2 odd",
        FermionParityFusionRule,
        [(Z2Irrep::ODD, 2)]
    );
    factor_conformance!(
        engine,
        "SU2 nonzero spin",
        SU2FusionRule,
        [(SU2Irrep::from_twice_spin(1), 2)]
    );
    factor_conformance!(
        engine,
        "fZ2xU1",
        Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        [
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
        ]
    );
    factor_conformance!(
        engine,
        "fZ2xU1xSU2",
        Fz2U1Su2Rule::try_new(
            Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
        [(
            product_sector(
                product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
                SU2Irrep::from_twice_spin(1),
            ),
            2,
        )]
    );
}
