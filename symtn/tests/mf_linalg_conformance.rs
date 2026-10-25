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
    ($actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        let error = actual
            .axpby(1.0, expected, -1.0)
            .unwrap()
            .norm(2.0)
            .unwrap();
        assert!(
            error <= 1e-9 * (1.0 + expected.norm(2.0).unwrap()),
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
    ($name:expr, $rule:expr, $pairs:expr) => {{
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
            TensorMap::from_subblock_fn(&rt, [&tall_space], [&wide_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 2.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&rt, [&wide_space], [&tall_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 2.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();

        let Svd { u, s, vh } = tall.svd_compact(&[0], &[1]).unwrap();
        assert_provider!(provider; u, s, vh);
        assert_close!(&u.compose(&s).unwrap().compose(&vh).unwrap(), &tall);
        let Svd { u, s, vh } = tall.svd_full(&[0], &[1]).unwrap();
        assert_provider!(provider; u, s, vh);
        assert_close!(&u.compose(&s).unwrap().compose(&vh).unwrap(), &tall);
        // Truncated SVD: svd_compact -> diagview -> find_truncated -> restrict.
        let Svd { u, s, vh } = tall.svd_compact(&[0], &[1]).unwrap();
        let found = s.domain()[0]
            .find_truncated(&s.diagview().unwrap(), &Truncation::rank(1))
            .unwrap();
        let u = u.restrict_leg(&[(u.codomain_rank(), &found.selection)]).unwrap();
        let s = s.restrict_leg(&[(0, &found.selection), (1, &found.selection)]).unwrap();
        let vh = vh.restrict_leg(&[(0, &found.selection)]).unwrap();
        assert_provider!(provider; u, s, vh);
        let reconstructed = u.compose(&s).unwrap().compose(&vh).unwrap();
        let error = reconstructed.axpby(1.0, &tall, -1.0).unwrap().norm(2.0).unwrap();
        assert!((error - found.error).abs() <= 1e-9 * (1.0 + found.error));
        assert!(found.error > 0.0, $name);
        let singular_values = tall.svd_vals(&[0], &[1]).unwrap();
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
        assert!((tall.norm(2.0).unwrap().powi(2) - weighted_norm_squared).abs() <= 1e-9);

        let Qr { q, r } = tall.qr_compact(&[0], &[1]).unwrap();
        assert_provider!(provider; q, r);
        assert_close!(&q.compose(&r).unwrap(), &tall);
        let id = TensorMap::isomorphism(&rt, q.domain().iter(), q.domain().iter()).unwrap();
        assert_close!(&q.adjoint().unwrap().compose(&q).unwrap(), &id);
        let Qr { q, r } = tall.qr_full(&[0], &[1]).unwrap();
        assert_provider!(provider; q, r);
        assert_close!(&q.compose(&r).unwrap(), &tall);
        let id = TensorMap::isomorphism(&rt, q.domain().iter(), q.domain().iter()).unwrap();
        assert_close!(&q.adjoint().unwrap().compose(&q).unwrap(), &id);

        let Lq { l, q } = wide.lq_compact(&[0], &[1]).unwrap();
        assert_provider!(provider; l, q);
        assert_close!(&l.compose(&q).unwrap(), &wide);
        let id = TensorMap::isomorphism(&rt, q.codomain().iter(), q.codomain().iter()).unwrap();
        assert_close!(&q.compose(&q.adjoint().unwrap()).unwrap(), &id);
        let Lq { l, q } = wide.lq_full(&[0], &[1]).unwrap();
        assert_provider!(provider; l, q);
        assert_close!(&l.compose(&q).unwrap(), &wide);
        let id = TensorMap::isomorphism(&rt, q.codomain().iter(), q.codomain().iter()).unwrap();
        assert_close!(&q.compose(&q.adjoint().unwrap()).unwrap(), &id);

        let left = tall.left_null(&[0], &[1]).unwrap();
        assert_provider!(provider; left);
        assert!(
            left.adjoint()
                .unwrap()
                .compose(&tall)
                .unwrap()
                .norm(2.0)
                .unwrap()
                <= 1e-9,
            $name
        );
        let left_id = TensorMap::isomorphism(&rt, left.domain().iter(), left.domain().iter()).unwrap();
        assert_close!(&left.adjoint().unwrap().compose(&left).unwrap(), &left_id);
        let right = wide.right_null(&[0], &[1]).unwrap();
        assert_provider!(provider; right);
        assert!(
            wide.compose(&right.adjoint().unwrap())
                .unwrap()
                .norm(2.0)
                .unwrap()
                <= 1e-9,
            $name
        );
        let right_id = TensorMap::isomorphism(&rt, right.codomain().iter(), right.codomain().iter()).unwrap();
        assert_close!(
            &right.compose(&right.adjoint().unwrap()).unwrap(),
            &right_id
        );

        let LeftPolar { w, p } = tall.left_polar(&[0], &[1]).unwrap();
        assert_provider!(provider; w, p);
        assert_close!(&w.compose(&p).unwrap(), &tall);
        let id = TensorMap::isomorphism(&rt, w.domain().iter(), w.domain().iter()).unwrap();
        assert_close!(&w.adjoint().unwrap().compose(&w).unwrap(), &id);
        assert!(is_hermitian!(p, 1e-10));
        assert!(p
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .flat_map(|s| &s.values)
            .all(|&x| x >= -1e-10));
        let RightPolar { p, wh: w } = wide.right_polar(&[0], &[1]).unwrap();
        assert_provider!(provider; p, w);
        assert_close!(&p.compose(&w).unwrap(), &wide);
        let id = TensorMap::isomorphism(&rt, w.codomain().iter(), w.codomain().iter()).unwrap();
        assert_close!(&w.compose(&w.adjoint().unwrap()).unwrap(), &id);
        assert!(is_hermitian!(p, 1e-10));
        assert!(p
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .flat_map(|s| &s.values)
            .all(|&x| x >= -1e-10));

        let h: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&rt, [&endo_space], [&endo_space], |_, index| {
                [[3.0, 1.0], [1.0, 3.0]][index[0]][index[1]]
            })
            .unwrap();
        let Eigh { d, v } = h.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
        assert_provider!(provider; d, v);
        assert_close!(
            &v.compose(&d)
                .unwrap()
                .compose(&v.adjoint().unwrap())
                .unwrap(),
            &h
        );
        assert!(h
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .all(|entry| entry.values == [2.0, 4.0]));
        let found = d.domain()[0]
            .find_truncated(&d.diagview().unwrap(), &Truncation::rank(1))
            .unwrap();
        let trunc_d = d.restrict_leg(&[(0, &found.selection), (1, &found.selection)]).unwrap();
        let trunc_v = v.restrict_leg(&[(v.codomain_rank(), &found.selection)]).unwrap();
        assert_provider!(provider; trunc_d, trunc_v);
        let reconstructed = trunc_v
            .compose(&trunc_d)
            .unwrap()
            .compose(&trunc_v.adjoint().unwrap())
            .unwrap();
        let error = reconstructed.axpby(1.0, &h, -1.0).unwrap().norm(2.0).unwrap();
        assert!((error - found.error).abs() <= 1e-9 * (1.0 + found.error));
        assert!(found.error > 0.0, $name);

        let g: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&rt, [&endo_space], [&endo_space], |_, index| {
                [[3.0, 1.0], [0.0, 1.0]][index[0]][index[1]]
            })
            .unwrap();
        let Eig { d, v } = g.eig_full(&[0], &[1]).unwrap();
        assert_provider!(provider; d, v);
        assert_complex_close!(
            &v.compose(&d).unwrap().compose(&v.inv(&[0], &[1]).unwrap()).unwrap(),
            &g.convert::<Complex64>()
        );
        assert!(g
            .eig_vals(&[0], &[1])
            .unwrap()
            .iter()
            .all(|entry| entry.values == [1.0.into(), 3.0.into()]));
        let found = d.domain()[0]
            .find_truncated(&d.diagview().unwrap(), &Truncation::rank(1))
            .unwrap();
        let trunc_d = d.restrict_leg(&[(0, &found.selection), (1, &found.selection)]).unwrap();
        let trunc_v = v.restrict_leg(&[(v.codomain_rank(), &found.selection)]).unwrap();
        assert_provider!(provider; trunc_d, trunc_v);
        assert_complex_close!(
            &g.convert::<Complex64>().compose(&trunc_v).unwrap(),
            &trunc_v.compose(&trunc_d).unwrap()
        );
        assert!(found.error > 0.0, $name);

        let id = TensorMap::isomorphism(&rt, h.domain().iter(), h.domain().iter()).unwrap();
        let inverse = h.inv(&[0], &[1]).unwrap();
        assert_provider!(provider; inverse);
        assert_close!(&h.compose(&inverse).unwrap(), &id);
        let exponential = h.exp(&[0], &[1]).unwrap();
        assert_provider!(provider; exponential);
        assert_close!(
            &h.scale(-1.0).unwrap()
                .exp(&[0], &[1])
                .unwrap()
                .compose(&exponential)
                .unwrap(),
            &id
        );
        let solved = h.solve(&[0], &[1], &id, &[0], &[1]).unwrap();
        assert_provider!(provider; solved);
        assert_close!(&solved, &inverse);
        let solved_right = h
            .adjoint()
            .unwrap()
            .solve(&[0], &[1], &id.adjoint().unwrap(), &[0], &[1])
            .unwrap()
            .adjoint()
            .unwrap();
        assert_provider!(provider; solved_right);
        assert_close!(&solved_right, &inverse);
        let pseudo = tall.pinv(&[0], &[1], 1e-12).unwrap();
        assert_provider!(provider; pseudo);
        assert_close!(
            &tall.compose(&pseudo).unwrap().compose(&tall).unwrap(),
            &tall
        );
        let aa_plus = tall.compose(&pseudo).unwrap();
        let a_plus_a = pseudo.compose(&tall).unwrap();
        assert_close!(&a_plus_a.compose(&pseudo).unwrap(), &pseudo);
        assert_close!(&aa_plus.adjoint().unwrap(), &aa_plus);
        assert_close!(&a_plus_a.adjoint().unwrap(), &a_plus_a);
        let left_projector = left.compose(&left.adjoint().unwrap()).unwrap();
        let id = TensorMap::isomorphism(&rt, tall.codomain().iter(), tall.codomain().iter()).unwrap();
        assert_close!(&left_projector.axpby(1.0, &aa_plus, 1.0).unwrap(), &id);
        let diagonal: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&rt, [&endo_space], [&endo_space], |_, index| {
                if index[0] == index[1] {
                    [4.0, 9.0][index[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        assert!(diagonal.map_diagonal(f64::sqrt).is_err());
        let compact =
            TensorMap::diagonal(&rt, &endo_space, diagonal.diagview().unwrap()).unwrap();
        let root = compact.map_diagonal(f64::sqrt).unwrap();
        assert_provider!(provider; root);
        assert!(root.materialize().unwrap().dense_data().unwrap().iter().any(|&value| value == 2.0));
        assert!(root.materialize().unwrap().dense_data().unwrap().iter().any(|&value| value == 3.0));
        assert!(root
            .materialize().unwrap().dense_data().unwrap()
            .iter()
            .all(|&value| value == 0.0 || value == 2.0 || value == 3.0));
        assert_close!(&root.compose(&root).unwrap(), &diagonal);
        let before = tall.dense_data().unwrap().to_vec();
        assert!(tall.pinv(&[0], &[1], -1.0).is_err());
        assert_eq!(tall.dense_data().unwrap(), before.as_slice());
    }};
}

#[test]
fn public_multiplicity_free_linalg_conformance() {
    factor_conformance!(
        "U(1)",
        U1FusionRule,
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]
    );
    let z3 = ZNFusionRule::new(3).unwrap();
    let z3_pairs = [(z3.irrep(0), 2), (z3.irrep(1), 2)];
    factor_conformance!("Z3", z3, z3_pairs);
    factor_conformance!(
        "CU1 vacuum/pseudo/charged",
        symtn::sector::CU1FusionRule,
        [
            (symtn::sector::CU1Irrep::VACUUM, 2),
            (symtn::sector::CU1Irrep::PSEUDOSCALAR, 2),
            (symtn::sector::CU1Irrep::from_twice_charge(1), 2),
        ]
    );
    factor_conformance!("fZ2 odd", FermionParityFusionRule, [(Z2Irrep::ODD, 2)]);
    factor_conformance!(
        "SU2 nonzero spin",
        SU2FusionRule,
        [(SU2Irrep::from_twice_spin(1), 2)]
    );
    factor_conformance!(
        "fZ2xU1",
        Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        [
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
        ]
    );
    factor_conformance!(
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
