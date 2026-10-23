//! `planar_contract` / `planar_contract_into` (#2284): six geometries on
//! Fibonacci and Ising (composition, both operands transposed, a rotated
//! golden-chain output, a full contraction, nothing contracted, a conjugated
//! operand), the destination form with complex α and β ≠ 0, and non-planar
//! specs that TensorKit rejects too.
//!
//! Four geometries are checked against an equivalent composition of other
//! operations (`compose`; crossing-free transposes then `compose`, as
//! `planarcontract!` runs it; `tr ∘ compose`; `otimes`). The rotated
//! output and the conjugated operand are checked against `norm` and the inner
//! product with a probe tensor, values checked against TensorKit 0.17.1
//! `@planar` on the same tree-hash operands (`support/table_fixture.rs`;
//! Ising entries carry AnyonWiki's vertex sign `u^{σσ}_ψ = −1`, #2277, so both
//! sides are the same morphism).
//!
//! Symmetric rules: a Bosonic planar contraction is the ordinary one (U(1),
//! SU(2)); a fermionic one is the twist-free transposes-and-compose, which
//! differs from `contract` and matches a TensorKit fℤ₂ `@planar` row, and
//! equals the same-run explicit steps on fℤ₂×U(1), through a lazy adjoint and
//! in the destination form with complex α and β ≠ 0.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep, TableRule,
    TableSector, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{
    BlockFusionTrees, ContractSpec, Error, GradedSpace, OperationError, Runtime, TensorMap,
};

#[allow(unused_macros)]
#[macro_use]
#[path = "support/table_fixture.rs"]
mod table_fixture;
use table_fixture::*;

type Table = TensorMap<TableRule, Complex64>;

fn spec<'a>(
    lhs: &'a [usize],
    rhs: &'a [usize],
    codomain: &'a [usize],
    domain: &'a [usize],
) -> ContractSpec<'a> {
    ContractSpec {
        lhs,
        rhs,
        codomain,
        domain,
    }
}

const G1: ContractSpec<'static> = ContractSpec {
    lhs: &[2, 3],
    rhs: &[0, 1],
    codomain: &[0, 1],
    domain: &[2, 3],
};
const G2: ContractSpec<'static> = ContractSpec {
    lhs: &[0, 2],
    rhs: &[2, 0],
    codomain: &[0, 1],
    domain: &[3, 2],
};
const G3: ContractSpec<'static> = ContractSpec {
    lhs: &[2, 3],
    rhs: &[1, 2],
    codomain: &[2, 0, 1],
    domain: &[3],
};
const G4: ContractSpec<'static> = ContractSpec {
    lhs: &[0, 1, 2, 3],
    rhs: &[2, 3, 0, 1],
    codomain: &[],
    domain: &[],
};
const G5: ContractSpec<'static> = ContractSpec {
    lhs: &[],
    rhs: &[],
    codomain: &[0, 2],
    domain: &[1, 3],
};

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 4096.0 * f64::EPSILON * want.abs().max(1.0)
}

struct Case {
    runtime: Runtime,
    v: GradedSpace<TableRule>,
}

impl Case {
    fn new(rule: TableRule, sectors: &[(u16, usize)]) -> Self {
        let v = GradedSpace::try_new(
            Arc::new(rule),
            sectors
                .iter()
                .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
        )
        .unwrap();
        Self {
            runtime: Runtime::builder().dense_threads(1).build().unwrap(),
            v,
        }
    }

    fn fibonacci() -> Self {
        Self::new(TableRule::fibonacci(), &[(1, 1), (2, 2)])
    }

    fn ising() -> Self {
        Self::new(TableRule::ising(), &[(1, 1), (2, 1), (3, 2)])
    }

    fn filled(&self, tag: &str, codomain: usize, domain: usize) -> Table {
        let v = &self.v;
        TensorMap::from_subblock_fn(&self.runtime, vec![v; codomain], vec![v; domain], |t, i| {
            entry(tag, t, i)
        })
        .unwrap()
    }

    /// The tree-hash tensor `tag` on `like`'s space.
    fn filled_like(&self, tag: &str, like: &Table) -> Table {
        TensorMap::from_subblock_fn(&self.runtime, &like.codomain(), &like.domain(), |t, i| {
            entry(tag, t, i)
        })
        .unwrap()
    }

    /// `C` against its `‖C‖` and `⟨probe, C⟩`, the probe the tree-hash
    /// tensor `p<what>` on `C`'s space.
    fn assert_matches(&self, what: &str, c: &Table, (norm, inner): (f64, Complex64)) {
        assert!(
            close(c.norm(2.0).unwrap(), norm),
            "{what} norm {} != {norm}",
            c.norm(2.0).unwrap()
        );
        let probe = self.filled_like(&format!("p{what}"), c);
        assert_close(&format!("{what} inner"), probe.inner(c).unwrap(), inner);
    }
}

fn assert_entries_close(what: &str, got: &Table, want: &Table) {
    assert_eq!(got.codomain(), want.codomain(), "{what}: codomain");
    assert_eq!(got.domain(), want.domain(), "{what}: domain");
    let (got, want) = (got.dense_data().unwrap(), want.dense_data().unwrap());
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (g, w) in got.iter().zip(want.iter()) {
        assert_close(what, *g, *w);
    }
}

/// `‖C‖` and `⟨probe, C⟩` of G3 and G6 (checked against TensorKit 0.17.1).
struct Rotated {
    g3: (f64, Complex64),
    g6: (f64, Complex64),
}

const FIBONACCI: Rotated = Rotated {
    g3: (
        4.943729839879726,
        Complex64::new(0.10601018294495584, 0.11150100931149448),
    ),
    g6: (
        11.686618512864229,
        Complex64::new(1.3809813920670846, -1.3653145363481314),
    ),
};

const ISING: Rotated = Rotated {
    g3: (
        5.467304381488151,
        Complex64::new(1.5700506713438183, 1.567336298573804),
    ),
    g6: (
        13.69410184860256,
        Complex64::new(-2.953462574037371, 2.9633251503559714),
    ),
};

fn geometries_match(case: &Case, rotated: &Rotated) {
    // G1, the composition geometry, is `compose` itself.
    let (a, b) = (case.filled("a1", 2, 2), case.filled("b1", 2, 2));
    assert_eq!(
        a.planar_contract(&b, &G1).unwrap().dense_data().unwrap(),
        a.compose(&b).unwrap().dense_data().unwrap()
    );

    // G2: `planarcontract!` transposes both operands crossing-free, then
    // composes.
    let (a, b) = (case.filled("a2", 2, 2), case.filled("b2", 2, 2));
    let steps = a
        .transpose(&[1, 3], &[0, 2])
        .unwrap()
        .compose(&b.transpose(&[2, 0], &[3, 1]).unwrap())
        .unwrap();
    assert_entries_close("g2", &a.planar_contract(&b, &G2).unwrap(), &steps);

    let (h, theta) = (case.filled("h3", 2, 2), case.filled("t3", 3, 1));
    case.assert_matches("g3", &h.planar_contract(&theta, &G3).unwrap(), rotated.g3);

    // G4: `A[a b; c d] * B[c d; a b]` closes into the planar trace `tr(A B)`.
    let (a, b) = (case.filled("a4", 2, 2), case.filled("b4", 2, 2));
    let c4 = a.planar_contract(&b, &G4).unwrap();
    assert_eq!((c4.codomain_rank(), c4.domain_rank()), (0, 0));
    let scalar = c4.dense_data().unwrap();
    assert_eq!(scalar.len(), 1);
    assert_close("g4 scalar", scalar[0], a.compose(&b).unwrap().tr().unwrap());

    // G5: `A[a; b] * B[c; d]` with nothing contracted is `A ⊗ B`.
    let (a, b) = (case.filled("a5", 1, 1), case.filled("b5", 1, 1));
    assert_entries_close(
        "g5",
        &a.planar_contract(&b, &G5).unwrap(),
        &a.otimes(&b).unwrap(),
    );

    // `θ[l s1 s2; a] * conj(θ[l s1 s2; b])`: conj is the lazy adjoint.
    let theta = case.filled("t6", 3, 1);
    let g6 = spec(&[0, 1, 2], &[1, 2, 3], &[0], &[1]);
    case.assert_matches(
        "g6",
        &theta.planar_contract(theta.adjoint_view(), &g6).unwrap(),
        rotated.g6,
    );
}

#[test]
fn fibonacci_planar_geometries_match() {
    geometries_match(&Case::fibonacci(), &FIBONACCI);
}

#[test]
fn ising_planar_geometries_match() {
    geometries_match(&Case::ising(), &ISING);
}

/// The destination form is `α · planar_contract + β · D`.
fn into_is_alpha_planar_plus_beta(case: &Case) {
    let (alpha, beta) = (Complex64::new(0.5, -0.25), Complex64::new(1.5, 0.75));
    let (a, b) = (case.filled("a1", 2, 2), case.filled("b1", 2, 2));
    let d1 = case.filled("d1", 2, 2);
    let mut got = case.filled("d1", 2, 2);
    a.planar_contract_into(&b, &G1, &mut got, alpha, beta)
        .unwrap();
    let want = a
        .planar_contract(&b, &G1)
        .unwrap()
        .axpby(alpha, &d1, beta)
        .unwrap();
    assert_entries_close("into g1", &got, &want);

    let (h, theta) = (case.filled("h3", 2, 2), case.filled("t3", 3, 1));
    let c3 = h.planar_contract(&theta, &G3).unwrap();
    let d3 = case.filled_like("d3", &c3);
    let mut got = case.filled_like("d3", &c3);
    h.planar_contract_into(&theta, &G3, &mut got, alpha, beta)
        .unwrap();
    assert_entries_close("into g3", &got, &c3.axpby(alpha, &d3, beta).unwrap());
}

#[test]
fn fibonacci_planar_into_is_alpha_planar_plus_beta() {
    into_is_alpha_planar_plus_beta(&Case::fibonacci());
}

#[test]
fn ising_planar_into_is_alpha_planar_plus_beta() {
    into_is_alpha_planar_plus_beta(&Case::ising());
}

/// TensorKit 0.17.1 `@planar` rejects each of these at expansion too.
fn non_planar_specs_are_rejected(case: &Case) {
    let (a, b) = (case.filled("a2", 2, 2), case.filled("b2", 2, 2));
    let rejected = [
        (
            "same_orientation",
            spec(&[2, 0], &[2, 0], &[0, 1], &[2, 3]),
            "rhs",
        ),
        (
            "not_an_arc",
            spec(&[0, 3], &[0, 1], &[0, 1], &[2, 3]),
            "lhs",
        ),
        (
            "not_a_rotation",
            spec(&[0, 2], &[2, 0], &[1, 0], &[3, 2]),
            "",
        ),
    ];
    for (name, spec, side) in rejected {
        let error = a.planar_contract(&b, &spec).unwrap_err();
        match (side, &error) {
            ("", Error::Operation(op)) => assert!(
                matches!(**op, OperationError::InvalidPermutation { .. }),
                "{name}: {error:?}"
            ),
            (side, Error::Operation(op)) => assert!(
                matches!(**op, OperationError::InvalidAxisSet { tensor, .. } if tensor == side),
                "{name}: {error:?}"
            ),
            _ => panic!("{name}: {error:?}"),
        }
        let mut destination = case.filled("dst", 2, 2);
        let before = destination.dense_data().unwrap().to_vec();
        let into = a
            .planar_contract_into(&b, &spec, &mut destination, Complex64::ONE, Complex64::ONE)
            .unwrap_err();
        assert_eq!(into, error, "{name}: the destination form reports the same");
        let after = destination.dense_data().unwrap();
        assert!(
            before
                .iter()
                .zip(after.iter())
                .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
            "{name}: destination written"
        );
    }
}

#[test]
fn fibonacci_non_planar_specs_are_rejected_like_tensorkit() {
    non_planar_specs_are_rejected(&Case::fibonacci());
}

#[test]
fn ising_non_planar_specs_are_rejected_like_tensorkit() {
    non_planar_specs_are_rejected(&Case::ising());
}

#[test]
fn ordinary_contract_still_rejects_anyonic_rules() {
    let case = Case::fibonacci();
    let (a, b) = (case.filled("a2", 2, 2), case.filled("b2", 2, 2));
    assert!(matches!(
        a.contract(&b, &G2).unwrap_err(),
        Error::Operation(op) if matches!(*op, OperationError::UnsupportedTensorContractScope { .. })
    ));
}

/// A malformed spec reports exactly what `contract` reports.
#[test]
fn malformed_specs_report_the_contract_errors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 1).unwrap();
    for spec in [
        spec(&[2], &[0, 1], &[0, 1], &[2, 3]),
        spec(&[2, 2], &[0, 1], &[0, 1], &[2, 3]),
        spec(&[2, 4], &[0, 1], &[0, 1], &[2, 3]),
        spec(&[2, 3], &[0, 1], &[0, 0], &[2, 3]),
        spec(&[2, 3], &[0, 1], &[0, 1], &[2]),
    ] {
        assert_eq!(
            a.planar_contract(&a, &spec).unwrap_err(),
            a.contract(&a, &spec).unwrap_err(),
            "{spec:?}"
        );
    }
}

/// Bosonic: the planar contraction is the ordinary one, run by its planner
/// (bit-identical), for every planar geometry above, eager and `_into`.
macro_rules! bosonic_planar_is_contract {
    ($runtime:expr, $v:expr) => {{
        let (runtime, v) = ($runtime, $v);
        let rank4 =
            |seed| TensorMap::<_, f64>::rand_with_seed(runtime, [v, v], [v, v], seed).unwrap();
        let theta = TensorMap::<_, f64>::rand_with_seed(runtime, [v, v, v], [v], 7).unwrap();
        let rank2 = |seed| TensorMap::<_, f64>::rand_with_seed(runtime, [v], [v], seed).unwrap();
        let (a, b) = (rank4(1), rank4(2));
        let cases = [(&a, &b, G1), (&a, &b, G2), (&a, &theta, G3), (&a, &b, G4)];
        for (lhs, rhs, spec) in cases {
            let planar = lhs.planar_contract(rhs, &spec).unwrap();
            let ordinary = lhs.contract(rhs, &spec).unwrap();
            assert_eq!(planar.dense_data().unwrap(), ordinary.dense_data().unwrap());
            let mut into = lhs.contract(rhs, &spec).unwrap();
            lhs.planar_contract_into(rhs, &spec, &mut into, 0.5, -1.25)
                .unwrap();
            let mut want = lhs.contract(rhs, &spec).unwrap();
            lhs.contract_into(rhs, &spec, &mut want, 0.5, -1.25)
                .unwrap();
            assert_eq!(into.dense_data().unwrap(), want.dense_data().unwrap());
        }
        let (x, y) = (rank2(3), rank2(4));
        assert_eq!(
            x.planar_contract(&y, &G5).unwrap().dense_data().unwrap(),
            x.contract(&y, &G5).unwrap().dense_data().unwrap()
        );
        // Planarity is checked for Bosonic rules too (as `@planar`).
        assert!(matches!(a
            .planar_contract(&b, &spec(&[0, 2], &[2, 0], &[1, 0], &[3, 2])), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 4, .. })));
        assert!(a
            .contract(&b, &spec(&[0, 2], &[2, 0], &[1, 0], &[3, 2]))
            .is_ok());
    }};
}

#[test]
fn bosonic_planar_contract_is_contract() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    bosonic_planar_is_contract!(&runtime, &u1);
    // SU(2) with j = ½ legs (Frobenius–Schur indicator −1).
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    bosonic_planar_is_contract!(&runtime, &su2);
}

/// The tree-hash `entry` for parity sectors, labelled 0 / 1.
fn parity_entry(tag: &str, trees: &BlockFusionTrees<Z2Irrep>, index: &[usize]) -> Complex64 {
    let label = |sectors: &[Z2Irrep]| {
        sectors
            .iter()
            .map(|&s| if s == Z2Irrep::ODD { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    };
    let key = format!(
        "{tag}|{}|{}|{}|{}|{}|{}",
        label(&[*trees.coupled()]),
        label(trees.codomain_uncoupled()),
        label(trees.codomain_innerlines()),
        label(trees.domain_uncoupled()),
        label(trees.domain_innerlines()),
        index
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    Complex64::new(
        unit_interval(fnv1a(&format!("{key}|re"))),
        unit_interval(fnv1a(&format!("{key}|im"))),
    )
}

/// Fermionic: `planar_contract` is the twist-free transposes-and-compose
/// (TensorKit `planarcontract!`), not `contract`'s supertrace-twisted
/// contraction, in the cyclic geometry `E[a z; c x] := X[a b; c d] *
/// Y[x d; z b]` with a dual odd contracted leg; checked against the explicit
/// steps, and against `‖E‖`, `⟨probe, E⟩` and the `@tensor` result's norm
/// checked against TensorKit 0.17.1 fℤ₂ on the same tree-hash operands.
#[test]
fn fermionic_planar_contract_is_twist_free_and_matches_tensorkit() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    let filled = |tag: &str| {
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&v, &v], [&v, &v], |t, i| {
            parity_entry(tag, t, i)
        })
        .unwrap()
    };
    let (x, y) = (filled("fx"), filled("fy"));
    let cyclic = spec(&[1, 3], &[3, 1], &[0, 3], &[1, 2]);
    let e = x.planar_contract(&y, &cyclic).unwrap();

    let norm = 5.503717993001285;
    assert!(close(e.norm(2.0).unwrap(), norm), "fz2 planar norm");
    let probe = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        &e.codomain(),
        &e.domain(),
        |t, i| parity_entry("pcyclic", t, i),
    )
    .unwrap();
    assert_close(
        "fz2 planar inner",
        probe.inner(&e).unwrap(),
        Complex64::new(1.0618683662524346, 1.0615065511839303),
    );

    let steps = x
        .transpose(&[2, 0], &[3, 1])
        .unwrap()
        .compose(&y.transpose(&[1, 3], &[0, 2]).unwrap())
        .unwrap()
        .transpose(&[1, 3], &[0, 2])
        .unwrap();
    for (got, want) in e
        .dense_data()
        .unwrap()
        .iter()
        .zip(steps.dense_data().unwrap().iter())
    {
        assert_close("fz2 planar vs steps", *got, *want);
    }

    // `@tensor` / `contract` is the other morphism here.
    let ordinary = x.contract(&y, &cyclic).unwrap();
    let tensor_norm = 5.87234692240772;
    assert!(
        close(ordinary.norm(2.0).unwrap(), tensor_norm),
        "fz2 contract norm"
    );
    let difference = e
        .axpby(Complex64::ONE, &ordinary, -Complex64::ONE)
        .unwrap()
        .norm(2.0)
        .unwrap();
    assert!(
        difference > 1e-3 * norm,
        "fermionic planar must differ from contract"
    );
}

fn fermionic_u1_space() -> GradedSpace<ProductFusionRule<FermionParityFusionRule, U1FusionRule>> {
    GradedSpace::try_new(
        Arc::new(
            ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::try_new(
                FermionParityFusionRule,
                U1FusionRule,
            )
            .unwrap(),
        ),
        [
            (ProductSector::new(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (ProductSector::new(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (ProductSector::new(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            (ProductSector::new(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap()
}

fn max_distance(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// fℤ₂×U(1) (design §7.5): the cyclic geometry with an operand transposed
/// on each side and a rotated output equals the explicit steps and differs
/// from `contract`; a lazy-adjoint operand equals the steps over the same
/// adjoint view.
#[test]
fn fermionic_u1_planar_contract_is_the_explicit_steps() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = fermionic_u1_space();
    let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 21).unwrap();
    let y = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 22).unwrap();
    let cyclic = spec(&[1, 3], &[3, 1], &[0, 3], &[1, 2]);
    let planar = x.planar_contract(&y, &cyclic).unwrap();
    let steps = x
        .transpose(&[2, 0], &[3, 1])
        .unwrap()
        .compose(&y.transpose(&[1, 3], &[0, 2]).unwrap())
        .unwrap()
        .transpose(&[1, 3], &[0, 2])
        .unwrap();
    let scale = planar.norm(2.0).unwrap();
    let (got, want) = (planar.dense_data().unwrap(), steps.dense_data().unwrap());
    assert!(max_distance(got, want) <= 1e-12 * scale.max(1.0), "steps");
    let ordinary = x.contract(&y, &cyclic).unwrap();
    assert!(
        max_distance(got, ordinary.dense_data().unwrap()) > 1e-3 * scale,
        "the supertrace twist distinguishes contract"
    );

    // `θ[l s1 s2; a] * conj(θ[l s1 s2; b])` through the lazy adjoint.
    let theta = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v, &v], [&v], 23).unwrap();
    let g6 = spec(&[0, 1, 2], &[1, 2, 3], &[0], &[1]);
    let planar = theta.planar_contract(theta.adjoint_view(), &g6).unwrap();
    let materialized = theta.adjoint().unwrap().materialize().unwrap();
    let explicit = theta
        .transpose(&[3], &[2, 1, 0])
        .unwrap()
        .compose(&materialized.transpose(&[3, 2, 1], &[0]).unwrap())
        .unwrap();
    let scale = planar.norm(2.0).unwrap();
    let got = planar.dense_data().unwrap();
    assert!(
        max_distance(got, explicit.dense_data().unwrap()) <= 1e-12 * scale.max(1.0),
        "lazy adjoint steps"
    );
    let ordinary = theta.contract(theta.adjoint_view(), &g6).unwrap();
    assert!(
        max_distance(got, ordinary.dense_data().unwrap()) > 1e-3 * scale,
        "the supertrace twist distinguishes contract on the adjoint"
    );
}

/// `planar_contract_into` with complex α and β ≠ 0 on fℤ₂: α · (explicit
/// steps) + β · destination, and not α · contract + β · destination.
#[test]
fn fermionic_planar_contract_into_is_alpha_steps_plus_beta() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    let filled = |tag: &str, codomain: usize, domain: usize| {
        TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            vec![&v; codomain],
            vec![&v; domain],
            |t, i| parity_entry(tag, t, i),
        )
        .unwrap()
    };
    let (x, y) = (filled("fx", 2, 2), filled("fy", 2, 2));
    let cyclic = spec(&[1, 3], &[3, 1], &[0, 3], &[1, 2]);
    let (alpha, beta) = (Complex64::new(0.5, -0.25), Complex64::new(1.5, 0.75));
    let steps = x
        .transpose(&[2, 0], &[3, 1])
        .unwrap()
        .compose(&y.transpose(&[1, 3], &[0, 2]).unwrap())
        .unwrap()
        .transpose(&[1, 3], &[0, 2])
        .unwrap();
    let ordinary = x.contract(&y, &cyclic).unwrap();
    let initial = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        &steps.codomain(),
        &steps.domain(),
        |t, i| parity_entry("fdst", t, i),
    )
    .unwrap();
    let mut destination = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        &steps.codomain(),
        &steps.domain(),
        |t, i| parity_entry("fdst", t, i),
    )
    .unwrap();
    x.planar_contract_into(&y, &cyclic, &mut destination, alpha, beta)
        .unwrap();
    let got = destination.dense_data().unwrap();
    let start = initial.dense_data().unwrap();
    let expect = |op: &[Complex64]| -> Vec<Complex64> {
        op.iter()
            .zip(start.iter())
            .map(|(o, d)| alpha * o + beta * d)
            .collect()
    };
    for (got, want) in got.iter().zip(expect(steps.dense_data().unwrap())) {
        assert_close("fz2 planar_contract_into", *got, want);
    }
    let twisted = expect(ordinary.dense_data().unwrap());
    assert!(
        got.iter()
            .zip(&twisted)
            .map(|(g, t)| (g - t).norm())
            .fold(0.0, f64::max)
            > 1e-3,
        "the destination form is the planar morphism, not contract's"
    );
}
