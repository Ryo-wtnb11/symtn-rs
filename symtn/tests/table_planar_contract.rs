//! `planar_contract` / `planar_contract_into` (#2284) against TensorKit
//! 0.17.1 `@planar` (`fixtures/tensorkit_table_linalg.jl`): six geometries on
//! Fibonacci and Ising (composition, both operands transposed, a rotated
//! golden-chain output, a full contraction, nothing contracted, a conjugated
//! operand), the destination form with complex α and β ≠ 0, and non-planar
//! specs that TensorKit rejects too.
//!
//! The tensors are the fixture's tree-hash entries, so the oracle builds them
//! without reading anything symtn computed; Ising rows are signed by the
//! recorded vertex-gauge witness in the fixture. Results are compared by
//! `norm` and by the inner product with a probe tensor on the result space.
//!
//! Symmetric rules: a Bosonic planar contraction is the ordinary one (U(1),
//! SU(2)); a fermionic one is the twist-free transposes-and-compose, which
//! differs from `contract` and matches a TensorKit fℤ₂ `@planar` row.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, TableRule, TableSector, U1FusionRule,
    U1Irrep, Z2Irrep,
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

const TENSORKIT: &str = include_str!("fixtures/tensorkit-0.17.1-table-linalg.txt");

fn tensorkit(category: &str, quantity: &str) -> Vec<f64> {
    TENSORKIT
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|fields| fields[0] == category && fields[1] == quantity)
        .unwrap_or_else(|| panic!("fixture is missing {category} {quantity}"))[3..]
        .iter()
        .map(|value| value.parse().unwrap())
        .collect()
}

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
    category: &'static str,
    runtime: Runtime,
    v: GradedSpace<TableRule>,
}

impl Case {
    fn new(category: &'static str, rule: TableRule, sectors: &[(u16, usize)]) -> Self {
        let v = GradedSpace::try_new(
            Arc::new(rule),
            sectors
                .iter()
                .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
        )
        .unwrap();
        Self {
            category,
            runtime: Runtime::builder().dense_threads(1).build().unwrap(),
            v,
        }
    }

    fn fibonacci() -> Self {
        Self::new("fibonacci", TableRule::fibonacci(), &[(1, 1), (2, 2)])
    }

    fn ising() -> Self {
        Self::new("ising", TableRule::ising(), &[(1, 1), (2, 1), (3, 2)])
    }

    fn filled(&self, tag: &str, codomain: usize, domain: usize) -> Table {
        let v = &self.v;
        TensorMap::from_subblock_fn(&self.runtime, vec![v; codomain], vec![v; domain], |t, i| {
            entry(tag, t, i)
        })
        .unwrap()
    }

    /// The fixture's tensor `tag` on `like`'s space.
    fn filled_like(&self, tag: &str, like: &Table) -> Table {
        TensorMap::from_subblock_fn(&self.runtime, &like.codomain(), &like.domain(), |t, i| {
            entry(tag, t, i)
        })
        .unwrap()
    }

    /// `C` against the fixture's `planar_<quantity>` norm and probe inner
    /// product.
    fn assert_matches(&self, quantity: &str, c: &Table) {
        let category = self.category;
        let norm = tensorkit(category, &format!("planar_{quantity}_norm"))[0];
        assert!(
            close(c.norm(2.0).unwrap(), norm),
            "{category} {quantity} norm {} != {norm}",
            c.norm(2.0).unwrap()
        );
        let probe = self.filled_like(&format!("p{quantity}"), c);
        let want = tensorkit(category, &format!("planar_{quantity}_inner"));
        assert_close(
            &format!("{category} {quantity} inner"),
            probe.inner(c).unwrap(),
            Complex64::new(want[0], want[1]),
        );
    }
}

fn geometries_match_tensorkit(case: &Case) {
    let (a, b) = (case.filled("a1", 2, 2), case.filled("b1", 2, 2));
    let c1 = a.planar_contract(&b, &G1).unwrap();
    case.assert_matches("g1", &c1);
    // The composition geometry is `compose` itself.
    assert_eq!(
        c1.dense_data().unwrap(),
        a.compose(&b).unwrap().dense_data().unwrap()
    );

    let (a, b) = (case.filled("a2", 2, 2), case.filled("b2", 2, 2));
    case.assert_matches("g2", &a.planar_contract(&b, &G2).unwrap());

    let (h, theta) = (case.filled("h3", 2, 2), case.filled("t3", 3, 1));
    let c3 = h.planar_contract(&theta, &G3).unwrap();
    case.assert_matches("g3", &c3);

    let (a, b) = (case.filled("a4", 2, 2), case.filled("b4", 2, 2));
    let c4 = a.planar_contract(&b, &G4).unwrap();
    assert_eq!((c4.codomain_rank(), c4.domain_rank()), (0, 0));
    let scalar = c4.dense_data().unwrap();
    let want = tensorkit(case.category, "planar_g4_scalar");
    assert_eq!(scalar.len(), 1);
    assert_close("g4 scalar", scalar[0], Complex64::new(want[0], want[1]));

    let (a, b) = (case.filled("a5", 1, 1), case.filled("b5", 1, 1));
    case.assert_matches("g5", &a.planar_contract(&b, &G5).unwrap());

    // `θ[l s1 s2; a] * conj(θ[l s1 s2; b])`: conj is the lazy adjoint.
    let theta = case.filled("t6", 3, 1);
    let g6 = spec(&[0, 1, 2], &[1, 2, 3], &[0], &[1]);
    case.assert_matches(
        "g6",
        &theta.planar_contract(theta.adjoint_view(), &g6).unwrap(),
    );
}

#[test]
fn fibonacci_planar_geometries_match_tensorkit() {
    geometries_match_tensorkit(&Case::fibonacci());
}

#[test]
fn ising_planar_geometries_match_tensorkit() {
    geometries_match_tensorkit(&Case::ising());
}

fn into_matches_tensorkit(case: &Case) {
    let (alpha, beta) = (Complex64::new(0.5, -0.25), Complex64::new(1.5, 0.75));
    let (a, b) = (case.filled("a1", 2, 2), case.filled("b1", 2, 2));
    let mut d1 = case.filled("d1", 2, 2);
    a.planar_contract_into(&b, &G1, &mut d1, alpha, beta)
        .unwrap();
    case.assert_matches("into_g1", &d1);

    let (h, theta) = (case.filled("h3", 2, 2), case.filled("t3", 3, 1));
    let shape = h.planar_contract(&theta, &G3).unwrap();
    let mut d3 = case.filled_like("d3", &shape);
    h.planar_contract_into(&theta, &G3, &mut d3, alpha, beta)
        .unwrap();
    case.assert_matches("into_g3", &d3);
}

#[test]
fn fibonacci_planar_into_matches_tensorkit() {
    into_matches_tensorkit(&Case::fibonacci());
}

#[test]
fn ising_planar_into_matches_tensorkit() {
    into_matches_tensorkit(&Case::ising());
}

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
        assert_eq!(
            tensorkit(case.category, &format!("planar_reject_{name}")),
            [1.0],
            "TensorKit `@planar` rejects {name}"
        );
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
        assert!(a
            .planar_contract(&b, &spec(&[0, 2], &[2, 0], &[1, 0], &[3, 2]))
            .is_err());
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

/// The fixture's `entry` for parity sectors, labelled 0 / 1.
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
/// Y[x d; z b]` with a dual odd contracted leg; checked against TensorKit's
/// fℤ₂ `@planar` row and against the same-run explicit steps.
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

    let norm = tensorkit("fz2", "planar_cyclic_norm")[0];
    assert!(close(e.norm(2.0).unwrap(), norm), "fz2 planar norm");
    let probe = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        &e.codomain(),
        &e.domain(),
        |t, i| parity_entry("pcyclic", t, i),
    )
    .unwrap();
    let want = tensorkit("fz2", "planar_cyclic_inner");
    assert_close(
        "fz2 planar inner",
        probe.inner(&e).unwrap(),
        Complex64::new(want[0], want[1]),
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
    let tensor_norm = tensorkit("fz2", "tensor_cyclic_norm")[0];
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
