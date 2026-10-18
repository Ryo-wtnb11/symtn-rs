//! Quantum-dimension-weighted reductions of complex-symbol rules (#2281):
//! `tr`, `norm`, `inner`, `scalar` and `GradedSpace::dim` on Fibonacci and
//! Ising through `TableRule`, against TensorKit 0.17.1; and the structural
//! operations the same bounds admit (unit legs, `restrict_leg`,
//! `is_diagonal`) against hand oracles.
//!
//! Every entry is a function of its fusion-tree pair and degeneracy index
//! alone (`entry`, the FNV-1a hash that `fixtures/tensorkit_table_reductions.jl`
//! also evaluates), so the TensorKit oracle fills the same tensors without
//! reading anything symtn computed. The printed quantities are invariant
//! under the vertex gauge by which AnyonWiki's Ising differs from
//! TensorKit's (see the fixture header).
//!
//! The negative control is a complex-symbol U(1) whose charged sectors carry
//! a quantum dimension with a nonzero imaginary part: every weighted
//! reduction refuses it with the typed error, and its real-dimension twin
//! equals the real-symbol U(1) exactly.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, U1FusionRule};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_sectors::{TableRule, TableSector};

#[macro_use]
#[path = "support/table_fixture.rs"]
mod table_fixture;
use table_fixture::*;

const TENSORKIT: &str = include_str!("fixtures/tensorkit-0.17.1-table-reductions.txt");

fn tensorkit(category: &str, quantity: &str) -> Complex64 {
    let row = TENSORKIT
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|fields| fields[0] == category && fields[1] == quantity)
        .unwrap_or_else(|| panic!("fixture is missing {category} {quantity}"));
    Complex64::new(row[2].parse().unwrap(), row[3].parse().unwrap())
}

fn check_against_tensorkit(category: &str, rule: TableRule, sectors: &[(u16, usize)]) {
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(rule),
        sectors
            .iter()
            .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
    )
    .unwrap();
    let filled = |tag: &'static str, codomain: Vec<&GradedSpace<TableRule>>, domain: Vec<_>| {
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, codomain, domain, |trees, index| {
            entry(tag, trees, index)
        })
        .unwrap()
    };
    let t = filled("t", vec![&v, &v], vec![&v, &v]);
    let s = filled("s", vec![&v, &v], vec![&v, &v]);
    let x = filled("x", vec![], vec![&v]);
    let y = filled("y", vec![&v], vec![]);

    let real = |value: f64| Complex64::new(value, 0.0);
    assert_close(
        &format!("{category} tr"),
        t.tr().unwrap(),
        tensorkit(category, "tr"),
    );
    for (p, quantity) in [(2.0, "norm2"), (1.0, "norm1"), (f64::INFINITY, "norminf")] {
        assert_close(
            &format!("{category} {quantity}"),
            real(t.norm(p).unwrap()),
            tensorkit(category, quantity),
        );
    }
    assert_close(
        &format!("{category} inner"),
        s.inner(&t).unwrap(),
        tensorkit(category, "inner"),
    );
    assert_close(
        &format!("{category} scalar"),
        x.compose(&y).unwrap().scalar().unwrap(),
        tensorkit(category, "scalar"),
    );
    assert_close(
        &format!("{category} dim"),
        real(v.dim().unwrap()),
        tensorkit(category, "dim"),
    );

    // The adjoint identities on the lazy view: tr(t^H) = conj(tr t),
    // norm(t^H) = norm(t), inner(s^H, t^H) = conj(inner(s, t)).
    let (t_h, s_h) = (t.adjoint().unwrap(), s.adjoint().unwrap());
    assert_close(
        &format!("{category} tr adjoint"),
        t_h.tr().unwrap(),
        t.tr().unwrap().conj(),
    );
    assert_close(
        &format!("{category} norm adjoint"),
        real(t_h.norm(2.0).unwrap()),
        real(t.norm(2.0).unwrap()),
    );
    assert_close(
        &format!("{category} inner adjoint"),
        s_h.inner(&t_h).unwrap(),
        s.inner(&t).unwrap().conj(),
    );
}

#[test]
fn fibonacci_reductions_match_tensorkit() {
    check_against_tensorkit("fibonacci", TableRule::fibonacci(), &[(1, 1), (2, 2)]);
}

#[test]
fn ising_reductions_match_tensorkit() {
    check_against_tensorkit("ising", TableRule::ising(), &[(1, 1), (2, 1), (3, 2)]);
}

#[test]
fn graded_space_dimension_is_the_hand_weighted_sum() {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let fib = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 1), (TableSector::new(2), 2)],
    )
    .unwrap();
    assert!((fib.dim().unwrap() - (1.0 + 2.0 * phi)).abs() <= 4.0 * f64::EPSILON);
    let ising = GradedSpace::try_new(
        Arc::new(TableRule::ising()),
        [
            (TableSector::new(1), 1),
            (TableSector::new(2), 1),
            (TableSector::new(3), 2),
        ],
    )
    .unwrap();
    assert!((ising.dim().unwrap() - (2.0 + 2.0 * 2.0_f64.sqrt())).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn a_non_real_quantum_dimension_is_refused_by_every_weighted_reduction() {
    // SU(2) is `Simple`: every reduction reads `dim(c)`.
    let runtime = Runtime::builder().build().unwrap();
    let (v, t) = operands!(&runtime, Embedded::<_, true>(SU2FusionRule), su2_sectors());
    for (what, error) in [
        ("tr", t.tr().unwrap_err()),
        ("norm", t.norm(2.0).unwrap_err()),
        ("inner", t.inner(&t).unwrap_err()),
        ("GradedSpace::dim", v.dim().unwrap_err()),
    ] {
        assert!(is_non_real_dimension(&error), "{what}: {error:?}");
    }
}

#[test]
fn unique_fusion_norm_and_inner_never_read_the_dimension() {
    // TensorKit `inner` (`vectorinterface.jl:124`) and `norm` (`linalg.jl:281`) skip `dim(c)`
    // for `UniqueFusion`, where every quantum dimension is one; symtn's
    // Unique reductions do too, so only `tr` and `dim` meet the tilt.
    let runtime = Runtime::builder().build().unwrap();
    let (v, t) = operands!(&runtime, Embedded::<_, true>(U1FusionRule), u1_sectors());
    let (_, real) = operands!(&runtime, U1FusionRule, u1_sectors());
    assert_eq!(
        t.norm(2.0).unwrap().to_bits(),
        real.norm(2.0).unwrap().to_bits()
    );
    assert_eq!(t.inner(&t).unwrap(), real.inner(&real).unwrap());
    assert!(is_non_real_dimension(&t.tr().unwrap_err()));
    assert!(is_non_real_dimension(&v.dim().unwrap_err()));
}

macro_rules! assert_same_reductions {
    ($runtime:expr, $base:expr, $sectors:expr) => {{
        let (complex_space, complex) = operands!($runtime, Embedded::<_, false>($base), $sectors);
        let (real_space, real) = operands!($runtime, $base, $sectors);
        assert_eq!(complex.dense_data().unwrap(), real.dense_data().unwrap());
        assert_eq!(complex.tr().unwrap(), real.tr().unwrap());
        assert_eq!(
            complex.norm(2.0).unwrap().to_bits(),
            real.norm(2.0).unwrap().to_bits()
        );
        assert_eq!(complex.inner(&complex).unwrap(), real.inner(&real).unwrap());
        assert_eq!(
            complex_space.dim().unwrap().to_bits(),
            real_space.dim().unwrap().to_bits()
        );
    }};
}

#[test]
fn a_real_complex_embedding_reduces_exactly_as_the_real_symbol_rule() {
    let runtime = Runtime::builder().build().unwrap();
    assert_same_reductions!(&runtime, U1FusionRule, u1_sectors());
    assert_same_reductions!(&runtime, SU2FusionRule, su2_sectors());
}

#[test]
fn ising_unit_legs_restriction_and_diagonality_follow_the_hand_oracle() {
    use symtn::typed::{Duality, LegSelection, Side};
    let runtime = Runtime::builder().build().unwrap();
    let (unit, psi, sigma) = (
        TableSector::new(1),
        TableSector::new(2),
        TableSector::new(3),
    );
    let v = GradedSpace::try_new(
        Arc::new(TableRule::ising()),
        [(unit, 1), (psi, 1), (sigma, 2)],
    )
    .unwrap();
    let t: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v, &v], |trees, index| {
            entry("t", trees, index)
        })
        .unwrap();

    // A unit leg carries no recoupling: insertion then removal is the
    // identity on the payload.
    let widened = t.insert_unit(1, Side::Codomain, Duality::Plain).unwrap();
    assert_eq!(widened.rank(), 5);
    let narrowed = widened.remove_unit(1).unwrap();
    assert_eq!(narrowed.dense_data().unwrap(), t.dense_data().unwrap());

    // Keep σ position 1 and the unit on leg 0: every kept entry is the
    // parent's entry at the selected degeneracy position.
    let selection = LegSelection::try_new(&v, [(sigma, vec![1]), (unit, vec![0])]).unwrap();
    let restricted = t.restrict_leg(&[(0, &selection)]).unwrap();
    let mut seen = 0;
    for (trees, block) in restricted.subblocks().unwrap() {
        let leg0 = trees.codomain_uncoupled()[0];
        assert_ne!(leg0, psi, "ψ was not selected");
        let shape = block.shape().to_vec();
        for flat in 0..shape.iter().product::<usize>() {
            let mut index = Vec::with_capacity(shape.len());
            let mut rest = flat;
            for &extent in &shape {
                index.push(rest % extent);
                rest /= extent;
            }
            let mut parent = index.clone();
            if leg0 == sigma {
                parent[0] = 1;
            }
            assert_eq!(*block.get(&index).unwrap(), entry("t", &trees, &parent));
            seen += 1;
        }
    }
    assert!(seen > 0);

    let identity: TensorMap<_, Complex64> = TensorMap::isomorphism(&runtime, [&v], [&v]).unwrap();
    assert!(symtn::expert::is_diagonal(&identity, 0.0).unwrap());
    let full: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&v], [&v], |trees, index| {
            entry("d", trees, index)
        })
        .unwrap();
    assert!(!symtn::expert::is_diagonal(&full, 0.0).unwrap());
}

#[test]
fn ising_compact_diagonal_agrees_with_its_dense_twin() {
    use symtn::typed::SectorSpectrum;
    let runtime = Runtime::builder().build().unwrap();
    let (unit, psi, sigma) = (
        TableSector::new(1),
        TableSector::new(2),
        TableSector::new(3),
    );
    let v = GradedSpace::try_new(
        Arc::new(TableRule::ising()),
        [(unit, 1), (psi, 1), (sigma, 2)],
    )
    .unwrap();
    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &v,
        [
            SectorSpectrum {
                sector: unit,
                values: vec![Complex64::new(0.5, -0.25)],
            },
            SectorSpectrum {
                sector: psi,
                values: vec![Complex64::new(-1.5, 0.75)],
            },
            SectorSpectrum {
                sector: sigma,
                values: vec![Complex64::new(2.0, 0.5), Complex64::new(-0.25, -1.0)],
            },
        ],
    )
    .unwrap();
    assert!(
        compact.dense_data().is_err(),
        "the diagonal is stored compact"
    );
    let dense = compact.materialize().unwrap();
    let x: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&v], [&v], |trees, index| {
            entry("x", trees, index)
        })
        .unwrap();

    assert_close("tr", compact.tr().unwrap(), dense.tr().unwrap());
    let real = |value: f64| Complex64::new(value, 0.0);
    for p in [1.0, 2.0, f64::INFINITY] {
        assert_close(
            &format!("norm({p})"),
            real(compact.norm(p).unwrap()),
            real(dense.norm(p).unwrap()),
        );
    }
    assert_close(
        "inner",
        compact.inner(&x).unwrap(),
        dense.inner(&x).unwrap(),
    );
    let flat = |t: TensorMap<TableRule, Complex64>| -> Vec<Complex64> {
        t.materialize().unwrap().dense_data().unwrap().to_vec()
    };
    for (what, got, want) in [
        (
            "compose D x",
            flat(compact.compose(&x).unwrap()),
            flat(dense.compose(&x).unwrap()),
        ),
        (
            "compose x D",
            flat(x.compose(&compact).unwrap()),
            flat(x.compose(&dense).unwrap()),
        ),
        (
            "transpose",
            flat(compact.transpose(&[1], &[0]).unwrap()),
            flat(dense.transpose(&[1], &[0]).unwrap()),
        ),
        (
            "braid",
            flat(compact.braid(&[1], &[0], &[0, 1]).unwrap()),
            flat(dense.braid(&[1], &[0], &[0, 1]).unwrap()),
        ),
        (
            "inverse braid",
            flat(compact.braid(&[1], &[0], &[1, 0]).unwrap()),
            flat(dense.braid(&[1], &[0], &[1, 0]).unwrap()),
        ),
    ] {
        assert_eq!(got.len(), want.len(), "{what}");
        for (g, w) in got.iter().zip(&want) {
            assert_close(what, *g, *w);
        }
    }
}
