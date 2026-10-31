//! Quantum-dimension-weighted reductions of complex-symbol rules (#2281):
//! `tr`, `norm`, `inner`, `scalar` and `GradedSpace::dim` on Fibonacci and
//! Ising through `TableRule`, against TensorKit's definitions written per
//! subblock; and the structural operations the same bounds admit (unit legs,
//! `restrict_leg`, `is_diagonal`) against hand oracles.
//!
//! The negative control is a complex-symbol U(1) whose charged sectors carry
//! a quantum dimension with a nonzero imaginary part: every weighted
//! reduction refuses it with the typed error, and its real-dimension twin
//! equals the real-symbol U(1) exactly.

use std::sync::Arc;
use symtn::typed::Engine;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, U1FusionRule};
use symtn::typed::{BlockFusionTrees, GradedSpace, Runtime, TensorMap};
use symtn_sectors::{TableRule, TableSector};

use crate::table_fixture;
use table_fixture::*;

/// Every multi-index of `shape`, first index fastest.
fn indices(shape: &[usize]) -> Vec<Vec<usize>> {
    (0..shape.iter().product::<usize>())
        .map(|flat| {
            let mut rest = flat;
            shape
                .iter()
                .map(|&extent| {
                    let i = rest % extent;
                    rest /= extent;
                    i
                })
                .collect()
        })
        .collect()
}

/// TensorKit's reductions (`tr`, `norm`, `inner`, `scalar`) on
/// `t, s ∈ V ⊗ V ← V ⊗ V`, `x ∈ 1 ← V`, `y ∈ V ← 1`, against their
/// definitions over the coupled-sector blocks `B_c`, read per subblock:
/// `tr t = Σ_c d_c tr B_c` (entries whose codomain and domain trees and
/// degeneracy indices agree), `‖t‖_p = (Σ_c d_c Σ |B_c|^p)^{1/p}`,
/// `‖t‖_∞ = max |B_c|`, `inner(s, t) = Σ_c d_c tr(S_c† T_c)`, and
/// `scalar(x y)` is the product of the two unit-sector entries.
fn check_closed_forms(
    category: &str,
    rule: TableRule,
    sectors: &[(u16, usize)],
    d: fn(u16) -> f64,
) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(rule),
        sectors
            .iter()
            .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
    )
    .unwrap();
    let mut filled = |tag: &'static str, codomain: Vec<&GradedSpace<TableRule>>, domain: Vec<_>| {
        TensorMap::<_, Complex64>::from_subblock_fn(
            engine,
            &runtime,
            codomain,
            domain,
            |trees, index| entry(tag, trees, index),
        )
        .unwrap()
    };
    let t = filled("t", vec![&v, &v], vec![&v, &v]);
    let s = filled("s", vec![&v, &v], vec![&v, &v]);
    let x = filled("x", vec![], vec![&v]);
    let y = filled("y", vec![&v], vec![]);

    let weight = |trees: &BlockFusionTrees<TableSector>| d(trees.coupled().label());
    let (mut tr, mut inner) = (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let (mut norm1, mut norm2, mut norminf) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut traced = 0;
    for ((trees, block), (s_trees, s_block)) in t.subblocks().unwrap().zip(s.subblocks().unwrap()) {
        assert_eq!(trees, s_trees);
        let w = weight(&trees);
        let shape = block.shape().to_vec();
        for index in indices(&shape) {
            let z = *block.get(&index).unwrap();
            norm1 += w * z.norm();
            norm2 += w * z.norm_sqr();
            norminf = norminf.max(z.norm());
            inner += w * s_block.get(&index).unwrap().conj() * z;
        }
        if trees.codomain_uncoupled() == trees.domain_uncoupled()
            && trees.codomain_innerlines() == trees.domain_innerlines()
        {
            let legs = trees.codomain_uncoupled().len();
            for row in indices(&shape[..legs]) {
                tr += w * block.get(&[row.clone(), row].concat()).unwrap();
                traced += 1;
            }
        }
    }
    assert!(traced > 0, "{category}: some trace entry");
    let unit_entry = |m: &TensorMap<TableRule, Complex64>| {
        let mut blocks = m.subblocks().unwrap();
        assert_eq!(blocks.len(), 1, "only the unit sector couples to 1");
        *blocks.next().unwrap().1.get(&[0]).unwrap()
    };

    let real = |value: f64| Complex64::new(value, 0.0);
    assert_close(&format!("{category} tr"), t.tr(engine).unwrap(), tr);
    for (p, want) in [(1.0, norm1), (2.0, norm2.sqrt()), (f64::INFINITY, norminf)] {
        assert_close(
            &format!("{category} norm({p})"),
            real(t.norm(engine, p).unwrap()),
            real(want),
        );
    }
    assert_close(
        &format!("{category} inner"),
        s.inner(engine, &t).unwrap(),
        inner,
    );
    assert_close(
        &format!("{category} scalar"),
        x.compose(engine, &y).unwrap().scalar(engine).unwrap(),
        unit_entry(&x) * unit_entry(&y),
    );

    // The adjoint identities on the lazy view: tr(t^H) = conj(tr t),
    // norm(t^H) = norm(t), inner(s^H, t^H) = conj(inner(s, t)).
    let (t_h, s_h) = (t.adjoint(engine).unwrap(), s.adjoint(engine).unwrap());
    assert_close(
        &format!("{category} tr adjoint"),
        t_h.tr(engine).unwrap(),
        t.tr(engine).unwrap().conj(),
    );
    assert_close(
        &format!("{category} norm adjoint"),
        real(t_h.norm(engine, 2.0).unwrap()),
        real(t.norm(engine, 2.0).unwrap()),
    );
    assert_close(
        &format!("{category} inner adjoint"),
        s_h.inner(engine, &t_h).unwrap(),
        s.inner(engine, &t).unwrap().conj(),
    );
}

#[test]
fn fibonacci_reductions_follow_the_weighted_block_formulas() {
    // AnyonWiki labels: 1 = I, 2 = τ (d_τ = φ).
    check_closed_forms(
        "fibonacci",
        TableRule::fibonacci(),
        &[(1, 1), (2, 2)],
        |label| {
            if label == 1 {
                1.0
            } else {
                (1.0 + 5.0_f64.sqrt()) / 2.0
            }
        },
    );
}

#[test]
fn ising_reductions_follow_the_weighted_block_formulas() {
    // AnyonWiki labels: 1 = I, 2 = ψ, 3 = σ (d_σ = √2).
    check_closed_forms(
        "ising",
        TableRule::ising(),
        &[(1, 1), (2, 1), (3, 2)],
        |label| if label == 3 { 2.0_f64.sqrt() } else { 1.0 },
    );
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    // SU(2) is `Simple`: every reduction reads `dim(c)`.
    let runtime = Runtime::builder().build().unwrap();
    let (v, t) = operands!(
        engine,
        &runtime,
        Embedded::<_, true>(SU2FusionRule),
        su2_sectors()
    );
    for (what, error) in [
        ("tr", t.tr(engine).unwrap_err()),
        ("norm", t.norm(engine, 2.0).unwrap_err()),
        ("inner", t.inner(engine, &t).unwrap_err()),
        ("GradedSpace::dim", v.dim().unwrap_err()),
    ] {
        assert!(is_non_real_dimension(&error), "{what}: {error:?}");
    }
}

#[test]
fn unique_fusion_norm_and_inner_never_read_the_dimension() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // TensorKit `inner` (`vectorinterface.jl:124`) and `norm` (`linalg.jl:281`) skip `dim(c)`
    // for `UniqueFusion`, where every quantum dimension is one; symtn's
    // Unique reductions do too, so only `tr` and `dim` meet the tilt.
    let runtime = Runtime::builder().build().unwrap();
    let (v, t) = operands!(
        engine,
        &runtime,
        Embedded::<_, true>(U1FusionRule),
        u1_sectors()
    );
    let (_, real) = operands!(engine, &runtime, U1FusionRule, u1_sectors());
    assert_eq!(
        t.norm(engine, 2.0).unwrap().to_bits(),
        real.norm(engine, 2.0).unwrap().to_bits()
    );
    assert_eq!(
        t.inner(engine, &t).unwrap(),
        real.inner(engine, &real).unwrap()
    );
    assert!(is_non_real_dimension(&t.tr(engine).unwrap_err()));
    assert!(is_non_real_dimension(&v.dim().unwrap_err()));
}

macro_rules! assert_same_reductions {
    ($engine:ident, $runtime:expr, $base:expr, $sectors:expr) => {{
        let (complex_space, complex) =
            operands!($engine, $runtime, Embedded::<_, false>($base), $sectors);
        let (real_space, real) = operands!($engine, $runtime, $base, $sectors);
        assert_eq!(complex.dense_data().unwrap(), real.dense_data().unwrap());
        assert_eq!(complex.tr($engine).unwrap(), real.tr($engine).unwrap());
        assert_eq!(
            complex.norm($engine, 2.0).unwrap().to_bits(),
            real.norm($engine, 2.0).unwrap().to_bits()
        );
        assert_eq!(
            complex.inner($engine, &complex).unwrap(),
            real.inner($engine, &real).unwrap()
        );
        assert_eq!(
            complex_space.dim().unwrap().to_bits(),
            real_space.dim().unwrap().to_bits()
        );
    }};
}

#[test]
fn a_real_complex_embedding_reduces_exactly_as_the_real_symbol_rule() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_same_reductions!(engine, &runtime, U1FusionRule, u1_sectors());
    assert_same_reductions!(engine, &runtime, SU2FusionRule, su2_sectors());
}

#[test]
fn ising_unit_legs_restriction_and_diagonality_follow_the_hand_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
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
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v, &v], |trees, index| {
            entry("t", trees, index)
        })
        .unwrap();

    // A unit leg carries no recoupling: insertion then removal is the
    // identity on the payload.
    let widened = t
        .insert_unit(engine, 1, Side::Codomain, Duality::Plain)
        .unwrap();
    assert_eq!(widened.rank(), 5);
    let narrowed = widened.remove_unit(engine, 1).unwrap();
    assert_eq!(narrowed.dense_data().unwrap(), t.dense_data().unwrap());

    // Keep σ position 1 and the unit on leg 0: every kept entry is the
    // parent's entry at the selected degeneracy position.
    let selection = LegSelection::try_new(&v, [(sigma, vec![1]), (unit, vec![0])]).unwrap();
    let restricted = t.restrict_leg(engine, &[(0, &selection)]).unwrap();
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

    let identity: TensorMap<_, Complex64> =
        TensorMap::isomorphism(engine, &runtime, [&v], [&v]).unwrap();
    assert!(symtn::expert::is_diagonal(engine, &identity, 0.0).unwrap());
    let full: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v], [&v], |trees, index| {
            entry("d", trees, index)
        })
        .unwrap();
    assert!(!symtn::expert::is_diagonal(engine, &full, 0.0).unwrap());
}

#[test]
fn ising_compact_diagonal_agrees_with_its_dense_twin() {
    let engine = &mut symtn_dense::CpuEngine::serial();
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
        engine,
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
    let dense = compact.materialize(engine).unwrap();
    let x: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v], [&v], |trees, index| {
            entry("x", trees, index)
        })
        .unwrap();

    assert_close("tr", compact.tr(engine).unwrap(), dense.tr(engine).unwrap());
    let real = |value: f64| Complex64::new(value, 0.0);
    for p in [1.0, 2.0, f64::INFINITY] {
        assert_close(
            &format!("norm({p})"),
            real(compact.norm(engine, p).unwrap()),
            real(dense.norm(engine, p).unwrap()),
        );
    }
    assert_close(
        "inner",
        compact.inner(engine, &x).unwrap(),
        dense.inner(engine, &x).unwrap(),
    );
    let flat = |engine: &mut dyn Engine, t: TensorMap<TableRule, Complex64>| -> Vec<Complex64> {
        t.materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    };
    for (what, got, want) in [
        (
            "compose D x",
            {
                let c = compact.compose(engine, &x).unwrap();
                flat(engine, c)
            },
            {
                let c = dense.compose(engine, &x).unwrap();
                flat(engine, c)
            },
        ),
        (
            "compose x D",
            {
                let c = x.compose(engine, &compact).unwrap();
                flat(engine, c)
            },
            {
                let c = x.compose(engine, &dense).unwrap();
                flat(engine, c)
            },
        ),
        (
            "transpose",
            {
                let c = compact.transpose(engine, &[1], &[0]).unwrap();
                flat(engine, c)
            },
            {
                let c = dense.transpose(engine, &[1], &[0]).unwrap();
                flat(engine, c)
            },
        ),
        (
            "braid",
            {
                let c = compact.braid(engine, &[1], &[0], &[0, 1]).unwrap();
                flat(engine, c)
            },
            {
                let c = dense.braid(engine, &[1], &[0], &[0, 1]).unwrap();
                flat(engine, c)
            },
        ),
        (
            "inverse braid",
            {
                let c = compact.braid(engine, &[1], &[0], &[1, 0]).unwrap();
                flat(engine, c)
            },
            {
                let c = dense.braid(engine, &[1], &[0], &[1, 0]).unwrap();
                flat(engine, c)
            },
        ),
    ] {
        assert_eq!(got.len(), want.len(), "{what}");
        for (g, w) in got.iter().zip(&want) {
            assert_close(what, *g, *w);
        }
    }
}
