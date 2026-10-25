//! `twist` and `flip` of complex-symbol rules (#2281) on Fibonacci and
//! Ising through `TableRule`: twist against the closed-form θ entrywise,
//! flip against a few values checked against TensorKit 0.17.1, TensorKit's
//! inverse identities, and the lazy adjoint; and the unchanged `NoBraiding`
//! refusals on Rep(S₃).

use std::sync::Arc;

use num_complex::Complex64;
use symtn::typed::{Direction, Error, GradedSpace, Runtime, SectorSpectrum, TensorMap};
use symtn_sectors::{TableRule, TableSector};

use crate::table_fixture;
use table_fixture::*;

type Table = TensorMap<TableRule, Complex64>;

/// Zero-based symtn legs and TensorKit's `inv`.
const TWISTS: [(bool, &[usize]); 5] = [
    (false, &[1]),
    (true, &[1]),
    (false, &[0, 2]),
    (true, &[1, 3]),
    (false, &[0, 1, 2, 3]),
];
const FLIPS: [(bool, &[usize]); 7] = [
    (false, &[0]),
    (false, &[1]),
    (false, &[2]),
    (false, &[3]),
    (true, &[0, 3]),
    (true, &[1, 2]),
    (false, &[0, 1, 2, 3]),
];

fn direction(inverse: bool) -> Direction {
    if inverse {
        Direction::Inverse
    } else {
        Direction::Forward
    }
}

struct Category {
    name: &'static str,
    v: GradedSpace<TableRule>,
    /// Closed-form θ by AnyonWiki label.
    theta: fn(u16) -> Complex64,
}

fn fibonacci() -> Category {
    Category {
        name: "fibonacci",
        v: space(TableRule::fibonacci(), &[(1, 2), (2, 2)]),
        theta: |label| match label {
            1 => Complex64::new(1.0, 0.0),
            // TensorKitSectors `twist_from_Rsymbol`:
            // θ_τ = R^{ττ}_1/φ + R^{ττ}_τ = e^{4πi/5}/φ + e^{-3πi/5} = e^{-4πi/5}.
            _ => Complex64::from_polar(1.0, -4.0 * std::f64::consts::PI / 5.0),
        },
    }
}

fn ising() -> Category {
    Category {
        name: "ising",
        v: space(TableRule::ising(), &[(1, 1), (2, 2), (3, 2)]),
        theta: |label| match label {
            1 => Complex64::new(1.0, 0.0),
            2 => Complex64::new(-1.0, 0.0),
            // θ_σ = (R^{σσ}_1 + R^{σσ}_ψ)/√2 = (e^{-iπ/8} + e^{3iπ/8})/√2 = e^{iπ/8}.
            _ => Complex64::from_polar(1.0, std::f64::consts::PI / 8.0),
        },
    }
}

fn space(rule: impl Into<Arc<TableRule>>, sectors: &[(u16, usize)]) -> GradedSpace<TableRule> {
    GradedSpace::try_new(
        rule.into(),
        sectors
            .iter()
            .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
    )
    .unwrap()
}

fn filled(
    runtime: &Runtime,
    tag: &'static str,
    codomain: &[GradedSpace<TableRule>],
    domain: &[GradedSpace<TableRule>],
) -> Table {
    TensorMap::from_subblock_fn(runtime, codomain, domain, |trees, index| {
        entry(tag, trees, index)
    })
    .unwrap()
}

/// `t ∈ V ⊗ V' ← V' ⊗ V`: dual and non-dual legs on both sides, and every
/// coupled sector of the category.
fn operand(runtime: &Runtime, category: &Category) -> Table {
    let dual = category.v.try_dual().unwrap();
    filled(
        runtime,
        "t",
        &[category.v.clone(), dual.clone()],
        &[dual, category.v.clone()],
    )
}

fn probe(runtime: &Runtime, result: &Table) -> Complex64 {
    filled(runtime, "p", &result.codomain(), &result.domain())
        .inner(result)
        .unwrap()
}

fn assert_same(what: &str, got: &Table, want: &Table) {
    assert_eq!(got.codomain(), want.codomain(), "{what}: codomain");
    assert_eq!(got.domain(), want.domain(), "{what}: domain");
    let (got, want) = (got.materialize().unwrap(), want.materialize().unwrap());
    let (got, want) = (got.dense_data().unwrap(), want.dense_data().unwrap());
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (x, y)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (x - y).norm() <= 16.0 * f64::EPSILON,
            "{what}: entry {index}: {x} != {y}"
        );
    }
}

/// `inner(probe, flip(t, legs))` checked against TensorKit 0.17.1 (`flip`
/// on the same `entry` operand and probe, AnyonWiki labels; one-based legs
/// `1`, inv `1,4` for Fibonacci and `4`, inv `2,3` for Ising). Each value is
/// invariant under AnyonWiki's Ising vertex sign u^{σσ}_ψ = −1 (#2277): flip
/// scales a tree pair by twists and Frobenius-Schur phases of its uncoupled
/// sectors, and `inner` pairs entries of equal trees.
const FLIP_ROWS: [(&str, bool, &[usize], Complex64); 4] = [
    (
        "fibonacci",
        false,
        &[0],
        Complex64::new(4.429009764303868, -0.00839304164666606),
    ),
    (
        "fibonacci",
        true,
        &[0, 3],
        Complex64::new(-1.8270041038594473, 2.030263134846873),
    ),
    (
        "ising",
        false,
        &[3],
        Complex64::new(1.194743771104495, 0.21559809479101277),
    ),
    (
        "ising",
        true,
        &[1, 2],
        Complex64::new(-5.795438516579738, 0.7743937638144475),
    ),
];

#[test]
fn flip_matches_values_checked_against_tensorkit() {
    let runtime = Runtime::builder().build().unwrap();
    for category in [fibonacci(), ising()] {
        let t = operand(&runtime, &category);
        for &(name, inverse, legs, want) in &FLIP_ROWS {
            if name == category.name {
                let result = t.flip(legs, direction(inverse)).unwrap();
                assert_close(
                    &format!("{name} flip {inverse} {legs:?}"),
                    probe(&runtime, &result),
                    want,
                );
            }
        }
    }
}

/// Every entry of the twisted tensor is the operand entry times the
/// closed-form `Π θ` of its block's uncoupled sectors on `legs` (θ̄ for the
/// inverse): θ_τ = e^{-4πi/5}; θ_σ = e^{iπ/8}, θ_ψ = −1.
#[test]
fn twist_scales_every_block_by_the_closed_form_theta() {
    let runtime = Runtime::builder().build().unwrap();
    for category in [fibonacci(), ising()] {
        let t = operand(&runtime, &category);
        for (inverse, legs) in TWISTS {
            let result = t.twist(legs, direction(inverse)).unwrap();
            let mut nontrivial = false;
            for (trees, block) in result.subblocks().unwrap() {
                let uncoupled = [trees.codomain_uncoupled(), trees.domain_uncoupled()].concat();
                let theta: Complex64 = legs
                    .iter()
                    .map(|&leg| (category.theta)(uncoupled[leg].label()))
                    .product();
                let factor = if inverse { theta.conj() } else { theta };
                nontrivial |= factor.im.abs() > 0.1;
                let shape = block.shape().to_vec();
                for linear in 0..shape.iter().product::<usize>() {
                    let mut rest = linear;
                    let index: Vec<usize> = shape
                        .iter()
                        .map(|&extent| {
                            let i = rest % extent;
                            rest /= extent;
                            i
                        })
                        .collect();
                    assert_close(
                        &format!("{} twist {inverse} {legs:?} {trees:?}", category.name),
                        *block.get(&index).unwrap(),
                        entry("t", &trees, &index) * factor,
                    );
                }
            }
            assert!(nontrivial, "{} {legs:?}: a complex θ acts", category.name);
        }
    }
}

/// TensorKit's identities: `twist(twist(t, I), I; inv) = t`, and
/// `flip(flip(t, I), I; inv) = t = flip(flip(t, I; inv), I)`
/// (`test/tensors/indexmanipulations.jl:103-104`). `flip` is not an
/// involution: flipping one leg twice scales each block by `χ_a θ_a`
/// (codomain) or `θ_a χ̄_a` (domain), which is the twist of that leg here
/// since every sector has χ = 1 (checked in TensorKit 0.17.1 too). So
/// `flip⁴` is `θ²`, not the identity, for these anyons.
#[test]
fn twist_and_flip_invert_as_tensorkit_defines() {
    let runtime = Runtime::builder().build().unwrap();
    let (f, b) = (Direction::Forward, Direction::Inverse);
    for category in [fibonacci(), ising()] {
        let t = operand(&runtime, &category);
        let name = category.name;
        for (_, legs) in TWISTS {
            let round = t.twist(legs, f).unwrap().twist(legs, b).unwrap();
            assert_same(&format!("{name} twist {legs:?}"), &round, &t);
        }
        for leg in 0..4 {
            let legs = &[leg];
            let once = t.flip(legs, f).unwrap();
            assert_same(
                &format!("{name} flip inv∘flip {leg}"),
                &once.flip(legs, b).unwrap(),
                &t,
            );
            assert_same(
                &format!("{name} flip∘flip inv {leg}"),
                &t.flip(legs, b).unwrap().flip(legs, f).unwrap(),
                &t,
            );
            assert_same(
                &format!("{name} flip∘flip {leg}"),
                &once.flip(legs, f).unwrap(),
                &t.twist(legs, f).unwrap(),
            );
        }
    }
}

/// A lazy adjoint twists and flips through its parent with the inverse
/// operation (conjugated θ), without materializing: the result equals the
/// same operation on the materialized adjoint.
#[test]
fn lazy_adjoint_twist_and_flip_equal_the_materialized_adjoint() {
    let runtime = Runtime::builder().build().unwrap();
    for category in [fibonacci(), ising()] {
        let t = operand(&runtime, &category);
        let lazy = t.adjoint().unwrap();
        assert!(lazy.dense_data().is_err(), "the adjoint is a lazy view");
        let eager = lazy.materialize().unwrap();
        let name = category.name;
        for (inverse, legs) in TWISTS {
            let got = lazy.twist(legs, direction(inverse)).unwrap();
            assert!(got.dense_data().is_err(), "{name} twist stays lazy");
            assert_same(
                &format!("{name} adjoint twist {inverse} {legs:?}"),
                &got,
                &eager.twist(legs, direction(inverse)).unwrap(),
            );
        }
        for (inverse, legs) in FLIPS {
            let got = lazy.flip(legs, direction(inverse)).unwrap();
            assert!(got.dense_data().is_err(), "{name} flip stays lazy");
            assert_same(
                &format!("{name} adjoint flip {inverse} {legs:?}"),
                &got,
                &eager.flip(legs, direction(inverse)).unwrap(),
            );
        }
    }
}

/// A compact diagonal stays compact (TensorKit `DiagonalTensorMap`): each
/// sector's values are scaled by the complex `θ(c)^{|legs|}` (θ̄ for the
/// inverse), and the result equals the dense twist of the materialized map.
#[test]
fn compact_diagonal_twist_scales_its_spectrum_by_complex_theta() {
    let runtime = Runtime::builder().build().unwrap();
    for category in [fibonacci(), ising()] {
        let sectors = category.v.sectors().unwrap();
        let spectrum = sectors
            .iter()
            .map(|&sector| SectorSpectrum {
                sector,
                values: (0..category.v.degeneracy(&sector).unwrap())
                    .map(|i| Complex64::new(1.0 + i as f64, 0.25 * f64::from(sector.label())))
                    .collect(),
            })
            .collect::<Vec<_>>();
        let compact = Table::diagonal(&runtime, &category.v, spectrum.clone()).unwrap();
        let dense = compact.materialize().unwrap();
        for legs in [&[0][..], &[1], &[0, 1]] {
            for inverse in [false, true] {
                let what = format!("{} compact twist {legs:?} {inverse}", category.name);
                let twisted = compact.twist(legs, direction(inverse)).unwrap();
                assert!(twisted.dense_data().is_err(), "{what}: stays compact");
                for (got, source) in twisted.diagview().unwrap().iter().zip(&spectrum) {
                    assert_eq!(got.sector, source.sector);
                    let theta = (category.theta)(source.sector.label()).powi(legs.len() as i32);
                    let factor = if inverse { theta.conj() } else { theta };
                    for (&x, &y) in got.values.iter().zip(&source.values) {
                        assert_close(&what, x, y * factor);
                    }
                }
                assert_same(
                    &what,
                    &twisted,
                    &dense.twist(legs, direction(inverse)).unwrap(),
                );
            }
        }
    }
}

/// Rep(S₃) has no braiding: a twist of unit legs is the identity, any other
/// twist and every flip is refused with the existing typed error, and the
/// input is untouched.
#[test]
fn no_braiding_table_keeps_its_typed_refusals() {
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(TableRule::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c").unwrap());
    let mixed = space(rule.clone(), &[(1, 2), (3, 1)]);
    let unit = space(rule, &[(1, 2)]);
    let t: Table = TensorMap::rand_with_seed(&runtime, [&mixed, &unit], [&mixed], 5).unwrap();
    let before = t.dense_data().unwrap().to_vec();

    assert_same(
        "unit-leg twist",
        &t.twist(&[1], Direction::Forward).unwrap(),
        &t,
    );
    let refused = |result: Result<Table, Error>, needle: &str| match result {
        Err(Error::InvalidArgument(message)) => {
            assert!(message.contains(needle), "{message}");
        }
        other => panic!("expected InvalidArgument, got {:?}", other.map(|_| ())),
    };
    refused(
        t.twist(&[0], Direction::Forward),
        "carries non-unit sectors but the fusion rule has no braiding",
    );
    refused(
        t.twist(&[2], Direction::Inverse),
        "carries non-unit sectors but the fusion rule has no braiding",
    );
    refused(
        t.flip(&[1], Direction::Forward),
        "needs the twist and Frobenius-Schur coefficients but the fusion rule has no braiding",
    );
    refused(
        t.adjoint().unwrap().flip(&[0], Direction::Inverse),
        "needs the twist and Frobenius-Schur coefficients but the fusion rule has no braiding",
    );
    assert_eq!(t.dense_data().unwrap(), &before[..]);
}

#[test]
fn fibonacci_flip_rejects_a_repeated_leg() {
    let runtime = Runtime::builder().build().unwrap();
    let t = operand(&runtime, &fibonacci());
    match t.flip(&[1, 1], Direction::Forward) {
        Err(Error::InvalidArgument(message)) => {
            assert!(message.contains("listed more than once"), "{message}");
        }
        other => panic!("expected InvalidArgument, got {:?}", other.map(|_| ())),
    }
}
