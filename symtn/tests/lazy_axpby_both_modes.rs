//! `axpby` and `axpby_into` with lazy-adjoint operands follow one rule in both
//! admission modes (#1874): one lazy and one owned operand (either order)
//! give an owned result, two lazy operands the lazy adjoint of their parents'
//! combination (A10), and a lazy `axpby_into` source is read in its logical
//! orientation.
//!
//! TensorKit `cfaa073e`: `VectorInterface.add` and `add!(ty, tx, α, β)`
//! (`src/tensors/vectorinterface.jl:67, 84`) loop over blocks, where an
//! `AdjointTensorMap` block is its parent's block adjoint
//! (`src/tensors/adjoint.jl:47-51` `subblock`); the CUDA extension adds no
//! method. QSpace (`dd2cc7e1`) has no lazy adjoint (`QSpace::plus_plain`
//! adds eager operands), so TensorKit semantics govern.
//!
//! Oracles, each independent of the route under test:
//! - (i) reference steps: `materialize` every lazy operand, then the same
//!   operation on owned operands;
//! - (ii) cross-mode agreement: SU(2) admitted multiplicity-free
//!   (`SU2FusionRule`) against checked Generic (`LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap())`).
//!
//! The device rows (multiplicity-free only) are `storage_parity_matrix.rs`
//! `mixed_lazy_axpby_cuda_rows_match_host`.

#![cfg(feature = "cgc-gen")]

use std::collections::BTreeMap;
use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::{LieGroupFusionRule, SU2FusionRule, SU2Irrep};
use symtn::typed::{BlockFusionTrees, Complex64, GradedSpace, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

/// `(2j, k)` per sector: nontrivial, distinct degeneracies.
const SECTORS: [(usize, usize); 3] = [(0, 2), (1, 3), (2, 2)];

type Rows = Vec<(String, BTreeMap<String, Vec<Complex64>>)>;

fn twice<S>(trees: &BlockFusionTrees<S>, spin: impl Fn(&S) -> usize) -> String {
    let map = |labels: &[S]| labels.iter().map(&spin).collect::<Vec<_>>();
    format!(
        "{:?}",
        (
            map(trees.codomain_uncoupled()),
            map(trees.codomain_innerlines()),
            map(trees.domain_uncoupled()),
            map(trees.domain_innerlines()),
            spin(trees.coupled()),
        )
    )
}

/// A value fixed by the block key, local index and `seed` only, so both
/// SU(2) providers fill the same reduced entries.
fn value(seed: u64, key: &str, index: &[usize]) -> Complex64 {
    let hash = format!("{seed}{key}{index:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    Complex64::new(
        ((hash % 2001) as f64 - 1000.0) / 257.0,
        (((hash >> 20) % 2001) as f64 - 1000.0) / 311.0,
    )
}

fn assert_close(
    what: &str,
    got: &BTreeMap<String, Vec<Complex64>>,
    want: &BTreeMap<String, Vec<Complex64>>,
) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{what}: blocks"
    );
    for (key, want) in want {
        let got = &got[key];
        assert_eq!(got.len(), want.len(), "{what} {key}: length");
        for (g, w) in got.iter().zip(want) {
            assert!(g.re.is_finite() && g.im.is_finite(), "{what} {key}: {g}");
            assert!(
                (g - w).norm() <= 1e-10 * (1.0 + w.norm()),
                "{what} {key}: {g} vs {w}"
            );
        }
    }
}

/// The subblock entries of an owned or lazy tensor, keyed by `key`.
macro_rules! entries {
    ($t:expr, $key:expr) => {{
        let key = $key;
        $t.subblocks()
            .unwrap()
            .map(|(trees, view)| {
                let shape = view.shape().to_vec();
                let len = shape.iter().product::<usize>();
                let values = (0..len)
                    .map(|mut linear| {
                        let index: Vec<usize> = shape
                            .iter()
                            .map(|&extent| {
                                let i = linear % extent;
                                linear /= extent;
                                i
                            })
                            .collect();
                        Complex64::from(*view.get(&index).unwrap())
                    })
                    .collect();
                (key(&trees), values)
            })
            .collect::<BTreeMap<String, Vec<Complex64>>>()
    }};
}

/// Every lazy/owned mix of `axpby` and a lazy `axpby_into` source on
/// `[V] <- [V, V]` (the adjoint of `[V, V] <- [V]`), against oracle (i), for
/// a general pair, a zero `alpha`, a zero `beta` (over a NaN destination for
/// `axpby_into`) and a unit pair.
macro_rules! axpby_rows {
    ($engine:ident, $rule:expr, $label:expr, $spin:expr, $D:ty, $conv:expr) => {{
        let runtime = host_runtime();
        let spin = $spin;
        let conv = $conv;
        let key = |trees: &BlockFusionTrees<_>| twice(trees, spin);
        let v =
            GradedSpace::try_new(Arc::new($rule), SECTORS.map(|(s, k)| ($label(s), k))).unwrap();
        let build = |$engine: &mut dyn Engine, seed: u64| {
            TensorMap::<_, $D>::from_subblock_fn($engine, &runtime, [&v, &v], [&v], |trees, ix| {
                conv(value(seed, &twice(trees, spin), ix))
            })
            .unwrap()
        };
        let (x, y) = (
            build($engine, 1).adjoint($engine).unwrap(),
            build($engine, 2).adjoint($engine).unwrap(),
        );
        let (x_owned, y_owned) = (
            x.materialize($engine).unwrap(),
            y.materialize($engine).unwrap(),
        );
        let nan = conv(Complex64::new(f64::NAN, f64::NAN));
        let poisoned = y_owned.scale($engine, nan).unwrap();
        let pairs = [
            (Complex64::new(0.7, -0.3), Complex64::new(-1.1, 0.4)),
            (Complex64::new(0.0, 0.0), Complex64::new(-1.1, 0.4)),
            (Complex64::new(0.7, -0.3), Complex64::new(0.0, 0.0)),
            (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
        ];
        let mut rows: Rows = Vec::new();
        for (alpha, beta) in pairs {
            let zero_beta = beta == Complex64::new(0.0, 0.0);
            let (alpha, beta): ($D, $D) = (conv(alpha), conv(beta));
            let mut row = |$engine: &mut dyn Engine,
                           name: &str,
                           actual: TensorMap<_, $D>,
                           expected: TensorMap<_, $D>| {
                let name = format!("{name} alpha {alpha:?} beta {beta:?}");
                let actual = entries!(actual.materialize($engine).unwrap(), key);
                assert_close(&name, &actual, &entries!(expected, key));
                rows.push((name, actual));
            };
            let oracle = x_owned.axpby($engine, alpha, &y_owned, beta).unwrap();
            let lazy_owned = x.axpby($engine, alpha, &y_owned, beta).unwrap();
            assert!(lazy_owned.dense_data().is_ok(), "mixed gives owned");
            row($engine, "lazy + owned", lazy_owned, oracle.clone());
            let owned_lazy = x_owned.axpby($engine, alpha, &y, beta).unwrap();
            assert!(owned_lazy.dense_data().is_ok(), "mixed gives owned");
            row($engine, "owned + lazy", owned_lazy, oracle.clone());
            let lazy_lazy = x.axpby($engine, alpha, &y, beta).unwrap();
            assert!(
                lazy_lazy.dense_data().is_err(),
                "A10: lazy + lazy stays lazy"
            );
            row($engine, "lazy + lazy", lazy_lazy, oracle);

            let start = if zero_beta { &poisoned } else { &y_owned };
            let mut actual = start.materialize($engine).unwrap();
            x.axpby_into($engine, &mut actual, alpha, beta).unwrap();
            let mut expected = start.materialize($engine).unwrap();
            x_owned
                .axpby_into($engine, &mut expected, alpha, beta)
                .unwrap();
            row($engine, "lazy axpby_into", actual, expected);
        }
        rows
    }};
}

fn sun2(twice: usize) -> Vec<i64> {
    vec![twice as i64]
}

#[expect(clippy::ptr_arg, reason = "the SU(N) sector label is a `Vec<i64>`")]
fn sun_spin(sector: &Vec<i64>) -> usize {
    sector[0] as usize
}

fn su2_spin(sector: &SU2Irrep) -> usize {
    sector.twice_spin()
}

fn assert_modes_agree(mf: Rows, checked: Rows) {
    assert_eq!(mf.len(), checked.len());
    for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
        assert_close(
            &format!("{name}: checked vs multiplicity-free"),
            checked,
            mf,
        );
    }
}

#[test]
fn lazy_axpby_rows_agree_across_modes_f64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let real = |z: Complex64| z.re;
    let mf = axpby_rows!(
        engine,
        SU2FusionRule,
        SU2Irrep::from_twice_spin,
        su2_spin,
        f64,
        real
    );
    let checked = axpby_rows!(
        engine,
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap(),
        sun2,
        sun_spin,
        f64,
        real
    );
    assert_modes_agree(mf, checked);
}

#[test]
fn lazy_axpby_rows_agree_across_modes_c64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let complex = |z: Complex64| z;
    let mf = axpby_rows!(
        engine,
        SU2FusionRule,
        SU2Irrep::from_twice_spin,
        su2_spin,
        Complex64,
        complex
    );
    let checked = axpby_rows!(
        engine,
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap(),
        sun2,
        sun_spin,
        Complex64,
        complex
    );
    assert_modes_agree(mf, checked);
}
