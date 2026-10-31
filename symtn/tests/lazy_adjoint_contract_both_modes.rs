//! A lazy adjoint is a storage-conjugate operand of contraction and
//! composition, and a tree transform of a lazy adjoint is the lazy adjoint
//! of the transformed parent, in both admission modes.
//!
//! TensorKit `cfaa073e`: `blas_contract!` (`src/tensors/tensoroperations.jl:383-450`)
//! contracts `A'` either as `mul!` over `block(parent, c)'` (no copy) or,
//! through `copyA`, as a `tensoradd!` from the adjoint's subblocks; `compose`
//! is `mul!` (`src/tensors/linalg.jl:38, 330-`). `permute(t'::AdjointTensorMap, p) =
//! adjoint(permute(parent(t'), adjointtensorindices(t', p)))`
//! (`src/tensors/indexmanipulations.jl:260-264`), and `braid` (L343-352),
//! `transpose` (L414-421) and `repartition` (through `transpose`) the same
//! way, for every fusion style. QSpace (`dd2cc7e1`) carries a conjugation
//! flag into its grouped block contraction (`contractQS.cc:icFlags`,
//! `wbarray.cc:3472-3482`) but has no lazy adjoint object (`QSpace::Permute`
//! conjugates eagerly), so TensorKit semantics govern.
//!
//! Oracles, each independent of the route under test:
//! - (i) reference steps: `copy(t')` (the uncached `materialize`), then the
//!   same operation on owned operands;
//! - (ii) cross-mode agreement: SU(2) admitted multiplicity-free
//!   (`SU2FusionRule`) against checked Generic (`LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap())`), two
//!   adapters to the same exact A1 provider on one input;
//! - SU(3) (checked only) with outer multiplicity two (`8 ⊗ 8 → 8`) and a
//!   dual leg against (i).
//!
//! Host only: checked Generic has no device API; the multiplicity-free
//! device lazy-adjoint gates live in `representation_gates/cuda.rs`.

#![cfg(feature = "cgc-gen")]

use std::collections::BTreeMap;
use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::{LieGroupFusionRule, SU2FusionRule, SU2Irrep};
use symtn::typed::__network::{network_reuse_class, NetworkReuseClass};
use symtn::typed::{BlockFusionTrees, Complex64, ContractSpec, GradedSpace, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

use crate::legs::su2_spin;

/// `(2j, k)` per sector: nontrivial, distinct degeneracies.
const SECTORS: [(usize, usize); 3] = [(0, 2), (1, 3), (2, 2)];

/// `(name, entries of the materialized result)` per transform.
type Rows = Vec<(&'static str, BTreeMap<String, Vec<Complex64>>)>;

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

/// A value fixed by the block key and local index only, so both SU(2)
/// providers fill the same reduced entries.
fn value(key: &str, index: &[usize]) -> f64 {
    let hash = format!("{key}{index:?}")
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    ((hash % 2001) as f64 - 1000.0) / 257.0
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

/// The lazy adjoint of `$source` through every tree transform: each result
/// stays a lazy adjoint, leaves its input lazy, and equals oracle (i).
macro_rules! transform_rows {
    ($engine:ident, $source:expr, $key:expr) => {{
        let key = $key;
        let entries = |t: &TensorMap<_, _>| entries!(t, key);
        let lazy = $source.adjoint($engine).unwrap();
        let eager = lazy.materialize($engine).unwrap();
        let mut rows: Rows = Vec::new();
        macro_rules! row {
            ($name:expr, |$t:ident| $op:expr) => {{
                assert!(matches!(
                    network_reuse_class(&lazy, false),
                    NetworkReuseClass::LazyAdjoint
                ));
                let actual = {
                    let $t = &lazy;
                    $op.unwrap()
                };
                let expected = {
                    let $t = &eager;
                    $op.unwrap()
                };
                assert!(matches!(
                    network_reuse_class(&lazy, false),
                    NetworkReuseClass::LazyAdjoint
                ));
                assert!(
                    matches!(
                        network_reuse_class(&actual, false),
                        NetworkReuseClass::LazyAdjoint
                    ),
                    "{}",
                    $name
                );
                let actual = entries(&actual.materialize($engine).unwrap());
                assert_close($name, &actual, &entries(&expected));
                rows.push(($name, actual));
            }};
        }
        row!("permute", |t| t.permute($engine, &[2, 0], &[1]));
        row!("braid", |t| t.braid($engine, &[2, 0], &[1], &[17, 3, 11]));
        row!("transpose", |t| t.transpose($engine, &[2, 1], &[0]));
        row!("repartition", |t| t.repartition($engine, 2));
        rows
    }};
}

/// Contractions and compositions with `$source`'s lazy adjoint `a` (on
/// either side, both, and beside its own parent `t`), over the canonical
/// core, a non-core contracted order and a permuted output: each equals
/// oracle (i) and leaves its lazy inputs lazy.
macro_rules! contract_rows {
    ($engine:ident, $source:expr, $key:expr) => {{
        let key = $key;
        let t = &$source;
        let lazy = t.adjoint($engine).unwrap();
        let eager = lazy.materialize($engine).unwrap();
        let mut rows: Rows = Vec::new();
        macro_rules! row {
            ($name:expr, |$a:ident| $op:expr) => {{
                let lazy_input = || {
                    assert!(matches!(
                        network_reuse_class(&lazy, false),
                        NetworkReuseClass::LazyAdjoint
                    ));
                };
                lazy_input();
                let actual = {
                    let $a = &lazy;
                    $op.unwrap()
                };
                lazy_input();
                let expected = {
                    let $a = &eager;
                    $op.unwrap()
                };
                let actual = entries!(actual, key);
                assert!(!actual.is_empty(), "{}", $name);
                assert_close($name, &actual, &entries!(expected, key));
                rows.push(($name, actual));
            }};
        }
        // `a = t'`: [V] <- [V, V], `t`: [V, V] <- [V].
        let spec = |lhs, rhs, codomain, domain| ContractSpec {
            lhs,
            rhs,
            codomain,
            domain,
        };
        row!("a·t core (t'·t)", |a| a.contract(
            $engine,
            t,
            &spec(&[1, 2], &[0, 1], &[0], &[1])
        ));
        row!("t·a core", |a| t.contract(
            $engine,
            a,
            &spec(&[2], &[0], &[0, 1], &[2, 3])
        ));
        row!("a·a", |a| a.contract(
            $engine,
            a,
            &spec(&[1], &[0], &[0, 1], &[2, 3])
        ));
        // Both operands conjugate and bend: each `a` contracts its codomain
        // leg against the other's domain leg.
        row!("a·a mixed legs", |a| a.contract(
            $engine,
            a,
            &spec(&[0, 1], &[1, 0], &[1], &[0])
        ));
        row!("a·t reversed contracted order", |a| a.contract(
            $engine,
            t,
            &spec(&[2, 1], &[1, 0], &[0], &[1])
        ));
        row!("a·t permuted output", |a| a.contract(
            $engine,
            t,
            &spec(&[1, 2], &[0, 1], &[], &[1, 0])
        ));
        row!("t·a permuted output", |a| t.contract(
            $engine,
            a,
            &spec(&[2], &[0], &[3, 0], &[2, 1])
        ));
        row!("a·t mixed legs", |a| a.contract(
            $engine,
            t,
            &spec(&[0, 1], &[2, 0], &[0], &[1])
        ));
        row!("t·a mixed legs", |a| t.contract(
            $engine,
            a,
            &spec(&[2, 0], &[0, 1], &[1], &[0])
        ));
        // `a` is a core operand as is (identity transform) while `t`
        // is permuted, so the DynamicTree core reads `a`'s parent with a
        // GEMM adjoint op instead of copying it.
        let u = t.permute($engine, &[0], &[1, 2]).unwrap();
        let w = t.permute($engine, &[2, 0], &[1]).unwrap();
        row!("a·u identity adjoint, transformed u", |a| a.contract(
            $engine,
            &u,
            &spec(&[1, 2], &[0, 1], &[0], &[1])
        ));
        row!("w·a identity adjoint, transformed w", |a| w.contract(
            $engine,
            a,
            &spec(&[0], &[0], &[0, 1], &[2, 3])
        ));
        // M1 (keep `a`'s contracted order: `a` free, `t` permuted)
        // and m2 (keep `t`'s: `a` permuted) both cost |t| under TensorKit
        // `contract_memcost`, and the tie goes to m1; charging the identity
        // adjoint as a copy took m2 instead.
        row!("a·t identity adjoint, reversed t order", |a| a.contract(
            $engine,
            t,
            &spec(&[1, 2], &[1, 0], &[0], &[1])
        ));
        row!("a∘t", |a| a.compose($engine, t));
        row!("t∘a", |a| t.compose($engine, a));
        rows
    }};
}

#[expect(clippy::ptr_arg, reason = "the SU(N) sector label is a `Vec<i64>`")]
fn sun_spin(sector: &Vec<i64>) -> usize {
    sector[0] as usize
}

fn su2_sources(
    engine: &mut dyn Engine,
) -> (
    TensorMap<SU2FusionRule, f64>,
    TensorMap<LieGroupFusionRule, f64>,
) {
    let runtime = host_runtime();
    let mf = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
    )
    .unwrap();
    let checked = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap()),
        SECTORS.map(|(s, k)| (vec![s as i64], k)),
    )
    .unwrap();
    (
        TensorMap::from_subblock_fn(engine, &runtime, [&mf, &mf], [&mf], |trees, index| {
            value(&twice(trees, su2_spin), index)
        })
        .unwrap(),
        TensorMap::from_subblock_fn(
            engine,
            &runtime,
            [&checked, &checked],
            [&checked],
            |trees, index| value(&twice(trees, sun_spin), index),
        )
        .unwrap(),
    )
}

fn assert_modes_agree(mf: Rows, checked: Rows) {
    for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
        assert_close(
            &format!("{name}: checked vs multiplicity-free"),
            checked,
            mf,
        );
    }
}

#[test]
fn su2_adjoint_transforms_stay_lazy_and_agree_across_modes_f64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (mf, checked) = su2_sources(engine);
    let mf = transform_rows!(engine, mf, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    let checked = transform_rows!(engine, checked, |t: &BlockFusionTrees<Vec<i64>>| twice(
        t, sun_spin
    ));
    assert_modes_agree(mf, checked);
}

#[test]
fn su2_adjoint_transforms_stay_lazy_and_agree_across_modes_c64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // `1 + 2i` makes the conjugation visible.
    let phase = Complex64::new(1.0, 2.0);
    let (mf, checked) = su2_sources(engine);
    let (mf, checked) = (
        mf.convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, phase)
            .unwrap(),
        checked
            .convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, phase)
            .unwrap(),
    );
    let mf = transform_rows!(engine, mf, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    let checked = transform_rows!(engine, checked, |t: &BlockFusionTrees<Vec<i64>>| twice(
        t, sun_spin
    ));
    assert_modes_agree(mf, checked);
}

#[test]
fn su3_adjoint_transforms_with_outer_multiplicity_stay_lazy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = host_runtime();
    let v = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v], |trees, index| {
            value(&format!("{trees:?}"), index)
        })
        .unwrap();
    let key = |t: &BlockFusionTrees<Vec<i64>>| format!("{t:?}");
    let f64_rows = transform_rows!(engine, source, key);
    let c64 = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 2.0))
        .unwrap();
    let c64_rows = transform_rows!(engine, c64, key);
    assert_eq!(f64_rows.len(), c64_rows.len());
    assert!(f64_rows.iter().all(|(_, entries)| !entries.is_empty()));
}

#[test]
fn su2_adjoint_operands_contract_and_agree_across_modes_f64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (mf, checked) = su2_sources(engine);
    let mf = contract_rows!(engine, mf, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    let checked = contract_rows!(engine, checked, |t: &BlockFusionTrees<Vec<i64>>| twice(
        t, sun_spin
    ));
    assert_modes_agree(mf, checked);
}

#[test]
fn su2_adjoint_operands_contract_and_agree_across_modes_c64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let phase = Complex64::new(1.0, 2.0);
    let (mf, checked) = su2_sources(engine);
    let (mf, checked) = (
        mf.convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, phase)
            .unwrap(),
        checked
            .convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, phase)
            .unwrap(),
    );
    let mf = contract_rows!(engine, mf, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    let checked = contract_rows!(engine, checked, |t: &BlockFusionTrees<Vec<i64>>| twice(
        t, sun_spin
    ));
    assert_modes_agree(mf, checked);
}

#[test]
fn su3_adjoint_operands_with_outer_multiplicity_contract() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = host_runtime();
    let v = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v], |trees, index| {
            value(&format!("{trees:?}"), index)
        })
        .unwrap();
    let key = |t: &BlockFusionTrees<Vec<i64>>| format!("{t:?}");
    let f64_rows = contract_rows!(engine, source, key);
    let c64 = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 2.0))
        .unwrap();
    let c64_rows = contract_rows!(engine, c64, key);
    assert_eq!(f64_rows.len(), c64_rows.len());
}

#[test]
fn su3_adjoint_transforms_with_a_dual_leg_stay_lazy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = host_runtime();
    let v = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    let dual = v.try_dual().unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &dual], [&v], |trees, index| {
            value(&format!("{trees:?}"), index)
        })
        .unwrap();
    let key = |t: &BlockFusionTrees<Vec<i64>>| format!("{t:?}");
    let f64_rows = transform_rows!(engine, source, key);
    let c64 = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 2.0))
        .unwrap();
    let c64_rows = transform_rows!(engine, c64, key);
    assert_eq!(f64_rows.len(), c64_rows.len());
    assert!(f64_rows.iter().all(|(_, entries)| !entries.is_empty()));
}
