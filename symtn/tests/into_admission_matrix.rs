//! `*_into` admits what eager admits, in every admission mode (#1870).
//!
//! TensorKit `cfaa073e`: `permute!`/`braid!`/`transpose!`/`repartition!`
//! (`src/tensors/indexmanipulations.jl:217, 303, 374, 435`) check the
//! destination space and run `add_transform!` (L547) with `α, β` for every
//! fusion style; `tensorcontract!` (`src/tensors/tensoroperations.jl:126`)
//! checks it and runs `blas_contract!` (L383), which scales a
//! `DiagonalTensorMap` operand's `Diagonal` blocks with `mul!` on the shared
//! pairing and copies it otherwise (L399-409). QSpace (`dd2cc7e1`) has no
//! accumulate-into-existing form
//! (`QSpace::Permute` is in place, `contract` overwrites), so TensorKit
//! semantics govern.
//!
//! Each cell runs one `_into` on an existing destination `dst₀` and asserts:
//! 1. `dst == α·eager + β·dst₀` within tolerance, with `β = 0` seeded by NaN
//!    (never read) and `β ∈ {0.5, 1}` by distinct finite values; for an owned
//!    source the reference-step oracle (`materialize`, then eager) is eager
//!    itself;
//! 2. multiplicity-free SU(2) (`SU2FusionRule`) == checked Generic SU(2)
//!    (`SUNFusionRule::new(2)`), two adapters to the same exact A1 provider on one input;
//! 3. the destination keeps its payload allocation and provider.
//!
//! 4. the reference-step oracle: eager on the materialized source equals
//!    eager on the source;
//! 5. the source stays cold: a lazy adjoint or compact diagonal is not
//!    materialized by the call.
//!
//! Rows: owned dense and lazy-adjoint rank-(2, 2) sources
//! × {permute, braid, transpose, repartition}; compact rank-(1, 1) diagonal
//! sources × {identity permute, swap permute, swap transpose,
//! braid, repartition to (0, 2) and (2, 0)}. Columns: multiplicity-free real
//! symbols (U(1), SU(2), fermion parity, each with dual legs),
//! multiplicity-free complex symbols (Fibonacci), checked Generic (SU(2),
//! SU(3) with outer multiplicity) × {f64, Complex64 scaled by `1 + 2i`}.
//! Contraction rows: `[v, w'] ← [v, w']` against `[v, w'] ← [v]`
//! over the domain as owned, lazy-adjoint and adjoint-view (`other`) operands
//! in the core and two reordered output orders; and every compact arm
//! pattern (`t · D`, `D · t` with a direct and a lazy `t`, `D · D`) in its own
//! order and reordered, plus a two-axis contraction the arm declines. The
//! oracle contracts the materialized operands; reordered `D · D` is the row
//! that proves `contract = permute(compose)` there (eager takes the dense
//! route).
//! Complex-symbol contraction rows: a `Complex64` embedding of U(1)
//! equals the real-symbol lane (exactly, up to the sign of zero) on owned
//! operands in every order, plus its compact arms and their decline; a
//! Bosonic Z3 × Z3 with an alternating-bicharacter `R` matches a hand
//! calculation of the braiding phase; Fibonacci `contract_into` is refused
//! like eager.
//! Trace rows: TensorKit `tensortrace!` → `trace_permute!`
//! (`tensoroperations.jl:79, 208`) checks `space(tdst)` before the pair
//! duality and runs `_trace_permute!` (L242, L267) with its `β` pass first.
//! Owned and lazy-adjoint `[v, w'] ← [v, w']` sources × {both partial
//! traces, the full trace}, and a compact diagonal traced over its only pair
//! (its spectrum read from the compiled terms), in every real-symbol column.
//!
//! A compact cell's destination is dense: its off-diagonal entries are
//! `β·dst₀` where the dense replay writes `α·(c·0) + β·dst₀`, so cells compare
//! within tolerance (signed zeros may differ).
//!
//! Compile-time capability absences, recorded and not counted as passing
//! cells: checked Generic and complex-symbol rules × CUDA (the device
//! transforms are bounded `MultiplicityFreeRigidSymbols<Scalar = f64>`,
//! #1756), and a complex-symbol rule × trace (neither eager nor `_into` has
//! one). The lazy adjoint of a complex-symbol rule is covered against its
//! materialized adjoint in `complex_symbol_lazy_adjoint.rs`. Fibonacci `permute` is anyonic and asserted as the
//! same rejection as eager.

use std::collections::BTreeMap;
use std::sync::Arc;

use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, TableRule, TableSector, U1FusionRule,
    U1Irrep, Z2Irrep,
};
use symtn::typed::{
    BlockFusionTrees, Complex64, ContractSpec, GradedSpace, SectorSpectrum, TensorMap,
};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

use fixtures::host_runtime;

const BETAS: [f64; 3] = [0.0, 0.5, 1.0];

/// `(cell, entries of the destination after the call)`.
type Rows = Vec<(String, BTreeMap<String, Vec<Complex64>>)>;

trait Lane: Copy + Into<Complex64> {
    fn lift(x: f64) -> Self;
    /// `alpha`: a non-real phase on the complex lane.
    fn alpha() -> Self;
}

impl Lane for f64 {
    fn lift(x: f64) -> Self {
        x
    }
    fn alpha() -> Self {
        1.5
    }
}

impl Lane for Complex64 {
    fn lift(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn alpha() -> Self {
        Complex64::new(1.5, -0.5)
    }
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

fn close(got: Complex64, want: Complex64) -> bool {
    (got - want).norm() <= 1e-10 * (1.0 + want.norm())
}

/// The dense image of `t`, materializing a lazy adjoint or compact diagonal.
fn dense<R, D: Lane + symtn::typed::TensorScalar>(t: &TensorMap<R, D>) -> Vec<Complex64> {
    t.materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|&x| x.into())
        .collect()
}

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

fn su2_spin(sector: &SU2Irrep) -> usize {
    sector.twice_spin()
}

#[cfg(feature = "cgc-gen")]
#[expect(clippy::ptr_arg, reason = "the SU(N) sector label is a `Vec<i64>`")]
fn sun_spin(sector: &Vec<i64>) -> usize {
    sector[0] as usize
}

/// The subblock entries of a tensor, keyed by `key`.
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
                        (*view.get(&index).unwrap()).into()
                    })
                    .collect::<Vec<Complex64>>();
                (key(&trees), values)
            })
            .collect::<BTreeMap<String, Vec<Complex64>>>()
    }};
}

/// One cell per `β`: `$into` on a fresh `dst₀` on eager's legs equals
/// `α·eager + β·dst₀` and keeps the destination's allocation and provider.
macro_rules! cell {
    ($rows:expr, $runtime:expr, $source:expr, $key:expr, $d:ty, $name:expr,
     |$t:ident| $eager:expr, |$dst:ident, $a:ident, $b:ident| $into:expr) => {
        cell!(dense; $rows, $runtime, $source, $key, $d, $name, |$t| $eager, |$dst, $a, $b| $into)
    };
    ($kind:ident; $rows:expr, $runtime:expr, $source:expr, $key:expr, $d:ty, $name:expr,
     |$t:ident| $eager:expr, |$dst:ident, $a:ident, $b:ident| $into:expr) => {{
        let cold = $source.dense_data().is_err();
        let eager = {
            let $t = &$source;
            $eager.unwrap()
        };
        let oracle = {
            let materialized = $source.materialize().unwrap();
            let $t = &materialized;
            $eager.unwrap()
        };
        beta_cells!(
            $kind,
            $rows,
            $runtime,
            eager,
            oracle,
            $key,
            $d,
            $name,
            |$dst, $a, $b| $into,
            || assert_eq!(
                $source.dense_data().is_err(),
                cold,
                "{}: source representation",
                $name
            )
        );
    }};
}

/// One cell per `β` for `$lhs.contract_into($rhs_arg, ..)`: the oracle
/// contracts the materialized operands, and both operands stay cold.
macro_rules! contract_cell {
    ($rows:expr, $runtime:expr, $lhs:expr, $rhs:expr, $rhs_arg:expr, $spec:expr, $key:expr,
     $d:ty, $name:expr) => {{
        let spec = $spec;
        let cold = ($lhs.dense_data().is_err(), $rhs.dense_data().is_err());
        let eager = $lhs.contract($rhs_arg, &spec).unwrap();
        let oracle = $lhs
            .materialize()
            .unwrap()
            .contract(&$rhs.materialize().unwrap(), &spec)
            .unwrap();
        beta_cells!(
            dense,
            $rows,
            $runtime,
            eager,
            oracle,
            $key,
            $d,
            $name,
            |d, a, b| $lhs.contract_into($rhs_arg, &spec, d, a, b),
            || assert_eq!(
                ($lhs.dense_data().is_err(), $rhs.dense_data().is_err()),
                cold,
                "{}: operand representation",
                $name
            )
        );
    }};
}

/// A fresh `dst₀` on `$eager`'s legs filled by `$seed`. `traced` also
/// takes a rank-0 (full-trace) result, which has no leg for
/// `from_subblock_fn`: its one entry is seeded by scaling eager's own into a
/// fresh payload (`scale` needs a real-symbol rule, so `dense` cells, which
/// include complex-symbol rules, never name it).
macro_rules! seeded_destination {
    (dense, $runtime:expr, $eager:expr, $seed:expr) => {
        TensorMap::from_subblock_fn($runtime, &$eager.codomain(), &$eager.domain(), $seed).unwrap()
    };
    (traced, $runtime:expr, $eager:expr, $seed:expr) => {
        if $eager.rank() == 0 {
            let (trees, _) = $eager.subblocks().unwrap().next().unwrap();
            $eager.scale($seed(&trees, &[])).unwrap()
        } else {
            seeded_destination!(dense, $runtime, $eager, $seed)
        }
    };
}

/// The `β` loop of a cell: `eager` matches the reference-step `oracle`,
/// then `$into` on a fresh `dst₀` equals `α·eager + β·dst₀`, keeps the
/// destination's allocation and provider, and `$cold` holds.
macro_rules! beta_cells {
    ($kind:ident, $rows:expr, $runtime:expr, $eager:expr, $oracle:expr, $key:expr, $d:ty, $name:expr,
     |$dst:ident, $a:ident, $b:ident| $into:expr, $cold:expr) => {{
        let key = $key;
        let eager = $eager;
        let expected_eager = dense(&eager);
        let oracle = dense(&$oracle);
        assert_eq!(oracle.len(), expected_eager.len(), "{}", $name);
        for (i, (&o, &e)) in oracle.iter().zip(&expected_eager).enumerate() {
            assert!(close(e, o), "{} [{i}]: eager {e} vs oracle {o}", $name);
        }
        for beta in BETAS {
            let seed = |trees: &BlockFusionTrees<_>, index: &[usize]| {
                if beta == 0.0 {
                    <$d>::lift(f64::NAN)
                } else {
                    <$d>::lift(value(&format!("dst{}", key(trees)), index))
                }
            };
            let mut destination: TensorMap<_, $d> =
                seeded_destination!($kind, $runtime, eager, seed);
            let before = dense(&destination);
            let payload = destination.dense_data().unwrap().as_ptr();
            let provider = destination.provider() as *const _;
            {
                let ($dst, $a, $b) = (&mut destination, <$d>::alpha(), <$d>::lift(beta));
                $into.unwrap();
            }
            let what = format!("{} beta {beta}", $name);
            assert_eq!(
                destination.dense_data().unwrap().as_ptr(),
                payload,
                "{what}"
            );
            assert!(std::ptr::eq(destination.provider(), provider), "{what}");
            let alpha: Complex64 = <$d>::alpha().into();
            let got = dense(&destination);
            assert_eq!(got.len(), expected_eager.len(), "{what}");
            for (i, (&got, &e)) in got.iter().zip(&expected_eager).enumerate() {
                let want = alpha * e
                    + if beta == 0.0 {
                        Complex64::new(0.0, 0.0)
                    } else {
                        before[i] * beta
                    };
                assert!(close(got, want), "{what} [{i}]: {got} vs {want}");
            }
            ($cold)();
            $rows.push((what, entries!(destination, key)));
        }
    }};
}

/// Every tree-transform `_into` of a compact rank-(1, 1) diagonal `$source`.
macro_rules! bond_cells {
    ($runtime:expr, $source:expr, $key:expr, $d:ty) => {{
        let mut rows: Rows = Vec::new();
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "bond identity permute",
            |t| t.permute(&[0], &[1]),
            |d, a, b| $source.permute_into(&[0], &[1], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "bond swap permute",
            |t| t.permute(&[1], &[0]),
            |d, a, b| $source.permute_into(&[1], &[0], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "bond swap transpose",
            |t| t.transpose(&[1], &[0]),
            |d, a, b| $source.transpose_into(&[1], &[0], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "bond braid",
            |t| t.braid(&[1], &[0], &[0, 1]),
            |d, a, b| $source.braid_into(&[1], &[0], &[0, 1], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "bond repartition (0, 2)",
            |t| t.repartition(0),
            |d, a, b| $source.repartition_into(d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "bond repartition (2, 0)",
            |t| t.repartition(2),
            |d, a, b| $source.repartition_into(d, a, b)
        );
        rows
    }};
}

/// A compact diagonal on `$leg` whose entries are fixed by the sector's
/// provider-neutral `$name` and the index only, lifted by `$lift`.
macro_rules! bond {
    ($runtime:expr, $leg:expr, $sectors:expr, $name:expr, $d:ty, $lift:expr) => {{
        let (lift, name) = ($lift, $name);
        let spectra = $sectors
            .into_iter()
            .map(|(sector, k)| SectorSpectrum {
                values: (0..k)
                    .map(|i| lift(value(&name(&sector), &[i])))
                    .collect::<Vec<$d>>(),
                sector,
            })
            .collect::<Vec<_>>();
        TensorMap::<_, $d>::diagonal($runtime, &$leg, spectra).unwrap()
    }};
}

/// Every tree-transform `_into` of an owned dense rank-(2, 2) `$source`.
macro_rules! transform_cells {
    ($runtime:expr, $source:expr, $key:expr, $d:ty) => {{
        let mut rows: Rows = Vec::new();
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "permute",
            |t| t.permute(&[2, 0], &[1, 3]),
            |d, a, b| $source.permute_into(&[2, 0], &[1, 3], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "braid",
            |t| t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]),
            |d, a, b| $source.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "transpose",
            |t| t.transpose(&[1, 3], &[0, 2]),
            |d, a, b| $source.transpose_into(&[1, 3], &[0, 2], d, a, b)
        );
        cell!(
            rows,
            $runtime,
            $source,
            $key,
            $d,
            "repartition",
            |t| t.repartition(1),
            |d, a, b| $source.repartition_into(d, a, b)
        );
        rows
    }};
}

/// A rank-(2, 2) source `[v, w'] ← [v, w']` with a dual leg on each side,
/// as f64.
macro_rules! source {
    ($runtime:expr, $v:expr, $w:expr, $key:expr) => {{
        let key = $key;
        let w = $w.try_dual().unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($runtime, [&$v, &w], [&$v, &w], |trees, index| {
                value(&key(trees), index)
            })
            .unwrap();
        source
    }};
}

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    (
        GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 3),
                (U1Irrep::new(1), 2),
            ],
        )
        .unwrap(),
        GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap(),
    )
}

/// `(2j, k)` per sector: nontrivial, distinct degeneracies.
const SU2_SECTORS: [(usize, usize); 3] = [(0, 2), (1, 3), (2, 2)];

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
    )
    .unwrap()
}

#[cfg(feature = "cgc-gen")]
fn sun2_leg() -> GradedSpace<symtn::sector::SUNFusionRule> {
    GradedSpace::try_new(
        Arc::new(symtn::sector::SUNFusionRule::new(2).unwrap()),
        SU2_SECTORS.map(|(s, k)| (vec![s as i64], k)),
    )
    .unwrap()
}

fn fz2_legs() -> (
    GradedSpace<FermionParityFusionRule>,
    GradedSpace<FermionParityFusionRule>,
) {
    let rule = Arc::new(FermionParityFusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)]).unwrap(),
        GradedSpace::try_new(rule, [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)]).unwrap(),
    )
}

fn debug_key<S: std::fmt::Debug>(trees: &BlockFusionTrees<S>) -> String {
    format!("{trees:?}")
}

/// The owned dense and lazy-adjoint rows of both lanes:
/// `[owned f64, owned c64, lazy f64, lazy c64]`.
macro_rules! both_lanes {
    ($runtime:expr, $v:expr, $w:expr, $key:expr) => {{
        let f64_rows = {
            let source = source!($runtime, $v, $w, $key);
            transform_cells!($runtime, source, $key, f64)
        };
        // `1 + 2i` makes a lost or doubled conjugation visible.
        let c64 = || {
            source!($runtime, $v, $w, $key)
                .convert::<Complex64>()
                .scale(Complex64::new(1.0, 2.0))
                .unwrap()
        };
        let c64_rows = {
            let source = c64();
            transform_cells!($runtime, source, $key, Complex64)
        };
        let lazy_f64_rows = {
            let source = source!($runtime, $v, $w, $key).adjoint().unwrap();
            transform_cells!($runtime, source, $key, f64)
        };
        let lazy_c64_rows = {
            let source = c64().adjoint().unwrap();
            transform_cells!($runtime, source, $key, Complex64)
        };
        [f64_rows, c64_rows, lazy_f64_rows, lazy_c64_rows]
    }};
}

/// The compact rows of both lanes on `$leg`: `[f64, c64]`.
macro_rules! bond_lanes {
    ($runtime:expr, $leg:expr, $sectors:expr, $name:expr, $key:expr) => {{
        let source = bond!($runtime, $leg, $sectors, $name, f64, |x: f64| x);
        let f64_rows = bond_cells!($runtime, source, $key, f64);
        let source = bond!($runtime, $leg, $sectors, $name, Complex64, |x: f64| {
            Complex64::new(x, 2.0 * x)
        });
        let c64_rows = bond_cells!($runtime, source, $key, Complex64);
        [f64_rows, c64_rows]
    }};
}

#[test]
fn multiplicity_free_real_symbol_transform_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    let (v, w) = u1_legs();
    both_lanes!(&runtime, v, w, debug_key::<U1Irrep>);
    let (v, w) = fz2_legs();
    both_lanes!(&runtime, v, w, debug_key::<Z2Irrep>);
    let v = su2_leg();
    both_lanes!(&runtime, v, v, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
}

#[test]
fn multiplicity_free_real_symbol_compact_transform_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    // Non-self-dual U(1): a swap moves each spectrum to the dual sector.
    let (v, _) = u1_legs();
    bond_lanes!(
        &runtime,
        v,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2)
        ],
        |s: &U1Irrep| format!("{s:?}"),
        debug_key::<U1Irrep>
    );
    // Fermion parity on a dual leg: the odd sector carries the swap sign.
    let (v, _) = fz2_legs();
    bond_lanes!(
        &runtime,
        v.try_dual().unwrap(),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
        |s: &Z2Irrep| format!("{s:?}"),
        debug_key::<Z2Irrep>
    );
    let v = su2_leg();
    bond_lanes!(
        &runtime,
        v,
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
        |s: &SU2Irrep| su2_spin(s).to_string(),
        |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin)
    );
}

#[test]
fn multiplicity_free_complex_symbol_transform_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    let leg = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 2), (TableSector::new(2), 3)],
    )
    .unwrap();
    let key = debug_key::<TableSector>;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            Complex64::new(
                value(&key(trees), index),
                value(&format!("im{}", key(trees)), index),
            )
        })
        .unwrap();
    let mut rows: Rows = Vec::new();
    cell!(
        rows,
        &runtime,
        source,
        key,
        Complex64,
        "braid",
        |t| t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]),
        |d, a, b| source.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, a, b)
    );
    cell!(
        rows,
        &runtime,
        source,
        key,
        Complex64,
        "transpose",
        |t| t.transpose(&[1, 3], &[0, 2]),
        |d, a, b| source.transpose_into(&[1, 3], &[0, 2], d, a, b)
    );
    cell!(
        rows,
        &runtime,
        source,
        key,
        Complex64,
        "repartition",
        |t| t.repartition(1),
        |d, a, b| source.repartition_into(d, a, b)
    );
    assert_eq!(rows.len(), 3 * BETAS.len());

    // A compact Fibonacci bond: everything but the anyonic permute.
    let bond = bond!(
        &runtime,
        leg,
        [(TableSector::new(1), 2), (TableSector::new(2), 3)],
        |s: &TableSector| format!("{s:?}"),
        Complex64,
        |x: f64| Complex64::new(x, 0.5 - x)
    );
    let mut rows: Rows = Vec::new();
    cell!(
        rows,
        &runtime,
        bond,
        key,
        Complex64,
        "bond swap transpose",
        |t| t.transpose(&[1], &[0]),
        |d, a, b| bond.transpose_into(&[1], &[0], d, a, b)
    );
    cell!(
        rows,
        &runtime,
        bond,
        key,
        Complex64,
        "bond braid",
        |t| t.braid(&[1], &[0], &[0, 1]),
        |d, a, b| bond.braid_into(&[1], &[0], &[0, 1], d, a, b)
    );
    cell!(
        rows,
        &runtime,
        bond,
        key,
        Complex64,
        "bond repartition (2, 0)",
        |t| t.repartition(2),
        |d, a, b| bond.repartition_into(d, a, b)
    );
    assert_eq!(rows.len(), 3 * BETAS.len());

    // Anyonic: `permute` is refused alike, and the destination is untouched.
    let eager = source.permute(&[2, 0], &[1, 3]).unwrap_err();
    let mut destination = source.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]).unwrap();
    let before = dense(&destination);
    let into = source
        .permute_into(
            &[1, 0],
            &[3, 2],
            &mut destination,
            Complex64::alpha(),
            Complex64::lift(0.5),
        )
        .unwrap_err();
    assert_eq!(
        std::mem::discriminant(&into),
        std::mem::discriminant(&eager),
        "{into:?} vs {eager:?}"
    );
    assert_eq!(dense(&destination), before);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_transform_into_is_alpha_eager_plus_beta_destination_and_matches_multiplicity_free(
) {
    let runtime = host_runtime();
    let mf = su2_leg();
    let checked = sun2_leg();
    let mf_rows = both_lanes!(&runtime, mf, mf, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    let checked_rows = both_lanes!(&runtime, checked, checked, |t: &BlockFusionTrees<
        Vec<i64>,
    >| twice(t, sun_spin));
    let mf_bond = bond_lanes!(
        &runtime,
        mf,
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
        |s: &SU2Irrep| su2_spin(s).to_string(),
        |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin)
    );
    let checked_bond = bond_lanes!(
        &runtime,
        checked,
        SU2_SECTORS.map(|(s, k)| (vec![s as i64], k)),
        |s: &Vec<i64>| sun_spin(s).to_string(),
        |t: &BlockFusionTrees<Vec<i64>>| twice(t, sun_spin)
    );
    for (mf, checked) in mf_rows
        .into_iter()
        .chain(mf_bond)
        .zip(checked_rows.into_iter().chain(checked_bond))
    {
        assert_eq!(mf.len(), checked.len());
        for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
            assert_eq!(
                mf.keys().collect::<Vec<_>>(),
                checked.keys().collect::<Vec<_>>(),
                "{name}"
            );
            for (key, mf) in mf {
                for (&c, &m) in checked[key].iter().zip(mf) {
                    assert!(
                        close(c, m),
                        "{name} {key}: checked {c} vs multiplicity-free {m}"
                    );
                }
            }
        }
    }

    // SU(3): `8 ⊗ 8 → 8` carries outer multiplicity two.
    let v = GradedSpace::try_new(
        Arc::new(symtn::sector::SUNFusionRule::new(3).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    both_lanes!(&runtime, v, v, debug_key::<Vec<i64>>);
    // A non-self-dual SU(3) bond on a dual leg.
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::SUNFusionRule::new(3).unwrap()),
        [(vec![1i64, 0], 2), (vec![0, 1], 1)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    bond_lanes!(
        &runtime,
        leg,
        // The dual leg holds the conjugate labels.
        [(vec![0i64, 1], 2), (vec![1, 0], 1)],
        |s: &Vec<i64>| format!("{s:?}"),
        debug_key::<Vec<i64>>
    );
}

/// A warm checked `_into` allocates no `n_C` result: it costs the eager
/// op's staging minus the payload, and makes no allocation of the payload's
/// size.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_warm_transform_into_allocates_no_result_payload() {
    let _serial = counting_alloc::serial();
    let runtime = host_runtime();
    let v = sun2_leg();
    let source = source!(&runtime, v, v, |t: &BlockFusionTrees<Vec<i64>>| twice(
        t, sun_spin
    ));
    let mut destination = source.permute(&[2, 0], &[1, 3]).unwrap();
    let payload_bytes = std::mem::size_of_val(destination.dense_data().unwrap());
    for _ in 0..2 {
        drop(source.permute(&[2, 0], &[1, 3]).unwrap());
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.5)
            .unwrap();
    }
    let (_, eager) = counting_alloc::measure(|| drop(source.permute(&[2, 0], &[1, 3]).unwrap()));
    let (_, into) = counting_alloc::measure_matching(payload_bytes..=usize::MAX, || {
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.5)
            .unwrap()
    });
    assert_eq!(into.matched_calls, 0, "{into:?}");
    assert!(
        eager.bytes >= into.bytes + payload_bytes as u64,
        "eager {eager:?} vs into {into:?}, payload {payload_bytes}"
    );
}

/// The warm `permute_into` of the lazy adjoint of `$source` makes no
/// allocation as large as the parent payload: the source is read through its
/// parent's storage, never materialized.
macro_rules! lazy_into_allocations {
    ($source:expr) => {{
        let parent = $source;
        let lazy = parent.adjoint().unwrap();
        let mut destination = lazy
            .permute(&[2, 0], &[1, 3])
            .unwrap()
            .materialize()
            .unwrap();
        let payload_bytes = std::mem::size_of_val(parent.dense_data().unwrap());
        for _ in 0..2 {
            lazy.permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.5)
                .unwrap();
        }
        let (_, into) = counting_alloc::measure_matching(payload_bytes..=usize::MAX, || {
            lazy.permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.5)
                .unwrap()
        });
        assert_eq!(into.matched_calls, 0, "{into:?}, payload {payload_bytes}");
        assert!(lazy.dense_data().is_err(), "the lazy source materialized");
    }};
}

#[test]
fn lazy_adjoint_transform_into_allocates_no_parent_sized_buffer() {
    let _serial = counting_alloc::serial();
    let runtime = host_runtime();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (U1Irrep::new(-1), 6),
            (U1Irrep::new(0), 8),
            (U1Irrep::new(1), 6),
        ],
    )
    .unwrap();
    lazy_into_allocations!(source!(&runtime, v, v, debug_key::<U1Irrep>));
    #[cfg(feature = "cgc-gen")]
    {
        let v = sun2_leg();
        lazy_into_allocations!(source!(&runtime, v, v, |t: &BlockFusionTrees<Vec<i64>>| {
            twice(t, sun_spin)
        }));
    }
}

/// `(name, pairs)` of every trace of `[v, w'] ← [v, w']`: the two partial
/// traces and the full trace to a scalar.
const TRACE_PAIRS: [(&str, &[(usize, usize)]); 3] = [
    ("trace (0, 2)", &[(0, 2)]),
    ("trace (1, 3)", &[(1, 3)]),
    ("full trace", &[(0, 2), (1, 3)]),
];

/// Every `trace_pairs_into` of a rank-(2, 2) `$source`.
macro_rules! trace_cells {
    ($runtime:expr, $source:expr, $key:expr, $d:ty) => {{
        let mut rows: Rows = Vec::new();
        for (name, pairs) in TRACE_PAIRS {
            cell!(
                traced;
                rows,
                $runtime,
                $source,
                $key,
                $d,
                name,
                |t| t.trace_pairs(pairs),
                |d, a, b| $source.trace_pairs_into(pairs, d, a, b)
            );
        }
        rows
    }};
}

/// The trace rows of both lanes: `[owned f64, owned c64, lazy f64, lazy
/// c64, compact f64, compact c64]`; the compact source is a diagonal on
/// `$v` traced over its only pair.
macro_rules! trace_lanes {
    ($runtime:expr, $v:expr, $w:expr, $sectors:expr, $name:expr, $key:expr) => {{
        let c64 = || {
            source!($runtime, $v, $w, $key)
                .convert::<Complex64>()
                .scale(Complex64::new(1.0, 2.0))
                .unwrap()
        };
        let owned_f64 = {
            let source = source!($runtime, $v, $w, $key);
            trace_cells!($runtime, source, $key, f64)
        };
        let owned_c64 = {
            let source = c64();
            trace_cells!($runtime, source, $key, Complex64)
        };
        let lazy_f64 = {
            let source = source!($runtime, $v, $w, $key).adjoint().unwrap();
            trace_cells!($runtime, source, $key, f64)
        };
        let lazy_c64 = {
            let source = c64().adjoint().unwrap();
            trace_cells!($runtime, source, $key, Complex64)
        };
        let compact_f64 = {
            let mut rows: Rows = Vec::new();
            let bond = bond!($runtime, $v, $sectors, $name, f64, |x: f64| x);
            cell!(
                traced;
                rows,
                $runtime,
                bond,
                $key,
                f64,
                "compact full trace",
                |t| t.trace_pairs(&[(0, 1)]),
                |d, a, b| bond.trace_pairs_into(&[(0, 1)], d, a, b)
            );
            rows
        };
        let compact_c64 = {
            let mut rows: Rows = Vec::new();
            let bond = bond!($runtime, $v, $sectors, $name, Complex64, |x: f64| {
                Complex64::new(x, 2.0 * x)
            });
            cell!(
                traced;
                rows,
                $runtime,
                bond,
                $key,
                Complex64,
                "compact full trace",
                |t| t.trace_pairs(&[(0, 1)]),
                |d, a, b| bond.trace_pairs_into(&[(0, 1)], d, a, b)
            );
            rows
        };
        [owned_f64, owned_c64, lazy_f64, lazy_c64, compact_f64, compact_c64]
    }};
}

#[test]
fn multiplicity_free_real_symbol_trace_pairs_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    let (v, w) = u1_legs();
    trace_lanes!(
        &runtime,
        v,
        w,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2)
        ],
        |s: &U1Irrep| format!("{s:?}"),
        debug_key::<U1Irrep>
    );
    // Fermion parity with a dual leg: the supertrace twist.
    let (v, w) = fz2_legs();
    trace_lanes!(
        &runtime,
        v,
        w,
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
        |s: &Z2Irrep| format!("{s:?}"),
        debug_key::<Z2Irrep>
    );
    let v = su2_leg();
    trace_lanes!(
        &runtime,
        v,
        v,
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
        |s: &SU2Irrep| su2_spin(s).to_string(),
        |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin)
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_trace_pairs_into_is_alpha_eager_plus_beta_destination_and_matches_multiplicity_free(
) {
    let runtime = host_runtime();
    let mf = su2_leg();
    let checked = sun2_leg();
    let mf_rows = trace_lanes!(
        &runtime,
        mf,
        mf,
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
        |s: &SU2Irrep| su2_spin(s).to_string(),
        |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin)
    );
    let checked_rows = trace_lanes!(
        &runtime,
        checked,
        checked,
        SU2_SECTORS.map(|(s, k)| (vec![s as i64], k)),
        |s: &Vec<i64>| sun_spin(s).to_string(),
        |t: &BlockFusionTrees<Vec<i64>>| twice(t, sun_spin)
    );
    for (mf, checked) in mf_rows.into_iter().zip(checked_rows) {
        assert_eq!(mf.len(), checked.len());
        for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
            assert_eq!(
                mf.keys().collect::<Vec<_>>(),
                checked.keys().collect::<Vec<_>>(),
                "{name}"
            );
            for (key, mf) in mf {
                for (&c, &m) in checked[key].iter().zip(mf) {
                    assert!(
                        close(c, m),
                        "{name} {key}: checked {c} vs multiplicity-free {m}"
                    );
                }
            }
        }
    }

    // SU(3): `8 ⊗ 8 → 8` carries outer multiplicity two.
    let v = GradedSpace::try_new(
        Arc::new(symtn::sector::SUNFusionRule::new(3).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    trace_lanes!(
        &runtime,
        v,
        v,
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
        |s: &Vec<i64>| format!("{s:?}"),
        debug_key::<Vec<i64>>
    );
}

/// A warm `trace_pairs_into` writes the caller's buffer: no allocation as
/// large as the result (checked, against eager), as the lazy source's parent
/// payload (read through it), or as a compact source's dense image (its
/// spectrum is read from the compiled terms).
#[test]
fn trace_pairs_into_allocates_no_result_parent_or_dense_compact_buffer() {
    let _serial = counting_alloc::serial();
    let runtime = host_runtime();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (U1Irrep::new(-1), 6),
            (U1Irrep::new(0), 8),
            (U1Irrep::new(1), 6),
        ],
    )
    .unwrap();
    macro_rules! lazy_trace {
        ($parent:expr) => {{
            let parent = $parent;
            let lazy = parent.adjoint().unwrap();
            let mut destination = lazy.trace_pairs(&[(0, 2)]).unwrap();
            let payload_bytes = std::mem::size_of_val(parent.dense_data().unwrap());
            for _ in 0..2 {
                lazy.trace_pairs_into(&[(0, 2)], &mut destination, 1.0, 0.5)
                    .unwrap();
            }
            let (_, into) = counting_alloc::measure_matching(payload_bytes..=usize::MAX, || {
                lazy.trace_pairs_into(&[(0, 2)], &mut destination, 1.0, 0.5)
                    .unwrap()
            });
            assert_eq!(into.matched_calls, 0, "{into:?}, payload {payload_bytes}");
            assert!(lazy.dense_data().is_err(), "the lazy source materialized");
        }};
    }
    lazy_trace!(source!(&runtime, v, v, debug_key::<U1Irrep>));

    // One sector of degeneracy k = 128: the dense image is P = k² entries.
    let k = 128;
    let leg = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), k)]).unwrap();
    let bond = bond!(
        &runtime,
        leg,
        [(U1Irrep::new(0), k)],
        |s: &U1Irrep| format!("{s:?}"),
        f64,
        |x: f64| x
    );
    let mut destination = bond.trace_pairs(&[(0, 1)]).unwrap();
    let p_bytes = k * k * std::mem::size_of::<f64>();
    for _ in 0..2 {
        bond.trace_pairs_into(&[(0, 1)], &mut destination, 1.0, 0.5)
            .unwrap();
    }
    let (_, into) = counting_alloc::measure_matching(p_bytes..=usize::MAX, || {
        bond.trace_pairs_into(&[(0, 1)], &mut destination, 1.0, 0.5)
            .unwrap()
    });
    assert_eq!(into.matched_calls, 0, "{into:?}, P {p_bytes}");
    assert!(into.bytes < p_bytes as u64, "{into:?}, P {p_bytes}");
    assert!(
        bond.dense_data().is_err(),
        "the compact source materialized"
    );
    // The control: its dense twin's trace reads the P-sized payload, which
    // the probe would see if the compact read declined.
    let dense_twin = bond.materialize().unwrap();
    let (_, control) = counting_alloc::measure_matching(p_bytes..=usize::MAX, || {
        drop(bond.materialize().unwrap());
        dense_twin
            .trace_pairs_into(&[(0, 1)], &mut destination, 1.0, 0.5)
            .unwrap()
    });
    assert!(control.matched_calls >= 1, "{control:?}");

    #[cfg(feature = "cgc-gen")]
    {
        let v = sun2_leg();
        let key = |t: &BlockFusionTrees<Vec<i64>>| twice(t, sun_spin);
        lazy_trace!(source!(&runtime, v, v, key));
        // A large untraced leg, so the result payload dominates the staging.
        let w = GradedSpace::try_new(
            Arc::new(symtn::sector::SUNFusionRule::new(2).unwrap()),
            [(vec![0i64], 24), (vec![1], 24), (vec![2], 24)],
        )
        .unwrap();
        let source = source!(&runtime, v, w, key);
        let mut destination = source.trace_pairs(&[(0, 2)]).unwrap();
        let payload_bytes = std::mem::size_of_val(destination.dense_data().unwrap());
        for _ in 0..2 {
            drop(source.trace_pairs(&[(0, 2)]).unwrap());
            source
                .trace_pairs_into(&[(0, 2)], &mut destination, 1.0, 0.5)
                .unwrap();
        }
        let (_, eager) = counting_alloc::measure(|| drop(source.trace_pairs(&[(0, 2)]).unwrap()));
        let (_, into) = counting_alloc::measure_matching(payload_bytes..=usize::MAX, || {
            source
                .trace_pairs_into(&[(0, 2)], &mut destination, 1.0, 0.5)
                .unwrap()
        });
        assert_eq!(into.matched_calls, 0, "{into:?}");
        assert!(
            eager.bytes >= into.bytes + payload_bytes as u64,
            "eager {eager:?} vs into {into:?}, payload {payload_bytes}"
        );
    }
}

/// `(name, codomain, domain)` of the rank-3 result of contracting
/// `[v, w'] ← [v, w']` with `[v, w'] ← [v]` over the domain: the core
/// (default) order and two reordered ones (copyC or `DynamicTree`, as the
/// planner's memcost picks).
const CONTRACT_ORDERS: [(&str, &[usize], &[usize]); 3] = [
    ("core", &[0, 1], &[2]),
    ("reordered [2, 0] ← [1]", &[2, 0], &[1]),
    ("reordered [1] ← [0, 2]", &[1], &[0, 2]),
];

/// Every `contract_into` of `$lhs` (`[v, w'] ← [v, w']`) with the
/// `[v, w'] ← [v]` operand: owned `$rhs`, and `$rhs_parent`'s lazy adjoint
/// and adjoint view; `$lazy_lhs` (`[v, w'] ← [v, w']`) against the owned one.
macro_rules! contract_cells {
    ($runtime:expr, $lhs:expr, $lazy_lhs:expr, $rhs:expr, $rhs_parent:expr, $key:expr,
     $d:ty) => {{
        let mut rows: Rows = Vec::new();
        let lazy_rhs = $rhs_parent.adjoint().unwrap();
        for (order, codomain, domain) in CONTRACT_ORDERS {
            let spec = ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain,
                domain,
            };
            contract_cell!(
                rows,
                $runtime,
                $lhs,
                $rhs,
                &$rhs,
                spec,
                $key,
                $d,
                format!("contract O·O {order}")
            );
            contract_cell!(
                rows,
                $runtime,
                $lazy_lhs,
                $rhs,
                &$rhs,
                spec,
                $key,
                $d,
                format!("contract L·O {order}")
            );
            contract_cell!(
                rows,
                $runtime,
                $lhs,
                lazy_rhs,
                &lazy_rhs,
                spec,
                $key,
                $d,
                format!("contract O·L {order}")
            );
            contract_cell!(
                rows,
                $runtime,
                $lhs,
                lazy_rhs,
                $rhs_parent.adjoint_view(),
                spec,
                $key,
                $d,
                format!("contract O·V {order}")
            );
        }
        rows
    }};
}

/// The contraction rows of both lanes on `[v, w'] ← [v, w']`: `[f64, c64]`.
macro_rules! contract_lanes {
    ($runtime:expr, $v:expr, $w:expr, $key:expr) => {{
        let key = $key;
        let w = $w.try_dual().unwrap();
        let rhs: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($runtime, [&$v, &w], [&$v], |trees, index| {
                value(&format!("rhs{}", key(trees)), index)
            })
            .unwrap();
        let rhs_parent: TensorMap<_, f64> =
            TensorMap::from_subblock_fn($runtime, [&$v], [&$v, &w], |trees, index| {
                value(&format!("parent{}", key(trees)), index)
            })
            .unwrap();
        let lhs = source!($runtime, $v, $w, key);
        let lazy_lhs = source!($runtime, $v, $w, key)
            .scale(-0.75)
            .unwrap()
            .adjoint()
            .unwrap();
        let f64_rows = contract_cells!($runtime, lhs, lazy_lhs, rhs, rhs_parent, key, f64);
        // `1 + 2i` makes a lost or doubled conjugation visible.
        let c64 = |t: &TensorMap<_, f64>| {
            t.convert::<Complex64>()
                .scale(Complex64::new(1.0, 2.0))
                .unwrap()
        };
        let lazy_lhs = c64(&source!($runtime, $v, $w, key).scale(-0.75).unwrap())
            .adjoint()
            .unwrap();
        let c64_rows = contract_cells!(
            $runtime,
            c64(&lhs),
            lazy_lhs,
            c64(&rhs),
            c64(&rhs_parent),
            key,
            Complex64
        );
        [f64_rows, c64_rows]
    }};
}

/// The compact contraction rows on a bond `$b` (spectrum on `$sectors`)
/// against `[v, w'] ← [b]`-shaped operands, lifted by `$lift`: every arm
/// pattern of `Self::contract` (`t · D`, `D · t` with a direct and a lazy
/// `t`, `D · D`), each in its own order and reordered, plus the two-axis
/// `D · D` the arm declines.
macro_rules! compact_contract_cells {
    ($runtime:expr, $v:expr, $w:expr, $b:expr, $sectors:expr, $name:expr, $key:expr, $d:ty,
     $lift:expr) => {{
        let (key, lift) = ($key, $lift);
        let w = $w.try_dual().unwrap();
        let fill = |prefix: &'static str| {
            move |trees: &BlockFusionTrees<_>, index: &[usize]| -> $d {
                lift(value(&format!("{prefix}{}", key(trees)), index))
            }
        };
        let d1 = bond!($runtime, $b, $sectors, $name, $d, lift);
        let d2 = bond!($runtime, $b, $sectors, $name, $d, |x: f64| lift(0.5 - x));
        // `t · D`: `[v, w'] ← [b]`, owned and as the adjoint of `[b] ← [v, w']`.
        let t: TensorMap<_, $d> =
            TensorMap::from_subblock_fn($runtime, [&$v, &w], [&$b], fill("t")).unwrap();
        let lazy_t = TensorMap::<_, $d>::from_subblock_fn($runtime, [&$b], [&$v, &w], fill("lt"))
            .unwrap()
            .adjoint()
            .unwrap();
        // `D · t`: `[b, v] ← [w']`, owned and as the adjoint of `[w'] ← [b, v]`.
        let u: TensorMap<_, $d> =
            TensorMap::from_subblock_fn($runtime, [&$b, &$v], [&w], fill("u")).unwrap();
        let lazy_u = TensorMap::<_, $d>::from_subblock_fn($runtime, [&w], [&$b, &$v], fill("lu"))
            .unwrap()
            .adjoint()
            .unwrap();
        let mut rows: Rows = Vec::new();
        for (pattern, codomain, domain) in [
            ("own order", &[0usize, 1][..], &[2usize][..]),
            ("reordered", &[2, 0][..], &[1][..]),
        ] {
            let spec = ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain,
                domain,
            };
            contract_cell!(
                rows,
                $runtime,
                t,
                d1,
                &d1,
                spec,
                key,
                $d,
                format!("compact t·D {pattern}")
            );
            contract_cell!(
                rows,
                $runtime,
                lazy_t,
                d1,
                &d1,
                spec,
                key,
                $d,
                format!("compact L(t)·D {pattern}")
            );
            let spec = ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain,
                domain,
            };
            contract_cell!(
                rows,
                $runtime,
                d1,
                u,
                &u,
                spec,
                key,
                $d,
                format!("compact D·t {pattern}")
            );
            contract_cell!(
                rows,
                $runtime,
                d1,
                lazy_u,
                &lazy_u,
                spec,
                key,
                $d,
                format!("compact D·L(t) {pattern}")
            );
        }
        for (pattern, codomain, domain) in [
            ("own order", &[0usize][..], &[1usize][..]),
            ("swapped", &[1][..], &[0][..]),
            ("(2, 0)", &[0, 1][..], &[][..]),
            ("(0, 2)", &[][..], &[1, 0][..]),
        ] {
            let spec = ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain,
                domain,
            };
            contract_cell!(
                rows,
                $runtime,
                d1,
                d2,
                &d2,
                spec,
                key,
                $d,
                format!("compact D·D {pattern}")
            );
        }
        // Both legs of `D` against `[v, b] ← [b]`: two contracted axes, so
        // the arm declines and `D` is densified (TensorKit copies it too).
        let x: TensorMap<_, $d> =
            TensorMap::from_subblock_fn($runtime, [&$v, &$b], [&$b], fill("x")).unwrap();
        let spec = ContractSpec {
            lhs: &[2, 1],
            rhs: &[0, 1],
            codomain: &[0],
            domain: &[],
        };
        contract_cell!(
            rows,
            $runtime,
            x,
            d1,
            &d1,
            spec,
            key,
            $d,
            "compact t·D two axes (declines)".to_string()
        );
        rows
    }};
}

/// The compact contraction rows of both lanes: `[f64, c64]`.
macro_rules! compact_contract_lanes {
    ($runtime:expr, $v:expr, $w:expr, $b:expr, $sectors:expr, $name:expr, $key:expr) => {{
        [
            compact_contract_cells!($runtime, $v, $w, $b, $sectors, $name, $key, f64, |x: f64| x),
            compact_contract_cells!(
                $runtime,
                $v,
                $w,
                $b,
                $sectors,
                $name,
                $key,
                Complex64,
                |x: f64| Complex64::new(x, 2.0 * x)
            ),
        ]
    }};
}

#[test]
fn multiplicity_free_real_symbol_contract_into_is_alpha_eager_plus_beta_destination() {
    let runtime = host_runtime();
    let (v, w) = u1_legs();
    contract_lanes!(&runtime, v, w, debug_key::<U1Irrep>);
    compact_contract_lanes!(
        &runtime,
        v,
        w,
        v,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2)
        ],
        |s: &U1Irrep| format!("{s:?}"),
        debug_key::<U1Irrep>
    );
    let (v, w) = fz2_legs();
    contract_lanes!(&runtime, v, w, debug_key::<Z2Irrep>);
    compact_contract_lanes!(
        &runtime,
        v,
        w,
        w,
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
        |s: &Z2Irrep| format!("{s:?}"),
        debug_key::<Z2Irrep>
    );
    // A bond on a dual leg: the fermionic twist on the contracted leg makes
    // every arm decline to the dense route (TensorKit's `copyA`/`copyB`).
    let dual = w.try_dual().unwrap();
    compact_contract_lanes!(
        &runtime,
        v,
        w,
        dual,
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
        |s: &Z2Irrep| format!("{s:?}"),
        debug_key::<Z2Irrep>
    );
    let v = su2_leg();
    contract_lanes!(&runtime, v, v, |t: &BlockFusionTrees<SU2Irrep>| twice(
        t, su2_spin
    ));
    compact_contract_lanes!(
        &runtime,
        v,
        v,
        v,
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
        |s: &SU2Irrep| su2_spin(s).to_string(),
        |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin)
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_contract_into_is_alpha_eager_plus_beta_destination_and_matches_multiplicity_free(
) {
    let runtime = host_runtime();
    let mf = su2_leg();
    let checked = sun2_leg();
    let mf_key = |t: &BlockFusionTrees<SU2Irrep>| twice(t, su2_spin);
    let checked_key = |t: &BlockFusionTrees<Vec<i64>>| twice(t, sun_spin);
    let mf_rows = contract_lanes!(&runtime, mf, mf, mf_key);
    let checked_rows = contract_lanes!(&runtime, checked, checked, checked_key);
    let mf_compact = compact_contract_lanes!(
        &runtime,
        mf,
        mf,
        mf,
        SU2_SECTORS.map(|(s, k)| (SU2Irrep::from_twice_spin(s), k)),
        |s: &SU2Irrep| su2_spin(s).to_string(),
        mf_key
    );
    let checked_compact = compact_contract_lanes!(
        &runtime,
        checked,
        checked,
        checked,
        SU2_SECTORS.map(|(s, k)| (vec![s as i64], k)),
        |s: &Vec<i64>| sun_spin(s).to_string(),
        checked_key
    );
    for (mf, checked) in mf_rows
        .into_iter()
        .chain(mf_compact)
        .zip(checked_rows.into_iter().chain(checked_compact))
    {
        assert_eq!(mf.len(), checked.len());
        for ((name, mf), (_, checked)) in mf.iter().zip(&checked) {
            assert_eq!(
                mf.keys().collect::<Vec<_>>(),
                checked.keys().collect::<Vec<_>>(),
                "{name}"
            );
            for (key, mf) in mf {
                for (&c, &m) in checked[key].iter().zip(mf) {
                    assert!(
                        close(c, m),
                        "{name} {key}: checked {c} vs multiplicity-free {m}"
                    );
                }
            }
        }
    }

    // SU(3): `8 ⊗ 8 → 8` carries outer multiplicity two.
    let v = GradedSpace::try_new(
        Arc::new(symtn::sector::SUNFusionRule::new(3).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    contract_lanes!(&runtime, v, v, debug_key::<Vec<i64>>);
    compact_contract_lanes!(
        &runtime,
        v,
        v,
        v,
        [(vec![0i64, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
        |s: &Vec<i64>| format!("{s:?}"),
        debug_key::<Vec<i64>>
    );
}

/// Warm compact `contract_into` on every arm pattern writes the eager arm's
/// algebra into the destination (TensorKit `mul!` scaling a `Diagonal`
/// block): no allocation of `P = k²` entries, and fewer bytes in total; a
/// direct `t` in its own order allocates no `n_C` buffer either. The
/// two-axis control declines and densifies `D`. `$v` (`dim 4`) and the bond
/// `$b` (`k = 128`) hold one sector, so `P` dominates `n_C`.
macro_rules! compact_contract_allocations {
    ($runtime:expr, $v:expr, $b:expr, $sector:expr) => {{
        let k = 128;
        let spectrum = |offset: f64| {
            vec![SectorSpectrum {
                sector: $sector,
                values: (0..k)
                    .map(|i| offset + i as f64 / 7.0)
                    .collect::<Vec<f64>>(),
            }]
        };
        let d1 = TensorMap::<_, f64>::diagonal($runtime, &$b, spectrum(1.0)).unwrap();
        let d2 = TensorMap::<_, f64>::diagonal($runtime, &$b, spectrum(-0.5)).unwrap();
        let fill = |_: &BlockFusionTrees<_>, index: &[usize]| value("t", index);
        let t = TensorMap::<_, f64>::from_subblock_fn($runtime, [&$v], [&$b], fill).unwrap();
        let lazy_t = TensorMap::<_, f64>::from_subblock_fn($runtime, [&$b], [&$v], fill)
            .unwrap()
            .adjoint()
            .unwrap();
        let u = TensorMap::<_, f64>::from_subblock_fn($runtime, [&$b], [&$v], fill).unwrap();
        let lazy_u = TensorMap::<_, f64>::from_subblock_fn($runtime, [&$v], [&$b], fill)
            .unwrap()
            .adjoint()
            .unwrap();
        let x = TensorMap::<_, f64>::from_subblock_fn($runtime, [&$v, &$b], [&$b], fill).unwrap();
        let p_bytes = k * k * std::mem::size_of::<f64>();
        let n_c_bytes = 4 * k * std::mem::size_of::<f64>();
        let own: (&[usize], &[usize]) = (&[0], &[1]);
        let swapped: (&[usize], &[usize]) = (&[1], &[0]);
        // (name, lhs, rhs, contracted, order, direct own-order, declines)
        let cases = [
            (
                "t·D own order",
                &t,
                &d1,
                (&[1usize][..], &[0usize][..]),
                own,
                true,
                false,
            ),
            (
                "t·D reordered",
                &t,
                &d1,
                (&[1][..], &[0][..]),
                swapped,
                false,
                false,
            ),
            (
                "L(t)·D own order",
                &lazy_t,
                &d1,
                (&[1][..], &[0][..]),
                own,
                false,
                false,
            ),
            (
                "L(t)·D reordered",
                &lazy_t,
                &d1,
                (&[1][..], &[0][..]),
                swapped,
                false,
                false,
            ),
            (
                "D·t own order",
                &d1,
                &u,
                (&[1][..], &[0][..]),
                own,
                true,
                false,
            ),
            (
                "D·t reordered",
                &d1,
                &u,
                (&[1][..], &[0][..]),
                swapped,
                false,
                false,
            ),
            (
                "D·L(t) own order",
                &d1,
                &lazy_u,
                (&[1][..], &[0][..]),
                own,
                false,
                false,
            ),
            (
                "D·L(t) reordered",
                &d1,
                &lazy_u,
                (&[1][..], &[0][..]),
                swapped,
                false,
                false,
            ),
            (
                "D·D own order",
                &d1,
                &d2,
                (&[1][..], &[0][..]),
                own,
                false,
                false,
            ),
            (
                "D·D reordered",
                &d1,
                &d2,
                (&[1][..], &[0][..]),
                swapped,
                false,
                false,
            ),
            (
                "t·D two axes",
                &x,
                &d1,
                (&[2, 1][..], &[0, 1][..]),
                (&[0][..], &[][..]),
                false,
                true,
            ),
        ];
        for (name, lhs, rhs, (lhs_axes, rhs_axes), (codomain, domain), direct, declines) in cases {
            let spec = ContractSpec {
                lhs: lhs_axes,
                rhs: rhs_axes,
                codomain,
                domain,
            };
            let mut destination = lhs.contract(rhs, &spec).unwrap().materialize().unwrap();
            for _ in 0..2 {
                lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5)
                    .unwrap();
            }
            let (_, large) = counting_alloc::measure_matching(p_bytes..=usize::MAX, || {
                lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5)
                    .unwrap()
            });
            if declines {
                assert!(large.matched_calls > 0, "{name}: the control densifies D");
                continue;
            }
            assert_eq!(large.matched_calls, 0, "{name}: {large:?}");
            // `bytes` sums every allocation of the call, not only the matched
            // ones: many small buffers must not add up to `P` either.
            assert!((large.bytes as usize) < p_bytes, "{name}: {large:?}");
            if direct {
                let (_, result) = counting_alloc::measure_matching(n_c_bytes..=usize::MAX, || {
                    lhs.contract_into(rhs, &spec, &mut destination, 1.0, 0.5)
                        .unwrap()
                });
                assert_eq!(result.matched_calls, 0, "{name}: {result:?}");
            }
        }
    }};
}

#[test]
fn compact_contract_into_allocates_no_dense_compact_operand_in_both_modes() {
    let _serial = counting_alloc::serial();
    let runtime = host_runtime();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 4)]).unwrap();
    let b = GradedSpace::try_new(rule, [(U1Irrep::new(0), 128)]).unwrap();
    compact_contract_allocations!(&runtime, v, b, U1Irrep::new(0));
    #[cfg(feature = "cgc-gen")]
    {
        let rule = Arc::new(symtn::sector::SUNFusionRule::new(2).unwrap());
        let v = GradedSpace::try_new(Arc::clone(&rule), [(vec![0i64], 4)]).unwrap();
        let b = GradedSpace::try_new(rule, [(vec![0i64], 128)]).unwrap();
        compact_contract_allocations!(&runtime, v, b, vec![0i64]);
    }
}

// ---- symmetric complex-symbol contraction ----
//
// TensorKit `tensorcontract!` has no symbol-type dispatch: a symmetric rule
// with `Complex64` symbols contracts like a real-symbol one. No in-tree
// complex-symbol rule is symmetric (Fibonacci is anyonic), so these two
// test-only rules stand in for a user's.

use symtn::sector::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionRule, FusionStyleKind,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorCodec, SectorId, SectorOrderKey, SectorVec,
};
use symtn::typed::FusionAlgebraError;

/// `U1FusionRule` with every symbol embedded in `Complex64`: the same
/// category, so its contractions equal the real-symbol lane's.
#[derive(Clone, Copy)]
struct ComplexU1;

impl symtn_core::testing::Provider for ComplexU1 {}

impl symtn::sector::TypedSectorAdmission for ComplexU1 {
    type Sector = <Self as symtn::sector::SectorCodec>::Sector;
    type Error = symtn::typed::FusionAlgebraError;
    type Mode = symtn::sector::MultiplicityFreeAdmissionMode;
    fn typed_rule_identity(&self) -> symtn::sector::RuleIdentity {
        symtn::sector::FusionRule::rule_identity(self)
    }
    fn try_encode_label(
        &self,
        sector: &Self::Sector,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::SectorCodec::encode_sector(self, sector)
    }
    fn try_decode_label(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<Self::Sector, Self::Error> {
        symtn::sector::SectorCodec::decode_sector(self, sector)
    }
    fn try_dual_id(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::CheckedFusionAlgebra::try_dual_sector(self, sector)
    }
}

impl FusionRule for ComplexU1 {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        U1FusionRule.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        U1FusionRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        U1FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        U1FusionRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        U1FusionRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        U1FusionRule.nsymbol(left, right, coupled)
    }
    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        U1FusionRule.sector_order_key(sector)
    }
}

impl MultiplicityFreeFusionRule for ComplexU1 {}

impl MultiplicityFreeFusionSymbols for ComplexU1 {
    type Scalar = Complex64;
    fn has_trivial_associator_gauge(&self) -> bool {
        U1FusionRule.has_trivial_associator_gauge()
    }
    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Complex64 {
        U1FusionRule.f_symbol_scalar(a, b, c, d, e, f).into()
    }
    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Complex64 {
        U1FusionRule.r_symbol_scalar(a, b, c).into()
    }
}

impl MultiplicityFreeRigidSymbols for ComplexU1 {
    fn dim_scalar(&self, s: SectorId) -> Complex64 {
        U1FusionRule.dim_scalar(s).into()
    }
    fn inv_dim_scalar(&self, s: SectorId) -> Complex64 {
        U1FusionRule.inv_dim_scalar(s).into()
    }
    fn sqrt_dim_scalar(&self, s: SectorId) -> Complex64 {
        U1FusionRule.sqrt_dim_scalar(s).into()
    }
    fn inv_sqrt_dim_scalar(&self, s: SectorId) -> Complex64 {
        U1FusionRule.inv_sqrt_dim_scalar(s).into()
    }
    fn twist_scalar(&self, s: SectorId) -> Complex64 {
        U1FusionRule.twist_scalar(s).into()
    }
    fn frobenius_schur_phase_scalar(&self, s: SectorId) -> Complex64 {
        U1FusionRule.frobenius_schur_phase_scalar(s).into()
    }
}

impl CheckedFusionAlgebra for ComplexU1 {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        U1FusionRule.try_dual_sector(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        U1FusionRule.try_fusion_channels(left, right)
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        U1FusionRule.try_nsymbol(left, right, coupled)
    }
}

impl SectorCodec for ComplexU1 {
    type Sector = U1Irrep;
    fn encode_sector(&self, value: &U1Irrep) -> Result<SectorId, FusionAlgebraError> {
        U1FusionRule.encode_sector(value)
    }
    fn decode_sector(&self, id: SectorId) -> Result<U1Irrep, FusionAlgebraError> {
        U1FusionRule.decode_sector(id)
    }
}

/// Bosonic Z3 × Z3 with trivial `F` and the alternating bicharacter
/// `R(a, b) = ω^{a₁b₂ − a₂b₁}`, `ω = e^{2πi/3}`: symmetric
/// (`R(a, b)·R(b, a) = 1`) with `θ = R(a, a) = 1`, so a swap of two
/// codomain legs is a phase a real-symbol lane cannot carry. Sector
/// `(a₁, a₂)` has id `a₁ + 3a₂`.
///
/// Why not certify `has_trivial_associator_gauge`: that selects the direct
/// Unique braid (`unique_braid.rs:unique_direct_braid`, TensorKit's
/// `SymmetricBraiding` branch, `braiding_manipulations.jl:221-229`), which
/// takes `R(b, a)` for `a` crossing `b`. That equals the Artin path's `R(a, b)`
/// only for a real `R`; for this one it is the conjugate (reported with
/// #1870, outside the contraction lane).
#[derive(Clone, Copy)]
struct AlternatingZ3Squared;

fn z3_squared(id: SectorId) -> (usize, usize) {
    (id.id() % 3, id.id() / 3)
}

fn z3_squared_id((a1, a2): (usize, usize)) -> SectorId {
    SectorId::new(a1 % 3 + 3 * (a2 % 3))
}

/// `R(a, b) = ω^{a₁b₂ − a₂b₁}`.
fn alternating_r((a1, a2): (usize, usize), (b1, b2): (usize, usize)) -> Complex64 {
    let exponent = (a1 * b2 + 2 * a2 * b1) % 3;
    Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * exponent as f64 / 3.0)
}

impl symtn_core::testing::Provider for AlternatingZ3Squared {}

impl symtn::sector::TypedSectorAdmission for AlternatingZ3Squared {
    type Sector = <Self as symtn::sector::SectorCodec>::Sector;
    type Error = symtn::typed::FusionAlgebraError;
    type Mode = symtn::sector::MultiplicityFreeAdmissionMode;
    fn typed_rule_identity(&self) -> symtn::sector::RuleIdentity {
        symtn::sector::FusionRule::rule_identity(self)
    }
    fn try_encode_label(
        &self,
        sector: &Self::Sector,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::SectorCodec::encode_sector(self, sector)
    }
    fn try_decode_label(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<Self::Sector, Self::Error> {
        symtn::sector::SectorCodec::decode_sector(self, sector)
    }
    fn try_dual_id(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::CheckedFusionAlgebra::try_dual_sector(self, sector)
    }
}

impl FusionRule for AlternatingZ3Squared {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_or_panic(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let ((a1, a2), (b1, b2)) = (z3_squared(left), z3_squared(right));
        core::iter::once(z3_squared_id((a1 + b1, a2 + b2))).collect()
    }
}

impl MultiplicityFreeFusionRule for AlternatingZ3Squared {}

impl MultiplicityFreeFusionSymbols for AlternatingZ3Squared {
    type Scalar = Complex64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, _: SectorId) -> Complex64 {
        alternating_r(z3_squared(a), z3_squared(b))
    }
}

impl MultiplicityFreeRigidSymbols for AlternatingZ3Squared {
    fn dim_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn inv_dim_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn twist_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
}

impl CheckedFusionAlgebra for AlternatingZ3Squared {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if sector.id() >= 9 {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        let (a1, a2) = z3_squared(sector);
        Ok(z3_squared_id((3 - a1, 3 - a2)))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        for sector in [left, right] {
            if sector.id() >= 9 {
                return Err(FusionAlgebraError::InvalidSector { sector });
            }
        }
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(usize::from(
            self.try_fusion_channels(left, right)?[0] == coupled,
        ))
    }
}

impl SectorCodec for AlternatingZ3Squared {
    type Sector = (u8, u8);
    fn encode_sector(&self, &(a1, a2): &(u8, u8)) -> Result<SectorId, FusionAlgebraError> {
        if a1 < 3 && a2 < 3 {
            Ok(z3_squared_id((usize::from(a1), usize::from(a2))))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: format!("Z3 x Z3 charge ({a1}, {a2})"),
            })
        }
    }
    fn decode_sector(&self, sector: SectorId) -> Result<(u8, u8), FusionAlgebraError> {
        if sector.id() >= 9 {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        let (a1, a2) = z3_squared(sector);
        Ok((a1 as u8, a2 as u8))
    }
}

/// `x` with `-0` folded into `+0`, so two routes compare exactly.
fn fold_zero(entries: BTreeMap<String, Vec<Complex64>>) -> BTreeMap<String, Vec<Complex64>> {
    entries
        .into_iter()
        .map(|(key, values)| {
            let values = values
                .into_iter()
                .map(|z| Complex64::new(z.re + 0.0, z.im + 0.0))
                .collect();
            (key, values)
        })
        .collect()
}

fn complex_value(key: &str, index: &[usize]) -> Complex64 {
    Complex64::new(value(key, index), value(&format!("im{key}"), index))
}

#[test]
fn symmetric_complex_symbol_contraction_equals_the_real_symbol_lane() {
    // What: the complex-symbol lane contracts (eager and `_into`), and on
    // the `Complex64` embedding of U(1) it is the real-symbol lane's value,
    // exactly up to the sign of zero, in every output order and operand
    // kind both lanes take.
    let runtime = host_runtime();
    macro_rules! legs {
        ($rule:expr) => {{
            let rule = Arc::new($rule);
            let v = GradedSpace::try_new(
                Arc::clone(&rule),
                [
                    (U1Irrep::new(-1), 2),
                    (U1Irrep::new(0), 3),
                    (U1Irrep::new(2), 1),
                ],
            )
            .unwrap();
            let w = GradedSpace::try_new(rule, [(U1Irrep::new(1), 2), (U1Irrep::new(-1), 1)])
                .unwrap()
                .try_dual()
                .unwrap();
            (v, w)
        }};
    }
    macro_rules! operands {
        ($rule:expr) => {{
            let (v, w) = legs!($rule);
            let lhs: TensorMap<_, Complex64> =
                TensorMap::from_subblock_fn(&runtime, [&v, &w], [&v, &w], |trees, index| {
                    complex_value(&debug_key(trees), index)
                })
                .unwrap();
            let rhs: TensorMap<_, Complex64> =
                TensorMap::from_subblock_fn(&runtime, [&v, &w], [&v], |trees, index| {
                    complex_value(&format!("rhs{}", debug_key(trees)), index)
                })
                .unwrap();
            (lhs, rhs)
        }};
    }
    let (lhs, rhs) = operands!(ComplexU1);
    let (real_lhs, real_rhs) = operands!(U1FusionRule);
    let key = debug_key::<U1Irrep>;
    let mut rows: Rows = Vec::new();
    for (order, codomain, domain) in CONTRACT_ORDERS {
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain,
            domain,
        };
        assert_eq!(
            fold_zero(entries!(lhs.contract(&rhs, &spec).unwrap(), key)),
            fold_zero(entries!(real_lhs.contract(&real_rhs, &spec).unwrap(), key)),
            "{order}"
        );
        contract_cell!(
            rows,
            &runtime,
            lhs,
            rhs,
            &rhs,
            spec,
            key,
            Complex64,
            format!("complex U(1) contract O·O {order}")
        );
    }

    // The compact arms and their decline, against owned operands.
    let (v, w) = legs!(ComplexU1);
    let lift = |x: f64| Complex64::new(x, 0.5 - x);
    let sectors = [
        (U1Irrep::new(-1), 2),
        (U1Irrep::new(0), 3),
        (U1Irrep::new(2), 1),
    ];
    let d1 = bond!(
        &runtime,
        v,
        sectors,
        |s: &U1Irrep| format!("{s:?}"),
        Complex64,
        lift
    );
    let d2 = bond!(
        &runtime,
        v,
        sectors,
        |s: &U1Irrep| format!("{s:?}"),
        Complex64,
        |x: f64| { lift(0.5 - x) }
    );
    let fill = |prefix: &'static str| {
        move |trees: &BlockFusionTrees<U1Irrep>, index: &[usize]| {
            complex_value(&format!("{prefix}{}", debug_key(trees)), index)
        }
    };
    let t: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &w], [&v], fill("t")).unwrap();
    let u: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &v], [&w], fill("u")).unwrap();
    let x: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v], fill("x")).unwrap();
    for (pattern, codomain, domain) in [
        ("own order", &[0usize, 1][..], &[2usize][..]),
        ("reordered", &[2, 0][..], &[1][..]),
    ] {
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain,
            domain,
        };
        contract_cell!(
            rows,
            &runtime,
            t,
            d1,
            &d1,
            spec,
            key,
            Complex64,
            format!("complex U(1) compact t·D {pattern}")
        );
        let spec = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain,
            domain,
        };
        contract_cell!(
            rows,
            &runtime,
            d1,
            u,
            &u,
            spec,
            key,
            Complex64,
            format!("complex U(1) compact D·t {pattern}")
        );
    }
    for (pattern, codomain, domain) in [
        ("own order", &[0usize][..], &[1usize][..]),
        ("swapped", &[1][..], &[0][..]),
        ("(2, 0)", &[0, 1][..], &[][..]),
    ] {
        let spec = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain,
            domain,
        };
        contract_cell!(
            rows,
            &runtime,
            d1,
            d2,
            &d2,
            spec,
            key,
            Complex64,
            format!("complex U(1) compact D·D {pattern}")
        );
    }
    let spec = ContractSpec {
        lhs: &[2, 1],
        rhs: &[0, 1],
        codomain: &[0],
        domain: &[],
    };
    contract_cell!(
        rows,
        &runtime,
        x,
        d1,
        &d1,
        spec,
        key,
        Complex64,
        "complex U(1) compact t·D two axes (declines)".to_string()
    );
    assert_eq!(rows.len(), (CONTRACT_ORDERS.len() + 8) * BETAS.len());
}

#[test]
fn alternating_z3_squared_contraction_carries_the_braiding_phase() {
    // What: `[v, w] ← [x]` contracted with `[x] ← [y]` over `x`, with the
    // output codomain legs swapped, is TensorKit's `blas_contract!` followed
    // by `artin_braid` on the codomain tree (`braiding_manipulations.jl:56`,
    // `R(a, b, c)` for uncoupled `(a, b)`): by hand, with `c = v + w = x = y`,
    // `C[(w, v); y][i_w, i_v, i_y] = R(v, w) Σ_{i_x} A[(v, w); x][i_v, i_w, i_x]
    // · B[x; y][i_x, i_y]`. The core order is the same sum without `R`. An
    // ignored or conjugated phase fails: `R(v, w)` and `R(w, v)` differ
    // whenever `v₁w₂ ≠ v₂w₁ (mod 3)`.
    let runtime = host_runtime();
    {
        let rule = Arc::new(AlternatingZ3Squared);
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [((0, 0), 2), ((1, 0), 1), ((0, 1), 2), ((1, 2), 1)],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::clone(&rule),
            [((0, 0), 1), ((2, 1), 2), ((1, 1), 1), ((0, 2), 2)],
        )
        .unwrap();
        let all = (0..9u8)
            .map(|id| ((id % 3, id / 3), 1 + usize::from(id % 2)))
            .collect::<Vec<_>>();
        let x = GradedSpace::try_new(rule, all).unwrap();
        let a_value = |v: (u8, u8), w: (u8, u8), x: (u8, u8), index: &[usize]| {
            complex_value(&format!("A{v:?}{w:?}{x:?}"), index)
        };
        let b_value = |x: (u8, u8), index: &[usize]| complex_value(&format!("B{x:?}"), index);
        let a: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&v, &w], [&x], |trees, index| {
                let (cod, dom) = (trees.codomain_uncoupled(), trees.domain_uncoupled());
                a_value(cod[0], cod[1], dom[0], index)
            })
            .unwrap();
        let b: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&x], [&x], |trees, index| {
                b_value(trees.domain_uncoupled()[0], index)
            })
            .unwrap();
        let key = debug_key::<(u8, u8)>;
        let mut rows: Rows = Vec::new();
        for (order, codomain, swapped) in [
            ("core", &[0usize, 1][..], false),
            ("swapped", &[1, 0][..], true),
        ] {
            let spec = ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain,
                domain: &[2],
            };
            let c = a.contract(&b, &spec).unwrap();
            let mut blocks = 0;
            for (trees, view) in c.subblocks().unwrap() {
                let cod = trees.codomain_uncoupled();
                let (v, w) = if swapped {
                    (cod[1], cod[0])
                } else {
                    (cod[0], cod[1])
                };
                let y = trees.domain_uncoupled()[0];
                let to = |s: (u8, u8)| (usize::from(s.0), usize::from(s.1));
                let phase = if swapped {
                    alternating_r(to(v), to(w))
                } else {
                    Complex64::new(1.0, 0.0)
                };
                let shape = view.shape().to_vec();
                let dx = 1 + usize::from((y.0 + 3 * y.1) % 2);
                for i0 in 0..shape[0] {
                    for i1 in 0..shape[1] {
                        for iy in 0..shape[2] {
                            let (iv, iw) = if swapped { (i1, i0) } else { (i0, i1) };
                            let sum: Complex64 = (0..dx)
                                .map(|ix| a_value(v, w, y, &[iv, iw, ix]) * b_value(y, &[ix, iy]))
                                .sum();
                            let got = *view.get(&[i0, i1, iy]).unwrap();
                            let want = phase * sum;
                            assert!(
                                close(got, want),
                                "{order} {v:?} {w:?} [{i0}, {i1}, {iy}]: \
                                 {got} vs {want}"
                            );
                        }
                    }
                }
                blocks += 1;
            }
            assert!(blocks > 0, "{order}");
            contract_cell!(
                rows,
                &runtime,
                a,
                b,
                &b,
                spec,
                key,
                Complex64,
                format!("Z3 x Z3 {order}")
            );
        }
        assert_eq!(rows.len(), 2 * BETAS.len());
    }
}

#[test]
fn anyonic_complex_symbol_contract_into_is_refused_like_eager() {
    // Fibonacci stays outside `contract` (anyonic), and `_into` refuses it
    // with eager's error, before touching the destination.
    let runtime = host_runtime();
    let leg = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 2), (TableSector::new(2), 3)],
    )
    .unwrap();
    let key = debug_key::<TableSector>;
    let a: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index| {
            complex_value(&key(trees), index)
        })
        .unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    let eager = a.contract(&a, &spec).unwrap_err();
    let mut destination: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index| {
            complex_value(&format!("dst{}", key(trees)), index)
        })
        .unwrap();
    let before = dense(&destination);
    let into = a
        .contract_into(
            &a,
            &spec,
            &mut destination,
            Complex64::alpha(),
            Complex64::lift(0.5),
        )
        .unwrap_err();
    assert_eq!(format!("{into}"), format!("{eager}"));
    assert_eq!(dense(&destination), before);
}
