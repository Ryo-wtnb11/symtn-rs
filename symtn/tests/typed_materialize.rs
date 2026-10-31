//! `TensorMap::materialize` (#1545), TensorKit `copy`.
//!
//! Oracles, none of which goes through the reduced-block adjoint map:
//! - U(1) and SU(2): the physical-basis expansion of the result is the
//!   conjugate transpose of the parent's (TensorKit `convert(Array, t')`).
//! - fZ2×U(1) and checked-Generic SU(3) (no physical basis): TensorKit's
//!   `subblock(t', (f₁, f₂)) = subblock(t, (f₂, f₁))'`, each block located by
//!   its decoded trees (uncoupled sectors, inner lines, vertex labels).
//! - Compact diagonals: the physical expansion is the diagonal matrix of the
//!   supplied values in TensorKit sector order.

use crate::common::ErrorKind as _;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::__network::NetworkReuseClass;
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap};

trait Scalar: Copy + std::fmt::Debug + PartialEq {
    fn conjugate(self) -> Self;
    fn distance(self, other: Self) -> f64;
}

impl Scalar for f64 {
    fn conjugate(self) -> Self {
        self
    }
    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl Scalar for Complex64 {
    fn conjugate(self) -> Self {
        self.conj()
    }
    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

use crate::fixtures;

use fixtures::host_runtime;

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
            (SU2Irrep::from_twice_spin(2), 2),
        ],
    )
    .unwrap()
}

macro_rules! fz2_u1_leg {
    () => {
        GradedSpace::try_new(
            Arc::new(
                symtn_sectors::ProductFusionRule::<_, _>::try_new(
                    FermionParityFusionRule,
                    U1FusionRule,
                )
                .unwrap(),
            ),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
            ],
        )
        .unwrap()
    };
}

/// The postconditions every input shares: same space, provider, runtime and
/// placement; owned dense; a payload allocation of its own; a write to the
/// result stays in the result.
macro_rules! assert_independent_owned_copy {
    ($engine:ident, $input:expr, $result:expr, $what:expr) => {{
        let (input, result, what) = (&$input, $result, $what);
        assert!(
            symtn::typed::__network::network_reuse_class(&result, false)
                == NetworkReuseClass::OwnedDense,
            "{what}: not lazy, not compact"
        );
        assert_eq!(result.codomain(), input.codomain(), "{what}: codomain");
        assert_eq!(result.domain(), input.domain(), "{what}: domain");
        assert!(
            std::ptr::eq(result.provider(), input.provider()),
            "{what}: provider allocation"
        );
        assert!(
            symtn::typed::__network::runtime_identity(result.runtime())
                == symtn::typed::__network::runtime_identity(input.runtime()),
            "{what}: runtime"
        );
        assert_eq!(result.placement(), input.placement(), "{what}: placement");
        let (payload, _) = symtn::typed::__network::network_owned_payload(&result).unwrap();
        if let Some((input_payload, _)) = symtn::typed::__network::network_owned_payload(&input) {
            assert_ne!(payload, input_payload, "{what}: fresh payload");
        }

        let before = input
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec();
        let mut written = result;
        let pointer = written.dense_data().unwrap().as_ptr();
        written.scale_assign($engine, 2.0.into()).unwrap();
        // In place means uniquely owned: nothing else holds this payload.
        assert_eq!(
            written.dense_data().unwrap().as_ptr(),
            pointer,
            "{what}: writable in place"
        );
        assert_eq!(
            input.materialize($engine).unwrap().dense_data().unwrap(),
            before.as_slice(),
            "{what}: input unchanged"
        );
    }};
}

/// Physical oracle: `adjoint` is the conjugate transpose of `parent` in the
/// physical basis. Returns how many entries would also match without
/// conjugation.
fn assert_physical_dagger<D: Scalar>(
    what: &str,
    shape: &[usize],
    nout: usize,
    parent: &[D],
    adjoint_shape: &[usize],
    adjoint: &[D],
) -> usize {
    let rows: usize = shape[..nout].iter().product();
    let cols: usize = shape[nout..].iter().product();
    let mut expected_shape = shape[nout..].to_vec();
    expected_shape.extend_from_slice(&shape[..nout]);
    assert_eq!(adjoint_shape, expected_shape.as_slice(), "{what}: shape");
    assert_eq!(adjoint.len(), rows * cols, "{what}: length");
    let mut unconjugated = 0;
    for row in 0..rows {
        for col in 0..cols {
            let parent_entry = parent[row + rows * col];
            let adjoint_entry = adjoint[col + cols * row];
            assert!(
                parent_entry.conjugate().distance(adjoint_entry) < 1e-12,
                "{what}: entry ({row},{col})"
            );
            if parent_entry.distance(adjoint_entry) < 1e-12 {
                unconjugated += 1;
            }
        }
    }
    unconjugated
}

macro_rules! physical_case {
    ($engine:ident, $leg:expr, $what:expr, $dtype:ty, $codomain:expr, $domain:expr, $seed:expr) => {{
        let runtime = host_runtime();
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let what: &str = $what;
        let tensor: TensorMap<_, $dtype> = TensorMap::rand_with_seed(
            $engine,
            &runtime,
            $codomain(&leg, &dual),
            $domain(&leg, &dual),
            $seed,
        )
        .unwrap();
        assert!(tensor.subblock_count() >= 2, "{what}: multi-block fixture");

        // Owned dense input: a copy with equal values.
        let copy = tensor.materialize($engine).unwrap();
        assert_eq!(
            copy.dense_data().unwrap(),
            tensor.dense_data().unwrap(),
            "{what}: owned copy values"
        );
        assert_independent_owned_copy!($engine, tensor, copy, what);

        // Lazy adjoint input.
        let lazy = tensor.adjoint($engine).unwrap();
        assert!(
            symtn::typed::__network::network_reuse_class(&lazy, false)
                == NetworkReuseClass::LazyAdjoint
        );
        let owned = lazy.materialize($engine).unwrap();
        let parent = tensor.to_physical_dense($engine).unwrap();
        let adjoint = owned.to_physical_dense($engine).unwrap();
        let unconjugated = assert_physical_dagger::<$dtype>(
            what,
            &parent.shape,
            tensor.codomain_rank(),
            &parent.data,
            &adjoint.shape,
            &adjoint.data,
        );
        assert_independent_owned_copy!($engine, lazy, owned, what);
        (unconjugated, adjoint.data.len())
    }};
}

#[test]
fn materialize_matches_the_physical_conjugate_transpose() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for (unconjugated, len) in [
        physical_case!(
            engine,
            u1_leg(),
            "U1 c64 2<-2",
            Complex64,
            |l, d| [l, d],
            |d, l| [l, d],
            2
        ),
        physical_case!(
            engine,
            u1_leg(),
            "U1 c64 1<-2",
            Complex64,
            |l, _d| [l],
            |l, d| [d, l],
            3
        ),
        physical_case!(
            engine,
            u1_leg(),
            "U1 c64 3<-1",
            Complex64,
            |l, d| [d, l, l],
            |_l, d| [d],
            4
        ),
        physical_case!(
            engine,
            su2_leg(),
            "SU2 c64 2<-2",
            Complex64,
            |l, d| [l, d],
            |d, l| [l, d],
            6
        ),
        physical_case!(
            engine,
            su2_leg(),
            "SU2 c64 1<-2",
            Complex64,
            |l, _d| [l],
            |l, d| [d, l],
            7
        ),
        physical_case!(
            engine,
            su2_leg(),
            "SU2 c64 3<-1",
            Complex64,
            |l, d| [d, l, l],
            |_l, d| [d],
            8
        ),
    ] {
        // Negative control: a transpose without conjugation must not pass.
        assert!(unconjugated < len, "conjugation is not observable");
    }
    physical_case!(
        engine,
        u1_leg(),
        "U1 f64 2<-2",
        f64,
        |l, d| [l, d],
        |l, d| [l, d],
        1
    );
    physical_case!(
        engine,
        u1_leg(),
        "U1 f64 1<-2",
        f64,
        |l, _d| [l],
        |l, d| [d, l],
        9
    );
    physical_case!(
        engine,
        su2_leg(),
        "SU2 f64 2<-2",
        f64,
        |l, d| [l, d],
        |l, d| [l, d],
        5
    );
    physical_case!(
        engine,
        su2_leg(),
        "SU2 f64 3<-1",
        f64,
        |l, d| [d, l, l],
        |_l, d| [d],
        10
    );
}

/// TensorKit `subblock(t', (f₁, f₂)) = subblock(t, (f₂, f₁))'`: every block
/// of `owned` is the conjugate transpose of the parent block with swapped
/// trees. Returns (entries matching without conjugation, entries, whether a
/// vertex label above 1 was seen).
macro_rules! assert_swapped_tree_dagger {
    ($parent:expr, $owned:expr, $what:expr) => {{
        let (parent, owned, what) = (&$parent, &$owned, $what);
        assert_eq!(
            owned.subblock_count(),
            parent.subblock_count(),
            "{what}: blocks"
        );
        let parent_blocks: Vec<_> = parent.subblocks().unwrap().collect();
        let adjoint_nout = owned.codomain_rank();
        let parent_nout = parent.codomain_rank();
        let (mut unconjugated, mut entries, mut multiplicity) = (0, 0, false);
        for (trees, view) in owned.subblocks().unwrap() {
            multiplicity |= trees
                .codomain_vertices()
                .iter()
                .chain(trees.domain_vertices())
                .any(|v| v.get() > 1);
            let (_, parent_view) = parent_blocks
                .iter()
                .find(|(p, _)| {
                    p.coupled() == trees.coupled()
                        && p.codomain_uncoupled() == trees.domain_uncoupled()
                        && p.domain_uncoupled() == trees.codomain_uncoupled()
                        && p.codomain_innerlines() == trees.domain_innerlines()
                        && p.domain_innerlines() == trees.codomain_innerlines()
                        && p.codomain_vertices() == trees.domain_vertices()
                        && p.domain_vertices() == trees.codomain_vertices()
                })
                .unwrap_or_else(|| panic!("{what}: no parent block with swapped trees"));
            let shape = view.shape().to_vec();
            let rank = shape.len();
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
                let parent_index: Vec<usize> = (0..rank)
                    .map(|axis| {
                        if axis < parent_nout {
                            index[adjoint_nout + axis]
                        } else {
                            index[axis - parent_nout]
                        }
                    })
                    .collect();
                let expected = *parent_view.get(&parent_index).unwrap();
                let actual = *view.get(&index).unwrap();
                assert!(
                    expected.conjugate().distance(actual) < 1e-12,
                    "{what}: entry {index:?}"
                );
                if expected.distance(actual) < 1e-12 {
                    unconjugated += 1;
                }
                entries += 1;
            }
        }
        (unconjugated, entries, multiplicity)
    }};
}

macro_rules! fermionic_case {
    ($engine:ident, $dtype:ty, $what:expr, $codomain:expr, $domain:expr, $seed:expr) => {{
        let runtime = host_runtime();
        let leg = fz2_u1_leg!();
        let dual = leg.try_dual().unwrap();
        let what: &str = $what;
        let tensor: TensorMap<_, $dtype> = TensorMap::rand_with_seed(
            $engine,
            &runtime,
            $codomain(&leg, &dual),
            $domain(&leg, &dual),
            $seed,
        )
        .unwrap();
        assert!(tensor.subblock_count() >= 2, "{what}: multi-block fixture");
        let copy = tensor.materialize($engine).unwrap();
        assert_eq!(
            copy.dense_data().unwrap(),
            tensor.dense_data().unwrap(),
            "{what}: owned copy values"
        );
        assert_independent_owned_copy!($engine, tensor, copy, what);

        let lazy = tensor.adjoint($engine).unwrap();
        let owned = lazy.materialize($engine).unwrap();
        let (unconjugated, entries, _) = assert_swapped_tree_dagger!(tensor, owned, what);
        assert_independent_owned_copy!($engine, lazy, owned, what);
        (unconjugated, entries)
    }};
}

#[test]
fn materialize_matches_swapped_trees_for_fermionic_z2_times_u1() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (unconjugated, entries) = fermionic_case!(
        engine,
        Complex64,
        "fZ2xU1 c64 2<-2",
        |l, d| [l, d],
        |d, l| [l, d],
        11
    );
    assert!(unconjugated < entries, "conjugation is not observable");
    let (unconjugated, entries) = fermionic_case!(
        engine,
        Complex64,
        "fZ2xU1 c64 1<-2",
        |l, _d| [l],
        |l, d| [d, l],
        12
    );
    assert!(unconjugated < entries, "conjugation is not observable");
    fermionic_case!(
        engine,
        f64,
        "fZ2xU1 f64 2<-2",
        |l, d| [l, d],
        |l, d| [l, d],
        13
    );
    fermionic_case!(
        engine,
        f64,
        "fZ2xU1 f64 3<-1",
        |l, d| [d, l, l],
        |_l, d| [d],
        14
    );
}

/// The physical expansion of a compact diagonal, in TensorKit order: sectors
/// by `sector_order`, then degeneracy, then carrier index (fastest).
fn expected_diagonal<D: Scalar + Default>(blocks: &[(Vec<D>, usize)]) -> (usize, Vec<D>) {
    let n: usize = blocks.iter().map(|(v, carrier)| v.len() * carrier).sum();
    let mut data = vec![D::default(); n * n];
    let mut position = 0;
    for (values, carrier) in blocks {
        for &value in values {
            for _ in 0..*carrier {
                data[position + n * position] = value;
                position += 1;
            }
        }
    }
    (n, data)
}

macro_rules! diagonal_case {
    ($engine:ident, $bond:expr, $dtype:ty, $spectra:expr, $expected:expr, $what:expr) => {{
        let runtime = host_runtime();
        let what: &str = $what;
        let bond = $bond;
        let diagonal: TensorMap<_, $dtype> =
            TensorMap::diagonal($engine, &runtime, &bond, $spectra).unwrap();
        assert!(
            symtn::typed::__network::network_reuse_class(&diagonal, false)
                == NetworkReuseClass::Compact
        );
        let dense = diagonal.materialize($engine).unwrap();
        let physical = dense.to_physical_dense($engine).unwrap();
        let (n, expected) = expected_diagonal::<$dtype>(&$expected);
        assert_eq!(physical.shape, vec![n, n], "{what}: shape");
        for (i, (&actual, &want)) in physical.data.iter().zip(&expected).enumerate() {
            assert!(actual.distance(want) < 1e-15, "{what}: entry {i}");
        }
        assert_independent_owned_copy!($engine, diagonal, dense, what);
        // The input stays compact.
        assert!(
            symtn::typed::__network::network_reuse_class(&diagonal, false)
                == NetworkReuseClass::Compact
        );
    }};
}

#[test]
fn materialize_densifies_a_compact_diagonal() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    // U(1) sector order is 0, 1, -1. The dual bond carries the dual labels
    // and is listed in the order of the space it dualizes.
    for (bond, sign) in [(u1_leg(), 1), (u1_leg().try_dual().unwrap(), -1)] {
        let q = |charge: i32| U1Irrep::new(sign * charge);
        diagonal_case!(
            engine,
            bond,
            Complex64,
            [
                SectorSpectrum {
                    sector: q(-1),
                    values: vec![c(5.0, 1.0), c(6.0, -1.0)]
                },
                SectorSpectrum {
                    sector: q(0),
                    values: vec![c(1.0, 0.5)]
                },
                SectorSpectrum {
                    sector: q(1),
                    values: vec![c(2.0, 0.0), c(3.0, 2.0), c(4.0, -3.0)]
                },
            ],
            [
                (vec![c(1.0, 0.5)], 1),
                (vec![c(2.0, 0.0), c(3.0, 2.0), c(4.0, -3.0)], 1),
                (vec![c(5.0, 1.0), c(6.0, -1.0)], 1),
            ],
            "U1 c64 diagonal"
        );
    }
    // SU(2) order is 0, 1/2, 1 with carrier dimensions 1, 2, 3.
    diagonal_case!(
        engine,
        su2_leg(),
        f64,
        [
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(2),
                values: vec![4.0, -5.0]
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(0),
                values: vec![1.0, -2.0]
            },
            SectorSpectrum {
                sector: SU2Irrep::from_twice_spin(1),
                values: vec![3.0]
            },
        ],
        [(vec![1.0, -2.0], 1), (vec![3.0], 2), (vec![4.0, -5.0], 3)],
        "SU2 f64 diagonal"
    );
}

#[test]
fn materialize_is_the_remedy_for_apis_that_reject_lazy_adjoints() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let leg = u1_leg();
    let tensor: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg], 21).unwrap();
    let lazy = tensor.adjoint(engine).unwrap();
    assert!(lazy.diagview(engine).is_invalid_argument("lazy adjoint"));
    let diagonal = lazy.materialize(engine).unwrap().diagview(engine).unwrap();
    let parent = tensor.diagview(engine).unwrap();
    for (entry, parent) in diagonal.iter().zip(&parent) {
        assert_eq!(entry.sector, parent.sector);
        let conjugated: Vec<_> = parent.values.iter().map(|v| v.conj()).collect();
        assert_eq!(entry.values, conjugated);
    }

    // An owned dense destination can be overwritten; the input is untouched.
    let before = tensor.dense_data().unwrap().to_vec();
    let mut destination = tensor.materialize(engine).unwrap();
    let pointer = destination.dense_data().unwrap().as_ptr();
    lazy.materialize(engine)
        .unwrap()
        .permute_into(
            engine,
            &[0],
            &[1],
            &mut destination,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
    assert_eq!(destination.dense_data().unwrap().as_ptr(), pointer);
    assert_eq!(tensor.dense_data().unwrap(), before.as_slice());
}

#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use super::*;
    use symtn::sector::LieGroupFusionRule;

    macro_rules! su3_case {
        ($engine:ident, $dtype:ty, $value:expr, $what:expr) => {{
            let (value, what): (fn(f64, f64) -> $dtype, &str) = ($value, $what);
            let runtime = host_runtime();
            let provider = Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            );
            // The adjoint irrep 8 (Dynkin [1, 1]) couples 8 ⊗ 8 → 8 with multiplicity 2.
            let a = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
            let b = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 3)]).unwrap();
            let b_dual = b.try_dual().unwrap();
            let tensor: TensorMap<_, $dtype> = TensorMap::from_subblock_fn(
                $engine,
                &runtime,
                [&a, &b],
                [&b_dual],
                |trees, index| {
                    let vertex = trees
                        .codomain_vertices()
                        .iter()
                        .map(|m| m.get())
                        .sum::<usize>() as f64;
                    let spread = index
                        .iter()
                        .enumerate()
                        .map(|(axis, i)| (i * (axis + 1) * 3) as f64)
                        .sum::<f64>();
                    value(1.0 + vertex * 10.0 + spread, 0.5 + index[0] as f64 - vertex)
                },
            )
            .unwrap();
            let copy = tensor.materialize($engine).unwrap();
            assert_eq!(
                copy.dense_data().unwrap(),
                tensor.dense_data().unwrap(),
                "{what}: owned copy values"
            );
            assert_independent_owned_copy!($engine, tensor, copy, what);

            let lazy = tensor.adjoint($engine).unwrap();
            assert!(
                symtn::typed::__network::network_reuse_class(&lazy, false)
                    == NetworkReuseClass::LazyAdjoint
            );
            // The #1545 consumer: a lazy adjoint's QR materializes it for the
            // call (#1755), so it factors exactly like the owned copy.
            let owned = lazy.materialize($engine).unwrap();
            assert_eq!(
                lazy.qr_compact($engine, &[0], &[1, 2])
                    .unwrap()
                    .r
                    .dense_data()
                    .unwrap(),
                owned
                    .qr_compact($engine, &[0], &[1, 2])
                    .unwrap()
                    .r
                    .dense_data()
                    .unwrap()
            );
            let (unconjugated, entries, multiplicity) =
                assert_swapped_tree_dagger!(tensor, owned, what);
            assert!(
                multiplicity,
                "{what}: fixture must carry a vertex label above 1"
            );
            assert_independent_owned_copy!($engine, lazy, owned, what);
            (unconjugated, entries)
        }};
    }

    #[test]
    fn materialize_matches_swapped_trees_for_su3_with_multiplicity() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let (unconjugated, entries) = su3_case!(engine, Complex64, Complex64::new, "SU3 c64");
        assert!(unconjugated < entries, "conjugation is not observable");
        su3_case!(engine, f64, |re, _| re, "SU3 f64");
    }
}
