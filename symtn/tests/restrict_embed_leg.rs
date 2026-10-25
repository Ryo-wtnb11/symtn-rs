//! Semantics of `TensorMap::restrict_leg` / `embed_leg` (#1285).
//!
//! Two oracles, both independent of the restriction kernel under test:
//!
//! * a **dense gather**: `to_physical_dense` of the result must equal a
//!   per-sector gather of `to_physical_dense` of the source, with the dense
//!   offsets of the restricted leg recomputed (the physical order is
//!   TensorKit's sector order, then degeneracy, then carrier index);
//! * an **inclusion isometry**: `ι_σ` hand-filled with `from_subblock_fn`, padded
//!   with `TensorMap::isomorphism(V, V)` through `otimes`, and applied with `compose`.
//!   `contract` is deliberately not used: it applies the fermionic supertrace
//!   twist to every dual contracted right-hand axis, so it would test a
//!   different map on odd sectors.

use crate::report;
use report::Report;

use std::ops::Range;
use std::sync::Arc;
use symtn::typed::ContractSpec;

use num_complex::Complex64;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep, ZNFusionRule,
};
use symtn::typed::BlockFusionTrees;
use symtn::typed::{GradedSpace, LegSelection, Runtime, TensorMap};

/// `(degeneracy, carrier dimension)` per sector, in the leg's physical order.
type AxisLayout = Vec<(usize, usize)>;

/// Destination dense index -> source dense index for one axis.
///
/// `selection[i]` is the kept degeneracy positions of the `i`-th sector, in
/// order, `None` when that sector is dropped entirely.
fn axis_gather(layout: &AxisLayout, selection: &[Option<Vec<usize>>]) -> Vec<usize> {
    let mut map = Vec::new();
    let mut offset = 0;
    for (position, &(degeneracy, carrier)) in layout.iter().enumerate() {
        if let Some(kept) = &selection[position] {
            for &index in kept {
                for basis in 0..carrier {
                    map.push(offset + index * carrier + basis);
                }
            }
        }
        offset += degeneracy * carrier;
    }
    map
}

fn identity_gather(layout: &AxisLayout) -> Vec<usize> {
    (0..layout
        .iter()
        .map(|&(degeneracy, carrier)| degeneracy * carrier)
        .sum())
        .collect()
}

/// Column-major gather of `data` along every axis.
fn gather<D: Copy>(shape: &[usize], data: &[D], maps: &[Vec<usize>]) -> (Vec<usize>, Vec<D>) {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let out_shape: Vec<usize> = maps.iter().map(Vec::len).collect();
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        let source: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &position)| maps[axis][position] * strides[axis])
            .sum();
        out.push(data[source]);
        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    (out_shape, out)
}

/// `data` with every entry outside the kept index sets replaced by zero.
fn masked(shape: &[usize], data: &[f64], maps: &[Vec<usize>]) -> Vec<f64> {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let mut out = vec![0.0; data.len()];
    let total: usize = maps.iter().map(Vec::len).product();
    let out_shape: Vec<usize> = maps.iter().map(Vec::len).collect();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        let position: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &offset)| maps[axis][offset] * strides[axis])
            .sum();
        out[position] = data[position];
        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// `source` placed into a zero buffer of `shape` at the gathered indices: the
/// dense form of an embedding.
fn scattered<D: Copy>(shape: &[usize], source: &[D], maps: &[Vec<usize>], zero: D) -> Vec<D> {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    let out_shape: Vec<usize> = maps.iter().map(Vec::len).collect();
    let mut out = vec![zero; shape.iter().product()];
    let total: usize = out_shape.iter().product();
    let mut index = vec![0usize; shape.len()];
    for &value in source.iter().take(total) {
        let destination: usize = index
            .iter()
            .enumerate()
            .map(|(axis, &offset)| maps[axis][offset] * strides[axis])
            .sum();
        out[destination] = value;
        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < out_shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

use crate::numerics;

/// Floating terms reaching one compared entry: a fusion-tree expansion to
/// the physical basis, or a composition with an inclusion isometry over a
/// leg of at most this many states.
const TERMS: usize = 8;

fn u1(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2(provider: &Arc<SU2FusionRule>, pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

fn fz2(
    provider: &Arc<FermionParityFusionRule>,
    pairs: &[(bool, usize)],
) -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(odd, degeneracy)| (if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN }, degeneracy)),
    )
    .unwrap()
}

/// A sector's TensorKitSectors `findindex` position on a leg. A dual leg is
/// listed in its parent's order, so the position is that of the dual sector.
trait TensorKitPosition {
    fn tensorkit_position(&self, is_dual: bool) -> u64;
}

impl TensorKitPosition for U1Irrep {
    /// TensorKit U(1) order `0, 1, -1, 2, -2, ...` (integer charges only).
    fn tensorkit_position(&self, is_dual: bool) -> u64 {
        let charge = if is_dual {
            -self.charge()
        } else {
            self.charge()
        };
        2 * u64::from(charge.unsigned_abs()) - u64::from(charge > 0)
    }
}

impl TensorKitPosition for SU2Irrep {
    fn tensorkit_position(&self, _is_dual: bool) -> u64 {
        self.twice_spin() as u64
    }
}

/// The physical-order `(degeneracy, carrier)` layout and the per-position
/// selected positions the dense oracle needs.
fn dense_plan<S: PartialEq + TensorKitPosition, P: Clone + IntoIterator<Item = usize>>(
    sectors: &[S],
    degeneracies: &[usize],
    is_dual: bool,
    carrier: impl Fn(&S) -> usize,
    selected: &[(S, P)],
) -> (AxisLayout, Vec<Option<Vec<usize>>>) {
    let mut order = sectors.iter().zip(degeneracies).collect::<Vec<_>>();
    order.sort_by_key(|(sector, _)| sector.tensorkit_position(is_dual));
    let (sectors, degeneracies): (Vec<_>, Vec<_>) = order.into_iter().unzip();
    let layout: AxisLayout = sectors
        .iter()
        .zip(degeneracies)
        .map(|(sector, &degeneracy)| (degeneracy, carrier(sector)))
        .collect();
    let selection = sectors
        .iter()
        .map(|sector| {
            selected
                .iter()
                .find(|(candidate, _)| candidate == *sector)
                .map(|(_, positions)| positions.clone().into_iter().collect())
        })
        .collect();
    (layout, selection)
}

#[test]
fn restrict_matches_the_dense_gather_on_a_dual_u1_codomain_leg() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    // Degeneracies asymmetric under c -> -c, so a label/dual mix-up cannot
    // survive by coincidence.
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 4)])
        .try_dual()
        .unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 3)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&other], 7).unwrap();

    // Selection in the dual leg's own labels: the caller never dualises.
    let selected = [(U1Irrep::new(-1), 1..2), (U1Irrep::new(1), 0..2)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = source.restrict_leg(&[(0, &selection)]).unwrap();

    assert_eq!(restricted.codomain()[0], *selection.subspace());
    assert_eq!(restricted.codomain()[1], other);
    assert_eq!(restricted.domain()[0], other);
    assert!(restricted.codomain()[0].is_dual());

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        |_| 1,
        &selected,
    );
    let (other_layout, _) = dense_plan::<U1Irrep, Range<usize>>(
        &other.sectors().unwrap(),
        other.degeneracies(),
        other.is_dual(),
        |_| 1,
        &[],
    );
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        axis_gather(&layout, &ranges),
        identity_gather(&other_layout),
        identity_gather(&other_layout),
    ];
    let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    numerics::assert_slices_close("physical entries", &actual.data, &expected, TERMS);
}

#[test]
fn restrict_keeps_su2_multiplets_intact_for_a_middle_sector_with_offset() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    // Three sectors so that dropping the middle one shifts the dense offsets
    // of the last, and j = 1/2, 1 carry nontrivial multiplet dimensions.
    let leg = su2(&provider, &[(0, 2), (1, 3), (2, 2)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&other, &other], 11).unwrap();

    let selected = [
        (SU2Irrep::from_twice_spin(0), 1..2),
        (SU2Irrep::from_twice_spin(2), 0..1),
    ];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = source.restrict_leg(&[(0, &selection)]).unwrap();
    assert_eq!(restricted.codomain()[0].degeneracies(), &[1, 1]);

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        |sector: &SU2Irrep| sector.twice_spin() + 1,
        &selected,
    );
    let (other_layout, _) = dense_plan::<SU2Irrep, Range<usize>>(
        &other.sectors().unwrap(),
        other.degeneracies(),
        other.is_dual(),
        |sector| sector.twice_spin() + 1,
        &[],
    );
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        axis_gather(&layout, &ranges),
        identity_gather(&other_layout),
        identity_gather(&other_layout),
    ];
    let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    numerics::assert_slices_close("physical entries", &actual.data, &expected, TERMS);
}

#[test]
fn restrict_equals_composition_with_the_inclusion_isometry_on_both_sides() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let spectator = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&spectator, &leg], [&leg, &spectator], 3).unwrap();

    let selected = [(U1Irrep::new(-1), 0..1), (U1Irrep::new(0), 1..3)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let start = |sector: &U1Irrep| {
        selected
            .iter()
            .find(|(candidate, _)| candidate == sector)
            .map_or(0, |(_, range)| range.start)
    };
    // ι_σ : V <- W, a one on every selected degeneracy coordinate.
    let iota = TensorMap::<_, f64>::from_subblock_fn(
        &runtime,
        [&leg],
        [selection.subspace()],
        |trees, indices| {
            if indices[0] == start(&trees.domain_uncoupled()[0]) + indices[1] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap();
    // The adjoint isometry, hand-filled rather than taken from `adjoint()`.
    let iota_dagger = TensorMap::<_, f64>::from_subblock_fn(
        &runtime,
        [selection.subspace()],
        [&leg],
        |trees, indices| {
            if indices[1] == start(&trees.codomain_uncoupled()[0]) + indices[0] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let spectator_id =
        TensorMap::<_, f64>::isomorphism(&runtime, [&spectator], [&spectator]).unwrap();

    // Codomain leg 1: (id ⊗ ι†) ∘ t.
    let projector = spectator_id.otimes(&iota_dagger).unwrap();
    let expected = projector.compose(&source).unwrap();
    let restricted = source.restrict_leg(&[(1, &selection)]).unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    assert_eq!(restricted.domain(), expected.domain());
    numerics::assert_slices_close(
        "entries",
        restricted.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        TERMS,
    );

    // Domain leg 2: t ∘ (ι ⊗ id).
    let inserter = iota.otimes(&spectator_id).unwrap();
    let expected = source.compose(&inserter).unwrap();
    let restricted = source.restrict_leg(&[(2, &selection)]).unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    assert_eq!(restricted.domain(), expected.domain());
    numerics::assert_slices_close(
        "entries",
        restricted.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        TERMS,
    );
}

#[test]
fn restrict_of_a_fermionic_odd_dual_leg_equals_the_isometry_composition() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let leg = fz2(&provider, &[(false, 2), (true, 3)]).try_dual().unwrap();
    let spectator = fz2(&provider, &[(false, 2), (true, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &spectator], [&spectator], 5).unwrap();

    let selected = [(Z2Irrep::ODD, 1..3)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let start = |sector: &Z2Irrep| if *sector == Z2Irrep::ODD { 1 } else { 0 };
    let iota_dagger = TensorMap::<_, f64>::from_subblock_fn(
        &runtime,
        [selection.subspace()],
        [&leg],
        |trees, indices| {
            if indices[1] == start(&trees.codomain_uncoupled()[0]) + indices[0] {
                1.0
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let spectator_id =
        TensorMap::<_, f64>::isomorphism(&runtime, [&spectator], [&spectator]).unwrap();
    let expected = iota_dagger
        .otimes(&spectator_id)
        .unwrap()
        .compose(&source)
        .unwrap();
    let restricted = source.restrict_leg(&[(0, &selection)]).unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    numerics::assert_slices_close(
        "entries",
        restricted.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        TERMS,
    );
}

#[test]
fn restrict_reads_a_complex_lazy_adjoint_domain_leg_with_a_multi_sector_selection() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = u1(&provider, &[(0, 2), (1, 3)]);
    let columns = u1(&provider, &[(0, 3), (1, 2)]);
    let parent: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&rows], [&columns], 13).unwrap();
    let lazy = parent.adjoint().unwrap();

    // The lazy adjoint's domain leg 1 is the parent's codomain leg.
    let leg = lazy.domain()[0].clone();
    let selected = [(U1Irrep::new(0), 1..2), (U1Irrep::new(1), 0..2)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = lazy.restrict_leg(&[(1, &selection)]).unwrap();

    // Oracle: the dense gather of the adjoint's own physical expansion, which
    // shares no code with the restriction kernel's storage remap and
    // conjugation.
    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        |_| 1,
        &selected,
    );
    let (row_layout, _) = dense_plan::<U1Irrep, Range<usize>>(
        &columns.sectors().unwrap(),
        columns.degeneracies(),
        columns.is_dual(),
        |_| 1,
        &[],
    );
    let dense = lazy.to_physical_dense().unwrap();
    let maps = vec![identity_gather(&row_layout), axis_gather(&layout, &ranges)];
    let (shape, gathered) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    numerics::assert_slices_close("physical entries", &actual.data, &gathered, TERMS);
}

#[test]
fn restrict_commutes_with_adjoint_under_the_codomain_domain_axis_map() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = u1(&provider, &[(0, 2), (1, 3)]);
    let columns = u1(&provider, &[(0, 3), (1, 2)]);
    let source: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&rows], [&columns], 17).unwrap();
    let selection = LegSelection::try_new(&columns, [(U1Irrep::new(1), 0..1)]).unwrap();

    // Domain leg 1 of `t` is codomain leg 0 of `t†`.
    let left = source
        .restrict_leg(&[(1, &selection)])
        .unwrap()
        .adjoint()
        .unwrap();
    let right = source
        .adjoint()
        .unwrap()
        .restrict_leg(&[(0, &selection)])
        .unwrap();
    assert_eq!(left.codomain(), right.codomain());
    assert_eq!(left.domain(), right.domain());
    numerics::assert_slices_close(
        "entries",
        left.materialize().unwrap().dense_data().unwrap(),
        right.dense_data().unwrap(),
        TERMS,
    );
}

#[test]
fn full_selection_is_the_identity_and_round_trips_are_exact() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 2), (1, 2), (2, 1)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&other], 23).unwrap();

    let full = LegSelection::try_new(
        &leg,
        leg.sectors()
            .unwrap()
            .into_iter()
            .zip(leg.degeneracies())
            .map(|(sector, &degeneracy)| (sector, 0..degeneracy)),
    )
    .unwrap();
    assert_eq!(full.subspace(), &leg);
    let restricted = source.restrict_leg(&[(0, &full)]).unwrap();
    assert_eq!(restricted.codomain(), source.codomain());
    assert_eq!(
        restricted.dense_data().unwrap(),
        source.dense_data().unwrap()
    );

    // restrict ∘ embed = id, bitwise.
    let partial = LegSelection::try_new(&leg, [(SU2Irrep::from_twice_spin(1), 1..2)]).unwrap();
    let small = source.restrict_leg(&[(0, &partial)]).unwrap();
    let embedded = small.embed_leg(0, &partial).unwrap();
    assert_eq!(embedded.codomain()[0], leg);
    assert_eq!(
        embedded
            .restrict_leg(&[(0, &partial)])
            .unwrap()
            .dense_data()
            .unwrap(),
        small.dense_data().unwrap()
    );
}

#[test]
fn embed_after_restrict_is_the_orthogonal_projector() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let other = u1(&provider, &[(0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&other, &other], 29).unwrap();

    let selected = [(U1Irrep::new(0), 1..3), (U1Irrep::new(1), 0..1)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let projected = source
        .restrict_leg(&[(0, &selection)])
        .unwrap()
        .embed_leg(0, &selection)
        .unwrap();
    assert_eq!(projected.codomain()[0], leg);

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        |_| 1,
        &selected,
    );
    let (other_layout, _) = dense_plan::<U1Irrep, Range<usize>>(
        &other.sectors().unwrap(),
        other.degeneracies(),
        other.is_dual(),
        |_| 1,
        &[],
    );
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        axis_gather(&layout, &ranges),
        identity_gather(&other_layout),
        identity_gather(&other_layout),
    ];
    let expected = masked(&dense.shape, &dense.data, &maps);
    let actual = projected.to_physical_dense().unwrap();
    assert_eq!(actual.shape, dense.shape);
    numerics::assert_slices_close("physical entries", &actual.data, &expected, TERMS);

    // Idempotent: projecting twice changes nothing.
    let twice = projected
        .restrict_leg(&[(0, &selection)])
        .unwrap()
        .embed_leg(0, &selection)
        .unwrap();
    assert_eq!(twice.dense_data().unwrap(), projected.dense_data().unwrap());
}

#[test]
fn a_valid_selection_with_no_admissible_block_returns_an_empty_tensor() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    // A rank-(1,0) map with a nonzero charge has no admissible fusion tree.
    let charged = u1(&provider, &[(1, 3)]);
    let source = TensorMap::<_, f64>::zeros(&runtime, [&charged], []).unwrap();
    assert_eq!(source.subblock_count(), 0);

    let selection = LegSelection::try_new(&charged, [(U1Irrep::new(1), 1..2)]).unwrap();
    let restricted = source.restrict_leg(&[(0, &selection)]).unwrap();
    assert_eq!(restricted.subblock_count(), 0);
    assert!(restricted.dense_data().unwrap().is_empty());
    assert_eq!(restricted.codomain()[0].degeneracies(), &[1]);

    let embedded = restricted.embed_leg(0, &selection).unwrap();
    assert_eq!(embedded.codomain()[0], charged);
    assert!(embedded.dense_data().unwrap().is_empty());

    // Leg-level validation still rejects malformed ranges there, because no
    // block would ever reach the kernel.
    assert!(LegSelection::try_new(&charged, [(U1Irrep::new(1), 2..4)])
        .err()
        .unwrap()
        .report()
        .contains("exceeds its"));
}

#[test]
fn selections_are_validated_against_their_parent_leg_before_anything_is_built() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(0, 2), (1, 3)]);
    let other = u1(&provider, &[(0, 2)]);

    assert!(LegSelection::try_new(&leg, std::iter::empty::<(U1Irrep, Range<usize>)>()).is_err());
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(0), 1..1)])
        .err()
        .unwrap()
        .report()
        .contains("must name at least one position"));
    assert!(
        LegSelection::try_new(&leg, [(U1Irrep::new(0), Range { start: 2, end: 1 })])
            .err()
            .unwrap()
            .report()
            .contains("must name at least one position")
    );
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..3)])
        .err()
        .unwrap()
        .report()
        .contains("exceeds its"));
    assert!(LegSelection::try_new(&leg, [(U1Irrep::new(5), 0..1)])
        .err()
        .unwrap()
        .report()
        .contains("absent from the selected leg"));
    assert!(LegSelection::try_new(
        &leg,
        [(U1Irrep::new(0), 0..1), (U1Irrep::new(0), 1..2)].into_iter()
    )
    .err()
    .unwrap()
    .report()
    .contains("selected more than once"));
    // A dual leg's own labels are the dualised ones; the non-dual label is
    // simply absent.
    let dual = leg.try_dual().unwrap();
    assert!(LegSelection::try_new(&dual, [(U1Irrep::new(1), 0..1)])
        .err()
        .unwrap()
        .report()
        .contains("absent from the selected leg"));
    assert!(LegSelection::try_new(&dual, [(U1Irrep::new(-1), 0..1)]).is_ok());

    let selection = LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..1)]).unwrap();
    let source = TensorMap::<_, f64>::zeros(&runtime, [&leg], [&other]).unwrap();
    assert!(source
        .restrict_leg(&[(2, &selection)])
        .err()
        .unwrap()
        .report()
        .contains("axis 2 is out of range"));
    assert!(source
        .restrict_leg(&[(1, &selection)])
        .err()
        .unwrap()
        .report()
        .contains("axis 1 is not the leg"));
    // `embed_leg` wants the subspace, not the parent.
    assert!(source
        .embed_leg(0, &selection)
        .err()
        .unwrap()
        .report()
        .contains("axis 0 is not the leg"));
    let small = source.restrict_leg(&[(0, &selection)]).unwrap();
    assert!(small
        .restrict_leg(&[(0, &selection)])
        .err()
        .unwrap()
        .report()
        .contains("axis 0 is not the leg"));
    assert!(small.embed_leg(0, &selection).is_ok());
}

#[test]
fn a_compact_diagonal_restriction_other_than_one_selection_on_both_legs_is_rejected() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(0, 3), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 31).unwrap();
    let diagonal = source.svd_compact(&[0], &[1]).unwrap().s;
    let bond = diagonal.domain()[0].clone();
    let selection = LegSelection::try_new(&bond, [(U1Irrep::new(0), 0..1)]).unwrap();
    let different = LegSelection::try_new(&bond, [(U1Irrep::new(0), 1..2)]).unwrap();
    // One leg only, or two different selections: the result would not be an
    // endomorphism, so it cannot stay compact, and densifying is the
    // caller's explicit `materialize`.
    for set in [
        vec![(1, &selection)],
        vec![(0, &selection)],
        vec![(0, &selection), (1, &different)],
    ] {
        assert_eq!(
            diagonal.restrict_leg(&set).err().unwrap(),
            symtn::typed::Error::Unsupported {
                operation: "restrict_leg",
                alternative: symtn::typed::Alternative::Materialize,
            }
        );
        assert!(diagonal.materialize().unwrap().restrict_leg(&set).is_ok());
    }
    // Per-pair checks come first and keep their single-axis errors.
    assert_eq!(
        format!(
            "{:?}",
            diagonal.restrict_leg(&[(2, &selection)]).err().unwrap()
        ),
        format!(
            "{:?}",
            diagonal
                .materialize()
                .unwrap()
                .restrict_leg(&[(2, &selection)])
                .err()
                .unwrap()
        )
    );
    assert!(diagonal.embed_leg(1, &selection).is_err());
}

#[test]
fn embed_scatters_a_complex_lazy_adjoint_dual_domain_leg_per_sector() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 4)])
        .try_dual()
        .unwrap();
    let selected = [(U1Irrep::new(1), 0..1), (U1Irrep::new(-1), 2..4)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let spectator = u1(&provider, &[(0, 2), (1, 1)]);
    let columns = u1(&provider, &[(-1, 1), (0, 2)]);

    // The parent already lives on the subspace, so the lazy adjoint itself is
    // the embed source: a dual leg, in the domain, complex, read through the
    // adjoint axis map, with a multi-sector table.
    let parent: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [selection.subspace(), &spectator], [&columns], 53)
            .unwrap();
    let lazy = parent.adjoint().unwrap();
    assert_eq!(lazy.domain()[0], *selection.subspace());

    let embedded = lazy.embed_leg(1, &selection).unwrap();
    assert_eq!(embedded.domain()[0], leg);
    assert_eq!(embedded.domain()[1], spectator);
    assert_eq!(embedded.codomain()[0], columns);

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        |_| 1,
        &selected,
    );
    let (column_layout, _) = dense_plan::<U1Irrep, Range<usize>>(
        &columns.sectors().unwrap(),
        columns.degeneracies(),
        columns.is_dual(),
        |_| 1,
        &[],
    );
    let (spectator_layout, _) = dense_plan::<U1Irrep, Range<usize>>(
        &spectator.sectors().unwrap(),
        spectator.degeneracies(),
        spectator.is_dual(),
        |_| 1,
        &[],
    );
    let maps = vec![
        identity_gather(&column_layout),
        axis_gather(&layout, &ranges),
        identity_gather(&spectator_layout),
    ];
    let source = lazy.to_physical_dense().unwrap();
    let actual = embedded.to_physical_dense().unwrap();
    let expected = scattered(&actual.shape, &source.data, &maps, Complex64::new(0.0, 0.0));
    numerics::assert_slices_close("physical entries", &actual.data, &expected, TERMS);

    // And the restriction of that embedding is the original, bitwise.
    assert_eq!(
        embedded
            .restrict_leg(&[(1, &selection)])
            .unwrap()
            .dense_data()
            .unwrap(),
        lazy.materialize().unwrap().dense_data().unwrap()
    );
}

#[test]
fn restrict_and_embed_on_a_product_fz2_u1_leg_match_the_isometry_composition() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 2),
            (label(true, 1), 3),
            (label(false, 2), 2),
        ],
    )
    .unwrap();
    let other = GradedSpace::try_new(
        Arc::clone(&provider),
        [(label(false, 0), 2), (label(true, 1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &other], [&other], 59).unwrap();

    let selected = [(label(true, 1), 1..3), (label(false, 2), 0..1)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let start = |sector: &_| {
        selected
            .iter()
            .find(|(candidate, _)| candidate == sector)
            .map_or(0, |(_, range)| range.start)
    };
    let restricted = source.restrict_leg(&[(0, &selection)]).unwrap();

    // The product rule has no physical-basis expansion, so the oracle here is
    // the inclusion isometry, applied with `compose` only.
    let iota_dagger = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        [selection.subspace()],
        [&leg],
        |trees, indices| {
            if indices[1] == start(&trees.codomain_uncoupled()[0]) + indices[0] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )
    .unwrap();
    let iota = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        [&leg],
        [selection.subspace()],
        |trees, indices| {
            if indices[0] == start(&trees.domain_uncoupled()[0]) + indices[1] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )
    .unwrap();
    let spectator_id =
        TensorMap::<_, Complex64>::isomorphism(&runtime, [&other], [&other]).unwrap();

    let expected = iota_dagger
        .otimes(&spectator_id)
        .unwrap()
        .compose(&source)
        .unwrap();
    assert_eq!(restricted.codomain(), expected.codomain());
    numerics::assert_slices_close(
        "entries",
        restricted.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        TERMS,
    );

    let embedded = restricted.embed_leg(0, &selection).unwrap();
    let expected = iota
        .otimes(&spectator_id)
        .unwrap()
        .compose(&restricted)
        .unwrap();
    assert_eq!(embedded.codomain(), expected.codomain());
    numerics::assert_slices_close(
        "entries",
        embedded.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        TERMS,
    );
}

#[test]
fn a_rank_five_restriction_still_gathers_one_axis_only() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 2), (1, 3)]);
    let small = u1(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&small, &leg, &small], [&small, &small], 61).unwrap();
    assert_eq!(source.rank(), 5);

    let selected = [(U1Irrep::new(1), 1..3)];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    let restricted = source.restrict_leg(&[(1, &selection)]).unwrap();

    let (layout, ranges) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        |_| 1,
        &selected,
    );
    let (small_layout, _) = dense_plan::<U1Irrep, Range<usize>>(
        &small.sectors().unwrap(),
        small.degeneracies(),
        small.is_dual(),
        |_| 1,
        &[],
    );
    let dense = source.to_physical_dense().unwrap();
    let maps = vec![
        identity_gather(&small_layout),
        axis_gather(&layout, &ranges),
        identity_gather(&small_layout),
        identity_gather(&small_layout),
        identity_gather(&small_layout),
    ];
    let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
    let actual = restricted.to_physical_dense().unwrap();
    assert_eq!(actual.shape, shape);
    numerics::assert_slices_close("physical entries", &actual.data, &expected, TERMS);
}

#[test]
fn a_selection_built_on_another_rule_is_a_rule_mismatch() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let two = Arc::new(ZNFusionRule::new(2).unwrap());
    let three = Arc::new(ZNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&two), [(two.irrep(0), 2), (two.irrep(1), 2)]).unwrap();
    let foreign = GradedSpace::try_new(
        Arc::clone(&three),
        [(three.irrep(0), 2), (three.irrep(1), 2)],
    )
    .unwrap();
    let source = TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg]).unwrap();
    let selection = LegSelection::try_new(&foreign, [(three.irrep(0), 0..1)]).unwrap();
    let error = source.restrict_leg(&[(0, &selection)]).unwrap_err();
    assert!(
        matches!(error, symtn::typed::Error::RuleMismatch),
        "expected a rule mismatch, got {error:?}"
    );
    assert!(matches!(
        source.embed_leg(0, &selection),
        Err(symtn::typed::Error::RuleMismatch)
    ));
}

#[test]
fn restriction_commutes_with_permute_and_with_a_contraction_over_an_untouched_leg() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let leg = fz2(&provider, &[(false, 2), (true, 3)]);
    let spectator = fz2(&provider, &[(false, 2), (true, 2)]);
    let bond = fz2(&provider, &[(false, 1), (true, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &spectator], [&bond], 67).unwrap();
    let partner: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&bond], [&spectator], 71).unwrap();
    let selection = LegSelection::try_new(&leg, [(Z2Irrep::ODD, 1..3)]).unwrap();

    // Swapping two codomain legs and restricting are independent.
    let restrict_then_permute = source
        .restrict_leg(&[(0, &selection)])
        .unwrap()
        .permute(&[1, 0], &[2])
        .unwrap();
    let permute_then_restrict = source
        .permute(&[1, 0], &[2])
        .unwrap()
        .restrict_leg(&[(1, &selection)])
        .unwrap();
    assert_eq!(
        restrict_then_permute.codomain(),
        permute_then_restrict.codomain()
    );
    numerics::assert_slices_close(
        "entries",
        restrict_then_permute.dense_data().unwrap(),
        permute_then_restrict.dense_data().unwrap(),
        TERMS,
    );

    // Contracting the untouched bond commutes with the restriction. Both
    // sides run the same `contract`, so the fermionic supertrace twist on the
    // dual contracted axis appears identically on each.
    let restrict_then_contract = source
        .restrict_leg(&[(0, &selection)])
        .unwrap()
        .contract(
            &partner,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    let contract_then_restrict = source
        .contract(
            &partner,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap()
        .restrict_leg(&[(0, &selection)])
        .unwrap();
    assert_eq!(
        restrict_then_contract.codomain(),
        contract_then_restrict.codomain()
    );
    numerics::assert_slices_close(
        "entries",
        restrict_then_contract.dense_data().unwrap(),
        contract_then_restrict.dense_data().unwrap(),
        TERMS,
    );
}

/// A deterministic, nonzero, finite entry per `(fusion-tree pair, degeneracy
/// index)`, so a block routed to the wrong tree or offset cannot coincide.
trait ExactEntry: Copy {
    const ZERO: Self;
    fn entry<S: std::hash::Hash>(trees: &BlockFusionTrees<S>, indices: &[usize]) -> Self;
    fn bits(self) -> (u64, u64);
}

fn hashed_nonzero<S: std::hash::Hash>(
    trees: &BlockFusionTrees<S>,
    indices: &[usize],
    salt: u8,
) -> f64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (trees, indices, salt).hash(&mut hasher);
    let bits = hasher.finish();
    let magnitude = 0.5 + (bits >> 11) as f64 / (1u64 << 53) as f64;
    let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
    sign * magnitude * f64::powi(2.0, ((bits >> 1) % 16) as i32 - 8)
}

impl ExactEntry for f64 {
    const ZERO: Self = 0.0;
    fn entry<S: std::hash::Hash>(trees: &BlockFusionTrees<S>, indices: &[usize]) -> Self {
        hashed_nonzero(trees, indices, 0)
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl ExactEntry for f32 {
    const ZERO: Self = 0.0;
    fn entry<S: std::hash::Hash>(trees: &BlockFusionTrees<S>, indices: &[usize]) -> Self {
        hashed_nonzero(trees, indices, 0) as f32
    }
    fn bits(self) -> (u64, u64) {
        (u64::from(self.to_bits()), 0)
    }
}

impl ExactEntry for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
    fn entry<S: std::hash::Hash>(trees: &BlockFusionTrees<S>, indices: &[usize]) -> Self {
        Complex64::new(
            hashed_nonzero(trees, indices, 0),
            hashed_nonzero(trees, indices, 1),
        )
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

fn exact_bits<D: ExactEntry>(data: &[D]) -> Vec<(u64, u64)> {
    data.iter().map(|value| value.bits()).collect()
}

/// The selected positions of the sector the block carries on `axis`.
fn selected_positions<S: PartialEq>(
    trees: &BlockFusionTrees<S>,
    axis: usize,
    codomain_rank: usize,
    selected: &[(S, Vec<usize>)],
) -> Option<Vec<usize>> {
    let sector = if axis < codomain_rank {
        &trees.codomain_uncoupled()[axis]
    } else {
        &trees.domain_uncoupled()[axis - codomain_rank]
    };
    selected
        .iter()
        .find(|(candidate, _)| candidate == sector)
        .map(|(_, range)| range.clone())
}

/// Where a selection sits inside one sector of degeneracy at least three.
#[derive(Clone, Copy)]
enum Place {
    Start,
    Middle,
    End,
    Single,
    /// The first and the last position: the largest `|λ|` of an ascending
    /// signed `eigh` spectrum, two runs (#2095).
    Ends,
    /// Every other position from the first: an arbitrary `eig` set, one run
    /// per kept position (#2095).
    Alternate,
}

impl Place {
    fn positions(self, degeneracy: usize) -> Vec<usize> {
        assert!(degeneracy >= 3);
        match self {
            Place::Start => (0..2).collect(),
            Place::Middle => (1..degeneracy - 1).collect(),
            Place::End => vec![degeneracy - 1],
            Place::Single => vec![1],
            Place::Ends => vec![0, degeneracy - 1],
            Place::Alternate => (0..degeneracy).step_by(2).collect(),
        }
    }
}

/// `restrict_leg`, `embed_leg` and their round trip, each against a reduced
/// block oracle filled independently with `from_subblock_fn`: restriction is a
/// pure copy, so every entry must match bit for bit.
macro_rules! assert_exact_restrict_embed {
    ($dtype:ty, [$($codomain:expr),+], [$($domain:expr),+], $axis:expr, $selected:expr) => {{
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let codomain = vec![$($codomain),+];
        let domain = vec![$($domain),+];
        let axis: usize = $axis;
        let rank = codomain.len();
        let parent = if axis < rank { codomain[axis] } else { domain[axis - rank] };
        // Positions in the parent's own stored order, so a dual leg is
        // selected in its own labels.
        let selected: Vec<_> = parent
            .sectors()
            .unwrap()
            .into_iter()
            .zip(parent.degeneracies())
            .zip($selected)
            .filter_map(|((sector, &degeneracy), place)| {
                place.map(|place: Place| (sector, place.positions(degeneracy)))
            })
            .collect();
        let selection = LegSelection::try_new(parent, selected.iter().cloned()).unwrap();
        let replaced = |legs: &[&GradedSpace<_>], base: usize| {
            legs.iter()
                .enumerate()
                .map(|(offset, &leg)| {
                    if base + offset == axis {
                        selection.subspace().clone()
                    } else {
                        leg.clone()
                    }
                })
                .collect::<Vec<_>>()
        };
        let sub_codomain = replaced(&codomain, 0);
        let sub_domain = replaced(&domain, rank);

        let source = TensorMap::<_, $dtype>::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |trees, indices| <$dtype as ExactEntry>::entry(trees, indices),
        )
        .unwrap();
        let expected = TensorMap::<_, $dtype>::from_subblock_fn(
            &runtime,
            &sub_codomain,
            &sub_domain,
            |trees, indices| {
                let positions = selected_positions(trees, axis, rank, &selected).unwrap();
                let mut shifted = indices.to_vec();
                shifted[axis] = positions[indices[axis]];
                <$dtype as ExactEntry>::entry(trees, &shifted)
            },
        )
        .unwrap();
        let restricted = source.restrict_leg(&[(axis, &selection)]).unwrap();
        assert_eq!(restricted.codomain(), expected.codomain());
        assert_eq!(restricted.domain(), expected.domain());
        assert_eq!(exact_bits(restricted.dense_data().unwrap()), exact_bits(expected.dense_data().unwrap()));

        let embedded = restricted.embed_leg(axis, &selection).unwrap();
        let projected = TensorMap::<_, $dtype>::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |trees, indices| match selected_positions(trees, axis, rank, &selected) {
                Some(positions) if positions.contains(&indices[axis]) => {
                    <$dtype as ExactEntry>::entry(trees, indices)
                }
                _ => <$dtype as ExactEntry>::ZERO,
            },
        )
        .unwrap();
        assert_eq!(embedded.codomain(), source.codomain());
        assert_eq!(exact_bits(embedded.dense_data().unwrap()), exact_bits(projected.dense_data().unwrap()));
        assert_eq!(
            exact_bits(embedded.restrict_leg(&[(axis, &selection)]).unwrap().dense_data().unwrap()),
            exact_bits(restricted.dense_data().unwrap())
        );
    }};
}

#[test]
fn restrict_and_embed_are_exact_block_copies_on_u1_legs() {
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 3), (0, 4), (1, 5)]);
    let dual = leg.try_dual().unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 3)]);
    // Ranges at the start, in the middle and at the end of a sector, plus a
    // single-state selection.
    let ranges = [Some(Place::Start), Some(Place::Middle), Some(Place::End)];
    let single = [None, Some(Place::Single), None];
    // Non-contiguous positions: two runs and one run per kept position.
    let gaps = [
        Some(Place::Alternate),
        Some(Place::Ends),
        Some(Place::Alternate),
    ];
    for selected in [ranges, single, gaps] {
        assert_exact_restrict_embed!(f64, [&leg, &other], [&other], 0, selected);
        assert_exact_restrict_embed!(Complex64, [&other, &dual], [&other], 1, selected);
        assert_exact_restrict_embed!(f64, [&other], [&other, &dual], 2, selected);
        assert_exact_restrict_embed!(Complex64, [&other], [&leg, &other], 1, selected);
        // Single precision, on the middle leg of a rank-four map.
        assert_exact_restrict_embed!(f32, [&other, &leg, &other], [&other], 1, selected);
    }
}

#[test]
fn restrict_and_embed_are_exact_block_copies_on_su2_legs() {
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 3), (1, 4), (2, 3)]);
    let other = su2(&provider, &[(0, 1), (1, 2), (2, 1)]);
    let ranges = [Some(Place::End), Some(Place::Middle), Some(Place::Start)];
    let single = [None, Some(Place::Single), None];
    // Non-contiguous positions: two runs and one run per kept position.
    let gaps = [
        Some(Place::Alternate),
        Some(Place::Ends),
        Some(Place::Alternate),
    ];
    for selected in [ranges, single, gaps] {
        assert_exact_restrict_embed!(f64, [&leg, &other], [&other], 0, selected);
        assert_exact_restrict_embed!(Complex64, [&other, &other], [&leg], 2, selected);
        assert_exact_restrict_embed!(Complex64, [&other, &leg], [&other], 1, selected);
        assert_exact_restrict_embed!(f64, [&other], [&other, &leg], 2, selected);
    }
}

#[test]
fn restrict_and_embed_are_exact_block_copies_on_fz2_u1_legs() {
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 3),
            (label(true, 1), 4),
            (label(false, 2), 3),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let other = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 2),
            (label(true, 1), 1),
            (label(true, -1), 2),
        ],
    )
    .unwrap();
    let ranges = [Some(Place::Start), Some(Place::Middle), Some(Place::End)];
    let single = [None, Some(Place::Single), None];
    // Non-contiguous positions: two runs and one run per kept position.
    let gaps = [
        Some(Place::Alternate),
        Some(Place::Ends),
        Some(Place::Alternate),
    ];
    for selected in [ranges, single, gaps] {
        assert_exact_restrict_embed!(Complex64, [&dual, &other], [&other], 0, selected);
        assert_exact_restrict_embed!(f64, [&other, &leg], [&other], 1, selected);
        assert_exact_restrict_embed!(f64, [&other], [&other, &dual], 2, selected);
        assert_exact_restrict_embed!(Complex64, [&other], [&leg, &other], 1, selected);
    }
}

/// A multi-axis `restrict_leg` against the same independent reduced-block
/// oracle: every restricted axis is shifted by its own selection's start
/// (#1561). Sequential one-axis restrictions, in the given order and
/// reversed, must give the same bits.
macro_rules! assert_exact_multi_restrict {
    ($dtype:ty, [$($codomain:expr),+], [$($domain:expr),*], [$(($axis:expr, $places:expr)),+]) => {{
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let codomain = vec![$($codomain),+];
        let domain: Vec<&GradedSpace<_>> = vec![$($domain),*];
        let rank = codomain.len();
        let parent_of = |axis: usize| if axis < rank { codomain[axis] } else { domain[axis - rank] };
        let mut selected = Vec::new();
        let mut selections = Vec::new();
        $(
            let parent = parent_of($axis);
            let kept: Vec<_> = parent
                .sectors()
                .unwrap()
                .into_iter()
                .zip(parent.degeneracies())
                .zip($places)
                .filter_map(|((sector, &degeneracy), place)| {
                    place.map(|place: Place| (sector, place.positions(degeneracy)))
                })
                .collect();
            selections.push(($axis, LegSelection::try_new(parent, kept.iter().cloned()).unwrap()));
            selected.push(($axis, kept));
        )+
        let set: Vec<(usize, &LegSelection<_>)> =
            selections.iter().map(|(axis, selection)| (*axis, selection)).collect();
        let replaced = |legs: &[&GradedSpace<_>], base: usize| {
            legs.iter()
                .enumerate()
                .map(|(offset, &leg)| {
                    selections
                        .iter()
                        .find(|(axis, _)| *axis == base + offset)
                        .map_or_else(|| leg.clone(), |(_, selection)| selection.subspace().clone())
                })
                .collect::<Vec<_>>()
        };
        let sub_codomain = replaced(&codomain, 0);
        let sub_domain = replaced(&domain, rank);
        let source = TensorMap::<_, $dtype>::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |trees, indices| <$dtype as ExactEntry>::entry(trees, indices),
        )
        .unwrap();
        let expected = TensorMap::<_, $dtype>::from_subblock_fn(
            &runtime,
            &sub_codomain,
            &sub_domain,
            |trees, indices| {
                let mut shifted = indices.to_vec();
                for (axis, kept) in &selected {
                    shifted[*axis] =
                        selected_positions(trees, *axis, rank, kept).unwrap()[indices[*axis]];
                }
                <$dtype as ExactEntry>::entry(trees, &shifted)
            },
        )
        .unwrap();
        let restricted = source.restrict_leg(&set).unwrap();
        assert_eq!(restricted.codomain(), expected.codomain());
        assert_eq!(restricted.domain(), expected.domain());
        assert_eq!(exact_bits(restricted.dense_data().unwrap()), exact_bits(expected.dense_data().unwrap()));
        for order in [set.clone(), set.iter().rev().copied().collect()] {
            let sequential = order.iter().fold(source.clone(), |tensor, &pair| {
                tensor.restrict_leg(&[pair]).unwrap()
            });
            assert_eq!(exact_bits(sequential.dense_data().unwrap()), exact_bits(restricted.dense_data().unwrap()));
        }
    }};
}

#[test]
fn multi_axis_restrict_is_one_exact_block_copy_on_u1_su2_and_fz2_u1_legs() {
    let ranges = [Some(Place::Start), Some(Place::Middle), Some(Place::End)];
    let other_ranges = [Some(Place::End), None, Some(Place::Middle)];
    let single = [None, Some(Place::Single), None];
    let gaps = [
        Some(Place::Alternate),
        Some(Place::Ends),
        Some(Place::Alternate),
    ];
    let other_gaps = [Some(Place::Ends), None, Some(Place::Alternate)];

    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 3), (0, 4), (1, 5)]);
    let dual = leg.try_dual().unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 3)]);
    assert_exact_multi_restrict!(f64, [&leg, &dual], [&leg], [(0, ranges), (2, other_ranges)]);
    assert_exact_multi_restrict!(
        Complex64,
        [&leg, &dual],
        [&leg, &dual],
        [(1, gaps), (3, other_gaps), (0, ranges)]
    );
    assert_exact_multi_restrict!(
        Complex64,
        [&dual, &other],
        [&leg, &dual],
        [(3, single), (0, ranges), (2, other_ranges)]
    );
    assert_exact_multi_restrict!(f64, [&leg], [&leg], [(1, ranges), (0, ranges)]);

    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 3), (1, 4), (2, 3)]);
    let other = su2(&provider, &[(0, 1), (1, 2), (2, 1)]);
    assert_exact_multi_restrict!(
        Complex64,
        [&leg, &other],
        [&leg],
        [(2, single), (0, ranges)]
    );
    assert_exact_multi_restrict!(
        f64,
        [&leg, &leg],
        [&leg, &other],
        [(0, ranges), (1, other_ranges), (2, single)]
    );
    assert_exact_multi_restrict!(
        Complex64,
        [&other, &leg],
        [&leg],
        [(1, gaps), (2, other_gaps)]
    );

    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 3),
            (label(true, 1), 4),
            (label(false, 2), 3),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let other = GradedSpace::try_new(
        Arc::clone(&provider),
        [(label(false, 0), 2), (label(true, 1), 1)],
    )
    .unwrap();
    assert_exact_multi_restrict!(
        Complex64,
        [&dual, &other],
        [&leg],
        [(2, other_ranges), (0, ranges)]
    );
    assert_exact_multi_restrict!(f64, [&leg], [&other, &dual], [(0, single), (2, ranges)]);
    assert_exact_multi_restrict!(f64, [&other, &leg], [&dual], [(1, gaps), (2, other_gaps)]);
}

/// A compact diagonal restricted on both legs with one selection stays
/// compact (TensorKit `truncate_diagonal!`): its values are the hand slice of
/// the source diagonal, and its dense form equals the dense restriction of
/// the materialized source, bit for bit.
macro_rules! assert_compact_stays_compact {
    ($dtype:ty, $leg:expr, $places:expr) => {{
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let leg = $leg;
        let source =
            TensorMap::<_, $dtype>::from_subblock_fn(&runtime, [&leg], [&leg], |trees, indices| {
                <$dtype as ExactEntry>::entry(trees, indices)
            })
            .unwrap();
        let s = source.svd_compact(&[0], &[1]).unwrap().s;
        let bond = s.domain()[0].clone();
        let kept: Vec<_> = bond
            .sectors()
            .unwrap()
            .into_iter()
            .zip(bond.degeneracies())
            .zip($places)
            .filter_map(|((sector, &degeneracy), place)| {
                place.map(|place: Place| (sector, place.positions(degeneracy)))
            })
            .collect();
        let selection = LegSelection::try_new(&bond, kept.iter().cloned()).unwrap();
        let expected: Vec<_> = s
            .diagview()
            .unwrap()
            .into_iter()
            .filter_map(|entry| {
                kept.iter()
                    .find(|(sector, _)| *sector == entry.sector)
                    .map(|(_, positions)| {
                        let values = positions.iter().map(|&position| entry.values[position]);
                        (entry.sector.clone(), values.collect::<Vec<_>>())
                    })
            })
            .collect();
        for set in [
            [(0, &selection), (1, &selection)],
            [(1, &selection), (0, &selection)],
        ] {
            let restricted = s.restrict_leg(&set).unwrap();
            let spectrum = symtn::expert::diagonal_spectrum(&restricted)
                .unwrap()
                .expect("a compact input restricted on both legs stays compact");
            let got: Vec<_> = spectrum
                .into_iter()
                .map(|entry| (entry.sector, entry.values))
                .collect();
            assert_eq!(got.len(), expected.len());
            for ((sector, values), (want_sector, want)) in got.iter().zip(&expected) {
                assert_eq!(sector, want_sector);
                assert_eq!(exact_bits(values), exact_bits(want));
            }
            let dense = s.materialize().unwrap().restrict_leg(&set).unwrap();
            assert!(symtn::expert::diagonal_spectrum(&dense).unwrap().is_none());
            assert_eq!(restricted.codomain(), dense.codomain());
            assert_eq!(restricted.domain(), dense.domain());
            assert_eq!(
                exact_bits(restricted.materialize().unwrap().dense_data().unwrap()),
                exact_bits(dense.dense_data().unwrap())
            );
        }
    }};
}

#[test]
fn a_compact_diagonal_restricted_on_both_legs_with_one_selection_stays_compact() {
    let places = [Some(Place::Start), Some(Place::Middle), None];
    let gaps = [Some(Place::Ends), Some(Place::Alternate), None];
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 3), (0, 4), (1, 5)]);
    assert_compact_stays_compact!(f64, leg.clone(), places);
    assert_compact_stays_compact!(f64, leg.clone(), gaps);
    assert_compact_stays_compact!(Complex64, leg.try_dual().unwrap(), places);

    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 3), (1, 4), (2, 3)]);
    assert_compact_stays_compact!(Complex64, leg.clone(), places);
    assert_compact_stays_compact!(Complex64, leg.clone(), gaps);
    assert_compact_stays_compact!(f64, leg, [None, Some(Place::Single), Some(Place::End)]);

    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 3),
            (label(true, 1), 4),
            (label(false, 2), 3),
        ],
    )
    .unwrap();
    assert_compact_stays_compact!(f64, leg.try_dual().unwrap(), places);
    assert_compact_stays_compact!(Complex64, leg.clone(), places);
    assert_compact_stays_compact!(f64, leg, gaps);
}

/// #2095: non-contiguous kept positions on a middle SU(2) leg of a rank-four
/// map and on the same leg of its lazy adjoint, against the dense gather of
/// the physical expansion (multiplets kept whole, physical offsets
/// recomputed), and `embed_leg` against the dense scatter into zeros.
#[test]
fn noncontiguous_positions_match_the_dense_gather_and_scatter_on_su2_and_a_lazy_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 4), (1, 5), (2, 3)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    let source: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&other, &leg, &other], [&other], 23).unwrap();
    let selected = [
        (SU2Irrep::from_twice_spin(0), vec![0, 3]),
        (SU2Irrep::from_twice_spin(1), vec![1, 2, 4]),
        (SU2Irrep::from_twice_spin(2), vec![0, 2]),
    ];
    let selection = LegSelection::try_new(&leg, selected.iter().cloned()).unwrap();
    assert_eq!(selection.subspace().degeneracies(), &[2, 3, 2]);
    let carrier = |sector: &SU2Irrep| sector.twice_spin() + 1;
    let (layout, kept) = dense_plan(
        &leg.sectors().unwrap(),
        leg.degeneracies(),
        leg.is_dual(),
        carrier,
        &selected,
    );
    let (other_layout, _) = dense_plan::<SU2Irrep, Vec<usize>>(
        &other.sectors().unwrap(),
        other.degeneracies(),
        other.is_dual(),
        carrier,
        &[],
    );
    let zero = Complex64::new(0.0, 0.0);
    let check = |tensor: &TensorMap<_, Complex64>, axis: usize| {
        let restricted = tensor.restrict_leg(&[(axis, &selection)]).unwrap();
        let dense = tensor.to_physical_dense().unwrap();
        let maps: Vec<Vec<usize>> = (0..dense.shape.len())
            .map(|candidate| {
                if candidate == axis {
                    axis_gather(&layout, &kept)
                } else {
                    identity_gather(&other_layout)
                }
            })
            .collect();
        let (shape, expected) = gather(&dense.shape, &dense.data, &maps);
        let actual = restricted.to_physical_dense().unwrap();
        assert_eq!(actual.shape, shape);
        numerics::assert_slices_close("restricted entries", &actual.data, &expected, TERMS);

        let embedded = restricted.embed_leg(axis, &selection).unwrap();
        let expected = scattered(&dense.shape, &actual.data, &maps, zero);
        let actual = embedded.to_physical_dense().unwrap();
        assert_eq!(actual.shape, dense.shape);
        numerics::assert_slices_close("embedded entries", &actual.data, &expected, TERMS);
    };
    check(&source, 1);
    let lazy = source.adjoint().unwrap();
    check(&lazy, lazy.codomain().len() + 1);
}

/// #2095: kept positions are a set. Any spelling of one set is the same
/// selection (canonical runs), `positions` reports it ascending, and
/// unsorted, repeated or out-of-range positions are rejected (stricter than
/// TensorKit's `view(b, :, I)`, which would permute or duplicate columns).
#[test]
fn kept_positions_are_a_canonical_strictly_increasing_set() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(0, 5), (1, 3), (2, 2)]);
    let zero = U1Irrep::new(0);
    let one = U1Irrep::new(1);

    let by_range = LegSelection::try_new(&leg, [(zero, 0..2), (one, 1..3)]).unwrap();
    let by_list = LegSelection::try_new(&leg, [(one, vec![1, 2]), (zero, vec![0, 1])]).unwrap();
    assert_eq!(by_range.subspace(), by_list.subspace());
    assert_eq!(by_list.positions(&zero).unwrap(), vec![0, 1]);
    assert_eq!(by_list.positions(&one).unwrap(), vec![1, 2]);
    // A parent sector that is not selected keeps nothing; a sector absent
    // from the parent is an error, not an empty set.
    assert!(by_list.positions(&U1Irrep::new(2)).unwrap().is_empty());
    assert!(by_list.positions(&U1Irrep::new(7)).is_err());

    let gaps = LegSelection::try_new(&leg, [(zero, vec![0, 2, 3]), (one, vec![0, 2])]).unwrap();
    assert_eq!(gaps.positions(&zero).unwrap(), vec![0, 2, 3]);
    assert_eq!(gaps.subspace().degeneracies(), &[3, 2]);
    assert!(!gaps.is_full());
    let full = LegSelection::try_new(
        &leg,
        [
            (zero, vec![0, 1, 2, 3, 4]),
            (one, vec![0, 1, 2]),
            (U1Irrep::new(2), vec![0, 1]),
        ],
    )
    .unwrap();
    assert!(full.is_full());

    // Equal sets spelled differently are one selection: a compact diagonal
    // restricted with both stays compact, which needs equal tables.
    let source: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 5).unwrap();
    let s = source.svd_compact(&[0], &[1]).unwrap().s;
    let bond = s.domain()[0].clone();
    let split = LegSelection::try_new(&bond, [(zero, vec![1, 2, 3])]).unwrap();
    let joined = LegSelection::try_new(&bond, [(zero, 1..4)]).unwrap();
    let restricted = s.restrict_leg(&[(0, &split), (1, &joined)]).unwrap();
    assert!(symtn::expert::diagonal_spectrum(&restricted)
        .unwrap()
        .is_some());

    for invalid in [vec![2, 1], vec![1, 1], vec![0, 5], vec![]] {
        let error = LegSelection::try_new(&leg, [(zero, invalid.clone())]).unwrap_err();
        assert!(
            matches!(error, symtn::typed::Error::InvalidArgument(ref m) if m.contains(match invalid.as_slice() {
                [2, 1] | [1, 1] => "strictly increasing",
                [0, 5] => "exceeds its",
                [] => "at least one position",
                _ => unreachable!(),
            })),
            "{invalid:?}: {error:?}"
        );
    }
}
