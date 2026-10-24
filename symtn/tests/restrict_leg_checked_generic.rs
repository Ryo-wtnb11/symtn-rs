//! `restrict_leg` / `embed_leg` on a checked Generic provider (#1285).
//!
//! SU(3) with the adjoint irrep gives fusion trees that differ only by their
//! outer-multiplicity vertex, which is the part of a block key a
//! degeneracy-only operation must leave alone. `otimes`/`id` are
//! multiplicity-free only, so the oracle here is a literal per-block gather
//! driven by the public fusion-tree keys and block geometry, which shares no
//! code with the restriction kernel.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::sector::LieGroupFusionRule;
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, LegSelection, TensorMap};

#[test]
fn restrict_and_embed_preserve_su3_multiplicity_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![2i64, 2];
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 3)]).unwrap();
    let spectator = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&leg, &spectator, &spectator],
        [],
        |trees, indices| {
            (100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>()) as f64
        },
    )
    .unwrap();
    assert!(
        (0..source.subblock_count()).any(|index| source
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "fixture must carry a Generic vertex key mu = 2"
    );

    let selection = LegSelection::try_new(&leg, [(adjoint.clone(), 1..3)]).unwrap();
    let restricted = source.restrict_leg(&[(0, &selection)]).unwrap();
    assert_eq!(restricted.codomain()[0], *selection.subspace());
    assert_eq!(restricted.subblock_count(), source.subblock_count());

    // Oracle: a literal per-block gather driven by the public fusion-tree
    // keys and block geometry, sharing no code with the restriction kernel.
    let expected = literal_slice(&source, &restricted, &[(0, &[1, 2])]);
    assert_eq!(restricted.dense_data().unwrap(), expected);

    // Embedding back writes the same rectangle into a zero payload and leaves
    // every vertex key in place.
    let embedded = restricted.embed_leg(0, &selection).unwrap();
    assert_eq!(embedded.codomain()[0], leg);
    let expected = literal_scatter(&restricted, &embedded, 0, &[1, 2]);
    assert_eq!(embedded.dense_data().unwrap(), expected);
    assert_eq!(
        embedded
            .restrict_leg(&[(0, &selection)])
            .unwrap()
            .dense_data()
            .unwrap(),
        restricted.dense_data().unwrap()
    );
}

#[test]
fn a_multi_axis_restriction_is_the_literal_slice_on_every_axis_at_once() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![2i64, 2];
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 3)]).unwrap();
    let spectator = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &spectator, &leg], [], |trees, indices| {
            (1000 * trees.codomain_vertices()[0].get()
                + 100 * indices[0]
                + 10 * indices[1]
                + indices[2]) as f64
        })
        .unwrap();
    let middle = LegSelection::try_new(&leg, [(adjoint.clone(), 1..3)]).unwrap();
    let head = LegSelection::try_new(&leg, [(adjoint.clone(), 0..2)]).unwrap();
    let set = [(2, &head), (0, &middle)];
    let restricted = source.restrict_leg(&set).unwrap();
    assert_eq!(restricted.codomain()[0], *middle.subspace());
    assert_eq!(restricted.codomain()[2], *head.subspace());
    assert_eq!(restricted.subblock_count(), source.subblock_count());
    assert_eq!(
        restricted.dense_data().unwrap(),
        literal_slice(&source, &restricted, &[(0, &[1, 2]), (2, &[0, 1])])
    );
    let sequential = source
        .restrict_leg(&[(2, &head)])
        .unwrap()
        .restrict_leg(&[(0, &middle)])
        .unwrap();
    assert_eq!(
        restricted
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        sequential
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_compact_eigh_spectrum_restricted_on_both_legs_stays_compact() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(vec![0i64, 0], 3), (vec![2i64, 2], 4)],
    )
    .unwrap();
    let mut state = 0x5eed_u64;
    let raw: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
    .unwrap();
    let hermitian = raw.axpby(1.0, &raw.adjoint().unwrap(), 1.0).unwrap();
    let d = hermitian
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .d;
    assert!(symtn::expert::diagonal_spectrum(&d).unwrap().is_some());
    let bond = d.domain()[0].clone();
    let kept = [(vec![0i64, 0], 1..3), (vec![2i64, 2], 0..1)];
    let selection = LegSelection::try_new(&bond, kept.iter().cloned()).unwrap();
    let restricted = d.restrict_leg(&[(1, &selection), (0, &selection)]).unwrap();
    let got = symtn::expert::diagonal_spectrum(&restricted)
        .unwrap()
        .expect("one selection on both legs keeps a compact payload compact");
    let source = d.diagview().unwrap();
    assert_eq!(got.len(), kept.len());
    for entry in got {
        let (_, range) = kept
            .iter()
            .find(|(sector, _)| *sector == entry.sector)
            .unwrap();
        let from = source
            .iter()
            .find(|candidate| candidate.sector == entry.sector)
            .unwrap();
        assert_eq!(entry.values, from.values[range.clone()].to_vec());
    }
    assert_eq!(
        restricted.materialize().unwrap().dense_data().unwrap(),
        d.materialize()
            .unwrap()
            .restrict_leg(&[(0, &selection), (1, &selection)])
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

/// #2095: non-contiguous kept positions on two SU(3) legs whose trees differ
/// by their outer-multiplicity vertex: the literal per-block gather and
/// scatter, and a compact `eigh` spectrum restricted to a two-run set.
#[test]
fn noncontiguous_positions_are_the_literal_gather_with_su3_multiplicity_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![2i64, 2];
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 4)]).unwrap();
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 3)]).unwrap();
    let spectator = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide, &spectator, &leg], [], |trees, indices| {
            (1000 * trees.codomain_vertices()[0].get()
                + 100 * indices[0]
                + 10 * indices[1]
                + indices[2]) as f64
        })
        .unwrap();
    let ends = LegSelection::try_new(&wide, [(adjoint.clone(), vec![0, 3])]).unwrap();
    let outer = LegSelection::try_new(&leg, [(adjoint.clone(), vec![0, 2])]).unwrap();
    assert_eq!(ends.positions(&adjoint).unwrap(), vec![0, 3]);
    let restricted = source.restrict_leg(&[(2, &outer), (0, &ends)]).unwrap();
    assert_eq!(restricted.subblock_count(), source.subblock_count());
    assert_eq!(
        restricted.dense_data().unwrap(),
        literal_slice(&source, &restricted, &[(0, &[0, 3]), (2, &[0, 2])])
    );

    let single = source.restrict_leg(&[(0, &ends)]).unwrap();
    let embedded = single.embed_leg(0, &ends).unwrap();
    assert_eq!(embedded.codomain()[0], wide);
    assert_eq!(
        embedded.dense_data().unwrap(),
        literal_scatter(&single, &embedded, 0, &[0, 3])
    );

    let mut state = 0x5eed_u64;
    let square = GradedSpace::try_new(
        Arc::clone(&provider),
        [(vec![0i64, 0], 3), (adjoint.clone(), 4)],
    )
    .unwrap();
    let raw: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&square], [&square], |_, _| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .unwrap();
    let hermitian = raw.axpby(1.0, &raw.adjoint().unwrap(), 1.0).unwrap();
    let d = hermitian
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .d;
    let bond = d.domain()[0].clone();
    let kept = [
        (vec![0i64, 0], vec![0, 2]),
        (adjoint.clone(), vec![0, 1, 3]),
    ];
    let selection = LegSelection::try_new(&bond, kept.iter().cloned()).unwrap();
    let restricted = d.restrict_leg(&[(0, &selection), (1, &selection)]).unwrap();
    let got = symtn::expert::diagonal_spectrum(&restricted)
        .unwrap()
        .expect("one selection on both legs keeps a compact payload compact");
    let source = d.diagview().unwrap();
    for entry in got {
        let (_, positions) = kept
            .iter()
            .find(|(sector, _)| *sector == entry.sector)
            .unwrap();
        let from = source
            .iter()
            .find(|candidate| candidate.sector == entry.sector)
            .unwrap();
        let expected: Vec<f64> = positions.iter().map(|&p| from.values[p]).collect();
        assert_eq!(entry.values, expected);
    }
}

/// The payload `destination` must hold when each of its blocks is the
/// gather of `source`'s block with the same fusion trees at the kept
/// `positions` on each listed `(axis, positions)`.
fn literal_slice(
    source: &TensorMap<LieGroupFusionRule, f64>,
    destination: &TensorMap<LieGroupFusionRule, f64>,
    kept: &[(usize, &[usize])],
) -> Vec<f64> {
    let mut payload = vec![f64::NAN; destination.dense_data().unwrap().len()];
    for index in 0..destination.subblock_count() {
        let trees = destination.subblock_fusion_trees(index).unwrap();
        let block = destination.subblock(index).unwrap();
        let matched = (0..source.subblock_count())
            .find(|&candidate| source.subblock_fusion_trees(candidate).unwrap() == trees)
            .expect("every destination tree pair survives from the source");
        let from = source.subblock(matched).unwrap();
        for_each_index(block.shape(), |position| {
            let to_offset: usize = block.offset()
                + position
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            let from_offset: usize = from.offset()
                + position
                    .iter()
                    .enumerate()
                    .zip(from.strides())
                    .map(|((dimension, &index), &stride)| {
                        let position = kept
                            .iter()
                            .find(|&&(axis, _)| axis == dimension)
                            .map_or(index, |&(_, positions)| positions[index]);
                        position * stride
                    })
                    .sum::<usize>();
            payload[to_offset] = source.dense_data().unwrap()[from_offset];
        });
    }
    payload
}

/// The payload `destination` must hold when each `source` block is written
/// into the kept `positions` on `axis` and everything else stays zero.
fn literal_scatter(
    source: &TensorMap<LieGroupFusionRule, f64>,
    destination: &TensorMap<LieGroupFusionRule, f64>,
    axis: usize,
    positions: &[usize],
) -> Vec<f64> {
    let mut payload = vec![0.0; destination.dense_data().unwrap().len()];
    for index in 0..source.subblock_count() {
        let trees = source.subblock_fusion_trees(index).unwrap();
        let block = source.subblock(index).unwrap();
        let matched = (0..destination.subblock_count())
            .find(|&candidate| destination.subblock_fusion_trees(candidate).unwrap() == trees)
            .expect("every source tree pair exists in the larger space");
        let into = destination.subblock(matched).unwrap();
        for_each_index(block.shape(), |position| {
            let from_offset: usize = block.offset()
                + position
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            let to_offset: usize = into.offset()
                + position
                    .iter()
                    .enumerate()
                    .zip(into.strides())
                    .map(|((dimension, &index), &stride)| {
                        (if dimension == axis {
                            positions[index]
                        } else {
                            index
                        }) * stride
                    })
                    .sum::<usize>();
            payload[to_offset] = source.dense_data().unwrap()[from_offset];
        });
    }
    payload
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.contains(&0) {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    loop {
        visit(&index);
        let mut axis = 0;
        loop {
            if axis == shape.len() {
                return;
            }
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
        }
    }
}
