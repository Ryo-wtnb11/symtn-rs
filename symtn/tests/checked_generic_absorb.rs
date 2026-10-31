#![cfg(feature = "cgc-gen")]

use std::fmt::Debug;
use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::LieGroupFusionRule;
use symtn::typed::{Complex64, Error, Runtime};
use symtn::typed::{GradedSpace, TensorMap};

// Why not fewer args: this is a recursive N-dimensional shape-copy helper;
// grouping the paired source/destination offset+shape+stride triples into a
// struct would add an abstraction with one call site for no clarity gain.
#[allow(clippy::too_many_arguments)]
fn copy_prefix<D: Copy>(
    destination: &mut [D],
    destination_offset: usize,
    destination_shape: &[usize],
    destination_strides: &[usize],
    source: &[D],
    source_offset: usize,
    source_shape: &[usize],
    source_strides: &[usize],
    axis: usize,
    destination_index: usize,
    source_index: usize,
) {
    if axis == destination_shape.len() {
        destination[destination_offset + destination_index] = source[source_offset + source_index];
        return;
    }
    for index in 0..destination_shape[axis].min(source_shape[axis]) {
        copy_prefix(
            destination,
            destination_offset,
            destination_shape,
            destination_strides,
            source,
            source_offset,
            source_shape,
            source_strides,
            axis + 1,
            destination_index + index * destination_strides[axis],
            source_index + index * source_strides[axis],
        );
    }
}

fn assert_absorb_case<D>(
    engine: &mut dyn Engine,
    n: usize,
    label: Vec<i64>,
    value: impl Fn(usize) -> D,
) where
    D: Copy + Debug + PartialEq + symtn::typed::TensorScalar,
{
    let runtime = Runtime::builder().build().unwrap();
    let destination_provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let source_provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let destination_legs = [3, 4, 2].map(|degeneracy| {
        GradedSpace::try_new(
            Arc::clone(&destination_provider),
            [(label.clone(), degeneracy)],
        )
        .unwrap()
    });
    let source_legs = [2, 1, 3].map(|degeneracy| {
        GradedSpace::try_new(Arc::clone(&source_provider), [(label.clone(), degeneracy)]).unwrap()
    });
    let destination: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [
            &destination_legs[0],
            &destination_legs[1],
            &destination_legs[2],
        ],
        [],
        |trees, indices| {
            value(10_000 + 100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>())
        },
    )
    .unwrap();
    let source: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&source_legs[0], &source_legs[1], &source_legs[2]],
        [],
        |trees, indices| {
            value(20_000 + 100 * trees.codomain_vertices()[0].get() + indices.iter().sum::<usize>())
        },
    )
    .unwrap();

    let destination_before = destination.dense_data().unwrap().to_vec();
    let mut expected = destination_before.clone();
    for destination_index in 0..destination.subblock_count() {
        let destination_trees = destination
            .subblock_fusion_trees(destination_index)
            .unwrap();
        let source_index = (0..source.subblock_count())
            .find(|&index| source.subblock_fusion_trees(index).unwrap() == destination_trees)
            .unwrap();
        let destination_block = destination.subblock(destination_index).unwrap();
        let source_block = source.subblock(source_index).unwrap();
        copy_prefix(
            &mut expected,
            destination_block.offset(),
            destination_block.shape(),
            destination_block.strides(),
            source.dense_data().unwrap(),
            source_block.offset(),
            source_block.shape(),
            source_block.strides(),
            0,
            0,
            0,
        );
    }
    assert!(
        (0..destination.subblock_count()).any(|index| {
            destination
                .subblock_fusion_trees(index)
                .unwrap()
                .codomain_vertices()
                .iter()
                .any(|vertex| vertex.get() == 2)
        }),
        "fixture must carry Generic vertex key μ=2"
    );

    let direct = destination.absorb(engine, &source).unwrap();
    assert_eq!(direct.dense_data().unwrap(), expected);
    assert!(std::ptr::eq(
        direct.provider(),
        destination_provider.as_ref()
    ));
    assert!(!std::ptr::eq(direct.provider(), source_provider.as_ref()));
    let source_adjoint = &source.adjoint(engine).unwrap();

    let lazy = destination
        .adjoint(engine)
        .unwrap()
        .absorb(engine, source_adjoint)
        .unwrap();
    let expected_lazy = direct.adjoint(engine).unwrap();
    assert_eq!(
        lazy.dense_data().unwrap(),
        expected_lazy
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    assert_eq!(destination.dense_data().unwrap(), destination_before);
}

#[test]
fn checked_generic_absorb_matches_su3_su4_vertex_keys_prefixes_and_lazy_adjoint() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // What: exact Generic fusion-tree keys (including multiplicity vertices),
    // distinct equal-identity provider Arcs, asymmetric min-prefix copies, and
    // direct/lazy-adjoint parity for f64 and c64 on both SU(3) and SU(4).
    assert_absorb_case(engine, 3, vec![2, 2], |value| value as f64);
    assert_absorb_case(engine, 3, vec![2, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
    assert_absorb_case(engine, 4, vec![2, 0, 2], |value| value as f64);
    assert_absorb_case(engine, 4, vec![2, 0, 2], |value| {
        Complex64::new(value as f64, -(value as f64))
    });
}

#[test]
fn checked_generic_absorb_validation_precedence_leaves_inputs_unchanged() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // What: rank, identity, runtime, and duality reject in that order before
    // copying; the input payloads remain the caller-owned originals.
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let equal_identity =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let wrong_identity =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(4).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let equal_leg = GradedSpace::try_new(Arc::clone(&equal_identity), [(vec![1, 1], 2)]).unwrap();
    let dual_leg = GradedSpace::try_new(Arc::clone(&equal_identity), [(vec![1, 1], 2)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let wrong_leg =
        GradedSpace::try_new(Arc::clone(&wrong_identity), [(vec![1, 0, 1], 2)]).unwrap();
    let destination: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg, &leg], [], |_, indices| {
            indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let before = destination.dense_data().unwrap().to_vec();
    let bad_rank: TensorMap<_, f64> =
        TensorMap::zeros(engine, &other_runtime, [&wrong_leg], []).unwrap();
    let bad_identity: TensorMap<_, f64> = TensorMap::zeros(
        engine,
        &other_runtime,
        [&wrong_leg, &wrong_leg, &wrong_leg],
        [],
    )
    .unwrap();
    let bad_runtime: TensorMap<_, f64> = TensorMap::zeros(
        engine,
        &other_runtime,
        [&equal_leg, &equal_leg, &equal_leg],
        [],
    )
    .unwrap();
    let bad_duality: TensorMap<_, f64> =
        TensorMap::zeros(engine, &runtime, [&dual_leg, &dual_leg, &dual_leg], []).unwrap();

    assert!(
        matches!(destination.absorb(engine, &bad_rank), Err(Error::InvalidArgument(message)) if message.contains("equal codomain/domain ranks"))
    );
    assert!(matches!(
        destination.absorb(engine, &bad_identity),
        Err(Error::RuleMismatch)
    ));
    assert!(matches!(
        destination.absorb(engine, &bad_runtime),
        Err(Error::RuntimeMismatch)
    ));
    assert!(
        matches!(destination.absorb(engine, &bad_duality), Err(Error::InvalidArgument(message)) if message.contains("equal duality"))
    );
    assert_eq!(destination.dense_data().unwrap(), before);
}
