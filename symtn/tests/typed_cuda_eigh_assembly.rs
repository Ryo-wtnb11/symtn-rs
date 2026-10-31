//! Device `eigh_full` assembles its eigenvectors without a GEMM when
//! cuSOLVER's order is already the published ascending one, and its
//! uploads do not depend on the number of fusion trees.
//!
//! [`cuda_transfer_stats`] snapshots count only the calling thread.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! integration typed_cuda_eigh_assembly:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::expert::cuda_transfer_stats;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Eigh, GradedSpace, Runtime, TensorMap};

use crate::fixtures;

use fixtures::{codomain_axes, domain_axes};

fn leg(charges: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        charges
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn close(
    engine: &mut dyn symtn::typed::Engine,
    got: &TensorMap<U1FusionRule, f64>,
    want: &TensorMap<U1FusionRule, f64>,
) {
    assert_eq!(
        got.materialize(engine).unwrap().dense_data().unwrap().len(),
        want.materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .len()
    );
    for (got, want) in got
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(want.materialize(engine).unwrap().dense_data().unwrap())
    {
        assert!((got - want).abs() <= 1e-10, "{got} vs {want}");
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn eigh_assembly_gemms_and_uploads_do_not_depend_on_the_tree_count() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let small = leg(&[(-1, 2), (0, 1), (1, 2)]);
    // The same five coupled sectors with the same matrix sizes (4, 4, 9, 4,
    // 4), reached through one codomain tree each (`fused`) or through nine
    // trees in total (`small x small`).
    let fused = leg(&[(-2, 4), (-1, 4), (0, 9), (1, 4), (2, 4)]);
    let one_tree = TensorMap::rand_with_seed(engine, &runtime, [&fused], [&fused], 7).unwrap();
    let many_trees =
        TensorMap::rand_with_seed(engine, &runtime, [&small, &small], [&small, &small], 7).unwrap();

    let mut counts = Vec::new();
    for source in [one_tree, many_trees] {
        let adjoint = source.adjoint(engine).unwrap();
        let source = source.axpby(engine, 1.0, &adjoint, 1.0).unwrap();
        let device = source.to_cuda(cuda).unwrap();

        let before = cuda_transfer_stats();
        let Eigh { d, v } = device
            .eigh_full(
                cuda,
                &codomain_axes(&device),
                &domain_axes(&device),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        let after = cuda_transfer_stats();
        let gemms = after.gemm_calls - before.gemm_calls;
        // cuSOLVER's ascending order is the published one: the
        // eigenvectors are copied, with no selector GEMM.
        assert_eq!(gemms, 0);
        counts.push((
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
        ));

        // Device vs host: the same ascending spectrum, and the device
        // eigenvectors (raw cuSOLVER gauge) satisfy the eigen equation.
        let Eigh { d: host_d, .. } = source
            .eigh_full(
                engine,
                &codomain_axes(&source),
                &domain_axes(&source),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        let d = d.to_host(cuda).unwrap();
        let v = v.to_host(cuda).unwrap();
        close(engine, &d, &host_d);
        let lhs = source.compose(engine, &v).unwrap();
        let rhs = v.compose(engine, &d).unwrap();
        close(engine, &lhs, &rhs);
    }
    // The same sectors and sizes, so every upload is independent of the tree
    // count. The selector-free copy counts and the forced selector path are
    // asserted in-crate (`typed_cuda_eigh_aligned_assembly_matches_the_per_tree_path_bitwise`).
    assert_eq!(counts[0], counts[1], "uploads do not depend on the trees");
}
