//! Device replay of *provider-compiled* recoupling structures.
//!
//! The structure-level device suite in `symtn-operations` proves the executor's
//! algebra on hand-written fixtures. This one proves it on the structures the
//! categorical layer actually compiles — SU(2) and fZ2 x SU(2) F moves reached
//! through the provider's compiled tree-transform structures, one level below
//! the typed `TensorMap` API —
//! whose recoupling matrices carry irrational 6j entries, mixed signs, and, for
//! the product rule, fermionic signs.
//!
//! Every value is compared with the structure-independent oracle in
//! `categorical_recoupling` *and* with the host executor replaying the same
//! compiled structure. The device is the unit under test, so every test is
//! `#[ignore]` like the rest of the device suite.

#![cfg(feature = "cuda")]

use crate::categorical_recoupling;

use std::sync::Arc;

use categorical_recoupling::{
    alphas, assert_close, compile, expected, expected_scaled, fixtures, four_leg_channels,
    host_replay, host_replay_scaled, non_symmetric_fixture, su2_space, Compiled, TestScalar,
};
use num_complex::Complex64;
use symtn_core::SU2FusionRule;
use symtn_dense::{cuda_transfer_stats, CudaDenseContext, CudaScalar, CudaTransferStats};
use symtn_operations::cuda::CudaStorage;
use symtn_operations::{
    CudaTreeTransformDestination, CudaTreeTransformExecutor, DEFAULT_COEFFICIENT_BUDGET_BYTES,
};
use symtn_tensors::{build_tree_pair_transform_group_plan, TreeTransformOperation};

/// Payload dtypes replayed on device, with the host arithmetic the oracle and
/// the host comparison need.
trait DeviceScalar:
    TestScalar
    + CudaScalar
    + symtn_operations::TreeTransformScalar
    + symtn_operations::RecouplingCoefficientAction<f64>
    + symtn_operations::DenseBlockScalar
{
}

impl DeviceScalar for f64 {}
impl DeviceScalar for Complex64 {}

fn engine() -> symtn_dense::CudaEngine {
    symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0))
        .expect("a CUDA device")
}

fn device_replay<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T> {
    device_replay_scaled(
        ctx,
        executor,
        fixture,
        source,
        destination,
        overwrite,
        T::from_parts(1.0, 0.0),
    )
}

#[allow(clippy::too_many_arguments)]
fn device_replay_scaled<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
    source: &[T],
    destination: &[T],
    overwrite: bool,
    alpha: T,
) -> Vec<T> {
    let mut device_dst = CudaStorage::<T>::upload(ctx, destination).unwrap();
    let device_src = CudaStorage::<T>::upload(ctx, source).unwrap();
    let mode = if overwrite {
        CudaTreeTransformDestination::Overwrite
    } else {
        CudaTreeTransformDestination::Axpby(T::from_parts(1.0, 0.0))
    };
    executor
        .replay(
            ctx,
            &fixture.structure,
            &fixture.space,
            &fixture.space,
            &mut device_dst,
            &device_src,
            alpha,
            mode,
        )
        .unwrap();
    device_dst.download(ctx).unwrap()
}

fn check<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
) {
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        let what = format!("{} / {} / overwrite = {overwrite}", fixture.name, T::NAME);
        let device = device_replay(ctx, executor, fixture, &source, &destination, overwrite);
        assert_close(
            &device,
            &expected(fixture, &source, &destination, overwrite),
            &format!("{what}: device vs oracle"),
        );
        assert_close(
            &device,
            &host_replay(fixture, &source, &destination, overwrite),
            &format!("{what}: device vs host"),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_replay_matches_provider_compiled_recoupling_structures() {
    // What: SU(2) and fZ2 x SU(2) permutes, braids and planar transposes at
    // ranks 4 and 6, with two- and five-channel recoupling, degeneracy 1 and 2,
    // conjugated sources, both payload dtypes and both destination modes.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in fixtures() {
        check::<f64>(&mut ctx, &mut executor, &fixture);
        check::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_device_applies_the_transposed_recoupling_matrix_on_provider_data() {
    // Negative control for the GEMM orientation, on the device and on real 6j
    // data: the fixture's five-channel matrix is not its own transpose, so a
    // device that contracted `U` instead of `Uᵀ` would produce the other
    // buffer. The CPU-only suite proves the two buffers differ; this asserts
    // the device lands on the oracle's one.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = non_symmetric_fixture();
    assert!(
        fixture.has_non_symmetric_matrix(),
        "the orientation control needs a non-symmetric U"
    );
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.len()];

    let device = device_replay(
        &mut ctx,
        &mut executor,
        &fixture,
        &source,
        &destination,
        true,
    );

    assert_close(
        &device,
        &expected(&fixture, &source, &destination, true),
        "su2_rank6_permute: device vs oracle",
    );
    assert_close(
        &device,
        &host_replay(&fixture, &source, &destination, true),
        "su2_rank6_permute: device vs host",
    );
}

fn stats_delta(before: CudaTransferStats, after: CudaTransferStats) -> CudaTransferStats {
    CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        device_allocs: after.device_allocs - before.device_allocs,
        gemm_calls: after.gemm_calls - before.gemm_calls,
        solver_calls: after.solver_calls - before.solver_calls,
        copy_calls: after.copy_calls - before.copy_calls,
        gauge_ops: after.gauge_ops - before.gauge_ops,
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_provider_structure_replay_transfers_nothing() {
    // What: the zero-transfer warm-replay contract holds for a structure the
    // categorical layer compiled, not only for hand-written fixtures — the
    // uploaded coefficient payload and the pack/scatter workspace are the whole
    // per-structure device state.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = non_symmetric_fixture();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.len()];
    let mut device_dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let device_src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &fixture.structure,
                &fixture.space,
                &fixture.space,
                dst,
                &device_src,
                1.0,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };

    replay(&mut ctx, &mut executor, &mut device_dst);
    let cold = cuda_transfer_stats();
    let cold_workspace = executor.workspace_device_bytes();
    assert!(
        cold_workspace > 0,
        "a recoupling structure needs a workspace"
    );
    replay(&mut ctx, &mut executor, &mut device_dst);
    let warm = stats_delta(cold, cuda_transfer_stats());

    assert_eq!(warm.h2d_calls, 0, "warm replay uploaded: {warm:?}");
    assert_eq!(warm.d2h_calls, 0, "warm replay downloaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "warm replay allocated: {warm:?}");
    assert_eq!(
        executor.workspace_device_bytes(),
        cold_workspace,
        "warm replay grew the workspace"
    );
    assert!(warm.gemm_calls > 0, "warm replay submitted nothing");
    assert_close(
        &device_dst.download(&ctx).unwrap(),
        &expected(&fixture, &source, &destination, true),
        "warm provider-structure replay",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn every_caller_scale_matches_the_host_on_provider_compiled_recoupling() {
    // What: the caller scale on genuine 6j data — Single blocks and Multi
    // scatters scaled, packs and the recoupling GEMM not — for
    // alpha in {1, -2.5, 0, -0.0, complex}, both dtypes and both destination
    // modes. The oracle applies alpha at the scatter alone and is pinned
    // against the host in CI, including an `alpha^2` negative control.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = non_symmetric_fixture();
    check_scales::<f64>(&mut ctx, &mut executor, &fixture);
    check_scales::<Complex64>(&mut ctx, &mut executor, &fixture);
}

fn check_scales<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
) {
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        for alpha in alphas::<T>() {
            let what = format!(
                "{} / {} / overwrite = {overwrite} / alpha = {alpha:?}",
                fixture.name,
                T::NAME
            );
            let device = device_replay_scaled(
                ctx,
                executor,
                fixture,
                &source,
                &destination,
                overwrite,
                alpha,
            );
            assert_close(
                &device,
                &expected_scaled(fixture, &source, &destination, overwrite, alpha),
                &format!("{what}: device vs oracle"),
            );
            assert_close(
                &device,
                &host_replay_scaled(fixture, &source, &destination, overwrite, alpha),
                &format!("{what}: device vs host"),
            );
        }
    }
}

/// One SU(2) F-move plan bound at degeneracies 3 and 2: two structures with
/// different layouts and one shared coefficient payload, as a degeneracy-only
/// rebinding of a cached group set produces.
fn degeneracy_pair() -> (Compiled, Compiled) {
    let operation = TreeTransformOperation::permute([0usize, 2, 1, 3], []);
    let space = |degeneracy| su2_space(&[1usize; 4], &four_leg_channels(), degeneracy);
    let plan = build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &space(2)).unwrap();
    let bind = |degeneracy| {
        let space = space(degeneracy);
        Compiled {
            name: format!("su2_permute_degeneracy_{degeneracy}"),
            structure: plan
                .compile_shared_structures_with_storage_conjugation(
                    Arc::clone(&space),
                    Arc::clone(&space),
                    false,
                )
                .unwrap(),
            space,
            conjugate: false,
            source_axes: vec![0, 2, 1, 3],
        }
    };
    let pair = (bind(3), bind(2));
    assert!(pair.0.structure.shares_coefficients_with(&pair.1.structure));
    pair
}

/// The same degeneracy-2 transform from a plan of its own: equal values, an
/// independent payload.
fn independent_small() -> Compiled {
    compile(
        "su2_permute_independent",
        &SU2FusionRule,
        TreeTransformOperation::permute([0usize, 2, 1, 3], []),
        &su2_space(&[1usize; 4], &four_leg_channels(), 2),
        false,
    )
}

/// Replays `fixture` (overwrite, unit scale), checks it against the oracle and
/// returns the host-to-device uploads the replay itself made. Its operands are
/// uploaded before counting, so on a context whose workspace and templates
/// already fit, every counted upload is a coefficient upload.
fn replay_uploads<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Compiled,
) -> u64 {
    let source = fixture.source::<T>();
    let destination = vec![T::from_parts(-1.0, 0.5); fixture.len()];
    let mut device_dst = CudaStorage::<T>::upload(ctx, &destination).unwrap();
    let device_src = CudaStorage::<T>::upload(ctx, &source).unwrap();
    let before = cuda_transfer_stats();
    executor
        .replay(
            ctx,
            &fixture.structure,
            &fixture.space,
            &fixture.space,
            &mut device_dst,
            &device_src,
            T::from_parts(1.0, 0.0),
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap();
    let uploads = cuda_transfer_stats().h2d_calls - before.h2d_calls;
    assert_close(
        &device_dst.download(ctx).unwrap(),
        &expected(fixture, &source, &destination, true),
        &format!("{} / {}", fixture.name, T::NAME),
    );
    uploads
}

/// Device bytes the executor's prepared structures retain (its coefficient
/// payloads), without the workspace.
fn payload_bytes(executor: &CudaTreeTransformExecutor) -> usize {
    executor.executor_device_bytes() - executor.workspace_device_bytes()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn degeneracy_only_structures_upload_and_charge_one_coefficient_payload() {
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let (large, small) = degeneracy_pair();
    // Larger first, so the smaller one grows no workspace or template.
    assert!(replay_uploads::<f64>(&mut ctx, &mut executor, &large) > 0);
    let one = payload_bytes(&executor);
    assert!(one > 0);

    assert_eq!(
        replay_uploads::<f64>(&mut ctx, &mut executor, &small),
        0,
        "a structure bound to a resident payload must not upload it again"
    );
    assert_eq!(
        executor.prepared_structures(),
        2,
        "entries are per structure"
    );
    assert_eq!(
        payload_bytes(&executor),
        one,
        "the shared payload is charged once"
    );

    // Control: an equal payload from another plan is another payload.
    let independent = independent_small();
    assert!(replay_uploads::<f64>(&mut ctx, &mut executor, &independent) > 0);
    assert!(payload_bytes(&executor) > one);
    assert_eq!(executor.prepared_structures(), 3);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_shared_payload_outlives_one_evicted_holder_and_is_released_with_the_last() {
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    // One entry: every insert evicts the previous structure.
    let mut executor =
        CudaTreeTransformExecutor::with_structure_entries(DEFAULT_COEFFICIENT_BUDGET_BYTES, 1);
    let (large, small) = degeneracy_pair();
    let other = non_symmetric_fixture();
    let other_bytes = {
        let mut alone = CudaTreeTransformExecutor::default();
        replay_uploads::<f64>(&mut ctx, &mut alone, &other);
        payload_bytes(&alone)
    };
    assert!(replay_uploads::<f64>(&mut ctx, &mut executor, &other) > 0);
    assert!(replay_uploads::<f64>(&mut ctx, &mut executor, &large) > 0);
    let one = payload_bytes(&executor);
    assert_ne!(one, other_bytes, "the control must be told apart by size");

    // `small` shares `large`'s payload, then its insert evicts `large`.
    assert_eq!(replay_uploads::<f64>(&mut ctx, &mut executor, &small), 0);
    assert_eq!(executor.prepared_structures(), 1);
    assert_eq!(payload_bytes(&executor), one, "the charge moved to `small`");
    // `large` is re-prepared from the payload `small` kept alive.
    assert_eq!(replay_uploads::<f64>(&mut ctx, &mut executor, &large), 0);
    assert_eq!(payload_bytes(&executor), one);

    // Evicting the last holder releases the payload.
    assert!(replay_uploads::<f64>(&mut ctx, &mut executor, &other) > 0);
    assert_eq!(payload_bytes(&executor), other_bytes);
    assert!(
        replay_uploads::<f64>(&mut ctx, &mut executor, &small) > 0,
        "a released payload is uploaded again"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn dead_holders_are_purged_and_reset_releases_every_payload() {
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let (large, small) = degeneracy_pair();
    let other = non_symmetric_fixture();
    replay_uploads::<f64>(&mut ctx, &mut executor, &large);
    let one = payload_bytes(&executor);
    assert_eq!(replay_uploads::<f64>(&mut ctx, &mut executor, &small), 0);

    // `large`'s core dies; the next insert purges its entry, and its charge
    // moves to `small`, which still holds the payload.
    drop(large);
    replay_uploads::<f64>(&mut ctx, &mut executor, &other);
    let with_other = payload_bytes(&executor);
    assert_eq!(executor.prepared_structures(), 2);
    assert!(with_other > one);
    assert_eq!(replay_uploads::<f64>(&mut ctx, &mut executor, &small), 0);

    // The last holder dies: the next insert releases the payload.
    drop(small);
    let independent = independent_small();
    replay_uploads::<f64>(&mut ctx, &mut executor, &independent);
    assert_eq!(executor.prepared_structures(), 2);
    assert_eq!(
        payload_bytes(&executor),
        with_other,
        "`independent` replaced the released payload byte for byte"
    );

    executor.clear();
    assert_eq!(executor.prepared_structures(), 0);
    assert_eq!(executor.executor_device_bytes(), 0);
    assert!(replay_uploads::<f64>(&mut ctx, &mut executor, &independent) > 0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn another_dtype_or_context_never_shares_a_payload() {
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let (large, small) = degeneracy_pair();
    replay_uploads::<f64>(&mut ctx, &mut executor, &large);
    replay_uploads::<Complex64>(&mut ctx, &mut executor, &large);
    let both = payload_bytes(&executor);

    // Each dtype shares within itself only.
    assert_eq!(
        replay_uploads::<Complex64>(&mut ctx, &mut executor, &small),
        0
    );
    assert_eq!(replay_uploads::<f64>(&mut ctx, &mut executor, &small), 0);
    assert_eq!(payload_bytes(&executor), both);

    let mut other_ctx_engine = engine();

    let mut other_ctx = CudaDenseContext::new(&mut other_ctx_engine).unwrap();
    assert_ne!(ctx.identity(), other_ctx.identity());
    assert!(
        replay_uploads::<f64>(&mut other_ctx, &mut executor, &small) > 0,
        "another context must upload its own payload"
    );
    assert!(payload_bytes(&executor) > both);
}
