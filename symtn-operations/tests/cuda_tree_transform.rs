//! Device tests for the tree-transform executor (issues #1304, #1310).
//!
//! Oracles: the explicit-index walk in `common`, which never reads a
//! `TreeTransformStructure` and is itself pinned against the host executor by
//! `tree_transform_device_oracle.rs` (CPU-only, runs in CI). Every value test
//! additionally compares against the host executor replaying the *same*
//! compiled structure, which catches a device/host divergence even where both
//! disagree with the fixture's intent.
//!
//! Numerical device tests are `#[ignore]`; metadata and admission tests run
//! without a device in ordinary CUDA-feature CI.

#![cfg(feature = "cuda")]

use crate::common;

use common::{
    all_fixtures, conjugated_recoupling, expert_interleaved_destination,
    expert_interleaved_recoupling_destination, inactive_destination_layouts,
    many_distinct_signatures, mixed_single_and_multi, rank_sweep, recoupling_non_symmetric_u,
    unit_coefficient_fixtures, Fixture, TestScalar,
};
use num_complex::{Complex32, Complex64};
use std::sync::Arc;
use symtn_core::BlockStructure;
use symtn_dense::{
    cuda_transfer_stats, reset_cuda_transfer_stats, CudaDenseContext, CudaScalar, CudaTransferStats,
};
use symtn_operations::cuda::CudaStorage;
use symtn_operations::cuda_transform::CudaSingleMemberRegions;
use symtn_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw,
    tree_transform_structure_with_strided_kernel_raw, CudaTreeTransformDestination,
    CudaTreeTransformExecutor, OperationError, StridedHostKernelAdapter, TreeTransformBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace,
};

#[test]
fn unit_single_member_preflight_rejects_cross_member_geometry() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![3]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
    )
    .unwrap();
    // The total allocation fits two five-element members, but a claimed
    // four-element source member cannot contain the compiled five-element view.
    assert!(matches!(
        CudaSingleMemberRegions::prepare(&transform, &structure, &structure, 5, 4, 10, 10, 2,),
        Err(OperationError::ElementCountMismatch { .. })
    ));
}

#[test]
fn single_member_admission_accepts_nonzero_and_rejects_zero() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2]]).unwrap());
    for coefficient in [-1.0, 1.0000000000000002] {
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, coefficient)],
        )
        .unwrap();
        CudaSingleMemberRegions::admit(&transform).unwrap();
        let regions =
            CudaSingleMemberRegions::prepare(&transform, &structure, &structure, 2, 2, 34, 34, 17)
                .unwrap();
        assert!(regions.has_scaled_moves());
        assert_eq!(regions.moves()[0].2, coefficient);
    }

    let zero = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 0.0)],
    )
    .unwrap();
    assert!(matches!(
        CudaSingleMemberRegions::admit(&zero),
        Err(OperationError::UnsupportedTensorContractScope { .. })
    ));

    let tiny = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, f64::from_bits(1))],
    )
    .unwrap();
    let regions =
        CudaSingleMemberRegions::prepare(&tiny, &structure, &structure, 2, 2, 2, 2, 1).unwrap();
    assert_eq!(regions.scaled_coefficients::<f32>(), [0.0]);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn unit_single_member_overwrite_zeros_inactive_layouts() {
    fn check<D: DeviceScalar + PartialEq + std::fmt::Debug>() {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![3]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap();
        let mut ctx_engine = engine();
        let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
        for members in [1, 2, 17] {
            let regions = CudaSingleMemberRegions::prepare(
                &transform,
                &structure,
                &structure,
                5,
                5,
                5 * members,
                5 * members,
                members,
            )
            .unwrap();
            assert_eq!(regions.moves().len(), 1);
            assert_eq!(regions.zeros().len(), 1);
            ctx.reserve_zero_template::<D>(regions.max_zero_len() * members)
                .unwrap();
            ctx.reserve_ones_template::<D>(1).unwrap();
            let source: Vec<D> = (0..5 * members)
                .map(|i| D::from_parts(1.0 + i as f64 / 8.0, -0.5 - i as f64 / 16.0))
                .collect();
            let input = CudaStorage::upload_owned(&ctx, source.clone()).unwrap();
            for _ in 0..2 {
                let mut output = CudaStorage::upload_owned(
                    &ctx,
                    vec![D::from_parts(f64::NAN, f64::NAN); 5 * members],
                )
                .unwrap();
                let before = cuda_transfer_stats();
                regions
                    .execute_overwrite(&mut ctx, &input, None, &mut output)
                    .unwrap();
                let after = cuda_transfer_stats();
                assert_eq!(after.copy_calls - before.copy_calls, 1);
                assert_eq!(after.h2d_calls - before.h2d_calls, 0);
                assert_eq!(after.d2h_calls - before.d2h_calls, 0);
                let actual = output.download(&ctx).unwrap();
                for member in 0..members {
                    assert_eq!(
                        &actual[member * 5..member * 5 + 2],
                        &source[member * 5..member * 5 + 2]
                    );
                    assert_eq!(
                        &actual[member * 5 + 2..member * 5 + 5],
                        &[D::from_parts(0.0, 0.0); 3]
                    );
                }
            }
        }
    }
    check::<f64>();
    check::<Complex64>();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn underflowed_f32_single_coefficient_remains_a_data_operand() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, f64::from_bits(1))],
    )
    .unwrap();
    let regions =
        CudaSingleMemberRegions::prepare(&transform, &structure, &structure, 2, 2, 2, 2, 1)
            .unwrap();
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let coefficients =
        CudaStorage::upload_owned(&ctx, regions.scaled_coefficients::<f32>()).unwrap();
    let input = CudaStorage::upload_owned(&ctx, vec![f32::NAN, 2.0]).unwrap();
    let mut output = CudaStorage::upload_owned(&ctx, vec![1.0_f32; 2]).unwrap();
    regions
        .execute_overwrite(&mut ctx, &input, Some(&coefficients.0), &mut output)
        .unwrap();
    let actual = output.download(&ctx).unwrap();
    assert!(
        actual[0].is_nan(),
        "zero coefficient still reads NaN source data"
    );
    assert_eq!(actual[1].to_bits(), 0.0_f32.to_bits());
}

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

impl DeviceScalar for f32 {}
impl DeviceScalar for f64 {}
impl DeviceScalar for Complex32 {}
impl DeviceScalar for Complex64 {}

fn engine() -> symtn_dense::CudaEngine {
    symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0))
        .expect("a CUDA device")
}

fn host_replay<T: DeviceScalar>(
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    overwrite: bool,
) -> Vec<T> {
    host_replay_scaled(
        fixture,
        source,
        destination,
        overwrite,
        T::from_parts(1.0, 0.0),
    )
}

/// The caller scales the executor must reproduce: one, minus one (the only
/// other value a fermionic twist folded into a descriptor ever takes),
/// a scale that is neither, both signed zeros, and a genuinely complex one
/// (its real part on `f64`).
fn alphas<T: DeviceScalar>() -> Vec<T> {
    vec![
        T::from_parts(1.0, 0.0),
        T::from_parts(-1.0, 0.0),
        T::from_parts(-2.5, 0.0),
        T::from_parts(0.0, 0.0),
        T::from_parts(-0.0, -0.0),
        T::from_parts(0.5, -1.25),
    ]
}

fn host_replay_scaled<T: DeviceScalar>(
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    overwrite: bool,
    alpha: T,
) -> Vec<T> {
    let structure = fixture.compile();
    let mut kernels = StridedHostKernelAdapter::default();
    let mut workspace = TreeTransformWorkspace::<T>::default();
    let mut data = destination.to_vec();
    let one = T::from_parts(1.0, 0.0);
    if overwrite {
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            strided_kernel::Exec::serial(),
            &mut kernels,
            &mut workspace,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut data,
            source,
            alpha,
        )
        .unwrap();
    } else {
        tree_transform_structure_with_strided_kernel_raw(
            strided_kernel::Exec::serial(),
            &mut kernels,
            &mut workspace,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut data,
            source,
            alpha,
            one,
        )
        .unwrap();
    }
    data
}

/// Replays `fixture` on device and downloads the destination.
fn device_replay<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
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
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    overwrite: bool,
    alpha: T,
) -> Vec<T> {
    let structure = fixture.compile();
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
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut device_dst,
            &device_src,
            alpha,
            mode,
        )
        .unwrap();
    device_dst.download(ctx).unwrap()
}

/// The bound a replayed move is held to: the `1e-12` this file has always used
/// for the double-precision payloads, and the same number of epsilons of its
/// own real lane for each single-precision one. A tighter double-precision
/// bound would assert the contraction order of whichever GPU runs the suite.
fn move_tolerance<T: TestScalar>() -> f64 {
    1e-12 * (T::EPSILON / f64::EPSILON)
}

fn assert_close<T: TestScalar>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            left.distance(*right) <= move_tolerance::<T>(),
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

fn check_fixture<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
) {
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        let what = format!("{} / {} / overwrite = {overwrite}", fixture.name, T::NAME);
        let device = device_replay(ctx, executor, fixture, &source, &destination, overwrite);
        assert_close(
            &device,
            &fixture.expected(&source, &destination, overwrite),
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
fn device_replay_matches_the_oracle_and_the_host_for_every_fixture() {
    // What: rank 2-6 permutes, transposes, fermionic signs, coefficients that
    // are neither 1 nor -1, conjugated sources, interleaved multi-block
    // layouts, inactive layouts and a zero-extent block, in all four payload
    // dtypes (#1326) and both destination modes.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in all_fixtures() {
        check_fixture::<f32>(&mut ctx, &mut executor, &fixture);
        check_fixture::<f64>(&mut ctx, &mut executor, &fixture);
        check_fixture::<Complex32>(&mut ctx, &mut executor, &fixture);
        check_fixture::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_cleans_a_nan_poisoned_destination_including_inactive_layouts() {
    // What: the Overwrite mode is destination-independent — every inactive
    // destination layout is zeroed, so a caller's poisoned buffer cannot leak
    // into a block the transform does not reach.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = inactive_destination_layouts();
    let source = fixture.source::<f64>();
    let poisoned = vec![f64::NAN; fixture.dst_len()];

    let device = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, true);

    assert!(
        device.iter().all(|value| !value.is_nan()),
        "overwrite left NaN behind: {device:?}"
    );
    assert_close(
        &device,
        &fixture.expected(&source, &poisoned, true),
        "poisoned destination",
    );

    // Negative control: accumulation does not promise a clean destination, so
    // the NaN must survive it — otherwise the test above proves nothing about
    // the zeroing of inactive layouts.
    let accumulated = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, false);
    assert!(
        accumulated.iter().any(|value| value.is_nan()),
        "accumulation must keep the destination's NaN: {accumulated:?}"
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
fn a_warm_replay_transfers_nothing_and_allocates_no_device_buffer() {
    // What: after the first replay of a structure, replaying it again moves
    // nothing across the host boundary and creates no device buffer; only the
    // per-block submissions remain.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = inactive_destination_layouts();
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut device_dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let device_src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                dst,
                &device_src,
                1.0,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };

    // Cold: the coefficient vector and the zero template are uploaded here.
    replay(&mut ctx, &mut executor, &mut device_dst);
    let cold = cuda_transfer_stats();
    replay(&mut ctx, &mut executor, &mut device_dst);
    let warm = stats_delta(cold, cuda_transfer_stats());

    assert_eq!(warm.h2d_calls, 0, "warm replay uploaded: {warm:?}");
    assert_eq!(warm.d2h_calls, 0, "warm replay downloaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "warm replay allocated: {warm:?}");
    // One submission per active block plus one per inactive destination layout.
    assert_eq!(warm.gemm_calls, 3, "submissions changed: {warm:?}");

    // Negative control: a *different* structure does upload its own
    // coefficients, so "no upload" above is a property of reuse, not of the
    // counters being dead.
    let other = many_distinct_signatures(3);
    let cold_other = cuda_transfer_stats();
    let _ = device_replay(
        &mut ctx,
        &mut executor,
        &other,
        &other.source::<f64>(),
        &vec![0.0_f64; other.dst_len()],
        true,
    );
    let delta = stats_delta(cold_other, cuda_transfer_stats());
    assert!(
        delta.h2d_calls > 0,
        "a new structure must upload its coefficients: {delta:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn alternating_structures_upload_their_coefficients_exactly_once_each() {
    // What: the coefficient cache holds more than one structure, so a network
    // replay alternating two transforms does not re-upload on every switch.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixtures = [rank_sweep().remove(2), interleaved()];
    let prepared: Vec<_> = fixtures
        .iter()
        .map(|fixture| {
            (
                fixture.compile(),
                fixture.dst_structure(),
                fixture.src_structure(),
                fixture.clone(),
            )
        })
        .collect();
    let mut buffers: Vec<_> = prepared
        .iter()
        .map(|(_, _, _, fixture)| {
            let source = fixture.source::<f64>();
            let destination = vec![0.0_f64; fixture.dst_len()];
            (
                CudaStorage::<f64>::upload(&ctx, &destination).unwrap(),
                CudaStorage::<f64>::upload(&ctx, &source).unwrap(),
            )
        })
        .collect();

    let before = cuda_transfer_stats();
    for round in 0..3 {
        for (index, (structure, dst_structure, src_structure, _)) in prepared.iter().enumerate() {
            let (dst, src) = &mut buffers[index];
            executor
                .replay(
                    &mut ctx,
                    structure,
                    dst_structure,
                    src_structure,
                    dst,
                    src,
                    1.0,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
            assert_eq!(
                executor.prepared_structures(),
                if round == 0 && index == 0 { 1 } else { 2 },
                "round {round}, structure {index}"
            );
        }
    }
    let delta = stats_delta(before, cuda_transfer_stats());

    assert_eq!(
        delta.h2d_calls, 2,
        "exactly one coefficient upload per structure: {delta:?}"
    );
}

fn interleaved() -> Fixture {
    common::interleaved_multi_block()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_dropped_structure_releases_its_cached_coefficients() {
    // What: the cache keys on a Weak witness of the structure, so device state
    // for a structure the transform cache has evicted cannot survive it.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let held = many_distinct_signatures(2);

    {
        let doomed = fixture.compile();
        let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
        let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
        executor
            .replay(
                &mut ctx,
                &doomed,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                1.0,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
        assert_eq!(executor.prepared_structures(), 1);
        assert!(executor.retained_device_bytes(&ctx) > 0);
    }

    // Preparing any other structure purges the dead entry.
    let _ = device_replay(
        &mut ctx,
        &mut executor,
        &held,
        &held.source::<f64>(),
        &vec![0.0_f64; held.dst_len()],
        true,
    );
    assert_eq!(
        executor.prepared_structures(),
        1,
        "the dropped structure's device state must be released"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_coefficient_of_one_moves_f64_payloads_bitwise() {
    // What: where the host copies bit-exactly, the device's multiply by the
    // uploaded `1` is exact too, so the two agree to the last bit for finite
    // f64 payloads. (Not a contract for other coefficients or for complex
    // payloads: the device multiply can deviate from the host at +-inf.)
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in unit_coefficient_fixtures() {
        let source = fixture.source::<f64>();
        let destination = vec![0.0_f64; fixture.dst_len()];
        let device = device_replay(
            &mut ctx,
            &mut executor,
            &fixture,
            &source,
            &destination,
            true,
        );
        assert_eq!(
            device,
            host_replay(&fixture, &source, &destination, true),
            "{} must match the host bitwise",
            fixture.name
        );
        assert_eq!(
            device,
            fixture.expected(&source, &destination, true),
            "{} must match the oracle bitwise",
            fixture.name
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn unsupported_modes_are_rejected_before_any_device_work() {
    // What: a destination layout the device cannot express is a typed
    // capability error reported before any upload, allocation or submission
    // — for a Single-block structure and for a recoupling
    // structure, whose pack columns and coefficient upload must not happen on
    // the strength of one unwritable scatter region. Every `beta` is
    // supported (#1550), so it is not a rejection input.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();

    // A structure whose destination layout the host proves injective only
    // through its exact overlap fallback, which the device region primitive
    // cannot express.
    let expert = expert_interleaved_destination();
    let expert_structure = expert.compile();
    let expert_destination: Vec<f64> = (0..expert.dst_len()).map(|i| 100.0 + i as f64).collect();
    let mut expert_dst = CudaStorage::<f64>::upload(&ctx, &expert_destination).unwrap();
    let expert_src = CudaStorage::<f64>::upload(&ctx, &expert.source::<f64>()).unwrap();

    let recoupling = expert_interleaved_recoupling_destination();
    let recoupling_structure = recoupling.compile();
    let recoupling_destination: Vec<f64> = (0..recoupling.dst_len())
        .map(|i| 200.0 + i as f64)
        .collect();
    let mut recoupling_dst = CudaStorage::<f64>::upload(&ctx, &recoupling_destination).unwrap();
    let recoupling_src = CudaStorage::<f64>::upload(&ctx, &recoupling.source::<f64>()).unwrap();

    reset_cuda_transfer_stats();
    let before = cuda_transfer_stats();

    let layout = executor
        .replay(
            &mut ctx,
            &expert_structure,
            &expert.dst_structure(),
            &expert.src_structure(),
            &mut expert_dst,
            &expert_src,
            1.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(
            layout,
            OperationError::UnsupportedDeviceTreeTransform { .. }
        ),
        "an inexpressible destination layout gave {layout:?}"
    );
    assert_eq!(
        expert_dst.download(&ctx).unwrap(),
        expert_destination,
        "a rejected layout must leave the caller's destination untouched"
    );

    let scatter = executor
        .replay(
            &mut ctx,
            &recoupling_structure,
            &recoupling.dst_structure(),
            &recoupling.src_structure(),
            &mut recoupling_dst,
            &recoupling_src,
            1.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(
            scatter,
            OperationError::UnsupportedDeviceTreeTransform { .. }
        ),
        "an inexpressible scatter destination gave {scatter:?}"
    );
    assert_eq!(
        recoupling_dst.download(&ctx).unwrap(),
        recoupling_destination,
        "a rejected recoupling must leave the caller's destination untouched"
    );
    assert_eq!(
        executor.workspace_device_bytes(),
        0,
        "a rejected recoupling must not have grown the workspace"
    );

    let delta = stats_delta(before, cuda_transfer_stats());
    // The two downloads above are the test's own read-backs.
    let read_back = (expert_destination.len() + recoupling_destination.len()) * size_of::<f64>();
    let delta = CudaTransferStats {
        d2h_calls: delta.d2h_calls - 2,
        d2h_bytes: delta.d2h_bytes - read_back as u64,
        ..delta
    };
    assert_eq!(delta, CudaTransferStats::default(), "rejection did work");
    assert_eq!(executor.prepared_structures(), 0);

    // Negative control: a writable structure replays, with a general beta
    // too, and a recoupling structure whose destinations *are* writable
    // replays, so the rejections above are about the layout, not about Multi
    // blocks or about the fixtures.
    let fixture = interleaved();
    let mut dst = CudaStorage::<f64>::upload(&ctx, &vec![0.0_f64; fixture.dst_len()]).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &fixture.source::<f64>()).unwrap();
    executor
        .replay(
            &mut ctx,
            &fixture.compile(),
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            1.0,
            CudaTreeTransformDestination::Axpby(2.0),
        )
        .unwrap();
    assert_eq!(executor.prepared_structures(), 1);

    let accepted = recoupling_non_symmetric_u();
    let accepted_source = accepted.source::<f64>();
    let accepted_destination = vec![0.0_f64; accepted.dst_len()];
    let device = device_replay(
        &mut ctx,
        &mut executor,
        &accepted,
        &accepted_source,
        &accepted_destination,
        true,
    );
    assert_close(
        &device,
        &accepted.expected(&accepted_source, &accepted_destination, true),
        "an accepted recoupling structure",
    );
    assert!(executor.workspace_device_bytes() > 0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn mismatched_structures_and_lengths_are_rejected_by_admission() {
    // Negative control for the admission wiring: Stage A still checks the
    // structures and the exact storage lengths, and the device buffer's
    // placement is the executor's own.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let other = inactive_destination_layouts();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let structure = fixture.compile();
    let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let mut short =
        CudaStorage::<f64>::upload(&ctx, &destination[..destination.len() - 1]).unwrap();

    let wrong_structure = executor
        .replay(
            &mut ctx,
            &structure,
            &other.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            1.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    // The Host replay's categories (#1763), not a collapsed InvalidArgument.
    assert_eq!(
        wrong_structure,
        OperationError::StructureMismatch { tensor: "dst" }
    );

    let wrong_length = executor
        .replay(
            &mut ctx,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut short,
            &src,
            1.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert_eq!(
        wrong_length,
        OperationError::ElementCountMismatch {
            expected: destination.len(),
            actual: destination.len() - 1,
        }
    );

    // The same call with the right structures and lengths succeeds.
    executor
        .replay(
            &mut ctx,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            1.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_second_engine_prepares_its_own_device_state() {
    // What: device state is keyed by the engine's allocation domain, so a
    // buffer of another engine (here on the same device) is never read back
    // through a cached coefficient vector, while a later context on the same
    // engine reuses the entry.
    let mut first_engine = engine();
    let mut first = CudaDenseContext::new(&mut first_engine).unwrap();
    let mut second_engine = engine();
    let mut second = CudaDenseContext::new(&mut second_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = interleaved();
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];

    for ctx in [&mut first, &mut second] {
        let mut dst = CudaStorage::<f64>::upload(ctx, &destination).unwrap();
        let src = CudaStorage::<f64>::upload(ctx, &source).unwrap();
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                1.0,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
        assert_close(
            &dst.download(ctx).unwrap(),
            &fixture.expected(&source, &destination, true),
            "per-context replay",
        );
    }
    assert_eq!(
        executor.prepared_structures(),
        2,
        "one entry per (structure, dtype, engine)"
    );
    drop(first);
    let mut again = CudaDenseContext::new(&mut first_engine).unwrap();
    let mut dst = CudaStorage::<f64>::upload(&again, &destination).unwrap();
    let src = CudaStorage::<f64>::upload(&again, &source).unwrap();
    let before = cuda_transfer_stats();
    executor
        .replay(
            &mut again,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut dst,
            &src,
            1.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap();
    assert_eq!(stats_delta(before, cuda_transfer_stats()).h2d_calls, 0);
    assert_eq!(
        executor.prepared_structures(),
        2,
        "the same engine reuses its entry"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn both_payload_dtypes_share_one_structure_with_their_own_coefficients() {
    // What: the same compiled structure replays for f64 and Complex64, each
    // with its own converted coefficient vector.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = rank_sweep().remove(4); // conjugated, fermionic sign, rank 4
    let structure = fixture.compile();

    for complex in [false, true] {
        if complex {
            let source = fixture.source::<Complex64>();
            let destination = vec![Complex64::new(0.0, 0.0); fixture.dst_len()];
            let mut dst = CudaStorage::<Complex64>::upload(&ctx, &destination).unwrap();
            let src = CudaStorage::<Complex64>::upload(&ctx, &source).unwrap();
            executor
                .replay(
                    &mut ctx,
                    &structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    &mut dst,
                    &src,
                    Complex64::new(1.0, 0.0),
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
            assert_close(
                &dst.download(&ctx).unwrap(),
                &fixture.expected(&source, &destination, true),
                "complex payload of a conjugated source",
            );
        } else {
            let source = fixture.source::<f64>();
            let destination = vec![0.0_f64; fixture.dst_len()];
            let mut dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
            let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
            executor
                .replay(
                    &mut ctx,
                    &structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    &mut dst,
                    &src,
                    1.0,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
            assert_close(
                &dst.download(&ctx).unwrap(),
                &fixture.expected(&source, &destination, true),
                "real payload",
            );
        }
    }
    assert_eq!(executor.prepared_structures(), 2, "one entry per dtype");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_recoupling_replay_transfers_nothing_and_reuses_the_workspace() {
    // What: the pack/scatter workspace and the uploaded recoupling matrices are
    // the whole per-structure device state, so replaying a Multi structure a
    // second time moves nothing across the boundary, allocates nothing, and
    // grows the workspace by nothing.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let mut device_dst = CudaStorage::<f64>::upload(&ctx, &destination).unwrap();
    let device_src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<f64>| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
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
    assert!(cold_workspace > 0, "a Multi structure needs a workspace");
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
    // Two GEMM jobs, four packs, four scatters, one Single block and one
    // inactive destination layout: every one of them is a submission.
    assert_eq!(warm.gemm_calls, 12, "submissions changed: {warm:?}");
    assert_close(
        &device_dst.download(&ctx).unwrap(),
        &fixture.expected(&source, &destination, true),
        "warm recoupling replay",
    );

    // Negative control: a wider recoupling structure does grow the workspace
    // and does upload, so the equalities above are reuse, not dead counters.
    let wider = many_distinct_recoupling_columns(8);
    let before = cuda_transfer_stats();
    let _ = device_replay(
        &mut ctx,
        &mut executor,
        &wider,
        &wider.source::<f64>(),
        &vec![0.0_f64; wider.dst_len()],
        true,
    );
    let delta = stats_delta(before, cuda_transfer_stats());
    assert!(
        delta.h2d_calls > 0,
        "a new structure must upload: {delta:?}"
    );
    assert!(
        executor.workspace_device_bytes() > cold_workspace,
        "a wider structure must grow the workspace"
    );

    // And replaying the narrow structure again neither shrinks the workspace
    // nor re-uploads: growth is monotonic and the buffers are shared.
    let before = cuda_transfer_stats();
    let grown = executor.workspace_device_bytes();
    replay(&mut ctx, &mut executor, &mut device_dst);
    let delta = stats_delta(before, cuda_transfer_stats());
    assert_eq!(delta.h2d_calls, 0, "re-upload after growth: {delta:?}");
    assert_eq!(delta.device_allocs, 0, "re-allocation after growth");
    assert_eq!(executor.workspace_device_bytes(), grown, "workspace shrank");
}

/// A recoupling structure with `columns` sources and `columns` destinations of
/// a larger degeneracy box than [`mixed_single_and_multi`], used as the
/// workspace-growth control.
fn many_distinct_recoupling_columns(columns: usize) -> Fixture {
    let element_count = 12;
    let mut dst_blocks = Vec::with_capacity(columns);
    let mut src_blocks = Vec::with_capacity(columns);
    for index in 0..columns {
        dst_blocks.push(common::Block::packed(vec![3, 4], index * element_count));
        src_blocks.push(common::Block::packed(vec![4, 3], index * element_count));
    }
    let u = (0..columns * columns)
        .map(|index| 0.5 + (index as f64) * 0.25)
        .collect();
    Fixture {
        name: "many_distinct_recoupling_columns",
        rank: 2,
        dst_blocks,
        src_blocks,
        pairs: Vec::new(),
        groups: vec![common::Group {
            dst_blocks: (0..columns).collect(),
            src_blocks: (0..columns).collect(),
            axes: vec![1, 0],
            u,
        }],
        conjugate: false,
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn alternating_recoupling_structures_upload_their_matrices_exactly_once_each() {
    // What: the recoupling matrices are cached with the block coefficients
    // under the same (structure, dtype, context) key, so a replay alternating
    // two Multi structures uploads each structure's matrices once and shares
    // one workspace between them.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    // Widest first, so the workspace is allocated once and the narrower
    // structure reuses it: growth is monotonic and shared across structures.
    let fixtures = [mixed_single_and_multi(), recoupling_non_symmetric_u()];
    let prepared: Vec<_> = fixtures
        .iter()
        .map(|fixture| (fixture.compile(), fixture.clone()))
        .collect();
    let mut buffers: Vec<_> = prepared
        .iter()
        .map(|(_, fixture)| {
            (
                CudaStorage::<f64>::upload(&ctx, &vec![0.0_f64; fixture.dst_len()]).unwrap(),
                CudaStorage::<f64>::upload(&ctx, &fixture.source::<f64>()).unwrap(),
            )
        })
        .collect();

    let before = cuda_transfer_stats();
    let mut after_first_round = before;
    for round in 0..3 {
        for (index, (structure, fixture)) in prepared.iter().enumerate() {
            let (dst, src) = &mut buffers[index];
            executor
                .replay(
                    &mut ctx,
                    structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    dst,
                    src,
                    1.0,
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
        }
        if round == 0 {
            after_first_round = cuda_transfer_stats();
        }
    }
    let cold = stats_delta(before, after_first_round);
    let warm = stats_delta(after_first_round, cuda_transfer_stats());

    assert_eq!(executor.prepared_structures(), 2);
    // Switching structures again uploads nothing at all: each structure's
    // matrices are resident and the workspace is shared.
    assert_eq!(warm.h2d_calls, 0, "a switch re-uploaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "a switch allocated: {warm:?}");
    // The first round pays six: one coefficient-and-matrix vector per
    // structure, the two workspace buffers the wider structure allocates and
    // the narrower one reuses, the context's shared `1` that pack and scatter
    // read as their coefficient, and the zero template the wider structure's
    // inactive destination layout needs.
    assert_eq!(cold.h2d_calls, 6, "cold uploads changed: {cold:?}");
    for (index, (_, fixture)) in prepared.iter().enumerate() {
        let (dst, _) = &buffers[index];
        assert_close(
            &dst.download(&ctx).unwrap(),
            &fixture.expected(
                &fixture.source::<f64>(),
                &vec![0.0_f64; fixture.dst_len()],
                true,
            ),
            fixture.name,
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn recoupling_replays_match_the_oracle_in_both_dtypes_and_modes() {
    // What: the Multi path itself — non-symmetric square U, rectangular U, a
    // conjugated source, and Single and Multi blocks in one structure — over
    // both payload dtypes and both destination modes, against the oracle and
    // against the host executor replaying the same compiled structure.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in common::recoupling_fixtures() {
        check_fixture::<f64>(&mut ctx, &mut executor, &fixture);
        check_fixture::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_cleans_a_nan_poisoned_destination_around_recoupling_blocks() {
    // What: Overwrite is destination-independent for a Multi structure too —
    // the scatter assigns and the untouched layout beside it is zeroed, so a
    // poisoned buffer cannot leak through a recoupling block.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let source = fixture.source::<f64>();
    let poisoned = vec![f64::NAN; fixture.dst_len()];

    let device = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, true);

    assert!(
        device.iter().all(|value| !value.is_nan()),
        "overwrite left NaN behind: {device:?}"
    );
    assert_close(
        &device,
        &fixture.expected(&source, &poisoned, true),
        "poisoned destination around recoupling",
    );

    // Negative control: accumulation keeps the NaN, so the cleanliness above is
    // the Overwrite rule and not an artefact of the fixture.
    let accumulated = device_replay(&mut ctx, &mut executor, &fixture, &source, &poisoned, false);
    assert!(
        accumulated.iter().any(|value| value.is_nan()),
        "accumulation must keep the destination's NaN: {accumulated:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_replay_after_a_non_finite_one_reuses_the_workspace_cleanly() {
    // What: the pack/scatter workspace is execution scratch, not state — every
    // column is fully written by a pack (beta = 0) or by the GEMM (beta = 0)
    // before it is read — so a replay whose source held NaN cannot leave
    // anything behind that the next replay's result depends on. Without this,
    // a workspace kept across replays would be an invisible channel between
    // two unrelated transforms.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let clean = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];
    let expected = fixture.expected(&clean, &destination, true);

    let reference = device_replay(
        &mut ctx,
        &mut executor,
        &fixture,
        &clean,
        &destination,
        true,
    );
    assert_close(&reference, &expected, "clean replay before poisoning");

    let poisoned: Vec<f64> = clean
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index % 3 == 0 {
                f64::NAN
            } else if index % 3 == 1 {
                f64::INFINITY
            } else {
                *value
            }
        })
        .collect();
    let poisoned_result = device_replay(
        &mut ctx,
        &mut executor,
        &fixture,
        &poisoned,
        &destination,
        true,
    );
    // Negative control: the poisoned source really does reach the workspace and
    // the destination, so the recovery below is not vacuous.
    assert!(
        poisoned_result.iter().any(|value| !value.is_finite()),
        "the poisoned source never reached the destination: {poisoned_result:?}"
    );
    let workspace_bytes = executor.workspace_device_bytes();

    let recovered = device_replay(
        &mut ctx,
        &mut executor,
        &fixture,
        &clean,
        &destination,
        true,
    );

    assert_close(&recovered, &expected, "replay after a non-finite one");
    assert_eq!(
        executor.workspace_device_bytes(),
        workspace_bytes,
        "the recovery replay reallocated the workspace instead of reusing it"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn alternating_complex_recoupling_structures_upload_their_matrices_once_each() {
    // What: the per-(structure, dtype, context) key and the shared workspace
    // behave the same for a complex payload, where every coefficient and every
    // packed column is twice as wide.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixtures = [mixed_single_and_multi(), recoupling_non_symmetric_u()];
    let prepared: Vec<_> = fixtures
        .iter()
        .map(|fixture| (fixture.compile(), fixture.clone()))
        .collect();
    let mut buffers: Vec<_> = prepared
        .iter()
        .map(|(_, fixture)| {
            (
                CudaStorage::<Complex64>::upload(
                    &ctx,
                    &vec![Complex64::new(0.0, 0.0); fixture.dst_len()],
                )
                .unwrap(),
                CudaStorage::<Complex64>::upload(&ctx, &fixture.source::<Complex64>()).unwrap(),
            )
        })
        .collect();

    let before = cuda_transfer_stats();
    let mut after_first_round = before;
    for round in 0..3 {
        for (index, (structure, fixture)) in prepared.iter().enumerate() {
            let (dst, src) = &mut buffers[index];
            executor
                .replay(
                    &mut ctx,
                    structure,
                    &fixture.dst_structure(),
                    &fixture.src_structure(),
                    dst,
                    src,
                    Complex64::new(1.0, 0.0),
                    CudaTreeTransformDestination::Overwrite,
                )
                .unwrap();
        }
        if round == 0 {
            after_first_round = cuda_transfer_stats();
        }
    }
    let warm = stats_delta(after_first_round, cuda_transfer_stats());

    assert_eq!(executor.prepared_structures(), 2);
    assert_eq!(warm.h2d_calls, 0, "a switch re-uploaded: {warm:?}");
    assert_eq!(warm.device_allocs, 0, "a switch allocated: {warm:?}");
    for (index, (_, fixture)) in prepared.iter().enumerate() {
        let (dst, _) = &buffers[index];
        assert_close(
            &dst.download(&ctx).unwrap(),
            &fixture.expected(
                &fixture.source::<Complex64>(),
                &vec![Complex64::new(0.0, 0.0); fixture.dst_len()],
                true,
            ),
            fixture.name,
        );
    }
}

/// Elementwise equality that treats NaN as a value, for the tests that pin the
/// device against the host exactly rather than within a tolerance.
fn assert_same<T: TestScalar>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        let same = if left.is_nan() || right.is_nan() {
            left.is_nan() && right.is_nan()
        } else {
            left == right
        };
        assert!(
            same,
            "{what}: element {index} is {left:?}, expected {right:?}"
        );
    }
}

/// The fixtures the caller-scale sweep replays: a Single-block transform with a
/// coefficient that is neither 1 nor -1, a conjugated source, and a structure
/// mixing Single blocks, two recoupling groups and an inactive destination
/// layout — so the sweep covers every place the scale may and may not appear.
fn caller_scale_fixtures() -> Vec<Fixture> {
    vec![
        rank_sweep().remove(3),
        rank_sweep().remove(4),
        mixed_single_and_multi(),
        // A recoupling group read from a conjugated source: with a complex
        // scale this is the one fixture where a scale applied on the wrong side
        // of the conjugation, or folded into the pack, is visible in the
        // imaginary part.
        conjugated_recoupling(),
    ]
}

#[test]
#[ignore = "requires a real CUDA device"]
fn every_caller_scale_matches_the_host_and_the_oracle() {
    // What: the caller scale reaches Single moves and Multi scatters and
    // nothing else, in both destination modes, all four payload dtypes and for
    // alpha in {1, -2.5, 0, -0.0, complex}. The oracle is the explicit index
    // walk, which applies alpha where the host does and is pinned against the
    // host in CI, so a device that scaled the packs or the GEMM instead fails
    // here even though both ends would still be "some multiple of U x".
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in caller_scale_fixtures() {
        for overwrite in [true, false] {
            check_scales::<f32>(&mut ctx, &mut executor, &fixture, overwrite);
            check_scales::<f64>(&mut ctx, &mut executor, &fixture, overwrite);
            check_scales::<Complex32>(&mut ctx, &mut executor, &fixture, overwrite);
            check_scales::<Complex64>(&mut ctx, &mut executor, &fixture, overwrite);
        }
    }
}

fn check_scales<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
    overwrite: bool,
) {
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
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
            &fixture.expected_scaled(&source, &destination, overwrite, alpha),
            &format!("{what}: device vs oracle"),
        );
        assert_close(
            &device,
            &host_replay_scaled(fixture, &source, &destination, overwrite, alpha),
            &format!("{what}: device vs host"),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_caller_scale_writes_zeros_over_a_nan_source_as_the_host() {
    // What: alpha = 0 follows VectorInterface's `scale(x, 0) = zero(x) * 0`
    // (#1438), as TensorKit's `permute!(tdst, tsrc, p, 0, 0)` does: Overwrite
    // writes zeros over every layout whatever the source holds, and
    // Axpby(1) and Axpby(0) behave as `dst + 0` and Overwrite. Compared
    // with the host and the oracle, whose zero rule CI pins.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let poisoned = vec![f64::NAN; fixture.src_len()];
    let destination: Vec<f64> = (0..fixture.dst_len()).map(|i| 1.0 + i as f64).collect();

    for alpha in [0.0_f64, -0.0_f64] {
        let device = device_replay_scaled(
            &mut ctx,
            &mut executor,
            &fixture,
            &poisoned,
            &destination,
            true,
            alpha,
        );
        assert!(
            device.iter().all(|value| *value == 0.0),
            "a zero scale writes zeros whatever the source holds: {device:?}"
        );
        assert_same(
            &device,
            &host_replay_scaled(&fixture, &poisoned, &destination, true, alpha),
            &format!("zero scale over a NaN source, alpha = {alpha}"),
        );
        let accumulated = device_replay_scaled(
            &mut ctx,
            &mut executor,
            &fixture,
            &poisoned,
            &destination,
            false,
            alpha,
        );
        assert_eq!(accumulated, destination, "Axpby(1) with alpha = {alpha}");

        let mut dst = CudaStorage::<f64>::upload(&ctx, &vec![f64::NAN; fixture.dst_len()]).unwrap();
        let src = CudaStorage::<f64>::upload(&ctx, &poisoned).unwrap();
        executor
            .replay(
                &mut ctx,
                &fixture.compile(),
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                alpha,
                CudaTreeTransformDestination::Axpby(0.0),
            )
            .unwrap();
        assert!(
            dst.download(&ctx)
                .unwrap()
                .iter()
                .all(|value| *value == 0.0),
            "Axpby(0) is Overwrite: scale(NaN, 0) = 0"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_caller_scale_over_a_poisoned_destination_matches_the_host() {
    // What: Overwrite with alpha = 0 writes zeros over a NaN destination — but
    // *which* elements come back finite is the host's answer, not an assumption,
    // so this asserts against the host rather than against "all zero".
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let source = fixture.source::<f64>();
    let poisoned = vec![f64::NAN; fixture.dst_len()];

    let device = device_replay_scaled(
        &mut ctx,
        &mut executor,
        &fixture,
        &source,
        &poisoned,
        true,
        0.0,
    );

    assert_same(
        &device,
        &host_replay_scaled(&fixture, &source, &poisoned, true, 0.0),
        "zero scale over a poisoned destination",
    );
    // Negative control: accumulation keeps the destination's NaN, so the clean
    // result above is the Overwrite mode's doing and not the counters' silence.
    let accumulated = device_replay_scaled(
        &mut ctx,
        &mut executor,
        &fixture,
        &source,
        &poisoned,
        false,
        0.0,
    );
    assert!(
        accumulated.iter().all(|value| value.is_nan()),
        "accumulation must keep the destination's NaN: {accumulated:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_nan_caller_scale_reproduces_the_hosts_nan_pattern() {
    // What: a non-finite scale is an ordinary descriptor multiplication on both
    // ends, so the device must poison exactly the elements the host poisons and
    // leave the zero fills exact. Pinned against the host, whose own pattern is
    // pinned against the oracle in CI.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.dst_len()];

    let device = device_replay_scaled(
        &mut ctx,
        &mut executor,
        &fixture,
        &source,
        &destination,
        true,
        f64::NAN,
    );

    assert!(
        device.iter().any(|value| value.is_nan()),
        "a NaN scale must reach the written elements: {device:?}"
    );
    assert_same(
        &device,
        &host_replay_scaled(&fixture, &source, &destination, true, f64::NAN),
        "NaN scale",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_scale_sizes_the_template_of_a_structure_with_no_zero_fill() {
    // What: a structure whose destination layouts are all written sizes no zero
    // template of its own, so the zero fills a zero scale writes over its
    // written layouts are the only reason one exists. It must still be
    // reserved — before the first submission — and the result must be the
    // host's: zeros, NaN source included (#1438).
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = rank_sweep().remove(3);
    let source = fixture.source::<f64>();
    let poisoned = vec![f64::NAN; fixture.src_len()];
    let destination = vec![-1.0_f64; fixture.dst_len()];

    for alpha in [0.0_f64, -0.0] {
        let device = device_replay_scaled(
            &mut ctx,
            &mut executor,
            &fixture,
            &source,
            &destination,
            true,
            alpha,
        );
        assert_same(
            &device,
            &host_replay_scaled(&fixture, &source, &destination, true, alpha),
            "no zero fill, finite source",
        );
        let poisoned_device = device_replay_scaled(
            &mut ctx,
            &mut executor,
            &fixture,
            &poisoned,
            &destination,
            true,
            alpha,
        );
        assert!(
            poisoned_device.iter().all(|value| *value == 0.0),
            "a zero scale must not read the source: {poisoned_device:?}"
        );
        assert_same(
            &poisoned_device,
            &host_replay_scaled(&fixture, &poisoned, &destination, true, alpha),
            "no zero fill, NaN source",
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_replay_is_transfer_free_for_every_caller_scale() {
    // What: the caller scale is an execution-time argument — it is in no cache
    // key, and the zero fills a zero scale writes read the same context zero
    // template the inactive-layout fills read. So a warm replay stays
    // transfer-free and allocation-free for every scale including 0, and the
    // prepared-structure count does not grow.
    warm_scale_sweep::<f32>();
    warm_scale_sweep::<f64>();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_complex_replay_is_transfer_free_for_every_caller_scale() {
    // The same contract for the complex payloads, whose scale, coefficient
    // operand and zero template are a different dtype's buffers — the real-only
    // twin above would not notice a complex one uploaded per call. Each dtype
    // owns its own context operand slot, so each needs its own sweep.
    warm_scale_sweep::<Complex32>();
    warm_scale_sweep::<Complex64>();
}

fn warm_scale_sweep<T: DeviceScalar>() {
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let structure = fixture.compile();
    let source = fixture.source::<T>();
    let destination = vec![T::from_parts(0.0, 0.0); fixture.dst_len()];
    let mut dst = CudaStorage::<T>::upload(&ctx, &destination).unwrap();
    let src = CudaStorage::<T>::upload(&ctx, &source).unwrap();
    let replay = |ctx: &mut CudaDenseContext,
                  executor: &mut CudaTreeTransformExecutor,
                  dst: &mut CudaStorage<T>,
                  alpha: T| {
        executor
            .replay(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                dst,
                &src,
                alpha,
                CudaTreeTransformDestination::Overwrite,
            )
            .unwrap();
    };

    // Cold, and cold again with a zero scale: the first call uploads the
    // coefficients and sizes the zero template, the second may grow that
    // template to the largest written layout its zero fills cover.
    replay(&mut ctx, &mut executor, &mut dst, T::from_parts(1.0, 0.0));
    replay(&mut ctx, &mut executor, &mut dst, T::from_parts(0.0, 0.0));
    let structures = executor.prepared_structures();
    let workspace = executor.workspace_device_bytes();
    let before = cuda_transfer_stats();

    let last = T::from_parts(0.5, -1.25);
    for alpha in alphas::<T>() {
        replay(&mut ctx, &mut executor, &mut dst, alpha);
    }
    let warm = stats_delta(before, cuda_transfer_stats());

    assert_eq!(warm.h2d_calls, 0, "a warm scaled replay uploaded: {warm:?}");
    assert_eq!(
        warm.d2h_calls, 0,
        "a warm scaled replay downloaded: {warm:?}"
    );
    assert_eq!(
        warm.device_allocs, 0,
        "a warm scaled replay allocated: {warm:?}"
    );
    assert_eq!(executor.prepared_structures(), structures);
    assert_eq!(executor.workspace_device_bytes(), workspace);
    assert_close(
        &dst.download(&ctx).unwrap(),
        &fixture.expected_scaled(&source, &destination, true, last),
        "the last warm scaled replay",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_caller_scale_is_rejected_in_the_same_order() {
    // What: the caller scale is not an admission input — an unwritable
    // destination layout is still reported before any device work, with a
    // zero scale exactly as with a unit one.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let expert = expert_interleaved_destination();
    let expert_structure = expert.compile();
    let mut expert_dst =
        CudaStorage::<f64>::upload(&ctx, &vec![0.0_f64; expert.dst_len()]).unwrap();
    let expert_src = CudaStorage::<f64>::upload(&ctx, &expert.source::<f64>()).unwrap();

    reset_cuda_transfer_stats();
    let before = cuda_transfer_stats();

    let layout = executor
        .replay(
            &mut ctx,
            &expert_structure,
            &expert.dst_structure(),
            &expert.src_structure(),
            &mut expert_dst,
            &expert_src,
            0.0,
            CudaTreeTransformDestination::Overwrite,
        )
        .unwrap_err();
    assert!(
        matches!(
            layout,
            OperationError::UnsupportedDeviceTreeTransform { .. }
        ),
        "{layout:?}"
    );

    let delta = stats_delta(before, cuda_transfer_stats());
    assert_eq!(delta.h2d_calls, 0, "a rejection uploaded: {delta:?}");
    assert_eq!(delta.device_allocs, 0, "a rejection allocated: {delta:?}");
    assert_eq!(delta.gemm_calls, 0, "a rejection submitted: {delta:?}");
    assert_eq!(executor.prepared_structures(), 0);
}

/// θ = −1 on every other non-empty destination block, starting with the
/// first (so a one-block fixture is scaled too), as the sorted
/// `(offset, θ)` list `replay_with_destination_scales` takes (#1347).
/// Alternating per block, not per Multi group: the executor scales each
/// write by its own block's θ, which is exact whether or not a group is
/// uniform.
fn alternating_scales(fixture: &Fixture) -> Vec<(usize, f64)> {
    let structure = fixture.dst_structure();
    let mut scales: Vec<(usize, f64)> = (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .filter(|block| !block.shape().contains(&0))
        .enumerate()
        .filter(|(position, _)| position % 2 == 0)
        .map(|(_, block)| (block.offset(), -1.0))
        .collect();
    scales.sort_unstable_by_key(|&(offset, _)| offset);
    scales
}

/// The host's order: the unscaled-by-θ replay, then an in-place scale of each
/// listed block, by an explicit index walk over the block's shape and strides
/// (not a production kernel).
fn host_replay_then_scale<T: DeviceScalar>(
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    alpha: T,
    scales: &[(usize, f64)],
) -> Vec<T> {
    let mut data = host_replay_scaled(fixture, source, destination, true, alpha);
    let structure = fixture.dst_structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let Ok(position) = scales.binary_search_by_key(&block.offset(), |&(offset, _)| offset)
        else {
            continue;
        };
        let shape = block.shape();
        let total: usize = shape.iter().product();
        let mut multi = vec![0usize; shape.len()];
        for _ in 0..total {
            let element = block.offset()
                + multi
                    .iter()
                    .zip(block.strides())
                    .map(|(i, stride)| i * stride)
                    .sum::<usize>();
            data[element] = data[element].scale(scales[position].1);
            for (axis, value) in multi.iter_mut().enumerate() {
                *value += 1;
                if *value < shape[axis] {
                    break;
                }
                *value = 0;
            }
        }
    }
    data
}

#[allow(clippy::too_many_arguments)]
fn device_replay_with_scales<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
    source: &[T],
    destination: &[T],
    alpha: T,
    scales: &[(usize, f64)],
) -> Vec<T> {
    let structure = fixture.compile();
    let mut device_dst = CudaStorage::<T>::upload(ctx, destination).unwrap();
    let device_src = CudaStorage::<T>::upload(ctx, source).unwrap();
    executor
        .replay_with_destination_scales(
            ctx,
            &structure,
            &fixture.dst_structure(),
            &fixture.src_structure(),
            &mut device_dst,
            &device_src,
            alpha,
            CudaTreeTransformDestination::Overwrite,
            scales,
        )
        .unwrap();
    device_dst.download(ctx).unwrap()
}

fn check_destination_scales<T: DeviceScalar>(
    ctx: &mut CudaDenseContext,
    executor: &mut CudaTreeTransformExecutor,
    fixture: &Fixture,
) {
    let scales = alternating_scales(fixture);
    assert!(!scales.is_empty(), "{}: no block to scale", fixture.name);
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for alpha in [
        T::from_parts(1.0, 0.0),
        T::from_parts(0.0, 0.0),
        T::from_parts(-2.5, 0.0),
    ] {
        let what = format!("{} / {} / alpha = {alpha:?}", fixture.name, T::NAME);
        let device = device_replay_with_scales(
            ctx,
            executor,
            fixture,
            &source,
            &destination,
            alpha,
            &scales,
        );
        let expected = host_replay_then_scale(fixture, &source, &destination, alpha, &scales);
        assert_close(
            &device,
            &expected,
            &format!("{what}: device vs host + scale"),
        );
        if alpha != T::from_parts(0.0, 0.0) {
            // Negative control: θ really moved something.
            let unscaled = host_replay_scaled(fixture, &source, &destination, true, alpha);
            assert!(
                device
                    .iter()
                    .zip(&unscaled)
                    .any(|(left, right)| left.distance(*right) > 1e3 * move_tolerance::<T>()),
                "{what}: the scales changed nothing"
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn destination_scales_match_the_host_replay_followed_by_an_in_place_block_scale() {
    // What: θ_b reaches exactly the Single move or Multi scatter writing block
    // b — never a pack, the recoupling GEMM or a zero fill — for alpha in
    // {1, 0, -2.5} x θ, all four payload dtypes, Single, conjugated and
    // recoupling fixtures.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for fixture in caller_scale_fixtures() {
        check_destination_scales::<f32>(&mut ctx, &mut executor, &fixture);
        check_destination_scales::<f64>(&mut ctx, &mut executor, &fixture);
        check_destination_scales::<Complex32>(&mut ctx, &mut executor, &fixture);
        check_destination_scales::<Complex64>(&mut ctx, &mut executor, &fixture);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_caller_scale_with_destination_scales_writes_the_hosts_zeros() {
    // What: alpha = 0 takes the zero route whatever θ is, so a NaN source
    // does not reach the destination, as the host's θ * scale(x, 0) (#1438).
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let scales = alternating_scales(&fixture);
    let poisoned = vec![f64::NAN; fixture.src_len()];
    let destination = vec![0.0_f64; fixture.dst_len()];
    for alpha in [0.0_f64, -0.0_f64] {
        let device = device_replay_with_scales(
            &mut ctx,
            &mut executor,
            &fixture,
            &poisoned,
            &destination,
            alpha,
            &scales,
        );
        assert!(device.iter().all(|value| *value == 0.0), "{device:?}");
        assert_same(
            &device,
            &host_replay_then_scale(&fixture, &poisoned, &destination, alpha, &scales),
            &format!("alpha = {alpha} with destination scales over a NaN source"),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn destination_scales_reuse_the_unscaled_structure_and_transfer_nothing_warm() {
    // What: θ is in no key and uploads nothing — a scaled replay after an
    // unscaled one of the same structure prepares no structure and moves no
    // byte.
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let scales = alternating_scales(&fixture);
    let structure = fixture.compile();
    let source = fixture.source::<f64>();
    let mut dst = CudaStorage::<f64>::upload(&ctx, &vec![0.0; fixture.dst_len()]).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &source).unwrap();
    let mut replay = |ctx: &mut CudaDenseContext, scales: &[(usize, f64)]| {
        executor
            .replay_with_destination_scales(
                ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                1.0,
                CudaTreeTransformDestination::Overwrite,
                scales,
            )
            .unwrap();
    };
    replay(&mut ctx, &[]);
    replay(&mut ctx, &[]);
    let before = cuda_transfer_stats();
    replay(&mut ctx, &scales);
    replay(&mut ctx, &scales);
    let warm = stats_delta(before, cuda_transfer_stats());
    assert_eq!(warm.h2d_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 0, "{warm:?}");
    assert_eq!(executor.prepared_structures(), 1);
    assert_close(
        &dst.download(&ctx).unwrap(),
        &host_replay_then_scale(
            &fixture,
            &source,
            &vec![0.0; fixture.dst_len()],
            1.0,
            &scales,
        ),
        "warm scaled replay",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn unsorted_or_vanishing_destination_scales_are_rejected_before_any_device_work() {
    let mut ctx_engine = engine();
    let mut ctx = CudaDenseContext::new(&mut ctx_engine).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    let fixture = mixed_single_and_multi();
    let mut scales = alternating_scales(&fixture);
    assert!(scales.len() >= 2, "fixture has too few blocks");
    scales.swap(0, 1);
    let structure = fixture.compile();
    let mut dst = CudaStorage::<f64>::upload(&ctx, &vec![0.0; fixture.dst_len()]).unwrap();
    let src = CudaStorage::<f64>::upload(&ctx, &fixture.source::<f64>()).unwrap();
    let vanishing = vec![(alternating_scales(&fixture)[0].0, 0.0)];
    let before = cuda_transfer_stats();
    for (scales, expected) in [
        (&scales, "strictly increasing"),
        (&vanishing, "must not vanish"),
    ] {
        let error = executor
            .replay_with_destination_scales(
                &mut ctx,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut dst,
                &src,
                1.0,
                CudaTreeTransformDestination::Overwrite,
                scales,
            )
            .unwrap_err();
        assert!(
            matches!(error, OperationError::InvalidArgument { message } if message.contains(expected)),
            "{error:?}"
        );
    }
    assert_eq!(
        stats_delta(before, cuda_transfer_stats()),
        CudaTransferStats::default()
    );
    assert_eq!(executor.prepared_structures(), 0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_single_member_overwrite_refuses_a_foreign_source_before_zeroing_inactive_layouts() {
    // What: the source belongs to engine A, the destination and context to
    // engine B on the same device. The overwrite is refused before its
    // inactive-layout zero fill, so B's destination keeps every element.
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![3]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
    )
    .unwrap();
    let regions =
        CudaSingleMemberRegions::prepare(&transform, &structure, &structure, 5, 5, 5, 5, 1)
            .unwrap();
    assert_eq!(regions.zeros().len(), 1);
    let (mut first, mut second) = (engine(), engine());
    let source = {
        let a = CudaDenseContext::new(&mut first).unwrap();
        CudaStorage::upload_owned(&a, vec![1.0_f64; 5]).unwrap()
    };
    let mut b = CudaDenseContext::new(&mut second).unwrap();
    b.reserve_zero_template::<f64>(5).unwrap();
    let before_values = vec![7.0_f64, 8.0, 9.0, 10.0, 11.0];
    let mut destination = CudaStorage::upload_owned(&b, before_values.clone()).unwrap();
    let before = cuda_transfer_stats();
    assert!(regions
        .execute_overwrite(&mut b, &source, None, &mut destination)
        .is_err());
    assert_eq!(cuda_transfer_stats(), before, "nothing reached the device");
    assert_eq!(destination.download(&b).unwrap(), before_values);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_replay_refuses_a_foreign_source_at_every_scale_before_any_device_work() {
    // What: a cold executor on engine B, given B's destination and a source of
    // engine A on the same device, is refused by admission before it prepares
    // (uploads) any state or writes the destination — also at a zero scale,
    // whose arithmetic never reads the source.
    let fixture = rank_sweep().remove(3);
    let structure = fixture.compile();
    let (mut first, mut second) = (engine(), engine());
    let source = {
        let a = CudaDenseContext::new(&mut first).unwrap();
        CudaStorage::<f64>::upload(&a, &fixture.source::<f64>()).unwrap()
    };
    let mut b = CudaDenseContext::new(&mut second).unwrap();
    let before_values = vec![-1.0_f64; fixture.dst_len()];
    let mut destination = CudaStorage::<f64>::upload(&b, &before_values).unwrap();
    let mut executor = CudaTreeTransformExecutor::default();
    for alpha in [1.0_f64, 0.0] {
        let before = cuda_transfer_stats();
        assert!(executor
            .replay(
                &mut b,
                &structure,
                &fixture.dst_structure(),
                &fixture.src_structure(),
                &mut destination,
                &source,
                alpha,
                CudaTreeTransformDestination::Overwrite,
            )
            .is_err());
        assert_eq!(
            cuda_transfer_stats(),
            before,
            "alpha {alpha}: no device work"
        );
    }
    assert_eq!(destination.download(&b).unwrap(), before_values);
}
