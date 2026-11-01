//! Host half of the Runtime device-state contract, run without a device and
//! without the `cuda` feature so ordinary CI covers it: a Runtime built
//! without a device reports no device transform state and still clears its
//! Host transform store.

use symtn::typed::Engine;

use std::sync::Arc;

use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::legs::u1_leg_2_1_2;

/// The fill `typed_cuda_transform.rs` uses.
fn real_fill<S: std::fmt::Debug>(_trees: &symtn::typed::BlockFusionTrees<S>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

#[test]
fn the_semantic_clear_empties_the_host_transform_caches() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Isolated: the completed-transformer counters are process-global.
    if run_isolated_or_return(
        "SYMTN_TYPED_TRANSFORM_HOST_SIDE_ISOLATED",
        "the_semantic_clear_empties_the_host_transform_caches",
    ) {
        return;
    }
    // The semantic clear empties the Host-side caches, including when the
    // `cuda` feature is compiled in.
    let runtime = Runtime::builder().build().unwrap();

    let v = u1_leg_2_1_2();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v, &v], real_fill).unwrap();
    let _ = tensor.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let entries = completed_transformers().entries();
    assert!(entries > 0);
    symtn::cache::clear();

    assert_eq!(completed_transformers().entries(), 0);
}

// ---------------------------------------------------------------------------
// The Host contract the device `*_into` mirrors
// ---------------------------------------------------------------------------

/// The positions that hold a NaN, so a NaN *pattern* can be compared rather
/// than merely "some NaN survived".
fn nan_positions(data: &[f64]) -> Vec<usize> {
    data.iter()
        .enumerate()
        .filter(|(_, value)| value.is_nan())
        .map(|(index, _)| index)
        .collect()
}

/// The Host precondition order and wording that `typed_cuda_transform_contracts
/// .rs` mirrors on the device. Pinned here, without a device and without the
/// `cuda` feature, so ordinary CI catches a Host drift that would silently
/// make the device mirror wrong rather than failing it.
///
/// Only the two storage nouns differ on device ("host" becomes "CUDA"), which
/// is why this test spells them out.
#[test]
fn the_host_into_preconditions_have_a_fixed_order_and_wording() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_leg_2_1_2();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v, &v], real_fill).unwrap();
    // Fresh, uniquely owned destinations: a `clone()` shares the payload `Arc`
    // and would trip the unique-ownership check instead.
    let destination = |engine: &mut dyn Engine| source.permute(engine, &[2, 0], &[1, 3]).unwrap();

    let message = |error: symtn::typed::Error| {
        std::iter::successors(Some(&error as &dyn std::error::Error), |&error| {
            error.source()
        })
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
    };

    // Runtime mismatch precedes a rule mismatch.
    let other = Runtime::builder().build().unwrap();
    let z2 = Arc::new(symtn::sector::ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(symtn::sector::ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2)]).unwrap();
    let zn_fill = |_: &_, indices: &[usize]| indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>();
    let z2_source =
        TensorMap::from_subblock_fn(engine, &runtime, [&z2_leg], [&z2_leg], zn_fill).unwrap();
    let mut foreign =
        TensorMap::from_subblock_fn(engine, &other, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    assert_eq!(
        z2_source
            .permute_into(engine, &[0], &[1], &mut foreign, 1.0, 0.0)
            .unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    );

    // Rule mismatch precedes a lazy-adjoint source.
    let mut z3_destination =
        TensorMap::from_subblock_fn(engine, &runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN)
            .unwrap();
    assert_eq!(
        z2_source
            .adjoint(engine)
            .unwrap()
            .permute_into(engine, &[0], &[1], &mut z3_destination, 1.0, 0.0)
            .unwrap_err(),
        symtn::typed::Error::RuleMismatch
    );

    // A lazy-adjoint source is admitted, so a lazy-adjoint
    // destination is what it reports.
    let lazy_source = source.adjoint(engine).unwrap();
    let mut lazy_destination = destination(engine).adjoint(engine).unwrap();
    assert_eq!(
        message(
            lazy_source
                .permute_into(engine, &[2, 0], &[1, 3], &mut lazy_destination, 1.0, 0.0)
                .unwrap_err()
        ),
        "invalid argument: destination must use ordinary dense host storage"
    );
    assert_eq!(
        message(
            source
                .permute_into(engine, &[2, 0], &[1, 3], &mut lazy_destination, 1.0, 0.0)
                .unwrap_err()
        ),
        "invalid argument: destination must use ordinary dense host storage"
    );

    // The alias check precedes the operation build, so malformed axes do not
    // mask it.
    let mut alias = source.clone();
    assert_eq!(
        message(
            source
                .permute_into(engine, &[0, 0], &[1, 3], &mut alias, 1.0, 0.0)
                .unwrap_err()
        ),
        "invalid argument: destination storage must not alias an input"
    );

    // The space check precedes the unique-ownership check.
    let mut wrong_space = source.transpose(engine, &[3, 2], &[1, 0]).unwrap();
    let wrong_space_handle = wrong_space.clone();
    assert_eq!(
        message(
            source
                .permute_into(engine, &[2, 0], &[1, 3], &mut wrong_space, 1.0, 0.0)
                .unwrap_err()
        ),
        "tensor operation failed: space mismatch: destination fusion space or block layout \
         does not match the operation result"
    );
    drop(wrong_space_handle);

    let mut shared = destination(engine);
    let shared_handle = shared.clone();
    assert_eq!(
        message(
            source
                .permute_into(engine, &[2, 0], &[1, 3], &mut shared, 1.0, 0.0)
                .unwrap_err()
        ),
        "the destination shares its storage with another tensor handle"
    );
    drop(shared_handle);

    // `repartition_into` onto a destination of another rank.
    let mut rank_three =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v], |_, _| f64::NAN).unwrap();
    assert_eq!(
        message(
            source
                .repartition_into(engine, &mut rank_three, 1.0, 0.0)
                .unwrap_err()
        ),
        "invalid argument: repartition destination rank 3 does not match source rank 4"
    );
}

#[test]
fn host_into_clears_a_poisoned_destination_and_zero_scales_to_zeros() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // The three Host destination semantics: Overwrite clears whatever the
    // destination held, `alpha == 0` writes zeros whatever the source holds,
    // and an identity axis list is still written rather than short
    // circuited.
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_leg_2_1_2();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v, &v], real_fill).unwrap();

    let mut poisoned = source
        .permute(engine, &[2, 0], &[1, 3])
        .unwrap()
        .scale(engine, f64::NAN)
        .unwrap();
    assert!(poisoned
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_nan()));
    source
        .permute_into(engine, &[2, 0], &[1, 3], &mut poisoned, 1.0, 0.0)
        .unwrap();
    assert!(
        poisoned
            .dense_data()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()),
        "Overwrite mode must clear every destination layout, inactive ones included"
    );
    assert_eq!(
        poisoned.dense_data().unwrap(),
        source
            .permute(engine, &[2, 0], &[1, 3])
            .unwrap()
            .dense_data()
            .unwrap()
    );

    // No identity short circuit: `alpha * self` is written.
    let mut identity = source.scale(engine, f64::NAN).unwrap();
    source
        .permute_into(engine, &[0, 1], &[2, 3], &mut identity, -2.5, 0.0)
        .unwrap();
    assert_eq!(
        identity.dense_data().unwrap(),
        source.scale(engine, -2.5).unwrap().dense_data().unwrap()
    );
    let mut same_split = source.scale(engine, f64::NAN).unwrap();
    source
        .repartition_into(engine, &mut same_split, 2.0, 0.0)
        .unwrap();
    assert_eq!(
        same_split.dense_data().unwrap(),
        source.scale(engine, 2.0).unwrap().dense_data().unwrap()
    );

    // `alpha == 0`, `-0.0` included, writes VectorInterface's
    // `scale(x, 0) = zero(x) * 0`: an exact zero at every position,
    // NaN sources included, as TensorKit's `permute!(tdst, tsrc, p, 0.0,
    // Zero())` does (observed: `0.0 + 0.0im` over a `NaN + 1.0im`
    // destination and an `Inf + 1.0im` source).
    let nan_source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v, &v], |trees, idx| {
            if idx.iter().sum::<usize>() % 3 == 0 {
                f64::NAN
            } else {
                real_fill(trees, idx)
            }
        })
        .unwrap();
    let permuted_source = nan_source.permute(engine, &[1, 2], &[3, 0]).unwrap();
    assert!(
        !nan_positions(permuted_source.dense_data().unwrap()).is_empty(),
        "the fixture must carry NaNs through the permute"
    );
    for alpha in [0.0, -0.0] {
        let mut destination = nan_source.permute(engine, &[1, 2], &[3, 0]).unwrap();
        nan_source
            .permute_into(engine, &[1, 2], &[3, 0], &mut destination, alpha, 0.0)
            .unwrap();
        assert!(
            destination
                .dense_data()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0),
            "alpha = {alpha}: every position must be an exact zero"
        );
    }
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}

/// Re-executes exactly one global-cache probe with fresh-process cache state.
fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
    if std::env::var_os(isolated_env).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("typed_transform_host_side::{test_path}"),
            "--include-ignored",
        ])
        .env(isolated_env, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
        "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    true
}
