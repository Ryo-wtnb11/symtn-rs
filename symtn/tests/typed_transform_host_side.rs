//! Host-side gates of the device structural transform that need
//! no CUDA device — and, deliberately, no `cuda` feature either, so ordinary
//! CI runs them.
//!
//! Two things live here:
//!
//! 1. the pin that makes the dense physical-basis oracle *independent*
//!    evidence. `typed_cuda_transform.rs` compares a device permute against
//!    `common::permute_dense` of the source's `to_physical_dense()`; that is
//!    only an oracle if the same function is first shown to agree with the
//!    Host for the providers it is used on;
//! 2. the Host half of the Runtime device-state contract: a Runtime built
//!    without a device reports no device transform state and still clears its
//!    Host transform store.

use crate::common;
use symtn::typed::Engine;

use std::sync::Arc;
use symtn::typed::Direction;

use common::permute_dense;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::numerics;

use crate::legs::{su2_leg_2_2_1, u1_leg_2_1_2};

/// Floating terms per physical entry: the fusion-tree expansion of a rank-4
/// SU(2) tensor with spins up to 1 sums over at most four intermediate spins.
const TERMS: usize = 4;

/// The slice-wide `numerics` bound also covers the zero entries, which this
/// oracle once held to `1e-12`; the shared fill must stay small enough that
/// the bound does not exceed it (`max|entry| <= 70` at `TERMS`).
fn assert_bound_within_entry_floor(data: &[f64]) {
    let scale = data.iter().fold(0.0_f64, |max, value| max.max(value.abs()));
    assert!(
        numerics::tolerance::<f64>(TERMS, scale) <= 1e-12,
        "fill scale {scale}"
    );
}

/// The fill `typed_cuda_transform.rs` uses, so the pin and the device gate
/// exercise the same values.
fn real_fill<S: std::fmt::Debug>(_trees: &symtn::typed::BlockFusionTrees<S>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

#[test]
fn the_dense_permute_oracle_agrees_with_the_host_for_u1_and_su2() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();

    let u1 = u1_leg_2_1_2();
    let u1_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let source = u1_tensor.to_physical_dense(engine).unwrap();
    let permuted = u1_tensor.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = permuted.to_physical_dense(engine).unwrap();
    assert_eq!(actual.shape, shape, "U(1) oracle shape");
    assert_bound_within_entry_floor(&data);
    numerics::assert_slices_close("U(1) oracle", &actual.data, &data, TERMS);

    // SU(2) is the case that matters: the reduced-block replay must recouple
    // to produce what is, physically, a bare axis permutation.
    let su2 = su2_leg_2_2_1();
    let su2_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&su2, &su2], [&su2, &su2], real_fill)
            .unwrap();
    let source = su2_tensor.to_physical_dense(engine).unwrap();
    let permuted = su2_tensor.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let (shape, data) = permute_dense(&source.shape, &source.data, &[1, 0, 3, 2]);
    let actual = permuted.to_physical_dense(engine).unwrap();
    assert_eq!(actual.shape, shape, "SU(2) oracle shape");
    assert_bound_within_entry_floor(&data);
    numerics::assert_slices_close("SU(2) oracle", &actual.data, &data, TERMS);

    // Non-vacuity: the reduced payload is not a reordering of the source, so
    // the oracle really did survive a recoupling.
    let sorted = |data: &[f64]| {
        let mut values = data.to_vec();
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        values
    };
    assert_ne!(
        sorted(su2_tensor.dense_data().unwrap()),
        sorted(permuted.dense_data().unwrap()),
        "SU(2) permute must apply coefficients other than 1"
    );
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

// ---------------------------------------------------------------------------
// Device twist: the Host-side decisions its lowering rests on
// ---------------------------------------------------------------------------

/// Every twist of `$tensor` over `$cases` keeps an entry or negates it, and
/// does both at least once — the `±1` domain, written once so each provider
/// is checked identically. A macro rather than a generic function: the typed
/// `twist` bound names the admission mode, which no test needs to spell out.
macro_rules! assert_signs_only_over {
    ($engine:ident, $what:expr, $tensor:expr, $cases:expr) => {
        for legs in $cases {
            for twisted in [
                $tensor.twist($engine, legs, Direction::Forward).unwrap(),
                $tensor.twist($engine, legs, Direction::Inverse).unwrap(),
            ] {
                let mut kept = 0usize;
                let mut negated = 0usize;
                for (&before, &after) in $tensor
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(twisted.dense_data().unwrap())
                {
                    if after == before {
                        kept += 1;
                    } else {
                        assert_eq!(
                            after, -before,
                            "{} twist {legs:?} scaled by something else",
                            $what
                        );
                        negated += 1;
                    }
                }
                assert!(negated > 0, "{} twist {legs:?} changed nothing", $what);
                assert!(
                    kept > 0,
                    "{} twist {legs:?} scaled every entry, so no unscaled block is covered",
                    $what
                );
            }
        }
    };
}

/// The device `twist` puts the per-block factor on the contraction
/// descriptor's own scale instead of uploading a factor table. That is only
/// sound because, for every provider the device impl admits (`Scalar = f64`),
/// a ribbon twist is a real sign and never zero — a zero would be rejected by
/// `cuda_region_axpby`, and would also let CUDA skip the source read.
///
/// This pins the value domain observationally, without a device: a Host twist
/// may keep an entry or negate it, and nothing else. It also pins that some
/// blocks really do keep their factor of one, which is why the device has to
/// move every block rather than only the scaled ones.
#[test]
fn a_fermionic_twist_only_ever_keeps_or_negates_an_entry() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();

    // The fixtures are the device gate's own: fZ2 with a dual leg, and the two
    // product providers, whose factors are products of the two rules' own.
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &dual], [&leg, &leg], real_fill)
            .unwrap();
    assert_signs_only_over!(
        engine,
        "fZ2",
        tensor,
        [
            &[0usize][..],
            &[1][..],
            &[3][..],
            &[0, 3][..],
            &[1, 2, 3][..]
        ]
    );

    // fZ2 x U(1).
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn::sector::FermionParityFusionRule,
            U1FusionRule,
        )
        .unwrap(),
    );
    let leg_u1 = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (
                symtn::sector::product_sector(symtn::sector::Z2Irrep::EVEN, U1Irrep::new(0)),
                2,
            ),
            (
                symtn::sector::product_sector(symtn::sector::Z2Irrep::ODD, U1Irrep::new(1)),
                1,
            ),
            (
                symtn::sector::product_sector(symtn::sector::Z2Irrep::EVEN, U1Irrep::new(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let product_u1: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg_u1, &leg_u1],
        [&leg_u1, &leg_u1],
        real_fill,
    )
    .unwrap();
    assert_signs_only_over!(
        engine,
        "fZ2 x U(1)",
        product_u1,
        [&[1usize][..], &[2][..], &[0, 2][..]]
    );

    // fZ2 (x) SU(2): the same sign domain with non-Abelian degeneracies.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn::sector::FermionParityFusionRule,
            SU2FusionRule,
        )
        .unwrap(),
    );
    let leg_su2 = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (
                symtn::sector::product_sector(
                    symtn::sector::Z2Irrep::EVEN,
                    SU2Irrep::from_twice_spin(0),
                ),
                2,
            ),
            (
                symtn::sector::product_sector(
                    symtn::sector::Z2Irrep::ODD,
                    SU2Irrep::from_twice_spin(1),
                ),
                1,
            ),
            (
                symtn::sector::product_sector(
                    symtn::sector::Z2Irrep::EVEN,
                    SU2Irrep::from_twice_spin(2),
                ),
                1,
            ),
        ],
    )
    .unwrap();
    let product_su2: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg_su2, &leg_su2],
        [&leg_su2, &leg_su2],
        real_fill,
    )
    .unwrap();
    assert_signs_only_over!(
        engine,
        "fZ2 (x) SU(2)",
        product_su2,
        [&[0usize][..], &[3][..], &[1, 3][..]]
    );

    // Twisting *every* leg is the identity in value on a parity-conserving
    // block: the factors multiply to the block's total parity, which is even.
    // It is not a short circuit — the detection tests each leg's own factor,
    // not the product — so the device gate expects the ordinary per-block
    // work here, and this pins the value it must produce.
    for legs in [&[0usize, 1, 2, 3][..], &[0, 1, 2, 3, 0, 1, 2, 3][..]] {
        assert_eq!(
            tensor
                .twist(engine, legs, Direction::Forward)
                .unwrap()
                .dense_data()
                .unwrap(),
            tensor.dense_data().unwrap()
        );
        assert_eq!(
            tensor
                .twist(engine, legs, Direction::Inverse)
                .unwrap()
                .dense_data()
                .unwrap(),
            tensor.dense_data().unwrap()
        );
    }

    // An inverse twist undoes a twist exactly, which is what lets the device
    // gate compare the two runs for equality rather than to a tolerance.
    let round_trip = tensor
        .twist(engine, &[0, 2], Direction::Forward)
        .unwrap()
        .twist(engine, &[0, 2], Direction::Inverse)
        .unwrap();
    assert_eq!(
        round_trip.dense_data().unwrap(),
        tensor.dense_data().unwrap()
    );
}

/// The zero-block fixture of the device gate. A space with no coupled sector
/// has no block, so `twist_is_identity_over_blocks` is vacuously true and the
/// call short-circuits to a clone on both Host and device: it proves the
/// degenerate space is handled, not that the zero-length upload path runs.
/// Reaching that path needs blocks with a zero extent, which is a device-only
/// distinction and is left untested.
#[test]
fn a_twist_of_a_space_with_no_coupled_sector_is_the_identity_short_circuit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let even = GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [(symtn::sector::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    let odd = GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [(symtn::sector::Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&even], [&odd], real_fill).unwrap();
    assert!(
        empty.dense_data().unwrap().is_empty(),
        "the fixture must carry no element"
    );
    assert!(empty
        .twist(engine, &[0, 1], Direction::Forward)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
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
