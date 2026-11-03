//! A configured execution context reaches every contraction a network call
//! makes: `contract!` (including its scalar readback) and
//! `Network::contract` run the context's lane allocation, not an engine's
//! default.
//!
//! On the native provider this is observed through the process-global
//! lane-allocation cache, which keys an allocation by its lane allocation:
//! a call that carries another allocation than the cached one searches. A
//! BLAS provider has no lane allocator, so the same call is refused there.

use std::sync::Arc;

use symtn::cache::StructureCacheKind;
use symtn::expert::DenseError;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    Configured, CpuEngine, Error, GradedSpace, LaneAllocation, OperationError, Pool, Runtime,
    TensorMap,
};
use symtn_network::{contract, Network};

const LANE_ALLOCATOR: bool = !cfg!(any(
    feature = "cpu-blas",
    feature = "blas-accelerate",
    feature = "blas-openblas",
    feature = "blas-mkl",
    feature = "provider-inject"
));

fn context(allocation: LaneAllocation) -> Configured<CpuEngine> {
    Configured::new(CpuEngine::new(Pool::builder().threads(4).build().unwrap()))
        .with_lane_allocation(allocation)
}

fn lane_misses() -> u64 {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::LaneAllocations)
        .unwrap()
        .misses()
}

fn is_refused(error: &Error) -> bool {
    matches!(
        error,
        Error::Operation(error) if matches!(
            **error,
            OperationError::Dense(DenseError::Unsupported { op: "lane_allocation", .. })
        )
    )
}

/// What: `contract!` and `Network::contract` forward a configured context:
/// after a default call, the other allocation searches its own lane
/// allocation on the native provider (and is refused on a provider without
/// the allocator), and the results agree with the default's. The scalar form
/// runs on the same context; its contraction has no grouped batch, so it
/// agrees with the default on every provider.
#[test]
fn network_calls_forward_a_configured_context() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    // Unequal blocks large enough that a four-thread engine searches lanes.
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-3..=3).map(|charge: i32| {
            (
                U1Irrep::new(charge),
                48 - 8 * charge.unsigned_abs() as usize,
            )
        }),
    )
    .unwrap();
    let seed = &mut CpuEngine::serial();
    let a = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 1).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 2).unwrap();
    let network = Network::new(
        [["i", "j"], ["j", "k"]],
        [false, false],
        [1, 1],
        ["i", "k"],
        Some(1),
    )
    .unwrap();
    let (default, equal) = (
        &mut context(LaneAllocation::Moldable),
        &mut context(LaneAllocation::EqualShare),
    );

    symtn::cache::clear();
    let want = contract!(default, [i; k] := a[i; j] * b[j; k]).unwrap();
    let want_scalar = contract!(default, a[i; j] * b[j; i]).unwrap();
    let mut searched = |call: &mut dyn FnMut() -> Result<TensorMap<U1FusionRule, f64>, Error>| {
        symtn::cache::clear();
        contract!(default, [i; k] := a[i; j] * b[j; k]).unwrap();
        let before = lane_misses();
        let result = call();
        (result, lane_misses() > before)
    };
    let by_macro = searched(&mut || contract!(equal, [i; k] := a[i; j] * b[j; k]));
    let by_network = searched(&mut || network.contract(equal, &[&a, &b]));
    let scalar = contract!(equal, a[i; j] * b[j; i]);
    for (what, (result, searched)) in [("contract!", by_macro), ("Network::contract", by_network)] {
        if LANE_ALLOCATOR {
            assert!(searched, "{what}: the other allocation searched");
            let got = result.unwrap();
            for (x, y) in got
                .dense_data()
                .unwrap()
                .iter()
                .zip(want.dense_data().unwrap())
            {
                assert!(
                    (x - y).abs() <= 1e-10 * (1.0 + y.abs()),
                    "{what}: {x} vs {y}"
                );
            }
        } else {
            assert!(is_refused(&result.unwrap_err()), "{what}");
        }
    }
    // The full contraction to a scalar has no grouped GEMM batch, so every
    // provider runs it with the configured context as with the default.
    let scalar = scalar.unwrap();
    assert!((scalar - want_scalar).abs() <= 1e-10 * (1.0 + want_scalar.abs()));
}
