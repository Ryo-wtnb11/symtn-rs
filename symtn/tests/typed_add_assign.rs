use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, OperationError};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "support/report.rs"]
mod report;
use report::Report;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn tensor(runtime: &Runtime, n: usize, value: f64) -> TensorMap<U1FusionRule, f64> {
    let rule = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(rule, [(U1Irrep::new(0), n)]).unwrap();
    TensorMap::from_subblock_fn(runtime, [&space], [&space], move |_, ij| {
        if ij[0] == ij[1] {
            value
        } else {
            0.0
        }
    })
    .unwrap()
}

#[test]
fn scale_assign_mutates_unique_dense_payload() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut value = tensor(&runtime, 3, 2.0);
    value.scale_assign(3.0).unwrap();
    assert_eq!(
        value.dense_data().unwrap(),
        &[6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0]
    );
}

#[test]
fn axpby_into_updates_a_unique_destination_and_refuses_a_shared_one() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let x = tensor(&runtime, 2, 3.0);
    let mut destination = tensor(&runtime, 2, 2.0);
    x.axpby_into(&mut destination, -1.0, 2.0).unwrap();
    assert_eq!(destination.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);

    // A clone shares the payload: writing it would change the clone too, and
    // replacing it would allocate behind the caller's back.
    let clone = destination.clone();
    assert_eq!(
        x.axpby_into(&mut destination, 1.0, 1.0),
        Err(Error::DestinationShared)
    );
    assert_eq!(destination.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(clone.dense_data().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn axpby_into_rejects_a_layout_mismatch_without_mutating_the_destination() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut destination = tensor(&runtime, 2, 2.0);
    let x = tensor(&runtime, 3, 3.0);
    let error = x.axpby_into(&mut destination, 1.0, 1.0).unwrap_err();
    assert!(
        matches!(
            &error,
            Error::Operation(operation)
                if matches!(**operation, OperationError::SpaceMismatch { .. })
        ),
        "{}",
        error.report()
    );
    assert_eq!(destination.dense_data().unwrap(), &[2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn unique_dense_destinations_are_allocation_free() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mut destination = tensor(&runtime, 64, 2.0);
    let x = tensor(&runtime, 64, 3.0);
    let ((), allocs) =
        counting_alloc::measure(|| x.axpby_into(&mut destination, -1.0, 2.0).unwrap());
    assert_eq!(allocs.calls, 0);
    let mut scaled = tensor(&runtime, 64, 2.0);
    let ((), allocs) = counting_alloc::measure(|| scaled.scale_assign(3.0).unwrap());
    assert_eq!(allocs.calls, 0);
}
