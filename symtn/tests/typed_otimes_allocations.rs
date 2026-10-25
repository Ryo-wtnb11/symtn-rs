//! Allocation gate for the multiplicity-free tensor-product executor.

use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

fn measured<T>(payload_bytes: usize, operation: impl FnOnce() -> T) -> (u64, usize) {
    let (output, allocs) =
        counting_alloc::measure_matching(payload_bytes..=payload_bytes, || black_box(operation()));
    black_box(output);
    (allocs.bytes, allocs.matched_calls as usize)
}

fn leg(rule: &Arc<U1FusionRule>, degeneracy: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::clone(rule), [(U1Irrep::new(0), degeneracy)]).unwrap()
}

#[test]
fn typed_otimes_allocates_one_output_payload_and_no_dense_kron_temporary() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let lhs_codomain = leg(&rule, 24);
    let lhs_domain = leg(&rule, 20);
    let rhs_codomain = leg(&rule, 18);
    let rhs_domain = leg(&rule, 16);
    let lhs =
        TensorMap::from_subblock_fn(&runtime, [&lhs_codomain], [&lhs_domain], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, _| 3.0).unwrap();
    let warm = lhs.otimes(&rhs).unwrap();
    let output_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    drop(warm);

    let (allocated, output_sized_allocations) =
        measured(output_bytes, || lhs.otimes(&rhs).unwrap());

    assert_eq!(output_sized_allocations, 1);
    assert!(
        allocated <= output_bytes as u64 + 128 * 1024,
        "otimes allocated {allocated} B for a {output_bytes} B output"
    );
}
