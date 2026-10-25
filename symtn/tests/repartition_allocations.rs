use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

#[test]
fn repartition_to_current_split_does_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    // What: a repartition which leaves the boundary unchanged only clones Arc
    // handles, preserves provider authority, and performs no heap allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let source: TensorMap<SU2FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 191).unwrap();

    black_box(source.repartition(source.codomain_rank()).unwrap());
    let (output, allocs) =
        counting_alloc::measure(|| black_box(source.repartition(source.codomain_rank()).unwrap()));
    black_box(&output);

    assert_eq!(allocs.calls, 0);
    assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    assert_eq!(
        output.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
}
