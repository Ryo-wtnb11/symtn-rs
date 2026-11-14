use crate::su2::{su2_irrep, su2_rule, Su2Rule};
use std::hint::black_box;
use std::sync::Arc;

use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

#[test]
fn repartition_to_current_split_does_not_allocate() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a repartition which leaves the boundary unchanged only clones Arc
    // handles, preserves provider authority, and performs no heap allocation.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(su2_rule());
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [(su2_irrep(0), 1), (su2_irrep(1), 2)],
    )
    .unwrap();
    let source: TensorMap<Su2Rule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 191).unwrap();

    black_box(source.repartition(engine, source.codomain_rank()).unwrap());
    let (output, allocs) = counting_alloc::measure(|| {
        black_box(source.repartition(engine, source.codomain_rank()).unwrap())
    });
    black_box(&output);

    assert_eq!(allocs.calls, 0);
    assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    assert_eq!(
        output.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    );
}
