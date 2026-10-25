//! Allocation contract of the batched failure rule (#1757): the buffers a
//! failed call keeps are reused, so the first call after a failure allocates
//! no more on the caller thread than a warm call. The allocation target
//! shares one counting allocator; single-threaded runtimes mean the
//! caller thread sees every symtn allocation.

use crate::prepared;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use symtn::typed::{
    ComposePlan, ContractPlan, ContractSpec, EighFullPlan, HermitianTol, Runtime, StackedTensorMap,
};

use prepared::eigh::hermitian_members;
use prepared::{members, u1_legs};

use crate::counting_alloc;

const COUNT: usize = 5;

fn allocations(f: impl FnOnce()) -> (u64, u64) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls, allocs.bytes)
}

/// `(cold, warm, after a failed call)` of `call`, with `fail` run in between.
fn sequence(mut call: impl FnMut(), fail: impl FnOnce()) -> [(u64, u64); 3] {
    let cold = allocations(&mut call);
    let warm = allocations(&mut call);
    fail();
    let recovered = allocations(&mut call);
    [cold, warm, recovered]
}

fn check(label: &str, [cold, warm, recovered]: [(u64, u64); 3]) {
    eprintln!("{label}: cold={cold:?} warm={warm:?} after failure={recovered:?} (calls, bytes)");
    assert!(cold.0 > warm.0, "{label}: the counter sees the cold call");
    assert!(
        recovered.0 <= warm.0 && recovered.1 <= warm.1,
        "{label}: recovery {recovered:?} allocates more than warm {warm:?}"
    );
}

#[test]
fn the_first_call_after_a_failure_allocates_no_more_than_a_warm_call() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], COUNT, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], COUNT, 2)).unwrap();

    let compose = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = compose.workspace().unwrap();
    let workspace = std::cell::RefCell::new(&mut workspace);
    check(
        "compose",
        sequence(
            || {
                compose
                    .execute(&lhs, &rhs, &mut workspace.borrow_mut())
                    .unwrap();
            },
            || {
                assert!(compose
                    .execute(&rhs, &lhs, &mut workspace.borrow_mut())
                    .is_err())
            },
        ),
    );

    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 1],
        domain: &[2],
    };
    let contract = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
    let mut workspace = contract.workspace().unwrap();
    let workspace = std::cell::RefCell::new(&mut workspace);
    check(
        "contract",
        sequence(
            || {
                contract
                    .execute(&lhs, &rhs, &mut workspace.borrow_mut())
                    .unwrap();
            },
            || {
                assert!(contract
                    .execute(&rhs, &lhs, &mut workspace.borrow_mut())
                    .is_err())
            },
        ),
    );

    let source = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], COUNT, 7)).unwrap();
    let other = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&w], 2, 9)).unwrap();
    let eigh = EighFullPlan::new(&source, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = eigh.workspace().unwrap();
    let workspace = std::cell::RefCell::new(&mut workspace);
    check(
        "eigh",
        sequence(
            || {
                eigh.execute(&source, &mut workspace.borrow_mut()).unwrap();
            },
            || assert!(eigh.execute(&other, &mut workspace.borrow_mut()).is_err()),
        ),
    );
}
