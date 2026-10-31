//! Allocation contract of the batched failure rule (#1757): the buffers a
//! failed call keeps are reused, so the first call after a failure allocates
//! no more on the caller thread than a warm call. The allocation target
//! shares one counting allocator; single-threaded runtimes mean the
//! caller thread sees every symtn allocation.

use crate::common::ErrorKind as _;
use crate::prepared;
use symtn::typed::Engine;

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
fn sequence(
    engine: &mut dyn Engine,
    mut call: impl FnMut(&mut dyn Engine),
    fail: impl FnOnce(&mut dyn Engine),
) -> [(u64, u64); 3] {
    let cold = allocations(|| call(engine));
    let warm = allocations(|| call(engine));
    // Why a BLAS build runs one more warm call: its eigh checks a workspace
    // and an output out of the engine's best-fit buffer pool
    // (`tlinalg-blas` eigh through `tenferro-cpu-basic` `buffer_pool`), and
    // the warm calls alternate between two pool states, measured (124,
    // 50803) and (126, 50843) without any failure (Tenferro 0.7.1 alternates
    // too). The recovered call is compared with the warm call of its own
    // phase. The faer eigh has no such checkout, so its warm state is steady.
    if !cfg!(feature = "cpu-faer") {
        call(engine);
    }
    fail(engine);
    let recovered = allocations(|| call(engine));
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], COUNT, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], COUNT, 2)).unwrap();

    let compose = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut workspace = compose.workspace().unwrap();
    let workspace = std::cell::RefCell::new(&mut workspace);
    check(
        "compose",
        sequence(
            engine,
            |engine| {
                compose
                    .execute(engine, &lhs, &rhs, &mut workspace.borrow_mut())
                    .unwrap();
            },
            |engine| {
                assert!(compose
                    .execute(engine, &rhs, &lhs, &mut workspace.borrow_mut())
                    .is_batch_signature_mismatch())
            },
        ),
    );

    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 1],
        domain: &[2],
    };
    let contract = ContractPlan::new(engine, &lhs, &rhs, &spec).unwrap();
    let mut workspace = contract.workspace().unwrap();
    let workspace = std::cell::RefCell::new(&mut workspace);
    check(
        "contract",
        sequence(
            engine,
            |engine| {
                contract
                    .execute(engine, &lhs, &rhs, &mut workspace.borrow_mut())
                    .unwrap();
            },
            |engine| {
                assert!(contract
                    .execute(engine, &rhs, &lhs, &mut workspace.borrow_mut())
                    .is_batch_signature_mismatch())
            },
        ),
    );

    let source = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], COUNT, 7)).unwrap();
    let other = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&w], 2, 9)).unwrap();
    let eigh = EighFullPlan::new(engine, &source, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = eigh.workspace().unwrap();
    let workspace = std::cell::RefCell::new(&mut workspace);
    check(
        "eigh",
        sequence(
            engine,
            |engine| {
                eigh.execute(engine, &source, &mut workspace.borrow_mut())
                    .unwrap();
            },
            |engine| {
                assert!(matches!(
                    eigh.execute(engine, &other, &mut workspace.borrow_mut()),
                    Err(symtn::typed::BatchError::Operation(
                        symtn::typed::Error::BatchSignatureMismatch { .. }
                    ))
                ))
            },
        ),
    );
}
