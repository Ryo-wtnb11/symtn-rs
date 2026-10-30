//! Allocation contract of the Host `EighFullPlan` (#1744): a warm call
//! allocates per member exactly what the dense solver allocates for that
//! member's coupled sectors, and nothing else. The allocation target
//! shares one counting allocator; every dense kernel runs on the
//! caller thread, so the counter sees all of them.

use crate::prepared;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use std::hint::black_box;
use symtn::expert::{DefaultDenseExecutor, DenseExecutor, DenseRead, DenseView};
use symtn::typed::{EighFullPlan, HermitianTol, Runtime, StackedTensorMap};

use prepared::eigh::hermitian_members;
use prepared::u1_legs;

use crate::counting_alloc;

/// Allocation calls of one warm `execute` over `count` members.
fn warm_calls(runtime: &Runtime, count: usize) -> (u64, Vec<usize>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (leg, _) = u1_legs();
    let stack =
        StackedTensorMap::pack(&hermitian_members(runtime, &[&leg, &leg], count, 5)).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0, 1], &[2, 3], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(engine, &stack, &mut workspace).unwrap();
    plan.execute(engine, &stack, &mut workspace).unwrap();
    let ((), allocs) = counting_alloc::measure(|| {
        black_box(plan.execute(engine, &stack, &mut workspace).unwrap());
    });
    let orders = plan
        .execute(engine, &stack, &mut workspace)
        .unwrap()
        .spectra[0]
        .iter()
        .map(|entry| entry.values.len())
        .collect();
    (allocs.calls, orders)
}

/// Allocation calls of one dense `eigh` per order, on a serial engine.
fn solver_calls(orders: &[usize]) -> u64 {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = DefaultDenseExecutor::new();
    let matrices: Vec<Vec<f64>> = orders
        .iter()
        .map(|&n| {
            (0..n * n)
                .map(|k| 1.0 / (1.0 + (k % n + k / n) as f64))
                .collect()
        })
        .collect();
    let mut solve = |dense: &mut DefaultDenseExecutor| {
        for (matrix, &n) in matrices.iter().zip(orders) {
            let (shape, strides) = ([n, n], [1, n]);
            let view = DenseView::new(matrix.as_slice(), &shape, &strides, 0).unwrap();
            black_box(dense.eigh(engine, DenseRead::F64(view)).unwrap());
        }
    };
    solve(&mut dense);
    let ((), allocs) = counting_alloc::measure(|| solve(&mut dense));
    allocs.calls
}

#[test]
fn warm_execute_allocates_per_member_only_the_dense_solver_outputs() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (small, orders) = warm_calls(&runtime, 4);
    let (large, _) = warm_calls(&runtime, 12);
    let solver = solver_calls(&orders);
    eprintln!(
        "warm calls: B=4 {small}, B=12 {large}; dense eigh per member {solver} over orders {orders:?}"
    );
    assert!(solver > 0, "the counter sees the solver");
    // Why one more per sector than `solver_calls`: Tenferro 0.8's faer eigh
    // checks its scratch out of the engine's buffer pool
    // (`tenferro-linalg src/cpu/tlinalg.rs` through
    // `tenferro-cpu-basic pooled_uninit_output`). Measured: the plan's
    // member replay takes one fresh checkout per sector, which the repeated
    // standalone sequence of `solver_calls` gets back from the pool. Engine
    // scratch, not a symtn temporary.
    assert_eq!(
        large - small,
        8 * (solver + orders.len() as u64),
        "per member, a warm call allocates only the dense solver's outputs and workspace"
    );
}
