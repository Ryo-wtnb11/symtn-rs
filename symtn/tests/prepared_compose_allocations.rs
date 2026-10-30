//! Warm Host compose workspace and plan/workspace allocation contract:
//! after one call at
//! a fixed `B`, `execute` and `execute_into` allocate nothing of symtn's on
//! the caller thread. The one remaining allocation is Tenferro 0.7.1's
//! grouped-GEMM validation (`tenferro-tensor src/backend.rs:
//! validate_grouped_gemm`), which every grouped submission pays, eager
//! included. The allocation target installs one shared counting allocator.

// The call budgets are measured on the faer provider; a BLAS provider owns
// its own scratch allocations (grouped-GEMM metadata, QR/LQ workspaces), which
// these gates do not model (#2107).
use crate::backend;

use crate::prepared;

use std::time::Instant;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use symtn::typed::{ComposePlan, Runtime, StackedTensorMap};

use prepared::{filled, members, u1_legs};

use crate::counting_alloc;

fn allocations(f: impl FnOnce()) -> (usize, usize, std::time::Duration) {
    let (elapsed, allocs) = counting_alloc::measure(|| {
        let started = Instant::now();
        f();
        started.elapsed()
    });
    (allocs.calls as usize, allocs.bytes as usize, elapsed)
}

#[test]
fn warm_host_calls_allocate_only_the_backend_grouped_validation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    if !backend::faer_is_default_backend() {
        return;
    }
    // What: at a fixed B, a second `execute` and a second `execute_into`
    // (whose plan has inactive destination blocks to zero-fill) reuse the
    // workspace's output, job list and fill strides, and the Runtime's pooled
    // context. The zero fill adds nothing (`execute_into` == `execute`), and
    // both stay within the one per-submission allocation of Tenferro's
    // grouped validator. The cold call's count is the control that shows the
    // counter sees this thread's allocations.
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let count = 5;
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], count, 2)).unwrap();
    let mut dst = StackedTensorMap::pack(&filled::<_, f64>(
        &runtime,
        &[&v, &v],
        &[&v],
        count,
        f64::NAN,
    ))
    .unwrap();
    let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut ws = plan.workspace().unwrap();

    let cold = allocations(|| {
        plan.execute(engine, &lhs, &rhs, &mut ws).unwrap();
    });
    assert!(
        cold.0 > 0,
        "the cold call allocates its output and job list"
    );
    plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut ws)
        .unwrap();

    let warm = allocations(|| {
        plan.execute(engine, &lhs, &rhs, &mut ws).unwrap();
    });
    let warm_into = allocations(|| {
        plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut ws)
            .unwrap();
    });
    assert!(warm.0 <= 1, "warm execute allocations: {warm:?}");
    assert_eq!(
        (warm_into.0, warm_into.1),
        (warm.0, warm.1),
        "the zero fill of execute_into allocates nothing"
    );
    assert!(cold.0 > warm.0 + 2, "cold {cold:?} against warm {warm:?}");

    let plan_runtime = Runtime::builder().build().unwrap();
    let plan_lhs = StackedTensorMap::pack(&members::<_, f64>(
        &plan_runtime,
        &[&v, &v],
        &[&w],
        count,
        1,
    ))
    .unwrap();
    let plan_rhs =
        StackedTensorMap::pack(&members::<_, f64>(&plan_runtime, &[&w], &[&v], count, 2)).unwrap();
    let plan = ComposePlan::new(engine, &plan_lhs, &plan_rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let plan_cold = allocations(|| {
        plan.execute(engine, &plan_lhs, &plan_rhs, &mut workspace)
            .unwrap();
    });
    let plan_warm = allocations(|| {
        plan.execute(engine, &plan_lhs, &plan_rhs, &mut workspace)
            .unwrap();
    });
    assert_eq!((plan_warm.0, plan_warm.1), (warm.0, warm.1));
    eprintln!(
        "B={count} first runtime cold={cold:?} warm={warm:?}; fresh runtime cold={plan_cold:?} warm={plan_warm:?}"
    );
}
