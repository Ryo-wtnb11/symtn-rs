//! A reused executor's warm grouped GEMM batch on four lanes, counted on
//! every thread of the process: on a target whose std mutex locks without
//! allocating it allocates nothing; on Apple targets, where each lane's fresh
//! std mutex allocates its pthread mutex on first lock (one per lane per
//! call, the disclosed runner residual until threadteam#37 offers a
//! mutable-share fork), repeated warm calls allocate the same and no more.
//!
//! Its own test binary, so the shared counting allocator's process-wide
//! measurement sees only this test.
//! That the fixture really runs on four lanes is asserted in the crate's
//! `the_warm_allocation_fixture_runs_on_four_lanes`, which can read the
//! placement.

#![cfg(all(
    feature = "cpu-faer",
    not(feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]

use symtn_dense::{
    strided_batch_runs, CpuEngine, DefaultDenseExecutor, DenseExecutor, DenseGemmBatchJob,
    DenseRead, DenseScalar, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/started_pool.rs"]
mod started_pool;
#[path = "support/warm_lanes.rs"]
mod warm_lanes;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn warm_multi_lane_grouped_batch_allocates_nothing_but_the_lane_locks() {
    // Every worker has started, so no worker start-up lands in a window.
    let pool = started_pool::started_pool(warm_lanes::THREADS);
    // `Outer` places the nonzero-estimate jobs on one-slot lanes.
    let engine =
        &mut CpuEngine::new(pool).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    let jobs = warm_lanes::jobs();
    let runs = strided_batch_runs(&jobs);
    let (lhs, rhs) = ([1.0; warm_lanes::LEN], [2.0; warm_lanes::LEN]);
    let mut output = [0.0; warm_lanes::LEN];
    let mut executor = DefaultDenseExecutor::new();
    let mut call = || {
        counting_alloc::measure_process(|| {
            executor
                .matmul_batch_axpby_with_ops_into(
                    engine,
                    DenseWrite::F64(
                        DenseViewMut::new(&mut output, &[warm_lanes::LEN], &[1], 0).unwrap(),
                    ),
                    DenseRead::F64(DenseView::new(&lhs, &[warm_lanes::LEN], &[1], 0).unwrap()),
                    DenseRead::F64(DenseView::new(&rhs, &[warm_lanes::LEN], &[1], 0).unwrap()),
                    &jobs,
                    &runs,
                    MatrixOp::Identity,
                    MatrixOp::Identity,
                    DenseScalar::F64(1.0),
                    DenseScalar::F64(0.0),
                )
                .unwrap()
        })
        .1
    };
    call();
    let warm = call();
    if cfg!(target_vendor = "apple") {
        assert_eq!(call(), warm, "warm calls grow");
    } else {
        assert_eq!(warm, 0);
    }
}
