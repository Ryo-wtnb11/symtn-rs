//! Timing harness for the Host `ContractPlan` batch (copyC, Core,
//! SwappedCore, signed direct and dynamic source-twist routes) against the
//! eager per-member loop. Run with `cargo bench -p symtn --bench
//! contract_batch_timing`.
//!
//! Timings and caller-thread allocation counts are evidence, never a CI
//! threshold.

#[path = "../tests/common/mod.rs"]
mod common;
#[path = "../../tests/support"]
mod support {
    use num_complex::{Complex32, Complex64};
    pub mod numerics;
}
use support::numerics;
#[macro_use]
#[allow(unused_macros)]
#[path = "../tests/contract_cases/mod.rs"]
mod contract_cases;
#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use contract_cases::{
    candidate_core_probes, fermion_u1, fermionic_twist_roles, fill, su2, su2_reordered, u1, Case,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use symtn::typed::{ContractPlan, ContractSpec, Runtime, StackedTensorMap, TensorMap};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn measure<T>(f: impl FnOnce() -> T) -> (T, Duration, usize, usize) {
    let ((value, elapsed), allocs) = counting_alloc::measure(|| {
        let start = Instant::now();
        let value = f();
        (value, start.elapsed())
    });
    (value, elapsed, allocs.calls as usize, allocs.bytes as usize)
}

fn public_copy_c_batch_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1(&[(0, 2), (1, 2)]);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 4).unwrap();
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[1, 0],
        domain: &[2, 3],
    };
    let (_, eager_cold, calls, bytes) = measure(|| black_box(a.contract(&b, &spec).unwrap()));
    eprintln!("copyC eager B=1 cold: {eager_cold:?}, {calls} calls, {bytes} bytes");
    a.contract(&b, &spec).unwrap();
    let (_, eager_warm, eager_warm_calls, eager_warm_bytes) =
        measure(|| black_box(a.contract(&b, &spec).unwrap()));
    eprintln!(
        "copyC eager B=1 warm: {eager_warm:?}, {eager_warm_calls} calls, {eager_warm_bytes} bytes"
    );
    for count in [1, 2, 17] {
        let (_, eager_batch_time, eager_batch_calls, eager_batch_bytes) = measure(|| {
            let results: Vec<_> = (0..count).map(|_| a.contract(&b, &spec).unwrap()).collect();
            black_box(results)
        });
        let mut eager_destinations: Vec<_> =
            (0..count).map(|_| a.contract(&b, &spec).unwrap()).collect();
        let (_, eager_into_time, eager_into_calls, eager_into_bytes) = measure(|| {
            for dst in &mut eager_destinations {
                a.contract_into(&b, &spec, dst, 1.0, 0.0).unwrap();
            }
        });
        let left = vec![&a; count];
        let right = vec![&b; count];
        let ((lhs, rhs), pack_time, pack_calls, pack_bytes) = measure(|| {
            (
                StackedTensorMap::pack(&left).unwrap(),
                StackedTensorMap::pack(&right).unwrap(),
            )
        });
        let (plan, plan_time, plan_calls, plan_bytes) =
            measure(|| ContractPlan::new(&lhs, &rhs, &spec).unwrap());
        let (mut workspace, ws_time, ws_calls, ws_bytes) = measure(|| plan.workspace().unwrap());
        let (_, cold_time, cold_calls, cold_bytes) =
            measure(|| black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap()));
        let (_, warm_time, warm_calls, warm_bytes) =
            measure(|| black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap()));
        eprintln!("copyC B={count}: eager returned {eager_batch_time:?}/{eager_batch_calls}/{eager_batch_bytes}, eager into {eager_into_time:?}/{eager_into_calls}/{eager_into_bytes}, pack {pack_time:?}/{pack_calls}/{pack_bytes}, plan {plan_time:?}/{plan_calls}/{plan_bytes}, workspace {ws_time:?}/{ws_calls}/{ws_bytes}, cold {cold_time:?}/{cold_calls}/{cold_bytes}, warm {warm_time:?}/{warm_calls}/{warm_bytes}, retained {} bytes", workspace.retained_bytes());
    }
}

fn public_host_batch_cold_warm_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let case = su2_reordered::<f64>(&runtime);
    let (_, eager_time, eager_calls, eager_bytes) =
        measure(|| black_box(case.lhs.contract(&case.rhs, &case.spec()).unwrap()));
    eprintln!("eager B=1 cold: {eager_time:?}, {eager_calls} calls, {eager_bytes} bytes");
    for count in [1, 2, 17] {
        let left = vec![&case.lhs; count];
        let right = vec![&case.rhs; count];
        let lhs = StackedTensorMap::pack(&left).unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap();
        let ((plan, mut workspace), cold_time, cold_calls, cold_bytes) = measure(|| {
            let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
            let mut workspace = plan.workspace().unwrap();
            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            (plan, workspace)
        });
        let mut samples = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            samples.push((elapsed, calls, bytes));
        }
        samples.sort_by_key(|sample| sample.0);
        let (warm_time, warm_calls, warm_bytes) = samples[15];
        eprintln!("B={count}: cold {cold_time:?} {cold_calls} calls {cold_bytes} bytes; warm median {warm_time:?} {warm_calls} calls {warm_bytes} bytes; retained {} bytes", workspace.retained_bytes());
    }
    let mut eager = Vec::new();
    for _ in 0..31 {
        let (_, elapsed, calls, bytes) = measure(|| {
            black_box(case.lhs.contract(&case.rhs, &case.spec()).unwrap());
        });
        eager.push((elapsed, calls, bytes));
    }
    eager.sort_by_key(|sample| sample.0);
    eprintln!(
        "eager B=1 warm median: {:?}, {} calls, {} bytes",
        eager[15].0, eager[15].1, eager[15].2
    );
}

fn public_host_swapped_core_release_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let case = candidate_core_probes::<_, f64>(&runtime, &su2())
        .into_iter()
        .find(|(case, _)| case.name == "C2")
        .unwrap()
        .0;
    for count in [1, 2, 17] {
        let ((lhs, rhs), pack_time, pack_calls, pack_bytes) = measure(|| {
            let left = vec![&case.lhs; count];
            let right = vec![&case.rhs; count];
            (
                StackedTensorMap::pack(&left).unwrap(),
                StackedTensorMap::pack(&right).unwrap(),
            )
        });
        let (plan, plan_time, plan_calls, plan_bytes) =
            measure(|| ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap());
        let (mut workspace, workspace_time, workspace_calls, workspace_bytes) =
            measure(|| plan.workspace().unwrap());
        let (_, cold_time, cold_calls, cold_bytes) = measure(|| {
            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        });
        let mut samples = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            samples.push((elapsed, calls, bytes));
        }
        samples.sort_by_key(|sample| sample.0);
        let (warm_time, warm_calls, warm_bytes) = samples[15];
        eprintln!("SwappedCore B={count}: pack {pack_time:?}/{pack_calls}/{pack_bytes}; plan {plan_time:?}/{plan_calls}/{plan_bytes}; workspace {workspace_time:?}/{workspace_calls}/{workspace_bytes}; cold execute {cold_time:?}/{cold_calls}/{cold_bytes}; warm median {warm_time:?}/{warm_calls}/{warm_bytes}; retained {} bytes", workspace.retained_bytes());
    }
    let mut samples = Vec::new();
    for _ in 0..31 {
        let (_, elapsed, calls, bytes) = measure(|| {
            black_box(case.host());
        });
        samples.push((elapsed, calls, bytes));
    }
    samples.sort_by_key(|sample| sample.0);
    eprintln!(
        "eager B=1 warm median: {:?}/{} calls/{} bytes",
        samples[15].0, samples[15].1, samples[15].2
    );
}

fn public_signed_direct_release_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = fermion_u1();
    let dual = v.try_dual().unwrap();
    let lhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&dual], fill(201)).unwrap();
    let rhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&v], fill(202)).unwrap();
    let case = Case {
        name: "signed direct fZ2xU1",
        lhs,
        rhs,
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    for count in [1, 2, 17] {
        let (pair, pack_time, pack_calls, pack_bytes) = measure(|| {
            let left = vec![&case.lhs; count];
            let right = vec![&case.rhs; count];
            (
                StackedTensorMap::pack(&left).unwrap(),
                StackedTensorMap::pack(&right).unwrap(),
            )
        });
        let (lhs, rhs) = pair;
        let (plan, plan_time, plan_calls, plan_bytes) =
            measure(|| ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap());
        let (mut workspace, workspace_time, workspace_calls, workspace_bytes) =
            measure(|| plan.workspace().unwrap());
        let (_, cold_time, cold_calls, cold_bytes) = measure(|| {
            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        });
        let mut returned = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            returned.push((elapsed, calls, bytes));
        }
        returned.sort_by_key(|sample| sample.0);
        let dst_member = case.host();
        let dst_refs = vec![&dst_member; count];
        let mut dst = StackedTensorMap::pack(&dst_refs).unwrap();
        let mut into = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
                black_box(&dst);
            });
            into.push((elapsed, calls, bytes));
        }
        into.sort_by_key(|sample| sample.0);
        let mut eager = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                for _ in 0..count {
                    black_box(case.host());
                }
            });
            eager.push((elapsed, calls, bytes));
        }
        eager.sort_by_key(|sample| sample.0);
        eprintln!("SignedDirect B={count}: pack {pack_time:?}/{pack_calls}/{pack_bytes}; plan {plan_time:?}/{plan_calls}/{plan_bytes}; workspace {workspace_time:?}/{workspace_calls}/{workspace_bytes}; cold {cold_time:?}/{cold_calls}/{cold_bytes}; returned warm median {:?}; into warm median {:?}; eager warm median {:?}; retained {} bytes", returned[15], into[15], eager[15], workspace.retained_bytes());
    }
}

fn public_dynamic_source_twist_release_measurement() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let case = fermionic_twist_roles::<_, f64>(
        &runtime,
        &fermion_u1(),
        ["A", "canonical", "B", "both"],
        71,
    )
    .into_iter()
    .next()
    .unwrap();
    for count in [1, 2, 17] {
        let ((lhs, rhs), pack_time, pack_calls, pack_bytes) = measure(|| {
            let left = vec![&case.lhs; count];
            let right = vec![&case.rhs; count];
            (
                StackedTensorMap::pack(&left).unwrap(),
                StackedTensorMap::pack(&right).unwrap(),
            )
        });
        let (plan, plan_time, plan_calls, plan_bytes) =
            measure(|| ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap());
        let (mut workspace, workspace_time, workspace_calls, workspace_bytes) =
            measure(|| plan.workspace().unwrap());
        let (_, cold_time, cold_calls, cold_bytes) = measure(|| {
            black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        });
        let mut returned = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                black_box(plan.execute(&lhs, &rhs, &mut workspace).unwrap());
            });
            returned.push((elapsed, calls, bytes));
        }
        returned.sort_by_key(|sample| sample.0);
        let dest_member = case.host();
        let dest_refs = vec![&dest_member; count];
        let mut destination = StackedTensorMap::pack(&dest_refs).unwrap();
        let mut into = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                plan.execute_into(&lhs, &rhs, &mut destination, &mut workspace)
                    .unwrap();
                black_box(&destination);
            });
            into.push((elapsed, calls, bytes));
        }
        into.sort_by_key(|sample| sample.0);
        let mut eager = Vec::new();
        for _ in 0..31 {
            let (_, elapsed, calls, bytes) = measure(|| {
                for _ in 0..count {
                    black_box(case.host());
                }
            });
            eager.push((elapsed, calls, bytes));
        }
        eager.sort_by_key(|sample| sample.0);
        eprintln!("DynamicSourceTwist B={count}: pack {pack_time:?}/{pack_calls}/{pack_bytes}; plan {plan_time:?}/{plan_calls}/{plan_bytes}; workspace {workspace_time:?}/{workspace_calls}/{workspace_bytes}; cold {cold_time:?}/{cold_calls}/{cold_bytes}; returned warm median {:?}; into warm median {:?}; eager warm median {:?}; retained {} bytes", returned[15], into[15], eager[15], workspace.retained_bytes());
    }
}

fn main() {
    public_copy_c_batch_measurement();
    public_host_batch_cold_warm_measurement();
    public_host_swapped_core_release_measurement();
    public_signed_direct_release_measurement();
    public_dynamic_source_twist_release_measurement();
}
