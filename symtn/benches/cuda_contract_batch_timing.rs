//! Timing harness for the CUDA `ContractPlan` batch (Core, SwappedCore,
//! signed Core and DynamicTree routes) against the eager per-member loop.
//! Run on a free GPU with `cargo bench -p symtn --no-default-features
//! --features cuda,cpu-faer --bench cuda_contract_batch_timing`.
//!
//! Timings are evidence, never a CI threshold. Host elapsed times bracket host
//! submission without an explicit device synchronization, so they are not GPU
//! completion times. Host allocation counts cover the calling thread and CUDA
//! counters cover symtn's submission boundary, not provider-internal kernels
//! or allocations. Fresh output is already zeroed, so warm minus cold GEMM
//! submissions counts inactive-region zero writes.

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

use common::DeviceRule;
use contract_cases::{
    candidate_core_probes, fermion_u1, fermionic_twist_roles, fill, su2, u1_inactive_cases,
    u1_reordered, Case, FermionU1,
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::typed::{
    ContractPlan, CpuEngine, CudaDeviceId, CudaEngine, Engine, Runtime, StackedTensorMap, TensorMap,
};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[derive(Clone, Copy)]
struct Observation {
    elapsed: Duration,
    host_calls: usize,
    host_bytes: usize,
    cuda: CudaTransferStats,
}

impl std::fmt::Display for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host_elapsed={:?}/{} alloc/{} bytes/{:?}",
            self.elapsed, self.host_calls, self.host_bytes, self.cuda
        )
    }
}

fn observe<T>(run: impl FnOnce() -> T) -> (T, Observation) {
    let ((value, elapsed, before, after), host) = counting_alloc::measure(|| {
        let before = cuda_transfer_stats();
        let start = Instant::now();
        let value = run();
        let elapsed = start.elapsed();
        (value, elapsed, before, cuda_transfer_stats())
    });
    let (host_calls, host_bytes) = (host.calls as usize, host.bytes as usize);
    let cuda = CudaTransferStats {
        h2d_calls: after.h2d_calls - before.h2d_calls,
        h2d_bytes: after.h2d_bytes - before.h2d_bytes,
        d2h_calls: after.d2h_calls - before.d2h_calls,
        d2h_bytes: after.d2h_bytes - before.d2h_bytes,
        device_allocs: after.device_allocs - before.device_allocs,
        gemm_calls: after.gemm_calls - before.gemm_calls,
        solver_calls: after.solver_calls - before.solver_calls,
        copy_calls: after.copy_calls - before.copy_calls,
        gauge_ops: after.gauge_ops - before.gauge_ops,
    };
    (
        value,
        Observation {
            elapsed,
            host_calls,
            host_bytes,
            cuda,
        },
    )
}

fn median(mut samples: Vec<Observation>) -> Observation {
    samples.sort_by_key(|sample| sample.elapsed);
    samples[samples.len() / 2]
}

fn dynamic_tree_release_measurement() {
    fn run<R: DeviceRule>(cpu: &mut dyn Engine, cuda: &mut dyn Engine, case: &Case<R, f64>) {
        for count in [1, 2, 17] {
            let left: Vec<_> = (0..count)
                .map(|i| case.lhs.scale(&mut *cpu, 1.0 + i as f64 / 8.0).unwrap())
                .collect();
            let right: Vec<_> = (0..count)
                .map(|i| case.rhs.scale(&mut *cpu, 1.0 - i as f64 / 32.0).unwrap())
                .collect();
            let lhs = StackedTensorMap::pack(&left)
                .unwrap()
                .to_cuda(cuda)
                .unwrap();
            let rhs = StackedTensorMap::pack(&right)
                .unwrap()
                .to_cuda(cuda)
                .unwrap();
            let (plan, compile) =
                observe(|| ContractPlan::new(&mut *cuda, &lhs, &rhs, &case.spec()).unwrap());
            let (mut workspace, reserve) = observe(|| plan.workspace().unwrap());
            let (_, cold) = observe(|| {
                black_box(
                    plan.execute(&mut *cuda, &lhs, &rhs, &mut workspace)
                        .unwrap(),
                );
            });
            let median_of =
                |run: &mut dyn FnMut()| median((0..11).map(|_| observe(&mut *run).1).collect());
            let warm = median_of(&mut || {
                black_box(
                    plan.execute(&mut *cuda, &lhs, &rhs, &mut workspace)
                        .unwrap(),
                );
            });
            let dest = case.host(&mut *cpu);
            let mut dst = StackedTensorMap::pack(&vec![&dest; count])
                .unwrap()
                .to_cuda(&mut *cuda)
                .unwrap();
            let warm_into = median_of(&mut || {
                plan.execute_into(&mut *cuda, &lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
            });
            let eager_inputs: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(a, b)| {
                    (
                        a.to_cuda(&mut *cuda).unwrap(),
                        b.to_cuda(&mut *cuda).unwrap(),
                    )
                })
                .collect();
            let eager = median_of(&mut || {
                for (a, b) in &eager_inputs {
                    black_box(a.contract(&mut *cuda, b, &case.spec()).unwrap());
                }
            });
            eprintln!(
                "{} B={count} f64: compile={compile}; workspace={reserve}; cold={cold}; warm={warm}; warm_into={warm_into}; eager_members={eager}; retained_workspace_bytes={}",
                case.name,
                workspace.retained_bytes()
            );
        }
    }
    let cpu = &mut CpuEngine::serial();
    let cuda = &mut CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    let case = &u1_inactive_cases::<f64>(cpu, &runtime)[1];
    run(cpu, cuda, case);
    let case = u1_reordered::<f64>(cpu, &runtime);
    run(cpu, cuda, &case);
    for case in fermionic_twist_roles::<FermionU1, f64>(
        cpu,
        &runtime,
        &fermion_u1(),
        [
            "fZ2xU1 A copied",
            "fZ2xU1 canonical",
            "fZ2xU1 B copied",
            "fZ2xU1 both",
        ],
        71,
    ) {
        run(cpu, cuda, &case);
    }
}

fn direct_core_release_measurement() {
    let cpu = &mut CpuEngine::serial();
    let cuda = &mut CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    for (case, _) in candidate_core_probes::<_, f64>(cpu, &runtime, &su2()) {
        if !matches!(case.name, "C1" | "C2") {
            continue;
        }
        for count in [1, 2, 17] {
            let left: Vec<_> = (0..count)
                .map(|i| case.lhs.scale(&mut *cpu, 1.0 + i as f64 / 8.0).unwrap())
                .collect();
            let right: Vec<_> = (0..count)
                .map(|i| case.rhs.scale(&mut *cpu, 1.0 - i as f64 / 32.0).unwrap())
                .collect();
            let ((host_lhs, host_rhs), pack) = observe(|| {
                (
                    StackedTensorMap::pack(&left).unwrap(),
                    StackedTensorMap::pack(&right).unwrap(),
                )
            });
            let ((lhs, rhs), upload) = observe(|| {
                (
                    host_lhs.to_cuda(&mut *cuda).unwrap(),
                    host_rhs.to_cuda(&mut *cuda).unwrap(),
                )
            });
            let (plan, compile) =
                observe(|| ContractPlan::new(&mut *cuda, &lhs, &rhs, &case.spec()).unwrap());
            let (mut workspace, reserve) = observe(|| plan.workspace().unwrap());
            let (_, cold_returned) = observe(|| {
                black_box(
                    plan.execute(&mut *cuda, &lhs, &rhs, &mut workspace)
                        .unwrap(),
                );
            });
            let warm_returned = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            black_box(
                                plan.execute(&mut *cuda, &lhs, &rhs, &mut workspace)
                                    .unwrap(),
                            );
                        })
                        .1
                    })
                    .collect(),
            );
            let dest = case.host(&mut *cpu);
            let mut dst = StackedTensorMap::pack(&vec![&dest; count])
                .unwrap()
                .to_cuda(&mut *cuda)
                .unwrap();
            let (_, first_into) = observe(|| {
                plan.execute_into(&mut *cuda, &lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
                black_box(&dst);
            });
            let warm_into = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            plan.execute_into(&mut *cuda, &lhs, &rhs, &mut dst, &mut workspace)
                                .unwrap();
                            black_box(&dst);
                        })
                        .1
                    })
                    .collect(),
            );
            let eager_inputs: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(a, b)| {
                    (
                        a.to_cuda(&mut *cuda).unwrap(),
                        b.to_cuda(&mut *cuda).unwrap(),
                    )
                })
                .collect();
            let eager = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            for (a, b) in &eager_inputs {
                                black_box(a.contract(&mut *cuda, b, &case.spec()).unwrap());
                            }
                        })
                        .1
                    })
                    .collect(),
            );
            let retained = workspace.retained_bytes();
            let (_, teardown) = observe(|| {
                drop(workspace);
                drop(plan);
            });
            let zero_submissions = warm_returned.cuda.gemm_calls - cold_returned.cuda.gemm_calls;
            eprintln!("{} B={count} f64 CUDA0: pack={pack}; upload={upload}; compile={compile}; workspace={reserve}; cold_returned={cold_returned}; warm_returned={warm_returned}; first_into={first_into}; warm_into={warm_into}; eager_members={eager}; zero_region_submissions_per_warm={zero_submissions}; retained_workspace_bytes={retained}; teardown={teardown}", case.name);
        }
    }
}

fn signed_core_release_measurement() {
    let cpu = &mut CpuEngine::serial();
    let cuda = &mut CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    let space = fermion_u1();
    let dual = space.try_dual().unwrap();
    let a =
        TensorMap::<_, f64>::from_subblock_fn(&mut *cpu, &runtime, [&space], [&dual], fill(201))
            .unwrap();
    let b =
        TensorMap::<_, f64>::from_subblock_fn(&mut *cpu, &runtime, [&dual], [&space], fill(202))
            .unwrap();
    for swapped in [false, true] {
        let case = Case {
            name: if swapped {
                "signed swapped"
            } else {
                "signed direct"
            },
            lhs: if swapped { b.clone() } else { a.clone() },
            rhs: if swapped { a.clone() } else { b.clone() },
            lhs_axes: vec![usize::from(!swapped)],
            rhs_axes: vec![usize::from(swapped)],
            output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
            dense: false,
        };
        for count in [1, 2, 17] {
            let left: Vec<_> = (0..count)
                .map(|i| case.lhs.scale(&mut *cpu, 1.0 + i as f64 / 8.0).unwrap())
                .collect();
            let right: Vec<_> = (0..count)
                .map(|i| case.rhs.scale(&mut *cpu, 1.0 - i as f64 / 32.0).unwrap())
                .collect();
            let ((host_lhs, host_rhs), pack) = observe(|| {
                (
                    StackedTensorMap::pack(&left).unwrap(),
                    StackedTensorMap::pack(&right).unwrap(),
                )
            });
            let ((lhs, rhs), upload) = observe(|| {
                (
                    host_lhs.to_cuda(&mut *cuda).unwrap(),
                    host_rhs.to_cuda(&mut *cuda).unwrap(),
                )
            });
            let (plan, compile) =
                observe(|| ContractPlan::new(&mut *cuda, &lhs, &rhs, &case.spec()).unwrap());
            let (mut workspace, reserve) = observe(|| plan.workspace().unwrap());
            let (_, cold) = observe(|| {
                black_box(
                    plan.execute(&mut *cuda, &lhs, &rhs, &mut workspace)
                        .unwrap(),
                );
            });
            let warm = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            black_box(
                                plan.execute(&mut *cuda, &lhs, &rhs, &mut workspace)
                                    .unwrap(),
                            );
                        })
                        .1
                    })
                    .collect(),
            );
            let dest = case.host(&mut *cpu);
            let mut dst = StackedTensorMap::pack(&vec![&dest; count])
                .unwrap()
                .to_cuda(&mut *cuda)
                .unwrap();
            let (_, first_into) = observe(|| {
                plan.execute_into(&mut *cuda, &lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap()
            });
            let warm_into = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            plan.execute_into(&mut *cuda, &lhs, &rhs, &mut dst, &mut workspace)
                                .unwrap();
                        })
                        .1
                    })
                    .collect(),
            );
            let eager_inputs: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(a, b)| {
                    (
                        a.to_cuda(&mut *cuda).unwrap(),
                        b.to_cuda(&mut *cuda).unwrap(),
                    )
                })
                .collect();
            let eager = median(
                (0..11)
                    .map(|_| {
                        observe(|| {
                            for (a, b) in &eager_inputs {
                                black_box(a.contract(&mut *cuda, b, &case.spec()).unwrap());
                            }
                        })
                        .1
                    })
                    .collect(),
            );
            let retained = workspace.retained_bytes();
            let (_, teardown) = observe(|| {
                drop(workspace);
                drop(plan);
            });
            eprintln!("{} B={count} f64 CUDA0 unsynchronized_host_submission: pack={pack}; upload={upload}; compile={compile}; workspace={reserve}; cold={cold}; warm_median={warm}; first_into={first_into}; warm_into_median={warm_into}; eager_members={eager}; retained_workspace_bytes={retained}; teardown={teardown}",
                case.name);
        }
    }
}

fn main() {
    dynamic_tree_release_measurement();
    direct_core_release_measurement();
    signed_core_release_measurement();
}
