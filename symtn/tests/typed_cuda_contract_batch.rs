//! Public C1/Core and C2/SwappedCore CUDA batch gates. Run on A100 with
//! `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test typed_cuda_contract_batch -- --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

mod braiding_probe;
mod common;
#[path = "../../tests/support"]
mod support {
    use num_complex::{Complex32, Complex64};
    pub mod numerics;
}
use support::numerics;
#[macro_use]
#[allow(unused_macros)]
mod contract_cases;

use braiding_probe::{ProbeSector, RealBraidingProbe};
use common::{DevicePayload, DeviceRule};
use contract_cases::{
    blas_contract_oracle, candidate_core_probes, dense_oracle, fermion_su2, fermion_u1,
    fermionic_blas_contract_oracle, fermionic_blas_contract_oracle_partitioned,
    fermionic_canonical_nonuniform, fermionic_twist_roles, fill, poisoned_destination, su2,
    su2_bent, u1, u1_inactive_cases, u1_non_self_dual, u1_reordered, Case, FermionU1, TwistRole,
};
use num_complex::Complex64;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};
use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{
    ContractPlan, ContractSpec, ContractWorkspace, CudaStorage, Direction, Error, GradedSpace,
    Runtime, StackedTensorMap, TensorMap,
};

fn check_signed_core<R: DeviceRule, D: DevicePayload>(
    runtime: &Runtime,
    space: &GradedSpace<R>,
    swapped: bool,
) {
    let dual = space.try_dual().unwrap();
    let a = TensorMap::<R, D>::from_subblock_fn(runtime, [space], [&dual], fill(201)).unwrap();
    let b = TensorMap::<R, D>::from_subblock_fn(runtime, [&dual], [space], fill(202)).unwrap();
    let base = Case {
        name: "signed CUDA core",
        lhs: if swapped { b.clone() } else { a.clone() },
        rhs: if swapped { a } else { b },
        lhs_axes: vec![usize::from(!swapped)],
        rhs_axes: vec![usize::from(swapped)],
        output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
        dense: false,
    };
    let first_lhs = StackedTensorMap::pack(&[&base.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let first_rhs = StackedTensorMap::pack(&[&base.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &base.spec()).unwrap();
    let reserved_before = runtime
        .cuda_plan_cache_stats()
        .unwrap()
        .unwrap()
        .reserved_entries;
    let mut workspace = plan.workspace().unwrap();
    let mut other = plan.workspace().unwrap();
    let mut expected_gemms = None;
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| Case {
                name: base.name,
                lhs: base
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                    .unwrap(),
                rhs: base
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap(),
                lhs_axes: base.lhs_axes.clone(),
                rhs_axes: base.rhs_axes.clone(),
                output_axes: base.output_axes.clone(),
                dense: false,
            })
            .collect();
        let left: Vec<_> = members.iter().map(|m| &m.lhs).collect();
        let right: Vec<_> = members.iter().map(|m| &m.rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap().to_cuda().unwrap();
        let (_, cold) = observe(|| plan.execute(&lhs, &rhs, &mut workspace).unwrap());
        assert!(cold.cuda.h2d_calls >= 1);
        assert_eq!(
            cold.cuda.gemm_calls, 3,
            "three signed structural jobs for this fixture"
        );
        let (_, warm) = observe(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        assert_eq!(warm.cuda.h2d_calls, 0);
        assert_eq!(warm.cuda.d2h_calls, 0);
        let result = plan
            .execute(&lhs, &rhs, &mut workspace)
            .unwrap()
            .to_host()
            .unwrap();
        let gemms = warm.cuda.gemm_calls;
        assert_eq!(
            gemms, cold.cuda.gemm_calls,
            "a reused output is born zero and not refilled (#2123)"
        );
        if let Some(expected) = expected_gemms {
            assert_eq!(gemms, expected);
        }
        expected_gemms = Some(gemms);
        for (i, member) in members.iter().enumerate() {
            let twist = |tensor: &TensorMap<R, D>, legs: &[usize]| {
                tensor.twist(legs, Direction::Forward).unwrap()
            };
            let role = if swapped { TwistRole::A } else { TwistRole::B };
            let oracle = fermionic_blas_contract_oracle_partitioned(member, role, 1, twist);
            let selected = if swapped {
                Case {
                    name: "selected signed swapped dispatch",
                    lhs: member.rhs.clone(),
                    rhs: member.lhs.clone(),
                    lhs_axes: vec![1],
                    rhs_axes: vec![0],
                    output_axes: vec![0, 1],
                    dense: false,
                }
            } else {
                Case {
                    name: member.name,
                    lhs: member.lhs.clone(),
                    rhs: member.rhs.clone(),
                    lhs_axes: member.lhs_axes.clone(),
                    rhs_axes: member.rhs_axes.clone(),
                    output_axes: member.output_axes.clone(),
                    dense: false,
                }
            };
            let selected_oracle =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::B, 1, twist);
            let untwisted =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::None, 1, twist);
            let expected = oracle.dense_data().unwrap();
            numerics::assert_nonzero_slices_close(
                base.name,
                expected,
                selected_oracle.dense_data().unwrap(),
                member.terms(),
            );
            let selected_data = selected_oracle.dense_data().unwrap();
            let scale = selected_data
                .iter()
                .map(|x| x.magnitude())
                .fold(0.0, f64::max);
            let tolerance = 64.0 * (member.terms().max(1) as f64).sqrt() * D::EPS * (1.0 + scale);
            assert!(
                selected_data
                    .iter()
                    .zip(untwisted.dense_data().unwrap())
                    .any(|(&x, &y)| x.distance(y) > tolerance),
                "twist must change signed {}",
                D::NAME
            );
            numerics::assert_nonzero_slices_close(
                base.name,
                result.member(i).unwrap().dense_data().unwrap(),
                expected,
                member.terms(),
            );
            let eager = member
                .lhs
                .to_cuda()
                .unwrap()
                .contract(&member.rhs.to_cuda().unwrap(), &member.spec())
                .unwrap()
                .to_host()
                .unwrap();
            numerics::assert_nonzero_slices_close(
                base.name,
                result.member(i).unwrap().dense_data().unwrap(),
                eager.dense_data().unwrap(),
                member.terms(),
            );
        }
        let poisoned: Vec<_> = members.iter().map(poisoned_destination).collect();
        let mut dst = StackedTensorMap::pack(&poisoned.iter().collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap();
        // The first `execute_into` at a new high-water B reserves the zero
        // template of its inactive-block fills, which `execute` never reads.
        let mut reserve = StackedTensorMap::pack(&poisoned.iter().collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap();
        let (_, first_into) = observe(|| {
            plan.execute_into(&lhs, &rhs, &mut reserve, &mut other)
                .unwrap()
        });
        let (_, into) = observe(|| plan.execute_into(&lhs, &rhs, &mut dst, &mut other).unwrap());
        assert_eq!(into.cuda.h2d_calls, 0);
        assert_eq!(into.cuda.d2h_calls, 0);
        assert_eq!(into.cuda.gemm_calls, first_into.cuda.gemm_calls);
        assert!(into.cuda.gemm_calls >= gemms);
        let written = dst.to_host().unwrap();
        for (i, member) in members.iter().enumerate() {
            numerics::assert_nonzero_slices_close(
                base.name,
                written.member(i).unwrap().dense_data().unwrap(),
                result.member(i).unwrap().dense_data().unwrap(),
                member.terms(),
            );
        }
        if count == 2 {
            let before = payload_snapshot(&dst);
            let short_rhs = rhs.select(&[0]).unwrap();
            assert!(plan
                .execute_into(&lhs, &short_rhs, &mut dst, &mut other)
                .is_err());
            assert_eq!(payload_snapshot(&dst), before);
            let mut wrong_dst = dst.select(&[0]).unwrap();
            let wrong_before = payload_snapshot(&wrong_dst);
            assert!(plan
                .execute_into(&lhs, &rhs, &mut wrong_dst, &mut other)
                .is_err());
            assert_eq!(payload_snapshot(&wrong_dst), wrong_before);
            assert_eq!(payload_snapshot(&dst), before);
        }
        let role = if swapped { TwistRole::A } else { TwistRole::B };
        check_changed_input_reuse(
            &plan,
            &members,
            &mut [&mut workspace, &mut other],
            |member| {
                fermionic_blas_contract_oracle_partitioned(member, role, 1, |tensor, legs| {
                    tensor.twist(legs, Direction::Forward).unwrap()
                })
            },
        );
    }
    drop(plan);
    drop(workspace);
    drop(other);
    assert_eq!(
        runtime
            .cuda_plan_cache_stats()
            .unwrap()
            .unwrap()
            .reserved_entries,
        reserved_before
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn signed_core_and_swapped_core_all_device_dtypes() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    macro_rules! cases {
        ($dtype:ty) => {
            for swapped in [false, true] {
                check_signed_core::<_, $dtype>(&runtime, &fermion_u1(), swapped);
                check_signed_core::<_, $dtype>(&runtime, &fermion_su2(), swapped);
            }
        };
    }
    cases!(f32);
    cases!(num_complex::Complex32);
    cases!(f64);
    cases!(Complex64);
}

#[test]
fn signed_fixture_twist_is_nonvacuous_without_a_device() {
    let runtime = Runtime::builder().build().unwrap();
    for space in [fermion_u1()] {
        let dual = space.try_dual().unwrap();
        let a =
            TensorMap::<_, f64>::from_subblock_fn(&runtime, [&space], [&dual], fill(201)).unwrap();
        let b =
            TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&space], fill(202)).unwrap();
        for swapped in [false, true] {
            let case = Case {
                name: "signed fixture",
                lhs: if swapped { b.clone() } else { a.clone() },
                rhs: if swapped { a.clone() } else { b.clone() },
                lhs_axes: vec![usize::from(!swapped)],
                rhs_axes: vec![usize::from(swapped)],
                output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
                dense: false,
            };
            let twist = |tensor: &TensorMap<_, f64>, legs: &[usize]| {
                tensor.twist(legs, Direction::Forward).unwrap()
            };
            let selected = if swapped {
                Case {
                    name: "selected signed swapped dispatch",
                    lhs: case.rhs.clone(),
                    rhs: case.lhs.clone(),
                    lhs_axes: vec![1],
                    rhs_axes: vec![0],
                    output_axes: vec![0, 1],
                    dense: false,
                }
            } else {
                case
            };
            let actual =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::B, 1, twist);
            let absent =
                fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::None, 1, twist);
            assert!(
                actual
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(absent.dense_data().unwrap())
                    .any(|(&x, &y)| (x - y).abs() > 1e-10),
                "swapped={swapped}"
            );
        }
    }
    let odd = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    let dual = odd.try_dual().unwrap();
    let case = Case {
        name: "negative-only signed core",
        lhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&odd], [&dual], |_, _| 1.0).unwrap(),
        rhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&dual], [&odd], |_, _| 1.0).unwrap(),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    let twist = |tensor: &TensorMap<_, f64>, legs: &[usize]| {
        tensor.twist(legs, Direction::Forward).unwrap()
    };
    let signed = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::B, 1, twist);
    let absent = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::None, 1, twist);
    assert!(signed
        .dense_data()
        .unwrap()
        .iter()
        .zip(absent.dense_data().unwrap())
        .any(|(&x, &y)| (x - y).abs() > 1e-10));
    numerics::assert_nonzero_slices_close(
        case.name,
        case.host().dense_data().unwrap(),
        signed.dense_data().unwrap(),
        case.terms(),
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn signed_core_zeroes_inactive_blocks_after_poisoning() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
        ],
    )
    .unwrap();
    let w = GradedSpace::try_new(
        rule,
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
    )
    .unwrap();
    let wd = w.try_dual().unwrap();
    let case = Case {
        name: "signed inactive",
        lhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&wd], fill(201)).unwrap(),
        rhs: TensorMap::<_, f64>::from_subblock_fn(&runtime, [&wd], [&v], fill(202)).unwrap(),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    let oracle = case.host();
    let expected = oracle.dense_data().unwrap();
    assert!(expected.contains(&0.0));
    assert!(expected.iter().any(|&x| x != 0.0));
    let lhs = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let poison = poisoned_destination(&case);
    for count in [1, 2, 17] {
        let lhs = StackedTensorMap::pack(&vec![&case.lhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let rhs = StackedTensorMap::pack(&vec![&case.rhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        for _ in 0..2 {
            let mut dst = StackedTensorMap::pack(&vec![&poison; count])
                .unwrap()
                .to_cuda()
                .unwrap();
            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                .unwrap();
            let result = dst.to_host().unwrap();
            for i in 0..count {
                numerics::assert_nonzero_slices_close(
                    case.name,
                    result.member(i).unwrap().dense_data().unwrap(),
                    expected,
                    case.terms(),
                );
            }
        }
        let members: Vec<_> = (0..count)
            .map(|_| Case {
                name: case.name,
                lhs: case.lhs.clone(),
                rhs: case.rhs.clone(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: case.dense,
            })
            .collect();
        check_changed_input_reuse(&plan, &members, &mut [&mut workspace], |member| {
            member.host()
        });
    }
}

fn payload_snapshot<R: DeviceRule, D: DevicePayload>(
    stack: &StackedTensorMap<R, D, CudaStorage<D>>,
) -> String {
    let host = stack.to_host().unwrap();
    format!(
        "{:?}",
        (0..host.len())
            .map(|i| host.member(i).unwrap().dense_data().unwrap().to_vec())
            .collect::<Vec<_>>()
    )
}

/// Changed-input workspace reuse (#1732). Every workspace in `workspaces` has
/// already executed this member count with `members`. Each member is
/// rescaled by its own negative factor, and both execution forms run again
/// through those same workspaces; `execute_into` writes a NaN-poisoned
/// destination. Each downloaded member must match `oracle` on the rescaled
/// Host operands. A stale output or scratch value would still have the old
/// value, which has the opposite sign.
fn check_changed_input_reuse<R: DeviceRule, D: DevicePayload>(
    plan: &ContractPlan<R, D, CudaStorage<D>>,
    members: &[Case<R, D>],
    workspaces: &mut [&mut ContractWorkspace<R, D, CudaStorage<D>>],
    oracle: impl Fn(&Case<R, D>) -> TensorMap<R, D>,
) {
    let changed: Vec<_> = members
        .iter()
        .enumerate()
        .map(|(i, member)| Case {
            name: member.name,
            lhs: member
                .lhs
                .scale(D::entry(-0.5 - i as f64 / 64.0, i as f64 / 32.0))
                .unwrap(),
            rhs: member
                .rhs
                .scale(D::entry(1.25 + i as f64 / 16.0, 0.0))
                .unwrap(),
            lhs_axes: member.lhs_axes.clone(),
            rhs_axes: member.rhs_axes.clone(),
            output_axes: member.output_axes.clone(),
            dense: member.dense,
        })
        .collect();
    let stack = |pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
        StackedTensorMap::pack(&changed.iter().map(pick).collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap()
    };
    let (lhs, rhs) = (stack(|m| &m.lhs), stack(|m| &m.rhs));
    let expected: Vec<_> = changed.iter().map(&oracle).collect();
    let poison: Vec<_> = changed.iter().map(poisoned_destination).collect();
    for workspace in workspaces.iter_mut() {
        let returned = plan
            .execute(&lhs, &rhs, workspace)
            .unwrap()
            .to_host()
            .unwrap();
        let mut dst = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, workspace).unwrap();
        let written = dst.to_host().unwrap();
        for (i, (member, expected)) in changed.iter().zip(&expected).enumerate() {
            for actual in [&returned, &written] {
                numerics::assert_nonzero_slices_close(
                    member.name,
                    actual.member(i).unwrap().dense_data().unwrap(),
                    expected.dense_data().unwrap(),
                    member.terms(),
                );
            }
        }
    }
}

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

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

fn check<R: DeviceRule, D: DevicePayload>(case: Case<R, D>) {
    let first_lhs = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let first_rhs = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut second = plan.workspace().unwrap();
    let mut warm_gemms = None;
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| {
                let lhs = case
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                    .unwrap();
                let rhs = case
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap();
                (lhs, rhs)
            })
            .collect();
        let left: Vec<_> = members.iter().map(|(lhs, _)| lhs).collect();
        let right: Vec<_> = members.iter().map(|(_, rhs)| rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap().to_cuda().unwrap();
        let cold_before = cuda_transfer_stats();
        let device_output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let cold_after = cuda_transfer_stats();
        let cold_h2d = cold_after.h2d_calls - cold_before.h2d_calls;
        let cold_bytes = cold_after.h2d_bytes - cold_before.h2d_bytes;
        let cold_gemms = cold_after.gemm_calls - cold_before.gemm_calls;
        assert!(cold_h2d >= 1, "new B needs #740 output zero upload");
        assert!(cold_gemms > 0, "direct core must submit work");
        let output = device_output.to_host().unwrap();
        assert!(
            cold_bytes
                >= (output.member(0).unwrap().dense_data().unwrap().len()
                    * count
                    * std::mem::size_of::<D>()) as u64
        );
        let warm_before = cuda_transfer_stats();
        plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let warm_after = cuda_transfer_stats();
        assert_eq!(warm_after.h2d_calls - warm_before.h2d_calls, 0);
        assert_eq!(warm_after.d2h_calls - warm_before.d2h_calls, 0);
        let submitted = warm_after.gemm_calls - warm_before.gemm_calls;
        assert!(submitted >= cold_gemms);
        if let Some(expected) = warm_gemms {
            assert_eq!(submitted, expected, "one submission count for every B");
        } else {
            warm_gemms = Some(submitted);
        }
        for (i, (left, right)) in members.iter().enumerate() {
            let member_case = Case {
                name: case.name,
                lhs: left.clone(),
                rhs: right.clone(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: case.dense,
            };
            let oracle = blas_contract_oracle(&member_case);
            let eager = left
                .to_cuda()
                .unwrap()
                .contract(&right.to_cuda().unwrap(), &case.spec())
                .unwrap()
                .to_host()
                .unwrap();
            let actual = output.member(i).unwrap();
            assert_eq!(actual.codomain_rank(), eager.codomain_rank());
            numerics::assert_nonzero_slices_close(
                case.name,
                actual.dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                case.terms(),
            );
            numerics::assert_nonzero_slices_close(
                case.name,
                actual.dense_data().unwrap(),
                eager.dense_data().unwrap(),
                case.terms(),
            );
        }
        let poison: Vec<_> = (0..count).map(|_| poisoned_destination(&case)).collect();
        let mut dst = StackedTensorMap::pack(&poison).unwrap().to_cuda().unwrap();
        let before = cuda_transfer_stats();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut second)
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(after.h2d_calls - before.h2d_calls, 0);
        assert_eq!(after.d2h_calls - before.d2h_calls, 0);
        assert_eq!(after.gemm_calls - before.gemm_calls, submitted);
        assert_eq!(after.copy_calls - before.copy_calls, 0);
        let result = dst.to_host().unwrap();
        for i in 0..count {
            numerics::assert_nonzero_slices_close(
                case.name,
                result.member(i).unwrap().dense_data().unwrap(),
                output.member(i).unwrap().dense_data().unwrap(),
                case.terms(),
            );
        }
        let wrong_indices: &[usize] = if count == 1 { &[0, 0] } else { &[0] };
        let mut wrong_dst = dst.select(wrong_indices).unwrap();
        assert_ne!(wrong_dst.len(), count);
        let before = payload_snapshot(&wrong_dst);
        assert!(matches!(
            plan.execute_into(&lhs, &rhs, &mut wrong_dst, &mut second),
            Err(Error::InvalidArgument(_))
        ));
        assert_eq!(payload_snapshot(&wrong_dst), before);
        assert!(workspace.retained_bytes() > 0);
        let member_cases: Vec<_> = members
            .iter()
            .map(|(lhs, rhs)| Case {
                name: case.name,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: case.dense,
            })
            .collect();
        check_changed_input_reuse(
            &plan,
            &member_cases,
            &mut [&mut workspace, &mut second],
            blas_contract_oracle,
        );
    }
    drop(plan);
    drop(workspace);
    drop(second);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn public_core_and_swapped_core_match_independent_oracle() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for (case, _) in candidate_core_probes::<_, f64>(&runtime, &u1_non_self_dual()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
    for (case, _) in candidate_core_probes::<_, Complex64>(&runtime, &u1_non_self_dual()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
    for (case, _) in candidate_core_probes::<_, f64>(&runtime, &su2()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
    for (case, _) in candidate_core_probes::<_, Complex64>(&runtime, &su2()) {
        if matches!(case.name, "C1" | "C2") {
            check(case);
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn swapped_core_uses_rhs_stride_and_nondefault_split() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = u1(&[(0, 2), (1, 1)]);
    let w = u1(&[(0, 1), (1, 2)]);
    let x = u1(&[(0, 2), (1, 3)]);
    let lhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&w], fill(171)).unwrap();
    let rhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&x, &x], [&v], fill(172)).unwrap();
    let spec = ContractSpec {
        lhs: &[0],
        rhs: &[2],
        codomain: &[1, 2],
        domain: &[0],
    };
    let left = StackedTensorMap::pack(&[&lhs, &lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let right = StackedTensorMap::pack(&[&rhs, &rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&left, &right, &spec).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let actual = plan
        .execute(&left, &right, &mut workspace)
        .unwrap()
        .to_host()
        .unwrap();
    let oracle = rhs.compose(&lhs).unwrap();
    for i in 0..2 {
        let member = actual.member(i).unwrap();
        assert_eq!(member.codomain_rank(), 2);
        numerics::assert_nonzero_slices_close(
            "swapped stride",
            member.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            64,
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn inactive_core_zeroes_each_poisoned_member_with_one_extra_submission() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_inactive_cases::<f64>(&runtime)
        .into_iter()
        .next()
        .unwrap();
    let first_lhs = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let first_rhs = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let oracle = blas_contract_oracle(&case);
    // Template uploads of the first `execute_into` per B on this fresh
    // Runtime: zero and ones templates at B = 1, then the zero template's
    // growth to each new high-water B. `execute` reserves none (#2123).
    for (count, template_uploads) in [(1, 2), (2, 1), (17, 1), (1, 0)] {
        let left = StackedTensorMap::pack(&vec![&case.lhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let right = StackedTensorMap::pack(&vec![&case.rhs; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let poison = poisoned_destination(&case);
        let mut dst = StackedTensorMap::pack(&vec![&poison; count])
            .unwrap()
            .to_cuda()
            .unwrap();
        let before = cuda_transfer_stats();
        plan.execute(&left, &right, &mut workspace).unwrap();
        let after = cuda_transfer_stats();
        let core_calls = after.gemm_calls - before.gemm_calls;
        assert!(core_calls > 0);
        let before = cuda_transfer_stats();
        plan.execute(&left, &right, &mut workspace).unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            after.gemm_calls - before.gemm_calls,
            core_calls,
            "the reused output's inactive region stays zero (#2123)"
        );
        assert_eq!(after.h2d_calls - before.h2d_calls, 0);
        assert_eq!(after.d2h_calls - before.d2h_calls, 0);
        let before = cuda_transfer_stats();
        plan.execute_into(&left, &right, &mut dst, &mut workspace)
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            after.gemm_calls - before.gemm_calls,
            core_calls + 1,
            "one inactive region is zeroed once over all B members"
        );
        assert_eq!(after.h2d_calls - before.h2d_calls, template_uploads);
        assert_eq!(after.d2h_calls - before.d2h_calls, 0);
        let actual = dst.to_host().unwrap();
        for i in 0..count {
            numerics::assert_nonzero_slices_close(
                case.name,
                actual.member(i).unwrap().dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                case.terms(),
            );
        }
        let members: Vec<_> = (0..count)
            .map(|_| Case {
                name: case.name,
                lhs: case.lhs.clone(),
                rhs: case.rhs.clone(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: case.dense,
            })
            .collect();
        check_changed_input_reuse(&plan, &members, &mut [&mut workspace], blas_contract_oracle);
    }
}

/// Device bytes (#2123): a caller that only calls `execute` never fills the
/// inactive core region of its born-zero output, so it pins no context zero
/// or ones template for it; the first `execute_into`, which fills it, does.
#[test]
#[ignore = "requires a real CUDA device"]
fn execute_only_caller_pins_no_core_zero_template() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_inactive_cases::<f64>(&runtime)
        .into_iter()
        .next()
        .unwrap();
    let stack = |tensor: &TensorMap<U1FusionRule, f64>, count| {
        StackedTensorMap::pack(&vec![tensor; count])
            .unwrap()
            .to_cuda()
            .unwrap()
    };
    let plan = ContractPlan::new(&stack(&case.lhs, 1), &stack(&case.rhs, 1), &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let template_bytes = || {
        runtime
            .cuda_tree_transform_stats()
            .unwrap()
            .context_scalar_operand_bytes
    };
    for count in [1, 2, 17, 1] {
        let (lhs, rhs) = (stack(&case.lhs, count), stack(&case.rhs, count));
        for _ in 0..2 {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            assert_eq!(template_bytes(), 0, "B = {count}");
        }
    }
    let (lhs, rhs) = (stack(&case.lhs, 17), stack(&case.rhs, 17));
    let mut dst = stack(&poisoned_destination(&case), 17);
    plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
        .unwrap();
    assert!(template_bytes() > 0);
    let oracle = blas_contract_oracle(&case);
    let written = dst.to_host().unwrap();
    for i in 0..17 {
        numerics::assert_nonzero_slices_close(
            case.name,
            written.member(i).unwrap().dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            case.terms(),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn invalid_bindings_preserve_poisoned_destination() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let reserved_before = runtime
        .cuda_plan_cache_stats()
        .unwrap()
        .unwrap()
        .reserved_entries;
    let case = candidate_core_probes::<_, f64>(&runtime, &u1_non_self_dual())
        .into_iter()
        .find(|(case, _)| case.name == "C1")
        .unwrap()
        .0;
    let lhs = StackedTensorMap::pack(&[&case.lhs, &case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs, &case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let other = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut foreign = other.workspace().unwrap();
    let poison = poisoned_destination(&case);
    let mut dst = StackedTensorMap::pack(&[&poison, &poison])
        .unwrap()
        .to_cuda()
        .unwrap();
    let before = payload_snapshot(&dst);
    let short = lhs.select(&[0]).unwrap();
    assert!(plan
        .execute_into(&short, &rhs, &mut dst, &mut workspace)
        .is_err());
    let other_case = candidate_core_probes::<_, f64>(&other_runtime, &u1_non_self_dual())
        .into_iter()
        .find(|(case, _)| case.name == "C1")
        .unwrap()
        .0;
    let wrong_runtime = StackedTensorMap::pack(&[&other_case.lhs, &other_case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(plan
        .execute_into(&wrong_runtime, &rhs, &mut dst, &mut workspace)
        .is_err());
    let other_space = u1(&[(0, 2)]);
    let wrong_tensor = TensorMap::<_, f64>::zeros(
        &runtime,
        [&other_space, &other_space],
        [&other_space, &other_space],
    )
    .unwrap();
    let wrong_signature = StackedTensorMap::pack(&[&wrong_tensor, &wrong_tensor])
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(plan
        .execute_into(&wrong_signature, &rhs, &mut dst, &mut workspace)
        .is_err());
    assert!(plan
        .execute_into(&lhs, &rhs, &mut dst, &mut foreign)
        .is_err());
    assert_eq!(payload_snapshot(&dst), before);
    drop(plan);
    drop(other);
    drop(workspace);
    drop(foreign);
    assert_eq!(
        runtime
            .cuda_plan_cache_stats()
            .unwrap()
            .unwrap()
            .reserved_entries,
        reserved_before,
        "workspaces return plan-ledger claims after the plan is dropped"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn negative_only_signed_core_matches_literal_twist() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let odd = symtn::typed::GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    let odd_dual = odd.try_dual().unwrap();
    let lhs =
        TensorMap::<_, f64>::from_subblock_fn(&runtime, [&odd], [&odd_dual], |_, _| 1.0).unwrap();
    let rhs =
        TensorMap::<_, f64>::from_subblock_fn(&runtime, [&odd_dual], [&odd], |_, _| 1.0).unwrap();
    let case = Case {
        name: "negative-only signed core",
        lhs,
        rhs,
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1],
        dense: false,
    };
    let twist = |tensor: &TensorMap<_, f64>, legs: &[usize]| {
        tensor.twist(legs, Direction::Forward).unwrap()
    };
    let oracle = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::B, 1, twist);
    let untwisted = fermionic_blas_contract_oracle_partitioned(&case, TwistRole::None, 1, twist);
    let expected = oracle.dense_data().unwrap();
    assert!(expected
        .iter()
        .zip(untwisted.dense_data().unwrap())
        .any(|(&x, &y)| (x - y).abs() > 1e-10));
    numerics::assert_nonzero_slices_close(
        case.name,
        case.host().dense_data().unwrap(),
        expected,
        case.terms(),
    );
    let left = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let right = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = ContractPlan::new(&left, &right, &case.spec()).unwrap();
    let actual = plan
        .execute(&left, &right, &mut plan.workspace().unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    numerics::assert_nonzero_slices_close(
        case.name,
        actual.member(0).unwrap().dense_data().unwrap(),
        expected,
        case.terms(),
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn copy_c_nonzero_single_public_admission() {
    fn check<D: DevicePayload>() {
        let runtime = Runtime::builder().cuda(0).build().unwrap();
        let v = su2();
        let a = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v, &v], [&v, &v], fill(3)).unwrap();
        let b = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v, &v], [&v, &v], fill(4)).unwrap();
        for (lhs_axes, rhs_axes, output) in [
            ([2, 3], [0, 1], [1, 0, 2, 3]),
            ([0, 1], [2, 3], [3, 2, 1, 0]),
        ] {
            let case = Case {
                name: "nonzero Single CUDA CopyC",
                lhs: a.clone(),
                rhs: b.clone(),
                lhs_axes: lhs_axes.to_vec(),
                rhs_axes: rhs_axes.to_vec(),
                output_axes: output.to_vec(),
                dense: lhs_axes == [2, 3],
            };
            let first_lhs = StackedTensorMap::pack(&[&case.lhs])
                .unwrap()
                .to_cuda()
                .unwrap();
            let first_rhs = StackedTensorMap::pack(&[&case.rhs])
                .unwrap()
                .to_cuda()
                .unwrap();
            let plan = ContractPlan::new(&first_lhs, &first_rhs, &case.spec())
                .expect("nonzero Single CopyC must be admitted");
            let reserved_before = runtime
                .cuda_plan_cache_stats()
                .unwrap()
                .unwrap()
                .reserved_entries;
            let mut workspace = plan.workspace().unwrap();
            let mut other = plan.workspace().unwrap();
            let workspace_reservations = runtime
                .cuda_plan_cache_stats()
                .unwrap()
                .unwrap()
                .reserved_entries
                - reserved_before;
            let mut expected_gemms = None;
            for count in [1, 2, 17] {
                let members: Vec<_> = (0..count)
                    .map(|i| Case {
                        name: case.name,
                        lhs: case
                            .lhs
                            .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                            .unwrap(),
                        rhs: case
                            .rhs
                            .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                            .unwrap(),
                        lhs_axes: case.lhs_axes.clone(),
                        rhs_axes: case.rhs_axes.clone(),
                        output_axes: case.output_axes.clone(),
                        dense: case.dense,
                    })
                    .collect();
                let lhs =
                    StackedTensorMap::pack(&members.iter().map(|m| &m.lhs).collect::<Vec<_>>())
                        .unwrap()
                        .to_cuda()
                        .unwrap();
                let rhs =
                    StackedTensorMap::pack(&members.iter().map(|m| &m.rhs).collect::<Vec<_>>())
                        .unwrap()
                        .to_cuda()
                        .unwrap();
                let (_, cold) = observe(|| plan.execute(&lhs, &rhs, &mut workspace).unwrap());
                assert!(cold.cuda.copy_calls > 0, "exact +1 moves remain copies");
                let (_, warm) = observe(|| {
                    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
                });
                assert_eq!(warm.cuda.copy_calls, cold.cuda.copy_calls);
                assert_eq!(warm.cuda.h2d_calls, 0);
                assert_eq!(warm.cuda.d2h_calls, 0);
                assert!(warm.cuda.gemm_calls > 0);
                if let Some(gemms) = expected_gemms {
                    assert_eq!(warm.cuda.gemm_calls, gemms);
                }
                expected_gemms = Some(warm.cuda.gemm_calls);
                eprintln!(
                    "CopyC {} swapped={} B={count} cold_host_submission={:?} warm_host_submission={:?} cold_host_allocs={}/{} warm_host_allocs={}/{} cold_cuda={:?} warm_cuda={:?} retained_workspace_bytes={} plans={:?}",
                    D::NAME, lhs_axes == [0, 1], cold.elapsed, warm.elapsed,
                    cold.host_calls, cold.host_bytes, warm.host_calls, warm.host_bytes,
                    cold.cuda, warm.cuda, workspace.retained_bytes(),
                    runtime.cuda_plan_cache_stats().unwrap().unwrap(),
                );
                let output = plan
                    .execute(&lhs, &rhs, &mut workspace)
                    .unwrap()
                    .to_host()
                    .unwrap();
                for (i, member) in members.iter().enumerate() {
                    let actual = output.member(i).unwrap();
                    let oracle = blas_contract_oracle(member);
                    numerics::assert_nonzero_slices_close(
                        case.name,
                        actual.dense_data().unwrap(),
                        oracle.dense_data().unwrap(),
                        member.terms(),
                    );
                    let eager = member
                        .lhs
                        .to_cuda()
                        .unwrap()
                        .contract(&member.rhs.to_cuda().unwrap(), &member.spec())
                        .unwrap()
                        .to_host()
                        .unwrap();
                    numerics::assert_nonzero_slices_close(
                        case.name,
                        actual.dense_data().unwrap(),
                        eager.dense_data().unwrap(),
                        member.terms(),
                    );
                    if member.dense {
                        let (_, physical) = dense_oracle(member);
                        numerics::assert_nonzero_slices_close(
                            case.name,
                            &actual.to_physical_dense().unwrap().data,
                            &physical,
                            member.terms(),
                        );
                    }
                }
                let poison: Vec<_> = members.iter().map(poisoned_destination).collect();
                let mut dst = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
                    .unwrap()
                    .to_cuda()
                    .unwrap();
                let (_, into) =
                    observe(|| plan.execute_into(&lhs, &rhs, &mut dst, &mut other).unwrap());
                assert_eq!(into.cuda.copy_calls, cold.cuda.copy_calls);
                assert!(
                    into.cuda.h2d_calls >= 1,
                    "cold second workspace uploads its temporary"
                );
                assert_eq!(into.cuda.d2h_calls, 0);
                let (_, warm_into) =
                    observe(|| plan.execute_into(&lhs, &rhs, &mut dst, &mut other).unwrap());
                assert_eq!(warm_into.cuda.copy_calls, cold.cuda.copy_calls);
                assert_eq!(warm_into.cuda.h2d_calls, 0);
                assert_eq!(warm_into.cuda.d2h_calls, 0);
                let eager_inputs: Vec<_> = members
                    .iter()
                    .map(|member| {
                        (
                            member.lhs.to_cuda().unwrap(),
                            member.rhs.to_cuda().unwrap(),
                            member.spec(),
                        )
                    })
                    .collect();
                let (_, eager) = observe(|| {
                    for (left, right, spec) in &eager_inputs {
                        black_box(left.contract(right, spec).unwrap());
                    }
                });
                eprintln!(
                    "CopyC {} swapped={} B={count} cold_into_host_submission={:?} warm_into_host_submission={:?} eager_member_loop_host_submission={:?} cold_into_allocs={}/{} warm_into_allocs={}/{} eager_allocs={}/{} cold_into_cuda={:?} warm_into_cuda={:?} eager_cuda={:?} retained_into_workspace_bytes={}",
                    D::NAME, lhs_axes == [0, 1], into.elapsed, warm_into.elapsed, eager.elapsed,
                    into.host_calls, into.host_bytes, warm_into.host_calls, warm_into.host_bytes,
                    eager.host_calls, eager.host_bytes, into.cuda, warm_into.cuda, eager.cuda,
                    other.retained_bytes(),
                );
                let written = dst.to_host().unwrap();
                for (i, member) in members.iter().enumerate() {
                    numerics::assert_nonzero_slices_close(
                        case.name,
                        written.member(i).unwrap().dense_data().unwrap(),
                        output.member(i).unwrap().dense_data().unwrap(),
                        member.terms(),
                    );
                }
                if count == 2 {
                    let mut invalid = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
                        .unwrap()
                        .to_cuda()
                        .unwrap();
                    let before = payload_snapshot(&invalid);
                    let short_rhs = rhs.select(&[0]).unwrap();
                    let (rejected, metrics) =
                        observe(|| plan.execute_into(&lhs, &short_rhs, &mut invalid, &mut other));
                    assert!(rejected.is_err());
                    assert_eq!(metrics.cuda.gemm_calls, 0);
                    assert_eq!(metrics.cuda.copy_calls, 0);
                    assert_eq!(payload_snapshot(&invalid), before);
                }
                check_changed_input_reuse(
                    &plan,
                    &members,
                    &mut [&mut workspace, &mut other],
                    blas_contract_oracle,
                );
            }
            check_high_water(&plan, &case, &blas_contract_oracle);
            let held_with_eager = runtime
                .cuda_plan_cache_stats()
                .unwrap()
                .unwrap()
                .reserved_entries;
            drop(other);
            drop(workspace);
            drop(plan);
            assert_eq!(
                runtime
                    .cuda_plan_cache_stats()
                    .unwrap()
                    .unwrap()
                    .reserved_entries
                    + workspace_reservations,
                held_with_eager,
                "CopyC workspaces must release their own plan reservations"
            );
        }
    }
    check::<f32>();
    check::<num_complex::Complex32>();
    check::<f64>();
    check::<Complex64>();
}

/// Batch-independent submission counts of one warm call.
fn submissions(observation: &Observation) -> (u64, u64) {
    (observation.cuda.gemm_calls, observation.cuda.copy_calls)
}

/// One transformed-tree (DynamicTree) case at B=1/2/17: every member equals
/// eager CUDA `contract` and the independent `oracle`; warm calls move no
/// payload across the Host boundary and submit the same work for every B;
/// a poisoned destination is fully overwritten through a second workspace.
/// `physical` returns `(actual, expected)` physical-basis arrays where the
/// provider has a physical basis.
fn check_dynamic_tree<R: DeviceRule, D: DevicePayload>(
    case: &Case<R, D>,
    oracle: impl Fn(&Case<R, D>) -> TensorMap<R, D>,
    physical: impl Fn(&Case<R, D>, &TensorMap<R, D>) -> Option<(Vec<D>, Vec<D>)>,
) {
    let stack = |members: &[Case<R, D>], lhs: bool| {
        StackedTensorMap::pack(
            &members
                .iter()
                .map(|m| if lhs { &m.lhs } else { &m.rhs })
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .to_cuda()
        .unwrap()
    };
    let one = [Case {
        name: case.name,
        lhs: case.lhs.clone(),
        rhs: case.rhs.clone(),
        lhs_axes: case.lhs_axes.clone(),
        rhs_axes: case.rhs_axes.clone(),
        output_axes: case.output_axes.clone(),
        dense: case.dense,
    }];
    let plan = ContractPlan::new(&stack(&one, true), &stack(&one, false), &case.spec())
        .unwrap_or_else(|error| panic!("{}: DynamicTree must be admitted: {error:?}", case.name));
    let mut workspace = plan.workspace().unwrap();
    let mut other = plan.workspace().unwrap();
    let mut warm_counts = None;
    let mut into_counts = None;
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| Case {
                name: case.name,
                lhs: case
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                    .unwrap(),
                rhs: case
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: case.dense,
            })
            .collect();
        let (lhs, rhs) = (stack(&members, true), stack(&members, false));
        let (_, cold) = observe(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        assert!(cold.cuda.gemm_calls > 0, "{}: core must submit", case.name);
        let (_, warm) = observe(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        assert_eq!(
            (
                warm.cuda.h2d_calls,
                warm.cuda.d2h_calls,
                warm.cuda.device_allocs
            ),
            (0, 0, 0),
            "{}: warm batch must not transfer or allocate",
            case.name
        );
        assert_eq!(
            *warm_counts.get_or_insert(submissions(&warm)),
            submissions(&warm),
            "{}: submissions independent of B",
            case.name
        );
        let output = plan
            .execute(&lhs, &rhs, &mut workspace)
            .unwrap()
            .to_host()
            .unwrap();
        for (i, member) in members.iter().enumerate() {
            let actual = output.member(i).unwrap();
            let eager = member
                .lhs
                .to_cuda()
                .unwrap()
                .contract(&member.rhs.to_cuda().unwrap(), &member.spec())
                .unwrap()
                .to_host()
                .unwrap();
            let expected = oracle(member);
            for reference in [&eager, &expected] {
                assert_eq!(actual.codomain_rank(), reference.codomain_rank());
                numerics::assert_nonzero_slices_close(
                    case.name,
                    actual.dense_data().unwrap(),
                    reference.dense_data().unwrap(),
                    member.terms(),
                );
            }
            if let Some((actual, expected)) = physical(member, &actual) {
                numerics::assert_nonzero_slices_close(
                    case.name,
                    &actual,
                    &expected,
                    member.terms(),
                );
            }
        }
        let poison: Vec<_> = members.iter().map(poisoned_destination).collect();
        let mut dst = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut other).unwrap();
        let (_, warm_into) = observe(|| {
            plan.execute_into(&lhs, &rhs, &mut dst, &mut other).unwrap();
        });
        assert_eq!(
            (
                warm_into.cuda.h2d_calls,
                warm_into.cuda.d2h_calls,
                warm_into.cuda.device_allocs
            ),
            (0, 0, 0),
            "{}: warm into must not transfer or allocate",
            case.name
        );
        assert_eq!(
            *into_counts.get_or_insert(submissions(&warm_into)),
            submissions(&warm_into),
            "{}: into submissions independent of B",
            case.name
        );
        let written = dst.to_host().unwrap();
        for (i, member) in members.iter().enumerate() {
            numerics::assert_nonzero_slices_close(
                case.name,
                written.member(i).unwrap().dense_data().unwrap(),
                output.member(i).unwrap().dense_data().unwrap(),
                member.terms(),
            );
        }
        if count == 2 {
            let mut invalid = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
                .unwrap()
                .to_cuda()
                .unwrap();
            let before = payload_snapshot(&invalid);
            let short_rhs = rhs.select(&[0]).unwrap();
            let (rejected, metrics) =
                observe(|| plan.execute_into(&lhs, &short_rhs, &mut invalid, &mut other));
            assert!(rejected.is_err());
            assert_eq!(submissions(&metrics), (0, 0));
            assert_eq!(payload_snapshot(&invalid), before);
        }
        check_changed_input_reuse(&plan, &members, &mut [&mut workspace, &mut other], &oracle);
        let eager_inputs: Vec<_> = members
            .iter()
            .map(|m| (m.lhs.to_cuda().unwrap(), m.rhs.to_cuda().unwrap()))
            .collect();
        let (_, eager) = observe(|| {
            for (left, right) in &eager_inputs {
                black_box(left.contract(right, &case.spec()).unwrap());
            }
        });
        eprintln!(
            "DynamicTree {} {} B={count}: cold={cold}; warm={warm}; warm_into={warm_into}; eager_members={eager}; retained_workspace_bytes={}",
            case.name,
            D::NAME,
            workspace.retained_bytes()
        );
    }
    check_high_water(&plan, case, &oracle);
}

/// High-water member stacks across B changes (#1746): the DynamicTree
/// workspace stacks and the CopyC temporary. Fresh workspaces run
/// B = 4, 2, 4 and B = 1, 17, 3 with new values at every call, so a stale
/// stack region left by an earlier, larger B would show. Each member of
/// `execute` and of `execute_into` (NaN-poisoned destination, so every
/// inactive output block must be written zero on device) matches `oracle`.
/// Through `execute_into`, which owns no output, a B at or below the high
/// water allocates and uploads nothing and keeps the retained bytes; through
/// `execute` the only upload is the owned output of the new B (#740).
fn check_high_water<R: DeviceRule, D: DevicePayload>(
    plan: &ContractPlan<R, D, CudaStorage<D>>,
    case: &Case<R, D>,
    oracle: &impl Fn(&Case<R, D>) -> TensorMap<R, D>,
) {
    let stack = |members: &[Case<R, D>], pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
        StackedTensorMap::pack(&members.iter().map(pick).collect::<Vec<_>>())
            .unwrap()
            .to_cuda()
            .unwrap()
    };
    for sequence in [[4, 2, 4], [1, 17, 3]] {
        let mut workspace = plan.workspace().unwrap();
        let mut into = plan.workspace().unwrap();
        let (mut high_water, mut high_bytes) = (0, 0);
        for (call, count) in sequence.into_iter().enumerate() {
            let call = call as f64;
            let members: Vec<_> = (0..count)
                .map(|i| Case {
                    name: case.name,
                    lhs: case
                        .lhs
                        .scale(D::entry(
                            -0.75 - call / 8.0 + i as f64 / 64.0,
                            (call + 1.0) / 16.0 - i as f64 / 128.0,
                        ))
                        .unwrap(),
                    rhs: case
                        .rhs
                        .scale(D::entry(1.5 + call / 4.0 + i as f64 / 32.0, 0.0))
                        .unwrap(),
                    lhs_axes: case.lhs_axes.clone(),
                    rhs_axes: case.rhs_axes.clone(),
                    output_axes: case.output_axes.clone(),
                    dense: case.dense,
                })
                .collect();
            let (lhs, rhs) = (stack(&members, |m| &m.lhs), stack(&members, |m| &m.rhs));
            let (_, owned) = observe(|| {
                plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            });
            let returned = plan
                .execute(&lhs, &rhs, &mut workspace)
                .unwrap()
                .to_host()
                .unwrap();
            let poison: Vec<_> = members.iter().map(poisoned_destination).collect();
            let mut dst = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
                .unwrap()
                .to_cuda()
                .unwrap();
            let (_, written_metrics) = observe(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut into).unwrap();
            });
            let written = dst.to_host().unwrap();
            for (i, member) in members.iter().enumerate() {
                let expected = oracle(member);
                for actual in [&returned, &written] {
                    numerics::assert_nonzero_slices_close(
                        case.name,
                        actual.member(i).unwrap().dense_data().unwrap(),
                        expected.dense_data().unwrap(),
                        member.terms(),
                    );
                }
            }
            let into_counts = (
                written_metrics.cuda.h2d_calls,
                written_metrics.cuda.device_allocs,
            );
            if count <= high_water {
                assert_eq!(
                    into_counts,
                    (0, 0),
                    "{}: B={count} under high water {high_water} must not allocate",
                    case.name
                );
                assert_eq!(
                    into.retained_bytes(),
                    high_bytes,
                    "{}: B={count} keeps the high-water stacks",
                    case.name
                );
                assert_eq!(
                    (owned.cuda.h2d_calls, owned.cuda.device_allocs),
                    (1, 1),
                    "{}: B={count} execute uploads only its owned output",
                    case.name
                );
            } else {
                assert!(
                    into_counts.0 >= 1,
                    "{}: B={count} grows the stacks",
                    case.name
                );
                assert!(into.retained_bytes() > high_bytes);
                (high_water, high_bytes) = (count, into.retained_bytes());
            }
            eprintln!(
                "high water {} {} B={count} (high water {high_water}): execute={owned}; execute_into={written_metrics}; retained_into_bytes={}",
                case.name,
                D::NAME,
                into.retained_bytes()
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn dynamic_tree_single_transform_routes_are_admitted() {
    fn bosonic<D: DevicePayload>(runtime: &Runtime) {
        // The second inactive fixture has a source transform and is pinned as
        // DynamicTree by storage_contract_tests::each_way_a_device_overwrite;
        // u1_reordered adds an output transform.
        let inactive = u1_inactive_cases::<D>(runtime).into_iter().nth(1).unwrap();
        for case in [inactive, u1_reordered::<D>(runtime)] {
            check_dynamic_tree(&case, blas_contract_oracle, |member, actual| {
                member.dense.then(|| {
                    (
                        actual.to_physical_dense().unwrap().data,
                        dense_oracle(member).1,
                    )
                })
            });
        }
    }
    fn fermionic<D: DevicePayload>(runtime: &Runtime) {
        let twist = |tensor: &TensorMap<FermionU1, D>, legs: &[usize]| {
            tensor.twist(legs, Direction::Forward).unwrap()
        };
        let roles = fermionic_twist_roles::<FermionU1, D>(
            runtime,
            &fermion_u1(),
            [
                "fZ2xU1 A copied",
                "fZ2xU1 canonical",
                "fZ2xU1 B copied",
                "fZ2xU1 both",
            ],
            71,
        )
        .into_iter()
        .zip([TwistRole::A, TwistRole::A, TwistRole::B, TwistRole::A])
        .chain([(
            fermionic_canonical_nonuniform::<FermionU1, D>(
                runtime,
                &fermion_u1(),
                "fZ2xU1 canonical nonuniform",
                39,
            ),
            TwistRole::A,
        )]);
        for (case, role) in roles {
            check_dynamic_tree(
                &case,
                |member| fermionic_blas_contract_oracle(member, role, twist),
                |_, _| None,
            );
        }
    }
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    bosonic::<f32>(&runtime);
    bosonic::<num_complex::Complex32>(&runtime);
    bosonic::<f64>(&runtime);
    bosonic::<Complex64>(&runtime);
    fermionic::<f32>(&runtime);
    fermionic::<num_complex::Complex32>(&runtime);
    fermionic::<f64>(&runtime);
    fermionic::<Complex64>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn dynamic_tree_multi_transform_stays_unsupported() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = su2_bent::<f64>(&runtime);
    let left = StackedTensorMap::pack(&[&case.lhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    let right = StackedTensorMap::pack(&[&case.rhs])
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(matches!(ContractPlan::new(&left, &right, &case.spec()),
    Err(Error::Operation(error)) if matches!(*error,
        symtn::typed::OperationError::UnsupportedTensorContractScope {
            message: "CUDA member transform requires unconjugated nonzero Single tasks"
        })));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn nonsymmetric_braiding_is_rejected_before_cuda_execution() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let leg =
        symtn::typed::GradedSpace::try_new(Arc::new(RealBraidingProbe::<true>), [(ProbeSector, 2)])
            .unwrap();
    let tensor = TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg]).unwrap();
    let stack = StackedTensorMap::pack(&[tensor])
        .unwrap()
        .to_cuda()
        .unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    assert!(matches!(ContractPlan::new(&stack, &stack, &spec),
    Err(Error::Operation(error)) if matches!(*error,
        symtn::typed::OperationError::UnsupportedTensorContractScope {
            message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
        })));
}
