//! Device-only gates for general-axes contraction on device tensors: device
//! == Host of the same public call (and the independent oracle) on the
//! zero-copy candidates and every split, borrowed identity adjoints,
//! `copyC` transfer counts, warm calls, and rejections before any device
//! work. The executor replaying one artifact is gated against the Host
//! replay in `symtn-tensors` (`storage_contract_tests.rs`); the independent
//! oracles on every fixture, owned and lazy, `contract` and `contract_into`,
//! run on both engines in `engine_suites/contract.rs`.
//!
//! The target-wide serial guard isolates the thread-local transfer counters.
//! Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test serial typed_cuda_contract:: -- --ignored
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

use crate::common;

use crate::contract_cases;
use crate::numerics;
use crate::report;
use report::Report;

use common::{DevicePayload, DeviceRule};
use contract_cases::{
    blas_contract_oracle, candidate_core_probes, copy_c_probes, fermion_su2, fermion_u1,
    fermionic_blas_contract_oracle, fermionic_general, identity_adjoint_cases,
    poisoned_destination, product_general, su2, su2_bent, u1_non_self_dual, u1_rank_five,
    u1_reordered, Case, FermionU1, TwistRole,
};
use num_complex::{Complex32, Complex64};
use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::typed::Direction;
use symtn::typed::{ContractSpec, Runtime, TensorMap};

// Host allocations are counted on the measuring thread only; device runtime
// threads never allocate for the warm host-allocation contract.

fn delta<T>(body: impl FnOnce() -> T) -> (T, CudaTransferStats) {
    let before = cuda_transfer_stats();
    let value = body();
    let after = cuda_transfer_stats();
    (
        value,
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
            gauge_ops: after.gauge_ops - before.gauge_ops,
        },
    )
}

/// Device == Host and device == `blas_contract!` for one fixture.
fn check<R, D>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    case: Case<R, D>,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let host = case.host(engine);
    let rhs_device = case.rhs.to_cuda(cuda).unwrap();
    let device = case
        .lhs
        .to_cuda(cuda)
        .unwrap()
        .contract(cuda, &rhs_device, &case.spec())
        .unwrap()
        .to_host(cuda)
        .unwrap();
    assert_eq!(
        device.codomain_rank(),
        host.codomain_rank(),
        "{}",
        case.name
    );
    assert_eq!(device.domain_rank(), host.domain_rank(), "{}", case.name);
    numerics::assert_nonzero_slices_close(
        case.name,
        device.dense_data().unwrap(),
        host.dense_data().unwrap(),
        case.terms(),
    );
    let oracle = blas_contract_oracle(engine, &case);
    numerics::assert_nonzero_slices_close(
        case.name,
        device.dense_data().unwrap(),
        oracle.dense_data().unwrap(),
        case.terms(),
    );
}

/// The zero-copy sorted/swapped candidates run the device Core
/// route, swapped operands and lazy-adjoint GEMM flags included. The
/// fermionic probes take the twisted `blas_contract!` oracle: C2 and L5
/// contract dual B legs in their literal sequence, which TensorKit twists.
#[test]
#[ignore = "requires a real CUDA device"]
fn zero_copy_candidates_match_the_host_at_every_dtype() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    fn at<D: DevicePayload>(
        engine: &mut dyn symtn::typed::Engine,
        cuda: &mut dyn symtn::typed::Engine,
        runtime: &Runtime,
    ) {
        for (case, _) in candidate_core_probes::<_, D>(engine, runtime, &u1_non_self_dual()) {
            check(engine, cuda, case);
        }
        for (case, _) in candidate_core_probes::<_, D>(engine, runtime, &su2()) {
            check(engine, cuda, case);
        }
        let twist =
            |engine: &mut dyn symtn::typed::Engine, t: &TensorMap<FermionU1, D>, legs: &[usize]| {
                t.twist(engine, legs, Direction::Forward).unwrap()
            };
        for (case, _) in candidate_core_probes::<_, D>(engine, runtime, &fermion_u1()) {
            check_fermionic(engine, cuda, case, twist);
        }
    }
    at::<f64>(engine, cuda, &runtime);
    at::<Complex64>(engine, cuda, &runtime);
    at::<f32>(engine, cuda, &runtime);
    at::<Complex32>(engine, cuda, &runtime);
}

/// A lazy adjoint whose tree transform is the identity, against a
/// permuted partner, runs the device DynamicTree with the adjoint read from
/// its parent's device buffer under a GEMM adjoint op (TensorKit
/// `blas_contract!` without `copyA`, `tensoroperations.jl:383-450` @cfaa073).
/// Values: device == Host == the materialize-then-contract oracle. (The
/// scratch-size route witness went with the Runtime-owned device scratch;
/// each device call now owns its scratch.)
#[test]
#[ignore = "requires a real CUDA device"]
fn identity_adjoint_operands_are_borrowed_by_the_device_dynamic_tree() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    fn at<R: DeviceRule, D: DevicePayload>(
        engine: &mut dyn symtn::typed::Engine,
        cuda: &mut dyn symtn::typed::Engine,
        v: &symtn::typed::GradedSpace<R>,
        symmetry: &str,
    ) {
        for index in 0..3 {
            let runtime = Runtime::builder().build().unwrap();
            let (case, _) = identity_adjoint_cases::<R, D>(engine, &runtime, v)
                .into_iter()
                .nth(index)
                .unwrap();
            let name = format!("{symmetry} {} {}", case.name, D::NAME);
            let host = case.host(engine);
            let rhs = case.rhs.materialize(engine).unwrap();
            let oracle = case
                .lhs
                .materialize(engine)
                .unwrap()
                .contract(engine, &rhs, &case.spec())
                .unwrap();
            let rhs_device = case.rhs.to_cuda(cuda).unwrap();
            let device = case
                .lhs
                .to_cuda(cuda)
                .unwrap()
                .contract(cuda, &rhs_device, &case.spec())
                .unwrap()
                .to_host(cuda)
                .unwrap();
            for (what, expected) in [("host", &host), ("materialized oracle", &oracle)] {
                numerics::assert_nonzero_slices_close(
                    &format!("{name} vs {what}"),
                    device.dense_data().unwrap(),
                    expected.dense_data().unwrap(),
                    case.terms(),
                );
            }
        }
    }
    at::<_, f64>(engine, cuda, &su2(), "SU(2)");
    at::<_, Complex64>(engine, cuda, &su2(), "SU(2)");
    at::<_, f64>(engine, cuda, &u1_non_self_dual(), "U(1)");
    at::<_, Complex64>(engine, cuda, &u1_non_self_dual(), "U(1)");
}

/// Device `contract` / `contract_into` take TensorKit's `copyC`
/// (`blas_contract!`, `tensoroperations.jl:436-446` @cfaa073) where the batched
/// `ContractPlan` does: the zero-copy core into the core-destination scratch,
/// then one permute. Values only: the call-local scratch is not observable.
#[test]
#[ignore = "requires a real CUDA device"]
fn copy_c_probes_move_only_c_and_match_the_host_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_shared();
    fn at<R: DeviceRule, D: DevicePayload>(
        engine: &mut dyn symtn::typed::Engine,
        cuda: &mut dyn symtn::typed::Engine,
        v: &symtn::typed::GradedSpace<R>,
        symmetry: &str,
        oracle: impl Fn(&mut dyn symtn::typed::Engine, &Case<R, D>) -> TensorMap<R, D>,
    ) {
        for index in 0..copy_c_probes::<R, D>(engine, &Runtime::builder().build().unwrap(), v).len()
        {
            // A fresh Runtime per probe: its scratch high-water mark is this
            // probe's alone.
            let runtime = Runtime::builder().build().unwrap();
            let (case, _) = copy_c_probes::<R, D>(engine, &runtime, v).swap_remove(index);
            let name = format!("{symmetry} {} {}", case.name, D::NAME);
            let host = case.host(engine);
            let lhs = case.lhs.to_cuda(cuda).unwrap();
            let rhs = case.rhs.to_cuda(cuda).unwrap();
            let device = lhs.contract(cuda, &rhs, &case.spec()).unwrap();
            numerics::assert_nonzero_slices_close(
                &name,
                device.to_host(cuda).unwrap().dense_data().unwrap(),
                host.dense_data().unwrap(),
                case.terms(),
            );
            numerics::assert_nonzero_slices_close(
                &name,
                device.to_host(cuda).unwrap().dense_data().unwrap(),
                oracle(engine, &case).dense_data().unwrap(),
                case.terms(),
            );
            let mut destination = poisoned_destination(engine, &case).to_cuda(cuda).unwrap();
            let (alpha, beta) = (D::entry(0.75, 0.25), D::entry(0.0, 0.0));
            lhs.contract_into(cuda, &rhs, &case.spec(), &mut destination, alpha, beta)
                .unwrap();
            numerics::assert_nonzero_slices_close(
                &name,
                destination.to_host(cuda).unwrap().dense_data().unwrap(),
                host.scale(engine, alpha).unwrap().dense_data().unwrap(),
                case.terms(),
            );
        }
    }
    at::<_, f64>(
        engine,
        cuda,
        &u1_non_self_dual(),
        "U(1)",
        blas_contract_oracle,
    );
    at::<_, Complex64>(
        engine,
        cuda,
        &u1_non_self_dual(),
        "U(1)",
        blas_contract_oracle,
    );
    at::<_, f32>(engine, cuda, &su2(), "SU(2)", blas_contract_oracle);
    at::<_, Complex32>(engine, cuda, &su2(), "SU(2)", blas_contract_oracle);
    at::<_, f64>(engine, cuda, &su2(), "SU(2)", blas_contract_oracle);
    // The swapped probes' literal sequence twists dual `B` legs the selected
    // candidate does not, so TensorKit's fermionic sequence is the oracle.
    at::<_, f64>(engine, cuda, &fermion_u1(), "fZ2xU(1)", |engine, case| {
        fermionic_blas_contract_oracle(engine, case, TwistRole::B, |engine, t, legs| {
            t.twist(engine, legs, Direction::Forward).unwrap()
        })
    });
}

/// Orders process-global cache clears against warm/cold count windows.
/// Cooperative: every new clearing test must take the write lock
/// and every new count-dependent test `cache_shared`.
static CACHE_TEST_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

fn cache_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    CACHE_TEST_LOCK.read().unwrap_or_else(|e| e.into_inner())
}

fn device_contract<R: DeviceRule, D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    case: &Case<R, D>,
) -> TensorMap<R, D> {
    let rhs_device = case.rhs.to_cuda(cuda).unwrap();
    case.lhs
        .to_cuda(cuda)
        .unwrap()
        .contract(cuda, &rhs_device, &case.spec())
        .unwrap()
        .to_host(cuda)
        .unwrap()
}

/// Device == Host and device == TensorKit's `blas_contract!` with the twist
/// on the B role and on the A role.
fn check_fermionic<R, D>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    case: Case<R, D>,
    twist: impl Fn(&mut dyn symtn::typed::Engine, &TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let device = device_contract(cuda, &case);
    let host = case.host(engine);
    assert_eq!(
        device.codomain_rank(),
        host.codomain_rank(),
        "{}",
        case.name
    );
    numerics::assert_nonzero_slices_close(
        case.name,
        device.dense_data().unwrap(),
        host.dense_data().unwrap(),
        case.terms(),
    );
    for role in [TwistRole::B, TwistRole::A] {
        let oracle = fermionic_blas_contract_oracle(engine, &case, role, twist);
        numerics::assert_nonzero_slices_close(
            &format!("{} vs the {role:?}-role oracle", case.name),
            device.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            case.terms(),
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rejections_happen_before_any_device_work_and_match_the_host() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    let case = u1_rank_five::<f64>(engine, &runtime);
    let host_lhs = &case.lhs;
    let host_rhs = &case.rhs;
    let lhs = host_lhs.to_cuda(cuda).unwrap();
    let rhs = host_rhs.to_cuda(cuda).unwrap();
    let stranger = u1_rank_five::<f64>(engine, &other)
        .rhs
        .to_cuda(cuda)
        .unwrap();
    let _ = lhs.contract(cuda, &rhs, &case.spec()).unwrap();
    // Malformed: an axis out of range, a repeated axis, an output order that
    // is not a permutation, and a pairing of two equal (not dual) legs.
    let (errors, counters) = delta(|| {
        let mut errors = Vec::new();
        for spec in malformed_rank_five_specs() {
            let expected = host_lhs
                .contract(engine, host_rhs, &spec)
                .unwrap_err()
                .report();
            let actual = lhs.contract(cuda, &rhs, &spec).unwrap_err().report();
            assert_eq!(actual, expected, "device error text must be the Host's");
            errors.push(actual);
        }
        errors.push(
            lhs.contract(cuda, &stranger, &case.spec())
                .unwrap_err()
                .to_string(),
        );
        errors
    });
    assert_eq!(errors.len(), 5);
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a rejected contraction must submit nothing: {counters:?}"
    );
}

/// Every split size of `case`'s open legs, in its own output order and
/// reversed: the device result is the Host one, which
/// `contract_spec_split.rs` pins against contract-then-permute. The returning
/// and the overwrite forms both run.
fn check_splits<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    case: Case<R, D>,
) {
    let lhs = case.lhs.to_cuda(cuda).unwrap();
    let rhs = case.rhs.to_cuda(cuda).unwrap();
    let mut orders = vec![case.output_axes.clone()];
    orders.push(case.output_axes.iter().rev().copied().collect());
    for order in &orders {
        for split in 0..=order.len() {
            let (codomain, domain) = order.split_at(split);
            let spec = ContractSpec {
                codomain,
                domain,
                ..case.spec()
            };
            let label = format!("{} {codomain:?} <- {domain:?}", case.name);
            let host = case.lhs.contract(engine, &case.rhs, &spec).unwrap();
            let device = lhs
                .contract(cuda, &rhs, &spec)
                .unwrap()
                .to_host(cuda)
                .unwrap();
            assert_eq!(device.codomain(), host.codomain(), "{label}");
            assert_eq!(device.domain(), host.domain(), "{label}");
            numerics::assert_nonzero_slices_close(
                &label,
                device.dense_data().unwrap(),
                host.dense_data().unwrap(),
                case.terms(),
            );
            let mut destination = host
                .scale(engine, D::entry(7.5, 0.0))
                .unwrap()
                .to_cuda(cuda)
                .unwrap();
            lhs.contract_into(
                cuda,
                &rhs,
                &spec,
                &mut destination,
                D::entry(1.0, 0.0),
                D::entry(0.0, 0.0),
            )
            .unwrap();
            numerics::assert_nonzero_slices_close(
                &label,
                destination.to_host(cuda).unwrap().dense_data().unwrap(),
                host.dense_data().unwrap(),
                case.terms(),
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn every_split_matches_the_host_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let fixture = u1_rank_five::<f64>(engine, &runtime);
    check_splits(engine, cuda, fixture);
    let fixture = su2_bent::<Complex64>(engine, &runtime);
    check_splits(engine, cuda, fixture);
    let fixture = product_general::<f32>(engine, &runtime);
    check_splits(engine, cuda, fixture);
    let fixture =
        fermionic_general::<_, Complex32>(engine, &runtime, &fermion_su2(), true, "fZ2xSU2", 52);
    check_splits(engine, cuda, fixture);
}

/// Malformed specs over `u1_rank_five` (rank 5 against rank 4): an axis out of
/// range, a repeated axis, an output order that is not a permutation, and a
/// pairing of two equal (not dual) legs.
fn malformed_rank_five_specs() -> [ContractSpec<'static>; 4] {
    [
        ContractSpec {
            lhs: &[3, 7],
            rhs: &[0, 3],
            codomain: &[2, 0, 4],
            domain: &[1, 3],
        },
        ContractSpec {
            lhs: &[3, 3],
            rhs: &[0, 3],
            codomain: &[2, 0, 4],
            domain: &[1, 3],
        },
        ContractSpec {
            lhs: &[3, 1],
            rhs: &[0, 3],
            codomain: &[2, 0, 4],
            domain: &[1, 1],
        },
        ContractSpec {
            lhs: &[0],
            rhs: &[0],
            codomain: &[0, 1, 2, 3],
            domain: &[4, 5, 6],
        },
    ]
}

// ---------------------------------------------------------------------------
// `contract_into`
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_rejections_match_the_host_in_order_and_leave_the_destination_untouched() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    let case = u1_rank_five::<f64>(engine, &runtime);
    let (host_lhs, host_rhs) = (&case.lhs, &case.rhs);
    let lhs = host_lhs.to_cuda(cuda).unwrap();
    let rhs = host_rhs.to_cuda(cuda).unwrap();
    let poison =
        |engine: &mut dyn symtn::typed::Engine| case.host(engine).scale(engine, 7.5).unwrap();
    let poison_data = poison(engine).dense_data().unwrap().to_vec();
    let _ = lhs.contract(cuda, &rhs, &case.spec()).unwrap();
    let device_text = |text: String| text.replace("host storage", "CUDA storage");

    // Host-shared rejections, each against a fresh poisoned destination:
    // malformed axes, an output order that is not a permutation, a pairing
    // of two equal (not dual) legs, and a destination of another space.
    let foreign = u1_reordered::<f64>(engine, &runtime)
        .host(engine)
        .scale(engine, 7.5)
        .unwrap();
    let cases: Vec<_> = malformed_rank_five_specs()
        .into_iter()
        .map(|spec| (spec, poison(engine)))
        .chain([(case.spec(), foreign)])
        .collect();
    for (spec, host_destination) in cases {
        let before = host_destination.dense_data().unwrap().to_vec();
        let mut destination = host_destination.to_cuda(cuda).unwrap();
        let mut host_destination = host_destination;
        let expected = host_lhs
            .contract_into(engine, host_rhs, &spec, &mut host_destination, 1.0, 0.0)
            .unwrap_err()
            .report();
        let (actual, counters) = delta(|| {
            lhs.contract_into(cuda, &rhs, &spec, &mut destination, 1.0, 0.0)
                .unwrap_err()
                .report()
        });
        assert_eq!(actual, device_text(expected), "{spec:?}");
        assert_eq!(counters, CudaTransferStats::default(), "{actual}");
        assert_eq!(
            destination.to_host(cuda).unwrap().dense_data().unwrap(),
            before.as_slice()
        );
    }

    // Runtime, shared ownership and alias. Every alpha and beta is supported,
    // so there is no device scale boundary left to reject.
    let mut destination = poison(engine).to_cuda(cuda).unwrap();
    let stranger = u1_rank_five::<f64>(engine, &other)
        .rhs
        .to_cuda(cuda)
        .unwrap();
    let shared = destination.clone();
    let (errors, counters) = delta(|| {
        let spec = case.spec();
        let mut errors = vec![
            lhs.contract_into(cuda, &stranger, &spec, &mut destination, 1.0, 0.0)
                .unwrap_err(),
            lhs.contract_into(cuda, &rhs, &spec, &mut destination, 1.0, 0.0)
                .unwrap_err(),
        ];
        let mut lhs_alias = lhs.clone();
        errors.push(
            lhs.contract_into(cuda, &rhs, &spec, &mut lhs_alias, 1.0, 0.0)
                .unwrap_err(),
        );
        errors
    });
    assert!(
        matches!(errors[0], symtn::typed::Error::RuntimeMismatch),
        "{:?}",
        errors[0]
    );
    assert!(
        errors[1].to_string().contains("shares its storage"),
        "{}",
        errors[1]
    );
    assert!(errors[2].to_string().contains("alias"), "{}", errors[2]);
    drop(shared);
    assert_eq!(counters, CudaTransferStats::default(), "{counters:?}");
    assert_eq!(
        destination.to_host(cuda).unwrap().dense_data().unwrap(),
        poison_data.as_slice()
    );
}
