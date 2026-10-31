//! Real-device gates for general-axes contraction on device tensors
//! (issue #1345): the Host `DynamicTree` artifact replayed by device
//! executors.
//!
//! Evidence chain:
//!
//! 1. the executor replaying one artifact is gated against the Host replay of
//!    the *same* artifact for both forced orientations and every axis-order
//!    candidate in `symtn-tensors` (`storage_contract_tests.rs`), where the
//!    test-only plan builder can force them; the NaN-poisoned core-destination
//!    scratch is gated there too;
//! 2. the typed lowering is gated here: device == Host of the same public
//!    call, at every device payload dtype, owned and lazy-adjoint;
//! 3. independence: device == TensorKit's `blas_contract!` step sequence on
//!    Host typed ops (bosonic providers), and device == the physical-basis
//!    dense contraction (U(1), SU(2)). Both oracles are pinned against the
//!    Host by the ungated `typed_contract_host_oracle.rs`;
//! 4. fermionic providers (#1347): device == Host == TensorKit's
//!    sequence with the twist on either role, and the TensorKit-valued FZ2
//!    loops as explicit device `contract` calls;
//! 5. `contract_into` (#1346): into a NaN-poisoned device
//!    destination, device == Host eager `contract` (whose overwrite twin is
//!    pinned ungated) == the same independent oracles, on every fixture
//!    above plus destinations with blocks no GEMM writes; warm calls transfer
//!    and allocate nothing; rejections leave the destination untouched.
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
    blas_contract_oracle, candidate_core_probes, copy_c_probes, dense_oracle, fermion_su2,
    fermion_u1, fermionic_blas_contract_oracle, fermionic_general, fz2_tensorkit_loops,
    identity_adjoint_cases, lazy_cases, poisoned_destination, product_general, su2, su2_bent,
    su2_core_form_lazy_cases, su2_reordered, u1_inactive_cases, u1_lhs_identity, u1_non_self_dual,
    u1_rank_five, u1_reordered, u1_rhs_identity, Case, FermionU1, TwistRole,
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

fn every_fixture<D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    engine: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) {
    let fixture = u1_rank_five::<D>(engine, runtime);
    check(engine, cuda, fixture);
    let fixture = u1_reordered::<D>(engine, runtime);
    check(engine, cuda, fixture);
    let fixture = su2_reordered::<D>(engine, runtime);
    check(engine, cuda, fixture);
    let fixture = su2_bent::<D>(engine, runtime);
    check(engine, cuda, fixture);
    let fixture = product_general::<D>(engine, runtime);
    check(engine, cuda, fixture);
    let fixture = u1_lhs_identity::<D>(engine, runtime);
    check(engine, cuda, fixture);
    let fixture = u1_rhs_identity::<D>(engine, runtime);
    check(engine, cuda, fixture);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn general_axes_match_the_host_and_the_blas_contract_oracle_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    every_fixture::<f64>(cuda, engine, &runtime);
    every_fixture::<Complex64>(cuda, engine, &runtime);
    every_fixture::<f32>(cuda, engine, &runtime);
    every_fixture::<Complex32>(cuda, engine, &runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn general_axes_match_the_physical_basis_contraction() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    fn dense<R, D>(
        engine: &mut dyn symtn::typed::Engine,
        cuda: &mut dyn symtn::typed::Engine,
        case: Case<R, D>,
    ) where
        R: DeviceRule + symtn::sector::PhysicalFusionBasis<Scalar = f64>,
        D: DevicePayload,
    {
        assert!(case.dense, "{}", case.name);
        let (shape, expected) = dense_oracle(engine, &case);
        let rhs = case.rhs.to_cuda(cuda).unwrap();
        let device = case
            .lhs
            .to_cuda(cuda)
            .unwrap()
            .contract(cuda, &rhs, &case.spec())
            .unwrap()
            .to_host(cuda)
            .unwrap()
            .to_physical_dense(engine)
            .unwrap();
        assert_eq!(device.shape, shape, "{}", case.name);
        numerics::assert_nonzero_slices_close(case.name, &device.data, &expected, case.terms());
    }
    let case = u1_reordered::<f64>(engine, &runtime);
    dense(engine, cuda, case);
    let case = u1_reordered::<Complex64>(engine, &runtime);
    dense(engine, cuda, case);
    let case = su2_reordered::<f64>(engine, &runtime);
    dense(engine, cuda, case);
    let case = su2_reordered::<Complex64>(engine, &runtime);
    dense(engine, cuda, case);
}

fn lazy_at<D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) {
    let operand = u1_rank_five::<D>(engine, runtime).lhs;
    for case in lazy_cases(engine, &operand, "U(1) rank 5 lazy") {
        check(engine, cuda, case);
    }
    let operand = su2_reordered::<D>(engine, runtime).lhs;
    for case in lazy_cases(engine, &operand, "SU(2) lazy") {
        check(engine, cuda, case);
    }
    let operand = product_general::<D>(engine, runtime).lhs;
    for case in lazy_cases(engine, &operand, "U(1) x SU(2) lazy") {
        check(engine, cuda, case);
    }
}

/// The zero-copy sorted/swapped candidates (#1468) run the device Core
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

#[test]
#[ignore = "requires a real CUDA device"]
fn core_form_lazy_contractions_match_the_host_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Values only. The planner's `copyC` route for this geometry is pinned by
    // the route-parity test in symtn-tensors (`storage_contract_tests.rs`).
    let runtime = Runtime::builder().build().unwrap();
    fn at<D: DevicePayload>(
        cuda: &mut dyn symtn::typed::Engine,
        engine: &mut dyn symtn::typed::Engine,
        runtime: &Runtime,
    ) {
        for case in su2_core_form_lazy_cases::<D>(engine, runtime) {
            check(engine, cuda, case);
        }
    }
    at::<f64>(cuda, engine, &runtime);
    at::<Complex64>(cuda, engine, &runtime);
    at::<f32>(cuda, engine, &runtime);
    at::<Complex32>(cuda, engine, &runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn lazy_adjoint_operands_match_the_host_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    lazy_at::<f64>(engine, cuda, &runtime);
    lazy_at::<Complex64>(engine, cuda, &runtime);
    lazy_at::<f32>(engine, cuda, &runtime);
    lazy_at::<Complex32>(engine, cuda, &runtime);
}

/// #2147: a lazy adjoint whose tree transform is the identity, against a
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

/// #1857: device `contract` / `contract_into` take TensorKit's `copyC`
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

/// Orders process-global cache clears against warm/cold count windows
/// (#2084). Cooperative: every new clearing test must take the write lock
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
/// on the B role and on the A role (#1347).
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
fn fermionic_contractions_match_the_host_and_both_tensorkit_twist_roles_at_every_dtype() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    // General axes with the twist on one or both contracted legs, the
    // canonical form whose θ varies within one coupled sector (lifted from
    // the storage core route to DynamicTree), and lazy adjoints on either
    // side; fZ2 x U(1) and fZ2 (x) SU(2). Non-vacuity — the twist changes
    // every one of these results — is pinned on the Host, ungated.
    let runtime = Runtime::builder().build().unwrap();
    for_each_fermionic_fixture!(engine, &runtime, f64, check_fermionic, cuda);
    for_each_fermionic_fixture!(engine, &runtime, Complex64, check_fermionic, cuda);
    for_each_fermionic_fixture!(engine, &runtime, f32, check_fermionic, cuda);
    for_each_fermionic_fixture!(engine, &runtime, Complex32, check_fermionic, cuda);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn fz2_loops_as_explicit_device_contracts_match_tensorkit() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // The helper's closures are `Fn`; each borrows the one device engine in
    // turn, never two at once.
    let cuda = std::cell::RefCell::new(crate::common::cuda_engine());
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let loops = fz2_tensorkit_loops(
        engine,
        &runtime,
        |tensor| tensor.to_cuda(&mut *cuda.borrow_mut()).unwrap(),
        |_, x, y, spec| x.contract(&mut *cuda.borrow_mut(), y, spec).unwrap(),
        |engine, tensor| {
            let host = tensor.to_host(&mut *cuda.borrow_mut()).unwrap();
            host.scalar(engine).unwrap()
        },
    );
    for (name, value, expected) in loops {
        assert!(
            (value - expected).abs() < 1e-12,
            "{name}: {value} vs {expected}"
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
/// reversed (#1549): the device result is the Host one, which
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
// `contract_into` (#1346)
// ---------------------------------------------------------------------------

/// Device `contract_into` of `case` into a NaN-poisoned device
/// destination, downloaded.
fn device_overwrite<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    case: &Case<R, D>,
) -> TensorMap<R, D> {
    let mut destination = poisoned_destination(engine, case).to_cuda(cuda).unwrap();
    let rhs_device = case.rhs.to_cuda(cuda).unwrap();
    case.lhs
        .to_cuda(cuda)
        .unwrap()
        .contract_into(
            cuda,
            &rhs_device,
            &case.spec(),
            &mut destination,
            D::entry(1.0, 0.0),
            D::entry(0.0, 0.0),
        )
        .unwrap();
    destination.to_host(cuda).unwrap()
}

fn check_overwrite<R: DeviceRule, D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    engine: &mut dyn symtn::typed::Engine,
    case: Case<R, D>,
) {
    let written = device_overwrite(engine, cuda, &case);
    numerics::assert_nonzero_slices_close(
        case.name,
        written.dense_data().unwrap(),
        case.host(engine).dense_data().unwrap(),
        case.terms(),
    );
    let oracle = blas_contract_oracle(engine, &case);
    numerics::assert_nonzero_slices_close(
        case.name,
        written.dense_data().unwrap(),
        oracle.dense_data().unwrap(),
        case.terms(),
    );
}

fn check_overwrite_fermionic<R, D>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    case: Case<R, D>,
    twist: impl Fn(&mut dyn symtn::typed::Engine, &TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let written = device_overwrite(engine, cuda, &case);
    numerics::assert_nonzero_slices_close(
        case.name,
        written.dense_data().unwrap(),
        case.host(engine).dense_data().unwrap(),
        case.terms(),
    );
    for role in [TwistRole::B, TwistRole::A] {
        let oracle = fermionic_blas_contract_oracle(engine, &case, role, twist);
        numerics::assert_nonzero_slices_close(
            &format!("{} overwrite vs the {role:?}-role oracle", case.name),
            written.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            case.terms(),
        );
    }
}

fn every_overwrite_fixture<D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    engine: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) {
    let fixture = u1_rank_five::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    let fixture = u1_reordered::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    let fixture = su2_reordered::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    let fixture = su2_bent::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    let fixture = product_general::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    let fixture = u1_lhs_identity::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    let fixture = u1_rhs_identity::<D>(engine, runtime);
    check_overwrite(cuda, engine, fixture);
    for case in u1_inactive_cases::<D>(engine, runtime) {
        check_overwrite(cuda, engine, case);
    }
    let fixture = u1_rank_five::<D>(engine, runtime).lhs;
    for case in lazy_cases(engine, &fixture, "U(1) rank 5 lazy") {
        check_overwrite(cuda, engine, case);
    }
    let fixture = su2_reordered::<D>(engine, runtime).lhs;
    for case in lazy_cases(engine, &fixture, "SU(2) lazy") {
        check_overwrite(cuda, engine, case);
    }
    let fixture = product_general::<D>(engine, runtime).lhs;
    for case in lazy_cases(engine, &fixture, "U(1) x SU(2) lazy") {
        check_overwrite(cuda, engine, case);
    }
    for case in su2_core_form_lazy_cases::<D>(engine, runtime) {
        check_overwrite(cuda, engine, case);
    }
    for_each_fermionic_fixture!(engine, runtime, D, check_overwrite_fermionic, cuda);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn contract_into_a_poisoned_destination_matches_the_host_and_the_oracles_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: general axes, lazy adjoints, the core-form lazy (`copyC`) class, both
    // fermionic twist roles, and destinations with blocks no GEMM writes —
    // by the core route, by an identity output and under an output transform
    // — each rewritten over NaN.
    let runtime = Runtime::builder().build().unwrap();
    every_overwrite_fixture::<f64>(cuda, engine, &runtime);
    every_overwrite_fixture::<Complex64>(cuda, engine, &runtime);
    every_overwrite_fixture::<f32>(cuda, engine, &runtime);
    every_overwrite_fixture::<Complex32>(cuda, engine, &runtime);
}

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

    // Runtime, shared ownership and alias. Every alpha and beta is supported
    // (#1550), so there is no device scale boundary left to reject.
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
