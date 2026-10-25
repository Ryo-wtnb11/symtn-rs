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
use crate::counting_alloc;

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
fn check<R, D>(case: Case<R, D>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let host = case.host();
    let device = case
        .lhs
        .to_cuda()
        .unwrap()
        .contract(&case.rhs.to_cuda().unwrap(), &case.spec())
        .unwrap()
        .to_host()
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
    let oracle = blas_contract_oracle(&case);
    numerics::assert_nonzero_slices_close(
        case.name,
        device.dense_data().unwrap(),
        oracle.dense_data().unwrap(),
        case.terms(),
    );
}

fn every_fixture<D: DevicePayload>(runtime: &Runtime) {
    check(u1_rank_five::<D>(runtime));
    check(u1_reordered::<D>(runtime));
    check(su2_reordered::<D>(runtime));
    check(su2_bent::<D>(runtime));
    check(product_general::<D>(runtime));
    check(u1_lhs_identity::<D>(runtime));
    check(u1_rhs_identity::<D>(runtime));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn general_axes_match_the_host_and_the_blas_contract_oracle_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    every_fixture::<f64>(&runtime);
    every_fixture::<Complex64>(&runtime);
    every_fixture::<f32>(&runtime);
    every_fixture::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn general_axes_match_the_physical_basis_contraction() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn dense<R, D>(case: Case<R, D>)
    where
        R: DeviceRule + symtn::sector::PhysicalFusionBasis<Scalar = f64>,
        D: DevicePayload,
    {
        assert!(case.dense, "{}", case.name);
        let (shape, expected) = dense_oracle(&case);
        let device = case
            .lhs
            .to_cuda()
            .unwrap()
            .contract(&case.rhs.to_cuda().unwrap(), &case.spec())
            .unwrap()
            .to_host()
            .unwrap()
            .to_physical_dense()
            .unwrap();
        assert_eq!(device.shape, shape, "{}", case.name);
        numerics::assert_nonzero_slices_close(case.name, &device.data, &expected, case.terms());
    }
    dense(u1_reordered::<f64>(&runtime));
    dense(u1_reordered::<Complex64>(&runtime));
    dense(su2_reordered::<f64>(&runtime));
    dense(su2_reordered::<Complex64>(&runtime));
}

fn lazy_at<D: DevicePayload>(runtime: &Runtime) {
    for case in lazy_cases(&u1_rank_five::<D>(runtime).lhs, "U(1) rank 5 lazy") {
        check(case);
    }
    for case in lazy_cases(&su2_reordered::<D>(runtime).lhs, "SU(2) lazy") {
        check(case);
    }
    for case in lazy_cases(&product_general::<D>(runtime).lhs, "U(1) x SU(2) lazy") {
        check(case);
    }
}

/// The zero-copy sorted/swapped candidates (#1468) run the device Core
/// route, swapped operands and lazy-adjoint GEMM flags included. The
/// fermionic probes take the twisted `blas_contract!` oracle: C2 and L5
/// contract dual B legs in their literal sequence, which TensorKit twists.
#[test]
#[ignore = "requires a real CUDA device"]
fn zero_copy_candidates_match_the_host_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn at<D: DevicePayload>(runtime: &Runtime) {
        for (case, _) in candidate_core_probes::<_, D>(runtime, &u1_non_self_dual()) {
            check(case);
        }
        for (case, _) in candidate_core_probes::<_, D>(runtime, &su2()) {
            check(case);
        }
        let twist = |t: &TensorMap<FermionU1, D>, legs: &[usize]| {
            t.twist(legs, Direction::Forward).unwrap()
        };
        for (case, _) in candidate_core_probes::<_, D>(runtime, &fermion_u1()) {
            check_fermionic(case, twist);
        }
    }
    at::<f64>(&runtime);
    at::<Complex64>(&runtime);
    at::<f32>(&runtime);
    at::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn core_form_lazy_contractions_match_the_host_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    // Values only. The planner's `copyC` route for this geometry is pinned by
    // the route-parity test in symtn-tensors (`storage_contract_tests.rs`).
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn at<D: DevicePayload>(runtime: &Runtime) {
        for case in su2_core_form_lazy_cases::<D>(runtime) {
            check(case);
        }
    }
    at::<f64>(&runtime);
    at::<Complex64>(&runtime);
    at::<f32>(&runtime);
    at::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn lazy_adjoint_operands_match_the_host_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    lazy_at::<f64>(&runtime);
    lazy_at::<Complex64>(&runtime);
    lazy_at::<f32>(&runtime);
    lazy_at::<Complex32>(&runtime);
}

/// #2147: a lazy adjoint whose tree transform is the identity, against a
/// permuted partner, runs the device DynamicTree with the adjoint read from
/// its parent's device buffer under a GEMM adjoint op (TensorKit
/// `blas_contract!` without `copyA`, `tensoroperations.jl:383-450` @cfaa073).
/// Route witness: the contraction scratch holds exactly the copied partner
/// (Core would hold nothing, CopyC the output, a copied adjoint `|t|` more).
/// Values: device == Host == the materialize-then-contract oracle.
#[test]
#[ignore = "requires a real CUDA device"]
fn identity_adjoint_operands_are_borrowed_by_the_device_dynamic_tree() {
    let _serial = crate::counting_alloc::serial();
    fn at<R: DeviceRule, D: DevicePayload>(v: &symtn::typed::GradedSpace<R>, symmetry: &str) {
        for index in 0..3 {
            // A fresh Runtime per case: its scratch high-water mark is this
            // case's alone.
            let runtime = Runtime::builder().cuda(0).build().unwrap();
            let (case, lazy_lhs) = identity_adjoint_cases::<R, D>(&runtime, v)
                .into_iter()
                .nth(index)
                .unwrap();
            let name = format!("{symmetry} {} {}", case.name, D::NAME);
            let host = case.host();
            let oracle = case
                .lhs
                .materialize()
                .unwrap()
                .contract(&case.rhs.materialize().unwrap(), &case.spec())
                .unwrap();
            let device = case
                .lhs
                .to_cuda()
                .unwrap()
                .contract(&case.rhs.to_cuda().unwrap(), &case.spec())
                .unwrap()
                .to_host()
                .unwrap();
            let partner = if lazy_lhs { &case.rhs } else { &case.lhs };
            let partner_bytes = std::mem::size_of_val(partner.dense_data().unwrap());
            assert_ne!(
                partner_bytes,
                std::mem::size_of_val(host.dense_data().unwrap()),
                "{name}: the witness must tell DynamicTree from CopyC"
            );
            assert_eq!(
                runtime.cuda_contract_scratch_bytes().unwrap(),
                partner_bytes,
                "{name}: only the permuted partner is copied"
            );
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
    at::<_, f64>(&su2(), "SU(2)");
    at::<_, Complex64>(&su2(), "SU(2)");
    at::<_, f64>(&u1_non_self_dual(), "U(1)");
    at::<_, Complex64>(&u1_non_self_dual(), "U(1)");
}

/// #1857: device `contract` / `contract_into` take TensorKit's `copyC`
/// (`blas_contract!`, `tensoroperations.jl:436-446` @cfaa073) where the batched
/// `ContractPlan` does: the zero-copy core into the core-destination scratch,
/// then one permute. Traffic: the only scratch is that temporary, `dim(C)`
/// elements — no operand, owned or lazy adjoint, is materialized (the
/// pre-#1857 device route copied the lazy adjoint `X`, `≥ |X|` more scratch).
#[test]
#[ignore = "requires a real CUDA device"]
fn copy_c_probes_move_only_c_and_match_the_host_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_shared();
    fn at<R: DeviceRule, D: DevicePayload>(
        v: &symtn::typed::GradedSpace<R>,
        symmetry: &str,
        oracle: impl Fn(&Case<R, D>) -> TensorMap<R, D>,
    ) {
        for index in 0..copy_c_probes::<R, D>(&Runtime::builder().build().unwrap(), v).len() {
            // A fresh Runtime per probe: its scratch high-water mark is this
            // probe's alone.
            let runtime = Runtime::builder().cuda(0).build().unwrap();
            let (case, _) = copy_c_probes::<R, D>(&runtime, v).swap_remove(index);
            let name = format!("{symmetry} {} {}", case.name, D::NAME);
            let host = case.host();
            let output_bytes = std::mem::size_of_val(host.dense_data().unwrap());
            let lhs = case.lhs.to_cuda().unwrap();
            let rhs = case.rhs.to_cuda().unwrap();
            let device = lhs.contract(&rhs, &case.spec()).unwrap();
            assert_eq!(
                runtime.cuda_contract_scratch_bytes().unwrap(),
                output_bytes,
                "{name}: the copyC temporary is the only scratch"
            );
            numerics::assert_nonzero_slices_close(
                &name,
                device.to_host().unwrap().dense_data().unwrap(),
                host.dense_data().unwrap(),
                case.terms(),
            );
            numerics::assert_nonzero_slices_close(
                &name,
                device.to_host().unwrap().dense_data().unwrap(),
                oracle(&case).dense_data().unwrap(),
                case.terms(),
            );
            let mut destination = poisoned_destination(&case).to_cuda().unwrap();
            let (alpha, beta) = (D::entry(0.75, 0.25), D::entry(0.0, 0.0));
            let (_, warm) = delta(|| {
                lhs.contract_into(&rhs, &case.spec(), &mut destination, alpha, beta)
                    .unwrap()
            });
            assert_eq!(
                (warm.h2d_calls, warm.d2h_calls, warm.device_allocs),
                (0, 0, 0),
                "{name}: {warm:?}"
            );
            assert_eq!(
                runtime.cuda_contract_scratch_bytes().unwrap(),
                output_bytes,
                "{name}: contract_into reuses the temporary"
            );
            numerics::assert_nonzero_slices_close(
                &name,
                destination.to_host().unwrap().dense_data().unwrap(),
                host.scale(alpha).unwrap().dense_data().unwrap(),
                case.terms(),
            );
        }
    }
    at::<_, f64>(&u1_non_self_dual(), "U(1)", blas_contract_oracle);
    at::<_, Complex64>(&u1_non_self_dual(), "U(1)", blas_contract_oracle);
    at::<_, f32>(&su2(), "SU(2)", blas_contract_oracle);
    at::<_, Complex32>(&su2(), "SU(2)", blas_contract_oracle);
    at::<_, f64>(&su2(), "SU(2)", blas_contract_oracle);
    // The swapped probes' literal sequence twists dual `B` legs the selected
    // candidate does not, so TensorKit's fermionic sequence is the oracle.
    at::<_, f64>(&fermion_u1(), "fZ2xU(1)", |case| {
        fermionic_blas_contract_oracle(case, TwistRole::B, |t, legs| {
            t.twist(legs, Direction::Forward).unwrap()
        })
    });
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_general_contraction_uploads_only_its_output() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_shared();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_rank_five::<f64>(&runtime);
    let output_bytes = std::mem::size_of_val(case.host().dense_data().unwrap()) as u64;
    let lhs = case.lhs.to_cuda().unwrap();
    let rhs = case.rhs.to_cuda().unwrap();
    let call = || lhs.contract(&rhs, &case.spec()).unwrap();

    // Cold: the output, the coefficient payloads and the scratch high-water
    // zero uploads.
    let (_, cold) = delta(call);
    assert!(cold.h2d_calls > 1, "{cold:?}");
    assert_eq!(cold.d2h_calls, 0, "{cold:?}");
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    assert!(scratch > 0, "a transformed operand lives in the scratch");
    let transforms = runtime.cuda_tree_transform_stats().unwrap();

    // Warm: exactly the #740 output initialisation.
    let (_, warm) = delta(call);
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(
        runtime.cuda_contract_scratch_bytes().unwrap(),
        scratch,
        "a warm call grows no scratch"
    );
    assert_eq!(
        runtime.cuda_tree_transform_stats().unwrap(),
        transforms,
        "a warm call prepares no transform state"
    );
    println!(
        "u1_rank_five f64: output {output_bytes} B; cold {cold:?}; warm {warm:?}; \
         scratch {scratch} B; {transforms:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn the_scratch_grows_only_at_a_high_water_mark_and_is_released_by_the_clear_path() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let large = u1_rank_five::<f64>(&runtime);
    let small = u1_reordered::<f64>(&runtime);
    let run = |case: &Case<_, f64>| {
        case.lhs
            .to_cuda()
            .unwrap()
            .contract(&case.rhs.to_cuda().unwrap(), &case.spec())
            .unwrap()
            .to_host()
            .unwrap()
    };
    let _ = run(&large);
    let small_lhs = small.lhs.to_cuda().unwrap();
    let small_rhs = small.rhs.to_cuda().unwrap();
    // Warm the small case's transform structures (and any buffer it needs
    // wider than the large case did) so only its output is left.
    let _ = small_lhs.contract(&small_rhs, &small.spec()).unwrap();
    let high_water = runtime.cuda_contract_scratch_bytes().unwrap();
    let (result, counters) = delta(|| small_lhs.contract(&small_rhs, &small.spec()).unwrap());
    // What: a smaller operand narrows the retained buffers instead of
    // reallocating them — one allocation, the returned output.
    assert_eq!(counters.device_allocs, 1, "{counters:?}");
    assert_eq!(counters.h2d_calls, 1, "{counters:?}");
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), high_water);
    numerics::assert_nonzero_slices_close(
        "narrowed scratch",
        result.to_host().unwrap().dense_data().unwrap(),
        small.host().dense_data().unwrap(),
        small.terms(),
    );
    // And the large one again, after the narrowing, is still right.
    numerics::assert_nonzero_slices_close(
        "re-widened",
        run(&large).dense_data().unwrap(),
        large.host().dense_data().unwrap(),
        large.terms(),
    );
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), high_water);

    runtime.clear_cuda_caches();
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), 0);
    numerics::assert_nonzero_slices_close(
        "after clear",
        run(&small).dense_data().unwrap(),
        small.host().dense_data().unwrap(),
        small.terms(),
    );
}

/// Orders process-global cache clears against warm/cold count windows
/// (#2084). Cooperative: every new clearing test must take `cache_exclusive`
/// and every new count-dependent test `cache_shared`.
static CACHE_TEST_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

fn cache_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    CACHE_TEST_LOCK.read().unwrap_or_else(|e| e.into_inner())
}

fn cache_exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
    CACHE_TEST_LOCK.write().unwrap_or_else(|e| e.into_inner())
}

fn device_contract<R: DeviceRule, D: DevicePayload>(case: &Case<R, D>) -> TensorMap<R, D> {
    case.lhs
        .to_cuda()
        .unwrap()
        .contract(&case.rhs.to_cuda().unwrap(), &case.spec())
        .unwrap()
        .to_host()
        .unwrap()
}

/// Device == Host and device == TensorKit's `blas_contract!` with the twist
/// on the B role and on the A role (#1347).
fn check_fermionic<R, D>(
    case: Case<R, D>,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let device = device_contract(&case);
    let host = case.host();
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
        let oracle = fermionic_blas_contract_oracle(&case, role, twist);
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
    let _serial = crate::counting_alloc::serial();
    // General axes with the twist on one or both contracted legs, the
    // canonical form whose θ varies within one coupled sector (lifted from
    // the storage core route to DynamicTree), and lazy adjoints on either
    // side; fZ2 x U(1) and fZ2 (x) SU(2). Non-vacuity — the twist changes
    // every one of these results — is pinned on the Host, ungated.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for_each_fermionic_fixture!(&runtime, f64, check_fermionic);
    for_each_fermionic_fixture!(&runtime, Complex64, check_fermionic);
    for_each_fermionic_fixture!(&runtime, f32, check_fermionic);
    for_each_fermionic_fixture!(&runtime, Complex32, check_fermionic);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn fz2_loops_as_explicit_device_contracts_match_tensorkit() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let loops = fz2_tensorkit_loops(
        &runtime,
        |tensor| tensor.to_cuda().unwrap(),
        |x, y, spec| x.contract(y, spec).unwrap(),
        |tensor| tensor.to_host().unwrap().scalar().unwrap(),
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
fn a_warm_fermionic_contraction_uploads_only_its_output() {
    let _serial = crate::counting_alloc::serial();
    // What: θ travels as descriptor scalars, so a warm twisted contraction
    // costs exactly what an untwisted one does — the #740 output upload.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case: Case<_, f64> = fermionic_general(&runtime, &fermion_su2(), true, "fZ2xSU2", 51);
    let output_bytes = std::mem::size_of_val(case.host().dense_data().unwrap()) as u64;
    let lhs = case.lhs.to_cuda().unwrap();
    let rhs = case.rhs.to_cuda().unwrap();
    let call = || lhs.contract(&rhs, &case.spec()).unwrap();
    let _ = call();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let (result, warm) = delta(call);
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    numerics::assert_nonzero_slices_close(
        "warm",
        result.to_host().unwrap().dense_data().unwrap(),
        case.host().dense_data().unwrap(),
        case.terms(),
    );
    println!("fZ2xSU2 mixed θ f64: output {output_bytes} B; warm {warm:?}; {transforms:?}");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rejections_happen_before_any_device_work_and_match_the_host() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_rank_five::<f64>(&runtime);
    let host_lhs = &case.lhs;
    let host_rhs = &case.rhs;
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    let stranger = u1_rank_five::<f64>(&other).rhs.to_cuda().unwrap();
    let _ = lhs.contract(&rhs, &case.spec()).unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    // Malformed: an axis out of range, a repeated axis, an output order that
    // is not a permutation, and a pairing of two equal (not dual) legs.
    let (errors, counters) = delta(|| {
        let mut errors = Vec::new();
        for spec in malformed_rank_five_specs() {
            let expected = host_lhs.contract(host_rhs, &spec).unwrap_err().report();
            let actual = lhs.contract(&rhs, &spec).unwrap_err().report();
            assert_eq!(actual, expected, "device error text must be the Host's");
            errors.push(actual);
        }
        errors.push(
            lhs.contract(&stranger, &case.spec())
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
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
}

/// Every split size of `case`'s open legs, in its own output order and
/// reversed (#1549): the device result is the Host one, which
/// `contract_spec_split.rs` pins against contract-then-permute. The returning
/// and the overwrite forms both run.
fn check_splits<R: DeviceRule, D: DevicePayload>(case: Case<R, D>) {
    let lhs = case.lhs.to_cuda().unwrap();
    let rhs = case.rhs.to_cuda().unwrap();
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
            let host = case.lhs.contract(&case.rhs, &spec).unwrap();
            let device = lhs.contract(&rhs, &spec).unwrap().to_host().unwrap();
            assert_eq!(device.codomain(), host.codomain(), "{label}");
            assert_eq!(device.domain(), host.domain(), "{label}");
            numerics::assert_nonzero_slices_close(
                &label,
                device.dense_data().unwrap(),
                host.dense_data().unwrap(),
                case.terms(),
            );
            let mut destination = host.scale(D::entry(7.5, 0.0)).unwrap().to_cuda().unwrap();
            lhs.contract_into(
                &rhs,
                &spec,
                &mut destination,
                D::entry(1.0, 0.0),
                D::entry(0.0, 0.0),
            )
            .unwrap();
            numerics::assert_nonzero_slices_close(
                &label,
                destination.to_host().unwrap().dense_data().unwrap(),
                host.dense_data().unwrap(),
                case.terms(),
            );
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn every_split_matches_the_host_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    check_splits(u1_rank_five::<f64>(&runtime));
    check_splits(su2_bent::<Complex64>(&runtime));
    check_splits(product_general::<f32>(&runtime));
    check_splits(fermionic_general::<_, Complex32>(
        &runtime,
        &fermion_su2(),
        true,
        "fZ2xSU2",
        52,
    ));
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
fn device_overwrite<R: DeviceRule, D: DevicePayload>(case: &Case<R, D>) -> TensorMap<R, D> {
    let mut destination = poisoned_destination(case).to_cuda().unwrap();
    case.lhs
        .to_cuda()
        .unwrap()
        .contract_into(
            &case.rhs.to_cuda().unwrap(),
            &case.spec(),
            &mut destination,
            D::entry(1.0, 0.0),
            D::entry(0.0, 0.0),
        )
        .unwrap();
    destination.to_host().unwrap()
}

fn check_overwrite<R: DeviceRule, D: DevicePayload>(case: Case<R, D>) {
    let written = device_overwrite(&case);
    numerics::assert_nonzero_slices_close(
        case.name,
        written.dense_data().unwrap(),
        case.host().dense_data().unwrap(),
        case.terms(),
    );
    let oracle = blas_contract_oracle(&case);
    numerics::assert_nonzero_slices_close(
        case.name,
        written.dense_data().unwrap(),
        oracle.dense_data().unwrap(),
        case.terms(),
    );
}

fn check_overwrite_fermionic<R, D>(
    case: Case<R, D>,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let written = device_overwrite(&case);
    numerics::assert_nonzero_slices_close(
        case.name,
        written.dense_data().unwrap(),
        case.host().dense_data().unwrap(),
        case.terms(),
    );
    for role in [TwistRole::B, TwistRole::A] {
        let oracle = fermionic_blas_contract_oracle(&case, role, twist);
        numerics::assert_nonzero_slices_close(
            &format!("{} overwrite vs the {role:?}-role oracle", case.name),
            written.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            case.terms(),
        );
    }
}

fn every_overwrite_fixture<D: DevicePayload>(runtime: &Runtime) {
    check_overwrite(u1_rank_five::<D>(runtime));
    check_overwrite(u1_reordered::<D>(runtime));
    check_overwrite(su2_reordered::<D>(runtime));
    check_overwrite(su2_bent::<D>(runtime));
    check_overwrite(product_general::<D>(runtime));
    check_overwrite(u1_lhs_identity::<D>(runtime));
    check_overwrite(u1_rhs_identity::<D>(runtime));
    for case in u1_inactive_cases::<D>(runtime) {
        check_overwrite(case);
    }
    for case in lazy_cases(&u1_rank_five::<D>(runtime).lhs, "U(1) rank 5 lazy") {
        check_overwrite(case);
    }
    for case in lazy_cases(&su2_reordered::<D>(runtime).lhs, "SU(2) lazy") {
        check_overwrite(case);
    }
    for case in lazy_cases(&product_general::<D>(runtime).lhs, "U(1) x SU(2) lazy") {
        check_overwrite(case);
    }
    for case in su2_core_form_lazy_cases::<D>(runtime) {
        check_overwrite(case);
    }
    for_each_fermionic_fixture!(runtime, D, check_overwrite_fermionic);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn contract_into_a_poisoned_destination_matches_the_host_and_the_oracles_at_every_dtype() {
    let _serial = crate::counting_alloc::serial();
    // What: general axes, lazy adjoints, the core-form lazy (`copyC`) class, both
    // fermionic twist roles, and destinations with blocks no GEMM writes —
    // by the core route, by an identity output and under an output transform
    // — each rewritten over NaN.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    every_overwrite_fixture::<f64>(&runtime);
    every_overwrite_fixture::<Complex64>(&runtime);
    every_overwrite_fixture::<f32>(&runtime);
    every_overwrite_fixture::<Complex32>(&runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_overwrite_transfers_and_allocates_nothing() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_shared();
    // What: no reset and no output initialisation — a warm overwrite costs
    // no upload, no download, no device allocation, no scratch growth and no
    // transform preparation, for a transformed general contraction, a
    // twisted one, and a core-route destination with an inactive block.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    fn warm<R: DeviceRule>(runtime: &Runtime, case: Case<R, f64>) {
        let lhs = case.lhs.to_cuda().unwrap();
        let rhs = case.rhs.to_cuda().unwrap();
        let mut destination = poisoned_destination(&case).to_cuda().unwrap();
        let mut call = || {
            lhs.contract_into(&rhs, &case.spec(), &mut destination, 1.0, 0.0)
                .unwrap()
        };
        call();
        let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
        let transforms = runtime.cuda_tree_transform_stats().unwrap();
        let plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        let ((), counters) = delta(&mut call);
        let after_plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        // #1348: a warm overwrite builds no cuTENSOR plan, observed through
        // the typed Runtime accessor rather than the executor.
        assert_eq!(
            (after_plans.misses, after_plans.evictions),
            (plans.misses, plans.evictions),
            "{}: cuTENSOR plan cache",
            case.name
        );
        assert!(after_plans.hits > plans.hits, "{}: vacuous", case.name);
        // #1348: the warm Host allocation count is a steady state — the
        // same on every warm call. The Host route is compiled on every call
        // (#1359 removed its rank-sized allocations); the inactive-region
        // list is rewritten in place in the lease's contract scratch.
        let host_allocations = [
            counting_alloc::measure(&mut call).1.calls,
            counting_alloc::measure(&mut call).1.calls,
            counting_alloc::measure(&mut call).1.calls,
        ];
        eprintln!("{}: warm host allocations {host_allocations:?}", case.name);
        assert!(host_allocations[0] > 0, "{}: vacuous count", case.name);
        assert!(
            host_allocations
                .iter()
                .all(|&count| count == host_allocations[0]),
            "{}: warm host allocations {host_allocations:?} are not steady",
            case.name
        );
        assert_eq!(
            (
                counters.h2d_calls,
                counters.d2h_calls,
                counters.device_allocs
            ),
            (0, 0, 0),
            "{}: {counters:?}",
            case.name
        );
        assert!(counters.gemm_calls > 0, "{}: vacuous", case.name);
        assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
        assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
        numerics::assert_nonzero_slices_close(
            case.name,
            destination.to_host().unwrap().dense_data().unwrap(),
            case.host().dense_data().unwrap(),
            case.terms(),
        );
    }
    warm(&runtime, u1_rank_five::<f64>(&runtime));
    warm(
        &runtime,
        fermionic_general(&runtime, &fermion_su2(), true, "fZ2xSU2 mixed θ", 51),
    );
    let [core, identity_output, output_transform] = u1_inactive_cases::<f64>(&runtime);
    warm(&runtime, core);
    warm(&runtime, identity_output);
    warm(&runtime, output_transform);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn overwrite_rejections_match_the_host_in_order_and_leave_the_destination_untouched() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other = Runtime::builder().cuda(0).build().unwrap();
    let case = u1_rank_five::<f64>(&runtime);
    let (host_lhs, host_rhs) = (&case.lhs, &case.rhs);
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    let poison = || case.host().scale(7.5).unwrap();
    let poison_data = poison().dense_data().unwrap().to_vec();
    let _ = lhs.contract(&rhs, &case.spec()).unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let scratch = runtime.cuda_contract_scratch_bytes().unwrap();
    let device_text = |text: String| text.replace("host storage", "CUDA storage");

    // Host-shared rejections, each against a fresh poisoned destination:
    // malformed axes, an output order that is not a permutation, a pairing
    // of two equal (not dual) legs, and a destination of another space.
    let foreign = || u1_reordered::<f64>(&runtime).host().scale(7.5).unwrap();
    let cases = malformed_rank_five_specs()
        .into_iter()
        .map(|spec| (spec, poison()))
        .chain([(case.spec(), foreign())]);
    for (spec, host_destination) in cases {
        let before = host_destination.dense_data().unwrap().to_vec();
        let mut destination = host_destination.to_cuda().unwrap();
        let mut host_destination = host_destination;
        let expected = host_lhs
            .contract_into(host_rhs, &spec, &mut host_destination, 1.0, 0.0)
            .unwrap_err()
            .report();
        let (actual, counters) = delta(|| {
            lhs.contract_into(&rhs, &spec, &mut destination, 1.0, 0.0)
                .unwrap_err()
                .report()
        });
        assert_eq!(actual, device_text(expected), "{spec:?}");
        assert_eq!(counters, CudaTransferStats::default(), "{actual}");
        assert_eq!(
            destination.to_host().unwrap().dense_data().unwrap(),
            before.as_slice()
        );
    }

    // Runtime, shared ownership and alias. Every alpha and beta is supported
    // (#1550), so there is no device scale boundary left to reject.
    let mut destination = poison().to_cuda().unwrap();
    let stranger = u1_rank_five::<f64>(&other).rhs.to_cuda().unwrap();
    let shared = destination.clone();
    let (errors, counters) = delta(|| {
        let spec = case.spec();
        let mut errors = vec![
            lhs.contract_into(&stranger, &spec, &mut destination, 1.0, 0.0)
                .unwrap_err(),
            lhs.contract_into(&rhs, &spec, &mut destination, 1.0, 0.0)
                .unwrap_err(),
        ];
        let mut lhs_alias = lhs.clone();
        errors.push(
            lhs.contract_into(&rhs, &spec, &mut lhs_alias, 1.0, 0.0)
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
        destination.to_host().unwrap().dense_data().unwrap(),
        poison_data.as_slice()
    );
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_eq!(runtime.cuda_contract_scratch_bytes().unwrap(), scratch);
}
