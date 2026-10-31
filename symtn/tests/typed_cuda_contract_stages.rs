//! The CUDA contraction stage sequence (#1859): eager `contract` /
//! `contract_into` on every route (Core, CopyC, DynamicTree) and member
//! replays of one `ContractPlan` on every route.
//!
//! The fixtures are exact in binary: dyadic entries, unit U(1) recoupling and
//! ±1 fermionic twists, scaled by dyadic factors. Every product and partial
//! sum is exact, so any summation order gives the same value and the device
//! is compared bit for bit with the Host on the same machine (±0 and NaN
//! folded): eager device against Host `contract` / `contract_into` at the
//! same alpha and beta, and each member against the Host member replay of
//! the same plan and against eager device `contract`.
//!
//! The transfer and submission counters
//! `(h2d_calls, h2d_bytes, d2h_calls, device_allocs, gemm_calls, copy_calls)`
//! are held to the documented costs of `TensorMap::contract`,
//! `TensorMap::contract_into`, `ContractPlan::execute` and
//! `ContractPlan::execute_into`, not to counts recorded at a base revision.
//!
//! `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test integration typed_cuda_contract_stages:: -- --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

use crate::common;

use crate::contract_cases;

use common::{DevicePayload, DeviceRule};
use contract_cases::{
    candidate_core_probes, copy_c_probes, fermion_u1, fermionic_twist_roles, fill,
    poisoned_destination, u1_inactive_cases, u1_non_self_dual, u1_reordered, Case, FermionU1,
};
use num_complex::Complex64;
use symtn::expert::cuda_transfer_stats;
use symtn::typed::GradedSpace;
use symtn::typed::{ContractPlan, Runtime, StackedTensorMap, TensorMap};

/// `(h2d_calls, h2d_bytes, d2h_calls, device_allocs, gemm_calls, copy_calls)`.
type Counters = [u64; 6];

fn delta<T>(body: impl FnOnce() -> T) -> (T, Counters) {
    let before = cuda_transfer_stats();
    let value = body();
    let after = cuda_transfer_stats();
    (
        value,
        [
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_calls - before.d2h_calls,
            after.device_allocs - before.device_allocs,
            after.gemm_calls - before.gemm_calls,
            after.copy_calls - before.copy_calls,
        ],
    )
}

/// One element as comparable bits: `-0` is `+0`, every NaN is one NaN.
fn folded<D: DevicePayload>(value: D) -> [u64; 2] {
    let fold = |x: f64| {
        if x.is_nan() {
            f64::NAN.to_bits()
        } else if x == 0.0 {
            0
        } else {
            x.to_bits()
        }
    };
    let (re, im) = value.parts();
    [fold(re), fold(im)]
}

fn assert_bit_equal<D: DevicePayload>(what: &str, got: &[D], want: &[D]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            folded(got),
            folded(want),
            "{what}[{index}]: device {got:?} != host {want:?}"
        );
    }
}

/// A destination of `case`'s result space with every block, including the
/// blocks no GEMM writes, holding a nonzero dyadic value.
fn filled_destination<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
    case: &Case<R, D>,
) -> TensorMap<R, D> {
    let host = case.host(engine);
    let (codomain, domain) = (host.codomain(), host.domain());
    TensorMap::from_subblock_fn(engine, runtime, codomain.iter(), domain.iter(), fill(97)).unwrap()
}

/// Eager device `contract` and an alpha/beta grid of `contract_into`, each
/// bit-equal to the Host call, within the documented costs. An eager call
/// builds its device state per call (transform coefficients and scalar
/// templates are uploaded by every call; only a prepared workspace keeps
/// them), so a repeated call costs exactly what the first one did, every
/// upload is a fresh allocation and nothing is downloaded. A `contract`
/// uploads at least its zeroed output (#740); on a core route that is all it
/// uploads, and a warm core `contract_into` transfers nothing. Submissions do not
/// depend on alpha or on beta, except that at `beta = 1` a core route skips
/// the region moves that write `beta * destination` into the coupled sectors
/// no GEMM reaches ([`UNREACHED_CORE_SECTORS`]).
fn eager<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
    case: &Case<R, D>,
    family: &str,
) {
    let name = format!("eager {family} {} {}", case.name, D::NAME);
    let spec = case.spec();
    let host = case.host(engine);
    let output_bytes = std::mem::size_of_val(host.dense_data().unwrap()) as u64;
    let (lhs, rhs) = (
        case.lhs.to_cuda(cuda).unwrap(),
        case.rhs.to_cuda(cuda).unwrap(),
    );
    let (device, cold) = delta(|| lhs.contract(cuda, &rhs, &spec).unwrap());
    assert_bit_equal(
        &format!("{name} contract"),
        device.to_host(cuda).unwrap().dense_data().unwrap(),
        host.dense_data().unwrap(),
    );
    let (_, warm) = delta(|| lhs.contract(cuda, &rhs, &spec).unwrap());
    assert_eq!(
        warm, cold,
        "{name} contract: a repeated eager call costs the first"
    );
    if family == "core" {
        assert_eq!(
            warm[..4],
            [1, output_bytes, 0, 1],
            "{name} contract warm: (h2d_calls, h2d_bytes, d2h_calls, device_allocs)"
        );
    } else {
        assert!(
            warm[1] >= output_bytes,
            "{name} contract uploads its output"
        );
        assert_eq!([warm[2], warm[0]], [0, warm[3]], "{name} contract warm");
    }
    let mut submissions = Vec::new();
    let filled = filled_destination(engine, runtime, case);
    let poisoned = poisoned_destination(engine, case);
    let zero = D::entry(0.0, 0.0);
    let negative_zero = D::entry(-0.0, 0.0);
    for (alpha_label, alpha) in [
        ("0", zero),
        ("1", D::entry(1.0, 0.0)),
        ("-1", D::entry(-1.0, 0.0)),
        ("2.5", D::entry(2.5, 0.0)),
    ] {
        for (beta_label, beta, start) in [
            ("0", zero, &filled),
            ("-0", negative_zero, &filled),
            ("1", D::entry(1.0, 0.0), &filled),
            ("2", D::entry(2.0, 0.0), &filled),
            ("0 poisoned", zero, &poisoned),
            ("-0 poisoned", negative_zero, &poisoned),
        ] {
            let what = format!("{name} into alpha={alpha_label} beta={beta_label}");
            // A deep copy: a clone shares the payload, which `contract_into`
            // rejects as a shared destination.
            let mut expected = start.scale(engine, D::entry(1.0, 0.0)).unwrap();
            case.lhs
                .contract_into(engine, &case.rhs, &spec, &mut expected, alpha, beta)
                .unwrap();
            let mut destination = start.to_cuda(cuda).unwrap();
            let (_, counters) = delta(|| {
                lhs.contract_into(cuda, &rhs, &spec, &mut destination, alpha, beta)
                    .unwrap()
            });
            assert_bit_equal(
                &what,
                destination.to_host(cuda).unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap(),
            );
            // Eager per-call device state: only a warm core call (after the
            // grid's first `contract_into` of this structure) uploads nothing.
            if family == "core" {
                if !submissions.is_empty() {
                    assert_eq!(counters[..4], [0; 4], "{what}: a warm call transfers");
                }
            } else {
                assert_eq!([counters[2], counters[0]], [0, counters[3]], "{what}");
            }
            submissions.push(((alpha_label, beta_label), counters[4..].to_vec()));
        }
    }
    let at = |alpha: &str, beta: &str| {
        submissions
            .iter()
            .find(|(labels, _)| *labels == (alpha, beta))
            .map(|(_, counters)| counters.clone())
            .unwrap()
    };
    let overwrite = at("1", "0");
    assert_eq!(
        at("2.5", "2"),
        overwrite,
        "{name}: beta = 2 against beta = 0"
    );
    assert_eq!(at("0", "0 poisoned"), overwrite, "{name}: alpha = 0");
    let [gemm, copy] = [0, 1].map(|index| (at("1", "1")[index], overwrite[index]));
    assert_eq!(copy.0, copy.1, "{name}: beta = 1 copy_calls");
    if UNREACHED_CORE_SECTORS.contains(&case.name) {
        assert!(gemm.0 < gemm.1, "{name}: beta = 1 skips no move: {gemm:?}");
    } else {
        assert_eq!(gemm.0, gemm.1, "{name}: beta = 1 gemm_calls");
    }
}

/// Fixtures whose core writes the output directly and has coupled sectors
/// no GEMM reaches. Every other fixture's core reaches all of its sectors,
/// or an output transform rewrites the whole destination at every beta.
const UNREACHED_CORE_SECTORS: &[&str] = &[
    "U(1) core route, inactive block",
    "U(1) transformed lhs, identity output, inactive block",
    "fA",
    "fB",
];

fn eager_routes<D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    engine: &mut dyn symtn::typed::Engine,
) {
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_non_self_dual();
    for (case, _) in candidate_core_probes::<_, D>(engine, &runtime, &v) {
        eager(engine, cuda, &runtime, &case, "core");
    }
    for (case, _) in copy_c_probes::<_, D>(engine, &runtime, &v) {
        eager(engine, cuda, &runtime, &case, "copyC");
    }
    for case in u1_inactive_cases::<D>(engine, &runtime) {
        eager(engine, cuda, &runtime, &case, "inactive");
    }
    let host_result = u1_reordered::<D>(engine, &runtime);
    eager(engine, cuda, &runtime, &host_result, "tree");
    for case in fermionic_twist_roles::<FermionU1, D>(
        engine,
        &runtime,
        &fermion_u1(),
        ["fA", "fcanonical", "fB", "fboth"],
        71,
    ) {
        eager(engine, cuda, &runtime, &case, "fermion");
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn eager_device_routes_are_bit_equal_to_the_host_within_documented_costs() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    eager_routes::<f64>(cuda, engine);
    eager_routes::<Complex64>(cuda, engine);
}

fn rescaled<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    case: &Case<R, D>,
    call: usize,
    member: usize,
) -> Case<R, D> {
    let (call, member) = (call as f64, member as f64);
    Case {
        name: case.name,
        lhs: case
            .lhs
            .scale(
                engine,
                D::entry(
                    -0.75 - call / 8.0 + member / 64.0,
                    (call + 1.0) / 16.0 - member / 128.0,
                ),
            )
            .unwrap(),
        rhs: case
            .rhs
            .scale(engine, D::entry(1.5 + call / 4.0 + member / 32.0, 0.0))
            .unwrap(),
        lhs_axes: case.lhs_axes.clone(),
        rhs_axes: case.rhs_axes.clone(),
        output_axes: case.output_axes.clone(),
        dense: case.dense,
    }
}

/// One member plan over the B sequences 1, 2, 17, 1 and 4, 2, 4
/// with fresh values on every call: `execute` (twice, the second warm, on
/// changed inputs after a rejected call) and `execute_into` a NaN-poisoned
/// destination through a second workspace.
/// Every member is bit-equal to the Host member replay of the same plan and
/// to eager device `contract`, within the documented costs: `execute` at a
/// new member count uploads its zeroed output, and a reused output transfers
/// nothing; `execute_into` transfers nothing below the member count's high
/// water; nothing is downloaded; and the submissions of each call do not
/// depend on `B`.
fn members<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    family: &str,
    case: &Case<R, D>,
) {
    let spec = case.spec();
    let member_bytes = std::mem::size_of_val(case.host(engine).dense_data().unwrap()) as u64;
    // Per call form, the `(gemm_calls, copy_calls)` of every call.
    let mut submissions: [Vec<Counters>; 3] = Default::default();
    let host_one =
        |pick: fn(&Case<R, D>) -> &TensorMap<R, D>| StackedTensorMap::pack(&[pick(case)]).unwrap();
    let lhs_one = host_one(|m| &m.lhs);
    let rhs_one = host_one(|m| &m.rhs);
    let plan = ContractPlan::new(
        engine,
        &lhs_one.to_cuda(cuda).unwrap(),
        &rhs_one.to_cuda(cuda).unwrap(),
        &spec,
    )
    .unwrap();
    let host_plan = ContractPlan::new(engine, &lhs_one, &rhs_one, &spec).unwrap();
    for sequence in [&[1usize, 2, 17, 1][..], &[4, 2, 4]] {
        let mut high_water = 0;
        let mut workspace = plan.workspace().unwrap();
        let mut into = plan.workspace().unwrap();
        let mut host_workspace = host_plan.workspace().unwrap();
        for (call, &count) in sequence.iter().enumerate() {
            let key = format!(
                "{family} {} {} seq={sequence:?} call={call} B={count}",
                case.name,
                D::NAME
            );
            let cases: Vec<_> = (0..count)
                .map(|i| rescaled(engine, case, call, i))
                .collect();
            let stack = |pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
                StackedTensorMap::pack(&cases.iter().map(pick).collect::<Vec<_>>()).unwrap()
            };
            let (host_lhs, host_rhs) = (stack(|m| &m.lhs), stack(|m| &m.rhs));
            let (lhs, rhs) = (
                host_lhs.to_cuda(cuda).unwrap(),
                host_rhs.to_cuda(cuda).unwrap(),
            );
            // The first call writes stale values (a later call's scaling) into
            // the output the warm call reuses; a rejected call in between keeps
            // that buffer as the spare (#2123).
            let stale: Vec<_> = (0..count)
                .map(|i| rescaled(engine, case, call + 8, i))
                .collect();
            let stale_stack =
                |cuda: &mut dyn symtn::typed::Engine, pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
                    StackedTensorMap::pack(&stale.iter().map(pick).collect::<Vec<_>>())
                        .unwrap()
                        .to_cuda(cuda)
                        .unwrap()
                };
            let (stale_lhs, stale_rhs) =
                (stale_stack(cuda, |m| &m.lhs), stale_stack(cuda, |m| &m.rhs));
            let longer = StackedTensorMap::pack(&vec![&case.rhs; count + 1])
                .unwrap()
                .to_cuda(cuda)
                .unwrap();
            let (_, first) = delta(|| {
                plan.execute(cuda, &stale_lhs, &stale_rhs, &mut workspace)
                    .unwrap();
            });
            assert!(plan.execute(cuda, &lhs, &longer, &mut workspace).is_err());
            assert!(workspace.take_output().is_none());
            let (_, warm) = delta(|| {
                plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
            });
            let returned = workspace.take_output().unwrap().to_host(cuda).unwrap();
            let poison: Vec<_> = cases
                .iter()
                .map(|case| poisoned_destination(engine, case))
                .collect();
            let mut dst = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
                .unwrap()
                .to_cuda(cuda)
                .unwrap();
            let (_, written_counters) = delta(|| {
                plan.execute_into(cuda, &lhs, &rhs, &mut dst, &mut into)
                    .unwrap();
            });
            let written = dst.to_host(cuda).unwrap();
            let expected = host_plan
                .execute(engine, &host_lhs, &host_rhs, &mut host_workspace)
                .unwrap();
            for (i, member) in cases.iter().enumerate() {
                let rhs_device = member.rhs.to_cuda(cuda).unwrap();
                let eager = member
                    .lhs
                    .to_cuda(cuda)
                    .unwrap()
                    .contract(cuda, &rhs_device, &spec)
                    .unwrap()
                    .to_host(cuda)
                    .unwrap();
                let host = expected.member(engine, i).unwrap();
                for (form, actual) in [("execute", &returned), ("execute_into", &written)] {
                    let actual = actual.member(engine, i).unwrap();
                    for (reference, value) in [("host member", &host), ("eager", &eager)] {
                        assert_bit_equal(
                            &format!("{key} {form} member {i} vs {reference}"),
                            actual.dense_data().unwrap(),
                            value.dense_data().unwrap(),
                        );
                    }
                }
            }
            // Consecutive counts differ, so `first` meets a new member count.
            let output_bytes = count as u64 * member_bytes;
            assert!(
                first[0] >= 1 && first[1] >= output_bytes && first[2] == 0,
                "{key} execute first: {first:?} for a {output_bytes}-byte output"
            );
            assert_eq!(warm[..4], [0; 4], "{key} execute warm");
            assert_eq!(written_counters[2], 0, "{key} execute_into downloads");
            if count <= high_water {
                assert_eq!(written_counters[..4], [0; 4], "{key} execute_into");
            }
            high_water = high_water.max(count);
            for (form, counters) in submissions.iter_mut().zip([first, warm, written_counters]) {
                form.push(counters);
            }
        }
    }
    for (form, counters) in ["execute first", "execute warm", "execute_into"]
        .iter()
        .zip(&submissions)
    {
        assert!(
            counters.iter().all(|c| c[4..] == counters[0][4..]),
            "{family} {} {}: {form} submissions depend on B: {counters:?}",
            case.name,
            D::NAME
        );
    }
}

fn member_routes<D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
) {
    let runtime = Runtime::builder().build().unwrap();
    let [_, identity_output, output_transform] = u1_inactive_cases::<D>(engine, &runtime);
    members(engine, cuda, "member", &identity_output);
    // `ContractPlan` resolves this geometry to CopyC (an output transform
    // over a direct core with one inactive block).
    members(engine, cuda, "member", &output_transform);
    let host_result = u1_reordered::<D>(engine, &runtime);
    members(engine, cuda, "member", &host_result);
    let [twisted_a, _, twisted_b, _] = fermionic_twist_roles::<FermionU1, D>(
        engine,
        &runtime,
        &fermion_u1(),
        ["fA", "fcanonical", "fB", "fboth"],
        71,
    );
    members(engine, cuda, "member", &twisted_a);
    members(engine, cuda, "member", &twisted_b);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn member_dynamic_tree_is_bit_equal_to_host_and_eager_within_documented_costs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    member_routes::<f64>(engine, cuda);
    member_routes::<Complex64>(engine, cuda);
}

/// Fermionic `V <- V'` against `V' <- V` over fZ2 x U(1): a direct core
/// with exact +1 and -1 job coefficients and one inactive destination block,
/// as Core (identity output) and CopyC (reversed output), unswapped and
/// swapped.
fn signed_cases<D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) -> [Case<FermionU1, D>; 4] {
    let v = fermion_u1();
    let dual = v.try_dual().unwrap();
    let map = |engine: &mut dyn symtn::typed::Engine,
               codomain: &GradedSpace<FermionU1>,
               domain: &GradedSpace<FermionU1>,
               salt| {
        TensorMap::<_, D>::from_subblock_fn(engine, runtime, [codomain], [domain], fill(salt))
            .unwrap()
    };
    let (a, b) = (map(engine, &v, &dual, 211), map(engine, &dual, &v, 212));
    let case = |name, swapped: bool, output: [usize; 2]| Case {
        name,
        lhs: if swapped { b.clone() } else { a.clone() },
        rhs: if swapped { a.clone() } else { b.clone() },
        lhs_axes: vec![usize::from(!swapped)],
        rhs_axes: vec![usize::from(swapped)],
        output_axes: output.to_vec(),
        dense: false,
    };
    [
        case("signed core", false, [0, 1]),
        case("signed copyC", false, [1, 0]),
        case("signed swapped core", true, [1, 0]),
        case("signed swapped copyC", true, [0, 1]),
    ]
}

/// Member Core and CopyC (#1859): unit and signed direct cores, swapped
/// and unswapped, with and without inactive destination blocks.
fn direct_member_routes<D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
) {
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_non_self_dual();
    for (case, _) in candidate_core_probes::<_, D>(engine, &runtime, &v) {
        if matches!(case.name, "C0" | "C1" | "C2") {
            members(engine, cuda, "direct member", &case);
        }
    }
    for (case, _) in copy_c_probes::<_, D>(engine, &runtime, &v) {
        if matches!(case.name, "C1p" | "C2p") {
            members(engine, cuda, "direct member", &case);
        }
    }
    let [core_inactive, _, _] = u1_inactive_cases::<D>(engine, &runtime);
    members(engine, cuda, "direct member", &core_inactive);
    for case in signed_cases::<D>(engine, &runtime) {
        members(engine, cuda, "direct member", &case);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn member_core_and_copy_c_are_bit_equal_to_host_and_eager_within_documented_costs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    direct_member_routes::<f64>(engine, cuda);
    direct_member_routes::<Complex64>(engine, cuda);
}
