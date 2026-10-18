//! Characterization pins of the CUDA contraction stage sequence (#1859):
//! eager `contract` / `contract_into` on every route (Core, CopyC,
//! DynamicTree) and member replays of one `ContractPlan` on every route.
//!
//! The fixtures are exact in binary: dyadic entries, unit U(1) recoupling and
//! ±1 fermionic twists, scaled by dyadic factors. Every product and partial
//! sum is exact, so any summation order gives the same value and the device
//! is compared bit for bit with the Host on the same machine (±0 and NaN
//! folded): eager device against Host `contract` / `contract_into` at the
//! same alpha and beta, and each member against the Host member replay of
//! the same plan and against eager device `contract`.
//!
//! Warm transfer and submission counters
//! `(h2d_calls, h2d_bytes, d2h_calls, device_allocs, gemm_calls, copy_calls,
//! retained_bytes)` are pinned in [`PINS`] at the base revisions named there: the stage
//! sequence may regroup the kernels, never add or remove one, with two
//! exceptions (#1859) that [`Recorder::check`] states exactly: member CopyC
//! does not refill its temporary ([`MEMBER_COPY_C`]), and member retained
//! bytes use the one member
//! workspace's Host region accounting, so they differ from the base by a
//! pinned per-case constant ([`RETAINED_METADATA_DELTA`]) while every device
//! byte, which grows with `B`, is equal. #2123 adds two more: a warm member
//! `execute` whose core writes its born-zero output directly does not
//! refill its inactive regions ([`BORN_ZERO_OUTPUT`]), and the core fill's
//! context templates are reserved by the first call that fills, an
//! `execute_into` ([`LAZY_CORE_TEMPLATE`]). The counters are process-wide,
//! hence `--test-threads=1`:
//!
//! `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test typed_cuda_contract_stages -- --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

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

use common::{DevicePayload, DeviceRule};
use contract_cases::{
    candidate_core_probes, copy_c_probes, fermion_u1, fermionic_twist_roles, fill,
    poisoned_destination, u1_inactive_cases, u1_non_self_dual, u1_reordered, Case, FermionU1,
};
use num_complex::Complex64;
use symtn::expert::cuda_transfer_stats;
use symtn::typed::GradedSpace;
use symtn::typed::{ContractPlan, Runtime, StackedTensorMap, TensorMap};

/// `(h2d_calls, h2d_bytes, d2h_calls, device_allocs, gemm_calls, copy_calls,
/// retained_bytes)`; `retained_bytes` is 0 for eager calls.
type Counters = [u64; 7];

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
            0,
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

/// Collects observed counters and compares them with [`PINS`] at the end.
#[derive(Default)]
struct Recorder(Vec<(String, Counters)>);

impl Recorder {
    fn record(&mut self, key: String, counters: Counters) {
        eprintln!("    (\"{key}\", {counters:?}),");
        self.0.push((key, counters));
    }

    fn check(self, prefix: &str) {
        let pinned: Vec<_> = PINS
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|&(key, counters)| (key.to_string(), counters))
            .collect();
        assert_eq!(
            self.0.len(),
            pinned.len(),
            "{prefix}: observed and pinned counter sets differ in size"
        );
        for ((key, observed), (pinned_key, pinned)) in self.0.iter().zip(&pinned) {
            assert_eq!(key, pinned_key, "{prefix}: pin order");
            let mut expected = *pinned;
            // One case's rows across both B sequences and both workspaces.
            let group = key.split(" seq=").next().unwrap();
            let copy_c = MEMBER_COPY_C
                .iter()
                .find(|(name, _)| group.contains(&format!(" {name} ")))
                .filter(|_| !key.starts_with("eager "));
            if let Some(&(_, inactive)) = copy_c {
                let base: Vec<u64> = PINS
                    .iter()
                    .filter(|(row, _)| row.starts_with(&format!("{group} seq=")))
                    .map(|(_, counters)| counters[4])
                    .collect();
                let fewest = *base.iter().min().unwrap();
                assert_eq!(
                    base.iter().max().unwrap() - fewest,
                    inactive,
                    "{group}: a base refill zeroes each inactive region once"
                );
                expected[4] = fewest;
            }
            if pinned[6] != 0 {
                let delta = RETAINED_METADATA_DELTA
                    .iter()
                    .find(|(seen, _)| *seen == group)
                    .map_or(0, |&(_, delta)| delta);
                expected[6] = expected[6].checked_add_signed(delta).unwrap();
            }
            if key.ends_with(" execute warm") {
                if let Some(&(_, inactive)) =
                    BORN_ZERO_OUTPUT.iter().find(|(name, _)| group == *name)
                {
                    expected[4] -= inactive;
                }
            }
            for (call, [calls, bytes]) in LAZY_CORE_TEMPLATE {
                let sign = match key.strip_prefix(call) {
                    Some(" execute first") => -1,
                    Some(" execute_into") => 1,
                    _ => continue,
                };
                for (index, change) in [(0, calls), (1, bytes), (3, calls)] {
                    expected[index] = expected[index].checked_add_signed(sign * change).unwrap();
                }
            }
            if let Some((_, [calls, bytes])) =
                ZERO_TEMPLATE_GROWTH.iter().find(|(row, _)| row == key)
            {
                for (index, change) in [(0, calls), (1, bytes), (3, calls)] {
                    expected[index] = expected[index].checked_add_signed(*change).unwrap();
                }
            }
            assert_eq!(
                observed, &expected,
                "{key}: counters differ from the base pin"
            );
        }
    }
}

/// Member CopyC cases and the inactive destination regions of their core
/// plan. The base (`run_cuda`) zeroed those regions of the CopyC temporary
/// on every call that reused it; the head never does, because the temporary
/// is the member workspace's core-destination stack, born zero and written
/// only by the core GEMMs (#1746). Every head call therefore submits the
/// base's no-refill count: exactly `R` fewer on each base refill call.
/// Cold rows whose context zero-template growth moved (#1859): the
/// template is a high-water mark of the runtime's device context, shared by
/// the cases that run one after another. The base grew it for the CopyC
/// temporary's per-call zero fills (6 elements per member, at B = 17 one
/// 816-byte upload); the head submits no such fill and reserves nothing for
/// it, so the next case that needs a larger template (fA at B = 2, 256
/// bytes) grows it instead. Only `h2d_calls`, `device_allocs` (one per
/// upload) and `h2d_bytes` change.
const ZERO_TEMPLATE_GROWTH: &[(&str, [i64; 2])] = &[
    (
        "member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=2 B=17 execute first",
        [-1, -816],
    ),
    ("member fA f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [1, 256]),
];

/// Per-case change of a member workspace's retained bytes against the base
/// (#1859), the same on every row of the case at every `B`; device bytes,
/// which grow with `B`, are therefore equal. One accounting covers every
/// member region list: inline vector bytes (56 per `CudaRegion`) plus each
/// region's heap capacity.
/// - DynamicTree with core zero regions: +56 per region, the inline bytes
///   the base left out (fA/fB: three regions).
/// - Core with one inactive region: +32, heap capacity where the base
///   counted `2 * dims.len()` words.
/// - CopyC with one inactive region: -88, the base's zero-region list for the
///   temporary, which the head does not keep.
const RETAINED_METADATA_DELTA: &[(&str, i64)] = &[
    ("direct member U(1) core route, inactive block f64", 32),
    ("direct member signed core f64", 32),
    ("direct member signed swapped core f64", 32),
    ("direct member signed copyC f64", -88),
    ("direct member signed swapped copyC f64", -88),
    (
        "member U(1) transformed lhs, identity output, inactive block f64",
        56,
    ),
    (
        "member U(1) output transform over an inactive core block f64",
        -88,
    ),
    ("member fA f64", 168),
    ("member fB f64", 168),
];

/// Member cases whose core writes the output directly, and the inactive
/// destination regions of their core plan (#2123). The base zeroed those
/// regions of a reused `execute` output on every warm call; the head never
/// does, because that output is born zero and only the core GEMMs write it.
/// Every warm `execute` row therefore submits exactly `R` fewer; CopyC and
/// output-transform rows keep their base counts.
const BORN_ZERO_OUTPUT: &[(&str, u64)] = &[
    (
        "member U(1) transformed lhs, identity output, inactive block f64",
        1,
    ),
    ("member fA f64", 3),
    ("member fB f64", 3),
    ("direct member U(1) core route, inactive block f64", 1),
    ("direct member signed core f64", 1),
    ("direct member signed swapped core f64", 1),
];

/// Calls whose context template growth moved (#2123) from the cold
/// `execute` to the `execute_into` of the same call: the base reserved the
/// core fill's zero template (and, first, the ones template) when `execute`
/// prepared a new B; the head reserves them only for a call that fills, the
/// `execute_into`. The uploads are the same; only the row that pays them
/// changes. Later calls find the template at its high-water size.
const LAZY_CORE_TEMPLATE: &[(&str, [i64; 2])] = &[
    (
        "direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1",
        [2, 56],
    ),
    (
        "direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2",
        [1, 96],
    ),
    (
        "direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17",
        [1, 816],
    ),
    (
        "member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1",
        [2, 16],
    ),
    (
        "member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2",
        [1, 16],
    ),
    (
        "member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17",
        [1, 136],
    ),
    ("member fA f64 seq=[1, 2, 17, 1] call=1 B=2", [1, 256]),
    ("member fA f64 seq=[1, 2, 17, 1] call=2 B=17", [1, 2176]),
];

const MEMBER_COPY_C: &[(&str, u64)] = &[
    ("U(1) output transform over an inactive core block", 1),
    ("C1p", 0),
    ("C2p", 0),
    ("signed copyC", 1),
    ("signed swapped copyC", 1),
];

/// A destination of `case`'s result space with every block, including the
/// blocks no GEMM writes, holding a nonzero dyadic value.
fn filled_destination<R: DeviceRule, D: DevicePayload>(
    runtime: &Runtime,
    case: &Case<R, D>,
) -> TensorMap<R, D> {
    let host = case.host();
    let (codomain, domain) = (host.codomain(), host.domain());
    TensorMap::from_subblock_fn(runtime, codomain.iter(), domain.iter(), fill(97)).unwrap()
}

/// Eager device `contract` and an alpha/beta grid of `contract_into`, each
/// bit-equal to the Host call; counters pinned at `f64` for a subset.
fn eager<R: DeviceRule, D: DevicePayload>(
    runtime: &Runtime,
    case: &Case<R, D>,
    family: &str,
    recorder: &mut Recorder,
) {
    let name = format!("eager {family} {} {}", case.name, D::NAME);
    let pin = D::NAME == "f64";
    let spec = case.spec();
    let host = case.host();
    let (lhs, rhs) = (case.lhs.to_cuda().unwrap(), case.rhs.to_cuda().unwrap());
    let device = lhs.contract(&rhs, &spec).unwrap();
    assert_bit_equal(
        &format!("{name} contract"),
        device.to_host().unwrap().dense_data().unwrap(),
        host.dense_data().unwrap(),
    );
    let (_, warm) = delta(|| lhs.contract(&rhs, &spec).unwrap());
    if pin {
        recorder.record(format!("{name} contract warm"), warm);
    }
    let filled = filled_destination(runtime, case);
    let poisoned = poisoned_destination(case);
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
            let mut expected = start.scale(D::entry(1.0, 0.0)).unwrap();
            case.lhs
                .contract_into(&case.rhs, &spec, &mut expected, alpha, beta)
                .unwrap();
            let mut destination = start.to_cuda().unwrap();
            let (_, counters) = delta(|| {
                lhs.contract_into(&rhs, &spec, &mut destination, alpha, beta)
                    .unwrap()
            });
            assert_bit_equal(
                &what,
                destination.to_host().unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap(),
            );
            if pin
                && matches!(
                    (alpha_label, beta_label),
                    ("1", "0") | ("2.5", "2") | ("1", "1") | ("0", "0 poisoned")
                )
            {
                recorder.record(what, counters);
            }
        }
    }
}

fn eager_routes<D: DevicePayload>(recorder: &mut Recorder) {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_non_self_dual();
    for (case, _) in candidate_core_probes::<_, D>(&runtime, &v) {
        eager(&runtime, &case, "core", recorder);
    }
    for (case, _) in copy_c_probes::<_, D>(&runtime, &v) {
        eager(&runtime, &case, "copyC", recorder);
    }
    for case in u1_inactive_cases::<D>(&runtime) {
        eager(&runtime, &case, "inactive", recorder);
    }
    eager(&runtime, &u1_reordered::<D>(&runtime), "tree", recorder);
    for case in fermionic_twist_roles::<FermionU1, D>(
        &runtime,
        &fermion_u1(),
        ["fA", "fcanonical", "fB", "fboth"],
        71,
    ) {
        eager(&runtime, &case, "fermion", recorder);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn eager_device_routes_are_bit_equal_to_the_host_with_base_counters() {
    let mut recorder = Recorder::default();
    eager_routes::<f64>(&mut recorder);
    eager_routes::<Complex64>(&mut recorder);
    recorder.check("eager ");
}

fn rescaled<R: DeviceRule, D: DevicePayload>(
    case: &Case<R, D>,
    call: usize,
    member: usize,
) -> Case<R, D> {
    let (call, member) = (call as f64, member as f64);
    Case {
        name: case.name,
        lhs: case
            .lhs
            .scale(D::entry(
                -0.75 - call / 8.0 + member / 64.0,
                (call + 1.0) / 16.0 - member / 128.0,
            ))
            .unwrap(),
        rhs: case
            .rhs
            .scale(D::entry(1.5 + call / 4.0 + member / 32.0, 0.0))
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
/// to eager device `contract`.
fn members<R: DeviceRule, D: DevicePayload>(
    family: &str,
    case: &Case<R, D>,
    recorder: &mut Recorder,
) {
    let pin = D::NAME == "f64";
    let spec = case.spec();
    let host_one =
        |pick: fn(&Case<R, D>) -> &TensorMap<R, D>| StackedTensorMap::pack(&[pick(case)]).unwrap();
    let lhs_one = host_one(|m| &m.lhs);
    let rhs_one = host_one(|m| &m.rhs);
    let plan = ContractPlan::new(
        &lhs_one.to_cuda().unwrap(),
        &rhs_one.to_cuda().unwrap(),
        &spec,
    )
    .unwrap();
    let host_plan = ContractPlan::new(&lhs_one, &rhs_one, &spec).unwrap();
    for sequence in [&[1usize, 2, 17, 1][..], &[4, 2, 4]] {
        let mut workspace = plan.workspace().unwrap();
        let mut into = plan.workspace().unwrap();
        let mut host_workspace = host_plan.workspace().unwrap();
        for (call, &count) in sequence.iter().enumerate() {
            let key = format!(
                "{family} {} {} seq={sequence:?} call={call} B={count}",
                case.name,
                D::NAME
            );
            let cases: Vec<_> = (0..count).map(|i| rescaled(case, call, i)).collect();
            let stack = |pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
                StackedTensorMap::pack(&cases.iter().map(pick).collect::<Vec<_>>()).unwrap()
            };
            let (host_lhs, host_rhs) = (stack(|m| &m.lhs), stack(|m| &m.rhs));
            let (lhs, rhs) = (host_lhs.to_cuda().unwrap(), host_rhs.to_cuda().unwrap());
            // The first call writes stale values (a later call's scaling) into
            // the output the warm call reuses; a rejected call in between keeps
            // that buffer as the spare (#2123).
            let stale: Vec<_> = (0..count).map(|i| rescaled(case, call + 8, i)).collect();
            let stale_stack = |pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
                StackedTensorMap::pack(&stale.iter().map(pick).collect::<Vec<_>>())
                    .unwrap()
                    .to_cuda()
                    .unwrap()
            };
            let (stale_lhs, stale_rhs) = (stale_stack(|m| &m.lhs), stale_stack(|m| &m.rhs));
            let longer = StackedTensorMap::pack(&vec![&case.rhs; count + 1])
                .unwrap()
                .to_cuda()
                .unwrap();
            let (_, first) = delta(|| {
                plan.execute(&stale_lhs, &stale_rhs, &mut workspace)
                    .unwrap();
            });
            assert!(plan.execute(&lhs, &longer, &mut workspace).is_err());
            assert!(workspace.take_output().is_none());
            let (_, mut warm) = delta(|| {
                plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            });
            warm[6] = workspace.retained_bytes() as u64;
            let returned = workspace.take_output().unwrap().to_host().unwrap();
            let poison: Vec<_> = cases.iter().map(poisoned_destination).collect();
            let mut dst = StackedTensorMap::pack(&poison.iter().collect::<Vec<_>>())
                .unwrap()
                .to_cuda()
                .unwrap();
            let (_, mut written_counters) = delta(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut into).unwrap();
            });
            written_counters[6] = into.retained_bytes() as u64;
            let written = dst.to_host().unwrap();
            let expected = host_plan
                .execute(&host_lhs, &host_rhs, &mut host_workspace)
                .unwrap();
            for (i, member) in cases.iter().enumerate() {
                let eager = member
                    .lhs
                    .to_cuda()
                    .unwrap()
                    .contract(&member.rhs.to_cuda().unwrap(), &spec)
                    .unwrap()
                    .to_host()
                    .unwrap();
                let host = expected.member(i).unwrap();
                for (form, actual) in [("execute", &returned), ("execute_into", &written)] {
                    let actual = actual.member(i).unwrap();
                    for (reference, value) in [("host member", &host), ("eager", &eager)] {
                        assert_bit_equal(
                            &format!("{key} {form} member {i} vs {reference}"),
                            actual.dense_data().unwrap(),
                            value.dense_data().unwrap(),
                        );
                    }
                }
            }
            if pin {
                recorder.record(format!("{key} execute first"), first);
                recorder.record(format!("{key} execute warm"), warm);
                recorder.record(format!("{key} execute_into"), written_counters);
            }
        }
    }
}

fn member_routes<D: DevicePayload>(recorder: &mut Recorder) {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let [_, identity_output, output_transform] = u1_inactive_cases::<D>(&runtime);
    members("member", &identity_output, recorder);
    // `ContractPlan` resolves this geometry to CopyC (an output transform
    // over a direct core with one inactive block).
    members("member", &output_transform, recorder);
    members("member", &u1_reordered::<D>(&runtime), recorder);
    let [twisted_a, _, twisted_b, _] = fermionic_twist_roles::<FermionU1, D>(
        &runtime,
        &fermion_u1(),
        ["fA", "fcanonical", "fB", "fboth"],
        71,
    );
    members("member", &twisted_a, recorder);
    members("member", &twisted_b, recorder);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn member_dynamic_tree_is_bit_equal_to_host_and_eager_with_base_counters() {
    let mut recorder = Recorder::default();
    member_routes::<f64>(&mut recorder);
    member_routes::<Complex64>(&mut recorder);
    recorder.check("member ");
}

/// Fermionic `V <- V'` against `V' <- V` over fZ2 x U(1): a direct core
/// with exact +1 and -1 job coefficients and one inactive destination block,
/// as Core (identity output) and CopyC (reversed output), unswapped and
/// swapped.
fn signed_cases<D: DevicePayload>(runtime: &Runtime) -> [Case<FermionU1, D>; 4] {
    let v = fermion_u1();
    let dual = v.try_dual().unwrap();
    let map = |codomain: &GradedSpace<FermionU1>, domain: &GradedSpace<FermionU1>, salt| {
        TensorMap::<_, D>::from_subblock_fn(runtime, [codomain], [domain], fill(salt)).unwrap()
    };
    let (a, b) = (map(&v, &dual, 211), map(&dual, &v, 212));
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
fn direct_member_routes<D: DevicePayload>(recorder: &mut Recorder) {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = u1_non_self_dual();
    for (case, _) in candidate_core_probes::<_, D>(&runtime, &v) {
        if matches!(case.name, "C0" | "C1" | "C2") {
            members("direct member", &case, recorder);
        }
    }
    for (case, _) in copy_c_probes::<_, D>(&runtime, &v) {
        if matches!(case.name, "C1p" | "C2p") {
            members("direct member", &case, recorder);
        }
    }
    let [core_inactive, _, _] = u1_inactive_cases::<D>(&runtime);
    members("direct member", &core_inactive, recorder);
    for case in signed_cases::<D>(&runtime) {
        members("direct member", &case, recorder);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn member_core_and_copy_c_are_bit_equal_to_host_and_eager_with_base_counters() {
    let mut recorder = Recorder::default();
    direct_member_routes::<f64>(&mut recorder);
    direct_member_routes::<Complex64>(&mut recorder);
    recorder.check("direct member ");
}

/// Recorded on qg1 (A100-SXM4-40GB): eager and member DynamicTree rows at
/// base `fae73f1c` (`a91af835`/`bc8f5de9`), member Core and CopyC rows at
/// base `160187c7`; see the module documentation.
#[rustfmt::skip]
const PINS: &[(&str, Counters)] = &[
    ("eager core C0 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core C0 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C0 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C0 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C0 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C1 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core C1 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C1 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C1 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C1 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C2 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core C2 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C2 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C2 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core C2 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L4 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core L4 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L4 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L4 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L4 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L3 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core L3 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L3 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L3 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L3 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L5 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core L5 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L5 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L5 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L5 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L7 f64 contract warm", [1, 2544, 0, 1, 5, 0, 0]),
    ("eager core L7 f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L7 f64 into alpha=1 beta=0", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L7 f64 into alpha=1 beta=1", [0, 0, 0, 0, 5, 0, 0]),
    ("eager core L7 f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 5, 0, 0]),
    ("eager copyC L3p f64 contract warm", [1, 2544, 0, 1, 24, 0, 0]),
    ("eager copyC L3p f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L3p f64 into alpha=1 beta=0", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L3p f64 into alpha=1 beta=1", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L3p f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L5p f64 contract warm", [1, 2544, 0, 1, 24, 0, 0]),
    ("eager copyC L5p f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L5p f64 into alpha=1 beta=0", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L5p f64 into alpha=1 beta=1", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L5p f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L7p f64 contract warm", [1, 2544, 0, 1, 24, 0, 0]),
    ("eager copyC L7p f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L7p f64 into alpha=1 beta=0", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L7p f64 into alpha=1 beta=1", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC L7p f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C1p f64 contract warm", [1, 2544, 0, 1, 24, 0, 0]),
    ("eager copyC C1p f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C1p f64 into alpha=1 beta=0", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C1p f64 into alpha=1 beta=1", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C1p f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C2p f64 contract warm", [1, 2544, 0, 1, 24, 0, 0]),
    ("eager copyC C2p f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C2p f64 into alpha=1 beta=0", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C2p f64 into alpha=1 beta=1", [0, 0, 0, 0, 24, 0, 0]),
    ("eager copyC C2p f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 24, 0, 0]),
    ("eager inactive U(1) core route, inactive block f64 contract warm", [1, 120, 0, 1, 2, 0, 0]),
    ("eager inactive U(1) core route, inactive block f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 3, 0, 0]),
    ("eager inactive U(1) core route, inactive block f64 into alpha=1 beta=0", [0, 0, 0, 0, 3, 0, 0]),
    ("eager inactive U(1) core route, inactive block f64 into alpha=1 beta=1", [0, 0, 0, 0, 2, 0, 0]),
    ("eager inactive U(1) core route, inactive block f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 3, 0, 0]),
    ("eager inactive U(1) transformed lhs, identity output, inactive block f64 contract warm", [1, 120, 0, 1, 7, 0, 0]),
    ("eager inactive U(1) transformed lhs, identity output, inactive block f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 8, 0, 0]),
    ("eager inactive U(1) transformed lhs, identity output, inactive block f64 into alpha=1 beta=0", [0, 0, 0, 0, 8, 0, 0]),
    ("eager inactive U(1) transformed lhs, identity output, inactive block f64 into alpha=1 beta=1", [0, 0, 0, 0, 7, 0, 0]),
    ("eager inactive U(1) transformed lhs, identity output, inactive block f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 8, 0, 0]),
    ("eager inactive U(1) output transform over an inactive core block f64 contract warm", [1, 120, 0, 1, 9, 0, 0]),
    ("eager inactive U(1) output transform over an inactive core block f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 9, 0, 0]),
    ("eager inactive U(1) output transform over an inactive core block f64 into alpha=1 beta=0", [0, 0, 0, 0, 9, 0, 0]),
    ("eager inactive U(1) output transform over an inactive core block f64 into alpha=1 beta=1", [0, 0, 0, 0, 9, 0, 0]),
    ("eager inactive U(1) output transform over an inactive core block f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 9, 0, 0]),
    ("eager tree U(1) reordered whole-side f64 contract warm", [1, 120, 0, 1, 15, 0, 0]),
    ("eager tree U(1) reordered whole-side f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 15, 0, 0]),
    ("eager tree U(1) reordered whole-side f64 into alpha=1 beta=0", [0, 0, 0, 0, 15, 0, 0]),
    ("eager tree U(1) reordered whole-side f64 into alpha=1 beta=1", [0, 0, 0, 0, 15, 0, 0]),
    ("eager tree U(1) reordered whole-side f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fA f64 contract warm", [1, 1472, 0, 1, 12, 0, 0]),
    ("eager fermion fA f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fA f64 into alpha=1 beta=0", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fA f64 into alpha=1 beta=1", [0, 0, 0, 0, 12, 0, 0]),
    ("eager fermion fA f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fcanonical f64 contract warm", [1, 320, 0, 1, 13, 0, 0]),
    ("eager fermion fcanonical f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 13, 0, 0]),
    ("eager fermion fcanonical f64 into alpha=1 beta=0", [0, 0, 0, 0, 13, 0, 0]),
    ("eager fermion fcanonical f64 into alpha=1 beta=1", [0, 0, 0, 0, 13, 0, 0]),
    ("eager fermion fcanonical f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 13, 0, 0]),
    ("eager fermion fB f64 contract warm", [1, 1472, 0, 1, 12, 0, 0]),
    ("eager fermion fB f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fB f64 into alpha=1 beta=0", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fB f64 into alpha=1 beta=1", [0, 0, 0, 0, 12, 0, 0]),
    ("eager fermion fB f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 15, 0, 0]),
    ("eager fermion fboth f64 contract warm", [1, 320, 0, 1, 44, 0, 0]),
    ("eager fermion fboth f64 into alpha=0 beta=0 poisoned", [0, 0, 0, 0, 44, 0, 0]),
    ("eager fermion fboth f64 into alpha=1 beta=0", [0, 0, 0, 0, 44, 0, 0]),
    ("eager fermion fboth f64 into alpha=1 beta=1", [0, 0, 0, 0, 44, 0, 0]),
    ("eager fermion fboth f64 into alpha=2.5 beta=2", [0, 0, 0, 0, 44, 0, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [4, 264, 0, 4, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 3, 5, 1696]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [1, 128, 0, 1, 3, 5, 1576]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [3, 512, 0, 3, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 5, 1944]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 256, 0, 1, 3, 5, 1704]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [3, 4352, 0, 3, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 3, 5, 5664]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 2176, 0, 1, 3, 5, 3624]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 120, 0, 1, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 3, 5, 3744]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 3, 5, 3624]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=0 B=4 execute first", [2, 992, 0, 2, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 3, 5, 2440]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=0 B=4 execute_into", [1, 512, 0, 1, 3, 5, 1960]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 240, 0, 1, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 5, 2200]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 3, 5, 1960]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 480, 0, 1, 2, 5, 0]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 3, 5, 2440]),
    ("member U(1) transformed lhs, identity output, inactive block f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 3, 5, 1960]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [2, 240, 0, 2, 2, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 3, 6, 1912]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [1, 120, 0, 1, 2, 6, 1792]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 480, 0, 2, 2, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 6, 2152]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 240, 0, 1, 2, 6, 1912]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [3, 4896, 0, 3, 2, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 3, 6, 5752]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 2040, 0, 1, 2, 6, 3712]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 120, 0, 1, 3, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 3, 6, 3832]),
    ("member U(1) output transform over an inactive core block f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 3, 6, 3712]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=0 B=4 execute first", [2, 960, 0, 2, 2, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 3, 6, 2632]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=0 B=4 execute_into", [1, 480, 0, 1, 2, 6, 2152]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 240, 0, 1, 3, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 6, 2392]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 3, 6, 2152]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 480, 0, 1, 3, 6, 0]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 3, 6, 2632]),
    ("member U(1) output transform over an inactive core block f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 3, 6, 2152]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [3, 464, 0, 3, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 3, 12, 3760]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [2, 344, 0, 2, 3, 12, 3640]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [3, 928, 0, 3, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 12, 4224]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [2, 688, 0, 2, 3, 12, 3984]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [3, 7888, 0, 3, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 3, 12, 11184]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [2, 5848, 0, 2, 3, 12, 9144]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 120, 0, 1, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 3, 12, 9264]),
    ("member U(1) reordered whole-side f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 3, 12, 9144]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=0 B=4 execute first", [3, 1856, 0, 3, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 3, 12, 5152]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=0 B=4 execute_into", [2, 1376, 0, 2, 3, 12, 4672]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 240, 0, 1, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 12, 4912]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 3, 12, 4672]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 480, 0, 1, 3, 12, 0]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 3, 12, 5152]),
    ("member U(1) reordered whole-side f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 3, 12, 4672]),
    ("member fA f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [3, 1808, 0, 3, 5, 7, 0]),
    ("member fA f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 8, 7, 4568]),
    ("member fA f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [2, 336, 0, 2, 8, 7, 3096]),
    ("member fA f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 3584, 0, 2, 5, 7, 0]),
    ("member fA f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 8, 7, 6360]),
    ("member fA f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 640, 0, 1, 8, 7, 3416]),
    ("member fA f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [3, 32640, 0, 3, 5, 7, 0]),
    ("member fA f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 8, 7, 33240]),
    ("member fA f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 5440, 0, 1, 8, 7, 8216]),
    ("member fA f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 1472, 0, 1, 5, 7, 0]),
    ("member fA f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 8, 7, 9688]),
    ("member fA f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 8, 7, 8216]),
    ("member fA f64 seq=[4, 2, 4] call=0 B=4 execute first", [3, 7184, 0, 3, 5, 7, 0]),
    ("member fA f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 8, 7, 9944]),
    ("member fA f64 seq=[4, 2, 4] call=0 B=4 execute_into", [2, 1296, 0, 2, 8, 7, 4056]),
    ("member fA f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 2944, 0, 1, 5, 7, 0]),
    ("member fA f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 8, 7, 7000]),
    ("member fA f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 8, 7, 4056]),
    ("member fA f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 5888, 0, 1, 5, 7, 0]),
    ("member fA f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 8, 7, 9944]),
    ("member fA f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 8, 7, 4056]),
    ("member fB f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [3, 1808, 0, 3, 5, 7, 0]),
    ("member fB f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 8, 7, 4568]),
    ("member fB f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [2, 336, 0, 2, 8, 7, 3096]),
    ("member fB f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 3584, 0, 2, 5, 7, 0]),
    ("member fB f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 8, 7, 6360]),
    ("member fB f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 640, 0, 1, 8, 7, 3416]),
    ("member fB f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [2, 30464, 0, 2, 5, 7, 0]),
    ("member fB f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 8, 7, 33240]),
    ("member fB f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 5440, 0, 1, 8, 7, 8216]),
    ("member fB f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 1472, 0, 1, 5, 7, 0]),
    ("member fB f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 8, 7, 9688]),
    ("member fB f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 8, 7, 8216]),
    ("member fB f64 seq=[4, 2, 4] call=0 B=4 execute first", [3, 7184, 0, 3, 5, 7, 0]),
    ("member fB f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 8, 7, 9944]),
    ("member fB f64 seq=[4, 2, 4] call=0 B=4 execute_into", [2, 1296, 0, 2, 8, 7, 4056]),
    ("member fB f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 2944, 0, 1, 5, 7, 0]),
    ("member fB f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 8, 7, 7000]),
    ("member fB f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 8, 7, 4056]),
    ("member fB f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 5888, 0, 1, 5, 7, 0]),
    ("member fB f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 8, 7, 9944]),
    ("member fB f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 8, 7, 4056]),
    // Member Core and CopyC: recorded at base `160187c7` (`74ffb5d7`).
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [1, 2544, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 5, 0, 2544]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 0, 5088]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [1, 43248, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 5, 0, 43248]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 2544, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 5, 0, 2544]),
    ("direct member C0 f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C0 f64 seq=[4, 2, 4] call=0 B=4 execute first", [1, 10176, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 5, 0, 10176]),
    ("direct member C0 f64 seq=[4, 2, 4] call=0 B=4 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C0 f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 0, 5088]),
    ("direct member C0 f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C0 f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 10176, 0, 1, 5, 0, 0]),
    ("direct member C0 f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 5, 0, 10176]),
    ("direct member C0 f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [1, 2544, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 5, 0, 2544]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 0, 5088]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [1, 43248, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 5, 0, 43248]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 2544, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 5, 0, 2544]),
    ("direct member C1 f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[4, 2, 4] call=0 B=4 execute first", [1, 10176, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 5, 0, 10176]),
    ("direct member C1 f64 seq=[4, 2, 4] call=0 B=4 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 0, 5088]),
    ("direct member C1 f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1 f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 10176, 0, 1, 5, 0, 0]),
    ("direct member C1 f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 5, 0, 10176]),
    ("direct member C1 f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [1, 2544, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 5, 0, 2544]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 0, 5088]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [1, 43248, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 5, 0, 43248]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 2544, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 5, 0, 2544]),
    ("direct member C2 f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[4, 2, 4] call=0 B=4 execute first", [1, 10176, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 5, 0, 10176]),
    ("direct member C2 f64 seq=[4, 2, 4] call=0 B=4 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 0, 5088]),
    ("direct member C2 f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C2 f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 10176, 0, 1, 5, 0, 0]),
    ("direct member C2 f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 5, 0, 10176]),
    ("direct member C2 f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 5, 0, 0]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [2, 5088, 0, 2, 5, 19, 0]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 5, 19, 10872]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [1, 2544, 0, 1, 5, 19, 8328]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 10176, 0, 2, 5, 19, 0]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 19, 15960]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 5088, 0, 1, 5, 19, 10872]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [2, 86496, 0, 2, 5, 19, 0]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 5, 19, 92280]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 43248, 0, 1, 5, 19, 49032]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 2544, 0, 1, 5, 19, 0]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 5, 19, 51576]),
    ("direct member C1p f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 5, 19, 49032]),
    ("direct member C1p f64 seq=[4, 2, 4] call=0 B=4 execute first", [2, 20352, 0, 2, 5, 19, 0]),
    ("direct member C1p f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 5, 19, 26136]),
    ("direct member C1p f64 seq=[4, 2, 4] call=0 B=4 execute_into", [1, 10176, 0, 1, 5, 19, 15960]),
    ("direct member C1p f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 19, 0]),
    ("direct member C1p f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 19, 21048]),
    ("direct member C1p f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 19, 15960]),
    ("direct member C1p f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 10176, 0, 1, 5, 19, 0]),
    ("direct member C1p f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 5, 19, 26136]),
    ("direct member C1p f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 5, 19, 15960]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [2, 5088, 0, 2, 5, 19, 0]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 5, 19, 10872]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [1, 2544, 0, 1, 5, 19, 8328]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 10176, 0, 2, 5, 19, 0]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 19, 15960]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 5088, 0, 1, 5, 19, 10872]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [2, 86496, 0, 2, 5, 19, 0]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 5, 19, 92280]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 43248, 0, 1, 5, 19, 49032]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 2544, 0, 1, 5, 19, 0]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 5, 19, 51576]),
    ("direct member C2p f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 5, 19, 49032]),
    ("direct member C2p f64 seq=[4, 2, 4] call=0 B=4 execute first", [2, 20352, 0, 2, 5, 19, 0]),
    ("direct member C2p f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 5, 19, 26136]),
    ("direct member C2p f64 seq=[4, 2, 4] call=0 B=4 execute_into", [1, 10176, 0, 1, 5, 19, 15960]),
    ("direct member C2p f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 5088, 0, 1, 5, 19, 0]),
    ("direct member C2p f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 5, 19, 21048]),
    ("direct member C2p f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 5, 19, 15960]),
    ("direct member C2p f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 10176, 0, 1, 5, 19, 0]),
    ("direct member C2p f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 5, 19, 26136]),
    ("direct member C2p f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 5, 19, 15960]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [3, 176, 0, 3, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 3, 0, 208]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 336, 0, 2, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 0, 328]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [2, 2856, 0, 2, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 3, 0, 2128]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 120, 0, 1, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 3, 0, 208]),
    ("direct member U(1) core route, inactive block f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=0 B=4 execute first", [1, 480, 0, 1, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 3, 0, 568]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=0 B=4 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 240, 0, 1, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 3, 0, 328]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 480, 0, 1, 2, 0, 0]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 3, 0, 568]),
    ("direct member U(1) core route, inactive block f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 3, 0, 88]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [1, 80, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 4, 0, 168]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [1, 160, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 4, 0, 248]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [1, 1360, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 4, 0, 1448]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 80, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 4, 0, 168]),
    ("direct member signed core f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed core f64 seq=[4, 2, 4] call=0 B=4 execute first", [1, 320, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 4, 0, 408]),
    ("direct member signed core f64 seq=[4, 2, 4] call=0 B=4 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed core f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 160, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 4, 0, 248]),
    ("direct member signed core f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed core f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 320, 0, 1, 3, 0, 0]),
    ("direct member signed core f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 4, 0, 408]),
    ("direct member signed core f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [3, 176, 0, 3, 5, 2, 0]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 6, 2, 1320]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [2, 96, 0, 2, 5, 2, 1240]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 320, 0, 2, 5, 2, 0]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 6, 2, 1480]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 160, 0, 1, 5, 2, 1320]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [2, 2720, 0, 2, 5, 2, 0]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 6, 2, 3880]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 1360, 0, 1, 5, 2, 2520]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 80, 0, 1, 6, 2, 0]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 6, 2, 2600]),
    ("direct member signed copyC f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 6, 2, 2520]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=0 B=4 execute first", [3, 656, 0, 3, 5, 2, 0]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 6, 2, 1800]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=0 B=4 execute_into", [2, 336, 0, 2, 5, 2, 1480]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 160, 0, 1, 6, 2, 0]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 6, 2, 1640]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 6, 2, 1480]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 320, 0, 1, 6, 2, 0]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 6, 2, 1800]),
    ("direct member signed copyC f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 6, 2, 1480]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [1, 80, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 4, 0, 168]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [1, 160, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 4, 0, 248]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [1, 1360, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 4, 0, 1448]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 80, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 4, 0, 168]),
    ("direct member signed swapped core f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=0 B=4 execute first", [1, 320, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 4, 0, 408]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=0 B=4 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 160, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 4, 0, 248]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 320, 0, 1, 3, 0, 0]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 4, 0, 408]),
    ("direct member signed swapped core f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 4, 0, 88]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=0 B=1 execute first", [3, 176, 0, 3, 5, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=0 B=1 execute warm", [0, 0, 0, 0, 6, 2, 1320]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=0 B=1 execute_into", [2, 96, 0, 2, 5, 2, 1240]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=1 B=2 execute first", [2, 320, 0, 2, 5, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=1 B=2 execute warm", [0, 0, 0, 0, 6, 2, 1480]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=1 B=2 execute_into", [1, 160, 0, 1, 5, 2, 1320]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=2 B=17 execute first", [2, 2720, 0, 2, 5, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=2 B=17 execute warm", [0, 0, 0, 0, 6, 2, 3880]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=2 B=17 execute_into", [1, 1360, 0, 1, 5, 2, 2520]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=3 B=1 execute first", [1, 80, 0, 1, 6, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=3 B=1 execute warm", [0, 0, 0, 0, 6, 2, 2600]),
    ("direct member signed swapped copyC f64 seq=[1, 2, 17, 1] call=3 B=1 execute_into", [0, 0, 0, 0, 6, 2, 2520]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=0 B=4 execute first", [3, 656, 0, 3, 5, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=0 B=4 execute warm", [0, 0, 0, 0, 6, 2, 1800]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=0 B=4 execute_into", [2, 336, 0, 2, 5, 2, 1480]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=1 B=2 execute first", [1, 160, 0, 1, 6, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=1 B=2 execute warm", [0, 0, 0, 0, 6, 2, 1640]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=1 B=2 execute_into", [0, 0, 0, 0, 6, 2, 1480]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=2 B=4 execute first", [1, 320, 0, 1, 6, 2, 0]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=2 B=4 execute warm", [0, 0, 0, 0, 6, 2, 1800]),
    ("direct member signed swapped copyC f64 seq=[4, 2, 4] call=2 B=4 execute_into", [0, 0, 0, 0, 6, 2, 1480]),
];
