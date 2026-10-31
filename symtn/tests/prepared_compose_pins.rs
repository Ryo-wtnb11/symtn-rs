//! Characterization pins of the Host `ComposePlan` (#1775), recorded on the
//! base `ec3dea5d` before `ComposePlan` moved onto `ContractPlan`.
//!
//! Over U(1), SU(2) and fZ2xU(1), f64 and c64, one plan and one workspace run
//! the member counts B = 1, 2, 17, 1 with new dyadic values on every call.
//! Every product and partial sum of the fixtures is exact, so the value
//! digest (±0 and NaN folded) does not depend on the dense kernel's
//! summation order. [`DIGESTS`] pins every value digest, recorded on the
//! base and unchanged by #1775; `expected_errors` lists the typed error of
//! every rejected call.
//!
//! Allocation counts and retained bytes are structural contracts, not exact
//! sizes, so a dense-backend release or a struct layout change cannot fail
//! them: a warm `execute` and `execute_into` allocate nothing at B = 1 (the
//! workspace keeps its core's grouped-GEMM plans,
//! `prepared_member1_core_reuse`) and at most the block GEMM kernel's
//! per-call staging at B > 1 (seven allocations measured on these
//! fixtures' serial engine; the count grows with distinct geometries and
//! lanes), until `B > 1` retention lands; `execute_into` retains nothing new; retained bytes grow with
//! `B > 1` and return to the B = 1 footprint; and at B = 1 the workspace
//! retains its output plus the charged grouped-GEMM bindings of its core (no
//! stacked replay; the base kept a one-member replay there, 208 bytes and 8
//! allocations on these fixtures).
//!
//! The target-wide serial guard keeps this test's cache warm-up ordered.

use crate::backend;

use crate::common;

use crate::prepared;

use crate::counting_alloc;

use std::fmt::Debug;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{ComposePlan, Error, GradedSpace, Runtime, SignatureField, StackedTensorMap};

use common::Payload;
use prepared::{filled, fz2u1_legs, members, su2_legs, u1_legs};

fn fold(state: &mut u64, values: impl IntoIterator<Item = f64>) {
    for value in values {
        let bits = if value.is_nan() {
            f64::NAN.to_bits()
        } else if value == 0.0 {
            0
        } else {
            value.to_bits()
        };
        for byte in bits.to_le_bytes() {
            *state ^= u64::from(byte);
            *state = state.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn stack_digest<R, D>(stack: &StackedTensorMap<R, D>) -> u64
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut state = 0xcbf2_9ce4_8422_2325;
    for index in 0..stack.len() {
        let member = stack.member(engine, index).unwrap();
        fold(
            &mut state,
            member.dense_data().unwrap().iter().flat_map(|value| {
                let (re, im) = value.parts();
                [re, im]
            }),
        );
    }
    state
}

fn calls<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let (value, allocs) = counting_alloc::measure(f);
    (value, allocs.calls, allocs.bytes)
}

fn record<R, D>(
    label: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
    rows: &mut Vec<(String, u64)>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let stacks = |count, salt| {
        (
            StackedTensorMap::pack(&members::<R, D>(&runtime, &[&v, &v], &[&w], count, salt))
                .unwrap(),
            StackedTensorMap::pack(&members::<R, D>(&runtime, &[&w], &[&v], count, salt + 1))
                .unwrap(),
        )
    };
    let (lhs, rhs) = stacks(1, 1);
    // A first composition on another plan creates the Runtime's context and
    // fills the process-wide caches, whose cold cost depends on the platform
    // and the enabled features; the rows below count this plan's own work.
    let warmup = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let warmup_workspace = &mut warmup.workspace().unwrap();
    warmup
        .execute(engine, &lhs, &rhs, warmup_workspace)
        .unwrap();
    let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut retained_by_call = Vec::new();
    for (call, count) in [1usize, 2, 17, 1].into_iter().enumerate() {
        let key = format!("{label} {} call={call} B={count}", D::NAME);
        let (lhs, rhs) = stacks(count, 10 * call + 3);
        let fills = [D::entry(f64::NAN, f64::NAN), D::entry(-3.0e17, 7.5)].map(|fill| {
            StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                .unwrap()
        });
        let (_, first, first_bytes) = calls(|| {
            plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
        });
        let (_, warm, warm_bytes) = calls(|| {
            plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
        });
        let output = plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
        let value = stack_digest(output);
        let output_bytes = output.len()
            * output
                .member(engine, 0)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
            * std::mem::size_of::<D>();
        let retained = workspace.retained_bytes();
        // A warm B = 1 call reuses the workspace's grouped core plans; at
        // B > 1 the core runs on the block GEMM kernel, whose per-call
        // staging (job records, one index entry and plan per distinct
        // geometry, lane tables) measured seven allocations on these
        // fixtures' serial engine; it is a fixture bound, not a kernel one.
        let warm_budget = if count == 1 { 0 } else { 7 };
        // The first `execute_into` may grow the workspace's fill scratch
        // once; the second is warm.
        for (warm_into, mut dst) in fills.into_iter().enumerate() {
            let (_, into, into_bytes) = calls(|| {
                plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
            });
            assert_eq!(stack_digest(&dst), value, "{key}: execute_into == execute");
            assert!(
                warm_into == 0 || into <= warm_budget,
                "{key}: warm execute_into allocated {into} ({into_bytes} B)"
            );
        }
        eprintln!(
            "    (\"{key}\", {value:#x}), // calls first={first} ({first_bytes} B) warm={warm} ({warm_bytes} B), retained {retained}"
        );
        assert!(warm <= warm_budget, "{key}: warm execute allocated {warm}");
        assert!(first > warm, "{key}: the counter sees the cold call");
        assert_eq!(
            workspace.retained_bytes(),
            retained,
            "{key}: execute_into retains nothing"
        );
        if count == 1 {
            assert!(
                retained >= output_bytes,
                "{key}: B = 1 retains the output and the core bindings"
            );
        }
        retained_by_call.push(retained);
        rows.push((key, value));
    }
    let [one, two, seventeen, again] = retained_by_call[..] else {
        unreachable!()
    };
    // B = 1 also holds the charged core bindings, which B > 1 releases.
    assert!(two <= seventeen, "{label}: retained grows with B");
    assert_eq!(
        again, one,
        "{label}: B = 1 again retains the B = 1 footprint"
    );
}

/// Every rejected call of `mismatched_stacks_are_typed_errors_before_any_work`
/// and the foreign workspace, as the error each returns.
fn error_matrix() -> Vec<(&'static str, Option<Error>)> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 3, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 3, 2)).unwrap();
    let short = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 2)).unwrap();
    let foreign =
        StackedTensorMap::pack(&members::<_, f64>(&other_runtime, &[&w], &[&v], 3, 2)).unwrap();
    let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let other_plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut ws = plan.workspace().unwrap();
    let mut other_ws = other_plan.workspace().unwrap();
    let mut wrong_dst =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&v], 3, 3)).unwrap();
    let mut short_dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 2, 1.0)).unwrap();
    let mut dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 3, 1.0)).unwrap();
    let show = |result: Result<(), Error>| result.err();
    vec![
        (
            "new: foreign runtime",
            show(ComposePlan::new(engine, &lhs, &foreign).map(|_| ())),
        ),
        (
            "new: member counts differ",
            show(ComposePlan::new(engine, &lhs, &short).map(|_| ())),
        ),
        (
            "execute: swapped operands",
            show(plan.execute(engine, &rhs, &lhs, &mut ws).map(|_| ())),
        ),
        (
            "execute: foreign runtime",
            show(plan.execute(engine, &lhs, &foreign, &mut ws).map(|_| ())),
        ),
        (
            "execute: member counts differ",
            show(plan.execute(engine, &lhs, &short, &mut ws).map(|_| ())),
        ),
        (
            "execute_into: wrong destination",
            show(plan.execute_into(engine, &lhs, &rhs, &mut wrong_dst, &mut ws)),
        ),
        (
            "execute_into: short destination",
            show(plan.execute_into(engine, &lhs, &rhs, &mut short_dst, &mut ws)),
        ),
        (
            "execute_into: foreign workspace",
            show(plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut other_ws)),
        ),
        (
            "execute: foreign workspace",
            show(plan.execute(engine, &lhs, &rhs, &mut other_ws).map(|_| ())),
        ),
    ]
}

#[test]
fn host_compose_plan_values_bytes_allocations_and_errors_are_pinned() {
    let _serial = crate::counting_alloc::serial();
    if !backend::faer_is_default_backend() {
        return;
    }
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let mut rows = Vec::new();
    record::<_, f64>("U1", u1_legs(), &mut rows);
    record::<_, Complex64>("U1", u1_legs(), &mut rows);
    record::<_, f64>("SU2", su2_legs(), &mut rows);
    record::<_, Complex64>("SU2", su2_legs(), &mut rows);
    record::<_, f64>("fZ2xU1", fz2u1_legs(), &mut rows);
    record::<_, Complex64>("fZ2xU1", fz2u1_legs(), &mut rows);
    let errors = error_matrix();
    let pinned: Vec<_> = DIGESTS
        .iter()
        .map(|&(key, digest)| (key.to_string(), digest))
        .collect();
    assert_eq!(rows, pinned, "observed digests are printed above");
    assert_eq!(errors, expected_errors());
}

/// The error of every [`error_matrix`] row. A member-count mismatch is
/// rejected at `new` (#1775, approved stricter errors A23).
fn expected_errors() -> Vec<(&'static str, Option<Error>)> {
    let signature = |field| {
        Some(Error::BatchSignatureMismatch {
            member: None,
            field,
        })
    };
    let invalid = |message: &str| Some(Error::InvalidArgument(message.to_string()));
    vec![
        ("new: foreign runtime", Some(Error::RuntimeMismatch)),
        (
            "new: member counts differ",
            invalid("operand stacks have different member counts"),
        ),
        (
            "execute: swapped operands",
            signature(SignatureField::HomSpace),
        ),
        (
            "execute: foreign runtime",
            signature(SignatureField::Runtime),
        ),
        (
            "execute: member counts differ",
            invalid("operand stacks have different member counts"),
        ),
        (
            "execute_into: wrong destination",
            signature(SignatureField::HomSpace),
        ),
        (
            "execute_into: short destination",
            invalid("destination member count differs from operands"),
        ),
        (
            "execute_into: foreign workspace",
            invalid("workspace belongs to another plan"),
        ),
        (
            "execute: foreign workspace",
            invalid("workspace belongs to another plan"),
        ),
    ]
}

/// Value digests, recorded on the base `ec3dea5d` (macOS aarch64 and qg1
/// x86_64 Linux alike) and unchanged by #1775.
#[rustfmt::skip]
const DIGESTS: &[(&str, u64)] = &[
    ("U1 f64 call=0 B=1", 0xd76442f7a019dd2),
    ("U1 f64 call=1 B=2", 0xcc7f92611b071413),
    ("U1 f64 call=2 B=17", 0x4aea74fa1b49b599),
    ("U1 f64 call=3 B=1", 0x6b6d317cdac66a13),
    ("U1 c64 call=0 B=1", 0x652cf4b4aef062b0),
    ("U1 c64 call=1 B=2", 0x297cf63f68f0d68f),
    ("U1 c64 call=2 B=17", 0x2dc5cbbc33f0b32e),
    ("U1 c64 call=3 B=1", 0xb75dc3454851fb8b),
    ("SU2 f64 call=0 B=1", 0x990d51ba85b0b569),
    ("SU2 f64 call=1 B=2", 0x9e89efa27937d69d),
    ("SU2 f64 call=2 B=17", 0x5bb560c4813e8837),
    ("SU2 f64 call=3 B=1", 0x43ce177b7d3eedd5),
    ("SU2 c64 call=0 B=1", 0x6d1183ab6294aee9),
    ("SU2 c64 call=1 B=2", 0x14750eff5eb8112b),
    ("SU2 c64 call=2 B=17", 0x5a1ded85d8111ec2),
    ("SU2 c64 call=3 B=1", 0x93d59f926f684fe5),
    ("fZ2xU1 f64 call=0 B=1", 0xe4b22a8e500beb3f),
    ("fZ2xU1 f64 call=1 B=2", 0xa0dc11d634d2c0ec),
    ("fZ2xU1 f64 call=2 B=17", 0x4b5950622b829e71),
    ("fZ2xU1 f64 call=3 B=1", 0x8e144ba44bb62cb5),
    ("fZ2xU1 c64 call=0 B=1", 0x6d763c21c6ac588b),
    ("fZ2xU1 c64 call=1 B=2", 0xf59530db98534636),
    ("fZ2xU1 c64 call=2 B=17", 0x758b6a9cb59609e),
    ("fZ2xU1 c64 call=3 B=1", 0xdf19f94b2386f04c),
];
