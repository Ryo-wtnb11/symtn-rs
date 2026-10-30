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
//! them: a warm `execute` and `execute_into` allocate at most once,
//! `execute_into` retains nothing new, retained bytes grow with `B` and
//! return to the B = 1 footprint, and at B = 1 the workspace retains its
//! output plus the charged grouped-GEMM bindings of its core (no stacked
//! replay; the base kept a one-member replay there, 208 bytes and 8
//! allocations on these fixtures). At B = 1 a warm call prepares nothing
//! (`prepared_member1_core_reuse`); at B > 1 the core's grouped plans are
//! still prepared per call, so those warm rows are over the one-allocation
//! budget until member retention lands.
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
    let mut state = 0xcbf2_9ce4_8422_2325;
    for index in 0..stack.len() {
        let member = stack.member(index).unwrap();
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
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
    let warmup = ComposePlan::new(&lhs, &rhs).unwrap();
    warmup
        .execute(&lhs, &rhs, &mut warmup.workspace().unwrap())
        .unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
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
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        let (_, warm, warm_bytes) = calls(|| {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        let output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let value = stack_digest(output);
        let output_bytes = output.len()
            * output.member(0).unwrap().dense_data().unwrap().len()
            * std::mem::size_of::<D>();
        let retained = workspace.retained_bytes();
        // The first `execute_into` may grow the Runtime context's fill
        // scratch once; the second is warm.
        for (warm_into, mut dst) in fills.into_iter().enumerate() {
            let (_, into, into_bytes) = calls(|| {
                plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
            });
            assert_eq!(stack_digest(&dst), value, "{key}: execute_into == execute");
            assert!(
                warm_into == 0 || into <= 1,
                "{key}: warm execute_into allocated {into} ({into_bytes} B)"
            );
        }
        eprintln!(
            "    (\"{key}\", {value:#x}), // calls first={first} ({first_bytes} B) warm={warm} ({warm_bytes} B), retained {retained}"
        );
        assert!(warm <= 1, "{key}: warm execute allocated {warm}");
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
    assert!(
        one <= two && two <= seventeen,
        "{label}: retained grows with B"
    );
    assert_eq!(
        again, one,
        "{label}: B = 1 again retains the B = 1 footprint"
    );
}

/// Every rejected call of `mismatched_stacks_are_typed_errors_before_any_work`
/// and the foreign workspace, as the error each returns.
fn error_matrix() -> Vec<(&'static str, Option<Error>)> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let lhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 3, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 3, 2)).unwrap();
    let short = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 2)).unwrap();
    let foreign =
        StackedTensorMap::pack(&members::<_, f64>(&other_runtime, &[&w], &[&v], 3, 2)).unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let other_plan = ComposePlan::new(&lhs, &rhs).unwrap();
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
            show(ComposePlan::new(&lhs, &foreign).map(|_| ())),
        ),
        (
            "new: member counts differ",
            show(ComposePlan::new(&lhs, &short).map(|_| ())),
        ),
        (
            "execute: swapped operands",
            show(plan.execute(&rhs, &lhs, &mut ws).map(|_| ())),
        ),
        (
            "execute: foreign runtime",
            show(plan.execute(&lhs, &foreign, &mut ws).map(|_| ())),
        ),
        (
            "execute: member counts differ",
            show(plan.execute(&lhs, &short, &mut ws).map(|_| ())),
        ),
        (
            "execute_into: wrong destination",
            show(plan.execute_into(&lhs, &rhs, &mut wrong_dst, &mut ws)),
        ),
        (
            "execute_into: short destination",
            show(plan.execute_into(&lhs, &rhs, &mut short_dst, &mut ws)),
        ),
        (
            "execute_into: foreign workspace",
            show(plan.execute_into(&lhs, &rhs, &mut dst, &mut other_ws)),
        ),
        (
            "execute: foreign workspace",
            show(plan.execute(&lhs, &rhs, &mut other_ws).map(|_| ())),
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
