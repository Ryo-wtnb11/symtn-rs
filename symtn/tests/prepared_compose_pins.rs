//! Allocation and retention contract of the Host `ComposePlan`.
//!
//! Over U(1), SU(2) and fZ2xU(1), f64 and c64, one plan and one workspace run
//! the member counts B = 1, 2, 17, 1 with new values on every call. The
//! contract is structural: a warm `execute` and `execute_into` allocate
//! nothing at any member count (the workspace's executor keeps the block
//! GEMM kernel's staging; the plans come from the process-global
//! geometry-plan cache, `prepared_member1_core_reuse`); `execute_into`
//! retains nothing new; retained bytes grow with `B > 1`, and B = 1 again
//! retains at least the B = 1 footprint (the GEMM staging keeps its
//! high-water capacity); and at B = 1 the workspace retains at least its
//! output. The values themselves are `prepared_compose`'s contract
//! (`prepared_compose_equals_eager_and_the_tree_oracle_per_member`), and
//! every rejected call returns its typed error before any work.
//!
//! The target-wide serial guard keeps this test's cache warm-up ordered.

use crate::common;

use crate::prepared;

use crate::counting_alloc;

use std::fmt::Debug;

use num_complex::Complex64;
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{ComposePlan, Error, GradedSpace, Runtime, SignatureField, StackedTensorMap};

use common::Payload;
use prepared::{filled, fz2u1_legs, members, su2_legs, u1_legs};

fn data<R, D>(stack: &StackedTensorMap<R, D>) -> Vec<D>
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
    (0..stack.len())
        .flat_map(|index| {
            stack
                .member(engine, index)
                .unwrap()
                .dense_data()
                .unwrap()
                .to_vec()
        })
        .collect()
}

fn calls<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let (value, allocs) = counting_alloc::measure(f);
    (value, allocs.calls, allocs.bytes)
}

fn record<R, D>(label: &str, (v, w): (GradedSpace<R>, GradedSpace<R>))
where
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
        let value = data(output);
        let output_bytes = output.len()
            * output
                .member(engine, 0)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
            * std::mem::size_of::<D>();
        let retained = workspace.retained_bytes();
        // The first `execute_into` may grow the workspace's fill scratch
        // once; the second is warm.
        for (warm_into, mut dst) in fills.into_iter().enumerate() {
            let (_, into, into_bytes) = calls(|| {
                plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut workspace)
                    .unwrap();
            });
            assert_eq!(data(&dst), value, "{key}: execute_into == execute");
            assert!(
                warm_into == 0 || into == 0,
                "{key}: warm execute_into allocated {into} ({into_bytes} B)"
            );
        }
        eprintln!(
            "{key}: calls first={first} ({first_bytes} B) warm={warm} ({warm_bytes} B), retained {retained}"
        );
        assert_eq!(warm, 0, "{key}: warm execute allocated");
        assert!(first > warm, "{key}: the counter sees the cold call");
        assert_eq!(
            workspace.retained_bytes(),
            retained,
            "{key}: execute_into retains nothing"
        );
        if count == 1 {
            assert!(retained >= output_bytes, "{key}: B = 1 retains the output");
        }
        retained_by_call.push(retained);
    }
    let [one, two, seventeen, again] = retained_by_call[..] else {
        unreachable!()
    };
    assert!(two <= seventeen, "{label}: retained grows with B");
    // The executor's GEMM staging keeps its high-water capacity from
    // B = 17; everything else returns to the B = 1 footprint.
    assert!(
        again >= one,
        "{label}: B = 1 again retains at least the B = 1 footprint"
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

/// Purpose: a warm Host compose allocates nothing at any member count and
/// its workspace retains no more than its high-water footprint.
#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "the warm-zero contract is the native grouped GEMM kernel's; a BLAS provider allocates its own per-call scratch"
)]
fn warm_host_compose_plan_allocates_nothing_and_retains_its_footprint() {
    let _serial = crate::counting_alloc::serial();
    record::<_, f64>("U1", u1_legs());
    record::<_, Complex64>("U1", u1_legs());
    record::<_, f64>("SU2", su2_legs());
    record::<_, Complex64>("SU2", su2_legs());
    record::<_, f64>("fZ2xU1", fz2u1_legs());
    record::<_, Complex64>("fZ2xU1", fz2u1_legs());
}

/// Purpose: every rejected Host compose call returns its typed error.
#[test]
fn rejected_host_compose_calls_return_typed_errors() {
    assert_eq!(error_matrix(), expected_errors());
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
