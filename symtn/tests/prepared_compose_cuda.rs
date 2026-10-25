//! Device gates of `ComposePlan` and its `ContractWorkspace` (#1639).
//!
//! CUDA transfer counters are thread-local, so these tests run concurrently.
//! Run with `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test integration prepared_compose_cuda:: -- --ignored`.

#![cfg(feature = "cuda")]

use crate::common;
use crate::numerics;
use crate::prepared;

use std::collections::{BTreeSet, HashSet};
use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use symtn::expert::cuda_transfer_stats;
use symtn::sector::U1FusionRule;
use symtn::typed::{ComposePlan, GradedSpace, Runtime, StackedTensorMap, TensorMap};

use common::{DevicePayload, DeviceRule};
use prepared::{
    compose_oracle, expected_plan_entries, filled, fz2u1_legs, members, su2_legs, u1_legs,
};

use crate::fixtures;

use fixtures::{delta, plans};

#[test]
#[ignore = "requires a real CUDA device"]
fn workspace_releases_its_claim_after_plan_is_dropped() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let (v, w) = u1_legs();
    let lhs = device(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 2, 41));
    let rhs = device(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 42));
    let baseline = plans(&runtime).reserved_entries;
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let first = plan.workspace().unwrap();
    let second = plan.workspace().unwrap();
    let claimed = plans(&runtime).reserved_entries - baseline;
    assert!(claimed > 0);
    assert_eq!(claimed % 2, 0, "each workspace owns the same claim");
    drop(plan);
    assert_eq!(plans(&runtime).reserved_entries, baseline + claimed);
    drop(first);
    assert_eq!(plans(&runtime).reserved_entries, baseline + claimed / 2);
    drop(second);
    assert_eq!(plans(&runtime).reserved_entries, baseline);
}

/// Coupled sectors both operands carry: the plan's direct GEMM jobs, read
/// from the operands' public block trees rather than from the plan.
fn active_sectors<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>) -> usize
where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
    D: DevicePayload,
{
    let coupled = |t: &TensorMap<R, D>| {
        (0..t.subblock_count())
            .map(|i| format!("{:?}", t.subblock_fusion_trees(i).unwrap().coupled()))
            .collect::<HashSet<_>>()
    };
    coupled(a).intersection(&coupled(b)).count()
}

fn device_gate<R, D>(label: &str, (v, w): (GradedSpace<R>, GradedSpace<R>))
where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
    D: DevicePayload,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let mut fills_per_b = BTreeSet::new();
    for count in [1, 2, 17] {
        let label = format!("{label} {} B={count}", D::NAME);
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, 2);
        let jobs = active_sectors(&a[0], &b[0]);
        let lhs = StackedTensorMap::pack(&a).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap().to_cuda().unwrap();
        let plan = ComposePlan::new(&lhs, &rhs).unwrap();
        let mut ws = plan.workspace().unwrap();

        // `terms` = len(A), an upper bound on the inner dimension of a block.
        let terms = a[0].dense_data().unwrap().len();
        let oracles: Vec<_> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| {
                let eager = x.compose(y).unwrap();
                let (oracle, unreached) = compose_oracle(x, y, &eager);
                assert!(unreached > 0, "{label}: fixture must have inactive blocks");
                numerics::assert_nonzero_slices_close(
                    &format!("{label}: Host eager"),
                    eager.dense_data().unwrap(),
                    &oracle,
                    terms,
                );
                oracle
            })
            .collect();
        let check = |stack: &StackedTensorMap<R, D, symtn::typed::CudaStorage<D>>, what: &str| {
            let host = stack.to_host().unwrap();
            for (index, oracle) in oracles.iter().enumerate() {
                numerics::assert_nonzero_slices_close(
                    &format!("{label}: {what}, member {index}"),
                    host.member(index).unwrap().dense_data().unwrap(),
                    oracle,
                    terms,
                );
            }
        };

        plan.execute(&lhs, &rhs, &mut ws).unwrap();
        let retained = ws.retained_bytes();
        let (before, plans_before) = (cuda_transfer_stats(), plans(&runtime));
        let output = plan.execute(&lhs, &rhs, &mut ws).unwrap();
        let warm = delta(cuda_transfer_stats(), before);
        let plans_after = plans(&runtime);
        assert_eq!(
            warm.gemm_calls, jobs as u64,
            "{label}: one GEMM per direct job"
        );
        assert_eq!(
            (warm.h2d_calls, warm.d2h_calls),
            (0, 0),
            "{label}: {warm:?}"
        );
        assert_eq!(warm.device_allocs, 0, "{label}: {warm:?}");
        assert_eq!(
            plans_after.misses, plans_before.misses,
            "{label}: warm misses"
        );
        assert_eq!(
            plans_after.evictions, plans_before.evictions,
            "{label}: warm evictions"
        );
        check(output, "execute");
        assert_eq!(
            ws.retained_bytes(),
            retained,
            "{label}: warm retained bytes"
        );

        for (fill, name) in [
            (D::entry(f64::NAN, f64::NAN), "NaN"),
            (D::entry(-3.0e17, 7.5), "garbage"),
        ] {
            let poisoned = || {
                StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                    .unwrap()
                    .to_cuda()
                    .unwrap()
            };
            let mut dst = poisoned();
            plan.execute_into(&lhs, &rhs, &mut dst, &mut ws).unwrap();
            check(&dst, &format!("cold execute_into over {name}"));

            let mut dst = poisoned();
            let (before, plans_before) = (cuda_transfer_stats(), plans(&runtime));
            plan.execute_into(&lhs, &rhs, &mut dst, &mut ws).unwrap();
            let warm = delta(cuda_transfer_stats(), before);
            let plans_after = plans(&runtime);
            assert_eq!(
                (warm.h2d_calls, warm.d2h_calls, warm.device_allocs),
                (0, 0, 0),
                "{label}: warm execute_into transfers nothing: {warm:?}"
            );
            assert_eq!(
                plans_after.misses, plans_before.misses,
                "{label}: into misses"
            );
            assert_eq!(
                plans_after.evictions, plans_before.evictions,
                "{label}: into evictions"
            );
            assert!(
                warm.gemm_calls > jobs as u64,
                "{label}: inactive blocks are filled"
            );
            fills_per_b.insert(warm.gemm_calls - jobs as u64);
            check(&dst, &format!("warm execute_into over {name}"));
        }
    }
    assert_eq!(
        fills_per_b.len(),
        1,
        "{label}: zero fills per call must not depend on B: {fills_per_b:?}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_handle_equals_the_oracle_with_b_independent_submissions() {
    device_gate::<_, f64>("U1", u1_legs());
    device_gate::<_, Complex64>("U1", u1_legs());
    device_gate::<_, f64>("SU2", su2_legs());
    device_gate::<_, Complex64>("SU2", su2_legs());
    device_gate::<_, f64>("fZ2xU1", fz2u1_legs());
    device_gate::<_, Complex64>("fZ2xU1", fz2u1_legs());
}

fn device<R: DeviceRule>(
    members: &[TensorMap<R, f64>],
) -> StackedTensorMap<R, f64, symtn::typed::CudaStorage<f64>> {
    StackedTensorMap::pack(members).unwrap().to_cuda().unwrap()
}

/// `V ⊗ W ← V ⊗ W` with 81 blocks of distinct extents (the #1508 fixture):
/// its device permute alone needs more than Tenferro's default 64 plans.
fn wide_permute_source(runtime: &Runtime) -> TensorMap<U1FusionRule, f64> {
    let u1 = |charges: Vec<(i32, usize)>| {
        GradedSpace::try_new(
            std::sync::Arc::new(U1FusionRule),
            charges
                .into_iter()
                .map(|(charge, degeneracy)| (symtn::sector::U1Irrep::new(charge), degeneracy)),
        )
        .unwrap()
    };
    let v = u1((0..9).map(|a| (a, a as usize + 1)).collect());
    let w = u1((0..9).map(|j| (100 * j, j as usize + 1)).collect());
    let source = members::<_, f64>(runtime, &[&v, &w], &[&v, &w], 1, 7).remove(0);
    assert_eq!(source.subblock_count(), 81);
    source
}

#[test]
#[ignore = "requires a real CUDA device"]
fn handles_and_an_eager_permute_past_the_default_bound_evict_no_plan() {
    // The pinned order of design §6: both handles reserve and run first, then
    // an eager permute whose executor total alone exceeds 64, then a warm
    // interleave of all three. An absolute raise by any consumer would let
    // one absorb another's reservation and evict here.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let count = 16;
    let (u1_v, u1_w) = u1_legs();
    let (su2_v, su2_w) = su2_legs();
    let u1_a = members::<_, f64>(&runtime, &[&u1_v, &u1_v], &[&u1_w], count, 1);
    let u1_b = members::<_, f64>(&runtime, &[&u1_w], &[&u1_v], count, 2);
    let su2_a = members::<_, f64>(&runtime, &[&su2_v, &su2_v], &[&su2_w], count, 3);
    let su2_b = members::<_, f64>(&runtime, &[&su2_w], &[&su2_v], count, 4);
    let (u1_lhs, u1_rhs) = (device(&u1_a), device(&u1_b));
    let (su2_lhs, su2_rhs) = (device(&su2_a), device(&su2_b));
    let u1_expected =
        expected_plan_entries(&u1_a[0], &u1_b[0], &u1_a[0].compose(&u1_b[0]).unwrap());
    let su2_expected =
        expected_plan_entries(&su2_a[0], &su2_b[0], &su2_a[0].compose(&su2_b[0]).unwrap());
    let source = wide_permute_source(&runtime).to_cuda().unwrap();

    let reserved = |runtime: &Runtime| plans(runtime).reserved_entries;
    let r0 = reserved(&runtime);
    let plan1 = ComposePlan::new(&u1_lhs, &u1_rhs).unwrap();
    let mut ws1 = plan1.workspace().unwrap();
    let r1 = reserved(&runtime);
    let plan2 = ComposePlan::new(&su2_lhs, &su2_rhs).unwrap();
    let mut ws2 = plan2.workspace().unwrap();
    let r2 = reserved(&runtime);
    let (h1_entries, h2_entries) = (r1 - r0, r2 - r1);
    assert_eq!(h1_entries, u1_expected, "U(1) handle reservation");
    assert_eq!(h2_entries, su2_expected, "SU(2) handle reservation");
    plan1.execute(&u1_lhs, &u1_rhs, &mut ws1).unwrap();
    plan2.execute(&su2_lhs, &su2_rhs, &mut ws2).unwrap();

    let _ = source.permute(&[1, 0], &[3, 2]).unwrap();
    let executor = runtime
        .cuda_tree_transform_stats()
        .unwrap()
        .required_plan_entries;
    assert!(
        executor > 64,
        "the permute alone must exceed the default bound"
    );
    let before = plans(&runtime);
    assert_eq!(
        before.reserved_entries,
        r0 + h1_entries + h2_entries + executor
    );

    for _ in 0..3 {
        plan1.execute(&u1_lhs, &u1_rhs, &mut ws1).unwrap();
        plan2.execute(&su2_lhs, &su2_rhs, &mut ws2).unwrap();
        let _ = source.permute(&[1, 0], &[3, 2]).unwrap();
    }
    let after = plans(&runtime);
    assert_eq!(after.evictions, before.evictions, "{before:?} -> {after:?}");
    assert_eq!(after.misses, before.misses, "{before:?} -> {after:?}");
    assert_eq!(after.reserved_entries, before.reserved_entries);

    // `execute_into` across a change of B on one handle: the zero regions
    // and the template are rebuilt for the new B, and the result is still
    // the oracle's.
    for wide in [count, 2 * count + 1] {
        let a = members::<_, f64>(&runtime, &[&u1_v, &u1_v], &[&u1_w], wide, 5);
        let b = members::<_, f64>(&runtime, &[&u1_w], &[&u1_v], wide, 6);
        let mut dst = device(&filled::<_, f64>(
            &runtime,
            &[&u1_v, &u1_v],
            &[&u1_v],
            wide,
            f64::NAN,
        ));
        plan1
            .execute_into(&device(&a), &device(&b), &mut dst, &mut ws1)
            .unwrap();
        let host = dst.to_host().unwrap();
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let eager = x.compose(y).unwrap();
            let (oracle, _) = compose_oracle(x, y, &eager);
            numerics::assert_nonzero_slices_close(
                &format!("execute_into at B={wide}, member {index}"),
                host.member(index).unwrap().dense_data().unwrap(),
                &oracle,
                x.dense_data().unwrap().len(),
            );
        }
    }
    assert_eq!(
        reserved(&runtime),
        before.reserved_entries,
        "a new B reserves nothing"
    );

    drop(ws1);
    assert_eq!(reserved(&runtime), before.reserved_entries - h1_entries);
    drop(ws2);
    assert_eq!(reserved(&runtime), r0 + executor);
}

/// `(h2d_calls, h2d_bytes, d2h_calls, device_allocs, gemm_calls, copy_calls,
/// 0 in place of retained bytes (checked as relations), plan-cache misses, plan-cache evictions)` of one call.
type Counters = [u64; 9];

fn counted<T>(runtime: &Runtime, body: impl FnOnce() -> T) -> (T, Counters) {
    let (before, plans_before) = (cuda_transfer_stats(), plans(runtime));
    let value = body();
    let (after, plans_after) = (delta(cuda_transfer_stats(), before), plans(runtime));
    (
        value,
        [
            after.h2d_calls,
            after.h2d_bytes,
            after.d2h_calls,
            after.device_allocs,
            after.gemm_calls,
            after.copy_calls,
            0,
            plans_after.misses - plans_before.misses,
            plans_after.evictions - plans_before.evictions,
        ],
    )
}

fn bits<D: DevicePayload>(values: &[D]) -> Vec<[u64; 2]> {
    let fold = |x: f64| {
        if x.is_nan() {
            f64::NAN.to_bits()
        } else if x == 0.0 {
            0
        } else {
            x.to_bits()
        }
    };
    values
        .iter()
        .map(|value| {
            let (re, im) = value.parts();
            [fold(re), fold(im)]
        })
        .collect()
}

/// One plan and two workspaces (one per call form) over B = 1, 2, 17, 1 with
/// new dyadic values per call: device members bit-equal to Host eager
/// `compose` (every product and sum is exact), and every call's counters.
fn record_counters<R, D>(
    label: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
    rows: &mut Vec<(String, Counters)>,
) where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
    D: DevicePayload,
{
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let stacks = |count, salt| {
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, salt);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, salt + 1);
        let lhs = StackedTensorMap::pack(&a).unwrap().to_cuda().unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap().to_cuda().unwrap();
        (a, b, lhs, rhs)
    };
    let (_, _, lhs, rhs) = stacks(1, 1);
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let reserved = plans(&runtime).reserved_entries;
    let mut workspace = plan.workspace().unwrap();
    let mut into = plan.workspace().unwrap();
    let claim = plans(&runtime).reserved_entries - reserved;
    let mut metadata = std::collections::BTreeSet::new();
    rows.push((
        format!("{label} {} workspaces", D::NAME),
        [0, 0, 0, 0, 0, 0, 0, 0, claim as u64],
    ));
    for (call, count) in [1usize, 2, 17, 1].into_iter().enumerate() {
        let key = format!("{label} {} call={call} B={count}", D::NAME);
        let (a, b, lhs, rhs) = stacks(count, 10 * call + 3);
        let (_, first) = counted(&runtime, || {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        let (_, mut warm) = counted(&runtime, || {
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        });
        warm[6] = workspace.retained_bytes() as u64;
        let returned = workspace.take_output().unwrap().to_host().unwrap();
        let mut dst = StackedTensorMap::pack(&filled::<R, D>(
            &runtime,
            &[&v, &v],
            &[&v],
            count,
            D::entry(f64::NAN, f64::NAN),
        ))
        .unwrap()
        .to_cuda()
        .unwrap();
        let (_, mut written) = counted(&runtime, || {
            plan.execute_into(&lhs, &rhs, &mut dst, &mut into).unwrap();
        });
        written[6] = into.retained_bytes() as u64;
        let written_host = dst.to_host().unwrap();
        let output_bytes = (count
            * returned.member(0).unwrap().dense_data().unwrap().len()
            * std::mem::size_of::<D>()) as u64;
        // The execute workspace holds its device output plus the same Host
        // metadata the execute_into workspace holds, independent of B.
        metadata.insert((warm[6] - output_bytes, written[6]));
        warm[6] = 0;
        written[6] = 0;
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let eager = x.compose(y).unwrap();
            for (form, stack) in [("execute", &returned), ("execute_into", &written_host)] {
                assert_eq!(
                    bits(stack.member(index).unwrap().dense_data().unwrap()),
                    bits(eager.dense_data().unwrap()),
                    "{key} {form} member {index}"
                );
            }
        }
        for (form, counters) in [
            ("execute first", first),
            ("execute warm", warm),
            ("execute_into", written),
        ] {
            rows.push((format!("{key} {form}"), counters));
        }
    }
    assert_eq!(
        metadata.len(),
        1,
        "{label}: retained metadata depends on B: {metadata:?}"
    );
    let [(execute, into)] = metadata.into_iter().collect::<Vec<_>>()[..] else {
        unreachable!()
    };
    assert_eq!(
        execute, into,
        "{label}: both workspaces hold the same metadata"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_counters_are_pinned_against_the_base() {
    let mut rows = Vec::new();
    record_counters::<_, f64>("U1", u1_legs(), &mut rows);
    record_counters::<_, Complex64>("U1", u1_legs(), &mut rows);
    record_counters::<_, f64>("SU2", su2_legs(), &mut rows);
    record_counters::<_, Complex64>("SU2", su2_legs(), &mut rows);
    record_counters::<_, f64>("fZ2xU1", fz2u1_legs(), &mut rows);
    record_counters::<_, Complex64>("fZ2xU1", fz2u1_legs(), &mut rows);
    for (key, counters) in &rows {
        eprintln!("    (\"{key}\", {counters:?}),");
    }
    let pinned: Vec<_> = PINS
        .iter()
        .map(|&(key, counters)| (key.to_string(), counters))
        .collect();
    assert_eq!(rows, pinned, "observed rows are printed above");
}

/// Recorded on qg1 (A100) at the base `ec3dea5d` with these tests and
/// unchanged by #1775. Every column is structural (transfers, allocations,
/// submissions, plan-cache traffic, the ledger claim); retained bytes
/// (column 6) are checked as relations in [`record_counters`] instead, so a
/// struct layout change cannot fail this table.
#[rustfmt::skip]
const PINS: &[(&str, Counters)] = &[
    ("U1 f64 workspaces", [0, 0, 0, 0, 0, 0, 0, 0, 6]),
    ("U1 f64 call=0 B=1 execute first", [1, 312, 0, 1, 2, 0, 0, 2, 0]),
    ("U1 f64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 f64 call=0 B=1 execute_into", [2, 72, 0, 2, 3, 0, 0, 1, 0]),
    ("U1 f64 call=1 B=2 execute first", [1, 624, 0, 1, 2, 0, 0, 2, 0]),
    ("U1 f64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 f64 call=1 B=2 execute_into", [1, 128, 0, 1, 3, 0, 0, 1, 0]),
    ("U1 f64 call=2 B=17 execute first", [1, 5304, 0, 1, 2, 0, 0, 2, 0]),
    ("U1 f64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 f64 call=2 B=17 execute_into", [1, 1088, 0, 1, 3, 0, 0, 1, 0]),
    ("U1 f64 call=3 B=1 execute first", [1, 312, 0, 1, 2, 0, 0, 0, 0]),
    ("U1 f64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 f64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0, 0, 0]),
    ("U1 c64 workspaces", [0, 0, 0, 0, 0, 0, 0, 0, 6]),
    ("U1 c64 call=0 B=1 execute first", [1, 624, 0, 1, 2, 0, 0, 2, 0]),
    ("U1 c64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 c64 call=0 B=1 execute_into", [2, 144, 0, 2, 3, 0, 0, 1, 0]),
    ("U1 c64 call=1 B=2 execute first", [1, 1248, 0, 1, 2, 0, 0, 2, 0]),
    ("U1 c64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 c64 call=1 B=2 execute_into", [1, 256, 0, 1, 3, 0, 0, 1, 0]),
    ("U1 c64 call=2 B=17 execute first", [1, 10608, 0, 1, 2, 0, 0, 2, 0]),
    ("U1 c64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 c64 call=2 B=17 execute_into", [1, 2176, 0, 1, 3, 0, 0, 1, 0]),
    ("U1 c64 call=3 B=1 execute first", [1, 624, 0, 1, 2, 0, 0, 0, 0]),
    ("U1 c64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("U1 c64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0, 0, 0]),
    ("SU2 f64 workspaces", [0, 0, 0, 0, 0, 0, 0, 0, 6]),
    ("SU2 f64 call=0 B=1 execute first", [1, 408, 0, 1, 2, 0, 0, 2, 0]),
    ("SU2 f64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 f64 call=0 B=1 execute_into", [2, 200, 0, 2, 3, 0, 0, 1, 0]),
    ("SU2 f64 call=1 B=2 execute first", [1, 816, 0, 1, 2, 0, 0, 2, 0]),
    ("SU2 f64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 f64 call=1 B=2 execute_into", [1, 384, 0, 1, 3, 0, 0, 1, 0]),
    ("SU2 f64 call=2 B=17 execute first", [1, 6936, 0, 1, 2, 0, 0, 2, 0]),
    ("SU2 f64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 f64 call=2 B=17 execute_into", [1, 3264, 0, 1, 3, 0, 0, 1, 0]),
    ("SU2 f64 call=3 B=1 execute first", [1, 408, 0, 1, 2, 0, 0, 0, 0]),
    ("SU2 f64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 f64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0, 0, 0]),
    ("SU2 c64 workspaces", [0, 0, 0, 0, 0, 0, 0, 0, 6]),
    ("SU2 c64 call=0 B=1 execute first", [1, 816, 0, 1, 2, 0, 0, 2, 0]),
    ("SU2 c64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 c64 call=0 B=1 execute_into", [2, 400, 0, 2, 3, 0, 0, 1, 0]),
    ("SU2 c64 call=1 B=2 execute first", [1, 1632, 0, 1, 2, 0, 0, 2, 0]),
    ("SU2 c64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 c64 call=1 B=2 execute_into", [1, 768, 0, 1, 3, 0, 0, 1, 0]),
    ("SU2 c64 call=2 B=17 execute first", [1, 13872, 0, 1, 2, 0, 0, 2, 0]),
    ("SU2 c64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 c64 call=2 B=17 execute_into", [1, 6528, 0, 1, 3, 0, 0, 1, 0]),
    ("SU2 c64 call=3 B=1 execute first", [1, 816, 0, 1, 2, 0, 0, 0, 0]),
    ("SU2 c64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("SU2 c64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0, 0, 0]),
    ("fZ2xU1 f64 workspaces", [0, 0, 0, 0, 0, 0, 0, 0, 6]),
    ("fZ2xU1 f64 call=0 B=1 execute first", [1, 288, 0, 1, 2, 0, 0, 2, 0]),
    ("fZ2xU1 f64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 f64 call=0 B=1 execute_into", [2, 136, 0, 2, 3, 0, 0, 1, 0]),
    ("fZ2xU1 f64 call=1 B=2 execute first", [1, 576, 0, 1, 2, 0, 0, 2, 0]),
    ("fZ2xU1 f64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 f64 call=1 B=2 execute_into", [1, 256, 0, 1, 3, 0, 0, 1, 0]),
    ("fZ2xU1 f64 call=2 B=17 execute first", [1, 4896, 0, 1, 2, 0, 0, 2, 0]),
    ("fZ2xU1 f64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 f64 call=2 B=17 execute_into", [1, 2176, 0, 1, 3, 0, 0, 1, 0]),
    ("fZ2xU1 f64 call=3 B=1 execute first", [1, 288, 0, 1, 2, 0, 0, 0, 0]),
    ("fZ2xU1 f64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 f64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0, 0, 0]),
    ("fZ2xU1 c64 workspaces", [0, 0, 0, 0, 0, 0, 0, 0, 6]),
    ("fZ2xU1 c64 call=0 B=1 execute first", [1, 576, 0, 1, 2, 0, 0, 2, 0]),
    ("fZ2xU1 c64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 c64 call=0 B=1 execute_into", [2, 272, 0, 2, 3, 0, 0, 1, 0]),
    ("fZ2xU1 c64 call=1 B=2 execute first", [1, 1152, 0, 1, 2, 0, 0, 2, 0]),
    ("fZ2xU1 c64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 c64 call=1 B=2 execute_into", [1, 512, 0, 1, 3, 0, 0, 1, 0]),
    ("fZ2xU1 c64 call=2 B=17 execute first", [1, 9792, 0, 1, 2, 0, 0, 2, 0]),
    ("fZ2xU1 c64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 c64 call=2 B=17 execute_into", [1, 4352, 0, 1, 3, 0, 0, 1, 0]),
    ("fZ2xU1 c64 call=3 B=1 execute first", [1, 576, 0, 1, 2, 0, 0, 0, 0]),
    ("fZ2xU1 c64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0, 0, 0]),
    ("fZ2xU1 c64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0, 0, 0]),
];
