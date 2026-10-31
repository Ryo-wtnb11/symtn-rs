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
use symtn::typed::{ComposePlan, GradedSpace, Runtime, StackedTensorMap, TensorMap};

use common::{DevicePayload, DeviceRule};
use prepared::{compose_oracle, filled, fz2u1_legs, members, su2_legs, u1_legs};

use crate::fixtures;

use fixtures::delta;

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

fn device_gate<R, D>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    label: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
) where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
    D: DevicePayload,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let runtime = Runtime::builder().build().unwrap();
    let mut fills_per_b = BTreeSet::new();
    for count in [1, 2, 17] {
        let label = format!("{label} {} B={count}", D::NAME);
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, 2);
        let jobs = active_sectors(&a[0], &b[0]);
        let lhs = StackedTensorMap::pack(&a).unwrap().to_cuda(cuda).unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap().to_cuda(cuda).unwrap();
        let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
        let mut ws = plan.workspace().unwrap();

        // `terms` = len(A), an upper bound on the inner dimension of a block.
        let terms = a[0].dense_data().unwrap().len();
        let oracles: Vec<_> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| {
                let eager = x.compose(engine, y).unwrap();
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
        let check = |engine: &mut dyn symtn::typed::Engine,
                     cuda: &mut dyn symtn::typed::Engine,
                     stack: &StackedTensorMap<R, D, symtn::typed::CudaStorage<D>>,
                     what: &str| {
            let host = stack.to_host(cuda).unwrap();
            for (index, oracle) in oracles.iter().enumerate() {
                numerics::assert_nonzero_slices_close(
                    &format!("{label}: {what}, member {index}"),
                    host.member(engine, index).unwrap().dense_data().unwrap(),
                    oracle,
                    terms,
                );
            }
        };

        plan.execute(cuda, &lhs, &rhs, &mut ws).unwrap();
        let retained = ws.retained_bytes();
        let before = cuda_transfer_stats();
        let output = plan.execute(cuda, &lhs, &rhs, &mut ws).unwrap();
        let warm = delta(cuda_transfer_stats(), before);
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
        check(engine, cuda, output, "execute");
        assert_eq!(
            ws.retained_bytes(),
            retained,
            "{label}: warm retained bytes"
        );

        for (fill, name) in [
            (D::entry(f64::NAN, f64::NAN), "NaN"),
            (D::entry(-3.0e17, 7.5), "garbage"),
        ] {
            let poisoned = |cuda: &mut dyn symtn::typed::Engine| {
                StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                    .unwrap()
                    .to_cuda(cuda)
                    .unwrap()
            };
            let mut dst = poisoned(cuda);
            plan.execute_into(cuda, &lhs, &rhs, &mut dst, &mut ws)
                .unwrap();
            check(
                engine,
                cuda,
                &dst,
                &format!("cold execute_into over {name}"),
            );

            let mut dst = poisoned(cuda);
            let before = cuda_transfer_stats();
            plan.execute_into(cuda, &lhs, &rhs, &mut dst, &mut ws)
                .unwrap();
            let warm = delta(cuda_transfer_stats(), before);
            assert_eq!(
                (warm.h2d_calls, warm.d2h_calls, warm.device_allocs),
                (0, 0, 0),
                "{label}: warm execute_into transfers nothing: {warm:?}"
            );
            assert!(
                warm.gemm_calls > jobs as u64,
                "{label}: inactive blocks are filled"
            );
            fills_per_b.insert(warm.gemm_calls - jobs as u64);
            check(
                engine,
                cuda,
                &dst,
                &format!("warm execute_into over {name}"),
            );
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
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    device_gate::<_, f64>(engine, cuda, "U1", u1_legs());
    device_gate::<_, Complex64>(engine, cuda, "U1", u1_legs());
    device_gate::<_, f64>(engine, cuda, "SU2", su2_legs());
    device_gate::<_, Complex64>(engine, cuda, "SU2", su2_legs());
    device_gate::<_, f64>(engine, cuda, "fZ2xU1", fz2u1_legs());
    device_gate::<_, Complex64>(engine, cuda, "fZ2xU1", fz2u1_legs());
}

/// `(h2d_calls, h2d_bytes, d2h_calls, device_allocs, gemm_calls, copy_calls,
/// 0 in place of retained bytes (checked as relations))` of one call.
type Counters = [u64; 7];

fn counted<T>(body: impl FnOnce() -> T) -> (T, Counters) {
    let before = cuda_transfer_stats();
    let value = body();
    let after = delta(cuda_transfer_stats(), before);
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
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    label: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
    rows: &mut Vec<(String, Counters)>,
) where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
    D: DevicePayload,
{
    let runtime = Runtime::builder().build().unwrap();
    let stacks = |cuda: &mut dyn symtn::typed::Engine, count, salt| {
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, salt);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, salt + 1);
        let lhs = StackedTensorMap::pack(&a).unwrap().to_cuda(cuda).unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap().to_cuda(cuda).unwrap();
        (a, b, lhs, rhs)
    };
    let (_, _, lhs, rhs) = stacks(cuda, 1, 1);
    let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut into = plan.workspace().unwrap();
    let mut metadata = std::collections::BTreeSet::new();
    for (call, count) in [1usize, 2, 17, 1].into_iter().enumerate() {
        let key = format!("{label} {} call={call} B={count}", D::NAME);
        let (a, b, lhs, rhs) = stacks(cuda, count, 10 * call + 3);
        let (_, first) = counted(|| {
            plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
        });
        let (_, mut warm) = counted(|| {
            plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
        });
        warm[6] = workspace.retained_bytes() as u64;
        let returned = workspace.take_output().unwrap().to_host(cuda).unwrap();
        let mut dst = StackedTensorMap::pack(&filled::<R, D>(
            &runtime,
            &[&v, &v],
            &[&v],
            count,
            D::entry(f64::NAN, f64::NAN),
        ))
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
        let (_, mut written) = counted(|| {
            plan.execute_into(cuda, &lhs, &rhs, &mut dst, &mut into)
                .unwrap();
        });
        written[6] = into.retained_bytes() as u64;
        let written_host = dst.to_host(cuda).unwrap();
        let output_bytes = (count
            * returned
                .member(engine, 0)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
            * std::mem::size_of::<D>()) as u64;
        // The execute workspace holds its device output plus the same Host
        // metadata the execute_into workspace holds, independent of B.
        metadata.insert((warm[6] - output_bytes, written[6]));
        warm[6] = 0;
        written[6] = 0;
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let eager = x.compose(engine, y).unwrap();
            for (form, stack) in [("execute", &returned), ("execute_into", &written_host)] {
                assert_eq!(
                    bits(stack.member(engine, index).unwrap().dense_data().unwrap()),
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
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let mut rows = Vec::new();
    record_counters::<_, f64>(engine, cuda, "U1", u1_legs(), &mut rows);
    record_counters::<_, Complex64>(engine, cuda, "U1", u1_legs(), &mut rows);
    record_counters::<_, f64>(engine, cuda, "SU2", su2_legs(), &mut rows);
    record_counters::<_, Complex64>(engine, cuda, "SU2", su2_legs(), &mut rows);
    record_counters::<_, f64>(engine, cuda, "fZ2xU1", fz2u1_legs(), &mut rows);
    record_counters::<_, Complex64>(engine, cuda, "fZ2xU1", fz2u1_legs(), &mut rows);
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
/// unchanged by #1775, less the dropped plan-cache columns. Every column is
/// structural (transfers, allocations, submissions); retained bytes
/// (column 6) are checked as relations in [`record_counters`] instead, so a
/// struct layout change cannot fail this table.
#[rustfmt::skip]
const PINS: &[(&str, Counters)] = &[
    ("U1 f64 call=0 B=1 execute first", [1, 312, 0, 1, 2, 0, 0]),
    ("U1 f64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 f64 call=0 B=1 execute_into", [2, 72, 0, 2, 3, 0, 0]),
    ("U1 f64 call=1 B=2 execute first", [1, 624, 0, 1, 2, 0, 0]),
    ("U1 f64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 f64 call=1 B=2 execute_into", [1, 128, 0, 1, 3, 0, 0]),
    ("U1 f64 call=2 B=17 execute first", [1, 5304, 0, 1, 2, 0, 0]),
    ("U1 f64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 f64 call=2 B=17 execute_into", [1, 1088, 0, 1, 3, 0, 0]),
    ("U1 f64 call=3 B=1 execute first", [1, 312, 0, 1, 2, 0, 0]),
    ("U1 f64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 f64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0]),
    ("U1 c64 call=0 B=1 execute first", [1, 624, 0, 1, 2, 0, 0]),
    ("U1 c64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 c64 call=0 B=1 execute_into", [2, 144, 0, 2, 3, 0, 0]),
    ("U1 c64 call=1 B=2 execute first", [1, 1248, 0, 1, 2, 0, 0]),
    ("U1 c64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 c64 call=1 B=2 execute_into", [1, 256, 0, 1, 3, 0, 0]),
    ("U1 c64 call=2 B=17 execute first", [1, 10608, 0, 1, 2, 0, 0]),
    ("U1 c64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 c64 call=2 B=17 execute_into", [1, 2176, 0, 1, 3, 0, 0]),
    ("U1 c64 call=3 B=1 execute first", [1, 624, 0, 1, 2, 0, 0]),
    ("U1 c64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("U1 c64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0]),
    ("SU2 f64 call=0 B=1 execute first", [1, 408, 0, 1, 2, 0, 0]),
    ("SU2 f64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 f64 call=0 B=1 execute_into", [2, 200, 0, 2, 3, 0, 0]),
    ("SU2 f64 call=1 B=2 execute first", [1, 816, 0, 1, 2, 0, 0]),
    ("SU2 f64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 f64 call=1 B=2 execute_into", [1, 384, 0, 1, 3, 0, 0]),
    ("SU2 f64 call=2 B=17 execute first", [1, 6936, 0, 1, 2, 0, 0]),
    ("SU2 f64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 f64 call=2 B=17 execute_into", [1, 3264, 0, 1, 3, 0, 0]),
    ("SU2 f64 call=3 B=1 execute first", [1, 408, 0, 1, 2, 0, 0]),
    ("SU2 f64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 f64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0]),
    ("SU2 c64 call=0 B=1 execute first", [1, 816, 0, 1, 2, 0, 0]),
    ("SU2 c64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 c64 call=0 B=1 execute_into", [2, 400, 0, 2, 3, 0, 0]),
    ("SU2 c64 call=1 B=2 execute first", [1, 1632, 0, 1, 2, 0, 0]),
    ("SU2 c64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 c64 call=1 B=2 execute_into", [1, 768, 0, 1, 3, 0, 0]),
    ("SU2 c64 call=2 B=17 execute first", [1, 13872, 0, 1, 2, 0, 0]),
    ("SU2 c64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 c64 call=2 B=17 execute_into", [1, 6528, 0, 1, 3, 0, 0]),
    ("SU2 c64 call=3 B=1 execute first", [1, 816, 0, 1, 2, 0, 0]),
    ("SU2 c64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("SU2 c64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0]),
    ("fZ2xU1 f64 call=0 B=1 execute first", [1, 288, 0, 1, 2, 0, 0]),
    ("fZ2xU1 f64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 f64 call=0 B=1 execute_into", [2, 136, 0, 2, 3, 0, 0]),
    ("fZ2xU1 f64 call=1 B=2 execute first", [1, 576, 0, 1, 2, 0, 0]),
    ("fZ2xU1 f64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 f64 call=1 B=2 execute_into", [1, 256, 0, 1, 3, 0, 0]),
    ("fZ2xU1 f64 call=2 B=17 execute first", [1, 4896, 0, 1, 2, 0, 0]),
    ("fZ2xU1 f64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 f64 call=2 B=17 execute_into", [1, 2176, 0, 1, 3, 0, 0]),
    ("fZ2xU1 f64 call=3 B=1 execute first", [1, 288, 0, 1, 2, 0, 0]),
    ("fZ2xU1 f64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 f64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0]),
    ("fZ2xU1 c64 call=0 B=1 execute first", [1, 576, 0, 1, 2, 0, 0]),
    ("fZ2xU1 c64 call=0 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 c64 call=0 B=1 execute_into", [2, 272, 0, 2, 3, 0, 0]),
    ("fZ2xU1 c64 call=1 B=2 execute first", [1, 1152, 0, 1, 2, 0, 0]),
    ("fZ2xU1 c64 call=1 B=2 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 c64 call=1 B=2 execute_into", [1, 512, 0, 1, 3, 0, 0]),
    ("fZ2xU1 c64 call=2 B=17 execute first", [1, 9792, 0, 1, 2, 0, 0]),
    ("fZ2xU1 c64 call=2 B=17 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 c64 call=2 B=17 execute_into", [1, 4352, 0, 1, 3, 0, 0]),
    ("fZ2xU1 c64 call=3 B=1 execute first", [1, 576, 0, 1, 2, 0, 0]),
    ("fZ2xU1 c64 call=3 B=1 execute warm", [0, 0, 0, 0, 2, 0, 0]),
    ("fZ2xU1 c64 call=3 B=1 execute_into", [0, 0, 0, 0, 3, 0, 0]),
];
