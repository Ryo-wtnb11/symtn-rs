//! The caller's lane allocation reaches every block search of a call, on
//! every executor path and on a reused prepared workspace; where the build
//! has no lane allocator (a BLAS provider), a non-default allocation is
//! refused with a typed `Unsupported` before any write, per participating
//! phase, and the workspace stays usable.
//!
//! Propagation is observed through the process-global lane-allocation
//! cache: an allocation is keyed by its lane allocation, so a call that
//! carries another allocation than the last one searches (a miss) where a
//! call that dropped it would hit the other allocation's entry.

use std::sync::Arc;

use symtn::cache::StructureCacheKind;
use symtn::expert::DenseError;
use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{
    Configured, ContractPlan, ContractSpec, CpuEngine, Error, ExecContext, GradedSpace,
    LaneAllocation, OperationError, Pool, Runtime, StackedTensorMap, TensorMap,
};

type U1Su2 = symtn::sector::ProductFusionRule<U1FusionRule, SU2FusionRule>;

const EQUAL: LaneAllocation = LaneAllocation::EqualShare;
const DEFAULT: LaneAllocation = LaneAllocation::Moldable;

fn context(threads: usize, allocation: LaneAllocation) -> Configured<CpuEngine> {
    let engine = if threads == 1 {
        CpuEngine::serial()
    } else {
        CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
    };
    Configured::new(engine).with_lane_allocation(allocation)
}

/// Unequal U(1) blocks large enough that a four-thread engine searches the
/// lanes of every grouped batch.
fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-3..=3).map(|charge: i32| {
            (
                U1Irrep::new(charge),
                48 - 8 * charge.unsigned_abs() as usize,
            )
        }),
    )
    .unwrap()
}

/// U(1) x SU(2), whose permute replays Multi recoupling blocks.
fn u1_su2_space() -> GradedSpace<U1Su2> {
    let provider = Arc::new(U1Su2::try_new(U1FusionRule, SU2FusionRule).unwrap());
    let sector =
        |q, twice_spin| product_sector(U1Irrep::new(q), SU2Irrep::from_twice_spin(twice_spin));
    GradedSpace::try_new(
        provider,
        [
            (sector(0, 0), 18),
            (sector(1, 1), 14),
            (sector(-1, 1), 14),
            (sector(0, 2), 12),
        ],
    )
    .unwrap()
}

const SPEC: ContractSpec<'static> = ContractSpec {
    lhs: &[1],
    rhs: &[0],
    codomain: &[0],
    domain: &[1],
};

fn lane_misses() -> u64 {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::LaneAllocations)
        .unwrap()
        .misses()
}

/// Whether `call` searched a lane allocation the cache did not hold.
fn searches(call: impl FnOnce()) -> bool {
    let before = lane_misses();
    call();
    lane_misses() > before
}

fn is_refused<T>(result: Result<T, Error>) -> bool {
    matches!(
        result,
        Err(Error::Operation(error)) if matches!(
            *error,
            OperationError::Dense(DenseError::Unsupported { op: "lane_allocation", .. })
        )
    )
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

/// What: each distinct executor path that searches (fresh and initialized
/// contraction GEMM, the Multi recoupling GEMM of a transform, a
/// factorization batch) runs the call's lane allocation: after the default
/// is cached, the other allocation searches once and then hits its own
/// entry.
#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "the lane search is the native provider's; a BLAS provider refuses another allocation"
)]
fn every_searching_path_runs_the_calls_lane_allocation() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let seed = &mut CpuEngine::serial();
    let v = u1_space();
    let a = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 6).unwrap();
    let w = u1_su2_space();
    let t = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&w, &w], [&w, &w], 7).unwrap();
    type Path<'a> = (&'a str, Box<dyn Fn(&mut dyn ExecContext) + 'a>);
    let paths: [Path<'_>; 4] = [
        (
            "contract",
            Box::new(|cx| drop(a.contract(cx, &b, &SPEC).unwrap())),
        ),
        (
            "contract_into",
            Box::new(|cx| {
                let mut destination = a.zeros_like(cx).unwrap();
                a.contract_into(cx, &b, &SPEC, &mut destination, 1.0, 0.0)
                    .unwrap();
            }),
        ),
        (
            "permute",
            Box::new(|cx| drop(t.permute(cx, &[0, 2], &[1, 3]).unwrap())),
        ),
        (
            "svd_compact",
            Box::new(|cx| drop(a.svd_compact(cx, &[0], &[1]).unwrap())),
        ),
    ];
    for (name, path) in &paths {
        symtn::cache::clear();
        let (mut default, mut equal) = (context(4, DEFAULT), context(4, EQUAL));
        assert!(
            searches(|| path(&mut default)),
            "{name}: premise: it searches"
        );
        assert!(!searches(|| path(&mut default)), "{name}: default hit");
        assert!(searches(|| path(&mut equal)), "{name}: own allocation");
        assert!(!searches(|| path(&mut equal)), "{name}: own allocation hit");
    }
}

/// What: a reused prepared workspace runs each call's lane allocation, never
/// an earlier call's, in both directions: default then the other allocation
/// searches again, and with only the other allocation cached, a default call
/// on the same workspace searches its own.
#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "the lane search is the native provider's; a BLAS provider refuses another allocation"
)]
fn a_prepared_workspace_runs_each_calls_lane_allocation() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let seed = &mut CpuEngine::serial();
    let v = u1_space();
    let mut stack = |salt: u64| {
        let members: Vec<_> = (0..2)
            .map(|member| {
                TensorMap::<_, f64>::rand_with_seed(
                    seed,
                    &runtime,
                    [&v],
                    [&v],
                    salt + member as u64,
                )
                .unwrap()
            })
            .collect();
        StackedTensorMap::pack(&members).unwrap()
    };
    let (lhs, rhs) = (stack(10), stack(20));
    let (mut default, mut equal) = (context(4, DEFAULT), context(4, EQUAL));
    let plan = ContractPlan::new(&mut default, &lhs, &rhs, &SPEC).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut run = |cx: &mut dyn ExecContext| {
        searches(|| {
            plan.execute(cx, &lhs, &rhs, &mut workspace).unwrap();
        })
    };
    symtn::cache::clear();
    assert!(run(&mut default), "premise: it searches");
    assert!(run(&mut equal), "default -> other allocation");
    symtn::cache::clear();
    assert!(run(&mut equal), "other allocation, cold");
    assert!(run(&mut default), "other allocation -> default");
    assert!(!run(&mut default), "default hit");
}

/// What: an allocation request with nothing to search (a one-slot engine)
/// runs on the native provider and gives the default's bits.
#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "a BLAS provider has no lane allocator and refuses another allocation"
)]
fn a_call_with_nothing_to_search_runs_any_lane_allocation() {
    let runtime = Runtime::builder().build().unwrap();
    let seed = &mut CpuEngine::serial();
    let v = u1_space();
    let a = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 6).unwrap();
    let default = a.contract(&mut context(1, DEFAULT), &b, &SPEC).unwrap();
    let equal = a.contract(&mut context(1, EQUAL), &b, &SPEC).unwrap();
    assert_eq!(
        bits(equal.dense_data().unwrap()),
        bits(default.dense_data().unwrap())
    );
}

/// What: on a provider without the lane allocator, another allocation is
/// refused with a typed `Unsupported` before any write: the caller's
/// destination keeps its bits, an eager and a prepared call alike, and the
/// prepared workspace serves a valid call afterwards.
#[test]
#[cfg_attr(
    not(any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    )),
    ignore = "the native provider has the lane allocator; the refusal is a BLAS provider's"
)]
fn a_provider_without_the_allocator_refuses_before_any_write() {
    let runtime = Runtime::builder().build().unwrap();
    let seed = &mut CpuEngine::serial();
    let v = u1_space();
    let a = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 6).unwrap();
    let mut destination =
        TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&v], [&v], 9).unwrap();
    let before = bits(destination.dense_data().unwrap());
    let equal = &mut context(4, EQUAL);
    assert!(is_refused(a.contract_into(
        equal,
        &b,
        &SPEC,
        &mut destination,
        1.0,
        1.0
    )));
    assert_eq!(bits(destination.dense_data().unwrap()), before);

    let (lhs, rhs) = (
        StackedTensorMap::pack(&[&a, &b]).unwrap(),
        StackedTensorMap::pack(&[&b, &a]).unwrap(),
    );
    let default = &mut context(4, DEFAULT);
    let plan = ContractPlan::new(default, &lhs, &rhs, &SPEC).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(default, &lhs, &rhs, &mut workspace).unwrap();
    let mut target = workspace.take_output().unwrap();
    let members = |stack: &StackedTensorMap<U1FusionRule, f64>| -> Vec<u64> {
        let engine = &mut CpuEngine::serial();
        (0..stack.len())
            .flat_map(|i| bits(stack.member(engine, i).unwrap().dense_data().unwrap()))
            .collect()
    };
    let kept = members(&target);
    assert!(is_refused(plan.execute_into(
        equal,
        &lhs,
        &rhs,
        &mut target,
        &mut workspace
    )));
    assert_eq!(members(&target), kept);
    plan.execute_into(default, &lhs, &rhs, &mut target, &mut workspace)
        .unwrap();
}

/// What: participation is per lowered phase: a general eigendecomposition,
/// whose dense stage searches no lanes, runs any allocation on its current
/// split, while a split that needs a Multi-recoupling permute first refuses
/// the allocation the provider cannot honor.
#[test]
#[cfg_attr(
    not(any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    )),
    ignore = "the native provider has the lane allocator; the refusal is a BLAS provider's"
)]
fn a_leg_role_permute_carries_the_lane_allocation() {
    let runtime = Runtime::builder().build().unwrap();
    let seed = &mut CpuEngine::serial();
    let w = u1_su2_space();
    let t = TensorMap::<_, f64>::rand_with_seed(seed, &runtime, [&w, &w], [&w, &w], 3).unwrap();
    let equal = &mut context(4, EQUAL);
    t.eig_full(equal, &[0, 1], &[2, 3]).unwrap();
    assert!(is_refused(t.eig_full(equal, &[0, 2], &[1, 3])));
}
