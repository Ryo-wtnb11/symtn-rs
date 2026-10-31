#![cfg(feature = "cuda")]

use std::sync::Arc;

use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex32, Complex64};
use symtn::typed::{ContractSpec, CudaStorage, Engine, GradedSpace, Runtime, TensorMap};
use symtn_network::{
    clear_plan_cache, configure_plan_cache, plan_cache_stats, ContractionPlan, ContractionStep,
    GreedyDenseOptimizer, Network, NetworkExecutionWorkspace, NetworkIR, NetworkOperand,
    PlanCacheConfig, TemporaryLabel, TensorId,
};

use crate::network_support;
use network_support::{conj, net, op};

use crate::numerics;

fn cuda_engine() -> symtn::typed::CudaEngine {
    symtn::typed::CudaEngine::open(symtn::typed::CudaDeviceId::from_ordinal(0)).unwrap()
}

fn pair_network() -> Network {
    Network::new(
        vec![vec!["a", "k"], vec!["k", "b"]],
        vec![false, false],
        vec![1, 1],
        vec!["a", "b"],
        Some(1),
    )
    .unwrap()
}

fn cuda_pair<R>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    space: &GradedSpace<R>,
    seed: u64,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let lhs = TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], seed).unwrap();
    let rhs =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], seed + 1).unwrap();
    let host_plan = pair_network()
        .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
        .unwrap();
    let lhs_cuda = lhs.to_cuda(cuda).unwrap();
    let rhs_cuda = rhs.to_cuda(cuda).unwrap();
    let cuda_refs: [&TensorMap<R, f64, CudaStorage>; 2] = [&lhs_cuda, &rhs_cuda];
    let cuda_plan = pair_network()
        .plan(&cuda_refs, &GreedyDenseOptimizer)
        .unwrap();
    assert_eq!(host_plan.plan().steps(), cuda_plan.plan().steps());
    let actual = cuda_plan
        .execute(cuda, &cuda_refs, &mut NetworkExecutionWorkspace::default())
        .unwrap();
    let before_cached = plan_cache_stats(runtime);
    let cached_actual = net(&[op(&["a"], &["k"]), op(&["k"], &["b"])], &["a"], &["b"])
        .contract(cuda, &[&lhs_cuda, &rhs_cuda])
        .unwrap();
    let cached_warm = net(&[op(&["a"], &["k"]), op(&["k"], &["b"])], &["a"], &["b"])
        .contract(cuda, &[&lhs_cuda, &rhs_cuda])
        .unwrap();
    let after_cached = plan_cache_stats(runtime);
    let manual = lhs_cuda
        .contract(
            cuda,
            &rhs_cuda,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let host_oracle = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(actual.placement(), lhs_cuda.placement());
    assert!(std::ptr::eq(actual.provider(), lhs_cuda.provider()));
    assert_eq!(actual.codomain(), host_oracle.codomain());
    assert_eq!(actual.domain(), host_oracle.domain());
    // Device GEMM order is the kernel's, so device results agree with the
    // host oracle (and the device manual contract) under the tolerance rule.
    let terms = space.dim().unwrap().ceil() as usize;
    let actual_host = actual.to_host(cuda).unwrap();
    let close = |what, got: &[f64], want: &[f64]| {
        numerics::assert_slices_close(what, got, want, terms);
    };
    close(
        "network vs device contract",
        actual_host.dense_data().unwrap(),
        manual.to_host(cuda).unwrap().dense_data().unwrap(),
    );
    close(
        "network vs host",
        actual_host.dense_data().unwrap(),
        host_oracle.dense_data().unwrap(),
    );
    close(
        "cached vs host",
        cached_actual.to_host(cuda).unwrap().dense_data().unwrap(),
        host_oracle.dense_data().unwrap(),
    );
    close(
        "warm cached vs host",
        cached_warm.to_host(cuda).unwrap().dense_data().unwrap(),
        host_oracle.dense_data().unwrap(),
    );
    assert_eq!(cached_actual.placement(), lhs_cuda.placement());
    // The warm device replay reproduces the returning result exactly: identical
    // submissions to the same device kernels, so f64 is bitwise equal.
    assert_eq!(
        cached_warm.to_host(cuda).unwrap().dense_data().unwrap(),
        actual.to_host(cuda).unwrap().dense_data().unwrap()
    );
    // This provider contributes exactly one device pool for `(R, f64,
    // CudaStorage<f64>)`; the second call leases that pool's idle workspace.
    assert_eq!(
        after_cached.workspaces_created - before_cached.workspaces_created,
        1,
        "one device workspace per (provider, dtype, storage)"
    );
    assert!(
        after_cached.workspace_reuses > before_cached.workspace_reuses,
        "the warm replay must reuse the leased workspace, not build a new one"
    );
    // Each cached plan keeps at most two typed pools of at most two idle
    // workspaces, so the runtime-wide idle count stays inside that bound.
    assert!(
        after_cached.idle_workspaces >= 1
            && after_cached.idle_workspaces <= 4 * after_cached.entries,
        "idle workspaces {} outside the plan-wide bound for {} entries",
        after_cached.idle_workspaces,
        after_cached.entries
    );
}

/// The reversed-output product the canonical predicate refused
/// runs on the device, equals the Host run, and shares the structural plan the
/// Host call published (one entry, a hit).
#[test]
#[ignore = "requires a real CUDA device"]
fn a_noncanonical_cuda_network_runs_and_shares_the_host_plan() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let a =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 748_091).unwrap();
    let b =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 748_092).unwrap();
    let a_cuda = a.to_cuda(cuda).unwrap();
    let b_cuda = b.to_cuda(cuda).unwrap();

    let host = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
        .contract(engine, &[&a, &b])
        .unwrap();
    let published = plan_cache_stats(&runtime);
    assert_eq!(published.entries, 1);
    for _ in 0..2 {
        let device = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
            .contract(cuda, &[&a_cuda, &b_cuda])
            .unwrap();
        assert_eq!(device.codomain(), host.codomain());
        assert_eq!(device.domain(), host.domain());
        assert_close_dyn(
            device.to_host(cuda).unwrap().dense_data().unwrap(),
            host.dense_data().unwrap(),
            f64::EPSILON,
            "reversed pair",
        );
    }
    let after = plan_cache_stats(&runtime);
    assert_eq!(
        after.entries, 1,
        "Host and device share one structural plan"
    );
    assert!(after.hits > published.hits);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn canonical_cuda_network_provider_matrix_chain_and_lazy_conj() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    cuda_pair(engine, cuda, &runtime, &u1, 748_100);
    let su2 =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 2)]).unwrap();
    cuda_pair(engine, cuda, &runtime, &su2, 748_110);
    let fz2 = GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 2)]).unwrap();
    cuda_pair(engine, cuda, &runtime, &fz2, 748_120);
    let product = GradedSpace::try_new(
        Arc::new(
            symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
                .unwrap(),
        ),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    cuda_pair(engine, cuda, &runtime, &product, 748_130);

    let tensors = (0..3)
        .map(|index| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(
                engine,
                &runtime,
                [&u1],
                [&u1],
                748_140 + index,
            )
            .unwrap()
            .to_cuda(cuda)
            .unwrap()
        })
        .collect::<Vec<_>>();
    let chain = Network::new(
        vec![vec!["a", "b"], vec!["b", "c"], vec!["c", "d"]],
        vec![false; 3],
        vec![1; 3],
        vec!["a", "d"],
        Some(1),
    )
    .unwrap();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let ir = NetworkIR::from_labels(
        [["a", "b"], ["b", "c"], ["c", "d"]]
            .map(|labels| labels.map(TemporaryLabel::from).to_vec())
            .to_vec(),
        ["a", "d"].map(TemporaryLabel::from).to_vec(),
    )
    .unwrap();
    let chain_order = ContractionPlan::from_steps(
        &ir,
        vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                ["a", "c"].map(TemporaryLabel::from).to_vec(),
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                ["a", "d"].map(TemporaryLabel::from).to_vec(),
            ),
        ],
    )
    .unwrap();
    let planned = chain.plan_with(&refs, chain_order).unwrap();
    let actual = planned
        .execute(cuda, &refs, &mut NetworkExecutionWorkspace::default())
        .unwrap();
    let manual = tensors[0]
        .contract(
            cuda,
            &tensors[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            cuda,
            &tensors[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let chain_cached = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
        &["a"],
        &["d"],
    )
    .contract(cuda, &[&(tensors[0]), &(tensors[1]), &(tensors[2])])
    .unwrap();
    assert_eq!(actual.codomain(), manual.codomain());
    assert_eq!(actual.domain(), manual.domain());
    // The planned chain, the cached contraction and the chained device contract may order
    // the two length-2 contractions differently: path agreement within the
    // tolerance rule over 2 * 2 terms.
    numerics::assert_slices_close(
        "chain vs device contract",
        actual.to_host(cuda).unwrap().dense_data().unwrap(),
        manual.to_host(cuda).unwrap().dense_data().unwrap(),
        4,
    );
    numerics::assert_slices_close(
        "chain cached vs device contract",
        chain_cached.to_host(cuda).unwrap().dense_data().unwrap(),
        manual.to_host(cuda).unwrap().dense_data().unwrap(),
        4,
    );
    assert_eq!(chain_cached.placement(), tensors[0].placement());

    let conj_network = Network::new(
        vec![vec!["k", "i"], vec!["k", "j"]],
        vec![true, false],
        vec![1, 1],
        vec!["i", "j"],
        Some(1),
    )
    .unwrap();
    let conj_refs = [&tensors[0], &tensors[1]];
    let conj_actual = conj_network
        .plan(&conj_refs, &GreedyDenseOptimizer)
        .unwrap()
        .execute(cuda, &conj_refs, &mut NetworkExecutionWorkspace::default())
        .unwrap();
    let conj_manual = tensors[0]
        .adjoint(cuda)
        .unwrap()
        .contract(
            cuda,
            &tensors[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let conj_cached = net(
        &[conj(op(&["k"], &["i"])), op(&["k"], &["j"])],
        &["i"],
        &["j"],
    )
    .contract(cuda, &[&(tensors[0]), &(tensors[1])])
    .unwrap();
    assert_eq!(conj_actual.codomain(), conj_manual.codomain());
    assert_eq!(conj_actual.domain(), conj_manual.domain());
    numerics::assert_slices_close(
        "conj network vs device contract",
        conj_actual.to_host(cuda).unwrap().dense_data().unwrap(),
        conj_manual.to_host(cuda).unwrap().dense_data().unwrap(),
        2,
    );
    numerics::assert_slices_close(
        "conj cached vs device contract",
        conj_cached.to_host(cuda).unwrap().dense_data().unwrap(),
        conj_manual.to_host(cuda).unwrap().dense_data().unwrap(),
        2,
    );

    let single = Network::new(
        vec![vec!["k", "i"]],
        vec![true],
        vec![1],
        vec!["i", "k"],
        Some(1),
    )
    .unwrap();
    let single_actual = single
        .plan(&[&tensors[0]], &GreedyDenseOptimizer)
        .unwrap()
        .execute(
            cuda,
            &[&tensors[0]],
            &mut NetworkExecutionWorkspace::default(),
        )
        .unwrap();
    let single_expected = tensors[0].adjoint(cuda).unwrap();
    let single_cached = net(&[conj(op(&["k"], &["i"]))], &["i"], &["k"])
        .contract(cuda, &[&(tensors[0])])
        .unwrap();
    assert!(std::ptr::eq(
        single_actual.provider(),
        tensors[0].provider()
    ));
    assert_eq!(single_actual.codomain(), single_expected.codomain());
    assert_eq!(single_actual.domain(), single_expected.domain());
    assert_eq!(
        single_actual
            .to_host(cuda)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        single_expected
            .to_host(cuda)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert_eq!(
        single_cached
            .to_host(cuda)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        single_expected
            .to_host(cuda)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    let ket = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        [&u1],
        748_150,
    )
    .unwrap()
    .to_cuda(cuda)
    .unwrap();
    let bra = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&u1],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        748_151,
    )
    .unwrap()
    .to_cuda(cuda)
    .unwrap();
    let scalar_network = Network::new(
        vec![vec!["k"], vec!["k"]],
        vec![false; 2],
        vec![0, 1],
        vec![],
        Some(0),
    )
    .unwrap();
    let scalar = scalar_network
        .plan(&[&ket, &bra], &GreedyDenseOptimizer)
        .unwrap()
        .execute(
            cuda,
            &[&ket, &bra],
            &mut NetworkExecutionWorkspace::default(),
        )
        .unwrap();
    let scalar_cached = net(&[op(&[], &["k"]), op(&["k"], &[])], &[], &[])
        .contract(cuda, &[&ket, &bra])
        .unwrap();
    assert_eq!(scalar.rank(), 0);
    assert!(scalar.codomain().is_empty());
    assert!(scalar.domain().is_empty());
    assert!(std::ptr::eq(scalar.provider(), ket.provider()));
    // Macro and planned network are two device paths for one length-2 inner
    // product; they agree within the tolerance rule.
    numerics::assert_slices_close(
        "scalar cached vs scalar network",
        scalar_cached.to_host(cuda).unwrap().dense_data().unwrap(),
        scalar.to_host(cuda).unwrap().dense_data().unwrap(),
        2,
    );
    let stats = plan_cache_stats(&runtime);
    assert!(
        stats.workspaces_created >= 4,
        "each of the four providers leases its own device workspace"
    );
    assert!(
        stats.idle_workspaces <= 4 * stats.entries,
        "idle workspaces {} outside the plan-wide bound for {} entries",
        stats.idle_workspaces,
        stats.entries
    );
}

/// Canonical `Network::contract` device execution with a genuinely complex
/// payload, including a lazy conjugate operand, a reversed output and
/// an intra-operand trace.
#[test]
#[ignore = "requires a real CUDA device"]
fn canonical_cuda_network_executes_complex_payloads_and_still_rejects_the_rest() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let complex_entry = |indices: &[usize], seed: f64| {
        let ramp = indices.iter().map(|&index| index as f64).sum::<f64>();
        Complex64::new(ramp + seed, -(ramp + seed + 0.75))
    };
    let host: Vec<TensorMap<U1FusionRule, Complex64>> = (0..3)
        .map(|index| {
            TensorMap::from_subblock_fn(engine, &runtime, [&u1], [&u1], |_, indices| {
                complex_entry(indices, 1.0 + index as f64)
            })
            .unwrap()
        })
        .collect();
    let device: Vec<_> = host
        .iter()
        .map(|tensor| tensor.to_cuda(cuda).unwrap())
        .collect();

    let refs: [&TensorMap<U1FusionRule, Complex64, CudaStorage<Complex64>>; 2] =
        [&device[0], &device[1]];
    let planned = pair_network().plan(&refs, &GreedyDenseOptimizer).unwrap();
    let executed = planned
        .execute(cuda, &refs, &mut NetworkExecutionWorkspace::default())
        .unwrap();
    let host_oracle = host[0]
        .contract(
            engine,
            &host[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(executed.placement(), device[0].placement());
    assert_close_c64(
        executed.to_host(cuda).unwrap().dense_data().unwrap(),
        host_oracle.dense_data().unwrap(),
    );

    let chain = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
        &["a"],
        &["d"],
    )
    .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
    .unwrap();
    let chain_oracle = host[0]
        .contract(
            engine,
            &host[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            engine,
            &host[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_close_c64(
        chain.to_host(cuda).unwrap().dense_data().unwrap(),
        chain_oracle.dense_data().unwrap(),
    );
    let chain_warm = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
        &["a"],
        &["d"],
    )
    .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
    .unwrap();
    assert_close_c64(
        chain_warm.to_host(cuda).unwrap().dense_data().unwrap(),
        chain_oracle.dense_data().unwrap(),
    );

    let conj = net(
        &[conj(op(&["k"], &["i"])), op(&["k"], &["j"])],
        &["i"],
        &["j"],
    )
    .contract(cuda, &[&(device[0]), &(device[1])])
    .unwrap();
    let conj_oracle = host[0]
        .adjoint(engine)
        .unwrap()
        .contract(
            engine,
            &host[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_close_c64(
        conj.to_host(cuda).unwrap().dense_data().unwrap(),
        conj_oracle.dense_data().unwrap(),
    );

    let trace = device[0].trace_pairs(cuda, &[(0, 1)]).unwrap();
    let trace_oracle = host[0].trace_pairs(engine, &[(0, 1)]).unwrap();
    assert_close_c64(
        trace.to_host(cuda).unwrap().dense_data().unwrap(),
        trace_oracle.dense_data().unwrap(),
    );
    // The reversed output the canonical predicate refused runs.
    let reversed = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
        .contract(cuda, &[&(device[0]), &(device[1])])
        .unwrap();
    let reversed_oracle = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
        .contract(engine, &[&(host[0]), &(host[1])])
        .unwrap();
    assert_close_c64(
        reversed.to_host(cuda).unwrap().dense_data().unwrap(),
        reversed_oracle.dense_data().unwrap(),
    );
}

fn assert_close_c64(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() <= 1e-12 * (1.0 + expected.norm()),
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

fn u1_pair(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    space: &GradedSpace<U1FusionRule>,
    seed: u64,
) -> (
    TensorMap<U1FusionRule, f64>,
    TensorMap<U1FusionRule, f64, CudaStorage>,
) {
    let host =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, runtime, [space], [space], seed)
            .unwrap();
    let device = host.to_cuda(cuda).unwrap();
    (host, device)
}

/// Host and device execution of one topology and one `(R, D)`
/// key separate workspace pools. A shared key would hand the device lease a
/// `NetworkExecutionWorkspace<_, _, Vec<f64>>` and panic in the registry
/// downcast.
#[test]
#[ignore = "requires a real CUDA device"]
fn host_and_cuda_contractions_of_one_topology_use_separate_workspace_pools() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let (a, a_cuda) = u1_pair(engine, cuda, &runtime, &u1, 748_400);
    let (b, b_cuda) = u1_pair(engine, cuda, &runtime, &u1, 748_401);

    let host = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&a, &b])
        .unwrap();
    let device = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(cuda, &[&a_cuda, &b_cuda])
        .unwrap();
    numerics::assert_slices_close(
        "device vs host",
        device.to_host(cuda).unwrap().dense_data().unwrap(),
        host.dense_data().unwrap(),
        2,
    );

    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 1, "one structural topology is shared");
    assert_eq!(
        stats.workspaces_created, 2,
        "storage is part of the pool key: one Host pool and one device pool"
    );
}

/// A device call rejected from metadata leases no workspace,
/// and the next valid call reuses the idle one. (A leg mismatch on a
/// plan-cache hit does not reach execution; quarantine of a failing lease is
/// the lease-level `panic_quarantines_typed_workspace_lease` unit test.)
#[test]
#[ignore = "requires a real CUDA device"]
fn a_rejected_cuda_call_leases_no_workspace_and_the_next_call_reuses_the_idle_one() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let wrong = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let (_, a_cuda) = u1_pair(engine, cuda, &runtime, &u1, 748_410);
    let (_, b_cuda) = u1_pair(engine, cuda, &runtime, &u1, 748_411);
    let (_, mismatched) = u1_pair(engine, cuda, &runtime, &wrong, 748_412);

    let warm = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(cuda, &[&a_cuda, &b_cuda])
        .unwrap();
    let after_warm = plan_cache_stats(&runtime);
    assert_eq!(
        after_warm.idle_workspaces, 1,
        "a successful device call recycles its lease"
    );

    // The contracted-leg mismatch is decided before the plan lookup,
    // so no workspace is leased, let alone quarantined.
    assert!(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(cuda, &[&a_cuda, &mismatched])
            .is_err()
    );
    assert_eq!(plan_cache_stats(&runtime), after_warm);

    let rebuilt = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(cuda, &[&a_cuda, &b_cuda])
        .unwrap();
    assert_eq!(
        rebuilt.to_host(cuda).unwrap().dense_data().unwrap(),
        warm.to_host(cuda).unwrap().dense_data().unwrap()
    );
    let after = plan_cache_stats(&runtime);
    assert_eq!(after.workspaces_created, after_warm.workspaces_created);
    assert_eq!(after.idle_workspaces, 1, "the idle workspace is reused");
}

/// Shape drift that keeps the
/// payload length identical but changes the block layout still discards the
/// cached replay state. The length check alone would accept it; the per-operand
/// sector snapshot must not. Because destinations are reused, a retained
/// device buffer of the other charge would survive the drift, so the drifted
/// result — and the undrifted result computed after it — are direct evidence
/// that the stale payload was dropped.
#[test]
#[ignore = "requires a real CUDA device"]
fn equal_length_block_layout_drift_discards_the_device_replay_state() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    // Both spaces carry one sector of degeneracy 2, so `[v; v]` has a single
    // 2x2 coupled block and exactly the same `required_len` either way.
    let charge_zero = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let charge_three =
        GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(3), 2)]).unwrap();
    let (a0, a0_cuda) = u1_pair(engine, cuda, &runtime, &charge_zero, 748_420);
    let (b0, b0_cuda) = u1_pair(engine, cuda, &runtime, &charge_zero, 748_421);
    let (a3, a3_cuda) = u1_pair(engine, cuda, &runtime, &charge_three, 748_422);
    let (b3, b3_cuda) = u1_pair(engine, cuda, &runtime, &charge_three, 748_423);
    assert_eq!(
        a0.dense_data().unwrap().len(),
        a3.dense_data().unwrap().len()
    );

    let first = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(cuda, &[&a0_cuda, &b0_cuda])
        .unwrap();
    numerics::assert_slices_close(
        "device vs host",
        first.to_host(cuda).unwrap().dense_data().unwrap(),
        a0.contract(
            engine,
            &b0,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .dense_data()
        .unwrap(),
        2,
    );
    let after_first = plan_cache_stats(&runtime);

    let drifted = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(cuda, &[&a3_cuda, &b3_cuda])
        .unwrap();
    numerics::assert_slices_close(
        "drifted device vs host",
        drifted.to_host(cuda).unwrap().dense_data().unwrap(),
        a3.contract(
            engine,
            &b3,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .dense_data()
        .unwrap(),
        2,
    );

    let after_drift = plan_cache_stats(&runtime);
    assert_eq!(
        after_drift.workspaces_created, after_first.workspaces_created,
        "drift discards the replay state inside the pooled workspace, not the workspace"
    );
    assert!(
        after_drift.workspace_reuses > after_first.workspace_reuses,
        "the drifted call still leases the pooled device workspace"
    );

    // Back to the original charges: the workspace rebuilt for the drifted
    // layout must not leak into this one either.
    let restored = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(cuda, &[&a0_cuda, &b0_cuda])
        .unwrap();
    assert_eq!(
        restored.to_host(cuda).unwrap().dense_data().unwrap(),
        first.to_host(cuda).unwrap().dense_data().unwrap()
    );
}

type ChainPair<R> = (Vec<TensorMap<R, f64>>, Vec<TensorMap<R, f64, CudaStorage>>);

/// A three-tensor canonical chain, which is the smallest schedule with a
/// reusable destination: step 0 overwrites the retained intermediate, step 1
/// produces the returned final output.
fn cuda_chain_tensors<R>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    space: &GradedSpace<R>,
    seed: u64,
) -> ChainPair<R>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let host: Vec<_> = (0..3)
        .map(|index| {
            TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], seed + index)
                .unwrap()
        })
        .collect();
    let device = host.iter().map(|t| t.to_cuda(cuda).unwrap()).collect();
    (host, device)
}

fn cuda_chain_reuse_matches_returning<R>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    space: &GradedSpace<R>,
    seed: u64,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let (host, device) = cuda_chain_tensors(engine, cuda, runtime, space, seed);
    let returning = device[0]
        .contract(
            cuda,
            &device[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            cuda,
            &device[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let host_oracle = host[0]
        .contract(
            engine,
            &host[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            engine,
            &host[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let cold = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
        &["a"],
        &["d"],
    )
    .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
    .unwrap();
    let warm = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
        &["a"],
        &["d"],
    )
    .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
    .unwrap();
    let terms = space.dim().unwrap().ceil() as usize;
    numerics::assert_slices_close(
        "cold device chain vs host",
        cold.to_host(cuda).unwrap().dense_data().unwrap(),
        host_oracle.dense_data().unwrap(),
        terms * terms,
    );
    // The warm replay submits the same kernels in the same order to the same
    // device, so f64 is bitwise equal to the returning result. This relies on
    // cuBLAS/cuTENSOR determinism for identical submissions, as the existing
    // device equality assertions already do.
    assert_eq!(
        warm.to_host(cuda).unwrap().dense_data().unwrap(),
        returning.to_host(cuda).unwrap().dense_data().unwrap()
    );
}

/// Warm `Network::contract` reuses its retained device destinations and
/// still returns exactly the returning contraction, for every admitted
/// categorical structure.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_cuda_destination_reuse_matches_the_returning_chain_for_every_provider() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    cuda_chain_reuse_matches_returning(engine, cuda, &runtime, &u1, 761_100);
    let su2 =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 2)]).unwrap();
    cuda_chain_reuse_matches_returning(engine, cuda, &runtime, &su2, 761_110);
    let fz2 = GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 2)]).unwrap();
    cuda_chain_reuse_matches_returning(engine, cuda, &runtime, &fz2, 761_120);
    let product = GradedSpace::try_new(
        Arc::new(
            symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
                .unwrap(),
        ),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
    )
    .unwrap();
    cuda_chain_reuse_matches_returning(engine, cuda, &runtime, &product, 761_130);

    // Complex payloads take the same route; the device kernels differ, so this
    // is a tolerance comparison against the Host oracle.
    let device: Vec<_> = (0..3)
        .map(|index| {
            TensorMap::<U1FusionRule, Complex64>::rand_with_seed(
                engine,
                &runtime,
                [&u1],
                [&u1],
                761_140 + index,
            )
            .unwrap()
        })
        .collect();
    let on_cuda: Vec<_> = device.iter().map(|t| t.to_cuda(cuda).unwrap()).collect();
    let oracle = device[0]
        .contract(
            engine,
            &device[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            engine,
            &device[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    drop(
        net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(cuda, &[&(on_cuda[0]), &(on_cuda[1]), &(on_cuda[2])])
        .unwrap(),
    );
    let warm = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
        &["a"],
        &["d"],
    )
    .contract(cuda, &[&(on_cuda[0]), &(on_cuda[1]), &(on_cuda[2])])
    .unwrap();
    assert_close_c64(
        warm.to_host(cuda).unwrap().dense_data().unwrap(),
        oracle.dense_data().unwrap(),
    );
}

/// The warm device replay of an N-tensor chain
/// performs no host-to-device traffic, no device allocation and no copy for
/// its N-2 intermediate steps: every destination block of this chain has a
/// GEMM, so a retained destination needs no reset at all. Only the final,
/// returned output still uploads its zeros.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_cuda_chain_uploads_nothing_for_its_reused_destinations() {
    use symtn::expert::cuda_transfer_stats;

    let engine = &mut symtn::typed::CpuEngine::serial();

    let cuda = &mut cuda_engine();

    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let (_, device) = cuda_chain_tensors(engine, cuda, &runtime, &u1, 761_200);
    let (_, extra) = cuda_chain_tensors(engine, cuda, &runtime, &u1, 761_300);

    // Four tensors: three steps, two of them over retained destinations.
    let mut run = || {
        net(
            &[
                op(&["a"], &["b"]),
                op(&["b"], &["c"]),
                op(&["c"], &["d"]),
                op(&["d"], &["e"]),
            ],
            &["a"],
            &["e"],
        )
        .contract(
            cuda,
            &[&(device[0]), &(device[1]), &(device[2]), &(extra[0])],
        )
        .unwrap()
    };
    // Call 1 has nothing retained yet, so every step returns.
    drop(run());
    // Call 2 is the first to overwrite: nothing but the returned output's own
    // zeros is uploaded, and nothing is reset.
    let before_second = cuda_transfer_stats();
    drop(run());
    let after_second = cuda_transfer_stats();
    assert_eq!(
        (
            after_second.h2d_calls - before_second.h2d_calls,
            after_second.copy_calls - before_second.copy_calls,
        ),
        (1, 0),
        "overwriting a retained destination uploads and copies nothing"
    );

    // Call 3 onward is the steady state this contract describes.
    let before = cuda_transfer_stats();
    let warm = run();
    let after = cuda_transfer_stats();

    assert_eq!(
        (
            after.h2d_calls - before.h2d_calls,
            after.device_allocs - before.device_allocs,
            after.copy_calls - before.copy_calls,
            after.d2h_calls - before.d2h_calls,
            after.gemm_calls - before.gemm_calls,
        ),
        // One GEMM per coupled sector (charges 0 and 1) of each of the
        // three steps.
        (1, 1, 0, 0, 3 * 2),
        "(h2d, device_allocs, d2d_copies, d2h, gemm) for the warm 4-tensor chain"
    );
    drop(warm);
}

/// An idle device workspace is charged to the one
/// `workspace_budget_bytes` ledger, so a budget one byte short of its charge
/// rejects it whole. The charge holds no reset scratch:
/// a retained device destination is overwritten without a reset.
#[test]
#[ignore = "requires a real CUDA device"]
fn the_device_workspace_is_charged_to_the_workspace_budget() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 8)]).unwrap();
    let (_, device) = cuda_chain_tensors(engine, cuda, &runtime, &u1, 761_400);

    drop(
        net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
        .unwrap(),
    );
    let charge = plan_cache_stats(&runtime).retained_workspace_bytes;
    assert!(
        charge > 0,
        "the idle device workspace retains its destinations"
    );

    clear_plan_cache(&runtime);
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge - 1,
            ..Default::default()
        },
    );
    drop(
        net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
        .unwrap(),
    );
    let rejected = plan_cache_stats(&runtime);
    assert_eq!(rejected.retained_workspace_bytes, 0);
    assert_eq!(rejected.idle_workspaces, 0);
    assert_eq!(rejected.workspace_byte_rejections, 1);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge,
            ..Default::default()
        },
    );
    drop(
        net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
        .unwrap(),
    );
    let admitted = plan_cache_stats(&runtime);
    assert_eq!(admitted.retained_workspace_bytes, charge);
    assert_eq!(admitted.idle_workspaces, 1);
}

// ---------------------------------------------------------------------------
// Single-precision device networks
// ---------------------------------------------------------------------------

/// The device network path is generic over the payload dtype, so the chain
/// contracts have single-precision twins that run the same schedule. The
/// oracle is the host chain *at the same dtype*; the tolerance is the payload's
/// own, not `f64`'s.
type TypedChainPair<R, D> = (Vec<TensorMap<R, D>>, Vec<TensorMap<R, D, CudaStorage<D>>>);

fn cuda_chain_tensors_at<R, D>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    space: &GradedSpace<R>,
    seed: u64,
) -> TypedChainPair<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: symtn::typed::CudaPayload,
{
    let host: Vec<_> = (0..3)
        .map(|index| {
            TensorMap::<R, D>::rand_with_seed(engine, runtime, [space], [space], seed + index)
                .unwrap()
        })
        .collect();
    let device = host.iter().map(|t| t.to_cuda(cuda).unwrap()).collect();
    (host, device)
}

/// With a non-Abelian provider, a
/// `Network::contract` device chain at `f32` and `Complex32` produces the host result of
/// the same dtype, cold and warm, and the warm replay keeps the
/// retained-destination contract of the `f64` chain.
#[test]
#[ignore = "requires a real CUDA device"]
fn single_precision_device_chains_match_the_host_and_reuse_their_destinations() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    // SU(2): several coupled sectors carrying different quantum dimensions, so
    // the chain's recoupling — not only its GEMMs — runs at single precision.
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();

    /// `K * sqrt(terms) * eps(real(D))` at the chain's own scale: derived from
    /// the payload's epsilon, never an absolute platform constant.
    fn chain_tolerance(epsilon: f64) -> f64 {
        32.0 * (64.0_f64).sqrt() * epsilon
    }

    fn assert_chain<R, D>(
        engine: &mut dyn Engine,
        cuda: &mut dyn Engine,
        runtime: &Runtime,
        space: &GradedSpace<R>,
        seed: u64,
        tolerance: f64,
    ) where
        R: TypedSectorAdmission<
                Sector = <R as symtn::sector::SectorCodec>::Sector,
                Error = FusionAlgebraError,
                Mode = MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync,
        D: symtn::typed::CudaPayload + Copy + std::fmt::Debug,
    {
        let (host, device) = cuda_chain_tensors_at::<R, D>(engine, cuda, runtime, space, seed);
        let oracle = host[0]
            .contract(
                engine,
                &host[1],
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .contract(
                engine,
                &host[2],
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        let cold = net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
        .unwrap();
        let warm = net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
        .unwrap();
        for (label, result) in [("cold", cold), ("warm", warm)] {
            let actual = result.to_host(cuda).unwrap();
            assert_eq!(
                actual.dense_data().unwrap().len(),
                oracle.dense_data().unwrap().len()
            );
            for (index, (&got, &want)) in actual
                .dense_data()
                .unwrap()
                .iter()
                .zip(oracle.dense_data().unwrap())
                .enumerate()
            {
                let distance = (got.widen_complex() - want.widen_complex()).norm();
                assert!(
                    distance <= tolerance * (1.0 + want.widen_complex().norm()),
                    "{label} chain element {index}: {got:?} vs {want:?} (distance {distance})"
                );
            }
        }
    }

    let double = chain_tolerance(f64::EPSILON);
    let single = chain_tolerance(f64::from(f32::EPSILON));

    assert_chain::<_, f64>(engine, cuda, &runtime, &u1, 762_100, double);
    assert_chain::<_, f32>(engine, cuda, &runtime, &u1, 762_200, single);
    assert_chain::<_, Complex64>(engine, cuda, &runtime, &u1, 762_300, double);
    assert_chain::<_, Complex32>(engine, cuda, &runtime, &u1, 762_400, single);

    assert_chain::<_, f64>(engine, cuda, &runtime, &su2, 762_500, double);
    assert_chain::<_, f32>(engine, cuda, &runtime, &su2, 762_600, single);
    assert_chain::<_, Complex64>(engine, cuda, &runtime, &su2, 762_700, double);
    assert_chain::<_, Complex32>(engine, cuda, &runtime, &su2, 762_800, single);
}

/// The warm single-precision chain costs the same device
/// calls as the `f64` chain of the same fixture and moves half the bytes.
///
/// Relative, same-process comparison: both dtypes run the same fixture in this
/// binary, so no absolute platform constant appears.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_single_precision_chain_costs_the_same_calls_and_half_the_bytes() {
    use symtn::expert::cuda_transfer_stats;

    let engine = &mut symtn::typed::CpuEngine::serial();

    let cuda = &mut cuda_engine();

    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)],
    )
    .unwrap();

    fn warm_cost<D>(
        engine: &mut dyn Engine,
        cuda: &mut dyn Engine,
        runtime: &Runtime,
        u1: &GradedSpace<U1FusionRule>,
        seed: u64,
    ) -> (u64, u64, u64, u64, u64, u64)
    where
        D: symtn::typed::CudaPayload,
    {
        let (_, device) = cuda_chain_tensors_at::<U1FusionRule, D>(engine, cuda, runtime, u1, seed);
        let mut run = || {
            net(
                &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
                &["a"],
                &["d"],
            )
            .contract(cuda, &[&(device[0]), &(device[1]), &(device[2])])
            .unwrap()
        };
        // Two warm-up runs: the first has nothing retained, the second is the
        // first to overwrite retained destinations.
        drop(run());
        drop(run());
        let before = cuda_transfer_stats();
        drop(run());
        let after = cuda_transfer_stats();
        (
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_calls - before.d2h_calls,
            after.device_allocs - before.device_allocs,
            after.copy_calls - before.copy_calls,
            after.gemm_calls - before.gemm_calls,
        )
    }

    // Both real and complex lanes: the complex twin is what shows the halving
    // is the *element size*, not the real/complex split.
    for (name, single, double) in [
        (
            "f32/f64",
            warm_cost::<f32>(engine, cuda, &runtime, &u1, 763_200),
            warm_cost::<f64>(engine, cuda, &runtime, &u1, 763_100),
        ),
        (
            "c32/c64",
            warm_cost::<Complex32>(engine, cuda, &runtime, &u1, 763_400),
            warm_cost::<Complex64>(engine, cuda, &runtime, &u1, 763_300),
        ),
    ] {
        eprintln!("warm chain {name}: narrow {single:?} wide {double:?}");
        assert_eq!(
            (single.0, single.2, single.3, single.4, single.5),
            (double.0, double.2, double.3, double.4, double.5),
            "{name}: (h2d_calls, d2h_calls, device_allocs, copy_calls, gemm_calls) must not \
             depend on the dtype"
        );
        assert!(double.1 > 0, "{name}: vacuous byte count");
        assert_eq!(
            single.1 * 2,
            double.1,
            "{name}: warm h2d bytes must be halved"
        );
    }
}

// ---------------------------------------------------------------------------
// General device networks
// ---------------------------------------------------------------------------

/// Device result against a reference of the same dtype, compared at the
/// payload's own epsilon (never bitwise: the device GEMM order is cuTENSOR's
/// and exact zeros may differ in sign).
fn assert_close_dyn<D: symtn::typed::CudaPayload + std::fmt::Debug>(
    actual: &[D],
    expected: &[D],
    epsilon: f64,
    what: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{what}: payload length");
    let scale = expected
        .iter()
        .map(|value| value.widen_complex().norm())
        .fold(0.0_f64, f64::max);
    assert!(scale > 0.0, "{what}: all-zero reference proves nothing");
    let tolerance = 256.0 * epsilon * (1.0 + scale);
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let distance = (got.widen_complex() - want.widen_complex()).norm();
        assert!(
            distance <= tolerance,
            "{what}: element {index} is {got:?}, expected {want:?} (tolerance {tolerance:e})"
        );
    }
}

/// Runs `network` once over the Host operands, then twice — cold and warm —
/// over the device operands.
fn host_cold_warm<H, G>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    network: &Network,
    host: &[&H],
    device: &[&G],
) -> (H, G, G)
where
    H: NetworkOperand,
    G: NetworkOperand,
    H::Error: std::fmt::Debug,
    G::Error: std::fmt::Debug,
{
    (
        network.contract(engine, host).unwrap(),
        network.contract(cuda, device).unwrap(),
        network.contract(cuda, device).unwrap(),
    )
}

/// The operands of the general networks, over three spaces `v`, `w`, `p` of
/// one provider (degeneracy > 1 in each):
///
/// - `a: [v, w; p]`, `b: [p, w; v, p]`, `c: [v; w]`, `e: [w; w]`, and `t` with
///   `conj(t)` shaped like `b` — codomain↔domain contractions only, so the
///   physical-basis dense expansion is a plain index sum;
/// - `f: [v, w; p]`, `g: [v*, w; p*]`, `h: [w*, v;]` — codomain↔codomain and
///   domain↔domain pairs, i.e. dual contracted legs (the fermionic twist).
struct Operands<T> {
    a: T,
    b: T,
    c: T,
    e: T,
    t: T,
    f: T,
    g: T,
    h: T,
}

impl<T> Operands<T> {
    fn map<U>(&self, mut lift: impl FnMut(&T) -> U) -> Operands<U> {
        Operands {
            a: lift(&self.a),
            b: lift(&self.b),
            c: lift(&self.c),
            e: lift(&self.e),
            t: lift(&self.t),
            f: lift(&self.f),
            g: lift(&self.g),
            h: lift(&self.h),
        }
    }
}

fn general_operands<R, D>(
    engine: &mut dyn Engine,
    runtime: &Runtime,
    [v, w, p]: [&GradedSpace<R>; 3],
    seed: u64,
) -> Operands<TensorMap<R, D>>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: symtn::typed::CudaPayload,
{
    let mut rand = |codomain: &[&GradedSpace<R>], domain: &[&GradedSpace<R>], salt: u64| {
        TensorMap::<R, D>::rand_with_seed(
            engine,
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            seed + salt,
        )
        .unwrap()
    };
    let (v_dual, w_dual, p_dual) = (
        v.try_dual().unwrap(),
        w.try_dual().unwrap(),
        p.try_dual().unwrap(),
    );
    Operands {
        a: rand(&[v, w], &[p], 0),
        b: rand(&[p, w], &[v, p], 1),
        c: rand(&[v], &[w], 2),
        e: rand(&[w], &[w], 3),
        t: rand(&[v, p], &[p, w], 4),
        f: rand(&[v, w], &[p], 5),
        g: rand(&[&v_dual, w], &[&p_dual], 6),
        h: rand(&[&w_dual, v], &[], 7),
    }
}

/// Every general network on Host and device (cold, warm): general contraction
/// axes, open outputs on both sides, result and final permutations (a
/// split-changing one included), a lazy conjugate operand, dual contracted
/// legs, and a four-tensor chain whose two intermediates are reused warm.
/// Returns `(name, host, device cold, device warm)`.
#[allow(clippy::type_complexity)]
fn general_networks<R, D>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    host: &Operands<TensorMap<R, D>>,
    device: &Operands<TensorMap<R, D, CudaStorage<D>>>,
) -> Vec<(
    &'static str,
    TensorMap<R, D>,
    TensorMap<R, D, CudaStorage<D>>,
    TensorMap<R, D, CudaStorage<D>>,
)>
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: symtn::typed::CudaPayload + Send + Sync + 'static,
{
    let mut runs = Vec::new();
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                op(&["p", "x"], &["a", "y"]),
                op(&["q"], &["w"]),
            ],
            &["x", "q"],
            &["y"],
        ),
        &[&host.a, &host.b, &host.c],
        &[&device.a, &device.b, &device.c],
    );
    runs.push(("xq;y", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                op(&["p", "x"], &["a", "y"]),
                op(&["q"], &["w"]),
            ],
            &["q", "x"],
            &["y"],
        ),
        &[&host.a, &host.b, &host.c],
        &[&device.a, &device.b, &device.c],
    );
    runs.push(("qx;y", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                op(&["p", "x"], &["a", "y"]),
                op(&["q"], &["w"]),
            ],
            &["y", "q"],
            &["x"],
        ),
        &[&host.a, &host.b, &host.c],
        &[&device.a, &device.b, &device.c],
    );
    runs.push(("yq;x split-changing", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                conj(op(&["a", "y"], &["p", "x"])),
                op(&["q"], &["w"]),
            ],
            &["x", "q"],
            &["y"],
        ),
        &[&host.a, &host.t, &host.c],
        &[&device.a, &device.t, &device.c],
    );
    runs.push(("xq;y lazy conj", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                op(&["p", "x"], &["a", "y"]),
                op(&["q"], &["u"]),
                op(&["u"], &["w"]),
            ],
            &["x", "q"],
            &["y"],
        ),
        &[&host.a, &host.b, &host.c, &host.e],
        &[&device.a, &device.b, &device.c, &device.e],
    );
    runs.push(("xq;y four tensors", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                op(&["a", "x"], &["p"]),
                op(&["w", "z"], &[]),
            ],
            &["z", "x"],
            &[],
        ),
        &[&host.f, &host.g, &host.h],
        &[&device.f, &device.g, &device.h],
    );
    runs.push(("zx; dual legs", host_result, cold, warm));
    let (host_result, cold, warm) = host_cold_warm(
        engine,
        cuda,
        &net(
            &[
                op(&["a", "w"], &["p"]),
                op(&["a", "x"], &["p"]),
                op(&["w", "z"], &[]),
            ],
            &["x"],
            &["z"],
        ),
        &[&host.f, &host.g, &host.h],
        &[&device.f, &device.g, &device.h],
    );
    runs.push(("x;z dual legs split-changing", host_result, cold, warm));
    runs
}

fn assert_general_networks_match_host<R, D>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    spaces: [&GradedSpace<R>; 3],
    seed: u64,
    epsilon: f64,
    provider: &str,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
    D: symtn::typed::CudaPayload + Send + Sync + 'static + std::fmt::Debug,
{
    let operands = general_operands::<R, D>(engine, runtime, spaces, seed);
    let device = operands.map(|tensor| tensor.to_cuda(cuda).unwrap());
    for (name, host, cold, warm) in general_networks(engine, cuda, &operands, &device) {
        let what = format!("{provider} {name}");
        for (phase, result) in [("cold", cold), ("warm", warm)] {
            assert_eq!(result.placement(), device.a.placement(), "{what} {phase}");
            assert!(std::ptr::eq(result.provider(), host.provider()), "{what}");
            assert_eq!(result.codomain(), host.codomain(), "{what} {phase}");
            assert_eq!(result.domain(), host.domain(), "{what} {phase}");
            assert_close_dyn(
                result.to_host(cuda).unwrap().dense_data().unwrap(),
                host.dense_data().unwrap(),
                epsilon,
                &format!("{what} {phase}"),
            );
        }
    }
}

fn column_major_offset(shape: &[usize], index: &[usize]) -> usize {
    index
        .iter()
        .zip(shape)
        .rev()
        .fold(0, |offset, (&i, &extent)| offset * extent + i)
}

/// Physical-basis einsum: every operand `(dense, labels)` in codomain-then-
/// domain axis order, summed over every label not in `output`. Plain index
/// pairing is exact for codomain↔domain contractions and legs that stay on
/// their side, which is all the dense-checked networks use.
fn dense_einsum(
    operands: &[(&symtn::typed::PhysicalDense<Complex64>, &[&str])],
    output: &[&str],
) -> (Vec<usize>, Vec<Complex64>) {
    let mut labels: Vec<&str> = Vec::new();
    let mut extents = Vec::new();
    for (dense, operand_labels) in operands {
        assert_eq!(dense.shape.len(), operand_labels.len());
        for (&label, &extent) in operand_labels.iter().zip(&dense.shape) {
            match labels.iter().position(|&known| known == label) {
                Some(position) => assert_eq!(extents[position], extent, "label {label}"),
                None => {
                    labels.push(label);
                    extents.push(extent);
                }
            }
        }
    }
    let position = |label: &str| labels.iter().position(|&known| known == label).unwrap();
    let output_shape: Vec<usize> = output
        .iter()
        .map(|&label| extents[position(label)])
        .collect();
    let mut result = vec![Complex64::new(0.0, 0.0); output_shape.iter().product()];
    let total: usize = extents.iter().product();
    let mut index = vec![0usize; labels.len()];
    for _ in 0..total {
        let mut product = Complex64::new(1.0, 0.0);
        for (dense, operand_labels) in operands {
            let local: Vec<usize> = operand_labels.iter().map(|&l| index[position(l)]).collect();
            product *= dense.data[column_major_offset(&dense.shape, &local)];
        }
        let out: Vec<usize> = output.iter().map(|&l| index[position(l)]).collect();
        result[column_major_offset(&output_shape, &out)] += product;
        for (axis, value) in index.iter_mut().enumerate() {
            *value += 1;
            if *value < extents[axis] {
                break;
            }
            *value = 0;
        }
    }
    (output_shape, result)
}

fn widened(dense: symtn::typed::PhysicalDense<f64>) -> symtn::typed::PhysicalDense<Complex64> {
    symtn::typed::PhysicalDense {
        shape: dense.shape,
        data: dense
            .data
            .into_iter()
            .map(|x| Complex64::new(x, 0.0))
            .collect(),
    }
}

/// Dense-expansion oracle for the codomain↔domain networks on a provider with
/// a physical basis: the Host result and the device results, cold and warm,
/// each equal the physical-basis einsum of the operands.
fn assert_dense_oracle<R>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    spaces: [&GradedSpace<R>; 3],
    seed: u64,
    provider: &str,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::PhysicalFusionBasis<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync,
{
    let o = general_operands::<R, f64>(engine, runtime, spaces, seed);
    let device = o.map(|tensor| tensor.to_cuda(cuda).unwrap());
    // Host, device cold and device warm results of one expression, each to be
    // checked against the einsum independently.
    type Runs<R> = (
        TensorMap<R, f64>,
        TensorMap<R, f64, CudaStorage>,
        TensorMap<R, f64, CudaStorage>,
    );
    let results = |(host, cold, warm): Runs<R>, cuda: &mut dyn Engine| {
        [
            host,
            cold.to_host(cuda).unwrap(),
            warm.to_host(cuda).unwrap(),
        ]
    };
    let dense = |engine: &mut dyn Engine, tensor: &TensorMap<R, f64>| {
        widened(tensor.to_physical_dense(engine).unwrap())
    };
    let (a, b, c, e) = (
        dense(engine, &o.a),
        dense(engine, &o.b),
        dense(engine, &o.c),
        dense(engine, &o.e),
    );
    let t_adjoint = o.t.adjoint(engine).unwrap();
    let t_adjoint = dense(engine, &t_adjoint);
    type DenseOperands<'a> = Vec<(&'a symtn::typed::PhysicalDense<Complex64>, &'a [&'a str])>;
    type DenseCase<'a, R> = (
        &'a str,
        [TensorMap<R, f64>; 3],
        DenseOperands<'a>,
        &'a [&'a str],
    );
    let cases: [DenseCase<'_, R>; 4] = [
        (
            "xq;y",
            results(
                host_cold_warm(
                    engine,
                    cuda,
                    &net(
                        &[
                            op(&["a", "w"], &["p"]),
                            op(&["p", "x"], &["a", "y"]),
                            op(&["q"], &["w"]),
                        ],
                        &["x", "q"],
                        &["y"],
                    ),
                    &[&o.a, &o.b, &o.c],
                    &[&device.a, &device.b, &device.c],
                ),
                cuda,
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&b, &["p", "x", "a", "y"][..]),
                (&c, &["q", "w"][..]),
            ],
            &["x", "q", "y"],
        ),
        (
            "qx;y",
            results(
                host_cold_warm(
                    engine,
                    cuda,
                    &net(
                        &[
                            op(&["a", "w"], &["p"]),
                            op(&["p", "x"], &["a", "y"]),
                            op(&["q"], &["w"]),
                        ],
                        &["q", "x"],
                        &["y"],
                    ),
                    &[&o.a, &o.b, &o.c],
                    &[&device.a, &device.b, &device.c],
                ),
                cuda,
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&b, &["p", "x", "a", "y"][..]),
                (&c, &["q", "w"][..]),
            ],
            &["q", "x", "y"],
        ),
        (
            "xq;y lazy conj",
            results(
                host_cold_warm(
                    engine,
                    cuda,
                    &net(
                        &[
                            op(&["a", "w"], &["p"]),
                            conj(op(&["a", "y"], &["p", "x"])),
                            op(&["q"], &["w"]),
                        ],
                        &["x", "q"],
                        &["y"],
                    ),
                    &[&o.a, &o.t, &o.c],
                    &[&device.a, &device.t, &device.c],
                ),
                cuda,
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&t_adjoint, &["p", "x", "a", "y"][..]),
                (&c, &["q", "w"][..]),
            ],
            &["x", "q", "y"],
        ),
        (
            "xq;y four tensors",
            results(
                host_cold_warm(
                    engine,
                    cuda,
                    &net(
                        &[
                            op(&["a", "w"], &["p"]),
                            op(&["p", "x"], &["a", "y"]),
                            op(&["q"], &["u"]),
                            op(&["u"], &["w"]),
                        ],
                        &["x", "q"],
                        &["y"],
                    ),
                    &[&o.a, &o.b, &o.c, &o.e],
                    &[&device.a, &device.b, &device.c, &device.e],
                ),
                cuda,
            ),
            vec![
                (&a, &["a", "w", "p"][..]),
                (&b, &["p", "x", "a", "y"][..]),
                (&c, &["q", "u"][..]),
                (&e, &["u", "w"][..]),
            ],
            &["x", "q", "y"],
        ),
    ];
    for (name, runs, operands, output) in cases {
        let (shape, expected) = dense_einsum(&operands, output);
        for (phase, result) in ["host", "device cold", "device warm"]
            .into_iter()
            .zip(&runs)
        {
            let actual = dense(engine, result);
            assert_eq!(
                actual.shape, shape,
                "{provider} {name} {phase}: dense shape"
            );
            assert_close_dyn(
                &actual.data,
                &expected,
                f64::EPSILON,
                &format!("{provider} {name} {phase} dense"),
            );
        }
    }
}

fn u1_space(sectors: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        sectors
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2_space(sectors: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        sectors
            .iter()
            .map(|&(twice, degeneracy)| (SU2Irrep::from_twice_spin(twice), degeneracy)),
    )
    .unwrap()
}

/// General `Network::contract` networks on device equal the Host run of
/// the same expression, cold and warm, for U(1), SU(2), U(1)×SU(2),
/// fZ2×U(1) and fZ2⊠SU(2) at `f64` and `Complex64` (U(1) also at the single
/// precision payloads); the codomain↔domain networks of U(1) and SU(2) also
/// equal the physical-basis dense expansion.
#[test]
#[ignore = "requires a real CUDA device"]
fn general_cuda_networks_match_host_and_dense_oracles() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();

    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    assert_general_networks_match_host::<_, f64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        790_000,
        f64::EPSILON,
        "U(1)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        790_100,
        f64::EPSILON,
        "U(1)",
    );
    assert_general_networks_match_host::<_, f32>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        790_200,
        f64::from(f32::EPSILON),
        "U(1)",
    );
    assert_general_networks_match_host::<_, Complex32>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        790_300,
        f64::from(f32::EPSILON),
        "U(1)",
    );
    assert_dense_oracle(engine, cuda, &runtime, [&v, &w, &p], 790_400, "U(1)");

    let (v, w, p) = (
        su2_space(&[(0, 2), (1, 1)]),
        su2_space(&[(1, 2), (2, 1)]),
        su2_space(&[(0, 1), (1, 2)]),
    );
    assert_general_networks_match_host::<_, f64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        791_000,
        f64::EPSILON,
        "SU(2)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        791_100,
        f64::EPSILON,
        "SU(2)",
    );
    assert_dense_oracle(engine, cuda, &runtime, [&v, &w, &p], 791_200, "SU(2)");

    let u1_su2 = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let product = |sectors: &[(i32, usize, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&u1_su2),
            sectors.iter().map(|&(charge, twice, degeneracy)| {
                (
                    product_sector(U1Irrep::new(charge), SU2Irrep::from_twice_spin(twice)),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        // Charge parity equals twice-spin parity in every sector (the
        // Hubbard-like correlation), so every fusion product meets its
        // partner and no network result is trivially empty.
        product(&[(0, 0, 2), (1, 1, 1)]),
        product(&[(0, 0, 1), (1, 1, 2), (-1, 1, 1)]),
        product(&[(0, 0, 1), (1, 1, 2), (2, 0, 1)]),
    );
    assert_general_networks_match_host::<_, f64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        792_000,
        f64::EPSILON,
        "U(1)xSU(2)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        792_100,
        f64::EPSILON,
        "U(1)xSU(2)",
    );

    let fz2_u1 = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let fermion_u1 = |sectors: &[(bool, i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&fz2_u1),
            sectors.iter().map(|&(odd, charge, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        U1Irrep::new(charge),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        fermion_u1(&[(false, 0, 2), (true, 1, 1), (true, -1, 1)]),
        fermion_u1(&[(false, 0, 1), (true, 1, 2)]),
        fermion_u1(&[(true, 0, 1), (false, 1, 2), (true, 1, 1)]),
    );
    assert_general_networks_match_host::<_, f64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        793_000,
        f64::EPSILON,
        "fZ2xU(1)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        793_100,
        f64::EPSILON,
        "fZ2xU(1)",
    );

    let fz2_su2 = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let fermion_su2 = |sectors: &[(bool, usize, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&fz2_su2),
            sectors.iter().map(|&(odd, twice, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        SU2Irrep::from_twice_spin(twice),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        fermion_su2(&[(false, 0, 2), (true, 1, 1)]),
        fermion_su2(&[(true, 1, 2), (false, 2, 1)]),
        fermion_su2(&[(false, 0, 1), (true, 1, 2)]),
    );
    assert_general_networks_match_host::<_, f64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        794_000,
        f64::EPSILON,
        "fZ2xSU(2)",
    );
    assert_general_networks_match_host::<_, Complex64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        794_100,
        f64::EPSILON,
        "fZ2xSU(2)",
    );
}

/// Device state a rejected or warm call must leave unchanged, or change only
/// as its contract states: the Runtime plan cache and the calling thread's
/// transfer counters. Device execution scratch is per call, so nothing else
/// outlives a call.
#[derive(Debug, PartialEq)]
struct DeviceState {
    plans: symtn_network::PlanCacheStats,
    transfers: symtn::expert::CudaTransferStats,
}

fn device_state(runtime: &Runtime) -> DeviceState {
    DeviceState {
        plans: plan_cache_stats(runtime),
        transfers: symtn::expert::cuda_transfer_stats(),
    }
}

/// Warm-cost contract: a warm general network — general axes,
/// result and final permutations, reused intermediates, and for the fermion
/// the dual-leg twist — downloads nothing and creates no plan-cache entry or
/// workspace. Every intermediate step overwrites its retained buffer.
///
/// Not a contract: zero H2D beyond the returned output. Each network step
/// is the eager `contract` / `contract_into` / `permute` on the caller's
/// engine, and the network workspace hands those steps no device state, so
/// every step's tree-transform coefficients, contraction scratch and scalar
/// templates are created per call (the eager rule). A warm call therefore
/// re-uploads them; the H2D count is printed for the record. Threading the
/// network workspace's device state into the steps would restore the former
/// one-H2D budget.
///
/// Transfer counters belong to this thread; the other state belongs to its Runtime.
#[test]
#[ignore = "requires a real CUDA device"]
fn warm_general_cuda_networks_transfer_only_the_returned_output() {
    fn warm<R, D>(
        engine: &mut dyn Engine,
        cuda: &mut dyn Engine,
        runtime: &Runtime,
        spaces: [&GradedSpace<R>; 3],
        seed: u64,
        what: &str,
    ) where
        R: TypedSectorAdmission<
                Sector = <R as symtn::sector::SectorCodec>::Sector,
                Error = FusionAlgebraError,
                Mode = MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync,
        D: symtn::typed::CudaPayload + Send + Sync + 'static,
    {
        let o = general_operands::<R, D>(engine, runtime, spaces, seed)
            .map(|tensor| tensor.to_cuda(cuda).unwrap());
        let chain = |cuda: &mut dyn Engine| {
            net(
                &[
                    op(&["a", "w"], &["p"]),
                    op(&["p", "x"], &["a", "y"]),
                    op(&["q"], &["u"]),
                    op(&["u"], &["w"]),
                ],
                &["y", "q"],
                &["x"],
            )
            .contract(cuda, &[&o.a, &o.b, &o.c, &o.e])
            .unwrap()
        };
        let dual = |cuda: &mut dyn Engine| {
            net(
                &[
                    op(&["a", "w"], &["p"]),
                    op(&["a", "x"], &["p"]),
                    op(&["w", "z"], &[]),
                ],
                &["x"],
                &["z"],
            )
            .contract(cuda, &[&o.f, &o.g, &o.h])
            .unwrap()
        };
        type Run<'a, R, D> = (
            &'a str,
            &'a dyn Fn(&mut dyn Engine) -> TensorMap<R, D, CudaStorage<D>>,
        );
        let runs: [Run<'_, R, D>; 2] = [("four-tensor chain", &chain), ("dual legs", &dual)];
        for (name, run) in runs {
            // Call 1 has nothing retained; call 2 is the first to overwrite.
            drop(run(cuda));
            drop(run(cuda));
            let before = device_state(runtime);
            let output = run(cuda);
            let after = device_state(runtime);
            let what = format!("{what} {name}");
            println!(
                "{what}: warm h2d calls {}, bytes {}, device allocations {}",
                after.transfers.h2d_calls - before.transfers.h2d_calls,
                after.transfers.h2d_bytes - before.transfers.h2d_bytes,
                after.transfers.device_allocs - before.transfers.device_allocs,
            );
            assert_eq!(
                after.transfers.d2h_calls, before.transfers.d2h_calls,
                "{what}: a warm call downloads nothing"
            );
            drop(output);
            assert_eq!(after.plans.entries, before.plans.entries, "{what}");
            assert_eq!(
                after.plans.workspaces_created, before.plans.workspaces_created,
                "{what}"
            );
        }
    }

    let engine = &mut symtn::typed::CpuEngine::serial();

    let cuda = &mut cuda_engine();

    let runtime = Runtime::builder().build().unwrap();
    let (v, w, p) = (
        u1_space(&[(-1, 2), (0, 1), (1, 2)]),
        u1_space(&[(0, 2), (1, 1)]),
        u1_space(&[(-1, 1), (0, 2), (1, 1)]),
    );
    warm::<_, f64>(engine, cuda, &runtime, [&v, &w, &p], 795_000, "U(1) f64");
    warm::<_, Complex64>(engine, cuda, &runtime, [&v, &w, &p], 795_100, "U(1) c64");
    let fz2_u1 = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let space = |sectors: &[(bool, i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&fz2_u1),
            sectors.iter().map(|&(odd, charge, degeneracy)| {
                (
                    product_sector(
                        if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                        U1Irrep::new(charge),
                    ),
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let (v, w, p) = (
        space(&[(false, 0, 2), (true, 1, 1), (true, -1, 1)]),
        space(&[(false, 0, 1), (true, 1, 2)]),
        space(&[(true, 0, 1), (false, 1, 2), (true, 1, 1)]),
    );
    warm::<_, f64>(
        engine,
        cuda,
        &runtime,
        [&v, &w, &p],
        795_200,
        "fZ2xU(1) f64",
    );
}

/// A contracted-leg mismatch on a fresh topology (the
/// plan-cache miss path) leaves plan cache, pools, transfer counters,
/// cuTENSOR plans, scratch and executor state unchanged. The topology-hit and
/// alias-hit paths of this and the non-symmetric class are the symtn-network
/// unit test `device_metadata_rejections_leave_the_plan_cache_and_device_untouched_on_every_path`;
/// the compact class, which is not constructible as a device operand
/// of this impl, is the device-free `device_operand_admission` test.
#[test]
#[ignore = "requires a real CUDA device"]
fn rejected_cuda_networks_leave_every_device_state_unchanged() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let space = u1_space(&[(0, 2), (1, 1)]);
    let wrong = u1_space(&[(0, 3)]);
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 796_000)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let mismatched =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&wrong], [&space], 796_001)
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
    // Warm the device context so lazily created state is not attributed to
    // the rejected calls.
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(cuda, &[&a, &a])
            .unwrap(),
    );
    let before = device_state(&runtime);

    // A fresh topology, i.e. the plan-cache miss path only.
    assert!(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
            .contract(cuda, &[&a, &mismatched])
            .is_err()
    );
    assert_eq!(
        device_state(&runtime),
        before,
        "contracted-leg space mismatch"
    );
}

use crate::braiding_probe;
use braiding_probe::{ProbeSector, RealBraidingProbe};

/// A device `Network::contract` contraction on a non-symmetric (unbraided or
/// anyonic) rule is rejected by the network preflight with the typed
/// `contract`'s error — the Host `Network::contract` error — even for the canonical
/// network, before any device work or plan publication.
fn assert_cuda_network_contraction_rejects_non_symmetric<const ANYONIC: bool>() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let leg =
        GradedSpace::try_new(Arc::new(RealBraidingProbe::<ANYONIC>), [(ProbeSector, 2)]).unwrap();
    let host_lhs =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg], [&leg], 1_372_100).unwrap();
    let host_rhs =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg], [&leg], 1_372_101).unwrap();
    let (lhs, rhs) = (
        host_lhs.to_cuda(cuda).unwrap(),
        host_rhs.to_cuda(cuda).unwrap(),
    );
    let host_error = net(&[op(&["a"], &["k"]), op(&["k"], &["b"])], &["a"], &["b"])
        .contract(engine, &[&host_lhs, &host_rhs])
        .unwrap_err();
    let before = device_state(&runtime);
    let device_error = net(&[op(&["a"], &["k"]), op(&["k"], &["b"])], &["a"], &["b"])
        .contract(cuda, &[&lhs, &rhs])
        .unwrap_err();
    assert!(
        matches!(
            &device_error,
            symtn::typed::Error::Operation(operation)
                if matches!(
                    **operation,
                    symtn::typed::OperationError::UnsupportedTensorContractScope {
                        message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
                    }
                )
        ),
        "{device_error:?}"
    );
    assert_eq!(device_error.to_string(), host_error.to_string());
    assert_eq!(device_state(&runtime), before, "ANYONIC = {ANYONIC}");
}

#[test]
#[ignore = "requires a real CUDA device"]
fn non_symmetric_cuda_network_contraction_rejects_like_host_before_device_work() {
    assert_cuda_network_contraction_rejects_non_symmetric::<false>();
    assert_cuda_network_contraction_rejects_non_symmetric::<true>();
}

#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_execution_reuses_a_caller_owned_workspace_and_network_contract_caches() {
    // What: device `execute` takes a caller-owned workspace that
    // repeated calls can pass again with the same result, and
    // `Network::contract` runs device operands through the Runtime's plan
    // cache like Host ones, leasing its pooled workspace on a hit.
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let host = (0..3)
        .map(|seed| {
            TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 60 + seed)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let host_refs = host.iter().collect::<Vec<_>>();
    let device = host
        .iter()
        .map(|tensor| tensor.to_cuda(cuda).unwrap())
        .collect::<Vec<_>>();
    let device_refs = device.iter().collect::<Vec<_>>();
    let network = Network::new(
        vec![vec!["i", "j"], vec!["j", "k"], vec!["k", "m"]],
        vec![false; 3],
        vec![1; 3],
        vec!["i", "m"],
        Some(1),
    )
    .unwrap();
    let expected = network
        .contract(engine, &host_refs)
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();

    let planned = network.plan(&device_refs, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for call in 0..3 {
        let actual = planned
            .execute(cuda, &device_refs, &mut workspace)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        numerics::assert_slices_close(
            &format!("workspace call {call}"),
            actual.dense_data().unwrap(),
            &expected,
            7 * 7,
        );
    }

    clear_plan_cache(&runtime);
    let first = network.contract(cuda, &device_refs).unwrap();
    let second = network.contract(cuda, &device_refs).unwrap();
    let stats = plan_cache_stats(&runtime);
    assert_eq!((stats.misses, stats.hits, stats.entries), (1, 1, 1));
    assert_eq!(stats.workspace_reuses, 1);
    for result in [first, second] {
        numerics::assert_slices_close(
            "Network::contract on device",
            result.to_host(cuda).unwrap().dense_data().unwrap(),
            &expected,
            7 * 7,
        );
    }
}
