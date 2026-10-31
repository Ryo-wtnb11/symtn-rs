use std::sync::Arc;
use symtn::typed::Engine;

use symtn::expert::SectorLeg;
#[cfg(feature = "cuda")]
use symtn::sector::product_sector;
use symtn::sector::FusionRule;
use symtn::sector::{FermionParityFusionRule, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep};
use symtn::typed::Complex64;
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap, TensorScalar};

use super::*;
use crate::optimizer::ContractionStep;
use crate::{plan_cache_stats, GreedyDenseOptimizer, Optimizer};

fn label(name: &str) -> TemporaryLabel {
    TemporaryLabel::from(name)
}

#[expect(
    clippy::type_complexity,
    reason = "the test fixture returns the three network inputs and its independent oracle"
)]
fn rank_four_orientation_inputs<D>(
    engine: &mut dyn Engine,
    runtime: &Runtime,
    provider: &Arc<U1FusionRule>,
    dimensions: [usize; 6],
    bias: f64,
    value: fn(f64, f64) -> D,
) -> (
    TensorMap<U1FusionRule, D>,
    TensorMap<U1FusionRule, D>,
    TensorMap<U1FusionRule, D>,
    TensorMap<U1FusionRule, D>,
)
where
    D: TensorScalar + Copy + std::ops::Add<Output = D> + std::ops::Mul<Output = D>,
{
    let [a_dim, f_dim, c_dim, b_dim, d_dim, e_dim] = dimensions;
    let make_space = |degeneracy| {
        GradedSpace::try_new(Arc::clone(provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let a = make_space(a_dim);
    let f = make_space(f_dim);
    let c = make_space(c_dim);
    let b = make_space(b_dim);
    let d = make_space(d_dim);
    let e = make_space(e_dim);
    let a_dual = a.try_dual().unwrap();
    let f_dual = f.try_dual().unwrap();
    let e_dual = e.try_dual().unwrap();
    let x = TensorMap::from_subblock_fn(engine, runtime, [&a, &f], [&c], |_, ijk| {
        value(
            1.0 + bias + ijk[0] as f64 + 2.0 * ijk[1] as f64 + 3.0 * ijk[2] as f64,
            1.0 + ijk[0] as f64 + ijk[2] as f64,
        )
    })
    .unwrap();
    let y = TensorMap::from_subblock_fn(engine, runtime, [&c], [&b, &d], |_, ijk| {
        value(
            2.0 + 2.0 * bias + 2.0 * ijk[0] as f64 + 3.0 * ijk[1] as f64 + 5.0 * ijk[2] as f64,
            2.0 + ijk[0] as f64 + 2.0 * ijk[1] as f64 + ijk[2] as f64,
        )
    })
    .unwrap();
    let z =
        TensorMap::from_subblock_fn(engine, runtime, [&a_dual, &f_dual, &d], [&e], |_, ijkl| {
            value(
                3.0 + 3.0 * bias
                    + 2.0 * ijkl[0] as f64
                    + 3.0 * ijkl[1] as f64
                    + 5.0 * ijkl[2] as f64
                    + 7.0 * ijkl[3] as f64,
                3.0 + ijkl[0] as f64 + ijkl[1] as f64 + ijkl[2] as f64 + 2.0 * ijkl[3] as f64,
            )
        })
        .unwrap();
    let expected = TensorMap::from_subblock_fn(engine, runtime, [&e_dual], [&b], |_, eb| {
        let mut sum = value(0.0, 0.0);
        for ai in 0..a_dim {
            for fi in 0..f_dim {
                for ci in 0..c_dim {
                    for di in 0..d_dim {
                        let x = value(
                            1.0 + bias + ai as f64 + 2.0 * fi as f64 + 3.0 * ci as f64,
                            1.0 + ai as f64 + ci as f64,
                        );
                        let y = value(
                            2.0 + 2.0 * bias
                                + 2.0 * ci as f64
                                + 3.0 * eb[1] as f64
                                + 5.0 * di as f64,
                            2.0 + ci as f64 + 2.0 * eb[1] as f64 + di as f64,
                        );
                        let z = value(
                            3.0 + 3.0 * bias
                                + 2.0 * ai as f64
                                + 3.0 * fi as f64
                                + 5.0 * di as f64
                                + 7.0 * eb[0] as f64,
                            3.0 + ai as f64 + fi as f64 + di as f64 + 2.0 * eb[0] as f64,
                        );
                        sum = sum + x * y * z;
                    }
                }
            }
        }
        sum
    })
    .unwrap();
    (x, y, z, expected)
}

fn rank_four_orientation_network() -> Network {
    Network::new(
        vec![
            vec![label("a"), label("f"), label("c")],
            vec![label("c"), label("b"), label("d")],
            vec![label("a"), label("f"), label("d"), label("e")],
        ],
        vec![false; 3],
        vec![2, 1, 3],
        vec![label("e"), label("b")],
        Some(1),
    )
    .unwrap()
}

fn assert_rank_four_orientation_replay<D>(
    engine: &mut dyn Engine,
    runtime: &Runtime,
    provider: &Arc<U1FusionRule>,
    make: fn(f64, f64) -> D,
) where
    D: TensorScalar
        + Copy
        + std::ops::Add<Output = D>
        + std::ops::Mul<Output = D>
        + PartialEq
        + std::fmt::Debug,
{
    let first =
        rank_four_orientation_inputs(engine, runtime, provider, [2, 2, 3, 2, 3, 5], 0.0, make);
    let second =
        rank_four_orientation_inputs(engine, runtime, provider, [3, 3, 4, 3, 4, 7], 7.0, make);
    let returned =
        rank_four_orientation_inputs(engine, runtime, provider, [2, 2, 3, 2, 3, 5], 11.0, make);
    assert_eq!(first.3.codomain(), returned.3.codomain());
    assert_eq!(first.3.domain(), returned.3.domain());
    assert_ne!(
        first.3.dense_data().unwrap(),
        returned.3.dense_data().unwrap()
    );

    let network = rank_four_orientation_network();
    let first_refs = [&first.0, &first.1, &first.2];
    let planned = network.plan(&first_refs, &GreedyDenseOptimizer).unwrap();
    assert_eq!(
        (
            planned.plan().steps()[0].lhs(),
            planned.plan().steps()[0].rhs()
        ),
        (TensorId::new(0), TensorId::new(1))
    );
    assert_eq!(
        (
            &planned.schedule.steps[0].codomain[..],
            &planned.schedule.steps[0].domain[..]
        ),
        (&[2][..], &[0, 1, 3][..])
    );

    let cached = Network::new(
        [
            vec!["a", "f", "c"],
            vec!["c", "b", "d"],
            vec!["a", "f", "d", "e"],
        ],
        [false; 3],
        [2, 1, 3],
        ["e", "b"],
        Some(1),
    )
    .unwrap();
    let execute = |engine: &mut dyn Engine, x, y, z| cached.contract(engine, &[x, y, z]).unwrap();
    for (x, y, z, expected) in [
        (&first.0, &first.1, &first.2, &first.3),
        (&second.0, &second.1, &second.2, &second.3),
        (&returned.0, &returned.1, &returned.2, &returned.3),
    ] {
        let actual = execute(engine, x, y, z);
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());

        let refs = [x, y, z];
        let fresh = network
            .plan(&refs, &GreedyDenseOptimizer)
            .unwrap()
            .execute(engine, &refs, &mut Default::default())
            .unwrap();
        assert_eq!(fresh.codomain(), expected.codomain());
        assert_eq!(fresh.domain(), expected.domain());
        assert_eq!(fresh.dense_data().unwrap(), expected.dense_data().unwrap());
    }
    let stats = plan_cache_stats(runtime);
    assert_eq!(
        (stats.misses, stats.hits, stats.replans, stats.entries),
        (1, 2, 0, 1)
    );
    assert_eq!((stats.workspaces_created, stats.workspace_reuses), (1, 2));
}

fn intermediate_payload_snapshot_calls<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    let recording = crate::probe::record();
    let result = operation();
    (
        result,
        recording.get(crate::probe::Event::IntermediatePayloadSnapshot),
    )
}

#[test]
fn ordinary_workspace_replay_skips_payload_meter_snapshots() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let mut make = |shift| {
        TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], move |_, indices| {
            shift + indices[0] as f64 + 2.0 * indices[1] as f64
        })
        .unwrap()
    };
    let tensors = [make(1.0), make(2.0), make(3.0)];
    let network = Network::new(
        vec![
            vec![label("a"), label("x")],
            vec![label("x"), label("y")],
            vec![label("y"), label("c")],
        ],
        vec![false; 3],
        vec![1; 3],
        vec![label("a"), label("c")],
        Some(1),
    )
    .unwrap();
    let refs = tensors.iter().collect::<Vec<_>>();
    let planned = network.plan(&refs, &GreedyDenseOptimizer).unwrap();
    assert!(planned.schedule.steps.len() > 1);
    let mut workspace = NetworkExecutionWorkspace::default();

    let steps = crate::probe::record();
    let (cold, cold_calls) = intermediate_payload_snapshot_calls(|| {
        planned.execute(engine, &refs, &mut workspace).unwrap()
    });
    let (warm, warm_calls) = intermediate_payload_snapshot_calls(|| {
        planned.execute(engine, &refs, &mut workspace).unwrap()
    });

    // The step probe fires on the CPU path too, so the device rejection
    // tests' zero-step assertions are not vacuous.
    assert!(steps.get(crate::probe::Event::ContractStep) > 0);
    assert_eq!(cold_calls, 0);
    assert_eq!(warm_calls, 0);
    assert_eq!(cold.dense_data().unwrap(), &[109.0, 160.0, 169.0, 248.0]);
    assert_eq!(warm.dense_data().unwrap(), &[109.0, 160.0, 169.0, 248.0]);
}

#[test]
fn symmetric_slice_binding_checks_adjoint_orientation_leg_and_rule() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(2), 1)]).unwrap();
    let domain = GradedSpace::try_new(provider, [(U1Irrep::new(-1), 2)]).unwrap();
    let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&codomain],
        [&domain],
        10_280,
    )
    .unwrap();
    let written = vec![label("p"), label("x")];
    let effective = vec![label("x"), label("p")];
    let network = Network::new(
        vec![written],
        vec![true],
        vec![1],
        effective.clone(),
        Some(1),
    )
    .unwrap();
    let ir = NetworkIR::from_labels(vec![effective.clone()], effective.clone()).unwrap();
    let order = ContractionPlan::from_steps(&ir, Vec::new()).unwrap();
    let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![2, 1])]).unwrap();
    let dense = SlicedPlan::new(
        order,
        crate::slice_plan_for(
            &ir,
            &ContractionPlan::from_steps(&ir, Vec::new()).unwrap(),
            &cost,
            &[label("x")],
        ),
    );
    let valid = network
        .lower_symmetric_sliced_plan(&[&tensor], dense)
        .unwrap();
    network
        .bind_symmetric_sliced_plan(&[&tensor], valid.clone())
        .unwrap();

    let index = &valid.slices().indices()[0];
    assert_eq!(index.authority_leg().sectors(), &[U1Irrep::new(-1).into()]);
    assert!(!index.authority_leg().is_dual());
    let forged_leg = SectorLeg::new(index.authority_leg().iter(), true);
    let forged = SymmetricSlicePlan::try_new(
        &ir,
        U1FusionRule.rule_identity(),
        vec![SymmetricSliceSpec::new(
            label("x"),
            index.authority(),
            forged_leg,
            index.pieces().to_vec(),
        )],
    )
    .unwrap();
    assert!(matches!(
        network.bind_symmetric_sliced_plan(
            &[&tensor],
            SymmetricSlicedPlan::new(valid.plan().clone(), forged)
        ),
        Err(SymmetricSliceLowerError::AuthorityLegMismatch { .. })
    ));

    let wrong_rule = SymmetricSlicePlan::try_new(
        &ir,
        Z2FusionRule.rule_identity(),
        vec![SymmetricSliceSpec::new(
            label("x"),
            index.authority(),
            index.authority_leg().clone(),
            index.pieces().to_vec(),
        )],
    )
    .unwrap();
    assert!(matches!(
        network.bind_symmetric_sliced_plan(
            &[&tensor],
            SymmetricSlicedPlan::new(valid.plan().clone(), wrong_rule)
        ),
        Err(SymmetricSliceLowerError::RuleMismatch { .. })
    ));
}

// The sliced-vs-unsliced tests below use small-integer payloads under
// abelian and fermion-parity rules (every coefficient is +-1), so each
// partial sum is an exactly representable integer and every accumulation
// order gives the same bits. Exact equality there checks slice
// selection and scatter without freezing an order.
#[test]
fn internal_symmetric_slices_match_unsliced_multisector_and_meter_cold_warm() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond =
        GradedSpace::try_new(provider, [(U1Irrep::new(0), 1), (U1Irrep::new(1), 3)]).unwrap();
    let mut make = |shift| {
        TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], move |trees, indices| {
            shift
                + trees.codomain_uncoupled()[0].charge() as f64
                + indices[0] as f64
                + 2.0 * indices[1] as f64
        })
        .unwrap()
    };
    let a = make(1.0);
    let b = make(2.0);
    let c = make(3.0);
    let inputs = vec![
        vec![label("a"), label("x")],
        vec![label("x"), label("y")],
        vec![label("y"), label("c")],
    ];
    let output = vec![label("a"), label("c")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 3],
        vec![1; 3],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&a, &b, &c];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![4, 4]),
            DenseTensorInfo::new(vec![4, 4]),
            DenseTensorInfo::new(vec![4, 4]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("x"), label("y")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    assert_eq!(sliced.slices().nslices(), 16);

    let ((cold, cold_stats), snapshot_calls) = intermediate_payload_snapshot_calls(|| {
        network
            .execute_symmetric_sliced(engine, &tensors, sliced.clone(), usize::MAX)
            .unwrap()
    });
    assert!(snapshot_calls > 0);
    let (warm, warm_stats) = network
        .execute_symmetric_sliced(engine, &tensors, sliced.clone(), usize::MAX)
        .unwrap();
    for actual in [&cold, &warm] {
        assert_eq!(actual.subblock_count(), expected.subblock_count());
        for index in 0..actual.subblock_count() {
            assert_eq!(
                actual.subblock(index).unwrap(),
                expected.subblock(index).unwrap()
            );
        }
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }
    assert_eq!(cold_stats, warm_stats);
    assert!(cold_stats.peak_total_bytes() > 0);
    assert_eq!(
        cold_stats.peak_total_bytes(),
        cold_stats
            .destination_bytes()
            .checked_add(cold_stats.peak_workspace_bytes())
            .unwrap()
    );
    assert!(cold_stats.peak_workspace_bytes() > 0);
    network
        .execute_symmetric_sliced(
            engine,
            &tensors,
            sliced.clone(),
            cold_stats.peak_total_bytes(),
        )
        .unwrap();
    assert!(matches!(
        network.execute_symmetric_sliced(
            engine,
            &tensors,
            sliced.clone(),
            cold_stats.peak_total_bytes() - 1,
        ),
        Err(SymmetricSliceExecutionError::WorkspaceLimitExceeded { .. })
    ));

    let mut saw_late_limit = false;
    for ceiling in 0..cold_stats.peak_total_bytes() {
        let jobs = crate::probe::record();
        if matches!(
            network.execute_symmetric_sliced(engine, &tensors, sliced.clone(), ceiling),
            Err(SymmetricSliceExecutionError::WorkspaceLimitExceeded { .. })
        ) && jobs.get(crate::probe::Event::SliceJobCompleted) > 0
        {
            saw_late_limit = true;
            break;
        }
    }
    assert!(
        saw_late_limit,
        "fixture must reject after a completed partial"
    );
}

#[test]
fn output_and_mixed_slices_scatter_by_output_position() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
        (1 + ij[0] + 3 * ij[1]) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
        (2 + 2 * ij[0] + ij[1]) as f64
    })
    .unwrap();
    let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
    // Canonical slice-label order is a,c,x, while final logical output
    // order is c,a. A positional zip would scatter into the wrong axes.
    let output = vec![label("c"), label("a")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![1; 2],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![3, 3]),
            DenseTensorInfo::new(vec![3, 3]),
        ],
    )
    .unwrap();

    for sliced_labels in [vec![label("a")], vec![label("a"), label("c"), label("x")]] {
        let dense = SlicedPlan::new(
            planned.plan().clone(),
            crate::slice_plan_for(&ir, planned.plan(), &cost, &sliced_labels),
        );
        let sliced = network
            .lower_symmetric_sliced_plan(&tensors, dense)
            .unwrap();
        let (actual, stats) = network
            .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
            .unwrap();
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
        assert_eq!(actual.subblock_count(), expected.subblock_count());
        for index in 0..actual.subblock_count() {
            assert_eq!(
                actual.subblock_fusion_trees(index).unwrap(),
                expected.subblock_fusion_trees(index).unwrap()
            );
            assert_eq!(
                actual.subblock(index).unwrap(),
                expected.subblock(index).unwrap()
            );
        }
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert_eq!(
            stats.peak_total_bytes(),
            stats
                .destination_bytes()
                .checked_add(stats.peak_workspace_bytes())
                .unwrap()
        );
    }
}

#[test]
fn mixed_output_slice_keeps_structural_zero_jobs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
        1.0 + trees.codomain_uncoupled()[0].charge() as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
        2.0 + trees.codomain_uncoupled()[0].charge() as f64
    })
    .unwrap();
    let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
    let output = vec![label("a"), label("c")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![1; 2],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("a"), label("x")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    // Two of the four Cartesian (a,x) sector jobs have no lhs block.
    assert_eq!(sliced.slices().nslices(), 4);
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.subblock_count(), expected.subblock_count());
}

#[test]
fn greedy_output_slice_plan_executes_end_to_end() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 6)]).unwrap();
    let unit = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&unit], |_, ij| {
        (1 + ij[0] + ij[1]) as f64
    })
    .unwrap();
    let middle =
        TensorMap::from_subblock_fn(engine, &runtime, [&unit], [&unit], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&unit], [&wide], |_, ij| {
        (3 + ij[0] + ij[1]) as f64
    })
    .unwrap();
    let inputs = vec![
        vec![label("a"), label("b")],
        vec![label("b"), label("c")],
        vec![label("c"), label("d")],
    ];
    let output = vec![label("a"), label("d")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 3],
        vec![1; 3],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&lhs, &middle, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![6, 1]),
            DenseTensorInfo::new(vec![1, 1]),
            DenseTensorInfo::new(vec![1, 6]),
        ],
    )
    .unwrap();
    let decision = crate::greedy_slice(
        &ir,
        planned.plan(),
        &cost,
        6,
        crate::SliceLabels::IncludeOutput,
    );
    assert!(decision.has_output_slices());
    assert!(decision.sliced_width() <= 6);
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, SlicedPlan::new(planned.plan().clone(), decision))
        .unwrap();
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
}

#[test]
fn mixed_slice_accumulator_uses_final_contraction_authority() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let providers = [
        Arc::new(U1FusionRule),
        Arc::new(U1FusionRule),
        Arc::new(U1FusionRule),
    ];
    let legs = providers
        .iter()
        .map(|provider| GradedSpace::try_new(Arc::clone(provider), [(U1Irrep::new(0), 2)]).unwrap())
        .collect::<Vec<_>>();
    let tensors = legs
        .iter()
        .enumerate()
        .map(|(operand, leg)| {
            TensorMap::from_subblock_fn(engine, &runtime, [leg], [leg], move |_, ij| {
                (1 + operand + ij[0] + (operand + 2) * ij[1]) as f64
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let inputs = vec![
        vec![label("a"), label("x")],
        vec![label("y"), label("c")],
        vec![label("x"), label("y")],
    ];
    let output = vec![label("c"), label("a")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 3],
        vec![1; 3],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let ir = NetworkIR::from_labels(inputs.clone(), output.clone()).unwrap();
    let order = ContractionPlan::from_steps(
        &ir,
        vec![
            ContractionStep::new(
                TensorId::new(1),
                TensorId::new(2),
                TensorId::new(3),
                0,
                vec![label("c"), label("x")],
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(0),
                TensorId::new(4),
                0,
                output.clone(),
            ),
        ],
    )
    .unwrap();
    let refs = tensors.iter().collect::<Vec<_>>();
    let planned = network.plan_with(&refs, order).unwrap();
    assert_eq!(
        planned.schedule.steps.last().unwrap().authority_input_slot,
        1
    );
    let expected = planned
        .execute(engine, &refs, &mut Default::default())
        .unwrap();
    assert!(std::ptr::eq(expected.provider(), providers[1].as_ref()));

    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("c"), label("x")]),
    );
    let sliced = network.lower_symmetric_sliced_plan(&refs, dense).unwrap();
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &refs, sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert!(std::ptr::eq(actual.provider(), providers[1].as_ref()));
    assert!(!std::ptr::eq(actual.provider(), providers[0].as_ref()));
}

#[test]
fn mixed_symmetric_slice_legal_empty_returns_unsliced_zero_layout() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let open = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let empty = GradedSpace::try_new(provider, [(U1Irrep::new(0), 0)]).unwrap();
    let a = TensorMap::<_, f64>::zeros(engine, &runtime, [&open], [&empty]).unwrap();
    let b = TensorMap::<_, f64>::zeros(engine, &runtime, [&empty], [&open]).unwrap();
    let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
    let output = vec![label("a"), label("c")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![1; 2],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&a, &b];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let effective = typed_effective_spaces(&a, false).unwrap();
    let output_authority = ir.edge(&label("a")).unwrap().occurrences()[0];
    let output_leg = __network::network_sector_leg(&effective[output_authority.axis()]).clone();
    let output_sector = output_leg.sectors()[0];
    let authority = ir.edge(&label("x")).unwrap().occurrences()[0];
    let leg = __network::network_sector_leg(&effective[authority.axis()]).clone();
    let slices = SymmetricSlicePlan::try_new(
        &ir,
        U1FusionRule.rule_identity(),
        vec![
            SymmetricSliceSpec::new(
                label("a"),
                output_authority,
                output_leg,
                vec![
                    crate::SectorSlice::new(
                        output_sector,
                        crate::DegeneracyRange::new(0, 1).unwrap(),
                    ),
                    crate::SectorSlice::new(
                        output_sector,
                        crate::DegeneracyRange::new(1, 2).unwrap(),
                    ),
                ],
            ),
            SymmetricSliceSpec::new(label("x"), authority, leg, Vec::new()),
        ],
    )
    .unwrap();
    assert_eq!(slices.nslices(), 0);
    let (actual, stats) = network
        .execute_symmetric_sliced(
            engine,
            &tensors,
            SymmetricSlicedPlan::new(planned.plan().clone(), slices),
            usize::MAX,
        )
        .unwrap();
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(
        stats.destination_bytes(),
        __network::network_owned_payload(&actual).unwrap().1
    );
    assert_eq!(stats.peak_workspace_bytes(), 0);
    assert_eq!(stats.peak_total_bytes(), stats.destination_bytes());
}

#[test]
fn internal_slice_maps_nonselfdual_partner_from_authority_leg() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = GradedSpace::try_new(provider, [(U1Irrep::new(1), 2)]).unwrap();
    let a = TensorMap::from_subblock_fn(engine, &runtime, [&plus], [&plus], |_, ij| {
        (1 + ij[0] + 2 * ij[1]) as f64
    })
    .unwrap();
    let b = TensorMap::from_subblock_fn(engine, &runtime, [&plus], [&plus], |_, ij| {
        (2 + 2 * ij[0] + ij[1]) as f64
    })
    .unwrap();
    let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
    let output = vec![label("a"), label("c")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![1; 2],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&a, &b];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("x")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    let authority = &sliced.slices().indices()[0];
    assert_eq!(
        authority.authority_leg().sectors(),
        &[U1Irrep::new(-1).into()]
    );
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
}

#[test]
fn output_slice_scatter_uses_domain_side_nonselfdual_logical_leg() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = GradedSpace::try_new(provider, [(U1Irrep::new(1), 3)]).unwrap();
    let tensor = TensorMap::from_subblock_fn(engine, &runtime, [&plus], [&plus], |_, ij| {
        (1 + ij[0] + 3 * ij[1]) as f64
    })
    .unwrap();
    let labels = vec![label("a"), label("b")];
    let network = Network::new(
        vec![labels.clone()],
        vec![false],
        vec![1],
        labels.clone(),
        Some(1),
    )
    .unwrap();
    let planned = network.plan(&[&tensor], &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &[&tensor], &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(vec![labels.clone()], labels).unwrap();
    let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![3, 3])]).unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("b")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&[&tensor], dense)
        .unwrap();
    let index = &sliced.slices().indices()[0];
    assert_eq!(index.output_position(), Some(1));
    assert_eq!(index.authority_leg().sectors(), &[U1Irrep::new(-1).into()]);
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &[&tensor], sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
}

#[test]
fn zero_step_conjugated_output_slice_scatter_reads_lazy_adjoint() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = U1Irrep::new(1);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(plus, 2)]).unwrap();
    let columns = GradedSpace::try_new(provider, [(plus, 3)]).unwrap();
    let tensor = TensorMap::from_subblock_fn(engine, &runtime, [&rows], [&columns], |_, ij| {
        Complex64::new((ij[0] + 2 * ij[1]) as f64, (1 + ij[0] + ij[1]) as f64)
    })
    .unwrap();
    let written = vec![label("p"), label("q")];
    let effective = vec![label("q"), label("p")];
    let network = Network::new(
        vec![written],
        vec![true],
        vec![1],
        effective.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&tensor];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    assert!(planned.plan().steps().is_empty());
    assert!(planned.schedule.final_permutation.is_none());

    let ir = NetworkIR::from_labels(vec![effective.clone()], effective).unwrap();
    let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![3, 2])]).unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("q")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    assert_eq!(sliced.slices().indices()[0].output_position(), Some(0));
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    let expected = (0..2)
        .flat_map(|column| {
            (0..3).map(move |row| {
                Complex64::new((column + 2 * row) as f64, -((1 + column + row) as f64))
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.dense_data().unwrap(), expected);
}

#[test]
fn compact_input_preflight_applies_to_output_slices() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let compact = TensorMap::<_, f64>::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0, 1.0],
        }],
    )
    .unwrap();
    assert!(__network::network_has_compact_payload(&compact));
    let labels = vec![label("a"), label("b")];
    let network = Network::new(
        vec![labels.clone()],
        vec![false],
        vec![1],
        labels.clone(),
        Some(1),
    )
    .unwrap();
    let ir = NetworkIR::from_labels(vec![labels.clone()], labels.clone()).unwrap();
    let plan = ContractionPlan::from_steps(&ir, Vec::new()).unwrap();
    let authority = ir.edge(&label("a")).unwrap().occurrences()[0];
    let effective = typed_effective_spaces(&compact, false).unwrap();
    let authority_leg = __network::network_sector_leg(&effective[authority.axis()]).clone();
    let sector = authority_leg.sectors()[0];
    let slices = SymmetricSlicePlan::try_new(
        &ir,
        U1FusionRule.rule_identity(),
        vec![SymmetricSliceSpec::new(
            label("a"),
            authority,
            authority_leg,
            vec![crate::SectorSlice::new(
                sector,
                crate::DegeneracyRange::new(0, 2).unwrap(),
            )],
        )],
    )
    .unwrap();
    assert!(matches!(
        network.execute_symmetric_sliced(
            engine,
            &[&compact],
            SymmetricSlicedPlan::new(plan, slices),
            usize::MAX,
        ),
        Err(SymmetricSliceExecutionError::Tensor(_))
    ));
    let empty = SymmetricSlicePlan::try_new(&ir, U1FusionRule.rule_identity(), Vec::new()).unwrap();
    assert!(matches!(
        network.execute_symmetric_sliced(
            engine,
            &[&compact],
            SymmetricSlicedPlan::new(ContractionPlan::from_steps(&ir, Vec::new()).unwrap(), empty,),
            usize::MAX,
        ),
        Err(SymmetricSliceExecutionError::Tensor(_))
    ));
}

fn assert_fermionic_mixed_slice<D>(engine: &mut dyn Engine)
where
    D: TensorScalar + PartialEq + std::fmt::Debug,
{
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = GradedSpace::try_new(provider, [(Z2Irrep::ODD, 2)]).unwrap();
    let a = TensorMap::from_subblock_fn(engine, &runtime, [&odd], [&odd], |_, ij| {
        D::from_real((1 + ij[0] + 2 * ij[1]) as f64)
    })
    .unwrap();
    let b = TensorMap::from_subblock_fn(engine, &runtime, [&odd], [&odd], |_, ij| {
        D::from_real((2 + 2 * ij[0] + ij[1]) as f64)
    })
    .unwrap();
    let inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
    let output = vec![label("a"), label("c")];
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![1; 2],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&a, &b];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("a"), label("x")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
}

#[test]
fn fermionic_mixed_slices_match_unsliced_real_and_complex() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    assert_fermionic_mixed_slice::<f64>(engine);
    assert_fermionic_mixed_slice::<Complex64>(engine);
}

#[test]
fn fermionic_complex_conjugated_operand_mixed_slice_matches_unsliced() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = GradedSpace::try_new(provider, [(Z2Irrep::ODD, 2)]).unwrap();
    let a = TensorMap::from_subblock_fn(engine, &runtime, [&odd], [&odd], |_, ij| {
        Complex64::new(
            (1 + ij[0] + 2 * ij[1]) as f64,
            (2 + 3 * ij[0] + ij[1]) as f64,
        )
    })
    .unwrap();
    let b = TensorMap::from_subblock_fn(engine, &runtime, [&odd], [&odd], |_, ij| {
        Complex64::new(
            (2 + 2 * ij[0] + ij[1]) as f64,
            -((1 + ij[0] + 4 * ij[1]) as f64),
        )
    })
    .unwrap();

    // The first operand is written as [x; a]. Network lowering rotates it
    // to the effective adjoint order [a; x]. Both the output a and the
    // contracted x are sliced. Nonzero imaginary parts make a missed
    // conjugation observable.
    let written_inputs = vec![vec![label("x"), label("a")], vec![label("x"), label("c")]];
    let effective_inputs = vec![vec![label("a"), label("x")], vec![label("x"), label("c")]];
    let output = vec![label("a"), label("c")];
    let network = Network::new(
        written_inputs,
        vec![true, false],
        vec![1; 2],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let tensors = [&a, &b];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(effective_inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        crate::slice_plan_for(&ir, planned.plan(), &cost, &[label("a"), label("x")]),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    let (actual, _) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));
}

fn crossed_plan() -> PlannedNetwork {
    let inputs = vec![
        vec![label("a"), label("b")],
        vec![label("b"), label("c")],
        vec![label("a"), label("d")],
    ];
    let output = vec![label("c"), label("d")];
    let ir = NetworkIR::from_labels(inputs.clone(), output.clone()).unwrap();
    let plan = ContractionPlan::from_steps(
        &ir,
        vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                vec![label("a"), label("c")],
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                vec![label("c"), label("d")],
            ),
        ],
    )
    .unwrap();
    let schedule = compile_schedule(&ir, &plan, Some(1), &[1, 1, 1]).unwrap();
    assert_eq!(
        (
            &schedule.steps[0].codomain[..],
            &schedule.steps[0].domain[..]
        ),
        (&[1][..], &[0][..])
    );
    PlannedNetwork {
        owner_token: NEXT_PLAN_OWNER_TOKEN.fetch_add(1, Ordering::Relaxed),
        plan,
        conj: vec![false; 3],
        input_codomain_ranks: vec![1; 3],
        schedule,
    }
}

#[test]
fn device_operand_admission_decides_each_class_from_metadata() {
    use symtn::sector::BraidingStyleKind;
    let dense = [
        NetworkReuseClass::OwnedDense,
        NetworkReuseClass::LazyAdjoint,
    ];
    for braiding in [BraidingStyleKind::Bosonic, BraidingStyleKind::Fermionic] {
        // What: symmetric braiding, fermionic included since the device
        // contraction carries the core-right twist, is admitted with or
        // without contraction steps.
        for contracts in [false, true] {
            assert!(device_operand_admission(contracts, braiding, dense).is_ok());
        }
    }
    // What: non-symmetric braiding (#1372: unbraided as well as anyonic)
    // is the contraction's boundary only, with the typed error.
    for braiding in [BraidingStyleKind::NoBraiding, BraidingStyleKind::Anyonic] {
        assert!(device_operand_admission(false, braiding, dense).is_ok());
        match device_operand_admission(true, braiding, dense) {
            Err(Error::Operation(error)) => assert!(matches!(
                error.as_ref(),
                OperationError::UnsupportedTensorContractScope {
                    message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
                }
            )),
            other => panic!("{braiding:?} contraction must be unsupported, got {other:?}"),
        }
    }
    // What: a compact operand is rejected whatever the schedule, and
    // before the braiding check.
    for (contracts, braiding) in [
        (false, BraidingStyleKind::Bosonic),
        (true, BraidingStyleKind::Anyonic),
    ] {
        assert!(matches!(
            device_operand_admission(
                contracts,
                braiding,
                [NetworkReuseClass::OwnedDense, NetworkReuseClass::Compact],
            ),
            Err(Error::UnsupportedOnDevice(_))
        ));
    }
}

#[test]
fn host_diagonal_operands_classify_as_compact_for_device_admission() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    // Why: the device preflight reads only `network_reuse_class`; this
    // pins that a diagonal payload reports `Compact` through it.
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let dense = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&space],
        [&space],
        748_200,
    )
    .unwrap();
    let symtn::typed::Svd { s: compact, .. } = dense.svd_compact(engine, &[0], &[1]).unwrap();
    assert!(device_operand_admission(
        true,
        symtn::sector::FusionRule::braiding_style(dense.provider()),
        [
            __network::network_reuse_class(&dense, false),
            __network::network_reuse_class(&dense, true)
        ],
    )
    .is_ok());
    assert!(matches!(
        device_operand_admission(
            true,
            symtn::sector::FusionRule::braiding_style(compact.provider()),
            [__network::network_reuse_class(&compact, false)],
        ),
        Err(Error::UnsupportedOnDevice(_))
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn device_network_admission_refuses_a_cpu_engine_before_any_operand_check() {
    // What: a device network runs on the caller's CUDA engine; a CPU engine
    // is a typed refusal decided from the engine alone, before operand
    // metadata or any step. No GPU is needed.
    let cpu = &mut symtn::typed::CpuEngine::serial();
    let error = cuda_operand_admission::<U1FusionRule, f64>(cpu, &[], true).unwrap_err();
    let mut chain = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        chain.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    assert!(
        chain.contains("unsupported") && chain.contains("CUDA engine"),
        "{chain}"
    );
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn assert_asymmetric_cuda_plan_parity<R>(
    engine: &mut dyn Engine,
    cuda: &mut dyn Engine,
    runtime: &Runtime,
    x0: &GradedSpace<R>,
    x1: &GradedSpace<R>,
    y: &GradedSpace<R>,
    z0: &GradedSpace<R>,
    z1: &GradedSpace<R>,
    seed: u64,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let a = TensorMap::<R, f64>::rand_with_seed(engine, runtime, [x0, x1], [y], seed).unwrap();
    let b = TensorMap::rand_with_seed(engine, runtime, [x1], [z0, z1], seed + 1).unwrap();
    let a_cuda = a.to_cuda(cuda).unwrap();
    let b_cuda = b.to_cuda(cuda).unwrap();
    let network = Network::new(
        vec![
            vec![label("i0"), label("i1"), label("j")],
            vec![label("i1"), label("k0"), label("k1")],
        ],
        vec![true, false],
        vec![2, 1],
        vec![label("j"), label("i0"), label("k0"), label("k1")],
        Some(2),
    )
    .unwrap();
    let host_refs = [&a, &b];
    let cuda_refs = [&a_cuda, &b_cuda];
    let host = network.plan(&host_refs, &GreedyDenseOptimizer).unwrap();
    let cuda = network.plan(&cuda_refs, &GreedyDenseOptimizer).unwrap();

    assert_eq!(host.plan.steps(), cuda.plan.steps());
    assert_eq!(host.schedule.input_ranks, cuda.schedule.input_ranks);
    assert_eq!(
        host.schedule.contracted_input_pairs,
        cuda.schedule.contracted_input_pairs
    );
    for (host_step, cuda_step) in host.schedule.steps.iter().zip(&cuda.schedule.steps) {
        assert_eq!(host_step.lhs_contract_axes, cuda_step.lhs_contract_axes);
        assert_eq!(host_step.rhs_contract_axes, cuda_step.rhs_contract_axes);
        assert_eq!(host_step.codomain, cuda_step.codomain);
        assert_eq!(host_step.domain, cuda_step.domain);
    }
    assert_eq!(
        host.schedule.final_permutation,
        cuda.schedule.final_permutation
    );
    for ((host_tensor, cuda_tensor), conj) in [(&a, &a_cuda), (&b, &b_cuda)]
        .into_iter()
        .zip([true, false])
    {
        assert_eq!(
            typed_effective_spaces(host_tensor, conj).unwrap(),
            typed_effective_spaces(cuda_tensor, conj).unwrap()
        );
        assert_eq!(
            host_tensor.leg_dims().unwrap(),
            cuda_tensor.leg_dims().unwrap()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_network_plans_match_host_and_reject_before_the_first_contract() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda =
        &mut symtn::typed::CudaEngine::open(symtn::typed::CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    let u1_rule = Arc::new(U1FusionRule);
    let u1_x0 = GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(2), 2)]).unwrap();
    let u1_x1 = GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(-1), 1)])
        .unwrap()
        .try_dual()
        .unwrap();
    let u1_y = GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(1), 3)]).unwrap();
    let u1_z0 = GradedSpace::try_new(Arc::clone(&u1_rule), [(U1Irrep::new(-2), 2)]).unwrap();
    let u1_z1 = GradedSpace::try_new(u1_rule, [(U1Irrep::new(0), 1)]).unwrap();
    assert_asymmetric_cuda_plan_parity(
        engine, cuda, &runtime, &u1_x0, &u1_x1, &u1_y, &u1_z0, &u1_z1, 748_210,
    );

    let product_rule = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let product_x0 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1)],
    )
    .unwrap();
    let product_x1 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let product_y = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2)],
    )
    .unwrap();
    let product_z0 = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [(product_sector(Z2Irrep::EVEN, U1Irrep::new(-2)), 1)],
    )
    .unwrap();
    let product_z1 = GradedSpace::try_new(
        product_rule,
        [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1)],
    )
    .unwrap();
    assert_asymmetric_cuda_plan_parity(
        engine,
        cuda,
        &runtime,
        &product_x0,
        &product_x1,
        &product_y,
        &product_z0,
        &product_z1,
        748_212,
    );

    let provider = Arc::new(U1FusionRule);
    let good = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let bad = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(1), 2)]).unwrap();
    let host_tensors = (0..3)
        .map(|seed| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(
                engine,
                &runtime,
                [&good],
                [&good],
                748_220 + seed,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let tensors = host_tensors
        .iter()
        .map(|tensor| tensor.to_cuda(cuda).unwrap())
        .collect::<Vec<_>>();
    let labels = |names: &[&str]| names.iter().copied().map(label).collect::<Vec<_>>();
    let network = Network::new(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
        ],
        vec![false; 3],
        vec![1; 3],
        labels(&["a", "d"]),
        Some(1),
    )
    .unwrap();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let ir = NetworkIR::from_labels(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
        ],
        labels(&["a", "d"]),
    )
    .unwrap();
    let canonical_order = || {
        ContractionPlan::from_steps(
            &ir,
            vec![
                ContractionStep::new(
                    TensorId::new(0),
                    TensorId::new(1),
                    TensorId::new(3),
                    0,
                    labels(&["a", "c"]),
                ),
                ContractionStep::new(
                    TensorId::new(3),
                    TensorId::new(2),
                    TensorId::new(4),
                    0,
                    labels(&["a", "d"]),
                ),
            ],
        )
        .unwrap()
    };
    let host_refs = [&host_tensors[0], &host_tensors[1], &host_tensors[2]];
    let host_plan = network.plan_with(&host_refs, canonical_order()).unwrap();
    let cuda_plan = network.plan_with(&refs, canonical_order()).unwrap();
    assert_eq!(host_plan.plan.steps(), cuda_plan.plan.steps());
    assert_eq!(
        host_plan.schedule.input_ranks,
        cuda_plan.schedule.input_ranks
    );
    assert_eq!(
        host_plan.schedule.contracted_input_pairs,
        cuda_plan.schedule.contracted_input_pairs
    );
    for (host, cuda) in host_plan
        .schedule
        .steps
        .iter()
        .zip(&cuda_plan.schedule.steps)
    {
        assert_eq!(host.lhs_contract_axes, cuda.lhs_contract_axes);
        assert_eq!(host.rhs_contract_axes, cuda.rhs_contract_axes);
        assert_eq!(host.codomain, cuda.codomain);
        assert_eq!(host.domain, cuda.domain);
    }
    assert_eq!(
        host_plan.schedule.final_permutation,
        cuda_plan.schedule.final_permutation
    );
    assert_eq!(
        typed_effective_spaces(&host_tensors[0], true).unwrap(),
        typed_effective_spaces(&tensors[0], true).unwrap()
    );
    assert_eq!(
        host_tensors[0].leg_dims().unwrap(),
        tensors[0].leg_dims().unwrap()
    );
    // A schedule the canonical predicate refused — a split-changing step
    // orientation, a non-identity pAB, a final permutation — is an ordinary
    // device schedule now: each edit runs the Host step sequence on the
    // device and equals the Host run of the same edited plan.
    type Edit = fn(&mut CompiledSchedule);
    let edits: [(&str, Edit); 3] = [
        ("split-changing step orientation", |schedule| {
            (schedule.steps[1].codomain, schedule.steps[1].domain) = (vec![0, 1], vec![]);
        }),
        ("non-identity pAB", |schedule| {
            (schedule.steps[1].codomain, schedule.steps[1].domain) = (vec![1], vec![0]);
        }),
        ("final permutation", |schedule| {
            schedule.final_permutation = Some((vec![1], vec![0]));
        }),
    ];
    for (what, edit) in edits {
        let mut host_edited = network.plan_with(&host_refs, canonical_order()).unwrap();
        edit(&mut host_edited.schedule);
        let mut cuda_edited = network.plan_with(&refs, canonical_order()).unwrap();
        edit(&mut cuda_edited.schedule);
        let host = host_edited
            .execute(engine, &host_refs, &mut Default::default())
            .unwrap();
        let device = cuda_edited
            .execute(cuda, &refs, &mut NetworkExecutionWorkspace::default())
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_eq!(device.codomain(), host.codomain(), "{what}");
        assert_eq!(device.domain(), host.domain(), "{what}");
        assert_eq!(
            device.dense_data().unwrap().len(),
            host.dense_data().unwrap().len(),
            "{what}"
        );
        for (&got, &want) in device
            .dense_data()
            .unwrap()
            .iter()
            .zip(host.dense_data().unwrap())
        {
            assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "{what}");
        }
    }

    let bad_rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&bad], [&good], 748_230)
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
    let pair = Network::new(
        vec![labels(&["a", "b"]), labels(&["b", "c"])],
        vec![false; 2],
        vec![1; 2],
        labels(&["a", "c"]),
        Some(1),
    )
    .unwrap();
    let planned = pair
        .plan(&[&tensors[0], &tensors[1]], &GreedyDenseOptimizer)
        .unwrap();
    let steps = crate::probe::record();
    assert!(planned
        .execute(
            cuda,
            &[&tensors[0], &bad_rhs],
            &mut NetworkExecutionWorkspace::default()
        )
        .is_err());
    assert_eq!(steps.get(crate::probe::Event::ContractStep), 0);

    let other_runtime = Runtime::builder().build().unwrap();
    let other_space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let foreign = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &other_runtime,
        [&other_space],
        [&other_space],
        748_231,
    )
    .unwrap()
    .to_cuda(cuda)
    .unwrap();
    let steps = crate::probe::record();
    assert!(planned
        .execute(
            cuda,
            &[&tensors[0], &foreign],
            &mut NetworkExecutionWorkspace::default()
        )
        .is_err());
    assert_eq!(steps.get(crate::probe::Event::ContractStep), 0);

    // A second engine on the same device is another allocation domain: its
    // buffer is refused before the first step, and so is a CPU engine.
    let other_engine =
        &mut symtn::typed::CudaEngine::open(symtn::typed::CudaDeviceId::from_ordinal(0)).unwrap();
    let other_domain = host_tensors[1].to_cuda(other_engine).unwrap();
    for (what, engine) in [
        ("foreign engine operand", &mut *cuda as &mut dyn Engine),
        ("CPU engine", engine as &mut dyn Engine),
    ] {
        let steps = crate::probe::record();
        let operands = if what == "CPU engine" {
            [&tensors[0], &tensors[1]]
        } else {
            [&tensors[0], &other_domain]
        };
        assert!(
            planned
                .execute(engine, &operands, &mut NetworkExecutionWorkspace::default())
                .is_err(),
            "{what}"
        );
        assert_eq!(steps.get(crate::probe::Event::ContractStep), 0, "{what}");
    }
}

#[test]
fn typed_crossed_schedule_reuses_the_actual_first_step_destination() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let other_provider = Arc::new(U1FusionRule);
    let space = |provider: &Arc<U1FusionRule>, degeneracy| {
        GradedSpace::try_new(Arc::clone(provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let left = space(&provider, 9);
    let bond = space(&provider, 8);
    let right = space(&provider, 10);
    let tail = space(&provider, 11);
    let left_dual = left.try_dual().unwrap();
    let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&left], [&bond], 1)
        .unwrap();
    let b = TensorMap::rand_with_seed(engine, &runtime, [&bond], [&right], 2).unwrap();
    let c = TensorMap::rand_with_seed(engine, &runtime, [&left_dual], [&tail], 3).unwrap();
    let planned = crossed_plan();
    let mut workspace = NetworkExecutionWorkspace::default();
    let refs = [&a, &b, &c];
    let oracle = a
        .contract(
            engine,
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap()
        .contract(
            engine,
            &c,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    drop(planned.execute(engine, &refs, &mut workspace).unwrap());
    let before = workspace.intermediates[0]
        .output
        .as_ref()
        .map(|tensor| {
            (
                tensor.dense_data().unwrap().as_ptr(),
                tensor.dense_data().unwrap().len(),
            )
        })
        .unwrap();
    let output = planned.execute(engine, &refs, &mut workspace).unwrap();
    let after = workspace.intermediates[0]
        .output
        .as_ref()
        .map(|tensor| {
            (
                tensor.dense_data().unwrap().as_ptr(),
                tensor.dense_data().unwrap().len(),
            )
        })
        .unwrap();
    assert_eq!(before, after);
    assert_ne!(after.0, output.dense_data().unwrap().as_ptr());
    // The crossed schedule groups the bond-8 and left-9 contractions
    // differently from the chained direct calls: 8 * 9 terms per entry.
    crate::test_numerics::numerics::assert_slices_close(
        "crossed schedule vs chained contract",
        output.dense_data().unwrap(),
        oracle.dense_data().unwrap(),
        8 * 9,
    );

    let other_bond = space(&other_provider, 8);
    let other_right = space(&other_provider, 10);
    let rhs_drift =
        TensorMap::rand_with_seed(engine, &runtime, [&other_bond], [&other_right], 4).unwrap();
    drop(
        planned
            .execute(engine, &[&a, &rhs_drift, &c], &mut workspace)
            .unwrap(),
    );
    let rhs_only = workspace.intermediates[0].output.as_ref().unwrap();
    assert_eq!(
        (
            rhs_only.dense_data().unwrap().as_ptr(),
            rhs_only.dense_data().unwrap().len()
        ),
        before
    );
    assert!(std::ptr::eq(rhs_only.provider(), provider.as_ref()));

    let retained = rhs_only.clone();
    assert!(planned.execute(engine, &refs[..2], &mut workspace).is_err());
    assert_eq!(
        workspace.intermediates[0]
            .output
            .as_ref()
            .unwrap()
            .dense_data()
            .unwrap()
            .as_ptr(),
        retained.dense_data().unwrap().as_ptr()
    );
    let bad_bond = space(&provider, 7);
    let bad_rhs = TensorMap::rand_with_seed(engine, &runtime, [&bad_bond], [&right], 5).unwrap();
    assert!(planned
        .execute(engine, &[&a, &bad_rhs, &c], &mut workspace)
        .is_err());
    assert_eq!(
        workspace.intermediates[0]
            .output
            .as_ref()
            .unwrap()
            .dense_data()
            .unwrap()
            .as_ptr(),
        retained.dense_data().unwrap().as_ptr()
    );

    let other_left = space(&other_provider, 9);
    let lhs_drift =
        TensorMap::rand_with_seed(engine, &runtime, [&other_left], [&other_bond], 6).unwrap();
    drop(
        planned
            .execute(engine, &[&lhs_drift, &b, &c], &mut workspace)
            .unwrap(),
    );
    let replaced = workspace.intermediates[0].output.as_ref().unwrap();
    assert_ne!(
        replaced.dense_data().unwrap().as_ptr(),
        retained.dense_data().unwrap().as_ptr()
    );
    assert!(std::ptr::eq(replaced.provider(), other_provider.as_ref()));

    let replaced = replaced.clone();
    let wide_left = space(&other_provider, 12);
    let other_tail = space(&other_provider, 11);
    let wide_c = TensorMap::rand_with_seed(
        engine,
        &runtime,
        [&wide_left.try_dual().unwrap()],
        [&other_tail],
        7,
    )
    .unwrap();
    let wide_a =
        TensorMap::rand_with_seed(engine, &runtime, [&wide_left], [&other_bond], 8).unwrap();
    drop(
        planned
            .execute(engine, &[&wide_a, &rhs_drift, &wide_c], &mut workspace)
            .unwrap(),
    );
    let widened = workspace.intermediates[0].output.as_ref().unwrap();
    assert_ne!(
        widened.dense_data().unwrap().len(),
        replaced.dense_data().unwrap().len()
    );

    let widened = widened.clone();
    let other_runtime = Runtime::builder().build().unwrap();
    let foreign_a =
        TensorMap::rand_with_seed(engine, &other_runtime, [&other_left], [&other_bond], 9).unwrap();
    let foreign_b =
        TensorMap::rand_with_seed(engine, &other_runtime, [&other_bond], [&other_right], 10)
            .unwrap();
    let foreign_c = TensorMap::rand_with_seed(
        engine,
        &other_runtime,
        [&other_left.try_dual().unwrap()],
        [&other_tail],
        11,
    )
    .unwrap();
    drop(
        planned
            .execute(
                engine,
                &[&foreign_a, &foreign_b, &foreign_c],
                &mut workspace,
            )
            .unwrap(),
    );
    assert_ne!(
        workspace.intermediates[0]
            .output
            .as_ref()
            .unwrap()
            .dense_data()
            .unwrap()
            .as_ptr(),
        widened.dense_data().unwrap().as_ptr()
    );

    let previous = workspace.intermediates[0].output.as_ref().unwrap().clone();
    let other_plan = crossed_plan();
    drop(
        other_plan
            .execute(
                engine,
                &[&foreign_a, &foreign_b, &foreign_c],
                &mut workspace,
            )
            .unwrap(),
    );
    assert_eq!(workspace.owner_token, Some(other_plan.owner_token));
    assert_ne!(
        workspace.intermediates[0]
            .output
            .as_ref()
            .unwrap()
            .dense_data()
            .unwrap()
            .as_ptr(),
        previous.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn typed_replay_restores_buffers_after_injected_failures() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let left = space(5);
    let bond = space(4);
    let right = space(6);
    let tail = space(7);
    let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&left], [&bond], 11)
        .unwrap();
    let b = TensorMap::rand_with_seed(engine, &runtime, [&bond], [&right], 12).unwrap();
    let c = TensorMap::rand_with_seed(engine, &runtime, [&left.try_dual().unwrap()], [&tail], 13)
        .unwrap();
    let refs = [&a, &b, &c];
    let mut planned = crossed_plan();
    let mut workspace = NetworkExecutionWorkspace::default();
    drop(planned.execute(engine, &refs, &mut workspace).unwrap());

    // A failure inside the step that owns the retained destination keeps
    // that exact allocation.
    let retained = |workspace: &NetworkExecutionWorkspace<U1FusionRule, f64>| {
        workspace.intermediates[0]
            .output
            .as_ref()
            .unwrap()
            .dense_data()
            .unwrap()
            .as_ptr()
    };
    let before = retained(&workspace);
    let codomain = std::mem::replace(&mut planned.schedule.steps[0].codomain, vec![usize::MAX]);
    assert!(planned.execute(engine, &refs, &mut workspace).is_err());
    assert_eq!(retained(&workspace), before);
    planned.schedule.steps[0].codomain = codomain;
    drop(planned.execute(engine, &refs, &mut workspace).unwrap());
    assert_eq!(retained(&workspace), before);

    let rhs_axes = std::mem::replace(
        &mut planned.schedule.steps[1].rhs_contract_axes,
        vec![usize::MAX],
    );
    assert!(planned.execute(engine, &refs, &mut workspace).is_err());
    assert!(workspace.slots[planned.schedule.steps[0].result_slot].is_some());
    planned.schedule.steps[1].rhs_contract_axes = rhs_axes;
    drop(planned.execute(engine, &refs, &mut workspace).unwrap());

    planned.schedule.final_permutation = Some((vec![usize::MAX], vec![0]));
    assert!(planned.execute(engine, &refs, &mut workspace).is_err());
    assert!(workspace.slots[planned.schedule.final_slot].is_some());
    planned.schedule.final_permutation = None;
    drop(planned.execute(engine, &refs, &mut workspace).unwrap());
}

#[test]
fn typed_split_moving_step_is_one_contract_equal_to_contract_then_permute() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let a =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 31)
            .unwrap();
    let b = TensorMap::rand_with_seed(engine, &runtime, [&space], [&space, &space], 32).unwrap();
    let c = TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 33).unwrap();
    let network = Network::new(
        vec![
            vec![label("a"), label("c")],
            vec![label("c"), label("b"), label("d")],
            vec![label("b"), label("d"), label("e")],
        ],
        vec![false; 3],
        vec![1, 1, 2],
        vec![label("e"), label("a")],
        Some(1),
    )
    .unwrap();
    let refs = [&a, &b, &c];
    let planned = network
        .plan(
            &refs,
            &crate::LabelOrderDenseOptimizer::new(vec![label("c"), label("b"), label("d")]),
        )
        .unwrap();
    // Step 0 contracts `c` of a[a; c] and b[c; b, d] into (a | b, d); its
    // consumer takes it as rhs against c[b, d; e], so the next-use
    // orientation is (b, d | a) — a split move, the contract's own spec.
    let step = &planned.schedule.steps[0];
    assert_eq!((step.lhs_slot, step.rhs_slot), (0, 1));
    assert_eq!(
        (&step.codomain[..], &step.domain[..]),
        (&[1, 2][..], &[0][..])
    );
    assert!(planned.schedule.final_permutation.is_none());
    // The pre-#1614 lowering, spelled out: default-split contract, then the
    // orientation permute, per step.
    let ab = a
        .contract(
            engine,
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1, 2],
            },
        )
        .unwrap()
        .permute(engine, &[1, 2], &[0])
        .unwrap();
    let oracle = c
        .contract(
            engine,
            &ab,
            &ContractSpec {
                lhs: &[0, 1],
                rhs: &[0, 1],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..2 {
        let output = planned.execute(engine, &refs, &mut workspace).unwrap();
        assert_eq!(output.codomain(), oracle.codomain());
        assert_eq!(output.domain(), oracle.domain());
        assert_eq!(output.dense_data().unwrap(), oracle.dense_data().unwrap());
    }
    assert!(workspace.intermediates[0].output.is_some());
}

#[test]
fn default_cached_contract_replays_rank_four_orientation_after_shape_drift_f64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(
        crate::plan_cache_config(&runtime).optimizer,
        Optimizer::Greedy
    );
    let provider = Arc::new(U1FusionRule);
    assert_rank_four_orientation_replay(engine, &runtime, &provider, |real, _| real);
}

#[test]
fn default_cached_contract_replays_rank_four_orientation_after_shape_drift_c64() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(
        crate::plan_cache_config(&runtime).optimizer,
        Optimizer::Greedy
    );
    let provider = Arc::new(U1FusionRule);
    assert_rank_four_orientation_replay(engine, &runtime, &provider, Complex64::new);
}

/// `A[i;j] * B[j;k]`, optionally times the closed pair `C[x;y] * D[y;x]`.
fn matrix_product_plan(with_closed_pair: bool, output: &[&str]) -> ContractionPlan {
    let mut inputs = vec![vec![label("i"), label("j")], vec![label("j"), label("k")]];
    let output = output.iter().map(|name| label(name)).collect::<Vec<_>>();
    let step = |lhs, rhs, result, labels| {
        ContractionStep::new(
            TensorId::new(lhs),
            TensorId::new(rhs),
            TensorId::new(result),
            0,
            labels,
        )
    };
    let steps = if with_closed_pair {
        inputs.push(vec![label("x"), label("y")]);
        inputs.push(vec![label("y"), label("x")]);
        vec![
            step(2, 3, 4, vec![]),
            step(0, 1, 5, vec![label("i"), label("k")]),
            step(5, 4, 6, output.clone()),
        ]
    } else {
        vec![step(0, 1, 2, output.clone())]
    };
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    ContractionPlan::from_steps(&ir, steps).unwrap()
}

#[test]
fn plan_with_rejects_a_plan_that_does_not_cover_the_network() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let (i, j, k, x, y) = (space(2), space(3), space(4), space(2), space(3));
    let a =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&i], [&j], 31).unwrap();
    let b = TensorMap::rand_with_seed(engine, &runtime, [&j], [&k], 32).unwrap();
    let c = TensorMap::rand_with_seed(engine, &runtime, [&x], [&y], 33).unwrap();
    let d = TensorMap::rand_with_seed(engine, &runtime, [&y], [&x], 34).unwrap();
    let s = Network::new(
        vec![
            vec![TemporaryLabel::from("x"), TemporaryLabel::from("y")],
            vec![TemporaryLabel::from("y"), TemporaryLabel::from("x")],
        ],
        vec![false, false],
        vec![1, 1],
        Vec::new(),
        Some(0),
    )
    .unwrap()
    .contract(engine, &[&c, &d])
    .unwrap();
    let network = |inputs: Vec<Vec<&str>>, splits: Vec<usize>| {
        let n = inputs.len();
        Network::new(inputs, vec![false; n], splits, ["i", "k"], Some(1)).unwrap()
    };
    let ab = matrix_product_plan(false, &["i", "k"]);
    let rejects = |result: Result<PlannedNetwork, Error>| {
        assert!(
            matches!(result, Err(Error::InvalidArgument(message)) if message.contains("plan topology"))
        );
    };

    // A rank-0 operand is a closed scalar subnetwork: a 2-tensor A*B plan
    // would otherwise drop it and return A*B.
    let with_scalar = network(vec![vec!["i", "j"], vec!["j", "k"], vec![]], vec![1, 1, 0]);
    rejects(with_scalar.plan_with(&[&a, &b, &s], ab.clone()));

    // A pair contracted only with each other is the same closed subnetwork.
    let with_pair = network(
        vec![
            vec!["i", "j"],
            vec!["j", "k"],
            vec!["x", "y"],
            vec!["y", "x"],
        ],
        vec![1; 4],
    );
    let abcd = [&a, &b, &c, &d];
    rejects(with_pair.plan_with(&abcd, ab.clone()));

    // A plan for more tensors than the network has is rejected up front.
    let product = network(vec![vec!["i", "j"], vec!["j", "k"]], vec![1; 2]);
    rejects(product.plan_with(&[&a, &b], matrix_product_plan(true, &["i", "k"])));

    // A plan for the same tensor count but another output is rejected.
    rejects(product.plan_with(&[&a, &b], matrix_product_plan(false, &["k", "i"])));

    // A matching plan still wraps and executes to the searched result.
    let supplied = with_pair
        .plan_with(&abcd, matrix_product_plan(true, &["i", "k"]))
        .unwrap()
        .execute(engine, &abcd, &mut Default::default())
        .unwrap();
    let searched = with_pair
        .plan(&abcd, &GreedyDenseOptimizer)
        .unwrap()
        .execute(engine, &abcd, &mut Default::default())
        .unwrap();
    assert_eq!(supplied.codomain(), searched.codomain());
    assert_eq!(supplied.domain(), searched.domain());
    let (supplied, searched) = (
        supplied.dense_data().unwrap(),
        searched.dense_data().unwrap(),
    );
    assert!(supplied
        .iter()
        .zip(searched)
        .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1e-12 * rhs.abs().max(1.0)));
}

/// What: one execution workspace driven A -> B -> A with the same chain
/// uploaded on each engine returns the Host value every time: the device
/// intermediates retained for one engine are dropped and rebuilt on the
/// other instead of being refused or submitted.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn a_network_workspace_follows_the_engine_of_each_call() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let open =
        || symtn::typed::CudaEngine::open(symtn::typed::CudaDeviceId::from_ordinal(0)).unwrap();
    let (first, second) = (&mut open(), &mut open());
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let host = (0..3)
        .map(|seed| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(
                engine,
                &runtime,
                [&leg],
                [&leg],
                748_240 + seed,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let labels = |names: &[&str]| names.iter().copied().map(label).collect::<Vec<_>>();
    let network = Network::new(
        vec![
            labels(&["a", "b"]),
            labels(&["b", "c"]),
            labels(&["c", "d"]),
        ],
        vec![false; 3],
        vec![1; 3],
        labels(&["a", "d"]),
        Some(1),
    )
    .unwrap();
    let planned = network
        .plan(&[&host[0], &host[1], &host[2]], &GreedyDenseOptimizer)
        .unwrap();
    let expected = planned
        .execute(
            engine,
            &[&host[0], &host[1], &host[2]],
            &mut NetworkExecutionWorkspace::default(),
        )
        .unwrap();
    let on = |cuda: &mut dyn Engine| {
        host.iter()
            .map(|tensor| tensor.to_cuda(cuda).unwrap())
            .collect::<Vec<_>>()
    };
    let (on_a, on_b) = (on(first), on(second));
    let mut workspace = NetworkExecutionWorkspace::default();
    for (what, on_b_engine) in [("A", false), ("B", true), ("A again", false)] {
        let (cuda, tensors): (&mut dyn Engine, _) = if on_b_engine {
            (&mut *second, &on_b)
        } else {
            (&mut *first, &on_a)
        };
        let result = planned
            .execute(
                cuda,
                &[&tensors[0], &tensors[1], &tensors[2]],
                &mut workspace,
            )
            .unwrap_or_else(|error| panic!("{what}: {error}"))
            .to_host(cuda)
            .unwrap();
        for (&got, &want) in result
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
        {
            assert!((got - want).abs() <= 1e-12 * (1.0 + want.abs()), "{what}");
        }
    }

    // A failing call on another engine (inputs of another rank, rejected by
    // the topology check) keeps no device intermediate of the old engine.
    assert!(workspace
        .intermediates
        .iter()
        .any(|buffers| buffers.output.is_some()));
    let wrong_rank = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg],
        748_250,
    )
    .unwrap()
    .to_cuda(second)
    .unwrap();
    assert!(planned
        .execute(second, &[&wrong_rank, &on_b[1], &on_b[2]], &mut workspace)
        .is_err());
    assert!(workspace
        .intermediates
        .iter()
        .all(|buffers| buffers.output.is_none()));
    assert!(workspace.slots.iter().all(Option::is_none));
}
