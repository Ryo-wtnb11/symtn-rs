use std::sync::Arc;

use symtn::sector::{SectorId, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};
use symtn_network::{
    slice_plan_for, ContractionPlan, ContractionStep, DenseCostModel, DenseTensorInfo, Network,
    NetworkIR, SlicedPlan, TemporaryLabel, TensorId,
};

fn label(name: &str) -> TemporaryLabel {
    TemporaryLabel::from(name)
}

#[test]
fn common_lowering_reconstructs_the_network_after_dense_roundtrip() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let two = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let vacuum = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)]).unwrap();
    let tensor = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        &runtime,
        [&two, &two, &vacuum],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        1028,
    )
    .unwrap();
    let labels = vec![label("a"), label("b"), label("c")];
    let ir = NetworkIR::from_labels(vec![labels.clone()], labels.clone()).unwrap();
    let order = ContractionPlan::from_steps(&ir, Vec::new()).unwrap();
    let cost = DenseCostModel::from_network(&ir, &[DenseTensorInfo::new(vec![3, 3, 1])]).unwrap();
    let dense = SlicedPlan::new(
        order,
        slice_plan_for(
            &ir,
            &ContractionPlan::from_steps(&ir, Vec::new()).unwrap(),
            &cost,
            &labels,
        ),
    );
    let network =
        Network::new(vec![labels.clone()], vec![false], vec![3], labels, Some(3)).unwrap();

    let lowered = network
        .lower_symmetric_sliced_plan(&[&tensor], dense.clone())
        .unwrap();
    let restored = SlicedPlan::from_text(&dense.to_text()).unwrap();
    let reconstructed = network
        .lower_symmetric_sliced_plan(&[&tensor], restored)
        .unwrap();

    assert_eq!(lowered, reconstructed);
    assert_eq!(lowered.slices().nslices(), 9);
    let first = lowered.slices().indices()[0].pieces();
    assert_eq!((first[0].range().start(), first[0].range().end()), (0, 1));
    assert_eq!((first[1].range().start(), first[1].range().end()), (1, 2));
    let q1 = SectorId::from(U1Irrep::new(1));
    let q0 = SectorId::from(U1Irrep::new(0));
    assert!(lowered.slices().combinations().any(|combination| {
        combination
            .iter()
            .map(|piece| piece.sector())
            .eq([q1, q1, q0])
    }));
}

#[test]
fn reconstruction_preserves_two_step_order_and_output_metadata() {
    let runtime = Runtime::builder().build().unwrap();
    let space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let tensors = (0..3)
        .map(|seed| {
            TensorMap::<U1FusionRule, f64>::rand_with_seed(
                &runtime,
                [&space],
                [&space],
                20_280 + seed,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let inputs = vec![
        vec![label("a"), label("b")],
        vec![label("b"), label("c")],
        vec![label("c"), label("d")],
    ];
    let output = vec![label("d"), label("a")];
    let ir = NetworkIR::from_labels(inputs.clone(), output.clone()).unwrap();
    let order = ContractionPlan::from_steps(
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
                output.clone(),
            ),
        ],
    )
    .unwrap();
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
        order.clone(),
        slice_plan_for(&ir, &order, &cost, &[label("b")]),
    );
    let restored = SlicedPlan::from_text(&dense.to_text()).unwrap();
    let network = Network::new(inputs, vec![false; 3], vec![1; 3], output, Some(1)).unwrap();
    let refs = [&tensors[0], &tensors[1], &tensors[2]];

    let original = network.lower_symmetric_sliced_plan(&refs, dense).unwrap();
    let reconstructed = network
        .lower_symmetric_sliced_plan(&refs, restored)
        .unwrap();

    assert_eq!(reconstructed, original);
    assert_eq!(reconstructed.plan(), &order);
    assert_eq!(reconstructed.plan().steps(), order.steps());
    assert_eq!(
        reconstructed.plan().output_labels(),
        &[label("d"), label("a")]
    );
}
