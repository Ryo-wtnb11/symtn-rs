use symtn_network::{
    active_pair_path_from_tree, dense_steps_from_active_pair_path, greedy_slice, ActivePair,
    ContractionPlan, DenseContractionOptimizer, DenseCostModel, DenseTensorInfo,
    GreedyDenseOptimizer, LabelOrderDenseOptimizer, NetworkIR, SliceLabels, TemporaryLabel,
};

// Format contract: `ContractionPlan::to_text` and `SlicePlan::to_text` write
// the versioned (`-v2`) text that applications persist in plan-cache files and
// restore with `from_text`, so a change to these strings is a format change
// that needs a new header version, not a refactor. The chain a-b-c-d contracts
// (0, 1) -> 3 with labels a c, then 3 with 2 -> 4 with labels a d, each step
// line reading `lhs rhs result cost "labels"`; slicing the internal label c of
// extent 8 yields 8 slices, each `slice <kind> "label"`.
const DENSE_PATH: &str =
    "symtn-contract-plan-v2\ntensor_count 3\noutput \"a\" \"d\"\nstep 0 1 3 32 \"a\" \"c\"\nstep 2 3 4 50 \"a\" \"d\"\n";
const DENSE_GREEDY: &str =
    "symtn-contract-plan-v2\ntensor_count 3\noutput \"a\" \"d\"\nstep 0 1 3 24 \"a\" \"c\"\nstep 3 2 4 40 \"a\" \"d\"\n";
const SLICE_INTERNAL: &str = "symtn-slice-plan-v2\nnslices 8\nsliced_width 4\nunsliced_width 16\nper_slice_flops 8\nslice internal \"c\"\n";
const SLICE_OUTPUT: &str = "symtn-slice-plan-v2\nnslices 6\nsliced_width 6\nunsliced_width 36\nper_slice_flops 7\nslice output \"a\"\n";

fn chain_ir() -> NetworkIR {
    NetworkIR::from_labels(
        vec![
            vec!["a".into(), "b".into()],
            vec!["b".into(), "c".into()],
            vec!["c".into(), "d".into()],
        ],
        vec!["a".into(), "d".into()],
    )
    .unwrap()
}

fn dense(ir: &NetworkIR, sizes: [[usize; 2]; 3]) -> DenseCostModel {
    DenseCostModel::from_network(
        ir,
        &sizes
            .into_iter()
            .map(|dims| DenseTensorInfo::new(dims.to_vec()))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Every plan producer (active-pair path, label order, greedy, tree round
/// trip) and both slice modes write the versioned plan-cache text format.
#[test]
fn plan_producers_write_the_versioned_text_format() {
    let ir = chain_ir();
    let cost = dense(&ir, [[2, 3], [3, 4], [4, 5]]);
    let path = [ActivePair::new(0, 1), ActivePair::new(0, 1)];
    let order = [TemporaryLabel::new("b"), TemporaryLabel::new("c")];
    let path_steps = dense_steps_from_active_pair_path(&ir, &path, &cost).unwrap();
    let path_plan = ContractionPlan::from_steps(&ir, path_steps).unwrap();
    assert_eq!(path_plan.to_text(), DENSE_PATH);

    let label_steps = DenseContractionOptimizer::optimize(
        &LabelOrderDenseOptimizer::new(order.to_vec()),
        &ir,
        &cost,
    )
    .unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, label_steps)
            .unwrap()
            .to_text(),
        DENSE_PATH
    );

    let greedy_steps = GreedyDenseOptimizer.optimize(&ir, &cost).unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, greedy_steps)
            .unwrap()
            .to_text(),
        DENSE_GREEDY
    );

    let tree = path_plan.tree().unwrap();
    let tree_path = active_pair_path_from_tree(ir.tensors().len(), &tree).unwrap();
    let tree_steps = dense_steps_from_active_pair_path(&ir, &tree_path, &cost).unwrap();
    assert_eq!(
        ContractionPlan::from_steps(&ir, tree_steps)
            .unwrap()
            .to_text(),
        DENSE_PATH
    );

    let internal_cost = dense(&ir, [[2, 2], [2, 8], [8, 2]]);
    let internal_plan = ContractionPlan::from_steps(
        &ir,
        dense_steps_from_active_pair_path(&ir, &path, &internal_cost).unwrap(),
    )
    .unwrap();
    assert_eq!(
        greedy_slice(
            &ir,
            &internal_plan,
            &internal_cost,
            8,
            SliceLabels::InternalOnly
        )
        .to_text(),
        SLICE_INTERNAL
    );

    let output_cost = dense(&ir, [[6, 1], [1, 1], [1, 6]]);
    let output_plan = ContractionPlan::from_steps(
        &ir,
        dense_steps_from_active_pair_path(&ir, &path, &output_cost).unwrap(),
    )
    .unwrap();
    assert_eq!(
        greedy_slice(
            &ir,
            &output_plan,
            &output_cost,
            6,
            SliceLabels::IncludeOutput
        )
        .to_text(),
        SLICE_OUTPUT
    );
}
