# Contraction order

`symtn-network` plans with `GreedyDenseOptimizer` by default: deterministic,
cheap, and the order `Network::contract` and the plan cache use.
`LabelOrderDenseOptimizer` contracts labels in a given order.

## omeco (feature `omeco`)

`OmecoOptimizer::Optimal` (exact minimum total pair cost, small networks) and
`OmecoOptimizer::TreeSa(omeco::TreeSA)` wrap the
[omeco](https://crates.io/crates/omeco) crate. Pass one to `Network::plan`;
no default path runs it. Its ties are not deterministic across processes
(omeco iterates a `std` `HashMap`), and TreeSA runs on the global rayon pool.
Slicing stays with `greedy_slice` on the chosen order.

## An order from cotengra or another planner

symtn does not call Python. Run cotengra in the application, take its
active-pair path (opt_einsum format: positions in the shrinking list of live
tensors, results appended), and build the plan with
`ContractionPlan::from_active_pair_path`; its rustdoc has the Python and Rust
sides. `Network::plan(&tensors, &plan)` validates the order against the
operands. For sliced execution, choose labels with `slice_plan_for` and pass a
`SlicedPlan`.
