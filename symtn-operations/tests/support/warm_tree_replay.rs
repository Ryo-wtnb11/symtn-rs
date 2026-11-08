//! The warm parallel tree replay fixture, measured process-wide by
//! `tests/warm_tree_replay.rs`.

use super::{Arc, BlockStructure, TreeTransformBlockSpec, TreeTransformStructure};

/// Worker threads of the fixture's pool.
pub const THREADS: usize = 4;

/// `groups` Multi groups of two `[rows, 3]` blocks each (24 elements at
/// eight rows keep every move's copy estimate nonzero), plus one Single,
/// recoupled by a non-identity matrix.
pub fn fixture(
    groups: usize,
    rows: usize,
) -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
    let structure = Arc::new(
        BlockStructure::packed_column_major(2, vec![vec![rows, 3]; 2 * groups + 1]).unwrap(),
    );
    let mut specs = (0..groups)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 2.0, -0.5, 1.5],
            )
        })
        .collect::<Vec<_>>();
    specs.push(TreeTransformBlockSpec::single(2 * groups, 2 * groups, -2.0));
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    let source = (0..structure.required_len().unwrap())
        .map(|value| value as f64 * 0.25)
        .collect();
    (structure, transform, source)
}
