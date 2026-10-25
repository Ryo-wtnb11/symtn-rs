use std::cell::Cell;

use symtn_core::{
    BlockKey, BlockSpec, BlockStructure, FusionProductSpace, FusionTreeHomSpace, FusionTreePairKey,
    SectorId, SectorLeg, Z2FusionRule,
};
use symtn_operations::{try_tensortrace_owned_raw, OwnedTraceTerm};

use crate::counting_alloc;

fn fusion_key() -> BlockKey {
    BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    )
}

fn canonical_structure_with_dim(dim: usize) -> BlockStructure {
    BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(fusion_key(), vec![dim, dim], vec![1, dim], 0).unwrap()],
    )
    .unwrap()
}

fn canonical_structure() -> BlockStructure {
    canonical_structure_with_dim(8)
}

fn padded_structure() -> BlockStructure {
    BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(fusion_key(), vec![8, 8], vec![1, 8], 1).unwrap()],
    )
    .unwrap()
}

fn incomplete_coupled_structure() -> (BlockStructure, BlockStructure) {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let canonical = BlockStructure::coupled_sector_matrix_with_keys(
        &rule,
        2,
        4,
        keys.iter().cloned().map(|key| (key, vec![1; 4])).collect(),
    )
    .unwrap();
    let incomplete = BlockStructure::from_blocks_with_rank(
        4,
        (0..canonical.block_count() - 1)
            .map(|index| {
                let block = canonical.block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    (canonical, incomplete)
}

fn execute(dst: &BlockStructure, src: &BlockStructure, source: &[f64]) -> Option<Vec<f64>> {
    let producer_indices = [0usize];
    let producer_offsets = [0, 1];
    try_tensortrace_owned_raw(
        dst,
        1,
        src,
        source,
        false,
        1,
        &producer_indices,
        &producer_offsets,
        |_| OwnedTraceTerm::new(0, 0, &[8, 8], &[], &[1, 8], &[], 1.0),
        1.0,
    )
    .unwrap()
}

fn execute_dim<F>(
    dst: &BlockStructure,
    src: &BlockStructure,
    source: &[f64],
    term_at: F,
) -> Option<Vec<f64>>
where
    F: Fn(usize) -> OwnedTraceTerm<'static, f64>,
{
    let producer_indices = [0usize];
    let producer_offsets = [0, 1];
    try_tensortrace_owned_raw(
        dst,
        1,
        src,
        source,
        false,
        1,
        &producer_indices,
        &producer_offsets,
        term_at,
        1.0,
    )
    .unwrap()
}

#[test]
fn warm_owned_trace_allocates_only_the_output_payload() {
    let _serial = crate::counting_alloc::serial();
    // What: after canonical-region proof is warm, the specialized trace writer
    // allocates only its returned payload; its bounded tile stays on the stack.
    let structure = canonical_structure();
    let source = vec![2.0; 64];
    drop(execute(&structure, &structure, &source).unwrap());
    counting_alloc::start();
    let output = execute(&structure, &structure, &source).unwrap();
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 1);
    assert_eq!(output, source);
}

#[test]
fn padded_destination_declines_without_allocating_output() {
    let _serial = crate::counting_alloc::serial();
    // What: a leading storage hole cannot claim complete physical coverage and
    // returns control to the initialized fallback before payload allocation.
    let source_structure = canonical_structure();
    let destination_structure = padded_structure();
    let source = vec![2.0; 64];
    assert!(execute(&destination_structure, &source_structure, &source).is_none());
    counting_alloc::start();
    let output = execute(&destination_structure, &source_structure, &source);
    let allocs = counting_alloc::stop();

    assert!(output.is_none());
    assert_eq!(allocs.calls, 0);
}

#[test]
fn incomplete_coupled_grid_declines_before_output_allocation() {
    let _serial = crate::counting_alloc::serial();
    // What: a missing tree-pair cannot claim complete coupled-sector coverage
    // and returns to the initialized caller before allocating an output.
    let (source_structure, destination_structure) = incomplete_coupled_structure();
    assert!(destination_structure
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let source = vec![1.0; source_structure.required_len().unwrap()];
    let producer_offsets = vec![0; destination_structure.block_count() + 1];
    counting_alloc::start();
    let output = try_tensortrace_owned_raw::<f64, f64, _>(
        &destination_structure,
        2,
        &source_structure,
        &source,
        false,
        0,
        &[],
        &producer_offsets,
        |_| unreachable!(),
        1.0,
    )
    .unwrap();
    let allocs = counting_alloc::stop();

    assert!(output.is_none());
    assert_eq!(allocs.calls, 0);
}

#[test]
fn rank_zero_and_zero_extent_outputs_commit_empty_or_scalar_storage() {
    let _serial = crate::counting_alloc::serial();
    // What: empty rank-zero storage and a zero degeneracy extent complete
    // without attempting a tile iteration or exposing uninitialized elements.
    let empty = BlockStructure::empty(0);
    let empty_offsets = [0];
    let empty_indices = [];
    let empty_output = try_tensortrace_owned_raw::<f64, f64, _>(
        &empty,
        0,
        &empty,
        &[],
        false,
        0,
        &empty_indices,
        &empty_offsets,
        |_| unreachable!(),
        1.0,
    )
    .unwrap()
    .unwrap();
    assert!(empty_output.is_empty());

    let zero = canonical_structure_with_dim(0);
    let zero_output = execute_dim(&zero, &zero, &[], |_| {
        OwnedTraceTerm::new(0, 0, &[0, 0], &[], &[1, 0], &[], 1.0)
    })
    .unwrap();
    assert!(zero_output.is_empty());
}

#[test]
fn canonical_block_without_a_producer_is_written_as_structural_zero() {
    let _serial = crate::counting_alloc::serial();
    // What: an active canonical destination with no matching trace term is
    // initialized exactly once to structural zero rather than left untouched.
    let structure = canonical_structure();
    let producer_indices = [];
    let producer_offsets = [0, 0];
    let output = try_tensortrace_owned_raw::<f64, f64, _>(
        &structure,
        1,
        &structure,
        &[0.0; 64],
        false,
        0,
        &producer_indices,
        &producer_offsets,
        |_| unreachable!(),
        1.0,
    )
    .unwrap()
    .unwrap();

    assert_eq!(output, [0.0; 64]);
}

#[test]
fn multiple_producers_accumulate_in_compiled_stable_order() {
    let _serial = crate::counting_alloc::serial();
    // What: producers for one destination retain compiler order; changing this
    // sequence changes the deliberately ill-conditioned result.
    let structure = canonical_structure_with_dim(1);
    let producer_indices = [0, 1, 2];
    let producer_offsets = [0, 3];
    let coefficients = [1.0e16, -1.0e16, 1.0];
    let output = try_tensortrace_owned_raw(
        &structure,
        1,
        &structure,
        &[1.0],
        false,
        3,
        &producer_indices,
        &producer_offsets,
        |index| OwnedTraceTerm::new(0, 0, &[1, 1], &[], &[1, 1], &[], coefficients[index]),
        1.0,
    )
    .unwrap()
    .unwrap();

    assert_eq!(output, [1.0]);
}

#[test]
fn preflight_error_and_mid_write_panic_never_return_partial_output() {
    let _serial = crate::counting_alloc::serial();
    // What: a source extent error happens before output allocation, and a panic
    // after the first tile scatter remains trapped inside the length-zero Vec.
    let structure = canonical_structure_with_dim(32);
    let short_source = vec![1.0; 1023];
    let producer_indices = [0usize];
    let producer_offsets = [0, 1];
    counting_alloc::start();
    let error = try_tensortrace_owned_raw(
        &structure,
        1,
        &structure,
        &short_source,
        false,
        1,
        &producer_indices,
        &producer_offsets,
        |_| OwnedTraceTerm::new(0, 0, &[32, 32], &[], &[1, 32], &[], 1.0),
        1.0,
    )
    .unwrap_err();
    let allocs = counting_alloc::stop();
    assert!(matches!(
        error,
        symtn_operations::OperationError::ElementCountMismatch { .. }
    ));
    assert_eq!(allocs.calls, 0);

    let source = vec![1.0; 1024];
    let calls = Cell::new(0usize);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = try_tensortrace_owned_raw(
            &structure,
            1,
            &structure,
            &source,
            false,
            1,
            &producer_indices,
            &producer_offsets,
            |_| {
                let call = calls.get() + 1;
                calls.set(call);
                if call == 4 {
                    panic!("injected after one completed output tile");
                }
                OwnedTraceTerm::new(0, 0, &[32, 32], &[], &[1, 32], &[], 1.0)
            },
            1.0,
        );
    }));
    assert!(panicked.is_err());
    assert_eq!(calls.get(), 4);
}
