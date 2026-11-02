//! Allocation calls of one compact block move, measured before any key is
//! materialized, so per-row scratch cannot cancel against other stages or
//! the owned per-source result.

use super::*;
use crate::tests::counting_alloc;

/// The pair keys of the complete SU(2) block `1^⊗5 ← coupled`: every move
/// below maps it onto a basis of the same size.
fn complete_spin_one_block(twice_coupled: usize) -> Vec<FusionTreePairKey> {
    let leg = |twice| SectorLeg::new([(SU2Irrep::from_twice_spin(twice).sector_id(), 1)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new(std::array::from_fn::<_, 5, _>(|_| leg(2))),
        FusionProductSpace::new([leg(twice_coupled)]),
    )
    .fusion_tree_keys(&SU2FusionRule)
    .to_vec()
}

fn identity_columns(rows: usize) -> DenseColumns<f64> {
    let mut columns = DenseColumns::with_capacity(rows, rows);
    for source in 0..rows {
        let row = columns.push_empty_row();
        columns.row_mut(row)[source] = Some(1.0);
    }
    columns
}

fn pair_basis(keys: &[FusionTreePairKey]) -> CompactMultiplicityFreeTreePairBasis {
    let group = validate_tree_pair_block_group_for_rule(&SU2FusionRule, keys)
        .unwrap()
        .expect("nonempty block");
    CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap()
}

fn tree_basis(keys: &[FusionTreePairKey]) -> CompactMultiplicityFreeTreeBasis {
    let trees = keys
        .iter()
        .map(|key| key.codomain_tree().clone())
        .collect::<Vec<_>>();
    let group = validate_fusion_tree_block_group_for_rule(&SU2FusionRule, &trees)
        .unwrap()
        .expect("nonempty block");
    CompactMultiplicityFreeTreeBasis::from_group(group).unwrap()
}

/// Warm calls of one composed move `apply` (returning its output rows) over
/// the complete block of `twice_coupled`, and the block's rows. The incoming
/// basis and source columns are built outside the measured window.
fn move_calls<B>(
    twice_coupled: usize,
    basis: fn(&[FusionTreePairKey]) -> B,
    apply: fn(B, &DenseColumns<f64>) -> usize,
) -> (usize, u64) {
    let keys = complete_spin_one_block(twice_coupled);
    let columns = identity_columns(keys.len());
    assert_eq!(apply(basis(&keys), &columns), keys.len());
    let incoming = basis(&keys);
    let (rows, allocs) = counting_alloc::measure(|| apply(incoming, &columns));
    (rows, allocs.calls)
}

/// Purpose: one compact move (codomain Artin of a tree-pair block, Artin of a
/// tree block, right and left bend) costs the same allocation calls for a
/// 4-row and a 15-row incoming basis: its row terms go through one reused
/// buffer, never a container per row.
#[test]
fn compact_moves_allocate_per_move_not_per_row() {
    // Index 2 swaps two spin-1 legs below an inner line, so the Artin moves
    // recouple (several terms per row), not only reorder.
    fn pair_artin(basis: CompactMultiplicityFreeTreePairBasis, c: &DenseColumns<f64>) -> usize {
        compact_codomain_artin_block(&SU2FusionRule, basis, Some(c), 2, false)
            .unwrap()
            .1
            .num_rows
    }
    fn tree_artin(basis: CompactMultiplicityFreeTreeBasis, c: &DenseColumns<f64>) -> usize {
        compact_artin_tree_block(&SU2FusionRule, basis, Some(c), 2, false)
            .unwrap()
            .1
            .num_rows
    }
    fn bend_right(basis: CompactMultiplicityFreeTreePairBasis, c: &DenseColumns<f64>) -> usize {
        compact_bendright_block(&SU2FusionRule, basis, Some(c))
            .unwrap()
            .1
            .num_rows
    }
    fn bend_left(basis: CompactMultiplicityFreeTreePairBasis, c: &DenseColumns<f64>) -> usize {
        compact_bendleft_block(&SU2FusionRule, basis, Some(c))
            .unwrap()
            .1
            .num_rows
    }
    let pair = |apply| {
        [
            move_calls(8, pair_basis, apply),
            move_calls(2, pair_basis, apply),
        ]
    };
    for (what, [(small_rows, small), (large_rows, large)]) in [
        ("tree-pair codomain Artin", pair(pair_artin)),
        (
            "tree Artin",
            [
                move_calls(8, tree_basis, tree_artin),
                move_calls(2, tree_basis, tree_artin),
            ],
        ),
        ("right bend", pair(bend_right)),
        ("left bend", pair(bend_left)),
    ] {
        assert_eq!((small_rows, large_rows), (4, 15), "{what}");
        assert_eq!(
            small, large,
            "{what}: {small} allocation calls for 4 rows, {large} for 15"
        );
    }
}
