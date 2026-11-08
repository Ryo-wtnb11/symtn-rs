use super::*;

pub(super) fn ensure_recoupling_coefficients<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    structure_identity: &Arc<()>,
) -> Result<bool, OperationError>
where
    D: RecouplingCoefficientAction<C>,
    C: Copy,
{
    let plan = task.recoupling_plan();
    // Why not key by the shared categorical payload alone: different layout
    // bindings can reorder Multi jobs by element count and therefore require
    // different packed RHS orders for the same categorical matrices.
    let same_structure = workspace
        .coefficient_structure_identity
        .as_ref()
        .and_then(Weak::upgrade)
        .is_some_and(|identity| Arc::ptr_eq(&identity, structure_identity));
    if same_structure && workspace.coefficient_scratch.len() == plan.coefficient_len() {
        return Ok(false);
    }

    workspace.coefficient_scratch.clear();
    workspace
        .coefficient_scratch
        .reserve(plan.coefficient_len());
    // Preserve entry order: each job's rhs_offset addresses this exact pack.
    for (block_index, _) in plan.entries() {
        let block = recoupling_multi_block(task, block_index)?;
        let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            ..
        } = *block
        else {
            continue;
        };
        let coefficient_len = src_count
            .checked_mul(dst_count)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let coefficients = task.block_matrix(block_index).unwrap_or_default();
        if coefficients.len() != coefficient_len {
            return Err(OperationError::CoefficientCountMismatch {
                expected: coefficient_len,
                actual: coefficients.len(),
            });
        }
        workspace.coefficient_scratch.extend(
            coefficients
                .iter()
                .map(|&coefficient| D::coefficient_as_data(coefficient)),
        );
    }
    if workspace.coefficient_scratch.len() != plan.coefficient_len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: plan.coefficient_len(),
            actual: workspace.coefficient_scratch.len(),
        });
    }
    workspace.coefficient_structure_identity = Some(Arc::downgrade(structure_identity));
    #[cfg(any(test, feature = "testing"))]
    {
        workspace.coefficient_pack_builds += 1;
    }
    Ok(true)
}

#[cfg(test)]
mod coefficient_cache_tests {
    use super::*;
    use crate::{TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan};
    use symtn_core::{BlockKey, BlockSpec, FusionTreePairKey};

    fn multi_recoupling_structure(coefficients: [f64; 4]) -> TreeTransformStructure<f64> {
        let block_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
        TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                coefficients.to_vec(),
            )],
        )
        .unwrap()
    }

    fn group_pair(group: usize, vertex: usize) -> FusionTreePairKey {
        FusionTreePairKey::try_pair_from_sector_ids(
            [group, group],
            [],
            group,
            [false, false],
            [],
            [],
            [],
            [vertex],
            [],
        )
        .unwrap()
    }

    fn two_group_layout(keys: &[FusionTreePairKey; 4], elements: [usize; 2]) -> BlockStructure {
        let mut offset = 0;
        BlockStructure::from_blocks_with_rank(
            2,
            keys.iter()
                .enumerate()
                .map(|(index, key)| {
                    let block_elements = elements[index / 2];
                    let block = BlockSpec::column_major_with_key(
                        BlockKey::from(key.clone()),
                        vec![block_elements, 1],
                        offset,
                    )
                    .unwrap();
                    offset += block_elements;
                    block
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn recoupling_coefficients_cache_uses_live_structure_identity() {
        let structure = multi_recoupling_structure([1.0, 2.0, 3.0, 4.0]);
        let mut workspace = TreeTransformWorkspace::<f64>::default();

        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            structure.task_view().unwrap(),
            structure.identity_marker(),
        )
        .unwrap());
        assert_eq!(workspace.coefficient_scratch, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!ensure_recoupling_coefficients(
            &mut workspace,
            structure.task_view().unwrap(),
            structure.identity_marker(),
        )
        .unwrap());

        let structure_clone = structure.clone();
        assert!(!ensure_recoupling_coefficients(
            &mut workspace,
            structure_clone.task_view().unwrap(),
            structure_clone.identity_marker(),
        )
        .unwrap());

        let equal_but_distinct = multi_recoupling_structure([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(structure, equal_but_distinct);
        workspace.coefficient_scratch.fill(-1.0);
        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            equal_but_distinct.task_view().unwrap(),
            equal_but_distinct.identity_marker(),
        )
        .unwrap());
        assert_eq!(workspace.coefficient_scratch, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn categorical_payload_is_repacked_when_layout_reorders_multi_jobs() {
        // What: structure identity protects a layout-packed execution buffer;
        // categorical payload identity alone would incorrectly reuse [UA, UB].
        let keys = [
            group_pair(0, 1),
            group_pair(0, 2),
            group_pair(1, 1),
            group_pair(1, 2),
        ];
        let plan = TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::try_multi(
                keys[..2].to_vec(),
                keys[..2].to_vec(),
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            TreeTransformGroupBlockSpec::try_multi(
                keys[2..].to_vec(),
                keys[2..].to_vec(),
                vec![5.0, 6.0, 7.0, 8.0],
            )
            .unwrap(),
        ]);
        let a_then_b = two_group_layout(&keys, [1, 2]);
        let b_then_a = two_group_layout(&keys, [3, 1]);
        let first = plan.compile_structures(&a_then_b, &a_then_b).unwrap();
        let second = plan.compile_structures(&b_then_a, &b_then_a).unwrap();
        assert!(first.shares_recoupling_matrices_with(&second));
        let mut workspace = TreeTransformWorkspace::<f64>::default();

        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            first.task_view().unwrap(),
            first.identity_marker(),
        )
        .unwrap());
        assert_eq!(
            workspace.coefficient_scratch,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert!(ensure_recoupling_coefficients(
            &mut workspace,
            second.task_view().unwrap(),
            second.identity_marker(),
        )
        .unwrap());
        assert_eq!(
            workspace.coefficient_scratch,
            vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
        );
    }
}
