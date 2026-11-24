use super::*;

/// The fermionic contraction twist of one materialized core operand `space`:
/// `θ` of each block is read from its contracted fusion tree — the codomain
/// tree of a core-right block, the domain tree of a core-left block (equal to
/// the codomain tree of every core-right block it meets) — at the positions
/// where `core_right`'s contracted leg is dual, as TensorKit twists `B`'s
/// dual codomain legs or the matching `A` domain legs (`blas_contract!`,
/// src/tensors/tensoroperations.jl:419/429 @cfaa073).
///
/// Returns `(block offset, θ_b)` for every non-empty block with `θ_b ≠ 1`,
/// sorted by offset: the destination scales the transform materializing
/// `space` folds into the moves writing those blocks (Host and device
/// alike). A zero-element block is left out because it can share its offset
/// with the next block.
pub(crate) fn compile_contract_twist<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    core_right: &FusionTreeHomSpace,
    space_is_core_left: bool,
    rhs_contracting_axes: &[usize],
) -> Result<Vec<(usize, R::Scalar)>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    if rule.braiding_style() != symtn_core::BraidingStyleKind::Fermionic {
        return Ok(Vec::new());
    }
    compile_contract_twist_with(
        space,
        core_right,
        space_is_core_left,
        rhs_contracting_axes,
        |sector| Ok(rule.twist_scalar(sector)),
    )
}

pub(crate) fn compile_contract_twist_with<C: DenseBlockScalar, E: From<OperationError>>(
    space: &DynamicFusionMapSpace,
    core_right: &FusionTreeHomSpace,
    space_is_core_left: bool,
    rhs_contracting_axes: &[usize],
    mut theta: impl FnMut(symtn_core::SectorId) -> Result<C, E>,
) -> Result<Vec<(usize, C)>, E> {
    let structure = space.structure();
    let mut scales = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let symtn_core::BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        if block.shape().contains(&0) {
            continue;
        }
        let contracted_tree = if space_is_core_left {
            key.domain_tree()
        } else {
            key.codomain_tree()
        };
        let factor = super::fusion::contract_twist_factor(
            symtn_core::OrientedFusionTreeHomSpace::new(
                core_right,
                symtn_core::FusionTreePairOrientation::Direct,
            ),
            rhs_contracting_axes,
            contracted_tree,
            &mut theta,
        )?;
        if factor != C::one() {
            scales.push((block.offset(), factor));
        }
    }
    scales.sort_unstable_by_key(|&(offset, _)| offset);
    Ok(scales)
}

/// θ_b of the destination block at `offset`: one where `scales` (sorted by
/// offset) has no entry.
fn destination_scale<C: DenseBlockScalar>(scales: &[(usize, C)], offset: usize) -> C {
    scales
        .binary_search_by_key(&offset, |&(block, _)| block)
        .map_or_else(|_| C::one(), |index| scales[index].1)
}

/// Asserts θ is uniform over the destination layouts of every Multi block of
/// the twisted operand's transform. Scaling each scatter by its own block's θ
/// would be exact either way; what this pins is the invariant behind it: θ
/// depends only on the destination's contracted uncoupled sectors (codomain
/// of a core-right block, domain of a core-left one), which one Multi block
/// shares, so a non-uniform list means the
/// twist and the transform disagree about block identity — an internal
/// inconsistency of the compiled artifact, reported before any replay.
pub(in crate::contract) fn validate_uniform_multi_scales<C: DenseBlockScalar>(
    structure: &TreeTransformStructure<C>,
    scales: &[(usize, C)],
) -> Result<(), OperationError> {
    let layouts = structure.layouts();
    for block in structure.blocks() {
        let symtn_operations::TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            ..
        } = *block
        else {
            continue;
        };
        let mut first = None;
        for index in dst_layout_start..dst_layout_start + dst_count {
            let layout = layouts.entry(index);
            if layout.element_count == 0 {
                continue;
            }
            let offset =
                usize::try_from(layout.offset).map_err(|_| OperationError::InvalidArgument {
                    message: "twisted transform destination layout has a negative offset",
                })?;
            let theta = destination_scale(scales, offset);
            match first {
                None => first = Some(theta),
                Some(expected) if expected == theta => {}
                Some(_) => {
                    return Err(OperationError::InvalidArgument {
                        message: "fermionic contraction twist is not uniform within one \
                                  recoupling block",
                    })
                }
            }
        }
    }
    Ok(())
}
