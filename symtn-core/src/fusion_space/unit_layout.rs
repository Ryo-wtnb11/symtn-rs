use super::*;

/// Proof that one exact fusion-tree block subset matches a HomSpace.
///
/// The proof is provider-free: it covers ranks, tree shapes, external-sector
/// membership, duality, and block degeneracies. Categorical validation remains
/// an explicit second step so malformed structure is rejected before provider
/// algebra is queried.
#[doc(hidden)]
pub struct StructurallyValidatedFusionTreeSubset<'homspace, 'structure> {
    homspace: &'homspace FusionTreeHomSpace,
    structure: &'structure BlockStructure,
}

#[doc(hidden)]
impl<'homspace, 'structure> StructurallyValidatedFusionTreeSubset<'homspace, 'structure> {
    pub fn try_new(
        homspace: &'homspace FusionTreeHomSpace,
        structure: &'structure BlockStructure,
    ) -> Result<Self, CoreError> {
        let expected_rank = homspace.rank();
        if structure.rank() != expected_rank {
            return Err(CoreError::StructureRankMismatch {
                expected: expected_rank,
                actual: structure.rank(),
            });
        }

        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                });
            };
            let codomain_tree = key.codomain_tree();
            let domain_tree = key.domain_tree();

            validate_fusion_tree_key_shape(codomain_tree.as_view())?;
            validate_fusion_tree_key_shape(domain_tree.as_view())?;
            if codomain_tree.uncoupled().len() != homspace.codomain().len()
                || domain_tree.uncoupled().len() != homspace.domain().len()
            {
                return Err(CoreError::FusionSpaceSplitMismatch {
                    expected_nout: homspace.codomain().len(),
                    expected_nin: homspace.domain().len(),
                    actual_nout: codomain_tree.uncoupled().len(),
                    actual_nin: domain_tree.uncoupled().len(),
                });
            }

            for (space, tree) in [
                (homspace.codomain(), codomain_tree),
                (homspace.domain(), domain_tree),
            ] {
                for (leg, &sector) in space.legs().iter().zip(tree.uncoupled()) {
                    if leg.degeneracy(sector).is_none() {
                        return Err(CoreError::MalformedFusionTree {
                            message: "fusion tree uses a sector absent from its HomSpace leg",
                        });
                    }
                }
                for (leg, &is_dual) in space.legs().iter().zip(tree.is_dual()) {
                    if leg.is_dual() != is_dual {
                        return Err(CoreError::MalformedFusionTree {
                            message: "fusion tree duality disagrees with its HomSpace leg",
                        });
                    }
                }
            }

            let (codomain_shape, domain_shape) = block.shape().split_at(homspace.codomain().len());
            for (space, tree, shape) in [
                (homspace.codomain(), codomain_tree, codomain_shape),
                (homspace.domain(), domain_tree, domain_shape),
            ] {
                for ((leg, &sector), &actual) in
                    space.legs().iter().zip(tree.uncoupled()).zip(shape)
                {
                    let expected =
                        leg.degeneracy(sector)
                            .ok_or(CoreError::MalformedFusionTree {
                                message: "fusion tree uses a sector absent from its HomSpace leg",
                            })?;
                    if expected != actual {
                        return Err(CoreError::LegDegeneracyMismatch {
                            sector,
                            expected,
                            actual,
                        });
                    }
                }
            }
        }
        Ok(Self {
            homspace,
            structure,
        })
    }

    pub fn validate_for_rule<R>(&self, rule: &R) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        for index in 0..self.structure.block_count() {
            let block = self.structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                });
            };
            key.validate_for_rule(rule)?;
        }
        Ok(())
    }

    pub fn validate_for_rule_checked<R>(&self, rule: &R) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        for index in 0..self.structure.block_count() {
            let block = self.structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                }
                .into());
            };
            validate_fusion_tree_pair_coupled(key.codomain_tree(), key.domain_tree())?;
        }
        for index in 0..self.structure.block_count() {
            let block = self.structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                }
                .into());
            };
            validate_fusion_tree_for_rule_checked_after_shape(rule, key.codomain_tree())?;
            validate_fusion_tree_for_rule_checked_after_shape(rule, key.domain_tree())?;
        }
        Ok(())
    }

    #[inline]
    pub fn homspace(&self) -> &'homspace FusionTreeHomSpace {
        self.homspace
    }
}

#[cfg(test)]
/// Validates that `larger` is the supplied `smaller` layout with one canonical
/// unit leg inserted in storage order.
///
/// Both structures must come from the ordinary layout authority; this proof
/// compares supplied metadata and does not enumerate missing blocks.
pub(crate) fn validate_unit_layout_correspondence<R>(
    rule: &R,
    smaller: (&FusionTreeHomSpace, &BlockStructure),
    larger: (&FusionTreeHomSpace, &BlockStructure),
    insertion: UnitLegInsertion,
) -> Result<(), CoreError>
where
    R: CanonicalUnitFusionRule,
{
    let (smaller_homspace, smaller_structure) = smaller;
    let (larger_homspace, larger_structure) = larger;
    let (is_codomain, local_position) = unit_insertion_side(smaller_homspace, insertion)
        .ok_or(CoreError::UnitLayoutCorrespondence)?;
    if larger_homspace.rank() != smaller_homspace.rank() + 1
        || larger_structure.rank() != smaller_structure.rank() + 1
    {
        return Err(CoreError::UnitLayoutCorrespondence);
    }
    StructurallyValidatedFusionTreeSubset::try_new(smaller_homspace, smaller_structure)?
        .validate_for_rule(rule)?;
    StructurallyValidatedFusionTreeSubset::try_new(larger_homspace, larger_structure)?
        .validate_for_rule(rule)?;
    validate_unit_layout_correspondence_after_preflight(
        rule.vacuum(),
        smaller,
        larger,
        insertion,
        is_codomain,
        local_position,
    )
}

/// Checked finite-algebra sibling of [`validate_unit_layout_correspondence`].
#[doc(hidden)]
pub fn validate_unit_layout_correspondence_checked<R>(
    rule: &R,
    smaller: (&FusionTreeHomSpace, &BlockStructure),
    larger: (&FusionTreeHomSpace, &BlockStructure),
    insertion: UnitLegInsertion,
) -> Result<(), CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    let (is_codomain, local_position) =
        unit_insertion_side(smaller.0, insertion).ok_or(CoreError::UnitLayoutCorrespondence)?;
    if larger.0.rank() != smaller.0.rank() + 1 || larger.1.rank() != smaller.1.rank() + 1 {
        return Err(CoreError::UnitLayoutCorrespondence.into());
    }
    let smaller_validated = StructurallyValidatedFusionTreeSubset::try_new(smaller.0, smaller.1)?;
    let larger_validated = StructurallyValidatedFusionTreeSubset::try_new(larger.0, larger.1)?;
    smaller_validated.validate_for_rule_checked(rule)?;
    larger_validated.validate_for_rule_checked(rule)?;
    validate_unit_layout_correspondence_after_preflight(
        FusionRule::vacuum(rule),
        smaller,
        larger,
        insertion,
        is_codomain,
        local_position,
    )
    .map_err(Into::into)
}

/// Checked Generic unit-layout correspondence after root admission.
///
/// The source and destination roots have already undergone Generic provider
/// validation; this seam checks only the exact structural/layout relation and
/// never falls back to the infallible provider API.
#[doc(hidden)]
pub fn validate_unit_layout_correspondence_generic_checked<R>(
    rule: &R,
    smaller: (&FusionTreeHomSpace, &BlockStructure),
    larger: (&FusionTreeHomSpace, &BlockStructure),
    insertion: UnitLegInsertion,
) -> Result<(), CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
{
    let (is_codomain, local_position) =
        unit_insertion_side(smaller.0, insertion).ok_or(CoreError::UnitLayoutCorrespondence)?;
    if larger.0.rank() != smaller.0.rank() + 1 || larger.1.rank() != smaller.1.rank() + 1 {
        return Err(CoreError::UnitLayoutCorrespondence.into());
    }
    StructurallyValidatedFusionTreeSubset::try_new(smaller.0, smaller.1)?;
    StructurallyValidatedFusionTreeSubset::try_new(larger.0, larger.1)?;
    validate_unit_layout_correspondence_after_preflight(
        rule.vacuum(),
        smaller,
        larger,
        insertion,
        is_codomain,
        local_position,
    )
    .map_err(Into::into)
}

fn unit_insertion_side(
    homspace: &FusionTreeHomSpace,
    insertion: UnitLegInsertion,
) -> Option<(bool, usize)> {
    let position = insertion.position();
    if position > homspace.rank() {
        return None;
    }
    match insertion {
        UnitLegInsertion::Left { .. } if position < homspace.codomain().len() => {
            Some((true, position))
        }
        UnitLegInsertion::Right { .. } if position <= homspace.codomain().len() => {
            Some((true, position))
        }
        _ => Some((false, position - homspace.codomain().len())),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_unit_layout_correspondence_after_preflight(
    unit: SectorId,
    smaller: (&FusionTreeHomSpace, &BlockStructure),
    larger: (&FusionTreeHomSpace, &BlockStructure),
    insertion: UnitLegInsertion,
    is_codomain: bool,
    local_position: usize,
) -> Result<(), CoreError> {
    let (smaller_homspace, smaller_structure) = smaller;
    let (larger_homspace, larger_structure) = larger;
    let (smaller_side, larger_side, smaller_other, larger_other) = if is_codomain {
        (
            smaller_homspace.codomain(),
            larger_homspace.codomain(),
            smaller_homspace.domain(),
            larger_homspace.domain(),
        )
    } else {
        (
            smaller_homspace.domain(),
            larger_homspace.domain(),
            smaller_homspace.codomain(),
            larger_homspace.codomain(),
        )
    };
    if larger_side.len() != smaller_side.len() + 1
        || larger_side.legs().get(local_position)
            != Some(&SectorLeg::new([(unit, 1)], insertion.dual()))
        || !slice_matches_without(larger_side.legs(), smaller_side.legs(), local_position)
        || smaller_other != larger_other
        || smaller_structure.block_count() != larger_structure.block_count()
        || smaller_structure.required_len()? != larger_structure.required_len()?
    {
        return Err(CoreError::UnitLayoutCorrespondence);
    }
    for index in 0..smaller_structure.block_count() {
        let small = smaller_structure.block(index)?;
        let large = larger_structure.block(index)?;
        if small.offset() != large.offset()
            || small.element_count()? != large.element_count()?
            || !unit_block_layout_matches(&small, &large, insertion.position())
        {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let (BlockKey::FusionTree(small_key), BlockKey::FusionTree(large_key)) =
            (small.key(), large.key())
        else {
            return Err(CoreError::UnitLayoutCorrespondence);
        };
        let (small_tree, large_tree) = if is_codomain {
            (small_key.codomain_tree(), large_key.codomain_tree())
        } else {
            (small_key.domain_tree(), large_key.domain_tree())
        };
        let (small_other, large_other) = if is_codomain {
            (small_key.domain_tree(), large_key.domain_tree())
        } else {
            (small_key.codomain_tree(), large_key.codomain_tree())
        };
        if small_other != large_other
            || !unit_tree_matches(
                unit,
                small_tree,
                large_tree,
                local_position,
                insertion.dual(),
            )
        {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
    }
    Ok(())
}

fn unit_block_layout_matches(small: &BlockRef<'_>, large: &BlockRef<'_>, axis: usize) -> bool {
    large.shape().get(axis) == Some(&1)
        && slice_matches_without(large.shape(), small.shape(), axis)
        && slice_matches_without(large.strides(), small.strides(), axis)
}

fn slice_matches_without<T: Eq>(larger: &[T], smaller: &[T], index: usize) -> bool {
    index < larger.len()
        && larger.len() == smaller.len() + 1
        && larger[..index] == smaller[..index]
        && larger[index + 1..] == smaller[index..]
}

fn unit_tree_matches(
    unit: SectorId,
    small: &FusionTreeKey,
    large: &FusionTreeKey,
    position: usize,
    dual: bool,
) -> bool {
    let rank = small.uncoupled().len();
    if position > rank
        || large.coupled() != small.coupled()
        || large.uncoupled().get(position) != Some(&unit)
        || large.is_dual().get(position) != Some(&dual)
        || !slice_matches_without(large.uncoupled(), small.uncoupled(), position)
        || !slice_matches_without(large.is_dual(), small.is_dual(), position)
    {
        return false;
    }
    if rank == 0 {
        return large.innerlines().is_empty() && large.vertices().is_empty();
    }
    let vertex = position.saturating_sub(1);
    if large.vertices().get(vertex) != Some(&MultiplicityIndex::ONE)
        || !slice_matches_without(large.vertices(), small.vertices(), vertex)
    {
        return false;
    }
    if rank == 1 {
        return large.innerlines().is_empty();
    }
    let innerline = position.saturating_sub(1).min(rank - 2);
    let expected = match position {
        0 | 1 => small.uncoupled()[0],
        p if p < rank => small.innerlines()[p - 2],
        _ => small.coupled(),
    };
    large.innerlines().get(innerline) == Some(&expected)
        && slice_matches_without(large.innerlines(), small.innerlines(), innerline)
}
