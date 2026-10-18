use super::*;

pub(crate) fn execute_unique_tree_braid<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    artin_steps: &[PreparedArtinStep],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let mut braided = UnhashedFusionTree::from(tree.clone());
    let coefficient = unique_tree_braid_in_place(
        rule,
        &mut braided,
        &mut UnhashedFusionTree::default(),
        permutation,
        artin_steps,
    )?;
    Ok((braided.freeze(), coefficient))
}

/// [`execute_unique_tree_braid`] of a tree edited in place; `spare` is the
/// direct path's rebuild buffer.
pub(crate) fn unique_tree_braid_in_place<R>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    spare: &mut UnhashedFusionTree,
    permutation: &[usize],
    artin_steps: &[PreparedArtinStep],
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.rank();
    if permutation.len() != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        });
    }
    unique_braid_in_place(
        rule,
        tree,
        spare,
        |position| permutation[position],
        artin_steps.iter().copied(),
    )
}

/// The borrowed-schedule form of [`unique_tree_braid_in_place`].
pub(crate) fn unique_tree_braid_borrowed_in_place<R>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    spare: &mut UnhashedFusionTree,
    braid: &UniqueBorrowedTreePairBraid<'_>,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = tree.rank();
    if braid.source_codomain_rank + braid.source_domain_rank != rank {
        return Err(CoreError::InvalidPermutation {
            permutation: (0..braid.source_codomain_rank + braid.source_domain_rank)
                .map(|position| braid.permutation_at(position))
                .collect(),
            rank,
        });
    }
    unique_braid_in_place(
        rule,
        tree,
        spare,
        |position| braid.permutation_at(position),
        braid.artin_steps(),
    )
}

#[cfg(test)]
pub(crate) fn execute_unique_tree_braid_borrowed<R>(
    rule: &R,
    tree: &FusionTreeKey,
    braid: &UniqueBorrowedTreePairBraid<'_>,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let mut braided = UnhashedFusionTree::from(tree.clone());
    let coefficient = unique_tree_braid_borrowed_in_place(
        rule,
        &mut braided,
        &mut UnhashedFusionTree::default(),
        braid,
    )?;
    Ok((braided.freeze(), coefficient))
}

/// The direct braid when its invariants hold, otherwise the Artin steps.
fn unique_braid_in_place<R, I>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    spare: &mut UnhashedFusionTree,
    permutation: impl Fn(usize) -> usize,
    steps: I,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
    I: IntoIterator<Item = PreparedArtinStep>,
{
    if let Some(coefficient) = unique_direct_braid(rule, tree, permutation, spare)? {
        core::mem::swap(tree, spare);
        return Ok(coefficient);
    }
    unique_tree_braid_steps_in_place(rule, tree, steps)
}

/// Symmetric-braiding unique-fusion braid in one step: TensorKit
/// `braid(f::FusionTree, p, levels)` `SymmetricBraiding` branch
/// (`braiding_manipulations.jl:217-234`) structure: the product over every
/// inverted pair of `R(a, b, a ⊗ b)` with `a` before `b` in the source (the
/// Artin-composition value), then the canonical tree of the permuted legs,
/// written to `out`. `None` when the source does not satisfy the direct-path
/// invariants.
///
/// `permutation(position)` is the source axis at output `position`, for
/// every position below the tree rank (validated by the caller).
pub(crate) fn unique_direct_braid<R>(
    rule: &R,
    tree: &UnhashedFusionTree,
    permutation: impl Fn(usize) -> usize,
    out: &mut UnhashedFusionTree,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    // Why not rebuild every Unique key directly: symtn's public checked
    // constructor still accepts noncanonical innerline data, whose given
    // Artin behavior must not be silently normalized into a different key.
    if !(rule.braiding_style().is_symmetric()
        && rule.has_trivial_associator_gauge()
        && is_unique_direct_braid_source(rule, tree.view()))
    {
        return Ok(None);
    }
    let rank = tree.rank();
    let uncoupled = tree.uncoupled();
    let mut coefficient = R::Scalar::one();
    for right_position in 0..rank {
        for left_position in 0..right_position {
            let left_axis = permutation(left_position);
            let right_axis = permutation(right_position);
            if left_axis > right_axis {
                // The crossing braids the source pair `first ⊗ second`
                // (source order) and acts by `R(first, second)`, exactly as
                // the Artin path does. Why not TensorKit's symmetric
                // shortcut order `R(second, first)`: it differs from its
                // own `artin_braid` whenever a trivial-gauge symmetric R is
                // not symmetric in its arguments (alternating bicharacters).
                let first = uncoupled[right_axis];
                let second = uncoupled[left_axis];
                // TensorKit treats a unit crossing as structural identity.
                // Why not ask the provider for R(unit, a): providers are
                // permitted to omit identity symbols and the Artin path
                // already skips them.
                if first == rule.vacuum() || second == rule.vacuum() {
                    continue;
                }
                let coupled = only_fusion_channel(rule, first, second)?;
                coefficient = coefficient * rule.r_symbol_scalar(first, second, coupled);
            }
        }
    }
    let is_dual = tree.is_dual();
    out.rebuild_unique_standard(
        rule,
        (0..rank).map(|position| uncoupled[permutation(position)]),
        tree.coupled(),
        (0..rank).map(|position| is_dual[permutation(position)]),
    )?;
    Ok(Some(coefficient))
}

pub(crate) fn execute_unique_tree_braid_steps<R, I>(
    rule: &R,
    tree: &FusionTreeKey,
    steps: I,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
    I: IntoIterator<Item = PreparedArtinStep>,
{
    let mut braided = UnhashedFusionTree::from(tree.clone());
    let coefficient = unique_tree_braid_steps_in_place(rule, &mut braided, steps)?;
    Ok((braided.freeze(), coefficient))
}

/// Apply an Artin schedule to a Unique tree in place.
fn unique_tree_braid_steps_in_place<R, I>(
    rule: &R,
    tree: &mut UnhashedFusionTree,
    steps: I,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
    I: IntoIterator<Item = PreparedArtinStep>,
{
    // Why not freeze a key after every step: this state is private to one
    // execution, while only the final categorical identity can escape.
    run_artin_steps(R::Scalar::one(), steps, |coefficient, step| {
        let step_coefficient =
            apply_unique_artin_braid_at_with_inverse(rule, tree, step.index, step.inverse)?;
        Ok::<_, CoreError>(coefficient * step_coefficient)
    })
}

pub(crate) fn is_unique_direct_braid_source<R>(rule: &R, tree: FusionTreeRef<'_>) -> bool
where
    R: MultiplicityFreeFusionRule,
{
    let rank = tree.uncoupled().len();
    if rank < 2
        || validate_fusion_tree_key_shape(tree).is_err()
        || tree
            .vertices()
            .iter()
            .any(|vertex| *vertex != MultiplicityIndex::ONE)
    {
        return false;
    }
    let coupled = tree.coupled();

    let mut running = tree.uncoupled()[0];
    for (offset, &right) in tree.uncoupled()[1..].iter().enumerate() {
        let Ok(next) = only_fusion_channel(rule, running, right) else {
            return false;
        };
        let is_last = offset + 2 == rank;
        if is_last {
            if next != coupled {
                return false;
            }
        } else if tree.innerlines().get(offset).copied() != Some(next) {
            return false;
        }
        running = next;
    }
    true
}

#[cfg(test)]
pub(crate) fn immutable_unique_artin_braid_at_with_inverse_oracle<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }

    let rank = tree.uncoupled().len();
    if index + 1 >= rank {
        return Err(CoreError::InvalidBraidIndex { index, rank });
    }

    let left = tree.uncoupled()[index];
    let right = tree.uncoupled()[index + 1];
    let mut uncoupled = tree.uncoupled().to_vec();
    uncoupled.swap(index, index + 1);
    let mut is_dual = tree.is_dual().to_vec();
    is_dual.swap(index, index + 1);
    let mut innerlines = tree.innerlines().to_vec();
    let mut vertices = tree.vertices().to_vec();

    if left == rule.vacuum() || right == rule.vacuum() {
        if index > 0 {
            let inner_source = if left == rule.vacuum() {
                inner_extended_sector(tree, index + 1)?
            } else {
                inner_extended_sector(tree, index - 1)?
            };
            *innerlines
                .get_mut(index - 1)
                .ok_or(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires an innerline",
                })? = inner_source;
            if vertices.len() <= index {
                return Err(CoreError::MalformedFusionTree {
                    message: "unit braid past the first adjacent pair requires adjacent vertices",
                });
            }
            vertices.swap(index - 1, index);
        }

        let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
        return Ok((braided, R::Scalar::one()));
    }

    if !rule.braiding_style().has_braiding() {
        return Err(CoreError::UnsupportedSectorBraid {
            left,
            right,
            style: rule.braiding_style(),
        });
    }

    if index == 0 {
        let coupled = if rank > 2 {
            tree.innerlines()
                .first()
                .copied()
                .ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a rank > 2 tree requires the first innerline",
                })?
        } else {
            tree.coupled()
        };

        let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
        let coefficient = if inverse {
            (rule.r_symbol_scalar(right, left, coupled)).conj()
        } else {
            rule.r_symbol_scalar(left, right, coupled)
        };
        return Ok((braided, coefficient));
    }

    let a = inner_extended_sector(tree, index - 1)?;
    let b = left;
    let c = inner_extended_sector(tree, index)?;
    let d = right;
    let e = inner_extended_sector(tree, index + 1)?;
    let c_prime = only_fusion_channel(rule, a, d)?;
    *innerlines
        .get_mut(index - 1)
        .ok_or(CoreError::MalformedFusionTree {
            message: "non-first braid requires an innerline to update",
        })? = c_prime;
    let braided = FusionTreeKey::new(uncoupled, tree.coupled(), is_dual, innerlines, vertices);
    let f_symbol = rule.f_symbol_scalar(d, a, b, e, c_prime, c);
    let coefficient = if inverse {
        let left = rule.r_symbol_scalar(d, c, e);
        let right = rule.r_symbol_scalar(d, a, c_prime);
        (left * f_symbol).conj() * right
    } else {
        let left = rule.r_symbol_scalar(c, d, e);
        let right = rule.r_symbol_scalar(a, d, c_prime);
        left * (f_symbol * right).conj()
    };
    Ok((braided, coefficient))
}
