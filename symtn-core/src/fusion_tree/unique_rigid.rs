use super::*;

/// Test-only oracle helper; production swaps read lines through `ArtinSite`.
#[cfg(test)]
pub(super) fn inner_extended_sector<T>(tree: &T, index: usize) -> Result<SectorId, CoreError>
where
    T: MultiplicityFreeTreeData + ?Sized,
{
    let rank = tree.uncoupled().len();
    if index == 0 {
        return tree
            .uncoupled()
            .first()
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "inner-extended tree requires at least one uncoupled sector",
            });
    }
    if index + 1 == rank {
        return Ok(tree.coupled());
    }
    tree.innerlines()
        .get(index - 1)
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "inner-extended tree is missing an innerline",
        })
}

pub(super) fn only_fusion_channel<R>(
    rule: &R,
    left: SectorId,
    right: SectorId,
) -> Result<SectorId, CoreError>
where
    R: FusionRule,
{
    let channels = rule.fusion_channels(left, right);
    match channels.as_slice() {
        [sector] => Ok(*sector),
        _ => Err(CoreError::FusionChannelCount {
            left,
            right,
            count: channels.len(),
        }),
    }
}

#[cfg(test)]
fn unique_rigid_bend_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    bend: Bend,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let mut scratch = UniqueTreePairScratch::default();
    scratch.load(tree_pair);
    check_multiplicity_free_projection(rule, &scratch)?;
    let coefficient = unique_rigid_bend_in_place(rule, &mut scratch, bend)?;
    scratch.codomain.reset_multiplicity_free_vertices();
    scratch.domain.reset_multiplicity_free_vertices();
    Ok((scratch.materialize(), coefficient))
}

/// The gate `project_multiplicity_free_tree_pair` applies before any
/// multiplicity-free move: the style, then every vertex label one.
fn check_multiplicity_free_projection<R>(
    rule: &R,
    scratch: &UniqueTreePairScratch,
) -> Result<(), CoreError>
where
    R: FusionRule,
{
    validate_multiplicity_free_execution_style(rule)?;
    scratch.codomain.check_multiplicity_free_vertices()?;
    scratch.domain.check_multiplicity_free_vertices()
}

/// One Unique bend of the scratch pair, in place. `bendleft` is `bendright`
/// through the swapped pair, conjugated (`duality_manipulations.jl:140-148`).
fn unique_rigid_bend_in_place<R>(
    rule: &R,
    scratch: &mut UniqueTreePairScratch,
    bend: Bend,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let UniqueTreePairScratch {
        codomain, domain, ..
    } = scratch;
    match bend {
        Bend::Right => unique_rigid_bendright_in_place(rule, codomain, domain),
        Bend::Left => Ok(unique_rigid_bendright_in_place(rule, domain, codomain)?.conj()),
    }
}

/// TensorKit `_bendright_treepair` (`duality_manipulations.jl:33-54`) on a
/// tree pair edited in place: [`MultiplicityFreeBend`]'s site, lines and
/// `UniqueK` coefficient, then its frame and local update.
///
/// Why not through [`MultiplicityFreeBend`]: its output frame owns the moved
/// externals (two allocations per side and bend), which a
/// one-tree-at-a-time Unique path would pay for every tree and bend.
fn unique_rigid_bendright_in_place<R>(
    rule: &R,
    codomain: &mut UnhashedFusionTree,
    domain: &mut UnhashedFusionTree,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let site = BendSite::new(codomain.uncoupled(), codomain.is_dual(), domain.rank())?;
    let (lines, coefficient) = bend_surgery(&UniqueK(rule), &site, &*codomain, &*domain)?;
    codomain.shrink_after_bend(lines.left_coupled);
    domain.grow_after_bend(
        rule.dual(site.bent_sector),
        !lines.bent_is_dual,
        lines.left_coupled,
        (site.domain_rank > 1).then_some(lines.coupled),
    );
    Ok(coefficient)
}

#[cfg(test)]
fn unique_rigid_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    // Why not `fold_surgery`: a unique fold has exactly one forward and one
    // inverse move, and threading that single term through the surgery's
    // move iterators measurably slows cold unique transposes.
    // These are the surgery's steps, unrolled, with the multiplicity-free
    // coefficient kernel.
    let kernel = SimpleK(rule);
    let codomain = tree_pair.codomain_tree();
    let site = FoldSite::new(codomain.uncoupled(), codomain.is_dual())?;
    let (fold, dual_first) = kernel.fold_begin(&site)?;
    let unique = UniqueK(rule);
    let (codomain_prime, coeff1) = multi_fmove_surgery(&unique, codomain)?;
    let tail_coupled = codomain_prime.coupled();
    let factors = kernel.fold_factors(&fold, tail_coupled, codomain.coupled())?;
    let (domain_prime, coeff2) = multi_fmove_inv_surgery(
        &unique,
        &(dual_first, !site.first_is_dual),
        tail_coupled,
        tree_pair.domain_tree(),
    )?;
    let coefficient = kernel.fold_coefficient(&fold, &factors, &coeff1, &coeff2)?;
    Ok((
        FusionTreePairKey::pair(codomain_prime, domain_prime),
        coefficient,
    ))
}

/// Key-form oracle of [`unique_rigid_cycle_in_place`].
#[cfg(test)]
pub(crate) fn unique_rigid_cycle_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    direction: PreparedCycleDirection,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    cycle(
        tree_pair,
        direction,
        |key, bend| unique_rigid_bend_tree_pair(rule, key, bend),
        |key| unique_rigid_foldright_tree_pair(rule, key),
    )
}

#[cfg(test)]
pub(crate) fn unique_rigid_repartition_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
) -> Result<(FusionTreePairKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let mut scratch = UniqueTreePairScratch::default();
    scratch.load(tree_pair);
    let coefficient = unique_rigid_repartition_in_place(rule, &mut scratch, target_codomain_rank)?;
    Ok((scratch.materialize(), coefficient))
}

/// One bend of a cycle step, as [`unique_rigid_bend_tree_pair`] applies it:
/// the projection gate, the bend, then every vertex label one.
fn unique_rigid_cycle_bend_in_place<R>(
    rule: &R,
    scratch: &mut UniqueTreePairScratch,
    bend: Bend,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    check_multiplicity_free_projection(rule, scratch)?;
    let coefficient = unique_rigid_bend_in_place(rule, scratch, bend)?;
    scratch.codomain.reset_multiplicity_free_vertices();
    scratch.domain.reset_multiplicity_free_vertices();
    Ok(coefficient)
}

/// [`unique_rigid_foldright_tree_pair`] on the scratch pair, in place, with
/// the same steps and coefficient association: `multi_Fmove` of the codomain
/// into the spare tree, then `multi_Fmove_inv` of the domain into it.
fn unique_rigid_foldright_in_place<R>(
    rule: &R,
    scratch: &mut UniqueTreePairScratch,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let kernel = SimpleK(rule);
    let UniqueTreePairScratch {
        codomain,
        domain,
        spare,
    } = scratch;
    let site = FoldSite::new(codomain.uncoupled(), codomain.is_dual())?;
    let (fold, dual_first) = kernel.fold_begin(&site)?;
    let coupled = codomain.coupled();
    // `UniqueK::tails`: the tail couples to the one channel of `ā ⊗ c`.
    let recoupled = only_fusion_channel(rule, rule.dual(site.first), coupled)?;
    spare.rebuild_unique_tree(
        rule,
        codomain.uncoupled()[1..].iter().copied(),
        recoupled,
        codomain.is_dual()[1..].iter().copied(),
    )?;
    let coeff1 = multiplicity_free_multi_associator_from_parts(
        rule,
        codomain.uncoupled(),
        codomain.is_dual(),
        &*codomain,
        spare.uncoupled(),
        spare.is_dual(),
        &*spare,
    )?
    .ok_or(CoreError::MalformedFusionTree {
        message: "unique multi_Fmove destination does not match the source tail",
    })?;
    let tail_coupled = spare.coupled();
    let factors = kernel.fold_factors(&fold, tail_coupled, coupled)?;
    std::mem::swap(codomain, spare);
    // `UniqueK::lifts`: `ā` joins the domain, which couples to `tail_coupled`.
    spare.rebuild_unique_tree(
        rule,
        std::iter::once(dual_first).chain(domain.uncoupled().iter().copied()),
        tail_coupled,
        std::iter::once(!site.first_is_dual).chain(domain.is_dual().iter().copied()),
    )?;
    let coeff2 = multiplicity_free_multi_associator_from_parts(
        rule,
        spare.uncoupled(),
        spare.is_dual(),
        &*spare,
        domain.uncoupled(),
        domain.is_dual(),
        &*domain,
    )?
    .ok_or(CoreError::MalformedFusionTree {
        message: "unique inverse multi_Fmove destination does not match the source tail",
    })?
    .conj();
    std::mem::swap(domain, spare);
    kernel.fold_coefficient(&fold, &factors, &coeff1, &coeff2)
}

/// One cyclic rotation of the scratch pair, in place: [`cycle`]'s step
/// order and `first · second` coefficient, `foldleft` as `foldright`
/// through the swapped pair, conjugated (`duality_manipulations.jl:401, 431`).
pub(crate) fn unique_rigid_cycle_in_place<R>(
    rule: &R,
    scratch: &mut UniqueTreePairScratch,
    direction: PreparedCycleDirection,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let foldleft = |scratch: &mut UniqueTreePairScratch| {
        std::mem::swap(&mut scratch.codomain, &mut scratch.domain);
        let coefficient = unique_rigid_foldright_in_place(rule, scratch);
        std::mem::swap(&mut scratch.codomain, &mut scratch.domain);
        coefficient.map(|coefficient| coefficient.conj())
    };
    match direction {
        PreparedCycleDirection::Clockwise => {
            if scratch.codomain.rank() == 0 {
                let first = unique_rigid_cycle_bend_in_place(rule, scratch, Bend::Left)?;
                Ok(first * unique_rigid_foldright_in_place(rule, scratch)?)
            } else {
                let first = unique_rigid_foldright_in_place(rule, scratch)?;
                Ok(first * unique_rigid_cycle_bend_in_place(rule, scratch, Bend::Left)?)
            }
        }
        PreparedCycleDirection::Anticlockwise => {
            if scratch.domain.rank() == 0 {
                let first = unique_rigid_cycle_bend_in_place(rule, scratch, Bend::Right)?;
                Ok(first * foldleft(scratch)?)
            } else {
                let first = foldleft(scratch)?;
                Ok(first * unique_rigid_cycle_bend_in_place(rule, scratch, Bend::Right)?)
            }
        }
    }
}

/// TensorKit `repartition` (`duality_manipulations.jl:460-505`) of a Unique
/// tree pair, in place: bend until the codomain has `target_codomain_rank`
/// legs. The trees end with every vertex label one, as a materialized
/// multiplicity-free frame writes them.
pub(crate) fn unique_rigid_repartition_in_place<R>(
    rule: &R,
    scratch: &mut UniqueTreePairScratch,
    target_codomain_rank: usize,
) -> Result<R::Scalar, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: rule.fusion_style(),
        });
    }
    let codomain_rank = scratch.codomain.rank();
    let total_rank = codomain_rank + scratch.domain.rank();
    if target_codomain_rank > total_rank {
        return Err(CoreError::DimensionMismatch {
            expected: total_rank,
            actual: target_codomain_rank,
        });
    }
    check_multiplicity_free_projection(rule, scratch)?;
    let coefficient = repartition_loop(
        R::Scalar::one(),
        codomain_rank,
        target_codomain_rank,
        |coefficient, bend| {
            Ok::<_, CoreError>(coefficient * unique_rigid_bend_in_place(rule, scratch, bend)?)
        },
    )?;
    scratch.codomain.reset_multiplicity_free_vertices();
    scratch.domain.reset_multiplicity_free_vertices();
    Ok(coefficient)
}

/// The forward move is TensorKit `multi_Fmove`'s `UniqueFusion` branch
/// (`basic_manipulations.jl:209-213`) at every rank: the tail couples to the
/// one channel of `ā ⊗ c`. The inverse is a symtn deviation: TensorKit
/// `multi_Fmove_inv` has no `UniqueFusion` branch, but with one channel per
/// fusion step its enumeration yields exactly one lifted tree, which is the
/// unique standard tree built here.
impl<R: MultiplicityFreeRigidSymbols> MultiFKernel for UniqueK<'_, R> {
    type Moves = (FusionTreeKey, R::Scalar);
    type Lift = (SectorId, bool);

    fn tails<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, CoreError> {
        let rule = self.0;
        let recoupled = only_fusion_channel(rule, rule.dual(tree.uncoupled()[0]), tree.coupled())?;
        let destination = unique_standard_fusion_tree(
            rule,
            &tree.uncoupled()[1..],
            recoupled,
            &tree.is_dual()[1..],
        )?;
        match multiplicity_free_multi_associator_from_parts(
            rule,
            tree.uncoupled(),
            tree.is_dual(),
            tree,
            destination.uncoupled(),
            destination.is_dual(),
            &destination,
        )? {
            Some(coefficient) => Ok((destination, coefficient)),
            None => Err(CoreError::MalformedFusionTree {
                message: "unique multi_Fmove destination does not match the source tail",
            }),
        }
    }

    fn lift_leading(&(leading, _): &(SectorId, bool)) -> SectorId {
        leading
    }

    // Why not check `c ∈ a ⊗ b` here as the other kernels do: for a
    // well-formed source tree (its coupled sector is the fusion of its
    // uncoupled legs), the unique standard tree below fails with a
    // channel-count error on exactly the inputs the admission rejects, and
    // the admission would add a fusion query per fold. Malformed source
    // trees are rejected by upstream validation before reaching this move.
    fn admit_lift(&self, _: SectorId, _: SectorId, _: SectorId) -> Result<(), CoreError> {
        Ok(())
    }

    fn lifts<T: FramedTree + ?Sized>(
        &self,
        &(leading, leading_is_dual): &(SectorId, bool),
        coupled: SectorId,
        tree: &T,
    ) -> Result<Self::Moves, CoreError> {
        let rule = self.0;
        let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
        uncoupled.push(leading);
        uncoupled.extend_from_slice(tree.uncoupled());
        let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
        is_dual.push(leading_is_dual);
        is_dual.extend_from_slice(tree.is_dual());
        let destination = unique_standard_fusion_tree(rule, &uncoupled, coupled, &is_dual)?;
        match multiplicity_free_multi_associator_from_parts(
            rule,
            destination.uncoupled(),
            destination.is_dual(),
            &destination,
            tree.uncoupled(),
            tree.is_dual(),
            tree,
        )? {
            Some(coefficient) => Ok((destination, coefficient.conj())),
            None => Err(CoreError::MalformedFusionTree {
                message: "unique inverse multi_Fmove destination does not match the source tail",
            }),
        }
    }
}

fn unique_standard_fusion_tree<R>(
    rule: &R,
    uncoupled: &[SectorId],
    coupled: SectorId,
    is_dual: &[bool],
) -> Result<FusionTreeKey, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    if uncoupled.len() != is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    let effective = uncoupled.to_vec();
    let trees = collect_fusion_trees_for_coupled(rule, uncoupled, is_dual, &effective, coupled);
    match trees.as_slice() {
        [tree] => Ok(tree.clone()),
        _ => Err(CoreError::FusionChannelCount {
            left: coupled,
            right: coupled,
            count: trees.len(),
        }),
    }
}
