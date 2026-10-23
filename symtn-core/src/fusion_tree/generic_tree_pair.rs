use super::*;

/// Generic-fusion `foldright`: bend the first codomain vertex `a ⊗ b ← c` to a
/// domain vertex `b ← dual(a) ⊗ c`. Verbatim mirror of TensorKit `foldright`
/// GenericFusion branch (`duality_manipulations.jl:238-289`), especially the
/// coefficient-vector × A × coefficient-vector contraction at `:277-284`:
///   `coeff₀ · (coeff₂' · (transpose(A) · coeff₁))`.
/// Structural twin of `multiplicity_free_foldright_tree_pair`, with the scalar
/// `coeff₁ · A · conj(coeff₂)` promoted to the vector–matrix–vector contraction
/// through the A-move matrix (which connects the two topmost `λ` vertices).
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let codomain = tree_pair.codomain_tree();
    let codomain_rank = codomain.uncoupled().len();
    if codomain_rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_foldright_tree_pair_validated(validated)
}

#[cfg(test)]
fn generic_foldright_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_foldright_tree_pair_unchecked(rule, tree_pair.key)
}

#[cfg(test)]
fn generic_foldright_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    generic_foldright_tree_pair_result(&InfallibleGenericFR(rule), tree_pair)
        .map_err(map_infallible_generic_symbol_error)
}

/// Generic-fusion `foldleft` = swap + conjugate of `foldright`, verbatim mirror
/// of TensorKit `foldleft((f₁,f₂))` (`duality_manipulations.jl:315-319`).
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_foldleft_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if tree_pair.domain_tree().uncoupled().is_empty() {
        return Err(CoreError::MalformedFusionTree {
            message: "foldleft requires at least one domain leg",
        });
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_foldleft_tree_pair_validated(validated)
}

#[cfg(test)]
fn generic_foldleft_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_foldleft_tree_pair_unchecked(rule, tree_pair.key)
}

#[cfg(test)]
fn generic_foldleft_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    left_move_by_swap(tree_pair, |key| {
        generic_foldright_tree_pair_unchecked(rule, key)
    })
}

/// Generic-fusion `cycleclockwise` = foldright ∘ bendleft (or the reverse order
/// when the codomain is empty), composing coefficient matrices. Verbatim mirror
/// of TensorKit `cycleclockwise` (`duality_manipulations.jl:401-410`) and
/// structural twin of `multiplicity_free_cycle_clockwise_tree_pair`.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_cycle_clockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_cycle_clockwise_tree_pair_validated(validated)
}

#[cfg(test)]
fn generic_cycle_clockwise_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_cycle_tree_pair_unchecked(rule, tree_pair.key, PreparedCycleDirection::Clockwise)
}

#[cfg(any(test, feature = "testing"))]
pub(super) fn generic_cycle_tree_pair_unchecked<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    direction: PreparedCycleDirection,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    generic_cycle_tree_pair_result(&InfallibleGenericFR(rule), tree_pair, direction)
        .map_err(map_infallible_generic_symbol_error)
}

/// Generic-fusion `cycleanticlockwise` = foldleft ∘ bendright (or the reverse
/// order when the domain is empty). Verbatim mirror of TensorKit
/// `cycleanticlockwise` (`duality_manipulations.jl:431-440`) and structural
/// twin of `multiplicity_free_cycle_anticlockwise_tree_pair`.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(test)]
pub(crate) fn generic_cycle_anticlockwise_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_cycle_anticlockwise_tree_pair_validated(validated)
}

#[cfg(test)]
fn generic_cycle_anticlockwise_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_cycle_tree_pair_unchecked(rule, tree_pair.key, PreparedCycleDirection::Anticlockwise)
}

pub(crate) fn generic_foldright_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    let mut terms = FusionTermAccumulator::new();
    fold_surgery(&GenericK(rule), tree_pair, |key, coefficient| {
        terms.push(key, coefficient)
    })?;
    Ok(terms.into_vec())
}

fn generic_cycle_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    direction: PreparedCycleDirection,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    cycle(
        tree_pair,
        direction,
        |key, bend| match bend {
            Bend::Left => generic_bendleft_tree_pair_result(rule, key),
            Bend::Right => generic_bendright_tree_pair_result(rule, key),
        },
        |key| generic_foldright_tree_pair_result(rule, key),
    )
}

/// Generic-fusion `braid` on a full tree pair: bend everything into the codomain,
/// braid there, bend back — the TensorKit `braid`/`fsbraid` decomposition.
/// Structural twin of [`multiplicity_free_braid_tree_pair`] (:829): the only
/// difference is the primitive family (`generic_repartition_tree_pair` /
/// `generic_braid_tree`) and the `one` seed;
/// no new recoupling formula is introduced.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_braid_tree_pair_proven(
        validated,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )
}

#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_braid_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if tree_pair.rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: tree_pair.rule.fusion_style(),
        });
    }
    let codomain_rank = tree_pair.key.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.key.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        });
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        });
    }
    let GenericBraidSchedule {
        permutation,
        identity,
        steps,
    } = generic_braid_schedule(
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?;
    generic_braid_tree_pair_validated(
        tree_pair,
        codomain_permutation.len(),
        &permutation,
        &steps,
        identity,
    )
}

#[cfg(any(test, feature = "testing"))]
fn generic_braid_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
    identity: bool,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGeneric::new(tree_pair.rule);
    generic_braid_tree_pair_result(
        &checked,
        tree_pair.key,
        target_codomain_rank,
        permutation,
        steps,
        identity,
    )
    .map_err(map_infallible_generic_symbol_error)
}

fn generic_braid_tree_pair_result<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
    identity: bool,
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    if identity {
        return Ok(vec![(tree_pair.clone(), C::Scalar::one())]);
    }
    let all_codomain = generic_repartition_tree_pair_result(rule, tree_pair, permutation.len())?;
    braid_via_codomain(
        all_codomain,
        |key| {
            generic_braid_tree_result(rule, key.codomain_tree(), permutation, steps)
                .map(|terms| with_domain(key, terms))
        },
        // Why not the whole-list `repartition_loop` that the multiplicity-free
        // and block schedules use: repartitioning each braided term on its own
        // keeps the established Generic association bit-for-bit (the list form
        // moves the last ulp of some coefficients).
        |braided| {
            compose_terms(braided, |key| {
                generic_repartition_tree_pair_result(rule, key, target_codomain_rank)
            })
        },
    )
}

/// Checked Generic-fusion braid on a full tree pair.
#[expect(
    clippy::type_complexity,
    reason = "the public checked API exposes destination tree-pair coefficient rows directly"
)]
pub fn generic_braid_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, C::Scalar)>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        }
        .into());
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        }
        .into());
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;
    let GenericBraidSchedule {
        permutation,
        identity,
        steps,
    } = generic_braid_schedule(
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?;
    generic_braid_tree_pair_result(
        rule,
        tree_pair,
        codomain_permutation.len(),
        &permutation,
        &steps,
        identity,
    )
}

/// Generic-fusion `permute` = [`generic_braid_tree_pair`] with the identity
/// level order (symmetric braiding only). Structural twin of
/// [`multiplicity_free_permute_tree_pair`] (:886).
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_permute_tree_pair_proven(validated, codomain_permutation, domain_permutation)
}

/// Checked Generic-fusion permutation on a full tree pair.
#[expect(
    clippy::type_complexity,
    reason = "the public checked API exposes destination tree-pair coefficient rows directly"
)]
pub fn generic_permute_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, C::Scalar)>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        }
        .into());
    }
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    generic_braid_tree_pair_checked(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_permute_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if !tree_pair.rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: tree_pair.rule.braiding_style(),
        });
    }
    if tree_pair.rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: tree_pair.rule.fusion_style(),
        });
    }
    let codomain_rank = tree_pair.key.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.key.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    let GenericBraidSchedule {
        permutation,
        identity,
        steps,
    } = generic_braid_schedule(
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )?;
    generic_braid_tree_pair_validated(
        tree_pair,
        codomain_permutation.len(),
        &permutation,
        &steps,
        identity,
    )
}

/// Generic-fusion `transpose` (planar cyclic permutation): bend into the target
/// partition, then cycle the coupled tree into place via fold/bend. Structural
/// twin of `multiplicity_free_transpose_tree_pair` (:916); braid-free, so it
/// runs on planar (non-symmetric) Generic rules too.
/// `tree_pair` follows [`FusionTreePairKey::validate_for_rule`]'s
/// provider-domain precondition.
#[cfg(any(test, feature = "testing"))]
pub fn generic_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        });
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let validated = validate_fusion_tree_pair_for_rule(rule, tree_pair)?;
    generic_transpose_tree_pair_proven(validated, codomain_permutation, domain_permutation)
}

#[cfg(any(test, feature = "testing"))]
pub(crate) fn generic_transpose_tree_pair_proven<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if tree_pair.rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: tree_pair.rule.fusion_style(),
        });
    }
    let codomain_rank = tree_pair.key.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.key.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        });
    }
    let position = permutation.iter().position(|&axis| axis == 0);
    generic_transpose_tree_pair_validated(
        tree_pair,
        codomain_permutation.len(),
        codomain_rank + domain_rank,
        position,
    )
}

#[cfg(any(test, feature = "testing"))]
fn generic_transpose_tree_pair_validated<R>(
    tree_pair: ValidatedFusionTreePair<'_, R>,
    target_codomain_rank: usize,
    total_rank: usize,
    position: Option<usize>,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree_pair.rule;
    generic_transpose_tree_pair_with(
        tree_pair.key,
        target_codomain_rank,
        total_rank,
        position,
        |key, rank| generic_repartition_tree_pair_unchecked(rule, key, rank),
        |key, direction| generic_cycle_tree_pair_unchecked(rule, key, direction),
    )
}

/// Checked Generic-fusion planar cyclic transpose.
///
/// Axis syntax and the complete source tree are validated before rigidity or
/// F-symbol queries; provider and symbol-shape failures remain typed.
pub fn generic_transpose_tree_pair_checked<C>(
    rule: &C,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<GenericTreePairTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let codomain_rank = tree_pair.codomain_tree().uncoupled().len();
    let domain_rank = tree_pair.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        }
        .into());
    }
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        }
        .into());
    }
    validate_generic_fusion_tree_pair_checked(rule, tree_pair)
        .map_err(map_checked_generic_structure_error)?;

    generic_transpose_tree_pair_with(
        tree_pair,
        codomain_permutation.len(),
        codomain_rank + domain_rank,
        permutation.iter().position(|&axis| axis == 0),
        |key, rank| generic_repartition_tree_pair_result(rule, key, rank),
        |key, direction| generic_cycle_tree_pair_result(rule, key, direction),
    )
}

fn generic_transpose_tree_pair_with<S, E, P, C>(
    tree_pair: &FusionTreePairKey,
    target_codomain_rank: usize,
    total_rank: usize,
    position: Option<usize>,
    mut repartition: P,
    cycle: C,
) -> Result<Vec<(FusionTreePairKey, S)>, E>
where
    S: CategoricalScalar,
    E: From<CoreError>,
    P: FnMut(&FusionTreePairKey, usize) -> Result<Vec<(FusionTreePairKey, S)>, E>,
    C: FnMut(&FusionTreePairKey, PreparedCycleDirection) -> Result<Vec<(FusionTreePairKey, S)>, E>,
{
    let Some(position) = position else {
        return Ok(vec![(tree_pair.clone(), S::one())]);
    };
    let current = repartition(tree_pair, target_codomain_rank)?;
    run_cycle_terms(current, transpose_cycles(position, total_rank), cycle)
}

/// The linearized permutation, identity flag and prepared Artin schedule of a
/// Generic tree-pair braid, the `fsbraid` preamble
/// (`braiding_manipulations.jl:302-306`). The permutation is validated once,
/// by the prepared schedule.
pub(super) struct GenericBraidSchedule {
    pub(super) permutation: Vec<usize>,
    pub(super) identity: bool,
    pub(super) steps: SmallVec<[PreparedArtinStep; 28]>,
}

pub(super) fn generic_braid_schedule(
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<GenericBraidSchedule, CoreError> {
    let (codomain_rank, domain_rank) = (codomain_levels.len(), domain_levels.len());
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    let identity = tree_pair_axis_map_is_identity(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    );
    let mut levels = Vec::with_capacity(codomain_rank + domain_rank);
    levels.extend_from_slice(codomain_levels);
    levels.extend(domain_levels.iter().rev().copied());
    let steps =
        PreparedTreeBraid::new(&permutation, &levels, codomain_rank + domain_rank)?.artin_steps;
    Ok(GenericBraidSchedule {
        permutation,
        identity,
        steps,
    })
}
