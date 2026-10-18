use super::*;

/// Elementary Artin braid of neighbouring uncoupled legs `index` and `index+1`
/// for an outer-multiplicity (`FusionStyleKind::Generic`) rule — the verbatim
/// mirror of TensorKit's `GenericFusion` branches of
/// `artin_braid(src::FusionTreeBlock, i; inv)`
/// (`fusiontrees/braiding_manipulations.jl:81-198`).
///
/// Where the multiplicity-free sibling
/// [`multiplicity_free_artin_braid_at_with_inverse`] returns a scalar per
/// output tree, here every vertex carries a one-based [`MultiplicityIndex`],
/// and one input tree can braid into several
/// output trees that differ *only* in their vertex labels. Each output's scalar
/// coefficient is the `R · F̄ · R̄` inner-index contraction TensorKit writes at
/// `braiding_manipulations.jl:181-182`.
/// The `inverse` flag is handled exactly as TensorKit does — the R-matrices
/// become adjoints (`Rsymbol(...)'`, `braiding_manipulations.jl:139,172-173`),
/// the F-symbol is *not* adjointed, and the contraction formula is otherwise
/// unchanged — rather than being derived here. Applying the `inverse=true`
/// braid to every output of the `inverse=false` braid recovers the original
/// tree with coefficient 1 (unit F/R), which the tests check.
#[cfg(test)]
pub(crate) fn generic_artin_braid_at_with_inverse<R>(
    rule: &R,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericFR(rule);
    generic_artin_braid_at_with_inverse_checked(&checked, tree, index, inverse)
        .map_err(map_infallible_generic_symbol_error)
}

pub(crate) fn generic_artin_braid_at_with_inverse_checked<C>(
    rule: &C,
    tree: &FusionTreeKey,
    index: usize,
    inverse: bool,
) -> Result<GenericTreeTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let kernel = GenericK(rule);
    let site = ArtinSite::new(&kernel, tree.uncoupled(), index, inverse)?;
    let mut out = KeyArtinTerms {
        tree,
        index,
        builder: TreeArenaBuilder::default(),
        terms: Vec::new(),
    };
    artin_surgery(&kernel, &site, tree, &mut out)?;
    Ok(out.into_terms())
}

/// Generic Artin outputs as full keys carrying their vertex labels, pushed
/// into one arena per braid step.
struct KeyArtinTerms<'t, S> {
    tree: &'t FusionTreeKey,
    index: usize,
    builder: TreeArenaBuilder,
    terms: Vec<(usize, S)>,
}

impl<S> KeyArtinTerms<'_, S> {
    fn into_terms(self) -> Vec<(FusionTreeKey, S)> {
        let arena = self.builder.freeze();
        self.terms
            .into_iter()
            .map(|(row, coefficient)| (arena.key(row), coefficient))
            .collect()
    }
}

impl<S, E> ArtinWriter<S, CheckedGenericSymbolError<E>> for KeyArtinTerms<'_, S> {
    type Slot = usize;

    fn reserve(&mut self, additional: usize) {
        self.terms.reserve_exact(additional);
        let tree = self.tree;
        self.builder.reserve_exact(
            additional,
            tree.uncoupled().len() + 1 + tree.innerlines().len(),
            tree.vertices().len(),
            tree.is_dual().len(),
        );
    }

    fn begin(
        &mut self,
        innerline: Option<(usize, SectorId)>,
        vertices: ArtinVertices,
    ) -> Result<Self::Slot, CheckedGenericSymbolError<E>> {
        let tree = self.tree;
        let index = self.index;
        if innerline.is_some_and(|(position, _)| position >= tree.innerlines().len()) {
            return Err(CoreError::MalformedFusionTree {
                message: artin_innerline_message(vertices),
            }
            .into());
        }
        let label = |value: usize| {
            MultiplicityIndex::new(value + 1)
                .expect("enumerated Generic multiplicity labels are one-based")
        };
        let mut labels: MultiplicityVec = tree.vertices().iter().copied().collect();
        match vertices {
            ArtinVertices::Keep => {}
            ArtinVertices::SwapUnit => {
                if labels.len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message:
                            "unit braid past the first adjacent pair requires adjacent vertices",
                    }
                    .into());
                }
                labels.swap(index - 1, index);
            }
            ArtinVertices::First { nu } => {
                *labels.get_mut(0).ok_or(CoreError::MalformedFusionTree {
                    message: "first braid of a Generic tree requires a vertex",
                })? = label(nu);
            }
            ArtinVertices::Pair { sigma, lambda } => {
                if labels.len() <= index {
                    return Err(CoreError::MalformedFusionTree {
                        message: "non-first Generic braid requires adjacent vertices",
                    }
                    .into());
                }
                labels[index - 1] = label(sigma);
                labels[index] = label(lambda);
            }
        }
        let swapped = |position: usize| {
            if position == index {
                index + 1
            } else if position == index + 1 {
                index
            } else {
                position
            }
        };
        let uncoupled = tree.uncoupled();
        let is_dual = tree.is_dual();
        Ok(self.builder.push_parts(
            (0..uncoupled.len()).map(|position| uncoupled[swapped(position)]),
            tree.coupled(),
            (0..is_dual.len()).map(|position| is_dual[swapped(position)]),
            tree.innerlines()
                .iter()
                .enumerate()
                .map(|(position, &sector)| match innerline {
                    Some((edited, replacement)) if edited == position => replacement,
                    _ => sector,
                }),
            labels,
        ))
    }

    fn finish(
        &mut self,
        key: Self::Slot,
        coefficient: S,
    ) -> Result<(), CheckedGenericSymbolError<E>> {
        self.terms.push((key, coefficient));
        Ok(())
    }
}

/// Braid the uncoupled legs of a Generic-fusion tree by `permutation` under the
/// given `levels`, the outer-multiplicity mirror of
/// [`multiplicity_free_braid_tree`] and of TensorKit's `braid(f, p, levels)`
/// swap-decomposition loop (`braiding_manipulations.jl:235-248`,
/// non-`SymmetricBraiding` branch). The permutation is decomposed into
/// neighbouring swaps; each swap is an `generic_artin_braid_at_with_inverse`
/// with `inverse = levels[s] > levels[s+1]` (:239), and the running level
/// tuple is swapped after each step (:243-244).
///
/// Because one input tree can fan out to several vertex-labelled outputs, the
/// coefficients are threaded through a `FusionTermAccumulator` (summing paths
/// that reconverge on the same output tree), exactly as the multiplicity-free
/// braid does.
/// `tree` follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[cfg(test)]
pub(crate) fn generic_braid_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    let rank = tree.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let steps = PreparedTreeBraid::new(permutation, levels, rank)?.artin_steps;
    let validated = validate_fusion_tree_for_rule(rule, tree)?;
    generic_braid_tree_validated(validated, permutation, &steps)
}

#[cfg(test)]
fn generic_braid_tree_validated<R>(
    tree: ValidatedFusionTree<'_, R>,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let rule = tree.rule;
    generic_braid_tree_unchecked(rule, tree.key, permutation, steps)
}

#[cfg(test)]
pub(super) fn generic_braid_tree_unchecked<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    let checked = InfallibleGenericFR(rule);
    generic_braid_tree_result(&checked, tree, permutation, steps)
        .map_err(map_infallible_generic_symbol_error)
}

pub(super) fn generic_braid_tree_result<C>(
    rule: &C,
    tree: &FusionTreeKey,
    permutation: &[usize],
    steps: &[PreparedArtinStep],
) -> Result<GenericTreeTerms<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericFRAccess,
{
    let rank = tree.uncoupled().len();
    if permutation.iter().copied().eq(0..rank) {
        return Ok(vec![(tree.clone(), C::Scalar::one())]);
    }
    braid_tree_steps(tree, steps.iter().copied(), |tree, step| {
        generic_artin_braid_at_with_inverse_checked(rule, tree, step.index, step.inverse)
    })
}
