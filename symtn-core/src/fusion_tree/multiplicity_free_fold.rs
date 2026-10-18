use super::*;

/// The one multiplicity-free `foldright` coefficient authority, shared by the
/// keyed, unique-rigid and compact-block folds.
///
/// TensorKit `foldright((f₁, f₂)::FusionTreePair)` and
/// `foldright(src::FusionTreeBlock)` (`duality_manipulations.jl:220-293`):
/// `coeff = √d_c·(1/√d_b) · conj(coeff₂) · A(a, b, c) · coeff₁`, times `κ_a`
/// when the folded leg is dual. The `(b, c)` factors are separable so a block
/// fold can cache them as TensorKit's `cache₃` does.
///
/// The operand order above is symtn's preserved evaluation order, which keeps
/// multiplicity-free coefficients bit-identical across the keyed,
/// unique-rigid and compact paths. TensorKit writes the same product as
/// `sqrtdim(c)·invsqrtdim(b)·coeff₁·A·conj(coeff₂)` (pair) and
/// `coeff₀·(coeff₂'·(Aᵀ·coeff₁))` (block).
pub(crate) struct MultiplicityFreeFoldCoefficient<S> {
    pub(super) first: SectorId,
    first_is_dual: bool,
    frobenius_schur_phase: S,
}

impl<S: CategoricalScalar> MultiplicityFreeFoldCoefficient<S> {
    pub(super) fn new<R>(rule: &R, first: SectorId, first_is_dual: bool) -> Self
    where
        R: MultiplicityFreeRigidSymbols<Scalar = S>,
    {
        Self {
            first,
            first_is_dual,
            frobenius_schur_phase: rule.frobenius_schur_phase_scalar(first),
        }
    }

    /// `(√d_c/√d_b, A(a, b, c))` for tail coupled `b` and coupled `c`.
    pub(super) fn sector_factors<R>(
        &self,
        rule: &R,
        tail_coupled: SectorId,
        coupled: SectorId,
    ) -> (S, S)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = S>,
    {
        (
            rule.sqrt_dim_scalar(coupled) * rule.inv_sqrt_dim_scalar(tail_coupled),
            rule.a_symbol_scalar(self.first, tail_coupled, coupled),
        )
    }

    pub(super) fn coefficient(
        &self,
        (normalization, a_symbol): &(S, S),
        codomain_coefficient: &S,
        domain_coefficient: &S,
    ) -> S {
        let coefficient = normalization.clone()
            * domain_coefficient.conj()
            * a_symbol.clone()
            * codomain_coefficient.clone();
        if self.first_is_dual {
            coefficient * self.frobenius_schur_phase.clone()
        } else {
            coefficient
        }
    }
}

/// The externals after a `foldright` (`duality_manipulations.jl:220-236`):
/// the first codomain leg moves to the front of the domain as `ā`.
pub(super) fn multiplicity_free_fold_output_frame(
    frame: &MultiplicityFreeTreePairFrame,
    dual_first: SectorId,
    first_is_dual: bool,
) -> MultiplicityFreeTreePairFrame {
    MultiplicityFreeTreePairFrame {
        codomain: MultiplicityFreeTreeFrame::from_externals(
            frame.codomain.uncoupled()[1..].iter().copied(),
            frame.codomain.is_dual()[1..].iter().copied(),
        ),
        domain: MultiplicityFreeTreeFrame::from_externals(
            std::iter::once(dual_first).chain(frame.domain.uncoupled().iter().copied()),
            std::iter::once(!first_is_dual).chain(frame.domain.is_dual().iter().copied()),
        ),
    }
}

pub(crate) fn multiplicity_free_foldright_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let mut terms = FusionTermAccumulator::new();
    fold_surgery(&SimpleK(rule), tree_pair, |key, coefficient| {
        terms.push(key, coefficient)
    })?;
    Ok(terms.into_vec())
}

pub(crate) fn multiplicity_free_cycle_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    direction: PreparedCycleDirection,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    cycle(
        tree_pair,
        direction,
        |key, bend| multiplicity_free_bend_tree_pair(rule, key, bend),
        |key| multiplicity_free_foldright_tree_pair(rule, key),
    )
}

fn collect_multiplicity_free_tree_locals_for_coupled<R>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<MultiplicityFreeTreeLocal>
where
    R: MultiplicityFreeFusionRule,
{
    collect_multiplicity_free_tree_locals_for_coupled_where(rule, effective, coupled, |_, _| true)
}

fn collect_multiplicity_free_tree_locals_for_coupled_where<R, P>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    prefix_allowed: P,
) -> Vec<MultiplicityFreeTreeLocal>
where
    R: MultiplicityFreeFusionRule,
    P: Fn(usize, SectorId) -> bool,
{
    let mut locals = Vec::new();
    let mut inner_rev = Vec::new();
    visit_fusion_trees_where(
        rule,
        effective,
        coupled,
        &mut inner_rev,
        &prefix_allowed,
        &mut |inner_rev| {
            locals.push(MultiplicityFreeTreeLocal {
                coupled,
                innerlines: inner_rev.iter().rev().copied().collect(),
            });
        },
    );
    locals
}

pub(super) fn collect_multiplicity_free_tree_pair_locals_for_frame<R>(
    rule: &R,
    frame: &MultiplicityFreeTreePairFrame,
) -> Vec<MultiplicityFreeTreePairLocal>
where
    R: MultiplicityFreeFusionRule,
{
    let codomain_coupled = reachable_coupled_sectors(rule, frame.codomain.uncoupled());
    let domain_coupled = reachable_coupled_sectors(rule, frame.domain.uncoupled());
    let mut codomain_index = 0usize;
    let mut domain_index = 0usize;
    let mut locals = Vec::new();
    while codomain_index < codomain_coupled.len() && domain_index < domain_coupled.len() {
        let codomain_sector = codomain_coupled[codomain_index];
        let domain_sector = domain_coupled[domain_index];
        match codomain_sector.cmp(&domain_sector) {
            std::cmp::Ordering::Less => codomain_index += 1,
            std::cmp::Ordering::Greater => domain_index += 1,
            std::cmp::Ordering::Equal => {
                let codomain = collect_multiplicity_free_tree_locals_for_coupled(
                    rule,
                    frame.codomain.uncoupled(),
                    codomain_sector,
                );
                let domain = collect_multiplicity_free_tree_locals_for_coupled(
                    rule,
                    frame.domain.uncoupled(),
                    domain_sector,
                );
                for domain_local in &domain {
                    for codomain_local in &codomain {
                        locals.push(MultiplicityFreeTreePairLocal {
                            codomain: codomain_local.clone(),
                            domain: domain_local.clone(),
                        });
                    }
                }
                codomain_index += 1;
                domain_index += 1;
            }
        }
    }
    locals
}

fn multiplicity_free_forward_prefix_targets<T>(
    long_uncoupled: &[SectorId],
    long: &T,
) -> Result<SectorVec, CoreError>
where
    T: TreeView + ?Sized,
{
    let mut targets = SectorVec::with_capacity(long_uncoupled.len().saturating_sub(2));
    for leg_index in 2..long_uncoupled.len() {
        let (_, right) = fusion_tree_vertex_neighbors_from_parts(long_uncoupled, long, leg_index)?;
        targets.push(right);
    }
    Ok(targets)
}

fn multiplicity_free_inverse_prefix_targets<T>(
    short_uncoupled: &[SectorId],
    short: &T,
) -> Result<SectorVec, CoreError>
where
    T: TreeView + ?Sized,
{
    let mut targets = SectorVec::with_capacity(short_uncoupled.len());
    for leg_index in 1..short_uncoupled.len() {
        let (left, right) =
            fusion_tree_vertex_neighbors_from_parts(short_uncoupled, short, leg_index)?;
        if leg_index == 1 {
            targets.push(left);
        }
        targets.push(right);
    }
    Ok(targets)
}

#[inline]
fn multiplicity_free_forward_prefix_allowed<R>(
    rule: &R,
    leading: SectorId,
    targets: &[SectorId],
    tail_prefix_len: usize,
    tail_prefix_coupled: SectorId,
) -> bool
where
    R: MultiplicityFreeFusionRule,
{
    tail_prefix_len < 2
        || rule.nsymbol(leading, tail_prefix_coupled, targets[tail_prefix_len - 2]) != 0
}

#[inline]
fn multiplicity_free_inverse_prefix_allowed<R>(
    rule: &R,
    leading: SectorId,
    targets: &[SectorId],
    candidate_prefix_len: usize,
    candidate_rank: usize,
    candidate_prefix_coupled: SectorId,
) -> bool
where
    R: MultiplicityFreeFusionRule,
{
    candidate_prefix_len <= 2
        || candidate_prefix_len >= candidate_rank
        || rule.nsymbol(
            leading,
            targets[candidate_prefix_len - 2],
            candidate_prefix_coupled,
        ) != 0
}

impl<R> MultiFKernel for SimpleK<'_, R>
where
    R: MultiplicityFreeRigidSymbols,
{
    type Moves = Vec<(MultiplicityFreeTreeLocal, R::Scalar)>;
    type Lift = MultiplicityFreeTreeFrame;

    fn check_externals<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<(), CoreError> {
        if tree.uncoupled().len() != tree.is_dual().len() {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion tree sectors and duality flags must have matching length",
            });
        }
        Ok(())
    }

    fn unit_tail<T: FramedTree + ?Sized>(&self, _: &T) -> Result<Self::Moves, CoreError> {
        Ok(vec![(
            MultiplicityFreeTreeLocal {
                coupled: self.0.vacuum(),
                innerlines: SectorVec::new(),
            },
            R::Scalar::one(),
        )])
    }

    fn single_tail<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, CoreError> {
        Ok(vec![(
            MultiplicityFreeTreeLocal {
                coupled: tree.uncoupled()[1],
                innerlines: SectorVec::new(),
            },
            R::Scalar::one(),
        )])
    }

    // Stage 1 with TensorKit's prefix pruning, then Stage 2 grouped by
    // F-symbol arguments per stage.
    fn tails<T: FramedTree + ?Sized>(&self, tree: &T) -> Result<Self::Moves, CoreError> {
        let rule = self.0;
        let first = tree.uncoupled()[0];
        let tail_uncoupled = &tree.uncoupled()[1..];
        let tail_is_dual = &tree.is_dual()[1..];
        let prefix_targets = multiplicity_free_forward_prefix_targets(tree.uncoupled(), tree)?;
        let mut candidates = Vec::new();
        for tail_coupled in rule.fusion_channels(rule.dual(first), tree.coupled()) {
            candidates.extend(collect_multiplicity_free_tree_locals_for_coupled_where(
                rule,
                tail_uncoupled,
                tail_coupled,
                |prefix_len, prefix_coupled| {
                    multiplicity_free_forward_prefix_allowed(
                        rule,
                        first,
                        &prefix_targets,
                        prefix_len,
                        prefix_coupled,
                    )
                },
            ));
        }
        let coefficients = multiplicity_free_multi_associator_grouped_fixed_long(
            rule,
            tree.uncoupled(),
            tree.is_dual(),
            tree,
            tail_uncoupled,
            tail_is_dual,
            &candidates,
        )?;
        Ok(candidates
            .into_iter()
            .zip(coefficients)
            .filter_map(|(candidate, coefficient)| {
                coefficient.map(|coefficient| (candidate, coefficient))
            })
            .collect())
    }

    fn lift_leading(lift: &MultiplicityFreeTreeFrame) -> SectorId {
        lift.uncoupled()[0]
    }

    fn check_lift<T: FramedTree + ?Sized>(
        &self,
        lift: &MultiplicityFreeTreeFrame,
        tree: &T,
    ) -> Result<(), CoreError> {
        if tree.uncoupled().len() != tree.is_dual().len()
            || lift.uncoupled().len() != lift.is_dual().len()
            || lift.uncoupled().len() != tree.uncoupled().len() + 1
            || lift.uncoupled()[1..] != *tree.uncoupled()
            || lift.is_dual()[1..] != *tree.is_dual()
        {
            return Err(CoreError::MalformedFusionTree {
                message: "multi_Fmove inverse requires one leading external sector",
            });
        }
        Ok(())
    }

    fn admit_lift(
        &self,
        leading: SectorId,
        tree_coupled: SectorId,
        coupled: SectorId,
    ) -> Result<(), CoreError> {
        if self.0.nsymbol(leading, tree_coupled, coupled) == 0 {
            return Err(CoreError::SectorMismatch {
                expected: coupled,
                actual: tree_coupled,
            });
        }
        Ok(())
    }

    fn lifts<T: FramedTree + ?Sized>(
        &self,
        lift: &MultiplicityFreeTreeFrame,
        coupled: SectorId,
        tree: &T,
    ) -> Result<Self::Moves, CoreError> {
        let rule = self.0;
        let leading = lift.uncoupled()[0];
        let prefix_targets = multiplicity_free_inverse_prefix_targets(tree.uncoupled(), tree)?;
        let candidates = collect_multiplicity_free_tree_locals_for_coupled_where(
            rule,
            lift.uncoupled(),
            coupled,
            |prefix_len, prefix_coupled| {
                multiplicity_free_inverse_prefix_allowed(
                    rule,
                    leading,
                    &prefix_targets,
                    prefix_len,
                    lift.uncoupled().len(),
                    prefix_coupled,
                )
            },
        );
        let coefficients = multiplicity_free_multi_associator_grouped_fixed_short(
            rule,
            lift.uncoupled(),
            lift.is_dual(),
            &candidates,
            tree.uncoupled(),
            tree.is_dual(),
            tree,
        )?;
        Ok(candidates
            .into_iter()
            .zip(coefficients)
            .filter_map(|(candidate, coefficient)| {
                coefficient.map(|coefficient| (candidate, coefficient.conj()))
            })
            .collect())
    }
}

/// The keyed multi-F-moves: project, run the surgery on the local, and
/// materialize against the tail (or lifted) externals.
impl<R: MultiplicityFreeRigidSymbols> KeyedMultiFKernel for SimpleK<'_, R> {
    type Moves = Vec<(FusionTreeKey, R::Scalar)>;

    fn multi_fmove(&self, tree: &FusionTreeKey) -> Result<Self::Moves, CoreError> {
        let (frame, local) = project_multiplicity_free_tree(self.0, tree)?;
        let terms = multi_fmove_surgery(
            self,
            &FramedLocal {
                frame: &frame,
                local: &local,
            },
        )?;
        let output_frame = MultiplicityFreeTreeFrame::from_externals(
            frame.uncoupled()[1..].iter().copied(),
            frame.is_dual()[1..].iter().copied(),
        );
        Ok(terms
            .into_iter()
            .map(|(local, coefficient)| (output_frame.materialize(local), coefficient))
            .collect())
    }

    fn multi_fmove_inv(
        &self,
        leading: SectorId,
        coupled: SectorId,
        tree: &FusionTreeKey,
        leading_is_dual: bool,
    ) -> Result<Self::Moves, CoreError> {
        let (source_frame, source) = project_multiplicity_free_tree(self.0, tree)?;
        let output_frame = MultiplicityFreeTreeFrame::from_externals(
            std::iter::once(leading).chain(tree.uncoupled().iter().copied()),
            std::iter::once(leading_is_dual).chain(tree.is_dual().iter().copied()),
        );
        Ok(multi_fmove_inv_surgery(
            self,
            &output_frame,
            coupled,
            &FramedLocal {
                frame: &source_frame,
                local: &source,
            },
        )?
        .into_iter()
        .map(|(local, coefficient)| (output_frame.materialize(local), coefficient))
        .collect())
    }
}

#[cfg(test)]
fn fusion_tree_vertex_neighbors_stepwise_oracle(
    tree: &FusionTreeKey,
    leg_index: usize,
) -> Result<(SectorId, SectorId), CoreError> {
    if leg_index == 0 || leg_index >= tree.uncoupled().len() {
        return Err(CoreError::MalformedFusionTree {
            message: "vertex_info requires a non-first uncoupled leg",
        });
    }
    let left = if leg_index == 1 {
        tree.uncoupled()[0]
    } else {
        tree.innerlines()
            .get(leg_index - 2)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a left innerline",
            })?
    };
    let right = if leg_index + 1 == tree.uncoupled().len() {
        tree.coupled()
    } else {
        tree.innerlines()
            .get(leg_index - 1)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a right innerline",
            })?
    };
    Ok((left, right))
}

#[cfg(test)]
fn multiplicity_free_multi_associator_scalar_stepwise_oracle<R>(
    rule: &R,
    long: &FusionTreeKey,
    short: &FusionTreeKey,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    let rank = long.uncoupled().len();
    if short.uncoupled().len() + 1 != rank
        || long.uncoupled()[1..] != *short.uncoupled()
        || long.is_dual()[1..] != *short.is_dual()
    {
        return Ok(None);
    }

    let mut coefficient = R::Scalar::one();
    let first = long.uncoupled()[0];
    for tensor_kit_k in 2..rank {
        let right = long.uncoupled()[tensor_kit_k];
        let (left_coupled, coupled) =
            fusion_tree_vertex_neighbors_stepwise_oracle(long, tensor_kit_k)?;
        let (middle, right_coupled) =
            fusion_tree_vertex_neighbors_stepwise_oracle(short, tensor_kit_k - 1)?;
        // Why not reuse the production admissibility helper: this oracle must
        // retain the pre-grouping decision path to expose a shared-helper bug.
        if rule.nsymbol(first, right_coupled, coupled) == 0 {
            return Ok(None);
        }
        coefficient = coefficient
            * rule.f_symbol_scalar(first, middle, right, coupled, left_coupled, right_coupled);
    }
    Ok(Some(coefficient))
}

#[cfg(test)]
pub(crate) fn multiplicity_free_multi_fmove_tree_stepwise_oracle<R>(
    rule: &R,
    tree: &FusionTreeKey,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let rank = tree.uncoupled().len();
    if rank == 0 {
        return Err(CoreError::MalformedFusionTree {
            message: "multi_Fmove requires at least one uncoupled sector",
        });
    }
    if rank == 1 {
        return Ok(vec![(
            FusionTreeKey::new(
                Vec::<SectorId>::new(),
                rule.vacuum(),
                Vec::<bool>::new(),
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
            R::Scalar::one(),
        )]);
    }
    if rank == 2 {
        return Ok(vec![(
            FusionTreeKey::new(
                vec![tree.uncoupled()[1]],
                tree.uncoupled()[1],
                vec![tree.is_dual()[1]],
                Vec::<SectorId>::new(),
                Vec::<MultiplicityIndex>::new(),
            ),
            R::Scalar::one(),
        )]);
    }

    let first = tree.uncoupled()[0];
    let coupled = tree.coupled();
    let tail_uncoupled = &tree.uncoupled()[1..];
    let tail_is_dual = &tree.is_dual()[1..];
    let mut terms = Vec::new();
    for tail_coupled in rule.fusion_channels(rule.dual(first), coupled) {
        let tail_effective = effective_sectors_for_uncoupled(rule, tail_uncoupled, tail_is_dual)?;
        for tail_tree in collect_fusion_trees_for_coupled(
            rule,
            tail_uncoupled,
            tail_is_dual,
            &tail_effective,
            tail_coupled,
        ) {
            if let Some(coefficient) =
                multiplicity_free_multi_associator_scalar_stepwise_oracle(rule, tree, &tail_tree)?
            {
                terms.push((tail_tree, coefficient));
            }
        }
    }
    Ok(terms)
}

#[cfg(test)]
pub(crate) fn multiplicity_free_multi_fmove_inv_tree_stepwise_oracle<R>(
    rule: &R,
    leading_sector: SectorId,
    coupled: SectorId,
    tree: &FusionTreeKey,
    leading_is_dual: bool,
) -> Result<Vec<(FusionTreeKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
{
    let tree_coupled = tree.coupled();
    if rule.nsymbol(leading_sector, tree_coupled, coupled) == 0 {
        return Err(CoreError::SectorMismatch {
            expected: coupled,
            actual: tree_coupled,
        });
    }

    let mut uncoupled = Vec::with_capacity(tree.uncoupled().len() + 1);
    uncoupled.push(leading_sector);
    uncoupled.extend_from_slice(tree.uncoupled());
    let mut is_dual = Vec::with_capacity(tree.is_dual().len() + 1);
    is_dual.push(leading_is_dual);
    is_dual.extend_from_slice(tree.is_dual());
    let effective = effective_sectors_for_uncoupled(rule, &uncoupled, &is_dual)?;
    let candidates =
        collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);

    let mut terms = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if let Some(coefficient) =
            multiplicity_free_multi_associator_scalar_stepwise_oracle(rule, &candidate, tree)?
        {
            terms.push((candidate, (coefficient).conj()));
        }
    }
    Ok(terms)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultiplicityFreeFsymbolArguments {
    left: SectorId,
    middle: SectorId,
    right: SectorId,
    coupled: SectorId,
    left_coupled: SectorId,
    right_coupled: SectorId,
}

impl MultiplicityFreeFsymbolArguments {
    #[inline]
    fn is_admissible<R>(self, rule: &R) -> bool
    where
        R: MultiplicityFreeFusionSymbols,
    {
        multi_associator_new_cross_channel_is_admissible(
            rule,
            self.left,
            self.right_coupled,
            self.coupled,
        )
    }

    #[inline]
    fn evaluate<R>(self, rule: &R) -> R::Scalar
    where
        R: MultiplicityFreeFusionSymbols,
    {
        rule.f_symbol_scalar(
            self.left,
            self.middle,
            self.right,
            self.coupled,
            self.left_coupled,
            self.right_coupled,
        )
    }
}

fn multiplicity_free_multi_associator_arguments_from_parts<L, S>(
    long_uncoupled: &[SectorId],
    long: &L,
    short_uncoupled: &[SectorId],
    short: &S,
    tensor_kit_k: usize,
) -> Result<MultiplicityFreeFsymbolArguments, CoreError>
where
    L: TreeView + ?Sized,
    S: TreeView + ?Sized,
{
    let right = long_uncoupled[tensor_kit_k];
    let (left_coupled, coupled) =
        fusion_tree_vertex_neighbors_from_parts(long_uncoupled, long, tensor_kit_k)?;
    let (middle, right_coupled) =
        fusion_tree_vertex_neighbors_from_parts(short_uncoupled, short, tensor_kit_k - 1)?;
    Ok(MultiplicityFreeFsymbolArguments {
        left: long_uncoupled[0],
        middle,
        right,
        coupled,
        left_coupled,
        right_coupled,
    })
}

fn multiplicity_free_multi_associator_grouped<R, F>(
    rule: &R,
    rank: usize,
    candidate_count: usize,
    arguments_for: F,
) -> Result<Vec<Option<R::Scalar>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
    F: Fn(usize, usize) -> Result<MultiplicityFreeFsymbolArguments, CoreError>,
{
    enum Coefficient<S> {
        Pending,
        Active(S),
        Rejected,
    }

    let mut coefficients = (0..candidate_count)
        .map(|_| Coefficient::Pending)
        .collect::<Vec<_>>();
    // Why not retain this table between stages or invocations: TensorKit's
    // grouping is stage-local, and a longer-lived table would create an
    // implicit provider cache for a pure eager reduction.
    let mut stage_symbols: FxHashMap<MultiplicityFreeFsymbolArguments, R::Scalar> =
        FxHashMap::default();
    stage_symbols.reserve(candidate_count);
    for tensor_kit_k in 2..rank {
        stage_symbols.clear();
        for (candidate, coefficient) in coefficients.iter_mut().enumerate() {
            if matches!(coefficient, Coefficient::Rejected) {
                continue;
            }
            let arguments = arguments_for(candidate, tensor_kit_k)?;
            if !arguments.is_admissible(rule) {
                *coefficient = Coefficient::Rejected;
                continue;
            }
            let symbol = match stage_symbols.entry(arguments) {
                Entry::Occupied(entry) => entry.get().clone(),
                Entry::Vacant(entry) => {
                    let symbol = arguments.evaluate(rule);
                    entry.insert(symbol.clone());
                    symbol
                }
            };
            *coefficient = match std::mem::replace(coefficient, Coefficient::Rejected) {
                Coefficient::Pending => Coefficient::Active(symbol),
                Coefficient::Active(prefix) => Coefficient::Active(prefix * symbol),
                Coefficient::Rejected => unreachable!("rejected coefficients are skipped"),
            };
        }
    }
    Ok(coefficients
        .into_iter()
        .map(|coefficient| match coefficient {
            Coefficient::Pending => Some(R::Scalar::one()),
            Coefficient::Active(value) => Some(value),
            Coefficient::Rejected => None,
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn multiplicity_free_multi_associator_grouped_fixed_long<R, L, S>(
    rule: &R,
    long_uncoupled: &[SectorId],
    long_is_dual: &[bool],
    long: &L,
    short_uncoupled: &[SectorId],
    short_is_dual: &[bool],
    shorts: &[S],
) -> Result<Vec<Option<R::Scalar>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
    L: TreeView + ?Sized,
    S: TreeView,
{
    let rank = long_uncoupled.len();
    if short_uncoupled.len() + 1 != rank
        || long_uncoupled[1..] != *short_uncoupled
        || long_is_dual[1..] != *short_is_dual
    {
        return Ok((0..shorts.len()).map(|_| None).collect());
    }
    multiplicity_free_multi_associator_grouped(
        rule,
        rank,
        shorts.len(),
        |candidate, tensor_kit_k| {
            multiplicity_free_multi_associator_arguments_from_parts(
                long_uncoupled,
                long,
                short_uncoupled,
                &shorts[candidate],
                tensor_kit_k,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn multiplicity_free_multi_associator_grouped_fixed_short<R, L, S>(
    rule: &R,
    long_uncoupled: &[SectorId],
    long_is_dual: &[bool],
    longs: &[L],
    short_uncoupled: &[SectorId],
    short_is_dual: &[bool],
    short: &S,
) -> Result<Vec<Option<R::Scalar>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Mul<Output = R::Scalar>,
    L: TreeView,
    S: TreeView + ?Sized,
{
    let rank = long_uncoupled.len();
    if short_uncoupled.len() + 1 != rank
        || long_uncoupled[1..] != *short_uncoupled
        || long_is_dual[1..] != *short_is_dual
    {
        return Ok((0..longs.len()).map(|_| None).collect());
    }
    multiplicity_free_multi_associator_grouped(
        rule,
        rank,
        longs.len(),
        |candidate, tensor_kit_k| {
            multiplicity_free_multi_associator_arguments_from_parts(
                long_uncoupled,
                &longs[candidate],
                short_uncoupled,
                short,
                tensor_kit_k,
            )
        },
    )
}

#[cfg(test)]
pub(crate) fn multiplicity_free_multi_associator_scalar<R>(
    rule: &R,
    long: &FusionTreeKey,
    short: &FusionTreeKey,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    multiplicity_free_multi_associator_from_parts(
        rule,
        long.uncoupled(),
        long.is_dual(),
        long,
        short.uncoupled(),
        short.is_dual(),
        short,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn multiplicity_free_multi_associator_from_parts<R, L, S>(
    rule: &R,
    long_uncoupled: &[SectorId],
    long_is_dual: &[bool],
    long: &L,
    short_uncoupled: &[SectorId],
    short_is_dual: &[bool],
    short: &S,
) -> Result<Option<R::Scalar>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
    L: TreeView + ?Sized,
    S: TreeView + ?Sized,
{
    let rank = long_uncoupled.len();
    if short_uncoupled.len() + 1 != rank {
        return Ok(None);
    }
    if long_uncoupled[1..] != *short_uncoupled || long_is_dual[1..] != *short_is_dual {
        return Ok(None);
    }

    let mut coefficient = R::Scalar::one();
    let first = long_uncoupled[0];
    for tensor_kit_k in 2..rank {
        let right_sector = long_uncoupled[tensor_kit_k];
        let (middle_left, middle_right) =
            fusion_tree_vertex_neighbors_from_parts(long_uncoupled, long, tensor_kit_k)?;
        let (short_left, short_right) =
            fusion_tree_vertex_neighbors_from_parts(short_uncoupled, short, tensor_kit_k - 1)?;
        if !multi_associator_new_cross_channel_is_admissible(rule, first, short_right, middle_right)
        {
            return Ok(None);
        }
        coefficient = coefficient
            * rule.f_symbol_scalar(
                first,
                short_left,
                right_sector,
                middle_right,
                middle_left,
                short_right,
            );
    }
    Ok(Some(coefficient))
}

#[cfg(test)]
pub(crate) fn fusion_tree_vertex_neighbors(
    tree: &FusionTreeKey,
    leg_index: usize,
) -> Result<(SectorId, SectorId), CoreError> {
    fusion_tree_vertex_neighbors_from_parts(tree.uncoupled(), tree, leg_index)
}

pub(super) fn fusion_tree_vertex_neighbors_from_parts<T>(
    uncoupled: &[SectorId],
    tree: &T,
    leg_index: usize,
) -> Result<(SectorId, SectorId), CoreError>
where
    T: TreeView + ?Sized,
{
    if leg_index == 0 || leg_index >= uncoupled.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "vertex_info requires a non-first uncoupled leg",
        });
    }
    let left = if leg_index == 1 {
        uncoupled[0]
    } else {
        tree.innerlines()
            .get(leg_index - 2)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a left innerline",
            })?
    };
    let right = if leg_index + 1 == uncoupled.len() {
        tree.coupled()
    } else {
        tree.innerlines()
            .get(leg_index - 1)
            .copied()
            .ok_or(CoreError::MalformedFusionTree {
                message: "fusion tree is missing a right innerline",
            })?
    };
    Ok((left, right))
}

#[cfg(test)]
fn effective_sectors_for_uncoupled<R>(
    _rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
) -> Result<Vec<SectorId>, CoreError>
where
    R: FusionRule,
{
    if uncoupled.len() != is_dual.len() {
        return Err(CoreError::MalformedFusionTree {
            message: "fusion tree sectors and duality flags must have matching length",
        });
    }
    Ok(uncoupled.to_vec())
}
