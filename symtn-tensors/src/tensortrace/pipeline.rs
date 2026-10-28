//! The one trace compile pipeline (#1856): a shared preflight (pair
//! duality, selected HomSpace) and a shared term builder (split, trace
//! channel match, channel factor, destination lookup), with the admission
//! mode supplying only its symbol access and errors.
//!
//! TensorKit `trace_permute!` / `_trace_permute!`
//! (`src/tensors/tensoroperations.jl`): check the pair duality and the
//! selected space, then per source tree pair permute to `(p…, q…)`, split at
//! the open ranks, keep `g₁ == g₂`, and scale by
//! `dim(coupled)/dim(first) · Π twist(non-dual)`.
//!
//! Why hook traits rather than one generic body: the multiplicity-free and
//! checked Generic modes keep different symbol access and error types; the
//! hooks are the narrowest layer that owns those differences. Both modes
//! detect the same misuses in TensorKit's order (#1872): braiding, axes,
//! selection, destination space (when there is one), pair duality.

use super::*;
use crate::tree_transform::{PendingCoefficientGroups, TraceTermReuse, TransformerMode};
use crate::TreeTransformOperation;
use symtn_core::{
    generic_permute_tree_pair_block_indexed_checked, BlockSourceColumns, BraidingStyleKind,
    CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode, RuleIdentity,
};

/// What the trace preflight asks of an admission mode.
pub(super) trait TracePreflightMode<R> {
    type Error;

    fn operation(error: OperationError) -> Self::Error;

    fn validate_source(src: &BoundDynamicFusionMapSpace<R>) -> Result<(), Self::Error>;

    fn select(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, Self::Error>;

    fn pair_is_dual(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        lhs_axis: usize,
        rhs_axis: usize,
    ) -> Result<bool, Self::Error>;

    fn braiding_style(rule: &R) -> BraidingStyleKind;
}

/// The one trace preflight, in TensorKit `trace_permute!`'s order: the
/// braiding gate, the axes, the selected output HomSpace, then the pair
/// duality, which the caller raises through
/// [`TracePreflight::require_dual_pairs`] once any destination space has been
/// compared.
pub(super) fn trace_preflight<M, R>(
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<TracePreflight, M::Error>
where
    M: TracePreflightMode<R>,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::TracePreflight);
    M::validate_source(src)?;
    // TensorKit `trace_permute!` gates the braiding first: a trace is defined
    // only for symmetric braiding.
    crate::admission::require_symmetric_braiding(
        M::braiding_style(src.provider()),
        crate::admission::SymmetricBraidingOp::Trace,
    )
    .map_err(M::operation)?;
    let orientation = if axes.source_conjugate() {
        FusionTreePairOrientation::Adjoint
    } else {
        FusionTreePairOrientation::Direct
    };
    let lowered_axes =
        lower_tensortrace_source_adjoint_axes_dyn(src.space().nout(), src.space().nin(), axes)
            .map_err(M::operation)?;
    let lowered_spec = lowered_axes.as_spec();
    let axis_plan = TensorTraceAxisPlan::compile(
        src.space().rank(),
        lowered_spec.output_axes().len(),
        lowered_spec,
    )
    .map_err(M::operation)?;
    trace_geometry::<M, R>(
        src.provider(),
        OrientedFusionTreeHomSpace::new(src.space().homspace(), orientation),
        &axis_plan,
        dst_nout,
    )
}

/// The preflight's geometry half, for a compile that already holds its axis
/// plan.
pub(super) fn trace_geometry<M, R>(
    rule: &R,
    src_homspace: OrientedFusionTreeHomSpace<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_nout: usize,
) -> Result<TracePreflight, M::Error>
where
    M: TracePreflightMode<R>,
{
    if dst_nout > axis_plan.output_axes.len() {
        return Err(M::operation(OperationError::RankMismatch {
            expected: axis_plan.output_axes.len(),
            actual: dst_nout,
        }));
    }
    let selected_homspace = M::select(
        rule,
        src_homspace,
        &axis_plan.output_axes[..dst_nout],
        &axis_plan.output_axes[dst_nout..],
    )?;
    let mut trace_pairs_match = true;
    for (&lhs_axis, &rhs_axis) in axis_plan
        .trace_lhs_axes
        .iter()
        .zip(axis_plan.trace_rhs_axes.iter())
    {
        trace_pairs_match &= M::pair_is_dual(rule, src_homspace, lhs_axis, rhs_axis)?;
    }
    Ok(TracePreflight {
        selected_homspace,
        trace_pairs_match,
    })
}

impl<R> TracePreflightMode<R> for MultiplicityFreeAdmissionMode
where
    R: FusionRule + CheckedFusionAlgebra,
{
    type Error = OperationError;

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        error
    }

    fn validate_source(src: &BoundDynamicFusionMapSpace<R>) -> Result<(), Self::Error> {
        src.space().validate_rule(src.provider())?;
        Ok(())
    }

    fn select(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, Self::Error> {
        // Why not widen the public rule bound: custom encoded rules retain their
        // established infallible contract while lowered built-ins close overflow.
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::TraceSelectedHomspaceDerivation);
        src.try_select_checked(rule, codomain_axes, domain_axes)
            .map_err(|error| match error {
                CheckedFusionSpaceError::FusionAlgebra(error) => {
                    OperationError::FusionAlgebra(error)
                }
                CheckedFusionSpaceError::Core(error) => OperationError::Core(*error),
                _ => OperationError::InvalidArgument {
                    message: "checked trace metadata error",
                },
            })
    }

    fn pair_is_dual(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        lhs_axis: usize,
        rhs_axis: usize,
    ) -> Result<bool, Self::Error> {
        let lhs = outward_axis_leg_checked(rule, src, lhs_axis)?;
        let rhs = outward_axis_leg_checked(rule, src, rhs_axis)?;
        let rhs_dual = rhs
            .try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        lhs.try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        Ok(lhs == rhs_dual)
    }

    fn braiding_style(rule: &R) -> BraidingStyleKind {
        FusionRule::braiding_style(rule)
    }
}

impl<R> TracePreflightMode<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericFusion,
{
    type Error = CheckedGenericPlanError<R::Error>;

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        CheckedGenericPlanError::Operation(error)
    }

    fn validate_source(_src: &BoundDynamicFusionMapSpace<R>) -> Result<(), Self::Error> {
        Ok(())
    }

    fn select(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, Self::Error> {
        src.try_select_generic_checked(rule, codomain_axes, domain_axes)
            .map_err(CheckedGenericPlanError::from)
    }

    fn pair_is_dual(
        rule: &R,
        src: OrientedFusionTreeHomSpace<'_>,
        lhs_axis: usize,
        rhs_axis: usize,
    ) -> Result<bool, Self::Error> {
        let lhs = src
            .try_external_axis_leg_generic(rule, lhs_axis)
            .map_err(CheckedGenericPlanError::from)?
            .ok_or_else(|| {
                CheckedGenericPlanError::Operation(OperationError::InvalidArgument {
                    message: "trace lhs axis has no external leg",
                })
            })?;
        let rhs = src
            .try_external_axis_leg_generic(rule, rhs_axis)
            .map_err(CheckedGenericPlanError::from)?
            .ok_or_else(|| {
                CheckedGenericPlanError::Operation(OperationError::InvalidArgument {
                    message: "trace rhs axis has no external leg",
                })
            })?;
        // Why the whole leg: TensorKit `trace_permute!` requires
        // `space(tsrc, q₁) == dual(space(tsrc, q₂))`; comparing only the first
        // sector admitted non-dual pairs and summed only the sectors they share.
        let rhs_dual = rhs
            .try_dual_generic(rule)
            .map_err(CheckedGenericPlanError::from)?;
        Ok(lhs == rhs_dual)
    }

    fn braiding_style(rule: &R) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(rule)
    }
}

/// What the trace term builder asks of an admission mode.
pub(super) trait TraceTermMode<R> {
    type Scalar: Clone + Mul<Output = Self::Scalar> + Send + Sync + 'static;
    type Error;

    /// The composer and admission family a cache-4 trace entry is keyed by.
    const MODE: TransformerMode;

    fn rule_identity(rule: &R) -> RuleIdentity;

    fn operation(error: OperationError) -> Self::Error;

    /// Hands every source block's permuted tree-pair rows, in source block
    /// order, to [`TraceLowering::lower`]; returns the permutation groups it
    /// built, for the caller to publish once its whole call succeeded.
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<PendingCoefficientGroups, Self::Error>;

    /// Split `tree` at `rank` into `rows`; returns the rows of both halves.
    fn split(
        rule: &R,
        tree: FusionTreeRef<'_>,
        rank: usize,
        rows: &mut TreeScratch,
    ) -> Result<(usize, usize), Self::Error>;

    /// `dim(coupled)/dim(first) · Π twist(non-dual)`: one division of the
    /// provider's dimensions, as TensorKit's `_trace_permute!` forms it.
    fn channel_factor(rule: &R, trace_tree: FusionTreeRef<'_>)
        -> Result<Self::Scalar, Self::Error>;
}

/// The trace geometry every source row is lowered against.
pub(super) struct TraceLowering<'a> {
    pub(super) dst_structure: &'a BlockStructure,
    pub(super) src: OrientedTraceSource<'a>,
    pub(super) axis_plan: &'a TensorTraceAxisPlan,
    pub(super) dst_codomain_rank: usize,
}

impl TraceLowering<'_> {
    /// Lowers permuted rows: split at the open ranks, keep the rows whose
    /// traced trees match, scale by the channel factor, and hand each
    /// `(open tree pair, coefficient)` to `keep`, in row order. The split
    /// halves live in `split` until the next row.
    #[inline]
    fn lower_rows<'r, M, R, I>(
        &self,
        rule: &R,
        rows: I,
        split: &mut TreeScratch,
        mut keep: impl FnMut(FusionTreePairRef<'_>, M::Scalar) -> Result<(), M::Error>,
    ) -> Result<(), M::Error>
    where
        M: TraceTermMode<R>,
        M::Scalar: 'r,
        I: IntoIterator<Item = (FusionTreePairRef<'r>, &'r M::Scalar)>,
    {
        for (permuted, permutation_coefficient) in rows {
            split.clear();
            let (dst_codomain_tree, trace_codomain_tree) = M::split(
                rule,
                permuted.codomain_tree(),
                self.dst_codomain_rank,
                split,
            )?;
            let (dst_domain_tree, trace_domain_tree) = M::split(
                rule,
                permuted.domain_tree(),
                self.axis_plan.output_axes.len() - self.dst_codomain_rank,
                split,
            )?;
            let trace_tree = split.view(trace_codomain_tree);
            if trace_tree != split.view(trace_domain_tree) {
                continue;
            }
            let trace_factor = M::channel_factor(rule, trace_tree)?;
            keep(
                FusionTreePairRef::new(split.view(dst_codomain_tree), split.view(dst_domain_tree)),
                permutation_coefficient.clone() * trace_factor,
            )?;
        }
        Ok(())
    }

    /// Lowers a whole group's permutation columns, column by column, into
    /// the value cache 4 keeps for it: destinations are the distinct open
    /// tree pairs, in first-appearance order.
    fn lower_group<M, R>(
        &self,
        rule: &R,
        permuted: &BlockSourceColumns<FusionTreePairKey, M::Scalar>,
        split: &mut TreeScratch,
    ) -> Result<BlockSourceColumns<FusionTreePairKey, M::Scalar>, M::Error>
    where
        M: TraceTermMode<R>,
    {
        let rows = permuted.destinations();
        let mut row_of = rustc_hash::FxHashMap::<FusionTreePairKey, usize>::default();
        let mut destinations = Vec::new();
        let mut column_start = Vec::with_capacity(permuted.source_count() + 1);
        let mut entries = Vec::with_capacity(permuted.entry_count());
        column_start.push(0);
        for source in 0..permuted.source_count() {
            self.lower_rows::<M, R, _>(
                rule,
                permuted
                    .column(source)
                    .iter()
                    .map(|(row, coefficient)| (rows[*row].as_view(), coefficient)),
                split,
                |dst, coefficient| {
                    let next = destinations.len();
                    let row = *row_of.entry(dst.to_key()).or_insert_with_key(|key| {
                        destinations.push(key.clone());
                        next
                    });
                    entries.push((row, coefficient));
                    Ok(())
                },
            )?;
            column_start.push(entries.len());
        }
        Ok(BlockSourceColumns::new(
            destinations.into(),
            column_start.into(),
            entries.into(),
        ))
    }

    /// Addresses one lowered row's destination block and records its term,
    /// holding the destination structure's own key.
    #[inline]
    fn push_term<M, R>(
        &self,
        src_block_index: usize,
        src_key: &impl Fn() -> Result<FusionTreePairKey, M::Error>,
        dst: FusionTreePairRef<'_>,
        coefficient: M::Scalar,
        terms: &mut Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
    ) -> Result<(), M::Error>
    where
        M: TraceTermMode<R>,
    {
        let missing = || {
            M::operation(OperationError::MissingBlockKey {
                key: Box::new(BlockKey::from(dst.to_key())),
            })
        };
        let dst_block = self
            .dst_structure
            .find_block_index_by_fusion_tree_pair_view(dst)
            .ok_or_else(missing)?;
        let dst_key = self
            .dst_structure
            .block(dst_block)
            .ok()
            .and_then(|block| block.key().as_fusion_tree_pair().cloned())
            .ok_or_else(missing)?;
        terms.push(TensorTraceFusionStructureTerm {
            dst_key,
            src_key: src_key()?,
            dst_block,
            src_block: src_block_index,
            coefficient,
        });
        Ok(())
    }
}

/// The one trace term builder: TensorKit's `(p…, q…)` permutation, then each
/// source block's rows lowered by [`TraceLowering::lower`]. Returns the
/// terms and the permutation groups built on a cache-4 miss; the caller
/// publishes them only after its whole call succeeded, or drops them.
#[allow(clippy::type_complexity)]
pub(super) fn build_trace_terms<M, R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src: OrientedTraceSource<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
) -> Result<
    (
        Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
        PendingCoefficientGroups,
    ),
    M::Error,
>
where
    M: TraceTermMode<R>,
{
    let mut codomain_permutation =
        Vec::with_capacity(dst_codomain_rank + axis_plan.trace_lhs_axes.len());
    codomain_permutation.extend_from_slice(&axis_plan.output_axes[..dst_codomain_rank]);
    codomain_permutation.extend_from_slice(&axis_plan.trace_lhs_axes);
    let mut domain_permutation = Vec::with_capacity(
        axis_plan.output_axes.len() - dst_codomain_rank + axis_plan.trace_rhs_axes.len(),
    );
    domain_permutation.extend_from_slice(&axis_plan.output_axes[dst_codomain_rank..]);
    domain_permutation.extend_from_slice(&axis_plan.trace_rhs_axes);

    let lowering = TraceLowering {
        dst_structure,
        src,
        axis_plan,
        dst_codomain_rank,
    };
    let mut terms = Vec::new();
    let pending = M::lower_source_rows(
        rule,
        &lowering,
        &codomain_permutation,
        &domain_permutation,
        &mut terms,
    )?;
    Ok((terms, pending))
}

/// Lowers every source block in block order, resolving a fusion group's
/// lowered terms once, when its first member is reached: from cache 4 when
/// resident (no admission, composition, split or channel-factor query, see
/// [`TraceTermReuse`]), otherwise by `compose_group` and
/// [`TraceLowering::lower_group`], staged for the caller to publish.
///
/// Why not compose every group up front: a later group's symbol or
/// admission error must not overtake an earlier source's lowering error.
/// The current group stays atomic, composed and lowered whole, as TensorKit
/// `_trace_permute!` lowers each fusion block's `U` right after permuting
/// it; per-source replay would repeat its F/R traversal. So provider
/// errors can change order only between members of one group (#2054).
fn lower_by_fusion_group<M, R>(
    rule: &R,
    lowering: &TraceLowering<'_>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    terms: &mut Vec<TensorTraceFusionStructureTerm<M::Scalar>>,
    mut compose_group: impl FnMut(
        &[usize],
    )
        -> Result<BlockSourceColumns<FusionTreePairKey, M::Scalar>, M::Error>,
) -> Result<PendingCoefficientGroups, M::Error>
where
    M: TraceTermMode<R>,
{
    let src = lowering.src;
    // Keyed by the storage keys and orientation (not the oriented keys): a
    // warm lookup then allocates no swapped key.
    let mut reuse = TraceTermReuse::<M::Scalar>::new(
        M::rule_identity(rule),
        M::MODE,
        TreeTransformOperation::permute(
            codomain_permutation.iter().copied(),
            domain_permutation.iter().copied(),
        ),
        src.orientation,
        lowering.dst_codomain_rank,
    );
    let mut storage_keys = Vec::new();
    let mut split = TreeScratch::default();
    let block_count = src.structure.block_count();
    let groups = src.structure.fusion_tree_group_slice();
    // Each source's group and its position (column) in that group.
    let mut member_of_source = vec![None; block_count];
    for (group_index, group) in groups.iter().enumerate() {
        for (position, &src_block_index) in group.block_indices().iter().enumerate() {
            member_of_source[src_block_index] = Some((group_index, position));
        }
    }
    let mut lowered_by_group = (0..groups.len()).map(|_| None).collect::<Vec<_>>();
    for (src_block_index, member) in member_of_source.into_iter().enumerate() {
        let (group_index, position) = member.ok_or_else(|| {
            M::operation(OperationError::InvalidArgument {
                message: "trace source block was not assigned to a fusion group",
            })
        })?;
        let lowered = match &mut lowered_by_group[group_index] {
            Some(lowered) => lowered,
            slot @ None => {
                let group = &groups[group_index];
                #[cfg(test)]
                crate::probe::push_trace_source(src_block_index);
                storage_keys.clear();
                for &index in group.block_indices() {
                    storage_keys.push(src.storage_key(index).map_err(M::operation)?);
                }
                let lowered = match reuse.lookup((group.group_key(), &storage_keys)) {
                    Ok(lowered) => lowered,
                    Err(hash) => {
                        let permuted = compose_group(group.block_indices())?;
                        if permuted.source_count() != group.block_indices().len() {
                            return Err(M::operation(OperationError::InvalidArgument {
                                message:
                                    "trace block transform returned the wrong source column count",
                            }));
                        }
                        let lowered = lowering.lower_group::<M, R>(rule, &permuted, &mut split)?;
                        reuse.stage(hash, (group.group_key(), &storage_keys), lowered)
                    }
                };
                slot.insert(lowered)
            }
        };
        let src_key = || src.source_key(src_block_index).map_err(M::operation);
        let destinations = lowered.destinations();
        for (row, coefficient) in lowered.column(position) {
            lowering.push_term::<M, R>(
                src_block_index,
                &src_key,
                destinations[*row].as_view(),
                coefficient.clone(),
                terms,
            )?;
        }
    }
    Ok(reuse.into_pending())
}

impl<R> TraceTermMode<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    type Scalar = R::Scalar;
    type Error = OperationError;

    const MODE: TransformerMode = TransformerMode::MultiplicityFree;

    fn rule_identity(rule: &R) -> RuleIdentity {
        FusionRule::rule_identity(rule)
    }

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        error
    }

    /// TensorKit's dispatch: one permuted row per tree pair for unique
    /// fusion (`_trace_permute!(::UniqueFusion)`), otherwise one block-indexed
    /// permute per fusion group (`fusionblocks`).
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<PendingCoefficientGroups, Self::Error> {
        let src = lowering.src;
        src.validate_source_keys()?;
        if rule.fusion_style() == FusionStyleKind::Unique {
            let (source_codomain_rank, source_domain_rank) = match src.orientation {
                FusionTreePairOrientation::Direct => (src.storage_nout, src.storage_nin),
                FusionTreePairOrientation::Adjoint => (src.storage_nin, src.storage_nout),
            };
            let prepared = PreparedTreePairOperation::prepare_permute(
                rule,
                source_codomain_rank,
                source_domain_rank,
                codomain_permutation,
                domain_permutation,
            )
            .map_err(OperationError::from_core_preserving_context)?;
            // Every intermediate and split tree lives in these two scratches,
            // as TensorKit's isbits trees do on the stack: no allocation per
            // tree (#2192).
            let rank = source_codomain_rank + source_domain_rank;
            let mut permuted = UniqueTreePairScratch::with_rank(rank);
            let mut split = TreeScratch::with_rank(rank);
            // At most one term per source tree.
            terms.reserve_exact(src.structure.block_count());
            for src_block_index in 0..src.structure.block_count() {
                #[cfg(test)]
                crate::probe::push_trace_source(src_block_index);
                let coefficient = prepared
                    .execute_unique_rigid_into(
                        rule,
                        &src.source_key(src_block_index)?,
                        &mut permuted,
                    )
                    .map_err(OperationError::from_core_preserving_context)?;
                lowering.lower_rows::<Self, R, _>(
                    rule,
                    core::iter::once((permuted.pair_view(), &coefficient)),
                    &mut split,
                    |dst, coefficient| {
                        lowering.push_term::<Self, R>(
                            src_block_index,
                            &|| src.source_key(src_block_index),
                            dst,
                            coefficient,
                            terms,
                        )
                    },
                )?;
            }
            // TensorKit `NoCache` for Unique: one uncached phase per tree on
            // a compiled-trace miss; a warm compile never reaches here.
            return Ok(PendingCoefficientGroups::default());
        }

        lower_by_fusion_group::<Self, R>(
            rule,
            lowering,
            codomain_permutation,
            domain_permutation,
            terms,
            |indices| {
                multiplicity_free_permute_tree_pair_block_indexed(
                    rule,
                    src.structure,
                    indices,
                    src.orientation,
                    codomain_permutation,
                    domain_permutation,
                )
                .map_err(OperationError::from_core_preserving_context)
            },
        )
    }

    #[inline]
    fn split(
        rule: &R,
        tree: FusionTreeRef<'_>,
        rank: usize,
        rows: &mut TreeScratch,
    ) -> Result<(usize, usize), Self::Error> {
        split_fusion_tree_into(rule, tree, rank, rows)
            .map_err(OperationError::from_core_preserving_context)
    }

    #[inline]
    fn channel_factor(
        rule: &R,
        trace_tree: FusionTreeRef<'_>,
    ) -> Result<Self::Scalar, Self::Error> {
        let coupled = trace_tree.coupled();
        let first = trace_tree.uncoupled().first().copied().ok_or(
            symtn_core::CoreError::MalformedFusionTree {
                message: "trace channel requires at least one uncoupled sector",
            },
        );
        let first = first.map_err(OperationError::from_core_preserving_context)?;
        // Why not `dim(c) · inv_dim(first)`: the reciprocal rounds twice
        // (SU(2) `5 · fl(1/3)` is an ulp below `5/3`, Fibonacci
        // `φ · fl(1/φ) = 1 - 2⁻⁵³`), unlike TensorKit and the checked mode.
        let mut factor = rule.dim_scalar(coupled) / rule.dim_scalar(first);
        for (&sector, &is_dual) in trace_tree
            .uncoupled()
            .iter()
            .zip(trace_tree.is_dual())
            .skip(1)
        {
            if !is_dual {
                factor = factor * rule.twist_scalar(sector);
            }
        }
        Ok(factor)
    }
}

impl<R> TraceTermMode<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericPivotal,
    R::Scalar: Clone + Mul<Output = R::Scalar> + Zero + 'static,
{
    type Scalar = R::Scalar;
    type Error = CheckedGenericPlanError<R::Error>;

    const MODE: TransformerMode = TransformerMode::CheckedGeneric;

    fn rule_identity(rule: &R) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(rule)
    }

    #[inline]
    fn operation(error: OperationError) -> Self::Error {
        CheckedGenericPlanError::Operation(error)
    }

    /// TensorKit's Generic dispatch: one whole-basis checked permute per
    /// fusion group, admitting each member once (#2054).
    fn lower_source_rows(
        rule: &R,
        lowering: &TraceLowering<'_>,
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        terms: &mut Vec<TensorTraceFusionStructureTerm<Self::Scalar>>,
    ) -> Result<PendingCoefficientGroups, Self::Error> {
        let src = lowering.src;
        src.validate_source_keys()
            .map_err(CheckedGenericPlanError::Operation)?;
        lower_by_fusion_group::<Self, R>(
            rule,
            lowering,
            codomain_permutation,
            domain_permutation,
            terms,
            |indices| {
                generic_permute_tree_pair_block_indexed_checked(
                    rule,
                    src.structure,
                    indices,
                    src.orientation,
                    codomain_permutation,
                    domain_permutation,
                )
                .map_err(map_checked_generic_trace_symbol_error)
            },
        )
    }

    fn split(
        rule: &R,
        tree: FusionTreeRef<'_>,
        rank: usize,
        rows: &mut TreeScratch,
    ) -> Result<(usize, usize), Self::Error> {
        split_fusion_tree_generic_checked_into(rule, tree, rank, rows)
            .map_err(map_checked_generic_trace_structure_error)
    }

    fn channel_factor(
        rule: &R,
        trace_tree: FusionTreeRef<'_>,
    ) -> Result<Self::Scalar, Self::Error> {
        let first = trace_tree.uncoupled().first().copied().ok_or_else(|| {
            CheckedGenericPlanError::Core(symtn_core::CoreError::MalformedFusionTree {
                message: "trace channel requires at least one uncoupled sector",
            })
        })?;
        let dim_coupled = rule
            .try_dim_scalar(trace_tree.coupled())
            .map_err(CheckedGenericPlanError::Provider)?;
        let dim_first = rule
            .try_dim_scalar(first)
            .map_err(CheckedGenericPlanError::Provider)?;
        // Why not `sqrt_dim(c)² · inv_sqrt_dim(first)²`: four roundings put an
        // integer-dimension quotient (SU(3) `15/3`) ulps off TensorKit's
        // `dim(c) / dim(first)`, which this single division reproduces.
        let mut factor = dim_coupled / dim_first;
        for (&sector, &is_dual) in trace_tree
            .uncoupled()
            .iter()
            .zip(trace_tree.is_dual())
            .skip(1)
        {
            if !is_dual {
                factor = factor
                    * rule
                        .try_twist_scalar(sector)
                        .map_err(CheckedGenericPlanError::Provider)?;
            }
        }
        Ok(factor)
    }
}

#[cfg(test)]
mod channel_factor_tests {
    use super::*;
    use num_complex::Complex64;
    use symtn_core::{FusionTreeKey, TableRule};

    /// Fibonacci `(τ) → τ` and `(τ, τ) → τ` trace trees, where TensorKit's
    /// `dim(τ)/dim(τ)` is exactly 1 and a non-dual second `τ` contributes
    /// `twist(τ) = cispi(-4/5)` (hand values). The previous `φ · fl(1/φ)`
    /// factor gave `1 - 2⁻⁵³`.
    #[test]
    fn multiplicity_free_channel_factor_is_the_dimension_quotient() {
        let rule = TableRule::fibonacci();
        // The table's derived twist (Σ_b d_b/d_τ R^{ττ}_b), which is within
        // (1, 1) ulp of the correctly rounded hand value `cispi(-4/5)`; the
        // factor multiplies it by an exact 1, so it is pinned bitwise.
        let twist = rule.twist_scalar(symtn_core::SectorId::new(1));
        let hand = Complex64::new(-0.8090169943749475, -0.5877852522924731);
        assert!(
            (twist - hand).norm() <= 2.0 * f64::EPSILON,
            "{twist} != {hand}"
        );
        let tree = |uncoupled: &[usize], coupled, dual: &[bool], vertices: &[usize]| {
            FusionTreeKey::try_from_sector_ids_for_rule(
                &rule,
                uncoupled.iter().copied(),
                coupled,
                dual.iter().copied(),
                [],
                vertices.iter().copied(),
            )
            .unwrap()
        };
        let cases = [
            (tree(&[1], 1, &[false], &[]), Complex64::new(1.0, 0.0)),
            (
                tree(&[1, 1], 1, &[false, true], &[1]),
                Complex64::new(1.0, 0.0),
            ),
            (tree(&[1, 1], 1, &[false, false], &[1]), twist),
        ];
        for (trace_tree, want) in cases {
            let actual = <MultiplicityFreeAdmissionMode as TraceTermMode<_>>::channel_factor(
                &rule,
                trace_tree.as_view(),
            )
            .unwrap();
            assert_eq!(
                (actual.re.to_bits(), actual.im.to_bits()),
                (want.re.to_bits(), want.im.to_bits()),
                "{trace_tree:?}: {actual} != {want}"
            );
        }
    }
}
