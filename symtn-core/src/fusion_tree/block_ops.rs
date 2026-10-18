use super::*;

#[derive(Clone, Copy)]
pub(super) struct ValidatedFusionTreeBlockGroup<'a, R> {
    rule: &'a R,
    pub(super) src_keys: &'a [FusionTreeKey],
    rank: usize,
    pub(super) projection: MultiplicityFreeTreeProjection<'a>,
}

#[cfg(any(test, feature = "testing"))]
fn validate_fusion_tree_block_group_for_rule<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreeKey],
) -> Result<Option<ValidatedFusionTreeBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    for source in src_keys {
        validate_fusion_tree_for_rule(rule, source)?;
    }
    validate_fusion_tree_block_group_proven(rule, src_keys)
}

fn validate_fusion_tree_block_group_proven<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreeKey],
) -> Result<Option<ValidatedFusionTreeBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let projection = MultiplicityFreeTreeProjection::from_validated(rule, src_keys)?;
    let Some(reference) = src_keys.first() else {
        return Ok(None);
    };
    let same_group = |key: &FusionTreeKey| {
        key.uncoupled() == reference.uncoupled() && key.is_dual() == reference.is_dual()
    };
    for source in src_keys {
        // Why not compare `coupled`: distinct coupled labels are basis states
        // within one external-sector group, notably for non-Abelian SU(2)
        // blocks. Group membership is checked source-by-source so the first
        // malformed source is reported before any later group mismatch.
        if !same_group(source) {
            return Err(CoreError::MalformedFusionTree {
                message: "fusion-tree keys must share one group",
            });
        }
    }
    Ok(Some(ValidatedFusionTreeBlockGroup {
        rule,
        src_keys,
        rank: reference.uncoupled().len(),
        projection,
    }))
}

#[allow(clippy::type_complexity)]
pub(crate) fn multiplicity_free_braid_tree_block_proven<R>(
    batch: ValidatedMultiplicityFreeTreeBatch<'_, R>,
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let rank = first.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
    let group = validate_fusion_tree_block_group_proven(rule, src_keys)?
        .expect("nonempty proven source block produces a group proof");
    multiplicity_free_braid_tree_block_validated(group, prepared)
}

/// Apply one braid to every source tree in an all-codomain block.
///
/// Sources and result rows retain source order. Floating-point summation can
/// differ from repeated scalar calls, so coefficients agree numerically rather
/// than necessarily bit-for-bit. Empty input returns an empty result. Every
/// nonempty source must share external sectors, duality, and fusion style;
/// malformed mixed groups fail before symbol evaluation.
///
/// Every source follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[allow(clippy::type_complexity)]
#[cfg(any(test, feature = "testing"))]
pub(crate) fn multiplicity_free_braid_tree_block<R>(
    rule: &R,
    src_keys: &[FusionTreeKey],
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let rank = first.uncoupled().len();
    if levels.len() != rank {
        return Err(CoreError::DimensionMismatch {
            expected: rank,
            actual: levels.len(),
        });
    }
    let prepared = PreparedTreeBraid::new(permutation, levels, rank)?;
    let group = validate_fusion_tree_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_block_validated(group, prepared)
}

#[allow(clippy::type_complexity)]
fn multiplicity_free_braid_tree_block_validated<R>(
    group: ValidatedFusionTreeBlockGroup<'_, R>,
    prepared: PreparedTreeBraid,
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let src_keys = group.src_keys;
    if prepared.permutation.iter().copied().eq(0..group.rank) {
        return Ok(src_keys
            .iter()
            .map(|key| vec![(key.clone(), R::Scalar::one())])
            .collect());
    }

    let (basis, columns) = run_artin_steps(
        (CompactMultiplicityFreeTreeBasis::from_group(group)?, None),
        prepared.artin_steps.iter().copied(),
        |(basis, columns), step| {
            let (basis, columns) =
                compact_artin_tree_block(rule, basis, columns.as_ref(), step.index, step.inverse)?;
            Ok::<_, CoreError>((basis, Some(columns)))
        },
    )?;

    // A validated non-identity permutation always contains at least one swap.
    let columns = columns.expect("non-identity permutation produces an Artin step");
    Ok(scatter_compact_tree_block(basis, &columns))
}

/// Symmetric-braiding convenience wrapper for
/// [`multiplicity_free_braid_tree_block`], with the same ordering, validation,
/// and empty-input contract.
///
/// Every source follows [`FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[allow(clippy::type_complexity)]
#[cfg(test)]
pub(crate) fn multiplicity_free_permute_tree_block<R>(
    rule: &R,
    src_keys: &[FusionTreeKey],
    permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let rank = first.uncoupled().len();
    let levels = (0..rank).collect::<SmallVec<[usize; 8]>>();
    let prepared = PreparedTreeBraid::new(permutation, &levels, rank)?;
    let group = validate_fusion_tree_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_block_validated(group, prepared)
}

#[derive(Clone, Copy)]
pub(crate) struct ValidatedTreePairBlockGroup<'a, R> {
    pub(super) rule: &'a R,
    pub(super) source_len: usize,
    pub(crate) codomain_rank: usize,
    pub(crate) domain_rank: usize,
    pub(super) projection: MultiplicityFreePairProjection<'a>,
}

pub(crate) const TREE_PAIR_BLOCK_GROUP_ERROR: &str = "fusion-tree block keys must share one group";

#[cfg(any(test, feature = "testing"))]
pub(crate) fn validate_tree_pair_block_group_for_rule<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreePairKey],
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    for source in src_keys {
        validate_fusion_tree_pair_for_rule(rule, source)?;
    }
    let projection = MultiplicityFreePairProjection::from_validated(rule, src_keys)?;
    validate_tree_pair_block_group_projection(rule, projection)
}

fn validate_tree_pair_block_group_proven<'a, R>(
    rule: &'a R,
    src_keys: &'a [FusionTreePairKey],
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let projection = MultiplicityFreePairProjection::from_validated(rule, src_keys)?;
    validate_tree_pair_block_group_projection(rule, projection)
}

fn validate_tree_pair_block_group_structure<'a, R>(
    rule: &'a R,
    structure: &'a BlockStructure,
    indices: &'a [usize],
    orientation: FusionTreePairOrientation,
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let projection =
        MultiplicityFreePairProjection::checked_structure(rule, structure, indices, orientation)?;
    validate_tree_pair_block_group_projection(rule, projection)
}

fn validate_tree_pair_block_group_projection<'a, R>(
    rule: &'a R,
    projection: MultiplicityFreePairProjection<'a>,
) -> Result<Option<ValidatedTreePairBlockGroup<'a, R>>, CoreError>
where
    R: FusionRule,
{
    let Some(reference) = projection.pair_at(0) else {
        return Ok(None);
    };
    let reference_codomain = reference.codomain().key();
    let reference_domain = reference.domain().key();
    let same_group = |pair: ValidatedMultiplicityFreeTreePair<'_>| {
        let codomain = pair.codomain().key();
        let domain = pair.domain().key();
        codomain.uncoupled() == reference_codomain.uncoupled()
            && domain.uncoupled() == reference_domain.uncoupled()
            && codomain.is_dual() == reference_codomain.is_dual()
            && domain.is_dual() == reference_domain.is_dual()
    };
    for index in 0..projection.len() {
        let source = projection
            .pair_at(index)
            .expect("validated projection covers every source");
        // Why not infer a block from matching ranks or tree shape:
        // coefficients share a basis only when every external sector,
        // orientation, and multiplicity invariant agrees. Coupled sectors are
        // basis states within one group and intentionally need not match.
        if !same_group(source) {
            return Err(CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            });
        }
    }
    Ok(Some(ValidatedTreePairBlockGroup {
        rule,
        source_len: projection.len(),
        codomain_rank: reference_codomain.uncoupled().len(),
        domain_rank: reference_domain.uncoupled().len(),
        projection,
    }))
}

#[allow(clippy::type_complexity)]
pub(crate) fn multiplicity_free_braid_tree_pair_block_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(Vec::new());
    };
    multiplicity_free_braid_tree_pair_block_validated(group, prepared)
}

pub(crate) fn multiplicity_free_braid_tree_pair_block_ordered_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_braid_tree_pair_block_ordered_validated(group, prepared)
}

#[allow(clippy::type_complexity)]
pub(crate) fn multiplicity_free_transpose_tree_pair_block_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(Vec::new());
    };
    multiplicity_free_transpose_tree_pair_block_validated(group, prepared)
}

pub(crate) fn multiplicity_free_transpose_tree_pair_block_ordered_proven<R>(
    batch: ValidatedMultiplicityFreePairBatch<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let (rule, src_keys) = batch.parts();
    let Some(group) = validate_tree_pair_block_group_proven(rule, src_keys)? else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_transpose_tree_pair_block_ordered_validated(group, prepared)
}

/// Batched [`multiplicity_free_braid_tree_pair`] over every source tree-pair of
/// a block (all sharing the same uncoupled sectors / duality). Returns, per
/// source (in `src_keys` order), its `(destination tree-pair, coefficient)`
/// rows — identical content to calling the per-source function on each, but the
/// bend/braid step structure is walked once for the block.
///
/// The floating-point *summation order* of coefficients that reach a
/// destination by several paths differs from the per-source accumulator, so
/// results agree with the per-source version to double-precision rounding, not
/// necessarily bit-for-bit.
///
/// `src_keys` must be empty or share one external-sector, duality, and
/// multiplicity group. Empty input returns an empty transform; mixed groups
/// return [`CoreError::MalformedFusionTree`] before any symbol evaluation.
///
/// Every source follows
/// [`FusionTreePairKey::validate_for_rule`]'s provider-domain precondition.
#[expect(
    clippy::type_complexity,
    reason = "the public block transform API exposes source-major coefficient rows directly"
)]
#[cfg(test)]
pub(crate) fn multiplicity_free_braid_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let prepared = PreparedTreePairOperation::prepare_braid(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?;
    let group = validate_tree_pair_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_pair_block_validated(group, &prepared)
}

#[allow(clippy::type_complexity)]
fn multiplicity_free_braid_tree_pair_block_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok((0..group.source_len)
            .map(|index| vec![validated_identity_tree_pair_row(&group, index)])
            .collect());
    }
    multiplicity_free_braid_tree_pair_block_compact_validated(group, prepared)
        .map(scatter_compact_tree_pair_block)
}

/// [`multiplicity_free_braid_tree_pair_block_validated`] as present
/// per-source columns, each column in ascending destination row as the rows
/// form lists them, so lowering visits the same rows in the same order.
fn multiplicity_free_braid_tree_pair_block_columns_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<BlockSourceColumns<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok(BlockSourceColumns::identity(
            (0..group.source_len)
                .map(|index| validated_identity_tree_pair_row(&group, index).0)
                .collect(),
        ));
    }
    multiplicity_free_braid_tree_pair_block_compact_validated(group, prepared)
        .map(columns_compact_tree_pair_block)
}

fn validated_identity_tree_pair_row<R>(
    group: &ValidatedTreePairBlockGroup<'_, R>,
    index: usize,
) -> (FusionTreePairKey, R::Scalar)
where
    R: MultiplicityFreeRigidSymbols,
{
    let key = group
        .projection
        .pair_at(index)
        .expect("validated projection covers every source")
        .materialize();
    (key, R::Scalar::one())
}

fn validated_identity_tree_pair_block_ordered<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
) -> OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>
where
    R: MultiplicityFreeRigidSymbols,
{
    // TensorKit's zero-move repartition returns the source with coefficient one:
    // https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/fusiontrees/duality_manipulations.jl#L477-L491
    // QSpace likewise avoids permutation staging for unchanged axes:
    // https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/QSpace.hh#lines-2837:2851
    // Validated singleton columns retain scalar conversion and repeated source
    // columns without constructing the nonidentity compact repartition state.
    let mut destinations = Vec::with_capacity(group.source_len);
    let mut destination_rows = Vec::with_capacity(group.source_len);
    let mut coefficients = Vec::with_capacity(group.source_len);
    let unique = group.projection.has_strictly_ordered_structure_indices();
    let mut rows = FxHashMap::default();
    for index in 0..group.source_len {
        let (key, coefficient) = validated_identity_tree_pair_row(&group, index);
        let next = destinations.len();
        let row = if unique {
            destinations.push(key);
            next
        } else {
            *rows.entry(key.clone()).or_insert_with(|| {
                destinations.push(key);
                next
            })
        };
        destination_rows.push(row);
        coefficients.push(coefficient);
    }
    OrderedBlockLinearMap {
        destinations,
        source_count: group.source_len,
        storage: OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        },
    }
}

fn multiplicity_free_braid_tree_pair_block_ordered_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok(validated_identity_tree_pair_block_ordered(group));
    }

    multiplicity_free_braid_tree_pair_block_compact_validated(group, prepared)
        .map(order_compact_tree_pair_block)
}

fn multiplicity_free_braid_tree_pair_block_compact_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let codomain_rank = group.codomain_rank;
    let domain_rank = group.domain_rank;
    match &prepared.plan {
        PreparedTreePairPlan::Identity => {
            return compact_repartition_tree_pair_block(group, codomain_rank);
        }
        PreparedTreePairPlan::Repartition => {
            return compact_repartition_tree_pair_block(group, prepared.target_codomain_rank);
        }
        PreparedTreePairPlan::Braid(_)
        | PreparedTreePairPlan::SimpleBraid(_)
        | PreparedTreePairPlan::UniqueBraid(_) => {}
        PreparedTreePairPlan::Transpose { .. } => {
            unreachable!("braid preparation cannot create a transpose plan")
        }
    }
    let all_rank = codomain_rank + domain_rank;

    // The first compact operator writes source columns directly; later
    // operators compose through the resulting dense coefficient matrix. The
    // braid swaps act on the all-codomain tree ONE adjacent swap at a time,
    // each batched across the whole block — the TensorKit 0.17
    // `artin_braid`-on-a-block scheme.
    let basis = CompactMultiplicityFreeTreePairBasis::from_group(group)?;
    let steps = prepared
        .plan
        .artin_steps()
        .ok_or(CoreError::MalformedFusionTree {
            message: "a block braid requires an Artin schedule",
        })?;
    let (basis, columns) = block_braid(
        &mut CompactTreePairDriver { rule },
        (basis, None),
        codomain_rank,
        all_rank,
        prepared.target_codomain_rank,
        steps,
    )?;

    // Why not materialize after each braid: both block runners keep the
    // external frame immutable and reconstruct full keys only at the API edge.
    Ok(CompactMultiplicityFreeTreePairBlock {
        basis,
        columns: columns.expect("nonidentity braid executes at least one compact operator"),
        records_dimensions: true,
    })
}

/// Batched [`multiplicity_free_permute_tree_pair`] over a block: symmetric
/// braiding with the trivial level ordering.
///
/// The group contract is identical to
/// [`multiplicity_free_braid_tree_pair_block`]. Symmetric braiding remains a
/// required capability even for an empty source block.
///
/// Every source follows
/// [`FusionTreePairKey::validate_for_rule`]'s provider-domain precondition.
#[expect(
    clippy::type_complexity,
    reason = "the public block transform API exposes source-major coefficient rows directly"
)]
#[cfg(any(test, feature = "testing"))]
pub(crate) fn multiplicity_free_permute_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let prepared = PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let group = validate_tree_pair_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_pair_block_validated(group, &prepared)
}

/// Whole-block multiplicity-free permutation of one fusion-tree group of
/// `structure`, read in `orientation`, as present per-source columns: the
/// trace lowering's permutation (TensorKit `_trace_permute!` permutes a
/// whole `FusionTreeBlock`).
#[doc(hidden)]
pub fn multiplicity_free_permute_tree_pair_block_indexed<R>(
    rule: &R,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<BlockSourceColumns<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    validate_multiplicity_free_execution_style(rule)?;
    let Some(&first_index) = src_indices.first() else {
        return Ok(BlockSourceColumns::empty());
    };
    let first = structure.block(first_index)?;
    let BlockKey::FusionTree(first) = first.key() else {
        return Err(CoreError::ExpectedFusionTreePairKey {
            actual: first.key().kind(),
        });
    };
    let (codomain_rank, domain_rank) = match orientation {
        FusionTreePairOrientation::Direct => (
            first.codomain_tree().uncoupled().len(),
            first.domain_tree().uncoupled().len(),
        ),
        FusionTreePairOrientation::Adjoint => (
            first.domain_tree().uncoupled().len(),
            first.codomain_tree().uncoupled().len(),
        ),
    };
    let prepared = PreparedTreePairOperation::prepare_permute(
        rule,
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let group =
        validate_tree_pair_block_group_structure(rule, structure, src_indices, orientation)?
            .expect("nonempty source block produces a validation proof");
    multiplicity_free_braid_tree_pair_block_columns_validated(group, &prepared)
}

/// Style-admitted seam of the `symtn-tensors` oriented tree-transform miss
/// path: the caller has already rejected the rule's fusion style with its own
/// `OperationError::UnsupportedFusionStyle` gate, so this checks the style
/// only in debug builds (#1950).
#[doc(hidden)]
pub fn multiplicity_free_braid_tree_pair_block_ordered_indexed<R>(
    rule: &R,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    debug_assert!(rule.fusion_style().is_multiplicity_free());
    prepared.validate_block_preflight(rule, PreparedTreePairFamily::BraidLike)?;
    let Some(group) =
        validate_tree_pair_block_group_structure(rule, structure, src_indices, orientation)?
    else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_braid_tree_pair_block_ordered_validated(group, prepared)
}

/// Style-admitted seam of the `symtn-tensors` oriented tree-transform miss
/// path: the caller has already rejected the rule's fusion style with its own
/// `OperationError::UnsupportedFusionStyle` gate, so this checks the style
/// only in debug builds (#1950).
#[doc(hidden)]
pub fn multiplicity_free_transpose_tree_pair_block_ordered_indexed<R>(
    rule: &R,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    debug_assert!(rule.fusion_style().is_multiplicity_free());
    prepared.validate_block_preflight(rule, PreparedTreePairFamily::Transpose)?;
    let Some(group) =
        validate_tree_pair_block_group_structure(rule, structure, src_indices, orientation)?
    else {
        return Ok(OrderedBlockLinearMap {
            destinations: Vec::new(),
            source_count: 0,
            storage: OrderedBlockLinearStorage::DenseDstSrc(Vec::new()),
        });
    };
    multiplicity_free_transpose_tree_pair_block_ordered_validated(group, prepared)
}

/// Batched [`multiplicity_free_transpose_tree_pair`] over every source
/// tree-pair of a block (all sharing uncoupled sectors / duality). The planar
/// cyclic-transpose step sequence — repartition to the target codomain rank,
/// then rotate the coupled loop one leg at a time — depends only on the ranks
/// and permutation, so it is identical for every source. Walk it once over the
/// shared `DenseColumns` matrix instead of replaying the repartition and cyclic
/// bends per source (TensorKit 0.17's block `fstranspose`). Returns, per source
/// in `src_keys` order, its `(destination tree-pair, coefficient)` rows.
///
/// As with the braid block port, coefficients that reach a destination by
/// several paths sum in a different order than the per-source accumulator, so
/// results agree to double-precision rounding, not necessarily bit-for-bit.
///
/// The empty/group contract is identical to
/// [`multiplicity_free_braid_tree_pair_block`].
///
/// Every source follows
/// [`FusionTreePairKey::validate_for_rule`]'s provider-domain precondition.
#[expect(
    clippy::type_complexity,
    reason = "the public block transform API exposes source-major coefficient rows directly"
)]
#[cfg(test)]
pub(crate) fn multiplicity_free_transpose_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    validate_multiplicity_free_execution_style(rule)?;
    let Some(first) = src_keys.first() else {
        return Ok(Vec::new());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let prepared = PreparedTreePairOperation::prepare_transpose(
        codomain_rank,
        domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let group = validate_tree_pair_block_group_for_rule(rule, src_keys)?
        .expect("nonempty source block produces a validation proof");
    multiplicity_free_transpose_tree_pair_block_validated(group, &prepared)
}

#[allow(clippy::type_complexity)]
fn multiplicity_free_transpose_tree_pair_block_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok((0..group.source_len)
            .map(|index| {
                let key = group
                    .projection
                    .pair_at(index)
                    .expect("validated projection covers every source")
                    .materialize();
                vec![(key, R::Scalar::one())]
            })
            .collect());
    }
    multiplicity_free_transpose_tree_pair_block_compact_validated(group, prepared)
        .map(scatter_compact_tree_pair_block)
}

pub(crate) fn multiplicity_free_transpose_tree_pair_block_ordered_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    prepared.validate_source_split(group.codomain_rank, group.domain_rank)?;
    if prepared.is_identity() {
        return Ok(validated_identity_tree_pair_block_ordered(group));
    }

    multiplicity_free_transpose_tree_pair_block_compact_validated(group, prepared)
        .map(order_compact_tree_pair_block)
}

fn multiplicity_free_transpose_tree_pair_block_compact_validated<R>(
    group: ValidatedTreePairBlockGroup<'_, R>,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<CompactMultiplicityFreeTreePairBlock<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let rule = group.rule;
    let codomain_rank = group.codomain_rank;
    let cycle = match &prepared.plan {
        PreparedTreePairPlan::Identity => {
            return compact_repartition_tree_pair_block(group, codomain_rank);
        }
        PreparedTreePairPlan::Repartition => {
            return compact_repartition_tree_pair_block(group, prepared.target_codomain_rank);
        }
        PreparedTreePairPlan::Transpose { direction, count } => (*direction, *count),
        PreparedTreePairPlan::Braid(_)
        | PreparedTreePairPlan::SimpleBraid(_)
        | PreparedTreePairPlan::UniqueBraid(_) => {
            unreachable!("transpose preparation cannot create a braid plan")
        }
    };

    let basis = CompactMultiplicityFreeTreePairBasis::from_group(group)?;
    let (basis, columns) = block_transpose(
        &mut CompactTreePairDriver { rule },
        (basis, None),
        codomain_rank,
        prepared.target_codomain_rank,
        Some(cycle),
    )?;

    // Why not materialize after each fold: the external frame is identical for
    // every local row and remains immutable until this ordered API boundary.
    #[cfg(test)]
    assert_compact_tree_pair_basis_in_homspace(rule, &basis);
    Ok(CompactMultiplicityFreeTreePairBlock {
        basis,
        columns: columns.expect("nonidentity transpose executes at least one compact operator"),
        records_dimensions: true,
    })
}

#[cfg(test)]
type MultiplicityFreeTreePairBlockRows<S> = Vec<Vec<(FusionTreePairKey, S)>>;

#[cfg(test)]
fn multiplicity_free_foldright_tree_pair_stepwise_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let codomain = tree_pair.codomain_tree();
    if codomain.uncoupled().is_empty() {
        return Err(CoreError::MalformedFusionTree {
            message: "foldright requires at least one codomain leg",
        });
    }
    let a = codomain.uncoupled()[0];
    let is_dual_a = codomain
        .is_dual()
        .first()
        .copied()
        .ok_or(CoreError::MalformedFusionTree {
            message: "codomain tree is missing the first duality flag",
        })?;
    let kappa = rule.frobenius_schur_phase_scalar(a);
    let c = codomain.coupled();

    let mut terms = FusionTermAccumulator::new();
    for (codomain_prime, coeff1) in
        multiplicity_free_multi_fmove_tree_stepwise_oracle(rule, codomain)?
    {
        let b = codomain_prime.coupled();
        let a_symbol = rule.a_symbol_scalar(a, b, c);
        let coeff0 = rule.sqrt_dim_scalar(c) * rule.inv_sqrt_dim_scalar(b);
        for (domain_prime, coeff2) in multiplicity_free_multi_fmove_inv_tree_stepwise_oracle(
            rule,
            rule.dual(a),
            b,
            tree_pair.domain_tree(),
            !is_dual_a,
        )? {
            let mut coefficient =
                coeff0.clone() * (coeff2).conj() * a_symbol.clone() * coeff1.clone();
            if is_dual_a {
                coefficient = coefficient * kappa.clone();
            }
            terms.push(
                FusionTreePairKey::pair(codomain_prime.clone(), domain_prime),
                coefficient,
            );
        }
    }
    Ok(terms.into_vec())
}

#[cfg(test)]
fn multiplicity_free_foldleft_tree_pair_stepwise_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let swapped = FusionTreePairKey::pair(
        tree_pair.domain_tree().clone(),
        tree_pair.codomain_tree().clone(),
    );
    Ok(
        multiplicity_free_foldright_tree_pair_stepwise_oracle(rule, &swapped)?
            .into_iter()
            .map(|(folded, coefficient)| {
                (
                    FusionTreePairKey::pair(
                        folded.domain_tree().clone(),
                        folded.codomain_tree().clone(),
                    ),
                    (coefficient).conj(),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
fn multiplicity_free_cycle_clockwise_tree_pair_stepwise_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.codomain_tree().uncoupled().is_empty() {
        multiplicity_free_bendleft_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldright_tree_pair_stepwise_oracle(rule, tree_pair)?
    };
    if tree_pair.codomain_tree().uncoupled().is_empty() {
        compose_terms(first, |key| {
            multiplicity_free_foldright_tree_pair_stepwise_oracle(rule, key)
        })
    } else {
        compose_terms(first, |key| multiplicity_free_bendleft_tree_pair(rule, key))
    }
}

#[cfg(test)]
fn multiplicity_free_cycle_anticlockwise_tree_pair_stepwise_oracle<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let first: Vec<_> = if tree_pair.domain_tree().uncoupled().is_empty() {
        multiplicity_free_bendright_tree_pair(rule, tree_pair)?
            .into_iter()
            .collect()
    } else {
        multiplicity_free_foldleft_tree_pair_stepwise_oracle(rule, tree_pair)?
    };
    if tree_pair.domain_tree().uncoupled().is_empty() {
        compose_terms(first, |key| {
            multiplicity_free_foldleft_tree_pair_stepwise_oracle(rule, key)
        })
    } else {
        compose_terms(first, |key| {
            multiplicity_free_bendright_tree_pair(rule, key)
        })
    }
}

#[cfg(test)]
pub(crate) fn multiplicity_free_transpose_tree_pair_block_full_key_oracle<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<MultiplicityFreeTreePairBlockRows<R::Scalar>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let Some(group) = validate_tree_pair_block_group_for_rule(rule, src_keys)? else {
        return Ok(Vec::new());
    };
    let prepared = PreparedTreePairOperation::prepare_transpose(
        group.codomain_rank,
        group.domain_rank,
        codomain_permutation,
        domain_permutation,
    )?;
    let cycle = match &prepared.plan {
        PreparedTreePairPlan::Identity => {
            return Ok(src_keys
                .iter()
                .map(|key| vec![(key.clone(), R::Scalar::one())])
                .collect());
        }
        PreparedTreePairPlan::Repartition => None,
        PreparedTreePairPlan::Transpose { direction, count } => Some((*direction, *count)),
        PreparedTreePairPlan::Braid(_)
        | PreparedTreePairPlan::SimpleBraid(_)
        | PreparedTreePairPlan::UniqueBraid(_) => {
            unreachable!("transpose preparation cannot create a braid plan")
        }
    };

    let num_src = src_keys.len();
    let mut columns = DenseColumns::with_capacity(num_src, num_src);
    for source in 0..num_src {
        let row = columns.push_empty_row();
        columns.row_mut(row)[source] = Some(R::Scalar::one());
    }

    let repartitioned = repartition_loop(
        (src_keys.to_vec(), columns),
        group.codomain_rank,
        prepared.target_codomain_rank,
        |(basis, columns), bend| {
            compose_block_terms(&basis, &columns, |key| match bend {
                Bend::Left => multiplicity_free_bendleft_tree_pair(rule, key),
                Bend::Right => multiplicity_free_bendright_tree_pair(rule, key),
            })
        },
    )?;
    let (basis, columns) = run_cycles(repartitioned, cycle, |(basis, columns), direction| {
        compose_block_terms(&basis, &columns, |key| match direction {
            PreparedCycleDirection::Clockwise => {
                multiplicity_free_cycle_clockwise_tree_pair_stepwise_oracle(rule, key)
            }
            PreparedCycleDirection::Anticlockwise => {
                multiplicity_free_cycle_anticlockwise_tree_pair_stepwise_oracle(rule, key)
            }
        })
    })?;

    let mut rows_per_source = vec![Vec::new(); num_src];
    for (destination_row, destination) in basis.iter().enumerate() {
        for (source, coefficient) in columns.row(destination_row).iter().enumerate() {
            if let Some(coefficient) = coefficient {
                rows_per_source[source].push((destination.clone(), coefficient.clone()));
            }
        }
    }
    Ok(rows_per_source)
}

/// The multiplicity-free column-batched driver of the shared block schedule.
struct CompactTreePairDriver<'a, R> {
    rule: &'a R,
}

impl<R> BlockDriver for CompactTreePairDriver<'_, R>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    type State = (
        CompactMultiplicityFreeTreePairBasis,
        Option<DenseColumns<R::Scalar>>,
    );
    type Error = CoreError;
    type BraidSchedule<'s> = PreparedTreePairArtinSteps<'s>;

    fn bend(
        &mut self,
        (basis, columns): Self::State,
        bend: Bend,
    ) -> Result<Self::State, CoreError> {
        let (basis, columns) = match bend {
            Bend::Left => compact_bendleft_block(self.rule, basis, columns.as_ref())?,
            Bend::Right => compact_bendright_block(self.rule, basis, columns.as_ref())?,
        };
        Ok((basis, Some(columns)))
    }

    fn braid_codomain(
        &mut self,
        state: Self::State,
        steps: PreparedTreePairArtinSteps<'_>,
    ) -> Result<Self::State, CoreError> {
        // Why a plain loop over a monomorphic move, not `run_artin_steps` with
        // a step closure: the closure form measured ~1-3% slower on the cold
        // braid plan build (#1852).
        let (mut basis, mut columns) = state;
        for step in steps {
            let (next_basis, next_columns) = compact_codomain_artin_block(
                self.rule,
                basis,
                columns.as_ref(),
                step.index,
                step.inverse,
            )?;
            basis = next_basis;
            columns = Some(next_columns);
        }
        Ok((basis, columns))
    }

    fn cycle(
        &mut self,
        (basis, columns): Self::State,
        direction: PreparedCycleDirection,
    ) -> Result<Self::State, CoreError> {
        let (basis, columns) = match direction {
            PreparedCycleDirection::Clockwise => {
                compact_cycle_clockwise_block(self.rule, basis, columns.as_ref())?
            }
            PreparedCycleDirection::Anticlockwise => {
                compact_cycle_anticlockwise_block(self.rule, basis, columns.as_ref())?
            }
        };
        Ok((basis, Some(columns)))
    }
}
