use std::collections::HashMap;

use symtn_core::{
    merge_fusion_trees_generic_checked, merge_fusion_trees_multiplicity_free, BlockKey,
    CanonicalUnitFusionRule, CategoricalScalar, CheckedFusionAlgebra, CheckedFusionSpaceError,
    CheckedGenericFusion, CheckedGenericRigidSymbols, CheckedGenericStructureError,
    CheckedGenericSymbolError, CoreError, FusionProductSpace, FusionTreeHomSpace,
    FusionTreePairKey, MultiplicityFreeRigidSymbols, MultiplicityIndex, RuleIdentity,
};
use symtn_matrixalgebra::SectorSpectrum;
use symtn_operations::TreeTransformBlock;
use symtn_tensors::{
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, RecouplingCoefficientAction,
    TreeTransformOperation, TreeTransformOperationKind, TreeTransformRuleCacheKey,
    TreeTransformStructure,
};

use crate::error::Error;
use crate::typed::ScalarOps;

/// Converts an internal coupled-layout invariant violation into the stable
/// crate error used by both the typed facade and the execution core.
pub(crate) fn internal_layout_error(what: &str) -> Error {
    Error::InvalidArgument(format!(
        "internal coupled-layout invariant violated ({what}); please report this"
    ))
}

/// Error from fallible checked-Generic tensor-product execution.
///
/// Provider failures and malformed F arrays remain distinct from local tensor
/// layout failures. Tensor product is not a tree-transform plan, so this type
/// deliberately does not reuse `CheckedGenericPlanError`.
#[derive(Debug)]
pub enum CheckedGenericTensorProductError<E> {
    /// The checked provider rejected an algebra or coefficient query.
    Provider(E),
    /// A returned F array did not have the shape fixed by its sectors.
    SymbolShape {
        /// The malformed categorical symbol.
        symbol: &'static str,
        /// Shape required by the fusion multiplicities.
        expected: Vec<usize>,
        /// Shape returned by the provider.
        actual: Vec<usize>,
    },
    /// A provider-independent categorical invariant failed.
    Core(CoreError),
    /// A local tensor layout or payload invariant failed.
    Operation(symtn_tensors::OperationError),
}

impl<E> From<CoreError> for CheckedGenericTensorProductError<E> {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

impl<E> From<symtn_tensors::OperationError> for CheckedGenericTensorProductError<E> {
    fn from(error: symtn_tensors::OperationError) -> Self {
        Self::Operation(error)
    }
}

impl<E> From<CheckedGenericStructureError<E>> for CheckedGenericTensorProductError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Provider(error) => Self::Provider(error),
            CheckedGenericStructureError::Core(error) => Self::Core(error),
        }
    }
}

impl<E: core::fmt::Display> core::fmt::Display for CheckedGenericTensorProductError<E> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Provider(_) => formatter.write_str("fusion provider failed"),
            Self::SymbolShape {
                symbol,
                expected,
                actual,
            } => write!(
                formatter,
                "{symbol} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::Core(_) => formatter.write_str("invalid tensor structure"),
            Self::Operation(_) => formatter.write_str("tensor operation failed"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericTensorProductError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::SymbolShape { .. } => None,
        }
    }
}

fn map_checked_tensor_product_symbol_error<E>(
    error: CheckedGenericSymbolError<E>,
) -> CheckedGenericTensorProductError<E> {
    match error {
        CheckedGenericSymbolError::Provider(error) => {
            CheckedGenericTensorProductError::Provider(error)
        }
        CheckedGenericSymbolError::Shape {
            symbol,
            expected,
            actual,
        } => CheckedGenericTensorProductError::SymbolShape {
            symbol,
            expected,
            actual,
        },
        CheckedGenericSymbolError::Core(error) => CheckedGenericTensorProductError::Core(error),
    }
}

/// Whether `operation` on a tensor of these ranks is the rank-(1,1) leg swap
/// (TensorKit `permute`/`transpose` of a `DiagonalTensorMap`,
/// `src/tensors/diagonal.jl:215-273`) whose compact result [`bond_spectrum`]
/// reads.
pub(crate) fn is_rank_one_diagonal_swap(
    codomain_rank: usize,
    domain_rank: usize,
    operation: &TreeTransformOperation,
) -> bool {
    codomain_rank == 1
        && domain_rank == 1
        && operation.codomain_permutation() == [1]
        && operation.domain_permutation() == [0]
        && matches!(
            operation.kind(),
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Transpose
        )
}

/// Whether `operation` is the rank-(1,1) braid whose dense result
/// [`bond_spectrum`] lets the compact arm read from the spectrum.
pub(crate) fn is_rank_one_diagonal_braid(
    codomain_rank: usize,
    domain_rank: usize,
    operation: &TreeTransformOperation,
) -> bool {
    codomain_rank == 1
        && domain_rank == 1
        && operation.kind() == TreeTransformOperationKind::Braid
        && operation.codomain_permutation() == [1]
        && operation.domain_permutation() == [0]
}

fn offsets_never_decrease(structure: &symtn_core::BlockStructure) -> bool {
    (1..structure.block_count()).all(|index| {
        match (structure.block(index - 1), structure.block(index)) {
            (Ok(previous), Ok(next)) => previous.offset() <= next.offset(),
            _ => false,
        }
    })
}

/// The non-empty block at `offset` in a structure whose block offsets never
/// decrease. Empty blocks may share that offset; at most one non-empty block
/// owns it, because destination storage is injective.
fn nonempty_block_at_sorted_offset(
    structure: &symtn_core::BlockStructure,
    offset: usize,
) -> Option<usize> {
    let (mut low, mut high) = (0, structure.block_count());
    while low < high {
        let middle = low + (high - low) / 2;
        if structure.block(middle).ok()?.offset() < offset {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    (low..structure.block_count())
        .map_while(|index| {
            let block = structure.block(index).ok()?;
            (block.offset() == offset).then_some((index, block))
        })
        .find(|(_, block)| block.shape().iter().product::<usize>() != 0)
        .map(|(index, _)| index)
}

/// The compact spectrum carried through a resolved rank-(1,1) transform, or
/// `None` unless `compiled` proves a one-to-one, single-term map between all
/// square bond blocks with equal shapes. Each destination entry is its source
/// entry scaled by the same compiled coefficient, through the same
/// [`RecouplingCoefficientAction::scale_by_coefficient`], that dense replay of
/// this structure applies, so the result is the dense route's diagonal by
/// construction. With `require_representable`, a coefficient that is not a
/// finite non-zero value of `D` also declines, so the dense replay's
/// off-diagonal `0 · c` products (NaN for an infinite `c`) are kept.
///
/// The output is in destination block order, which is the compact payload
/// order of the destination space.
pub(crate) fn bond_spectrum<C, D>(
    sources: &symtn_core::BlockStructure,
    destinations: &symtn_core::BlockStructure,
    compiled: &TreeTransformStructure<C>,
    spectrum: &[SectorSpectrum<D>],
    require_representable: bool,
) -> Option<Vec<SectorSpectrum<D>>>
where
    C: Copy,
    D: ScalarOps + RecouplingCoefficientAction<C>,
{
    if sources.block_count() != spectrum.len()
        || sources.block_count() != destinations.block_count()
        || compiled.blocks().len() != sources.block_count()
    {
        return None;
    }
    // A compact bond has one physical block per spectrum entry in sector order.
    // Check that correspondence before using compiled layout offsets as identities.
    for (index, entry) in spectrum.iter().enumerate() {
        let source_block = sources.block(index).ok()?;
        let source_pair = source_block.key().as_fusion_tree_pair()?;
        let [rows, columns] = source_block.shape() else {
            return None;
        };
        if source_pair.codomain_tree().coupled() != entry.sector
            || rows != columns
            || entry.values.len() != *rows
        {
            return None;
        }
    }
    let mut source_offsets = None;
    let mut destination_sorted = None;
    let mut seen_source = smallvec::SmallVec::<[bool; 16]>::from_elem(false, sources.block_count());
    let mut seen_destination =
        smallvec::SmallVec::<[bool; 16]>::from_elem(false, destinations.block_count());
    let mut writes = smallvec::SmallVec::<[(usize, usize, C); 16]>::new();
    let layouts = compiled.layouts();
    for prepared in compiled.blocks() {
        let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = prepared
        else {
            return None;
        };
        let src = layouts.entry(*src_layout);
        let dst = layouts.entry(*dst_layout);
        // Single layouts are inserted as destination/source pairs before the
        // replay blocks are weight-sorted. Check that this pair's source is
        // the compact spectrum ordinal; only unusual ordering needs a map.
        let ordinal = src_layout
            .checked_sub(1)
            .filter(|_| src_layout % 2 == 1 && dst_layout.checked_add(1) == Some(*src_layout))
            .map(|index| index / 2)
            .filter(|&index| index < sources.block_count())
            .filter(|&index| {
                sources
                    .block(index)
                    .ok()
                    .and_then(|block| isize::try_from(block.offset()).ok())
                    == Some(src.offset)
            });
        let source_index = if let Some(ordinal) = ordinal {
            ordinal
        } else {
            let offsets = source_offsets.get_or_insert_with(|| {
                (0..sources.block_count())
                    .filter_map(|index| {
                        let block = sources.block(index).ok()?;
                        Some((isize::try_from(block.offset()).ok()?, index))
                    })
                    .collect::<HashMap<_, _>>()
            });
            if offsets.len() != sources.block_count() {
                return None;
            }
            *offsets.get(&src.offset)?
        };
        if std::mem::replace(&mut seen_source[source_index], true) {
            return None;
        }
        let source_block = sources.block(source_index).ok()?;
        let [rows, cols] = source_block.shape() else {
            return None;
        };
        // Zero degeneracy is an absent sector, so a bond block is never
        // empty; an empty one is not a bond and has no offset identity.
        if src.element_count != rows.checked_mul(*cols)?
            || dst.element_count != src.element_count
            || dst.element_count == 0
        {
            return None;
        }
        let coefficient = compiled.single_coefficient(*coefficient)?;
        if require_representable {
            let converted = D::coefficient_as_data(coefficient);
            if converted == D::from_real(0.0) || !converted.abs_value().is_finite() {
                return None;
            }
        }
        // The destination block's geometry comes from the destination
        // structure, not the compiled table: a completed transform retains only
        // its normalized replay roles, whose axis order and rank need not be
        // the block's. A non-empty block is identified by its offset, because
        // compilation rejects aliased destinations.
        //
        // Why the source-axis order needs no proof here: the compiled Single
        // maps a square source block onto a destination block of the same
        // shape, either identically or transposed, and both fix a diagonal.
        let destination_index = destinations
            .block(source_index)
            .ok()
            .filter(|block| isize::try_from(block.offset()).ok() == Some(dst.offset))
            .map(|_| source_index);
        let destination_index = if let Some(index) = destination_index {
            index
        } else {
            // A braid or swap sends sector c to its dual, which sits at
            // another ordinal whenever the sector set is not self-dual, so
            // this path is common. Canonical structures store blocks in offset
            // order, where a binary search finds the block without allocating.
            //
            // Why decline instead of indexing an unordered layout: every
            // destination is canonical, and the general dense replay is
            // correct for any layout, so an offset map here would only add an
            // allocation no caller needs.
            if !*destination_sorted.get_or_insert_with(|| offsets_never_decrease(destinations)) {
                return None;
            }
            nonempty_block_at_sorted_offset(destinations, usize::try_from(dst.offset).ok()?)?
        };
        if std::mem::replace(&mut seen_destination[destination_index], true)
            || destinations.block(destination_index).ok()?.shape() != source_block.shape()
        {
            return None;
        }
        writes.push((destination_index, source_index, coefficient));
    }
    if seen_source.iter().any(|&covered| !covered) {
        return None;
    }
    writes.sort_unstable_by_key(|&(destination_index, ..)| destination_index);
    writes
        .into_iter()
        .map(|(destination_index, source_index, coefficient)| {
            let destination = destinations.block(destination_index).ok()?;
            Some(SectorSpectrum {
                sector: destination
                    .key()
                    .as_fusion_tree_pair()?
                    .codomain_tree()
                    .coupled(),
                values: spectrum[source_index]
                    .values
                    .iter()
                    .map(|&value| value.scale_by_coefficient(coefficient))
                    .collect(),
            })
        })
        .collect()
}

/// TensorKit tensor product: merge codomain trees with codomain trees and
/// domain trees with domain trees, without crossing either pair of legs.
pub(crate) fn tensorproduct_owned_multiplicity_free<R, D, C>(
    lhs: BoundDynamicTensorRef<'_, R, D>,
    rhs: BoundDynamicTensorRef<'_, R, D>,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), symtn_tensors::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C>
        + CheckedFusionAlgebra
        + CanonicalUnitFusionRule
        + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    C: CategoricalScalar,
    D: ScalarOps + RecouplingCoefficientAction<C>,
{
    let rule = lhs.space().provider();
    if rule.rule_identity() != rhs.space().provider().rule_identity() {
        return Err(symtn_tensors::OperationError::Core(
            CoreError::FusionRuleMismatch {
                expected: rule.rule_identity(),
                actual: rhs.space().provider().rule_identity(),
            },
        ));
    }
    let lhs_hom = lhs.space().space().homspace();
    let rhs_hom = rhs.space().space().homspace();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(
            lhs_hom
                .codomain()
                .legs()
                .iter()
                .chain(rhs_hom.codomain().legs())
                .cloned(),
        ),
        FusionProductSpace::new(
            lhs_hom
                .domain()
                .legs()
                .iter()
                .chain(rhs_hom.domain().legs())
                .cloned(),
        ),
    );
    let prepared_destination = homspace
        .prepare_fusion_tree_layout_checked(rule)
        .map_err(|error| symtn_tensors::OperationError::FusionAlgebra(Box::new(error)))?;
    let destination_indices = prepared_destination
        .keys()
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<HashMap<_, _>>();

    struct Contribution<C> {
        lhs: usize,
        rhs: usize,
        destination: usize,
        coefficient: C,
    }
    let lhs_structure = lhs.space().space().structure();
    let rhs_structure = rhs.space().space().structure();
    let mut contributions = Vec::new();
    for lhs_index in 0..lhs_structure.block_count() {
        let BlockKey::FusionTree(lhs_key) = lhs_structure.block(lhs_index)?.key() else {
            return Err(symtn_tensors::OperationError::ExpectedFusionTreeBlock {
                tensor: "lhs",
                index: lhs_index,
            });
        };
        for rhs_index in 0..rhs_structure.block_count() {
            let BlockKey::FusionTree(rhs_key) = rhs_structure.block(rhs_index)?.key() else {
                return Err(symtn_tensors::OperationError::ExpectedFusionTreeBlock {
                    tensor: "rhs",
                    index: rhs_index,
                });
            };
            for coupled in rule
                .try_fusion_channels(
                    lhs_key.codomain_tree().coupled(),
                    rhs_key.codomain_tree().coupled(),
                )
                .map_err(|error| symtn_tensors::OperationError::FusionAlgebra(Box::new(error)))?
            {
                let codomain = merge_fusion_trees_multiplicity_free(
                    rule,
                    lhs_key.codomain_tree(),
                    rhs_key.codomain_tree(),
                    coupled,
                )
                .map_err(tensor_product_checked_error)?;
                let domain = merge_fusion_trees_multiplicity_free(
                    rule,
                    lhs_key.domain_tree(),
                    rhs_key.domain_tree(),
                    coupled,
                )
                .map_err(tensor_product_checked_error)?;
                for (codomain, codomain_coefficient) in &codomain {
                    for (domain, domain_coefficient) in &domain {
                        let key = FusionTreePairKey::pair(codomain.clone(), domain.clone());
                        let destination_index =
                            destination_indices.get(&key).copied().ok_or_else(|| {
                                symtn_tensors::OperationError::MissingBlockKey {
                                    key: Box::new(BlockKey::FusionTree(key)),
                                }
                            })?;
                        contributions.push(Contribution {
                            lhs: lhs_index,
                            rhs: rhs_index,
                            destination: destination_index,
                            coefficient: codomain_coefficient.clone() * domain_coefficient.conj(),
                        });
                    }
                }
            }
        }
    }

    // Checked provider work and exact destination-key validation precede
    // destination admission. Only deterministic prepared-key replay remains.
    let destination = lhs.space().derive_from_final_homspace(homspace)?;
    let dst_structure = destination.space().structure();
    if dst_structure.block_count() != prepared_destination.keys().len()
        || prepared_destination
            .keys()
            .iter()
            .enumerate()
            .any(|(index, key)| {
                dst_structure.find_block_index_by_fusion_tree_pair(key) != Some(index)
            })
    {
        return Err(symtn_tensors::OperationError::StructureMismatch {
            tensor: "tensor-product destination",
        });
    }

    // Multiple recoupling paths deliberately accumulate below.
    let mut data = vec![D::from_real(0.0); destination.space().required_len()?];
    let lhs_nout = lhs.space().space().nout();
    let rhs_nout = rhs.space().space().nout();
    for contribution in contributions {
        let lhs_block = lhs_structure.block(contribution.lhs)?;
        let rhs_block = rhs_structure.block(contribution.rhs)?;
        let dst_block = dst_structure.block(contribution.destination)?;
        scatter_tensor_product_block(
            lhs.data(),
            lhs_block,
            lhs_nout,
            rhs.data(),
            rhs_block,
            rhs_nout,
            &mut data,
            dst_block,
            contribution.coefficient,
        )?;
    }
    Ok((destination, data))
}

type CheckedGenericTensorProductResult<R, D> = Result<
    (BoundDynamicFusionMapSpace<R>, Vec<D>),
    CheckedGenericTensorProductError<<R as CheckedGenericFusion>::Error>,
>;

fn validate_tensor_product_source_structure<E>(
    tensor: &'static str,
    structure: &symtn_core::BlockStructure,
) -> Result<(), CheckedGenericTensorProductError<E>> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(
                symtn_tensors::OperationError::ExpectedFusionTreeBlock { tensor, index }.into(),
            );
        };
        if block.shape().len() != structure.rank() || block.strides().len() != structure.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: structure.rank(),
                actual: block.shape().len(),
            }
            .into());
        }
        let key_rank = key.codomain_tree().uncoupled().len() + key.domain_tree().uncoupled().len();
        if key_rank != structure.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: structure.rank(),
                actual: key_rank,
            }
            .into());
        }
        for tree in [key.codomain_tree(), key.domain_tree()] {
            let rank = tree.uncoupled().len();
            if tree.is_dual().len() != rank
                || tree.innerlines().len() != rank.saturating_sub(2)
                || tree.vertices().len() != rank.saturating_sub(1)
            {
                return Err(CoreError::MalformedFusionTree {
                    message: "tensor-product source fusion-tree arrays have inconsistent lengths",
                }
                .into());
            }
        }
        if key.codomain_tree().coupled() != key.domain_tree().coupled() {
            return Err(CoreError::MalformedFusionTree {
                message: "tensor-product source tree pair has mismatched coupled sectors",
            }
            .into());
        }
    }
    Ok(())
}

fn validate_tensor_product_source_structures<E>(
    lhs: &symtn_core::BlockStructure,
    rhs: &symtn_core::BlockStructure,
) -> Result<(), CheckedGenericTensorProductError<E>> {
    validate_tensor_product_source_structure("lhs", lhs)?;
    validate_tensor_product_source_structure("rhs", rhs)
}

pub(crate) fn tensorproduct_owned_checked_generic<R, D>(
    lhs_space: &BoundDynamicFusionMapSpace<R>,
    lhs_data: &[D],
    rhs_space: &BoundDynamicFusionMapSpace<R>,
    rhs_data: &[D],
) -> CheckedGenericTensorProductResult<R, D>
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
    D: ScalarOps,
{
    let rule = symtn_tensors::admit_checked_generic_pair(lhs_space, rhs_space)?;

    // Local storage validation is deliberately after identity/style rejection
    // and before any provider enumeration or symbol query.
    let lhs = BoundDynamicTensorRef::try_new(lhs_space, lhs_data)?;
    let rhs = BoundDynamicTensorRef::try_new(rhs_space, rhs_data)?;
    let lhs_structure = lhs.space().space().structure();
    let rhs_structure = rhs.space().space().structure();
    validate_tensor_product_source_structures::<R::Error>(lhs_structure, rhs_structure)?;
    let lhs_hom = lhs.space().space().homspace();
    let rhs_hom = rhs.space().space().homspace();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(
            lhs_hom
                .codomain()
                .legs()
                .iter()
                .chain(rhs_hom.codomain().legs())
                .cloned(),
        ),
        FusionProductSpace::new(
            lhs_hom
                .domain()
                .legs()
                .iter()
                .chain(rhs_hom.domain().legs())
                .cloned(),
        ),
    );
    let prepared = lhs
        .space()
        .prepare_final_homspace_generic_with_checked(rule, homspace)?;
    let destination_indices = (0..prepared.structure().block_count())
        .map(|index| {
            let block = prepared.structure().block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(symtn_tensors::OperationError::ExpectedFusionTreeBlock {
                    tensor: "tensor-product destination",
                    index,
                });
            };
            Ok((key.clone(), index))
        })
        .collect::<Result<HashMap<_, _>, symtn_tensors::OperationError>>()?;

    struct Contribution {
        lhs: usize,
        rhs: usize,
        destination: usize,
        coefficient: f64,
    }
    let mut contributions = Vec::new();
    for lhs_index in 0..lhs_structure.block_count() {
        let BlockKey::FusionTree(lhs_key) = lhs_structure.block(lhs_index)?.key() else {
            return Err(symtn_tensors::OperationError::ExpectedFusionTreeBlock {
                tensor: "lhs",
                index: lhs_index,
            }
            .into());
        };
        for rhs_index in 0..rhs_structure.block_count() {
            let BlockKey::FusionTree(rhs_key) = rhs_structure.block(rhs_index)?.key() else {
                return Err(symtn_tensors::OperationError::ExpectedFusionTreeBlock {
                    tensor: "rhs",
                    index: rhs_index,
                }
                .into());
            };
            let left_root = lhs_key.codomain_tree().coupled();
            let right_root = rhs_key.codomain_tree().coupled();
            let channels = rule
                .try_fusion_channels(left_root, right_root)
                .map_err(CheckedGenericTensorProductError::Provider)?;
            for coupled in channels {
                let multiplicity = rule
                    .try_nsymbol(left_root, right_root, coupled)
                    .map_err(CheckedGenericTensorProductError::Provider)?;
                for mu in 1..=multiplicity {
                    let mu = MultiplicityIndex::new(mu).ok_or(
                        symtn_tensors::OperationError::InvalidArgument {
                            message: "invalid Generic root multiplicity",
                        },
                    )?;
                    let codomain = merge_fusion_trees_generic_checked(
                        rule,
                        lhs_key.codomain_tree(),
                        rhs_key.codomain_tree(),
                        coupled,
                        mu,
                    )
                    .map_err(map_checked_tensor_product_symbol_error)?;
                    let domain = merge_fusion_trees_generic_checked(
                        rule,
                        lhs_key.domain_tree(),
                        rhs_key.domain_tree(),
                        coupled,
                        mu,
                    )
                    .map_err(map_checked_tensor_product_symbol_error)?;
                    for (codomain, codomain_coefficient) in &codomain {
                        for (domain, domain_coefficient) in &domain {
                            let key = FusionTreePairKey::pair(codomain.clone(), domain.clone());
                            let destination =
                                destination_indices.get(&key).copied().ok_or_else(|| {
                                    symtn_tensors::OperationError::MissingBlockKey {
                                        key: Box::new(BlockKey::FusionTree(key)),
                                    }
                                })?;
                            contributions.push(Contribution {
                                lhs: lhs_index,
                                rhs: rhs_index,
                                destination,
                                coefficient: *codomain_coefficient * domain_coefficient.conj(),
                            });
                        }
                    }
                }
            }
        }
    }

    // All provider and F-array work is complete before payload allocation.
    // Scatter targets the staged structure; publishing the bound destination
    // is the final operation below.
    let mut data = vec![D::from_real(0.0); prepared.required_len()];
    #[cfg(test)]
    if crate::typed::probe::fires(crate::typed::probe::Fault::TensorProductScatter) {
        return Err(symtn_tensors::OperationError::StructureMismatch {
            tensor: "forced late tensor-product scatter failure",
        }
        .into());
    }
    let lhs_nout = lhs.space().space().nout();
    let rhs_nout = rhs.space().space().nout();
    for contribution in contributions {
        scatter_tensor_product_block(
            lhs.data(),
            lhs_structure.block(contribution.lhs)?,
            lhs_nout,
            rhs.data(),
            rhs_structure.block(contribution.rhs)?,
            rhs_nout,
            &mut data,
            prepared.structure().block(contribution.destination)?,
            contribution.coefficient,
        )?;
    }
    let destination = lhs
        .space()
        .commit_final_homspace_generic_bound_checked(prepared)?;
    #[cfg(test)]
    crate::typed::probe::hit(crate::typed::probe::Event::TensorProductCommit);
    Ok((destination, data))
}

fn tensor_product_checked_error(error: CheckedFusionSpaceError) -> symtn_tensors::OperationError {
    match error {
        CheckedFusionSpaceError::Core(error) => symtn_tensors::OperationError::Core(*error),
        CheckedFusionSpaceError::FusionAlgebra(error) => {
            symtn_tensors::OperationError::FusionAlgebra(error)
        }
        _ => symtn_tensors::OperationError::InvalidArgument {
            message: "unknown checked fusion-tree merge error",
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_tensor_product_block<D, C>(
    lhs_data: &[D],
    lhs: symtn_core::BlockRef<'_>,
    lhs_nout: usize,
    rhs_data: &[D],
    rhs: symtn_core::BlockRef<'_>,
    rhs_nout: usize,
    destination_data: &mut [D],
    destination: symtn_core::BlockRef<'_>,
    coefficient: C,
) -> Result<(), symtn_tensors::OperationError>
where
    D: ScalarOps + RecouplingCoefficientAction<C>,
    C: Clone,
{
    if !destination
        .shape()
        .iter()
        .copied()
        .eq(lhs.shape()[..lhs_nout]
            .iter()
            .chain(&rhs.shape()[..rhs_nout])
            .chain(&lhs.shape()[lhs_nout..])
            .chain(&rhs.shape()[rhs_nout..])
            .copied())
    {
        return Err(symtn_tensors::OperationError::StructureMismatch {
            tensor: "tensor-product destination",
        });
    }

    let lhs_count = lhs.element_count()?;
    let rhs_count = rhs.element_count()?;
    let mut lhs_coordinates = vec![0; lhs.shape().len()];
    let mut rhs_coordinates = vec![0; rhs.shape().len()];
    for _ in 0..lhs_count {
        let lhs_position = block_position(lhs.offset(), lhs.strides(), &lhs_coordinates);
        rhs_coordinates.fill(0);
        for _ in 0..rhs_count {
            let rhs_position = block_position(rhs.offset(), rhs.strides(), &rhs_coordinates);
            let destination_position = destination.offset()
                + lhs_coordinates[..lhs_nout]
                    .iter()
                    .zip(&destination.strides()[..lhs_nout])
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()
                + rhs_coordinates[..rhs_nout]
                    .iter()
                    .zip(&destination.strides()[lhs_nout..lhs_nout + rhs_nout])
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()
                + lhs_coordinates[lhs_nout..]
                    .iter()
                    .zip(
                        &destination.strides()[lhs_nout + rhs_nout
                            ..lhs_nout + rhs_nout + lhs.shape().len() - lhs_nout],
                    )
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>()
                + rhs_coordinates[rhs_nout..]
                    .iter()
                    .zip(
                        &destination.strides()
                            [lhs_nout + rhs_nout + lhs.shape().len() - lhs_nout..],
                    )
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            let value = (lhs_data[lhs_position] * rhs_data[rhs_position])
                .scale_by_coefficient(coefficient.clone());
            destination_data[destination_position] = destination_data[destination_position] + value;
            increment_coordinates(&mut rhs_coordinates, rhs.shape());
        }
        increment_coordinates(&mut lhs_coordinates, lhs.shape());
    }
    Ok(())
}

fn block_position(offset: usize, strides: &[usize], coordinates: &[usize]) -> usize {
    offset
        + coordinates
            .iter()
            .zip(strides)
            .map(|(&index, &stride)| index * stride)
            .sum::<usize>()
}

fn increment_coordinates(coordinates: &mut [usize], shape: &[usize]) {
    for (coordinate, &extent) in coordinates.iter_mut().zip(shape) {
        *coordinate += 1;
        if *coordinate < extent {
            return;
        }
        *coordinate = 0;
    }
}

/// Categorical composition (TensorKit `A * B` / `mul!`) of two owned
/// multiplicity-free operands.
///
#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use symtn_dense::Engine;

    use crate::typed::ContractSpec;
    use symtn_matrixalgebra::SectorSpectrum;

    use symtn_core::{
        BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, CheckedGenericFusion,
        CheckedGenericRigidSymbols, CoreError, FusionProductSpace, FusionRule, FusionStyleKind,
        FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey, GenericFArray, GenericRMatrix,
        MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
        RuleIdentity, SectorId, SectorLeg, SectorVec, Z2FusionRule,
    };
    use symtn_tensors::{
        BoundDynamicFusionMapSpace, FusionOperand, OutputAxisOrder, TreeTransformOperation,
    };

    use super::{scatter_tensor_product_block, tensorproduct_owned_checked_generic};
    use crate::runtime::Ctx;
    use crate::typed::probe::{self, Event, Fault};

    struct CheckedTensorProductSpy {
        algebra_queries: AtomicUsize,
        f_queries: AtomicUsize,
    }

    impl CheckedTensorProductSpy {
        fn reset(&self) {
            self.algebra_queries.store(0, Ordering::Relaxed);
            self.f_queries.store(0, Ordering::Relaxed);
        }
    }

    impl symtn_core::testing::Provider for CheckedTensorProductSpy {}

    impl CheckedGenericFusion for CheckedTensorProductSpy {
        type Error = Infallible;

        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            FusionStyleKind::Generic
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            BraidingStyleKind::Bosonic
        }

        fn vacuum(&self) -> SectorId {
            SectorId::new(0)
        }

        fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
            self.algebra_queries.fetch_add(1, Ordering::Relaxed);
            Ok(sector)
        }

        fn try_fusion_channels(
            &self,
            _left: SectorId,
            _right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.algebra_queries.fetch_add(1, Ordering::Relaxed);
            Ok([SectorId::new(0)].into_iter().collect())
        }

        fn try_fusion_channels_in_table(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.try_fusion_channels(left, right)
        }

        fn try_nsymbol(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
        ) -> Result<usize, Self::Error> {
            self.algebra_queries.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        }
    }

    impl CheckedGenericRigidSymbols for CheckedTensorProductSpy {
        type Scalar = f64;

        fn try_sqrt_dim_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
            Ok(1.0)
        }

        fn try_inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
            Ok(1.0)
        }

        fn try_frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
            Ok(1.0)
        }

        fn try_f_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
            _d: SectorId,
            _e: SectorId,
            _f: SectorId,
        ) -> Result<GenericFArray<f64>, Self::Error> {
            self.f_queries.fetch_add(1, Ordering::Relaxed);
            Ok(GenericFArray::new(vec![1.0], (1, 1, 1, 1)))
        }

        fn try_r_symbol_generic(
            &self,
            _a: SectorId,
            _b: SectorId,
            _c: SectorId,
        ) -> Result<GenericRMatrix<f64>, Self::Error> {
            Ok(GenericRMatrix::new(vec![1.0], 1, 1))
        }
    }

    fn checked_tensor_product_source(
        provider: Arc<CheckedTensorProductSpy>,
    ) -> (
        BoundDynamicFusionMapSpace<CheckedTensorProductSpy>,
        Vec<f64>,
    ) {
        let leg = || SectorLeg::new([(SectorId::new(0), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        );
        let space =
            BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
                .unwrap();
        let data = vec![1.0; space.space().required_len().unwrap()];
        (space, data)
    }

    #[test]
    fn tensor_product_source_validator_rejects_a_later_malformed_key() {
        let sector = SectorId::new(0);
        let tree =
            FusionTreeKey::try_new_for_rule(&Z2FusionRule, [sector], sector, [false], [], [])
                .unwrap();
        let valid = BlockSpec::column_major_with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(tree.clone(), tree)),
            vec![1, 1],
            0,
        )
        .unwrap();
        let rank_zero =
            FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], sector, [], [], []).unwrap();
        let malformed = BlockSpec::column_major_with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(rank_zero.clone(), rank_zero)),
            vec![1, 1],
            1,
        )
        .unwrap();
        let structure = BlockStructure::from_blocks(vec![valid, malformed]).unwrap();

        let error =
            super::validate_tensor_product_source_structure::<Infallible>("rhs", &structure)
                .unwrap_err();

        assert!(matches!(
            error,
            super::CheckedGenericTensorProductError::Core(CoreError::StructureRankMismatch {
                expected: 2,
                actual: 0
            })
        ));
    }

    #[test]
    fn checked_tensor_product_late_scatter_failure_precedes_publication() {
        let provider = Arc::new(CheckedTensorProductSpy {
            algebra_queries: AtomicUsize::new(0),
            f_queries: AtomicUsize::new(0),
        });
        let (source, data) = checked_tensor_product_source(Arc::clone(&provider));
        provider.reset();
        let commits = probe::record();
        probe::arm(Fault::TensorProductScatter, 1);

        let error =
            tensorproduct_owned_checked_generic(&source, &data, &source, &data).unwrap_err();

        assert!(matches!(
            error,
            super::CheckedGenericTensorProductError::Operation(_)
        ));
        assert!(provider.algebra_queries.load(Ordering::Relaxed) > 0);
        assert!(provider.f_queries.load(Ordering::Relaxed) > 0);
        assert_eq!(commits.get(Event::TensorProductCommit), 0);
    }

    #[test]
    fn tensor_product_scatter_accumulates_asymmetric_strided_blocks() {
        // What: two contributions to one destination add rather than
        // overwrite. The gapped source layouts make a contiguous-slice
        // shortcut fail, and the unequal shapes pin the external order.
        let lhs_structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::trivial(),
            vec![2, 3],
            vec![2, 5],
            0,
        )
        .unwrap()])
        .unwrap();
        let rhs_structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            BlockKey::trivial(),
            vec![4, 2],
            vec![1, 7],
            0,
        )
        .unwrap()])
        .unwrap();
        let destination_structure = BlockStructure::trivial(&[2, 4, 3, 2]).unwrap();
        let lhs_block = lhs_structure.only_block().unwrap();
        let rhs_block = rhs_structure.only_block().unwrap();
        let destination_block = destination_structure.only_block().unwrap();
        let mut lhs = vec![f64::NAN; 13];
        let mut rhs = vec![f64::NAN; 11];
        for j in 0..3 {
            for i in 0..2 {
                lhs[i * 2 + j * 5] = (1 + i + 10 * j) as f64;
            }
        }
        for j in 0..2 {
            for i in 0..4 {
                rhs[i + j * 7] = (2 + i + 10 * j) as f64;
            }
        }
        let mut actual = vec![0.0; 48];
        for _ in 0..2 {
            scatter_tensor_product_block(
                &lhs,
                lhs_block,
                1,
                &rhs,
                rhs_block,
                1,
                &mut actual,
                destination_block,
                0.5,
            )
            .unwrap();
        }

        let mut expected = vec![0.0; 48];
        for rhs_domain in 0..2 {
            for lhs_domain in 0..3 {
                for rhs_codomain in 0..4 {
                    for lhs_codomain in 0..2 {
                        let position =
                            lhs_codomain + 2 * rhs_codomain + 8 * lhs_domain + 24 * rhs_domain;
                        expected[position] = (1 + lhs_codomain + 10 * lhs_domain) as f64
                            * (2 + rhs_codomain + 10 * rhs_domain) as f64;
                    }
                }
            }
        }
        assert_eq!(actual, expected);
    }

    /// Deliberately outside the user-layer rule enum: this exercises the typed
    /// core with a provider an application can define without `LoweredMultiplicityFreeAlgebra`.
    struct ExternalZ2<const R_SCALE: u8>;

    impl<const R_SCALE: u8> symtn_core::testing::Provider for ExternalZ2<R_SCALE> {}

    impl<const R_SCALE: u8> FusionRule for ExternalZ2<R_SCALE> {
        fn rule_identity(&self) -> RuleIdentity {
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            Z2FusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            Z2FusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            Z2FusionRule.vacuum()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            Z2FusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            Z2FusionRule.fusion_channels(left, right)
        }
    }

    impl<const R_SCALE: u8> MultiplicityFreeFusionRule for ExternalZ2<R_SCALE> {}

    impl<const R_SCALE: u8> MultiplicityFreeFusionSymbols for ExternalZ2<R_SCALE> {
        type Scalar = f64;

        fn has_trivial_associator_gauge(&self) -> bool {
            Z2FusionRule.has_trivial_associator_gauge()
        }

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
        }

        fn r_symbol_scalar(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Self::Scalar {
            Z2FusionRule.r_symbol_scalar(left, right, coupled)
                * match R_SCALE {
                    1 => 1e-100,
                    2 => 1e100,
                    _ => 1.0,
                }
        }
    }

    impl<const R_SCALE: u8> MultiplicityFreeRigidSymbols for ExternalZ2<R_SCALE> {
        fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
            Z2FusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    /// Compiles the rank-one braid of a two-sector bond with each source
    /// block sent to the destination block of its shape (the degeneracies
    /// differ, so the shape names the braided sector), runs the compact
    /// route, and checks value i of each spectrum entry at offset +
    /// i * (row stride + column stride) of that destination block. Returns
    /// whether some destination sits at another ordinal than its source.
    /// The compact braid's dense result: [`super::bond_spectrum`] with
    /// representability required, densified on the destination.
    fn braid_bond_data<D: crate::typed::ScalarOps>(
        source: &symtn_tensors::DynamicFusionMapSpace,
        destination: &symtn_tensors::DynamicFusionMapSpace,
        compiled: &symtn_tensors::TreeTransformStructure<f64>,
        spectrum: &[SectorSpectrum<D>],
    ) -> Option<Vec<D>> {
        let carried = super::bond_spectrum(
            source.structure(),
            destination.structure(),
            compiled,
            spectrum,
            true,
        )?;
        Some(symtn_matrixalgebra::seam::diagonal_bond_data(destination, &carried, &|v| v).unwrap())
    }

    macro_rules! check_compact_braid_diagonal {
        ($provider:expr, $sectors:expr) => {{
            let provider = Arc::new($provider);
            let sectors: [(SectorId, usize); 2] = $sectors;
            let leg = SectorLeg::new(sectors, false);
            let source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
                Arc::clone(&provider),
                symtn_core::FusionTreeHomSpace::new(
                    FusionProductSpace::new([leg.clone()]),
                    FusionProductSpace::new([leg]),
                ),
            )
            .unwrap();
            let operation = TreeTransformOperation::braid([1], [0], [0], [1]);
            let destination = source.transformed_multiplicity_free(&operation).unwrap();
            let sources = source.space().structure();
            let destinations = destination.space().structure();
            let spectrum = sectors.map(|(sector, n)| SectorSpectrum {
                sector,
                values: (0..n).map(|i| 1.0 + i as f64).collect::<Vec<f64>>(),
            });
            let coefficients = [2.0, -0.5];
            let targets = (0..2)
                .map(|index| {
                    (0..2)
                        .find(|&target| {
                            destinations.block(target).unwrap().shape()
                                == sources.block(index).unwrap().shape()
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>();
            // Reverse spec order, so the source lookup leaves the paired
            // ordinal too.
            let specs = [1, 0].map(|index| {
                symtn_operations::TreeTransformBlockSpec::single(
                    targets[index],
                    index,
                    coefficients[index],
                )
                .with_source_axes([1, 0])
            });
            let compiled = symtn_tensors::TreeTransformStructure::compile_structures(
                destinations,
                sources,
                &specs,
            )
            .unwrap();

            let actual =
                braid_bond_data::<f64>(source.space(), destination.space(), &compiled, &spectrum)
                    .expect("a square diagonal braid takes the compact route");

            let mut expected = vec![0.0; destinations.required_len().unwrap()];
            for (index, entry) in spectrum.iter().enumerate() {
                let block = destinations.block(targets[index]).unwrap();
                for (i, value) in entry.values.iter().enumerate() {
                    let offset = block.offset() + i * block.strides()[0] + i * block.strides()[1];
                    expected[offset] = coefficients[index] * value;
                }
            }
            assert_eq!(actual, expected);
            targets
                .iter()
                .enumerate()
                .any(|(index, &target)| index != target)
        }};
    }

    #[test]
    fn compact_braid_steps_along_each_destination_block_diagonal() {
        // What: the compact braid writes each spectrum along its destination
        // block's diagonal for a 1x1 block (whose compiled role is a single
        // unit axis with no second stride to index) and a 3x3 block. Z2 is
        // self-dual, so each destination keeps its source's ordinal; U(1)
        // charges ±1 swap under the braid, so the destination is found by
        // its offset in the ordered structure.
        let self_dual = check_compact_braid_diagonal!(
            ExternalZ2::<0>,
            [(SectorId::new(0), 1), (SectorId::new(1), 3)]
        );
        assert!(!self_dual, "Z2 destinations keep their source ordinal");
        let reordered = check_compact_braid_diagonal!(
            symtn_core::U1FusionRule,
            [
                (symtn_core::U1Irrep::new(-1).sector_id(), 1),
                (symtn_core::U1Irrep::new(1).sector_id(), 3),
            ]
        );
        assert!(reordered, "the U(1) braid moves a destination ordinal");
    }

    #[test]
    fn compact_braid_declines_coefficients_outside_the_payload_dtype() {
        let provider = Arc::new(ExternalZ2::<2>);
        let leg = SectorLeg::new([(SectorId::new(1), 2)], false);
        let source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            symtn_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone()]),
                FusionProductSpace::new([leg]),
            ),
        )
        .unwrap();
        let operation = TreeTransformOperation::braid([1], [0], [0], [1]);
        let destination = source.transformed_multiplicity_free(&operation).unwrap();
        let f32_spectrum = [SectorSpectrum {
            sector: SectorId::new(1),
            values: vec![1.0f32, 2.0],
        }];
        let compiled = symtn_tensors::TreeTransformStructure::compile_structures(
            destination.space().structure(),
            source.space().structure(),
            &[
                symtn_operations::TreeTransformBlockSpec::single(0, 0, 1e100)
                    .with_source_axes([1, 0]),
            ],
        )
        .unwrap();
        assert!(braid_bond_data::<f32>(
            source.space(),
            destination.space(),
            &compiled,
            &f32_spectrum,
        )
        .is_none());
        let f64_spectrum = [SectorSpectrum {
            sector: SectorId::new(1),
            values: vec![1.0f64, 2.0],
        }];
        assert!(braid_bond_data::<f64>(
            source.space(),
            destination.space(),
            &compiled,
            &f64_spectrum,
        )
        .is_some());
        let underflow_provider = Arc::new(ExternalZ2::<1>);
        let underflow_source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&underflow_provider),
            source.space().homspace().clone(),
        )
        .unwrap();
        let underflow_destination = underflow_source
            .transformed_multiplicity_free(&operation)
            .unwrap();
        let underflow_compiled = symtn_tensors::TreeTransformStructure::compile_structures(
            underflow_destination.space().structure(),
            underflow_source.space().structure(),
            &[
                symtn_operations::TreeTransformBlockSpec::single(0, 0, 1e-100)
                    .with_source_axes([1, 0]),
            ],
        )
        .unwrap();
        assert!(braid_bond_data::<f32>(
            underflow_source.space(),
            underflow_destination.space(),
            &underflow_compiled,
            &f32_spectrum,
        )
        .is_none());
    }

    #[test]
    fn external_multiplicity_free_provider_matches_direct_transform() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let provider = Arc::new(ExternalZ2::<0>);
        let leg = SectorLeg::new([(SectorId::new(0), 2)], false);
        let source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            symtn_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone()]),
                FusionProductSpace::new([leg]),
            ),
        )
        .unwrap();
        let operation = TreeTransformOperation::permute([1], [0]);
        let source_data = vec![1.0, 2.0, 3.0, 4.0];

        let expected_destination = source.transformed_multiplicity_free(&operation).unwrap();
        let mut expected_data = vec![0.0; expected_destination.space().required_len().unwrap()];
        let mut direct = Ctx::<f64, RuleIdentity>::default();
        direct
            .tree_context_mut()
            .tree_transform_dyn_into(
                engine,
                provider.as_ref(),
                operation.clone(),
                expected_destination.space().structure(),
                source.space().structure(),
                &mut expected_data,
                &source_data,
                1.0,
                0.0,
            )
            .unwrap();

        let mut context = Ctx::<f64, RuleIdentity>::default();
        let (actual_destination, actual_data) = context
            .tree_context_mut()
            .tree_transform_owned_multiplicity_free_in(
                engine,
                &source,
                &source_data,
                &operation,
                1.0,
            )
            .unwrap();

        assert_eq!(actual_destination.space(), expected_destination.space());
        assert_eq!(actual_data, expected_data);
    }

    #[test]
    fn external_multiplicity_free_provider_contracts_direct_with_output_order() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let provider = Arc::new(ExternalZ2::<0>);
        let lhs_codomain = SectorLeg::new([(SectorId::new(0), 2)], false);
        let contracted = SectorLeg::new([(SectorId::new(0), 3)], false);
        let rhs_domain = SectorLeg::new([(SectorId::new(0), 4)], false);
        let lhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            symtn_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([lhs_codomain.clone()]),
                FusionProductSpace::new([contracted.clone()]),
            ),
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            symtn_core::FusionTreeHomSpace::new(
                FusionProductSpace::new([contracted]),
                FusionProductSpace::new([rhs_domain.clone()]),
            ),
        )
        .unwrap();
        let lhs_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data = vec![
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ];

        let mut context = Ctx::<f64, RuleIdentity>::default();
        let destination = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &lhs,
            &rhs,
            &[1],
            &[0],
            OutputAxisOrder::from_axes(&[1, 0]),
        )
        .unwrap();
        let mut data = vec![0.0; destination.space().required_len().unwrap()];
        context
            .tensorcontract_planned_into(
                engine,
                &destination,
                &mut data,
                (FusionOperand::direct(lhs.space()), &lhs_data),
                (FusionOperand::direct(rhs.space()), &rhs_data),
                &[1],
                &[0],
                &[1, 0],
                1.0,
                symtn_tensors::ContractDestinationInit::Zeroed,
            )
            .unwrap();

        let expected_destination = symtn_core::FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(0), 4)], true)]),
            FusionProductSpace::new([SectorLeg::new([(SectorId::new(0), 2)], true)]),
        );
        assert_eq!(destination.space().homspace(), &expected_destination);
        assert_eq!(
            data,
            [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
        );
    }

    /// A rank-3 Z2 tensor map built through the typed facade, so the gate
    /// below measures the whole typed route, not this module in isolation.
    fn typed_z2_facade_tensor(
        engine: &mut dyn Engine,
    ) -> (
        crate::typed::Runtime,
        crate::typed::TensorMap<Z2FusionRule, f64>,
    ) {
        let runtime = crate::typed::Runtime::builder().build().unwrap();
        let leg = crate::typed::GradedSpace::try_new(
            Arc::new(Z2FusionRule),
            [
                (symtn_core::Z2Irrep::EVEN, 2),
                (symtn_core::Z2Irrep::ODD, 3),
            ],
        )
        .unwrap();
        let tensor = crate::typed::TensorMap::from_subblock_fn(
            engine,
            &runtime,
            [&leg, &leg],
            [&leg],
            |_, indices| (indices[0] * 7 + indices[1] * 3 + indices[2]) as f64 + 1.0,
        )
        .unwrap();
        (runtime, tensor)
    }

    /// Runs `operation` and returns
    /// `(contract seam calls, tree-transform seam calls)`.
    fn seam_calls<T>(operation: impl FnOnce() -> T) -> (usize, usize) {
        let calls = probe::record();
        let _output = operation();
        (
            calls.get(Event::ContractSeamCall),
            calls.get(Event::TreeTransformSeamCall),
        )
    }

    #[test]
    fn typed_ordered_contract_is_one_fused_seam_call_and_no_permute_transform() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: a typed `contract` with a
        // non-identity output order runs the fused contraction seam exactly
        // once and never a separate permute transform.
        let (_runtime, tensor) = typed_z2_facade_tensor(engine);

        // Negative control: the counters can see the sequential shape at all.
        // A contract followed by a public permute is one fused call plus one
        // transform, which is what a regressed contract-then-permute alias
        // would look like.
        let default = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2, 3],
        };
        let (fused, transforms) = seam_calls(|| {
            tensor
                .contract(engine, &tensor, &default)
                .unwrap()
                .permute(engine, &[1, 0], &[3, 2])
                .unwrap()
        });
        assert_eq!((fused, transforms), (1, 1));

        // The gate: the non-identity order is folded into the one fused call.
        let ordered = ContractSpec {
            codomain: &[1, 0],
            domain: &[3, 2],
            ..default
        };
        let (fused, transforms) =
            seam_calls(|| tensor.contract(engine, &tensor, &ordered).unwrap());
        assert_eq!(
            (fused, transforms),
            (1, 0),
            "ordered contract must be one fused contraction, not contract-then-permute"
        );
    }

    #[test]
    fn typed_split_moving_contract_is_one_fused_seam_call_and_no_permute_transform() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: a `ContractSpec` that moves open legs across
        // the codomain/domain split runs the fused contraction seam once and
        // no separate permute transform, where the composition it is defined
        // by (default-split contract, then permute) runs one of each.
        let (_runtime, tensor) = typed_z2_facade_tensor(engine);
        let default = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2, 3],
        };
        for (codomain, domain) in [
            (&[0, 2, 1][..], &[3][..]),
            (&[3][..], &[1, 0, 2][..]),
            (&[][..], &[2, 0, 3, 1][..]),
        ] {
            let (fused, transforms) = seam_calls(|| {
                tensor
                    .contract(engine, &tensor, &default)
                    .unwrap()
                    .permute(engine, codomain, domain)
                    .unwrap()
            });
            assert_eq!((fused, transforms), (1, 1), "{codomain:?} <- {domain:?}");
            let spec = ContractSpec {
                codomain,
                domain,
                ..default
            };
            let (fused, transforms) =
                seam_calls(|| tensor.contract(engine, &tensor, &spec).unwrap());
            assert_eq!(
                (fused, transforms),
                (1, 0),
                "{codomain:?} <- {domain:?}: split-moving contract must not permute afterwards"
            );
        }
    }
}
