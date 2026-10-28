use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use symtn_core::{
    BlockKey, FusionSpaceAdmission, FusionTreeHomSpace, FusionTreePairKey,
    FusionTreePairOrientation, MultiplicityFreeFusionRule, MultiplicityFreeRigidSymbols,
    OrientedFusionTreeHomSpace,
};

use super::metadata::{dispatch_prepare, LayoutKeyBuilder, MetadataOutput, MetadataRequest};
use super::{tree_transform_operation_axes, DynamicFusionMapSpace, TransformedLayoutProbe};
use crate::tree_transform::OrientedBasisOrder;
use crate::{OperationError, TreeTransformOperation};

/// Internal contraction operand separating categorical and storage authority.
///
/// Naming note: `prelowered` in the `*_prelowered_*` contraction entry
/// points names THIS lazy-adjoint operand split — logical geometry prepared
/// separately from parent storage before plan compilation. It is unrelated to
/// the `*_lowered` layout-staging family (`prepare_fusion_tree_layout_lowered`
/// cold sector enumeration); the two families collide only in name.
///
/// The parent space defines physical storage; orientation derives the logical
/// HomSpace and user-axis view. Why not retain a logical adjoint space: it is
/// duplicate authority that can drift from the parent representation.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct FusionOperand<'a> {
    storage_space: &'a DynamicFusionMapSpace,
    orientation: FusionTreePairOrientation,
}

pub(crate) struct FusionOperandLayout<'a> {
    operand: FusionOperand<'a>,
    homspace: Cow<'a, FusionTreeHomSpace>,
    projection: FusionOperandProjection,
    basis_order: OrientedBasisOrder,
}

enum FusionOperandProjection {
    Direct,
    Adjoint {
        logical_keys: Arc<[FusionTreePairKey]>,
        /// Filled on first use. Why not in `prepare`: a warm call takes its
        /// plans from the Runtime store and never reads this map, while the
        /// per-block lookups and the Vec cost a Complete parent on every call.
        storage_indices: OnceLock<Result<Vec<usize>, OperationError>>,
    },
}

impl<'a> FusionOperandLayout<'a> {
    #[inline]
    pub(crate) fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace.as_ref()
    }

    #[inline]
    pub(crate) fn nout(&self) -> usize {
        self.operand.oriented_homspace().nout()
    }

    #[inline]
    pub(crate) fn rank(&self) -> usize {
        self.operand.storage_space().rank()
    }

    #[inline]
    pub(crate) fn logical_block_count(&self) -> usize {
        match &self.projection {
            FusionOperandProjection::Direct => self.storage_space().structure().block_count(),
            FusionOperandProjection::Adjoint { logical_keys, .. } => logical_keys.len(),
        }
    }

    pub(crate) fn logical_key(
        &self,
        logical_index: usize,
    ) -> Result<&FusionTreePairKey, OperationError> {
        match &self.projection {
            FusionOperandProjection::Direct => {
                match self.storage_space().structure().block(logical_index)?.key() {
                    BlockKey::FusionTree(key) => Ok(key),
                    _ => Err(OperationError::StructureMismatch {
                        tensor: "direct fusion operand",
                    }),
                }
            }
            FusionOperandProjection::Adjoint { logical_keys, .. } => logical_keys
                .get(logical_index)
                .ok_or_else(|| OperationError::BlockIndexOutOfBounds {
                    tensor: "logical src",
                    index: logical_index,
                    count: logical_keys.len(),
                }),
        }
    }

    #[inline]
    pub(crate) fn storage_index(&self, logical_index: usize) -> Result<usize, OperationError> {
        match &self.projection {
            FusionOperandProjection::Direct => {
                self.storage_space().structure().block(logical_index)?;
                Ok(logical_index)
            }
            FusionOperandProjection::Adjoint { logical_keys, .. } => self
                .adjoint_storage_indices()?
                .get(logical_index)
                .copied()
                .ok_or_else(|| OperationError::BlockIndexOutOfBounds {
                    tensor: "logical src",
                    index: logical_index,
                    count: logical_keys.len(),
                }),
        }
    }

    /// The parent storage index of each logical key, in logical order.
    ///
    /// Canonical preparation proves the parent-key bijection; storage-ordered
    /// preparation derives each logical key from its parent index. Both allow
    /// this projection to be deferred until a compiler miss.
    ///
    /// # Panics
    ///
    /// On a direct operand, which has no projection.
    pub(crate) fn adjoint_storage_indices(&self) -> Result<&[usize], OperationError> {
        let FusionOperandProjection::Adjoint {
            logical_keys,
            storage_indices,
        } = &self.projection
        else {
            unreachable!("only adjoint operands carry a storage projection")
        };
        storage_indices
            .get_or_init(|| match self.basis_order {
                OrientedBasisOrder::Canonical => {
                    adjoint_storage_indices(self.storage_space(), logical_keys)
                }
                OrientedBasisOrder::Storage => Ok((0..logical_keys.len()).collect()),
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    #[inline]
    pub(crate) fn is_direct(&self) -> bool {
        matches!(self.projection, FusionOperandProjection::Direct)
    }

    /// The logical keys in the prepared basis order; `None` when direct.
    #[inline]
    pub(crate) fn adjoint_logical_keys(&self) -> Option<&[FusionTreePairKey]> {
        match &self.projection {
            FusionOperandProjection::Direct => None,
            FusionOperandProjection::Adjoint { logical_keys, .. } => Some(logical_keys),
        }
    }

    #[inline]
    pub(crate) fn storage_space(&self) -> &'a DynamicFusionMapSpace {
        self.operand.storage_space()
    }

    #[inline]
    pub(crate) fn storage_conjugate(&self) -> bool {
        self.operand.storage_conjugate()
    }

    #[inline]
    pub(crate) fn basis_order(&self) -> OrientedBasisOrder {
        self.basis_order
    }

    #[inline]
    pub(crate) fn orientation(&self) -> FusionTreePairOrientation {
        self.operand.orientation()
    }

    #[inline]
    pub(crate) fn storage_axis(&self, logical_axis: usize) -> Result<usize, OperationError> {
        self.operand.storage_axis(logical_axis)
    }

    #[inline]
    pub(crate) fn admission(&self) -> &FusionSpaceAdmission {
        self.storage_space().admission()
    }

    pub(crate) fn transformed_layout_probe<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<TransformedLayoutProbe, OperationError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let homspace = match primer(
            rule,
            MetadataRequest::Permute {
                homspace: self.homspace(),
                codomain_axes,
                domain_axes,
            },
        )? {
            MetadataOutput::HomSpace { homspace, .. } => homspace,
            _ => unreachable!("metadata dispatcher returned a non-HomSpace response"),
        };
        // Why not build an oriented BlockStructure: a conjugated source is
        // borrowed only through the artifact's canonical storage-adjoint
        // proof, never through this equality, which is only observed for
        // the direct orientation where the parent structure is authoritative.
        let (required_len, source_structure_matches) = homspace
            .coupled_subblock_layout_probe_uncached(rule, self.storage_space().structure())
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(TransformedLayoutProbe {
            nout: codomain_axes.len(),
            homspace,
            required_len,
            source_structure_matches,
        })
    }

    pub(crate) fn transformed_space<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<DynamicFusionMapSpace, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        DynamicFusionMapSpace::permuted_memoized(
            rule,
            self.storage_space(),
            self.orientation(),
            self.homspace(),
            operation,
            primer,
        )
    }
}

impl<'a> FusionOperand<'a> {
    pub fn direct(space: &'a DynamicFusionMapSpace) -> Self {
        Self {
            storage_space: space,
            orientation: FusionTreePairOrientation::Direct,
        }
    }

    pub fn adjoint(storage_space: &'a DynamicFusionMapSpace) -> Self {
        Self {
            storage_space,
            orientation: FusionTreePairOrientation::Adjoint,
        }
    }

    #[inline]
    pub fn storage_space(self) -> &'a DynamicFusionMapSpace {
        self.storage_space
    }

    #[inline]
    pub fn storage_conjugate(self) -> bool {
        self.orientation == FusionTreePairOrientation::Adjoint
    }

    #[inline]
    pub(crate) fn orientation(self) -> FusionTreePairOrientation {
        self.orientation
    }

    #[inline]
    pub(crate) fn oriented_homspace(self) -> OrientedFusionTreeHomSpace<'a> {
        OrientedFusionTreeHomSpace::new(self.storage_space.homspace(), self.orientation())
    }

    /// The logical hom space: the parent's own, or its memoized adjoint.
    pub(crate) fn logical_homspace(self) -> &'a FusionTreeHomSpace {
        match self.orientation() {
            FusionTreePairOrientation::Direct => self.storage_space.homspace(),
            FusionTreePairOrientation::Adjoint => {
                self.storage_space.adjoint_memo().homspace.as_ref()
            }
        }
    }

    pub(crate) fn prepare<R>(
        self,
        rule: &R,
        layout_primer: LayoutKeyBuilder<R>,
    ) -> Result<FusionOperandLayout<'a>, OperationError>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.storage_space.validate_rule(rule)?;
        if self.orientation() == FusionTreePairOrientation::Direct {
            return Ok(FusionOperandLayout {
                operand: self,
                homspace: Cow::Borrowed(self.storage_space.homspace()),
                projection: FusionOperandProjection::Direct,
                basis_order: OrientedBasisOrder::Canonical,
            });
        }

        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::FusionOperandProjectionPrepare);

        // The parent's memoized adjoint HomSpace equals
        // `oriented_homspace().materialize()` without a per-call build.
        let homspace = Cow::Borrowed(self.storage_space.adjoint_memo().homspace.as_ref());
        let prepared = dispatch_prepare(layout_primer, rule, homspace.as_ref())?;
        let all_logical_keys = prepared.keys(rule, homspace.as_ref());
        let projection = if matches!(
            self.storage_space.admission(),
            FusionSpaceAdmission::Complete(_)
        ) {
            // A Complete parent holds every canonical key, so the logical keys
            // are the canonical ones and the storage map can wait for a reader.
            FusionOperandProjection::Adjoint {
                logical_keys: all_logical_keys,
                storage_indices: OnceLock::new(),
            }
        } else {
            let structure = self.storage_space.structure();
            let mut logical_keys = Vec::with_capacity(structure.block_count());
            let mut storage_indices = Vec::with_capacity(structure.block_count());
            for logical_key in all_logical_keys.iter() {
                if let Some(index) =
                    structure.find_block_index_by_adjoint_fusion_tree_pair(logical_key)
                {
                    logical_keys.push(logical_key.clone());
                    storage_indices.push(index);
                }
            }
            if storage_indices.len() != structure.block_count() {
                return Err(OperationError::StructureMismatch {
                    tensor: "operand block projection",
                });
            }
            FusionOperandProjection::Adjoint {
                logical_keys: Arc::from(logical_keys),
                storage_indices: OnceLock::from(Ok(storage_indices)),
            }
        };
        prepared.commit();
        Ok(FusionOperandLayout {
            operand: self,
            homspace,
            projection,
            basis_order: OrientedBasisOrder::Canonical,
        })
    }

    /// Static adjoint views retain the parent's block order, including an
    /// admitted subset and arbitrary strides. Canonical enumeration would
    /// change group/provider and accumulation order; derive only swapped keys.
    /// TensorKit preserves parent block traversal while swapping trees and axes:
    /// <https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/adjoint.jl#L22-L50>.
    /// The borrowed parent and index projection are the Rust ownership adaptation.
    pub(crate) fn prepare_storage_ordered_adjoint<R>(
        storage_space: &'a DynamicFusionMapSpace,
        rule: &R,
    ) -> Result<FusionOperandLayout<'a>, OperationError>
    where
        R: symtn_core::FusionRule,
    {
        storage_space.validate_rule(rule)?;
        Self::adjoint(storage_space).storage_ordered_layout()
    }

    /// This operand's layout in its parent's block order, read from the
    /// stored keys alone: no provider is consulted, so a checked planner
    /// may build it after its destination is staged (#2046). The caller
    /// has checked the storage space against its rule.
    pub(crate) fn storage_ordered_layout(self) -> Result<FusionOperandLayout<'a>, OperationError> {
        let storage_space = self.storage_space;
        if self.orientation() == FusionTreePairOrientation::Direct {
            return Ok(FusionOperandLayout {
                operand: self,
                homspace: Cow::Borrowed(storage_space.homspace()),
                projection: FusionOperandProjection::Direct,
                basis_order: OrientedBasisOrder::Canonical,
            });
        }
        let structure = storage_space.structure();
        let mut logical_keys = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index,
                });
            };
            logical_keys.push(FusionTreePairKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ));
        }
        Ok(FusionOperandLayout {
            operand: self,
            homspace: Cow::Owned(self.oriented_homspace().materialize()),
            projection: FusionOperandProjection::Adjoint {
                logical_keys: logical_keys.into(),
                storage_indices: OnceLock::new(),
            },
            basis_order: OrientedBasisOrder::Storage,
        })
    }

    pub(crate) fn storage_axis(self, logical_axis: usize) -> Result<usize, OperationError> {
        if logical_axis >= self.storage_space.rank() {
            return Err(OperationError::InvalidAxisSet {
                tensor: "logical src",
                axes: vec![logical_axis],
                rank: self.storage_space.rank(),
            });
        }
        Ok(match self.orientation() {
            FusionTreePairOrientation::Direct => logical_axis,
            FusionTreePairOrientation::Adjoint => {
                if logical_axis < self.storage_space.nin() {
                    self.storage_space.nout() + logical_axis
                } else {
                    logical_axis - self.storage_space.nin()
                }
            }
        })
    }

    pub(crate) fn storage_block_index(
        self,
        logical_key: &FusionTreePairKey,
    ) -> Result<usize, OperationError> {
        let structure = self.storage_space().structure();
        let index = if self.storage_conjugate() {
            structure.find_block_index_by_adjoint_fusion_tree_pair(logical_key)
        } else {
            structure.find_block_index_by_fusion_tree_pair(logical_key)
        };
        index.ok_or_else(|| OperationError::MissingBlockKey {
            key: Box::new(BlockKey::from(logical_key.clone())),
        })
    }
}

/// Maps each canonical logical key of a Complete parent to its storage block.
fn adjoint_storage_indices(
    storage_space: &DynamicFusionMapSpace,
    logical_keys: &[FusionTreePairKey],
) -> Result<Vec<usize>, OperationError> {
    let structure = storage_space.structure();
    let storage_indices = logical_keys
        .iter()
        .map(|logical_key| {
            structure
                .find_block_index_by_adjoint_fusion_tree_pair(logical_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(BlockKey::from(logical_key.clone())),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if storage_indices.len() != structure.block_count() {
        return Err(OperationError::StructureMismatch {
            tensor: "operand block projection",
        });
    }
    Ok(storage_indices)
}
