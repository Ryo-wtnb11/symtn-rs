use symtn_core::{
    multiplicity_free_permute_tree_pair, BlockKey, BraidingStyleKind, CategoricalScalar,
    FusionRule, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey, FusionTreePairOrientation,
    MultiplicityFreeRigidSymbols, OrientedFusionTreeHomSpace, SectorId,
};

use crate::{DenseBlockScalar, OperationError};
use symtn_operations::TensorContractSpec;

use super::super::dynamic_space::{DynamicFusionMapSpace, FusionOperandLayout};
use super::super::fusion_block::validate_fusion_contract_rule;
use super::super::structure::{TensorContractAxisPlan, TensorContractBlockSpec};

trait ContractBlockSource {
    fn homspace(&self) -> &FusionTreeHomSpace;
    fn rank(&self) -> usize;
    fn block_count(&self) -> usize;
    fn logical_key(&self, index: usize) -> Result<Option<&FusionTreePairKey>, OperationError>;
    fn storage_index(&self, logical_index: usize) -> Result<usize, OperationError>;
}

impl ContractBlockSource for DynamicFusionMapSpace {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn block_count(&self) -> usize {
        self.structure().block_count()
    }

    fn logical_key(&self, index: usize) -> Result<Option<&FusionTreePairKey>, OperationError> {
        Ok(match self.structure().block(index)?.key() {
            BlockKey::FusionTree(key) => Some(key),
            _ => None,
        })
    }

    fn storage_index(&self, logical_index: usize) -> Result<usize, OperationError> {
        Ok(logical_index)
    }
}

impl ContractBlockSource for FusionOperandLayout<'_> {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn block_count(&self) -> usize {
        self.logical_block_count()
    }

    fn logical_key(&self, index: usize) -> Result<Option<&FusionTreePairKey>, OperationError> {
        self.logical_key(index).map(Some)
    }

    fn storage_index(&self, logical_index: usize) -> Result<usize, OperationError> {
        self.storage_index(logical_index)
    }
}

pub fn tensorcontract_fusion_block_specs<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Vec<TensorContractBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
    reject_fusion_contract_conjugation(axes)?;
    tensorcontract_fusion_block_specs_lowered(rule, dst, lhs, rhs, axes)
}

fn tensorcontract_fusion_block_specs_lowered<R, S>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axes: TensorContractSpec<'_>,
) -> Result<Vec<TensorContractBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    S: ContractBlockSource,
{
    let dst_nout = dst.nout();
    let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    if !FusionTreeHomSpace::tensorcontract_homspace_matches(
        rule,
        lhs.homspace(),
        rhs.homspace(),
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axis_plan.output_axes.as_slice(),
        dst_nout,
        dst.homspace(),
    )
    .map_err(OperationError::from_core_preserving_context)?
    {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    if !is_core_form_fusion_source_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
    ) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: SOURCE_TRANSFORM_REQUIRES_EXPLICIT,
        });
    }
    if is_core_form_fusion_compose_contract(
        lhs.homspace(),
        rhs.homspace(),
        axis_plan.lhs_contracting_axes.as_slice(),
        axis_plan.rhs_contracting_axes.as_slice(),
        axis_plan.output_axes.as_slice(),
        dst_nout,
    ) {
        return tensorcontract_core_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan);
    }

    tensorcontract_transformed_fusion_block_specs(rule, dst, lhs, rhs, &axis_plan, dst_nout)
}

fn tensorcontract_core_fusion_block_specs<R, S>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axis_plan: &TensorContractAxisPlan,
) -> Result<Vec<TensorContractBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    S: ContractBlockSource,
{
    let mut specs = Vec::new();
    for lhs_index in 0..lhs.block_count() {
        let Some(lhs_key) = lhs.logical_key(lhs_index)? else {
            continue;
        };
        let lhs_external = lhs_key.external_sectors(rule);
        for rhs_index in 0..rhs.block_count() {
            let Some(rhs_key) = rhs.logical_key(rhs_index)? else {
                continue;
            };
            let rhs_external = rhs_key.external_sectors(rule);
            if !contracted_external_sectors_match(
                rule,
                &lhs_external,
                &rhs_external,
                axis_plan.lhs_contracting_axes.as_slice(),
                axis_plan.rhs_contracting_axes.as_slice(),
            ) {
                continue;
            }
            if !contracted_fusion_tree_basis_matches(lhs_key.domain_tree(), rhs_key.codomain_tree())
            {
                continue;
            }
            let dst_key = FusionTreePairKey::pair(
                lhs_key.codomain_tree().clone(),
                rhs_key.domain_tree().clone(),
            );
            let dst_external = dst_key.external_sectors(rule);
            let expected_external =
                contracted_output_external_sectors(&lhs_external, &rhs_external, axis_plan);
            if dst_external != expected_external {
                return Err(OperationError::StructureMismatch { tensor: "dst" });
            }
            // The sorted, exact-key block lookup below already reports a
            // missing dst key. A prior `fusion_tree_keys_from_external_sectors`
            // + `contains` membership test is redundant — dst's structure
            // blocks are exactly the homspace's valid fusion trees, so
            // `find_block_index_by_fusion_tree_pair` is `Some` iff the key is a
            // valid output tree — and it costs an uncached fusion-tree
            // enumeration + allocation per block pair (issue #52).
            let dst_index = dst
                .structure()
                .find_block_index_by_fusion_tree_pair(&dst_key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(BlockKey::from(dst_key.clone())),
                })?;
            let coefficient = rhs_contract_twist_factor(
                rule,
                rhs.homspace(),
                axis_plan.rhs_contracting_axes.as_slice(),
                rhs_key.codomain_tree(),
            )?;
            specs.push(TensorContractBlockSpec::with_coefficient(
                dst_index,
                lhs.storage_index(lhs_index)?,
                rhs.storage_index(rhs_index)?,
                coefficient,
            ));
        }
    }
    Ok(specs)
}

fn tensorcontract_transformed_fusion_block_specs<R, S>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axis_plan: &TensorContractAxisPlan,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorContractBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    S: ContractBlockSource,
{
    let output_codomain_axes = &axis_plan.output_axes[..dst_codomain_rank];
    let output_domain_axes = &axis_plan.output_axes[dst_codomain_rank..];
    let mut rhs_prepared = Vec::new();
    for rhs_index in 0..rhs.block_count() {
        let Some(rhs_key) = rhs.logical_key(rhs_index)? else {
            continue;
        };
        let rhs_terms = multiplicity_free_permute_tree_pair(
            rule,
            rhs_key,
            axis_plan.rhs_contracting_axes.as_slice(),
            axis_plan.rhs_open_axes.as_slice(),
        )
        .map_err(OperationError::from_core_preserving_context)?;
        rhs_prepared.push((
            rhs.storage_index(rhs_index)?,
            rhs_key.external_sectors(rule),
            rhs_terms,
        ));
    }

    let mut specs = Vec::new();
    for lhs_index in 0..lhs.block_count() {
        let Some(lhs_key) = lhs.logical_key(lhs_index)? else {
            continue;
        };
        let lhs_storage_index = lhs.storage_index(lhs_index)?;
        let lhs_terms = multiplicity_free_permute_tree_pair(
            rule,
            lhs_key,
            axis_plan.lhs_open_axes.as_slice(),
            axis_plan.lhs_contracting_axes.as_slice(),
        )
        .map_err(OperationError::from_core_preserving_context)?;
        let lhs_external = lhs_key.external_sectors(rule);
        for (rhs_index, rhs_external, rhs_terms) in &rhs_prepared {
            if !contracted_external_sectors_match(
                rule,
                &lhs_external,
                rhs_external,
                axis_plan.lhs_contracting_axes.as_slice(),
                axis_plan.rhs_contracting_axes.as_slice(),
            ) {
                continue;
            }

            for (lhs_core, lhs_coeff) in &lhs_terms {
                for (rhs_core, rhs_coeff) in rhs_terms {
                    if !contracted_fusion_tree_basis_matches(
                        lhs_core.domain_tree(),
                        rhs_core.codomain_tree(),
                    ) {
                        continue;
                    }
                    let core_dst_key = FusionTreePairKey::pair(
                        lhs_core.codomain_tree().clone(),
                        rhs_core.domain_tree().clone(),
                    );
                    let rhs_twist = rhs_contract_twist_factor(
                        rule,
                        rhs.homspace(),
                        axis_plan.rhs_contracting_axes.as_slice(),
                        rhs_core.codomain_tree(),
                    )?;
                    let dst_terms = multiplicity_free_permute_tree_pair(
                        rule,
                        &core_dst_key,
                        output_codomain_axes,
                        output_domain_axes,
                    )
                    .map_err(OperationError::from_core_preserving_context)?;
                    for (dst_key, dst_coeff) in dst_terms {
                        let dst_index = dst
                            .structure()
                            .find_block_index_by_fusion_tree_pair(&dst_key)
                            .ok_or_else(|| OperationError::MissingBlockKey {
                                key: Box::new(BlockKey::from(dst_key.clone())),
                            })?;
                        specs.push(TensorContractBlockSpec::with_coefficient(
                            dst_index,
                            lhs_storage_index,
                            *rhs_index,
                            *lhs_coeff * *rhs_coeff * dst_coeff * rhs_twist,
                        ));
                    }
                }
            }
        }
    }
    Ok(specs)
}

pub(crate) const FUSION_TENSORCONTRACT_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT: &str =
    "fusion tensorcontract with conjugation requires categorical adjoint lowering";
pub(crate) const SOURCE_TRANSFORM_REQUIRES_EXPLICIT: &str =
    "fusion contraction requiring source tree-pair transforms is not implemented; pre-transform operands explicitly";

pub(crate) fn reject_fusion_contract_conjugation(
    axes: TensorContractSpec<'_>,
) -> Result<(), OperationError> {
    if axes.lhs_conjugate() || axes.rhs_conjugate() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: FUSION_TENSORCONTRACT_CONJUGATION_REQUIRES_CATEGORICAL_ADJOINT,
        });
    }
    Ok(())
}

pub(crate) fn rhs_contract_twist_factor<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    rhs_contracting_axes: &[usize],
    rhs_core_codomain: &FusionTreeKey,
) -> Result<R::Scalar, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    rhs_contract_twist_factor_oriented(
        rule,
        OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
        rhs_contracting_axes,
        rhs_core_codomain,
    )
}

pub(crate) fn rhs_contract_twist_factor_oriented<R>(
    rule: &R,
    rhs: OrientedFusionTreeHomSpace<'_>,
    rhs_contracting_axes: &[usize],
    rhs_core_codomain: &FusionTreeKey,
) -> Result<R::Scalar, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    if rule.braiding_style() != BraidingStyleKind::Fermionic {
        return Ok(R::Scalar::one());
    }
    if rhs_contracting_axes.len() != rhs_core_codomain.uncoupled().len() {
        return Err(OperationError::StructureRankMismatch {
            expected: rhs_contracting_axes.len(),
            actual: rhs_core_codomain.uncoupled().len(),
        });
    }
    let mut factor = R::Scalar::one();
    for (position, &axis) in rhs_contracting_axes.iter().enumerate() {
        if rhs
            .external_axis_is_dual(axis)
            .ok_or_else(|| OperationError::InvalidAxisSet {
                tensor: "rhs",
                axes: vec![axis],
                rank: rhs.rank(),
            })?
        {
            factor = factor * rule.twist_scalar(rhs_core_codomain.uncoupled()[position]);
        }
    }
    Ok(factor)
}

pub(crate) fn external_axis_is_dual(
    homspace: &FusionTreeHomSpace,
    axis: usize,
) -> Result<bool, OperationError> {
    homspace
        .external_axis_is_dual(axis)
        .ok_or_else(|| OperationError::InvalidAxisSet {
            tensor: "rhs",
            axes: vec![axis],
            rank: homspace.rank(),
        })
}

fn contracted_external_sectors_match<R>(
    rule: &R,
    lhs_external: &[SectorId],
    rhs_external: &[SectorId],
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> bool
where
    R: FusionRule,
{
    lhs_axes
        .iter()
        .zip(rhs_axes)
        .all(|(&lhs_axis, &rhs_axis)| rule.dual(lhs_external[lhs_axis]) == rhs_external[rhs_axis])
}

pub(crate) fn contracted_fusion_tree_basis_matches(
    lhs_domain: &FusionTreeKey,
    rhs_codomain: &FusionTreeKey,
) -> bool {
    lhs_domain == rhs_codomain
}

fn contracted_output_external_sectors(
    lhs_external: &[SectorId],
    rhs_external: &[SectorId],
    axis_plan: &TensorContractAxisPlan,
) -> Vec<SectorId> {
    let mut core = axis_plan
        .lhs_open_axes
        .iter()
        .map(|&axis| lhs_external[axis])
        .collect::<Vec<_>>();
    core.extend(
        axis_plan
            .rhs_open_axes
            .iter()
            .map(|&axis| rhs_external[axis]),
    );
    axis_plan
        .output_axes
        .iter()
        .map(|&axis| core[axis])
        .collect()
}

fn is_core_form_fusion_compose_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_codomain_rank: usize,
) -> bool {
    let core_output_rank = lhs.codomain().len() + rhs.domain().len();
    is_core_form_fusion_source_contract(lhs, rhs, lhs_contracting_axes, rhs_contracting_axes)
        && output_axes.iter().copied().eq(0..core_output_rank)
        && dst_codomain_rank == lhs.codomain().len()
}

fn is_core_form_fusion_source_contract(
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
) -> bool {
    lhs_contracting_axes
        .iter()
        .copied()
        .eq(lhs.codomain().len()..lhs.rank())
        && rhs_contracting_axes
            .iter()
            .copied()
            .eq(0..rhs.codomain().len())
}
