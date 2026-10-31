use std::borrow::Cow;

use symtn_core::{FusionSpaceAdmission, FusionTreeHomSpace, MultiplicityFreeRigidSymbols};

use smallvec::SmallVec;

use crate::{DenseBlockScalar, OperationError, TreeTransformOperation, TreeTransformOperationKind};
use symtn_operations::{AxisVec, OutputAxisOrder, TensorContractSpec, TensorContractSpecOwned};

use super::super::dynamic_space::{
    encoded_layout_primer, BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    FusionOperandLayout, LayoutKeyBuilder, MetadataOutput, MetadataRequest, TransformedLayoutProbe,
};
use super::super::structure::TensorContractAxisPlan;

/// A paired ordering of the contracted axes.
///
/// The two vectors are a single permutation: entries at the same position
/// remain paired. Keeping this as one value prevents a cost model from
/// accidentally sorting one operand independently and changing the
/// contraction semantics. This is deliberately only a preparation primitive;
/// it does not select a runtime winner yet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ContractAxisOrderCandidate {
    lhs: AxisVec,
    rhs: AxisVec,
}

impl ContractAxisOrderCandidate {
    #[inline]
    pub(crate) fn lhs(&self) -> &[usize] {
        &self.lhs
    }

    #[inline]
    pub(crate) fn rhs(&self) -> &[usize] {
        &self.rhs
    }
}

/// Operand orientation for a fusion-level contraction candidate.
///
/// Keeping orientation separate prevents reversed candidates from overloading
/// paired-axis ordering with a second meaning.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusionContractOrientation {
    LhsRhs,
    RhsLhs,
}

const FORWARD_ORIENTATIONS: [FusionContractOrientation; 1] = [FusionContractOrientation::LhsRhs];
/// TensorKit `contract!` tie order: A·B (m1, m2) before B·A (m3, m4).
pub(crate) const CACHED_ORIENTATIONS: [FusionContractOrientation; 2] = [
    FusionContractOrientation::LhsRhs,
    FusionContractOrientation::RhsLhs,
];

/// Provider-independent structural facts used by the current fusion selector.
///
/// The value owns no provider, transformed HomSpace, backend, or cache handle.
/// It is therefore safe to carry into later route lowering without retaining
/// the semantic machinery used to prove these facts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FusionContractCandidateFacts {
    axis_order: ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
    lhs_exact_identity_borrowable: bool,
    rhs_exact_identity_borrowable: bool,
    lhs_requires_twist: bool,
    rhs_requires_twist: bool,
    output_exact_identity: bool,
    lhs_materialized_elements: usize,
    rhs_materialized_elements: usize,
    output_materialized_elements: usize,
    total_materialized_elements: usize,
}

impl FusionContractCandidateFacts {
    #[cfg(test)]
    #[inline]
    pub(crate) fn axis_order(&self) -> &ContractAxisOrderCandidate {
        &self.axis_order
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn orientation(&self) -> FusionContractOrientation {
        self.orientation
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_exact_identity_borrowable(&self) -> bool {
        self.lhs_exact_identity_borrowable
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_exact_identity_borrowable(&self) -> bool {
        self.rhs_exact_identity_borrowable
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_requires_twist(&self) -> bool {
        self.lhs_requires_twist
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_requires_twist(&self) -> bool {
        self.rhs_requires_twist
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn output_exact_identity(&self) -> bool {
        self.output_exact_identity
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn lhs_materialized_elements(&self) -> usize {
        self.lhs_materialized_elements
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rhs_materialized_elements(&self) -> usize {
        self.rhs_materialized_elements
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn output_materialized_elements(&self) -> usize {
        self.output_materialized_elements
    }

    #[inline]
    pub(crate) fn total_materialized_elements(&self) -> usize {
        self.total_materialized_elements
    }
}

#[derive(Debug)]
struct ScoredFusionContractCandidate {
    plan: FusionContractPlan,
    facts: FusionContractCandidateFacts,
}

/// Build the side-sorted candidates for a contraction.
///
/// Sorting is stable and always applies the same permutation to both sides.
/// No fermionic sign is computed here: that belongs to the tree-transform
/// execution of the selected candidate.
pub(crate) fn contracted_axis_order_candidates(
    lhs: &[usize],
    rhs: &[usize],
) -> SmallVec<[ContractAxisOrderCandidate; 2]> {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "paired contraction axes must have equal length"
    );
    let sorted_by = |keys: &[usize]| {
        let mut order = (0..keys.len()).collect::<AxisVec>();
        order.sort_by_key(|&i| keys[i]);
        ContractAxisOrderCandidate {
            lhs: order.iter().map(|&i| lhs[i]).collect(),
            rhs: order.iter().map(|&i| rhs[i]).collect(),
        }
    };
    let lhs_sorted = sorted_by(lhs);
    let rhs_sorted = sorted_by(rhs);
    let mut candidates = SmallVec::new();
    candidates.push(lhs_sorted);
    if !candidates.contains(&rhs_sorted) {
        candidates.push(rhs_sorted);
    }
    candidates
}

fn select_best_scored_contract_candidate<P, F, E>(
    axes: TensorContractSpec<'_>,
    orientations: &[FusionContractOrientation],
    mut score: F,
) -> Result<(P, FusionContractCandidateFacts), E>
where
    F: FnMut(
        &ContractAxisOrderCandidate,
        FusionContractOrientation,
    ) -> Result<(P, FusionContractCandidateFacts), E>,
{
    let candidates =
        contracted_axis_order_candidates(axes.lhs_contracting_axes(), axes.rhs_contracting_axes());
    let mut best: Option<(P, FusionContractCandidateFacts)> = None;
    let mut first_error = None;
    for &orientation in orientations {
        for candidate in &candidates {
            let scored = match score(candidate, orientation) {
                Ok(scored) => scored,
                Err(error) => {
                    // Candidate-local layout capability does not invalidate a
                    // later stable candidate. Provider errors remain terminal
                    // in callers that cannot classify them as local.
                    first_error.get_or_insert(error);
                    continue;
                }
            };
            if best.as_ref().is_none_or(|best| {
                scored.1.total_materialized_elements() < best.1.total_materialized_elements()
            }) {
                best = Some(scored);
            }
        }
    }
    best.ok_or_else(|| {
        first_error.expect("paired contraction always has at least one stable candidate")
    })
}

/// One candidate [`FusionContractPlan`] with its permutations still as
/// stack axis lists.
///
/// Why not score materialized plans: every `TreeTransformOperation` owns a
/// shared heap slice, and only the selected candidate's operations outlive
/// the selection. This is the single authority for the candidate lowering;
/// [`CandidatePlan::materialize`] builds the plan value from it.
#[derive(Clone, Debug)]
struct CandidatePlan {
    orientation: FusionContractOrientation,
    lhs: [AxisVec; 2],
    rhs: [AxisVec; 2],
    output: [AxisVec; 2],
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    lhs_open_rank: usize,
    lhs_contract_rank: usize,
    rhs_contract_rank: usize,
    rhs_open_rank: usize,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
}

impl CandidatePlan {
    #[allow(clippy::too_many_arguments)]
    fn from_ranks(
        dst_nout: usize,
        dst_rank: usize,
        lhs_rank: usize,
        rhs_rank: usize,
        axes: TensorContractSpec<'_>,
        lhs_source_conjugate: bool,
        rhs_source_conjugate: bool,
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile(lhs_rank, rhs_rank, dst_rank, axes)?;
        let lhs_open_rank = axis_plan.lhs_open_axes.len();
        let lhs_contract_rank = axis_plan.lhs_contracting_axes.len();
        let rhs_contract_rank = axis_plan.rhs_contracting_axes.len();
        let rhs_open_rank = axis_plan.rhs_open_axes.len();
        Ok(Self {
            orientation: FusionContractOrientation::LhsRhs,
            lhs: [axis_plan.lhs_open_axes, axis_plan.lhs_contracting_axes],
            rhs: [axis_plan.rhs_contracting_axes, axis_plan.rhs_open_axes],
            output: [
                AxisVec::from_slice(&axis_plan.output_axes[..dst_nout]),
                AxisVec::from_slice(&axis_plan.output_axes[dst_nout..]),
            ],
            core_dst_open_lhs_rank: lhs_open_rank,
            core_dst_open_rhs_rank: rhs_open_rank,
            lhs_open_rank,
            lhs_contract_rank,
            rhs_contract_rank,
            rhs_open_rank,
            lhs_source_conjugate,
            rhs_source_conjugate,
        })
    }

    #[cfg(test)]
    fn from_plan(plan: &FusionContractPlan) -> Self {
        let permutation = |operation: &TreeTransformOperation| {
            if operation.kind() != TreeTransformOperationKind::Permute {
                unreachable!("fusion contraction lowering uses permutations");
            }
            [
                AxisVec::from_slice(operation.codomain_permutation()),
                AxisVec::from_slice(operation.domain_permutation()),
            ]
        };
        Self {
            orientation: plan.orientation,
            lhs: permutation(&plan.lhs_transform),
            rhs: permutation(&plan.rhs_transform),
            output: permutation(&plan.output_transform),
            core_dst_open_lhs_rank: plan.core_dst_open_lhs_rank,
            core_dst_open_rhs_rank: plan.core_dst_open_rhs_rank,
            lhs_open_rank: plan.lhs_open_rank,
            lhs_contract_rank: plan.lhs_contract_rank,
            rhs_contract_rank: plan.rhs_contract_rank,
            rhs_open_rank: plan.rhs_open_rank,
            lhs_source_conjugate: plan.lhs_source_conjugate,
            rhs_source_conjugate: plan.rhs_source_conjugate,
        }
    }

    fn orient(mut self, orientation: FusionContractOrientation) -> Self {
        if orientation == FusionContractOrientation::RhsLhs {
            let lhs_open_rank = self.lhs_open_rank;
            let rhs_open_rank = self.rhs_open_rank;
            let semantic_to_core = |axis: usize| {
                if axis < lhs_open_rank {
                    axis + rhs_open_rank
                } else {
                    axis - lhs_open_rank
                }
            };
            let [output_codomain, output_domain] = &self.output;
            self.output = [
                output_codomain
                    .iter()
                    .copied()
                    .map(semantic_to_core)
                    .collect(),
                output_domain
                    .iter()
                    .copied()
                    .map(semantic_to_core)
                    .collect(),
            ];
            let [lhs_open_axes, lhs_contracting_axes] = std::mem::take(&mut self.lhs);
            let [rhs_contracting_axes, rhs_open_axes] = std::mem::take(&mut self.rhs);
            self.orientation = orientation;
            self.lhs = [lhs_contracting_axes, lhs_open_axes];
            self.rhs = [rhs_open_axes, rhs_contracting_axes];
            self.core_dst_open_lhs_rank = rhs_open_rank;
            self.core_dst_open_rhs_rank = lhs_open_rank;
        }
        self
    }

    fn output_transform_is_identity(&self) -> bool {
        output_permutation_is_identity(
            &self.output[0],
            &self.output[1],
            self.core_dst_open_lhs_rank,
            self.core_dst_open_rhs_rank,
        )
    }

    fn materialize(&self) -> FusionContractPlan {
        let permute = |[codomain, domain]: &[AxisVec; 2]| {
            TreeTransformOperation::permute(codomain.iter().copied(), domain.iter().copied())
        };
        let open_lhs = self.core_dst_open_lhs_rank;
        FusionContractPlan {
            orientation: self.orientation,
            lhs_transform: permute(&self.lhs),
            rhs_transform: permute(&self.rhs),
            output_transform: permute(&self.output),
            core_axes: TensorContractSpecOwned::from_axis_vecs(
                (open_lhs..open_lhs + self.lhs_contract_rank).collect(),
                (0..self.rhs_contract_rank).collect(),
                (0..open_lhs + self.core_dst_open_rhs_rank).collect(),
                false,
                false,
            ),
            core_dst_open_lhs_rank: self.core_dst_open_lhs_rank,
            core_dst_open_rhs_rank: self.core_dst_open_rhs_rank,
            lhs_open_rank: self.lhs_open_rank,
            lhs_contract_rank: self.lhs_contract_rank,
            rhs_contract_rank: self.rhs_contract_rank,
            rhs_open_rank: self.rhs_open_rank,
            lhs_source_conjugate: self.lhs_source_conjugate,
            rhs_source_conjugate: self.rhs_source_conjugate,
            supertrace_twist: true,
        }
    }
}

/// True when `(codomain, domain)` is the exact current axis order and split.
fn output_permutation_is_identity(
    codomain: &[usize],
    domain: &[usize],
    codomain_rank: usize,
    domain_rank: usize,
) -> bool {
    codomain.iter().copied().eq(0..codomain_rank)
        && domain
            .iter()
            .copied()
            .eq(codomain_rank..codomain_rank + domain_rank)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionContractPlan {
    orientation: FusionContractOrientation,
    lhs_transform: TreeTransformOperation,
    rhs_transform: TreeTransformOperation,
    output_transform: TreeTransformOperation,
    core_axes: TensorContractSpecOwned,
    core_dst_open_lhs_rank: usize,
    core_dst_open_rhs_rank: usize,
    lhs_open_rank: usize,
    lhs_contract_rank: usize,
    rhs_contract_rank: usize,
    rhs_open_rank: usize,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    /// Whether dual rhs-contracted legs take the fermionic supertrace twist:
    /// a contraction's (`blas_contract!`), never a planar contraction's
    /// (`planarcontract!` is transposes and `mul!`).
    supertrace_twist: bool,
}

impl FusionContractPlan {
    /// TensorKit `planarcontract!`'s steps for admitted `orders`: each
    /// operand transposed onto its core layout, the core GEMMs, and the
    /// output transpose, with no candidate walk and no twist. An identity
    /// transpose is borrowed like an identity permute.
    pub(crate) fn planar(
        orders: &super::super::PlanarContractOrders,
        lhs_source_conjugate: bool,
        rhs_source_conjugate: bool,
    ) -> Self {
        let transpose = |[codomain, domain]: &[AxisVec; 2]| {
            TreeTransformOperation::transpose(codomain.iter().copied(), domain.iter().copied())
        };
        let lhs_open_rank = orders.lhs[0].len();
        let contract_rank = orders.lhs[1].len();
        let rhs_open_rank = orders.rhs[1].len();
        Self {
            orientation: FusionContractOrientation::LhsRhs,
            lhs_transform: transpose(&orders.lhs),
            rhs_transform: transpose(&orders.rhs),
            output_transform: transpose(&orders.output),
            core_axes: TensorContractSpecOwned::from_axis_vecs(
                (lhs_open_rank..lhs_open_rank + contract_rank).collect(),
                (0..contract_rank).collect(),
                (0..lhs_open_rank + rhs_open_rank).collect(),
                false,
                false,
            ),
            core_dst_open_lhs_rank: lhs_open_rank,
            core_dst_open_rhs_rank: rhs_open_rank,
            lhs_open_rank,
            lhs_contract_rank: contract_rank,
            rhs_contract_rank: contract_rank,
            rhs_open_rank,
            lhs_source_conjugate,
            rhs_source_conjugate,
            supertrace_twist: false,
        }
    }

    #[inline]
    pub(crate) fn supertrace_twist(&self) -> bool {
        self.supertrace_twist
    }

    #[inline]
    pub(crate) fn orientation(&self) -> FusionContractOrientation {
        self.orientation
    }

    #[inline]
    pub fn lhs_transform(&self) -> &TreeTransformOperation {
        &self.lhs_transform
    }

    #[inline]
    pub fn rhs_transform(&self) -> &TreeTransformOperation {
        &self.rhs_transform
    }

    #[inline]
    pub fn output_transform(&self) -> &TreeTransformOperation {
        &self.output_transform
    }

    #[inline]
    pub fn core_axes(&self) -> &TensorContractSpecOwned {
        &self.core_axes
    }

    #[inline]
    pub fn core_dst_open_lhs_rank(&self) -> usize {
        self.core_dst_open_lhs_rank
    }

    #[inline]
    pub fn core_dst_open_rhs_rank(&self) -> usize {
        self.core_dst_open_rhs_rank
    }

    pub(crate) fn output_transform_is_identity(&self) -> bool {
        self.output_transform.kind() != TreeTransformOperationKind::Braid
            && output_permutation_is_identity(
                self.output_transform.codomain_permutation(),
                self.output_transform.domain_permutation(),
                self.core_dst_open_lhs_rank,
                self.core_dst_open_rhs_rank,
            )
    }

    #[inline]
    pub fn lhs_open_rank(&self) -> usize {
        self.lhs_open_rank
    }

    #[inline]
    pub fn lhs_contract_rank(&self) -> usize {
        self.lhs_contract_rank
    }

    #[inline]
    pub fn rhs_contract_rank(&self) -> usize {
        self.rhs_contract_rank
    }

    #[inline]
    pub fn rhs_open_rank(&self) -> usize {
        self.rhs_open_rank
    }

    #[inline]
    pub fn lhs_source_conjugate(&self) -> bool {
        self.lhs_source_conjugate
    }

    #[inline]
    pub fn rhs_source_conjugate(&self) -> bool {
        self.rhs_source_conjugate
    }
}

/// The fusion contraction plan of `dst = contract(lhs, rhs)`, the planning
/// step of [`TensorContractFusionExecutionContext::tensorcontract_planned_into`]
/// with the same operand convention: a [`FusionOperand::adjoint`] operand is
/// the lazy adjoint of its storage space, and its axes index the adjoint's
/// legs, as in TensorKit's planned `contract!`, which receives `A'` after
/// `tensorcontract!` has lowered `conjA`.
///
/// [`TensorContractFusionExecutionContext::tensorcontract_planned_into`]:
///     crate::TensorContractFusionExecutionContext::tensorcontract_planned_into
pub fn prepare_tensorcontract_fusion_plan_dyn<R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    // Why not accept a separate rule: the bound destination is the semantic
    // authority, as in the eager entry. The raw core still checks every
    // operand space against it without re-enumerating any fusion-tree grid.
    prepare_tensorcontract_fusion_plan_dyn_raw(
        dst.provider(),
        dst.space(),
        lhs.storage_space(),
        rhs.storage_space(),
        TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::from_axes(output_axes),
            lhs.storage_conjugate(),
            rhs.storage_conjugate(),
        ),
    )
}

/// `space`, or its lazy adjoint view when `adjoint` is set: the logical
/// space whose legs a conjugated operand's axes index.
fn logical_space(
    space: &DynamicFusionMapSpace,
    adjoint: bool,
) -> Result<Cow<'_, DynamicFusionMapSpace>, OperationError> {
    if adjoint {
        space.adjoint_view().map(Cow::Owned)
    } else {
        Ok(Cow::Borrowed(space))
    }
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    prepare_tensorcontract_fusion_plan_dyn_raw_with_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        &FORWARD_ORIENTATIONS,
    )
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw_canonical<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    prepare_tensorcontract_fusion_plan_dyn_raw_with_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        &CACHED_ORIENTATIONS,
    )
}

fn prepare_tensorcontract_fusion_plan_dyn_raw_with_orientations<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    orientations: &[FusionContractOrientation],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    let lhs = logical_space(lhs, axes.lhs_conjugate())?;
    let lhs = &*lhs;
    let rhs = logical_space(rhs, axes.rhs_conjugate())?;
    let rhs = &*rhs;
    select_tensorcontract_fusion_plan_from_spaces_with_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        axes.lhs_conjugate(),
        axes.rhs_conjugate(),
        orientations,
    )
}

pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    primer: LayoutKeyBuilder<R>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    prepare_tensorcontract_fusion_plan_from_operands(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        |rule, source: &FusionOperandLayout<'_>, operation, primer| {
            source.transformed_layout_probe(
                rule,
                operation,
                primer.ok_or(OperationError::InvalidArgument {
                    message: "prelowered plan requires a layout primer",
                })?,
            )
        },
        primer,
    )
}

/// Prepare a plan with an explicitly paired contraction-axis ordering.
///
/// This is intentionally crate-private: it is an oracle seam for validating
/// candidate permutations before a layout cost model is enabled. The caller
/// must provide the same axis sets as `axes`; only their order may differ.
#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    candidate: &ContractAxisOrderCandidate,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    fn same_axes(a: &[usize], b: &[usize]) -> bool {
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        a == b
    }
    if !same_axes(axes.lhs_contracting_axes(), candidate.lhs())
        || !same_axes(axes.rhs_contracting_axes(), candidate.rhs())
    {
        return Err(OperationError::InvalidArgument {
            message: "candidate must preserve contracted axis sets",
        });
    }
    let candidate_axes = TensorContractSpec::new_with_conjugation(
        candidate.lhs(),
        candidate.rhs(),
        axes.output_permutation(),
        axes.lhs_conjugate(),
        axes.rhs_conjugate(),
    );
    prepare_tensorcontract_fusion_plan_dyn_raw_fixed(rule, dst, lhs, rhs, candidate_axes)
}

#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    candidate: &ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    let plan = prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order(
        rule, dst, lhs, rhs, axes, candidate,
    )?;
    Ok(orient_fusion_contract_plan(plan, orientation))
}

#[cfg(test)]
pub(crate) fn orient_fusion_contract_plan(
    plan: FusionContractPlan,
    orientation: FusionContractOrientation,
) -> FusionContractPlan {
    if orientation == FusionContractOrientation::RhsLhs {
        return CandidatePlan::from_plan(&plan)
            .orient(orientation)
            .materialize();
    }
    plan
}

#[cfg(test)]
fn prepare_tensorcontract_fusion_plan_dyn_raw_fixed<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    let lhs = logical_space(lhs, axes.lhs_conjugate())?;
    let lhs = &*lhs;
    let rhs = logical_space(rhs, axes.rhs_conjugate())?;
    let rhs = &*rhs;
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, None)?;
    compile_tensorcontract_fusion_plan_from_spaces(
        dst,
        lhs,
        rhs,
        axes,
        axes.lhs_conjugate(),
        axes.rhs_conjugate(),
    )
}

#[cfg(test)]
pub(crate) fn prepare_tensorcontract_fusion_candidate_facts_dyn_raw<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<Vec<FusionContractCandidateFacts>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    let lhs = logical_space(lhs, axes.lhs_conjugate())?;
    let lhs = &*lhs;
    let rhs = logical_space(rhs, axes.rhs_conjugate())?;
    let rhs = &*rhs;
    fusion_contract_candidate_facts_from_spaces_with_probe(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        axes.lhs_conjugate(),
        axes.rhs_conjugate(),
        encoded_layout_probe::<R>,
        None,
        &CACHED_ORIENTATIONS,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_tensorcontract_fusion_plan_from_spaces_with_orientations<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    orientations: &[FusionContractOrientation],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
        encoded_layout_probe::<R>,
        None,
        orientations,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fusion_contract_candidate_facts_from_spaces_with_probe<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    probe: LayoutProbeBuilder<R>,
    primer: Option<LayoutKeyBuilder<R>>,
    orientations: &[FusionContractOrientation],
) -> Result<Vec<FusionContractCandidateFacts>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, primer)?;
    orientations
        .iter()
        .flat_map(|&orientation| {
            contracted_axis_order_candidates(
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
            )
            .into_iter()
            .map(move |candidate| (orientation, candidate))
        })
        .map(|(orientation, candidate)| {
            let candidate_axes = TensorContractSpec::new(
                candidate.lhs(),
                candidate.rhs(),
                axes.output_permutation(),
            );
            let plan = orient_fusion_contract_plan(
                compile_tensorcontract_fusion_plan_from_spaces(
                    dst,
                    lhs,
                    rhs,
                    candidate_axes,
                    lhs_source_conjugate,
                    rhs_source_conjugate,
                )?,
                orientation,
            );
            score_fusion_contract_candidate(rule, dst, lhs, rhs, candidate, plan, probe, primer)
                .map(|scored| scored.facts)
        })
        .collect()
}

#[cfg(test)]
type LayoutProbeBuilder<R> = for<'a> fn(
    &'a R,
    &'a DynamicFusionMapSpace,
    &'a TreeTransformOperation,
    Option<LayoutKeyBuilder<R>>,
) -> Result<TransformedLayoutProbe, OperationError>;

trait ContractPlanSource {
    fn homspace(&self) -> &FusionTreeHomSpace;
    fn nout(&self) -> usize;
    fn rank(&self) -> usize;
    fn admission(&self) -> &FusionSpaceAdmission;
    fn storage_space(&self) -> &DynamicFusionMapSpace;
}

impl ContractPlanSource for DynamicFusionMapSpace {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace()
    }

    fn nout(&self) -> usize {
        self.nout()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn admission(&self) -> &FusionSpaceAdmission {
        self.admission()
    }

    fn storage_space(&self) -> &DynamicFusionMapSpace {
        self
    }
}

impl ContractPlanSource for FusionOperand<'_> {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.logical_homspace()
    }

    fn nout(&self) -> usize {
        self.oriented_homspace().nout()
    }

    fn rank(&self) -> usize {
        self.storage_space().rank()
    }

    fn admission(&self) -> &FusionSpaceAdmission {
        self.storage_space().admission()
    }

    fn storage_space(&self) -> &DynamicFusionMapSpace {
        FusionOperand::storage_space(*self)
    }
}

impl ContractPlanSource for FusionOperandLayout<'_> {
    fn homspace(&self) -> &FusionTreeHomSpace {
        self.homspace()
    }

    fn nout(&self) -> usize {
        self.nout()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn admission(&self) -> &FusionSpaceAdmission {
        self.admission()
    }

    fn storage_space(&self) -> &DynamicFusionMapSpace {
        self.storage_space()
    }
}

fn encoded_layout_probe<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    _primer: Option<LayoutKeyBuilder<R>>,
) -> Result<TransformedLayoutProbe, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    space.transformed_layout_probe(rule, operation)
}

#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "the test seam keeps injected layout probes and each planning input independently controllable"
)]
fn select_tensorcontract_fusion_plan_from_spaces_with_probe<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    probe: LayoutProbeBuilder<R>,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations(
        rule,
        dst,
        lhs,
        rhs,
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
        probe,
        primer,
        &FORWARD_ORIENTATIONS,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations<R, S, P>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    probe: P,
    primer: Option<LayoutKeyBuilder<R>>,
    orientations: &[FusionContractOrientation],
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    S: ContractPlanSource,
    P: for<'a> Fn(
            &'a R,
            &'a S,
            &'a TreeTransformOperation,
            Option<LayoutKeyBuilder<R>>,
        ) -> Result<TransformedLayoutProbe, OperationError>
        + Copy,
{
    validate_tensorcontract_fusion_plan_inputs(rule, dst, lhs, rhs, axes, primer)?;
    let complete = matches!(dst.admission(), FusionSpaceAdmission::Complete(_))
        && matches!(lhs.admission(), FusionSpaceAdmission::Complete(_))
        && matches!(rhs.admission(), FusionSpaceAdmission::Complete(_));
    let canonical = |space: &DynamicFusionMapSpace| {
        super::super::dynamic::storage_layout_is_canonical(rule, space)
    };
    let layouts = CanonicalLayouts::new(&canonical);

    select_best_scored_contract_candidate(axes, orientations, |candidate, orientation| {
        let candidate_axes =
            TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
        let shape = CandidatePlan::from_ranks(
            dst.nout(),
            dst.rank(),
            lhs.rank(),
            rhs.rank(),
            candidate_axes,
            lhs_source_conjugate,
            rhs_source_conjugate,
        )?
        .orient(orientation);
        if complete {
            let facts = score_complete_fusion_contract_candidate(
                |homspace, axes| source_contract_homspace_requires_twist(rule, homspace, axes),
                &layouts,
                dst,
                lhs,
                rhs,
                candidate.clone(),
                &shape,
            )?;
            Ok((SelectedPlan::Candidate(shape), facts))
        } else {
            let scored = score_fusion_contract_candidate(
                rule,
                dst,
                lhs,
                rhs,
                candidate.clone(),
                shape.materialize(),
                probe,
                primer,
            )?;
            Ok((SelectedPlan::Materialized(scored.plan), scored.facts))
        }
    })
    .map(|(best, _)| match best {
        SelectedPlan::Candidate(shape) => shape.materialize(),
        SelectedPlan::Materialized(plan) => plan,
    })
}

/// The winner of a candidate selection: probing scorers already built the
/// plan to probe its transformed layouts; Complete scoring never did.
enum SelectedPlan {
    Candidate(CandidatePlan),
    Materialized(FusionContractPlan),
}

#[allow(clippy::too_many_arguments)]
fn prepare_tensorcontract_fusion_plan_from_operands<R, P>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
    axes: TensorContractSpec<'_>,
    probe: P,
    primer: LayoutKeyBuilder<R>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    P: for<'a> Fn(
            &'a R,
            &'a FusionOperandLayout<'_>,
            &'a TreeTransformOperation,
            Option<LayoutKeyBuilder<R>>,
        ) -> Result<TransformedLayoutProbe, OperationError>
        + Copy,
{
    dst.validate_rule(rule)?;
    if axes.lhs_conjugate() != lhs.storage_conjugate()
        || axes.rhs_conjugate() != rhs.storage_conjugate()
    {
        return Err(OperationError::InvalidArgument {
            message: "prelowered operand flags must match the contraction cache key",
        });
    }
    let logical_axes = TensorContractSpec::new(
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    );
    select_tensorcontract_fusion_plan_from_spaces_with_probe_and_orientations(
        rule,
        dst,
        lhs,
        rhs,
        logical_axes,
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
        probe,
        Some(primer),
        &CACHED_ORIENTATIONS,
    )
}

fn source_contract_homspace_requires_twist<R>(
    rule: &R,
    source: &FusionTreeHomSpace,
    contracting_axes: &[usize],
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    // Why not permute a loser HomSpace: external-axis duality is unchanged by
    // the candidate permutation, so core-right twist can be read at its source axes.
    super::super::resolution::rhs_contract_homspace_requires_twist(
        rule,
        source,
        TensorContractSpec::new(&[], contracting_axes, OutputAxisOrder::identity()),
    )
}

// Why not resolve every length eagerly: identity-borrowed storage never uses
// that count and must not gain an overflow/error from an unused route.
#[derive(Clone, Copy)]
enum CandidateRequiredLen<'a> {
    Known(usize),
    Space(&'a DynamicFusionMapSpace),
}

impl CandidateRequiredLen<'_> {
    fn get(&self) -> Result<usize, OperationError> {
        match self {
            Self::Known(len) => Ok(*len),
            Self::Space(space) => space
                .required_len()
                .map_err(OperationError::from_core_preserving_context),
        }
    }
}

/// Whether the fermionic contraction twist goes on the physical lhs (else the
/// physical rhs), given which operands the layout alone already copies: the
/// copied one when exactly one is, otherwise the smaller core operand, the
/// core-right one on a tie — TensorKit `blas_contract!`
/// (src/tensors/tensoroperations.jl:398-409 @cfaa073), with core-left as its
/// `A` and core-right as its `B`. Both sides carry the same `θ`: a core-left
/// block's domain tree equals the codomain tree of every core-right block it
/// contracts with, so twisting either operand of the pair scales the same
/// products. `lens` is `(lhs, rhs)` core lengths, read only when needed.
pub(crate) fn contract_twist_on_physical_lhs(
    reverse: bool,
    lhs_copied: bool,
    rhs_copied: bool,
    lens: impl FnOnce() -> Result<(usize, usize), OperationError>,
) -> Result<bool, OperationError> {
    let (core_left_copied, core_right_copied) = if reverse {
        (rhs_copied, lhs_copied)
    } else {
        (lhs_copied, rhs_copied)
    };
    let twist_core_left = if core_left_copied != core_right_copied {
        core_left_copied
    } else {
        let (lhs_len, rhs_len) = lens()?;
        let (core_left_len, core_right_len) = if reverse {
            (rhs_len, lhs_len)
        } else {
            (lhs_len, rhs_len)
        };
        core_left_len < core_right_len
    };
    Ok(twist_core_left != reverse)
}

struct FusionContractMaterializationInputs<'a> {
    lhs_exact_identity_borrowable: bool,
    rhs_exact_identity_borrowable: bool,
    core_right_requires_twist: bool,
    lhs_required_len: CandidateRequiredLen<'a>,
    rhs_required_len: CandidateRequiredLen<'a>,
    output_required_len: CandidateRequiredLen<'a>,
}

/// The orientation, conjugation, and output-identity facts one candidate's
/// cost depends on, read from either plan form.
#[derive(Clone, Copy)]
struct CandidateRoute {
    orientation: FusionContractOrientation,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
    output_exact_identity: bool,
}

impl CandidateRoute {
    fn of_plan(plan: &FusionContractPlan) -> Self {
        Self {
            orientation: plan.orientation(),
            lhs_source_conjugate: plan.lhs_source_conjugate(),
            rhs_source_conjugate: plan.rhs_source_conjugate(),
            output_exact_identity: plan.output_transform_is_identity(),
        }
    }

    fn of_candidate(plan: &CandidatePlan) -> Self {
        Self {
            orientation: plan.orientation,
            lhs_source_conjugate: plan.lhs_source_conjugate,
            rhs_source_conjugate: plan.rhs_source_conjugate,
            output_exact_identity: plan.output_transform_is_identity(),
        }
    }
}

fn fusion_contract_candidate_facts(
    axis_order: ContractAxisOrderCandidate,
    plan: CandidateRoute,
    inputs: FusionContractMaterializationInputs<'_>,
) -> Result<FusionContractCandidateFacts, OperationError> {
    let reverse = plan.orientation == FusionContractOrientation::RhsLhs;
    let twist_lhs = inputs
        .core_right_requires_twist
        .then(|| {
            contract_twist_on_physical_lhs(
                reverse,
                !inputs.lhs_exact_identity_borrowable,
                !inputs.rhs_exact_identity_borrowable,
                || {
                    Ok((
                        inputs.lhs_required_len.get()?,
                        inputs.rhs_required_len.get()?,
                    ))
                },
            )
        })
        .transpose()?;
    let lhs_requires_twist = twist_lhs == Some(true);
    let rhs_requires_twist = twist_lhs == Some(false);
    let lhs_materialized_elements = if inputs.lhs_exact_identity_borrowable && !lhs_requires_twist {
        0
    } else {
        inputs.lhs_required_len.get()?
    };
    let rhs_materialized_elements = if inputs.rhs_exact_identity_borrowable && !rhs_requires_twist {
        0
    } else {
        inputs.rhs_required_len.get()?
    };
    let output_exact_identity = plan.output_exact_identity;
    let output_materialized_elements = if output_exact_identity {
        0
    } else {
        inputs.output_required_len.get()?
    };
    let total_materialized_elements = lhs_materialized_elements
        .checked_add(rhs_materialized_elements)
        .and_then(|total| total.checked_add(output_materialized_elements))
        .ok_or_else(|| {
            OperationError::from_core_preserving_context(
                symtn_core::CoreError::ElementCountOverflow,
            )
        })?;
    Ok(FusionContractCandidateFacts {
        axis_order,
        orientation: plan.orientation,
        lhs_conjugate: plan.lhs_source_conjugate,
        rhs_conjugate: plan.rhs_source_conjugate,
        lhs_exact_identity_borrowable: inputs.lhs_exact_identity_borrowable,
        rhs_exact_identity_borrowable: inputs.rhs_exact_identity_borrowable,
        lhs_requires_twist,
        rhs_requires_twist,
        output_exact_identity,
        lhs_materialized_elements,
        rhs_materialized_elements,
        output_materialized_elements,
        total_materialized_elements,
    })
}

/// Whether each operand's storage and the destination are their HomSpaces'
/// canonical coupled-sector layouts (`dynamic::storage_layout_is_canonical`
/// in multiplicity-free planning), asked lazily and at most once per
/// selection: slots 0, 1 and 2 are the physical lhs, rhs and destination.
pub(crate) struct CanonicalLayouts<'a, E> {
    canonical: &'a dyn Fn(&DynamicFusionMapSpace) -> Result<bool, E>,
    answers: [std::cell::OnceCell<bool>; 3],
}

impl<'a, E> CanonicalLayouts<'a, E> {
    pub(crate) fn new(canonical: &'a dyn Fn(&DynamicFusionMapSpace) -> Result<bool, E>) -> Self {
        Self {
            canonical,
            answers: Default::default(),
        }
    }

    fn get(&self, slot: usize, space: &DynamicFusionMapSpace) -> Result<bool, E> {
        if let Some(&answer) = self.answers[slot].get() {
            return Ok(answer);
        }
        let answer = (self.canonical)(space)?;
        let _ = self.answers[slot].set(answer);
        Ok(answer)
    }
}

/// The fewest elements any DynamicTree candidate of `axes` over
/// `orientations` materializes into `dst`: TensorKit `_contract_memcost`
/// (tensoroperations.jl L378) minimized over `_contract_candidates` (L360).
/// This is the complete-admission scorer of the real selector
/// (`score_complete_fusion_contract_candidate`), run without building a
/// plan; `None` when an admission is not Complete, where the selector would
/// probe transformed layouts instead.
pub(crate) fn min_dynamic_tree_materialized_elements<E>(
    twist: impl Fn(&FusionTreeHomSpace, &[usize]) -> Result<bool, E>,
    canonical: &dyn Fn(&DynamicFusionMapSpace) -> Result<bool, E>,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    orientations: &[FusionContractOrientation],
) -> Result<Option<usize>, E>
where
    E: From<OperationError>,
{
    let complete = [dst.admission(), lhs.admission(), rhs.admission()]
        .into_iter()
        .all(|admission| matches!(admission, FusionSpaceAdmission::Complete(_)));
    if !complete {
        return Ok(None);
    }
    let layouts = CanonicalLayouts::new(canonical);
    select_best_scored_contract_candidate(axes, orientations, |candidate, orientation| {
        let candidate_axes =
            TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
        let shape = CandidatePlan::from_ranks(
            dst.nout(),
            dst.rank(),
            ContractPlanSource::rank(&lhs),
            ContractPlanSource::rank(&rhs),
            candidate_axes,
            lhs.storage_conjugate(),
            rhs.storage_conjugate(),
        )?
        .orient(orientation);
        let facts = score_complete_fusion_contract_candidate(
            &twist,
            &layouts,
            dst,
            &lhs,
            &rhs,
            candidate.clone(),
            &shape,
        )?;
        Ok(((), facts))
    })
    .map(|((), facts)| Some(facts.total_materialized_elements()))
}

/// The `DynamicTree` candidate of two operands over Complete logical spaces,
/// a lazy adjoint marked by `axes`' conjugation flags (borrowed only under
/// an identity transform, read in its parent's storage), the destination
/// derived from them by construction:
/// the Complete scorer
/// of the selector ([`prepare_tensorcontract_fusion_plan_dyn_raw_canonical`]),
/// over both orientations, with `twist` reading the core-right twist.
/// Every layout is canonical: checked admission derives the canonical
/// layout of every operand and destination (`with_test_structure` is the
/// only, test-only, exception).
pub(crate) fn select_complete_tensorcontract_fusion_plan<E>(
    twist: impl Fn(&FusionTreeHomSpace, &[usize]) -> Result<bool, E>,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, E>
where
    E: From<OperationError>,
{
    let canonical = |_: &DynamicFusionMapSpace| Ok(true);
    let layouts = CanonicalLayouts::new(&canonical);
    select_best_scored_contract_candidate(axes, &CACHED_ORIENTATIONS, |candidate, orientation| {
        let candidate_axes =
            TensorContractSpec::new(candidate.lhs(), candidate.rhs(), axes.output_permutation());
        let shape = CandidatePlan::from_ranks(
            dst.nout(),
            dst.rank(),
            lhs.rank(),
            rhs.rank(),
            candidate_axes,
            axes.lhs_conjugate(),
            axes.rhs_conjugate(),
        )?
        .orient(orientation);
        let facts = score_complete_fusion_contract_candidate(
            &twist,
            &layouts,
            dst,
            lhs,
            rhs,
            candidate.clone(),
            &shape,
        )?;
        Ok((shape, facts))
    })
    .map(|(shape, _)| shape.materialize())
}

fn score_complete_fusion_contract_candidate<S, E>(
    twist: impl Fn(&FusionTreeHomSpace, &[usize]) -> Result<bool, E>,
    layouts: &CanonicalLayouts<'_, E>,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axis_order: ContractAxisOrderCandidate,
    plan: &CandidatePlan,
) -> Result<FusionContractCandidateFacts, E>
where
    S: ContractPlanSource,
    E: From<OperationError>,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::CandidateScore);
    // Why not probe the transformed layouts: Complete admission proves the
    // full basis, so permutation preserves reduced element count. An identity
    // side is read in place exactly when the executor reads it so: a direct
    // source over its canonical layout (the derived core layout), a storage
    // adjoint over its parent's canonical layout and, when the output is the
    // identity, a canonical destination, the coupled-region proof's operands
    // (a copied partner or core destination is canonical by derivation).
    let output_exact_identity = plan.output_transform_is_identity();
    let borrowable = |slot, source: &S, [codomain, domain]: &[AxisVec; 2], conjugate| {
        let direct = super::super::dynamic::source_layout_permutation_is_borrowable(
            source.storage_space(),
            source.nout(),
            source.rank(),
            || true,
            codomain,
            domain,
            conjugate,
        );
        let adjoint = super::super::dynamic::source_is_storage_adjoint(
            conjugate,
            source.nout(),
            source.rank(),
            codomain,
            domain,
        );
        if !(direct || adjoint) || !layouts.get(slot, source.storage_space())? {
            return Ok::<_, E>(false);
        }
        Ok(direct || !output_exact_identity || layouts.get(2, dst)?)
    };
    let lhs_exact_identity_borrowable = borrowable(0, lhs, &plan.lhs, plan.lhs_source_conjugate)?;
    let rhs_exact_identity_borrowable = borrowable(1, rhs, &plan.rhs, plan.rhs_source_conjugate)?;
    let (core_right, core_right_axes) = match plan.orientation {
        FusionContractOrientation::LhsRhs => (rhs, axis_order.rhs()),
        FusionContractOrientation::RhsLhs => (lhs, axis_order.lhs()),
    };
    let core_right_requires_twist = twist(core_right.homspace(), core_right_axes)?;
    Ok(fusion_contract_candidate_facts(
        axis_order,
        CandidateRoute::of_candidate(plan),
        FusionContractMaterializationInputs {
            lhs_exact_identity_borrowable,
            rhs_exact_identity_borrowable,
            core_right_requires_twist,
            lhs_required_len: CandidateRequiredLen::Space(lhs.storage_space()),
            rhs_required_len: CandidateRequiredLen::Space(rhs.storage_space()),
            output_required_len: CandidateRequiredLen::Space(dst),
        },
    )?)
}

#[expect(
    clippy::too_many_arguments,
    reason = "candidate scoring keeps rule, three spaces, axis order, plan, probe, and optional primer independently meaningful"
)]
fn score_fusion_contract_candidate<R, S, P>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axis_order: ContractAxisOrderCandidate,
    plan: FusionContractPlan,
    probe: P,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<ScoredFusionContractCandidate, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    S: ContractPlanSource,
    P: for<'a> Fn(
            &'a R,
            &'a S,
            &'a TreeTransformOperation,
            Option<LayoutKeyBuilder<R>>,
        ) -> Result<TransformedLayoutProbe, OperationError>
        + Copy,
{
    #[cfg(test)]
    {
        crate::probe::hit(crate::probe::Event::CandidateScore);
        // One layout probe per operand side.
        crate::probe::hit(crate::probe::Event::CandidateLayoutProbe);
        crate::probe::hit(crate::probe::Event::CandidateLayoutProbe);
    }
    let lhs_core = probe(rule, lhs, plan.lhs_transform(), primer)?;
    let rhs_core = probe(rule, rhs, plan.rhs_transform(), primer)?;
    // Why an identity adjoint stays charged here, unlike the Complete scorer:
    // the artifact borrows it only over canonical coupled-sector layouts of
    // the parent, the other core operand and the destination, and a
    // non-Complete (expert) tiling certifies none of them; the copy is the
    // executor's fallback, so the charge is an upper bound.
    let lhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        lhs.storage_space(),
        lhs_core.nout,
        lhs_core.homspace.rank(),
        || lhs_core.homspace == *lhs.homspace(),
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    ) && lhs_core.source_structure_matches;
    let rhs_exact_identity_borrowable = super::super::dynamic::source_layout_metadata_is_borrowable(
        rhs.storage_space(),
        rhs_core.nout,
        rhs_core.homspace.rank(),
        || rhs_core.homspace == *rhs.homspace(),
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    ) && rhs_core.source_structure_matches;
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let core_right = if reverse { &lhs_core } else { &rhs_core };
    let core_right_requires_twist = super::super::resolution::rhs_contract_homspace_requires_twist(
        rule,
        &core_right.homspace,
        plan.core_axes().as_spec(),
    )?;
    let facts = fusion_contract_candidate_facts(
        axis_order,
        CandidateRoute::of_plan(&plan),
        FusionContractMaterializationInputs {
            lhs_exact_identity_borrowable,
            rhs_exact_identity_borrowable,
            core_right_requires_twist,
            lhs_required_len: CandidateRequiredLen::Known(lhs_core.required_len),
            rhs_required_len: CandidateRequiredLen::Known(rhs_core.required_len),
            output_required_len: CandidateRequiredLen::Space(dst),
        },
    )?;
    Ok(ScoredFusionContractCandidate { plan, facts })
}

fn validate_tensorcontract_fusion_plan_inputs<R, S>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &S,
    rhs: &S,
    axes: TensorContractSpec<'_>,
    primer: Option<LayoutKeyBuilder<R>>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    S: ContractPlanSource,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::ContractPreflight);
    let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), dst.rank(), axes)?;
    let primer = primer.unwrap_or(encoded_layout_primer::<R>);
    let matches = match primer(
        rule,
        MetadataRequest::ContractHomSpaceMatches {
            lhs: lhs.homspace(),
            rhs: rhs.homspace(),
            lhs_axes: axes.lhs_contracting_axes(),
            rhs_axes: axes.rhs_contracting_axes(),
            output_axes: axis_plan.output_axes.as_slice(),
            dst_codomain_rank: dst.nout(),
            expected: dst.homspace(),
        },
    )? {
        MetadataOutput::Matches(matches) => matches,
        _ => unreachable!("metadata dispatcher returned a non-match response"),
    };
    if !matches {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    Ok(())
}

#[cfg(test)]
fn compile_tensorcontract_fusion_plan_from_spaces(
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
) -> Result<FusionContractPlan, OperationError> {
    compile_tensorcontract_fusion_plan_from_ranks(
        dst.nout(),
        dst.rank(),
        lhs.rank(),
        rhs.rank(),
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_tensorcontract_fusion_plan_from_ranks(
    dst_nout: usize,
    dst_rank: usize,
    lhs_rank: usize,
    rhs_rank: usize,
    axes: TensorContractSpec<'_>,
    lhs_source_conjugate: bool,
    rhs_source_conjugate: bool,
) -> Result<FusionContractPlan, OperationError> {
    CandidatePlan::from_ranks(
        dst_nout,
        dst_rank,
        lhs_rank,
        rhs_rank,
        axes,
        lhs_source_conjugate,
        rhs_source_conjugate,
    )
    .map(|plan| plan.materialize())
}

#[cfg(test)]
mod tests {
    use super::{
        contracted_axis_order_candidates, prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
        prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical,
        prepare_tensorcontract_fusion_plan_dyn_raw,
        prepare_tensorcontract_fusion_plan_dyn_raw_canonical,
        prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation,
        FusionContractOrientation,
    };
    use crate::contract::dynamic_space::{encoded_layout_primer, MetadataOutput, MetadataRequest};
    use crate::contract::{DynamicFusionMapSpace, FusionOperand};
    use crate::probe::{self, Event};
    use crate::{TreeTransformOperation, TreeTransformOperationKind};
    use symtn_core::{
        BlockKey, BlockStructure, FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace,
        SU2FusionRule, SectorId, SectorLeg, U1FusionRule,
    };
    use symtn_operations::{OutputAxisOrder, TensorContractSpec};

    std::thread_local! {
        static REJECT_NEXT_PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static OPERAND_PREFLIGHT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn rejecting_contract_preflight(
        _rule: &U1FusionRule,
        request: MetadataRequest<'_>,
    ) -> Result<MetadataOutput, crate::OperationError> {
        OPERAND_PREFLIGHT_CALLS.set(OPERAND_PREFLIGHT_CALLS.get() + 1);
        if !matches!(request, MetadataRequest::ContractHomSpaceMatches { .. }) {
            return Err(crate::OperationError::InvalidArgument {
                message: "operand plan prepared a layout during preflight",
            });
        }
        Err(crate::OperationError::InvalidArgument {
            message: "operand plan called bound contract preflight",
        })
    }

    fn reject_next_probe(
        rule: &U1FusionRule,
        space: &DynamicFusionMapSpace,
        operation: &TreeTransformOperation,
        primer: Option<super::LayoutKeyBuilder<U1FusionRule>>,
    ) -> Result<super::TransformedLayoutProbe, crate::OperationError> {
        if REJECT_NEXT_PROBE.replace(false) {
            Err(crate::OperationError::UnsupportedTensorContractScope {
                message: "test candidate layout unavailable",
            })
        } else {
            super::encoded_layout_probe(rule, space, operation, primer)
        }
    }

    fn single_sector_space(rule: &U1FusionRule, dimensions: [usize; 4]) -> DynamicFusionMapSpace {
        let homspace = FusionTreeHomSpace::from_sector_ids(
            [(0, dimensions[0]), (0, dimensions[1])],
            [(0, dimensions[2]), (0, dimensions[3])],
        );
        DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap()
    }

    fn subset_copy(rule: &U1FusionRule, complete: &DynamicFusionMapSpace) -> DynamicFusionMapSpace {
        DynamicFusionMapSpace::try_from_layout(
            complete.homspace().clone(),
            BlockStructure::clone(complete.structure()),
        )
        .unwrap()
        .try_bind_rule(rule)
        .unwrap()
    }

    fn selected_contract_axes(
        lhs_dimensions: [usize; 4],
        rhs_dimensions: [usize; 4],
    ) -> (Vec<usize>, Vec<usize>) {
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, lhs_dimensions);
        let rhs = single_sector_space(&rule, rhs_dimensions);
        let dst = single_sector_space(
            &rule,
            [
                lhs_dimensions[0],
                lhs_dimensions[1],
                rhs_dimensions[2],
                rhs_dimensions[3],
            ],
        );
        let plan = prepare_tensorcontract_fusion_plan_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]),
        )
        .unwrap();
        assert_eq!(
            plan.lhs_transform().kind(),
            TreeTransformOperationKind::Permute,
            "lhs lowering must be a permutation"
        );
        assert_eq!(
            plan.rhs_transform().kind(),
            TreeTransformOperationKind::Permute,
            "rhs lowering must be a permutation"
        );
        (
            plan.lhs_transform().domain_permutation().to_vec(),
            plan.rhs_transform().codomain_permutation().to_vec(),
        )
    }

    #[test]
    fn operand_plan_preserves_the_bound_layout_capability() {
        // What: prelowered candidate planning reaches the bound dispatcher
        // exactly once through an unprepared contraction-HomSpace request.
        let rule = U1FusionRule;
        let lhs_typed = single_sector_space(&rule, [2, 3, 4, 5]);
        let rhs_typed = single_sector_space(&rule, [5, 4, 6, 7]);
        let lhs = subset_copy(&rule, &lhs_typed);
        let rhs = subset_copy(&rule, &rhs_typed);
        let dst = single_sector_space(&rule, [2, 3, 6, 7]);
        let lhs_layout = FusionOperand::direct(&lhs)
            .prepare(&rule, encoded_layout_primer::<U1FusionRule>)
            .unwrap();
        let rhs_layout = FusionOperand::direct(&rhs)
            .prepare(&rule, encoded_layout_primer::<U1FusionRule>)
            .unwrap();
        let axes = || TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]);

        OPERAND_PREFLIGHT_CALLS.set(0);
        assert_eq!(
            prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical(
                &rule,
                &dst,
                &lhs_layout,
                &rhs_layout,
                axes(),
                rejecting_contract_preflight,
            )
            .unwrap_err(),
            crate::OperationError::InvalidArgument {
                message: "operand plan called bound contract preflight",
            }
        );
        assert_eq!(OPERAND_PREFLIGHT_CALLS.get(), 1);
    }

    #[test]
    fn candidates_keep_lhs_rhs_pairs_intact() {
        let candidates = contracted_axis_order_candidates(&[3, 1, 2], &[6, 8, 4]);
        assert_eq!(candidates[0].lhs(), &[1, 2, 3]);
        assert_eq!(candidates[0].rhs(), &[8, 4, 6]);
        assert_eq!(candidates[1].lhs(), &[2, 3, 1]);
        assert_eq!(candidates[1].rhs(), &[4, 6, 8]);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn already_canonical_axes_do_not_duplicate_candidates() {
        let candidates = contracted_axis_order_candidates(&[0, 2], &[1, 3]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].lhs(), &[0, 2]);
        assert_eq!(candidates[0].rhs(), &[1, 3]);
    }

    #[test]
    fn duplicate_axis_values_use_stable_pair_order() {
        let candidates = contracted_axis_order_candidates(&[2, 1, 2], &[7, 3, 5]);
        assert_eq!(candidates[0].lhs(), &[1, 2, 2]);
        assert_eq!(candidates[0].rhs(), &[3, 7, 5]);
    }

    #[test]
    fn lhs_sorted_candidate_wins_when_rhs_is_the_smaller_replay() {
        // What: crossed pairs lower to the LHS-sorted plan when only the small RHS is materialized.
        assert_eq!(
            selected_contract_axes([16, 16, 2, 3], [3, 2, 1, 1]),
            (vec![2, 3], vec![1, 0])
        );
    }

    #[test]
    fn rhs_sorted_candidate_wins_when_lhs_is_the_smaller_replay() {
        // What: the same crossed pairs lower to the RHS-sorted plan after operand sizes reverse.
        assert_eq!(
            selected_contract_axes([1, 1, 2, 3], [3, 2, 16, 16]),
            (vec![3, 2], vec![0, 1])
        );
    }

    #[test]
    fn canonical_selector_chooses_reverse_when_it_avoids_both_forward_transforms() {
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [2, 3, 5, 7]);
        let rhs = single_sector_space(&rule, [11, 13, 3, 2]);
        let dst = single_sector_space(&rule, [11, 13, 5, 7]);
        let axes =
            TensorContractSpec::new(&[1, 0], &[2, 3], OutputAxisOrder::from_axes(&[2, 3, 0, 1]));
        let facts =
            prepare_tensorcontract_fusion_candidate_facts_dyn_raw(&rule, &dst, &lhs, &rhs, axes)
                .unwrap();
        let selected = super::prepare_tensorcontract_fusion_plan_dyn_raw_canonical(
            &rule, &dst, &lhs, &rhs, axes,
        )
        .unwrap();

        assert_eq!(facts.len(), 4);
        assert!(facts[..2]
            .iter()
            .all(|candidate| candidate.total_materialized_elements() > 0));
        assert!(
            facts[2..]
                .iter()
                .map(|candidate| candidate.total_materialized_elements())
                .min()
                .unwrap()
                < facts[..2]
                    .iter()
                    .map(|candidate| candidate.total_materialized_elements())
                    .min()
                    .unwrap()
        );
        assert_eq!(selected.orientation(), FusionContractOrientation::RhsLhs);
        assert_eq!(
            selected.lhs_transform(),
            &TreeTransformOperation::permute([1, 0], [2, 3])
        );
        assert_eq!(
            selected.rhs_transform(),
            &TreeTransformOperation::permute([0, 1], [2, 3])
        );
        assert!(selected.output_transform_is_identity());
    }

    #[test]
    fn equal_cost_keeps_lhs_first_across_repeats_and_thread_counts() {
        // What: an equal allocation cost always selects the stable LHS-sorted candidate.
        for threads in [1, 2, 4] {
            let mut pool = symtn_operations::host_pool::test_pool(threads);
            for _ in 0..4 {
                assert_eq!(
                    pool.install(|_engine| selected_contract_axes([1, 1, 2, 3], [3, 2, 1, 1])),
                    (vec![2, 3], vec![1, 0])
                );
            }
        }
    }

    #[test]
    fn complete_selector_skips_loser_layouts_and_matches_exact_fallback() {
        let rule = U1FusionRule;
        let lhs_complete = single_sector_space(&rule, [16, 16, 2, 3]);
        let rhs_complete = single_sector_space(&rule, [3, 2, 1, 1]);
        let dst_complete = single_sector_space(&rule, [16, 16, 1, 1]);
        let lhs = lhs_complete.clone();
        let rhs = rhs_complete.clone();
        let dst = dst_complete.clone();
        let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]);

        let candidate_score_calls_rec = probe::record();
        let candidate_build_calls_rec = probe::record();
        let complete =
            prepare_tensorcontract_fusion_plan_dyn_raw_canonical(&rule, &dst, &lhs, &rhs, axes)
                .unwrap();
        // What: Complete cold selection performs one semantic HomSpace preflight
        // and scores all stable candidates without constructing their layouts.
        assert_eq!(candidate_score_calls_rec.get(Event::CandidateScore), 4);
        assert_eq!(
            (
                candidate_build_calls_rec.get(Event::ContractPreflight),
                candidate_build_calls_rec.get(Event::CandidateLayoutProbe)
            ),
            (1, 0)
        );

        let candidate_score_calls_rec = probe::record();
        let candidate_build_calls_rec = probe::record();
        let exact = prepare_tensorcontract_fusion_plan_dyn_raw_canonical(
            &rule,
            &subset_copy(&rule, &dst_complete),
            &subset_copy(&rule, &lhs_complete),
            &subset_copy(&rule, &rhs_complete),
            axes,
        )
        .unwrap();
        // What: the conservative expert fallback probes both operands for each
        // candidate and selects the same mathematical winner.
        assert_eq!(candidate_score_calls_rec.get(Event::CandidateScore), 4);
        assert_eq!(
            (
                candidate_build_calls_rec.get(Event::ContractPreflight),
                candidate_build_calls_rec.get(Event::CandidateLayoutProbe)
            ),
            (1, 8)
        );
        assert_eq!(complete, exact);
    }

    #[test]
    fn complete_fz2_twist_cost_matches_exact_probe() {
        let rule = FermionParityFusionRule;
        let odd = SectorId::new(1);
        let space = |codomain_dual: bool, domain_dual: bool| {
            DynamicFusionMapSpace::from_final_homspace(
                &rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([SectorLeg::new([(odd, 1)], codomain_dual)]),
                    FusionProductSpace::new([SectorLeg::new([(odd, 1)], domain_dual)]),
                ),
            )
            .unwrap()
        };
        let lhs = space(false, true);
        let rhs = space(true, false);
        let dst = space(false, false);
        let axis_order = contracted_axis_order_candidates(&[1], &[0]).remove(0);
        let axes =
            TensorContractSpec::with_default_output_order(axis_order.lhs(), axis_order.rhs());
        let plan = super::compile_tensorcontract_fusion_plan_from_spaces(
            &dst, &lhs, &rhs, axes, false, false,
        )
        .unwrap();

        let canonical = |space: &DynamicFusionMapSpace| {
            super::super::super::dynamic::storage_layout_is_canonical(&rule, space)
        };
        let complete = super::score_complete_fusion_contract_candidate(
            |homspace, axes| super::source_contract_homspace_requires_twist(&rule, homspace, axes),
            &super::CanonicalLayouts::new(&canonical),
            &dst,
            &lhs,
            &rhs,
            axis_order.clone(),
            &super::CandidatePlan::from_plan(&plan),
        )
        .unwrap();
        let exact = super::score_fusion_contract_candidate(
            &rule,
            &dst,
            &lhs,
            &rhs,
            axis_order,
            plan,
            super::encoded_layout_probe,
            None,
        )
        .unwrap()
        .facts;

        // What: Complete structural scoring observes the same odd RHS twist
        // and materialization cost as the exact transformed-layout probe.
        assert!(complete.rhs_requires_twist());
        assert_eq!(complete, exact);
    }

    #[test]
    fn exact_fallback_skips_an_unavailable_loser() {
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [1, 1, 2, 3]);
        let rhs = single_sector_space(&rule, [3, 2, 16, 16]);
        let dst = single_sector_space(&rule, [1, 1, 16, 16]);
        let lhs = subset_copy(&rule, &lhs);
        let rhs = subset_copy(&rule, &rhs);
        let dst = subset_copy(&rule, &dst);
        let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]);
        let expected =
            prepare_tensorcontract_fusion_plan_dyn_raw(&rule, &dst, &lhs, &rhs, axes).unwrap();

        REJECT_NEXT_PROBE.set(true);
        let actual = super::select_tensorcontract_fusion_plan_from_spaces_with_probe(
            &rule,
            &dst,
            &lhs,
            &rhs,
            axes,
            false,
            false,
            reject_next_probe,
            None,
        )
        .unwrap();

        // What: a candidate-local capability failure cannot reject a later
        // stable candidate that selects the same valid winner.
        assert_eq!(actual, expected);
    }

    #[test]
    fn asymmetric_u1_candidates_expose_separate_source_components() {
        // What: each paired U1 candidate reports the exact source it must
        // materialize, rather than exposing only the winning total.
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [16, 16, 2, 3]);
        let rhs = single_sector_space(&rule, [3, 2, 1, 1]);
        let dst = single_sector_space(&rule, [16, 16, 1, 1]);
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::with_default_output_order(&[2, 3], &[1, 0]),
        )
        .unwrap();

        assert_eq!(facts.len(), 4);
        assert_eq!(facts[0].axis_order().lhs(), &[2, 3]);
        assert_eq!(facts[0].axis_order().rhs(), &[1, 0]);
        assert_eq!(facts[0].orientation(), FusionContractOrientation::LhsRhs);
        assert!(!facts[0].lhs_conjugate());
        assert!(!facts[0].rhs_conjugate());
        assert!(facts[0].lhs_exact_identity_borrowable());
        assert!(!facts[0].rhs_exact_identity_borrowable());
        assert!(!facts[0].rhs_requires_twist());
        assert!(facts[0].output_exact_identity());
        assert_eq!(facts[0].lhs_materialized_elements(), 0);
        assert_eq!(facts[0].rhs_materialized_elements(), 6);
        assert_eq!(facts[0].output_materialized_elements(), 0);
        assert_eq!(facts[0].total_materialized_elements(), 6);

        assert_eq!(facts[1].axis_order().lhs(), &[3, 2]);
        assert_eq!(facts[1].axis_order().rhs(), &[0, 1]);
        assert!(!facts[1].lhs_exact_identity_borrowable());
        assert!(facts[1].rhs_exact_identity_borrowable());
        assert_eq!(facts[1].lhs_materialized_elements(), 1_536);
        assert_eq!(facts[1].rhs_materialized_elements(), 0);
        assert_eq!(facts[1].output_materialized_elements(), 0);
        assert_eq!(facts[1].total_materialized_elements(), 1_536);
        assert_eq!(facts[2].orientation(), FusionContractOrientation::RhsLhs);
        assert_eq!(facts[3].orientation(), FusionContractOrientation::RhsLhs);
    }

    #[test]
    fn nonidentity_output_is_one_checked_score_component() {
        // What: a requested nonidentity output order contributes exactly one
        // complete destination buffer to every current candidate.
        let rule = U1FusionRule;
        let lhs = single_sector_space(&rule, [11, 13, 2, 3]);
        let rhs = single_sector_space(&rule, [3, 2, 5, 7]);
        let dst = single_sector_space(&rule, [13, 11, 7, 5]);
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new(&[2, 3], &[1, 0], OutputAxisOrder::from_axes(&[1, 0, 3, 2])),
        )
        .unwrap();
        let dst_elements = dst.required_len().unwrap();

        assert_eq!(facts.len(), 4);
        for candidate in facts {
            assert!(!candidate.output_exact_identity());
            assert_eq!(candidate.output_materialized_elements(), dst_elements);
            assert_eq!(
                candidate.total_materialized_elements(),
                candidate.lhs_materialized_elements()
                    + candidate.rhs_materialized_elements()
                    + dst_elements
            );
        }
    }

    #[test]
    fn source_conjugation_is_an_explicit_candidate_fact() {
        // What: adjoint lowering records storage conjugation independently
        // from axis order and charges the affected source exactly once.
        let rule = U1FusionRule;
        let sector = symtn_core::U1Irrep::new(0).sector_id();
        let vector_space = || {
            DynamicFusionMapSpace::from_final_homspace(
                &rule,
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]),
                    FusionProductSpace::new(Vec::<SectorLeg>::new()),
                ),
            )
            .unwrap()
        };
        let lhs = vector_space();
        let rhs = vector_space();
        let dst = DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
                FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
            ),
        )
        .unwrap();
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new_with_conjugation(
                &[],
                &[],
                OutputAxisOrder::identity(),
                true,
                false,
            ),
        )
        .unwrap();

        assert_eq!(facts.len(), 2);
        assert!(facts[0].lhs_conjugate());
        assert!(!facts[0].rhs_conjugate());
        assert!(!facts[0].lhs_exact_identity_borrowable());
        assert_eq!(facts[0].lhs_materialized_elements(), 1);
        assert_eq!(
            facts[0].total_materialized_elements(),
            facts[0].lhs_materialized_elements()
                + facts[0].rhs_materialized_elements()
                + facts[0].output_materialized_elements()
        );
        assert_eq!(facts[1].orientation(), FusionContractOrientation::RhsLhs);
        assert!(!facts[1].lhs_exact_identity_borrowable());

        let candidate = contracted_axis_order_candidates(&[], &[]).remove(0);
        let reverse = prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new_with_conjugation(
                &[],
                &[],
                OutputAxisOrder::identity(),
                true,
                false,
            ),
            &candidate,
            FusionContractOrientation::RhsLhs,
        )
        .unwrap();
        assert_eq!(reverse.orientation(), FusionContractOrientation::RhsLhs);
        assert!(reverse.lhs_source_conjugate());
        assert!(!reverse.rhs_source_conjugate());
    }

    #[test]
    fn selector_cost_materializes_identity_operand_with_missing_structural_zero() {
        // What: an identity-axis operand with an incomplete SU2 tree grid is
        // charged for the complete core layout instead of treated as borrowed.
        let rule = SU2FusionRule;
        let lhs = DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::from_sector_ids([], []),
        )
        .unwrap();
        let rhs_hom = FusionTreeHomSpace::from_sector_ids([], [(1, 1), (1, 1), (1, 1), (1, 1)]);
        let rhs_keys = rhs_hom.fusion_tree_keys(&rule);
        assert_eq!(rhs_keys.len(), 2);
        let rhs = DynamicFusionMapSpace::try_from_layout(
            rhs_hom.clone(),
            crate::tests::packed_fixture_structure(
                4,
                [(BlockKey::from(rhs_keys[0].clone()), vec![1, 1, 1, 1])],
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let dst_hom = rhs_hom.permute(&rule, &[0, 1, 2, 3], &[]).unwrap();
        let dst_keys = dst_hom.fusion_tree_keys(&rule);
        let dst = DynamicFusionMapSpace::try_from_layout(
            dst_hom,
            crate::tests::packed_fixture_structure(
                4,
                dst_keys
                    .iter()
                    .cloned()
                    .map(|key| (BlockKey::from(key), vec![1, 1, 1, 1])),
            )
            .unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
        let facts = prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
            &rule,
            &dst,
            &lhs,
            &rhs,
            TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        )
        .unwrap();

        assert_eq!(facts.len(), 2);
        assert!(!facts[0].rhs_exact_identity_borrowable());
        assert_eq!(facts[0].lhs_materialized_elements(), 0);
        assert_eq!(facts[0].rhs_materialized_elements(), 2);
        assert_eq!(facts[0].output_materialized_elements(), 2);
        assert_eq!(facts[0].total_materialized_elements(), 4);
    }
}
