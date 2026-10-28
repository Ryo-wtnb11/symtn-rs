//! Eager contraction route compilation.
//!
//! Ordinary calls resolve per operation, as TensorKit and QSpace do. Explicit
//! prepared handles own the returned [`StorageContractResolution`] for
//! lookup-free replay.

use std::cmp::Ordering;
use std::sync::Arc;

use symtn_core::{
    BlockStructure, FusionSpaceAdmission, FusionTreeHomSpace, FusionTreePairOrientation,
    MultiplicityFreeRigidSymbols,
};

use crate::mode::PlanningAlgebra;
use crate::{DenseBlockScalar, OperationError};
use symtn_operations::axis::{OutputAxisOrder, TensorContractSpec};
use symtn_operations::fusion_replay::FusionBlockContractPlan;
use symtn_operations::TreeTransformStructure;

use super::context::PlanTarget;
use super::dynamic_space::{DynamicFusionMapSpace, FusionOperand};
use super::fusion::{
    contracted_axis_order_candidates, external_axis_is_dual,
    min_dynamic_tree_materialized_elements, FusionContractOrientation, CACHED_ORIENTATIONS,
};
use super::fusion_block::{
    compile_fusion_block_contract_plan_core_geometry, try_compile_oriented_canonical_core_plan,
    try_compile_scaled_canonical_core_plan, CoreContractPreflight, ValidatedCoreContract,
};

/// Host-compiled, owned route of one contraction whose payloads are not
/// host slices (the device path): everything categorical — route choice,
/// orientation and axis order, source/output transform structures, borrow
/// decisions, core plan and twist classification — is decided here, on the
/// host, by the same compilers the Host contraction runs; a storage executor
/// only replays it.
///
/// Created by
/// [`TensorContractFusionExecutionContext::plan_contract`](super::TensorContractFusionExecutionContext::plan_contract).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct StorageContractResolution<C = f64> {
    pub(crate) route: ContractRoute<C>,
}

impl<C: DenseBlockScalar> ContractRoute<C> {
    /// The core plan whose GEMMs this route runs.
    pub(crate) fn block_plan(&self) -> &FusionBlockContractPlan<C> {
        match self {
            Self::Core { plan, .. } => plan,
            Self::DynamicTree(artifact) => artifact.block_plan(),
            Self::CopyC(copy) => &copy.core,
        }
    }
}

/// The one route vocabulary of a multiplicity-free storage contraction,
/// chosen by `plan_contract` (TensorKit `contract!` / `blas_contract!`,
/// `tensoroperations.jl:314-447` @cfaa073).
#[derive(Clone, Debug)]
pub(crate) enum ContractRoute<C> {
    /// Canonical fully-direct coupled-sector GEMM batch over the parent
    /// buffers (lazy adjoints as GEMM operand flags, a uniform fermionic twist
    /// as per-job alpha). `swapped` runs the B·A candidate: the plan's left
    /// operand is the caller's rhs (TensorKit's
    /// `blas_contract!(C, B, reverse(pB), A, reverse(pA), pAB′)`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Core {
        plan: Arc<FusionBlockContractPlan<C>>,
        swapped: bool,
    },
    /// Source tree transforms → fully-direct core GEMM → output transform.
    DynamicTree(Arc<super::dynamic::DynamicTreeExecutionArtifact<C>>),
    /// TensorKit's `copyC` (`blas_contract!`, `tensoroperations.jl:436-446` @cfaa073):
    /// a zero-copy core into a temporary in its own default output, then one
    /// tree transform into the destination.
    CopyC(CopyCRoute<C>),
}

/// Which TensorKit operation a planned contraction is, decided before route
/// planning: the axes alone cannot tell them apart, because a fermionic
/// composition and the contraction over the same legs differ by the
/// supertrace twist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContractKind {
    /// TensorKit `tensorcontract!` / `blas_contract!`
    /// (`tensoroperations.jl:383-452` @cfaa073): the candidate walk, the
    /// supertrace twist on dual rhs-contracted legs, then Core, `copyC` and
    /// `DynamicTree`.
    Contract,
    /// TensorKit `mul!` (`linalg.jl:330-370` @cfaa073): A's domain against
    /// B's codomain in order and the identity output, with no twist, no
    /// candidate walk and no transform route, so only the canonical Core.
    Compose,
}

/// The planner's first rung: the canonical core over the parent buffers, or
/// what its candidate walk learned for the rest of the planner.
#[doc(hidden)]
pub enum CoreRoute<C> {
    Hit(StorageContractResolution<C>),
    Miss(CoreMiss),
}

impl<C> CoreRoute<C> {
    #[doc(hidden)]
    pub fn hit(self) -> Option<StorageContractResolution<C>> {
        match self {
            Self::Hit(resolution) => Some(resolution),
            Self::Miss(_) => None,
        }
    }
}

/// A canonical-core miss of one contraction request: the input of
/// `plan_contract_beyond_core` for that same request, carrying whether the
/// requested order had a zero-copy candidate (the core rung then declined
/// it), so the planner walks the requested order once.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct CoreMiss {
    pub(crate) requested_zero_copy: bool,
}

/// The plans of a [`ContractRoute::CopyC`] route: shared plans, held inline
/// (one allocation fewer per planned contraction).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct CopyCRoute<C> {
    /// The core into the temporary, over the caller's operands.
    pub(crate) core: Arc<FusionBlockContractPlan<C>>,
    /// Whether the core's left operand is the caller's rhs.
    pub(crate) swapped: bool,
    pub(crate) temporary: Arc<BlockStructure>,
    pub(crate) temporary_len: usize,
    /// Permutes the temporary into the destination's order and split.
    pub(crate) transform: TreeTransformStructure<C>,
}

impl<C> CopyCRoute<C> {
    #[doc(hidden)]
    pub fn temporary_structure(&self) -> &Arc<BlockStructure> {
        &self.temporary
    }

    /// Elements of one temporary.
    #[doc(hidden)]
    pub fn temporary_len(&self) -> usize {
        self.temporary_len
    }

    #[doc(hidden)]
    pub fn transform(&self) -> &TreeTransformStructure<C> {
        &self.transform
    }
}

impl<C: DenseBlockScalar> StorageContractResolution<C> {
    /// Backend-neutral: a Host resolution never runs a device check, so it
    /// resolves identically with and without the `cuda` feature. Device
    /// replays admit the core's inactive blocks themselves
    /// ([`Self::admit_cuda_inactive_regions`]).
    pub(crate) fn new(route: ContractRoute<C>) -> Self {
        Self { route }
    }

    /// The CUDA admission of the core plan's inactive destination blocks:
    /// each must have a device zero region (unsigned strides and offset).
    /// Run by every CUDA entry before device work.
    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    pub fn admit_cuda_inactive_regions(&self) -> Result<(), OperationError>
    where
        C: Copy + PartialEq + num_traits::One,
    {
        symtn_operations::cuda_transform::CudaMemberZeroRegions::admit(
            self.route.block_plan().inactive_destination_regions(),
        )
    }

    /// True when the route needs the fermionic contraction twist of one
    /// materialized operand: some non-empty block of that operand's
    /// transformed source has `θ_b ≠ 1`. Host and device alike fold `θ_b`
    /// into the source-transform move writing block `b`. A twist that falls
    /// only on zero-element blocks scales no data and reports `false`.
    pub fn requires_source_twist(&self) -> bool {
        match &self.route {
            ContractRoute::Core { .. } | ContractRoute::CopyC(_) => false,
            ContractRoute::DynamicTree(artifact) => artifact.requires_source_twist(),
        }
    }

    /// The core plan's inactive destination blocks when the core GEMMs write
    /// the caller's destination directly (the `Core` route, or an identity
    /// output); `None` when an output transform writes it.
    #[cfg(test)]
    pub(crate) fn direct_destination_inactive_blocks(&self) -> Option<usize> {
        match &self.route {
            ContractRoute::Core { plan, .. } => Some(plan.inactive_destination_regions().len()),
            ContractRoute::DynamicTree(artifact) => artifact.direct_destination_inactive_blocks(),
            ContractRoute::CopyC(_) => None,
        }
    }

    /// The core plan whose GEMMs this route runs. A `DynamicTree` core is
    /// unit-coefficient: the fermionic twist rides its source transforms.
    /// Member replays state their admission on it once, independent of `B`.
    #[doc(hidden)]
    pub fn core_plan(&self) -> &FusionBlockContractPlan<C> {
        self.route.block_plan()
    }

    /// True when the route runs source/output tree transforms around the core.
    pub fn is_dynamic_tree(&self) -> bool {
        matches!(self.route, ContractRoute::DynamicTree(_))
    }

    /// The core plan whose GEMMs read the caller's operands directly (the
    /// `Core` and `CopyC` routes), and whether the plan's left operand is the
    /// caller's rhs; `None` for a DynamicTree route. Under `CopyC` the core
    /// writes the temporary ([`Self::copy_c`]). Callers state the replay
    /// contract they need on the plan (`require_identity_(signed_)direct_replay`)
    /// or build it (`StackedDirectReplay::new_signed`).
    #[doc(hidden)]
    pub fn direct_core(&self) -> Option<(&Arc<FusionBlockContractPlan<C>>, bool)> {
        match &self.route {
            ContractRoute::Core { plan, swapped } => Some((plan, *swapped)),
            ContractRoute::CopyC(copy) => Some((&copy.core, copy.swapped)),
            ContractRoute::DynamicTree(_) => None,
        }
    }

    /// The temporary and output transform of a `CopyC` route.
    #[doc(hidden)]
    pub fn copy_c(&self) -> Option<&CopyCRoute<C>> {
        match &self.route {
            ContractRoute::CopyC(copy) => Some(copy),
            ContractRoute::Core { .. } | ContractRoute::DynamicTree(_) => None,
        }
    }

    /// The derived intermediates this route replays over: a `CopyC`
    /// temporary, or a `DynamicTree`'s core sources and core destination.
    pub(crate) fn derived_structures(&self) -> smallvec::SmallVec<[&Arc<BlockStructure>; 3]> {
        match &self.route {
            ContractRoute::Core { .. } => smallvec::SmallVec::new(),
            ContractRoute::CopyC(copy) => smallvec::smallvec![&copy.temporary],
            ContractRoute::DynamicTree(artifact) => {
                let mut structures = smallvec::smallvec![
                    artifact.lhs_transform.space.structure(),
                    artifact.rhs_transform.space.structure()
                ];
                if let Some(core) = &artifact.core_dst {
                    structures.push(core.space.structure());
                }
                structures
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn is_swapped_core(&self) -> bool {
        matches!(self.route, ContractRoute::Core { swapped: true, .. })
    }
}

#[cfg(feature = "cuda")]
impl StorageContractResolution<f64> {
    /// Checks, without a device, that this route has a CUDA member replay,
    /// and returns the device plan entries one member workspace holds:
    /// distinct core GEMM shapes, core zero fills, and each transform stage's
    /// zero fills and scaled moves.
    ///
    /// Core and CopyC admit exact +1/-1 job coefficients; a DynamicTree core
    /// must be unit, its fermionic twist riding the source moves. Every
    /// transform stage must be unconjugated nonzero Single moves.
    #[doc(hidden)]
    pub fn admit_cuda_members(&self) -> Result<usize, OperationError> {
        use symtn_operations::cuda_transform::CudaSingleMemberRegions;
        match &self.route {
            ContractRoute::Core { plan, .. } => {
                plan.require_identity_signed_direct_replay()?;
                self.admit_cuda_inactive_regions()?;
                Ok(plan.cuda_direct_plan_entries())
            }
            ContractRoute::CopyC(copy) => {
                copy.core.require_identity_signed_direct_replay()?;
                let output = CudaSingleMemberRegions::admit(&copy.transform)?;
                self.admit_cuda_inactive_regions()?;
                Ok(copy.core.cuda_direct_plan_entries() + output)
            }
            ContractRoute::DynamicTree(artifact) => {
                self.admit_cuda_inactive_regions()?;
                artifact.block_plan.require_identity_direct_replay()?;
                let mut entries = artifact.block_plan.cuda_direct_plan_entries();
                let [lhs_scales, rhs_scales] = artifact.stage_scales();
                for (borrowed, transform, scales) in [
                    (artifact.lhs_borrowed, &artifact.lhs_transform, lhs_scales),
                    (artifact.rhs_borrowed, &artifact.rhs_transform, rhs_scales),
                ] {
                    if let (false, Some(structure)) = (borrowed, &transform.transform_structure) {
                        entries += CudaSingleMemberRegions::admit_scaled(structure, scales)?;
                    }
                }
                if let Some(output) = &artifact.core_dst {
                    entries += CudaSingleMemberRegions::admit(&output.output_transform_structure)?;
                }
                Ok(entries)
            }
        }
    }
}

/// Runs `compile` on the TensorKit `contract!` candidates
/// (`_contract_candidates`: pairs sorted by lhs or by rhs keys, times A·B or
/// B·A) that need no source or output materialization, in TensorKit's tie
/// order m1..m4, and returns the first compiled route.
///
/// A candidate qualifies when, over the *oriented* operand HomSpaces, its
/// core-left operand contracts exactly its domain and its core-right operand
/// exactly its codomain, both in order, and its output is the identity: the
/// core GEMM then reads both parent buffers, a lazy adjoint as a GEMM
/// adjoint flag. This is TensorKit `_contract_memcost == 0` with
/// `has_shared_permute(::AdjointTensorMap)` (indexmanipulations.jl L282)
/// delegating to the parent: a lazy adjoint whose permuted index order is its
/// natural adjoint order is free. Unless `twist_consumable`, a candidate
/// whose core-right contracted legs carry the fermionic supertrace twist is
/// excluded, since `blas_contract!` must then copy one operand to twist it.
///
/// Why before the DynamicTree scorer, not inside it: a zero-cost candidate
/// runs on the Core route with no DynamicTree artifact, and over a
/// non-Complete admission that scorer still charges an identity adjoint it
/// cannot certify as borrowable (`score_fusion_contract_candidate`).
/// Every qualifying candidate has cost zero, the global minimum, so taking
/// the first in tie order selects what `select_best_scored_contract_candidate`
/// would under TensorKit's cost; when none compiles, the DynamicTree scorer
/// runs unchanged. The checks here are axis and dual-flag comparisons only;
/// `compile` runs the categorical preflight for a qualifying candidate.
///
/// `candidate_seen` becomes true when some candidate qualifies (passes the
/// geometry checks, and the twist check unless `twist_consumable`): the
/// question the `copyC` decision asks of the requested order, answered by
/// this walk so the planner walks it once. `twist_possible` answers whether
/// the rule's braiding can twist at all (Fermionic), asked once per walk
/// that reaches the candidates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_zero_copy_contract_candidates<'a, T, E>(
    twist_possible: impl FnOnce() -> Result<bool, E>,
    dst_nout: usize,
    lhs: FusionOperand<'a>,
    rhs: FusionOperand<'a>,
    axes: TensorContractSpec<'_>,
    twist_consumable: bool,
    candidate_seen: &mut bool,
    mut compile: impl FnMut(
        FusionOperand<'a>,
        FusionOperand<'a>,
        TensorContractSpec<'_>,
        FusionContractOrientation,
    ) -> Result<Option<T>, E>,
) -> Result<Option<T>, E> {
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::CandidateWalk);
    let (lhs_axes, rhs_axes) = (axes.lhs_contracting_axes(), axes.rhs_contracting_axes());
    let contracted = lhs_axes.len();
    let (lhs_rank, rhs_rank) = (lhs.storage_space().rank(), rhs.storage_space().rank());
    if contracted != rhs_axes.len() || contracted > lhs_rank || contracted > rhs_rank {
        return Ok(None);
    }
    let lhs_open = lhs_rank - contracted;
    let open = lhs_open + rhs_rank - contracted;
    let output_is = |expected: &mut dyn Iterator<Item = usize>| match axes.output_permutation() {
        OutputAxisOrder::Identity => (0..open).eq(expected),
        OutputAxisOrder::Axes(output) => output.iter().copied().eq(expected),
    };
    // pAB′: B·A writes (rhs open | lhs open).
    let output_ok = [
        output_is(&mut (0..open)),
        output_is(&mut (lhs_open..open).chain(0..lhs_open)),
    ];
    if !output_ok.contains(&true) {
        return Ok(None);
    }
    let fermionic = twist_possible()?;
    let candidates = contracted_axis_order_candidates(lhs_axes, rhs_axes);
    for (orientation, output_ok) in CACHED_ORIENTATIONS.into_iter().zip(output_ok) {
        if !output_ok {
            continue;
        }
        for candidate in &candidates {
            let (core_lhs, core_rhs, core_lhs_axes, core_rhs_axes) = match orientation {
                FusionContractOrientation::LhsRhs => (lhs, rhs, candidate.lhs(), candidate.rhs()),
                FusionContractOrientation::RhsLhs => (rhs, lhs, candidate.rhs(), candidate.lhs()),
            };
            let (left, right) = (core_lhs.oriented_homspace(), core_rhs.oriented_homspace());
            if dst_nout != left.nout()
                || !core_lhs_axes.iter().copied().eq(left.nout()..left.rank())
                || !core_rhs_axes.iter().copied().eq(0..right.nout())
            {
                continue;
            }
            let twisted = fermionic
                && core_rhs_axes
                    .iter()
                    .any(|&axis| right.external_axis_is_dual(axis) == Some(true));
            if twisted && !twist_consumable {
                continue;
            }
            *candidate_seen = true;
            let core_axes = TensorContractSpec::new_with_conjugation(
                core_lhs_axes,
                core_rhs_axes,
                OutputAxisOrder::identity(),
                core_lhs.storage_conjugate(),
                core_rhs.storage_conjugate(),
            );
            if let Some(route) = compile(core_lhs, core_rhs, core_axes, orientation)? {
                return Ok(Some(route));
            }
        }
    }
    Ok(None)
}

/// The canonical core with a uniform fermionic twist as per-job alpha. A
/// twist that varies within one RHS coupled-sector matrix has no per-job
/// alpha: the core declines (`Ok(None)`), so the `DynamicTree` artifact
/// applies it per block.
fn try_compile_scaled_storage_contract_plan<M, R>(
    rule: &R,
    validated: &ValidatedCoreContract<'_, R>,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    rhs_orientation: FusionTreePairOrientation,
) -> Result<Option<FusionBlockContractPlan<M::Scalar>>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let Some(regions) = rhs
        .structure()
        .coupled_sector_regions(rhs.nout())
        .map_err(OperationError::from_core_preserving_context)?
    else {
        return Ok(None);
    };
    let mut alpha_by_coupled = Vec::with_capacity(regions.len());
    for region in regions.iter() {
        let row_trees = match rhs_orientation {
            FusionTreePairOrientation::Direct => region.row_trees(),
            FusionTreePairOrientation::Adjoint => region.col_trees(),
        };
        let Some(alpha) = M::core_alpha(
            rule,
            validated.rhs_homspace(),
            validated.rhs_contracting_axes(),
            row_trees.iter().map(|extent| extent.tree()),
        )?
        else {
            return Ok(None);
        };
        alpha_by_coupled.push((region.coupled(), alpha));
    }
    Ok(try_compile_scaled_canonical_core_plan(
        validated,
        dst,
        lhs,
        rhs,
        &alpha_by_coupled,
    )?)
}

/// Compiles the core plan of one zero-copy candidate from parent spaces and
/// lazy operand orientation: the canonical coupled-region plan, and, for an
/// executor with [`ExecCaps::IRREGULAR_CORE`] (`irregular`), the mode's
/// packed plan of a non-canonical tiling (#1517) when the canonical one
/// declines. A twisted [`ContractKind::Contract`] candidate takes only the
/// canonical scaled plan; a [`ContractKind::Compose`] is never twisted.
#[allow(clippy::type_complexity)]
pub(crate) fn try_compile_oriented_storage_contract_plan<M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    kind: ContractKind,
    irregular: bool,
) -> Result<Option<Arc<FusionBlockContractPlan<M::Scalar>>>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let PlanTarget {
        rule,
        space: dst,
        authority,
    } = target;
    let preflight = CoreContractPreflight::compile_oriented(
        rule,
        dst.homspace(),
        lhs.oriented_homspace(),
        rhs.oriented_homspace(),
        axes,
    )?;
    let Some(validated) = preflight.validate_core_geometry_in::<M>()? else {
        return Ok(None);
    };
    let twisted = kind == ContractKind::Contract
        && validated_rhs_contract_requires_twist::<M, R>(&validated, authority)?;
    let plan = if twisted {
        try_compile_scaled_storage_contract_plan::<M, R>(
            rule,
            &validated,
            dst,
            lhs.storage_space(),
            rhs.storage_space(),
            rhs.orientation(),
        )?
    } else {
        match try_compile_oriented_canonical_core_plan(
            &validated,
            dst,
            lhs.storage_space(),
            rhs.storage_space(),
        )? {
            Some(plan) => Some(plan),
            // Only a non-canonical tiling reaches the irregular core: a
            // canonical parent-owned region never prepares its projection.
            None if irregular => Some(M::irregular_core(validated, authority, dst, lhs, rhs)?),
            None => None,
        }
    };
    Ok(plan.map(Arc::new))
}

/// The route capabilities of one contraction executor, which the planner
/// reads at compile time (design §2.3): a route an executor cannot run is
/// never planned for it.
#[doc(hidden)]
pub trait ExecCaps: exec_caps::Sealed {
    /// The executor runs a core plan that is not fully direct: the packed
    /// pack/GEMM/scatter replay of a non-canonical coupled-sector tiling
    /// (#1517).
    const IRREGULAR_CORE: bool;
}

mod exec_caps {
    pub trait Sealed {}
    impl Sealed for super::HostEagerExecutor {}
    impl Sealed for super::DirectCoreExecutor {}
}

/// The Host eager executor: it packs a non-canonical tiling around the core
/// GEMMs.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct HostEagerExecutor;

impl ExecCaps for HostEagerExecutor {
    const IRREGULAR_CORE: bool = true;
}

/// An executor with the direct core GEMMs only: the device executors and the
/// stacked Host replay of `ContractPlan`.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct DirectCoreExecutor;

impl ExecCaps for DirectCoreExecutor {
    const IRREGULAR_CORE: bool = false;
}

/// [`try_compile_oriented_storage_contract_plan`] over every zero-copy
/// TensorKit candidate ([`try_zero_copy_contract_candidates`]); a uniform
/// twist is consumed as per-job alpha, a nonuniform one declines, and so does
/// a candidate whose plan is not fully direct unless `irregular` (a storage
/// replay has no irregular pack/scatter; the next candidate, or
/// `DynamicTree`, applies — TensorKit's `mul!` takes every candidate, so
/// declining never rejects).
pub(crate) fn try_compile_oriented_storage_contract_candidate_plan<M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    candidate_seen: &mut bool,
    irregular: bool,
) -> Result<Option<ContractRoute<M::Scalar>>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    try_zero_copy_contract_candidates(
        || M::contract_twist_possible(target.rule, target.authority),
        target.space.nout(),
        lhs,
        rhs,
        axes,
        true,
        candidate_seen,
        |core_lhs, core_rhs, core_axes, orientation| {
            let plan = try_compile_oriented_storage_contract_plan::<M, R>(
                target,
                core_lhs,
                core_rhs,
                core_axes,
                ContractKind::Contract,
                irregular,
            )?;
            Ok(plan
                .filter(|plan| irregular || plan.is_fully_direct())
                .map(|plan| ContractRoute::Core {
                    plan,
                    swapped: orientation == FusionContractOrientation::RhsLhs,
                }))
        },
    )
}

/// Compiles the coupled block plan of the core operands a
/// [`FusionContractPlan`] derived from an already validated contraction.
///
/// Why no second [`CoreContractPreflight`]: the plan's source transforms
/// place each operand's open and contracted legs in core order and its core
/// axes are the canonical `(open, contracted) x (contracted, open)` pairing,
/// so the core source and output form hold by construction. The core
/// destination is either derived from these two operands with those core
/// axes or, for an identity output transform, the destination whose
/// contracted HomSpace the plan compile already checked. The contracted leg
/// pairs are the checked pairs in candidate order, and every derived space
/// is bound to `rule`. Re-running the preflight would only repeat those
/// checks on every warm call.
pub(crate) fn compile_derived_core_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    core_axes: TensorContractSpec<'_>,
) -> Result<Arc<FusionBlockContractPlan<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    // Debug builds re-check the construction argument above, so an internal
    // inconsistency (a wrong space-cache entry, diverging HomSpace
    // derivations) still fails loudly in tests instead of reaching the block
    // plan.
    #[cfg(debug_assertions)]
    super::fusion_block::debug_assert_core_geometry(rule, dst, lhs, rhs, core_axes);
    #[cfg(not(debug_assertions))]
    let _ = core_axes;
    compile_fusion_block_contract_plan_core_geometry(rule, dst, lhs, rhs).map(Arc::new)
}

/// True when the fermionic supertrace twist can be nontrivial: such
/// contractions take the dynamic route, where the twist is applied during
/// rhs materialization; the core direct-GEMM route stays
/// coefficient-free (TensorKit mul! parity).
pub(crate) fn rhs_contract_requires_twist<R>(
    rule: &R,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    rhs_contract_homspace_requires_twist(rule, rhs.homspace(), axes)
}

fn validated_rhs_contract_requires_twist<M, R>(
    validated: &ValidatedCoreContract<'_, R>,
    authority: M::SpaceAuthority<'_>,
) -> Result<bool, M::Error>
where
    M: PlanningAlgebra<R>,
{
    if !M::contract_twist_possible(validated.rule(), authority)? {
        return Ok(false);
    }
    Ok(validated
        .rhs_contracting_axes()
        .iter()
        .copied()
        .any(|axis| {
            validated
                .rhs_homspace()
                .external_axis_is_dual(axis)
                .expect("core preflight validated every rhs contraction axis")
        }))
}

pub(crate) fn rhs_contract_homspace_requires_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    axes: TensorContractSpec<'_>,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    rhs_contract_axes_require_twist(rule, rhs, axes.rhs_contracting_axes())
}

/// The permute a TensorKit `copyC` (`blas_contract!`,
/// `tensoroperations.jl:436-446` @cfaa073) applies to its temporary: the temporary is
/// `first·second` in `orientation` with its own default output (`first`'s
/// open axes, then `second`'s), and this operation moves those axes into the
/// caller's `output_axes`, split after `codomain_rank`.
#[doc(hidden)]
pub fn copy_c_output_transform(
    orientation: FusionContractOrientation,
    lhs_rank: usize,
    rhs_rank: usize,
    lhs_contract: &[usize],
    rhs_contract: &[usize],
    output_axes: &[usize],
    codomain_rank: usize,
) -> crate::TreeTransformOperation {
    let lhs_open = lhs_rank - lhs_contract.len();
    let rhs_open = rhs_rank - rhs_contract.len();
    let (lhs_offset, rhs_offset) = match orientation {
        FusionContractOrientation::LhsRhs => (0, 0),
        FusionContractOrientation::RhsLhs => (rhs_open, lhs_open),
    };
    let position = |axis: usize| {
        if axis < lhs_open {
            axis + lhs_offset
        } else {
            axis - rhs_offset
        }
    };
    let (codomain, domain) = output_axes.split_at(codomain_rank);
    crate::TreeTransformOperation::permute(
        codomain.iter().copied().map(position),
        domain.iter().copied().map(position),
    )
}

/// Whether a contraction into `dst`, the destination of the requested
/// `output_axes`, takes TensorKit `blas_contract!`'s `copyC` shape, and in
/// which operand order: the returned order's default-output contraction runs
/// a zero-copy candidate (see [`try_zero_copy_contract_candidates`]) into a
/// temporary, and one permute into `dst` then gives `output_axes`. `LhsRhs`
/// contracts `lhs·rhs`, `RhsLhs` contracts `rhs·lhs`. A lazy adjoint stays a
/// GEMM flag there (`has_shared_permute(::AdjointTensorMap)`).
///
/// The choice is TensorKit `contract!`'s (tensoroperations.jl L318-357)
/// under `_contract_memcost` (L378): `copyC` costs `dim(C)`, the required
/// length of `dst` (the temporary has the same blocks), and it is taken only
/// when that is no more than the cheapest DynamicTree candidate for the
/// requested order ([`min_dynamic_tree_materialized_elements`], the real
/// selector's scorer). Ties go to the earlier candidate in TensorKit's
/// order: a B·A `copyC` (m3/m4) loses a tie to an A·B DynamicTree candidate
/// (m1/m2) and wins one against a B·A candidate, whose equal cost is then
/// the same output-only copy. A large `C` from small operands therefore
/// keeps the one-call route, which copies the operands instead.
///
/// `None` also when `output_axes` is not a permutation of the open axes,
/// when it is already zero-copy, when no candidate is, or when an admission
/// is not Complete: the one-call route then applies and keeps its errors.
/// Candidate checks are axis and dual-flag comparisons, with the fermionic
/// twist declined as on the Host Core route.
///
/// The requested-order question is optionally answered (`requested_zero_copy`: the `candidate_seen`
/// of the planner's core walk over the requested order, which therefore runs
/// once) and the twist rule of the core the temporary will run.
///
/// `twist_possible` answers whether the rule can twist at all, `twist`
/// whether given core-right contracted legs carry the twist (both read
/// where the walk and the scorer ask, as their rule reads were).
///
/// `twist_consumable`: the planner's core carries a twist uniform per RHS
/// coupled sector as per-job alpha (#1858), so a twisted default order is
/// still a zero-copy temporary for it (a nonuniform one makes the
/// temporary's core decline, and `DynamicTree` applies). TensorKit's
/// `blas_contract!` instead copies an operand to twist it
/// (`tensoroperations.jl:399-429` @cfaa073), which `false` keeps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_c_order<E>(
    twist_possible: impl Fn() -> Result<bool, E> + Copy,
    twist: impl Fn(&FusionTreeHomSpace, &[usize]) -> Result<bool, E> + Copy,
    canonical: &dyn Fn(&DynamicFusionMapSpace) -> Result<bool, E>,
    dst: &DynamicFusionMapSpace,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    requested_zero_copy: Option<bool>,
    twist_consumable: bool,
) -> Option<FusionContractOrientation>
where
    E: From<OperationError>,
{
    let lhs_open = lhs.storage_space().rank().checked_sub(lhs_axes.len())?;
    let rhs_open = rhs.storage_space().rank().checked_sub(rhs_axes.len())?;
    let open = lhs_open + rhs_open;
    if output_axes.len() != open || !(0..open).all(|axis| output_axes.contains(&axis)) {
        return None;
    }
    let zero_copy = |lhs, rhs, lhs_axes, rhs_axes, output, dst_nout| {
        matches!(
            try_zero_copy_contract_candidates(
                twist_possible,
                dst_nout,
                lhs,
                rhs,
                TensorContractSpec::new(lhs_axes, rhs_axes, output),
                twist_consumable,
                &mut false,
                |_, _, _, _| Ok(Some(())),
            ),
            Ok(Some(()))
        )
    };
    let requested = OutputAxisOrder::from_axes(output_axes);
    // The requested split is `dst`'s; only the temporaries below use the
    // default one.
    if requested_zero_copy
        .unwrap_or_else(|| zero_copy(lhs, rhs, lhs_axes, rhs_axes, requested, dst.nout()))
    {
        return None;
    }
    let identity = OutputAxisOrder::identity();
    let order = if zero_copy(lhs, rhs, lhs_axes, rhs_axes, identity, lhs_open) {
        FusionContractOrientation::LhsRhs
    } else if zero_copy(rhs, lhs, rhs_axes, lhs_axes, identity, rhs_open) {
        FusionContractOrientation::RhsLhs
    } else {
        return None;
    };
    let output_len = dst.required_len().ok()?;
    let complete = [
        dst.admission(),
        lhs.storage_space().admission(),
        rhs.storage_space().admission(),
    ]
    .into_iter()
    .all(|admission| matches!(admission, FusionSpaceAdmission::Complete(_)));
    if !complete {
        return None;
    }
    let dynamic = |orientation| {
        min_dynamic_tree_materialized_elements(
            twist,
            canonical,
            dst,
            lhs,
            rhs,
            TensorContractSpec::new(lhs_axes, rhs_axes, requested),
            &[orientation],
        )
        .ok()
        .flatten()
    };
    let scored = || -> Option<bool> {
        let (a_b, b_a) = (
            dynamic(FusionContractOrientation::LhsRhs)?,
            dynamic(FusionContractOrientation::RhsLhs)?,
        );
        Some(match order {
            FusionContractOrientation::LhsRhs => output_len <= a_b.min(b_a),
            FusionContractOrientation::RhsLhs => output_len < a_b && output_len <= b_a,
        })
    };
    // Why not always score: a DynamicTree candidate that copies nothing is a
    // zero-copy candidate for `output_axes` (the same axis, borrowability and
    // twist conditions), which returned `None` above, so every candidate
    // copies an operand (at least `operand_len`) or the output
    // (`output_len`). Under `RhsLhs` no A·B candidate borrows both operands
    // either, or the `LhsRhs` walk would have found it, so every A·B
    // candidate costs at least `operand_len`, plus `output_len` unless
    // `output_axes` is the identity (its only output-free order). Only an
    // `RhsLhs` tie of the two lengths with an identity (or empty) output
    // needs scores, and only the A·B ones: the B·A ones are then at least
    // `output_len`.
    let operand_len = lhs
        .storage_space()
        .required_len()
        .ok()?
        .min(rhs.storage_space().required_len().ok()?);
    let identity_output = output_axes.iter().copied().eq(0..open) && dst.nout() == lhs_open;
    let copy_c = match (order, output_len.cmp(&operand_len)) {
        (FusionContractOrientation::LhsRhs, Ordering::Less | Ordering::Equal)
        | (FusionContractOrientation::RhsLhs, Ordering::Less) => Some(true),
        (FusionContractOrientation::RhsLhs, Ordering::Equal)
            if output_len > 0 && !identity_output =>
        {
            Some(true)
        }
        (FusionContractOrientation::RhsLhs, Ordering::Equal) => {
            Some(dynamic(FusionContractOrientation::LhsRhs).is_some_and(|a_b| output_len < a_b))
        }
        _ => None,
    };
    debug_assert!(
        copy_c.is_none() || copy_c == Some(scored() == Some(true)),
        "copyC shortcut disagrees with full candidate scoring"
    );
    copy_c.or_else(scored).unwrap_or(false).then_some(order)
}

/// Whether the core-right contracted legs `rhs_contracting_axes` of
/// `homspace` carry TensorKit's fermionic supertrace twist in mode `M`: the
/// rule can twist and one of them is dual (`blas_contract!`,
/// `tensoroperations.jl:398-410` @cfaa073).
pub(crate) fn contract_axes_require_twist<M, R>(
    rule: &R,
    authority: M::SpaceAuthority<'_>,
    homspace: &FusionTreeHomSpace,
    rhs_contracting_axes: &[usize],
) -> Result<bool, M::Error>
where
    M: PlanningAlgebra<R>,
{
    if !M::contract_twist_possible(rule, authority)? {
        return Ok(false);
    }
    for &axis in rhs_contracting_axes {
        if external_axis_is_dual(homspace, axis)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn rhs_contract_axes_require_twist<R>(
    rule: &R,
    rhs: &FusionTreeHomSpace,
    rhs_contracting_axes: &[usize],
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    if rule.braiding_style() != symtn_core::BraidingStyleKind::Fermionic {
        return Ok(false);
    }
    for &axis in rhs_contracting_axes {
        if external_axis_is_dual(rhs, axis)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtn_core::{
        FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, ProductFusionRule,
        SU2FusionRule, SU2Irrep, SectorId, SectorLeg, U1FusionRule, U1Irrep,
    };

    fn single_sector_matrix_space<R>(
        rule: &R,
        sector: SectorId,
        codomain_dual: bool,
        domain_dual: bool,
    ) -> DynamicFusionMapSpace
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], codomain_dual)]),
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], domain_dual)]),
        );
        DynamicFusionMapSpace::from_final_homspace(rule, hom).unwrap()
    }

    #[test]
    fn rhs_twist_requirement_uses_external_domain_dual_and_product_parity() {
        let fermion = FermionParityFusionRule;
        let odd = SectorId::new(1);
        let cases = [
            (false, false, 0, false),
            (true, false, 0, true),
            (false, false, 1, true),
            (false, true, 1, false),
        ];
        for (codomain_dual, domain_dual, rhs_axis, expected) in cases {
            let rhs = single_sector_matrix_space(&fermion, odd, codomain_dual, domain_dual);
            let rhs_axes = [rhs_axis];
            let axes = TensorContractSpec::with_default_output_order(&[0], &rhs_axes);
            // What: codomain uses its stored dual flag, while domain external
            // duality is the inverse of its stored flag.
            assert_eq!(
                rhs_contract_requires_twist(&fermion, &rhs, axes).unwrap(),
                expected
            );
        }

        let fp_u1 =
            ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
        let odd_charge = fp_u1.encode_component_ids(odd, U1Irrep::new(0).sector_id());
        let product = ProductFusionRule::<_, _>::try_new(fp_u1, SU2FusionRule).unwrap();
        let odd_product =
            product.encode_component_ids(odd_charge, SU2Irrep::from_twice_spin(0).sector_id());
        let rhs = single_sector_matrix_space(&product, odd_product, true, false);
        // What: a bosonic U(1) x SU(2) component does not erase the odd fZ2
        // twist on an externally dual product-sector axis.
        assert!(rhs_contract_requires_twist(
            &product,
            &rhs,
            TensorContractSpec::with_default_output_order(&[0], &[0]),
        )
        .unwrap());
    }

    #[test]
    fn storage_core_route_derives_geometry_once() {
        let rule = std::sync::Arc::new(U1FusionRule);
        let zero = U1Irrep::new(0).sector_id();
        let space = |space| {
            crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(space, rule.clone()).unwrap()
        };
        let lhs = space(single_sector_matrix_space(&*rule, zero, false, false));
        let rhs = space(single_sector_matrix_space(&*rule, zero, false, false));
        let dst = space(single_sector_matrix_space(&*rule, zero, false, false));
        let core_rec = crate::probe::record();

        let route = crate::try_compile_storage_contract_core_route::<HostEagerExecutor, _>(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
        )
        .unwrap();

        // What: the core rung's one candidate derives each geometry once.
        assert!(route.hit().is_some_and(|core| !core.is_dynamic_tree()));
        assert_eq!(
            (
                core_rec.get(crate::probe::Event::ContractAxisPlanCompile),
                core_rec.get(crate::probe::Event::CoreHomspaceDerivation),
            ),
            (1, 1)
        );
    }

    mod warm_compile_counts {
        use std::sync::Arc;

        use symtn_core::{
            FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep,
        };
        use symtn_operations::{OutputAxisOrder, TensorContractSpec};

        use crate::contract::{BoundDynamicFusionMapSpace, FusionOperand};
        use crate::probe::{self, Event};
        use crate::{ContractDestinationInit, RuleIdentity, TensorContractFusionExecutionContext};

        type Space = BoundDynamicFusionMapSpace<U1FusionRule>;

        fn space(provider: &Arc<U1FusionRule>, codomain: usize, domain: usize) -> Space {
            let leg = || {
                SectorLeg::new(
                    [
                        (U1Irrep::new(-1).sector_id(), 2),
                        (U1Irrep::new(0).sector_id(), 3),
                        (U1Irrep::new(1).sector_id(), 1),
                    ],
                    false,
                )
            };
            Space::from_final_homspace_multiplicity_free_checked(
                Arc::clone(provider),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new((0..codomain).map(|_| leg())),
                    FusionProductSpace::new((0..domain).map(|_| leg())),
                ),
            )
            .unwrap()
        }

        /// What one contraction compile did: core preflights, core
        /// destination checks, and permuted or contracted HomSpaces built.
        #[derive(Debug, Eq, PartialEq)]
        struct ContractCompileCounts {
            preflights: usize,
            core_destination_checks: usize,
            derived_homspace_builds: usize,
        }

        /// Counts of the third of three identical calls.
        fn warm_counts(mut call: impl FnMut()) -> ContractCompileCounts {
            call();
            call();
            let rec = probe::record();
            call();
            ContractCompileCounts {
                preflights: rec.get(Event::CoreContractPreflight),
                core_destination_checks: rec.get(Event::CoreHomspaceDerivation),
                derived_homspace_builds: rec.get(Event::DerivedHomspaceBuild),
            }
        }

        #[test]
        fn warm_contract_compile_runs_one_preflight_and_rebuilds_no_homspace() {
            // Isolated: the warm counts rely on each transformed source's
            // destination memo staying resident across the three calls, and a
            // concurrent test's cache clear would drop it mid-measurement.
            if crate::test_support::run_isolated_or_return(
                "SYMTN_WARM_COMPILE_COUNTS_ISOLATED",
                "contract::resolution::tests::warm_compile_counts::warm_contract_compile_runs_one_preflight_and_rebuilds_no_homspace",
            ) {
                return;
            }
            let provider = Arc::new(U1FusionRule);
            let lhs = space(&provider, 2, 1);
            let matrix = space(&provider, 1, 1);
            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
            let matrix_data = vec![0.5; matrix.space().required_len().unwrap()];
            for output in [[0usize, 1, 2], [1, 0, 2]] {
                let axes = TensorContractSpec::new(&[0], &[1], OutputAxisOrder::from_axes(&output));
                let dst = Space::contracted_multiplicity_free_ordered(
                    &lhs,
                    &matrix,
                    &[0],
                    &[1],
                    OutputAxisOrder::from_axes(&output),
                )
                .unwrap();
                let mut dst_data = vec![0.0; dst.space().required_len().unwrap()];
                let host = warm_counts(|| {
                    context
                        .tensorcontract_prelowered_into(
                            &dst,
                            &mut dst_data,
                            crate::FusionOperand::direct(lhs.space()),
                            &lhs_data,
                            crate::FusionOperand::direct(matrix.space()),
                            &matrix_data,
                            axes,
                            1.0,
                            ContractDestinationInit::Axpby(0.0),
                        )
                        .unwrap();
                });
                let mut core_dst = false;
                let storage = warm_counts(|| {
                    let resolution = context
                        .compile_storage_contract_resolution(
                            &dst,
                            FusionOperand::direct(lhs.space()),
                            FusionOperand::direct(matrix.space()),
                            axes,
                        )
                        .unwrap();
                    assert!(resolution.is_dynamic_tree());
                    core_dst = resolution.direct_destination_inactive_blocks().is_none();
                });
                // What: no core preflight (Host and device share the planner,
                // which rejects every candidate on axes alone), no core
                // destination check
                // for a non-core request, and fewer HomSpaces than TensorKit
                // forms: each transformed source's permuted HomSpace comes
                // from its source's destination memo, so only the core
                // destination is built, when the output transform is not
                // the identity. The fixture covers both output forms.
                assert_eq!(core_dst, output != [0, 1, 2]);
                let core_dst = usize::from(core_dst);
                let expected = ContractCompileCounts {
                    preflights: 0,
                    core_destination_checks: 0,
                    derived_homspace_builds: core_dst,
                };
                assert_eq!(host, expected, "host output order {output:?}");
                assert_eq!(storage, expected, "storage output order {output:?}");
            }
        }

        #[test]
        fn warm_core_compile_checks_the_destination_once_without_building_it() {
            let provider = Arc::new(U1FusionRule);
            let lhs = space(&provider, 2, 2);
            let square = space(&provider, 2, 2);
            let dst = Space::contracted_multiplicity_free_ordered(
                &lhs,
                &square,
                &[2, 3],
                &[0, 1],
                OutputAxisOrder::identity(),
            )
            .unwrap();
            let axes = TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]);
            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
            let square_data = vec![0.5; square.space().required_len().unwrap()];
            let mut dst_data = vec![0.0; dst.space().required_len().unwrap()];
            let counts = warm_counts(|| {
                context
                    .tensorcontract_prelowered_into(
                        &dst,
                        &mut dst_data,
                        crate::FusionOperand::direct(lhs.space()),
                        &lhs_data,
                        crate::FusionOperand::direct(square.space()),
                        &square_data,
                        axes,
                        1.0,
                        ContractDestinationInit::Zeroed,
                    )
                    .unwrap();
            });
            assert_eq!(
                counts,
                ContractCompileCounts {
                    preflights: 1,
                    core_destination_checks: 1,
                    derived_homspace_builds: 0,
                }
            );
        }
    }
}
