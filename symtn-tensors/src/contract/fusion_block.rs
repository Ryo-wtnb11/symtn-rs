use std::collections::HashSet;
use std::mem::MaybeUninit;
use symtn_dense::Engine;

use rustc_hash::FxHashMap;
use std::sync::Arc;

use symtn_core::{
    BlockKey, CategoricalScalar, CoupledMatrixSide, CoupledTreePlacement, FusionRule,
    FusionTreeHomSpace, FusionTreeKey, FusionTreePairOrientation, MultiplicityFreeRigidSymbols,
    OrientedFusionTreeHomSpace, SectorId,
};
#[cfg(test)]
use symtn_core::{HostReadableStorage, HostWritableStorage};

use crate::strided::{
    column_major_strides_isize, column_major_strides_usize, element_count, offset_to_isize,
    strides_to_isize,
};
use crate::{DenseBlockScalar, OperationError, RecouplingCoefficientAction};
#[cfg(test)]
use symtn_operations::HostKernelAdapter;
use symtn_operations::TensorContractSpec;

#[cfg(test)]
pub(crate) use symtn_operations::fusion_replay::StorageGemm;
use symtn_operations::fusion_replay::{
    direct_group_matrix_offset_generic, fusion_scale_block_layouts_excluding,
    FusionBlockContractGroupPlan, FusionBlockMatrixGroup, FusionStridedBlockLayout,
    FusionSubblockMatrixLayout, MatrixOp, Rank2GemmBatchJob,
};
pub(crate) use symtn_operations::fusion_replay::{
    FusionBlockContractPlan, FusionBlockContractWorkspace, Rank2Gemm,
};

/// Validate category identity before a contraction route reads sectors or
/// symbols through the supplied rule.
///
/// Why-not validate during replay: a compiled plan is already tied to
/// validated spaces, so replay checks would charge every warm execution for a
/// compile-time invariant.
pub(crate) fn validate_fusion_contract_rule<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)
}

use super::backend::TensorContractBackend;
use super::dynamic_space::{DynamicFusionMapSpace, FusionOperandLayout};
// Used by the debug-build core geometry assertion and the test-only compile.
#[cfg(any(test, debug_assertions))]
use super::fusion::reject_fusion_contract_conjugation;
use super::structure::TensorContractAxisPlan;

mod compile;
#[cfg(test)]
mod tests;
pub(crate) use compile::*;

pub(super) struct CoreContractPreflight<'a, R> {
    rule: &'a R,
    dst_homspace: &'a FusionTreeHomSpace,
    lhs_homspace: OrientedFusionTreeHomSpace<'a>,
    rhs_homspace: OrientedFusionTreeHomSpace<'a>,
    axis_plan: TensorContractAxisPlan,
}

pub(crate) struct ValidatedCoreContract<'a, R> {
    preflight: CoreContractPreflight<'a, R>,
}

impl<'a, R> CoreContractPreflight<'a, R> {
    pub(super) fn compile_oriented(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: OrientedFusionTreeHomSpace<'a>,
        rhs_homspace: OrientedFusionTreeHomSpace<'a>,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::CoreContractPreflight);
        let axis_plan = TensorContractAxisPlan::compile(
            lhs_homspace.rank(),
            rhs_homspace.rank(),
            dst_homspace.rank(),
            axes,
        )?;
        Ok(Self::with_axis_plan(
            rule,
            dst_homspace,
            lhs_homspace,
            rhs_homspace,
            axis_plan,
        ))
    }

    /// [`Self::compile_oriented`] without the test compile counts, for the
    /// debug re-check of work production already did.
    #[cfg(debug_assertions)]
    fn compile_oriented_unobserved(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: OrientedFusionTreeHomSpace<'a>,
        rhs_homspace: OrientedFusionTreeHomSpace<'a>,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        let axis_plan = TensorContractAxisPlan::compile_unobserved(
            lhs_homspace.rank(),
            rhs_homspace.rank(),
            dst_homspace.rank(),
            axes,
        )?;
        Ok(Self::with_axis_plan(
            rule,
            dst_homspace,
            lhs_homspace,
            rhs_homspace,
            axis_plan,
        ))
    }

    fn with_axis_plan(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: OrientedFusionTreeHomSpace<'a>,
        rhs_homspace: OrientedFusionTreeHomSpace<'a>,
        axis_plan: TensorContractAxisPlan,
    ) -> Self {
        Self {
            rule,
            dst_homspace,
            lhs_homspace,
            rhs_homspace,
            axis_plan,
        }
    }

    /// The core form check (geometry only), then the mode's destination
    /// check.
    pub(super) fn validate_core_geometry_in<M>(
        self,
    ) -> Result<Option<ValidatedCoreContract<'a, R>>, M::Error>
    where
        M: crate::mode::PlanningAlgebra<R>,
    {
        self.validate_core_geometry_with(M::destination_matches)
    }

    #[allow(clippy::type_complexity)]
    fn validate_core_geometry_with<E>(
        self,
        destination_matches: impl FnOnce(
            &R,
            OrientedFusionTreeHomSpace<'_>,
            OrientedFusionTreeHomSpace<'_>,
            &[usize],
            &[usize],
            &[usize],
            &FusionTreeHomSpace,
        ) -> Result<bool, E>,
    ) -> Result<Option<ValidatedCoreContract<'a, R>>, E>
    where
        E: From<OperationError>,
    {
        if !is_core_form_source(
            self.lhs_homspace.rank(),
            self.lhs_homspace.nout(),
            self.rhs_homspace.nout(),
            &self.axis_plan,
        ) || !is_core_form_output(
            self.dst_homspace.codomain().len(),
            self.lhs_homspace.nout(),
            self.rhs_homspace.rank(),
            self.rhs_homspace.nout(),
            &self.axis_plan,
        ) {
            return Ok(None);
        }
        if !destination_matches(
            self.rule,
            self.lhs_homspace,
            self.rhs_homspace,
            self.axis_plan.lhs_contracting_axes.as_slice(),
            self.axis_plan.rhs_contracting_axes.as_slice(),
            self.axis_plan.output_axes.as_slice(),
            self.dst_homspace,
        )? {
            return Err(OperationError::StructureMismatch { tensor: "dst" }.into());
        }
        Ok(Some(ValidatedCoreContract { preflight: self }))
    }
}

/// The direct-homspace preflight of the test-only
/// `compile_fusion_block_contract_plan`; production compiles go through
/// [`CoreContractPreflight::compile_oriented`].
#[cfg(test)]
impl<'a, R> CoreContractPreflight<'a, R>
where
    R: FusionRule,
{
    pub(super) fn compile_homspaces(
        rule: &'a R,
        dst_homspace: &'a FusionTreeHomSpace,
        lhs_homspace: &'a FusionTreeHomSpace,
        rhs_homspace: &'a FusionTreeHomSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError> {
        Self::compile_oriented(
            rule,
            dst_homspace,
            OrientedFusionTreeHomSpace::new(lhs_homspace, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs_homspace, FusionTreePairOrientation::Direct),
            axes,
        )
    }

    pub(super) fn validate_core_geometry(
        self,
    ) -> Result<Option<ValidatedCoreContract<'a, R>>, OperationError> {
        self.validate_core_geometry_with(core_homspace_matches)
    }

    pub(super) fn require_core_geometry(
        self,
    ) -> Result<ValidatedCoreContract<'a, R>, OperationError> {
        require_core_form(self.validate_core_geometry()?)
    }
}

#[cfg(any(test, debug_assertions))]
fn require_core_form<T>(validated: Option<T>) -> Result<T, OperationError> {
    validated.ok_or(OperationError::UnsupportedTensorContractScope {
        message: "core fusion-block contraction requires core source and output axes",
    })
}

impl<'a, R> ValidatedCoreContract<'a, R> {
    pub(crate) fn rule(&self) -> &'a R {
        self.preflight.rule
    }

    pub(super) fn rhs_homspace(&self) -> OrientedFusionTreeHomSpace<'a> {
        self.preflight.rhs_homspace
    }

    pub(super) fn rhs_contracting_axes(&self) -> &[usize] {
        &self.preflight.axis_plan.rhs_contracting_axes
    }
}

/// The core destination check: `dst` must be the contracted HomSpace of
/// the two oriented operands. Decided by leg comparison, without building
/// that HomSpace, with its errors in its order.
pub(crate) fn core_homspace_matches<R>(
    rule: &R,
    lhs: OrientedFusionTreeHomSpace<'_>,
    rhs: OrientedFusionTreeHomSpace<'_>,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst: &FusionTreeHomSpace,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::CoreHomspaceDerivation);
    contracted_homspace_matches(
        rule,
        lhs,
        rhs,
        lhs_contracting_axes,
        rhs_contracting_axes,
        output_axes,
        dst,
    )
}

/// [`core_homspace_matches`] without the test check count.
fn contracted_homspace_matches<R>(
    rule: &R,
    lhs: OrientedFusionTreeHomSpace<'_>,
    rhs: OrientedFusionTreeHomSpace<'_>,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst: &FusionTreeHomSpace,
) -> Result<bool, OperationError>
where
    R: FusionRule,
{
    OrientedFusionTreeHomSpace::tensorcontract_homspace_matches(
        rule,
        lhs,
        rhs,
        lhs_contracting_axes,
        rhs_contracting_axes,
        output_axes,
        dst.codomain().len(),
        dst,
    )
    .map_err(OperationError::from_core_preserving_context)
}

/// Asserts what [`super::resolution::compile_derived_core_plan`] relies on:
/// the full core preflight and destination check accept the plan-derived
/// operands. Runs the uncounted compile and check, so tests count the
/// production work only.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_core_geometry<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    core_axes: TensorContractSpec<'_>,
) where
    R: FusionRule,
{
    let checked = reject_fusion_contract_conjugation(core_axes).and_then(|()| {
        validate_fusion_contract_rule(rule, dst, lhs, rhs)?;
        let preflight = CoreContractPreflight::compile_oriented_unobserved(
            rule,
            dst.homspace(),
            OrientedFusionTreeHomSpace::new(lhs.homspace(), FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs.homspace(), FusionTreePairOrientation::Direct),
            core_axes,
        )?;
        require_core_form(preflight.validate_core_geometry_with(contracted_homspace_matches)?)
            .map(|_| ())
    });
    debug_assert!(
        checked.is_ok(),
        "plan-derived core operands failed the core preflight: {checked:?}"
    );
}

/// Adapts a [`TensorContractBackend`] + workspace pair onto the replay
/// layer's [`Rank2Gemm`] seam.
pub(crate) struct BackendRank2Gemm<'a, B: ?Sized, W, C = f64> {
    pub(crate) backend: &'a mut B,
    pub(crate) workspace: &'a mut W,
    pub(crate) coefficient: std::marker::PhantomData<C>,
}

impl<'a, B: ?Sized, W, C> BackendRank2Gemm<'a, B, W, C> {
    pub(crate) fn new(backend: &'a mut B, workspace: &'a mut W) -> Self {
        Self {
            backend,
            workspace,
            coefficient: std::marker::PhantomData,
        }
    }
}

impl<'a, B, D, C> Rank2Gemm<D> for BackendRank2Gemm<'a, B, B::Workspace, C>
where
    B: TensorContractBackend<D, C> + ?Sized,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + num_traits::One,
{
    fn matmul_rank2(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.backend.matmul_rank2_axpby_into_raw(
            engine,
            self.workspace,
            dst,
            lhs,
            rhs,
            rows,
            contracted,
            cols,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_batch(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.backend.matmul_rank2_batch_axpby_into_raw(
            engine,
            self.workspace,
            dst,
            lhs,
            rhs,
            jobs,
            runs,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_batch_with_ops(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.backend.matmul_rank2_batch_with_ops_axpby_into_raw(
            engine,
            self.workspace,
            dst,
            lhs,
            rhs,
            jobs,
            runs,
            lhs_op,
            rhs_op,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_batch_fresh<'f>(
        &mut self,
        engine: &mut dyn Engine,
        segments: &mut [(usize, &'f mut [MaybeUninit<D>])],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        ops: [MatrixOp; 2],
        alpha: D,
        written: &mut dyn FnMut(usize, &'f mut [D]),
    ) -> Result<(), OperationError>
    where
        D: Copy + num_traits::Zero,
    {
        self.backend.matmul_rank2_batch_fresh_raw(
            engine,
            self.workspace,
            segments,
            lhs,
            rhs,
            jobs,
            ops,
            alpha,
            written,
        )
    }
}

/// A [`Rank2Gemm`] that keeps the trait's zero-then-overwrite fresh
/// default over `G`'s initialized GEMMs: the default of
/// [`TensorContractBackend::matmul_rank2_batch_fresh_raw`].
pub(crate) struct ZeroThenInto<G>(pub(crate) G);

impl<D, G: Rank2Gemm<D>> Rank2Gemm<D> for ZeroThenInto<G> {
    fn matmul_rank2(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.0
            .matmul_rank2(engine, dst, lhs, rhs, rows, contracted, cols, alpha, beta)
    }

    fn matmul_rank2_batch_with_ops(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        self.0.matmul_rank2_batch_with_ops(
            engine, dst, lhs, rhs, jobs, runs, lhs_op, rhs_op, alpha, beta,
        )
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_core_fusion_blocks_into_raw<A, B, R, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, C>,
    R: MultiplicityFreeRigidSymbols<Scalar = C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let plan = compile_fusion_block_contract_plan(rule, dst_space, lhs_space, rhs_space, axes)?;
    tensorcontract_core_fusion_blocks_with_plan_into_raw(
        engine, kernels, backend, workspace, &plan, dst_space, dst_data, lhs_space, lhs_data,
        rhs_space, rhs_data, alpha, beta,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_core_fusion_blocks_with_plan_into_raw<A, B, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    backend: &mut B,
    workspace: &mut B::Workspace,
    plan: &FusionBlockContractPlan<C>,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let mut fusion_workspace = FusionBlockContractWorkspace::<D>::default();
    plan.execute_raw(
        engine,
        kernels,
        &mut BackendRank2Gemm::new(backend, workspace),
        &mut fusion_workspace,
        dst_space.structure(),
        dst_data,
        lhs_space.structure(),
        lhs_data,
        rhs_space.structure(),
        rhs_data,
        alpha,
        beta,
    )
}

fn is_core_form_source(
    lhs_rank: usize,
    lhs_nout: usize,
    rhs_nout: usize,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    axis_plan
        .lhs_contracting_axes
        .iter()
        .copied()
        .eq(lhs_nout..lhs_rank)
        && axis_plan
            .rhs_contracting_axes
            .iter()
            .copied()
            .eq(0..rhs_nout)
}

fn is_core_form_output(
    dst_nout: usize,
    lhs_nout: usize,
    rhs_rank: usize,
    rhs_nout: usize,
    axis_plan: &TensorContractAxisPlan,
) -> bool {
    let output_rank = lhs_nout + (rhs_rank - rhs_nout);
    dst_nout == lhs_nout && axis_plan.output_axes.iter().copied().eq(0..output_rank)
}
