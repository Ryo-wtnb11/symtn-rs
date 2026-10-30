use core::ops::{Add, Mul};
use std::sync::Arc;

use num_traits::{One, Zero};
use symtn_core::{
    multiplicity_free_permute_tree_pair_block_indexed, split_fusion_tree_generic_checked_into,
    split_fusion_tree_into, BlockKey, BlockStructure, CheckedFusionAlgebra,
    CheckedFusionSpaceError, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericSymbolError, FusionRule, FusionStyleKind, FusionTreeHomSpace, FusionTreePairKey,
    FusionTreePairOrientation, FusionTreePairRef, FusionTreeRef, MultiplicityFreeRigidSymbols,
    OrientedFusionTreeHomSpace, PreparedTreePairOperation, SectorLeg, TreeScratch,
    UniqueTreePairScratch,
};

use crate::contract::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, PreparedCheckedGenericDynamicSpace,
};
use crate::lowering::lower_tensortrace_source_adjoint_axes_dyn;
use crate::strided::offset_to_isize;
use crate::tree_transform::{
    lookup_trace, publish_staged_trace, CheckedGenericPlanError, CheckedPendingCoefficients,
    TraceKeyRef, TraceSplit, TransformerMode,
};
use std::any::TypeId;
use symtn_core::RuleIdentity;
use symtn_operations::structure_identity::validate_structure_identity;
use symtn_operations::tensortrace_raw_strided_kernel_add_with_coefficient_trusted;
use symtn_operations::transform_structure::validate_destination_layouts_injective;
use symtn_operations::TensorTraceAxisSpec;
use symtn_operations::{axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted};
use symtn_operations::{try_tensortrace_owned_raw, OperationError, OwnedTraceTerm};
use symtn_operations::{ConjugateValue, RealStructuralCoefficient, RecouplingCoefficientAction};

#[cfg(feature = "cuda")]
mod cuda;
mod pipeline;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use cuda::tensortrace_fusion_structure_into_on_cuda;
use pipeline::{build_trace_terms, trace_geometry, trace_preflight};
use symtn_core::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};

#[derive(Clone, Copy)]
struct OrientedTraceSource<'a> {
    homspace: OrientedFusionTreeHomSpace<'a>,
    structure: &'a Arc<BlockStructure>,
    orientation: FusionTreePairOrientation,
    storage_nout: usize,
    storage_nin: usize,
}

impl<'a> OrientedTraceSource<'a> {
    fn new(
        homspace: &'a FusionTreeHomSpace,
        structure: &'a Arc<BlockStructure>,
        storage_nout: usize,
        storage_nin: usize,
        orientation: FusionTreePairOrientation,
    ) -> Self {
        Self {
            homspace: OrientedFusionTreeHomSpace::new(homspace, orientation),
            structure,
            orientation,
            storage_nout,
            storage_nin,
        }
    }

    fn logical_axis_to_storage(self, axis: usize) -> usize {
        match self.orientation {
            FusionTreePairOrientation::Direct => axis,
            FusionTreePairOrientation::Adjoint => {
                if axis < self.storage_nin {
                    self.storage_nout + axis
                } else {
                    axis - self.storage_nin
                }
            }
        }
    }

    fn source_key(self, index: usize) -> Result<FusionTreePairKey, OperationError> {
        let block = self.structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index,
            });
        };
        Ok(match self.orientation {
            FusionTreePairOrientation::Direct => key.clone(),
            FusionTreePairOrientation::Adjoint => {
                FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
            }
        })
    }

    /// Block `index`'s stored key (no orientation applied).
    fn storage_key(self, index: usize) -> Result<&'a FusionTreePairKey, OperationError> {
        let block = self.structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index,
            });
        };
        Ok(key)
    }

    fn validate_source_keys(self) -> Result<(), OperationError> {
        for index in 0..self.structure.block_count() {
            let block = self.structure.block(index)?;
            if !matches!(block.key(), BlockKey::FusionTree(_)) {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index,
                });
            }
        }
        Ok(())
    }
}

/// A compiled fusion-aware trace whose valid terms retain global source order.
///
/// Valid terms and their source accumulation follow global source-block order.
/// For expert layouts built from an incomplete explicit block structure,
/// errors between distinct external-sector groups likewise follow the first
/// source block encountered globally. On the non-`Unique` grouped path
/// (currently `Simple` fusion), one external-sector group is transformed and
/// lowered atomically at its first member, so a symbol, tree-transform or
/// split error from a later member of that group can precede an earlier
/// member's destination [`OperationError::MissingBlockKey`].
///
/// This uses the same block-level lowering granularity as TensorKit and QSpace;
/// those references do not establish identical malformed-layout error
/// ordering. Why not recover per-source error precedence within a group: doing
/// so would replay the same F/R traversal for every source tree.
///
/// A compile over canonical source and destination contents is resident in
/// the process-global completed-transformer cache: a warm compile is one
/// lookup that rebinds the cached terms to the caller's structures, with no
/// per-group lookup, symbol query, split or descriptor assembly. Otherwise a
/// fusion group whose lowered terms are resident in the process-global
/// transformation-coefficient cache makes no symbol queries and no split;
/// only a whole successful compile publishes what it built.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorTraceFusionStructure<C> {
    core: Arc<TraceCore<C>>,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

/// The structure-free part of a compiled trace, as the completed-transformer
/// cache keeps it: everything but the two structures it is bound to.
#[derive(Debug, PartialEq)]
pub(crate) struct TraceCore<C> {
    dst_rank: usize,
    src_rank: usize,
    output_axes: Vec<usize>,
    trace_lhs_axes: Vec<usize>,
    trace_rhs_axes: Vec<usize>,
    terms: Vec<TensorTraceFusionStructureTerm<C>>,
    descriptor: TensorTraceDescriptor,
}

impl<C> TraceCore<C> {
    /// Bytes this core retains. The term keys hold the tree arenas of the
    /// structures it was compiled over, so their contents are charged too:
    /// a resident core keeps them alive after their own entries leave.
    pub(crate) fn charged_bytes(&self, dst: &BlockStructure, src: &BlockStructure) -> usize {
        let words = self.output_axes.capacity()
            + self.trace_lhs_axes.capacity()
            + self.trace_rhs_axes.capacity();
        core::mem::size_of::<Self>()
            .saturating_add(words.saturating_mul(core::mem::size_of::<usize>()))
            .saturating_add(
                self.terms
                    .capacity()
                    .saturating_mul(core::mem::size_of::<TensorTraceFusionStructureTerm<C>>()),
            )
            .saturating_add(self.descriptor.heap_bytes())
            .saturating_add(dst.charged_retained_bytes())
            .saturating_add(src.charged_retained_bytes())
    }
}

impl<C> TensorTraceFusionStructure<C> {
    #[cfg(test)]
    pub(crate) fn compile_fusion_dyn_raw<R>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        src: &DynamicFusionMapSpace,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone
            + Add<Output = C>
            + Mul<Output = C>
            + Zero
            + RealStructuralCoefficient
            + Send
            + Sync
            + 'static,
    {
        Self::compile_fusion_dyn_raw_with_preflight(rule, dst, src, axes, |_, _| {
            Ok(TraceAdmission::Unchecked)
        })
    }

    /// The compile [`tensortrace_fusion_dyn_owned_checked`] runs, returned
    /// instead of executed, so a device executor replays exactly the Host's
    /// terms, coefficients and strides.
    #[doc(hidden)]
    pub fn compile_fusion_dyn_checked<R>(
        dst: &BoundDynamicFusionMapSpace<R>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + CheckedFusionAlgebra,
        C: Clone
            + Add<Output = C>
            + Mul<Output = C>
            + Zero
            + RealStructuralCoefficient
            + Send
            + Sync
            + 'static,
    {
        Self::compile_fusion_dyn_checked_raw(src.provider(), dst.space(), src.space(), axes)
    }

    fn compile_fusion_dyn_checked_raw<R>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        src: &DynamicFusionMapSpace,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + CheckedFusionAlgebra,
        C: Clone
            + Add<Output = C>
            + Mul<Output = C>
            + Zero
            + RealStructuralCoefficient
            + Send
            + Sync
            + 'static,
    {
        Self::compile_fusion_dyn_raw_with_preflight(
            rule,
            dst,
            src,
            axes,
            |logical_src, axis_plan| {
                trace_geometry::<MultiplicityFreeAdmissionMode, R>(
                    rule,
                    logical_src,
                    axis_plan,
                    dst.nout(),
                )
                .map(TraceAdmission::Checked)
            },
        )
    }

    fn compile_fusion_dyn_raw_with_preflight<R, F>(
        rule: &R,
        dst: &DynamicFusionMapSpace,
        src: &DynamicFusionMapSpace,
        axes: TensorTraceAxisSpec<'_>,
        preflight: F,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone
            + Add<Output = C>
            + Mul<Output = C>
            + Zero
            + RealStructuralCoefficient
            + Send
            + Sync
            + 'static,
        F: FnOnce(
            OrientedFusionTreeHomSpace<'_>,
            &TensorTraceAxisPlan,
        ) -> Result<TraceAdmission, OperationError>,
    {
        dst.validate_rule(rule)?;
        src.validate_rule(rule)?;
        let orientation = if axes.source_conjugate() {
            FusionTreePairOrientation::Adjoint
        } else {
            FusionTreePairOrientation::Direct
        };
        let lowered_axes = lower_tensortrace_source_adjoint_axes_dyn(src.nout(), src.nin(), axes)?;
        let source = OrientedTraceSource::new(
            src.homspace(),
            src.structure(),
            src.nout(),
            src.nin(),
            orientation,
        );
        Self::compile_fusion_parts_with_preflight(
            rule,
            dst.homspace(),
            Arc::clone(dst.structure()),
            source,
            dst.nout(),
            lowered_axes.as_spec(),
            preflight,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_fusion_parts_with_preflight<R, F>(
        rule: &R,
        dst_homspace: &FusionTreeHomSpace,
        dst_structure: Arc<BlockStructure>,
        src: OrientedTraceSource<'_>,
        dst_codomain_rank: usize,
        axes: TensorTraceAxisSpec<'_>,
        preflight: F,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>,
        C: Clone
            + Add<Output = C>
            + Mul<Output = C>
            + Zero
            + RealStructuralCoefficient
            + Send
            + Sync
            + 'static,
        F: FnOnce(
            OrientedFusionTreeHomSpace<'_>,
            &TensorTraceAxisPlan,
        ) -> Result<TraceAdmission, OperationError>,
    {
        // Captured before any cache-4 lookup: a reset during the compile
        // leaves its built groups unpublished.
        let epoch = symtn_core::core_reset_epoch();
        crate::admission::require_symmetric_braiding(
            rule.braiding_style(),
            crate::admission::SymmetricBraidingOp::Trace,
        )?;
        let axis_plan =
            TensorTraceAxisPlan::compile(src.structure.rank(), dst_structure.rank(), axes)?;
        let admission = preflight(src.homspace, &axis_plan)?;
        // Why not cache this per-call result: eager compilation only needs to
        // pass the checked selection forward instead of deriving it again.
        validate_fusion_trace_homspace(
            rule,
            dst_homspace,
            src.homspace,
            &axis_plan,
            dst_codomain_rank,
            admission,
        )?;
        // After every admission check: a hit skips only the lowering, so
        // errors keep their order and checked admission still runs.
        let identity = rule.rule_identity();
        let completed = CompletedTrace::new(&axis_plan, src, dst_codomain_rank);
        let key = completed.key::<C>(
            &identity,
            TransformerMode::MultiplicityFree,
            &dst_structure,
            src,
        );
        if let Some(core) = lookup_trace::<TraceCore<C>>(&key) {
            return Ok(Self::bind(core, dst_structure, src));
        }
        let (terms, pending) = build_trace_terms::<MultiplicityFreeAdmissionMode, R>(
            rule,
            &dst_structure,
            src,
            &axis_plan,
            dst_codomain_rank,
        )?;
        let structure = Self::assemble(axis_plan, terms, dst_structure, src)?;
        pending.publish(epoch);
        if let Some(staged) = completed.staged::<C>(
            &identity,
            TransformerMode::MultiplicityFree,
            &structure,
            src,
        ) {
            publish_staged_trace(staged, epoch);
        }
        Ok(structure)
    }

    /// A resident core bound to the caller's structures. The key's content
    /// ids make them the structures the core was assembled over, value for
    /// value, so its pairing and layout checks still hold.
    fn bind(
        core: Arc<TraceCore<C>>,
        dst_structure: Arc<BlockStructure>,
        src: OrientedTraceSource<'_>,
    ) -> Self {
        Self {
            core,
            dst_structure,
            src_structure: Arc::clone(src.structure),
        }
    }

    /// The only builder of a core. The descriptor is compiled here from
    /// `terms` (one descriptor term per fusion term, in order, or an error),
    /// so `descriptor.terms()[i]` addresses `terms[i]`'s block pair for every
    /// `i`; the fields are private, so replay can pair them without checking.
    fn assemble(
        axis_plan: TensorTraceAxisPlan,
        terms: Vec<TensorTraceFusionStructureTerm<C>>,
        dst_structure: Arc<BlockStructure>,
        src: OrientedTraceSource<'_>,
    ) -> Result<Self, OperationError> {
        let descriptor = TensorTraceDescriptor::compile_oriented(
            &axis_plan,
            terms.iter().map(|term| (term.dst_block, term.src_block)),
            &dst_structure,
            src,
        )?;
        validate_destination_layouts_injective(
            &dst_structure,
            "tensor trace destination layouts overlap",
        )?;

        Ok(Self {
            core: Arc::new(TraceCore {
                dst_rank: dst_structure.rank(),
                src_rank: src.structure.rank(),
                output_axes: axis_plan.output_axes,
                trace_lhs_axes: axis_plan.trace_lhs_axes,
                trace_rhs_axes: axis_plan.trace_rhs_axes,
                terms,
                descriptor,
            }),
            dst_structure,
            src_structure: Arc::clone(src.structure),
        })
    }

    /// Each descriptor term with the fusion term it was compiled from.
    pub(crate) fn paired_terms(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            &TensorTraceDescriptorTerm,
            &TensorTraceFusionStructureTerm<C>,
        ),
    > {
        self.core.descriptor.terms().iter().zip(&self.core.terms)
    }

    /// The `index`-th pair of [`Self::paired_terms`].
    pub(crate) fn paired_term(
        &self,
        index: usize,
    ) -> (
        &TensorTraceDescriptorTerm,
        &TensorTraceFusionStructureTerm<C>,
    ) {
        (
            &self.core.descriptor.terms()[index],
            &self.core.terms[index],
        )
    }

    #[inline]
    pub fn dst_rank(&self) -> usize {
        self.core.dst_rank
    }

    #[inline]
    pub fn src_rank(&self) -> usize {
        self.core.src_rank
    }

    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        &self.core.output_axes
    }

    #[inline]
    pub fn trace_lhs_axes(&self) -> &[usize] {
        &self.core.trace_lhs_axes
    }

    #[inline]
    pub fn trace_rhs_axes(&self) -> &[usize] {
        &self.core.trace_rhs_axes
    }

    #[inline]
    pub fn terms(&self) -> &[TensorTraceFusionStructureTerm<C>] {
        &self.core.terms
    }

    #[inline]
    pub(crate) fn descriptor(&self) -> &TensorTraceDescriptor {
        &self.core.descriptor
    }

    /// The destination block structure the terms were compiled for.
    #[doc(hidden)]
    #[inline]
    pub fn dst_structure(&self) -> &Arc<BlockStructure> {
        &self.dst_structure
    }

    pub(crate) fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorTraceFusionStructureTerm<C> {
    dst_key: FusionTreePairKey,
    src_key: FusionTreePairKey,
    dst_block: usize,
    src_block: usize,
    coefficient: C,
}

impl<C> TensorTraceFusionStructureTerm<C> {
    #[inline]
    pub fn dst_key(&self) -> &FusionTreePairKey {
        &self.dst_key
    }

    #[inline]
    pub fn src_key(&self) -> &FusionTreePairKey {
        &self.src_key
    }

    #[inline]
    pub fn dst_block(&self) -> usize {
        self.dst_block
    }

    #[inline]
    pub fn src_block(&self) -> usize {
        self.src_block
    }

    #[inline]
    pub fn coefficient(&self) -> &C {
        &self.coefficient
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TensorTraceDescriptor {
    source_conjugate: bool,
    terms: Vec<TensorTraceDescriptorTerm>,
    destination_layouts: Vec<TensorTraceDestinationLayout>,
    destination_shape: Vec<usize>,
    destination_strides: Vec<isize>,
    destination_zero_strides: Vec<isize>,
    output_shape: Vec<usize>,
    trace_shape: Vec<usize>,
    dst_strides: Vec<isize>,
    src_output_strides: Vec<isize>,
    src_trace_strides: Vec<isize>,
    destination_producers: Vec<usize>,
}

impl TensorTraceDescriptor {
    /// Heap bytes of every buffer this descriptor owns.
    fn heap_bytes(&self) -> usize {
        let words = self.destination_shape.capacity()
            + self.destination_strides.capacity()
            + self.destination_zero_strides.capacity()
            + self.output_shape.capacity()
            + self.trace_shape.capacity()
            + self.dst_strides.capacity()
            + self.src_output_strides.capacity()
            + self.src_trace_strides.capacity()
            + self.destination_producers.capacity();
        words
            .saturating_mul(core::mem::size_of::<usize>())
            .saturating_add(
                self.terms
                    .capacity()
                    .saturating_mul(core::mem::size_of::<TensorTraceDescriptorTerm>()),
            )
            .saturating_add(
                self.destination_layouts
                    .capacity()
                    .saturating_mul(core::mem::size_of::<TensorTraceDestinationLayout>()),
            )
    }

    #[inline]
    pub(crate) fn terms(&self) -> &[TensorTraceDescriptorTerm] {
        &self.terms
    }

    #[inline]
    pub(crate) fn source_conjugate(&self) -> bool {
        self.source_conjugate
    }

    #[inline]
    fn destination_layouts(&self) -> &[TensorTraceDestinationLayout] {
        &self.destination_layouts
    }

    fn destination_shape(&self, layout: &TensorTraceDestinationLayout) -> &[usize] {
        &self.destination_shape[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn destination_strides(&self, layout: &TensorTraceDestinationLayout) -> &[isize] {
        &self.destination_strides[layout.layout_start..layout.layout_start + layout.rank]
    }

    fn destination_zero_strides(&self, layout: &TensorTraceDestinationLayout) -> &[isize] {
        &self.destination_zero_strides[..layout.rank]
    }

    pub(crate) fn output_shape(&self, term: &TensorTraceDescriptorTerm) -> &[usize] {
        &self.output_shape[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(crate) fn trace_shape(&self, term: &TensorTraceDescriptorTerm) -> &[usize] {
        &self.trace_shape[term.trace_layout_start..term.trace_layout_start + term.trace_rank]
    }

    pub(crate) fn dst_strides(&self, term: &TensorTraceDescriptorTerm) -> &[isize] {
        &self.dst_strides[term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(crate) fn src_output_strides(&self, term: &TensorTraceDescriptorTerm) -> &[isize] {
        &self.src_output_strides
            [term.output_layout_start..term.output_layout_start + term.output_rank]
    }

    pub(crate) fn src_trace_strides(&self, term: &TensorTraceDescriptorTerm) -> &[isize] {
        &self.src_trace_strides[term.trace_layout_start..term.trace_layout_start + term.trace_rank]
    }

    fn destination_producer_offsets(&self) -> &[usize] {
        &self.destination_producers[..self.destination_layouts.len() + 1]
    }

    fn destination_producer_indices(&self) -> &[usize] {
        &self.destination_producers[self.destination_layouts.len() + 1..]
    }

    fn compile_oriented<I>(
        axis_plan: &TensorTraceAxisPlan,
        terms: I,
        dst_structure: &BlockStructure,
        src: OrientedTraceSource<'_>,
    ) -> Result<Self, OperationError>
    where
        I: ExactSizeIterator<Item = (usize, usize)>,
    {
        Self::compile_with_source_axis_map(axis_plan, terms, dst_structure, src.structure, |axis| {
            src.logical_axis_to_storage(axis)
        })
    }

    fn compile_with_source_axis_map<F, I>(
        axis_plan: &TensorTraceAxisPlan,
        terms: I,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        source_axis: F,
    ) -> Result<Self, OperationError>
    where
        F: Fn(usize) -> usize,
        I: ExactSizeIterator<Item = (usize, usize)>,
    {
        let mut descriptor = Self {
            source_conjugate: axis_plan.source_conjugate,
            ..Self::default()
        };
        descriptor.terms.reserve(terms.len());
        descriptor
            .output_shape
            .reserve(terms.len().saturating_mul(axis_plan.output_axes.len()));
        descriptor
            .trace_shape
            .reserve(terms.len().saturating_mul(axis_plan.trace_lhs_axes.len()));
        descriptor
            .dst_strides
            .reserve(terms.len().saturating_mul(axis_plan.output_axes.len()));
        descriptor
            .src_output_strides
            .reserve(terms.len().saturating_mul(axis_plan.output_axes.len()));
        descriptor
            .src_trace_strides
            .reserve(terms.len().saturating_mul(axis_plan.trace_lhs_axes.len()));
        descriptor
            .destination_layouts
            .reserve(dst_structure.block_count());
        let destination_layout_len = dst_structure
            .block_count()
            .saturating_mul(dst_structure.rank());
        descriptor
            .destination_shape
            .reserve_exact(destination_layout_len);
        descriptor
            .destination_strides
            .reserve_exact(destination_layout_len);
        descriptor
            .destination_zero_strides
            .resize(dst_structure.rank(), 0);

        for (dst_block_index, src_block_index) in terms {
            let dst_block = dst_structure.block(dst_block_index)?;
            let src_block = src_structure.block(src_block_index)?;
            let output_layout_start = descriptor.output_shape.len();
            for (dst_axis, &src_axis) in axis_plan.output_axes.iter().enumerate() {
                let src_axis = source_axis(src_axis);
                let dst_dim = dst_block.shape()[dst_axis];
                let src_dim = src_block.shape()[src_axis];
                if dst_dim != src_dim {
                    let src_shape = axis_plan
                        .output_axes
                        .iter()
                        .map(|&axis| src_block.shape()[source_axis(axis)])
                        .collect::<Vec<_>>();
                    return Err(OperationError::ShapeMismatch {
                        dst: dst_block.shape().to_vec(),
                        src: src_shape,
                    });
                }
                descriptor.output_shape.push(dst_dim);
                descriptor
                    .dst_strides
                    .push(stride_to_isize(dst_block.strides()[dst_axis])?);
                descriptor
                    .src_output_strides
                    .push(stride_to_isize(src_block.strides()[src_axis])?);
            }

            let trace_layout_start = descriptor.trace_shape.len();
            for (&lhs_axis, &rhs_axis) in axis_plan
                .trace_lhs_axes
                .iter()
                .zip(axis_plan.trace_rhs_axes.iter())
            {
                let lhs_axis = source_axis(lhs_axis);
                let rhs_axis = source_axis(rhs_axis);
                let lhs_dim = src_block.shape()[lhs_axis];
                let rhs_dim = src_block.shape()[rhs_axis];
                if lhs_dim != rhs_dim {
                    return Err(OperationError::ShapeMismatch {
                        dst: vec![lhs_dim],
                        src: vec![rhs_dim],
                    });
                }
                descriptor.trace_shape.push(lhs_dim);
                let trace_stride = src_block.strides()[lhs_axis]
                    .checked_add(src_block.strides()[rhs_axis])
                    .ok_or(OperationError::ElementCountOverflow)?;
                descriptor
                    .src_trace_strides
                    .push(stride_to_isize(trace_stride)?);
            }

            descriptor.terms.push(TensorTraceDescriptorTerm {
                dst_block: dst_block_index,
                src_block: src_block_index,
                output_layout_start,
                trace_layout_start,
                output_rank: axis_plan.output_axes.len(),
                trace_rank: axis_plan.trace_lhs_axes.len(),
                dst_offset: offset_to_isize(dst_block.offset())?,
                src_offset: offset_to_isize(src_block.offset())?,
            });
        }

        for dst_block in 0..dst_structure.block_count() {
            let block = dst_structure.block(dst_block)?;
            let layout_start = descriptor.destination_shape.len();
            descriptor
                .destination_shape
                .extend_from_slice(block.shape());
            for &stride in block.strides() {
                descriptor
                    .destination_strides
                    .push(stride_to_isize(stride)?);
            }
            descriptor
                .destination_layouts
                .push(TensorTraceDestinationLayout {
                    layout_start,
                    rank: block.shape().len(),
                    offset: offset_to_isize(block.offset())?,
                });
        }
        descriptor.compile_destination_producer_partition(dst_structure.block_count())?;

        Ok(descriptor)
    }

    fn compile_destination_producer_partition(
        &mut self,
        block_count: usize,
    ) -> Result<(), OperationError> {
        let offsets_len = block_count
            .checked_add(1)
            .ok_or(OperationError::ElementCountOverflow)?;
        let total_len = offsets_len
            .checked_add(self.terms.len())
            .ok_or(OperationError::ElementCountOverflow)?;
        self.destination_producers.resize(total_len, 0);
        let (offsets, indices) = self.destination_producers.split_at_mut(offsets_len);
        for term in &self.terms {
            let end = offsets.get_mut(term.dst_block + 1).ok_or(
                OperationError::BlockIndexOutOfBounds {
                    tensor: "trace destination",
                    index: term.dst_block,
                    count: block_count,
                },
            )?;
            *end = end
                .checked_add(1)
                .ok_or(OperationError::ElementCountOverflow)?;
        }

        for block in 0..block_count {
            offsets[block + 1] = offsets[block]
                .checked_add(offsets[block + 1])
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        for (term_index, term) in self.terms.iter().enumerate().rev() {
            let cursor = &mut offsets[term.dst_block + 1];
            *cursor = cursor
                .checked_sub(1)
                .ok_or(OperationError::ElementCountOverflow)?;
            indices[*cursor] = term_index;
        }
        for block in 0..block_count {
            offsets[block] = offsets[block + 1];
        }
        offsets[block_count] = self.terms.len();
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorTraceDestinationLayout {
    layout_start: usize,
    rank: usize,
    offset: isize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TensorTraceDescriptorTerm {
    pub(crate) dst_block: usize,
    pub(crate) src_block: usize,
    pub(crate) output_layout_start: usize,
    pub(crate) trace_layout_start: usize,
    pub(crate) output_rank: usize,
    pub(crate) trace_rank: usize,
    pub(crate) dst_offset: isize,
    pub(crate) src_offset: isize,
}

/// The completed-transformer key of one compile, held across the lowering
/// that consumes its axis plan: the `(p…, q…)` permutation halves (inline
/// up to rank 12) and the split that recovers the plan from them.
struct CompletedTrace {
    codomain_permutation: smallvec::SmallVec<[usize; 12]>,
    domain_permutation: smallvec::SmallVec<[usize; 12]>,
    split: TraceSplit,
}

impl CompletedTrace {
    fn new(
        axis_plan: &TensorTraceAxisPlan,
        src: OrientedTraceSource<'_>,
        dst_codomain_rank: usize,
    ) -> Self {
        let (front, back) = axis_plan.output_axes.split_at(dst_codomain_rank);
        Self {
            codomain_permutation: front
                .iter()
                .chain(&axis_plan.trace_lhs_axes)
                .copied()
                .collect(),
            domain_permutation: back
                .iter()
                .chain(&axis_plan.trace_rhs_axes)
                .copied()
                .collect(),
            split: TraceSplit {
                open_codomain_rank: dst_codomain_rank,
                pairs: axis_plan.trace_lhs_axes.len(),
                storage_nout: src.storage_nout,
                storage_nin: src.storage_nin,
            },
        }
    }

    fn key<'a, C: 'static>(
        &'a self,
        identity: &'a RuleIdentity,
        mode: TransformerMode,
        dst: &'a BlockStructure,
        src: OrientedTraceSource<'a>,
    ) -> TraceKeyRef<'a> {
        TraceKeyRef {
            rule: identity,
            mode,
            coefficient: TypeId::of::<C>(),
            codomain_permutation: &self.codomain_permutation,
            domain_permutation: &self.domain_permutation,
            orientation: src.orientation,
            split: self.split,
            dst,
            src: src.structure,
        }
    }

    /// The entry to publish for `structure`, or `None` when a keyed content
    /// is not canonical (an expert or staged structure gets a fresh id per
    /// call, so its entry could only evict live ones).
    fn staged<C>(
        &self,
        identity: &RuleIdentity,
        mode: TransformerMode,
        structure: &TensorTraceFusionStructure<C>,
        src: OrientedTraceSource<'_>,
    ) -> Option<(
        crate::tree_transform::CompletedTransformerKey,
        Arc<dyn std::any::Any + Send + Sync>,
        u64,
    )>
    where
        C: 'static + Send + Sync,
    {
        let key = self.key::<C>(identity, mode, &structure.dst_structure, src);
        key.publishable()
            .then(|| self.entry(identity, mode, structure, src))
    }

    /// The entry of `structure` under this key, publishable or not.
    fn entry<C>(
        &self,
        identity: &RuleIdentity,
        mode: TransformerMode,
        structure: &TensorTraceFusionStructure<C>,
        src: OrientedTraceSource<'_>,
    ) -> (
        crate::tree_transform::CompletedTransformerKey,
        Arc<dyn std::any::Any + Send + Sync>,
        u64,
    )
    where
        C: 'static + Send + Sync,
    {
        self.key::<C>(identity, mode, &structure.dst_structure, src)
            .staged_entry(
                Arc::clone(&structure.core),
                structure
                    .core
                    .charged_bytes(&structure.dst_structure, src.structure),
            )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorTraceAxisPlan {
    output_axes: Vec<usize>,
    trace_lhs_axes: Vec<usize>,
    trace_rhs_axes: Vec<usize>,
    source_conjugate: bool,
}

impl TensorTraceAxisPlan {
    fn compile(
        src_rank: usize,
        dst_rank: usize,
        axes: TensorTraceAxisSpec<'_>,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::TraceAxisPlanDerivation);
        if axes.trace_lhs_axes().len() != axes.trace_rhs_axes().len() {
            return Err(OperationError::TraceAxisCountMismatch {
                lhs: axes.trace_lhs_axes().len(),
                rhs: axes.trace_rhs_axes().len(),
            });
        }
        if axes.output_axes().len() != dst_rank {
            return Err(OperationError::RankMismatch {
                expected: dst_rank,
                actual: axes.output_axes().len(),
            });
        }

        let mut seen = symtn_core::axes::AxisMask::new(src_rank);
        mark_axes("trace output", axes.output_axes(), src_rank, &mut seen)?;
        mark_axes("trace lhs", axes.trace_lhs_axes(), src_rank, &mut seen)?;
        mark_axes("trace rhs", axes.trace_rhs_axes(), src_rank, &mut seen)?;
        if !seen.is_full() {
            let mut all_axes = Vec::with_capacity(
                axes.output_axes().len()
                    + axes.trace_lhs_axes().len()
                    + axes.trace_rhs_axes().len(),
            );
            all_axes.extend_from_slice(axes.output_axes());
            all_axes.extend_from_slice(axes.trace_lhs_axes());
            all_axes.extend_from_slice(axes.trace_rhs_axes());
            return Err(OperationError::InvalidAxisSet {
                tensor: "trace source",
                axes: all_axes,
                rank: src_rank,
            });
        }

        Ok(Self {
            output_axes: axes.output_axes().to_vec(),
            trace_lhs_axes: axes.trace_lhs_axes().to_vec(),
            trace_rhs_axes: axes.trace_rhs_axes().to_vec(),
            source_conjugate: axes.source_conjugate(),
        })
    }
}

/// What the trace preflight decided: the selected output HomSpace, and
/// whether every traced pair joins mutually dual legs.
///
/// Why the duality is a check and not the preflight's own error: TensorKit
/// `trace_permute!` compares `space(tdst)` before the pair duality, so a
/// trace into a destination raises the destination's space error first; an
/// eager trace has no destination and raises it right away.
#[doc(hidden)]
#[derive(Debug)]
pub struct TracePreflight {
    selected_homspace: FusionTreeHomSpace,
    trace_pairs_match: bool,
}

impl TracePreflight {
    #[inline]
    pub fn selected_homspace(&self) -> &FusionTreeHomSpace {
        &self.selected_homspace
    }

    #[inline]
    pub fn into_selected_homspace(self) -> FusionTreeHomSpace {
        self.selected_homspace
    }

    /// `StructureMismatch { "trace axes" }` when a traced pair joins legs
    /// that are not mutually dual (TensorKit
    /// `space(tsrc, q₁) == dual(space(tsrc, q₂))`).
    #[inline]
    pub fn require_dual_pairs(&self) -> Result<(), OperationError> {
        if self.trace_pairs_match {
            Ok(())
        } else {
            Err(OperationError::StructureMismatch {
                tensor: "trace axes",
            })
        }
    }
}

/// The checked Generic trace preflight (see
/// [`tensortrace_fusion_dyn_preflight_checked`]). No payload or destination
/// layout is allocated here.
#[doc(hidden)]
pub fn tensortrace_fusion_dyn_preflight_generic_checked<R>(
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<TracePreflight, CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    trace_preflight::<CheckedGenericAdmissionMode, R>(src, axes, dst_nout)
}

/// The checked Generic trace terms and descriptor
/// [`tensortrace_fusion_dyn_owned_generic_checked`] executes: the structural
/// half of a checked Generic trace, before any payload is read.
///
/// Its permutation groups built on a cache-4 miss are staged into `pending`;
/// the caller flushes them once its whole call succeeded (#2043).
pub(crate) fn compile_fusion_dyn_generic_checked<R>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    src_space: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    pending: &mut CheckedPendingCoefficients,
) -> Result<TensorTraceFusionStructure<R::Scalar>, CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericPivotal,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + 'static,
{
    compile_fusion_generic_checked_parts(
        dst_space.space().homspace(),
        Arc::clone(dst_space.space().structure()),
        dst_space.space().nout(),
        src_space,
        axes,
        false,
        pending,
    )
}

fn compile_fusion_generic_checked_parts<R>(
    dst_homspace: &FusionTreeHomSpace,
    dst_structure: Arc<BlockStructure>,
    dst_nout: usize,
    src_space: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    admitted: bool,
    pending: &mut CheckedPendingCoefficients,
) -> Result<TensorTraceFusionStructure<R::Scalar>, CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericPivotal,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + 'static,
{
    let orientation = if axes.source_conjugate() {
        FusionTreePairOrientation::Adjoint
    } else {
        FusionTreePairOrientation::Direct
    };
    // `admitted`: the destination was selected from this source and these
    // axes by a preflight that also proved the pair duality (#2146).
    if !admitted {
        let preflight =
            tensortrace_fusion_dyn_preflight_generic_checked(src_space, axes, dst_nout)?;
        // TensorKit's order: the destination space, then the pair duality.
        if preflight.selected_homspace() != dst_homspace {
            return Err(CheckedGenericPlanError::Operation(
                OperationError::StructureMismatch { tensor: "dst" },
            ));
        }
        preflight
            .require_dual_pairs()
            .map_err(CheckedGenericPlanError::Operation)?;
    }
    let lowered_axes = lower_tensortrace_source_adjoint_axes_dyn(
        src_space.space().nout(),
        src_space.space().nin(),
        axes,
    )
    .map_err(CheckedGenericPlanError::Operation)?;
    let axis_plan = TensorTraceAxisPlan::compile(
        src_space.space().structure().rank(),
        dst_structure.rank(),
        lowered_axes.as_spec(),
    )
    .map_err(CheckedGenericPlanError::Operation)?;
    let source = OrientedTraceSource::new(
        src_space.space().homspace(),
        src_space.space().structure(),
        src_space.space().nout(),
        src_space.space().nin(),
        orientation,
    );
    // After every admission check, as the multiplicity-free compile.
    let identity = src_space.provider().rule_identity();
    let completed = CompletedTrace::new(&axis_plan, source, dst_nout);
    let key = completed.key::<R::Scalar>(
        &identity,
        TransformerMode::CheckedGeneric,
        &dst_structure,
        source,
    );
    if let Some(core) = lookup_trace::<TraceCore<R::Scalar>>(&key) {
        return Ok(TensorTraceFusionStructure::bind(
            core,
            dst_structure,
            source,
        ));
    }
    let (terms, built) = build_trace_terms::<CheckedGenericAdmissionMode, R>(
        src_space.provider(),
        &dst_structure,
        source,
        &axis_plan,
        dst_nout,
    )?;
    let structure = TensorTraceFusionStructure::assemble(axis_plan, terms, dst_structure, source)
        .map_err(CheckedGenericPlanError::Operation)?;
    pending.stage(built);
    // Published with the groups, only after the destination commit, keyed
    // then on the committed destination (a cold call compiles on a preview).
    pending.stage_trace(
        completed.entry::<R::Scalar>(
            &identity,
            TransformerMode::CheckedGeneric,
            &structure,
            source,
        ),
        Arc::clone(&structure.dst_structure),
        Arc::clone(source.structure),
    );
    Ok(structure)
}

/// The destination steps of the one owned trace ([`tensortrace_owned_in`],
/// #2142): multiplicity-free derives and publishes its destination when it
/// stages; checked Generic prepares it and commits after execution (#2063).
///
/// Why a sibling of [`crate::PivotalCoefficientAlgebra`] and not more
/// methods on it: the checked stage and pending coefficients are
/// crate-internal, and that trait is public.
pub(crate) trait TraceStaging<R>: crate::PivotalCoefficientAlgebra<R> {
    /// Multiplicity-free: the derived, already-published destination.
    /// Checked: the prepared, unpublished destination.
    type TraceStage<'a>
    where
        R: 'a;
    /// Multiplicity-free: nothing. Checked: the coefficients the compile
    /// built, published only after the commit.
    type TracePending;

    /// The braiding, axes and selected HomSpace, the duality left open.
    fn preflight_trace(
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<TracePreflight, Self::Error>;

    /// The pair duality, then the destination of `preflight`.
    fn stage_preflighted_trace<'a>(
        src: &'a BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'a>,
        preflight: TracePreflight,
    ) -> Result<Self::TraceStage<'a>, Self::Error>;

    /// Whether `dst` is the staged destination by value; provider-free and
    /// allocation-free.
    fn trace_destination_matches(stage: &Self::TraceStage<'_>, dst: &DynamicFusionMapSpace)
        -> bool;

    /// Preflight, then the pair duality (with no destination it follows at
    /// once, TensorKit `trace_permute!`), then the destination.
    fn stage_trace<'a>(
        src: &'a BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'a>,
        dst_nout: usize,
    ) -> Result<Self::TraceStage<'a>, Self::Error> {
        let preflight = Self::preflight_trace(src, axes, dst_nout)?;
        Self::stage_preflighted_trace(src, axes, preflight)
    }

    /// Opened before the compile, so a checked reset during it leaves the
    /// built groups unpublished.
    fn open_trace() -> Self::TracePending;

    fn trace_nout(stage: &Self::TraceStage<'_>) -> usize;

    fn compile_trace(
        stage: &Self::TraceStage<'_>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
        pending: &mut Self::TracePending,
    ) -> Result<TensorTraceFusionStructure<Self::Coeff>, Self::Error>;

    /// Publishes the destination and what the compile staged.
    fn commit_trace(
        src: &BoundDynamicFusionMapSpace<R>,
        stage: Self::TraceStage<'_>,
        pending: Self::TracePending,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::Error>;
}

impl<R> TraceStaging<R> for MultiplicityFreeAdmissionMode
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra,
{
    type TraceStage<'a>
        = TensorTraceStage<'a, R, BoundDynamicFusionMapSpace<R>>
    where
        R: 'a;
    type TracePending = ();

    fn preflight_trace(
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<TracePreflight, OperationError> {
        tensortrace_fusion_dyn_preflight_checked(src, axes, dst_nout)
    }

    fn stage_preflighted_trace<'a>(
        src: &'a BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'a>,
        preflight: TracePreflight,
    ) -> Result<TensorTraceStage<'a, R, BoundDynamicFusionMapSpace<R>>, OperationError> {
        preflight.require_dual_pairs()?;
        let destination = src.derive_from_final_homspace(preflight.into_selected_homspace())?;
        Ok(TensorTraceStage {
            destination,
            src,
            axes,
        })
    }

    fn trace_destination_matches(
        stage: &TensorTraceStage<'_, R, BoundDynamicFusionMapSpace<R>>,
        dst: &DynamicFusionMapSpace,
    ) -> bool {
        stage.destination.space() == dst
    }

    fn open_trace() {}

    fn trace_nout(stage: &TensorTraceStage<'_, R, BoundDynamicFusionMapSpace<R>>) -> usize {
        stage.destination.space().nout()
    }

    fn compile_trace(
        stage: &TensorTraceStage<'_, R, BoundDynamicFusionMapSpace<R>>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
        (): &mut (),
    ) -> Result<TensorTraceFusionStructure<f64>, OperationError> {
        if !stage.admits(src, axes) {
            return <Self as crate::PivotalCoefficientAlgebra<R>>::trace_terms(
                &stage.destination,
                src,
                axes,
            );
        }
        TensorTraceFusionStructure::compile_fusion_dyn_raw_with_preflight(
            src.provider(),
            stage.destination.space(),
            src.space(),
            axes,
            |_, _| Ok(TraceAdmission::Admitted),
        )
    }

    fn commit_trace(
        _: &BoundDynamicFusionMapSpace<R>,
        stage: TensorTraceStage<'_, R, BoundDynamicFusionMapSpace<R>>,
        (): (),
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError> {
        Ok(stage.destination)
    }
}

impl<R> TraceStaging<R> for CheckedGenericAdmissionMode
where
    R: CheckedGenericPivotal<Scalar = f64>,
{
    type TraceStage<'a>
        = TensorTraceStage<'a, R, PreparedCheckedGenericDynamicSpace>
    where
        R: 'a;
    type TracePending = CheckedPendingCoefficients;

    fn preflight_trace(
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<TracePreflight, CheckedGenericPlanError<R::Error>> {
        tensortrace_fusion_dyn_preflight_generic_checked(src, axes, dst_nout)
    }

    fn stage_preflighted_trace<'a>(
        src: &'a BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'a>,
        preflight: TracePreflight,
    ) -> Result<
        TensorTraceStage<'a, R, PreparedCheckedGenericDynamicSpace>,
        CheckedGenericPlanError<R::Error>,
    > {
        preflight
            .require_dual_pairs()
            .map_err(CheckedGenericPlanError::Operation)?;
        let destination = src
            .prepare_final_homspace_generic_with_checked(
                src.provider(),
                preflight.into_selected_homspace(),
            )
            .map_err(CheckedGenericPlanError::from)?;
        Ok(TensorTraceStage {
            destination,
            src,
            axes,
        })
    }

    fn trace_destination_matches(
        stage: &TensorTraceStage<'_, R, PreparedCheckedGenericDynamicSpace>,
        dst: &DynamicFusionMapSpace,
    ) -> bool {
        stage.destination.matches(dst)
    }

    fn open_trace() -> CheckedPendingCoefficients {
        CheckedPendingCoefficients::new()
    }

    fn trace_nout(stage: &TensorTraceStage<'_, R, PreparedCheckedGenericDynamicSpace>) -> usize {
        stage.destination.nout()
    }

    /// A warm complete hit compiles against the live canonical wrapper, so
    /// the published coupled-region memo is reused rather than rebuilt.
    fn compile_trace(
        stage: &TensorTraceStage<'_, R, PreparedCheckedGenericDynamicSpace>,
        src: &BoundDynamicFusionMapSpace<R>,
        axes: TensorTraceAxisSpec<'_>,
        pending: &mut CheckedPendingCoefficients,
    ) -> Result<TensorTraceFusionStructure<f64>, CheckedGenericPlanError<R::Error>> {
        let destination = &stage.destination;
        src.validate_prepared_final_homspace_generic_checked(destination)
            .map_err(CheckedGenericPlanError::Operation)?;
        compile_fusion_generic_checked_parts(
            destination.homspace(),
            destination.shared_structure(),
            destination.nout(),
            src,
            axes,
            stage.admits(src, axes),
            pending,
        )
    }

    /// The final provider guard remains defensive.
    fn commit_trace(
        src: &BoundDynamicFusionMapSpace<R>,
        stage: TensorTraceStage<'_, R, PreparedCheckedGenericDynamicSpace>,
        pending: CheckedPendingCoefficients,
    ) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericPlanError<R::Error>> {
        let preview = stage.destination.shared_structure();
        let space = src
            .commit_final_homspace_generic_bound_checked(stage.destination)
            .map_err(CheckedGenericPlanError::Operation)?;
        pending.flush_committed(&[(preview, Arc::clone(space.space().structure()))]);
        Ok(space)
    }
}

/// The one owned trace into a staged destination: compile, then read the
/// payload, then execute, then commit. Payload production follows all
/// structural admission and pivotal compilation; a failed compile or
/// execution commits nothing.
///
/// `read` may answer a single-scalar, unconjugated destination from the
/// compiled terms alone (a compact-diagonal source, #1866); its value is
/// scaled by `alpha` and neither the payload nor the execution runs. It is
/// asked nothing for any other destination, and on `None` the same compiled
/// structure executes in the same transaction.
///
/// [TensorKit trace_permute!](https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/tensoroperations.jl#L208-L302)
/// fixes pair selection, splitting and pivotal factors; [QSpace::trace](https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/QSpace.cc#lines-4501:4625)
/// and its [traceQS](https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/traceQS.cc)
/// entry retain grouped reduced-block execution. Neither reference has a
/// checked layout-publication transaction: deferring publication until
/// execution succeeds is a Rust ownership/error adaptation, not another
/// numerical path.
#[allow(clippy::type_complexity)]
#[cfg(test)]
fn tensortrace_owned_in<M, R, D, P>(
    exec: symtn_operations::host_pool::Exec<'_>,
    stage: M::TraceStage<'_>,
    src: &BoundDynamicFusionMapSpace<R>,
    read: impl FnOnce(&TensorTraceFusionStructure<M::Coeff>) -> Option<D>,
    payload: impl FnOnce() -> P,
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), M::Error>
where
    M: TraceStaging<R>,
    M::Error: From<OperationError>,
    M::Coeff: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<M::Coeff>
        + Send
        + Sync,
    P: AsRef<[D]>,
{
    trace_staged_in::<M, R, D, _, M::Error>(stage, src, axes, read, |structure, value, nout| {
        Ok(match value {
            Some(value) => vec![value * alpha],
            None => tensortrace_fusion_structure_owned_host(
                exec,
                structure,
                nout,
                src.space().structure(),
                payload().as_ref(),
                alpha,
            )?,
        })
    })
}

/// The skeleton every trace runs on a staged destination: open, compile,
/// then `execute` (given `read`'s answer for a single-scalar, unconjugated
/// destination), then commit — so the owned and the existing-destination
/// traces stage, compile and publish identically, and a failed compile or
/// execution commits nothing.
fn trace_staged_in<M, R, D, T, E>(
    stage: M::TraceStage<'_>,
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    read: impl FnOnce(&TensorTraceFusionStructure<M::Coeff>) -> Option<D>,
    execute: impl FnOnce(&TensorTraceFusionStructure<M::Coeff>, Option<D>, usize) -> Result<T, E>,
) -> Result<(BoundDynamicFusionMapSpace<R>, T), E>
where
    M: TraceStaging<R>,
    E: From<M::Error>,
{
    let mut pending = M::open_trace();
    let structure = M::compile_trace(&stage, src, axes, &mut pending)?;
    let scalar = structure.dst_rank() == 0
        && structure.dst_structure.block_count() == 1
        && !structure.descriptor().source_conjugate();
    let value = scalar.then(|| read(&structure)).flatten();
    let out = execute(&structure, value, M::trace_nout(&stage))?;
    Ok((M::commit_trace(src, stage, pending)?, out))
}

/// The skeleton of every trace into an existing destination, on any
/// storage: preflight, then in TensorKit's order the staged HomSpace against
/// `dst_space` before the pair duality and the staged layout after it (both
/// the caller's `SpaceMismatch` in every mode), then `dst` for the buffer,
/// then [`trace_staged_in`] with `execute` writing into it — so the checked
/// commit follows the write (#2063).
fn trace_into_staged_in<M, R, D, X, E>(
    src: &BoundDynamicFusionMapSpace<R>,
    read: impl FnOnce(&TensorTraceFusionStructure<M::Coeff>) -> Option<D>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
    dst_space: &DynamicFusionMapSpace,
    dst: impl FnOnce() -> Result<X, E>,
    execute: impl FnOnce(&TensorTraceFusionStructure<M::Coeff>, Option<D>, X) -> Result<(), E>,
) -> Result<(), E>
where
    M: TraceStaging<R>,
    E: From<M::Error> + From<OperationError>,
{
    // Why `E` directly: a destination mismatch is the caller's argument
    // error in every mode, never a checked plan failure.
    let mismatch = || {
        E::from(OperationError::SpaceMismatch {
            message: "destination fusion space or block layout does not match the operation result",
        })
    };
    let preflight = M::preflight_trace(src, axes, dst_nout)?;
    if preflight.selected_homspace() != dst_space.homspace() {
        return Err(mismatch());
    }
    let stage = M::stage_preflighted_trace(src, axes, preflight)?;
    if !M::trace_destination_matches(&stage, dst_space) {
        return Err(mismatch());
    }
    let dst = dst()?;
    trace_staged_in::<M, R, D, (), E>(stage, src, axes, read, |structure, value, _| {
        execute(structure, value, dst)
    })?;
    Ok(())
}

/// A trace destination (`S`: derived and published for multiplicity-free,
/// prepared and unpublished for checked Generic) with the source and axes
/// whose preflight selected it.
///
/// Why it borrows the source rather than trusting the caller: the `_in`
/// compile skips its own preflight and `dst` comparison only for that same
/// source and those same axes (#2146); any other pair is compiled as against
/// a committed destination.
#[doc(hidden)]
pub struct TensorTraceStage<'a, R, S> {
    pub(crate) destination: S,
    pub(crate) src: &'a BoundDynamicFusionMapSpace<R>,
    pub(crate) axes: TensorTraceAxisSpec<'a>,
}

impl<R, S> TensorTraceStage<'_, R, S> {
    #[inline]
    pub fn destination(&self) -> &S {
        &self.destination
    }

    #[inline]
    pub fn into_destination(self) -> S {
        self.destination
    }

    /// Whether the staging preflight covers a compile of `axes` over `src`:
    /// the shared borrow keeps the very source it ran on unchanged.
    fn admits(&self, src: &BoundDynamicFusionMapSpace<R>, axes: TensorTraceAxisSpec<'_>) -> bool {
        std::ptr::eq(self.src, src) && self.axes == axes
    }
}

/// Stages a multiplicity-free trace destination, derived and published.
#[doc(hidden)]
pub fn tensortrace_stage_multiplicity_free<'a, R>(
    src: &'a BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'a>,
    dst_nout: usize,
) -> Result<TensorTraceStage<'a, R, BoundDynamicFusionMapSpace<R>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra,
{
    MultiplicityFreeAdmissionMode::stage_trace(src, axes, dst_nout)
}

/// Stages a checked Generic trace destination, prepared and unpublished;
/// [`tensortrace_checked_generic_in`] commits it after execution.
#[doc(hidden)]
pub fn tensortrace_stage_checked_generic<'a, R>(
    src: &'a BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'a>,
    dst_nout: usize,
) -> Result<
    TensorTraceStage<'a, R, PreparedCheckedGenericDynamicSpace>,
    CheckedGenericPlanError<R::Error>,
>
where
    R: CheckedGenericPivotal<Scalar = f64>,
{
    CheckedGenericAdmissionMode::stage_trace(src, axes, dst_nout)
}

/// The owned multiplicity-free trace into a
/// [`tensortrace_stage_multiplicity_free`] destination; `read` may answer a
/// scalar destination from the compiled terms without the payload.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
#[cfg(test)]
pub(crate) fn tensortrace_multiplicity_free_in<R, D, P>(
    exec: symtn_operations::host_pool::Exec<'_>,
    stage: TensorTraceStage<'_, R, BoundDynamicFusionMapSpace<R>>,
    src: &BoundDynamicFusionMapSpace<R>,
    read: impl FnOnce(&TensorTraceFusionStructure<f64>) -> Option<D>,
    payload: impl FnOnce() -> P,
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<f64>
        + Send
        + Sync,
    P: AsRef<[D]>,
{
    tensortrace_owned_in::<MultiplicityFreeAdmissionMode, R, D, P>(
        exec, stage, src, read, payload, axes, alpha,
    )
}

/// The owned checked Generic trace into a
/// [`tensortrace_stage_checked_generic`] destination; `read` may answer a
/// scalar destination from the compiled terms without the payload.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
#[cfg(test)]
pub(crate) fn tensortrace_checked_generic_in<R, D, P>(
    exec: symtn_operations::host_pool::Exec<'_>,
    stage: TensorTraceStage<'_, R, PreparedCheckedGenericDynamicSpace>,
    src: &BoundDynamicFusionMapSpace<R>,
    read: impl FnOnce(&TensorTraceFusionStructure<f64>) -> Option<D>,
    payload: impl FnOnce() -> P,
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericPivotal<Scalar = f64>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<f64>
        + Send
        + Sync,
    P: AsRef<[D]>,
{
    tensortrace_owned_in::<CheckedGenericAdmissionMode, R, D, P>(
        exec, stage, src, read, payload, axes, alpha,
    )
}

/// The owned multiplicity-free trace into a
/// [`tensortrace_stage_multiplicity_free`] destination on any storage: the
/// compile, then `execute` (given `read`'s answer for a single-scalar,
/// unconjugated destination, and the destination codomain rank) writes the
/// payload, then the commit.
#[doc(hidden)]
pub fn tensortrace_multiplicity_free_with<R, D, T, E>(
    stage: TensorTraceStage<'_, R, BoundDynamicFusionMapSpace<R>>,
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    read: impl FnOnce(&TensorTraceFusionStructure<f64>) -> Option<D>,
    execute: impl FnOnce(&TensorTraceFusionStructure<f64>, Option<D>, usize) -> Result<T, E>,
) -> Result<(BoundDynamicFusionMapSpace<R>, T), E>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra,
    E: From<OperationError>,
{
    trace_staged_in::<MultiplicityFreeAdmissionMode, R, D, T, E>(stage, src, axes, read, execute)
}

/// [`tensortrace_multiplicity_free_with`] for a
/// [`tensortrace_stage_checked_generic`] destination, committed and
/// published only after `execute` succeeds (#2063).
#[doc(hidden)]
pub fn tensortrace_checked_generic_with<R, D, T, E>(
    stage: TensorTraceStage<'_, R, PreparedCheckedGenericDynamicSpace>,
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    read: impl FnOnce(&TensorTraceFusionStructure<f64>) -> Option<D>,
    execute: impl FnOnce(&TensorTraceFusionStructure<f64>, Option<D>, usize) -> Result<T, E>,
) -> Result<(BoundDynamicFusionMapSpace<R>, T), E>
where
    R: CheckedGenericPivotal<Scalar = f64>,
    E: From<CheckedGenericPlanError<R::Error>>,
{
    trace_staged_in::<CheckedGenericAdmissionMode, R, D, T, E>(stage, src, axes, read, execute)
}

/// The multiplicity-free trace into an existing destination on any storage:
/// staging, `dst_space` against the staged destination (HomSpace before the
/// pair duality, then the layout), then `dst` for the destination, then
/// `execute` writes `alpha · trace + beta · dst` into it.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_into_multiplicity_free_with<R, D, X, E>(
    src: &BoundDynamicFusionMapSpace<R>,
    read: impl FnOnce(&TensorTraceFusionStructure<f64>) -> Option<D>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
    dst_space: &DynamicFusionMapSpace,
    dst: impl FnOnce() -> Result<X, E>,
    execute: impl FnOnce(&TensorTraceFusionStructure<f64>, Option<D>, X) -> Result<(), E>,
) -> Result<(), E>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra,
    E: From<OperationError>,
{
    trace_into_staged_in::<MultiplicityFreeAdmissionMode, R, D, X, E>(
        src, read, axes, dst_nout, dst_space, dst, execute,
    )
}

/// [`tensortrace_into_multiplicity_free_with`] in checked Generic: the same
/// commit and publication after the write (#2063).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_into_checked_generic_with<R, D, X, E>(
    src: &BoundDynamicFusionMapSpace<R>,
    read: impl FnOnce(&TensorTraceFusionStructure<f64>) -> Option<D>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
    dst_space: &DynamicFusionMapSpace,
    dst: impl FnOnce() -> Result<X, E>,
    execute: impl FnOnce(&TensorTraceFusionStructure<f64>, Option<D>, X) -> Result<(), E>,
) -> Result<(), E>
where
    R: CheckedGenericPivotal<Scalar = f64>,
    E: From<CheckedGenericPlanError<R::Error>> + From<OperationError>,
{
    trace_into_staged_in::<CheckedGenericAdmissionMode, R, D, X, E>(
        src, read, axes, dst_nout, dst_space, dst, execute,
    )
}

/// Checked Generic trace lowering through the existing strided executor.
///
/// This is a lower-layer seam only; typed dispatch can adopt it once the
/// Generic output-space admission is part of the facade contract.
#[doc(hidden)]
#[cfg(test)]
pub(crate) fn tensortrace_fusion_dyn_owned_generic_checked<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<Vec<D>, CheckedGenericPlanError<R::Error>>
where
    R: CheckedGenericPivotal,
    R::Scalar: Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + 'static,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    let mut coefficients = CheckedPendingCoefficients::new();
    let structure =
        compile_fusion_dyn_generic_checked(dst_space, src_space, axes, &mut coefficients)?;
    let data = tensortrace_fusion_dyn_structure_owned(
        exec,
        &structure,
        dst_space.space(),
        src_space.space(),
        src_data,
        alpha,
    )
    .map_err(CheckedGenericPlanError::Operation)?;
    coefficients.flush();
    Ok(data)
}

fn map_checked_generic_trace_symbol_error<E>(
    error: CheckedGenericSymbolError<E>,
) -> CheckedGenericPlanError<E> {
    match error {
        CheckedGenericSymbolError::Provider(error) => CheckedGenericPlanError::Provider(error),
        CheckedGenericSymbolError::Shape {
            symbol,
            expected,
            actual,
        } => CheckedGenericPlanError::SymbolShape {
            symbol,
            expected,
            actual,
        },
        CheckedGenericSymbolError::Core(error) => CheckedGenericPlanError::Core(error),
    }
}

fn map_checked_generic_trace_structure_error<E>(
    error: symtn_core::CheckedGenericStructureError<E>,
) -> CheckedGenericPlanError<E> {
    match error {
        symtn_core::CheckedGenericStructureError::Provider(error) => {
            CheckedGenericPlanError::Provider(error)
        }
        symtn_core::CheckedGenericStructureError::Core(error) => {
            CheckedGenericPlanError::Core(error)
        }
    }
}

#[cfg(test)]
pub(crate) fn build_fusion_trace_terms_for_test<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    axes: TensorTraceAxisSpec<'_>,
    dst_codomain_rank: usize,
) -> Result<Vec<TensorTraceFusionStructureTerm<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let axis_plan = TensorTraceAxisPlan::compile(src_structure.rank(), dst_structure.rank(), axes)?;
    let homspace = FusionTreeHomSpace::new(
        symtn_core::FusionProductSpace::new([]),
        symtn_core::FusionProductSpace::new([]),
    );
    let src_structure = Arc::new(src_structure.clone());
    let src = OrientedTraceSource::new(
        &homspace,
        &src_structure,
        src_structure.rank(),
        0,
        FusionTreePairOrientation::Direct,
    );
    // Publishes nothing: the helper probes terms, not reuse.
    build_trace_terms::<MultiplicityFreeAdmissionMode, R>(
        rule,
        dst_structure,
        src,
        &axis_plan,
        dst_codomain_rank,
    )
    .map(|(terms, _)| terms)
}

/// What a multiplicity-free compile knows about its destination's selection.
enum TraceAdmission {
    /// Selected from this source and these axes by a stage preflight that
    /// also proved the pair duality (#2146): nothing is left to compare.
    Admitted,
    /// This compile's checked preflight: compare it with the destination.
    Checked(TracePreflight),
    /// No preflight ran: select and check here.
    #[cfg(test)]
    Unchecked,
}

fn validate_fusion_trace_homspace<R>(
    rule: &R,
    dst: &FusionTreeHomSpace,
    src: OrientedFusionTreeHomSpace<'_>,
    axis_plan: &TensorTraceAxisPlan,
    dst_codomain_rank: usize,
    admission: TraceAdmission,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    let (selected, checked_trace_pairs_match) = match admission {
        TraceAdmission::Admitted => return Ok(()),
        TraceAdmission::Checked(TracePreflight {
            selected_homspace,
            trace_pairs_match,
        }) => (selected_homspace, Some(trace_pairs_match)),
        #[cfg(test)]
        TraceAdmission::Unchecked => {
            crate::probe::hit(crate::probe::Event::TraceSelectedHomspaceDerivation);
            (
                src.select(
                    rule,
                    &axis_plan.output_axes[..dst_codomain_rank],
                    &axis_plan.output_axes[dst_codomain_rank..],
                )
                .map_err(OperationError::from_core_preserving_context)?,
                None,
            )
        }
    };
    if selected != *dst {
        return Err(OperationError::StructureMismatch { tensor: "dst" });
    }
    let dst_domain_rank = axis_plan.output_axes.len() - dst_codomain_rank;
    if dst.domain().len() != dst_domain_rank {
        return Err(OperationError::StructureRankMismatch {
            expected: dst_domain_rank,
            actual: dst.domain().len(),
        });
    }
    if let Some(trace_pairs_match) = checked_trace_pairs_match {
        if !trace_pairs_match {
            return Err(OperationError::StructureMismatch {
                tensor: "trace axes",
            });
        }
        return Ok(());
    }
    for (&lhs_axis, &rhs_axis) in axis_plan
        .trace_lhs_axes
        .iter()
        .zip(axis_plan.trace_rhs_axes.iter())
    {
        let lhs = outward_axis_leg(rule, src, lhs_axis)?;
        let rhs = outward_axis_leg(rule, src, rhs_axis)?;
        if lhs != dual_sector_leg(rule, &rhs) {
            return Err(OperationError::StructureMismatch {
                tensor: "trace axes",
            });
        }
    }
    Ok(())
}

fn outward_axis_leg<R>(
    rule: &R,
    homspace: OrientedFusionTreeHomSpace<'_>,
    axis: usize,
) -> Result<SectorLeg, OperationError>
where
    R: FusionRule,
{
    homspace
        .external_axis_leg(rule, axis)
        .ok_or_else(|| OperationError::InvalidAxisSet {
            tensor: "trace source",
            axes: vec![axis],
            rank: homspace.rank(),
        })
}

fn dual_sector_leg<R>(rule: &R, leg: &SectorLeg) -> SectorLeg
where
    R: FusionRule,
{
    leg.dual(rule)
}

fn outward_axis_leg_checked<R>(
    rule: &R,
    homspace: OrientedFusionTreeHomSpace<'_>,
    axis: usize,
) -> Result<SectorLeg, OperationError>
where
    R: FusionRule + CheckedFusionAlgebra,
{
    homspace
        .try_external_axis_leg(rule, axis)
        .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?
        .ok_or_else(|| OperationError::InvalidAxisSet {
            tensor: "trace source",
            axes: vec![axis],
            rank: homspace.rank(),
        })
}

/// The one Host replay of a compiled fusion trace: structure identity and
/// data extents, the `beta` pass over every destination layout, then one
/// strided add per term.
#[allow(clippy::too_many_arguments)]
fn replay_fusion_trace_host<C, T>(
    exec: symtn_operations::host_pool::Exec<'_>,
    structure: &TensorTraceFusionStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    dst_data: &mut [T],
    src_structure: &Arc<BlockStructure>,
    src_data: &[T],
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    C: Copy,
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + Send
        + Sync,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_trace_data_extents(dst_structure, dst_data.len(), src_structure, src_data.len())?;
    let descriptor = structure.descriptor();
    scale_trace_destination_layouts(exec, descriptor, dst_data, beta)?;
    for (term, fusion_term) in structure.paired_terms() {
        tensortrace_raw_strided_kernel_add_with_coefficient_trusted(
            dst_data,
            src_data,
            descriptor.output_shape(term),
            descriptor.trace_shape(term),
            descriptor.dst_strides(term),
            descriptor.src_output_strides(term),
            descriptor.src_trace_strides(term),
            term.dst_offset,
            term.src_offset,
            descriptor.source_conjugate(),
            alpha,
            fusion_term.coefficient,
        )?;
    }
    Ok(())
}

#[cfg(test)]
/// Dynamic-rank fusion tensortrace: partial (or full) trace of `src` over
/// the `axes` trace pairs into caller-allocated `dst_data`
/// (`dst = beta * dst + alpha * trace(src)`), operating on
/// provider-bound dynamic spaces plus raw coupled-layout slices (TensorKit
/// `tensortrace!` semantics: quantum dimension factors and twists, i.e. the
/// fermionic supertrace).
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_fusion_dyn_into<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst_data: &mut [D],
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    tensortrace_fusion_dyn_into_raw(
        exec,
        src_space.provider(),
        dst_space.space(),
        dst_data,
        src_space.space(),
        src_data,
        axes,
        alpha,
        beta,
    )
}

/// Checked lowered dynamic trace. Built-in lowered rules should use this
/// entry point when typed machine-boundary errors are required; the unchecked
/// wrapper above is available to custom fusion rules.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn tensortrace_fusion_dyn_into_checked<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst_data: &mut [D],
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    tensortrace_fusion_dyn_into_checked_raw(
        exec,
        src_space.provider(),
        dst_space.space(),
        dst_data,
        src_space.space(),
        src_data,
        axes,
        alpha,
        beta,
    )
}

#[cfg(test)]
/// Internal owned-output trace path for built-in host storage.
///
/// This compiles the oriented trace semantics once, then either uses the
/// private-initialization writer or replays that same structure into an
/// initialized destination when canonical physical coverage is unavailable.
#[doc(hidden)]
pub fn tensortrace_fusion_dyn_owned<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    let structure = TensorTraceFusionStructure::compile_fusion_dyn_raw(
        src_space.provider(),
        dst_space.space(),
        src_space.space(),
        axes,
    )?;
    tensortrace_fusion_dyn_structure_owned(
        exec,
        &structure,
        dst_space.space(),
        src_space.space(),
        src_data,
        alpha,
    )
}

/// Checked built-in counterpart of [`tensortrace_fusion_dyn_owned`].
#[doc(hidden)]
#[cfg(test)]
pub(crate) fn tensortrace_fusion_dyn_owned_checked<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    src_space: &BoundDynamicFusionMapSpace<R>,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    let structure = TensorTraceFusionStructure::compile_fusion_dyn_checked_raw(
        src_space.provider(),
        dst_space.space(),
        src_space.space(),
        axes,
    )?;
    tensortrace_fusion_dyn_structure_owned(
        exec,
        &structure,
        dst_space.space(),
        src_space.space(),
        src_data,
        alpha,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensortrace_fusion_dyn_into_raw<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    let structure =
        TensorTraceFusionStructure::compile_fusion_dyn_raw(rule, dst_space, src_space, axes)?;
    tensortrace_fusion_dyn_structure_into_raw(
        exec, &structure, dst_space, dst_data, src_space, src_data, alpha, beta,
    )
}

/// Executes compiled trace terms into a new payload (the execution half of
/// the owned trace entries).
#[doc(hidden)]
#[cfg(test)]
pub(crate) fn tensortrace_fusion_dyn_structure_owned<C, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    structure: &TensorTraceFusionStructure<C>,
    dst_space: &DynamicFusionMapSpace,
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + Send
        + Sync,
{
    trace_structure_owned_parts(
        exec,
        structure,
        dst_space.structure(),
        dst_space.nout(),
        src_space.structure(),
        src_data,
        alpha,
    )
}

fn trace_structure_owned_parts<C, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    structure: &TensorTraceFusionStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    dst_nout: usize,
    src_structure: &Arc<BlockStructure>,
    src_data: &[D],
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + Send
        + Sync,
{
    // Why: a public executor must not replay terms compiled for other spaces;
    // the pointer-equality fast path keeps compiled-in-place callers free.
    structure.validate_replay_structures(dst_structure, src_structure)?;
    let descriptor = structure.descriptor();

    #[cfg(test)]
    if let Some(hook) = crate::probe::take_trace_raw_execution_hook() {
        hook()?;
    }
    if let Some(data) = try_tensortrace_owned_raw(
        dst_structure,
        dst_nout,
        src_structure,
        src_data,
        descriptor.source_conjugate(),
        descriptor.terms().len(),
        descriptor.destination_producer_indices(),
        descriptor.destination_producer_offsets(),
        |term_index| {
            let (term, fusion_term) = structure.paired_term(term_index);
            OwnedTraceTerm::new(
                term.dst_block,
                term.src_block,
                descriptor.output_shape(term),
                descriptor.trace_shape(term),
                descriptor.src_output_strides(term),
                descriptor.src_trace_strides(term),
                fusion_term.coefficient,
            )
        },
        alpha,
    )? {
        return Ok(data);
    }

    let required_len = dst_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); required_len];
    replay_fusion_trace_host(
        exec,
        structure,
        dst_structure,
        &mut data,
        src_structure,
        src_data,
        alpha,
        D::zero(),
    )?;
    Ok(data)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
/// Executes compiled trace terms into `dst_data` (the execution half of the
/// trace-into entries).
#[doc(hidden)]
pub(crate) fn tensortrace_fusion_dyn_structure_into_raw<C, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    structure: &TensorTraceFusionStructure<C>,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + Send
        + Sync,
{
    // Why: a public executor must not replay terms compiled for other spaces;
    // the pointer-equality fast path keeps compiled-in-place callers free.
    replay_fusion_trace_host(
        exec,
        structure,
        dst_space.structure(),
        dst_data,
        src_space.structure(),
        src_data,
        alpha,
        beta,
    )
}

/// The Host kernel of a staged owned trace: `alpha · trace(src)` into a new
/// payload of the compiled destination (`dst_nout` its codomain rank).
#[doc(hidden)]
pub fn tensortrace_fusion_structure_owned_host<C, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    structure: &TensorTraceFusionStructure<C>,
    dst_nout: usize,
    src_structure: &Arc<BlockStructure>,
    src_data: &[D],
    alpha: D,
) -> Result<Vec<D>, OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + Send
        + Sync,
{
    trace_structure_owned_parts(
        exec,
        structure,
        &structure.dst_structure,
        dst_nout,
        src_structure,
        src_data,
        alpha,
    )
}

/// The Host kernel of a staged trace into an existing destination:
/// `dst = alpha · trace(src) + beta · dst` over the compiled destination.
#[doc(hidden)]
pub fn tensortrace_fusion_structure_into_host<C, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    structure: &TensorTraceFusionStructure<C>,
    dst_data: &mut [D],
    src_structure: &Arc<BlockStructure>,
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + Send
        + Sync,
{
    replay_fusion_trace_host(
        exec,
        structure,
        &structure.dst_structure,
        dst_data,
        src_structure,
        src_data,
        alpha,
        beta,
    )
}

/// Preflights finite-label trace metadata: the braiding gate, the axes and
/// the selected result HomSpace, with the pair duality left to
/// [`TracePreflight::require_dual_pairs`].
///
/// The user facade calls this before materializing source data or constructing
/// a result layout, so a checked algebra failure cannot publish either derived
/// state first.
#[doc(hidden)]
pub fn tensortrace_fusion_dyn_preflight_checked<R>(
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Result<TracePreflight, OperationError>
where
    R: FusionRule + CheckedFusionAlgebra,
{
    trace_preflight::<MultiplicityFreeAdmissionMode, R>(src, axes, dst_nout)
}

/// Checked lowered trace entry point. The unchecked dynamic API intentionally
/// keeps its `MultiplicityFreeRigidSymbols` bound for custom rules; lowered
/// built-in callers use this wrapper to validate metadata before compilation.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn tensortrace_fusion_dyn_into_checked_raw<R, D>(
    exec: symtn_operations::host_pool::Exec<'_>,
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    src_space: &DynamicFusionMapSpace,
    src_data: &[D],
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + Send
        + Sync,
{
    let structure = TensorTraceFusionStructure::compile_fusion_dyn_checked_raw(
        rule, dst_space, src_space, axes,
    )?;
    tensortrace_fusion_dyn_structure_into_raw(
        exec, &structure, dst_space, dst_data, src_space, src_data, alpha, beta,
    )
}

fn validate_trace_data_extents(
    dst_structure: &BlockStructure,
    dst_len: usize,
    src_structure: &BlockStructure,
    src_len: usize,
) -> Result<(), OperationError> {
    let expected_dst = dst_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if dst_len != expected_dst {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_dst,
            actual: dst_len,
        });
    }
    let expected_src = src_structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if src_len != expected_src {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_src,
            actual: src_len,
        });
    }
    Ok(())
}

fn scale_trace_destination_layouts<T>(
    mut exec: symtn_operations::host_pool::Exec<'_>,
    descriptor: &TensorTraceDescriptor,
    dst: &mut [T],
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if beta.is_one() {
        return Ok(());
    }
    if beta.is_zero() {
        let zero = [T::zero()];
        for layout in descriptor.destination_layouts() {
            axpby_raw_strided_kernel_trusted(
                exec.reborrow(),
                dst,
                &zero,
                descriptor.destination_shape(layout),
                descriptor.destination_strides(layout),
                descriptor.destination_zero_strides(layout),
                layout.offset,
                0,
                T::one(),
                beta,
            )?;
        }
        return Ok(());
    }
    for layout in descriptor.destination_layouts() {
        scale_raw_strided_kernel_trusted(
            dst,
            descriptor.destination_shape(layout),
            descriptor.destination_strides(layout),
            layout.offset,
            beta,
        )?;
    }
    Ok(())
}

fn mark_axes(
    tensor: &'static str,
    axes: &[usize],
    rank: usize,
    seen: &mut symtn_core::axes::AxisMask,
) -> Result<(), OperationError> {
    seen.insert_all(axes)
        .map_err(|_| OperationError::InvalidAxisSet {
            tensor,
            axes: axes.to_vec(),
            rank,
        })
}

fn stride_to_isize(stride: usize) -> Result<isize, OperationError> {
    isize::try_from(stride).map_err(|_| OperationError::StrideOverflow { value: stride })
}
