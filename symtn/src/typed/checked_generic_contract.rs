use super::*;

impl<R, D> TypedTensorTraceDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    fn trace<S: TraceExec<R, D>>(
        source: &TensorMap<R, D, S>,
        read: impl FnOnce(&symtn_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        axes: symtn_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        let space = &source.storage_body().space;
        let stage = symtn_tensors::tensortrace_stage_multiplicity_free(space, axes, dst_nout)?;
        symtn_tensors::tensortrace_multiplicity_free_with(
            stage,
            space,
            axes,
            read,
            |structure, compact, nout| {
                S::trace_owned(EXEC, structure, nout, compact, source, D::from_real(1.0))
                    .map_err(Error::from)
            },
        )
    }

    fn trace_into<S: TraceExec<R, D>>(
        source: &TensorMap<R, D, S>,
        read: impl FnOnce(&symtn_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        axes: symtn_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let destination_space = destination.logical_space().space().clone();
        symtn_tensors::tensortrace_into_multiplicity_free_with(
            &source.storage_body().space,
            read,
            axes,
            dst_nout,
            &destination_space,
            || destination_storage_mut(destination),
            |structure, compact, destination| {
                S::trace_into(EXEC, structure, compact, source, destination, alpha, beta)
                    .map_err(Error::from)
            },
        )
    }
}

impl<R, D> TypedTensorTraceDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar + symtn_tensors::RecouplingCoefficientAction<f64>,
{
    fn trace<S: TraceExec<R, D>>(
        source: &TensorMap<R, D, S>,
        read: impl FnOnce(&symtn_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        axes: symtn_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError> {
        let space = &source.storage_body().space;
        let stage = symtn_tensors::tensortrace_stage_checked_generic(space, axes, dst_nout)?;
        symtn_tensors::tensortrace_checked_generic_with(
            stage,
            space,
            axes,
            read,
            |structure, compact, nout| {
                S::trace_owned(EXEC, structure, nout, compact, source, D::from_real(1.0))
                    .map_err(checked_generic_exec_error)
            },
        )
    }

    fn trace_into<S: TraceExec<R, D>>(
        source: &TensorMap<R, D, S>,
        read: impl FnOnce(&symtn_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        axes: symtn_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
    ) -> Result<(), Self::FacadeError> {
        let destination_space = destination.logical_space().space().clone();
        symtn_tensors::tensortrace_into_checked_generic_with(
            &source.storage_body().space,
            read,
            axes,
            dst_nout,
            &destination_space,
            || destination_storage_mut(destination).map_err(Self::FacadeError::from),
            |structure, compact, destination| {
                S::trace_into(EXEC, structure, compact, source, destination, alpha, beta)
                    .map_err(checked_generic_exec_error)
            },
        )
    }
}

/// The storage kernels of the one `trace_pairs` / `trace_pairs_into` body:
/// admission, staging, the compile, the destination checks and the commit
/// are shared; a kernel only replays the compiled terms on its storage.
///
/// `compact` is the compact arm's answer for a single-scalar destination
/// (a compact spectrum traced over its only pair, read through
/// [`TypedStorage::as_host`]); it is always `None` on a storage without
/// compact data.
#[doc(hidden)]
pub trait TraceExec<R, D>: ElementwiseExec<R, D> {
    /// `alpha · trace(source)` into a fresh payload of `structure`'s destination
    /// (`dst_nout` its codomain rank), reading `source`'s stored payload (a
    /// lazy adjoint's parent).
    fn trace_owned(
        _: ExecToken,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
        dst_nout: usize,
        compact: Option<D>,
        source: &TensorMap<R, D, Self>,
        alpha: D,
    ) -> Result<Self, TransformExecError>;

    /// `destination = alpha · trace(source) + beta · destination` into a
    /// destination that passed every admission check, with TensorKit's
    /// `beta` pass over every destination layout first.
    fn trace_into(
        _: ExecToken,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
        compact: Option<D>,
        source: &TensorMap<R, D, Self>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>;
}

/// The Host kernels: the strided partial-trace replay inside the Runtime's
/// Host pool, where a compact source other than the scalar arm is
/// materialized.
impl<R, D: TensorScalar> TraceExec<R, D> for Vec<D> {
    fn trace_owned(
        _: ExecToken,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
        dst_nout: usize,
        compact: Option<D>,
        source: &TensorMap<R, D>,
        alpha: D,
    ) -> Result<Self, TransformExecError> {
        if let Some(value) = compact {
            return Ok(vec![value * alpha]);
        }
        let _host_pool = source.runtime.enter_host_pool();
        let body = source.storage_body();
        symtn_tensors::tensortrace_fusion_structure_owned_host(
            structure,
            dst_nout,
            body.space.space().structure(),
            &body.materialized_dense_data(),
            alpha,
        )
        .map_err(TransformExecError::Replay)
    }

    fn trace_into(
        _: ExecToken,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
        compact: Option<D>,
        source: &TensorMap<R, D>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError> {
        if let (Some(value), [slot]) = (compact, destination.as_mut_slice()) {
            *slot = if beta == D::zero() {
                value * alpha
            } else {
                value * alpha + beta * *slot
            };
            return Ok(());
        }
        let _host_pool = source.runtime.enter_host_pool();
        let body = source.storage_body();
        symtn_tensors::tensortrace_fusion_structure_into_host(
            structure,
            destination,
            body.space.space().structure(),
            &body.materialized_dense_data(),
            alpha,
            beta,
        )
        .map_err(TransformExecError::Replay)
    }
}

/// The axis lists a validated `trace_pairs` pair list stands for.
pub(super) struct TracePairAxes {
    pub(super) output_axes: Vec<usize>,
    pub(super) destination_codomain_rank: usize,
    pub(super) trace_lhs: Vec<usize>,
    pub(super) trace_rhs: Vec<usize>,
}

/// Validates a `trace_pairs` pair list and derives the untraced output axes in
/// order; `None` for an empty list, whose trace is a clone. Shared by every
/// storage so the Host and device reject a malformed list with one message.
///
/// Why this validation is kept rather than left to the seam: `seen` is not a
/// check, it is the derivation of `output_axes` — the seam cannot supply it,
/// and a malformed list would otherwise produce a silently wrong output order
/// rather than an error. Same precedent as `braid`'s levels pre-check.
fn trace_pair_axes(
    rank: usize,
    codomain_rank: usize,
    pairs: &[(usize, usize)],
) -> Result<Option<TracePairAxes>, Error> {
    let mut seen = symtn_core::axes::AxisMask::new(rank);
    for &(lhs, rhs) in pairs {
        if seen.insert(lhs).and_then(|()| seen.insert(rhs)).is_err() {
            // The traced axes are an axis subset, reported as every other
            // subset misuse (#1873), with the pairs flattened in order.
            return Err(symtn_tensors::OperationError::InvalidAxisSet {
                tensor: "trace pairs",
                axes: pairs.iter().flat_map(|&(lhs, rhs)| [lhs, rhs]).collect(),
                rank,
            }
            .into());
        }
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    // The pairs are validated distinct, so exactly `rank - 2·pairs` axes stay
    // open; the complement iterator carries no size hint.
    let mut output_axes = Vec::with_capacity(rank - 2 * pairs.len());
    output_axes.extend(seen.complement());
    let destination_codomain_rank = output_axes
        .iter()
        .filter(|&&axis| axis < codomain_rank)
        .count();
    Ok(Some(TracePairAxes {
        output_axes,
        destination_codomain_rank,
        trace_lhs: pairs.iter().map(|&(lhs, _)| lhs).collect(),
        trace_rhs: pairs.iter().map(|&(_, rhs)| rhs).collect(),
    }))
}

/// A trace's source in storage orientation: an owned body as it is, or a
/// lazy adjoint's parent with the pair axes mapped onto the parent's legs
/// and the source conjugated. Shared by every trace entry point and
/// placement, so they lower an adjoint identically.
pub(super) struct TraceSource<'t, R, D, S> {
    pub(super) body: &'t Arc<TypedTensorBody<R, D, S>>,
    pub(super) axes: TracePairAxes,
    pub(super) conjugate: bool,
}

impl<R, D, S> TraceSource<'_, R, D, S> {
    pub(super) fn spec(&self) -> symtn_tensors::TensorTraceAxisSpec<'_> {
        symtn_tensors::TensorTraceAxisSpec::new_with_conjugation(
            &self.axes.output_axes,
            &self.axes.trace_lhs,
            &self.axes.trace_rhs,
            self.conjugate,
        )
    }
}

/// Gates the provider's braiding, validates `pairs` and lowers `tensor` to
/// its trace source; `None` for an empty list.
///
/// The braiding gate comes first, as in TensorKit `trace_permute!`, which
/// rejects a non-symmetric braiding before it looks at the indices, even for
/// an empty trace.
pub(super) fn trace_source<'t, R, D, S>(
    tensor: &'t TensorMap<R, D, S>,
    braiding: symtn_core::BraidingStyleKind,
    pairs: &[(usize, usize)],
) -> Result<Option<TraceSource<'t, R, D, S>>, Error> {
    symtn_tensors::require_symmetric_braiding(braiding, symtn_tensors::SymmetricBraidingOp::Trace)?;
    let Some(mut axes) = trace_pair_axes(tensor.rank(), tensor.codomain_rank(), pairs)? else {
        return Ok(None);
    };
    Ok(Some(match &tensor.repr {
        TypedTensorRepr::Owned(body) => TraceSource {
            body,
            axes,
            conjugate: false,
        },
        TypedTensorRepr::Adjoint(view) => {
            let parent = view.parent.space.space();
            let (nout, nin) = (parent.nout(), parent.nin());
            for list in [
                &mut axes.output_axes,
                &mut axes.trace_lhs,
                &mut axes.trace_rhs,
            ] {
                *list = logical_adjoint_axes_to_parent(nout, nin, list);
            }
            TraceSource {
                body: &view.parent,
                axes,
                conjugate: true,
            }
        }
    }))
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTraceDispatch<R, D>,
    D: TensorScalar,
    S: TraceExec<R, D>,
{
    /// Traces mutually dual axis pairs and returns the tensor on the remaining
    /// legs, preserving their order and codomain/domain side.
    ///
    /// Each pair uses flat zero-based axes with codomain axes first. Every axis
    /// must be in range and appear at most once; past the braiding gate,
    /// `pairs.is_empty()` returns a clone. This is the categorical contraction trace, so provider pivotal
    /// coefficients and fermionic twists are included. It can therefore be a
    /// supertrace and need not equal [`Self::tr`].
    ///
    /// Checked Generic requires [`crate::sector::CheckedGenericPivotal`] and
    /// admits the output with the source's exact provider `Arc`. Lazy
    /// adjoints are read through their parent without materializing.
    ///
    /// # Complexity
    ///
    /// Dense storage runs the partial-trace engine over the whole payload. A
    /// compact spectrum traced over its only pair reduces the stored spectrum
    /// in `O(Σ_c k_c)` without materializing, in both admission modes (#604,
    /// #1866): the destination is the empty tree, so each source block
    /// contributes its compiled trace coefficient times its spectrum sum —
    /// the coefficients the dense route replays, equal to it within dtype
    /// tolerance. Other compact cases materialize inside the runtime's Host
    /// pool.
    ///
    /// # Errors
    ///
    /// In TensorKit `trace_permute!`'s order: `UnsupportedTensorContractScope`
    /// for a non-symmetric braiding (even with an empty `pairs`), then
    /// [`Error::Operation`] with `InvalidAxisSet { tensor: "trace pairs" }`
    /// for an out-of-range or repeated axis, then
    /// `StructureMismatch { "trace axes" }` for legs that are not mutually
    /// dual, in both admission modes. Trace execution failures return their
    /// corresponding [`Error`] variant. If required pivotal data is unavailable, the original
    /// provider error is available as the source. On CUDA storage,
    /// [`Error::PlacementMismatch`] last. A failed trace returns no output
    /// tensor.
    ///
    /// # CUDA storage
    ///
    /// The same body, compile and terms: the device only replays them, one
    /// contraction per term, reading the source block through one merged
    /// diagonal axis per pair (extent `t_k`, stride `s_lhs + s_rhs`; pairs are
    /// never merged with each other) against the coefficient repeated (the
    /// context ones template for a unit coefficient, the scaled template
    /// otherwise), so each traced element is scaled before the sum as on the
    /// Host, accumulated (`beta = 1`) into the zero-initialised output, so a
    /// destination block with several producers receives their sum. A lazy
    /// adjoint traces its parent with the Host's parent axes and a conjugated
    /// read.
    ///
    /// A warm call transfers only the #740 output initialisation: one H2D of
    /// `required_len * size_of::<D>()` bytes, one device allocation (the
    /// output), no download and no new cuTENSOR plan while the trace's
    /// distinct term signatures fit the plan-cache budget the transform
    /// executor raises the bound under (about 585 plans together with the
    /// prepared transforms; beyond it a warm call rebuilds plans). The first
    /// call whose largest traced extent `prod t_k` exceeds the resident ones
    /// or scaled template grows it once (one upload each, reported by
    /// `CudaTreeTransformStats::context_scalar_operand_bytes`).
    /// One submission per term with a non-zero coefficient, plus one device
    /// refill of the scaled template per distinct non-unit coefficient; FLOPs
    /// are the Host's `sum_terms |out| * prod t_k` multiply-adds.
    ///
    /// Device and Host agree to dtype tolerance, never bitwise: the sum order
    /// is cuTENSOR's. Both scale each traced element by `alpha * c` before
    /// the sum and skip a zero coefficient, as TensorKit's `_trace_permute!`
    /// and TensorOperations' `tensortrace!` do (#1438). A rejected call
    /// leaves the device and the Runtime unchanged.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// assert_eq!(id.trace_pairs(&[(0, 1)])?.scalar()?, 2.0);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn trace_pairs(&self, pairs: &[(usize, usize)]) -> Result<Self, FacadeError<R>> {
        let braiding = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider());
        let Some(source) = trace_source(self, braiding, pairs)? else {
            return Ok(self.clone());
        };
        let (space, data) = <R::Mode as TypedTensorTraceDispatch<R, D>>::trace(
            self,
            |structure| self.compact_full_trace(structure),
            source.spec(),
            source.axes.destination_codomain_rank,
        )?;
        Ok(self.published(space, data))
    }

    /// `destination = alpha * self.trace_pairs(pairs) + beta * destination`,
    /// TensorKit `tensortrace!(C, A, p, q, false, α, β)`, in every admission
    /// mode that has [`Self::trace_pairs`].
    ///
    /// TensorKit's `_trace_permute!` order: every destination layout becomes
    /// `beta * destination` first (a strong zero for `beta = 0`, which never
    /// reads the destination; nothing for `beta = 1`), then every trace term
    /// adds `alpha * coefficient * trace(block)`. That `beta` pass is the
    /// reference's own: several source blocks feed one destination block, so
    /// no single term's write can carry it. Past the braiding gate, an empty
    /// `pairs` is exactly [`Self::axpby_into`], with its rules and errors.
    ///
    /// It admits exactly the sources [`Self::trace_pairs`] admits, through
    /// the same staging and compiled terms: a lazy adjoint is read through
    /// its parent, and a compact (diagonal) source traced over its only pair
    /// writes `alpha` times its spectrum's compiled trace without
    /// materializing (`O(Σ_c k_c)`).
    ///
    /// On CUDA storage the `beta` pass is a zero-source region move per
    /// destination layout (zeros for `beta = 0`, nothing for `beta = 1`),
    /// then every term accumulates as in [`Self::trace_pairs`].
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`], [`Error::RuleMismatch`];
    /// [`crate::typed::OperationError::UnsupportedTensorContractScope`] for a
    /// non-symmetric braiding, even with an empty `pairs`; the pair-list errors
    /// of [`Self::trace_pairs`]; [`Error::InvalidArgument`] for a destination
    /// that is not owned dense storage of its placement or aliases the
    /// source's storage (a lazy adjoint's parent included); then, in TensorKit
    /// `trace_permute!`'s order, the staging errors of [`Self::trace_pairs`]
    /// (checked Generic: provider and preflight errors), `SpaceMismatch` for
    /// a destination whose HomSpace is not the result's, the duality error of
    /// [`Self::trace_pairs`], and `SpaceMismatch` for a destination whose
    /// layout is not the result's; then [`Error::InvalidArgument`] for the
    /// wrong length and [`Error::DestinationShared`]; on CUDA storage,
    /// [`Error::PlacementMismatch`] last.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write, and a checked trace
    /// publishes its staged layout and coefficients only after the write
    /// succeeds. On failure after validation (a backend or other execution
    /// error) the destination's contents are unspecified, while its space and
    /// block structure stay intact.
    pub fn trace_pairs_into(
        &self,
        pairs: &[(usize, usize)],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        // The held admitted identities: provider-free, and the provider's
        // own by #2046.
        if self.logical_space().space().admission().rule_identity()
            != destination
                .logical_space()
                .space()
                .admission()
                .rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        let braiding = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider());
        let Some(source) = trace_source(self, braiding, pairs)? else {
            return Ok(axpby_into_checked(self, destination, alpha, beta)?);
        };
        // Why the storage body: a lazy adjoint is read through its parent.
        destination_precheck(destination, &source.body.data, S::NOUN)?;
        <R::Mode as TypedTensorTraceDispatch<R, D>>::trace_into(
            self,
            |structure| self.compact_full_trace(structure),
            source.spec(),
            source.axes.destination_codomain_rank,
            destination,
            alpha,
            beta,
        )
    }

    /// The compact arm: a compact spectrum's full trace into a
    /// single-scalar destination, from the compiled terms alone.
    fn compact_full_trace(
        &self,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
    ) -> Option<D> {
        let spectrum = S::as_host(EXEC, self)?.spectrum()?;
        compact_arms::full_trace_terms(
            structure,
            self.storage_body().space.space().structure(),
            spectrum,
        )
    }
}

/// The multiplicity-free owned contraction or composition on any storage,
/// one body for every coefficient lane (TensorKit `tensorcontract!` has no
/// symbol-type dispatch): the staged destination, the lock-free canonical
/// core for the storage's executor, else the rest of the planner under one
/// context lease, the storage's kernel, then the commit.
#[allow(private_bounds)]
fn contract_owned_multiplicity_free<'a, R, D, S>(
    lhs: &'a TensorMap<R, D, S>,
    rhs: &'a TensorMap<R, D, S>,
    request: impl FnOnce(
        symtn_tensors::FusionOperand<'a>,
        symtn_tensors::FusionOperand<'a>,
    ) -> symtn_tensors::ContractRequest<'a>,
) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    R::Scalar: CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar + crate::runtime::MultiplicityFreeCoefficientLane<R::Scalar>,
    S: ContractExec<R, D>,
{
    let (lhs_operand, rhs_operand) = (lhs.fusion_operand(), rhs.fusion_operand());
    let mut staged = symtn_tensors::StagedContract::multiplicity_free(
        (
            lhs.logical_space(),
            lhs_operand,
            transform_source_len(lhs.storage_body())?,
        ),
        (
            rhs.logical_space(),
            rhs_operand,
            transform_source_len(rhs.storage_body())?,
        ),
        request(lhs_operand, rhs_operand),
    )?;
    // Why the lease waits for a core miss: the canonical core resolves from
    // the operands alone, and the device kernel must not hold a context
    // lease; the Host kernel takes the one lease when planning took none.
    let (lease, resolution) = match staged.try_core::<S::Caps>()? {
        symtn_tensors::CoreRoute::Hit(core) => (None, core),
        symtn_tensors::CoreRoute::Miss(miss) => {
            let mut lease = lhs.runtime.lease_context()?;
            let resolution = D::lane(lease.context())?
                .plan_staged_contract_beyond_core::<S::Caps, R>(&mut staged, miss)?;
            (Some(lease), resolution)
        }
    };
    let data = S::contract_owned(
        EXEC,
        ContractReplay::new::<MultiplicityFreeTreeLane>(
            &lhs.runtime,
            lease,
            &resolution,
            (staged.destination(), staged.destination_len()),
            (lhs_operand, rhs_operand),
        ),
        lhs,
        rhs,
    )?;
    Ok((staged.commit(&resolution)?, data))
}

#[allow(private_bounds)]
fn contract_multiplicity_free<R, D, S>(
    lhs: &TensorMap<R, D, S>,
    rhs: &TensorMap<R, D, S>,
    spec: &ContractSpec<'_>,
    planar: Option<&symtn_tensors::PlanarContractOrders>,
) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    R::Scalar: CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar + crate::runtime::MultiplicityFreeCoefficientLane<R::Scalar>,
    S: ContractExec<R, D>,
{
    let output_axes = spec.output_axes();
    #[cfg(test)]
    super::probe::hit(super::probe::Event::ContractSeamCall);
    contract_owned_multiplicity_free(lhs, rhs, |lhs, rhs| {
        contract_request(lhs, rhs, spec, &output_axes, planar)
    })
}

/// [`contract_multiplicity_free`] written into an existing destination: one
/// lease first (leasing counts as validation), the same staging, the staged
/// space against the destination's, its length and ownership, planning, the
/// storage's write, then the commit.
#[allow(private_bounds)]
fn contract_into_multiplicity_free<R, D, S>(
    lhs: &TensorMap<R, D, S>,
    rhs: &TensorMap<R, D, S>,
    spec: &ContractSpec<'_>,
    planar: Option<&symtn_tensors::PlanarContractOrders>,
    destination: &mut TensorMap<R, D, S>,
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    R::Scalar: CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar + crate::runtime::MultiplicityFreeCoefficientLane<R::Scalar>,
    S: ContractExec<R, D>,
{
    let output_axes = spec.output_axes();
    let mut lease = lhs.runtime.lease_context()?;
    D::lane(lease.context())?;
    let (lhs_operand, rhs_operand) = (lhs.fusion_operand(), rhs.fusion_operand());
    let mut staged = symtn_tensors::StagedContract::multiplicity_free(
        (
            lhs.logical_space(),
            lhs_operand,
            transform_source_len(lhs.storage_body())?,
        ),
        (
            rhs.logical_space(),
            rhs_operand,
            transform_source_len(rhs.storage_body())?,
        ),
        contract_request(lhs_operand, rhs_operand, spec, &output_axes, planar),
    )?;
    staged.require_destination(destination.logical_space().space())?;
    let destination = destination_storage_mut(destination)?;
    let resolution = match staged.try_core::<S::Caps>()? {
        symtn_tensors::CoreRoute::Hit(core) => core,
        symtn_tensors::CoreRoute::Miss(miss) => D::lane(lease.context())?
            .plan_staged_contract_beyond_core::<S::Caps, R>(&mut staged, miss)?,
    };
    S::contract_into(
        EXEC,
        ContractReplay::new::<MultiplicityFreeTreeLane>(
            &lhs.runtime,
            Some(lease),
            &resolution,
            (staged.destination(), staged.destination_len()),
            (lhs_operand, rhs_operand),
        ),
        lhs,
        rhs,
        destination,
        alpha,
        beta,
    )?;
    staged.commit(&resolution)?;
    Ok(())
}

impl<R, D> MultiplicityFreeContractExecution<R> for D
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    R::Scalar: CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar + crate::runtime::MultiplicityFreeCoefficientLane<R::Scalar>,
{
    fn contract<S: ContractExec<R, Self>>(
        lhs: &TensorMap<R, Self, S>,
        rhs: &TensorMap<R, Self, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        contract_multiplicity_free(lhs, rhs, spec, planar)
    }

    fn contract_into<S: ContractExec<R, Self>>(
        lhs: &TensorMap<R, Self, S>,
        rhs: &TensorMap<R, Self, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
        destination: &mut TensorMap<R, Self, S>,
        alpha: Self,
        beta: Self,
    ) -> Result<(), Error> {
        contract_into_multiplicity_free(lhs, rhs, spec, planar, destination, alpha, beta)
    }

    fn compose<S: ContractExec<R, Self>>(
        lhs: &TensorMap<R, Self, S>,
        rhs: &TensorMap<R, Self, S>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        contract_owned_multiplicity_free(lhs, rhs, |_, _| symtn_tensors::ContractRequest::Compose)
    }
}

/// Writes TensorKit's `one!` into every coupled-sector block: `δ_ij` with
/// rows and columns taken in TensorKit `fusiontrees` order, so the partial
/// identity of a rectangular block selects the trees TensorKit selects.
///
/// `order_key(sector, dual)` is the provider's `sector_order_key` of `sector`,
/// or of its dual when `dual` is set.
pub(super) fn write_identity_blocks_generic<R, D, E>(
    tensor: &mut TensorMap<R, D>,
    order_key: impl FnMut(SectorId, bool) -> Result<symtn_core::SectorOrderKey, E>,
) -> Result<(), E>
where
    D: TensorScalar,
    E: From<Error>,
{
    let TypedTensorRepr::Owned(body) = &mut tensor.repr else {
        unreachable!()
    };
    write_dense_identity_blocks(Arc::get_mut(body).expect("fresh identity body"), order_key)
}

fn write_dense_identity_blocks<R, D, E>(
    body: &mut TypedTensorBody<R, D>,
    mut order_key: impl FnMut(SectorId, bool) -> Result<symtn_core::SectorOrderKey, E>,
) -> Result<(), E>
where
    D: TensorScalar,
    E: From<Error>,
{
    let regions = {
        let space = body.space.space();
        sector_regions(space.structure(), space.nout())?
    };
    let payload = Arc::get_mut(&mut body.data).expect("fresh identity payload");
    let TypedData::Dense(data) = payload else {
        unreachable!()
    };
    for region in regions.iter() {
        let rows = tensorkit_tree_order(region.row_trees(), &mut order_key)?;
        let cols = tensorkit_tree_order(region.col_trees(), &mut order_key)?;
        let start = region.range().start;
        for (row, col) in rows.into_iter().flatten().zip(cols.into_iter().flatten()) {
            data[start + row + col * region.rows()] = D::from_real(1.0);
        }
    }
    Ok(())
}

/// The matrix index ranges of `trees` (one side of a coupled block, in
/// storage order) listed in TensorKit `fusiontrees` order.
///
/// TensorKit 0.17.1 `FusionTreeIterator` (`src/fusiontrees/iterator.jl`,
/// `_fusiontree_iterate`) varies the last uncoupled sector slowest, then the
/// inner line feeding the top vertex, then that vertex, then recurses into
/// the remaining tree; for two legs `(u₂, u₁, μ)`. Uncoupled sectors iterate
/// as `sectors(V)`, which for a dual leg is `dual(s)` in `V`'s order; inner
/// lines iterate as `c ⊗ dual(b)`, which is sector order for every simple
/// TensorKitSectors type but not for every product or SU(N) (see
/// `docs/sector_id_compatibility.md`). Within a tree, indices stay in storage order, which
/// is TensorKit's column-major subblock order.
fn tensorkit_tree_order<E>(
    trees: &[CoupledTreeExtent],
    order_key: &mut impl FnMut(SectorId, bool) -> Result<symtn_core::SectorOrderKey, E>,
) -> Result<Vec<core::ops::Range<usize>>, E>
where
    E: From<Error>,
{
    let mut ranked = Vec::with_capacity(trees.len());
    for extent in trees {
        let tree = extent.tree();
        let (uncoupled, lines, vertices) = (tree.uncoupled(), tree.innerlines(), tree.vertices());
        let vertex = |k: usize| symtn_core::SectorOrderKey::position(vertices[k].get() as u64);
        let mut key = Vec::with_capacity(3 * uncoupled.len());
        for leg in (1..uncoupled.len()).rev() {
            key.push(order_key(uncoupled[leg], tree.is_dual()[leg])?);
            if leg >= 2 {
                key.push(order_key(lines[leg - 2], false)?);
            } else {
                key.push(order_key(uncoupled[0], tree.is_dual()[0])?);
            }
            key.push(vertex(leg - 1));
        }
        let len = extent
            .extent()
            .map_err(|error| E::from(Error::from(error)))?;
        ranked.push((key, extent.offset()..extent.offset() + len));
    }
    ranked.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(ranked.into_iter().map(|(_, range)| range).collect())
}

#[cfg(test)]
mod copy_c_scratch_tests {
    use std::sync::Arc;

    use crate::typed::probe::{self, Event};

    use symtn_core::{U1FusionRule, U1Irrep};

    use crate::typed::{ContractSpec, GradedSpace, Runtime, TensorMap, TypedData, TypedTensorRepr};

    /// Grows the pooled copyC scratch to a length no temporary here reaches;
    /// a shorter length afterwards proves the copyC route ran.
    const MARKER_LEN: usize = 100_000;

    fn mark_scratch(runtime: &Runtime) {
        let mut lease = runtime.lease_context().unwrap();
        let lane = lease.context().multiplicity_free_lane::<f64>().unwrap();
        let mut scratch = lane.take_copy_c_scratch();
        scratch.resize_filled(MARKER_LEN, 0.0);
        lane.restore_copy_c_scratch(scratch);
    }

    fn retained_len(runtime: &Runtime) -> usize {
        let mut lease = runtime.lease_context().unwrap();
        lease
            .context()
            .multiplicity_free_lane::<f64>()
            .unwrap()
            .copy_c_scratch_len()
    }

    /// The copyC temporary lives in the leased context and is sized by the
    /// latest temporary, so a Runtime retains at most one largest temporary
    /// per idle context — the same bound as the context's other scratch.
    #[test]
    fn copy_c_scratch_holds_the_latest_temporary_only() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let space =
            |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 2],
            domain: &[1],
        };
        let run = |n: usize| {
            let (p, c) = (space(n), space(3));
            let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&p, &p], [&c], 1)
                .unwrap();
            let b = TensorMap::rand_with_seed(&runtime, [&c], [&p], 2).unwrap();
            drop(a.contract(&b, &spec).unwrap());
            retained_len(&runtime)
        };
        assert_eq!(retained_len(&runtime), 0);
        assert_eq!(run(5), 125);
        assert_eq!(run(7), 343);
        assert_eq!(run(5), 125);
    }

    /// The pooled temporary is written with a strong-zero `Axpby(0)`, so a
    /// stale buffer never leaks: a NaN-poisoned buffer longer than the
    /// temporary (so resizing writes nothing) still gives exactly the
    /// contract-then-permute result, including the temporary's inactive block.
    #[test]
    fn stale_copy_c_scratch_never_reaches_the_result() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        // `c` carries charge 0 only, so the temporary `p ⊗ q ← r` block of
        // coupled charge 1 receives no GEMM contribution.
        let open = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
        )
        .unwrap();
        let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 4)]).unwrap();
        let a =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&open, &open], [&bond], 3)
                .unwrap();
        let b = TensorMap::rand_with_seed(&runtime, [&bond], [&open], 4).unwrap();
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 2],
            domain: &[1],
        };
        let expected = a
            .contract(
                &b,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2],
                },
            )
            .unwrap()
            .permute(&[0, 2], &[1])
            .unwrap();
        const POISONED_LEN: usize = 100_000;
        {
            let mut lease = runtime.lease_context().unwrap();
            let lane = lease.context().multiplicity_free_lane::<f64>().unwrap();
            let mut scratch = lane.take_copy_c_scratch();
            scratch.resize_filled(POISONED_LEN, f64::NAN);
            lane.restore_copy_c_scratch(scratch);
        }
        let actual = a.contract(&b, &spec).unwrap();
        // What: the copyC route ran on the poisoned buffer.
        assert!(retained_len(&runtime) < POISONED_LEN);
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }

    /// The partial-coverage fixture of `stale_copy_c_scratch_never_reaches_the_result`:
    /// `c` carries charge 0 only, so the result's coupled-charge-1 block of
    /// `(p, r | q)` is reached by no GEMM.
    fn partial_copy_c_fixture(
        runtime: &Runtime,
    ) -> (TensorMap<U1FusionRule, f64>, TensorMap<U1FusionRule, f64>) {
        let provider = Arc::new(U1FusionRule);
        let open = GradedSpace::try_new(
            Arc::clone(&provider),
            [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
        )
        .unwrap();
        let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 4)]).unwrap();
        (
            TensorMap::rand_with_seed(runtime, [&open, &open], [&bond], 3).unwrap(),
            TensorMap::rand_with_seed(runtime, [&bond], [&open], 4).unwrap(),
        )
    }

    const MOVED: ContractSpec<'static> = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 2],
        domain: &[1],
    };

    /// #1631: on the copyC route the output transform writes every element,
    /// so an unreached one becomes `alpha * (+0) + beta * dst` — a `-0.0`
    /// becomes `+0.0` under `beta = 1` and an infinite `alpha` gives NaN, as
    /// TensorKit's `tensoradd!` — while `alpha = 0` still never reads the
    /// operands.
    #[test]
    fn copy_c_contract_into_writes_unreached_elements_through_the_transform() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let (a, b) = partial_copy_c_fixture(&runtime);
        let result = a.contract(&b, &MOVED).unwrap();
        let unreached: Vec<usize> = result
            .dense_data()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 0.0)
            .map(|(index, _)| index)
            .collect();
        assert!(!unreached.is_empty(), "the fixture must miss a sector");
        let run =
            |seed: f64, alpha: f64, beta: f64, operands: (&TensorMap<_, _>, &TensorMap<_, _>)| {
                let mut destination = result.scale(0.0).unwrap();
                let TypedTensorRepr::Owned(body) = &mut destination.repr else {
                    unreachable!("a scaled dense tensor is owned")
                };
                let Some(TypedData::Dense(data)) =
                    Arc::get_mut(body).and_then(|body| Arc::get_mut(&mut body.data))
                else {
                    unreachable!("a fresh scaled tensor is uniquely owned and dense")
                };
                data.fill(seed);
                mark_scratch(&runtime);
                operands
                    .0
                    .contract_into(operands.1, &MOVED, &mut destination, alpha, beta)
                    .unwrap();
                assert!(
                    retained_len(&runtime) < MARKER_LEN,
                    "the copyC route must run"
                );
                destination
            };
        let negative_zero = run(-0.0, 1.0, 1.0, (&a, &b));
        for &index in &unreached {
            let value = negative_zero.dense_data().unwrap()[index];
            assert_eq!(value.to_bits(), 0.0f64.to_bits(), "-0.0 at {index}");
        }
        let infinite = run(1.0, f64::INFINITY, 1.0, (&a, &b));
        for &index in &unreached {
            assert!(infinite.dense_data().unwrap()[index].is_nan(), "{index}");
        }
        let (nan_a, nan_b) = (a.scale(f64::NAN).unwrap(), b.scale(f64::NAN).unwrap());
        let silent = run(0.5, 0.0, 2.0, (&nan_a, &nan_b));
        assert!(silent
            .dense_data()
            .unwrap()
            .iter()
            .all(|&value| value == 1.0));
    }

    /// #1870: a
    /// reordered `t · D` writes the scaled `t` through one transform `_into`
    /// (TensorKit's `copyC` of a `mul!` with a `Diagonal`), so the compact
    /// operand is never densified and the contraction's copyC scratch is
    /// not used.
    #[test]
    fn reordered_compact_contract_into_scales_without_densifying() {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let space =
            |n| GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), n)]).unwrap();
        let (p, q, c) = (space(2), space(3), space(4));
        let t =
            TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&p, &q], [&c], 5).unwrap();
        let spectra = c
            .sectors()
            .unwrap()
            .into_iter()
            .map(|sector| crate::typed::SectorSpectrum {
                values: vec![1.5; c.degeneracy(&sector).unwrap()],
                sector,
            })
            .collect::<Vec<_>>();
        let d = TensorMap::diagonal(&runtime, &c, spectra).unwrap();
        let mut destination = t.contract(&d, &MOVED).unwrap().scale(1.0).unwrap();
        mark_scratch(&runtime);
        let densified = probe::record();
        t.contract_into(&d, &MOVED, &mut destination, 1.0, 0.0)
            .unwrap();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(retained_len(&runtime), MARKER_LEN, "no contraction ran");
        let expected = t.contract(&d.materialize().unwrap(), &MOVED).unwrap();
        assert_eq!(
            destination.dense_data().unwrap(),
            expected.dense_data().unwrap()
        );
    }
}
