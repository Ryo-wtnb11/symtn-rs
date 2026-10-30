use super::*;
use symtn_dense::Engine;

/// The factors of one [`EighFullPlan::execute`], borrowed from its workspace.
///
/// Member `i` of `d` and `v` is the eager `eigh_full` of the same placement
/// on member `i` of the source; `spectra[i]` is that member's eigenvalues per
/// coupled sector, in sector-label order, each ascending as in `d`.
/// `d` is stored dense: a stack holds dense payloads only, so the eager Host
/// compact diagonal is materialized.
pub struct EighStackOutput<'a, R: SectorCodec, D, S = Vec<D>> {
    /// The eigenvalues on the diagonal of `bond <- bond`.
    pub d: &'a StackedTensorMap<R, D, S>,
    /// The eigenvectors `codomain <- bond`, one column per eigenvalue.
    pub v: &'a StackedTensorMap<R, D, S>,
    /// Per member, per coupled sector: the eigenvalues, already on the host.
    pub spectra: &'a [Vec<SectorSpectrum<<R as SectorCodec>::Sector>>],
}

/// The Hermitian eigendecomposition `t = v d v†` (MatrixAlgebraKit
/// `eigh_full`) of every member of a stack, prepared once for one structure
/// signature (#1287). Real payloads only; a complex payload is rejected by
/// [`Self::new`] until Tenferro's batched complex solver works
/// (tenferro-rs#1923).
///
/// Per member it is the eager [`TensorMap::eigh_full`] of the same placement,
/// within that operation's gauge contract. Both placements compile the eager
/// plan (coupled-sector routes, the eigenvector and diagonal factor spaces,
/// each sector's eigenvector placement and `d` diagonal) once here from the
/// structure alone, since `eigh_full` keeps every eigenpair:
///
/// - **CUDA**: each call admits the whole batch (at most three downloads),
///   solves each coupled sector of all members with one batched solver
///   call, downloads every spectrum
///   once, orders each member ascending on the host as eager does (an O(n)
///   check: the solver already returns ascending values),
///   and assembles `v` by one batched column gather per coupled sector
///   followed by one strided copy per aligned sector (per codomain tree
///   otherwise). Submissions, host syncs and
///   downloads per call do not depend on the member count `B`. The raw
///   cuSOLVER gauge is kept, as eager keeps it; with `B > 1` the batched
///   solver may differ from the eager one in the last ULPs, so results are
///   deterministic for a fixed `B` but not bit-stable across `B`. At `B = 1`
///   the solver and host code are the eager ones.
/// - **Host**: after the whole batch is admitted, each member runs eager's
///   per-sector solver stage, gauge-fixed exactly as Host eager (the
///   largest-magnitude entry of each eigenvector real and positive), and is
///   written straight into its slice of the output stacks; results are bit
///   for bit eager's at every `B`.
///
/// # Supported scope
///
/// Endomorphisms with packed coupled-sector regions (every structure a
/// public constructor builds). A batch fails as a whole
/// ([`BatchError`]): on any error no output is returned, and none is
/// observable afterwards ([`EighFullWorkspace::take_output`] returns `None`)
/// until the next successful call, which rewrites every member.
///
/// # Retained state and shared effects
///
/// Each workspace owns its output stacks and host spectra, reported by
/// [`EighFullWorkspace::retained_bytes`] and freed on drop.
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{
///     EighFullPlan, GradedSpace, HermitianTol, Runtime, StackedTensorMap, TensorMap,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let engine = &mut symtn::typed::CpuEngine::serial();
/// let runtime = Runtime::builder().build()?;
/// let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
/// let stack = StackedTensorMap::pack(&[TensorMap::<_, f64>::zeros(engine,
///     &runtime, [&leg], [&leg],
/// )?])?;
/// let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT)?;
/// let mut workspace = plan.workspace()?;
/// let _output = plan.execute(engine, &stack, &mut workspace)?;
/// # Ok(())
/// # }
/// ```
pub struct EighFullPlan<R: SectorCodec, D, S = Vec<D>> {
    identity: Arc<()>,
    runtime: Runtime,
    source: StructureSignature,
    member_len: usize,
    /// The source's coupled sectors in label order, as eager reports them.
    labels: Vec<(SectorId, <R as SectorCodec>::Sector)>,
    hermitian_tol: HermitianTol,
    pub(super) routes: EighRoutePlan<R>,
    _payload: PhantomData<(D, S)>,
}

/// Caller-owned mutable state for repeated execution of an [`EighFullPlan`].
pub struct EighFullWorkspace<R: SectorCodec, D, S = Vec<D>> {
    binding: Arc<()>,
    output: OutputSlot<StackPair<R, D, S>>,
    spectra: Vec<Vec<SectorSpectrum<<R as SectorCodec>::Sector>>>,
    /// Every member's ascending eigenvalues, member-major (`Σ n` each).
    pub(in crate::typed) sorted: Vec<f64>,
    /// The solver-to-ascending column order of `sorted`'s entries.
    #[cfg(feature = "cuda")]
    pub(in crate::typed) orders: Vec<usize>,
}

/// `(d, v)` of a prepared eigendecomposition.
pub(in crate::typed) type StackPair<R, D, S> =
    (StackedTensorMap<R, D, S>, StackedTensorMap<R, D, S>);

/// The data-independent part of the eigendecomposition, compiled once from
/// the structure for both placements (`eigh_full` keeps every eigenpair, so
/// nothing depends on the payload).
pub(super) struct EighRoutePlan<R> {
    /// The source and eigenvector regions of the compact factor plan, all
    /// EIGH reads of it besides the routes.
    source_regions: Arc<[CoupledSectorRegion]>,
    /// The plan's executed (nonzero-rank) routes; every other per-route
    /// table below follows their order.
    routes: Vec<symtn_matrixalgebra::seam::CompactFactorRoute>,
    left_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
    middle_space: symtn_tensors::BoundDynamicFusionMapSpace<R>,
    /// `(offset, n)` of every source coupled sector, for admission.
    #[cfg(feature = "cuda")]
    admission: Vec<(usize, usize)>,
    /// First eigenvalue of each route in a member's concatenated spectra.
    spectrum_offsets: Vec<usize>,
    /// For each of the plan's `labels`, its route; `None` for an empty
    /// sector, whose spectrum is empty as in eager.
    pub(super) label_routes: Vec<Option<usize>>,
    /// Each route's eigenvector region within a member: its range and rows.
    targets: Vec<(std::ops::Range<usize>, usize)>,
    /// The largest coupled-sector order, sizing the Host reorder scratch.
    max_n: usize,
    /// Eigenvalues per member (`Σ n` over the routes).
    spectrum_len: usize,
    /// Each route's diagonal in `d`: `(offset, step)` within a member.
    diagonals: Vec<(usize, usize)>,
    /// Each route's `(source row, target row, rows)` copies within its
    /// target region: one for a layout-aligned route, one per codomain tree
    /// otherwise.
    pub(super) copies: Vec<Vec<(usize, usize, usize)>>,
    d_len: usize,
    v_len: usize,
    d_signature: StructureSignature,
    v_signature: StructureSignature,
}

impl<R, D, S> EighFullPlan<R, D, S>
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    /// Prepares the eigendecomposition of stacks with `source`'s signature.
    /// Its payload is not read, and `B` is not fixed.
    ///
    /// Why a stack and not a bare signature: the plan needs the bound space
    /// and its provider, which a signature does not carry.
    ///
    /// `rows`, `cols` and `hermitian_tol` are the leg roles and the
    /// Hermiticity admission of the eager [`TensorMap::eigh_full`]; the plan
    /// fixes the tolerance for every execution. A stack has no batched
    /// permute, so only the
    /// members' current split (`rows = 0..nout`, `cols = nout..rank`) is
    /// supported here.
    ///
    /// # Errors
    ///
    /// The eager structural errors (not an endomorphism, a non-square
    /// coupled-sector block), `UnsupportedTensorContractScope` for a complex
    /// payload, a layout without packed coupled-sector regions, or leg roles
    /// other than the current split.
    pub fn new(
        engine: &mut dyn Engine,
        source: &StackedTensorMap<R, D, S>,
        rows: &[usize],
        cols: &[usize],
        hermitian_tol: HermitianTol,
    ) -> Result<Self, Error> {
        // Host-local work: the engine is the uniform call shape.
        let _ = engine;
        let nout = source.space.space().nout();
        if !rows.iter().copied().eq(0..nout)
            || !cols.iter().copied().eq(nout..source.space.space().rank())
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh supports only the members' current split; \
                          permute the members before pack",
            }
            .into());
        }
        if !D::CONJUGATION_IS_IDENTITY {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh supports real payloads until tenferro-rs#1923",
            }
            .into());
        }
        let space = source.space.space();
        symtn_matrixalgebra::seam::EndomorphismFamily::EighFull
            .require_endomorphism(space.homspace())?;
        let regions = space
            .structure()
            .coupled_sector_regions(space.nout())?
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "a prepared eigh needs packed coupled-sector regions",
            })?;
        if regions.iter().any(|region| region.rows() != region.cols()) {
            return Err(OperationError::SpaceMismatch {
                message: symtn_matrixalgebra::seam::EIGH_NOT_SQUARE,
            }
            .into());
        }
        symtn_matrixalgebra::seam::validate_endomorphism_region_stacking(
            &regions,
            symtn_matrixalgebra::seam::EndomorphismFamily::EighFull,
        )?;
        let provider = source.space.provider();
        let mut labels = regions
            .iter()
            .map(|region| Ok((region.coupled(), provider.decode_sector(region.coupled())?)))
            .collect::<Result<Vec<_>, Error>>()?;
        labels.sort_by(|left, right| left.1.cmp(&right.1));
        let routes = EighRoutePlan::new(source, &labels)?;
        Ok(Self {
            identity: Arc::new(()),
            runtime: source.runtime.clone(),
            source: source.signature.clone(),
            member_len: source.member_len,
            labels,
            hermitian_tol,
            routes,
            _payload: PhantomData,
        })
    }

    /// Creates independent mutable execution state for this plan.
    pub fn workspace(&self) -> Result<EighFullWorkspace<R, D, S>, Error> {
        Ok(EighFullWorkspace {
            binding: Arc::clone(&self.identity),
            output: OutputSlot::default(),
            spectra: Vec::new(),
            sorted: Vec::new(),
            #[cfg(feature = "cuda")]
            orders: Vec::new(),
        })
    }
}

impl<R: SectorCodec, D, S> EighFullWorkspace<R, D, S> {
    /// Moves the workspace-owned `(d, v)` of the last successful call out; the
    /// next execute allocates new ones. `None` after a failed call: a batch
    /// fails as a whole, so no partially written factor is ever handed out.
    pub fn take_output(&mut self) -> Option<StackPair<R, D, S>> {
        self.output.take_output()
    }

    /// The output buffers for a call over `members`, taken after the source
    /// is admitted so a rejected call keeps buffers of any member count.
    fn take_buffers(&mut self, members: usize) -> Option<StackPair<R, D, S>> {
        self.output.take().filter(|(d, _)| d.members == members)
    }

    /// Bytes retained exclusively by this workspace: output payloads, host
    /// spectra, and Host/CUDA staging buffers. The shared plan and backend
    /// context resources are not included.
    pub fn retained_bytes(&self) -> usize {
        let payloads = self
            .output
            .buffers()
            .map(|(d, v)| {
                (d.members * d.member_len + v.members * v.member_len) * std::mem::size_of::<D>()
            })
            .sum::<usize>();
        let spectra = self.spectra.capacity()
            * std::mem::size_of::<Vec<SectorSpectrum<<R as SectorCodec>::Sector>>>()
            + self
                .spectra
                .iter()
                .map(|member| {
                    member.capacity()
                        * std::mem::size_of::<SectorSpectrum<<R as SectorCodec>::Sector>>()
                        + member
                            .iter()
                            .map(|entry| entry.values.capacity() * std::mem::size_of::<f64>())
                            .sum::<usize>()
                })
                .sum::<usize>();
        let sorted = self.sorted.capacity() * std::mem::size_of::<f64>();
        #[cfg(feature = "cuda")]
        let device_scratch = self.orders.capacity() * std::mem::size_of::<usize>();
        #[cfg(not(feature = "cuda"))]
        let device_scratch = 0;
        payloads + spectra + sorted + device_scratch
    }
}

impl<R: SectorCodec, D, S> EighFullPlan<R, D, S> {
    fn check_workspace(&self, workspace: &EighFullWorkspace<R, D, S>) -> Result<(), Error> {
        if Arc::ptr_eq(&self.identity, &workspace.binding) {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "eigh workspace belongs to another plan".into(),
            ))
        }
    }

    fn check_source(&self, source: &StackedTensorMap<R, D, S>) -> Result<(), Error> {
        match self.source.first_mismatch(&source.signature) {
            Some(field) => Err(Error::BatchSignatureMismatch {
                member: None,
                field,
            }),
            None => Ok(()),
        }
    }

    fn stack(
        &self,
        space: BoundDynamicFusionMapSpace<R>,
        signature: StructureSignature,
        storage: S,
        members: usize,
        member_len: usize,
    ) -> StackedTensorMap<R, D, S> {
        StackedTensorMap {
            runtime: self.runtime.clone(),
            space,
            signature,
            storage,
            members,
            member_len,
            _payload: PhantomData,
        }
    }

    fn output_ref<'a>(
        &self,
        workspace: &'a EighFullWorkspace<R, D, S>,
    ) -> Result<EighStackOutput<'a, R, D, S>, Error> {
        let (d, v) = workspace
            .output
            .output
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument("eigh output is missing".into()))?;
        Ok(EighStackOutput {
            d,
            v,
            spectra: &workspace.spectra,
        })
    }
}

impl<R, D, S> EighFullPlan<R, D, S>
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    D: FactorizationScalar,
    S: FactorExec<R, D>,
{
    /// Decomposes every member into the workspace-owned `(d, v)` and borrows
    /// them with the host spectra.
    ///
    /// # Host
    ///
    /// Every member is admitted (the eager Hermitian rule, per coupled
    /// sector, one read of its `Σ n²` region entries) before any member is
    /// solved; then each member runs eager's finite-input check and, per
    /// coupled sector, eager's solver stage, compiled once by [`Self::new`].
    /// The stage's eigenvectors are written straight into the member's
    /// slice of `v` (one copy of `Σ n²` per member, from the solver's own
    /// output buffer) and its ascending values onto `d`'s diagonal and the
    /// spectra. Per member symtn allocates nothing: the only per-member
    /// allocations are the dense solver's outputs, one set per coupled
    /// sector. The outputs are allocated zeroed when absent or when `B`
    /// changes and reused otherwise: `v` is overwritten, and `d` only on its
    /// diagonal, since the workspace never writes `d` elsewhere.
    ///
    /// # CUDA
    ///
    /// Per call, independent of `B`: at most three admission downloads and
    /// one spectra download; one batched solver call per coupled sector
    /// (each reads its solver status, a host barrier, inside the backend);
    /// one column gather per coupled sector, then one strided copy per
    /// aligned coupled sector (per codomain tree otherwise); one strided copy
    /// per coupled sector onto `d`'s diagonals. Host to device, per call:
    /// the `B · Σ n` sorted eigenvalues (the only payload of `d` that
    /// changes), one `[B · n, 2]` index table per coupled sector and at most
    /// two `[B]` normalizer vectors per coupled sector.
    ///
    /// `d` and `v` are reused across calls at the same `B`. A new `B`, or a
    /// call after [`EighFullWorkspace::take_output`], allocates both with one
    /// zero upload each (`B · Σ n²` and `B · L_v` elements, the only device
    /// allocation path, #740); `d`'s off-diagonal entries are never
    /// written.
    ///
    /// # Errors
    ///
    /// [`BatchError::Operation`] with [`Error::BatchSignatureMismatch`] for a
    /// stack of another signature; [`BatchError::MemberRejected`] listing
    /// every non-Hermitian member (before any solve) or, after the solves,
    /// every member with a non-finite eigenvalue; otherwise the eager
    /// operation's errors as [`BatchError::Operation`]. A workspace from
    /// another plan is rejected before it is touched. Once the workspace is
    /// bound to this plan, any execution error leaves no observable output.
    ///
    /// On CUDA, non-Hermitian members are found by the batched admission
    /// before any solver launch, and [`BatchError::Solver`] is returned when
    /// the solver reports a numerical failure on a block. Argument, bounds
    /// and backend failures are [`BatchError::Operation`].
    pub fn execute<'a>(
        &self,
        engine: &mut dyn Engine,
        source: &StackedTensorMap<R, D, S>,
        workspace: &'a mut EighFullWorkspace<R, D, S>,
    ) -> Result<EighStackOutput<'a, R, D, S>, BatchError> {
        self.check_workspace(workspace)?;
        if let Err(error) = self.check_source(source) {
            return Err(workspace.output.hide(error.into()));
        }
        let mut buffers = workspace.take_buffers(source.members);
        if let Err(error) = S::eigh_full_stack(engine, EXEC, self, source, &mut buffers, workspace)
        {
            return Err(workspace.output.fail(buffers, error));
        }
        if let Err(error) = self.publish(workspace, source.members) {
            return Err(workspace.output.fail(buffers, error.into()));
        }
        if let Some(buffers) = buffers {
            workspace.output.publish(buffers);
        }
        Ok(self.output_ref(workspace)?)
    }
}

impl<R, D> EighFullPlan<R, D>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar,
{
    /// The Host call into `buffers`, allocated zeroed when absent; writes
    /// every member's ascending eigenvalues into `sorted`. `d`'s
    /// off-diagonal entries are never written, so a reused `d` needs only
    /// its diagonals.
    pub(in crate::typed) fn run_host(
        &self,
        engine: &mut dyn Engine,
        source: &StackedTensorMap<R, D>,
        buffers: &mut Option<StackPair<R, D, Vec<D>>>,
        sorted: &mut Vec<f64>,
    ) -> Result<(), BatchError> {
        let members = source.members;
        let len = self.member_len;
        let plan = &self.routes;
        let rejected = source
            .storage
            .chunks_exact(len.max(1))
            .take(members)
            .enumerate()
            .filter_map(|(member, data)| {
                match symtn_matrixalgebra::seam::validate_hermitian_regions(
                    data,
                    &plan.source_regions,
                    self.hermitian_tol,
                ) {
                    Ok(()) => None,
                    // Shapes were admitted by `new`, so this is the content
                    // rule, the only other error the check reports.
                    Err(OperationError::InvalidArgument { .. }) => {
                        Some(Ok((member, MemberFault::NotHermitian)))
                    }
                    Err(other) => Some(Err(Error::from(other))),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !rejected.is_empty() {
            return Err(BatchError::MemberRejected { members: rejected });
        }

        if buffers.is_none() {
            *buffers = Some((
                self.stack(
                    plan.middle_space.clone(),
                    plan.d_signature.clone(),
                    vec![D::zero(); plan.d_len * members],
                    members,
                    plan.d_len,
                ),
                self.stack(
                    plan.left_space.clone(),
                    plan.v_signature.clone(),
                    vec![D::zero(); plan.v_len * members],
                    members,
                    plan.v_len,
                ),
            ));
        }
        let Some((d, v)) = buffers.as_mut() else {
            return Err(internal_layout_error("eigh output stacks are missing").into());
        };
        let total = plan.spectrum_len;
        sorted.clear();
        sorted.resize(total * members, 0.0);
        let mut scratch = symtn_matrixalgebra::seam::EighScratch::<D>::for_order(plan.max_n);
        // The scope body is `Send`; the bound spaces are not, so it borrows
        // only the plan's tables and the payloads.
        let storage = source.storage.as_slice();
        let (d_len, v_len) = (plan.d_len, plan.v_len);
        let (d_storage, v_storage) = (d.storage.as_mut_slice(), v.storage.as_mut_slice());
        let EighRoutePlan {
            source_regions,
            routes,
            spectrum_offsets,
            targets,
            copies,
            diagonals,
            ..
        } = plan;
        let runtime = self.runtime.clone();
        let mut dense = runtime.lease_dense();
        // One linear-algebra scope for the batch: eager opens one per member.
        let faults =
            symtn_matrixalgebra::seam::in_linalg_scope(engine, dense.dense(), |dense, engine| {
                let mut faults = Vec::new();
                for member in 0..members {
                    let data = &storage[member * len..(member + 1) * len];
                    symtn_matrixalgebra::seam::require_finite_eigh_input(data)?;
                    let values = &mut sorted[member * total..(member + 1) * total];
                    let d_member = &mut d_storage[member * d_len..(member + 1) * d_len];
                    let v_member = &mut v_storage[member * v_len..(member + 1) * v_len];
                    for (index, route) in routes.iter().enumerate() {
                        let n = route.rank();
                        let region = &source_regions[route.source_region()];
                        let values = &mut values[spectrum_offsets[index]..][..n];
                        let vectors = match symtn_matrixalgebra::seam::eigh_sector_into(
                            engine,
                            dense,
                            &data[region.range()],
                            n,
                            &mut scratch,
                            values,
                        ) {
                            Ok(vectors) => vectors,
                            // The eager rejection of a non-finite eigenvalue;
                            // every other member is still solved so all of them
                            // are named. An executor's own numerical failure is
                            // not a member fault.
                            Err(OperationError::Dense(
                                symtn_dense::DenseError::NumericalFailure {
                                    op: symtn_matrixalgebra::seam::EIGH_EIGENVALUE_CHECK,
                                    ..
                                },
                            )) => {
                                faults.push((member, MemberFault::NonFiniteEigenvalue));
                                break;
                            }
                            Err(other) => return Err(other),
                        };
                        // One contiguous copy for an aligned route, one row slice
                        // per column and codomain tree otherwise.
                        let (ref range, target_rows) = targets[index];
                        let target = &mut v_member[range.clone()];
                        for &(source_row, target_row, rows) in &copies[index] {
                            if rows == n && target_rows == n {
                                target.copy_from_slice(&vectors);
                                continue;
                            }
                            for column in 0..n {
                                target[column * target_rows + target_row..][..rows]
                                    .copy_from_slice(&vectors[column * n + source_row..][..rows]);
                            }
                        }
                        let (offset, step) = diagonals[index];
                        for (position, &value) in values.iter().enumerate() {
                            d_member[offset + position * step] = D::from_real(value);
                        }
                    }
                }
                Ok(faults)
            })
            .map_err(Error::from)?;
        drop(dense);
        if !faults.is_empty() {
            return Err(BatchError::MemberRejected { members: faults });
        }
        Ok(())
    }
}

impl<R: SectorCodec, D, S> EighFullPlan<R, D, S> {
    /// Rewrites the workspace spectra from its member-major `sorted` values,
    /// in label order, reusing every buffer of the same shape.
    fn publish(
        &self,
        workspace: &mut EighFullWorkspace<R, D, S>,
        members: usize,
    ) -> Result<(), Error> {
        let plan = &self.routes;
        let total = plan.spectrum_len;
        let sorted = &workspace.sorted;
        publish_spectra(
            &mut workspace.spectra,
            &self.labels,
            members,
            |member, position| match plan.label_routes.get(position).copied().flatten() {
                None => Ok(&[][..]),
                Some(route) => plan
                    .spectrum_offsets
                    .get(route)
                    .zip(plan.routes.get(route))
                    .and_then(|(&offset, route)| {
                        sorted.get(member * total + offset..member * total + offset + route.rank())
                    })
                    .ok_or_else(|| internal_layout_error("a coupled sector has no route")),
            },
        )
    }
}

/// Every diagonal fusion-tree block of a `bond <- bond` structure as
/// `(coupled sector, offset, step, count)`, read as eager
/// `diagonal_bond_data` reads it.
fn sector_diagonals(
    structure: &symtn_core::BlockStructure,
) -> Result<Vec<(SectorId, usize, usize, usize)>, Error> {
    let mut diagonals = Vec::with_capacity(structure.block_count());
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let symtn_core::BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let strides = block.strides();
        let shape = block.shape();
        diagonals.push((
            tree.codomain_tree().coupled(),
            block.offset(),
            strides[0] + strides[1],
            shape[0].min(shape[1]),
        ));
    }
    Ok(diagonals)
}

/// Resizes `spectra` to `members` and rewrites each member's coupled
/// sectors, in `labels` order, from `values(member, label position)`,
/// reusing every buffer of the same shape.
fn publish_spectra<'v, S: Clone>(
    spectra: &mut Vec<Vec<SectorSpectrum<S>>>,
    labels: &[(SectorId, S)],
    members: usize,
    mut values: impl FnMut(usize, usize) -> Result<&'v [f64], Error>,
) -> Result<(), Error> {
    spectra.resize_with(members, Vec::new);
    for (member, entries) in spectra.iter_mut().enumerate() {
        entries.truncate(labels.len());
        for (position, (_, label)) in labels.iter().enumerate() {
            let source = values(member, position)?;
            match entries.get_mut(position) {
                Some(entry) => {
                    entry.sector.clone_from(label);
                    entry.values.clear();
                    entry.values.extend_from_slice(source);
                }
                None => entries.push(SectorSpectrum {
                    sector: label.clone(),
                    values: source.to_vec(),
                }),
            }
        }
    }
    Ok(())
}

impl<R> EighRoutePlan<R>
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
{
    /// Compiles the eager plan from the structure: `eigh_full` keeps every
    /// eigenpair, so each coupled sector's rank is its full `n`.
    fn new<D, S>(
        source: &StackedTensorMap<R, D, S>,
        labels: &[(SectorId, <R as SectorCodec>::Sector)],
    ) -> Result<Self, Error> {
        let plan = symtn_matrixalgebra::seam::compact_factor_routes(&source.space)?;
        let routes: Vec<_> = plan.executed_routes().map(|(route, _, _)| *route).collect();
        let left_space = plan.left_space(&source.space)?;
        let middle_space = plan.bond_space(&source.space)?;
        let label_routes = labels
            .iter()
            .map(|(sector, _)| routes.iter().position(|route| route.sector() == *sector))
            .collect();
        let max_n = routes.iter().map(|route| route.rank()).max().unwrap_or(0);
        let targets = routes
            .iter()
            .map(|route| {
                let region = &plan.left_regions()[left_region(route)?];
                Ok((region.range(), region.rows()))
            })
            .collect::<Result<_, Error>>()?;
        #[cfg(feature = "cuda")]
        let admission = plan
            .source_regions()
            .iter()
            .map(|region| (region.range().start, region.rows()))
            .collect();
        let mut spectrum_offsets = Vec::with_capacity(routes.len());
        let mut copies = Vec::with_capacity(routes.len());
        let mut spectrum_len = 0usize;
        let policy = crate::typed::probe::RoutePolicy::current();
        for route in &routes {
            spectrum_offsets.push(spectrum_len);
            spectrum_len += route.rank();
            copies.push(route_copies(&plan, route, policy)?);
        }
        let diagonals = route_diagonals(&routes, &middle_space)?;
        let placement = source.signature.placement;
        let d_signature = StructureSignature::of_space(&middle_space, placement, &source.runtime);
        let v_signature = StructureSignature::of_space(&left_space, placement, &source.runtime);
        let d_len = middle_space.space().required_len()?;
        let v_len = left_space.space().required_len()?;
        Ok(Self {
            source_regions: Arc::clone(plan.source_regions()),
            routes,
            left_space,
            middle_space,
            #[cfg(feature = "cuda")]
            admission,
            spectrum_offsets,
            label_routes,
            targets,
            max_n,
            spectrum_len,
            diagonals,
            copies,
            d_len,
            v_len,
            d_signature,
            v_signature,
        })
    }
}

/// A route's `(source row, target row, rows)` copies within its target region,
/// exactly the placements of eager's assembly: the whole region when the
/// plan proved it layout-aligned, else one row slice per codomain tree,
/// matched by tree identity.
fn route_copies(
    plan: &symtn_matrixalgebra::seam::CompactFactorPlan,
    route: &symtn_matrixalgebra::seam::CompactFactorRoute,
    policy: crate::typed::probe::RoutePolicy,
) -> Result<Vec<(usize, usize, usize)>, Error> {
    let target = &plan.left_regions()[left_region(route)?];
    if plan.left_preserves_trees(route) && !policy.eigh_treewise {
        return Ok(vec![(0, 0, target.rows())]);
    }
    let source = &plan.source_regions()[route.source_region()];
    let mut copies = Vec::with_capacity(target.row_trees().len());
    for target_tree in target.row_trees() {
        let rows = target_tree.extent()?;
        if rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|tree| tree.tree() == target_tree.tree())
            .map(|tree| tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        copies.push((src_row, target_tree.offset(), rows));
    }
    Ok(copies)
}

/// An executed route's eigenvector region.
fn left_region(route: &symtn_matrixalgebra::seam::CompactFactorRoute) -> Result<usize, Error> {
    route
        .left_region()
        .ok_or_else(|| internal_layout_error("an executed EIGH route has no eigenvector region"))
}

/// Each route's diagonal in the dense `d` of one member, `(offset, step)`.
fn route_diagonals<R>(
    routes: &[symtn_matrixalgebra::seam::CompactFactorRoute],
    middle_space: &symtn_tensors::BoundDynamicFusionMapSpace<R>,
) -> Result<Vec<(usize, usize)>, Error> {
    let diagonals = sector_diagonals(middle_space.space().structure())?;
    routes
        .iter()
        .map(|route| {
            let sector = route.sector();
            match diagonals.iter().find(|entry| entry.0 == sector) {
                Some(&(_, offset, step, count)) if count == route.rank() => Ok((offset, step)),
                _ => Err(internal_layout_error(
                    "an EIGH route has no matching diagonal block",
                )),
            }
        })
        .collect()
}

#[cfg(feature = "cuda")]
impl<R, D> EighFullPlan<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// The device call into `buffers`, allocated zeroed when absent; returns
    /// every member's sorted eigenvalues, member-major (`Σ n` per member).
    pub(in crate::typed) fn run_cuda(
        &self,
        engine: &mut dyn Engine,
        source: &StackedTensorMap<R, D, CudaStorage<D>>,
        buffers: &mut Option<StackPair<R, D, CudaStorage<D>>>,
        sorted: &mut Vec<f64>,
        orders: &mut Vec<usize>,
    ) -> Result<(), BatchError> {
        let members = source.members;
        let device = &self.routes;
        let mut lease = crate::runtime::cuda_call(engine)?;
        let cuda = &mut *lease;
        let routes = &device.routes;

        let accepted = symtn_dense::cuda_hermitian_regions_batched::<D>(
            cuda,
            &source.storage.0,
            &device.admission,
            members,
            self.member_len,
            self.hermitian_tol
                .resolve(<D as symtn_matrixalgebra::FactorScalar>::epsilon()),
        )
        .map_err(dense_err)?;
        let rejected: Vec<_> = accepted
            .chunks(device.admission.len().max(1))
            .enumerate()
            .filter(|(_, member)| member.contains(&false))
            .map(|(member, _)| (member, MemberFault::NotHermitian))
            .collect();
        if !rejected.is_empty() {
            return Err(BatchError::MemberRejected { members: rejected });
        }

        let mut device_spectra = Vec::with_capacity(routes.len());
        let mut vectors = Vec::with_capacity(routes.len());
        for route in routes {
            let region = &device.source_regions[route.source_region()];
            let (values, vector) = symtn_dense::cuda_eigh_region_batched::<D>(
                cuda,
                &source.storage.0,
                region.range().start,
                route.rank(),
                members,
                self.member_len,
            )
            .map_err(|error| match error {
                // Only the solver's own status names a block; argument,
                // bounds and backend failures are the operation's.
                error @ symtn_dense::DenseError::NumericalFailure { .. } => BatchError::Solver {
                    block: route.source_region(),
                    source: dense_err(error),
                },
                other => BatchError::Operation(dense_err(other)),
            })?;
            device_spectra.push(values);
            vectors.push(vector);
        }
        let raw = symtn_dense::cuda_download_batched_spectra::<D>(cuda, &device_spectra, members)
            .map_err(dense_err)?;
        drop(device_spectra);

        // Per member and route, eager's order (one authority).
        let total = device.spectrum_len;
        sorted.clear();
        sorted.resize(total * members, 0.0);
        orders.clear();
        orders.resize(total * members, 0);
        let mut faults = Vec::new();
        for member in 0..members {
            let mut finite = true;
            for (route, &offset) in routes.iter().zip(&device.spectrum_offsets) {
                let n = route.rank();
                let base = member * total + offset;
                let values = &raw[base..base + n];
                finite &= values.iter().all(|value| value.is_finite());
                let order = &mut orders[base..base + n];
                symtn_matrixalgebra::seam::ascending_eigh_order(values, order);
                for (slot, &index) in sorted[base..base + n].iter_mut().zip(order.iter()) {
                    *slot = values[index];
                }
            }
            if !finite {
                faults.push((member, MemberFault::NonFiniteEigenvalue));
            }
        }
        if !faults.is_empty() {
            return Err(BatchError::MemberRejected { members: faults });
        }

        // `d` and `v` are allocated zeroed once per `B` (the only device
        // allocation path, #740). The workspace never writes `d` off its
        // diagonal, so each call moves only the `B · Σ n` sorted values.
        if buffers.is_none() {
            let zeros = |len: usize| {
                CudaStorage::<D>::upload_members(cuda, vec![D::zero(); len * members], len, members)
                    .map_err(Error::from)
            };
            *buffers = Some((
                self.stack(
                    device.middle_space.clone(),
                    device.d_signature.clone(),
                    zeros(device.d_len)?,
                    members,
                    device.d_len,
                ),
                self.stack(
                    device.left_space.clone(),
                    device.v_signature.clone(),
                    zeros(device.v_len)?,
                    members,
                    device.v_len,
                ),
            ));
        }
        let Some((d, v)) = buffers.as_mut() else {
            return Err(internal_layout_error("eigh output stacks are missing").into());
        };
        let values = CudaStorage::<D>::upload_owned(
            cuda,
            sorted.iter().map(|&value| D::from_real(value)).collect(),
        )
        .map_err(Error::from)?;
        for ((&(offset, step), &spectrum), route) in device
            .diagonals
            .iter()
            .zip(&device.spectrum_offsets)
            .zip(routes)
        {
            let region = |dims: Vec<usize>, strides: Vec<usize>, offset: usize| {
                symtn_dense::CudaRegion::new(dims, strides, offset).map_err(dense_err)
            };
            symtn_dense::cuda_copy_strided_into::<D>(
                cuda,
                &values.0,
                &region(vec![route.rank(), members], vec![1, total], spectrum)?,
                &mut d.storage.0,
                &region(
                    vec![route.rank(), members],
                    vec![step, device.d_len],
                    offset,
                )?,
            )
            .map_err(dense_err)?;
        }
        for ((((route, raw_vectors), &offset), copies), (target, target_rows)) in routes
            .iter()
            .zip(&vectors)
            .zip(&device.spectrum_offsets)
            .zip(&device.copies)
            .zip(&device.targets)
        {
            let columns: Vec<usize> = (0..members)
                .flat_map(|member| {
                    orders[member * total + offset..][..route.rank()]
                        .iter()
                        .copied()
                })
                .collect();
            symtn_dense::cuda_gather_columns_batched_into::<D>(
                cuda,
                &mut v.storage.0,
                target.start,
                *target_rows,
                device.v_len,
                raw_vectors,
                route.rank(),
                members,
                &columns,
                copies,
            )
            .map_err(dense_err)?;
        }
        Ok(())
    }
}
