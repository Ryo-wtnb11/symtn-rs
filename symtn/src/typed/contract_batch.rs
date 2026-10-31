//! Restricted binding for one ordinary contraction over owned-dense stacks.

use super::*;
use crate::typed::PreparedExec;
use symtn_dense::Engine;
use symtn_tensors::{HostContractMembersWorkspace, OutputAxisOrder, StorageContractResolution};

#[cfg(test)]
#[path = "compose_plan_tests.rs"]
mod compose_plan_tests;
#[cfg(test)]
#[path = "contract_batch_copy_c.rs"]
mod copy_c;
#[cfg(test)]
#[path = "contract_batch_members.rs"]
mod members;

/// Immutable contraction structure for owned-dense Host or CUDA stacks.
///
/// The route is the contraction planner's (`plan_contract`), the one the
/// device [`TensorMap::contract`] replays. This binding admits Core, CopyC
/// and owned-source transformed-tree routes whose core is fully direct with
/// exact +1/-1 coefficients, checked once at construction so the admitted
/// set does not depend on the member count. CUDA admits CopyC when every
/// output move is an unconjugated nonzero Single task, and transformed-tree
/// routes whose source and output moves are all such tasks over a unit-alpha
/// direct core. Direct
/// composition is served by [`ComposePlan`]. The plan fixes structure and
/// axes, but not member count. On CUDA the member axis is a dense stride of
/// every submission, so the GEMMs, region moves and zero fills of a call do
/// not depend on the member count.
pub struct ContractPlan<R, D, S = Vec<D>> {
    pub(in crate::typed) runtime: Runtime,
    lhs: StructureSignature,
    rhs: StructureSignature,
    output_signature: StructureSignature,
    pub(in crate::typed) space: BoundDynamicFusionMapSpace<R>,
    pub(in crate::typed) resolution: Arc<StorageContractResolution<f64>>,
    member_len: usize,
    _payload: PhantomData<(D, S)>,
}

/// Caller-owned output and execution resources for a [`ContractPlan`].
///
/// The workspace-owned output's inactive blocks (those no core GEMM writes)
/// are zero for the buffer's whole life, so `execute` never refills them
/// (#2123). This is a Rust-specific deviation: TensorKit `mul!` rescales
/// every block with no product by `β` on each call, because its destination
/// is the caller's; here the workspace owns the buffer, and its inactive
/// blocks are provably zero: the buffer is born zero, and a changed member
/// count replaces it with a fresh zero buffer; the inactive set is fixed,
/// because the workspace is bound to one resolution (`Arc::ptr_eq`); a route
/// whose core writes the output directly (Core and DynamicTree identity
/// output) has GEMMs that write only the active blocks, a disjoint tiling, so a failure
/// part-way leaves only partial active writes, which the next call
/// overwrites; a CopyC or DynamicTree output transform overwrites the whole
/// destination; and the caller never reaches the buffer mutably until
/// `take_output` moves it out for good. `execute_into` cannot rely on any of
/// this for a caller's destination and rewrites every inactive block.
///
/// On Host with the faer provider, the dense GEMM plans of the grouped core
/// submissions are not workspace state: the block GEMM kernel takes them
/// from the process-global geometry-plan cache
/// ([`StructureCacheKind::GemmPlans`](crate::cache::StructureCacheKind::GemmPlans)),
/// shared by every workspace, member count and eager call, so a warm call
/// prepares no grouped core plan while the cache keeps it. The workspace's
/// dense executor keeps the kernel's per-call staging, so a warm call on a
/// serial engine allocates nothing; [`Self::retained_bytes`] includes that
/// staging, which keeps its high-water capacity. One-job and affine
/// core batches keep Tenferro's binary-plan cache, whose misses are outside
/// this guarantee.
pub struct ContractWorkspace<R, D, S = Vec<D>> {
    binding: Arc<StorageContractResolution<f64>>,
    pub(super) output: OutputSlot<StackedTensorMap<R, D, S>>,
    pub(in crate::typed) members: HostContractMembersWorkspace<D>,
    /// The Host replay's execution scratch (tree and contract lanes), kept
    /// for this workspace's replays instead of rebuilt per call.
    pub(in crate::typed) host_context: crate::runtime::ContextLease,
    #[cfg(feature = "cuda")]
    pub(in crate::typed) members_cuda: symtn_tensors::CudaContractMembersWorkspace<S>,
    /// The engine ([`symtn_dense::CudaDenseContext::identity`]) the device
    /// state above belongs to; `None` while it holds none.
    #[cfg(feature = "cuda")]
    pub(in crate::typed) cuda_engine: Option<u64>,
}

#[cfg(feature = "cuda")]
impl<R, D, S> ContractWorkspace<R, D, S> {
    /// Drops the outputs, the spare and the member scratch and templates, and
    /// binds the emptied workspace to `engine`.
    pub(in crate::typed) fn drop_device_state(&mut self, engine: Option<u64>) {
        self.output = OutputSlot::default();
        self.members_cuda = Default::default();
        self.cuda_engine = engine;
    }
}

impl<R, D, S> ContractPlan<R, D, S>
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: PreparedExec<R, D>,
{
    /// Fixes an ordinary contraction without reading operand payloads.
    /// CUDA admits exact-sign direct Core/SwappedCore, nonzero Single CopyC,
    /// and transformed-tree routes whose transforms are nonzero Single moves.
    pub fn new(
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        spec: &super::super::ContractSpec<'_>,
    ) -> Result<Self, Error> {
        Self::prepare(engine, lhs, rhs, Some(spec))
    }

    /// The one constructor of both prepared operations: `Some(spec)` is the
    /// contraction ([`Self::new`]), `None` the composition
    /// ([`ComposePlan::new`]). The kind is fixed here, before planning,
    /// because a fermionic composition and the contraction over the same
    /// legs differ by the supertrace twist.
    fn prepare(
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        spec: Option<&super::super::ContractSpec<'_>>,
    ) -> Result<Self, Error> {
        if !lhs.runtime.same_runtime(&rhs.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if lhs.members != rhs.members {
            return Err(Error::InvalidArgument(
                "operand stacks have different member counts".into(),
            ));
        }
        let placement = lhs.signature.placement;
        if rhs.signature.placement != placement {
            return Err(Error::PlacementMismatch);
        }
        let operands = (
            FusionOperand::direct(lhs.space.space()),
            FusionOperand::direct(rhs.space.space()),
        );
        let (space, resolution) = match spec {
            Some(spec) => {
                symtn_tensors::reject_non_symmetric_contraction(
                    lhs.space.provider().braiding_style(),
                )?;
                let output_axes = spec.output_axes();
                let order = OutputAxisOrder::from_axes(&output_axes);
                let space = BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
                    &lhs.space,
                    &rhs.space,
                    spec.lhs,
                    spec.rhs,
                    order,
                    spec.codomain.len(),
                )?;
                let mut lease = crate::runtime::ContextLease::new();
                let lane = lease.context().multiplicity_free_lane::<D>()?;
                let resolution = lane.plan_contract::<symtn_tensors::DirectCoreExecutor, _>(
                    engine,
                    &space,
                    operands.0,
                    operands.1,
                    spec.lhs,
                    spec.rhs,
                    &output_axes,
                )?;
                (space, resolution)
            }
            None => {
                // The composition's core needs no context state, so planning
                // it builds no execution context (#1531).
                let (left, right) = (lhs.space.space(), rhs.space.space());
                let lhs_axes: Vec<usize> = (left.nout()..left.nout() + left.nin()).collect();
                let rhs_axes: Vec<usize> = (0..right.nout()).collect();
                let space = BoundDynamicFusionMapSpace::contracted_multiplicity_free(
                    &lhs.space, &rhs.space, &lhs_axes, &rhs_axes,
                )?;
                let resolution = symtn_tensors::plan_compose::<symtn_tensors::DirectCoreExecutor, _>(
                    &space, operands.0, operands.1,
                )?;
                (space, resolution)
            }
        };
        // One admission for every route and member count: a plan accepted at
        // B = 1 replays at any B.
        resolution
            .core_plan()
            .require_identity_signed_direct_replay()?;
        #[cfg(feature = "cuda")]
        if matches!(placement, Placement::Cuda(_)) {
            resolution.admit_cuda_members()?;
        }
        let member_len = space.space().required_len()?;
        Ok(Self {
            runtime: lhs.runtime.clone(),
            lhs: lhs.signature.clone(),
            rhs: rhs.signature.clone(),
            output_signature: StructureSignature::of_space(&space, placement, &lhs.runtime),
            space,
            resolution: Arc::new(resolution),
            member_len,
            _payload: PhantomData,
        })
    }

    #[cfg(test)]
    fn copy_c(&self) -> Option<&symtn_tensors::CopyCRoute<f64>> {
        self.resolution.copy_c()
    }

    /// Signature required of every output member.
    pub fn output_signature(&self) -> &StructureSignature {
        &self.output_signature
    }

    fn check_workspace(&self, workspace: &ContractWorkspace<R, D, S>) -> Result<(), Error> {
        if Arc::ptr_eq(&self.resolution, &workspace.binding) {
            Ok(())
        } else {
            Err(Error::InvalidArgument(
                "workspace belongs to another plan".into(),
            ))
        }
    }

    /// Checks both operand stacks and returns their common member count.
    fn check(
        &self,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
    ) -> Result<usize, Error> {
        for (actual, expected) in [(&lhs.signature, &self.lhs), (&rhs.signature, &self.rhs)] {
            if let Some(field) = expected.first_mismatch(actual) {
                return Err(Error::BatchSignatureMismatch {
                    member: None,
                    field,
                });
            }
        }
        if lhs.members != rhs.members {
            return Err(Error::InvalidArgument(
                "operand stacks have different member counts".into(),
            ));
        }
        for (len, actual) in [
            (lhs.member_len, lhs.storage.len()),
            (rhs.member_len, rhs.storage.len()),
        ] {
            if actual != self.total_len(len, lhs.members)? {
                return Err(Error::InvalidArgument(
                    "stacked payload length differs from its structure".into(),
                ));
            }
        }
        Ok(lhs.members)
    }

    fn check_destination(
        &self,
        dst: &StackedTensorMap<R, D, S>,
        members: usize,
    ) -> Result<(), Error> {
        if let Some(field) = self.output_signature.first_mismatch(&dst.signature) {
            return Err(Error::BatchSignatureMismatch {
                member: None,
                field,
            });
        }
        if dst.members != members {
            return Err(Error::InvalidArgument(
                "destination member count differs from operands".into(),
            ));
        }
        if dst.storage.len() != self.total_len(self.member_len, members)? {
            return Err(Error::InvalidArgument(
                "destination payload length differs from its structure".into(),
            ));
        }
        Ok(())
    }

    /// Refuses a call whose operand or destination belongs to another engine
    /// than the lease's, before any output, template or scratch upload.
    fn admit_members(lease: &S::MembersLease<'_>, buffers: &[&S]) -> Result<(), Error> {
        if buffers
            .iter()
            .all(|buffer| S::owns_output(EXEC, lease, buffer))
        {
            Ok(())
        } else {
            Err(Error::PlacementMismatch)
        }
    }

    fn total_len(&self, len: usize, members: usize) -> Result<usize, Error> {
        let total = len.checked_mul(members).ok_or_else(|| {
            Error::InvalidArgument("stacked payload length overflows usize".into())
        })?;
        std::alloc::Layout::array::<D>(total).map_err(|_| {
            Error::InvalidArgument("stacked payload allocation overflows usize".into())
        })?;
        Ok(total)
    }

    /// Creates independent mutable state; member count can change later. It
    /// allocates nothing: device buffers are created by the first execution
    /// that needs them.
    pub fn workspace(&self) -> Result<ContractWorkspace<R, D, S>, Error> {
        Ok(ContractWorkspace {
            binding: Arc::clone(&self.resolution),
            output: OutputSlot::default(),
            members: HostContractMembersWorkspace::default(),
            host_context: crate::runtime::ContextLease::new(),
            #[cfg(feature = "cuda")]
            members_cuda: Default::default(),
            #[cfg(feature = "cuda")]
            cuda_engine: None,
        })
    }

    /// Overwrites the workspace-owned output and borrows it until the next
    /// call. Any error follows the batched module's failure rule. The
    /// output's inactive blocks are not rewritten: they are already zero
    /// (see [`ContractWorkspace`]).
    ///
    /// On CUDA a new member count, or an engine other than the one the
    /// retained output belongs to, uploads one zeroed output buffer (#740);
    /// an output of another engine is dropped, never written or kept. A
    /// reused buffer, including one kept after a failed call, transfers no
    /// payload. Operands of another engine are refused
    /// ([`Error::PlacementMismatch`]) before any upload.
    pub fn execute<'a>(
        &self,
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        workspace: &'a mut ContractWorkspace<R, D, S>,
    ) -> Result<&'a StackedTensorMap<R, D, S>, Error> {
        self.check_workspace(workspace)?;
        // First, so no later error of this call keeps another engine's state.
        let mut lease = match S::lease_members(EXEC, engine, workspace) {
            Ok(lease) => lease,
            Err(error) => return Err(workspace.output.hide(error)),
        };
        let buffers = workspace.output.take();
        let prepared = self.check(lhs, rhs).and_then(|members| {
            let total = self.total_len(self.member_len, members)?;
            Self::admit_members(&lease, &[&lhs.storage, &rhs.storage])?;
            Ok((members, total))
        });
        let (members, total) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return Err(workspace.output.fail(buffers, error)),
        };
        let mut output = match buffers.filter(|output| output.members == members) {
            Some(output) => output,
            None => StackedTensorMap {
                runtime: self.runtime.clone(),
                space: self.space.clone(),
                signature: self.output_signature.clone(),
                storage: S::zeroed_members(EXEC, &mut lease, total, self.member_len, members)?,
                members,
                member_len: self.member_len,
                _payload: PhantomData,
            },
        };
        let result = S::contract_members(
            EXEC,
            self,
            workspace,
            &mut lease,
            &mut output.storage,
            (lhs.space.space().structure(), &lhs.storage),
            (rhs.space.space().structure(), &rhs.storage),
            members,
            symtn_tensors::ContractDestinationInit::Zeroed,
        );
        drop(lease);
        workspace.output.settle(output, result)
    }

    /// Overwrites a checked caller destination. Validation and capability
    /// errors leave it unchanged; after a backend error its payload may be
    /// partially overwritten while its metadata stays valid. Any error
    /// follows the batched module's failure rule.
    ///
    /// On CUDA each inactive region is zeroed across all members; the first
    /// such call at a new high-water member count grows the workspace's zero
    /// template, and later calls transfer nothing.
    pub fn execute_into(
        &self,
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        dst: &mut StackedTensorMap<R, D, S>,
        workspace: &mut ContractWorkspace<R, D, S>,
    ) -> Result<(), Error> {
        self.check_workspace(workspace)?;
        // First, so no later error of this call keeps another engine's state.
        let mut lease = match S::lease_members(EXEC, engine, workspace) {
            Ok(lease) => lease,
            Err(error) => return Err(workspace.output.hide(error)),
        };
        let result = self.check(lhs, rhs).and_then(|members| {
            self.check_destination(dst, members)?;
            Self::admit_members(&lease, &[&lhs.storage, &rhs.storage, &dst.storage])?;
            S::contract_members(
                EXEC,
                self,
                workspace,
                &mut lease,
                &mut dst.storage,
                (lhs.space.space().structure(), &lhs.storage),
                (rhs.space.space().structure(), &rhs.storage),
                members,
                symtn_tensors::ContractDestinationInit::Axpby(D::from_real(0.0)),
            )
        });
        result.map_err(|error| workspace.output.hide(error))
    }
}

/// Immutable structural plan for composing every member pair of two stacks.
///
/// Per member it is the eager [`TensorMap::compose`]: TensorKit `mul!`
/// (`linalg.jl:330-370` @cfaa073), `A`'s domain against `B`'s codomain in
/// order, with no twist, for every braiding style. It is a [`ContractPlan`]
/// whose kind is fixed to composition before planning, so it shares that
/// plan's planner, executors, workspace and failure rule; the composition
/// always resolves to the canonical fully-direct core with unit
/// coefficients, compiled once here for every member count. The member
/// axis exists only as a dense stride:
///
/// - **CUDA**: one batched GEMM per coupled sector per call, independent of
///   the member count `B`.
/// - **Host**: one grouped GEMM submission per call over all `B × blocks`
///   matrices, through the Runtime's dense backend.
///
/// The plan is independent of `B` and can be shared across threads. Each
/// caller owns a separate [`ContractWorkspace`] for the mutable output,
/// replay layout and CUDA templates. Inputs and an optional
/// destination are bound and validated on every call.
///
/// # Supported scope
///
/// Identity orientation only: an adjoint operand must be materialized before
/// it is packed. A composition whose plan is not the canonical fully-direct
/// one (for example an expert-layer tiling whose tree stackings differ) is
/// rejected by [`Self::new`] with the same `UnsupportedTensorContractScope`
/// the eager device composition reports.
///
/// [`Self::workspace`] creates the B-dependent state. On CUDA each workspace
/// also keeps the zero and `1` templates of its fills, bound to the engine
/// that created them: used with another engine, it drops its device state,
/// retained outputs included, and rebuilds it there.
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{ComposePlan, Error, GradedSpace, Runtime, StackedTensorMap, TensorMap};
///
/// # fn main() -> Result<(), Error> {
/// let engine = &mut symtn::typed::CpuEngine::serial();
/// let runtime = Runtime::builder().build()?;
/// let rule = Arc::new(U1FusionRule);
/// let v = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)])?;
/// let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 3)])?;
/// let lhs = StackedTensorMap::pack(&[TensorMap::<_, f64>::zeros(engine,
///     &runtime, [&v, &v], [&w],
/// )?])?;
/// let rhs = StackedTensorMap::pack(&[TensorMap::<_, f64>::zeros(engine,
///     &runtime, [&w], [&v],
/// )?])?;
/// let mut destination = StackedTensorMap::pack(&[TensorMap::<_, f64>::zeros(engine,
///     &runtime, [&v, &v], [&v],
/// )?])?;
///
/// let plan = ComposePlan::new(engine, &lhs, &rhs)?;
/// let mut workspace = plan.workspace()?;
/// let _output = plan.execute(engine, &lhs, &rhs, &mut workspace)?;
/// plan.execute_into(engine, &lhs, &rhs, &mut destination, &mut workspace)?;
/// # Ok(())
/// # }
/// ```
pub struct ComposePlan<R, D, S = Vec<D>>(ContractPlan<R, D, S>);

impl<R, D, S> ComposePlan<R, D, S>
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: PreparedExec<R, D>,
{
    /// Prepares the composition of stacks with `lhs`'s and `rhs`'s
    /// signatures. Their payloads are not read, and `B` is not fixed.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`] and [`Error::PlacementMismatch`] for
    /// operands of different Runtimes or placements;
    /// [`Error::InvalidArgument`] for stacks of different member counts; the
    /// eager composition's structural errors for spaces that do not compose;
    /// and `UnsupportedTensorContractScope` for a plan that is not fully
    /// direct.
    pub fn new(
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
    ) -> Result<Self, Error> {
        ContractPlan::prepare(engine, lhs, rhs, None).map(Self)
    }

    /// The signature of every output member.
    pub fn output_signature(&self) -> &StructureSignature {
        self.0.output_signature()
    }

    /// Creates independent mutable execution state for this plan; on CUDA,
    /// the B-dependent output and zero-fill buffers, created by the first
    /// execution that needs them.
    pub fn workspace(&self) -> Result<ContractWorkspace<R, D, S>, Error> {
        self.0.workspace()
    }

    /// Composes every member pair into the workspace-owned output and borrows
    /// it for the workspace borrow's lifetime; [`ContractPlan::execute`]. On
    /// CUDA: one batched GEMM per coupled sector, and no transfer on a warm
    /// call.
    ///
    /// # Errors
    ///
    /// [`Error::BatchSignatureMismatch`] with `member: None` for a stack of
    /// another signature, and [`Error::InvalidArgument`] for operand stacks
    /// of different member counts, all before any work. Any error follows
    /// the batched module's failure rule.
    pub fn execute<'a>(
        &self,
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        workspace: &'a mut ContractWorkspace<R, D, S>,
    ) -> Result<&'a StackedTensorMap<R, D, S>, Error> {
        self.0.execute(engine, lhs, rhs, workspace)
    }

    /// Composes every member pair into the caller's `dst`, overwriting it;
    /// [`ContractPlan::execute_into`]. `dst` must carry
    /// [`Self::output_signature`] and the operands' member count. Its prior
    /// contents never reach the result: the blocks no GEMM writes are
    /// zero-filled on every call (on CUDA, one zero fill per inactive block
    /// per call over all members).
    ///
    /// # Errors
    ///
    /// As [`Self::execute`], plus the same typed errors for `dst`, before
    /// any write; any error follows the batched module's failure rule.
    pub fn execute_into(
        &self,
        engine: &mut dyn Engine,
        lhs: &StackedTensorMap<R, D, S>,
        rhs: &StackedTensorMap<R, D, S>,
        dst: &mut StackedTensorMap<R, D, S>,
        workspace: &mut ContractWorkspace<R, D, S>,
    ) -> Result<(), Error> {
        self.0.execute_into(engine, lhs, rhs, dst, workspace)
    }
}

impl<R, D, S> ContractWorkspace<R, D, S> {
    /// Moves the output of the last successful call to the caller; `None`
    /// after a failed call (the batched module's failure rule).
    pub fn take_output(&mut self) -> Option<StackedTensorMap<R, D, S>> {
        self.output.take_output()
    }
}

impl<R, D, S: PreparedExec<R, D>> ContractWorkspace<R, D, S> {
    /// Retained output payload and replay scratch (on CUDA, device payload
    /// and Host layout scratch), plus on Host the dense executors' kept GEMM
    /// staging (job records, geometry index, plan-handle slots, costs and
    /// lane tables, charged by capacity). GEMM plans are not counted: they
    /// live in the process-global geometry-plan cache. On CUDA the zero and
    /// `1` templates of the fills are not counted: they are bounded by the
    /// largest fill of one call, not by what the member count sizes.
    pub fn retained_bytes(&self) -> usize {
        self.members.retained_bytes()
            + self.host_context.dense_scratch_bytes()
            + S::retained_scratch_bytes(EXEC, self)
            + self
                .output
                .buffers()
                .map(|output| {
                    S::retained_payload_bytes(
                        EXEC,
                        &output.storage,
                        output.members.saturating_mul(output.member_len),
                    )
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod failure_tests {
    use super::*;
    use crate::sector::{U1FusionRule, U1Irrep};
    use crate::typed::GradedSpace;

    #[test]
    fn a_failure_after_the_output_is_taken_keeps_it_unobservable() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: on the direct Core and the CopyC route, an error inside the
        // replay, after the output buffer is taken (forced by an operand
        // whose layout and payload are consistently one entry short, which
        // only a corrupted stack carries), leaves no observable output; the
        // buffer is kept and the next call recovers.
        let runtime = Runtime::builder().build().unwrap();
        let rule = Arc::new(U1FusionRule);
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 3),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v, &v], [&w], 5).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&w], [&v], 6).unwrap();
        let lhs = StackedTensorMap::pack(&[&a, &a]).unwrap();
        let rhs = StackedTensorMap::pack(&[&b, &b]).unwrap();
        let mut corrupt = StackedTensorMap::pack(&[&a, &a]).unwrap();
        corrupt.member_len -= 1;
        corrupt
            .storage
            .truncate(corrupt.member_len * corrupt.members);
        for (codomain, copy_c) in [([0, 1], false), ([1, 0], true)] {
            let spec = super::super::super::ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &codomain,
                domain: &[2],
            };
            let plan = ContractPlan::new(engine, &lhs, &rhs, &spec).unwrap();
            assert_eq!(plan.copy_c().is_some(), copy_c);
            let mut workspace = plan.workspace().unwrap();
            let expected = plan
                .execute(engine, &lhs, &rhs, &mut workspace)
                .unwrap()
                .storage
                .clone();
            let bytes = workspace.retained_bytes();

            // Only the route executor compares the payload with the plan's
            // operand structure.
            let Err(crate::error::Error::Operation(error)) = plan
                .execute(engine, &corrupt, &rhs, &mut workspace)
                .map(|_| ())
            else {
                panic!("the corrupted stack must fail as an operation error");
            };
            assert!(
                matches!(
                    *error,
                    symtn_tensors::OperationError::ElementCountMismatch { .. }
                ),
                "{error:?}"
            );
            assert!(
                workspace.output.spare.is_some(),
                "the error is after the take"
            );
            assert!(workspace.take_output().is_none());
            assert_eq!(workspace.retained_bytes(), bytes);
            assert_eq!(
                plan.execute(engine, &lhs, &rhs, &mut workspace)
                    .unwrap()
                    .storage,
                expected
            );
            assert!(workspace.take_output().is_some());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_copy_c_failure_keeps_the_temporary_and_the_output() {
        let cpu = &mut crate::typed::CpuEngine::serial();
        let cuda = &mut crate::typed::CudaEngine::open(crate::typed::CudaDeviceId::from_ordinal(0))
            .unwrap();
        // What: on the CUDA CopyC route, an error after the core GEMMs have
        // written the temporary and before the output transform (injected:
        // the member replay checks every operand length before its first
        // submission, so a corrupted stack never reaches the device)
        // keeps the temporary and the output in the workspace, so the next
        // call uploads neither and recovers.
        let runtime = Runtime::builder().build().unwrap();
        let rule = Arc::new(U1FusionRule);
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 1),
                (U1Irrep::new(1), 3),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
        let a = TensorMap::<_, f64>::rand_with_seed(cpu, &runtime, [&v, &v], [&w], 5).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(cpu, &runtime, [&w], [&v], 6).unwrap();
        let lhs = StackedTensorMap::pack(&[&a, &a])
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
        let rhs = StackedTensorMap::pack(&[&b, &b])
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
        let spec = super::super::super::ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        };
        let plan = ContractPlan::new(cuda, &lhs, &rhs, &spec).unwrap();
        assert!(plan.copy_c().is_some());
        let mut workspace = plan.workspace().unwrap();
        let expected = plan
            .execute(cuda, &lhs, &rhs, &mut workspace)
            .unwrap()
            .to_host(cuda)
            .unwrap()
            .storage;
        let bytes = workspace.retained_bytes();

        workspace.members_cuda.fail_next_before_output();
        let before = symtn_dense::cuda_transfer_stats();
        let error = plan
            .execute(cuda, &lhs, &rhs, &mut workspace)
            .map(|_| ())
            .unwrap_err();
        let failed = symtn_dense::cuda_transfer_stats();
        eprintln!("cuda CopyC error after the core: {error:?}");
        assert!(
            failed.gemm_calls > before.gemm_calls,
            "the core ran before the error"
        );
        assert_eq!(failed.copy_calls, before.copy_calls, "no output move ran");
        assert!(workspace.output.spare.is_some(), "the output is kept");
        assert!(workspace.take_output().is_none());
        assert_eq!(workspace.retained_bytes(), bytes, "the temporary is kept");
        let output = plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
        let recovered = symtn_dense::cuda_transfer_stats();
        assert_eq!(recovered.h2d_calls, failed.h2d_calls, "nothing re-uploaded");
        assert_eq!(output.to_host(cuda).unwrap().storage, expected);
    }
}

#[cfg(test)]
mod fermionic_unit_tests {
    use super::*;
    use crate::sector::{
        product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
        U1FusionRule, U1Irrep, Z2Irrep,
    };
    use crate::typed::GradedSpace;
    use symtn_operations::{fusion_replay::Rank2GemmBatchJob, Rank2Gemm, StridedHostKernelAdapter};

    #[derive(Default)]
    struct Count {
        calls: usize,
        jobs: usize,
        classes: Vec<(f64, usize)>,
        fail_negative: bool,
    }

    impl Rank2Gemm<f64> for Count {
        fn matmul_rank2(
            &mut self,
            _engine: &mut dyn Engine,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            _: usize,
            _: usize,
            _: usize,
            _: f64,
            _: f64,
        ) -> Result<(), OperationError> {
            panic!("direct member replay must submit one batch")
        }

        fn matmul_rank2_batch(
            &mut self,
            _engine: &mut dyn Engine,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            jobs: &[Rank2GemmBatchJob],
            _: &[usize],
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            assert_eq!(beta, 0.0);
            self.calls += 1;
            self.jobs += jobs.len();
            self.classes.push((alpha, jobs.len()));
            if self.fail_negative && alpha < 0.0 {
                return Err(OperationError::InvalidArgument {
                    message: "recording negative-class backend failure",
                });
            }
            Ok(())
        }
    }

    #[test]
    fn fermionic_unit_routes_submit_one_member_expanded_core_batch() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let u1 = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            ),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            ],
        )
        .unwrap();
        let su2 = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule).unwrap(),
            ),
            [
                (
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                    2,
                ),
                (
                    product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                    2,
                ),
                (
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                    1,
                ),
            ],
        )
        .unwrap();
        check_routes(engine, &runtime, &u1);
        check_routes(engine, &runtime, &su2);
    }

    fn check_routes<R>(engine: &mut dyn Engine, runtime: &Runtime, v: &GradedSpace<R>)
    where
        R: TypedSectorAdmission<
                Sector = <R as SectorCodec>::Sector,
                Error = symtn_core::FusionAlgebraError,
                Mode = symtn_core::MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let a = TensorMap::<_, f64>::rand_with_seed(engine, runtime, [v, v], [v, v], 61).unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed(engine, runtime, [v, v], [v, v], 62).unwrap();
        let cases = [
            ("C0", [2, 3], [0, 1], [0, 1, 2, 3], 2, false, false),
            ("C1", [3, 2], [1, 0], [0, 1, 2, 3], 2, false, false),
            ("C2", [0, 1], [2, 3], [2, 3, 0, 1], 2, false, true),
            ("C1p", [3, 2], [1, 0], [1, 0, 3, 2], 1, true, false),
            ("C2p", [0, 1], [2, 3], [3, 2, 1, 0], 1, true, false),
        ];
        for (name, lhs_axes, rhs_axes, output, split, copy_c, swapped_expected) in cases {
            let spec = super::super::super::ContractSpec {
                lhs: &lhs_axes,
                rhs: &rhs_axes,
                codomain: &output[..split],
                domain: &output[split..],
            };
            let left = StackedTensorMap::pack(&[&a]).unwrap();
            let right = StackedTensorMap::pack(&[&b]).unwrap();
            let plan = ContractPlan::new(engine, &left, &right, &spec).unwrap();
            assert_eq!(plan.copy_c().is_some(), copy_c, "{name}");
            assert!(!plan.resolution.is_dynamic_tree(), "{name}");
            let mut jobs_per_member = 0;
            for members in [1, 2, 17] {
                let (core, swapped) = plan.resolution.direct_core().expect("unit direct core");
                core.require_identity_direct_replay().unwrap();
                let replay = StackedDirectReplay::new_signed(Arc::clone(core), members).unwrap();
                if !copy_c {
                    assert_eq!(swapped, swapped_expected, "{name}");
                }
                let [dst_len, lhs_len, rhs_len] = replay.member_lens();
                let mut dst = vec![f64::NAN; dst_len * members];
                let lhs = vec![1.0; lhs_len * members];
                let rhs = vec![1.0; rhs_len * members];
                let mut count = Count::default();
                replay
                    .execute_host(
                        engine,
                        &mut StridedHostKernelAdapter::default(),
                        &mut count,
                        &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                            .unwrap(),
                        &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                        &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                        true,
                    )
                    .unwrap();
                if members == 1 {
                    jobs_per_member = count.jobs;
                    assert!(jobs_per_member > 0, "{name}");
                }
                assert_eq!(
                    (count.calls, count.jobs),
                    (1, jobs_per_member * members),
                    "{name}"
                );
                assert_eq!(count.classes, [(1.0, jobs_per_member * members)]);
            }
        }
    }

    #[test]
    fn signed_direct_submits_one_batch_per_present_sign_class() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let u1 = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            ),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            ],
        )
        .unwrap();
        let u1_full = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            ),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(1)), 1),
            ],
        )
        .unwrap();
        let su2 = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule).unwrap(),
            ),
            [
                (
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                    2,
                ),
                (
                    product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                    2,
                ),
                (
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                    1,
                ),
            ],
        )
        .unwrap();
        let odd_only = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            ),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
        )
        .unwrap();
        for swapped in [false, true] {
            check_signed_classes(engine, &runtime, &u1, 1, 2, swapped);
            check_signed_classes(engine, &runtime, &u1_full, 1, 2, swapped);
            check_signed_classes(engine, &runtime, &su2, 2, 1, swapped);
            check_signed_classes(engine, &runtime, &odd_only, 0, 1, swapped);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn signed_direct_can_leave_inactive_destination_blocks() {
        let cpu = &mut crate::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let rule = Arc::new(
            ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        );
        let v = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
            ],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            rule,
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            ],
        )
        .unwrap();
        let wd = w.try_dual().unwrap();
        let lhs =
            TensorMap::<_, f64>::from_subblock_fn(cpu, &runtime, [&v], [&wd], |_, _| 1.0).unwrap();
        let rhs =
            TensorMap::<_, f64>::from_subblock_fn(cpu, &runtime, [&wd], [&v], |_, _| 1.0).unwrap();
        let left = StackedTensorMap::pack(&[&lhs]).unwrap();
        let right = StackedTensorMap::pack(&[&rhs]).unwrap();
        let spec = super::super::super::ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        };
        let plan = ContractPlan::new(cpu, &left, &right, &spec).unwrap();
        let core = plan.resolution.direct_core().unwrap().0;
        core.require_identity_signed_direct_replay().unwrap();
        assert!(core.require_identity_direct_replay().is_err());
        assert!(!core.inactive_destination_regions().is_empty());
        assert!(core.distinct_direct_gemm_shapes() > 0);
        // Member Core and CopyC admit the signed core; only a DynamicTree
        // core must be unit.
        plan.resolution.admit_cuda_members().unwrap();
        let reversed = super::super::super::ContractSpec {
            codomain: &[1],
            domain: &[0],
            ..spec
        };
        let copy_c = ContractPlan::new(cpu, &left, &right, &reversed).unwrap();
        assert!(copy_c.copy_c().is_some());
        copy_c.resolution.admit_cuda_members().unwrap();
    }

    #[cfg(feature = "cuda")]
    fn cuda_dynamic_admission<R>(
        runtime: &Runtime,
        codomains: [&[&GradedSpace<R>]; 2],
        domains: [&[&GradedSpace<R>]; 2],
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output: &[usize],
    ) -> (bool, Result<(), OperationError>)
    where
        R: symtn_core::TypedSectorAdmission<
                Sector = <R as symtn_core::SectorCodec>::Sector,
                Error = symtn_core::FusionAlgebraError,
                Mode = symtn_core::MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let cpu = &mut crate::typed::CpuEngine::serial();
        let [a, b] = [0, 1].map(|i| {
            TensorMap::<_, f64>::from_subblock_fn(
                &mut *cpu,
                runtime,
                codomains[i].iter().copied(),
                domains[i].iter().copied(),
                |_, _| 1.0,
            )
            .unwrap()
        });
        let split = codomains[0].len() + domains[0].len() - lhs_axes.len();
        let spec = super::super::super::ContractSpec {
            lhs: lhs_axes,
            rhs: rhs_axes,
            codomain: &output[..split],
            domain: &output[split..],
        };
        let plan = ContractPlan::new(
            cpu,
            &StackedTensorMap::pack(&[&a]).unwrap(),
            &StackedTensorMap::pack(&[&b]).unwrap(),
            &spec,
        )
        .unwrap();
        (
            plan.resolution.is_dynamic_tree(),
            plan.resolution.admit_cuda_members(),
        )
    }

    /// Device-free pin of the public CUDA fixtures: U(1) and fermionic
    /// transformed-tree routes have Single-only transforms, SU(2) recoupling
    /// has a Multi transform and stays a typed capability error.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_dynamic_tree_admission_is_single_only() {
        let runtime = Runtime::builder().build().unwrap();
        let u1 = |charges: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                charges
                    .iter()
                    .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
            )
            .unwrap()
        };
        let v = u1(&[(0, 1), (1, 2), (2, 1)]);
        let w = u1(&[(0, 2), (1, 1)]);
        let x = u1(&[(-1, 1), (0, 2), (1, 2)]);
        for (name, codomains, domains, lhs_axes, rhs_axes, output) in [
            (
                "U(1) transformed lhs, inactive block",
                [&[&v][..], &[&w]],
                [&[&w, &v][..], &[&v]],
                &[1][..],
                &[0][..],
                &[0, 1, 2][..],
            ),
            (
                "U(1) reordered whole-side",
                [&[&v, &v], &[&x, &x]],
                [&[&x, &x], &[&v]],
                &[3, 2],
                &[0, 1],
                &[1, 0, 2],
            ),
        ] {
            let (dynamic, admitted) =
                cuda_dynamic_admission(&runtime, codomains, domains, lhs_axes, rhs_axes, output);
            assert!(dynamic, "{name}");
            admitted.unwrap_or_else(|err| panic!("{name}: {err}"));
        }
        let f = GradedSpace::try_new(
            Arc::new(
                ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            ),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(1)), 1),
            ],
        )
        .unwrap();
        let fd = f.try_dual().unwrap();
        // contract_cases::fermionic_twist_roles "A": the copied A carries the twist.
        let (dynamic, admitted) = cuda_dynamic_admission(
            &runtime,
            [&[&f, &f], &[&fd]],
            [&[&f], &[&f, &f]],
            &[1],
            &[0],
            &[0, 1, 2, 3],
        );
        assert!(dynamic);
        admitted.unwrap();
        let s = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let (dynamic, admitted) = cuda_dynamic_admission(
            &runtime,
            [&[&s, &s], &[&s]],
            [&[&s], &[&s, &s]],
            &[0],
            &[2],
            &[3, 0, 2, 1],
        );
        assert!(dynamic);
        assert!(matches!(
            admitted,
            Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA member transform requires unconjugated nonzero Single tasks"
            })
        ));
    }

    fn check_signed_classes<R>(
        engine: &mut dyn Engine,
        runtime: &Runtime,
        v: &GradedSpace<R>,
        positive: usize,
        negative: usize,
        swapped_expected: bool,
    ) where
        R: TypedSectorAdmission<
                Sector = <R as SectorCodec>::Sector,
                Error = symtn_core::FusionAlgebraError,
                Mode = symtn_core::MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let dual = v.try_dual().unwrap();
        let a = TensorMap::<_, f64>::from_subblock_fn(engine, runtime, [v], [&dual], |_, _| 1.0)
            .unwrap();
        let b = TensorMap::<_, f64>::from_subblock_fn(engine, runtime, [&dual], [v], |_, _| 1.0)
            .unwrap();
        let (lhs, rhs) = if swapped_expected { (b, a) } else { (a, b) };
        let left = StackedTensorMap::pack(&[&lhs]).unwrap();
        let right = StackedTensorMap::pack(&[&rhs]).unwrap();
        let lhs_axes = [usize::from(!swapped_expected)];
        let rhs_axes = [usize::from(swapped_expected)];
        let output = if swapped_expected { [1, 0] } else { [0, 1] };
        let spec = super::super::super::ContractSpec {
            lhs: &lhs_axes,
            rhs: &rhs_axes,
            codomain: &output[..1],
            domain: &output[1..],
        };
        let plan = ContractPlan::new(engine, &left, &right, &spec).unwrap();
        assert!(plan.copy_c().is_none());
        assert!(!plan.resolution.is_dynamic_tree());
        let (core, swapped) = plan.resolution.direct_core().unwrap();
        assert_eq!(core.require_identity_direct_replay().is_ok(), negative == 0);
        core.require_identity_signed_direct_replay().unwrap();
        for members in [1, 2, 17] {
            let replay = StackedDirectReplay::new_signed(Arc::clone(core), members).unwrap();
            assert_eq!(swapped, swapped_expected);
            let [dst_len, lhs_len, rhs_len] = replay.member_lens();
            let mut dst = vec![f64::NAN; dst_len * members];
            let lhs = vec![1.0; lhs_len * members];
            let rhs = vec![1.0; rhs_len * members];
            let mut count = Count::default();
            if negative > 0 && members == 1 {
                assert!(replay
                    .execute_host(
                        engine,
                        &mut StridedHostKernelAdapter::default(),
                        &mut count,
                        &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                            .unwrap(),
                        &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                        &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                        true,
                    )
                    .is_err());
                assert!(dst.iter().all(|value| value.is_nan()));
                assert_eq!(count.calls, 0);
            }
            replay
                .execute_signed_host(
                    engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut count,
                    &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                        .unwrap(),
                    &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                    &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                    true,
                )
                .unwrap();
            let expected: Vec<_> = [(1.0, positive), (-1.0, negative)]
                .into_iter()
                .filter(|(_, jobs)| *jobs > 0)
                .map(|(alpha, jobs)| (alpha, jobs * members))
                .collect();
            assert_eq!(count.classes, expected);
            assert_eq!(
                count.calls,
                usize::from(positive > 0) + usize::from(negative > 0)
            );
            assert_eq!(count.jobs, (positive + negative) * members);
            if negative > 0 && members == 1 {
                count.fail_negative = true;
                let error = replay
                    .execute_signed_host(
                        engine,
                        &mut StridedHostKernelAdapter::default(),
                        &mut count,
                        &mut StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len)
                            .unwrap(),
                        &StackedStorageView::new::<f64>(&lhs, lhs_len, members, lhs_len).unwrap(),
                        &StackedStorageView::new::<f64>(&rhs, rhs_len, members, rhs_len).unwrap(),
                        true,
                    )
                    .unwrap_err();
                assert!(matches!(
                    error,
                    OperationError::InvalidArgument {
                        message: "recording negative-class backend failure"
                    }
                ));
            }
        }
    }
}
