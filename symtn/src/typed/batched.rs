//! Structure signatures: the key under which tensors can share one prepared
//! plan and one stacked payload layout (#1287).
//!
//! Neither TensorKit nor QSpace has a batched path, so the signature is
//! defined by what symtn's eager operations read from an operand: the rule
//! authority, the logical hom space, the fusion-tree block structure, the
//! payload placement and the Runtime. The payload dtype is the type parameter
//! `D` of every consumer and is therefore checked by the compiler, not here.
//!
//! # Failure rule of every batched plan
//!
//! [`ComposePlan`], [`ContractPlan`] and [`EighFullPlan`] (Host and CUDA)
//! share one rule. A workspace created by another plan is rejected before it
//! is touched. Otherwise a failed `execute` or `execute_into` (a validation
//! error, a signature mismatch or a backend error) leaves no observable
//! workspace output: `take_output` returns `None` until the next successful
//! `execute`, which rewrites every member. The failed call's buffers stay in
//! the workspace and are reused, so the next call allocates no more than a
//! warm one. After a failed `execute_into` the caller's destination contents
//! are unspecified; a validation error never writes them.

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{
    BlockStructureContent, CheckedFusionAlgebra, CoupledSectorRegion, HomSpaceId,
    MultiplicityFreeRigidSymbols, Placement, RuleIdentity, SectorCodec, SectorId, TensorStorage,
};
use symtn_matrixalgebra::HermitianTol;
#[cfg(test)]
use symtn_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use symtn_tensors::{FusionOperand, OperationError};

#[cfg(feature = "cuda")]
use super::{dense_err, CudaFactorizationPayload, CudaPayload, CudaStorage};
use super::{
    owned_repr, require_restriction_set, restricted_space, restriction_runs,
    BoundDynamicFusionMapSpace, ExecToken, FacadeError, FactorExec, FactorizationScalar,
    LegSelection, Runtime, SectorSpectrum, TensorMap, TensorScalar, TypedData,
    TypedSectorAdmission, TypedTensorBody, TypedTensorRepr, TypedTensorRootDispatch, EXEC,
};
use crate::error::Error;
use crate::runtime::RuntimeIdentity;
use crate::tensor_product::internal_layout_error;

#[path = "contract_batch.rs"]
mod contract_batch;
pub use contract_batch::{ComposePlan, ContractPlan, ContractWorkspace};

mod eigh;
pub(in crate::typed) use eigh::StackPair;
pub use eigh::{EighFullPlan, EighFullWorkspace, EighStackOutput};

#[cfg(test)]
mod tests;

/// Content identity of a tensor's structure, placement and Runtime.
///
/// Two signatures are equal exactly when the fusion-rule identity, the hom
/// space (every codomain and domain leg's sectors, degeneracies and dual
/// flags), the block structure content (fusion-tree blocks, their order and
/// dense offsets), the placement and the Runtime are all equal. Equality
/// never depends on process-local intern ids, so it survives interner
/// eviction, `symtn::cache::clear` and oversized structures that bypass
/// the interner.
///
/// The signature describes the logical structure only. It does not record
/// whether the payload is an owned dense buffer, a lazy adjoint or a compact
/// diagonal spectrum; a consumer that relies on the dense layout must admit
/// the representation separately.
///
/// Comparison is O(1) while the compared structures share their interned
/// allocations, and O(blocks) otherwise. `Hash` covers only O(1) prehashes;
/// signatures that differ only in block structure share a bucket and are
/// separated by `Eq`. A signature holds a non-owning Runtime handle and does
/// not keep the Runtime alive.
///
/// Likewise, a rule whose identity is a category or group value (such as
/// `CategoryRule` or `LieGroupFusionRule`) is held weakly: once every rule
/// over that value is dropped, its signatures equal only their own clones,
/// never a signature built later over an equal value.
#[derive(Clone)]
pub struct StructureSignature {
    rule: RuleIdentity,
    homspace: HomSpaceId,
    // Why the content and not `content_id`: the id is a process-local
    // insertion counter reissued after eviction or reset, and oversized
    // contents never enter the interner, so equal content can carry
    // different ids.
    structure: Arc<BlockStructureContent>,
    placement: Placement,
    runtime: RuntimeIdentity,
}

impl PartialEq for StructureSignature {
    fn eq(&self, other: &Self) -> bool {
        self.first_mismatch(other).is_none()
    }
}

impl Eq for StructureSignature {}

impl Hash for StructureSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rule.hash(state);
        self.homspace.hash(state);
        self.placement.hash(state);
        self.runtime.hash(state);
    }
}

/// The first determinant in which two [`StructureSignature`]s differ.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SignatureField {
    /// Host or device ordinal.
    Placement,
    /// The owning Runtime.
    Runtime,
    /// The fusion-rule identity.
    Rule,
    /// A codomain or domain leg's sectors, degeneracies or dual flag.
    HomSpace,
    /// The fusion-tree blocks, their order or dense offsets.
    BlockStructure,
}

/// A payload representation a [`StackedTensorMap`] cannot hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BatchMemberRepresentation {
    /// A lazy adjoint view over its parent's payload.
    LazyAdjoint,
    /// A compact diagonal spectrum, whose dense layout is not stored.
    CompactDiagonal,
}

impl StructureSignature {
    /// The first field in which `other` differs, in the order `==` checks
    /// them, or `None` when the signatures are equal.
    pub(crate) fn first_mismatch(&self, other: &Self) -> Option<SignatureField> {
        if self.placement != other.placement {
            Some(SignatureField::Placement)
        } else if self.runtime != other.runtime {
            Some(SignatureField::Runtime)
        } else if self.rule != other.rule {
            Some(SignatureField::Rule)
        } else if self.homspace != other.homspace {
            Some(SignatureField::HomSpace)
        } else if !(Arc::ptr_eq(&self.structure, &other.structure)
            || self.structure == other.structure)
        {
            Some(SignatureField::BlockStructure)
        } else {
            None
        }
    }
}

impl std::fmt::Debug for StructureSignature {
    // Why not derive: the hom-space key and block list are O(legs + blocks)
    // and would make a diagnostic proportional to the tensor's structure.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StructureSignature")
            .field("rule", &self.rule)
            .field("content_id", &self.structure.id())
            .field("placement", &self.placement)
            .field("runtime_alive", &self.runtime.is_alive())
            .finish_non_exhaustive()
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    S: TensorStorage<D>,
{
    /// The [`StructureSignature`] of this tensor's logical structure.
    ///
    /// Builds no new structure: it clones the shared hom-space and block
    /// structure handles and a weak Runtime handle.
    pub fn structure_signature(&self) -> StructureSignature {
        StructureSignature::of_space(self.logical_space(), self.placement(), &self.runtime)
    }
}

impl StructureSignature {
    /// The signature of a tensor over `space` with `placement` on `runtime`.
    fn of_space<R: TypedSectorAdmission>(
        space: &BoundDynamicFusionMapSpace<R>,
        placement: Placement,
        runtime: &Runtime,
    ) -> Self {
        // Why the admission's identity first: it is the identity the layout
        // was validated against. Every bound space carries one; the provider
        // fallback clones the identity the rule stored at construction.
        let rule = match space.space().admission().rule_identity() {
            Some(identity) => identity.clone(),
            None => TypedSectorAdmission::typed_rule_identity(space.provider()),
        };
        StructureSignature {
            rule,
            homspace: space.space().homspace().id(),
            structure: space.space().structure().content_key(),
            placement,
            runtime: runtime.identity(),
        }
    }
}

/// `B` tensors of one [`StructureSignature`] in one buffer: member `i`
/// occupies `[i * L, (i + 1) * L)`, where `L` is the signature's required
/// payload length. There is no padding and no member leg; the member axis
/// exists only as this stride.
///
/// A stack is an operand of a batched operation, not a tensor. It
/// deliberately does not implement `TensorStorage`, whose `len` would be
/// `B * L` and would let an ordinary per-tensor kernel run on member 0 only:
///
/// ```compile_fail
/// use symtn::expert::TensorStorage;
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::StackedTensorMap;
///
/// fn as_storage<S: TensorStorage<f64>>(_: &S) {}
/// fn reject(stack: &StackedTensorMap<U1FusionRule, f64>) {
///     as_storage(stack);
/// }
/// ```
///
/// The same imports compile when the stack is used as a stack:
///
/// ```
/// use symtn::expert::TensorStorage;
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::StackedTensorMap;
///
/// fn as_storage<S: TensorStorage<f64>>(_: &S) {}
/// fn accept(stack: &StackedTensorMap<U1FusionRule, f64>) {
///     let _ = stack.signature();
///     as_storage(&vec![0.0_f64]);
/// }
/// ```
pub struct StackedTensorMap<R, D, S = Vec<D>> {
    runtime: Runtime,
    space: BoundDynamicFusionMapSpace<R>,
    signature: StructureSignature,
    storage: S,
    members: usize,
    member_len: usize,
    _payload: PhantomData<D>,
}

impl<R, D, S> StackedTensorMap<R, D, S> {
    /// The signature every member shares, with the stack's placement.
    pub fn signature(&self) -> &StructureSignature {
        &self.signature
    }

    /// Number of legs of every member.
    fn rank(&self) -> usize {
        let homspace = self.space.space().homspace();
        homspace.codomain().len() + homspace.domain().len()
    }

    /// The member count `B`. Never zero: [`Self::pack`] rejects an empty
    /// batch.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.members
    }

    /// Checks a [`Self::select`] index list and returns the selected payload
    /// length `B' * L`.
    fn selection_len(&self, members: &[usize]) -> Result<usize, Error> {
        if members.is_empty() {
            return Err(Error::InvalidArgument(
                "a stack needs at least one member".into(),
            ));
        }
        if let Some(&member) = members.iter().find(|&&member| member >= self.members) {
            return Err(Error::BatchMemberOutOfRange {
                member,
                len: self.members,
            });
        }
        self.member_len
            .checked_mul(members.len())
            .ok_or_else(|| Error::InvalidArgument("stacked payload length overflows usize".into()))
    }

    /// A stack of this one's member count and Runtime over `space`, holding
    /// `storage`; the placement is read from `storage`.
    fn with_space<T: TensorStorage<D>>(
        &self,
        space: BoundDynamicFusionMapSpace<R>,
        storage: T,
    ) -> Result<StackedTensorMap<R, D, T>, Error>
    where
        R: TypedSectorAdmission,
    {
        Ok(StackedTensorMap {
            runtime: self.runtime.clone(),
            signature: StructureSignature::of_space(&space, storage.placement(), &self.runtime),
            member_len: space.space().required_len()?,
            space,
            storage,
            members: self.members,
            _payload: PhantomData,
        })
    }

    /// This stack's space, signature and Runtime over `storage` holding
    /// `members` members; the placement is read from `storage`.
    fn with_storage<T: TensorStorage<D>>(
        &self,
        storage: T,
        members: usize,
    ) -> StackedTensorMap<R, D, T> {
        StackedTensorMap {
            runtime: self.runtime.clone(),
            space: self.space.clone(),
            signature: StructureSignature {
                placement: storage.placement(),
                ..self.signature.clone()
            },
            storage,
            members,
            member_len: self.member_len,
            _payload: PhantomData,
        }
    }
}

impl<R, D> StackedTensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: Copy,
{
    /// Copies `members` into one stacked buffer: one allocation of `B * L`
    /// elements and `B` copies.
    ///
    /// Every member is checked before anything is copied. A lazy adjoint or
    /// a compact diagonal member returns [`Error::UnsupportedBatchMember`];
    /// a member whose signature differs from member 0's returns
    /// [`Error::BatchSignatureMismatch`] naming that member and the first
    /// differing field. An empty batch is an [`Error::InvalidArgument`].
    ///
    /// Lazy adjoints are not packed. Materialize one before passing it to the
    /// identity-oriented compose plan; oriented batching is outside this API.
    pub fn pack<T: AsRef<TensorMap<R, D>>>(members: &[T]) -> Result<Self, Error> {
        let first = members
            .first()
            .ok_or_else(|| Error::InvalidArgument("a stack needs at least one member".into()))?
            .as_ref();
        let signature = first.structure_signature();
        let payloads = members
            .iter()
            .enumerate()
            .map(|(member, tensor)| {
                let tensor = tensor.as_ref();
                let payload = dense_payload(tensor).map_err(|representation| {
                    Error::UnsupportedBatchMember {
                        member,
                        representation,
                    }
                })?;
                match signature.first_mismatch(&tensor.structure_signature()) {
                    Some(field) => Err(Error::BatchSignatureMismatch {
                        member: Some(member),
                        field,
                    }),
                    None => Ok(payload),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let space = first.logical_space().clone();
        let member_len = space.space().required_len()?;
        let total = member_len.checked_mul(members.len()).ok_or_else(|| {
            Error::InvalidArgument("stacked payload length overflows usize".into())
        })?;
        let mut storage = Vec::with_capacity(total);
        for payload in payloads {
            storage.extend_from_slice(payload);
        }
        Ok(Self {
            runtime: first.runtime.clone(),
            space,
            signature,
            storage,
            members: members.len(),
            member_len,
            _payload: PhantomData,
        })
    }

    /// An owned, bit-identical copy of member `i`.
    ///
    /// An index at or past [`Self::len`] is
    /// [`Error::BatchMemberOutOfRange`].
    pub fn member(&self, engine: &mut dyn Engine, i: usize) -> Result<TensorMap<R, D>, Error> {
        super::admit_host_engine(engine)?;
        if i >= self.members {
            return Err(Error::BatchMemberOutOfRange {
                member: i,
                len: self.members,
            });
        }
        let start = i * self.member_len;
        let data = self.storage[start..start + self.member_len].to_vec();
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.space.clone(), data)),
        })
    }
}

impl<R, D, S: StackExec<D>> StackedTensorMap<R, D, S> {
    /// A new stack whose member `j` is member `members[j]` of this one, with
    /// the same signature, placement and Runtime.
    ///
    /// Indices may appear in any order and may repeat (a repeated index
    /// replicates that member), so the result can be longer than this stack.
    /// An index at or past [`Self::len`] is
    /// [`Error::BatchMemberOutOfRange`]; an empty list is an
    /// [`Error::InvalidArgument`], because a stack is never empty. Both are
    /// checked before anything is allocated.
    ///
    /// This is how a caller drops the members a batched handle rejected, or
    /// splits a stack into groups whose truncated signatures agree.
    ///
    /// # Cost
    ///
    /// Host: one allocation of `members.len() * L` elements and one slice
    /// copy per selected member. CUDA: one upload of the `members.len()`
    /// member offsets and one Tenferro `gather` launch per call, whatever
    /// `B` and the block structure; the gather allocates the result buffer.
    /// Nothing is downloaded and nothing waits on the device.
    pub fn select(&self, engine: &mut dyn Engine, members: &[usize]) -> Result<Self, Error> {
        let len = self.selection_len(members)?;
        S::select_members(engine, EXEC, self, members, len)
    }
}

impl<R, D, S> StackedTensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
    S: StackExec<D>,
{
    /// Restricts the listed legs of every member: member `i` of the result
    /// is `self.member(i)?.restrict_leg(engine, legs)?`, bit for bit, with the same
    /// space and signature.
    ///
    /// One set of selections serves every member, which is what grouping
    /// members by their truncated signature (then [`Self::select`])
    /// produces.
    ///
    /// # Cost
    ///
    /// Host: the eager block plan is built once; each destination block is
    /// then one strided copy for all `B` members and all restricted legs, the
    /// member axis being one more dimension of the copy. `B · L'` elements
    /// move into one output allocation, left unfilled when the blocks tile it
    /// (as eager), plus `O(rank + blocks)` structural work independent of
    /// `B`.
    ///
    /// CUDA: one upload of an `L'`-entry element table and one Tenferro
    /// `gather` launch per call, whatever `B`, the block count and the number
    /// of restricted legs. Nothing is downloaded and nothing waits on the
    /// device. The table maps each destination element to its source element
    /// within a member. It is the eager restriction kernel applied once, on
    /// the Host, to the payload `0, 1, …, L - 1`, so the block plan has one
    /// implementation; the gather then reads that element of every member
    /// (its window is the member axis) and allocates the `[L', B]` result.
    ///
    /// # Errors
    ///
    /// Exactly the eager [`TensorMap::restrict_leg`] errors for a dense
    /// payload: an empty set, an out-of-range or repeated axis, or a leg
    /// other than [`LegSelection::parent`] is an [`Error::InvalidArgument`],
    /// a selection of another rule is [`Error::RuleMismatch`]. Nothing is
    /// allocated before every check.
    pub fn restrict_leg(
        &self,
        engine: &mut dyn Engine,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<Self, FacadeError<R>>
    where
        R::Mode: TypedTensorRootDispatch<R>,
    {
        require_restriction_set(&self.space, legs)?;
        let destination = restricted_space(&self.space, legs)?;
        Ok(S::restrict_members(engine, EXEC, self, destination, legs)?)
    }
}

/// The member-movement kernels of a stack for one storage (#1756):
/// [`StackedTensorMap::select`] and [`StackedTensorMap::restrict_leg`] own
/// the index checks and the result space, and call one of these, last. Each
/// returns the finished stack, so a storage's resource scope (the Host pool,
/// the device lease) spans exactly what it spanned before.
///
/// Why not a subtrait of [`super::TypedStorage`]: these kernels move member
/// elements and never read the rule, and the CUDA `TypedStorage` impl is
/// bounded by the multiplicity-free rules, which would remove the device
/// `select` and `restrict_leg` of a checked Generic stack.
#[doc(hidden)]
pub trait StackExec<D>: TensorStorage<D> + super::storage::sealed::Sealed + Sized {
    /// The stack of `stack`'s members `members`, whose checked payload
    /// length is `len`.
    fn select_members<R>(
        engine: &mut dyn Engine,
        _: ExecToken,
        stack: &StackedTensorMap<R, D, Self>,
        members: &[usize],
        len: usize,
    ) -> Result<StackedTensorMap<R, D, Self>, Error>;

    /// The stack over the checked `destination` holding every member of
    /// `stack` restricted by `legs`.
    fn restrict_members<R: TypedSectorAdmission>(
        engine: &mut dyn Engine,
        _: ExecToken,
        stack: &StackedTensorMap<R, D, Self>,
        destination: BoundDynamicFusionMapSpace<R>,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<StackedTensorMap<R, D, Self>, Error>
    where
        D: TensorScalar;
}

impl<D: Copy> StackExec<D> for Vec<D> {
    fn select_members<R>(
        engine: &mut dyn Engine,
        _: ExecToken,
        stack: &StackedTensorMap<R, D>,
        members: &[usize],
        len: usize,
    ) -> Result<StackedTensorMap<R, D>, Error> {
        super::admit_host_engine(engine)?;
        let mut storage = Vec::with_capacity(len);
        for &member in members {
            storage
                .extend_from_slice(&stack.storage[member * stack.member_len..][..stack.member_len]);
        }
        Ok(stack.with_storage(storage, members.len()))
    }

    fn restrict_members<R: TypedSectorAdmission>(
        engine: &mut dyn Engine,
        _: ExecToken,
        stack: &StackedTensorMap<R, D>,
        destination: BoundDynamicFusionMapSpace<R>,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<StackedTensorMap<R, D>, Error>
    where
        D: TensorScalar,
    {
        super::admit_host_engine(engine)?;
        let data = symtn_tensors::stacked_fusion_restrict_owned(
            destination.space().structure(),
            FusionOperand::direct(stack.space.space()),
            &stack.storage,
            stack.members,
            &restriction_runs(stack.rank(), legs),
        )
        .map_err(Error::from)?;
        stack.with_space(destination, data)
    }
}

/// The workspace-owned output of a batched plan under the module's failure
/// rule: at most one of `output` and `spare` is set.
struct OutputSlot<T> {
    /// The output of the last successful call: the only observable one.
    output: Option<T>,
    /// Buffers kept for reuse after a failed call. Never observable: a failed
    /// call may have written part of them.
    spare: Option<T>,
}

impl<T> Default for OutputSlot<T> {
    fn default() -> Self {
        Self {
            output: None,
            spare: None,
        }
    }
}

impl<T> OutputSlot<T> {
    fn take_output(&mut self) -> Option<T> {
        self.output.take()
    }

    /// Takes the buffers for a call; the previous output stays unobservable
    /// until the call publishes.
    fn take(&mut self) -> Option<T> {
        let spare = self.spare.take();
        self.output.take().or(spare)
    }

    /// Keeps `buffers` as the unobservable spare and returns `error`.
    fn fail<E>(&mut self, buffers: Option<T>, error: E) -> E {
        self.spare = buffers;
        error
    }

    /// Hides the output after a failed call that did not take the buffers.
    fn hide<E>(&mut self, error: E) -> E {
        let buffers = self.take();
        self.fail(buffers, error)
    }

    /// Makes `buffers` the observable output of a successful call.
    fn publish(&mut self, buffers: T) -> &T {
        self.output.insert(buffers)
    }

    /// Publishes `buffers` after a successful call, or keeps them as the spare.
    fn settle<E>(&mut self, buffers: T, result: Result<(), E>) -> Result<&T, E> {
        match result {
            Ok(()) => Ok(self.publish(buffers)),
            Err(error) => Err(self.fail(Some(buffers), error)),
        }
    }

    /// The output and the spare, for retained-byte accounting.
    fn buffers(&self) -> impl Iterator<Item = &T> {
        self.output.iter().chain(&self.spare)
    }
}

/// Why a [`BatchError::MemberRejected`] member failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MemberFault {
    /// A coupled-sector block is not Hermitian under the eager rule
    /// `||(A - A†)/2||_F <= 64 eps ||A||_F`, or holds a non-finite entry.
    NotHermitian,
    /// The solver returned a non-finite eigenvalue.
    NonFiniteEigenvalue,
}

/// The error of a prepared batched factorization. A batch fails as a whole:
/// no member's output is returned when any member fails.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BatchError {
    /// Every member that failed, in member order, with its reason. Admission
    /// rejections are found before any solver work.
    MemberRejected {
        /// `(member, fault)` for each failing member.
        members: Vec<(usize, MemberFault)>,
    },
    /// The device solver failed on coupled-sector block `block` (the index of
    /// the block among the source's coupled sectors, in layout order).
    ///
    /// The failing member is not known: Tenferro 0.7.1 reads the solver
    /// status of a whole batch and reports its first failure without the
    /// member index.
    Solver {
        /// The coupled-sector block whose batched solve failed.
        block: usize,
        /// The backend's error.
        source: Error,
    },
    /// A structural, capability or backend error, as the eager operation
    /// reports it. Nothing is attributed to a member.
    Operation(Error),
}

impl From<Error> for BatchError {
    fn from(error: Error) -> Self {
        Self::Operation(error)
    }
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemberRejected { members } => write!(f, "batch members rejected: {members:?}"),
            Self::Solver { block, .. } => {
                write!(f, "solver failed on coupled-sector block {block}")
            }
            // Transparent: the eager operation's own error, reported as is.
            Self::Operation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for BatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MemberRejected { .. } => None,
            Self::Solver { source, .. } => Some(source),
            Self::Operation(error) => std::error::Error::source(error),
        }
    }
}

/// The owned dense payload of `tensor`, or the representation a stack
/// cannot hold.
fn dense_payload<R, D>(tensor: &TensorMap<R, D>) -> Result<&[D], BatchMemberRepresentation> {
    match &tensor.repr {
        TypedTensorRepr::Adjoint(_) => Err(BatchMemberRepresentation::LazyAdjoint),
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(data) => Ok(data),
            TypedData::Diagonal(_) => Err(BatchMemberRepresentation::CompactDiagonal),
        },
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> StackedTensorMap<R, D> {
    /// Uploads the whole stack with one host-to-device transfer into the
    /// caller's CUDA engine.
    pub fn to_cuda(
        &self,
        engine: &mut dyn Engine,
    ) -> Result<StackedTensorMap<R, D, CudaStorage<D>>, Error> {
        let lease = crate::runtime::cuda_call(engine)?;
        let storage = CudaStorage::upload_members(
            &lease,
            self.storage.clone(),
            self.member_len,
            self.members,
        )?;
        Ok(self.with_storage(storage, self.members))
    }
}

#[cfg(feature = "cuda")]
impl<R, D: CudaPayload> StackedTensorMap<R, D, CudaStorage<D>> {
    /// Downloads the whole stack with one device-to-host transfer, through
    /// the caller's CUDA engine that owns it.
    pub fn to_host(&self, engine: &mut dyn Engine) -> Result<StackedTensorMap<R, D>, Error> {
        let lease = crate::runtime::cuda_call(engine)?;
        let storage = self.storage.download(&lease)?;
        Ok(self.with_storage(storage, self.members))
    }
}

#[cfg(feature = "cuda")]
impl<D: CudaPayload> StackExec<D> for CudaStorage<D> {
    fn select_members<R>(
        engine: &mut dyn Engine,
        _: ExecToken,
        stack: &StackedTensorMap<R, D, Self>,
        members: &[usize],
        _len: usize,
    ) -> Result<StackedTensorMap<R, D, Self>, Error> {
        let mut lease = crate::runtime::cuda_call(engine)?;
        let storage =
            stack
                .storage
                .gather_members(&mut lease, stack.member_len, stack.members, members)?;
        Ok(stack.with_storage(storage, members.len()))
    }

    fn restrict_members<R: TypedSectorAdmission>(
        engine: &mut dyn Engine,
        _: ExecToken,
        stack: &StackedTensorMap<R, D, Self>,
        destination: BoundDynamicFusionMapSpace<R>,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<StackedTensorMap<R, D, Self>, Error>
    where
        D: TensorScalar,
    {
        // The element table is the eager kernel itself run over the payload
        // `0, 1, …, L - 1` (in `i64`, the gather's index type), so the block
        // plan keeps one implementation. Why not enumerate the destination
        // blocks here instead: that would restate the sector lookup, start
        // offsets and rectangle checks of `restrict_block`. The price is one
        // `L`-element iota per call, host work `O(L + L')`, independent of `B`
        // and of the number of restricted legs.
        let elements = {
            let iota = (0..stack.member_len)
                .map(i64::try_from)
                .collect::<Result<Vec<i64>, _>>()
                .map_err(|_| Error::InvalidArgument("restrict_leg: a member exceeds i64".into()))?;
            symtn_tensors::oriented_fusion_restrict_owned(
                destination.space().structure(),
                FusionOperand::direct(stack.space.space()),
                &iota,
                &restriction_runs(stack.rank(), legs),
            )
            .map_err(Error::from)?
        };
        let mut lease = crate::runtime::cuda_call(engine)?;
        let storage = stack
            .storage
            .gather_member_elements(&mut lease, stack.member_len, stack.members, elements)
            .map_err(Error::from)?;
        drop(lease);
        stack.with_space(destination, storage)
    }
}
