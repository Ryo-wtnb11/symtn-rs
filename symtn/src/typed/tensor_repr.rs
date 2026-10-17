use super::*;

/// Storage shared by every clone of one typed tensor map: the admitted space
/// and its block payload.
pub(super) struct TypedTensorBody<R, D, S = Vec<D>> {
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    /// Why the payload carries its own reference count rather than sitting
    /// inline in the body: an operation that rewrites only the *space* and
    /// leaves every stored value where it is — inserting or removing a unit
    /// leg, whose trivial sector adds no block and reorders nothing — must be
    /// an O(1) metadata edit. Inline, such an operation would have to copy the
    /// whole `Vec<D>` to build a body with the new space; behind this `Arc` it
    /// clones a pointer. That O(1)-reuse argument holds for **dense payloads
    /// only**: `TypedData::Diagonal` is a bond-space-only representation, and
    /// reusing one under a rewritten non-bond space would leave `spectrum()`
    /// returning `Some` and send `exp`/`inv`/`pinv`/`scale` down their compact
    /// elementwise arms on a tensor that is no longer an endomorphism. The
    /// Group 4 (#580) contract is therefore: materialize a compact payload to
    /// dense *before* changing the space, exactly as the references do —
    /// TensorKit 0.17 shares `t.data` only for ordinary `TensorMap` and routes
    /// `DiagonalTensorMap` through the generic similar+block-copy branch
    /// (`src/tensors/indexmanipulations.jl:124-136,158-195`). The Group 4
    /// slice (#580 PR 5) holds that contract in
    /// [`TensorMap::shareable_dense_payload`]: a dense
    /// payload is shared at pointer cost, a compact one is materialized into
    /// a *fresh* dense payload (one copy). And `TypedData::Diagonal` must not be
    /// broadened to non-bond spaces without separately proving every compact
    /// fast path (`spectrum`/`exp`/`inv`/`pinv`/`scale`).
    /// Clone-then-modify keeps the same property from the
    /// other side: two *bodies* can share one payload until one of them
    /// writes (the unit-leg operations build new bodies over old dense
    /// payloads; every write route publishes a new payload instead of
    /// reaching through the `Arc`).
    pub(super) data: Arc<TypedData<D, S>>,
}

impl<R, D, S> TypedTensorBody<R, D, S> {
    /// A body holding an already-dense payload.
    pub(super) fn dense(space: BoundDynamicFusionMapSpace<R>, data: S) -> Self {
        Self::new(space, TypedData::Dense(data))
    }

    pub(super) fn new(space: BoundDynamicFusionMapSpace<R>, data: TypedData<D, S>) -> Self {
        Self {
            space,
            data: Arc::new(data),
        }
    }

    /// A body installing an already-shared payload under a (usually
    /// rewritten) space — the unit-leg operations' O(1) dense reuse
    /// (#580 PR 5).
    pub(super) fn with_shared_payload(
        space: BoundDynamicFusionMapSpace<R>,
        data: Arc<TypedData<D, S>>,
    ) -> Self {
        Self { space, data }
    }
}

impl<R, D, S> TypedTensorBody<R, D, S> {
    /// A body holding a compact spectrum payload.
    pub(super) fn diagonal(
        space: BoundDynamicFusionMapSpace<R>,
        spectrum: Vec<symtn_matrixalgebra::SectorSpectrum<D>>,
    ) -> Self {
        Self::new(space, TypedData::Diagonal(spectrum))
    }
}

pub(super) struct TypedAdjointView<R, D, S = Vec<D>> {
    pub(super) parent: Arc<TypedTensorBody<R, D, S>>,
    pub(super) logical_space: BoundDynamicFusionMapSpace<R>,
    /// Set only on the operation-local header a [`TensorRef`] resolves to:
    /// the materialization path refuses it instead of copying.
    pub(super) borrowed: bool,
}

impl<R, D, S> TypedAdjointView<R, D, S> {
    /// A lazy adjoint over a dense parent.
    ///
    /// Why not a compact diagonal parent: densifying its unstored zeros
    /// through conjugation would publish them as `0-0i`, so every `adjoint`
    /// entry point returns the owned conjugated diagonal through
    /// `compact_adjoint` first, and flip, twist and device transfer only
    /// re-wrap parents that are already dense. Why not a dense-only parent
    /// type: `parent` is the same `Arc` an owned tensor holds, which keeps
    /// adjoint-of-adjoint an `O(1)` handle swap.
    pub(super) fn new(
        parent: Arc<TypedTensorBody<R, D, S>>,
        logical_space: BoundDynamicFusionMapSpace<R>,
    ) -> Self {
        debug_assert!(
            matches!(parent.data.as_ref(), TypedData::Dense(_)),
            "a lazy adjoint never holds a compact diagonal parent"
        );
        Self {
            parent,
            logical_space,
            borrowed: false,
        }
    }
}

/// Borrowed view of a [`TensorMap`], optionally as its adjoint.
///
/// `&t` converts into the plain view and [`TensorMap::adjoint_view`] gives
/// the adjoint view (TensorKit `adjoint(t)`, TensorOperations' `conjA`
/// flag). Operations take their second tensor as `impl Into<TensorRef>`, so
/// both are passed the same way. The view copies nothing: it resolves, inside
/// the call, to exactly the operand `&t.adjoint()?` would have been and runs
/// the same code, so the result is the same.
///
/// An operation that would have to materialize the adjoint returns
/// [`Error::Unsupported`] with [`crate::error::Alternative::Materialize`]
/// instead of copying; pass `&t.adjoint()?.materialize()?` there.
///
/// ```
/// use std::sync::Arc;
///
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{GradedSpace, Runtime, TensorMap};
///
/// let runtime = Runtime::builder().build()?;
/// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
/// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1)?;
/// let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 2)?;
/// assert_eq!(a.inner(b.adjoint_view())?, a.inner(&b.adjoint()?)?);
/// # Ok::<(), symtn::typed::Error>(())
/// ```
pub struct TensorRef<'a, R, D, S = Vec<D>> {
    pub(super) base: &'a TensorMap<R, D, S>,
    /// The base's own `adjoint`, captured where it exists; `None` for the
    /// plain view. Why a `fn` pointer and not a bound on each operation:
    /// `adjoint` needs the mode's adjoint space (`TypedAdjointSpace`), which
    /// `contract` and `compose` do not.
    pub(super) adjoint: Option<ViewAdjoint<R, D, S>>,
}

pub(super) type ViewAdjoint<R, D, S> = fn(&TensorMap<R, D, S>) -> Result<TensorMap<R, D, S>, Error>;

// Why hand-written: the derives would demand `R`, `D`, `S: Clone`.
impl<R, D, S> Clone for TensorRef<'_, R, D, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R, D, S> Copy for TensorRef<'_, R, D, S> {}

impl<'a, R, D, S> From<&'a TensorMap<R, D, S>> for TensorRef<'a, R, D, S> {
    fn from(base: &'a TensorMap<R, D, S>) -> Self {
        Self {
            base,
            adjoint: None,
        }
    }
}

impl<'a, R, D, S> TensorRef<'a, R, D, S> {
    /// The operand an owned-lazy-adjoint call would have received. A plain
    /// view is its base, unchanged, so `&t` keeps today's behavior, including
    /// the owned lazy adjoint's implicit materialization until #1548. An
    /// adjoint view is `adjoint(base)`, marked so that the materialization
    /// path refuses it.
    pub(super) fn operand(self) -> Result<std::borrow::Cow<'a, TensorMap<R, D, S>>, Error> {
        let Some(adjoint) = self.adjoint else {
            return Ok(std::borrow::Cow::Borrowed(self.base));
        };
        let resolved = adjoint(self.base)?;
        Ok(std::borrow::Cow::Owned(resolved.into_borrowed_operand()))
    }
}

pub(super) fn borrowed_view_unsupported(operation: &'static str) -> Error {
    Error::Unsupported {
        operation,
        alternative: crate::error::Alternative::Materialize,
    }
}

impl<R, D, S> TensorMap<R, D, S> {
    /// Refuses a resolved adjoint view before `operation` does any work,
    /// for operations whose only route for a lazy adjoint materializes it.
    pub(super) fn refuse_borrowed_view(&self, operation: &'static str) -> Result<(), Error> {
        match &self.repr {
            TypedTensorRepr::Adjoint(view) if view.borrowed => {
                Err(borrowed_view_unsupported(operation))
            }
            _ => Ok(()),
        }
    }

    fn into_borrowed_operand(mut self) -> Self {
        let TypedTensorRepr::Adjoint(view) = &mut self.repr else {
            return self;
        };
        if let Some(unique) = Arc::get_mut(view) {
            unique.borrowed = true;
            return self;
        }
        let mut marked =
            TypedAdjointView::new(Arc::clone(&view.parent), view.logical_space.clone());
        marked.borrowed = true;
        *view = Arc::new(marked);
        self
    }
}

pub(super) enum TypedTensorRepr<R, D, S = Vec<D>> {
    Owned(Arc<TypedTensorBody<R, D, S>>),
    Adjoint(Arc<TypedAdjointView<R, D, S>>),
}

/// Owned physical-basis tensor data on the Host.
///
/// Axes are ordered as all codomain legs followed by all domain legs. Data is
/// column-major (axis 0 varies fastest). Within each leg, entries follow
/// TensorKit's sector order (the provider's `sector_order_key`; U(1):
/// `0, 1, -1, 2, ...`), then degeneracy, then carrier-basis index, with the
/// carrier index varying fastest. A dual space `V'` is listed in `V`'s sector
/// order, the dual basis vector `e^i` at the position of `e_i`.
///
/// This is TensorKit's `convert(Array, t)` layout, so the side of a leg does
/// not change its index order; see [`TensorMap::to_physical_dense`].
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicalDense<D> {
    /// Physical dimension of each axis, in codomain-then-domain order.
    pub shape: Vec<usize>,
    /// Column-major physical entries.
    pub data: Vec<D>,
}

pub(super) fn owned_repr<R, D, S>(body: TypedTensorBody<R, D, S>) -> TypedTensorRepr<R, D, S> {
    TypedTensorRepr::Owned(Arc::new(body))
}

/// A block-sparse symmetric tensor map that keeps its provider type.
///
/// `D` is the payload dtype ([`f64`] or [`num_complex::Complex64`]) and is
/// independent of the provider's real categorical coefficient scalar — the
/// same separation TensorKit makes between a tensor's `T` and its sector type.
///
/// `S` is the owned payload storage and defaults to [`Vec<D>`]. Runtime
/// placement is diagnostic metadata from [`TensorStorage::placement`], not an
/// operation-dispatch mechanism; the current arithmetic impls remain on the
/// default host storage.
///
/// Host readback exists only when the storage implements
/// [`HostReadableStorage`]:
///
/// ```compile_fail
/// use symtn::expert::{Placement, TensorStorage};
/// use symtn::typed::TensorMap;
///
/// struct DeviceStorage(usize);
/// impl TensorStorage<f64> for DeviceStorage {
///     fn len(&self) -> usize { self.0 }
///     fn placement(&self) -> Placement { Placement::Cuda(0) }
/// }
///
/// fn cannot_read<R>(tensor: &TensorMap<R, f64, DeviceStorage>) {
///     let _ = tensor.dense_data();
/// }
/// ```
///
/// Naming a storage type and cloning its handle do not require the storage
/// itself to implement [`Clone`]:
///
/// ```
/// use symtn::expert::{Placement, TensorStorage};
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::TensorMap;
///
/// struct OpaqueStorage;
/// impl TensorStorage<f64> for OpaqueStorage {
///     fn len(&self) -> usize { 0 }
///     fn placement(&self) -> Placement { Placement::Cuda(0) }
/// }
///
/// fn clone_handle(tensor: &TensorMap<U1FusionRule, f64, OpaqueStorage>) {
///     let _: TensorMap<U1FusionRule, f64, OpaqueStorage> = tensor.clone();
/// }
/// ```
///
/// Cloning is cheap: the runtime handle and the shared body are both
/// reference-counted, and cloning does not require `S: Clone`.
pub struct TensorMap<R, D, S = Vec<D>> {
    pub(super) runtime: Runtime,
    pub(super) repr: TypedTensorRepr<R, D, S>,
}

impl<R, D, S> AsRef<Self> for TensorMap<R, D, S> {
    fn as_ref(&self) -> &Self {
        self
    }
}

/// Tensor authority parked without retaining its [`Runtime`].
///
/// This is an internal ownership seam for the Runtime-owned network workspace
/// pool. It retains only a validated provider-neutral layout and an owned dense
/// payload; detach/attach never copies or materializes tensor data, and attach
/// binds the layout to the current execution authority's exact provider.
///
/// `S` is the parked payload storage. Host storage parks a `Vec<D>` allocation;
/// a device storage parks its resident device buffer. Parking is a *Runtime and
/// provider* ownership operation only — it never moves or frees the payload.
#[doc(hidden)]
pub struct RuntimeDetachedTensorMap<D, S = Vec<D>> {
    runtime: RuntimeIdentity,
    layout: ValidatedDynamicFusionLayout,
    data: Arc<TypedData<D, S>>,
}

mod network_payload_sealed {
    pub trait Sealed {}
}

/// Payload storage whose retained-byte cost the network workspace budget can
/// charge.
///
/// Host storage reports the bytes its allocation *holds* ([`Vec::capacity`]),
/// because a pooled Host destination keeps its spare capacity and reuses it.
/// Device storage reports `len` bytes: a device buffer is allocated at its
/// exact length and has no separate capacity.
#[doc(hidden)]
pub trait NetworkPayloadStorage<D>:
    TensorStorage<D> + network_payload_sealed::Sealed + 'static
{
    /// Bytes this allocation retains while it is held for reuse.
    fn network_retained_bytes(&self) -> usize;
}

impl<D: 'static> network_payload_sealed::Sealed for Vec<D> {}

impl<D: 'static> NetworkPayloadStorage<D> for Vec<D> {
    fn network_retained_bytes(&self) -> usize {
        self.capacity().saturating_mul(std::mem::size_of::<D>())
    }
}

#[cfg(feature = "cuda")]
impl<D: symtn_dense::CudaScalar> network_payload_sealed::Sealed for CudaStorage<D> {}

#[cfg(feature = "cuda")]
impl<D: CudaPayload> NetworkPayloadStorage<D> for CudaStorage<D> {
    fn network_retained_bytes(&self) -> usize {
        TensorStorage::len(self).saturating_mul(std::mem::size_of::<D>())
    }
}

/// Storage route used by typed network replay admission.
#[doc(hidden)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum NetworkReuseClass {
    OwnedDense,
    Compact,
    LazyAdjoint,
}

// Why hand-written: the derive would demand `R: Clone`, `D: Clone`, and
// `S: Clone`; none is needed because the representation sits behind `Arc`.
impl<R, D, S> Clone for TensorMap<R, D, S> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: match &self.repr {
                TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(Arc::clone(body)),
                TypedTensorRepr::Adjoint(view) => TypedTensorRepr::Adjoint(Arc::clone(view)),
            },
        }
    }
}

impl<R, D, S> TensorMap<R, D, S> {
    /// Removes Runtime and provider ownership from an ordinary dense
    /// destination while preserving its validated layout and payload allocation.
    ///
    /// This is storage-generic: a device destination parks its resident buffer
    /// exactly as a Host destination parks its `Vec`, because detaching only
    /// drops the [`Runtime`] and provider handles.
    pub(crate) fn detach_runtime(self) -> Option<RuntimeDetachedTensorMap<D, S>> {
        let TensorMap { runtime, repr } = self;
        let TypedTensorRepr::Owned(body) = repr else {
            return None;
        };
        let body = Arc::try_unwrap(body).ok()?;
        if Arc::strong_count(&body.data) != 1 || !matches!(body.data.as_ref(), TypedData::Dense(_))
        {
            return None;
        }
        Some(RuntimeDetachedTensorMap {
            runtime: runtime.identity(),
            layout: body.space.validated_layout(),
            data: body.data,
        })
    }
}

impl<D, S> RuntimeDetachedTensorMap<D, S> {
    /// Dense allocation bytes and conservative dependent-layout bytes retained
    /// by this parked destination.
    ///
    /// Shared layout descendants are charged per parked destination. This can
    /// over-count, but it keeps the Runtime budget a true ceiling even when the
    /// workspace is the last owner of an Arc-backed layout allocation.
    #[doc(hidden)]
    pub fn retained_dense_capacity_bytes(&self) -> usize
    where
        S: NetworkPayloadStorage<D>,
    {
        let payload_capacity = match self.data.as_ref() {
            TypedData::Dense(data) => data.network_retained_bytes(),
            TypedData::Diagonal(_) => {
                debug_assert!(false, "runtime-detached destinations are always dense");
                0
            }
        };
        payload_capacity
            .saturating_add(std::mem::size_of::<TypedData<D, S>>())
            .saturating_add(2 * std::mem::size_of::<usize>())
            .saturating_add(self.layout.charged_retained_bytes())
    }

    /// Whether this parked tensor belongs to `runtime`.
    #[doc(hidden)]
    pub fn matches_runtime(&self, runtime: &Runtime) -> bool {
        self.runtime.matches(runtime)
    }

    /// Validates Runtime identity and layout rebinding against the current
    /// authority without consuming this parked destination.
    #[doc(hidden)]
    pub fn can_attach<R>(
        &self,
        runtime: &Runtime,
        authority: &TensorMap<R, D, S>,
    ) -> Result<(), Error> {
        if !self.matches_runtime(runtime) {
            return Err(Error::RuntimeMismatch);
        }
        authority.logical_space().rebind_validated(&self.layout)?;
        Ok(())
    }

    /// Rebinds this provider-neutral destination to the current authority's
    /// exact provider allocation after [`Self::can_attach`] succeeds.
    #[doc(hidden)]
    pub fn attach_runtime<R>(
        self,
        runtime: &Runtime,
        authority: &TensorMap<R, D, S>,
    ) -> Result<TensorMap<R, D, S>, Error> {
        if !self.matches_runtime(runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let space = authority.logical_space().rebind_validated(&self.layout)?;
        Ok(TensorMap {
            runtime: runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(space, self.data)),
        })
    }
}

impl<R, D, S> core::fmt::Debug for TensorMap<R, D, S>
where
    S: TensorStorage<D>,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let stored = &self.storage_body().data;
        formatter
            .debug_struct("TensorMap")
            // Storage-shaped, deliberately: a compact spectrum payload reports
            // the values it stores, and forcing its dense materialization for a
            // `{:?}` would make a diagnostic the most expensive call on the type.
            .field(
                "elements",
                &match stored.as_ref() {
                    TypedData::Dense(data) => data.len(),
                    TypedData::Diagonal(spectrum) => {
                        spectrum.iter().map(|entry| entry.values.len()).sum()
                    }
                },
            )
            .finish_non_exhaustive()
    }
}

impl<R, D, S> TensorMap<R, D, S> {
    /// Identity and retained bytes of the dense payload allocation owned by
    /// this tensor. Network metering uses the identity to avoid charging Arc
    /// aliases twice.
    pub(crate) fn network_owned_payload(&self) -> Option<(usize, usize)>
    where
        S: NetworkPayloadStorage<D>,
    {
        let body = self.storage_body();
        let TypedData::Dense(data) = body.data.as_ref() else {
            return None;
        };
        Some((
            Arc::as_ptr(&body.data) as usize,
            data.network_retained_bytes(),
        ))
    }

    /// Classifies the representation produced after an optional adjoint.
    pub(crate) fn network_reuse_class(&self, adjoint: bool) -> NetworkReuseClass {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Diagonal(_) => NetworkReuseClass::Compact,
                TypedData::Dense(_) if adjoint => NetworkReuseClass::LazyAdjoint,
                TypedData::Dense(_) => NetworkReuseClass::OwnedDense,
            },
            TypedTensorRepr::Adjoint(_) if adjoint => NetworkReuseClass::OwnedDense,
            TypedTensorRepr::Adjoint(_) => NetworkReuseClass::LazyAdjoint,
        }
    }

    /// Matches the metadata produced after an optional adjoint without allocation.
    pub(crate) fn network_input_metadata_matches(
        &self,
        adjoint: bool,
        expected_legs: &[SectorLeg],
        expected_class: NetworkReuseClass,
    ) -> bool
    where
        R: TypedSectorAdmission,
    {
        if self.network_reuse_class(adjoint) != expected_class || self.rank() != expected_legs.len()
        {
            return false;
        }
        (0..self.rank()).all(|axis| {
            let source_axis = if adjoint {
                (axis + self.codomain_rank()) % self.rank()
            } else {
                axis
            };
            let Some(actual) = self.network_source_leg(source_axis) else {
                return false;
            };
            actual == &expected_legs[axis]
        })
    }

    /// The logical leg of source axis `axis` as stored (a domain leg is not
    /// dualised), borrowed so network admission compares legs without
    /// allocating.
    pub(crate) fn network_source_leg(&self, axis: usize) -> Option<&SectorLeg> {
        let homspace = self.logical_space().space().homspace();
        if axis < self.codomain_rank() {
            homspace.codomain().legs().get(axis)
        } else {
            homspace.domain().legs().get(axis - self.codomain_rank())
        }
    }

    pub(super) fn storage_body(&self) -> &Arc<TypedTensorBody<R, D, S>> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => body,
            TypedTensorRepr::Adjoint(view) => &view.parent,
        }
    }

    pub(super) fn logical_space(&self) -> &BoundDynamicFusionMapSpace<R> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => &body.space,
            TypedTensorRepr::Adjoint(view) => &view.logical_space,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_bound_space(&self) -> &BoundDynamicFusionMapSpace<R> {
        self.logical_space()
    }

    /// Whether the tensor owns a dense payload (neither a lazy adjoint nor a
    /// compact diagonal): the representations `axpby_into` writes into.
    #[cfg(feature = "krylov")]
    pub(crate) fn is_owned_dense(&self) -> bool {
        self.owned_body()
            .is_some_and(|body| matches!(body.data.as_ref(), TypedData::Dense(_)))
    }

    pub(super) fn owned_body(&self) -> Option<&Arc<TypedTensorBody<R, D, S>>> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => Some(body),
            TypedTensorRepr::Adjoint(_) => None,
        }
    }

    /// The provider allocation that owns this tensor's categorical layout.
    #[inline]
    pub fn provider(&self) -> &R {
        self.logical_space().provider()
    }

    /// Number of codomain legs.
    #[inline]
    pub fn codomain_rank(&self) -> usize {
        self.logical_space().space().homspace().codomain().len()
    }

    /// Number of domain legs.
    #[inline]
    pub fn domain_rank(&self) -> usize {
        self.logical_space().space().homspace().domain().len()
    }

    /// Total number of legs.
    #[inline]
    pub fn rank(&self) -> usize {
        self.codomain_rank() + self.domain_rank()
    }

    /// Exact current split and axis order, checked without constructing a
    /// transform operation (whose inline axis storage can spill at high rank).
    ///
    /// Storage-generic so the host and device transform facades cannot drift
    /// on what "identity" means, which is what each of them short-circuits on.
    #[inline]
    pub(super) fn axes_are_identity(&self, codomain_axes: &[usize], domain_axes: &[usize]) -> bool {
        let codomain_rank = self.codomain_rank();
        codomain_axes.iter().copied().eq(0..codomain_rank)
            && domain_axes.iter().copied().eq(codomain_rank..self.rank())
    }

    /// Layout metadata of one fusion-tree subblock, by its index in
    /// [`Self::subblocks`] order. For the coupled-sector matrix use
    /// [`Self::block`].
    ///
    /// Metadata only: reading an adjoint subblock does not materialize its data.
    pub fn subblock(&self, index: usize) -> Result<BlockRef<'_>, Error> {
        self.logical_space()
            .space()
            .structure()
            .block(index)
            .map_err(Error::from)
    }

    /// Runtime bound to this tensor map.
    #[inline]
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// The codomain legs, in axis order.
    ///
    /// Allocates: each call builds a fresh `Vec` and clones every leg's
    /// sector table (the provider travels by `Arc` bump). Hold the result
    /// rather than re-calling in a loop.
    pub fn codomain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.logical_space().space().homspace().codomain())
    }

    /// The domain legs, in axis order.
    ///
    /// Allocates per call, exactly as [`Self::codomain`].
    pub fn domain(&self) -> Vec<GradedSpace<R>> {
        self.legs(self.logical_space().space().homspace().domain())
    }

    fn legs(&self, product: &FusionProductSpace) -> Vec<GradedSpace<R>> {
        product
            .legs()
            .iter()
            .map(|leg| GradedSpace {
                provider: Arc::clone(self.logical_space().provider_arc()),
                leg: leg.clone(),
            })
            .collect()
    }

    /// Number of fusion-tree subblocks: TensorKit `length(fusiontrees(t))`,
    /// not the number of coupled sectors.
    #[inline]
    pub fn subblock_count(&self) -> usize {
        self.logical_space().space().structure().block_count()
    }
}

impl<R, D, S> TensorMap<R, D, S> {
    /// An operation's fresh dense result on `space`, bound to this tensor's
    /// runtime.
    pub(super) fn published(&self, space: BoundDynamicFusionMapSpace<R>, data: S) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        }
    }
}

impl<R, D> TensorMap<R, D> {
    pub(super) fn with_data(&self, data: Vec<D>) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), data)),
        }
    }
}

// Generic over storage: the one `adjoint` body applies the compact rule
// (#1452) on every storage.
impl<R, D, S> TensorMap<R, D, S> {
    pub(super) fn with_spectrum(
        &self,
        spectrum: Vec<symtn_matrixalgebra::SectorSpectrum<D>>,
    ) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(
                self.logical_space().clone(),
                spectrum,
            )),
        }
    }

    pub(super) fn with_spectrum_on(
        &self,
        space: BoundDynamicFusionMapSpace<R>,
        spectrum: Vec<symtn_matrixalgebra::SectorSpectrum<D>>,
    ) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::diagonal(space, spectrum)),
        }
    }

    pub(super) fn spectrum(&self) -> Option<&[symtn_matrixalgebra::SectorSpectrum<D>]> {
        match self.owned_body()?.data.as_ref() {
            TypedData::Diagonal(spectrum) => Some(spectrum),
            TypedData::Dense(_) => None,
        }
    }

    /// TensorKit `adjoint(::DiagonalTensorMap)`: `d` itself for a real
    /// payload, else the conjugated diagonal. A bond space is its own adjoint
    /// (`codomain == domain`), so this is `O(Σ_c k_c)` with no dense buffer
    /// and no bend, and a real payload shares the body without copying.
    /// [`None`] unless the tensor owns a compact diagonal.
    pub(super) fn compact_adjoint(&self) -> Option<Self>
    where
        D: TensorScalar,
    {
        let spectrum = self.spectrum()?;
        if D::CONJUGATION_IS_IDENTITY {
            return Some(self.clone());
        }
        Some(
            self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| symtn_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| FactorScalar::adjoint(value))
                            .collect(),
                    })
                    .collect(),
            ),
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// TensorKit `absorb(tdst, tsrc)` (which copies and delegates to
    /// `absorb!`): copies the common per-axis prefix of every shared
    /// fusion-tree block of `source` into a deep copy of `self` (TK takes
    /// the `min` of the two block shapes per axis). Blocks whose key the source
    /// does not carry are untouched, so the caller owns the initialization of
    /// the non-shared region — TK documents the same contract.
    ///
    /// The result keeps `self`'s spaces and dtype. Equal `D` is required by
    /// the signature; widen with [`Self::convert`] first. A compact `source`
    /// is read directly, including its structural zero entries. A compact
    /// receiver is expanded into the returned dense output.
    ///
    /// # Complexity
    ///
    /// One output allocation for owned dense inputs plus `O(min-prefix)`
    /// overwrites per shared block. A lazy input currently adds one
    /// operation-local logical payload, released with the operation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on unequal codomain/domain ranks (TK throws
    /// its `DimensionError` for the same), [`Error::RuleMismatch`] when the
    /// already-admitted layouts carry differing rule-identity stamps,
    /// [`Error::RuntimeMismatch`] on differing runtimes, and
    /// [`Error::InvalidArgument`] when corresponding legs differ in duality.
    ///
    /// An adjoint view `source` (`t.adjoint_view()`) returns
    /// [`Error::Unsupported`]: this operation would copy it. Pass
    /// `&t.adjoint()?.materialize()?` instead.
    pub fn absorb<'a>(&self, source: impl Into<TensorRef<'a, R, D>>) -> Result<Self, Error> {
        let source = source.into().operand()?;
        let source = &*source;
        source.refuse_borrowed_view("absorb")?;
        let destination_space = self.logical_space().space();
        let source_space = source.logical_space().space();
        if destination_space.nout() != source_space.nout()
            || destination_space.nin() != source_space.nin()
        {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::absorb requires equal codomain/domain ranks, got {}|{} and {}|{}",
                destination_space.nout(),
                destination_space.nin(),
                source_space.nout(),
                source_space.nin()
            )));
        }
        if destination_space.admission().rule_identity() != source_space.admission().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        if !self.runtime.same_runtime(&source.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        for (destination_leg, source_leg) in destination_space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(destination_space.homspace().domain().legs())
            .zip(
                source_space
                    .homspace()
                    .codomain()
                    .legs()
                    .iter()
                    .chain(source_space.homspace().domain().legs()),
            )
        {
            if destination_leg.is_dual() != source_leg.is_dual() {
                return Err(Error::InvalidArgument(
                    "TensorMap::absorb requires corresponding legs to have equal duality"
                        .to_string(),
                ));
            }
        }
        // A lazy input still takes an operation-local dense payload.
        let destination = self.materialized_tensor_uncached()?;
        let source = source.materialized_tensor_uncached()?;
        let destination_body = destination
            .owned_body()
            .expect("uncached materialization is owned");
        let source_body = source
            .owned_body()
            .expect("uncached materialization is owned");
        let mut output = match destination_body.data.as_ref() {
            TypedData::Dense(data) => data.clone(),
            TypedData::Diagonal(spectrum) => symtn_matrixalgebra::seam::diagonal_bond_data(
                destination_space,
                spectrum,
                &|value| value,
            )?,
        };
        let source_required_len = source_space.structure().required_len()?;
        if destination_space.structure().required_len()? != output.len()
            || matches!(source_body.data.as_ref(), TypedData::Dense(data)
                if source_required_len != data.len())
        {
            return Err(internal_layout_error(
                "absorb block layout does not cover scalar storage",
            ));
        }
        match source_body.data.as_ref() {
            TypedData::Dense(data) => absorb_mapped(
                destination_space.structure(),
                &mut output,
                source_space.structure(),
                data,
                Ok,
            )?,
            TypedData::Diagonal(spectrum) => absorb_compact_source(
                destination_space.structure(),
                &mut output,
                source_space.structure(),
                spectrum,
            )?,
        }
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), output)),
        })
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    /// Quantum-dimension-weighted total dimension of every leg, in flat order
    /// (codomain legs first, then domain legs) — TensorKit's `dim(space(t,
    /// i))` per leg.
    /// Contraction planners use it as a size/FLOP proxy.
    ///
    /// The rounding formula is
    /// `Σ_sector round(degeneracy * dim(sector))` per leg, with `dim` read
    /// through the mode's `RigidCoefficientAlgebra::dim` (crate helper
    /// `multiplicity_free_dim`), so there is deliberately no group-specific
    /// branch.
    ///
    /// # Complexity
    ///
    /// `O(Σ_leg sectors)`; allocates the returned `Vec<usize>` only, never a
    /// payload.
    ///
    /// # Errors
    ///
    /// None today; the `Result` leaves room for future fallible dimension
    /// providers without changing the method shape.
    pub fn leg_dims(&self) -> Result<Vec<usize>, Error> {
        let hom = self.logical_space().space().homspace();
        let provider = self.logical_space().provider();
        hom.codomain()
            .legs()
            .iter()
            .chain(hom.domain().legs())
            .map(|leg| Self::weighted_leg_dim(provider, leg))
            .collect()
    }

    /// Quantum dimensions are generally irrational (SU(2) `sqrt` products,
    /// anyonic golden ratios), so the per-sector weight is computed in `f64`
    /// and rounded once.
    fn weighted_leg_dim(provider: &R, leg: &SectorLeg) -> Result<usize, Error> {
        leg.sectors()
            .iter()
            .zip(leg.degeneracies())
            .map(|(&sector, &degeneracy)| {
                Ok((degeneracy as f64 * multiplicity_free_dim(provider, sector)?).round() as usize)
            })
            .sum()
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    S: TensorStorage<D>,
{
    /// Reports where the canonical payload lives.
    ///
    /// This is diagnostic metadata only. Arithmetic never dispatches on
    /// placement, and transfers are always explicit and type-changing.
    pub fn placement(&self) -> Placement {
        match self.storage_body().data.as_ref() {
            TypedData::Dense(storage) => storage.placement(),
            TypedData::Diagonal(_) => Placement::Host,
        }
    }
}

// Explicit payload precision conversions (#1323).
//
// TensorKit converts with `complex(t)`, `convert(TensorMap{T,…}, t)` and
// `similar(t, T)` + `copy!`, and promotes mixed scalar types implicitly
// (`promote_rule` on `TensorMap`), so a `Float64 -> Float32` `convert` rounds
// silently. symtn keeps mixed payload dtypes a compile error and converts only
// through the explicit `convert::<To>()`, whose sealed pair list marks the two
// lossy narrowings. There are deliberately no `From` impls.
//
// Only the host `Vec<D>` storage has `convert`; a device tensor is
// downloaded, converted and uploaded explicitly.
impl<R, D> TensorMap<R, D>
where
    D: TensorScalar + FactorScalar,
{
    /// Converts the payload elementwise into an owned tensor.
    ///
    /// * Owned dense or compact diagonal: one pass into one new payload of
    ///   the same form.
    /// * Lazy adjoint (always of a dense parent; see
    ///   `TypedAdjointView::new`): materialized in the source dtype,
    ///   then converted — an operation-local source-sized buffer plus the
    ///   output. Why not a lazy view over a converted parent: several
    ///   operations (`diagview`, the
    ///   `permute_into` source, the device path after `to_cuda`) reject
    ///   lazy adjoints, so a lazy
    ///   result would accept less than the owned one `to_c64` always
    ///   returned. Why not one fused pass: `materialize_adjoint_data_dyn` has
    ///   no mapped variant, and adding one would duplicate its block walk.
    fn convert_payload<E>(&self, convert: impl Fn(D) -> E) -> TensorMap<R, E> {
        let body = match &self.repr {
            TypedTensorRepr::Owned(body) => body.convert_payload(convert),
            TypedTensorRepr::Adjoint(_) => self
                .materialized_tensor_uncached()
                .expect("a pre-admitted typed adjoint must materialize")
                .owned_body()
                .expect("uncached materialization is owned")
                .convert_payload(convert),
        };
        TensorMap {
            runtime: self.runtime.clone(),
            repr: TypedTensorRepr::Owned(body),
        }
    }
}

impl<R, D: Copy> TypedTensorBody<R, D> {
    fn convert_payload<E>(&self, convert: impl Fn(D) -> E) -> Arc<TypedTensorBody<R, E>> {
        let data = match self.data.as_ref() {
            TypedData::Dense(values) => {
                TypedData::Dense(values.iter().map(|&value| convert(value)).collect())
            }
            TypedData::Diagonal(spectrum) => {
                TypedData::Diagonal(map_spectrum_dtype(spectrum, convert))
            }
        };
        Arc::new(TypedTensorBody::new(self.space.clone(), data))
    }
}

/// An elementwise payload dtype conversion accepted by [`TensorMap::convert`].
///
/// Implemented for exactly these pairs:
///
/// | from | to | loss |
/// |---|---|---|
/// | `f32` | `f64`, `Complex32`, `Complex64` | exact |
/// | `f64` | `Complex64` | exact |
/// | `Complex32` | `Complex64` | exact |
/// | `f64` | `f32` | **lossy** |
/// | `Complex64` | `Complex32` | **lossy**, componentwise |
///
/// A lossy entry is `value as f32`: IEEE 754 round to nearest, ties to even.
/// A finite value whose rounding exceeds `f32::MAX` in magnitude becomes
/// `±inf`; a tiny value rounds to the nearest `f32` subnormal or `±0` (no
/// flush to zero); `±0` and `±inf` keep their sign; NaN stays NaN (Rust does
/// not promise its payload bits). An exact entry represents every source
/// value, including subnormals, `±0`, `±inf` and NaN; a real source gets a
/// zero imaginary part. There is no complex -> real pair (TensorKit's
/// `real(t)` discards data; it is not a precision change).
///
/// Sealed: the pairs above are the whole contract.
pub trait PayloadConversion<To>: Copy + typed_payload_conversion_private::Sealed<To> {
    #[doc(hidden)]
    fn convert_value(self) -> To;
}

mod typed_payload_conversion_private {
    pub trait Sealed<To> {}
}

macro_rules! payload_conversion {
    ($($from:ty => $to:ty, |$value:ident| $body:expr;)*) => {$(
        impl typed_payload_conversion_private::Sealed<$to> for $from {}
        impl PayloadConversion<$to> for $from {
            fn convert_value(self) -> $to {
                let $value = self;
                $body
            }
        }
    )*};
}

payload_conversion! {
    f32 => f64, |value| f64::from(value);
    f32 => Complex32, |value| Complex32::new(value, 0.0);
    f32 => Complex64, |value| Complex64::new(f64::from(value), 0.0);
    f64 => Complex64, |value| Complex64::new(value, 0.0);
    Complex32 => Complex64, |value| Complex64::new(f64::from(value.re), f64::from(value.im));
    f64 => f32, |value| value as f32;
    Complex64 => Complex32, |value| Complex32::new(value.re as f32, value.im as f32);
}

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar + FactorScalar,
{
    /// Converts the payload dtype elementwise (TensorKit
    /// `convert(TensorMap{T}, t)` and `complex(t)`). The accepted pairs, and
    /// which of them are lossy, are listed on [`PayloadConversion`].
    ///
    /// Spaces, block structure, gauge and the provider are unchanged, and
    /// the result is always owned. Dense and compact diagonal storage keep
    /// their form: one pass and one payload-sized allocation (a compact
    /// diagonal allocates the per-sector spectra its form holds). A lazy
    /// adjoint of a compact diagonal becomes the owned compact diagonal its
    /// multiplicity-free `adjoint` would be; a dense lazy adjoint is
    /// materialized in the source dtype and then converted, which is two
    /// payload-sized allocations (the operation-local materialization and the
    /// output).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use num_complex::Complex64;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    /// )?;
    /// let single: TensorMap<_, f32> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 7)?;
    /// let double = single.convert::<f64>();
    /// assert_eq!(double.convert::<f32>().dense_data()?, single.dense_data()?);
    ///
    /// let widened = double.convert::<Complex64>();
    /// // The real part round-trips exactly; the imaginary part is zero.
    /// assert_eq!(widened.re().dense_data()?, double.dense_data()?);
    /// assert_eq!(widened.im().norm(2.0)?, 0.0);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// Mixing payload dtypes stays a compile error; nothing converts
    /// implicitly:
    ///
    /// ```compile_fail
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn mixed(a: &TensorMap<U1FusionRule, f32>, b: &TensorMap<U1FusionRule, f64>) {
    ///     let _ = a.axpby(1.0, b, 1.0);
    /// }
    /// ```
    ///
    /// and there is no `From`/`Into` between payload dtypes:
    ///
    /// ```compile_fail
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn into(a: TensorMap<U1FusionRule, f32>) -> TensorMap<U1FusionRule, f64> {
    ///     a.into()
    /// }
    /// ```
    ///
    /// nor a complex -> real conversion:
    ///
    /// ```compile_fail
    /// use num_complex::Complex64;
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn real(a: &TensorMap<U1FusionRule, Complex64>) -> TensorMap<U1FusionRule, f64> {
    ///     a.convert::<f64>()
    /// }
    /// ```
    ///
    /// The conversion is host-only: a tensor on another storage (such as
    /// `CudaStorage`) is converted after an explicit download.
    ///
    /// ```compile_fail
    /// use symtn::expert::{Placement, TensorStorage};
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// struct DeviceStorage(usize);
    /// impl TensorStorage<f32> for DeviceStorage {
    ///     fn len(&self) -> usize { self.0 }
    ///     fn placement(&self) -> Placement { Placement::Cuda(0) }
    /// }
    ///
    /// fn device(tensor: &TensorMap<U1FusionRule, f32, DeviceStorage>) {
    ///     let _ = tensor.convert::<f64>();
    /// }
    /// ```
    pub fn convert<To>(&self) -> TensorMap<R, To>
    where
        D: PayloadConversion<To>,
    {
        self.convert_payload(PayloadConversion::convert_value)
    }
}
