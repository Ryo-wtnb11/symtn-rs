use super::*;

/// The error every fallible [`TensorMap`] / [`GradedSpace`] operation returns
/// for fusion rule `R`: [`Error`] for a multiplicity-free rule, and
/// [`GenericTensorError`]`<R::Error>` for a checked Generic rule.
///
/// The error type depends on the rule's admission mode, so generic code over
/// `R` names it through this alias rather than through the mode-dispatch
/// traits. Both types convert from [`Error`].
pub type FacadeError<R> =
    <<R as TypedSectorAdmission>::Mode as TypedTensorModeDispatch<R>>::FacadeError;

/// Tensor-side layout admission selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorModeDispatch<R>: typed_admission_private::Sealed
where
    R: TypedSectorAdmission,
{
    /// Error returned by the ordinary typed facade.
    type FacadeError: std::error::Error + From<Error> + From<symtn_tensors::OperationError>;

    /// Preserves a provider-side admission error.
    fn map_provider_error(error: R::Error) -> Self::FacadeError;

    /// The provider's fusion style, which selects TensorKit's
    /// `UniqueFusion` whole-buffer reductions.
    #[doc(hidden)]
    fn fusion_style(provider: &R) -> symtn_core::FusionStyleKind;

    /// The provider's braiding style, which admits contraction and selects
    /// the twist/flip early returns.
    #[doc(hidden)]
    fn braiding_style(provider: &R) -> symtn_core::BraidingStyleKind;

    /// Lowers a facade error to [`Error`] for the provider-neutral
    /// [`TensorRef`] adjoint constructor.
    #[doc(hidden)]
    fn facade_error_into_error(error: Self::FacadeError) -> Error;

    /// TensorKit `blockdim(P, c)`: the reduced dimension of coupled sector
    /// `coupled` in the product space `product`, zero when it does not occur.
    #[doc(hidden)]
    fn coupled_block_dimension(
        provider: &R,
        product: &FusionProductSpace,
        coupled: SectorId,
    ) -> Result<usize, Self::FacadeError>;

    /// TensorKit `fuse` of two sector contents, `N`-symbol weighted and in
    /// sector-id order.
    #[doc(hidden)]
    fn fuse_sector_content(
        provider: &R,
        left: &[(SectorId, usize)],
        right: &[(SectorId, usize)],
    ) -> Result<Vec<(SectorId, usize)>, Self::FacadeError>;
}

/// Tensor-side root construction selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorRootDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    /// Builds one fully admitted root while retaining `provider`.
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;

    /// Derives the final HomSpace before the existing root admission, in one epoch.
    fn build_root_with(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, Self::FacadeError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

/// Tensor-side construction gate for payload/provider coefficient pairs.
#[doc(hidden)]
pub trait TypedTensorConstructionDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Builds one fully admitted root while retaining `provider`.
    fn build_construction_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

/// Tensor-side tree-transform execution selected by a provider-owned mode.
///
/// ```
/// use symtn::sector::{CheckedGenericAdmissionMode, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use symtn::typed::TensorMap;
///
/// fn checked_transpose<R>(tensor: &TensorMap<R, f64>)
/// where
///     R: TypedSectorAdmission<
///             Error = <R as symtn::sector::CheckedGenericFusion>::Error,
///             Mode = CheckedGenericAdmissionMode,
///         > + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.transpose(&[1], &[0]);
/// }
/// ```
#[doc(hidden)]
pub trait TypedTensorTransformDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// `operation` (a rank-(1,1) leg swap or braid) applied to the compact
    /// `spectrum` of `tensor` in one staging: the committed destination and
    /// either the spectrum carried through the resolved single-term square
    /// bijection, or (when that proof or `require_representable` fails) the
    /// dense replay of the densified spectrum with the same structure.
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, D>,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, D>, Self::FacadeError>;

    /// The lazy adjoint of the transformed parent for a lazy-adjoint input,
    /// else `None`. The method is where a mode discharges
    /// [`TypedAdjointSpace`]; the default `None` is for a mode that cannot
    /// hold a lazy adjoint.
    fn try_lazy_adjoint_transform<S: TransformExec<R, D>>(
        _tensor: &TensorMap<R, D, S>,
        _operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D, S>>, Self::FacadeError> {
        Ok(None)
    }

    /// Transforms the owned dense payload of `tensor` (a compact diagonal
    /// densified) into a fresh destination payload of its storage. A lazy
    /// adjoint never reaches it: `try_lazy_adjoint_transform` takes it first.
    fn transform<S: TransformExec<R, D>>(
        tensor: &TensorMap<R, D, S>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError>;

    /// `destination = alpha · operation(source) + beta · destination` for a
    /// `source` of any representation (a lazy adjoint read through its
    /// parent's storage, a compact diagonal through its bond structure) and a
    /// `destination` that passed [`destination_precheck`]: one lease,
    /// [`Self::transform`]'s staging, the staged space against the
    /// destination's, the destination's length and ownership, the storage's
    /// write, then the same commit. A mode with an exact-layout memo offers it
    /// the borrowed request of an owned dense source through
    /// `exact_layout_probe` before `operation` builds the owned one.
    fn transform_into<S: TransformExec<R, D>>(
        source: &TensorMap<R, D, S>,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
        operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
        exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
    ) -> Result<(), Self::FacadeError>;
}

/// The committed destination of a compact rank-(1,1) transform and its
/// payload.
///
/// Internal and unstable despite being public for the dispatch traits.
#[doc(hidden)]
pub struct BondTransform<R, D> {
    pub destination: BoundDynamicFusionMapSpace<R>,
    pub output: BondOutput<D>,
}

/// A compact rank-(1,1) transform's payload: the carried spectrum, or the
/// dense replay of the densified spectrum when the bijection proof declined.
///
/// Internal and unstable despite being public for the dispatch traits.
#[doc(hidden)]
pub enum BondOutput<D> {
    Spectrum(Vec<symtn_matrixalgebra::SectorSpectrum<D>>),
    Dense(Vec<D>),
}

pub(super) trait MultiplicityFreeTransformExecution<R>: TensorScalar
where
    R: TypedSectorAdmission,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, Self>,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<Self>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, Self>, Error>;

    fn try_lazy_adjoint<S: TransformExec<R, Self>>(
        tensor: &TensorMap<R, Self, S>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, Self, S>>, Error>;

    fn transform<S: TransformExec<R, Self>>(
        tensor: &TensorMap<R, Self, S>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error>;

    fn transform_into<S: TransformExec<R, Self>>(
        source: &TensorMap<R, Self, S>,
        destination: &mut TensorMap<R, Self, S>,
        alpha: Self,
        beta: Self,
        operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
        exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
    ) -> Result<(), Error>;
}

pub(super) trait MultiplicityFreeContractExecution<R: TypedSectorAdmission>:
    TensorScalar
{
    fn contract<S: ContractExec<R, Self>>(
        lhs: &TensorMap<R, Self, S>,
        rhs: &TensorMap<R, Self, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error>;

    fn contract_into<S: ContractExec<R, Self>>(
        lhs: &TensorMap<R, Self, S>,
        rhs: &TensorMap<R, Self, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
        destination: &mut TensorMap<R, Self, S>,
        alpha: Self,
        beta: Self,
    ) -> Result<(), Error>;

    fn compose<S: ContractExec<R, Self>>(
        lhs: &TensorMap<R, Self, S>,
        rhs: &TensorMap<R, Self, S>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error>;
}

/// A twist or Z-isomorphism coefficient (θ, χ) acting on payload `D`: the
/// provider's own categorical scalar, so a complex-symbol rule's θ is never
/// narrowed to `f64`.
///
/// Why a bound on the coefficient rather than `D: RecouplingCoefficientAction<C>`
/// at each call site: bounds on an associated type are implied wherever the
/// dispatch trait is, so generic callers of `twist`/`flip` keep naming only
/// `R::Mode: TypedTensorTwistDispatch<R, D>`.
#[doc(hidden)]
pub trait TwistCoefficient<D>: CategoricalScalar + Copy + PartialEq {
    /// This coefficient as a payload scalar.
    fn as_data(self) -> D;
}

impl<C, D> TwistCoefficient<D> for C
where
    C: CategoricalScalar + Copy + PartialEq,
    D: RecouplingCoefficientAction<C>,
{
    #[inline]
    fn as_data(self) -> D {
        D::coefficient_as_data(self)
    }
}

/// Ribbon-twist execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorTwistDispatch<R, D>: TypedSpaceModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// The coefficient type of θ: the provider's categorical scalar
    /// (TensorKit `braidingscalartype`).
    type Coeff: TwistCoefficient<D>;

    /// θ of every sector that `legs` carry in `structure`'s fusion-tree
    /// blocks, or `None` when every one of them is one. Every fallible
    /// provider query happens here, so scaling by the returned values never
    /// fails on the provider.
    fn twist_values<'a>(
        provider: &'a R,
        structure: &symtn_core::BlockStructure,
        codomain_rank: usize,
        legs: &[usize],
    ) -> Result<Option<impl Fn(SectorId) -> Self::Coeff + 'a>, Self::FacadeError>;
}

/// Z-isomorphism execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorFlipDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// The coefficient type of (χ, θ): the provider's categorical scalar
    /// (TensorKit `sectorscalartype`).
    type Coeff: TwistCoefficient<D>;

    /// The admitted space of `homspace`, bound to `space`'s provider.
    fn root(
        space: &BoundDynamicFusionMapSpace<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;

    /// (χ, θ) of every sector the flip `occurrences` carry in `structure`'s
    /// fusion-tree blocks, with every fallible provider query made here.
    #[expect(
        clippy::type_complexity,
        reason = "the staged (χ, θ) table is one opaque closure"
    )]
    fn pivotal_values<'a>(
        provider: &'a R,
        structure: &symtn_core::BlockStructure,
        codomain_rank: usize,
        occurrences: &[(usize, bool)],
    ) -> Result<impl Fn(SectorId) -> (Self::Coeff, Self::Coeff) + 'a, Self::FacadeError>;
}

/// Tensor-product execution selected by a provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorProductDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Executes the F-only product of two owned dense payloads.
    fn product(
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError>;
}

/// Contraction execution selected by a provider-owned mode.
///
/// Why a transform supertrait: the shared bodies call the mode-free compact
/// arms, whose reordered `t · D` / `D · t` result is laid out by one
/// [`TensorMap::permute`]-style tree transform.
#[doc(hidden)]
pub trait TypedTensorContractDispatch<R, D>: TypedTensorTransformDispatch<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Contracts two operands, a lazy adjoint read through its parent's
    /// payload, into a fresh destination payload of their storage: the
    /// staging, the storage's planner capabilities, its kernel, the commit.
    /// With `planar`, the planar contraction over those admitted orders
    /// ([`TensorMap::planar_contract`]) instead of the ordinary one.
    fn contract<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError>;

    /// `destination = alpha · contract(lhs, rhs) + beta · destination` on
    /// the dense route (a lazy adjoint read through its parent's payload, a
    /// compact operand densified) for a `destination` that passed
    /// [`destination_precheck`]: one lease, [`Self::contract`]'s staging, the
    /// staged space against the destination's, [`destination_storage_mut`],
    /// planning, the storage's write, then the same commit.
    fn contract_into<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
    ) -> Result<(), Self::FacadeError>;

    /// Composes two operands, a lazy adjoint read through its parent's
    /// payload, into a fresh destination payload of their storage.
    fn compose<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError>;
}

/// Partial categorical trace execution selected by provider-owned mode.
#[doc(hidden)]
pub trait TypedTensorTraceDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Stages the destination of `source`'s admitted trace (read through its
    /// storage body) and runs the one owned trace into it: the compile, the
    /// storage's [`TraceExec::trace_owned`] kernel — given `read`'s answer
    /// for a single-scalar destination (a compact spectrum) — then the
    /// commit.
    fn trace<S: TraceExec<R, D>>(
        source: &TensorMap<R, D, S>,
        read: impl FnOnce(&symtn_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        axes: symtn_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError>;

    /// `destination = alpha · trace + beta · destination` for a
    /// `destination` that passed [`destination_precheck`]:
    /// [`Self::trace`]'s staging, the staged space against the
    /// destination's (HomSpace before the pair duality, then the layout),
    /// [`destination_storage_mut`], the storage's
    /// [`TraceExec::trace_into`] kernel, then the same commit.
    #[allow(clippy::too_many_arguments)]
    fn trace_into<S: TraceExec<R, D>>(
        source: &TensorMap<R, D, S>,
        read: impl FnOnce(&symtn_tensors::TensorTraceFusionStructure<f64>) -> Option<D>,
        axes: symtn_tensors::TensorTraceAxisSpec<'_>,
        dst_nout: usize,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
    ) -> Result<(), Self::FacadeError>;
}

mod typed_admission_private {
    use super::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};

    pub trait Sealed {}

    impl Sealed for MultiplicityFreeAdmissionMode {}
    impl Sealed for CheckedGenericAdmissionMode {}
}

#[doc(hidden)]
pub trait TypedSpaceModeDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn vacuum(provider: &R) -> SectorId;
    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
    fn dim(
        provider: &R,
        sector: SectorId,
    ) -> Result<f64, <Self as TypedTensorModeDispatch<R>>::FacadeError>;
}

/// The bond-truncation decision, selected by a provider-owned mode.
///
/// Both arms call `symtn_matrixalgebra::seam::decide_bond_truncation`, the one
/// adapter over `select_truncation` that supplies the quantum-dimension weight
/// and TensorKit's sector order.
#[doc(hidden)]
pub trait TypedTruncationDispatch<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn decide_bond_truncation<V>(
        provider: &R,
        spectra: &[symtn_matrixalgebra::SectorSpectrum<V>],
        truncation: &Truncation,
    ) -> Result<symtn_matrixalgebra::TruncationDecision, Self::FacadeError>
    where
        V: SpectrumMagnitude;
}

/// The one coefficient step `adjoint` needs from a mode: the adjoint space
/// (`symtn_tensors::CoefficientAlgebra::adjoint_space`) with its error in the
/// facade's type. Implemented for every mode that has both, so `adjoint` and
/// the operations built on it are one body for every mode.
#[doc(hidden)]
pub trait TypedAdjointSpace<R>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
{
    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError>;
}

impl<R, M> TypedAdjointSpace<R> for M
where
    R: TypedSectorAdmission,
    M: TypedTensorModeDispatch<R> + symtn_tensors::CoefficientAlgebra<R>,
    M::FacadeError: From<<M as symtn_tensors::CoefficientAlgebra<R>>::Error>,
{
    fn adjoint_space(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        <M as symtn_tensors::CoefficientAlgebra<R>>::adjoint_space(space).map_err(Into::into)
    }
}

/// Canonical-unit leg execution selected by the admitted provider mode.
#[doc(hidden)]
pub trait TypedTensorUnitDispatch<R, D>: TypedTensorModeDispatch<R>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    fn insert_unit(
        tensor: &TensorMap<R, D>,
        insertion: UnitLegInsertion,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;

    fn remove_unit(
        tensor: &TensorMap<R, D>,
        axis: usize,
    ) -> Result<TensorMap<R, D>, Self::FacadeError>;
}

impl<R, D> TypedTensorUnitDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    fn insert_unit(
        tensor: &TensorMap<R, D>,
        insertion: UnitLegInsertion,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.insert_unit_multiplicity_free(insertion)
    }

    fn remove_unit(
        tensor: &TensorMap<R, D>,
        axis: usize,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        tensor.remove_unit_multiplicity_free(axis)
    }
}

impl<R, D> TypedTensorUnitDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedCanonicalUnitFusionRule,
    D: TensorScalar,
{
    fn insert_unit(
        tensor: &TensorMap<R, D>,
        insertion: UnitLegInsertion,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        generic_insert_unit(tensor, insertion)
    }

    fn remove_unit(
        tensor: &TensorMap<R, D>,
        axis: usize,
    ) -> Result<TensorMap<R, D>, Self::FacadeError> {
        generic_remove_unit(tensor, axis)
    }
}
