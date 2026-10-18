use super::*;

/// The facade half of a fusion mode's factorization contract. Every
/// factorization has one body over it; the mode chooses only what differs by
/// design or is still owned by a unification leaf (#1862, table in its body):
/// the error type and provider-label decode (D4, permanent). The lazy-adjoint
/// rule is the mode-independent [`FactorOp::adjoint_rule`] (#1755). The
/// numerical stages and factor spaces come from the matrix-algebra
/// [`FactorMode`](symtn_matrixalgebra::seam::FactorMode).
#[doc(hidden)]
pub trait FusionMode<R>: TypedAdjointSpace<R> + symtn_matrixalgebra::seam::FactorMode<R>
where
    R: TypedSectorAdmission,
{
    /// A matrix-algebra error as this mode's facade error.
    fn map_factor_error(
        error: <Self as symtn_matrixalgebra::seam::FactorMode<R>>::Error,
    ) -> Self::FacadeError;

    /// The public label of a coupled sector the provider's own algebra
    /// produced.
    fn decode_label(provider: &R, sector: SectorId) -> Result<R::Sector, Self::FacadeError>;

    /// A factor-space root constructor error as this mode's facade error.
    fn map_root_error(
        error: <Self as symtn_matrixalgebra::seam::FactorMode<R>>::RootError,
    ) -> Self::FacadeError;
}

/// A factorization that reads a possibly lazy-adjoint input.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorOp {
    SvdVals,
    EighVals,
    EigVals,
    QrCompact,
    QrFull,
    LqCompact,
    LqFull,
    SvdCompact,
    SvdFull,
    LeftNull,
    RightNull,
    LeftPolar,
    RightPolar,
    EighFull,
    EigFull,
    Inv,
    Pinv,
    Exp,
}

impl FactorOp {
    /// How `op` reads a lazy adjoint, in every fusion mode (#1755): the
    /// identity relating the operation of `A^H` to one of `A` is fixed by the
    /// mathematics, so only the dense stages that execute it are per mode.
    pub(super) const fn adjoint_rule(self) -> AdjointRule {
        match self {
            // Singular values and coupled sectors are invariant under adjoint.
            Self::SvdVals => AdjointRule::Parent,
            // `A^H = V S U^H`: the mode's adjoint stage factors the parent and
            // gauges the final left factor `V`, as materialize-then-SVD
            // would; and `(A^H)^+ = U S^+ Vh` from the same parent SVD.
            // LQ of `A^H` is computed as the QR of its adjoint, which is the
            // parent `A` itself, so the stage reads `A` and publishes
            // `A^H = R^H Q^H` (#2070).
            Self::SvdCompact | Self::SvdFull | Self::Pinv | Self::LqCompact | Self::LqFull => {
                AdjointRule::AdjointSeam
            }
            // The null space of `A^H` is the adjoint of the opposite null
            // space of `A`, `left_polar(A^H)` is the adjoint-swapped
            // `right_polar(A)` (and vice versa), and `(A^H)^-1 = (A^-1)^H`.
            Self::LeftNull | Self::RightNull | Self::LeftPolar | Self::RightPolar | Self::Inv => {
                AdjointRule::Redirect
            }
            // Why not read QR's parent through LQ (TensorKit's
            // `qr_compact!` on an `AdjointTensorMap`): LQ is itself the QR of
            // the adjoint, so it would form this copy anyway and add two
            // factor adjoints.
            Self::QrCompact | Self::QrFull => AdjointRule::Materialize,
            // Why not read the parent: an admitted near-Hermitian input
            // differs from its adjoint and the solver reads one triangle; the
            // values of `B^H` are `conj` of those of `B` only as a multiset,
            // so the published order would change; and the right
            // eigenvectors of `B^H` are the left ones of `B`.
            // `exp` likewise dispatches a near-Hermitian input to the
            // spectral route. The materialization runs on the dense route,
            // after the shared preflight and before the lease.
            Self::EighVals | Self::EighFull | Self::EigVals | Self::EigFull | Self::Exp => {
                AdjointRule::Materialize
            }
        }
    }
}

/// What a factorization reads when its input is a lazy adjoint.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjointRule {
    /// The dense parent, for a result that is invariant under adjoint.
    Parent,
    /// An operation-local materialized adjoint.
    Materialize,
    /// The partner operation on the parent, whose results are adjointed
    /// back and detached (the left null space of `t^H` from the right null
    /// space of `t`).
    Redirect,
    /// The dense parent through the factorization's adjoint-aware stage,
    /// which returns the factors of the adjoint without forming it.
    AdjointSeam,
}

impl<R> FusionMode<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn map_factor_error(error: symtn_tensors::OperationError) -> Error {
        error.into()
    }

    fn decode_label(
        provider: &R,
        sector: SectorId,
    ) -> Result<<R as TypedSectorAdmission>::Sector, Error> {
        // The blanket `TypedSectorAdmission` impl (the only one a
        // `CheckedFusionAlgebra + SectorCodec` rule can have) decodes through
        // `SectorCodec::decode_sector`.
        Ok(provider.try_decode_label(sector)?)
    }

    fn map_root_error(error: symtn_tensors::OperationError) -> Error {
        error.into()
    }
}

impl<R> FusionMode<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    fn map_factor_error(
        error: symtn_matrixalgebra::seam::CheckedGenericFactorPlanError<
            <R as CheckedGenericFusion>::Error,
        >,
    ) -> Self::FacadeError {
        error.into()
    }

    fn decode_label(provider: &R, sector: SectorId) -> Result<R::Sector, Self::FacadeError> {
        provider
            .try_decode_label(sector)
            .map_err(|error| GenericTensorError::Plan(CheckedGenericPlanError::Provider(error)))
    }

    fn map_root_error(
        error: CheckedGenericStructureError<<R as CheckedGenericFusion>::Error>,
    ) -> Self::FacadeError {
        error.into()
    }
}
