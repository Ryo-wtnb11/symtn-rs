//! Shared fixtures, spy executors and imports for symtn-matrixalgebra's
//! test suite, split by responsibility into sibling files.

use symtn_core::{
    BlockKey, BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, CoreError, CoupledSectorFold, FermionParityFusionRule,
    FusionProductSpace, FusionRule, FusionStyleKind, FusionTreeHomSpace, FusionTreeKey,
    GenericFArray, GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols, InfallibleGeneric,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorVec, U1FusionRule, U1Irrep,
    Z2FusionRule,
};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, OperationError,
    OutputAxisOrder, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{
    pinv_cutoff, typed_from_bound_factor, validate_inverse_region_routes_for_test, BoundTensorMap,
    TestTensor,
};
use crate::test_numerics::numerics;
use crate::*;
use crate::{LeftPolar, Lq, Qr, RightPolar, Svd};
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use std::{cell::Cell, convert::Infallible, fmt, sync::Arc};
use symtn_dense::{
    arity_mismatch, DenseBackend, DenseError, DenseExecutor, DenseOwned, DenseRead, DenseTensor,
    DenseWrite,
};

macro_rules! bound_tensor_ref {
    ($provider:expr, $tensor:expr) => {
        bound_tensor($provider, $tensor).as_ref()
    };
}

mod fixtures;
mod generic_fixtures;
pub(crate) mod scripted_executor;
mod spies;
use fixtures::*;
use generic_fixtures::*;
use spies::*;

// The checked values entries under their earlier names: the one entry per
// family in checked mode, on a source or on dense storage.
fn svd_vals_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    svd_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(engine, lease, source)
}

// The checked inverse and solve admission, read through the mode's
// factor-space authority.
fn factor_isomorphic_checked_generic<R: CheckedGenericFusion>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<bool, CheckedGenericFactorPlanError<R::Error>> {
    use crate::factorize::{FactorMode, FactorSpaceAuthority};
    <CheckedGenericAdmissionMode as FactorMode<R>>::authority(space).isomorphic(space)
}

fn eig_vals_checked_generic<L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, CheckedGenericFactorPlanError<R::Error>>
where
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    eig_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(engine, lease, source)
}

fn dense_source<'a, R, D>(input: &'a BoundDynamicTensorRef<'_, R, D>) -> FactorSource<'a, R, D> {
    FactorSource::Dense(BoundDynamicTensorRef::try_new(input.space(), input.data()).unwrap())
}

fn svd_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_vals_checked_generic(dense, dense_source(input))
}

fn eigh_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    eigh_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
        engine,
        dense,
        dense_source(input),
        HermitianTol::DEFAULT,
    )
}

fn eig_vals_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    eig_vals_checked_generic(dense, dense_source(input))
}

mod eigh_eig;
mod exact_generic_dim;
mod generic_dispatch;
mod matrix_functions;
mod null_space;
mod polar;
mod qr_lq;
mod svd;
