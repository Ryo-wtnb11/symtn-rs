//! Test oracle: composition of factor maps through the fusion contraction
//! engine. No production path recomposes factors this way; pinv, exp and
//! polar write each coupled sector by one dense GEMM (#1752), and their tests
//! check them against this independent recomposition.

use std::hash::Hash;
use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::MultiplicityFreeRigidSymbols;
use symtn_tensors::{
    ContractDestinationInit, FusionOperand, OperationError, TensorContractFusionExecutionContext,
    TreeTransformRuleCacheKey,
};

use crate::factorize::{BoundDynFactor, FactorScalar, TestTensor};

/// `lhs * rhs` of provider-bound factors by fusion contraction. The
/// destination is created from the operands' shared provider and remains
/// bound to that exact allocation.
pub(crate) fn compose_bound_dyn<RuleKey, BT, BC, R, D>(
    engine: &mut dyn Engine,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    lhs: &BoundDynFactor<R, D>,
    rhs: &BoundDynFactor<R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: symtn_tensors::TreeTransformBackend<D, f64>,
    BC: symtn_tensors::TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + symtn_tensors::RecouplingCoefficientAction<f64>,
{
    let lhs_space = lhs.space();
    let rhs_space = rhs.space();
    if lhs_space.space().nin() != rhs_space.space().nout() {
        return Err(OperationError::RankMismatch {
            expected: lhs_space.space().nin(),
            actual: rhs_space.space().nout(),
        });
    }
    let lhs_axes: Vec<usize> = (lhs_space.space().nout()..lhs_space.space().rank()).collect();
    let rhs_axes: Vec<usize> = (0..rhs_space.space().nout()).collect();
    let dst_space = symtn_tensors::BoundDynamicFusionMapSpace::contracted_multiplicity_free(
        lhs_space, rhs_space, &lhs_axes, &rhs_axes,
    )?;
    let mut dst_data = vec![
        D::zero();
        dst_space
            .space()
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?
    ];
    let output_axes: Vec<usize> = (0..dst_space.space().rank()).collect();
    context.tensorcontract_planned_into(
        engine,
        &dst_space,
        &mut dst_data,
        (FusionOperand::direct(lhs_space.space()), lhs.data()),
        (FusionOperand::direct(rhs_space.space()), rhs.data()),
        &lhs_axes,
        &rhs_axes,
        &output_axes,
        D::one(),
        ContractDestinationInit::Axpby(D::zero()),
    )?;
    let nout = dst_space.space().nout();
    let nin = dst_space.space().nin();
    BoundDynFactor::from_bound(dst_space, dst_data, nout, nin)
}

/// [`compose_bound_dyn`] over unbound test operands, binding both to `rule`.
pub(crate) fn compose<RuleKey, BT, BC, R, D>(
    engine: &mut dyn Engine,
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC>,
    rule: &R,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
) -> Result<TestTensor<D>, OperationError>
where
    RuleKey: Clone + Eq + Hash + Send + Sync + 'static,
    BT: symtn_tensors::TreeTransformBackend<D, f64>,
    BC: symtn_tensors::TensorContractBackend<D, f64>,
    R: Clone
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: FactorScalar + symtn_tensors::RecouplingCoefficientAction<f64>,
{
    let provider = Arc::new(rule.clone());
    let lhs = crate::factorize::BoundTensorMap::try_new(Arc::clone(&provider), lhs.clone())?;
    let rhs = crate::factorize::BoundTensorMap::try_new(Arc::clone(&provider), rhs.clone())?;
    let bind = |tensor: crate::factorize::BoundTensorMap<R, D>| {
        let (nout, nin) = (tensor.fusion_space().nout(), tensor.fusion_space().nin());
        BoundDynFactor::from_bound(tensor.space().clone(), tensor.data().to_vec(), nout, nin)
    };
    let out = compose_bound_dyn(engine, context, &bind(lhs)?, &bind(rhs)?)?;
    Ok(crate::factorize::typed_from_bound_factor(out)?
        .into_parts()
        .1)
}
