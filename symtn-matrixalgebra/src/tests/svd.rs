//! SVD factorization tests.

use super::*;

mod checked;
mod compact;
mod derived;
mod full;

/// The dense executor of a stage that must not run, counting the leases.
pub(super) struct CountingLease<'a>(pub(super) &'a Cell<usize>);

impl ExecutorLease for CountingLease<'_> {
    type Executor = symtn_dense::DefaultDenseExecutor;

    fn run<T>(self, stage: impl FnOnce(&mut Self::Executor) -> T) -> T {
        self.0.set(self.0.get() + 1);
        stage(&mut symtn_dense::DefaultDenseExecutor::new())
    }
}

#[test]
fn factorization_without_adjoint_route_refuses_lazy_adjoint_before_leasing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let bound = bound_tensor(
        Arc::new(ZNFusionRule::new(2).unwrap()),
        &rectangular_svd_tensor(3, 2),
    );
    let leases = Cell::new(0);
    let refused = factor::<symtn_core::MultiplicityFreeAdmissionMode, _, _, _, _>(
        engine,
        CountingLease(&leases),
        op::SvdVals,
        FactorSource::AdjointOf {
            parent: bound.as_ref().dynamic(),
            adjoint_space: bound.space(),
        },
    );
    assert!(matches!(
        refused,
        Err(OperationError::UnsupportedTensorContractScope { .. })
    ));
    assert_eq!(leases.get(), 0);
}
