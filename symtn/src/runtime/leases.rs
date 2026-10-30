use super::*;

/// The execution scratch of one standalone operation (tree-transform and
/// contraction lanes), built for the call and dropped when it returns, or kept
/// by a prepared workspace for its replays. It holds only the lanes the
/// operation used (each boxed on first use, so moving the context moves a few
/// pointers, #228), structural state and scratch; the operation runs on the
/// caller's engine.
pub(crate) struct ContextLease(TensorExecutionContext);

impl ContextLease {
    pub(crate) fn new() -> Self {
        Self(TensorExecutionContext::new())
    }

    pub(crate) fn context(&mut self) -> &mut TensorExecutionContext {
        &mut self.0
    }
}

/// The dense executor one factorization owns for its duration. In unit
/// tests it may be one the probe injected, which returns to the probe on drop.
pub(crate) struct DenseLease {
    pub(crate) executor: Option<Box<dyn symtn_dense::DenseExecutor + Send>>,
    #[cfg(test)]
    pub(super) injected: bool,
}

impl DenseLease {
    pub(crate) fn dense(&mut self) -> &mut (dyn symtn_dense::DenseExecutor + Send) {
        &mut **self
            .executor
            .as_mut()
            .expect("dense lease always owns an executor")
    }
}

#[cfg(test)]
impl Drop for DenseLease {
    fn drop(&mut self) {
        if self.injected && !std::thread::panicking() {
            if let Some(executor) = self.executor.take() {
                crate::typed::probe::return_injected_dense(executor);
            }
        }
    }
}
