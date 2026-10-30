use super::*;

/// The device-local state of one CUDA operation: the dense context that
/// borrows the caller's CUDA engine, the tree-transform executor holding the
/// per-structure device state (uploaded coefficient vectors, pack/scatter
/// workspaces), and the general contraction's operand and core-destination
/// scratch.
///
/// Invocation-local: it lives for one public operation and is dropped with
/// it, so neither a Runtime nor any global holds an engine or device buffers
/// for later calls. A replay needs the context and the executor at the same
/// instant, which [`Self::split`] hands out.
pub(crate) struct CudaCall<'e> {
    dense: symtn_dense::CudaDenseContext<'e>,
    tree_transform: symtn_operations::CudaTreeTransformExecutor,
    contract_scratch: symtn_tensors::CudaContractScratch,
}

impl<'e> CudaCall<'e> {
    /// Borrows the dense context, the tree-transform executor and the
    /// contraction scratch at once.
    pub(crate) fn split_contract(
        &mut self,
    ) -> (
        &mut symtn_dense::CudaDenseContext<'e>,
        &mut symtn_operations::CudaTreeTransformExecutor,
        &mut symtn_tensors::CudaContractScratch,
    ) {
        (
            &mut self.dense,
            &mut self.tree_transform,
            &mut self.contract_scratch,
        )
    }

    /// Borrows the dense context and the tree-transform executor at once,
    /// which a replay needs because the executor submits its work through the
    /// context.
    pub(crate) fn split(
        &mut self,
    ) -> (
        &mut symtn_dense::CudaDenseContext<'e>,
        &mut symtn_operations::CudaTreeTransformExecutor,
    ) {
        (&mut self.dense, &mut self.tree_transform)
    }
}

impl<'e> std::ops::Deref for CudaCall<'e> {
    type Target = symtn_dense::CudaDenseContext<'e>;

    fn deref(&self) -> &Self::Target {
        &self.dense
    }
}

impl std::ops::DerefMut for CudaCall<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dense
    }
}

/// Opens one CUDA operation on the caller's engine. A non-CUDA engine is a
/// typed refusal before any upload or output mutation.
pub(crate) fn cuda_call(engine: &mut dyn symtn_dense::Engine) -> Result<CudaCall<'_>, Error> {
    let dense =
        symtn_dense::CudaDenseContext::new(engine).map_err(symtn_tensors::OperationError::Dense)?;
    Ok(CudaCall {
        dense,
        tree_transform: symtn_operations::CudaTreeTransformExecutor::default(),
        contract_scratch: symtn_tensors::CudaContractScratch::default(),
    })
}
