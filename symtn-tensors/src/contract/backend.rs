use num_traits::One;
use std::sync::Arc;
use symtn_core::{BlockStructure, Placement};
use symtn_dense::Engine;
use symtn_dense::{DenseExecutor, DenseView, DenseViewMut};
use symtn_operations::fusion_replay::{
    direct_slice, direct_slice_mut, MatrixOp, Rank2GemmBatchJob,
};

use crate::host_scratch::HostScratchBuffer;
use symtn_operations::{scale_raw_strided_kernel_trusted, tensoradd_raw_strided_kernel};

use crate::{
    ConjugateValue, DenseBlockScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction, ReportsPlacement,
};

use super::structure::{
    TensorContractDenseRouteOrder, TensorContractDescriptor, TensorContractDescriptorTerm,
    TensorContractStructure,
};

/// Tensor-contraction execution contract over host-accessible data.
///
/// The raw replay methods take host slices. Code that specifically depends on
/// this host-slice contract may use `HostTensorContractBackend`. Placement-aware
/// (device or distributed) contraction traits do not inherit from this raw-slice
/// API.
pub trait TensorContractBackend<D, C = f64>
where
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    type Workspace;

    #[expect(
        clippy::too_many_arguments,
        reason = "the raw backend trait keeps three structure/data pairs, workspace, and alpha/beta explicit"
    )]
    fn tensorcontract_structure_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;

    /// Writes the full rank-2 matrix product `lhs * rhs` into `dst_data`.
    ///
    /// Implementations must overwrite every element of the `rows x cols`
    /// output buffer; callers may reuse dirty workspace and must not rely on
    /// pre-cleared destination storage.
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError>;

    /// Accumulate-form rank-2 GEMM: `dst = alpha * lhs * rhs + beta * dst`
    /// over column-major slices (BLAS semantics). The default only covers the
    /// overwrite case; accumulate-capable backends override it.
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_axpby_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_rank2_into_raw(
                engine, workspace, dst_data, lhs_data, rhs_data, rows, contracted, cols,
            );
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "contract backend does not implement the accumulate-form rank-2 GEMM",
        })
    }

    /// Batched accumulate-form rank-2 GEMM addressed by element offsets into
    /// shared column-major buffers (see [`Rank2GemmBatchJob`]). The caller
    /// guarantees the batch's destination ranges are pairwise disjoint, so
    /// implementations may run jobs in any order or concurrently; the default
    /// executes them serially through `matmul_rank2_axpby_into_raw`.
    ///
    /// `runs` is the batch's plan-time run partition (see issue #103), forwarded
    /// to backends that route runs; the serial default ignores it.
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_batch_axpby_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let _ = runs;
        for job in jobs {
            let lhs_slice = direct_slice(lhs_data, job.lhs_offset, job.rows, job.contracted)?;
            let rhs_slice = direct_slice(rhs_data, job.rhs_offset, job.contracted, job.cols)?;
            let dst_slice = direct_slice_mut(dst_data, job.dst_offset, job.rows, job.cols)?;
            self.matmul_rank2_axpby_into_raw(
                engine,
                workspace,
                dst_slice,
                lhs_slice,
                rhs_slice,
                job.rows,
                job.contracted,
                job.cols,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_batch_with_ops_axpby_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
            return self.matmul_rank2_batch_axpby_into_raw(
                engine, workspace, dst_data, lhs_data, rhs_data, jobs, runs, alpha, beta,
            );
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "contract backend does not implement transpose/adjoint rank-2 batches",
        })
    }
}

/// Explicit marker for the host-slice tensor-contract backend family.
///
/// `TensorContractBackend` carries the methods; this marker means “implements the
/// host-slice replay contract,” not necessarily “physically CPU-native.”
/// Device or distributed contraction backends use separate placement-aware
/// execution traits.
pub trait HostTensorContractBackend<D, C = f64>: TensorContractBackend<D, C>
where
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
}

impl<B, D, C> HostTensorContractBackend<D, C> for B
where
    B: TensorContractBackend<D, C> + ?Sized,
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
}

/// Host scratch/replay workspace backed by `Vec<T>`.
///
/// Raw contraction replay using this workspace operates on host slices. Device
/// execution should use a separate device workspace instead of hiding device
/// storage behind this type.
#[derive(Clone, Debug)]
pub struct HostTensorContractWorkspace<T> {
    output: HostScratchBuffer<T>,
    zero_strides: Vec<isize>,
}

pub type TensorContractWorkspace<T> = HostTensorContractWorkspace<T>;

impl<T> Default for HostTensorContractWorkspace<T> {
    fn default() -> Self {
        Self {
            output: HostScratchBuffer::default(),
            zero_strides: Vec::new(),
        }
    }
}

impl<T> HostTensorContractWorkspace<T> {
    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    fn prepare_output(&mut self, len: usize, zero: T)
    where
        T: Clone,
    {
        self.output.resize_filled(len, zero);
    }
}

impl<T> ReportsPlacement for HostTensorContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<E, D, C> TensorContractBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    type Workspace = TensorContractWorkspace<D>;

    fn tensorcontract_structure_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TensorContractStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        tensorcontract_structure_with_dense_executor_raw(
            engine,
            self.dense_mut(),
            workspace,
            structure,
            dst_structure,
            lhs_structure,
            rhs_structure,
            dst_data,
            lhs_data,
            rhs_data,
            alpha,
            beta,
        )
    }

    fn matmul_rank2_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        _workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        let lhs_shape = [rows, contracted];
        let lhs_strides = [1, rows];
        let rhs_shape = [contracted, cols];
        let rhs_strides = [1, contracted];
        let dst_shape = [rows, cols];
        let dst_strides = [1, rows];
        // Layout validated at plan compile: callers hand slices whose lengths
        // were bounds-checked against rows/contracted/cols when the fusion
        // block plan (or its scratch) was built, so replay uses the trusted
        // constructors (debug builds re-validate).
        let lhs = D::dense_read(DenseView::new_trusted(
            lhs_data,
            &lhs_shape,
            &lhs_strides,
            0,
        ));
        let rhs = D::dense_read(DenseView::new_trusted(
            rhs_data,
            &rhs_shape,
            &rhs_strides,
            0,
        ));
        let output = D::dense_write(DenseViewMut::new_trusted(
            dst_data,
            &dst_shape,
            &dst_strides,
            0,
        ));
        self.dense_mut()
            .matmul_into(engine, output, lhs, rhs)
            .map_err(OperationError::Dense)
    }

    fn matmul_rank2_axpby_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        _workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let lhs_shape = [rows, contracted];
        let lhs_strides = [1, rows];
        let rhs_shape = [contracted, cols];
        let rhs_strides = [1, contracted];
        let dst_shape = [rows, cols];
        let dst_strides = [1, rows];
        // Same plan-compile validation contract as matmul_rank2_into_raw.
        let lhs = D::dense_read(DenseView::new_trusted(
            lhs_data,
            &lhs_shape,
            &lhs_strides,
            0,
        ));
        let rhs = D::dense_read(DenseView::new_trusted(
            rhs_data,
            &rhs_shape,
            &rhs_strides,
            0,
        ));
        let output = D::dense_write(DenseViewMut::new_trusted(
            dst_data,
            &dst_shape,
            &dst_strides,
            0,
        ));
        self.dense_mut()
            .matmul_axpby_into(
                engine,
                output,
                lhs,
                rhs,
                alpha.dense_scalar(),
                beta.dense_scalar(),
            )
            .map_err(OperationError::Dense)
    }

    fn matmul_rank2_batch_axpby_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        _workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        // Whole storage buffers as flat rank-1 views; the batch jobs carry the
        // per-matrix offsets. Same plan-compile validation contract as
        // matmul_rank2_axpby_into_raw.
        let dst_shape = [dst_data.len()];
        let lhs_shape = [lhs_data.len()];
        let rhs_shape = [rhs_data.len()];
        let flat_strides = [1];
        let lhs = D::dense_read(DenseView::new_trusted(
            lhs_data,
            &lhs_shape,
            &flat_strides,
            0,
        ));
        let rhs = D::dense_read(DenseView::new_trusted(
            rhs_data,
            &rhs_shape,
            &flat_strides,
            0,
        ));
        let output = D::dense_write(DenseViewMut::new_trusted(
            dst_data,
            &dst_shape,
            &flat_strides,
            0,
        ));
        self.dense_mut()
            .matmul_batch_axpby_into(
                engine,
                output,
                lhs,
                rhs,
                jobs,
                runs,
                alpha.dense_scalar(),
                beta.dense_scalar(),
            )
            .map_err(OperationError::Dense)
    }

    fn matmul_rank2_batch_with_ops_axpby_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        _workspace: &mut Self::Workspace,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let dst_shape = [dst_data.len()];
        let lhs_shape = [lhs_data.len()];
        let rhs_shape = [rhs_data.len()];
        let flat_strides = [1];
        let lhs = D::dense_read(DenseView::new_trusted(
            lhs_data,
            &lhs_shape,
            &flat_strides,
            0,
        ));
        let rhs = D::dense_read(DenseView::new_trusted(
            rhs_data,
            &rhs_shape,
            &flat_strides,
            0,
        ));
        let output = D::dense_write(DenseViewMut::new_trusted(
            dst_data,
            &dst_shape,
            &flat_strides,
            0,
        ));
        self.dense_mut()
            .matmul_batch_axpby_with_ops_into(
                engine,
                output,
                lhs,
                rhs,
                jobs,
                runs,
                lhs_op,
                rhs_op,
                alpha.dense_scalar(),
                beta.dense_scalar(),
            )
            .map_err(OperationError::Dense)
    }
}

/// Replays a prepared tensor contraction structure on host slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_structure_with_dense_executor_raw<E, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TensorContractWorkspace<D>,
    structure: &TensorContractStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    lhs_structure: &Arc<BlockStructure>,
    rhs_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    structure.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
    validate_contract_replay_storage_len(dst_structure, dst_data.len())?;
    validate_contract_replay_storage_len(lhs_structure, lhs_data.len())?;
    validate_contract_replay_storage_len(rhs_structure, rhs_data.len())?;
    scale_inactive_contract_destinations(structure, dst_data, beta)?;
    let descriptor = structure.descriptor();
    for term in descriptor.terms() {
        workspace.prepare_output(term.workspace_len, D::zero());
        tensorcontract_descriptor_term_with_output_scratch(
            engine,
            dense,
            &mut workspace.zero_strides,
            descriptor,
            term,
            dst_data,
            workspace.output.as_mut_slice(),
            lhs_data,
            rhs_data,
            alpha,
            beta,
        )?;
    }
    Ok(())
}

fn validate_contract_replay_storage_len(
    structure: &BlockStructure,
    actual: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual != expected {
        return Err(OperationError::ElementCountMismatch { expected, actual });
    }
    Ok(())
}

fn scale_inactive_contract_destinations<D, C>(
    structure: &TensorContractStructure<C>,
    dst_data: &mut [D],
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar,
    C: Copy + One,
{
    if beta == D::one() {
        return Ok(());
    }
    for layout in structure.inactive_destination_scale_blocks() {
        scale_raw_strided_kernel_trusted(
            dst_data,
            &layout.block.shape,
            &layout.block.strides,
            layout.block.offset,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tensorcontract_descriptor_term_with_output_scratch<E, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    zero_strides: &mut Vec<isize>,
    descriptor: &TensorContractDescriptor<C>,
    term: &TensorContractDescriptorTerm<C>,
    dst_data: &mut [D],
    output_scratch: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + One,
{
    // Conjugation is carried by `descriptor.dot_config()` (dot-operand order)
    // and folded into the dense kernel, so conjugated and non-conjugated
    // contractions share this single GEMM path — no scalar-loop fallback.
    let logical_lhs = DenseView::new(
        lhs_data,
        descriptor.lhs_shape(term),
        descriptor.lhs_strides(term),
        term.lhs_offset,
    )
    .map_err(OperationError::Dense)?;
    let logical_rhs = DenseView::new(
        rhs_data,
        descriptor.rhs_shape(term),
        descriptor.rhs_strides(term),
        term.rhs_offset,
    )
    .map_err(OperationError::Dense)?;
    let (lhs, rhs) = match descriptor.dense_route_order() {
        TensorContractDenseRouteOrder::LhsRhs => {
            (D::dense_read(logical_lhs), D::dense_read(logical_rhs))
        }
        TensorContractDenseRouteOrder::RhsLhs => {
            (D::dense_read(logical_rhs), D::dense_read(logical_lhs))
        }
    };
    let output = D::dense_write(
        DenseViewMut::new(
            output_scratch,
            descriptor.output_shape(term),
            descriptor.output_strides(term),
            0,
        )
        .map_err(OperationError::Dense)?,
    );
    let term_alpha = alpha.scale_by_coefficient(term.coefficient);
    // A zero `alpha` never forms `A * B`, as BLAS `gemm` and TensorKit's
    // `mul!` do not (#1442). The skip only saves cost: the zero arm of the
    // add below (#1447) already contributes an exact zero whatever the
    // scratch holds.
    if !term_alpha.is_zero() {
        dense
            .dot_general_into(engine, output, lhs, rhs, descriptor.dot_config())
            .map_err(OperationError::Dense)?;
    }
    let term_beta = if term.apply_beta { beta } else { D::one() };
    symtn_operations::host_pool::with_exec(engine, |exec| {
        tensoradd_raw_strided_kernel(
            exec,
            zero_strides,
            dst_data,
            output_scratch,
            descriptor.scatter_shape(term),
            descriptor.dst_strides(term),
            descriptor.workspace_strides(term),
            term.dst_offset,
            0,
            false,
            term_alpha,
            term_beta,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DefaultBatchBackend;

    impl TensorContractBackend<f64, f64> for DefaultBatchBackend {
        type Workspace = usize;

        fn tensorcontract_structure_into_raw(
            &mut self,
            _engine: &mut dyn Engine,
            _workspace: &mut Self::Workspace,
            _structure: &TensorContractStructure<f64>,
            _dst_structure: &Arc<BlockStructure>,
            _lhs_structure: &Arc<BlockStructure>,
            _rhs_structure: &Arc<BlockStructure>,
            _dst_data: &mut [f64],
            _lhs_data: &[f64],
            _rhs_data: &[f64],
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            unreachable!("default batch tests do not execute structure replay")
        }

        fn matmul_rank2_into_raw(
            &mut self,
            _engine: &mut dyn Engine,
            workspace: &mut Self::Workspace,
            dst_data: &mut [f64],
            lhs_data: &[f64],
            rhs_data: &[f64],
            rows: usize,
            contracted: usize,
            cols: usize,
        ) -> Result<(), OperationError> {
            *workspace += 1;
            for col in 0..cols {
                for row in 0..rows {
                    dst_data[row + rows * col] = (0..contracted)
                        .map(|inner| {
                            lhs_data[row + rows * inner] * rhs_data[inner + contracted * col]
                        })
                        .sum();
                }
            }
            Ok(())
        }
    }

    fn assert_tensor_contract_backend<B, D, C>()
    where
        B: TensorContractBackend<D, C>,
        D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
        C: Copy + One,
    {
    }

    fn assert_host_tensor_contract_backend<B, D, C>()
    where
        B: HostTensorContractBackend<D, C>,
        D: DenseBlockScalar + ConjugateValue + RecouplingCoefficientAction<C>,
        C: Copy + One,
    {
    }

    #[test]
    fn dense_tree_transform_operations_implements_both_contract_backend_traits() {
        assert_tensor_contract_backend::<DenseTreeTransformOperations, f64, f64>();
        assert_host_tensor_contract_backend::<DenseTreeTransformOperations, f64, f64>();
    }

    #[test]
    fn default_op_batch_delegates_identity_and_rejects_matrix_views() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: the backend-neutral default preserves existing Identity batches
        // and declines transpose/adjoint unless a backend opts into them.
        let jobs = [Rank2GemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows: 1,
            contracted: 1,
            cols: 1,
        }];
        let mut backend = DefaultBatchBackend;
        let mut calls = 0;
        let mut dst = [0.0];
        backend
            .matmul_rank2_batch_with_ops_axpby_into_raw(
                engine,
                &mut calls,
                &mut dst,
                &[2.0],
                &[3.0],
                &jobs,
                &[1],
                MatrixOp::Identity,
                MatrixOp::Identity,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(dst, [6.0]);
        assert_eq!(calls, 1);

        let error = backend
            .matmul_rank2_batch_with_ops_axpby_into_raw(
                engine,
                &mut calls,
                &mut dst,
                &[2.0],
                &[3.0],
                &jobs,
                &[1],
                MatrixOp::Adjoint,
                MatrixOp::Identity,
                1.0,
                0.0,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            OperationError::UnsupportedTensorContractScope {
                message: "contract backend does not implement transpose/adjoint rank-2 batches"
            }
        ));
        assert_eq!(calls, 1);
    }

    #[test]
    fn dense_tree_backend_executes_op_bearing_batch() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: the built-in backend forwards matrix operations to its dense
        // executor instead of taking the backend-neutral rejection path.
        let jobs = [Rank2GemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows: 1,
            contracted: 1,
            cols: 1,
        }];
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TensorContractWorkspace::default();
        let mut dst = [0.0];
        <DenseTreeTransformOperations as TensorContractBackend<f64, f64>>::matmul_rank2_batch_with_ops_axpby_into_raw(
                &mut backend, engine,
                &mut workspace,
                &mut dst,
                &[2.0],
                &[3.0],
                &jobs,
                &[1],
                MatrixOp::Adjoint,
                MatrixOp::Identity,
                1.0,
                0.0,
            )
            .unwrap();
        assert_eq!(dst, [6.0]);
    }
}
