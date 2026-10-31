use std::sync::Arc;

use symtn_core::{BlockStructure, Placement};
use symtn_dense::{DefaultDenseExecutor, DenseExecutor, Engine};

use crate::ReportsPlacement;

use crate::{
    ConjugateValue, DenseRecouplingScalar, HostKernelAdapter, OperationError,
    RecouplingCoefficientAction, StridedHostKernelAdapter, TreeTransformReplayProfile,
    TreeTransformScalar, TreeTransformStructure, TreeTransformWorkspace,
};

/// Tree-transform execution contract over host-accessible data.
///
/// The raw replay methods take host slices. Code that specifically depends on
/// this host-slice contract may use `HostTreeTransformBackend`. Placement-aware
/// (device or distributed) transform traits do not inherit from this raw-slice
/// API.
pub trait TreeTransformBackend<D, C>
where
    D: TreeTransformScalar,
    C: Copy,
{
    type Workspace;

    #[expect(
        clippy::too_many_arguments,
        reason = "the raw backend contract keeps workspace, replay structures, buffers, and alpha/beta explicit"
    )]
    fn tree_transform_structure_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;

    /// Overwrites `dst` with `alpha * T(src)`, the move writing destination
    /// block `b` scaled by `θ_b` from `destination_scales` (`(b, θ_b)`,
    /// strictly increasing offsets; unlisted blocks have `θ_b = 1`).
    ///
    /// The provided implementation has no per-move access: after its zero
    /// and accumulate passes it scales each listed block in place, one extra
    /// pass over those blocks. The structural-recoupling backends fold the
    /// scales into their moves instead.
    #[expect(
        clippy::too_many_arguments,
        reason = "the overwrite backend contract keeps workspace, replay structures, buffers, alpha and destination scales explicit"
    )]
    fn tree_transform_structure_overwrite_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        validate_destination_scales(destination_scales)?;
        structure.validate_replay_structures(dst_structure, src_structure)?;
        crate::transform_replay::validate_replay_storage_len(src_structure, src_data.len())?;
        let mut kernels = StridedHostKernelAdapter::default();
        let mut zero_strides = Vec::new();
        crate::host_pool::with_exec(engine, |exec| {
            crate::transform_replay::zero_tree_transform_destination(
                exec,
                &mut kernels,
                &mut zero_strides,
                dst_structure,
                dst_data,
            )
        })?;
        self.tree_transform_structure_into_raw(
            engine,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            D::one(),
        )?;
        crate::host_pool::with_exec(engine, |exec| {
            scale_destination_blocks(
                exec,
                &mut kernels,
                dst_structure,
                dst_data,
                destination_scales,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_transform_structure_into_raw_profiled(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let start = std::time::Instant::now();
        let result = self.tree_transform_structure_into_raw(
            engine,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        );
        profile.total += start.elapsed();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_transform_structure_overwrite_into_raw_profiled(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        destination_scales: &[(usize, C)],
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let start = std::time::Instant::now();
        let result = self.tree_transform_structure_overwrite_into_raw(
            engine,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            destination_scales,
        );
        profile.total += start.elapsed();
        result
    }
}

fn validate_destination_scales<C>(scales: &[(usize, C)]) -> Result<(), OperationError> {
    if scales.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err(OperationError::InvalidArgument {
            message: "tree transform destination scales must have strictly increasing offsets",
        });
    }
    Ok(())
}

/// Multiplies every non-empty destination block whose offset `scales` lists
/// by its `θ_b` in place: the provided overwrite's explicit scale pass.
fn scale_destination_blocks<D, C>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut StridedHostKernelAdapter,
    dst_structure: &BlockStructure,
    dst_data: &mut [D],
    scales: &[(usize, C)],
) -> Result<(), OperationError>
where
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    if scales.is_empty() {
        return Ok(());
    }
    for index in 0..dst_structure.block_count() {
        let block = dst_structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        if block.shape().contains(&0) {
            continue;
        }
        let Ok(at) = scales.binary_search_by_key(&block.offset(), |&(offset, _)| offset) else {
            continue;
        };
        kernels.scale_strided(
            exec.reborrow(),
            dst_data,
            block.shape(),
            &crate::strided::strides_to_isize(block.strides())?,
            crate::strided::offset_to_isize(block.offset())?,
            D::coefficient_as_data(scales[at].1),
        )?;
    }
    Ok(())
}

/// Explicit marker for the host-slice tree-transform backend family.
///
/// `TreeTransformBackend` carries the methods; this marker means “implements the
/// host-slice replay contract,” not necessarily “physically CPU-native.”
/// Device or distributed transform backends use separate placement-aware
/// execution traits.
pub trait HostTreeTransformBackend<D, C>: TreeTransformBackend<D, C>
where
    D: TreeTransformScalar,
    C: Copy,
{
}

impl<B, D, C> HostTreeTransformBackend<D, C> for B
where
    B: TreeTransformBackend<D, C> + ?Sized,
    D: TreeTransformScalar,
    C: Copy,
{
}

#[derive(Debug)]
pub struct DenseTreeTransformOperations<E = DefaultDenseExecutor> {
    dense: E,
}

impl DenseTreeTransformOperations<DefaultDenseExecutor> {
    pub fn default_executor() -> Self {
        Self::new(DefaultDenseExecutor::new())
    }
}

impl<E> DenseTreeTransformOperations<E> {
    pub fn new(dense: E) -> Self {
        Self { dense }
    }

    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    pub fn dense(&self) -> &E {
        &self.dense
    }

    pub fn dense_mut(&mut self) -> &mut E {
        &mut self.dense
    }

    #[inline]
    fn kernel_adapter(&self) -> StridedHostKernelAdapter {
        StridedHostKernelAdapter::default()
    }

    /// Worker count for one replay into `dst_len` elements: the caller
    /// engine's width when the engine's own cost rule admits the section,
    /// otherwise 1 (see [`crate::host_pool::copy_width`]).
    #[inline]
    fn effective_recoupling_threads(
        &self,
        engine: &mut dyn Engine,
        dst_len: usize,
    ) -> Result<usize, OperationError> {
        crate::host_pool::copy_width(engine, dst_len)
    }
}

impl<E: DenseExecutor> DenseTreeTransformOperations<E> {
    /// Owned-output transform under this backend's replay worker count and
    /// size gate, exactly as `tree_transform_structure_overwrite_into_raw`
    /// would replay into a caller buffer of the same length.
    ///
    /// Internal and unstable; see
    /// [`crate::try_tree_transform_structure_overwrite_owned_raw`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tree_transform_structure_overwrite_owned_raw<D, C>(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut TreeTransformWorkspace<D>,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        nout: usize,
        src_data: &[D],
        alpha: D,
    ) -> Result<Option<Vec<D>>, OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
        C: Copy + Sync,
    {
        // A malformed destination reports its error from the overwrite proof
        // in the same order as before; the serial count only sizes scratch.
        let threads = match dst_structure.required_len() {
            Ok(len) => self.effective_recoupling_threads(engine, len)?,
            Err(_) => 1,
        };
        crate::try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            nout,
            src_data,
            alpha,
            threads,
        )
    }
}

impl<E> ReportsPlacement for DenseTreeTransformOperations<E> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl Default for DenseTreeTransformOperations<DefaultDenseExecutor> {
    fn default() -> Self {
        Self::default_executor()
    }
}

impl<E, D, C> TreeTransformBackend<D, C> for DenseTreeTransformOperations<E>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Workspace = TreeTransformWorkspace<D>;

    fn tree_transform_structure_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(engine, dst_data.len())?;
        crate::tree_transform_structure_with_structural_recoupling_raw(
            engine,
            &mut self.kernel_adapter(),
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
            threads,
        )
    }

    fn tree_transform_structure_overwrite_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(engine, dst_data.len())?;
        crate::tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut self.kernel_adapter(),
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            destination_scales,
            threads,
        )
    }

    fn tree_transform_structure_into_raw_profiled(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(engine, dst_data.len())?;
        crate::tree_transform_structure_with_structural_recoupling_raw_profiled(
            engine,
            &mut self.kernel_adapter(),
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
            threads,
            profile,
        )
    }

    fn tree_transform_structure_overwrite_into_raw_profiled(
        &mut self,
        engine: &mut dyn Engine,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        destination_scales: &[(usize, C)],
        profile: &mut TreeTransformReplayProfile,
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(engine, dst_data.len())?;
        crate::tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
            engine,
            &mut self.kernel_adapter(),
            &mut self.dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            destination_scales,
            threads,
            profile,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overwrite_fixture() -> (
        Arc<BlockStructure>,
        Arc<BlockStructure>,
        TreeTransformStructure<f64>,
    ) {
        let src = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
        let dst = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let structure = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[crate::TreeTransformBlockSpec::single(0, 0, 2.0)],
        )
        .unwrap();
        (dst, src, structure)
    }

    struct RequiredMethodsOnlyBackend;

    impl TreeTransformBackend<f64, f64> for RequiredMethodsOnlyBackend {
        type Workspace = TreeTransformWorkspace<f64>;

        fn tree_transform_structure_into_raw(
            &mut self,
            engine: &mut dyn Engine,
            workspace: &mut Self::Workspace,
            structure: &TreeTransformStructure<f64>,
            dst_structure: &Arc<BlockStructure>,
            src_structure: &Arc<BlockStructure>,
            dst_data: &mut [f64],
            src_data: &[f64],
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            DenseTreeTransformOperations::default_executor().tree_transform_structure_into_raw(
                engine,
                workspace,
                structure,
                dst_structure,
                src_structure,
                dst_data,
                src_data,
                alpha,
                beta,
            )
        }
    }

    #[test]
    fn replay_width_follows_the_engine_cost_rule() {
        let backend = DenseTreeTransformOperations::default();
        // A pool that declares free dispatch admits even a one-element section.
        let mut eager = crate::host_pool::eager_test_pool(4);
        eager.install(|engine| {
            assert_eq!(backend.effective_recoupling_threads(engine, 1).unwrap(), 4);
        });
        // A measured pool forks a large copy but not a tiny one.
        let mut measured = crate::host_pool::test_pool(4);
        measured.install(|engine| {
            assert_eq!(backend.effective_recoupling_threads(engine, 1).unwrap(), 1);
            assert_eq!(
                backend
                    .effective_recoupling_threads(engine, 1 << 26)
                    .unwrap(),
                4
            );
        });
        // A serial engine never forks.
        assert_eq!(
            backend
                .effective_recoupling_threads(&mut symtn_dense::CpuEngine::serial(), 1 << 26)
                .unwrap(),
            1
        );
    }

    #[test]
    fn provided_overwrite_keeps_existing_backend_implementations_source_compatible() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (dst_structure, src_structure, structure) = overwrite_fixture();
        let mut dst = [f64::NAN; 2];

        RequiredMethodsOnlyBackend
            .tree_transform_structure_overwrite_into_raw(
                engine,
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[3.0],
                1.0,
                &[],
            )
            .unwrap();

        assert_eq!(dst, [6.0, 0.0]);
    }

    #[test]
    fn provided_overwrite_scales_listed_blocks_as_a_folding_backend_does() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: the provided overwrite's explicit scale pass gives the same
        // destination as the structural backend folding θ_b into its moves,
        // for a Single and both destinations of a Multi block; an unsorted
        // list is rejected before the destination is touched.
        let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 3]).unwrap());
        let dst = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 4]).unwrap());
        let structure = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[
                crate::TreeTransformBlockSpec::single(0, 0, 2.0).with_source_axes([1, 0]),
                crate::TreeTransformBlockSpec::multi(
                    vec![1, 2],
                    vec![1, 2],
                    vec![1.0, 2.0, 3.0, 4.0],
                ),
            ],
        )
        .unwrap();
        let source: Vec<f64> = (0..12).map(|index| 0.5 * index as f64 + 1.0).collect();
        let scales = [(0, -1.0), (4, -1.0), (8, -1.0)];
        let mut run = |provided: bool, scales: &[(usize, f64)]| {
            let mut out = vec![f64::NAN; 16];
            let mut workspace = TreeTransformWorkspace::default();
            if provided {
                RequiredMethodsOnlyBackend.tree_transform_structure_overwrite_into_raw(
                    engine,
                    &mut workspace,
                    &structure,
                    &dst,
                    &src,
                    &mut out,
                    &source,
                    1.0,
                    scales,
                )
            } else {
                DenseTreeTransformOperations::default_executor()
                    .tree_transform_structure_overwrite_into_raw(
                        engine,
                        &mut workspace,
                        &structure,
                        &dst,
                        &src,
                        &mut out,
                        &source,
                        1.0,
                        scales,
                    )
            }
            .unwrap();
            out
        };
        let expected = run(false, &scales);
        assert_ne!(expected[..12], run(false, &[])[..12], "vacuous scales");
        assert_eq!(run(true, &scales), expected);

        let mut untouched = [5.0; 16];
        let error = RequiredMethodsOnlyBackend
            .tree_transform_structure_overwrite_into_raw(
                engine,
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst,
                &src,
                &mut untouched,
                &source,
                1.0,
                &[(4, -1.0), (0, -1.0)],
            )
            .unwrap_err();
        assert!(
            matches!(error, OperationError::InvalidArgument { message } if message.contains("strictly increasing"))
        );
        assert_eq!(untouched, [5.0; 16]);
    }

    #[test]
    fn provided_profiled_methods_keep_required_only_backends_correct() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (dst_structure, src_structure, structure) = overwrite_fixture();
        let mut workspace = TreeTransformWorkspace::default();
        let mut backend = RequiredMethodsOnlyBackend;
        let mut generic_dst = [10.0, 20.0];
        let mut generic_profile = TreeTransformReplayProfile::default();
        backend
            .tree_transform_structure_into_raw_profiled(
                engine,
                &mut workspace,
                &structure,
                &dst_structure,
                &src_structure,
                &mut generic_dst,
                &[3.0],
                1.0,
                0.5,
                &mut generic_profile,
            )
            .unwrap();
        assert_eq!(generic_dst, [11.0, 10.0]);

        let mut overwrite_dst = [f64::NAN; 2];
        let mut overwrite_profile = TreeTransformReplayProfile::default();
        backend
            .tree_transform_structure_overwrite_into_raw_profiled(
                engine,
                &mut workspace,
                &structure,
                &dst_structure,
                &src_structure,
                &mut overwrite_dst,
                &[3.0],
                1.0,
                &[],
                &mut overwrite_profile,
            )
            .unwrap();
        assert_eq!(overwrite_dst, [6.0, 0.0]);
    }

    #[test]
    fn builtin_overwrite_backends_execute_raw_and_profiled_one_pass_routes() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (dst_structure, src_structure, structure) = overwrite_fixture();
        let mut workspace = TreeTransformWorkspace::default();

        let mut dense = DenseTreeTransformOperations::default();
        let mut dense_dst = [f64::NAN; 2];
        dense
            .tree_transform_structure_overwrite_into_raw(
                engine,
                &mut workspace,
                &structure,
                &dst_structure,
                &src_structure,
                &mut dense_dst,
                &[3.0],
                1.0,
                &[],
            )
            .unwrap();
        assert_eq!(dense_dst, [6.0, 0.0]);

        dense_dst.fill(f64::NAN);
        let mut profile = TreeTransformReplayProfile::default();
        dense
            .tree_transform_structure_overwrite_into_raw_profiled(
                engine,
                &mut workspace,
                &structure,
                &dst_structure,
                &src_structure,
                &mut dense_dst,
                &[3.0],
                1.0,
                &[],
                &mut profile,
            )
            .unwrap();
        assert_eq!(dense_dst, [6.0, 0.0]);
        assert_eq!(profile.single_blocks, 1);
    }
}
