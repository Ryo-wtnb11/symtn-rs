use std::num::NonZeroUsize;
use std::sync::Arc;

use symtn_core::{BlockStructure, Placement};
use symtn_dense::{DefaultDenseExecutor, DenseExecutor};

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

    /// Worker count this backend is configured to run transforms with.
    /// The execution context mirrors it into the plan-compile cache, so the
    /// one configured knob drives both replay and compile parallelism;
    /// backends without a thread setting stay serial.
    fn recoupling_threads(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the raw backend contract keeps workspace, replay structures, buffers, and alpha/beta explicit"
    )]
    fn tree_transform_structure_into_raw(
        &mut self,
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
        crate::transform_replay::zero_tree_transform_destination(
            &mut kernels,
            &mut zero_strides,
            dst_structure,
            dst_data,
        )?;
        self.tree_transform_structure_into_raw(
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            D::one(),
        )?;
        scale_destination_blocks(&mut kernels, dst_structure, dst_data, destination_scales)
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_transform_structure_into_raw_profiled(
        &mut self,
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

/// Default minimum destination length before a `recoupling_threads > 1`
/// setting actually goes parallel; below it the replay stays serial.
///
/// Mirrors TensorKit's gate `length(t.data) > Strided.MINTHREADLENGTH`
/// (Strided.jl `MINTHREADLENGTH = 1 << 15`, the same value strided-kernel
/// uses internally but does not export). Configurable per backend via
/// [`DenseTreeTransformOperations::set_transform_parallel_min_len`].
pub const TRANSFORM_PARALLEL_MIN_LEN: usize = 1 << 15;

#[derive(Debug)]
pub struct DenseTreeTransformOperations<E = DefaultDenseExecutor> {
    dense: E,
    // Replay parallelism is a property of this backend: worker count for the
    // tree-transform replay phases (1 = serial, the default).
    recoupling_threads: NonZeroUsize,
    // Size gate paired with recoupling_threads; see TRANSFORM_PARALLEL_MIN_LEN.
    transform_parallel_min_len: usize,
}

impl DenseTreeTransformOperations<DefaultDenseExecutor> {
    pub fn default_executor() -> Self {
        Self::new(DefaultDenseExecutor::new())
    }

    pub fn with_threads(threads: usize) -> Result<Self, OperationError> {
        Ok(Self::new(
            DefaultDenseExecutor::with_threads(threads).map_err(OperationError::Dense)?,
        ))
    }
}

impl<E> DenseTreeTransformOperations<E> {
    pub fn new(dense: E) -> Self {
        Self {
            dense,
            recoupling_threads: NonZeroUsize::MIN,
            transform_parallel_min_len: TRANSFORM_PARALLEL_MIN_LEN,
        }
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

    /// Worker count for tree-transform replays (default 1 = serial).
    #[inline]
    pub fn recoupling_threads(&self) -> NonZeroUsize {
        self.recoupling_threads
    }

    /// Sets the tree-transform replay worker count (1 = serial).
    /// Values `> 1` parallelize replays whose destination length
    /// exceeds [`Self::transform_parallel_min_len`].
    pub fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
        self.recoupling_threads = threads;
    }

    #[inline]
    fn kernel_adapter(&self) -> StridedHostKernelAdapter {
        StridedHostKernelAdapter::default()
    }

    /// Minimum destination length before `recoupling_threads > 1` goes
    /// parallel (default [`TRANSFORM_PARALLEL_MIN_LEN`]).
    #[inline]
    pub fn transform_parallel_min_len(&self) -> usize {
        self.transform_parallel_min_len
    }

    pub fn set_transform_parallel_min_len(&mut self, min_len: usize) {
        self.transform_parallel_min_len = min_len;
    }

    /// Effective worker count for one replay: the configured count when
    /// parallelism is enabled and the destination is past the size gate,
    /// otherwise 1 (the untouched serial path).
    #[inline]
    fn effective_recoupling_threads(&self, dst_len: usize) -> usize {
        if self.recoupling_threads.get() > 1 && dst_len > self.transform_parallel_min_len {
            self.recoupling_threads.get()
        } else {
            1
        }
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
        let threads = dst_structure
            .required_len()
            .map_or(1, |len| self.effective_recoupling_threads(len));
        crate::try_tree_transform_structure_overwrite_owned_raw(
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

    #[inline]
    fn recoupling_threads(&self) -> NonZeroUsize {
        DenseTreeTransformOperations::recoupling_threads(self)
    }

    fn tree_transform_structure_into_raw(
        &mut self,
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(dst_data.len());
        crate::tree_transform_structure_with_structural_recoupling_raw(
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
        workspace: &mut Self::Workspace,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError> {
        let threads = self.effective_recoupling_threads(dst_data.len());
        crate::tree_transform_structure_overwrite_with_structural_recoupling_raw(
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
        let threads = self.effective_recoupling_threads(dst_data.len());
        crate::tree_transform_structure_with_structural_recoupling_raw_profiled(
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
        let threads = self.effective_recoupling_threads(dst_data.len());
        crate::tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
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
    fn transform_parallel_gate_is_strictly_past_the_threshold() {
        let mut backend = DenseTreeTransformOperations::default();
        backend.set_recoupling_threads(NonZeroUsize::new(4).unwrap());
        backend.set_transform_parallel_min_len(8);

        assert_eq!(backend.effective_recoupling_threads(8), 1);
        assert_eq!(backend.effective_recoupling_threads(9), 4);
    }

    #[test]
    fn provided_overwrite_keeps_existing_backend_implementations_source_compatible() {
        let (dst_structure, src_structure, structure) = overwrite_fixture();
        let mut dst = [f64::NAN; 2];

        RequiredMethodsOnlyBackend
            .tree_transform_structure_overwrite_into_raw(
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
        let run = |provided: bool, scales: &[(usize, f64)]| {
            let mut out = vec![f64::NAN; 16];
            let mut workspace = TreeTransformWorkspace::default();
            if provided {
                RequiredMethodsOnlyBackend.tree_transform_structure_overwrite_into_raw(
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
        let (dst_structure, src_structure, structure) = overwrite_fixture();
        let mut workspace = TreeTransformWorkspace::default();
        let mut backend = RequiredMethodsOnlyBackend;
        let mut generic_dst = [10.0, 20.0];
        let mut generic_profile = TreeTransformReplayProfile::default();
        backend
            .tree_transform_structure_into_raw_profiled(
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
        let (dst_structure, src_structure, structure) = overwrite_fixture();
        let mut workspace = TreeTransformWorkspace::default();

        let mut dense = DenseTreeTransformOperations::default();
        let mut dense_dst = [f64::NAN; 2];
        dense
            .tree_transform_structure_overwrite_into_raw(
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
