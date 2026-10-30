use super::*;

/// One kernel call as seen by [`ObservedKernels`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelCall {
    Add,
    Axpby,
    CopyScale { dst_offset: isize },
    Scale { dst_offset: isize },
    Recoupling,
}

/// The one test adapter: forwards every primitive to
/// [`StridedHostKernelAdapter`] after recording it in `calls` and reporting
/// it to `observe`.
///
/// `transform_strided_baked` is deliberately not forwarded, so replay
/// reaches the observer through the trait's promote-and-forward default.
#[derive(Clone)]
pub(crate) struct ObservedKernels<F = fn(KernelCall)> {
    inner: StridedHostKernelAdapter,
    observe: F,
    pub(crate) calls: Vec<KernelCall>,
}

impl ObservedKernels {
    pub(crate) fn recording() -> Self {
        Self::new(|_| {})
    }
}

impl<F: FnMut(KernelCall)> ObservedKernels<F> {
    pub(crate) fn new(observe: F) -> Self {
        Self {
            inner: StridedHostKernelAdapter::default(),
            observe,
            calls: Vec::new(),
        }
    }

    fn record(&mut self, call: KernelCall) {
        self.calls.push(call);
        (self.observe)(call);
    }

    pub(crate) fn count(&self, call: KernelCall) -> usize {
        self.calls.iter().filter(|&&seen| seen == call).count()
    }

    pub(crate) fn copy_offsets(&self) -> Vec<isize> {
        self.calls
            .iter()
            .filter_map(|call| match call {
                KernelCall::CopyScale { dst_offset } => Some(*dst_offset),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn scale_offsets(&self) -> Vec<isize> {
        self.calls
            .iter()
            .filter_map(|call| match call {
                KernelCall::Scale { dst_offset } => Some(*dst_offset),
                _ => None,
            })
            .collect()
    }
}

impl<F> super::adapter::sealed::Sealed for ObservedKernels<F> {}

impl<T, F> HostKernelAdapter<T> for ObservedKernels<F>
where
    StridedHostKernelAdapter: HostKernelAdapter<T>,
    F: FnMut(KernelCall),
{
    fn add_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        self.record(KernelCall::Add);
        self.inner.add_strided_baked(
            exec,
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
            baked,
            index,
        )
    }

    fn axpby_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        self.record(KernelCall::Axpby);
        self.inner.axpby_strided_baked(
            exec,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
            baked,
            index,
        )
    }

    fn copy_scale_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        self.record(KernelCall::CopyScale { dst_offset });
        self.inner.copy_scale_strided_baked(
            exec,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            baked,
            index,
        )
    }

    fn scale_strided(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError> {
        self.record(KernelCall::Scale { dst_offset });
        self.inner
            .scale_strided(exec, dst_data, shape, dst_strides, dst_offset, beta)
    }

    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>,
    {
        self.record(KernelCall::Recoupling);
        self.inner.recoupling_src_times_u_transpose(
            exec,
            destination,
            source,
            recoupling_coefficients_dst_src,
            element_count,
            src_count,
            dst_count,
        )
    }
}
