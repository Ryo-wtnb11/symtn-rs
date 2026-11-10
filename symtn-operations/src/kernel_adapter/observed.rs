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
    /// With [`Self::addressing`], every element a write primitive writes:
    /// its absolute address, in elements (the buffer address over the
    /// element size, plus its index there). Clones share it, so the kernel
    /// clones of a split replay record into one list.
    pub(crate) writes: Option<std::sync::Arc<std::sync::Mutex<Vec<usize>>>>,
    /// With [`Self::failing_at`], the absolute element address whose write
    /// fails: the primitive that would write it returns an error instead.
    fail_at: Option<usize>,
}

impl ObservedKernels {
    pub(crate) fn recording() -> Self {
        Self::new(|_| {})
    }

    /// [`Self::recording`] that also records every written element.
    pub(crate) fn addressing() -> Self {
        Self {
            writes: Some(Default::default()),
            ..Self::recording()
        }
    }

    /// [`Self::addressing`] whose write primitive that would write the
    /// element at absolute address `at` (in elements) fails before writing.
    pub(crate) fn failing_at(at: usize) -> Self {
        Self {
            fail_at: Some(at),
            ..Self::addressing()
        }
    }
}

impl<F: FnMut(KernelCall)> ObservedKernels<F> {
    pub(crate) fn new(observe: F) -> Self {
        Self {
            inner: StridedHostKernelAdapter::default(),
            observe,
            calls: Vec::new(),
            writes: None,
            fail_at: None,
        }
    }

    fn record(&mut self, call: KernelCall) {
        self.calls.push(call);
        (self.observe)(call);
    }

    /// Records the elements `shape` / `strides` from `offset` address in
    /// `dst`, when addressing; fails when one is the failing address.
    fn record_writes<T>(
        &mut self,
        dst: &[T],
        shape: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<(), OperationError> {
        let Some(writes) = self.writes.as_ref() else {
            return Ok(());
        };
        let mut at = vec![offset];
        for (&extent, &stride) in shape.iter().zip(strides) {
            at = at
                .iter()
                .flat_map(|&base| (0..extent as isize).map(move |i| base + i * stride))
                .collect();
        }
        let base = dst.as_ptr() as usize / core::mem::size_of::<T>().max(1);
        let at = at
            .into_iter()
            .map(|index| base.wrapping_add_signed(index))
            .collect::<Vec<_>>();
        if self.fail_at.is_some_and(|fail| at.contains(&fail)) {
            return Err(OperationError::InvalidArgument {
                message: "observed kernels injected a movement failure",
            });
        }
        writes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(at);
        Ok(())
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
        self.record_writes(dst_data, shape, dst_strides, dst_offset)?;
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
        self.record_writes(dst_data, shape, dst_strides, dst_offset)?;
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
        self.record_writes(dst_data, shape, dst_strides, dst_offset)?;
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
        self.record_writes(dst_data, shape, dst_strides, dst_offset)?;
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
