use super::*;

/// Flat [`CudaScalar`] buffer owned by one CUDA engine's allocation domain.
///
/// The handle itself is dtype-erased; every typed access names the payload
/// dtype and reports a mismatch as [`DenseError::DTypeMismatch`].
pub struct CudaDenseStorage {
    pub(super) tensor: Tensor,
    // Fixed by the typed constructor, so `dtype()` never re-maps Tenferro's
    // dtype, which since 0.6.0 includes the unmappable `DType::External`.
    pub(super) dtype: DenseDType,
    pub(super) len: usize,
    pub(super) device: CudaPlacement,
}
impl CudaDenseStorage {
    /// Uploads borrowed host data as a flat device buffer.
    ///
    /// Tenferro uploads an owned host tensor (`upload_tensor`, tenferro-gpu
    /// `cubecl/memory.rs:31`), so borrowed data costs exactly one host copy
    /// here: the payload is duplicated into the host tensor that is then
    /// staged to the device. That copy is required by the backend contract,
    /// not by symtn. A caller that already owns its buffer — a zero buffer, a
    /// selector, a coefficient or diagonal vector — uses
    /// [`Self::upload_owned`] instead and pays none.
    pub fn upload<D: CudaScalar>(ctx: &CudaDenseContext, data: &[D]) -> Result<Self, DenseError> {
        Self::upload_owned(ctx, data.to_vec())
    }

    /// Uploads owned host data as a flat device buffer, moving `data` into the
    /// host tensor Tenferro uploads instead of copying it.
    ///
    /// Counting, device traffic and the resulting buffer are identical to
    /// [`Self::upload`]; only the redundant host copy is gone.
    pub fn upload_owned<D: CudaScalar>(
        ctx: &CudaDenseContext,
        data: Vec<D>,
    ) -> Result<Self, DenseError> {
        let len = data.len();
        Self::upload_shaped(ctx, data, vec![len])
    }

    /// [`Self::upload_owned`] for a stack of `members` members of
    /// `member_len` elements each, stored member-major: the device tensor is
    /// `[member_len, members]` (column-major, the same bytes as flat), so a
    /// gather can address the member axis ([`cuda_gather_member_elements`]).
    /// Counting and traffic are those of [`Self::upload_owned`].
    #[doc(hidden)]
    pub fn upload_members<D: CudaScalar>(
        ctx: &CudaDenseContext,
        data: Vec<D>,
        member_len: usize,
        members: usize,
    ) -> Result<Self, DenseError> {
        if member_len.checked_mul(members) != Some(data.len()) {
            return Err(cuda_error(
                "cuda_upload",
                "a stack upload must hold exactly `members * member_len` elements",
            ));
        }
        Self::upload_shaped(ctx, data, vec![member_len, members])
    }

    fn upload_shaped<D: CudaScalar>(
        ctx: &CudaDenseContext,
        data: Vec<D>,
        shape: Vec<usize>,
    ) -> Result<Self, DenseError> {
        let len = data.len();
        let bytes = std::mem::size_of_val(data.as_slice());
        let host = Tensor::from_vec_col_major(shape, data)
            .map_err(|err| cuda_error("cuda_upload", err))?;
        let tensor =
            upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error("cuda_upload", err))?;
        record_h2d(bytes);
        Ok(Self {
            tensor,
            dtype: D::DTYPE,
            len,
            device: ctx.device,
        })
    }

    /// Downloads the flat device buffer back to host data.
    ///
    /// The returned vector is the one Tenferro's download produced: the host
    /// tensor is consumed (`TypedTensor::into_host_vec`, tenferro-tensor
    /// `types.rs:7248`) rather than copied out of again.
    pub fn download<D: CudaScalar>(&self, ctx: &CudaDenseContext) -> Result<Vec<D>, DenseError> {
        ensure_cuda_device(ctx.device, "cuda_download", &[("source", self.device)])?;
        let host = download_tensor(&ctx.runtime, &self.tensor)
            .map_err(|err| cuda_error("cuda_download", err))?;
        if host.dtype() != D::dtype() {
            return Err(dtype_mismatch::<D>("cuda_download", &host));
        }
        let typed = D::into_typed(host).map_err(|err| cuda_error("cuda_download", err))?;
        let mut data = typed
            .into_host_vec()
            .map_err(|err| cuda_error("cuda_download", err))?;
        let bytes = std::mem::size_of_val(data.as_slice());
        // A narrowed buffer (`set_active_len`) still transfers its whole
        // allocation; only the active prefix is the value.
        data.truncate(self.len);
        #[cfg(test)]
        crate::probe::add(crate::probe::Event::CudaFullDownloadBytes, bytes);
        record_d2h(bytes);
        Ok(data)
    }

    /// The payload dtype this device buffer owns.
    pub fn dtype(&self) -> DenseDType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The CUDA device ordinal of the engine that owns this buffer.
    pub fn device(&self) -> usize {
        self.device.ordinal
    }

    /// Elements the device allocation holds; at least [`Self::len`].
    #[doc(hidden)]
    pub fn capacity(&self) -> usize {
        self.tensor.shape().iter().product()
    }

    /// Sets the active prefix [`Self::len`] reports, and every region bound
    /// is checked against, to `len` elements of the allocation.
    ///
    /// Why: a grow-only device scratch reused across operands of different
    /// sizes must present exactly the length each consumer admits without a
    /// reallocation (and without the zero upload a reallocation costs, #740).
    /// Only the bound narrows; the allocation and its contents are unchanged.
    #[doc(hidden)]
    pub fn set_active_len(&mut self, len: usize) -> Result<(), DenseError> {
        if len > self.capacity() {
            return Err(DenseError::OutOfBounds);
        }
        self.len = len;
        Ok(())
    }

    /// Bounds a matrix view by the active length rather than the
    /// allocation: after [`Self::set_active_len`] the two differ, and the
    /// backend view itself only checks the allocation.
    pub(super) fn check_matrix_bound(
        &self,
        shape: [usize; 2],
        strides: [usize; 2],
        offset: usize,
    ) -> Result<(), DenseError> {
        if shape.contains(&0) {
            return Ok(());
        }
        let last = shape
            .iter()
            .zip(strides)
            .try_fold(offset, |end, (&dim, stride)| {
                (dim - 1).checked_mul(stride)?.checked_add(end)
            })
            .ok_or(DenseError::ElementCountOverflow)?;
        if last >= self.len {
            return Err(DenseError::OutOfBounds);
        }
        Ok(())
    }

    /// Wraps a device tensor produced by a tenferro op (e.g. a cuSOLVER
    /// factor) as flat storage, after proving it carries `D`'s payload — the
    /// dtype recorded here.
    pub(super) fn from_tensor<D: CudaScalar>(
        op: &'static str,
        tensor: Tensor,
        device: CudaPlacement,
    ) -> Result<Self, DenseError> {
        if D::typed(&tensor).is_none() {
            return Err(dtype_mismatch::<D>(op, &tensor));
        }
        record(|stats| stats.device_allocs += 1);
        let len = tensor.shape().iter().product();
        Ok(Self {
            tensor,
            dtype: D::DTYPE,
            len,
            device,
        })
    }

    /// Column-major matrix view over a buffer region with an explicit
    /// leading dimension (`ld >= rows`, `ld == rows` for a packed region).
    pub(super) fn region_view<D: CudaScalar>(
        &self,
        rows: usize,
        cols: usize,
        ld: usize,
        offset: usize,
    ) -> Result<TensorView<'_>, DenseError> {
        self.region_view_strided::<D>([rows, cols], [1, ld], offset)
    }

    pub(super) fn region_view_strided<D: CudaScalar>(
        &self,
        shape: [usize; 2],
        strides: [usize; 2],
        offset: usize,
    ) -> Result<TensorView<'_>, DenseError> {
        self.check_matrix_bound(shape, strides, offset)?;
        let Some(tensor) = D::typed(&self.tensor) else {
            return Err(dtype_mismatch::<D>("cuda_region", &self.tensor));
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_region", "offset does not fit in isize"))?;
        let strides = strides
            .map(isize::try_from)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| cuda_error("cuda_region", "stride does not fit in isize"))?;
        tensor
            .backend_region_view(shape.to_vec(), strides, offset)
            .map(D::tensor_view)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    /// Rank-N counterpart of [`Self::region_view_strided`]. The caller owns
    /// the bounds and dtype proof only in the sense that both are re-checked
    /// here (dtype) and by [`validate_region`] (bounds) before any submission.
    pub(super) fn region_view_nd<D: CudaScalar>(
        &self,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<TensorView<'_>, DenseError> {
        let Some(tensor) = D::typed(&self.tensor) else {
            return Err(dtype_mismatch::<D>("cuda_region", &self.tensor));
        };
        tensor
            .backend_region_view(dims.to_vec(), strides.to_vec(), offset)
            .map(D::tensor_view)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    pub(super) fn region_view_nd_mut<D: CudaScalar>(
        &mut self,
        dims: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> Result<TensorViewMut<'_>, DenseError> {
        let actual = self.dtype;
        let Some(tensor) = D::typed_mut(&mut self.tensor) else {
            return Err(DenseError::DTypeMismatch {
                op: "cuda_region",
                expected: D::DTYPE,
                actual,
            });
        };
        tensor
            .backend_region_view_mut(dims.to_vec(), strides.to_vec(), offset)
            .map(D::tensor_view_mut)
            .map_err(|err| cuda_error("cuda_region", err))
    }

    pub(super) fn region_view_mut<D: CudaScalar>(
        &mut self,
        rows: usize,
        cols: usize,
        ld: usize,
        offset: usize,
    ) -> Result<TensorViewMut<'_>, DenseError> {
        self.check_matrix_bound([rows, cols], [1, ld], offset)?;
        let actual = self.dtype;
        let Some(tensor) = D::typed_mut(&mut self.tensor) else {
            return Err(DenseError::DTypeMismatch {
                op: "cuda_region",
                expected: D::DTYPE,
                actual,
            });
        };
        let offset = isize::try_from(offset)
            .map_err(|_| cuda_error("cuda_region", "offset does not fit in isize"))?;
        let ld_isize = isize::try_from(ld)
            .map_err(|_| cuda_error("cuda_region", "leading dimension does not fit in isize"))?;
        tensor
            .backend_region_view_mut(vec![rows, cols], vec![1, ld_isize], offset)
            .map(D::tensor_view_mut)
            .map_err(|err| cuda_error("cuda_region", err))
    }
}
