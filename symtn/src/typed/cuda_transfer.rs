use super::*;

impl<R, D: CudaPayload> TensorMap<R, D> {
    /// Uploads host ownership of a device-capable payload into the caller's
    /// CUDA engine. A non-CUDA engine is refused before any upload.
    ///
    /// Dense storage uploads directly. Compact diagonal storage is expanded
    /// operation-locally and becomes dense on device; a roundtrip remains
    /// dense rather than recovering compactness. A lazy adjoint transfers only its canonical
    /// parent and rebuilds a lazy view over the device parent.
    ///
    /// Every payload of the base family uploads: `f64`, `Complex64`, `f32`
    /// and `Complex32`. Single precision moves exactly half the bytes of its
    /// double-precision twin for the same fixture, in the same number of
    /// transfer and allocation calls. What single precision does *not* open is
    /// the device factorization family — see [`CudaFactorizationPayload`].
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn c32_upload(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, Complex32>) {
    ///     let _ = tensor.to_cuda(engine);
    /// }
    /// ```
    ///
    /// ```
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn f32_upload(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, f32>) {
    ///     let _ = tensor.to_cuda(engine);
    /// }
    /// ```
    ///
    /// A rule with complex categorical coefficients has no device path. The
    /// payload itself uploads (storage only), but every device operation
    /// needs `Scalar = f64` symbols, so using it is a compile-time boundary
    /// rather than a runtime error:
    ///
    /// ```compile_fail
    /// use num_complex::Complex64;
    /// use symtn::sector::TableRule;
    /// use symtn::typed::{Error, TensorMap};
    ///
    /// fn no_table_device_svd(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<TableRule, Complex64>) -> Result<(), Error> {
    ///     let device = tensor.to_cuda(engine)?;
    ///     let _ = device.svd_compact(engine, &[0], &[1]);
    ///     Ok(())
    /// }
    /// ```
    ///
    /// The Host `twist` of such a rule exists, but the device twist keeps
    /// its `Scalar = f64` bound:
    ///
    /// ```compile_fail
    /// use num_complex::Complex64;
    /// use symtn::sector::TableRule;
    /// use symtn::typed::{Direction, Error, TensorMap};
    ///
    /// fn no_table_device_twist(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<TableRule, Complex64>) -> Result<(), Error> {
    ///     let device = tensor.to_cuda(engine)?;
    ///     let _ = device.twist(engine, &[0], Direction::Forward);
    ///     Ok(())
    /// }
    /// ```
    pub fn to_cuda(
        &self,
        engine: &mut dyn Engine,
    ) -> Result<TensorMap<R, D, CudaStorage<D>>, Error> {
        let mut lease = crate::runtime::cuda_call(engine)?;
        let cuda = &mut *lease;
        let upload = |body: &Arc<TypedTensorBody<R, D>>| {
            let storage = match body.data.as_ref() {
                TypedData::Dense(data) => CudaStorage::upload(cuda, data)?,
                TypedData::Diagonal(spectrum) => {
                    let dense = symtn_matrixalgebra::seam::diagonal_bond_data(
                        body.space.space(),
                        spectrum,
                        &|value| value,
                    )?;
                    CudaStorage::upload_owned(cuda, dense)?
                }
            };
            Ok::<_, Error>(Arc::new(TypedTensorBody::dense(
                body.space.clone(),
                storage,
            )))
        };

        let repr = match &self.repr {
            TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(upload(body)?),
            TypedTensorRepr::Adjoint(view) => TypedTensorRepr::Adjoint(Arc::new(
                TypedAdjointView::new(upload(&view.parent)?, view.logical_space.clone()),
            )),
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr,
        })
    }
}

impl<R, D: CudaPayload> TensorMap<R, D, CudaStorage<D>> {
    /// Downloads device ownership into one final dense host buffer, through
    /// the caller's CUDA engine that owns the payload.
    ///
    /// A lazy adjoint downloads only its canonical parent and rebuilds a cold
    /// host lazy view. No receiver-sized logical adjoint is materialized.
    /// Device storage is never implicitly host-readable:
    ///
    /// ```compile_fail
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::{CudaStorage, TensorMap};
    ///
    /// fn no_device_slice(tensor: &TensorMap<U1FusionRule, f64, CudaStorage>) {
    ///     let _ = tensor.dense_data();
    /// }
    /// ```
    pub fn to_host(&self, engine: &mut dyn Engine) -> Result<TensorMap<R, D>, Error> {
        let mut lease = crate::runtime::cuda_call(engine)?;
        let cuda = &mut *lease;
        let download = |body: &Arc<TypedTensorBody<R, D, CudaStorage<D>>>| {
            let TypedData::Dense(storage) = body.data.as_ref() else {
                unreachable!("typed CUDA transfer never produces compact storage")
            };
            let data = storage.download(cuda)?;
            Ok::<_, Error>(Arc::new(TypedTensorBody::dense(body.space.clone(), data)))
        };

        let repr = match &self.repr {
            TypedTensorRepr::Owned(body) => TypedTensorRepr::Owned(download(body)?),
            TypedTensorRepr::Adjoint(view) => TypedTensorRepr::Adjoint(Arc::new(
                TypedAdjointView::new(download(&view.parent)?, view.logical_space.clone()),
            )),
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr,
        })
    }
}

impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    pub(super) fn direct_cuda_storage(
        &self,
        operation: &'static str,
    ) -> Result<&CudaStorage<D>, Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(storage) => Ok(storage),
                TypedData::Diagonal(_) => Err(Error::UnsupportedOnDevice(format!(
                    "{operation} requires dense CUDA storage"
                ))),
            },
            TypedTensorRepr::Adjoint(_) => Err(Error::UnsupportedOnDevice(format!(
                "{operation} does not support lazy adjoint CUDA operands"
            ))),
        }
    }

    pub(super) fn validate_cuda_owned_metadata(
        expected: Placement,
        actual: Placement,
        required_len: usize,
        actual_len: usize,
    ) -> Result<(), Error> {
        if actual != expected {
            return Err(Error::PlacementMismatch);
        }
        if actual_len != required_len {
            return Err(internal_layout_error(
                "CUDA payload length does not match its admitted tensor space",
            ));
        }
        Ok(())
    }

    /// An owned dense result on this operand's logical space.
    pub(super) fn with_owned_cuda_storage(&self, storage: CudaStorage<D>) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(
                self.logical_space().clone(),
                storage,
            )),
        }
    }
}

impl<R, D> MaterializeExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn materialize(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
        source: &Self,
        map: Option<&symtn_tensors::AdjointBlockMap<'_>>,
    ) -> Result<Self, Error> {
        let dense_err = |err| Error::from(symtn_tensors::OperationError::Dense(err));
        let required_len = t.logical_space().space().required_len()?;
        let adjoint = map
            .map(|map| cuda_adjoint_lowering(map, D::IS_COMPLEX))
            .transpose()?;

        let mut lease = crate::runtime::cuda_call(engine)?;
        let cuda = &mut *lease;
        if source.placement() != Placement::Cuda(cuda.device()) {
            return Err(Error::PlacementMismatch);
        }
        let output = match adjoint {
            None => source.gather_members(cuda, required_len, 1, &[0])?,
            Some(CudaAdjointLowering {
                copies,
                table,
                zeros,
            }) => {
                let mut output = if D::IS_COMPLEX {
                    source.gather_elements(cuda, &table)?.conj(cuda)?
                } else {
                    let mut output = source.gather_members(cuda, required_len, 1, &[0])?;
                    for (source_region, destination_region) in &copies {
                        symtn_dense::cuda_copy_strided_into::<D>(
                            cuda,
                            &source.0,
                            source_region,
                            &mut output.0,
                            destination_region,
                        )
                        .map_err(dense_err)?;
                    }
                    output
                };
                for zero in &zeros {
                    symtn_dense::cuda_region_zero::<D>(cuda, &mut output.0, zero)
                        .map_err(dense_err)?;
                }
                output
            }
        };
        drop(lease);
        Ok(output)
    }
}

/// The device lowering of an [`symtn_tensors::AdjointBlockMap`], on the Host
/// before the lease: the strided copies, or with `table` the element table,
/// that transpose the parent (the caller conjugates), and the regions of
/// blocks that read as zero. `materialize` takes the table for a complex
/// payload (one gather and one `conj`); the elementwise kernels take the
/// copies at every dtype and conjugate in the strided read itself (#1874).
pub(super) struct CudaAdjointLowering {
    /// (parent region, output region) per block, without `table`.
    pub(super) copies: Vec<(symtn_dense::CudaRegion, symtn_dense::CudaRegion)>,
    /// The parent element of each output element, with `table`.
    pub(super) table: Vec<usize>,
    pub(super) zeros: Vec<symtn_dense::CudaRegion>,
}

pub(super) fn cuda_adjoint_lowering(
    map: &symtn_tensors::AdjointBlockMap<'_>,
    table: bool,
) -> Result<CudaAdjointLowering, Error> {
    let region = |dims: Vec<usize>, strides: Vec<usize>, offset: usize| {
        symtn_dense::CudaRegion::new(dims, strides, offset)
            .map_err(|err| Error::from(symtn_tensors::OperationError::Dense(err)))
    };
    let space = map.adjoint_space();
    let required_len = space.required_len()?;
    if map.parent_space().required_len()? != required_len {
        return Err(internal_layout_error(
            "an adjoint layout has its parent's payload length",
        ));
    }
    let structure = space.structure();
    let mut copies = Vec::new();
    // Per block: (shape, destination strides and offset, source strides and
    // offset, or `None` for a block that reads as zero).
    let mut walks = Vec::new();
    let mut zeros = Vec::new();
    let mut covered = 0usize;
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        covered += shape.iter().product::<usize>();
        let destination = region(shape.to_vec(), block.strides().to_vec(), block.offset())?;
        let Some(source) = map.source(block)? else {
            zeros.push(destination);
            if table {
                walks.push((shape, block.strides(), block.offset(), None));
            }
            continue;
        };
        let source_strides = (0..shape.len())
            .map(|axis| source.stride(axis))
            .collect::<Result<Vec<_>, _>>()?;
        if table {
            walks.push((
                shape,
                block.strides(),
                block.offset(),
                Some((source_strides, source.offset())),
            ));
        } else {
            copies.push((
                region(shape.to_vec(), source_strides, source.offset())?,
                destination,
            ));
        }
    }
    if covered != required_len {
        return Err(internal_layout_error(
            "adjoint blocks do not tile the payload",
        ));
    }
    // The table maps each output element to its parent element. A canonical
    // layout lays blocks out compactly in block order, so the walk visits
    // output positions 0, 1, 2, ... and each entry is pushed once; any other
    // order falls back to an indexed fill.
    let mut elements = Vec::with_capacity(if table { required_len } else { 0 });
    let mut sequential = true;
    for (shape, strides, offset, source) in &walks {
        for_each_block_element(shape, strides, *offset, source, |dst, src| {
            sequential &= dst == elements.len();
            elements.push(src);
        });
    }
    if !sequential {
        // ponytail: non-canonical output layouts only; zero-filled then
        // overwritten, as tiling was checked above.
        elements = vec![0; required_len];
        for (shape, strides, offset, source) in &walks {
            for_each_block_element(shape, strides, *offset, source, |dst, src| {
                elements[dst] = src;
            });
        }
    }
    Ok(CudaAdjointLowering {
        copies,
        table: elements,
        zeros,
    })
}

/// Calls `visit(destination, source)` for every element of one block, first
/// axis fastest; a `None` source (a block that reads as zero) visits `0`.
fn for_each_block_element(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    source: &Option<(Vec<usize>, usize)>,
    mut visit: impl FnMut(usize, usize),
) {
    let mut position = vec![0usize; shape.len()];
    let at = |position: &[usize], strides: &[usize], base: usize| {
        base + position
            .iter()
            .zip(strides)
            .map(|(i, stride)| i * stride)
            .sum::<usize>()
    };
    for _ in 0..shape.iter().product::<usize>() {
        let src = source
            .as_ref()
            .map_or(0, |(strides, base)| at(&position, strides, *base));
        visit(at(&position, strides, offset), src);
        for (axis, extent) in shape.iter().enumerate() {
            position[axis] += 1;
            if position[axis] < *extent {
                break;
            }
            position[axis] = 0;
        }
    }
}
