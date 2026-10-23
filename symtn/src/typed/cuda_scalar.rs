use super::*;

/// Device transfer, arithmetic, reductions, and contraction over a
/// [`CudaStorage`] payload.
///
/// The payload dtype is the only degree of freedom: structural coefficients
/// stay real (`R::Scalar = f64`), and operand conjugation is carried as a GEMM
/// flag, never as a materialized conjugated buffer. All four payload dtypes are
/// admitted (#1336); see [`CudaPayload`].
///
/// Checked Generic providers deliberately have no device execution methods:
///
/// ```compile_fail
/// use symtn::sector::{CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use symtn::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_operations<R>(
///     lhs: &TensorMap<R, f64, CudaStorage>,
///     rhs: &TensorMap<R, f64, CudaStorage>,
/// ) where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = lhs.norm(2.0);
///     let _ = lhs.inner(rhs);
///     let _ = lhs.scale(2.0);
///     let _ = lhs.axpby(2.0, rhs, -3.0);
///     let _ = lhs.zeros_like();
/// }
/// ```
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload,
{
    fn cuda_zeros_owned(
        &self,
        required_len: usize,
        source: &CudaStorage<D>,
    ) -> Result<CudaStorage<D>, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        Self::validate_cuda_owned_metadata(
            Placement::Cuda(cuda.device()),
            source.placement(),
            required_len,
            source.len(),
        )?;
        let output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        #[cfg(test)]
        probe::hit(probe::Event::CudaZeroUpload);
        Ok(output)
    }

    /// Quantum-dimension-weighted Frobenius reduction over owned CUDA storage,
    /// **conjugate-linear in `lhs`** like the Host `coupled_region_inner`,
    /// returned in the wide `f64` lane before any narrowing to `D`.
    ///
    /// The conjugation is the GEMM operand flag `MatrixOp::Adjoint`: the left
    /// row-vector view of a coupled region has extent 1 along its first axis,
    /// so the adjoint strides `[len, 1]` address exactly the same elements as
    /// the identity strides `[1, 1]`, and the orientation only sets
    /// `lhs_conj`. For `D = f64` that flag is a no-op, so the reduction has
    /// the same operands and produces the same values as the unconjugated
    /// form it replaces.
    ///
    /// The device returns one scalar per coupled sector. Category weights stay
    /// with the tensor and are applied only after releasing the Runtime lock.
    ///
    /// # Accumulation contract
    ///
    /// Like the Host `coupled_region_inner`, which widens every entry, the sum
    /// accumulates in `f64` within *and* across coupled sectors at every
    /// payload dtype, so an `f32`/`Complex32` result is finite wherever the
    /// Host's is (#1344 for `norm`, #1383 for `inner`).
    ///
    /// * **Within one coupled sector** the sum runs on the device inside the
    ///   backend GEMM, in the dtype of its operands. Tenferro 0.7.1 offers no
    ///   widening dot or reduction: `dot_general` dispatches on the one dtype
    ///   shared by both operands and the destination (tenferro-gpu 0.7.1
    ///   `src/cubecl/gemm.rs` `dot_general_read_into_accum`), and
    ///   `DotGeneralAccumulation` carries no compute type; the BLAS-1
    ///   `vdot_read` is `cublasSdot_v2`/`cublasCdotc_v2` (`src/cubecl/blas1.rs`)
    ///   and the reduce kernels are generic over one float type
    ///   (`src/kernels/reduce/launch.rs` `launch_sum_float`). A
    ///   single-precision payload is therefore widened on the device first
    ///   ([`CudaStorage::widened`], Tenferro's `cast`; exact, since every
    ///   `f32` is an `f64`) and reduced in `f64`. That costs one device
    ///   allocation of twice the payload bytes and one elementwise pass per
    ///   *distinct* operand buffer — one for `norm` or `inner(a, a)`, two for
    ///   `inner(a, b)` — and no host transfer. `f64`/`Complex64` payloads take
    ///   the reduction directly and pay nothing extra.
    /// * **Across coupled sectors** the quantum-dimension weighting and the
    ///   final sum run on the host in `WideScalar::Wide`, the accumulator the
    ///   Host reductions use.
    ///
    /// Why not TensorKit's accumulation: TensorKit 0.17.1 `dot`
    /// (`src/tensors/linalg.jl`) sums `dim(c) * dot(b1, b2)` in the payload
    /// type, a `Float32` BLAS `sdot` per block, so it saturates exactly where
    /// the pre-#1383 device did. symtn's Host semantics are strictly wider,
    /// and the device must agree with its own Host.
    fn weighted_inner_cuda(
        &self,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
    ) -> Result<num_complex::Complex64, Error> {
        match D::DTYPE {
            symtn_dense::DenseDType::F32 => self.weighted_inner_widened_cuda::<f64>(lhs, rhs),
            symtn_dense::DenseDType::C32 => {
                self.weighted_inner_widened_cuda::<num_complex::Complex64>(lhs, rhs)
            }
            _ => Ok(self.weighted_inner_cuda_wide(lhs, rhs)?.widen_complex()),
        }
    }

    /// [`Self::weighted_inner_cuda`] for a single-precision payload: both
    /// operands widened to the double-precision lane `E` on the device, one
    /// cast when they are the same buffer.
    ///
    /// Why widen rather than rescale by `norm(Inf)` (#1344): the reference
    /// only needs *some* overflow-safe sum — LinearAlgebra's `generic_norm2`
    /// accumulates a `Float32` sum in `Float64` and rescales by `normInf`
    /// only when `length * maxabs^2` leaves the payload range — and the Host
    /// already meets it by accumulating every square in `f64`, where an `f32`
    /// payload can neither overflow nor underflow. Widening reproduces that
    /// Host arithmetic exactly in one extra device pass; a scaled norm would
    /// need a max-abs pass, a host round trip for the scale and a scaled copy
    /// before the same reduction, and would still round each square in `f32`.
    ///
    /// The selected reference dispatches are narrower. Julia routes a strided
    /// `BlasFloat` array of `length >= NRM2_CUTOFF` to `BLAS.nrm2`
    /// (`LinearAlgebra/src/dense.jl:107`), a scaled sum in the payload
    /// precision. TensorKit 0.17.1 `norm` (`src/tensors/linalg.jl:277`) takes
    /// `norm(t.data)` for `UniqueFusion` and otherwise `_norm` (`:261`), which
    /// adds `dim(c) * norm(b)^2` in `float(real(scalartype))` — `Float32` for
    /// a `Float32` tensor — so it returns `Inf` once one weighted block square
    /// exceeds `f32::MAX` (the "big" SU(2) fixture of
    /// `typed_cuda_single_precision`). The Host and this device norm
    /// accumulate every square and the cross-sector combine in `f64`, which is
    /// strictly wider than that dispatch. For `inner` a rescale is not even
    /// available: a cancelling sum has no scale that keeps every product in
    /// range and the result accurate.
    fn weighted_inner_widened_cuda<E: CudaPayload>(
        &self,
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
    ) -> Result<num_complex::Complex64, Error> {
        let mut lease = self.runtime.lease_cuda()?;
        let wide_lhs = lhs.widened::<E>(&mut lease)?;
        let wide_rhs = if std::ptr::eq(lhs, rhs) {
            None
        } else {
            Some(rhs.widened::<E>(&mut lease)?)
        };
        drop(lease);
        let wide_rhs = wide_rhs.as_ref().unwrap_or(&wide_lhs);
        Ok(self
            .weighted_inner_cuda_wide(&wide_lhs, wide_rhs)?
            .widen_complex())
    }

    /// [`Self::weighted_inner_cuda`] over operands already in the lane `E`: the
    /// cross-sector total in `E`'s wide accumulator.
    fn weighted_inner_cuda_wide<E: CudaPayload>(
        &self,
        lhs: &CudaStorage<E>,
        rhs: &CudaStorage<E>,
    ) -> Result<<E as symtn_tensors::WideScalar>::Wide, Error> {
        let space = self.logical_space().space();
        let regions = sector_regions(space.structure(), space.nout())?;
        let mut lease = self.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        validate_cuda_reduction_placement(
            Placement::Cuda(cuda.device()),
            lhs.placement(),
            rhs.placement(),
        )?;
        // ponytail: #740 keeps the proven host-zero upload until a native
        // allocation has correct cross-stream publication and measured value.
        let mut partials =
            CudaStorage::upload_owned(cuda, vec![E::from_real(0.0); regions.len().max(1)])?;
        {
            let mut gemm = CudaStorageGemm::new(cuda);
            for (index, region) in regions.iter().enumerate() {
                let len = region.rows() * region.cols();
                if len == 0 {
                    continue;
                }
                gemm.matmul_range_with_ops_scaled_into(
                    &mut partials,
                    index,
                    lhs,
                    region.range().start,
                    rhs,
                    region.range().start,
                    1,
                    len,
                    1,
                    symtn_dense::MatrixOp::Adjoint,
                    symtn_dense::MatrixOp::Identity,
                    E::from_real(1.0),
                )?;
            }
        }
        let values = download_cuda_reduction_partials(&partials, cuda)?;
        drop(lease);

        regions.iter().zip(values).try_fold(
            <<E as symtn_tensors::WideScalar>::Wide as FactorScalar>::from_real(0.0),
            |total, (region, value)| {
                let dim = multiplicity_free_dim(self.logical_space().provider(), region.coupled())?;
                Ok(total
                    + value.widen()
                        * <<E as symtn_tensors::WideScalar>::Wide as FactorScalar>::from_real(dim))
            },
        )
    }
}

impl<R, D> ReduceExec<R, D> for CudaStorage<D>
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload,
{
    fn dense_norm(_: ExecToken, t: &TensorMap<R, D, Self>, p: f64) -> Result<f64, Error> {
        if p != 2.0 {
            return Err(Error::UnsupportedOnDevice(format!(
                "device norm supports only p = 2, got {p}"
            )));
        }
        let storage = t.direct_cuda_storage("norm")?;
        // `<t, t>` is real up to rounding; the norm is its real part's root,
        // matching the Host norm.
        Ok(t.weighted_inner_cuda(storage, storage)?.re.sqrt())
    }

    fn dense_inner(
        _: ExecToken,
        x: &TensorMap<R, D, Self>,
        y: &TensorMap<R, D, Self>,
    ) -> Result<D, Error> {
        let lhs = x.direct_cuda_storage("inner")?;
        let rhs = y.direct_cuda_storage("inner")?;
        x.weighted_inner_cuda(lhs, rhs).map(D::from_complex64)
    }
}

impl<R, D> ElementwiseExec<R, D> for CudaStorage<D>
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload,
{
    fn zeros(_: ExecToken, t: &TensorMap<R, D, Self>) -> Result<TensorMap<R, D, Self>, Error> {
        let required_len = t.logical_space().space().required_len()?;
        let source = t.direct_cuda_storage("zeros_like")?;
        let output = t.cuda_zeros_owned(required_len, source)?;
        Ok(t.with_owned_cuda_storage(output))
    }

    fn combine(
        _: ExecToken,
        x: &TensorMap<R, D, Self>,
        alpha: D,
        y: Option<(&TensorMap<R, D, Self>, D)>,
    ) -> Result<TensorMap<R, D, Self>, Error> {
        let required_len = x.logical_space().space().required_len()?;
        let operation = if y.is_some() { "axpby" } else { "scale" };
        // Every representation takes one body (#2187), as TensorKit's
        // blockwise `add!`: each nonzero-coefficient operand is read in its
        // logical orientation (an owned one as one packed region, a lazy one
        // through its parent's strided, conjugated blocks; two lazy operands
        // never reach here), the first overwriting and the second
        // accumulating, its coefficient as the descriptor scale. A zero
        // coefficient drops its operand, as on the Host; every operand is
        // still validated, whatever its coefficient.
        let reads = std::iter::once((x, alpha))
            .chain(y)
            .map(|(t, coefficient)| Ok((t.cuda_oriented_read(operation)?, coefficient)))
            .collect::<Result<Vec<_>, Error>>()?;
        let dense_err = |err| Error::from(symtn_tensors::OperationError::Dense(err));
        let mut lease = x.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        let expected = Placement::Cuda(cuda.device());
        for (read, _) in &reads {
            TensorMap::<R, D, Self>::validate_cuda_owned_metadata(
                expected,
                read.source.placement(),
                required_len,
                read.source.len(),
            )?;
        }
        // ponytail: #740 keeps this proven Host upload (the output is born
        // zero) until native device allocation publishes cross-stream writes
        // correctly and wins a bench.
        let mut output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        #[cfg(test)]
        probe::hit(probe::Event::CudaZeroUpload);
        let mut mode = symtn_dense::CudaRegionBeta::Overwrite;
        for (read, coefficient) in reads.iter().filter(|(_, c)| !c.is_zero()) {
            for (source_region, output_region) in &read.moves {
                symtn_dense::cuda_region_axpby::<D>(
                    cuda,
                    &read.source.0,
                    source_region,
                    read.conj,
                    *coefficient,
                    symtn_dense::CudaRegionCoefficient::One,
                    mode,
                    &mut output.0,
                    output_region,
                )
                .map_err(dense_err)?;
                #[cfg(test)]
                if !source_region.is_empty() {
                    probe::hit(probe::Event::CudaArithmeticKernel);
                }
            }
            mode = symtn_dense::CudaRegionBeta::Accumulate;
        }
        drop(lease);
        Ok(x.with_owned_cuda_storage(output))
    }

    /// `beta = 0` never reads the destination, and a zero `alpha` does not
    /// read `x` (the destination becomes `beta * destination` through a
    /// zero-source move).
    fn axpby_into(
        _: ExecToken,
        x: &TensorMap<R, D, Self>,
        destination: &mut TensorMap<R, D, Self>,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let read = x.cuda_oriented_read("axpby_into")?;
        let source = read.source;
        let device = Placement::Cuda(x.runtime.cuda_device_ordinal_checked()?);
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error(
                "ordinary CUDA destination checked above",
            ));
        };
        let destination_data = Arc::get_mut(destination_body)
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or_else(|| internal_layout_error("unique CUDA destination checked above"))?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error(
                "dense CUDA destination checked above",
            ));
        };
        if source.placement() != device || destination_data.placement() != device {
            return Err(Error::PlacementMismatch);
        }
        let len = TensorStorage::len(source);
        if len == 0 {
            return Ok(());
        }
        let dense_err = |err| Error::from(symtn_tensors::OperationError::Dense(err));
        let mut lease = x.runtime.lease_cuda()?;
        let cuda = &mut *lease;
        if alpha.is_zero() {
            let region = symtn_dense::CudaRegion::packed(&[len], 0).map_err(dense_err)?;
            return symtn_dense::cuda_region_scale::<D>(
                cuda,
                &mut destination_data.0,
                &region,
                beta,
            )
            .map_err(dense_err);
        }
        // A logical block with no parent block reads as zero: only `beta`.
        for zero in &read.zeros {
            symtn_dense::cuda_region_scale::<D>(cuda, &mut destination_data.0, zero, beta)
                .map_err(dense_err)?;
        }
        let beta = if beta.is_zero() {
            symtn_dense::CudaRegionBeta::Overwrite
        } else if beta == D::from_real(1.0) {
            symtn_dense::CudaRegionBeta::Accumulate
        } else {
            symtn_dense::CudaRegionBeta::Scale(beta)
        };
        for (source_region, destination_region) in &read.moves {
            symtn_dense::cuda_region_axpby::<D>(
                cuda,
                &source.0,
                source_region,
                read.conj,
                alpha,
                symtn_dense::CudaRegionCoefficient::One,
                beta,
                &mut destination_data.0,
                destination_region,
            )
            .map_err(dense_err)?;
        }
        Ok(())
    }
}

/// A dense CUDA operand read in its logical orientation (#1874): its stored
/// payload and the (stored region, logical region) moves that cover its
/// logical layout, read conjugated when `conj`, plus the logical regions
/// that read as zero.
struct CudaOrientedRead<'a, D: CudaPayload> {
    source: &'a CudaStorage<D>,
    moves: Vec<(symtn_dense::CudaRegion, symtn_dense::CudaRegion)>,
    conj: bool,
    zeros: Vec<symtn_dense::CudaRegion>,
}

impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    /// An owned payload is one packed move. A lazy adjoint is its parent's
    /// blocks through the one [`symtn_tensors::AdjointBlockMap`] lowering
    /// (`materialize` uses the same), conjugated in the strided read: the
    /// Host oriented kernels' per-block `(f₁, f₂) → (f₂, f₁)` read, without
    /// materializing the adjoint.
    fn cuda_oriented_read(
        &self,
        operation: &'static str,
    ) -> Result<CudaOrientedRead<'_, D>, Error> {
        let dense_err = |err| Error::from(symtn_tensors::OperationError::Dense(err));
        let TypedTensorRepr::Adjoint(view) = &self.repr else {
            let source = self.direct_cuda_storage(operation)?;
            let region = symtn_dense::CudaRegion::packed(&[TensorStorage::len(source)], 0)
                .map_err(dense_err)?;
            return Ok(CudaOrientedRead {
                source,
                moves: vec![(region.clone(), region)],
                conj: false,
                zeros: Vec::new(),
            });
        };
        let TypedData::Dense(source) = view.parent.data.as_ref() else {
            return Err(Error::UnsupportedOnDevice(format!(
                "{operation} requires dense CUDA storage"
            )));
        };
        let lowering = super::cuda_transfer::cuda_adjoint_lowering(
            &symtn_tensors::AdjointBlockMap::new(
                view.parent.space.space(),
                view.logical_space.space(),
            ),
            false,
        )?;
        Ok(CudaOrientedRead {
            source,
            moves: lowering.copies,
            conj: true,
            zeros: lowering.zeros,
        })
    }
}
