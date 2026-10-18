use super::*;

/// Device compact SVD (`svd_compact`) and Hermitian eigendecomposition
/// (`eigh_full`) over either device payload (`f64` or `Complex64`).
///
/// A truncated factorization is a composition, and its decision over
/// quantum-dimension-weighted spectra across all coupled sectors belongs on
/// the host; placement stays the caller's explicit choice:
///
/// ```no_run
/// use std::sync::Arc;
///
/// use symtn::sector::U1Irrep;
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{Runtime, Svd, TensorMap, Truncation};
/// use symtn::typed::GradedSpace;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let runtime = Runtime::builder().cuda(0).build()?;
/// let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)])?;
/// let device = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
///     (index[0] + 2 * index[1]) as f64
/// })?
/// .to_cuda()?;
///
/// let Svd { u, s, vh } = device.svd_compact(&[0], &[1])?;
/// // Until a device `restrict_leg` lands, the factors move to the host once.
/// let (u, s, vh) = (u.to_host()?, s.to_host()?, vh.to_host()?);
/// let found = s.domain()[0].find_truncated(&s.diagview()?, &Truncation::rank(2))?;
/// let u = u.restrict_leg(&[(u.codomain_rank(), &found.selection)])?;
/// let s = s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])?;
/// let vh = vh.restrict_leg(&[(0, &found.selection)])?;
/// # let _ = (u, s, vh, found.error);
/// # Ok(())
/// # }
/// ```
///
/// The `d` and `v` of `eigh_full` are truncated the same way.
///
/// `u` and `vh` follow the Host largest-pivot gauge for every payload, as
/// TensorKit's (MatrixAlgebraKit `gaugefix!`) does. `eigh_full`
/// admits a block only when it equals its *conjugate* transpose, so a
/// complex-symmetric non-Hermitian block is rejected before any
/// factorization.
///
/// Compact QR is not here: it lives in its own impl below.
///
/// Checked Generic providers deliberately have no device factorizations, and
/// compact SVD has the same deliberately narrow typed CUDA surface:
///
/// ```compile_fail
/// use symtn::sector::{CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use symtn::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_svd<R>(tensor: &TensorMap<R, f64, CudaStorage>)
/// where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// The device payload marker alone does not admit a device factorization, and
/// neither does it together with the *host* factorization marker: the gate is
/// [`CudaFactorizationPayload`]. Every payload implements it (#1341), but
/// a *generic* body must still name it — the two bounds below are not enough,
/// because neither says anything about the device gauge or the device
/// Hermitian rule.
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{CudaPayload, CudaStorage, TensorMap};
///
/// fn device_payload_only<D: CudaPayload>(tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::FactorizationScalar;
/// use symtn::typed::{CudaPayload, CudaStorage, TensorMap};
///
/// fn device_payload_and_host_factorizing<D: CudaPayload + FactorizationScalar>(
///     tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>,
/// ) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{CudaFactorizationPayload, CudaStorage, TensorMap};
///
/// fn device_factorizing<D: CudaFactorizationPayload>(
///     tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>,
/// ) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// Concretely, at all four device payloads — the single-precision twins of the
/// pins #1336 left here, which differ only in the dtype:
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{CudaStorage, TensorMap};
///
/// fn f32_device_svd(tensor: &TensorMap<U1FusionRule, f32, CudaStorage<f32>>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
///
/// fn f64_device_svd(tensor: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>) {
///     let _ = tensor.svd_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use num_complex::{Complex32, Complex64};
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{CudaStorage, HermitianTol, TensorMap};
///
/// fn c32_device_eigh(tensor: &TensorMap<U1FusionRule, Complex32, CudaStorage<Complex32>>) {
///     let _ = tensor.eigh_full(&[0], &[1], HermitianTol::DEFAULT);
/// }
///
/// fn c64_device_eigh(tensor: &TensorMap<U1FusionRule, Complex64, CudaStorage<Complex64>>) {
///     let _ = tensor.eigh_full(&[0], &[1], HermitianTol::DEFAULT);
/// }
/// ```
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// The `k x k` identity selector the non-aligned assembly GEMMs need.
    /// A fully aligned route assembles by copy and uploads nothing.
    fn identity_selector(
        cuda: &mut CudaDenseContext,
        rank: usize,
        aligned_left: bool,
        aligned_right: bool,
    ) -> Result<Option<CudaStorage<D>>, Error> {
        if aligned_left && aligned_right {
            return Ok(None);
        }
        upload_selector(
            cuda,
            rank,
            rank,
            (0..rank).map(|index| (index, index, D::ONE)),
        )
        .map(Some)
    }

    fn route_selector(selector: &Option<CudaStorage<D>>) -> Result<&CudaStorage<D>, Error> {
        selector.as_ref().ok_or_else(|| {
            internal_layout_error("a non-aligned factor route has no assembly selector")
        })
    }

    /// The Host compact factor plan of this tensor's matrix view: the one
    /// authority for the bond, the factor spaces and the sector routes.
    pub(super) fn compact_factor_plan(&self) -> Result<CompactFactorPlan, Error> {
        Ok(compact_factor_routes(self.logical_space())?)
    }

    /// The admission every device factorization shares, in one order:
    /// dense storage, the matrix algebra's endomorphism admission of
    /// `endomorphism` (if any), regions, placement, then the Host compact
    /// factor plan. No device lock is taken and no output exists yet.
    fn cuda_factor_preflight(
        &self,
        operation: &'static str,
        endomorphism: Option<symtn_matrixalgebra::seam::EndomorphismFamily>,
    ) -> Result<(&CudaStorage<D>, CompactFactorPlan), Error> {
        let source = self.direct_cuda_storage(operation)?;
        let source_space = self.logical_space().space();
        if let Some(family) = endomorphism {
            family.require_endomorphism(source_space.homspace())?;
        }
        let required_len = source_space.required_len()?;
        // Region admission precedes the placement preflight; the plan reads
        // the same cached regions.
        let regions = sector_regions(source_space.structure(), source_space.nout())?;
        if let Some(family) = endomorphism {
            if regions.iter().any(|region| region.rows() != region.cols()) {
                return Err(internal_layout_error(
                    "CUDA EIGH source contains a non-square coupled-sector region",
                ));
            }
            symtn_matrixalgebra::seam::validate_endomorphism_region_stacking(&regions, family)?;
        }
        {
            // Preflight only: the ordinal is immutable, so this placement
            // check takes no device lock at all.
            let device = self.runtime.cuda_device_ordinal_checked()?;
            Self::validate_cuda_owned_metadata(
                Placement::Cuda(device),
                source.placement(),
                required_len,
                source.len(),
            )?;
        }
        Ok((source, self.compact_factor_plan()?))
    }

    /// The device kernel of [`TensorMap::svd_compact`] on the current split.
    fn svd_compact_matrix(&self) -> Result<Svd<Self>, Error> {
        // As for typed CUDA QR, all provider work and final-space admission
        // complete before the execution lock and before any output exists.
        let (source, plan) = self.cuda_factor_preflight("svd_compact", None)?;
        let treewise = probe::RoutePolicy::current().svd_treewise;
        let left_space = plan.left_space(self.logical_space())?;
        let right_space = plan.right_space(self.logical_space())?;
        let middle_space = plan.bond_space(self.logical_space())?;
        let middle_regions = sector_regions(
            middle_space.space().structure(),
            middle_space.space().nout(),
        )?;
        let diagonals = validate_cuda_svd_middle_regions(&plan, &middle_regions)?;
        let left_len = left_space.space().required_len()?;
        let middle_len = middle_space.space().required_len()?;
        let right_len = right_space.space().required_len()?;
        let (left_data, middle_data, right_data) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            require_finite_cuda_input::<D>(
                cuda,
                &source.0,
                symtn_matrixalgebra::seam::FactorFamily::Svd,
            )?;
            let mut left_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; left_len])?;
            #[cfg(test)]
            probe::hit(probe::Event::CudaSvdOutputUpload);
            let mut right_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; right_len])?;
            #[cfg(test)]
            probe::hit(probe::Event::CudaSvdOutputUpload);
            // The zero upload is the only device allocation path (#740);
            // every value then arrives by a device copy.
            let mut middle_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; middle_len])?;
            #[cfg(test)]
            probe::hit(probe::Event::CudaSvdOutputUpload);

            let max_rows = executed_routes(&plan)
                .map(|(route, _, _)| plan.source_regions()[route.source_region()].rows())
                .max();
            let weights = max_rows
                .map(|rows| CudaSvdGaugeWeights::upload(cuda, rows))
                .transpose()
                .map_err(dense_err)?;
            for ((route, left, right), &diagonal) in executed_routes(&plan).zip(&diagonals) {
                let rank = route.rank();
                let source_region = &plan.source_regions()[route.source_region()];
                let left_region = &plan.left_regions()[left];
                let right_region = &plan.right_regions()[right];
                let aligned_left = !treewise && plan.left_preserves_trees(route);
                let aligned_right = !treewise && plan.right_preserves_trees(route);
                let (raw_left, spectrum, raw_right) = cuda_svd_region::<D>(
                    cuda,
                    &source.0,
                    source_region.range().start,
                    source_region.rows(),
                    source_region.cols(),
                )?;
                if spectrum.len() != rank {
                    return Err(internal_layout_error(
                        "compact SVD spectrum length does not match its source route",
                    ));
                }
                let weights = weights.as_ref().ok_or_else(|| {
                    internal_layout_error("a compact SVD route has no gauge weights")
                })?;
                // The Host gauge (first largest-|U| entry of each column real
                // and non-negative) is folded into the selector a non-aligned
                // side multiplies by anyway; only an aligned side, which is a
                // plain copy, pays a separate scaling pass.
                let phases = cuda_svd_gauge_phases::<D>(
                    cuda,
                    &raw_left,
                    source_region.rows(),
                    rank,
                    weights,
                )
                .map_err(dense_err)?;
                let (left, left_selector) = if aligned_left {
                    let left = phases
                        .scale_left::<D>(cuda, &raw_left, source_region.rows())
                        .map_err(dense_err)?;
                    (left, None)
                } else {
                    let selector = phases.left_selector::<D>(cuda).map_err(dense_err)?;
                    (raw_left, Some(selector))
                };
                let (right, right_selector) = if aligned_right {
                    let right = phases
                        .scale_right::<D>(cuda, &raw_right, source_region.cols())
                        .map_err(dense_err)?;
                    (right, None)
                } else {
                    let selector = phases.right_selector::<D>(cuda).map_err(dense_err)?;
                    (raw_right, Some(selector))
                };
                drop(phases);
                // The scratch owns all route-local allocations. It is dropped
                // at the end of this iteration, bounding peak raw-factor
                // storage independently of the number of sectors.
                let scratch =
                    TypedCudaSvdScratch::<D>::new(left, right, left_selector, right_selector);
                // Compact SVD keeps the full rank by construction, so the same
                // proved layout identity applies as for QR.
                if aligned_left {
                    copy_whole_factor(cuda, &mut left_data, left_region, &scratch.left)?;
                } else {
                    assemble_left_factor(
                        cuda,
                        &mut left_data,
                        left_region,
                        source_region,
                        &scratch.left,
                        rank,
                        TypedCudaSvdScratch::<D>::selector(&scratch.left_selector)?,
                        0,
                        rank,
                    )?;
                }
                if aligned_right {
                    copy_whole_factor(cuda, &mut right_data, right_region, &scratch.right)?;
                } else {
                    assemble_right_factor(
                        cuda,
                        &mut right_data,
                        right_region,
                        source_region,
                        TypedCudaSvdScratch::<D>::selector(&scratch.right_selector)?,
                        rank,
                        rank,
                        &scratch.right,
                    )?;
                }
                symtn_dense::cuda_copy_spectrum_into::<D>(
                    cuda,
                    spectrum,
                    &mut middle_data.0,
                    diagonal,
                    rank + 1,
                )
                .map_err(dense_err)?;
            }
            (left_data, middle_data, right_data)
        };

        Ok(Svd {
            u: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(left_space, left_data)),
            },
            s: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(middle_space, middle_data)),
            },
            vh: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(right_space, right_data)),
            },
        })
    }

    /// The device kernel of [`TensorMap::eigh_full`] on the current split.
    fn eigh_full_matrix(&self, hermitian_tol: HermitianTol) -> Result<Eigh<Self>, Error> {
        // The Host compact factor plan: an endomorphism's compact bond keeps
        // every eigenpair (`min(n, n) = n`), so its left route is EIGH's.
        let (source, plan) = self.cuda_factor_preflight(
            "eigh_full",
            Some(symtn_matrixalgebra::seam::EndomorphismFamily::EighFull),
        )?;
        let policy = probe::RoutePolicy::current();

        // Admission is complete. Validate every block before the first EIGH so
        // a late nonfinite or non-Hermitian sector cannot trigger partial
        // numerical work; finiteness first, as on the Host.
        {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            require_finite_cuda_input::<D>(
                cuda,
                &source.0,
                symtn_matrixalgebra::seam::FactorFamily::Eigh,
            )?;
            let regions: Vec<_> = plan
                .source_regions()
                .iter()
                .map(|region| (region.range().start, region.rows()))
                .collect();
            let tol = hermitian_tol.resolve(<D as FactorScalar>::epsilon());
            if cuda_hermitian_regions::<D>(cuda, &source.0, &regions, tol)?.contains(&false) {
                return Err(symtn_tensors::OperationError::InvalidArgument {
                    message: symtn_matrixalgebra::seam::EIGH_NOT_HERMITIAN,
                }
                .into());
            }
        }

        let (spectra, mut raw_vectors, orders, permuted) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let mut device_spectra = Vec::with_capacity(plan.source_regions().len());
            let mut vectors = Vec::with_capacity(plan.source_regions().len());
            for region in plan.source_regions().iter() {
                let n = region.rows();
                if n == 0 {
                    vectors.push(None);
                    continue;
                }
                let (values, vector) =
                    typed_cuda_eigh_region::<D>(cuda, &source.0, region.range().start, n)?;
                #[cfg(test)]
                Self::inject_cuda_eigh_failure("decomposition")?;
                device_spectra.push(values);
                vectors.push(Some(vector));
            }
            let mut downloaded = cuda_download_spectra::<D>(cuda, &device_spectra)?.into_iter();
            let mut spectra = Vec::with_capacity(plan.source_regions().len());
            let mut orders = Vec::with_capacity(plan.source_regions().len());
            let mut permuted = false;
            for (region, sector_index) in plan.source_regions().iter().zip(0usize..) {
                let n = region.rows();
                let values = if n == 0 {
                    Vec::new()
                } else {
                    downloaded.next().ok_or_else(|| {
                        internal_layout_error("CUDA EIGH spectrum count does not match its blocks")
                    })?
                };
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(internal_layout_error(
                        "CUDA EIGH returned a non-finite eigenvalue",
                    ));
                }
                let mut order = vec![0; n];
                let reorder = symtn_matrixalgebra::seam::ascending_eigh_order(&values, &mut order);
                // A test policy reverses every odd sector's order, so the
                // selector path sees a real, non-identity permutation.
                let reorder = reorder
                    || (policy.eigh_reverse_odd_sectors && sector_index % 2 == 1 && n >= 2 && {
                        order.reverse();
                        true
                    });
                let sorted = if reorder {
                    permuted = true;
                    order.iter().map(|&index| values[index]).collect()
                } else {
                    values
                };
                spectra.push(symtn_matrixalgebra::SectorSpectrum {
                    sector: region.coupled(),
                    values: sorted,
                });
                orders.push(order);
            }
            (
                spectra,
                vectors,
                orders,
                permuted || policy.eigh_force_selector,
            )
        };

        // Every factor-space admission is provider work on the host: no CUDA
        // lease is held from here until the assembly below.
        let left_space = plan.left_space(self.logical_space())?;
        let middle_space = plan.bond_space(self.logical_space())?;
        let vector_len = left_space.space().required_len()?;
        let diagonal_len = middle_space.space().required_len()?;
        let mut diagonal_host = vec![D::ZERO; diagonal_len];
        fill_diagonal_values(
            middle_space.space().structure(),
            &mut diagonal_host,
            &spectra,
        )?;

        let (diagonal_data, vector_data) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            let diagonal_data = CudaStorage::upload_owned(cuda, diagonal_host)?;
            let mut vector_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; vector_len])?;
            // Every route's `full_rank x kept` column selector, packed back
            // to back so the call makes one upload whatever its route count.
            let mut selector_offsets = Vec::with_capacity(plan.routes().len());
            let mut selector_len = 0usize;
            for (route, _, _) in executed_routes(&plan) {
                if route.rank() > orders[route.source_region()].len() {
                    return Err(internal_layout_error(
                        "CUDA EIGH rank exceeds its eigenvector order",
                    ));
                }
                selector_offsets.push(selector_len);
                selector_len += route.rank() * route.rank();
            }
            // cuSOLVER `syevd` is ascending, so every order is normally the
            // identity: then each route is a plain column copy and no
            // selector is uploaded or multiplied.
            let selector = if selector_len == 0 || !permuted {
                None
            } else {
                Some(upload_selector(
                    cuda,
                    selector_len,
                    1,
                    executed_routes(&plan).zip(&selector_offsets).flat_map(
                        |((route, _, _), &offset)| {
                            let n = route.rank();
                            orders[route.source_region()][..n]
                                .iter()
                                .enumerate()
                                .map(move |(column, &row)| (offset + row + n * column, 0, D::ONE))
                        },
                    ),
                )?)
            };
            for ((route, left, _), &selector_offset) in
                executed_routes(&plan).zip(&selector_offsets)
            {
                #[cfg(test)]
                Self::inject_cuda_eigh_failure("assembly")?;
                let raw = raw_vectors[route.source_region()]
                    .take()
                    .ok_or_else(|| internal_layout_error("CUDA EIGH route has no eigenvectors"))?;
                let left_region = &plan.left_regions()[left];
                let n = route.rank();
                let aligned = plan.left_preserves_trees(route) && !policy.eigh_treewise;
                let Some(selector) = selector.as_ref() else {
                    if aligned {
                        copy_whole_factor(cuda, &mut vector_data, left_region, &raw)?;
                    } else {
                        copy_left_factor_treewise(
                            cuda,
                            &mut vector_data,
                            left_region,
                            &plan.source_regions()[route.source_region()],
                            &raw,
                            n,
                        )?;
                    }
                    continue;
                };
                if aligned {
                    assemble_aligned_left_factor(
                        cuda,
                        &mut vector_data,
                        left_region,
                        &raw,
                        n,
                        selector,
                        selector_offset,
                        n,
                    )?;
                } else {
                    assemble_left_factor(
                        cuda,
                        &mut vector_data,
                        left_region,
                        &plan.source_regions()[route.source_region()],
                        &raw,
                        n,
                        &selector.0,
                        selector_offset,
                        n,
                    )?;
                }
            }
            (diagonal_data, vector_data)
        };

        Ok(Eigh {
            d: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(middle_space, diagonal_data)),
            },
            v: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(left_space, vector_data)),
            },
        })
    }

    #[cfg(test)]
    fn inject_cuda_eigh_failure(stage: &'static str) -> Result<(), Error> {
        if probe::fires(probe::Fault::CudaEigh(stage)) {
            return Err(Error::InvalidArgument(format!(
                "injected CUDA EIGH {stage} failure"
            )));
        }
        Ok(())
    }
}

/// Device compact QR, every device factorization payload (`f64`, `f32`,
/// `Complex64`, `Complex32`).
///
/// `qr_compact` returns the Host gauge exactly: positive diagonal (`R_jj` real
/// and non-negative, phase 1 kept where `R_jj == 0`, MatrixAlgebraKit
/// `positive = true`), applied on device by the backend's own QR primitive
/// (`QrGauge::PositiveDiagonal`: `Q -> Q D`, `R -> D^H R` with
/// `D = diag(R_jj / |R_jj|)`) rather than re-derived here.
///
/// The payload dtype is the only degree of freedom; the plan, the routes and
/// the assembly are the ones the device SVD shares.
///
/// Checked Generic providers have no device QR:
///
/// ```compile_fail
/// use symtn::sector::{CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use symtn::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_qr<R>(tensor: &TensorMap<R, f64, CudaStorage>)
/// where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.qr_compact(&[0], &[1]);
/// }
/// ```
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{CudaFactorizationPayload, CudaStorage, TensorMap};
///
/// fn device_qr<D: CudaFactorizationPayload>(
///     tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>,
/// ) {
///     let _ = tensor.qr_compact(&[0], &[1]);
/// }
/// ```
impl<R, D> TensorMap<R, D, CudaStorage<D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    /// The device kernel of [`TensorMap::qr_compact`] on the current split.
    fn qr_compact_matrix(&self) -> Result<Qr<Self>, Error> {
        // Provider queries and final HomSpace admission belong outside the
        // execution lock; the plan owns every source-to-factor route.
        let (source, plan) = self.cuda_factor_preflight("qr_compact", None)?;
        let left_space = plan.left_space(self.logical_space())?;
        let right_space = plan.right_space(self.logical_space())?;
        let left_len = left_space.space().required_len()?;
        let right_len = right_space.space().required_len()?;

        let (left_data, right_data) = {
            let mut lease = self.runtime.lease_cuda()?;
            let cuda = &mut *lease;
            require_finite_cuda_input::<D>(
                cuda,
                &source.0,
                symtn_matrixalgebra::seam::FactorFamily::Qr,
            )?;
            let mut left_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; left_len])?;
            #[cfg(test)]
            probe::hit(probe::Event::CudaQrOutputUpload);
            let mut right_data = CudaStorage::upload_owned(cuda, vec![D::ZERO; right_len])?;
            #[cfg(test)]
            probe::hit(probe::Event::CudaQrOutputUpload);
            for (route, left, right) in executed_routes(&plan) {
                let rank = route.rank();
                let source_region = &plan.source_regions()[route.source_region()];
                let left_region = &plan.left_regions()[left];
                let right_region = &plan.right_regions()[right];
                let aligned_left = plan.left_preserves_trees(route);
                let aligned_right = plan.right_preserves_trees(route);
                let (raw_left, raw_right) = cuda_qr_region::<D>(
                    cuda,
                    &source.0,
                    source_region.range().start,
                    source_region.rows(),
                    source_region.cols(),
                )?;
                let selector = Self::identity_selector(cuda, rank, aligned_left, aligned_right)?;
                let scratch = TypedCudaQrScratch::new(raw_left, raw_right, selector);
                if aligned_left {
                    copy_whole_factor(cuda, &mut left_data, left_region, &scratch.left)?;
                } else {
                    assemble_left_factor(
                        cuda,
                        &mut left_data,
                        left_region,
                        source_region,
                        &scratch.left,
                        rank,
                        &Self::route_selector(&scratch.selector)?.0,
                        0,
                        rank,
                    )?;
                }
                if aligned_right {
                    copy_whole_factor(cuda, &mut right_data, right_region, &scratch.right)?;
                } else {
                    assemble_right_factor(
                        cuda,
                        &mut right_data,
                        right_region,
                        source_region,
                        &Self::route_selector(&scratch.selector)?.0,
                        rank,
                        rank,
                        &scratch.right,
                    )?;
                }
            }
            (left_data, right_data)
        };

        Ok(Qr {
            q: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(left_space, left_data)),
            },
            r: Self {
                runtime: self.runtime.clone(),
                repr: owned_repr(TypedTensorBody::dense(right_space, right_data)),
            },
        })
    }
}

impl<R, D> FactorExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    fn qr_compact(
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
    ) -> Result<Qr<TensorMap<R, D, Self>>, Error> {
        t.qr_compact_matrix()
    }

    fn svd_compact(
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
    ) -> Result<Svd<TensorMap<R, D, Self>>, Error> {
        t.svd_compact_matrix()
    }

    fn eigh_full(
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<TensorMap<R, D, Self>>, Error> {
        t.eigh_full_matrix(hermitian_tol)
    }

    fn eigh_full_stack(
        _: ExecToken,
        plan: &EighFullPlan<R, D, Self>,
        source: &StackedTensorMap<R, D, Self>,
        buffers: &mut Option<super::batched::StackPair<R, D, Self>>,
        workspace: &mut EighFullWorkspace<R, D, Self>,
    ) -> Result<(), BatchError> {
        plan.run_cuda(
            source,
            buffers,
            &mut workspace.sorted,
            &mut workspace.orders,
        )
    }
}
