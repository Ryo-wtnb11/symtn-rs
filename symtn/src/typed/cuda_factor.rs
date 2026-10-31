use super::*;

pub(crate) fn dense_err(err: symtn_dense::DenseError) -> Error {
    Error::from(symtn_tensors::OperationError::Dense(err))
}

/// Uploads a small host-built selector matrix (`rows x cols`, column-major,
/// zero except `entries`) used by the assembly GEMMs.
pub(crate) fn upload_selector<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    rows: usize,
    cols: usize,
    entries: impl Iterator<Item = (usize, usize, D)>,
) -> Result<CudaStorage<D>, Error> {
    let mut data = vec![D::ZERO; rows * cols];
    for (row, col, value) in entries {
        data[row + rows * col] = value;
    }
    let selector = CudaStorage::upload_owned(cuda, data).map_err(Error::from)?;
    #[cfg(test)]
    probe::hit(probe::Event::CudaSelectorUpload);
    Ok(selector)
}

#[inline]
pub(crate) fn cuda_qr_region<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaDenseStorage), Error> {
    let factors = dense_cuda_qr_region::<D>(cuda, source, offset, rows, cols).map_err(dense_err)?;
    #[cfg(test)]
    probe::hit(probe::Event::CudaQrDecomposition);
    Ok(factors)
}

#[inline]
pub(crate) fn cuda_svd_region<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<
    (
        CudaDenseStorage,
        symtn_dense::CudaSpectrum,
        CudaDenseStorage,
    ),
    Error,
> {
    let factors =
        dense_cuda_svd_region::<D>(cuda, source, offset, rows, cols).map_err(dense_err)?;
    #[cfg(test)]
    {
        probe::hit(probe::Event::CudaSvdDecomposition);
        probe::add(probe::Event::CudaSvdSpectrumScalar, factors.1.len());
    }
    Ok(factors)
}

#[inline]
pub(crate) fn cuda_hermitian_regions<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    regions: &[(usize, usize)],
    relative_tolerance: f64,
) -> Result<Vec<bool>, Error> {
    dense_cuda_hermitian_regions::<D>(cuda, source, regions, relative_tolerance).map_err(dense_err)
}

/// The Host finite-input stage of the eager factorizations (#1986) on a
/// device payload: one device scan and one scalar D2H, so a nonfinite input
/// is refused with the Host error before any output or solver call.
pub(crate) fn require_finite_cuda_input<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    family: symtn_matrixalgebra::seam::FactorFamily,
) -> Result<(), Error> {
    if symtn_dense::cuda_all_finite::<D>(cuda, source).map_err(dense_err)? {
        Ok(())
    } else {
        Err(symtn_matrixalgebra::seam::nonfinite_factor_input(family).into())
    }
}

#[inline]
pub(crate) fn typed_cuda_eigh_region<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    source: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<(symtn_dense::CudaSpectrum, CudaDenseStorage), Error> {
    cuda_eigh_region::<D>(cuda, source, offset, n).map_err(dense_err)
}

/// Downloads the device spectra of one factorization call with one transfer.
#[inline]
pub(crate) fn cuda_download_spectra<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    spectra: &[symtn_dense::CudaSpectrum],
) -> Result<Vec<Vec<f64>>, Error> {
    symtn_dense::cuda_download_spectra::<D>(cuda, spectra).map_err(dense_err)
}

/// Writes `factor_rows x kept` slices of `factor * selector` into the target
/// sector region of a left factor (`codomain <- bond`), one GEMM per
/// codomain tree so correctness never relies on tree enumeration order
/// matching between the source and factor spaces.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_left_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    source: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
    k_full: usize,
    selector: &CudaDenseStorage,
    selector_offset: usize,
    kept: usize,
) -> Result<(), Error> {
    for target_tree in target.row_trees() {
        let sub_rows = target_tree.extent()?;
        if sub_rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        cuda_gemm_region_into::<D>(
            cuda,
            &mut dst.0,
            target.range().start + target_tree.offset(),
            target.rows(),
            factor,
            src_row,
            source.rows(),
            selector,
            selector_offset,
            k_full,
            sub_rows,
            k_full,
            kept,
            D::ONE,
            D::ZERO,
        )
        .map_err(dense_err)?;
        #[cfg(test)]
        probe::hit(probe::Event::CudaAssemblyGemm);
    }
    Ok(())
}

/// Writes `factor * selector` into a left-factor region with one GEMM.
///
/// Only valid on a route the plan proved layout-aligned: the target's
/// codomain trees tile it exactly as the source's do, so the per-tree GEMMs
/// of [`assemble_left_factor`] are row slices of this one product.
///
/// The product is exact data movement: each selector column holds a single
/// unit entry. Why not Tenferro `gather`: at 0.7.1 it reads owned index
/// tensors only, so one index upload per call would need a device slice copy
/// per route to feed it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_aligned_left_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
    k_full: usize,
    selector: &CudaStorage<D>,
    selector_offset: usize,
    kept: usize,
) -> Result<(), Error> {
    if target.rows() == 0 {
        return Ok(());
    }
    cuda_gemm_region_into::<D>(
        cuda,
        &mut dst.0,
        target.range().start,
        target.rows(),
        factor,
        0,
        target.rows(),
        &selector.0,
        selector_offset,
        k_full,
        target.rows(),
        k_full,
        kept,
        D::ONE,
        D::ZERO,
    )
    .map_err(dense_err)?;
    #[cfg(test)]
    probe::hit(probe::Event::CudaAssemblyGemm);
    Ok(())
}

/// Writes `kept x factor_cols` slices of `selector * factor` into the target
/// sector region of a right factor (`bond <- domain`), one GEMM per domain
/// tree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_right_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    source: &CoupledSectorRegion,
    selector: &CudaDenseStorage,
    kept: usize,
    k_full: usize,
    factor: &CudaDenseStorage,
) -> Result<(), Error> {
    for target_tree in target.col_trees() {
        let sub_cols = target_tree.extent()?;
        if sub_cols == 0 {
            continue;
        }
        let src_col = source
            .col_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("domain tree missing in the source sector"))?;
        cuda_gemm_region_into::<D>(
            cuda,
            &mut dst.0,
            target.range().start + target.rows() * target_tree.offset(),
            target.rows(),
            selector,
            0,
            kept,
            factor,
            k_full * src_col,
            k_full,
            kept,
            k_full,
            sub_cols,
            D::ONE,
            D::ZERO,
        )
        .map_err(dense_err)?;
        #[cfg(test)]
        probe::hit(probe::Event::CudaAssemblyGemm);
    }
    Ok(())
}

/// Copies a whole compact device factor into its target sector region.
///
/// Only valid on a route the plan proved layout-aligned: the target region's
/// tree sequence equals the source's (same trees, extents and offsets), so the
/// region is exactly the compact factor and the identity-selector GEMM reduces
/// to one device copy. The per-tree GEMM stays the general assembly path.
pub(crate) fn copy_whole_factor<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
) -> Result<(), Error> {
    // `cuda_copy_region_into` accepts a source at least as long as the region
    // so one long buffer can serve every shorter region; this route instead
    // promises the factor *is* the region, so assert that here rather
    // than silently copying a prefix of a longer factor.
    if factor.len() != target.rows().saturating_mul(target.cols()) {
        return Err(internal_layout_error(
            "device factor length does not match its aligned target region",
        ));
    }
    cuda_copy_region_into::<D>(
        cuda,
        &mut dst.0,
        target.range().start,
        target.rows(),
        factor,
        target.rows(),
        target.cols(),
    )
    .map_err(dense_err)?;
    #[cfg(test)]
    probe::hit(probe::Event::CudaFactorCopy);
    Ok(())
}

/// Copies every column of a full `source.rows() x cols` device factor into a
/// left-factor region, one strided copy per codomain tree: the per-tree
/// assembly of [`assemble_left_factor`] when its selector is the identity.
pub(crate) fn copy_left_factor_treewise<D: CudaPayload>(
    cuda: &mut CudaDenseContext,
    dst: &mut CudaStorage<D>,
    target: &CoupledSectorRegion,
    source: &CoupledSectorRegion,
    factor: &CudaDenseStorage,
    cols: usize,
) -> Result<(), Error> {
    let region = |rows: usize, leading: usize, offset: usize| {
        symtn_dense::CudaRegion::new(vec![rows, cols], vec![1, leading], offset).map_err(dense_err)
    };
    for target_tree in target.row_trees() {
        let sub_rows = target_tree.extent()?;
        if sub_rows == 0 {
            continue;
        }
        let src_row = source
            .row_trees()
            .iter()
            .find(|source_tree| source_tree.tree() == target_tree.tree())
            .map(|source_tree| source_tree.offset())
            .ok_or_else(|| internal_layout_error("codomain tree missing in the source sector"))?;
        symtn_dense::cuda_copy_strided_into::<D>(
            cuda,
            factor,
            &region(sub_rows, source.rows(), src_row)?,
            &mut dst.0,
            &region(
                sub_rows,
                target.rows(),
                target.range().start + target_tree.offset(),
            )?,
        )
        .map_err(dense_err)?;
        #[cfg(test)]
        probe::hit(probe::Event::CudaFactorCopy);
    }
    Ok(())
}

/// Fills the diagonal of a coupled-layout `W <- W` buffer from per-sector
/// spectra, mirroring the host `diagonal_bond_tensor_dyn`.
pub(crate) fn fill_diagonal_values<D: CudaPayload>(
    structure: &BlockStructure,
    data: &mut [D],
    spectra: &[symtn_matrixalgebra::SectorSpectrum<f64>],
) -> Result<(), Error> {
    let mut cursor = super::fusion_tree::SpectrumCursor::default();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(entry) = cursor.find(spectra, sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            data[offset + position * (strides[0] + strides[1])] =
                D::from_real(entry.values[position]);
        }
    }
    Ok(())
}

/// The plan's nonzero-rank routes with their `(left, right)` factor regions:
/// a zero-rank sector has nothing to decompose or publish.
pub(super) fn executed_routes(
    plan: &CompactFactorPlan,
) -> impl Iterator<Item = (&CompactFactorRoute, usize, usize)> {
    plan.executed_routes()
}

pub(super) struct TypedCudaQrScratch<D: CudaPayload> {
    pub(super) left: CudaDenseStorage,
    pub(super) right: CudaDenseStorage,
    /// `None` when both assembly routes are layout-aligned and copy instead.
    pub(super) selector: Option<CudaStorage<D>>,
}

impl<D: CudaPayload> TypedCudaQrScratch<D> {
    pub(super) fn new(
        left: CudaDenseStorage,
        right: CudaDenseStorage,
        selector: Option<CudaStorage<D>>,
    ) -> Self {
        #[cfg(test)]
        probe::enter(probe::Gauge::CudaQrScratch);
        Self {
            left,
            right,
            selector,
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
impl<D: CudaPayload> Drop for TypedCudaQrScratch<D> {
    fn drop(&mut self) {
        probe::leave(probe::Gauge::CudaQrScratch);
    }
}

pub(super) struct TypedCudaSvdScratch<D: CudaPayload> {
    pub(super) left: CudaDenseStorage,
    pub(super) right: CudaDenseStorage,
    /// `None` on a layout-aligned side, whose gauged factor is copied.
    pub(super) left_selector: Option<CudaDenseStorage>,
    pub(super) right_selector: Option<CudaDenseStorage>,
    payload: std::marker::PhantomData<D>,
}

impl<D: CudaPayload> TypedCudaSvdScratch<D> {
    pub(super) fn new(
        left: CudaDenseStorage,
        right: CudaDenseStorage,
        left_selector: Option<CudaDenseStorage>,
        right_selector: Option<CudaDenseStorage>,
    ) -> Self {
        #[cfg(test)]
        probe::enter(probe::Gauge::CudaSvdScratch);
        Self {
            left,
            right,
            left_selector,
            right_selector,
            payload: std::marker::PhantomData,
        }
    }

    pub(super) fn selector(
        selector: &Option<CudaDenseStorage>,
    ) -> Result<&CudaDenseStorage, Error> {
        selector.as_ref().ok_or_else(|| {
            internal_layout_error("a non-aligned factor route has no gauge selector")
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
impl<D: CudaPayload> Drop for TypedCudaSvdScratch<D> {
    fn drop(&mut self) {
        probe::leave(probe::Gauge::CudaSvdScratch);
    }
}

/// Validates the compact SVD diagonal factor against its routes and returns
/// each executed route's diagonal start: the packed `rank x rank` region's
/// offset, whose diagonal then has element stride `rank + 1`.
pub(super) fn validate_cuda_svd_middle_regions(
    plan: &CompactFactorPlan,
    middle_regions: &[CoupledSectorRegion],
) -> Result<Vec<usize>, Error> {
    let mut by_sector = HashMap::with_capacity(middle_regions.len());
    for region in middle_regions {
        if by_sector.insert(region.coupled(), region).is_some() {
            return Err(internal_layout_error(
                "compact SVD diagonal factor contains a duplicate coupled sector",
            ));
        }
    }
    let mut diagonals = Vec::with_capacity(middle_regions.len());
    for (route, _, _) in executed_routes(plan) {
        let rank = route.rank();
        let middle = by_sector.get(&route.sector()).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor is missing a source sector")
        })?;
        let range_len = middle.range().len();
        let expected_len = rank.checked_mul(rank).ok_or_else(|| {
            internal_layout_error("compact SVD diagonal factor region length overflows")
        })?;
        if (middle.rows(), middle.cols()) != (rank, rank)
            || range_len != expected_len
            || !middle.has_aligned_diagonal()
        {
            return Err(internal_layout_error(
                "compact SVD diagonal factor region does not match its source route",
            ));
        }
        diagonals.push(middle.range().start);
    }
    if diagonals.len() != middle_regions.len() {
        return Err(internal_layout_error(
            "compact SVD diagonal factor does not have exactly one region per route",
        ));
    }
    Ok(diagonals)
}

/// Admits a CUDA operation's device inputs and destinations on the caller's
/// engine, before it uploads, allocates or writes anything. A buffer of
/// another engine is [`Error::PlacementMismatch`] even on the same device
/// ordinal: engines are separate allocation domains.
pub(crate) fn ensure_cuda_owns(
    cuda: &CudaDenseContext,
    op: &'static str,
    operands: &[&CudaDenseStorage],
) -> Result<(), Error> {
    cuda.ensure_owns(op, operands)
        .map_err(|_| Error::PlacementMismatch)
}

pub(super) fn download_cuda_reduction_partials<D: CudaPayload>(
    partials: &CudaStorage<D>,
    cuda: &CudaDenseContext,
) -> Result<Vec<D>, Error> {
    #[cfg(test)]
    {
        probe::hit(probe::Event::CudaReductionDownload);
        probe::add(probe::Event::CudaReductionDevicePartials, partials.len());
    }
    let values = partials.download(cuda)?;
    #[cfg(test)]
    probe::add(probe::Event::CudaReductionHostPartials, values.len());
    Ok(values)
}
