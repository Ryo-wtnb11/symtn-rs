use super::*;

#[cfg(test)]
/// Compact (thin, untruncated) fusion-tensor SVD `t = U * S * Vh`
/// (MatrixAlgebraKit `svd_compact`).
///
/// This is the pure device-boundary factorization: the dense per-sector SVDs
/// run through the [`DenseExecutor`] and no truncation logic is involved.
/// Per block the bond is `min(rows, cols)`; the square-`U` variant is
/// MatrixAlgebraKit `svd_full` (later batch).
#[derive(Clone, Debug)]
pub(crate) struct SvdCompact<R, D> {
    pub u: BoundTensorMap<R, D>,
    pub s: BoundTensorMap<R, D>,
    pub vh: BoundTensorMap<R, D>,
    pub singular_values: Vec<SectorSpectrum>,
}

#[cfg(test)]
/// Dynamic-rank [`SvdCompact`].
#[derive(Clone, Debug)]
pub(crate) struct SvdCompactDyn<R, D> {
    pub(super) u: BoundDynFactor<R, D>,
    pub(super) s: BoundDynFactor<R, D>,
    pub(super) vh: BoundDynFactor<R, D>,
    pub(super) singular_values: Vec<SectorSpectrum>,
}

#[cfg(test)]
impl<R, D> SvdCompactDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }

    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }

    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }

    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }
}

#[cfg(test)]
pub(super) fn diagonal_bond_svd_factor<R, D, V>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
    V: Copy,
{
    #[cfg(test)]
    {
        crate::probe::hit(crate::probe::Event::DiagonalBondBuilds);
        crate::probe::add(
            crate::probe::Event::DiagonalBondValues,
            spectrum.iter().map(|entry| entry.values.len()).sum(),
        );
    }
    let space = spectrum_bond::<MultiplicityFreeAdmissionMode, _, _>(authority, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)?;
    BoundDynFactor::from_bound(space, data, 1, 1)
}

/// Fills the dense block-diagonal data of `space` from `spectrum`, mapping
/// each value through `to_scalar`. Only the
/// per-block diagonal is written; the rest stays zero.
#[doc(hidden)]
pub fn diagonal_bond_data<D, V>(
    space: &DynamicFusionMapSpace,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<Vec<D>, OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let len = space
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        copy_mapped_to_strided_diagonal(
            &mut data,
            offset,
            strides[0] + strides[1],
            &entry.values[..count],
            to_scalar,
        );
    }
    Ok(data)
}

/// Scales one bond axis of `data` (laid out per `space`) by the per-sector
/// `spectrum`, in place — the block-local realization of TensorKit's
/// `DiagonalTensorMap` multiplication. `axis = None` scales each block's
/// trailing axis (`t * D`, `rmul!`, column scaling); `axis = Some(0)` scales
/// the leading axis (`D * t`, `lmul!`, row scaling). Verified twist-free
/// against TK `diagonal.jl`: diagonal multiplication is pure per-block scaling
/// with no braiding or fusion-tree recoupling (`block(D, c)` is a `Diagonal`,
/// so LinearAlgebra dispatches to scaling, not GEMM). A real `spectrum` on a
/// complex `data` promotes each entry the same way (`D::from_real`).
#[cfg(test)]
pub(crate) fn scale_axis_by_spectrum<D>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum],
) -> Result<(), OperationError>
where
    D: FactorScalar,
{
    scale_axis_by_spectrum_mapped(space, data, axis, spectrum, D::from_real)
}

/// Value-generic sibling of `scale_axis_by_spectrum`. Why not convert the
/// spectrum before this call: a complex spectrum cannot pass through the
/// real-only `SectorSpectrum` alias without losing its imaginary component.
pub fn scale_axis_by_spectrum_mapped<D, V>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: impl Fn(V) -> D,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let nout = space.nout();
    let structure = Arc::clone(space.structure());
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let shape = block.shape();
        if shape.is_empty() {
            continue;
        }
        let strides = block.strides();
        let offset = block.offset();
        let bond_axis = axis.unwrap_or(shape.len() - 1);
        // Index the spectrum by the charge ON THE SCALED LEG — its uncoupled
        // charge in this block's fusion tree — NOT the block's coupled charge.
        // For an SVD/eigh factor's sole bond leg the two coincide, but scaling a
        // general tensor leg (diagonal-aware `contract`, #75) is only correct per
        // leg charge.
        let leg_charge = if bond_axis < nout {
            tree.codomain_tree().uncoupled()[bond_axis]
        } else {
            tree.domain_tree().uncoupled()[bond_axis - nout]
        };
        // Absent charge => this leg slice is structurally zero for the spectrum;
        // nothing to scale (mirrors `diagonal_bond_tensor_dyn`'s `unwrap_or(0)`).
        let Some(&entry) = spectrum_by_sector.get(&leg_charge) else {
            continue;
        };
        let bond = shape[bond_axis];
        let bond_stride = strides[bond_axis];
        debug_assert_eq!(
            bond,
            entry.values.len(),
            "bond degeneracy must match the spectrum length"
        );
        let bond = bond.min(entry.values.len());
        // Walk every combination of the non-bond axes; for each, scale the
        // `bond` entries along `bond_axis` by the spectrum.
        let lead_axes: Vec<usize> = (0..shape.len()).filter(|&a| a != bond_axis).collect();
        let outer: usize = lead_axes.iter().map(|&a| shape[a]).product();
        let mut coord = vec![0usize; lead_axes.len()];
        for _ in 0..outer {
            let mut base = offset;
            for (k, &a) in lead_axes.iter().enumerate() {
                base += coord[k] * strides[a];
            }
            for j in 0..bond {
                let scale = to_scalar(entry.values[j]);
                let idx = base + j * bond_stride;
                data[idx] = data[idx] * scale;
            }
            for k in (0..coord.len()).rev() {
                coord[k] += 1;
                if coord[k] < shape[lead_axes[k]] {
                    break;
                }
                coord[k] = 0;
            }
        }
    }
    Ok(())
}

/// `dst = beta · dst + alpha · scale_axis(src)`, where `scale_axis` is
/// [`scale_axis_by_spectrum_mapped`] on `src` (laid out per `space`, the
/// layout `dst` shares): TensorKit `mul!(C, A, Diagonal, α, β)` in one pass,
/// without the scaled copy of `src`. Every element of `dst` is written; one
/// [`scale_axis_by_spectrum_mapped`] leaves unscaled is `alpha · src`.
/// `beta == 0` never reads `dst` and `alpha == 0` never reads `src`.
#[allow(clippy::too_many_arguments)]
pub fn axpby_axis_by_spectrum_into<D, V>(
    space: &DynamicFusionMapSpace,
    dst: &mut [D],
    src: &[D],
    axis: Option<usize>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: impl Fn(V) -> D,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    V: Copy,
{
    if dst.len() != src.len() {
        return Err(OperationError::InvalidArgument {
            message: "scaled-axis axpby source and destination lengths differ",
        });
    }
    if alpha.is_zero() {
        if beta.is_zero() {
            dst.fill(D::zero());
        } else if beta != D::one() {
            for value in dst.iter_mut() {
                *value = beta * *value;
            }
        }
        return Ok(());
    }
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum<V>> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    let nout = space.nout();
    let structure = space.structure();
    // One coordinate buffer for every block. Why not a `SmallVec`: this
    // crate does not depend on `smallvec`, and one reused buffer already
    // makes the walk's allocations independent of the block count.
    let mut coord = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let shape = block.shape();
        let strides = block.strides();
        // The same per-leg-charge lookup as `scale_axis_by_spectrum_mapped`.
        let scaled = match block.key() {
            BlockKey::FusionTree(tree) if !shape.is_empty() => {
                let bond_axis = axis.unwrap_or(shape.len() - 1);
                let leg_charge = if bond_axis < nout {
                    tree.codomain_tree().uncoupled()[bond_axis]
                } else {
                    tree.domain_tree().uncoupled()[bond_axis - nout]
                };
                spectrum_by_sector
                    .get(&leg_charge)
                    .map(|&entry| (bond_axis, entry))
            }
            _ => None,
        };
        coord.clear();
        coord.resize(shape.len(), 0usize);
        let count: usize = shape.iter().product();
        for _ in 0..count {
            let idx = block.offset()
                + coord
                    .iter()
                    .zip(strides)
                    .map(|(&c, &stride)| c * stride)
                    .sum::<usize>();
            let mut value = src[idx];
            if let Some((bond_axis, entry)) = scaled {
                if let Some(&factor) = entry.values.get(coord[bond_axis]) {
                    value = value * to_scalar(factor);
                }
            }
            dst[idx] = if beta.is_zero() {
                alpha * value
            } else {
                beta * dst[idx] + alpha * value
            };
            for k in (0..coord.len()).rev() {
                coord[k] += 1;
                if coord[k] < shape[k] {
                    break;
                }
                coord[k] = 0;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
/// All singular values per coupled sector, descending (MatrixAlgebraKit
/// `svd_vals`). Runs the dense SVD per sector through the executor and keeps
/// only the spectra.
pub(crate) fn svd_vals<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_vals_dyn(dense, &input.dynamic())
}

/// Dynamic-rank `svd_vals`: the dense stage, the same in every fusion mode.
pub fn svd_vals_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    // Values-only: per coupled sector call the no-vector SVD (`svd_vals`,
    // LAPACK `job='N'`) and keep the spectrum. Unlike `svd_compact_dyn` this
    // never builds the U/Vt spaces, allocates the factor buffers, gauge-fixes,
    // or scatters blocks into the fusion-tree layout — all of which the old
    // `svd_compact_dyn(..).map(|svd| svd.singular_values)` computed then threw
    // away. Valid no-vector and full-factor drivers can differ in the last
    // bits, so comparisons use dtype-appropriate tolerances.
    let matricizations = value_matricizations(space.structure(), input.data(), space.nout())?;
    svd_vals_spectra(dense, &matricizations)
}

/// Singular values of every coupled-sector matrix through the no-vector SVD,
/// truncated to the sector rank.
pub(super) fn svd_vals_spectra<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut singular_values = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let rank = matrix.rows.min(matrix.cols);
        let input_shape = [matrix.rows, matrix.cols];
        let input_strides = [1usize, matrix.rows];
        let input = DenseView::new(matrix.data, &input_shape, &input_strides, 0)
            .map_err(OperationError::Dense)?;
        let s_tensor = dense
            .svd_vals(D::dense_read(input))
            .map_err(OperationError::Dense)?;
        let mut s = D::real_spectrum(&s_tensor).map_err(OperationError::Dense)?;
        s.truncate(rank);
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s,
        });
    }
    Ok(singular_values)
}

/// Singular values of a compact diagonal: MAK `svd_vals!(::DiagonalAlgorithm)`
/// (`abs` then a descending sort), per coupled sector.
fn svd_vals_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Vec<SectorSpectrum>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Svd)?;
    let mut values = Vec::with_capacity(bond.len());
    for region in bond.iter() {
        let mut sorted: Vec<f64> = bond
            .entry(region)
            .values
            .iter()
            .map(|&value| {
                D::from_real(value.widen_complex().norm())
                    .widen_complex()
                    .re
            })
            .collect();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        values.push(SectorSpectrum {
            sector: region.coupled(),
            values: sorted,
        });
    }
    Ok(values)
}

/// Singular values of `source` in fusion mode `M`, descending per coupled
/// sector. A compact diagonal is read directly; only dense storage leases an
/// executor.
#[doc(hidden)]
pub fn svd_vals_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        Some(FactorFamily::Svd),
        |space, spectrum| svd_vals_diagonal(&M::authority(space), space, spectrum),
        |dense, input| Ok(svd_vals_dyn(dense, input)?),
    )
}

#[cfg(test)]
/// Compact (untruncated) fusion-tensor SVD through the device boundary.
pub(crate) fn svd_compact<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<SvdCompact<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = svd_compact_dyn(dense, &input.dynamic())?;
    Ok(SvdCompact {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
        singular_values: out.singular_values,
    })
}

/// The compact-SVD factors without materializing the diagonal `S`:
/// `(U, Vh, spectrum)`. The shared core of every SVD entry point.
/// `svd_compact_dyn` wraps this and adds the dense `S` as a tensor for callers
/// that want it; polar and the matrix-function paths scale by the spectrum
/// directly (TensorKit `DiagonalTensorMap` `rmul!`) and never build `S`.
pub type SvdFactorsDyn<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
);

/// Compact diagonal input, MAK `svd_full!(::DiagonalAlgorithm)`: sort each
/// sector's magnitudes and write the permutation/phase factors directly into
/// the dense factor layout. Compact and full SVD coincide on a diagonal.
#[doc(hidden)]
pub(crate) fn svd_compact_diagonal_factors_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let bond = diagonal_bond(
        &MfAuthority(authority),
        authority,
        spectrum,
        FactorFamily::Svd,
    )?;
    let authority = bond.space();
    let plan =
        compact_factor_plan(authority)?.ok_or(OperationError::UnsupportedTensorContractScope {
            message: "canonical bond layout has no compact factor plan",
        })?;
    let (u_data, vh_data, singular_values) = compact_diagonal_svd_factor_data(
        &plan.routes,
        plan.left_regions.len(),
        plan.right_regions.len(),
        plan.left_layout.required_len()?,
        plan.right_layout.required_len()?,
        &bond.by_sector,
    )?;
    let u = BoundDynFactor::from_bound(
        authority.rebind_validated(&plan.left_layout)?,
        u_data,
        authority.space().nout(),
        1,
    )?;
    let vh = BoundDynFactor::from_bound(
        authority.rebind_validated(&plan.right_layout)?,
        vh_data,
        1,
        authority.space().nin(),
    )?;
    Ok((u, vh, singular_values))
}

#[expect(clippy::type_complexity)]
fn compact_diagonal_svd_factor_data<D: FactorScalar>(
    routes: &[CompactFactorRoute],
    left_region_count: usize,
    right_region_count: usize,
    left_len: usize,
    right_len: usize,
    by_sector: &FxHashMap<SectorId, &SectorSpectrum<D>>,
) -> Result<(Vec<D>, Vec<D>, Vec<SectorSpectrum>), OperationError> {
    if by_sector.len() != routes.len() {
        return Err(OperationError::InvalidArgument {
            message: "compact diagonal spectrum does not match source sectors",
        });
    }
    let mut u_regions = vec![None; left_region_count];
    let mut vh_regions = vec![None; right_region_count];
    let mut singular_values = Vec::with_capacity(routes.len());
    for route in routes {
        let entry = by_sector
            .get(&route.sector)
            .ok_or(OperationError::InvalidArgument {
                message: "compact diagonal spectrum is missing a source sector",
            })?;
        let k = route.rank;
        if entry.values.len() != k {
            return Err(OperationError::InvalidArgument {
                message: "compact diagonal spectrum length does not match source region",
            });
        }
        let (u, vh, values) = compact_diagonal_svd_sector(&entry.values);
        if k != 0 {
            u_regions[route.left_region.expect("nonzero route has left region")] = Some(u);
            vh_regions[route.right_region.expect("nonzero route has right region")] = Some(vh);
        }
        singular_values.push(SectorSpectrum {
            sector: route.sector,
            values,
        });
    }
    let u_data = concat_compact_svd_factor_regions(u_regions, left_len);
    let vh_data = concat_compact_svd_factor_regions(vh_regions, right_len);
    Ok((u_data, vh_data, singular_values))
}

fn compact_diagonal_svd_sector<D: FactorScalar>(values: &[D]) -> (Vec<D>, Vec<D>, Vec<f64>) {
    let k = values.len();
    let mut order = values
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value.widen_complex().norm()))
        .collect::<Vec<_>>();
    // MAK `sortperm(Ad; by = abs, rev = true)`, stable.
    order.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut u = vec![D::zero(); k * k];
    let mut vh = vec![D::zero(); k * k];
    for (column, &(source, _)) in order.iter().enumerate() {
        u[source + column * k] = D::from_real(1.0);
        vh[column + source * k] = diagonal_phase_magnitude(values[source]).0;
    }
    (
        u,
        vh,
        order.into_iter().map(|(_, magnitude)| magnitude).collect(),
    )
}

/// Checked-provider diagonal SVD factors, MAK `svd_full!(::DiagonalAlgorithm)`;
/// outputs use the usual checked factor builder.
pub(super) fn svd_compact_diagonal_factors_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let bond = diagonal_bond(
        &CheckedAuthority(authority.provider_arc()),
        authority,
        spectrum,
        FactorFamily::Svd,
    )?;
    let authority = bond.space();
    let source = authority.space();
    let source_regions = &bond.regions;
    let mut pairs = Vec::with_capacity(source_regions.len());
    let mut singular_values = Vec::with_capacity(source_regions.len());
    for region in source_regions.iter() {
        let entry = bond.entry(region);
        let (left, right, values) = compact_diagonal_svd_sector(&entry.values);
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: region.rows(),
            left,
            left_rows: region.rows(),
            right,
            right_leading: region.rows(),
        });
        singular_values.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    if factor_bond_is_input_bond(source, || {
        SectorLeg::new(pairs.iter().map(|pair| (pair.sector, pair.kept)), false)
    }) {
        let (left, right): (Vec<_>, Vec<_>) = pairs
            .into_iter()
            .map(|pair| (pair.left, pair.right))
            .unzip();
        let u = factor_on_input_space(authority, source_regions, left)?;
        let vh = factor_on_input_space(authority, source_regions, right)?;
        return Ok((u, vh, singular_values));
    }
    let (u, vh) = build_left_right_bound_pair_generic_checked(
        authority.provider_arc(),
        source.homspace(),
        source_regions,
        pairs,
    )?;
    Ok((u, vh, singular_values))
}

pub(crate) fn svd_compact_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_compact_factors_dyn_with_gauge(dense, input, CompactSvdGauge::Left)
}

/// Compact SVD factors for the logical adjoint without constructing its input:
/// if `A = U S Vh`, returns `(V, Uh, spectrum)` with the phase gauge applied
/// to the final left factor `V`.
#[doc(hidden)]
pub(crate) fn svd_compact_adjoint_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_compact_factors_dyn_with_gauge(dense, input, CompactSvdGauge::AdjointLeft)
}

pub(super) struct CompactSvdNumericalStage<D> {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) rank: usize,
    pub(super) u: Vec<D>,
    pub(super) singular_values: Vec<f64>,
    pub(super) vt: Vec<D>,
}

pub(super) fn compact_svd_numerical_stage<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<CompactSvdNumericalStage<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    compact_svd_stage(
        dense,
        matrix,
        rows,
        cols,
        CompactSvdGauge::Left,
        |_u, _vt| {
            #[cfg(test)]
            crate::probe::checked_svd_stage(_u, _vt);
        },
    )
}

/// Compact SVD of one coupled-sector matrix in `gauge`; an empty sector makes
/// no dense call. `observe` sees the gauged `u` and `vt`.
pub(super) fn compact_svd_stage<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
    gauge: CompactSvdGauge,
    observe: impl FnOnce(&[D], &[D]),
) -> Result<CompactSvdNumericalStage<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let rank = rows.min(cols);
    if rank == 0 {
        return Ok(CompactSvdNumericalStage {
            rows,
            cols,
            rank,
            u: Vec::new(),
            singular_values: Vec::new(),
            vt: Vec::new(),
        });
    }
    let (mut u, singular_values, mut vt) = compact_svd_owned(dense, matrix, rows, cols)?;
    match gauge {
        CompactSvdGauge::Left => svd_compact_gauge(&mut u, rows, rows, &mut vt, rank, cols, rank),
        CompactSvdGauge::AdjointLeft => {
            svd_compact_adjoint_gauge(&mut u, rows, rows, &mut vt, rank, cols, rank)
        }
    }
    observe(&u, &vt);
    Ok(CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        singular_values,
        vt,
    })
}

#[cfg(test)]
pub(crate) fn compact_svd_numerical_stage_lengths_for_test<E, D>(
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
) -> Result<(usize, usize, usize), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let stage = compact_svd_numerical_stage(dense, matrix, rows, cols)?;
    Ok((stage.u.len(), stage.singular_values.len(), stage.vt.len()))
}

#[derive(Clone, Copy)]
pub(super) enum CompactSvdGauge {
    Left,
    AdjointLeft,
}

fn svd_compact_factors_dyn_with_gauge<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    gauge: CompactSvdGauge,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    if let Some(plan) = compact_factor_plan(input.space())? {
        let adjoint_spaces = if matches!(gauge, CompactSvdGauge::AdjointLeft) {
            let adjoint = input.space().adjoint_view()?;
            let bond = compact_bond_leg(&plan.source_regions);
            Some((
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond.clone(),
                    FactorSide::Left,
                )?,
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond,
                    FactorSide::Right,
                )?,
            ))
        } else {
            None
        };
        return svd_compact_direct_regions(dense, input, &plan, gauge, adjoint_spaces);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    crate::probe::input_pack(
        crate::probe::CopyOp::Svd,
        matricizations
            .iter()
            .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
    );

    let ranks = matricizations
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector,
            kept: matrix.rows.min(matrix.cols),
        })
        .collect::<Vec<_>>();
    let bond = SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false);
    let (left_space, right_space) = match gauge {
        CompactSvdGauge::Left => (
            build_bound_factor_space(
                input.space(),
                space.homspace(),
                bond.clone(),
                FactorSide::Left,
            )?,
            build_bound_factor_space(input.space(), space.homspace(), bond, FactorSide::Right)?,
        ),
        CompactSvdGauge::AdjointLeft => {
            let adjoint = input.space().adjoint_view()?;
            (
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond.clone(),
                    FactorSide::Left,
                )?,
                build_bound_factor_space(
                    &adjoint,
                    adjoint.space().homspace(),
                    bond,
                    FactorSide::Right,
                )?,
            )
        }
    };
    let u_len = left_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut u_data = vec![D::zero(); u_len];
    let vt_len = right_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut vt_data = vec![D::zero(); vt_len];

    let mut singular_values = Vec::with_capacity(matricizations.len());

    let index = PlacementIndex::new(&matricizations, &[FactorSide::Left, FactorSide::Right]);
    let u_groups = SectorBlockGroups::new(left_space.space().structure(), FactorSide::Left)?;
    let vt_groups = SectorBlockGroups::new(right_space.space().structure(), FactorSide::Right)?;
    let (u_target, vt_target) = (left_space.space(), right_space.space());
    in_linalg_scope(dense, |dense| {
        for matrix in &matricizations {
            let CompactSvdNumericalStage {
                rank,
                u,
                singular_values: values,
                vt,
                ..
            } = compact_svd_stage(
                dense,
                &matrix.data,
                matrix.rows,
                matrix.cols,
                gauge,
                |_u, _vt| {
                    #[cfg(test)]
                    crate::probe::mf_svd_fallback(_u, _vt);
                },
            )?;
            singular_values.push(SectorSpectrum {
                sector: matrix.sector,
                values,
            });
            match gauge {
                CompactSvdGauge::Left => scatter_left_sector_blocks(
                    u_target,
                    &mut u_data,
                    matrix,
                    &index,
                    &u_groups,
                    &u,
                    matrix.rows,
                )?,
                CompactSvdGauge::AdjointLeft => scatter_adjoint_svd_factor(
                    u_target,
                    &mut u_data,
                    matrix,
                    &index,
                    &u_groups,
                    &vt,
                    rank,
                    FactorSide::Left,
                )?,
            }
            #[cfg(test)]
            crate::probe::output_scatter::<D>(crate::probe::CopyOp::Svd, 1, matrix.rows * rank);
            match gauge {
                CompactSvdGauge::Left => scatter_right_sector_blocks(
                    vt_target,
                    &mut vt_data,
                    matrix,
                    &index,
                    &vt_groups,
                    &vt,
                    rank,
                )?,
                CompactSvdGauge::AdjointLeft => scatter_adjoint_svd_factor(
                    vt_target,
                    &mut vt_data,
                    matrix,
                    &index,
                    &vt_groups,
                    &u,
                    matrix.rows,
                    FactorSide::Right,
                )?,
            }
            #[cfg(test)]
            crate::probe::output_scatter::<D>(crate::probe::CopyOp::Svd, 1, rank * matrix.cols);
        }
        Ok(())
    })?;

    let (u_nout, u_nin) = (u_target.nout(), u_target.nin());
    let (vh_nout, vh_nin) = (vt_target.nout(), vt_target.nin());
    let u = BoundDynFactor::from_bound(left_space, u_data, u_nout, u_nin)?;
    let vh = BoundDynFactor::from_bound(right_space, vt_data, vh_nout, vh_nin)?;
    Ok((u, vh, singular_values))
}

pub(super) fn svd_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
    gauge: CompactSvdGauge,
    adjoint_spaces: Option<(BoundDynamicFusionMapSpace<R>, BoundDynamicFusionMapSpace<R>)>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let (left_space, right_space) = match adjoint_spaces {
        Some(spaces) => spaces,
        None => (
            input.space().rebind_validated(&plan.left_layout)?,
            input.space().rebind_validated(&plan.right_layout)?,
        ),
    };
    let adjoint_routes = if matches!(gauge, CompactSvdGauge::AdjointLeft) {
        let left =
            checked_sector_regions(left_space.space().structure(), left_space.space().nout())?;
        let right =
            checked_sector_regions(right_space.space().structure(), right_space.space().nout())?;
        match (left, right) {
            (Some(left), Some(right)) => {
                let left_by_sector = SectorRegionIndex::new(&left)?;
                let right_by_sector = SectorRegionIndex::new(&right)?;
                let mut routes = Vec::with_capacity(plan.routes.len());
                let mut compatible = true;
                for route in &plan.routes {
                    let source = &plan.source_regions[route.source_region];
                    let pair = left_by_sector
                        .get(route.sector)
                        .zip(right_by_sector.get(route.sector));
                    if let Some((li, ri)) = pair {
                        compatible &= left[li].row_trees() == source.col_trees()
                            && right[ri].col_trees() == source.row_trees()
                            && left[li].rows() == source.cols()
                            && right[ri].cols() == source.rows();
                        routes.push((li, ri));
                    } else {
                        compatible = false;
                        break;
                    }
                }
                compatible.then_some((routes, left.len(), right.len()))
            }
            _ => None,
        }
    } else {
        None
    };
    let reuse_regions = matches!(gauge, CompactSvdGauge::Left) || adjoint_routes.is_some();
    let mut u_regions = if reuse_regions {
        vec![
            None;
            adjoint_routes
                .as_ref()
                .map_or(plan.left_regions.len(), |r| r.1)
        ]
    } else {
        Vec::new()
    };
    let mut vh_regions = if reuse_regions {
        vec![
            None;
            adjoint_routes
                .as_ref()
                .map_or(plan.right_regions.len(), |r| r.2)
        ]
    } else {
        Vec::new()
    };
    let mut singular_values = Vec::with_capacity(plan.routes.len());
    let mut left_data = if matches!(gauge, CompactSvdGauge::AdjointLeft) && !reuse_regions {
        vec![D::zero(); left_space.space().required_len()?]
    } else {
        Vec::new()
    };
    let mut right_data = if matches!(gauge, CompactSvdGauge::AdjointLeft) && !reuse_regions {
        vec![D::zero(); right_space.space().required_len()?]
    } else {
        Vec::new()
    };
    let adjoint_replay = if matches!(gauge, CompactSvdGauge::AdjointLeft) && !reuse_regions {
        Some((
            PlacementIndex::new(&plan.source_regions, &[FactorSide::Left, FactorSide::Right]),
            SectorBlockGroups::new(left_space.space().structure(), FactorSide::Left)?,
            SectorBlockGroups::new(right_space.space().structure(), FactorSide::Right)?,
        ))
    } else {
        None
    };

    let blocks = plan
        .routes
        .iter()
        .filter(|route| route.rank != 0)
        .map(|route| {
            let region = &plan.source_regions[route.source_region];
            (&input.data()[region.range()], region.rows(), region.cols())
        })
        .collect::<Vec<_>>();
    let mut factors = compact_svd_owned_batch(dense, &blocks)?.into_iter();

    for (route_index, route) in plan.routes.iter().copied().enumerate() {
        let region = &plan.source_regions[route.source_region];
        let rank = route.rank;
        if rank == 0 {
            singular_values.push(SectorSpectrum {
                sector: route.sector,
                values: Vec::new(),
            });
            continue;
        }
        let left_region = route.left_region.expect("nonzero route has left region");
        let right_region = route.right_region.expect("nonzero route has right region");
        let (mut u, spectrum, mut vh) = factors
            .next()
            .expect("one batch output per nonzero-rank route");
        match gauge {
            CompactSvdGauge::Left => svd_compact_gauge(
                &mut u,
                region.rows(),
                region.rows(),
                &mut vh,
                rank,
                region.cols(),
                rank,
            ),
            CompactSvdGauge::AdjointLeft => svd_compact_adjoint_gauge(
                &mut u,
                region.rows(),
                region.rows(),
                &mut vh,
                rank,
                region.cols(),
                rank,
            ),
        }
        match gauge {
            CompactSvdGauge::Left => {
                u_regions[left_region] = Some(u);
                vh_regions[right_region] = Some(vh);
            }
            CompactSvdGauge::AdjointLeft => {
                if let Some((routes, _, _)) = &adjoint_routes {
                    adjoint_col_major_in_place(&mut vh, rank, region.cols());
                    adjoint_col_major_in_place(&mut u, region.rows(), rank);
                    let (li, ri) = routes[route_index];
                    u_regions[li] = Some(vh);
                    vh_regions[ri] = Some(u);
                } else {
                    let (index, left_groups, right_groups) = adjoint_replay
                        .as_ref()
                        .expect("adjoint replay is present for adjoint gauge");
                    scatter_adjoint_svd_factor(
                        left_space.space(),
                        &mut left_data,
                        region,
                        index,
                        left_groups,
                        &vh,
                        rank,
                        FactorSide::Left,
                    )?;
                    scatter_adjoint_svd_factor(
                        right_space.space(),
                        &mut right_data,
                        region,
                        index,
                        right_groups,
                        &u,
                        region.rows(),
                        FactorSide::Right,
                    )?;
                }
            }
        }
        singular_values.push(SectorSpectrum {
            sector: route.sector,
            values: spectrum,
        });
    }

    let (u_data, vh_data) = match gauge {
        CompactSvdGauge::Left => (
            concat_compact_svd_factor_regions(u_regions, plan.left_layout.required_len()?),
            concat_compact_svd_factor_regions(vh_regions, plan.right_layout.required_len()?),
        ),
        CompactSvdGauge::AdjointLeft if reuse_regions => (
            concat_compact_svd_factor_regions(u_regions, left_space.space().required_len()?),
            concat_compact_svd_factor_regions(vh_regions, right_space.space().required_len()?),
        ),
        CompactSvdGauge::AdjointLeft => (left_data, right_data),
    };
    let u = BoundDynFactor::from_bound(
        left_space,
        u_data,
        match gauge {
            CompactSvdGauge::Left => space.nout(),
            CompactSvdGauge::AdjointLeft => space.nin(),
        },
        1,
    )?;
    let vh = BoundDynFactor::from_bound(
        right_space,
        vh_data,
        1,
        match gauge {
            CompactSvdGauge::Left => space.nin(),
            CompactSvdGauge::AdjointLeft => space.nout(),
        },
    )?;
    Ok((u, vh, singular_values))
}

#[allow(clippy::too_many_arguments)]
fn scatter_adjoint_svd_factor<D, M>(
    target: &DynamicFusionMapSpace,
    data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    source_rows: usize,
    side: FactorSide,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let source_side = match side {
        FactorSide::Left => FactorSide::Right,
        FactorSide::Right => FactorSide::Left,
    };
    for block_index in groups.blocks(matrix.sector()) {
        let block = target.structure().block(block_index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let tree = match side {
            FactorSide::Left => key.codomain_tree(),
            FactorSide::Right => key.domain_tree(),
        };
        let (side_offset, _) = index.placement(matrix.sector(), source_side, tree)?;
        let shape = block.shape();
        let strides = block.strides();
        let bond_axis = match side {
            FactorSide::Left => shape.len() - 1,
            FactorSide::Right => 0,
        };
        let side_extent = shape
            .iter()
            .enumerate()
            .filter(|&(axis, _)| axis != bond_axis)
            .map(|(_, &extent)| extent)
            .product::<usize>();
        for side_index in 0..side_extent {
            let mut remaining = side_index;
            let mut target_offset = block.offset();
            for axis in 0..shape.len() {
                if axis != bond_axis {
                    let coordinate = remaining % shape[axis];
                    remaining /= shape[axis];
                    target_offset += coordinate * strides[axis];
                }
            }
            for bond in 0..shape[bond_axis] {
                let source_index = match side {
                    FactorSide::Left => bond + source_rows * (side_offset + side_index),
                    FactorSide::Right => side_offset + side_index + source_rows * bond,
                };
                data[target_offset + bond * strides[bond_axis]] =
                    FactorScalar::adjoint(factor[source_index]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
/// Dynamic-rank [`svd_compact`]: the [`svd_compact_factors_dyn`] core plus the
/// diagonal `S` materialized as a `bond <- bond` tensor.
pub(crate) fn svd_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdCompactDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn(dense, input)?;
    let s = diagonal_bond_svd_factor(input.space(), &singular_values, &D::from_real)?;
    Ok(SvdCompactDyn {
        u,
        s,
        vh,
        singular_values,
    })
}

/// Host-side truncation decision over the spectra of a bond factorization:
/// the selection magnitude is `|value|`, each `spectra` entry is in the
/// factorization's stored order (any order), and the kept set is a per-sector
/// mask over those stored positions (TensorKit's `findtruncated`).
///
/// The mode `M` supplies the `dim(c)` weight (exact `try_dim_scalar` on the
/// checked tier, #1871). The rule identity and sector order are arguments
/// because the multiplicity-free and checked providers expose them through
/// different traits (`FusionRule` and `CheckedGenericFusion`).
///
/// Public and `doc(hidden)` only because the typed facade's
/// `GradedSpace::find_truncated`, in the `symtn` crate, is its caller; it is
/// not an API of its own.
#[doc(hidden)]
pub fn decide_bond_truncation<M, R, V>(
    rule: &R,
    spectra: &[SectorSpectrum<V>],
    truncation: &Truncation,
    rule_identity: &RuleIdentity,
    sector_order_key: impl Fn(SectorId) -> SectorOrderKey,
) -> Result<crate::truncation::TruncationDecision, M::Error>
where
    M: RigidCoefficientAlgebra<R>,
    V: SpectrumMagnitude,
{
    let magnitudes: Vec<Vec<f64>> = spectra
        .iter()
        .map(|entry| entry.values.iter().map(|value| value.magnitude()).collect())
        .collect();
    let mut weighted = Vec::with_capacity(spectra.len());
    for (entry, values) in spectra.iter().zip(&magnitudes) {
        weighted.push(WeightedSpectrum {
            sector: entry.sector,
            weight: M::dim(rule, entry.sector)?,
            values,
        });
    }
    select_truncation(&weighted, truncation, rule_identity, sector_order_key)
        .map_err(|error| M::Error::from(OperationError::from(error)))
}

#[cfg(test)]
/// Full fusion-tensor SVD `t = U * S * Vh` (MatrixAlgebraKit `svd_full`):
/// per sector `U` is the square `m x m` unitary, `S` the rectangular
/// `m x n` diagonal, and `Vh` the square `n x n` unitary.
#[derive(Clone, Debug)]
pub(crate) struct SvdFull<R, D> {
    pub u: BoundTensorMap<R, D>,
    pub s: BoundTensorMap<R, D>,
    pub vh: BoundTensorMap<R, D>,
}

#[cfg(test)]
/// Dynamic-rank [`SvdFull`].
#[derive(Clone, Debug)]
pub(crate) struct SvdFullDyn<R, D> {
    pub(super) u: BoundDynFactor<R, D>,
    pub(super) s: BoundDynFactor<R, D>,
    pub(super) vh: BoundDynFactor<R, D>,
    pub(super) singular_values: Vec<SectorSpectrum>,
}

/// Numerical full-SVD factors before choosing the storage of `S`.
#[doc(hidden)]
pub struct SvdFullFactorsDyn<R, D> {
    u: BoundDynFactor<R, D>,
    vh: BoundDynFactor<R, D>,
    singular_values: Vec<SectorSpectrum>,
    row_dimensions: BTreeMap<SectorId, usize>,
    col_dimensions: BTreeMap<SectorId, usize>,
    #[cfg(test)]
    adjoint_space: Option<BoundDynamicFusionMapSpace<R>>,
}

impl<R, D> SvdFullFactorsDyn<R, D> {
    #[expect(
        clippy::type_complexity,
        reason = "full-SVD factors retain both output bond maps"
    )]
    pub fn into_parts(
        self,
    ) -> (
        BoundDynFactor<R, D>,
        BoundDynFactor<R, D>,
        Vec<SectorSpectrum>,
        BTreeMap<SectorId, usize>,
        BTreeMap<SectorId, usize>,
    ) {
        (
            self.u,
            self.vh,
            self.singular_values,
            self.row_dimensions,
            self.col_dimensions,
        )
    }
}

#[cfg(test)]
impl<R, D> SvdFullDyn<R, D> {
    pub fn u(&self) -> &BoundDynFactor<R, D> {
        &self.u
    }
    pub fn s(&self) -> &BoundDynFactor<R, D> {
        &self.s
    }
    pub fn vh(&self) -> &BoundDynFactor<R, D> {
        &self.vh
    }
    pub fn singular_values(&self) -> &[SectorSpectrum] {
        &self.singular_values
    }
}

#[cfg(test)]
/// Full fusion-tensor SVD through the device boundary.
///
/// A provider that advertises [`DenseExecutor::supports_svd_full`] factorizes
/// the full matrices directly, one call per sector. Otherwise the unitaries
/// are completed from the compact factors with an extra economy QR of
/// `[U1 | I]` per sector (any orthonormal completion is exact because the
/// corresponding rows/columns of `S` are zero), so every provider stays on the
/// existing dense-executor boundary. The two routes agree on the reconstructed
/// input, the spectrum and unitarity, but the basis spanning the null space is
/// not unique and differs between them.
pub(crate) fn svd_full<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<SvdFull<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = svd_full_dyn(dense, &input.dynamic())?;
    Ok(SvdFull {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
    })
}

#[cfg(test)]
/// Dynamic-rank [`svd_full`].
pub(crate) fn svd_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_dyn(dense, input, FactorPlacement::Direct)
}

/// Full-SVD numerical factors without publishing a dense `S` tensor.
#[doc(hidden)]
pub(crate) fn svd_full_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn(dense, input, FactorPlacement::Direct)
}

#[doc(hidden)]
pub(crate) fn svd_full_adjoint_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn(dense, input, FactorPlacement::Adjoint)
}

#[cfg(test)]
/// Full SVD factors for the logical adjoint without constructing its input.
pub(crate) fn svd_full_adjoint_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_dyn(dense, input, FactorPlacement::Adjoint)
}

#[expect(
    clippy::type_complexity,
    reason = "the private stage returns the fixed dense full-SVD tuple without another wrapper"
)]
pub(super) fn owned_full_svd_stage<E, D>(
    dense: &mut E,
    data: &mut Vec<D>,
    rows: usize,
    cols: usize,
) -> Result<Option<(Vec<D>, Vec<f64>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if !dense.supports_svd_full() {
        return Ok(None);
    }
    let input = match D::dense_into_owned(std::mem::take(data)) {
        Ok(input) => input,
        Err(input_data) => {
            *data = input_data;
            return Ok(None);
        }
    };
    let mut outputs = dense
        .svd_full_owned(input, rows, cols)
        .map_err(OperationError::Dense)?;
    if outputs.len() != 3 {
        return Err(OperationError::Dense(arity_mismatch(
            "svd_full_owned",
            3,
            outputs.len(),
        )));
    }
    let u = compact_factor_output_owned(outputs.remove(0), &[rows, rows], "svd_full_owned")?;
    let singular_values =
        compact_real_spectrum_owned::<D>(outputs.remove(0), &[rows.min(cols)], "svd_full_owned")?;
    let vh = compact_factor_output_owned(outputs.remove(0), &[cols, cols], "svd_full_owned")?;
    Ok(Some((u, singular_values, vh)))
}

/// The right factor of one full-SVD sector: the dense provider's `Vh`, or
/// the completed `V` of the thin-SVD fallback. Callers place whichever they
/// need, so neither orientation pays an extra adjoint.
pub(super) enum FullSvdRight<D> {
    Vh(Vec<D>),
    V(Vec<D>),
}

/// Thin-SVD scratch for the full-SVD fallback, sized once for the largest
/// sector and allocated on first use.
pub(super) struct FullSvdWorkspace<D: FactorScalar> {
    max_rows: usize,
    max_cols: usize,
    max_rank: usize,
    u: Vec<D>,
    s: Vec<D::Real>,
    vt: Vec<D>,
}

impl<D: FactorScalar> FullSvdWorkspace<D> {
    pub(super) fn for_sectors(shapes: impl Iterator<Item = (usize, usize)>) -> Self {
        let (mut max_rows, mut max_cols, mut max_rank) = (0, 0, 0);
        for (rows, cols) in shapes {
            max_rows = max_rows.max(rows);
            max_cols = max_cols.max(cols);
            max_rank = max_rank.max(rows.min(cols));
        }
        Self {
            max_rows,
            max_cols,
            max_rank,
            u: Vec::new(),
            s: Vec::new(),
            vt: Vec::new(),
        }
    }
}

/// Full SVD of one `rows x cols` coupled-sector matrix, ungauged: the
/// provider's owned full SVD when available (it takes `data`), otherwise the
/// thin `svd_into` completed to square factors.
#[allow(clippy::type_complexity)]
pub(super) fn full_svd_numerical_stage<E, D>(
    dense: &mut E,
    data: &mut Vec<D>,
    rows: usize,
    cols: usize,
    workspace: &mut FullSvdWorkspace<D>,
) -> Result<(Vec<D>, Vec<f64>, FullSvdRight<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if let Some((u_full, s_values, vh_full)) = owned_full_svd_stage(dense, data, rows, cols)? {
        return Ok((u_full, s_values, FullSvdRight::Vh(vh_full)));
    }
    let rank = rows.min(cols);
    let FullSvdWorkspace {
        max_rows,
        max_cols,
        max_rank,
        ..
    } = *workspace;
    if workspace.u.is_empty() && max_rows != 0 && max_rank != 0 {
        workspace.u = vec![D::zero(); max_rows * max_rank];
        workspace.s = vec![D::Real::zero(); max_rank];
        workspace.vt = vec![D::zero(); max_rank * max_cols];
    }
    let shape = [rows, cols];
    let strides = [1usize, rows];
    let view = DenseView::new(data, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let u_shape = [rows, rank];
    let u_strides = [1usize, max_rows];
    let s_shape = [rank];
    let s_strides = [1usize];
    let vt_shape = [rank, cols];
    let vt_strides = [1usize, max_rank];
    let u_view = DenseViewMut::new(&mut workspace.u, &u_shape, &u_strides, 0)
        .map_err(OperationError::Dense)?;
    let s_view = DenseViewMut::new(&mut workspace.s, &s_shape, &s_strides, 0)
        .map_err(OperationError::Dense)?;
    let vt_view = DenseViewMut::new(&mut workspace.vt, &vt_shape, &vt_strides, 0)
        .map_err(OperationError::Dense)?;
    dense
        .svd_into(
            D::dense_read(view),
            D::dense_write(u_view),
            D::Real::dense_write(s_view),
            D::dense_write(vt_view),
        )
        .map_err(OperationError::Dense)?;
    let mut u_thin = vec![D::zero(); rows * rank];
    let mut vt_thin = vec![D::zero(); rank * cols];
    copy_col_major_strided(&workspace.u, rows, rank, max_rows, &mut u_thin, rows);
    copy_col_major_strided(&workspace.vt, rank, cols, max_rank, &mut vt_thin, rank);
    let u_full = orthonormal_completion(dense, &u_thin, rows, rank)?;
    let v_thin = adjoint_col_major(&vt_thin, rank, cols);
    let v_full = orthonormal_completion(dense, &v_thin, cols, rank)?;
    let s_values = workspace.s[..rank]
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    Ok((u_full, s_values, FullSvdRight::V(v_full)))
}

#[cfg(test)]
pub(super) fn svd_full_oriented_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let parts = svd_full_oriented_factors_dyn(dense, input, placement)?;
    let authority = parts.adjoint_space.as_ref().unwrap_or(input.space());
    let s = rectangular_diagonal_bond_tensor(
        &MfAuthority(authority),
        &parts.singular_values,
        &parts.row_dimensions,
        &parts.col_dimensions,
    )?;
    Ok(SvdFullDyn {
        u: parts.u,
        s,
        vh: parts.vh,
        singular_values: parts.singular_values,
    })
}

fn svd_full_oriented_factors_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    // Why pack rather than borrow admitted regions: `owned_full_svd_stage`
    // moves each matrix into the dense provider, so a borrowed region would
    // be copied into an owned buffer anyway.
    let mut matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    let authority = MfAuthority(input.space());
    let row_dimensions = authority.coupled_dimensions(space.homspace().codomain())?;
    let col_dimensions = authority.coupled_dimensions(space.homspace().domain())?;

    let mut pairs = Vec::with_capacity(matricizations.len());
    let mut singular_values = Vec::with_capacity(matricizations.len());
    let mut workspace = FullSvdWorkspace::for_sectors(
        matricizations
            .iter()
            .map(|matrix| (matrix.rows, matrix.cols)),
    );
    for matrix in &mut matricizations {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let (mut u_full, s_values, right) =
            full_svd_numerical_stage(dense, &mut matrix.data, rows, cols, &mut workspace)?;
        let (mut left, left_rows, mut right, right_leading) = match (placement, right) {
            (FactorPlacement::Direct, FullSvdRight::Vh(vh_full)) => (u_full, rows, vh_full, cols),
            (FactorPlacement::Direct, FullSvdRight::V(v_full)) => {
                (u_full, rows, adjoint_col_major(&v_full, cols, cols), cols)
            }
            (FactorPlacement::Adjoint, FullSvdRight::Vh(mut vh_full)) => {
                adjoint_square_col_major_in_place(&mut vh_full, cols);
                adjoint_square_col_major_in_place(&mut u_full, rows);
                (vh_full, cols, u_full, rows)
            }
            (FactorPlacement::Adjoint, FullSvdRight::V(v_full)) => {
                adjoint_square_col_major_in_place(&mut u_full, rows);
                (v_full, cols, u_full, rows)
            }
        };
        svd_full_gauge(
            &mut left,
            left_rows,
            left_rows,
            &mut right,
            right_leading,
            right_leading,
        );

        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: left_rows,
            left,
            left_rows,
            right,
            right_leading,
        });
    }

    let adjoint_space = match placement {
        FactorPlacement::Direct => None,
        FactorPlacement::Adjoint => Some(<MultiplicityFreeAdmissionMode as CoefficientAlgebra<
            R,
        >>::adjoint_space(input.space())?),
    };
    let authority = adjoint_space.as_ref().unwrap_or(input.space());
    let homspace = authority.space().homspace();
    let (output_row_dimensions, output_col_dimensions) = match placement {
        FactorPlacement::Direct => (&row_dimensions, &col_dimensions),
        FactorPlacement::Adjoint => (&col_dimensions, &row_dimensions),
    };

    // The left/right bond legs differ in the full SVD (rows vs columns), so
    // build the two factors with separate bond dimensions.
    let u_factor = build_bound_factor_with_placement(
        authority,
        homspace,
        &matricizations,
        &mut pairs,
        output_row_dimensions,
        FactorSide::Left,
        placement,
    )?;
    let vh_factor = build_bound_factor_with_placement(
        authority,
        homspace,
        &matricizations,
        &mut pairs,
        output_col_dimensions,
        FactorSide::Right,
        placement,
    )?;
    let (row_dimensions, col_dimensions) = match placement {
        FactorPlacement::Direct => (row_dimensions, col_dimensions),
        FactorPlacement::Adjoint => (col_dimensions, row_dimensions),
    };
    Ok(SvdFullFactorsDyn {
        u: u_factor,
        vh: vh_factor,
        singular_values,
        row_dimensions,
        col_dimensions,
        #[cfg(test)]
        adjoint_space,
    })
}

fn adjoint_square_col_major_in_place<D: FactorScalar>(data: &mut [D], dimension: usize) {
    for column in 0..dimension {
        let diagonal = column * dimension + column;
        data[diagonal] = FactorScalar::adjoint(data[diagonal]);
        for row in 0..column {
            let a = row + dimension * column;
            let b = column + dimension * row;
            let value = data[a];
            data[a] = FactorScalar::adjoint(data[b]);
            data[b] = FactorScalar::adjoint(value);
        }
    }
}

fn adjoint_col_major_in_place<D: FactorScalar>(data: &mut [D], rows: usize, cols: usize) {
    if rows == cols {
        adjoint_square_col_major_in_place(data, rows);
        return;
    }
    if rows <= 1 || cols <= 1 {
        for value in data {
            *value = FactorScalar::adjoint(*value);
        }
        return;
    }
    let mut visited = vec![false; data.len()];
    for start in 0..data.len() {
        if visited[start] {
            continue;
        }
        let mut current = start;
        let mut value = data[current];
        loop {
            let next = current / rows + cols * (current % rows);
            let displaced = data[next];
            data[next] = FactorScalar::adjoint(value);
            visited[current] = true;
            current = next;
            value = displaced;
            if current == start {
                break;
            }
        }
    }
}

/// Completes `k` orthonormal columns (`m x k`, column-major) to a full
/// `m x m` orthonormal basis via [`augmented_identity_qr`] of `[Q1 | I]`;
/// the first `k` columns are returned unchanged.
pub(super) fn orthonormal_completion<E, D>(
    dense: &mut E,
    thin: &[D],
    rows: usize,
    rank: usize,
) -> Result<Vec<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    if rank == rows {
        return Ok(thin.to_vec());
    }
    let (mut full, _) = augmented_identity_qr(dense, rows, rank, |block| {
        block.copy_from_slice(thin);
    })?;
    full[..rows * rank].copy_from_slice(thin);
    Ok(full)
}

/// Rectangular diagonal `W_row <- W_col` bond factor (the `S` of the full
/// SVD): per sector shape `[rows, cols]` with the spectrum on the diagonal.
/// One body for both modes; `authority` owns the output space, so the mode
/// shows only in that space's derivation and in the error type.
#[doc(hidden)]
pub fn rectangular_diagonal_bond_tensor<A, R, D>(
    authority: &A,
    spectra: &[SectorSpectrum],
    row_dimensions: &BTreeMap<SectorId, usize>,
    col_dimensions: &BTreeMap<SectorId, usize>,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<A::RootError>,
    D: FactorScalar,
{
    let row_leg = SectorLeg::new(
        row_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let col_leg = SectorLeg::new(
        col_dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([row_leg]),
        FusionProductSpace::new([col_leg]),
    );
    let space = authority.output_space(homspace)?;
    let len = space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut data = vec![D::zero(); len];
    let structure = Arc::clone(space.space().structure());
    let spectrum_by_sector: FxHashMap<SectorId, &SectorSpectrum> =
        spectra.iter().map(|entry| (entry.sector, entry)).collect();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(tree) = block.key() else {
            continue;
        };
        let sector = tree.codomain_tree().coupled();
        let Some(&entry) = spectrum_by_sector.get(&sector) else {
            continue;
        };
        let strides = block.strides();
        let offset = block.offset();
        let count = block.shape()[0].min(block.shape()[1]);
        for position in 0..count {
            let Some(&value) = entry.values.get(position) else {
                break;
            };
            data[offset + position * (strides[0] + strides[1])] = D::from_real(value);
        }
    }
    Ok(BoundDynFactor::from_bound(space, data, 1, 1)?)
}

/// Positive-diagonal gauge (MatrixAlgebraKit `positive = true`, the default
/// of the Householder QR/LQ algorithms since MAK 0.6.8 / TensorKit 0.17):
/// absorbs the unitary phase `D = diag(phase(R_jj))` into `Q`, i.e.
/// `Q -> Q * D`, `R -> D^H * R`, leaving `Q * R` unchanged with real
/// non-negative `R_jj`. Zero diagonal entries keep phase `1`, exactly like
/// MAK `sign_safe` (no epsilon threshold).
///
/// `q` is column-major `q_rows x nq` (`nq >= min(r_rows, r_cols)`), `r` is
/// column-major `r_rows x r_cols`.
pub(crate) fn positive_diagonal_gauge<D: FactorScalar>(
    q: &mut [D],
    q_rows: usize,
    r: &mut [D],
    r_rows: usize,
    r_cols: usize,
) {
    positive_diagonal_gauge_strided(q, q_rows, q_rows, r, r_rows, r_rows, r_cols);
}

pub(super) fn positive_diagonal_gauge_strided<D: FactorScalar>(
    q: &mut [D],
    q_rows: usize,
    q_leading: usize,
    r: &mut [D],
    r_rows: usize,
    r_leading: usize,
    r_cols: usize,
) {
    for j in 0..r_rows.min(r_cols) {
        let z = r[j + r_leading * j].widen_complex();
        let norm = z.norm();
        if norm == 0.0 {
            continue; // phase 1: nothing to scale
        }
        let phase = D::from_complex64(z / norm);
        let conj_phase = FactorScalar::adjoint(phase);
        for row in 0..q_rows {
            let index = row + q_leading * j;
            q[index] = q[index] * phase;
        }
        for col in 0..r_cols {
            let index = j + r_leading * col;
            r[index] = conj_phase * r[index];
        }
    }
}

pub(crate) fn svd_compact_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
    vh_leading: usize,
) {
    for j in 0..vh_rows {
        let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
        if needs_scaling {
            scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
            scale_row(vh, vh_cols, vh_leading, j, phase);
        }
    }
}

pub(crate) fn svd_compact_adjoint_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
    vh_leading: usize,
) {
    for j in 0..vh_rows {
        let (phase, needs_scaling) = phase_of_largest_abs_row(vh, vh_cols, vh_leading, j);
        if needs_scaling {
            scale_col(u, u_rows, u_leading, j, phase);
            scale_row(vh, vh_cols, vh_leading, j, FactorScalar::adjoint(phase));
        }
    }
}

pub(crate) fn svd_full_gauge<D: FactorScalar>(
    u: &mut [D],
    u_rows: usize,
    u_leading: usize,
    vh: &mut [D],
    vh_rows: usize,
    vh_cols: usize,
) {
    let paired = u_leading.min(vh_rows);
    for j in 0..u_leading.max(vh_rows) {
        if j < paired {
            let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
            if needs_scaling {
                scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
                scale_row(vh, vh_cols, vh_rows, j, phase);
            }
        } else if j < u_leading {
            let (phase, needs_scaling) = phase_of_largest_abs_col(u, u_rows, u_leading, j);
            if needs_scaling {
                scale_col(u, u_rows, u_leading, j, FactorScalar::adjoint(phase));
            }
        } else {
            let (phase, needs_scaling) = phase_of_largest_abs_row(vh, vh_cols, vh_rows, j);
            if needs_scaling {
                scale_row(vh, vh_cols, vh_rows, j, FactorScalar::adjoint(phase));
            }
        }
    }
}

pub(super) fn phase_of_largest_abs_col<D: FactorScalar>(
    data: &[D],
    rows: usize,
    leading: usize,
    col: usize,
) -> (D, bool) {
    let mut best = Complex64::new(0.0, 0.0);
    let mut best_norm_sqr = 0.0;
    for row in 0..rows {
        let value = data[row + leading * col].widen_complex();
        let norm_sqr = value.norm_sqr();
        if best_norm_sqr < norm_sqr {
            best = value;
            best_norm_sqr = norm_sqr;
        }
    }
    unit_phase(best, best_norm_sqr)
}

pub(super) fn phase_of_largest_abs_row<D: FactorScalar>(
    data: &[D],
    cols: usize,
    leading: usize,
    row: usize,
) -> (D, bool) {
    let mut best = Complex64::new(0.0, 0.0);
    let mut best_norm_sqr = 0.0;
    for col in 0..cols {
        let value = data[row + leading * col].widen_complex();
        let norm_sqr = value.norm_sqr();
        if best_norm_sqr < norm_sqr {
            best = value;
            best_norm_sqr = norm_sqr;
        }
    }
    unit_phase(best, best_norm_sqr)
}

pub(super) fn unit_phase<D: FactorScalar>(value: Complex64, norm_sqr: f64) -> (D, bool) {
    if norm_sqr == 0.0 || (value.im == 0.0 && value.re >= 0.0) {
        (D::from_real(1.0), false)
    } else {
        (D::from_complex64(value / norm_sqr.sqrt()), true)
    }
}

/// Compact SVD owns U and Vt, while S remains the host-side spectrum used by
/// the existing truncation and diagonal construction paths.
#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
pub(super) fn compact_svd_owned<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .svd(D::dense_read(input_view))
        .map_err(OperationError::Dense)?;
    compact_svd_outputs(outputs, rows, cols)
}

/// Compact SVD of each column-major `(data, rows, cols)` block, submitted as one
/// executor batch so the backend admits the whole coupled-sector loop once.
#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
pub(super) fn compact_svd_owned_batch<E, D>(
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
) -> Result<Vec<(Vec<D>, Vec<f64>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factorize_col_major_batch(dense, DenseFactorization::Svd, blocks)?
        .into_iter()
        .zip(blocks)
        .map(|(outputs, &(_, rows, cols))| compact_svd_outputs(outputs, rows, cols))
        .collect()
}

#[expect(
    clippy::type_complexity,
    reason = "the dense SVD ownership boundary returns its documented U, S, Vt tuple"
)]
pub(super) fn compact_svd_outputs<D: FactorScalar>(
    mut outputs: Vec<DenseTensor>,
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<f64>, Vec<D>), OperationError> {
    let rank = rows.min(cols);
    if outputs.len() != 3 {
        return Err(OperationError::Dense(arity_mismatch(
            "svd_into",
            3,
            outputs.len(),
        )));
    }
    let u = compact_factor_output_owned::<D>(outputs.remove(0), &[rows, rank], "svd_into")?;
    let singular_values = compact_real_spectrum_owned::<D>(outputs.remove(0), &[rank], "svd_into")?;
    let vt = compact_factor_output_owned::<D>(outputs.remove(0), &[rank, cols], "svd_into")?;
    Ok((u, singular_values, vt))
}

pub(super) fn concat_compact_svd_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    #[cfg(test)]
    let first = regions
        .iter()
        .find_map(|region| region.as_ref().map(Vec::as_ptr));
    let output = concat_owned_factor_regions(regions, required_len);
    #[cfg(test)]
    crate::probe::owned_output(
        crate::probe::CopyOp::Svd,
        first.is_some_and(|pointer| std::ptr::eq(pointer, output.as_ptr())),
    );
    output
}

#[doc(hidden)]
pub fn diagonal_bond_svd_factor_generic_checked<R, D, V>(
    source: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<V>],
    to_scalar: &dyn Fn(V) -> D,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
    V: Copy,
{
    let space = spectrum_bond::<CheckedGenericAdmissionMode, _, _>(source, spectrum)?;
    let data = diagonal_bond_data(space.space(), spectrum, to_scalar)
        .map_err(CheckedGenericFactorPlanError::from)?;
    BoundDynFactor::from_bound(space, data, 1, 1).map_err(CheckedGenericFactorPlanError::from)
}

/// Checked-Generic compact SVD. Dense SVD is unchanged; all provider-bound
/// output spaces are admitted through the checked staging boundary.
#[doc(hidden)]
pub fn svd_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Svd<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (u, vh, singular_values) =
        svd_compact_factors_with_spectrum_dyn_checked_generic(dense, input)?;
    let s =
        diagonal_bond_svd_factor_generic_checked(input.space(), &singular_values, &D::from_real)?;
    Ok(Svd { u, s, vh })
}

#[doc(hidden)]
pub type CheckedCompactSvdFactorsWithSpectrum<R, D> = (
    BoundDynFactor<R, D>,
    BoundDynFactor<R, D>,
    Vec<SectorSpectrum>,
);

/// Checked compact SVD factors and sector values before diagonal publication.
#[doc(hidden)]
pub(crate) fn svd_compact_factors_with_spectrum_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_compact_oriented_factors_dyn_checked_generic(dense, input, FactorPlacement::Direct)
}

/// Checked compact SVD factors of the logical adjoint of `parent`, read in
/// place: `A = U S Vh` gives `(V, Uh)` on the swapped hom space, with the
/// phase gauge applied to the final left factor `V` (the multiplicity-free
/// [`svd_compact_adjoint_factors_dyn`] contract).
pub(crate) fn svd_compact_adjoint_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_compact_oriented_factors_dyn_checked_generic(dense, parent, FactorPlacement::Adjoint)
}

fn svd_compact_oriented_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    #[cfg(test)]
    if let InputMatricizations::Packed(matrices) = &matrices {
        crate::probe::input_pack(
            crate::probe::CopyOp::Svd,
            matrices
                .iter()
                .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
        );
    }
    let gauge = match placement {
        FactorPlacement::Direct => CompactSvdGauge::Left,
        FactorPlacement::Adjoint => CompactSvdGauge::AdjointLeft,
    };
    let mut pairs = Vec::with_capacity(matrices.len());
    let mut ranks = BTreeMap::new();
    let mut singular_values = Vec::with_capacity(matrices.len());
    #[cfg(test)]
    let data = input.data();
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            #[cfg(test)]
            crate::probe::checked_compact_input(
                crate::probe::CheckedCompactOperation::Svd,
                data,
                matrix.data,
                None,
            );
            let CompactSvdNumericalStage {
                rows,
                cols,
                rank,
                mut u,
                singular_values: values,
                mut vt,
            } = compact_svd_stage(
                dense,
                matrix.data,
                matrix.rows,
                matrix.cols,
                gauge,
                |_u, _vt| {
                    #[cfg(test)]
                    if matches!(gauge, CompactSvdGauge::Left) {
                        crate::probe::checked_svd_stage(_u, _vt);
                    }
                },
            )?;
            singular_values.push(SectorSpectrum {
                sector: matrix.sector,
                values,
            });
            pairs.push(match placement {
                FactorPlacement::Direct => FactorPair {
                    sector: matrix.sector,
                    kept: rank,
                    left: u,
                    left_rows: rows,
                    right: vt,
                    right_leading: rank,
                },
                // `A^H = V S U^H`: the adjoint's left factor is `vt^H`.
                FactorPlacement::Adjoint => {
                    ranks.insert(matrix.sector, rank);
                    adjoint_col_major_in_place(&mut vt, rank, cols);
                    adjoint_col_major_in_place(&mut u, rows, rank);
                    FactorPair {
                        sector: matrix.sector,
                        kept: rank,
                        left: vt,
                        left_rows: cols,
                        right: u,
                        right_leading: rank,
                    }
                }
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    if matches!(placement, FactorPlacement::Direct) {
        let (u, vh) = build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)?;
        return Ok((u, vh, singular_values));
    }
    // Why not the direct pair builder: it orders and admits factor keys by
    // the input's own sides, while the adjoint publishes on the swapped hom
    // space with the parent's sides exchanged.
    let adjoint = FusionTreeHomSpace::new(
        space.homspace().domain().clone(),
        space.homspace().codomain().clone(),
    );
    let authority = CheckedAuthority(provider);
    let mut publish = |side| match &matrices {
        InputMatricizations::Regions { regions, .. } => publish_one_sided_factor(
            &authority,
            &adjoint,
            regions.as_ref(),
            &mut pairs,
            &ranks,
            side,
            FactorPlacement::Adjoint,
        ),
        InputMatricizations::Packed(packed) => publish_one_sided_factor(
            &authority,
            &adjoint,
            packed,
            &mut pairs,
            &ranks,
            side,
            FactorPlacement::Adjoint,
        ),
    };
    let u = publish(FactorSide::Left)?;
    let vh = publish(FactorSide::Right)?;
    Ok((u, vh, singular_values))
}

#[cfg(test)]
/// Checked-Generic full SVD. Dense work is performed before any output-space
/// publication; checked factor builders then admit square outer factors and
/// the rectangular diagonal, including unmatched structural sectors.
pub(crate) fn svd_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let parts = svd_full_factors_dyn_checked_generic(dense, input)?;
    let (u, vh, singular_values, row_dimensions, col_dimensions) = parts.into_parts();
    let s = rectangular_diagonal_bond_tensor(
        &CheckedAuthority(input.space().provider_arc()),
        &singular_values,
        &row_dimensions,
        &col_dimensions,
    )?;
    Ok(SvdFullDyn {
        u,
        s,
        vh,
        singular_values,
    })
}

/// Checked full-SVD numerical factors of a dense input, before choosing `S`
/// storage.
pub(crate) fn svd_full_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn_checked_generic(dense, input, FactorPlacement::Direct)
}

/// Checked full-SVD factors of the logical adjoint of `parent`, read in
/// place (the multiplicity-free [`svd_full_adjoint_factors_dyn`] contract).
pub(crate) fn svd_full_adjoint_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn_checked_generic(dense, parent, FactorPlacement::Adjoint)
}

fn svd_full_oriented_factors_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    // Why pack rather than borrow admitted regions: `owned_full_svd_stage`
    // moves each matrix into the dense provider, so a borrowed region would
    // be copied into an owned buffer anyway.
    let mut matrices = sector_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let checked = CheckedAuthority(provider);
    let row_dimensions = checked.coupled_dimensions(space.homspace().codomain())?;
    let col_dimensions = checked.coupled_dimensions(space.homspace().domain())?;
    let mut workspace =
        FullSvdWorkspace::for_sectors(matrices.iter().map(|matrix| (matrix.rows, matrix.cols)));
    let mut pairs = Vec::with_capacity(matrices.len());
    let mut singular_values = Vec::with_capacity(matrices.len());
    for matrix in &mut matrices {
        let (rows, cols) = (matrix.rows, matrix.cols);
        let (mut u_full, s_values, right) =
            full_svd_numerical_stage(dense, &mut matrix.data, rows, cols, &mut workspace)?;
        let (mut left, left_rows, mut right, right_leading) = match (placement, right) {
            (FactorPlacement::Direct, FullSvdRight::Vh(vh_full)) => (u_full, rows, vh_full, cols),
            (FactorPlacement::Direct, FullSvdRight::V(v_full)) => {
                (u_full, rows, adjoint_col_major(&v_full, cols, cols), cols)
            }
            (FactorPlacement::Adjoint, FullSvdRight::Vh(mut vh_full)) => {
                adjoint_square_col_major_in_place(&mut vh_full, cols);
                adjoint_square_col_major_in_place(&mut u_full, rows);
                (vh_full, cols, u_full, rows)
            }
            (FactorPlacement::Adjoint, FullSvdRight::V(v_full)) => {
                adjoint_square_col_major_in_place(&mut u_full, rows);
                (v_full, cols, u_full, rows)
            }
        };
        svd_full_gauge(
            &mut left,
            left_rows,
            left_rows,
            &mut right,
            right_leading,
            right_leading,
        );
        singular_values.push(SectorSpectrum {
            sector: matrix.sector,
            values: s_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: left_rows,
            left,
            left_rows,
            right,
            right_leading,
        });
    }
    let (homspace, row_dimensions, col_dimensions) = match placement {
        FactorPlacement::Direct => (space.homspace().clone(), row_dimensions, col_dimensions),
        FactorPlacement::Adjoint => (
            FusionTreeHomSpace::new(
                space.homspace().domain().clone(),
                space.homspace().codomain().clone(),
            ),
            col_dimensions,
            row_dimensions,
        ),
    };
    let u = publish_one_sided_factor(
        &checked,
        &homspace,
        &matrices,
        &mut pairs,
        &row_dimensions,
        FactorSide::Left,
        placement,
    )?;
    let vh = publish_one_sided_factor(
        &checked,
        &homspace,
        &matrices,
        &mut pairs,
        &col_dimensions,
        FactorSide::Right,
        placement,
    )?;
    Ok(SvdFullFactorsDyn {
        u,
        vh,
        singular_values,
        row_dimensions,
        col_dimensions,
        #[cfg(test)]
        adjoint_space: None,
    })
}

/// Compact SVD factors and singular values of `source` in fusion mode `M`;
/// only dense storage leases an executor.
#[doc(hidden)]
pub fn svd_compact_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        Some(FactorFamily::Svd),
        M::svd_compact_diagonal,
        M::svd_compact_dense,
    )
}

/// Compact SVD factors of the lazy adjoint of the dense `parent`, read in place,
/// in fusion mode `M`.
#[doc(hidden)]
pub fn svd_compact_adjoint_from_parent<M, L, E, R, D>(
    lease: L,
    parent: BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    // The adjoint is finite exactly when its parent is.
    require_finite_factor_input(parent.data().iter().copied(), FactorFamily::Svd)?;
    lease.run(|dense| M::svd_compact_adjoint_dense(dense, &parent))
}

/// Full SVD factors and singular values of dense `source` in fusion mode
/// `M`. A compact diagonal is refused: its compact and full SVD coincide
/// (TensorKit `svd_compact!(::DiagonalAlgorithm)` is `svd_full!`), so every
/// mode factors it through [`svd_compact_from_source`].
#[doc(hidden)]
pub fn svd_full_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        Some(FactorFamily::Svd),
        |_, _| {
            Err(OperationError::UnsupportedTensorContractScope {
                message: "full SVD of a compact diagonal runs the compact route",
            }
            .into())
        },
        M::svd_full_dense,
    )
}

/// Full SVD factors of the lazy adjoint of the dense `parent`, read in place,
/// in fusion mode `M`.
#[doc(hidden)]
pub fn svd_full_adjoint_from_parent<M, L, E, R, D>(
    lease: L,
    parent: BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    require_finite_factor_input(parent.data().iter().copied(), FactorFamily::Svd)?;
    lease.run(|dense| M::svd_full_adjoint_dense(dense, &parent))
}
