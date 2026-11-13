use super::*;
use symtn_dense::Engine;

#[cfg(test)]
/// Compact (thin, untruncated) fusion-tensor SVD `t = U * S * Vh`
/// (MatrixAlgebraKit `svd_compact`).
///
/// This is the pure device-boundary factorization: the dense per-sector SVDs
/// run through the [`DenseExecutor`] and no truncation logic is involved.
/// Per block the bond is `min(rows, cols)`; the square-`U` variant is
/// MatrixAlgebraKit `svd_full` ([`SvdFull`]).
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
            for (j, &value) in entry.values[..bond].iter().enumerate() {
                let scale = to_scalar(value);
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_vals_dyn(engine, dense, &input.dynamic())
}

/// Dynamic-rank `svd_vals`: the dense stage, the same in every fusion mode.
pub(crate) fn svd_vals_dyn<E, R, D>(
    engine: &mut dyn Engine,
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
    svd_vals_spectra(engine, dense, &matricizations)
}

/// Singular values of every coupled-sector matrix through one no-vector SVD
/// batch, truncated to the sector rank.
pub(super) fn svd_vals_spectra<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (sectors, blocks) = sector_blocks(matrices)?;
    let mut layouts = Vec::new();
    let inputs = col_major_reads(&blocks, &mut layouts)?;
    let mut singular_values = Vec::with_capacity(blocks.len());
    factorize_batch_staged(
        engine,
        dense,
        DenseFactorization::SvdVals,
        &inputs,
        |index, outcome| {
            let s_tensor = single_output(outcome, "svd_vals")?;
            let (_, rows, cols) = blocks[index];
            let mut s = D::real_spectrum(&s_tensor).map_err(OperationError::Dense)?;
            s.truncate(rows.min(cols));
            singular_values.push(SectorSpectrum {
                sector: sectors[index],
                values: s,
            });
            Ok(())
        },
    )?;
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

impl<R, D: FactorScalar> FactorOp<R, D> for op::SvdVals {
    type Output = Vec<SectorSpectrum>;
}

/// Singular values, descending per coupled sector. A compact diagonal is
/// read directly; only dense storage leases an executor. The facade reads a
/// lazy adjoint's parent as dense storage (the values are invariant).
impl<R, D: FactorScalar> Run<R, D> for op::SvdVals {
    fn run<M, L>(
        self,
        _: Token,
        engine: &mut dyn Engine,
        lease: L,
        source: FactorSource<'_, R, D>,
    ) -> Result<<Self as FactorOp<R, D>>::Output, M::Error>
    where
        M: FactorMode<R>,
        L: ExecutorLease,
    {
        factor_from_source(
            engine,
            lease,
            source,
            Some(FactorFamily::Svd),
            |space, spectrum| svd_vals_diagonal(&M::authority(space), space, spectrum),
            |engine, dense, input| Ok(svd_vals_dyn(engine, dense, input)?),
            no_adjoint(),
        )
    }
}

#[cfg(test)]
/// Compact (untruncated) fusion-tensor SVD through the device boundary.
pub(crate) fn svd_compact<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<SvdCompact<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = svd_compact_dyn(engine, dense, &input.dynamic())?;
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_compact_factors_dyn_with_gauge(engine, dense, input, CompactSvdGauge::Left)
}

/// Compact SVD factors for the logical adjoint without constructing its input:
/// if `A = U S Vh`, returns `(V, Uh, spectrum)` with the phase gauge applied
/// to the final left factor `V`.
#[doc(hidden)]
pub(crate) fn svd_compact_adjoint_factors_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_compact_factors_dyn_with_gauge(engine, dense, input, CompactSvdGauge::AdjointLeft)
}

/// One sector of [`compact_svd_sectors_staged`]: its gauged factors, in
/// place in the batch's storage at `u_range`/`vt_range` (the whole
/// per-sector factor under [`SvdStorage::PerSector`]), and its spectrum.
pub(super) struct CompactSvdSector<'s, D> {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) rank: usize,
    pub(super) u: &'s mut [D],
    pub(super) u_range: std::ops::Range<usize>,
    pub(super) singular_values: Vec<f64>,
    pub(super) vt: &'s mut [D],
    pub(super) vt_range: std::ops::Range<usize>,
}

/// Where [`compact_svd_sectors_staged`] keeps the solved factors.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SvdStorage {
    /// One packed transient per factor, written by the destination batch:
    /// for a caller that scatters or publishes the factors, where it
    /// replaces the per-sector owners and their copy.
    Packed,
    /// The owned batch's per-sector factors, as the executor publishes
    /// them: for a caller that only reads the factors in place. Why not
    /// packed there: the transient must be zero-initialized before the
    /// destination entry overwrites it, a serial pass the owned batch's
    /// provider-published outputs do not pay and that such a caller never
    /// recovers (TensorKit's `pinv` likewise solves each block into its own
    /// temporaries).
    PerSector,
}

/// The factors [`compact_svd_sectors_staged`] solved, in sector order.
pub(super) enum StagedSvdFactors<D> {
    /// Every sector's `u` and `vt` concatenated.
    Packed { u: Vec<D>, vt: Vec<D> },
    /// Each sector's own `(u, vt)`; empty for a zero-rank sector.
    PerSector(Vec<(Vec<D>, Vec<D>)>),
}

impl<D> StagedSvdFactors<D> {
    /// Sector `position`'s `(u, vt)`, at `u_range`/`vt_range` of the
    /// packed transient.
    pub(super) fn sector_mut(
        &mut self,
        position: usize,
        u_range: std::ops::Range<usize>,
        vt_range: std::ops::Range<usize>,
    ) -> (&mut [D], &mut [D]) {
        match self {
            Self::Packed { u, vt } => (&mut u[u_range], &mut vt[vt_range]),
            Self::PerSector(sectors) => {
                let (u, vt) = &mut sectors[position];
                (u, vt)
            }
        }
    }
}

/// Compact SVDs of the column-major sectors `blocks` (`(data, rows, cols)`)
/// as one executor batch, the factors kept as `storage` says: `stage`
/// receives every sector in sector order, each solved sector gauged in place
/// (`None` keeps the solver's gauge) and a zero-rank sector as an empty
/// sector that makes no dense call. Each zero-rank sector is staged as soon
/// as every sector before it has been, so a per-matrix executor sees the
/// same call sequence as a sector loop. Either failure of a sector stops the
/// batch. `observe` sees each solved sector's gauged `u` and `vt`.
#[allow(clippy::too_many_arguments)]
pub(super) fn compact_svd_sectors_staged<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
    gauge: Option<CompactSvdGauge>,
    storage: SvdStorage,
    mut observe: impl FnMut(&[D], &[D]),
    mut stage: impl FnMut(usize, CompactSvdSector<'_, D>) -> Result<(), OperationError>,
) -> Result<StagedSvdFactors<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let rank_of = |position: usize| blocks[position].1.min(blocks[position].2);
    let u_ranges = packed_ranges((0..blocks.len()).map(|p| blocks[p].1 * rank_of(p)));
    let vt_ranges = packed_ranges((0..blocks.len()).map(|p| rank_of(p) * blocks[p].2));
    let empty = |position: usize| CompactSvdSector {
        rows: blocks[position].1,
        cols: blocks[position].2,
        rank: 0,
        u: &mut [],
        u_range: u_ranges[position].clone(),
        singular_values: Vec::new(),
        vt: &mut [],
        vt_range: vt_ranges[position].clone(),
    };
    let solved = (0..blocks.len())
        .filter(|&position| rank_of(position) != 0)
        .collect::<Vec<_>>();
    let solved_blocks = solved.iter().map(|&p| blocks[p]).collect::<Vec<_>>();
    let mut layouts = Vec::new();
    let inputs = col_major_reads(&solved_blocks, &mut layouts)?;
    let mut next = 0;
    while next < blocks.len() && rank_of(next) == 0 {
        stage(next, empty(next))?;
        next += 1;
    }
    // One solved sector, gauged and staged, then the zero-rank sectors after
    // it.
    let mut deliver = |position: usize,
                       u: &mut [D],
                       singular_values: Vec<f64>,
                       vt: &mut [D],
                       ranges: [std::ops::Range<usize>; 2]|
     -> Result<(), OperationError> {
        let (_, rows, cols) = blocks[position];
        let rank = rows.min(cols);
        match gauge {
            Some(CompactSvdGauge::Left) => svd_compact_gauge(u, rows, rows, vt, rank, cols, rank),
            Some(CompactSvdGauge::AdjointLeft) => {
                svd_compact_adjoint_gauge(u, rows, rows, vt, rank, cols, rank)
            }
            None => {}
        }
        observe(u, vt);
        let [u_range, vt_range] = ranges;
        stage(
            position,
            CompactSvdSector {
                rows,
                cols,
                rank,
                u,
                u_range,
                singular_values,
                vt,
                vt_range,
            },
        )?;
        next = position + 1;
        while next < blocks.len() && rank_of(next) == 0 {
            stage(next, empty(next))?;
            next += 1;
        }
        Ok(())
    };
    if storage == SvdStorage::PerSector {
        let mut sectors = (0..blocks.len())
            .map(|_| (Vec::new(), Vec::new()))
            .collect::<Vec<_>>();
        if !inputs.is_empty() {
            factorize_batch_staged(
                engine,
                dense,
                DenseFactorization::Svd,
                &inputs,
                |index, outcome| {
                    let position = solved[index];
                    let (_, rows, cols) = blocks[position];
                    let (mut u, singular_values, mut vt) =
                        compact_svd_outputs(outcome.map_err(OperationError::Dense)?, rows, cols)?;
                    let ranges = [0..u.len(), 0..vt.len()];
                    deliver(position, &mut u, singular_values, &mut vt, ranges)?;
                    sectors[position] = (u, vt);
                    Ok(())
                },
            )?;
        }
        return Ok(StagedSvdFactors::PerSector(sectors));
    }
    let s_ranges = packed_ranges((0..blocks.len()).map(rank_of));
    let end = |ranges: &[std::ops::Range<usize>]| ranges.last().map_or(0, |range| range.end);
    let mut u = vec![D::zero(); end(&u_ranges)];
    let mut vt = vec![D::zero(); end(&vt_ranges)];
    let mut s = vec![D::Real::zero(); end(&s_ranges)];
    let dest_layouts = solved_blocks
        .iter()
        .map(|&(_, rows, cols)| {
            let rank = rows.min(cols);
            [
                matrix_layout(rows, rank),
                vector_layout(rank),
                matrix_layout(rank, cols),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    {
        let mut u_slices = split_ranges(&mut u, &u_ranges)?;
        let mut vt_slices = split_ranges(&mut vt, &vt_ranges)?;
        let mut s_slices = split_ranges(&mut s, &s_ranges)?;
        let mut dests = solved
            .iter()
            .zip(&dest_layouts)
            .map(|(&position, layout)| {
                Ok(FactorDest::new(
                    [
                        Some(dest_write(
                            std::mem::take(&mut u_slices[position]),
                            &layout[0],
                        )?),
                        Some(dest_write(
                            std::mem::take(&mut s_slices[position]),
                            &layout[1],
                        )?),
                        Some(dest_write(
                            std::mem::take(&mut vt_slices[position]),
                            &layout[2],
                        )?),
                    ],
                    symtn_dense::FactorPlacement::AsComputed,
                ))
            })
            .collect::<Result<Vec<_>, OperationError>>()?;
        if !inputs.is_empty() {
            factorize_batch_into_staged(
                engine,
                dense,
                DenseFactorization::Svd,
                &inputs,
                &mut dests,
                |index, dest, outcome| {
                    outcome.map_err(|failure| OperationError::Dense(failure.into_error()))?;
                    let position = solved[index];
                    let [u, s, vt] = dest.outputs_mut();
                    let (u, vt) = (
                        dest_data::<D>(u, "svd_into")?,
                        dest_data::<D>(vt, "svd_into")?,
                    );
                    let singular_values = widened_spectrum(dest_data::<D::Real>(s, "svd_into")?);
                    let ranges = [u_ranges[position].clone(), vt_ranges[position].clone()];
                    deliver(position, u, singular_values, vt, ranges)
                },
            )?;
        }
    }
    Ok(StagedSvdFactors::Packed { u, vt })
}

#[derive(Clone, Copy)]
pub(super) enum CompactSvdGauge {
    Left,
    AdjointLeft,
}

fn svd_compact_factors_dyn_with_gauge<E, R, D>(
    engine: &mut dyn Engine,
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
        return svd_compact_direct_regions(engine, dense, input, &plan, gauge, adjoint_spaces);
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
    let blocks = matricizations
        .iter()
        .map(|matrix| (&matrix.data[..], matrix.rows, matrix.cols))
        .collect::<Vec<_>>();
    in_linalg_scope(engine, dense, |dense, engine| {
        compact_svd_sectors_staged(
            engine,
            dense,
            &blocks,
            Some(gauge),
            SvdStorage::Packed,
            |_u, _vt| {
                #[cfg(test)]
                crate::probe::mf_svd_fallback(_u, _vt);
            },
            |position, stage| {
                let matrix = &matricizations[position];
                let CompactSvdSector {
                    rank,
                    u,
                    singular_values: values,
                    vt,
                    ..
                } = stage;
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
                        u,
                        matrix.rows,
                    )?,
                    CompactSvdGauge::AdjointLeft => scatter_adjoint_svd_factor(
                        u_target,
                        &mut u_data,
                        matrix,
                        &index,
                        &u_groups,
                        vt,
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
                        vt,
                        rank,
                    )?,
                    CompactSvdGauge::AdjointLeft => scatter_adjoint_svd_factor(
                        vt_target,
                        &mut vt_data,
                        matrix,
                        &index,
                        &vt_groups,
                        u,
                        matrix.rows,
                        FactorSide::Right,
                    )?,
                }
                #[cfg(test)]
                crate::probe::output_scatter::<D>(crate::probe::CopyOp::Svd, 1, rank * matrix.cols);
                Ok(())
            },
        )
    })
    .map(drop)?;

    let (u_nout, u_nin) = (u_target.nout(), u_target.nin());
    let (vh_nout, vh_nin) = (vt_target.nout(), vt_target.nin());
    let u = BoundDynFactor::from_bound(left_space, u_data, u_nout, u_nin)?;
    let vh = BoundDynFactor::from_bound(right_space, vt_data, vh_nout, vh_nin)?;
    Ok((u, vh, singular_values))
}

pub(super) fn svd_compact_direct_regions<E, R, D>(
    engine: &mut dyn Engine,
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
    // The adjoint's own factor regions, per route, when its trees match the
    // parent's so each route's adjoint factors are one contiguous region.
    let adjoint_regions = if matches!(gauge, CompactSvdGauge::AdjointLeft) {
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
                        routes.push((left[li].range(), right[ri].range()));
                    } else {
                        compatible = false;
                        break;
                    }
                }
                compatible.then_some(routes)
            }
            _ => None,
        }
    } else {
        None
    };
    // A zero-rank route is not solved; its spectrum stays empty.
    let mut singular_values = plan
        .routes
        .iter()
        .map(|route| SectorSpectrum {
            sector: route.sector,
            values: Vec::new(),
        })
        .collect::<Vec<_>>();
    let solved = (0..plan.routes.len())
        .filter(|&position| plan.routes[position].rank != 0)
        .collect::<Vec<_>>();
    let blocks = solved
        .iter()
        .map(|&position| {
            let region = &plan.source_regions[plan.routes[position].source_region];
            (&input.data()[region.range()], region.rows(), region.cols())
        })
        .collect::<Vec<_>>();
    let mut u_data = vec![D::zero(); left_space.space().required_len()?];
    let mut vh_data = vec![D::zero(); right_space.space().required_len()?];

    // Where each solved route's factors land: the final regions (as
    // computed, or the adjoint problem's factors in the adjoint's regions),
    // or, for an adjoint whose trees do not match, one packed transient that
    // is scattered once the batch has succeeded.
    let scattered = matches!(gauge, CompactSvdGauge::AdjointLeft) && adjoint_regions.is_none();
    let (u_ranges, vh_ranges, placement) = match (gauge, &adjoint_regions) {
        (CompactSvdGauge::Left, _) => (
            solved
                .iter()
                .map(|&p| {
                    plan.left_regions[plan.routes[p]
                        .left_region
                        .expect("nonzero route has left region")]
                    .range()
                })
                .collect::<Vec<_>>(),
            solved
                .iter()
                .map(|&p| {
                    plan.right_regions[plan.routes[p]
                        .right_region
                        .expect("nonzero route has right region")]
                    .range()
                })
                .collect::<Vec<_>>(),
            symtn_dense::FactorPlacement::AsComputed,
        ),
        (CompactSvdGauge::AdjointLeft, Some(routes)) => (
            solved.iter().map(|&p| routes[p].0.clone()).collect(),
            solved.iter().map(|&p| routes[p].1.clone()).collect(),
            symtn_dense::FactorPlacement::Adjoint,
        ),
        (CompactSvdGauge::AdjointLeft, None) => (
            packed_ranges(blocks.iter().map(|&(_, rows, cols)| rows * rows.min(cols))),
            packed_ranges(blocks.iter().map(|&(_, rows, cols)| rows.min(cols) * cols)),
            symtn_dense::FactorPlacement::AsComputed,
        ),
    };
    let end = |ranges: &[std::ops::Range<usize>]| ranges.last().map_or(0, |range| range.end);
    let (mut u_transient, mut vh_transient) = if scattered {
        (
            vec![D::zero(); end(&u_ranges)],
            vec![D::zero(); end(&vh_ranges)],
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let s_ranges = packed_ranges(blocks.iter().map(|&(_, rows, cols)| rows.min(cols)));
    let mut s_data = vec![D::Real::zero(); end(&s_ranges)];
    let layouts = blocks
        .iter()
        .map(|&(_, rows, cols)| {
            let rank = rows.min(cols);
            match placement {
                symtn_dense::FactorPlacement::Adjoint => [
                    matrix_layout(cols, rank),
                    vector_layout(rank),
                    matrix_layout(rank, rows),
                ],
                _ => [
                    matrix_layout(rows, rank),
                    vector_layout(rank),
                    matrix_layout(rank, cols),
                ],
            }
        })
        .collect::<Vec<DestLayouts>>();
    {
        let (u_target, vh_target) = if scattered {
            (&mut u_transient[..], &mut vh_transient[..])
        } else {
            (&mut u_data[..], &mut vh_data[..])
        };
        let u_slices = split_ranges(u_target, &u_ranges)?;
        let vh_slices = split_ranges(vh_target, &vh_ranges)?;
        let s_slices = split_ranges(&mut s_data, &s_ranges)?;
        let mut dests = u_slices
            .into_iter()
            .zip(s_slices)
            .zip(vh_slices)
            .zip(&layouts)
            .map(|(((u, s), vh), layout)| {
                Ok(FactorDest::new(
                    [
                        Some(dest_write(u, &layout[0])?),
                        Some(dest_write(s, &layout[1])?),
                        Some(dest_write(vh, &layout[2])?),
                    ],
                    placement,
                ))
            })
            .collect::<Result<Vec<_>, OperationError>>()?;
        let mut layouts = Vec::new();
        let inputs = col_major_reads(&blocks, &mut layouts)?;
        factorize_into_solver_first(
            engine,
            dense,
            DenseFactorization::Svd,
            &inputs,
            &mut dests,
            |index, dest| {
                let (_, rows, cols) = blocks[index];
                let rank = rows.min(cols);
                let [u, s, vh] = dest.outputs_mut();
                let (u, vh) = (
                    dest_data::<D>(u, "svd_into")?,
                    dest_data::<D>(vh, "svd_into")?,
                );
                match (gauge, placement) {
                    (CompactSvdGauge::Left, _) => {
                        svd_compact_gauge(u, rows, rows, vh, rank, cols, rank)
                    }
                    // The adjoint's `u` columns are the conjugated parent `Vh`
                    // rows, so its left gauge is the parent-row gauge.
                    (CompactSvdGauge::AdjointLeft, symtn_dense::FactorPlacement::Adjoint) => {
                        svd_compact_gauge(u, cols, cols, vh, rank, rows, rank)
                    }
                    (CompactSvdGauge::AdjointLeft, _) => {
                        svd_compact_adjoint_gauge(u, rows, rows, vh, rank, cols, rank)
                    }
                }
                singular_values[solved[index]].values =
                    widened_spectrum(dest_data::<D::Real>(s, "svd_into")?);
                Ok(())
            },
        )?;
    }
    if scattered {
        let index =
            PlacementIndex::new(&plan.source_regions, &[FactorSide::Left, FactorSide::Right]);
        let left_groups = SectorBlockGroups::new(left_space.space().structure(), FactorSide::Left)?;
        let right_groups =
            SectorBlockGroups::new(right_space.space().structure(), FactorSide::Right)?;
        for (solved_index, &position) in solved.iter().enumerate() {
            let region = &plan.source_regions[plan.routes[position].source_region];
            let rank = plan.routes[position].rank;
            scatter_adjoint_svd_factor(
                left_space.space(),
                &mut u_data,
                region,
                &index,
                &left_groups,
                &vh_transient[vh_ranges[solved_index].clone()],
                rank,
                FactorSide::Left,
            )?;
            scatter_adjoint_svd_factor(
                right_space.space(),
                &mut vh_data,
                region,
                &index,
                &right_groups,
                &u_transient[u_ranges[solved_index].clone()],
                region.rows(),
                FactorSide::Right,
            )?;
        }
    }
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdCompactDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let (u, vh, singular_values) = svd_compact_factors_dyn(engine, dense, input)?;
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
/// Full fusion-tensor SVD through the device boundary: every sector in one
/// [`DenseFactorization::SvdFull`] batch (see [`full_svd_packed`]). The
/// native provider solves the blocks on its lanes; any other executor keeps
/// its per-matrix route below the seam ([`DenseExecutor::svd_full_owned`]
/// when it [`DenseExecutor::supports_svd_full`], otherwise the compact
/// [`DenseExecutor::svd_into`] completed by an economy QR of `[U1 | I]`,
/// exact because the matching rows/columns of `S` are zero). The routes
/// agree on the reconstructed input, the spectrum and unitarity; the basis
/// spanning the null space is not unique and may differ between them.
pub(crate) fn svd_full<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<SvdFull<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = svd_full_dyn(engine, dense, &input.dynamic())?;
    Ok(SvdFull {
        u: typed_from_bound_factor(out.u)?,
        s: typed_from_bound_factor(out.s)?,
        vh: typed_from_bound_factor(out.vh)?,
    })
}

#[cfg(test)]
/// Dynamic-rank [`svd_full`].
pub(crate) fn svd_full_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_dyn(engine, dense, input, FactorPlacement::Direct)
}

/// Full-SVD numerical factors without publishing a dense `S` tensor.
#[doc(hidden)]
pub(crate) fn svd_full_factors_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn(engine, dense, input, FactorPlacement::Direct)
}

#[doc(hidden)]
pub(crate) fn svd_full_adjoint_factors_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn(engine, dense, input, FactorPlacement::Adjoint)
}

#[cfg(test)]
/// Full SVD factors for the logical adjoint without constructing its input.
pub(crate) fn svd_full_adjoint_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    svd_full_oriented_dyn(engine, dense, input, FactorPlacement::Adjoint)
}

/// Full SVD of every coupled-sector matrix in one solver-first executor
/// batch ([`DenseFactorization::SvdFull`]), each solved block gauged in
/// place ([`svd_full_gauge`]), as square `u` (left) and `vh` (right) factors
/// written once into one buffer per side in matrix order, laid out from the
/// matrix shapes before the dense call, as TensorKit's `svd_full!` writes
/// each block of its preallocated `U`/`Vᴴ` in place; plus each sector's
/// spectrum. Under [`FactorPlacement::Adjoint`] the matrices are the parent
/// blocks `A` of a lazy adjoint `X = A^H`, read in place and placed as the
/// full SVD of `X` (`u = Vh^H`, `vh = U^H`), so no adjoint is staged.
/// Past `min(rows, cols)`, `u`'s columns and `vh`'s rows are any
/// orthonormal basis of the complement, which may differ by executor.
fn full_svd_packed<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    placement: FactorPlacement,
) -> Result<(PackedFactorPairs<D>, Vec<SectorSpectrum>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (sectors, blocks) = sector_blocks(matrices)?;
    // `X`'s `(rows, cols)`: an adjoint parent block is `X^H`.
    let dims = |&(_, rows, cols): &(&[D], usize, usize)| match placement {
        FactorPlacement::Direct => (rows, cols),
        FactorPlacement::Adjoint => (cols, rows),
    };
    let left_ranges = packed_ranges(blocks.iter().map(|block| dims(block).0.pow(2)));
    let right_ranges = packed_ranges(blocks.iter().map(|block| dims(block).1.pow(2)));
    let s_ranges = packed_ranges(blocks.iter().map(|&(_, rows, cols)| rows.min(cols)));
    let end = |ranges: &[std::ops::Range<usize>]| ranges.last().map_or(0, |range| range.end);
    let mut left = vec![D::zero(); end(&left_ranges)];
    let mut right = vec![D::zero(); end(&right_ranges)];
    let mut s = vec![D::Real::zero(); end(&s_ranges)];
    let mut read_layouts = Vec::new();
    let inputs = col_major_reads(&blocks, &mut read_layouts)?;
    let layouts = blocks
        .iter()
        .map(|block| {
            let (rows, cols) = dims(block);
            [
                matrix_layout(rows, rows),
                vector_layout(rows.min(cols)),
                matrix_layout(cols, cols),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    let dense_placement = match placement {
        FactorPlacement::Direct => symtn_dense::FactorPlacement::AsComputed,
        FactorPlacement::Adjoint => symtn_dense::FactorPlacement::Adjoint,
    };
    let mut singular_values = sectors
        .iter()
        .map(|&sector| SectorSpectrum {
            sector,
            values: Vec::new(),
        })
        .collect::<Vec<_>>();
    {
        let mut dests = split_ranges(&mut left, &left_ranges)?
            .into_iter()
            .zip(split_ranges(&mut s, &s_ranges)?)
            .zip(split_ranges(&mut right, &right_ranges)?)
            .zip(&layouts)
            .map(|(((u, s), vh), layout)| {
                Ok(FactorDest::new(
                    [
                        Some(dest_write(u, &layout[0])?),
                        Some(dest_write(s, &layout[1])?),
                        Some(dest_write(vh, &layout[2])?),
                    ],
                    dense_placement,
                ))
            })
            .collect::<Result<Vec<_>, OperationError>>()?;
        factorize_into_solver_first(
            engine,
            dense,
            DenseFactorization::SvdFull,
            &inputs,
            &mut dests,
            |index, dest| {
                let (rows, cols) = dims(&blocks[index]);
                let [u, s, vh] = dest.outputs_mut();
                svd_full_gauge(
                    dest_data::<D>(u, "svd_into")?,
                    rows,
                    rows,
                    dest_data::<D>(vh, "svd_into")?,
                    cols,
                    cols,
                );
                singular_values[index].values =
                    widened_spectrum(dest_data::<D::Real>(s, "svd_into")?);
                Ok(())
            },
        )?;
    }
    let pairs = blocks
        .iter()
        .zip(sectors)
        .zip(left_ranges.into_iter().zip(right_ranges))
        .map(|((block, sector), (left, right))| {
            let (rows, cols) = dims(block);
            PackedFactorPair {
                sector,
                kept: rows,
                left,
                left_rows: rows,
                right,
                right_leading: cols,
            }
        })
        .collect();
    Ok((PackedFactorPairs { pairs, left, right }, singular_values))
}

#[cfg(test)]
pub(super) fn svd_full_oriented_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    placement: FactorPlacement,
) -> Result<SvdFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let parts = svd_full_oriented_factors_dyn(engine, dense, input, placement)?;
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
    engine: &mut dyn Engine,
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
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    let authority = MfAuthority(input.space());
    let row_dimensions = authority.coupled_dimensions(space.homspace().codomain())?;
    let col_dimensions = authority.coupled_dimensions(space.homspace().domain())?;
    let (mut pairs, singular_values) = full_svd_packed(engine, dense, &matrices, placement)?;

    let adjoint_space = match placement {
        FactorPlacement::Direct => None,
        FactorPlacement::Adjoint => Some(<MultiplicityFreeAdmissionMode as CoefficientAlgebra<
            R,
        >>::adjoint_space(input.space())?),
    };
    let authority = adjoint_space.as_ref().unwrap_or(input.space());
    let homspace = authority.space().homspace();
    let (row_dimensions, col_dimensions) = match placement {
        FactorPlacement::Direct => (row_dimensions, col_dimensions),
        FactorPlacement::Adjoint => (col_dimensions, row_dimensions),
    };
    // The left/right bond legs differ in the full SVD (rows vs columns), so
    // build the two factors with separate bond dimensions.
    let (u, vh) = with_input_geometry!(&matrices, |geometry| {
        let mut publish = |dimensions, side| {
            build_bound_factor_with_placement(
                authority, homspace, geometry, &mut pairs, dimensions, side, placement,
            )
        };
        Ok::<_, OperationError>((
            publish(&row_dimensions, FactorSide::Left)?,
            publish(&col_dimensions, FactorSide::Right)?,
        ))
    })?;
    Ok(SvdFullFactorsDyn {
        u,
        vh,
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

/// The compact diagonal SVD's per-route factors concatenated in region order.
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

#[cfg(test)]
pub(crate) fn diagonal_bond_svd_factor_generic_checked<R, D, V>(
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
#[cfg(test)]
pub(crate) fn svd_compact_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Svd<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let (u, vh, singular_values) =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, dense, input)?;
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_compact_oriented_factors_dyn_checked_generic(engine, dense, input, FactorPlacement::Direct)
}

/// Checked compact SVD factors of the logical adjoint of `parent`, read in
/// place: `A = U S Vh` gives `(V, Uh)` on the swapped hom space, with the
/// phase gauge applied to the final left factor `V` (the multiplicity-free
/// [`svd_compact_adjoint_factors_dyn`] contract).
pub(crate) fn svd_compact_adjoint_factors_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<CheckedCompactSvdFactorsWithSpectrum<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_compact_oriented_factors_dyn_checked_generic(
        engine,
        dense,
        parent,
        FactorPlacement::Adjoint,
    )
}

fn svd_compact_oriented_factors_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
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
    let packed = in_linalg_scope(engine, dense, |dense, engine| {
        let (sectors, blocks) = sector_blocks(&matrices)?;
        #[cfg(test)]
        for &(matrix, _, _) in &blocks {
            crate::probe::checked_compact_input(
                crate::probe::CheckedCompactOperation::Svd,
                data,
                matrix,
            );
        }
        compact_svd_sectors_staged(
            engine,
            dense,
            &blocks,
            Some(gauge),
            SvdStorage::Packed,
            |_u, _vt| {
                #[cfg(test)]
                if matches!(gauge, CompactSvdGauge::Left) {
                    crate::probe::checked_svd_stage(_u, _vt);
                }
            },
            |index, stage| {
                let sector = sectors[index];
                let CompactSvdSector {
                    rows,
                    cols,
                    rank,
                    u,
                    u_range,
                    singular_values: values,
                    vt,
                    vt_range,
                } = stage;
                singular_values.push(SectorSpectrum { sector, values });
                pairs.push(match placement {
                    FactorPlacement::Direct => PackedFactorPair {
                        sector,
                        kept: rank,
                        left: u_range,
                        left_rows: rows,
                        right: vt_range,
                        right_leading: rank,
                    },
                    // `A^H = V S U^H`: the adjoint's left factor is `vt^H`,
                    // adjointed in place in the packed `vt`.
                    FactorPlacement::Adjoint => {
                        ranks.insert(sector, rank);
                        adjoint_col_major_in_place(vt, rank, cols);
                        adjoint_col_major_in_place(u, rows, rank);
                        PackedFactorPair {
                            sector,
                            kept: rank,
                            left: vt_range,
                            left_rows: cols,
                            right: u_range,
                            right_leading: rank,
                        }
                    }
                });
                Ok(())
            },
        )
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    let StagedSvdFactors::Packed {
        u: packed_u,
        vt: packed_vt,
    } = packed
    else {
        unreachable!("the packed storage was requested")
    };
    if matches!(placement, FactorPlacement::Direct) {
        let pairs = PackedFactorPairs {
            pairs,
            left: packed_u,
            right: packed_vt,
        };
        let (u, vh) = build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)?;
        return Ok((u, vh, singular_values));
    }
    let mut pairs = PackedFactorPairs {
        pairs,
        left: packed_vt,
        right: packed_u,
    };
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let parts = svd_full_factors_dyn_checked_generic(engine, dense, input)?;
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn_checked_generic(engine, dense, input, FactorPlacement::Direct)
}

/// Checked full-SVD factors of the logical adjoint of `parent`, read in
/// place (the multiplicity-free [`svd_full_adjoint_factors_dyn`] contract).
pub(crate) fn svd_full_adjoint_factors_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<SvdFullFactorsDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    svd_full_oriented_factors_dyn_checked_generic(engine, dense, parent, FactorPlacement::Adjoint)
}

fn svd_full_oriented_factors_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
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
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let checked = CheckedAuthority(provider);
    let row_dimensions = checked.coupled_dimensions(space.homspace().codomain())?;
    let col_dimensions = checked.coupled_dimensions(space.homspace().domain())?;
    let (mut pairs, singular_values) = full_svd_packed(engine, dense, &matrices, placement)
        .map_err(CheckedGenericFactorPlanError::from)?;
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
    let (u, vh) = with_input_geometry!(&matrices, |geometry| {
        let mut publish = |dimensions, side| {
            publish_one_sided_factor(
                &checked, &homspace, geometry, &mut pairs, dimensions, side, placement,
            )
        };
        Ok::<_, CheckedGenericFactorPlanError<R::Error>>((
            publish(&row_dimensions, FactorSide::Left)?,
            publish(&col_dimensions, FactorSide::Right)?,
        ))
    })?;
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

impl<R, D: FactorScalar> FactorOp<R, D> for op::SvdCompact {
    type Output = SvdFactorsDyn<R, D>;
}

/// Compact SVD factors and singular values; only dense storage leases an
/// executor. A lazy adjoint's dense parent is read in place, and the mode's
/// adjoint stage gauges the final left factor as materialize-then-SVD would.
impl<R, D: FactorScalar> Run<R, D> for op::SvdCompact {
    fn run<M, L>(
        self,
        _: Token,
        engine: &mut dyn Engine,
        lease: L,
        source: FactorSource<'_, R, D>,
    ) -> Result<<Self as FactorOp<R, D>>::Output, M::Error>
    where
        M: FactorMode<R>,
        L: ExecutorLease,
    {
        factor_from_source(
            engine,
            lease,
            source,
            Some(FactorFamily::Svd),
            |space, spectrum| M::svd_compact_diagonal(TOKEN, space, spectrum),
            |engine, dense, input| M::svd_compact_dense(TOKEN, engine, dense, input),
            Some(
                |engine: &mut dyn Engine,
                 dense: &mut L::Executor,
                 parent: &BoundDynamicTensorRef<'_, R, D>,
                 _: &BoundDynamicFusionMapSpace<R>| {
                    M::svd_compact_adjoint_dense(TOKEN, engine, dense, parent)
                },
            ),
        )
    }
}

impl<R, D: FactorScalar> FactorOp<R, D> for op::SvdFull {
    type Output = SvdFullFactorsDyn<R, D>;
}

/// Full SVD factors and singular values of dense storage or of a lazy
/// adjoint's dense parent. A compact diagonal is refused: its compact and
/// full SVD coincide (TensorKit `svd_compact!(::DiagonalAlgorithm)` is
/// `svd_full!`), so every mode factors it through [`op::SvdCompact`].
impl<R, D: FactorScalar> Run<R, D> for op::SvdFull {
    fn run<M, L>(
        self,
        _: Token,
        engine: &mut dyn Engine,
        lease: L,
        source: FactorSource<'_, R, D>,
    ) -> Result<<Self as FactorOp<R, D>>::Output, M::Error>
    where
        M: FactorMode<R>,
        L: ExecutorLease,
    {
        factor_from_source(
            engine,
            lease,
            source,
            Some(FactorFamily::Svd),
            |_, _| {
                Err(OperationError::UnsupportedTensorContractScope {
                    message: "full SVD of a compact diagonal runs the compact route",
                }
                .into())
            },
            |engine, dense, input| M::svd_full_dense(TOKEN, engine, dense, input),
            Some(
                |engine: &mut dyn Engine,
                 dense: &mut L::Executor,
                 parent: &BoundDynamicTensorRef<'_, R, D>,
                 _: &BoundDynamicFusionMapSpace<R>| {
                    M::svd_full_adjoint_dense(TOKEN, engine, dense, parent)
                },
            ),
        )
    }
}
