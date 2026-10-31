use super::*;
use symtn_dense::Engine;

pub(super) trait HermitianReal: Float {
    fn from_f64(value: f64) -> Self;
}

impl HermitianReal for f32 {
    fn from_f64(value: f64) -> Self {
        value as Self
    }
}

impl HermitianReal for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

/// The tolerance of eigh's Hermiticity admission, MatrixAlgebraKit's
/// `hermitian_tol`: every coupled-sector block `A` must satisfy
/// `‖(A − Aᴴ)/2‖_F ≤ tol · ‖A‖_F` at the payload's working precision.
///
/// MAK uses an absolute `atol`, defaulting to `eps(norm(A, Inf))^(3/4)`
/// (`common/defaults.jl:default_hermitian_tol`). symtn keeps a relative
/// threshold so admission is invariant under rescaling `A` (#1983, #1987).
/// The defaults agree for a scalar of unit magnitude, not at arbitrary scales.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HermitianTol {
    relative: Option<f64>,
}

impl HermitianTol {
    /// `eps(real(D))^(3/4)`.
    pub const DEFAULT: Self = Self { relative: None };

    /// An explicit relative tolerance.
    ///
    /// # Errors
    ///
    /// [`OperationError::InvalidArgument`] unless `tol` is finite and
    /// non-negative.
    pub fn relative(tol: f64) -> Result<Self, OperationError> {
        if !tol.is_finite() || tol < 0.0 {
            return Err(OperationError::InvalidArgument {
                message: "hermitian_tol must be finite and non-negative",
            });
        }
        Ok(Self {
            relative: Some(tol),
        })
    }

    /// The relative tolerance at a payload precision with machine epsilon
    /// `epsilon`.
    #[doc(hidden)]
    pub fn resolve(self, epsilon: f64) -> f64 {
        self.relative.unwrap_or_else(|| epsilon.powf(0.75))
    }
}

/// `exp`'s spectral-route predicate in machine epsilons. Why not
/// [`HermitianTol::DEFAULT`]: this picks an algorithm, not an admission.
/// The eigensolver reads a Hermitian triangle; retaining this threshold avoids
/// sending additional nearly-Hermitian inputs from Padé to that spectral route.
pub(crate) const EXP_SPECTRAL_ROUTE_EPSILONS: f64 = 64.0;

/// symtn's EIGH value order within one coupled sector (#1985): ascending.
///
/// Writes into `order` (`order.len() == values.len()`) the stable permutation
/// with `values[order[k]]` ascending, and returns whether it is not the
/// identity. Already-ascending input, the order every LAPACK/faer/cuSOLVER
/// `syev`-family solver returns, costs one O(n) scan and no sort, so callers
/// skip the value and eigenvector permutation on `false`.
///
/// Reference: MatrixAlgebraKit v0.6.8 dense `eigh_full!`/`eigh_vals!`
/// forward LAPACK's ascending order unsorted, and `DiagonalAlgorithm`
/// sorts by `real` with the stable `sortperm`
/// (<https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/33d77fdf032d5ffa36310a04c21aef0c51a554f3/src/implementations/eigh.jl#L179>).
pub fn ascending_eigh_order(values: &[f64], order: &mut [usize]) -> bool {
    stable_ascending_order(values, order, f64::total_cmp)
}

/// symtn's EIG value order within one coupled sector (#1985): ascending
/// lexicographic `(real(λ), imag(λ))`, approved 2026-10-07 for every route.
///
/// Same contract as [`ascending_eigh_order`]. Reference: MatrixAlgebraKit
/// v0.6.8 `eig_sortby` for `DiagonalAlgorithm`
/// (<https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/33d77fdf032d5ffa36310a04c21aef0c51a554f3/src/implementations/eig.jl#L162>).
/// Why symtn also sorts dense sectors, where MAK `eig_full_qr_iteration!`
/// keeps `geev`'s unspecified order: one representation-independent public
/// order (a deliberate deviation from MatrixAlgebraKit, #1985), at O(n log n) per sector next to the
/// O(n³) solver.
pub fn lexicographic_eig_order(values: &[Complex64], order: &mut [usize]) -> bool {
    stable_ascending_order(values, order, lexicographic_eig_cmp)
}

fn lexicographic_eig_cmp(left: &Complex64, right: &Complex64) -> std::cmp::Ordering {
    left.re
        .total_cmp(&right.re)
        .then(left.im.total_cmp(&right.im))
}

fn stable_ascending_order<T>(
    values: &[T],
    order: &mut [usize],
    cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
) -> bool {
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }
    if values.is_sorted_by(|left, right| cmp(left, right).is_le()) {
        return false;
    }
    order.sort_by(|&left, &right| cmp(&values[left], &values[right]));
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::EigOrderPermutation);
    true
}

/// Values-only form of the same order: sorts `values` in place, skipping the
/// sort when already ordered.
fn sort_spectrum<T>(values: &mut [T], cmp: impl Fn(&T, &T) -> std::cmp::Ordering) {
    if !values.is_sorted_by(|left, right| cmp(left, right).is_le()) {
        values.sort_by(cmp);
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::EigOrderPermutation);
    }
}

#[cfg(test)]
/// Full (untruncated) Hermitian eigendecomposition `t = V * D * Vh`.
///
/// Requires an endomorphism (`codomain == domain`) with Hermitian coupled
/// blocks. Bond states are stored ascending by eigenvalue per sector
/// ([`ascending_eigh_order`]); `eigenvalues` keeps the signed values in that
/// order and `D : W <- W` is their diagonal tensor.
#[derive(Clone, Debug)]
pub(crate) struct EighFull<R, D> {
    pub d: BoundTensorMap<R, D>,
    pub v: BoundTensorMap<R, D>,
    pub eigenvalues: Vec<SectorSpectrum>,
}

/// Dynamic-rank `EighFull`. Carries only the eigenvector map and the O(rank)
/// spectrum; the dense diagonal `D` is built on demand by the test-only
/// `eigh_full` wrapper, so callers that keep `D` diagonal
/// (the user layer, via compact diagonal storage) never pay the O(rank²)
/// materialization.
#[derive(Clone, Debug)]
pub struct EighFullDyn<R, D> {
    pub(super) v: BoundDynFactor<R, D>,
    pub(super) eigenvalues: Vec<SectorSpectrum>,
}

impl<R, D> EighFullDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum] {
        &self.eigenvalues
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D>, Vec<SectorSpectrum>) {
        (self.v, self.eigenvalues)
    }
}

#[cfg(test)]
/// Full Hermitian eigendecomposition through the device boundary.
///
/// Before any dense call, every coupled-sector block must pass the
/// Hermiticity admission at [`HermitianTol::DEFAULT`].
pub(crate) fn eigh_full<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<EighFull<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eigh_full_dyn(engine, dense, &dynamic, HermitianTol::DEFAULT)?;
    // Materialize the dense diagonal here (this wrapper returns a dense `D`);
    // the dyn producer does not build it.
    let d = diagonal_bond_svd_factor(dynamic.space(), &out.eigenvalues, &D::from_real)?;
    Ok(EighFull {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank `eigh_full`: the shared core, admitting Hermitian blocks
/// at `hermitian_tol`.
pub fn eigh_full_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    EndomorphismFamily::EighFull.require_endomorphism(space.homspace())?;
    if let Some(plan) = compact_factor_plan(input.space())? {
        // The plan proves the stacking.
        require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
        return eigh_full_direct_regions(engine, dense, input, &plan, hermitian_tol);
    }
    let matricizations = admit_endomorphism_stacking(
        space,
        input.data(),
        EndomorphismFamily::EighFull,
        |structure, data, nout| {
            let matricizations = multiplicity_free_input_matricizations(structure, data, nout)?;
            #[cfg(test)]
            if let InputMatricizations::Packed(matrices) = &matricizations {
                crate::probe::input_pack(
                    crate::probe::CopyOp::Eigh,
                    matrices
                        .iter()
                        .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
                );
            }
            Ok(matricizations)
        },
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
    matricizations.validate_hermitian(hermitian_tol)?;
    with_input_geometry!(&matricizations, |geometry| eigh_full_scattered(
        engine,
        dense,
        input,
        &matricizations,
        geometry
    ))
}

/// Eigendecomposition of an input whose tree order the positional direct
/// path cannot prove; eigenvectors scatter by tree identity.
pub(super) fn eigh_full_scattered<E, R, D, M>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    matricizations: &InputMatricizations<'_, D>,
    geometry: &[M],
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
    M: SectorGeometry + Sync,
{
    let space = input.space().space();
    let ranks = geometry
        .iter()
        .map(|matrix| SectorRank {
            sector: matrix.sector(),
            kept: matrix.rows(),
        })
        .collect::<Vec<_>>();
    let v_space = build_bound_factor_space(
        input.space(),
        space.homspace(),
        SectorLeg::new(ranks.iter().map(|rank| (rank.sector, rank.kept)), false),
        FactorSide::Left,
    )?;
    let v_len = v_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let mut v_data = vec![D::zero(); v_len];
    let max_n = geometry.iter().map(SectorGeometry::rows).max().unwrap_or(0);
    let mut scratch = EighScratch::for_order(max_n);
    let mut eigenvalues = Vec::with_capacity(geometry.len());
    let index = PlacementIndex::new(geometry, &[FactorSide::Left]);
    let v_groups = SectorBlockGroups::new(v_space.space().structure(), FactorSide::Left)?;
    let v_target = v_space.space();
    in_linalg_scope(engine, dense, |dense, engine| {
        for (position, sector_geometry) in geometry.iter().enumerate() {
            let matrix = matricizations.get(position)?;
            let n = matrix.rows;
            let (sorted_values, vectors) =
                eigh_sector_stage(engine, dense, matrix.data, n, &mut scratch)?;
            eigenvalues.push(SectorSpectrum {
                sector: matrix.sector,
                values: sorted_values,
            });
            #[cfg(test)]
            crate::probe::eigh_owned_vector(&vectors);
            scatter_left_sector_blocks(
                v_target,
                &mut v_data,
                sector_geometry,
                &index,
                &v_groups,
                &vectors,
                n,
            )?;
            #[cfg(test)]
            crate::probe::output_scatter::<D>(crate::probe::CopyOp::Eigh, 1, n * n);
        }
        Ok(())
    })?;

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

pub(super) fn eigh_full_direct_regions<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    validate_endomorphism_tree_stacking(
        plan.source_regions.as_ref(),
        EndomorphismFamily::EighFull,
    )?;
    validate_hermitian_regions(input.data(), &plan.source_regions, hermitian_tol)?;

    let v_space = input.space().rebind_validated(&plan.left_layout)?;
    let v_len = plan.left_layout.required_len()?;
    let max_n = plan
        .source_regions
        .iter()
        .map(CoupledSectorRegion::rows)
        .max()
        .unwrap_or(0);
    let mut scratch = EighScratch::for_order(max_n);
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    let mut regions = vec![None; plan.left_regions.len()];
    let mut next_left_region = 0;
    let mut output = None;

    let data = input.data();
    in_linalg_scope(engine, dense, |dense, engine| {
        for route in plan.routes.iter().copied() {
            let source = &plan.source_regions[route.source_region];
            let n = source.rows();
            if n == 0 {
                eigenvalues.push(SectorSpectrum {
                    sector: route.sector,
                    values: Vec::new(),
                });
                continue;
            }
            let (sorted_values, vectors) =
                eigh_sector_stage(engine, dense, &data[source.range()], n, &mut scratch)?;
            regions[route.left_region.expect("nonzero route has left region")] = Some(vectors);
            while next_left_region < plan.left_regions.len() {
                if plan.left_regions[next_left_region].range().is_empty() {
                    next_left_region += 1;
                    continue;
                }
                let Some(region) = regions[next_left_region].take() else {
                    break;
                };
                append_owned_factor(&mut output, region, v_len);
                next_left_region += 1;
            }
            eigenvalues.push(SectorSpectrum {
                sector: route.sector,
                values: sorted_values,
            });
        }
        Ok(())
    })?;

    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, output.unwrap_or_default(), space.nout(), 1)?,
        eigenvalues,
    })
}

/// Forms the full eigenbasis of an owned compact diagonal directly, MAK
/// `eigh_full!(::Diagonal, …, ::DiagonalAlgorithm)` in symtn's eigenvalue
/// order.
pub(super) fn eigh_full_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let bond =
        hermitian_diagonal_bond(&MfAuthority(authority), authority, spectrum, hermitian_tol)?;
    let authority = bond.space();
    let space = authority.space();
    let plan = compact_factor_plan(authority)?.ok_or(missing_compact_plan())?;
    let by_sector = &bond.by_sector;
    let v_space = authority.rebind_validated(&plan.left_layout)?;
    let mut v_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    for route in plan.routes.iter().copied() {
        let entry = by_sector[&route.sector];
        let n = entry.values.len();
        // An empty sector has no left region and writes nothing.
        let start = route
            .left_region
            .map_or(0, |index| plan.left_regions[index].range().start);
        eigenvalues.push(SectorSpectrum {
            sector: route.sector,
            values: compact_diagonal_eigh_sector(&entry.values, &mut v_data[start..start + n * n]),
        });
    }
    Ok(EighFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

/// Sorts one admitted real diagonal sector into [`ascending_eigh_order`]
/// and writes the matching permutation into the zeroed column-major `n x n`
/// `vectors` (MAK `eigh_full!(::Diagonal, …, ::DiagonalAlgorithm)`).
fn compact_diagonal_eigh_sector<D: FactorScalar>(values: &[D], vectors: &mut [D]) -> Vec<f64> {
    let n = values.len();
    let real: Vec<f64> = values
        .iter()
        .map(|value| value.widen_complex().re)
        .collect();
    let mut order = vec![0; n];
    ascending_eigh_order(&real, &mut order);
    for (column, &row) in order.iter().enumerate() {
        vectors[column * n + row] = D::from_real(1.0);
    }
    order.into_iter().map(|index| real[index]).collect()
}

/// Checked-provider full eigenbasis of an owned compact diagonal; output
/// uses the checked factor builder over the bond's region geometry.
pub(super) fn eigh_full_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let bond = hermitian_diagonal_bond(
        &CheckedAuthority(authority.provider_arc()),
        authority,
        spectrum,
        hermitian_tol,
    )?;
    let authority = bond.space();
    let space = authority.space();
    let regions = &bond.regions;
    let by_sector = &bond.by_sector;
    let mut eigenvalues = Vec::with_capacity(regions.len());
    let mut pairs = Vec::with_capacity(regions.len());
    let mut dimensions = BTreeMap::new();
    for region in regions.iter() {
        let entry = by_sector[&region.coupled()];
        let n = region.rows();
        let mut vectors = vec![D::zero(); n * n];
        eigenvalues.push(SectorSpectrum {
            sector: region.coupled(),
            values: compact_diagonal_eigh_sector(&entry.values, &mut vectors),
        });
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: n,
            left: vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
        dimensions.insert(region.coupled(), n);
    }
    let v = if factor_bond_is_input_bond(space, || {
        SectorLeg::new(
            dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
            false,
        )
    }) {
        factor_on_input_space(authority, regions, pairs.into_iter().map(|pair| pair.left))?
    } else {
        build_bound_factor_generic_checked(
            authority.provider_arc(),
            space.homspace(),
            regions,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?
    };
    Ok(EighFullDyn { v, eigenvalues })
}

pub(crate) fn eigenvector_gauge<D: FactorScalar>(
    vectors: &mut [D],
    rows: usize,
    leading: usize,
    cols: usize,
) {
    for j in 0..cols {
        let (phase, needs_scaling) = phase_of_largest_abs_col(vectors, rows, leading, j);
        if needs_scaling {
            scale_col(vectors, rows, leading, j, FactorScalar::adjoint(phase));
        }
    }
}

#[cfg(test)]
/// Full general eigendecomposition `t = V * D * V^-1` (MatrixAlgebraKit
/// `eig_full`): always complex, requires an endomorphism. Bond states are
/// stored per sector in [`lexicographic_eig_order`].
#[derive(Clone, Debug)]
pub(crate) struct EigFull<R, D: FactorScalar> {
    pub d: BoundTensorMap<R, D::Eig>,
    pub v: BoundTensorMap<R, D::Eig>,
    pub eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

/// Dynamic-rank `EigFull`. Spectrum + eigenvectors only; the dense diagonal
/// is materialized by the typed `eig_full` wrapper (see [`EighFullDyn`]).
#[derive(Clone, Debug)]
pub struct EigFullDyn<R, D: FactorScalar> {
    pub(super) v: BoundDynFactor<R, D::Eig>,
    pub(super) eigenvalues: Vec<SectorSpectrum<Complex64>>,
}

impl<R, D: FactorScalar> EigFullDyn<R, D> {
    pub fn v(&self) -> &BoundDynFactor<R, D::Eig> {
        &self.v
    }

    pub fn eigenvalues(&self) -> &[SectorSpectrum<Complex64>] {
        &self.eigenvalues
    }

    pub fn into_parts(self) -> (BoundDynFactor<R, D::Eig>, Vec<SectorSpectrum<Complex64>>) {
        (self.v, self.eigenvalues)
    }
}

#[cfg(test)]
/// Full general eigendecomposition through the device boundary.
pub(crate) fn eig_full<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<EigFull<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let dynamic = input.dynamic();
    let out = eig_full_dyn::<E, R, D>(engine, dense, &dynamic)?;
    // Materialize the dense diagonal here (this wrapper returns a dense `D`); the
    // dyn producer does not build it.
    let d = diagonal_bond_svd_factor(
        dynamic.space(),
        &out.eigenvalues,
        &<D::Eig as FactorScalar>::from_complex64,
    )?;
    Ok(EigFull {
        d: typed_from_bound_factor(d)?,
        v: typed_from_bound_factor(out.v)?,
        eigenvalues: out.eigenvalues,
    })
}

/// Dynamic-rank `eig_full`.
pub fn eig_full_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    let matricizations = admit_endomorphism(
        space,
        input.data(),
        EndomorphismFamily::EigFull,
        multiplicity_free_input_matricizations,
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eig)?;

    let mut pairs: Vec<FactorPair<D::Eig>> = Vec::with_capacity(matricizations.len());
    let mut eigenvalues = Vec::with_capacity(matricizations.len());
    for index in 0..matricizations.len() {
        let matrix = matricizations.get(index)?;
        let n = matrix.rows;
        let (sorted_values, sorted_vectors) = eig_sector_stage(engine, dense, matrix.data, n)?;
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: n,
            left: sorted_vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: n,
        });
    }

    let v_factor = with_input_geometry!(&matricizations, |geometry| build_left_bound_factor(
        input.space(),
        space.homspace(),
        geometry,
        &mut pairs,
    ))?;
    Ok(EigFullDyn {
        v: v_factor,
        eigenvalues,
    })
}

/// Forms the full eigenbasis of an admitted owned compact diagonal directly.
pub(super) fn eig_full_diagonal_dyn<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<EigFullDyn<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let bond = diagonal_bond(
        &MfAuthority(authority),
        authority,
        spectrum,
        FactorFamily::Eig,
    )?;
    validate_diagonal_eigenvalues(&bond)?;
    let authority = bond.space();
    let space = authority.space();
    let plan = compact_factor_plan(authority)?.ok_or(missing_compact_plan())?;
    let by_sector = &bond.by_sector;
    let v_space = authority.rebind_validated(&plan.left_layout)?;
    let mut v_data = vec![<D::Eig as num_traits::Zero>::zero(); plan.left_layout.required_len()?];
    let mut eigenvalues = Vec::with_capacity(plan.routes.len());
    for route in plan.routes.iter().copied() {
        let entry = by_sector[&route.sector];
        let n = entry.values.len();
        // An empty sector has no left region and writes nothing.
        let start = route
            .left_region
            .map_or(0, |index| plan.left_regions[index].range().start);
        eigenvalues.push(SectorSpectrum {
            sector: route.sector,
            values: compact_diagonal_eig_sector(&entry.values, &mut v_data[start..start + n * n]),
        });
    }
    Ok(EigFullDyn {
        v: BoundDynFactor::from_bound(v_space, v_data, space.nout(), 1)?,
        eigenvalues,
    })
}

/// Sorts one admitted complex diagonal sector into
/// [`lexicographic_eig_order`] and writes the matching permutation into the
/// zeroed column-major `n x n` `vectors` (MAK `eig_full!(::Diagonal, …,
/// ::DiagonalAlgorithm)`).
fn compact_diagonal_eig_sector<D: FactorScalar>(
    values: &[D],
    vectors: &mut [D::Eig],
) -> Vec<Complex64> {
    let n = values.len();
    let complex: Vec<Complex64> = values.iter().map(|value| value.widen_complex()).collect();
    let mut order = vec![0; n];
    lexicographic_eig_order(&complex, &mut order);
    for (column, &row) in order.iter().enumerate() {
        vectors[column * n + row] =
            <D::Eig as FactorScalar>::from_complex64(Complex64::new(1.0, 0.0));
    }
    order.into_iter().map(|index| complex[index]).collect()
}

/// Checked-provider full general eigenbasis of an owned compact diagonal;
/// output uses the checked factor builder over the bond's region geometry.
pub(super) fn eig_full_diagonal_dyn_checked_generic<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<EigFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let bond = diagonal_bond(
        &CheckedAuthority(authority.provider_arc()),
        authority,
        spectrum,
        FactorFamily::Eig,
    )?;
    validate_diagonal_eigenvalues(&bond)?;
    let authority = bond.space();
    let space = authority.space();
    let regions = &bond.regions;
    let by_sector = &bond.by_sector;
    let mut eigenvalues = Vec::with_capacity(regions.len());
    let mut pairs = Vec::with_capacity(regions.len());
    let mut dimensions = BTreeMap::new();
    for region in regions.iter() {
        let entry = by_sector[&region.coupled()];
        let n = region.rows();
        let mut vectors = vec![D::Eig::zero(); n * n];
        eigenvalues.push(SectorSpectrum {
            sector: region.coupled(),
            values: compact_diagonal_eig_sector(&entry.values, &mut vectors),
        });
        pairs.push(FactorPair {
            sector: region.coupled(),
            kept: n,
            left: vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
        dimensions.insert(region.coupled(), n);
    }
    let v = if factor_bond_is_input_bond(space, || {
        SectorLeg::new(
            dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
            false,
        )
    }) {
        factor_on_input_space(authority, regions, pairs.into_iter().map(|pair| pair.left))?
    } else {
        build_bound_factor_generic_checked(
            authority.provider_arc(),
            space.homspace(),
            regions,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?
    };
    Ok(EigFullDyn { v, eigenvalues })
}

/// Hermitian eigenvalues of a compact diagonal: the real parts, after the
/// dense route's Hermiticity check, in symtn's eigenvalue order.
fn eigh_vals_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<Vec<SectorSpectrum>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = hermitian_diagonal_bond(authority, space, spectrum, hermitian_tol)?;
    let mut result = Vec::with_capacity(bond.len());
    for region in bond.iter() {
        let mut values: Vec<f64> = bond
            .entry(region)
            .values
            .iter()
            .map(|&value| value.widen_complex().re)
            .collect();
        validate_real_eigenvalues(&values)?;
        sort_spectrum(&mut values, f64::total_cmp);
        result.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    Ok(result)
}

/// A finite compact diagonal that passes the dense route's
/// relative-Frobenius Hermiticity check at `hermitian_tol`, applied to the
/// diagonal matrix itself; its real parts are its eigenvalues.
fn hermitian_diagonal_bond<'a, A, R, D>(
    authority: &A,
    space: &'a BoundDynamicFusionMapSpace<R>,
    spectrum: &'a [SectorSpectrum<D>],
    hermitian_tol: HermitianTol,
) -> Result<DiagonalBond<'a, R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Eigh)?;
    let tol = hermitian_tol.resolve(D::epsilon());
    for region in bond.iter() {
        let values = &bond.entry(region).values;
        let hermitian = if D::epsilon() == f32::EPSILON as f64 {
            normwise_hermitian_diagonal::<D, f32>(values, f32::from_f64(tol))
        } else if D::epsilon() == f64::EPSILON {
            normwise_hermitian_diagonal::<D, f64>(values, tol)
        } else {
            false
        };
        if !hermitian {
            return Err(OperationError::InvalidArgument {
                message: EIGH_NOT_HERMITIAN,
            }
            .into());
        }
    }
    Ok(bond)
}

/// [`normwise_hermitian`] of the diagonal matrix with diagonal `values`,
/// without forming it. Off-diagonal zeros and the zero real residuals of the
/// diagonal add nothing to either scaled sum, and both sums visit the
/// diagonal in the same order as the dense scan, so the decision is equal.
fn normwise_hermitian_diagonal<D: FactorScalar, R: HermitianReal>(values: &[D], tol: R) -> bool {
    let mut input = ScaledFrobenius::zero();
    for &value in values {
        let value = value.widen_complex();
        let re = R::from_f64(value.re);
        let im = R::from_f64(value.im);
        if !re.is_finite() || !im.is_finite() || !input.add_complex(re, im) {
            return false;
        }
    }
    if input.scale == R::zero() {
        return true;
    }
    let mut residual = ScaledFrobenius::zero();
    for &value in values {
        let value = value.widen_complex();
        let re = R::from_f64(value.re) / input.scale - R::from_f64(value.re) / input.scale;
        let im = R::from_f64(value.im) / input.scale + R::from_f64(value.im) / input.scale;
        if !residual.add_complex(re, im) {
            return false;
        }
    }
    residual.scaled_norm(R::one()) <= (R::one() + R::one()) * tol * input.sum_squares.sqrt()
}

/// General eigenvalues of a compact diagonal: the values themselves in
/// [`lexicographic_eig_order`], after the dense route's eigenvalue check.
fn eig_vals_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<Vec<SectorSpectrum<Complex64>>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Eig)?;
    let mut result = Vec::with_capacity(bond.len());
    for region in bond.iter() {
        let mut values: Vec<Complex64> = bond
            .entry(region)
            .values
            .iter()
            .map(|&value| value.widen_complex())
            .collect();
        validate_complex_eigenvalues(&values)?;
        sort_spectrum(&mut values, lexicographic_eig_cmp);
        result.push(SectorSpectrum {
            sector: region.coupled(),
            values,
        });
    }
    Ok(result)
}

fn missing_compact_plan() -> OperationError {
    OperationError::UnsupportedTensorContractScope {
        message: "canonical bond layout has no compact factor plan",
    }
}

#[cfg(test)]
/// All Hermitian eigenvalues per coupled sector, ascending
/// (MatrixAlgebraKit `eigh_vals`).
///
/// Admits Hermitian blocks at [`HermitianTol::DEFAULT`].
pub(crate) fn eigh_vals<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    eigh_vals_dyn(engine, dense, &input.dynamic(), HermitianTol::DEFAULT)
}

/// Dynamic-rank `eigh_vals`: the dense stage, the same in every fusion mode.
pub fn eigh_vals_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    // Values-only: per sector call the no-vector Hermitian eig (`eigh_vals`,
    // LAPACK `job='N'`) and publish the spectrum ascending. Skips the
    // eigenvector space/buffer, the vector reorder, gauge-fixing, and the block
    // scatter that `eigh_full_dyn` did only to discard here.
    let matricizations = admit_endomorphism(
        space,
        input.data(),
        EndomorphismFamily::EighVals,
        value_matricizations,
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
    matricizations.validate_hermitian(hermitian_tol)?;
    eigh_vals_spectra(engine, dense, &matricizations)
}

#[cfg(test)]
/// All general eigenvalues per coupled sector in [`lexicographic_eig_order`]
/// (MatrixAlgebraKit `eig_vals`).
pub(crate) fn eig_vals<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    eig_vals_dyn::<E, R, D>(engine, dense, &input.dynamic())
}

/// Dynamic-rank `eig_vals`: the dense stage, the same in every fusion mode.
pub fn eig_vals_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    // Values-only: per sector call the no-vector general eig (`eig_vals`, LAPACK
    // `job='N'`) and publish the complex spectrum in lexicographic order.
    // Skips the eigenvector reorder, gauge-fixing, and the factor-pair block
    // assembly that `eig_full_dyn` did only to discard here.
    let matricizations = admit_endomorphism(
        space,
        input.data(),
        EndomorphismFamily::EigVals,
        value_matricizations,
    )?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eig)?;
    eig_vals_spectra(engine, dense, &matricizations)
}

/// Reorder scratch for [`eigh_sector_stage`] and [`eigh_sector_into`], sized
/// for the largest sector.
#[doc(hidden)]
pub struct EighScratch<D: FactorScalar> {
    order: Vec<usize>,
    visited: Vec<bool>,
    column: Vec<D>,
    /// The solver's values of [`eigh_sector_into`], in its order.
    real: Vec<f64>,
}

impl<D: FactorScalar> EighScratch<D> {
    pub fn for_order(max_n: usize) -> Self {
        Self {
            order: Vec::with_capacity(max_n),
            visited: vec![false; max_n],
            column: vec![D::zero(); max_n],
            real: Vec::with_capacity(max_n),
        }
    }
}

/// Orders one solved sector: validates the solver's `real_values`, reorders
/// the `n x n` eigenvectors to [`ascending_eigh_order`] (only when the
/// solver's order is not already ascending) and phase-gauges them. Returns
/// whether `scratch.order` holds a reordering.
#[inline]
fn order_eigh_sector<D: FactorScalar>(
    real_values: &[f64],
    vectors: &mut [D],
    n: usize,
    scratch: &mut EighScratch<D>,
) -> Result<bool, OperationError> {
    validate_real_eigenvalues(real_values)?;
    let EighScratch {
        order,
        visited,
        column,
        ..
    } = scratch;
    order.clear();
    order.resize(n, 0);
    let reordered = ascending_eigh_order(real_values, order);
    if reordered {
        reorder_columns_in_place(vectors, n, order, visited, column);
    }
    eigenvector_gauge(vectors, n, n, n);
    Ok(reordered)
}

/// Hermitian eigendecomposition of one `n x n` coupled-sector matrix: values
/// in [`ascending_eigh_order`], eigenvectors reordered to match and
/// phase-gauged ([`order_eigh_sector`]).
#[inline]
fn eigh_sector_stage<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrix: &[D],
    n: usize,
    scratch: &mut EighScratch<D>,
) -> Result<(Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (real_values, mut vectors) = compact_eigh_owned(engine, dense, matrix, n)?;
    let sorted_values = if order_eigh_sector(&real_values, &mut vectors, n, scratch)? {
        scratch
            .order
            .iter()
            .map(|&index| real_values[index])
            .collect()
    } else {
        real_values
    };
    Ok((sorted_values, vectors))
}

/// [`eigh_sector_stage`] for a prepared caller: the ascending values land in
/// `values` (length `n`) and the ordered, gauge-fixed eigenvectors
/// (column-major `n x n`) are returned in the solver's own output buffer for
/// the caller to place. Same solver, order and gauge as eager, and no
/// allocation beyond the executor's outputs. The input must already be
/// admitted (endomorphism stacking, finite, Hermitian).
///
/// Why not the executor's `eigh_into`: its default copies the owned outputs
/// into the destinations through a strided kernel that allocates per call,
/// more than the one contiguous copy the caller makes.
#[doc(hidden)]
pub fn eigh_sector_into<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrix: &[D],
    n: usize,
    scratch: &mut EighScratch<D>,
    values: &mut [f64],
) -> Result<Vec<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (values_tensor, vectors_tensor) = compact_eigh_outputs(engine, dense, matrix, n)?;
    let mut real = std::mem::take(&mut scratch.real);
    let result = (|| {
        D::real_spectrum_into(&values_tensor, &mut real).map_err(OperationError::Dense)?;
        check_contiguous_output("eigh_into", real.len(), values_tensor.shape(), &[n])
            .map_err(OperationError::Dense)?;
        let mut vectors = compact_factor_output_owned::<D>(vectors_tensor, &[n, n], "eigh_into")?;
        if order_eigh_sector(&real, &mut vectors, n, scratch)? {
            for (slot, &index) in values.iter_mut().zip(&scratch.order) {
                *slot = real[index];
            }
        } else {
            values.copy_from_slice(&real);
        }
        Ok(vectors)
    })();
    scratch.real = real;
    result
}

/// The finite-input admission of [`eigh_full_dyn`], for a prepared caller.
#[doc(hidden)]
pub fn require_finite_eigh_input<D: FactorScalar>(data: &[D]) -> Result<(), OperationError> {
    require_finite_factor_input(data.iter().copied(), FactorFamily::Eigh)
}

/// General eigendecomposition of one `n x n` endomorphism sector, shared by
/// both fusion modes: values in [`lexicographic_eig_order`], eigenvectors
/// reordered to match, scaled to unit 2-norm and phase-gauged.
///
/// Result checks, in this order: eigenvalues finite
/// ([`validate_complex_eigenvalues`]), then every eigenvector entry finite
/// ([`validate_eigenvectors`]). Both are the solver's numerical failure on an
/// admitted finite input; faer's back-substitution overflows to NaN on large
/// Jordan blocks where LAPACK `xTREVC` rescales, and a nonfinite column has
/// no unit-norm representative to publish.
///
/// Borrowed contract: MatrixAlgebraKit v0.6.8 `eig_full_qr_iteration!`
/// (<https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/33d77fdf032d5ffa36310a04c21aef0c51a554f3/src/implementations/eig.jl#L117>)
/// runs `geev!` (unit 2-norm columns) and then `gaugefix!`, promising
/// `A V = V D` to backward error and certifying no eigenvector rank, so a
/// defective sector returns its computed eigenvectors. Why no rank SVD here: a numerical-rank threshold on
/// `V` neither detects every defective input nor changes the eigenpair
/// equation, and `V^-1` reconstruction is promised only for diagonalizable
/// inputs. The callers' strict finite-input stage is symtn's approved
/// deviation (#1983); MAK passes non-finite input to LAPACK.
fn eig_sector_stage<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrix: &[D],
    n: usize,
) -> Result<(Vec<Complex64>, Vec<D::Eig>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [n, n];
    let strides = [1usize, n];
    let view = DenseView::new(matrix, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .eig(engine, D::dense_read(view))
        .map_err(OperationError::Dense)?;
    if outputs.len() != 2 {
        return Err(OperationError::Dense(arity_mismatch(
            "eig",
            2,
            outputs.len(),
        )));
    }
    validate_dense_shape(outputs[0].shape(), &[n])?;
    validate_dense_shape(outputs[1].shape(), &[n, n])?;
    let values =
        <D::Eig as FactorScalar>::dense_slice(&outputs[0]).map_err(OperationError::Dense)?;
    let vectors =
        <D::Eig as FactorScalar>::dense_slice(&outputs[1]).map_err(OperationError::Dense)?;
    let complex_values: Vec<Complex64> =
        values.iter().map(|&value| value.widen_complex()).collect();
    validate_complex_eigenvalues(&complex_values)?;
    validate_eigenvectors(vectors)?;
    let mut order = vec![0; n];
    lexicographic_eig_order(&complex_values, &mut order);
    let sorted_values: Vec<Complex64> = order.iter().map(|&index| complex_values[index]).collect();
    let mut sorted_vectors = vec![<D::Eig as num_traits::Zero>::zero(); n * n];
    for (position, &index) in order.iter().enumerate() {
        unit_norm_column_into(
            &vectors[index * n..(index + 1) * n],
            &mut sorted_vectors[position * n..(position + 1) * n],
        );
    }
    eigenvector_gauge(&mut sorted_vectors, n, n, n);
    Ok((sorted_values, sorted_vectors))
}

/// Writes `source / ‖source‖₂` into `dest`: LAPACK `geev`'s unit 2-norm
/// eigenvector columns, which MAK `eig_full_qr_iteration!` publishes before
/// `gaugefix!` (#2073). Providers that already normalize (LAPACK) are
/// unchanged up to rounding; faer does not normalize. The phase gauge runs
/// after, as in MAK; it is a unit-modulus scaling, so the two commute.
///
/// The column is first divided by its largest magnitude and the arithmetic
/// runs in `f64`, so neither huge nor tiny finite columns overflow or
/// underflow, at single precision too. The caller has rejected nonfinite
/// entries ([`validate_eigenvectors`]); an all-zero column (no eigenvector)
/// is copied unscaled rather than turned into NaN.
fn unit_norm_column_into<D: FactorScalar>(source: &[D], dest: &mut [D]) {
    let largest = source.iter().fold(0.0f64, |largest, value| {
        largest.max(value.widen_complex().norm())
    });
    if !(largest > 0.0 && largest.is_finite()) {
        dest.copy_from_slice(source);
        return;
    }
    let relative_norm = source
        .iter()
        .map(|value| (value.widen_complex() / largest).norm_sqr())
        .sum::<f64>()
        .sqrt();
    for (slot, &value) in dest.iter_mut().zip(source) {
        *slot = D::from_complex64(value.widen_complex() / largest / relative_norm);
    }
}

/// Hermitian eigenvalues of every sector through the no-vector solver,
/// ascending.
fn eigh_vals_spectra<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<Vec<SectorSpectrum>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eigh_vals(engine, D::dense_read(view))
            .map_err(OperationError::Dense)?;
        let mut sorted = D::real_spectrum(&values_tensor).map_err(OperationError::Dense)?;
        sorted.truncate(n);
        validate_real_eigenvalues(&sorted)?;
        sort_spectrum(&mut sorted, f64::total_cmp);
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted,
        });
    }
    Ok(eigenvalues)
}

/// General eigenvalues of every sector through the no-vector solver, in
/// [`lexicographic_eig_order`].
fn eig_vals_spectra<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<Vec<SectorSpectrum<Complex64>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let n = matrix.rows;
        let shape = [matrix.rows, matrix.cols];
        let strides = [1usize, matrix.rows];
        let view =
            DenseView::new(matrix.data, &shape, &strides, 0).map_err(OperationError::Dense)?;
        let values_tensor = dense
            .eig_vals(engine, D::dense_read(view))
            .map_err(OperationError::Dense)?;
        validate_dense_shape(values_tensor.shape(), &[n])?;
        let values =
            <D::Eig as FactorScalar>::dense_slice(&values_tensor).map_err(OperationError::Dense)?;
        let mut sorted: Vec<Complex64> = values[..n].iter().map(|&v| v.widen_complex()).collect();
        validate_complex_eigenvalues(&sorted)?;
        sort_spectrum(&mut sorted, lexicographic_eig_cmp);
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted,
        });
    }
    Ok(eigenvalues)
}

pub(crate) fn compact_eigh_owned<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &[D],
    order: usize,
) -> Result<(Vec<f64>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (values, vectors) = compact_eigh_outputs(engine, dense, input, order)?;
    let values = compact_real_spectrum_owned::<D>(values, &[order], "eigh_into")?;
    let vectors = compact_factor_output_owned::<D>(vectors, &[order, order], "eigh_into")?;
    Ok((values, vectors))
}

/// The executor's `(values, vectors)` of one `order x order` Hermitian
/// matrix, arity checked.
fn compact_eigh_outputs<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &[D],
    order: usize,
) -> Result<(DenseTensor, DenseTensor), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [order, order];
    let strides = [1usize, order];
    let input = DenseView::new(input, &shape, &strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .eigh(engine, D::dense_read(input))
        .map_err(OperationError::Dense)?;
    let [values, vectors]: [DenseTensor; 2] = outputs.try_into().map_err(|outputs: Vec<_>| {
        OperationError::Dense(arity_mismatch("eigh_into", 2, outputs.len()))
    })?;
    Ok((values, vectors))
}

pub(super) fn scaled_antihermitian_difference<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    row: usize,
    col: usize,
    scale: R,
) -> (R, R) {
    let upper = data[row + n * col].widen_complex();
    let lower = data[col + n * row].widen_complex();
    let residual_re = R::from_f64(upper.re) / scale - R::from_f64(lower.re) / scale;
    let residual_im = R::from_f64(upper.im) / scale + R::from_f64(lower.im) / scale;
    (residual_re, residual_im)
}

#[derive(Clone, Copy)]
pub(super) struct ScaledFrobenius<R> {
    pub(super) scale: R,
    pub(super) sum_squares: R,
}

impl<R: HermitianReal> ScaledFrobenius<R> {
    pub(super) fn zero() -> Self {
        Self {
            scale: R::zero(),
            sum_squares: R::one(),
        }
    }

    pub(super) fn add(&mut self, magnitude: R) -> bool {
        if !magnitude.is_finite() {
            return false;
        }
        if magnitude == R::zero() {
            return true;
        }
        if self.scale < magnitude {
            let ratio = self.scale / magnitude;
            self.sum_squares = R::one() + self.sum_squares * ratio * ratio;
            self.scale = magnitude;
        } else {
            let ratio = magnitude / self.scale;
            self.sum_squares = self.sum_squares + ratio * ratio;
        }
        true
    }

    pub(super) fn add_complex(&mut self, re: R, im: R) -> bool {
        self.add(re.abs()) && self.add(im.abs())
    }

    pub(super) fn scaled_norm(self, scale: R) -> R {
        if self.scale == R::zero() {
            R::zero()
        } else {
            (self.scale / scale) * self.sum_squares.sqrt()
        }
    }
}

pub(super) fn hermitian_residual_norm<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    input_scale: R,
) -> Option<ScaledFrobenius<R>> {
    const BLOCK_SIZE: usize = 32;
    let mut residual = ScaledFrobenius::zero();
    for block_col in (0..n).step_by(BLOCK_SIZE) {
        let block_width = BLOCK_SIZE.min(n - block_col);
        for local_col in 0..block_width {
            let col = block_col + local_col;
            for local_row in 0..=local_col {
                let row = block_col + local_row;
                let (re, im) =
                    scaled_antihermitian_difference::<D, R>(data, n, row, col, input_scale);
                if !residual.add_complex(re, im) || (row != col && !residual.add_complex(re, im)) {
                    return None;
                }
            }
        }

        for block_row in (0..block_col).step_by(BLOCK_SIZE) {
            for local_col in 0..block_width {
                let col = block_col + local_col;
                for local_row in 0..BLOCK_SIZE {
                    let row = block_row + local_row;
                    let (re, im) =
                        scaled_antihermitian_difference::<D, R>(data, n, row, col, input_scale);
                    if !residual.add_complex(re, im) || !residual.add_complex(re, im) {
                        return None;
                    }
                }
            }
        }
    }
    Some(residual)
}

/// Tests `||(A - A†)/2||_F <= tol * ||A||_F`.
///
/// The scaled sums keep that relative decision stable when either norm would
/// overflow or underflow if formed directly.
pub(super) fn normwise_hermitian<D: FactorScalar, R: HermitianReal>(
    data: &[D],
    n: usize,
    tol: R,
) -> bool {
    let mut input = ScaledFrobenius::zero();
    for &value in data {
        let value = value.widen_complex();
        let re = R::from_f64(value.re);
        let im = R::from_f64(value.im);
        if !re.is_finite() || !im.is_finite() {
            return false;
        }
        if !input.add_complex(re, im) {
            return false;
        }
    }

    if input.scale == R::zero() {
        return true;
    }
    let Some(residual) = hermitian_residual_norm::<D, R>(data, n, input.scale) else {
        return false;
    };
    residual.scaled_norm(R::one()) <= (R::one() + R::one()) * tol * input.sum_squares.sqrt()
}

pub(super) fn validate_hermitian_matrix_shape<D>(
    data: &[D],
    rows: usize,
    cols: usize,
) -> Result<(), OperationError> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(OperationError::ElementCountOverflow)?;
    if data.len() != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: data.len(),
        });
    }
    if rows != cols {
        return Err(OperationError::SpaceMismatch {
            message: EIGH_NOT_SQUARE,
        });
    }
    Ok(())
}

/// Scale-invariant normwise hermiticity predicate at the working precision of
/// `D`, with relative tolerance `tol`.
///
/// Extracted from [`validate_hermitian_matrix_contents`] so the `exp` dispatch
/// (issue #577) can ask the question without provoking — and then having to
/// interpret — an EIGH failure. Same measure; `exp` passes its own route
/// threshold ([`EXP_SPECTRAL_ROUTE_EPSILONS`]).
pub(super) fn hermitian_matrix_contents<D: FactorScalar>(data: &[D], n: usize, tol: f64) -> bool {
    if D::epsilon() == f32::EPSILON as f64 {
        normwise_hermitian::<D, f32>(data, n, f32::from_f64(tol))
    } else if D::epsilon() == f64::EPSILON {
        normwise_hermitian::<D, f64>(data, n, tol)
    } else {
        false
    }
}

pub(super) fn validate_hermitian_matrix_contents<D: FactorScalar>(
    data: &[D],
    n: usize,
    hermitian_tol: HermitianTol,
) -> Result<(), OperationError> {
    if !hermitian_matrix_contents(data, n, hermitian_tol.resolve(D::epsilon())) {
        return Err(OperationError::InvalidArgument {
            message: EIGH_NOT_HERMITIAN,
        });
    }
    Ok(())
}

pub(super) fn validate_hermitian_matricizations<D: FactorScalar>(
    matricizations: &[SectorMatricization<D>],
    hermitian_tol: HermitianTol,
) -> Result<(), OperationError> {
    for matrix in matricizations {
        validate_hermitian_matrix_shape(&matrix.data, matrix.rows, matrix.cols)?;
    }
    for matrix in matricizations {
        validate_hermitian_matrix_contents(&matrix.data, matrix.rows, hermitian_tol)?;
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_hermitian_regions<D: FactorScalar>(
    data: &[D],
    regions: &[CoupledSectorRegion],
    hermitian_tol: HermitianTol,
) -> Result<(), OperationError> {
    for region in regions {
        let range = region.range();
        let matrix = data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: data.len(),
            })?;
        validate_hermitian_matrix_shape(matrix, region.rows(), region.cols())?;
    }
    for region in regions {
        let range = region.range();
        let matrix = data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: data.len(),
            })?;
        validate_hermitian_matrix_contents(matrix, region.rows(), hermitian_tol)?;
    }
    Ok(())
}

/// Is this an endomorphism whose coupled-sector blocks are all Hermitian?
///
/// The `exp` dispatch (issue #577) needs the Hermitian question answered
/// *separately* from the eigendecomposition: the spectral route stays for
/// Hermitian input, and everything else goes to blockwise Padé. Inferring it
/// from a failed EIGH would conflate hermiticity with a backend failure, so
/// this asks directly, over the same direct-region / packed matricization
/// split and relative Frobenius measure as [`eigh_full_dyn`], with the
/// separate threshold [`EXP_SPECTRAL_ROUTE_EPSILONS`].
///
/// A non-endomorphism, a malformed layout or a non-square block is still an
/// error — only non-hermiticity is `Ok(false)`. Nonfinite entries make
/// the predicate `false`, so they arrive at the Padé route, which rejects them
/// in its own words.
///
/// Cost is `O(Σ_c n_c²)`, one pass over the blocks, against the `O(Σ_c n_c³)`
/// factorization that follows.
pub(crate) fn is_hermitian_endomorphism_dyn<R, D>(
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<bool, OperationError>
where
    D: FactorScalar,
{
    let space = input.space().space();
    // `exp` asks its own endomorphism question first, so this refusal is
    // reached only by a direct caller.
    EndomorphismFamily::EighFull.require_endomorphism(space.homspace())?;
    let exp_route_tol = EXP_SPECTRAL_ROUTE_EPSILONS * D::epsilon();
    // Why `checked_sector_regions` and not `compact_factor_plan` as
    // `eigh_full_dyn` does: the plan is `Some` exactly when the regions are
    // (`build_compact_factor_plan` returns early otherwise), and building it
    // also builds the factor bond spaces — work a yes/no question must not pay
    // for on the retained Hermitian route.
    if let Some(regions) = checked_sector_regions(space.structure(), space.nout())? {
        validate_endomorphism_tree_stacking(regions.as_ref(), EndomorphismFamily::Exp)?;
        for region in regions.iter() {
            let range = region.range();
            let matrix = data_region(input.data(), &range)?;
            validate_hermitian_matrix_shape(matrix, region.rows(), region.cols())?;
        }
        for region in regions.iter() {
            let range = region.range();
            let matrix = data_region(input.data(), &range)?;
            if !hermitian_matrix_contents(matrix, region.rows(), exp_route_tol) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    validate_endomorphism_tree_stacking(&matricizations, EndomorphismFamily::Exp)?;
    for matrix in &matricizations {
        validate_hermitian_matrix_shape(&matrix.data, matrix.rows, matrix.cols)?;
    }
    Ok(matricizations
        .iter()
        .all(|matrix| hermitian_matrix_contents(&matrix.data, matrix.rows, exp_route_tol)))
}

/// The `op` of symtn's own Hermitian eigenvalue check, so a caller can tell
/// it from an executor's `NumericalFailure` by matching it exactly.
#[doc(hidden)]
pub const EIGH_EIGENVALUE_CHECK: &str = "eigh eigenvalue check";

/// The `op` of symtn's own general eigenvalue check.
const EIG_EIGENVALUE_CHECK: &str = "eig eigenvalue check";

/// A nonfinite eigenvalue of an admitted finite input is the eigensolver's
/// numerical failure, not a misuse: the input already passed
/// [`require_finite_factor_input`].
pub(super) fn invalid_eigenvalues(op: &'static str) -> OperationError {
    OperationError::Dense(DenseError::NumericalFailure {
        backend: DenseBackend::Tenferro,
        op,
        message: "eigenvalues must be finite".to_string(),
    })
}

pub(crate) fn validate_real_eigenvalues(values: &[f64]) -> Result<(), OperationError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_eigenvalues(EIGH_EIGENVALUE_CHECK))
    }
}

#[cfg(test)]
pub(crate) fn validate_real_eigenvalues_for_test(values: &[f64]) -> Result<(), OperationError> {
    validate_real_eigenvalues(values)
}

/// The `op` of symtn's own general eigenvector check.
const EIG_EIGENVECTOR_CHECK: &str = "eig eigenvector check";

/// A nonfinite eigenvector entry of an admitted finite input is the
/// eigensolver's numerical failure, reported as [`invalid_eigenvalues`] is.
fn validate_eigenvectors<D: FactorScalar>(vectors: &[D]) -> Result<(), OperationError> {
    if vectors.iter().all(|value| {
        let value = value.widen_complex();
        value.re.is_finite() && value.im.is_finite()
    }) {
        Ok(())
    } else {
        Err(OperationError::Dense(DenseError::NumericalFailure {
            backend: DenseBackend::Tenferro,
            op: EIG_EIGENVECTOR_CHECK,
            message: "eigenvectors must be finite".to_string(),
        }))
    }
}

pub(super) fn validate_complex_eigenvalues(values: &[Complex64]) -> Result<(), OperationError> {
    if values
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite() && value.norm().is_finite())
    {
        Ok(())
    } else {
        Err(invalid_eigenvalues(EIG_EIGENVALUE_CHECK))
    }
}

/// Checked-Generic full Hermitian eigendecomposition. The exact source
/// provider remains the authority for the eigenvector factor.
#[doc(hidden)]
pub(crate) fn eigh_full_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    // Why not trust equal product spaces alone: outer-multiplicity vertices
    // are part of a tree key, so Hermitian coordinates require identical full
    // tree stacking, not merely equal coupled-sector dimensions.
    let matrices = admit_endomorphism(
        space,
        input.data(),
        EndomorphismFamily::EighFull,
        generic_input_matricizations,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Eigh)?;
    matrices
        .validate_hermitian(hermitian_tol)
        .map_err(CheckedGenericFactorPlanError::from)?;

    let max_n = with_input_geometry!(&matrices, |geometry| geometry
        .iter()
        .map(SectorGeometry::rows)
        .max()
        .unwrap_or(0));
    let mut scratch = EighScratch::for_order(max_n);
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    let mut pairs = Vec::with_capacity(matrices.len());
    in_linalg_scope(engine, dense, |dense, engine| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let n = matrix.rows;
            let (sorted_values, vectors) =
                eigh_sector_stage(engine, dense, matrix.data, n, &mut scratch)?;
            eigenvalues.push(SectorSpectrum {
                sector: matrix.sector,
                values: sorted_values,
            });
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: n,
                left: vectors,
                left_rows: n,
                right: Vec::new(),
                right_leading: 0,
            });
        }
        Ok(())
    })
    .map_err(CheckedGenericFactorPlanError::from)?;
    let dimensions = with_input_geometry!(&matrices, |geometry| geometry
        .iter()
        .map(|matrix| (matrix.sector(), matrix.rows()))
        .collect::<BTreeMap<_, _>>());
    #[cfg(test)]
    crate::probe::checked_eigh_pairs(
        pairs
            .iter()
            .map(|pair| pair.left.as_ptr() as usize)
            .collect(),
    );
    let v = with_input_geometry!(&matrices, |geometry| build_bound_factor_generic_checked(
        provider,
        space.homspace(),
        geometry,
        &mut pairs,
        &dimensions,
        FactorSide::Left,
    ))?;
    Ok(EighFullDyn { v, eigenvalues })
}

/// Checked-Generic full general eigendecomposition. All input components and
/// dense results are validated before the exact source provider is asked to
/// admit either output factor.
#[doc(hidden)]
pub(crate) fn eig_full_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = admit_endomorphism(
        space,
        input.data(),
        EndomorphismFamily::EigFull,
        generic_input_matricizations,
    )
    .map_err(CheckedGenericFactorPlanError::from)?;
    // Every sector is read before the shared finite-input stage reports, so
    // a matricization error still precedes it.
    let mut finite = Ok(());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        if finite.is_ok() {
            finite = require_finite_factor_input(matrix.data.iter().copied(), FactorFamily::Eig);
        }
    }
    finite.map_err(CheckedGenericFactorPlanError::from)?;

    let mut pairs: Vec<FactorPair<D::Eig>> = Vec::with_capacity(matrices.len());
    let mut eigenvalues = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let n = matrix.rows;
        let (sorted_values, sorted_vectors) = eig_sector_stage(engine, dense, matrix.data, n)
            .map_err(CheckedGenericFactorPlanError::from)?;
        eigenvalues.push(SectorSpectrum {
            sector: matrix.sector,
            values: sorted_values,
        });
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: n,
            left: sorted_vectors,
            left_rows: n,
            right: Vec::new(),
            right_leading: 0,
        });
    }

    let v = with_input_geometry!(&matrices, |geometry| {
        let dimensions = geometry
            .iter()
            .map(|matrix| (matrix.sector(), matrix.rows()))
            .collect::<BTreeMap<_, _>>();
        build_bound_factor_generic_checked(
            provider,
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )
    })?;
    Ok(EigFullDyn { v, eigenvalues })
}

/// Hermitian eigenvalues of `source` in fusion mode `M`, ascending per
/// coupled sector. A compact diagonal is read directly; only
/// dense storage leases an executor.
#[doc(hidden)]
pub fn eigh_vals_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
    hermitian_tol: HermitianTol,
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
        None,
        |space, spectrum| eigh_vals_diagonal(&M::authority(space), space, spectrum, hermitian_tol),
        |dense, input| Ok(eigh_vals_dyn(engine, dense, input, hermitian_tol)?),
    )
}

/// General eigenvalues of `source` in fusion mode `M`, in
/// [`lexicographic_eig_order`] per coupled sector. A compact diagonal is read directly; only
/// dense storage leases an executor.
#[doc(hidden)]
pub fn eig_vals_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Vec<SectorSpectrum<Complex64>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        None,
        |space, spectrum| eig_vals_diagonal(&M::authority(space), space, spectrum),
        |dense, input| Ok(eig_vals_dyn(engine, dense, input)?),
    )
}

/// Hermitian eigendecomposition of `source` in fusion mode `M`;
/// only dense storage leases an executor.
#[doc(hidden)]
pub fn eigh_full_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
    hermitian_tol: HermitianTol,
) -> Result<EighFullDyn<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        None,
        |space, spectrum| M::eigh_full_diagonal(space, spectrum, hermitian_tol),
        |dense, input| M::eigh_full_dense(engine, dense, input, hermitian_tol),
    )
}

/// General eigendecomposition of `source` in fusion mode `M`;
/// only dense storage leases an executor.
#[doc(hidden)]
pub fn eig_full_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<EigFullDyn<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(lease, source, None, M::eig_full_diagonal, |dense, input| {
        M::eig_full_dense(engine, dense, input)
    })
}

/// The dense route's eigenvalue check (`validate_complex_eigenvalues`) on a
/// compact diagonal's eigenvalues, which are its values: an overflowing
/// magnitude is refused as the dense route refuses it.
fn validate_diagonal_eigenvalues<R, D: FactorScalar>(
    bond: &DiagonalBond<'_, R, D>,
) -> Result<(), OperationError> {
    for region in bond.iter() {
        for &value in &bond.entry(region).values {
            let value = value.widen_complex();
            if !(value.re.is_finite() && value.im.is_finite() && value.norm().is_finite()) {
                return Err(invalid_eigenvalues(EIG_EIGENVALUE_CHECK));
            }
        }
    }
    Ok(())
}
