use super::*;

/// QR of `source` in fusion mode `M`: a compact diagonal factors directly on
/// its bond (`family` names its finite-input error), dense storage runs
/// `dense_qr` with a leased executor.
fn qr_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
    family: FactorFamily,
    dense_qr: impl FnOnce(
        &mut E,
        &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Qr<BoundDynFactor<R, D>>, M::Error>,
) -> Result<Qr<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        Some(family),
        |space, spectrum| {
            let (q, r) =
                diagonal_phase_magnitude_spectra(&M::authority(space), space, spectrum, family)?;
            let on_input = |values| FactorOutput::Diagonal {
                space: space.clone(),
                values,
            };
            Ok(Qr {
                q: on_input(q),
                r: on_input(r),
            })
        },
        |dense, input| {
            dense_qr(dense, input).map(|Qr { q, r }| Qr {
                q: FactorOutput::Dense(q),
                r: FactorOutput::Dense(r),
            })
        },
    )
}

/// Compact QR of `source` in fusion mode `M`.
#[doc(hidden)]
pub fn qr_compact_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Qr<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    qr_from_source::<M, _, _, _, _>(lease, source, FactorFamily::Qr, M::qr_compact_dense)
}

/// Full QR of `source` in fusion mode `M`.
#[doc(hidden)]
pub fn qr_full_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Qr<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    qr_from_source::<M, _, _, _, _>(lease, source, FactorFamily::Qr, M::qr_full_dense)
}

/// LQ of `source` in fusion mode `M`: a compact diagonal's `l` holds the
/// magnitudes and `q` the phases (see [`diagonal_phase_magnitude_spectra`]).
fn lq_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
    dense_lq: impl FnOnce(
        &mut E,
        &BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<Lq<BoundDynFactor<R, D>>, M::Error>,
) -> Result<Lq<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        lease,
        source,
        Some(FactorFamily::Lq),
        |space, spectrum| {
            let (q, r) = diagonal_phase_magnitude_spectra(
                &M::authority(space),
                space,
                spectrum,
                FactorFamily::Lq,
            )?;
            let on_input = |values| FactorOutput::Diagonal {
                space: space.clone(),
                values,
            };
            Ok(Lq {
                l: on_input(r),
                q: on_input(q),
            })
        },
        |dense, input| {
            dense_lq(dense, input).map(|Lq { l, q }| Lq {
                l: FactorOutput::Dense(l),
                q: FactorOutput::Dense(q),
            })
        },
    )
}

/// Compact LQ of `source` in fusion mode `M`.
#[doc(hidden)]
pub fn lq_compact_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Lq<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    lq_from_source::<M, _, _, _, _>(lease, source, M::lq_compact_dense)
}

/// Full LQ of `source` in fusion mode `M`.
#[doc(hidden)]
pub fn lq_full_from_source<M, L, E, R, D>(
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Lq<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    lq_from_source::<M, _, _, _, _>(lease, source, M::lq_full_dense)
}

pub(super) fn full_qr_numerical_stage<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (mut q, mut r) = if rows <= cols {
        let mut q = vec![D::zero(); rows * rows];
        let mut r = vec![D::zero(); rows * cols];
        qr_into_workspace(
            dense, input, rows, cols, rows, &mut q, rows, rows, rows, &mut r, rows, cols, rows,
        )?;
        (q, r)
    } else {
        let (q, work_r) = augmented_identity_qr(dense, rows, cols, |block| {
            block.copy_from_slice(input);
        })?;
        (q, work_r[..rows * cols].to_vec())
    };
    positive_diagonal_gauge(&mut q, rows, &mut r, rows, cols);
    Ok((q, r))
}

/// The economy QR of the `m x (n + m)` matrix `[X | I_m]`, `X` the `m x n`
/// block `fill` writes (column-major): returns its `m x m` unitary `Q` and
/// its `m x (n + m)` upper-trapezoidal `R`. Because `R` is upper-trapezoidal,
/// the first `n` columns of `Q` span a superspace of `X`'s columns whatever
/// its rank, and the last `m - n` span their orthogonal complement.
///
/// This is the single owner of the full-`Q` cost the dense backend imposes:
/// it exposes only economy QR, so a full `Q` costs
/// `O(m^2 (n + m))` here against MatrixAlgebraKit's `geqrf` + `unmqr`/`ungqr`
/// `O(m n^2 + m n (m - n))`. Its consumers are the tall branch of
/// [`full_qr_numerical_stage`] (`qr_full`, `lq_full`), the full-SVD
/// [`orthonormal_completion`](super::svd::orthonormal_completion), and the
/// shape-based `left_null`/`right_null` kernel (`qr_null_basis`).
pub(super) fn augmented_identity_qr<E, D>(
    dense: &mut E,
    m: usize,
    n: usize,
    fill: impl FnOnce(&mut [D]),
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut augmented = vec![D::zero(); m * (n + m)];
    fill(&mut augmented[..m * n]);
    for row in 0..m {
        augmented[m * n + row * m + row] = D::one();
    }
    let mut q = vec![D::zero(); m * m];
    let mut r = vec![D::zero(); m * (n + m)];
    qr_into_workspace(
        dense,
        &augmented,
        m,
        n + m,
        m,
        &mut q,
        m,
        m,
        m,
        &mut r,
        m,
        n + m,
        m,
    )?;
    Ok((q, r))
}

pub(super) fn scale_col<D: FactorScalar>(
    data: &mut [D],
    rows: usize,
    leading: usize,
    col: usize,
    phase: D,
) {
    for row in 0..rows {
        let index = row + leading * col;
        data[index] = data[index] * phase;
    }
}

pub(super) fn scale_row<D: FactorScalar>(
    data: &mut [D],
    cols: usize,
    leading: usize,
    row: usize,
    phase: D,
) {
    for col in 0..cols {
        let index = row + leading * col;
        data[index] = data[index] * phase;
    }
}

#[cfg(test)]
/// Full QR `t = Q * R` (MatrixAlgebraKit `qr_full`): per sector `Q` is the
/// square `m x m` unitary and `R` the upper-trapezoidal `m x n`, obtained
/// from one economy QR, augmenting with identity columns only when `m > n`.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
pub(crate) fn qr_full<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Qr<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Qr { q, r } = qr_full_dyn(dense, &input.dynamic())?;
    Ok(Qr {
        q: typed_from_bound_factor(q)?,
        r: typed_from_bound_factor(r)?,
    })
}

/// Provider-bound dynamic-rank [`qr_full`].
pub(crate) fn qr_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let rows = matrix.rows;
        let cols = matrix.cols;
        let (q, r) = full_qr_numerical_stage(dense, matrix.data, rows, cols)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }
    let dimensions = MfAuthority(input.space()).coupled_dimensions(space.homspace().codomain())?;
    with_input_geometry!(&matrices, |geometry| Ok(Qr {
        q: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        r: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    }))
}

#[cfg(test)]
/// Full LQ `t = L * Q` (MatrixAlgebraKit `lq_full`): per sector `L` is the
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, via the full
/// QR of the adjoint sector matrices.
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
pub(crate) fn lq_full<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Lq<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Lq { l, q } = lq_full_dyn(dense, &input.dynamic())?;
    Ok(Lq {
        l: typed_from_bound_factor(l)?,
        q: typed_from_bound_factor(q)?,
    })
}

/// Provider-bound dynamic-rank [`lq_full`].
pub(crate) fn lq_full_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices =
        multiplicity_free_input_matricizations(space.structure(), input.data(), space.nout())?;
    let mut pairs = full_lq_pairs(dense, &matrices, FactorPlacement::Direct)?;
    let dimensions = MfAuthority(input.space()).coupled_dimensions(space.homspace().domain())?;
    with_input_geometry!(&matrices, |geometry| Ok(Lq {
        l: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        q: build_bound_factor(
            input.space(),
            space.homspace(),
            geometry,
            &mut pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    }))
}

#[cfg(test)]
/// Compact QR `t = Q * R` (MatrixAlgebraKit `qr_compact`):
/// `Q : codomain <- W` has orthonormal columns per coupled sector and
/// `R : W <- domain` with per-sector bond `min(rows, cols)`. The
/// positive-diagonal gauge is applied (MAK / TensorKit 0.17 default
/// `positive = true`): `R`'s diagonal is real non-negative per sector.
pub(crate) fn qr_compact<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Qr<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Qr { q, r } = qr_compact_dyn(dense, &input.dynamic())?;
    Ok(Qr {
        q: typed_from_bound_factor(q)?,
        r: typed_from_bound_factor(r)?,
    })
}

/// Provider-bound compact QR used by authority-preserving callers.
pub(crate) fn qr_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    // Why the checked entry has no direct-region plan: a plan needs both
    // provider-built factor spaces before any dense work, and the checked
    // contract reports dense errors before provider errors (#1960).
    if let Some(plan) = compact_factor_plan(input.space())? {
        return qr_compact_direct_regions(dense, input, &plan).map(|(q, r)| Qr { q, r });
    }
    let space = input.space().space();
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    crate::probe::input_pack(
        crate::probe::CopyOp::Qr,
        matricizations
            .iter()
            .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
    );
    let mut pairs = compact_qr_pairs(dense, matricizations.as_slice(), |_| {})?;
    #[cfg(test)]
    for pair in &pairs {
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Qr, 1, pair.left.len());
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Qr, 1, pair.right.len());
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
        .map(|(q, r)| Qr { q, r })
}

/// Compact QR of every coupled-sector matrix in one batched dense call, in the
/// positive-diagonal gauge, as `q` (left) and `r` (right) factor pairs.
/// `observe` sees each matrix before the dense call.
pub(super) fn compact_qr_pairs<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    mut observe: impl FnMut(&SectorMatrixRef<'_, D>),
) -> Result<Vec<FactorPair<D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut blocks = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        observe(&matrix);
        blocks.push((matrix.data, matrix.rows, matrix.cols));
    }
    let factors = compact_qr_owned_batch(dense, &blocks)?;
    let mut pairs = Vec::with_capacity(blocks.len());
    for (index, (mut q, mut r)) in factors.into_iter().enumerate() {
        let (_, rows, cols) = blocks[index];
        let rank = rows.min(cols);
        positive_diagonal_gauge_strided(&mut q, rows, rows, &mut r, rank, rank, cols);
        pairs.push(FactorPair {
            sector: matrices.get(index)?.sector,
            kept: rank,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rank,
        });
    }
    Ok(pairs)
}

/// The `rows x cols` matrix `X` whose LQ a sector computes, and `X^H`, the
/// matrix its QR reads: under [`FactorPlacement::Adjoint`] `matrix` is the
/// parent block `A` of the lazy adjoint `X = A^H`, read in place (#2070);
/// otherwise `matrix` is `X` and `X^H` is written into `scratch`.
fn lq_qr_input<'a, D: FactorScalar>(
    matrix: &SectorMatrixRef<'a, D>,
    placement: FactorPlacement,
    scratch: &'a mut Vec<D>,
) -> (usize, usize, &'a [D]) {
    match placement {
        FactorPlacement::Direct => {
            scratch.clear();
            extend_adjoint_col_major(scratch, matrix.data, matrix.rows, matrix.cols);
            (matrix.rows, matrix.cols, scratch)
        }
        FactorPlacement::Adjoint => (matrix.cols, matrix.rows, matrix.data),
    }
}

/// Compact LQ of every coupled-sector matrix through the adjoint's QR, in the
/// positive-diagonal gauge; under [`FactorPlacement::Adjoint`] the matrices
/// are the parent blocks of a lazy adjoint (see [`lq_qr_input`]). Every
/// direct sector's `X^H` lands in one scratch sized for the largest sector,
/// where MatrixAlgebraKit's `lq_via_qr!` allocates one per block. `observe` sees
/// each direct matrix with its adjoint before the dense call; `publish` sees
/// each sector's `X` shape (`rows x cols`), rank, and gauged `Q'`
/// (`cols x rank`) and `R'` (`rank x rows`), whose adjoints are `q` and `l`.
fn for_each_compact_lq<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + Sync + ?Sized),
    placement: FactorPlacement,
    mut observe: impl FnMut(&SectorMatrixRef<'_, D>, &[D]) + Send,
    mut publish: impl FnMut(SectorId, usize, usize, usize, &[D], &[D]) + Send,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut scratch_len = 0;
    if matches!(placement, FactorPlacement::Direct) {
        for index in 0..matrices.len() {
            scratch_len = scratch_len.max(matrices.get(index)?.data.len());
        }
    }
    let mut scratch = Vec::with_capacity(scratch_len);
    in_linalg_scope(dense, |dense| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let rank = matrix.rows.min(matrix.cols);
            let (rows, cols, qr_input) = lq_qr_input(&matrix, placement, &mut scratch);
            if matches!(placement, FactorPlacement::Direct) {
                observe(&matrix, qr_input);
            }
            let (mut q_prime, mut r_prime) = compact_qr_owned(dense, qr_input, cols, rows)?;
            positive_diagonal_gauge_strided(
                &mut q_prime,
                cols,
                cols,
                &mut r_prime,
                rank,
                rank,
                rows,
            );
            publish(matrix.sector, rows, cols, rank, &q_prime, &r_prime);
        }
        Ok(())
    })
}

/// [`for_each_compact_lq`] as `l` (left) and `q` (right) factor pairs owned
/// per sector, for the publishers that place each sector separately.
pub(super) fn compact_lq_pairs<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + Sync + ?Sized),
    placement: FactorPlacement,
    observe: impl FnMut(&SectorMatrixRef<'_, D>, &[D]) + Send,
) -> Result<Vec<FactorPair<D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut pairs = Vec::with_capacity(matrices.len());
    for_each_compact_lq(
        dense,
        matrices,
        placement,
        observe,
        |sector, rows, cols, rank, q_prime, r_prime| {
            pairs.push(FactorPair {
                sector,
                kept: rank,
                left: adjoint_col_major(r_prime, rank, rows),
                left_rows: rows,
                right: adjoint_col_major(q_prime, cols, rank),
                right_leading: rank,
            });
        },
    )?;
    Ok(pairs)
}

/// [`for_each_compact_lq`] with each final adjoint written once into one
/// buffer per side, in matrix order: the checked paired builder publishes
/// those buffers as they are when its output is canonical, as TensorKit's
/// `lq_compact!` writes each block of its preallocated `L`/`Q` in place.
pub(super) fn compact_lq_packed<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + Sync + ?Sized),
    placement: FactorPlacement,
    observe: impl FnMut(&SectorMatrixRef<'_, D>, &[D]) + Send,
) -> Result<PackedFactorPairs<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (mut left_len, mut right_len) = (0, 0);
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let rank = matrix.rows.min(matrix.cols);
        // `l` is `rows x rank` and `q` is `rank x cols` for `X`; an adjoint
        // parent block is `X^H`, whose extents are swapped below.
        left_len += rank * matrix.rows;
        right_len += rank * matrix.cols;
    }
    if matches!(placement, FactorPlacement::Adjoint) {
        std::mem::swap(&mut left_len, &mut right_len);
    }
    let mut packed = PackedFactorPairs {
        pairs: Vec::with_capacity(matrices.len()),
        left: Vec::with_capacity(left_len),
        right: Vec::with_capacity(right_len),
    };
    for_each_compact_lq(
        dense,
        matrices,
        placement,
        observe,
        |sector, rows, cols, rank, q_prime, r_prime| {
            let (left_start, right_start) = (packed.left.len(), packed.right.len());
            extend_adjoint_col_major(&mut packed.left, r_prime, rank, rows);
            extend_adjoint_col_major(&mut packed.right, q_prime, cols, rank);
            packed.pairs.push(PackedFactorPair {
                sector,
                kept: rank,
                left: left_start..packed.left.len(),
                left_rows: rows,
                right: right_start..packed.right.len(),
                right_leading: rank,
            });
        },
    )?;
    Ok(packed)
}

/// Full LQ of every coupled-sector matrix through the full QR of its adjoint,
/// as `l` (left) and square `q` (right) factor pairs; `placement` as for
/// [`compact_lq_pairs`].
fn full_lq_pairs<E, D>(
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    placement: FactorPlacement,
) -> Result<Vec<FactorPair<D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        let mut adjoint = Vec::new();
        let (rows, cols, qr_input) = lq_qr_input(&matrix, placement, &mut adjoint);
        let (q_prime, r_prime) = full_qr_numerical_stage(dense, qr_input, cols, rows)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: cols,
            left: adjoint_col_major(&r_prime, cols, rows),
            left_rows: rows,
            right: adjoint_col_major(&q_prime, cols, cols),
            right_leading: cols,
        });
    }
    Ok(pairs)
}

/// The parent blocks of a lazy adjoint `X = A^H`, listed in the region order
/// of the adjoint's own layout: region `i` of that layout is `X_c`, and
/// `get(i)` is the parent block `A_c = X_c^H` that LQ of `X_c` hands to QR.
pub(super) struct AdjointParentMatrices<'a, D> {
    data: &'a [D],
    regions: &'a [CoupledSectorRegion],
    by_adjoint: &'a [usize],
}

impl<D: FactorScalar> SectorMatrices<D> for AdjointParentMatrices<'_, D> {
    fn len(&self) -> usize {
        self.by_adjoint.len()
    }

    fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError> {
        let region = &self.regions[self.by_adjoint[index]];
        Ok(SectorMatrixRef {
            sector: region_sector(region),
            rows: region.rows(),
            cols: region.cols(),
            data: data_region(self.data, &region.range())?,
        })
    }
}

/// A parent's coupled-sector regions and, per adjoint region, the index of
/// the parent region it adjoints.
type ParentRegions = (Arc<[CoupledSectorRegion]>, Vec<usize>);

/// The parent's coupled-sector regions and, for each region of the adjoint
/// layout `adjoint`, the parent region it is the adjoint of: same sector,
/// rows the parent's column trees and columns its row trees, tree for tree.
/// `None` when the parent is not region-tiled or the trees are not so
/// transposed; the caller then publishes by tree identity instead.
fn adjoint_parent_regions(
    parent: &BlockStructure,
    parent_nout: usize,
    adjoint: &[CoupledSectorRegion],
) -> Result<Option<ParentRegions>, OperationError> {
    let Some(regions) = input_regions(parent, parent_nout)? else {
        return Ok(None);
    };
    if regions.len() != adjoint.len() {
        return Ok(None);
    }
    let mut by_adjoint = Vec::with_capacity(adjoint.len());
    {
        let index = SectorRegionIndex::new(&regions)?;
        for region in adjoint {
            let Some(parent_index) = index.get(region_sector(region)) else {
                return Ok(None);
            };
            let source = &regions[parent_index];
            if source.rows() != region.cols()
                || source.cols() != region.rows()
                || source.row_trees() != region.col_trees()
                || source.col_trees() != region.row_trees()
            {
                return Ok(None);
            }
            by_adjoint.push(parent_index);
        }
    }
    Ok(Some((regions, by_adjoint)))
}

/// LQ of the lazy adjoint `X = A^H` of the dense `parent = A` in fusion mode
/// `M`, read in place and published by tree identity on the swapped hom
/// space: the route for a parent or adjoint layout that is not
/// region-tiled.
///
/// Sector by sector, LQ of the materialized `X` is computed as the QR of
/// `X^H = A`; here that QR (the same dense kernel, input and positive-diagonal
/// gauge) reads `A` itself, and `X = R^H Q^H` publishes `l = R^H`,
/// `q = Q^H`. TensorKit (`src/factorizations/adjoint.jl`, `lq_compact!` /
/// `lq_full!` on an `AdjointTensorMap`) likewise factors the parent by QR and
/// adjoints the factors.
///
/// Why not a redirect to the facade QR of the parent: detaching `R^H` and
/// `Q^H` copies `k (m + n) >= m n` elements per sector; here they are written
/// once, as LQ of the materialized adjoint writes them.
fn lq_adjoint_by_tree<M, E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    full: bool,
) -> Result<Lq<BoundDynFactor<R, D>>, M::Error>
where
    M: FactorMode<R>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = parent.space().space();
    let matrices = generic_input_matricizations(space.structure(), parent.data(), space.nout())?;
    let mut pairs = if full {
        full_lq_pairs(dense, &matrices, FactorPlacement::Adjoint)?
    } else {
        compact_lq_pairs(dense, &matrices, FactorPlacement::Adjoint, |_, _| {})?
    };
    let authority = M::authority(parent.space());
    // The bond of full LQ is the domain of `A^H`, the codomain of `A`.
    let dimensions = if full {
        authority.coupled_dimensions(space.homspace().codomain())?
    } else {
        pairs.iter().map(|pair| (pair.sector, pair.kept)).collect()
    };
    let adjoint = FusionTreeHomSpace::new(
        space.homspace().domain().clone(),
        space.homspace().codomain().clone(),
    );
    with_input_geometry!(&matrices, |geometry| {
        let mut publish = |side| {
            publish_one_sided_factor(
                &authority,
                &adjoint,
                geometry,
                &mut pairs,
                &dimensions,
                side,
                FactorPlacement::Adjoint,
            )
        };
        Ok(Lq {
            l: publish(FactorSide::Left)?,
            q: publish(FactorSide::Right)?,
        })
    })
}

/// Multiplicity-free compact LQ of the lazy adjoint of `parent`, whose own
/// layout is `adjoint_space`: when both are region-tiled with transposed
/// trees, the direct-region plan of `adjoint_space` (the plan LQ of the
/// materialized adjoint runs) reads each QR input from the parent in place;
/// otherwise [`lq_adjoint_by_tree`].
pub(crate) fn lq_compact_adjoint_dyn<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let parent_space = parent.space().space();
    if let Some(plan) = compact_factor_plan(adjoint_space)? {
        if let Some((regions, by_adjoint)) = adjoint_parent_regions(
            parent_space.structure(),
            parent_space.nout(),
            &plan.source_regions,
        )? {
            let reader = LqRegionSource::Parent(AdjointParentMatrices {
                data: parent.data(),
                regions: &regions,
                by_adjoint: &by_adjoint,
            });
            return lq_compact_direct_regions(dense, adjoint_space, reader, &plan)
                .map(|(l, q)| Lq { l, q });
        }
    }
    lq_adjoint_by_tree::<MultiplicityFreeAdmissionMode, _, _, _>(dense, parent, false)
}

/// A lazy adjoint's own region tiling and the parent regions it adjoints.
struct AdjointLayout {
    regions: Arc<[CoupledSectorRegion]>,
    parent: ParentRegions,
}

/// The region tiling of `adjoint_space` and the parent blocks it adjoints,
/// when both layouts are region-tiled with transposed trees.
fn adjoint_layout_parent<R, D>(
    parent: &BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<AdjointLayout>, OperationError> {
    let space = adjoint_space.space();
    let Some(adjoint_regions) = input_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let parent_space = parent.space().space();
    Ok(adjoint_parent_regions(
        parent_space.structure(),
        parent_space.nout(),
        &adjoint_regions,
    )?
    .map(|parent| AdjointLayout {
        regions: adjoint_regions,
        parent,
    }))
}

/// Multiplicity-free full LQ of the lazy adjoint of `parent`: pairs from the
/// parent blocks in place, published on `adjoint_space` as LQ of the
/// materialized adjoint publishes them; otherwise [`lq_adjoint_by_tree`].
pub(crate) fn lq_full_adjoint_dyn<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Some(AdjointLayout {
        regions: geometry,
        parent: (regions, by_adjoint),
    }) = adjoint_layout_parent(parent, adjoint_space)?
    else {
        return lq_adjoint_by_tree::<MultiplicityFreeAdmissionMode, _, _, _>(dense, parent, true);
    };
    let matrices = AdjointParentMatrices {
        data: parent.data(),
        regions: &regions,
        by_adjoint: &by_adjoint,
    };
    let mut pairs = full_lq_pairs(dense, &matrices, FactorPlacement::Adjoint)?;
    let space = adjoint_space.space();
    let dimensions = MfAuthority(adjoint_space).coupled_dimensions(space.homspace().domain())?;
    let mut publish = |side| {
        build_bound_factor(
            adjoint_space,
            space.homspace(),
            geometry.as_ref(),
            &mut pairs,
            &dimensions,
            side,
        )
    };
    Ok(Lq {
        l: publish(FactorSide::Left)?,
        q: publish(FactorSide::Right)?,
    })
}

/// Checked-Generic compact LQ of the lazy adjoint of `parent`; see
/// [`lq_full_adjoint_dyn`].
pub(crate) fn lq_compact_adjoint_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Lq<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let Some(AdjointLayout {
        regions: geometry,
        parent: (regions, by_adjoint),
    }) = adjoint_layout_parent(parent, adjoint_space)?
    else {
        return lq_adjoint_by_tree::<CheckedGenericAdmissionMode, _, _, _>(dense, parent, false);
    };
    let matrices = AdjointParentMatrices {
        data: parent.data(),
        regions: &regions,
        by_adjoint: &by_adjoint,
    };
    let pairs = compact_lq_packed(dense, &matrices, FactorPlacement::Adjoint, |_, _| {})?;
    build_left_right_bound_pair_generic_checked(
        adjoint_space.provider_arc(),
        adjoint_space.space().homspace(),
        geometry.as_ref(),
        pairs,
    )
    .map(|(l, q)| Lq { l, q })
}

/// Checked-Generic full LQ of the lazy adjoint of `parent`; see
/// [`lq_full_adjoint_dyn`].
pub(crate) fn lq_full_adjoint_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Lq<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let Some(AdjointLayout {
        regions: geometry,
        parent: (regions, by_adjoint),
    }) = adjoint_layout_parent(parent, adjoint_space)?
    else {
        return lq_adjoint_by_tree::<CheckedGenericAdmissionMode, _, _, _>(dense, parent, true);
    };
    let matrices = AdjointParentMatrices {
        data: parent.data(),
        regions: &regions,
        by_adjoint: &by_adjoint,
    };
    let pairs = full_lq_pairs(dense, &matrices, FactorPlacement::Adjoint)?;
    let provider = adjoint_space.provider_arc();
    let homspace = adjoint_space.space().homspace();
    let dimensions = CheckedAuthority(provider).coupled_dimensions(homspace.domain())?;
    checked_full_factor_pair(provider, homspace, geometry.as_ref(), pairs, &dimensions)
        .map(|(l, q)| Lq { l, q })
}

/// Compact LQ of the lazy adjoint of the dense `parent`, read in place, in
/// fusion mode `M`; `adjoint_space` is the adjoint's own layout, on which LQ
/// of the materialized adjoint would publish (see [`lq_adjoint_by_tree`]).
#[doc(hidden)]
pub fn lq_compact_adjoint_from_parent<M, L, E, R, D>(
    lease: L,
    parent: BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Lq<BoundDynFactor<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    // The adjoint is finite exactly when its parent is.
    require_finite_factor_input(parent.data().iter().copied(), FactorFamily::Lq)?;
    lease.run(|dense| M::lq_compact_adjoint_dense(dense, &parent, adjoint_space))
}

/// Full LQ of the lazy adjoint of the dense `parent`, read in place, in
/// fusion mode `M` (see [`lq_compact_adjoint_from_parent`]).
#[doc(hidden)]
pub fn lq_full_adjoint_from_parent<M, L, E, R, D>(
    lease: L,
    parent: BoundDynamicTensorRef<'_, R, D>,
    adjoint_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<Lq<BoundDynFactor<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    require_finite_factor_input(parent.data().iter().copied(), FactorFamily::Lq)?;
    lease.run(|dense| M::lq_full_adjoint_dense(dense, &parent, adjoint_space))
}

pub(super) fn qr_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    plan: &CompactFactorPlan,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input.space().space();
    debug_assert_eq!(plan.source_layout, input.space().validated_layout());
    let left_space = input.space().rebind_validated(&plan.left_layout)?;
    let right_space = input.space().rebind_validated(&plan.right_layout)?;
    let mut left_regions = (0..plan.left_regions.len())
        .map(|_| None)
        .collect::<Vec<_>>();
    let mut right_regions = (0..plan.right_regions.len())
        .map(|_| None)
        .collect::<Vec<_>>();

    let routes = plan
        .routes
        .iter()
        .copied()
        .filter(|route| route.rank != 0)
        .collect::<Vec<_>>();
    let blocks = routes
        .iter()
        .map(|route| {
            let source = &plan.source_regions[route.source_region];
            (&input.data()[source.range()], source.rows(), source.cols())
        })
        .collect::<Vec<_>>();
    let factors = compact_qr_owned_batch(dense, &blocks)?;
    for (route, (mut left_data, mut right_data)) in routes.into_iter().zip(factors) {
        let source = &plan.source_regions[route.source_region];
        positive_diagonal_gauge_strided(
            &mut left_data,
            source.rows(),
            source.rows(),
            &mut right_data,
            route.rank,
            route.rank,
            source.cols(),
        );
        left_regions[route.left_region.expect("nonzero route has left region")] = Some(left_data);
        right_regions[route.right_region.expect("nonzero route has right region")] =
            Some(right_data);
    }

    let left_data = concat_compact_factor_regions(left_regions, plan.left_layout.required_len()?);
    let right_data =
        concat_compact_factor_regions(right_regions, plan.right_layout.required_len()?);

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

#[cfg(test)]
/// Compact LQ `t = L * Q` (MatrixAlgebraKit `lq_compact`, via the QR of the
/// transposed sector matrices): `Q : W <- domain` has orthonormal rows per
/// coupled sector and `L : codomain <- W`. The positive-diagonal gauge is
/// applied (MAK / TensorKit 0.17 default `positive = true`): `L`'s diagonal
/// is real non-negative per sector.
pub(crate) fn lq_compact<E, R, D>(
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Lq<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Lq { l, q } = lq_compact_dyn(dense, &input.dynamic())?;
    Ok(Lq {
        l: typed_from_bound_factor(l)?,
        q: typed_from_bound_factor(q)?,
    })
}

/// Provider-bound compact LQ used by authority-preserving callers.
pub(crate) fn lq_compact_dyn<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    if let Some(plan) = compact_factor_plan(input.space())? {
        return lq_compact_direct_regions(
            dense,
            input.space(),
            LqRegionSource::Input(input.data()),
            &plan,
        )
        .map(|(l, q)| Lq { l, q });
    }
    let space = input.space().space();
    let matricizations = sector_matricizations(space.structure(), input.data(), space.nout())?;
    #[cfg(test)]
    crate::probe::input_pack(
        crate::probe::CopyOp::Lq,
        matricizations
            .iter()
            .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
    );
    let mut pairs = compact_lq_pairs(
        dense,
        matricizations.as_slice(),
        FactorPlacement::Direct,
        |_, _| {},
    )?;
    #[cfg(test)]
    for pair in &pairs {
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Lq, 1, pair.left.len());
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Lq, 1, pair.right.len());
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
        .map(|(l, q)| Lq { l, q })
}

/// What the QR of each compact-LQ route reads.
pub(super) enum LqRegionSource<'a, D> {
    /// The input payload: the QR reads the adjoint of each source region.
    Input(&'a [D]),
    /// A lazy adjoint's parent blocks, already the QR input (#2070).
    Parent(AdjointParentMatrices<'a, D>),
}

/// Compact LQ of `input_space` over the direct-region `plan`, reading `reader`.
pub(super) fn lq_compact_direct_regions<E, R, D>(
    dense: &mut E,
    input_space: &BoundDynamicFusionMapSpace<R>,
    reader: LqRegionSource<'_, D>,
    plan: &CompactFactorPlan,
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: FusionRule,
    D: FactorScalar,
{
    let space = input_space.space();
    debug_assert_eq!(plan.source_layout, input_space.validated_layout());
    let left_space = input_space.rebind_validated(&plan.left_layout)?;
    let right_space = input_space.rebind_validated(&plan.right_layout)?;
    let left_len = plan.left_layout.required_len()?;
    let right_len = plan.right_layout.required_len()?;
    // Every output element is written once by the final adjoints. When the
    // nonzero routes visit both factor regions in storage order, append them
    // to empty buffers instead of zero-filling first; otherwise overwrite a
    // zeroed buffer region by region.
    let append = lq_routes_append_in_storage_order(plan, left_len, right_len);
    #[cfg(test)]
    let append = append && !crate::probe::lq_zeroed_publication_forced();
    let (mut left_data, mut right_data) = if append {
        (Vec::with_capacity(left_len), Vec::with_capacity(right_len))
    } else {
        (vec![D::zero(); left_len], vec![D::zero(); right_len])
    };

    let max_adjoint_len = match reader {
        LqRegionSource::Input(_) => plan
            .routes
            .iter()
            .map(|route| plan.source_regions[route.source_region].range().len())
            .max()
            .unwrap_or(0),
        LqRegionSource::Parent(_) => 0,
    };
    let mut adjoint_scratch = Vec::with_capacity(max_adjoint_len);
    #[cfg(test)]
    crate::probe::lq_scratch::<D>(max_adjoint_len);

    in_linalg_scope(dense, |dense| {
        for route in plan.routes.iter().copied() {
            if route.rank == 0 {
                continue;
            }
            let source = &plan.source_regions[route.source_region];
            let left =
                &plan.left_regions[route.left_region.expect("nonzero route has left region")];
            let right =
                &plan.right_regions[route.right_region.expect("nonzero route has right region")];
            let qr_input = match &reader {
                LqRegionSource::Input(data) => {
                    let source_data = &data[source.range()];
                    adjoint_scratch.clear();
                    extend_adjoint_col_major(
                        &mut adjoint_scratch,
                        source_data,
                        source.rows(),
                        source.cols(),
                    );
                    #[cfg(test)]
                    crate::probe::lq_adjoint_fill::<D>(source_data.len());
                    adjoint_scratch.as_slice()
                }
                LqRegionSource::Parent(parent) => parent.get(route.source_region)?.data,
            };

            let (mut q_prime, mut r_prime) =
                compact_qr_owned(dense, qr_input, source.cols(), source.rows())?;
            positive_diagonal_gauge_strided(
                &mut q_prime,
                source.cols(),
                source.cols(),
                &mut r_prime,
                route.rank,
                route.rank,
                source.rows(),
            );
            if append {
                extend_adjoint_col_major(&mut left_data, &r_prime, route.rank, source.rows());
                extend_adjoint_col_major(&mut right_data, &q_prime, source.cols(), route.rank);
            } else {
                adjoint_col_major_into(
                    &r_prime,
                    route.rank,
                    source.rows(),
                    &mut left_data[left.range()],
                );
                adjoint_col_major_into(
                    &q_prime,
                    source.cols(),
                    route.rank,
                    &mut right_data[right.range()],
                );
            }
            #[cfg(test)]
            crate::probe::lq_final_adjoint_copy::<D>(r_prime.len());
            #[cfg(test)]
            crate::probe::lq_final_adjoint_copy::<D>(q_prime.len());
        }
        Ok(())
    })?;
    #[cfg(test)]
    crate::probe::lq_output_prefill::<D>(if append { 0 } else { left_len + right_len });

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

/// [`lq_routes_append_in_storage_order`] after replacing the plan's routes.
#[cfg(test)]
pub(crate) fn lq_routes_append_with_routes_for_test(
    plan: &mut CompactFactorPlan,
    routes: Vec<CompactFactorRoute>,
) -> Result<bool, OperationError> {
    plan.routes = routes;
    Ok(lq_routes_append_in_storage_order(
        plan,
        plan.left_layout.required_len()?,
        plan.right_layout.required_len()?,
    ))
}

/// Whether the nonzero routes of `plan` reach the left and right factor
/// regions contiguously from offset zero, in route order, and cover
/// `left_len`/`right_len` exactly, so appending each route's factors in turn
/// builds both outputs. Rank-zero routes own empty regions.
pub(super) fn lq_routes_append_in_storage_order(
    plan: &CompactFactorPlan,
    left_len: usize,
    right_len: usize,
) -> bool {
    let (mut left_end, mut right_end) = (0usize, 0usize);
    for route in &plan.routes {
        let (Some(left), Some(right)) = (route.left_region, route.right_region) else {
            continue;
        };
        let (left, right) = (
            plan.left_regions[left].range(),
            plan.right_regions[right].range(),
        );
        if left.start != left_end || right.start != right_end {
            return false;
        }
        (left_end, right_end) = (left.end, right.end);
    }
    left_end == left_len && right_end == right_len
}

/// Appends the adjoint of the column-major `rows x cols` matrix `data` to
/// `output` as a column-major `cols x rows` matrix.
pub(super) fn extend_adjoint_col_major<D: FactorScalar>(
    output: &mut Vec<D>,
    data: &[D],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(data.len(), rows * cols);
    output.reserve(data.len());
    for row in 0..rows {
        // The StepBy/Take chain retained an out-of-line fold per row in Release.
        output.extend((0..cols).map(|col| FactorScalar::adjoint(data[row + col * rows])));
    }
}

#[test]
fn append_adjoint_preserves_rectangular_order_conjugation_and_existing_values() {
    macro_rules! check {
        ($ty:ty) => {{
            let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(<$ty as FactorScalar>::from_real);
            let mut output = vec![<$ty as FactorScalar>::from_real(99.0)];
            extend_adjoint_col_major(&mut output, &input, 2, 3);
            assert_eq!(
                output,
                [99.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0].map(<$ty as FactorScalar>::from_real)
            );
            for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
                let before = output.clone();
                extend_adjoint_col_major(&mut output, &[], rows, cols);
                assert_eq!(output, before);
            }
        }};
    }
    check!(f32);
    check!(f64);
    check!(num_complex::Complex32);
    check!(Complex64);
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
        Complex64::new(9.0, 10.0),
        Complex64::new(11.0, 12.0),
    ];
    let mut output = vec![Complex64::new(99.0, 1.0)];
    extend_adjoint_col_major(&mut output, &input, 2, 3);
    assert_eq!(
        output,
        [
            Complex64::new(99.0, 1.0),
            Complex64::new(1.0, -2.0),
            Complex64::new(5.0, -6.0),
            Complex64::new(9.0, -10.0),
            Complex64::new(3.0, -4.0),
            Complex64::new(7.0, -8.0),
            Complex64::new(11.0, -12.0)
        ]
    );
}

/// Transposes a column-major `rows x cols` matrix into column-major
/// `cols x rows`.
/// Adjoint (conjugate transpose) of a column-major `rows x cols` matrix.
pub(super) fn adjoint_col_major<D: FactorScalar>(data: &[D], rows: usize, cols: usize) -> Vec<D> {
    let mut adjoint = vec![D::zero(); data.len()];
    adjoint_col_major_into(data, rows, cols, &mut adjoint);
    adjoint
}

pub(super) fn adjoint_col_major_into<D: FactorScalar>(
    data: &[D],
    rows: usize,
    cols: usize,
    adjoint: &mut [D],
) {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(adjoint.len(), data.len());
    for col in 0..cols {
        for row in 0..rows {
            adjoint[col + cols * row] = FactorScalar::adjoint(data[row + rows * col]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qr_into_workspace<E, D>(
    dense: &mut E,
    input: &[D],
    input_rows: usize,
    input_cols: usize,
    input_leading: usize,
    q: &mut [D],
    q_rows: usize,
    q_cols: usize,
    q_leading: usize,
    r: &mut [D],
    r_rows: usize,
    r_cols: usize,
    r_leading: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let input_shape = [input_rows, input_cols];
    let input_strides = [1usize, input_leading];
    let q_shape = [q_rows, q_cols];
    let q_strides = [1usize, q_leading];
    let r_shape = [r_rows, r_cols];
    let r_strides = [1usize, r_leading];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let q_view = DenseViewMut::new(q, &q_shape, &q_strides, 0).map_err(OperationError::Dense)?;
    let r_view = DenseViewMut::new(r, &r_shape, &r_strides, 0).map_err(OperationError::Dense)?;
    dense
        .qr_into(
            D::dense_read(input_view),
            D::dense_write(q_view),
            D::dense_write(r_view),
        )
        .map_err(OperationError::Dense)
}

/// Compact QR owns both dense outputs, so it can transfer the executor's host
/// buffers directly. Full QR keeps its caller-owned workspace contract above.
pub(super) fn compact_qr_owned<E, D>(
    dense: &mut E,
    input: &[D],
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let input_shape = [rows, cols];
    let input_strides = [1usize, rows];
    let input_view =
        DenseView::new(input, &input_shape, &input_strides, 0).map_err(OperationError::Dense)?;
    let outputs = dense
        .qr(D::dense_read(input_view))
        .map_err(OperationError::Dense)?;
    compact_qr_outputs(outputs, rows, cols)
}

/// Compact QR of each column-major `(data, rows, cols)` block, submitted as one
/// executor batch so the backend admits the whole coupled-sector loop once.
#[expect(
    clippy::type_complexity,
    reason = "the dense QR ownership boundary returns one documented Q, R pair per block"
)]
pub(super) fn compact_qr_owned_batch<E, D>(
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
) -> Result<Vec<(Vec<D>, Vec<D>)>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factorize_col_major_batch(dense, DenseFactorization::Qr, blocks)?
        .into_iter()
        .zip(blocks)
        .map(|(outputs, &(_, rows, cols))| compact_qr_outputs(outputs, rows, cols))
        .collect()
}

pub(super) fn compact_qr_outputs<D: FactorScalar>(
    mut outputs: Vec<DenseTensor>,
    rows: usize,
    cols: usize,
) -> Result<(Vec<D>, Vec<D>), OperationError> {
    let rank = rows.min(cols);
    if outputs.len() != 2 {
        return Err(OperationError::Dense(arity_mismatch(
            "qr_into",
            2,
            outputs.len(),
        )));
    }
    let q = compact_factor_output_owned::<D>(outputs.remove(0), &[rows, rank], "qr_into")?;
    let r = compact_factor_output_owned::<D>(outputs.remove(0), &[rank, cols], "qr_into")?;
    Ok((q, r))
}

/// Checked-Generic compact QR. Provider-bound output spaces are admitted
/// through the checked staging boundary; dense QR itself performs no provider
/// queries and therefore needs no Tenferro-specific capability.
#[doc(hidden)]
pub(crate) fn qr_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
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
            crate::probe::CopyOp::Qr,
            matrices
                .iter()
                .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
        );
    }
    #[cfg(test)]
    let data = input.data();
    let pairs = compact_qr_pairs(dense, &matrices, |_matrix| {
        #[cfg(test)]
        crate::probe::checked_compact_input(
            crate::probe::CheckedCompactOperation::Qr,
            data,
            _matrix.data,
            None,
        );
    })?;
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
        .map(|(q, r)| Qr { q, r })
}

/// Checked-Generic compact LQ, implemented through the existing host
/// adjoint-plus-QR boundary; no borrowed conjugated-dot capability is needed.
#[doc(hidden)]
pub(crate) fn lq_compact_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
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
            crate::probe::CopyOp::Lq,
            matrices
                .iter()
                .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
        );
    }
    #[cfg(test)]
    let data = input.data();
    let pairs = compact_lq_packed(
        dense,
        &matrices,
        FactorPlacement::Direct,
        |_matrix, _adjoint| {
            #[cfg(test)]
            {
                crate::probe::lq_adjoint_fill::<D>(_matrix.data.len());
                crate::probe::checked_compact_input(
                    crate::probe::CheckedCompactOperation::Lq,
                    data,
                    _matrix.data,
                    Some(_adjoint),
                );
            }
        },
    )?;
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
        .map(|(l, q)| Lq { l, q })
}

/// Checked-Generic full QR, augmenting only sectors that require completion.
#[doc(hidden)]
pub(crate) fn qr_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Qr<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let mut pairs = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices
            .get(index)
            .map_err(CheckedGenericFactorPlanError::from)?;
        let rows = matrix.rows;
        let cols = matrix.cols;
        let (q, r) = full_qr_numerical_stage(dense, matrix.data, rows, cols)
            .map_err(CheckedGenericFactorPlanError::from)?;
        pairs.push(FactorPair {
            sector: matrix.sector,
            kept: rows,
            left: q,
            left_rows: rows,
            right: r,
            right_leading: rows,
        });
    }
    let dimensions = CheckedAuthority(provider).coupled_dimensions(space.homspace().codomain())?;
    with_input_geometry!(&matrices, |geometry| checked_full_factor_pair(
        provider,
        space.homspace(),
        geometry,
        pairs,
        &dimensions
    ))
    .map(|(q, r)| Qr { q, r })
}

/// Checked-Generic full LQ via the full QR of each sector's adjoint matrix.
#[doc(hidden)]
pub(crate) fn lq_full_dyn_checked_generic<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<Lq<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let provider = input.space().provider_arc();
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())
        .map_err(CheckedGenericFactorPlanError::from)?;
    let pairs = full_lq_pairs(dense, &matrices, FactorPlacement::Direct)
        .map_err(CheckedGenericFactorPlanError::from)?;
    let dimensions = CheckedAuthority(provider).coupled_dimensions(space.homspace().domain())?;
    with_input_geometry!(&matrices, |geometry| checked_full_factor_pair(
        provider,
        space.homspace(),
        geometry,
        pairs,
        &dimensions
    ))
    .map(|(l, q)| Lq { l, q })
}
