use super::*;
use symtn_dense::Engine;

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
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Qr<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    qr_from_source::<M, _, _, _, _>(lease, source, FactorFamily::Qr, |dense, input| {
        M::qr_compact_dense(engine, dense, input)
    })
}

/// Full QR of `source` in fusion mode `M`.
#[doc(hidden)]
pub fn qr_full_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Qr<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    qr_from_source::<M, _, _, _, _>(lease, source, FactorFamily::Qr, |dense, input| {
        M::qr_full_dense(engine, dense, input)
    })
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
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Lq<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    lq_from_source::<M, _, _, _, _>(lease, source, |dense, input| {
        M::lq_compact_dense(engine, dense, input)
    })
}

/// Full LQ of `source` in fusion mode `M`.
#[doc(hidden)]
pub fn lq_full_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<Lq<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    lq_from_source::<M, _, _, _, _>(lease, source, |dense, input| {
        M::lq_full_dense(engine, dense, input)
    })
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
/// square `m x m` unitary and `R` the upper-trapezoidal `m x n`, from one
/// [`DenseFactorization::QrFull`] batch (see [`full_qr_packed`]).
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
pub(crate) fn qr_full<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Qr<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Qr { q, r } = qr_full_dyn(engine, dense, &input.dynamic())?;
    Ok(Qr {
        q: typed_from_bound_factor(q)?,
        r: typed_from_bound_factor(r)?,
    })
}

/// Provider-bound dynamic-rank [`qr_full`].
pub(crate) fn qr_full_dyn<E, R, D>(
    engine: &mut dyn Engine,
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
    let mut pairs = full_qr_packed(engine, dense, &matrices)?;
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
/// lower-trapezoidal `m x n` and `Q` the square `n x n` unitary, from one
/// [`DenseFactorization::QrFull`] batch on the sector matrices' transposes,
/// read in place (see [`full_lq_packed`]).
/// The positive-diagonal gauge is applied (MAK / TensorKit 0.17 default).
pub(crate) fn lq_full<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Lq<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Lq { l, q } = lq_full_dyn(engine, dense, &input.dynamic())?;
    Ok(Lq {
        l: typed_from_bound_factor(l)?,
        q: typed_from_bound_factor(q)?,
    })
}

/// Provider-bound dynamic-rank [`lq_full`].
pub(crate) fn lq_full_dyn<E, R, D>(
    engine: &mut dyn Engine,
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
    let mut pairs = full_lq_packed(engine, dense, &matrices, FactorPlacement::Direct)?;
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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Qr<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Qr { q, r } = qr_compact_dyn(engine, dense, &input.dynamic())?;
    Ok(Qr {
        q: typed_from_bound_factor(q)?,
        r: typed_from_bound_factor(r)?,
    })
}

/// Provider-bound compact QR used by authority-preserving callers.
pub(crate) fn qr_compact_dyn<E, R, D>(
    engine: &mut dyn Engine,
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
        return qr_compact_direct_regions(engine, dense, input, &plan).map(|(q, r)| Qr { q, r });
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
    let mut pairs = compact_qr_pairs(engine, dense, matricizations.as_slice(), |_| {})?;
    #[cfg(test)]
    for pair in &pairs.pairs {
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Qr, 1, pair.left.len());
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Qr, 1, pair.right.len());
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
        .map(|(q, r)| Qr { q, r })
}

/// Compact QR of every coupled-sector matrix in one batched dense call, in the
/// positive-diagonal gauge, as `q` (left) and `r` (right) factors written
/// once into one buffer per side in matrix order, laid out from the
/// matricization shapes before the dense call (the checked builder then
/// publishes them as they are when its output is canonical). `observe` sees
/// each matrix before the dense call.
pub(super) fn compact_qr_pairs<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    mut observe: impl FnMut(&SectorMatrixRef<'_, D>),
) -> Result<PackedFactorPairs<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut blocks = Vec::with_capacity(matrices.len());
    let mut sectors = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        observe(&matrix);
        sectors.push(matrix.sector);
        blocks.push((matrix.data, matrix.rows, matrix.cols));
    }
    let rank = |&(_, rows, cols): &(&[D], usize, usize)| rows.min(cols);
    let left_ranges = packed_ranges(blocks.iter().map(|block| block.1 * rank(block)));
    let right_ranges = packed_ranges(blocks.iter().map(|block| rank(block) * block.2));
    let mut left = vec![D::zero(); left_ranges.last().map_or(0, |range| range.end)];
    let mut right = vec![D::zero(); right_ranges.last().map_or(0, |range| range.end)];
    compact_qr_into_batch(
        engine,
        dense,
        &blocks,
        split_ranges(&mut left, &left_ranges)?,
        split_ranges(&mut right, &right_ranges)?,
    )?;
    let pairs = blocks
        .iter()
        .zip(sectors)
        .zip(left_ranges.into_iter().zip(right_ranges))
        .map(|((block, sector), (left, right))| PackedFactorPair {
            sector,
            kept: rank(block),
            left,
            left_rows: block.1,
            right,
            right_leading: rank(block),
        })
        .collect();
    Ok(PackedFactorPairs { pairs, left, right })
}

/// Full QR of every coupled-sector matrix in one solver-first executor
/// batch ([`DenseFactorization::QrFull`]), in the positive-diagonal gauge,
/// as square `q` (left) and `r` (right) factors written once into one
/// buffer per side in matrix order, laid out from the matrix shapes before
/// the dense call, as TensorKit's `qr_full!` writes each block of its
/// preallocated `Q`/`R` in place. `Q`'s columns past `min(rows, cols)` are
/// any orthonormal basis of the complement, which may differ by executor.
fn full_qr_packed<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<PackedFactorPairs<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (sectors, blocks) = sector_blocks(matrices)?;
    let left_ranges = packed_ranges(blocks.iter().map(|&(_, rows, _)| rows * rows));
    let right_ranges = packed_ranges(blocks.iter().map(|&(_, rows, cols)| rows * cols));
    let mut left = vec![D::zero(); left_ranges.last().map_or(0, |range| range.end)];
    let mut right = vec![D::zero(); right_ranges.last().map_or(0, |range| range.end)];
    let mut read_layouts = Vec::new();
    let inputs = col_major_reads(&blocks, &mut read_layouts)?;
    let layouts = blocks
        .iter()
        .map(|&(_, rows, cols)| {
            [
                matrix_layout(rows, rows),
                matrix_layout(rows, cols),
                vector_layout(0),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    let mut dests = split_ranges(&mut left, &left_ranges)?
        .into_iter()
        .zip(split_ranges(&mut right, &right_ranges)?)
        .zip(&layouts)
        .map(|((q, r), layout)| {
            Ok(FactorDest::new(
                [
                    Some(dest_write(q, &layout[0])?),
                    Some(dest_write(r, &layout[1])?),
                    None,
                ],
                symtn_dense::FactorPlacement::AsComputed,
            ))
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    factorize_into_solver_first(
        engine,
        dense,
        DenseFactorization::QrFull,
        &inputs,
        &mut dests,
        |index, dest| {
            let (_, rows, cols) = blocks[index];
            let [q, r, _] = dest.outputs_mut();
            positive_diagonal_gauge(
                dest_data::<D>(q, "qr_into")?,
                rows,
                dest_data::<D>(r, "qr_into")?,
                rows,
                cols,
            );
            Ok(())
        },
    )?;
    let pairs = blocks
        .iter()
        .zip(sectors)
        .zip(left_ranges.into_iter().zip(right_ranges))
        .map(
            |((&(_, rows, _), sector), (left, right))| PackedFactorPair {
                sector,
                kept: rows,
                left,
                left_rows: rows,
                right,
                right_leading: rows,
            },
        )
        .collect();
    Ok(PackedFactorPairs { pairs, left, right })
}

/// Compact LQ `X = L Q` of each column-major `(data, rows, cols)` block into
/// its destination slices `l` (`X` rows `x k`) and `q` (`k x` `X` cols),
/// `k = min(rows, cols)`, as one solver-first executor batch inside one
/// linear-algebra scope; each solved block is positive-diagonal gauged in
/// place on its final placement.
/// Under [`FactorPlacement::Direct`] a block is `X`, factored by
/// [`DenseFactorization::Lq`]. Under [`FactorPlacement::Adjoint`] it is the
/// parent block `A` of a lazy adjoint `X = A^H`, read in place (#2070) and
/// factored by [`DenseFactorization::Qr`] placed as the LQ of `A^H`
/// (`A = Q' R'`, `l = R'^H`, `q = Q'^H`). TensorKit
/// (`src/factorizations/adjoint.jl`) likewise factors the parent of an
/// adjoint by QR and adjoints the factors.
fn compact_lq_into_batch<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
    placement: FactorPlacement,
    l: Vec<&mut [D]>,
    q: Vec<&mut [D]>,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (op, dense_placement) = match placement {
        FactorPlacement::Direct => (
            DenseFactorization::Lq,
            symtn_dense::FactorPlacement::AsComputed,
        ),
        FactorPlacement::Adjoint => (
            DenseFactorization::Qr,
            symtn_dense::FactorPlacement::Adjoint,
        ),
    };
    // The `rows x cols` extents of `X`.
    let lq_dims = |&(_, rows, cols): &(&[D], usize, usize)| match placement {
        FactorPlacement::Direct => (rows, cols),
        FactorPlacement::Adjoint => (cols, rows),
    };
    let mut read_layouts = Vec::new();
    let inputs = col_major_reads(blocks, &mut read_layouts)?;
    let layouts = blocks
        .iter()
        .map(|block| {
            let (rows, cols) = lq_dims(block);
            let rank = rows.min(cols);
            [
                matrix_layout(rows, rank),
                matrix_layout(rank, cols),
                vector_layout(0),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    let mut dests = l
        .into_iter()
        .zip(q)
        .zip(&layouts)
        .map(|((l, q), layout)| {
            Ok(FactorDest::new(
                [
                    Some(dest_write(l, &layout[0])?),
                    Some(dest_write(q, &layout[1])?),
                    None,
                ],
                dense_placement,
            ))
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    // One executor scope around the batch, as compact LQ has always run:
    // the scope's own admission and its refusal take precedence over the
    // batch's outcome.
    in_linalg_scope(engine, dense, |dense, engine| {
        factorize_into_solver_first(engine, dense, op, &inputs, &mut dests, |index, dest| {
            let (rows, cols) = lq_dims(&blocks[index]);
            let [l, q, _] = dest.outputs_mut();
            positive_diagonal_gauge_lq(
                dest_data::<D>(l, "lq")?,
                rows,
                dest_data::<D>(q, "lq")?,
                rows.min(cols),
                cols,
            );
            Ok(())
        })
    })
}

/// [`positive_diagonal_gauge`] of an LQ pair: absorbs `D = diag(phase(L_jj))`
/// as `L -> L D^H`, `Q -> D Q`, leaving `L Q` unchanged with real
/// non-negative `L_jj`; zero diagonal entries keep phase `1`. It is the QR
/// gauge of `X^H = Q^H L^H`, adjointed. `l` is column-major `rows x rank`,
/// `q` column-major `rank x cols`.
fn positive_diagonal_gauge_lq<D: FactorScalar>(
    l: &mut [D],
    rows: usize,
    q: &mut [D],
    rank: usize,
    cols: usize,
) {
    for j in 0..rank {
        let z = l[j + rows * j].widen_complex();
        let norm = z.norm();
        if norm == 0.0 {
            continue;
        }
        let phase = D::from_complex64(z / norm);
        scale_col(l, rows, rows, j, FactorScalar::adjoint(phase));
        scale_row(q, cols, rank, j, phase);
    }
}

/// Compact LQ of every coupled-sector matrix in one batched dense call (see
/// [`compact_lq_into_batch`]), as `l` (left) and `q` (right) factors written
/// once into one buffer per side in matrix order, laid out from the matrix
/// shapes before the dense call: the checked paired builder publishes those
/// buffers as they are when its output is canonical, as TensorKit's
/// `lq_compact!` writes each block of its preallocated `L`/`Q` in place.
/// Under [`FactorPlacement::Adjoint`] the matrices are the parent blocks of
/// a lazy adjoint, and each pair has the adjoint's (`X`'s) geometry.
/// `observe` sees each matrix before the dense call.
pub(super) fn compact_lq_packed<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    placement: FactorPlacement,
    mut observe: impl FnMut(&SectorMatrixRef<'_, D>),
) -> Result<PackedFactorPairs<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut blocks = Vec::with_capacity(matrices.len());
    let mut sectors = Vec::with_capacity(matrices.len());
    for index in 0..matrices.len() {
        let matrix = matrices.get(index)?;
        observe(&matrix);
        sectors.push(matrix.sector);
        blocks.push((matrix.data, matrix.rows, matrix.cols));
    }
    // `X`'s `(rows, cols)`: an adjoint parent block is `X^H`.
    let lq_dims = |&(_, rows, cols): &(&[D], usize, usize)| match placement {
        FactorPlacement::Direct => (rows, cols),
        FactorPlacement::Adjoint => (cols, rows),
    };
    let left_ranges = packed_ranges(blocks.iter().map(|block| {
        let (rows, cols) = lq_dims(block);
        rows * rows.min(cols)
    }));
    let right_ranges = packed_ranges(blocks.iter().map(|block| {
        let (rows, cols) = lq_dims(block);
        rows.min(cols) * cols
    }));
    let mut left = vec![D::zero(); left_ranges.last().map_or(0, |range| range.end)];
    let mut right = vec![D::zero(); right_ranges.last().map_or(0, |range| range.end)];
    compact_lq_into_batch(
        engine,
        dense,
        &blocks,
        placement,
        split_ranges(&mut left, &left_ranges)?,
        split_ranges(&mut right, &right_ranges)?,
    )?;
    let pairs = blocks
        .iter()
        .zip(sectors)
        .zip(left_ranges.into_iter().zip(right_ranges))
        .map(|((block, sector), (left, right))| {
            let (rows, cols) = lq_dims(block);
            PackedFactorPair {
                sector,
                kept: rows.min(cols),
                left,
                left_rows: rows,
                right,
                right_leading: rows.min(cols),
            }
        })
        .collect();
    Ok(PackedFactorPairs { pairs, left, right })
}

/// Full LQ `X = L Q` of every coupled-sector matrix in one solver-first
/// executor batch ([`DenseFactorization::QrFull`]), in the positive-diagonal
/// gauge, as `l` (left) and square `q` (right) factors written into one
/// buffer per side in matrix order; `placement` as for
/// [`compact_lq_packed`].
///
/// Under [`FactorPlacement::Direct`] the batch reads each block's transpose
/// in place (a strided view): `X^T = Q' R'` gives `X = R'^T Q'^T`, so
/// `L = R'^T` and `Q = Q'^T` (`Q'^T conj(Q') = conj(Q'^H Q') = I`). Under
/// [`FactorPlacement::Adjoint`] it reads the parent block `A = X^H`:
/// `A = Q' R'` gives `L = R'^H`, `Q = Q'^H`. Either way `L`'s diagonal is
/// `R'`'s, real non-negative once `(Q', R')` is gauged. `Q'` is solved into
/// its final slice and transposed there in place; `R'` is solved into one
/// packed scratch the size of `L` and written to `L` once.
///
/// Why the transpose, not a staged `X^H`: a strided read can swap axes but
/// not conjugate, and the QR of `X^T` factors `X` itself.
fn full_lq_packed<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    placement: FactorPlacement,
) -> Result<PackedFactorPairs<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (sectors, blocks) = sector_blocks(matrices)?;
    // `X`'s `(rows, cols)`; the QR input is `cols x rows`.
    let lq_dims = |&(_, rows, cols): &(&[D], usize, usize)| match placement {
        FactorPlacement::Direct => (rows, cols),
        FactorPlacement::Adjoint => (cols, rows),
    };
    let left_ranges = packed_ranges(blocks.iter().map(|block| {
        let (rows, cols) = lq_dims(block);
        rows * cols
    }));
    let right_ranges = packed_ranges(blocks.iter().map(|block| {
        let (_, cols) = lq_dims(block);
        cols * cols
    }));
    let mut left = vec![D::zero(); left_ranges.last().map_or(0, |range| range.end)];
    let mut right = vec![D::zero(); right_ranges.last().map_or(0, |range| range.end)];
    let mut r_prime = vec![D::zero(); left.len()];
    let mut read_layouts = Vec::new();
    let transposed = matches!(placement, FactorPlacement::Direct);
    let inputs = block_reads(&blocks, &mut read_layouts, transposed)?;
    let layouts = blocks
        .iter()
        .map(|block| {
            let (rows, cols) = lq_dims(block);
            [
                matrix_layout(cols, cols),
                matrix_layout(cols, rows),
                vector_layout(0),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    let mut dests = split_ranges(&mut right, &right_ranges)?
        .into_iter()
        .zip(split_ranges(&mut r_prime, &left_ranges)?)
        .zip(&layouts)
        .map(|((q, r), layout)| {
            Ok(FactorDest::new(
                [
                    Some(dest_write(q, &layout[0])?),
                    Some(dest_write(r, &layout[1])?),
                    None,
                ],
                symtn_dense::FactorPlacement::AsComputed,
            ))
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    factorize_into_solver_first(
        engine,
        dense,
        DenseFactorization::QrFull,
        &inputs,
        &mut dests,
        |index, dest| {
            let (rows, cols) = lq_dims(&blocks[index]);
            let [q, r, _] = dest.outputs_mut();
            positive_diagonal_gauge(
                dest_data::<D>(q, "qr_into")?,
                cols,
                dest_data::<D>(r, "qr_into")?,
                cols,
                rows,
            );
            Ok(())
        },
    )?;
    let map: fn(D) -> D = match placement {
        FactorPlacement::Direct => |value| value,
        FactorPlacement::Adjoint => FactorScalar::adjoint,
    };
    let pairs = blocks
        .iter()
        .zip(sectors)
        .zip(left_ranges.into_iter().zip(right_ranges))
        .map(|((block, sector), (l, q))| {
            let (rows, cols) = lq_dims(block);
            transpose_col_major_into(&r_prime[l.clone()], cols, rows, &mut left[l.clone()], map);
            transpose_square_in_place(&mut right[q.clone()], cols, map);
            PackedFactorPair {
                sector,
                kept: cols,
                left: l,
                left_rows: rows,
                right: q,
                right_leading: cols,
            }
        })
        .collect();
    Ok(PackedFactorPairs { pairs, left, right })
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
/// Sector by sector, the QR `X^H = A = Q R` in the positive-diagonal gauge
/// reads `A` itself, and `X = R^H Q^H` publishes `l = R^H`, `q = Q^H`: LQ
/// of the materialized `X`, up to the gauge freedom of a singular leading
/// triangular factor (see `TensorMap::lq_compact`). TensorKit (`src/factorizations/adjoint.jl`, `lq_compact!` /
/// `lq_full!` on an `AdjointTensorMap`) likewise factors the parent by QR and
/// adjoints the factors.
///
/// Why not a redirect to the facade QR of the parent: detaching `R^H` and
/// `Q^H` copies `k (m + n) >= m n` elements per sector; here they are written
/// once, as LQ of the materialized adjoint writes them.
fn lq_adjoint_by_tree<M, E, R, D>(
    engine: &mut dyn Engine,
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
    let (mut full_pairs, mut compact_pairs);
    let pairs: &mut dyn SidedFactors<D> = if full {
        full_pairs = full_lq_packed(engine, dense, &matrices, FactorPlacement::Adjoint)?;
        &mut full_pairs
    } else {
        compact_pairs =
            compact_lq_packed(engine, dense, &matrices, FactorPlacement::Adjoint, |_| {})?;
        &mut compact_pairs
    };
    let authority = M::authority(parent.space());
    // The bond of full LQ is the domain of `A^H`, the codomain of `A`.
    let dimensions = if full {
        authority.coupled_dimensions(space.homspace().codomain())?
    } else {
        (0..pairs.len())
            .map(|index| {
                let pair = pairs.pair(index);
                (pair.sector, pair.kept)
            })
            .collect()
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
                &mut *pairs,
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
    engine: &mut dyn Engine,
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
            return lq_compact_direct_regions(engine, dense, adjoint_space, reader, &plan)
                .map(|(l, q)| Lq { l, q });
        }
    }
    lq_adjoint_by_tree::<MultiplicityFreeAdmissionMode, _, _, _>(engine, dense, parent, false)
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
    engine: &mut dyn Engine,
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
        return lq_adjoint_by_tree::<MultiplicityFreeAdmissionMode, _, _, _>(
            engine, dense, parent, true,
        );
    };
    let matrices = AdjointParentMatrices {
        data: parent.data(),
        regions: &regions,
        by_adjoint: &by_adjoint,
    };
    let mut pairs = full_lq_packed(engine, dense, &matrices, FactorPlacement::Adjoint)?;
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
    engine: &mut dyn Engine,
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
        return lq_adjoint_by_tree::<CheckedGenericAdmissionMode, _, _, _>(
            engine, dense, parent, false,
        );
    };
    let matrices = AdjointParentMatrices {
        data: parent.data(),
        regions: &regions,
        by_adjoint: &by_adjoint,
    };
    let pairs = compact_lq_packed(engine, dense, &matrices, FactorPlacement::Adjoint, |_| {})?;
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
    engine: &mut dyn Engine,
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
        return lq_adjoint_by_tree::<CheckedGenericAdmissionMode, _, _, _>(
            engine, dense, parent, true,
        );
    };
    let matrices = AdjointParentMatrices {
        data: parent.data(),
        regions: &regions,
        by_adjoint: &by_adjoint,
    };
    let pairs = full_lq_packed(engine, dense, &matrices, FactorPlacement::Adjoint)?;
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
    engine: &mut dyn Engine,
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
    lease.run(|dense| M::lq_compact_adjoint_dense(engine, dense, &parent, adjoint_space))
}

/// Full LQ of the lazy adjoint of the dense `parent`, read in place, in
/// fusion mode `M` (see [`lq_compact_adjoint_from_parent`]).
#[doc(hidden)]
pub fn lq_full_adjoint_from_parent<M, L, E, R, D>(
    engine: &mut dyn Engine,
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
    lease.run(|dense| M::lq_full_adjoint_dense(engine, dense, &parent, adjoint_space))
}

pub(super) fn qr_compact_direct_regions<E, R, D>(
    engine: &mut dyn Engine,
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
    let mut left_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut right_data = vec![D::zero(); plan.right_layout.required_len()?];

    let routes = plan
        .routes
        .iter()
        .filter(|route| route.rank != 0)
        .collect::<Vec<_>>();
    let blocks = routes
        .iter()
        .map(|route| {
            let source = &plan.source_regions[route.source_region];
            (&input.data()[source.range()], source.rows(), source.cols())
        })
        .collect::<Vec<_>>();
    let left_ranges = routes
        .iter()
        .map(|route| {
            plan.left_regions[route.left_region.expect("nonzero route has left region")].range()
        })
        .collect::<Vec<_>>();
    let right_ranges = routes
        .iter()
        .map(|route| {
            plan.right_regions[route.right_region.expect("nonzero route has right region")].range()
        })
        .collect::<Vec<_>>();
    compact_qr_into_batch(
        engine,
        dense,
        &blocks,
        split_ranges(&mut left_data, &left_ranges)?,
        split_ranges(&mut right_data, &right_ranges)?,
    )?;

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
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<Lq<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let Lq { l, q } = lq_compact_dyn(engine, dense, &input.dynamic())?;
    Ok(Lq {
        l: typed_from_bound_factor(l)?,
        q: typed_from_bound_factor(q)?,
    })
}

/// Provider-bound compact LQ used by authority-preserving callers.
pub(crate) fn lq_compact_dyn<E, R, D>(
    engine: &mut dyn Engine,
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
            engine,
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
    let mut pairs = compact_lq_packed(
        engine,
        dense,
        matricizations.as_slice(),
        FactorPlacement::Direct,
        |_| {},
    )?;
    #[cfg(test)]
    for pair in &pairs.pairs {
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Lq, 1, pair.left.len());
        crate::probe::output_scatter::<D>(crate::probe::CopyOp::Lq, 1, pair.right.len());
    }
    build_left_right_bound_pair(input.space(), space.homspace(), &matricizations, &mut pairs)
        .map(|(l, q)| Lq { l, q })
}

/// What each compact-LQ route reads.
pub(super) enum LqRegionSource<'a, D> {
    /// The input payload: each source region is `X`, factored by the
    /// executor's LQ (the native provider reads its transpose in place; the
    /// default and BLAS stage its adjoint for QR).
    Input(&'a [D]),
    /// A lazy adjoint's parent blocks, factored by QR in place (#2070).
    Parent(AdjointParentMatrices<'a, D>),
}

/// Compact LQ of `input_space` over the direct-region `plan`, reading `reader`.
pub(super) fn lq_compact_direct_regions<E, R, D>(
    engine: &mut dyn Engine,
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
    let mut left_data = vec![D::zero(); plan.left_layout.required_len()?];
    let mut right_data = vec![D::zero(); plan.right_layout.required_len()?];
    let routes = plan
        .routes
        .iter()
        .filter(|route| route.rank != 0)
        .collect::<Vec<_>>();
    let (placement, blocks) = match &reader {
        LqRegionSource::Input(data) => (
            FactorPlacement::Direct,
            routes
                .iter()
                .map(|route| {
                    let source = &plan.source_regions[route.source_region];
                    (&data[source.range()], source.rows(), source.cols())
                })
                .collect::<Vec<_>>(),
        ),
        LqRegionSource::Parent(parent) => (
            FactorPlacement::Adjoint,
            routes
                .iter()
                .map(|route| {
                    let block = parent.get(route.source_region)?;
                    Ok((block.data, block.rows, block.cols))
                })
                .collect::<Result<Vec<_>, OperationError>>()?,
        ),
    };
    let left_ranges = routes
        .iter()
        .map(|route| {
            plan.left_regions[route.left_region.expect("nonzero route has left region")].range()
        })
        .collect::<Vec<_>>();
    let right_ranges = routes
        .iter()
        .map(|route| {
            plan.right_regions[route.right_region.expect("nonzero route has right region")].range()
        })
        .collect::<Vec<_>>();
    compact_lq_into_batch(
        engine,
        dense,
        &blocks,
        placement,
        split_ranges(&mut left_data, &left_ranges)?,
        split_ranges(&mut right_data, &right_ranges)?,
    )?;

    let left = BoundDynFactor::from_bound(left_space, left_data, space.nout(), 1)?;
    let right = BoundDynFactor::from_bound(right_space, right_data, 1, space.nin())?;
    Ok((left, right))
}

/// Writes `map` of each entry of the column-major `rows x cols` matrix
/// `data`, transposed, into the column-major `cols x rows` matrix `out`.
pub(super) fn transpose_col_major_into<D: Copy>(
    data: &[D],
    rows: usize,
    cols: usize,
    out: &mut [D],
    map: impl Fn(D) -> D,
) {
    debug_assert_eq!(data.len(), rows * cols);
    debug_assert_eq!(out.len(), data.len());
    for col in 0..cols {
        for row in 0..rows {
            out[col + cols * row] = map(data[row + rows * col]);
        }
    }
}

/// Replaces the column-major `n x n` matrix `data` by `map` of each entry of
/// its transpose, in place.
fn transpose_square_in_place<D: Copy>(data: &mut [D], n: usize, map: impl Fn(D) -> D) {
    debug_assert_eq!(data.len(), n * n);
    for col in 0..n {
        data[col + n * col] = map(data[col + n * col]);
        for row in col + 1..n {
            let (lower, upper) = (row + n * col, col + n * row);
            let value = data[upper];
            data[upper] = map(data[lower]);
            data[lower] = map(value);
        }
    }
}

/// Compact QR of each column-major `(data, rows, cols)` block into its
/// destination slices `q` (`rows x rank`) and `r` (`rank x cols`),
/// submitted as one solver-first executor batch so the backend admits the
/// whole coupled-sector loop once; each solved block is positive-diagonal
/// gauged in place.
pub(super) fn compact_qr_into_batch<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    blocks: &[(&[D], usize, usize)],
    q: Vec<&mut [D]>,
    r: Vec<&mut [D]>,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let mut read_layouts = Vec::new();
    let inputs = col_major_reads(blocks, &mut read_layouts)?;
    let layouts = blocks
        .iter()
        .map(|&(_, rows, cols)| {
            let rank = rows.min(cols);
            [
                matrix_layout(rows, rank),
                matrix_layout(rank, cols),
                vector_layout(0),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    let mut dests = q
        .into_iter()
        .zip(r)
        .zip(&layouts)
        .map(|((q, r), layout)| {
            Ok(FactorDest::new(
                [
                    Some(dest_write(q, &layout[0])?),
                    Some(dest_write(r, &layout[1])?),
                    None,
                ],
                symtn_dense::FactorPlacement::AsComputed,
            ))
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    factorize_into_solver_first(
        engine,
        dense,
        DenseFactorization::Qr,
        &inputs,
        &mut dests,
        |index, dest| {
            let (_, rows, cols) = blocks[index];
            let rank = rows.min(cols);
            let [q, r, _] = dest.outputs_mut();
            positive_diagonal_gauge_strided(
                dest_data::<D>(q, "qr_into")?,
                rows,
                rows,
                dest_data::<D>(r, "qr_into")?,
                rank,
                rank,
                cols,
            );
            Ok(())
        },
    )
}

/// Checked-Generic compact QR. Provider-bound output spaces are admitted
/// through the checked staging boundary; dense QR itself performs no provider
/// queries and therefore needs no Tenferro-specific capability.
#[doc(hidden)]
pub(crate) fn qr_compact_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
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
    let pairs = compact_qr_pairs(engine, dense, &matrices, |_matrix| {
        #[cfg(test)]
        crate::probe::checked_compact_input(
            crate::probe::CheckedCompactOperation::Qr,
            data,
            _matrix.data,
        );
    })?;
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
        .map(|(q, r)| Qr { q, r })
}

/// Checked-Generic compact LQ through the executor's compact LQ batch; no
/// borrowed conjugated-dot capability is needed.
#[doc(hidden)]
pub(crate) fn lq_compact_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
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
        engine,
        dense,
        &matrices,
        FactorPlacement::Direct,
        |_matrix| {
            #[cfg(test)]
            crate::probe::checked_compact_input(
                crate::probe::CheckedCompactOperation::Lq,
                data,
                _matrix.data,
            );
        },
    )?;
    build_checked_pair_from_input(provider, space.homspace(), &matrices, pairs)
        .map(|(l, q)| Lq { l, q })
}

/// Checked-Generic full QR: one executor batch over every sector (see
/// [`full_qr_packed`]).
#[doc(hidden)]
pub(crate) fn qr_full_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
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
    let pairs =
        full_qr_packed(engine, dense, &matrices).map_err(CheckedGenericFactorPlanError::from)?;
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

/// Checked-Generic full LQ: one executor batch over every sector (see
/// [`full_lq_packed`]).
#[doc(hidden)]
pub(crate) fn lq_full_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
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
    let pairs = full_lq_packed(engine, dense, &matrices, FactorPlacement::Direct)
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
