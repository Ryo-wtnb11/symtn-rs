use super::*;
use symtn_dense::Engine;

/// Runs `body` once inside a single executor linear-algebra scope, so the
/// backend admits the body's dense calls once. The scope's own error takes
/// precedence over the body's error, which is returned otherwise. The body
/// decides the working set: a per-block loop streams one block's factors at
/// a time, while a `factorize_batch` inside it may retain every block's
/// factors until they are delivered.
#[doc(hidden)]
pub fn in_linalg_scope<E, T>(
    engine: &mut dyn Engine,
    dense: &mut E,
    body: impl FnOnce(&mut dyn DenseExecutor, &mut dyn Engine) -> Result<T, OperationError> + Send,
) -> Result<T, OperationError>
where
    E: DenseExecutor + ?Sized,
    T: Send,
{
    let mut body = Some(body);
    let mut outcome = None;
    #[cfg(test)]
    let mut probes = Some(crate::probe::Carried::take());
    let scoped = dense.with_linalg_scope(engine, &mut |dense, engine| {
        if let Some(body) = body.take() {
            #[cfg(test)]
            if let Some(probes) = probes.take() {
                probes.put();
            }
            outcome = Some(body(dense, engine));
            #[cfg(test)]
            {
                probes = Some(crate::probe::Carried::take());
            }
        }
        Ok(())
    });
    #[cfg(test)]
    if let Some(probes) = probes.take() {
        probes.put();
    }
    scoped.map_err(OperationError::Dense)?;
    outcome.unwrap_or_else(|| {
        Err(OperationError::Dense(DenseError::Unsupported {
            op: "with_linalg_scope",
            message: "dense executor returned without running the scope body".to_string(),
        }))
    })
}

/// Runs `op` over `inputs` as one executor batch and hands `stage` each
/// input's outcome in input order; a `stage` error stops the batch and is
/// returned.
///
/// An overriding executor is outside this crate, so its delivery is checked:
/// an input delivered twice, out of order or not at all is symtn's typed
/// arity error, never a dropped or misplaced sector.
pub(super) fn factorize_batch_staged<E>(
    engine: &mut dyn Engine,
    dense: &mut E,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    mut stage: impl FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), OperationError>,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
{
    let mut delivered = 0;
    let mut failure = None;
    let batch = dense.factorize_batch(engine, op, inputs, &mut |index, outcome| {
        let staged = if index != delivered || index >= inputs.len() {
            Err(OperationError::Dense(arity_mismatch(
                "factorize_batch",
                delivered,
                index,
            )))
        } else {
            delivered += 1;
            stage(index, outcome)
        };
        // The executor only needs to stop; the error that stopped it is
        // kept here, so the stop signal carries nothing.
        staged.map_err(|error| {
            failure = Some(error);
            DenseError::Unsupported {
                op: "factorize_batch",
                message: String::new(),
            }
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    batch.map_err(OperationError::Dense)?;
    if delivered != inputs.len() {
        return Err(OperationError::Dense(arity_mismatch(
            "factorize_batch",
            inputs.len(),
            delivered,
        )));
    }
    Ok(())
}

/// [`factorize_batch_staged`] into caller-owned destinations: `stage` sees
/// each delivered input's destination and tagged outcome in input order;
/// a `stage` error stops the batch and is returned. The same delivery check
/// guards an executor that overrides the destination entry.
pub(super) fn factorize_batch_into_staged<E>(
    engine: &mut dyn Engine,
    dense: &mut E,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    mut stage: impl FnMut(
        usize,
        &mut FactorDest<'_>,
        Result<(), FactorFailure>,
    ) -> Result<(), OperationError>,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
{
    let mut delivered = 0;
    let mut failure = None;
    let batch =
        dense.factorize_batch_into(engine, op, inputs, dests, &mut |index, dest, outcome| {
            let staged = if index != delivered || index >= inputs.len() {
                Err(OperationError::Dense(arity_mismatch(
                    "factorize_batch",
                    delivered,
                    index,
                )))
            } else {
                delivered += 1;
                #[cfg(test)]
                crate::probe::staged_destination(index, dest);
                stage(index, dest, outcome)
            };
            staged.map_err(|error| {
                failure = Some(error);
                DenseError::Unsupported {
                    op: "factorize_batch",
                    message: String::new(),
                }
            })
        });
    if let Some(error) = failure {
        return Err(error);
    }
    batch.map_err(OperationError::Dense)?;
    if delivered != inputs.len() {
        return Err(OperationError::Dense(arity_mismatch(
            "factorize_batch",
            inputs.len(),
            delivered,
        )));
    }
    Ok(())
}

/// The solver-first destination batch: the first solver failure stops it;
/// the first malformed output is returned only once the whole batch has
/// succeeded, so a later solver failure takes precedence. `solved` sees
/// each successfully delivered destination.
pub(super) fn factorize_into_solver_first<E>(
    engine: &mut dyn Engine,
    dense: &mut E,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    mut solved: impl FnMut(usize, &mut FactorDest<'_>) -> Result<(), OperationError>,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
{
    let mut malformed = None;
    factorize_batch_into_staged(engine, dense, op, inputs, dests, |index, dest, outcome| {
        match outcome {
            Ok(()) => solved(index, dest),
            Err(FactorFailure::Solver(error)) => Err(OperationError::Dense(error)),
            Err(FactorFailure::Output(error)) => {
                malformed.get_or_insert(error);
                Ok(())
            }
        }
    })?;
    match malformed {
        Some(error) => Err(OperationError::Dense(error)),
        None => Ok(()),
    }
}

/// The column-major layouts of one destination's outputs: `(shape,
/// strides, rank)`, a spectrum using the first extent only.
pub(super) type DestLayouts = [([usize; 2], [usize; 2], usize); 3];

pub(super) fn matrix_layout(rows: usize, cols: usize) -> ([usize; 2], [usize; 2], usize) {
    ([rows, cols], [1, rows], 2)
}

pub(super) fn vector_layout(len: usize) -> ([usize; 2], [usize; 2], usize) {
    ([len, 0], [1, 0], 1)
}

/// A destination view over all of `data` with layout `layout`.
pub(super) fn dest_write<'a, T: DenseBlockScalar>(
    data: &'a mut [T],
    (shape, strides, rank): &'a ([usize; 2], [usize; 2], usize),
) -> Result<DenseWrite<'a>, OperationError> {
    DenseViewMut::new(data, &shape[..*rank], &strides[..*rank], 0)
        .map(T::dense_write)
        .map_err(OperationError::Dense)
}

/// The storage of a delivered destination output this crate built.
pub(super) fn dest_data<'b, T: DenseBlockScalar>(
    output: &'b mut Option<DenseWrite<'_>>,
    op: &'static str,
) -> Result<&'b mut [T], OperationError> {
    output
        .as_mut()
        .and_then(T::dense_write_data)
        .ok_or_else(|| OperationError::Dense(arity_mismatch(op, 1, 0)))
}

/// Splits `data` into the disjoint `ranges` (any order), returned in the
/// order given; ranges are cut in storage order with `split_at_mut`.
pub(super) fn split_ranges<'a, T>(
    data: &'a mut [T],
    ranges: &[std::ops::Range<usize>],
) -> Result<Vec<&'a mut [T]>, OperationError> {
    let mut order = (0..ranges.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| ranges[index].start);
    let mut slices = (0..ranges.len())
        .map(|_| None)
        .collect::<Vec<Option<&mut [T]>>>();
    let mut rest = data;
    let mut cursor = 0usize;
    for index in order {
        let range = &ranges[index];
        let mismatch = OperationError::ElementCountMismatch {
            expected: cursor,
            actual: range.start,
        };
        let skip = range.start.checked_sub(cursor).ok_or(mismatch)?;
        let len = range.len();
        if skip + len > rest.len() {
            return Err(OperationError::ElementCountMismatch {
                expected: rest.len(),
                actual: skip + len,
            });
        }
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(skip);
        let (slice, tail) = tail.split_at_mut(len);
        slices[index] = Some(slice);
        rest = tail;
        cursor = range.end;
    }
    Ok(slices
        .into_iter()
        .map(|slice| slice.expect("every range was cut"))
        .collect())
}

/// Consecutive ranges of `lens`, in order.
pub(super) fn packed_ranges(lens: impl Iterator<Item = usize>) -> Vec<std::ops::Range<usize>> {
    let mut start = 0usize;
    lens.map(|len| {
        let range = start..start + len;
        start += len;
        range
    })
    .collect()
}

/// A delivered spectrum slice widened to the public `f64` owner.
pub(super) fn widened_spectrum<R: Copy + Into<f64>>(values: &[R]) -> Vec<f64> {
    values.iter().map(|&value| value.into()).collect()
}

/// The one tensor of a values-only batch outcome, arity checked.
pub(super) fn single_output(
    outcome: Result<Vec<DenseTensor>, DenseError>,
    op: &'static str,
) -> Result<DenseTensor, OperationError> {
    let [tensor]: [DenseTensor; 1] = outcome
        .map_err(OperationError::Dense)?
        .try_into()
        .map_err(|outputs: Vec<_>| OperationError::Dense(arity_mismatch(op, 1, outputs.len())))?;
    Ok(tensor)
}

/// Each sector's id and its column-major `(data, rows, cols)` matrix, in
/// sector order.
#[expect(
    clippy::type_complexity,
    reason = "the sector ids and the batch's borrowed matrices are parallel lists"
)]
pub(super) fn sector_blocks<D>(
    matrices: &(impl SectorMatrices<D> + ?Sized),
) -> Result<(Vec<SectorId>, Vec<(&[D], usize, usize)>), OperationError> {
    (0..matrices.len())
        .map(|index| {
            let matrix = matrices.get(index)?;
            Ok((matrix.sector, (matrix.data, matrix.rows, matrix.cols)))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|pairs| pairs.into_iter().unzip())
}

/// Column-major `(data, rows, cols)` blocks as dense reads, in order.
pub(super) fn col_major_reads<'a, D: FactorScalar>(
    blocks: &[(&'a [D], usize, usize)],
    layouts: &'a mut Vec<([usize; 2], [usize; 2])>,
) -> Result<Vec<DenseRead<'a>>, OperationError> {
    block_reads(blocks, layouts, false)
}

/// Column-major `(data, rows, cols)` blocks as dense reads, in order: each
/// block itself, or with `transposed` its `cols x rows` transpose (extents
/// and strides swapped, no copy).
pub(super) fn block_reads<'a, D: FactorScalar>(
    blocks: &[(&'a [D], usize, usize)],
    layouts: &'a mut Vec<([usize; 2], [usize; 2])>,
    transposed: bool,
) -> Result<Vec<DenseRead<'a>>, OperationError> {
    layouts.clear();
    layouts.extend(blocks.iter().map(|&(_, rows, cols)| {
        if transposed {
            ([cols, rows], [rows, 1usize])
        } else {
            ([rows, cols], [1usize, rows])
        }
    }));
    let layouts: &'a [_] = layouts;
    blocks
        .iter()
        .zip(layouts)
        .map(|(&(data, _, _), (shape, strides))| {
            DenseView::new(data, shape, strides, 0).map(D::dense_read)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(OperationError::Dense)
}

pub(super) fn compact_factor_output_owned<D: FactorScalar>(
    tensor: DenseTensor,
    expected_shape: &[usize],
    op: &'static str,
) -> Result<Vec<D>, OperationError> {
    let source = D::dense_slice(&tensor).map_err(OperationError::Dense)?;
    check_contiguous_output(op, source.len(), tensor.shape(), expected_shape)
        .map_err(OperationError::Dense)?;
    D::dense_into_vec(tensor).map_err(OperationError::Dense)
}

pub(super) fn compact_real_spectrum_owned<D: FactorScalar>(
    tensor: DenseTensor,
    expected_shape: &[usize],
    op: &'static str,
) -> Result<Vec<f64>, OperationError> {
    let spectrum = D::real_spectrum(&tensor).map_err(OperationError::Dense)?;
    check_contiguous_output(op, spectrum.len(), tensor.shape(), expected_shape)
        .map_err(OperationError::Dense)?;
    Ok(spectrum)
}

pub(super) fn concat_owned_factor_regions<D>(
    regions: Vec<Option<Vec<D>>>,
    required_len: usize,
) -> Vec<D> {
    let mut output = None;
    for region in regions.into_iter().flatten() {
        append_owned_factor(&mut output, region, required_len);
    }
    output.unwrap_or_default()
}

pub(super) fn copy_col_major_strided<D: Copy>(
    source: &[D],
    rows: usize,
    cols: usize,
    source_leading: usize,
    destination: &mut [D],
    destination_leading: usize,
) {
    for col in 0..cols {
        let src_start = source_leading * col;
        let dst_start = destination_leading * col;
        destination[dst_start..dst_start + rows]
            .copy_from_slice(&source[src_start..src_start + rows]);
    }
}

pub(super) fn advance_outer_index(index: &mut [usize], shape: &[usize]) {
    for axis in 1..shape.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            break;
        }
        index[axis] = 0;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_tensor_block_to_matrix<D: Copy>(
    source: &[D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    nout: usize,
    matrix: &mut [D],
    matrix_rows: usize,
    row_offset: usize,
    col_offset: usize,
) {
    if shape.is_empty() {
        matrix[row_offset + matrix_rows * col_offset] = source[offset];
        return;
    }
    let run = shape[0];
    let src_lane_stride = strides[0];
    let dst_lane_stride = if nout > 0 { 1 } else { matrix_rows };
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..outer_count {
        let mut src_start = offset;
        let mut row = 0usize;
        let mut row_stride = if nout > 0 { shape[0] } else { 1 };
        let mut col = 0usize;
        let mut col_stride = if nout == 0 { shape[0] } else { 1 };
        for axis in 1..shape.len() {
            src_start += index[axis] * strides[axis];
            if axis < nout {
                row += index[axis] * row_stride;
                row_stride *= shape[axis];
            } else {
                col += index[axis] * col_stride;
                col_stride *= shape[axis];
            }
        }
        let dst_start = (row_offset + row) + matrix_rows * (col_offset + col);
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            matrix[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                matrix[dst_start + lane * dst_lane_stride] =
                    source[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

pub(super) fn copy_mapped_to_strided_diagonal<D, V, F>(
    data: &mut [D],
    offset: usize,
    diagonal_stride: usize,
    values: &[V],
    to_scalar: &F,
) where
    V: Copy,
    F: Fn(V) -> D + ?Sized,
{
    for (position, &value) in values.iter().enumerate() {
        data[offset + position * diagonal_stride] = to_scalar(value);
    }
}

/// Copies a dense column-major matrix region into one fusion-tree subblock.
///
/// `matrix_axis` names the block axis that walks the matrix's own leading
/// dimension side; the remaining axes enumerate the offset side column-major.
/// For `U` the matrix axis is the trailing (new leg) axis and the codomain
/// axes select rows at `side_offset`; for `Vt` the matrix axis is the leading
/// (new leg) axis and the domain axes select columns at `side_offset`.
/// `factor_side` names the factor layout (`Left`: `a x b`, element `(o, j)` at
/// `F[o + a*j]`; `Right`: `b x cols`, element `(j, o)` at `F[j + b*o]`).
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_matrix_block<D: Copy>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    factor_side: FactorSide,
    matrix: &[D],
    matrix_rows: usize,
    side_offset: usize,
) {
    if shape.is_empty() {
        data[offset] = matrix[side_offset];
        return;
    }
    let rank = shape.len();
    let run = shape[0];
    let dst_lane_stride = strides[0];
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; rank];
    for _ in 0..outer_count {
        let mut dst_start = offset;
        let mut side = 0usize;
        let mut side_stride = if matrix_axis == 0 { 1 } else { shape[0] };
        let mut matrix_index = 0usize;
        for axis in 1..rank {
            dst_start += index[axis] * strides[axis];
            if axis == matrix_axis {
                matrix_index = index[axis];
            } else {
                side += index[axis] * side_stride;
                side_stride *= shape[axis];
            }
        }
        // A rank-1 block (zero-leg tree) has `matrix_axis == 0 == rank - 1` on
        // both sides, so `matrix_axis` alone cannot encode the source layout;
        // only a Left factor walks the bond with stride `matrix_rows`.
        let (src_start, src_lane_stride) = if matrix_axis == 0 {
            if matrix_axis == rank - 1 && factor_side == FactorSide::Left {
                (side_offset + side, matrix_rows)
            } else {
                (matrix_rows * (side_offset + side), 1)
            }
        } else if matrix_axis == rank - 1 {
            ((side_offset + side) + matrix_rows * matrix_index, 1)
        } else {
            (
                matrix_index + matrix_rows * (side_offset + side),
                matrix_rows,
            )
        };
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            data[dst_start..dst_start + run].copy_from_slice(&matrix[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                data[dst_start + lane * dst_lane_stride] =
                    matrix[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_identity_matrix_block<D: FactorScalar>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    matrix_axis: usize,
    matrix_dimension: usize,
    side_offset: usize,
    side_extent: usize,
) -> Result<(), OperationError> {
    if shape.get(matrix_axis).copied() != Some(matrix_dimension) {
        return Err(OperationError::ElementCountMismatch {
            expected: matrix_dimension,
            actual: shape.get(matrix_axis).copied().unwrap_or(0),
        });
    }
    for local_side in 0..side_extent {
        let matrix_index = side_offset
            .checked_add(local_side)
            .ok_or(OperationError::ElementCountOverflow)?;
        let mut destination = offset + matrix_index * strides[matrix_axis];
        let mut remaining = local_side;
        for axis in 0..shape.len() {
            if axis == matrix_axis {
                continue;
            }
            let coordinate = remaining % shape[axis];
            remaining /= shape[axis];
            destination += coordinate * strides[axis];
        }
        data[destination] = D::one();
    }
    Ok(())
}
