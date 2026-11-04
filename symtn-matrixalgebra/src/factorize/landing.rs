use super::*;

/// Where one input sector's result matrix lands in a canonical output region.
///
/// A per-sector kernel computes its result column-major in the input
/// sector's own basis order: rows indexed by the trees of one input side,
/// columns by those of another. The output region usually lists the same
/// trees at the same offsets, and the kernel then writes the region in place.
/// When it does not (a packed input whose first-appearance tree order differs
/// from the derived output layout), the result is staged and copied by tree
/// extent.
pub(super) struct SectorLanding {
    pub(super) output: usize,
    reorder: Option<(Vec<InverseBasisExtent>, Vec<InverseBasisExtent>)>,
}

impl SectorLanding {
    /// Runs `write` on this sector's contiguous `rows x cols` destination
    /// (leading dimension `rows`), in place or through `scratch`.
    pub(super) fn write<D: FactorScalar>(
        &self,
        output_data: &mut [D],
        output: &CoupledSectorRegion,
        scratch: &mut Vec<D>,
        write: impl FnOnce(&mut [D]) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let destination = &mut output_data[output.range()];
        if self.reorder.is_none() {
            return write(destination);
        }
        scratch.clear();
        scratch.resize(destination.len(), D::zero());
        write(scratch)?;
        self.land(scratch, destination, output);
        Ok(())
    }

    /// Whether this sector's result is staged and reordered rather than
    /// written in place.
    pub(super) fn reorders(&self) -> bool {
        self.reorder.is_some()
    }

    /// Copies the `staged` result, in the input sector's basis order, by
    /// tree extent into `destination`, the output region `output`'s data; a
    /// sector that lands in place has nothing to copy.
    pub(super) fn land<D: Copy>(
        &self,
        staged: &[D],
        destination: &mut [D],
        output: &CoupledSectorRegion,
    ) {
        if let Some((rows, cols)) = &self.reorder {
            reorder_inverse_solution(
                staged,
                output.rows(),
                destination,
                output.rows(),
                rows,
                cols,
            );
        }
    }
}

/// Compiles the landing of every `source` sector in `output`, whose rows are
/// the trees of the input side `rows` and columns those of `cols`. Every
/// output sector must receive exactly one input sector.
pub(super) fn compile_sector_landings<M: SectorGeometry>(
    source: &[M],
    output: &[CoupledSectorRegion],
    output_len: usize,
    rows: FactorSide,
    cols: FactorSide,
) -> Result<Vec<SectorLanding>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut landings = Vec::with_capacity(source.len());
    for matrix in source {
        let index = output_by_sector.get(&matrix.sector()).copied().ok_or(
            OperationError::SpaceMismatch {
                message: "factor output is missing a source coupled sector",
            },
        )?;
        let region = &output[index];
        if region.rows() != side_dimension(matrix, rows)
            || region.cols() != side_dimension(matrix, cols)
        {
            return Err(OperationError::SpaceMismatch {
                message: "factor output does not match the source coupled-sector dimensions",
            });
        }
        validate_region_range(region, output_len)?;
        let reorder = if same_basis(matrix, rows, region.row_trees())
            && same_basis(matrix, cols, region.col_trees())
        {
            None
        } else {
            Some((
                compile_basis_extents(matrix, rows, region.row_trees())?,
                compile_basis_extents(matrix, cols, region.col_trees())?,
            ))
        };
        used[index] = true;
        landings.push(SectorLanding {
            output: index,
            reorder,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::SpaceMismatch {
            message: "factor output contains a coupled sector absent from the source",
        });
    }
    Ok(landings)
}

fn side_dimension<M: SectorGeometry>(matrix: &M, side: FactorSide) -> usize {
    match side {
        FactorSide::Left => matrix.rows(),
        FactorSide::Right => matrix.cols(),
    }
}

/// Whether `side` of `matrix` lists exactly `output`'s trees at the same
/// offsets and shapes, so its basis order is the output's.
fn same_basis<M: SectorGeometry>(
    matrix: &M,
    side: FactorSide,
    output: &[CoupledTreeExtent],
) -> bool {
    matrix.tree_count(side) == output.len()
        && output.iter().enumerate().all(|(index, extent)| {
            matrix.tree(side, index).is_some_and(|tree| {
                tree.tree == extent.tree()
                    && tree.offset == extent.offset()
                    && tree.shape == extent.shape()
            })
        })
}

/// Maps every tree of `side` of `matrix` onto its extent in `output`, which
/// must list each of those trees once with the same degeneracy shape.
pub(super) fn compile_basis_extents<M: SectorGeometry>(
    matrix: &M,
    side: FactorSide,
    output: &[CoupledTreeExtent],
) -> Result<Vec<InverseBasisExtent>, OperationError> {
    let output_by_tree = output
        .iter()
        .enumerate()
        .map(|(index, extent)| (extent.tree(), index))
        .collect::<FxHashMap<_, _>>();
    if output_by_tree.len() != output.len() {
        return Err(OperationError::SpaceMismatch {
            message: "inverse output contains a duplicate tree basis",
        });
    }
    let mut used = vec![false; output.len()];
    let count = matrix.tree_count(side);
    let mut extents = Vec::with_capacity(count);
    for index in 0..count {
        let Some(source) = matrix.tree(side, index) else {
            break;
        };
        let output_index =
            output_by_tree
                .get(source.tree)
                .copied()
                .ok_or(OperationError::SpaceMismatch {
                    message: "inverse output is missing a source tree basis",
                })?;
        let output_extent = &output[output_index];
        if source.shape != output_extent.shape() {
            return Err(OperationError::SpaceMismatch {
                message: "inverse output tree basis has an unexpected shape",
            });
        }
        let extent = output_extent
            .extent()
            .map_err(OperationError::from_core_preserving_context)?;
        used[output_index] = true;
        extents.push(InverseBasisExtent {
            source_offset: source.offset,
            output_offset: output_extent.offset(),
            extent,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::SpaceMismatch {
            message: "inverse output contains a tree basis absent from the source",
        });
    }
    Ok(extents)
}
