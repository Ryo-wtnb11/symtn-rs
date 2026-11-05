use super::*;

/// The canonical output region one input sector's result matrix lands in.
///
/// A per-sector kernel computes its result column-major in the input
/// sector's own basis order, and writes the output region in place: every
/// factorization input and output is built in its hom space's canonical tree
/// order, which enumerates each side's trees per coupled sector from that
/// side's legs alone, so a sector's row and column trees are the output's at
/// the same offsets. [`compile_sector_landings`] checks it.
pub(super) struct SectorLanding {
    pub(super) output: usize,
}

/// Compiles the landing of every `source` sector in `output`, whose rows are
/// the trees of the input side `rows` and columns those of `cols`. Every
/// output sector must receive exactly one input sector, with the same trees at
/// the same offsets.
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
        if !same_basis(matrix, rows, region.row_trees())
            || !same_basis(matrix, cols, region.col_trees())
        {
            return Err(OperationError::SpaceMismatch {
                message: "factor output tree basis differs from the source coupled sector",
            });
        }
        used[index] = true;
        landings.push(SectorLanding { output: index });
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
