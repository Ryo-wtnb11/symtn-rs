use super::*;

/// One side of a tensor map `codomain <- domain`.
///
/// Used where an operation acts on exactly one side, such as
/// [`TensorMap::cat`] and the seam choice of [`TensorMap::insert_unit`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Side {
    /// The codomain (output) legs.
    Codomain,
    /// The domain (input) legs.
    Domain,
}

/// Which of a pair of mutually inverse maps to apply.
///
/// Used by [`TensorMap::twist`] and [`TensorMap::flip`], whose inverse
/// TensorKit spells with the `inv` keyword.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Direction {
    /// The map itself.
    Forward,
    /// Its inverse.
    Inverse,
}

impl Direction {
    pub(super) fn is_inverse(self) -> bool {
        self == Self::Inverse
    }
}

/// Duality flag of a leg created by an operation, such as the unit leg of
/// [`TensorMap::insert_unit`] (TensorKit's `dual` keyword).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Duality {
    /// A leg on the space itself.
    Plain,
    /// A leg on the dual space.
    Dual,
}

impl Duality {
    pub(super) fn is_dual(self) -> bool {
        self == Self::Dual
    }
}

#[derive(Clone, Copy)]
pub(crate) enum TensorOrientation {
    Owned,
    Adjoint,
}

/// Structure-level description of one concatenation operand.
pub(crate) struct CatOperandLayout<'a> {
    structure: &'a Arc<BlockStructure>,
    regions: Arc<[CoupledSectorRegion]>,
    orientation: TensorOrientation,
    logical_nout: usize,
    storage_nout: usize,
    rank: usize,
}

impl<'a> CatOperandLayout<'a> {
    pub(crate) fn owned(
        structure: &'a Arc<BlockStructure>,
        nout: usize,
        nin: usize,
    ) -> Result<Self, Error> {
        Ok(Self::from_regions(
            structure,
            sector_regions(structure, nout)?,
            TensorOrientation::Owned,
            nout,
            nout,
            nout + nin,
        ))
    }

    pub(crate) fn adjoint(
        structure: &'a Arc<BlockStructure>,
        storage_nout: usize,
        storage_nin: usize,
    ) -> Result<Self, Error> {
        Ok(Self::from_regions(
            structure,
            sector_regions(structure, storage_nout)?,
            TensorOrientation::Adjoint,
            storage_nin,
            storage_nout,
            storage_nout + storage_nin,
        ))
    }

    pub(crate) fn from_regions(
        structure: &'a Arc<BlockStructure>,
        regions: Arc<[CoupledSectorRegion]>,
        orientation: TensorOrientation,
        logical_nout: usize,
        storage_nout: usize,
        rank: usize,
    ) -> Self {
        Self {
            structure,
            regions,
            orientation,
            logical_nout,
            storage_nout,
            rank,
        }
    }
}

/// Compiled per-fusion-tree copy plan for one concatenation.
pub(crate) struct CatCopyPlan {
    required_len: usize,
    copies: Vec<OwnedCatCopy>,
    side: Side,
}

#[derive(Clone, Copy)]
pub(crate) enum CatOperandData<'a, D> {
    Dense(&'a [D]),
    Diagonal {
        structure: &'a Arc<BlockStructure>,
        spectrum: &'a [symtn_matrixalgebra::SectorSpectrum<D>],
    },
}

impl CatCopyPlan {
    pub(crate) fn execute<D: ScalarOps>(
        &self,
        engine: &mut dyn Engine,
        sources: [CatOperandData<'_, D>; 2],
    ) -> Result<Vec<D>, Error> {
        let side = match self.side {
            Side::Domain => OwnedCatSide::Domain,
            Side::Codomain => OwnedCatSide::Codomain,
        };
        if let [CatOperandData::Dense(lhs), CatOperandData::Dense(rhs)] = sources {
            return symtn_operations::host_pool::region(
                engine,
                symtn_operations::host_pool::HostPoolSite::Strided,
                |exec| {
                    symtn_tensors::try_cat_owned_raw(
                        exec,
                        self.required_len,
                        side,
                        &self.copies,
                        [lhs, rhs],
                    )
                },
            )?
            .ok_or_else(|| {
                internal_layout_error(
                    "cat copy plan declined by the fast-path prover; no known geometry reaches \
                     this",
                )
            });
        }
        let source_lengths = sources.map(|source| match source {
            CatOperandData::Dense(values) => Ok(values.len()),
            CatOperandData::Diagonal { structure, .. } => structure.required_len(),
        });
        let [lhs_len, rhs_len] = source_lengths;
        let source_lengths = [lhs_len?, rhs_len?];
        for source in &sources {
            let CatOperandData::Diagonal {
                structure,
                spectrum,
            } = source
            else {
                continue;
            };
            if structure.block_count() != spectrum.len() {
                return Err(internal_layout_error(
                    "compact cat spectrum block count disagrees",
                ));
            }
            for (block_index, entry) in spectrum.iter().enumerate() {
                let block = structure.block(block_index)?;
                let pair = block.key().as_fusion_tree_pair().ok_or_else(|| {
                    internal_layout_error("compact cat source is not fusion-tree keyed")
                })?;
                if pair.codomain_tree().coupled() != entry.sector {
                    return Err(internal_layout_error(
                        "compact cat spectrum sector is absent",
                    ));
                }
                let [rows, cols] = block.shape() else {
                    return Err(internal_layout_error("compact cat source is not a matrix"));
                };
                if rows != cols || entry.values.len() != *rows {
                    return Err(internal_layout_error(
                        "compact cat spectrum shape disagrees",
                    ));
                }
            }
        }
        let dense_sources = sources.map(|source| match source {
            CatOperandData::Dense(values) => Some(values),
            CatOperandData::Diagonal { .. } => None,
        });
        // The compiled copies follow increasing coupled sectors; a compact
        // rank-(1,1) bond has exactly one physical block in each sector.
        let mut source_positions = [0usize; 2];
        let compact_entries = self
            .copies
            .iter()
            .map(|copy| {
                let Some(CatOperandData::Diagonal {
                    structure,
                    spectrum,
                }) = sources.get(copy.source())
                else {
                    return Ok(None);
                };
                let position = &mut source_positions[copy.source()];
                let block = structure.block(*position)?;
                let entry = spectrum
                    .get(*position)
                    .ok_or_else(|| internal_layout_error("compact cat source region is absent"))?;
                *position += 1;
                if copy.rows() != copy.cols()
                    || block.offset() != copy.source_offset()
                    || entry.values.len() != copy.rows()
                    || copy.source_row_stride() != 1
                    || copy.source_column_stride() != copy.rows()
                    || copy.conjugate()
                {
                    return Err(internal_layout_error("compact cat copy geometry disagrees"));
                }
                Ok(Some(entry.values.as_slice()))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        for (index, source) in sources.iter().enumerate() {
            if let CatOperandData::Diagonal { structure, .. } = source {
                if source_positions[index] != structure.block_count() {
                    return Err(internal_layout_error("compact cat source region is absent"));
                }
            }
        }
        symtn_operations::host_pool::region(
            engine,
            symtn_operations::host_pool::HostPoolSite::Strided,
            |exec| {
                symtn_operations::try_cat_owned_mixed_raw(
                    exec,
                    self.required_len,
                    side,
                    &self.copies,
                    dense_sources,
                    source_lengths,
                    &compact_entries,
                )
            },
        )?
        .ok_or_else(|| internal_layout_error("cat copy plan declined by the fast-path prover"))
    }
}

/// Compiles the shared structure-level copy plan of `catdomain` and
/// `catcodomain`. Owned facade storage is canonical, so both orientations map
/// monotonically onto the destination with matching tree order; a mismatch is
/// an internal layout error.
pub(crate) fn compile_cat_plan(
    destination: &Arc<BlockStructure>,
    destination_nout: usize,
    operands: [CatOperandLayout<'_>; 2],
    axis: usize,
    side: Side,
) -> Result<CatCopyPlan, Error> {
    {
        let sources = [operands[0].structure, operands[1].structure];
        let source_blocks = [
            cat_source_blocks(destination, sources[0], operands[0].orientation)?,
            cat_source_blocks(destination, sources[1], operands[1].orientation)?,
        ];
        let source_metadata = [&operands[0], &operands[1]];
        for (destination_block, (&lhs_source_block, &rhs_source_block)) in
            source_blocks[0].iter().zip(&source_blocks[1]).enumerate()
        {
            let dst = destination.block(destination_block)?;
            let source_for_destination = [lhs_source_block, rhs_source_block];
            if source_for_destination.iter().all(Option::is_none) {
                return Err(internal_layout_error(
                    "concatenated fusion-tree key has no source",
                ));
            }
            let mut destination_axis_offset = 0usize;
            for source in 0..2 {
                let Some(source_block) = source_for_destination[source] else {
                    continue;
                };
                let src = sources[source].block(source_block)?;
                if src.shape().len() != dst.shape().len()
                    || (0..dst.shape().len()).any(|logical_axis| {
                        if logical_axis == axis {
                            return false;
                        }
                        match cat_storage_axis(source_metadata[source], logical_axis) {
                            Ok(storage_axis) => {
                                src.shape()[storage_axis] != dst.shape()[logical_axis]
                            }
                            Err(_) => true,
                        }
                    })
                {
                    return Err(internal_layout_error(
                        "concatenated source and destination subblock shapes disagree",
                    ));
                }
                let storage_axis = cat_storage_axis(source_metadata[source], axis)?;
                destination_axis_offset = destination_axis_offset
                    .checked_add(src.shape()[storage_axis])
                    .ok_or_else(|| internal_layout_error("concatenated axis offset overflow"))?;
            }
            if destination_axis_offset != dst.shape()[axis] {
                return Err(internal_layout_error(
                    "concatenated source slabs do not cover the destination axis",
                ));
            }
        }
    }
    let destination_regions = sector_regions(destination, destination_nout)?;
    let source_regions = [&operands[0].regions, &operands[1].regions];
    let oriented = matches!(operands[0].orientation, TensorOrientation::Adjoint)
        || matches!(operands[1].orientation, TensorOrientation::Adjoint);
    let source_region_indices = [
        cat_source_regions(&destination_regions, source_regions[0])?,
        cat_source_regions(&destination_regions, source_regions[1])?,
    ];
    if oriented
        && !cat_region_tree_orders_match(
            &destination_regions,
            [source_regions[0], source_regions[1]],
            [&source_region_indices[0], &source_region_indices[1]],
            [operands[0].orientation, operands[1].orientation],
            side,
        )
    {
        return Err(internal_layout_error(
            "oriented cat source tree order disagrees with the destination",
        ));
    }
    let mut copies = Vec::with_capacity(source_regions[0].len() + source_regions[1].len());
    for (destination_index, destination) in destination_regions.iter().enumerate() {
        let mut changed_axis_offset = 0usize;
        for source in 0..2 {
            let Some(source_index) = source_region_indices[source][destination_index] else {
                continue;
            };
            let src = &source_regions[source][source_index];
            let (source_rows, source_cols, source_row_stride, source_column_stride) =
                match operands[source].orientation {
                    TensorOrientation::Owned => (src.rows(), src.cols(), 1, src.rows()),
                    TensorOrientation::Adjoint => (src.cols(), src.rows(), src.rows(), 1),
                };
            let (rows, cols, destination_offset) = match side {
                Side::Codomain => {
                    if source_cols != destination.cols() {
                        return Err(internal_layout_error(
                            "cat(Side::Codomain) coupled-sector columns disagree",
                        ));
                    }
                    (
                        source_rows,
                        source_cols,
                        destination
                            .range()
                            .start
                            .checked_add(changed_axis_offset)
                            .ok_or_else(|| {
                                internal_layout_error(
                                    "cat(Side::Codomain) destination row offset overflow",
                                )
                            })?,
                    )
                }
                Side::Domain => {
                    if source_rows != destination.rows() {
                        return Err(internal_layout_error(
                            "cat(Side::Domain) coupled-sector rows disagree",
                        ));
                    }
                    (
                        source_rows,
                        source_cols,
                        destination
                            .rows()
                            .checked_mul(changed_axis_offset)
                            .and_then(|offset| destination.range().start.checked_add(offset))
                            .ok_or_else(|| {
                                internal_layout_error(
                                    "cat(Side::Domain) destination column offset overflow",
                                )
                            })?,
                    )
                }
            };
            copies.push(OwnedCatCopy::new(
                source,
                src.range().start,
                destination_offset,
                [rows, cols],
                [source_row_stride, source_column_stride],
                destination.rows(),
                destination.range(),
                matches!(operands[source].orientation, TensorOrientation::Adjoint),
            ));
            changed_axis_offset = changed_axis_offset
                .checked_add(match side {
                    Side::Codomain => source_rows,
                    Side::Domain => source_cols,
                })
                .ok_or_else(|| internal_layout_error("concatenated region offset overflow"))?;
        }
        let expected = match side {
            Side::Codomain => destination.rows(),
            Side::Domain => destination.cols(),
        };
        if changed_axis_offset != expected {
            return Err(internal_layout_error(
                "concatenated source regions do not cover the destination matrix",
            ));
        }
    }
    Ok(CatCopyPlan {
        required_len: destination.required_len()?,
        copies,
        side,
    })
}

pub(super) fn cat_source_blocks(
    destination: &BlockStructure,
    source: &BlockStructure,
    orientation: TensorOrientation,
) -> Result<Vec<Option<usize>>, Error> {
    if matches!(orientation, TensorOrientation::Adjoint) {
        let mut source_for_destination = vec![None; destination.block_count()];
        for source_index in 0..source.block_count() {
            let logical_key = cat_logical_block_key(source.block(source_index)?.key())?;
            let destination_index = destination
                .find_block_index_by_key(&logical_key)
                .ok_or_else(|| {
                    internal_layout_error(
                        "source fusion-tree key is absent from concatenated destination",
                    )
                })?;
            if source_for_destination[destination_index]
                .replace(source_index)
                .is_some()
            {
                return Err(internal_layout_error(
                    "multiple source fusion-tree keys map to one destination",
                ));
            }
        }
        return Ok(source_for_destination);
    }

    let destination_sector = destination.sector_structure();
    let source_sector = source.sector_structure();
    let mut source_for_destination = vec![None; destination.block_count()];
    let mut destination_position = 0usize;
    let mut source_position = 0usize;
    while destination_position < destination_sector.sorted_indices().len()
        && source_position < source_sector.sorted_indices().len()
    {
        let destination_index = destination_sector.sorted_indices()[destination_position];
        let source_index = source_sector.sorted_indices()[source_position];
        let destination_key = destination_sector.key(destination_index)?;
        let source_key = source_sector.key(source_index)?;
        match destination_key.cmp(source_key) {
            std::cmp::Ordering::Less => destination_position += 1,
            std::cmp::Ordering::Greater => {
                return Err(internal_layout_error(
                    "source fusion-tree key is absent from concatenated destination",
                ));
            }
            std::cmp::Ordering::Equal => {
                source_for_destination[destination_index] = Some(source_index);
                destination_position += 1;
                source_position += 1;
            }
        }
    }
    if source_position != source_sector.sorted_indices().len() {
        return Err(internal_layout_error(
            "source fusion-tree key is absent from concatenated destination",
        ));
    }
    Ok(source_for_destination)
}

pub(crate) fn cat_logical_block_key(key: &BlockKey) -> Result<BlockKey, Error> {
    match key {
        BlockKey::FusionTree(key) => Ok(BlockKey::from(FusionTreePairKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ))),
        _ => Err(internal_layout_error(
            "unsupported block key in adjoint concatenation",
        )),
    }
}

pub(super) fn cat_storage_axis(
    source: &CatOperandLayout<'_>,
    logical_axis: usize,
) -> Result<usize, Error> {
    if logical_axis >= source.rank {
        return Err(internal_layout_error(
            "concatenated logical axis exceeds source rank",
        ));
    }
    match source.orientation {
        TensorOrientation::Owned => Ok(logical_axis),
        TensorOrientation::Adjoint => {
            if logical_axis < source.logical_nout {
                source
                    .storage_nout
                    .checked_add(logical_axis)
                    .ok_or_else(|| internal_layout_error("concatenated storage axis overflow"))
            } else {
                Ok(logical_axis - source.logical_nout)
            }
        }
    }
}

pub(super) fn cat_source_regions(
    destination: &[CoupledSectorRegion],
    source: &[CoupledSectorRegion],
) -> Result<Vec<Option<usize>>, Error> {
    cat_source_regions_if_monotone(destination, source).ok_or_else(|| {
        internal_layout_error("source coupled sectors do not map monotonically to destination")
    })
}

pub(super) fn cat_source_regions_if_monotone(
    destination: &[CoupledSectorRegion],
    source: &[CoupledSectorRegion],
) -> Option<Vec<Option<usize>>> {
    if destination
        .windows(2)
        .any(|pair| pair[0].coupled() >= pair[1].coupled())
        || source
            .windows(2)
            .any(|pair| pair[0].coupled() >= pair[1].coupled())
    {
        return None;
    }
    let mut source_for_destination = vec![None; destination.len()];
    let mut destination_position = 0usize;
    let mut source_position = 0usize;
    while destination_position < destination.len() && source_position < source.len() {
        match destination[destination_position]
            .coupled()
            .cmp(&source[source_position].coupled())
        {
            std::cmp::Ordering::Less => destination_position += 1,
            std::cmp::Ordering::Greater => return None,
            std::cmp::Ordering::Equal => {
                source_for_destination[destination_position] = Some(source_position);
                destination_position += 1;
                source_position += 1;
            }
        }
    }
    if source_position != source.len() {
        return None;
    }
    Some(source_for_destination)
}

pub(super) fn cat_region_tree_orders_match(
    destination: &[CoupledSectorRegion],
    sources: [&[CoupledSectorRegion]; 2],
    source_indices: [&[Option<usize>]; 2],
    orientations: [TensorOrientation; 2],
    side: Side,
) -> bool {
    destination
        .iter()
        .enumerate()
        .all(|(destination_index, destination)| {
            (0..2).all(|source| {
                let Some(source_index) = source_indices[source][destination_index] else {
                    return true;
                };
                let source_region = &sources[source][source_index];
                match (side, orientations[source]) {
                    (Side::Domain, TensorOrientation::Owned) => {
                        source_region.row_trees() == destination.row_trees()
                    }
                    (Side::Domain, TensorOrientation::Adjoint) => {
                        source_region.col_trees() == destination.row_trees()
                    }
                    (Side::Codomain, TensorOrientation::Owned) => {
                        source_region.col_trees() == destination.col_trees()
                    }
                    (Side::Codomain, TensorOrientation::Adjoint) => {
                        source_region.row_trees() == destination.col_trees()
                    }
                }
            })
        })
}

#[cfg(test)]
mod cat_plan_tests {
    use super::*;

    #[test]
    fn declined_plan_is_an_internal_layout_error() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let plan = CatCopyPlan {
            required_len: 10,
            copies: vec![
                OwnedCatCopy::new(0, 2, 4, [2, 1], [1, 2], 2, 4..10, false),
                OwnedCatCopy::new(1, 2, 6, [2, 2], [2, 1], 2, 4..10, false),
                OwnedCatCopy::new(0, 0, 0, [2, 1], [1, 2], 2, 0..4, false),
                OwnedCatCopy::new(1, 0, 2, [2, 1], [1, 2], 2, 0..4, false),
            ],
            side: Side::Domain,
        };
        let error = plan
            .execute(
                engine,
                [
                    CatOperandData::Dense(&[1.0, 2.0, 3.0, 4.0]),
                    CatOperandData::Dense(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                ],
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidArgument(message) if message.contains("declined by the fast-path prover")
        ));
    }
}

/// Shared validation-and-output-homspace core of `catdomain`/`catcodomain`:
/// checks the rank-1 changed side and the identical unchanged product space,
/// then direct-sums the changed legs through [`oplus_sector_legs`]. Rule
/// identity is checked by callers before this runs.
pub(crate) fn cat_homspace(
    lhs_codomain: &FusionProductSpace,
    lhs_domain: &FusionProductSpace,
    rhs_codomain: &FusionProductSpace,
    rhs_domain: &FusionProductSpace,
    side: Side,
) -> Result<(usize, FusionTreeHomSpace), Error> {
    match side {
        Side::Domain => {
            if lhs_domain.len() != 1 || rhs_domain.len() != 1 {
                return Err(Error::InvalidArgument(
                    "cat(Side::Domain) requires exactly one domain leg on each tensor".to_string(),
                ));
            }
            if lhs_codomain != rhs_codomain {
                return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
                    message: "cat(Side::Domain) requires identical codomain product spaces",
                }));
            }
            let leg = oplus_sector_legs(&lhs_domain.legs()[0], &rhs_domain.legs()[0])?;
            Ok((
                lhs_codomain.len(),
                FusionTreeHomSpace::new(lhs_codomain.clone(), FusionProductSpace::new([leg])),
            ))
        }
        Side::Codomain => {
            if lhs_codomain.len() != 1 || rhs_codomain.len() != 1 {
                return Err(Error::InvalidArgument(
                    "cat(Side::Codomain) requires exactly one codomain leg on each tensor"
                        .to_string(),
                ));
            }
            if lhs_domain != rhs_domain {
                return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
                    message: "cat(Side::Codomain) requires identical domain product spaces",
                }));
            }
            let leg = oplus_sector_legs(&lhs_codomain.legs()[0], &rhs_codomain.legs()[0])?;
            Ok((
                0,
                FusionTreeHomSpace::new(FusionProductSpace::new([leg]), lhs_domain.clone()),
            ))
        }
    }
}

/// How a freshly built tensor is filled.
pub(crate) enum Fill<'f, D> {
    Zeros,
    Rand(u64),
    BlockFn(&'f mut dyn FnMut(&BlockKey, &[usize]) -> D),
}

/// Keeps checked unit-leg validation failures in the public error classes used
/// by the typed facade.
pub(crate) fn map_checked_unit_layout_error(error: symtn_core::CheckedFusionSpaceError) -> Error {
    match error {
        symtn_core::CheckedFusionSpaceError::Core(error) => Error::Core(error),
        symtn_core::CheckedFusionSpaceError::FusionAlgebra(error) => Error::FusionAlgebra(error),
        other => Error::InvalidArgument(format!("unit layout correspondence failed: {other}")),
    }
}

/// Fills a freshly-built coupled space without inspecting its provider.
pub(crate) fn apply_fill<S: ScalarOps>(
    space: &DynamicFusionMapSpace,
    fill: Fill<'_, S>,
) -> Result<Vec<S>, Error> {
    let len = space.required_len()?;
    let mut data = vec![S::from_real(0.0); len];
    match fill {
        Fill::Zeros => {}
        Fill::Rand(seed) => {
            let mut state = seed;
            for value in &mut data {
                *value = S::rand_unit(&mut state);
            }
        }
        Fill::BlockFn(fill) => {
            fill_block_elements(space.structure(), &mut data, fill)?;
        }
    }
    Ok(data)
}

/// Fills every symmetry-allowed block in coupled-layout storage order.
pub(super) fn fill_block_elements<D: ScalarOps>(
    structure: &symtn_core::BlockStructure,
    data: &mut [D],
    fill: &mut dyn FnMut(&BlockKey, &[usize]) -> D,
) -> Result<(), Error> {
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        let count: usize = shape.iter().product();
        let mut indices = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = offset
                + indices
                    .iter()
                    .zip(strides)
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            data[position] = fill(block.key(), &indices);
            for axis in 0..shape.len() {
                indices[axis] += 1;
                if indices[axis] < shape[axis] {
                    break;
                }
                indices[axis] = 0;
            }
        }
    }
    Ok(())
}

/// Scales every fusion-tree block by its categorical factor.
pub(crate) fn scale_blocks_impl<D: ScalarOps, C: TwistCoefficient<D>>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    factor_of: &dyn Fn(&BlockKey) -> C,
) -> Result<(), Error> {
    let structure = space.structure();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let factor = factor_of(block.key());
        if factor == C::one() {
            continue;
        }
        scale_strided_block(
            data,
            block.shape(),
            block.strides(),
            block.offset(),
            factor.as_data(),
        )?;
    }
    Ok(())
}

pub(super) fn scale_strided_block<D: ScalarOps>(
    data: &mut [D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    factor: D,
) -> Result<(), Error> {
    if shape.contains(&0) {
        return Ok(());
    }
    let strides = shape
        .iter()
        .zip(strides)
        .map(|(&extent, &stride)| {
            if extent <= 1 {
                Ok(0)
            } else {
                isize::try_from(stride)
                    .map_err(|_| symtn_operations::OperationError::StrideOverflow { value: stride })
            }
        })
        .collect::<Result<SmallVec<[isize; 8]>, _>>()?;
    let offset = symtn_operations::strided::offset_to_isize(offset)?;
    symtn_operations::scale_raw_strided_kernel_trusted(data, shape, &strides, offset, factor)?;
    Ok(())
}

#[cfg(test)]
mod scale_strided_block_tests {
    use super::*;

    const VISITED: [usize; 8] = [2, 3, 6, 7, 12, 13, 16, 17];

    #[test]
    fn rank_three_scale_touches_only_literal_strided_addresses() {
        let mut real = vec![-99.0; 20];
        for (value, index) in VISITED.into_iter().enumerate() {
            real[index] = f64::from(value as u32 + 1);
        }
        let mut expected_real = vec![-99.0; 20];
        for (value, index) in VISITED.into_iter().enumerate() {
            expected_real[index] = -2.0 * f64::from(value as u32 + 1);
        }
        scale_strided_block(&mut real, &[2, 2, 2], &[1, 4, 10], 2, -2.0).unwrap();
        assert_eq!(real, expected_real);

        let sentinel = Complex64::new(-99.0, 99.0);
        let mut complex = vec![sentinel; 20];
        for (value, index) in VISITED.into_iter().enumerate() {
            let value = f64::from(value as u32 + 1);
            complex[index] = Complex64::new(value, -value);
        }
        let mut expected_complex = vec![sentinel; 20];
        for (value, index) in VISITED.into_iter().enumerate() {
            let value = 3.0 * f64::from(value as u32 + 1);
            expected_complex[index] = Complex64::new(value, -value);
        }
        scale_strided_block(
            &mut complex,
            &[2, 2, 2],
            &[1, 4, 10],
            2,
            Complex64::new(3.0, 0.0),
        )
        .unwrap();
        assert_eq!(complex, expected_complex);
    }

    #[test]
    fn scalar_empty_and_singleton_layouts_keep_their_existing_meaning() {
        let mut scalar = vec![11.0, 13.0, 17.0];
        scale_strided_block(&mut scalar, &[], &[], 1, -1.0).unwrap();
        assert_eq!(scalar, [11.0, -13.0, 17.0]);

        let mut empty = Vec::<f64>::new();
        scale_strided_block(
            &mut empty,
            &[0, 2],
            &[usize::MAX, usize::MAX],
            usize::MAX,
            2.0,
        )
        .unwrap();
        assert!(empty.is_empty());

        let mut singleton = vec![2.0, 3.0, 5.0, 7.0];
        scale_strided_block(&mut singleton, &[1, 2], &[usize::MAX, 2], 1, 10.0).unwrap();
        assert_eq!(singleton, [2.0, 30.0, 5.0, 70.0]);
    }
}

pub(super) fn uncoupled_sector_of_leg(
    key: &FusionTreePairKey,
    nout: usize,
    leg: usize,
) -> SectorId {
    if leg < nout {
        key.codomain_uncoupled()[leg]
    } else {
        key.domain_uncoupled()[leg - nout]
    }
}

/// θ for the twist, `θ̄` for its inverse. Why the conjugate is the inverse:
/// every admitted rule is unitary, so a twist is a phase (`|θ| = 1`), and
/// TensorKit `twist!` uses `θ'` the same way without a runtime check
/// (`src/tensors/indexmanipulations.jl:74`).
pub(crate) fn twist_factor_with_inverse<C: CategoricalScalar>(factor: C, inverse: bool) -> C {
    if inverse {
        factor.conj()
    } else {
        factor
    }
}

pub(crate) fn twist_is_identity_over_blocks<R>(
    rule: &R,
    structure: &symtn_core::BlockStructure,
    nout: usize,
    legs: &[usize],
) -> Result<bool, Error>
where
    R: MultiplicityFreeRigidSymbols + ?Sized,
    R::Scalar: PartialEq,
{
    if rule.braiding_style().is_bosonic() {
        return Ok(true);
    }
    let one = <R::Scalar as CategoricalScalar>::one();
    (0..structure.block_count()).try_fold(true, |noop, index| {
        let block = structure.block(index)?;
        Ok::<_, Error>(
            noop && match block.key() {
                BlockKey::FusionTree(key) => legs
                    .iter()
                    .all(|&leg| rule.twist_scalar(uncoupled_sector_of_leg(key, nout, leg)) == one),
                _ => true,
            },
        )
    })
}

/// TensorKit-compatible Z-isomorphism phase for distinct flipped legs,
/// reading each uncoupled sector's (χ, θ) from `values`.
pub(crate) fn flip_block_factor<C: CategoricalScalar + Copy>(
    values: &impl Fn(SectorId) -> (C, C),
    key: &FusionTreePairKey,
    nout: usize,
    occurrences: &[(usize, bool)],
    inverse: bool,
) -> C {
    occurrences
        .iter()
        .map(|&(leg, dual)| {
            let (chi, theta) = values(uncoupled_sector_of_leg(key, nout, leg));
            if leg < nout {
                if dual {
                    if inverse {
                        C::one()
                    } else {
                        chi * theta
                    }
                } else if inverse {
                    (chi * theta).conj()
                } else {
                    C::one()
                }
            } else if dual {
                if inverse {
                    theta.conj()
                } else {
                    chi.conj()
                }
            } else if inverse {
                chi
            } else {
                theta
            }
        })
        .fold(C::one(), |product, factor| product * factor)
}

/// Returns the duality-toggled HomSpace and each distinct leg's stored duality.
pub(crate) fn flip_toggled_homspace(
    hom: &FusionTreeHomSpace,
    legs: &[usize],
) -> (FusionTreeHomSpace, Vec<(usize, bool)>) {
    let nout = hom.codomain().len();
    let leg_of = |leg: usize| {
        if leg < nout {
            &hom.codomain().legs()[leg]
        } else {
            &hom.domain().legs()[leg - nout]
        }
    };
    let occurrences = legs
        .iter()
        .map(|&leg| (leg, leg_of(leg).is_dual()))
        .collect();

    let toggled_leg = |index: usize, leg: &SectorLeg| {
        if legs.contains(&index) {
            SectorLeg::new(leg.iter(), !leg.is_dual())
        } else {
            leg.clone()
        }
    };
    let toggled = FusionTreeHomSpace::new(
        FusionProductSpace::new(
            hom.codomain()
                .legs()
                .iter()
                .enumerate()
                .map(|(index, leg)| toggled_leg(index, leg)),
        ),
        FusionProductSpace::new(
            hom.domain()
                .legs()
                .iter()
                .enumerate()
                .map(|(index, leg)| toggled_leg(nout + index, leg)),
        ),
    );
    (toggled, occurrences)
}

pub(crate) fn check_flip_layout_identity(
    old_structure: &symtn_core::BlockStructure,
    new_structure: &symtn_core::BlockStructure,
) -> Result<(), Error> {
    if new_structure.block_count() != old_structure.block_count() {
        return Err(internal_layout_error("flip changed the block count"));
    }
    if new_structure.required_len()? != old_structure.required_len()? {
        return Err(internal_layout_error(
            "flip changed the required payload length",
        ));
    }
    for index in 0..old_structure.block_count() {
        let old_block = old_structure.block(index)?;
        let new_block = new_structure.block(index)?;
        let same_key = match (
            old_block.key().as_fusion_tree_pair(),
            new_block.key().as_fusion_tree_pair(),
        ) {
            (Some(old), Some(new)) => {
                old.codomain_uncoupled() == new.codomain_uncoupled()
                    && old.domain_uncoupled() == new.domain_uncoupled()
                    && old.coupled() == new.coupled()
                    && old.codomain_innerlines() == new.codomain_innerlines()
                    && old.domain_innerlines() == new.domain_innerlines()
                    && old.codomain_vertices() == new.codomain_vertices()
                    && old.domain_vertices() == new.domain_vertices()
            }
            (None, None) => old_block.key() == new_block.key(),
            _ => false,
        };
        if !same_key
            || old_block.shape() != new_block.shape()
            || old_block.strides() != new_block.strides()
            || old_block.offset() != new_block.offset()
        {
            return Err(internal_layout_error("flip changed the block layout"));
        }
    }
    Ok(())
}

pub(crate) enum PlanarRequestKind<'a> {
    Explicit {
        codomain_axes: &'a [usize],
        domain_axes: &'a [usize],
    },
    Repartition {
        num_codomain: usize,
    },
}

pub(crate) fn with_planar_axes<T>(
    source_codomain_rank: usize,
    source_rank: usize,
    kind: PlanarRequestKind<'_>,
    apply: impl FnOnce(&[usize], &[usize]) -> Result<T, Error>,
) -> Result<T, Error> {
    let source_domain_rank = source_rank - source_codomain_rank;
    let checked_apply = |codomain_axes: &[usize], domain_axes: &[usize]| {
        PreparedTreePairOperation::validate_transpose_syntax(
            source_codomain_rank,
            source_domain_rank,
            codomain_axes,
            domain_axes,
        )?;
        apply(codomain_axes, domain_axes)
    };
    match kind {
        PlanarRequestKind::Explicit {
            codomain_axes,
            domain_axes,
        } => checked_apply(codomain_axes, domain_axes),
        PlanarRequestKind::Repartition { num_codomain } => {
            if num_codomain > source_rank {
                return Err(Error::InvalidArgument(format!(
                    "repartition: num_codomain {num_codomain} exceeds rank {source_rank}",
                )));
            }
            let mut axes = (0..source_codomain_rank)
                .chain((source_codomain_rank..source_rank).rev())
                .collect::<Vec<_>>();
            axes[num_codomain..].reverse();
            let (codomain_axes, domain_axes) = axes.split_at(num_codomain);
            checked_apply(codomain_axes, domain_axes)
        }
    }
}

pub(super) fn logical_adjoint_axis_to_parent(
    parent_codomain_rank: usize,
    parent_domain_rank: usize,
    axis: usize,
) -> usize {
    debug_assert!(axis < parent_codomain_rank + parent_domain_rank);
    if axis < parent_domain_rank {
        parent_codomain_rank + axis
    } else {
        axis - parent_domain_rank
    }
}

pub(crate) fn logical_adjoint_axes_to_parent(
    parent_codomain_rank: usize,
    parent_domain_rank: usize,
    axes: &[usize],
) -> Vec<usize> {
    axes.iter()
        .map(|&axis| logical_adjoint_axis_to_parent(parent_codomain_rank, parent_domain_rank, axis))
        .collect()
}

pub(crate) fn lower_adjoint_tree_transform_operation(
    parent_codomain_rank: usize,
    parent_domain_rank: usize,
    operation: &TreeTransformOperation,
) -> Result<TreeTransformOperation, Error> {
    let rank = parent_codomain_rank
        .checked_add(parent_domain_rank)
        .ok_or_else(|| Error::InvalidArgument("tensor rank overflow".to_string()))?;
    let logical_axes = operation
        .codomain_permutation()
        .iter()
        .chain(operation.domain_permutation())
        .copied()
        .collect::<Vec<_>>();
    validate_axis_permutation(&logical_axes, rank)?;

    let codomain_axes = operation
        .domain_permutation()
        .iter()
        .copied()
        .map(|axis| logical_adjoint_axis_to_parent(parent_codomain_rank, parent_domain_rank, axis))
        .collect::<Vec<_>>();
    let domain_axes = operation
        .codomain_permutation()
        .iter()
        .copied()
        .map(|axis| logical_adjoint_axis_to_parent(parent_codomain_rank, parent_domain_rank, axis))
        .collect::<Vec<_>>();
    Ok(match operation.kind() {
        TreeTransformOperationKind::Permute => {
            TreeTransformOperation::permute(codomain_axes, domain_axes)
        }
        TreeTransformOperationKind::Transpose => {
            TreeTransformOperation::transpose(codomain_axes, domain_axes)
        }
        TreeTransformOperationKind::Braid => {
            let level_count = operation
                .codomain_levels()
                .len()
                .checked_add(operation.domain_levels().len())
                .ok_or_else(|| Error::InvalidArgument("braid level count overflow".to_string()))?;
            if level_count != rank {
                return Err(Error::InvalidArgument(format!(
                    "braid levels must list one level per source axis \
                     (expected {rank}, got {level_count})",
                )));
            }
            // Why not reflect the level values: the outer adjoint conjugates
            // coefficients; TensorKit only reindexes levels into parent order.
            TreeTransformOperation::braid(
                codomain_axes,
                domain_axes,
                operation.domain_levels().iter().copied(),
                operation.codomain_levels().iter().copied(),
            )
        }
    })
}

pub(crate) fn validate_axis_permutation(axes: &[usize], rank: usize) -> Result<(), Error> {
    symtn_core::axes::validate_permutation(axes, rank).map_err(|_| {
        symtn_tensors::OperationError::InvalidPermutation {
            axes: axes.to_vec(),
            rank,
        }
        .into()
    })
}

/// Direct-sums two sector legs by adding matching degeneracies.
pub(crate) fn oplus_sector_legs(lhs: &SectorLeg, rhs: &SectorLeg) -> Result<SectorLeg, Error> {
    if lhs.is_dual() != rhs.is_dual() {
        return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
            message: "oplus: cannot direct-sum spaces of opposite duality (dualize one first)",
        }));
    }
    let mut sectors: Vec<(SectorId, usize)> = lhs.iter().collect();
    for (sector, deg) in rhs.iter() {
        match sectors.iter_mut().find(|(s, _)| *s == sector) {
            Some(entry) => {
                entry.1 = entry.1.checked_add(deg).ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "oplus: degeneracy overflow for sector {sector:?}"
                    ))
                })?;
            }
            None => sectors.push((sector, deg)),
        }
    }
    sectors.retain(|&(_, deg)| deg > 0);
    sectors.sort_by_key(|(sector, _)| *sector);
    Ok(SectorLeg::new(sectors, lhs.is_dual()))
}

/// Fuses sector content, including fusion multiplicities, in sector-id order.
pub(super) fn fuse_sector_content<E>(
    left: &[(SectorId, usize)],
    right: &[(SectorId, usize)],
    channels: impl Fn(SectorId, SectorId) -> Result<symtn_core::SectorVec, E>,
    nsymbol: impl Fn(SectorId, SectorId, SectorId) -> Result<usize, E>,
) -> Result<Vec<(SectorId, usize)>, E> {
    let mut out = std::collections::BTreeMap::<SectorId, usize>::new();
    for &(a, deg_a) in left {
        for &(b, deg_b) in right {
            for c in channels(a, b)? {
                *out.entry(c).or_insert(0) += nsymbol(a, b, c)? * deg_a * deg_b;
            }
        }
    }
    Ok(out.into_iter().collect())
}
