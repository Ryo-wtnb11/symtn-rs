use core::ops::{Add, Mul, Range};

use num_traits::{One, Zero};
use smallvec::SmallVec;
use symtn_core::{BlockKey, BlockRef, BlockStructure, FusionTreePairKey, SectorId};
use symtn_operations::{
    bilinear_raw_strided_kernel_mapped, check_axis_runs, for_each_run_piece,
    overwrite_owned_blocks, overwrite_owned_member_blocks, AxisRuns, BlockOverwrite,
    ConjugateValue, OperationError, RecouplingCoefficientAction, StridedHostKernelAdapter,
    WideScalar,
};

use crate::FusionOperand;

fn checked_offset(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

fn validate_logical_block_shape(
    operand: FusionOperand<'_>,
    logical_shape: &[usize],
    storage_shape: &[usize],
) -> Result<(), OperationError> {
    if logical_shape.len() != storage_shape.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented elementwise block",
        });
    }
    for (axis, &extent) in logical_shape.iter().enumerate() {
        if storage_shape[operand.storage_axis(axis)?] != extent {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented elementwise block",
            });
        }
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_oriented_fusion_layout(
    logical: &BlockStructure,
    operand: FusionOperand<'_>,
) -> Result<(), OperationError> {
    if logical.block_count() != operand.storage_space().structure().block_count() {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented elementwise layout",
        });
    }
    for logical_index in 0..logical.block_count() {
        let logical_block = logical.block(logical_index)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented elementwise logical layout",
            });
        };
        let storage_block = operand
            .storage_space()
            .structure()
            .block(operand.storage_block_index(logical_key)?)?;
        validate_logical_block_shape(operand, logical_block.shape(), storage_block.shape())?;
    }
    Ok(())
}

/// The uncoupled sector a fusion-tree pair key carries on one logical axis.
///
/// Codomain axes index the codomain tree, the remaining axes the domain tree,
/// which is the same split the block shape uses.
fn logical_axis_sector(key: &FusionTreePairKey, axis: usize) -> Result<SectorId, OperationError> {
    let codomain = key.codomain_uncoupled();
    if let Some(&sector) = codomain.get(axis) {
        return Ok(sector);
    }
    key.domain_uncoupled()
        .get(axis - codomain.len())
        .copied()
        .ok_or_else(|| OperationError::StructureMismatch {
            tensor: "oriented elementwise axis sector",
        })
}

/// Looks a block's sector up in a per-axis selection table.
///
/// Tables are sorted by [`SectorId`]; an unsorted table can only make this
/// miss, which every caller turns into `StructureMismatch`, never into a
/// different sector's entry.
fn selected_for_sector<T>(table: &[(SectorId, T)], sector: SectorId) -> Option<&T> {
    table
        .binary_search_by_key(&sector, |(candidate, _)| *candidate)
        .ok()
        .map(|index| &table[index].1)
}

/// The kept degeneracy positions of one sector, as sorted, disjoint,
/// nonempty runs. A contiguous selection is one run, held inline.
#[doc(hidden)]
pub type SelectedRuns = SmallVec<[Range<usize>; 1]>;

/// One axis' selected positions per sector, keyed by the sector a block
/// carries on that axis and sorted by [`SectorId`], or `None` when the axis
/// is not selected (every position, in order).
///
/// The positions are on the parent leg's side: the source of a restriction,
/// the destination of a scatter. The other side holds them consecutively.
#[doc(hidden)]
pub type SectorRunTable<'a> = Option<&'a [(SectorId, SelectedRuns)]>;

/// One block's per-axis runs out of the per-axis tables: `None` for an
/// unselected axis, the runs of the block's own sector otherwise. A block
/// whose sector is absent from a `Some` table is a `StructureMismatch`
/// rather than an implicit whole axis, so a caller that forgot a sector
/// cannot get a silently misaligned copy. The selected positions (all of
/// `0..packed[axis]` for `None`) must end within `sparse_extent(axis)`, the
/// parent-side extent; their count against `packed` is checked by the
/// writer that consumes them.
fn block_axis_runs<'a>(
    key: &FusionTreePairKey,
    packed: &[usize],
    tables: &[SectorRunTable<'a>],
    mut sparse_extent: impl FnMut(usize) -> Result<usize, OperationError>,
    tensor: &'static str,
) -> Result<SmallVec<[AxisRuns<'a>; 8]>, OperationError> {
    if packed.len() != tables.len() {
        return Err(OperationError::StructureMismatch { tensor });
    }
    let mut runs = SmallVec::with_capacity(tables.len());
    for (axis, (&extent, table)) in packed.iter().zip(tables).enumerate() {
        let selected = match table {
            None => None,
            Some(table) => Some(
                selected_for_sector(table, logical_axis_sector(key, axis)?)
                    .ok_or(OperationError::StructureMismatch { tensor })?
                    .as_slice(),
            ),
        };
        let end = match selected {
            None => extent,
            Some(selected) => selected.last().map_or(0, |run| run.end),
        };
        if end > sparse_extent(axis)? {
            return Err(OperationError::StructureMismatch { tensor });
        }
        runs.push(selected);
    }
    Ok(runs)
}

/// Gathers the selected degeneracy positions of every block from an owned
/// or lazy-adjoint fusion tensor into a newly allocated `destination`
/// payload, in order.
///
/// `logical_runs` holds one entry per logical axis: `None` keeps the axis
/// whole, `Some(table)` gives the source positions per uncoupled sector of
/// the destination block's own key ([`SectorRunTable`]). The runs must end
/// within the source extent and cover the destination block's extent
/// exactly, both checked per block and axis before the block is written.
#[doc(hidden)]
pub fn oriented_fusion_restrict_owned<D>(
    destination: &BlockStructure,
    source: FusionOperand<'_>,
    source_data: &[D],
    logical_runs: &[SectorRunTable<'_>],
) -> Result<Vec<D>, OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if source_data.len() != source.storage_space().required_len()?
        || logical_runs.len() != destination.rank()
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented degeneracy restriction storage",
        });
    }
    overwrite_owned_blocks(destination, |destination_block, writer| {
        restrict_block(destination_block, writer, source, source_data, logical_runs)
    })
}

/// [`oriented_fusion_restrict_owned`] for `members` source payloads of one
/// structure stacked end to end in `source_data`: the output stacks the
/// `members` restrictions at stride `destination.required_len()`.
///
/// Each piece of a destination block is one strided copy for every member at
/// once (the member axis is a trailing dimension of the copy), from the same
/// per-block plan as the single-payload form.
#[doc(hidden)]
pub fn stacked_fusion_restrict_owned<D>(
    destination: &BlockStructure,
    source: FusionOperand<'_>,
    source_data: &[D],
    members: usize,
    logical_runs: &[SectorRunTable<'_>],
) -> Result<Vec<D>, OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    let member_len = source.storage_space().required_len()?;
    if Some(source_data.len()) != member_len.checked_mul(members)
        || logical_runs.len() != destination.rank()
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented degeneracy restriction storage",
        });
    }
    overwrite_owned_member_blocks(
        destination,
        members,
        member_len,
        |destination_block, writer| {
            restrict_block(destination_block, writer, source, source_data, logical_runs)
        },
    )
}

/// Writes one destination block of a restriction: the source block of the
/// same key, read at the selected positions on every axis.
fn restrict_block<D>(
    destination_block: BlockRef<'_>,
    writer: &mut BlockOverwrite<'_, D>,
    source: FusionOperand<'_>,
    source_data: &[D],
    logical_runs: &[SectorRunTable<'_>],
) -> Result<(), OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    let BlockKey::FusionTree(logical_key) = destination_block.key() else {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented degeneracy restriction destination",
        });
    };
    let source_block = source
        .storage_space()
        .structure()
        .block(source.storage_block_index(logical_key)?)?;
    let runs = block_axis_runs(
        logical_key,
        destination_block.shape(),
        logical_runs,
        |axis| Ok(source_block.shape()[source.storage_axis(axis)?]),
        "oriented degeneracy restriction rectangle",
    )?;
    writer.copy_runs(
        |axis| Ok(source_block.strides()[source.storage_axis(axis)?]),
        source_data,
        source_block.offset(),
        source.storage_conjugate(),
        D::one(),
        &runs,
    )
}

struct ScatterBlock {
    shape: Vec<usize>,
    destination_strides: Vec<isize>,
    source_strides: Vec<isize>,
    destination_offset: isize,
    source_offset: isize,
}

fn preflight_scatter_bounds(
    len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<(), OperationError> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| OperationError::ElementCountOverflow)
    })?;
    if shape.contains(&0) {
        return Ok(());
    }
    let maximum = shape
        .iter()
        .zip(strides)
        .try_fold(offset, |maximum, (&extent, &stride)| {
            let steps =
                isize::try_from(extent - 1).map_err(|_| OperationError::ElementCountOverflow)?;
            maximum
                .checked_add(
                    stride
                        .checked_mul(steps)
                        .ok_or_else(|| OperationError::ElementCountOverflow)?,
                )
                .ok_or_else(|| OperationError::ElementCountOverflow)
        })?;
    let maximum = usize::try_from(maximum)
        .map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })?;
    if maximum >= len {
        return Err(OperationError::OffsetOverflow { value: maximum });
    }
    Ok(())
}

/// One block of a scatter: one stride record for the block and its per-axis
/// destination runs; the pieces are walked from the runs at write time, so a
/// `k`-run selection costs no allocation per piece.
struct ScatterRunsBlock<'a> {
    block: ScatterBlock,
    runs: SmallVec<[AxisRuns<'a>; 8]>,
}

/// `origin + sum_a position[a] * stride[a]` in signed offsets, checked.
fn piece_offset(origin: isize, position: &[usize], strides: &[isize]) -> Option<isize> {
    position
        .iter()
        .zip(strides)
        .try_fold(origin, |offset, (&position, &stride)| {
            isize::try_from(position)
                .ok()?
                .checked_mul(stride)
                .and_then(|step| offset.checked_add(step))
        })
}

/// Adds source fusion-tree blocks into the selected degeneracy positions of
/// a full destination, in order: the adjoint of
/// [`oriented_fusion_restrict_owned`].
///
/// All keys, ranks, shapes, runs, strides, offsets, and storage bounds are
/// preflighted for every block before the first destination element is changed.
///
/// `runs` holds one entry per logical axis: `None` requires the source and
/// destination extents to be equal, `Some(table)` gives the destination
/// positions per uncoupled sector of the source block's own key
/// ([`SectorRunTable`]); they must cover the source extent exactly and end
/// within the destination extent. A block whose sector is absent from a
/// `Some` table is a preflight `StructureMismatch`. Accumulation is
/// `beta = 1`. One strided add per piece of the per-axis runs' product.
#[doc(hidden)]
pub fn fusion_scatter_add_assign<D>(
    mut exec: symtn_operations::host_pool::Exec<'_>,
    destination: &BlockStructure,
    destination_data: &mut [D],
    logical_source: &BlockStructure,
    source: FusionOperand<'_>,
    source_data: &[D],
    runs: &[SectorRunTable<'_>],
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if destination_data.len() != destination.required_len()?
        || source_data.len() != source.storage_space().required_len()?
        || destination.rank() != logical_source.rank()
        || runs.len() != destination.rank()
    {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion scatter accumulation storage",
        });
    }
    let mut blocks = Vec::with_capacity(logical_source.block_count());
    for source_index in 0..logical_source.block_count() {
        let logical_block = logical_source.block(source_index)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion scatter accumulation source key",
            });
        };
        let destination_index = destination
            .find_block_index_by_key(logical_block.key())
            .ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(logical_block.key().clone()),
            })?;
        let destination_block = destination.block(destination_index)?;
        let storage_block = source
            .storage_space()
            .structure()
            .block(source.storage_block_index(logical_key)?)?;
        validate_logical_block_shape(source, logical_block.shape(), storage_block.shape())?;
        if logical_block.shape().len() != destination.rank()
            || destination_block.shape().len() != destination.rank()
        {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion scatter accumulation block rank",
            });
        }
        let block_runs = block_axis_runs(
            logical_key,
            logical_block.shape(),
            runs,
            |axis| Ok(destination_block.shape()[axis]),
            "fusion scatter sliced extent",
        )?;
        for (axis, selected) in block_runs.iter().enumerate() {
            if selected.is_none() && logical_block.shape()[axis] != destination_block.shape()[axis]
            {
                return Err(OperationError::StructureMismatch {
                    tensor: "fusion scatter unsliced extent",
                });
            }
        }
        check_axis_runs(logical_block.shape(), &block_runs).map_err(|_| {
            OperationError::StructureMismatch {
                tensor: "fusion scatter sliced extent",
            }
        })?;
        let destination_strides = destination_block
            .strides()
            .iter()
            .map(|&stride| {
                isize::try_from(stride)
                    .map_err(|_| OperationError::StrideOverflow { value: stride })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_strides = (0..destination.rank())
            .map(|axis| {
                let stride = storage_block.strides()[source.storage_axis(axis)?];
                isize::try_from(stride)
                    .map_err(|_| OperationError::StrideOverflow { value: stride })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let destination_offset = checked_offset(destination_block.offset())?;
        let source_offset = checked_offset(storage_block.offset())?;
        // Strides are non-negative, so the pieces' destination envelope is
        // the block origin up to every axis' last selected position.
        let envelope: SmallVec<[usize; 8]> = block_runs
            .iter()
            .zip(logical_block.shape())
            .map(|(selected, &extent)| match selected {
                None => extent,
                Some(selected) => selected.last().map_or(0, |run| run.end),
            })
            .collect();
        preflight_scatter_bounds(
            destination_data.len(),
            &envelope,
            &destination_strides,
            destination_offset,
        )?;
        preflight_scatter_bounds(
            source_data.len(),
            logical_block.shape(),
            &source_strides,
            source_offset,
        )?;
        blocks.push(ScatterRunsBlock {
            block: ScatterBlock {
                shape: logical_block.shape().to_vec(),
                destination_strides,
                source_strides,
                destination_offset,
                source_offset,
            },
            runs: block_runs,
        });
    }
    let mut kernels = StridedHostKernelAdapter::default();
    for ScatterRunsBlock { block, runs } in &blocks {
        for_each_run_piece(&block.shape, runs, |piece, packed, sparse| {
            // Within the preflighted envelopes, so neither can overflow.
            let (Some(destination_offset), Some(source_offset)) = (
                piece_offset(block.destination_offset, sparse, &block.destination_strides),
                piece_offset(block.source_offset, packed, &block.source_strides),
            ) else {
                return Err(OperationError::ElementCountOverflow);
            };
            kernels.tensoradd_strided_checked(
                exec.reborrow(),
                destination_data,
                source_data,
                piece,
                &block.destination_strides,
                &block.source_strides,
                destination_offset,
                source_offset,
                source.storage_conjugate(),
                D::one(),
                D::one(),
            )
        })?;
    }
    Ok(())
}

/// `alpha * op(lhs) + beta * op(rhs)` per logical block of `destination`,
/// from owned or lazy-adjoint operands, into a newly allocated payload.
#[doc(hidden)]
pub fn oriented_fusion_add_owned<D>(
    destination: &BlockStructure,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<Vec<D>, OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if lhs_data.len() != lhs.storage_space().required_len()?
        || rhs_data.len() != rhs.storage_space().required_len()?
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented elementwise storage",
        });
    }
    validate_oriented_fusion_layout(destination, lhs)?;
    validate_oriented_fusion_layout(destination, rhs)?;
    overwrite_owned_blocks(destination, |destination_block, writer| {
        let BlockKey::FusionTree(logical_key) = destination_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented elementwise destination",
            });
        };
        let lhs_block = lhs
            .storage_space()
            .structure()
            .block(lhs.storage_block_index(logical_key)?)?;
        let rhs_block = rhs
            .storage_space()
            .structure()
            .block(rhs.storage_block_index(logical_key)?)?;
        writer.add(
            |axis| {
                Ok((
                    lhs_block.strides()[lhs.storage_axis(axis)?],
                    rhs_block.strides()[rhs.storage_axis(axis)?],
                ))
            },
            (lhs_data, lhs_block.offset(), lhs.storage_conjugate()),
            (rhs_data, rhs_block.offset(), rhs.storage_conjugate()),
            alpha,
            beta,
        )
    })
}

/// `destination = alpha * op(source) + beta * destination` per logical block,
/// in place, for an owned or lazy-adjoint `source` on the destination's own
/// logical space: BLAS `axpby` / VectorInterface `add!(y, x, α, β)`.
///
/// Every block is validated before the first element is written. Each block
/// is one strided pass carrying both coefficients, so `beta` is applied once
/// per element; `beta == 0` assigns without reading the destination. A zero
/// `alpha` does not read the source: VectorInterface's `scale(x, 0) =
/// zero(x)`, the destination becoming `beta * destination`.
#[doc(hidden)]
pub fn oriented_fusion_axpby_into<D>(
    mut exec: symtn_operations::host_pool::Exec<'_>,
    destination: &BlockStructure,
    destination_data: &mut [D],
    source: FusionOperand<'_>,
    source_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if destination_data.len() != destination.required_len()?
        || source_data.len() != source.storage_space().required_len()?
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented axpby storage",
        });
    }
    validate_oriented_fusion_layout(destination, source)?;
    let mut blocks = Vec::with_capacity(destination.block_count());
    for index in 0..destination.block_count() {
        let destination_block = destination.block(index)?;
        let BlockKey::FusionTree(logical_key) = destination_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented axpby destination",
            });
        };
        let storage_block = source
            .storage_space()
            .structure()
            .block(source.storage_block_index(logical_key)?)?;
        let signed = |stride: usize| {
            isize::try_from(stride).map_err(|_| OperationError::StrideOverflow { value: stride })
        };
        let destination_strides = destination_block
            .strides()
            .iter()
            .map(|&stride| signed(stride))
            .collect::<Result<Vec<_>, _>>()?;
        let source_strides = (0..destination_block.shape().len())
            .map(|axis| signed(storage_block.strides()[source.storage_axis(axis)?]))
            .collect::<Result<Vec<_>, _>>()?;
        validate_logical_block_shape(source, destination_block.shape(), storage_block.shape())?;
        let destination_offset = checked_offset(destination_block.offset())?;
        let source_offset = checked_offset(storage_block.offset())?;
        preflight_scatter_bounds(
            destination_data.len(),
            destination_block.shape(),
            &destination_strides,
            destination_offset,
        )?;
        preflight_scatter_bounds(
            source_data.len(),
            destination_block.shape(),
            &source_strides,
            source_offset,
        )?;
        blocks.push(ScatterBlock {
            shape: destination_block.shape().to_vec(),
            destination_strides,
            source_strides,
            destination_offset,
            source_offset,
        });
    }
    let mut kernels = StridedHostKernelAdapter::default();
    for block in &blocks {
        if alpha.is_zero() {
            symtn_operations::scale_raw_strided_kernel_trusted(
                destination_data,
                &block.shape,
                &block.destination_strides,
                block.destination_offset,
                beta,
            )?;
            continue;
        }
        kernels.tensoradd_strided_checked(
            exec.reborrow(),
            destination_data,
            source_data,
            &block.shape,
            &block.destination_strides,
            &block.source_strides,
            block.destination_offset,
            block.source_offset,
            source.storage_conjugate(),
            alpha,
            beta,
        )?;
    }
    Ok(())
}

/// Quantum-dimension-weighted oriented inner product.
///
/// `sector_weight` supplies `dim(c)` per logical block, as the provider's own
/// `f64` scalar: both the per-block sums and the weighted total accumulate in
/// [`WideScalar::Wide`], so a single-precision payload neither narrows the
/// weight nor sums the products in single precision. The result is narrowed to
/// the payload type once, at the end. For `f64`/`Complex64` `Wide` is the
/// payload type and every step is the arithmetic this function always did.
#[doc(hidden)]
pub fn oriented_fusion_inner<D>(
    logical: &BlockStructure,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    mut sector_weight: impl FnMut(SectorId) -> f64,
) -> Result<D, OperationError>
where
    D: WideScalar,
{
    oriented_fusion_inner_with(logical, lhs, lhs_data, rhs, rhs_data, |sector| {
        Ok::<_, OperationError>(sector_weight(sector))
    })
}

/// Fallible variant of [`oriented_fusion_inner`] for checked providers.
#[doc(hidden)]
pub fn oriented_fusion_inner_with<D, E>(
    logical: &BlockStructure,
    lhs: FusionOperand<'_>,
    lhs_data: &[D],
    rhs: FusionOperand<'_>,
    rhs_data: &[D],
    mut sector_weight: impl FnMut(SectorId) -> Result<f64, E>,
) -> Result<D, E>
where
    D: WideScalar,
    E: From<OperationError>,
{
    if lhs_data.len()
        != lhs
            .storage_space()
            .required_len()
            .map_err(OperationError::from)?
        || rhs_data.len()
            != rhs
                .storage_space()
                .required_len()
                .map_err(OperationError::from)?
    {
        return Err(OperationError::StructureMismatch {
            tensor: "oriented inner storage",
        }
        .into());
    }
    validate_oriented_fusion_layout(logical, lhs)?;
    validate_oriented_fusion_layout(logical, rhs)?;
    let mut total = D::Wide::zero();
    for logical_index in 0..logical.block_count() {
        let logical_block = logical.block(logical_index).map_err(OperationError::from)?;
        let BlockKey::FusionTree(logical_key) = logical_block.key() else {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented inner logical layout",
            }
            .into());
        };
        let lhs_block = lhs
            .storage_space()
            .structure()
            .block(lhs.storage_block_index(logical_key)?)
            .map_err(OperationError::from)?;
        let rhs_block = rhs
            .storage_space()
            .structure()
            .block(rhs.storage_block_index(logical_key)?)
            .map_err(OperationError::from)?;
        let lhs_stride = |axis| {
            let storage_axis = lhs.storage_axis(axis)?;
            isize::try_from(lhs_block.strides()[storage_axis])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let rhs_stride = |axis| {
            let storage_axis = rhs.storage_axis(axis)?;
            isize::try_from(rhs_block.strides()[storage_axis])
                .map_err(|_| OperationError::ElementCountOverflow)
        };
        let partial = bilinear_raw_strided_kernel_mapped(
            lhs_data,
            rhs_data,
            logical_block.shape(),
            lhs_stride,
            rhs_stride,
            checked_offset(lhs_block.offset())?,
            checked_offset(rhs_block.offset())?,
            !lhs.storage_conjugate(),
            rhs.storage_conjugate(),
        )?;
        // `coefficient_as_data` then `*`, not `scale_by_coefficient`: for a
        // complex accumulator the former is `partial * (w + 0i)`, the exact
        // expression this loop used before the accumulator was widened, so
        // `Complex64` keeps its results bit for bit — including the signed
        // zeros and non-finite cases where the two forms differ.
        let weight = <D::Wide as RecouplingCoefficientAction<f64>>::coefficient_as_data(
            sector_weight(logical_key.codomain_tree().coupled())?,
        );
        total = total + partial * weight;
    }
    Ok(D::narrow(total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DynamicFusionMapSpace;
    use num_complex::Complex64;
    use symtn_core::{
        BlockStructure, FusionProductSpace, FusionTreeHomSpace, SectorId, SectorLeg, Z2FusionRule,
    };

    /// A rectangular Z2 parent, its logical adjoint space and a direct
    /// payload on that logical space; the parent payload is the canonical
    /// `[2, 3]` column-major block.
    fn adjoint_fixture() -> (
        DynamicFusionMapSpace,
        DynamicFusionMapSpace,
        Vec<Complex64>,
        Vec<Complex64>,
    ) {
        let rule = Z2FusionRule;
        let vacuum = SectorId::new(0);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(vacuum, 3)], false)]),
        );
        let canonical = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
        let (logical, _) =
            crate::adjoint::adjoint_dyn(&rule, &canonical, &[Complex64::zero(); 6]).unwrap();
        let direct = (0..6)
            .map(|index| Complex64::new(index as f64 + 1.0, 0.5 - index as f64))
            .collect();
        let mut parent = vec![Complex64::zero(); 6];
        for column in 0..3 {
            for row in 0..2 {
                parent[row + 2 * column] =
                    Complex64::new((row + 2 * column) as f64, row as f64 - column as f64);
            }
        }
        (logical, canonical, direct, parent)
    }

    /// Both Z2 sectors sliced to the same destination range, sorted by id, as
    /// every scatter caller passes it.
    fn runs(runs: &[Range<usize>]) -> SelectedRuns {
        runs.iter().cloned().collect()
    }

    fn sliced_both_sectors() -> [(SectorId, SelectedRuns); 2] {
        [
            (SectorId::new(0), SelectedRuns::from_elem(1..2, 1)),
            (SectorId::new(1), SelectedRuns::from_elem(1..2, 1)),
        ]
    }

    fn two_key_destination() -> BlockStructure {
        let rule = Z2FusionRule;
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(even, 3), (odd, 3)], false)]),
            FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
        );
        DynamicFusionMapSpace::from_final_homspace(&rule, homspace)
            .unwrap()
            .structure()
            .as_ref()
            .clone()
    }

    /// The canonical single-leg Z2 space whose even and odd blocks have
    /// `shapes`.
    fn source_with_shapes(shapes: &[Vec<usize>]) -> crate::DynamicFusionMapSpace {
        let rule = Z2FusionRule;
        let even = SectorId::new(0);
        let odd = SectorId::new(1);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(even, shapes[0][0]), (odd, shapes[1][0])],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(even, shapes[0][1]), (odd, shapes[1][1])],
                false,
            )]),
        );
        DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap()
    }

    #[test]
    fn fusion_scatter_adds_nonprefix_rectangles_for_exact_tree_keys() {
        let destination = two_key_destination();
        let source = source_with_shapes(&[vec![1, 2], vec![1, 2]]);
        let mut output = vec![0.0; destination.required_len().unwrap()];
        fusion_scatter_add_assign(
            strided_kernel::Exec::serial(),
            &destination,
            &mut output,
            source.structure(),
            FusionOperand::direct(&source),
            &[1.0, 2.0, 3.0, 4.0],
            &[Some(&sliced_both_sectors()[..]), None],
        )
        .unwrap();
        assert_eq!(
            output,
            vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0]
        );
    }

    #[test]
    fn fusion_scatter_preflights_all_shapes_before_mutation() {
        let destination = two_key_destination();
        let source = source_with_shapes(&[vec![1, 2], vec![2, 2]]);
        let mut output = vec![7.0; destination.required_len().unwrap()];
        let before = output.clone();
        assert!(matches!(
            fusion_scatter_add_assign(
                strided_kernel::Exec::serial(),
                &destination,
                &mut output,
                source.structure(),
                FusionOperand::direct(&source),
                &[1.0; 6],
                &[Some(&sliced_both_sectors()[..]), None],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
        assert_eq!(output, before);
    }

    #[test]
    fn per_sector_tables_reject_a_missing_sector_instead_of_starting_at_zero() {
        // A table that names only one of the two Z2 sectors must not let the
        // other block through with an implicit start of 0.
        let destination = two_key_destination();
        let source = source_with_shapes(&[vec![1, 2], vec![1, 2]]);
        let mut output = vec![7.0; destination.required_len().unwrap()];
        let before = output.clone();
        assert!(matches!(
            fusion_scatter_add_assign(
                strided_kernel::Exec::serial(),
                &destination,
                &mut output,
                source.structure(),
                FusionOperand::direct(&source),
                &[1.0, 2.0, 3.0, 4.0],
                &[
                    Some(&[(SectorId::new(0), SelectedRuns::from_elem(1..2, 1))][..]),
                    None
                ],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
        assert_eq!(output, before);

        assert!(matches!(
            oriented_fusion_restrict_owned(
                source.structure(),
                FusionOperand::direct(&destination_operand()),
                &[0.0; 12],
                &[
                    Some(&[(SectorId::new(0), SelectedRuns::from_elem(1..2, 1))][..]),
                    None
                ],
            ),
            Err(OperationError::StructureMismatch { .. })
        ));
    }

    /// What: a two-run selection gathers rows `{0, 2}` of every sector in
    /// order, and the scatter puts them back there (zeros elsewhere): the
    /// dense row gather and its adjoint, by hand.
    #[test]
    fn two_run_selection_gathers_and_scatters_rows_in_order() {
        let full = destination_operand();
        let kept = [
            (SectorId::new(0), runs(&[0..1, 2..3])),
            (SectorId::new(1), runs(&[0..1, 2..3])),
        ];
        let restricted = source_with_shapes(&[vec![2, 2], vec![2, 2]]);
        // Column-major 3x2 per sector: block 0 holds 0..6, block 1 6..12.
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let gathered = oriented_fusion_restrict_owned(
            restricted.structure(),
            FusionOperand::direct(&full),
            &data,
            &[Some(&kept[..]), None],
        )
        .unwrap();
        assert_eq!(gathered, vec![0.0, 2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0]);
        let mut scattered = vec![0.0; 12];
        fusion_scatter_add_assign(
            strided_kernel::Exec::serial(),
            full.structure(),
            &mut scattered,
            restricted.structure(),
            FusionOperand::direct(&restricted),
            &gathered,
            &[Some(&kept[..]), None],
        )
        .unwrap();
        assert_eq!(
            scattered,
            vec![0.0, 0.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0, 9.0, 0.0, 11.0]
        );
    }

    /// The full two-sector space, read as a restriction source.
    fn destination_operand() -> crate::DynamicFusionMapSpace {
        source_with_shapes(&[vec![3, 2], vec![3, 2]])
    }

    /// Per block, lazy `add` normalizes one joint layout and walks it
    /// once (it made two adapter calls, each converting and then normalizing
    /// its own layout), and restriction and adjoint materialization normalize
    /// once per block.
    #[test]
    fn checked_block_paths_make_one_layout_pass_and_one_walk_per_block() {
        use symtn_operations::{
            take_checked_block_passes, take_owned_block_prefills, CheckedBlockPasses,
        };
        let passes = |count| CheckedBlockPasses {
            layout_passes: count,
            span_walks: count,
        };
        let destination = two_key_destination();
        let operand = destination_operand();
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        take_checked_block_passes();
        take_owned_block_prefills();
        let output = oriented_fusion_add_owned(
            &destination,
            FusionOperand::direct(&operand),
            &data,
            FusionOperand::direct(&operand),
            &data,
            2.0,
            0.5,
        )
        .unwrap();
        assert_eq!(take_checked_block_passes(), passes(2));
        assert_eq!(
            output,
            data.iter().map(|x| 2.0 * x + 0.5 * x).collect::<Vec<_>>()
        );

        let output = oriented_fusion_restrict_owned(
            &destination,
            FusionOperand::direct(&operand),
            &data,
            &[None, None],
        )
        .unwrap();
        assert_eq!(take_checked_block_passes(), passes(2));
        assert_eq!(output, data);

        let (logical, parent_space, _, parent) = adjoint_fixture();
        take_checked_block_passes();
        let materialized =
            crate::adjoint::materialize_adjoint_data_dyn(&parent_space, &logical, &parent).unwrap();
        assert_eq!(take_checked_block_passes(), passes(1));
        assert_eq!(materialized.len(), 6);
        // Each owned output tiles its storage, so none is zero-filled
        // before the blocks overwrite it.
        assert_eq!(take_owned_block_prefills(), 0);
    }

    #[test]
    fn oriented_add_and_inner_read_the_adjoint_parent_in_both_orders() {
        let (logical, parent_space, direct, parent) = adjoint_fixture();
        let direct_operand = FusionOperand::direct(&logical);
        let adjoint_operand = FusionOperand::adjoint(&parent_space);
        let alpha = Complex64::new(0.5, -0.25);
        let beta = Complex64::new(-1.0, 0.75);
        let mut expected = vec![Complex64::zero(); 6];
        for column in 0..2 {
            for row in 0..3 {
                let logical_index = row + 3 * column;
                let parent_index = column + 2 * row;
                expected[logical_index] =
                    alpha * direct[logical_index] + beta * parent[parent_index].conj();
            }
        }
        for (lhs, lhs_data, rhs, rhs_data, lhs_factor, rhs_factor) in [
            (
                direct_operand,
                direct.as_slice(),
                adjoint_operand,
                parent.as_slice(),
                alpha,
                beta,
            ),
            (
                adjoint_operand,
                parent.as_slice(),
                direct_operand,
                direct.as_slice(),
                beta,
                alpha,
            ),
        ] {
            let output = oriented_fusion_add_owned(
                logical.structure(),
                lhs,
                lhs_data,
                rhs,
                rhs_data,
                lhs_factor,
                rhs_factor,
            )
            .unwrap();
            assert_eq!(output, expected);
        }

        let expected_inner = (0..6).fold(Complex64::zero(), |sum, logical_index| {
            let row = logical_index % 3;
            let column = logical_index / 3;
            let parent_index = column + 2 * row;
            sum + direct[logical_index].conj() * parent[parent_index].conj()
        });
        assert_eq!(
            oriented_fusion_inner(
                logical.structure(),
                direct_operand,
                &direct,
                adjoint_operand,
                &parent,
                |_| 1.0,
            )
            .unwrap(),
            expected_inner
        );
        assert_eq!(
            oriented_fusion_inner_with(
                logical.structure(),
                direct_operand,
                &direct,
                adjoint_operand,
                &parent,
                |_| Ok::<_, OperationError>(1.0),
            )
            .unwrap(),
            expected_inner
        );
        assert_eq!(
            oriented_fusion_inner(
                logical.structure(),
                adjoint_operand,
                &parent,
                direct_operand,
                &direct,
                |_| 1.0,
            )
            .unwrap(),
            expected_inner.conj()
        );

        let inactive = vec![Complex64::new(f64::NAN, f64::NAN); parent.len()];
        let output = oriented_fusion_add_owned(
            logical.structure(),
            direct_operand,
            &direct,
            adjoint_operand,
            &inactive,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
        assert_eq!(output, direct);

        let output = oriented_fusion_add_owned(
            logical.structure(),
            direct_operand,
            &vec![Complex64::new(f64::NAN, f64::NAN); direct.len()],
            adjoint_operand,
            &parent,
            Complex64::zero(),
            Complex64::one(),
        )
        .unwrap();
        let expected = (0..6)
            .map(|logical_index| {
                let row = logical_index % 3;
                let column = logical_index / 3;
                parent[column + 2 * row].conj()
            })
            .collect::<Vec<_>>();
        assert_eq!(output, expected);

        let output = oriented_fusion_add_owned(
            logical.structure(),
            direct_operand,
            &vec![Complex64::new(f64::NAN, f64::NAN); direct.len()],
            adjoint_operand,
            &inactive,
            Complex64::zero(),
            Complex64::zero(),
        )
        .unwrap();
        assert_eq!(output, vec![Complex64::zero(); direct.len()]);
    }
}
