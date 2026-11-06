//! The recording writer and the fixtures the native and barrier lane
//! replay tests share.

use super::*;
use crate::kernel_adapter::movement::{Store, Write};
use crate::transform_structure::TreeTransformDirectItem;
use std::thread::ThreadId;

use super::super::writers::{crop_role, inactive_role, DestinationWriter, ReplayParts};

/// One destination element as the recording writer leaves it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Mark {
    pub(super) writes: u32,
    pub(super) worker: Option<ThreadId>,
}

/// A movement store that records each write's thread instead of a value.
enum Record {}

impl Store<f64> for Record {
    type Slot = Mark;

    fn put(slot: &mut Mark, _: f64) {
        slot.writes += 1;
        slot.worker = Some(std::thread::current().id());
    }
}

/// A writer that addresses exactly the elements the real writers address,
/// whole or cropped to a band, and records who wrote each.
pub(super) struct Recorder<'a> {
    pub(super) parts: ReplayParts<'a, f64>,
    pub(super) src: &'a [f64],
}

impl DestinationWriter<Mark, f64> for Recorder<'_> {
    type Kernels = ();

    fn kernels(&self) {}

    fn overwrites(&self) -> bool {
        true
    }

    fn direct(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        dst: &mut [Mark],
        dst_start: isize,
        band: Option<super::super::writers::Band>,
        _: crate::host_pool::Exec<'_>,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let mut cropped = [0; crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK];
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let (role, shift) = inactive_role(layouts, layout, band, &mut cropped)?;
                span::<Record, _, f64>(
                    dst,
                    &[0.0],
                    role,
                    layouts.entry(layout).offset - dst_start + shift,
                    0,
                    &mut index[..role.dims().len()],
                    &Write(|value: f64| value),
                );
            }
            TreeTransformDirectItem::Single { block, .. } => {
                let (dst_layout, src_layout, _) = scheduled_single(self.parts.blocks, block)?;
                let role = layouts.role(dst_layout)?;
                let (role, dst_shift, src_shift) = match band {
                    None => (role, 0, 0),
                    Some(band) => crop_role(role, band, &mut cropped)?,
                };
                span::<Record, _, f64>(
                    dst,
                    self.src,
                    role,
                    layouts.entry(dst_layout).offset - dst_start + dst_shift,
                    layouts.entry(src_layout).offset + src_shift,
                    &mut index[..role.dims().len()],
                    &Write(|value: f64| value),
                );
            }
        }
        Ok(())
    }

    fn scatter(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        dst_layout: usize,
        packed: &[f64],
        packed_offset: usize,
        dst: &mut [Mark],
        dst_start: isize,
        band: Option<super::super::writers::Band>,
        _: crate::host_pool::Exec<'_>,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let mut cropped = [0; crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK];
        let role = layouts.role(dst_layout)?;
        let (role, dst_shift, src_shift) = match band {
            None => (role, 0, 0),
            Some(band) => crop_role(role, band, &mut cropped)?,
        };
        span::<Record, _, f64>(
            dst,
            packed,
            role,
            layouts.entry(dst_layout).offset - dst_start + dst_shift,
            offset_to_isize(packed_offset)? + src_shift,
            &mut index[..role.dims().len()],
            &Write(|value: f64| value),
        );
        Ok(())
    }
}

/// Destination element addresses of a block `(offset, dims, strides)`.
pub(super) fn addresses(offset: usize, dims: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut out = vec![offset];
    for (&dim, &stride) in dims.iter().zip(strides) {
        out = out
            .iter()
            .flat_map(|&base| (0..dim).map(move |i| base + i * stride))
            .collect();
    }
    out
}

/// Columns of the interleaved bandable blocks.
pub(super) const COLUMNS: usize = 32;

/// Two `[2, COLUMNS]` blocks of row stride 4 interleave in one atom
/// bandable by its columns, a Single and a Multi group's first column; the
/// group's second column, `[2, COLUMNS]` of row stride 2, fuses to one
/// contiguous atom; with `injective`, a Single over the `[3, 2]` / `[2, 3]`
/// layout, which no band certificate can split, forms a second component.
/// Returns the destination blocks `(offset, dims, strides)`, the structures
/// and the transform.
#[allow(clippy::type_complexity)]
pub(super) fn interleaved_bandable(
    injective: bool,
) -> (
    Vec<(usize, [usize; 2], [usize; 2])>,
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let mut blocks = vec![
        (0, [2, COLUMNS], [1, 4]),
        (2, [2, COLUMNS], [1, 4]),
        (4 * COLUMNS, [2, COLUMNS], [1, 2]),
    ];
    if injective {
        blocks.push((6 * COLUMNS, [3, 2], [2, 3]));
    }
    let dst = Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            blocks
                .iter()
                .enumerate()
                .map(|(key, &(offset, dims, strides))| {
                    BlockSpec::with_key(
                        BlockKey::ordinal(key),
                        dims.to_vec(),
                        strides.to_vec(),
                        offset,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap(),
    );
    let src = Arc::new(
        BlockStructure::packed_column_major(2, blocks.iter().map(|&(_, dims, _)| dims.to_vec()))
            .unwrap(),
    );
    let mut specs = vec![
        TreeTransformBlockSpec::single(0, 0, -2.0),
        TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0]),
    ];
    if injective {
        specs.push(TreeTransformBlockSpec::single(3, 3, 1.0));
    }
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    (blocks, dst, src, transform)
}

/// The error a test executor injects for recoupling group `group`.
pub(super) fn injected(group: usize) -> DenseError {
    DenseError::Unsupported {
        op: "spy",
        message: format!("group {group}"),
    }
}

/// The group message of an injected error.
pub(super) fn group_of(error: OperationError) -> String {
    match error {
        OperationError::Dense(DenseError::Unsupported { op: "spy", message }) => message,
        other => panic!("not an injected error: {other:?}"),
    }
}
