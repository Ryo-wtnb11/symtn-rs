//! The destination writes of a replay, one per item kind, for the
//! initialized destination (`T = D`) and the owned uninitialized one
//! (`T = MaybeUninit<D>`): every replay phase writes through these.

use super::*;
use crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK;
use crate::transform_structure::TreeTransformDirectItem;

/// The parts of a completed transform every writer and worker reads: shared
/// by reference only, so workers need no bound on the coefficient type beyond
/// `Sync`.
#[derive(Clone, Copy)]
pub(super) struct ReplayParts<'a, C> {
    pub layouts: &'a TreeTransformLayoutTable,
    pub blocks: &'a [TreeTransformBlock],
    pub single_coefficients: &'a [C],
    pub storage_conjugate: bool,
}

impl<'a, C: Copy> ReplayParts<'a, C> {
    pub fn new(task: TreeTransformTaskView<'a, C>) -> Self {
        Self {
            layouts: task.layouts(),
            blocks: task.blocks(),
            single_coefficients: task.single_coefficients(),
            storage_conjugate: task.storage_conjugate(),
        }
    }
}

/// The `(dst_layout, src_layout, coefficient)` of the Single block `item`
/// names.
pub(super) fn scheduled_single(
    blocks: &[TreeTransformBlock],
    item: u32,
) -> Result<(usize, usize, usize), OperationError> {
    match blocks.get(item as usize) {
        Some(&TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        }) => Ok((dst_layout, src_layout, coefficient)),
        _ => Err(OperationError::InvalidArgument {
            message: "tree transform direct item names a non-Single block",
        }),
    }
}

/// Outer indices `[k0, k1)` of a bandable atom's items: the item's baked
/// layout with its outermost extent replaced by `k1 - k0` and both operand
/// offsets advanced by `k0` outer strides.
pub(super) type Band = (usize, usize);

/// `dims` with its outermost extent replaced by the band's length, copied
/// into `scratch`, and the destination and source offset shifts of the
/// band's first outer index. Compilation admits a band only for a rank
/// within the scratch length.
///
/// Why `scratch` is an `Option` the crop fills: a whole-item write never
/// crops, and an array the caller zeroes up front costs every item its
/// store (measured per Direct item on the owned permute replay).
fn crop_dims<'s>(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    (k0, k1): Band,
    scratch: &'s mut Option<[usize; FUSED_LAYOUT_INLINE_RANK]>,
) -> Result<(&'s [usize], isize, isize), OperationError> {
    let rank = dims.len();
    let (Some(outer), true) = (rank.checked_sub(1), rank <= FUSED_LAYOUT_INLINE_RANK) else {
        return Err(OperationError::InvalidArgument {
            message: "tree transform band crops a layout it cannot hold",
        });
    };
    let scratch = scratch.insert([0; FUSED_LAYOUT_INLINE_RANK]);
    scratch[..rank].copy_from_slice(dims);
    scratch[outer] = k1 - k0;
    let k0 = offset_to_isize(k0)?;
    let shift = |stride: isize| {
        k0.checked_mul(stride)
            .ok_or(OperationError::ElementCountOverflow)
    };
    Ok((
        &scratch[..rank],
        shift(dst_strides[outer])?,
        shift(src_strides.get(outer).copied().unwrap_or(0))?,
    ))
}

/// A two-sided role cropped to `band`: the baked layout over `scratch`'s
/// dims and the destination and source offset shifts.
pub(super) fn crop_role<'s>(
    role: crate::BakedFusedLayout<'s>,
    band: Band,
    scratch: &'s mut Option<[usize; FUSED_LAYOUT_INLINE_RANK]>,
) -> Result<(crate::BakedFusedLayout<'s>, isize, isize), OperationError> {
    let (dims, dst_shift, src_shift) = crop_dims(
        role.dims(),
        role.dst_strides(),
        role.src_strides(),
        band,
        scratch,
    )?;
    Ok((
        crate::BakedFusedLayout::from_compiled_normalized_slices(
            dims,
            role.dst_strides(),
            role.src_strides(),
            None,
        ),
        dst_shift,
        src_shift,
    ))
}

/// An inactive layout's one-sided role, cropped to `band` when given, as a
/// baked layout against the compiled zero source strides (its source is one
/// zero element), and the destination offset shift.
pub(super) fn inactive_role<'s>(
    layouts: &'s TreeTransformLayoutTable,
    layout: usize,
    band: Option<Band>,
    scratch: &'s mut Option<[usize; FUSED_LAYOUT_INLINE_RANK]>,
) -> Result<(crate::BakedFusedLayout<'s>, isize), OperationError> {
    let (dims, strides) = layouts.inactive_role(layout)?;
    let zeros = layouts.zero_src_strides(dims.len())?;
    let (dims, shift) = match band {
        Some(band) => {
            let (dims, shift, _) = crop_dims(dims, strides, &[], band, scratch)?;
            (dims, shift)
        }
        None => (dims, 0),
    };
    Ok((
        crate::BakedFusedLayout::from_compiled_normalized_slices(dims, strides, zeros, None),
        shift,
    ))
}

/// Writes the destination items of a replay into a destination of `T`.
/// `dst` starts at destination element `dst_start`; `band` crops the item to
/// outer indices of its bandable atom, `None` writes it whole on its
/// compiled path (its plan when `gate` admits it on `exec`). A writer is
/// shared by every worker of a replay split; each worker brings its own
/// [`Self::Kernels`], traversal scratch and gate.
pub(super) trait DestinationWriter<T, D>: Sync {
    /// One worker's kernel adapter state.
    type Kernels: Send + Sync;

    /// A fresh worker's kernel state.
    fn kernels(&self) -> Self::Kernels;

    /// The write of direct item `item` from the replayed source `src` (one
    /// member's, in a member replay).
    ///
    /// Why the writers mark this `#[inline]`: it runs once per item inside
    /// the item loops of `batched.rs` and `chunk.rs`, which live in other
    /// modules. Without the hint rustc gives each writer instance one
    /// home codegen unit (this module's), so every item pays an out-of-line
    /// call and prologue that LLVM never sees across (measured on the owned
    /// permute replay of many small Singles, where the call is a visible
    /// share of each item's work); with it each caller's codegen unit gets
    /// a local copy, as for `movement::span`.
    #[allow(clippy::too_many_arguments)]
    fn direct(
        &self,
        kernels: &mut Self::Kernels,
        zero_strides: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        src: &[D],
        dst: &mut [T],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &mut LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError>;

    /// The scatter of destination layout `dst_layout` from `packed`, whose
    /// column starts at `packed_offset`.
    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &self,
        kernels: &mut Self::Kernels,
        zero_strides: &mut Vec<isize>,
        dst_layout: usize,
        packed: &[D],
        packed_offset: usize,
        dst: &mut [T],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &mut LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError>;
}

/// The initialized replay's writes: `alpha` (with its destination scales)
/// and the destination mode.
pub(super) struct InitializedWriter<'a, A, D, C> {
    pub prototype: A,
    pub task: ReplayParts<'a, C>,
    pub alpha: DestinationAlpha<'a, D, C>,
    pub mode: DestinationMode<D>,
}

impl<A, D, C> DestinationWriter<D, D> for InitializedWriter<'_, A, D, C>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Kernels = A;

    fn kernels(&self) -> A {
        self.prototype.clone()
    }

    #[inline]
    fn direct(
        &self,
        kernels: &mut A,
        zero_strides: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        src: &[D],
        dst: &mut [D],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &mut LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let mut cropped = None;
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let (role, shift) = inactive_role(layouts, layout, band, &mut cropped)?;
                let offset = layouts.entry(layout).offset - dst_start + shift;
                match self.mode {
                    DestinationMode::Axpby(beta) if beta.is_one() => Ok(()),
                    DestinationMode::Axpby(beta) => kernels.scale_strided(
                        exec,
                        dst,
                        role.dims(),
                        role.dst_strides(),
                        offset,
                        beta,
                    ),
                    DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
                        exec,
                        dst,
                        &[D::zero()],
                        role.dims(),
                        role.dst_strides(),
                        role.src_strides(),
                        offset,
                        0,
                        false,
                        D::one(),
                        Some(role),
                        Some(index),
                    ),
                }
            }
            TreeTransformDirectItem::Single { block, .. } => {
                let (dst_layout, src_layout, coefficient) =
                    scheduled_single(self.task.blocks, block)?;
                let coefficient = self.task.single_coefficients[coefficient];
                let alpha = self.alpha.at(layouts.entry(dst_layout).offset);
                let Some(band) = band else {
                    return tree_transform_single_on_token(
                        exec,
                        gate,
                        kernels,
                        zero_strides,
                        Some(index),
                        layouts,
                        dst_layout,
                        src_layout,
                        coefficient,
                        self.task.storage_conjugate,
                        dst,
                        dst_start,
                        src,
                        alpha,
                        self.mode,
                    );
                };
                let (baked, dst_shift, src_shift) =
                    crop_role(layouts.role(dst_layout)?, band, &mut cropped)?;
                kernels.transform_strided_baked(
                    exec,
                    zero_strides,
                    dst,
                    src,
                    baked.dims(),
                    baked.dst_strides(),
                    baked.src_strides(),
                    layouts.entry(dst_layout).offset - dst_start + dst_shift,
                    layouts.entry(src_layout).offset + src_shift,
                    self.task.storage_conjugate,
                    TransformScale::new(alpha, coefficient),
                    match self.mode {
                        DestinationMode::Axpby(beta) => Some(beta),
                        DestinationMode::Overwrite => None,
                    },
                    Some(baked),
                    Some(index),
                )
            }
        }
    }

    fn scatter(
        &self,
        kernels: &mut A,
        zero_strides: &mut Vec<isize>,
        dst_layout: usize,
        packed: &[D],
        packed_offset: usize,
        dst: &mut [D],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &mut LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let offset = layouts.entry(dst_layout).offset;
        let alpha = self.alpha.at(offset);
        let Some(band) = band else {
            return scatter_column_on_token(
                exec,
                gate,
                kernels,
                zero_strides,
                Some(index),
                layouts,
                dst_layout,
                packed,
                packed_offset,
                dst,
                dst_start,
                alpha,
                self.mode,
            );
        };
        let mut cropped = None;
        let (baked, dst_shift, src_shift) =
            crop_role(layouts.role(dst_layout)?, band, &mut cropped)?;
        scatter_role(
            exec,
            kernels,
            zero_strides,
            Some(index),
            baked,
            offset - dst_start + dst_shift,
            packed,
            offset_to_isize(packed_offset)? + src_shift,
            dst,
            alpha,
            self.mode,
        )
    }
}

/// The owned writer's writes into uninitialized memory: `alpha` and the
/// overwrite mode; every element its items address is written once.
pub(super) struct OwnedWriter<'a, D, C> {
    pub task: ReplayParts<'a, C>,
    pub alpha: D,
}

impl<D, C> DestinationWriter<MaybeUninit<D>, D> for OwnedWriter<'_, D, C>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Kernels = ();

    fn kernels(&self) {}

    #[inline]
    fn direct(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        src: &[D],
        dst: &mut [MaybeUninit<D>],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &mut LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let mut cropped = None;
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let (role, shift) = inactive_role(layouts, layout, band, &mut cropped)?;
                let index =
                    index
                        .get_mut(..role.dims().len())
                        .ok_or(OperationError::InvalidArgument {
                            message: "fused traversal scratch is shorter than the normalized rank",
                        })?;
                span::<Uninit, _, D>(
                    dst,
                    &[D::zero()],
                    role,
                    layouts.entry(layout).offset - dst_start + shift,
                    0,
                    index,
                    &Write(|value: D| value),
                );
                Ok(())
            }
            TreeTransformDirectItem::Single { block, .. } => {
                let (dst_layout, src_layout, coefficient) =
                    scheduled_single(self.task.blocks, block)?;
                let role = layouts.role(dst_layout)?;
                let (role, elements, dst_shift, src_shift) = match band {
                    None => (role, layouts.entry(dst_layout).element_count, 0, 0),
                    Some(band) => {
                        let (baked, dst_shift, src_shift) = crop_role(role, band, &mut cropped)?;
                        (baked, baked.dims().iter().product(), dst_shift, src_shift)
                    }
                };
                let scale =
                    TransformScale::new(self.alpha, self.task.single_coefficients[coefficient]);
                with_transform_map!(D, scale, self.task.storage_conjugate, |map| {
                    write_role_uninit(
                        exec,
                        gate,
                        role,
                        elements,
                        dst,
                        src,
                        layouts.entry(dst_layout).offset - dst_start + dst_shift,
                        layouts.entry(src_layout).offset + src_shift,
                        index,
                        map,
                    )
                })
            }
        }
    }

    fn scatter(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        dst_layout: usize,
        packed: &[D],
        packed_offset: usize,
        dst: &mut [MaybeUninit<D>],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &mut LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let role = layouts.role(dst_layout)?;
        let mut cropped = None;
        let (role, elements, dst_shift, src_shift) = match band {
            None => (role, layouts.entry(dst_layout).element_count, 0, 0),
            Some(band) => {
                let (baked, dst_shift, src_shift) = crop_role(role, band, &mut cropped)?;
                (baked, baked.dims().iter().product(), dst_shift, src_shift)
            }
        };
        // A zero alpha gives an exact zero even where the recoupling GEMM
        // left a NaN in `packed`, as TensorKit's scatter `tensoradd!` does.
        with_payload_map!(D, self.alpha, false, |map| {
            write_role_uninit(
                exec,
                gate,
                role,
                elements,
                dst,
                packed,
                layouts.entry(dst_layout).offset - dst_start + dst_shift,
                offset_to_isize(packed_offset)? + src_shift,
                index,
                map,
            )
        })
    }
}
