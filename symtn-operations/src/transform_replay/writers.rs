//! The destination writes of a replay, one per item kind, for the
//! initialised destination (`T = D`) and the owned uninitialised one
//! (`T = MaybeUninit<D>`): every runner (lane or staged, any width) writes
//! through these, whole or cropped to a band of its atom.

use super::*;
use crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK;
use crate::transform_structure::TreeTransformDirectItem;

/// The parts of a completed transform every lane reads: shared by
/// reference only, so lanes need no bound on the coefficient type beyond
/// `Sync`.
#[derive(Clone, Copy)]
pub(super) struct ReplayParts<'a, C> {
    pub layouts: &'a TreeTransformLayoutTable,
    pub blocks: &'a [TreeTransformBlock],
    pub single_coefficients: &'a [C],
    pub storage_conjugate: bool,
    pub plan: &'a crate::transform_structure::TreeTransformRecouplingPlan,
}

impl<'a, C: Copy> ReplayParts<'a, C> {
    pub fn new(task: TreeTransformTaskView<'a, C>) -> Self {
        Self {
            layouts: task.layouts(),
            blocks: task.blocks(),
            single_coefficients: task.single_coefficients(),
            storage_conjugate: task.storage_conjugate(),
            plan: task.recoupling_plan(),
        }
    }
}

/// Outer indices `[k0, k1)` of a bandable atom's items: the item's baked
/// layout with its outermost extent replaced by `k1 - k0` and both operand
/// offsets advanced by `k0` outer strides.
pub(super) type Band = (usize, usize);

/// `dims` with its outermost extent replaced by the band's length, copied
/// into `scratch`, and the destination and source offset shifts of the
/// band's first outer index. Compilation admits a band only for a rank
/// within `scratch`'s length.
#[inline]
pub(super) fn crop_dims<'s>(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    (k0, k1): Band,
    scratch: &'s mut [usize; FUSED_LAYOUT_INLINE_RANK],
) -> Result<(&'s [usize], isize, isize), OperationError> {
    let rank = dims.len();
    let (Some(outer), true) = (rank.checked_sub(1), rank <= FUSED_LAYOUT_INLINE_RANK) else {
        return Err(OperationError::InvalidArgument {
            message: "tree transform band crops a layout it cannot hold",
        });
    };
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

/// Zero source strides for a one-sided (inactive) layout.
const ZERO_STRIDES: [isize; FUSED_LAYOUT_INLINE_RANK] = [0; FUSED_LAYOUT_INLINE_RANK];

/// `rank` zero source strides: the inline ones, or above their rank (a
/// whole inactive item, which compilation admits at any rank; only bands
/// are limited to the inline rank) the lane's `scratch` resized to `rank`.
fn zero_source_strides(scratch: &mut Vec<isize>, rank: usize) -> &[isize] {
    match ZERO_STRIDES.get(..rank) {
        Some(zeros) => zeros,
        None => {
            scratch.clear();
            scratch.resize(rank, 0);
            scratch
        }
    }
}

/// Writes the destination items of a replay into a destination of `T`.
/// `dst` starts at absolute element `dst_start`; `band` crops the item to
/// outer indices of its bandable atom (see [`crop_dims`]), `None` writes it
/// whole on its compiled path. A writer is shared by every lane; each lane
/// brings its own [`Self::Kernels`] and traversal scratch.
pub(super) trait DestinationWriter<T, D>: Sync {
    /// A lane's kernel adapter state.
    type Kernels: Send;

    /// A fresh lane's kernel state.
    fn kernels(&self) -> Self::Kernels;

    /// Whether every write ignores the destination: only such a write may
    /// run on a compiled plan.
    fn overwrites(&self) -> bool;

    #[allow(clippy::too_many_arguments)]
    fn direct(
        &self,
        kernels: &mut Self::Kernels,
        zero_strides: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        dst: &mut [T],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError>;

    /// One scatter column from `packed`, whose column starts at
    /// `packed_offset`.
    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &self,
        kernels: &mut Self::Kernels,
        zero_strides: &mut Vec<isize>,
        column: &TreeTransformScatterReplay,
        packed: &[D],
        packed_offset: usize,
        dst: &mut [T],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError>;
}

/// The initialised replay's writes: `alpha` (with its structural scales)
/// and the destination mode.
pub(super) struct InitializedWriter<'a, A, D, C> {
    pub prototype: A,
    pub task: ReplayParts<'a, C>,
    pub src: &'a [D],
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

    fn overwrites(&self) -> bool {
        self.mode.overwrites()
    }

    fn direct(
        &self,
        kernels: &mut A,
        zero_strides: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        dst: &mut [D],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let layout = layout as usize;
                let (dims, strides) = layouts.inactive_role(layout)?;
                let mut cropped = [0; FUSED_LAYOUT_INLINE_RANK];
                let (dims, shift, _) = match band {
                    Some(band) => crop_dims(dims, strides, &[], band, &mut cropped)?,
                    None => (dims, 0, 0),
                };
                let offset = layouts.entry(layout).offset - dst_start + shift;
                match self.mode {
                    DestinationMode::Axpby(beta) if beta.is_one() => Ok(()),
                    DestinationMode::Axpby(beta) => {
                        kernels.scale_strided(exec, dst, dims, strides, offset, beta)
                    }
                    DestinationMode::Overwrite => {
                        let zero = [D::zero()];
                        kernels.copy_scale_strided_baked(
                            exec,
                            dst,
                            &zero,
                            dims,
                            strides,
                            zero_source_strides(zero_strides, dims.len()),
                            offset,
                            0,
                            false,
                            D::one(),
                            None,
                            None,
                        )
                    }
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
                        self.src,
                        alpha,
                        self.mode,
                    );
                };
                let role = layouts.role(dst_layout)?;
                let mut cropped = [0; FUSED_LAYOUT_INLINE_RANK];
                let (dims, dst_shift, src_shift) = crop_dims(
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    band,
                    &mut cropped,
                )?;
                let baked = crate::BakedFusedLayout::from_compiled_normalized_slices(
                    dims,
                    role.dst_strides(),
                    role.src_strides(),
                    None,
                );
                kernels.transform_strided_baked(
                    exec,
                    zero_strides,
                    dst,
                    self.src,
                    dims,
                    role.dst_strides(),
                    role.src_strides(),
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
        column: &TreeTransformScatterReplay,
        packed: &[D],
        packed_offset: usize,
        dst: &mut [D],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let offset = layouts.entry(column.dst_layout).offset;
        let alpha = self.alpha.at(offset);
        let Some(band) = band else {
            return scatter_column_on_token(
                exec,
                gate,
                kernels,
                zero_strides,
                Some(index),
                layouts,
                column.dst_layout,
                packed,
                packed_offset,
                dst,
                dst_start,
                alpha,
                self.mode,
            );
        };
        let role = layouts.role(column.dst_layout)?;
        let mut cropped = [0; FUSED_LAYOUT_INLINE_RANK];
        let (dims, dst_shift, src_shift) = crop_dims(
            role.dims(),
            role.dst_strides(),
            role.src_strides(),
            band,
            &mut cropped,
        )?;
        let baked = crate::BakedFusedLayout::from_compiled_normalized_slices(
            dims,
            role.dst_strides(),
            role.src_strides(),
            None,
        );
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

/// The owned writer's writes into uninitialised memory: `alpha` and the
/// overwrite mode; every element its items address is written once.
pub(super) struct OwnedWriter<'a, D, C> {
    pub task: ReplayParts<'a, C>,
    pub src: &'a [D],
    pub alpha: D,
}

impl<D, C> DestinationWriter<MaybeUninit<D>, D> for OwnedWriter<'_, D, C>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Kernels = ();

    fn kernels(&self) {}

    fn overwrites(&self) -> bool {
        true
    }

    fn direct(
        &self,
        _: &mut (),
        zero_strides: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        dst: &mut [MaybeUninit<D>],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let mut cropped = [0; FUSED_LAYOUT_INLINE_RANK];
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let layout = layout as usize;
                let (dims, strides) = layouts.inactive_role(layout)?;
                let (dims, shift, _) = match band {
                    Some(band) => crop_dims(dims, strides, &[], band, &mut cropped)?,
                    None => (dims, 0, 0),
                };
                let zero = [D::zero()];
                write_fused_uninit(
                    crate::BakedFusedLayout::from_compiled_normalized_slices(
                        dims,
                        strides,
                        zero_source_strides(zero_strides, dims.len()),
                        None,
                    ),
                    dst,
                    &zero,
                    layouts.entry(layout).offset - dst_start + shift,
                    0,
                    index,
                    |value| value,
                )
            }
            TreeTransformDirectItem::Single { block, .. } => {
                let (dst_layout, src_layout, coefficient) =
                    scheduled_single(self.task.blocks, block)?;
                let role = layouts.role(dst_layout)?;
                let (role, elements, dst_shift, src_shift) = match band {
                    None => (role, layouts.entry(dst_layout).element_count, 0, 0),
                    Some(band) => {
                        let (dims, dst_shift, src_shift) = crop_dims(
                            role.dims(),
                            role.dst_strides(),
                            role.src_strides(),
                            band,
                            &mut cropped,
                        )?;
                        (
                            crate::BakedFusedLayout::from_compiled_normalized_slices(
                                dims,
                                role.dst_strides(),
                                role.src_strides(),
                                None,
                            ),
                            dims.iter().product(),
                            dst_shift,
                            src_shift,
                        )
                    }
                };
                write_uninit_layout_from_source(
                    exec,
                    gate,
                    role,
                    elements,
                    dst,
                    layouts.entry(dst_layout).offset - dst_start + dst_shift,
                    self.src,
                    layouts.entry(src_layout).offset + src_shift,
                    self.task.storage_conjugate,
                    TransformScale::new(self.alpha, self.task.single_coefficients[coefficient]),
                    index,
                )
            }
        }
    }

    fn scatter(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        column: &TreeTransformScatterReplay,
        packed: &[D],
        packed_offset: usize,
        dst: &mut [MaybeUninit<D>],
        dst_start: isize,
        band: Option<Band>,
        exec: Exec<'_>,
        gate: &LeafGate,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.task.layouts;
        let role = layouts.role(column.dst_layout)?;
        let mut cropped = [0; FUSED_LAYOUT_INLINE_RANK];
        let (role, elements, dst_shift, src_shift) = match band {
            None => (role, layouts.entry(column.dst_layout).element_count, 0, 0),
            Some(band) => {
                let (dims, dst_shift, src_shift) = crop_dims(
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    band,
                    &mut cropped,
                )?;
                (
                    crate::BakedFusedLayout::from_compiled_normalized_slices(
                        dims,
                        role.dst_strides(),
                        role.src_strides(),
                        None,
                    ),
                    dims.iter().product(),
                    dst_shift,
                    src_shift,
                )
            }
        };
        write_uninit_layout_from_packed(
            exec,
            gate,
            role,
            elements,
            dst,
            layouts.entry(column.dst_layout).offset - dst_start + dst_shift,
            packed,
            offset_to_isize(packed_offset)? + src_shift,
            self.alpha,
            index,
        )
    }
}
