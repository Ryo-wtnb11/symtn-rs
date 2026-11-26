//! The one host movement leaf: every two-operand block move that is not a
//! whole planned role (Strided's `UpdatePlan` runs those) walks its fused
//! layout through [`span`], generic over the destination [`Store`] and the
//! per-element [`Action`].
//!
//! The element map is chosen once per role, outside the element loop, by
//! [`with_conjugation!`], [`with_payload_map!`] or [`with_transform_map!`]
//! (the only places that decide identity, zero and scale arms), and each arm
//! binds it as a closure of its own type, so every arm's walk is a separate
//! monomorphic body.

use core::mem::{size_of, MaybeUninit};
use core::ops::{Add, Mul};

use crate::BakedFusedLayout;

/// How a destination slot receives a value.
pub(crate) trait Store<T: Copy> {
    type Slot;

    fn put(slot: &mut Self::Slot, value: T);
}

/// An initialized destination, which an update may read.
pub(crate) enum Init {}

/// An uninitialized destination: only [`Write`] implements [`Action`] for
/// it, so no action can read a slot.
pub(crate) enum Uninit {}

impl<T: Copy> Store<T> for Init {
    type Slot = T;

    fn put(slot: &mut T, value: T) {
        *slot = value;
    }
}

impl<T: Copy> Store<T> for Uninit {
    type Slot = MaybeUninit<T>;

    fn put(slot: &mut MaybeUninit<T>, value: T) {
        slot.write(value);
    }
}

/// What a source element does to its destination slot.
pub(crate) trait Action<S: Store<T>, T: Copy> {
    fn element(&self, slot: &mut S::Slot, value: T);
}

/// `dst = map(src)`, the only action on an uninitialized store.
pub(crate) struct Write<F>(pub(crate) F);

impl<S: Store<T>, T: Copy, F: Fn(T) -> T> Action<S, T> for Write<F> {
    fn element(&self, slot: &mut S::Slot, value: T) {
        S::put(slot, (self.0)(value));
    }
}

/// `dst = dst + map(src)`: the exact accumulate, never `1 * dst + ...`.
pub(crate) struct Accumulate<F>(pub(crate) F);

impl<T: Copy + Add<T, Output = T>, F: Fn(T) -> T> Action<Init, T> for Accumulate<F> {
    fn element(&self, slot: &mut T, value: T) {
        *slot = *slot + (self.0)(value);
    }
}

/// `dst = beta * dst + map(src)`.
pub(crate) struct Axpby<T, F> {
    pub(crate) beta: T,
    pub(crate) map: F,
}

impl<T, F> Action<Init, T> for Axpby<T, F>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
    F: Fn(T) -> T,
{
    fn element(&self, slot: &mut T, value: T) {
        *slot = self.beta * *slot + (self.map)(value);
    }
}

/// The update an initialized destination receives.
#[derive(Clone, Copy)]
pub(crate) enum Update<T> {
    /// `dst = map(src)`; the destination is never read.
    Overwrite,
    /// `dst = dst + map(src)`.
    Accumulate,
    /// `dst = beta * dst + map(src)`.
    Axpby(T),
}

/// Applies `action` to every `(dst, src)` element pair `layout` addresses
/// from the two offsets, in one odometer over the fused axes with axis 0
/// innermost; a run that steps by one in both operands is walked as two
/// slices. `index` is the caller's traversal scratch of the layout's rank.
///
/// Why `#[inline(always)]`: every leaf path calls this once per item, and
/// whether that call stays out of line was an inlining outcome that small
/// source or flag changes flipped (an out-of-line `span` per item measured
/// slower on short items). Forced, the walk of each item is part of the
/// caller's item loop, which runs once per item list (a leaf's job, a pack
/// column list, a serial phase), on every leaf path alike.
///
/// # Panics
///
/// On an offset outside either slice (bounds-checked indexing).
#[inline(always)]
pub(crate) fn span<S, A, T>(
    dst: &mut [S::Slot],
    src: &[T],
    layout: BakedFusedLayout<'_>,
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    action: &A,
) where
    T: Copy,
    S: Store<T>,
    A: Action<S, T>,
{
    let (dims, dst_strides, src_strides) =
        (layout.dims(), layout.dst_strides(), layout.src_strides());
    let rank = dims.len();
    debug_assert_eq!(rank, dst_strides.len());
    debug_assert_eq!(rank, src_strides.len());
    if rank == 0 || dims.contains(&0) {
        return;
    }
    debug_assert_eq!(rank, index.len());
    let (len, inner_dst, inner_src) = (dims[0], dst_strides[0], src_strides[0]);
    // Why the run kernel is chosen here, once per call, and not per run:
    // the inner strides are a layout property, and a per-run test of them
    // ahead of the general kernel's bodies keeps the odometer's invariants
    // out of registers (spilled and reloaded on every run).
    if inner_dst == 1 && inner_src == 1 {
        // Why the length class is chosen per call: `len` is the layout's, so
        // one test selects the run kernel for every run of the call.
        if len <= contiguous_piece::<T>() {
            odometer(
                dims,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                index,
                |dst_base, src_base| {
                    let (dst_start, src_start) = (dst_base as usize, src_base as usize);
                    contiguous_blocks::<S, A, T>(
                        &mut dst[dst_start..dst_start + len],
                        &src[src_start..src_start + len],
                        action,
                    );
                },
            );
        } else {
            odometer(
                dims,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                index,
                |dst_base, src_base| {
                    let (dst_start, src_start) = (dst_base as usize, src_base as usize);
                    contiguous_pieces::<S, A, T>(
                        &mut dst[dst_start..dst_start + len],
                        &src[src_start..src_start + len],
                        action,
                    );
                },
            );
        }
        return;
    }
    if inner_dst == 0 && inner_src == 0 {
        odometer(
            dims,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            index,
            |dst_base, src_base| {
                let slot = &mut dst[dst_base as usize];
                let value = src[src_base as usize];
                // Why the bound stays `usize`: a zero-stride extent may
                // exceed `isize::MAX`.
                for _ in 0..len {
                    action.element(slot, value);
                }
            },
        );
        return;
    }
    if size_of::<T>() == 0 || size_of::<S::Slot>() == 0 {
        // Why zero-sized operands take the mathematical run: their slices
        // may exceed `isize::MAX`, so a valid window bounds neither a
        // wrapped position nor a negative base, and the windowed walk below
        // could revisit or never leave it. Each of the `len` positions is
        // addressed exactly, and an offset outside `isize`-scaled arithmetic
        // or either slice panics.
        let offset = |base: isize, stride: isize, position: usize| {
            (stride as i128)
                .checked_mul(position as i128)
                .and_then(|offset| offset.checked_add(base as i128))
                .and_then(|offset| usize::try_from(offset).ok())
                .expect("run offset outside usize")
        };
        odometer(
            dims,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            index,
            |dst_base, src_base| {
                for position in 0..len {
                    let slot = &mut dst[offset(dst_base, inner_dst, position)];
                    action.element(slot, src[offset(src_base, inner_src, position)]);
                }
            },
        );
        return;
    }
    // Each operand's run lies in a window of `(len - 1) * |stride| + 1`
    // slots: a layout property, so it is checked once per call (a size that
    // overflows has no slice to address). A run starts at its window's
    // first slot, or its last when the stride is negative.
    let last = len - 1;
    let size = |stride: isize| {
        last.checked_mul(stride.unsigned_abs())
            .and_then(|extent| extent.checked_add(1))
            .expect("run extent overflows usize")
    };
    let (dst_size, src_size) = (size(inner_dst), size(inner_src));
    let first = |stride: isize, size: usize| if stride < 0 { size - 1 } else { 0 };
    let (dst_first, src_first) = (first(inner_dst, dst_size), first(inner_src, src_size));
    // Why wrapping arithmetic: a slice of sized elements holds at most
    // `isize::MAX` of them, so a negative base (at least `2^(BITS-1)` as
    // `usize`) or a wrapped bound yields a range no such slice satisfies
    // (`start > end` or `end > len`), and slicing panics.
    let window = |base: isize, first: usize, size: usize| {
        let low = (base as usize).wrapping_sub(first);
        low..low.wrapping_add(size)
    };
    // Why not strided's `replay_map_raw[_uninit]`: it requires `Send + Sync`
    // scalars and allocates above fused rank 8; see #2519.
    if inner_dst == 1 && inner_src != 0 {
        let step = inner_src.unsigned_abs();
        // Why `chunks_exact`, not `step_by`: zipping two TrustedRandomAccess
        // iterators gives one counted exit test per element; `step_by` keeps
        // a second remaining-length test per element.
        if inner_src > 0 {
            odometer(
                dims,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                index,
                |dst_base, src_base| {
                    // Why the rebinding: an outlined run body otherwise
                    // reloads the captured step from memory on every element.
                    let (step, last, src_size) = (step, last, src_size);
                    let slots = &mut dst[window(dst_base, dst_first, dst_size)];
                    let values = &src[window(src_base, src_first, src_size)];
                    let (body, tail) = slots.split_at_mut(last);
                    for (slot, chunk) in body.iter_mut().zip(values.chunks_exact(step)) {
                        action.element(slot, chunk[0]);
                    }
                    action.element(&mut tail[0], values[src_size - 1]);
                },
            );
        } else {
            odometer(
                dims,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                index,
                |dst_base, src_base| {
                    let (step, last, src_size) = (step, last, src_size);
                    let slots = &mut dst[window(dst_base, dst_first, dst_size)];
                    let values = &src[window(src_base, src_first, src_size)];
                    let (body, tail) = slots.split_at_mut(last);
                    for (slot, chunk) in body.iter_mut().zip(values.rchunks_exact(step)) {
                        action.element(slot, chunk[step - 1]);
                    }
                    action.element(&mut tail[0], values[0]);
                },
            );
        }
        return;
    }
    // Why the run is walked by two in-window positions rather than by
    // `step_by` or `base + position * stride`: both bounds are validated
    // once per run by slicing its windows, after which the loop condition
    // is the only check each position needs; `step_by` checks every step
    // against the remaining length, and indexing from the bases checks both
    // operands as well as counting positions. Each position steps by its
    // stride with wrapping; as a window holds at most `isize::MAX` slots, a
    // positive stride leaves it without wrapping and a negative one wraps
    // past zero to at least `2^(BITS-1)`, so the run ends after exactly
    // `len` positions. A zero-stride
    // operand stays on its one slot while the other ends the run; a
    // repeated destination is thus updated in source order.
    let (dst_delta, src_delta) = (inner_dst as usize, inner_src as usize);
    odometer(
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        index,
        |dst_base, src_base| {
            let slots = &mut dst[window(dst_base, dst_first, dst_size)];
            let values = &src[window(src_base, src_first, src_size)];
            let (mut slot, mut value) = (dst_first, src_first);
            while slot < slots.len() && value < values.len() {
                action.element(&mut slots[slot], values[value]);
                slot = slot.wrapping_add(dst_delta);
                value = value.wrapping_add(src_delta);
            }
        },
    );
}

/// The most bytes of one contiguous run [`span`] walks in blocks; a longer
/// run is walked in pieces of at most this size, each one element loop.
///
/// Why pieces: an identity map's loop over a contiguous run is lowered by
/// LLVM to one `memcpy` of the whole run, and libc switches a copy above its
/// non-temporal threshold to stores that bypass the cache (several times
/// slower for a multi-megabyte block on a many-core Xeon). glibc's x86
/// tunable rejects thresholds below 0x4040 bytes and its default derives from
/// the shared cache size, so a piece of at most 16 KiB never reaches that
/// switch, whatever the host; one call per 16 KiB is a negligible share of
/// the copy. Scaled and conjugated maps are no `memcpy`, and the pieces only
/// add one outer loop to them.
const CONTIGUOUS_PIECE_BYTES: usize = 16 << 10;

/// [`CONTIGUOUS_PIECE_BYTES`] in elements of `T` (at least one).
#[inline(always)]
fn contiguous_piece<T>() -> usize {
    (CONTIGUOUS_PIECE_BYTES / size_of::<T>().max(1)).max(1)
}

/// Elements one block of [`contiguous_blocks`] moves: a small power of two
/// (one 64-byte line of `f64`).
const CONTIGUOUS_BLOCK: usize = 8;

/// Applies `action` to every element pair of two equal-length contiguous
/// runs of at most one piece: whole blocks of [`CONTIGUOUS_BLOCK`] elements,
/// then the shorter tail, without any libc call.
///
/// Why the barriers: LLVM's loop-idiom pass turns a loop of stores of
/// consecutive loads into one variable-length `memcpy` (a libc call per
/// run, measured slower for the 128-byte runs of small sectors), and it
/// merges a loop of constant-size block copies into one such call as well.
/// It does so only when nothing else in the loop may access memory. An
/// empty `black_box` after each block and each tail element is an opaque
/// statement that may (an empty `asm` with a memory clobber, no
/// instruction), so no loop here is a pure copy: each block stays a
/// constant-size move that codegen lowers to register or vector moves, and
/// the tail stays an element loop. Nothing depends on the host: the block
/// size is a structural constant, and the property holds for every length.
#[inline(always)]
fn contiguous_blocks<S, A, T>(dst: &mut [S::Slot], src: &[T], action: &A)
where
    T: Copy,
    S: Store<T>,
    A: Action<S, T>,
{
    #[cfg(test)]
    tests::observe_blocked_run();
    let (dst_blocks, dst_tail) = dst.as_chunks_mut::<CONTIGUOUS_BLOCK>();
    let (src_blocks, src_tail) = src.as_chunks::<CONTIGUOUS_BLOCK>();
    for (slots, values) in dst_blocks.iter_mut().zip(src_blocks) {
        for (slot, &value) in slots.iter_mut().zip(values) {
            action.element(slot, value);
        }
        core::hint::black_box(());
    }
    for (slot, &value) in dst_tail.iter_mut().zip(src_tail) {
        action.element(slot, value);
        core::hint::black_box(());
    }
}

/// Applies `action` to every element pair of two equal-length contiguous
/// runs, one bounded piece at a time.
///
/// Why `chunks`, not `chunks_exact`: a piece loop of constant-length copies
/// is what LLVM's loop-idiom pass merges back into one whole-run `memcpy`;
/// the last piece's shorter length keeps each piece's length a runtime value.
#[inline(always)]
fn contiguous_pieces<S, A, T>(dst: &mut [S::Slot], src: &[T], action: &A)
where
    T: Copy,
    S: Store<T>,
    A: Action<S, T>,
{
    let piece = contiguous_piece::<T>();
    for (slots, values) in dst.chunks_mut(piece).zip(src.chunks(piece)) {
        #[cfg(test)]
        tests::observe_piece(slots.len());
        for (slot, &value) in slots.iter_mut().zip(values) {
            action.element(slot, value);
        }
    }
}

/// Calls `run(dst_base, src_base)` for every run of axis 0 `dims` addresses
/// from the two offsets, the outer axes stepped as one odometer with axis 1
/// fastest. `index` is the caller's traversal scratch of the layout's rank.
///
/// Why `#[inline(always)]`: this is [`span`]'s own loop, called once per
/// span with that call's run kernel, and it is forced for the same reason.
#[inline(always)]
fn odometer(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    mut run: impl FnMut(isize, isize),
) {
    let rank = dims.len();
    // Why the reslices: they state the equal ranks once, so the carry loop
    // indexes all four slices by `axis < rank` without a check per slice.
    let (index, dst_strides, src_strides) = (
        &mut index[..rank],
        &dst_strides[..rank],
        &src_strides[..rank],
    );
    index.fill(0);
    let (mut dst_base, mut src_base) = (dst_offset, src_offset);
    loop {
        run(dst_base, src_base);
        let mut axis = 1;
        loop {
            if axis >= rank {
                return;
            }
            index[axis] += 1;
            dst_base += dst_strides[axis];
            src_base += src_strides[axis];
            if index[axis] < dims[axis] {
                break;
            }
            dst_base -= dims[axis] as isize * dst_strides[axis];
            src_base -= dims[axis] as isize * src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

/// Binds `$map` to `|x| x` or `|x| conj(x)` over `$t` and evaluates `$body`
/// with it. A pure copy closure lets LLVM lower a contiguous run to
/// `memcpy`.
macro_rules! with_conjugation {
    ($t:ty, $conjugate:expr, |$map:ident| $body:expr) => {
        if $conjugate {
            let $map = |value: $t| $crate::ConjugateValue::maybe_conj(value, true);
            $body
        } else {
            let $map = |value: $t| value;
            $body
        }
    };
}

/// Binds `$map` to the element map of `alpha * op(src)` over `$t` for a
/// payload-typed `alpha` and evaluates `$body` with it.
///
/// Why the identity arm: `1 * (inf + 0i)` is `inf + NaN i`, so `alpha = 1`
/// must copy rather than multiply. Why the zero arm: VectorInterface's
/// `scale(x, 0) = zero(x) * 0`, so a zero scale ignores the source.
macro_rules! with_payload_map {
    ($t:ty, $alpha:expr, $conjugate:expr, |$map:ident| $body:expr) => {{
        let alpha: $t = $alpha;
        let conjugate: bool = $conjugate;
        if ::num_traits::One::is_one(&alpha) {
            $crate::kernel_adapter::movement::with_conjugation!($t, conjugate, |$map| $body)
        } else if ::num_traits::Zero::is_zero(&alpha) {
            let zero = <$t as ::num_traits::Zero>::zero() * alpha;
            let $map = move |_: $t| zero;
            $body
        } else {
            let $map =
                move |value: $t| alpha * $crate::ConjugateValue::maybe_conj(value, conjugate);
            $body
        }
    }};
}

/// Binds `$map` to the element map of a tree-transform scale (see
/// [`TransformScale`](crate::TransformScale)) on `op(src)` over `$t` and
/// evaluates `$body` with it: TensorKit's three source forms, no scale for
/// `One()`, a structural coefficient in its own type, and a payload multiply
/// once `α` is folded in, plus the zero arm of [`with_payload_map!`].
///
/// The structural closure holds the coefficient by reference, so a threaded
/// plan needs only `C: Sync`.
macro_rules! with_transform_map {
    ($t:ty, $scale:expr, $conjugate:expr, |$map:ident| $body:expr) => {{
        let scale = $scale;
        let conjugate: bool = $conjugate;
        if scale.is_identity() {
            $crate::kernel_adapter::movement::with_conjugation!($t, conjugate, |$map| $body)
        } else if scale.is_zero() {
            let zero = scale.apply(<$t as ::num_traits::Zero>::zero());
            let $map = move |_: $t| zero;
            $body
        } else {
            match scale {
                $crate::TransformScale::Structural(coefficient) => {
                    let coefficient = &coefficient;
                    let $map = move |value: $t| {
                        $crate::RecouplingCoefficientAction::scale_by_coefficient(
                            $crate::ConjugateValue::maybe_conj(value, conjugate),
                            *coefficient,
                        )
                    };
                    $body
                }
                $crate::TransformScale::Data(alpha) => {
                    let $map = move |value: $t| {
                        alpha * $crate::ConjugateValue::maybe_conj(value, conjugate)
                    };
                    $body
                }
            }
        }
    }};
}

pub(crate) use {with_conjugation, with_payload_map, with_transform_map};

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    std::thread_local! {
        /// The pieces [`contiguous_pieces`] walked on this thread, and the
        /// longest of them.
        static PIECES: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
        /// The runs [`contiguous_blocks`] walked on this thread.
        static BLOCKED: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn observe_blocked_run() {
        BLOCKED.with(|runs| runs.set(runs.get() + 1));
    }

    fn take_blocked_runs() -> usize {
        BLOCKED.with(|runs| runs.replace(0))
    }

    pub(super) fn observe_piece(len: usize) {
        PIECES.with(|pieces| {
            let (count, longest) = pieces.get();
            pieces.set((count + 1, longest.max(len)));
        });
    }

    fn take_pieces() -> (usize, usize) {
        PIECES.with(|pieces| pieces.replace((0, 0)))
    }

    /// What: a contiguous run of at most one piece is walked in blocks and a
    /// longer one in pieces of at most [`CONTIGUOUS_PIECE_BYTES`] (each run
    /// takes exactly one of the two paths), and on both stores every element
    /// lands where a plain copy puts it, for lengths around the block and
    /// piece sizes and unaligned starts.
    #[test]
    fn contiguous_runs_copy_in_blocks_or_bounded_pieces() {
        let piece = contiguous_piece::<f64>();
        assert_eq!(piece * size_of::<f64>(), CONTIGUOUS_PIECE_BYTES);
        let block = CONTIGUOUS_BLOCK;
        let src: Vec<f64> = (0..3 * piece + 8).map(|i| i as f64 + 0.5).collect();
        let lengths = [
            0,
            1,
            block - 1,
            block,
            block + 1,
            2 * block + 1,
            piece - 1,
            piece,
            piece + 1,
            2 * piece + 1,
        ];
        for len in lengths {
            for (dst_offset, src_offset) in [(0, 0), (1, 3), (5, 2)] {
                let layout = [len];
                let baked =
                    BakedFusedLayout::from_compiled_normalized_slices(&layout, &[1], &[1], None);
                let expected = &src[src_offset..src_offset + len];
                let (pieces, blocked) = match len {
                    0 => (0, 0),
                    len if len > piece => (len.div_ceil(piece), 0),
                    _ => (0, 1),
                };
                let mut dst = vec![-1.0; len + dst_offset + 1];
                take_pieces();
                take_blocked_runs();
                span::<Init, _, f64>(
                    &mut dst,
                    &src,
                    baked,
                    dst_offset as isize,
                    src_offset as isize,
                    &mut [0],
                    &Write(|value: f64| value),
                );
                let (count, longest) = take_pieces();
                assert_eq!((count, take_blocked_runs()), (pieces, blocked));
                assert!(longest <= piece);
                assert_eq!(&dst[dst_offset..dst_offset + len], expected);
                assert!(dst[..dst_offset].iter().all(|&value| value == -1.0));
                assert_eq!(dst[dst_offset + len], -1.0);

                let mut uninit = vec![MaybeUninit::new(-1.0); len + dst_offset + 1];
                span::<Uninit, _, f64>(
                    &mut uninit,
                    &src,
                    baked,
                    dst_offset as isize,
                    src_offset as isize,
                    &mut [0],
                    &Write(|value: f64| value),
                );
                assert_eq!((take_pieces().0, take_blocked_runs()), (pieces, blocked));
                // SAFETY: every slot was initialized by `MaybeUninit::new`,
                // and `span` only writes initialized values into slots.
                let written: Vec<f64> = uninit
                    .iter()
                    .map(|slot| unsafe { slot.assume_init() })
                    .collect();
                assert_eq!(written, dst);
            }
        }
    }

    /// What: an outer axis of long contiguous runs pieces every run at
    /// the same bound and keeps each run at its own offsets.
    #[test]
    fn every_long_run_of_a_layout_is_pieced() {
        let piece = contiguous_piece::<f64>();
        let len = piece + 3;
        let src: Vec<f64> = (0..3 * len + 7).map(|i| i as f64).collect();
        let mut dst = vec![0.0; 2 * (len + 2)];
        take_pieces();
        span::<Init, _, f64>(
            &mut dst,
            &src,
            BakedFusedLayout::from_compiled_normalized_slices(
                &[len, 2],
                &[1, (len + 2) as isize],
                &[1, (len + 7) as isize],
                None,
            ),
            0,
            0,
            &mut [0; 2],
            &Write(|value: f64| value),
        );
        assert_eq!(take_pieces(), (4, piece));
        assert_eq!(dst[..len], src[..len]);
        assert_eq!(dst[len + 2..2 * len + 2], src[len + 7..2 * len + 7]);
    }

    /// Stops the traversal at its first element.
    struct StopAtFirst;

    impl Action<Init, f64> for StopAtFirst {
        fn element(&self, slot: &mut f64, value: f64) {
            assert_eq!((*slot, value), (7.0, 42.0));
            panic!("first element visited");
        }
    }

    /// What: a zero-stride extent above `isize::MAX` still visits its first
    /// element rather than walking an empty range.
    #[test]
    #[should_panic(expected = "first element visited")]
    fn span_visits_a_zero_stride_extent_above_isize_max() {
        let dims = [isize::MAX as usize + 1];
        let mut index = [0];
        span::<Init, _, f64>(
            &mut [7.0],
            &[42.0],
            BakedFusedLayout::from_compiled_normalized_slices(&dims, &[0], &[0], None),
            0,
            0,
            &mut index,
            &StopAtFirst,
        );
    }

    /// Counts the visits of a zero-sized run.
    struct Count<'a>(&'a core::cell::Cell<usize>);

    impl Action<Init, ()> for Count<'_> {
        fn element(&self, _: &mut (), _: ()) {
            self.0.set(self.0.get() + 1);
        }
    }

    /// What: a zero-sized source longer than `isize::MAX` with a stride of a
    /// quarter of the address space is walked as exactly its `len`
    /// positions, where a wrapped position would cycle inside its window.
    #[test]
    fn span_walks_exactly_len_positions_of_a_long_zero_sized_run() {
        let quarter = 1usize << (usize::BITS - 2);
        let src = vec![(); 3 * quarter + 1];
        let visits = core::cell::Cell::new(0);
        let mut index = [0];
        span::<Init, _, ()>(
            &mut [()],
            &src,
            BakedFusedLayout::from_compiled_normalized_slices(
                &[4],
                &[0],
                &[quarter as isize],
                None,
            ),
            0,
            0,
            &mut index,
            &Count(&visits),
        );
        assert_eq!(visits.get(), 4);
    }

    /// Records each visited `(dst, src)` offset pair.
    struct Record<'a>(&'a core::cell::RefCell<Vec<(usize, usize)>>);

    impl Action<Init, usize> for Record<'_> {
        fn element(&self, slot: &mut usize, value: usize) {
            self.0.borrow_mut().push((*slot, value));
        }
    }

    /// What: the odometer keeps axis 0 innermost with exact signed offset
    /// arithmetic on negative strides; slot and source values are their
    /// own offsets, so each visit records the pair it addressed.
    #[test]
    fn span_order_keeps_axis_zero_fastest_with_negative_strides() {
        let mut dst: Vec<usize> = (0..400).collect();
        let src: Vec<usize> = (0..400).collect();
        let visits = core::cell::RefCell::new(Vec::new());
        let mut index = [0; 3];
        span::<Init, _, usize>(
            &mut dst,
            &src,
            BakedFusedLayout::from_compiled_normalized_slices(
                &[3, 2, 2],
                &[1, 10, -100],
                &[2, -20, 200],
                None,
            ),
            300,
            60,
            &mut index,
            &Record(&visits),
        );
        let starts = visits
            .borrow()
            .chunks(3)
            .map(|run| run[0])
            .collect::<Vec<_>>();
        assert_eq!(starts, [(300, 60), (310, 40), (200, 260), (210, 240)]);
        assert_eq!(visits.borrow()[..3], [(300, 60), (301, 62), (302, 64)]);

        visits.borrow_mut().clear();
        span::<Init, _, usize>(
            &mut dst,
            &src,
            BakedFusedLayout::from_compiled_normalized_slices(&[3], &[1], &[-2], None),
            0,
            4,
            &mut index[..1],
            &Record(&visits),
        );
        assert_eq!(*visits.borrow(), [(0, 4), (1, 2), (2, 0)]);
    }

    /// What: a repeated destination with a negative source stride is
    /// updated in source position order.
    #[test]
    fn span_updates_a_repeated_destination_in_source_order() {
        let src: Vec<usize> = (0..5).collect();
        let visits = core::cell::RefCell::new(Vec::new());
        let mut index = [0];
        span::<Init, _, usize>(
            &mut [7],
            &src,
            BakedFusedLayout::from_compiled_normalized_slices(&[3], &[0], &[-2], None),
            0,
            4,
            &mut index,
            &Record(&visits),
        );
        assert_eq!(*visits.borrow(), [(7, 4), (7, 2), (7, 0)]);
    }
}
