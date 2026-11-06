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

use core::mem::MaybeUninit;
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
/// Why `#[inline]`: without it, rustc gives each instance of this shared
/// generic leaf one home codegen unit (the defining module's), and the
/// role loops that call it live in other modules, so every role pays an
/// out-of-line call that LLVM never sees across. `#[inline]` is rustc's
/// `InlineAttr::Hint`, which makes the instance `InstantiationMode::LocalCopy`
/// in optimized builds: each caller's codegen unit gets its own copy, and
/// LLVM decides inlining with the caller in view. It forces nothing; this and
/// its [`odometer`] are the only such attributes in the movement layer.
///
/// # Panics
///
/// On an offset outside either slice (bounds-checked indexing).
#[inline]
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
        odometer(
            dims,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            index,
            |dst_base, src_base| {
                let (dst_start, src_start) = (dst_base as usize, src_base as usize);
                let run = dst[dst_start..dst_start + len].iter_mut();
                for (slot, &value) in run.zip(&src[src_start..src_start + len]) {
                    action.element(slot, value);
                }
            },
        );
        return;
    }
    odometer(
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        index,
        |dst_base, src_base| {
            // Why iterators over once-validated runs rather than indexing
            // `base + position * stride`: the indexed loop bound-checks both
            // operands on every element and keeps their lengths live in the
            // caller's register budget. Slicing each operand's run once
            // validates its first and last offsets, and `step_by` then walks
            // it with no per-element check.
            //
            // The offset `position * stride` of the run's last element; a
            // value no slice reaches when it overflows, so slicing panics.
            let last = len - 1;
            let reach = |base: isize, stride: isize| {
                isize::try_from(last)
                    .ok()
                    .and_then(|last| last.checked_mul(stride))
                    .and_then(|offset| base.checked_add(offset))
                    .unwrap_or(isize::MAX)
            };
            if inner_dst == 0 {
                // A repeated destination: the source order is the update
                // order, so the run keeps `position` ascending.
                let slot = &mut dst[dst_base as usize];
                if inner_src == 0 {
                    let value = src[src_base as usize];
                    // Why the bound stays `usize`: a zero-stride extent may
                    // exceed `isize::MAX`.
                    for _ in 0..len {
                        action.element(slot, value);
                    }
                } else if inner_src > 0 {
                    let run = &src[src_base as usize..=reach(src_base, inner_src) as usize];
                    for &value in run.iter().step_by(inner_src as usize) {
                        action.element(slot, value);
                    }
                } else {
                    let run = &src[reach(src_base, inner_src) as usize..=src_base as usize];
                    for &value in run.iter().rev().step_by(inner_src.unsigned_abs()) {
                        action.element(slot, value);
                    }
                }
            } else {
                // Distinct destination slots: each is written once per run,
                // so the run may be walked from its lowest destination
                // offset, with the source stepping the opposite way when the
                // destination stride is negative.
                let (dst_low, dst_step, src_first, src_step) = if inner_dst > 0 {
                    (dst_base, inner_dst, src_base, inner_src)
                } else {
                    (
                        reach(dst_base, inner_dst),
                        -inner_dst,
                        reach(src_base, inner_src),
                        -inner_src,
                    )
                };
                let slots = dst[dst_low as usize..=reach(dst_low, dst_step) as usize]
                    .iter_mut()
                    .step_by(dst_step as usize);
                if src_step == 0 {
                    let value = src[src_first as usize];
                    for slot in slots {
                        action.element(slot, value);
                    }
                } else if src_step > 0 {
                    let run = &src[src_first as usize..=reach(src_first, src_step) as usize];
                    for (slot, &value) in slots.zip(run.iter().step_by(src_step as usize)) {
                        action.element(slot, value);
                    }
                } else {
                    let run = &src[reach(src_first, src_step) as usize..=src_first as usize];
                    let values = run.iter().rev().step_by(src_step.unsigned_abs());
                    for (slot, &value) in slots.zip(values) {
                        action.element(slot, value);
                    }
                }
            }
        },
    );
}

/// Calls `run(dst_base, src_base)` for every run of axis 0 `dims` addresses
/// from the two offsets, the outer axes stepped as one odometer with axis 1
/// fastest. `index` is the caller's traversal scratch of the layout's rank.
///
/// Why `#[inline]`: this is [`span`]'s own loop, called once per span with
/// that call's run kernel, and it needs [`span`]'s local copy for the same
/// codegen-unit reason.
#[inline]
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
    }
}
