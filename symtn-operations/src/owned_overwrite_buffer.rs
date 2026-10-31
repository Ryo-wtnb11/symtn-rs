use core::alloc::Layout;
use core::mem::MaybeUninit;
use std::alloc::{alloc_zeroed, handle_alloc_error};

/// Scalars whose all-zero byte pattern is a valid value equal to `Self::zero()`.
///
/// # Safety
///
/// Implementors promise that `len * size_of::<Self>()` zeroed bytes with
/// `Self`'s alignment are `len` valid, initialised values, each equal to the
/// additive zero, and that `Self` is not zero-sized (`alloc_zeroed` with a
/// zero-size layout is undefined behaviour). IEEE floats and `#[repr(C)]`
/// pairs of them satisfy this.
pub unsafe trait ZeroBytes: Copy {}

// SAFETY: IEEE 754 binary64 encodes `+0.0` as all-zero bits, `f64` has no
// padding and no niche, and `size_of::<f64>() == 8 != 0`. Any number of zeroed
// bytes at `align_of::<f64>()` is therefore that many initialised `+0.0`
// values, and `+0.0 == <f64 as Zero>::zero()`.
unsafe impl ZeroBytes for f64 {}
// SAFETY: `num_complex::Complex<T>` is `#[repr(C)] { re: T, im: T }` (num-complex
// 0.4 `src/lib.rs`), so `Complex64` is exactly two `f64` with no padding and no
// niche: all-zero bytes are `0.0 + 0.0i`, which is `Complex64::zero()`.
// `size_of::<Complex64>() == 16 != 0`.
unsafe impl ZeroBytes for num_complex::Complex64 {}
// SAFETY: IEEE 754 binary32 encodes `+0.0` as all-zero bits, `f32` has no
// padding and no niche, and `size_of::<f32>() == 4 != 0`; same argument as the
// `f64` impl above.
unsafe impl ZeroBytes for f32 {}
// SAFETY: same `#[repr(C)]` two-field layout as `Complex64`, over `f32`: no
// padding, no niche, all-zero bytes are `0.0 + 0.0i == Complex32::zero()`, and
// `size_of::<Complex32>() == 8 != 0`.
unsafe impl ZeroBytes for num_complex::Complex32 {}

/// `len` zero scalars zeroed by the allocator (calloc), not by a scalar loop.
///
/// Why not `vec![D::zero(); len]`: std lowers that to `alloc_zeroed` only for
/// types with its private `IsZero` specialisation, which `num_complex`
/// scalars lack, so `Complex64` would be filled by an element loop over `len`.
pub fn zeroed_payload<D: ZeroBytes>(len: usize) -> Vec<D> {
    // Checked here as well so the allocation does not rest on the trait
    // contract alone.
    const {
        assert!(
            size_of::<D>() != 0,
            "ZeroBytes types must not be zero-sized"
        )
    };
    if len == 0 {
        return Vec::new();
    }
    let Ok(layout) = Layout::array::<D>(len) else {
        capacity_overflow()
    };
    // SAFETY: `ZeroBytes` is implemented only for types whose all-zero byte
    // pattern is a valid `D::zero()`; `alloc_zeroed` returns
    // `len * size_of::<D>()` zeroed bytes with `D`'s alignment (a null result
    // is routed to `handle_alloc_error` exactly as `Vec` does), so all `len`
    // elements are initialised before `Vec::from_raw_parts` receives the same
    // pointer with the matching `Layout` (`capacity == len`, same alignment).
    // `len == 0` returned above, so the zero-size allocation path is never
    // taken, and `layout.size() != 0` holds as `alloc_zeroed` requires.
    unsafe {
        let pointer = alloc_zeroed(layout);
        if pointer.is_null() {
            handle_alloc_error(layout);
        }
        Vec::from_raw_parts(pointer.cast::<D>(), len, len)
    }
}

#[cold]
fn capacity_overflow() -> ! {
    panic!("capacity overflow")
}

struct OwnedOverwriteBuffer<D: Copy> {
    data: Vec<D>,
    target_len: usize,
}

impl<D: Copy> OwnedOverwriteBuffer<D> {
    fn new(target_len: usize) -> Self {
        Self {
            data: Vec::with_capacity(target_len),
            target_len,
        }
    }

    fn spare_mut(&mut self) -> &mut [MaybeUninit<D>] {
        &mut self.data.spare_capacity_mut()[..self.target_len]
    }

    fn finish(mut self) -> Vec<D> {
        // Why not expose initialized length during replay: an executor error
        // may occur after partial writes. The physical overwrite proof is the
        // only authority that permits the completed buffer to escape.
        unsafe {
            self.data.set_len(self.target_len);
        }
        self.data
    }
}

/// Writes `D::zero()` to every element of `piece` and returns it initialized.
///
/// Why not `<[MaybeUninit<D>]>::write_filled`: it is unstable at the
/// workspace MSRV.
pub(crate) fn write_zeros<D: Copy + num_traits::Zero>(piece: &mut [MaybeUninit<D>]) -> &mut [D] {
    for slot in piece.iter_mut() {
        slot.write(D::zero());
    }
    // SAFETY: the loop above wrote a valid `D` to every element of `piece`,
    // and the returned borrow is `piece`'s own, so nothing can de-initialize
    // an element while the initialized view is live. `D: Copy` has no drop.
    unsafe { piece.assume_init_mut() }
}

/// Runs `write` over `target_len` uninitialized elements and returns them
/// as a `Vec` of length `target_len` on `Ok`; on `Err` or a panic the
/// storage is dropped at length zero.
///
/// The caller's `write` must initialize every element of its argument
/// before returning `Ok`; this is the only obligation the length transition
/// rests on.
pub(crate) fn initialize_owned<D, E>(
    target_len: usize,
    write: impl FnOnce(&mut [MaybeUninit<D>]) -> Result<(), E>,
) -> Result<Vec<D>, E>
where
    D: Copy,
{
    let mut buffer = OwnedOverwriteBuffer::new(target_len);
    write(buffer.spare_mut())?;
    Ok(buffer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_payload_holds_len_zero_values_for_every_scalar_type() {
        // What: `zeroed_payload` returns `len` values bit-equal to the
        // positive zero of each supported scalar; Miri checks that every
        // element is initialised and the allocation layout round-trips.
        for len in [0usize, 1, 7, 64] {
            let real = zeroed_payload::<f64>(len);
            assert_eq!(real.len(), len);
            assert_eq!(real.capacity(), len);
            assert!(real.iter().all(|value| value.to_bits() == 0));

            let complex = zeroed_payload::<num_complex::Complex64>(len);
            assert_eq!(complex.len(), len);
            assert_eq!(complex.capacity(), len);
            assert!(complex
                .iter()
                .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));

            let single = zeroed_payload::<f32>(len);
            assert_eq!(single.len(), len);
            assert_eq!(single.capacity(), len);
            assert!(single.iter().all(|value| value.to_bits() == 0));

            let single_complex = zeroed_payload::<num_complex::Complex32>(len);
            assert_eq!(single_complex.len(), len);
            assert_eq!(single_complex.capacity(), len);
            assert!(single_complex
                .iter()
                .all(|value| value.re.to_bits() == 0 && value.im.to_bits() == 0));
        }
    }

    #[test]
    fn zeroed_payload_vec_can_grow_and_drop() {
        // What: the `alloc_zeroed` allocation carries a `Vec`-compatible
        // layout for every scalar width, so reallocation and drop of the
        // returned buffer are sound (the Miri check).
        fn grow<T: ZeroBytes + Copy + PartialEq + std::fmt::Debug>(len: usize, value: T) {
            let mut values = zeroed_payload::<T>(len);
            values.push(value);
            assert_eq!(values.len(), len + 1);
            assert_eq!(values[len], value);
        }
        grow(5, 1.5_f32);
        grow(5, 1.5_f64);
        grow(3, num_complex::Complex32::new(1.0, 2.0));
        grow(3, num_complex::Complex64::new(1.0, 2.0));
    }

    #[test]
    fn completed_transaction_returns_initialized_values() {
        // What: the sole unsafe length transition follows complete writes and
        // remains suitable for Miri's uninitialized-memory checks.
        let values = initialize_owned(4, |dst| -> Result<(), ()> {
            for (index, slot) in dst.iter_mut().enumerate() {
                slot.write(index + 1);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(values, [1, 2, 3, 4]);
    }

    #[test]
    fn failed_partial_transaction_does_not_return_storage() {
        // What: an error after a partial write drops the length-zero buffer
        // instead of exposing a Vec containing uninitialized elements.
        let result = initialize_owned(4, |dst| {
            dst[0].write(7u64);
            Err("injected writer failure")
        });
        assert_eq!(result, Err("injected writer failure"));
    }

    #[test]
    fn panicking_partial_transaction_cannot_expose_initialized_length() {
        // What: unwinding after a partial write drops only a length-zero Vec;
        // no partially initialized owned output can escape the transaction.
        let result = std::panic::catch_unwind(|| {
            let _ = initialize_owned(4, |dst| -> Result<(), ()> {
                dst[0].write(7u64);
                panic!("injected writer panic");
            });
        });
        assert!(result.is_err());
    }
}
