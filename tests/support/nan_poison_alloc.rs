//! A global allocator that fills every fresh, non-zeroed byte with `0xFF`.
//!
//! Every IEEE `f32`/`f64` word of all-one bits is a NaN, so a payload
//! element that an operation publishes without writing it reads as NaN (and
//! any arithmetic on it stays NaN) instead of happening to hold a stale zero.
//! `alloc_zeroed` is left untouched: its zeros are the allocator's promise.
//!
//! Install with
//! `#[global_allocator] static ALLOCATOR: nan_poison_alloc::NanPoisonAllocator = nan_poison_alloc::NanPoisonAllocator;`

use std::alloc::{GlobalAlloc, Layout, System};

pub struct NanPoisonAllocator;

// SAFETY: every hook forwards its arguments unchanged to `System`; the only
// addition writes bytes inside the block `System` just returned, between the
// old and the new size, which that block owns and nothing has read yet.
unsafe impl GlobalAlloc for NanPoisonAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            unsafe { pointer.write_bytes(0xFF, layout.size()) };
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !pointer.is_null() && new_size > layout.size() {
            unsafe {
                pointer
                    .add(layout.size())
                    .write_bytes(0xFF, new_size - layout.size())
            };
        }
        pointer
    }
}
