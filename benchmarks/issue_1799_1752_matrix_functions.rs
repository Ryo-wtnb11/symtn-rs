//! Copy into symtn/examples/issue_1799_1752_matrix_functions.rs, then run with
//! --release. Input setup is outside timing; publication and drop are inside
//! every call. Multiplicity-free dense `pinv`, Hermitian and general `exp`,
//! left/right polar and their lazy-adjoint routes on `[V, V] <- [V, V]`.
use num_complex::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, LinalgBackend, Runtime, TensorMap};

struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

fn grow(delta: isize) {
    let live = LIVE.fetch_add(delta, Relaxed) + delta;
    PEAK.fetch_max(live, Relaxed);
}
// SAFETY: every pointer/layout is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
            grow(layout.size() as isize);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNT.load(Relaxed) {
            grow(-(layout.size() as isize));
        }
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNT.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
            grow(size as isize - layout.size() as isize);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOC: Allocator = Allocator;

/// Median ns, allocation calls and bytes of one call, and its peak live bytes
/// above the live bytes at its start (frees of storage allocated before the
/// call count against the peak only once it is reached).
fn measure(mut f: impl FnMut()) -> (u128, usize, usize, isize) {
    for _ in 0..3 {
        f();
    }
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    COUNT.store(true, Relaxed);
    f();
    COUNT.store(false, Relaxed);
    let (calls, bytes, peak) = (CALLS.load(Relaxed), BYTES.load(Relaxed), PEAK.load(Relaxed));
    let mut samples = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        for _ in 0..5 {
            f();
        }
        samples.push(start.elapsed().as_nanos() / 5);
    }
    samples.sort_unstable();
    (samples[5], calls, bytes, peak)
}

macro_rules! cases {
    ($runtime:expr, $family:literal, $rule:expr, $irrep:expr, $count:expr, $k:expr, $dtype:ty, $name:literal, $value:expr) => {{
        let leg = GradedSpace::try_new(Arc::new($rule), (0..$count).map(|c| (($irrep)(c), $k)))
            .unwrap();
        let general: TensorMap<_, $dtype> =
            TensorMap::from_subblock_fn($runtime, [&leg, &leg], [&leg, &leg], |_, i| {
                ($value)(i)
            })
            .unwrap();
        let hermitian = general
            .compose(&general.adjoint().unwrap())
            .unwrap()
            .scale(<$dtype>::from(1.0e-3)).unwrap();
        let adjoint = general.adjoint().unwrap();
        let small = general.scale(<$dtype>::from(1.0e-3)).unwrap();
        let ops: [(&str, &dyn Fn()); 8] = [
            ("pinv", &|| drop(black_box(general.pinv(&[0, 1], &[2, 3], 1e-12).unwrap()))),
            ("pinv_adjoint", &|| drop(black_box(adjoint.pinv(&[0, 1], &[2, 3], 1e-12).unwrap()))),
            ("exp_hermitian", &|| drop(black_box(hermitian.exp(&[0, 1], &[2, 3]).unwrap()))),
            ("exp_general", &|| drop(black_box(small.exp(&[0, 1], &[2, 3]).unwrap()))),
            ("left_polar", &|| drop(black_box(general.left_polar(&[0, 1], &[2, 3]).unwrap()))),
            ("right_polar", &|| drop(black_box(general.right_polar(&[0, 1], &[2, 3]).unwrap()))),
            ("left_polar_adjoint", &|| drop(black_box(adjoint.left_polar(&[0, 1], &[2, 3]).unwrap()))),
            ("right_polar_adjoint", &|| drop(black_box(adjoint.right_polar(&[0, 1], &[2, 3]).unwrap()))),
        ];
        for (op, f) in ops {
            let (median, calls, bytes, peak) = measure(|| f());
            println!("{},{},{},{},{op},{median},{calls},{bytes},{peak}", $family, $name, $count, $k);
        }
    }};
}

fn main() {
    let runtime = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    println!("family,dtype,sectors,k,operation,median_ns,alloc_calls,alloc_bytes,peak_live_bytes");
    let real = |i: &[usize]| {
        ((i[0] * 7 + i[1] * 3 + i[2] * 5 + i[3] * 11) % 13) as f64 / 13.0 - 0.5
    };
    let complex = |i: &[usize]| Complex64::new(real(i), ((i[0] + 2 * i[3]) % 5) as f64 / 5.0);
    for (count, k) in [(3, 2), (3, 6), (7, 2)] {
        cases!(&runtime, "U1", U1FusionRule, |c: i32| U1Irrep::new(c - 1), count, k, f64, "f64", real);
        cases!(&runtime, "U1", U1FusionRule, |c: i32| U1Irrep::new(c - 1), count, k, Complex64, "c64", complex);
        cases!(&runtime, "SU2", SU2FusionRule, |c: i32| SU2Irrep::from_twice_spin(c as usize), count, k, f64, "f64", real);
        cases!(&runtime, "SU2", SU2FusionRule, |c: i32| SU2Irrep::from_twice_spin(c as usize), count, k, Complex64, "c64", complex);
    }
}
