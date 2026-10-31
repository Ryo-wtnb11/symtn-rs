//! Copy into symtn/examples/issue_1684_null.rs and build with --release.
//! Every measured call includes public factor publication and output drop.
use num_complex::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, LinalgBackend, Runtime, SectorSpectrum, TensorMap};

struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
// SAFETY: pointer and layout are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNT.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOC: Allocator = Allocator;

fn measure(mut f: impl FnMut()) -> (u128, u128, usize, usize) {
    let start = Instant::now();
    f();
    let first = start.elapsed().as_nanos();
    for _ in 0..3 {
        f();
    }
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    COUNT.store(true, Relaxed);
    f();
    COUNT.store(false, Relaxed);
    let calls = CALLS.load(Relaxed);
    let bytes = BYTES.load(Relaxed);
    let mut samples = [0; 7];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..5 {
            f();
        }
        *sample = start.elapsed().as_nanos() / 5;
    }
    samples.sort_unstable();
    (first, samples[3], calls, bytes)
}

fn main() {
    let runtime = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    println!("family,dtype,sectors,k,dual,storage,operation,setup_ns,first_ns,median_ns,alloc_calls,alloc_bytes");
    macro_rules! family {
        ($dtype:ty, $name:literal, $make:expr, $eps:expr, $family:literal, $rule:expr, $sector:expr) => {
            for (sectors, k, dual) in [(1, 8, false), (4, 64, true)] {
                for storage in ["diagonal", "dense", "near_cutoff"] {
                    let setup = Instant::now();
                    let mut leg = GradedSpace::try_new(Arc::new($rule),
                        (0..sectors).map(|c| (($sector)(c, false), k))).unwrap();
                    if dual { leg = leg.try_dual().unwrap(); }
                    let mut input: TensorMap<_, $dtype> = TensorMap::diagonal(&runtime, &leg,
                        (0..sectors).map(|c| SectorSpectrum {
                            sector: ($sector)(c, dual),
                            values: (0..k).map(|i| {
                                let value = if storage == "near_cutoff" && i == 0 {
                                    2.0 * $eps * k as f64 * k as f64
                                } else if i % 3 == 0 { 0.0 } else { (i + 1) as f64 };
                                ($make)(value)
                            }).collect(),
                        })).unwrap();
                    if storage == "dense" { input = input.materialize().unwrap(); }
                    let setup_ns = setup.elapsed().as_nanos();
                    for operation in ["left_null", "right_null"] {
                        let (first, median, calls, bytes) = measure(|| {
                            if operation == "left_null" {
                                black_box(input.left_null(&[0], &[1]).unwrap());
                            } else {
                                black_box(input.right_null(&[0], &[1]).unwrap());
                            }
                        });
                        println!("{},{},{sectors},{k},{dual},{storage},{operation},{setup_ns},{first},{median},{calls},{bytes}", $family, $name);
                    }
                }
            }
        };
    }
    macro_rules! scalar {
        ($dtype:ty, $name:literal, $make:expr, $eps:expr) => {
            family!(
                $dtype,
                $name,
                $make,
                $eps,
                "U1",
                U1FusionRule,
                |c: i32, dual| U1Irrep::new(if dual { -c } else { c })
            );
            family!(
                $dtype,
                $name,
                $make,
                $eps,
                "SU2",
                SU2FusionRule,
                |c: i32, _dual| SU2Irrep::from_twice_spin(c as usize)
            );
        };
    }
    scalar!(f64, "f64", |x: f64| x, f64::EPSILON);
    scalar!(
        Complex64,
        "c64",
        |x: f64| Complex64::new(x, 0.25 * x),
        f64::EPSILON
    );
}
