//! Copy into symtn/examples/issue_1692_polar.rs, then run with --release.
//! Input setup is separate; factor publication and drop are inside every call.
use num_complex::{Complex32, Complex64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::{GradedSpace, LinalgBackend, Runtime, SectorSpectrum, TensorMap};

struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
// SAFETY: every pointer/layout is forwarded unchanged to the system allocator.
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
    let cold = start.elapsed().as_nanos();
    for _ in 0..5 {
        f();
    }
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    COUNT.store(true, Relaxed);
    f();
    COUNT.store(false, Relaxed);
    let calls = CALLS.load(Relaxed);
    let bytes = BYTES.load(Relaxed);
    let mut samples = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        for _ in 0..20 {
            f();
        }
        samples.push(start.elapsed().as_nanos() / 20);
    }
    samples.sort_unstable();
    (cold, samples[5], calls, bytes)
}

fn main() {
    let runtime = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    if std::env::args().any(|arg| arg == "--probe") {
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 3)])
            .unwrap()
            .try_dual()
            .unwrap();
        let input: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 0.0, 3.0],
            }],
        )
        .unwrap();
        let left = input.left_polar(&[0], &[1]).unwrap();
        let right = input.right_polar(&[0], &[1]).unwrap();
        println!(
            "left_w_diagonal={},left_p_diagonal={},right_wh_diagonal={},right_p_diagonal={},spaces_preserved={}",
            left.w.dense_data().is_err() && left.w.diagview().is_ok(),
            left.p.dense_data().is_err() && left.p.diagview().is_ok(),
            right.wh.dense_data().is_err() && right.wh.diagview().is_ok(),
            right.p.dense_data().is_err() && right.p.diagview().is_ok(),
            left.w.domain() == input.domain() && right.p.codomain() == input.codomain()
        );
        return;
    }
    println!(
        "family,dtype,sectors,k,dual,storage,operation,setup_ns,first_ns,median_ns,alloc_calls,alloc_bytes"
    );
    macro_rules! family {
        ($dtype:ty, $name:literal, $value:expr, $family:literal, $rule:expr, $sector:expr) => {
            for (sectors, k, dual) in [(1, 8, false), (4, 64, true)] {
                for storage in ["diagonal", "dense"] {
                    let setup = Instant::now();
                    let mut leg = GradedSpace::try_new(Arc::new($rule),
                        (0..sectors).map(|c| (($sector)(c, false), k))).unwrap();
                    if dual { leg = leg.try_dual().unwrap(); }
                    let mut input: TensorMap<_, $dtype> = TensorMap::diagonal(&runtime, &leg,
                        (0..sectors).map(|c| SectorSpectrum {
                            sector: ($sector)(c, dual),
                            values: (0..k).map($value).collect(),
                        })).unwrap();
                    if storage == "dense" { input = input.materialize().unwrap(); }
                    let setup_ns = setup.elapsed().as_nanos();
                    for op in ["left_polar", "right_polar"] {
                        let (cold, median, calls, bytes) = measure(|| {
                            match op {
                                "left_polar" => { black_box(input.left_polar(&[0], &[1]).unwrap()); },
                                _ => { black_box(input.right_polar(&[0], &[1]).unwrap()); },
                            }
                        });
                        println!("{},{},{sectors},{k},{dual},{storage},{op},{setup_ns},{cold},{median},{calls},{bytes}", $family, $name);
                    }
                }
            }
        }
    }
    macro_rules! run {
        ($dtype:ty, $name:literal, $value:expr) => {
            family!($dtype, $name, $value, "U1", U1FusionRule, |c: i32, dual| {
                U1Irrep::new(if dual { -c } else { c })
            });
            family!(
                $dtype,
                $name,
                $value,
                "SU2",
                SU2FusionRule,
                |c: i32, _dual| SU2Irrep::from_twice_spin(c as usize)
            );
            family!(
                $dtype,
                $name,
                $value,
                "U1xF",
                symtn::sector::ProductFusionRule::<_, _>::try_new(
                    U1FusionRule,
                    FermionParityFusionRule
                )
                .unwrap(),
                |c: i32, dual| product_sector(
                    U1Irrep::new(if dual { -c } else { c }),
                    if c % 2 == 0 {
                        Z2Irrep::EVEN
                    } else {
                        Z2Irrep::ODD
                    }
                )
            );
        };
    }
    run!(f32, "f32", |i| if i % 3 == 0 {
        0.0
    } else {
        -(i as f32 + 1.0)
    });
    run!(f64, "f64", |i| if i % 3 == 0 {
        0.0
    } else {
        -(i as f64 + 1.0)
    });
    run!(Complex32, "c32", |i| Complex32::new(
        i as f32 - 3.0,
        0.5 * i as f32
    ));
    run!(Complex64, "c64", |i| Complex64::new(
        i as f64 - 3.0,
        0.5 * i as f64
    ));
}
