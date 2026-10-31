//! Serial Host vector operations. Run with an optional substring:
//! `cargo bench -p symtn --bench vector_ops --offline -- U1`.
//! Finite, bounded random inputs; setup and correctness checks are untimed.
//! Allocations include result destruction and count only the calling thread.
use std::{
    hint::black_box,
    ops::{Add, Mul},
    sync::Arc,
    time::{Duration, Instant},
};
use symtn::{
    sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep},
    typed::{Complex64, GradedSpace, Runtime, TensorMap},
};
use symtn_sectors::{TableRule, TableSector};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

// W is f64 for f32 and f64, Complex64 for Complex64: no single-precision sum.
fn dot<D: Copy, W: Copy + Default + Add<Output = W> + Mul<Output = W>, const N: usize>(
    a: &[D],
    b: &[D],
    widen: impl Fn(D) -> W,
    conj: impl Fn(W) -> W,
) -> W {
    assert_eq!(a.len(), b.len());
    let mut sums = [W::default(); N];
    let (aa, a_remainder) = a.as_chunks::<N>();
    let (bb, b_remainder) = b.as_chunks::<N>();
    for (a, b) in aa.iter().zip(bb) {
        for i in 0..N {
            sums[i] = sums[i] + conj(widen(a[i])) * widen(b[i]);
        }
    }
    for (i, (&a, &b)) in a_remainder.iter().zip(b_remainder).enumerate() {
        sums[i] = sums[i] + conj(widen(a)) * widen(b);
    }
    let mut width = N;
    while width > 1 {
        for i in 0..width / 2 {
            sums[i] = sums[2 * i] + sums[2 * i + 1];
        }
        width /= 2;
    }
    sums[0]
}

fn run<T>(filter: &str, prefix: &str, op: &str, elements: usize, mut call: impl FnMut() -> T) {
    let key = format!("{prefix},{op}");
    if !key.contains(filter) {
        return;
    }
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(10) {
        black_box(call());
    }
    let start = Instant::now();
    let mut calls = 0u64;
    while start.elapsed() < Duration::from_millis(200) {
        black_box(call());
        calls += 1;
    }
    let ns = start.elapsed().as_nanos() as f64 / calls as f64;
    let ((), allocs) = counting_alloc::measure(|| drop(black_box(call())));
    println!(
        "{key},{ns:.2},{:.4},{},{}",
        ns / elements as f64,
        allocs.calls,
        allocs.bytes
    );
}

// Like eager_overhead_ledger: facade dispatch bounds differ between operations.
macro_rules! bench {
    ($engine:ident, $runtime:expr, $filter:expr, $case:literal, $rule:expr, $labels:expr, $weights:expr,
     $dtype:ty, $name:literal, $widen:expr, $conj:expr, $narrow:expr, $abs:expr, $one:expr) => {{
        let engine = &mut *$engine;
        for target in [1_000usize, 10_000, 100_000, 1_000_000] {
            let labels = $labels;
            let weights: &[f64] = &$weights;
            let deg = ((target as f64 / labels.len() as f64).sqrt().round() as usize).max(1);
            let leg = GradedSpace::try_new(Arc::new($rule), labels.into_iter().map(|s| (s, deg)))?;
            let a = TensorMap::<_, $dtype>::rand_with_seed(engine, $runtime, [&leg], [&leg], 1)?;
            let b = TensorMap::<_, $dtype>::rand_with_seed(engine, $runtime, [&leg], [&leg], 2)?;
            let x = a.dense_data()?;
            let y = b.dense_data()?;
            assert_eq!(x.len(), weights.len() * deg * deg);
            let prefix = format!("{},{},{target},{}", $case, $name, x.len());
            let weighted = |x: &[$dtype], y: &[$dtype]| {
                if $case == "U1" {
                    return dot::<_, _, 1>(x, y, $widen, $conj);
                }
                x.chunks_exact(deg * deg)
                    .zip(y.chunks_exact(deg * deg))
                    .zip(weights)
                    .map(|((x, y), &w)| dot::<_, _, 1>(x, y, $widen, $conj) * w)
                    .sum::<_>()
            };
            let want = weighted(x, y);
            let got = a.inner(engine, &b)?;
            assert!(
                ($abs)(($widen)(got) - want) <= 2e-5 * (1.0 + ($abs)(want)),
                "inner {prefix}"
            );
            let norm = ($abs)(weighted(x, x)).sqrt();
            assert!((a.norm(engine, 2.0)? - norm).abs() <= 2e-5 * (1.0 + norm));
            macro_rules! lanes {
                ($n:literal) => {{
                    let v = dot::<_, _, $n>(x, y, $widen, $conj);
                    let serial = dot::<_, _, 1>(x, y, $widen, $conj);
                    assert!(($abs)(v - serial) <= 1e-9 * (1.0 + ($abs)(serial)));
                    run(
                        $filter,
                        &prefix,
                        concat!("dot_lanes_", stringify!($n)),
                        x.len(),
                        || dot::<_, _, $n>(black_box(x), black_box(y), $widen, $conj),
                    );
                }};
            }
            lanes!(1);
            lanes!(4);
            lanes!(8);
            lanes!(16);
            run($filter, &prefix, "dot_unweighted", x.len(), || {
                dot::<_, _, 1>(black_box(x), black_box(y), $widen, $conj)
            });
            run($filter, &prefix, "inner_slice", x.len(), || {
                ($narrow)(weighted(black_box(x), black_box(y)))
            });
            run($filter, &prefix, "norm_slice", x.len(), || {
                ($abs)(weighted(black_box(x), black_box(x))).sqrt()
            });
            run($filter, &prefix, "inner", x.len(), || {
                black_box(&a).inner(engine, black_box(&b)).unwrap()
            });
            run($filter, &prefix, "norm", x.len(), || {
                black_box(&a).norm(engine, black_box(2.0)).unwrap()
            });
            let one: $dtype = $one;
            let alpha = one * 0.75;
            let beta = one * 0.25;
            let scaled = a.scale(engine, alpha)?;
            for (&got, &src) in scaled.dense_data()?.iter().zip(x) {
                assert!(($abs)(($widen)(got - alpha * src)) <= 1e-6);
            }
            let mut dst = b.scale(engine, one)?;
            a.axpby_into(engine, &mut dst, alpha, beta)?;
            for ((&got, &a), &b) in dst.dense_data()?.iter().zip(x).zip(y) {
                assert!(($abs)(($widen)(got - (beta * b + alpha * a))) <= 1e-6);
            }
            // Convex updates and unit scaling keep repeated in-place inputs bounded.
            run($filter, &prefix, "axpby_into", x.len(), || {
                black_box(&a)
                    .axpby_into(
                        engine,
                        black_box(&mut dst),
                        black_box(alpha),
                        black_box(beta),
                    )
                    .unwrap()
            });
            let mut buf = y.to_vec();
            run($filter, &prefix, "axpby_slice", x.len(), || {
                let (alpha, beta) = black_box((alpha, beta));
                for (d, &s) in black_box(&mut buf).iter_mut().zip(black_box(x)) {
                    *d = beta * *d + alpha * s;
                }
            });
            run($filter, &prefix, "scale", x.len(), || {
                black_box(&a).scale(engine, black_box(one)).unwrap()
            });
            run($filter, &prefix, "scale_slice", x.len(), || {
                let factor = black_box(one);
                black_box(x).iter().map(|&v| factor * v).collect::<Vec<_>>()
            });
            let before = dst.dense_data()?.to_vec();
            dst.scale_assign(engine, alpha)?;
            for (&got, &src) in dst.dense_data()?.iter().zip(&before) {
                assert!(($abs)(($widen)(got - alpha * src)) <= 1e-6);
            }
            run($filter, &prefix, "scale_assign", x.len(), || {
                black_box(&mut dst)
                    .scale_assign(engine, black_box(one))
                    .unwrap()
            });
            run($filter, &prefix, "scale_assign_slice", x.len(), || {
                let factor = black_box(one);
                // Keep factor on the left, as in the public scale kernel.
                #[allow(clippy::assign_op_pattern)]
                for v in black_box(&mut buf).iter_mut() {
                    *v = factor * *v;
                }
            });
        }
    }};
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serial: these rows time per-call overhead and single-thread kernels.
    let engine = &mut symtn::typed::CpuEngine::serial();
    let filter = std::env::args()
        .skip(1)
        .find(|a| a != "--bench")
        .unwrap_or_default();
    let runtime = Runtime::builder().build()?;
    println!("case,dtype,target,elements,op,ns/call,ns/element,allocations/call,bytes/call");
    macro_rules! cases {
        ($engine:ident, $d:ty, $name:literal, $w:expr, $c:expr, $n:expr, $abs:expr, $one:expr) => {
            bench!(
                $engine,
                &runtime,
                &filter,
                "U1",
                U1FusionRule,
                vec![U1Irrep::new(0), U1Irrep::new(1), U1Irrep::new(2)],
                [1.0, 1.0, 1.0],
                $d,
                $name,
                $w,
                $c,
                $n,
                $abs,
                $one
            );
            bench!(
                $engine,
                &runtime,
                &filter,
                "SU2",
                SU2FusionRule,
                vec![
                    SU2Irrep::from_twice_spin(0),
                    SU2Irrep::from_twice_spin(1),
                    SU2Irrep::from_twice_spin(2)
                ],
                [1.0, 2.0, 3.0],
                $d,
                $name,
                $w,
                $c,
                $n,
                $abs,
                $one
            );
        };
    }
    cases!(
        engine,
        f64,
        "f64",
        |x: f64| x,
        |x: f64| x,
        |x: f64| x,
        |x: f64| x.abs(),
        1.0
    );
    cases!(
        engine,
        f32,
        "f32",
        |x: f32| f64::from(x),
        |x: f64| x,
        |x: f64| x as f32,
        |x: f64| x.abs(),
        1.0
    );
    cases!(
        engine,
        Complex64,
        "Complex64",
        |x: Complex64| x,
        |x: Complex64| x.conj(),
        |x: Complex64| x,
        |x: Complex64| x.norm(),
        Complex64::new(1.0, 0.0)
    );
    bench!(
        engine,
        &runtime,
        &filter,
        "Fibonacci",
        TableRule::fibonacci(),
        vec![TableSector::new(1), TableSector::new(2)],
        [1.0, (1.0 + 5.0_f64.sqrt()) / 2.0],
        Complex64,
        "Complex64",
        |x: Complex64| x,
        |x: Complex64| x.conj(),
        |x: Complex64| x,
        |x: Complex64| x.norm(),
        Complex64::new(1.0, 0.0)
    );
    Ok(())
}
