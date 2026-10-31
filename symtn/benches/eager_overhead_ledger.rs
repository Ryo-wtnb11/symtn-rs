//! Per-call cost of the public eager primitives on small many-block tensors
//! (#1313): warm per-call minimum/median and allocation calls/bytes for
//! `compose`, `contract`, `permute`, `repartition`, `qr_compact`,
//! `svd_compact`, `lq_compact`, `left_null`, `eigh_full` (of the Hermitian
//! `square' * square`), `restrict_leg`, `scale`, `axpby`, and `norm` over U(1), fZ2×U(1), and SU(2),
//! `f64` and `Complex64`, ranks 2–5. Four lazy-adjoint rows follow them:
//! `add_adjoint` (`a.adjoint() + b` on the adjoint space), `adjoint_data`
//! (a fresh `a.adjoint()`, materialized and read with `dense_data()`), and
//! `contract_conj` / `compose_conj` (`contract` / `compose` with the lazy
//! `a.adjoint()` as the conjugated left operand).
//!
//! ```text
//! cargo bench -p symtn --bench eager_overhead_ledger -- [filter ...]
//! ```
//!
//! Each `filter` must equal one field of `symmetry,dtype,case,op` for a row
//! to run (`U1 c64 qr_compact`). `LEDGER_THREADS` selects the thread layout:
//!
//! - `one` (default): `Runtime::builder()`, a runtime CPU
//!   pool of one worker for all of its Host work;
//! - `default`: no thread configuration at all (the pool uses the process's
//!   available parallelism).
//!
//! `LEDGER_SAMPLE=<path>` with exactly one selected row loops that operation
//! for `LEDGER_SECONDS` (default 4) while `/usr/bin/sample` (macOS) records
//! the process call tree into `<path>`; `benchmarks/eager_overhead_phases.py`
//! folds that tree into phases. No production code carries a hook for this.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use symtn::typed::HermitianTol;

use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{Complex64, ContractSpec, GradedSpace, LegSelection, Runtime, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// One tensor shape: `nc` codomain and `nd` domain copies of a leg with
/// `sectors` sectors of degeneracy `deg` each.
struct Case {
    name: &'static str,
    nc: usize,
    nd: usize,
    sectors: usize,
    deg: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "r2_s8_d2",
        nc: 1,
        nd: 1,
        sectors: 8,
        deg: 2,
    },
    Case {
        name: "r2_s8_d16",
        nc: 1,
        nd: 1,
        sectors: 8,
        deg: 16,
    },
    Case {
        name: "r3_s4_d4",
        nc: 2,
        nd: 1,
        sectors: 4,
        deg: 4,
    },
    Case {
        name: "r4_s3_d4",
        nc: 2,
        nd: 2,
        sectors: 3,
        deg: 4,
    },
    Case {
        name: "r5_s2_d2",
        nc: 3,
        nd: 2,
        sectors: 2,
        deg: 2,
    },
];

const WARMUP: Duration = Duration::from_millis(30);
const BUDGET: Duration = Duration::from_millis(250);
const MAX_SAMPLES: usize = 4000;

struct Config {
    threads: String,
    filters: Vec<String>,
    sample: Option<(String, u64)>,
}

impl Config {
    fn selects(&self, key: &str) -> bool {
        self.filters
            .iter()
            .all(|filter| key.split(',').any(|field| field == filter))
    }
}

/// Times warm calls in batches of at least ~10 µs (the host timer ticks at
/// ~42 ns); returns `(calls, min_ns, median_ns)` per call over the batches.
fn time_calls<T>(mut call: impl FnMut() -> T) -> (usize, f64, f64) {
    let warm_start = Instant::now();
    let mut warm_calls = 0u32;
    while warm_start.elapsed() < WARMUP {
        black_box(call());
        warm_calls += 1;
    }
    let per_call = WARMUP.as_nanos() as f64 / f64::from(warm_calls.max(1));
    let batch = ((10_000.0 / per_call).ceil() as usize).max(1);
    let mut samples = Vec::with_capacity(MAX_SAMPLES);
    let start = Instant::now();
    while samples.len() < MAX_SAMPLES && (samples.len() < 50 || start.elapsed() < BUDGET) {
        let t = Instant::now();
        for _ in 0..batch {
            black_box(call());
        }
        samples.push(t.elapsed().as_nanos() as f64 / batch as f64);
    }
    samples.sort_unstable_by(f64::total_cmp);
    (
        samples.len() * batch,
        samples[0],
        samples[samples.len() / 2],
    )
}

/// Allocation calls and requested bytes of one warm call on this thread,
/// including the drop of its result.
fn count_allocations<T>(mut call: impl FnMut() -> T) -> (usize, usize) {
    black_box(call());
    let ((), allocs) = counting_alloc::measure(|| drop(black_box(call())));
    (allocs.calls as usize, allocs.bytes as usize)
}

/// Loops `call` while `/usr/bin/sample` records this process. Never inlined:
/// the phase folder anchors on this frame.
#[inline(never)]
fn sample_calls<T>(path: &str, seconds: u64, mut call: impl FnMut() -> T) {
    let warm_start = Instant::now();
    while warm_start.elapsed() < WARMUP {
        black_box(call());
    }
    let mut sampler = std::process::Command::new("/usr/bin/sample")
        .args([
            &std::process::id().to_string(),
            &seconds.to_string(),
            "1",
            "-mayDie",
            "-file",
            path,
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn /usr/bin/sample");
    let start = Instant::now();
    let mut calls = 0u64;
    while start.elapsed() < Duration::from_secs(seconds + 1) {
        black_box(call());
        calls += 1;
    }
    sampler.wait().expect("wait for /usr/bin/sample");
    eprintln!("sampled {calls} calls into {path}");
}

fn run_op<T>(config: &Config, prefix: &str, op: &str, call: impl FnMut() -> T) {
    let key = format!("{prefix},{op}");
    if !config.selects(&key) {
        return;
    }
    if let Some((path, seconds)) = &config.sample {
        sample_calls(path, *seconds, call);
        return;
    }
    let mut call = call;
    let (iterations, min_ns, median_ns) = time_calls(&mut call);
    let (alloc_calls, alloc_bytes) = count_allocations(&mut call);
    println!(
        "{},{key},{iterations},{min_ns:.1},{median_ns:.1},{alloc_calls},{alloc_bytes}",
        config.threads
    );
}

/// Runs every case and operation for one rule and dtype. A macro rather than
/// a generic function: the public methods' dispatch bounds differ per method
/// and naming them all would only restate the facade.
macro_rules! ledger {
    ($engine:ident, $config:expr, $runtime:expr, $symmetry:literal, $provider:expr, $dtype:ty, $dname:literal,
     $labels:expr, $one:expr) => {{
        for case in CASES {
            let sectors: Vec<_> = ($labels)(case.sectors);
            let leg = GradedSpace::try_new(
                std::sync::Arc::new($provider),
                sectors.iter().map(|s| (s.clone(), case.deg)),
            )?;
            let codomain = vec![&leg; case.nc];
            let domain = vec![&leg; case.nd];
            let a = TensorMap::<_, $dtype>::rand_with_seed(
                $engine,
                $runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                1,
            )?;
            let a2 = TensorMap::<_, $dtype>::rand_with_seed(
                $engine,
                $runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                2,
            )?;
            let square = TensorMap::<_, $dtype>::rand_with_seed(
                $engine,
                $runtime,
                domain.iter().copied(),
                domain.iter().copied(),
                3,
            )?;
            let matrix =
                TensorMap::<_, $dtype>::rand_with_seed($engine, $runtime, [&leg], [&leg], 4)?;
            let on_adjoint = TensorMap::<_, $dtype>::rand_with_seed(
                $engine,
                $runtime,
                domain.iter().copied(),
                codomain.iter().copied(),
                5,
            )?;
            let hermitian = square.adjoint($engine)?.compose($engine, &square)?;
            let lazy = a.adjoint($engine)?;
            // Leg roles of `a` (`codomain <- domain`) and of the square
            // `domain <- domain` matrices: each factorization acts on the
            // tensor's own split.
            let rows: Vec<usize> = (0..case.nc).collect();
            let cols: Vec<usize> = (case.nc..case.nc + case.nd).collect();
            let square_rows: Vec<usize> = (0..case.nd).collect();
            let square_cols: Vec<usize> = (case.nd..2 * case.nd).collect();
            let selection = LegSelection::try_new(
                &leg,
                sectors.iter().map(|s| (s.clone(), 0..case.deg.div_ceil(2))),
            )?;
            let rank = case.nc + case.nd;
            let rotated: Vec<usize> = (1..rank).chain([0]).collect();
            let (perm_codomain, perm_domain) = rotated.split_at(case.nc);
            let repartition_to = if case.nd >= 2 {
                case.nc + 1
            } else {
                case.nc - 1
            };
            let open: Vec<usize> = (0..rank).collect();
            let (open_codomain, open_domain) = open.split_at(rank - 1);
            let one_leg = ContractSpec {
                lhs: &[0],
                rhs: &[1],
                codomain: open_codomain,
                domain: open_domain,
            };
            let mut coupled: Vec<_> = (0..a.subblock_count())
                .map(|i| a.subblock_fusion_trees(i).map(|t| t.coupled().clone()))
                .collect::<Result<_, _>>()?;
            coupled.sort();
            coupled.dedup();
            let prefix = format!(
                "{},{},{},{},{},{},{}",
                $symmetry,
                $dname,
                case.name,
                rank,
                a.subblock_count(),
                coupled.len(),
                a.dense_data().unwrap().len()
            );
            let one: $dtype = $one;
            let config: &Config = $config;
            // `black_box(&a)`: the receiver is loop-invariant, so an inlined pure
            // operation (`norm`) could otherwise be hoisted out of the timed loop.
            run_op(config, &prefix, "compose", || {
                black_box(&a).compose($engine, &square).unwrap()
            });
            run_op(config, &prefix, "contract", || {
                black_box(&a).contract($engine, &matrix, &one_leg).unwrap()
            });
            run_op(config, &prefix, "permute", || {
                black_box(&a)
                    .permute($engine, perm_codomain, perm_domain)
                    .unwrap()
            });
            run_op(config, &prefix, "repartition", || {
                black_box(&a).repartition($engine, repartition_to).unwrap()
            });
            run_op(config, &prefix, "qr_compact", || {
                black_box(&a).qr_compact($engine, &rows, &cols).unwrap()
            });
            run_op(config, &prefix, "svd_compact", || {
                black_box(&a).svd_compact($engine, &rows, &cols).unwrap()
            });
            run_op(config, &prefix, "lq_compact", || {
                black_box(&a).lq_compact($engine, &rows, &cols).unwrap()
            });
            run_op(config, &prefix, "left_null", || {
                black_box(&a).left_null($engine, &rows, &cols).unwrap()
            });
            run_op(config, &prefix, "eigh_full", || {
                black_box(&hermitian)
                    .eigh_full($engine, &square_rows, &square_cols, HermitianTol::DEFAULT)
                    .unwrap()
            });
            run_op(config, &prefix, "restrict_leg", || {
                black_box(&a)
                    .restrict_leg($engine, &[(0, &selection)])
                    .unwrap()
            });
            run_op(config, &prefix, "scale", || {
                black_box(&a).scale($engine, one + one).unwrap()
            });
            run_op(config, &prefix, "add", || {
                black_box(&a).axpby($engine, one, &a2, one).unwrap()
            });
            run_op(config, &prefix, "norm", || {
                black_box(&a).norm($engine, 2.0).unwrap()
            });
            run_op(config, &prefix, "add_adjoint", || {
                black_box(&lazy)
                    .axpby($engine, one, &on_adjoint, one)
                    .unwrap()
            });
            run_op(config, &prefix, "adjoint_data", || {
                black_box(&a)
                    .adjoint($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .len()
            });
            run_op(config, &prefix, "contract_conj", || {
                black_box(&lazy)
                    .contract($engine, &matrix, &one_leg)
                    .unwrap()
            });
            run_op(config, &prefix, "compose_conj", || {
                black_box(&lazy).compose($engine, &a2).unwrap()
            });
        }
    }};
}

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn centered(count: usize) -> impl Iterator<Item = i32> {
    let low = -((count as i32 - 1) / 2);
    (0..count as i32).map(move |i| low + i)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Serial: these rows time per-call overhead and single-thread kernels.
    let engine = &mut symtn::typed::CpuEngine::serial();
    let threads = std::env::var("LEDGER_THREADS").unwrap_or_else(|_| "one".to_string());
    let builder = match threads.as_str() {
        "one" => Runtime::builder(),
        "default" => Runtime::builder(),
        other => return Err(format!("unknown LEDGER_THREADS={other}").into()),
    };
    let runtime = builder.build()?;
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect();
    let sample = std::env::var("LEDGER_SAMPLE").ok().map(|path| {
        let seconds = std::env::var("LEDGER_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        (path, seconds)
    });
    let config = Config {
        threads,
        filters,
        sample,
    };
    eprintln!("# rayon_global_threads={}", rayon::current_num_threads());
    if config.sample.is_none() {
        println!(
            "threads,symmetry,dtype,case,rank,blocks,coupled,elements,op,iterations,min_ns,median_ns,alloc_calls,alloc_bytes"
        );
    }

    let u1 = |n: usize| centered(n).map(U1Irrep::new).collect::<Vec<_>>();
    let fz2u1 = |n: usize| {
        centered(n)
            .map(|q| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                ProductSector::new(parity, U1Irrep::new(q))
            })
            .collect::<Vec<_>>()
    };
    let su2 = |n: usize| (0..n).map(SU2Irrep::from_twice_spin).collect::<Vec<_>>();
    let fz2u1_rule = || Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();

    ledger!(
        engine,
        &config,
        &runtime,
        "U1",
        U1FusionRule,
        f64,
        "f64",
        u1,
        1.0
    );
    ledger!(
        engine,
        &config,
        &runtime,
        "U1",
        U1FusionRule,
        Complex64,
        "c64",
        u1,
        Complex64::new(1.0, 0.0)
    );
    ledger!(
        engine,
        &config,
        &runtime,
        "fZ2xU1",
        fz2u1_rule(),
        f64,
        "f64",
        fz2u1,
        1.0
    );
    ledger!(
        engine,
        &config,
        &runtime,
        "fZ2xU1",
        fz2u1_rule(),
        Complex64,
        "c64",
        fz2u1,
        Complex64::new(1.0, 0.0)
    );
    ledger!(
        engine,
        &config,
        &runtime,
        "SU2",
        SU2FusionRule,
        f64,
        "f64",
        su2,
        1.0
    );
    ledger!(
        engine,
        &config,
        &runtime,
        "SU2",
        SU2FusionRule,
        Complex64,
        "c64",
        su2,
        Complex64::new(1.0, 0.0)
    );
    Ok(())
}
