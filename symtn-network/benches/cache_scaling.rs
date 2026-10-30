//! Multi-thread scaling of the process-global structure caches and the
//! per-Runtime plan cache: warm-hit and cold-miss throughput under an
//! explicit list of scoped threads.
//!
//! ```text
//! cargo bench -p symtn-network --bench cache_scaling -- [filter ...]
//! ```
//!
//! `CACHE_SCALING_THREADS` (default `1,2,4,8,16,32,48,96`) lists the thread
//! counts; `CACHE_SCALING_WARM_MS` (default 500) is the warm window and
//! `CACHE_SCALING_COLD_ROUNDS` (default 3) the cold windows per row. A row
//! runs when every `filter` equals one of its `scenario,symmetry,case,
//! runtime,phase` fields. Every worker holds its own serial engine, so each
//! worker runs its operation on its own thread and the requested count is the
//! thread count.
//!
//! Scenarios:
//! - `raw_front` / `raw_shard` / `raw_spread`: one standalone
//!   `StructureCache<u64, u64>` (4096 resident keys). `raw_front` gets the
//!   last-admitted key (the front path), `raw_shard` gets one other key from
//!   every thread (shard path, one shared `Arc`), `raw_spread` gets
//!   per-thread rotating keys (shard path, distinct `Arc`s).
//!   `raw_publish` admits fresh disjoint keys (the admission path).
//! - `permute`, `contract` (pairwise `TensorMap::contract`) and `network`
//!   (`Network::contract`, two operands) on U(1) and SU(2) rank-4 tensors
//!   with 3 or 6 sectors per leg and degeneracies 2 to 4.
//!   - `warm`: every thread repeats the same operation on equal spaces (the
//!     same cache keys) for the warm window.
//!   - `cold_disjoint`: after `symtn::cache::clear()` (and the plan caches),
//!     each thread runs 64 operations on spaces no other thread uses; the
//!     sector structures and coefficient groups are still shared.
//!   - `cold_shared`: as `cold_disjoint`, but every thread runs the same 64
//!     spaces in the same order (concurrent misses of one key).
//!   - `runtime=own` gives each thread its own Runtime; `runtime=shared`
//!     builds every thread's tensors from one Runtime (shared plan cache and
//!     context pool).
//!
//! `cpu_s` is the process user+system time over the measured window from
//! `/proc/self/stat` (Linux; `nan` elsewhere), `os_threads` the process
//! thread count sampled during it. The cache columns are deltas of
//! `symtn::cache::stats()` over the window; a lost publication race counts
//! in none of them.

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use symtn::cache::{StructureCacheInfo, StructureCacheKind};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, CpuEngine, Engine, GradedSpace, Runtime, TensorMap};
use symtn_core::{core_reset_epoch, StructureCache};
use symtn_network::{clear_plan_cache, Network};

const COLD_OPS: usize = 64;
const RAW_KEYS: u64 = 4096;

struct Config {
    threads: Vec<usize>,
    warm: Duration,
    cold_rounds: usize,
    filters: Vec<String>,
}

impl Config {
    fn selects(&self, key: &str) -> bool {
        self.filters
            .iter()
            .all(|filter| key.split(',').any(|field| field == filter))
    }
}

/// Process CPU seconds (user + system), Linux only.
fn cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return f64::NAN;
    };
    let Some((_, rest)) = stat.rsplit_once(") ") else {
        return f64::NAN;
    };
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let ticks = |i: usize| fields.get(i).and_then(|f| f.parse::<f64>().ok());
    match (ticks(11), ticks(12)) {
        // ponytail: assumes CLK_TCK = 100 (Linux default); the job records `getconf CLK_TCK`.
        (Some(user), Some(system)) => (user + system) / 100.0,
        _ => f64::NAN,
    }
}

fn os_threads() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("Threads:").map(|n| n.trim().to_string()))
        })
        .unwrap_or_else(|| "na".to_string())
}

fn cache_snapshot() -> Vec<StructureCacheInfo> {
    symtn::cache::stats()
}

/// `(hits, misses)` per kind in [`StructureCacheKind::ALL`] order, then total
/// admissions and evictions, as deltas.
fn cache_delta(before: &[StructureCacheInfo], after: &[StructureCacheInfo]) -> String {
    let mut out = Vec::new();
    let (mut admissions, mut evictions) = (0i64, 0i64);
    for kind in StructureCacheKind::ALL {
        let find = |infos: &[StructureCacheInfo]| {
            infos
                .iter()
                .copied()
                .find(|info| info.kind() == kind)
                .expect("every kind reports")
        };
        let (b, a) = (find(before), find(after));
        // A clear inside the window resets the counters; the after value is
        // then the window's count.
        let delta = |x: u64, y: u64| if y >= x { (y - x) as i64 } else { y as i64 };
        out.push(delta(b.hits(), a.hits()).to_string());
        out.push(delta(b.misses(), a.misses()).to_string());
        admissions += delta(b.admissions(), a.admissions());
        evictions += delta(b.evictions(), a.evictions());
    }
    out.push(admissions.to_string());
    out.push(evictions.to_string());
    out.join(",")
}

struct Row<'a> {
    key: &'a str,
    blocks: usize,
    threads: usize,
    per_thread: Vec<u64>,
    wall: f64,
    cpu: f64,
    os_threads: String,
    caches: String,
}

fn print_row(row: Row<'_>) {
    let ops: u64 = row.per_thread.iter().sum();
    let active = row.per_thread.iter().filter(|&&n| n > 0).count();
    let min = row.per_thread.iter().min().copied().unwrap_or(0);
    let max = row.per_thread.iter().max().copied().unwrap_or(0);
    println!(
        "{},{},{},{},{},{ops},{:.6},{:.1},{:.3},{:.2},{min},{max},{}",
        row.key,
        row.blocks,
        row.threads,
        active,
        row.os_threads,
        row.wall,
        ops as f64 / row.wall,
        row.cpu,
        row.cpu / row.wall,
        row.caches
    );
}

/// Runs `threads` workers. Each worker calls `setup` once, then waits at the
/// start barrier and calls `run` (which returns its op count). The main
/// thread calls `between` before releasing them and times the window from the
/// release to the last worker's finish. `stop` ends warm loops.
#[allow(clippy::too_many_arguments)]
fn harness<S>(
    config: &Config,
    key: &str,
    blocks: usize,
    threads: usize,
    setup: impl Fn(usize) -> S + Sync,
    run: impl Fn(usize, &S, &AtomicBool) -> u64 + Sync,
    between: impl FnOnce(),
    warm: bool,
) {
    let ready = Barrier::new(threads + 1);
    let start = Barrier::new(threads + 1);
    let done = Barrier::new(threads + 1);
    let stop = AtomicBool::new(false);
    let counts: Vec<std::sync::Mutex<u64>> =
        (0..threads).map(|_| std::sync::Mutex::new(0)).collect();
    std::thread::scope(|scope| {
        for t in 0..threads {
            let (ready, start, done, stop, counts) = (&ready, &start, &done, &stop, &counts);
            let (setup, run) = (&setup, &run);
            scope.spawn(move || {
                let state = setup(t);
                ready.wait();
                start.wait();
                let n = run(t, &state, stop);
                *counts[t].lock().unwrap() = n;
                done.wait();
                drop(state);
            });
        }
        ready.wait();
        between();
        let before = cache_snapshot();
        let cpu0 = cpu_seconds();
        start.wait();
        let t0 = Instant::now();
        let sampled;
        if warm {
            std::thread::sleep(config.warm / 2);
            sampled = os_threads();
            std::thread::sleep(config.warm - config.warm / 2);
            stop.store(true, Ordering::Relaxed);
        } else {
            sampled = os_threads();
        }
        done.wait();
        let wall = t0.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - cpu0;
        let after = cache_snapshot();
        print_row(Row {
            key,
            blocks,
            threads,
            per_thread: counts.iter().map(|c| *c.lock().unwrap()).collect(),
            wall,
            cpu,
            os_threads: sampled,
            caches: cache_delta(&before, &after),
        });
    });
}

fn warm_loop(stop: &AtomicBool, mut call: impl FnMut()) -> u64 {
    let mut n = 0;
    while !stop.load(Ordering::Relaxed) {
        call();
        n += 1;
    }
    n
}

fn runtime() -> Runtime {
    Runtime::builder().build().expect("runtime")
}

fn raw(config: &Config) {
    let cache: StructureCache<u64, u64> =
        StructureCache::new(StructureCacheKind::SectorStructure, 1 << 30);
    for k in 0..RAW_KEYS {
        cache.publish(&k, Arc::new(k), 64, core_reset_epoch());
    }
    let front = RAW_KEYS - 1;
    let cache = &cache;
    for (scenario, which) in [("raw_front", 0), ("raw_shard", 1), ("raw_spread", 2)] {
        let key = format!("{scenario},raw,k4096,own,warm");
        if !config.selects(&key) {
            continue;
        }
        for &threads in &config.threads {
            harness(
                config,
                &key,
                0,
                threads,
                |_| (),
                |t, _, stop| {
                    let mut i = (t as u64 * 977) % (RAW_KEYS - 1);
                    warm_loop(stop, || {
                        let k = match which {
                            0 => front,
                            1 => 0,
                            _ => {
                                i = (i + 1) % (RAW_KEYS - 1);
                                i
                            }
                        };
                        black_box(cache.get(&k).expect("resident"));
                    })
                },
                || {},
                true,
            );
        }
    }
    let key = "raw_publish,raw,fresh,own,cold".to_string();
    if config.selects(&key) {
        for &threads in &config.threads {
            let fresh = std::sync::atomic::AtomicU64::new(1 << 40);
            harness(
                config,
                &key,
                0,
                threads,
                |_| (),
                |_, _, stop| {
                    warm_loop(stop, || {
                        let k = fresh.fetch_add(1, Ordering::Relaxed);
                        black_box(cache.publish(&k, Arc::new(k), 64, core_reset_epoch()));
                    })
                },
                || {},
                true,
            );
        }
    }
}

/// The degeneracy of each `(leg, sector)` slot for space id `id`: 2, 3 or 4 by
/// the base-3 digits of `id`, so distinct ids (below 3^12) give distinct HomSpaces.
fn degs(id: usize, legs: usize, sectors: usize) -> Vec<Vec<usize>> {
    (0..legs)
        .map(|leg| {
            (0..sectors)
                .map(|s| 2 + (id / 3usize.pow((leg * sectors + s) as u32)) % 3)
                .collect()
        })
        .collect()
}

fn u1_leg(degs: &[usize]) -> GradedSpace<U1FusionRule> {
    let low = -((degs.len() as i32 - 1) / 2);
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        degs.iter()
            .enumerate()
            .map(|(i, &d)| (U1Irrep::new(low + i as i32), d)),
    )
    .expect("U(1) leg")
}

fn su2_leg(degs: &[usize]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        degs.iter()
            .enumerate()
            .map(|(i, &d)| (SU2Irrep::from_twice_spin(i), d)),
    )
    .expect("SU(2) leg")
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Permute,
    Contract,
    Network,
}

fn network() -> Network {
    // a[i j; k l] * b[m; i] -> [m j; k l]
    Network::new(
        [vec!["i", "j", "k", "l"], vec!["m", "i"]],
        [false, false],
        [2, 1],
        ["m", "j", "k", "l"],
        Some(2),
    )
    .expect("network")
}

/// Every operation, case, runtime layout and phase for one symmetry. A macro
/// rather than a generic function: the facade's dispatch bounds differ per
/// method and naming them all would only restate the facade.
macro_rules! ops_rows {
    ($config:expr, $name:literal, $leg:expr) => {{
        let config: &Config = $config;
        // The operands of one space id: a rank-4 `a = (l0 l1 <- l2 l3)` and
        // the matrix `b = (l0 <- l0)`.
        let operands = |rt: &Runtime, id: usize, sectors: usize| {
            let legs: Vec<_> = degs(id, 4, sectors).iter().map(|d| $leg(d)).collect();
            let engine = &mut CpuEngine::serial();
            let a = TensorMap::<_, f64>::rand_with_seed(
                engine,
                rt,
                [&legs[0], &legs[1]],
                [&legs[2], &legs[3]],
                1,
            )
            .expect("a");
            let b = TensorMap::<_, f64>::rand_with_seed(engine, rt, [&legs[0]], [&legs[0]], 2)
                .expect("b");
            (a, b)
        };
        let spec = ContractSpec {
            lhs: &[0],
            rhs: &[1],
            codomain: &[0, 1, 2],
            domain: &[3],
        };
        let apply = |engine: &mut dyn Engine,
                     op: Op,
                     (a, b): &(TensorMap<_, f64>, TensorMap<_, f64>),
                     net: &Network| match op {
            Op::Permute => drop(black_box(
                a.permute(engine, &[1, 2], &[3, 0]).expect("permute"),
            )),
            Op::Contract => drop(black_box(a.contract(engine, b, &spec).expect("contract"))),
            Op::Network => drop(black_box(net.contract(engine, &[a, b]).expect("network"))),
        };
        for (op, op_name) in [
            (Op::Permute, "permute"),
            (Op::Contract, "contract"),
            (Op::Network, "network"),
        ] {
            for sectors in [3usize, 6] {
                let case = format!("s{sectors}");
                let blocks = operands(&runtime(), 0, sectors).0.subblock_count();
                for shared_rt in [false, true] {
                    let rt_name = if shared_rt { "shared" } else { "own" };
                    for phase in ["warm", "cold_disjoint", "cold_shared"] {
                        let key = format!("{op_name},{},{case},{rt_name},{phase}", $name);
                        if !config.selects(&key) {
                            continue;
                        }
                        let warm = phase == "warm";
                        for &threads in &config.threads {
                            let shared = runtime();
                            let ids = |t: usize| -> Vec<usize> {
                                match phase {
                                    "warm" => vec![0],
                                    "cold_shared" => (1..=COLD_OPS).collect(),
                                    _ => (0..COLD_OPS).map(|i| 1 + t * COLD_OPS + i).collect(),
                                }
                            };
                            let setup = |t: usize| {
                                let own = (!shared_rt).then(runtime);
                                let rt = own.as_ref().unwrap_or(&shared);
                                let inputs: Vec<_> = ids(t)
                                    .into_iter()
                                    .map(|id| operands(rt, id, sectors))
                                    .collect();
                                let net = network();
                                let mut engine = CpuEngine::serial();
                                if warm {
                                    for _ in 0..2 {
                                        apply(&mut engine, op, &inputs[0], &net);
                                    }
                                }
                                (own, inputs, net, std::sync::Mutex::new(engine))
                            };
                            // Cold rows run `cold_rounds` windows; each starts
                            // from cleared caches (setup rebuilds the inputs
                            // and own runtimes, so their plan caches are
                            // fresh).
                            let rounds = if warm { 1 } else { config.cold_rounds };
                            for _ in 0..rounds {
                                harness(
                                    config,
                                    &key,
                                    blocks,
                                    threads,
                                    &setup,
                                    |_, (_, inputs, net, engine), stop| {
                                        // Uncontended: each worker locks only its own engine.
                                        let engine = &mut *engine.lock().unwrap();
                                        if warm {
                                            warm_loop(stop, || apply(engine, op, &inputs[0], net))
                                        } else {
                                            for input in inputs {
                                                apply(engine, op, input, net);
                                            }
                                            inputs.len() as u64
                                        }
                                    },
                                    || {
                                        if !warm {
                                            symtn::cache::clear();
                                            clear_plan_cache(&shared);
                                        }
                                    },
                                    warm,
                                );
                            }
                        }
                    }
                }
            }
        }
    }};
}

fn main() {
    let threads = std::env::var("CACHE_SCALING_THREADS")
        .unwrap_or_else(|_| "1,2,4,8,16,32,48,96".to_string())
        .split(',')
        .map(|t| t.trim().parse().expect("thread count"))
        .collect();
    let warm_ms = std::env::var("CACHE_SCALING_WARM_MS")
        .ok()
        .and_then(|ms| ms.parse().ok())
        .unwrap_or(500);
    let filters = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect();
    let config = Config {
        threads,
        warm: Duration::from_millis(warm_ms),
        cold_rounds: std::env::var("CACHE_SCALING_COLD_ROUNDS")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(3),
        filters,
    };
    println!(
        "scenario,symmetry,case,runtime,phase,blocks,threads,active_threads,os_threads,ops,wall_s,ops_per_s,cpu_s,cpu_util,min_thread_ops,max_thread_ops,sector_hits,sector_misses,deg_hits,deg_misses,ctt_hits,ctt_misses,coef_hits,coef_misses,admissions,evictions"
    );
    raw(&config);
    ops_rows!(&config, "U1", u1_leg);
    ops_rows!(&config, "SU2", su2_leg);
}
