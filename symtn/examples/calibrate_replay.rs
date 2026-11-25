//! Offline CPU calibration of the compiled, owned tree-transform replay.
//! Usage: calibrate_replay <profile-path> [width]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::hint::black_box;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;
use symtn::sector::{SectorId, U1Irrep};
use symtn::typed::{CpuEngine, Pool};
use symtn_core::{BlockSpec, BlockStructure, FusionTreePairKey};
use symtn_dense::DefaultDenseExecutor;
use symtn_operations::host_pool::{lane_region, HostPoolSite};
use symtn_operations::{
    try_tree_transform_structure_overwrite_owned_raw, TreeTransformBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace,
};

type Res<T> = Result<T, Box<dyn std::error::Error>>;
const TOLERANCE: f64 = 0.20;

struct Fixture {
    structure: Arc<BlockStructure>,
    transform: TreeTransformStructure<f64>,
    data: Vec<f64>,
    workspace: TreeTransformWorkspace<f64>,
    dense: DefaultDenseExecutor,
    roles: usize,
    elements: usize,
    rank: usize,
}

impl Fixture {
    fn new(roles: usize, elements: usize, rank: usize, coefficient: f64) -> Res<Self> {
        let shape = if rank == 6 {
            vec![4, 4, 4, 4, 4, elements / 1024]
        } else {
            vec![elements, 1]
        };
        roles
            .checked_mul(elements)
            .and_then(|n| n.checked_mul(16))
            .ok_or("fixture size overflow")?;
        let mut blocks = Vec::new();
        for role in 0..roles {
            let charge = SectorId::from(U1Irrep::new(i32::try_from(role)?)?).id();
            let mut legs = vec![SectorId::from(U1Irrep::new(0)?).id(); rank / 2];
            legs[0] = charge;
            let key = FusionTreePairKey::try_pair_from_sector_ids(
                legs.clone(),
                legs,
                charge,
                vec![false; rank / 2],
                vec![false; rank / 2],
                vec![charge; (rank / 2).saturating_sub(2)],
                vec![charge; (rank / 2).saturating_sub(2)],
                vec![1; rank / 2 - 1],
                vec![1; rank / 2 - 1],
            )?;
            blocks.push(BlockSpec::column_major_with_key(
                key.into(),
                shape.clone(),
                role * elements,
            )?);
        }
        let structure = Arc::new(BlockStructure::from_blocks_with_rank(rank, blocks)?);
        let specs: Vec<_> = (0..roles)
            .map(|role| {
                let spec = TreeTransformBlockSpec::single(role, role, coefficient);
                if rank == 6 {
                    spec.with_source_axes([4, 3, 2, 1, 0, 5])
                } else {
                    spec
                }
            })
            .collect();
        let transform = TreeTransformStructure::compile_structures(&structure, &structure, &specs)?;
        assert_eq!(transform.blocks().len(), roles);
        assert!(!transform.has_pack_gemm_scatter_blocks());
        if rank == 6 {
            assert_eq!(
                transform.layout_diagnostics().entry(0).unwrap().dims.len(),
                6
            );
        }
        Ok(Self {
            structure,
            transform,
            data: Vec::new(),
            workspace: TreeTransformWorkspace::default(),
            dense: DefaultDenseExecutor::new(),
            roles,
            elements,
            rank,
        })
    }

    fn seed(&mut self, engine: &mut CpuEngine) -> Res<()> {
        // Inactive replay first-writes each source range on its computing worker.
        let empty = Arc::new(BlockStructure::empty(self.rank));
        let fill = TreeTransformStructure::<f64>::compile_structures(&self.structure, &empty, &[])?;
        self.data = try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut self.dense,
            &mut self.workspace,
            &fill,
            &self.structure,
            &empty,
            self.rank / 2,
            &[],
            1.0,
            None,
        )?
        .ok_or("source fill lacks an owned-overwrite proof")?;
        Ok(())
    }

    fn replay(&mut self, engine: &mut CpuEngine) -> Res<Vec<f64>> {
        Ok(try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut self.dense,
            &mut self.workspace,
            &self.transform,
            &self.structure,
            &self.structure,
            self.rank / 2,
            &self.data,
            1.0,
            None,
        )?
        .ok_or("fixture lacks an owned-overwrite proof")?)
    }

    fn bytes(&self) -> f64 {
        (self.roles * self.elements * 16) as f64
    }
}

#[derive(Clone, Copy)]
struct Samples {
    raw: [f64; 5],
    median: f64,
    spread: f64,
    cold: f64,
}

fn measure(mut call: impl FnMut() -> Res<()>, max_reps: usize) -> Res<Samples> {
    let start = Instant::now();
    call()?;
    let cold = start.elapsed().as_secs_f64();
    let reps = ((0.002 / cold.max(1e-9)) as usize).clamp(1, max_reps);
    let mut raw = [0.0; 5];
    for sample in &mut raw {
        let start = Instant::now();
        for _ in 0..reps {
            call()?;
        }
        *sample = start.elapsed().as_secs_f64() * 1e9 / reps as f64;
    }
    let mut sorted = raw;
    sorted.sort_by(f64::total_cmp);
    Ok(Samples {
        raw,
        median: sorted[2],
        spread: sorted[4] - sorted[0],
        cold,
    })
}

fn print_samples(label: &str, samples: Samples) {
    println!(
        "{label}: batches_ns={:?} median_ns={:.3} spread_max_min_ns={:.3} warmup_s={:.6}",
        samples.raw, samples.median, samples.spread, samples.cold
    );
}

// Two-variable NNLS: test the unconstrained solution and each boundary.
fn fit(rows: &[(f64, f64, f64)]) -> (f64, f64) {
    let (mut xx, mut xy, mut yy, mut xt, mut yt) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y, t) in rows {
        xx += x * x;
        xy += x * y;
        yy += y * y;
        xt += x * t;
        yt += y * t;
    }
    let det = xx * yy - xy * xy;
    let mut candidates = vec![
        (0.0, 0.0),
        ((xt / xx).max(0.0), 0.0),
        (0.0, (yt / yy).max(0.0)),
    ];
    if det > 0.0 {
        let (a, b) = ((xt * yy - yt * xy) / det, (yt * xx - xt * xy) / det);
        if a >= 0.0 && b >= 0.0 {
            candidates.push((a, b));
        }
    }
    candidates
        .into_iter()
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .min_by(|&(a, b), &(c, d)| {
            let error = |a: f64, b: f64| {
                rows.iter()
                    .map(|&(x, y, t)| (a * x + b * y - t).powi(2))
                    .sum::<f64>()
            };
            error(a, b).total_cmp(&error(c, d))
        })
        .unwrap_or((0.0, 0.0))
}

fn cpus(value: &str) -> Option<BTreeSet<usize>> {
    let mut set = BTreeSet::new();
    for part in value.trim().split(',') {
        let mut range = part.split('-');
        let lo: usize = range.next()?.parse().ok()?;
        let hi: usize = range.next().map_or(Some(lo), |v| v.parse().ok())?;
        if range.next().is_some() || hi < lo || hi > 1_000_000 {
            return None;
        }
        set.extend(lo..=hi);
    }
    Some(set)
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(path)
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "unknown".into())
}

fn sysctl(key: &str) -> String {
    Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn host() -> (BTreeMap<String, String>, Option<usize>) {
    let mut h = BTreeMap::new();
    let info = read("/proc/cpuinfo");
    for (key, field) in [("cpu", "model name"), ("isa", "flags")] {
        h.insert(
            format!("host.{key}"),
            info.lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    (k.trim() == field).then(|| v.trim().to_owned())
                })
                .unwrap_or_else(|| {
                    if cfg!(target_os = "macos") {
                        sysctl(if key == "cpu" {
                            "machdep.cpu.brand_string"
                        } else {
                            "hw.optional.arm64"
                        })
                    } else {
                        "unknown".into()
                    }
                }),
        );
    }
    let status = read("/proc/self/status");
    let allowed = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .map(str::trim)
        .unwrap_or("unknown");
    let online = read("/sys/devices/system/cpu/online");
    let online_set = cpus(&online);
    let allowed_set = cpus(allowed);
    h.insert(
        "host.allowed_cpus".into(),
        allowed_set
            .as_ref()
            .map(|s| s.iter().map(usize::to_string).collect::<Vec<_>>().join(","))
            .unwrap_or_else(|| "unknown".into()),
    );
    h.insert(
        "host.logical_cpus".into(),
        online_set
            .as_ref()
            .map(|s| s.len().to_string())
            .unwrap_or_else(|| {
                if cfg!(target_os = "macos") {
                    sysctl("hw.logicalcpu")
                } else {
                    "unknown".into()
                }
            }),
    );
    let mut cores = BTreeSet::new();
    let mut cores_known = true;
    let mut dies_known = true;
    let mut dies: BTreeMap<String, Vec<String>> = BTreeMap::new();
    if let Some(online) = &online_set {
        for cpu in online {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            let package = read(format!("{base}/physical_package_id"));
            let core = read(format!("{base}/core_id"));
            if package.parse::<usize>().is_ok() && core.parse::<usize>().is_ok() {
                cores.insert(format!("{package}:{core}"));
            } else {
                cores_known = false;
            }
            let die = read(format!("{base}/die_id"));
            if package.parse::<usize>().is_ok() && die.parse::<usize>().is_ok() {
                dies.entry(format!("{package}:{die}"))
                    .or_default()
                    .push(cpu.to_string());
            } else {
                dies_known = false;
            }
        }
    }
    h.insert(
        "host.physical_cores".into(),
        if cores.is_empty() || !cores_known {
            if cfg!(target_os = "macos") {
                sysctl("hw.physicalcpu")
            } else {
                "unknown".into()
            }
        } else {
            cores.len().to_string()
        },
    );
    h.insert(
        "host.die_cpu_sets".into(),
        if dies.is_empty() || !dies_known {
            "unknown".into()
        } else {
            dies.values()
                .map(|v| v.join(","))
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    let mut nodes = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name
                .strip_prefix("node")
                .is_some_and(|s| s.parse::<usize>().is_ok())
            {
                nodes.push((name, read(entry.path().join("cpulist"))));
            }
        }
    }
    nodes.sort();
    h.insert(
        "host.numa_cpu_sets".into(),
        if nodes.is_empty() {
            "unknown".into()
        } else {
            nodes
                .into_iter()
                .map(|(_, s)| s)
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    h.insert("placement".into(), "unknown".into());
    h.insert(
        "pool_pin_strategy".into(),
        if cfg!(target_os = "linux") {
            "threadteam-default-best-effort-pinned"
        } else {
            "threadteam-default-unpinned"
        }
        .into(),
    );
    h.insert(
        "numa_balancing".into(),
        read("/proc/sys/kernel/numa_balancing"),
    );
    h.insert("numactl_policy".into(), "unknown".into());
    // Deduplicate shared caches before summing each CPU's highest data/unified level.
    let mut caches = BTreeMap::new();
    let mut all_caches_known = true;
    if let Some(allowed) = &allowed_set {
        for cpu in allowed {
            let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
            let mut levels = Vec::new();
            if let Ok(entries) = fs::read_dir(base) {
                for entry in entries.flatten() {
                    if !entry
                        .file_name()
                        .to_string_lossy()
                        .strip_prefix("index")
                        .is_some_and(|s| s.parse::<usize>().is_ok())
                    {
                        continue;
                    }
                    let p = entry.path();
                    let kind = read(p.join("type"));
                    if kind == "unknown" {
                        all_caches_known = false;
                    }
                    if kind != "Unified" && kind != "Data" {
                        continue;
                    }
                    let level = read(p.join("level")).parse::<usize>().ok();
                    let size = read(p.join("size"));
                    let bytes = size
                        .strip_suffix('K')
                        .and_then(|s| s.parse::<usize>().ok())
                        .and_then(|n| n.checked_mul(1024))
                        .or_else(|| {
                            size.strip_suffix('M')
                                .and_then(|s| s.parse::<usize>().ok())
                                .and_then(|n| n.checked_mul(1024 * 1024))
                        })
                        .or_else(|| size.parse::<usize>().ok());
                    let shared = read(p.join("shared_cpu_list"));
                    if let (Some(level), Some(bytes), Some(shared)) = (level, bytes, cpus(&shared))
                    {
                        levels.push((level, format!("{level}:{shared:?}"), bytes));
                    } else {
                        all_caches_known = false;
                    }
                }
            }
            if let Some(level) = levels.iter().map(|x| x.0).max() {
                for (_, key, bytes) in levels.into_iter().filter(|x| x.0 == level) {
                    caches.insert(key, bytes);
                }
            } else {
                all_caches_known = false;
            }
        }
    }
    let llc = if caches.is_empty() || !all_caches_known {
        None
    } else {
        caches
            .values()
            .try_fold(0usize, |sum, &n| sum.checked_add(n))
    };
    (h, llc)
}

fn profile_text(fields: &BTreeMap<String, String>) -> Res<String> {
    let mut output = String::new();
    for (key, value) in fields {
        if key.is_empty() || key.contains(['=', '\n', '\r']) || value.contains(['\n', '\r']) {
            return Err("invalid profile field".into());
        }
        output.push_str(&format!("{key} = {value}\n"));
    }
    if output.len() > 64 * 1024 {
        return Err("profile exceeds 64 KiB".into());
    }
    Ok(output)
}

fn save(path: &Path, fields: &BTreeMap<String, String>) -> Res<()> {
    let output = profile_text(fields)?;
    let temp = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .ok_or("output path needs a filename")?
            .to_string_lossy(),
        std::process::id()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| -> io::Result<()> {
        file.write_all(output.as_bytes())?;
        file.sync_all()?;
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    Ok(result?)
}

fn main() -> Res<()> {
    let start = Instant::now();
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|s| s == "--help") {
        println!("Usage: calibrate_replay <profile-path> [width]\nwidth defaults to available parallelism; run a release build.\nFive median batches after one warmup; held-out/stability tolerance = 20%.\nMeasures CPU f64 owned-overwrite structural Singles; no profile loader is installed.");
        return Ok(());
    }
    if args.is_empty() || args.len() > 2 {
        return Err("usage: calibrate_replay <profile-path> [width]".into());
    }
    let width = if let Some(arg) = args.get(1) {
        arg.to_str()
            .ok_or("width must be UTF-8")?
            .parse::<usize>()?
    } else {
        std::thread::available_parallelism()?.get()
    };
    if width == 0 || width > u16::MAX as usize {
        return Err("width must be in 1..=65535".into());
    }
    let (mut profile, llc) = host();
    println!("Compiled replay calibration: width={width}, cpu-faer, f64 owned-overwrite copy; tolerance=20%");
    for (key, value) in &profile {
        println!("{key} = {value}");
    }
    println!("LLC aggregate bytes = {}; per-q engines use their whole budget; actual replay fan-out remains production-selected.", llc.map_or("unknown".into(),|n|n.to_string()));
    println!("Bandwidth source first touch: inactive owned production replay; output first touch: owned Single replay. Participant CPU mapping is unknown; bandwidth is not portable placement wisdom.");
    let revision = Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    for (k, v) in [
        ("format_version", "1"),
        ("model_version", "1"),
        ("replay_abi", "symtn-replay-v2"),
        ("dispatch_abi", "threadteam-0.5.1"),
        ("class", "f64-owned-overwrite-copy"),
        ("adapter", "StridedHostKernelAdapter-owned-writer"),
        ("reference_id", "unknown"),
        ("bandwidth_regime", "unknown"),
        ("c_ns_per_application", "unknown"),
    ] {
        profile.insert(k.into(), v.into());
    }
    profile.insert("producer_revision".into(), "unknown".into());
    profile.insert("observed_checkout_revision".into(), revision.clone());
    profile.insert(
        "producer_build".into(),
        format!(
            "{}-{}-cpu-faer;compiler-and-flags-unknown",
            std::env::consts::ARCH,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        ),
    );
    profile.insert("pool_width".into(), width.to_string());
    println!(
        "observed_checkout_revision = {revision}; producer_revision = unknown; producer_build = {}",
        profile["producer_build"]
    );

    println!(
        "host identity section cold_s={:.6}",
        start.elapsed().as_secs_f64()
    );
    let mut serial = CpuEngine::serial();
    let section = Instant::now();
    let mut empty = Fixture::new(0, 0, 2, 1.0)?;
    let setup_s = section.elapsed().as_secs_f64();
    let baseline = measure(
        || {
            black_box(empty.replay(&mut serial)?);
            Ok(())
        },
        1000,
    )?;
    print_samples("no-item serial engine entry", baseline);
    let mut serial_cold_s = setup_s + baseline.cold;
    let mut accepted = [true, true];
    let mut rows = [Vec::new(), Vec::new()];
    let mut pairs = Vec::new();
    let mut payload_peak = 0usize;
    for roles in [1, 16, 64] {
        for elements in [16, 64, 256, 4096] {
            let mut samples = Vec::new();
            for (class, coefficient) in [1.0, -0.75].into_iter().enumerate() {
                let cold = Instant::now();
                let mut fixture = Fixture::new(roles, elements, 2, coefficient)?;
                fixture.data = (0..roles * elements)
                    .map(|i| (i % 97) as f64 / 97.0)
                    .collect();
                payload_peak = payload_peak.max(roles * elements * 16);
                let setup_s = cold.elapsed().as_secs_f64();
                let time = measure(
                    || {
                        black_box(fixture.replay(&mut serial)?);
                        Ok(())
                    },
                    1000,
                )?;
                print_samples(
                    &format!(
                        "serial map={} roles={roles} elements_per_role={elements} logical_bytes={}",
                        if class == 0 {
                            "identity"
                        } else {
                            "structural(-0.75)"
                        },
                        fixture.bytes()
                    ),
                    time,
                );
                serial_cold_s += setup_s + time.cold;
                let residual = time.median - baseline.median;
                let resolved = residual > time.spread + baseline.spread;
                accepted[class] &= resolved;
                println!("serial class={class} signed_residual_ns={residual:.3} resolved={resolved} setup_s={setup_s:.6}");
                rows[class].push((roles as f64, fixture.bytes(), residual));
                samples.push(time);
            }
            pairs.push((
                roles * elements,
                samples[1].median - samples[0].median,
                samples[0].spread + samples[1].spread,
            ));
        }
    }
    let copy_fit = fit(&rows[0]);
    let scale_fit = fit(&rows[1]);
    println!("fit identity: a_ns={:.6} b_ns_per_byte={:.9}; structural(-0.75): a_ns={:.6} b_ns_per_byte={:.9}",copy_fit.0,copy_fit.1,scale_fit.0,scale_fit.1);
    for rank in [2, 6] {
        for (class, coefficient) in [1.0, -0.75].into_iter().enumerate() {
            let cold = Instant::now();
            let mut held = Fixture::new(1, 65536, rank, coefficient)?;
            held.data = (0..65536).map(|i| (i % 97) as f64 / 97.0).collect();
            let setup_s = cold.elapsed().as_secs_f64();
            let s = measure(
                || {
                    black_box(held.replay(&mut serial)?);
                    Ok(())
                },
                100,
            )?;
            print_samples(
                &format!("held-out map={class} roles=1 elements_per_role=65536 rank={rank}"),
                s,
            );
            serial_cold_s += setup_s + s.cold;
            println!(
                "held-out normalized_dims={:?}",
                held.transform.layout_diagnostics().entry(0).unwrap().dims
            );
            let (a, b) = if class == 0 { copy_fit } else { scale_fit };
            let predicted = baseline.median + a + b * held.bytes();
            let error = (predicted - s.median).abs() / s.median;
            accepted[class] &= error.is_finite() && error <= TOLERANCE;
            println!(
                "held-out prediction_ns={predicted:.3} relative_error={error:.6} accepted={}",
                error <= TOLERANCE
            );
        }
    }
    let denominator = pairs
        .iter()
        .map(|&(n, _, _)| (n as f64).powi(2))
        .sum::<f64>();
    let c = pairs.iter().map(|&(n, d, _)| n as f64 * d).sum::<f64>() / denominator;
    let c_identified = accepted.into_iter().all(|v| v)
        && c.is_finite()
        && c > 0.0
        && pairs
            .iter()
            .all(|&(n, d, spread)| d > spread && (d - c * n as f64).abs() <= TOLERANCE * d);
    if c_identified {
        println!("c_ns_per_application={c:.9} (paired identical-layout controlled structural coefficients)");
    } else {
        println!(
            "c not identified; b reported separately per identity/structural(-0.75) map class"
        );
    }
    for (key, value) in [("a_ns", copy_fit.0), ("b_ns_per_byte", copy_fit.1)] {
        profile.insert(
            key.into(),
            if accepted[0] {
                format!("{value:.12e}")
            } else {
                "unknown".into()
            },
        );
        profile.insert(format!("measured.{key}"), format!("{value:.12e}"));
    }
    if c_identified {
        profile.insert("c_ns_per_application".into(), format!("{c:.12e}"));
    }
    profile.insert(
        "fit_status".into(),
        if accepted[0] {
            "held-out accepted; no trusted campaign reference"
        } else {
            "rejected: unresolved training residual or held-out error exceeds 20%"
        }
        .into(),
    );
    println!(
        "map fit acceptance: identity={} structural(-0.75)={}",
        accepted[0], accepted[1]
    );
    profile.insert(
        "measured.structural.a_ns".into(),
        format!("{:.12e}", scale_fit.0),
    );
    profile.insert(
        "measured.structural.b_ns_per_byte".into(),
        format!("{:.12e}", scale_fit.1),
    );
    profile.insert(
        "structural_fit_status".into(),
        if accepted[1] {
            "held-out accepted; no trusted campaign reference"
        } else {
            "rejected: unresolved training residual or held-out error exceeds 20%"
        }
        .into(),
    );
    println!(
        "serial fit section cold_s={serial_cold_s:.6} total_s={:.6}",
        section.elapsed().as_secs_f64()
    );

    let ladder: Vec<_> = std::iter::successors(Some(1usize), |&q| {
        (q < width).then(|| q.saturating_mul(2).min(width))
    })
    .collect();
    let mut dispatch_widths = Vec::new();
    let mut dispatch = Vec::new();
    let mut bandwidth_widths = Vec::new();
    let mut bandwidth = Vec::new();
    let per_array = llc
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(8));
    let mut all_bandwidth_stable = llc.is_some();
    let mut narrow_reference = None;
    let mut working_set = None;
    for &q in &ladder {
        let section = Instant::now();
        let mut engine = if q == 1 {
            CpuEngine::serial()
        } else {
            CpuEngine::new(Pool::builder().threads(q).build()?)
        };
        let mut dispatch_cold_s = section.elapsed().as_secs_f64();
        if q == 1 {
            println!("D(1) = 0 ns (serial engine)");
        } else {
            let (s, table) =
                lane_region(&mut engine, HostPoolSite::Replay, |mut exec| -> Res<_> {
                    let table = exec
                        .token_ref()
                        .dispatch_cost(NonZeroUsize::new(q).unwrap())
                        .as_secs_f64()
                        * 1e9;
                    let s = if table == 0.0 {
                        None
                    } else {
                        Some(measure(
                            || {
                                exec.reborrow().into_token().fork(q, |_, _| {});
                                Ok(())
                            },
                            1000,
                        )?)
                    };
                    Ok((s, table))
                })??;
            let value = if let Some(s) = s {
                dispatch_cold_s += s.cold;
                print_samples(&format!("empty-fork q={q} table_ns={table:.3}"), s);
                let accepted = s.median >= table / 2.0 && s.median < 20.0 * table;
                println!("D({q}) measured_ns={:.3} accepted={accepted}; rejected samples retain the unmodified table",s.median);
                if accepted {
                    s.median
                } else {
                    table
                }
            } else {
                println!("D({q}) = 0 ns; zero table override preserved without measurement");
                0.0
            };
            dispatch_widths.push(q.to_string());
            dispatch.push(format!("{value:.9}"));
        }
        println!(
            "dispatch section q={q} cold_s={dispatch_cold_s:.6} total_s={:.6}",
            section.elapsed().as_secs_f64()
        );
        let section = Instant::now();
        let mut bandwidth_cold_s = 0.0;
        for elements in [4096, 65536] {
            let cold = Instant::now();
            let mut fixture = Fixture::new(64, elements, 2, 1.0)?;
            fixture.seed(&mut engine)?;
            payload_peak = payload_peak.max(64 * elements * 16);
            let setup_s = cold.elapsed().as_secs_f64();
            let s = measure(
                || {
                    black_box(fixture.replay(&mut engine)?);
                    Ok(())
                },
                5,
            )?;
            bandwidth_cold_s += setup_s + s.cold;
            print_samples(
                &format!("cache replay q={q} roles=64 elements_per_role={elements}"),
                s,
            );
        }
        if let Some(per_array) = per_array {
            let roles = width.max(64);
            let base_elements = per_array
                .checked_add(7)
                .map(|n| n / 8)
                .and_then(|n| n.checked_add(roles - 1))
                .map(|n| n / roles)
                .ok_or("bandwidth fixture size overflow")?;
            let mut rates = Vec::new();
            for multiplier in [1, 2] {
                let elements = base_elements
                    .checked_mul(multiplier)
                    .ok_or("bandwidth fixture size overflow")?;
                let cold = Instant::now();
                let mut fixture = Fixture::new(roles, elements, 2, 1.0)?;
                fixture.seed(&mut engine)?;
                payload_peak = payload_peak.max(roles * elements * 16);
                let setup_s = cold.elapsed().as_secs_f64();
                let s = measure(
                    || {
                        black_box(fixture.replay(&mut engine)?);
                        Ok(())
                    },
                    1,
                )?;
                print_samples(&format!("DRAM candidate q={q} roles={roles} elements_per_role={elements} bytes_per_array={}",roles*elements*8),s);
                bandwidth_cold_s += setup_s + s.cold;
                let rate = fixture.bytes() / s.median;
                println!("bandwidth q={q} bytes_per_ns={rate:.6} (effective replay, no setup subtraction)");
                rates.push(rate);
                if multiplier == 2 {
                    working_set = Some(roles * elements * 8);
                    if q == 1 {
                        narrow_reference = Some(fixture.bytes() / s.median);
                    }
                    if q == width {
                        let retest = measure(
                            || {
                                black_box(fixture.replay(&mut serial)?);
                                Ok(())
                            },
                            1,
                        )?;
                        bandwidth_cold_s += retest.cold;
                        print_samples(
                            "narrow serial DRAM retest on widest-engine-first-touched source",
                            retest,
                        );
                        let rate = fixture.bytes() / retest.median;
                        let stable = narrow_reference
                            .is_some_and(|first: f64| (rate / first - 1.0).abs() <= TOLERANCE);
                        all_bandwidth_stable &= stable;
                        println!("narrow_retest_bytes_per_ns={rate:.6} stable={stable} (includes placement and cache changes)");
                    }
                }
            }
            let stable = rates.iter().all(|r| r.is_finite() && *r > 0.0)
                && (rates[0] / rates[1] - 1.0).abs() <= TOLERANCE;
            println!("bandwidth q={q} twice_size_stable={stable}; placement verification unknown");
            all_bandwidth_stable &= stable;
            bandwidth_widths.push(q.to_string());
            bandwidth.push(if rates[1].is_finite() && rates[1] > 0.0 {
                format!("{:.9}", rates[1])
            } else {
                "unknown".into()
            });
        } else {
            println!("B({q}) unavailable: aggregate LLC size unknown");
        }
        println!(
            "bandwidth section q={q} cold_s={bandwidth_cold_s:.6} total_s={:.6}",
            section.elapsed().as_secs_f64()
        );
    }
    profile.insert("dispatch_widths".into(), dispatch_widths.join(","));
    profile.insert("dispatch_team_ns".into(), dispatch.join(","));
    profile.insert(
        "measured.bandwidth_widths".into(),
        bandwidth_widths.join(","),
    );
    profile.insert(
        "measured.bandwidth_bytes_per_ns".into(),
        bandwidth.join(","),
    );
    profile.insert(
        "bandwidth_status".into(),
        format!(
            "{}; participant placement unknown; unusable for offline loading",
            if all_bandwidth_stable {
                "size/retest stable"
            } else {
                "rejected or unavailable"
            }
        ),
    );
    profile.insert(
        "working_set_bytes_per_array".into(),
        working_set.map_or("unknown".into(), |n: usize| n.to_string()),
    );
    profile.insert("status".into(), "incomplete, base fallback".into());
    profile.insert("payload_peak_bytes".into(), payload_peak.to_string());
    let peak = read("/proc/self/status")
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .map(str::trim)
        .map(str::to_owned)
        .unwrap_or_else(|| "unknown".into());
    println!("peak_process_rss={peak}; peak_simultaneous_payload_bytes={payload_peak} (excludes metadata/pool/allocator retention)");
    println!("incomplete, base fallback: initial measurements have no trusted campaign reference; offline placement identity remains unknown");
    save(Path::new(&args[0]), &profile)?;
    println!(
        "profile saved: {}; total_runtime_s={:.6}",
        Path::new(&args[0]).display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonnegative_fit_handles_interior_and_boundary_optima() {
        // Purpose: recover the least-squares solution subject to nonnegative constants.
        for (rows, expected) in [
            (
                &[(1.0, 1.0, 5.0), (1.0, 2.0, 8.0), (2.0, 1.0, 7.0)][..],
                (2.0, 3.0),
            ),
            (&[(1.0, 0.0, -2.0), (0.0, 1.0, 3.0)][..], (0.0, 3.0)),
        ] {
            let actual = fit(rows);
            assert!((actual.0 - expected.0).abs() <= 1e-12);
            assert!((actual.1 - expected.1).abs() <= 1e-12);
        }
    }

    #[test]
    fn profile_fields_remain_single_line_and_bounded() {
        // Purpose: emitted profiles preserve key/value boundaries and the format size limit.
        let mut fields = BTreeMap::from([
            ("format_version".into(), "1".into()),
            ("a_ns".into(), "2".into()),
        ]);
        assert_eq!(
            profile_text(&fields).unwrap(),
            "a_ns = 2\nformat_version = 1\n"
        );
        fields.insert("bad".into(), "value\nstatus = complete".into());
        assert!(profile_text(&fields).is_err());
        fields.insert("bad".into(), "x".repeat(64 * 1024));
        assert!(profile_text(&fields).is_err());
    }

    #[test]
    fn cpu_lists_expand_inclusive_ranges_and_reject_reversed_ranges() {
        // Purpose: cache aggregation uses the complete allowed CPU set.
        assert_eq!(cpus("0-2,4").unwrap(), BTreeSet::from([0, 1, 2, 4]));
        assert!(cpus("4-2").is_none());
    }
}
