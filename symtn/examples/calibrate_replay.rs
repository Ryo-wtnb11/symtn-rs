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
    TreeTransformReplayProfile, TreeTransformStructure, TreeTransformWorkspace,
};

type Res<T> = Result<T, Box<dyn std::error::Error>>;
const TOLERANCE: f64 = 0.20;

struct Fixture {
    structure: Arc<BlockStructure>,
    source: Arc<BlockStructure>,
    transform: TreeTransformStructure<f64>,
    data: Vec<f64>,
    workspace: TreeTransformWorkspace<f64>,
    dense: DefaultDenseExecutor,
    roles: usize,
    elements: usize,
    rank: usize,
}

impl Fixture {
    fn new(roles: usize, shape: &[usize], axes: &[usize], interleaved: bool) -> Res<Self> {
        let rank = shape.len();
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or("fixture size overflow")?;
        roles
            .checked_mul(elements)
            .and_then(|n| n.checked_mul(16))
            .ok_or("fixture size overflow")?;
        let mut source_shape = vec![0; rank];
        for (i, &axis) in axes.iter().enumerate() {
            source_shape[axis] = shape[i];
        }
        let mut destination = Vec::new();
        let mut source = Vec::new();
        for role in 0..roles {
            let zero = SectorId::from(U1Irrep::new(0)?).id();
            let q = i32::try_from(role)?;
            let charge = if interleaved {
                zero
            } else {
                SectorId::from(U1Irrep::new(q)?).id()
            };
            let mut outgoing = vec![zero; rank / 2];
            let mut incoming = outgoing.clone();
            if interleaved {
                outgoing = vec![
                    SectorId::from(U1Irrep::new(q)?).id(),
                    SectorId::from(U1Irrep::new(-q)?).id(),
                ];
            } else {
                outgoing[0] = charge;
                incoming[0] = charge;
            }
            let key = FusionTreePairKey::try_pair_from_sector_ids(
                outgoing,
                incoming,
                charge,
                vec![false; rank / 2],
                vec![false; rank / 2],
                vec![charge; (rank / 2).saturating_sub(2)],
                vec![charge; (rank / 2).saturating_sub(2)],
                vec![1; rank / 2 - 1],
                vec![1; rank / 2 - 1],
            )?;
            if interleaved {
                // Two interleaved matrix row blocks reproduce the small-item leading dimension.
                let strides = vec![1, shape[0], roles * shape[0], roles * elements];
                let offset = role * shape[0];
                destination.push(BlockSpec::with_key(
                    key.clone().into(),
                    shape.to_vec(),
                    strides.clone(),
                    offset,
                )?);
                source.push(BlockSpec::with_key(
                    key.into(),
                    source_shape.clone(),
                    strides,
                    offset,
                )?);
            } else {
                destination.push(BlockSpec::column_major_with_key(
                    key.clone().into(),
                    shape.to_vec(),
                    role * elements,
                )?);
                source.push(BlockSpec::column_major_with_key(
                    key.into(),
                    source_shape.clone(),
                    role * elements,
                )?);
            }
        }
        let structure = Arc::new(BlockStructure::from_blocks_with_rank(rank, destination)?);
        let source = Arc::new(BlockStructure::from_blocks_with_rank(rank, source)?);
        let specs: Vec<_> = (0..roles)
            .map(|role| {
                TreeTransformBlockSpec::single(role, role, 1.0)
                    .with_source_axes(axes.iter().copied())
            })
            .collect();
        let transform = TreeTransformStructure::compile_structures(&structure, &source, &specs)?;
        assert_eq!(transform.blocks().len(), roles);
        assert!(!transform.has_pack_gemm_scatter_blocks());
        Ok(Self {
            structure,
            source,
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

    fn replay(
        &mut self,
        engine: &mut CpuEngine,
        profile: Option<&mut TreeTransformReplayProfile>,
    ) -> Res<Vec<f64>> {
        Ok(try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut self.dense,
            &mut self.workspace,
            &self.transform,
            &self.structure,
            &self.source,
            self.rank / 2,
            &self.data,
            1.0,
            profile,
        )?
        .ok_or("fixture lacks an owned-overwrite proof")?)
    }

    fn phase(&mut self, engine: &mut CpuEngine) -> Res<f64> {
        let mut profile = TreeTransformReplayProfile::default();
        black_box(self.replay(engine, Some(&mut profile))?);
        Ok(profile.single_total.as_secs_f64() * 1e9)
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

fn measure(mut call: impl FnMut() -> Res<Option<f64>>, max_reps: usize) -> Res<Samples> {
    let start = Instant::now();
    call()?;
    let cold = start.elapsed().as_secs_f64();
    let reps = ((0.002 / cold.max(1e-9)) as usize).clamp(1, max_reps);
    let mut raw = [0.0; 5];
    for sample in &mut raw {
        let start = Instant::now();
        let mut phase = None;
        for _ in 0..reps {
            if let Some(ns) = call()? {
                *phase.get_or_insert(0.0) += ns;
            }
        }
        *sample = phase.unwrap_or_else(|| start.elapsed().as_secs_f64() * 1e9) / reps as f64;
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

// Column-scaled, relative least squares; singular designs have no identified constants.
fn fit(rows: &[(Vec<f64>, f64)]) -> Option<Vec<f64>> {
    let n = rows.first()?.0.len();
    let mut scale = vec![0.0_f64; n];
    for (x, t) in rows {
        if x.len() != n || !t.is_finite() || *t <= 0.0 || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        for (s, v) in scale.iter_mut().zip(x) {
            *s = s.max((v / t).abs());
        }
    }
    if n == 0 || scale.contains(&0.0) {
        return None;
    }
    let mut matrix = vec![vec![0.0; n + 1]; n];
    for (x, t) in rows {
        for i in 0..n {
            let xi = x[i] / t / scale[i];
            for j in 0..n {
                matrix[i][j] += xi * x[j] / t / scale[j];
            }
            matrix[i][n] += xi;
        }
    }
    for i in 0..n {
        let pivot = (i..n).max_by(|&a, &b| matrix[a][i].abs().total_cmp(&matrix[b][i].abs()))?;
        matrix.swap(i, pivot);
        let diagonal = matrix[i][i];
        if diagonal.abs() <= 1e-10 {
            return None;
        }
        for j in i..=n {
            matrix[i][j] /= diagonal;
        }
        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = matrix[k][i];
            for j in i..=n {
                let entry = matrix[i][j];
                matrix[k][j] -= factor * entry;
            }
        }
    }
    let constants: Vec<_> = (0..n).map(|i| matrix[i][n] / scale[i]).collect();
    constants.iter().all(|v| v.is_finite()).then_some(constants)
}

fn cache_bytes(value: &str) -> Option<usize> {
    let value = value.trim();
    let (number, multiplier) = match value.as_bytes().last()? {
        b'K' => (&value[..value.len() - 1], 1024usize),
        b'M' => (&value[..value.len() - 1], 1024usize.pow(2)),
        b'G' => (&value[..value.len() - 1], 1024usize.pow(3)),
        _ => (value, 1),
    };
    let bytes = number.parse::<usize>().ok()?.checked_mul(multiplier)?;
    (bytes > 0).then_some(bytes)
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

fn host() -> (
    BTreeMap<String, String>,
    Option<usize>,
    Vec<(String, usize)>,
) {
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
                    let bytes = cache_bytes(&size);
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
    let mut levels = BTreeMap::new();
    let mut cache_complete = true;
    if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu/cpu0/cache") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name
                .strip_prefix("index")
                .is_some_and(|n| n.parse::<usize>().is_ok())
            {
                continue;
            }
            let path = entry.path();
            let level = read(path.join("level"));
            let size = read(path.join("size"));
            let kind = read(path.join("type"));
            for (field, value) in [("level", &level), ("size", &size), ("type", &kind)] {
                h.insert(format!("host.cache.cpu0.{name}.{field}"), value.clone());
            }
            if kind == "Data" || kind == "Unified" {
                if let (Ok(level), Some(bytes)) = (level.parse::<usize>(), cache_bytes(&size)) {
                    levels
                        .entry(level)
                        .and_modify(|n: &mut usize| *n = (*n).max(bytes))
                        .or_insert(bytes);
                } else {
                    cache_complete = false;
                }
            } else if kind != "Instruction" {
                cache_complete = false;
            }
        }
    } else {
        cache_complete = false;
    }
    cache_complete &= !levels.is_empty();
    h.insert(
        "host.cache.cpu0.status".into(),
        if cache_complete {
            "complete"
        } else {
            "unavailable"
        }
        .into(),
    );
    if !cache_complete {
        levels.clear();
    }
    let levels = levels
        .into_iter()
        .map(|(level, bytes)| (format!("L{level}"), bytes))
        .collect();
    (h, llc, levels)
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
        println!("Usage: calibrate_replay <profile-path> [width]\nwidth defaults to available parallelism; run a release build.\nFive median batches after one warmup; held-out/stability tolerance = 20%.\nMeasures CPU f64 identity owned-overwrite Singles, traversal/cache fits and held-out errors; no profile loader is installed.");
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
    let (mut profile, llc, caches) = host();
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
        ("model_version", "2"),
        ("replay_abi", "symtn-replay-v2"),
        ("dispatch_abi", "threadteam-0.5.1"),
        ("class", "f64-owned-overwrite-copy"),
        ("adapter", "StridedHostKernelAdapter-owned-writer"),
        ("reference_id", "unknown"),
        ("bandwidth_regime", "unknown"),
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
    let mut empty = Fixture::new(0, &[0, 1], &[0, 1], false)?;
    let setup_s = section.elapsed().as_secs_f64();
    let baseline = measure(|| Ok(Some(empty.phase(&mut serial)?)), 1000)?;
    print_samples("no-item serial direct phase", baseline);
    let mut serial_cold_s = setup_s + baseline.cold;
    let mut payload_peak = 0usize;
    let mut regimes: Vec<_> = caches.iter().map(|(name, _)| name.clone()).collect();
    regimes.push("memory".into());
    let classes = ["contiguous-run", "strided-inner"];
    let columns = 3 + classes.len() * regimes.len();
    let mut rows = Vec::new();
    let mut training_resolved = !caches.is_empty();
    let mut targets = BTreeSet::new();
    let mut previous = 0usize;
    for &(_, bytes) in &caches {
        if bytes > previous {
            for part in [1, 2] {
                targets.insert(previous + (bytes - previous) / 3 * part);
            }
            previous = bytes;
        }
    }
    if previous > 0 {
        for multiplier in [2, 4] {
            targets.insert(
                previous
                    .checked_mul(multiplier)
                    .ok_or("cache fixture size overflow")?,
            );
        }
    }
    println!("Serial fits use the production direct-phase timer minus its no-item baseline; output allocation/proof/validation are excluded, first stores remain inside items.");
    println!("Raw inner lengths: 1,2,4,16,64,256; normalization removes unit axes and may fuse others: all regressors use printed baked geometry. Planned constants unavailable: admitted_plan requires more than one worker.");
    for bytes in targets {
        let total = bytes
            .checked_add(8191)
            .ok_or("cache fixture size overflow")?
            / 8192
            * 512;
        for roles in [1, 4] {
            let elements = (total / roles / 512).max(1) * 512;
            let working_set = roles * elements * 16;
            let regime = caches
                .iter()
                .position(|(_, size)| working_set <= *size)
                .unwrap_or(caches.len());
            for inner in [1, 2, 4, 16, 64, 256] {
                for axes in [[0, 2, 1, 3], [1, 0, 2, 3]] {
                    let cold = Instant::now();
                    let shape = [inner, 2, elements / (2 * inner), 1];
                    let mut fixture = Fixture::new(roles, &shape, &axes, false)?;
                    fixture.data = (0..roles * elements)
                        .map(|i| (i % 97) as f64 / 97.0)
                        .collect();
                    payload_peak = payload_peak.max(working_set);
                    let diagnostics = fixture.transform.layout_diagnostics();
                    let layout = diagnostics.entry(0).ok_or("missing destination layout")?;
                    let source = layout.partner_strides.ok_or("missing source strides")?;
                    let class = usize::from(layout.block_strides[0] != 1 || source[0] != 1);
                    let runs = roles * elements / layout.dims[0];
                    let setup_s = cold.elapsed().as_secs_f64();
                    let time = measure(|| Ok(Some(fixture.phase(&mut serial)?)), 1000)?;
                    print_samples(&format!("serial k={} regime={} roles={roles} raw_dims={shape:?} axes={axes:?} dims={:?} dst_strides={:?} src_strides={source:?} elements_per_item={elements} runs_per_item={} working_set_bytes={working_set}", classes[class], regimes[regime], layout.dims, layout.block_strides, runs / roles), time);
                    serial_cold_s += setup_s + time.cold;
                    let residual = time.median - baseline.median;
                    let resolved = residual > time.spread + baseline.spread;
                    training_resolved &= resolved;
                    let mut x = vec![0.0; columns];
                    x[0] = roles as f64;
                    x[1 + class] = runs as f64;
                    x[3 + class * regimes.len() + regime] = (roles * elements) as f64;
                    rows.push((x, residual));
                    println!("signed_phase_residual_ns={residual:.3} resolved={resolved}");
                }
            }
        }
    }
    let constants = fit(&rows);
    let nonnegative = constants
        .as_ref()
        .is_some_and(|c| c.iter().all(|v| *v >= 0.0));
    let mut accepted = training_resolved && nonnegative;
    let mut training_errors = [None::<f64>; 2];
    if let Some(constants) = &constants {
        for (x, observed) in &rows {
            let class = usize::from(x[1] == 0.0);
            let predicted: f64 = x.iter().zip(constants).map(|(x, c)| x * c).sum();
            let error = (predicted - observed).abs() / observed;
            training_errors[class] = Some(training_errors[class].unwrap_or(0.0).max(error));
        }
    }
    for (class, error) in classes.iter().zip(training_errors) {
        let valid = error.is_some_and(|e| e.is_finite() && e <= TOLERANCE);
        accepted &= valid;
        println!("training k={class} max_relative_error={error:?} within_tolerance={valid}");
        profile.insert(
            format!("measured.item.{class}.training_max_relative_error"),
            error.map_or("unknown".into(), |e| format!("{e:.12e}")),
        );
        profile.insert(
            format!("item.{class}.training_status"),
            if valid {
                "within 20%"
            } else {
                "rejected or unidentified"
            }
            .into(),
        );
    }
    let mut names = vec!["a_ns".to_owned()];
    names.extend(classes.iter().map(|k| format!("{k}.r_ns_per_run")));
    for k in classes {
        names.extend(
            regimes
                .iter()
                .map(|regime| format!("{k}.{regime}.e_ns_per_element")),
        );
    }
    for (i, name) in names.iter().enumerate() {
        let value = constants
            .as_ref()
            .map_or("unknown".into(), |c| format!("{:.12e}", c[i]));
        println!("joint_fit.{name} = {value}");
        profile.insert(format!("measured.item.{name}"), value);
    }
    profile.insert(
        "item_model".into(),
        "a + r_k*runs + e_k(W)*elements; W=phase source+destination bytes; cache cpu0".into(),
    );
    profile.insert(
        "measured.item.phase_baseline_ns".into(),
        format!("{:.12e}", baseline.median),
    );
    profile.insert(
        "item.planned.status".into(),
        "unavailable: serial token cannot admit a planned kernel".into(),
    );
    profile.insert("item.planned.r_ns_per_run".into(), "unknown".into());
    for regime in &regimes {
        profile.insert(
            format!("item.planned.{regime}.e_ns_per_element"),
            "unknown".into(),
        );
    }
    for (label, roles, shape, interleaved) in [
        ("SU2-small-shaped", 2, [16, 1, 16, 1], true),
        ("dom-large-shaped", 1, [768, 1, 768, 1], false),
    ] {
        let cold = Instant::now();
        let mut held = Fixture::new(roles, &shape, &[0, 3, 2, 1], interleaved)?;
        held.data = (0..roles * held.elements)
            .map(|i| (i % 97) as f64 / 97.0)
            .collect();
        let working_set = roles * held.elements * 16;
        payload_peak = payload_peak.max(working_set);
        let diagnostics = held.transform.layout_diagnostics();
        let layout = diagnostics.entry(0).ok_or("missing held-out layout")?;
        let source = layout
            .partner_strides
            .ok_or("missing held-out source strides")?;
        let class = usize::from(layout.block_strides[0] != 1 || source[0] != 1);
        let regime = caches
            .iter()
            .position(|(_, size)| working_set <= *size)
            .unwrap_or(caches.len());
        let runs = roles * held.elements / layout.dims[0];
        let expected_dims: &[usize] = if interleaved { &[16, 16] } else { &[589824] };
        let expected_strides: &[isize] = if interleaved { &[1, 32] } else { &[1] };
        assert_eq!(layout.dims, expected_dims);
        assert_eq!(layout.block_strides, expected_strides);
        assert_eq!(source, expected_strides);
        assert_eq!(held.replay(&mut serial, None)?, held.data);
        let setup_s = cold.elapsed().as_secs_f64();
        let time = measure(|| Ok(Some(held.phase(&mut serial)?)), 1000)?;
        let whole = measure(
            || {
                black_box(held.replay(&mut serial, None)?);
                Ok(None)
            },
            1000,
        )?;
        serial_cold_s += setup_s + time.cold + whole.cold;
        print_samples(&format!("held-out {label} k={} regime={} roles={roles} elements_per_item={} dims={:?} dst_strides={:?} src_strides={source:?} runs_per_item={} working_set_bytes={working_set}", classes[class], regimes[regime], held.elements, layout.dims, layout.block_strides, runs / roles), time);
        print_samples(&format!("held-out {label} whole-owned-call"), whole);
        let predicted = constants.as_ref().map(|c| {
            baseline.median
                + c[0] * roles as f64
                + c[1 + class] * runs as f64
                + c[3 + class * regimes.len() + regime] * (roles * held.elements) as f64
        });
        let error = predicted.map(|ns| (ns - time.median).abs() / time.median);
        accepted &= error.is_some_and(|e| e.is_finite() && e <= TOLERANCE);
        println!("held-out {label} phase_prediction_ns={predicted:?} relative_error={error:?}; whole_call_minus_phase_ns={:.3} (unpaired, unprofiled whole call; not an allocation attribution)", whole.median - time.median);
        profile.insert(
            format!("measured.heldout.{label}.phase_ns"),
            format!("{:.12e}", time.median),
        );
        profile.insert(
            format!("measured.heldout.{label}.predicted_phase_ns"),
            predicted.map_or("unknown".into(), |ns| format!("{ns:.12e}")),
        );
        profile.insert(
            format!("measured.heldout.{label}.relative_error"),
            error.map_or("unknown".into(), |e| format!("{e:.12e}")),
        );
    }
    profile.insert("item_fit_status".into(), if accepted { "held-out accepted; no trusted campaign reference" } else { "rejected: cache topology unavailable, singular/negative fit, unresolved residual, or held-out error exceeds 20%" }.into());
    for (i, name) in names.iter().enumerate() {
        profile.insert(
            format!("item.{name}"),
            if accepted {
                format!("{:.12e}", constants.as_ref().unwrap()[i])
            } else {
                "unknown".into()
            },
        );
    }
    println!("serial joint fit accepted={accepted}; planned unavailable; no structural-coefficient class measured");
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
                                Ok(None)
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
            let mut fixture = Fixture::new(64, &[elements, 1], &[0, 1], false)?;
            fixture.seed(&mut engine)?;
            payload_peak = payload_peak.max(64 * elements * 16);
            let setup_s = cold.elapsed().as_secs_f64();
            let s = measure(
                || {
                    black_box(fixture.replay(&mut engine, None)?);
                    Ok(None)
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
                let mut fixture = Fixture::new(roles, &[elements, 1], &[0, 1], false)?;
                fixture.seed(&mut engine)?;
                payload_peak = payload_peak.max(roles * elements * 16);
                let setup_s = cold.elapsed().as_secs_f64();
                let s = measure(
                    || {
                        black_box(fixture.replay(&mut engine, None)?);
                        Ok(None)
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
                                black_box(fixture.replay(&mut serial, None)?);
                                Ok(None)
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
    fn fit_identifies_shared_setup_runs_and_regime_slopes() {
        // Purpose: independent predictors recover shared setup, run and regime element costs.
        let rows = [
            (vec![1.0, 2.0, 4.0, 0.0], 24.0),
            (vec![1.0, 4.0, 4.0, 0.0], 30.0),
            (vec![2.0, 2.0, 8.0, 0.0], 42.0),
            (vec![1.0, 2.0, 0.0, 4.0], 36.0),
            (vec![1.0, 4.0, 0.0, 8.0], 70.0),
        ];
        let actual = fit(&rows).unwrap();
        for (got, expected) in actual.iter().zip([2.0, 3.0, 4.0, 7.0]) {
            assert!((got - expected).abs() < 1e-9);
        }
        assert!(fit(&[(vec![1.0, 1.0], 2.0), (vec![2.0, 2.0], 4.0)]).is_none());
    }

    #[test]
    fn cache_sizes_parse_sysfs_units_and_reject_invalid_sizes() {
        // Purpose: cache regime boundaries use checked byte counts from sysfs.
        for (text, expected) in [
            (" 32K\n", Some(32768)),
            ("2M", Some(2097152)),
            ("1G", Some(1073741824)),
            ("4096", Some(4096)),
            ("0", None),
            ("unknown", None),
            ("2KB", None),
        ] {
            assert_eq!(cache_bytes(text), expected);
        }
        assert_eq!(cache_bytes(&format!("{}K", usize::MAX)), None);
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
