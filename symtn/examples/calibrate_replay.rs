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
use symtn::sector::{
    GroupId, LieGroupFusionRule, MultiplicityFreeAdmissionMode, SectorId, U1FusionRule, U1Irrep,
};
use symtn::typed::{CpuEngine, GradedSpace, Pool, Runtime, TensorMap, TypedSectorAdmission};
use symtn_core::{
    BlockSpec, BlockStructure, FusionProductSpace, FusionTreeHomSpace, FusionTreePairKey,
    RuleIdentity, SectorLeg,
};
use symtn_dense::DefaultDenseExecutor;
use symtn_operations::host_pool::{lane_region, HostPoolSite};
use symtn_operations::{
    try_tree_transform_structure_overwrite_owned_raw, TreeTransformBlock, TreeTransformBlockSpec,
    TreeTransformReplayProfile, TreeTransformStructure, TreeTransformWorkspace,
};

type Res<T> = Result<T, Box<dyn std::error::Error>>;
const TOLERANCE: f64 = 0.25;
const BANDWIDTH_TOLERANCE: f64 = 0.20;
const STABILITY: f64 = 0.05;
const MAX_WORKING_SET: usize = 1024 * 1024 * 1024;
const KINDS: [&str; 2] = ["contiguous-run", "strided-inner"];

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

    // Replicate row-tree members to fill the destination's matrix leading dimension.
    fn geometry(dims: &[usize], dst: &[isize], src: &[isize]) -> Res<Self> {
        if !(1..=2).contains(&dims.len())
            || dims.len() != dst.len()
            || dims.len() != src.len()
            || dims.contains(&0)
            || dst[0] != 1
            || src.iter().any(|&s| s <= 0)
        {
            return Err("unsupported calibration geometry".into());
        }
        let inner = dims[0];
        let outer = dims.get(1).copied().unwrap_or(1);
        let lead = usize::try_from(dst.get(1).copied().unwrap_or(inner as isize))?;
        if lead < inner || lead % inner != 0 {
            return Err("destination padding does not tile with identical row-tree members".into());
        }
        let roles = lead / inner;
        let source_inner = usize::try_from(src[0])?;
        let source_lead = usize::try_from(
            src.get(1)
                .copied()
                .unwrap_or((source_inner * inner) as isize),
        )?;
        let source_span = (inner - 1)
            .checked_mul(source_inner)
            .and_then(|n| {
                (outer - 1)
                    .checked_mul(source_lead)
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| n.checked_add(1))
            .ok_or("fixture size overflow")?;
        let shape = [inner, 1, outer, 1];
        let mut fixture = Self::new(roles, &shape, &[0, 1, 2, 3], true)?;
        let mut source = Vec::new();
        for role in 0..roles {
            source.push(BlockSpec::with_key(
                fixture.structure.block(role)?.key().clone(),
                shape.to_vec(),
                vec![source_inner, source_inner * inner, source_lead, source_span],
                role * source_span,
            )?);
        }
        fixture.source = Arc::new(BlockStructure::from_blocks_with_rank(4, source)?);
        let specs: Vec<_> = (0..roles)
            .map(|i| TreeTransformBlockSpec::single(i, i, 1.0))
            .collect();
        fixture.transform = TreeTransformStructure::compile_structures(
            &fixture.structure,
            &fixture.source,
            &specs,
        )?;
        let diagnostics = fixture.transform.layout_diagnostics();
        for block in fixture.transform.blocks() {
            let TreeTransformBlock::Single { dst_layout, .. } = *block else {
                return Err("expected only Singles".into());
            };
            let layout = diagnostics
                .entry(dst_layout)
                .ok_or("missing compiled geometry")?;
            if layout.dims != dims
                || layout.block_strides != dst
                || layout.partner_strides != Some(src)
            {
                return Err("calibration geometry changed during normalization".into());
            }
        }
        fixture.data = (0..fixture.source.required_len()?)
            .map(|i| (i % 97) as f64 / 97.0)
            .collect();
        Ok(fixture)
    }

    fn traversal(elements: usize, runs: usize, kind: usize) -> Res<Self> {
        let inner = elements / runs;
        let step = kind + 1;
        if runs == 1 {
            Self::geometry(&[elements], &[1], &[step as isize])
        } else {
            Self::geometry(
                &[inner, runs],
                &[1, inner as isize],
                &[step as isize, (step * inner + 1) as isize],
            )
        }
    }

    fn print_geometry(&self, label: &str) -> Res<()> {
        let TreeTransformBlock::Single { dst_layout, .. } = self.transform.blocks()[0] else {
            return Err("expected a Single geometry".into());
        };
        let diagnostics = self.transform.layout_diagnostics();
        let layout = diagnostics
            .entry(dst_layout)
            .ok_or("missing calibration geometry")?;
        println!("{label}: baked_dims={:?} dst_strides={:?} src_strides={:?} elements_per_item={} runs_per_item={}", layout.dims, layout.block_strides, layout.partner_strides, layout.element_count, layout.element_count / layout.dims[0]);
        Ok(())
    }

    fn payload_bytes(&self) -> Res<usize> {
        Ok((self.source.required_len()? + self.structure.required_len()?) * 8)
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
    repetitions: [usize; 5],
    count: usize,
    median: f64,
    spread: f64,
    cold: f64,
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.0
}

fn measure(mut call: impl FnMut() -> Res<Option<f64>>, max_reps: usize) -> Res<Samples> {
    let start = Instant::now();
    call()?;
    let cold = start.elapsed().as_secs_f64();
    let max_reps = max_reps.min(((0.001 / cold.max(1e-9)) as usize).max(1));
    let mut reps = ((0.0002 / cold.max(1e-9)) as usize).clamp(1, max_reps);
    let mut raw = [0.0; 5];
    let mut repetitions = [0; 5];
    let mut count = 0;
    for (i, batch_reps) in repetitions.iter_mut().enumerate() {
        *batch_reps = reps;
        let start = Instant::now();
        let mut phase = None;
        for _ in 0..reps {
            if let Some(ns) = call()? {
                *phase.get_or_insert(0.0) += ns;
            }
        }
        raw[i] = phase.unwrap_or_else(|| start.elapsed().as_secs_f64() * 1e9) / reps as f64;
        count = i + 1;
        if count >= 3 {
            let center = median(&raw[..count]);
            let spread = raw[..count]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                - raw[..count].iter().copied().fold(f64::INFINITY, f64::min);
            if count == 3 && spread <= STABILITY * center {
                break;
            }
            // More short calls average timer noise; expensive calls keep one repetition.
            reps = reps.saturating_mul(2).min(max_reps);
        }
    }
    Ok(Samples {
        raw,
        repetitions,
        count,
        median: median(&raw[..count]),
        spread: raw[..count]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - raw[..count].iter().copied().fold(f64::INFINITY, f64::min),
        cold,
    })
}

fn print_samples(label: &str, samples: Samples) {
    println!(
        "{label}: batches_ns={:?} repetitions={:?} median_ns={:.3} spread_max_min_ns={:.3} warmup_s={:.6}",
        &samples.raw[..samples.count],
        &samples.repetitions[..samples.count],
        samples.median,
        samples.spread,
        samples.cold
    );
}

fn paired_slope(rows: &[(f64, f64)]) -> Option<f64> {
    let mut slopes = Vec::new();
    for (i, &(x, y)) in rows.iter().enumerate() {
        for &(other_x, other_y) in &rows[i + 1..] {
            if other_x != x {
                slopes.push((other_y - y) / (other_x - x));
            }
        }
    }
    (!slopes.is_empty() && slopes.iter().all(|v| v.is_finite())).then(|| median(&slopes))
}

fn tiny_intercept(rows: &[(f64, f64)]) -> Option<f64> {
    let slope = paired_slope(rows)?;
    Some(median(
        &rows.iter().map(|&(n, t)| t - slope * n).collect::<Vec<_>>(),
    ))
}

fn interpolate(curve: &[(usize, f64)], bytes: usize) -> f64 {
    let Some(&(first_w, first)) = curve.first() else {
        return f64::NAN;
    };
    if bytes <= first_w {
        return first;
    }
    for pair in curve.windows(2) {
        let [(lo_w, lo), (hi_w, hi)] = pair else {
            unreachable!()
        };
        if bytes <= *hi_w {
            return lo
                + (hi - lo)
                    * ((bytes as f64 / *lo_w as f64).ln() / (*hi_w as f64 / *lo_w as f64).ln());
        }
    }
    curve.last().unwrap().1
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ItemClass {
    dims: Vec<usize>,
    dst: Vec<isize>,
    src: Vec<isize>,
}

impl ItemClass {
    fn elements(&self) -> usize {
        self.dims.iter().product()
    }
    fn runs(&self) -> usize {
        self.elements() / self.dims[0]
    }
    fn kind(&self) -> usize {
        usize::from(self.dst[0] != 1 || self.src[0] != 1)
    }
}

type RealItems = Vec<(String, usize, BTreeMap<ItemClass, usize>)>;

fn real_items(engine: &mut CpuEngine) -> Res<RealItems> {
    let runtime = Runtime::builder().build()?;
    let mut fixtures = Vec::new();
    // Typed public permute and lower staging use the same spaces and production compiler.
    macro_rules! collect {
        ($label:expr, $rule:expr, $w:expr, $p:expr) => {{
            let rule = Arc::new($rule);
            let w = $w;
            let p = $p;
            let encode = |pairs: &Vec<_>| -> Res<SectorLeg> {
                let pairs = pairs
                    .iter()
                    .map(|(q, n)| {
                        Ok((
                            TypedSectorAdmission::try_encode_label(rule.as_ref(), q)?,
                            *n,
                        ))
                    })
                    .collect::<Res<Vec<_>>>()?;
                Ok(SectorLeg::try_new(pairs, false)?)
            };
            let bond = encode(&w)?;
            let physical = encode(&p)?;
            let w = GradedSpace::try_new(rule.clone(), w)?;
            let p = GradedSpace::try_new(rule.clone(), p)?;
            let x: TensorMap<_, f64> = TensorMap::from_subblock_fn(
                engine,
                &runtime,
                [&w, &p],
                [&w, &p],
                |tree, index| {
                    let charge =
                        TypedSectorAdmission::try_encode_label(rule.as_ref(), tree.coupled())
                            .expect("fixture provider encodes its own sector")
                            .id();
                    (index.iter().fold(charge % 97, |n, &i| (n * 17 + i) % 97) as f64 + 1.0) / 97.0
                },
            )?;
            let y = x.permute(engine, &[0, 3], &[2, 1])?;
            let product = FusionProductSpace::new([bond, physical]);
            let source =
                symtn_tensors::BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
                    rule,
                    FusionTreeHomSpace::new(product.clone(), product),
                )?;
            let mut context =
                symtn_tensors::TreeTransformExecutionContext::<f64, RuleIdentity>::default();
            let staged = context.stage_tree_transform_multiplicity_free_in(
                engine,
                symtn_tensors::FusionSource::Direct(&source),
                x.dense_data()?.len(),
                &symtn_tensors::TreeTransformOperation::permute([0, 3], [2, 1]),
            )?;
            let structure = staged.structure();
            if structure.has_pack_gemm_scatter_blocks() {
                return Err("real held-out fixture contains a Multi".into());
            }
            // This also checks that the staged compiler is the public permute's execution path.
            let mut replay = Fixture {
                structure: staged.destination_structure().clone(),
                source: staged.source_structure().clone(),
                transform: structure.clone(),
                data: x.dense_data()?.to_vec(),
                workspace: TreeTransformWorkspace::default(),
                dense: DefaultDenseExecutor::new(),
                roles: structure.blocks().len(),
                elements: 0,
                rank: 4,
            };
            assert_eq!(replay.replay(engine, None)?, y.dense_data()?);
            let diagnostics = structure.layout_diagnostics();
            let mut classes = BTreeMap::new();
            for block in structure.blocks() {
                let TreeTransformBlock::Single { dst_layout, .. } = *block else {
                    unreachable!()
                };
                let layout = diagnostics
                    .entry(dst_layout)
                    .ok_or("missing real Single geometry")?;
                let class = ItemClass {
                    dims: layout.dims.to_vec(),
                    dst: layout.block_strides.to_vec(),
                    src: layout
                        .partner_strides
                        .ok_or("missing real source geometry")?
                        .to_vec(),
                };
                *classes.entry(class).or_insert(0) += 1;
            }
            let working_set = (x.dense_data()?.len() + y.dense_data()?.len()) * 8;
            println!(
                "real permute {}: items={} phase_working_set_bytes={working_set}",
                $label,
                structure.blocks().len()
            );
            fixtures.push(($label.to_owned(), working_set, classes));
        }};
    }
    for dominant in [false, true] {
        let w = (-100..=100)
            .map(|q| {
                Ok((
                    U1Irrep::new(q)?,
                    if dominant {
                        if q == 0 {
                            768
                        } else {
                            48
                        }
                    } else {
                        16
                    },
                ))
            })
            .collect::<Res<Vec<_>>>()?;
        let p = if dominant {
            vec![(U1Irrep::new(0)?, 1)]
        } else {
            (-1..=1)
                .map(|q| Ok((U1Irrep::new(q)?, 1)))
                .collect::<Res<Vec<_>>>()?
        };
        collect!(
            if dominant { "U1-dom" } else { "U1-small" },
            U1FusionRule,
            w,
            p
        );
        let rule = LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(GroupId::su(2)?)?;
        let w = (0..=99)
            .map(|j| {
                Ok((
                    rule.category().irrep(&[j])?,
                    if dominant {
                        if j == 0 {
                            768
                        } else {
                            48
                        }
                    } else {
                        16
                    },
                ))
            })
            .collect::<Res<Vec<_>>>()?;
        let p = vec![(rule.category().irrep(&[if dominant { 0 } else { 1 }])?, 1)];
        collect!(if dominant { "SU2-dom" } else { "SU2-small" }, rule, w, p);
    }
    Ok(fixtures)
}

fn memory_limit() -> usize {
    let available = read("/proc/meminfo")
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse::<usize>()
                .ok()?
                .checked_mul(1024)
        })
        .or_else(|| {
            if cfg!(target_os = "macos") {
                sysctl("hw.memsize").parse().ok()
            } else {
                None
            }
        });
    let mut limit = available.unwrap_or(MAX_WORKING_SET);
    for root in ["/sys/fs/cgroup", "/sys/fs/cgroup/memory"] {
        let capacity = read(format!("{root}/memory.max"))
            .parse::<usize>()
            .ok()
            .or_else(|| read(format!("{root}/memory.limit_in_bytes")).parse().ok());
        let used = read(format!("{root}/memory.current"))
            .parse::<usize>()
            .ok()
            .or_else(|| read(format!("{root}/memory.usage_in_bytes")).parse().ok());
        if let Some(remaining) = capacity.zip(used).map(|(n, used)| n.saturating_sub(used)) {
            limit = limit.min(remaining);
        }
    }
    // Sparse source hulls use up to twice the addressed bytes; leave room for allocator retention.
    (limit / 4).min(MAX_WORKING_SET)
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

/// Host identity fields, the aggregate LLC bytes, and each cache level's
/// `(name, bytes)`.
type Host = (
    BTreeMap<String, String>,
    Option<usize>,
    Vec<(String, usize)>,
);

fn host() -> Host {
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
        println!("Usage: calibrate_replay <profile-path> [width]\nwidth defaults to available parallelism; run a release build.\nThree batches when spread <=5%, otherwise five; held-out tolerance = 25%.\nMeasures CPU f64 owned-overwrite Singles with measured working-set curves and decoupled fits; no profile loader is installed.");
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
    println!("Compiled replay calibration: width={width}, cpu-faer, f64 owned-overwrite copy; held-out tolerance=25%");
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
        ("model_version", "3"),
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
    let largest_cache = caches.iter().map(|(_, bytes)| *bytes).max();
    let requested = largest_cache
        .unwrap_or(64 * 1024 * 1024)
        .saturating_mul(4)
        .min(MAX_WORKING_SET);
    let allowance = memory_limit();
    let limit = requested.min(allowance);
    if limit < 64 * 1024 {
        return Err("insufficient memory for the 64 KiB ladder".into());
    }
    let mut targets = vec![64 * 1024usize];
    while *targets.last().unwrap() < limit {
        let next = targets.last().unwrap() * 2;
        if next > allowance || next > MAX_WORKING_SET {
            break;
        }
        targets.push(next);
    }
    let largest = *targets.last().unwrap();
    println!("Working-set ladder: {:?}; reported caches are information only; requested_bytes={requested} memory_allowance_bytes={allowance}; W counts distinct source+destination element bytes, sparse padding is reported separately.", targets);
    println!("Serial phase minus no-item baseline excludes output allocation/proof validation, includes first stores. Raw e(W)=median residual/elements; corrected e subtracts independently fitted a and r(W) to avoid double counting. All curves interpolate in log W with flat endpoints.");
    let mut identified = true;
    let mut raw_curves: [Vec<(usize, f64)>; 2] = Default::default();
    for (kind, _) in KINDS.iter().enumerate() {
        for &working_set in &targets {
            let elements = working_set / 16;
            let cold = Instant::now();
            let mut fixture = Fixture::traversal(elements, 1, kind)?;
            fixture.print_geometry("curve")?;
            payload_peak = payload_peak.max(fixture.payload_bytes()?);
            let setup = cold.elapsed().as_secs_f64();
            let time = measure(|| Ok(Some(fixture.phase(&mut serial)?)), 256)?;
            serial_cold_s += setup + time.cold;
            print_samples(
                &format!(
                    "curve k={} W={working_set} elements={elements} runs=1 payload_bytes={}",
                    KINDS[kind],
                    fixture.payload_bytes()?
                ),
                time,
            );
            let resolved = time.median - baseline.median > time.spread + baseline.spread;
            identified &= resolved;
            println!("curve residual_resolved={resolved}");
            raw_curves[kind].push((
                working_set,
                (time.median - baseline.median) / elements as f64,
            ));
        }
    }
    let mut run_targets = BTreeSet::from([targets[0], largest]);
    run_targets.insert(targets[targets.len() / 2]);
    let mut run_curves: [Vec<(usize, f64)>; 2] = Default::default();
    for (kind, _) in KINDS.iter().enumerate() {
        for &working_set in &run_targets {
            let elements = working_set / 16;
            let mut rows = Vec::new();
            let mut spreads = Vec::new();
            for runs in [1, elements / 64, elements / 4, elements / 2] {
                let cold = Instant::now();
                let mut fixture = Fixture::traversal(elements, runs, kind)?;
                fixture.print_geometry("paired")?;
                payload_peak = payload_peak.max(fixture.payload_bytes()?);
                let setup = cold.elapsed().as_secs_f64();
                let time = measure(|| Ok(Some(fixture.phase(&mut serial)?)), 256)?;
                serial_cold_s += setup + time.cold;
                print_samples(&format!("paired k={} W={working_set} elements={elements} runs={runs} payload_bytes={}", KINDS[kind], fixture.payload_bytes()?), time);
                rows.push((runs as f64, time.median));
                spreads.push(time.spread);
            }
            let slope = paired_slope(&rows).ok_or("unidentified paired run slope")?;
            let resolved =
                rows.last().unwrap().1 - rows[0].1 > spreads[0] + spreads.last().unwrap();
            identified &= resolved;
            println!("paired endpoint_difference_resolved={resolved}");
            println!(
                "paired k={} W={working_set} r_ns_per_run={slope:.9}",
                KINDS[kind]
            );
            run_curves[kind].push((working_set, slope));
        }
    }
    let mut tiny = Vec::new();
    let mut tiny_spreads = Vec::new();
    for elements in 1..=16 {
        let mut fixture = Fixture::new(1, &[elements, 1], &[0, 1], false)?;
        fixture.data = vec![1.0; elements];
        let time = measure(|| Ok(Some(fixture.phase(&mut serial)?)), 256)?;
        print_samples(&format!("tiny elements={elements} roles=1 runs=1"), time);
        serial_cold_s += time.cold;
        if elements > 1 {
            tiny.push((elements as f64, time.median - baseline.median));
            tiny_spreads.push(time.spread);
        }
    }
    let tiny_slope = paired_slope(&tiny).ok_or("unidentified tiny intercept")?;
    let intercept = tiny_intercept(&tiny).ok_or("unidentified tiny intercept")?;
    let a = intercept - interpolate(&run_curves[0], 16 * 16);
    let role_resolved = a > baseline.spread + median(&tiny_spreads);
    identified &= role_resolved;
    println!(
        "a_resolved={role_resolved}; baseline_plus_tiny_median_spread_ns={:.9}",
        baseline.spread + median(&tiny_spreads)
    );
    println!("a_ns={a:.9}; tiny_element_slope_ns={tiny_slope:.9}; tiny_intercept_minus_baseline_ns={intercept:.9}; n=1 is a scalar-normalization control excluded from the copy intercept");
    let mut element_curves = raw_curves.clone();
    for (kind, _) in KINDS.iter().enumerate() {
        for (working_set, e) in &mut element_curves[kind] {
            *e -= (a + interpolate(&run_curves[kind], *working_set)) / (*working_set / 16) as f64;
        }
        for (curve_name, curve) in [
            ("raw_e_ns_per_element", &raw_curves[kind]),
            ("e_ns_per_element", &element_curves[kind]),
            ("r_ns_per_run", &run_curves[kind]),
        ] {
            println!("measured.item.{}.{}={curve:?}", KINDS[kind], curve_name);
            profile.insert(
                format!(
                    "measured.item.{}.{}.working_set_bytes",
                    KINDS[kind], curve_name
                ),
                curve
                    .iter()
                    .map(|(w, _)| w.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            profile.insert(
                format!("measured.item.{}.{}", KINDS[kind], curve_name),
                curve
                    .iter()
                    .map(|(_, value)| format!("{value:.12e}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }
    let mut accepted = identified
        && a.is_finite()
        && a >= 0.0
        && run_curves
            .iter()
            .chain(&element_curves)
            .flatten()
            .all(|(_, value)| value.is_finite() && *value >= 0.0);
    profile.insert("measured.item.a_ns".into(), format!("{a:.12e}"));
    profile.insert("item_model".into(), "a + r_k(W)*runs + e_k(W)*elements; log-W interpolation, flat endpoints; W=distinct phase source+destination element bytes; sysfs informational only".into());
    profile.insert(
        "measured.item.phase_baseline_ns".into(),
        format!("{:.12e}", baseline.median),
    );
    profile.insert(
        "item.planned.status".into(),
        "unavailable: serial token cannot admit a planned kernel".into(),
    );
    let heldout_cold = Instant::now();
    let real = real_items(&mut serial)?;
    serial_cold_s += heldout_cold.elapsed().as_secs_f64();
    for (label, phase_w, classes) in real {
        payload_peak = payload_peak.max(2 * phase_w);
        for (index, (class, original_roles)) in classes.iter().enumerate() {
            let cold = Instant::now();
            let mut held = Fixture::geometry(&class.dims, &class.dst, &class.src)?;
            payload_peak = payload_peak.max(held.payload_bytes()?);
            let setup = cold.elapsed().as_secs_f64();
            let time = measure(|| Ok(Some(held.phase(&mut serial)?)), 256)?;
            serial_cold_s += setup + time.cold;
            let working_set = held.roles * held.elements * 16;
            let kind = class.kind();
            let per_item = (time.median - baseline.median) / held.roles as f64;
            let predicted = a
                + interpolate(&run_curves[kind], working_set) * class.runs() as f64
                + interpolate(&element_curves[kind], working_set) * class.elements() as f64;
            let error = (predicted - per_item).abs() / per_item.abs();
            let valid = error.is_finite() && per_item > 0.0 && error <= TOLERANCE;
            accepted &= valid;
            print_samples(&format!("held-out {label}.{index} k={} elements={} runs={} dims={:?} dst_strides={:?} src_strides={:?} original_items={original_roles} original_phase_W={phase_w} isolated_items={} isolated_W={working_set} payload_bytes={}", KINDS[kind], class.elements(), class.runs(), class.dims, class.dst, class.src, held.roles, held.payload_bytes()?), time);
            println!("held-out {label}.{index}: measured_ns_per_item={per_item:.3} predicted_ns_per_item={predicted:.3} relative_error={error:.6} within_25_percent={valid}; isolated exact geometry, not the original phase cache state");
            for (field, value) in [
                ("measured_ns_per_item", per_item),
                ("predicted_ns_per_item", predicted),
                ("relative_error", error),
            ] {
                profile.insert(
                    format!("measured.heldout.{label}.{index}.{field}"),
                    format!("{value:.12e}"),
                );
            }
        }
    }
    profile.insert(
        "item_fit_status".into(),
        if accepted {
            "held-out accepted; no trusted campaign reference"
        } else {
            "rejected: unresolved/nonfinite/negative constant or per-class held-out error exceeds 25%"
        }
        .into(),
    );
    println!("serial decoupled fit accepted={accepted}; planned unavailable; no structural-coefficient class measured");
    println!(
        "serial fit section cold_s={serial_cold_s:.6} total_s={:.6}",
        section.elapsed().as_secs_f64()
    );
    let serial_elapsed = start.elapsed().as_secs_f64();

    let ladder: Vec<_> = std::iter::successors(Some(1usize), |&q| {
        (q < width).then(|| q.saturating_mul(2).min(width))
    })
    .collect();
    let mut dispatch_widths = Vec::new();
    let mut dispatch = Vec::new();
    let mut bandwidth_widths = Vec::new();
    let mut bandwidth = Vec::new();
    let per_array = largest / 4;
    let mut all_bandwidth_stable = true;
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
        for elements in [64, (largest.min(1024 * 1024) / 16 / 64).max(1)] {
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
        {
            let roles = 64;
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
                print_samples(&format!("measured large-W candidate q={q} roles={roles} elements_per_role={elements} bytes_per_array={}",roles*elements*8),s);
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
                            "narrow serial large-W retest on widest-engine-first-touched source",
                            retest,
                        );
                        let rate = fixture.bytes() / retest.median;
                        let stable = narrow_reference.is_some_and(|first: f64| {
                            (rate / first - 1.0).abs() <= BANDWIDTH_TOLERANCE
                        });
                        all_bandwidth_stable &= stable;
                        println!("narrow_retest_bytes_per_ns={rate:.6} stable={stable} (includes placement and cache changes)");
                    }
                }
            }
            let stable = rates.iter().all(|r| r.is_finite() && *r > 0.0)
                && (rates[0] / rates[1] - 1.0).abs() <= BANDWIDTH_TOLERANCE;
            println!("bandwidth q={q} twice_size_stable={stable}; placement verification unknown");
            all_bandwidth_stable &= stable;
            bandwidth_widths.push(q.to_string());
            bandwidth.push(if rates[1].is_finite() && rates[1] > 0.0 {
                format!("{:.9}", rates[1])
            } else {
                "unknown".into()
            });
        }
        println!(
            "bandwidth section q={q} cold_s={bandwidth_cold_s:.6} total_s={:.6}",
            section.elapsed().as_secs_f64()
        );
    }
    let bandwidth_elapsed = start.elapsed().as_secs_f64() - serial_elapsed;
    let projected_p4 = serial_elapsed + bandwidth_elapsed * 3.0 / ladder.len() as f64;
    let projected_p96 = serial_elapsed + bandwidth_elapsed * 8.0 / ladder.len() as f64;
    println!("runtime projection: p=4 {projected_p4:.3}s, p=96 {projected_p96:.3}s; serial section fixed, per-width dispatch/bandwidth scaled by ladder counts 3/8; projection excludes host contention and different topology. p4_within_60s={}", projected_p4 <= 60.0);
    profile.insert(
        "projected_runtime_p4_s".into(),
        format!("{projected_p4:.6}"),
    );
    profile.insert(
        "projected_runtime_p96_s".into(),
        format!("{projected_p96:.6}"),
    );
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
    fn paired_differences_identify_run_cost_without_element_columns() {
        // Purpose: fixed element cost cancels, and one contaminated row does not set the median slope.
        assert_eq!(
            paired_slope(&[(1.0, 105.0), (4.0, 120.0), (16.0, 180.0), (32.0, 260.0)]),
            Some(5.0)
        );
        assert_eq!(
            paired_slope(&[
                (1.0, 105.0),
                (4.0, 120.0),
                (16.0, 180.0),
                (32.0, 999.0),
                (64.0, 420.0)
            ]),
            Some(5.0)
        );
        assert_eq!(paired_slope(&[(1.0, 2.0), (1.0, 3.0)]), None);
    }

    #[test]
    fn tiny_intercept_separates_one_run_from_per_role_cost() {
        // Purpose: hand values with a=7, r=3 and element slope=2 recover the role cost.
        assert_eq!(
            tiny_intercept(&[(2.0, 14.0), (8.0, 26.0), (16.0, 42.0)]).unwrap() - 3.0,
            7.0
        );
    }

    #[test]
    fn measured_curves_interpolate_in_log_working_set_and_clamp_endpoints() {
        // Purpose: half a log interval has half the cost change, with no extrapolation.
        let curve = [(64, 2.0), (256, 6.0), (1024, 10.0)];
        for (w, expected) in [
            (1, 2.0),
            (64, 2.0),
            (128, 4.0),
            (256, 6.0),
            (512, 8.0),
            (2048, 10.0),
        ] {
            assert!((interpolate(&curve, w) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn cache_sizes_parse_sysfs_units_and_reject_invalid_sizes() {
        // Purpose: informational cache sizes and the ladder extent use checked sysfs byte counts.
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
