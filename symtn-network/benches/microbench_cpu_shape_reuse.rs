use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hint::black_box;
use std::ops::{AddAssign, Mul};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt, U1FusionRule,
    U1Irrep, Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, TensorScalar};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{
    LabelOrderDenseOptimizer, Network, NetworkExecutionWorkspace, PlannedNetwork, TensorId,
};

struct CountingAllocator;

static MEASURING: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REQUESTED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

fn add_live(bytes: u64, measuring: bool) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    if measuring {
        PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && layout.size() != 0 {
            let measuring = MEASURING.load(Ordering::Acquire);
            add_live(layout.size() as u64, measuring);
            if measuring {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                REQUESTED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if !pointer.is_null() && layout.size() != 0 {
            LIVE_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            let measuring = MEASURING.load(Ordering::Acquire);
            if new_size >= layout.size() {
                add_live((new_size - layout.size()) as u64, measuring);
            } else {
                LIVE_BYTES.fetch_sub((layout.size() - new_size) as u64, Ordering::Relaxed);
            }
            if measuring {
                REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                REQUESTED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
            }
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

trait BenchScalar:
    TensorScalar + Copy + Default + Debug + AddAssign + Mul<Output = Self> + PartialEq
{
    const NAME: &'static str;
    const COMPLEX: bool;
    fn close(self, rhs: Self) -> bool;
    fn has_imaginary_part(self) -> bool;
}

impl BenchScalar for f64 {
    const NAME: &'static str = "f64";
    const COMPLEX: bool = false;

    fn close(self, rhs: Self) -> bool {
        (self - rhs).abs() <= 1.0e-11 * self.abs().max(rhs.abs()).max(1.0)
    }

    fn has_imaginary_part(self) -> bool {
        false
    }
}

impl BenchScalar for Complex64 {
    const NAME: &'static str = "c64";
    const COMPLEX: bool = true;

    fn close(self, rhs: Self) -> bool {
        (self - rhs).norm() <= 1.0e-11 * self.norm().max(rhs.norm()).max(1.0)
    }

    fn has_imaginary_part(self) -> bool {
        self.im != 0.0
    }
}

#[derive(Clone)]
struct Row {
    case: String,
    dtype: String,
    phase: String,
    workspace: String,
    iterations: u64,
    ns_per_iter: u64,
    alloc_calls: u64,
    realloc_calls: u64,
    requested_bytes: u64,
    live_start_bytes: u64,
    live_peak_bytes: u64,
    live_with_output_bytes: u64,
    live_idle_bytes: u64,
}

impl Row {
    fn csv(&self, stat: &str, sample: &str) -> String {
        format!(
            "{stat},{sample},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.case,
            self.dtype,
            self.phase,
            self.workspace,
            self.iterations,
            self.ns_per_iter,
            self.alloc_calls,
            self.realloc_calls,
            self.requested_bytes,
            self.live_start_bytes,
            self.live_peak_bytes,
            self.live_with_output_bytes,
            self.live_idle_bytes,
        )
    }

    fn parse(line: &str) -> Self {
        let fields = line.split(',').collect::<Vec<_>>();
        assert_eq!(fields.len(), 13, "unexpected child row: {line}");
        Self {
            case: fields[0].to_owned(),
            dtype: fields[1].to_owned(),
            phase: fields[2].to_owned(),
            workspace: fields[3].to_owned(),
            iterations: fields[4].parse().unwrap(),
            ns_per_iter: fields[5].parse().unwrap(),
            alloc_calls: fields[6].parse().unwrap(),
            realloc_calls: fields[7].parse().unwrap(),
            requested_bytes: fields[8].parse().unwrap(),
            live_start_bytes: fields[9].parse().unwrap(),
            live_peak_bytes: fields[10].parse().unwrap(),
            live_with_output_bytes: fields[11].parse().unwrap(),
            live_idle_bytes: fields[12].parse().unwrap(),
        }
    }

    fn child_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.case,
            self.dtype,
            self.phase,
            self.workspace,
            self.iterations,
            self.ns_per_iter,
            self.alloc_calls,
            self.realloc_calls,
            self.requested_bytes,
            self.live_start_bytes,
            self.live_peak_bytes,
            self.live_with_output_bytes,
            self.live_idle_bytes,
        )
    }
}

fn measure<T>(
    case: &str,
    dtype: &str,
    phase: &str,
    workspace: &str,
    iterations: usize,
    run: impl FnOnce() -> T,
) -> Row {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    REQUESTED_BYTES.store(0, Ordering::Relaxed);
    let live_start_bytes = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(live_start_bytes, Ordering::Relaxed);
    let start = Instant::now();
    MEASURING.store(true, Ordering::SeqCst);
    let output = run();
    MEASURING.store(false, Ordering::SeqCst);
    let elapsed = start.elapsed();
    let live_with_output_bytes = LIVE_BYTES.load(Ordering::Relaxed);
    let live_peak_bytes = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    let alloc_calls = ALLOC_CALLS.load(Ordering::Relaxed);
    let realloc_calls = REALLOC_CALLS.load(Ordering::Relaxed);
    let requested_bytes = REQUESTED_BYTES.load(Ordering::Relaxed);
    drop(output);
    let live_idle_bytes = LIVE_BYTES.load(Ordering::Relaxed);
    Row {
        case: case.to_owned(),
        dtype: dtype.to_owned(),
        phase: phase.to_owned(),
        workspace: workspace.to_owned(),
        iterations: iterations as u64,
        ns_per_iter: (elapsed.as_nanos() / iterations as u128) as u64,
        alloc_calls,
        realloc_calls,
        requested_bytes,
        live_start_bytes,
        live_peak_bytes,
        live_with_output_bytes,
        live_idle_bytes,
    }
}

fn chain_network() -> Network {
    Network::new(
        vec![vec!["a", "b"], vec!["b", "c"], vec!["c", "d"]],
        vec![false; 3],
        vec![1; 3],
        vec!["a", "d"],
        Some(1),
    )
    .unwrap()
}

fn chain_plan<R, D>(tensors: &[TensorMap<R, D>; 3]) -> PlannedNetwork
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    D: BenchScalar,
{
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    let planned = chain_network()
        .plan(
            &refs,
            &LabelOrderDenseOptimizer::new(vec!["b".into(), "c".into()]),
        )
        .unwrap();
    assert_eq!(
        planned
            .plan()
            .steps()
            .iter()
            .map(|step| (step.lhs(), step.rhs()))
            .collect::<Vec<_>>(),
        [
            (TensorId::new(0), TensorId::new(1)),
            (TensorId::new(2), TensorId::new(3)),
        ]
    );
    planned
}

fn fixture<R, D>(runtime: &Runtime, space: &GradedSpace<R>, seed: u64) -> [TensorMap<R, D>; 3]
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    D: BenchScalar,
{
    let tensors = std::array::from_fn(|index| {
        TensorMap::rand_with_seed(runtime, [space], [space], seed + index as u64).unwrap()
    });
    if D::COMPLEX {
        assert!(
            tensors
                .iter()
                .flat_map(|tensor| tensor.dense_data().unwrap())
                .copied()
                .any(D::has_imaginary_part),
            "complex fixture must contain a nonzero imaginary component"
        );
    }
    tensors
}

fn block_value<R, D>(tensor: &TensorMap<R, D>, block: usize, row: usize, column: usize) -> D
where
    R: TypedSectorAdmission,
    D: BenchScalar,
{
    let metadata = tensor.subblock(block).unwrap();
    tensor.dense_data().unwrap()
        [metadata.offset() + row * metadata.strides()[0] + column * metadata.strides()[1]]
}

fn assert_indexed_chain<R, D>(tensors: &[TensorMap<R, D>; 3], actual: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    <R as TypedSectorAdmission>::Sector: Eq,
    D: BenchScalar,
{
    assert_eq!(actual.subblock_count(), tensors[0].subblock_count());
    for tensor in tensors {
        assert_eq!(tensor.subblock_count(), actual.subblock_count());
        for input_index in 0..tensor.subblock_count() {
            let input_trees = tensor.subblock_fusion_trees(input_index).unwrap();
            assert!(
                (0..actual.subblock_count())
                    .any(|index| actual.subblock_fusion_trees(index).unwrap() == input_trees),
                "result is missing an input block"
            );
        }
    }
    for output_index in 0..actual.subblock_count() {
        let trees = actual.subblock_fusion_trees(output_index).unwrap();
        let input_indices = std::array::from_fn(|slot| {
            let tensor = &tensors[slot];
            (0..tensor.subblock_count())
                .find(|&index| tensor.subblock_fusion_trees(index).unwrap() == trees)
                .expect("square one-leg chain has the output sector in every operand")
        });
        let [a_index, b_index, c_index] = input_indices;
        let shape = actual.subblock(output_index).unwrap().shape().to_vec();
        assert_eq!(shape.len(), 2);
        let contracted = tensors[0].subblock(a_index).unwrap().shape()[1];
        assert_eq!(
            tensors[1].subblock(b_index).unwrap().shape(),
            [contracted, contracted]
        );
        assert_eq!(tensors[2].subblock(c_index).unwrap().shape()[0], contracted);
        for column in 0..shape[1] {
            for row in 0..shape[0] {
                let mut expected = D::default();
                for middle in 0..contracted {
                    for inner in 0..contracted {
                        expected += block_value(&tensors[0], a_index, row, inner)
                            * block_value(&tensors[1], b_index, inner, middle)
                            * block_value(&tensors[2], c_index, middle, column);
                    }
                }
                assert!(
                    block_value(actual, output_index, row, column).close(expected),
                    "indexed three-matrix oracle mismatch in block {output_index}, ({row}, {column})"
                );
            }
        }
    }
}

fn execute_once<R, D>(
    planned: &PlannedNetwork,
    tensors: &[TensorMap<R, D>; 3],
    workspace: Option<&mut NetworkExecutionWorkspace<R, D>>,
) -> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    D: BenchScalar,
{
    let refs = [&tensors[0], &tensors[1], &tensors[2]];
    match workspace {
        Some(workspace) => planned.execute(&refs, workspace).unwrap(),
        None => planned.execute(&refs, &mut Default::default()).unwrap(),
    }
}

fn run_batch<R, D>(
    planned: &PlannedNetwork,
    fixtures: &[[TensorMap<R, D>; 3]],
    mut workspace: Option<&mut NetworkExecutionWorkspace<R, D>>,
    iterations: usize,
) -> TensorMap<R, D>
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    D: BenchScalar,
{
    let mut output = None;
    for iteration in 0..iterations {
        let tensors = &fixtures[iteration % fixtures.len()];
        output = Some(black_box(execute_once(
            planned,
            tensors,
            workspace.as_deref_mut(),
        )));
    }
    output.unwrap()
}

fn assert_fixtures<R, D>(
    planned: &PlannedNetwork,
    fixtures: &[[TensorMap<R, D>; 3]],
    workspace_mode: &str,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    <R as TypedSectorAdmission>::Sector: Eq,
    D: BenchScalar,
{
    let mut workspace = NetworkExecutionWorkspace::default();
    for tensors in fixtures {
        let actual = if workspace_mode == "reuse" {
            execute_once(planned, tensors, Some(&mut workspace))
        } else {
            execute_once(planned, tensors, None)
        };
        assert_indexed_chain(tensors, &actual);
    }
}

fn run_case<R, D>(
    case: &str,
    provider: Arc<R>,
    patterns: Vec<Vec<(<R as TypedSectorAdmission>::Sector, usize)>>,
    dynamic_phase: Option<&str>,
    workspace_mode: &str,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec<Sector = <R as TypedSectorAdmission>::Sector>,
    <R as TypedSectorAdmission>::Sector: Clone + Eq,
    D: BenchScalar,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let cold_pattern = patterns[0].clone();
    let mut cold_workspace = NetworkExecutionWorkspace::default();
    let cold = measure(
        case,
        D::NAME,
        "cold_structure_input_plan_first_execute",
        workspace_mode,
        1,
        || {
            let space = GradedSpace::try_new(Arc::clone(&provider), cold_pattern).unwrap();
            let tensors = fixture::<R, D>(&runtime, &space, 11_000);
            let planned = chain_plan(&tensors);
            if workspace_mode == "reuse" {
                execute_once(&planned, &tensors, Some(&mut cold_workspace))
            } else {
                execute_once(&planned, &tensors, None)
            }
        },
    );
    println!("{}", cold.child_csv());
    drop(cold);
    drop(cold_workspace);

    let space = GradedSpace::try_new(Arc::clone(&provider), patterns[0].clone()).unwrap();
    let steady = (0..4)
        .map(|index| fixture::<R, D>(&runtime, &space, 12_000 + 10 * index))
        .collect::<Vec<_>>();
    let planned = chain_plan(&steady[0]);
    assert_fixtures(&planned, &steady, workspace_mode);
    let reuse = workspace_mode == "reuse";
    let mut steady_workspace = NetworkExecutionWorkspace::default();
    for _ in 0..2 {
        drop(run_batch(
            &planned,
            &steady,
            reuse.then_some(&mut steady_workspace),
            steady.len(),
        ));
    }
    let warm = measure(
        case,
        D::NAME,
        "warm_same_structure_values_change",
        workspace_mode,
        12,
        || {
            run_batch(
                &planned,
                &steady,
                reuse.then_some(&mut steady_workspace),
                12,
            )
        },
    );
    println!("{}", warm.child_csv());
    drop(warm);
    drop(steady_workspace);

    if let Some(phase) = dynamic_phase {
        let changing = patterns
            .into_iter()
            .skip(1)
            .enumerate()
            .flat_map(|(index, pattern)| {
                let space = GradedSpace::try_new(Arc::clone(&provider), pattern).unwrap();
                [
                    fixture::<R, D>(&runtime, &space, 13_000 + 20 * index as u64),
                    fixture::<R, D>(&runtime, &space, 13_010 + 20 * index as u64),
                ]
            })
            .collect::<Vec<_>>();
        assert_fixtures(&planned, &changing, workspace_mode);
        let mut changing_workspace = NetworkExecutionWorkspace::default();
        for _ in 0..2 {
            drop(run_batch(
                &planned,
                &changing,
                reuse.then_some(&mut changing_workspace),
                changing.len(),
            ));
        }
        let row = measure(
            case,
            D::NAME,
            phase,
            workspace_mode,
            changing.len() * 3,
            || {
                run_batch(
                    &planned,
                    &changing,
                    reuse.then_some(&mut changing_workspace),
                    changing.len() * 3,
                )
            },
        );
        println!("{}", row.child_csv());
        drop(row);
    }
}

fn u1_many_small() -> Vec<(U1Irrep, usize)> {
    (-7..=8).map(|charge| (U1Irrep::new(charge), 3)).collect()
}

fn u1_redistribution() -> Vec<Vec<(U1Irrep, usize)>> {
    vec![
        vec![
            (U1Irrep::new(-1), 16),
            (U1Irrep::new(0), 16),
            (U1Irrep::new(1), 16),
        ],
        vec![(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
        vec![(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
        vec![(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        vec![(U1Irrep::new(0), 1)],
        vec![(U1Irrep::new(-1), 2), (U1Irrep::new(2), 1)],
        vec![(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    ]
}

type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn fermion_u1_patterns() -> Vec<Vec<(symtn::sector::ProductSector<Z2Irrep, U1Irrep>, usize)>> {
    let sector = product_sector;
    vec![
        vec![
            (sector(Z2Irrep::EVEN, U1Irrep::new(0)), 16),
            (sector(Z2Irrep::ODD, U1Irrep::new(1)), 16),
        ],
        vec![
            (sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
        vec![
            (sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
            (sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
        vec![
            (sector(Z2Irrep::EVEN, U1Irrep::new(0)), 3),
            (sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
        vec![(sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1)],
        vec![
            (sector(Z2Irrep::ODD, U1Irrep::new(-1)), 2),
            (sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
        vec![
            (sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
    ]
}

fn run_child(case: &str, workspace_mode: &str) {
    match case {
        "dense_equivalent_f64" => run_case::<U1FusionRule, f64>(
            case,
            Arc::new(U1FusionRule),
            [24, 40, 12, 32, 16, 24]
                .map(|size| vec![(U1Irrep::new(0), size)])
                .into(),
            Some("continuous_grow_shrink"),
            workspace_mode,
        ),
        "dense_equivalent_c64" => run_case::<U1FusionRule, Complex64>(
            case,
            Arc::new(U1FusionRule),
            [24, 40, 12, 32, 16, 24]
                .map(|size| vec![(U1Irrep::new(0), size)])
                .into(),
            Some("continuous_grow_shrink"),
            workspace_mode,
        ),
        "u1_few_large_f64" => run_case::<U1FusionRule, f64>(
            case,
            Arc::new(U1FusionRule),
            u1_redistribution(),
            Some("same_total_redistribute_appear_disappear"),
            workspace_mode,
        ),
        "u1_few_large_c64" => run_case::<U1FusionRule, Complex64>(
            case,
            Arc::new(U1FusionRule),
            u1_redistribution(),
            Some("same_total_redistribute_appear_disappear"),
            workspace_mode,
        ),
        "u1_many_small_f64" => run_case::<U1FusionRule, f64>(
            case,
            Arc::new(U1FusionRule),
            vec![u1_many_small()],
            None,
            workspace_mode,
        ),
        "u1_many_small_c64" => run_case::<U1FusionRule, Complex64>(
            case,
            Arc::new(U1FusionRule),
            vec![u1_many_small()],
            None,
            workspace_mode,
        ),
        "fz2xu1_f64" => run_case::<FermionU1, f64>(
            case,
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            fermion_u1_patterns(),
            Some("same_total_redistribute_appear_disappear"),
            workspace_mode,
        ),
        "fz2xu1_c64" => run_case::<FermionU1, Complex64>(
            case,
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            fermion_u1_patterns(),
            Some("same_total_redistribute_appear_disappear"),
            workspace_mode,
        ),
        _ => panic!("unknown case {case}"),
    }
}

const CASES: [&str; 8] = [
    "dense_equivalent_f64",
    "dense_equivalent_c64",
    "u1_few_large_f64",
    "u1_few_large_c64",
    "u1_many_small_f64",
    "u1_many_small_c64",
    "fz2xu1_f64",
    "fz2xu1_c64",
];

fn main() {
    if let Ok(case) = std::env::var("SYMTN_CPU_SHAPE_CHILD") {
        let workspace = std::env::var("SYMTN_CPU_SHAPE_WORKSPACE").unwrap();
        run_child(&case, &workspace);
        return;
    }

    let samples = std::env::var("SYMTN_CPU_SHAPE_SAMPLES")
        .ok()
        .map(|value| value.parse::<usize>().expect("positive sample count"))
        .unwrap_or(5);
    assert!(samples > 0);
    let workspace_filter = std::env::var("SYMTN_CPU_SHAPE_WORKSPACE_FILTER").ok();
    let mut grouped = BTreeMap::<(String, String, String, String), Vec<Row>>::new();
    let executable = std::env::current_exe().unwrap();
    println!("stat,sample,case,dtype,phase,workspace,iterations,ns_per_iter,alloc_calls,realloc_calls,requested_bytes,live_start_bytes,live_peak_bytes,live_with_output_bytes,live_idle_bytes");
    for case in CASES {
        for workspace in ["fresh", "reuse"] {
            if workspace_filter
                .as_deref()
                .is_some_and(|selected| selected != workspace)
            {
                continue;
            }
            for sample in 0..samples {
                let output = Command::new(&executable)
                    .env("SYMTN_CPU_SHAPE_CHILD", case)
                    .env("SYMTN_CPU_SHAPE_WORKSPACE", workspace)
                    .env("RAYON_NUM_THREADS", "1")
                    .env("OMP_NUM_THREADS", "1")
                    .env("OPENBLAS_NUM_THREADS", "1")
                    .env("MKL_NUM_THREADS", "1")
                    .env("BLIS_NUM_THREADS", "1")
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "child failed for {case}/{workspace}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                for line in String::from_utf8(output.stdout).unwrap().lines() {
                    let row = Row::parse(line);
                    println!("{}", row.csv("raw", &sample.to_string()));
                    grouped
                        .entry((
                            row.case.clone(),
                            row.dtype.clone(),
                            row.phase.clone(),
                            row.workspace.clone(),
                        ))
                        .or_default()
                        .push(row);
                }
            }
        }
    }
    for rows in grouped.values_mut() {
        rows.sort_by_key(|row| row.ns_per_iter);
        println!("{}", rows[rows.len() / 2].csv("median", "-"));
    }
}
