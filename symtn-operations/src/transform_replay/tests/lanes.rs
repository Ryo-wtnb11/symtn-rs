//! The lane route's placement: the one-slot iteration in source block
//! order, the cost classes, and Singles that run whole on their compiled
//! plan with each writer's own gate.

use super::threaded::first_count;
use super::*;
use crate::kernel_adapter::observed::ObservedKernels;
use crate::kernel_adapter::sealed::GATE_SCANS;
use crate::transform_replay::lanes::{component_time, place_components, Admission, LanePlacement};

/// The lane route: the native executor, whose lane GEMMs a BLAS provider
/// does not have.
fn lanes() -> DefaultDenseExecutor {
    DefaultDenseExecutor::new()
}

/// The multi-slot schedule of `transform`, built on first use.
fn lane_schedule(
    transform: &TreeTransformStructure<f64>,
) -> &crate::transform_structure::TreeTransformLaneSchedule {
    transform.task_view().unwrap().lanes().unwrap()
}

/// `blocks` Singles `b -> b` of scale two over a canonical U(1) layout of
/// `blocks` charges, one `degeneracy x degeneracy` block each: separately
/// compiled roles of equal geometry, each its own component.
fn equal_singles(
    blocks: i32,
    degeneracy: usize,
) -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
    let leg = || {
        symtn_core::SectorLeg::new(
            (0..blocks).map(|charge| (symtn_core::U1Irrep::new(charge), degeneracy)),
            false,
        )
    };
    let structure = symtn_core::FusionTreeHomSpace::new(
        symtn_core::FusionProductSpace::new([leg()]),
        symtn_core::FusionProductSpace::new([leg()]),
    )
    .coupled_subblock_structure_from_leg_degeneracies(&symtn_core::U1FusionRule)
    .unwrap();
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &(0..blocks as usize)
            .map(|block| TreeTransformBlockSpec::single(block, block, 2.0))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    (structure, transform)
}

/// Nine equal Singles whose blocks Strided gives all four workers of a
/// four-worker pool, so each runs alone on the whole token.
fn wide_equal_singles() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
    let large = first_count(4, |width| width == 4);
    let degeneracy = (1..).find(|d: &usize| d * d >= large).unwrap();
    let (structure, transform) = equal_singles(9, degeneracy);
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.component_count(), 9);
    assert!(lane_schedule(&transform).components[..9]
        .iter()
        .all(|component| component.planned_single));
    assert_eq!(transform.layouts().retained_plan_count(), 9);
    (structure, transform)
}

fn source(len: usize) -> Vec<f64> {
    (0..len).map(|i| (i % 89) as f64 - 44.0).collect()
}

/// The initialised replay of `transform` on `threads` lanes: an overwrite
/// (`beta = None`) or `dst = alpha f(src) + beta dst`.
#[allow(clippy::too_many_arguments)]
fn initialized<A: HostKernelAdapter<f64> + Clone + Send + Sync>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    transform: &TreeTransformStructure<f64>,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    destination: &mut [f64],
    source: &[f64],
    beta: Option<f64>,
    threads: usize,
) {
    match beta {
        None => tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            kernels,
            &mut lanes(),
            &mut TreeTransformWorkspace::default(),
            transform,
            dst,
            src,
            destination,
            source,
            1.0,
            &[],
            threads,
        ),
        Some(beta) => tree_transform_structure_with_structural_recoupling_raw(
            engine,
            kernels,
            &mut lanes(),
            &mut TreeTransformWorkspace::default(),
            transform,
            dst,
            src,
            destination,
            source,
            1.0,
            beta,
            threads,
        ),
    }
    .unwrap();
}

fn owned(
    engine: &mut dyn Engine,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
    nout: usize,
    source: &[f64],
    alpha: f64,
    threads: usize,
) -> Vec<f64> {
    try_tree_transform_structure_overwrite_owned_raw(
        engine,
        &mut lanes(),
        &mut TreeTransformWorkspace::default(),
        transform,
        structure,
        structure,
        nout,
        source,
        alpha,
        threads,
        None,
    )
    .unwrap()
    .expect("a canonical layout admits the owned writer")
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn one_slot_lanes_match_the_block_oracle_with_groups_whole() {
    // What: on one slot, interleaved canonical blocks whose group-free
    // components sit between components with Multi groups replay to the
    // per-block oracle in the initialised overwrite and the owned writer,
    // bit for bit alike.
    let (structure, transform) = su2_interleaved_fixture();
    let source = (1..=60).map(|v| f64::from(v) * 0.25).collect::<Vec<_>>();
    let expected = su2_oracle(&source, 1.0);
    let mut pool = crate::host_pool::test_pool(1);
    let mut overwritten = vec![f64::NAN; 60];
    let owned = pool.install(|engine| {
        initialized(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &transform,
            &structure,
            &structure,
            &mut overwritten,
            &source,
            None,
            1,
        );
        owned(engine, &transform, &structure, 2, &source, 1.0, 1)
    });
    assert_eq!(overwritten, expected);
    assert_eq!(
        owned.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        overwritten.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn one_slot_lanes_accumulate_onto_the_destination() {
    // What: `dst = f(src) + dst` on one slot over the same mix of
    // group-free and grouped components is the oracle plus the initial
    // destination.
    let (structure, transform) = su2_interleaved_fixture();
    let source = (1..=60).map(|v| f64::from(v) * 0.25).collect::<Vec<_>>();
    let initial = (1..=60).map(|v| -f64::from(v)).collect::<Vec<_>>();
    let expected = su2_oracle(&source, 1.0)
        .iter()
        .zip(&initial)
        .map(|(value, old)| value + old)
        .collect::<Vec<_>>();
    let mut accumulated = initial.clone();
    crate::host_pool::test_pool(1).install(|engine| {
        initialized(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &transform,
            &structure,
            &structure,
            &mut accumulated,
            &source,
            Some(1.0),
            1,
        )
    });
    assert_eq!(accumulated, expected);
}

/// Packed one-axis blocks: Singles of 64, 64 and 32 elements, then Multi
/// pairs of 16-, 16- and 8-element blocks, each its own component in that
/// order.
fn class_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
    let lens = [64, 64, 32, 16, 16, 16, 16, 8, 8];
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, lens.map(|len| vec![len])).unwrap());
    let pair = |first: usize| {
        TreeTransformBlockSpec::multi(
            vec![first, first + 1],
            vec![first, first + 1],
            vec![1.0, 0.5, -0.5, 1.0],
        )
    };
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::single(1, 1, 3.0),
            TreeTransformBlockSpec::single(2, 2, 1.0),
            pair(3),
            pair(5),
            pair(7),
        ],
    )
    .unwrap();
    (structure, transform)
}

#[test]
fn equal_cost_inputs_share_a_class_and_unequal_ones_do_not() {
    // What: components with equal moved elements, band units, plan
    // eligibility and Multi job shapes (the two 64-element Singles, the two
    // 16-element pairs) share a class; a different size does not, whatever
    // the coefficients.
    let (_, transform) = class_fixture();
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.component_count(), 6);
    let lanes = lane_schedule(&transform);
    let class = &lanes.component_class;
    let distinct = class.iter().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(lanes.class_count, 4);
    assert_eq!(distinct.len(), 4);
    assert_eq!(class[0], class[1]);
    assert_eq!(class[3], class[4]);
    for (a, b) in [(0, 2), (0, 3), (2, 3), (3, 5), (2, 5)] {
        assert_ne!(class[a], class[b], "components {a} and {b}");
    }
}

/// Column-major coupled matrices of `cols` columns whose trees are row
/// ranges (shape `[rows, cols]`, strides `[1, total rows]`): per matrix a
/// two-member group of 16-row trees, a 40-row Single and a three-member
/// group of 8-row trees, interleaved in its columns, so each matrix is one
/// component with two groups.
fn heterogeneous_fixture(cols: &[usize]) -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
    let trees = [(16, 0), (40, 2), (8, 1), (16, 0), (8, 1), (8, 1)];
    let rows: usize = trees.iter().map(|tree| tree.0).sum();
    let (mut blocks, mut specs, mut offset) = (Vec::new(), Vec::new(), 0);
    for &cols in cols {
        let mut groups = [Vec::new(), Vec::new()];
        let mut row = 0;
        for &(tree_rows, role) in &trees {
            let index = blocks.len();
            blocks.push(
                BlockSpec::with_key(
                    BlockKey::ordinal(index),
                    vec![tree_rows, cols],
                    vec![1, rows],
                    offset + row,
                )
                .unwrap(),
            );
            row += tree_rows;
            match role {
                2 => specs.push(TreeTransformBlockSpec::single(index, index, 0.5)),
                group => groups[group].push(index),
            }
        }
        for members in groups {
            let m = members.len();
            let matrix = (0..m * m)
                .map(|i| if i / m == i % m { 1.0 } else { 0.25 })
                .collect();
            specs.push(TreeTransformBlockSpec::multi(
                members.clone(),
                members,
                matrix,
            ));
        }
        offset += rows * cols;
    }
    let structure = Arc::new(BlockStructure::from_blocks(blocks).unwrap());
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    (structure, transform)
}

/// `n` destination blocks of shape `[16]`, stride `[n]` at offsets
/// `0..n`, all in one atom, each the destination of a two-source Multi
/// group of packed blocks: one component of `n` groups.
fn coalesced_fixture(n: usize) -> TreeTransformStructure<f64> {
    let dst = Arc::new(
        BlockStructure::from_blocks(
            (0..n)
                .map(|j| BlockSpec::with_key(BlockKey::ordinal(j), vec![16], vec![n], j).unwrap())
                .collect(),
        )
        .unwrap(),
    );
    let src = Arc::new(BlockStructure::packed_column_major(1, vec![vec![16]; 2 * n]).unwrap());
    TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &(0..n)
            .map(|j| TreeTransformBlockSpec::multi(vec![j], vec![2 * j, 2 * j + 1], vec![1.0, 2.0]))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn class_placement_equals_the_per_component_placement() {
    // What: placing one member per cost class and copying it to the others
    // gives every component the width and times its own search over the
    // candidate widths gives, on heterogeneous grouped components with
    // repeated shapes and on one coalesced component of many groups.
    const THREADS: usize = 8;
    let mut engine = crate::host_pool::test_engine(THREADS);
    for transform in [
        heterogeneous_fixture(&[32, 32, 40, 48, 40, 32]).1,
        coalesced_fixture(64),
        class_fixture().1,
    ] {
        let schedule = transform.parallel_schedule();
        let lanes_of = lane_schedule(&transform);
        let jobs = transform.recoupling_plan().jobs();
        let shapes = jobs
            .iter()
            .map(|job| symtn_dense::LaneGemmShape {
                rows: job.rows,
                k: job.contracted,
                n: job.cols,
            })
            .collect::<Vec<_>>();
        let group_sizes = |component: usize| {
            schedule.component_group_list(component).iter().fold(
                (0usize, 0usize),
                |(source, destination), &group| {
                    let job = &jobs[group as usize];
                    (
                        source.max(job.rows * job.contracted),
                        destination.max(job.rows * job.cols),
                    )
                },
            )
        };
        let mut checked = false;
        lanes()
            .with_lane_gemms(
                &mut engine,
                symtn_dense::DenseDType::F64,
                &shapes,
                &mut |mut token, gemms| {
                    let budget = THREADS.min(token.budget().get());
                    let admission = Admission::new(true);
                    let (mut classes, mut placements) = (Vec::new(), Vec::new());
                    place_components(
                        &mut token,
                        gemms,
                        schedule,
                        lanes_of,
                        budget,
                        &group_sizes,
                        admission,
                        &mut classes,
                        &mut placements,
                    );
                    assert_eq!(placements.len(), schedule.component_count());
                    for (component, placed) in placements.iter().enumerate() {
                        let mut time = |width| {
                            component_time(
                                &mut token, gemms, schedule, lanes_of, component, width, admission,
                            )
                        };
                        let cap = budget.min(lanes_of.components[component].units).max(1);
                        let (serial, _) = time(1);
                        let (mut width, mut timed) = (1, serial);
                        let mut candidate = 2;
                        while candidate <= cap {
                            let (at, occupied) = time(candidate);
                            if at < timed {
                                (width, timed) = (occupied, at);
                            }
                            if candidate == cap {
                                break;
                            }
                            candidate = (candidate * 2).min(cap);
                        }
                        let (source, destination) = group_sizes(component);
                        assert_eq!(
                            *placed,
                            LanePlacement {
                                width,
                                serial,
                                timed,
                                source,
                                destination,
                            },
                            "component {component}"
                        );
                    }
                    checked = true;
                },
            )
            .unwrap();
        assert!(checked);
    }
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn a_planned_single_runs_whole_on_its_plan_and_owned_matches_initialised() {
    // What: an overwrite of nine separate whole-block Singles on four
    // workers writes the dense oracle on the role's compiled plan, and the
    // owned writer's result is bit for bit the initialised one.
    let (structure, transform) = wide_equal_singles();
    let len = structure.required_len().unwrap();
    let source = source(len);
    let expected = source.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
    let mut pool = crate::host_pool::test_pool(4);
    let mut overwritten = vec![f64::NAN; len];
    let owned = pool.install(|engine| {
        initialized(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &transform,
            &structure,
            &structure,
            &mut overwritten,
            &source,
            None,
            4,
        );
        owned(engine, &transform, &structure, 1, &source, 1.0, 4)
    });
    assert_eq!(overwritten, expected);
    assert_eq!(
        owned.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        overwritten.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// One rank-9 Single whose axes cannot fuse (its first eight destination
/// strides reverse the source's), so its role carries no plan, while its
/// outer axis of stride 256 still bands it.
fn planless_bandable_single(
    outer: usize,
) -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let dims = [vec![2; 8], vec![outer]].concat();
    let dst_strides = (0..8)
        .map(|axis| 1usize << (7 - axis))
        .chain([256])
        .collect::<Vec<_>>();
    let dst = Arc::new(
        BlockStructure::from_blocks_with_rank(
            9,
            vec![BlockSpec::with_key(BlockKey::ordinal(0), dims.clone(), dst_strides, 0).unwrap()],
        )
        .unwrap(),
    );
    let src = Arc::new(BlockStructure::packed_column_major(9, vec![dims]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    (dst, src, transform)
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn accumulations_and_planless_roles_stay_banded() {
    // What: a wide accumulation of the whole-block Singles, and a wide
    // overwrite of a bandable role without a compiled plan, write their
    // items in band pieces (each smaller than its role, together all of
    // it) instead of one whole-role write.
    let mut pool = crate::host_pool::test_pool(4);
    let pieces = |pool: &mut crate::host_pool::TestPool,
                  transform: &TreeTransformStructure<f64>,
                  dst: &Arc<BlockStructure>,
                  src: &Arc<BlockStructure>,
                  beta: Option<f64>| {
        let mut kernels = ObservedKernels::recording();
        let writes = Arc::clone(&kernels.write_budgets);
        let mut destination = vec![0.0; dst.required_len().unwrap()];
        let source = source(src.required_len().unwrap());
        pool.install(|engine| {
            initialized(
                engine,
                &mut kernels,
                transform,
                dst,
                src,
                &mut destination,
                &source,
                beta,
                4,
            )
        });
        let writes = writes.lock().unwrap().clone();
        writes
            .into_iter()
            .map(|(elements, _)| elements)
            .collect::<Vec<_>>()
    };

    let (structure, transform) = wide_equal_singles();
    let role = structure.required_len().unwrap() / 9;
    let accumulated = pieces(&mut pool, &transform, &structure, &structure, Some(1.0));
    assert!(accumulated.len() > 9);
    assert!(accumulated.iter().all(|&elements| elements < role));
    assert_eq!(accumulated.iter().sum::<usize>(), 9 * role);

    let large = first_count(4, |width| width == 4);
    let (dst, src, planless) = planless_bandable_single(large.div_ceil(256).max(2));
    assert_eq!(planless.layouts().retained_plan_count(), 0);
    assert!(lane_schedule(&planless).bands[0].bandable());
    let total = dst.required_len().unwrap();
    let overwritten = pieces(&mut pool, &planless, &dst, &src, None);
    assert!(overwritten.len() > 1);
    assert!(overwritten.iter().all(|&elements| elements < total));
    assert_eq!(overwritten.iter().sum::<usize>(), total);
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn a_wide_accumulation_of_planned_singles_matches_the_oracle() {
    // What: `dst = 2 src + dst` of the whole-block Singles on four workers,
    // banded, onto a nonzero destination.
    let (structure, transform) = wide_equal_singles();
    let len = structure.required_len().unwrap();
    let source = source(len);
    let initial = (0..len).map(|i| (i % 13) as f64 + 0.5).collect::<Vec<_>>();
    let expected = source
        .iter()
        .zip(&initial)
        .map(|(v, old)| 2.0 * v + old)
        .collect::<Vec<_>>();
    let mut accumulated = initial.clone();
    crate::host_pool::test_pool(4).install(|engine| {
        initialized(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &transform,
            &structure,
            &structure,
            &mut accumulated,
            &source,
            Some(1.0),
            4,
        )
    });
    assert_eq!(accumulated, expected);
}

#[test]
fn only_a_multi_slot_replay_builds_the_multi_slot_schedule() {
    // What: a one-slot replay of a compiled structure leaves its multi-slot
    // schedule unbuilt; the first replay on two slots builds it; the
    // structure's retained charge is the same before and after, and both
    // replays write the dense oracle; the built schedule retains exactly
    // the bytes charged for it.
    let (structure, transform) = equal_singles(4, 8);
    let len = structure.required_len().unwrap();
    let source = source(len);
    let expected = source.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
    let charged = transform.charged_payload_bytes();
    let built = || transform.parallel_schedule().lanes.get().is_some();
    for (threads, built_after) in [(1, false), (2, true)] {
        let mut overwritten = vec![f64::NAN; len];
        crate::host_pool::test_pool(threads).install(|engine| {
            initialized(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &transform,
                &structure,
                &structure,
                &mut overwritten,
                &source,
                None,
                threads,
            )
        });
        assert_eq!(overwritten, expected, "{threads} slots");
        assert_eq!(built(), built_after, "{threads} slots");
        assert_eq!(transform.charged_payload_bytes(), charged);
    }
    let schedule = transform.parallel_schedule();
    let lanes = schedule.lanes.get().unwrap();
    let retained = lanes.bands.capacity() * core::mem::size_of_val(&lanes.bands[0])
        + lanes.components.capacity() * core::mem::size_of_val(&lanes.components[0])
        + lanes.group_widest.capacity() * core::mem::size_of::<usize>()
        + lanes.component_class.capacity() * core::mem::size_of::<u32>();
    assert_eq!(retained, schedule.lane_bytes());
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn whole_token_singles_take_one_gate_scan_on_the_calling_thread() {
    // What: nine separately compiled equal-geometry Singles, each run whole
    // on its own plan across the whole token (the alone tier), take exactly
    // one gate scan on the calling thread for the whole call: the placement
    // scans nothing and the alone tier's components share one gate, for the
    // initialised and the owned writer; the values are the dense oracle.
    let (structure, transform) = wide_equal_singles();
    let len = structure.required_len().unwrap();
    let role = len / 9;
    let source = source(len);
    let expected = source.iter().map(|v| 2.0 * v).collect::<Vec<_>>();
    let mut pool = crate::host_pool::test_pool(4);
    let scans = |run: &mut dyn FnMut()| {
        let before = GATE_SCANS.with(|scans| scans.get());
        run();
        GATE_SCANS.with(|scans| scans.get()) - before
    };

    let mut kernels = ObservedKernels::recording();
    let writes = Arc::clone(&kernels.write_budgets);
    let mut overwritten = vec![f64::NAN; len];
    let initialized_scans = scans(&mut || {
        pool.install(|engine| {
            initialized(
                engine,
                &mut kernels,
                &transform,
                &structure,
                &structure,
                &mut overwritten,
                &source,
                None,
                4,
            )
        })
    });
    assert_eq!(overwritten, expected);
    assert_eq!(*writes.lock().unwrap(), vec![(role, 4); 9]);
    assert_eq!(initialized_scans, 1);

    let mut written = Vec::new();
    let owned_scans = scans(&mut || {
        written = pool.install(|engine| owned(engine, &transform, &structure, 1, &source, 2.0, 4));
    });
    assert_eq!(written, source.iter().map(|v| 4.0 * v).collect::<Vec<_>>());
    assert_eq!(owned_scans, 1);
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn a_single_strided_runs_narrower_than_the_token_leaves_the_alone_tier() {
    // What: two separately compiled equal-geometry planned Singles on an
    // eight-worker pool whose dispatch costs make Strided copy each block
    // on six workers of the whole token (and on all four of a half token)
    // occupy six slots, not the eight they were offered: neither runs alone
    // on the whole token, each runs whole on its own plan on a narrower
    // lane, and the values are the dense oracle.
    use std::time::Duration;
    use strided_kernel::execution::{parallel_width, KernelClass};
    use tenferro_cpu::threadteam::WidthCosts;
    const THREADS: usize = 8;
    let row = WidthCosts::from_anchors(&[
        (2, Duration::from_nanos(100)),
        (6, Duration::from_nanos(100)),
        (7, Duration::from_millis(10)),
        (8, Duration::from_millis(10)),
    ])
    .unwrap();
    let mut pool = crate::host_pool::test_pool_with_costs(THREADS, row);
    let (structure, transform) = equal_singles(2, 256);
    let role = structure.required_len().unwrap() / 2;
    let widths = pool.with_exec(|mut exec| {
        let whole = parallel_width(&exec, KernelClass::Copy, role).get();
        let half = granted_width(exec.reborrow(), THREADS / 2);
        (parallel_width(&half, KernelClass::Copy, role).get(), whole)
    });
    assert_eq!(widths, (4, 6));
    assert!(lane_schedule(&transform).components[..2]
        .iter()
        .all(|component| component.planned_single));

    let source = source(2 * role);
    let mut kernels = ObservedKernels::recording();
    let writes = Arc::clone(&kernels.write_budgets);
    let mut overwritten = vec![f64::NAN; 2 * role];
    pool.install(|engine| {
        initialized(
            engine,
            &mut kernels,
            &transform,
            &structure,
            &structure,
            &mut overwritten,
            &source,
            None,
            THREADS,
        )
    });
    assert_eq!(
        overwritten,
        source.iter().map(|v| 2.0 * v).collect::<Vec<_>>()
    );
    let writes = writes.lock().unwrap().clone();
    assert_eq!(writes.len(), 2);
    assert!(writes
        .iter()
        .all(|&(elements, budget)| elements == role && budget < THREADS));
}
