//! Measurement of the existing borrowed tree-transform admission seam.
//! Tenferro 0.3 exposes no cache statistics here, so fresh versus reused
//! `DenseTreeTransformOperations` is the recorded GEMM-analysis proxy.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symtn_core::BlockStructure;
use symtn_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw, DenseTreeTransformOperations,
    StridedHostKernelAdapter, TreeTransformBackend, TreeTransformBlockSpec,
    TreeTransformReplayProfile, TreeTransformStructure, TreeTransformWorkspace,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

struct Fixture {
    name: &'static str,
    structure: Arc<BlockStructure>,
    a: TreeTransformStructure<f64>,
    b: TreeTransformStructure<f64>,
    src: Vec<f64>,
    dst: Vec<f64>,
}

impl Fixture {
    fn new(name: &'static str, groups: usize, width: usize, elements: usize) -> Self {
        let blocks = groups * width;
        let structure = Arc::new(
            BlockStructure::packed_column_major(1, (0..blocks).map(|_| vec![elements])).unwrap(),
        );
        let specs = |diagonal: f64| {
            (0..groups)
                .map(|group| {
                    let indices = (group * width..(group + 1) * width).collect::<Vec<_>>();
                    let coefficients = (0..width * width)
                        .map(|index| {
                            if index / width == index % width {
                                diagonal
                            } else {
                                0.125
                            }
                        })
                        .collect();
                    TreeTransformBlockSpec::multi(indices.clone(), indices, coefficients)
                })
                .collect::<Vec<_>>()
        };
        let a = TreeTransformStructure::compile_structures(&structure, &structure, &specs(1.0))
            .unwrap();
        let b = TreeTransformStructure::compile_structures(&structure, &structure, &specs(0.75))
            .unwrap();
        let len = blocks * elements;
        Self {
            name,
            structure,
            a,
            b,
            src: (0..len).map(|index| (index % 17) as f64).collect(),
            dst: vec![0.0; len],
        }
    }

    fn describe(&self) {
        let plan = self.a.recoupling_plan();
        let workspace_payload_lower_bound_bytes =
            (plan.source_len() + plan.destination_len() + plan.coefficient_len())
                * size_of::<f64>();
        println!(
            "fixture={} tasks={} jobs={} structure_charged_bytes={} workspace_payload_lower_bound_bytes={}",
            self.name,
            self.a.block_count(),
            plan.jobs().len(),
            self.a.charged_payload_bytes(),
            workspace_payload_lower_bound_bytes,
        );
    }
}

fn replay(
    engine: &mut dyn symtn_dense::Engine,
    backend: &mut DenseTreeTransformOperations,
    workspace: &mut TreeTransformWorkspace<f64>,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
    dst: &mut [f64],
    src: &[f64],
) {
    backend
        .tree_transform_structure_overwrite_into_raw(
            engine,
            workspace,
            transform,
            structure,
            structure,
            dst,
            src,
            1.0,
            &[],
        )
        .unwrap();
}

fn profiled_replay(
    engine: &mut dyn symtn_dense::Engine,
    backend: &mut DenseTreeTransformOperations,
    workspace: &mut TreeTransformWorkspace<f64>,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
    dst: &mut [f64],
    src: &[f64],
) -> TreeTransformReplayProfile {
    let mut profile = TreeTransformReplayProfile::default();
    backend
        .tree_transform_structure_overwrite_into_raw_profiled(
            engine,
            workspace,
            transform,
            structure,
            structure,
            dst,
            src,
            1.0,
            &[],
            &mut profile,
        )
        .unwrap();
    profile
}

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    counting_alloc::start();
    let value = operation();
    let allocs = counting_alloc::stop();
    (value, allocs.calls as usize, allocs.bytes as usize)
}

fn print_sample(label: &str, profile: TreeTransformReplayProfile, calls: usize, bytes: usize) {
    let replay_phases = profile.single_total
        + profile.multi_workspace_prepare
        + profile.multi_pack
        + profile.multi_coefficient_prepare
        + profile.multi_matmul_total
        + profile.multi_scatter
        + profile.strided_kernel;
    println!(
        "phase={} total_ns={} admission_ns={} coefficient_prepare_ns={} numerical_replay_phases_ns={} caller_allocations={} caller_requested_bytes={}",
        label,
        profile.total.as_nanos(),
        profile.validate.as_nanos(),
        profile.multi_coefficient_prepare.as_nanos(),
        replay_phases.saturating_sub(profile.multi_coefficient_prepare).as_nanos(),
        calls,
        bytes,
    );
}

fn report_first_and_warm(fixture: &mut Fixture) {
    fixture.describe();
    let engine = &mut symtn_dense::CpuEngine::serial();
    let ((profile, _backend, _workspace), calls, bytes) = measured(|| {
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        let profile = profiled_replay(
            engine,
            &mut backend,
            &mut workspace,
            &fixture.a,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
        );
        (profile, backend, workspace)
    });
    print_sample(
        "cold_fresh_workspace_fresh_executor_A",
        profile,
        calls,
        bytes,
    );

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay(
        engine,
        &mut backend,
        &mut workspace,
        &fixture.a,
        &fixture.structure,
        &mut fixture.dst,
        &fixture.src,
    );
    let (profile, calls, bytes) = measured(|| {
        profiled_replay(
            engine,
            &mut backend,
            &mut workspace,
            &fixture.a,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
        )
    });
    print_sample(
        "warm_reused_workspace_reused_executor_AA",
        profile,
        calls,
        bytes,
    );

    let (profile, calls, bytes) = measured(|| {
        profiled_replay(
            engine,
            &mut backend,
            &mut workspace,
            &fixture.b,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
        )
    });
    print_sample(
        "warm_reused_workspace_reused_executor_AB",
        profile,
        calls,
        bytes,
    );
}

fn bench_fixture(c: &mut Criterion, mut fixture: Fixture) {
    report_first_and_warm(&mut fixture);
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut group = c.benchmark_group(fixture.name);
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    group.bench_function("fresh_workspace_fresh_executor_A", |bencher| {
        bencher.iter(|| {
            let mut backend = DenseTreeTransformOperations::default();
            let mut workspace = TreeTransformWorkspace::default();
            replay(
                engine,
                &mut backend,
                &mut workspace,
                &fixture.a,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
            black_box((backend, workspace));
        });
    });

    let mut workspace = TreeTransformWorkspace::default();
    group.bench_function("reused_workspace_fresh_executor_A", |bencher| {
        bencher.iter(|| {
            let mut backend = DenseTreeTransformOperations::default();
            replay(
                engine,
                &mut backend,
                &mut workspace,
                &fixture.a,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
            black_box(backend);
        });
    });

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    group.bench_function("reused_workspace_reused_executor_AA", |bencher| {
        bencher.iter(|| {
            replay(
                engine,
                &mut backend,
                &mut workspace,
                &fixture.a,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
        });
    });

    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let mut use_a = false;
    group.bench_function("reused_workspace_reused_executor_AB", |bencher| {
        bencher.iter(|| {
            use_a = !use_a;
            let transform = if use_a { &fixture.a } else { &fixture.b };
            replay(
                engine,
                &mut backend,
                &mut workspace,
                transform,
                &fixture.structure,
                &mut fixture.dst,
                &fixture.src,
            );
        });
    });
    group.finish();
}

fn bench_tree_transform_lowering(c: &mut Criterion) {
    bench_fixture(c, Fixture::new("many_small", 32, 2, 8));
    bench_fixture(c, Fixture::new("few_large", 2, 8, 256));
    bench_effective_rank_strided(c);
}

// This is deliberately a benchmark-local descriptor. `BakedFusedLayout` is
// compiled-plan internals, so exposing it merely to test a hypothetical
// specialization would change the production API without evidence.
#[derive(Clone, Debug)]
struct FusedDescriptor {
    dims: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
}

fn normalized_descriptor(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> FusedDescriptor {
    let mut descriptor = FusedDescriptor {
        dims: Vec::with_capacity(shape.len()),
        dst_strides: Vec::with_capacity(shape.len()),
        src_strides: Vec::with_capacity(shape.len()),
    };
    for axis in 0..shape.len() {
        let mut position = descriptor.dims.len();
        while position > 0 && descriptor.dst_strides[position - 1] > dst_strides[axis] {
            position -= 1;
        }
        descriptor.dims.insert(position, shape[axis]);
        descriptor.dst_strides.insert(position, dst_strides[axis]);
        descriptor.src_strides.insert(position, src_strides[axis]);
    }
    let mut fused = 0;
    for axis in 1..descriptor.dims.len() {
        let extent = descriptor.dims[fused] as isize;
        if descriptor.dst_strides[fused].checked_mul(extent) == Some(descriptor.dst_strides[axis])
            && descriptor.src_strides[fused].checked_mul(extent)
                == Some(descriptor.src_strides[axis])
        {
            descriptor.dims[fused] *= descriptor.dims[axis];
        } else {
            fused += 1;
            descriptor.dims[fused] = descriptor.dims[axis];
            descriptor.dst_strides[fused] = descriptor.dst_strides[axis];
            descriptor.src_strides[fused] = descriptor.src_strides[axis];
        }
    }
    descriptor.dims.truncate(fused + 1);
    descriptor.dst_strides.truncate(fused + 1);
    descriptor.src_strides.truncate(fused + 1);
    descriptor
}

fn fused_copy_generic(descriptor: &FusedDescriptor, dst: &mut [f64], src: &[f64]) {
    let rank = descriptor.dims.len();
    let mut index = vec![0; rank];
    let mut dst_base = 0isize;
    let mut src_base = 0isize;
    loop {
        for position in 0..descriptor.dims[0] {
            dst[(dst_base + position as isize * descriptor.dst_strides[0]) as usize] =
                src[(src_base + position as isize * descriptor.src_strides[0]) as usize];
        }
        let mut axis = 1;
        loop {
            if axis == rank {
                return;
            }
            index[axis] += 1;
            dst_base += descriptor.dst_strides[axis];
            src_base += descriptor.src_strides[axis];
            if index[axis] < descriptor.dims[axis] {
                break;
            }
            index[axis] = 0;
            dst_base -= descriptor.dims[axis] as isize * descriptor.dst_strides[axis];
            src_base -= descriptor.dims[axis] as isize * descriptor.src_strides[axis];
            axis += 1;
        }
    }
}

fn fused_copy_rank_2(descriptor: &FusedDescriptor, dst: &mut [f64], src: &[f64]) {
    assert_eq!(descriptor.dims.len(), 2);
    for second in 0..descriptor.dims[1] {
        let dst_base = second as isize * descriptor.dst_strides[1];
        let src_base = second as isize * descriptor.src_strides[1];
        for first in 0..descriptor.dims[0] {
            dst[(dst_base + first as isize * descriptor.dst_strides[0]) as usize] =
                src[(src_base + first as isize * descriptor.src_strides[0]) as usize];
        }
    }
}

fn fused_copy_rank_4(descriptor: &FusedDescriptor, dst: &mut [f64], src: &[f64]) {
    assert_eq!(descriptor.dims.len(), 4);
    for fourth in 0..descriptor.dims[3] {
        for third in 0..descriptor.dims[2] {
            for second in 0..descriptor.dims[1] {
                let dst_base = fourth as isize * descriptor.dst_strides[3]
                    + third as isize * descriptor.dst_strides[2]
                    + second as isize * descriptor.dst_strides[1];
                let src_base = fourth as isize * descriptor.src_strides[3]
                    + third as isize * descriptor.src_strides[2]
                    + second as isize * descriptor.src_strides[1];
                for first in 0..descriptor.dims[0] {
                    dst[(dst_base + first as isize * descriptor.dst_strides[0]) as usize] =
                        src[(src_base + first as isize * descriptor.src_strides[0]) as usize];
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Permutation {
    Identity,
    AdjacentTranspose,
    Reverse,
}

impl Permutation {
    const ALL: [Self; 3] = [Self::Identity, Self::AdjacentTranspose, Self::Reverse];

    fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::AdjacentTranspose => "adjacent_transpose",
            Self::Reverse => "reverse",
        }
    }

    fn axes(self, rank: usize) -> Vec<usize> {
        match self {
            Self::Identity => (0..rank).collect(),
            Self::AdjacentTranspose => {
                let mut axes = (0..rank).collect::<Vec<_>>();
                axes.swap(0, 1);
                axes
            }
            Self::Reverse => (0..rank).rev().collect(),
        }
    }
}

struct RankFixture {
    name: String,
    logical_rank: usize,
    descriptor: FusedDescriptor,
    structure: Arc<BlockStructure>,
    transform: TreeTransformStructure<f64>,
    src: Vec<f64>,
    dst: Vec<f64>,
}

fn rank_fixture(logical_rank: usize, permutation: Permutation, blocks: usize) -> RankFixture {
    // Equal dimensions keep every requested permutation shape-preserving.
    // These regimes intentionally change the *block* size: 32 x 256 elements
    // is metadata/scheduling dominated, 8 x 4096 is mixed, and 2 x 65536 is
    // dense-kernel dominated for ranks where the dimensions can realize it.
    let target_per_block = match blocks {
        32 => 1 << 8,
        8 => 1 << 12,
        2 => 1 << 16,
        _ => unreachable!("the benchmark defines exactly three block regimes"),
    };
    let mut side = 2usize;
    while side
        .checked_mul(2)
        .and_then(|next| next.checked_pow(logical_rank as u32))
        .is_some_and(|elements| elements <= target_per_block)
    {
        side *= 2;
    }
    let shape = vec![side; logical_rank];
    let elements_per_block = shape.iter().product::<usize>();
    let mut strides = Vec::with_capacity(logical_rank);
    let mut stride = 1usize;
    for &dim in &shape {
        strides.push(stride);
        stride *= dim;
    }
    let structure = Arc::new(
        BlockStructure::packed_column_major(logical_rank, (0..blocks).map(|_| shape.clone()))
            .unwrap(),
    );
    let axes = permutation.axes(logical_rank);
    let specs = (0..blocks)
        .map(|block| {
            TreeTransformBlockSpec::single(block, block, 1.0).with_source_axes(axes.clone())
        })
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    let src_strides = axes
        .iter()
        .map(|&axis| strides[axis] as isize)
        .collect::<Vec<_>>();
    let descriptor = normalized_descriptor(
        &shape,
        &strides
            .iter()
            .map(|&value| value as isize)
            .collect::<Vec<_>>(),
        &src_strides,
    );
    let len = elements_per_block * blocks;
    let src = (0..len)
        .map(|index| f64::from_bits(0x3ff0_0000_0000_0000 | index as u64))
        .collect::<Vec<_>>();
    let mut dst = vec![f64::NAN; len];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_structure_overwrite_with_strided_kernel_raw(
        strided_kernel::Exec::serial(),
        &mut kernels,
        &mut workspace,
        &transform,
        &structure,
        &structure,
        &mut dst,
        &src,
        1.0,
    )
    .unwrap();
    let mut expected = vec![f64::NAN; len];
    for block in 0..blocks {
        let offset = block * elements_per_block;
        fused_copy_generic(
            &descriptor,
            &mut expected[offset..offset + elements_per_block],
            &src[offset..offset + elements_per_block],
        );
    }
    assert!(dst
        .iter()
        .zip(&expected)
        .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    if descriptor.dims.len() == 2 || descriptor.dims.len() == 4 {
        let mut fixed = vec![f64::NAN; len];
        for block in 0..blocks {
            let start = block * elements_per_block;
            let end = start + elements_per_block;
            match descriptor.dims.len() {
                2 => fused_copy_rank_2(&descriptor, &mut fixed[start..end], &src[start..end]),
                4 => fused_copy_rank_4(&descriptor, &mut fixed[start..end], &src[start..end]),
                _ => unreachable!(),
            }
        }
        assert!(fixed
            .iter()
            .zip(&expected)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    }
    RankFixture {
        name: format!(
            "rank_{logical_rank}/{} /blocks_{blocks}/effective_rank_{}",
            permutation.name(),
            descriptor.dims.len()
        )
        .replace(" /", "/"),
        logical_rank,
        descriptor,
        structure,
        transform,
        src,
        dst,
    }
}

fn bench_rank_fixture(c: &mut Criterion, mut fixture: RankFixture) {
    println!(
        "effective_rank_fixture={} logical_rank={} effective_rank={} blocks={} bytes={}",
        fixture.name,
        fixture.logical_rank,
        fixture.descriptor.dims.len(),
        fixture.structure.block_count(),
        fixture.src.len() * size_of::<f64>(),
    );
    let mut group = c.benchmark_group(&fixture.name);
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);
    let mut kernels = StridedHostKernelAdapter::default();
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_structure_overwrite_with_strided_kernel_raw(
        strided_kernel::Exec::serial(),
        &mut kernels,
        &mut workspace,
        &fixture.transform,
        &fixture.structure,
        &fixture.structure,
        &mut fixture.dst,
        &fixture.src,
        1.0,
    )
    .unwrap();
    let (_, warm_allocations, warm_bytes) = measured(|| {
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            strided_kernel::Exec::serial(),
            &mut kernels,
            &mut workspace,
            &fixture.transform,
            &fixture.structure,
            &fixture.structure,
            &mut fixture.dst,
            &fixture.src,
            1.0,
        )
        .unwrap();
    });
    println!(
        "effective_rank_warm_allocations fixture={} calls={} bytes={}",
        fixture.name, warm_allocations, warm_bytes
    );
    group.bench_function("warm_strided_replay", |bencher| {
        bencher.iter(|| {
            tree_transform_structure_overwrite_with_strided_kernel_raw(
                strided_kernel::Exec::serial(),
                &mut kernels,
                &mut workspace,
                &fixture.transform,
                &fixture.structure,
                &fixture.structure,
                black_box(&mut fixture.dst),
                black_box(&fixture.src),
                1.0,
            )
            .unwrap();
        });
    });
    if fixture.descriptor.dims.len() == 2 || fixture.descriptor.dims.len() == 4 {
        let descriptor = fixture.descriptor.clone();
        let src = fixture.src.clone();
        let mut dst = vec![f64::NAN; src.len()];
        let block_len = src.len() / fixture.structure.block_count();
        group.bench_function("prototype_fixed_effective_rank", |bencher| {
            bencher.iter(|| {
                let src = black_box(&src);
                let dst = black_box(&mut dst);
                for block in 0..fixture.structure.block_count() {
                    let start = block * block_len;
                    let end = start + block_len;
                    match descriptor.dims.len() {
                        2 => fused_copy_rank_2(&descriptor, &mut dst[start..end], &src[start..end]),
                        4 => fused_copy_rank_4(&descriptor, &mut dst[start..end], &src[start..end]),
                        _ => unreachable!(),
                    }
                }
            });
        });
    }
    group.finish();
}

fn bench_effective_rank_strided(c: &mut Criterion) {
    for logical_rank in [2, 4, 8, 16] {
        for permutation in Permutation::ALL {
            for blocks in [32, 8, 2] {
                bench_rank_fixture(c, rank_fixture(logical_rank, permutation, blocks));
            }
        }
    }
}

criterion_group!(benches, bench_tree_transform_lowering);
criterion_main!(benches);
