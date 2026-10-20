//! Same-process cold/warm measurements for public basic tensor operations.

use std::{
    convert::Infallible,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{
    Complex64, ContractSpec, Eig, Error, GradedSpace, LinalgBackend, Qr, Runtime, TensorMap,
    TensorScalar,
};

use symtn::cache::{StructureCacheInfo, StructureCacheKind};
use symtn::expert::CpuBackendKind;
use symtn::expert::DefaultDenseExecutor;
use symtn::expert::{BlockRef, SectorLeg};
use symtn::expert::{DenseExecutor, DenseGemmBatchJob, MatrixOp};
use symtn::expert::{DenseView, DenseViewMut};
use symtn::sector::SectorId;
use symtn::sector::{
    BraidingStyleKind, CheckedGenericFusion, FusionRule, FusionStyleKind, RuleIdentity, SectorVec,
};
use symtn::typed::FactorScalar;
use symtn::typed::FusionTreeKey;
use symtn_core::{BlockSpec, BlockStructure, FusionProductSpace, FusionTreeHomSpace};
use symtn_dense::strided_batch_runs;
use symtn_matrixalgebra::seam::{
    eig_full_from_source, lq_compact_from_source, lq_full_from_source, qr_compact_from_source,
    qr_full_from_source, svd_compact_dyn_checked_generic, CheckedGenericFactorPlanError,
    EigFullDyn, FactorSource,
};
use symtn_matrixalgebra::BoundDynFactor;
use symtn_tensors::{BoundDynamicFusionMapSpace, BoundDynamicTensorRef, DynamicFusionMapSpace};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[derive(Clone, Copy)]
struct Allocations {
    calls: usize,
    requested_bytes: usize,
}

fn measure_allocations<T, E>(
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<(T, Allocations), E> {
    let (result, allocs) = counting_alloc::measure(operation);
    result.map(|value| {
        (
            value,
            Allocations {
                calls: allocs.calls as usize,
                requested_bytes: allocs.bytes as usize,
            },
        )
    })
}

#[derive(Clone, Copy)]
struct Counters {
    completed: StructureCacheInfo,
    fusion_layout: StructureCacheInfo,
    complete_hom: StructureCacheInfo,
}

fn structure_cache_info(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .expect("every structure cache kind reports")
}

fn counters(_: &Runtime) -> Counters {
    Counters {
        completed: structure_cache_info(StructureCacheKind::CompletedTreeTransformer),
        fusion_layout: structure_cache_info(StructureCacheKind::SectorStructure),
        complete_hom: structure_cache_info(StructureCacheKind::DegeneracyStructure),
    }
}

fn delta(after: usize, before: usize) -> isize {
    after as isize - before as isize
}

#[expect(
    clippy::too_many_arguments,
    reason = "the example printer exposes one local benchmark row without introducing a reporting abstraction"
)]
fn print_sample(
    symmetry: &str,
    operation: &str,
    form: &str,
    phase: &str,
    iterations: u64,
    elapsed: Duration,
    allocations: Allocations,
    before: Counters,
    after: Counters,
) {
    let tree_before = before.completed;
    let tree_after = after.completed;
    let layout_before = before.fusion_layout;
    let layout_after = after.fusion_layout;
    let hom_before = before.complete_hom;
    let hom_after = after.complete_hom;
    println!(
        "{symmetry},{operation},{form},{phase},{iterations},{us:.3},\
         {tree_hits},{tree_misses},{tree_evictions},{tree_bypasses},{tree_entries_delta},\
         {tree_bytes_before},{tree_bytes_after},{tree_bytes_delta},\
         {layout_misses},{layout_evictions},{layout_bypasses},{layout_entries_delta},\
         {layout_bytes_before},{layout_bytes_after},{layout_bytes_delta},\
         {hom_hits},{hom_misses},{hom_admissions},{hom_evictions},{hom_bypasses},\
         {hom_entries_delta},{hom_bytes_before},{hom_bytes_after},{hom_bytes_delta},\
         NA,{allocation_calls},{requested_bytes},NA,NA,NA,NA,NA",
        us = elapsed.as_secs_f64() * 1e6 / iterations as f64,
        tree_hits = tree_after.hits() - tree_before.hits(),
        tree_misses = tree_after.misses() - tree_before.misses(),
        tree_evictions = tree_after.evictions() - tree_before.evictions(),
        tree_bypasses = tree_after.rejections() - tree_before.rejections(),
        tree_entries_delta = delta(tree_after.entries(), tree_before.entries()),
        tree_bytes_before = tree_before.charged_bytes(),
        tree_bytes_after = tree_after.charged_bytes(),
        tree_bytes_delta = delta(
            tree_after.charged_bytes() as usize,
            tree_before.charged_bytes() as usize,
        ),
        layout_misses = layout_after.misses() - layout_before.misses(),
        layout_evictions = layout_after.evictions() - layout_before.evictions(),
        layout_bypasses = layout_after.rejections() - layout_before.rejections(),
        layout_entries_delta = delta(layout_after.entries(), layout_before.entries()),
        layout_bytes_before = layout_before.charged_bytes(),
        layout_bytes_after = layout_after.charged_bytes(),
        layout_bytes_delta =
            layout_after.charged_bytes() as i128 - layout_before.charged_bytes() as i128,
        hom_hits = hom_after.hits() - hom_before.hits(),
        hom_misses = hom_after.misses() - hom_before.misses(),
        hom_admissions = hom_after.admissions() - hom_before.admissions(),
        hom_evictions = hom_after.evictions() - hom_before.evictions(),
        hom_bypasses = hom_after.rejections() - hom_before.rejections(),
        hom_entries_delta = delta(hom_after.entries(), hom_before.entries()),
        hom_bytes_before = hom_before.charged_bytes(),
        hom_bytes_after = hom_after.charged_bytes(),
        hom_bytes_delta = hom_after.charged_bytes() as i128 - hom_before.charged_bytes() as i128,
        allocation_calls = allocations.calls,
        requested_bytes = allocations.requested_bytes,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the example harness keeps local labels, phases, timing policy, and measured closure explicit"
)]
fn bench<T, E>(
    runtime: &Runtime,
    symmetry: &str,
    operation: &str,
    form: &str,
    first_phase: &str,
    repeated_phase: &str,
    min_time: Duration,
    mut operation_fn: impl FnMut() -> Result<T, E>,
) -> Result<T, E> {
    let cold_before = counters(runtime);
    let cold_start = Instant::now();
    let (cold_output, cold_allocations) = measure_allocations(&mut operation_fn)?;
    black_box(&cold_output);
    let cold_elapsed = cold_start.elapsed();
    let cold_after = counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        first_phase,
        1,
        cold_elapsed,
        cold_allocations,
        cold_before,
        cold_after,
    );

    black_box(operation_fn()?);
    black_box(operation_fn()?);
    let warm_before = counters(runtime);
    let warm_start = Instant::now();
    let mut iterations = 0;
    let (_, warm_allocations) = measure_allocations(|| {
        while iterations < 2 || warm_start.elapsed() < min_time {
            black_box(operation_fn()?);
            iterations += 1;
        }
        Ok(())
    })?;
    let warm_elapsed = warm_start.elapsed();
    let warm_after = counters(runtime);
    print_sample(
        symmetry,
        operation,
        form,
        repeated_phase,
        iterations,
        warm_elapsed,
        warm_allocations,
        warm_before,
        warm_after,
    );
    if let Ok(milliseconds) = std::env::var("OP_MATRIX_PROFILE_PAUSE_MS") {
        std::thread::sleep(Duration::from_millis(
            milliseconds
                .parse()
                .expect("OP_MATRIX_PROFILE_PAUSE_MS must be an integer"),
        ));
    }
    Ok(cold_output)
}

fn benchmark_runtime() -> Result<Runtime, Error> {
    let backend = match std::env::var("OP_MATRIX_GEMM_BACKEND").as_deref() {
        Ok("blas") => LinalgBackend::Blas,
        Ok("faer") | Err(_) => LinalgBackend::Faer,
        Ok(other) => {
            return Err(Error::InvalidArgument(format!(
                "OP_MATRIX_GEMM_BACKEND must be `faer` or `blas`, got `{other}`"
            )))
        }
    };
    let builder = Runtime::builder().dense_threads(1).gemm_backend(backend);
    if std::env::var("OP_MATRIX_CACHE").as_deref() == Ok("disabled") {
        // Process-global: every build is offered and rejected.
        symtn::cache::configure_budgets([
            (StructureCacheKind::CompletedTreeTransformer, 0),
            (StructureCacheKind::TreeTransformCoefficients, 0),
        ]);
    }
    builder.build()
}

fn benchmark_dense_executor() -> Result<DefaultDenseExecutor, symtn::expert::DenseError> {
    let kind = match std::env::var("OP_MATRIX_GEMM_BACKEND").as_deref() {
        Ok("blas") => CpuBackendKind::Blas,
        Ok("faer") | Err(_) => CpuBackendKind::Faer,
        Ok(other) => {
            return Err(symtn::expert::DenseError::Unsupported {
                op: "oriented_uniform_run",
                message: format!("OP_MATRIX_GEMM_BACKEND must be `faer` or `blas`, got `{other}`"),
            })
        }
    };
    DefaultDenseExecutor::with_threads_and_kind(1, kind)
}

fn operation_enabled(operation: &str) -> bool {
    std::env::var("OP_MATRIX_OPERATION").map_or(true, |selected| selected == operation)
}

fn form_enabled(form: &str) -> bool {
    std::env::var("OP_MATRIX_FORM").map_or(true, |selected| selected == form)
}

fn assert_f64_payload_close(actual: &[f64], expected: &[f64]) {
    const ABS_TOLERANCE: f64 = 64.0 * f64::EPSILON;
    const REL_TOLERANCE: f64 = 256.0 * f64::EPSILON;

    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "non-finite payload at index {index}: actual={actual}, expected={expected}"
        );
        let error = (actual - expected).abs();
        let tolerance = ABS_TOLERANCE + REL_TOLERANCE * actual.abs().max(expected.abs());
        assert!(
            error <= tolerance,
            "payload mismatch at index {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
        );
    }
}

trait OrientedScalar: HarnessScalar + TensorScalar + std::fmt::Debug {
    fn sentinel() -> Self {
        Self::from_parts(-0.375, 0.625)
    }
    fn sample(index: usize) -> Self {
        Self::from_parts(
            0.125 + ((index * 17 + 3) % 97) as f64 / 29.0,
            -0.25 + ((index * 11 + 7) % 89) as f64 / 31.0,
        )
    }
    fn alpha() -> Self {
        Self::from_parts(0.75, -0.375)
    }
    fn error(self, other: Self) -> f64 {
        (self.as_complex() - other.as_complex()).norm()
    }
    fn magnitude(self) -> f64 {
        self.as_complex().norm()
    }
}

impl OrientedScalar for f64 {}

impl OrientedScalar for Complex64 {}

fn assert_oriented_close<T: OrientedScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-11 * (1.0 + actual.magnitude().max(expected.magnitude()));
        assert!(
            actual.error(expected) <= tolerance,
            "payload mismatch at {index}: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[derive(Clone)]
struct AdapterFixture<T> {
    lhs: Vec<T>,
    rhs: Vec<T>,
    output: Vec<T>,
    jobs: Vec<DenseGemmBatchJob>,
    runs: Vec<usize>,
    lhs_base: usize,
    rhs_base: usize,
    dst_base: usize,
}

fn adapter_fixture<T: OrientedScalar>(shapes: &[(usize, usize, usize)]) -> AdapterFixture<T> {
    let uniform = shapes.windows(2).all(|pair| pair[0] == pair[1]);
    let (lhs_base, rhs_base, dst_base) = (3, 5, 7);
    let mut lhs_offset = 0usize;
    let mut rhs_offset = 0usize;
    let mut dst_offset = 0usize;
    let mut jobs = Vec::with_capacity(shapes.len());
    for (index, &(rows, contracted, cols)) in shapes.iter().enumerate() {
        jobs.push(DenseGemmBatchJob {
            dst_offset,
            lhs_offset,
            rhs_offset,
            rows,
            contracted,
            cols,
        });
        if index + 1 != shapes.len() {
            if uniform {
                let lhs_stride = if (rows, contracted, cols) == (64, 48, 56) {
                    3080
                } else if rows >= 32 {
                    rows * contracted + 8
                } else {
                    rows * contracted + 4
                };
                let rhs_stride = if (rows, contracted, cols) == (4, 3, 5) {
                    18
                } else if (rows, contracted, cols) == (64, 48, 56) {
                    2696
                } else if rows >= 32 {
                    contracted * cols + 8
                } else {
                    contracted * cols + 6
                };
                let dst_stride = if (rows, contracted, cols) == (64, 48, 56) {
                    3592
                } else if rows >= 32 {
                    rows * cols + 8
                } else {
                    rows * cols + 4
                };
                lhs_offset += lhs_stride;
                rhs_offset += rhs_stride;
                dst_offset += dst_stride;
            } else {
                lhs_offset += rows * contracted + 3;
                rhs_offset += contracted * cols + 5;
                dst_offset += rows * cols + 7;
            }
        }
    }
    let &(last_rows, last_contracted, last_cols) = shapes.last().expect("nonempty fixture");
    let lhs_len = lhs_base + lhs_offset + last_rows * last_contracted;
    let rhs_len = rhs_base + rhs_offset + last_contracted * last_cols;
    let dst_len = dst_base + dst_offset + last_rows * last_cols;
    let lhs = (0..lhs_len).map(T::sample).collect();
    let rhs = (0..rhs_len).map(|index| T::sample(index + 101)).collect();
    let output = vec![T::sentinel(); dst_len];
    let runs = strided_batch_runs(&jobs);
    AdapterFixture {
        lhs,
        rhs,
        output,
        jobs,
        runs,
        lhs_base,
        rhs_base,
        dst_base,
    }
}

fn expected_adapter<T: OrientedScalar>(
    fixture: &AdapterFixture<T>,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    alpha: T,
    beta: T,
) -> Vec<T> {
    let mut expected = fixture.output.clone();
    for job in &fixture.jobs {
        for col in 0..job.cols {
            for row in 0..job.rows {
                let mut sum = T::zero();
                for inner in 0..job.contracted {
                    let lhs_index = match lhs_op {
                        MatrixOp::Identity => row + job.rows * inner,
                        MatrixOp::Transpose | MatrixOp::Adjoint => inner + job.contracted * row,
                    };
                    let rhs_index = match rhs_op {
                        MatrixOp::Identity => inner + job.contracted * col,
                        MatrixOp::Transpose | MatrixOp::Adjoint => col + job.cols * inner,
                    };
                    let mut lhs = fixture.lhs[fixture.lhs_base + job.lhs_offset + lhs_index];
                    let mut rhs = fixture.rhs[fixture.rhs_base + job.rhs_offset + rhs_index];
                    if lhs_op == MatrixOp::Adjoint {
                        lhs = lhs.maybe_conj(true);
                    }
                    if rhs_op == MatrixOp::Adjoint {
                        rhs = rhs.maybe_conj(true);
                    }
                    sum = sum + lhs * rhs;
                }
                let index = fixture.dst_base + job.dst_offset + row + job.rows * col;
                expected[index] = alpha * sum + beta * expected[index];
            }
        }
    }
    expected
}

fn execute_adapter<T: OrientedScalar>(
    executor: &mut DefaultDenseExecutor,
    fixture: &mut AdapterFixture<T>,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    alpha: T,
    beta: T,
) -> Result<(T, usize), symtn::expert::DenseError> {
    let lhs_shape = [fixture.lhs.len() - fixture.lhs_base];
    let rhs_shape = [fixture.rhs.len() - fixture.rhs_base];
    let dst_shape = [fixture.output.len() - fixture.dst_base];
    let strides = [1];
    let lhs = DenseView::new(&fixture.lhs, &lhs_shape, &strides, fixture.lhs_base)?;
    let rhs = DenseView::new(&fixture.rhs, &rhs_shape, &strides, fixture.rhs_base)?;
    let output = DenseViewMut::new(&mut fixture.output, &dst_shape, &strides, fixture.dst_base)?;
    executor.matmul_batch_axpby_with_ops_into(
        T::dense_write(output),
        T::dense_read(lhs),
        T::dense_read(rhs),
        &fixture.jobs,
        &fixture.runs,
        lhs_op,
        rhs_op,
        alpha.dense_scalar(),
        beta.dense_scalar(),
    )?;
    let marker = (fixture.output[fixture.dst_base], fixture.output.len());
    black_box(marker);
    Ok(marker)
}

fn print_adapter_sample(
    form: &str,
    phase: &str,
    iterations: u64,
    elapsed: Duration,
    allocations: Allocations,
) {
    let mut fields = vec![
        "DenseAdapter".to_string(),
        "oriented_uniform_run".to_string(),
        form.to_string(),
        phase.to_string(),
        iterations.to_string(),
        format!("{:.3}", elapsed.as_secs_f64() * 1e6 / iterations as f64),
    ];
    fields.extend(std::iter::repeat_n("NA".to_string(), 25));
    fields.push(allocations.calls.to_string());
    fields.push(allocations.requested_bytes.to_string());
    fields.extend(std::iter::repeat_n("NA".to_string(), 5));
    println!("{}", fields.join(","));
}

fn bench_adapter<T: OrientedScalar>(
    form: &str,
    min_time: Duration,
    expected_marker: (T, usize),
    mut operation: impl FnMut() -> Result<(T, usize), symtn::expert::DenseError>,
) -> Result<(), symtn::expert::DenseError> {
    let started = Instant::now();
    let (marker, allocations) = measure_allocations(&mut operation)?;
    let elapsed = started.elapsed();
    print_adapter_sample(
        form,
        "first_fresh_executor_after_preflight",
        1,
        elapsed,
        allocations,
    );
    assert_oriented_close(&[marker.0], &[expected_marker.0]);
    assert_eq!(marker.1, expected_marker.1);

    for _ in 0..2 {
        let marker = operation()?;
        assert_oriented_close(&[marker.0], &[expected_marker.0]);
        assert_eq!(marker.1, expected_marker.1);
    }
    let started = Instant::now();
    let mut iterations = 0;
    let (marker, allocations) = measure_allocations(|| {
        let mut marker = expected_marker;
        while iterations < 2 || started.elapsed() < min_time {
            marker = operation()?;
            iterations += 1;
        }
        Ok(marker)
    })?;
    let elapsed = started.elapsed();
    print_adapter_sample(form, "warm_fixed", iterations, elapsed, allocations);
    assert_oriented_close(&[marker.0], &[expected_marker.0]);
    assert_eq!(marker.1, expected_marker.1);
    Ok(())
}

fn bench_adapter_shape<T: OrientedScalar>(
    form: &str,
    min_time: Duration,
    expected: &[(T, usize)],
    mut operation: impl FnMut() -> Result<(usize, T, usize), symtn::expert::DenseError>,
) -> Result<(), symtn::expert::DenseError> {
    for _ in 0..expected.len() {
        let (case, first, len) = operation()?;
        assert_oriented_close(&[first], &[expected[case].0]);
        assert_eq!(len, expected[case].1);
    }
    let started = Instant::now();
    let mut iterations = 0;
    let (marker, allocations) = measure_allocations(|| {
        let mut marker = None;
        while iterations == 0 || started.elapsed() < min_time {
            for _ in 0..expected.len() {
                marker = Some(operation()?);
                iterations += 1;
            }
        }
        Ok::<_, symtn::expert::DenseError>(marker.expect("shape cycle is nonempty"))
    })?;
    let elapsed = started.elapsed();
    assert_eq!(iterations % expected.len() as u64, 0);
    print_adapter_sample(form, "warm_shape_cycle", iterations, elapsed, allocations);
    assert_oriented_close(&[marker.1], &[expected[marker.0].0]);
    assert_eq!(marker.2, expected[marker.0].1);
    Ok(())
}

fn bench_public_shape<T: OrientedScalar>(
    runtime: &Runtime,
    form: &str,
    min_time: Duration,
    expected: &[(T, usize)],
    mut operation: impl FnMut() -> Result<(usize, T, usize), Error>,
) -> Result<(), Error> {
    for _ in 0..expected.len() {
        let (case, first, len) = operation()?;
        assert_oriented_close(&[first], &[expected[case].0]);
        assert_eq!(len, expected[case].1);
    }
    let before = counters(runtime);
    let started = Instant::now();
    let mut iterations = 0;
    let (marker, allocations) = measure_allocations(|| {
        let mut marker = None;
        while iterations == 0 || started.elapsed() < min_time {
            for _ in 0..expected.len() {
                marker = Some(operation()?);
                iterations += 1;
            }
        }
        Ok::<_, Error>(marker.expect("shape cycle is nonempty"))
    })?;
    let elapsed = started.elapsed();
    assert_eq!(iterations % expected.len() as u64, 0);
    let after = counters(runtime);
    print_sample(
        "U1Public",
        "oriented_uniform_run",
        form,
        "warm_shape_cycle",
        iterations,
        elapsed,
        allocations,
        before,
        after,
    );
    assert_oriented_close(&[marker.1], &[expected[marker.0].0]);
    assert_eq!(marker.2, expected[marker.0].1);
    Ok(())
}

struct PublicFixture<T: TensorScalar> {
    lhs: TensorMap<U1FusionRule, T>,
    lhs_adjoint: TensorMap<U1FusionRule, T>,
    rhs: TensorMap<U1FusionRule, T>,
    expected_direct: TensorMap<U1FusionRule, T>,
    expected_adjoint: TensorMap<U1FusionRule, T>,
}

struct PublicInputs<T: TensorScalar> {
    lhs: TensorMap<U1FusionRule, T>,
    lhs_adjoint: TensorMap<U1FusionRule, T>,
    rhs: TensorMap<U1FusionRule, T>,
}

fn discard_public_oracles<T: OrientedScalar>(
    fixture: PublicFixture<T>,
) -> (PublicInputs<T>, (T, usize), (T, usize)) {
    let PublicFixture {
        lhs,
        lhs_adjoint,
        rhs,
        expected_direct,
        expected_adjoint,
    } = fixture;
    let direct_marker = (
        expected_direct.dense_data().unwrap()[0],
        expected_direct.dense_data().unwrap().len(),
    );
    let adjoint_marker = (
        expected_adjoint.dense_data().unwrap()[0],
        expected_adjoint.dense_data().unwrap().len(),
    );
    drop(expected_direct);
    drop(expected_adjoint);
    (
        PublicInputs {
            lhs,
            lhs_adjoint,
            rhs,
        },
        direct_marker,
        adjoint_marker,
    )
}

fn public_fixture<T: OrientedScalar>(
    runtime: &Runtime,
    sectors: usize,
    degeneracy: usize,
) -> Result<PublicFixture<T>, Error> {
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        (0..sectors).map(|charge| (U1Irrep::new(charge as i32), degeneracy)),
    )?;
    let lhs_value = |charge: i32, indices: &[usize]| {
        T::sample(charge as usize * 1009 + indices[0] + 3 * indices[1] + 211)
    };
    let rhs_value = |charge: i32, indices: &[usize]| {
        T::sample(charge as usize * 1013 + 2 * indices[0] + 5 * indices[1] + 401)
    };
    let lhs = TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |trees, indices| {
        lhs_value(trees.coupled().charge(), indices)
    })?;
    let lhs_adjoint = lhs.adjoint()?;
    let rhs = TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |trees, indices| {
        rhs_value(trees.coupled().charge(), indices)
    })?;
    let expected_direct =
        TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |trees, indices| {
            let charge = trees.coupled().charge();
            let mut sum = T::zero();
            for inner in 0..degeneracy {
                sum = sum
                    + lhs_value(charge, &[indices[0], inner])
                        * rhs_value(charge, &[inner, indices[1]]);
            }
            sum
        })?;
    let expected_adjoint =
        TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |trees, indices| {
            let charge = trees.coupled().charge();
            let mut sum = T::zero();
            for inner in 0..degeneracy {
                sum = sum
                    + lhs_value(charge, &[inner, indices[0]]).maybe_conj(true)
                        * rhs_value(charge, &[inner, indices[1]]);
            }
            sum
        })?;
    Ok(PublicFixture {
        lhs,
        lhs_adjoint,
        rhs,
        expected_direct,
        expected_adjoint,
    })
}

fn assert_public_fixture<T: OrientedScalar>(
    actual: &TensorMap<U1FusionRule, T>,
    expected: &TensorMap<U1FusionRule, T>,
) -> Result<(), Error> {
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    for index in 0..actual.subblock_count() {
        assert_eq!(actual.subblock(index)?, expected.subblock(index)?);
        assert_eq!(
            actual.subblock_fusion_trees(index)?,
            expected.subblock_fusion_trees(index)?
        );
    }
    assert_oriented_close(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    Ok(())
}

macro_rules! assert_same_tensor {
    ($actual:expr, $expected:expr, $authority:expr) => {{
        assert!(std::ptr::eq($actual.provider(), $authority.provider()));
        assert_eq!($actual.codomain(), $expected.codomain());
        assert_eq!($actual.domain(), $expected.domain());
        assert_eq!($actual.subblock_count(), $expected.subblock_count());
        for index in 0..$actual.subblock_count() {
            assert_eq!($actual.subblock(index)?, $expected.subblock(index)?);
            assert_eq!(
                $actual.subblock_fusion_trees(index)?,
                $expected.subblock_fusion_trees(index)?
            );
        }
        assert_f64_payload_close(
            $actual.dense_data().unwrap(),
            $expected.dense_data().unwrap(),
        );
    }};
}

#[derive(Clone, Copy)]
struct LayoutGenericRule;

impl FusionRule for LayoutGenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        if left.id() < 16 && right.id() < 16 {
            [SectorId::new(left.id() ^ right.id())]
                .into_iter()
                .collect()
        } else {
            SectorVec::new()
        }
    }
}

impl CheckedGenericFusion for LayoutGenericRule {
    type Error = Infallible;

    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(self)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(self)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(self)
    }

    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(self)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(FusionRule::dual(self, sector))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(FusionRule::fusion_channels(self, left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(FusionRule::fusion_channels(self, left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(FusionRule::nsymbol(self, left, right, coupled))
    }
}

trait HarnessScalar: FactorScalar + Copy {
    const NAME: &'static str;

    fn from_parts(real: f64, imaginary: f64) -> Self;
    fn as_complex(self) -> Complex64;
}

impl HarnessScalar for f64 {
    const NAME: &'static str = "f64";

    fn from_parts(real: f64, _imaginary: f64) -> Self {
        real
    }

    fn as_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl HarnessScalar for Complex64 {
    const NAME: &'static str = "c64";

    fn from_parts(real: f64, imaginary: f64) -> Self {
        Complex64::new(real, imaginary)
    }

    fn as_complex(self) -> Complex64 {
        self
    }
}

fn layout_generic_value(sector: SectorId, row: usize, column: usize) -> f64 {
    let diagonal = f64::from(row == column) * (2.0 + sector.id() as f64);
    diagonal + ((row + 1) * 3 + (column + 1) * 5 + sector.id()) as f64 / 32.0
}

fn block_value(data: &[f64], block: BlockRef<'_>, row: usize, column: usize) -> f64 {
    data[block.offset() + row * block.strides()[0] + column * block.strides()[1]]
}

fn fill_layout_generic_data(space: &BoundDynamicFusionMapSpace<LayoutGenericRule>) -> Vec<f64> {
    let structure = space.space().structure();
    let mut data = vec![0.0; space.space().required_len().unwrap()];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                let offset =
                    block.offset() + row * block.strides()[0] + column * block.strides()[1];
                data[offset] = layout_generic_value(sector, row, column);
            }
        }
    }
    data
}

struct LayoutGenericInput {
    space: BoundDynamicFusionMapSpace<LayoutGenericRule>,
    data: Vec<f64>,
}

fn layout_generic_qr_fixture(
    degeneracy: usize,
) -> Result<(LayoutGenericInput, LayoutGenericInput), Box<dyn std::error::Error>> {
    let provider = Arc::new(LayoutGenericRule);
    let vacuum = SectorId::new(0);
    let charge = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(vacuum, degeneracy), (charge, 2 * degeneracy)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(vacuum, 2 * degeneracy), (charge, degeneracy)],
            false,
        )]),
    );
    let ordinary = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::clone(&provider),
        homspace.clone(),
    )?;
    let expert = |reverse: bool| {
        let ordinary_structure = ordinary.space().structure();
        let mut offset = 1;
        let mut blocks = Vec::with_capacity(ordinary_structure.block_count());
        let mut indices = (0..ordinary_structure.block_count()).collect::<Vec<_>>();
        if reverse {
            indices.reverse();
        }
        for index in indices {
            let block = ordinary_structure.block(index)?;
            blocks.push(BlockSpec::column_major_with_key(
                block.key().clone(),
                block.shape().to_vec(),
                offset,
            )?);
            offset += block.element_count()? + 1;
        }
        let structure = BlockStructure::from_blocks_with_rank(2, blocks)?;
        let layout = DynamicFusionMapSpace::try_from_layout(homspace.clone(), structure)?
            .try_bind_rule(provider.as_ref())?;
        let dynamic = layout;
        let bound = BoundDynamicFusionMapSpace::bind_generic(dynamic, Arc::clone(&provider))?;
        let data = fill_layout_generic_data(&bound);
        Ok::<_, Box<dyn std::error::Error>>(LayoutGenericInput { space: bound, data })
    };
    Ok((expert(false)?, expert(true)?))
}

fn assert_layout_generic_source(
    space: &BoundDynamicFusionMapSpace<LayoutGenericRule>,
    data: &[f64],
) {
    let structure = space.space().structure();
    assert_eq!(structure.block_count(), 2);
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                assert_eq!(
                    block_value(data, block, row, column),
                    layout_generic_value(sector, row, column)
                );
            }
        }
    }
}

fn factor_block_for_sector<'a>(
    factor: &'a BoundDynFactor<LayoutGenericRule, f64>,
    sector: SectorId,
) -> BlockRef<'a> {
    let structure = factor.space().space().structure();
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
        .expect("the compact factor retains every source sector")
}

fn assert_layout_generic_factors_equal(
    actual: &BoundDynFactor<LayoutGenericRule, f64>,
    expected: &BoundDynFactor<LayoutGenericRule, f64>,
) {
    assert_eq!(
        actual.space().space().homspace(),
        expected.space().space().homspace()
    );
    let actual_structure = actual.space().space().structure();
    let expected_structure = expected.space().space().structure();
    assert_eq!(
        actual_structure.block_count(),
        expected_structure.block_count()
    );
    for index in 0..expected_structure.block_count() {
        let expected_block = expected_structure.block(index).unwrap();
        let actual_block = actual_structure.block_by_key(expected_block.key()).unwrap();
        assert_eq!(actual_block.shape(), expected_block.shape());
        for column in 0..expected_block.shape()[1] {
            for row in 0..expected_block.shape()[0] {
                let actual_value = block_value(actual.data(), actual_block, row, column);
                let expected_value = block_value(expected.data(), expected_block, row, column);
                let tolerance = 512.0 * f64::EPSILON * expected_value.abs().max(1.0);
                assert!((actual_value - expected_value).abs() <= tolerance);
            }
        }
    }
}

fn assert_layout_generic_qr_reconstructs(
    left: &BoundDynFactor<LayoutGenericRule, f64>,
    right: &BoundDynFactor<LayoutGenericRule, f64>,
) {
    for sector in [SectorId::new(0), SectorId::new(1)] {
        let left_block = factor_block_for_sector(left, sector);
        let right_block = factor_block_for_sector(right, sector);
        assert_eq!(left_block.shape()[1], right_block.shape()[0]);
        for column in 0..right_block.shape()[1] {
            for row in 0..left_block.shape()[0] {
                let actual = (0..left_block.shape()[1])
                    .map(|inner| {
                        block_value(left.data(), left_block, row, inner)
                            * block_value(right.data(), right_block, inner, column)
                    })
                    .sum::<f64>();
                let expected = layout_generic_value(sector, row, column);
                let tolerance = 2048.0 * f64::EPSILON * expected.abs().max(1.0);
                assert!((actual - expected).abs() <= tolerance);
            }
        }
    }
}

fn run_layout_generic_qr(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    if !operation_enabled("qr_compact_generic_layout") || !form_enabled("owned") {
        return Ok(());
    }
    let (ordered, reordered) = layout_generic_qr_fixture(degeneracy)?;
    assert_layout_generic_source(&ordered.space, &ordered.data);
    assert_layout_generic_source(&reordered.space, &reordered.data);
    let ordered_structure = ordered.space.space().structure();
    let reordered_structure = reordered.space.space().structure();
    assert_eq!(
        ordered_structure.block(0)?.key(),
        reordered_structure.block(1)?.key()
    );
    assert_eq!(
        ordered_structure.block(1)?.key(),
        reordered_structure.block(0)?.key()
    );

    let ordered_input = BoundDynamicTensorRef::try_new(&ordered.space, &ordered.data)?;
    let reordered_input = BoundDynamicTensorRef::try_new(&reordered.space, &reordered.data)?;
    let mut preflight_dense = DefaultDenseExecutor::new();
    let ordered_expected = qr_compact_dense(&mut preflight_dense, &ordered_input)
        .map_err(checked_compact_example_error)?;
    let reordered_expected = qr_compact_dense(&mut preflight_dense, &reordered_input)
        .map_err(checked_compact_example_error)?;
    assert_layout_generic_factors_equal(&reordered_expected.q, &ordered_expected.q);
    assert_layout_generic_factors_equal(&reordered_expected.r, &ordered_expected.r);
    assert_layout_generic_qr_reconstructs(&ordered_expected.q, &ordered_expected.r);
    assert_layout_generic_qr_reconstructs(&reordered_expected.q, &reordered_expected.r);

    println!(
        "# GenericLayout: qr_fixture_matrices=2 row_trees=2 col_trees=2 source_blocks=2 matrix_shapes={}x{},{}x{}",
        degeneracy,
        2 * degeneracy,
        2 * degeneracy,
        degeneracy
    );
    for (symmetry, input) in [
        ("GenericLayout-ordered", &ordered_input),
        ("GenericLayout-reordered", &reordered_input),
    ] {
        let runtime = benchmark_runtime()?;
        let mut dense = DefaultDenseExecutor::new();
        let Qr { q: left, r: right } = bench(
            &runtime,
            symmetry,
            "qr_compact_generic_layout",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || qr_compact_dense(&mut dense, input).map_err(checked_compact_example_error),
        )?;
        assert_layout_generic_qr_reconstructs(&left, &right);
    }
    Ok(())
}

struct CheckedLayoutInput<D> {
    space: BoundDynamicFusionMapSpace<LayoutGenericRule>,
    data: Vec<D>,
}

#[derive(Clone, Copy)]
enum CheckedMatrixAspect {
    Alternating,
    Square,
    Wide,
    Tall,
}

fn checked_layout_value<D: HarnessScalar>(sector: SectorId, row: usize, column: usize) -> D {
    let real = layout_generic_value(sector, row, column);
    let imaginary = if D::NAME == "c64" {
        ((row + 2) * 7 + (column + 1) * 11 + sector.id() * 3) as f64 / 37.0
    } else {
        0.0
    };
    D::from_parts(real, imaginary)
}

fn fill_checked_layout_data<D: HarnessScalar>(
    space: &BoundDynamicFusionMapSpace<LayoutGenericRule>,
) -> Vec<D> {
    let structure = space.space().structure();
    let mut data = vec![D::zero(); space.space().required_len().unwrap()];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                let offset =
                    block.offset() + row * block.strides()[0] + column * block.strides()[1];
                data[offset] = checked_layout_value(sector, row, column);
            }
        }
    }
    data
}

fn checked_layout_fixture<D: HarnessScalar>(
    degeneracy: usize,
    sector_count: usize,
) -> Result<(CheckedLayoutInput<D>, CheckedLayoutInput<D>), Box<dyn std::error::Error>> {
    checked_layout_fixture_with_aspect(degeneracy, sector_count, CheckedMatrixAspect::Alternating)
}

fn checked_layout_fixture_with_aspect<D: HarnessScalar>(
    degeneracy: usize,
    sector_count: usize,
    aspect: CheckedMatrixAspect,
) -> Result<(CheckedLayoutInput<D>, CheckedLayoutInput<D>), Box<dyn std::error::Error>> {
    let provider = Arc::new(LayoutGenericRule);
    let codomain = (0..sector_count)
        .map(|sector| {
            (
                SectorId::new(sector),
                match aspect {
                    CheckedMatrixAspect::Alternating if sector % 2 != 0 => 2 * degeneracy,
                    CheckedMatrixAspect::Tall => 2 * degeneracy,
                    _ => degeneracy,
                },
            )
        })
        .collect::<Vec<_>>();
    let domain = (0..sector_count)
        .map(|sector| {
            (
                SectorId::new(sector),
                match aspect {
                    CheckedMatrixAspect::Alternating if sector % 2 == 0 => 2 * degeneracy,
                    CheckedMatrixAspect::Wide => 2 * degeneracy,
                    _ => degeneracy,
                },
            )
        })
        .collect::<Vec<_>>();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(codomain, false)]),
        FusionProductSpace::new([SectorLeg::new(domain, false)]),
    );
    let canonical = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )?;
    let canonical_data = fill_checked_layout_data(&canonical);

    let canonical_structure = canonical.space().structure();
    let mut offset = 1;
    let mut blocks = Vec::with_capacity(canonical_structure.block_count());
    for index in (0..canonical_structure.block_count()).rev() {
        let block = canonical_structure.block(index)?;
        blocks.push(BlockSpec::column_major_with_key(
            block.key().clone(),
            block.shape().to_vec(),
            offset,
        )?);
        offset += block.element_count()? + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(2, blocks)?;
    let layout = DynamicFusionMapSpace::try_from_layout(homspace, structure)?
        .try_bind_rule(provider.as_ref())?;
    let fallback = BoundDynamicFusionMapSpace::bind_generic(layout, provider)?;
    let fallback_data = fill_checked_layout_data(&fallback);
    Ok((
        CheckedLayoutInput {
            space: canonical,
            data: canonical_data,
        },
        CheckedLayoutInput {
            space: fallback,
            data: fallback_data,
        },
    ))
}

fn factor_block<'a, D>(
    factor: &'a BoundDynFactor<LayoutGenericRule, D>,
    sector: SectorId,
) -> BlockRef<'a> {
    let structure = factor.space().space().structure();
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
        .expect("checked compact factor retains every source sector")
}

fn checked_block_value<D: HarnessScalar>(
    data: &[D],
    block: BlockRef<'_>,
    row: usize,
    column: usize,
) -> Complex64 {
    data[block.offset() + row * block.strides()[0] + column * block.strides()[1]].as_complex()
}

fn assert_checked_source_unchanged<D: HarnessScalar>(actual: &[D], expected: &[D]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_eq!(actual.as_complex(), expected.as_complex());
    }
}

fn assert_checked_pair_reconstructs<D: HarnessScalar>(
    left: &BoundDynFactor<LayoutGenericRule, D>,
    right: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    assert_eq!(left.space().space().structure().block_count(), sector_count);
    assert_eq!(
        right.space().space().structure().block_count(),
        sector_count
    );
    for sector in (0..sector_count).map(SectorId::new) {
        let left_block = factor_block(left, sector);
        let right_block = factor_block(right, sector);
        let rows = left_block.shape()[0];
        let kept = left_block.shape()[1];
        let columns = right_block.shape()[1];
        assert_eq!(right_block.shape()[0], kept);
        for column in 0..columns {
            for row in 0..rows {
                let actual = (0..kept).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                    sum + checked_block_value(left.data(), left_block, row, inner)
                        * checked_block_value(right.data(), right_block, inner, column)
                });
                let expected = checked_layout_value::<D>(sector, row, column).as_complex();
                assert!((actual - expected).norm() <= 2.0e-10 * expected.norm().max(1.0));
            }
        }
    }
}

fn assert_checked_svd_reconstructs<D: HarnessScalar>(
    u: &BoundDynFactor<LayoutGenericRule, D>,
    s: &BoundDynFactor<LayoutGenericRule, D>,
    vh: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    assert_eq!(u.space().space().structure().block_count(), sector_count);
    assert_eq!(s.space().space().structure().block_count(), sector_count);
    assert_eq!(vh.space().space().structure().block_count(), sector_count);
    for sector in (0..sector_count).map(SectorId::new) {
        let u_block = factor_block(u, sector);
        let s_block = factor_block(s, sector);
        let vh_block = factor_block(vh, sector);
        let rows = u_block.shape()[0];
        let kept = u_block.shape()[1];
        let columns = vh_block.shape()[1];
        assert_eq!(s_block.shape(), [kept, kept]);
        assert_eq!(vh_block.shape()[0], kept);
        let singular_values = (0..kept)
            .map(|index| checked_block_value(s.data(), s_block, index, index))
            .collect::<Vec<_>>();
        for value in &singular_values {
            assert!(value.im.abs() <= 2.0e-12 && value.re >= 0.0);
        }
        for adjacent in singular_values.windows(2) {
            assert!(adjacent[0].re >= adjacent[1].re);
        }
        for column in 0..columns {
            for row in 0..rows {
                let actual = (0..kept).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                    sum + checked_block_value(u.data(), u_block, row, inner)
                        * checked_block_value(s.data(), s_block, inner, inner)
                        * checked_block_value(vh.data(), vh_block, inner, column)
                });
                let expected = checked_layout_value::<D>(sector, row, column).as_complex();
                assert!((actual - expected).norm() <= 4.0e-10 * expected.norm().max(1.0));
            }
        }
    }
}

fn assert_columns_orthonormal<D: HarnessScalar>(
    factor: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    for sector in (0..sector_count).map(SectorId::new) {
        let block = factor_block(factor, sector);
        for right in 0..block.shape()[1] {
            for left in 0..block.shape()[1] {
                let actual = (0..block.shape()[0]).fold(Complex64::new(0.0, 0.0), |sum, row| {
                    sum + checked_block_value(factor.data(), block, row, left).conj()
                        * checked_block_value(factor.data(), block, row, right)
                });
                let expected = Complex64::new(f64::from(left == right), 0.0);
                assert!((actual - expected).norm() <= 2.0e-10);
            }
        }
    }
}

fn assert_rows_orthonormal<D: HarnessScalar>(
    factor: &BoundDynFactor<LayoutGenericRule, D>,
    sector_count: usize,
) {
    for sector in (0..sector_count).map(SectorId::new) {
        let block = factor_block(factor, sector);
        for lower in 0..block.shape()[0] {
            for upper in 0..block.shape()[0] {
                let actual = (0..block.shape()[1]).fold(Complex64::new(0.0, 0.0), |sum, column| {
                    sum + checked_block_value(factor.data(), block, upper, column)
                        * checked_block_value(factor.data(), block, lower, column).conj()
                });
                let expected = Complex64::new(f64::from(upper == lower), 0.0);
                assert!((actual - expected).norm() <= 2.0e-10);
            }
        }
    }
}

fn checked_compact_example_error(error: CheckedGenericFactorPlanError<Infallible>) -> Error {
    Error::InvalidArgument(format!("checked compact input fixture failed: {error:?}"))
}

fn preflight_checked_compact_input<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    sector_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut dense = DefaultDenseExecutor::new();
    let qr = qr_compact_dense(&mut dense, input).map_err(checked_compact_example_error)?;
    assert_checked_pair_reconstructs(&qr.q, &qr.r, sector_count);
    assert_columns_orthonormal(&qr.q, sector_count);
    drop(qr);
    let svd = svd_compact_dyn_checked_generic(&mut dense, input)
        .map_err(checked_compact_example_error)?;
    assert_checked_svd_reconstructs(&svd.u, &svd.s, &svd.vh, sector_count);
    assert_columns_orthonormal(&svd.u, sector_count);
    assert_rows_orthonormal(&svd.vh, sector_count);
    drop(svd);
    let lq = lq_compact_dense(&mut dense, input).map_err(checked_compact_example_error)?;
    assert_checked_pair_reconstructs(&lq.l, &lq.q, sector_count);
    assert_rows_orthonormal(&lq.q, sector_count);
    Ok(())
}

fn run_checked_compact_operation<D: HarnessScalar>(
    operation: &str,
    dense: &mut DefaultDenseExecutor,
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
) -> Result<(), Box<dyn std::error::Error>> {
    match operation {
        "qr" => drop(black_box(
            qr_compact_dense(dense, input).map_err(checked_compact_example_error)?,
        )),
        "svd" => drop(black_box(
            svd_compact_dyn_checked_generic(dense, input).map_err(checked_compact_example_error)?,
        )),
        "lq" => drop(black_box(
            lq_compact_dense(dense, input).map_err(checked_compact_example_error)?,
        )),
        _ => unreachable!("fixed checked compact operation table"),
    }
    Ok(())
}

fn run_checked_compact_input_fixture<D: HarnessScalar>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_runtime = benchmark_runtime()?;
    let setup_symmetry = format!("GenericCheckedInput-{workload}-{}", D::NAME);
    drop(bench(
        &setup_runtime,
        &setup_symmetry,
        "checked_compact_input_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || checked_layout_fixture::<D>(degeneracy, sector_count),
    )?);
    let (canonical, fallback) = checked_layout_fixture::<D>(degeneracy, sector_count)?;
    let (canonical_changed, fallback_changed) =
        checked_layout_fixture::<D>(degeneracy + 1, sector_count)?;
    let canonical_structure = canonical.space.space().structure();
    assert_eq!(canonical_structure.block_count(), sector_count);
    assert_eq!(
        fallback.space.space().structure().block_count(),
        sector_count
    );
    let mut matrix_shapes = Vec::with_capacity(sector_count);
    for sector in 0..sector_count {
        let block = canonical_structure.block(sector)?;
        assert_eq!(
            block.key().as_fusion_tree_pair().unwrap().coupled(),
            SectorId::new(sector)
        );
        matrix_shapes.push(format!("{}x{}", block.shape()[0], block.shape()[1]));
    }
    println!(
        "# GenericCheckedInput: workload={} dtype={} labels=0..{} G={} row_trees={} col_trees={} source_blocks={} matrix_shapes={} layouts=canonical,padded_reordered setup=fixture+binding+literal_preflight_outside_timer measured=public_owned_factorization+fresh_return+drop caller_allocations=requested_only peak_native_worker=NA",
        workload,
        D::NAME,
        sector_count - 1,
        canonical_structure.block_count(),
        canonical_structure.block_count(),
        canonical_structure.block_count(),
        canonical_structure.block_count(),
        matrix_shapes.join(";")
    );
    println!(
        "# GenericCheckedInput: shape_change=alternating_preconstructed_d{}_d{} fixture_construction=excluded region_tables=preinitialized_by_literal_preflight executor=reused_per_operation returned_factors=dropped_inside_timed_closure",
        degeneracy,
        degeneracy + 1
    );
    for (layout, fixture, changed_fixture) in [
        ("canonical", canonical, canonical_changed),
        ("fallback", fallback, fallback_changed),
    ] {
        let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data)?;
        let changed_input =
            BoundDynamicTensorRef::try_new(&changed_fixture.space, &changed_fixture.data)?;
        let original = fixture.data.clone();
        let changed_original = changed_fixture.data.clone();
        preflight_checked_compact_input(&input, sector_count)?;
        preflight_checked_compact_input(&changed_input, sector_count)?;
        assert_checked_source_unchanged(&fixture.data, &original);
        assert_checked_source_unchanged(&changed_fixture.data, &changed_original);

        let symmetry = format!("GenericCheckedInput-{workload}-{layout}-{}", D::NAME);
        let runtime = benchmark_runtime()?;
        let mut dense = DefaultDenseExecutor::new();
        let qr = bench(
            &runtime,
            &symmetry,
            "checked_compact_input_qr",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || qr_compact_dense(&mut dense, &input).map_err(checked_compact_example_error),
        )?;
        assert_checked_pair_reconstructs(&qr.q, &qr.r, sector_count);
        drop(qr);
        let svd = bench(
            &runtime,
            &symmetry,
            "checked_compact_input_svd",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || {
                svd_compact_dyn_checked_generic(&mut dense, &input)
                    .map_err(checked_compact_example_error)
            },
        )?;
        assert_checked_svd_reconstructs(&svd.u, &svd.s, &svd.vh, sector_count);
        drop(svd);
        let lq = bench(
            &runtime,
            &symmetry,
            "checked_compact_input_lq",
            "owned",
            "first_after_setup",
            "warm_after_setup",
            min_time,
            || lq_compact_dense(&mut dense, &input).map_err(checked_compact_example_error),
        )?;
        assert_checked_pair_reconstructs(&lq.l, &lq.q, sector_count);
        drop(lq);
        assert_checked_source_unchanged(&fixture.data, &original);

        for operation in ["qr", "svd", "lq"] {
            let runtime = benchmark_runtime()?;
            let mut dense = DefaultDenseExecutor::new();
            let mut changed = false;
            bench(
                &runtime,
                &symmetry,
                &format!("checked_compact_input_{operation}_shape_alternating"),
                "owned",
                "first_after_inputs_and_regions_setup",
                "warm_shape_alternating",
                min_time,
                || {
                    let selected = if changed { &changed_input } else { &input };
                    changed = !changed;
                    run_checked_compact_operation(operation, &mut dense, selected)
                },
            )?;
        }
        assert_checked_source_unchanged(&fixture.data, &original);
        assert_checked_source_unchanged(&changed_fixture.data, &changed_original);
    }
    Ok(())
}

fn run_checked_compact_input_for<D: HarnessScalar>(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    run_checked_compact_input_fixture::<D>("few-large", degeneracy, 2, min_time)?;
    run_checked_compact_input_fixture::<D>("many-small", (degeneracy / 16).max(1), 16, min_time)
}

fn run_checked_compact_input(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    if !operation_enabled("checked_compact_input") || !form_enabled("owned") {
        return Ok(());
    }
    run_checked_compact_input_for::<f64>(degeneracy, min_time)?;
    run_checked_compact_input_for::<Complex64>(degeneracy, min_time)
}

fn eig_value<D: HarnessScalar>(sector: usize, row: usize, column: usize) -> D {
    let diagonal = (sector * 32 + row + 1) as f64;
    let adjacent = row + 1 == column && diagonal > 1.0;
    D::from_parts(
        if row == column {
            diagonal
        } else if adjacent {
            if D::NAME == "c64" {
                0.25
            } else {
                0.5
            }
        } else {
            0.0
        },
        if D::NAME == "c64" && adjacent {
            -0.375
        } else {
            0.0
        },
    )
}

fn eig_fixture<D: HarnessScalar>(
    degeneracy: usize,
    sector_count: usize,
) -> Result<(CheckedLayoutInput<D>, CheckedLayoutInput<D>), Box<dyn std::error::Error>> {
    let (mut canonical, mut fallback) =
        checked_layout_fixture_with_aspect(degeneracy, sector_count, CheckedMatrixAspect::Square)?;
    for fixture in [&mut canonical, &mut fallback] {
        let structure = fixture.space.space().structure();
        for block_index in 0..structure.block_count() {
            let block = structure.block(block_index)?;
            let sector = block.key().as_fusion_tree_pair().unwrap().coupled().id();
            for column in 0..block.shape()[1] {
                for row in 0..block.shape()[0] {
                    fixture.data
                        [block.offset() + row * block.strides()[0] + column * block.strides()[1]] =
                        eig_value(sector, row, column);
                }
            }
        }
    }
    Ok((canonical, fallback))
}

fn multitree_eig_value<D: HarnessScalar>(
    sector: usize,
    dimension: usize,
    row: usize,
    column: usize,
) -> D {
    let adjacent = row + 1 == column;
    D::from_parts(
        if row == column {
            (sector * dimension + row + 1) as f64
        } else if adjacent {
            if D::NAME == "c64" {
                0.25
            } else {
                0.5
            }
        } else {
            0.0
        },
        if D::NAME == "c64" && adjacent {
            -0.375
        } else {
            0.0
        },
    )
}

fn multitree_label(tree: &FusionTreeKey, sector_count: usize) -> (usize, usize) {
    assert_eq!(tree.uncoupled().len(), 2);
    assert_eq!(tree.is_dual(), [false, false]);
    assert!(tree.innerlines().is_empty());
    assert_eq!(tree.vertices().len(), 1);
    assert_eq!(tree.vertices()[0].get(), 1);
    let sector = tree.coupled().id();
    let first = tree.uncoupled()[0].id();
    let second = tree.uncoupled()[1].id();
    assert!(sector < sector_count && first < sector_count && second < sector_count);
    assert_eq!(first ^ second, sector);
    (sector, first)
}

fn assert_bond_tree(tree: &FusionTreeKey, sector: usize) {
    assert_eq!(tree.uncoupled(), [SectorId::new(sector)]);
    assert_eq!(tree.coupled(), SectorId::new(sector));
    assert_eq!(tree.is_dual(), [false]);
    assert!(tree.innerlines().is_empty());
    assert!(tree.vertices().is_empty());
}

fn fill_multitree_eig_data<D: HarnessScalar>(
    space: &BoundDynamicFusionMapSpace<LayoutGenericRule>,
    degeneracy: usize,
    sector_count: usize,
) -> Vec<D> {
    let structure = space.space().structure();
    assert_eq!(structure.block_count(), sector_count.pow(3));
    let dimension = sector_count * degeneracy;
    let mut data = vec![D::zero(); space.space().required_len().unwrap()];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let key = block.key().as_fusion_tree_pair().unwrap();
        let (sector, row_tree) = multitree_label(key.codomain_tree(), sector_count);
        let (domain_sector, column_tree) = multitree_label(key.domain_tree(), sector_count);
        assert_eq!(domain_sector, sector);
        assert_eq!(block.shape(), [degeneracy, 1, degeneracy, 1]);
        for column in 0..degeneracy {
            for row in 0..degeneracy {
                let offset =
                    block.offset() + row * block.strides()[0] + column * block.strides()[2];
                data[offset] = multitree_eig_value(
                    sector,
                    dimension,
                    row_tree * degeneracy + row,
                    column_tree * degeneracy + column,
                );
            }
        }
    }
    data
}

fn multitree_eig_fixture<D: HarnessScalar>(
    degeneracy: usize,
    sector_count: usize,
) -> Result<(CheckedLayoutInput<D>, CheckedLayoutInput<D>), Box<dyn std::error::Error>> {
    let provider = Arc::new(LayoutGenericRule);
    let first = SectorLeg::new(
        (0..sector_count).map(|sector| (SectorId::new(sector), degeneracy)),
        false,
    );
    let second = SectorLeg::new(
        (0..sector_count).map(|sector| (SectorId::new(sector), 1)),
        false,
    );
    let product = FusionProductSpace::new([first, second]);
    let homspace = FusionTreeHomSpace::new(product.clone(), product);
    let canonical = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )?;
    let canonical_data = fill_multitree_eig_data(&canonical, degeneracy, sector_count);

    let canonical_structure = canonical.space().structure();
    let mut offset = 1;
    let mut blocks = Vec::with_capacity(canonical_structure.block_count());
    for index in (0..canonical_structure.block_count()).rev() {
        let block = canonical_structure.block(index)?;
        blocks.push(BlockSpec::column_major_with_key(
            block.key().clone(),
            block.shape().to_vec(),
            offset,
        )?);
        offset += block.element_count()? + 1;
    }
    let structure = BlockStructure::from_blocks_with_rank(4, blocks)?;
    let layout = DynamicFusionMapSpace::try_from_layout(homspace, structure)?
        .try_bind_rule(provider.as_ref())?;
    let fallback = BoundDynamicFusionMapSpace::bind_generic(layout, provider)?;
    let fallback_data = fill_multitree_eig_data(&fallback, degeneracy, sector_count);
    Ok((
        CheckedLayoutInput {
            space: canonical,
            data: canonical_data,
        },
        CheckedLayoutInput {
            space: fallback,
            data: fallback_data,
        },
    ))
}

fn assert_multitree_source<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    degeneracy: usize,
    sector_count: usize,
) {
    let structure = input.space().space().structure();
    assert_eq!(structure.block_count(), sector_count.pow(3));
    assert_eq!(input.data().len(), structure.required_len().unwrap());
    let mut visited = vec![false; sector_count.pow(3)];
    let dimension = sector_count * degeneracy;
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let key = block.key().as_fusion_tree_pair().unwrap();
        let (sector, row_tree) = multitree_label(key.codomain_tree(), sector_count);
        let (domain_sector, column_tree) = multitree_label(key.domain_tree(), sector_count);
        assert_eq!(domain_sector, sector);
        assert_eq!(block.shape(), [degeneracy, 1, degeneracy, 1]);
        let visit = (sector * sector_count + row_tree) * sector_count + column_tree;
        assert!(!visited[visit]);
        visited[visit] = true;
        for column in 0..degeneracy {
            for row in 0..degeneracy {
                assert_eq!(
                    input.data()
                        [block.offset() + row * block.strides()[0] + column * block.strides()[2]]
                        .as_complex(),
                    multitree_eig_value::<D>(
                        sector,
                        dimension,
                        row_tree * degeneracy + row,
                        column_tree * degeneracy + column,
                    )
                    .as_complex()
                );
            }
        }
    }
    assert!(visited.into_iter().all(|value| value));
}

fn assert_multitree_layouts<D>(
    canonical: &CheckedLayoutInput<D>,
    fallback: &CheckedLayoutInput<D>,
    degeneracy: usize,
    sector_count: usize,
) {
    let canonical = canonical.space.space().structure();
    let fallback = fallback.space.space().structure();
    assert_eq!(canonical.block_count(), fallback.block_count());
    for index in 0..canonical.block_count() {
        let canonical_block = canonical.block(index).unwrap();
        let fallback_block = fallback.block(canonical.block_count() - index - 1).unwrap();
        assert_eq!(canonical_block.key(), fallback_block.key());
        assert_eq!(canonical_block.shape(), fallback_block.shape());
    }
    let mut fallback_offset = 1;
    for index in 0..canonical.block_count() {
        let fallback_block = fallback.block(index).unwrap();
        assert_eq!(fallback_block.offset(), fallback_offset);
        fallback_offset += fallback_block.element_count().unwrap() + 1;
    }
    let matrix_dimension = sector_count * degeneracy;
    assert_eq!(
        canonical.required_len().unwrap(),
        sector_count * matrix_dimension * matrix_dimension
    );
    assert_eq!(fallback.required_len().unwrap(), fallback_offset - 1);
}

fn multitree_factor_matrices<D: HarnessScalar>(
    factor: &BoundDynFactor<LayoutGenericRule, D>,
    degeneracy: usize,
    sector_count: usize,
    left: bool,
) -> Vec<Vec<Complex64>> {
    let structure = factor.space().space().structure();
    assert_eq!(structure.block_count(), sector_count * sector_count);
    let dimension = sector_count * degeneracy;
    let mut matrices = vec![vec![Complex64::new(0.0, 0.0); dimension * dimension]; sector_count];
    let mut visited = vec![false; sector_count * sector_count];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let key = block.key().as_fusion_tree_pair().unwrap();
        let (sector, tree) = if left {
            let value = multitree_label(key.codomain_tree(), sector_count);
            assert_bond_tree(key.domain_tree(), value.0);
            assert_eq!(block.shape(), [degeneracy, 1, dimension]);
            value
        } else {
            let value = multitree_label(key.domain_tree(), sector_count);
            assert_bond_tree(key.codomain_tree(), value.0);
            assert_eq!(block.shape(), [dimension, degeneracy, 1]);
            value
        };
        let visit = sector * sector_count + tree;
        assert!(!visited[visit]);
        visited[visit] = true;
        for column in 0..dimension {
            for row in 0..degeneracy {
                let (matrix_row, matrix_column, offset) = if left {
                    (
                        tree * degeneracy + row,
                        column,
                        block.offset() + row * block.strides()[0] + column * block.strides()[2],
                    )
                } else {
                    (
                        column,
                        tree * degeneracy + row,
                        block.offset() + column * block.strides()[0] + row * block.strides()[1],
                    )
                };
                matrices[sector][matrix_row + dimension * matrix_column] =
                    factor.data()[offset].as_complex();
            }
        }
    }
    assert!(visited.into_iter().all(|value| value));
    matrices
}

fn assert_multitree_eig<D: HarnessScalar<Eig = Complex64>>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    result: &symtn_matrixalgebra::seam::EigFullDyn<LayoutGenericRule, D>,
    degeneracy: usize,
    sector_count: usize,
) {
    assert!(Arc::ptr_eq(
        result.v().space().provider_arc(),
        input.space().provider_arc()
    ));
    assert_eq!(result.eigenvalues().len(), sector_count);
    let dimension = sector_count * degeneracy;
    let vectors = multitree_factor_matrices(result.v(), degeneracy, sector_count, true);
    for (sector, sector_vectors) in vectors.iter().enumerate() {
        let values = &result
            .eigenvalues()
            .iter()
            .find(|entry| entry.sector == SectorId::new(sector))
            .expect("EIG result contains every sector")
            .values;
        assert_eq!(values.len(), dimension);
        for column in 0..dimension {
            let eigenvalue = values[column];
            let expected = Complex64::new((sector * dimension + dimension - column) as f64, 0.0);
            assert!((eigenvalue - expected).norm() <= 1.0e-10 * expected.norm().max(1.0));
            let norm = (0..dimension)
                .map(|row| sector_vectors[row + dimension * column].norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(norm.is_finite() && norm > 0.0);
            for row in 0..dimension {
                let av = (0..dimension).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                    sum + multitree_eig_value::<D>(sector, dimension, row, inner).as_complex()
                        * sector_vectors[inner + dimension * column]
                }) / norm;
                let vd = sector_vectors[row + dimension * column] * eigenvalue / norm;
                assert!((av - vd).norm() <= 1.0e-9 * av.norm().max(vd.norm()).max(1.0));
            }
        }
    }
}

fn assert_multitree_qr<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    left: &BoundDynFactor<LayoutGenericRule, D>,
    right: &BoundDynFactor<LayoutGenericRule, D>,
    degeneracy: usize,
    sector_count: usize,
) {
    assert!(Arc::ptr_eq(
        left.space().provider_arc(),
        input.space().provider_arc()
    ));
    assert!(Arc::ptr_eq(
        right.space().provider_arc(),
        input.space().provider_arc()
    ));
    let dimension = sector_count * degeneracy;
    let left = multitree_factor_matrices(left, degeneracy, sector_count, true);
    let right = multitree_factor_matrices(right, degeneracy, sector_count, false);
    for sector in 0..sector_count {
        for column in 0..dimension {
            for row in 0..dimension {
                let actual = (0..dimension).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                    sum + left[sector][row + dimension * inner]
                        * right[sector][inner + dimension * column]
                });
                let expected =
                    multitree_eig_value::<D>(sector, dimension, row, column).as_complex();
                assert!((actual - expected).norm() <= 2.0e-9 * expected.norm().max(1.0));
            }
        }
        for right_column in 0..dimension {
            for left_column in 0..dimension {
                let actual = (0..dimension).fold(Complex64::new(0.0, 0.0), |sum, row| {
                    sum + left[sector][row + dimension * left_column].conj()
                        * left[sector][row + dimension * right_column]
                });
                let expected = Complex64::new(f64::from(left_column == right_column), 0.0);
                assert!((actual - expected).norm() <= 2.0e-9);
            }
        }
    }
}

fn checked_source_block<'a, D>(
    input: &'a BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    sector: SectorId,
) -> BlockRef<'a> {
    let structure = input.space().space().structure();
    (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
        .expect("EIG fixture contains every requested sector")
}

fn assert_checked_eig_source<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
) {
    let structure = input.space().space().structure();
    for block_index in 0..structure.block_count() {
        let block = structure.block(block_index).unwrap();
        let sector = block.key().as_fusion_tree_pair().unwrap().coupled().id();
        for column in 0..block.shape()[1] {
            for row in 0..block.shape()[0] {
                assert_eq!(
                    checked_block_value(input.data(), block, row, column),
                    eig_value::<D>(sector, row, column).as_complex()
                );
            }
        }
    }
}

fn assert_eig_block(
    sector: usize,
    n: usize,
    value: impl Fn(usize) -> Complex64,
    source: impl Fn(usize, usize) -> Complex64,
    vector: impl Fn(usize, usize) -> Complex64,
) {
    for column in 0..n {
        let eigenvalue = value(column);
        let expected = Complex64::new((sector * 32 + n - column) as f64, 0.0);
        assert!((eigenvalue - expected).norm() <= 1.0e-10 * expected.norm().max(1.0));
        let column_norm = (0..n)
            .map(|row| vector(row, column).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(column_norm.is_finite() && column_norm > 0.0);
        for row in 0..n {
            let av = (0..n).fold(Complex64::new(0.0, 0.0), |sum, inner| {
                sum + source(row, inner) * vector(inner, column)
            }) / column_norm;
            let vd = vector(row, column) * eigenvalue / column_norm;
            assert!((av - vd).norm() <= 1.0e-9 * av.norm().max(vd.norm()).max(1.0));
        }
    }
}

fn assert_checked_eig<D: HarnessScalar<Eig = Complex64>>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    result: &symtn_matrixalgebra::seam::EigFullDyn<LayoutGenericRule, D>,
    sector_count: usize,
) {
    assert!(Arc::ptr_eq(
        result.v().space().provider_arc(),
        input.space().provider_arc()
    ));
    assert_eq!(
        input.space().space().structure().block_count(),
        sector_count
    );
    assert_eq!(
        result.v().space().space().structure().block_count(),
        sector_count
    );
    assert_eq!(result.eigenvalues().len(), sector_count);
    for sector in (0..sector_count).map(SectorId::new) {
        let source = checked_source_block(input, sector);
        let vectors = factor_block(result.v(), sector);
        let values = &result
            .eigenvalues()
            .iter()
            .find(|entry| entry.sector == sector)
            .expect("EIG result contains every requested sector")
            .values;
        let n = source.shape()[0];
        assert_eq!(source.shape(), [n, n]);
        assert_eq!(vectors.shape(), [n, n]);
        assert_eq!(values.len(), n);
        assert_eig_block(
            sector.id(),
            n,
            |column| values[column],
            |row, column| checked_block_value(input.data(), source, row, column),
            |row, column| checked_block_value(result.v().data(), vectors, row, column),
        );
    }
}

fn assert_mf_eig<D>(
    source: &TensorMap<U1FusionRule, D>,
    d: &TensorMap<U1FusionRule, Complex64>,
    v: &TensorMap<U1FusionRule, Complex64>,
    sector_count: usize,
) -> Result<(), Error>
where
    D: HarnessScalar<Eig = Complex64> + symtn::typed::TensorScalar,
{
    assert!(std::ptr::eq(source.provider(), d.provider()));
    assert!(std::ptr::eq(source.provider(), v.provider()));
    assert_eq!(
        (
            source.subblock_count(),
            d.subblock_count(),
            v.subblock_count()
        ),
        (sector_count, sector_count, sector_count)
    );
    for block_index in 0..v.subblock_count() {
        let vectors = v.subblock(block_index)?;
        let trees = v.subblock_fusion_trees(block_index)?;
        let sector = trees.coupled();
        let source_index = (0..source.subblock_count())
            .find(|&index| source.subblock_fusion_trees(index).unwrap().coupled() == sector)
            .unwrap();
        let value_index = (0..d.subblock_count())
            .find(|&index| d.subblock_fusion_trees(index).unwrap().coupled() == sector)
            .unwrap();
        let source_block = source.subblock(source_index)?;
        let values = d.subblock(value_index)?;
        let n = vectors.shape()[0];
        assert_eq!(source_block.shape(), [n, n]);
        assert_eq!(vectors.shape(), [n, n]);
        assert_eq!(values.shape(), [n, n]);
        for column in 0..n {
            for row in 0..n {
                if row != column {
                    let value = d.dense_data().unwrap()[values.offset()
                        + row * values.strides()[0]
                        + column * values.strides()[1]];
                    assert!(value.norm() <= 1.0e-12);
                }
            }
        }
        assert_eig_block(
            sector.charge() as usize,
            n,
            |column| {
                d.dense_data().unwrap()
                    [values.offset() + column * values.strides()[0] + column * values.strides()[1]]
            },
            |row, column| {
                source.dense_data().unwrap()[source_block.offset()
                    + row * source_block.strides()[0]
                    + column * source_block.strides()[1]]
                    .as_complex()
            },
            |row, column| {
                v.dense_data().unwrap()
                    [vectors.offset() + row * vectors.strides()[0] + column * vectors.strides()[1]]
            },
        );
    }
    Ok(())
}

fn run_checked_eig<D: HarnessScalar<Eig = Complex64>>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_runtime = benchmark_runtime()?;
    bench(
        &setup_runtime,
        &format!("EigGeometry-{workload}-{}", D::NAME),
        "checked_eig_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || {
            drop(black_box(eig_fixture::<D>(degeneracy, sector_count)?));
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    let (canonical, fallback) = eig_fixture::<D>(degeneracy, sector_count)?;
    let (canonical_changed, fallback_changed) = eig_fixture::<D>(degeneracy + 1, sector_count)?;
    let (control_canonical, control_fallback) =
        checked_layout_fixture_with_aspect(degeneracy, sector_count, CheckedMatrixAspect::Square)?;
    let (control_canonical_changed, control_fallback_changed) = checked_layout_fixture_with_aspect(
        degeneracy + 1,
        sector_count,
        CheckedMatrixAspect::Square,
    )?;
    for (layout, fixture, changed_fixture, control, changed_control) in [
        (
            "canonical",
            canonical,
            canonical_changed,
            control_canonical,
            control_canonical_changed,
        ),
        (
            "padded_reordered",
            fallback,
            fallback_changed,
            control_fallback,
            control_fallback_changed,
        ),
    ] {
        let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data)?;
        let changed_input =
            BoundDynamicTensorRef::try_new(&changed_fixture.space, &changed_fixture.data)?;
        let control_input = BoundDynamicTensorRef::try_new(&control.space, &control.data)?;
        let changed_control_input =
            BoundDynamicTensorRef::try_new(&changed_control.space, &changed_control.data)?;
        let mut preflight = DefaultDenseExecutor::new();
        for selected in [&input, &changed_input] {
            let eig =
                eig_full_dense(&mut preflight, selected).map_err(checked_compact_example_error)?;
            assert_checked_eig(selected, &eig, sector_count);
            drop(eig);
        }
        for selected in [&control_input, &changed_control_input] {
            let qr = qr_compact_dense(&mut preflight, selected)
                .map_err(checked_compact_example_error)?;
            assert_checked_pair_reconstructs(&qr.q, &qr.r, sector_count);
            drop(qr);
        }
        drop(preflight);
        let symmetry = format!("EigGeometry-{workload}-{layout}-{}", D::NAME);
        for (operation, is_eig) in [("checked_eig", true), ("compact_qr_control", false)] {
            let runtime = benchmark_runtime()?;
            let mut dense = DefaultDenseExecutor::new();
            let fixed = if is_eig { &input } else { &control_input };
            let shape_changed = if is_eig {
                &changed_input
            } else {
                &changed_control_input
            };
            bench(
                &runtime,
                &symmetry,
                operation,
                "owned",
                "first_after_preflight",
                "warm_after_preflight",
                min_time,
                || {
                    if is_eig {
                        drop(black_box(
                            eig_full_dense(&mut dense, &input)
                                .map_err(checked_compact_example_error)?,
                        ));
                    } else {
                        drop(black_box(
                            qr_compact_dense(&mut dense, fixed)
                                .map_err(checked_compact_example_error)?,
                        ));
                    }
                    Ok::<_, Error>(())
                },
            )?;
            let mut changed = false;
            bench(
                &runtime,
                &symmetry,
                &format!("{operation}_shape_alternating"),
                "owned",
                "first_after_preflight",
                "warm_shape_alternating",
                min_time,
                || {
                    let selected = if changed { shape_changed } else { fixed };
                    changed = !changed;
                    if is_eig {
                        drop(black_box(
                            eig_full_dense(&mut dense, selected)
                                .map_err(checked_compact_example_error)?,
                        ));
                    } else {
                        drop(black_box(
                            qr_compact_dense(&mut dense, selected)
                                .map_err(checked_compact_example_error)?,
                        ));
                    }
                    Ok::<_, Error>(())
                },
            )?;
        }
        assert_checked_eig_source(&input);
        assert_checked_eig_source(&changed_input);
    }
    Ok(())
}

fn run_mf_eig<D>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>>
where
    D: HarnessScalar<Eig = Complex64> + symtn::typed::AdvancedLinalgScalar,
{
    let runtime = benchmark_runtime()?;
    let make = |d| -> Result<_, Box<dyn std::error::Error>> {
        let space = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..sector_count).map(|sector| (U1Irrep::new(sector as i32), d)),
        )?;
        Ok(TensorMap::<U1FusionRule, D>::from_subblock_fn(
            &runtime,
            [&space],
            [&space],
            |trees, index| eig_value(trees.coupled().charge() as usize, index[0], index[1]),
        )?)
    };
    bench(
        &runtime,
        &format!("EigGeometry-MF-{workload}-{}", D::NAME),
        "mf_eig_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || {
            drop(black_box(make(degeneracy)?));
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    let source = make(degeneracy)?;
    let changed_source = make(degeneracy + 1)?;
    for selected in [&source, &changed_source] {
        let Eig { d, v } = selected.eig_full(&[0], &[1])?;
        assert_mf_eig(selected, &d, &v, sector_count)?;
        drop((d, v));
    }
    let symmetry = format!("EigGeometry-MF-{workload}-{}", D::NAME);
    bench(
        &runtime,
        &symmetry,
        "mf_eig",
        "owned",
        "first_after_preflight",
        "warm_after_preflight",
        min_time,
        || {
            drop(black_box(source.eig_full(&[0], &[1])?));
            Ok::<_, Error>(())
        },
    )?;
    let mut changed = false;
    bench(
        &runtime,
        &symmetry,
        "mf_eig_shape_alternating",
        "owned",
        "first_after_preflight",
        "warm_shape_alternating",
        min_time,
        || {
            let selected = if changed { &changed_source } else { &source };
            changed = !changed;
            drop(black_box(selected.eig_full(&[0], &[1])?));
            Ok::<_, Error>(())
        },
    )?;
    for selected in [&source, &changed_source] {
        for block_index in 0..selected.subblock_count() {
            let block = selected.subblock(block_index)?;
            let sector = selected
                .subblock_fusion_trees(block_index)?
                .coupled()
                .charge() as usize;
            for column in 0..block.shape()[1] {
                for row in 0..block.shape()[0] {
                    assert_eq!(
                        selected.dense_data().unwrap()[block.offset()
                            + row * block.strides()[0]
                            + column * block.strides()[1]]
                            .as_complex(),
                        eig_value::<D>(sector, row, column).as_complex()
                    );
                }
            }
        }
    }
    Ok(())
}

fn run_eig_source_geometry(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    if !operation_enabled("eig_source_geometry") || !form_enabled("owned") {
        return Ok(());
    }
    for (workload, d, sectors) in [
        ("few-large", degeneracy, 2),
        ("many-small", (degeneracy / 16).max(1), 16),
    ] {
        run_checked_eig::<f64>(workload, d, sectors, min_time)?;
        run_checked_eig::<Complex64>(workload, d, sectors, min_time)?;
        run_mf_eig::<f64>(workload, d, sectors, min_time)?;
        run_mf_eig::<Complex64>(workload, d, sectors, min_time)?;
    }
    Ok(())
}

fn run_checked_multitree_eig<D: HarnessScalar<Eig = Complex64>>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_runtime = benchmark_runtime()?;
    bench(
        &setup_runtime,
        &format!("EigPlacement-{workload}-{}", D::NAME),
        "checked_multitree_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || {
            drop(black_box(multitree_eig_fixture::<D>(
                degeneracy,
                sector_count,
            )?));
            Ok::<_, Box<dyn std::error::Error>>(())
        },
    )?;
    let (canonical, fallback) = multitree_eig_fixture::<D>(degeneracy, sector_count)?;
    let (canonical_changed, fallback_changed) =
        multitree_eig_fixture::<D>(degeneracy + 1, sector_count)?;
    assert_multitree_layouts(&canonical, &fallback, degeneracy, sector_count);
    assert_multitree_layouts(
        &canonical_changed,
        &fallback_changed,
        degeneracy + 1,
        sector_count,
    );
    for (layout, fixture, changed_fixture) in [
        ("canonical", canonical, canonical_changed),
        ("padded_reordered", fallback, fallback_changed),
    ] {
        let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data)?;
        let changed_input =
            BoundDynamicTensorRef::try_new(&changed_fixture.space, &changed_fixture.data)?;
        assert_multitree_source(&input, degeneracy, sector_count);
        assert_multitree_source(&changed_input, degeneracy + 1, sector_count);
        let mut preflight = DefaultDenseExecutor::new();
        for (selected, selected_degeneracy) in
            [(&input, degeneracy), (&changed_input, degeneracy + 1)]
        {
            let eig =
                eig_full_dense(&mut preflight, selected).map_err(checked_compact_example_error)?;
            assert_multitree_eig(selected, &eig, selected_degeneracy, sector_count);
            drop(eig);
            let qr = qr_compact_dense(&mut preflight, selected)
                .map_err(checked_compact_example_error)?;
            assert_multitree_qr(selected, &qr.q, &qr.r, selected_degeneracy, sector_count);
            drop(qr);
        }
        drop(preflight);

        let symmetry = format!("EigPlacement-{workload}-{layout}-{}", D::NAME);
        for (operation, is_eig) in [
            ("checked_multitree_eig", true),
            ("compact_qr_multitree_control", false),
        ] {
            let runtime = benchmark_runtime()?;
            let mut dense = DefaultDenseExecutor::new();
            bench(
                &runtime,
                &symmetry,
                operation,
                "owned",
                "first_after_preflight",
                "warm_after_preflight",
                min_time,
                || {
                    if is_eig {
                        drop(black_box(
                            eig_full_dense(&mut dense, &input)
                                .map_err(checked_compact_example_error)?,
                        ));
                    } else {
                        drop(black_box(
                            qr_compact_dense(&mut dense, &input)
                                .map_err(checked_compact_example_error)?,
                        ));
                    }
                    Ok::<_, Error>(())
                },
            )?;
            let mut changed = false;
            bench(
                &runtime,
                &symmetry,
                &format!("{operation}_shape_alternating"),
                "owned",
                "first_after_preflight",
                "warm_shape_alternating",
                min_time,
                || {
                    let selected = if changed { &changed_input } else { &input };
                    changed = !changed;
                    if is_eig {
                        drop(black_box(
                            eig_full_dense(&mut dense, selected)
                                .map_err(checked_compact_example_error)?,
                        ));
                    } else {
                        drop(black_box(
                            qr_compact_dense(&mut dense, selected)
                                .map_err(checked_compact_example_error)?,
                        ));
                    }
                    Ok::<_, Error>(())
                },
            )?;
        }
        assert_multitree_source(&input, degeneracy, sector_count);
        assert_multitree_source(&changed_input, degeneracy + 1, sector_count);
    }
    Ok(())
}

fn run_eig_output_placement(
    degeneracy: usize,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    if !operation_enabled("eig_output_placement") || !form_enabled("owned") {
        return Ok(());
    }
    for (workload, d, sectors) in [
        ("few-large", degeneracy, 2),
        ("many-small", (degeneracy / 16).max(1), 16),
    ] {
        run_checked_multitree_eig::<f64>(workload, d, sectors, min_time)?;
        run_checked_multitree_eig::<Complex64>(workload, d, sectors, min_time)?;
        run_mf_eig::<f64>(workload, d, sectors, min_time)?;
        run_mf_eig::<Complex64>(workload, d, sectors, min_time)?;
    }
    Ok(())
}

fn assert_full_qr_lq<D: HarnessScalar>(
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
    left: &BoundDynFactor<LayoutGenericRule, D>,
    right: &BoundDynFactor<LayoutGenericRule, D>,
    operation: &str,
    sector_count: usize,
) {
    assert!(Arc::ptr_eq(
        input.space().provider_arc(),
        left.space().provider_arc()
    ));
    assert!(Arc::ptr_eq(
        input.space().provider_arc(),
        right.space().provider_arc()
    ));
    assert_checked_pair_reconstructs(left, right, sector_count);
    for sector in (0..sector_count).map(SectorId::new) {
        let source = (0..input.space().space().structure().block_count())
            .map(|index| input.space().space().structure().block(index).unwrap())
            .find(|block| block.key().as_fusion_tree_pair().unwrap().coupled() == sector)
            .unwrap();
        let left_block = factor_block(left, sector);
        let right_block = factor_block(right, sector);
        let (rows, columns) = (source.shape()[0], source.shape()[1]);
        let gauge_factor = if operation == "qr" { right } else { left };
        let gauge_block = factor_block(gauge_factor, sector);
        if operation == "qr" {
            assert_eq!(left_block.shape(), [rows, rows]);
            assert_eq!(right_block.shape(), [rows, columns]);
        } else {
            assert_eq!(left_block.shape(), [rows, columns]);
            assert_eq!(right_block.shape(), [columns, columns]);
        }
        for diagonal in 0..rows.min(columns) {
            let value = checked_block_value(gauge_factor.data(), gauge_block, diagonal, diagonal);
            assert!(value.im.abs() <= 2.0e-10 && value.re >= 0.0);
        }
    }
    if operation == "qr" {
        assert_columns_orthonormal(left, sector_count);
    } else {
        assert_rows_orthonormal(right, sector_count);
    }
}

fn run_checked_full_qr_lq<D: HarnessScalar>(
    operation: &str,
    dense: &mut DefaultDenseExecutor,
    input: &BoundDynamicTensorRef<'_, LayoutGenericRule, D>,
) -> Result<(), Error> {
    if operation == "qr" {
        drop(black_box(
            qr_full_dense(dense, input).map_err(checked_compact_example_error)?,
        ));
    } else {
        drop(black_box(
            lq_full_dense(dense, input).map_err(checked_compact_example_error)?,
        ));
    }
    Ok(())
}

fn run_full_qr_fixture<D: HarnessScalar>(
    workload: &str,
    degeneracy: usize,
    sector_count: usize,
    shape_name: &str,
    aspect: CheckedMatrixAspect,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let setup_runtime = benchmark_runtime()?;
    let setup_symmetry = format!("FullQr-{workload}-{shape_name}-{}", D::NAME);
    drop(bench(
        &setup_runtime,
        &setup_symmetry,
        "full_qr_fixture",
        "setup",
        "fixture_first",
        "fixture_repeat",
        min_time,
        || checked_layout_fixture_with_aspect::<D>(degeneracy, sector_count, aspect),
    )?);
    let (canonical, padded) =
        checked_layout_fixture_with_aspect::<D>(degeneracy, sector_count, aspect)?;
    let (canonical_changed, padded_changed) =
        checked_layout_fixture_with_aspect::<D>(degeneracy + 1, sector_count, aspect)?;
    for (layout, fixture, changed_fixture) in [
        ("canonical", canonical, canonical_changed),
        ("padded", padded, padded_changed),
    ] {
        let input = BoundDynamicTensorRef::try_new(&fixture.space, &fixture.data)?;
        let changed =
            BoundDynamicTensorRef::try_new(&changed_fixture.space, &changed_fixture.data)?;
        let original = fixture.data.clone();
        let changed_original = changed_fixture.data.clone();
        let mut preflight_dense = DefaultDenseExecutor::new();
        for selected in [&input, &changed] {
            let qr = qr_full_dense(&mut preflight_dense, selected)
                .map_err(checked_compact_example_error)?;
            assert_full_qr_lq(selected, &qr.q, &qr.r, "qr", sector_count);
            drop(qr);
            let lq = lq_full_dense(&mut preflight_dense, selected)
                .map_err(checked_compact_example_error)?;
            assert_full_qr_lq(selected, &lq.l, &lq.q, "lq", sector_count);
            drop(lq);
        }
        drop(preflight_dense);
        let symmetry = format!("FullQr-{workload}-{shape_name}-{layout}-{}", D::NAME);
        for operation in ["qr", "lq"] {
            let runtime = benchmark_runtime()?;
            let mut dense = DefaultDenseExecutor::new();
            bench(
                &runtime,
                &symmetry,
                &format!("checked_full_{operation}"),
                "owned",
                "first_after_setup",
                "warm_after_setup",
                min_time,
                || run_checked_full_qr_lq(operation, &mut dense, &input),
            )?;
            let mut alternate = false;
            bench(
                &runtime,
                &symmetry,
                &format!("checked_full_{operation}_shape_alternating"),
                "owned",
                "first_after_inputs_setup",
                "warm_shape_alternating",
                min_time,
                || {
                    let selected = if alternate { &changed } else { &input };
                    alternate = !alternate;
                    run_checked_full_qr_lq(operation, &mut dense, selected)
                },
            )?;
        }
        assert_checked_source_unchanged(&fixture.data, &original);
        assert_checked_source_unchanged(&changed_fixture.data, &changed_original);
    }
    Ok(())
}

fn run_full_qr_lowering(min_time: Duration) -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("OP_MATRIX_OPERATION").as_deref() != Ok("full_qr_lowering")
        || !form_enabled("owned")
    {
        return Ok(());
    }
    let min_time = min_time.max(Duration::from_millis(100));
    println!("# FullQr: fixed_global_D=32 geometries=few-large:G2:d32,many-small:G16:d2 shapes=square,wide,tall layouts=canonical,padded dtypes=f64,c64 fixture_and_literal_preflight_outside_factor_timers dense_executor=DefaultDenseExecutor factorization_provider=compiled_cpu_features OP_MATRIX_GEMM_BACKEND_does_not_select_factorization_provider threads=wrapper_RAYON_BLAS_1");
    for (workload, degeneracy, sectors) in [("few-large", 32, 2), ("many-small", 2, 16)] {
        for (shape_name, aspect) in [
            ("square", CheckedMatrixAspect::Square),
            ("wide", CheckedMatrixAspect::Wide),
            ("tall", CheckedMatrixAspect::Tall),
        ] {
            run_full_qr_fixture::<f64>(
                workload, degeneracy, sectors, shape_name, aspect, min_time,
            )?;
            run_full_qr_fixture::<Complex64>(
                workload, degeneracy, sectors, shape_name, aspect, min_time,
            )?;
        }
    }
    Ok(())
}

macro_rules! run_provider {
    ($symmetry:literal, $rule:ty, $space:expr, $min_time:expr) => {{
        let space = $space;
        for operation in ["permute", "transpose", "repartition"] {
            if !operation_enabled(operation) {
                continue;
            }
            for form in ["owned", "destination"] {
                if !form_enabled(form) {
                    continue;
                }
                let runtime = benchmark_runtime()?;
                let source = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space],
                    724,
                )?;
                if form == "owned" {
                    let cold = bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "cold",
                        "warm",
                        $min_time,
                        || match operation {
                            "permute" => source.permute(&[1], &[2, 0]),
                            "transpose" => source.transpose(&[2], &[1, 0]),
                            "repartition" => source.repartition(1),
                            _ => unreachable!("fixed tree-operation table"),
                        },
                    )?;
                    assert!(cold.norm(2.0)?.is_finite());
                    let expected = match operation {
                        "permute" => source.permute(&[1], &[2, 0])?,
                        "transpose" => source.transpose(&[2], &[1, 0])?,
                        "repartition" => source.repartition(1)?,
                        _ => unreachable!("fixed tree-operation table"),
                    };
                    assert_same_tensor!(cold, expected, source);
                } else {
                    let expected = match operation {
                        "permute" => source.permute(&[1], &[2, 0])?,
                        "transpose" => source.transpose(&[2], &[1, 0])?,
                        "repartition" => source.repartition(1)?,
                        _ => unreachable!("fixed tree-operation table"),
                    };
                    let mut destination = expected.zeros_like().unwrap();
                    bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "first_after_setup",
                        "warm_after_setup",
                        $min_time,
                        || match operation {
                            "permute" => {
                                source.permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
                            }
                            "transpose" => {
                                source.transpose_into(&[2], &[1, 0], &mut destination, 1.0, 0.0)
                            }
                            "repartition" => source.repartition_into(&mut destination, 1.0, 0.0),
                            _ => unreachable!("fixed tree-operation table"),
                        },
                    )?;
                    assert_same_tensor!(destination, expected, source);
                }
            }
        }

        for operation in ["trace", "trace_adjoint"] {
            if !operation_enabled(operation) || !form_enabled("owned") {
                continue;
            }
            let runtime = benchmark_runtime()?;
            let source = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                727,
            )?;
            let logical_source = match operation {
                "trace" => source,
                "trace_adjoint" => source.adjoint()?,
                _ => unreachable!("fixed trace-operation table"),
            };
            let traced = bench(
                &runtime,
                $symmetry,
                operation,
                "owned",
                "cold",
                "warm",
                $min_time,
                || logical_source.trace_pairs(&[(1, 2)]),
            )?;
            assert!(traced.norm(2.0)?.is_finite());
            let expected = logical_source.trace_pairs(&[(1, 2)])?;
            assert_same_tensor!(traced, expected, logical_source);
        }

        let runtime = benchmark_runtime()?;
        let lhs = TensorMap::<$rule, f64>::rand_with_seed(
            &runtime,
            [&space, &space],
            [&space, &space],
            728,
        )?;
        let rhs = TensorMap::<$rule, f64>::rand_with_seed(
            &runtime,
            [&space, &space],
            [&space, &space],
            729,
        )?;
        let lhs_adjoint = lhs.adjoint()?;
        let rhs_adjoint = rhs.adjoint()?;
        for (suffix, left, right) in [("", &lhs, &rhs), ("_adjoint", &lhs_adjoint, &rhs_adjoint)] {
            let scale_name = format!("scale{suffix}");
            if operation_enabled(&scale_name) && form_enabled("owned") {
                let scaled = bench(
                    &runtime,
                    $symmetry,
                    &scale_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || Ok::<_, Error>(left.scale(0.5).unwrap()),
                )?;
                let error = (scaled.norm(2.0)? - 0.5 * left.norm(2.0)?).abs();
                assert!(error <= 256.0 * f64::EPSILON * left.norm(2.0)?.max(1.0));
            }

            let add_name = format!("add{suffix}");
            if operation_enabled(&add_name) && form_enabled("owned") {
                let alpha = 0.75;
                let beta = -0.25;
                let added = bench(
                    &runtime,
                    $symmetry,
                    &add_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || left.axpby(alpha, right, beta),
                )?;
                let expected_norm_squared = alpha * alpha * left.norm(2.0)?.powi(2)
                    + beta * beta * right.norm(2.0)?.powi(2)
                    + 2.0 * alpha * beta * left.inner(right)?;
                let error = (added.norm(2.0)?.powi(2) - expected_norm_squared).abs();
                assert!(error <= 1024.0 * f64::EPSILON * expected_norm_squared.abs().max(1.0));
            }

            let norm_name = format!("norm{suffix}");
            if operation_enabled(&norm_name) && form_enabled("owned") {
                let value = bench(
                    &runtime,
                    $symmetry,
                    &norm_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || left.norm(2.0),
                )?;
                assert!(value.is_finite());
            }

            let inner_name = format!("inner{suffix}");
            if operation_enabled(&inner_name) && form_enabled("owned") {
                let value = bench(
                    &runtime,
                    $symmetry,
                    &inner_name,
                    "owned",
                    "cold",
                    "warm",
                    $min_time,
                    || left.inner(right),
                )?;
                assert!(value.is_finite());
            }
        }

        if operation_enabled("compose") && form_enabled("owned") {
            let runtime = benchmark_runtime()?;
            let lhs = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                725,
            )?;
            let rhs = TensorMap::<$rule, f64>::rand_with_seed(
                &runtime,
                [&space, &space],
                [&space, &space],
                726,
            )?;
            let composed = bench(
                &runtime,
                $symmetry,
                "compose",
                "owned",
                "cold",
                "warm",
                $min_time,
                || lhs.compose(&rhs),
            )?;
            let contracted = lhs.contract(
                &rhs,
                &ContractSpec {
                    lhs: &[2, 3],
                    rhs: &[0, 1],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
            )?;
            assert_same_tensor!(composed, contracted, lhs);
        }

        for (operation, lhs_axes, rhs_axes, output_axes) in [
            (
                "contract_identity",
                &[2, 3][..],
                &[0, 1][..],
                &[0, 1, 2, 3][..],
            ),
            (
                "contract_input_swap",
                &[3, 2][..],
                &[0, 1][..],
                &[0, 1, 2, 3][..],
            ),
            (
                "contract_input_output_swap",
                &[3, 2][..],
                &[0, 1][..],
                &[1, 0, 2, 3][..],
            ),
        ] {
            if !operation_enabled(operation) {
                continue;
            }
            // Both operands are rank (2, 2) and contract two legs.
            let (codomain, domain) = output_axes.split_at(2);
            let spec = ContractSpec {
                lhs: lhs_axes,
                rhs: rhs_axes,
                codomain,
                domain,
            };
            for form in ["owned", "destination"] {
                if !form_enabled(form) {
                    continue;
                }
                let runtime = benchmark_runtime()?;
                let lhs = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    725,
                )?;
                let rhs = TensorMap::<$rule, f64>::rand_with_seed(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    726,
                )?;
                if form == "owned" {
                    let cold = bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "cold",
                        "warm",
                        $min_time,
                        || lhs.contract(&rhs, &spec),
                    )?;
                    assert!(cold.norm(2.0)?.is_finite());
                    let expected = lhs.contract(&rhs, &spec)?;
                    assert_same_tensor!(cold, expected, lhs);
                } else {
                    let expected = lhs.contract(&rhs, &spec)?;
                    let mut destination = expected.zeros_like().unwrap();
                    bench(
                        &runtime,
                        $symmetry,
                        operation,
                        form,
                        "first_after_setup",
                        "warm_after_setup",
                        $min_time,
                        || lhs.contract_into(&rhs, &spec, &mut destination, 1.0, 0.0),
                    )?;
                    assert_same_tensor!(destination, expected, lhs);
                }
            }
        }
    }};
}

#[cfg(feature = "cgc-gen")]
fn run_checked_sun(
    symmetry: &str,
    provider: std::sync::Arc<symtn::sector::SUNFusionRule>,
    label: Vec<i64>,
    degeneracy: usize,
    min_time: Duration,
    qr_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use symtn::sector::SUNFusionRule;

    let space = GradedSpace::try_new(provider, [(label, degeneracy)])?;
    if form_enabled("destination") {
        println!("# {symmetry}: destination rows excluded: the public destination methods retain multiplicity-free dispatch bounds, so the exact SUN fixtures cannot call them");
    }
    for operation in ["permute", "transpose", "repartition"] {
        if qr_only || !operation_enabled(operation) || !form_enabled("owned") {
            continue;
        }
        let runtime = benchmark_runtime()?;
        let source = TensorMap::<SUNFusionRule, f64>::rand_with_seed(
            &runtime,
            [&space, &space],
            [&space],
            724,
        )?;
        let cold = bench(
            &runtime,
            symmetry,
            operation,
            "owned",
            "cold",
            "warm",
            min_time,
            || match operation {
                "permute" => source.permute(&[1], &[2, 0]),
                "transpose" => source.transpose(&[2], &[1, 0]),
                "repartition" => source.repartition(1),
                _ => unreachable!("fixed tree-operation table"),
            },
        )?;
        let expected = match operation {
            "permute" => source.permute(&[1], &[2, 0])?,
            "transpose" => source.transpose(&[2], &[1, 0])?,
            "repartition" => source.repartition(1)?,
            _ => unreachable!("fixed tree-operation table"),
        };
        assert_same_tensor!(cold, expected, source);
    }

    if !qr_only && (operation_enabled("trace") || operation_enabled("trace_adjoint")) {
        println!("# {symmetry}: trace rows excluded: checked-Generic trace dispatch exists, but SUNFusionRule lacks the required SectorCodec");
    }

    let runtime = benchmark_runtime()?;
    let lhs = TensorMap::<SUNFusionRule, f64>::rand_with_seed(
        &runtime,
        [&space, &space],
        [&space, &space],
        725,
    )?;
    let rhs = TensorMap::<SUNFusionRule, f64>::rand_with_seed(
        &runtime,
        [&space, &space],
        [&space, &space],
        726,
    )?;
    if operation_enabled("qr_compact") && form_enabled("owned") {
        let mut sectors = Vec::new();
        let mut row_trees = Vec::new();
        let mut col_trees = Vec::new();
        for index in 0..lhs.subblock_count() {
            let trees = lhs.subblock_fusion_trees(index)?;
            if !sectors.contains(trees.coupled()) {
                sectors.push(trees.coupled().clone());
            }
            let row = (
                trees.coupled().clone(),
                trees.codomain_uncoupled().to_vec(),
                trees.codomain_innerlines().to_vec(),
                trees.codomain_vertices().to_vec(),
            );
            if !row_trees.contains(&row) {
                row_trees.push(row);
            }
            let column = (
                trees.coupled().clone(),
                trees.domain_uncoupled().to_vec(),
                trees.domain_innerlines().to_vec(),
                trees.domain_vertices().to_vec(),
            );
            if !col_trees.contains(&column) {
                col_trees.push(column);
            }
        }
        println!(
            "# {symmetry}: qr_fixture_matrices={} row_trees={} col_trees={} source_blocks={}",
            sectors.len(),
            row_trees.len(),
            col_trees.len(),
            lhs.subblock_count()
        );
    }
    for (operation, action) in [
        ("scale", 0),
        ("add", 1),
        ("norm", 2),
        ("inner", 3),
        ("compose", 4),
        ("contract_identity", 5),
        ("contract_input_swap", 6),
        ("contract_input_output_swap", 7),
        ("qr_compact", 8),
    ] {
        if (qr_only && action != 8) || !operation_enabled(operation) || !form_enabled("owned") {
            continue;
        }
        match action {
            0 => {
                let scaled = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || {
                        Ok::<_, symtn::typed::GenericTensorError<symtn::sector::SUNFusionRuleError>>(
                            lhs.scale(0.5).unwrap(),
                        )
                    },
                )?;
                for (&actual, &input) in scaled
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(lhs.dense_data().unwrap())
                {
                    assert_eq!(actual, 0.5 * input);
                }
                assert_same_tensor!(scaled, lhs.scale(0.5).unwrap(), lhs);
            }
            1 => {
                let added = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.axpby(0.75, &rhs, -0.25),
                )?;
                for ((&actual, &left), &right) in added
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(lhs.dense_data().unwrap())
                    .zip(rhs.dense_data().unwrap())
                {
                    assert_eq!(actual, 0.75 * left - 0.25 * right);
                }
                assert_same_tensor!(added, lhs.axpby(0.75, &rhs, -0.25)?, lhs);
            }
            2 => {
                let norm = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.norm(2.0),
                )?;
                let inner_self = lhs.inner(&lhs)?;
                assert!(
                    (norm * norm - inner_self).abs()
                        <= 256.0 * f64::EPSILON * inner_self.abs().max(1.0)
                );
            }
            3 => {
                let inner = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.inner(&rhs),
                )?;
                let reverse = rhs.inner(&lhs)?;
                assert!((inner - reverse).abs() <= 256.0 * f64::EPSILON * inner.abs().max(1.0));
            }
            4 => {
                let composed = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.compose(&rhs),
                )?;
                assert_same_tensor!(
                    composed,
                    lhs.contract(
                        &rhs,
                        &ContractSpec {
                            lhs: &[2, 3],
                            rhs: &[0, 1],
                            codomain: &[0, 1],
                            domain: &[2, 3]
                        }
                    )?,
                    lhs
                );
            }
            8 => {
                let Qr { q, r } = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.qr_compact(&[0, 1], &[2, 3]),
                )?;
                assert_same_tensor!(q.compose(&r)?, lhs, lhs);
            }
            _ => {
                let (lhs_axes, output_axes) = match action {
                    5 => (&[2, 3][..], &[0, 1, 2, 3][..]),
                    6 => (&[3, 2][..], &[0, 1, 2, 3][..]),
                    7 => (&[3, 2][..], &[1, 0, 2, 3][..]),
                    _ => unreachable!(),
                };
                let (codomain, domain) = output_axes.split_at(2);
                let spec = ContractSpec {
                    lhs: lhs_axes,
                    rhs: &[0, 1],
                    codomain,
                    domain,
                };
                let output = bench(
                    &runtime,
                    symmetry,
                    operation,
                    "owned",
                    "cold",
                    "warm",
                    min_time,
                    || lhs.contract(&rhs, &spec),
                )?;
                assert_same_tensor!(output, lhs.contract(&rhs, &spec)?, lhs);
            }
        }
    }
    Ok(())
}

fn matrix_op_tag(op: MatrixOp) -> &'static str {
    match op {
        MatrixOp::Identity => "I",
        MatrixOp::Transpose => "T",
        MatrixOp::Adjoint => "A",
    }
}

fn run_adapter_case<T: OrientedScalar>(
    class: &str,
    shapes: &[(usize, usize, usize)],
    expected_runs: &[usize],
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let form = format!(
        "{class}_{}_{}{}",
        T::NAME,
        matrix_op_tag(lhs_op),
        matrix_op_tag(rhs_op)
    );
    let mut preflight = adapter_fixture::<T>(shapes);
    assert_eq!(preflight.runs, expected_runs);
    let alpha = T::alpha();
    let beta = T::sample(997);
    let expected = expected_adapter(&preflight, lhs_op, rhs_op, alpha, beta);
    let mut preflight_executor = benchmark_dense_executor()?;
    execute_adapter(
        &mut preflight_executor,
        &mut preflight,
        lhs_op,
        rhs_op,
        alpha,
        beta,
    )?;
    assert_oriented_close(&preflight.output, &expected);
    drop(preflight_executor);
    drop(preflight);
    drop(expected);

    let mut fixture = adapter_fixture::<T>(shapes);
    let expected = expected_adapter(&fixture, lhs_op, rhs_op, alpha, T::zero());
    let expected_marker = (expected[fixture.dst_base], expected.len());
    drop(expected);
    let mut executor = benchmark_dense_executor()?;
    bench_adapter(&form, min_time, expected_marker, || {
        execute_adapter(
            &mut executor,
            &mut fixture,
            lhs_op,
            rhs_op,
            alpha,
            T::zero(),
        )
    })?;
    Ok(())
}

fn run_adapter_shape_cycle<T: OrientedScalar>(
    class: &str,
    geometries: &[(usize, usize, usize)],
    jobs: usize,
    lhs_op: MatrixOp,
    min_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let form = format!("{class}_{}_{}I_shape_cycle", T::NAME, matrix_op_tag(lhs_op));
    for &geometry in geometries {
        let shapes = vec![geometry; jobs];
        let mut fixture = adapter_fixture::<T>(&shapes);
        assert_eq!(fixture.runs, [jobs]);
        let expected = expected_adapter(
            &fixture,
            lhs_op,
            MatrixOp::Identity,
            T::alpha(),
            T::sample(997),
        );
        let mut executor = benchmark_dense_executor()?;
        execute_adapter(
            &mut executor,
            &mut fixture,
            lhs_op,
            MatrixOp::Identity,
            T::alpha(),
            T::sample(997),
        )?;
        assert_oriented_close(&fixture.output, &expected);
    }

    let mut fixtures: Vec<_> = geometries
        .iter()
        .map(|&geometry| adapter_fixture::<T>(&vec![geometry; jobs]))
        .collect();
    let expected: Vec<_> = fixtures
        .iter()
        .map(|fixture| {
            let data = expected_adapter(fixture, lhs_op, MatrixOp::Identity, T::alpha(), T::zero());
            (data[fixture.dst_base], data.len())
        })
        .collect();
    let mut executor = benchmark_dense_executor()?;
    let mut next = 0usize;
    bench_adapter_shape(&form, min_time, &expected, || {
        let case = next;
        next = (next + 1) % fixtures.len();
        let (first, len) = execute_adapter(
            &mut executor,
            &mut fixtures[case],
            lhs_op,
            MatrixOp::Identity,
            T::alpha(),
            T::zero(),
        )?;
        Ok((case, first, len))
    })?;
    Ok(())
}

fn observe_owned<T: OrientedScalar>(output: TensorMap<U1FusionRule, T>) -> (usize, T, usize) {
    let marker = (
        0,
        output.dense_data().unwrap()[0],
        output.dense_data().unwrap().len(),
    );
    black_box(marker);
    drop(output);
    marker
}

macro_rules! preflight_public {
    ($fixture:expr) => {{
        let direct = $fixture.lhs.compose(&$fixture.rhs)?;
        assert_public_fixture(&direct, &$fixture.expected_direct)?;
        let composed = $fixture.lhs_adjoint.compose(&$fixture.rhs)?;
        assert_public_fixture(&composed, &$fixture.expected_adjoint)?;
        let contracted = $fixture.lhs_adjoint.contract(
            &$fixture.rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )?;
        assert_public_fixture(&contracted, &$fixture.expected_adjoint)?;
    }};
}

macro_rules! run_public_dtype {
    ($ty:ty, $class:expr, $sectors:expr, $degeneracy:expr, $min_time:expr) => {{
        for operation in ["direct_compose", "lazy_lhs_compose", "lazy_lhs_contract"] {
            let preflight_runtime = benchmark_runtime()?;
            let preflight = public_fixture::<$ty>(&preflight_runtime, $sectors, $degeneracy)?;
            preflight_public!(preflight);
            drop(preflight);
            drop(preflight_runtime);

            let runtime = benchmark_runtime()?;
            let (fixture, direct_marker, adjoint_marker) =
                discard_public_oracles(public_fixture::<$ty>(&runtime, $sectors, $degeneracy)?);
            let expected_marker = if operation == "direct_compose" {
                direct_marker
            } else {
                adjoint_marker
            };
            let form = format!("{}_{}_{}", $class, <$ty as HarnessScalar>::NAME, operation);
            let marker = match operation {
                "direct_compose" => bench(
                    &runtime,
                    "U1Public",
                    "oriented_uniform_run",
                    &form,
                    "first_fresh_runtime_after_preflight",
                    "warm_fixed",
                    $min_time,
                    || Ok::<_, Error>(observe_owned(fixture.lhs.compose(&fixture.rhs)?)),
                )?,
                "lazy_lhs_compose" => bench(
                    &runtime,
                    "U1Public",
                    "oriented_uniform_run",
                    &form,
                    "first_fresh_runtime_after_preflight",
                    "warm_fixed",
                    $min_time,
                    || Ok::<_, Error>(observe_owned(fixture.lhs_adjoint.compose(&fixture.rhs)?)),
                )?,
                "lazy_lhs_contract" => bench(
                    &runtime,
                    "U1Public",
                    "oriented_uniform_run",
                    &form,
                    "first_fresh_runtime_after_preflight",
                    "warm_fixed",
                    $min_time,
                    || {
                        Ok::<_, Error>(observe_owned(fixture.lhs_adjoint.contract(
                            &fixture.rhs,
                            &ContractSpec {
                                lhs: &[1],
                                rhs: &[0],
                                codomain: &[0],
                                domain: &[1],
                            },
                        )?))
                    },
                )?,
                _ => unreachable!(),
            };
            assert_oriented_close(&[marker.1], &[expected_marker.0]);
            assert_eq!(marker.2, expected_marker.1);
        }
    }};
}

macro_rules! run_public_shape_dtype {
    ($ty:ty, $class:expr, $sectors:expr, $degeneracies:expr, $min_time:expr) => {{
        let preflight_runtime = benchmark_runtime()?;
        for &degeneracy in $degeneracies {
            let fixture = public_fixture::<$ty>(&preflight_runtime, $sectors, degeneracy)?;
            preflight_public!(fixture);
        }
        drop(preflight_runtime);

        let runtime = benchmark_runtime()?;
        let mut fixtures = Vec::with_capacity($degeneracies.len());
        let mut expected = Vec::with_capacity($degeneracies.len());
        for &degeneracy in $degeneracies {
            let (fixture, _, adjoint_marker) =
                discard_public_oracles(public_fixture::<$ty>(&runtime, $sectors, degeneracy)?);
            fixtures.push(fixture);
            expected.push(adjoint_marker);
        }
        let form = format!(
            "{}_{}_lazy_lhs_compose_shape_cycle",
            $class,
            <$ty as HarnessScalar>::NAME
        );
        let mut next = 0usize;
        bench_public_shape(&runtime, &form, $min_time, &expected, || {
            let case = next;
            next = (next + 1) % fixtures.len();
            let output = fixtures[case].lhs_adjoint.compose(&fixtures[case].rhs)?;
            let (_, first, len) = observe_owned(output);
            Ok((case, first, len))
        })?;
    }};
}

fn run_oriented_uniform_run(min_time: Duration) -> Result<(), Box<dyn std::error::Error>> {
    let many_small = vec![(4, 3, 5); 32];
    let few_large = vec![(64, 48, 56); 4];
    let minimum_run = vec![(4, 3, 5); 2];
    let singleton = vec![(64, 48, 56)];
    let heterogeneous: Vec<_> = (0..8)
        .map(|index| if index % 2 == 0 { (4, 3, 5) } else { (5, 4, 3) })
        .collect();

    for (lhs_op, rhs_op) in [
        (MatrixOp::Identity, MatrixOp::Identity),
        (MatrixOp::Transpose, MatrixOp::Identity),
    ] {
        run_adapter_case::<f64>("many_small", &many_small, &[32], lhs_op, rhs_op, min_time)?;
        run_adapter_case::<f64>("few_large", &few_large, &[4], lhs_op, rhs_op, min_time)?;
    }
    for (lhs_op, rhs_op) in [
        (MatrixOp::Identity, MatrixOp::Identity),
        (MatrixOp::Adjoint, MatrixOp::Identity),
        (MatrixOp::Adjoint, MatrixOp::Adjoint),
    ] {
        run_adapter_case::<Complex64>("many_small", &many_small, &[32], lhs_op, rhs_op, min_time)?;
        run_adapter_case::<Complex64>("few_large", &few_large, &[4], lhs_op, rhs_op, min_time)?;
    }
    run_adapter_case::<f64>(
        "minimum_run",
        &minimum_run,
        &[2],
        MatrixOp::Transpose,
        MatrixOp::Identity,
        min_time,
    )?;
    run_adapter_case::<Complex64>(
        "minimum_run",
        &minimum_run,
        &[2],
        MatrixOp::Adjoint,
        MatrixOp::Identity,
        min_time,
    )?;
    run_adapter_case::<Complex64>(
        "singleton",
        &singleton,
        &[1],
        MatrixOp::Adjoint,
        MatrixOp::Identity,
        min_time,
    )?;
    run_adapter_case::<Complex64>(
        "heterogeneous",
        &heterogeneous,
        &[1; 8],
        MatrixOp::Adjoint,
        MatrixOp::Identity,
        min_time,
    )?;

    let small_shapes = [(4, 3, 5), (5, 4, 6), (3, 6, 4)];
    let large_shapes = [(64, 48, 56), (56, 40, 64), (72, 56, 48)];
    run_adapter_shape_cycle::<f64>(
        "many_small",
        &small_shapes,
        32,
        MatrixOp::Transpose,
        min_time,
    )?;
    run_adapter_shape_cycle::<Complex64>(
        "many_small",
        &small_shapes,
        32,
        MatrixOp::Adjoint,
        min_time,
    )?;
    run_adapter_shape_cycle::<f64>("few_large", &large_shapes, 4, MatrixOp::Transpose, min_time)?;
    run_adapter_shape_cycle::<Complex64>(
        "few_large",
        &large_shapes,
        4,
        MatrixOp::Adjoint,
        min_time,
    )?;

    run_public_dtype!(f64, "many_small", 32, 4, min_time);
    run_public_dtype!(Complex64, "many_small", 32, 4, min_time);
    run_public_dtype!(f64, "few_large", 4, 32, min_time);
    run_public_dtype!(Complex64, "few_large", 4, 32, min_time);
    run_public_shape_dtype!(f64, "many_small", 32, &[4, 5, 3], min_time);
    run_public_shape_dtype!(Complex64, "many_small", 32, &[4, 5, 3], min_time);
    run_public_shape_dtype!(f64, "few_large", 4, &[32, 33, 31], min_time);
    run_public_shape_dtype!(Complex64, "few_large", 4, &[32, 33, 31], min_time);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(operation) = std::env::var("OP_MATRIX_OPERATION") {
        if !matches!(
            operation.as_str(),
            "permute"
                | "transpose"
                | "repartition"
                | "trace"
                | "trace_adjoint"
                | "scale"
                | "scale_adjoint"
                | "add"
                | "add_adjoint"
                | "norm"
                | "norm_adjoint"
                | "inner"
                | "inner_adjoint"
                | "compose"
                | "contract_identity"
                | "contract_input_swap"
                | "contract_input_output_swap"
                | "qr_compact"
                | "qr_compact_generic_layout"
                | "checked_compact_input"
                | "eig_source_geometry"
                | "eig_output_placement"
                | "full_qr_lowering"
                | "oriented_uniform_run"
        ) {
            return Err(Box::new(Error::InvalidArgument(format!(
                "unknown OP_MATRIX_OPERATION `{operation}`"
            ))));
        }
    }
    if let Ok(form) = std::env::var("OP_MATRIX_FORM") {
        if !matches!(form.as_str(), "owned" | "destination") {
            return Err(Box::new(Error::InvalidArgument(format!(
                "unknown OP_MATRIX_FORM `{form}`"
            ))));
        }
    }
    #[cfg(not(feature = "cgc-gen"))]
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("qr_compact") {
        return Err(Box::new(Error::InvalidArgument(
            "operation-matrix qr_compact requires the cgc-gen feature".into(),
        )));
    }
    let min_ms = std::env::var("OP_MATRIX_MIN_MS")
        .ok()
        .map(|value| value.parse().expect("OP_MATRIX_MIN_MS must be an integer"))
        .unwrap_or(20);
    let degeneracy = std::env::var("OP_MATRIX_DEGENERACY")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("OP_MATRIX_DEGENERACY must be an integer")
        })
        .unwrap_or(8);
    println!(
        "# symtn_authority={}",
        std::env::var("SYMTN_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# tenferro_authority={}",
        std::env::var("TENFERRO_AUTHORITY").unwrap_or_else(|_| "unknown".into())
    );
    println!(
        "# features=cpu-faer:{} blas-provider:{} cuda:{} gemm_backend={}",
        cfg!(feature = "cpu-faer"),
        cfg!(any(
            feature = "cpu-blas",
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl"
        )),
        cfg!(feature = "cuda"),
        std::env::var("OP_MATRIX_GEMM_BACKEND").unwrap_or_else(|_| "faer".into())
    );
    println!("# degeneracy={degeneracy}");
    println!("# threads=RAYON_NUM_THREADS:{} OPENBLAS_NUM_THREADS:{} OMP_NUM_THREADS:{} MKL_NUM_THREADS:{}", env_or_unset("RAYON_NUM_THREADS"), env_or_unset("OPENBLAS_NUM_THREADS"), env_or_unset("OMP_NUM_THREADS"), env_or_unset("MKL_NUM_THREADS"));
    println!("# cold_scope=fresh Runtime tree-transform store; process-global interned structures may already be warm");
    println!(
        "# cache_mode={}",
        std::env::var("OP_MATRIX_CACHE").unwrap_or_else(|_| "enabled".into())
    );
    println!("# allocation_scope=caller-thread Rust allocation calls and requested bytes during the measured phase; excludes worker threads, native BLAS allocation, frees, and peak/live bytes");
    println!("# unavailable_counters=exact_layout_admission,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("oriented_uniform_run") {
        println!("# oriented_uniform_run_scope=DenseAdapter caller-owned destination retained; U1Public owned result first-value/length observation and drop inside every measured call");
        println!("# oriented_uniform_run_fixtures=many_small:L32:4x3x5;few_large:L4:64x48x56;minimum_run:L2:4x3x5;singleton:L1:64x48x56;heterogeneous:L8:alternating");
        println!("# oriented_uniform_run_shape_cycles=many_small:L32:[4x3x5,5x4x6,3x6x4];few_large:L4:[64x48x56,56x40x64,72x56x48]");
        println!("# oriented_uniform_run_public=many_small:32_sectors:d4;few_large:4_sectors:d32;shape_degeneracies:[d,d+1,d-1]");
        println!("# oriented_uniform_run_first_scope=fresh executor or Runtime after separate full oracle preflight; process-global metadata may already be warm");
        println!("# adapter_runtime_cache_provider_counters=NA");
        println!("# comparison_protocol=unconditional A-baseline,B-candidate,B-candidate,A-baseline; three fresh child processes per position");
    }
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("eig_source_geometry") {
        println!("# eig_source_geometry_scope=rank2 MF and checked Generic EIG complete owned result dropped inside every measured call; checked canonical and padded/reordered layouts are both affected");
        println!("# eig_source_geometry_fixtures=few-large:G2:d{degeneracy};many-small:G16:d{};changed_shape=d+1;dtypes=f64,c64;complex_input_has_nonzero_imaginary_part", (degeneracy / 16).max(1));
        println!("# eig_source_geometry_oracle=known spectrum,finite nonzero eigenvectors,scale-invariant AV=VL,provider identity,block geometry,source preservation; full preflight outside timers");
        println!("# eig_source_geometry_control=checked compact QR over identical input geometry and scalar types");
        println!("# eig_source_geometry_first_scope=first_after_preflight; Runtime and process-global metadata may already be warm");
        println!("# eig_source_geometry_cache_counters=MF rows observe their execution Runtime; checked EIG and QR use DefaultDenseExecutor directly, so printed Runtime cache counters are not execution-owned and are NA for interpretation");
        println!("# eig_source_geometry_unavailable=native_worker_allocations,frees,live_peak_bytes,exact_source_copy_bytes,backend_materialization_bytes");
        println!("# comparison_protocol=unconditional A-baseline,B-candidate,B-candidate,A-baseline; three fresh child processes per position");
    }
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("eig_output_placement") {
        println!("# eig_output_placement_scope=rank4 checked Generic full EIG affected; rank4 checked compact QR paired-publication control; rank2 MF full EIG affected");
        println!("# eig_output_placement_fixtures=few-large:G2:T2:d{degeneracy}:source_blocks8;many-small:G16:T16:d{}:source_blocks4096;changed_shape=d+1;layouts=canonical,padded_reordered;dtypes=f64,c64", (degeneracy / 16).max(1));
        println!("# eig_output_placement_oracle=literal_full_keys,known_distinct_spectrum,finite_nonzero_eigenvectors,scale_invariant_AV_equals_VLambda,QR_reconstruction_and_orthogonality,provider_identity,source_preservation;full_preflight_outside_timers");
        println!("# eig_output_placement_first_scope=first_after_preflight; Runtime and process-global metadata may already be warm; complete owned result dropped inside every measured call");
        println!("# eig_output_placement_cache_counters=MF rows observe their execution Runtime; checked EIG and QR use DefaultDenseExecutor directly, so printed Runtime cache counters are not execution-owned and are NA for interpretation");
        println!("# eig_output_placement_unavailable=native_worker_allocations,frees,live_peak_bytes,placement_comparisons,exact_copied_bytes,isolated_solver_time");
        println!("# comparison_protocol=unconditional A-baseline,B-candidate,B-candidate,A-baseline; three fresh child processes per position");
    }
    println!("symmetry,operation,form,phase,iterations,us_per_iter,tree_hits,tree_misses,tree_evictions,tree_bypasses,tree_entries_delta,tree_charged_payload_bytes_before,tree_charged_payload_bytes_after,tree_charged_payload_bytes_delta,fusion_layout_misses,fusion_layout_evictions,fusion_layout_bypasses,fusion_layout_entries_delta,fusion_layout_charged_payload_bytes_before,fusion_layout_charged_payload_bytes_after,fusion_layout_charged_payload_bytes_delta,complete_hom_hits,complete_hom_misses,complete_hom_admissions,complete_hom_evictions,complete_hom_bypasses,complete_hom_entries_delta,complete_hom_charged_bytes_before,complete_hom_charged_bytes_after,complete_hom_charged_bytes_delta,exact_layout_admission,caller_allocation_calls,caller_requested_allocation_bytes,operation_local_scratch_bytes,provider_queries,transform_passes,gemm_calls,host_device_transfers");

    let min_time = Duration::from_millis(min_ms);
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("oriented_uniform_run") {
        return run_oriented_uniform_run(min_time);
    }
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("eig_source_geometry") {
        return run_eig_source_geometry(degeneracy, min_time);
    }
    if std::env::var("OP_MATRIX_OPERATION").as_deref() == Ok("eig_output_placement") {
        return run_eig_output_placement(degeneracy, min_time);
    }
    run_layout_generic_qr(degeneracy, min_time)?;
    run_checked_compact_input(degeneracy, min_time)?;
    run_full_qr_lowering(min_time)?;
    run_provider!(
        "U1",
        U1FusionRule,
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), degeneracy),
                (U1Irrep::new(0), degeneracy),
                (U1Irrep::new(1), degeneracy),
            ]
        )?,
        min_time
    );
    run_provider!(
        "SU2",
        SU2FusionRule,
        GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), degeneracy),
                (SU2Irrep::from_twice_spin(1), degeneracy),
                (SU2Irrep::from_twice_spin(2), degeneracy),
            ]
        )?,
        min_time
    );
    #[cfg(feature = "cgc-gen")]
    {
        use std::sync::Arc;
        use symtn::sector::SUNFusionRule;

        run_checked_sun(
            "SU3[0;0]",
            Arc::new(SUNFusionRule::new(3)?),
            vec![0, 0],
            degeneracy,
            min_time,
            true,
        )?;
        run_checked_sun(
            "SU3[1;1]",
            Arc::new(SUNFusionRule::new(3)?),
            vec![1, 1],
            degeneracy,
            min_time,
            false,
        )?;
        run_checked_sun(
            "SU4[1;0;1]",
            Arc::new(SUNFusionRule::new(4)?),
            vec![1, 0, 1],
            degeneracy,
            min_time,
            false,
        )?;
    }
    Ok(())
}

fn env_or_unset(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".into())
}

// Dense-input checked factorizations through the seam's one entry per op;
// a dense source yields dense factors only.
fn dense_source<'a, R, D>(input: &'a BoundDynamicTensorRef<'_, R, D>) -> FactorSource<'a, R, D> {
    FactorSource::Dense(
        BoundDynamicTensorRef::try_new(input.space(), input.data())
            .expect("a validated input revalidates"),
    )
}

type CheckedQr<R, D> = Result<
    symtn_matrixalgebra::Qr<BoundDynFactor<R, D>>,
    CheckedGenericFactorPlanError<<R as CheckedGenericFusion>::Error>,
>;
type CheckedLq<R, D> = Result<
    symtn_matrixalgebra::Lq<BoundDynFactor<R, D>>,
    CheckedGenericFactorPlanError<<R as CheckedGenericFusion>::Error>,
>;

fn qr_dense<R, D>(
    qr: symtn_matrixalgebra::Qr<symtn_matrixalgebra::seam::FactorOutput<R, D>>,
) -> symtn_matrixalgebra::Qr<BoundDynFactor<R, D>> {
    symtn_matrixalgebra::Qr {
        q: qr.q.into_dense().expect("dense source"),
        r: qr.r.into_dense().expect("dense source"),
    }
}

fn lq_dense<R, D>(
    lq: symtn_matrixalgebra::Lq<symtn_matrixalgebra::seam::FactorOutput<R, D>>,
) -> symtn_matrixalgebra::Lq<BoundDynFactor<R, D>> {
    symtn_matrixalgebra::Lq {
        l: lq.l.into_dense().expect("dense source"),
        q: lq.q.into_dense().expect("dense source"),
    }
}

fn qr_compact_dense<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> CheckedQr<R, D>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    qr_compact_from_source::<symtn::sector::CheckedGenericAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
    .map(qr_dense)
}

fn qr_full_dense<E, R, D>(dense: &mut E, input: &BoundDynamicTensorRef<'_, R, D>) -> CheckedQr<R, D>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    qr_full_from_source::<symtn::sector::CheckedGenericAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
    .map(qr_dense)
}

fn lq_compact_dense<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> CheckedLq<R, D>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    lq_compact_from_source::<symtn::sector::CheckedGenericAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
    .map(lq_dense)
}

fn lq_full_dense<E, R, D>(dense: &mut E, input: &BoundDynamicTensorRef<'_, R, D>) -> CheckedLq<R, D>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    lq_full_from_source::<symtn::sector::CheckedGenericAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
    .map(lq_dense)
}

fn eig_full_dense<E, R, D>(
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<EigFullDyn<R, D>, CheckedGenericFactorPlanError<<R as CheckedGenericFusion>::Error>>
where
    E: symtn_dense::DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    eig_full_from_source::<symtn::sector::CheckedGenericAdmissionMode, _, _, _, _>(
        dense,
        dense_source(input),
    )
}
