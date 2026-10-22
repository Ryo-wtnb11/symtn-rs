//! Revision-pinned device performance baseline for the existing CUDA path.
//!
//! Mirrors the Host protocol of `symtn/benches/operation_matrix.rs`: a fresh
//! `Runtime` per row, the fixture built before the timer, one cold call then a
//! warm-up then `--iterations` measured calls, correctness checked after the
//! timer, environment header lines, a CSV table, and caller-thread allocation
//! counters. It adds the device columns: host/device
//! transfer calls and bytes, device buffer allocations, GEMM submissions and
//! cuSOLVER calls, read from `symtn::expert::cuda_transfer_stats`.
//!
//! This is a validation fixture. It never gates CI on wall clock, and no
//! measured value feeds a dispatch decision.
//!
//! Run through `benchmarks/cuda_operation_matrix.sh`.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "cuda_operation_matrix measures the CUDA device path and is unsupported without the \
         `cuda` feature; rebuild with `--no-default-features --features cuda,cpu-faer`"
    );
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() {
    device::run();
}

#[cfg(feature = "cuda")]
mod device {
    use super::counting_alloc;
    use std::{hint::black_box, sync::Arc, time::Instant};

    use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
    use symtn::sector::{
        product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
        U1FusionRule, U1Irrep, Z2Irrep,
    };
    use symtn::sector::{
        CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols,
        SectorCodec, TypedSectorAdmission,
    };
    use symtn::typed::FusionAlgebraError;
    use symtn::typed::{Complex32, Complex64};
    use symtn::typed::{
        ContractSpec, CudaStorage, Eigh, GradedSpace, Qr, Runtime, SpectrumMagnitude, Svd,
        TensorMap, Truncation,
    };
    use symtn_network::Network;

    /// Fixture families. The parameters are explicit CLI inputs; nothing in
    /// symtn or in this harness dispatches on them.
    const FAMILIES: [&str; 2] = ["many-small", "few-large"];

    pub(super) struct Config {
        /// Coupled block count per family, in `FAMILIES` order.
        pub blocks: [usize; 2],
        /// Per-sector degeneracy per family, in `FAMILIES` order.
        pub degeneracy: [usize; 2],
        pub iterations: usize,
        pub warmup: usize,
        pub device: usize,
        pub precision: Precision,
    }

    /// Which payload lanes one invocation emits.
    ///
    /// Not "all four by default", and the reason is a resource limit rather
    /// than a preference: the protocol builds a **fresh `Runtime` per row**,
    /// and each one creates a CUDA context and a cuTENSOR handle. Emitting all
    /// four lanes in one process roughly triples the row count and the run
    /// dies before the last provider with `cutensorCreate returned status 14`
    /// (observed on an A100, CUDA 12.6, cuTENSOR 2.5.0; see
    /// `benchmarks/history/cuda-typed-single-precision-factorizations-2026-09-21.md`).
    /// Splitting the lanes across invocations keeps each process inside the
    /// limit and leaves the pinned double-precision baseline exactly the rows
    /// and order it always had.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(super) enum Precision {
        Double,
        Single,
        All,
    }

    impl Precision {
        fn double(self) -> bool {
            matches!(self, Self::Double | Self::All)
        }

        fn single(self) -> bool {
            matches!(self, Self::Single | Self::All)
        }
    }

    fn parse_pair(name: &str, raw: &str) -> [usize; 2] {
        let mut parts = raw.split(',');
        let first = parts.next().unwrap_or_default();
        let second = parts.next().unwrap_or(first);
        if parts.next().is_some() {
            panic!("{name} takes at most two comma-separated values (many-small,few-large)");
        }
        [
            first
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} many-small value must be an integer")),
            second
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} few-large value must be an integer")),
        ]
    }

    fn parse_config() -> Config {
        let mut config = Config {
            blocks: [64, 4],
            degeneracy: [4, 64],
            iterations: 20,
            warmup: 3,
            device: 0,
            precision: Precision::Double,
        };
        let mut args = std::env::args().skip(1).filter(|arg| arg != "--bench");
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .unwrap_or_else(|| panic!("{flag} requires a value"));
            match flag.as_str() {
                "--blocks" => config.blocks = parse_pair("--blocks", &value),
                "--degeneracy" => config.degeneracy = parse_pair("--degeneracy", &value),
                "--iterations" => {
                    config.iterations = value.parse().expect("--iterations must be an integer")
                }
                "--warmup" => config.warmup = value.parse().expect("--warmup must be an integer"),
                "--device" => config.device = value.parse().expect("--device must be an integer"),
                "--precision" => {
                    config.precision = match value.as_str() {
                        "double" => Precision::Double,
                        "single" => Precision::Single,
                        "all" => Precision::All,
                        other => {
                            panic!("--precision takes `double`, `single` or `all`, not `{other}`")
                        }
                    }
                }
                other => panic!(
                    "unknown flag `{other}`; supported: --blocks, --degeneracy, --iterations, \
                     --warmup, --device, --precision"
                ),
            }
        }
        assert!(config.iterations > 0, "--iterations must be positive");
        config
    }

    /// Payload dtypes this baseline covers: all four device payloads, which
    /// differ in element size and in whether they carry an imaginary part.
    ///
    /// The bound stays `CudaFactorizationPayload` because the matrix runs
    /// device factorizations. #1336 could not add single-precision rows
    /// because that marker excluded `f32`/`Complex32`, and expected a split
    /// into a base half and a factorization half; #1341 admitted both dtypes
    /// to the marker instead, so the rows are an instantiation and the harness
    /// stays one piece. Device QR keeps its own row set (below).
    ///
    /// `CHECK_TOLERANCE` is the correctness verdict's bound, not a measured
    /// quantity. It is stated per real lane rather than shared, so a
    /// single-precision row is not judged against a double-precision epsilon.
    /// The double-precision values are unchanged, so the pinned baseline CSVs
    /// keep their `check` column bit for bit.
    pub(super) trait HarnessScalar:
        symtn::typed::CudaFactorizationPayload + SpectrumMagnitude
    {
        const NAME: &'static str;
        const CHECK_TOLERANCE: f64;
        fn entry(real: f64, imaginary: f64) -> Self;
        fn distance(self, other: Self) -> f64;
    }

    /// The double-precision verdict bound, unchanged from the revision the
    /// baseline CSVs were pinned at.
    const DOUBLE_CHECK_TOLERANCE: f64 = 1.0e-9;

    impl HarnessScalar for f64 {
        const NAME: &'static str = "f64";
        const CHECK_TOLERANCE: f64 = DOUBLE_CHECK_TOLERANCE;
        fn entry(real: f64, _imaginary: f64) -> Self {
            real
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).abs()
        }
    }

    impl HarnessScalar for Complex64 {
        const NAME: &'static str = "c64";
        const CHECK_TOLERANCE: f64 = DOUBLE_CHECK_TOLERANCE;
        fn entry(real: f64, imaginary: f64) -> Self {
            Complex64::new(real, imaginary)
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).norm()
        }
    }

    /// The verdict threshold on the single-precision lane, `2^15` times the
    /// double one (about `3.3e-5`, roughly `275 * eps(f32)`). It is an
    /// **empirical** verdict threshold, checked against the A100 smoke run
    /// (zero `not-ok` rows), not derived from an error bound. The tests derive their bounds as
    /// `K * sqrt(n) * kappa * eps(real(D))` instead.
    const SINGLE_CHECK_TOLERANCE: f64 = DOUBLE_CHECK_TOLERANCE * 32768.0;

    impl HarnessScalar for f32 {
        const NAME: &'static str = "f32";
        const CHECK_TOLERANCE: f64 = SINGLE_CHECK_TOLERANCE;
        fn entry(real: f64, _imaginary: f64) -> Self {
            real as f32
        }
        fn distance(self, other: Self) -> f64 {
            f64::from((self - other).abs())
        }
    }

    impl HarnessScalar for Complex32 {
        const NAME: &'static str = "c32";
        const CHECK_TOLERANCE: f64 = SINGLE_CHECK_TOLERANCE;
        fn entry(real: f64, imaginary: f64) -> Self {
            Complex32::new(real as f32, imaginary as f32)
        }
        fn distance(self, other: Self) -> f64 {
            f64::from((self - other).norm())
        }
    }

    #[derive(Clone, Copy, Default)]
    struct Sample {
        nanos: u64,
        alloc_calls: u64,
        alloc_bytes: u64,
        device: CudaTransferStats,
        barrier_d2h_calls: u64,
        barrier_d2h_bytes: u64,
    }

    fn stats_delta(after: CudaTransferStats, before: CudaTransferStats) -> CudaTransferStats {
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
            gauge_ops: after.gauge_ops - before.gauge_ops,
        }
    }

    /// One measured call. Setup stays outside the timed region, the owned
    /// result is handed back rather than dropped inside it, and the completion
    /// barrier runs after the clock stops with its own device traffic recorded
    /// in separate columns.
    fn measure<T>(mut operation: impl FnMut() -> T, mut barrier: impl FnMut()) -> (T, Sample) {
        let before = cuda_transfer_stats();
        counting_alloc::start();
        let start = Instant::now();
        let output = operation();
        let nanos = start.elapsed().as_nanos() as u64;
        let allocs = counting_alloc::stop();
        black_box(&output);
        let after = cuda_transfer_stats();
        barrier();
        let after_barrier = cuda_transfer_stats();
        (
            output,
            Sample {
                nanos,
                alloc_calls: allocs.calls,
                alloc_bytes: allocs.bytes,
                device: stats_delta(after, before),
                barrier_d2h_calls: after_barrier.d2h_calls - after.d2h_calls,
                barrier_d2h_bytes: after_barrier.d2h_bytes - after.d2h_bytes,
            },
        )
    }

    fn median(values: &mut [u64]) -> u64 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn summarize(samples: &[Sample]) -> (Sample, u64) {
        let field = |get: fn(&Sample) -> u64| {
            let mut values: Vec<u64> = samples.iter().map(get).collect();
            median(&mut values)
        };
        let minimum = samples.iter().map(|sample| sample.nanos).min().unwrap_or(0);
        (
            Sample {
                nanos: field(|sample| sample.nanos),
                alloc_calls: field(|sample| sample.alloc_calls),
                alloc_bytes: field(|sample| sample.alloc_bytes),
                device: CudaTransferStats {
                    h2d_calls: field(|sample| sample.device.h2d_calls),
                    h2d_bytes: field(|sample| sample.device.h2d_bytes),
                    d2h_calls: field(|sample| sample.device.d2h_calls),
                    d2h_bytes: field(|sample| sample.device.d2h_bytes),
                    device_allocs: field(|sample| sample.device.device_allocs),
                    gemm_calls: field(|sample| sample.device.gemm_calls),
                    solver_calls: field(|sample| sample.device.solver_calls),
                    copy_calls: field(|sample| sample.device.copy_calls),
                    gauge_ops: field(|sample| sample.device.gauge_ops),
                },
                barrier_d2h_calls: field(|sample| sample.barrier_d2h_calls),
                barrier_d2h_bytes: field(|sample| sample.barrier_d2h_bytes),
            },
            minimum,
        )
    }

    #[derive(Clone, Copy)]
    struct Labels<'a> {
        provider: &'a str,
        dtype: &'a str,
        family: &'a str,
        blocks: usize,
        degeneracy: usize,
        operation: &'a str,
    }

    /// One printable phase of a row. Rows are buffered rather than printed as
    /// they are measured, because the `check` column is only known once both
    /// phases have run.
    struct PhaseRow {
        phase: &'static str,
        iterations: usize,
        sample: Sample,
        minimum: u64,
    }

    /// Host arms cannot fail: their `Result` exists only so both arms share one
    /// `bench` signature.
    type Never = std::convert::Infallible;

    #[allow(clippy::too_many_arguments)]
    fn print_row(
        labels: Labels<'_>,
        target: &str,
        phase: &str,
        iterations: usize,
        sample: &Sample,
        minimum: u64,
        check: &str,
    ) {
        println!(
            "{provider},{dtype},{family},{blocks},{degeneracy},{operation},{target},{phase},\
             {iterations},{median},{minimum},{alloc_calls},{alloc_bytes},{h2d_calls},{h2d_bytes},\
             {d2h_calls},{d2h_bytes},{device_allocs},{gemm_calls},{solver_calls},{copy_calls},\
             {barrier_calls},{barrier_bytes},{check}",
            provider = labels.provider,
            dtype = labels.dtype,
            family = labels.family,
            blocks = labels.blocks,
            degeneracy = labels.degeneracy,
            operation = labels.operation,
            median = sample.nanos,
            alloc_calls = sample.alloc_calls,
            alloc_bytes = sample.alloc_bytes,
            h2d_calls = sample.device.h2d_calls,
            h2d_bytes = sample.device.h2d_bytes,
            d2h_calls = sample.device.d2h_calls,
            d2h_bytes = sample.device.d2h_bytes,
            device_allocs = sample.device.device_allocs,
            gemm_calls = sample.device.gemm_calls,
            solver_calls = sample.device.solver_calls,
            copy_calls = sample.device.copy_calls,
            barrier_calls = sample.barrier_d2h_calls,
            barrier_bytes = sample.barrier_d2h_bytes,
        );
    }

    fn print_rows(labels: Labels<'_>, target: &str, rows: &[PhaseRow], check: &str) {
        for row in rows {
            print_row(
                labels,
                target,
                row.phase,
                row.iterations,
                &row.sample,
                row.minimum,
                check,
            );
        }
    }

    fn print_skip(labels: Labels<'_>, target: &str, reason: &str) {
        print_row(
            labels,
            target,
            "skipped",
            0,
            &Sample::default(),
            0,
            &sanitize(reason),
        );
    }

    fn sanitize(text: &str) -> String {
        text.replace([',', '\n', '\r'], "; ")
    }

    /// Runs the first call, the warm-up, and the measured warm iterations of
    /// one operation, and hands back the first call's output together with the
    /// unprinted phase rows. Nothing is validated here: the caller checks the
    /// returned output after both phases have been measured.
    fn bench<T, E: std::fmt::Display>(
        config: &Config,
        first_phase: &'static str,
        mut operation: impl FnMut() -> Result<T, E>,
        mut barrier: impl FnMut(),
    ) -> Result<(T, [PhaseRow; 2]), String> {
        let (first, cold) = measure(&mut operation, &mut barrier);
        let first = first.map_err(|error| sanitize(&error.to_string()))?;
        for _ in 0..config.warmup {
            black_box(operation().map_err(|error| sanitize(&error.to_string()))?);
        }
        let mut samples = Vec::with_capacity(config.iterations);
        for _ in 0..config.iterations {
            let (output, sample) = measure(&mut operation, &mut barrier);
            drop(output.map_err(|error| sanitize(&error.to_string()))?);
            samples.push(sample);
        }
        let (warm, minimum) = summarize(&samples);
        Ok((
            first,
            [
                PhaseRow {
                    phase: first_phase,
                    iterations: 1,
                    sample: cold,
                    minimum: cold.nanos,
                },
                PhaseRow {
                    phase: "warm",
                    iterations: config.iterations,
                    sample: warm,
                    minimum,
                },
            ],
        ))
    }

    fn payload_close<D: HarnessScalar + Copy>(
        actual: &[D],
        expected: &[D],
        tolerance: f64,
    ) -> bool {
        actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(&actual, &expected)| {
                actual.distance(expected) <= tolerance * (1.0 + expected.magnitude())
            })
    }

    fn verdict(ok: bool, name: &str) -> String {
        if ok {
            format!("ok:{name}")
        } else {
            format!("MISMATCH:{name}")
        }
    }

    pub(super) fn run() {
        let config = parse_config();
        print_header(&config);
        run_all(&config);
    }

    fn print_header(config: &Config) {
        println!(
            "# symtn_authority={}",
            std::env::var("SYMTN_AUTHORITY").unwrap_or_else(|_| "unknown".into())
        );
        println!(
            "# tenferro_authority={}",
            std::env::var("TENFERRO_AUTHORITY").unwrap_or_else(|_| "unknown".into())
        );
        println!(
            "# cuda_env TENFERRO_CUTENSOR_PATH={} CUDA_VISIBLE_DEVICES={} device_ordinal={}",
            std::env::var("TENFERRO_CUTENSOR_PATH").unwrap_or_else(|_| "unset".into()),
            std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "unset".into()),
            config.device
        );
        println!(
            "# fixtures families={FAMILIES:?} blocks={:?} degeneracy={:?} rank=2 \
             shape=endomorphism[V;V]",
            config.blocks, config.degeneracy
        );
        println!(
            "# protocol fresh_Runtime_per_row fixture_before_timer first_call warmup={} \
             warm_iterations={} statistic=per_iteration_median(+ns_min) \
             correctness_checked_after_both_timed_phases",
            config.warmup, config.iterations
        );
        println!(
            "# cold_scope=cold is the first call of the measured operation on a fresh Runtime; \
             the fixture's own upload has already run, so process-global CUDA state and \
             interned space structures may be warm. Rows whose measured operation is itself \
             performed during fixture setup report first_after_setup instead of cold"
        );
        println!(
            "# sampling=one process per invocation; unlike the Host operation_matrix wrapper \
             there is no three-process median, so a row carries this process's per-iteration \
             median only"
        );
        println!(
            "# allocation_scope=caller-thread Rust allocation calls and requested bytes during \
             the timed region; excludes worker threads, frees, and device memory"
        );
        println!(
            "# device_counter_scope=caller-thread symtn_dense::cuda_transfer_stats deltas over the timed \
             region; device_allocs counts uploads plus tenferro tensors wrapped as \
             CudaDenseStorage and excludes tenferro-internal solver workspaces"
        );
        println!(
            "# peak_device_memory=NA: neither symtn nor tenferro exposes a device allocation \
             high-water mark, and a per-row nvidia-smi sample would observe the whole device \
             rather than this process's buffers"
        );
        println!(
            "# synchronization=NA: no device synchronization is reachable through symtn's \
             public API, and tenferro's with_cubecl success path does not synchronize, so a \
             timed region measures submission unless the operation itself downloads. The \
             completion barrier after each timed call is norm() on one fixture input, which \
             submits one GEMM per coupled block and downloads the per-block partials; only its \
             downloads are recorded, in the barrier_* columns, and none of its traffic is \
             included in the phase columns"
        );
        // The tenferro backend library initialization (cuTENSOR,
        // cuSOLVER/cuBLAS handles) happens at Runtime construction (#1278),
        // so that cost is reported here rather than in the `cold` cells. `first` is this
        // process's first CUDA Runtime and also pays CUDA context creation;
        // `second` is a second fresh Runtime in the same process, where the
        // context is warm but the backend instance, and therefore the
        // warm-up, is new. Both are one observation, not a median.
        let build_start = Instant::now();
        let first_runtime = Runtime::builder()
            .cuda(config.device)
            .dense_threads(1)
            .build()
            .expect("CUDA Runtime for the construction measurement");
        let first_build_ns = build_start.elapsed().as_nanos();
        drop(first_runtime);
        let build_start = Instant::now();
        let second_runtime = Runtime::builder()
            .cuda(config.device)
            .dense_threads(1)
            .build()
            .expect("CUDA Runtime for the construction measurement");
        let second_build_ns = build_start.elapsed().as_nanos();
        drop(second_runtime);
        println!(
            "# runtime_build_ns first={first_build_ns} second={second_build_ns} \
             scope=one Runtime::builder().cuda(device).dense_threads(1).build() each, one \
             observation, first includes CUDA context creation, both include the #1278 \
             backend warm-up"
        );
        println!(
            "# tensorkit_cuda_comparison=absent: neither TensorKit nor QSpace has a matched \
             CUDA fixture for these rows at this revision"
        );
        println!(
            "provider,dtype,family,blocks,degeneracy,operation,target,phase,iterations,\
             ns_median,ns_min,alloc_calls,alloc_bytes,h2d_calls,h2d_bytes,d2h_calls,d2h_bytes,\
             device_allocs,gemm_calls,solver_calls,copy_calls,barrier_d2h_calls,\
             barrier_d2h_bytes,check"
        );
    }

    struct Fixture<R, D>
    where
        R: SectorCodec,
        D: HarnessScalar,
    {
        /// Kept so the row's tensors outlive their Runtime borrow; the field
        /// itself is never read.
        #[allow(dead_code)]
        runtime: Runtime,
        host: Vec<TensorMap<R, D>>,
        device: Vec<TensorMap<R, D, CudaStorage<D>>>,
        hermitian_host: Option<TensorMap<R, D>>,
        hermitian_device: Option<TensorMap<R, D, CudaStorage<D>>>,
    }

    /// Builds one row's Runtime, Host fixture, and device upload. Everything
    /// here happens before any timer starts.
    fn fixture<R, D>(config: &Config, space: &GradedSpace<R>, count: usize) -> Fixture<R, D>
    where
        R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
            + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: HarnessScalar,
    {
        let runtime = Runtime::builder()
            .cuda(config.device)
            .dense_threads(1)
            .build()
            .expect("CUDA Runtime for the measured row");
        let host: Vec<TensorMap<R, D>> = (0..count)
            .map(|index| {
                let shift = index as f64;
                TensorMap::from_subblock_fn(&runtime, [space], [space], move |_, indices| {
                    let row = indices[0] as f64;
                    let column = indices[1] as f64;
                    D::entry(
                        1.0 + shift + 0.5 * row + 0.25 * column,
                        0.125 * row - 0.0625 * column - 0.03125 * shift,
                    )
                })
                .expect("Host fixture tensor")
            })
            .collect();
        let device = host
            .iter()
            .map(|tensor| tensor.to_cuda().expect("fixture upload"))
            .collect();
        let hermitian_host =
            TensorMap::<R, D>::from_subblock_fn(&runtime, [space], [space], |_, indices| {
                let (row, column) = (indices[0], indices[1]);
                if row == column {
                    D::entry(row as f64 + 1.0, 0.0)
                } else {
                    let magnitude = 0.25 / (1.0 + row.abs_diff(column) as f64);
                    D::entry(magnitude, if row < column { 0.125 } else { -0.125 })
                }
            })
            .ok();
        let hermitian_device = hermitian_host
            .as_ref()
            .and_then(|tensor| tensor.to_cuda().ok());
        Fixture {
            runtime,
            host,
            device,
            hermitian_host,
            hermitian_device,
        }
    }

    /// Prints both arms of a row whose device probe failed, so an unsupported
    /// boundary stays visible in the table instead of vanishing.
    fn skip_row(labels: Labels<'_>, reason: &str) {
        print_skip(labels, "cuda", reason);
        print_skip(labels, "host", reason);
    }

    fn run_dtype<R, D>(
        config: &Config,
        provider: &str,
        family: &str,
        blocks: usize,
        degeneracy: usize,
        space: &GradedSpace<R>,
    ) where
        R: TypedSectorAdmission<
                Sector = <R as symtn::sector::SectorCodec>::Sector,
                Error = FusionAlgebraError,
                Mode = MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: HarnessScalar,
    {
        let label = |operation: &'static str| Labels {
            provider,
            dtype: D::NAME,
            family,
            blocks,
            degeneracy,
            operation,
        };
        let tolerance = D::CHECK_TOLERANCE;
        let alpha = D::entry(1.5, 0.25);
        let beta = D::entry(-0.5, 0.125);
        let factor = D::entry(2.0, -0.5);

        // transfer roundtrip; the measured operations also run during fixture
        // setup, so their first phase is reported as first_after_setup
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let uploaded = &fixture.device[0];
            let barrier = || {
                let _ = uploaded.norm(2.0);
            };
            match bench(config, "first_after_setup", || source.to_cuda(), barrier) {
                Err(reason) => skip_row(label("to_cuda"), &reason),
                Ok((first, rows)) => {
                    let check = verdict(
                        payload_close(
                            first
                                .to_host()
                                .expect("roundtrip download")
                                .dense_data()
                                .unwrap(),
                            source.dense_data().unwrap(),
                            0.0,
                        ),
                        "roundtrip_bit_equal",
                    );
                    print_rows(label("to_cuda"), "cuda", &rows, &check);
                    print_skip(label("to_cuda"), "host", "no Host counterpart");
                }
            }
            match bench(config, "first_after_setup", || uploaded.to_host(), barrier) {
                Err(reason) => skip_row(label("to_host"), &reason),
                Ok((first, rows)) => {
                    let check = verdict(
                        payload_close(
                            first.dense_data().unwrap(),
                            source.dense_data().unwrap(),
                            0.0,
                        ),
                        "roundtrip_bit_equal",
                    );
                    print_rows(label("to_host"), "cuda", &rows, &check);
                    print_skip(label("to_host"), "host", "no Host counterpart");
                }
            }
        }

        // contract, canonical direct
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm(2.0);
            };
            match bench(
                config,
                "cold",
                || {
                    lhs_device.contract(
                        rhs_device,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                },
                barrier,
            ) {
                Err(reason) => skip_row(label("contract_direct"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || {
                            Ok::<_, Never>(
                                lhs.contract(
                                    rhs,
                                    &ContractSpec {
                                        lhs: &[1],
                                        rhs: &[0],
                                        codomain: &[0],
                                        domain: &[1],
                                    },
                                )
                                .expect("Host contract"),
                            )
                        },
                        || {},
                    )
                    .expect("Host contract arm");
                    let check = verdict(
                        payload_close(
                            device_first
                                .to_host()
                                .expect("download")
                                .dense_data()
                                .unwrap(),
                            host_first.dense_data().unwrap(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("contract_direct"), "cuda", &device_rows, &check);
                    print_rows(label("contract_direct"), "host", &host_rows, &check);
                }
            }
        }

        // contract with a lazy-adjoint left operand; the lazy views are built
        // once as operands, outside every timed region
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm(2.0);
            };
            match lhs_device.adjoint() {
                Err(error) => skip_row(label("contract_lazy_adjoint_lhs"), &error.to_string()),
                Ok(lhs_adjoint) => {
                    let host_adjoint = lhs.adjoint().expect("Host adjoint");
                    match bench(
                        config,
                        "cold",
                        || {
                            lhs_adjoint.contract(
                                rhs_device,
                                &ContractSpec {
                                    lhs: &[1],
                                    rhs: &[0],
                                    codomain: &[0],
                                    domain: &[1],
                                },
                            )
                        },
                        barrier,
                    ) {
                        Err(reason) => skip_row(label("contract_lazy_adjoint_lhs"), &reason),
                        Ok((device_first, device_rows)) => {
                            let (host_first, host_rows) = bench(
                                config,
                                "cold",
                                || {
                                    Ok::<_, Never>(
                                        host_adjoint
                                            .contract(
                                                rhs,
                                                &ContractSpec {
                                                    lhs: &[1],
                                                    rhs: &[0],
                                                    codomain: &[0],
                                                    domain: &[1],
                                                },
                                            )
                                            .expect("Host adjoint contract"),
                                    )
                                },
                                || {},
                            )
                            .expect("Host adjoint contract arm");
                            let check = verdict(
                                payload_close(
                                    device_first
                                        .to_host()
                                        .expect("download")
                                        .dense_data()
                                        .unwrap(),
                                    host_first.dense_data().unwrap(),
                                    tolerance,
                                ),
                                "host_value_equality",
                            );
                            print_rows(
                                label("contract_lazy_adjoint_lhs"),
                                "cuda",
                                &device_rows,
                                &check,
                            );
                            print_rows(
                                label("contract_lazy_adjoint_lhs"),
                                "host",
                                &host_rows,
                                &check,
                            );
                        }
                    }
                }
            }
        }

        // compose
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm(2.0);
            };
            match bench(config, "cold", || lhs_device.compose(rhs_device), barrier) {
                Err(reason) => skip_row(label("compose"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(lhs.compose(rhs).expect("Host compose")),
                        || {},
                    )
                    .expect("Host compose arm");
                    let check = verdict(
                        payload_close(
                            device_first
                                .to_host()
                                .expect("download")
                                .dense_data()
                                .unwrap(),
                            host_first.dense_data().unwrap(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("compose"), "cuda", &device_rows, &check);
                    print_rows(label("compose"), "host", &host_rows, &check);
                }
            }
        }

        // structural transforms (#1322): the same public call on both targets,
        // written once per row and expanded against the device and the Host
        // receiver, so the two arms cannot drift.
        macro_rules! transform_row {
            ($name:expr, |$t:ident| $call:expr) => {{
                let fixture = fixture::<R, D>(config, space, 1);
                let host_source = &fixture.host[0];
                let device_source = &fixture.device[0];
                let barrier = || {
                    let _ = device_source.norm(2.0);
                };
                match bench(
                    config,
                    "cold",
                    || {
                        let $t = device_source;
                        $call
                    },
                    barrier,
                ) {
                    Err(reason) => skip_row(label($name), &reason),
                    Ok((device_first, device_rows)) => {
                        let (host_first, host_rows) = bench(
                            config,
                            "cold",
                            || {
                                Ok::<_, Never>(
                                    {
                                        let $t = host_source;
                                        $call
                                    }
                                    .expect("Host transform"),
                                )
                            },
                            || {},
                        )
                        .expect("Host transform arm");
                        let check = verdict(
                            payload_close(
                                device_first
                                    .to_host()
                                    .expect("download")
                                    .dense_data()
                                    .unwrap(),
                                host_first.dense_data().unwrap(),
                                tolerance,
                            ),
                            "host_value_equality",
                        );
                        print_rows(label($name), "cuda", &device_rows, &check);
                        print_rows(label($name), "host", &host_rows, &check);
                    }
                }
            }};
        }
        transform_row!("permute", |t| t.permute(&[1], &[0]));
        transform_row!("braid", |t| t.braid(&[1], &[0], &[1, 2]));
        transform_row!("transpose", |t| t.transpose(&[1], &[0]));
        transform_row!("repartition", |t| t.repartition(0));

        // scale
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            let barrier = || {
                let _ = source_device.norm(2.0);
            };
            match bench(config, "cold", || source_device.scale(factor), barrier) {
                Err(reason) => skip_row(label("scale"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(source.scale(factor).unwrap()),
                        || {},
                    )
                    .expect("Host scale arm");
                    let check = verdict(
                        payload_close(
                            device_first
                                .to_host()
                                .expect("download")
                                .dense_data()
                                .unwrap(),
                            host_first.dense_data().unwrap(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("scale"), "cuda", &device_rows, &check);
                    print_rows(label("scale"), "host", &host_rows, &check);
                }
            }
        }

        // add, owned operands
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm(2.0);
            };
            match bench(
                config,
                "cold",
                || lhs_device.axpby(alpha, rhs_device, beta),
                barrier,
            ) {
                Err(reason) => skip_row(label("add_owned"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(lhs.axpby(alpha, rhs, beta).expect("Host add")),
                        || {},
                    )
                    .expect("Host add arm");
                    let check = verdict(
                        payload_close(
                            device_first
                                .to_host()
                                .expect("download")
                                .dense_data()
                                .unwrap(),
                            host_first.dense_data().unwrap(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("add_owned"), "cuda", &device_rows, &check);
                    print_rows(label("add_owned"), "host", &host_rows, &check);
                }
            }
        }

        // add, lazy-adjoint fold on both operands
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            let barrier = || {
                let _ = lhs_device.norm(2.0);
            };
            match (lhs_device.adjoint(), rhs_device.adjoint()) {
                (Err(error), _) | (_, Err(error)) => {
                    skip_row(label("add_lazy_fold"), &error.to_string())
                }
                (Ok(lhs_adjoint), Ok(rhs_adjoint)) => {
                    let host_lhs = lhs.adjoint().expect("Host adjoint");
                    let host_rhs = rhs.adjoint().expect("Host adjoint");
                    match bench(
                        config,
                        "cold",
                        || lhs_adjoint.axpby(alpha, &rhs_adjoint, beta),
                        barrier,
                    ) {
                        Err(reason) => skip_row(label("add_lazy_fold"), &reason),
                        Ok((device_first, device_rows)) => {
                            let (host_first, host_rows) = bench(
                                config,
                                "cold",
                                || {
                                    Ok::<_, Never>(
                                        host_lhs
                                            .axpby(alpha, &host_rhs, beta)
                                            .expect("Host lazy add"),
                                    )
                                },
                                || {},
                            )
                            .expect("Host lazy add arm");
                            let check = verdict(
                                payload_close(
                                    device_first
                                        .to_host()
                                        .expect("download")
                                        .materialize()
                                        .expect("materialize")
                                        .dense_data()
                                        .unwrap(),
                                    host_first
                                        .materialize()
                                        .expect("materialize")
                                        .dense_data()
                                        .unwrap(),
                                    tolerance,
                                ),
                                "host_value_equality",
                            );
                            print_rows(label("add_lazy_fold"), "cuda", &device_rows, &check);
                            print_rows(label("add_lazy_fold"), "host", &host_rows, &check);
                        }
                    }
                }
            }
        }

        // norm
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            match bench(config, "cold", || source_device.norm(2.0), || {}) {
                Err(reason) => skip_row(label("norm"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(source.norm(2.0).expect("Host norm")),
                        || {},
                    )
                    .expect("Host norm arm");
                    let check = verdict(
                        (device_first - host_first).abs() <= tolerance * (1.0 + host_first.abs()),
                        "host_scalar_equality",
                    );
                    print_rows(label("norm"), "cuda", &device_rows, &check);
                    print_rows(label("norm"), "host", &host_rows, &check);
                }
            }
        }

        // inner
        {
            let fixture = fixture::<R, D>(config, space, 2);
            let (lhs, rhs) = (&fixture.host[0], &fixture.host[1]);
            let (lhs_device, rhs_device) = (&fixture.device[0], &fixture.device[1]);
            match bench(config, "cold", || lhs_device.inner(rhs_device), || {}) {
                Err(reason) => skip_row(label("inner"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(lhs.inner(rhs).expect("Host inner")),
                        || {},
                    )
                    .expect("Host inner arm");
                    let check = verdict(
                        device_first.distance(host_first)
                            <= tolerance * (1.0 + host_first.magnitude()),
                        "host_scalar_equality",
                    );
                    print_rows(label("inner"), "cuda", &device_rows, &check);
                    print_rows(label("inner"), "host", &host_rows, &check);
                }
            }
        }

        // svd_compact
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            let barrier = || {
                let _ = source_device.norm(2.0);
            };
            match bench(
                config,
                "cold",
                || source_device.svd_compact(&[0], &[1]),
                barrier,
            ) {
                Err(reason) => skip_row(label("svd_compact"), &reason),
                Ok((Svd { u, s, vh }, device_rows)) => {
                    let (_, host_rows) = bench(
                        config,
                        "cold",
                        || {
                            Ok::<_, Never>(
                                source.svd_compact(&[0], &[1]).expect("Host svd_compact"),
                            )
                        },
                        || {},
                    )
                    .expect("Host svd_compact arm");
                    let rebuilt = u
                        .compose(&s)
                        .and_then(|left| left.compose(&vh))
                        .expect("device SVD reconstruction")
                        .to_host()
                        .expect("download");
                    let check = verdict(
                        payload_close(
                            rebuilt.dense_data().unwrap(),
                            source.dense_data().unwrap(),
                            tolerance,
                        ),
                        "device_reconstruction",
                    );
                    print_rows(label("svd_compact"), "cuda", &device_rows, &check);
                    print_rows(label("svd_compact"), "host", &host_rows, &check);
                }
            }
        }

        // Truncated SVD as a composition: device compact SVD, one D2H per
        // factor, then the Host selection and restriction, against the same
        // composition on the Host. The two arms are therefore not the same
        // amount of work; the row prices the recipe the rustdoc gives, not
        // one kernel against another.
        {
            let fixture = fixture::<R, D>(config, space, 1);
            let source = &fixture.host[0];
            let source_device = &fixture.device[0];
            let truncation = Truncation::rank(degeneracy.max(1));
            let barrier = || {
                let _ = source_device.norm(2.0);
            };
            let composed = || -> Result<_, String> {
                let Svd { u, s, vh } = source_device
                    .svd_compact(&[0], &[1])
                    .map_err(|e| e.to_string())?;
                let (u, s, vh) = (
                    u.to_host().map_err(|e| e.to_string())?,
                    s.to_host().map_err(|e| e.to_string())?,
                    vh.to_host().map_err(|e| e.to_string())?,
                );
                let found = s.domain()[0]
                    .find_truncated(&s.diagview().map_err(|e| e.to_string())?, &truncation)
                    .map_err(|e| e.to_string())?;
                let u = u
                    .restrict_leg(&[(u.codomain_rank(), &found.selection)])
                    .map_err(|e| e.to_string())?;
                let s = s
                    .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
                    .map_err(|e| e.to_string())?;
                let vh = vh
                    .restrict_leg(&[(0, &found.selection)])
                    .map_err(|e| e.to_string())?;
                Ok((u, s, vh, found.error))
            };
            match bench(config, "cold", composed, barrier) {
                Err(reason) => skip_row(label("svd_trunc_composition"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || {
                            let Svd { u, s, vh } =
                                source.svd_compact(&[0], &[1]).expect("Host svd_compact");
                            let found = s.domain()[0]
                                .find_truncated(&s.diagview().expect("Host spectrum"), &truncation)
                                .expect("Host find_truncated");
                            let restrict = "Host restriction";
                            Ok::<_, Never>((
                                u.restrict_leg(&[(u.codomain_rank(), &found.selection)])
                                    .expect(restrict),
                                s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])
                                    .expect(restrict),
                                vh.restrict_leg(&[(0, &found.selection)]).expect(restrict),
                                found.error,
                            ))
                        },
                        || {},
                    )
                    .expect("Host truncated SVD arm");
                    let kept = device_first.1.diagview().expect("kept spectrum");
                    let expected_kept = host_first.1.diagview().expect("Host kept spectrum");
                    let zero = D::entry(0.0, 0.0);
                    let kept_match = kept.len() == expected_kept.len()
                        && kept.iter().zip(&expected_kept).all(|(actual, expected)| {
                            actual.values.len() == expected.values.len()
                                && actual.values.iter().zip(&expected.values).all(
                                    |(&actual, &expected)| {
                                        actual.distance(expected)
                                            <= tolerance * (1.0 + expected.distance(zero))
                                    },
                                )
                        });
                    let check = verdict(
                        kept_match
                            && (device_first.3 - host_first.3).abs()
                                <= tolerance * (1.0 + host_first.3.abs()),
                        "host_spectrum_and_discard_weight",
                    );
                    print_rows(label("svd_trunc_composition"), "cuda", &device_rows, &check);
                    print_rows(label("svd_trunc_composition"), "host", &host_rows, &check);
                }
            }
        }

        // eigh_full, on the Hermitian fixture
        {
            let fixture = fixture::<R, D>(config, space, 1);
            match (&fixture.hermitian_host, &fixture.hermitian_device) {
                (Some(source), Some(source_device)) => {
                    let barrier = || {
                        let _ = source_device.norm(2.0);
                    };
                    match bench(
                        config,
                        "cold",
                        || source_device.eigh_full(&[0], &[1], symtn::typed::HermitianTol::DEFAULT),
                        barrier,
                    ) {
                        Err(reason) => skip_row(label("eigh_full"), &reason),
                        Ok((Eigh { d, v }, device_rows)) => {
                            let (_, host_rows) = bench(
                                config,
                                "cold",
                                || {
                                    Ok::<_, Never>(
                                        source
                                            .eigh_full(
                                                &[0],
                                                &[1],
                                                symtn::typed::HermitianTol::DEFAULT,
                                            )
                                            .expect("Host eigh_full"),
                                    )
                                },
                                || {},
                            )
                            .expect("Host eigh_full arm");
                            let rebuilt = v
                                .compose(&d)
                                .and_then(|left| {
                                    v.adjoint().and_then(|adjoint| left.compose(&adjoint))
                                })
                                .expect("device EIGH reconstruction")
                                .to_host()
                                .expect("download");
                            let check = verdict(
                                payload_close(
                                    rebuilt.dense_data().unwrap(),
                                    source.dense_data().unwrap(),
                                    tolerance,
                                ),
                                "device_reconstruction",
                            );
                            print_rows(label("eigh_full"), "cuda", &device_rows, &check);
                            print_rows(label("eigh_full"), "host", &host_rows, &check);
                        }
                    }
                }
                _ => skip_row(label("eigh_full"), "Hermitian fixture unavailable"),
            }
        }

        // canonical three-tensor network chain
        {
            let chain = Network::new(
                vec![vec!["p", "q"], vec!["q", "r"], vec!["r", "s"]],
                vec![false, false, false],
                vec![1, 1, 1],
                vec!["p", "s"],
                Some(1),
            )
            .expect("chain network");
            let fixture = fixture::<R, D>(config, space, 3);
            let (a, b, c) = (&fixture.host[0], &fixture.host[1], &fixture.host[2]);
            let (ad, bd, cd) = (&fixture.device[0], &fixture.device[1], &fixture.device[2]);
            let barrier = || {
                let _ = ad.norm(2.0);
            };
            match bench(config, "cold", || chain.contract(&[ad, bd, cd]), barrier) {
                Err(reason) => skip_row(label("network_chain3"), &reason),
                Ok((device_first, device_rows)) => {
                    let (host_first, host_rows) = bench(
                        config,
                        "cold",
                        || Ok::<_, Never>(chain.contract(&[a, b, c]).expect("Host chain")),
                        || {},
                    )
                    .expect("Host chain arm");
                    let check = verdict(
                        payload_close(
                            device_first
                                .to_host()
                                .expect("download")
                                .dense_data()
                                .unwrap(),
                            host_first.dense_data().unwrap(),
                            tolerance,
                        ),
                        "host_value_equality",
                    );
                    print_rows(label("network_chain3"), "cuda", &device_rows, &check);
                    print_rows(label("network_chain3"), "host", &host_rows, &check);
                }
            }
        }
    }

    /// `qr_compact` rows, kept apart from `run_dtype` and real-payload only so
    /// the pinned baseline CSVs keep their row set; device QR itself admits
    /// every `CudaFactorizationPayload` (#1271).
    fn run_qr<R, D>(
        config: &Config,
        provider: &str,
        family: &str,
        blocks: usize,
        degeneracy: usize,
        space: &GradedSpace<R>,
    ) where
        R: TypedSectorAdmission<
                Sector = <R as symtn::sector::SectorCodec>::Sector,
                Error = FusionAlgebraError,
                Mode = MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + Send
            + Sync
            + 'static,
        D: HarnessScalar,
    {
        let labels = Labels {
            provider,
            dtype: D::NAME,
            family,
            blocks,
            degeneracy,
            operation: "qr_compact",
        };
        let fixture = fixture::<R, D>(config, space, 1);
        let source = &fixture.host[0];
        let source_device = &fixture.device[0];
        let barrier = || {
            let _ = source_device.norm(2.0);
        };
        match bench(
            config,
            "cold",
            || source_device.qr_compact(&[0], &[1]),
            barrier,
        ) {
            Err(reason) => skip_row(labels, &reason),
            Ok((Qr { q, r }, device_rows)) => {
                let (_, host_rows) = bench(
                    config,
                    "cold",
                    || Ok::<_, Never>(source.qr_compact(&[0], &[1]).expect("Host qr_compact")),
                    || {},
                )
                .expect("Host qr_compact arm");
                let rebuilt = q
                    .compose(&r)
                    .expect("device QR reconstruction")
                    .to_host()
                    .expect("download");
                let check = verdict(
                    payload_close(
                        rebuilt.dense_data().unwrap(),
                        source.dense_data().unwrap(),
                        D::CHECK_TOLERANCE,
                    ),
                    "device_reconstruction",
                );
                print_rows(labels, "cuda", &device_rows, &check);
                print_rows(labels, "host", &host_rows, &check);
            }
        }
    }

    macro_rules! provider_rows {
        ($config:expr, $name:literal, $build:expr) => {{
            for (index, family) in FAMILIES.iter().enumerate() {
                let blocks = $config.blocks[index];
                let degeneracy = $config.degeneracy[index];
                let space = ($build)(blocks, degeneracy);
                if $config.precision.double() {
                    run_dtype::<_, f64>($config, $name, family, blocks, degeneracy, &space);
                    run_qr::<_, f64>($config, $name, family, blocks, degeneracy, &space);
                    run_dtype::<_, Complex64>($config, $name, family, blocks, degeneracy, &space);
                }
                // Single precision (#1341). Under `--precision all` these rows
                // follow the double rows of the same provider and size index,
                // so the two lanes interleave; the baseline row order is only
                // reproduced by the default `--precision double`.
                if $config.precision.single() {
                    run_dtype::<_, f32>($config, $name, family, blocks, degeneracy, &space);
                    run_qr::<_, f32>($config, $name, family, blocks, degeneracy, &space);
                    run_dtype::<_, Complex32>($config, $name, family, blocks, degeneracy, &space);
                }
            }
        }};
    }

    fn run_all(config: &Config) {
        provider_rows!(config, "U1", |blocks: usize, degeneracy: usize| {
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                (0..blocks).map(|index| (U1Irrep::new(index as i32), degeneracy)),
            )
            .expect("U(1) fixture leg")
        });
        provider_rows!(config, "fZ2", |_blocks: usize, degeneracy: usize| {
            // Fermion parity has exactly two irreps, so the realized block
            // count of this provider is 2 in both families. That is a provider
            // capability, not a size decision made by the harness.
            GradedSpace::try_new(
                Arc::new(FermionParityFusionRule),
                [(Z2Irrep::EVEN, degeneracy), (Z2Irrep::ODD, degeneracy)],
            )
            .expect("fZ2 fixture leg")
        });
        provider_rows!(config, "SU2", |blocks: usize, degeneracy: usize| {
            GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                (0..blocks).map(|index| (SU2Irrep::from_twice_spin(index), degeneracy)),
            )
            .expect("SU(2) fixture leg")
        });
        provider_rows!(config, "U1xfZ2", |blocks: usize, degeneracy: usize| {
            GradedSpace::try_new(
                Arc::new(FermionParityFusionRule.product(U1FusionRule)),
                (0..blocks).map(|index| {
                    (
                        product_sector(
                            if index % 2 == 0 {
                                Z2Irrep::EVEN
                            } else {
                                Z2Irrep::ODD
                            },
                            U1Irrep::new((index / 2) as i32),
                        ),
                        degeneracy,
                    )
                }),
            )
            .expect("fZ2 x U(1) fixture leg")
        });
    }
}
