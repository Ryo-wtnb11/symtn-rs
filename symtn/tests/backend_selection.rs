//! `RuntimeBuilder::with_dense_executor` lets a caller select the CPU
//! linear-algebra backend by injecting a `DenseExecutor` (issue #64). This
//! checks the runtime actually drives the injected executor and that doing so
//! is numerically identical to the unset built-in provider.

use std::sync::Arc;

use symtn::expert::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::OperationError;
use symtn::typed::{Error, Svd};
use symtn::typed::{GradedSpace, Runtime, RuntimeConfigError, SectorSpectrum, TensorMap};

#[test]
fn compact_diagonal_svd_submits_no_dense_svd() {
    let counts = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(SpyExecutor::counting(&counts)))
        .build()
        .unwrap();
    let leg = u1_space([(-1, 2), (0, 3), (1, 1)]);
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![1.0, -3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![0.0, -4.0, 2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![-2.0],
            },
        ],
    )
    .unwrap();
    let Svd { u, s, vh } = input.svd_compact(&[0], &[1]).unwrap();
    assert_eq!(read(&counts), (0, 0, 0, 0));
    assert_eq!(counts.total(), 0);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    for (actual, expected) in rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize().unwrap().dense_data().unwrap())
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

fn u1_space(entries: [(i32, usize); 3]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        entries.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

include!("common/spy_executor.rs");

/// `(svd, eigh, gemm, solve)` entries the spy saw, so a test can prove both
/// that the injected backend is the one the runtime drives and that a
/// storage-local route never reaches it.
fn read(counts: &SpyCounts) -> (usize, usize, usize, usize) {
    (
        counts.of(Kernel::SVD) + counts.get(Kernel::SvdVals),
        counts.of(Kernel::EIGH) + counts.get(Kernel::EighVals),
        counts.of(Kernel::GEMM),
        counts.get(Kernel::Solve),
    )
}

#[test]
fn injected_dense_executor_is_used_and_preserves_results() {
    let counts = Arc::new(SpyCounts::default());
    let spy = SpyExecutor::counting(&counts);
    let rt = Runtime::builder()
        .with_dense_executor(Box::new(spy))
        .build()
        .unwrap();

    let v = u1_space([(-1, 2), (0, 2), (1, 1)]);
    let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 99).unwrap();
    let Svd { s, .. } = t.svd_compact(&[0, 1], &[2, 3]).unwrap();

    assert!(
        counts.of(Kernel::SVD) > 0,
        "the injected executor's svd was never called — the runtime is not \
         driving the injected backend"
    );

    // No behavior change: the same seeded tensor on an unset built-in runtime
    // yields identical singular values.
    let rt_default = Runtime::builder().build().unwrap();
    let t_default =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt_default, [&v, &v], [&v, &v], 99)
            .unwrap();
    let Svd { s: s_default, .. } = t_default.svd_compact(&[0, 1], &[2, 3]).unwrap();
    assert_eq!(
        s.materialize().unwrap().dense_data().unwrap().len(),
        s_default.materialize().unwrap().dense_data().unwrap().len()
    );
    for (a, b) in s
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(s_default.materialize().unwrap().dense_data().unwrap())
    {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "singular value differs from the default backend: {a} vs {b}"
        );
    }
}

#[test]
fn compact_diagonal_exp_drives_no_dense_kernel() {
    // Issue #578: `exp` on compact diagonal storage is elementwise on the
    // stored spectrum, so the injected backend sees nothing at all — not the
    // Hermitian eigendecomposition the dense route runs, not the GEMM that
    // reassembles `V exp(D) V^H`, not a solve. The spy is the direct
    // observation of that; the allocation gates only see its cost.
    let counts = Arc::new(SpyCounts::default());
    let rt = Runtime::builder()
        .with_dense_executor(Box::new(SpyExecutor::counting(&counts)))
        .build()
        .unwrap();

    let v = u1_space([(-1, 3), (0, 4), (1, 3)]);
    let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v], [&v], 578).unwrap();
    let s = t.svd_compact(&[0], &[1]).unwrap().s;
    let (svd, ..) = read(&counts);
    assert!(svd > 0, "the fixture never reached the injected backend");

    let before = read(&counts);
    let before_total = counts.total();
    let image = s.exp(&[0], &[1]).unwrap();
    let (_, eigh, gemm, solve) = read(&counts);
    assert_eq!(
        (eigh, gemm, solve),
        (before.1, before.2, before.3),
        "compact exp drove a dense kernel"
    );
    assert_eq!(
        counts.total(),
        before_total,
        "compact exp drove a dense kernel"
    );

    // And it computed the right thing: every singular value exponentiated.
    // Compared per sector, since `svd_vals` sorts within a sector, and the
    // exponential is monotone so the order carries over.
    let source_values = s.svd_vals(&[0], &[1]).unwrap();
    let image_values = image.svd_vals(&[0], &[1]).unwrap();
    assert_eq!(source_values.len(), image_values.len());
    for (source, image) in source_values.iter().zip(&image_values) {
        assert_eq!(source.sector, image.sector);
        assert_eq!(source.values.len(), image.values.len());
        for (source, image) in source.values.iter().zip(&image.values) {
            let expected = source.exp();
            assert!(
                (expected - image).abs() <= 1e-12 * (1.0 + expected.abs()),
                "exp({source}) is {expected}, got {image}"
            );
        }
    }
}

#[test]
fn injected_executor_without_eig_reports_unsupported() {
    // The spy reports `eig` as Unsupported, as an executor that leaves it at
    // the trait default does: the missing-capability case. The error
    // must name the injected executor's gap, not a Tenferro backend failure.
    let rt = Runtime::builder()
        .with_dense_executor(Box::new(SpyExecutor::default().without(
            &[Kernel::Eig],
            "executor does not implement the general eigendecomposition",
        )))
        .build()
        .unwrap();

    let v = u1_space([(-1, 1), (0, 2), (1, 1)]);
    let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(&rt, [&v], [&v], 1266).unwrap();

    let error = t.eig_full(&[0], &[1]).unwrap_err();
    let Error::Operation(operation) = error else {
        panic!("a missing executor capability must surface as an operation error, got {error:?}")
    };
    assert!(
        matches!(
            *operation,
            OperationError::Dense(DenseError::Unsupported { op: "eig", .. })
        ),
        "a missing executor capability must be Unsupported, got {operation:?}"
    );
}

#[test]
fn zero_dense_threads_is_a_typed_build_error() {
    let error = Runtime::builder().dense_threads(0).build().unwrap_err();
    assert_eq!(
        error,
        Error::RuntimeConfig(RuntimeConfigError::ZeroDenseThreads)
    );
}

#[test]
fn zero_recoupling_threads_is_a_typed_build_error() {
    let error = Runtime::builder()
        .recoupling_threads(0)
        .build()
        .unwrap_err();
    assert_eq!(
        error,
        Error::RuntimeConfig(RuntimeConfigError::ZeroRecouplingThreads)
    );
}
