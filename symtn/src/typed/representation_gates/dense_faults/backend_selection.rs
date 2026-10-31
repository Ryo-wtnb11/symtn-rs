//! Storage-local routes on a compact diagonal never reach the dense
//! executor: the spy sees no SVD, EIGH, GEMM or solve.

use super::*;

fn u1_space(entries: [(i32, usize); 3]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        entries.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

/// `(svd, eigh, gemm, solve)` entries the spy saw.
fn read(counts: &SpyCounts) -> (usize, usize, usize, usize) {
    (
        counts.of(Kernel::SVD) + counts.get(Kernel::SvdVals),
        counts.of(Kernel::EIGH) + counts.get(Kernel::EighVals),
        counts.of(Kernel::GEMM),
        counts.get(Kernel::Solve),
    )
}

#[test]
fn compact_diagonal_svd_submits_no_dense_svd() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let counts = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(SpyExecutor::counting(&counts)));
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_space([(-1, 2), (0, 3), (1, 1)]);
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
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
    let Svd { u, s, vh } = input.svd_compact(engine, &[0], &[1]).unwrap();
    assert_eq!(read(&counts), (0, 0, 0, 0));
    assert_eq!(counts.total(), 0);
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    for (actual, expected) in rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize(engine).unwrap().dense_data().unwrap())
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn compact_diagonal_exp_drives_no_dense_kernel() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // `exp` on compact diagonal storage is elementwise on the stored spectrum
    // (#578), so the injected backend sees nothing at all: not the Hermitian
    // eigendecomposition the dense route runs, not the GEMM that reassembles
    // `V exp(D) V^H`, not a solve.
    let counts = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(SpyExecutor::counting(&counts)));
    let runtime = Runtime::builder().build().unwrap();

    let v = u1_space([(-1, 3), (0, 4), (1, 3)]);
    let t =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&v], [&v], 578).unwrap();
    let s = t.svd_compact(engine, &[0], &[1]).unwrap().s;
    let (svd, ..) = read(&counts);
    assert!(svd > 0, "the fixture never reached the injected backend");

    let before = read(&counts);
    let before_total = counts.total();
    let image = s.exp(engine, &[0], &[1]).unwrap();
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

    // Every singular value exponentiated. Compared per sector, since
    // `svd_vals` sorts within a sector, and the exponential is monotone so
    // the order carries over.
    let source_values = s.svd_vals(engine, &[0], &[1]).unwrap();
    let image_values = image.svd_vals(engine, &[0], &[1]).unwrap();
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
