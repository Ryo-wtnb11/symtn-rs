use super::*;
use symtn_dense::{DenseDType, DenseView};

/// Owned EIGH entries; the destination form is refused, so a checked
/// Generic EIGH must use owned output.
const EIGH_OWNED: &[Kernel] = &[Kernel::Eigh, Kernel::EighVals];

/// Fails every dense EIG call.
fn eig_fault_spy() -> SpyExecutor {
    SpyExecutor::default().failing(
        &[Kernel::Eig, Kernel::EigVals],
        None,
        "injected checked Generic EIG failure",
    )
}
/// Counts owned EIGH calls; the `fail_at`-th fails.
fn eigh_fault_spy(counts: &Arc<SpyCounts>, fail_at: Option<usize>) -> SpyExecutor {
    let spy = SpyExecutor::counting(counts).failing(
        &[Kernel::EighInto],
        None,
        "checked Generic EIGH must use owned output",
    );
    match fail_at {
        Some(nth) => spy.failing(
            EIGH_OWNED,
            Some(nth),
            "injected checked Generic EIGH failure",
        ),
        None => spy,
    }
}
fn assert_checked_generic_eig_reconstruction(
    source: &TensorMap<CheckedOnlyToy, Complex64>,
    d: &TensorMap<CheckedOnlyToy, Complex64>,
    v: &TensorMap<CheckedOnlyToy, Complex64>,
) {
    let engine = &mut crate::typed::CpuEngine::serial();
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(symtn::typed::__network::runtime_identity(d.runtime()).matches(source.runtime()));
    assert!(symtn::typed::__network::runtime_identity(v.runtime()).matches(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    let av = source.compose(engine, v).unwrap();
    let vd = v.compose(engine, d).unwrap();
    let scale = source
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    assert!(av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() <= 1.0e-11 * scale));
    let inverse = v.inv(engine, &[0], &[1]).unwrap();
    let rebuilt = v
        .compose(engine, d)
        .unwrap()
        .compose(engine, &inverse)
        .unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, source, |actual, expected| {
        (actual - expected).norm()
    });
}
/// Independent dense oracle for one coupled block: every returned pair
/// satisfies `A v = lambda v` to backward error relative to `|A| |v|` and no
/// eigenvector vanishes. No inverse of `V` is formed: a defective block has
/// no eigenbasis.
fn assert_defective_eigenpairs(
    n: usize,
    a: impl Fn(usize, usize) -> Complex64,
    v: impl Fn(usize, usize) -> Complex64,
    lambda: impl Fn(usize) -> Complex64,
    exact: Complex64,
) {
    let scale = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| a(i, j).norm())
        .fold(1.0_f64, f64::max);
    // A size-n Jordan block's eigenvalue moves by about (eps |A|)^(1/n) under a
    // backward-stable solver, so the spectrum bound follows that theory; the
    // eigenpair residual below is the main oracle.
    let spectrum_tol = 10.0 * (f64::EPSILON * scale).powf(1.0 / n as f64);
    for k in 0..n {
        assert!((lambda(k) - exact).norm() <= spectrum_tol);
        let norm = (0..n).map(|i| v(i, k).norm_sqr()).sum::<f64>().sqrt();
        assert!(norm > 0.5);
        for i in 0..n {
            let av = (0..n).map(|j| a(i, j) * v(j, k)).sum::<Complex64>();
            assert!((av - lambda(k) * v(i, k)).norm() <= 1.0e-12 * n as f64 * scale * norm);
        }
    }
}
#[test]
fn checked_generic_eig_dense_failure_preserves_the_source() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let _dense = probe::inject_dense(Box::new(eig_fault_spy()));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x40));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(matches!(
        source.eig_full(engine, &[0], &[1]),
        Err(GenericTensorError::Plan(
            CheckedGenericPlanError::Operation(OperationError::Dense(DenseError::Backend { .. }))
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}
#[test]
fn checked_generic_eig_full_is_complex_and_reconstructs_nonnormal_inputs() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x41));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let Eig {
        d: real_d,
        v: real_v,
    } = real.eig_full(engine, &[0], &[1]).unwrap();
    assert!(real_d
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im.abs() > 1.0));
    for column in 0..2 {
        let pivot = (0..2)
            .max_by(|&left, &right| {
                real_v.dense_data().unwrap()[left + column * 2]
                    .norm()
                    .total_cmp(&real_v.dense_data().unwrap()[right + column * 2].norm())
            })
            .unwrap();
        let pivot = real_v.dense_data().unwrap()[pivot + column * 2];
        assert!(pivot.im.abs() < 1.0e-12);
        assert!(pivot.re >= 0.0);
    }
    assert_checked_generic_eig_reconstruction(
        &real.convert::<Complex64>(engine).unwrap(),
        &real_d,
        &real_v,
    );

    let complex = real
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.25))
        .unwrap();
    let Eig {
        d: complex_d,
        v: complex_v,
    } = complex.eig_full(engine, &[0], &[1]).unwrap();
    assert!(complex_d
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im.abs() > 1.0));
    assert_checked_generic_eig_reconstruction(&complex, &complex_d, &complex_v);

    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -3.0)],
        }],
    )
    .unwrap();
    let Eig { d, v } = compact.eig_full(engine, &[0], &[1]).unwrap();
    assert_eq!(
        d.diagview(engine).unwrap()[0].values,
        [Complex64::new(0.0, -3.0), Complex64::new(1.0, 1.0)]
    );
    let dense = compact.materialize(engine).unwrap();
    assert_checked_generic_eig_reconstruction(&dense, &d, &v);
    let _dense = probe::inject_dense(Box::new(eig_fault_spy()));
    let fault_runtime = Runtime::builder().build().unwrap();
    let fault_provider = Arc::new(CheckedOnlyToy::new(0x42));
    let fault_leg = GradedSpace::try_new(Arc::clone(&fault_provider), [(Label::X, 2)]).unwrap();
    let compact_on = |engine: &mut dyn Engine, leg: &GradedSpace<_>| -> TensorMap<_, Complex64> {
        TensorMap::diagonal(
            engine,
            &fault_runtime,
            leg,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -3.0)],
            }],
        )
        .unwrap()
    };
    let nondual_compact = compact_on(engine, &fault_leg);
    let fault_compact = compact_on(engine, &fault_leg.try_dual().unwrap());
    fault_provider.invalid_style.store(true, Ordering::Relaxed);
    // A nondual bond is its own eigenbasis bond (TensorKit `fuse(V) = V`): `v`
    // and `d` live on the input space, so the provider is never queried and
    // the dense eigensolver never runs.
    reset_provider_queries(&fault_provider);
    let Eig { d, v } = nondual_compact.eig_full(engine, &[0], &[1]).unwrap();
    assert_eq!(
        fault_provider.queries_since_reset.load(Ordering::Relaxed),
        0
    );
    assert_eq!(v.codomain(), nondual_compact.codomain());
    assert_eq!(d.codomain(), nondual_compact.codomain());
    // A dual bond's eigenbasis bond is the fresh nondual `fuse(V)`: the
    // eligible compact path reaches checked V publication; falling back to
    // dense EIG would return the injected dense error instead.
    let compact_error = fault_compact.eig_full(engine, &[0], &[1]).unwrap_err();
    assert!(
        matches!(
            compact_error,
            GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::Core(
                    symtn::typed::CoreError::UnsupportedFusionStyle { .. }
                )
            ))
        ),
        "{compact_error:?}"
    );
}
#[test]
fn checked_generic_eigh_dense_failure_preserves_the_source() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eigh_fault_spy(&calls, Some(1))));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x43));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(matches!(
        source.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT),
        Err(GenericTensorError::Plan(
            CheckedGenericPlanError::Operation(OperationError::Dense(DenseError::Backend { .. }))
        ))
    ));
    assert_eq!(calls.of(EIGH_OWNED), 1);
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}
#[test]
fn checked_generic_eigh_preflights_all_sectors_and_runs_once_per_sector() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eigh_fault_spy(&calls, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x44));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, index| {
            if index[0] == index[1] {
                if trees.coupled() == &Label::Vacuum {
                    1.0
                } else {
                    2.0
                }
            } else {
                0.0
            }
        })
        .unwrap();
    hermitian
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(calls.of(EIGH_OWNED), 2);

    calls.reset();
    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, index| {
            if trees.coupled() == &Label::X && index == [0, 1] {
                1.0
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(matches!(nonhermitian
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::InvalidArgument { message }))) if message.contains("Hermitian coupled-sector blocks")));
    assert_eq!(calls.of(EIGH_OWNED), 0);
    assert_eq!(calls.total(), 0);
}
#[test]
fn eig_full_accepts_finite_defective_sectors_in_both_modes_without_a_rank_svd() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn::sector::{SU2FusionRule, SU2Irrep};

    let counts = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(SpyExecutor::counting(&counts)));
    let runtime = Runtime::builder().build().unwrap();
    // `lambda I + N` with `N` the upper shift: a single Jordan chain, so each
    // block of dimension two or more is defective.
    let jordan = |lambda: Complex64, row: usize, col: usize| {
        if row == col {
            lambda
        } else if col == row + 1 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    };
    let checked_leg = GradedSpace::try_new(
        Arc::new(CheckedOnlyToy::new(0x45)),
        [(Label::Vacuum, 2), (Label::X, 3)],
    )
    .unwrap();
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 3),
        ],
    )
    .unwrap();
    macro_rules! check {
        ($leg:expr, $first:expr, $dtype:ty, $from:expr, $lambda:expr) => {{
            let from: fn(Complex64) -> $dtype = $from;
            let lambda = |sector: &_| $lambda[usize::from(sector != &$first)];
            let source: TensorMap<_, $dtype> =
                TensorMap::from_subblock_fn(engine, &runtime, [&$leg], [&$leg], |trees, index| {
                    from(jordan(lambda(trees.coupled()), index[0], index[1]))
                })
                .unwrap();
            counts.reset();
            let Eig { d, v } = source.eig_full(engine, &[0], &[1]).unwrap();
            // One dense EIG per coupled sector and nothing else: no
            // eigenvector SVD runs in either mode.
            assert_eq!(counts.get(Kernel::Eig), 2);
            assert_eq!(counts.others(&[Kernel::Eig]), 0);
            let mut sectors = 0;
            for (sector, a) in source.blocks().unwrap() {
                let (v, d) = (v.block(&sector).unwrap(), d.block(&sector).unwrap());
                assert_defective_eigenpairs(
                    a.rows(),
                    |i, j| Complex64::from(a.get(i, j).unwrap()),
                    |i, j| v.get(i, j).unwrap(),
                    |k| d.get(k, k).unwrap(),
                    lambda(&sector),
                );
                sectors += 1;
            }
            assert_eq!(sectors, 2);
        }};
    }
    let real = [Complex64::new(2.0, 0.0), Complex64::new(-0.5, 0.0)];
    let complex = [Complex64::new(2.0, 1.0), Complex64::new(-0.5, -0.25)];
    check!(checked_leg, Label::Vacuum, f64, |z| z.re, real);
    check!(checked_leg, Label::Vacuum, Complex64, |z| z, complex);
    let spin_half = SU2Irrep::from_twice_spin(1);
    check!(mf_leg, spin_half, f64, |z| z.re, real);
    check!(mf_leg, spin_half, Complex64, |z| z, complex);
}

/// The default executor, except that `eig` returns the provider's finite
/// eigenvalues with an all-NaN eigenvector matrix of the same order and
/// dtype: a provider that reports success with nonfinite eigenvectors. The
/// matrix is the `Q` of an all-NaN QR, so it is a real provider tensor.
struct NonfiniteEigenvectors(DefaultDenseExecutor);

impl DenseExecutor for NonfiniteEigenvectors {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.svd(engine, input)
    }
    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.qr(engine, input)
    }
    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eigh(engine, input)
    }
    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.0.dot_general_into(engine, output, lhs, rhs, config)
    }
    fn eig(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let mut outputs = self.0.eig(engine, input)?;
        let n = outputs[0].shape()[0];
        let (shape, strides) = ([n, n], [1, n]);
        let mut q = match outputs[1].dtype() {
            DenseDType::C64 => {
                let nan = vec![Complex64::new(f64::NAN, 0.0); n * n];
                let view = DenseView::new(&nan, &shape, &strides, 0)?;
                self.0.qr(engine, DenseRead::C64(view))?
            }
            dtype => {
                assert_eq!(dtype, DenseDType::C32);
                let nan = vec![Complex32::new(f32::NAN, 0.0); n * n];
                let view = DenseView::new(&nan, &shape, &strides, 0)?;
                self.0.qr(engine, DenseRead::C32(view))?
            }
        };
        assert!(q[0].shape() == [n, n], "the forged eigenvectors are n x n");
        outputs[1] = q.swap_remove(0);
        Ok(outputs)
    }
}

/// symtn's own eigenvector check: a provider that reports success with a
/// nonfinite eigenvector of a finite input is a `NumericalFailure`, in both
/// fusion modes and every dtype. The default provider cannot reach it (faer
/// 0.27 rescales its back substitution like LAPACK `xTREVC` and reports
/// `NoConvergence` for a nonfinite column), so the provider is forged.
#[test]
fn eig_full_rejects_provider_nonfinite_eigenvectors_in_both_modes() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn::sector::{SU2FusionRule, SU2Irrep};

    let runtime = Runtime::builder().build().unwrap();
    let checked_leg =
        GradedSpace::try_new(Arc::new(CheckedOnlyToy::new(0)), [(Label::Vacuum, 3)]).unwrap();
    let mf_leg =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(1), 3)]).unwrap();
    // `2 I + N`: finite, with finite eigenvalues.
    let jordan = |row: usize, col: usize| match col.checked_sub(row) {
        Some(0) => 2.0,
        Some(1) => 1.0,
        _ => 0.0,
    };
    macro_rules! check {
        ($leg:expr, $dtype:ty, $from:expr) => {{
            let from: fn(f64) -> $dtype = $from;
            let source: TensorMap<_, $dtype> =
                TensorMap::from_subblock_fn(engine, &runtime, [&$leg], [&$leg], |_, index| {
                    from(jordan(index[0], index[1]))
                })
                .unwrap();
            let _dense =
                probe::inject_dense(Box::new(NonfiniteEigenvectors(DefaultDenseExecutor::new())));
            let error = source.eig_full(engine, &[0], &[1]).unwrap_err();
            assert!(
                matches!(
                    cause::<symtn_tensors::OperationError>(&error),
                    Some(symtn_tensors::OperationError::Dense(
                        DenseError::NumericalFailure {
                            op: "eig eigenvector check",
                            ..
                        }
                    ))
                ),
                "{error:?}"
            );
        }};
    }
    check!(checked_leg, f32, |x| x as f32);
    check!(checked_leg, f64, |x| x);
    check!(checked_leg, Complex32, |x| Complex32::new(x as f32, 0.0));
    check!(checked_leg, Complex64, |x| Complex64::new(x, 0.0));
    check!(mf_leg, f32, |x| x as f32);
    check!(mf_leg, f64, |x| x);
    check!(mf_leg, Complex32, |x| Complex32::new(x as f32, 0.0));
    check!(mf_leg, Complex64, |x| Complex64::new(x, 0.0));
}
