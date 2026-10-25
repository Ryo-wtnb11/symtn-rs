use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_qr_preserves_provider_and_reconstructs() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Qr { q, r } = source.qr_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(&r).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>();
    let Qr {
        q: complex_q,
        r: complex_r,
    } = complex.qr_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(&complex_r).unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[test]
fn checked_compact_diagonal_qr_lq_all_modes_use_hand_phase_and_magnitude() {
    let _serial = crate::counting_alloc::serial();
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(SpyExecutor::counting(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(-2.0, 0.0), Complex64::new(0.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![
                    Complex64::new(0.0, 3.0),
                    Complex64::new(4.0, 0.0),
                    Complex64::new(-5.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let expected_phase = [
        vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    ];
    let expected_magnitude = [
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ],
    ];
    let assert_factors =
        |phase: &TensorMap<CheckedOnlyToy, Complex64>,
         magnitude: &TensorMap<CheckedOnlyToy, Complex64>| {
            for factor in [phase, magnitude] {
                assert!(std::ptr::eq(factor.provider(), provider.as_ref()));
                assert_eq!(factor.codomain(), input.codomain());
                assert_eq!(factor.domain(), input.domain());
                assert!(
                    symtn::typed::__network::network_reuse_class(factor, false)
                        == NetworkReuseClass::Compact
                );
                assert!(factor.dense_data().is_err());
            }
            for (actual, expected) in phase.diagview().unwrap().iter().zip(&expected_phase) {
                assert_eq!(&actual.values, expected);
            }
            for (actual, expected) in magnitude
                .diagview()
                .unwrap()
                .iter()
                .zip(&expected_magnitude)
            {
                assert_eq!(&actual.values, expected);
            }
            let rebuilt = phase.compose(magnitude).unwrap();
            assert!(rebuilt.dense_data().is_err(), "D * D stays compact");
            numerics::assert_slices_close(
                "checked compact diagonal QR/LQ reconstruction",
                rebuilt.materialize().unwrap().dense_data().unwrap(),
                input.materialize().unwrap().dense_data().unwrap(),
                3,
            );
        };
    for Qr { q, r } in [
        input.qr_compact(&[0], &[1]).unwrap(),
        input.qr_full(&[0], &[1]).unwrap(),
    ] {
        assert_factors(&q, &r);
    }
    for Lq { l, q } in [
        input.lq_compact(&[0], &[1]).unwrap(),
        input.lq_full(&[0], &[1]).unwrap(),
    ] {
        assert_factors(&q, &l);
    }
    assert_eq!(calls.of(Kernel::QR), 0);
    assert_eq!(calls.total(), 0);

    let narrow = input.convert::<Complex32>();
    for factor in [
        narrow.qr_compact(&[0], &[1]).unwrap().r,
        narrow.qr_full(&[0], &[1]).unwrap().r,
        narrow.lq_compact(&[0], &[1]).unwrap().l,
        narrow.lq_full(&[0], &[1]).unwrap().l,
    ] {
        assert!(
            symtn::typed::__network::network_reuse_class(&factor, false)
                == NetworkReuseClass::Compact
        );
    }
    let Qr { q, r } = narrow.qr_compact(&[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "checked compact Complex32 QR reconstruction",
        q.compose(&r)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        narrow.materialize().unwrap().dense_data().unwrap(),
        3,
    );
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-2.0, 0.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![3.0, 4.0, -5.0],
            },
        ],
    )
    .unwrap();
    for factor in [
        real.qr_compact(&[0], &[1]).unwrap().r,
        real.qr_full(&[0], &[1]).unwrap().r,
        real.lq_compact(&[0], &[1]).unwrap().l,
        real.lq_full(&[0], &[1]).unwrap().l,
    ] {
        assert!(
            symtn::typed::__network::network_reuse_class(&factor, false)
                == NetworkReuseClass::Compact
        );
    }
    let Lq { l, q } = real.lq_compact(&[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "checked compact f64 LQ reconstruction",
        l.compose(&q)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        real.materialize().unwrap().dense_data().unwrap(),
        3,
    );
    let narrow_real = real.convert::<f32>();
    for factor in [
        narrow_real.qr_compact(&[0], &[1]).unwrap().r,
        narrow_real.qr_full(&[0], &[1]).unwrap().r,
        narrow_real.lq_compact(&[0], &[1]).unwrap().l,
        narrow_real.lq_full(&[0], &[1]).unwrap().l,
    ] {
        assert!(
            symtn::typed::__network::network_reuse_class(&factor, false)
                == NetworkReuseClass::Compact
        );
    }
    let Qr { q, r } = narrow_real.qr_compact(&[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "checked compact f32 QR reconstruction",
        q.compose(&r)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        narrow_real.materialize().unwrap().dense_data().unwrap(),
        3,
    );

    let dual_bond = bond.try_dual().unwrap();
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-2.0, 0.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![3.0, 4.0, -5.0],
            },
        ],
    )
    .unwrap();
    // #1729: TensorKit's diagonal dispatch keeps `W = V`, dual flag included.
    for factor in [
        dual.qr_compact(&[0], &[1]).unwrap().r,
        dual.qr_full(&[0], &[1]).unwrap().r,
        dual.lq_compact(&[0], &[1]).unwrap().l,
        dual.lq_full(&[0], &[1]).unwrap().l,
    ] {
        assert!(
            symtn::typed::__network::network_reuse_class(&factor, false)
                == NetworkReuseClass::Compact
        );
        assert!(factor.codomain()[0].is_dual());
        assert!(factor.domain()[0].is_dual());
        assert_eq!(factor.codomain(), dual.codomain());
        assert_eq!(factor.domain(), dual.domain());
    }
    assert_eq!(calls.of(Kernel::QR), 0);
    assert_eq!(calls.total(), 0);

    // Materialized dual input keeps the dense route and its fresh nondual W.
    let dense_dual = dual.materialize().unwrap();
    for factor in [
        dense_dual.qr_compact(&[0], &[1]).unwrap().r,
        dense_dual.qr_full(&[0], &[1]).unwrap().r,
    ] {
        assert!(
            symtn::typed::__network::network_reuse_class(&factor, false)
                == NetworkReuseClass::OwnedDense
        );
        assert!(!factor.codomain()[0].is_dual());
    }
    for factor in [
        dense_dual.lq_compact(&[0], &[1]).unwrap().l,
        dense_dual.lq_full(&[0], &[1]).unwrap().l,
    ] {
        assert!(
            symtn::typed::__network::network_reuse_class(&factor, false)
                == NetworkReuseClass::OwnedDense
        );
        assert!(!factor.domain()[0].is_dual());
    }
    assert_eq!(calls.of(Kernel::QR), 8);
}

/// A nonfinite compact diagonal is refused by the shared finite-input stage
/// (as dense input is, #1986), and a finite one is
/// factorized on its input bond without consulting the provider, so a failing
/// provider does not fail `qr_full` (#1751).
#[test]
fn checked_compact_diagonal_qr_rejects_nonfinite_and_skips_the_provider_when_finite() {
    let _serial = crate::counting_alloc::serial();
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(SpyExecutor::counting(&calls)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![f64::NAN, -1.0],
        }],
    )
    .unwrap();
    let error = nonfinite.qr_compact(&[0], &[1]).map(drop).unwrap_err();
    assert_eq!(
        crate::report::cause::<symtn::typed::OperationError>(&error),
        Some(&symtn::typed::OperationError::InvalidArgument {
            message: "qr input components must be finite"
        }),
        "{error:?}"
    );
    assert_eq!(calls.of(Kernel::QR), 0);

    let failing_provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
    let failing_bond =
        GradedSpace::try_new(Arc::clone(&failing_provider), [(Label::X, 2)]).unwrap();
    let finite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &failing_bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![2.0, -1.0],
        }],
    )
    .unwrap();
    failing_provider.fail_algebra.store(true, Ordering::Relaxed);
    reset_provider_queries(&failing_provider);
    let Qr { q, r } = finite.qr_full(&[0], &[1]).unwrap();
    assert_eq!(
        failing_provider.queries_since_reset.load(Ordering::Relaxed),
        0
    );
    for factor in [&q, &r] {
        assert!(
            symtn::typed::__network::network_reuse_class(factor, false)
                == NetworkReuseClass::Compact
        );
        assert_eq!(factor.codomain(), finite.codomain());
    }
    assert_eq!(calls.of(Kernel::QR), 0);
}

/// A bosonic Abelian rule exposed only through the checked-Generic contract,
/// so dual QR/LQ is checked on self-dual Z2 and non-self-dual U(1) labels.
struct CheckedAbelian<R>(R);

impl<R: symtn::sector::FusionRule> symtn_core::testing::Provider for CheckedAbelian<R> {}

impl<R: symtn::sector::FusionRule> CheckedGenericFusion for CheckedAbelian<R> {
    type Error = ToyError;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x1729, Arc::<[u8]>::from([]))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.0.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.0.vacuum()
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, ToyError> {
        Ok(self.0.dual(sector))
    }

    fn try_fusion_channels(&self, left: SectorId, right: SectorId) -> Result<SectorVec, ToyError> {
        Ok(self.0.fusion_channels(left, right))
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, ToyError> {
        Ok(self.0.fusion_channels(left, right))
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, ToyError> {
        Ok(self.0.nsymbol(left, right, coupled))
    }

    fn sector_order_key(&self, sector: SectorId) -> symtn::sector::SectorOrderKey {
        self.0.sector_order_key(sector)
    }
}

impl<R: symtn::sector::FusionRule> CheckedGenericRigidSymbols for CheckedAbelian<R> {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, ToyError> {
        Ok(1.0)
    }

    fn try_inv_sqrt_dim_scalar(&self, _: SectorId) -> Result<f64, ToyError> {
        Ok(1.0)
    }

    fn try_frobenius_schur_phase_scalar(&self, _: SectorId) -> Result<f64, ToyError> {
        Ok(1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, ToyError> {
        let shape = (
            self.0.nsymbol(a, b, e),
            self.0.nsymbol(e, c, d),
            self.0.nsymbol(b, c, f),
            self.0.nsymbol(a, f, d),
        );
        Ok(GenericFArray::new(
            vec![1.0; shape.0 * shape.1 * shape.2 * shape.3],
            shape,
        ))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, ToyError> {
        let rows = self.0.nsymbol(a, b, c);
        Ok(GenericRMatrix::new(vec![1.0; rows * rows], rows, rows))
    }
}

impl<R: symtn::sector::SectorCodec> TypedSectorAdmission for CheckedAbelian<R> {
    type Sector = R::Sector;
    type Error = ToyError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &R::Sector) -> Result<SectorId, ToyError> {
        self.0
            .encode_sector(sector)
            .map_err(|_| ToyError::InvalidSector)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<R::Sector, ToyError> {
        self.0.decode_sector(sector).map_err(|_| ToyError::Decode)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, ToyError> {
        Ok(self.0.dual(sector))
    }
}

fn assert_wide_close<D: numerics::Numeric>(what: &str, got: D, want: Complex64) {
    let tolerance = numerics::tolerance::<D>(2, want.norm());
    assert!(
        (got.wide() - want).norm() <= tolerance,
        "{what}: {got:?} != {want}"
    );
}

/// All four QR/LQ methods on an admitted dual diagonal `A : V <- V`. Oracle:
/// TensorKit `diagonal.jl:16-42` keeps `space(d)` for both factors, and per
/// entry `phase(d) = d / |d|` (1 at zero) and `|d|`, computed here in
/// Complex64 independently of symtn's descriptor.
macro_rules! assert_dual_diagonal_qr_lq {
    ($runtime:expr, $input:expr, $calls:expr) => {{
        use numerics::Numeric as _;
        let input = &$input;
        let bond = &input.codomain()[0];
        assert!(bond.is_dual());
        let source = input.diagview().unwrap();
        let dense = input.materialize().unwrap();
        let identity = TensorMap::isomorphism(&$runtime, [bond], [bond]).unwrap();
        let before = $calls.total();
        let results = [
            input.qr_compact(&[0], &[1]).map(|Qr { q, r }| (q, r, true)),
            input.qr_full(&[0], &[1]).map(|Qr { q, r }| (q, r, true)),
            input
                .lq_compact(&[0], &[1])
                .map(|Lq { l, q }| (q, l, false)),
            input.lq_full(&[0], &[1]).map(|Lq { l, q }| (q, l, false)),
        ];
        assert_eq!($calls.total(), before);
        for result in results {
            let (phase, magnitude, qr) = result.unwrap();
            for factor in [&phase, &magnitude] {
                assert!(std::ptr::eq(factor.provider(), input.provider()));
                assert_eq!(factor.codomain(), input.codomain());
                assert_eq!(factor.domain(), input.domain());
                assert!(
                    symtn::typed::__network::network_reuse_class(&factor, false)
                        == NetworkReuseClass::Compact
                );
                assert!(factor.dense_data().is_err());
            }
            let phases = phase.diagview().unwrap();
            let magnitudes = magnitude.diagview().unwrap();
            assert_eq!(phases.len(), source.len());
            assert_eq!(magnitudes.len(), source.len());
            for entry in &source {
                let p = phases.iter().find(|s| s.sector == entry.sector).unwrap();
                let m = magnitudes
                    .iter()
                    .find(|s| s.sector == entry.sector)
                    .unwrap();
                assert_eq!(p.values.len(), entry.values.len());
                assert_eq!(m.values.len(), entry.values.len());
                for ((&d, &p), &m) in entry.values.iter().zip(&p.values).zip(&m.values) {
                    let d = d.wide();
                    let (want_phase, want_magnitude) = if d.norm() == 0.0 {
                        (Complex64::new(1.0, 0.0), 0.0)
                    } else {
                        (d / d.norm(), d.norm())
                    };
                    assert_wide_close("phase", p, want_phase);
                    assert_wide_close("magnitude", m, Complex64::new(want_magnitude, 0.0));
                    // Positive gauge: the magnitude is exactly real and nonnegative.
                    assert!(m.wide().im == 0.0 && m.wide().re >= 0.0);
                }
            }
            let rebuilt = if qr {
                phase.compose(&magnitude)
            } else {
                magnitude.compose(&phase)
            }
            .unwrap();
            numerics::assert_slices_close(
                "dual diagonal QR/LQ reconstruction",
                rebuilt.materialize().unwrap().dense_data().unwrap(),
                dense.dense_data().unwrap(),
                2,
            );
            let gram = phase.adjoint().unwrap().compose(&phase).unwrap();
            numerics::assert_slices_close(
                "dual diagonal QR/LQ isometry",
                gram.materialize().unwrap().dense_data().unwrap(),
                identity.dense_data().unwrap(),
                2,
            );
        }
    }};
}

fn real_spectra<S: Clone>(sectors: [S; 2], values: [&[f64]; 2]) -> Vec<SectorSpectrum<S, f64>> {
    sectors
        .into_iter()
        .zip(values)
        .map(|(sector, values)| SectorSpectrum {
            sector,
            values: values.to_vec(),
        })
        .collect()
}

#[test]
fn checked_dual_diagonal_qr_lq_keeps_dual_bond_for_self_dual_and_non_self_dual_rules() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::{U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep};

    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(SpyExecutor::counting(&calls)))
        .build()
        .unwrap();
    // Unequal degeneracies, zero and nonreal entries in every case.
    let complex_values: [&[(f64, f64)]; 2] = [
        &[(-2.0, 0.0), (0.0, 0.0)],
        &[(3.0, 4.0), (0.0, -7.0), (-1.0, 0.0)],
    ];
    let real_values: [&[f64]; 2] = [&[-2.0, 0.0], &[5.0, -7.0, 1.0]];

    let z2 = GradedSpace::try_new(
        Arc::new(CheckedAbelian(Z2FusionRule)),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let z2_dual = z2.try_dual().unwrap();
    // Self-dual labels: only the orientation flag distinguishes V' from V.
    assert!(z2_dual.is_dual());
    assert_eq!(z2_dual.degeneracy(&Z2Irrep::EVEN).unwrap(), 2);
    assert_eq!(z2_dual.degeneracy(&Z2Irrep::ODD).unwrap(), 3);
    let z2_sectors = [Z2Irrep::EVEN, Z2Irrep::ODD];

    // TensorKit `dual(Rep[U1](1 => 2, 2 => 3))` carries physical sectors
    // -1 and -2 with the same degeneracies.
    let u1 = GradedSpace::try_new(
        Arc::new(CheckedAbelian(U1FusionRule)),
        [(U1Irrep::new(1), 2), (U1Irrep::new(2), 3)],
    )
    .unwrap();
    let u1_dual = u1.try_dual().unwrap();
    assert!(u1_dual.is_dual());
    let mut u1_sectors_seen = u1_dual.sectors().unwrap();
    u1_sectors_seen.sort();
    assert_eq!(u1_sectors_seen, [U1Irrep::new(-2), U1Irrep::new(-1)]);
    assert_eq!(u1_dual.degeneracy(&U1Irrep::new(-1)).unwrap(), 2);
    assert_eq!(u1_dual.degeneracy(&U1Irrep::new(-2)).unwrap(), 3);
    assert_eq!(u1_dual.degeneracy(&U1Irrep::new(1)).unwrap(), 0);
    let u1_sectors = [U1Irrep::new(-1), U1Irrep::new(-2)];

    let z2_complex: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &z2_dual,
        complex_spectra(z2_sectors, complex_values),
    )
    .unwrap();
    let z2_real: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &z2_dual, real_spectra(z2_sectors, real_values)).unwrap();
    assert_dual_diagonal_qr_lq!(runtime, z2_complex, calls);
    assert_dual_diagonal_qr_lq!(runtime, z2_complex.convert::<Complex32>(), calls);
    assert_dual_diagonal_qr_lq!(runtime, z2_real, calls);
    assert_dual_diagonal_qr_lq!(runtime, z2_real.convert::<f32>(), calls);

    let u1_complex: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &u1_dual,
        complex_spectra(u1_sectors, complex_values),
    )
    .unwrap();
    let u1_real: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &u1_dual, real_spectra(u1_sectors, real_values)).unwrap();
    assert_dual_diagonal_qr_lq!(runtime, u1_complex, calls);
    assert_dual_diagonal_qr_lq!(runtime, u1_complex.convert::<Complex32>(), calls);
    assert_dual_diagonal_qr_lq!(runtime, u1_real, calls);
    assert_dual_diagonal_qr_lq!(runtime, u1_real.convert::<f32>(), calls);
    // The compact adjoint of an admitted diagonal is itself admitted.
    assert_dual_diagonal_qr_lq!(runtime, u1_complex.adjoint().unwrap(), calls);
    assert_eq!(calls.of(Kernel::QR), 0);
    assert_eq!(calls.total(), 0);

    // Swapped leg roles: the swap of a compact diagonal stays compact on
    // TensorKit's `dual(d.domain)` (#1866), so it is a dual diagonal whose
    // QR/LQ take the compact route above, and the swapped-role factorization
    // of the source runs that same route with no dense QR.
    let nondual: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &u1,
        complex_spectra([U1Irrep::new(1), U1Irrep::new(2)], complex_values),
    )
    .unwrap();
    let swapped = nondual.permute(&[1], &[0]).unwrap();
    assert!(
        symtn::typed::__network::network_reuse_class(&swapped, false) == NetworkReuseClass::Compact
    );
    assert_eq!(swapped.codomain()[0], u1_dual);
    assert_eq!(swapped.domain()[0], u1_dual);
    assert_dual_diagonal_qr_lq!(runtime, swapped, calls);
    let Qr { q, r } = nondual.qr_compact(&[1], &[0]).unwrap();
    let Lq { l, q: lq_q } = nondual.lq_compact(&[1], &[0]).unwrap();
    assert_eq!(calls.of(Kernel::QR), 0);
    for factor in [&q, &r, &l, &lq_q] {
        assert!(
            symtn::typed::__network::network_reuse_class(factor, false)
                == NetworkReuseClass::Compact
        );
    }
    for rebuilt in [q.compose(&r).unwrap(), l.compose(&lq_q).unwrap()] {
        numerics::assert_slices_close(
            "swapped-role compact QR/LQ reconstruction",
            rebuilt.materialize().unwrap().dense_data().unwrap(),
            swapped.materialize().unwrap().dense_data().unwrap(),
            2,
        );
    }
    calls.reset();

    // A nonfinite dual diagonal is refused by the shared finite-input stage,
    // with no dense QR.
    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &u1_dual,
        real_spectra(u1_sectors, [&[f64::NAN, 1.0], &[2.0, 3.0, 4.0]]),
    )
    .unwrap();
    assert!(
        matches!(nonfinite.qr_compact(&[0], &[1]), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::InvalidArgument { message }))) if message.contains("qr input components must be finite"))
    );
    assert!(
        matches!(nonfinite.lq_full(&[0], &[1]), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::InvalidArgument { message }))) if message.contains("lq input components must be finite"))
    );
    assert_eq!(calls.of(Kernel::QR), 0);

    // A materialized dual payload keeps the dense route and its fresh nondual
    // W. U(1) W is `flip(V')`, nondual with the same physical sectors -1 and -2.
    for (index, input) in [u1_real.materialize().unwrap()].iter().enumerate() {
        let before = calls.of(Kernel::QR);
        let Qr { r, .. } = input.qr_compact(&[0], &[1]).unwrap();
        let Qr { r: full_r, .. } = input.qr_full(&[0], &[1]).unwrap();
        let Lq { l, .. } = input.lq_compact(&[0], &[1]).unwrap();
        let Lq { l: full_l, .. } = input.lq_full(&[0], &[1]).unwrap();
        assert_eq!(calls.of(Kernel::QR), before + 8, "input {index}");
        for bond in [
            &r.codomain()[0],
            &full_r.codomain()[0],
            &l.domain()[0],
            &full_l.domain()[0],
        ] {
            assert!(!bond.is_dual());
            assert_eq!(bond.degeneracy(&U1Irrep::new(-1)).unwrap(), 2);
            assert_eq!(bond.degeneracy(&U1Irrep::new(-2)).unwrap(), 3);
        }
        for factor in [&r, &full_r, &l, &full_l] {
            assert!(
                symtn::typed::__network::network_reuse_class(factor, false)
                    == NetworkReuseClass::OwnedDense
            );
        }
    }

    // A lazy adjoint of a dense dual tensor factors like its materialized
    // adjoint (#1755): both QR and LQ materialize it for the call.
    let lazy = u1_real.materialize().unwrap().adjoint().unwrap();
    let eager = lazy.materialize().unwrap();
    let close = |a: &TensorMap<_, f64>, b: &TensorMap<_, f64>| {
        assert_eq!(a.codomain(), b.codomain());
        assert_eq!(a.domain(), b.domain());
        let (a, b) = (a.materialize().unwrap(), b.materialize().unwrap());
        assert!(a
            .dense_data()
            .unwrap()
            .iter()
            .zip(b.dense_data().unwrap())
            .all(|(x, y)| (x - y).abs() < 1.0e-12));
    };
    for (lazy_qr, eager_qr) in [
        (lazy.qr_compact(&[0], &[1]), eager.qr_compact(&[0], &[1])),
        (lazy.qr_full(&[0], &[1]), eager.qr_full(&[0], &[1])),
    ] {
        let (Qr { q, r }, Qr { q: eq, r: er }) = (lazy_qr.unwrap(), eager_qr.unwrap());
        close(&q, &eq);
        close(&r, &er);
    }
    for (lazy_lq, eager_lq) in [
        (lazy.lq_compact(&[0], &[1]), eager.lq_compact(&[0], &[1])),
        (lazy.lq_full(&[0], &[1]), eager.lq_full(&[0], &[1])),
    ] {
        let (Lq { l, q }, Lq { l: el, q: eq }) = (lazy_lq.unwrap(), eager_lq.unwrap());
        close(&l, &el);
        close(&q, &eq);
    }
}

/// A compact diagonal's QR and LQ factors, full and compact alike, live on
/// the admitted input bond `V <- V`, dual orientation included: they are
/// published on the input space itself, so the provider is never queried for
/// an output space and a failing provider cannot fail them.
#[test]
fn checked_dual_diagonal_qr_lq_publish_on_the_input_bond_without_provider_queries() {
    let _serial = crate::counting_alloc::serial();
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(SpyExecutor::counting(&calls)))
        .build()
        .unwrap();
    for dual in [false, true] {
        let provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
        let mut bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
        if dual {
            bond = bond.try_dual().unwrap();
        }
        let finite: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, -1.0],
            }],
        )
        .unwrap();
        provider.fail_algebra.store(true, Ordering::Relaxed);
        reset_provider_queries(&provider);
        let Qr { q, r } = finite.qr_compact(&[0], &[1]).unwrap();
        let Lq { l, q: lq_q } = finite.lq_compact(&[0], &[1]).unwrap();
        let Qr {
            q: full_q,
            r: full_r,
        } = finite.qr_full(&[0], &[1]).unwrap();
        let Lq {
            l: full_l,
            q: full_lq_q,
        } = finite.lq_full(&[0], &[1]).unwrap();
        assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
        for factor in [&q, &r, &l, &lq_q, &full_q, &full_r, &full_l, &full_lq_q] {
            assert!(
                symtn::typed::__network::network_reuse_class(factor, false)
                    == NetworkReuseClass::Compact
            );
            assert_eq!(factor.codomain(), finite.codomain());
            assert_eq!(factor.domain(), finite.domain());
            assert_eq!(factor.codomain()[0].is_dual(), dual);
        }
    }
    assert_eq!(calls.of(Kernel::QR), 0);
    assert_eq!(calls.total(), 0);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_lq_preserves_provider_and_reconstructs() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Lq { l, q } = source.lq_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(&q).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>();
    let Lq {
        l: complex_l,
        q: complex_q,
    } = complex.lq_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    let complex_rebuilt = complex_l.compose(&complex_q).unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_full_qr_preserves_provider_and_reconstructs() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Qr { q, r } = source.qr_full(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(&r).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>();
    let Qr {
        q: complex_q,
        r: complex_r,
    } = complex.qr_full(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(&complex_r).unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[test]
fn checked_generic_compact_qr_failure_is_typed_and_nonpublishing() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(120));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    // The construction published the layouts this operation walks; a
    // provider whose answers change is not one identity, so the cached
    // layouts go first (#2030). The tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_compact(&[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_compact_lq_failure_is_typed_and_nonpublishing() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(121));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    // The construction published the layouts this operation walks; a
    // provider whose answers change is not one identity, so the cached
    // layouts go first (#2030). The tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_compact(&[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_full_qr_failure_is_typed_and_nonpublishing() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_full(&[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_full_lq_reconstructs_and_preserves_provider() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let Lq { l, q } = source.lq_full(&[0, 1], &[2]).unwrap();
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(&q).unwrap();
    assert_eq!(
        rebuilt.dense_data().unwrap().len(),
        source.dense_data().unwrap().len()
    );
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
}

#[test]
fn checked_generic_full_lq_supports_complex_scalars() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
            Complex64::new(2.0, 1.0)
        })
        .unwrap();
    let Lq { l, q } = source.lq_full(&[0, 1], &[2]).unwrap();
    let rebuilt = l.compose(&q).unwrap();
    assert!((rebuilt.norm(2.0).unwrap() - source.norm(2.0).unwrap()).abs() < 1e-12);
}

#[test]
fn checked_generic_full_lq_failure_is_typed_and_nonpublishing() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_full(&[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}
