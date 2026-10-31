//! SVD, QR/LQ, null spaces, `eigh`, `eig` and polar. `l = unit ⊕ charged`
//! (one state each); `[l, l] <- [l]` is tall in every coupled sector and
//! `[l] <- [l, l]` wide.

use super::*;
use symtn::typed::{Eig, Eigh, HermitianTol, LeftPolar, Lq, Qr, RightPolar, Svd, Truncation};

fn magnitude<T: numerics::Numeric>(value: T) -> f64 {
    value.wide().norm()
}

/// Deterministic values in `[-0.5, 0.5)` with no low-rank structure.
fn scattered(salt: &mut u64) -> f64 {
    *salt += 1;
    (salt.wrapping_mul(2_654_435_761) % 1009) as f64 / 1009.0 - 0.5
}

/// `w† w` is the identity on `w`'s domain.
macro_rules! assert_isometry {
    ($engine:ident, $runtime:expr, $w:expr, $what:expr) => {{
        let w = $w;
        let gram = w
            .adjoint($engine)
            .unwrap()
            .materialize($engine)
            .unwrap()
            .compose($engine, w)
            .unwrap();
        let identity = TensorMap::isomorphism($engine, $runtime, &w.domain(), &w.domain()).unwrap();
        dense_close(
            $what,
            gram.dense_data().unwrap(),
            identity.dense_data().unwrap(),
        );
    }};
}

/// `w w†` is the identity on `w`'s codomain.
macro_rules! assert_coisometry {
    ($engine:ident, $runtime:expr, $w:expr, $what:expr) => {{
        let w = $w;
        let wh = w.adjoint($engine).unwrap().materialize($engine).unwrap();
        let gram = w.compose($engine, &wh).unwrap();
        let identity =
            TensorMap::isomorphism($engine, $runtime, &w.codomain(), &w.codomain()).unwrap();
        dense_close(
            $what,
            gram.dense_data().unwrap(),
            identity.dense_data().unwrap(),
        );
    }};
}

/// Compact and full SVD, QR and LQ of `$a`: each reconstructs `$a`, has
/// isometric outer factors on `$a`'s own spaces and keeps the source
/// provider; compact `S` is a compact diagonal sorted by magnitude in every
/// sector. A tall input separates compact from full QR, a wide one LQ.
macro_rules! assert_decompositions {
    ($engine:ident, $runtime:expr, $provider:expr, $a:expr, $tall:expr) => {{
        let a = $a;
        let rows: Vec<usize> = (0..a.codomain_rank()).collect();
        let cols: Vec<usize> = (a.codomain_rank()..a.rank()).collect();
        for (name, Svd { u, s, vh }) in [
            ("svd_compact", a.svd_compact($engine, &rows, &cols).unwrap()),
            ("svd_full", a.svd_full($engine, &rows, &cols).unwrap()),
        ] {
            for factor in [&u, &s, &vh] {
                assert!(
                    std::ptr::eq(factor.provider(), $provider.as_ref()),
                    "{name}"
                );
            }
            if name == "svd_compact" {
                assert!(s.dense_data().is_err(), "compact S is a compact diagonal");
                assert!(
                    symtn::typed::__network::network_reuse_class(&s, false)
                        == symtn::typed::__network::NetworkReuseClass::Compact
                );
                for entry in s.diagview($engine).unwrap() {
                    assert!(entry
                        .values
                        .windows(2)
                        .all(|pair| magnitude(pair[0]) >= magnitude(pair[1])));
                }
            }
            assert_eq!(u.codomain(), a.codomain(), "{name}");
            assert_eq!(vh.domain(), a.domain(), "{name}");
            assert_eq!(u.domain(), s.codomain(), "{name}");
            assert_eq!(s.domain(), vh.codomain(), "{name}");
            let rebuilt = u
                .compose($engine, &s)
                .unwrap()
                .compose($engine, &vh)
                .unwrap();
            dense_close(name, rebuilt.dense_data().unwrap(), a.dense_data().unwrap());
            assert_isometry!($engine, $runtime, &u, name);
            assert_coisometry!($engine, $runtime, &vh, name);
        }

        let qr = [
            a.qr_compact($engine, &rows, &cols).unwrap(),
            a.qr_full($engine, &rows, &cols).unwrap(),
        ];
        for Qr { q, r } in &qr {
            assert!(std::ptr::eq(q.provider(), $provider.as_ref()));
            assert_eq!(q.codomain(), a.codomain());
            assert_eq!(r.domain(), a.domain());
            assert_eq!(q.domain(), r.codomain());
            let rebuilt = q.compose($engine, r).unwrap();
            dense_close(
                "Q R",
                rebuilt.dense_data().unwrap(),
                a.dense_data().unwrap(),
            );
            assert_isometry!($engine, $runtime, q, "Q");
        }
        let lq = [
            a.lq_compact($engine, &rows, &cols).unwrap(),
            a.lq_full($engine, &rows, &cols).unwrap(),
        ];
        for Lq { l, q } in &lq {
            assert!(std::ptr::eq(q.provider(), $provider.as_ref()));
            assert_eq!(l.codomain(), a.codomain());
            assert_eq!(q.domain(), a.domain());
            assert_eq!(l.domain(), q.codomain());
            let rebuilt = l.compose($engine, q).unwrap();
            dense_close(
                "L Q",
                rebuilt.dense_data().unwrap(),
                a.dense_data().unwrap(),
            );
            assert_coisometry!($engine, $runtime, q, "Q of LQ");
        }
        if $tall {
            assert_ne!(
                qr[0].q.domain(),
                qr[1].q.domain(),
                "tall: compact and full QR differ"
            );
        } else {
            assert_ne!(
                lq[0].q.codomain(),
                lq[1].q.codomain(),
                "wide: compact and full LQ differ"
            );
        }
    }};
}

in_both_modes! {
    /// Compact and full SVD, QR and LQ of a tall and a wide map, real and
    /// complex (imaginary part independent of the real one), reconstruct the
    /// source with isometric factors on the source's spaces and provider.
    fn svd_qr_lq_reconstruct_with_isometric_factors<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
            .unwrap();
        let mut salt = 0;
        let tall: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| {
                scattered(&mut salt)
            })
            .unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg, &leg], |_, _| {
                scattered(&mut salt)
            })
            .unwrap();
        let complex_tall: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| {
                Complex64::new(scattered(&mut salt), scattered(&mut salt))
            })
            .unwrap();
        let complex_wide: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg, &leg], |_, _| {
                Complex64::new(scattered(&mut salt), scattered(&mut salt))
            })
            .unwrap();
        assert_decompositions!(engine, &runtime, provider, &tall, true);
        assert_decompositions!(engine, &runtime, provider, &wide, false);
        assert_decompositions!(engine, &runtime, provider, &complex_tall, true);
        assert_decompositions!(engine, &runtime, provider, &complex_wide, false);
    }
}

in_both_modes! {
    /// A truncated SVD is `find_truncated` plus `restrict_leg`; its reported
    /// error is the quantum-dimension-weighted 2-norm of the discarded
    /// singular values and equals the reconstruction residual. Keeping
    /// nothing is a policy, not an error.
    fn truncated_svd_error_is_the_weighted_discarded_spectrum_and_the_residual<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
            .unwrap();
        let mut salt = 0;
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| {
                scattered(&mut salt)
            })
            .unwrap();
        let Svd { u, s, vh } = source.svd_compact(engine, &[0, 1], &[2]).unwrap();
        let spectrum = s.diagview(engine).unwrap();
        let mut values: Vec<f64> = spectrum.iter().flat_map(|e| e.values.clone()).collect();
        values.sort_by(|a, b| b.total_cmp(a));
        // Between the two largest values: one sector loses its only value.
        let cutoff = 0.5 * (values[0] + values[1]);
        let expected = spectrum
            .iter()
            .flat_map(|e| e.values.iter().map(|&v| (e.sector, v)))
            .filter(|&(_, v)| v < cutoff)
            .map(|(sector, v)| M::dim(sector) * v * v)
            .sum::<f64>()
            .sqrt();
        assert!(expected > 0.0, "the fixture must discard a nonzero value");

        let found = s.domain()[0]
            .find_truncated(&spectrum, &Truncation::absolute_cutoff(cutoff).unwrap())
            .unwrap();
        numerics::assert_close("reported error", found.error, expected, 4);
        let u = u.restrict_leg(engine, &[(2, &found.selection)]).unwrap();
        let s = s
            .restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
            .unwrap();
        let vh = vh.restrict_leg(engine, &[(0, &found.selection)]).unwrap();
        for factor in [&u, &s, &vh] {
            assert!(std::ptr::eq(factor.provider(), provider.as_ref()));
        }
        let kept: Vec<f64> =
            s.diagview(engine).unwrap().into_iter().flat_map(|e| e.values).collect();
        assert_eq!(kept, [values[0]]);
        let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
        let residual = source
            .axpby(engine, 1.0, &rebuilt, -1.0)
            .unwrap()
            .norm(engine, 2.0)
            .unwrap();
        numerics::assert_close("residual", residual, expected, 16);

        let none = s.domain()[0]
            .find_truncated(&s.diagview(engine).unwrap(), &Truncation::Rank(0))
            .unwrap();
        let empty = s
            .restrict_leg(engine, &[(0, &none.selection), (1, &none.selection)])
            .unwrap();
        assert!(empty.diagview(engine).unwrap().iter().all(|e| e.values.is_empty()));
    }
}

in_both_modes! {
    /// `svd_vals` reports `|σ|` per coupled sector, descending, in label
    /// order, for dense storage and for the same diagonal stored compactly;
    /// `svd_full` of the endomorphism publishes the same values as a compact
    /// `S`. Oracle: signed, out-of-order diagonal blocks.
    fn svd_vals_are_sorted_magnitudes_per_sector_in_label_order<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 3), (M::charged(), 2)])
            .unwrap();
        let (unit_values, charged_values) = ([-2.0, 5.0, -1.0], [3.0, -4.0]);
        let dense: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                if ij[0] != ij[1] {
                    0.0
                } else if trees.coupled() == &M::unit() {
                    unit_values[ij[0]]
                } else {
                    charged_values[ij[0]]
                }
            })
            .unwrap();
        let compact: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: unit_values.to_vec() },
                SectorSpectrum { sector: M::charged(), values: charged_values.to_vec() },
            ],
        )
        .unwrap();
        for (name, tensor) in [("dense", &dense), ("compact", &compact)] {
            let spectrum: Vec<_> = tensor
                .svd_vals(engine, &[0], &[1])
                .unwrap()
                .into_iter()
                .map(|e| (e.sector, e.values))
                .collect();
            assert_eq!(
                spectrum,
                [(M::unit(), vec![5.0, 2.0, 1.0]), (M::charged(), vec![4.0, 3.0])],
                "{name}"
            );
        }
        let s = dense.svd_full(engine, &[0], &[1]).unwrap().s;
        assert!(s.dense_data().is_err(), "full S of an endomorphism is compact");
        let values: Vec<_> = s.diagview(engine).unwrap().into_iter().map(|e| e.values).collect();
        dense_close("svd_full S", &values.concat(), &[5.0, 2.0, 1.0, 4.0, 3.0]);
        assert_eq!(s.materialize(engine).unwrap().diagview(engine).unwrap(), s.diagview(engine).unwrap());
    }
}

in_both_modes! {
    /// Null spaces are shape-based: `left_null` of a tall map has
    /// `rows - cols` states per coupled sector, is isometric and annihilates
    /// the map from the left; `right_null` of a wide map mirrors it; the
    /// other side is empty.
    fn null_spaces_complete_the_range_and_annihilate_the_source<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
            .unwrap();
        let mut salt = 0;
        let tall: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| {
                scattered(&mut salt)
            })
            .unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg, &leg], |_, _| {
                scattered(&mut salt)
            })
            .unwrap();

        let left = tall.left_null(engine, &[0, 1], &[2]).unwrap();
        assert_eq!(left.codomain(), tall.codomain());
        assert_eq!(left.domain()[0].degeneracies(), M::tall_null_dims());
        assert_isometry!(engine, &runtime, &left, "left_null");
        let annihilated = left
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .compose(engine, &tall)
            .unwrap();
        assert!(annihilated.dense_data().unwrap().iter().all(|v| v.abs() < 1e-12));
        assert!(tall
            .right_null(engine, &[0, 1], &[2])
            .unwrap()
            .codomain()[0]
            .degeneracies()
            .is_empty());

        let right = wide.right_null(engine, &[0], &[1, 2]).unwrap();
        assert_eq!(right.domain(), wide.domain());
        assert_eq!(right.codomain()[0].degeneracies(), M::tall_null_dims());
        assert_coisometry!(engine, &runtime, &right, "right_null");
        let right_adjoint = right.adjoint(engine).unwrap().materialize(engine).unwrap();
        let annihilated = wide.compose(engine, &right_adjoint).unwrap();
        assert!(annihilated.dense_data().unwrap().iter().all(|v| v.abs() < 1e-12));
        assert!(wide
            .left_null(engine, &[0], &[1, 2])
            .unwrap()
            .domain()[0]
            .degeneracies()
            .is_empty());
    }
}

// ---------------------------------------------------------------------------
// Eigendecompositions.
// ---------------------------------------------------------------------------

/// `unit ⊕ charged` blocks `[[2,1],[1,2]]` (eigenvalues 1, 3) and
/// `diag(-3, 2, 1)`, in either dtype.
macro_rules! hermitian_fixture {
    ($engine:ident, $runtime:expr, $bond:expr, $mode:ty, $dtype:ty) => {
        TensorMap::<_, $dtype>::from_subblock_fn(
            $engine,
            $runtime,
            [$bond],
            [$bond],
            |trees, ij| {
                let value = if trees.coupled() == &<$mode>::unit() {
                    [[2.0, 1.0], [1.0, 2.0]][ij[0]][ij[1]]
                } else if ij[0] == ij[1] {
                    [-3.0, 2.0, 1.0][ij[0]]
                } else {
                    0.0
                };
                <$dtype>::from(value)
            },
        )
        .unwrap()
    };
}

in_both_modes! {
    /// `eigh` of a Hermitian endomorphism: ascending real eigenvalues per
    /// sector in label order (also for complex and compact storage),
    /// `V D V† = H` with unitary `V` on the source provider, and a truncation
    /// whose error is the dimension-weighted norm of the dropped values.
    /// Oracle: hand-diagonalized blocks.
    fn eigh_matches_hand_spectra_and_reconstructs<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 3)])
            .unwrap();
        let real = hermitian_fixture!(engine, &runtime, &bond, M, f64);
        let complex = hermitian_fixture!(engine, &runtime, &bond, M, Complex64);
        let compact: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: vec![3.0, 1.0] },
                SectorSpectrum { sector: M::charged(), values: vec![-3.0, 2.0, 1.0] },
            ],
        )
        .unwrap();
        let expected = [vec![1.0, 3.0], vec![-3.0, 1.0, 2.0]];
        let flat: Vec<f64> = expected.concat();
        for (name, spectrum) in [
            ("dense", real.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT).unwrap()),
            ("compact", compact.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT).unwrap()),
        ] {
            assert_eq!(
                spectrum.iter().map(|e| e.sector).collect::<Vec<_>>(),
                [M::unit(), M::charged()],
                "{name}"
            );
            let got: Vec<f64> = spectrum.into_iter().flat_map(|e| e.values).collect();
            dense_close(name, &got, &flat);
        }
        let got: Vec<f64> = complex
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .into_iter()
            .flat_map(|e| e.values)
            .collect();
        dense_close("complex", &got, &flat);

        let Eigh { d, v } = real.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT).unwrap();
        assert!(std::ptr::eq(d.provider(), provider.as_ref()));
        assert!(std::ptr::eq(v.provider(), provider.as_ref()));
        assert!(d.dense_data().is_err(), "D is a compact diagonal");
        assert_eq!(v.codomain(), real.codomain());
        let v_adjoint = v.adjoint(engine).unwrap().materialize(engine).unwrap();
        let rebuilt = v.compose(engine, &d).unwrap().compose(engine, &v_adjoint).unwrap();
        dense_close("V D V†", rebuilt.dense_data().unwrap(), real.dense_data().unwrap());
        assert_isometry!(engine, &runtime, &v, "V");
        let Eigh { d: complex_d, v: complex_v } =
            complex.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT).unwrap();
        assert!(complex_d
            .diagview(engine)
            .unwrap()
            .iter()
            .flat_map(|e| &e.values)
            .all(|value| value.im == 0.0));
        let complex_v_adjoint = complex_v.adjoint(engine).unwrap().materialize(engine).unwrap();
        let rebuilt = complex_v
            .compose(engine, &complex_d)
            .unwrap()
            .compose(engine, &complex_v_adjoint)
            .unwrap();
        dense_close("complex V D V†", rebuilt.dense_data().unwrap(), complex.dense_data().unwrap());

        // |λ| < 1.5 drops the unit sector's 1 and the charged sector's 1.
        let found = d.domain()[0]
            .find_truncated(
                &d.diagview(engine).unwrap(),
                &Truncation::absolute_cutoff(1.5).unwrap(),
            )
            .unwrap();
        let truncated = d
            .restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
            .unwrap();
        let kept: Vec<f64> =
            truncated.diagview(engine).unwrap().into_iter().flat_map(|e| e.values).collect();
        dense_close("kept", &kept, &[3.0, -3.0, 2.0]);
        let error = (M::dim(M::unit()) + M::dim(M::charged())).sqrt();
        numerics::assert_close("eigh truncation error", found.error, error, 4);
    }
}

in_both_modes! {
    /// `eigh` refuses a non-endomorphism, a non-Hermitian block (for the
    /// values too) and a nonfinite block, each with its own reason.
    fn eigh_refuses_invalid_inputs<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let wide = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 2)]).unwrap();
        let narrow = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 1)]).unwrap();
        let reason = |error: OperationError| match error {
            OperationError::InvalidArgument { message }
            | OperationError::SpaceMismatch { message } => message,
            other => panic!("{other:?}"),
        };
        let nonendomorphism: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
        let error = operation_err(nonendomorphism.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT));
        assert!(matches!(error, OperationError::SpaceMismatch { .. }), "{error:?}");
        assert!(reason(error).contains("endomorphism"));

        let nonhermitian: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, ij| {
                [[0.0, 1.0], [0.0, 0.0]][ij[0]][ij[1]]
            })
            .unwrap();
        for error in [
            operation_err(nonhermitian.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)),
            operation_err(nonhermitian.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)),
        ] {
            assert!(matches!(error, OperationError::InvalidArgument { .. }), "{error:?}");
            assert!(reason(error).contains("Hermitian coupled-sector blocks"));
        }

        let nonfinite: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, ij| {
                if ij[0] == ij[1] { f64::NAN } else { 0.0 }
            })
            .unwrap();
        let error = operation_err(nonfinite.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT));
        assert!(matches!(error, OperationError::InvalidArgument { .. }), "{error:?}");
        assert!(reason(error).contains("input components must be finite"));
    }
}

in_both_modes! {
    /// `eig` of a non-normal endomorphism: complex eigenpairs with
    /// `A V = V D`, eigenvalues per sector in label order, a truncation error
    /// that is the dimension-weighted norm of the dropped values, and for a
    /// complex input a genuinely nonreal spectrum whose adjoint conjugates.
    /// Oracle: triangular `[[2,1],[0,3]]` (unit) and `diag(-3, 2, 1)`.
    fn eig_matches_hand_spectra_and_the_eigen_equation<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 3)])
            .unwrap();
        let real: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                if trees.coupled() == &M::unit() {
                    [[2.0, 1.0], [0.0, 3.0]][ij[0]][ij[1]]
                } else if ij[0] == ij[1] {
                    [-3.0, 2.0, 1.0][ij[0]]
                } else {
                    0.0
                }
            })
            .unwrap();
        let sorted = |mut values: Vec<Complex64>| {
            values.sort_by(|a, b| a.re.total_cmp(&b.re));
            values
        };
        let spectrum = real.eig_vals(engine, &[0], &[1]).unwrap();
        assert_eq!(spectrum.iter().map(|e| e.sector).collect::<Vec<_>>(), [M::unit(), M::charged()]);
        let c = |re: f64| Complex64::new(re, 0.0);
        dense_close("unit", &sorted(spectrum[0].values.clone()), &[c(2.0), c(3.0)]);
        dense_close("charged", &sorted(spectrum[1].values.clone()), &[c(-3.0), c(1.0), c(2.0)]);

        let Eig { d, v } = real.eig_full(engine, &[0], &[1]).unwrap();
        assert!(std::ptr::eq(d.provider(), provider.as_ref()));
        assert!(std::ptr::eq(v.provider(), provider.as_ref()));
        let av = real.convert::<Complex64>(engine).unwrap().compose(engine, &v).unwrap();
        let vd = v.compose(engine, &d).unwrap();
        dense_close("A V = V D", av.dense_data().unwrap(), vd.dense_data().unwrap());

        // |λ| < 1.5 drops only the charged sector's 1.
        let found = d.domain()[0]
            .find_truncated(
                &d.diagview(engine).unwrap(),
                &Truncation::absolute_cutoff(1.5).unwrap(),
            )
            .unwrap();
        numerics::assert_close("eig truncation error", found.error, M::dim(M::charged()).sqrt(), 4);

        let complex = real
            .convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, Complex64::new(1.0, 1.0))
            .unwrap();
        let Eig { d, v } = complex.eig_full(engine, &[0], &[1]).unwrap();
        let av = complex.compose(engine, &v).unwrap();
        let vd = v.compose(engine, &d).unwrap();
        dense_close("complex A V = V D", av.dense_data().unwrap(), vd.dense_data().unwrap());
        let values: Vec<Complex64> =
            d.diagview(engine).unwrap().into_iter().flat_map(|e| e.values).collect();
        assert!(values.iter().all(|value| value.im.abs() > 0.5), "{values:?}");
        let conjugated: Vec<Complex64> = d
            .adjoint(engine)
            .unwrap()
            .diagview(engine)
            .unwrap()
            .into_iter()
            .flat_map(|e| e.values)
            .collect();
        assert_eq!(conjugated, values.iter().map(|v| v.conj()).collect::<Vec<_>>());
    }
}

// ---------------------------------------------------------------------------
// Polar.
// ---------------------------------------------------------------------------

fn product_2x2<D: symtn::typed::TensorScalar>(a: [[D; 2]; 2], b: [[D; 2]; 2]) -> [[D; 2]; 2] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j]))
}

in_both_modes! {
    /// Polar factors are unique (`P` positive definite), so `left_polar(Q H)`
    /// is `(Q, H)` and `right_polar(H Q)` is `(H, Q)`, real and complex, in
    /// every sector. Oracle: hand-built `Q` (unitary) and `H` (positive).
    fn polar_recovers_hand_built_q_and_h<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 2)])
            .unwrap();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let c = Complex64::new;
        let real = ([[0.0, -1.0], [1.0, 0.0]], [[2.0, 0.5], [0.5, 3.0]]);
        let complex = (
            [[c(s, 0.0), c(0.0, s)], [c(0.0, s), c(s, 0.0)]],
            [[c(2.0, 0.0), c(0.25, 0.5)], [c(0.25, -0.5), c(3.0, 0.0)]],
        );
        let ((q, h), (cq, ch)) = (real, complex);
        let build = |engine: &mut symtn_dense::CpuEngine, m: [[f64; 2]; 2]| {
            TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&bond], [&bond], |_, ij| {
                m[ij[0]][ij[1]]
            })
            .unwrap()
        };
        let (q_tensor, h_tensor) = (build(engine, q), build(engine, h));
        let LeftPolar { w, p } =
            build(engine, product_2x2(q, h)).left_polar(engine, &[0], &[1]).unwrap();
        assert!(std::ptr::eq(w.provider(), provider.as_ref()));
        dense_close("left W", w.dense_data().unwrap(), q_tensor.dense_data().unwrap());
        dense_close("left P", p.dense_data().unwrap(), h_tensor.dense_data().unwrap());
        let RightPolar { p, wh } =
            build(engine, product_2x2(h, q)).right_polar(engine, &[0], &[1]).unwrap();
        dense_close("right W", wh.dense_data().unwrap(), q_tensor.dense_data().unwrap());
        dense_close("right P", p.dense_data().unwrap(), h_tensor.dense_data().unwrap());

        let build = |engine: &mut symtn_dense::CpuEngine, m: [[Complex64; 2]; 2]| {
            TensorMap::<_, Complex64>::from_subblock_fn(engine, &runtime, [&bond], [&bond], |_, ij| {
                m[ij[0]][ij[1]]
            })
            .unwrap()
        };
        let (q_tensor, h_tensor) = (build(engine, cq), build(engine, ch));
        let LeftPolar { w, p } =
            build(engine, product_2x2(cq, ch)).left_polar(engine, &[0], &[1]).unwrap();
        dense_close("complex left W", w.dense_data().unwrap(), q_tensor.dense_data().unwrap());
        dense_close("complex left P", p.dense_data().unwrap(), h_tensor.dense_data().unwrap());
        let RightPolar { p, wh } =
            build(engine, product_2x2(ch, cq)).right_polar(engine, &[0], &[1]).unwrap();
        dense_close("complex right W", wh.dense_data().unwrap(), q_tensor.dense_data().unwrap());
        dense_close("complex right P", p.dense_data().unwrap(), h_tensor.dense_data().unwrap());
    }
}

in_both_modes! {
    /// Rectangular polar with a dual leg on each side: `left_polar` gives `W`
    /// on the source's space (isometric) and positive `P` on
    /// `domain <- domain` with `P² = A† A`; `right_polar` mirrors it. The
    /// wrong side is refused by complete dimensions, also for a side-only
    /// map, and an empty map factors to empty factors.
    fn polar_factor_laws_spaces_and_direction<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
            .unwrap();
        let dual = leg.try_dual().unwrap();
        let mut salt = 0;
        let tall: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &dual], [&dual], |_, _| {
                Complex64::new(scattered(&mut salt), scattered(&mut salt))
            })
            .unwrap();
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&dual], [&leg, &dual], |_, _| {
                scattered(&mut salt)
            })
            .unwrap();

        let LeftPolar { w, p } = tall.left_polar(engine, &[0, 1], &[2]).unwrap();
        assert_eq!((w.codomain(), w.domain()), (tall.codomain(), tall.domain()));
        assert_eq!((p.codomain(), p.domain()), (tall.domain(), tall.domain()));
        dense_close("W P", w.compose(engine, &p).unwrap().dense_data().unwrap(), tall.dense_data().unwrap());
        assert_isometry!(engine, &runtime, &w, "left W");
        let p_adjoint = p.adjoint(engine).unwrap().materialize(engine).unwrap();
        dense_close("P = P†", p_adjoint.dense_data().unwrap(), p.dense_data().unwrap());
        let tall_adjoint = tall.adjoint(engine).unwrap().materialize(engine).unwrap();
        dense_close(
            "P² = A† A",
            p.compose(engine, &p).unwrap().dense_data().unwrap(),
            tall_adjoint.compose(engine, &tall).unwrap().dense_data().unwrap(),
        );
        assert!(p
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .iter()
            .flat_map(|e| &e.values)
            .all(|&value| value >= -1e-12));

        let RightPolar { p, wh } = wide.right_polar(engine, &[0], &[1, 2]).unwrap();
        assert_eq!((wh.codomain(), wh.domain()), (wide.codomain(), wide.domain()));
        assert_eq!((p.codomain(), p.domain()), (wide.codomain(), wide.codomain()));
        dense_close("P W", p.compose(engine, &wh).unwrap().dense_data().unwrap(), wide.dense_data().unwrap());
        assert_coisometry!(engine, &runtime, &wh, "right W");
        let wide_adjoint = wide.adjoint(engine).unwrap().materialize(engine).unwrap();
        dense_close(
            "P² = A A†",
            p.compose(engine, &p).unwrap().dense_data().unwrap(),
            wide.compose(engine, &wide_adjoint).unwrap().dense_data().unwrap(),
        );

        let message = |error: OperationError| match error {
            OperationError::InvalidArgument { message } => message,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            message(operation_err(tall.right_polar(engine, &[0, 1], &[2]))),
            "right_polar requires columns >= rows in every coupled-sector matrix"
        );
        assert_eq!(
            message(operation_err(wide.left_polar(engine, &[0], &[1, 2]))),
            "left_polar requires rows >= columns in every coupled-sector matrix"
        );

        let domain_only = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1)]).unwrap();
        let side_only: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&domain_only], |_, _| 0.0)
                .unwrap();
        assert!(side_only.left_polar(engine, &[0], &[1]).is_ok());
        assert!(message(operation_err(side_only.right_polar(engine, &[0], &[1])))
            .contains("right_polar requires columns >= rows"));
        let empty = GradedSpace::try_new(Arc::clone(&provider), []).unwrap();
        let empty_map: TensorMap<_, f64> =
            TensorMap::zeros(engine, &runtime, [&empty], [&empty]).unwrap();
        assert!(empty_map.left_polar(engine, &[0], &[1]).unwrap().w.dense_data().unwrap().is_empty());
        assert!(empty_map.right_polar(engine, &[0], &[1]).unwrap().p.dense_data().unwrap().is_empty());
    }
}
