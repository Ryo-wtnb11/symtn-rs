//! #1996: every factorization leases its dense executor once, only on the
//! dense route, the same way in both fusion modes and on both runtime lease
//! arms (pooled executor, injected executor behind the state lock). A compact
//! diagonal and every error raised before the dense stage (preflight, leg
//! roles, the shared finite-input stage) take no lease. Every adjoint
//! materialization and compact densification runs inside the runtime's Host
//! pool.

use super::*;
use num_complex::Complex64;
#[cfg(feature = "cgc-gen")]
use symtn_core::SUNFusionRule;

/// What `f` did on this thread: whether it succeeded, its dense leases,
/// and its materializations inside and outside a runtime Host pool.
fn leases<T, E>(f: impl FnOnce() -> Result<T, E>) -> Probe {
    let recording = probe::record();
    let ok = f().is_ok();
    Probe {
        ok,
        leases: recording.get(Event::DenseLease),
        pooled: recording.get(Event::PooledMaterialization),
        unpooled: recording.get(Event::UnpooledMaterialization),
    }
}

#[derive(Debug)]
struct Probe {
    ok: bool,
    leases: usize,
    pooled: usize,
    unpooled: usize,
}

fn spectrum_of<R, D>(
    bond: &GradedSpace<R>,
    value: impl Fn(usize) -> D,
) -> Vec<SectorSpectrum<R::Sector, D>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let mut next = 0;
    bond.sectors()
        .unwrap()
        .into_iter()
        .map(|sector| {
            let values = (0..bond.degeneracy(&sector).unwrap())
                .map(|_| {
                    next += 1;
                    value(next)
                })
                .collect();
            SectorSpectrum { sector, values }
        })
        .collect()
}

/// `(row, succeeded, leases)` of every family on one rule, dtype and
/// runtime: `v` and `w` are non-isomorphic legs.
macro_rules! lease_rows {
    ($runtime:expr, $v:expr, $w:expr, $dtype:ty) => {{
        let (runtime, v, w) = (&$runtime, &$v, &$w);
        let one = <$dtype>::from_real(1.0);
        let rand = |cod: &GradedSpace<_>, dom: &GradedSpace<_>, seed| {
            TensorMap::<_, $dtype>::rand_with_seed(runtime, [cod], [dom], seed).unwrap()
        };
        let a = rand(v, v, 1);
        let y = rand(v, v, 2);
        let vw = rand(v, w, 3);
        let h = a.axpby(one, a.adjoint_view(), one).unwrap();
        let nan = a
            .axpby(<$dtype>::from_real(f64::NAN), &a, <$dtype>::from_real(0.0))
            .unwrap();
        let d: TensorMap<_, $dtype> = TensorMap::diagonal(
            runtime,
            v,
            spectrum_of(v, |k| <$dtype>::from_real(1.0 + 0.5 * k as f64)),
        )
        .unwrap();
        let tol = HermitianTol::DEFAULT;
        let mut rows: Vec<(String, Probe)> = Vec::new();
        let mut row = |name: String, probe: Probe| rows.push((name, probe));
        // Dense owned, lazy adjoint (each family's adjoint rule) and compact.
        let lazy_a = a.adjoint().unwrap();
        let lazy_h = h.adjoint().unwrap();
        for (input, hermitian, kind) in [
            (&a, &h, "dense"),
            (&lazy_a, &lazy_h, "lazy"),
            (&d, &d, "compact"),
        ] {
            let t = |op: &str| format!("{op} {kind}");
            row(t("svd_vals"), leases(|| input.svd_vals(&[0], &[1])));
            row(
                t("eigh_vals"),
                leases(|| hermitian.eigh_vals(&[0], &[1], tol)),
            );
            row(t("eig_vals"), leases(|| input.eig_vals(&[0], &[1])));
            row(t("qr_compact"), leases(|| input.qr_compact(&[0], &[1])));
            row(t("qr_full"), leases(|| input.qr_full(&[0], &[1])));
            row(t("lq_compact"), leases(|| input.lq_compact(&[0], &[1])));
            row(t("lq_full"), leases(|| input.lq_full(&[0], &[1])));
            row(t("svd_compact"), leases(|| input.svd_compact(&[0], &[1])));
            row(t("svd_full"), leases(|| input.svd_full(&[0], &[1])));
            row(t("left_null"), leases(|| input.left_null(&[0], &[1])));
            row(t("right_null"), leases(|| input.right_null(&[0], &[1])));
            row(t("left_polar"), leases(|| input.left_polar(&[0], &[1])));
            row(t("right_polar"), leases(|| input.right_polar(&[0], &[1])));
            row(
                t("eigh_full"),
                leases(|| hermitian.eigh_full(&[0], &[1], tol)),
            );
            row(t("eig_full"), leases(|| input.eig_full(&[0], &[1])));
            row(t("inv"), leases(|| input.inv(&[0], &[1])));
            row(t("pinv"), leases(|| input.pinv(&[0], &[1], 0.0)));
            row(t("exp"), leases(|| input.exp(&[0], &[1])));
            row(
                t("solve divisor"),
                leases(|| input.solve(&[0], &[1], &y, &[0], &[1])),
            );
        }
        // A dense divisor densifies a compact or materializes a lazy rhs.
        row(
            "solve compact rhs dense".into(),
            leases(|| a.solve(&[0], &[1], &d, &[0], &[1])),
        );
        row(
            "solve lazy rhs dense".into(),
            leases(|| a.solve(&[0], &[1], &lazy_a, &[0], &[1])),
        );
        // Errors before the dense stage.
        row(
            "error leg roles".into(),
            leases(|| a.svd_compact(&[0], &[0])),
        );
        row(
            "error inv non-isomorphic".into(),
            leases(|| vw.inv(&[0], &[1])),
        );
        row(
            "error exp non-endomorphic".into(),
            leases(|| vw.exp(&[0], &[1])),
        );
        row(
            "error pinv rcond".into(),
            leases(|| a.pinv(&[0], &[1], -1.0)),
        );
        row(
            "error solve codomain".into(),
            leases(|| a.solve(&[0], &[1], &vw.adjoint().unwrap(), &[0], &[1])),
        );
        row(
            "error solve borrowed rhs".into(),
            leases(|| a.solve(&[0], &[1], y.adjoint_view(), &[0], &[1])),
        );
        row(
            "error svd_vals nonfinite".into(),
            leases(|| nan.svd_vals(&[0], &[1])),
        );
        row(
            "error qr nonfinite".into(),
            leases(|| nan.qr_compact(&[0], &[1])),
        );
        row(
            "error lq nonfinite".into(),
            leases(|| nan.lq_compact(&[0], &[1])),
        );
        row(
            "error svd nonfinite".into(),
            leases(|| nan.svd_compact(&[0], &[1])),
        );
        row(
            "error null nonfinite".into(),
            leases(|| nan.left_null(&[0], &[1])),
        );
        rows
    }};
}

/// Rows whose route must materialize or densify, so the pool check is not
/// vacuous: a compact rhs densified for a dense divisor, and a lazy adjoint
/// copied by every `Materialize` family.
const MATERIALIZING: &[&str] = &[
    "solve compact rhs dense",
    "solve lazy rhs dense",
    "solve divisor lazy",
    "exp lazy",
    "qr_compact lazy",
    "qr_full lazy",
    "eigh_full lazy",
    "eig_vals lazy",
];

fn check(case: &str, rows: Vec<(String, Probe)>) {
    let wrong: Vec<_> = rows
        .iter()
        .filter(|(name, probe)| {
            let error = name.starts_with("error");
            let expected = usize::from(name.ends_with("dense") || name.ends_with("lazy"));
            probe.ok == error
                || probe.leases != expected
                || probe.unpooled != 0
                || (MATERIALIZING.contains(&name.as_str()) && probe.pooled == 0)
        })
        .collect();
    assert!(wrong.is_empty(), "{case}: {wrong:?}");
}

fn runtimes() -> [(&'static str, Runtime); 2] {
    [
        (
            "pooled",
            Runtime::builder().dense_threads(1).build().unwrap(),
        ),
        (
            "locked",
            Runtime::builder()
                .with_dense_executor(Box::new(DefaultDenseExecutor::default()))
                .build()
                .unwrap(),
        ),
    ]
}

#[test]
fn factorizations_lease_once_on_the_dense_route_in_both_modes() {
    let u1 = Arc::new(U1FusionRule);
    let u1_leg = |pairs: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&u1),
            pairs.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = |pairs: &[(usize, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&su2),
            pairs
                .iter()
                .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
        )
        .unwrap()
    };
    for (arm, runtime) in runtimes() {
        let (v, w) = (u1_leg(&[(0, 2), (1, 3)]), u1_leg(&[(0, 2)]));
        check(&format!("U1 f64 {arm}"), lease_rows!(runtime, v, w, f64));
        check(
            &format!("U1 c64 {arm}"),
            lease_rows!(runtime, v, w, Complex64),
        );
        let (v, w) = (su2_leg(&[(0, 2), (1, 2)]), su2_leg(&[(0, 2)]));
        check(&format!("SU2 f64 {arm}"), lease_rows!(runtime, v, w, f64));
        #[cfg(feature = "cgc-gen")]
        {
            let su3 = Arc::new(SUNFusionRule::new(3).unwrap());
            let leg = |pairs: &[(&[i64], usize)]| {
                GradedSpace::try_new(
                    Arc::clone(&su3),
                    pairs.iter().map(|&(l, d)| (l.to_vec(), d)),
                )
                .unwrap()
            };
            let (v, w) = (leg(&[(&[0, 0], 2), (&[1, 0], 3)]), leg(&[(&[0, 0], 2)]));
            check(&format!("SU3 f64 {arm}"), lease_rows!(runtime, v, w, f64));
            check(
                &format!("SU3 c64 {arm}"),
                lease_rows!(runtime, v, w, Complex64),
            );
        }
    }
}

/// #1866: a checked Generic compact full trace reads the spectrum, as the
/// multiplicity-free one does, so it materializes nothing (#2142).
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_trace_reads_the_spectrum_without_materializing() {
    let su3 = Arc::new(SUNFusionRule::new(3).unwrap());
    let v = GradedSpace::try_new(Arc::clone(&su3), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let diagonal: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &v, spectrum_of(&v, |k| 1.0 + 0.5 * k as f64)).unwrap();
    let probe = leases(|| diagonal.trace_pairs(&[(0, 1)]));
    assert!(probe.ok, "{probe:?}");
    assert_eq!((probe.pooled, probe.unpooled), (0, 0), "{probe:?}");
}
