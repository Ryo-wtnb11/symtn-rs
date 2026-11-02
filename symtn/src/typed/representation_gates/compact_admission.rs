//! #1800 + #1994: a compact diagonal is admitted, and its S/D bond built,
//! through one matrix-algebra authority in both fusion modes.
//!
//! The same spectra run on a multiplicity-free U(1) and SU(2) rule and on
//! the checked SU(3) rule (fusion-symbols), on the canonical bond layout and its
//! dual. Every op must take the same route
//! class with bit-identical values in every row. The oracle is independent of
//! the route under test: compact `inv`/`pinv`/`exp`/`\` are elementwise
//! (TensorKit `src/tensors/diagonal.jl`), `S`/`D` of a diagonal are its
//! sorted magnitudes / values, and the expected values below are written out.

use super::*;
use num_complex::{Complex32, Complex64};
use symtn_core::LieGroupFusionRule;
use symtn_matrixalgebra::FactorScalar;

/// Route class and bit pattern of a result. Value lists are sorted by
/// length (the fixtures' degeneracies differ), so sector ids of different
/// rules compare alike.
#[derive(Clone, Debug, PartialEq)]
enum Route {
    Compact(Vec<Vec<(u64, u64)>>),
    Dense,
    Failed(String),
}

fn bits<D: FactorScalar>(value: D) -> (u64, u64) {
    let value = value.widen_complex();
    (value.re.to_bits(), value.im.to_bits())
}

fn sorted(mut values: Vec<Vec<(u64, u64)>>) -> Vec<Vec<(u64, u64)>> {
    values.sort_by_key(Vec::len);
    values
}

fn route<R, D: FactorScalar>(tensor: &TensorMap<R, D>) -> Route {
    match owned(tensor).data.as_ref() {
        TypedData::Diagonal(spectrum) => Route::Compact(sorted(
            spectrum
                .iter()
                .map(|entry| entry.values.iter().map(|&value| bits(value)).collect())
                .collect(),
        )),
        TypedData::Dense(_) => Route::Dense,
    }
}

/// Whether a compact `S`/`D` sits on a canonical, nondual bond: TensorKit's
/// `fuse(V)`, a fresh bond whatever the input's layout or duality.
fn on_fresh_bond<R, D>(tensor: &TensorMap<R, D>) -> bool {
    let space = owned(tensor).space.space();
    let leg = &space.homspace().codomain().legs()[0];
    !leg.is_dual()
        && matches!(
            space.structure().coupled_sector_regions(1),
            Ok(Some(regions)) if regions.iter().all(|region| region.has_aligned_diagonal())
        )
}

macro_rules! row {
    ($result:expr, $map:expr) => {
        match $result {
            Ok(value) => $map(value),
            // The facade error type is per mode by design (#1862); the
            // root cause is what both must agree on.
            Err(error) => Route::Failed(super::root_cause(&error)),
        }
    };
}

/// One row per op of the matrix, for compact input `$t`.
macro_rules! matrix {
    ($engine:ident, $t:expr, $rcond:expr) => {{
        let t = &$t;
        let flag = |value: bool| Route::Failed(format!("fresh bond: {value}"));
        vec![
            (
                "svd_vals",
                row!(t.svd_vals($engine, &[0], &[1]), |values: Vec<
                    SectorSpectrum<_, f64>,
                >| {
                    Route::Compact(sorted(
                        values
                            .into_iter()
                            .map(|entry| entry.values.into_iter().map(bits).collect())
                            .collect(),
                    ))
                }),
            ),
            (
                "svd_compact.s",
                row!(t.svd_compact($engine, &[0], &[1]), |f: Svd<_>| route(&f.s)),
            ),
            (
                "svd_compact.bond",
                row!(t.svd_compact($engine, &[0], &[1]), |f: Svd<_>| flag(
                    on_fresh_bond(&f.s)
                )),
            ),
            (
                "svd_full.s",
                row!(t.svd_full($engine, &[0], &[1]), |f: Svd<_>| route(&f.s)),
            ),
            (
                "svd_full.bond",
                row!(t.svd_full($engine, &[0], &[1]), |f: Svd<_>| flag(
                    on_fresh_bond(&f.s)
                )),
            ),
            (
                "eig_full.d",
                row!(t.eig_full($engine, &[0], &[1]), |f: Eig<_>| route(&f.d)),
            ),
            (
                "eig_full.bond",
                row!(t.eig_full($engine, &[0], &[1]), |f: Eig<_>| flag(
                    on_fresh_bond(&f.d)
                )),
            ),
            (
                "eigh_full.d",
                row!(
                    t.eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT),
                    |f: Eigh<_>| route(&f.d)
                ),
            ),
            ("inv", row!(t.inv($engine, &[0], &[1]), |x| route(&x))),
            (
                "pinv",
                row!(t.pinv($engine, &[0], &[1], $rcond), |x| route(&x)),
            ),
            ("exp", row!(t.exp($engine, &[0], &[1]), |x| route(&x))),
            (
                "solve",
                row!(t.solve($engine, &[0], &[1], t, &[0], &[1]), |x| route(&x)),
            ),
        ]
    }};
}

type Rows = Vec<(&'static str, Route)>;

/// `two` on the bond's two-fold sector and `three` on its three-fold one,
/// labelled as the (possibly dual) bond labels them.
fn spectra<R, D: Copy>(
    bond: &GradedSpace<R>,
    two: [D; 2],
    three: [D; 3],
) -> Vec<SectorSpectrum<R::Sector, D>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let Ok(sectors) = bond.sectors() else {
        panic!("bond sectors");
    };
    sectors
        .into_iter()
        .map(|sector| {
            let Ok(degeneracy) = bond.degeneracy(&sector) else {
                panic!("bond degeneracy");
            };
            let values = if degeneracy == 2 {
                two.to_vec()
            } else {
                three.to_vec()
            };
            SectorSpectrum { sector, values }
        })
        .collect()
}

/// The matrix of one dtype's spectra, `(two values, three values)`, on the
/// canonical layout of U(1), SU(2) and SU(3) bonds and their duals; every
/// rule must agree row by row. Returns the agreed rows.
fn agreed_matrix<D>(engine: &mut dyn Engine, two: [D; 2], three: [D; 3], rcond: f64) -> Rows
where
    D: AdvancedLinalgScalar + FactorizationScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    let runtime = Runtime::builder().build().unwrap();
    let mut all: Vec<(&str, Rows)> = Vec::new();

    let u1 = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    for (name, bond) in [("u1", bond.clone()), ("u1 dual", bond.try_dual().unwrap())] {
        let t: TensorMap<_, D> =
            TensorMap::diagonal(engine, &runtime, &bond, spectra(&bond, two, three)).unwrap();
        all.push((name, matrix!(engine, t, rcond)));
    }

    let su2 = Arc::new(SU2FusionRule);
    let (zero, half) = (SU2Irrep::from_twice_spin(0), SU2Irrep::from_twice_spin(1));
    let bond = GradedSpace::try_new(Arc::clone(&su2), [(zero, 2), (half, 3)]).unwrap();
    let t: TensorMap<_, D> =
        TensorMap::diagonal(engine, &runtime, &bond, spectra(&bond, two, three)).unwrap();
    all.push(("su2", matrix!(engine, t, rcond)));

    let su3 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let bond = GradedSpace::try_new(Arc::clone(&su3), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    for (name, bond) in [
        ("su3", bond.clone()),
        ("su3 dual", bond.try_dual().unwrap()),
    ] {
        let t: TensorMap<_, D> =
            TensorMap::diagonal(engine, &runtime, &bond, spectra(&bond, two, three)).unwrap();
        all.push((name, matrix!(engine, t, rcond)));
    }

    let (reference_name, reference) = all[0].clone();
    for (name, rows) in &all[1..] {
        for ((op, expected), (_, actual)) in reference.iter().zip(rows) {
            assert_eq!(
                actual, expected,
                "{op}: {name} differs from {reference_name}"
            );
        }
    }
    reference
}

fn get<'a>(rows: &'a Rows, op: &str) -> &'a Route {
    &rows.iter().find(|(name, _)| *name == op).unwrap().1
}

fn compact<D: FactorScalar>(values: &[&[D]]) -> Route {
    Route::Compact(sorted(
        values
            .iter()
            .map(|values| values.iter().map(|&value| bits(value)).collect())
            .collect(),
    ))
}

#[test]
fn finite_compact_diagonals_route_alike_on_every_rule_and_layout() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let rows = agreed_matrix(engine, [3.0_f64, -0.5], [2.0, 0.25, -1.5], 0.0);
    for op in ["svd_compact.s", "svd_full.s", "eig_full.d", "eigh_full.d"] {
        assert!(matches!(get(&rows, op), Route::Compact(_)), "{op}");
    }
    for op in ["svd_compact.bond", "svd_full.bond", "eig_full.bond"] {
        assert_eq!(get(&rows, op), &Route::Failed("fresh bond: true".into()));
    }
    assert_eq!(
        get(&rows, "inv"),
        &compact::<f64>(&[&[1.0 / 3.0, -2.0], &[0.5, 4.0, -1.0 / 1.5]])
    );
    assert_eq!(get(&rows, "pinv"), get(&rows, "inv"));
    assert_eq!(
        get(&rows, "exp"),
        &compact::<f64>(&[
            &[3.0_f64.exp(), (-0.5_f64).exp()],
            &[2.0_f64.exp(), 0.25_f64.exp(), (-1.5_f64).exp()]
        ])
    );
    assert_eq!(
        get(&rows, "solve"),
        &compact::<f64>(&[&[1.0, 1.0], &[1.0; 3]])
    );

    let c = Complex64::new;
    let rows = agreed_matrix(
        engine,
        [c(3.0, 1.0), c(-0.5, 0.0)],
        [c(2.0, 0.0), c(0.0, 0.25), c(-1.5, 0.5)],
        0.0,
    );
    for op in ["inv", "pinv", "exp", "solve", "svd_compact.s", "svd_full.s"] {
        assert!(matches!(get(&rows, op), Route::Compact(_)), "{op}");
    }
}

#[test]
fn retained_subnormals_stay_compact_with_ieee_reciprocals() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // MF semantics (#1800, TensorKit `pinv(::DiagonalTensorMap)`): a
    // retained subnormal is inverted to `Inf`, not sent to a dense SVD.
    let rows = agreed_matrix(engine, [5e-324_f64, 1.0], [2.0, 1e-310, -1.0], 0.0);
    assert_eq!(
        get(&rows, "pinv"),
        &compact::<f64>(&[&[f64::INFINITY, 1.0], &[0.5, f64::INFINITY, -1.0]])
    );
    assert_eq!(get(&rows, "pinv"), get(&rows, "inv"));
    let Route::Compact(s) = get(&rows, "svd_compact.s") else {
        panic!("S must stay compact");
    };
    assert_eq!(s[0], [bits(1.0_f64), bits(5e-324_f64)]);
}

#[test]
fn nonfinite_compact_values_fail_or_map_through_alike() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let rows = agreed_matrix(engine, [f64::NAN, 1.0], [2.0, 1.0, f64::INFINITY], 0.5);
    let failed = |message: &str| Route::Failed(message.to_string());
    assert_eq!(
        get(&rows, "svd_compact.s"),
        &failed("svd input components must be finite")
    );
    assert_eq!(
        get(&rows, "eig_full.d"),
        &failed("eig input components must be finite")
    );
    // #1986 covers factorizations only; the elementwise matrix functions map
    // nonfinite entries through, as TensorKit's `inv.`/`exp.` do.
    assert_eq!(
        get(&rows, "inv"),
        &compact::<f64>(&[&[f64::NAN, 1.0], &[0.5, 1.0, 0.0]])
    );
    assert!(matches!(get(&rows, "exp"), Route::Compact(_)));
    assert!(matches!(get(&rows, "solve"), Route::Compact(_)));
    // One pinv refusal in both modes: the dense cutoff's typed error.
    assert_eq!(
        get(&rows, "pinv"),
        &failed("pinv singular values must be finite")
    );
}

#[test]
fn pinv_cutoff_compares_unrounded_magnitudes() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // `|1 + 2i| = √5` rounds up in f32; with the cutoff at exactly √5 the
    // unrounded magnitude sits on it and is cut (strict `>`), as are the
    // ones below it.
    let root5 = Complex64::new(1.0, 2.0).norm();
    let one = Complex32::new(1.0, 0.0);
    let rows = agreed_matrix(
        engine,
        [Complex32::new(4.0, 0.0), Complex32::new(1.0, 2.0)],
        [one; 3],
        root5 / 4.0,
    );
    assert_eq!(
        get(&rows, "pinv"),
        &compact::<Complex32>(&[
            &[Complex32::new(0.25, -0.0), Complex32::new(0.0, 0.0)],
            &[Complex32::new(0.0, 0.0); 3]
        ])
    );
}
