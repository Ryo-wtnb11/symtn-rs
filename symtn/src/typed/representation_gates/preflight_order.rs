//! #1995: inv, pinv, exp and solve report the same first error for the same
//! misuse in both fusion modes, in one order — runtime, rule, operand shape
//! (codomain equality, isomorphism, endomorphism), borrowed-view admission,
//! then representation — on multiplicity-free U(1) and SU(2) and on the
//! checked SU(3) rule (fusion-symbols; those rows need `cgc-gen`). The expected first error of every row is the
//! one TensorKit's precedence implies (`src/tensors/linalg.jl` `inv`, `\`,
//! `exp!`; `src/tensors/diagonal.jl` `D \ t`), written out below rather than
//! derived from either mode.

use super::*;
use num_complex::Complex64;
#[cfg(feature = "cgc-gen")]
use symtn_core::LieGroupFusionRule;
use symtn_core::ZNFusionRule;

const RUNTIME: &str = "operands belong to different runtimes";
const RULE: &str = "operands use different fusion rules";
const CODOMAIN: &str = "space mismatch: solve requires equal divisor and right-hand-side codomains";
const NOT_ISO_SOLVE: &str =
    "space mismatch: solve requires an isomorphic divisor codomain and domain";
const NOT_ISO_INV: &str = "space mismatch: inv requires isomorphic codomain and domain";
const NOT_ENDO: &str = "space mismatch: exp requires an endomorphism (codomain == domain)";

fn unsupported_solve() -> String {
    Error::Unsupported {
        operation: "solve",
        alternative: crate::error::Alternative::Materialize,
    }
    .to_string()
}

/// The row text of a result: `ok`, or the error's root cause without the
/// per-mode wrapper (the facade error type is per mode by design, #1862).
fn text<T, E: std::error::Error + 'static>(result: &Result<T, E>) -> String {
    match result {
        Ok(_) => "ok".to_string(),
        Err(error) => super::root_cause(error),
    }
}

/// Runs `f`, returning its value and the adjoint payload materializations it
/// made.
fn materializations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let adjoint = probe::record();
    let value = f();
    (value, adjoint.get(Event::AdjointMaterialization))
}

fn spectrum<R, D>(
    bond: &GradedSpace<R>,
    value: impl Fn(usize) -> D,
) -> Vec<SectorSpectrum<R::Sector, D>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let mut next = 0;
    bond.sectors()
        .unwrap_or_else(|_| panic!("bond sectors"))
        .into_iter()
        .map(|sector| {
            let degeneracy = bond
                .degeneracy(&sector)
                .unwrap_or_else(|_| panic!("bond degeneracy"));
            let values = (0..degeneracy)
                .map(|_| {
                    next += 1;
                    value(next)
                })
                .collect();
            SectorSpectrum { sector, values }
        })
        .collect()
}

/// The misuse matrix of one rule: `v` and `w` are non-isomorphic legs.
macro_rules! misuse {
    ($engine:ident, $v:expr, $w:expr, $dtype:ty, $value:expr) => {{
        let (v, w) = (&$v, &$w);
        let value: fn(f64) -> $dtype = $value;
        let runtime = Runtime::builder().build().unwrap();
        let other = Runtime::builder().build().unwrap();
        let mut rand = |rt: &Runtime, cod: &GradedSpace<_>, dom: &GradedSpace<_>, seed| {
            TensorMap::<_, $dtype>::rand_with_seed($engine, rt, [cod], [dom], seed).unwrap()
        };
        let vv = rand(&runtime, v, v, 1);
        let vw = rand(&runtime, v, w, 2);
        let wv = rand(&runtime, w, v, 3);
        let wv_other = rand(&other, w, v, 4);
        let vw_other = rand(&other, v, w, 5);
        let y = rand(&runtime, v, v, 6);
        let d: TensorMap<_, $dtype> = TensorMap::diagonal(
            $engine,
            &runtime,
            v,
            spectrum(v, |k| value(1.0 + 0.5 * k as f64)),
        )
        .unwrap();
        let singular: TensorMap<_, $dtype> = TensorMap::diagonal(
            $engine,
            &runtime,
            v,
            spectrum(v, |k| {
                <$dtype>::from_real(if k == 2 { 0.0 } else { k as f64 })
            }),
        )
        .unwrap();
        let solve = |$engine: &mut dyn Engine,
                     lhs: &TensorMap<_, $dtype>,
                     rhs: TensorRef<'_, _, $dtype>| {
            text(&lhs.solve($engine, &[0], &[1], rhs, &[0], &[1]))
        };
        let lazy_wv = wv.adjoint($engine).unwrap();
        let (lazy_exp, lazy_exp_copies) = materializations(|| lazy_wv.exp($engine, &[0], &[1]));

        // `D \ y'` reads the view: equal to the owned adjoint, and to the
        // dense LU route on the materialized divisor (an independent oracle).
        let (compact_view, copies) =
            materializations(|| d.solve($engine, &[0], &[1], y.adjoint_view(), &[0], &[1]));
        let y_owned = y.adjoint($engine).unwrap().materialize($engine).unwrap();
        let compact_view_ok = compact_view.as_ref().map(|x| {
            let lu = d
                .materialize($engine)
                .unwrap()
                .solve($engine, &[0], &[1], &y_owned, &[0], &[1])
                .unwrap();
            let owned = d.solve($engine, &[0], &[1], &y_owned, &[0], &[1]).unwrap();
            assert_eq!(x.dense_data().unwrap(), owned.dense_data().unwrap());
            let diff = x
                .axpby(
                    $engine,
                    <$dtype>::from_real(1.0),
                    &lu,
                    <$dtype>::from_real(-1.0),
                )
                .unwrap()
                .norm($engine, 2.0)
                .unwrap();
            assert!(
                diff < 1e-12 * lu.norm($engine, 2.0).unwrap().max(1.0),
                "{diff}"
            );
            copies
        });

        vec![
            ("solve runtime", solve($engine, &vw, (&wv_other).into())),
            (
                "solve runtime borrowed",
                solve($engine, &vw, vw_other.adjoint_view()),
            ),
            ("solve codomain", solve($engine, &vv, (&wv).into())),
            (
                "solve codomain and non-isomorphic",
                solve($engine, &vw, (&wv).into()),
            ),
            (
                "solve codomain borrowed",
                solve($engine, &vv, vw.adjoint_view()),
            ),
            ("solve non-isomorphic", solve($engine, &vw, (&vv).into())),
            (
                "solve non-isomorphic borrowed",
                solve($engine, &vw, y.adjoint_view()),
            ),
            (
                "solve borrowed dense divisor",
                solve($engine, &vv, y.adjoint_view()),
            ),
            (
                "solve borrowed compact divisor",
                match compact_view_ok {
                    Ok(copies) => format!("ok, {copies} copies"),
                    Err(error) => error.to_string(),
                },
            ),
            (
                "solve singular compact divisor",
                solve($engine, &singular, (&y).into()),
            ),
            (
                "solve singular compact divisor borrowed",
                solve($engine, &singular, y.adjoint_view()),
            ),
            ("inv non-isomorphic", text(&vw.inv($engine, &[0], &[1]))),
            (
                "inv non-isomorphic lazy",
                text(&lazy_wv.inv($engine, &[0], &[1])),
            ),
            ("exp non-endomorphic", text(&vw.exp($engine, &[0], &[1]))),
            (
                "exp non-endomorphic lazy",
                format!("{}, {lazy_exp_copies} copies", text(&lazy_exp)),
            ),
            ("pinv rcond", text(&vw.pinv($engine, &[0], &[1], -1.0))),
        ]
    }};
}

fn expected_misuse(singular: &str) -> Vec<(&'static str, String)> {
    vec![
        ("solve runtime", RUNTIME.to_string()),
        ("solve runtime borrowed", RUNTIME.to_string()),
        ("solve codomain", CODOMAIN.to_string()),
        ("solve codomain and non-isomorphic", CODOMAIN.to_string()),
        ("solve codomain borrowed", CODOMAIN.to_string()),
        ("solve non-isomorphic", NOT_ISO_SOLVE.to_string()),
        ("solve non-isomorphic borrowed", NOT_ISO_SOLVE.to_string()),
        ("solve borrowed dense divisor", unsupported_solve()),
        ("solve borrowed compact divisor", "ok, 0 copies".to_string()),
        ("solve singular compact divisor", singular.to_string()),
        (
            "solve singular compact divisor borrowed",
            singular.to_string(),
        ),
        ("inv non-isomorphic", NOT_ISO_INV.to_string()),
        ("inv non-isomorphic lazy", NOT_ISO_INV.to_string()),
        ("exp non-endomorphic", NOT_ENDO.to_string()),
        ("exp non-endomorphic lazy", format!("{NOT_ENDO}, 0 copies")),
        (
            "pinv rcond",
            "invalid argument: pinv rcond must be finite and non-negative".to_string(),
        ),
    ]
}

fn differing(name: &str, actual: &[(&str, String)], expected: &[(&str, String)]) -> Vec<String> {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .filter(|((_, actual), (_, expected))| actual != expected)
        .map(|((op, actual), (_, expected))| {
            format!("{name}: {op}: got `{actual}`, want `{expected}`")
        })
        .collect()
}

#[test]
fn misuse_reports_one_first_error_in_both_modes() {
    let engine = &mut crate::typed::CpuEngine::serial();
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
    #[cfg(feature = "cgc-gen")]
    let su3 =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    #[cfg(feature = "cgc-gen")]
    let su3_leg = |pairs: &[(&[i64], usize)]| {
        GradedSpace::try_new(
            Arc::clone(&su3),
            pairs.iter().map(|&(l, d)| (l.to_vec(), d)),
        )
        .unwrap()
    };
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.5 - x);

    // The singular-divisor error is the dense route's own, in every mode;
    // take its text from one row and require every rule and dtype to agree.
    let reference = misuse!(
        engine,
        u1_leg(&[(0, 2), (1, 3)]),
        u1_leg(&[(0, 2)]),
        f64,
        real
    );
    let singular = reference
        .iter()
        .find(|(op, _)| *op == "solve singular compact divisor")
        .unwrap()
        .1
        .clone();
    assert_ne!(singular, "ok");
    let expected = expected_misuse(&singular);
    let mut differing_rows = differing("U1 f64", &reference, &expected);
    let mut assert_rows = |name: &str, actual: &[(&str, String)], expected: &[(&str, String)]| {
        differing_rows.extend(differing(name, actual, expected));
    };
    assert_rows(
        "U1 c64",
        &misuse!(
            engine,
            u1_leg(&[(0, 2), (1, 3)]),
            u1_leg(&[(0, 2)]),
            Complex64,
            complex
        ),
        &expected,
    );
    assert_rows(
        "SU2 f64",
        &misuse!(
            engine,
            su2_leg(&[(0, 2), (1, 3)]),
            su2_leg(&[(0, 2)]),
            f64,
            real
        ),
        &expected,
    );
    assert_rows(
        "SU2 c64",
        &misuse!(
            engine,
            su2_leg(&[(0, 2), (1, 3)]),
            su2_leg(&[(0, 2)]),
            Complex64,
            complex
        ),
        &expected,
    );
    #[cfg(feature = "cgc-gen")]
    assert_rows(
        "SU3 checked f64",
        &misuse!(
            engine,
            su3_leg(&[(&[0, 0], 2), (&[1, 0], 3)]),
            su3_leg(&[(&[0, 0], 2)]),
            f64,
            real
        ),
        &expected,
    );
    #[cfg(feature = "cgc-gen")]
    assert_rows(
        "SU3 checked c64",
        &misuse!(
            engine,
            su3_leg(&[(&[0, 0], 2), (&[1, 0], 3)]),
            su3_leg(&[(&[0, 0], 2)]),
            Complex64,
            complex
        ),
        &expected,
    );
    assert!(differing_rows.is_empty(), "{}", differing_rows.join("\n"));
}

/// A rule mismatch precedes every later check, including a borrowed rhs and
/// one whose moved roles would copy it.
macro_rules! rule_rows {
    ($engine:ident, $divisor_leg:expr, $rhs_leg:expr, $dtype:ty) => {{
        let (a, b) = (&$divisor_leg, &$rhs_leg);
        let runtime = Runtime::builder().build().unwrap();
        let other = Runtime::builder().build().unwrap();
        let divisor =
            TensorMap::<_, $dtype>::rand_with_seed($engine, &runtime, [a], [a], 1).unwrap();
        let rhs = TensorMap::<_, $dtype>::rand_with_seed($engine, &runtime, [b], [b], 2).unwrap();
        let rhs_other =
            TensorMap::<_, $dtype>::rand_with_seed($engine, &other, [b], [b], 3).unwrap();
        vec![
            text(&divisor.solve($engine, &[0], &[1], &rhs, &[0], &[1])),
            text(&divisor.solve($engine, &[0], &[1], rhs.adjoint_view(), &[0], &[1])),
            text(&divisor.solve($engine, &[1], &[0], rhs.adjoint_view(), &[1], &[0])),
            // Malformed roles: the rule still precedes the permute's own error,
            // for an owned and a borrowed rhs alike.
            text(&divisor.solve($engine, &[5], &[0], &rhs, &[0], &[1])),
            text(&divisor.solve($engine, &[5], &[0], rhs.adjoint_view(), &[0], &[1])),
            // Runtime precedes rule.
            text(&divisor.solve($engine, &[0], &[1], &rhs_other, &[0], &[1])),
        ]
    }};
}

#[test]
fn rule_mismatch_is_the_first_error_in_both_modes() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let za = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2), (z2.irrep(1), 1)]).unwrap();
    let zb = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)]).unwrap();
    let mut expected = vec![RULE.to_string(); 5];
    expected.push(RUNTIME.to_string());
    assert_eq!(rule_rows!(engine, za, zb, f64), expected, "Z2/Z3 f64");
    assert_eq!(rule_rows!(engine, za, zb, Complex64), expected, "Z2/Z3 c64");
    #[cfg(feature = "cgc-gen")]
    {
        let su3 = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        let su4 = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(4).unwrap()).unwrap(),
        );
        let sa =
            GradedSpace::try_new(Arc::clone(&su3), [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
        let sb = GradedSpace::try_new(Arc::clone(&su4), [(vec![0, 0, 0], 1)]).unwrap();
        assert_eq!(rule_rows!(engine, sa, sb, f64), expected, "SU3/SU4 f64");
        assert_eq!(
            rule_rows!(engine, sa, sb, Complex64),
            expected,
            "SU3/SU4 c64"
        );
    }
}

/// `space` re-laid as a padded (expert) layout of its own hom space, with
/// the payload offset of every block.
fn padded(
    space: &BoundDynamicFusionMapSpace<U1FusionRule>,
) -> (BoundDynamicFusionMapSpace<U1FusionRule>, Vec<usize>) {
    let structure = space.space().structure();
    let (mut offset, mut blocks, mut offsets) = (1, Vec::new(), Vec::new());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        blocks.push(
            BlockSpec::column_major_with_key(block.key().clone(), block.shape().to_vec(), offset)
                .unwrap(),
        );
        offsets.push(offset);
        offset += block.shape().iter().product::<usize>() + 1;
    }
    let layout = DynamicFusionMapSpace::try_from_layout(
        space.space().homspace().clone(),
        BlockStructure::from_blocks_with_rank(2, blocks).unwrap(),
    )
    .unwrap()
    .try_bind_rule(space.provider())
    .unwrap();
    let expert = BoundDynamicFusionMapSpace::bind_multiplicity_free_checked(
        layout.clone(),
        Arc::clone(space.provider_arc()),
    )
    .unwrap();
    assert_ne!(expert.space().structure(), structure);
    (expert, offsets)
}

/// The two layout side effects of one compact-divisor solve (#1995): `D \ t`
/// lands on the canonical output space whatever `t`'s layout, scaled along
/// its leading axis (TensorKit `similar(D, domain(D) ← domain(t))`), and
/// `D \ D'` stays compact on `D`'s own space (`d1.domain`), expert or not.
#[test]
fn compact_divisor_solve_lands_on_the_documented_spaces() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    // Row `a` of every sector divides by `1 + a`.
    let by_row = |bond: &GradedSpace<U1FusionRule>, scale: f64| {
        bond.sectors()
            .unwrap()
            .into_iter()
            .map(|sector| SectorSpectrum {
                values: (0..bond.degeneracy(&sector).unwrap())
                    .map(|a| scale * (1.0 + a as f64))
                    .collect(),
                sector,
            })
            .collect::<Vec<_>>()
    };
    let d: TensorMap<_, f64> = TensorMap::diagonal(engine, &runtime, &v, by_row(&v, 1.0)).unwrap();
    let canonical = owned(&TensorMap::<_, f64>::zeros(engine, &runtime, [&v], [&v]).unwrap())
        .space
        .clone();

    // An owned dense rhs in a padded layout; values name their block and cell.
    let (expert, offsets) = padded(&canonical);
    let structure = expert.space().structure();
    let mut data = vec![f64::NAN; expert.space().required_len().unwrap()];
    for (index, &offset) in offsets.iter().enumerate() {
        let shape = structure.block(index).unwrap().shape().to_vec();
        for b in 0..shape[1] {
            for a in 0..shape[0] {
                data[offset + a + shape[0] * b] = (100 * index + 10 * a + b) as f64 + 1.0;
            }
        }
    }
    let rhs = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(expert.clone(), data)),
    };
    let solved = d.solve(engine, &[0], &[1], &rhs, &[0], &[1]).unwrap();
    let body = owned(&solved);
    assert_eq!(
        body.space.space().structure(),
        canonical.space().structure()
    );
    let TypedData::Dense(values) = body.data.as_ref() else {
        panic!("D \\ t must be dense");
    };
    let output = body.space.space().structure();
    for index in 0..structure.block_count() {
        let source = structure.block(index).unwrap();
        let target = (0..output.block_count())
            .map(|j| output.block(j).unwrap())
            .find(|block| block.key() == source.key())
            .unwrap();
        for b in 0..source.shape()[1] {
            for a in 0..source.shape()[0] {
                // The reciprocal spectrum scales the row (`inv_spectrum`).
                let expected = ((100 * index + 10 * a + b) as f64 + 1.0) * (1.0 / (1.0 + a as f64));
                let at = target.offset() + a * target.strides()[0] + b * target.strides()[1];
                assert_eq!(values[at], expected, "block {index} ({a}, {b})");
            }
        }
    }

    // `D \ D'` with `D` on the padded bond layout.
    let d_expert = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(
            expert.clone(),
            TensorMap::<_, f64>::diagonal(engine, &runtime, &v, by_row(&v, 2.0))
                .unwrap()
                .spectrum()
                .unwrap()
                .to_vec(),
        )),
    };
    let quotient = d_expert.solve(engine, &[0], &[1], &d, &[0], &[1]).unwrap();
    let body = owned(&quotient);
    assert_eq!(body.space.space().structure(), expert.space().structure());
    let TypedData::Diagonal(spectrum) = body.data.as_ref() else {
        panic!("D \\ D' must stay compact");
    };
    assert!(spectrum
        .iter()
        .all(|entry| entry.values.iter().all(|&value| value == 0.5)));
}

/// Neither a square stored-sector intersection nor an equal total dimension
/// substitutes for the coupled-sector isomorphism the preflight checks;
/// the `_into` seams assume an admitted operand.
#[test]
fn isomorphism_is_checked_per_coupled_sector() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = Arc::new(U1FusionRule);
    let leg = |pairs: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&u1),
            pairs.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
        )
        .unwrap()
    };
    for (codomain, domain) in [
        (leg(&[(0, 1), (1, 1)]), leg(&[(0, 1)])),
        (leg(&[(0, 1), (1, 1)]), leg(&[(0, 1), (2, 1)])),
    ] {
        let t = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&codomain], [&domain], 1)
            .unwrap();
        let rhs =
            TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&codomain], [&codomain], 2)
                .unwrap();
        assert_eq!(text(&t.inv(engine, &[0], &[1])), NOT_ISO_INV);
        assert_eq!(
            text(&t.solve(engine, &[0], &[1], &rhs, &[0], &[1])),
            NOT_ISO_SOLVE
        );
    }
}
