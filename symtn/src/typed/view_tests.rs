//! #1546: an operation given `t.adjoint_view()` equals the same operation
//! given the owned lazy adjoint `&t.adjoint()?`, and either consumes it with
//! zero entries into adjoint materialization or refuses it with
//! `Unsupported { Materialize }`.
//!
//! The oracle is the owned-lazy-adjoint call, which reaches the operation's
//! existing body without the view; results are compared bit for bit because
//! both run the same code on the same parent payload.

use std::sync::Arc;

use num_complex::Complex64;
use symtn_core::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};

use super::probe::{self, Event, RoutePolicy};
use super::{ContractSpec, GradedSpace, Side, TensorMap};
use crate::error::{Alternative, Error};
use crate::runtime::Runtime;

fn runtime() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

/// Runs `f` and returns its value with the number of entries into adjoint
/// payload materialization it made.
fn adjoint_entries<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let adjoint = probe::record();
    let value = f();
    (value, adjoint.get(Event::AdjointMaterialization))
}

/// Forces the declined adjoint-oriented cat plan.
const DECLINE_ORIENTED_CAT: RoutePolicy = RoutePolicy {
    cat_declines_oriented: true,
    ..RoutePolicy::DEFAULT
};

/// The refusal names the operation and the `materialize` remedy.
fn is_unsupported(error: &impl std::fmt::Display, operation: &'static str) -> bool {
    let expected = Error::Unsupported {
        operation,
        alternative: Alternative::Materialize,
    }
    .to_string();
    assert!(expected.contains(operation) && expected.contains("materialize"));
    error.to_string() == expected
}

macro_rules! assert_same_tensor {
    ($actual:expr, $expected:expr, $what:expr) => {{
        let (actual, expected) = (&$actual, &$expected);
        assert_eq!(
            actual.codomain(),
            expected.codomain(),
            "{}: codomain",
            $what
        );
        assert_eq!(actual.domain(), expected.domain(), "{}: domain", $what);
        assert_eq!(
            actual.materialize().unwrap().dense_data().unwrap(),
            expected.materialize().unwrap().dense_data().unwrap(),
            "{}: payload",
            $what
        );
    }};
}

/// A direct-consume operation: same result as the owned lazy adjoint, zero
/// materialization entries.
macro_rules! direct {
    ($what:expr, $view_call:expr, $owned_call:expr) => {{
        let (view, entries) = adjoint_entries(|| $view_call);
        assert_eq!(entries, 0, "{}: view entered materialization", $what);
        let view = view.unwrap_or_else(|error| panic!("{}: view: {error}", $what));
        let owned = $owned_call.unwrap_or_else(|error| panic!("{}: owned: {error}", $what));
        (view, owned)
    }};
}

/// A refused operation: `Unsupported { Materialize }`, zero entries; the
/// owned lazy adjoint still materializes implicitly (positive control), and
/// the remedy gives the owned-lazy result.
macro_rules! refused {
    ($what:expr, $view_call:expr, $owned_call:expr, $remedy_call:expr) => {{
        let (view, entries) = adjoint_entries(|| $view_call);
        assert_eq!(
            entries, 0,
            "{}: refused view entered materialization",
            $what
        );
        let error = view.err().expect(concat!($what, ": view must be refused"));
        assert!(is_unsupported(&error, $what), "{}: {error}", $what);
        let (owned, owned_entries) = adjoint_entries(|| $owned_call);
        assert!(owned_entries >= 1, "{}: positive control", $what);
        assert_same_tensor!($remedy_call.unwrap(), owned.unwrap(), $what);
    }};
}

/// Operations every provider mode shares. `a: [V, W] <- [V]`,
/// `b: [V] <- [V, W]`, and `a` is built in `b'`'s layout.
macro_rules! shared_suite {
    ($a:expr, $b:expr, $s:expr, $alpha:expr, $beta:expr, $what:expr) => {{
        let (a, b, s, what) = (&$a, &$b, &$s, $what);
        let lazy = b.adjoint().unwrap();

        let (view, owned) = direct!("inner", a.inner(b.adjoint_view()), a.inner(&lazy));
        assert_eq!(view, owned, "{what}: inner");

        let (view, owned) = direct!(
            "axpby",
            a.axpby($alpha, b.adjoint_view(), $beta),
            a.axpby($alpha, &lazy, $beta)
        );
        assert_same_tensor!(view, owned, format!("{what}: axpby"));

        let mut view = a.materialize().unwrap();
        let mut owned = a.materialize().unwrap();
        let (result, entries) = adjoint_entries(|| lazy.axpby_into(&mut view, $beta, $alpha));
        assert_eq!(entries, 0, "{what}: axpby_into");
        result.unwrap();
        lazy.materialize()
            .unwrap()
            .axpby_into(&mut owned, $beta, $alpha)
            .unwrap();
        assert_same_tensor!(view, owned, format!("{what}: axpby_into"));

        let (view, owned) = direct!(
            "cat",
            a.cat(b.adjoint_view(), Side::Domain),
            a.cat(&lazy, Side::Domain)
        );
        assert_same_tensor!(view, owned, format!("{what}: cat"));

        refused!(
            "solve",
            s.solve(&[0, 1], &[2, 3], b.adjoint_view(), &[0, 1], &[2]),
            s.solve(&[0, 1], &[2, 3], &lazy, &[0, 1], &[2]),
            s.solve(&[0, 1], &[2, 3], &lazy.materialize().unwrap(), &[0, 1], &[2])
        );
        refused!(
            "otimes",
            a.otimes(b.adjoint_view()),
            a.otimes(&lazy),
            a.otimes(&lazy.materialize().unwrap())
        );
        refused!(
            "absorb",
            a.absorb(b.adjoint_view()),
            a.absorb(&lazy),
            a.absorb(&lazy.materialize().unwrap())
        );

        // Refusal precedes the work these operations do on an owned lazy
        // receiver: no entry into materialization at all.
        let a_lazy = a.adjoint().unwrap();
        let (view, entries) = adjoint_entries(|| a_lazy.otimes(b.adjoint_view()));
        assert_eq!(entries, 0, "{what}: otimes on a lazy receiver");
        assert!(is_unsupported(&view.err().unwrap(), "otimes"));
        let s_lazy = s.adjoint().unwrap();
        let (view, entries) = adjoint_entries(|| s_lazy.solve(&[0, 1], &[2, 3], b.adjoint_view(), &[0, 1], &[2]));
        assert_eq!(entries, 0, "{what}: solve on a lazy receiver");
        assert!(is_unsupported(&view.err().unwrap(), "solve"));

        // The adjoint view of a lazy adjoint is its owned parent.
        let (view, owned) = direct!("inner of parent", b.inner(lazy.adjoint_view()), b.inner(b));
        assert_eq!(view, owned, "{what}: inner of parent");
    }};
}

/// Multiplicity-free: contraction and composition consume the view.
macro_rules! mf_suite {
    ($a:expr, $b:expr, $s:expr, $alpha:expr, $beta:expr, $what:expr) => {{
        shared_suite!($a, $b, $s, $alpha, $beta, $what);
        let (a, b, what) = (&$a, &$b, $what);
        let lazy = b.adjoint().unwrap();

        let (view, owned) = direct!(
            "contract",
            a.contract(
                b.adjoint_view(),
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3]
                }
            ),
            a.contract(
                &lazy,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3]
                }
            )
        );
        assert_same_tensor!(view, owned, format!("{what}: contract"));

        let mut destination = owned.zeros_like().unwrap();
        let mut expected = owned.zeros_like().unwrap();
        let (result, entries) = adjoint_entries(|| {
            a.contract_into(
                b.adjoint_view(),
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
                &mut destination,
                $alpha,
                $alpha * 0.0,
            )
        });
        assert_eq!(entries, 0, "{what}: contract_into");
        result.unwrap();
        a.contract_into(
            &lazy,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
            &mut expected,
            $alpha,
            $alpha * 0.0,
        )
        .unwrap();
        assert_same_tensor!(destination, expected, format!("{what}: contract_into"));

        let (view, owned) = direct!("compose", b.compose(b.adjoint_view()), b.compose(&lazy));
        assert_same_tensor!(view, owned, format!("{what}: compose"));
    }};
}

macro_rules! mf_case {
    ($v:expr, $w:expr, $dtype:ty, $alpha:expr, $beta:expr, $what:expr) => {{
        let runtime = runtime();
        let (v, w) = ($v, $w);
        let a: TensorMap<_, $dtype> =
            // Built as an adjoint so that it shares the lazy adjoint's layout.
            TensorMap::rand_with_seed(&runtime, [&v], [&v, &w], 1).unwrap().adjoint().unwrap().materialize().unwrap();
        let b: TensorMap<_, $dtype> =
            TensorMap::rand_with_seed(&runtime, [&v], [&v, &w], 2).unwrap();
        let s: TensorMap<_, $dtype> =
            TensorMap::rand_with_seed(&runtime, [&v, &w], [&v, &w], 3).unwrap();
        mf_suite!(a, b, s, $alpha, $beta, $what);
    }};
}

fn u1(pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        pairs.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
    )
    .unwrap()
}

fn su2(pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        pairs
            .iter()
            .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
    )
    .unwrap()
}

const RE: (f64, f64) = (0.5, -1.25);
const CX: (Complex64, Complex64) = (Complex64::new(0.5, 0.25), Complex64::new(-1.25, 0.75));

#[test]
fn u1_view_matches_owned_lazy_adjoint() {
    let v = || u1(&[(-1, 2), (0, 1), (1, 3)]);
    let w = || u1(&[(0, 2), (1, 1)]).try_dual().unwrap();
    mf_case!(v(), w(), f64, RE.0, RE.1, "U1 f64");
    mf_case!(v(), w(), Complex64, CX.0, CX.1, "U1 c64");
}

#[test]
fn su2_view_matches_owned_lazy_adjoint() {
    let v = || su2(&[(0, 2), (1, 1), (2, 2)]);
    let w = || su2(&[(1, 2), (2, 1)]);
    mf_case!(v(), w(), f64, RE.0, RE.1, "SU2 f64");
    mf_case!(v(), w(), Complex64, CX.0, CX.1, "SU2 c64");
}

#[test]
fn fermionic_z2_times_u1_view_matches_owned_lazy_adjoint() {
    let leg = |pairs: &[(bool, i32, usize)]| {
        GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            pairs.iter().map(|&(odd, q, d)| {
                let parity = if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN };
                (product_sector(parity, U1Irrep::new(q)), d)
            }),
        )
        .unwrap()
    };
    let v = || leg(&[(false, 0, 2), (true, 1, 1), (true, -1, 2), (false, 2, 1)]);
    let w = || leg(&[(true, 1, 2), (false, 0, 1)]).try_dual().unwrap();
    mf_case!(v(), w(), f64, RE.0, RE.1, "fZ2xU1 f64");
    mf_case!(v(), w(), Complex64, CX.0, CX.1, "fZ2xU1 c64");
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_su3_view_matches_owned_lazy_adjoint() {
    use symtn_core::SUNFusionRule;

    macro_rules! su3_case {
        ($dtype:ty, $alpha:expr, $beta:expr, $what:expr) => {{
            let runtime = runtime();
            let provider = Arc::new(SUNFusionRule::new(3).unwrap());
            // 8 ⊗ 8 → 8 has multiplicity 2.
            let v = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
            let w = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1), (vec![1, 0], 1)])
                .unwrap();
            let a: TensorMap<_, $dtype> =
                // Built as an adjoint so that it shares the lazy adjoint's layout.
            TensorMap::rand_with_seed(&runtime, [&v], [&v, &w], 1).unwrap().adjoint().unwrap().materialize().unwrap();
            let b: TensorMap<_, $dtype> =
                TensorMap::rand_with_seed(&runtime, [&v], [&v, &w], 2).unwrap();
            let s: TensorMap<_, $dtype> =
                TensorMap::rand_with_seed(&runtime, [&v, &w], [&v, &w], 3).unwrap();
            shared_suite!(a, b, s, $alpha, $beta, $what);

            // Contraction, its `_into` (#1870) and composition consume
            // the view as the owned lazy adjoint (#1865).
            let lazy = b.adjoint().unwrap();
            let spec = ContractSpec { lhs: &[2], rhs: &[0], codomain: &[0, 1], domain: &[2, 3] };
            let (view, owned) =
                direct!("contract", a.contract(b.adjoint_view(), &spec), a.contract(&lazy, &spec));
            assert_same_tensor!(view, owned, format!("{}: contract", $what));
            let mut destination = owned.scale($alpha * 0.0).unwrap();
            let mut expected = owned.scale($alpha * 0.0).unwrap();
            let (result, entries) = adjoint_entries(|| {
                a.contract_into(b.adjoint_view(), &spec, &mut destination, $alpha, $alpha * 0.0)
            });
            assert_eq!(entries, 0, "{}: contract_into", $what);
            result.unwrap();
            a.contract_into(&lazy, &spec, &mut expected, $alpha, $alpha * 0.0).unwrap();
            assert_same_tensor!(destination, expected, format!("{}: contract_into", $what));
            let (view, owned) = direct!("compose", b.compose(b.adjoint_view()), b.compose(&lazy));
            assert_same_tensor!(view, owned, format!("{}: compose", $what));
        }};
    }
    su3_case!(f64, RE.0, RE.1, "SU3 f64");
    su3_case!(Complex64, CX.0, CX.1, "SU3 c64");
}

#[test]
fn deligne_product_refuses_an_adjoint_view() {
    let runtime = runtime();
    let v = u1(&[(0, 2), (1, 1)]);
    let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 1).unwrap();
    let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v, &v], [&v], 2).unwrap();
    let product = Arc::new(U1FusionRule.product(U1FusionRule));
    let lazy = b.adjoint().unwrap();
    refused!(
        "deligne_product",
        a.deligne_product(b.adjoint_view(), Arc::clone(&product)),
        a.deligne_product(&lazy, Arc::clone(&product)),
        a.deligne_product(&lazy.materialize().unwrap(), Arc::clone(&product))
    );
    // The left operand is not committed before the refusal.
    let a_lazy = a.adjoint().unwrap();
    let (view, entries) = adjoint_entries(|| a_lazy.deligne_product(b.adjoint_view(), product));
    assert_eq!(entries, 0);
    assert!(is_unsupported(&view.err().unwrap(), "deligne_product"));
}

#[test]
fn otimes_materializes_a_lazy_operand_and_a_plain_view_is_the_tensor_itself() {
    let runtime = runtime();
    let v = u1(&[(0, 2), (1, 1)]);
    let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v, &v], 1).unwrap();
    let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v, &v], [&v], 2).unwrap();
    let lazy = b.adjoint().unwrap();
    // `&lazy` is not marked: `otimes` materializes an owned lazy adjoint
    // operation-locally, as its cost section documents.
    let (result, entries) = adjoint_entries(|| a.otimes(&lazy));
    assert!(entries >= 1);
    result.unwrap();
    let view: super::TensorRef<'_, _, f64> = (&a).into();
    let copied = view;
    assert_eq!(a.inner(view).unwrap(), a.inner(&a).unwrap());
    assert_eq!(a.inner(copied).unwrap(), a.inner(&a).unwrap());
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// Device operations that take the view. Multiplicity-free only: the
    /// device facade has no checked-Generic operations.
    macro_rules! cuda_case {
        ($v:expr, $w:expr, $dtype:ty, $alpha:expr, $beta:expr, $what:expr) => {{
            let what = $what;
            let runtime = Runtime::builder().cuda(0).build().unwrap();
            let (v, w) = ($v, $w);
            let host = |seed| {
                TensorMap::<_, $dtype>::rand_with_seed(&runtime, [&v], [&v, &w], seed).unwrap()
            };
            let b = host(2).to_cuda().unwrap();
            let c = host(1).to_cuda().unwrap();
            let lazy = b.adjoint().unwrap();
            // `a` is owned in `b'`'s layout; `a_lazy` is a lazy adjoint.
            let a = c.adjoint().unwrap().materialize().unwrap();
            let a_lazy = c.adjoint().unwrap();

            let (view, owned) = direct!(
                "cuda contract",
                a.contract(
                    b.adjoint_view(),
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3]
                    }
                ),
                a.contract(
                    &lazy,
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3]
                    }
                )
            );
            assert_same_tensor!(
                view.to_host().unwrap(),
                owned.to_host().unwrap(),
                format!("{what}: contract")
            );

            let mut destination = owned.zeros_like().unwrap();
            let mut expected = owned.zeros_like().unwrap();
            let one = <$dtype>::from(1.0);
            let (result, entries) = adjoint_entries(|| {
                a.contract_into(
                    b.adjoint_view(),
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                    &mut destination,
                    one,
                    one * 0.0,
                )
            });
            assert_eq!(entries, 0, "{what}: contract_into");
            result.unwrap();
            a.contract_into(
                &lazy,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
                &mut expected,
                one,
                one * 0.0,
            )
            .unwrap();
            assert_same_tensor!(
                destination.to_host().unwrap(),
                expected.to_host().unwrap(),
                format!("{what}: contract_into")
            );

            let (view, owned) = direct!(
                "cuda compose",
                b.compose(b.adjoint_view()),
                b.compose(&lazy)
            );
            assert_same_tensor!(
                view.to_host().unwrap(),
                owned.to_host().unwrap(),
                format!("{what}: compose")
            );

            let (view, owned) = direct!(
                "cuda axpby (both lazy)",
                a_lazy.axpby($alpha, b.adjoint_view(), $beta),
                a_lazy.axpby($alpha, &lazy, $beta)
            );
            assert_same_tensor!(
                view.to_host().unwrap(),
                owned.to_host().unwrap(),
                format!("{what}: axpby")
            );

            let (view, owned) = direct!(
                "cuda axpby (mixed)",
                a.axpby($alpha, b.adjoint_view(), $beta),
                a.axpby($alpha, &lazy, $beta)
            );
            assert_same_tensor!(
                view.to_host().unwrap(),
                owned.to_host().unwrap(),
                format!("{what}: mixed axpby")
            );

            // `inner` still rejects a lazy operand on the device.
            let (view, entries) = adjoint_entries(|| a.inner(b.adjoint_view()));
            assert_eq!(entries, 0);
            assert_eq!(
                view.err().unwrap(),
                a.inner(&lazy).err().unwrap(),
                "{what}: inner"
            );

            // Positive control for the device probe.
            let (_, entries) = adjoint_entries(|| lazy.materialize().unwrap());
            assert_eq!(entries, 1, "{what}: device probe");
        }};
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_view_matches_owned_lazy_adjoint() {
        let v = || u1(&[(-1, 2), (0, 1), (1, 3)]);
        let w = || u1(&[(0, 2), (1, 1)]).try_dual().unwrap();
        cuda_case!(v(), w(), f64, RE.0, RE.1, "CUDA U1 f64");
        cuda_case!(v(), w(), Complex64, CX.0, CX.1, "CUDA U1 c64");
        let v = || su2(&[(0, 2), (1, 1), (2, 2)]);
        let w = || su2(&[(1, 2), (2, 1)]);
        cuda_case!(v(), w(), f64, RE.0, RE.1, "CUDA SU2 f64");
        cuda_case!(v(), w(), Complex64, CX.0, CX.1, "CUDA SU2 c64");
        let leg = |pairs: &[(bool, i32, usize)]| {
            GradedSpace::try_new(
                Arc::new(FermionParityFusionRule.product(U1FusionRule)),
                pairs.iter().map(|&(odd, q, d)| {
                    let parity = if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN };
                    (product_sector(parity, U1Irrep::new(q)), d)
                }),
            )
            .unwrap()
        };
        let v = || leg(&[(false, 0, 2), (true, 1, 1), (true, -1, 2), (false, 2, 1)]);
        let w = || leg(&[(true, 1, 2), (false, 0, 1)]).try_dual().unwrap();
        cuda_case!(v(), w(), f64, RE.0, RE.1, "CUDA fZ2xU1 f64");
        cuda_case!(v(), w(), Complex64, CX.0, CX.1, "CUDA fZ2xU1 c64");
    }
}

/// A compact diagonal `D` meets an adjoint view: every pairing reads the view
/// in place (`scaled_axis`, compact `solve` through `compose`, compact
/// `axpby`/`inner`), bit-identical to `&x.adjoint()?`.
macro_rules! compact_suite {
    ($bond:expr, $v:expr, $dtype:ty, $value:expr, $alpha:expr, $beta:expr, $what:expr) => {{
        let (bond, v, what) = ($bond, $v, $what);
        let value: fn(f64) -> $dtype = $value;
        let runtime = runtime();
        let spectra = bond
            .sectors()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, sector)| super::SectorSpectrum {
                values: (0..bond.degeneracy(&sector).unwrap())
                    .map(|k| value(1.5 + i as f64 + 0.25 * k as f64))
                    .collect(),
                sector,
            })
            .collect::<Vec<_>>();
        let d: TensorMap<_, $dtype> = TensorMap::diagonal(&runtime, &bond, spectra).unwrap();
        // x: [v] <- [bond], so x': [bond] <- [v].
        let x: TensorMap<_, $dtype> =
            TensorMap::rand_with_seed(&runtime, [&v], [&bond], 5).unwrap();
        // y: [bond] <- [v], so y': [v] <- [bond].
        let y: TensorMap<_, $dtype> =
            TensorMap::rand_with_seed(&runtime, [&bond], [&v], 6).unwrap();
        let x_lazy = x.adjoint().unwrap();
        let y_lazy = y.adjoint().unwrap();
        let d_adjoint = d.adjoint().unwrap();

        let (view, owned) = direct!(
            "D.compose(x')",
            d.compose(x.adjoint_view()),
            d.compose(&x_lazy)
        );
        assert_same_tensor!(view, owned, format!("{what}: D.compose(x')"));
        let (view, owned) = direct!(
            "y'.compose(D')",
            y_lazy.compose(d.adjoint_view()),
            y_lazy.compose(&d_adjoint)
        );
        assert_same_tensor!(view, owned, format!("{what}: y'.compose(D')"));
        let (view, owned) = direct!(
            "x.compose(D')",
            x.compose(d.adjoint_view()),
            x.compose(&d_adjoint)
        );
        assert_same_tensor!(view, owned, format!("{what}: x.compose(D')"));
        let (view, owned) = direct!(
            "D.contract(x')",
            d.contract(
                x.adjoint_view(),
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            ),
            d.contract(
                &x_lazy,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
        );
        assert_same_tensor!(view, owned, format!("{what}: D.contract(x')"));
        let (view, owned) = direct!(
            "compact solve",
            d.solve(&[0], &[1], x.adjoint_view(), &[0], &[1]),
            d.solve(&[0], &[1], &x_lazy, &[0], &[1])
        );
        assert_same_tensor!(view, owned, format!("{what}: compact solve"));

        // e: [bond] <- [bond] dense, so e' has D's hom space.
        let e: TensorMap<_, $dtype> =
            TensorMap::rand_with_seed(&runtime, [&bond], [&bond], 7).unwrap();
        let e_lazy = e.adjoint().unwrap();
        let (view, owned) = direct!(
            "compact axpby",
            d.axpby($alpha, e.adjoint_view(), $beta),
            d.axpby($alpha, &e_lazy, $beta)
        );
        assert_same_tensor!(view, owned, format!("{what}: compact axpby"));
        let (view, owned) = direct!("compact inner", d.inner(e.adjoint_view()), d.inner(&e_lazy));
        assert_eq!(view, owned, "{what}: compact inner");
    }};
}

#[test]
fn compact_diagonal_partners_read_the_view_in_place() {
    let bond = || u1(&[(-1, 2), (0, 3), (1, 1)]);
    let v = || u1(&[(-1, 1), (0, 2), (1, 2)]);
    compact_suite!(bond(), v(), f64, |x| x, RE.0, RE.1, "U1 f64");
    compact_suite!(
        bond(),
        v(),
        Complex64,
        |x| Complex64::new(x, 0.5 - x),
        CX.0,
        CX.1,
        "U1 c64"
    );
    let bond = || su2(&[(0, 2), (1, 1), (2, 2)]);
    let v = || su2(&[(0, 1), (1, 2), (2, 1)]);
    compact_suite!(bond(), v(), f64, |x| x, RE.0, RE.1, "SU2 f64");
    compact_suite!(
        bond(),
        v(),
        Complex64,
        |x| Complex64::new(x, 0.5 - x),
        CX.0,
        CX.1,
        "SU2 c64"
    );
}

/// `cat` refuses an adjoint view only when its copy plan declines, which no
/// public geometry is known to reach; the test hook forces the decline.
#[test]
fn cat_refuses_an_adjoint_view_when_its_plan_declines() {
    let runtime = runtime();
    let v = u1(&[(-1, 2), (0, 1), (1, 3)]);
    let w = u1(&[(0, 2), (1, 1)]);
    let a: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&v, &v], [&w], 1).unwrap();
    let b: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&w], [&v, &v], 2).unwrap();
    let lazy = b.adjoint().unwrap();
    let fast = a.cat(&lazy, Side::Domain).unwrap();
    probe::with_routes(DECLINE_ORIENTED_CAT, || {
        refused!(
            "cat",
            a.cat(b.adjoint_view(), Side::Domain),
            a.cat(&lazy, Side::Domain),
            a.cat(&lazy.materialize().unwrap(), Side::Domain)
        );
        // The declined fallback of the owned lazy adjoint equals the plan result.
        assert_same_tensor!(a.cat(&lazy, Side::Domain).unwrap(), fast, "cat fallback");
    });
}

/// The declined-plan fallback materializes a lazy adjoint while reading a
/// compact diagonal directly. The result is bitwise the fast-plan result.
#[test]
fn cat_fallback_reads_compact_operand_without_densifying() {
    let runtime = runtime();
    let bond = u1(&[(-1, 2), (0, 1), (1, 3)]);
    let spectra = bond
        .sectors()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, sector)| super::SectorSpectrum {
            values: (0..bond.degeneracy(&sector).unwrap())
                .map(|k| 1.25 + i as f64 + 0.5 * k as f64)
                .collect(),
            sector,
        })
        .collect::<Vec<_>>();
    let d: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &bond, spectra).unwrap();
    let x: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&bond], [&bond], 7).unwrap();
    let y: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&bond], [&bond], 8).unwrap();
    let lazy = y.adjoint().unwrap();
    let eager = lazy.materialize().unwrap();
    let bits = |t: &TensorMap<_, f64>| {
        t.dense_data()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    for side in [Side::Domain, Side::Codomain] {
        for (name, lhs, rhs, compact) in [
            ("d|lazy", &d, &lazy, 0usize),
            ("lazy|d", &lazy, &d, 0),
            ("x|lazy", &x, &lazy, 0),
            ("lazy|x", &lazy, &x, 0),
        ] {
            let oracle_lhs = if std::ptr::eq(lhs, &lazy) {
                &eager
            } else {
                lhs
            };
            let oracle_rhs = if std::ptr::eq(rhs, &lazy) {
                &eager
            } else {
                rhs
            };
            let oracle = oracle_lhs
                .materialize()
                .unwrap()
                .cat(&oracle_rhs.materialize().unwrap(), side)
                .unwrap();
            let fast = lhs.cat(rhs, side).unwrap();
            let diagonal = probe::record();
            let slow = probe::with_routes(DECLINE_ORIENTED_CAT, || lhs.cat(rhs, side).unwrap());
            let entries = diagonal.get(Event::DiagonalMaterialization);
            assert_eq!(
                bits(&slow),
                bits(&fast),
                "{name} {side:?}: fallback vs plan"
            );
            assert_eq!(
                bits(&slow),
                bits(&oracle),
                "{name} {side:?}: fallback vs oracle"
            );
            assert_eq!(slow.codomain(), oracle.codomain(), "{name} {side:?}");
            assert_eq!(slow.domain(), oracle.domain(), "{name} {side:?}");
            assert_eq!(slow.codomain(), fast.codomain(), "{name} {side:?}");
            assert_eq!(slow.domain(), fast.domain(), "{name} {side:?}");
            assert_eq!(entries, compact, "{name} {side:?}: densifications");
        }
    }
}

/// #1547/#1548: `dense_data` borrows dense Host storage and refuses a lazy
/// adjoint or a compact diagonal with `Unsupported { Materialize }`,
/// entering neither adjoint materialization nor diagonal densification.
#[test]
fn dense_data_borrows_dense_storage_and_never_materializes() {
    let runtime = runtime();
    let v = su2(&[(0, 2), (1, 1)]);
    let t: TensorMap<_, Complex64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 3).unwrap();
    let (dense, entries) = adjoint_entries(|| t.dense_data().unwrap());
    assert_eq!(entries, 0);
    assert_eq!(dense.as_ptr(), t.dense_data().unwrap().as_ptr());

    let adjoint = t.adjoint().unwrap();
    let (refused, entries) = adjoint_entries(|| adjoint.dense_data().map(<[_]>::len));
    assert_eq!(entries, 0, "dense_data entered adjoint materialization");
    assert!(is_unsupported(&refused.unwrap_err(), "dense_data"));
    // Positive control: the explicit path enters the probe.
    let (materialized, entries) = adjoint_entries(|| adjoint.materialize().unwrap());
    assert_eq!(entries, 1);
    assert_eq!(materialized.dense_data().unwrap().len(), dense.len());

    let bond = u1(&[(0, 2), (1, 1)]);
    let spectra = bond
        .sectors()
        .unwrap()
        .into_iter()
        .map(|sector| super::SectorSpectrum {
            values: vec![1.5; bond.degeneracy(&sector).unwrap()],
            sector,
        })
        .collect::<Vec<_>>();
    let d: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &bond, spectra).unwrap();
    let densified = probe::record();
    assert!(is_unsupported(&d.dense_data().unwrap_err(), "dense_data"));
    assert!(is_unsupported(
        &d.subblocks().map(|_| ()).unwrap_err(),
        "subblocks"
    ));
    assert_eq!(
        densified.get(Event::DiagonalMaterialization),
        0,
        "dense_data densified a compact diagonal"
    );
    let dense_d = d.materialize().unwrap();
    assert_eq!(dense_d.dense_data().unwrap().len(), 2 * 2 + 1);
    assert!(dense_d.subblocks().is_ok());
    let (refused, entries) = adjoint_entries(|| adjoint.subblocks().map(|_| ()));
    assert_eq!(entries, 0, "subblocks entered adjoint materialization");
    assert!(is_unsupported(&refused.unwrap_err(), "subblocks"));
}
