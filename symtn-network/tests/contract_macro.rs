//! `contract!` (#2029): every written form builds the same `Network` as the
//! equivalent `Network::new` call and returns the same result.
//!
//! "Same `Network`" is observed through the Runtime's plan cache, whose key
//! hashes every `Network` field (labels, `conj`, splits, output, output
//! split): after the macro plans once (a miss), the equivalent
//! `Network::new` contraction is a hit and adds no entry. A `Network` that
//! differs only in its output split misses, which shows the check
//! discriminates.

use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Complex64, GradedSpace, Runtime, TensorMap};
use symtn_network::{contract, plan_cache_stats, Network};

/// Asserts the call ran exactly one more contraction through an existing
/// cached plan (`hit`) or through a new one.
fn assert_cache(runtime: &Runtime, before: (u64, u64), hit: bool) {
    let stats = plan_cache_stats(runtime);
    let expected = if hit {
        (before.0 + 1, before.1)
    } else {
        (before.0, before.1 + 1)
    };
    assert_eq!((stats.hits, stats.misses), expected);
}

fn counts(runtime: &Runtime) -> (u64, u64) {
    let stats = plan_cache_stats(runtime);
    (stats.hits, stats.misses)
}

struct Svd<T> {
    u: T,
}

/// `[i; k] := (svd.u)[i; s] * conj(w.first()?)[k; s]`: a parenthesized
/// field chain and a general expression whose `?` is the caller's.
fn parenthesized<T: symtn_network::NetworkOperand>(
    engine: &mut dyn Engine,
    svd: &Svd<T>,
    w: &[T],
) -> Option<T> {
    contract!(engine, [i; k] := (svd.u)[i; s] * conj(w.first()?)[k; s]).ok()
}

macro_rules! forms {
    ($name:ident, $d:ty, $space:expr) => {
        #[test]
        fn $name() {
            let engine = &mut symtn::typed::CpuEngine::serial();
            let runtime = Runtime::builder().build().unwrap();
            let v = $space;
            let rand = |engine: &mut dyn Engine, codomain: &[&_], domain: &[&_], seed| {
                TensorMap::<_, $d>::rand_with_seed(engine,
                    &runtime,
                    codomain.iter().copied(),
                    domain.iter().copied(),
                    seed,
                )
                .unwrap()
            };
            let a = rand(engine, &[&v], &[&v], 1);
            let b = rand(engine, &[&v], &[&v], 2);

            // `:=` with `;` on every list.
            let before = counts(&runtime);
            let c = contract!(engine, [i; k] := a[i; j] * b[j; k]).unwrap();
            assert_cache(&runtime, before, false);
            let before = counts(&runtime);
            let expected = Network::new(
                [["i", "j"], ["j", "k"]],
                [false, false],
                [1, 1],
                ["i", "k"],
                Some(1),
            )
            .unwrap()
            .contract(engine, &[&a, &b])
            .unwrap();
            assert_cache(&runtime, before, true);
            assert_eq!((c.codomain(), c.domain()), (expected.codomain(), expected.domain()));
            assert_eq!(c.dense_data().unwrap(), expected.dense_data().unwrap());

            // A nested tuple field chain (`0.0` lexes as one float) is the
            // same operand.
            let tup = ((a.clone(),),);
            let before = counts(&runtime);
            let nested = contract!(engine, [i; k] := tup.0.0[i; j] * b[j; k]).unwrap();
            assert_cache(&runtime, before, true);
            assert_eq!(nested.dense_data().unwrap(), expected.dense_data().unwrap());

            // An output without `;`: every output leg in the codomain;
            // labels separated by spaces build the same network.
            let x = rand(engine, &[&v], &[&v], 3);
            let y = rand(engine, &[&v], &[&v, &v], 4);
            let before = counts(&runtime);
            let t = contract!(engine, [a, b, c] := x[a; i] * y[i; b, c]).unwrap();
            assert_cache(&runtime, before, false);
            let before = counts(&runtime);
            let spaced = contract!(engine, [a b c] := x[a; i] * y[i; b c]).unwrap();
            assert_cache(&runtime, before, true);
            let all_codomain = |engine: &mut dyn Engine, output_split| {
                Network::new(
                    [vec!["a", "i"], vec!["i", "b", "c"]],
                    [false, false],
                    [1, 1],
                    ["a", "b", "c"],
                    output_split,
                )
                .unwrap()
                .contract(engine, &[&x, &y])
                .unwrap()
            };
            let before = counts(&runtime);
            let expected = all_codomain(engine, None);
            assert_cache(&runtime, before, true);
            assert_eq!((t.codomain_rank(), t.domain_rank()), (3, 0));
            assert_eq!(t.dense_data().unwrap(), expected.dense_data().unwrap());
            assert_eq!(spaced.dense_data().unwrap(), expected.dense_data().unwrap());
            // Negative control: an explicit output split is another network.
            let before = counts(&runtime);
            all_codomain(engine, Some(3));
            assert_cache(&runtime, before, false);

            // No left-hand side: the scalar, through the rank-0 tensor.
            let psi = rand(engine, &[&v], &[&v, &v], 5);
            let h = rand(engine, &[&v], &[&v], 6);
            let before = counts(&runtime);
            let e: $d = contract!(engine, conj(psi)[p; l, r] * h[p; q] * psi[q; l, r]).unwrap();
            assert_cache(&runtime, before, false);
            let before = counts(&runtime);
            let expected = Network::new(
                [vec!["p", "l", "r"], vec!["p", "q"], vec!["q", "l", "r"]],
                [true, false, false],
                [1, 1, 1],
                Vec::<&str>::new(),
                None,
            )
            .unwrap()
            .contract(engine, &[&psi, &h, &psi])
            .unwrap();
            assert_cache(&runtime, before, true);
            assert_eq!(e, expected.scalar(engine).unwrap());

            // `[] :=` keeps the rank-0 tensor; its network is the scalar's.
            let before = counts(&runtime);
            let rank0 = contract!(engine, [] := conj(psi)[p; l, r] * h[p; q] * psi[q; l, r]).unwrap();
            assert_cache(&runtime, before, true);
            assert_eq!(rank0.rank(), 0);
            assert_eq!(rank0.scalar(engine).unwrap(), e);

            // Operand expressions: a parenthesized field chain, a bare field
            // chain, and a `?` expression yielding a reference.
            let svd = Svd { u: a.clone() };
            let w = vec![b.clone()];
            let before = counts(&runtime);
            let d = parenthesized(engine, &svd, &w).unwrap();
            assert_cache(&runtime, before, false);
            let before = counts(&runtime);
            let bare = contract!(engine, [i; k] := svd.u[i; s] * conj(w[0])[k; s]).unwrap();
            assert_cache(&runtime, before, true);
            let before = counts(&runtime);
            let expected = Network::new(
                [["i", "s"], ["k", "s"]],
                [false, true],
                [1, 1],
                ["i", "k"],
                Some(1),
            )
            .unwrap()
            .contract(engine, &[&a, &b])
            .unwrap();
            assert_cache(&runtime, before, true);
            assert_eq!(d.dense_data().unwrap(), expected.dense_data().unwrap());
            assert_eq!(bare.dense_data().unwrap(), expected.dense_data().unwrap());

            // Every operand's split is checked at run time, before the plan
            // lookup: a misplaced `;`, and a list without `;` (all codomain)
            // on a tensor with a domain leg.
            let before = counts(&runtime);
            let misplaced = contract!(engine, [i, k] := a[i, j;] * b[j; k]).unwrap_err();
            assert!(
                misplaced
                    .to_string()
                    .contains("operand 0 puts 2 label(s) before `;` but the tensor's codomain rank is 1"),
                "{misplaced}"
            );
            let unsplit = contract!(engine, [i; k] := a[i; j] * b[j, k]).unwrap_err();
            assert!(
                unsplit
                    .to_string()
                    .contains("operand 1 puts 2 label(s) before `;` but the tensor's codomain rank is 1"),
                "{unsplit}"
            );
            assert_eq!(counts(&runtime), before);
        }
    };
}

fn u1() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap()
}

fn su2() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

forms!(u1_f64_forms_match_network_new, f64, u1());
forms!(su2_c64_forms_match_network_new, Complex64, su2());
