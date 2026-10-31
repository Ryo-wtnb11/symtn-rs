//! `isomorphism`, `isometry` and `rand_with_seed`.

use super::*;

/// The identity written sector by sector from its definition: `1` exactly
/// where codomain and domain fusion trees agree (uncoupled sectors, inner
/// lines and vertices) and the degeneracy indices agree.
macro_rules! equal_tree_identity {
    ($engine:ident, $runtime:expr, $legs:expr) => {{
        let legs = $legs;
        let rank = legs.len();
        TensorMap::<_, f64>::from_subblock_fn(
            $engine,
            $runtime,
            legs.iter().copied(),
            legs.iter().copied(),
            |trees, ij| {
                let same_tree = trees.codomain_uncoupled() == trees.domain_uncoupled()
                    && trees.codomain_innerlines() == trees.domain_innerlines()
                    && trees.codomain_vertices() == trees.domain_vertices();
                f64::from(u8::from(same_tree && ij[..rank] == ij[rank..]))
            },
        )
        .unwrap()
    }};
}

/// `w† w` against `isomorphism(domain, domain)`, exactly.
macro_rules! assert_exact_isometry {
    ($engine:ident, $runtime:expr, $w:expr) => {{
        let w = $w;
        let gram = w
            .adjoint($engine)
            .unwrap()
            .materialize($engine)
            .unwrap()
            .compose($engine, w)
            .unwrap();
        let identity: TensorMap<_, f64> =
            TensorMap::isomorphism($engine, $runtime, &w.domain(), &w.domain()).unwrap();
        assert_eq!(gram.dense_data().unwrap(), identity.dense_data().unwrap());
    }};
}

in_both_modes! {
    /// `isomorphism(V, V)` is the equal-tree identity for one to three legs
    /// (real and complex) and composes neutrally; an isomorphism between
    /// opposite dual arrangements and an isometry into a strictly larger or
    /// dual space satisfy `w† w = 1`. A pair that is not isomorphic (also when
    /// it embeds) or does not embed (also sector by sector with equal sector
    /// sets) is a space mismatch.
    fn structural_constructors_are_identities_and_isometries<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let space = |unit: usize, charged: usize| {
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), unit), (M::charged(), charged)])
                .unwrap()
        };
        let leg = space(1, 2);
        for legs in [vec![&leg], vec![&leg, &leg], vec![&leg, &leg, &leg]] {
            let oracle = equal_tree_identity!(engine, &runtime, &legs);
            let real: TensorMap<_, f64> =
                TensorMap::isomorphism(engine, &runtime, legs.iter().copied(), legs.iter().copied())
                    .unwrap();
            assert_eq!(real.dense_data().unwrap(), oracle.dense_data().unwrap());
            let complex: TensorMap<_, Complex64> =
                TensorMap::isomorphism(engine, &runtime, legs.iter().copied(), legs.iter().copied())
                    .unwrap();
            let oracle = oracle.convert::<Complex64>(engine).unwrap();
            assert_eq!(complex.dense_data().unwrap(), oracle.dense_data().unwrap());
        }

        let t: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, ij| {
                0.25 + (ij[0] * 5 + ij[1] * 3 + ij[2]) as f64
            })
            .unwrap();
        let left: TensorMap<_, f64> =
            TensorMap::isomorphism(engine, &runtime, [&leg, &leg], [&leg, &leg]).unwrap();
        let right: TensorMap<_, f64> =
            TensorMap::isomorphism(engine, &runtime, [&leg], [&leg]).unwrap();
        assert_eq!(left.compose(engine, &t).unwrap().dense_data().unwrap(), t.dense_data().unwrap());
        assert_eq!(t.compose(engine, &right).unwrap().dense_data().unwrap(), t.dense_data().unwrap());

        let dual = leg.try_dual().unwrap();
        let f: TensorMap<_, f64> =
            TensorMap::isomorphism(engine, &runtime, [&dual, &leg], [&leg, &dual]).unwrap();
        assert_exact_isometry!(engine, &runtime, &f);
        let (small, big) = (space(1, 2), space(2, 3));
        for domain in [small.clone(), small.try_dual().unwrap()] {
            let w: TensorMap<_, f64> =
                TensorMap::isometry(engine, &runtime, [&big], [&domain]).unwrap();
            assert_exact_isometry!(engine, &runtime, &w);
        }
        let w: TensorMap<_, f64> =
            TensorMap::isometry(engine, &runtime, [&leg, &leg], [&leg]).unwrap();
        assert_exact_isometry!(engine, &runtime, &w);

        for error in [
            facade_err(TensorMap::<_, f64>::isomorphism(engine, &runtime, [&leg, &leg], [&leg])),
            facade_err(TensorMap::<_, f64>::isomorphism(engine, &runtime, [&big], [&small])),
            facade_err(TensorMap::<_, f64>::isometry(engine, &runtime, [&leg], [&leg, &leg])),
            facade_err(TensorMap::<_, f64>::isometry(engine, &runtime, [&small], [&big])),
            facade_err(TensorMap::<_, f64>::isometry(engine, &runtime, [&space(1, 2)], [&space(2, 1)])),
        ] {
            assert!(is_space_mismatch(&error, ""), "{error:?}");
        }
    }
}

in_both_modes! {
    /// `rand_with_seed` is a function of the seed: the same seed replays the
    /// payload, another seed changes it, and the complex payload has a
    /// nonzero imaginary part in the real one's layout.
    fn rand_with_seed_is_a_function_of_the_seed<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
            .unwrap();
        let real = |engine: &mut symtn_dense::CpuEngine, seed| {
            TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], seed)
                .unwrap()
        };
        let first = real(engine, 7);
        assert!(!first.dense_data().unwrap().is_empty());
        assert_eq!(first.dense_data().unwrap(), real(engine, 7).dense_data().unwrap());
        assert_ne!(first.dense_data().unwrap(), real(engine, 8).dense_data().unwrap());
        let complex = |engine: &mut symtn_dense::CpuEngine, seed| {
            TensorMap::<_, Complex64>::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], seed)
                .unwrap()
        };
        let first_complex = complex(engine, 11);
        assert_eq!(first_complex.dense_data().unwrap().len(), first.dense_data().unwrap().len());
        assert!(first_complex.dense_data().unwrap().iter().any(|value| value.im != 0.0));
        assert_eq!(
            first_complex.dense_data().unwrap(),
            complex(engine, 11).dense_data().unwrap()
        );
        assert_ne!(
            first_complex.dense_data().unwrap(),
            complex(engine, 12).dense_data().unwrap()
        );
    }
}
