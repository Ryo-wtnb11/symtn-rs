//! `permute`, `braid`, `transpose`, `repartition`, `cat` and `absorb`: the
//! round-trip law and the argument validation both admission modes share.
//! The source is codomain-only because the toy provider's F-symbols cannot
//! represent a recoupling across both sides of a tree.

use super::*;
use symtn::typed::Side;

fn is_invalid_permutation(error: &symtn::typed::Error, rank: usize) -> bool {
    matches!(
        error,
        symtn::typed::Error::Operation(error) if matches!(
            **error,
            OperationError::InvalidPermutation { rank: actual, .. } if actual == rank
        )
    )
}

in_both_modes! {
    /// Permuting by `p` and then by `p⁻¹` restores the source's spaces,
    /// fusion trees and payload on the source provider, and a braid with a
    /// full level list is the permute of the same axes for a bosonic rule.
    fn permute_round_trips_and_a_bosonic_braid_is_the_permute<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
            .unwrap();
        let mut next = 0.0;
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg, &leg], [], |_, _| {
                next += 1.0;
                next
            })
            .unwrap();

        for (forward, inverse) in [([1, 0, 2], [1, 0, 2]), ([1, 2, 0], [2, 0, 1])] {
            let there = source.permute(engine, &forward, &[]).unwrap();
            assert!(std::ptr::eq(there.provider(), provider.as_ref()));
            assert_ne!(there.dense_data().unwrap(), source.dense_data().unwrap());
            let back = there.permute(engine, &inverse, &[]).unwrap();
            assert_eq!(back.codomain(), source.codomain());
            assert_eq!(back.subblock_count(), source.subblock_count());
            for index in 0..source.subblock_count() {
                assert_eq!(
                    back.subblock_fusion_trees(index).unwrap(),
                    source.subblock_fusion_trees(index).unwrap()
                );
            }
            dense_close(
                &format!("permute {forward:?} then {inverse:?}"),
                back.dense_data().unwrap(),
                source.dense_data().unwrap(),
            );

            for levels in [[0, 1, 2], [2, 1, 0]] {
                let braided = source.braid(engine, &forward, &[], &levels).unwrap();
                assert!(std::ptr::eq(braided.provider(), provider.as_ref()));
                assert_eq!(braided.dense_data().unwrap(), there.dense_data().unwrap());
            }
        }
    }
}

in_both_modes! {
    /// Malformed axes are `InvalidPermutation` for `permute`, `braid` and
    /// `transpose` (which also refuses a non-planar order rather than
    /// braiding); `braid` needs one level per source axis and `repartition`
    /// a split within the rank. The level-count refusal precedes provider
    /// work, and no refusal touches the source.
    fn transforms_reject_malformed_arguments_before_provider_work<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg, &leg], [], |_, ij| {
                (1 + ij[0] + 2 * ij[1] + 4 * ij[2]) as f64
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();

        for (codomain, domain) in [(&[0, 9, 1][..], &[][..]), (&[0, 0, 1], &[]), (&[0, 1], &[])] {
            let error = facade_err(source.permute(engine, codomain, domain));
            assert!(is_invalid_permutation(&error, 3), "permute: {error:?}");
            let error = facade_err(source.braid(engine, codomain, domain, &[0, 1, 2]));
            assert!(is_invalid_permutation(&error, 3), "braid: {error:?}");
            let error = facade_err(source.transpose(engine, codomain, domain));
            assert!(is_invalid_permutation(&error, 3), "transpose: {error:?}");
        }
        let error = facade_err(source.transpose(engine, &[1, 0, 2], &[]));
        assert!(is_invalid_permutation(&error, 3), "non-planar transpose: {error:?}");

        for levels in [&[0, 1][..], &[0; 4]] {
            M::reset_queries(&provider);
            match facade_err(source.braid(engine, &[1, 0, 2], &[], levels)) {
                symtn::typed::Error::InvalidArgument(message) => assert!(
                    message.contains("one level per source axis") && message.contains("expected 3"),
                    "{message}"
                ),
                other => panic!("levels {levels:?}: {other:?}"),
            }
            M::assert_no_queries(&provider);
        }

        let error = facade_err(source.repartition(engine, 4));
        assert!(error.to_string().contains("exceeds rank 3"), "{error}");
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}

in_both_modes! {
    /// `cat` and `absorb` keep their own validation classes: a runtime
    /// mismatch, a changed side with more than one leg, an unequal unchanged
    /// side, a changed leg of opposite duality, and `absorb`'s rank and
    /// duality equalities. Each is refused before provider work and leaves
    /// both operands untouched.
    fn cat_and_absorb_validate_their_arguments_before_provider_work<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = |unit: usize, charged: usize| {
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), unit), (M::charged(), charged)])
                .unwrap()
        };
        let (a, b) = (leg(1, 2), leg(2, 1).try_dual().unwrap());
        let fill = |_: &_, ij: &[usize]| ij.iter().sum::<usize>() as f64 + 1.0;
        let lhs: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&a, &b], [&leg(1, 1)], fill).unwrap();
        let lhs_before = lhs.dense_data().unwrap().to_vec();

        let two_domain_legs: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&a, &b], [&leg(1, 1), &leg(1, 1)], fill)
                .unwrap();
        let other_codomain: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg(3, 2), &b], [&leg(1, 1)], fill)
                .unwrap();
        let dual_domain: TensorMap<_, f64> = TensorMap::from_subblock_fn(
            engine,
            &runtime,
            [&a, &b],
            [&leg(1, 1).try_dual().unwrap()],
            fill,
        )
        .unwrap();
        let mut stack = |domain: GradedSpace<_>| {
            TensorMap::<_, f64>::from_subblock_fn(
                engine,
                &runtime,
                [&leg(2, 1)],
                [&a, &domain],
                fill,
            )
            .unwrap()
        };
        let (stack_lhs, stack_rhs) = (stack(leg(1, 1)), stack(leg(2, 1)));
        let other_runtime = Runtime::builder().build().unwrap();
        let foreign_runtime: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &other_runtime, [&a, &b], [&leg(1, 1)], fill)
                .unwrap();
        let operands_before: Vec<Vec<f64>> =
            [&two_domain_legs, &other_codomain, &dual_domain, &stack_rhs]
                .map(|t| t.dense_data().unwrap().to_vec())
                .to_vec();

        M::reset_queries(&provider);
        for error in [
            facade_err(lhs.cat(engine, &foreign_runtime, Side::Domain)),
            facade_err(lhs.cat(engine, &foreign_runtime, Side::Codomain)),
            facade_err(lhs.absorb(engine, &foreign_runtime)),
        ] {
            assert!(matches!(error, symtn::typed::Error::RuntimeMismatch), "{error:?}");
        }
        let invalid = |error: symtn::typed::Error, reason: &str| match error {
            symtn::typed::Error::InvalidArgument(message) => {
                assert!(message.contains(reason), "{reason}: {message}")
            }
            other => panic!("{reason}: {other:?}"),
        };
        invalid(
            facade_err(lhs.cat(engine, &two_domain_legs, Side::Domain)),
            "exactly one domain leg",
        );
        invalid(
            facade_err(lhs.absorb(engine, &two_domain_legs)),
            "equal codomain/domain ranks",
        );
        invalid(facade_err(lhs.absorb(engine, &dual_domain)), "equal duality");
        for (error, reason) in [
            (
                facade_err(lhs.cat(engine, &other_codomain, Side::Domain)),
                "identical codomain product spaces",
            ),
            (
                facade_err(stack_lhs.cat(engine, &stack_rhs, Side::Codomain)),
                "identical domain product spaces",
            ),
            (
                facade_err(lhs.cat(engine, &dual_domain, Side::Domain)),
                "opposite duality",
            ),
        ] {
            assert!(is_space_mismatch(&error, reason), "{reason}: {error:?}");
        }
        M::assert_no_queries(&provider);

        assert_eq!(lhs.dense_data().unwrap(), lhs_before.as_slice());
        for (tensor, before) in [&two_domain_legs, &other_codomain, &dual_domain, &stack_rhs]
            .into_iter()
            .zip(&operands_before)
        {
            assert_eq!(tensor.dense_data().unwrap(), before.as_slice());
        }
    }
}
