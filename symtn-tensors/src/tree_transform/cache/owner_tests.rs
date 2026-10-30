//! The process-global completed-transformer owner: key
//! completeness, canonical-only publication, the reset epoch and failure
//! non-publication, and concurrent cold misses.

use std::hash::{BuildHasher, Hash};
use std::sync::{Arc, Barrier};

use num_complex::Complex64;
use symtn_core::{
    BlockStructure, FusionProductSpace, FusionTreeHomSpace, FusionTreePairOrientation,
    RuleIdentity, SectorLeg, StructureCacheEquivalent, U1FusionRule, U1Irrep,
};

use super::{
    resolve, CompletedTransformerKey, OrientedBasisOrder, TransformerMode, TreePairKeyRef,
    TreeTransformOperationView, TreeTransformPlanning,
};
use crate::probe::take_completed_transformer_activity;
use crate::{BoundDynamicFusionMapSpace, OperationError, TreeTransformOperation};

/// A canonical U(1) space `V^2 ← V` whose degeneracies (`7 + seed`) no other
/// test uses, so its transformer keys are this test's alone.
fn canonical_space(seed: usize) -> BoundDynamicFusionMapSpace<U1FusionRule> {
    let leg = |dual| {
        SectorLeg::new(
            [-1, 0, 1].map(|charge| (U1Irrep::new(charge).sector_id(), 7 + seed)),
            dual,
        )
    };
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(false), leg(false)]),
            FusionProductSpace::new([leg(false)]),
        ),
    )
    .unwrap()
}

fn fixture(
    seed: usize,
) -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformOperation,
) {
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let source = canonical_space(seed);
    let destination = source.transformed_multiplicity_free(&operation).unwrap();
    (
        Arc::clone(destination.space().structure()),
        Arc::clone(source.space().structure()),
        operation,
    )
}

fn same_core<T>(
    left: &crate::TreeTransformStructure<T>,
    right: &crate::TreeTransformStructure<T>,
) -> bool {
    Arc::ptr_eq(left.replay_core(), right.replay_core())
}

#[test]
fn every_key_determinant_splits_entries() {
    let (dst, src, operation) = fixture(0);
    let other_structure = fixture(1).0;
    let key = |rule: RuleIdentity,
               mode,
               operation: &TreeTransformOperation,
               orientation,
               basis_order,
               storage_conjugate,
               logical_source: Option<&BlockStructure>,
               dst: &BlockStructure,
               src: &BlockStructure| {
        CompletedTransformerKey::new::<f64>(
            rule,
            mode,
            operation,
            orientation,
            basis_order,
            storage_conjugate,
            logical_source,
            dst,
            src,
        )
    };
    let rule = RuleIdentity::of_type::<U1FusionRule>();
    let base = || {
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        )
    };
    let variants = [
        key(
            RuleIdentity::of_type::<()>(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::CheckedGeneric,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &TreeTransformOperation::transpose([1, 0], [2]),
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Adjoint,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Storage,
            false,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            true,
            None,
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            Some(&src),
            &dst,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &other_structure,
            &src,
        ),
        key(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &other_structure,
        ),
        CompletedTransformerKey::new::<Complex64>(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            &operation,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
            false,
            None,
            &dst,
            &src,
        ),
    ];
    // What: each determinant, varied alone, is a distinct key.
    for (index, variant) in variants.iter().enumerate() {
        assert_ne!(&base(), variant, "determinant {index}");
    }
    // An equal operation built independently is the same key.
    let rebuilt = key(
        rule.clone(),
        TransformerMode::MultiplicityFree,
        &TreeTransformOperation::permute(vec![1, 0], vec![2]),
        FusionTreePairOrientation::Direct,
        OrientedBasisOrder::Canonical,
        false,
        None,
        &dst,
        &src,
    );
    assert_eq!(base(), rebuilt);

    // What: the borrowed exact-layout probe hashes and compares like the
    // owned key it stands for.
    let probe = TreePairKeyRef {
        rule: &rule,
        coefficient: std::any::TypeId::of::<f64>(),
        operation: TreeTransformOperationView::of(&operation),
        dst: dst.content_id(),
        src: src.content_id(),
    };
    let hash = |value: &dyn Fn(&mut rustc_hash::FxHasher)| {
        let mut hasher = rustc_hash::FxBuildHasher.build_hasher();
        value(&mut hasher);
        std::hash::Hasher::finish(&hasher)
    };
    assert_eq!(hash(&|h| probe.hash(h)), hash(&|h| base().hash(h)));
    assert!(probe.equivalent(&base()));
    assert!(!probe.equivalent(&variants[7]));
}

#[test]
fn only_canonical_keys_publish_and_hits_rebind_the_callers_structures() {
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (dst, src, operation) = fixture(2);
    assert!(dst.is_canonical() && src.is_canonical());
    let mut planning = TreeTransformPlanning::default();
    take_completed_transformer_activity();
    let cold = planning
        .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
        .unwrap();
    let warm = planning
        .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
        .unwrap();
    let activity = take_completed_transformer_activity();
    assert_eq!(
        (activity.builds, activity.publications, activity.hits),
        (1, 1, 1)
    );
    assert!(same_core(&cold, &warm));
    // A hit binds the cached core to the structures it was asked about.
    assert!(Arc::ptr_eq(warm.dst_structure(), &dst));
    assert!(Arc::ptr_eq(warm.src_structure(), &src));

    // What: an equal never-admitted copy is lookup-only: every call builds,
    // nothing is published, and the result equals the canonical one.
    let copy = |structure: &BlockStructure| {
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                structure.rank(),
                (0..structure.block_count())
                    .map(|index| {
                        let block = structure.block(index).unwrap();
                        symtn_core::BlockSpec::with_key(
                            block.key().clone(),
                            block.shape().to_vec(),
                            block.strides().to_vec(),
                            block.offset(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
        )
    };
    let (expert_dst, expert_src) = (copy(&dst), copy(&src));
    assert!(!expert_dst.is_canonical() && !expert_src.is_canonical());
    for _ in 0..2 {
        let expert = planning
            .resolve_tree_pair(&U1FusionRule, &operation, &expert_dst, &expert_src, false)
            .unwrap();
        assert_eq!(expert, cold);
    }
    let activity = take_completed_transformer_activity();
    assert_eq!(
        (activity.builds, activity.publications, activity.hits),
        (2, 0, 0)
    );
}

#[test]
fn coefficient_types_are_separate_entries() {
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (dst, src, _) = fixture(3);
    let epoch = symtn_core::core_reset_epoch();
    let key = |complex| {
        let rule = RuleIdentity::of_type::<U1FusionRule>();
        let operation = TreeTransformOperation::permute([0, 1], [2]);
        let args = (
            rule,
            TransformerMode::MultiplicityFree,
            FusionTreePairOrientation::Direct,
            OrientedBasisOrder::Canonical,
        );
        if complex {
            CompletedTransformerKey::new::<Complex64>(
                args.0, args.1, &operation, args.2, args.3, false, None, &dst, &src,
            )
        } else {
            CompletedTransformerKey::new::<f64>(
                args.0, args.1, &operation, args.2, args.3, false, None, &dst, &src,
            )
        }
    };
    let compile_f64 = || {
        crate::TreeTransformStructure::<f64>::compile_structures(
            &dst,
            &src,
            &[crate::TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .map(|built| {
            built
                .with_canonical_structures(Arc::clone(&dst), Arc::clone(&src))
                .unwrap()
        })
    };
    let compile_c64 = || {
        crate::TreeTransformStructure::<Complex64>::compile_structures(
            &dst,
            &src,
            &[crate::TreeTransformBlockSpec::single(
                0,
                0,
                Complex64::new(0.0, 1.0),
            )],
        )
        .map(|built| {
            built
                .with_canonical_structures(Arc::clone(&dst), Arc::clone(&src))
                .unwrap()
        })
    };
    let real = resolve(key(false), true, epoch, &dst, &src, compile_f64).unwrap();
    let complex = resolve(key(true), true, epoch, &dst, &src, compile_c64).unwrap();
    // What: both are hits now, each of its own type: the coefficient TypeId
    // in the key fixes the erased value's type, so the downcast cannot fail.
    let real_hit = resolve(
        key(false),
        true,
        epoch,
        &dst,
        &src,
        || -> Result<_, OperationError> { panic!("f64 entry must hit") },
    )
    .unwrap();
    let complex_hit = resolve(
        key(true),
        true,
        epoch,
        &dst,
        &src,
        || -> Result<_, OperationError> { panic!("Complex64 entry must hit") },
    )
    .unwrap();
    assert!(same_core(&real, &real_hit) && same_core(&complex, &complex_hit));
    assert_eq!(
        complex_hit.single_coefficient(0),
        Some(Complex64::new(0.0, 1.0))
    );
}

#[test]
fn a_build_straddling_a_clear_or_failing_is_not_published() {
    // Clears every structure cache: serialize with the other clearing tests.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (dst, src, operation) = fixture(4);
    let mut planning = TreeTransformPlanning::default();
    take_completed_transformer_activity();
    crate::probe::set_before_publish_hook(Box::new(symtn_core::clear_structure_caches));
    let straddling = planning
        .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
        .unwrap();
    // What: the build that a clear straddled is returned but not resident,
    // so the next call builds again (and publishes).
    let rebuilt = planning
        .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
        .unwrap();
    assert!(!same_core(&straddling, &rebuilt));
    assert_eq!(straddling, rebuilt);
    let activity = take_completed_transformer_activity();
    assert_eq!(
        (activity.builds, activity.publications, activity.hits),
        (2, 2, 0)
    );
    let warm = planning
        .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
        .unwrap();
    assert!(same_core(&rebuilt, &warm));

    // What: a build error is returned and leaves no entry behind.
    // `V^2 <- V` is not the `V <- V^2` result of this permute: the build
    // finds no destination block for a transformed tree.
    let (_, other_src, _) = fixture(5);
    let wrong = TreeTransformOperation::permute([0], [1, 2]);
    let error = planning.resolve_tree_pair(&U1FusionRule, &wrong, &other_src, &other_src, false);
    assert!(error.is_err());
    let again = planning.resolve_tree_pair(&U1FusionRule, &wrong, &other_src, &other_src, false);
    assert!(again.is_err());
    // The warm hit above, then two builds that failed and published nothing.
    let activity = take_completed_transformer_activity();
    assert_eq!(
        (activity.builds, activity.publications, activity.hits),
        (2, 0, 1)
    );
}

#[test]
fn a_zero_degeneracy_budget_admits_no_completed_transformer() {
    if crate::test_support::run_isolated_or_return(
        "SYMTN_TENSORS_ZERO_DEGENERACY_BUDGET_ISOLATED",
        "tree_transform::cache::owner_tests::a_zero_degeneracy_budget_admits_no_completed_transformer",
    ) {
        return;
    }
    let kind = symtn_core::StructureCacheKind::DegeneracyStructure;
    let budget = symtn_core::structure_cache_info(kind).byte_budget();
    symtn_core::set_structure_cache_byte_budget(kind, 0);
    let completed = || {
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::CompletedTreeTransformer)
    };
    let before = completed();
    // What: with nothing resident in the degeneracy cache no
    // content is canonical, so every transformer key is lookup-only.
    let (dst, src, operation) = fixture(6);
    assert!(!dst.is_canonical() && !src.is_canonical());
    let mut planning = TreeTransformPlanning::default();
    for _ in 0..2 {
        planning
            .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
            .unwrap();
    }
    let after = completed();
    assert_eq!(after.admissions(), before.admissions());
    assert_eq!(after.misses(), before.misses());
    symtn_core::set_structure_cache_byte_budget(kind, budget);
}

#[test]
fn concurrent_cold_misses_build_without_waiting_and_admit_once() {
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (dst, src, operation) = fixture(7);
    let threads = 8;
    let barrier = Arc::new(Barrier::new(threads));
    let results = std::thread::scope(|scope| {
        (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let (dst, src, operation) = (&dst, &src, &operation);
                scope.spawn(move || {
                    let mut planning = TreeTransformPlanning::default();
                    planning.set_recoupling_threads(std::num::NonZeroUsize::new(4).unwrap());
                    barrier.wait();
                    let built = planning
                        .resolve_tree_pair(&U1FusionRule, operation, dst, src, false)
                        .unwrap();
                    (built, take_completed_transformer_activity())
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    // What: no caller parks on another's build (no single-flight): every
    // thread finishes, all results are equal, and they converge on one
    // resident core (the first admission wins; later offers get it).
    let resident = TreeTransformPlanning::default()
        .resolve_tree_pair(&U1FusionRule, &operation, &dst, &src, false)
        .unwrap();
    for (built, activity) in &results {
        assert_eq!(built, &resident);
        assert!(activity.builds + activity.hits == 1);
        if activity.builds == 1 {
            assert!(same_core(built, &resident));
        }
    }
}

#[test]
fn structure_cache_clear_does_not_reuse_content_ids() {
    // What: the clear-all races any test that assumes the shared
    // symtn-core structure caches stay stable across two builds. See
    // `test_support` for why this is a shared lock.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Content ids are never reused across a clear: id-keyed completed
    // transformers could otherwise alias.
    let base = 700_000_000usize;
    let before = BlockStructure::trivial(&[base]).unwrap();
    let id_before = before.content_id();
    symtn_core::clear_structure_caches();
    assert_eq!(before.required_len().unwrap(), base);
    let id_after = BlockStructure::trivial(&[base]).unwrap().content_id();
    assert!(
        id_after > id_before,
        "reset chain must not reuse content ids, got before={id_before} after={id_after}"
    );
}

#[test]
fn core_and_racah_resets_have_separate_owners() {
    use symtn_core::{FusionRule, MultiplicityFreeFusionSymbols, SU2FusionRule, SectorId};

    if crate::test_support::run_isolated_or_return(
        "SYMTN_CORE_RACAH_OWNERS_ISOLATED",
        "tree_transform::cache::owner_tests::core_and_racah_resets_have_separate_owners",
    ) {
        return;
    }
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    fusion_symbols::cache::reset();

    let rule = SU2FusionRule;
    let identity = rule.rule_identity();
    let structure = BlockStructure::trivial(&[700_000_001]).unwrap();
    let structure_id = structure.content_id();
    let sector = SectorId::new;
    let _ = rule.f_symbol_scalar(
        sector(1),
        sector(1),
        sector(1),
        sector(1),
        sector(0),
        sector(0),
    );
    fusion_symbols::cache::reset();
    assert_eq!(rule.rule_identity(), identity);
    let rebuilt = BlockStructure::trivial(&[700_000_001]).unwrap();
    assert_eq!(rebuilt, structure);
    assert_ne!(rebuilt.content_id(), structure_id);

    let _ = rule.f_symbol_scalar(
        sector(1),
        sector(1),
        sector(1),
        sector(1),
        sector(0),
        sector(0),
    );
    let base_tiers = || {
        use fusion_symbols::cache::{tier_stats, CoefficientCacheTier as Tier};
        [Tier::Wigner3j, Tier::Wigner6j, Tier::FSymbol].map(tier_stats)
    };
    let populated = base_tiers();
    assert_ne!(populated.iter().map(|tier| tier.entries).sum::<usize>(), 0);
    symtn_core::clear_structure_caches();
    assert_eq!(base_tiers(), populated);
}
