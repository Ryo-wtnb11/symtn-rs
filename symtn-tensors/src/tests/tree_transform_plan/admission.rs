use super::*;
use crate::probe::{self, Event};

fn malformed_simple_su2_tree_pair_tensors() -> (TestTensor<f64>, TestTensor<f64>) {
    let valid = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1],
        0,
        [false, false],
        [],
        [1],
    );
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(2, [(malformed, vec![1, 1])]).unwrap();
    let space = vec![1, 1];
    let dst = TestTensor::<f64>::from_vec_with_structure(vec![0.0], space.clone(), dst_structure)
        .unwrap();
    let src = TestTensor::<f64>::from_vec_with_structure(vec![1.0], space, src_structure).unwrap();
    (dst, src)
}

pub(super) fn simple_su2_vertex_structure(vertex: usize) -> Arc<BlockStructure> {
    let vertices = (vertex != 0).then_some(vertex);
    let key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            vertices,
            [],
        )
        .unwrap(),
    );
    Arc::new(packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap())
}

#[derive(Clone)]
pub(super) struct AdmissionCountingSu2Rule {
    pub(super) nsymbol_calls: Arc<AtomicUsize>,
    pub(super) fusion_style: Option<FusionStyleKind>,
    pub(super) braiding_style: Option<BraidingStyleKind>,
}

impl symtn_core::testing::Provider for AdmissionCountingSu2Rule {}

impl FusionRule for AdmissionCountingSu2Rule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        SU2FusionRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.fusion_style
            .unwrap_or_else(|| SU2FusionRule.fusion_style())
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.braiding_style
            .unwrap_or_else(|| SU2FusionRule.braiding_style())
    }

    fn vacuum(&self) -> SectorId {
        SU2FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SU2FusionRule.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        SU2FusionRule.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_calls.fetch_add(1, Ordering::Relaxed);
        SU2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for AdmissionCountingSu2Rule {}

impl MultiplicityFreeFusionSymbols for AdmissionCountingSu2Rule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        SU2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        SU2FusionRule.r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for AdmissionCountingSu2Rule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        SU2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

/// This thread's completed-transformer activity since the last call.
pub(super) fn owner_activity() -> crate::probe::CompletedActivity {
    crate::probe::take_completed_transformer_activity()
}

/// A rejected request observed no published transformer and published none.
/// (A failure inside the build itself still counts that build.)
pub(super) fn assert_owner_untouched(activity: crate::probe::CompletedActivity) {
    assert_eq!(
        (activity.hits, activity.publications),
        (0, 0),
        "{activity:?}"
    );
}

fn assert_invalid_simple_vertex(error: OperationError) {
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of vertices",
        })
    );
}

fn valid_simple_source_and_nonalias_malformed_destination() -> (TestTensor<f64>, TestTensor<f64>) {
    let valid = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1],
        0,
        [false, false],
        [],
        [1],
    );
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(malformed, vec![2, 1])]).unwrap();
    assert_ne!(src_structure.content_id(), dst_structure.content_id());
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![3.0], vec![1, 1], src_structure).unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![17.0, 19.0], vec![2, 1], dst_structure)
            .unwrap();
    (dst, src)
}

#[test]
fn callback_builder_admits_whole_source_before_first_callback() {
    let valid = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let malformed_later = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            2,
            [false, false],
            [],
            [],
            [],
            [],
            [],
        )
        .unwrap(),
    );
    let structure =
        packed_fixture_structure(2, [(valid, vec![1, 1]), (malformed_later, vec![1, 1])]).unwrap();
    let callbacks = std::cell::Cell::new(0usize);

    let error = build_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &structure,
        |source| {
            callbacks.set(callbacks.get() + 1);
            Ok(vec![(source.clone(), 1.0)])
        },
    )
    .unwrap_err();

    // What: a malformed later source block prevents source-major callbacks
    // from observing the valid prefix.
    assert_invalid_simple_vertex(error);
    assert_eq!(callbacks.get(), 0);
}

#[test]
fn callback_builder_rejects_malformed_destination_before_assembly() {
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], SectorId::new(0), [], [], []).unwrap(),
    );
    let malformed_destination = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [],
        0,
        [false, false],
        [],
        [],
        [],
        [],
        [],
    )
    .unwrap();
    assert_ne!(source, malformed_destination);
    let structure = packed_fixture_structure(2, [(BlockKey::from(source), vec![1, 1])]).unwrap();

    let error = build_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &structure,
        |_| Ok(vec![(malformed_destination.clone(), 1.0)]),
    )
    .unwrap_err();

    // What: a malformed callback destination is rejected before group assembly
    // even though its external sectors match the admitted source.
    assert_invalid_simple_vertex(error);
}

#[test]
fn warm_structure_aliases_are_rejected_before_owner_lookup() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();

    let valid_structure = simple_su2_vertex_structure(1);
    let invalid_structure = simple_su2_vertex_structure(0);
    assert_ne!(valid_structure.content_id(), invalid_structure.content_id());
    assert_ne!(
        valid_structure.block(0).unwrap().key(),
        invalid_structure.block(0).unwrap().key()
    );
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    mark_canonical([valid_structure.as_ref(), invalid_structure.as_ref()]);
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &valid_structure,
            &valid_structure,
            false,
        )
        .unwrap();
    owner_activity();

    let error = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &invalid_structure,
            &valid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);

    // What: a malformed destination is rejected before the owner is
    // consulted, even after a valid sibling structure is warm.
    assert_owner_untouched(owner_activity());

    let error = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &valid_structure,
            &invalid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);
    assert_owner_untouched(owner_activity());

    let mut independent = crate::tree_transform::TreeTransformPlanning::default();
    let error = independent
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &invalid_structure,
            &valid_structure,
            false,
        )
        .unwrap_err();
    assert_invalid_simple_vertex(error);

    // What: no other resolution's published transformer admits a malformed
    // raw structure.
    assert_owner_untouched(owner_activity());
}

#[test]
fn exact_warm_structure_reuses_prior_local_admission_proof() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
        fusion_style: None,
        braiding_style: None,
    };
    let structure = simple_su2_vertex_structure(1);
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    mark_canonical([structure.as_ref()]);
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();

    let cold = cache
        .resolve_tree_pair(engine, &rule, &operation, &structure, &structure, false)
        .unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    let same_content = Arc::new((*structure).clone());
    assert_eq!(same_content.content_id(), structure.content_id());
    assert!(!Arc::ptr_eq(&same_content, &structure));
    let warm = cache
        .resolve_tree_pair(
            engine,
            &rule,
            &operation,
            &same_content,
            &same_content,
            false,
        )
        .unwrap();

    // What: an exact semantic-key/content replay reuses the LOCAL admission
    // proof carried by the published compiled structure.
    assert!(same_core(&cold, &warm));
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    symtn_core::clear_structure_caches();
    calls.store(0, Ordering::Relaxed);
    cache
        .resolve_tree_pair(engine, &rule, &operation, &structure, &structure, false)
        .unwrap();

    // What: clearing retained entries removes proof reuse; the next call
    // performs LOCAL admission again.
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[test]
fn completed_structure_miss_and_hit_validate_capability_once() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = simple_su2_vertex_structure(1);
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    mark_canonical([structure.as_ref()]);
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();

    let multiplicity_free_capability_validations_rec = probe::record();
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    assert_eq!(
        multiplicity_free_capability_validations_rec
            .get(Event::MultiplicityFreeCapabilityValidation),
        1
    );

    let multiplicity_free_capability_validations_rec = probe::record();
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &structure,
            &structure,
            false,
        )
        .unwrap();

    // What: misses hand the prior capability proof to the whole-HomSpace
    // compiler, while hits still perform their one admission check.
    assert_eq!(
        multiplicity_free_capability_validations_rec
            .get(Event::MultiplicityFreeCapabilityValidation),
        1
    );
}

#[test]
fn prelowered_same_content_roles_share_one_local_admission() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
        fusion_style: None,
        braiding_style: None,
    };
    let logical = simple_su2_vertex_structure(1);
    let storage = Arc::new((*logical).clone());
    let destination = Arc::new((*logical).clone());
    assert_eq!(logical.content_id(), storage.content_id());
    assert_eq!(logical.content_id(), destination.content_id());
    let cache = crate::tree_transform::TreeTransformPlanning::default();

    cache
        .get_or_compile_tree_pair_prelowered(
            engine,
            &rule,
            &TreeTransformOperation::braid([1, 0], [], [0, 1], []),
            &destination,
            &logical,
            &storage,
            false,
            Ok,
            Ok,
        )
        .unwrap();

    // What: logical, storage, and destination roles sharing immutable content
    // perform one LOCAL categorical admission, even through distinct Arcs.
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn expert_layout_tree_transform_paths_compile_without_publishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = simple_su2_vertex_structure(1);
    let space = vec![1, 1];
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0], space.clone(), (*structure).clone())
            .unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0], space, (*structure).clone()).unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);

    let assert_uncached = |f: &mut dyn FnMut(&mut crate::tree_transform::TreeTransformPlanning)| {
        let mut cache = crate::tree_transform::TreeTransformPlanning::default();
        owner_activity();
        owner_activity();
        f(&mut cache);

        // What: an expert (never admitted) layout compiles eagerly; its
        // keys are lookup-only, so nothing is published or hit.
        let activity = owner_activity();
        assert_eq!((activity.hits, activity.publications), (0, 0));
    };

    assert_uncached(&mut |cache| {
        cache
            .resolve_tree_pair(
                engine,
                &SU2FusionRule,
                &operation.clone(),
                dst.structure(),
                src.structure(),
                false,
            )
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        cache
            .resolve_tree_pair(
                engine,
                &SU2FusionRule,
                &operation,
                &structure,
                &structure,
                false,
            )
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        cache
            .get_or_compile_tree_pair_prelowered(
                engine,
                &SU2FusionRule,
                &operation,
                &structure,
                &structure,
                &structure,
                false,
                Ok,
                Ok,
            )
            .unwrap();
    });
    assert_uncached(&mut |cache| {
        let logical_keys = (0..structure.block_count())
            .map(|index| match structure.block(index).unwrap().key() {
                BlockKey::FusionTree(key) => key.clone(),
                _ => unreachable!("SU2 fixture uses fusion-tree keys"),
            })
            .collect::<Vec<_>>();
        let storage_indices = (0..logical_keys.len()).collect::<Vec<_>>();
        cache
            .resolve_tree_pair_oriented(
                engine,
                &SU2FusionRule,
                &operation,
                &structure,
                &logical_keys,
                || Ok(&storage_indices),
                &structure,
                symtn_core::FusionTreePairOrientation::Direct,
                crate::tree_transform::OrientedBasisOrder::Canonical,
                structure.rank(),
                Ok,
            )
            .unwrap();
    });
}

#[test]
fn oriented_adjoint_projection_matches_materialized_logical_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let make_key = |coupled: usize| {
        let coupled = SectorId::new(coupled);
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(1), SectorId::new(1)],
                coupled,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(2), SectorId::new(2)],
                coupled,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
        )
    };
    let storage_keys = [make_key(2), make_key(0)];
    let adjoint_key = |key: &FusionTreePairKey| {
        FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
    };
    let logical_keys = [adjoint_key(&storage_keys[1]), adjoint_key(&storage_keys[0])];
    let storage = Arc::new(
        packed_fixture_structure(
            4,
            storage_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let logical = Arc::new(
        packed_fixture_structure(
            4,
            logical_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let storage_indices = [1, 0];
    let storage_axes = [2, 3, 0, 1];

    let old_cache = crate::tree_transform::TreeTransformPlanning::default();
    let old = old_cache
        .get_or_compile_tree_pair_prelowered(
            engine,
            &SU2FusionRule,
            &operation,
            &logical,
            &logical,
            &storage,
            true,
            |index| Ok(storage_indices[index]),
            |axis| Ok(storage_axes[axis]),
        )
        .unwrap();
    let mut oriented_cache = crate::tree_transform::TreeTransformPlanning::default();
    let oriented = oriented_cache
        .resolve_tree_pair_oriented(
            engine,
            &SU2FusionRule,
            &operation,
            &logical,
            &logical_keys,
            || Ok(&storage_indices),
            &storage,
            symtn_core::FusionTreePairOrientation::Adjoint,
            crate::tree_transform::OrientedBasisOrder::Canonical,
            4,
            |axis| Ok(storage_axes[axis]),
        )
        .unwrap();

    // What: an adjoint operand with noncanonical parent block order compiles
    // and replays exactly like the materialized logical structure.
    assert_eq!(oriented, old);
    let space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        (*storage).clone(),
    )
    .unwrap();
    let make_dst = || {
        TestTensor::<f64>::from_vec_with_structure(
            vec![0.0, 0.0],
            space.clone(),
            (*logical).clone(),
        )
        .unwrap()
    };
    let mut old_dst = make_dst();
    let mut oriented_dst = make_dst();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        engine,
        &mut backend,
        &mut workspace,
        &old,
        &mut old_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    replay_into(
        engine,
        &mut backend,
        &mut workspace,
        &oriented,
        &mut oriented_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(oriented_dst.data(), old_dst.data());
}

#[test]
fn adjoint_oriented_transformer_is_reused_and_keyed_by_orientation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // Both trees carry sectors [1, 1], so the parent block set is closed under
    // the adjoint and the ordinary and oriented compilers accept the same
    // destination and parent structures: only the key separates their plans.
    let tree = |coupled: usize| {
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(coupled),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let storage_keys = [
        FusionTreePairKey::pair(tree(2), tree(2)),
        FusionTreePairKey::pair(tree(0), tree(0)),
    ];
    let storage = Arc::new(
        packed_fixture_structure(
            4,
            storage_keys
                .iter()
                .cloned()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let destination = Arc::new(
        packed_fixture_structure(
            4,
            [storage_keys[1].clone(), storage_keys[0].clone()]
                .into_iter()
                .map(|key| (key, vec![1usize; 4])),
        )
        .unwrap(),
    );
    let logical_keys = [storage_keys[1].clone(), storage_keys[0].clone()];
    let storage_indices = [1, 0];
    let storage_axes = [2, 3, 0, 1];
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let other_operation = TreeTransformOperation::braid([0, 1], [3, 2], [0, 1], [2, 3]);
    let oriented = |engine: &mut dyn symtn_dense::Engine,
                    cache: &mut crate::tree_transform::TreeTransformPlanning,
                    operation: &TreeTransformOperation,
                    orientation: symtn_core::FusionTreePairOrientation| {
        let direct = orientation == symtn_core::FusionTreePairOrientation::Direct;
        cache
            .resolve_tree_pair_oriented(
                engine,
                &SU2FusionRule,
                operation,
                &destination,
                if direct { &storage_keys } else { &logical_keys },
                || Ok(if direct { &[0, 1] } else { &storage_indices }),
                &storage,
                orientation,
                crate::tree_transform::OrientedBasisOrder::Canonical,
                4,
                |axis| Ok(if direct { axis } else { storage_axes[axis] }),
            )
            .unwrap()
    };
    let adjoint = symtn_core::FusionTreePairOrientation::Adjoint;
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([storage.as_ref(), destination.as_ref()]);
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    crate::probe::take(crate::probe::Event::OrientedTreePairCompile);
    owner_activity();
    crate::probe::take_coefficient_group_activity();

    let first = oriented(engine, &mut cache, &operation, adjoint);
    let second = oriented(engine, &mut cache, &operation, adjoint);

    // What: the second call hits the published adjoint transformer, keyed by
    // the parent content, orientation and basis order; only the first looks
    // up its one source group's coefficients.
    assert_eq!(
        crate::probe::take(crate::probe::Event::OrientedTreePairCompile),
        1
    );
    assert!(same_core(&first, &second));
    let activity = owner_activity();
    assert_eq!((activity.hits, activity.builds), (1, 1));
    let groups = crate::probe::take_coefficient_group_activity();
    assert_eq!(groups.hits + groups.misses, 1);
    assert_eq!(groups.publications, groups.misses);
    // What: the retained transformer equals a fresh compile of equal
    // (never-admitted) layouts.
    let fresh = {
        let storage = expert_copy(&storage);
        let destination = expert_copy(&destination);
        cache
            .resolve_tree_pair_oriented(
                engine,
                &SU2FusionRule,
                &operation,
                &destination,
                &logical_keys,
                || Ok(&storage_indices),
                &storage,
                adjoint,
                crate::tree_transform::OrientedBasisOrder::Canonical,
                4,
                |axis| Ok(storage_axes[axis]),
            )
            .unwrap()
    };
    assert!(!same_core(&first, &fresh));
    assert_eq!(first, fresh);
    owner_activity();

    let conjugated = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &destination,
            &storage,
            true,
        )
        .unwrap();
    let plain = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation,
            &destination,
            &storage,
            false,
        )
        .unwrap();
    let other = oriented(engine, &mut cache, &other_operation, adjoint);

    // What: an ordinary transformer with either conjugation flag, and an
    // adjoint one for another operation, each miss instead of aliasing the
    // adjoint entry.
    assert!(!same_core(&first, &conjugated));
    assert!(!same_core(&first, &plain));
    assert!(!same_core(&first, &other));
    let activity = owner_activity();
    assert_eq!((activity.hits, activity.builds), (0, 3));

    let direct = symtn_core::FusionTreePairOrientation::Direct;
    let direct_first = oriented(engine, &mut cache, &operation, direct);
    let direct_second = oriented(engine, &mut cache, &operation, direct);

    // What: a direct oriented transformer is neither served from nor
    // published under the ordinary tree-pair key it would share.
    assert_eq!(
        crate::probe::take(crate::probe::Event::OrientedTreePairCompile),
        4
    );
    assert!(!same_core(&direct_first, &plain));
    assert!(!same_core(&direct_first, &direct_second));
    let activity = owner_activity();
    assert_eq!((activity.hits, activity.publications), (0, 0));

    let storage_ordered =
        |engine: &mut dyn symtn_dense::Engine,
         cache: &mut crate::tree_transform::TreeTransformPlanning| {
            cache
                .resolve_tree_pair_oriented(
                    engine,
                    &SU2FusionRule,
                    &operation,
                    &destination,
                    &storage_keys,
                    || Ok(&[0, 1]),
                    &storage,
                    adjoint,
                    crate::tree_transform::OrientedBasisOrder::Storage,
                    4,
                    |axis| Ok(storage_axes[axis]),
                )
                .unwrap()
        };
    let storage_first = storage_ordered(engine, &mut cache);
    assert!(!same_core(&first, &storage_first));
    assert!(same_core(
        &storage_first,
        &storage_ordered(engine, &mut cache)
    ));
    assert!(same_core(
        &first,
        &oriented(engine, &mut cache, &operation, adjoint)
    ));
}

#[test]
fn adjoint_oriented_degeneracy_change_reuses_the_composed_coefficients() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What: a new parent layout with the same sectors misses the completed-
    // transformer cache but recomposes no source group, and the result
    // equals the uncached eager producer on that layout bit for bit.
    let tree = |coupled: usize| {
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(coupled),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let storage_keys = [
        FusionTreePairKey::pair(tree(2), tree(2)),
        FusionTreePairKey::pair(tree(0), tree(0)),
    ];
    let logical_keys = [storage_keys[1].clone(), storage_keys[0].clone()];
    let storage_indices = [1, 0];
    let storage_axes = [2, 3, 0, 1];
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let layouts = |degeneracy: usize| {
        let structure = |keys: [FusionTreePairKey; 2]| {
            Arc::new(
                packed_fixture_structure(4, keys.into_iter().map(|key| (key, vec![degeneracy; 4])))
                    .unwrap(),
            )
        };
        (
            structure(storage_keys.clone()),
            structure([storage_keys[1].clone(), storage_keys[0].clone()]),
        )
    };
    let mut oriented =
        |cache: &mut crate::tree_transform::TreeTransformPlanning,
         (storage, destination): &(Arc<BlockStructure>, Arc<BlockStructure>)| {
            cache
                .resolve_tree_pair_oriented(
                    engine,
                    &SU2FusionRule,
                    &operation,
                    destination,
                    &logical_keys,
                    || Ok(&storage_indices),
                    storage,
                    symtn_core::FusionTreePairOrientation::Adjoint,
                    crate::tree_transform::OrientedBasisOrder::Canonical,
                    4,
                    |axis| Ok(storage_axes[axis]),
                )
                .unwrap()
        };
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    let narrow = layouts(1);
    let wider = layouts(2);
    mark_canonical(
        [&narrow, &wider]
            .into_iter()
            .flat_map(|(storage, destination)| [storage.as_ref(), destination.as_ref()]),
    );
    owner_activity();

    oriented(&mut cache, &narrow);
    crate::probe::take_coefficient_group_activity();
    let warm = oriented(&mut cache, &wider);

    assert_eq!(owner_activity().builds, 2);
    let groups = crate::probe::take_coefficient_group_activity();
    assert_eq!((groups.hits, groups.misses, groups.publications), (1, 0, 0));
    let mut uncached = crate::tree_transform::TreeTransformPlanning::default();
    let expert = (expert_copy(&wider.0), expert_copy(&wider.1));
    let rebuilt = oriented(&mut uncached, &expert);
    assert_eq!(warm, rebuilt);
    // The uncached eager producer on the same layout: no owner, no cache 4.
    let projection = logical_keys
        .iter()
        .zip(storage_indices)
        .collect::<rustc_hash::FxHashMap<_, _>>();
    let eager =
        crate::tree_transform::build_oriented_tree_pair_transform_group_plan_capability_validated(
            &SU2FusionRule,
            operation.clone(),
            &logical_keys,
            &wider.0,
            symtn_core::FusionTreePairOrientation::Adjoint,
            crate::tree_transform::OrientedBasisOrder::Canonical,
            4,
            &projection,
            &mut symtn_operations::host_pool::test_engine(1),
            None,
        )
        .unwrap()
        .compile_shared_structures_with_source_projection(
            Arc::clone(&wider.1),
            Arc::clone(&wider.0),
            4,
            |key| Ok(projection[key]),
            |axis| Ok(storage_axes[axis]),
            true,
        )
        .unwrap();
    assert_eq!(warm, eager);
}

#[test]
fn runtime_bound_adjoint_oriented_projection_errors_match_the_local_path() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tree = |coupled: usize| {
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(coupled),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap()
    };
    let keys = [
        FusionTreePairKey::pair(tree(2), tree(2)),
        FusionTreePairKey::pair(tree(0), tree(0)),
    ];
    let storage = Arc::new(
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1usize; 4])))
            .unwrap(),
    );
    let operation = TreeTransformOperation::braid([1, 0], [3, 2], [0, 1], [2, 3]);
    let duplicate = [keys[0].clone(), keys[0].clone()];
    owner_activity();
    crate::probe::take_coefficient_group_activity();
    let mut bound = crate::tree_transform::TreeTransformPlanning::default();
    let engine = &mut symtn_operations::host_pool::test_engine(4);
    let mut local = crate::tree_transform::TreeTransformPlanning::default();
    let malformed: [(&[FusionTreePairKey], &[usize]); 4] = [
        (&keys, &[1]),
        (&keys, &[1, 2]),
        (&duplicate, &[1, 0]),
        (&duplicate, &[1, 2]),
    ];
    for (logical_keys, storage_indices) in malformed {
        let mut compile = |cache: &mut crate::tree_transform::TreeTransformPlanning| {
            let error = cache
                .resolve_tree_pair_oriented(
                    engine,
                    &SU2FusionRule,
                    &operation,
                    &storage,
                    logical_keys,
                    || Ok(storage_indices),
                    &storage,
                    symtn_core::FusionTreePairOrientation::Adjoint,
                    crate::tree_transform::OrientedBasisOrder::Canonical,
                    4,
                    Ok,
                )
                .unwrap_err();
            format!("{error:?}")
        };

        // What: a multi-threaded miss reports the same first projection
        // error as a serial one and admits nothing.
        assert_eq!(compile(&mut bound), compile(&mut local));
    }
    // What: every malformed projection is read only after an owner miss, so
    // each one counts a build and publishes nothing; no coefficient group is
    // looked up before the projection is proved.
    let activity = owner_activity();
    assert_eq!((activity.builds, activity.publications), (8, 0));
    assert_eq!(
        crate::probe::take_coefficient_group_activity(),
        Default::default()
    );
}

#[test]
fn prelowered_compile_does_not_enter_the_completed_transformer_cache() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = simple_su2_vertex_structure(1);
    mark_canonical([structure.as_ref()]);
    let operation_a = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let operation_b = TreeTransformOperation::permute([0, 1], []);
    let operation_c = TreeTransformOperation::braid([1, 0], [], [1, 0], []);
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    owner_activity();

    let first_a = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    cache
        .get_or_compile_tree_pair_prelowered(
            engine,
            &SU2FusionRule,
            &operation_b,
            &structure,
            &structure,
            &structure,
            false,
            Ok,
            Ok,
        )
        .unwrap();
    let warm_a = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    assert!(same_core(&first_a, &warm_a));
    assert_eq!(owner_activity().publications, 1);
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_c,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_b,
            &structure,
            &structure,
            false,
        )
        .unwrap();

    // What: the prelowered compile never reaches the completed-transformer
    // owner; the later ordinary B and C calls are both builds.
    let activity = owner_activity();
    assert_eq!((activity.hits, activity.builds), (0, 2));
}

#[test]
fn failed_structure_compile_publishes_nothing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = simple_su2_vertex_structure(1);
    let wrong_destination = Arc::new(
        packed_fixture_structure(
            2,
            [(
                BlockKey::from(
                    FusionTreePairKey::try_pair_from_sector_ids(
                        [2, 2],
                        [],
                        0,
                        [false, false],
                        [],
                        [],
                        [],
                        [1],
                        [],
                    )
                    .unwrap(),
                ),
                vec![1, 1],
            )],
        )
        .unwrap(),
    );
    let operation_a = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let operation_b = TreeTransformOperation::permute([0, 1], []);
    let operation_c = TreeTransformOperation::braid([1, 0], [], [1, 0], []);
    mark_canonical([structure.as_ref(), wrong_destination.as_ref()]);
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    owner_activity();

    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_b,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    let first_a = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_b,
            &wrong_destination,
            &structure,
            false,
        )
        .unwrap_err();
    // The failed build of a publishable key published nothing.
    let activity = owner_activity();
    assert_eq!((activity.builds, activity.publications), (3, 2));
    let warm_a = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    assert!(same_core(&first_a, &warm_a));
    cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_c,
            &structure,
            &structure,
            false,
        )
        .unwrap();
    let retained_a = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &operation_a,
            &structure,
            &structure,
            false,
        )
        .unwrap();

    // What: a failed compile publishes nothing; the A entry stays resident.
    assert!(same_core(&first_a, &retained_a));
    let activity = owner_activity();
    assert_eq!((activity.hits, activity.builds), (2, 1));
}

#[test]
fn warm_prelowered_raw_arc_aliases_do_not_observe_or_mutate_cache_state() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();

    let valid = simple_su2_vertex_structure(1);
    let invalid = simple_su2_vertex_structure(0);
    assert_ne!(valid.content_id(), invalid.content_id());
    assert!(!Arc::ptr_eq(&valid, &invalid));
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let cache = crate::tree_transform::TreeTransformPlanning::default();
    owner_activity();
    cache
        .get_or_compile_tree_pair_prelowered(
            engine,
            &SU2FusionRule,
            &operation,
            &valid,
            &valid,
            &valid,
            false,
            Ok,
            Ok,
        )
        .unwrap();
    owner_activity();

    for (destination, logical_source, storage_source) in [
        (&valid, &invalid, &valid),
        (&valid, &valid, &invalid),
        (&invalid, &valid, &valid),
    ] {
        let error = cache
            .get_or_compile_tree_pair_prelowered(
                engine,
                &SU2FusionRule,
                &operation,
                destination,
                logical_source,
                storage_source,
                false,
                Ok,
                Ok,
            )
            .unwrap_err();
        assert_invalid_simple_vertex(error);

        // What: each raw-Arc role is categorically admitted before any
        // completed-transformer lookup can observe its shared content identity.
        assert_owner_untouched(owner_activity());
    }
}

#[test]
fn invalid_simple_source_does_not_mutate_cached_or_no_cache_state() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What: malformed categorical input is rejected before cache statistics or
    // retained completed-structure state change under either execution policy.
    for _ in 0..2 {
        let (dst, src) = malformed_simple_su2_tree_pair_tensors();
        let mut cache = crate::tree_transform::TreeTransformPlanning::default();
        owner_activity();
        owner_activity();

        let error = cache
            .resolve_tree_pair(
                engine,
                &SU2FusionRule,
                &TreeTransformOperation::braid([0, 1], [], [0, 1], []),
                dst.structure(),
                src.structure(),
                false,
            )
            .unwrap_err();

        assert_eq!(
            error,
            OperationError::Core(CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            })
        );
        assert_owner_untouched(owner_activity());
    }
}

#[test]
fn malformed_source_precedes_noncategorical_destination_without_cache_mutation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (_, src) = malformed_simple_su2_tree_pair_tensors();
    let dst_structure = packed_fixture_structure(2, [(BlockKey::opaque([9]), vec![1, 1])]).unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0], vec![1, 1], dst_structure).unwrap();
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    owner_activity();

    let error = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &TreeTransformOperation::permute([0, 1], []),
            dst.structure(),
            src.structure(),
            false,
        )
        .unwrap_err();

    // What: cold admission proves the categorical source before diagnosing
    // the destination namespace, without touching cache state.
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        })
    );
    assert_owner_untouched(owner_activity());
}

#[test]
fn invalid_operation_precedes_malformed_simple_source_without_cache_mutation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // What: operation syntax has deterministic precedence over categorical
    // source admission and neither failure is counted as a cache miss.
    let (dst, src) = malformed_simple_su2_tree_pair_tensors();
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    owner_activity();

    let tree_pair_operation_preparations_rec = probe::record();
    let error = cache
        .resolve_tree_pair(
            engine,
            &SU2FusionRule,
            &TreeTransformOperation::braid([0, 0], [], [0, 1], []),
            dst.structure(),
            src.structure(),
            false,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidPermutation {
            axes: vec![0, 0],
            rank: 2
        }
    );
    assert_eq!(
        tree_pair_operation_preparations_rec.get(Event::TreePairOperationPreparation),
        0
    );
    assert_owner_untouched(owner_activity());
}

#[test]
fn invalid_operation_precedes_non_categorical_namespace_without_cache_mutation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.clone(),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();
        let space = vec![1, 1];
        let dst =
            TestTensor::<f64>::from_vec_with_structure(vec![0.0], space.clone(), structure.clone())
                .unwrap();
        let src = TestTensor::<f64>::from_vec_with_structure(vec![1.0], space, structure).unwrap();
        let mut cache = crate::tree_transform::TreeTransformPlanning::default();
        owner_activity();

        let error = cache
            .resolve_tree_pair(
                engine,
                &SU2FusionRule,
                &TreeTransformOperation::permute([0, 0], []),
                dst.structure(),
                src.structure(),
                false,
            )
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::InvalidPermutation {
                axes: vec![0, 0],
                rank: 2
            }
        );
        assert_owner_untouched(owner_activity());

        let error = cache
            .resolve_tree_pair(
                engine,
                &SU2FusionRule,
                &TreeTransformOperation::permute([0, 1], []),
                dst.structure(),
                src.structure(),
                false,
            )
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::Core(CoreError::ExpectedFusionTreePairKey { actual: key.kind() })
        );
        // What: syntax and namespace failures occur before any cache lookup or
        // categorical coefficient provider can compile a row.
        assert_owner_untouched(owner_activity());
    }
}

#[test]
fn invalid_unique_source_does_not_count_eager_compile_misses() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What: a failed Unique eager build reports no successful structure compile
    // in cache statistics.
    let valid = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
    let malformed = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let dst_structure = packed_fixture_structure(2, [(valid, vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(2, [(malformed, vec![1, 1])]).unwrap();
    let space = vec![1, 1];
    let dst = TestTensor::<f64>::from_vec_with_structure(vec![0.0], space.clone(), dst_structure)
        .unwrap();
    let src = TestTensor::<f64>::from_vec_with_structure(vec![1.0], space, src_structure).unwrap();
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();
    owner_activity();

    let error = cache
        .resolve_tree_pair(
            engine,
            &Z2FusionRule,
            &TreeTransformOperation::permute([0, 1], []),
            dst.structure(),
            src.structure(),
            false,
        )
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        })
    );
    assert_owner_untouched(owner_activity());
}

#[test]
fn direct_tree_pair_rejects_nonalias_malformed_destination_without_writing() {
    let (mut dst, src) = valid_simple_source_and_nonalias_malformed_destination();
    let before = dst.data().to_vec();

    let error = eager_tree_transform_into(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1], []),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap_err();

    // What: noncached tree-pair admission validates the destination before
    // replay can modify caller-owned storage.
    assert_invalid_simple_vertex(error);
    assert_eq!(dst.data(), before);
}

#[test]
fn all_codomain_pair_mismatch_is_rejected_before_source_scope() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What: whole-pair categorical admission precedes the all-codomain source
    // restriction, so mismatched coupled sectors retain the core error.
    let nonempty_domain = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(0)],
            SectorId::new(0),
            [false],
            [],
            [],
        )
        .unwrap(),
    ));
    let mismatched = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            SectorId::new(0),
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [SectorId::new(2)],
            SectorId::new(2),
            [false],
            [],
            [],
        )
        .unwrap(),
    ));
    let src_structure = packed_fixture_structure(
        3,
        [
            (nonempty_domain, vec![1, 1, 1]),
            (mismatched, vec![1, 1, 1]),
        ],
    )
    .unwrap();
    let dst_key =
        all_codomain_fusion_tree_test_key([0, 0, 0], 0, [false, false, false], [0], [1, 1]);
    let dst_structure = packed_fixture_structure(3, [(dst_key, vec![1, 1, 1])]).unwrap();
    let src_space = vec![1, 1, 1];
    let dst_space = vec![1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(vec![1.0, 2.0], src_space, src_structure)
        .unwrap();
    let _dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0], dst_space, dst_structure).unwrap();

    let error = crate::build_all_codomain_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 1, 2], []),
        src.structure(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    let error = crate::build_all_codomain_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([0, 0, 2], []),
        src.structure(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );

    let error = crate::build_all_codomain_tree_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::transpose([0, 1, 2], []),
        src.structure(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        })
    );
}

#[test]
fn oriented_cold_and_warm_admission_rejects_equal_key_capability_changes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let storage = simple_su2_vertex_structure(1);
    let BlockKey::FusionTree(parent) = storage.block(0).unwrap().key() else {
        unreachable!()
    };
    let logical_keys = [FusionTreePairKey::pair(
        parent.domain_tree().clone(),
        parent.codomain_tree().clone(),
    )];
    let destination =
        Arc::new(packed_fixture_structure(2, [(logical_keys[0].clone(), vec![1, 1])]).unwrap());
    let operation = TreeTransformOperation::permute([], [1, 0]);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut rule = AdmissionCountingSu2Rule {
        nsymbol_calls: calls,
        fusion_style: None,
        braiding_style: None,
    };
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for _pass in 0..2 {
        // A fresh canonical parent per pass: the second pass must start cold.
        let storage = expert_copy(&storage);
        let destination = expert_copy(&destination);
        mark_canonical([storage.as_ref(), destination.as_ref()]);
        let mut cache = crate::tree_transform::TreeTransformPlanning::default();
        owner_activity();
        let mut compile = |cache: &mut crate::tree_transform::TreeTransformPlanning,
                           rule: &AdmissionCountingSu2Rule| {
            cache.resolve_tree_pair_oriented(
                engine,
                rule,
                &operation,
                &destination,
                &logical_keys,
                || Ok(&[0]),
                &storage,
                symtn_core::FusionTreePairOrientation::Adjoint,
                crate::tree_transform::OrientedBasisOrder::Storage,
                2,
                Ok,
            )
        };
        for warm in [false, true] {
            rule.fusion_style = Some(FusionStyleKind::Generic);
            assert_eq!(
                compile(&mut cache, &rule).unwrap_err(),
                OperationError::UnsupportedFusionStyle {
                    operation: Box::new(operation.clone()),
                    style: FusionStyleKind::Generic,
                }
            );
            rule.fusion_style = None;
            rule.braiding_style = Some(BraidingStyleKind::Anyonic);
            assert_eq!(
                compile(&mut cache, &rule).unwrap_err(),
                OperationError::UnsupportedBraidingStyle {
                    operation: Box::new(operation.clone()),
                    style: BraidingStyleKind::Anyonic,
                }
            );
            rule.braiding_style = None;
            let multiplicity_free_capability_validations_rec = probe::record();
            compile(&mut cache, &rule).unwrap();
            assert_eq!(
                multiplicity_free_capability_validations_rec
                    .get(Event::MultiplicityFreeCapabilityValidation),
                1
            );
            // What: capability checks run before the owner on cold and warm
            // calls alike; the warm call hits.
            let activity = owner_activity();
            assert_eq!(
                (activity.hits, activity.builds),
                (usize::from(warm), usize::from(!warm))
            );
        }
    }
}

#[test]
fn every_tree_transform_entry_reports_the_operation_fusion_style_error() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = simple_su2_vertex_structure(1);
    let operation = TreeTransformOperation::braid([1, 0], [], [0, 1], []);
    let generic = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::new(AtomicUsize::new(0)),
        fusion_style: Some(FusionStyleKind::Generic),
        braiding_style: None,
    };
    let rejected = |style| OperationError::UnsupportedFusionStyle {
        operation: Box::new(operation.clone()),
        style,
    };
    let mut cache = crate::tree_transform::TreeTransformPlanning::default();

    // What: each tensors-layer tree-transform entry rejects a style it
    // cannot lower with the operation-carrying tensors variant, never the
    // core variant wrapped in `OperationError::Core` (#1950).
    assert_eq!(
        crate::build_tree_pair_transform_group_plan(&generic, operation.clone(), &structure)
            .unwrap_err(),
        rejected(FusionStyleKind::Generic)
    );
    assert_eq!(
        crate::build_multiplicity_free_tree_pair_transform_group_plan(
            &generic,
            operation.clone(),
            &structure
        )
        .unwrap_err(),
        rejected(FusionStyleKind::Generic)
    );
    assert_eq!(
        crate::build_all_codomain_tree_transform_group_plan(
            &generic,
            operation.clone(),
            &structure
        )
        .unwrap_err(),
        rejected(FusionStyleKind::Generic)
    );
    assert_eq!(
        cache
            .resolve_tree_pair(engine, &generic, &operation, &structure, &structure, false)
            .unwrap_err(),
        rejected(FusionStyleKind::Generic)
    );
    assert_eq!(
        crate::build_unique_tree_pair_transform_group_plan(
            &SU2FusionRule,
            operation.clone(),
            &structure
        )
        .unwrap_err(),
        rejected(FusionStyleKind::Simple)
    );
    assert_eq!(
        crate::build_unique_all_codomain_tree_transform_group_plan(
            &SU2FusionRule,
            operation.clone(),
            &structure
        )
        .unwrap_err(),
        rejected(FusionStyleKind::Simple)
    );
    assert_eq!(generic.nsymbol_calls.load(Ordering::Relaxed), 0);
}
