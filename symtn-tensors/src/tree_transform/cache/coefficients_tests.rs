//! The process-global composed-coefficient owner: key
//! completeness, staged and epoch-guarded publication, degeneracy-only reuse
//! and output identity against the uncached eager producer.

use std::sync::Arc;

use num_complex::Complex64;
use symtn_core::{
    BlockKey, BlockStructure, FusionProductSpace, FusionTreeHomSpace, FusionTreePairKey,
    FusionTreePairOrientation, RuleIdentity, SU2FusionRule, SU2Irrep, SectorLeg,
};

use super::super::{TransformerMode, TreeTransformPlanning, TreeTransformScope};
use super::*;
use crate::probe::take_coefficient_group_activity;
use crate::{BoundDynamicFusionMapSpace, TreeTransformGroupBlockSpec, TreeTransformOperation};

/// A rule identity no other test publishes under.
struct KeyMarker;

/// The canonical SU(2) `V^2 ← V^2` structure, `V = 0 ⊕ ½ ⊕ 1`, each sector
/// `degeneracy`-fold: a group holds every coupled sector of its legs.
fn su2_structure(degeneracy: usize) -> Arc<BlockStructure> {
    let leg = SectorLeg::new(
        [0, 1, 2].map(|spin| (SU2Irrep::from_twice_spin(spin).sector_id(), degeneracy)),
        false,
    );
    Arc::clone(
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(SU2FusionRule),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone(), leg.clone()]),
                FusionProductSpace::new([leg.clone(), leg]),
            ),
        )
        .unwrap()
        .space()
        .structure(),
    )
}

/// Each source group's key and its ordered tree pairs.
fn group_keys(structure: &BlockStructure) -> Vec<(FusionTreeGroupKey, Vec<FusionTreePairKey>)> {
    structure
        .fusion_tree_group_slice()
        .iter()
        .map(|group| {
            let keys = group
                .block_indices()
                .iter()
                .map(|&index| match structure.block(index).unwrap().key() {
                    BlockKey::FusionTree(key) => key.clone(),
                    _ => unreachable!("fusion-tree structure"),
                })
                .collect();
            (group.group_key().clone(), keys)
        })
        .collect()
}

fn reuse<T: 'static + Send + Sync>(
    rule: RuleIdentity,
    mode: TransformerMode,
    scope: TreeTransformScope,
    operation: &TreeTransformOperation,
    orientation: FusionTreePairOrientation,
) -> CoefficientGroupReuse<T> {
    CoefficientGroupReuse::new(rule, mode, scope, operation, orientation)
}

fn lookup_one<T: 'static + Send + Sync>(
    reuse: &CoefficientGroupReuse<T>,
    (group_key, keys): &(FusionTreeGroupKey, Vec<FusionTreePairKey>),
) -> GroupSlot<T> {
    let refs = keys.iter().collect::<Vec<_>>();
    reuse.lookup(&[(group_key, &refs)]).pop().unwrap()
}

fn stage_one<T: 'static + Send + Sync + Clone>(
    reuse: &CoefficientGroupReuse<T>,
    group: &(FusionTreeGroupKey, Vec<FusionTreePairKey>),
    coefficient: T,
) -> Arc<CoefficientGroupEntry<T>> {
    let GroupSlot::Miss(hash) = lookup_one(reuse, group) else {
        panic!("the group must start absent");
    };
    let refs = group.1.iter().collect::<Vec<_>>();
    let specs = vec![TreeTransformGroupBlockSpec::single(
        group.1[0].clone(),
        group.1[0].clone(),
        coefficient,
    )];
    reuse.stage(hash, (&group.0, &refs), specs)
}

#[test]
fn every_key_determinant_splits_entries_and_equal_keys_hit() {
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = su2_structure(1);
    let groups = group_keys(&structure);
    let multi = groups
        .iter()
        .find(|(_, keys)| keys.len() > 1)
        .expect("an SU(2) group with several trees")
        .clone();
    let rule = RuleIdentity::of_type::<KeyMarker>();
    let operation = TreeTransformOperation::braid([1, 0], [2, 3], [0, 1], [2, 3]);
    let direct = FusionTreePairOrientation::Direct;
    let base = || {
        reuse::<f64>(
            rule.clone(),
            TransformerMode::MultiplicityFree,
            TreeTransformScope::TreePair,
            &operation,
            direct,
        )
    };
    let epoch = symtn_core::core_reset_epoch();
    let published = base();
    let entry = stage_one(&published, &multi, 2.0);
    published.into_pending().publish(epoch);

    // What: an equal key hits the very published entry.
    match lookup_one(&base(), &multi) {
        GroupSlot::Hit(hit) => assert!(Arc::ptr_eq(&hit, &entry)),
        GroupSlot::Miss(_) => panic!("an equal key must hit"),
    }
    let misses = |handle: CoefficientGroupReuse<f64>| {
        matches!(lookup_one(&handle, &multi), GroupSlot::Miss(_))
    };
    // What: every keyed determinant, varied alone, misses.
    assert!(misses(reuse(
        RuleIdentity::of_type::<SU2FusionRule>(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &operation,
        direct,
    )));
    assert!(misses(reuse(
        rule.clone(),
        TransformerMode::CheckedGeneric,
        TreeTransformScope::TreePair,
        &operation,
        direct,
    )));
    assert!(misses(reuse(
        rule.clone(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TraceTerms {
            open_codomain_rank: 0,
        },
        &operation,
        direct,
    )));
    assert!(misses(reuse(
        rule.clone(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &TreeTransformOperation::permute([1, 0], [2, 3]),
        direct,
    )));
    assert!(misses(reuse(
        rule.clone(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &TreeTransformOperation::braid([1, 0], [2, 3], [1, 0], [2, 3]),
        direct,
    )));
    assert!(misses(reuse(
        rule.clone(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &operation,
        FusionTreePairOrientation::Adjoint,
    )));
    // The coefficient type: a Complex64 lookup of the same group misses.
    let complex = reuse::<Complex64>(
        rule.clone(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &operation,
        direct,
    );
    assert!(matches!(lookup_one(&complex, &multi), GroupSlot::Miss(_)));
    // The ordered tree subset: a reordered or partial group misses.
    let mut reordered = multi.clone();
    reordered.1.reverse();
    assert!(matches!(
        lookup_one(&base(), &reordered),
        GroupSlot::Miss(_)
    ));
    let partial = (multi.0.clone(), multi.1[..1].to_vec());
    assert!(matches!(lookup_one(&base(), &partial), GroupSlot::Miss(_)));
}

#[test]
fn staged_groups_publish_only_when_flushed_in_their_epoch() {
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let structure = su2_structure(2);
    let groups = group_keys(&structure);
    struct EpochMarker;
    let handle = || {
        reuse::<f64>(
            RuleIdentity::of_type::<EpochMarker>(),
            TransformerMode::CheckedGeneric,
            TreeTransformScope::TreePair,
            &TreeTransformOperation::permute([1, 0], [2, 3]),
            FusionTreePairOrientation::Direct,
        )
    };
    take_coefficient_group_activity();

    // What: groups staged by a transaction that never publishes stay
    // invisible.
    let abandoned = handle();
    stage_one(&abandoned, &groups[0], 1.0);
    drop(abandoned.into_pending());
    assert!(matches!(
        lookup_one(&handle(), &groups[0]),
        GroupSlot::Miss(_)
    ));

    // What: a flush whose epoch a reset overtook is refused.
    let mut stale = CheckedPendingCoefficients {
        pending: PendingCoefficientGroups::default(),
        transformers: Vec::new(),
        traces: Vec::new(),
        epoch: symtn_core::core_reset_epoch().wrapping_add(2),
    };
    let built = handle();
    stage_one(&built, &groups[0], 1.0);
    stale.stage(built.into_pending());
    stale.flush();
    assert!(matches!(
        lookup_one(&handle(), &groups[0]),
        GroupSlot::Miss(_)
    ));

    // What: a current flush publishes every staged group.
    let mut current = CheckedPendingCoefficients::new();
    let built = handle();
    for group in &groups {
        stage_one(&built, group, 1.0);
    }
    current.stage(built.into_pending());
    current.flush();
    for group in &groups {
        assert!(matches!(lookup_one(&handle(), group), GroupSlot::Hit(_)));
    }
    let activity = take_coefficient_group_activity();
    assert_eq!(activity.publications, 1 + groups.len());
}

#[test]
fn an_entry_charges_its_recoupling_matrix_and_trees() {
    let structure = su2_structure(1);
    let multi = group_keys(&structure)
        .into_iter()
        .find(|(_, keys)| keys.len() > 1)
        .unwrap();
    let width = multi.1.len();
    let spec = TreeTransformGroupBlockSpec::try_multi(
        multi.1.clone(),
        multi.1.clone(),
        vec![0.5_f64; width * width],
    )
    .unwrap();
    let handle = reuse::<f64>(
        RuleIdentity::of_type::<KeyMarker>(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &TreeTransformOperation::permute([1, 0], [2, 3]),
        FusionTreePairOrientation::Direct,
    );
    let refs = multi.1.iter().collect::<Vec<_>>();
    let mut arena = symtn_core::DedupTreeArenaBuilder::default();
    let (key, _) = handle
        .context
        .rehomed_key(0, (&multi.0, &refs), &mut arena, std::iter::empty());
    let bytes =
        charged_key_bytes(&key).saturating_add(charged_entry_bytes(std::slice::from_ref(&spec)));
    // What: the n x n matrix, the two key lists and the entry's tree arena
    // are inside the charge.
    let floor = width * width * core::mem::size_of::<f64>()
        + 2 * width * core::mem::size_of::<FusionTreePairKey>()
        + key.group_key.codomain_tree().arena_retained_bytes()
        + ENTRY_OVERHEAD_BYTES;
    assert!(bytes > floor, "{bytes} <= {floor}");

    // What: the key charge moves with its entry arena, byte for byte. The
    // same key over a lone two-row group arena differs only in that arena.
    let lone = CoefficientGroupKey {
        context: Arc::clone(&key.context),
        group_key: FusionTreeGroupKey::new(
            multi.0.codomain_uncoupled().iter().copied(),
            multi.0.domain_uncoupled().iter().copied(),
            multi.0.codomain_is_dual().iter().copied(),
            multi.0.domain_is_dual().iter().copied(),
        ),
        src_keys: key.src_keys.clone(),
        hash: key.hash,
    };
    let arena = |key: &CoefficientGroupKey| key.group_key.codomain_tree().arena_retained_bytes();
    assert_ne!(arena(&key), arena(&lone));
    assert_eq!(
        charged_key_bytes(&key) as i64 - charged_key_bytes(&lone) as i64,
        arena(&key) as i64 - arena(&lone) as i64,
        "the key charge must include its entry arena"
    );
}

/// Standalone context (no Runtime): a degeneracy-only change is a cache-3
/// miss whose every source group hits cache 4, the result equals the
/// uncached eager producer bit for bit, and the recoupling thread count and
/// storage conjugation do not split entries.
#[test]
fn degeneracy_only_change_recomposes_no_group_and_matches_the_uncached_producer() {
    // Unlocked constructors can evict groups even while CACHE_TEST_LOCK is held.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_TENSORS_COEFFICIENT_DEGENERACY_REUSE_ISOLATED",
        "tree_transform::cache::coefficients::tests::degeneracy_only_change_recomposes_no_group_and_matches_the_uncached_producer",
    ) {
        return;
    }
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let operation = TreeTransformOperation::braid([1, 0], [2, 3], [1, 0], [2, 3]);
    // Swapping two equal legs maps `V^2 ← V^2` onto itself.
    let layouts = |degeneracy: usize| {
        let source = su2_structure(degeneracy);
        (Arc::clone(&source), source)
    };
    let mut planning = TreeTransformPlanning::default();
    let engine = &mut symtn_operations::host_pool::test_engine(4);
    // Degeneracies no other test uses, so no completed transformer of
    // these layouts is resident.
    let (dst, src) = layouts(23);
    let group_count = src.fusion_tree_group_slice().len();
    take_coefficient_group_activity();
    planning
        .resolve_tree_pair(engine, &SU2FusionRule, &operation, &dst, &src, false)
        .unwrap();
    let cold = take_coefficient_group_activity();
    assert_eq!(cold.hits + cold.misses, group_count);
    assert_eq!(cold.publications, cold.misses);

    for (degeneracy, storage_conjugate) in [(29, false), (31, true)] {
        let (dst, src) = layouts(degeneracy);
        let warm = planning
            .resolve_tree_pair(
                engine,
                &SU2FusionRule,
                &operation,
                &dst,
                &src,
                storage_conjugate,
            )
            .unwrap();
        let activity = take_coefficient_group_activity();
        assert_eq!(
            (activity.hits, activity.misses, activity.publications),
            (group_count, 0, 0),
            "degeneracy {degeneracy}, conjugate {storage_conjugate}"
        );
        let eager =
            crate::build_tree_pair_transform_group_plan(&SU2FusionRule, operation.clone(), &src)
                .unwrap()
                .compile_shared_structures_with_storage_conjugation(
                    Arc::clone(&dst),
                    Arc::clone(&src),
                    storage_conjugate,
                )
                .unwrap();
        assert_eq!(warm, eager);
        let bits = |structure: &crate::TreeTransformStructure<f64>| {
            let mut coefficients = Vec::new();
            structure.gather_recoupling_coefficients_into(&mut coefficients);
            coefficients
                .iter()
                .map(|value: &f64| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&warm), bits(&eager));
    }
}

/// Degeneracy-only bindings of one cached group set share one
/// coefficient payload allocation (so a device uploads it once), with the
/// values of an uncached build; another operation's group set does not.
#[test]
fn degeneracy_only_bindings_of_one_group_set_share_one_coefficient_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // Hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let operation = TreeTransformOperation::braid([1, 0], [2, 3], [1, 0], [2, 3]);
    let mut planning = TreeTransformPlanning::default();
    // Degeneracies no other test uses, so each binding is a cache-3 miss.
    let mut bind = |degeneracy: usize, operation: &TreeTransformOperation| {
        let source = su2_structure(degeneracy);
        planning
            .resolve_tree_pair(engine, &SU2FusionRule, operation, &source, &source, false)
            .unwrap()
    };
    let first = bind(43, &operation);
    let second = bind(47, &operation);
    take_coefficient_group_activity();
    let third = bind(53, &operation);
    assert_eq!(take_coefficient_group_activity().misses, 0);

    assert!(first.shares_coefficients_with(&second));
    assert!(second.shares_coefficients_with(&third));
    let bits = |structure: &crate::TreeTransformStructure<f64>| {
        let mut coefficients = Vec::new();
        structure.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let source = su2_structure(47);
    let eager = crate::build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &source)
        .unwrap()
        .compile_shared_structures_with_storage_conjugation(Arc::clone(&source), source, false)
        .unwrap();
    assert!(!eager.shares_coefficients_with(&second));
    assert_eq!(bits(&second), bits(&eager));

    let other = TreeTransformOperation::permute([1, 0], [2, 3]);
    assert!(!bind(43, &other).shares_coefficients_with(&first));
}

#[test]
fn a_staged_entry_owns_one_arena_for_its_key_and_value() {
    let structure = su2_structure(1);
    let multi = group_keys(&structure)
        .into_iter()
        .find(|(_, keys)| keys.len() > 1)
        .unwrap();
    let handle = reuse::<f64>(
        RuleIdentity::of_type::<KeyMarker>(),
        TransformerMode::MultiplicityFree,
        TreeTransformScope::TreePair,
        &TreeTransformOperation::permute([0, 1], [3, 2]),
        FusionTreePairOrientation::Direct,
    );
    let width = multi.1.len();
    let spec = TreeTransformGroupBlockSpec::try_multi(
        multi.1.clone(),
        multi.1.clone(),
        vec![0.5_f64; width * width],
    )
    .unwrap();
    let refs = multi.1.iter().collect::<Vec<_>>();
    let entry = handle.stage(0, (&multi.0, &refs), vec![spec]);
    let pending = handle.into_pending();
    let (key, _, _) = &pending.groups[0];

    // What: the key's group and source pairs and the value's destination and
    // source pairs all point into one arena (charged once), never into the
    // source structure's, and are equal by content (single-arena ownership).
    let arena = key.group_key.codomain_tree();
    assert!(arena.shares_arena(key.group_key.domain_tree()));
    let spec = &entry.specs()[0];
    for pair in key
        .src_keys
        .iter()
        .chain(spec.dst_keys())
        .chain(spec.src_keys())
    {
        assert!(arena.shares_arena(pair.codomain_tree()));
        assert!(arena.shares_arena(pair.domain_tree()));
    }
    assert!(!arena.shares_arena(multi.1[0].codomain_tree()));
    assert_eq!(key.group_key, multi.0);
    assert_eq!(&*key.src_keys, &multi.1[..]);
    assert_eq!(spec.dst_keys(), &multi.1[..]);
}

#[test]
fn a_staged_trace_entry_owns_one_arena_for_its_key_and_destinations() {
    let structure = su2_structure(1);
    let multi = group_keys(&structure)
        .into_iter()
        .find(|(_, keys)| keys.len() > 1)
        .unwrap();
    let mut handle = TraceTermReuse::<f64>::new(
        RuleIdentity::of_type::<KeyMarker>(),
        TransformerMode::MultiplicityFree,
        TreeTransformOperation::permute([0, 1], [3, 2]),
        FusionTreePairOrientation::Direct,
        1,
    );
    let width = multi.1.len();
    let columns = BlockSourceColumns::new(
        multi.1.clone().into_boxed_slice(),
        Box::new([0, width]),
        (0..width).map(|row| (row, 1.0)).collect(),
    );
    let refs = multi.1.iter().collect::<Vec<_>>();
    let entry = handle.stage(0, (&multi.0, &refs), columns);
    let pending = handle.into_pending();
    let (key, _, _) = &pending.groups[0];

    // What: the key and every destination of a trace entry point into one
    // arena (charged once), never into the source structure's (single-arena
    // ownership), and keep their content.
    let arena = key.group_key.codomain_tree();
    assert!(arena.shares_arena(key.group_key.domain_tree()));
    for pair in key.src_keys.iter().chain(entry.destinations()) {
        assert!(arena.shares_arena(pair.codomain_tree()));
        assert!(arena.shares_arena(pair.domain_tree()));
    }
    assert!(!arena.shares_arena(multi.1[0].codomain_tree()));
    assert_eq!(entry.destinations(), &multi.1[..]);
}
