use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use symtn_dense::Engine;

use symtn_core::{
    BlockKey, BlockStructure, BraidingStyleKind, DegeneracyStructure, FusionProductSpace,
    FusionRule, FusionStyleKind, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    MultiplicityIndex, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorStructure, SectorVec,
    U1FusionRule, U1Irrep,
};
use symtn_tensors::{
    build_tree_pair_transform_group_plan, BoundDynamicFusionMapSpace, RuleIdentity,
    TreeTransformBlockSpec, TreeTransformExecutionContext, TreeTransformOperation,
    TreeTransformStructure,
};

/// The canonical all-codomain space of `legs`, bound to `rule`: the source
/// an eager transform stages.
fn bound_source<R: MultiplicityFreeRigidSymbols>(
    rule: Arc<R>,
    legs: impl IntoIterator<Item = SectorLeg>,
) -> BoundDynamicFusionMapSpace<R> {
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new(legs), FusionProductSpace::new([]));
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(rule, homspace).unwrap()
}

/// The eager staging of `operation` on `source`: destination derivation plus
/// the completed-transformer resolution.
fn stage<R>(
    engine: &mut dyn Engine,
    context: &mut TreeTransformExecutionContext<f64, RuleIdentity>,
    source: &BoundDynamicFusionMapSpace<R>,
    operation: &TreeTransformOperation,
) -> TreeTransformStructure<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn_tensors::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let src_len = source.space().required_len().unwrap();
    context
        .stage_tree_transform_multiplicity_free_in(engine, source, None, src_len, operation)
        .unwrap()
        .structure()
        .clone()
}

/// The destination derivation alone, which the staging runs before it
/// resolves the structure.
fn derive<R: MultiplicityFreeRigidSymbols>(
    source: &BoundDynamicFusionMapSpace<R>,
    operation: &TreeTransformOperation,
) -> BoundDynamicFusionMapSpace<R> {
    source.transformed_multiplicity_free(operation).unwrap()
}

use crate::counting_alloc;

fn su2_f_move_structure() -> BlockStructure {
    let keys = [[0, 1], [2, 1]].map(|inner| {
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(1); 4],
                SectorId::new(0),
                [false; 4],
                inner.map(SectorId::new),
                [MultiplicityIndex::ONE; 3],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], SectorId::new(0), [], [], [])
                .unwrap(),
        ))
    });
    BlockStructure::from_parts(
        SectorStructure::from_keys(4, keys).unwrap(),
        DegeneracyStructure::packed_column_major(4, [vec![1; 4], vec![1; 4]]).unwrap(),
    )
    .unwrap()
}

#[derive(Clone)]
struct AdmissionCountingSu2Rule {
    nsymbol_calls: Arc<AtomicUsize>,
}

impl symtn_core::testing::Provider for AdmissionCountingSu2Rule {}

impl FusionRule for AdmissionCountingSu2Rule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        SU2FusionRule.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        SU2FusionRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        SU2FusionRule.braiding_style()
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

#[test]
fn rank_129_second_exact_warm_staging_resolves_without_allocation_or_provider_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    symtn_core::clear_structure_caches();

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
    };
    let vacuum = SU2FusionRule.vacuum();
    let source = bound_source(
        Arc::new(rule),
        (0..129).map(|_| SectorLeg::new([(vacuum, 1)], false)),
    );
    let operation = TreeTransformOperation::permute(0..129, []);
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let cold = stage(engine, &mut context, &source, &operation);
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    counting_alloc::start();
    let warm = stage(engine, &mut context, &source, &operation);
    let staged = counting_alloc::stop();
    let staged_calls = calls.swap(0, Ordering::Relaxed);
    counting_alloc::start();
    let derived = derive(&source, &operation);
    let derivation = counting_alloc::stop();
    let derivation_calls = calls.load(Ordering::Relaxed);
    std::hint::black_box(derived);

    eprintln!(
        "rank-129 warm staging: {} allocations, derivation alone {}",
        staged.calls, derivation.calls
    );
    // What: on an exact warm hit the staging's structure resolution — the
    // runtime-rank operation cloned into the completed-transformer lookup key
    // and the cached core bound to the caller's structures — adds no
    // allocation and no provider work to the destination derivation.
    assert!(Arc::ptr_eq(cold.replay_core(), warm.replay_core()));
    assert_eq!(staged.calls, derivation.calls);
    assert_eq!(staged_calls, derivation_calls);
}

/// Rank eight with `half_legs` spin-1/2 legs and vacuum legs elsewhere: the
/// complete layout has one block per vacuum-coupled tree of the spin-1/2
/// legs (1, 2, 5, 14 for 2, 4, 6, 8 of them).
fn rank_eight_su2_layout(half_legs: usize) -> BoundDynamicFusionMapSpace<SU2FusionRule> {
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let vacuum = SU2FusionRule.vacuum();
    let legs = (0..8)
        .map(|leg| SectorLeg::new([(if leg < half_legs { half } else { vacuum }, 1)], false))
        .collect::<Vec<_>>();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(legs.clone()),
        FusionProductSpace::new([]),
    );
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(Arc::new(SU2FusionRule), hom)
        .unwrap()
}

/// Cold allocation calls of the staging of `operation` on `source`, less
/// those of its destination derivation alone, each from cleared caches.
fn cold_resolution_allocations<R>(
    engine: &mut dyn Engine,
    source: &BoundDynamicFusionMapSpace<R>,
    operation: &TreeTransformOperation,
) -> (u64, u64, u64)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn_tensors::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    symtn_core::clear_structure_caches();
    counting_alloc::start();
    std::hint::black_box(derive(source, operation));
    let derivation = counting_alloc::stop();
    symtn_core::clear_structure_caches();
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    counting_alloc::start();
    let structure = stage(engine, &mut context, source, operation);
    let staged = counting_alloc::stop();
    std::hint::black_box(structure);
    (
        staged.calls - derivation.calls,
        staged.bytes.saturating_sub(derivation.bytes),
        staged.calls,
    )
}

/// Purpose: the cold ordered whole-block resolution grows at most linearly
/// in the source blocks: the calls each further block adds never exceed
/// what the second block added.
#[test]
fn cold_ordered_tree_pair_staging_grows_at_most_linearly_in_blocks() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();

    // A first cold run grows the process-wide registries' capacity, so the
    // rows below measure only the plan compilation itself.
    let operation = TreeTransformOperation::permute(0..8, []);
    cold_resolution_allocations(engine, &rank_eight_su2_layout(2), &operation);

    let rows = [2, 4, 6, 8].map(|half_legs| {
        let source = rank_eight_su2_layout(half_legs);
        let blocks = source.space().structure().block_count() as u64;
        (
            blocks,
            cold_resolution_allocations(engine, &source, &operation).0,
        )
    });
    let slope = |[(b0, r0), (b1, r1)]: [(u64, u64); 2]| (r1 - r0, b1 - b0);
    let (first_calls, first_blocks) = slope([rows[0], rows[1]]);
    for pair in rows.windows(2).skip(1) {
        let (calls, blocks) = slope([pair[0], pair[1]]);
        // calls / blocks <= first_calls / first_blocks
        assert!(
            calls * first_blocks <= first_calls * blocks,
            "per-block resolution cost grew: {rows:?} (blocks, calls)"
        );
    }
}

fn rank_one_u1_pair_structure(count: usize) -> BlockStructure {
    let keys = (0..count).map(|charge| {
        let sector = U1Irrep::new(charge as i32).sector_id();
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [sector], sector, [false], [], [])
                .unwrap(),
            FusionTreeKey::try_new_for_rule(&U1FusionRule, [sector], sector, [false], [], [])
                .unwrap(),
        ))
    });
    BlockStructure::from_parts(
        SectorStructure::from_keys(2, keys).unwrap(),
        DegeneracyStructure::packed_column_major(2, (0..count).map(|_| vec![1, 1])).unwrap(),
    )
    .unwrap()
}

/// Purpose: building a Unique rank-one U(1) plan costs the same at every
/// source block count, and binding it borrows the plan's coefficients.
#[test]
fn unique_rank_one_u1_plan_allocations_do_not_scale_with_source_blocks() {
    let _serial = crate::counting_alloc::serial();
    let mut one_block = None;
    for count in [1, 2, 4, 8, 16] {
        let structure = rank_one_u1_pair_structure(count);
        let operation = TreeTransformOperation::permute([0], [1]);
        counting_alloc::start();
        let plan =
            build_tree_pair_transform_group_plan(&U1FusionRule, operation, &structure).unwrap();
        let allocs = counting_alloc::stop();

        // What: every Unique source block is an inline Single and shares the
        // operation axis map, so plan construction costs what one block does.
        let one_block = *one_block.get_or_insert(allocs.calls);
        counting_alloc::assert_same_calls(
            &format!("source blocks 1 -> {count}"),
            one_block,
            allocs.calls,
        );
        assert_eq!(plan.specs().len(), count);

        let indexed_specs = (0..count)
            .map(|index| TreeTransformBlockSpec::single(index, index, 1.0).with_source_axes([0, 1]))
            .collect::<Vec<_>>();
        counting_alloc::start();
        let direct =
            TreeTransformStructure::compile_structures(&structure, &structure, &indexed_specs)
                .unwrap();
        let allocs = counting_alloc::stop();
        let direct_allocations = allocs.calls as usize;
        counting_alloc::start();
        let grouped = plan.compile_structures(&structure, &structure).unwrap();
        let allocs = counting_alloc::stop();
        let grouped_allocations = allocs.calls as usize;

        // What: resolving grouped Single entries owns one descriptor arena but
        // borrows coefficients and shared source axes from the plan.
        assert_eq!(
            grouped_allocations,
            direct_allocations + 1,
            "source_blocks={count}"
        );
        assert_eq!(grouped, direct);
    }
}

/// Purpose: binding a grouped Multi spec borrows its coefficient matrix
/// instead of copying it.
#[test]
fn grouped_multi_compile_borrows_plan_coefficient_matrix() {
    let _serial = crate::counting_alloc::serial();
    const BLOCKS: usize = 2;
    const COEFFICIENT_BYTES: usize = 64 * 1024;

    let structure = su2_f_move_structure();
    let keys = (0..BLOCKS)
        .map(|block| {
            structure
                .block(block)
                .unwrap()
                .key()
                .as_fusion_tree_pair()
                .unwrap()
                .clone()
        })
        .collect::<Vec<_>>();
    let coefficients = vec![[1_u8; COEFFICIENT_BYTES]; BLOCKS * BLOCKS];
    let grouped_spec = symtn_tensors::TreeTransformGroupBlockSpec::try_multi(
        keys.clone(),
        keys,
        coefficients.clone(),
    )
    .unwrap();
    let grouped_plan = symtn_tensors::TreeTransformGroupPlan::new(vec![grouped_spec]);
    let direct_specs = [TreeTransformBlockSpec::multi(
        (0..BLOCKS).collect(),
        (0..BLOCKS).collect(),
        coefficients,
    )];
    // A copy of the matrix allocates at least one coefficient's bytes; the
    // binding's own index and descriptor scratch is far smaller.
    let coefficient_sized = COEFFICIENT_BYTES..=usize::MAX;
    let (_, cold) = counting_alloc::measure_matching(coefficient_sized.clone(), || {
        grouped_plan
            .compile_structures(&structure, &structure)
            .unwrap()
    });
    let direct =
        TreeTransformStructure::compile_structures(&structure, &structure, &direct_specs).unwrap();
    let (grouped, warm) = counting_alloc::measure_matching(coefficient_sized, || {
        grouped_plan
            .compile_structures(&structure, &structure)
            .unwrap()
    });

    // What: neither the first nor a later binding copies the spec's matrix.
    assert_eq!(
        cold.matched_calls, 0,
        "cold binding copied coefficients: {cold:?}"
    );
    assert_eq!(
        warm.matched_calls, 0,
        "warm binding copied coefficients: {warm:?}"
    );
    assert_eq!(
        grouped.gathered_coefficients(),
        direct.gathered_coefficients()
    );
}

/// `[s, s, s] <- [d]` with `s = {1/2, 1}` and the domain leg's sectors `d`.
fn su2_three_to_one_structure(domain_twice_spins: &[usize]) -> Arc<BlockStructure> {
    let leg = |twice_spins: &[usize]| {
        SectorLeg::new(
            twice_spins
                .iter()
                .map(|&j| (SU2Irrep::from_twice_spin(j).sector_id(), 1)),
            false,
        )
    };
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(&[1, 2]), leg(&[1, 2]), leg(&[1, 2])]),
        FusionProductSpace::new([leg(domain_twice_spins)]),
    );
    let keys = hom
        .fusion_tree_keys(&SU2FusionRule)
        .iter()
        .cloned()
        .map(BlockKey::from)
        .collect::<Vec<_>>();
    let count = keys.len();
    Arc::new(
        BlockStructure::from_parts(
            SectorStructure::from_keys(4, keys).unwrap(),
            DegeneracyStructure::packed_column_major(4, (0..count).map(|_| vec![1usize; 4]))
                .unwrap(),
        )
        .unwrap(),
    )
}

fn convert_spec<T: Copy>(
    spec: &symtn_tensors::TreeTransformGroupBlockSpec<f64>,
    convert: fn(f64) -> T,
) -> symtn_tensors::TreeTransformGroupBlockSpec<T> {
    let coefficients = spec
        .recoupling_coefficients_dst_src()
        .iter()
        .map(|&value| convert(value))
        .collect::<Vec<_>>();
    let converted = if coefficients.len() == 1 {
        symtn_tensors::TreeTransformGroupBlockSpec::single(
            spec.dst_keys()[0].clone(),
            spec.src_keys()[0].clone(),
            coefficients[0],
        )
    } else {
        symtn_tensors::TreeTransformGroupBlockSpec::try_multi(
            spec.dst_keys().iter().cloned(),
            spec.src_keys().iter().cloned(),
            coefficients,
        )
        .unwrap()
    };
    match spec.source_axes() {
        Some(axes) => converted.with_source_axes(axes.iter().copied()),
        None => converted,
    }
}

/// The `c` plan a warm Runtime assembles after a sector change: every group
/// `a` already built is the cached spec handle (`plan.rs` clones group specs),
/// and only the added groups are new. Returns the plan, the coefficient count
/// of the added groups, and the number of Single (1x1) specs.
fn reused_sector_change_plan<T: Copy>(
    a: &symtn_tensors::TreeTransformGroupPlan<f64>,
    c: &symtn_tensors::TreeTransformGroupPlan<f64>,
    convert: fn(f64) -> T,
) -> (symtn_tensors::TreeTransformGroupPlan<T>, usize, usize) {
    let cached = a
        .specs()
        .iter()
        .map(|spec| convert_spec(spec, convert))
        .collect::<Vec<_>>();
    let mut changed_coefficients = 0;
    let mut singles = 0;
    let specs = c
        .specs()
        .iter()
        .map(|spec| {
            if spec.recoupling_coefficients_dst_src().len() == 1 {
                singles += 1;
            }
            cached
                .iter()
                .find(|hit| {
                    hit.group_key() == spec.group_key()
                        && hit.src_keys() == spec.src_keys()
                        && hit.dst_keys() == spec.dst_keys()
                })
                .cloned()
                .unwrap_or_else(|| {
                    changed_coefficients += spec.recoupling_coefficients_dst_src().len();
                    convert_spec(spec, convert)
                })
        })
        .collect();
    (
        symtn_tensors::TreeTransformGroupPlan::new(specs),
        changed_coefficients,
        singles,
    )
}

#[test]
fn sector_change_binding_copies_coefficients_of_changed_groups_only() {
    let _serial = crate::counting_alloc::serial();
    // Why only operations that keep the space: the binding needs the
    // destination structure, and these map `[s, s, s] <- [d]` onto itself.
    for operation in [
        TreeTransformOperation::permute([0, 2, 1], [3]),
        TreeTransformOperation::braid([0, 2, 1], [3], [0, 1, 2], [3]),
    ] {
        assert_sector_change_binding_copies_only_changed_groups(operation);
    }
}

fn assert_sector_change_binding_copies_only_changed_groups(operation: TreeTransformOperation) {
    let a = su2_three_to_one_structure(&[1, 2]);
    let c = su2_three_to_one_structure(&[1, 2, 3]);
    let plan_a =
        build_tree_pair_transform_group_plan(&SU2FusionRule, operation.clone(), &a).unwrap();
    let plan_c = build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &c).unwrap();

    let bind_bytes = |bind: &dyn Fn()| {
        counting_alloc::start();
        bind();
        let allocs = counting_alloc::stop();
        allocs.bytes as usize
    };
    // Why compare two scalar widths: the bound plans have identical keys,
    // blocks and layouts, so their binding allocations differ only by the
    // coefficient values the binding stores, 8 extra bytes per copied value.
    let (real, changed, singles) = reused_sector_change_plan(&plan_a, &plan_c, |value| value);
    let (complex, _, _) = reused_sector_change_plan(&plan_a, &plan_c, |value| {
        num_complex::Complex64::new(value, 0.0)
    });
    let total = plan_c
        .specs()
        .iter()
        .map(|spec| spec.recoupling_coefficients_dst_src().len())
        .sum::<usize>();
    assert!(
        changed > 0 && total > changed + singles,
        "fixture must reuse matrix groups"
    );

    let bind_real = || {
        std::hint::black_box(
            real.compile_shared_structures_with_storage_conjugation(
                Arc::clone(&c),
                Arc::clone(&c),
                false,
            )
            .unwrap(),
        );
    };
    let bind_complex = || {
        std::hint::black_box(
            complex
                .compile_shared_structures_with_storage_conjugation(
                    Arc::clone(&c),
                    Arc::clone(&c),
                    false,
                )
                .unwrap(),
        );
    };
    let real_bytes = bind_bytes(&bind_real);
    let complex_bytes = bind_bytes(&bind_complex);
    let copied = complex_bytes.saturating_sub(real_bytes) / 8;

    // What: a plan miss copies coefficients of the groups it built plus the
    // Single scalars, never the reused groups' recoupling matrices.
    assert!(
        copied <= changed + singles,
        "copied={copied} coefficients, changed={changed}, singles={singles}, total={total}"
    );

    // What: binding the already-bound plan again (a structure-tier miss on a
    // plan-tier hit) shares the plan's payload and copies no coefficient.
    let rebound_real = bind_bytes(&bind_real);
    let rebound_complex = bind_bytes(&bind_complex);
    assert_eq!(
        rebound_complex, rebound_real,
        "a re-binding must not store any coefficient"
    );
}

/// Test view of the logical coefficient payload (an explicit copy).
trait GatheredCoefficients<T> {
    fn gathered_coefficients(&self) -> Vec<T>;
}

impl<T: Copy> GatheredCoefficients<T> for symtn_tensors::TreeTransformStructure<T> {
    fn gathered_coefficients(&self) -> Vec<T> {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }
}

/// U(1) on six `q ∈ {0, 1}` legs split three and three, every block of
/// degeneracy one.
fn rank_six_u1_layout() -> BoundDynamicFusionMapSpace<U1FusionRule> {
    let leg = || {
        SectorLeg::new(
            [0, 1].map(|charge| (U1Irrep::new(charge).sector_id(), 1)),
            false,
        )
    };
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new((0..3).map(|_| leg())),
        FusionProductSpace::new((0..3).map(|_| leg())),
    );
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(Arc::new(U1FusionRule), hom)
        .unwrap()
}

#[test]
fn unique_cache_three_misses_reuse_the_context_tree_pair_scratch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the Unique compile's tree-pair scratch (three trees of three
    // buffers) belongs to the execution context, so only a context's first
    // cache-3 miss allocates it, for the direct and the lazy-adjoint compile.
    const SCRATCH_BUFFERS: u64 = 9;
    let parent = rank_six_u1_layout();
    let adjoint = symtn_tensors::adjoint_bound_space_dyn(&parent).unwrap();
    let parent_len = parent.space().required_len().unwrap();
    let operation = TreeTransformOperation::permute([5, 4, 3], [2, 1, 0]);
    let mut miss = |context: &mut TreeTransformExecutionContext<f64, RuleIdentity>, lazy: bool| {
        symtn_core::clear_structure_caches();
        let (logical, adjoint_of) = if lazy {
            (&adjoint, Some(&parent))
        } else {
            (&parent, None)
        };
        counting_alloc::start();
        let staged = context
            .stage_tree_transform_multiplicity_free_in(
                engine, logical, adjoint_of, parent_len, &operation,
            )
            .unwrap();
        let allocs = counting_alloc::stop();
        std::hint::black_box(staged);
        allocs.calls
    };
    for lazy in [false, true] {
        // Why a throwaway context first: the first miss in the process also
        // grows the cleared caches' tables, which `clear` keeps.
        miss(&mut TreeTransformExecutionContext::default(), lazy);
        let mut context = TreeTransformExecutionContext::default();
        let first = miss(&mut context, lazy);
        let second = miss(&mut context, lazy);
        let third = miss(&mut context, lazy);
        eprintln!("lazy={lazy}: misses {first}, {second}, {third}");
        assert_eq!(first, second + SCRATCH_BUFFERS, "lazy={lazy}");
        assert_eq!(second, third, "lazy={lazy}");
    }
}
