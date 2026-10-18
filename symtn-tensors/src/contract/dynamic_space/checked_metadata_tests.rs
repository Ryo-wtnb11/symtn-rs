use super::*;
use crate::probe::{self, Event};
use crate::test_support::CACHE_TEST_LOCK;
use std::cell::Cell;
use symtn_core::{
    structure_cache_info, FermionParityFusionRule, FusionAlgebraError, FusionProductSpace,
    Fz2SectorLayout, PackedProductCodec, ProductFusionRule, ProductSectorCodec,
    ProductSectorLayout, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, StructureCacheKind,
    Su2SectorLayout, U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
};

type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

thread_local! {
    static PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn rule() -> TripleRule {
    TripleRule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    )
}

fn sector(parity: usize, charge: i32, twice_spin: usize) -> SectorId {
    TripleCodec::encode(
        Fz2U1Codec::encode(
            Z2Irrep::new(parity as u8).sector_id(),
            U1Irrep::new(charge).sector_id(),
        ),
        SU2Irrep::from_twice_spin(twice_spin).sector_id(),
    )
}

fn homspace() -> FusionTreeHomSpace {
    let vacuum = sector(0, 0, 0);
    let charged = sector(1, 1, 1);
    let leg = |dual| SectorLeg::new([(vacuum, 1), (charged, 1)], dual);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    )
}

fn counting_primer(
    rule: &TripleRule,
    request: MetadataRequest<'_>,
) -> Result<MetadataOutput, OperationError> {
    PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
    checked_metadata_dispatcher(rule, request)
}

fn reset_primer_calls() {
    PRIMER_CALLS.with(|calls| calls.set(0));
}

fn primer_calls() -> usize {
    PRIMER_CALLS.with(Cell::get)
}

fn source(rule: &TripleRule) -> DynamicFusionMapSpace {
    let homspace = homspace();
    checked_layout_primer(rule, &homspace).unwrap();
    let count = homspace.fusion_tree_keys(rule).len();
    DynamicFusionMapSpace::from_degeneracy_shapes(rule, homspace, vec![vec![1; 4]; count]).unwrap()
}

fn shapes_from_tree_keys<R>(rule: &R, homspace: &FusionTreeHomSpace) -> Vec<Vec<usize>>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    homspace
        .prepare_fusion_tree_layout_checked(rule)
        .unwrap()
        .commit()
        .iter()
        .map(|key| {
            homspace
                .codomain()
                .legs()
                .iter()
                .chain(homspace.domain().legs())
                .zip(
                    key.codomain_uncoupled()
                        .iter()
                        .chain(key.domain_uncoupled()),
                )
                .map(|(leg, &sector)| leg.degeneracy(sector).unwrap())
                .collect()
        })
        .collect()
}

fn layout_snapshot(
    space: &DynamicFusionMapSpace,
) -> Vec<(BlockKey, Vec<usize>, Vec<usize>, usize)> {
    (0..space.structure().block_count())
        .map(|index| {
            let block = space.structure().block(index).unwrap();
            (
                block.key().clone(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
        })
        .collect()
}

fn assert_final_homspace_matches_shape_oracle<R>(provider: Arc<R>, homspace: FusionTreeHomSpace)
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
{
    let shapes = shapes_from_tree_keys(provider.as_ref(), &homspace);
    let oracle = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        homspace.clone(),
        shapes,
    )
    .unwrap();
    let expected_layout = layout_snapshot(oracle.space());
    let expected_len = oracle.space().required_len().unwrap();

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let legacy_shape_path_builds_rec = probe::record();
    let scratch_structure_builds_rec = probe::record();
    let expected_rule = provider.rule_identity();
    let actual = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();

    assert!(Arc::ptr_eq(actual.provider_arc(), &provider));
    assert_eq!(
        actual.space().admission().rule_identity(),
        Some(&expected_rule)
    );
    assert_eq!(layout_snapshot(actual.space()), expected_layout);
    assert_eq!(actual.space().required_len().unwrap(), expected_len);
    assert_eq!(
        legacy_shape_path_builds_rec.get(Event::LegacyShapePathBuild),
        0
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn derived_factor_shapes_use_one_authority_key_build() {
    // What: each cold SVD/QR/EIGH-style output derives its shapes and
    // storage grid from one authority-selected fusion-tree key set.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let rule = Arc::new(rule());
    let source = source(rule.as_ref());
    let authority = BoundDynamicFusionMapSpace::bind_multiplicity_free(source, rule)
        .unwrap()
        .with_test_layout_primer(counting_primer);

    for _operation in ["svd_compact", "qr_compact", "eigh_full"] {
        reset_primer_calls();
        let shape_builds = Cell::new(0usize);
        let derived = authority
            .derive_from_fusion_tree_shapes(homspace(), |keys| {
                shape_builds.set(shape_builds.get() + 1);
                Ok(keys.iter().map(|_| vec![1; 4]).collect::<Vec<_>>())
            })
            .unwrap();
        assert_eq!(primer_calls(), 1);
        assert_eq!(shape_builds.get(), 1);
        assert!(derived.space().structure().block_count() > 0);
    }
}

#[test]
fn final_derived_layout_selects_authority_keys_once() {
    // What: a final built-in layout derives both degeneracy shapes and
    // storage ordering from one authority-selected key Arc.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let rule = Arc::new(rule());
    let authority = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        source(rule.as_ref()),
        Arc::clone(&rule),
    )
    .unwrap()
    .with_test_layout_primer(counting_primer);

    reset_primer_calls();
    let derived = authority.derive_from_final_homspace(homspace()).unwrap();
    assert_eq!(primer_calls(), 1);
    let encoded = DynamicFusionMapSpace::from_final_homspace(rule.as_ref(), homspace()).unwrap();
    assert_eq!(derived.space(), &encoded);
}

#[test]
fn final_homspace_layout_matches_shape_oracle_across_supported_rules() {
    // What: U1, SU2, fZ2, their supported products, and scalar/vector
    // boundaries produce the exact shape-oracle block layout without
    // entering the per-tree shape or scratch-cache path.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let u1_minus = U1Irrep::new(-1).sector_id();
    let u1_zero = U1Irrep::new(0).sector_id();
    let u1_plus = U1Irrep::new(1).sector_id();
    let u1_codomain = SectorLeg::new([(u1_minus, 2), (u1_zero, 1), (u1_plus, 3)], false);
    let u1_domain = SectorLeg::new([(u1_minus, 4), (u1_zero, 2), (u1_plus, 1)], true);
    assert_final_homspace_matches_shape_oracle(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_codomain]),
            FusionProductSpace::new([u1_domain]),
        ),
    );

    let su2_zero = SU2Irrep::from_twice_spin(0).sector_id();
    let su2_half = SU2Irrep::from_twice_spin(1).sector_id();
    let su2_left = SectorLeg::new([(su2_zero, 1), (su2_half, 2)], false);
    let su2_right = SectorLeg::new([(su2_zero, 3), (su2_half, 1)], true);
    assert_final_homspace_matches_shape_oracle(
        Arc::new(SU2FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([su2_left.clone(), su2_right.clone()]),
            FusionProductSpace::new([su2_right, su2_left]),
        ),
    );

    let even = Z2Irrep::EVEN.sector_id();
    let odd = Z2Irrep::ODD.sector_id();
    let fz2_left = SectorLeg::new([(even, 1), (odd, 3)], false);
    let fz2_right = SectorLeg::new([(even, 2), (odd, 1)], true);
    assert_final_homspace_matches_shape_oracle(
        Arc::new(FermionParityFusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([fz2_left.clone(), fz2_right.clone()]),
            FusionProductSpace::new([fz2_right, fz2_left]),
        ),
    );

    // The nested fixture covers both ProductFusionRule levels:
    // fZ2 x U1 and (fZ2 x U1) x SU2.
    assert_final_homspace_matches_shape_oracle(Arc::new(rule()), homspace());
    assert_final_homspace_matches_shape_oracle(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
    );
    assert_final_homspace_matches_shape_oracle(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1_zero, 5)], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
    );
}

#[test]
fn final_homspace_normalizes_zero_degeneracy_sectors() {
    // What: explicitly zero-degenerate sectors and omitted sectors yield
    // identical canonical HomSpaces and lowered block layouts.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();

    let vacuum = U1Irrep::new(0).sector_id();
    let absent = U1Irrep::new(1).sector_id();
    let explicit_zero = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 3), (absent, 0)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2), (absent, 0)], true)]),
    );
    let omitted = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], true)]),
    );
    assert_eq!(explicit_zero, omitted);

    let provider = Arc::new(U1FusionRule);
    let explicit = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        Arc::clone(&provider),
        explicit_zero,
    )
    .unwrap();
    let omitted = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        provider, omitted,
    )
    .unwrap();
    assert_eq!(
        layout_snapshot(explicit.space()),
        layout_snapshot(omitted.space())
    );
    assert_eq!(
        explicit.space().required_len().unwrap(),
        omitted.space().required_len().unwrap()
    );
}

#[test]
fn final_and_derived_homspaces_preserve_provider_and_skip_shape_cache() {
    // What: root and derived canonical spaces retain the caller's provider
    // allocation while avoiding per-tree shape reconstruction and scratch
    // admission; the explicit expert shape constructor remains available.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();

    let provider = Arc::new(U1FusionRule);
    let vacuum = U1Irrep::new(0).sector_id();
    let root_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 3)], true)]),
    );
    let legacy_shape_path_builds_rec = probe::record();
    let scratch_structure_builds_rec = probe::record();
    let root = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_lowered(
        Arc::clone(&provider),
        root_homspace.clone(),
    )
    .unwrap();
    assert!(Arc::ptr_eq(root.provider_arc(), &provider));
    assert_eq!(
        legacy_shape_path_builds_rec.get(Event::LegacyShapePathBuild),
        0
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );

    let derived_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 5)], false)]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let legacy_shape_path_builds_rec = probe::record();
    let scratch_structure_builds_rec = probe::record();
    let derived = root
        .derive_from_final_homspace(derived_homspace.clone())
        .unwrap();
    assert!(Arc::ptr_eq(derived.provider_arc(), &provider));
    assert_eq!(
        legacy_shape_path_builds_rec.get(Event::LegacyShapePathBuild),
        0
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
    let encoded =
        DynamicFusionMapSpace::from_final_homspace(provider.as_ref(), derived_homspace).unwrap();
    assert_eq!(derived.space(), &encoded);

    let rebound = root.rebind_validated(&root.validated_layout()).unwrap();
    assert!(Arc::ptr_eq(rebound.provider_arc(), &provider));
    assert!(Arc::ptr_eq(
        root.space().structure(),
        rebound.space().structure()
    ));

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let legacy_shape_path_builds_rec = probe::record();
    let scratch_structure_builds_rec = probe::record();
    let shapes = shapes_from_tree_keys(provider.as_ref(), &root_homspace);
    let _expert =
        BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(provider, root_homspace, shapes)
            .unwrap();
    assert_eq!(
        legacy_shape_path_builds_rec.get(Event::LegacyShapePathBuild),
        1
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn lowered_final_homspace_keeps_single_pass_and_publishes_only_success() {
    // What: the lowered final builder matches the encoded single-pass
    // layout, never materializes the per-tree shape batch, and an
    // extent failure reaches no layout commit or scratch publication.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let rule = rule();
    reset_primer_calls();
    let legacy_shape_path_builds_rec = probe::record();

    let lowered =
        DynamicFusionMapSpace::from_final_homspace_with_primer(&rule, homspace(), counting_primer)
            .unwrap();
    assert_eq!(primer_calls(), 1);
    assert_eq!(
        legacy_shape_path_builds_rec.get(Event::LegacyShapePathBuild),
        0
    );

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let encoded = DynamicFusionMapSpace::from_final_homspace(&rule, homspace()).unwrap();
    assert_eq!(lowered, encoded);

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let vacuum = U1Irrep::new(0).sector_id();
    let overflowing = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, usize::MAX)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
    );
    let scratch_structure_builds_rec = probe::record();
    let error = DynamicFusionMapSpace::from_final_homspace_with_primer(
        &U1FusionRule,
        overflowing,
        checked_metadata_dispatcher::<U1FusionRule>,
    )
    .unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
    // Why not compare process-global cache totals: coverage runs unit
    // tests in parallel. These counters attribute publication to this
    // failing transaction only.
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn lowered_metadata_routes_every_eager_result_through_the_primer() {
    // What: final, transform, and ordered contraction metadata enter the
    // lowered primer once per eager result-space derivation.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let rule = rule();
    let source = source(&rule);

    reset_primer_calls();
    let final_space =
        DynamicFusionMapSpace::from_final_homspace_with_primer(&rule, homspace(), counting_primer)
            .unwrap();
    assert_eq!(primer_calls(), 1);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let encoded_final = DynamicFusionMapSpace::from_final_homspace(&rule, homspace()).unwrap();
    assert_eq!(final_space, encoded_final);

    symtn_core::clear_structure_caches();
    reset_primer_calls();
    let operation = TreeTransformOperation::permute([1, 0], [3, 2]);
    let transformed = source
        .transformed_with_primer(&rule, &operation, counting_primer)
        .unwrap();
    let repeated = source
        .transformed_with_primer(&rule, &operation, counting_primer)
        .unwrap();
    assert_eq!(primer_calls(), 2);
    assert_eq!(transformed, repeated);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let encoded_transformed = source.transformed(&rule, &operation).unwrap();
    assert_eq!(transformed, encoded_transformed);

    symtn_core::clear_structure_caches();
    reset_primer_calls();
    let axes = TensorContractSpec::new(
        &[],
        &[],
        OutputAxisOrder::from_axes(&[1, 0, 2, 3, 4, 5, 6, 7]),
    );
    let contracted = DynamicFusionMapSpace::contracted_with_spec_and_primer(
        &rule,
        &source,
        &source,
        axes,
        None,
        counting_primer,
    )
    .unwrap();
    let repeated = DynamicFusionMapSpace::contracted_with_spec_and_primer(
        &rule,
        &source,
        &source,
        axes,
        None,
        counting_primer,
    )
    .unwrap();
    assert_eq!(primer_calls(), 2);
    assert_eq!(contracted, repeated);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let encoded_contracted =
        DynamicFusionMapSpace::contracted_with_spec(&rule, &source, &source, axes).unwrap();
    assert_eq!(contracted, encoded_contracted);
}

#[test]
fn metadata_error_maps_to_the_exact_codec_cause() {
    // What: a malformed product ID crosses into the operation layer with
    // its codec cause intact rather than as a static message: the checked
    // enumerator preserves the typed cause instead of flattening it to
    // InvalidArgument.
    let malformed = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(usize::MAX), 1)], false)]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let error = checked_layout_primer(&rule(), &malformed).unwrap_err();
    assert!(matches!(
        error,
        OperationError::FusionAlgebra(ref cause)
            if matches!(**cause, FusionAlgebraError::ProductCodec(_))
    ));
}

#[test]
fn algebra_error_maps_to_exact_operation_cause() {
    // What: the operation boundary owns the exact lowered U1 closure
    // cause and exposes it through the standard error source chain.
    let overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(U1Irrep::new(i32::MAX), 1)], false),
            SectorLeg::new([(U1Irrep::new(1), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let expected = FusionAlgebraError::U1FusionOverflow {
        left: i32::MAX,
        right: 1,
    };
    let error = checked_layout_primer(&U1FusionRule, &overflow).unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(expected.clone()))
    );
    assert_eq!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<FusionAlgebraError>()),
        Some(&expected)
    );
}

fn assert_lowered_root_failure<R>(
    rule: Arc<R>,
    homspace: FusionTreeHomSpace,
    expected: FusionAlgebraError,
) where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
{
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let scratch_structure_builds_rec = probe::record();
    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        rule,
        homspace,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap_err();
    assert_eq!(error, OperationError::FusionAlgebra(Box::new(expected)));
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn failed_lowered_roots_build_and_admit_no_scratch() {
    // What: invalid U1, SU2, and product algebra reaches its exact typed
    // cause before scratch construction, identity, or admission.
    let u1_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(U1Irrep::new(i32::MAX).sector_id(), 1)], false),
            SectorLeg::new([(U1Irrep::new(1).sector_id(), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_lowered_root_failure(
        Arc::new(U1FusionRule),
        u1_overflow,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        },
    );

    let su2_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SU2Irrep::from_twice_spin(128).sector_id(), 1)], false),
            SectorLeg::new([(SU2Irrep::from_twice_spin(127).sector_id(), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_lowered_root_failure(
        Arc::new(SU2FusionRule),
        su2_overflow,
        FusionAlgebraError::FusionNotRepresentable {
            left: SU2Irrep::from_twice_spin(128).sector_id(),
            right: SU2Irrep::from_twice_spin(127).sector_id(),
        },
    );

    let product_rule = Arc::new(Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule));
    let product_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new(
                [(
                    Fz2U1Codec::encode(
                        Z2Irrep::EVEN.sector_id(),
                        U1Irrep::new(i32::MAX).sector_id(),
                    ),
                    1,
                )],
                false,
            ),
            SectorLeg::new(
                [(
                    Fz2U1Codec::encode(Z2Irrep::ODD.sector_id(), U1Irrep::new(1).sector_id()),
                    1,
                )],
                false,
            ),
        ]),
        FusionProductSpace::new([]),
    );
    assert_lowered_root_failure(
        product_rule,
        product_overflow,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        },
    );
}

#[test]
fn lowered_builder_runs_before_homspace_intern() {
    // What: a no-ID HomSpace reaches its exact lowered algebra failure while
    // the key builder still observes that no semantic identity was published.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(U1Irrep::new(i32::MAX).sector_id(), 1)], false),
            SectorLeg::new([(U1Irrep::new(1).sector_id(), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let scratch_structure_builds_rec = probe::record();
    let error = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        &U1FusionRule,
        overflow,
        Vec::<Vec<usize>>::new(),
        |rule, homspace| {
            assert!(homspace.existing_id().is_none());
            checked_layout_primer(rule, homspace)
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }))
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn lowered_shape_validation_runs_before_homspace_intern() {
    // What: valid U1 lowering followed by an invalid shape reports the exact
    // rank cause before HomSpace identity or scratch publication.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let scratch_structure_builds_rec = probe::record();
    let error = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        &U1FusionRule,
        homspace,
        [vec![1]],
        |rule, homspace| {
            assert!(homspace.existing_id().is_none());
            checked_layout_primer(rule, homspace)
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::Core(symtn_core::CoreError::StructureRankMismatch {
            expected: 0,
            actual: 1,
        })
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn lowered_count_failure_builds_and_admits_no_scratch() {
    // What: one valid scalar key paired with zero caller shapes reports
    // the exact count error and abandons its cold prepared layout.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let scratch_structure_builds_rec = probe::record();

    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::new(U1FusionRule),
        homspace,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(symtn_core::CoreError::BlockCountMismatch {
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn encoded_cold_invalid_count_does_not_publish_layouts() {
    // Isolated like #649/#650's checked_bind_failure test: this asserts an
    // absolute process-global cache snapshot, which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked encoded constructions.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_ENCODED_COLD_INVALID_COUNT_ISOLATED",
        "contract::dynamic_space::checked_metadata_tests::encoded_cold_invalid_count_does_not_publish_layouts",
    ) {
        return;
    }
    // What: an invalid explicit encoded U1 shape count leaves both staged
    // process-global cache snapshots unchanged before any layout commit.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );

    let error = DynamicFusionMapSpace::from_degeneracy_shapes(
        &U1FusionRule,
        homspace,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::BlockCountMismatch {
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        ),
        before
    );
}

#[test]
fn encoded_existing_candidate_invalid_shape_does_not_publish_again() {
    // Isolated like #649/#650's checked_bind_failure test: this asserts an
    // absolute process-global cache snapshot, which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked encoded constructions.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_ENCODED_EXISTING_CANDIDATE_INVALID_SHAPE_ISOLATED",
        "contract::dynamic_space::checked_metadata_tests::encoded_existing_candidate_invalid_shape_does_not_publish_again",
    ) {
        return;
    }
    // What: an already committed encoded U1 candidate remains
    // observationally unchanged when a later explicit shape is invalid.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    DynamicFusionMapSpace::from_degeneracy_shapes(
        &U1FusionRule,
        homspace.clone(),
        [Vec::<usize>::new()],
    )
    .unwrap();
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );

    let error = DynamicFusionMapSpace::from_degeneracy_shapes(&U1FusionRule, homspace, [vec![1]])
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(CoreError::StructureRankMismatch {
            expected: 0,
            actual: 1,
        })
    );
    // The rejected call looks its layout up (one hit) and publishes nothing.
    let after = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );
    assert_eq!(after.0.hits(), before.0.hits() + 1);
    assert_eq!(
        (after.0.entries(), after.0.misses(), after.0.admissions()),
        (before.0.entries(), before.0.misses(), before.0.admissions())
    );
    assert_eq!(after.1, before.1);
}

#[test]
fn encoded_cold_extent_overflow_does_not_publish_layouts() {
    // Isolated like #649/#650's checked_bind_failure test: this asserts an
    // absolute process-global cache snapshot, which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked encoded constructions.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_ENCODED_COLD_EXTENT_OVERFLOW_ISOLATED",
        "contract::dynamic_space::checked_metadata_tests::encoded_cold_extent_overflow_does_not_publish_layouts",
    ) {
        return;
    }
    // What: a valid explicit U1 shape whose final extent overflows leaves
    // both staged cache snapshots unchanged before any layout commit.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let vacuum = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, usize::MAX)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
    );
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );

    let error = DynamicFusionMapSpace::from_degeneracy_shapes(
        &U1FusionRule,
        homspace,
        [vec![usize::MAX, 2]],
    )
    .unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
    assert_eq!(
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        ),
        before
    );
}

#[test]
fn encoded_and_lowered_explicit_layouts_share_checked_frozen_content() {
    // Isolated like #649/#650's checked_bind_failure test: this asserts
    // absolute process-global cache counts (entries()==1, misses()==1,
    // admissions()==1 from a fresh reset), which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked encoded
    // constructions landing between the reset and this test's own build.
    // Found flaking during #1598/#1606 verification (566 passed, 1 failed
    // at this file's entries()==1 assertion, actual 2, under
    // `cargo test -p symtn-tensors --lib`); same class as the four named
    // instances in that pair of issues, so it gets the same fix.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_ENCODED_AND_LOWERED_SHARE_FROZEN_CONTENT_ISOLATED",
        "contract::dynamic_space::checked_metadata_tests::encoded_and_lowered_explicit_layouts_share_checked_frozen_content",
    ) {
        return;
    }
    // What: checked encoded and lowered explicit constructors retain exact
    // block order, shape, stride, offset, and storage length while one
    // encoded transaction publishes the canonical frozen content once.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = rule();
    let homspace = homspace();
    let shapes = shapes_from_tree_keys(&rule, &homspace);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();

    let encoded =
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace.clone(), shapes.clone())
            .unwrap();
    let after_encoded = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );
    assert_eq!(after_encoded.0.entries(), 1);
    assert_eq!(after_encoded.0.misses(), 1);
    assert_eq!(after_encoded.1.admissions(), 1);

    let lowered = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        &rule,
        homspace,
        shapes,
        checked_layout_primer,
    )
    .unwrap();

    assert_eq!(layout_snapshot(&encoded), layout_snapshot(&lowered));
    assert_eq!(encoded.required_len(), lowered.required_len());
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure).admissions(),
        after_encoded.1.admissions()
    );
}

#[test]
fn encoded_explicit_u1_and_su2_complete_layouts_are_valid() {
    // What: the staged explicit path accepts canonical U1 and SU2
    // multiplicity-free storage layouts; the product case is covered by
    // the encoded/lowered layout oracle above.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let u1_vacuum = U1Irrep::new(0).sector_id();
    let u1 = DynamicFusionMapSpace::from_degeneracy_shapes(
        &U1FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1_vacuum, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1_vacuum, 3)], false)]),
        ),
        [vec![2, 3]],
    )
    .unwrap();
    let su2_vacuum = SU2Irrep::from_twice_spin(0).sector_id();
    let su2 = DynamicFusionMapSpace::from_degeneracy_shapes(
        &SU2FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(su2_vacuum, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(su2_vacuum, 3)], false)]),
        ),
        [vec![2, 3]],
    )
    .unwrap();

    assert_eq!(u1.required_len(), Ok(6));
    assert_eq!(su2.required_len(), Ok(6));
}

#[test]
fn lowered_extent_overflow_precedes_identity_and_scratch_admission() {
    // What: valid U1 keys and shapes whose block extent overflows usize
    // fail locally before layout, HomSpace, or scratch publication.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let vacuum = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, usize::MAX)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
    );
    let scratch_structure_builds_rec = probe::record();

    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::new(U1FusionRule),
        homspace,
        [vec![usize::MAX, 2]],
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(symtn_core::CoreError::ElementCountOverflow)
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn cached_layout_shape_failure_is_observationally_read_only() {
    // What: a core-layout hit followed by invalid caller shapes performs
    // no scratch build, layout commit, HomSpace identity, or admission.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    homspace
        .prepare_fusion_tree_layout_checked(&U1FusionRule)
        .unwrap()
        .commit();
    assert!(homspace.existing_id().is_none());
    let scratch_structure_builds_rec = probe::record();

    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::new(U1FusionRule),
        homspace,
        [vec![1]],
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(symtn_core::CoreError::StructureRankMismatch {
            expected: 0,
            actual: 1,
        })
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn existing_id_failure_builds_and_admits_no_scratch() {
    // What: an existing HomSpace ID that misses scratch lookup and then
    // fails lowered algebra builds and admits no scratch structure.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(U1Irrep::new(i32::MAX).sector_id(), 1)], false),
            SectorLeg::new([(U1Irrep::new(1).sector_id(), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let _ = overflow.id();
    let scratch_structure_builds_rec = probe::record();
    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::new(U1FusionRule),
        overflow,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }))
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn canonical_explicit_rebuilds_staged_lowered_keys_before_core_reuse() {
    // What: explicit caller shapes are validated on every request before
    // the core cache reuses their canonical frozen content.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let builds = Cell::new(0);
    let first = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        &U1FusionRule,
        homspace,
        [Vec::<usize>::new()],
        |rule, homspace| {
            builds.set(builds.get() + 1);
            checked_layout_primer(rule, homspace)
        },
    )
    .unwrap();
    assert_eq!(builds.get(), 1);

    builds.set(0);
    let second = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
        &U1FusionRule,
        first.homspace().clone(),
        [Vec::<usize>::new()],
        |rule, homspace| {
            builds.set(builds.get() + 1);
            checked_layout_primer(rule, homspace)
        },
    )
    .unwrap();
    assert_eq!(builds.get(), 1);
    assert_eq!(
        first.structure().content_id(),
        second.structure().content_id()
    );
    assert_eq!(first, second);
}

#[test]
fn admission_rejects_the_excluded_u1_id_without_publication() {
    // What: a raw ID excluded by the label constructor is rejected before
    // the space publishes anything.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let min = SectorId::new(u32::MAX as usize);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(min, 1)], false)]),
        FusionProductSpace::new([]),
    );
    let scratch_structure_builds_rec = probe::record();

    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::new(U1FusionRule),
        homspace,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector { sector: min }))
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn admission_rejects_a_product_containing_the_excluded_u1_id_without_publication() {
    // What: a packed product containing the excluded U1 ID is rejected
    // before any result layout publication.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let provider = Arc::new(Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule));
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let lhs = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        scalar,
        [Vec::<usize>::new()],
    )
    .unwrap();
    let excluded_u1 = SectorId::new(u32::MAX as usize);
    let odd_min = Fz2U1Codec::encode(Z2Irrep::ODD.sector_id(), excluded_u1);
    let rhs_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(odd_min, 1)], false)]),
        FusionProductSpace::new([]),
    );
    let scratch_structure_builds_rec = probe::record();

    let _ = &lhs;
    let error = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        rhs_homspace,
        Vec::<Vec<usize>>::new(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector {
            sector: excluded_u1,
        }))
    );
    assert_eq!(
        scratch_structure_builds_rec.get(Event::ScratchStructureBuild),
        0
    );
}

#[test]
fn transform_invalid_axis_error_is_reported_exactly() {
    // What: malformed transform axes retain their exact structural error
    // precedence even when a legal boundary crossing would overflow U1 dual.
    // The min-dual leg is rejected at admission, so axis validation and
    // dual representability do not race: build the precedence
    // fixture on a leg whose dual exists.
    let representable = U1Irrep::new(1).sector_id();
    let source = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(representable, 1)], false)]),
            FusionProductSpace::new([]),
        ),
        Vec::<Vec<usize>>::new(),
    )
    .unwrap();

    let error = source
        .transformed_multiplicity_free(&TreeTransformOperation::permute([], [1]))
        .unwrap_err();
    assert_eq!(
        error,
        OperationError::InvalidPermutation {
            axes: vec![1],
            rank: 1
        }
    );
}

#[test]
fn fz2_lowered_transform_contract_and_mixed_plan_match_encoded_oracle() {
    // What: the closed fZ2 algebra produces the exact encoded structure for
    // transform and contract, and mixed strategy planning is operand-order safe.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = Z2Irrep::ODD.sector_id();
    let leg = || FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]);
    let homspace = FusionTreeHomSpace::new(leg(), leg());
    let count = homspace.fusion_tree_keys(provider.as_ref()).len();
    let shapes = vec![vec![1, 1]; count];
    let lowered = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        homspace.clone(),
        shapes.clone(),
    )
    .unwrap();
    let hits_before_repeated_lowered =
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
            .hits();
    let repeated_lowered = BoundDynamicFusionMapSpace::from_degeneracy_shapes_lowered(
        Arc::clone(&provider),
        homspace.clone(),
        shapes.clone(),
    )
    .unwrap();
    assert_eq!(
        lowered.space().structure().content_id(),
        repeated_lowered.space().structure().content_id()
    );
    assert!(
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
            .hits()
            > hits_before_repeated_lowered,
        "same-content lowered construction must reuse the complete layout"
    );
    let encoded =
        BoundDynamicFusionMapSpace::from_degeneracy_shapes(Arc::clone(&provider), homspace, shapes)
            .unwrap();
    let operation = TreeTransformOperation::permute([1], [0]);
    let lowered_transform = lowered.transformed_multiplicity_free(&operation).unwrap();
    let encoded_transform = encoded.transformed_multiplicity_free(&operation).unwrap();
    assert_eq!(lowered_transform.space(), encoded_transform.space());

    let lowered_dst =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lowered, &lowered, &[1], &[0])
            .unwrap();
    let encoded_dst =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&encoded, &encoded, &[1], &[0])
            .unwrap();
    assert_eq!(lowered_dst.space(), encoded_dst.space());
    let axes = symtn_operations::TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mixed =
        crate::prepare_tensorcontract_fusion_plan_dyn(&lowered_dst, &lowered, &encoded, axes)
            .unwrap();
    let lowered_only =
        crate::prepare_tensorcontract_fusion_plan_dyn(&lowered_dst, &lowered, &lowered, axes)
            .unwrap();
    assert_eq!(mixed, lowered_only);
}
