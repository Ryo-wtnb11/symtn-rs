use super::*;
use crate::test_support::CACHE_TEST_LOCK;
use std::cell::Cell;
use symtn_core::{
    structure_cache_info, FermionParityFusionRule, FusionAlgebraError, FusionProductSpace,
    FusionTreeKey, Fz2SectorLayout, PackedProductCodec, ProductFusionRule, ProductSectorCodec,
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
    TripleRule::try_new(
        Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap()
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
    DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap()
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

/// One block's key, shape, strides, and offset.
type LayoutRow = (BlockKey, Vec<usize>, Vec<usize>, usize);

fn layout_snapshot(space: &DynamicFusionMapSpace) -> Vec<LayoutRow> {
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

/// TensorKit's coupled-sector matrix layout, computed by hand from the
/// per-tree shapes: for each coupled sector (in key order) one column-major
/// matrix whose rows are the codomain trees and whose columns are the domain
/// trees, each tree block a strided window of it.
fn tensorkit_layout_oracle<R>(rule: &R, homspace: &FusionTreeHomSpace) -> (Vec<LayoutRow>, usize)
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let nout = homspace.codomain().len();
    let keys = homspace
        .prepare_fusion_tree_layout_checked(rule)
        .unwrap()
        .commit();
    let shapes = shapes_from_tree_keys(rule, homspace);
    let mut coupled = Vec::new();
    for key in keys.iter() {
        if !coupled.contains(&key.coupled()) {
            coupled.push(key.coupled());
        }
    }
    let mut layout = Vec::new();
    let mut base = 0usize;
    for sector in coupled {
        let mut rows = Vec::<(FusionTreeKey, usize)>::new();
        let mut cols = Vec::<(FusionTreeKey, usize)>::new();
        for (key, shape) in keys.iter().zip(&shapes) {
            if key.coupled() != sector {
                continue;
            }
            if !rows.iter().any(|(tree, _)| tree == key.codomain_tree()) {
                rows.push((key.codomain_tree().clone(), shape[..nout].iter().product()));
            }
            if !cols.iter().any(|(tree, _)| tree == key.domain_tree()) {
                cols.push((key.domain_tree().clone(), shape[nout..].iter().product()));
            }
        }
        let offset_of = |trees: &[(FusionTreeKey, usize)], tree: &FusionTreeKey| {
            trees
                .iter()
                .take_while(|(candidate, _)| candidate != tree)
                .map(|(_, extent)| extent)
                .sum::<usize>()
        };
        let nrows = rows.iter().map(|(_, extent)| extent).sum::<usize>();
        let ncols = cols.iter().map(|(_, extent)| extent).sum::<usize>();
        for (key, shape) in keys.iter().zip(&shapes) {
            if key.coupled() != sector {
                continue;
            }
            let mut strides = Vec::with_capacity(shape.len());
            let mut stride = 1;
            for (axis, &extent) in shape.iter().enumerate() {
                if axis == nout {
                    stride = nrows;
                }
                strides.push(stride);
                stride *= extent;
            }
            let offset = base
                + offset_of(&rows, key.codomain_tree())
                + offset_of(&cols, key.domain_tree()) * nrows;
            layout.push((
                BlockKey::FusionTree(key.clone()),
                shape.clone(),
                strides,
                offset,
            ));
        }
        base += nrows * ncols;
    }
    (layout, base)
}

fn assert_final_homspace_matches_tensorkit_layout<R>(provider: Arc<R>, homspace: FusionTreeHomSpace)
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
{
    let (expected_layout, expected_len) = tensorkit_layout_oracle(provider.as_ref(), &homspace);

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let expected_rule = provider.rule_identity();
    let actual = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
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
fn final_homspace_layout_matches_tensorkit_layout_across_supported_rules() {
    // What: U1, SU2, fZ2, their supported products, and scalar/vector
    // boundaries produce TensorKit's coupled-sector block layout.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let u1_minus = U1Irrep::new(-1).sector_id();
    let u1_zero = U1Irrep::new(0).sector_id();
    let u1_plus = U1Irrep::new(1).sector_id();
    let u1_codomain = SectorLeg::new([(u1_minus, 2), (u1_zero, 1), (u1_plus, 3)], false);
    let u1_domain = SectorLeg::new([(u1_minus, 4), (u1_zero, 2), (u1_plus, 1)], true);
    assert_final_homspace_matches_tensorkit_layout(
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
    assert_final_homspace_matches_tensorkit_layout(
        Arc::new(SU2FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([su2_left.clone(), su2_right.clone()]),
            FusionProductSpace::new([su2_right.clone(), su2_left.clone()]),
        ),
    );
    // Non-square coupled blocks with several domain trees pin the column
    // offsets and the domain stride.
    let su2_wide = SectorLeg::new([(su2_zero, 2), (su2_half, 3)], false);
    assert_final_homspace_matches_tensorkit_layout(
        Arc::new(SU2FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([su2_left.clone(), su2_left]),
            FusionProductSpace::new([su2_wide, su2_right]),
        ),
    );

    let even = Z2Irrep::EVEN.sector_id();
    let odd = Z2Irrep::ODD.sector_id();
    let fz2_left = SectorLeg::new([(even, 1), (odd, 3)], false);
    let fz2_right = SectorLeg::new([(even, 2), (odd, 1)], true);
    assert_final_homspace_matches_tensorkit_layout(
        Arc::new(FermionParityFusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([fz2_left.clone(), fz2_right.clone()]),
            FusionProductSpace::new([fz2_right, fz2_left]),
        ),
    );

    // The nested fixture covers both ProductFusionRule levels:
    // fZ2 x U1 and (fZ2 x U1) x SU2.
    assert_final_homspace_matches_tensorkit_layout(Arc::new(rule()), homspace());
    assert_final_homspace_matches_tensorkit_layout(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
    );
    assert_final_homspace_matches_tensorkit_layout(
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
    let explicit = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        explicit_zero,
    )
    .unwrap();
    let omitted = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
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
fn final_and_derived_homspaces_preserve_provider() {
    // What: root and derived canonical spaces retain the caller's provider
    // allocation and share the encoded layout.
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
    let root = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        root_homspace.clone(),
    )
    .unwrap();
    assert!(Arc::ptr_eq(root.provider_arc(), &provider));

    let derived_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, 5)], false)]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let derived = root
        .derive_from_final_homspace(derived_homspace.clone())
        .unwrap();
    assert!(Arc::ptr_eq(derived.provider_arc(), &provider));
    let encoded =
        DynamicFusionMapSpace::from_final_homspace(provider.as_ref(), derived_homspace).unwrap();
    assert_eq!(derived.space(), &encoded);

    let rebound = root.rebind_validated(&root.validated_layout()).unwrap();
    assert!(Arc::ptr_eq(rebound.provider_arc(), &provider));
    assert!(Arc::ptr_eq(
        root.space().structure(),
        rebound.space().structure()
    ));
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

    let lowered =
        DynamicFusionMapSpace::from_final_homspace_with_primer(&rule, homspace(), counting_primer)
            .unwrap();
    assert_eq!(primer_calls(), 1);

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
    let error = DynamicFusionMapSpace::from_final_homspace_with_primer(
        &U1FusionRule,
        overflowing,
        checked_metadata_dispatcher::<U1FusionRule>,
    )
    .unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
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
    let error =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(rule, homspace)
            .unwrap_err();
    assert_eq!(error, OperationError::FusionAlgebra(Box::new(expected)));
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

    let product_rule = Arc::new(Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap());
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
fn encoded_cold_extent_overflow_does_not_publish_layouts() {
    // Isolated like the other process-global cache-count tests: this asserts an
    // absolute process-global cache snapshot, which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked encoded constructions.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_ENCODED_COLD_EXTENT_OVERFLOW_ISOLATED",
        "contract::dynamic_space::checked_metadata_tests::encoded_cold_extent_overflow_does_not_publish_layouts",
    ) {
        return;
    }
    // What: a U1 root whose final extent overflows publishes no degeneracy
    // layout. Its fusion-tree enumeration depends on sectors alone, so the
    // sector structure stays a valid cache entry.
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
    let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);

    let error = DynamicFusionMapSpace::from_final_homspace(&U1FusionRule, homspace).unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
        before
    );
}

#[test]
fn encoded_and_lowered_final_layouts_share_checked_frozen_content() {
    // Isolated like the other process-global cache-count tests: this asserts
    // absolute process-global cache counts (entries()==1, misses()==1,
    // admissions()==1 from a fresh reset), which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked encoded
    // constructions landing between the reset and this test's own build.
    // Without isolation the entries()==1 assertion flakes (actual 2) when run
    // with the rest of the lib tests, so it gets the same fix.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_ENCODED_AND_LOWERED_SHARE_FROZEN_CONTENT_ISOLATED",
        "contract::dynamic_space::checked_metadata_tests::encoded_and_lowered_final_layouts_share_checked_frozen_content",
    ) {
        return;
    }
    // What: checked encoded and lowered final constructors retain exact
    // block order, shape, stride, offset, and storage length while one
    // encoded transaction publishes the canonical frozen content once.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = rule();
    let homspace = homspace();
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();

    let encoded = DynamicFusionMapSpace::from_final_homspace(&rule, homspace.clone()).unwrap();
    let after_encoded = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );
    assert_eq!(after_encoded.0.entries(), 1);
    assert_eq!(after_encoded.0.misses(), 1);
    assert_eq!(after_encoded.1.admissions(), 1);

    let lowered = DynamicFusionMapSpace::from_final_homspace_with_primer(
        &rule,
        homspace,
        checked_metadata_dispatcher::<TripleRule>,
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
fn encoded_u1_and_su2_root_layouts_have_hand_extent() {
    // What: canonical U1 and SU2 multiplicity-free roots have the hand
    // extent 2 x 3; the product case is covered by the layout oracle above.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let u1_vacuum = U1Irrep::new(0).sector_id();
    let u1 = DynamicFusionMapSpace::from_final_homspace(
        &U1FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(u1_vacuum, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(u1_vacuum, 3)], false)]),
        ),
    )
    .unwrap();
    let su2_vacuum = SU2Irrep::from_twice_spin(0).sector_id();
    let su2 = DynamicFusionMapSpace::from_final_homspace(
        &SU2FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(su2_vacuum, 2)], false)]),
            FusionProductSpace::new([SectorLeg::new([(su2_vacuum, 3)], false)]),
        ),
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

    let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(U1FusionRule),
        homspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::Core(symtn_core::CoreError::ElementCountOverflow)
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
    let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(U1FusionRule),
        overflow,
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }))
    );
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

    let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(U1FusionRule),
        homspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector { sector: min }))
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
    let provider = Arc::new(Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap());
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let lhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        scalar,
    )
    .unwrap();
    let excluded_u1 = SectorId::new(u32::MAX as usize);
    let odd_min = Fz2U1Codec::encode(Z2Irrep::ODD.sector_id(), excluded_u1);
    let rhs_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(odd_min, 1)], false)]),
        FusionProductSpace::new([]),
    );

    let _ = &lhs;
    let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        rhs_homspace,
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(FusionAlgebraError::InvalidSector {
            sector: excluded_u1,
        }))
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
    let source = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(representable, 1)], false)]),
            FusionProductSpace::new([]),
        ),
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
fn fz2_lowered_transform_and_contract_build_hand_keys_and_mixed_plans_agree() {
    // What: the closed fZ2 algebra builds the hand-written transform and
    // contract layouts, and a contraction plan does not depend on whether
    // the rhs operand was bound through the lowered or the encoded strategy.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let provider = Arc::new(FermionParityFusionRule);
    let even = Z2Irrep::EVEN.sector_id();
    let odd = Z2Irrep::ODD.sector_id();
    let leg = || FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 3)], false)]);
    let homspace = FusionTreeHomSpace::new(leg(), leg());
    let lowered = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )
    .unwrap();
    let hits_before_repeated_lowered =
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
            .hits();
    let repeated_lowered =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            Arc::clone(&provider),
            homspace.clone(),
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
    let encoded = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();
    // Rank-1 trees carry no inner line or vertex; each parity couples to
    // itself with a (degeneracy x degeneracy) block.
    let blocks = |space: &DynamicFusionMapSpace| {
        let mut blocks = layout_snapshot(space)
            .into_iter()
            .map(|(key, shape, _, _)| (key, shape))
            .collect::<Vec<_>>();
        blocks.sort();
        blocks
    };
    let expected = |dual: bool| {
        let mut blocks = [(even, 2), (odd, 3)]
            .map(|(sector, degeneracy)| {
                let key = FusionTreePairKey::try_pair_from_sector_ids(
                    [sector.id()],
                    [sector.id()],
                    sector.id(),
                    [dual],
                    [dual],
                    [],
                    [],
                    [],
                    [],
                )
                .unwrap();
                (BlockKey::FusionTree(key), vec![degeneracy, degeneracy])
            })
            .to_vec();
        blocks.sort();
        blocks
    };
    // Moving a leg across the codomain/domain split dualizes it (TensorKit
    // `permute`), so both legs of the swapped map are dual.
    let lowered_transform = lowered
        .transformed_multiplicity_free(&TreeTransformOperation::permute([1], [0]))
        .unwrap();
    assert_eq!(blocks(lowered_transform.space()), expected(true));

    let lowered_dst =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lowered, &lowered, &[1], &[0])
            .unwrap();
    assert_eq!(blocks(lowered_dst.space()), expected(false));
    let output = (0..lowered_dst.space().rank()).collect::<Vec<_>>();
    let plan = |rhs: &BoundDynamicFusionMapSpace<_>| {
        crate::prepare_tensorcontract_fusion_plan_dyn(
            &lowered_dst,
            crate::FusionOperand::direct(lowered.space()),
            crate::FusionOperand::direct(rhs.space()),
            &[1],
            &[0],
            &output,
        )
        .unwrap()
    };
    let mixed = plan(&encoded);
    let lowered_only = plan(&lowered);
    assert_eq!(mixed, lowered_only);
}
