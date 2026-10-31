use super::*;

#[test]
fn tensor_map_zeros_carries_a_complex_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the payload dtype is independent of the provider's real
    // categorical coefficient scalar.
    let provider = Arc::new(external_su2());
    let leg = su2_leg(&provider, false);
    let runtime = runtime();

    let tensor: TensorMap<ExternalSu2Rule, Complex64> =
        TensorMap::zeros(engine, &runtime, [&leg, &leg], [&leg, &leg]).unwrap();

    assert!(tensor.subblock_count() >= 2);
    assert!(tensor
        .dense_data()
        .unwrap()
        .iter()
        .all(|&value| value == Complex64::new(0.0, 0.0)));
}

#[test]
fn tensor_map_accepts_separately_allocated_equal_identity_providers() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: two independent allocations of one rule interoperate; the facade
    // keys on `RuleIdentity`, never on `Arc` identity.
    let first = Arc::new(external_z3());
    let second = Arc::new(external_z3());
    assert!(!Arc::ptr_eq(&first, &second));
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3Rule, f64> = TensorMap::zeros(
        engine,
        &runtime,
        [&z3_leg(&first, false)],
        [&z3_leg(&second, true)],
    )
    .unwrap();

    assert!(tensor.subblock_count() >= 1);
}

#[test]
fn tensor_map_rejects_distinct_rule_identities_before_provider_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: two providers of the same Rust type but different identities are a
    // rule mismatch, reported before any layout is staged — the caches stay
    // exactly as they were.
    let _guard = cache_lock();
    let first = Arc::new(external_z3_tagged(0));
    let second = Arc::new(external_z3_tagged(1));
    assert_ne!(first.rule_identity(), second.rule_identity());
    let runtime = runtime();
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );

    let error = TensorMap::<ExternalZ3Rule, f64>::zeros(
        engine,
        &runtime,
        [&z3_leg(&first, false)],
        [&z3_leg(&second, true)],
    )
    .unwrap_err();

    assert!(matches!(error, symtn::typed::Error::RuleMismatch));
    assert_eq!(
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        ),
        before
    );
}

#[test]
fn enumerated_rules_share_identity_by_category_value() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a fusion-symbols category under `EnumeratedCodec` is identified by
    // its value. Two separately built Z3 rules mix, and Z3 and Z5 (whose
    // fusion-symbols fingerprints are equal) are a rule mismatch.
    use fusion_symbols::finite::{ZNIrrep, ZN};
    use symtn::sector::EnumeratedCodec;

    type ZnRule = CategoryRule<ZN, EnumeratedCodec<MultiplicityFreeAdmissionMode>>;
    let rule = |modulus| {
        Arc::new(ZnRule::try_new(ZN::new(modulus).unwrap(), EnumeratedCodec::new()).unwrap())
    };
    let leg = |rule: &Arc<ZnRule>, modulus| {
        GradedSpace::try_new(
            Arc::clone(rule),
            [
                (ZNIrrep::from_canonical(0, modulus).unwrap(), 2),
                (ZNIrrep::from_canonical(1, modulus).unwrap(), 1),
            ],
        )
        .unwrap()
    };
    let runtime = runtime();
    let (first, second, z5) = (rule(3), rule(3), rule(5));
    assert!(!Arc::ptr_eq(&first, &second));

    let tensor =
        TensorMap::<ZnRule, f64>::zeros(engine, &runtime, [&leg(&first, 3)], [&leg(&second, 3)])
            .unwrap();
    assert_eq!(tensor.subblock_count(), 2);
    let error =
        TensorMap::<ZnRule, f64>::zeros(engine, &runtime, [&leg(&first, 3)], [&leg(&z5, 5)])
            .unwrap_err();
    assert!(matches!(error, symtn::typed::Error::RuleMismatch));
}

#[test]
fn a_composed_codec_identity_never_shares_a_layout_across_categories() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: an expert codec composing `EnumeratedCodec`'s identity before
    // any rule resolves it must still separate Z3 from Z5. Charges (2, 2)
    // fuse to 1 in Z3 and to 4 in Z5, so the [2, 2] <- [1] layout has one
    // block for Z3 and none for Z5, whichever was built first.
    use fusion_symbols::finite::{ZNIrrep, ZN};
    use symtn::sector::EnumeratedCodec;

    type Enumerated = EnumeratedCodec<MultiplicityFreeAdmissionMode>;

    #[derive(Clone, Copy, Debug)]
    struct Wrapped;

    impl CategoryCodec<ZN> for Wrapped {
        type Sector = ZNIrrep;
        type Error = FusionAlgebraError;
        type Mode = MultiplicityFreeAdmissionMode;
        type Unit = CanonicalUnitAdmission;

        fn encode_label<'a>(
            &self,
            category: impl Fn() -> &'a ZN + Copy,
            rule: &RuleIdentity,
            label: &ZNIrrep,
        ) -> Result<SectorId, FusionAlgebraError> {
            Enumerated::new().encode_label(category, rule, label)
        }
        fn decode_label<'a>(
            &self,
            category: impl Fn() -> &'a ZN + Copy,
            id: SectorId,
        ) -> Result<ZNIrrep, FusionAlgebraError> {
            Enumerated::new().decode_label(category, id)
        }
        fn object<'a>(
            &self,
            category: impl Fn() -> &'a ZN + Copy,
            id: SectorId,
        ) -> Result<ZNIrrep, FusionAlgebraError> {
            Enumerated::new().object(category, id)
        }
        fn sector_id<'a>(
            &self,
            category: impl Fn() -> &'a ZN + Copy,
            rule: &RuleIdentity,
            object: &ZNIrrep,
        ) -> Result<SectorId, FusionAlgebraError> {
            Enumerated::new().sector_id(category, rule, object)
        }
        fn rule_identity(&self, category: &ZN) -> Option<RuleIdentity> {
            Some(RuleIdentity::compose_with_codec::<Self>(
                Enumerated::new().rule_identity(category)?,
                RuleIdentity::of_type::<Self>(),
            ))
        }
        fn map_error(
            &self,
            inputs: &[SectorId],
            error: CategoryAdapterError<core::convert::Infallible>,
        ) -> FusionAlgebraError {
            <Enumerated as CategoryCodec<ZN>>::map_error(&Enumerated::new(), inputs, error)
        }
    }

    type WrappedRule = CategoryRule<ZN, Wrapped>;
    let runtime = runtime();
    let mut blocks = |modulus: u32| {
        let rule = Arc::new(WrappedRule::try_new(ZN::new(modulus).unwrap(), Wrapped).unwrap());
        let leg = |charge| {
            GradedSpace::try_new(
                Arc::clone(&rule),
                [(ZNIrrep::from_canonical(charge, modulus).unwrap(), 1)],
            )
            .unwrap()
        };
        TensorMap::<WrappedRule, f64>::zeros(engine, &runtime, [&leg(2), &leg(2)], [&leg(1)])
            .unwrap()
            .subblock_count()
    };
    assert_eq!(blocks(3), 1);
    assert_eq!(blocks(5), 0);
}

#[test]
fn checked_construction_failure_publishes_no_cache_state() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Isolated: the completed-transformer counters are process-global.
    if crate::typed_facade::run_isolated_or_return(
        "SYMTN_TYPED_FACADE_CHECKED_CONSTRUCTION_FAILURE_PUBLISHES_N",
        "construction::checked_construction_failure_publishes_no_cache_state",
    ) {
        return;
    }
    // What: a provider that fails mid-staging returns a typed error and leaves
    // both process-global layout caches and the runtime's own cache untouched,
    // which is the transactional guarantee the checked path promises.
    let _guard = cache_lock();
    // The broken provider must be the layout authority (the first leg's
    // provider), otherwise the staging never calls its failing primitive.
    let broken = Arc::new(external_z3_with(Quirk::FailDual));
    let codomain = z3_leg(&broken, false);
    let healthy = Arc::new(external_z3());
    let domain = GradedSpace::try_new(
        healthy,
        [(Z3Charge(0), 2), (Z3Charge(2), 3), (Z3Charge(1), 1)],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let runtime = runtime();
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );
    let runtime_before = crate::typed_facade::completed_transformers();

    let error = TensorMap::<ExternalZ3Rule, f64>::zeros(engine, &runtime, [&codomain], [&domain])
        .unwrap_err();

    assert!(matches!(
        error,
        symtn::typed::Error::FusionAlgebra(_) | symtn::typed::Error::Operation(_)
    ));
    assert_eq!(
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        ),
        before
    );
    assert_eq!(
        crate::typed_facade::completed_transformers(),
        runtime_before
    );
}

// ---------------------------------------------------------------------------
// `from_subblock_fn` and inspection.
// ---------------------------------------------------------------------------

#[test]
fn from_block_fn_surfaces_a_decode_failure_as_the_codec_error() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a codec that cannot decode an id the engine produced fails the
    // construction with the provider's own error instead of panicking inside
    // the fill.
    let provider = Arc::new(external_z3_with(Quirk::NarrowDecode));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 2)]).unwrap();
    let runtime = runtime();

    // Two charge-1 codomain legs couple to charge 2, the id this codec refuses.
    let error = TensorMap::<ExternalZ3Rule, f64>::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |_, _| 1.0,
    )
    .unwrap_err();

    assert!(matches!(error, symtn::typed::Error::FusionAlgebra(_)));
}

#[test]
fn tensor_map_inspection_round_trips_the_spaces_and_blocks() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the legs come back as typed graded spaces with their labels and
    // dual flags intact, and every block reports decoded labels plus a data
    // view consistent with the buffer.
    let provider = Arc::new(external_z3());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::zeros(engine, &runtime, [&leg, &leg], [&dual]).unwrap();

    let codomain = tensor.codomain();
    let domain = tensor.domain();
    assert_eq!(codomain.len(), 2);
    assert_eq!(domain.len(), 1);
    assert_eq!(codomain[0].sectors().unwrap(), leg.sectors().unwrap());
    assert!(!codomain[0].is_dual());
    assert_eq!(domain[0].sectors().unwrap(), dual.sectors().unwrap());
    assert!(domain[0].is_dual());

    let mut elements = 0;
    for index in 0..tensor.subblock_count() {
        let sectors = tensor.subblock_fusion_trees(index).unwrap();
        assert_eq!(sectors.codomain_uncoupled().len(), 2);
        assert_eq!(sectors.domain_uncoupled().len(), 1);
        // Unique fusion: the two codomain charges fuse to the coupled charge.
        let sum = (sectors.codomain_uncoupled()[0].0 + sectors.codomain_uncoupled()[1].0) % 3;
        assert_eq!(&Z3Charge(sum), sectors.coupled());

        let block = tensor.subblock(index).unwrap();
        assert!(block.storage_end_exclusive().unwrap() <= tensor.dense_data().unwrap().len());
        elements += block.element_count().unwrap();
    }
    assert_eq!(elements, tensor.dense_data().unwrap().len());
}

#[test]
fn labelled_subblocks_borrow_u1_and_su2_values_in_canonical_order() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (symtn::sector::U1Irrep::new(-1), 1),
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, indices| {
            100.0 * trees.coupled().charge() as f64
                + 10.0 * indices[0] as f64
                + 3.0 * indices[1] as f64
                + indices[2] as f64
        })
        .unwrap();

    let mut u1_blocks = u1.subblocks().unwrap();
    assert_eq!(u1_blocks.len(), u1.subblock_count());
    for (index, (trees, values)) in u1_blocks.by_ref().enumerate() {
        let raw = u1.subblock(index).unwrap();
        assert_eq!(values.shape(), raw.shape());
        assert_eq!(values.strides(), raw.strides());
        assert_eq!(values.data().as_ptr(), u1.dense_data().unwrap().as_ptr());
        assert_eq!(trees.codomain_uncoupled().len(), 2);
        assert_eq!(
            trees.domain_uncoupled(),
            std::slice::from_ref(trees.coupled())
        );
        assert_eq!(
            trees.codomain_uncoupled()[0].charge() + trees.codomain_uncoupled()[1].charge(),
            trees.coupled().charge()
        );
        for i in 0..values.shape()[0] {
            for j in 0..values.shape()[1] {
                for k in 0..values.shape()[2] {
                    assert_eq!(
                        values.get(&[i, j, k]),
                        Some(
                            &(100.0 * trees.coupled().charge() as f64
                                + 10.0 * i as f64
                                + 3.0 * j as f64
                                + k as f64)
                        )
                    );
                }
            }
        }
    }
    assert!(matches!(
        u1.subblock(u1.subblock_count()),
        Err(symtn::typed::Error::Core(error))
            if matches!(*error, symtn::typed::CoreError::BlockIndexOutOfBounds { index, count }
                if index == count && count == u1.subblock_count())
    ));

    let su2 = su2_tensor_split(&runtime, 2);
    let su2_blocks = su2.subblocks().unwrap();
    assert_eq!(su2_blocks.len(), su2.subblock_count());
    for (index, (trees, values)) in su2_blocks.enumerate() {
        let raw = su2.subblock(index).unwrap();
        assert_eq!(trees.coupled(), &SU2Irrep::from_twice_spin(0));
        assert!(trees.domain_uncoupled().is_empty());
        assert!(SU2FusionRule
            .fusion_channels(
                SU2FusionRule
                    .encode_sector(&trees.codomain_uncoupled()[0])
                    .unwrap(),
                SU2FusionRule
                    .encode_sector(&trees.codomain_uncoupled()[1])
                    .unwrap(),
            )
            .contains(&SU2FusionRule.encode_sector(trees.coupled()).unwrap()));
        assert_eq!(values.shape(), raw.shape());
        assert_eq!(values.strides(), raw.strides());
        assert_eq!(values.data().as_ptr(), su2.dense_data().unwrap().as_ptr());
        assert_eq!(
            values.get(&vec![0; values.shape().len()]),
            Some(&su2.dense_data().unwrap()[raw.offset()])
        );
    }
}

#[test]
fn subblock_fusion_trees_reports_a_non_self_dual_domain_label() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a dual domain leg carrying charge 2 — whose dual is charge 1, so a
    // confusion between the two would show — is decoded as charge 2, matching
    // the convention that a tree labels a domain leg with the space's own
    // sector rather than its dual.
    let provider = Arc::new(external_z3());
    let codomain = GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(1), 1)]).unwrap();
    // A dual constructor interprets its key through the orientation, so key
    // charge 1 is stored and reported as the external charge 2.
    let domain = GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(1), 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    assert_eq!(
        domain.try_dual().unwrap().sectors().unwrap(),
        vec![Z3Charge(1)]
    );
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::zeros(engine, &runtime, [&codomain, &codomain], [&domain]).unwrap();

    assert_eq!(tensor.subblock_count(), 1);
    let sectors = tensor.subblock_fusion_trees(0).unwrap();
    assert_eq!(sectors.coupled(), &Z3Charge(2));
    assert_eq!(sectors.codomain_uncoupled(), &[Z3Charge(1), Z3Charge(1)]);
    assert_eq!(sectors.domain_uncoupled(), &[Z3Charge(2)]);
    assert!(tensor.domain()[0].is_dual());
}

// ---------------------------------------------------------------------------
// Cross-cutting gates.
// ---------------------------------------------------------------------------

#[test]
fn simple_fusion_provider_round_trips_construction_fill_and_inspection() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the whole phase-2 surface works for a non-abelian (Simple) external
    // provider with a complex payload, not just for the abelian fixture.
    let provider = Arc::new(external_su2());
    let leg = su2_leg(&provider, false);
    let runtime = runtime();

    let tensor: TensorMap<ExternalSu2Rule, Complex64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |sectors, indices| {
            Complex64::new(
                sectors.coupled().twice_spin() as f64,
                indices.iter().sum::<usize>() as f64,
            )
        },
    )
    .unwrap();

    // Two spin-1/2 legs couple to spin 0 and spin 1 on each side.
    assert!(tensor.subblock_count() >= 2);
    let coupled: Vec<usize> = (0..tensor.subblock_count())
        .map(|index| {
            tensor
                .subblock_fusion_trees(index)
                .unwrap()
                .coupled()
                .twice_spin()
        })
        .collect();
    assert!(coupled.contains(&0) && coupled.contains(&2));
    assert!(tensor
        .dense_data()
        .unwrap()
        .iter()
        .zip(0..)
        .all(|(value, _)| value.re == 0.0 || value.re == 2.0));
    assert_eq!(tensor.codomain().len(), 2);
    assert_eq!(tensor.domain().len(), 2);
}

#[test]
fn typed_block_fill_preserves_tree_and_storage_order() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: each typed block exposes the same tree and column-major indices
    // that drove its fill closure, without borrowing the retired facade as an
    // oracle.
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::Z2FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed: TensorMap<symtn::sector::Z2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], typed_fill_value)
            .unwrap();

    assert!(typed.subblock_count() > 1);
    for index in 0..typed.subblock_count() {
        let block = typed.subblock(index).unwrap();
        let sectors = typed.subblock_fusion_trees(index).unwrap();
        for local in 0..block.element_count().unwrap() {
            let mut remainder = local;
            let indices: Vec<_> = block
                .shape()
                .iter()
                .map(|&extent| {
                    let index = remainder % extent;
                    remainder /= extent;
                    index
                })
                .collect();
            let offset = block.offset()
                + indices
                    .iter()
                    .zip(block.strides())
                    .map(|(&index, &stride)| index * stride)
                    .sum::<usize>();
            assert_eq!(
                typed.dense_data().unwrap()[offset],
                typed_fill_value(&sectors, &indices)
            );
        }
    }
}

#[test]
fn rand_and_isomorphism_build_on_an_external_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: nothing in the new constructors needs a built-in rule — a
    // downstream Z3 provider gets a populated random tensor and a lawful
    // structural isomorphism from the same public vocabulary.
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();

    let random: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg], 0x9E37_79B9_7F4A_7C15).unwrap();
    assert!(random
        .dense_data()
        .unwrap()
        .iter()
        .any(|&value| value != 0.0));

    // The fused non-dual isomorph of the dual leg has the same sector content.
    let f: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&leg, &dual], [&dual, &leg]).unwrap();
    let roundtrip = f.adjoint(engine).unwrap().compose(engine, &f).unwrap();
    // Explicit `D`: under cuda,cpu-faer serde_json adds a `PartialEq` impl
    // that makes the `assert_eq!` unable to pin `D` on its own (E0283).
    let id: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&dual, &leg], [&dual, &leg]).unwrap();
    assert_eq!(roundtrip.dense_data().unwrap(), id.dense_data().unwrap());
}

#[test]
fn typed_structural_constructors_reject_an_empty_leg_list() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the provider is inferred from the legs, so an empty construction
    // has nothing to infer it from.
    let runtime = runtime();
    let empty: [&GradedSpace<ExternalZ3Rule>; 0] = [];

    assert!(
        TensorMap::<ExternalZ3Rule, f64>::zeros(engine, &runtime, empty, empty)
            .is_invalid_argument("at least one leg")
    );
    assert!(TensorMap::<ExternalZ3Rule, f64>::rand_with_seed(
        engine,
        &runtime,
        empty,
        empty,
        0x9E37_79B9_7F4A_7C15
    )
    .is_invalid_argument("at least one leg"));
    assert!(matches!(
        TensorMap::<ExternalZ3Rule, f64>::isomorphism(engine, &runtime, empty, empty).unwrap_err(),
        symtn::typed::Error::InvalidArgument(message) if message.contains("at least one leg")
    ));
    assert!(
        TensorMap::<ExternalZ3Rule, f64>::isometry(engine, &runtime, empty, empty)
            .is_invalid_argument("at least one leg")
    );
}

#[test]
fn typed_rand_rejects_mismatched_providers_like_zeros() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: two providers with distinct rule identities are a rule mismatch on
    // the random constructor exactly as on `zeros`.
    let first = Arc::new(external_z3_tagged(0));
    let second = Arc::new(external_z3_tagged(1));
    let runtime = runtime();

    let error = TensorMap::<ExternalZ3Rule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&z3_leg(&first, false)],
        [&z3_leg(&second, true)],
        0x9E37_79B9_7F4A_7C15,
    )
    .unwrap_err();
    assert!(matches!(error, symtn::typed::Error::RuleMismatch));
}

#[test]
fn typed_isomorphism_is_unitary_on_the_norm_fuser_shape() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the norm-fuser shape carries an asymmetric dual arrangement and
    // still satisfies the defining isomorphism law.
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let typed_v = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (symtn::sector::U1Irrep::new(0), 1),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let typed_dual = typed_v.try_dual().unwrap();
    // fuse(dual(v) ⊗ v) by hand: charges -1, 0 (twice), 1.
    let typed_fused = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (symtn::sector::U1Irrep::new(-1), 1),
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let typed: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&typed_fused], [&typed_dual, &typed_v]).unwrap();
    let identity: TensorMap<symtn::sector::U1FusionRule, f64> = TensorMap::isomorphism(
        engine,
        &runtime,
        [&typed_dual, &typed_v],
        [&typed_dual, &typed_v],
    )
    .unwrap();

    assert_eq!(
        typed
            .adjoint(engine)
            .unwrap()
            .compose(engine, &typed)
            .unwrap()
            .dense_data()
            .unwrap(),
        identity.dense_data().unwrap()
    );
}

#[test]
fn typed_c64_unitary_satisfies_the_identity_law() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let leg = fz2_typed_leg();
    let leg_dual = leg.try_dual().unwrap();
    let typed: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        TensorMap::isomorphism(engine, &runtime, [&leg, &leg_dual], [&leg_dual, &leg]).unwrap();
    let identity: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        TensorMap::isomorphism(engine, &runtime, [&leg_dual, &leg], [&leg_dual, &leg]).unwrap();

    assert_eq!(
        typed
            .adjoint(engine)
            .unwrap()
            .compose(engine, &typed)
            .unwrap()
            .dense_data()
            .unwrap(),
        identity.dense_data().unwrap()
    );
}

#[test]
fn typed_su2_isomorphism_satisfies_the_identity_law() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: nothing in the structural constructors is abelian-specific — a
    // simple-fusion SU(2) provider's isomorphism still satisfies `f† ∘ f =
    // id` through genuinely non-trivial recoupling.
    let runtime = runtime();
    let provider = Arc::new(external_su2());
    let leg = su2_leg(&provider, false);

    let f: TensorMap<ExternalSu2Rule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&leg, &leg], [&leg, &leg]).unwrap();
    let roundtrip = f.adjoint(engine).unwrap().compose(engine, &f).unwrap();
    // Explicit `D` for the cuda,cpu-faer feature set (see the ExternalZ3
    // identity-law test above).
    let id: TensorMap<ExternalSu2Rule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&leg, &leg], [&leg, &leg]).unwrap();
    assert_eq!(roundtrip.dense_data().unwrap(), id.dense_data().unwrap());
}

#[test]
fn typed_leg_dims_include_quantum_dimensions() {
    let _serial = crate::counting_alloc::serial();
    // Gate 2: `leg_dims` values on a built-in abelian rule and on
    // built-in SU(2), whose non-abelian sectors
    // are what make the quantum-dimension weighting visible (spin-1/2 with
    // degeneracy 2 contributes 4, not 2).
    let runtime = runtime();

    let typed = z2_tensor(&runtime);
    assert_eq!(typed.leg_dims().unwrap(), vec![5, 5, 5]);

    let su2_typed = su2_tensor(&runtime);
    assert_eq!(su2_typed.leg_dims().unwrap(), vec![5, 5]);
}

#[test]
fn typed_leg_dims_carry_an_external_provider_with_nontrivial_dimensions() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 2's external-provider leg: `ExternalSu2` reports quantum dimension
    // 2 for twice-spin 1, so the degeneracy-2 leg weighs 4. This pins an
    // external provider rather than a built-in special case.
    let runtime = runtime();
    let provider = Arc::new(external_su2());
    let leg = su2_leg(&provider, false);
    let tensor: TensorMap<ExternalSu2Rule, f64> =
        TensorMap::zeros(engine, &runtime, [&leg], [&leg]).unwrap();

    assert_eq!(tensor.leg_dims().unwrap(), vec![4, 4]);
}

#[test]
fn typed_scalar_reads_a_complex_full_contraction() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 4's c64 leg: a complex endomorphism traced to rank 0.
    let runtime = runtime();
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed: TensorMap<symtn::sector::Z2FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |sectors, indices| {
            complex(typed_fill_value(sectors, indices))
        })
        .unwrap();

    let typed_value: Complex64 = typed
        .trace_pairs(engine, &[(0, 1)])
        .unwrap()
        .scalar(engine)
        .unwrap();
    assert_eq!(typed_value, typed.tr(engine).unwrap());
}

#[test]
fn typed_scalar_on_a_tensor_with_legs_is_an_invalid_argument() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let typed_error = typed.scalar(engine).unwrap_err();
    assert!(matches!(
        typed_error,
        symtn::typed::Error::InvalidArgument(_)
    ));
}

#[test]
fn typed_to_c64_widens_dense_and_compact_values_exactly() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 6: widening adds an exactly zero imaginary component on dense and
    // compact diagonal payloads.
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let typed_wide: TensorMap<symtn::sector::Z2FusionRule, Complex64> =
        typed.convert::<Complex64>(engine).unwrap();
    assert!(typed_wide
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.dense_data().unwrap())
        .all(|(&wide, &real)| wide == Complex64::new(real, 0.0)));

    let typed_s: TensorMap<symtn::sector::Z2FusionRule, f64> =
        typed.svd_compact(engine, &[0, 1], &[2]).unwrap().s;
    let typed_s_wide: TensorMap<symtn::sector::Z2FusionRule, Complex64> =
        typed_s.convert::<Complex64>(engine).unwrap();
    assert!(typed_s_wide
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed_s.materialize(engine).unwrap().dense_data().unwrap())
        .all(|(&wide, &real)| wide == Complex64::new(real, 0.0)));
}

#[test]
fn typed_re_im_reconstruct_the_complex_tensor_byte_exactly() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 6's law checks: `re(t) + i*im(t)` rebuilds `t` byte-exactly through
    // `to_c64` + `add`,
    // and on a real tensor `re(to_c64(x)) == x`, `im(to_c64(x)) ==
    // zeros_like(x)`, byte-exactly.
    let runtime = runtime();
    let complex_typed = z2_complex_tensor(&runtime);

    let real_part: TensorMap<symtn::sector::Z2FusionRule, f64> = complex_typed.re(engine).unwrap();
    let imag_part: TensorMap<symtn::sector::Z2FusionRule, f64> = complex_typed.im(engine).unwrap();
    let operand = Complex64::new(0.0, 1.0);
    let imag_part_c64 = imag_part.convert::<Complex64>(engine).unwrap();
    let rebuilt: TensorMap<symtn::sector::Z2FusionRule, Complex64> = real_part
        .convert::<Complex64>(engine)
        .unwrap()
        .axpby(engine, Complex64::new(1.0, 0.0), &imag_part_c64, operand)
        .unwrap();
    assert_eq!(
        rebuilt.dense_data().unwrap(),
        complex_typed.dense_data().unwrap()
    );

    let real_typed = z2_tensor(&runtime);
    let round_trip: TensorMap<symtn::sector::Z2FusionRule, f64> = real_typed
        .convert::<Complex64>(engine)
        .unwrap()
        .re(engine)
        .unwrap();
    assert_eq!(
        round_trip.dense_data().unwrap(),
        real_typed.dense_data().unwrap()
    );
    let vanished: TensorMap<symtn::sector::Z2FusionRule, f64> = real_typed
        .convert::<Complex64>(engine)
        .unwrap()
        .im(engine)
        .unwrap();
    assert_eq!(
        vanished.dense_data().unwrap(),
        real_typed.zeros_like(engine).unwrap().dense_data().unwrap()
    );
}

#[test]
fn typed_re_im_keep_a_compact_spectrum_on_its_bond_space() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 6's diagonal leg: `re`/`im` of a complex compact spectrum factor
    // map spectrum-to-spectrum, and the law `re + i*im == t` holds on the
    // materialized payloads.
    let runtime = runtime();
    let complex_typed = z2_complex_tensor(&runtime);
    let s: TensorMap<symtn::sector::Z2FusionRule, Complex64> =
        complex_typed.svd_compact(engine, &[0, 1], &[2]).unwrap().s;

    let real_part: TensorMap<symtn::sector::Z2FusionRule, f64> = s.re(engine).unwrap();
    let imag_part: TensorMap<symtn::sector::Z2FusionRule, f64> = s.im(engine).unwrap();
    assert_same_legs(&real_part.codomain(), &s.codomain());
    let operand = Complex64::new(0.0, 1.0);
    let imag_part_c64 = imag_part.convert::<Complex64>(engine).unwrap();
    let rebuilt: TensorMap<symtn::sector::Z2FusionRule, Complex64> = real_part
        .convert::<Complex64>(engine)
        .unwrap()
        .axpby(engine, Complex64::new(1.0, 0.0), &imag_part_c64, operand)
        .unwrap();
    assert_eq!(
        rebuilt.materialize(engine).unwrap().dense_data().unwrap(),
        s.materialize(engine).unwrap().dense_data().unwrap()
    );
}

#[test]
fn typed_leg_dims_route_each_axis_to_its_own_leg() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 2's axis-routing leg: every earlier fixture has homogeneous leg
    // dimensions, so a leg_dims that reads the wrong leg (e.g. domain[0] for
    // the last codomain axis) still passes them. Three pairwise-distinct
    // dimensions in a 2 <- 1 split make any misrouting visible, per axis.
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::Z2FusionRule);
    let z2_pairs = |even: usize, odd: usize| {
        [
            (symtn::sector::Z2Irrep::EVEN, even),
            (symtn::sector::Z2Irrep::ODD, odd),
        ]
    };
    let a = GradedSpace::try_new(Arc::clone(&provider), z2_pairs(2, 3)).unwrap();
    let b = GradedSpace::try_new(Arc::clone(&provider), z2_pairs(1, 1)).unwrap();
    let c = GradedSpace::try_new(Arc::clone(&provider), z2_pairs(3, 4)).unwrap();
    let typed: TensorMap<symtn::sector::Z2FusionRule, f64> =
        TensorMap::zeros(engine, &runtime, [&a, &b], [&c]).unwrap();
    assert_eq!(typed.leg_dims().unwrap(), vec![5, 2, 7]);
}

#[test]
fn tensor_map_zeros_builds_a_multi_block_checked_layout() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `zeros` admits a runtime-rank multi-block layout through the
    // checked path and returns a zero buffer of exactly the layout's length.
    let provider = Arc::new(external_z3());
    let leg = z3_leg(&provider, false);
    let dual = leg.try_dual().unwrap();
    let runtime = runtime();

    let tensor: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::zeros(engine, &runtime, [&leg, &leg], [&dual, &dual]).unwrap();

    assert!(tensor.subblock_count() >= 2);
    assert!(!tensor.dense_data().unwrap().is_empty());
    assert!(tensor
        .dense_data()
        .unwrap()
        .iter()
        .all(|&value| value == 0.0));
}

#[test]
fn id_writes_the_nonuniform_fused_diagonal() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the identity on a leg list whose per-sector degeneracies differ
    // from leg to leg — the case where the
    // diagonal offsets inside a coupled-sector block are not all the same.
    let runtime = runtime();
    let z2 = Arc::new(symtn::sector::Z2FusionRule);
    let typed_leg = |even, odd| {
        GradedSpace::try_new(
            Arc::clone(&z2),
            [
                (symtn::sector::Z2Irrep::EVEN, even),
                (symtn::sector::Z2Irrep::ODD, odd),
            ],
        )
        .unwrap()
    };

    let wide = typed_leg(2, 3);
    let narrow = typed_leg(1, 4);
    let typed =
        TensorMap::<_, f64>::isomorphism(engine, &runtime, [&wide, &narrow], [&wide, &narrow])
            .unwrap();

    // Not the zero tensor, and not the all-ones one either: a genuine diagonal.
    // 14 even + 11 odd fused states: `2*1 + 3*4` and `2*4 + 3*1`.
    assert_eq!(
        typed
            .dense_data()
            .unwrap()
            .iter()
            .filter(|&&v| v == 1.0)
            .count(),
        25
    );
    assert!(typed.dense_data().unwrap().contains(&0.0));
}
