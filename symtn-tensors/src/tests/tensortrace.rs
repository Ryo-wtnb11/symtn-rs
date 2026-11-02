use super::*;
use crate::probe::{self, Event};
use std::sync::Arc;

/// Its answers follow `TRACE_TEST_WRONG_STYLE` (admission) and
/// `TRACE_TEST_FAIL_TWIST` (lowering only, never cached), so its identity
/// is per instance and changes with the style switch: equal identity means
/// equal admission and symbol answers.
#[derive(Clone, Debug)]
struct CheckedTraceToy(symtn_core::BraidingStyleKind, symtn_core::RuleIdentity);

impl CheckedTraceToy {
    fn new(braiding: symtn_core::BraidingStyleKind) -> Self {
        Self(braiding, symtn_core::RuleIdentity::new_unique::<Self>())
    }
}

/// The identity of every `CheckedTraceToy` while its fusion style is wrong.
struct WrongStyleCheckedTraceToy;

impl symtn_core::testing::Provider for CheckedTraceToy {}

impl symtn_core::CheckedGenericFusion for CheckedTraceToy {
    type Error = std::io::Error;
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        if TRACE_TEST_WRONG_STYLE.get() {
            symtn_core::RuleIdentity::of_type::<WrongStyleCheckedTraceToy>()
        } else {
            self.1.clone()
        }
    }
    fn fusion_style(&self) -> FusionStyleKind {
        if TRACE_TEST_WRONG_STYLE.get() {
            FusionStyleKind::Unique
        } else {
            FusionStyleKind::Generic
        }
    }
    fn braiding_style(&self) -> symtn_core::BraidingStyleKind {
        self.0
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(sector)
    }
    fn try_fusion_channels(
        &self,
        _left: SectorId,
        _right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(vec![SectorId::new(0)].into())
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }
    fn try_nsymbol(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(1)
    }
}

impl symtn_core::CheckedGenericRigidSymbols for CheckedTraceToy {
    type Scalar = f64;
    fn try_sqrt_dim_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn try_inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn try_frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn try_f_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
        _d: SectorId,
        _e: SectorId,
        _f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        Ok(GenericFArray::new(vec![1.0], (1, 1, 1, 1)))
    }
    fn try_r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        Ok(GenericRMatrix::new(vec![1.0], 1, 1))
    }
}

impl symtn_core::CheckedGenericPivotal for CheckedTraceToy {
    fn try_twist_scalar(&self, _sector: SectorId) -> Result<f64, Self::Error> {
        TRACE_TEST_TWIST_CALLS.set(TRACE_TEST_TWIST_CALLS.get() + 1);
        if TRACE_TEST_FAIL_TWIST.get() {
            return Err(std::io::Error::other("trace pivotal failure"));
        }
        Ok(1.0)
    }
}

#[test]
fn checked_generic_owned_trace_returns_the_traced_scalar_value() {
    let leg = || SectorLeg::new([(SectorId::new(0), 1)], false);
    let source_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let destination_hom =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let provider = Arc::new(CheckedTraceToy::new(symtn_core::BraidingStyleKind::Bosonic));
    let src = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        source_hom,
    )
    .unwrap();
    let dst =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, destination_hom)
            .unwrap();
    let output = crate::tensortrace::tensortrace_fusion_dyn_owned_generic_checked(
        strided_kernel::Exec::serial(),
        &dst,
        &src,
        &[2.0],
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
    )
    .unwrap();
    assert_eq!(output, vec![2.0]);
}

#[test]
fn checked_generic_trace_preflight_gates_non_symmetric_braiding_first() {
    // What: as TensorKit `trace_permute!`, the Generic preflight rejects a
    // non-symmetric braiding before it looks at the pairs, as the
    // multiplicity-free compile already did.
    let leg = || SectorLeg::new([(SectorId::new(0), 1)], false);
    let src = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(CheckedTraceToy::new(symtn_core::BraidingStyleKind::Anyonic)),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    for axes in [
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        // A malformed pair (an axis traced with itself): the braiding still wins.
        TensorTraceAxisSpec::new(&[1], &[0], &[0]),
    ] {
        let error =
            crate::tensortrace::tensortrace_fusion_dyn_preflight_generic_checked(&src, axes, 0)
                .unwrap_err();
        assert!(
            matches!(
                error,
                CheckedGenericPlanError::Operation(
                    OperationError::UnsupportedTensorContractScope {
                        message: crate::admission::FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING
                    }
                )
            ),
            "{error:?}"
        );
    }
}

fn assert_owned_trace_matches_initialized_for_real_and_complex<R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    src: &BoundDynamicFusionMapSpace<R>,
    axes: TensorTraceAxisSpec<'_>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let src_len = src.space().required_len().unwrap();
    let dst_len = dst.space().required_len().unwrap();
    let real_source = (0..src_len)
        .map(|index| index as f64 + 0.25)
        .collect::<Vec<_>>();
    let mut expected_real = vec![0.0; dst_len];
    tensortrace_fusion_dyn_into(
        strided_kernel::Exec::serial(),
        dst,
        &mut expected_real,
        src,
        &real_source,
        axes,
        1.0,
        0.0,
    )
    .unwrap();
    let actual_real = tensortrace_fusion_dyn_owned(
        strided_kernel::Exec::serial(),
        dst,
        src,
        &real_source,
        axes,
        1.0,
    )
    .unwrap();
    // The owned writer and the initialized writer are two paths whose
    // agreement is the contract; each destination entry sums at most
    // `src_len` recoupled source entries.
    crate::test_numerics::numerics::assert_slices_close(
        "owned vs initialized real trace",
        &actual_real,
        &expected_real,
        src_len,
    );

    let complex_source = real_source
        .iter()
        .enumerate()
        .map(|(index, &real)| Complex64::new(real, index as f64 + 0.5))
        .collect::<Vec<_>>();
    let mut expected_complex = vec![Complex64::new(0.0, 0.0); dst_len];
    tensortrace_fusion_dyn_into(
        strided_kernel::Exec::serial(),
        dst,
        &mut expected_complex,
        src,
        &complex_source,
        axes,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    let actual_complex = tensortrace_fusion_dyn_owned(
        strided_kernel::Exec::serial(),
        dst,
        src,
        &complex_source,
        axes,
        Complex64::new(1.0, 0.0),
    )
    .unwrap();
    crate::test_numerics::numerics::assert_slices_close(
        "owned vs initialized complex trace",
        &actual_complex,
        &expected_complex,
        src_len,
    );
}

#[test]
fn public_dynamic_fz2_trace_validates_exact_extents_before_mutation() {
    struct Case {
        name: &'static str,
        dst: Vec<f64>,
        src: Vec<f64>,
        error: Option<(usize, usize)>,
        expected_dst: Vec<f64>,
    }

    let provider = std::sync::Arc::new(FermionParityFusionRule);
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let traced_leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([traced_leg()]),
        FusionProductSpace::new([traced_leg()]),
    );
    let src_space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&provider),
        src_homspace,
    )
    .unwrap();
    let dst_space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        provider,
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
    )
    .unwrap();
    let cases = [
        Case {
            name: "oversized destination",
            dst: vec![10.0, 999.0],
            src: vec![2.0, 3.0],
            error: Some((1, 2)),
            expected_dst: vec![10.0, 999.0],
        },
        Case {
            name: "short destination",
            dst: vec![],
            src: vec![2.0, 3.0],
            error: Some((1, 0)),
            expected_dst: vec![],
        },
        Case {
            name: "oversized source",
            dst: vec![10.0],
            src: vec![2.0, 3.0, 999.0],
            error: Some((2, 3)),
            expected_dst: vec![10.0],
        },
        Case {
            name: "short source",
            dst: vec![10.0],
            src: vec![2.0],
            error: Some((2, 1)),
            expected_dst: vec![10.0],
        },
        Case {
            name: "exact destination and source",
            dst: vec![10.0],
            src: vec![2.0, 3.0],
            error: None,
            expected_dst: vec![4.0],
        },
    ];

    for case in cases {
        let mut dst = case.dst;
        let result = tensortrace_fusion_dyn_into(
            strided_kernel::Exec::serial(),
            &dst_space,
            &mut dst,
            &src_space,
            &case.src,
            TensorTraceAxisSpec::new(&[], &[0], &[1]),
            1.0,
            0.5,
        );
        match case.error {
            Some((expected, actual)) => {
                assert_eq!(
                    result,
                    Err(OperationError::ElementCountMismatch { expected, actual }),
                    "{}",
                    case.name
                );
                assert_eq!(dst, case.expected_dst, "{}", case.name);
            }
            None => {
                result.unwrap();
                assert_eq!(dst, case.expected_dst, "{}", case.name);
            }
        }
    }
}

#[test]
fn owned_trace_matches_initialized_across_supported_multiplicity_free_rules() {
    // What: the owned writer preserves the initialized trace oracle for both
    // storage dtypes across U1, SU2, fermionic, and product coefficient paths.
    fn scalar_destination<R>(provider: std::sync::Arc<R>) -> BoundDynamicFusionMapSpace<R>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            provider,
            FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
        )
        .unwrap()
    }

    let u1_provider = std::sync::Arc::new(U1FusionRule);
    let u1_leg = || SectorLeg::new([(U1Irrep::new(1).sector_id(), 2)], false);
    let u1_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&u1_provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([u1_leg()]),
            FusionProductSpace::new([u1_leg()]),
        ),
    )
    .unwrap();
    let u1_dst = scalar_destination(u1_provider);
    assert_owned_trace_matches_initialized_for_real_and_complex(
        &u1_dst,
        &u1_src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
    );

    let su2_provider = std::sync::Arc::new(SU2FusionRule);
    let su2_leg = || SectorLeg::new([(SU2Irrep::from_twice_spin(1).sector_id(), 2)], false);
    let su2_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&su2_provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([su2_leg()]),
            FusionProductSpace::new([su2_leg()]),
        ),
    )
    .unwrap();
    let su2_dst = scalar_destination(su2_provider);
    assert_owned_trace_matches_initialized_for_real_and_complex(
        &su2_dst,
        &su2_src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
    );

    let fz2_provider = std::sync::Arc::new(FermionParityFusionRule);
    let fz2_leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let fz2_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([fz2_leg()]),
        FusionProductSpace::new([fz2_leg()]),
    );
    let fz2_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&fz2_provider),
        fz2_hom.clone(),
    )
    .unwrap();
    let fz2_dst = scalar_destination(fz2_provider);
    assert_owned_trace_matches_initialized_for_real_and_complex(
        &fz2_dst,
        &fz2_src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
    );

    let left_rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let product_provider = std::sync::Arc::new(
        FpU1Su2Rule::try_new(
            FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let product_sector = product_provider.encode_component_ids(
        left_rule.encode_component_ids(SectorId::new(1), U1Irrep::new(1).sector_id()),
        SU2Irrep::from_twice_spin(1).sector_id(),
    );
    let product_leg = || SectorLeg::new([(product_sector, 2)], false);
    let product_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&product_provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([product_leg()]),
            FusionProductSpace::new([product_leg()]),
        ),
    )
    .unwrap();
    let product_dst = scalar_destination(product_provider);
    assert_owned_trace_matches_initialized_for_real_and_complex(
        &product_dst,
        &product_src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
    );
}

/// One canonical fZ2 space used as both trace destination and source.
fn fz2_identity_trace_space() -> DynamicFusionMapSpace {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    DynamicFusionMapSpace::from_final_homspace(&FermionParityFusionRule, homspace).unwrap()
}

#[test]
fn fz2_trace_raw_entry_validates_host_slice_extents_before_mutation() {
    struct Case {
        name: &'static str,
        dst: Vec<f64>,
        src: Vec<f64>,
        expected: usize,
        actual: usize,
    }

    let rule = FermionParityFusionRule;
    let space = fz2_identity_trace_space();
    let (dst_space, src_space) = (&space, &space);
    let exact = vec![10.0, 20.0, 30.0, 40.0];
    let cases = [
        Case {
            name: "short destination",
            dst: exact[..3].to_vec(),
            src: exact.clone(),
            expected: 4,
            actual: 3,
        },
        Case {
            name: "oversized destination",
            dst: [exact.as_slice(), &[55.0]].concat(),
            src: exact.clone(),
            expected: 4,
            actual: 5,
        },
        Case {
            name: "short source",
            dst: exact.clone(),
            src: exact[..3].to_vec(),
            expected: 4,
            actual: 3,
        },
        Case {
            name: "oversized source",
            dst: exact.clone(),
            src: [exact.as_slice(), &[55.0]].concat(),
            expected: 4,
            actual: 5,
        },
    ];

    for case in cases {
        let initial_dst = case.dst.clone();
        let mut dst = case.dst;
        let result = crate::tensortrace::tensortrace_fusion_dyn_into_raw(
            strided_kernel::Exec::serial(),
            &rule,
            dst_space,
            &mut dst,
            src_space,
            &case.src,
            TensorTraceAxisSpec::new(&[0, 1], &[], &[]),
            1.0,
            0.5,
        );

        // What: replay enforces the exact destination-first extent contract
        // before beta can reach a logical layout.
        assert_eq!(
            result,
            Err(OperationError::ElementCountMismatch {
                expected: case.expected,
                actual: case.actual,
            }),
            "{}",
            case.name
        );
        assert_eq!(dst, initial_dst, "{}", case.name);
    }
}

#[test]
fn fz2_lowered_dynamic_trace_matches_encoded_data_oracle() {
    // What: the closed fZ2 trace produces byte-for-byte identical reduced data
    // through encoded and lowered bindings.
    let provider = std::sync::Arc::new(symtn_core::FermionParityFusionRule);
    let odd = symtn_core::Z2Irrep::ODD.sector_id();
    let leg = || FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]);
    let src_homspace = FusionTreeHomSpace::new(leg(), leg());
    let encoded_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&provider),
        src_homspace.clone(),
    )
    .unwrap();
    let lowered_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        std::sync::Arc::clone(&provider),
        src_homspace,
    )
    .unwrap();
    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let encoded_dst = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&provider),
        scalar.clone(),
    )
    .unwrap();
    let lowered_dst =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(provider, scalar)
            .unwrap();
    let axes = TensorTraceAxisSpec::new(&[], &[0], &[1]);
    let mut encoded_data = vec![7.0_f64];
    let mut lowered_data = encoded_data.clone();
    tensortrace_fusion_dyn_into(
        strided_kernel::Exec::serial(),
        &encoded_dst,
        &mut encoded_data,
        &encoded_src,
        &[5.0],
        axes,
        2.0,
        3.0,
    )
    .unwrap();
    tensortrace_fusion_dyn_into(
        strided_kernel::Exec::serial(),
        &lowered_dst,
        &mut lowered_data,
        &lowered_src,
        &[5.0],
        axes,
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(lowered_data, encoded_data);
}

#[test]
fn tensortrace_fusion_rejects_invalid_axis_sets_at_compile_time() {
    let rule = FermionParityFusionRule;
    let parity = [(SectorId::new(0), 1), (SectorId::new(1), 1)];
    let src_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(parity, false)]),
            FusionProductSpace::new([SectorLeg::new(parity, false)]),
        ),
    )
    .unwrap();
    let src: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![2.0, 3.0], src_space).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
    )
    .unwrap();
    let dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    let err = dynamic_tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[0]),
    )
    .unwrap_err();
    assert!(matches!(err, OperationError::InvalidAxisSet { .. }));

    let err = dynamic_tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[]),
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::TraceAxisCountMismatch { lhs: 1, rhs: 0 }
    );
}

#[test]
fn tensortrace_fusion_fermion_parity_matches_tensorkit_supertrace() {
    // The probe below watches the compile itself: alone in a child process
    // on emptied caches, so no sibling's resident compiled trace answers it.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_SUPERTRACE_ISOLATED",
        "tests::tensortrace::tensortrace_fusion_fermion_parity_matches_tensorkit_supertrace",
    ) {
        return;
    }
    symtn_core::clear_structure_caches();
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![2.0, 3.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    probe::reset_trace_sources();
    let structure = dynamic_tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
    )
    .unwrap();
    assert_eq!(structure.terms().len(), 2);
    // What: Unique fusion retains the scalar direct path instead of constructing
    // a fusion-group recoupling matrix.
    assert_eq!(
        probe::take_trace_sources().len(),
        src.structure().block_count()
    );

    replay_tensortrace_fusion_into(&structure, &mut dst, &src, 1.0, 0.0).unwrap();

    assert_eq!(dst.data(), &[-1.0]);
}

#[test]
fn tensortrace_fusion_fermion_supertrace_uses_degeneracy_diagonals() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 2), (odd, 2)], false)]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![
            1.0, 2.0, 3.0, 4.0, // even block, column-major diagonal 1 + 4
            5.0, 6.0, 7.0, 8.0, // odd block, column-major diagonal 5 + 8
        ],
        src_space,
    )
    .unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![10.0], dst_space).unwrap();

    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[14.0]);
}

#[test]
fn tensortrace_fusion_with_conjugation_lowers_lazy_adjoint_supertrace() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<Complex64> = TestTensor::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, 4.0)],
        src_space,
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(vec![Complex64::new(10.0, 20.0)], dst_space)
            .unwrap();

    let adjoint_view_build_count_rec = probe::record();
    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new_with_conjugation(&[], &[0], &[1], true),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(-1.0, 3.0)]);
    // What: typed conjugating trace reads the parent block grid through its
    // orientation instead of constructing an adjoint block structure.
    assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);
}

#[test]
fn tensortrace_fusion_rank_zero_adjoint_keeps_parent_structure() {
    let rule = FermionParityFusionRule;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let space = || DynamicFusionMapSpace::from_final_homspace(&rule, homspace.clone()).unwrap();
    let src: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(vec![Complex64::new(2.0, 3.0)], space()).unwrap();
    let mut dst: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(vec![Complex64::new(0.0, 0.0)], space()).unwrap();

    let adjoint_view_build_count_rec = probe::record();
    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new_with_conjugation(&[], &[], &[], true),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    // What: the empty tree pair is still an oriented parent block, and source
    // conjugation does not require an owned rank-zero adjoint structure.
    assert_eq!(dst.data(), &[Complex64::new(2.0, -3.0)]);
    assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);
}

#[test]
fn dynamic_u1_conjugating_trace_matches_hand_indexed_logical_adjoint() {
    let rule = U1FusionRule;
    let charge = U1Irrep::new(1).sector_id();
    let neutral = U1Irrep::new(0).sector_id();
    let charged_leg = || SectorLeg::new([(charge, 2)], false);
    let neutral_leg = || SectorLeg::new([(neutral, 3)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([charged_leg(), neutral_leg()]),
        FusionProductSpace::new([charged_leg()]),
    );
    let src_fusion = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom.clone()).unwrap();
    let src_data = (1..=12)
        .map(|value| Complex64::new(value as f64, value as f64 + 0.25))
        .collect::<Vec<_>>();
    let src: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(src_data.clone(), src_fusion).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([]),
        FusionProductSpace::new([neutral_leg()]),
    );
    let dst_fusion = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom.clone()).unwrap();
    let initial = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
    ];
    let alpha = Complex64::new(2.0, -1.0);
    let beta = Complex64::new(-0.5, 0.25);
    let axes = TensorTraceAxisSpec::new_with_conjugation(&[1], &[0], &[2], true);

    // What: source storage is column-major [charged, neutral, charged].
    // The logical adjoint swaps the HomSpace sides, remaps axes, and traces
    // exactly the original storage diagonal before conjugating each value.
    let hand_trace = [
        src_data[0].conj() + src_data[7].conj(),
        src_data[2].conj() + src_data[9].conj(),
        src_data[4].conj() + src_data[11].conj(),
    ];
    let expected = initial
        .iter()
        .zip(hand_trace)
        .map(|(&dst, trace)| beta * dst + alpha * trace)
        .collect::<Vec<_>>();

    let adjoint_view_build_count_rec = probe::record();
    let mut typed: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(initial.clone(), dst_fusion).unwrap();
    dynamic_tensortrace_fusion_into(&rule, &mut typed, &src, axes, alpha, beta).unwrap();
    assert_eq!(typed.data(), expected);

    let provider = std::sync::Arc::new(rule);
    let dynamic_src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&provider),
        src_hom,
    )
    .unwrap();
    let dynamic_dst =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(provider, dst_hom)
            .unwrap();
    let mut unchecked = initial.clone();
    tensortrace_fusion_dyn_into(
        strided_kernel::Exec::serial(),
        &dynamic_dst,
        &mut unchecked,
        &dynamic_src,
        &src_data,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(unchecked, expected);

    let mut checked = initial.clone();
    tensortrace_fusion_dyn_into_checked(
        strided_kernel::Exec::serial(),
        &dynamic_dst,
        &mut checked,
        &dynamic_src,
        &src_data,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(checked, expected);
    let _ = (
        probe::take(Event::TraceAxisPlanDerivation),
        probe::take(Event::TraceSelectedHomspaceDerivation),
    );
    let owned = tensortrace_fusion_dyn_owned_checked(
        strided_kernel::Exec::serial(),
        &dynamic_dst,
        &dynamic_src,
        &src_data,
        axes,
        alpha,
    )
    .unwrap();
    // What: one checked compile derives one axis plan and one selected HomSpace.
    assert_eq!(
        (
            probe::take(Event::TraceAxisPlanDerivation),
            probe::take(Event::TraceSelectedHomspaceDerivation)
        ),
        (1, 1)
    );
    assert_eq!(
        owned,
        hand_trace
            .into_iter()
            .map(|trace| alpha * trace)
            .collect::<Vec<_>>()
    );
    // What: the eager multiplicity-free trace selects its output
    // HomSpace once, in its stage; the compile reuses that admission (it
    // still compiles its own axis plan, a provider-free derivation).
    let stage =
        crate::tensortrace_stage_multiplicity_free(&dynamic_src, axes, dynamic_dst.space().nout())
            .unwrap();
    let (eager_dst, eager) = crate::tensortrace_multiplicity_free_in(
        strided_kernel::Exec::serial(),
        stage,
        &dynamic_src,
        |_| None,
        || &src_data[..],
        axes,
        alpha,
    )
    .unwrap();
    assert_eq!(
        (
            probe::take(Event::TraceAxisPlanDerivation),
            probe::take(Event::TraceSelectedHomspaceDerivation)
        ),
        (2, 1)
    );
    assert_eq!(eager_dst.space().homspace(), dynamic_dst.space().homspace());
    assert_eq!(eager, owned);
    // A stage of another source binding is compiled as against a committed
    // destination: its compile selects and compares again.
    let other = dynamic_src.clone();
    let stage =
        crate::tensortrace_stage_multiplicity_free(&other, axes, dynamic_dst.space().nout())
            .unwrap();
    let (_, fallback) = crate::tensortrace_multiplicity_free_in(
        strided_kernel::Exec::serial(),
        stage,
        &dynamic_src,
        |_| None,
        || &src_data[..],
        axes,
        alpha,
    )
    .unwrap();
    assert_eq!(
        (
            probe::take(Event::TraceAxisPlanDerivation),
            probe::take(Event::TraceSelectedHomspaceDerivation)
        ),
        (2, 2)
    );
    assert_eq!(fallback, owned);

    let mut unchanged = initial.clone();
    let error = tensortrace_fusion_dyn_into_checked(
        strided_kernel::Exec::serial(),
        &dynamic_dst,
        &mut unchanged,
        &dynamic_src,
        &src_data[..11],
        axes,
        alpha,
        beta,
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::ElementCountMismatch {
            expected: 12,
            actual: 11,
        }
    );
    assert_eq!(unchanged, initial);
    assert_eq!(
        tensortrace_fusion_dyn_owned_checked(
            strided_kernel::Exec::serial(),
            &dynamic_dst,
            &dynamic_src,
            &src_data[..11],
            axes,
            alpha,
        ),
        Err(OperationError::ElementCountMismatch {
            expected: 12,
            actual: 11,
        })
    );
    // What: typed, dynamic unchecked, dynamic checked, and their error path all
    // retain the parent block grid for asymmetric non-self-dual U1.
    assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);
}

#[test]
fn dynamic_u1_nonconjugating_trace_keeps_parent_axis_geometry() {
    let charge = U1Irrep::new(1).sector_id();
    let neutral = U1Irrep::new(0).sector_id();
    let charged_leg = || SectorLeg::new([(charge, 2)], false);
    let neutral_leg = || SectorLeg::new([(neutral, 3)], false);
    let provider = std::sync::Arc::new(U1FusionRule);
    let src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([charged_leg(), neutral_leg()]),
            FusionProductSpace::new([charged_leg()]),
        ),
    )
    .unwrap();
    let dst = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        provider,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([neutral_leg()]),
            FusionProductSpace::new([]),
        ),
    )
    .unwrap();
    let src_data = (1..=12)
        .map(|value| Complex64::new(value as f64, value as f64 + 0.25))
        .collect::<Vec<_>>();
    let expected = vec![
        src_data[0] + src_data[7],
        src_data[2] + src_data[9],
        src_data[4] + src_data[11],
    ];
    let mut actual = vec![Complex64::new(0.0, 0.0); 3];

    tensortrace_fusion_dyn_into(
        strided_kernel::Exec::serial(),
        &dst,
        &mut actual,
        &src,
        &src_data,
        TensorTraceAxisSpec::new(&[1], &[0], &[2]),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    // What: disabling conjugation retains the source HomSpace and values.
    assert_eq!(actual, expected);
    // What: checked algebra can also detect the neutral/charged trace mismatch,
    // but the incompatible selected destination remains the earlier error.
    assert_eq!(
        tensortrace_fusion_dyn_owned_checked(
            strided_kernel::Exec::serial(),
            &dst,
            &src,
            &src_data,
            TensorTraceAxisSpec::new(&[0], &[1], &[2]),
            Complex64::new(1.0, 0.0),
        )
        .unwrap_err(),
        OperationError::StructureMismatch { tensor: "dst" }
    );
}

#[test]
fn tensortrace_fusion_scales_destination_once_for_multiple_source_terms() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 1), (odd, 1)], false)]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![2.0, 3.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![10.0], dst_space).unwrap();

    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        5.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[49.0]);
}

// Expected data checked against TensorKit 0.17.1 `TO.tensortrace!`.
#[test]
fn tensortrace_fusion_fermion_open_output_matches_tensorkit_oracle() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        (1..=8).map(|value| value as f64).collect(),
        src_space,
    )
    .unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();

    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        // TensorKit 1-based: output ((1,), (3,)), trace ((2,), (4,)).
        TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[16.0, 62.0]);
}

// Expected data checked against TensorKit 0.17.1 `TO.tensortrace!`.
#[test]
fn tensortrace_fusion_fermion_two_trace_pairs_match_tensorkit_oracle() {
    let rule = FermionParityFusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        (1..=8).map(|value| value as f64).collect(),
        src_space,
    )
    .unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![5.0], dst_space).unwrap();

    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        // TensorKit 1-based: output ((), ()), trace ((1, 2), (3, 4)).
        TensorTraceAxisSpec::new(&[], &[0, 1], &[2, 3]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[-1.0]);
}

#[test]
fn tensortrace_fusion_rejects_nonsymmetric_braiding_like_tensorkit_tensortrace() {
    let rule = UniqueAnyonicRule;
    let src_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]),
    )
    .unwrap();
    let src: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![2.0], src_space).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::from_sector_ids([], []),
    )
    .unwrap();
    let dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    let err = dynamic_tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTensorContractScope {
            message: "fusion tensortrace requires symmetric braiding"
        }
    );
}

#[test]
fn tensortrace_fusion_su2_includes_quantum_dimension_factor() {
    let rule = SU2FusionRule;
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let src: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![7.0], src_space).unwrap();

    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let mut dst: TestTensor<f64> =
        TestTensor::from_vec_with_fusion_space(vec![0.0], dst_space).unwrap();

    dynamic_tensortrace_fusion_into(
        &rule,
        &mut dst,
        &src,
        TensorTraceAxisSpec::new(&[], &[0], &[1]),
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 14.0).abs() < 1.0e-12);
}

fn scalar_trace_term_oracle<R>(
    rule: &R,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    output_axes: &[usize],
    trace_lhs_axes: &[usize],
    trace_rhs_axes: &[usize],
    dst_codomain_rank: usize,
) -> Vec<(FusionTreePairKey, FusionTreePairKey, usize, usize, f64)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    use symtn_core::{multiplicity_free_permute_tree_pair, split_fusion_tree};

    let mut codomain_permutation = Vec::with_capacity(dst_codomain_rank + trace_lhs_axes.len());
    codomain_permutation.extend_from_slice(&output_axes[..dst_codomain_rank]);
    codomain_permutation.extend_from_slice(trace_lhs_axes);
    let mut domain_permutation =
        Vec::with_capacity(output_axes.len() - dst_codomain_rank + trace_rhs_axes.len());
    domain_permutation.extend_from_slice(&output_axes[dst_codomain_rank..]);
    domain_permutation.extend_from_slice(trace_rhs_axes);

    let mut terms = Vec::new();
    for src_block_index in 0..src_structure.block_count() {
        let src_key = expect_tree_key(src_structure.block(src_block_index).unwrap().key());
        for (permuted_key, permutation_coefficient) in multiplicity_free_permute_tree_pair(
            rule,
            &src_key,
            &codomain_permutation,
            &domain_permutation,
        )
        .unwrap()
        {
            let (dst_codomain_tree, trace_codomain_tree) =
                split_fusion_tree(rule, permuted_key.codomain_tree(), dst_codomain_rank).unwrap();
            let (dst_domain_tree, trace_domain_tree) = split_fusion_tree(
                rule,
                permuted_key.domain_tree(),
                output_axes.len() - dst_codomain_rank,
            )
            .unwrap();
            if trace_codomain_tree != trace_domain_tree {
                continue;
            }

            let coupled = trace_codomain_tree.coupled();
            let first = trace_codomain_tree.uncoupled()[0];
            let mut trace_factor = rule.dim_scalar(coupled) / rule.dim_scalar(first);
            for (&sector, &is_dual) in trace_codomain_tree
                .uncoupled()
                .iter()
                .zip(trace_codomain_tree.is_dual())
                .skip(1)
            {
                if !is_dual {
                    trace_factor *= rule.twist_scalar(sector);
                }
            }
            let dst_key = FusionTreePairKey::pair(dst_codomain_tree, dst_domain_tree);
            let dst_block = dst_structure
                .find_block_index_by_fusion_tree_pair(&dst_key)
                .unwrap();
            terms.push((
                dst_key,
                src_key.clone(),
                dst_block,
                src_block_index,
                permutation_coefficient * trace_factor,
            ));
        }
    }
    terms
}

fn assert_trace_terms_match_scalar_oracle<R>(
    rule: &R,
    structure: &TensorTraceFusionStructure<f64>,
    dst: &TestTensor<f64>,
    src_structure: &BlockStructure,
    output_axes: &[usize],
    trace_lhs_axes: &[usize],
    trace_rhs_axes: &[usize],
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let oracle = scalar_trace_term_oracle(
        rule,
        dst.structure(),
        src_structure,
        output_axes,
        trace_lhs_axes,
        trace_rhs_axes,
        1,
    );
    assert_eq!(structure.terms().len(), oracle.len());
    for (actual, (dst_key, src_key, dst_block, src_block, coefficient)) in
        structure.terms().iter().zip(oracle)
    {
        // What: block lowering preserves scalar-path term identity and global
        // source/destination order, and the retained public destination key
        // agrees with its destination block; only floating reduction may round differently.
        assert_eq!(actual.dst_key(), &dst_key);
        assert_eq!(actual.src_key(), &src_key);
        assert_eq!(actual.dst_block(), dst_block);
        assert_eq!(actual.src_block(), src_block);
        let BlockKey::FusionTree(block_key) = dst.structure().block(dst_block).unwrap().key()
        else {
            panic!("fusion trace destination must use a fusion-tree block key");
        };
        assert_eq!(actual.dst_key(), block_key);
        assert!((actual.coefficient() - coefficient).abs() <= 1.0e-12 * (1.0 + coefficient.abs()));
    }
}

#[test]
fn tensortrace_fusion_recouples_once_per_simple_fusion_group() {
    let rule = SU2FusionRule;
    let spin_one = SU2Irrep::from_twice_spin(2).sector_id();
    let leg = || SectorLeg::new([(spin_one, 1)], false);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom.clone()).unwrap();
    let src: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![0.0; src_space.required_len().unwrap()],
        src_space,
    )
    .unwrap();

    let dst_hom = src_hom.select(&rule, &[1], &[3]).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let dst: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();
    let group_count = src.structure().fusion_tree_groups().len();
    assert!(
        src.structure().block_count() > group_count,
        "fixture must contain multiple source trees in one fusion group"
    );

    probe::reset_trace_sources();
    let structure = dynamic_tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]),
    )
    .unwrap();

    // What: non-Abelian partial trace walks the recoupling transform once per
    // external-sector group, not once per source fusion tree.
    assert_eq!(probe::take_trace_sources().len(), group_count);
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &structure,
        &dst,
        src.structure(),
        &[1, 3],
        &[0],
        &[2],
    );
    assert!(structure
        .terms()
        .windows(2)
        .all(|pair| pair[0].src_block() <= pair[1].src_block()));
    let provider = std::sync::Arc::new(SU2FusionRule);
    let dynamic_src = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        src.fusion_space().unwrap().as_ref().clone(),
        std::sync::Arc::clone(&provider),
    )
    .unwrap();
    let dynamic_dst = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst.fusion_space().unwrap().as_ref().clone(),
        provider,
    )
    .unwrap();
    assert_owned_trace_matches_initialized_for_real_and_complex(
        &dynamic_dst,
        &dynamic_src,
        TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]),
    );

    let conjugate_axes = TensorTraceAxisSpec::new_with_conjugation(&[1, 3], &[0], &[2], true);
    let adjoint_src = src.dynamic_space().adjoint_view().unwrap();
    let lowered_axes =
        crate::lowering::lower_tensortrace_source_adjoint_axes_dyn(2, 2, conjugate_axes).unwrap();
    let lowered_axes = lowered_axes.as_spec();
    let conjugate_dst_hom = adjoint_src
        .homspace()
        .select(
            &rule,
            &lowered_axes.output_axes()[..1],
            &lowered_axes.output_axes()[1..],
        )
        .unwrap();
    let conjugate_dst_space =
        DynamicFusionMapSpace::from_final_homspace(&rule, conjugate_dst_hom).unwrap();
    let conjugate_dst: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![0.0; conjugate_dst_space.required_len().unwrap()],
        conjugate_dst_space,
    )
    .unwrap();
    let adjoint_view_build_count_rec = probe::record();
    let conjugate_structure =
        dynamic_tensortrace_fusion_structure(&rule, &conjugate_dst, &src, conjugate_axes).unwrap();
    // What: source-conjugate lowering still batches the logical adjoint tree
    // groups while retaining the scalar adjoint-axis trace semantics.
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &conjugate_structure,
        &conjugate_dst,
        adjoint_src.structure(),
        lowered_axes.output_axes(),
        lowered_axes.trace_lhs_axes(),
        lowered_axes.trace_rhs_axes(),
    );
    assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);
}

#[test]
fn tensortrace_fusion_product_block_matches_scalar_trace_terms() {
    let left_rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let rule = FpU1Su2Rule::try_new(
        FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let odd = SectorId::new(1);
    let product_sector = |charge| {
        rule.encode_component_ids(
            left_rule.encode_component_ids(odd, U1Irrep::new(charge).sector_id()),
            SU2Irrep::from_twice_spin(1).sector_id(),
        )
    };
    let a = product_sector(1);
    let b = product_sector(-1);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(b, 1)], false),
            SectorLeg::new([(a, 1)], false),
        ]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom.clone()).unwrap();
    let src: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![0.0; src_space.required_len().unwrap()],
        src_space,
    )
    .unwrap();
    let dst_hom = src_hom.select(&rule, &[1], &[2]).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let dst: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();

    let structure = dynamic_tensortrace_fusion_structure(
        &rule,
        &dst,
        &src,
        TensorTraceAxisSpec::new(&[1, 2], &[0], &[3]),
    )
    .unwrap();

    assert_trace_terms_match_scalar_oracle(
        &rule,
        &structure,
        &dst,
        src.structure(),
        &[1, 2],
        &[0],
        &[3],
    );

    let conjugate_axes = TensorTraceAxisSpec::new_with_conjugation(&[1, 2], &[0], &[3], true);
    let adjoint_src = src.dynamic_space().adjoint_view().unwrap();
    let lowered_axes =
        crate::lowering::lower_tensortrace_source_adjoint_axes_dyn(2, 2, conjugate_axes).unwrap();
    let lowered_axes = lowered_axes.as_spec();
    let conjugate_dst_hom = adjoint_src
        .homspace()
        .select(
            &rule,
            &lowered_axes.output_axes()[..1],
            &lowered_axes.output_axes()[1..],
        )
        .unwrap();
    let conjugate_dst_space =
        DynamicFusionMapSpace::from_final_homspace(&rule, conjugate_dst_hom).unwrap();
    let conjugate_dst: TestTensor<f64> = TestTensor::from_vec_with_fusion_space(
        vec![0.0; conjugate_dst_space.required_len().unwrap()],
        conjugate_dst_space,
    )
    .unwrap();
    let adjoint_view_build_count_rec = probe::record();
    let conjugate_structure =
        dynamic_tensortrace_fusion_structure(&rule, &conjugate_dst, &src, conjugate_axes).unwrap();

    // What: the Simple product path preserves source-conjugate adjoint lowering,
    // asymmetric U(1), half-integer SU(2), and the fZ2 odd supertrace sign.
    assert_trace_terms_match_scalar_oracle(
        &rule,
        &conjugate_structure,
        &conjugate_dst,
        adjoint_src.structure(),
        lowered_axes.output_axes(),
        lowered_axes.trace_lhs_axes(),
        lowered_axes.trace_rhs_axes(),
    );
    assert!(
        conjugate_structure
            .terms()
            .iter()
            .any(|term| *term.coefficient() < 0.0),
        "odd-sector conjugate trace fixture must retain a negative structural coefficient"
    );
    let provider = std::sync::Arc::new(
        FpU1Su2Rule::try_new(
            FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let dynamic_src = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        src.fusion_space().unwrap().as_ref().clone(),
        std::sync::Arc::clone(&provider),
    )
    .unwrap();
    let dynamic_dst = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        conjugate_dst.fusion_space().unwrap().as_ref().clone(),
        provider,
    )
    .unwrap();
    assert_owned_trace_matches_initialized_for_real_and_complex(
        &dynamic_dst,
        &dynamic_src,
        conjugate_axes,
    );
    assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);
}

#[test]
fn excluded_u1_sector_id_is_rejected_at_admission() {
    // What: the raw zigzag ID excluded by the label constructor never becomes
    // a space, so no trace can observe it.
    let min = SectorId::new(u32::MAX as usize);
    let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        std::sync::Arc::new(U1FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(min, 1)], false)]),
            FusionProductSpace::new([]),
        ),
    )
    .unwrap_err();
    assert_eq!(
        error,
        OperationError::FusionAlgebra(Box::new(symtn_core::FusionAlgebraError::InvalidSector {
            sector: min
        }))
    );
}

#[test]
fn structure_executors_reject_terms_compiled_for_other_spaces() {
    // What: the public trace executors replay only terms compiled for the
    // spaces they are given.
    let provider = std::sync::Arc::new(U1FusionRule);
    let source = |degeneracy| {
        let leg = || SectorLeg::new([(U1Irrep::new(1).sector_id(), degeneracy)], false);
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            std::sync::Arc::clone(&provider),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    let dst = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        std::sync::Arc::clone(&provider),
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
    )
    .unwrap();
    let (compiled_for, other) = (source(2), source(3));
    let axes = TensorTraceAxisSpec::new(&[], &[0], &[1]);
    let structure =
        TensorTraceFusionStructure::compile_fusion_dyn_checked(&dst, &compiled_for, axes).unwrap();
    let data = vec![1.0; 9];

    assert_eq!(
        crate::tensortrace_fusion_dyn_structure_owned(
            strided_kernel::Exec::serial(),
            &structure,
            dst.space(),
            other.space(),
            &data,
            1.0
        ),
        Err(OperationError::StructureMismatch { tensor: "src" })
    );
    let mut out = vec![0.0];
    assert_eq!(
        crate::tensortrace_fusion_dyn_structure_into_raw(
            strided_kernel::Exec::serial(),
            &structure,
            dst.space(),
            &mut out,
            other.space(),
            &data,
            1.0,
            0.0
        ),
        Err(OperationError::StructureMismatch { tensor: "src" })
    );
    assert!(crate::tensortrace_fusion_dyn_structure_owned(
        strided_kernel::Exec::serial(),
        &structure,
        dst.space(),
        compiled_for.space(),
        &data[..4],
        1.0
    )
    .is_ok());
}

thread_local! {
    static TRACE_TEST_WRONG_STYLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TRACE_TEST_FAIL_TWIST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static TRACE_TEST_TWIST_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(feature = "cgc-gen")]
mod checked_whole_basis;
mod staged_publication;
mod trace_cache4;
