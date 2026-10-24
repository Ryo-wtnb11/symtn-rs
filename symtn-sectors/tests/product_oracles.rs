//! Product symbols against independent TensorKitSectors coefficients and hand values.

use num_complex::{Complex32, Complex64};
use symtn_sectors::*;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn s(id: usize) -> SectorId {
    SectorId::new(id)
}

#[test]
fn category_rule_factors_preserve_channels_duals_and_factor_errors() {
    use fusion_symbols::finite::ZN;
    use std::error::Error;
    type MF = MultiplicityFreeAdmissionMode;
    type CG = CheckedGenericAdmissionMode;

    let z3 = CategoryRule::try_new(ZN::new(3).unwrap(), EnumeratedCodec::<MF>::new()).unwrap();
    let mf = ProductFusionRule::<_, _>::try_new(z3, U1FusionRule).unwrap();
    let x = mf.encode_component_ids(s(1), U1Irrep::new(2).sector_id());
    let y = mf.encode_component_ids(s(2), U1Irrep::new(4).sector_id());
    let dual = mf.encode_component_ids(s(2), U1Irrep::new(-2).sector_id());
    // (1,2) + (1,2) = (2,4) in Z3 x U1; -(1,2) = (2,-2).
    assert_eq!(mf.try_fusion_channels(x, x).unwrap().as_slice(), &[y]);
    assert_eq!(mf.try_nsymbol(x, x, y).unwrap(), 1);
    assert_eq!(mf.try_dual_sector(x).unwrap(), dual);
    assert_eq!(mf.r_symbol_scalar(x, x, y), 1.0);
    assert_eq!(mf.twist_scalar(x), 1.0);
    assert_eq!(FusionRule::sector_order_key(&mf, x).positions(), &[1, 7]);
    // Negative controls: the wrong output and a foreign Z3 id stay distinct.
    assert_eq!(mf.try_nsymbol(x, x, x).unwrap(), 0);
    let bad = mf.encode_component_ids(s(3), s(0));
    assert_eq!(
        mf.try_dual_sector(bad),
        Err(FusionAlgebraError::InvalidSector { sector: s(3) })
    );

    let z3 = CategoryRule::try_new(ZN::new(3).unwrap(), EnumeratedCodec::<CG>::new()).unwrap();
    let left_cg = ProductFusionRule::<_, _>::try_new(z3.clone(), U1FusionRule).unwrap();
    assert!(matches!(
        left_cg.try_dual(left_cg.encode_component_ids(s(3), s(0))),
        Err(ProductFusionRuleError::Left(FusionAlgebraError::InvalidSector { sector }))
            if sector == s(3)
    ));
    let cg = ProductFusionRule::<_, _>::try_new(U1FusionRule, z3).unwrap();
    assert_eq!(cg.fusion_style(), FusionStyleKind::Generic);
    let x = cg.encode_component_ids(U1Irrep::new(2).sector_id(), s(1));
    let y = cg.encode_component_ids(U1Irrep::new(4).sector_id(), s(2));
    let dual = cg.encode_component_ids(U1Irrep::new(-2).sector_id(), s(2));
    assert_eq!(cg.try_fusion_channels(x, x).unwrap().as_slice(), &[y]);
    assert_eq!(
        cg.try_fusion_channels_in_table(x, x).unwrap().as_slice(),
        &[y]
    );
    assert_eq!(cg.try_nsymbol(x, x, y).unwrap(), 1);
    assert_eq!(cg.try_dual(x).unwrap(), dual);
    assert_eq!(cg.try_r_symbol_generic(x, x, y).unwrap().data(), &[1.0]);
    assert_eq!(cg.try_twist_scalar(x).unwrap(), 1.0);
    assert_eq!(
        CheckedGenericFusion::sector_order_key(&cg, x).positions(),
        &[7, 1]
    );
    assert_eq!(cg.try_nsymbol(x, x, x).unwrap(), 0);
    let bad = cg.encode_component_ids(s(0), s(3));
    assert!(matches!(
        cg.try_dual(bad),
        Err(ProductFusionRuleError::Right(FusionAlgebraError::InvalidSector { sector }))
            if sector == s(3)
    ));
    assert!(matches!(
        cg.try_r_symbol_generic(x, x, bad),
        Err(ProductFusionRuleError::Right(FusionAlgebraError::InvalidSector { sector }))
            if sector == s(3)
    ));
    assert!(matches!(
        cg.try_twist_scalar(bad),
        Err(ProductFusionRuleError::Right(FusionAlgebraError::InvalidSector { sector }))
            if sector == s(3)
    ));
    let mapped =
        CategoryProvider::map_error(cg.right_rule(), &[s(1)], CategoryAdapterError::NoBraiding);
    let FusionAlgebraError::CategoryFailure(failure) = mapped else {
        panic!("{mapped:?}")
    };
    assert_eq!(
        failure
            .source()
            .unwrap()
            .downcast_ref::<CategoryAdapterError<std::convert::Infallible>>(),
        Some(&CategoryAdapterError::NoBraiding)
    );
}

#[test]
fn product_errors_display_context_and_retain_each_source() {
    use std::error::Error;
    use std::io;
    type E = ProductFusionRuleError<io::Error, io::Error>;

    for (error, text, source) in [
        (
            E::Left(io::Error::other("left probe")),
            "the left factor rejected its component",
            Some("left probe"),
        ),
        (
            E::Right(io::Error::other("right probe")),
            "the right factor rejected its component",
            Some("right probe"),
        ),
        (
            E::Codec(ProductSectorCodecError::CodecRejected),
            "the product sector id cannot be packed",
            Some("product sector codec rejected the value"),
        ),
        (
            E::DimensionNotRepresentable { sector: s(7) },
            "the dimension of SectorId(7) is not a finite positive f64",
            None,
        ),
        (
            E::Category(CategoryFailure::new(io::Error::other("category probe"))),
            "the product category failed",
            Some("category probe"),
        ),
    ] {
        assert_eq!(error.to_string(), text);
        assert_eq!(error.source().map(ToString::to_string).as_deref(), source);
        match &error {
            E::Left(_) | E::Right(_) => assert!(error.source().unwrap().is::<io::Error>()),
            E::Codec(_) => assert!(error.source().unwrap().is::<ProductSectorCodecError>()),
            E::Category(_) => assert!(error.source().unwrap().is::<CategoryFailure>()),
            // Negative control: a dimension failure has context but no underlying error.
            E::DimensionNotRepresentable { .. } => assert!(error.source().is_none()),
            _ => panic!("unexpected product error: {error:?}"),
        }
    }
}

#[test]
fn product_admission_maps_adapter_errors_without_erasing_sources() {
    use std::error::Error;
    type MF = MultiplicityFreeAdmissionMode;
    type CG = CheckedGenericAdmissionMode;
    type A = CategoryAdapterError<FusionAlgebraError>;
    type F = FusionAlgebraError;

    for adapter in [
        A::DimensionNotRepresentable { sector: s(7) },
        A::Provider(F::InvalidSector { sector: s(9) }),
        A::NoBraiding,
        A::ForeignObject {
            sector: Some(s(11)),
        },
    ] {
        let mf = <(MF, MF) as ProductAdmission<F, F>>::adapter(adapter.clone());
        if let A::ForeignObject {
            sector: Some(sector),
        } = adapter
        {
            // Negative control: foreign ids map to InvalidSector, not CategoryFailure.
            assert_eq!(mf, F::InvalidSector { sector });
        } else {
            let F::CategoryFailure(failure) = mf else {
                panic!("{mf:?}")
            };
            match &adapter {
                A::Provider(expected) => assert_eq!(
                    failure.source().unwrap().downcast_ref::<F>(),
                    Some(expected)
                ),
                other => assert_eq!(failure.source().unwrap().downcast_ref::<A>(), Some(other)),
            }
        }

        for map in [
            <(MF, CG) as ProductAdmission<F, F>>::adapter::<F>,
            <(CG, MF) as ProductAdmission<F, F>>::adapter::<F>,
            <(CG, CG) as ProductAdmission<F, F>>::adapter::<F>,
        ] {
            let error = map(adapter.clone());
            match &adapter {
                A::DimensionNotRepresentable { sector: expected } => {
                    assert!(
                        matches!(error, ProductFusionRuleError::DimensionNotRepresentable { sector } if sector == *expected)
                    );
                    assert!(error.source().is_none());
                }
                other => {
                    let ProductFusionRuleError::Category(failure) = error else {
                        panic!("{error:?}")
                    };
                    match other {
                        A::Provider(expected) => assert_eq!(
                            failure.source().unwrap().downcast_ref::<F>(),
                            Some(expected)
                        ),
                        expected => assert_eq!(
                            failure.source().unwrap().downcast_ref::<A>(),
                            Some(expected)
                        ),
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "MultiplicityFreeRigidSymbols::inv_dim_scalar")]
fn mf_product_inverse_dimension_rejects_a_foreign_factor_id() {
    // docs/provider_interface.md: infallible queries may panic outside their domain.
    let rule = ProductFusionRule::<_, _>::try_new(SU2FusionRule, Z2FusionRule).unwrap();
    assert_eq!(
        rule.inv_dim_scalar(rule.encode_component_ids(s(1), s(1))),
        0.5
    );
    // Negative control: the packing is valid, but Z2 has no sector 2.
    rule.inv_dim_scalar(rule.encode_component_ids(s(1), s(2)));
}

#[test]
fn su2_fermion_product_has_the_independent_spin_and_parity_signs() {
    let rule = ProductFusionRule::<_, _>::try_new(SU2FusionRule, FermionParityFusionRule).unwrap();
    let half_odd = rule.encode_component_ids(s(1), s(1));
    let singlet = rule.encode_component_ids(s(0), s(0));
    let triplet = rule.encode_component_ids(s(2), s(0));
    // Condon--Shortley: R(1/2,1/2;J) = (-1)^(1-J), multiplied by -1 for two odd lines.
    assert_eq!(rule.r_symbol_scalar(half_odd, half_odd, singlet), 1.0);
    assert_eq!(rule.r_symbol_scalar(half_odd, half_odd, triplet), -1.0);
    numerics::assert_close(
        "spin-half F times the trivial parity F",
        rule.f_symbol_scalar(half_odd, half_odd, half_odd, half_odd, singlet, singlet),
        -0.5,
        4,
    );
    assert_eq!(rule.frobenius_schur_phase_scalar(half_odd), -1.0);
    assert_eq!(rule.twist_scalar(half_odd), -1.0);
    assert_eq!(rule.dim_scalar(half_odd), 2.0);
    // Negative control: omitting the parity braid reverses both R signs.
    assert_eq!(SU2FusionRule.r_symbol_scalar(s(1), s(1), s(0)), -1.0);
    assert_eq!(SU2FusionRule.r_symbol_scalar(s(1), s(1), s(2)), 1.0);
}

#[test]
fn fermions_in_different_factors_have_no_cross_braid() {
    let rule = ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, FermionParityFusionRule)
        .unwrap();
    let left_odd = rule.encode_component_ids(s(1), s(0));
    let right_odd = rule.encode_component_ids(s(0), s(1));
    let both_odd = rule.encode_component_ids(s(1), s(1));
    assert_eq!(rule.r_symbol_scalar(left_odd, right_odd, both_odd), 1.0);
    assert_eq!(rule.r_symbol_scalar(right_odd, left_odd, both_odd), 1.0);
    // Negative control: two odd lines in the same factor do braid with -1.
    assert_eq!(
        rule.r_symbol_scalar(left_odd, left_odd, rule.vacuum()),
        -1.0
    );
}

#[test]
#[should_panic(expected = "MultiplicityFreeFusionSymbols::r_symbol_scalar")]
fn zero_left_r_still_rejects_an_invalid_right_mf_sector() {
    let rule = ProductFusionRule::<_, _>::try_new(SU2FusionRule, Z2FusionRule).unwrap();
    let vacuum = rule.encode_component_ids(s(0), s(0));
    let absent = rule.encode_component_ids(s(2), s(0));
    // Negative control: a valid right factor preserves the absent SU(2) channel.
    assert_eq!(rule.r_symbol_scalar(vacuum, vacuum, absent), 0.0);
    let invalid = rule.encode_component_ids(s(0), s(2));
    rule.r_symbol_scalar(invalid, vacuum, absent);
}

#[cfg(feature = "cgc-gen")]
mod generated {
    use super::*;
    use fusion_symbols::category::{
        BraidingStyle, EnumerationError, FusionCategory, FusionStyle, SymbolBlock,
    };
    use fusion_symbols::lie::{GroupId, LieError, LieGroup, LieIrrep};

    // SUNRepresentations 0.4.0 / TensorKitSectors 0.3.9 literals retained in
    // fusion-symbols 0.8.0 tests/ab_symbols.rs::multiplicity_blocks_match_sunrepresentations.
    const B_15: [f64; 4] = [
        -0.353_553_390_593_273_73,
        0.935_414_346_693_485_7,
        -0.935_414_346_693_486_3,
        -0.353_553_390_593_273_9,
    ];

    #[test]
    fn category_authority_a_b_match_octet_and_asymmetric_product_literals() {
        let su3 = SUNFusionRule::new(3).unwrap();
        let [octet, fifteen, fifteen_bar, twenty_seven] =
            [[1, 1], [2, 1], [1, 2], [2, 2]].map(|labels| su3.encode_dynkin(&labels).unwrap());
        for block in [
            su3.try_a_symbol_generic(octet, octet, octet).unwrap(),
            su3.try_b_symbol_generic(octet, octet, octet).unwrap(),
        ] {
            numerics::assert_slices_close(
                "SU3 octet A/B",
                block.data(),
                &[1.0, 0.0, 0.0, 1.0],
                CGC_TERMS,
            );
        }
        let rule = ProductFusionRule::<_, _>::try_new(su3.clone(), su3).unwrap();
        // TensorKitSectors product.jl::_kron: right multiplicity digit fastest.
        let left = rule
            .try_b_symbol_generic(
                rule.encode_component_ids(fifteen, octet),
                rule.encode_component_ids(fifteen_bar, octet),
                rule.encode_component_ids(twenty_seven, octet),
            )
            .unwrap();
        let right = rule
            .try_b_symbol_generic(
                rule.encode_component_ids(octet, fifteen),
                rule.encode_component_ids(octet, fifteen_bar),
                rule.encode_component_ids(octet, twenty_seven),
            )
            .unwrap();
        let [r, s, t, u] = B_15;
        #[rustfmt::skip]
        let kron_left = [r,0.0,s,0.0, 0.0,r,0.0,s, t,0.0,u,0.0, 0.0,t,0.0,u];
        #[rustfmt::skip]
        let kron_right = [r,s,0.0,0.0, t,u,0.0,0.0, 0.0,0.0,r,s, 0.0,0.0,t,u];
        numerics::assert_slices_close("B15 x I2", left.data(), &kron_left, CGC_TERMS);
        numerics::assert_slices_close("I2 x B15", right.data(), &kron_right, CGC_TERMS);
        // Transposing B reverses its off-diagonal sign; reversing product
        // digits moves the nonzero (0,2) entry to (0,1).
        assert!((*left.get(0, 2) - *left.get(2, 0)).abs() > 1.0);
        assert!((*left.get(0, 2) - *right.get(0, 2)).abs() > 0.9);
    }

    struct RotatedF {
        axes: [usize; 4],
        data: Vec<Complex64>,
    }
    impl SymbolBlock<4> for RotatedF {
        type Scalar = Complex64;
        fn dims(&self) -> [usize; 4] {
            self.axes
        }
        fn get(&self, index: [usize; 4]) -> Complex64 {
            self.data[index
                .into_iter()
                .zip(self.axes)
                .fold(0, |p, (i, n)| p * n + i)]
        }
        fn as_slice(&self) -> &[Complex64] {
            &self.data
        }
    }

    /// One SU3 vertex basis change C'_mu = sum_rho C_rho U[rho,mu].
    /// Real U=[[0,1],[-1,0]] gives A'=U^T A=J; replacing -1 by -i
    /// exercises conjugation. TensorKitSectors::Fsymbol_from_fusiontensor
    /// is linear in left-tree C and antilinear in right-tree C. The real
    /// gauge and J literal also appear in fs 0.8.0 tests/ab_symbols.rs::Rotated.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    struct Rotated {
        group: LieGroup,
        complex: bool,
    }
    impl FusionCategory for Rotated {
        type Object = LieIrrep;
        type Scalar = Complex64;
        type FSymbol = RotatedF;
        type RSymbol = Complex64;
        type Fusion<'a> = <LieGroup as FusionCategory>::Fusion<'a>;
        type Error = LieError;
        fn fusion_style(&self) -> FusionStyle {
            self.group.fusion_style()
        }
        fn braiding_style(&self) -> BraidingStyle {
            BraidingStyle::None
        }
        fn unit(&self) -> LieIrrep {
            self.group.unit()
        }
        fn contains(&self, a: &LieIrrep) -> bool {
            self.group.contains(a)
        }
        fn dual(&self, a: &LieIrrep) -> LieIrrep {
            self.group.dual(a)
        }
        fn dim(&self, a: &LieIrrep) -> f64 {
            self.group.dim(a)
        }
        fn frobenius_schur_phase(&self, a: &LieIrrep) -> Complex64 {
            self.group.frobenius_schur_phase(a).into()
        }
        fn frobenius_schur_indicator(&self, a: &LieIrrep) -> Complex64 {
            self.group.frobenius_schur_indicator(a).into()
        }
        fn twist(&self, a: &LieIrrep) -> Complex64 {
            self.group.twist(a).into()
        }
        fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
            self.group.fusion(a, b)
        }
        fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
            self.group.nsymbol(a, b, c)
        }
        fn has_canonical_unit(&self) -> bool {
            self.group.has_canonical_unit()
        }
        fn f_symbol(
            &self,
            a: &LieIrrep,
            b: &LieIrrep,
            c: &LieIrrep,
            d: &LieIrrep,
            e: &LieIrrep,
            f: &LieIrrep,
        ) -> Result<RotatedF, LieError> {
            let vertices = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)];
            let axes = vertices.map(|(x, y, z)| self.nsymbol(x, y, z) as usize);
            let base = self.group.f_symbol(a, b, c, d, e, f)?;
            let mut data = if axes.contains(&0) {
                vec![]
            } else {
                base.data()
                    .iter()
                    .copied()
                    .map(Complex64::from)
                    .collect::<Vec<_>>()
            };
            for (axis, (x, y, z)) in vertices.into_iter().enumerate() {
                if data.is_empty()
                    || !x.dynkin_labels().eq([2, 1])
                    || !y.dynkin_labels().eq([1, 2])
                    || !z.dynkin_labels().eq([2, 2])
                {
                    continue;
                }
                let inner: usize = axes[axis + 1..].iter().product();
                let old = data.clone();
                for (flat, value) in data.iter_mut().enumerate() {
                    let i = flat / inner % 2;
                    let other = flat + inner * (1 - i) - inner * i;
                    let phase = if i == 1 {
                        Complex64::new(1.0, 0.0)
                    } else if self.complex {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::new(-1.0, 0.0)
                    };
                    *value = old[other] * if axis < 2 { phase } else { phase.conj() };
                }
            }
            Ok(RotatedF { axes, data })
        }
        fn r_symbol(
            &self,
            _: &LieIrrep,
            _: &LieIrrep,
            _: &LieIrrep,
        ) -> Result<Complex64, LieError> {
            Err(LieError::Unsupported {
                requires: "braiding in the rotated gauge",
            })
        }
        fn authority_fingerprint(&self) -> &[u8] {
            if self.complex {
                b"test:su3-complex-vertex-gauge"
            } else {
                b"test:su3-real-vertex-gauge"
            }
        }
        fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
            self.group.object_count()
        }
        fn object_at(&self, index: u64) -> Result<LieIrrep, EnumerationError> {
            self.group.object_at(index)
        }
        fn object_index(&self, a: &LieIrrep) -> Result<u64, EnumerationError> {
            self.group.object_index(a)
        }
    }

    #[test]
    fn rotated_gauge_a_detects_transposition_and_complex_conjugation() {
        let group = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let objects = [[2, 1], [1, 2], [2, 2]].map(|labels| group.irrep(&labels).unwrap());
        for complex in [false, true] {
            let rule = CategoryRule::try_new(
                Rotated {
                    group: group.clone(),
                    complex,
                },
                EnumeratedCodec::<CheckedGenericAdmissionMode>::new(),
            )
            .unwrap();
            let [a, b, c] = objects
                .each_ref()
                .map(|x| rule.try_encode_label(x).unwrap());
            let block = rule.try_a_symbol_generic(a, b, c).unwrap();
            let off_diagonal = if complex {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(-1.0, 0.0)
            };
            let expected = [
                Complex64::new(0.0, 0.0),
                off_diagonal,
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ];
            numerics::assert_slices_close("rotated SU3 A", block.data(), &expected, CGC_TERMS);
            assert!((*block.get(0, 1) - *block.get(1, 0)).norm() > 1.0);
            if complex {
                assert!((*block.get(0, 1) - block.get(0, 1).conj()).norm() > 1.0);
            }
        }
    }

    #[test]
    fn checked_generic_product_surface_preserves_hand_derived_factor_data() {
        let su3 = SUNFusionRule::new(3).unwrap();
        let three = su3.encode_dynkin(&[1, 0]).unwrap();
        let anti_three = su3.encode_dynkin(&[0, 1]).unwrap();
        let six = su3.encode_dynkin(&[2, 0]).unwrap();
        let rule = ProductFusionRule::<_, _>::try_new(SU2FusionRule, su3.clone()).unwrap();
        let swapped = ProductFusionRule::<_, _>::try_new(su3, SU2FusionRule).unwrap();
        assert_eq!(rule.rule_identity(), rule.clone().rule_identity());
        assert_ne!(rule.rule_identity(), swapped.rule_identity());
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Bosonic);
        assert_eq!(rule.vacuum(), s(0));
        let debug = format!("{rule:?}");
        assert!(debug.starts_with("ProductFusionRule { left: SU2FusionRule, right: SUNFusionRule"));
        assert!(debug.contains(&format!("identity: {:?}", rule.rule_identity())));
        assert_ne!(debug, format!("{swapped:?}"));

        let x = rule.encode_component_ids(s(1), three);
        let dual = rule.encode_component_ids(s(1), anti_three);
        assert_eq!(rule.try_dual(x).unwrap(), dual);
        assert_ne!(dual, x);
        // (1/2 x 1/2) = 0 + 1; 3 x 3 = bar(3) + 6 (Dynkin order).
        // fusion-symbols Product enumerates the first factor fastest.
        let channels = [
            rule.encode_component_ids(s(0), anti_three),
            rule.encode_component_ids(s(2), anti_three),
            rule.encode_component_ids(s(0), six),
            rule.encode_component_ids(s(2), six),
        ];
        assert_eq!(
            rule.try_fusion_channels(x, x).unwrap().as_slice(),
            &channels
        );
        assert_eq!(
            rule.try_fusion_channels_in_table(x, x).unwrap().as_slice(),
            &channels
        );
        for output in channels {
            assert_eq!(rule.try_nsymbol(x, x, output).unwrap(), 1);
        }
        assert_eq!(rule.try_dim_scalar(x).unwrap(), 2.0 * 3.0);
        numerics::assert_close(
            "sqrt product dimension",
            rule.try_sqrt_dim_scalar(x).unwrap(),
            6.0_f64.sqrt(),
            2,
        );
        numerics::assert_close(
            "inverse sqrt product dimension",
            rule.try_inv_sqrt_dim_scalar(x).unwrap(),
            1.0 / 6.0_f64.sqrt(),
            2,
        );
        assert_eq!(rule.try_frobenius_schur_phase_scalar(x).unwrap(), -1.0);
        assert_eq!(rule.try_twist_scalar(x).unwrap(), 1.0);
        // Negative control: neither the input nor the unit is in x tensor x.
        assert_eq!(rule.try_nsymbol(x, x, x).unwrap(), 0);
        assert_eq!(rule.try_nsymbol(x, x, rule.vacuum()).unwrap(), 0);
        let invalid = rule.encode_component_ids(s(255), three);
        for error in [
            rule.try_dual(invalid).unwrap_err(),
            rule.try_fusion_channels(invalid, x).unwrap_err(),
            rule.try_fusion_channels_in_table(invalid, x).unwrap_err(),
            rule.try_nsymbol(invalid, x, channels[0]).unwrap_err(),
            rule.try_dim_scalar(invalid).unwrap_err(),
            rule.try_sqrt_dim_scalar(invalid).unwrap_err(),
            rule.try_inv_sqrt_dim_scalar(invalid).unwrap_err(),
            rule.try_frobenius_schur_phase_scalar(invalid).unwrap_err(),
            rule.try_twist_scalar(invalid).unwrap_err(),
        ] {
            assert!(matches!(error,
                ProductFusionRuleError::Left(FusionAlgebraError::InvalidSector { sector })
                    if sector == s(255)
            ));
        }

        let nested = ProductFusionRule::<_, _>::try_new(rule, FermionParityFusionRule).unwrap();
        assert_ne!(nested.rule_identity(), nested.left_rule().rule_identity());
        let odd = nested.encode_component_ids(x, s(1));
        let odd_dual = nested.encode_component_ids(dual, s(1));
        assert_eq!(nested.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(nested.braiding_style(), BraidingStyleKind::Fermionic);
        assert_eq!(nested.vacuum(), s(0));
        assert_eq!(nested.try_dual(odd).unwrap(), odd_dual);
        let even_channels = channels.map(|id| nested.encode_component_ids(id, s(0)));
        assert_eq!(
            nested.try_fusion_channels(odd, odd).unwrap().as_slice(),
            &even_channels
        );
        assert_eq!(
            nested
                .try_fusion_channels_in_table(odd, odd)
                .unwrap()
                .as_slice(),
            &even_channels
        );
        assert_eq!(nested.try_nsymbol(odd, odd, even_channels[0]).unwrap(), 1);
        assert_eq!(nested.try_dim_scalar(odd).unwrap(), 6.0);
        assert_eq!(nested.try_twist_scalar(odd).unwrap(), -1.0);
        // Negative control: odd x odd has even parity, and removing parity removes its twist.
        assert_eq!(
            nested
                .try_nsymbol(odd, odd, nested.encode_component_ids(channels[0], s(1)))
                .unwrap(),
            0
        );
        assert_eq!(
            nested
                .try_twist_scalar(nested.encode_component_ids(x, s(0)))
                .unwrap(),
            1.0
        );
    }

    // TensorKitSectors 0.3.9 / SUNRepresentations 0.4.0:
    // o = SUNIrrep{3}(1,1); Fsymbol(o,o,o,o,o,o), Rsymbol(o,o,o).
    // Exported once as [F[i,j,k,l] for i ... for j ... for k ... for l ...]
    // and [R[i,j] for i ... for j ...], explicitly in Rust row-major order.
    const OCTET_F: [f64; 16] = [
        0.857_142_857_142_856,
        1.491_862_189_340_054e-16,
        -2.255_140_518_769_849_2e-17,
        -0.142_857_142_857_142_63,
        8.326_672_684_688_674e-17,
        -0.142_857_142_857_142_5,
        -0.142_857_142_857_142_7,
        -0.383_325_938_999_963_37,
        -2.255_140_518_769_849_2e-17,
        -0.142_857_142_857_142_85,
        -0.142_857_142_857_142_46,
        -0.383_325_938_999_963_15,
        -0.142_857_142_857_142_74,
        -0.383_325_938_999_963_53,
        -0.383_325_938_999_963_26,
        0.628_571_428_571_427_4,
    ];
    const OCTET_R: [f64; 4] = [
        -0.285_714_285_714_285_3,
        0.958_314_847_499_908_8,
        0.958_314_847_499_908_9,
        0.285_714_285_714_285_53,
    ];
    // The largest exported representation here is (2,3), dimension 42.
    // Allow the roundoff of two independent dense physical-CGC contractions.
    const CGC_TERMS: usize = 2 * 42 * 42 * 42;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn zero_left_r_still_rejects_an_invalid_right_generic_sector() {
        let rule =
            ProductFusionRule::<_, _>::try_new(SU2FusionRule, SUNFusionRule::new(2).unwrap())
                .unwrap();
        let vacuum = rule.encode_component_ids(s(0), s(0));
        let absent = rule.encode_component_ids(s(2), s(0));
        // Negative control: a valid right factor preserves the absent SU(2) channel.
        let zero = rule.try_r_symbol_generic(vacuum, vacuum, absent).unwrap();
        assert_eq!(zero.shape(), (0, 0));
        assert!(zero.data().is_empty());
        let invalid = rule.encode_component_ids(s(0), s(u32::MAX as usize + 1));
        assert!(matches!(
            rule.try_nsymbol(invalid, vacuum, absent),
            Err(ProductFusionRuleError::Right(_))
        ));
        assert!(matches!(
            rule.try_r_symbol_generic(invalid, vacuum, absent),
            Err(ProductFusionRuleError::Right(_))
        ));
    }

    #[test]
    fn su3_u1_keeps_the_full_octet_blocks_and_nonselfdual_phase() {
        let su3 = SUNFusionRule::new(3).unwrap();
        let octet = su3.encode_dynkin(&[1, 1]).unwrap();
        let three = su3.encode_dynkin(&[1, 0]).unwrap();
        let rule = ProductFusionRule::<_, _>::try_new(su3, U1FusionRule).unwrap();
        let oq = |q| rule.encode_component_ids(octet, U1Irrep::new(q).sector_id());
        let f = rule
            .try_f_symbol_generic(oq(1), oq(1), oq(1), oq(3), oq(2), oq(2))
            .unwrap();
        assert_eq!(f.shape(), (2, 2, 2, 2));
        numerics::assert_slices_close("SU3 x U1 F", f.data(), &OCTET_F, CGC_TERMS);
        let r = rule.try_r_symbol_generic(oq(1), oq(1), oq(2)).unwrap();
        assert_eq!(r.shape(), (2, 2));
        numerics::assert_slices_close("SU3 x U1 R", r.data(), &OCTET_R, CGC_TERMS);
        // TensorKitSectors 0.3.9: Asymbol(o,o,o) and Bsymbol(o,o,o),
        // o = SUNIrrep{3}(1,1), both equal the 2 x 2 identity to roundoff.
        for block in [
            rule.try_a_symbol_generic(oq(1), oq(1), oq(2)).unwrap(),
            rule.try_b_symbol_generic(oq(1), oq(1), oq(2)).unwrap(),
        ] {
            assert_eq!(block.shape(), (2, 2));
            numerics::assert_slices_close(
                "SU3 x U1 A/B",
                block.data(),
                &[1.0, 0.0, 0.0, 1.0],
                CGC_TERMS,
            );
        }
        let fundamental = rule.encode_component_ids(three, U1Irrep::new(1).sector_id());
        assert_eq!(
            rule.try_frobenius_schur_phase_scalar(fundamental).unwrap(),
            1.0
        );
        // TensorKitSectors: frobenius_schur_phase(SUNIrrep{3}(1,0)) = 1,
        // frobenius_schur_indicator(...) = 0; the charge also makes it non-self-dual.
        let object = CategoryProvider::object(&rule, fundamental).unwrap();
        assert_eq!(rule.category().frobenius_schur_indicator(&object), 0.0);
        assert_ne!(rule.try_dual(fundamental).unwrap(), fundamental);

        // Negative controls: a wrong charge removes the vertex, rather than
        // leaving a copied octet block or a scalar zero of the wrong shape.
        let absent_r = rule.try_r_symbol_generic(oq(1), oq(1), oq(3)).unwrap();
        assert_eq!(absent_r.shape(), (0, 0));
        assert!(absent_r.data().is_empty());
        let absent_f = rule
            .try_f_symbol_generic(oq(1), oq(1), oq(1), oq(3), oq(3), oq(2))
            .unwrap();
        assert_eq!(absent_f.shape(), (0, 0, 2, 2));
        assert!(absent_f.data().is_empty());
        for absent in [
            rule.try_a_symbol_generic(oq(1), oq(1), oq(3)).unwrap(),
            rule.try_b_symbol_generic(oq(1), oq(1), oq(3)).unwrap(),
        ] {
            assert_eq!(absent.shape(), (0, 0));
            assert!(absent.data().is_empty());
        }
    }

    #[test]
    fn su3_fermion_product_negates_every_octet_r_entry() {
        let su3 = SUNFusionRule::new(3).unwrap();
        let octet = su3.encode_dynkin(&[1, 1]).unwrap();
        let rule = ProductFusionRule::<_, _>::try_new(su3, FermionParityFusionRule).unwrap();
        let odd = rule.encode_component_ids(octet, s(1));
        let even = rule.encode_component_ids(octet, s(0));
        let f = rule
            .try_f_symbol_generic(odd, odd, odd, odd, even, even)
            .unwrap();
        assert_eq!(f.shape(), (2, 2, 2, 2));
        numerics::assert_slices_close("SU3 x fZ2 F", f.data(), &OCTET_F, CGC_TERMS);
        let r = rule.try_r_symbol_generic(odd, odd, even).unwrap();
        assert_eq!(r.shape(), (2, 2));
        numerics::assert_slices_close("SU3 x fZ2 odd R", r.data(), &OCTET_R.map(|x| -x), CGC_TERMS);
        // Negative control: removing one odd parity removes the minus sign.
        let ungraded = rule.try_r_symbol_generic(odd, even, odd).unwrap();
        numerics::assert_slices_close("SU3 x fZ2 mixed R", ungraded.data(), &OCTET_R, CGC_TERMS);
        assert!((r.data()[1] - ungraded.data()[1]).abs() > 1.0);
    }

    #[test]
    fn su3_su3_octet_checks_every_entry_of_the_fourfold_multiplicity_blocks() {
        let su3 = SUNFusionRule::new(3).unwrap();
        let octet = su3.encode_dynkin(&[1, 1]).unwrap();
        let rule = ProductFusionRule::<_, _>::try_new(su3.clone(), su3).unwrap();
        let x = rule.encode_component_ids(octet, octet);
        assert_eq!(rule.try_nsymbol(x, x, x).unwrap(), 4);
        let f = rule.try_f_symbol_generic(x, x, x, x, x, x).unwrap();
        let r = rule.try_r_symbol_generic(x, x, x).unwrap();
        assert_eq!(f.shape(), (4, 4, 4, 4));
        assert_eq!(r.shape(), (4, 4));
        // TensorKitSectors product.jl: _kron on each logical multiplicity axis,
        // with the right factor's digit fastest. This oracle uses only literals.
        for mu in 0..4 {
            for nu in 0..4 {
                for kappa in 0..4 {
                    for lambda in 0..4 {
                        let left = ((mu / 2 * 2 + nu / 2) * 2 + kappa / 2) * 2 + lambda / 2;
                        let right = ((mu % 2 * 2 + nu % 2) * 2 + kappa % 2) * 2 + lambda % 2;
                        numerics::assert_close(
                            "SU3 x SU3 F",
                            *f.get(mu, nu, kappa, lambda),
                            OCTET_F[left] * OCTET_F[right],
                            CGC_TERMS,
                        );
                    }
                }
                numerics::assert_close(
                    "SU3 x SU3 R",
                    *r.get(mu, nu),
                    OCTET_R[mu / 2 * 2 + nu / 2] * OCTET_R[mu % 2 * 2 + nu % 2],
                    CGC_TERMS,
                );
                let gram: f64 = (0..4).map(|k| *r.get(mu, k) * *r.get(nu, k)).sum();
                numerics::assert_close("product R unitarity", gram, f64::from(mu == nu), CGC_TERMS);
            }
        }
        // Negative control: flattening each factor before taking its Kronecker
        // product puts OCTET_F[0]*OCTET_F[3] here, rather than the logical-axis entry.
        assert!((*f.get(0, 0, 0, 3) - OCTET_F[0] * OCTET_F[3]).abs() > 0.1);
    }

    #[test]
    fn asymmetric_su3_product_pins_unequal_f_and_r_digit_orders() {
        let su3 = SUNFusionRule::new(3).unwrap();
        let octet = su3.encode_dynkin(&[1, 1]).unwrap();
        let twenty_seven = su3.encode_dynkin(&[2, 2]).unwrap();
        let anti_three = su3.encode_dynkin(&[0, 1]).unwrap();
        let twenty_four = su3.encode_dynkin(&[3, 1]).unwrap();
        let forty_two = su3.encode_dynkin(&[2, 3]).unwrap();
        let rule = ProductFusionRule::<_, _>::try_new(su3.clone(), su3).unwrap();
        let pair = |right| rule.encode_component_ids(octet, right);
        // TensorKitSectors Fsymbol(p,p,b,b,o,c), with p=(2,2), b=(0,1),
        // o=(1,1), c=(3,1); exported as F[i,j,k,l] in Rust row-major order.
        let asymmetric_f = [0.259_499_648_053_841_24, 0.797_724_035_217_465_4];
        let f = rule
            .try_f_symbol_generic(
                pair(twenty_seven),
                pair(twenty_seven),
                pair(anti_three),
                pair(anti_three),
                pair(octet),
                pair(twenty_four),
            )
            .unwrap();
        assert_eq!(f.shape(), (4, 2, 2, 2));
        for mu in 0..4 {
            for nu in 0..2 {
                for kappa in 0..2 {
                    for lambda in 0..2 {
                        let left = ((mu / 2 * 2 + nu) * 2 + kappa) * 2 + lambda;
                        numerics::assert_close(
                            "asymmetric product F",
                            *f.get(mu, nu, kappa, lambda),
                            OCTET_F[left] * asymmetric_f[mu % 2],
                            CGC_TERMS,
                        );
                    }
                }
            }
        }
        // TensorKitSectors Rsymbol(p,t,t), p=(2,2), t=(2,3); R[i,j] row-major.
        let asymmetric_r = [
            0.136_914_793_673_601_93,
            -0.439_764_978_446_179_9,
            0.887_615_402_640_999_7,
            -0.374_954_863_181_612_3,
            0.806_373_884_496_817_5,
            0.457_351_078_470_322_36,
            0.916_876_867_332_671_5,
            0.395_433_840_400_314_8,
            0.054_487_503_305_696_45,
        ];
        let r = rule
            .try_r_symbol_generic(pair(twenty_seven), pair(forty_two), pair(forty_two))
            .unwrap();
        assert_eq!(r.shape(), (6, 6));
        for row in 0..6 {
            for col in 0..6 {
                numerics::assert_close(
                    "asymmetric product R",
                    *r.get(row, col),
                    OCTET_R[row / 3 * 2 + col / 3] * asymmetric_r[row % 3 * 3 + col % 3],
                    CGC_TERMS,
                );
            }
        }
        // Negative controls: reversing factor digits cannot be hidden by identical factors.
        assert!((*f.get(1, 0, 0, 0) - OCTET_F[8] * asymmetric_f[0]).abs() > 0.1);
        assert!((*r.get(0, 2) - OCTET_R[0] * asymmetric_r[1]).abs() > 0.1);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checked_generic_products_complete_the_left_admission_before_the_right() {
        let su2 = SUNFusionRule::new(2).unwrap();
        let invalid_sun = s(u32::MAX as usize + 1);
        let max_charge = U1Irrep::new(i32::MAX).sector_id();
        let one_charge = U1Irrep::new(1).sector_id();
        let left_mf = ProductFusionRule::<_, _>::try_new(U1FusionRule, su2.clone()).unwrap();
        let a = left_mf.encode_component_ids(max_charge, s(0));
        let b = left_mf.encode_component_ids(one_charge, invalid_sun);
        assert!(matches!(
            left_mf.try_nsymbol(a, b, s(0)),
            Err(ProductFusionRuleError::Left(
                FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1
                }
            ))
        ));
        // Negative control: repairing the left overflow exposes the right failure.
        let a = left_mf.encode_component_ids(s(0), s(0));
        assert!(
            matches!(left_mf.try_nsymbol(a, b, s(0)), Err(ProductFusionRuleError::Right(SUNFusionRuleError::UnrepresentableLabel { labels })) if labels == [u32::MAX as i64 + 1])
        );

        let left_cg = ProductFusionRule::<_, _>::try_new(su2, U1FusionRule).unwrap();
        let a = left_cg.encode_component_ids(s(0), max_charge);
        let b = left_cg.encode_component_ids(invalid_sun, one_charge);
        assert!(
            matches!(left_cg.try_nsymbol(a, b, s(0)), Err(ProductFusionRuleError::Left(SUNFusionRuleError::UnrepresentableLabel { labels })) if labels == [u32::MAX as i64 + 1])
        );
        let b = left_cg.encode_component_ids(s(0), one_charge);
        assert!(matches!(
            left_cg.try_nsymbol(a, b, s(0)),
            Err(ProductFusionRuleError::Right(
                FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1
                }
            ))
        ));
    }
}

mod storage {
    use super::*;
    use fusion_symbols::category::{
        BraidingStyle, EnumerationError, FusionCategory, FusionStyle, RBlock, SymbolBlock,
    };
    use fusion_symbols::product::ProductError;
    use std::convert::Infallible;
    use std::error::Error;

    #[derive(Clone, Debug)]
    struct Block<const RANK: usize> {
        dims: [usize; RANK],
        data: Vec<f64>,
    }

    impl<const RANK: usize> SymbolBlock<RANK> for Block<RANK> {
        type Scalar = f64;
        fn dims(&self) -> [usize; RANK] {
            self.dims
        }
        fn as_slice(&self) -> &[f64] {
            &self.data
        }
        fn get(&self, index: [usize; RANK]) -> f64 {
            self.data[index
                .into_iter()
                .zip(self.dims)
                .fold(0, |flat, (i, n)| flat * n + i)]
        }
    }

    // A storage probe, not a physical category oracle: nonidentity, unequal
    // multiplicity matrices expose permutations that the SU3 identity A/B hide.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    struct Symbols(u32, bool);

    impl FusionCategory for Symbols {
        type Object = u8;
        type Scalar = f64;
        type FSymbol = Block<4>;
        type RSymbol = f64;
        type Fusion<'a> = std::vec::IntoIter<u8>;
        type Error = Infallible;

        fn fusion_style(&self) -> FusionStyle {
            FusionStyle::Generic
        }
        fn braiding_style(&self) -> BraidingStyle {
            BraidingStyle::Bosonic
        }
        fn unit(&self) -> u8 {
            0
        }
        fn contains(&self, a: &u8) -> bool {
            *a < 2
        }
        fn dual(&self, a: &u8) -> u8 {
            *a
        }
        fn dim(&self, _: &u8) -> f64 {
            1.0
        }
        fn frobenius_schur_phase(&self, _: &u8) -> f64 {
            1.0
        }
        fn frobenius_schur_indicator(&self, _: &u8) -> f64 {
            1.0
        }
        fn twist(&self, _: &u8) -> f64 {
            1.0
        }
        fn has_canonical_unit(&self) -> bool {
            self.1
        }
        fn fusion<'a>(&'a self, a: &u8, b: &u8) -> Result<Self::Fusion<'a>, Infallible> {
            Ok(if *a == 0 {
                vec![*b]
            } else if *b == 0 {
                vec![*a]
            } else {
                vec![0, 1]
            }
            .into_iter())
        }
        fn nsymbol(&self, a: &u8, b: &u8, c: &u8) -> u32 {
            if *a == 0 {
                u32::from(b == c)
            } else if *b == 0 {
                u32::from(a == c)
            } else if *c == 0 {
                1
            } else {
                self.0
            }
        }
        fn f_symbol(
            &self,
            a: &u8,
            b: &u8,
            c: &u8,
            d: &u8,
            e: &u8,
            f: &u8,
        ) -> Result<Block<4>, Infallible> {
            let dims = [
                self.nsymbol(a, b, e),
                self.nsymbol(e, c, d),
                self.nsymbol(b, c, f),
                self.nsymbol(a, f, d),
            ]
            .map(|n| n as usize);
            let data = match (a, b, c, d, e, f, self.0) {
                (1, 1, 1, 1, 0, 1, 2) => vec![0.0, -1.0, 1.0, 0.0],
                (1, 1, 1, 1, 0, 1, 3) => vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                (1, 1, 1, 1, 1, 0, 2) => vec![1.0, 0.0, 0.0, -1.0],
                (1, 1, 1, 1, 1, 0, 3) => vec![0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                _ => (0..dims.iter().product())
                    .map(|flat| {
                        let mut rest = flat;
                        let mut active = Vec::new();
                        for n in dims.into_iter().rev() {
                            if n > 1 {
                                active.push(rest % n);
                            }
                            rest /= n;
                        }
                        f64::from(active.windows(2).all(|pair| pair[0] == pair[1]))
                    })
                    .collect(),
            };
            Ok(Block { dims, data })
        }
        fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Infallible> {
            Ok(1.0)
        }
        fn authority_fingerprint(&self) -> &[u8] {
            b"test:product-symbol-storage"
        }
        fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
            Ok(Some(2))
        }
        fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
            if index < 2 {
                Ok(index as u8)
            } else {
                Err(EnumerationError::OutOfRange)
            }
        }
        fn object_index(&self, a: &u8) -> Result<u64, EnumerationError> {
            Ok(u64::from(*a))
        }
    }

    fn cg<C>(category: C) -> CategoryRule<C, EnumeratedCodec<CheckedGenericAdmissionMode>>
    where
        C: FusionCategory + Send + Sync + 'static,
        C::Error: Send + Sync + 'static,
    {
        CategoryRule::try_new(category, EnumeratedCodec::new()).unwrap()
    }

    #[test]
    fn product_requires_the_category_unit_certificate_even_with_uncertified_admission() {
        for canonical in [false, true] {
            let factor = CategoryRule::try_new(
                Symbols(2, canonical),
                EnumeratedCodec::<CheckedGenericAdmissionMode, UncertifiedUnitAdmission>::new(),
            )
            .unwrap();
            let left = ProductFusionRule::<_, _>::try_new(factor.clone(), U1FusionRule);
            let right = ProductFusionRule::<_, _>::try_new(U1FusionRule, factor);
            if canonical {
                // Negative control: the admission selector alone does not reject a product.
                assert!(left.is_ok());
                assert!(right.is_ok());
            } else {
                assert!(matches!(
                    left,
                    Err(CategoryAdmissionError::MissingCanonicalUnit)
                ));
                assert!(matches!(
                    right,
                    Err(CategoryAdmissionError::MissingCanonicalUnit)
                ));
            }
        }
    }

    #[test]
    fn asymmetric_a_and_b_use_the_right_factor_digit_fastest() {
        let rule =
            ProductFusionRule::<_, _>::try_new(cg(Symbols(2, true)), cg(Symbols(3, true))).unwrap();
        let x = rule.encode_component_ids(s(1), s(1));
        // Explicit factor matrices from the storage probe's F blocks above;
        // exact Kronecker formula M[(i,j),(k,l)] = L[i,k] R[j,l].
        let factors = [
            (
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            ),
            (
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
        ];
        let blocks = [
            rule.try_a_symbol_generic(x, x, x).unwrap(),
            rule.try_b_symbol_generic(x, x, x).unwrap(),
        ];
        for (block, (left, right)) in blocks.iter().zip(factors) {
            assert_eq!(block.shape(), (6, 6));
            for row in 0..6 {
                for col in 0..6 {
                    assert_eq!(
                        *block.get(row, col),
                        left[row / 3 * 2 + col / 3] * right[row % 3 * 3 + col % 3]
                    );
                }
            }
        }
        // Negative controls: first-factor-fastest packing gives zero at both entries.
        assert_eq!(*blocks[0].get(0, 4), -1.0);
        assert_eq!(*blocks[1].get(0, 2), -1.0);
        assert_eq!(factors[0].0[0] * factors[0].1[2], 0.0);
        assert_eq!(factors[1].0[0] * factors[1].1[1], 0.0);
    }

    #[test]
    fn product_n_keeps_fs_overflow_and_a_later_zero_annuls_it() {
        let inner =
            ProductFusionRule::<_, _>::try_new(cg(Symbols(u32::MAX, true)), cg(Symbols(2, true)))
                .unwrap();
        let x = inner.encode_component_ids(s(1), s(1));
        let error = inner.try_nsymbol(x, x, x).unwrap_err();
        let ProductFusionRuleError::Category(failure) = error else {
            panic!("{error:?}")
        };
        assert!(matches!(
            failure
                .source()
                .unwrap()
                .downcast_ref::<ProductError<Infallible, Infallible>>()
                .unwrap(),
            ProductError::Overflow
        ));

        let nested = ProductFusionRule::<_, _>::try_new(inner, Z2FusionRule).unwrap();
        let even = nested.encode_component_ids(x, s(0));
        let odd = nested.encode_component_ids(x, s(1));
        // N_Z2(0,0;1)=0 fixes the product at zero despite the inner overflow.
        assert_eq!(nested.try_nsymbol(even, even, odd).unwrap(), 0);
        // Negative control: restoring the Z2 channel exposes the same overflow,
        // now retaining the nested left-factor position through the public error.
        let error = nested.try_nsymbol(even, even, even).unwrap_err();
        let ProductFusionRuleError::Left(ProductFusionRuleError::Category(failure)) = error else {
            panic!("{error:?}")
        };
        assert!(matches!(
            failure
                .source()
                .unwrap()
                .downcast_ref::<ProductError<Infallible, Infallible>>()
                .unwrap(),
            ProductError::Overflow
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn checked_generic_outer_decode_precedes_factor_admission() {
        let rule = ProductFusionRule::<_, _, PackedProductCodec<U1SectorLayout, Fz2SectorLayout>>::try_new(
            U1FusionRule,
            cg(fusion_symbols::finite::ZN::new(2).unwrap()),
        ).unwrap();
        let invalid_u1 = s(u32::MAX as usize);
        let invalid_outer = s(1 << 33);
        assert!(
            matches!(rule.try_nsymbol(invalid_u1, s(0), invalid_outer), Err(ProductFusionRuleError::Codec(ProductSectorCodecError::InvalidHighBits { sector, total_bits: 33 })) if sector == invalid_outer)
        );
        // Negative control: fixing outer packing exposes the left factor error.
        assert!(
            matches!(rule.try_nsymbol(invalid_u1, s(0), s(0)), Err(ProductFusionRuleError::Left(FusionAlgebraError::InvalidSector { sector })) if sector == invalid_u1)
        );
    }

    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    struct Malformed;

    impl FusionCategory for Malformed {
        type Object = u8;
        type Scalar = f64;
        type FSymbol = Block<4>;
        type RSymbol = f64;
        type Fusion<'a> = std::vec::IntoIter<u8>;
        type Error = Infallible;

        fn fusion_style(&self) -> FusionStyle {
            FusionStyle::Generic
        }
        fn braiding_style(&self) -> BraidingStyle {
            BraidingStyle::Bosonic
        }
        fn unit(&self) -> u8 {
            0
        }
        fn contains(&self, a: &u8) -> bool {
            *a < 2
        }
        fn dual(&self, a: &u8) -> u8 {
            *a
        }
        fn dim(&self, _: &u8) -> f64 {
            1.0
        }
        fn frobenius_schur_phase(&self, _: &u8) -> f64 {
            1.0
        }
        fn frobenius_schur_indicator(&self, _: &u8) -> f64 {
            1.0
        }
        fn twist(&self, _: &u8) -> f64 {
            1.0
        }
        fn has_canonical_unit(&self) -> bool {
            true
        }
        fn fusion<'a>(&'a self, a: &u8, b: &u8) -> Result<Self::Fusion<'a>, Infallible> {
            Symbols(2, true).fusion(a, b)
        }
        fn nsymbol(&self, a: &u8, b: &u8, c: &u8) -> u32 {
            Symbols(2, true).nsymbol(a, b, c)
        }
        fn f_symbol(
            &self,
            a: &u8,
            b: &u8,
            c: &u8,
            d: &u8,
            e: &u8,
            f: &u8,
        ) -> Result<Block<4>, Infallible> {
            if [a, b, c, d, e, f].into_iter().all(|id| *id == 0) {
                Ok(Block {
                    dims: [1; 4],
                    data: vec![1.0],
                })
            } else {
                // The element count is right; only the logical axes are wrong.
                Ok(Block {
                    dims: [4, 1, 2, 2],
                    data: vec![1.0; 16],
                })
            }
        }
        fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Infallible> {
            Ok(1.0)
        }
        fn a_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Infallible> {
            Ok(RBlock::one(1.0))
        }
        fn b_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Infallible> {
            Ok(RBlock::one(1.0))
        }
        fn authority_fingerprint(&self) -> &[u8] {
            b"test:malformed-product-symbols"
        }
        fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
            Ok(Some(2))
        }
        fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
            Symbols(2, true).object_at(index)
        }
        fn object_index(&self, a: &u8) -> Result<u64, EnumerationError> {
            Ok(u64::from(*a))
        }
    }

    #[test]
    fn malformed_product_f_r_a_b_keep_the_fs_shape_error_and_factor_position() {
        let rule = ProductFusionRule::<_, _>::try_new(cg(Malformed), Z2FusionRule).unwrap();
        let x = rule.encode_component_ids(s(1), s(0));
        let errors = [
            (
                rule.try_f_symbol_generic(x, x, x, x, x, x).unwrap_err(),
                vec![2; 4],
                vec![4, 1, 2, 2],
                16,
            ),
            (
                rule.try_r_symbol_generic(x, x, x).unwrap_err(),
                vec![2; 2],
                vec![1; 2],
                1,
            ),
            (
                rule.try_a_symbol_generic(x, x, x).unwrap_err(),
                vec![2; 2],
                vec![1; 2],
                1,
            ),
            (
                rule.try_b_symbol_generic(x, x, x).unwrap_err(),
                vec![2; 2],
                vec![1; 2],
                1,
            ),
        ];
        for (error, expected_shape, actual_shape, len) in errors {
            let ProductFusionRuleError::Category(failure) = &error else {
                panic!("{error:?}")
            };
            let source = failure
                .source()
                .unwrap()
                .downcast_ref::<ProductError<Infallible, Infallible>>()
                .unwrap();
            assert!(
                matches!(source, ProductError::MalformedSymbolData { factor: 0, expected, actual, storage_len } if expected == &expected_shape && actual == &actual_shape && *storage_len == len)
            );
            assert!(error
                .source()
                .unwrap()
                .downcast_ref::<CategoryFailure>()
                .is_some());
        }
        // Negative control: the same storage is correct at the unit's one-dimensional vertex.
        let v = rule.vacuum();
        assert_eq!(
            rule.try_f_symbol_generic(v, v, v, v, v, v).unwrap().data(),
            &[1.0]
        );
        for block in [
            rule.try_r_symbol_generic(v, v, v).unwrap(),
            rule.try_a_symbol_generic(v, v, v).unwrap(),
            rule.try_b_symbol_generic(v, v, v).unwrap(),
        ] {
            assert_eq!(block.shape(), (1, 1));
            assert_eq!(block.data(), &[1.0]);
        }
        let reversed = ProductFusionRule::<_, _>::try_new(Z2FusionRule, cg(Malformed)).unwrap();
        let x = reversed.encode_component_ids(s(0), s(1));
        let error = reversed.try_a_symbol_generic(x, x, x).unwrap_err();
        let ProductFusionRuleError::Category(failure) = error else {
            panic!("{error:?}")
        };
        assert!(matches!(
            failure
                .source()
                .unwrap()
                .downcast_ref::<ProductError<Infallible, Infallible>>()
                .unwrap(),
            ProductError::MalformedSymbolData { factor: 1, .. }
        ));
    }
}
