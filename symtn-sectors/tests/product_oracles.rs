//! Product symbols against independent TensorKitSectors coefficients and hand values.

use num_complex::{Complex32, Complex64};
use symtn_sectors::*;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn s(id: usize) -> SectorId {
    SectorId::new(id)
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
    use fusion_symbols::category::FusionCategory;

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
    struct Symbols(u32);

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
            true
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
    fn asymmetric_a_and_b_use_the_right_factor_digit_fastest() {
        let rule = ProductFusionRule::<_, _>::try_new(cg(Symbols(2)), cg(Symbols(3))).unwrap();
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
            ProductFusionRule::<_, _>::try_new(cg(Symbols(u32::MAX)), cg(Symbols(2))).unwrap();
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
            Symbols(2).fusion(a, b)
        }
        fn nsymbol(&self, a: &u8, b: &u8, c: &u8) -> u32 {
            Symbols(2).nsymbol(a, b, c)
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
            Symbols(2).object_at(index)
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
