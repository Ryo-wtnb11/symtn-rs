//! `CategoryRule`: a `fusion_symbols` category under a codec is a symtn
//! provider, with symtn's identity, admission, and error contracts.
//!
//! Expected values are hand-derived from each category's definition; the
//! merged-admission tests compare against the literals the built-in providers
//! are frozen to in `tests/provider_contracts/errors.rs`.

use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use fusion_symbols::category::{
    BraidingStyle, EnumerationError, FusionCategory, FusionStyle, RBlock, SymbolBlock,
};
use fusion_symbols::finite::{ZNIrrep, ZN};
use fusion_symbols::lie::{GroupId, LieError, LieGroup, LieIrrep, U1Irrep as FsU1Irrep, U1};
use fusion_symbols::table::TableCategory;
use num_complex::Complex64;
use symtn_sectors::*;

use crate::numerics;

type MF = MultiplicityFreeAdmissionMode;
type CG = CheckedGenericAdmissionMode;

fn s(id: usize) -> SectorId {
    SectorId::new(id)
}

fn invalid<T>(id: usize) -> Result<T, FusionAlgebraError> {
    Err(FusionAlgebraError::InvalidSector { sector: s(id) })
}

fn mf<C>(category: C) -> CategoryRule<C, EnumeratedCodec<MF>>
where
    C: FusionCategory + Send + Sync + 'static,
    C::Error: Send + Sync + 'static,
{
    CategoryRule::try_new(category, EnumeratedCodec::new()).unwrap()
}

fn cg<C>(category: C) -> CategoryRule<C, EnumeratedCodec<CG>>
where
    C: FusionCategory + Send + Sync + 'static,
    C::Error: Send + Sync + 'static,
{
    CategoryRule::try_new(category, EnumeratedCodec::new()).unwrap()
}

/// The adapter error a `CategoryFailure` retains.
fn adapter_source<E: std::error::Error + 'static>(
    error: &FusionAlgebraError,
) -> &CategoryAdapterError<E> {
    let FusionAlgebraError::CategoryFailure(failure) = error else {
        panic!("expected a category failure, got {error:?}");
    };
    std::error::Error::source(failure)
        .and_then(|source| source.downcast_ref::<CategoryAdapterError<E>>())
        .unwrap_or_else(|| panic!("unexpected failure source {failure:?}"))
}

// ---------------------------------------------------------------------------
// A downstream category: Z4 anyons with R^{ab} = i^{ab}, implementing only
// `FusionCategory`. Trivial F, so the bicharacter R satisfies the hexagons and
// the twist is θ_a = R^{aa} = i^{a²}.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Z4Anyons;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Z4(u8);

fn i_power(power: u32) -> Complex64 {
    [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ][(power % 4) as usize]
}

impl FusionCategory for Z4Anyons {
    type Object = Z4;
    type Scalar = Complex64;
    type FSymbol = Complex64;
    type RSymbol = Complex64;
    type Fusion<'a> = std::iter::Once<Z4>;
    type Error = std::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Anyonic
    }
    fn unit(&self) -> Z4 {
        Z4(0)
    }
    fn contains(&self, a: &Z4) -> bool {
        a.0 < 4
    }
    fn dual(&self, a: &Z4) -> Z4 {
        Z4((4 - a.0) % 4)
    }
    fn dim(&self, _: &Z4) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _: &Z4) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn frobenius_schur_indicator(&self, a: &Z4) -> Complex64 {
        Complex64::new(f64::from(u8::from(a.0.is_multiple_of(2))), 0.0)
    }
    fn twist(&self, a: &Z4) -> Complex64 {
        i_power(u32::from(a.0) * u32::from(a.0))
    }
    fn fusion<'a>(&'a self, a: &Z4, b: &Z4) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(Z4((a.0 + b.0) % 4)))
    }
    fn nsymbol(&self, a: &Z4, b: &Z4, c: &Z4) -> u32 {
        u32::from((a.0 + b.0) % 4 == c.0)
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(
        &self,
        a: &Z4,
        b: &Z4,
        c: &Z4,
        d: &Z4,
        e: &Z4,
        f: &Z4,
    ) -> Result<Complex64, Self::Error> {
        let n = self.nsymbol(a, b, e)
            * self.nsymbol(e, c, d)
            * self.nsymbol(b, c, f)
            * self.nsymbol(a, f, d);
        Ok(Complex64::new(f64::from(n), 0.0))
    }
    fn r_symbol(&self, a: &Z4, b: &Z4, c: &Z4) -> Result<Complex64, Self::Error> {
        Ok(i_power(u32::from(a.0) * u32::from(b.0)) * f64::from(self.nsymbol(a, b, c)))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:z4-anyons"
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(4))
    }
    fn object_at(&self, index: u64) -> Result<Z4, EnumerationError> {
        u8::try_from(index)
            .ok()
            .filter(|&charge| charge < 4)
            .map(Z4)
            .ok_or(EnumerationError::OutOfRange)
    }
    fn object_index(&self, a: &Z4) -> Result<u64, EnumerationError> {
        assert!(self.contains(a), "foreign object");
        Ok(u64::from(a.0))
    }
}

#[test]
fn a_downstream_category_is_a_provider_without_any_symtn_impl() {
    let rule = mf(Z4Anyons);
    let i = Complex64::new(0.0, 1.0);

    assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
    assert_eq!(rule.braiding_style(), BraidingStyleKind::Anyonic);
    assert_eq!(rule.vacuum(), s(0));
    assert_eq!(rule.try_dual_sector(s(1)), Ok(s(3)));
    assert_eq!(
        rule.try_fusion_channels(s(3), s(2)).unwrap().as_slice(),
        &[s(1)]
    );
    assert_eq!(rule.try_nsymbol(s(1), s(2), s(3)), Ok(1));
    assert_eq!(rule.try_nsymbol(s(1), s(2), s(0)), Ok(0));
    assert_eq!(
        rule.r_symbol_scalar(s(1), s(2), s(3)),
        Complex64::new(-1.0, 0.0)
    );
    assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(2)), i);
    assert_eq!(rule.twist_scalar(s(3)), i);
    assert_eq!(
        rule.f_symbol_scalar(s(1), s(1), s(1), s(3), s(2), s(2)),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(rule.dim_scalar(s(2)), Complex64::new(1.0, 0.0));
    assert_eq!(
        rule.a_symbol_scalar(s(1), s(2), s(3)),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(
        rule.b_symbol_scalar(s(1), s(2), s(3)),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(rule.encode_sector(&Z4(2)), Ok(s(2)));
    assert_eq!(rule.decode_sector(s(3)), Ok(Z4(3)));
    assert_eq!(
        FusionRule::sector_order_key(&rule, s(2)),
        SectorOrderKey::position(2)
    );
    assert_eq!(rule.try_dual_sector(s(7)), invalid(7));
    assert_eq!(rule.decode_sector(s(4)), invalid(4));
    assert_eq!(
        rule.encode_sector(&Z4(9)),
        Err(FusionAlgebraError::UnrepresentableSectorLabel {
            rule: FusionRule::rule_identity(&rule),
            label: "Z4(9)".to_owned(),
        })
    );
    assert_eq!(rule.try_dual_id(s(2)), Ok(s(2)));
}

// ---------------------------------------------------------------------------
// Admission mode, unit selector, and runtime admission.
// ---------------------------------------------------------------------------

fn multiplicity_free<R>(_: &R)
where
    R: MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule
        + TypedSectorAdmission<Error = FusionAlgebraError, Mode = MF>,
{
}

fn checked_generic<R>(_: &R)
where
    R: CheckedGenericPivotal
        + CheckedCanonicalUnitFusionRule
        + TypedSectorAdmission<Error = <R as CheckedGenericFusion>::Error, Mode = CG>,
{
}

fn uncertified_multiplicity_free<R>(_: &R)
where
    R: MultiplicityFreeRigidSymbols + TypedSectorAdmission<Mode = MF>,
{
}

#[test]
fn both_modes_and_both_scalars_admit_their_traits() {
    multiplicity_free(&mf(ZN::new(3).unwrap()));
    multiplicity_free(&mf(Z4Anyons));
    checked_generic(&cg(ZN::new(3).unwrap()));
    checked_generic(&cg(Z4Anyons));
    let uncertified = CategoryRule::try_new(
        ZN::new(3).unwrap(),
        EnumeratedCodec::<MF, UncertifiedUnitAdmission>::new(),
    )
    .unwrap();
    uncertified_multiplicity_free(&uncertified);

    // Checked Generic admission reports Generic whatever the category's style.
    assert_eq!(
        cg(ZN::new(3).unwrap()).fusion_style(),
        FusionStyleKind::Generic
    );
    assert_eq!(
        mf(ZN::new(3).unwrap()).fusion_style(),
        FusionStyleKind::Unique
    );
}

/// A category that claims outer multiplicity but certifies nothing else.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ClaimsGeneric {
    canonical_unit: bool,
}

impl FusionCategory for ClaimsGeneric {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::convert::Infallible;

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
        *a == 0
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        1
    }
    fn has_canonical_unit(&self) -> bool {
        self.canonical_unit
    }
    fn f_symbol(&self, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:claims-generic"
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(1))
    }
    fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
        if index == 0 {
            Ok(0)
        } else {
            Err(EnumerationError::OutOfRange)
        }
    }
    fn object_index(&self, _: &u8) -> Result<u64, EnumerationError> {
        Ok(0)
    }
}

#[test]
fn value_dependent_admission_is_checked_at_construction() {
    let generic = ClaimsGeneric {
        canonical_unit: true,
    };
    assert_eq!(
        CategoryRule::try_new(generic.clone(), EnumeratedCodec::<MF>::new()).unwrap_err(),
        CategoryAdmissionError::IncompatibleFusionStyle {
            actual: FusionStyle::Generic
        }
    );
    assert!(CategoryRule::try_new(generic, EnumeratedCodec::<CG>::new()).is_ok());

    let uncertified = ClaimsGeneric {
        canonical_unit: false,
    };
    assert_eq!(
        CategoryRule::try_new(uncertified.clone(), EnumeratedCodec::<CG>::new()).unwrap_err(),
        CategoryAdmissionError::MissingCanonicalUnit
    );
    assert!(CategoryRule::try_new(
        uncertified,
        EnumeratedCodec::<CG, UncertifiedUnitAdmission>::new()
    )
    .is_ok());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_is_refused_as_multiplicity_free_and_admitted_as_checked_generic() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    assert!(matches!(
        CategoryRule::try_new(su3.clone(), EnumeratedCodec::<MF>::new()),
        Err(CategoryAdmissionError::IncompatibleFusionStyle {
            actual: FusionStyle::Generic
        })
    ));
    let rule = cg(su3);
    // Octet (1,1) is enumeration index 4 (grade 2: (0,2), (1,1), (2,0) after
    // grade 1's (0,1), (1,0)); 8 x 8 contains the octet twice.
    let octet = rule
        .try_encode_label(&rule.category().irrep(&[1, 1]).unwrap())
        .unwrap();
    assert_eq!(octet, s(4));
    assert_eq!(rule.try_nsymbol(octet, octet, octet), Ok(2));
    let a = rule.try_a_symbol_generic(octet, octet, octet).unwrap();
    let b = rule.try_b_symbol_generic(octet, octet, octet).unwrap();
    assert_eq!(a.shape(), (2, 2));
    assert_eq!(b.shape(), (2, 2));
}

#[test]
fn multiplicity_free_a_and_b_match_the_hand_values_and_the_builtin_derivation() {
    // SU(2), a = b = 1/2, c = 0: F(1/2,1/2,1/2,1/2; 0, 0) = -1/2 and the
    // Frobenius-Schur phase of spin 1/2 is -1, so A = 2 conj(-1 * -1/2) = 1
    // and B = 2 * (-1/2) = -1 (TensorKitSectors `Asymbol`/`Bsymbol`).
    let rule = mf(LieGroup::new(GroupId::su(2).unwrap()).unwrap());
    numerics::assert_close(
        "A(1/2,1/2,0)",
        rule.a_symbol_scalar(s(1), s(1), s(0)),
        1.0,
        4,
    );
    numerics::assert_close(
        "B(1/2,1/2,0)",
        rule.b_symbol_scalar(s(1), s(1), s(0)),
        -1.0,
        4,
    );
    // The built-in rule still derives A and B from its F in symtn; both
    // routes agree within the tolerance.
    for (a, b, c) in [(1, 1, 0), (1, 1, 2), (2, 1, 1), (2, 2, 2), (3, 2, 1)] {
        let (a, b, c) = (s(a), s(b), s(c));
        numerics::assert_close(
            "A",
            rule.a_symbol_scalar(a, b, c),
            SU2FusionRule.a_symbol_scalar(a, b, c),
            4,
        );
        numerics::assert_close(
            "B",
            rule.b_symbol_scalar(a, b, c),
            SU2FusionRule.b_symbol_scalar(a, b, c),
            4,
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn a_and_b_blocks_are_the_category_authority() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let rule = cg(su3.clone());
    let sun = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let octet = su3.irrep(&[1, 1]).unwrap();
    let rule_octet = rule.try_encode_label(&octet).unwrap();
    let lie_group_octet = sun.encode_dynkin(&[1, 1]).unwrap();
    let authority_a = su3.a_symbol(&octet, &octet, &octet).unwrap();
    let authority_b = su3.b_symbol(&octet, &octet, &octet).unwrap();
    for block in [
        rule.try_a_symbol_generic(rule_octet, rule_octet, rule_octet)
            .unwrap(),
        sun.try_a_symbol_generic(lie_group_octet, lie_group_octet, lie_group_octet)
            .unwrap(),
    ] {
        assert_eq!(block.shape(), (authority_a.dim(), authority_a.dim()));
        assert_eq!(block.data(), authority_a.data());
    }
    for block in [
        rule.try_b_symbol_generic(rule_octet, rule_octet, rule_octet)
            .unwrap(),
        sun.try_b_symbol_generic(lie_group_octet, lie_group_octet, lie_group_octet)
            .unwrap(),
    ] {
        assert_eq!(block.shape(), (authority_b.dim(), authority_b.dim()));
        assert_eq!(block.data(), authority_b.data());
    }
}

// ---------------------------------------------------------------------------
// Rule identity.
// ---------------------------------------------------------------------------

fn hash_of(identity: &RuleIdentity) -> u64 {
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn equal_category_values_share_one_identity() {
    let first = mf(ZN::new(3).unwrap());
    let second = mf(ZN::new(3).unwrap());
    assert_eq!(
        FusionRule::rule_identity(&first),
        FusionRule::rule_identity(&second)
    );
    assert_eq!(
        hash_of(&FusionRule::rule_identity(&first)),
        hash_of(&FusionRule::rule_identity(&second))
    );
    assert_eq!(
        FusionRule::rule_identity(&first.clone()),
        FusionRule::rule_identity(&first)
    );
}

#[test]
fn a_shared_fingerprint_does_not_alias_different_moduli() {
    let z3 = ZN::new(3).unwrap();
    let z5 = ZN::new(5).unwrap();
    assert_eq!(z3.authority_fingerprint(), z5.authority_fingerprint());
    assert_ne!(
        FusionRule::rule_identity(&mf(z3)),
        FusionRule::rule_identity(&mf(z5))
    );
}

/// Values whose hash is constant, so only equality can tell them apart.
#[derive(Clone, Debug, Eq, PartialEq)]
struct CollidingHash(u8);

impl Hash for CollidingHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        0_u8.hash(state);
    }
}

impl FusionCategory for CollidingHash {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn contains(&self, a: &u8) -> bool {
        *a == 0
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        1
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(&self, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:colliding-hash"
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(1))
    }
    fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
        if index == 0 {
            Ok(0)
        } else {
            Err(EnumerationError::OutOfRange)
        }
    }
    fn object_index(&self, _: &u8) -> Result<u64, EnumerationError> {
        Ok(0)
    }
}

#[test]
fn colliding_hashes_stay_distinct_identities() {
    let (one_rule, two_rule) = (mf(CollidingHash(1)), mf(CollidingHash(2)));
    let one = FusionRule::rule_identity(&one_rule);
    let two = FusionRule::rule_identity(&two_rule);
    assert_eq!(hash_of(&one), hash_of(&two));
    assert_ne!(one, two);
    assert_eq!(one, FusionRule::rule_identity(&mf(CollidingHash(1))));
}

#[test]
fn identity_separates_mode_unit_codec_table_and_factor_structure() {
    let z3 = || ZN::new(3).unwrap();
    let mf_id = FusionRule::rule_identity(&mf(z3()));
    let cg_id = CheckedGenericFusion::rule_identity(&cg(z3()));
    let uncertified =
        CategoryRule::try_new(z3(), EnumeratedCodec::<MF, UncertifiedUnitAdmission>::new())
            .unwrap();
    let uncertified_id = FusionRule::rule_identity(&uncertified);
    assert_ne!(mf_id, cg_id);
    assert_ne!(mf_id, uncertified_id);

    // A built-in rule over the same category keeps its own identity.
    assert_ne!(
        mf_id,
        FusionRule::rule_identity(&ZNFusionRule::new(3).unwrap())
    );

    // A codec without a canonical identity: each construction is its own
    // identity, shared by clones.
    let fresh = CategoryRule::try_new(z3(), FreshZ3Codec).unwrap();
    let again = CategoryRule::try_new(z3(), FreshZ3Codec).unwrap();
    assert_ne!(
        FusionRule::rule_identity(&fresh),
        FusionRule::rule_identity(&again)
    );
    assert_eq!(
        FusionRule::rule_identity(&fresh.clone()),
        FusionRule::rule_identity(&fresh)
    );

    let fibonacci = mf(TableCategory::fibonacci());
    let ising = mf(TableCategory::ising());
    assert_ne!(
        FusionRule::rule_identity(&fibonacci),
        FusionRule::rule_identity(&ising)
    );
    assert_eq!(
        FusionRule::rule_identity(&fibonacci),
        FusionRule::rule_identity(&mf(TableCategory::fibonacci()))
    );

    let z2 = || ZN::new(2).unwrap();
    let left_first = FusionRule::rule_identity(&mf(fusion_symbols::product(z2(), z3())));
    let right_first = FusionRule::rule_identity(&mf(fusion_symbols::product(z3(), z2())));
    assert_ne!(left_first, right_first);
    let nested = FusionRule::rule_identity(&mf(fusion_symbols::product(
        fusion_symbols::product(z2(), z3()),
        z2(),
    )));
    let flat = FusionRule::rule_identity(&mf(fusion_symbols::product![z2(), z3(), z2()]));
    assert_ne!(nested, flat);
}

/// A category holding a large table, observed through its `Arc` count.
#[derive(Clone, Debug)]
struct Tracked(Arc<Vec<u8>>);

impl PartialEq for Tracked {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Tracked {}

impl Hash for Tracked {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hashes far less than it holds, as `Hash` permits.
        self.0.len().hash(state);
    }
}

impl FusionCategory for Tracked {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn contains(&self, a: &u8) -> bool {
        *a == 0
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        1
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(&self, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:tracked"
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(1))
    }
    fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
        if index == 0 {
            Ok(0)
        } else {
            Err(EnumerationError::OutOfRange)
        }
    }
    fn object_index(&self, _: &u8) -> Result<u64, EnumerationError> {
        Ok(0)
    }
}

#[test]
fn an_identity_does_not_keep_its_category_alive() {
    let table = Arc::new(vec![7_u8; 1 << 20]);
    let rule = mf(Tracked(Arc::clone(&table)));
    let clone = rule.clone();
    let identity = FusionRule::rule_identity(&rule);
    // The key's charge does not depend on what the category holds.
    let charge = identity.charged_retained_bytes();
    assert!(charge < 256, "{charge}");
    assert_eq!(
        FusionRule::rule_identity(&mf(ZN::new(3).unwrap())).charged_retained_bytes()
            - std::mem::size_of::<ZN>(),
        charge - std::mem::size_of::<Tracked>()
    );
    assert_eq!(Arc::strong_count(&table), 2);

    drop(rule);
    assert_eq!(Arc::strong_count(&table), 2);
    drop(clone);
    // Only the identity (as a cache key would) survives, and it released the
    // category.
    assert_eq!(Arc::strong_count(&table), 1);
    assert_eq!(identity.clone(), identity);
}

#[test]
fn the_public_codec_hook_never_aliases_categories() {
    let codec = EnumeratedCodec::<MF>::new();
    let z3 = CategoryCodec::<ZN>::rule_identity(&codec, &ZN::new(3).unwrap()).unwrap();
    let z5 = CategoryCodec::<ZN>::rule_identity(&codec, &ZN::new(5).unwrap()).unwrap();
    assert_ne!(z3, z5);
    // Without its rule the request cannot compare categories, so even an
    // equal category gives a distinct identity; clones stay equal.
    assert_ne!(
        z3,
        CategoryCodec::<ZN>::rule_identity(&codec, &ZN::new(3).unwrap()).unwrap()
    );
    assert_eq!(z3.clone(), z3);
    // Composition before resolution stays distinct as well.
    let tag = RuleIdentity::of_type::<WrappedZn>();
    assert_ne!(
        RuleIdentity::compose_with_codec::<WrappedZn>(z3.clone(), tag.clone()),
        RuleIdentity::compose_with_codec::<WrappedZn>(z5, tag)
    );
    // A codec that composes the hook's identity gives each rule its own.
    let rule3 = CategoryRule::try_new(ZN::new(3).unwrap(), WrappedZn).unwrap();
    let rule5 = CategoryRule::try_new(ZN::new(5).unwrap(), WrappedZn).unwrap();
    assert_ne!(
        FusionRule::rule_identity(&rule3),
        FusionRule::rule_identity(&rule5)
    );
    assert_eq!(
        FusionRule::rule_identity(&rule3.clone()),
        FusionRule::rule_identity(&rule3)
    );
}

/// An expert codec built on `EnumeratedCodec` whose identity composes the
/// enumerated identity before any rule resolves it.
#[derive(Clone, Copy, Debug)]
struct WrappedZn;

type Enumerated = EnumeratedCodec<MF>;

impl CategoryCodec<ZN> for WrappedZn {
    type Sector = ZNIrrep;
    type Error = FusionAlgebraError;
    type Mode = MF;
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
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        <Enumerated as CategoryCodec<ZN>>::map_error(&Enumerated::new(), inputs, error)
    }
}

#[test]
fn a_dropped_category_identity_equals_no_live_identity() {
    let first = mf(ZN::new(3).unwrap());
    let dead = FusionRule::rule_identity(&first);
    let live_rule = mf(ZN::new(3).unwrap());
    let live = FusionRule::rule_identity(&live_rule);
    assert_eq!(dead, live);
    drop(first);
    assert_ne!(dead, live);
    assert_ne!(live, dead);
    assert_eq!(hash_of(&dead), hash_of(&live));
    // A rule built later over an equal category shares the live identity.
    assert_eq!(live, FusionRule::rule_identity(&mf(ZN::new(3).unwrap())));
}

/// Z3 under ids equal to its charges, with no canonical identity.
#[derive(Clone, Copy, Debug)]
struct FreshZ3Codec;

impl CategoryCodec<ZN> for FreshZ3Codec {
    type Sector = ZNIrrep;
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        rule: &RuleIdentity,
        label: &ZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        self.sector_id(category, rule, label)
    }
    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<ZNIrrep, FusionAlgebraError> {
        self.object(category, id)
    }
    fn object<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<ZNIrrep, FusionAlgebraError> {
        u32::try_from(id.id())
            .ok()
            .and_then(|charge| ZNIrrep::from_canonical(charge, category().modulus()).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a ZN + Copy,
        _: &RuleIdentity,
        object: &ZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(object.charge() as usize))
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        match error {
            CategoryAdapterError::ForeignObject {
                sector: Some(sector),
            } => FusionAlgebraError::InvalidSector { sector },
            other => FusionAlgebraError::CategoryFailure(CategoryFailure::new(other)),
        }
    }
}

// ---------------------------------------------------------------------------
// Enumeration-backed ids.
// ---------------------------------------------------------------------------

#[test]
fn enumeration_ids_round_trip_and_reach_past_32_bits() {
    let rule = mf(U1);
    let u1 = |q: i32| FsU1Irrep::new(q).unwrap();
    assert_eq!(rule.encode_sector(&u1(0)), Ok(s(0)));
    assert_eq!(rule.encode_sector(&u1(1)), Ok(s(1)));
    assert_eq!(rule.encode_sector(&u1(-1)), Ok(s(2)));
    assert_eq!(rule.decode_sector(s(3)), Ok(u1(2)));

    // The last label, -i32::MAX, is index 2^32 - 2; index 2^32 - 1 would be
    // charge 2^31, which no label holds.
    let last = u32::MAX as usize - 1;
    assert_eq!(rule.encode_sector(&u1(-i32::MAX)), Ok(s(last)));
    assert_eq!(rule.decode_sector(s(last)), Ok(u1(-i32::MAX)));
    assert_eq!(rule.decode_sector(s(last + 1)), invalid(last + 1));

    // (-2^16, -2^16) in U(1) x U(1) is the coordinate pair (2^17, 2^17):
    // sum 2^18, index 2^18 (2^18 + 1) / 2 + 2^17, past every 32-bit id.
    let pair = mf(fusion_symbols::product(U1, U1));
    let label = (u1(-65_536), u1(-65_536));
    #[cfg(target_pointer_width = "64")]
    {
        let id = s(34_360_000_512);
        assert_eq!(pair.encode_sector(&label), Ok(id));
        assert_eq!(pair.decode_sector(id), Ok(label));
    }
    #[cfg(target_pointer_width = "32")]
    assert!(matches!(
        pair.encode_sector(&label),
        Err(FusionAlgebraError::UnrepresentableSectorLabel { .. })
    ));

    // Channels keep the category's order; sector order is the index.
    assert_eq!(
        rule.try_fusion_channels(s(1), s(1)).unwrap().as_slice(),
        &[s(3)]
    );
    assert_eq!(
        FusionRule::sector_order_key(&rule, s(2)),
        SectorOrderKey::position(2)
    );
}

#[test]
fn a_product_category_is_numbered_in_bounded_manhattan_order() {
    let z2 = ZN::new(2).unwrap();
    let z3 = ZN::new(3).unwrap();
    let rule = mf(fusion_symbols::product(z2, z3));
    let label = |a: u32, b: u32| {
        (
            ZNIrrep::from_canonical(a, 2).unwrap(),
            ZNIrrep::from_canonical(b, 3).unwrap(),
        )
    };
    // Bounds (2, 3), by coordinate sum, then lexicographically.
    let order = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)];
    for (id, &(a, b)) in order.iter().enumerate() {
        assert_eq!(rule.decode_sector(s(id)), Ok(label(a, b)));
        assert_eq!(rule.encode_sector(&label(a, b)), Ok(s(id)));
    }
    assert_eq!(rule.decode_sector(s(6)), invalid(6));
    // (1,1) x (1,2) = (0,0).
    assert_eq!(
        rule.try_fusion_channels(s(4), s(5)).unwrap().as_slice(),
        &[s(0)]
    );
    assert_eq!(rule.try_dual_sector(s(5)), Ok(s(4)));

    // An unbounded factor: (2, ∞) with U(1)'s 0, +1, -1, ... second.
    let mixed = mf(fusion_symbols::product(z2, U1));
    let pair = |a: u32, q: i32| {
        (
            ZNIrrep::from_canonical(a, 2).unwrap(),
            FsU1Irrep::new(q).unwrap(),
        )
    };
    let order = [(0, 0), (0, 1), (1, 0), (0, -1), (1, 1), (0, 2)];
    for (id, &(a, q)) in order.iter().enumerate() {
        assert_eq!(mixed.decode_sector(s(id)), Ok(pair(a, q)));
    }
}

// ---------------------------------------------------------------------------
// Checked Generic storage, rejection, and error sources.
// ---------------------------------------------------------------------------

#[test]
fn checked_generic_blocks_take_their_shape_from_the_multiplicities() {
    let rule = cg(ZN::new(3).unwrap());
    let admissible = rule
        .try_f_symbol_generic(s(1), s(1), s(1), s(0), s(2), s(2))
        .unwrap();
    assert_eq!(admissible.shape(), (1, 1, 1, 1));
    assert_eq!(admissible.data(), &[1.0]);
    // fusion-symbols stores the absent vertex as one zero; its block is empty.
    let absent = rule
        .try_f_symbol_generic(s(1), s(1), s(1), s(0), s(0), s(2))
        .unwrap();
    // N(1,1,0) = N(0,1,0) = 0, N(1,1,2) = N(1,2,0) = 1.
    assert_eq!(absent.shape(), (0, 0, 1, 1));
    assert!(absent.data().is_empty());
    let r = rule.try_r_symbol_generic(s(1), s(1), s(0)).unwrap();
    assert_eq!(r.shape(), (0, 0));
    let r = rule.try_r_symbol_generic(s(1), s(1), s(2)).unwrap();
    assert_eq!((r.shape(), r.data()), ((1, 1), &[1.0][..]));
    let a = rule.try_a_symbol_generic(s(1), s(2), s(0)).unwrap();
    assert_eq!((a.shape(), a.data()), ((1, 1), &[1.0][..]));
    let b = rule.try_b_symbol_generic(s(1), s(1), s(0)).unwrap();
    assert_eq!(b.shape(), (0, 0));
    assert_eq!(rule.try_dim_scalar(s(2)), Ok(1.0));
    assert_eq!(rule.try_twist_scalar(s(2)), Ok(1.0));
    assert_eq!(
        rule.try_f_symbol_generic(s(3), s(0), s(0), s(0), s(0), s(0))
            .unwrap_err(),
        FusionAlgebraError::InvalidSector { sector: s(3) }
    );
}

/// Deliberately independent storage axes and checked multiplicities.
#[derive(Clone, Debug, PartialEq)]
struct ProbeBlock<const RANK: usize> {
    axes: [usize; RANK],
    data: &'static [f64],
}

impl<const RANK: usize> Eq for ProbeBlock<RANK> {}
impl<const RANK: usize> Hash for ProbeBlock<RANK> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.axes.hash(state);
        for value in self.data {
            (if *value == 0.0 { 0 } else { value.to_bits() }).hash(state);
        }
    }
}

impl<const RANK: usize> SymbolBlock<RANK> for ProbeBlock<RANK> {
    type Scalar = f64;
    fn dims(&self) -> [usize; RANK] {
        self.axes
    }
    fn get(&self, index: [usize; RANK]) -> f64 {
        let offset = index
            .into_iter()
            .zip(self.axes)
            .fold(0, |offset, (i, n)| offset * n + i);
        self.data[offset]
    }
    fn as_slice(&self) -> &[f64] {
        self.data
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StorageProbe {
    n: u32,
    claimed_mf: bool,
    checked_n: Option<(u32, u32)>,
    f: ProbeBlock<4>,
    r: ProbeBlock<2>,
}

thread_local! {
    static STORAGE_N_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl StorageProbe {
    fn new(n: u32, data: &'static [f64]) -> Self {
        Self {
            n,
            claimed_mf: false,
            checked_n: None,
            f: ProbeBlock { axes: [1; 4], data },
            r: ProbeBlock { axes: [1; 2], data },
        }
    }
}

impl FusionCategory for StorageProbe {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = ProbeBlock<4>;
    type RSymbol = ProbeBlock<2>;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        if self.n > 1 && !self.claimed_mf {
            FusionStyle::Generic
        } else {
            FusionStyle::Simple
        }
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
        1 - a
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        STORAGE_N_CALLS.with(|n| n.set(n.get() + 1));
        self.n
    }
    fn try_nsymbol(&self, a: &u8, b: &u8, c: &u8) -> Result<u32, Self::Error> {
        STORAGE_N_CALLS.with(|n| n.set(n.get() + 1));
        Ok(self.checked_n.map_or(
            self.n,
            |(row, col)| if [*a, *b, *c] == [1; 3] { row } else { col },
        ))
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(
        &self,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
    ) -> Result<Self::FSymbol, Self::Error> {
        Ok(self.f.clone())
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<Self::RSymbol, Self::Error> {
        Ok(self.r.clone())
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:storage-probe"
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(2))
    }
    fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
        u8::try_from(index)
            .ok()
            .filter(|&a| a < 2)
            .ok_or(EnumerationError::OutOfRange)
    }
    fn object_index(&self, a: &u8) -> Result<u64, EnumerationError> {
        Ok(u64::from(*a))
    }
}

#[test]
fn scalar_validation_uses_n_before_accepting_empty_or_bare_zero_storage() {
    for (n, data, expected) in [
        (
            1,
            &[][..],
            Err(SymbolShapeError {
                actual_len: 0,
                expected_len: Some(1),
            }),
        ),
        (0, &[][..], Ok(0.0)),
        (0, &[0.0][..], Ok(0.0)),
        (
            0,
            &[1.0][..],
            Err(SymbolShapeError {
                actual_len: 1,
                expected_len: Some(0),
            }),
        ),
        (
            0,
            &[0.0, 0.0][..],
            Err(SymbolShapeError {
                actual_len: 2,
                expected_len: Some(0),
            }),
        ),
        (1, &[0.0][..], Ok(0.0)),
        (1, &[1.0][..], Ok(1.0)),
    ] {
        let scalar = mf(StorageProbe::new(n, data));
        for result in [
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scalar.f_symbol_scalar(s(1), s(1), s(1), s(1), s(1), s(1))
            })),
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scalar.r_symbol_scalar(s(1), s(1), s(1))
            })),
        ] {
            match expected {
                Ok(value) => assert_eq!(result.unwrap(), value),
                Err(_) => assert!(
                    result.is_err(),
                    "malformed multiplicity-free storage must panic"
                ),
            }
        }
        let generic = cg(StorageProbe::new(n, data));
        for result in [
            generic
                .try_f_symbol_generic(s(1), s(1), s(1), s(1), s(1), s(1))
                .map(|block| block.data().to_vec()),
            generic
                .try_r_symbol_generic(s(1), s(1), s(1))
                .map(|block| block.data().to_vec()),
        ] {
            match expected {
                Ok(value) => assert_eq!(result.unwrap(), if n == 0 { vec![] } else { vec![value] }),
                Err(shape) => assert_eq!(
                    adapter_source::<std::convert::Infallible>(&result.unwrap_err()),
                    &CategoryAdapterError::MalformedSymbolData(shape)
                ),
            }
        }
    }
}

#[test]
fn falsely_multiplicity_free_categories_cannot_discard_extra_coefficients() {
    let mut category = StorageProbe::new(2, &[1.0; 16]);
    category.claimed_mf = true;
    category.f.axes = [2; 4];
    category.r = ProbeBlock {
        axes: [2; 2],
        data: &[1.0; 4],
    };
    let rule = mf(category);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rule.f_symbol_scalar(
            s(1),
            s(1),
            s(1),
            s(1),
            s(1),
            s(1)
        )))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rule.r_symbol_scalar(
            s(1),
            s(1),
            s(1)
        )))
        .is_err()
    );
}

#[test]
fn equal_volume_axis_corruption_is_rejected_for_f_r_a_and_b() {
    let mut category = StorageProbe::new(2, &[1.0; 16]);
    category.f.axes = [1, 4, 2, 2];
    category.r = ProbeBlock {
        axes: [1, 4],
        data: &[1.0; 4],
    };
    let rule = cg(category.clone());
    for (error, expected, actual) in [
        (
            rule.try_f_symbol_generic(s(1), s(1), s(1), s(1), s(1), s(1))
                .unwrap_err(),
            vec![2; 4],
            vec![1, 4, 2, 2],
        ),
        (
            rule.try_r_symbol_generic(s(1), s(1), s(1)).unwrap_err(),
            vec![2; 2],
            vec![1, 4],
        ),
    ] {
        assert_eq!(
            adapter_source::<std::convert::Infallible>(&error),
            &CategoryAdapterError::MalformedSymbolAxes { expected, actual }
        );
    }
    // Numerical zero coefficients do not make a live vertex absent.
    let mut zero = category.clone();
    zero.f.data = &[0.0; 16];
    zero.r.data = &[0.0; 4];
    let zero = cg(zero);
    assert!(matches!(
        adapter_source::<std::convert::Infallible>(
            &zero
                .try_f_symbol_generic(s(1), s(1), s(1), s(1), s(1), s(1))
                .unwrap_err()
        ),
        CategoryAdapterError::MalformedSymbolAxes { .. }
    ));
    assert!(matches!(
        adapter_source::<std::convert::Infallible>(
            &zero.try_r_symbol_generic(s(1), s(1), s(1)).unwrap_err()
        ),
        CategoryAdapterError::MalformedSymbolAxes { .. }
    ));
    category.f.axes = [2; 4];
    category.r.axes = [2; 2];
    let valid = cg(category.clone());
    assert!(valid
        .try_f_symbol_generic(s(1), s(1), s(1), s(1), s(1), s(1))
        .is_ok());
    assert!(valid.try_r_symbol_generic(s(1), s(1), s(1)).is_ok());

    // Default fs A/B constructs 2 x 2 storage; checked N reports 1 x 4.
    category.f.data = &[1.0; 4];
    category.checked_n = Some((1, 4));
    let rule = cg(category.clone());
    for error in [
        rule.try_a_symbol_generic(s(1), s(1), s(1)).unwrap_err(),
        rule.try_b_symbol_generic(s(1), s(1), s(1)).unwrap_err(),
    ] {
        assert_eq!(
            adapter_source::<std::convert::Infallible>(&error),
            &CategoryAdapterError::MalformedSymbolAxes {
                expected: vec![1, 4],
                actual: vec![2, 2]
            }
        );
    }
    category.checked_n = None;
    let valid = cg(category);
    assert!(valid.try_a_symbol_generic(s(1), s(1), s(1)).is_ok());
    assert!(valid.try_b_symbol_generic(s(1), s(1), s(1)).is_ok());
}

#[test]
fn adapter_n_query_budgets_cover_leaf_and_both_product_nestings() {
    fn count(query: impl FnOnce()) -> usize {
        STORAGE_N_CALLS.with(|n| n.set(0));
        query();
        STORAGE_N_CALLS.with(std::cell::Cell::get)
    }
    fn check<C>(category: C, id: SectorId, expected: [usize; 4])
    where
        C: FusionCategory<Scalar = f64> + Send + Sync + 'static,
        C::Error: Send + Sync + 'static,
    {
        let generic = cg(category.clone());
        let scalar = mf(category);
        assert_eq!(
            [
                count(|| {
                    assert_eq!(
                        generic
                            .try_f_symbol_generic(id, id, id, id, id, id)
                            .unwrap()
                            .data(),
                        &[1.0]
                    );
                }),
                count(|| {
                    assert_eq!(
                        generic.try_r_symbol_generic(id, id, id).unwrap().data(),
                        &[1.0]
                    );
                }),
                count(|| {
                    assert_eq!(
                        generic.try_a_symbol_generic(id, id, id).unwrap().data(),
                        &[1.0]
                    );
                }),
                count(|| {
                    assert_eq!(
                        generic.try_b_symbol_generic(id, id, id).unwrap().data(),
                        &[1.0]
                    );
                }),
            ],
            expected
        );
        assert_eq!(
            [
                count(|| {
                    assert_eq!(scalar.f_symbol_scalar(id, id, id, id, id, id), 1.0);
                }),
                count(|| {
                    assert_eq!(scalar.r_symbol_scalar(id, id, id), 1.0);
                }),
                count(|| {
                    assert_eq!(scalar.a_symbol_scalar(id, id, id), 1.0);
                }),
                count(|| {
                    assert_eq!(scalar.b_symbol_scalar(id, id, id), 1.0);
                }),
            ],
            expected
        );
    }
    let leaf = StorageProbe::new(1, &[1.0]);
    check(leaf.clone(), s(1), [4, 2, 3, 3]);
    // fs checks product construction and leaf outputs; the adapter adds one
    // logical-axis pass. Nesting does not multiply the number of leaf queries.
    check(
        fusion_symbols::product(
            fusion_symbols::product(leaf.clone(), leaf.clone()),
            leaf.clone(),
        ),
        s(7),
        [36, 18, 21, 21],
    );
    check(
        fusion_symbols::product(leaf.clone(), fusion_symbols::product(leaf.clone(), leaf)),
        s(7),
        [36, 18, 21, 21],
    );
}

#[test]
fn a_fusion_only_category_rejects_braiding_with_a_typed_source() {
    // Rep(S3), multiplicity-free and without braiding.
    let rep_s3 = TableCategory::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c").unwrap();
    let rule = cg(rep_s3);
    assert_eq!(rule.braiding_style(), BraidingStyleKind::NoBraiding);
    let error = rule.try_twist_scalar(s(1)).unwrap_err();
    assert_eq!(
        adapter_source::<fusion_symbols::table::TableError>(&error),
        &CategoryAdapterError::NoBraiding
    );
    let error = rule.try_r_symbol_generic(s(1), s(1), s(0)).unwrap_err();
    assert_eq!(
        adapter_source::<fusion_symbols::table::TableError>(&error),
        &CategoryAdapterError::NoBraiding
    );
    // F needs no braiding.
    assert!(rule
        .try_f_symbol_generic(s(0), s(0), s(0), s(0), s(0), s(0))
        .is_ok());
}

/// A category whose authority fails, reports a nonpositive dimension, and
/// returns storage its multiplicities do not admit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Faulty;

#[derive(Debug, PartialEq)]
struct AuthorityFailed;

impl std::fmt::Display for AuthorityFailed {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("authority failed")
    }
}

impl std::error::Error for AuthorityFailed {}

impl FusionCategory for Faulty {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = AuthorityFailed;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn contains(&self, a: &u8) -> bool {
        *a < 3
    }
    fn dual(&self, a: &u8) -> u8 {
        *a
    }
    fn dim(&self, a: &u8) -> f64 {
        if *a == 2 {
            0.0
        } else {
            1.0
        }
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
    fn fusion<'a>(&'a self, a: &u8, _: &u8) -> Result<Self::Fusion<'a>, AuthorityFailed> {
        match a {
            1 => Err(AuthorityFailed),
            // An outcome outside the category.
            2 => Ok(std::iter::once(7)),
            _ => Ok(std::iter::once(0)),
        }
    }
    fn nsymbol(&self, a: &u8, b: &u8, c: &u8) -> u32 {
        u32::from(a ^ b == *c)
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(
        &self,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
    ) -> Result<f64, AuthorityFailed> {
        // Nonzero even where a vertex is absent.
        Ok(1.0)
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, AuthorityFailed> {
        Ok(1.0)
    }
    fn a_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, AuthorityFailed> {
        // A nonzero 1 x 1 block even where N(a,b,c) = 0.
        Ok(RBlock::one(1.0))
    }
    fn b_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, AuthorityFailed> {
        Ok(RBlock::one(1.0))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:faulty"
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(3))
    }
    fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
        // Index 3 unranks to an object outside the category.
        u8::try_from(index)
            .ok()
            .filter(|&object| object <= 3)
            .ok_or(EnumerationError::OutOfRange)
    }
    fn object_index(&self, a: &u8) -> Result<u64, EnumerationError> {
        Ok(u64::from(*a))
    }
}

#[test]
fn authority_dimension_and_storage_failures_keep_their_source() {
    let rule = cg(Faulty);
    let error = rule.try_fusion_channels(s(1), s(0)).unwrap_err();
    let FusionAlgebraError::CategoryFailure(failure) = &error else {
        panic!("expected a category failure, got {error:?}");
    };
    assert!(std::error::Error::source(failure)
        .and_then(|source| source.downcast_ref::<AuthorityFailed>())
        .is_some());
    assert_eq!(error.clone(), error);

    let error = rule.try_dim_scalar(s(2)).unwrap_err();
    assert_eq!(
        adapter_source::<AuthorityFailed>(&error),
        &CategoryAdapterError::DimensionNotRepresentable { sector: s(2) }
    );
    // F(1,1,1,1; e=1, f=1): N(1,1,1) = 0, yet the authority stores 1.
    let error = rule
        .try_f_symbol_generic(s(1), s(1), s(1), s(1), s(1), s(1))
        .unwrap_err();
    assert_eq!(
        adapter_source::<AuthorityFailed>(&error),
        &CategoryAdapterError::MalformedSymbolData(SymbolShapeError {
            actual_len: 1,
            expected_len: Some(0),
        })
    );
}

#[test]
fn a_and_b_take_their_shape_from_n_and_require_finite_dimensions() {
    let rule = cg(Faulty);
    // N(1,1,1) = 0, yet the authority returns a nonzero 1 x 1 block.
    for error in [
        rule.try_a_symbol_generic(s(1), s(1), s(1)).unwrap_err(),
        rule.try_b_symbol_generic(s(1), s(1), s(1)).unwrap_err(),
    ] {
        assert_eq!(
            adapter_source::<AuthorityFailed>(&error),
            &CategoryAdapterError::MalformedSymbolData(SymbolShapeError {
                actual_len: 1,
                expected_len: Some(0),
            })
        );
    }
    // N(2,0,2) = 1, but dim(2) = 0: refused before the symbols are asked,
    // not a NaN normalization inside `Ok`.
    for error in [
        rule.try_a_symbol_generic(s(2), s(0), s(2)).unwrap_err(),
        rule.try_b_symbol_generic(s(2), s(0), s(2)).unwrap_err(),
    ] {
        assert_eq!(
            adapter_source::<AuthorityFailed>(&error),
            &CategoryAdapterError::DimensionNotRepresentable { sector: s(2) }
        );
    }
    // The admissible vertex keeps its 1 x 1 block.
    let a = rule.try_a_symbol_generic(s(0), s(0), s(0)).unwrap();
    assert_eq!((a.shape(), a.data()), ((1, 1), &[1.0][..]));
}

#[test]
fn a_codec_that_cannot_encode_the_unit_is_refused() {
    #[derive(Clone, Copy, Debug)]
    struct NoUnit;
    impl CategoryCodec<ZN> for NoUnit {
        type Sector = ZNIrrep;
        type Error = FusionAlgebraError;
        type Mode = MF;
        type Unit = CanonicalUnitAdmission;
        fn encode_label<'a>(
            &self,
            category: impl Fn() -> &'a ZN + Copy,
            rule: &RuleIdentity,
            label: &ZNIrrep,
        ) -> Result<SectorId, FusionAlgebraError> {
            self.sector_id(category, rule, label)
        }
        fn decode_label<'a>(
            &self,
            category: impl Fn() -> &'a ZN + Copy,
            id: SectorId,
        ) -> Result<ZNIrrep, FusionAlgebraError> {
            self.object(category, id)
        }
        fn object<'a>(
            &self,
            _: impl Fn() -> &'a ZN + Copy,
            id: SectorId,
        ) -> Result<ZNIrrep, FusionAlgebraError> {
            invalid(id.id())
        }
        fn sector_id<'a>(
            &self,
            _: impl Fn() -> &'a ZN + Copy,
            _: &RuleIdentity,
            _: &ZNIrrep,
        ) -> Result<SectorId, FusionAlgebraError> {
            invalid(7)
        }
        fn map_error(
            &self,
            _: &[SectorId],
            error: CategoryAdapterError<std::convert::Infallible>,
        ) -> FusionAlgebraError {
            FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
        }
    }
    assert_eq!(
        CategoryRule::try_new(ZN::new(3).unwrap(), NoUnit).unwrap_err(),
        CategoryAdmissionError::Codec(FusionAlgebraError::InvalidSector { sector: s(7) })
    );
}

/// A category whose unit is not its member: a broken authority.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ForeignUnit;

impl FusionCategory for ForeignUnit {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn contains(&self, a: &u8) -> bool {
        *a == 1
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(1))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        1
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(&self, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:foreign-unit"
    }
}

/// Ids equal to the `u8` objects, with no membership checks of its own and
/// every adapter failure kept whole as the source.
#[derive(Clone, Copy, Debug)]
struct RawU8;

impl<C> CategoryCodec<C> for RawU8
where
    C: FusionCategory<Object = u8> + 'static,
    C::Error: Send + Sync + 'static,
{
    type Sector = u8;
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a C + Copy,
        _: &RuleIdentity,
        label: &u8,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(usize::from(*label)))
    }
    fn decode_label<'a>(
        &self,
        _: impl Fn() -> &'a C + Copy,
        id: SectorId,
    ) -> Result<u8, FusionAlgebraError> {
        u8::try_from(id.id()).map_err(|_| FusionAlgebraError::InvalidSector { sector: id })
    }
    fn object<'a>(
        &self,
        category: impl Fn() -> &'a C + Copy,
        id: SectorId,
    ) -> Result<u8, FusionAlgebraError> {
        <Self as CategoryCodec<C>>::decode_label(self, category, id)
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a C + Copy,
        _: &RuleIdentity,
        object: &u8,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(usize::from(*object)))
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<C::Error>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn a_unit_outside_the_category_is_refused_even_when_it_round_trips() {
    let error = CategoryRule::try_new(ForeignUnit, RawU8).unwrap_err();
    let CategoryAdmissionError::Codec(error) = &error else {
        panic!("expected a codec error, got {error:?}");
    };
    assert_eq!(
        adapter_source::<std::convert::Infallible>(error),
        &CategoryAdapterError::ForeignObject { sector: Some(s(0)) }
    );
}

// ---------------------------------------------------------------------------
// One merged `admit` hook expresses every frozen validation schedule
// (`tests/provider_contracts/errors.rs`).
// ---------------------------------------------------------------------------

/// The first error of `checks`, in order.
fn first_error(checks: &[(bool, usize)]) -> Result<(), FusionAlgebraError> {
    match checks.iter().find(|(valid, _)| !valid) {
        Some(&(_, id)) => invalid(id),
        None => Ok(()),
    }
}

/// ZN with ids equal to charges and a chosen N schedule.
#[derive(Clone, Copy, Debug)]
struct ScheduledZn {
    coupled_first: bool,
}

impl CategoryCodec<ZN> for ScheduledZn {
    type Sector = ZNIrrep;
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        rule: &RuleIdentity,
        label: &ZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        self.sector_id(category, rule, label)
    }
    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<ZNIrrep, FusionAlgebraError> {
        self.object(category, id)
    }
    fn object<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<ZNIrrep, FusionAlgebraError> {
        u32::try_from(id.id())
            .ok()
            .and_then(|charge| ZNIrrep::from_canonical(charge, category().modulus()).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a ZN + Copy,
        _: &RuleIdentity,
        object: &ZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(object.charge() as usize))
    }
    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        _: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[ZNIrrep; N], FusionAlgebraError> {
        let modulus = category().modulus() as usize;
        let valid = |id: SectorId| (id.id() < modulus, id.id());
        let checks: Vec<_> = match (operation, self.coupled_first, ids.len()) {
            (CategoryOperation::N, true, 3) => vec![valid(ids[2]), valid(ids[0]), valid(ids[1])],
            _ => ids.iter().map(|&id| valid(id)).collect(),
        };
        first_error(&checks)?;
        let mut objects = [category().unit(); N];
        for (object, id) in objects.iter_mut().zip(ids) {
            *object = self.object(category, id)?;
        }
        Ok(objects)
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn admission_expresses_the_zn_and_parity_schedules() {
    let z5 = CategoryRule::try_new(
        ZN::new(5).unwrap(),
        ScheduledZn {
            coupled_first: true,
        },
    )
    .unwrap();
    // ZN: coupled, then left, then right.
    assert_eq!(z5.try_nsymbol(s(5), s(6), s(7)), invalid(7));
    assert_eq!(z5.try_nsymbol(s(5), s(6), s(0)), invalid(5));
    assert_eq!(z5.try_nsymbol(s(0), s(6), s(0)), invalid(6));
    assert_eq!(z5.try_fusion_channels(s(5), s(6)), invalid(5));
    assert_eq!(z5.try_dual_sector(s(5)), invalid(5));

    // Z2, fermion parity, and tables: left, right, then coupled, which is
    // the default schedule.
    let z2 = mf(ZN::new(2).unwrap());
    assert_eq!(z2.try_nsymbol(s(2), s(3), s(4)), invalid(2));
    assert_eq!(z2.try_nsymbol(s(0), s(3), s(4)), invalid(3));
    assert_eq!(z2.try_nsymbol(s(0), s(0), s(4)), invalid(4));
    let fibonacci = mf(TableCategory::fibonacci());
    assert_eq!(fibonacci.try_nsymbol(s(2), s(3), s(4)), invalid(2));
    assert_eq!(fibonacci.try_nsymbol(s(0), s(3), s(4)), invalid(3));
    assert_eq!(fibonacci.try_nsymbol(s(0), s(0), s(4)), invalid(4));
}

/// U(1) with zigzag ids that, as `U1FusionRule`, refuses a fusion whose
/// total charge leaves `i32` after validating every label.
#[derive(Clone, Copy, Debug)]
struct ClosedU1;

impl CategoryCodec<U1> for ClosedU1 {
    type Sector = FsU1Irrep;
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a U1 + Copy,
        rule: &RuleIdentity,
        label: &FsU1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        self.sector_id(category, rule, label)
    }
    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a U1 + Copy,
        id: SectorId,
    ) -> Result<FsU1Irrep, FusionAlgebraError> {
        self.object(category, id)
    }
    fn object<'a>(
        &self,
        _: impl Fn() -> &'a U1 + Copy,
        id: SectorId,
    ) -> Result<FsU1Irrep, FusionAlgebraError> {
        U1Irrep::from_sector_id(id)
            .and_then(|irrep| FsU1Irrep::new(irrep.charge()).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a U1 + Copy,
        _: &RuleIdentity,
        object: &FsU1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(U1Irrep::new(object.charge()).sector_id())
    }
    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a U1 + Copy,
        _: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[FsU1Irrep; N], FusionAlgebraError> {
        let mut objects = [category().unit(); N];
        for (object, id) in objects.iter_mut().zip(ids) {
            *object = self.object(category, id)?;
        }
        if matches!(operation, CategoryOperation::N | CategoryOperation::Fusion) {
            let (left, right) = (objects[0].charge(), objects[1].charge());
            // The sum must be a label, and i32::MIN has no representable dual.
            if left.checked_add(right).is_none_or(|sum| sum == i32::MIN) {
                return Err(FusionAlgebraError::U1FusionOverflow { left, right });
            }
        }
        Ok(objects)
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<fusion_symbols::lie::U1Error>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn admission_expresses_the_u1_schedule() {
    let rule = CategoryRule::try_new(U1, ClosedU1).unwrap();
    let u1 = |q: i32| U1Irrep::new(q).sector_id();
    let excluded = u32::MAX as usize;
    assert_eq!(
        rule.try_nsymbol(u1(i32::MAX), u1(1), s(excluded)),
        invalid(excluded)
    );
    assert_eq!(
        rule.try_nsymbol(u1(i32::MAX), u1(1), u1(0)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        })
    );
    assert_eq!(
        rule.try_fusion_channels(u1(i32::MIN + 1), u1(-1)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MIN + 1,
            right: -1,
        })
    );
}

/// Compact SU(2): doubled spins `0..=254`, N checked coupled, left, right,
/// then the whole closure, and an R known to be zero before the coupled id
/// is converted.
struct CompactSu2<M>(std::marker::PhantomData<M>);

impl<M> CompactSu2<M> {
    const NEW: Self = Self(std::marker::PhantomData);

    fn checked(id: SectorId) -> Result<i64, FusionAlgebraError> {
        if id.id() <= SU2_MAX_DOUBLED_SPIN {
            Ok(id.id() as i64)
        } else {
            invalid(id.id())
        }
    }
}

impl<M: 'static> CategoryCodec<LieGroup> for CompactSu2<M> {
    type Sector = SU2Irrep;
    type Error = FusionAlgebraError;
    type Mode = M;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a LieGroup + Copy,
        _: &RuleIdentity,
        label: &SU2Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(label.sector_id())
    }
    fn decode_label<'a>(
        &self,
        _: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<SU2Irrep, FusionAlgebraError> {
        SU2Irrep::try_from_sector_id(id).ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }
    fn object<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<LieIrrep, FusionAlgebraError> {
        let twice_spin = Self::checked(id)?;
        category()
            .irrep(&[twice_spin])
            .map_err(|_| FusionAlgebraError::InvalidSector { sector: id })
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a LieGroup + Copy,
        _: &RuleIdentity,
        object: &LieIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        let twice_spin = object.dynkin_labels().next().unwrap_or(0) as usize;
        Ok(s(twice_spin))
    }
    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        _: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[LieIrrep; N], FusionAlgebraError> {
        if operation == CategoryOperation::N {
            Self::checked(ids[2])?;
        }
        let twice_spins = ids.map(Self::checked);
        for twice_spin in &twice_spins {
            twice_spin.clone()?;
        }
        if matches!(operation, CategoryOperation::N | CategoryOperation::Fusion) {
            let (left, right) = (ids[0].id(), ids[1].id());
            if left + right > SU2_MAX_DOUBLED_SPIN {
                return Err(FusionAlgebraError::FusionNotRepresentable {
                    left: ids[0],
                    right: ids[1],
                });
            }
        }
        let group = category();
        Ok(twice_spins.map(|twice_spin| group.irrep(&[twice_spin.unwrap()]).unwrap()))
    }
    fn admit_r<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        rule: &RuleIdentity,
        ids: [SectorId; 3],
    ) -> Result<Option<[LieIrrep; 3]>, FusionAlgebraError> {
        let (left, right) = (Self::checked(ids[0])?, Self::checked(ids[1])?);
        let coupled = ids[2].id() as i64;
        let in_closure = (left - right).abs() <= coupled
            && coupled <= left + right
            && (left + right - coupled) % 2 == 0;
        if !in_closure {
            return Ok(None);
        }
        self.admit(category, rule, CategoryOperation::R, ids)
            .map(Some)
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<LieError>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn admission_expresses_the_compact_su2_schedule_and_r_zero_shortcut() {
    let rule = CategoryRule::try_new(
        LieGroup::new(GroupId::su(2).unwrap()).unwrap(),
        CompactSu2::<MF>::NEW,
    )
    .unwrap();
    let closure = Err(FusionAlgebraError::FusionNotRepresentable {
        left: s(128),
        right: s(127),
    });
    assert_eq!(rule.try_nsymbol(s(128), s(127), s(1)), closure);
    assert_eq!(rule.try_nsymbol(s(128), s(127), s(254)), closure);
    assert_eq!(rule.try_nsymbol(s(128), s(127), s(255)), invalid(255));
    assert_eq!(rule.try_nsymbol(s(127), s(127), s(254)), Ok(1));
    assert_eq!(rule.try_dual_sector(s(255)), invalid(255));

    // R of an absent channel is zero before the coupled id is converted,
    // even for an id past the codec.
    assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(300)), 0.0);
    assert_eq!(
        rule.r_symbol_scalar(s(1), s(1), s(300)),
        SU2FusionRule.r_symbol_scalar(s(1), s(1), s(300))
    );
    assert_eq!(rule.r_symbol_scalar(s(1), s(1), s(0)), -1.0);
}

/// A category that counts how often it is touched after construction.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Watched;

thread_local! {
    static WATCHED_TOUCHES: Cell<usize> = const { Cell::new(0) };
}

impl FusionCategory for Watched {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::convert::Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn contains(&self, a: &u8) -> bool {
        WATCHED_TOUCHES.set(WATCHED_TOUCHES.get() + 1);
        *a == 0
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        1
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn f_symbol(&self, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"test:watched"
    }
}

/// Rejects ids past 0 before asking the category, preserving labels-first
/// validation independently of the category closure.
#[derive(Clone, Copy, Debug)]
struct LabelsFirst;

impl CategoryCodec<Watched> for LabelsFirst {
    type Sector = u8;
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a Watched + Copy,
        _: &RuleIdentity,
        label: &u8,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(usize::from(*label)))
    }
    fn decode_label<'a>(
        &self,
        _: impl Fn() -> &'a Watched + Copy,
        id: SectorId,
    ) -> Result<u8, FusionAlgebraError> {
        if id == s(0) {
            Ok(0)
        } else {
            invalid(id.id())
        }
    }
    fn object<'a>(
        &self,
        category: impl Fn() -> &'a Watched + Copy,
        id: SectorId,
    ) -> Result<u8, FusionAlgebraError> {
        self.decode_label(category, id)
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a Watched + Copy,
        _: &RuleIdentity,
        object: &u8,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(usize::from(*object)))
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn admission_can_reject_labels_before_touching_the_category() {
    let rule = CategoryRule::try_new(Watched, LabelsFirst).unwrap();
    let before = WATCHED_TOUCHES.get();
    assert_eq!(rule.try_nsymbol(s(1), s(0), s(0)), invalid(1));
    assert_eq!(rule.try_dual_sector(s(3)), invalid(3));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rule.a_symbol_scalar(
            s(1),
            s(0),
            s(0)
        )))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rule.b_symbol_scalar(
            s(1),
            s(0),
            s(0)
        )))
        .is_err()
    );
    assert_eq!(WATCHED_TOUCHES.get(), before);
}

/// `Z2 ⊠ Z2` under Cantor pairing, admitted as `ProductFusionRule` does:
/// decode every outer id, then admit the left factor completely, then the
/// right factor.
#[derive(Clone, Copy, Debug)]
struct CantorZ2Z2;

type Z2Z2 = fusion_symbols::product::Product<(ZN, ZN)>;

impl CantorZ2Z2 {
    fn factor(id: SectorId) -> Result<ZNIrrep, FusionAlgebraError> {
        u32::try_from(id.id())
            .ok()
            .and_then(|charge| ZNIrrep::from_canonical(charge, 2).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }
}

impl CategoryCodec<Z2Z2> for CantorZ2Z2 {
    type Sector = (ZNIrrep, ZNIrrep);
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a Z2Z2 + Copy,
        rule: &RuleIdentity,
        label: &(ZNIrrep, ZNIrrep),
    ) -> Result<SectorId, FusionAlgebraError> {
        self.sector_id(category, rule, label)
    }
    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a Z2Z2 + Copy,
        id: SectorId,
    ) -> Result<(ZNIrrep, ZNIrrep), FusionAlgebraError> {
        self.object(category, id)
    }
    fn object<'a>(
        &self,
        _: impl Fn() -> &'a Z2Z2 + Copy,
        id: SectorId,
    ) -> Result<(ZNIrrep, ZNIrrep), FusionAlgebraError> {
        let (left, right) = TensorKitProductCodec::decode_checked(id)?;
        Ok((Self::factor(left)?, Self::factor(right)?))
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a Z2Z2 + Copy,
        _: &RuleIdentity,
        object: &(ZNIrrep, ZNIrrep),
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(TensorKitProductCodec::encode_checked(
            s(object.0.charge() as usize),
            s(object.1.charge() as usize),
        )?)
    }
    fn admit<'a, const N: usize>(
        &self,
        _: impl Fn() -> &'a Z2Z2 + Copy,
        _: &RuleIdentity,
        _: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[(ZNIrrep, ZNIrrep); N], FusionAlgebraError> {
        let mut pairs = [(s(0), s(0)); N];
        for (pair, id) in pairs.iter_mut().zip(ids) {
            *pair = TensorKitProductCodec::decode_checked(id)?;
        }
        let mut objects = [(ZNIrrep::unit(2).unwrap(), ZNIrrep::unit(2).unwrap()); N];
        for (object, pair) in objects.iter_mut().zip(pairs) {
            object.0 = Self::factor(pair.0)?;
        }
        for (object, pair) in objects.iter_mut().zip(pairs) {
            object.1 = Self::factor(pair.1)?;
        }
        Ok(objects)
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<<Z2Z2 as FusionCategory>::Error>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn admission_expresses_the_product_schedule() {
    let z2 = || ZN::new(2).unwrap();
    let rule = CategoryRule::try_new(fusion_symbols::product(z2(), z2()), CantorZ2Z2).unwrap();
    // Cantor (2, 3) = 17: both components invalid; the left factor wins.
    assert_eq!(rule.try_nsymbol(s(17), s(0), s(0)), invalid(2));
    assert_eq!(rule.try_fusion_channels(s(17), s(0)), invalid(2));
    assert_eq!(rule.try_dual_sector(s(17)), invalid(2));
    // Cantor (0, 3) = 6: only the right component is invalid.
    assert_eq!(rule.try_nsymbol(s(6), s(0), s(0)), invalid(3));
    // Cantor (2, 0) = 5 as the coupled id: the left factor checks it.
    assert_eq!(rule.try_nsymbol(s(0), s(0), s(5)), invalid(2));
    // The same literals the built-in product is frozen to.
    let builtin =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(Z2FusionRule, Z2FusionRule).unwrap();
    for ids in [[17, 0, 0], [6, 0, 0], [0, 0, 5]] {
        let [a, b, c] = ids.map(s);
        assert_eq!(rule.try_nsymbol(a, b, c), builtin.try_nsymbol(a, b, c));
    }
}

#[test]
fn checked_generic_r_of_a_codec_admitted_zero_block_is_empty() {
    let rule = CategoryRule::try_new(
        LieGroup::new(GroupId::su(2).unwrap()).unwrap(),
        CompactSu2::<CG>::NEW,
    )
    .unwrap();
    // 1/2 x 1/2 cannot reach 300; the codec admits a zero block unconverted.
    let r = rule.try_r_symbol_generic(s(1), s(1), s(300)).unwrap();
    assert_eq!(r.shape(), (0, 0));
    assert!(r.data().is_empty());
}

// ---------------------------------------------------------------------------
// What the error types report.
// ---------------------------------------------------------------------------

#[test]
fn admission_errors_report_their_cause_and_keep_the_codec_error_as_source() {
    use std::error::Error as _;

    let style = CategoryRule::try_new(
        ClaimsGeneric {
            canonical_unit: true,
        },
        EnumeratedCodec::<MF>::new(),
    )
    .unwrap_err();
    assert_eq!(
        style.to_string(),
        "multiplicity-free admission requires a multiplicity-free category, got Generic fusion"
    );
    assert!(style.source().is_none());

    let unit = CategoryRule::try_new(
        ClaimsGeneric {
            canonical_unit: false,
        },
        EnumeratedCodec::<CG>::new(),
    )
    .unwrap_err();
    assert_eq!(
        unit.to_string(),
        "canonical-unit admission requires a canonical-unit category"
    );
    assert!(unit.source().is_none());

    let codec = CategoryRule::try_new(ForeignUnit, RawU8).unwrap_err();
    assert_eq!(codec.to_string(), "the category codec rejected the unit");
    let source = codec.source().unwrap().downcast_ref::<FusionAlgebraError>();
    assert!(matches!(
        source,
        Some(FusionAlgebraError::CategoryFailure(_))
    ));
}

#[test]
fn adapter_errors_name_their_failure_and_keep_authority_and_storage_sources() {
    use std::error::Error as _;

    // The authority's own error, kept whole by an expert codec.
    let rule = CategoryRule::try_new(Faulty, RawU8).unwrap();
    let error = rule.try_fusion_channels(s(1), s(0)).unwrap_err();
    let provider = adapter_source::<AuthorityFailed>(&error);
    assert_eq!(provider, &CategoryAdapterError::Provider(AuthorityFailed));
    assert_eq!(provider.to_string(), "the fusion category failed");
    assert_eq!(provider.source().unwrap().to_string(), "authority failed");

    // A decoded object outside the category (index 3 unranks to 3).
    let error = rule.try_dual_sector(s(3)).unwrap_err();
    let foreign = adapter_source::<AuthorityFailed>(&error);
    assert_eq!(
        foreign,
        &CategoryAdapterError::ForeignObject { sector: Some(s(3)) }
    );
    assert_eq!(
        foreign.to_string(),
        "SectorId(3) names an object outside the category"
    );
    assert!(foreign.source().is_none());

    // A fusion outcome outside the category, through the default
    // `encode_channels`; a member outcome is encoded.
    let error = rule.try_fusion_channels(s(2), s(0)).unwrap_err();
    let outcome = adapter_source::<AuthorityFailed>(&error);
    assert_eq!(
        outcome,
        &CategoryAdapterError::ForeignObject { sector: None }
    );
    assert_eq!(
        outcome.to_string(),
        "the category produced an object outside itself"
    );
    assert_eq!(
        rule.try_fusion_channels(s(0), s(0)).unwrap().as_slice(),
        &[s(0)]
    );

    let checked = cg(Faulty);
    let error = checked.try_dim_scalar(s(2)).unwrap_err();
    let dimension = adapter_source::<AuthorityFailed>(&error);
    assert_eq!(
        dimension.to_string(),
        "the dimension of SectorId(2) is not a finite positive f64"
    );
    assert!(dimension.source().is_none());

    let error = checked
        .try_f_symbol_generic(s(1), s(1), s(1), s(1), s(1), s(1))
        .unwrap_err();
    let storage = adapter_source::<AuthorityFailed>(&error);
    assert_eq!(
        storage.to_string(),
        "symbol storage disagrees with its fusion multiplicities"
    );
    assert_eq!(
        storage.source().unwrap().downcast_ref::<SymbolShapeError>(),
        Some(&SymbolShapeError {
            actual_len: 1,
            expected_len: Some(0),
        })
    );

    let axes = CategoryAdapterError::<AuthorityFailed>::MalformedSymbolAxes {
        expected: vec![2, 2],
        actual: vec![1, 4],
    };
    assert_eq!(
        axes.to_string(),
        "symbol axes [1, 4] disagree with the fusion multiplicities [2, 2]"
    );
    assert!(axes.source().is_none());

    let rep_s3 = cg(TableCategory::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c").unwrap());
    let error = rep_s3.try_twist_scalar(s(1)).unwrap_err();
    assert_eq!(
        adapter_source::<fusion_symbols::table::TableError>(&error).to_string(),
        "the category has no braiding"
    );
}

#[test]
fn a_category_failure_reports_and_exposes_the_failure_it_keeps() {
    use std::error::Error as _;

    let error = cg(Faulty).try_fusion_channels(s(1), s(0)).unwrap_err();
    assert_eq!(error.to_string(), "fusion category failed");
    let failure = error.source().unwrap();
    // The failure's message and source are the authority's own.
    assert_eq!(failure.to_string(), "authority failed");
    assert!(failure
        .source()
        .unwrap()
        .downcast_ref::<AuthorityFailed>()
        .is_some());
    let FusionAlgebraError::CategoryFailure(failure) = &error else {
        unreachable!();
    };
    // Equality is identity of the kept failure.
    assert_eq!(failure.clone(), *failure);
    assert_ne!(CategoryFailure::new(AuthorityFailed), *failure);
}

// ---------------------------------------------------------------------------
// Enumerated codec boundaries.
// ---------------------------------------------------------------------------

#[test]
fn enumerated_codec_rejects_foreign_objects_and_reports_missing_enumeration() {
    let rule = cg(Faulty);
    // An authority that unranks a foreign object names an invalid sector.
    assert_eq!(rule.try_dual(s(3)), invalid(3));
    // A foreign fusion outcome is a category failure, not a filtered channel.
    let error = rule.try_fusion_channels(s(2), s(0)).unwrap_err();
    assert_eq!(
        adapter_source::<AuthorityFailed>(&error),
        &CategoryAdapterError::ForeignObject { sector: None }
    );

    // Without enumeration, neither the unit nor any id can be numbered.
    let error = CategoryRule::try_new(Watched, EnumeratedCodec::<MF>::new()).unwrap_err();
    let not_enumerable = |error: &FusionAlgebraError| {
        let FusionAlgebraError::CategoryFailure(failure) = error else {
            return false;
        };
        std::error::Error::source(failure)
            .and_then(|source| source.downcast_ref::<EnumerationError>())
            == Some(&EnumerationError::NotEnumerable)
    };
    let CategoryAdmissionError::Codec(error) = &error else {
        panic!("expected a codec error, got {error:?}");
    };
    assert!(not_enumerable(error));
    let codec = EnumeratedCodec::<MF>::new();
    let decoded = CategoryCodec::<Watched>::object(&codec, || &Watched, s(0)).unwrap_err();
    assert!(not_enumerable(&decoded));
}

/// Z3 whose objects or labels are offset from its ids, so that the unit does
/// not round-trip.
#[derive(Debug)]
struct ShiftedZ3 {
    object: u32,
    label: usize,
}

impl CategoryCodec<ZN> for ShiftedZ3 {
    type Sector = usize;
    type Error = FusionAlgebraError;
    type Mode = MF;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _: impl Fn() -> &'a ZN + Copy,
        _: &RuleIdentity,
        label: &usize,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s((label + self.label) % 3))
    }
    fn decode_label<'a>(
        &self,
        _: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(id.id())
    }
    fn object<'a>(
        &self,
        _: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<ZNIrrep, FusionAlgebraError> {
        Ok(ZNIrrep::from_canonical((id.id() as u32 + self.object) % 3, 3).unwrap())
    }
    fn sector_id<'a>(
        &self,
        _: impl Fn() -> &'a ZN + Copy,
        _: &RuleIdentity,
        object: &ZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(s(object.charge() as usize))
    }
    fn map_error(
        &self,
        _: &[SectorId],
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
    }
}

#[test]
fn construction_refuses_a_unit_that_does_not_round_trip() {
    let refused = |codec| {
        let error = CategoryRule::try_new(ZN::new(3).unwrap(), codec).unwrap_err();
        let CategoryAdmissionError::Codec(error) = &error else {
            panic!("expected a codec error, got {error:?}");
        };
        adapter_source::<std::convert::Infallible>(error).clone()
    };
    let unit_mismatch = CategoryAdapterError::ForeignObject { sector: Some(s(0)) };
    // Id 0 decodes to charge 1, not the unit.
    assert_eq!(
        refused(ShiftedZ3 {
            object: 1,
            label: 0
        }),
        unit_mismatch
    );
    // The unit's label encodes to id 1, not 0.
    assert_eq!(
        refused(ShiftedZ3 {
            object: 0,
            label: 1
        }),
        unit_mismatch
    );
    assert!(CategoryRule::try_new(
        ZN::new(3).unwrap(),
        ShiftedZ3 {
            object: 0,
            label: 0
        }
    )
    .is_ok());
}

// ---------------------------------------------------------------------------
// Every provider query answers from the category.
// ---------------------------------------------------------------------------

#[test]
fn a_multiplicity_free_rule_answers_every_query_from_its_category() {
    // SU(2) by doubled spin: 1/2 has dimension 2 and FS phase -1.
    let su2 = || LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let rule = mf(su2());
    assert_eq!(rule.category(), &su2());
    assert_eq!(rule.typed_rule_identity(), FusionRule::rule_identity(&rule));
    assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
    assert_eq!(rule.dual(s(1)), s(1));
    assert_eq!(rule.fusion_channels(s(1), s(1)).as_slice(), &[s(0), s(2)]);
    assert_eq!(rule.nsymbol(s(1), s(1), s(2)), 1);
    assert_eq!(rule.nsymbol(s(1), s(1), s(1)), 0);
    assert!(!rule.has_trivial_associator_gauge());
    assert_eq!(rule.dim_scalar(s(1)), 2.0);
    assert_eq!(rule.inv_dim_scalar(s(1)), 0.5);
    numerics::assert_close("sqrt dim", rule.sqrt_dim_scalar(s(1)), 2f64.sqrt(), 4);
    numerics::assert_close(
        "inv sqrt dim",
        rule.inv_sqrt_dim_scalar(s(1)),
        0.5f64.sqrt(),
        4,
    );
    assert_eq!(rule.frobenius_schur_phase_scalar(s(1)), -1.0);
    assert_eq!(rule.twist_scalar(s(1)), 1.0);

    // ZN's associator is the unit; fermion parity braids fermionically.
    assert!(mf(ZN::new(3).unwrap()).has_trivial_associator_gauge());
    assert_eq!(
        mf(fusion_symbols::fermion::FermionParity).braiding_style(),
        BraidingStyleKind::Fermionic
    );
}

#[test]
fn a_checked_generic_rule_answers_every_query_from_its_category() {
    let rule = cg(LieGroup::new(GroupId::su(2).unwrap()).unwrap());
    assert_eq!(
        CheckedGenericFusion::rule_identity(&rule),
        rule.typed_rule_identity()
    );
    assert_eq!(CheckedGenericFusion::vacuum(&rule), s(0));
    assert_eq!(rule.try_dual(s(3)), Ok(s(3)));
    assert_eq!(
        rule.try_fusion_channels_in_table(s(1), s(2)).unwrap(),
        rule.try_fusion_channels(s(1), s(2)).unwrap()
    );
    assert_eq!(
        rule.try_fusion_channels(s(1), s(2)).unwrap().as_slice(),
        &[s(1), s(3)]
    );
    assert_eq!(rule.try_nsymbol(s(1), s(2), s(3)), Ok(1));
    numerics::assert_close(
        "sqrt dim",
        rule.try_sqrt_dim_scalar(s(2)).unwrap(),
        3f64.sqrt(),
        4,
    );
    numerics::assert_close(
        "inv sqrt dim",
        rule.try_inv_sqrt_dim_scalar(s(2)).unwrap(),
        3f64.sqrt().recip(),
        4,
    );
    assert_eq!(rule.try_frobenius_schur_phase_scalar(s(1)), Ok(-1.0));
    assert_eq!(
        CheckedGenericFusion::sector_order_key(&rule, s(2)),
        SectorOrderKey::position(2)
    );
}

// ---------------------------------------------------------------------------
// The testing-tier A/B composition of a native probe.
// ---------------------------------------------------------------------------

/// A native probe that supplies F and rigidity data but no A/B, so it
/// inherits the testing-only TensorKitSectors composition.
struct ComposedAb(CategoryRule<LieGroup, EnumeratedCodec<CG>>);

impl symtn_sectors::testing::Provider for ComposedAb {}

impl CheckedGenericFusion for ComposedAb {
    type Error = FusionAlgebraError;
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(&self.0)
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.0)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.0.try_dual(sector)
    }
    fn try_fusion_channels(&self, l: SectorId, r: SectorId) -> Result<SectorVec, Self::Error> {
        self.0.try_fusion_channels(l, r)
    }
    fn try_fusion_channels_in_table(
        &self,
        l: SectorId,
        r: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.0.try_fusion_channels_in_table(l, r)
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        CheckedGenericFusion::try_nsymbol(&self.0, a, b, c)
    }
}

impl CheckedGenericRigidSymbols for ComposedAb {
    type Scalar = f64;
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.0.try_sqrt_dim_scalar(sector)
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.0.try_inv_sqrt_dim_scalar(sector)
    }
    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.0.try_frobenius_schur_phase_scalar(sector)
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.0.try_f_symbol_generic(a, b, c, d, e, f)
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.0.try_r_symbol_generic(a, b, c)
    }
}

#[test]
fn the_testing_a_and_b_composition_agrees_with_the_category_authority() {
    // Independent of the composition: fusion-symbols' own A and B, which
    // `CategoryRule` forwards, against the TensorKitSectors F-based formula.
    let authority = cg(LieGroup::new(GroupId::su(2).unwrap()).unwrap());
    let probe = ComposedAb(cg(LieGroup::new(GroupId::su(2).unwrap()).unwrap()));
    for (a, b, c) in [
        (1, 1, 0),
        (1, 1, 2),
        (2, 1, 1),
        (2, 2, 2),
        (3, 2, 1),
        (1, 2, 3),
    ] {
        let (a, b, c) = (s(a), s(b), s(c));
        for (composed, forwarded) in [
            (
                probe.try_a_symbol_generic(a, b, c).unwrap(),
                authority.try_a_symbol_generic(a, b, c).unwrap(),
            ),
            (
                probe.try_b_symbol_generic(a, b, c).unwrap(),
                authority.try_b_symbol_generic(a, b, c).unwrap(),
            ),
        ] {
            assert_eq!(composed.shape(), forwarded.shape());
            numerics::assert_slices_close("A/B", composed.data(), forwarded.data(), 8);
        }
    }
}

// ---------------------------------------------------------------------------
// Identity requests that escape resolution, and identity formatting.
// ---------------------------------------------------------------------------

/// A codec over ZN that asks for the value identity of another category type.
#[derive(Clone, Debug)]
struct MismatchedRequest;

impl CategoryCodec<ZN> for MismatchedRequest {
    type Sector = ZNIrrep;
    type Error = FusionAlgebraError;
    type Mode = MF;
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
    fn rule_identity(&self, _: &ZN) -> Option<RuleIdentity> {
        CategoryCodec::<U1>::rule_identity(&Enumerated::new(), &U1)
    }
    fn map_error(
        &self,
        inputs: &[SectorId],
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        <Enumerated as CategoryCodec<ZN>>::map_error(&Enumerated::new(), inputs, error)
    }
}

#[test]
fn a_request_for_another_category_type_never_aliases() {
    let first = CategoryRule::try_new(ZN::new(3).unwrap(), MismatchedRequest).unwrap();
    let second = CategoryRule::try_new(ZN::new(3).unwrap(), MismatchedRequest).unwrap();
    assert_ne!(
        FusionRule::rule_identity(&first),
        FusionRule::rule_identity(&second)
    );
    assert_eq!(
        FusionRule::rule_identity(&first),
        FusionRule::rule_identity(&first.clone())
    );
}
