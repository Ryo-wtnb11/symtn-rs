//! Run with `cargo run -p symtn-production-contract`: workspace tests enable
//! core's `testing` dev-dependency and cannot check the production query path.

use std::cell::{Cell, RefCell};

use fusion_symbols::category::{
    BraidingStyle, EnumerationError, FusionCategory, FusionStyle, RBlock,
};
use symtn_core::*;

thread_local! {
    static QUERIES: RefCell<Vec<CategoryOperation>> = const { RefCell::new(Vec::new()) };
    static REJECT: Cell<Option<CategoryOperation>> = const { Cell::new(None) };
    static FAIL: Cell<Option<CategoryOperation>> = const { Cell::new(None) };
    static SYMBOLS: RefCell<Vec<CategoryOperation>> = const { RefCell::new(Vec::new()) };
    static PHASES: Cell<usize> = const { Cell::new(0) };
}

// A routing probe, not a coherent category: distinct A/B sentinels must survive
// core lowering, whereas an F-derived answer would be one.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Probe;

impl FusionCategory for Probe {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = std::io::Error;

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
    fn frobenius_schur_phase(&self, _: &u8) -> Result<f64, Self::Error> {
        PHASES.set(PHASES.get() + 1);
        Ok(1.0)
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
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> Result<u32, Self::Error> {
        Ok(1)
    }
    fn f_symbol(&self, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        panic!("A/B forwarding must not query F")
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<f64, Self::Error> {
        panic!("planar transpose must not query R")
    }
    fn a_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Self::Error> {
        symbol(CategoryOperation::A, 2.0)
    }
    fn b_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Self::Error> {
        symbol(CategoryOperation::B, 3.0)
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

fn symbol(operation: CategoryOperation, value: f64) -> Result<RBlock<f64>, std::io::Error> {
    SYMBOLS.with_borrow_mut(|calls| calls.push(operation));
    if FAIL.get() == Some(operation) {
        Err(std::io::Error::other(format!("{operation:?}")))
    } else {
        Ok(RBlock::one(value))
    }
}

type Enumerated = EnumeratedCodec<CheckedGenericAdmissionMode>;

struct Codec;

impl CategoryCodec<Probe> for Codec {
    type Sector = u8;
    type Error = FusionAlgebraError;
    type Mode = CheckedGenericAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        c: impl Fn() -> &'a Probe + Copy,
        r: &RuleIdentity,
        a: &u8,
    ) -> Result<SectorId, Self::Error> {
        Enumerated::new().encode_label(c, r, a)
    }
    fn decode_label<'a>(
        &self,
        c: impl Fn() -> &'a Probe + Copy,
        id: SectorId,
    ) -> Result<u8, Self::Error> {
        Enumerated::new().decode_label(c, id)
    }
    fn object<'a>(
        &self,
        c: impl Fn() -> &'a Probe + Copy,
        id: SectorId,
    ) -> Result<u8, Self::Error> {
        Enumerated::new().object(c, id)
    }
    fn sector_id<'a>(
        &self,
        c: impl Fn() -> &'a Probe + Copy,
        r: &RuleIdentity,
        a: &u8,
    ) -> Result<SectorId, Self::Error> {
        Enumerated::new().sector_id(c, r, a)
    }
    fn admit<'a, const N: usize>(
        &self,
        c: impl Fn() -> &'a Probe + Copy,
        r: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[u8; N], Self::Error> {
        // Adapter-internal category N/dual/dimension checks do not cross this
        // seam; the log counts only queries issued by core (plus A/B entry).
        QUERIES.with_borrow_mut(|calls| calls.push(operation));
        if REJECT.get() == Some(operation) {
            return Err(FusionAlgebraError::InvalidSector {
                sector: SectorId::new(99),
            });
        }
        Enumerated::new().admit(c, r, operation, ids)
    }
    fn map_error(
        &self,
        ids: &[SectorId],
        error: CategoryAdapterError<std::io::Error>,
    ) -> Self::Error {
        <Enumerated as CategoryCodec<Probe>>::map_error(&Enumerated::new(), ids, error)
    }
}

fn main() {
    use CategoryOperation::*;
    let rule = CategoryRule::try_new(Probe, Codec).unwrap();
    let unit = SectorId::new(0);
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [unit; 2],
            unit,
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], unit, [], [], []).unwrap(),
    );

    // Rank two avoids recoupling: repartition uses B; a cyclic transpose uses
    // A then B. Dimensions are all one, so the independent coefficients are
    // B = 3 and A*B = 6, rather than the F-derived value one.
    // Fusion/N validate the source vertex. Bend needs Dual and two dimensions.
    // Fold needs Dual, N for its one-hot vector, two dimensions and N for its
    // inverse lift; its dual output adds Dual/phase to the following bend.
    // No N/dual/F/rigidity prefix or suffix is allowed around either A/B entry.
    for (codomain, domain, expected, symbols, queries, phases) in [
        (
            &[0][..],
            &[1][..],
            3.0,
            &[B][..],
            &[Fusion, N, Dual, Dimension, Dimension, B][..],
            0,
        ),
        (
            &[1, 0][..],
            &[][..],
            6.0,
            &[A, B][..],
            &[
                Fusion, N, Dual, N, A, Dimension, Dimension, N, Dual, Dimension, Dimension, Dual, B,
            ][..],
            2,
        ),
    ] {
        QUERIES.with_borrow_mut(Vec::clear);
        SYMBOLS.with_borrow_mut(Vec::clear);
        PHASES.set(0);
        let terms = generic_transpose_tree_pair_checked(&rule, &source, codomain, domain).unwrap();
        assert_eq!(terms.len(), 1);
        assert!((terms[0].1 - expected).abs() < 1e-12);
        SYMBOLS.with_borrow(|calls| assert_eq!(calls, symbols));
        QUERIES.with_borrow(|calls| assert_eq!(calls, queries));
        assert_eq!(PHASES.get(), phases);

        for &operation in symbols {
            // Codec rejection wins even when the category would also fail.
            REJECT.set(Some(operation));
            FAIL.set(Some(operation));
            SYMBOLS.with_borrow_mut(Vec::clear);
            QUERIES.with_borrow_mut(Vec::clear);
            let error =
                generic_transpose_tree_pair_checked(&rule, &source, codomain, domain).unwrap_err();
            assert!(
                matches!(error, CheckedGenericSymbolError::Provider(FusionAlgebraError::InvalidSector { sector }) if sector == SectorId::new(99))
            );
            SYMBOLS.with_borrow(|calls| assert!(!calls.contains(&operation)));
            let prefix = &queries[..=queries
                .iter()
                .position(|&query| query == operation)
                .unwrap()];
            QUERIES.with_borrow(|calls| assert_eq!(calls, prefix));
            REJECT.set(None);
            QUERIES.with_borrow_mut(Vec::clear);
            let error =
                generic_transpose_tree_pair_checked(&rule, &source, codomain, domain).unwrap_err();
            QUERIES.with_borrow(|calls| assert_eq!(calls, prefix));
            let CheckedGenericSymbolError::Provider(FusionAlgebraError::CategoryFailure(failure)) =
                error
            else {
                panic!("provider failure lost its type: {error:?}");
            };
            let source = std::error::Error::source(&failure).unwrap();
            assert_eq!(
                source.downcast_ref::<std::io::Error>().unwrap().to_string(),
                format!("{operation:?}")
            );
            FAIL.set(None);
        }
    }
    println!("PRODUCTION_CONTRACT_OK");
}
