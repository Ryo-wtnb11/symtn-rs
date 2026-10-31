//! Facade contracts that both admission modes promise, written once and run
//! in each: multiplicity-free typed over Z2, and checked Generic over the
//! counted toy provider. A cell that only one mode can express (outer
//! multiplicity, injected provider faults, a category-specific oracle) stays
//! in that mode's own facade suite.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorId, SectorVec, TypedSectorAdmission, Z2FusionRule, Z2Irrep,
};
use symtn::typed::{
    Complex64, GenericTensorError, GradedSpace, OperationError, Runtime, SectorSpectrum, TensorMap,
};

use crate::numerics;

include!("../common/checked_only_toy.rs");

/// Multiplicity-free typed mode: Z2, whose charged sector is `ODD`.
struct Mf;

impl Mf {
    fn provider() -> Arc<Z2FusionRule> {
        Arc::new(Z2FusionRule)
    }
    fn unit() -> Z2Irrep {
        Z2Irrep::EVEN
    }
    fn charged() -> Z2Irrep {
        Z2Irrep::ODD
    }
    /// A built-in rule has no query counter; the checked mode observes the
    /// "before provider work" half of a cell.
    fn reset_queries(_: &Z2FusionRule) {}
    fn assert_no_queries(_: &Z2FusionRule) {}
}

/// Checked Generic mode: the toy provider, whose charged sector is `X`.
struct Checked;

impl Checked {
    fn provider() -> Arc<CheckedOnlyToy> {
        Arc::new(CheckedOnlyToy::new_product_probe(0))
    }
    fn unit() -> Label {
        Label::Vacuum
    }
    fn charged() -> Label {
        Label::X
    }
    fn reset_queries(provider: &CheckedOnlyToy) {
        reset_provider_queries(provider);
    }
    fn assert_no_queries(provider: &CheckedOnlyToy) {
        assert_no_provider_queries(provider);
    }
}

/// The facade `Error` inside either mode's error type: multiplicity-free
/// returns it directly, checked Generic wraps it in `Facade`.
trait IntoFacade {
    fn into_facade(self) -> symtn::typed::Error;
}

impl IntoFacade for symtn::typed::Error {
    fn into_facade(self) -> symtn::typed::Error {
        self
    }
}

impl<E: fmt::Debug> IntoFacade for GenericTensorError<E> {
    fn into_facade(self) -> symtn::typed::Error {
        match self {
            GenericTensorError::Facade(error) => error,
            other => panic!("expected a facade error, got {other:?}"),
        }
    }
}

#[track_caller]
fn facade_err<T, E: IntoFacade>(result: Result<T, E>) -> symtn::typed::Error {
    match result {
        Ok(_) => panic!("the call succeeded"),
        Err(error) => error.into_facade(),
    }
}

/// Generates `$name::multiplicity_free` and `$name::checked_generic`, each
/// running `$body` with `$mode` bound to that mode.
macro_rules! in_both_modes {
    ($(#[doc = $doc:literal])* fn $name:ident<$mode:ident>() $body:block) => {
        $(#[doc = $doc])*
        mod $name {
            use super::*;

            #[test]
            fn multiplicity_free() {
                type $mode = crate::admission_modes::Mf;
                let _serial = crate::counting_alloc::serial();
                $body
            }

            #[test]
            fn checked_generic() {
                type $mode = crate::admission_modes::Checked;
                let _serial = crate::counting_alloc::serial();
                $body
            }
        }
    };
}

mod matrix_fn;
