//! What: every facade error wrapper names only its own layer, so a reporter
//! that walks `source()` prints each message of the chain exactly once, and
//! generic code names the per-rule facade error as `FacadeError<R>`.

use std::error::Error as StdError;

use symtn::sector::U1FusionRule;
use symtn::typed::{
    BatchError, CheckedGenericPlanError, CheckedGenericStructureError,
    CheckedGenericTensorProductError, CoreError, Error, FacadeError, FusionAlgebraError,
    GenericTensorError, OperationError,
};

/// The chain as a standard reporter prints it: each `Display`, outermost
/// first.
fn chain(error: &(dyn StdError + 'static)) -> Vec<String> {
    std::iter::successors(Some(error), |&error| error.source())
        .map(ToString::to_string)
        .collect()
}

fn core() -> CoreError {
    CoreError::DimensionMismatch {
        expected: 3,
        actual: 5,
    }
}

/// `error`'s chain shows `root`'s message exactly once, at its end.
fn assert_printed_once(error: &(dyn StdError + 'static), root: &dyn StdError) {
    let messages = chain(error);
    let root = root.to_string();
    let printed = messages.join(": ");
    assert_eq!(printed.matches(&root).count(), 1, "{printed}");
    assert_eq!(messages.last(), Some(&root), "{printed}");
}

#[test]
fn facade_wrappers_print_each_message_once() {
    let operation = OperationError::from(core());
    let algebra = FusionAlgebraError::U1FusionOverflow {
        left: i32::MAX,
        right: 1,
    };
    let cases: Vec<(Error, Box<dyn StdError>)> = vec![
        (Error::Core(Box::new(core())), Box::new(core())),
        (
            Error::Operation(Box::new(operation.clone())),
            Box::new(operation.clone()),
        ),
        (
            Error::FusionAlgebra(Box::new(algebra.clone())),
            Box::new(algebra),
        ),
    ];
    for (error, root) in &cases {
        assert_printed_once(error, root.as_ref());
    }
    assert_eq!(
        chain(&Error::Operation(Box::new(operation))),
        [
            "tensor operation failed",
            "dimension mismatch: expected 3, got 5"
        ]
    );
}

#[test]
fn batch_wrappers_print_each_message_once() {
    let inner = Error::InvalidArgument("bad axis".into());
    let solver = BatchError::Solver {
        block: 2,
        source: inner.clone(),
    };
    assert_printed_once(&solver, &inner);
    assert_eq!(
        solver.to_string(),
        "solver failed on coupled-sector block 2"
    );

    // `Operation` is the eager error itself, as that operation reports it.
    let operation = BatchError::Operation(Error::Operation(Box::new(core().into())));
    assert_eq!(
        chain(&operation),
        chain(&Error::Operation(Box::new(core().into())))
    );
    assert_printed_once(&operation, &core());
}

#[test]
fn generic_wrappers_print_each_message_once() {
    type Generic = GenericTensorError<FusionAlgebraError>;
    let cases: Vec<Generic> = vec![
        GenericTensorError::Facade(Error::Core(Box::new(core()))),
        GenericTensorError::Structure(CheckedGenericStructureError::Core(core())),
        GenericTensorError::Plan(CheckedGenericPlanError::Core(core())),
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::Core(core())),
    ];
    for error in &cases {
        assert_printed_once(error, &core());
    }
    // The checked mode renders a facade failure as the multiplicity-free
    // mode does.
    let facade = Error::InvalidArgument("bad axis".into());
    assert_eq!(chain(&Generic::Facade(facade.clone())), chain(&facade));
}

#[test]
fn facade_error_is_nameable_per_rule() {
    let error: FacadeError<U1FusionRule> = Error::RuleMismatch;
    assert_eq!(error, Error::RuleMismatch);
    #[cfg(feature = "cgc-gen")]
    {
        use symtn::sector::{LieGroupFusionRule, LieGroupFusionRuleError};
        let _: FacadeError<LieGroupFusionRule> =
            GenericTensorError::<LieGroupFusionRuleError>::Facade(Error::RuleMismatch);
    }
}
