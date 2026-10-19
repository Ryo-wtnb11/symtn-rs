/// Every factorization of `$t` (an endomorphism) as `(family, error text)`,
/// for inputs that must all be refused by the shared finite-input stage
/// (#1986), each error reduced by [`refusal`] so both fusion modes' facade
/// errors compare alike.
macro_rules! nonfinite_factor_errors {
    ($t:expr) => {{
        let t = $t;
        let rows: &[usize] = &[0];
        let cols: &[usize] = &[1];
        let text = |result: Result<(), _>| nonfinite_factor_input::refusal(&result.unwrap_err());
        vec![
            ("svd", text(t.svd_vals(rows, cols).map(drop))),
            ("svd", text(t.svd_compact(rows, cols).map(drop))),
            ("svd", text(t.svd_full(rows, cols).map(drop))),
            ("qr", text(t.qr_compact(rows, cols).map(drop))),
            ("qr", text(t.qr_full(rows, cols).map(drop))),
            ("lq", text(t.lq_compact(rows, cols).map(drop))),
            ("lq", text(t.lq_full(rows, cols).map(drop))),
            ("null", text(t.left_null(rows, cols).map(drop))),
            ("null", text(t.right_null(rows, cols).map(drop))),
            ("polar", text(t.left_polar(rows, cols).map(drop))),
            ("polar", text(t.right_polar(rows, cols).map(drop))),
            (
                "eigh",
                text(
                    t.eigh_vals(rows, cols, symtn::typed::HermitianTol::DEFAULT)
                        .map(drop),
                ),
            ),
            (
                "eigh",
                text(
                    t.eigh_full(rows, cols, symtn::typed::HermitianTol::DEFAULT)
                        .map(drop),
                ),
            ),
            ("eig", text(t.eig_vals(rows, cols).map(drop))),
            ("eig", text(t.eig_full(rows, cols).map(drop))),
        ]
    }};
}

/// The operation-layer `InvalidArgument` message on `error`'s `source()`
/// chain, with the error's `Debug` text for a failure message.
pub(crate) fn refusal<E: std::error::Error + 'static>(error: &E) -> (Option<&'static str>, String) {
    let message = std::iter::successors(
        Some(error as &(dyn std::error::Error + 'static)),
        |&error| error.source(),
    )
    .find_map(
        |error| match error.downcast_ref::<symtn::typed::OperationError>() {
            Some(symtn::typed::OperationError::InvalidArgument { message }) => Some(*message),
            _ => None,
        },
    );
    (message, format!("{error:?}"))
}

/// Asserts that every entry of [`nonfinite_factor_errors`] is its family's
/// finite-input refusal, the documented `InvalidArgument` message.
pub(crate) fn assert_finite_input_refusals(
    errors: Vec<(&str, (Option<&'static str>, String))>,
    case: &str,
) {
    for (family, (message, error)) in errors {
        let expected = format!("{family} input components must be finite");
        assert_eq!(message, Some(expected.as_str()), "{case}: {error}");
    }
}
