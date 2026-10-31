use super::*;

// Canary against `CoreError` regrowing past the clippy
// `result_large_err` threshold: `{Missing,Duplicate}BlockKey` box their
// `BlockKey` payload precisely to keep every `Result<_, CoreError>` return
// pointer-cheap on the hot paths that propagate it with `?`.
#[test]
fn core_error_size_has_not_silently_grown() {
    assert!(std::mem::size_of::<CoreError>() <= 128);
}

#[test]
fn checked_fusion_space_error_exposes_its_typed_source() {
    // What: callers can inspect either the structural or algebraic source
    // through the standard error chain without parsing display text.
    let core = CheckedFusionSpaceError::from(CoreError::DimensionMismatch {
        expected: 1,
        actual: 2,
    });
    assert!(std::error::Error::source(&core)
        .is_some_and(|source| source.downcast_ref::<CoreError>().is_some()));
    let algebra = CheckedFusionSpaceError::from(FusionAlgebraError::U1FusionOverflow {
        left: i32::MAX,
        right: 1,
    });
    assert!(std::error::Error::source(&algebra)
        .is_some_and(|source| source.downcast_ref::<FusionAlgebraError>().is_some()));
}
