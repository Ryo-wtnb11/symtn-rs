//! Variant predicates on a typed `Result`, so a test asserts which error it
//! got and not only that there was one. `InvalidArgument` is stringly: only
//! its reason is matched.

#![allow(dead_code)]

pub trait Rejection {
    /// `InvalidArgument` whose reason contains `reason`.
    fn is_invalid_argument(&self, reason: &str) -> bool;
    /// The operands belong to different runtimes.
    fn is_runtime_mismatch(&self) -> bool;
}

impl<T> Rejection for Result<T, symtn::typed::Error> {
    fn is_invalid_argument(&self, reason: &str) -> bool {
        matches!(self, Err(symtn::typed::Error::InvalidArgument(text)) if text.contains(reason))
    }

    fn is_runtime_mismatch(&self) -> bool {
        matches!(self, Err(symtn::typed::Error::RuntimeMismatch))
    }
}
