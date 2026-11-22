//! What: the network's wrapping errors name only their own layer, so a
//! reporter that walks `source()` reaches the tensor cause and prints it
//! exactly once.

use std::error::Error as StdError;

use symtn::typed::{CoreError, Error};
use symtn_network::{SymmetricSliceExecutionError, SymmetricSliceLowerError};

fn chain(error: &(dyn StdError + 'static)) -> Vec<String> {
    std::iter::successors(Some(error), |&error| error.source())
        .map(ToString::to_string)
        .collect()
}

fn tensor_error() -> Error {
    Error::Operation(Box::new(
        CoreError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
        .into(),
    ))
}

const ROOT: &str = "dimension mismatch: expected 1, got 2";

#[test]
fn symmetric_slice_errors_print_the_tensor_cause_once() {
    let cases: [(SymmetricSliceExecutionError<Error>, &[&str]); 3] = [
        (
            SymmetricSliceExecutionError::Tensor(tensor_error()),
            &[
                "symmetric slice execution failed",
                "tensor operation failed",
                "invalid tensor structure",
                ROOT,
            ],
        ),
        (
            SymmetricSliceExecutionError::Bind(SymmetricSliceLowerError::Tensor(tensor_error())),
            &[
                "symmetric slice binding failed",
                "typed network lowering failed",
                "tensor operation failed",
                "invalid tensor structure",
                ROOT,
            ],
        ),
        (
            SymmetricSliceExecutionError::WorkspaceArithmeticOverflow,
            &["symmetric sliced workspace byte count overflowed"],
        ),
    ];
    for (error, expected) in &cases {
        // Exact chains: the root cause appears once, at the end.
        assert_eq!(chain(error), *expected);
    }
}
