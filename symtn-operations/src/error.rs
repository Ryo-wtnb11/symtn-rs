use core::fmt;

use symtn_core::{BlockKey, BraidingStyleKind, CoreError, FusionAlgebraError, FusionStyleKind};
use symtn_dense::DenseError;

use crate::TreeTransformOperation;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OperationError {
    Core(CoreError),
    Dense(DenseError),
    /// A finite encoded fusion algebra cannot represent a lowered layout.
    FusionAlgebra(Box<FusionAlgebraError>),
    BlockIndexOutOfBounds {
        tensor: &'static str,
        index: usize,
        count: usize,
    },
    BlockCountMismatch {
        dst: usize,
        src: usize,
    },
    CoefficientCountMismatch {
        expected: usize,
        actual: usize,
    },
    ContractAxisCountMismatch {
        lhs: usize,
        rhs: usize,
    },
    TraceAxisCountMismatch {
        lhs: usize,
        rhs: usize,
    },
    DuplicateTransformDestination {
        dst_block: usize,
    },
    DuplicateTreeTransformKey {
        tensor: &'static str,
        index: usize,
    },
    ElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    ElementCountOverflow,
    EmptyTransformBlock,
    ExpectedFusionTreeBlock {
        tensor: &'static str,
        index: usize,
    },
    ExpectedAllCodomainFusionTree {
        index: usize,
    },
    InvalidPermutation {
        axes: Vec<usize>,
        rank: usize,
    },
    InvalidAxisSet {
        tensor: &'static str,
        axes: Vec<usize>,
        rank: usize,
    },
    InvalidArgument {
        message: &'static str,
    },
    FusionTreeGroupMismatch {
        tensor: &'static str,
        index: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    StructureMismatch {
        tensor: &'static str,
    },
    StructureRankMismatch {
        expected: usize,
        actual: usize,
    },
    /// A tree transform (permute, braid or transpose) rejected the rule's
    /// fusion style at its admission.
    ///
    /// Each layer reports its own fusion-style error (#1950): every
    /// tree-transform entry of `symtn-tensors` and `symtn-operations`,
    /// multiplicity-free, Unique-only or checked Generic, reports this
    /// operation-carrying variant, while `symtn-core` APIs report
    /// [`CoreError::UnsupportedFusionStyle`]. Operations without a
    /// [`TreeTransformOperation`] (contraction admission and checked-Generic
    /// space construction) pass the core provider requirement through as
    /// [`Self::Core`], as does a provider whose style changes after
    /// admission. Why not one variant for both layers: the core error cannot
    /// name the operation, and wrapping it here would drop that context
    /// from every tree-transform caller.
    UnsupportedFusionStyle {
        operation: Box<TreeTransformOperation>,
        style: FusionStyleKind,
    },
    UnsupportedBraidingStyle {
        operation: Box<TreeTransformOperation>,
        style: BraidingStyleKind,
    },
    UnsupportedTreeTransformScope {
        operation: Box<TreeTransformOperation>,
        message: &'static str,
    },
    UnsupportedTensorContractScope {
        message: &'static str,
    },
    /// A hom-space relation the operation requires does not hold (TensorKit
    /// `SpaceMismatch`): an endomorphism or an isomorphism, matching
    /// codomains, or an output or factor space that disagrees with its
    /// source. A misuse, unlike [`Self::UnsupportedTensorContractScope`].
    SpaceMismatch {
        message: &'static str,
    },
    /// A tree transform the device executor cannot replay: a recoupling
    /// (Multi) block, a destination mode outside its capability, or a layout
    /// the device region primitive cannot express. The host executor replays
    /// all of them, so this is a placement boundary, never a compile defect.
    UnsupportedDeviceTreeTransform {
        message: &'static str,
    },
    MissingBlockKey {
        key: Box<BlockKey>,
    },
    ShapeMismatch {
        dst: Vec<usize>,
        src: Vec<usize>,
    },
    StrideOverflow {
        value: usize,
    },
    OffsetOverflow {
        value: usize,
    },
    StridedKernel {
        message: String,
    },
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => err.fmt(f),
            Self::Dense(err) => err.fmt(f),
            Self::FusionAlgebra(_) => f.write_str(
                "the fusion algebra cannot represent the requested dual or fusion output",
            ),
            Self::BlockIndexOutOfBounds {
                tensor,
                index,
                count,
            } => {
                write!(
                    f,
                    "{tensor} block index {index} is out of bounds for {count} blocks"
                )
            }
            Self::BlockCountMismatch { dst, src } => {
                write!(f, "block count mismatch: dst {dst}, src {src}")
            }
            Self::CoefficientCountMismatch { expected, actual } => {
                write!(
                    f,
                    "coefficient count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ContractAxisCountMismatch { lhs, rhs } => {
                write!(f, "contracting axis count mismatch: lhs {lhs}, rhs {rhs}")
            }
            Self::TraceAxisCountMismatch { lhs, rhs } => {
                write!(f, "trace axis count mismatch: lhs {lhs}, rhs {rhs}")
            }
            Self::DuplicateTransformDestination { dst_block } => {
                write!(
                    f,
                    "tree transform destination block {dst_block} appears in more than one block"
                )
            }
            Self::DuplicateTreeTransformKey { tensor, index } => {
                write!(
                    f,
                    "{tensor} fusion-tree key at index {index} duplicates an earlier key"
                )
            }
            Self::ElementCountMismatch { expected, actual } => {
                write!(
                    f,
                    "element count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ElementCountOverflow => write!(f, "element count overflow"),
            Self::EmptyTransformBlock => {
                write!(f, "tree transform block has no source or destination")
            }
            Self::ExpectedFusionTreeBlock { tensor, index } => {
                write!(f, "{tensor} block {index} is not a fusion-tree block")
            }
            Self::ExpectedAllCodomainFusionTree { index } => {
                write!(
                    f,
                    "source fusion-tree block {index} is not an all-codomain tree"
                )
            }
            Self::InvalidPermutation { axes, rank } => {
                write!(f, "invalid axis permutation {axes:?} for rank {rank}")
            }
            Self::InvalidAxisSet { tensor, axes, rank } => {
                write!(f, "invalid {tensor} axis set {axes:?} for rank {rank}")
            }
            Self::InvalidArgument { message } => f.write_str(message),
            Self::FusionTreeGroupMismatch { tensor, index } => {
                write!(
                    f,
                    "{tensor} block {index} does not match the fusion-tree group"
                )
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "rank mismatch: expected {expected}, got {actual}")
            }
            Self::StructureMismatch { tensor } => {
                write!(
                    f,
                    "{tensor} tensor structure does not match compiled structure"
                )
            }
            Self::StructureRankMismatch { expected, actual } => {
                write!(
                    f,
                    "block structure rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::UnsupportedFusionStyle { operation, style } => {
                write!(
                    f,
                    "unsupported fusion style {style:?} for tree transform operation {operation:?}"
                )
            }
            Self::UnsupportedBraidingStyle { operation, style } => {
                write!(
                    f,
                    "unsupported braiding style {style:?} for tree transform operation {operation:?}"
                )
            }
            Self::UnsupportedTreeTransformScope { operation, message } => {
                write!(
                    f,
                    "unsupported tree transform scope for operation {operation:?}: {message}"
                )
            }
            Self::UnsupportedTensorContractScope { message } => {
                write!(f, "unsupported tensor contraction scope: {message}")
            }
            Self::SpaceMismatch { message } => write!(f, "space mismatch: {message}"),
            Self::UnsupportedDeviceTreeTransform { message } => {
                write!(f, "unsupported device tree transform: {message}")
            }
            Self::MissingBlockKey { key } => {
                write!(f, "missing matching block for key {key:?}")
            }
            Self::ShapeMismatch { dst, src } => {
                write!(f, "shape mismatch: dst {dst:?}, src {src:?}")
            }
            Self::StrideOverflow { value } => {
                write!(f, "stride {value} does not fit in strided-rs isize")
            }
            Self::OffsetOverflow { value } => {
                write!(f, "offset {value} does not fit in strided-rs isize")
            }
            Self::StridedKernel { message } => write!(f, "strided kernel error: {message}"),
        }
    }
}

impl std::error::Error for OperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FusionAlgebra(err) => Some(err.as_ref()),
            _ => None,
        }
    }
}

/// Core's axis-list rejections become this crate's own variants, so a
/// permutation misuse is always [`OperationError::InvalidPermutation`] and an
/// axis-subset misuse [`OperationError::InvalidAxisSet`], whichever layer
/// validated the list (#1989).
impl From<CoreError> for OperationError {
    fn from(value: CoreError) -> Self {
        match value {
            CoreError::InvalidPermutation { permutation, rank } => Self::InvalidPermutation {
                axes: permutation,
                rank,
            },
            CoreError::InvalidAxisSet { tensor, axes, rank } => {
                Self::InvalidAxisSet { tensor, axes, rank }
            }
            other => Self::Core(other),
        }
    }
}

impl OperationError {
    /// Wraps a core error without collapsing its context; shared with the
    /// matrix-algebra crate.
    pub fn from_core_preserving_context(value: CoreError) -> Self {
        match value {
            CoreError::MissingBlockKey { key } => Self::MissingBlockKey { key },
            other => Self::from(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Canary against `OperationError` regrowing past the clippy
    // `result_large_err` threshold: `MissingBlockKey` boxes `BlockKey` and the
    // `TreeTransformOperation` payload on `Unsupported{FusionStyle,
    // BraidingStyle,TreeTransformScope}` is boxed, keeping every
    // `Result<_, OperationError>` return pointer-cheap on the hot paths that
    // propagate it with `?`.
    #[test]
    fn operation_error_size_has_not_silently_grown() {
        assert!(std::mem::size_of::<OperationError>() <= 128);
    }
}
