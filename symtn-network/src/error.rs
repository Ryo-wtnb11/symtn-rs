use std::fmt::{Display, Formatter};

use symtn::expert::SectorLeg;
use symtn::sector::RuleIdentity;
use symtn::sector::SectorId;

use crate::labels::{TemporaryLabel, TensorAxis};
use crate::slice::SliceKind;

/// Validation failures for coefficient-free symmetric slicing descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceError {
    EmptyRange {
        at: usize,
    },
    ReversedRange {
        start: usize,
        end: usize,
    },
    UnknownLabel(TemporaryLabel),
    DuplicateLabel(TemporaryLabel),
    InvalidAuthority {
        label: TemporaryLabel,
        expected: TensorAxis,
        actual: TensorAxis,
    },
    UnknownSector {
        label: TemporaryLabel,
        sector: SectorId,
    },
    RangeOutOfBounds {
        label: TemporaryLabel,
        sector: SectorId,
        start: usize,
        end: usize,
        degeneracy: usize,
    },
    OverlappingRanges {
        label: TemporaryLabel,
        sector: SectorId,
        previous_end: usize,
        next_start: usize,
    },
    IncompleteCoverage {
        label: TemporaryLabel,
        sector: SectorId,
        expected_start: usize,
        actual_start: usize,
    },
    SliceCountOverflow,
}

pub type SliceResult<T> = std::result::Result<T, SliceError>;

/// Failure while lowering or binding a symmetric sliced plan.
#[derive(Debug)]
pub enum SymmetricSliceLowerError<E> {
    Tensor(E),
    InvalidPlan(ContractError),
    InvalidSlice(SliceError),
    RuleMismatch {
        expected: RuleIdentity,
        actual: RuleIdentity,
    },
    AuthorityLegMismatch {
        label: TemporaryLabel,
        authority: TensorAxis,
        expected: SectorLeg,
        actual: SectorLeg,
    },
    MissingAuthority {
        label: TemporaryLabel,
        authority: TensorAxis,
    },
}

/// Failure while binding or executing a symmetric sliced plan.
#[derive(Debug)]
pub enum SymmetricSliceExecutionError<E> {
    Bind(SymmetricSliceLowerError<E>),
    Tensor(E),
    WorkspaceLimitExceeded { limit: usize, required: usize },
    WorkspaceArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContractError {
    MarkerCountMismatch {
        operands: usize,
        conj: usize,
        codomain_splits: usize,
    },
    OutputCodomainRankOutOfRange {
        codomain_rank: usize,
        output_rank: usize,
    },
    EmptyInput,
    DuplicateOutputLabel(String),
    UnknownOutputLabel(String),
    TensorCountMismatch {
        expected: usize,
        actual: usize,
    },
    RankMismatch {
        tensor: usize,
        expected: usize,
        actual: usize,
    },
    DimensionMismatch {
        label: String,
        expected: usize,
        actual: usize,
    },
    InvalidTensorId {
        tensor: usize,
        tensor_count: usize,
    },
    InvalidContractionPlan(String),
    NotEnoughTensors,
    /// A label occurs more than once WITHIN a single operand (a diagonal /
    /// trace-on-one-tensor, e.g. `aa->a`). Not supported by the pairwise executor.
    UnsupportedDiagonal {
        label: String,
        tensor: usize,
    },
    /// A label is shared by more than two operands (a hyperedge, e.g.
    /// `a,a,a->`). The pairwise executor contracts a label across exactly two
    /// operands; >2 is unsupported.
    UnsupportedHyperedge {
        label: String,
        operand_count: usize,
    },
    /// An OUTPUT label is carried by more than one input operand (a
    /// batch/hadamard index, e.g. `ab,ab->ab`). Not supported.
    UnsupportedBatchLabel {
        label: String,
        operand_count: usize,
    },
    /// A contracted (non-output) label occurs on only ONE operand, so it would
    /// need a single-operand reduction/sum (e.g. `a->` or `ab->a`). Not supported.
    UnsupportedReduction {
        label: String,
    },
    UnsupportedPlannerProjection {
        label: String,
    },
    UnknownPlannerSliceLabel {
        label: String,
    },
    PlannerSliceKindMismatch {
        label: String,
        expected: SliceKind,
        actual: SliceKind,
    },
}

pub type Result<T> = std::result::Result<T, ContractError>;

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractError::MarkerCountMismatch {
                operands,
                conj,
                codomain_splits,
            } => write!(
                f,
                "operand marker lists must match operand count {operands}: got {conj} conj markers and {codomain_splits} codomain splits"
            ),
            ContractError::OutputCodomainRankOutOfRange {
                codomain_rank,
                output_rank,
            } => write!(
                f,
                "output codomain rank {codomain_rank} exceeds output rank {output_rank}"
            ),
            ContractError::EmptyInput => write!(f, "contraction equation has an empty input"),
            ContractError::DuplicateOutputLabel(label) => {
                write!(f, "duplicate output label `{label}`")
            }
            ContractError::UnknownOutputLabel(label) => {
                write!(f, "output label `{label}` does not occur in inputs")
            }
            ContractError::TensorCountMismatch { expected, actual } => {
                write!(f, "expected {expected} tensor infos, got {actual}")
            }
            ContractError::RankMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "rank mismatch for tensor {tensor}: expected {expected}, got {actual}"
            ),
            ContractError::DimensionMismatch {
                label,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch for label `{label}`: expected {expected}, got {actual}"
            ),
            ContractError::InvalidTensorId {
                tensor,
                tensor_count,
            } => write!(
                f,
                "invalid tensor id {tensor}; network has {tensor_count} tensors"
            ),
            ContractError::InvalidContractionPlan(message) => {
                write!(f, "invalid contraction plan: {message}")
            }
            ContractError::NotEnoughTensors => write!(f, "need at least two active tensors"),
            ContractError::UnsupportedDiagonal { label, tensor } => write!(
                f,
                "einsum: repeated label `{label}` within one operand (tensor {tensor}) \
                 is a diagonal/trace — not supported"
            ),
            ContractError::UnsupportedHyperedge {
                label,
                operand_count,
            } => write!(
                f,
                "einsum: label `{label}` appears on {operand_count} operands (>2, a \
                 hyperedge) — not supported"
            ),
            ContractError::UnsupportedBatchLabel {
                label,
                operand_count,
            } => write!(
                f,
                "einsum: output label `{label}` is shared by {operand_count} inputs \
                 (batch/hadamard) — not supported"
            ),
            ContractError::UnsupportedReduction { label } => write!(
                f,
                "einsum: contracted label `{label}` occurs on a single operand \
                 (single-operand reduction) — not supported"
            ),
            ContractError::UnsupportedPlannerProjection { label } => write!(
                f,
                "planner projected sliced index `{label}`; only complete index slicing is supported"
            ),
            ContractError::UnknownPlannerSliceLabel { label } => {
                write!(f, "planner returned unknown sliced index `{label}`")
            }
            ContractError::PlannerSliceKindMismatch {
                label,
                expected,
                actual,
            } => write!(
                f,
                "planner sliced index `{label}` kind mismatch: expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for ContractError {}

impl Display for SliceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRange { at } => write!(f, "empty degeneracy range at {at}"),
            Self::ReversedRange { start, end } => {
                write!(f, "reversed degeneracy range [{start}, {end})")
            }
            Self::UnknownLabel(label) => write!(f, "unknown sliced label `{label}`"),
            Self::DuplicateLabel(label) => write!(f, "duplicate sliced label `{label}`"),
            Self::InvalidAuthority {
                label,
                expected,
                actual,
            } => write!(
                f,
                "invalid authority for `{label}`: expected {expected:?}, got {actual:?}"
            ),
            Self::UnknownSector { label, sector } => {
                write!(f, "unknown sector {sector:?} for sliced label `{label}`")
            }
            Self::RangeOutOfBounds {
                label,
                sector,
                start,
                end,
                degeneracy,
            } => write!(
                f,
                "range [{start}, {end}) for `{label}` sector {sector:?} exceeds degeneracy {degeneracy}"
            ),
            Self::OverlappingRanges {
                label,
                sector,
                previous_end,
                next_start,
            } => write!(
                f,
                "ranges for `{label}` sector {sector:?} overlap at {next_start} before {previous_end}"
            ),
            Self::IncompleteCoverage {
                label,
                sector,
                expected_start,
                actual_start,
            } => write!(
                f,
                "incomplete coverage for `{label}` sector {sector:?}: expected next boundary {expected_start}, got {actual_start}"
            ),
            Self::SliceCountOverflow => write!(f, "symmetric slice count exceeds u128"),
        }
    }
}

impl std::error::Error for SliceError {}

impl<E: Display> Display for SymmetricSliceLowerError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tensor(_) => f.write_str("typed network lowering failed"),
            Self::InvalidPlan(_) => f.write_str("invalid sliced contraction plan"),
            Self::InvalidSlice(_) => f.write_str("invalid symmetric slice"),
            Self::RuleMismatch { expected, actual } => write!(
                f,
                "symmetric slice rule mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::AuthorityLegMismatch {
                label,
                authority,
                expected,
                actual,
            } => write!(
                f,
                "symmetric slice authority leg mismatch for `{label}` at {authority:?}: expected {expected:?}, got {actual:?}"
            ),
            Self::MissingAuthority { label, authority } => write!(
                f,
                "symmetric slice authority {authority:?} for `{label}` has no actual effective leg"
            ),
        }
    }
}

impl<E> std::error::Error for SymmetricSliceLowerError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tensor(error) => Some(error),
            Self::InvalidPlan(error) => Some(error),
            Self::InvalidSlice(error) => Some(error),
            Self::RuleMismatch { .. }
            | Self::AuthorityLegMismatch { .. }
            | Self::MissingAuthority { .. } => None,
        }
    }
}

impl<E: Display> Display for SymmetricSliceExecutionError<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind(_) => formatter.write_str("symmetric slice binding failed"),
            Self::Tensor(_) => formatter.write_str("symmetric slice execution failed"),
            Self::WorkspaceLimitExceeded { limit, required } => write!(
                formatter,
                "symmetric sliced execution observed {required} network-owned payload bytes, measured ceiling is {limit}"
            ),
            Self::WorkspaceArithmeticOverflow => {
                write!(formatter, "symmetric sliced workspace byte count overflowed")
            }
        }
    }
}

impl<E> std::error::Error for SymmetricSliceExecutionError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind(error) => Some(error),
            Self::Tensor(error) => Some(error),
            Self::WorkspaceLimitExceeded { .. } | Self::WorkspaceArithmeticOverflow => None,
        }
    }
}
