use smallvec::SmallVec;

use crate::OperationError;

/// Rank-sized axis list kept on the stack up to rank 8 (PEPS/MPS ranks), so
/// per-call axis arithmetic does not allocate.
pub type AxisVec = SmallVec<[usize; 8]>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputAxisOrder<'a> {
    Identity,
    Axes(&'a [usize]),
}

impl<'a> OutputAxisOrder<'a> {
    #[inline]
    pub fn identity() -> Self {
        Self::Identity
    }

    #[inline]
    pub fn from_axes(axes: &'a [usize]) -> Self {
        Self::Axes(axes)
    }
}

/// Full index lowering for a pairwise tensor contraction.
///
/// TensorKit / TensorOperations.jl correspondence:
/// `pA = (open axes of lhs, contracted axes of lhs)`,
/// `pB = (contracted axes of rhs, open axes of rhs)`,
/// `pAB = output axis order` (here [`OutputAxisOrder`]), plus the
/// `conjA` / `conjB` conjugation flags (`lhs_conjugate` / `rhs_conjugate`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorContractSpec<'a> {
    lhs_contracting_axes: &'a [usize],
    rhs_contracting_axes: &'a [usize],
    output_permutation: OutputAxisOrder<'a>,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl<'a> TensorContractSpec<'a> {
    pub fn new(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        output_permutation: OutputAxisOrder<'a>,
    ) -> Self {
        Self::new_with_conjugation(
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_permutation,
            false,
            false,
        )
    }

    pub fn new_with_conjugation(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        output_permutation: OutputAxisOrder<'a>,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_permutation,
            lhs_conjugate,
            rhs_conjugate,
        }
    }

    /// Contract the given axes with the default output order (`pAB` omitted):
    /// lhs open axes in original order, then rhs open axes in original order.
    pub fn with_default_output_order(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
    ) -> Self {
        Self::new(
            lhs_contracting_axes,
            rhs_contracting_axes,
            OutputAxisOrder::identity(),
        )
    }

    pub fn with_default_output_order_and_conjugation(
        lhs_contracting_axes: &'a [usize],
        rhs_contracting_axes: &'a [usize],
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self::new_with_conjugation(
            lhs_contracting_axes,
            rhs_contracting_axes,
            OutputAxisOrder::identity(),
            lhs_conjugate,
            rhs_conjugate,
        )
    }

    #[inline]
    pub fn lhs_contracting_axes(&self) -> &'a [usize] {
        self.lhs_contracting_axes
    }

    #[inline]
    pub fn rhs_contracting_axes(&self) -> &'a [usize] {
        self.rhs_contracting_axes
    }

    #[inline]
    pub fn output_permutation(&self) -> OutputAxisOrder<'a> {
        self.output_permutation
    }

    #[inline]
    pub fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[inline]
    pub fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TensorContractSpecOwned {
    lhs_contracting_axes: AxisVec,
    rhs_contracting_axes: AxisVec,
    output_axes: AxisVec,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
}

impl TensorContractSpecOwned {
    pub fn new(
        lhs_contracting_axes: Vec<usize>,
        rhs_contracting_axes: Vec<usize>,
        output_axes: Vec<usize>,
    ) -> Self {
        Self::new_with_conjugation(
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            false,
            false,
        )
    }

    pub fn new_with_conjugation(
        lhs_contracting_axes: Vec<usize>,
        rhs_contracting_axes: Vec<usize>,
        output_axes: Vec<usize>,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self::from_axis_vecs(
            AxisVec::from_vec(lhs_contracting_axes),
            AxisVec::from_vec(rhs_contracting_axes),
            AxisVec::from_vec(output_axes),
            lhs_conjugate,
            rhs_conjugate,
        )
    }

    /// Allocation-free constructor for rank-sized axis lists.
    pub fn from_axis_vecs(
        lhs_contracting_axes: AxisVec,
        rhs_contracting_axes: AxisVec,
        output_axes: AxisVec,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
    ) -> Self {
        Self {
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            lhs_conjugate,
            rhs_conjugate,
        }
    }

    #[inline]
    pub fn as_spec(&self) -> TensorContractSpec<'_> {
        TensorContractSpec::new_with_conjugation(
            self.lhs_contracting_axes.as_slice(),
            self.rhs_contracting_axes.as_slice(),
            OutputAxisOrder::from_axes(self.output_axes.as_slice()),
            self.lhs_conjugate,
            self.rhs_conjugate,
        )
    }

    #[inline]
    pub fn lhs_contracting_axes(&self) -> &[usize] {
        self.lhs_contracting_axes.as_slice()
    }

    #[inline]
    pub fn rhs_contracting_axes(&self) -> &[usize] {
        self.rhs_contracting_axes.as_slice()
    }

    #[inline]
    pub fn output_axes(&self) -> &[usize] {
        self.output_axes.as_slice()
    }

    #[inline]
    pub fn lhs_conjugate(&self) -> bool {
        self.lhs_conjugate
    }

    #[inline]
    pub fn rhs_conjugate(&self) -> bool {
        self.rhs_conjugate
    }
}

pub(crate) fn validate_permutation(axes: &[usize], rank: usize) -> Result<(), OperationError> {
    symtn_core::axes::validate_permutation(axes, rank).map_err(|_| {
        OperationError::InvalidPermutation {
            axes: axes.to_vec(),
            rank,
        }
    })
}

/// The axes of `permutation` over `rank` axes, without a heap allocation up
/// to rank 8.
pub fn permutation_axes_inline(
    permutation: OutputAxisOrder<'_>,
    rank: usize,
) -> Result<AxisVec, OperationError> {
    match permutation {
        OutputAxisOrder::Identity => Ok((0..rank).collect()),
        OutputAxisOrder::Axes(axes) => {
            validate_permutation(axes, rank)?;
            Ok(AxisVec::from_slice(axes))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorTraceAxisSpec<'a> {
    output_axes: &'a [usize],
    trace_lhs_axes: &'a [usize],
    trace_rhs_axes: &'a [usize],
    source_conjugate: bool,
}

impl<'a> TensorTraceAxisSpec<'a> {
    pub fn new(
        output_axes: &'a [usize],
        trace_lhs_axes: &'a [usize],
        trace_rhs_axes: &'a [usize],
    ) -> Self {
        Self::new_with_conjugation(output_axes, trace_lhs_axes, trace_rhs_axes, false)
    }

    pub fn new_with_conjugation(
        output_axes: &'a [usize],
        trace_lhs_axes: &'a [usize],
        trace_rhs_axes: &'a [usize],
        source_conjugate: bool,
    ) -> Self {
        Self {
            output_axes,
            trace_lhs_axes,
            trace_rhs_axes,
            source_conjugate,
        }
    }

    #[inline]
    pub fn output_axes(&self) -> &'a [usize] {
        self.output_axes
    }

    #[inline]
    pub fn trace_lhs_axes(&self) -> &'a [usize] {
        self.trace_lhs_axes
    }

    #[inline]
    pub fn trace_rhs_axes(&self) -> &'a [usize] {
        self.trace_rhs_axes
    }

    #[inline]
    pub fn source_conjugate(&self) -> bool {
        self.source_conjugate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permutation_axes(
        permutation: OutputAxisOrder<'_>,
        rank: usize,
    ) -> Result<Vec<usize>, OperationError> {
        permutation_axes_inline(permutation, rank).map(AxisVec::into_vec)
    }

    fn vec_permutation_axes(
        permutation: OutputAxisOrder<'_>,
        rank: usize,
    ) -> Result<Vec<usize>, OperationError> {
        match permutation {
            OutputAxisOrder::Identity => Ok((0..rank).collect()),
            OutputAxisOrder::Axes(axes) => {
                if axes.len() != rank {
                    return Err(OperationError::InvalidPermutation {
                        axes: axes.to_vec(),
                        rank,
                    });
                }
                let mut seen = vec![false; rank];
                for &axis in axes {
                    if axis >= rank || seen[axis] {
                        return Err(OperationError::InvalidPermutation {
                            axes: axes.to_vec(),
                            rank,
                        });
                    }
                    seen[axis] = true;
                }
                Ok(axes.to_vec())
            }
        }
    }

    #[test]
    fn permutation_axes_identity_empty_inline_and_spilled_rank() {
        // rank <= 8 stays in inline SmallVec storage; rank > 8 spills to the heap.
        for rank in [0, 4, 12] {
            let axes = permutation_axes(OutputAxisOrder::identity(), rank).unwrap();
            assert_eq!(axes, (0..rank).collect::<Vec<_>>());
        }
    }

    #[test]
    fn permutation_axes_explicit_inline_and_spilled_round_trip() {
        let inline = vec![3usize, 1, 0, 2];
        let spilled: Vec<usize> = (0..12).rev().collect();
        for requested in [inline, spilled] {
            let rank = requested.len();
            let axes = permutation_axes(OutputAxisOrder::from_axes(&requested), rank).unwrap();
            assert_eq!(axes, requested);
            assert_eq!(
                axes,
                vec_permutation_axes(OutputAxisOrder::from_axes(&requested), rank).unwrap()
            );
        }
    }

    #[test]
    fn permutation_axes_wrong_length_errors() {
        let requested = [0usize, 1];
        let err = permutation_axes(OutputAxisOrder::from_axes(&requested), 3).unwrap_err();
        match err {
            OperationError::InvalidPermutation { axes, rank } => {
                assert_eq!(axes, requested.to_vec());
                assert_eq!(rank, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn permutation_axes_out_of_range_errors() {
        let requested = [0usize, 1, 5];
        let err = permutation_axes(OutputAxisOrder::from_axes(&requested), 3).unwrap_err();
        assert!(matches!(err, OperationError::InvalidPermutation { .. }));
    }

    #[test]
    fn permutation_axes_duplicate_errors() {
        let requested = [0usize, 1, 1];
        let err = permutation_axes(OutputAxisOrder::from_axes(&requested), 3).unwrap_err();
        assert!(matches!(err, OperationError::InvalidPermutation { .. }));
    }

    #[test]
    fn permutation_axes_inline_matches_vec_variant_on_error() {
        let requested = [2usize, 2, 0];
        let inline_err =
            permutation_axes_inline(OutputAxisOrder::from_axes(&requested), 3).unwrap_err();
        let vec_err = vec_permutation_axes(OutputAxisOrder::from_axes(&requested), 3).unwrap_err();
        assert_eq!(inline_err, vec_err);
    }

    #[test]
    fn from_axis_vecs_inline_round_trips_new() {
        let lhs = vec![0usize, 1];
        let rhs = vec![1usize, 0];
        let out = vec![2usize, 3];
        let via_new = TensorContractSpecOwned::new(lhs.clone(), rhs.clone(), out.clone());
        let via_axis_vecs = TensorContractSpecOwned::from_axis_vecs(
            AxisVec::from_vec(lhs.clone()),
            AxisVec::from_vec(rhs.clone()),
            AxisVec::from_vec(out.clone()),
            false,
            false,
        );
        assert_eq!(via_new, via_axis_vecs);
        assert_eq!(via_new.lhs_contracting_axes(), lhs.as_slice());
        assert_eq!(via_new.rhs_contracting_axes(), rhs.as_slice());
        assert_eq!(via_new.output_axes(), out.as_slice());
        assert!(!via_new.lhs_conjugate());
        assert!(!via_new.rhs_conjugate());

        let conjugated = TensorContractSpecOwned::new_with_conjugation(lhs, rhs, out, true, false);
        assert!(conjugated.lhs_conjugate());
        assert!(!conjugated.rhs_conjugate());
        assert_eq!(
            conjugated.output_axes(),
            via_axis_vecs.output_axes(),
            "conjugation flags leave the axes unchanged"
        );
    }

    #[test]
    fn from_axis_vecs_empty_axes() {
        let spec = TensorContractSpecOwned::from_axis_vecs(
            AxisVec::new(),
            AxisVec::new(),
            AxisVec::new(),
            false,
            false,
        );
        assert!(spec.lhs_contracting_axes().is_empty());
        assert!(spec.rhs_contracting_axes().is_empty());
        assert!(spec.output_axes().is_empty());
    }

    #[test]
    fn from_axis_vecs_spilled_rank_with_conjugation() {
        // rank > 8 forces the SmallVec to spill; conjugation flags and
        // as_spec() must still reflect the stored axes faithfully.
        let lhs: Vec<usize> = (0..12).collect();
        let rhs: Vec<usize> = (0..10).collect();
        let out: Vec<usize> = (0..12).chain(0..10).collect();
        let spec = TensorContractSpecOwned::from_axis_vecs(
            AxisVec::from_vec(lhs.clone()),
            AxisVec::from_vec(rhs.clone()),
            AxisVec::from_vec(out.clone()),
            true,
            true,
        );
        assert_eq!(spec.lhs_contracting_axes(), lhs.as_slice());
        assert_eq!(spec.rhs_contracting_axes(), rhs.as_slice());
        assert_eq!(spec.output_axes(), out.as_slice());
        assert!(spec.lhs_conjugate());
        assert!(spec.rhs_conjugate());

        let as_spec = spec.as_spec();
        assert_eq!(as_spec.lhs_contracting_axes(), lhs.as_slice());
        assert_eq!(as_spec.rhs_contracting_axes(), rhs.as_slice());
        assert_eq!(
            as_spec.output_permutation(),
            OutputAxisOrder::from_axes(&out)
        );
        assert!(as_spec.lhs_conjugate());
        assert!(as_spec.rhs_conjugate());
    }
}
