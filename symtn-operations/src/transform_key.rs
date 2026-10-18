//! Tree-transform operation keys: pure permutation/braid descriptions with
//! no symmetry knowledge (the rule-aware cache keys stay in the symmetric
//! execution crate).

use std::fmt;
use std::sync::Arc;

use smallvec::SmallVec;

const INVALID_RAW_AXIS_POSITION: usize = usize::MAX - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TreeTransformOperationKind {
    /// Planar transpose, including codomain/domain repartitioning.
    Transpose,
    /// Permutation using the rule's symmetric braiding.
    Permute,
    /// Explicit braid with one level per source axis.
    Braid,
}

/// Immutable runtime-rank description of a tree-transform operation.
///
/// Accessors expose the TensorKit-compatible logical operation while keeping
/// its Rust storage independent of tensor rank and free to evolve.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct TreeTransformOperation {
    kind: TreeTransformOperationKind,
    data: Arc<[usize]>,
    ends: [usize; 4],
}

impl TreeTransformOperation {
    /// Heap storage retained by this operation beyond its inline value.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();
        ARC_CONTROL_BYTES.saturating_add(
            self.data
                .len()
                .saturating_mul(core::mem::size_of::<usize>()),
        )
    }

    /// Build a planar transpose operation.
    ///
    /// The two permutations follow TensorKit's `Index2Tuple` convention:
    /// both `codomain_permutation` and `domain_permutation` contain source
    /// tensor axis numbers in the full `0..numind` range. They are not local
    /// permutations within the old codomain/domain parts. For example, for a
    /// `(NOUT, NIN) = (2, 1)` tensor, keeping the domain leg in the domain uses
    /// `domain_permutation = [2]`, not `[0]`.
    pub fn transpose<Codomain, Domain>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::from_parts(
            TreeTransformOperationKind::Transpose,
            codomain_permutation,
            domain_permutation,
            std::iter::empty(),
            std::iter::empty(),
        )
    }

    /// Build a symmetric-braiding permutation operation.
    ///
    /// Axis numbering follows TensorKit's `Index2Tuple` convention; see
    /// [`Self::transpose`].
    pub fn permute<Codomain, Domain>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
    {
        Self::from_parts(
            TreeTransformOperationKind::Permute,
            codomain_permutation,
            domain_permutation,
            std::iter::empty(),
            std::iter::empty(),
        )
    }

    /// Build an explicit braid operation with source-axis permutations and levels.
    ///
    /// Axis numbering follows TensorKit's `Index2Tuple` convention; see
    /// [`Self::transpose`]. `codomain_levels` and `domain_levels` are split by
    /// the source tensor's codomain/domain tree axes, independent of the output
    /// tuple positions selected by `codomain_permutation` and
    /// `domain_permutation`. This mirrors TensorKit's `add_braid!`, which
    /// splits the full source `levels` tuple with `codomainind(tsrc)` and
    /// `domainind(tsrc)`.
    pub fn braid<Codomain, Domain, CodomainLevels, DomainLevels>(
        codomain_permutation: Codomain,
        domain_permutation: Domain,
        codomain_levels: CodomainLevels,
        domain_levels: DomainLevels,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainLevels: IntoIterator<Item = usize>,
        DomainLevels: IntoIterator<Item = usize>,
    {
        Self::from_parts(
            TreeTransformOperationKind::Braid,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    fn from_parts<Codomain, Domain, CodomainLevels, DomainLevels>(
        kind: TreeTransformOperationKind,
        codomain_permutation: Codomain,
        domain_permutation: Domain,
        codomain_levels: CodomainLevels,
        domain_levels: DomainLevels,
    ) -> Self
    where
        Codomain: IntoIterator<Item = usize>,
        Domain: IntoIterator<Item = usize>,
        CodomainLevels: IntoIterator<Item = usize>,
        DomainLevels: IntoIterator<Item = usize>,
    {
        // Why a stack buffer: the operation is rebuilt on every eager
        // contraction plan, and its only heap value should be the shared
        // `Arc<[usize]>` itself.
        let mut data = SmallVec::<[usize; 32]>::new();
        data.extend(codomain_permutation);
        let codomain_permutation_end = data.len();
        data.extend(domain_permutation);
        let domain_permutation_end = data.len();
        data.extend(codomain_levels);
        let codomain_levels_end = data.len();
        data.extend(domain_levels);
        let domain_levels_end = data.len();
        if matches!(
            kind,
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid
        ) {
            let rank = domain_permutation_end;
            data.extend(std::iter::repeat_n(usize::MAX, rank));
            for position in 0..domain_permutation_end {
                let axis = data[position];
                if axis >= rank {
                    continue;
                }
                let raw = &mut data[domain_levels_end + axis];
                if *raw != usize::MAX {
                    *raw = INVALID_RAW_AXIS_POSITION;
                    continue;
                }
                *raw = position;
            }
        }
        Self {
            kind,
            data: Arc::from(data.as_slice()),
            ends: [
                codomain_permutation_end,
                domain_permutation_end,
                codomain_levels_end,
                domain_levels_end,
            ],
        }
    }

    /// Return the operation kind.
    #[inline]
    pub fn kind(&self) -> TreeTransformOperationKind {
        self.kind
    }

    /// Return source axes selected for the destination codomain.
    #[inline]
    pub fn codomain_permutation(&self) -> &[usize] {
        &self.data[..self.ends[0]]
    }

    /// Return source axes selected for the destination domain.
    #[inline]
    pub fn domain_permutation(&self) -> &[usize] {
        &self.data[self.ends[0]..self.ends[1]]
    }

    /// Return explicit braid levels for source codomain axes.
    #[inline]
    pub fn codomain_levels(&self) -> &[usize] {
        &self.data[self.ends[1]..self.ends[2]]
    }

    /// Return explicit braid levels for source domain axes.
    #[inline]
    pub fn domain_levels(&self) -> &[usize] {
        &self.data[self.ends[2]..self.ends[3]]
    }

    /// Return source-logical-axis positions for a permutation or braid.
    ///
    /// The table is derived with the operation, before any source split is
    /// known. Invalid logical maps retain sentinels so construction keeps the
    /// public API's deferred validation and error precedence.
    #[doc(hidden)]
    #[inline]
    pub fn raw_axis_positions(&self) -> Option<&[usize]> {
        match self.kind {
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => {
                Some(&self.data[self.ends[3]..])
            }
            TreeTransformOperationKind::Transpose => None,
        }
    }

    pub fn requires_symmetric_braiding(&self) -> bool {
        self.kind == TreeTransformOperationKind::Permute
    }

    /// Whether this operation describes the exact current axis order and
    /// codomain/domain split for a source with the given ranks.
    ///
    /// Braid levels must also describe every source leg. Their values do not
    /// matter once the permutation has zero adjacent swaps. A transpose is
    /// identity only at the exact current split. Why not infer from cyclic
    /// position alone: a split-changing transpose carries bend and dual
    /// semantics even when no ordinary axis swap is visible.
    pub fn is_identity_for(&self, codomain_rank: usize, domain_rank: usize) -> bool {
        let axes_are_identity = |codomain: &[usize], domain: &[usize]| {
            codomain.iter().copied().eq(0..codomain_rank)
                && domain
                    .iter()
                    .copied()
                    .eq(codomain_rank..codomain_rank + domain_rank)
        };
        match self.kind {
            TreeTransformOperationKind::Braid => {
                self.codomain_levels().len() == codomain_rank
                    && self.domain_levels().len() == domain_rank
                    && axes_are_identity(self.codomain_permutation(), self.domain_permutation())
            }
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Transpose => {
                axes_are_identity(self.codomain_permutation(), self.domain_permutation())
            }
        }
    }
}

impl fmt::Debug for TreeTransformOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct(match self.kind {
            TreeTransformOperationKind::Transpose => "Transpose",
            TreeTransformOperationKind::Permute => "Permute",
            TreeTransformOperationKind::Braid => "Braid",
        });
        debug
            .field("codomain_permutation", &self.codomain_permutation())
            .field("domain_permutation", &self.domain_permutation());
        if self.kind == TreeTransformOperationKind::Braid {
            debug
                .field("codomain_levels", &self.codomain_levels())
                .field("domain_levels", &self.domain_levels());
        }
        debug.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    use super::{TreeTransformOperation, TreeTransformOperationKind, INVALID_RAW_AXIS_POSITION};

    #[test]
    fn operation_values_expose_all_logical_segments_at_runtime_rank() {
        // What: every operation kind preserves its logical segments, including
        // values above the inline-storage rank.
        let transpose = TreeTransformOperation::transpose(0..9, [9]);
        assert_eq!(transpose.kind(), TreeTransformOperationKind::Transpose);
        assert_eq!(
            transpose.codomain_permutation(),
            &(0..9).collect::<Vec<_>>()
        );
        assert_eq!(transpose.domain_permutation(), &[9]);
        assert!(transpose.codomain_levels().is_empty());
        assert!(transpose.domain_levels().is_empty());

        let permute = TreeTransformOperation::permute([1, 0], [2]);
        assert_eq!(permute.kind(), TreeTransformOperationKind::Permute);
        assert_eq!(permute.codomain_permutation(), &[1, 0]);
        assert_eq!(permute.domain_permutation(), &[2]);
        assert!(permute.codomain_levels().is_empty());
        assert!(permute.domain_levels().is_empty());
        assert_eq!(permute.raw_axis_positions(), Some(&[1, 0, 2][..]));

        let braid = TreeTransformOperation::braid([1, 0], [2], [7, 3], [5]);
        assert_eq!(braid.kind(), TreeTransformOperationKind::Braid);
        assert_eq!(braid.codomain_permutation(), &[1, 0]);
        assert_eq!(braid.domain_permutation(), &[2]);
        assert_eq!(braid.codomain_levels(), &[7, 3]);
        assert_eq!(braid.domain_levels(), &[5]);
        assert_eq!(braid.raw_axis_positions(), Some(&[1, 0, 2][..]));
        assert_eq!(transpose.raw_axis_positions(), None);

        let equal = TreeTransformOperation::braid(vec![1, 0], vec![2], vec![7, 3], vec![5]);
        let mut left_hash = DefaultHasher::new();
        let mut right_hash = DefaultHasher::new();
        braid.hash(&mut left_hash);
        equal.hash(&mut right_hash);
        assert!(!Arc::ptr_eq(&braid.data, &equal.data));
        assert_eq!(braid, equal);
        assert_eq!(left_hash.finish(), right_hash.finish());
    }

    #[test]
    fn raw_axis_positions_preserve_invalid_operation_construction() {
        // What: malformed operations remain constructible and defer their
        // established validation errors until a rule-aware execution path.
        let invalid = TreeTransformOperation::permute([0, 0], []);
        assert_eq!(
            invalid.raw_axis_positions(),
            Some(&[INVALID_RAW_AXIS_POSITION, usize::MAX][..])
        );
    }

    #[test]
    fn retained_bytes_include_operation_local_inverse_positions() {
        // What: cache resource accounting charges the full immutable backing
        // allocation, including the derived position segment.
        let word = core::mem::size_of::<usize>();
        let operation = TreeTransformOperation::permute([1, 0], [2]);
        assert_eq!(operation.charged_retained_bytes(), 8 * word);
    }

    #[test]
    fn debug_preserves_the_public_variant_shaped_text() {
        // What: opaque storage leaves the public operation text unchanged.
        assert_eq!(
            format!("{:?}", TreeTransformOperation::transpose([2, 0], [1])),
            "Transpose { codomain_permutation: [2, 0], domain_permutation: [1] }"
        );
        assert_eq!(
            format!("{:?}", TreeTransformOperation::permute([1, 0], [2])),
            "Permute { codomain_permutation: [1, 0], domain_permutation: [2] }"
        );
        assert_eq!(
            format!("{:?}", TreeTransformOperation::braid([1, 0], [2], [0, 1], [2])),
            "Braid { codomain_permutation: [1, 0], domain_permutation: [2], codomain_levels: [0, 1], domain_levels: [2] }"
        );

        let error = crate::OperationError::UnsupportedTreeTransformScope {
            operation: Box::new(TreeTransformOperation::permute([1, 0], [2])),
            message: "representative scope",
        };
        assert_eq!(
            error.to_string(),
            "unsupported tree transform scope for operation Permute { codomain_permutation: [1, 0], domain_permutation: [2] }: representative scope"
        );
    }

    #[test]
    fn identity_axis_map_requires_the_current_split_and_valid_braid_levels() {
        // What: identity classification accepts exact axes only after the
        // operation itself has a complete source-level description.
        assert!(TreeTransformOperation::permute([0, 1], [2]).is_identity_for(2, 1));
        assert!(TreeTransformOperation::braid([0, 1], [2], [7, 3], [5]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::braid([0, 1], [2], [7], [5]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::permute([0], [1, 2]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::permute([1, 0], [2]).is_identity_for(2, 1));
        assert!(TreeTransformOperation::transpose([0, 1], [2]).is_identity_for(2, 1));
        assert!(!TreeTransformOperation::transpose([0], [1, 2]).is_identity_for(2, 1));
    }

    #[test]
    fn rank_zero_identity_axis_map_is_well_formed() {
        // What: empty Permute/Braid descriptions are valid rank-zero
        // identities, including an exact same-split Transpose.
        assert!(TreeTransformOperation::permute([], []).is_identity_for(0, 0));
        assert!(TreeTransformOperation::braid([], [], [], []).is_identity_for(0, 0));
        assert!(TreeTransformOperation::transpose([], []).is_identity_for(0, 0));
    }
}
