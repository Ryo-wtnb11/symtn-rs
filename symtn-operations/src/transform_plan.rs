//! Tree-transform plan data: block/group specs (destination trees, source
//! trees, recoupling matrices) and the grouped plan container. Pure data —
//! the symmetric compile layer in `symtn-tensors` builds these from fusion
//! rules; replay consumes them without any symmetry knowledge.

use std::borrow::Cow;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use symtn_core::{
    BlockKey, BlockStructure, FusionTreeBlockGroup, FusionTreeGroupKey, FusionTreePairKey,
};

use crate::transform_helpers::{
    duplicate_fusion_tree_pair_indices, fusion_tree_group_block_keys,
    fusion_tree_pair_matches_group, fusion_tree_pairs_share_group,
};
use crate::transform_structure::{TreeTransformCoefficients, TreeTransformStructure};
use crate::OperationError;

/// Why shared slices: a Runtime reuses one group's specs across the plans of
/// every sector structure containing that group, so a clone must be O(1)
/// rather than a copy of the `|dst|·|src|` recoupling matrix.
#[derive(Clone, Debug)]
enum SpecEntries<K, T> {
    Single {
        dst: K,
        src: K,
        coefficient: T,
    },
    Multi {
        dst: Arc<[K]>,
        src: Arc<[K]>,
        coefficients: Arc<[T]>,
    },
}

impl<K, T> SpecEntries<K, T> {
    #[inline]
    fn dst(&self) -> &[K] {
        match self {
            Self::Single { dst, .. } => std::slice::from_ref(dst),
            Self::Multi { dst, .. } => dst,
        }
    }

    #[inline]
    fn src(&self) -> &[K] {
        match self {
            Self::Single { src, .. } => std::slice::from_ref(src),
            Self::Multi { src, .. } => src,
        }
    }

    #[inline]
    fn coefficients(&self) -> &[T] {
        match self {
            Self::Single { coefficient, .. } => std::slice::from_ref(coefficient),
            Self::Multi { coefficients, .. } => coefficients,
        }
    }

    #[inline]
    fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        match self {
            Self::Single { .. } => None,
            Self::Multi { coefficients, .. } => Some(coefficients),
        }
    }
}

#[derive(Debug)]
enum ResolvedSpecEntries<'a, T> {
    Single {
        dst: usize,
        src: usize,
        coefficient: &'a T,
    },
    Multi {
        dst: Vec<usize>,
        src: Vec<usize>,
        coefficients: &'a Arc<[T]>,
    },
}

impl<T> ResolvedSpecEntries<'_, T> {
    #[inline]
    fn dst(&self) -> &[usize] {
        match self {
            Self::Single { dst, .. } => std::slice::from_ref(dst),
            Self::Multi { dst, .. } => dst,
        }
    }

    #[inline]
    fn src(&self) -> &[usize] {
        match self {
            Self::Single { src, .. } => std::slice::from_ref(src),
            Self::Multi { src, .. } => src,
        }
    }

    #[inline]
    fn coefficients(&self) -> &[T] {
        match self {
            Self::Single { coefficient, .. } => std::slice::from_ref(coefficient),
            Self::Multi { coefficients, .. } => coefficients,
        }
    }

    #[inline]
    fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        match self {
            Self::Single { .. } => None,
            Self::Multi { coefficients, .. } => Some(coefficients),
        }
    }

    fn map_source_blocks<F>(&mut self, logical_to_storage_block: &F) -> Result<(), OperationError>
    where
        F: Fn(usize) -> Result<usize, OperationError>,
    {
        match self {
            Self::Single { src, .. } => *src = logical_to_storage_block(*src)?,
            Self::Multi { src, .. } => {
                for block in src {
                    *block = logical_to_storage_block(*block)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedTreeTransformBlockSpec<'a, T> {
    entries: ResolvedSpecEntries<'a, T>,
    source_axes: Option<Cow<'a, [usize]>>,
}

impl<'a, T> ResolvedTreeTransformBlockSpec<'a, T> {
    fn from_entries<K, FindIndex, EraseKey>(
        entries: &'a SpecEntries<K, T>,
        source_axes: Option<&'a [usize]>,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        find_index: FindIndex,
        erase_key: EraseKey,
    ) -> Result<Self, OperationError>
    where
        FindIndex: Fn(&BlockStructure, &K) -> Option<usize>,
        EraseKey: Fn(&K) -> BlockKey,
    {
        let resolve = |structure: &BlockStructure, key: &K| {
            find_index(structure, key).ok_or_else(|| OperationError::MissingBlockKey {
                key: Box::new(erase_key(key)),
            })
        };
        Self::from_entries_with_resolvers(
            entries,
            source_axes,
            |key| resolve(dst_structure, key),
            |key| resolve(src_structure, key),
        )
    }

    fn from_entries_with_resolvers<K, ResolveDst, ResolveSrc>(
        entries: &'a SpecEntries<K, T>,
        source_axes: Option<&'a [usize]>,
        resolve_dst: ResolveDst,
        resolve_src: ResolveSrc,
    ) -> Result<Self, OperationError>
    where
        ResolveDst: Fn(&K) -> Result<usize, OperationError>,
        ResolveSrc: Fn(&K) -> Result<usize, OperationError>,
    {
        let entries = match entries {
            SpecEntries::Single {
                dst,
                src,
                coefficient,
            } => ResolvedSpecEntries::Single {
                dst: resolve_dst(dst)?,
                src: resolve_src(src)?,
                coefficient,
            },
            SpecEntries::Multi {
                dst,
                src,
                coefficients,
            } => ResolvedSpecEntries::Multi {
                dst: dst.iter().map(&resolve_dst).collect::<Result<_, _>>()?,
                src: src.iter().map(resolve_src).collect::<Result<_, _>>()?,
                coefficients,
            },
        };
        Ok(Self {
            entries,
            source_axes: source_axes.map(Cow::Borrowed),
        })
    }

    pub(crate) fn dst_blocks(&self) -> &[usize] {
        self.entries.dst()
    }

    pub(crate) fn src_blocks(&self) -> &[usize] {
        self.entries.src()
    }

    pub(crate) fn coefficients(&self) -> &[T] {
        self.entries.coefficients()
    }

    pub(crate) fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        self.entries.shared_coefficients()
    }

    pub(crate) fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }

    fn map_storage<FBlock, FAxis>(
        mut self,
        logical_rank: usize,
        logical_to_storage_block: &FBlock,
        logical_to_storage_axis: &FAxis,
    ) -> Result<Self, OperationError>
    where
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        self.entries.map_source_blocks(logical_to_storage_block)?;
        let storage_axes = match self.source_axes.as_deref() {
            Some(logical_axes) => logical_axes
                .iter()
                .copied()
                .map(logical_to_storage_axis)
                .collect::<Result<Vec<_>, _>>()?,
            None => (0..logical_rank)
                .map(logical_to_storage_axis)
                .collect::<Result<Vec<_>, _>>()?,
        };
        self.source_axes = Some(Cow::Owned(storage_axes));
        Ok(self)
    }
}

impl<K: PartialEq, T: PartialEq> PartialEq for SpecEntries<K, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dst() == other.dst()
            && self.src() == other.src()
            && self.coefficients() == other.coefficients()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformBlockSpec<T> {
    entries: SpecEntries<usize, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformBlockSpec<T> {
    pub fn single(dst_block: usize, src_block: usize, coefficient: T) -> Self {
        Self {
            entries: SpecEntries::Single {
                dst: dst_block,
                src: src_block,
                coefficient,
            },
            source_axes: None,
        }
    }

    pub fn multi(
        dst_blocks: Vec<usize>,
        src_blocks: Vec<usize>,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Self {
        Self {
            entries: SpecEntries::Multi {
                dst: dst_blocks.into(),
                src: src_blocks.into(),
                coefficients: recoupling_coefficients_dst_src.into(),
            },
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    /// As [`Self::with_source_axes`], sharing `axes` instead of copying them,
    /// so specs that read their sources in one axis order share one buffer.
    #[doc(hidden)]
    pub fn with_shared_source_axes(mut self, axes: Arc<[usize]>) -> Self {
        self.source_axes = Some(axes);
        self
    }

    #[inline]
    pub fn dst_blocks(&self) -> &[usize] {
        self.entries.dst()
    }

    #[inline]
    pub fn src_blocks(&self) -> &[usize] {
        self.entries.src()
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        self.entries.coefficients()
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }

    #[inline]
    pub(crate) fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        self.entries.shared_coefficients()
    }
}

/// Explicit replay descriptor keyed by application block labels.
///
/// This is the namespace-neutral entry point for Dense, Opaque, and
/// FusionTree block labels. Categorical grouped recoupling uses
/// [`TreeTransformGroupBlockSpec`] instead.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformKeyBlockSpec<T> {
    entries: SpecEntries<BlockKey, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformKeyBlockSpec<T> {
    pub fn single<KDst, KSrc>(dst_key: KDst, src_key: KSrc, coefficient: T) -> Self
    where
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            entries: SpecEntries::Single {
                dst: dst_key.into(),
                src: src_key.into(),
                coefficient,
            },
            source_axes: None,
        }
    }

    pub fn multi<DstKeys, SrcKeys, KDst, KSrc>(
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Self
    where
        DstKeys: IntoIterator<Item = KDst>,
        SrcKeys: IntoIterator<Item = KSrc>,
        KDst: Into<BlockKey>,
        KSrc: Into<BlockKey>,
    {
        Self {
            entries: SpecEntries::Multi {
                dst: dst_keys.into_iter().map(Into::into).collect(),
                src: src_keys.into_iter().map(Into::into).collect(),
                coefficients: recoupling_coefficients_dst_src.into(),
            },
            source_axes: None,
        }
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    #[inline]
    pub fn dst_keys(&self) -> &[BlockKey] {
        self.entries.dst()
    }

    #[inline]
    pub fn src_keys(&self) -> &[BlockKey] {
        self.entries.src()
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        self.entries.coefficients()
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }
}

impl<T> TreeTransformKeyBlockSpec<T> {
    pub(crate) fn resolve(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<ResolvedTreeTransformBlockSpec<'_, T>, OperationError> {
        ResolvedTreeTransformBlockSpec::from_entries(
            &self.entries,
            self.source_axes(),
            dst_structure,
            src_structure,
            BlockStructure::find_block_index_by_key,
            Clone::clone,
        )
    }
}

/// Categorical grouped-recoupling descriptor.
///
/// Source and destination identities are fusion-tree pairs. Use
/// [`TreeTransformKeyBlockSpec`] when replay must address arbitrary Dense or
/// Opaque application labels.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformGroupBlockSpec<T> {
    entries: SpecEntries<FusionTreePairKey, T>,
    source_axes: Option<Arc<[usize]>>,
}

impl<T> TreeTransformGroupBlockSpec<T> {
    /// Creates one categorical row/column transform.
    ///
    /// The stored group identity is the source key's fusion-tree group.
    pub fn single(dst_key: FusionTreePairKey, src_key: FusionTreePairKey, coefficient: T) -> Self {
        Self {
            entries: SpecEntries::Single {
                dst: dst_key,
                src: src_key,
                coefficient,
            },
            source_axes: None,
        }
    }

    /// Creates a categorical transform with coefficients ordered as `U[dst, src]`.
    ///
    /// Each side must be nonempty, internally group-coherent, and free of
    /// duplicate fusion-tree identities. The source and destination groups may
    /// differ. Violations return [`OperationError::EmptyTransformBlock`],
    /// [`OperationError::FusionTreeGroupMismatch`],
    /// [`OperationError::DuplicateTreeTransformKey`], or
    /// [`OperationError::CoefficientCountMismatch`]; an unrepresentable
    /// row/column product returns [`OperationError::ElementCountOverflow`].
    pub fn try_multi<DstKeys, SrcKeys>(
        dst_keys: DstKeys,
        src_keys: SrcKeys,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Result<Self, OperationError>
    where
        DstKeys: IntoIterator<Item = FusionTreePairKey>,
        SrcKeys: IntoIterator<Item = FusionTreePairKey>,
    {
        let dst_keys = dst_keys.into_iter().collect::<Vec<_>>();
        let src_keys = src_keys.into_iter().collect::<Vec<_>>();
        let Some(first_src) = src_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        let Some(first_dst) = dst_keys.first() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        let group_key = first_src.group_key();
        if let Some(index) = src_keys
            .iter()
            .position(|key| !fusion_tree_pair_matches_group(key, &group_key))
        {
            return Err(OperationError::FusionTreeGroupMismatch {
                tensor: "src",
                index,
            });
        }
        if let Some(index) = dst_keys
            .iter()
            .position(|key| !fusion_tree_pairs_share_group(key, first_dst))
        {
            return Err(OperationError::FusionTreeGroupMismatch {
                tensor: "dst",
                index,
            });
        }
        let (duplicate_src, duplicate_dst) =
            duplicate_fusion_tree_pair_indices(&src_keys, &dst_keys);
        if let Some(index) = duplicate_src {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index,
            });
        }
        if let Some(index) = duplicate_dst {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "dst",
                index,
            });
        }
        Self::multi_from_validated(dst_keys, src_keys, recoupling_coefficients_dst_src)
    }

    fn multi_from_validated(
        dst_keys: Vec<FusionTreePairKey>,
        src_keys: Vec<FusionTreePairKey>,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Result<Self, OperationError> {
        let expected = dst_keys
            .len()
            .checked_mul(src_keys.len())
            .ok_or(OperationError::ElementCountOverflow)?;
        if recoupling_coefficients_dst_src.len() != expected {
            return Err(OperationError::CoefficientCountMismatch {
                expected,
                actual: recoupling_coefficients_dst_src.len(),
            });
        }
        Ok(Self {
            entries: SpecEntries::Multi {
                dst: dst_keys.into(),
                src: src_keys.into(),
                coefficients: recoupling_coefficients_dst_src.into(),
            },
            source_axes: None,
        })
    }

    pub fn with_source_axes<I>(mut self, axes: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        self.source_axes = Some(axes.into_iter().collect());
        self
    }

    /// Reuse one immutable source-axis map across plan entries.
    #[doc(hidden)]
    pub fn with_shared_source_axes(mut self, axes: Arc<[usize]>) -> Self {
        self.source_axes = Some(axes);
        self
    }

    pub fn from_block_groups(
        dst_structure: &BlockStructure,
        dst_group: &FusionTreeBlockGroup,
        src_structure: &BlockStructure,
        src_group: &FusionTreeBlockGroup,
        recoupling_coefficients_dst_src: Vec<T>,
    ) -> Result<Self, OperationError> {
        let dst_keys = fusion_tree_group_block_keys(dst_structure, dst_group, "dst")?;
        let src_keys = fusion_tree_group_block_keys(src_structure, src_group, "src")?;
        if src_keys.is_empty() || dst_keys.is_empty() {
            return Err(OperationError::EmptyTransformBlock);
        }
        let (duplicate_src, duplicate_dst) =
            duplicate_fusion_tree_pair_indices(&src_keys, &dst_keys);
        if let Some(index) = duplicate_src {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "src",
                index,
            });
        }
        if let Some(index) = duplicate_dst {
            return Err(OperationError::DuplicateTreeTransformKey {
                tensor: "dst",
                index,
            });
        }
        Self::multi_from_validated(dst_keys, src_keys, recoupling_coefficients_dst_src)
    }

    /// The source fusion-tree group. Why derived rather than stored: every
    /// constructor takes it from the first source key, so a stored copy is
    /// a group key per spec that two refcount bumps rebuild (#1998, #2192).
    #[inline]
    pub fn group_key(&self) -> FusionTreeGroupKey {
        self.entries.src()[0].group_key()
    }

    /// Whether the keys and coefficients live in shared heap slices (`Multi`)
    /// rather than inline (`Single`). Cache charging only.
    #[doc(hidden)]
    #[inline]
    pub fn has_shared_slices(&self) -> bool {
        matches!(self.entries, SpecEntries::Multi { .. })
    }

    #[inline]
    pub fn dst_keys(&self) -> &[FusionTreePairKey] {
        self.entries.dst()
    }

    /// Replace every key in place, destinations first, then sources, each in
    /// order; a cache re-homes a freshly built spec's keys into the entry's
    /// own arena this way (Rule O, #2192). A shared key slice is copied once.
    #[doc(hidden)]
    pub fn rehome_keys(&mut self, mut rehome: impl FnMut(&FusionTreePairKey) -> FusionTreePairKey) {
        fn in_place(
            keys: &mut Arc<[FusionTreePairKey]>,
            rehome: &mut impl FnMut(&FusionTreePairKey) -> FusionTreePairKey,
        ) {
            match Arc::get_mut(keys) {
                Some(keys) => keys.iter_mut().for_each(|key| *key = rehome(key)),
                None => *keys = keys.iter().map(rehome).collect(),
            }
        }
        match &mut self.entries {
            SpecEntries::Single { dst, src, .. } => {
                *dst = rehome(dst);
                *src = rehome(src);
            }
            SpecEntries::Multi { dst, src, .. } => {
                in_place(dst, &mut rehome);
                in_place(src, &mut rehome);
            }
        }
    }

    #[inline]
    pub fn src_keys(&self) -> &[FusionTreePairKey] {
        self.entries.src()
    }

    /// Recoupling matrix coefficients stored as `U[dst, src]` in row-major
    /// destination-by-source order: `coeff[src + dst * src_count]`.
    #[inline]
    pub fn recoupling_coefficients_dst_src(&self) -> &[T] {
        self.entries.coefficients()
    }

    #[inline]
    pub fn source_axes(&self) -> Option<&[usize]> {
        self.source_axes.as_deref()
    }
}

impl<T> TreeTransformGroupBlockSpec<T> {
    pub(crate) fn resolve(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<ResolvedTreeTransformBlockSpec<'_, T>, OperationError> {
        ResolvedTreeTransformBlockSpec::from_entries(
            &self.entries,
            self.source_axes(),
            dst_structure,
            src_structure,
            BlockStructure::find_block_index_by_fusion_tree_pair,
            |key| BlockKey::FusionTree(key.clone()),
        )
    }

    #[inline]
    pub(crate) fn shared_coefficients(&self) -> Option<&Arc<[T]>> {
        self.entries.shared_coefficients()
    }

    fn resolve_with_source_projection<F>(
        &self,
        dst_structure: &BlockStructure,
        source_index: &F,
    ) -> Result<ResolvedTreeTransformBlockSpec<'_, T>, OperationError>
    where
        F: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
    {
        let resolve_dst = |key: &FusionTreePairKey| {
            dst_structure
                .find_block_index_by_fusion_tree_pair(key)
                .ok_or_else(|| OperationError::MissingBlockKey {
                    key: Box::new(BlockKey::FusionTree(key.clone())),
                })
        };
        ResolvedTreeTransformBlockSpec::from_entries_with_resolvers(
            &self.entries,
            self.source_axes(),
            resolve_dst,
            source_index,
        )
    }
}

/// Immutable categorical fusion-tree transform, independent of storage layout.
///
/// The first binding builds one coefficient payload that every later binding
/// shares by an `Arc` bump. It references each group's recoupling matrix
/// through the spec's shared `Arc<[T]>`, as TensorKit's transformer references
/// the cached per-`FusionTreeBlock` `U`, and copies only Single scalars. Why
/// not one flattened payload per plan: plans are assembled from cached group
/// specs, and flattening would copy every reused matrix.
#[derive(Clone)]
pub struct TreeTransformGroupPlan<T> {
    specs: Vec<TreeTransformGroupBlockSpec<T>>,
    coefficients: OnceLock<Arc<TreeTransformCoefficients<T>>>,
}

impl<T: fmt::Debug> fmt::Debug for TreeTransformGroupPlan<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TreeTransformGroupPlan")
            .field("specs", &self.specs)
            .finish()
    }
}

impl<T: PartialEq> PartialEq for TreeTransformGroupPlan<T> {
    fn eq(&self, other: &Self) -> bool {
        self.specs == other.specs
    }
}

impl<T> TreeTransformGroupPlan<T> {
    pub fn new(specs: Vec<TreeTransformGroupBlockSpec<T>>) -> Self {
        Self {
            specs,
            coefficients: OnceLock::new(),
        }
    }

    pub fn from_specs<I>(specs: I) -> Self
    where
        I: IntoIterator<Item = TreeTransformGroupBlockSpec<T>>,
    {
        Self::new(specs.into_iter().collect())
    }

    #[inline]
    pub fn specs(&self) -> &[TreeTransformGroupBlockSpec<T>] {
        &self.specs
    }

    pub fn into_specs(self) -> Vec<TreeTransformGroupBlockSpec<T>> {
        self.specs
    }
}

/// The coefficient payload of `specs`: Single scalars copied, each Multi
/// matrix referenced through its shared `Arc`.
fn coefficients_of<'s, T: Copy + 's>(
    specs: impl Iterator<Item = &'s TreeTransformGroupBlockSpec<T>> + Clone,
) -> Result<TreeTransformCoefficients<T>, OperationError> {
    TreeTransformCoefficients::from_specs(specs.map(|spec| {
        (
            spec.dst_keys().len(),
            spec.src_keys().len(),
            spec.recoupling_coefficients_dst_src(),
            spec.shared_coefficients(),
        )
    }))
}

fn bind_with_storage_conjugation<'s, T: Copy + 's>(
    specs: impl ExactSizeIterator<Item = &'s TreeTransformGroupBlockSpec<T>>,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
    storage_conjugate: bool,
    coefficients: impl FnOnce() -> Result<Arc<TreeTransformCoefficients<T>>, OperationError>,
) -> Result<TreeTransformStructure<T>, OperationError> {
    let mut resolved = Vec::with_capacity(specs.len());
    for spec in specs {
        resolved.push(spec.resolve(&dst_structure, &src_structure)?);
    }
    TreeTransformStructure::compile_resolved_shared_structures(
        dst_structure,
        src_structure,
        &resolved,
        storage_conjugate,
        coefficients()?,
    )
}

#[allow(clippy::too_many_arguments)]
fn bind_with_storage_mapping<'s, T: Copy + 's, FBlock, FAxis>(
    specs: impl ExactSizeIterator<Item = &'s TreeTransformGroupBlockSpec<T>>,
    dst_structure: Arc<BlockStructure>,
    logical_src_structure: &BlockStructure,
    storage_src_structure: Arc<BlockStructure>,
    logical_to_storage_block: FBlock,
    logical_to_storage_axis: FAxis,
    storage_conjugate: bool,
    coefficients: impl FnOnce() -> Result<Arc<TreeTransformCoefficients<T>>, OperationError>,
) -> Result<TreeTransformStructure<T>, OperationError>
where
    FBlock: Fn(usize) -> Result<usize, OperationError>,
    FAxis: Fn(usize) -> Result<usize, OperationError>,
{
    let mut resolved = Vec::with_capacity(specs.len());
    // Why not resolve every key first: fallible block/axis mapping completes
    // per spec before the next key lookup, preserving callback error order.
    for spec in specs {
        resolved.push(
            spec.resolve(&dst_structure, logical_src_structure)?
                .map_storage(
                    logical_src_structure.rank(),
                    &logical_to_storage_block,
                    &logical_to_storage_axis,
                )?,
        );
    }
    TreeTransformStructure::compile_resolved_shared_structures(
        dst_structure,
        storage_src_structure,
        &resolved,
        storage_conjugate,
        coefficients()?,
    )
}

#[allow(clippy::too_many_arguments)]
fn bind_with_source_projection<'s, T: Copy + 's, FSource, FAxis>(
    specs: impl ExactSizeIterator<Item = &'s TreeTransformGroupBlockSpec<T>>,
    dst_structure: Arc<BlockStructure>,
    storage_src_structure: Arc<BlockStructure>,
    logical_rank: usize,
    source_index: FSource,
    logical_to_storage_axis: FAxis,
    storage_conjugate: bool,
    coefficients: impl FnOnce() -> Result<Arc<TreeTransformCoefficients<T>>, OperationError>,
) -> Result<TreeTransformStructure<T>, OperationError>
where
    FSource: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
    FAxis: Fn(usize) -> Result<usize, OperationError>,
{
    let identity_block = |index| Ok(index);
    let mut resolved = Vec::with_capacity(specs.len());
    for spec in specs {
        resolved.push(
            spec.resolve_with_source_projection(&dst_structure, &source_index)?
                .map_storage(logical_rank, &identity_block, &logical_to_storage_axis)?,
        );
    }
    TreeTransformStructure::compile_resolved_shared_structures(
        dst_structure,
        storage_src_structure,
        &resolved,
        storage_conjugate,
        coefficients()?,
    )
}

impl<T: Copy> TreeTransformGroupPlan<T> {
    fn shared_coefficients(&self) -> Result<Arc<TreeTransformCoefficients<T>>, OperationError> {
        if let Some(coefficients) = self.coefficients.get() {
            return Ok(Arc::clone(coefficients));
        }
        let built = coefficients_of(self.specs.iter())?;
        Ok(Arc::clone(
            self.coefficients.get_or_init(|| Arc::new(built)),
        ))
    }

    fn compile_shared_structures_internal(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        bind_with_storage_conjugation(
            self.specs.iter(),
            dst_structure,
            src_structure,
            storage_conjugate,
            || self.shared_coefficients(),
        )
    }

    pub fn compile_structures(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        self.compile_structures_with_storage_conjugation(dst_structure, src_structure, false)
    }

    pub fn compile_structures_with_storage_conjugation(
        &self,
        dst_structure: &BlockStructure,
        src_structure: &BlockStructure,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        self.compile_shared_structures_internal(
            Arc::new(dst_structure.clone()),
            Arc::new(src_structure.clone()),
            storage_conjugate,
        )
    }

    pub fn compile_shared_structures_with_storage_conjugation(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        self.compile_shared_structures_internal(dst_structure, src_structure, storage_conjugate)
    }

    pub fn compile_shared_structures_with_storage_mapping<FBlock, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        logical_src_structure: &BlockStructure,
        storage_src_structure: Arc<BlockStructure>,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        bind_with_storage_mapping(
            self.specs.iter(),
            dst_structure,
            logical_src_structure,
            storage_src_structure,
            logical_to_storage_block,
            logical_to_storage_axis,
            storage_conjugate,
            || self.shared_coefficients(),
        )
    }

    #[doc(hidden)]
    pub fn compile_shared_structures_with_source_projection<FSource, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        storage_src_structure: Arc<BlockStructure>,
        logical_rank: usize,
        source_index: FSource,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        FSource: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        bind_with_source_projection(
            self.specs.iter(),
            dst_structure,
            storage_src_structure,
            logical_rank,
            source_index,
            logical_to_storage_axis,
            storage_conjugate,
            || self.shared_coefficients(),
        )
    }
}

/// One source group's specs as the composed-coefficient cache
/// holds them, plus the payload last bound from a group set that starts
/// with this group.
///
/// The payload is derived data of that set, which is a sequence of these
/// cached groups, so it lives with them instead of in a cache of its own.
/// The slot is weak: it shares a payload while some binding holds it and
/// retains nothing once the last one is dropped. One slot per group: two sets
/// that start with one group and alternate rebuild their payload per
/// binding, as before this slot existed (#1753).
///
/// A payload names the groups it was built from by their `id`s, which are
/// never reused in a process, so it retains no group after eviction (#2251).
#[doc(hidden)]
pub struct TreeTransformSpecGroup<T> {
    id: u64,
    specs: Box<[TreeTransformGroupBlockSpec<T>]>,
    payload: Mutex<Weak<TreeTransformCoefficients<T>>>,
}

impl<T> TreeTransformSpecGroup<T> {
    pub fn new(specs: Vec<TreeTransformGroupBlockSpec<T>>) -> Self {
        // One counter for every `T`; 2^64 ids cannot wrap in a process.
        static NEXT_GROUP_ID: AtomicU64 = AtomicU64::new(0);
        Self {
            id: NEXT_GROUP_ID.fetch_add(1, Ordering::Relaxed),
            specs: specs.into_boxed_slice(),
            payload: Mutex::new(Weak::new()),
        }
    }

    /// This group's process-unique identity.
    #[inline]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    #[inline]
    pub fn specs(&self) -> &[TreeTransformGroupBlockSpec<T>] {
        &self.specs
    }

    fn payload(&self) -> std::sync::MutexGuard<'_, Weak<TreeTransformCoefficients<T>>> {
        // The slot holds a `Weak` only; a panic mid-store leaves a valid one.
        self.payload.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A categorical transform assembled from shared cached groups (the
/// composed-coefficient cache's entries), bound without copying
/// any spec: a plan miss whose groups all hit allocates only the group list.
/// Every binding of one group set shares one coefficient payload, so a
/// degeneracy-only rebinding neither rebuilds it on the host nor, through
/// its identity, uploads it again on a device (#1753).
#[doc(hidden)]
#[derive(Clone)]
pub struct SharedTreeTransformGroupPlan<T> {
    groups: Vec<Arc<TreeTransformSpecGroup<T>>>,
    len: usize,
}

impl<T> SharedTreeTransformGroupPlan<T> {
    pub fn new(groups: Vec<Arc<TreeTransformSpecGroup<T>>>) -> Self {
        let len = groups.iter().map(|group| group.specs.len()).sum();
        Self { groups, len }
    }

    /// Every spec, group by group.
    pub fn specs(&self) -> impl ExactSizeIterator<Item = &TreeTransformGroupBlockSpec<T>> + Clone {
        SpecsIter {
            groups: self.groups.iter(),
            current: [].iter(),
            remaining: self.len,
        }
    }
}

struct SpecsIter<'a, T> {
    groups: std::slice::Iter<'a, Arc<TreeTransformSpecGroup<T>>>,
    current: std::slice::Iter<'a, TreeTransformGroupBlockSpec<T>>,
    remaining: usize,
}

// Why manual: a derive would bound `T: Clone`.
impl<T> Clone for SpecsIter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            groups: self.groups.clone(),
            current: self.current.clone(),
            remaining: self.remaining,
        }
    }
}

impl<'a, T> Iterator for SpecsIter<'a, T> {
    type Item = &'a TreeTransformGroupBlockSpec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(spec) = self.current.next() {
                self.remaining -= 1;
                return Some(spec);
            }
            self.current = self.groups.next()?.specs.iter();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for SpecsIter<'_, T> {}

impl<T: Copy> SharedTreeTransformGroupPlan<T> {
    /// The payload of this group set: the live one another binding of the
    /// same set built, else a new one, recorded for the next binding.
    fn coefficients(&self) -> Result<Arc<TreeTransformCoefficients<T>>, OperationError> {
        let Some(first) = self.groups.first() else {
            return coefficients_of(self.specs()).map(Arc::new);
        };
        if let Some(resident) = first.payload().upgrade() {
            if resident.is_built_from(&self.groups) {
                return Ok(resident);
            }
        }
        // Built outside the lock: concurrent first bindings may each build
        // one, equal in value; the last recorded is the one later bindings share.
        let built = Arc::new(coefficients_of(self.specs())?.built_from(&self.groups));
        *first.payload() = Arc::downgrade(&built);
        Ok(built)
    }

    pub fn compile_shared_structures_with_storage_conjugation(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError> {
        bind_with_storage_conjugation(
            self.specs(),
            dst_structure,
            src_structure,
            storage_conjugate,
            || self.coefficients(),
        )
    }

    pub fn compile_shared_structures_with_storage_mapping<FBlock, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        logical_src_structure: &BlockStructure,
        storage_src_structure: Arc<BlockStructure>,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        bind_with_storage_mapping(
            self.specs(),
            dst_structure,
            logical_src_structure,
            storage_src_structure,
            logical_to_storage_block,
            logical_to_storage_axis,
            storage_conjugate,
            || self.coefficients(),
        )
    }

    pub fn compile_shared_structures_with_source_projection<FSource, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        storage_src_structure: Arc<BlockStructure>,
        logical_rank: usize,
        source_index: FSource,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<TreeTransformStructure<T>, OperationError>
    where
        FSource: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        bind_with_source_projection(
            self.specs(),
            dst_structure,
            storage_src_structure,
            logical_rank,
            source_index,
            logical_to_storage_axis,
            storage_conjugate,
            || self.coefficients(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtn_core::BlockSpec;

    fn tree_pair(vertex: usize) -> FusionTreePairKey {
        FusionTreePairKey::try_pair_from_sector_ids(
            [0, 0],
            [],
            0,
            [false, false],
            [],
            [],
            [],
            [vertex],
            [],
        )
        .unwrap()
    }

    fn structure_with_rank(
        keys: &[FusionTreePairKey],
        elements: usize,
        rank: usize,
    ) -> BlockStructure {
        BlockStructure::from_blocks_with_rank(
            rank,
            keys.iter()
                .enumerate()
                .map(|(index, key)| {
                    BlockSpec::column_major_with_key(
                        BlockKey::from(key.clone()),
                        std::iter::once(elements)
                            .chain(std::iter::repeat_n(1, rank - 1))
                            .collect(),
                        index * elements,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn structure(keys: &[FusionTreePairKey], elements: usize) -> BlockStructure {
        structure_with_rank(keys, elements, 2)
    }

    fn matrix_plan(keys: &[FusionTreePairKey; 2]) -> TreeTransformGroupPlan<f64> {
        TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::try_multi(
            keys.clone(),
            keys.clone(),
            vec![1.0_f64, 2.0, 3.0, 4.0],
        )
        .unwrap()])
    }

    #[test]
    fn categorical_coefficients_are_shared_across_layout_bindings() {
        // What: one categorical plan binds to different degeneracy layouts
        // without copying its recoupling matrices into either replay plan.
        let keys = [tree_pair(1), tree_pair(2)];
        let plan = matrix_plan(&keys);
        let first_layout = structure(&keys, 2);
        let second_layout = structure(&keys, 3);

        let first = plan
            .compile_structures(&first_layout, &first_layout)
            .unwrap();
        let second = plan
            .compile_structures(&second_layout, &second_layout)
            .unwrap();

        assert!(first.shares_recoupling_matrices_with(&second));
        assert_eq!(
            first.block_coefficients(0).unwrap().as_ptr(),
            plan.specs()[0].recoupling_coefficients_dst_src().as_ptr()
        );
        assert_eq!(first.gathered_coefficients(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(second.gathered_coefficients(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn concurrent_cold_bindings_share_the_spec_matrices() {
        // What: racing cold layout bindings both reference the spec's matrix
        // instead of each retaining its own copy.
        let keys = [tree_pair(1), tree_pair(2)];
        let plan = matrix_plan(&keys);
        let first_layout = structure(&keys, 2);
        let second_layout = structure(&keys, 3);
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let (first, second) = std::thread::scope(|scope| {
            let first_barrier = Arc::clone(&barrier);
            let first_plan = &plan;
            let first_layout = &first_layout;
            let first = scope.spawn(move || {
                first_barrier.wait();
                first_plan
                    .compile_structures(first_layout, first_layout)
                    .unwrap()
            });
            let second_barrier = Arc::clone(&barrier);
            let second_plan = &plan;
            let second_layout = &second_layout;
            let second = scope.spawn(move || {
                second_barrier.wait();
                second_plan
                    .compile_structures(second_layout, second_layout)
                    .unwrap()
            });
            (first.join().unwrap(), second.join().unwrap())
        });

        assert!(first.shares_recoupling_matrices_with(&second));
    }

    #[test]
    fn debug_output_is_independent_of_binding_and_spec_kind() {
        // What: Debug of a plan is its categorical specs, unchanged by binding,
        // and a grouped binding prints like the equivalent direct binding.
        let keys = [tree_pair(1), tree_pair(2)];
        let plan = matrix_plan(&keys);
        let layout = structure(&keys, 2);
        let wrong_rank = structure_with_rank(&keys, 2, 3);
        let before = format!("{plan:?}");

        let error = plan.compile_structures(&layout, &wrong_rank).unwrap_err();
        assert!(matches!(
            error,
            OperationError::StructureRankMismatch {
                expected: 2,
                actual: 3
            }
        ));
        assert_eq!(format!("{plan:?}"), before);

        let shared = plan.compile_structures(&layout, &layout).unwrap();
        assert_eq!(format!("{plan:?}"), before);
        let owned = TreeTransformStructure::compile_structures(
            &layout,
            &layout,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 2.0, 3.0, 4.0],
            )],
        )
        .unwrap();
        let shared_debug = format!("{shared:?}");
        assert_eq!(shared_debug, format!("{owned:?}"));
    }
}
