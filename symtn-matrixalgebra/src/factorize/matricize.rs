use super::*;

/// A packed coupled-sector matricization: the test oracle for publication
/// over borrowed regions.
#[cfg(test)]
pub(super) struct SectorMatricization<D> {
    pub(super) sector: SectorId,
    pub(super) rows: usize,
    pub(super) cols: usize,
    /// (codomain tree, row offset, codomain degeneracy shape)
    pub(super) row_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// (domain tree, column offset, domain degeneracy shape)
    pub(super) col_trees: Vec<(FusionTreeKey, usize, Vec<usize>)>,
    /// Column-major `rows x cols` matrix.
    pub(super) data: Vec<D>,
}

#[derive(Clone, Copy)]
pub(super) struct TreeExtentRef<'a> {
    pub(super) tree: &'a FusionTreeKey,
    pub(super) offset: usize,
    pub(super) shape: &'a [usize],
}

// Keeps publication generic over owned packs and cached regions without
// allocating a second list of borrowed tree descriptors.
pub(super) trait SectorGeometry {
    fn sector(&self) -> SectorId;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn tree_count(&self, side: FactorSide) -> usize;
    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>>;
}

#[cfg(test)]
impl<D> SectorGeometry for SectorMatricization<D> {
    fn sector(&self) -> SectorId {
        self.sector
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn tree_count(&self, side: FactorSide) -> usize {
        match side {
            FactorSide::Left => self.row_trees.len(),
            FactorSide::Right => self.col_trees.len(),
        }
    }

    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
        let trees = match side {
            FactorSide::Left => &self.row_trees,
            FactorSide::Right => &self.col_trees,
        };
        trees.get(index).map(|(tree, offset, shape)| TreeExtentRef {
            tree,
            offset: *offset,
            shape,
        })
    }
}

impl SectorGeometry for CoupledSectorRegion {
    fn sector(&self) -> SectorId {
        self.coupled()
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn tree_count(&self, side: FactorSide) -> usize {
        match side {
            FactorSide::Left => self.row_trees().len(),
            FactorSide::Right => self.col_trees().len(),
        }
    }

    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
        let extent = match side {
            FactorSide::Left => self.row_trees().get(index),
            FactorSide::Right => self.col_trees().get(index),
        }?;
        Some(TreeExtentRef {
            tree: extent.tree(),
            offset: extent.offset(),
            shape: extent.shape(),
        })
    }
}

pub(super) struct SectorMatrixRef<'a, D> {
    pub(super) sector: SectorId,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) data: &'a [D],
}

/// Indexed access to the per-sector matrices a factorization stage reads,
/// borrowed regions or packed copies alike.
pub(super) trait SectorMatrices<D> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError>;
}

impl<D: FactorScalar> SectorMatrices<D> for InputMatricizations<'_, D> {
    fn len(&self) -> usize {
        InputMatricizations::len(self)
    }

    fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError> {
        InputMatricizations::get(self, index)
    }
}

#[cfg(test)]
impl<D> SectorMatrices<D> for [SectorMatricization<D>] {
    fn len(&self) -> usize {
        <[SectorMatricization<D>]>::len(self)
    }

    fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError> {
        let matrix = &self[index];
        Ok(SectorMatrixRef {
            sector: matrix.sector,
            rows: matrix.rows,
            cols: matrix.cols,
            data: &matrix.data,
        })
    }
}

/// The input's coupled-sector matrices, borrowed in place from its
/// canonical layout.
pub(super) struct InputMatricizations<'a, D> {
    pub(super) data: &'a [D],
    pub(super) regions: Arc<[CoupledSectorRegion]>,
}

impl<'a, D: FactorScalar> InputMatricizations<'a, D> {
    pub(super) fn len(&self) -> usize {
        self.regions.len()
    }

    pub(super) fn get(&self, index: usize) -> Result<SectorMatrixRef<'_, D>, OperationError> {
        let region = &self.regions[index];
        let range = region.range();
        let matrix = self
            .data
            .get(range.clone())
            .ok_or(OperationError::ElementCountMismatch {
                expected: range.end,
                actual: self.data.len(),
            })?;
        Ok(SectorMatrixRef {
            sector: region_sector(region),
            rows: region.rows(),
            cols: region.cols(),
            data: matrix,
        })
    }

    /// Eigenvalues are basis-invariant only when row `i` and column `i` name
    /// the same tree state, so each sector must stack its row and column trees
    /// identically (outer-multiplicity vertices included).
    pub(super) fn validate_endomorphism_stacking(
        &self,
        family: EndomorphismFamily,
    ) -> Result<(), OperationError> {
        validate_endomorphism_tree_stacking(self.regions.as_ref(), family)
    }

    pub(super) fn validate_hermitian(
        &self,
        hermitian_tol: HermitianTol,
    ) -> Result<(), OperationError> {
        validate_hermitian_regions(self.data, &self.regions, hermitian_tol)
    }
}

pub(super) fn coupled_of(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

pub(super) fn matricization_map<M: SectorGeometry>(
    matricizations: &[M],
) -> FxHashMap<SectorId, &M> {
    matricizations
        .iter()
        .map(|matrix| (matrix.sector(), matrix))
        .collect()
}

pub(super) fn matricization_of<'a, M>(
    matricizations: &FxHashMap<SectorId, &'a M>,
    sector: SectorId,
) -> Result<&'a M, OperationError> {
    matricizations
        .get(&sector)
        .copied()
        .ok_or(OperationError::SpaceMismatch {
            message: "factor tree references a coupled sector absent from the source tensor",
        })
}

pub(super) fn validate_dense_shape(
    actual: &[usize],
    expected: &[usize],
) -> Result<(), OperationError> {
    if actual != expected {
        return Err(OperationError::ShapeMismatch {
            dst: expected.to_vec(),
            src: actual.to_vec(),
        });
    }
    Ok(())
}

pub(super) fn data_region<'a, D>(
    data: &'a [D],
    range: &std::ops::Range<usize>,
) -> Result<&'a [D], OperationError> {
    data.get(range.clone())
        .ok_or(OperationError::ElementCountMismatch {
            expected: range.end,
            actual: data.len(),
        })
}

pub(super) fn checked_sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    structure
        .coupled_sector_regions(nout)
        .map_err(OperationError::from_core_preserving_context)
}

/// The coupled-sector matrices of an input, borrowed from its regions.
///
/// A region lists its sector and tree extents in first-appearance block order,
/// which is the canonical tree order, so the factor outputs (also canonical)
/// list the same trees in the same order, and the eigenvalue ops check
/// endomorphism stacking. Why no tree-order admission (`FusionTreeKey` Ord):
/// canonical multi-leg order is the fusion walk's, not that `Ord`.
pub(super) fn input_matricizations<'a, D>(
    structure: &BlockStructure,
    data: &'a [D],
    nout: usize,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    Ok(InputMatricizations {
        data,
        regions: input_regions(structure, nout)?,
    })
}

/// The coupled-sector regions of a dynamic space's layout.
///
/// Why an internal error, not a packed fallback: every dynamic space is its
/// HomSpace's canonical coupled-sector layout, which the region compiler
/// always admits, so a missing region table is a broken layout invariant.
pub(super) fn input_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Arc<[CoupledSectorRegion]>, OperationError> {
    let regions = checked_sector_regions(structure, nout)?;
    debug_assert!(regions.is_some(), "dynamic space layout is not canonical");
    regions.ok_or(OperationError::InvalidArgument {
        message: "internal layout invariant violated (not a coupled-sector matrix layout); please report this",
    })
}

// ============================================================================
// Generic-fusion (SU(N)) siblings.
//
// Parallel `*_generic` siblings of the mult-free factorization entry points.
// The block-level engine — the dense SVD/QR per coupled sector, the gauges,
// the workspace scatters, `diagonal_bond_data`, and every copy helper — is
// symmetry-agnostic and SHARED. A sibling differs from its original in
// exactly three substitutions:
//   1. bound: `MultiplicityFreeRigidSymbols` -> `FusionRule` (or
//      `GenericRigidSymbols` where the truncation weight needs rigid data);
//   2. key enumeration: `fusion_tree_keys` -> `fusion_tree_keys_generic` and
//      `from_final_homspace` -> `from_final_homspace_generic`, so the
//      factor spaces carry multiplicity-aware (vertex-labelled) trees — the
//      matricization already stacks ALL trees of a coupled sector into one
//      dense block (TensorKit `block(t, c)`), so outer multiplicity rides the
//      row/col tree lists with no math change;
//   3. truncation dim weight: `dim_scalar(c)` -> the checked provider's
//      `try_dim_scalar(c)`, both through `RigidCoefficientAlgebra::dim`.
// Duplicated rather than bound-relaxed so the mult-free path stays
// byte-for-byte untouched (the B-series byte-invariance rule).
// ============================================================================

pub(super) fn coupled_of_generic(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}

/// Fallible coupled-sector reduced dimensions for checked Generic providers:
/// core's [`FusionProductSpace::coupled_sector_block_dimensions_checked`] with
/// this crate's error wrapping.
#[doc(hidden)]
pub fn coupled_sector_block_dimensions_generic_checked<R>(
    product: &FusionProductSpace,
    rule: &R,
) -> Result<BTreeMap<SectorId, usize>, CheckedGenericFactorPlanError<R::Error>>
where
    R: CheckedGenericFusion,
{
    product
        .coupled_sector_block_dimensions_checked(rule)
        .map_err(|error| match error {
            CheckedGenericStructureError::Provider(error) => {
                CheckedGenericFactorPlanError::Provider(error)
            }
            CheckedGenericStructureError::Core(CoreError::ElementCountOverflow) => {
                CheckedGenericFactorPlanError::Operation(OperationError::ElementCountOverflow)
            }
            CheckedGenericStructureError::Core(error) => CheckedGenericFactorPlanError::Operation(
                OperationError::from_core_preserving_context(error),
            ),
        })
}

/// Row `i` and column `i` of every sector name the same tree state (tree key
/// including outer-multiplicity vertices, offset and shape), so the block is
/// an endomorphism matrix in one basis and its spectrum is basis-invariant.
/// Generic over [`SectorGeometry`] so packed matricizations and borrowed
/// regions share this one predicate.
pub(super) fn endomorphism_tree_stacking_is_identical<M: SectorGeometry>(matrices: &[M]) -> bool {
    matrices.iter().all(|matrix| {
        let count = matrix.tree_count(FactorSide::Left);
        matrix.rows() == matrix.cols()
            && count == matrix.tree_count(FactorSide::Right)
            && (0..count).all(|index| {
                match (
                    matrix.tree(FactorSide::Left, index),
                    matrix.tree(FactorSide::Right, index),
                ) {
                    (Some(row), Some(col)) => {
                        row.tree == col.tree && row.offset == col.offset && row.shape == col.shape
                    }
                    _ => false,
                }
            })
    })
}

/// `eigh` refusal of a block that is not Hermitian at the admitted tolerance.
#[doc(hidden)]
pub const EIGH_NOT_HERMITIAN: &str = "eigh requires Hermitian coupled-sector blocks";

/// `eigh` refusal of a non-square coupled-sector block.
#[doc(hidden)]
pub const EIGH_NOT_SQUARE: &str = "eigh requires square coupled-sector matrices";

/// An operation defined only on endomorphisms. The one owner of its
/// endomorphism refusal and stacking-invariant texts, for every storage.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndomorphismFamily {
    EighFull,
    EighVals,
    EigFull,
    EigVals,
    Exp,
}

impl EndomorphismFamily {
    const fn not_endomorphism(self) -> &'static str {
        match self {
            Self::EighFull | Self::EighVals => "eigh requires an endomorphism (codomain == domain)",
            Self::EigFull | Self::EigVals => "eig requires an endomorphism (codomain == domain)",
            Self::Exp => "exp requires an endomorphism (codomain == domain)",
        }
    }

    /// Why an internal error, not a refusal: an endomorphism's canonical
    /// layout stacks row and column trees identically, so a mismatch is a
    /// broken layout invariant.
    const fn stacking(self) -> &'static str {
        match self {
            Self::EighFull => "internal eigh_full layout invariant violated (row/column tree stacking differs); please report this",
            Self::EighVals => "internal eigh_vals layout invariant violated (row/column tree stacking differs); please report this",
            Self::EigFull => "internal eig_full layout invariant violated (row/column tree stacking differs); please report this",
            Self::EigVals => "internal eig_vals layout invariant violated (row/column tree stacking differs); please report this",
            Self::Exp => "internal exp layout invariant violated (row/column tree stacking differs); please report this",
        }
    }

    /// TensorKit's `domain(t) == codomain(t)` precondition, asked first.
    pub fn require_endomorphism(self, homspace: &FusionTreeHomSpace) -> Result<(), OperationError> {
        if homspace.codomain() == homspace.domain() {
            Ok(())
        } else {
            Err(OperationError::SpaceMismatch {
                message: self.not_endomorphism(),
            })
        }
    }
}

/// The endomorphism admission of every Host eigenvalue-type operation:
/// endomorphism, then `matricize`, then identical row/column stacking.
pub(super) fn admit_endomorphism<'a, D>(
    space: &DynamicFusionMapSpace,
    data: &'a [D],
    family: EndomorphismFamily,
    matricize: impl FnOnce(
        &BlockStructure,
        &'a [D],
        usize,
    ) -> Result<InputMatricizations<'a, D>, OperationError>,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    family.require_endomorphism(space.homspace())?;
    admit_endomorphism_stacking(space, data, family, matricize)
}

/// [`admit_endomorphism`] after the caller has already asked
/// [`EndomorphismFamily::require_endomorphism`] (for a route taken between
/// the two).
pub(super) fn admit_endomorphism_stacking<'a, D>(
    space: &DynamicFusionMapSpace,
    data: &'a [D],
    family: EndomorphismFamily,
    matricize: impl FnOnce(
        &BlockStructure,
        &'a [D],
        usize,
    ) -> Result<InputMatricizations<'a, D>, OperationError>,
) -> Result<InputMatricizations<'a, D>, OperationError>
where
    D: FactorScalar,
{
    let matricizations = matricize(space.structure(), data, space.nout())?;
    matricizations.validate_endomorphism_stacking(family)?;
    Ok(matricizations)
}

/// [`validate_endomorphism_tree_stacking`] over canonical coupled-sector
/// regions, for device paths outside this crate.
#[doc(hidden)]
pub fn validate_endomorphism_region_stacking(
    regions: &[CoupledSectorRegion],
    family: EndomorphismFamily,
) -> Result<(), OperationError> {
    validate_endomorphism_tree_stacking(regions, family)
}

pub(super) fn validate_endomorphism_tree_stacking<M: SectorGeometry>(
    matrices: &[M],
    family: EndomorphismFamily,
) -> Result<(), OperationError> {
    if endomorphism_tree_stacking_is_identical(matrices) {
        Ok(())
    } else {
        Err(OperationError::InvalidArgument {
            message: family.stacking(),
        })
    }
}

#[cfg(test)]
/// Packs every coupled sector of the source data into its dense column-major
/// matricization, independent of the storage layout. Core's
/// [`CoupledMatricizationBuilder`] owns the sector, row and column placement;
/// this only records the trees and moves the data.
pub(super) fn sector_matricizations<D>(
    structure: &BlockStructure,
    data: &[D],
    nout: usize,
) -> Result<Vec<SectorMatricization<D>>, OperationError>
where
    D: FactorScalar,
{
    let mut layout = symtn_core::CoupledMatricizationBuilder::new();
    let mut matricizations: Vec<SectorMatricization<D>> = Vec::new();
    let mut routes = Vec::with_capacity(structure.block_count());

    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "tsvd",
                index,
            });
        };
        let (row_shape, col_shape) = block.shape().split_at(nout);
        let overflow = |_| OperationError::ElementCountOverflow;
        let placement = layout
            .place(
                key,
                symtn_core::checked_product(row_shape).map_err(overflow)?,
                symtn_core::checked_product(col_shape).map_err(overflow)?,
            )
            .map_err(overflow)?;
        if placement.new_sector {
            matricizations.push(SectorMatricization::<D> {
                sector: key.codomain_tree().coupled(),
                rows: 0,
                cols: 0,
                row_trees: Vec::new(),
                col_trees: Vec::new(),
                data: Vec::new(),
            });
        }
        let matrix = &mut matricizations[placement.sector];
        if placement.new_row {
            matrix.row_trees.push((
                key.codomain_tree().clone(),
                placement.row_offset,
                row_shape.to_vec(),
            ));
        }
        if placement.new_col {
            matrix.col_trees.push((
                key.domain_tree().clone(),
                placement.col_offset,
                col_shape.to_vec(),
            ));
        }
        routes.push((placement.sector, placement.row_offset, placement.col_offset));
    }
    for matrix in &mut matricizations {
        (matrix.rows, matrix.cols) = layout.extents(matrix.sector);
        let len = matrix
            .rows
            .checked_mul(matrix.cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        matrix.data = vec![D::zero(); len];
    }

    for (index, (matrix_index, row_offset, col_offset)) in routes.into_iter().enumerate() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let matrix = &mut matricizations[matrix_index];
        let rows = matrix.rows;
        copy_tensor_block_to_matrix(
            data,
            block.shape(),
            block.strides(),
            block.offset(),
            nout,
            &mut matrix.data,
            rows,
            row_offset,
            col_offset,
        );
    }
    #[cfg(test)]
    crate::probe::input_pack_bytes(
        matricizations
            .iter()
            .map(|matrix| std::mem::size_of_val(&matrix.data[..])),
    );
    Ok(matricizations)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn copy_tensor_block_to_matrix<D: Copy>(
    source: &[D],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    nout: usize,
    matrix: &mut [D],
    matrix_rows: usize,
    row_offset: usize,
    col_offset: usize,
) {
    if shape.is_empty() {
        matrix[row_offset + matrix_rows * col_offset] = source[offset];
        return;
    }
    let run = shape[0];
    let src_lane_stride = strides[0];
    let dst_lane_stride = if nout > 0 { 1 } else { matrix_rows };
    let outer_count: usize = shape[1..].iter().product();
    let mut index = vec![0usize; shape.len()];
    for _ in 0..outer_count {
        let mut src_start = offset;
        let mut row = 0usize;
        let mut row_stride = if nout > 0 { shape[0] } else { 1 };
        let mut col = 0usize;
        let mut col_stride = if nout == 0 { shape[0] } else { 1 };
        for axis in 1..shape.len() {
            src_start += index[axis] * strides[axis];
            if axis < nout {
                row += index[axis] * row_stride;
                row_stride *= shape[axis];
            } else {
                col += index[axis] * col_stride;
                col_stride *= shape[axis];
            }
        }
        let dst_start = (row_offset + row) + matrix_rows * (col_offset + col);
        if src_lane_stride == 1 && dst_lane_stride == 1 {
            matrix[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                matrix[dst_start + lane * dst_lane_stride] =
                    source[src_start + lane * src_lane_stride];
            }
        }
        advance_outer_index(&mut index, shape);
    }
}
