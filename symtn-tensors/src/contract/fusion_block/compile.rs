use super::*;

fn compile_direct_coupled_region_plan<C>(
    dst_space: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
) -> Result<Option<FusionBlockContractPlan<C>>, OperationError>
where
    C: DenseBlockScalar,
{
    FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_generic(
        dst_space.structure(),
        dst_space.nout(),
        lhs_storage.structure(),
        lhs_storage.nout(),
        rhs_storage.structure(),
        rhs_storage.nout(),
        lhs_op,
        rhs_op,
    )
}

fn compile_scaled_direct_coupled_region_plan<C>(
    dst_space: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    alpha_by_coupled: &[(SectorId, C)],
) -> Result<Option<FusionBlockContractPlan<C>>, OperationError>
where
    C: DenseBlockScalar,
{
    FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_and_alpha_generic(
        dst_space.structure(),
        dst_space.nout(),
        lhs_storage.structure(),
        lhs_storage.nout(),
        rhs_storage.structure(),
        rhs_storage.nout(),
        lhs_op,
        rhs_op,
        |coupled| {
            alpha_by_coupled
                .binary_search_by_key(&coupled, |&(sector, _)| sector)
                .map(|index| alpha_by_coupled[index].1)
                .map_err(|_| OperationError::UnsupportedTensorContractScope {
                    message:
                        "canonical storage plan is missing a verified coupled-sector coefficient",
                })
        },
    )
}

pub(crate) fn try_compile_oriented_canonical_core_plan<R, C>(
    validated: &ValidatedCoreContract<'_, R>,
    dst_space: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
) -> Result<Option<FusionBlockContractPlan<C>>, OperationError>
where
    C: DenseBlockScalar,
{
    let matrix_op = |orientation| match orientation {
        FusionTreePairOrientation::Direct => MatrixOp::Identity,
        FusionTreePairOrientation::Adjoint => MatrixOp::Adjoint,
    };
    compile_direct_coupled_region_plan(
        dst_space,
        lhs_storage,
        rhs_storage,
        matrix_op(validated.preflight.lhs_homspace.orientation()),
        matrix_op(validated.preflight.rhs_homspace.orientation()),
    )
}

pub(crate) fn try_compile_scaled_canonical_core_plan<R, C>(
    validated: &ValidatedCoreContract<'_, R>,
    dst_space: &DynamicFusionMapSpace,
    lhs_storage: &DynamicFusionMapSpace,
    rhs_storage: &DynamicFusionMapSpace,
    alpha_by_coupled: &[(SectorId, C)],
) -> Result<Option<FusionBlockContractPlan<C>>, OperationError>
where
    C: DenseBlockScalar,
{
    let matrix_op = |orientation| match orientation {
        FusionTreePairOrientation::Direct => MatrixOp::Identity,
        FusionTreePairOrientation::Adjoint => MatrixOp::Adjoint,
    };
    compile_scaled_direct_coupled_region_plan(
        dst_space,
        lhs_storage,
        rhs_storage,
        matrix_op(validated.preflight.lhs_homspace.orientation()),
        matrix_op(validated.preflight.rhs_homspace.orientation()),
        alpha_by_coupled,
    )
}

#[cfg(test)]
pub(crate) fn compile_fusion_block_contract_plan<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionBlockContractPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    validate_fusion_contract_rule(rule, dst_space, lhs_space, rhs_space)?;
    reject_fusion_contract_conjugation(axes)?;
    let validated = CoreContractPreflight::compile_homspaces(
        rule,
        dst_space.homspace(),
        lhs_space.homspace(),
        rhs_space.homspace(),
        axes,
    )?
    .require_core_geometry()?;
    compile_fusion_block_contract_plan_core_geometry(
        validated.preflight.rule,
        dst_space,
        lhs_space,
        rhs_space,
    )
}

/// The coupled block plan of three spaces whose core geometry is already
/// established: by a [`ValidatedCoreContract`], or by construction for the
/// core operands a [`super::fusion::FusionContractPlan`] derives.
pub(crate) fn compile_fusion_block_contract_plan_core_geometry<R>(
    rule: &R,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    rhs_space: &DynamicFusionMapSpace,
) -> Result<FusionBlockContractPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    if let Some(plan) = compile_direct_coupled_region_plan(
        dst_space,
        lhs_space,
        rhs_space,
        MatrixOp::Identity,
        MatrixOp::Identity,
    )? {
        return Ok(plan);
    }

    let lhs_layout = FusionBlockMatrixLayout::compile(lhs_space)?;
    let rhs_layout = FusionBlockMatrixLayout::compile(rhs_space)?;
    let dst_layout = FusionBlockMatrixLayout::compile(dst_space)?;

    let (groups, active_dst_blocks) = pair_coupled_groups(
        lhs_layout,
        rhs_layout,
        dst_layout,
        |group| group.finish(rule, lhs_space),
        |group| group.finish(rule, rhs_space),
        |group| group.finish(rule, dst_space),
    )?;
    FusionBlockContractPlan::from_parts_generic(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs_space.structure()),
        Arc::clone(rhs_space.structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
    )
}

pub(crate) fn compile_fusion_block_contract_plan_prelowered_validated<R>(
    validated: ValidatedCoreContract<'_, R>,
    dst_space: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
) -> Result<FusionBlockContractPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    if let Some(plan) = try_compile_oriented_canonical_core_plan(
        &validated,
        dst_space,
        lhs.storage_space(),
        rhs.storage_space(),
    )? {
        return Ok(plan);
    }
    compile_operand_core_plan_general(dst_space, lhs, rhs)
}

/// The packed core plan of two operands whose coupled-sector tiling is not
/// canonical (#1517), a lazy adjoint read through its parent's storage with
/// [`MatrixOp::Adjoint`]. Structure-only: the core glues `lhs.domain` to
/// `rhs.codomain` without crossing legs, so every coefficient is 1 in both
/// modes.
pub(crate) fn compile_operand_core_plan_general<C>(
    dst_space: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
) -> Result<FusionBlockContractPlan<C>, OperationError>
where
    C: DenseBlockScalar,
{
    let matrix_op = |source: &FusionOperandLayout<'_>| match source.orientation() {
        FusionTreePairOrientation::Direct => MatrixOp::Identity,
        FusionTreePairOrientation::Adjoint => MatrixOp::Adjoint,
    };
    let (lhs_op, rhs_op) = (matrix_op(lhs), matrix_op(rhs));
    let compile_source = |source: &FusionOperandLayout<'_>| {
        if source.is_direct() {
            FusionBlockMatrixLayout::compile(source.storage_space())
        } else {
            FusionBlockMatrixLayout::compile_operand(source)
        }
    };
    let finish_source =
        |group: FusionBlockMatrixGroupBuilder, source: &FusionOperandLayout<'_>, op| {
            if source.is_direct() {
                let storage = source.storage_space();
                group.finish_generic(storage.structure(), storage.nout())
            } else {
                group.finish_operand_generic(source, op)
            }
        };
    let lhs_layout = compile_source(lhs)?;
    let rhs_layout = compile_source(rhs)?;
    let dst_layout = FusionBlockMatrixLayout::compile(dst_space)?;

    let (groups, active_dst_blocks) = pair_coupled_groups(
        lhs_layout,
        rhs_layout,
        dst_layout,
        |group| finish_source(group, lhs, lhs_op),
        |group| finish_source(group, rhs, rhs_op),
        |group| group.finish_generic(dst_space.structure(), dst_space.nout()),
    )?;
    FusionBlockContractPlan::from_parts_with_ops_generic(
        Arc::clone(dst_space.structure()),
        Arc::clone(lhs.storage_space().structure()),
        Arc::clone(rhs.storage_space().structure()),
        fusion_scale_block_layouts_excluding(dst_space.structure(), &active_dst_blocks)?,
        groups,
        lhs_op,
        rhs_op,
    )
}

/// The per-subblock checked core builder: re-bases expert tilings whose
/// coupled-sector trees are not canonically aligned, or reports the same
/// structure error as before the coupled-region route.
pub(crate) fn compile_checked_generic_core_plan_general(
    dst_structure: &Arc<symtn_core::BlockStructure>,
    dst_nout: usize,
    lhs_structure: &Arc<symtn_core::BlockStructure>,
    lhs_nout: usize,
    rhs_structure: &Arc<symtn_core::BlockStructure>,
    rhs_nout: usize,
) -> Result<FusionBlockContractPlan, OperationError> {
    let lhs_layout = FusionBlockMatrixLayout::compile_parts(lhs_structure, lhs_nout)?;
    let rhs_layout = FusionBlockMatrixLayout::compile_parts(rhs_structure, rhs_nout)?;
    let dst_layout = FusionBlockMatrixLayout::compile_parts(dst_structure, dst_nout)?;
    let (groups, active_dst_blocks) = pair_coupled_groups(
        lhs_layout,
        rhs_layout,
        dst_layout,
        |group| group.finish_generic(lhs_structure, lhs_nout),
        |group| group.finish_generic(rhs_structure, rhs_nout),
        |group| group.finish_generic(dst_structure, dst_nout),
    )?;
    FusionBlockContractPlan::from_parts(
        Arc::clone(dst_structure),
        Arc::clone(lhs_structure),
        Arc::clone(rhs_structure),
        fusion_scale_block_layouts_excluding(dst_structure, &active_dst_blocks)?,
        groups,
    )
}

#[cfg(test)]
/// Host implementation of [`StorageGemm`] over host-readable storages.
pub(crate) struct HostStorageGemm<'a, B, W> {
    engine: &'a mut dyn Engine,
    backend: &'a mut B,
    workspace: &'a mut W,
}

#[cfg(test)]
impl<'a, B, W> HostStorageGemm<'a, B, W> {
    pub(crate) fn new(
        engine: &'a mut dyn Engine,
        backend: &'a mut B,
        workspace: &'a mut W,
    ) -> Self {
        Self {
            engine,
            backend,
            workspace,
        }
    }
}

#[cfg(test)]
impl<'a, B, D, DDst, DLhs, DRhs> StorageGemm<D, DDst, DLhs, DRhs>
    for HostStorageGemm<'a, B, B::Workspace>
where
    B: TensorContractBackend<D, f64>,
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
    DDst: HostWritableStorage<D>,
    DLhs: HostReadableStorage<D>,
    DRhs: HostReadableStorage<D>,
{
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
    }

    fn matmul_range_into(
        &mut self,
        dst: &mut DDst,
        dst_offset: usize,
        lhs: &DLhs,
        lhs_offset: usize,
        rhs: &DRhs,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        let dst_len = rows * cols;
        let lhs_len = rows * contracted;
        let rhs_len = contracted * cols;
        self.backend.matmul_rank2_into_raw(
            &mut *self.engine,
            self.workspace,
            &mut dst.as_mut_slice()[dst_offset..dst_offset + dst_len],
            &lhs.as_slice()[lhs_offset..lhs_offset + lhs_len],
            &rhs.as_slice()[rhs_offset..rhs_offset + rhs_len],
            rows,
            contracted,
            cols,
        )
    }

    fn matmul_range_with_ops_scaled_into(
        &mut self,
        dst: &mut DDst,
        dst_offset: usize,
        lhs: &DLhs,
        lhs_offset: usize,
        rhs: &DRhs,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
    ) -> Result<(), OperationError> {
        if lhs_op != MatrixOp::Identity || rhs_op != MatrixOp::Identity {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "host storage GEMM does not implement transformed operands",
            });
        }
        let dst_len = rows * cols;
        let lhs_len = rows * contracted;
        let rhs_len = contracted * cols;
        self.backend.matmul_rank2_axpby_into_raw(
            &mut *self.engine,
            self.workspace,
            &mut dst.as_mut_slice()[dst_offset..dst_offset + dst_len],
            &lhs.as_slice()[lhs_offset..lhs_offset + lhs_len],
            &rhs.as_slice()[rhs_offset..rhs_offset + rhs_len],
            rows,
            contracted,
            cols,
            alpha,
            D::zero(),
        )
    }
}

/// One operand's coupled-sector groups with tree offsets collected but not
/// yet finished, so [`pair_coupled_groups`] can give every shared GEMM index
/// one tree order before any subblock offset is fixed.
#[derive(Clone, Debug)]
pub(super) struct FusionBlockMatrixLayout {
    groups: Vec<Option<FusionBlockMatrixGroupBuilder>>,
    // Why not sort for a merge walk: expert block layouts retain first
    // destination occurrence order, while this compile-time map preserves it.
    group_indices: FxHashMap<SectorId, usize>,
}

impl FusionBlockMatrixLayout {
    pub(super) fn compile(space: &DynamicFusionMapSpace) -> Result<Self, OperationError> {
        Self::compile_parts(space.structure(), space.nout())
    }

    pub(super) fn compile_operand(
        source: &FusionOperandLayout<'_>,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::FusionLayoutCompile);
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = FxHashMap::<SectorId, usize>::default();
        let storage = source.storage_space();
        for logical_index in 0..source.logical_block_count() {
            let key = source.logical_key(logical_index)?;
            let storage_index = source.storage_index(logical_index)?;
            let block = storage.structure().block(storage_index)?;
            let coupled = coupled_sector(key.codomain_tree());
            if coupled != coupled_sector(key.domain_tree()) {
                return Err(OperationError::FusionTreeGroupMismatch {
                    tensor: "fusion",
                    index: logical_index,
                });
            }
            let group_index = if let Some(&group_index) = group_indices.get(&coupled) {
                group_index
            } else {
                let group_index = builders.len();
                group_indices.insert(coupled, group_index);
                builders.push(FusionBlockMatrixGroupBuilder::new(coupled));
                group_index
            };
            let split = storage.nout();
            let (row_shape, col_shape) = match source.orientation() {
                FusionTreePairOrientation::Direct => {
                    (&block.shape()[..split], &block.shape()[split..])
                }
                FusionTreePairOrientation::Adjoint => {
                    (&block.shape()[split..], &block.shape()[..split])
                }
            };
            builders[group_index].add_tree_pair_mapped(
                key.codomain_tree().clone(),
                element_count(row_shape)?,
                key.domain_tree().clone(),
                element_count(col_shape)?,
                logical_index,
                storage_index,
            )?;
        }
        Ok(Self::from_builders(builders, group_indices))
    }

    fn compile_parts(
        structure: &Arc<symtn_core::BlockStructure>,
        nout: usize,
    ) -> Result<Self, OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::FusionLayoutCompile);
        let mut builders = Vec::<FusionBlockMatrixGroupBuilder>::new();
        let mut group_indices = FxHashMap::<SectorId, usize>::default();
        for block_index in 0..structure.block_count() {
            let block = structure.block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let coupled = coupled_sector(key.codomain_tree());
            if coupled != coupled_sector(key.domain_tree()) {
                return Err(OperationError::FusionTreeGroupMismatch {
                    tensor: "fusion",
                    index: block_index,
                });
            }
            let group_index = if let Some(&group_index) = group_indices.get(&coupled) {
                group_index
            } else {
                let group_index = builders.len();
                group_indices.insert(coupled, group_index);
                builders.push(FusionBlockMatrixGroupBuilder::new(coupled));
                group_index
            };
            let row_dim = element_count(&block.shape()[..nout])?;
            let col_dim = element_count(&block.shape()[nout..])?;
            builders[group_index].add_tree_pair(
                key.codomain_tree().clone(),
                row_dim,
                key.domain_tree().clone(),
                col_dim,
                block_index,
            )?;
        }
        Ok(Self::from_builders(builders, group_indices))
    }

    fn from_builders(
        builders: Vec<FusionBlockMatrixGroupBuilder>,
        group_indices: FxHashMap<SectorId, usize>,
    ) -> Self {
        Self {
            groups: builders.into_iter().map(Some).collect(),
            group_indices,
        }
    }

    #[cfg(test)]
    pub(super) fn finish_all<C>(
        self,
        finish: impl FnMut(
            FusionBlockMatrixGroupBuilder,
        ) -> Result<FusionBlockMatrixGroup<C>, OperationError>,
    ) -> FinishedLayout<C> {
        FinishedLayout {
            groups: self
                .groups
                .into_iter()
                .flatten()
                .map(finish)
                .collect::<Result<_, _>>()
                .unwrap(),
        }
    }

    /// Moves out the group of `coupled`; one destination group pairs it once.
    pub(super) fn take_group(
        &mut self,
        coupled: SectorId,
    ) -> Option<FusionBlockMatrixGroupBuilder> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::FusionLayoutLookup);
        self.group_indices
            .get(&coupled)
            .and_then(|&group_index| self.groups.get_mut(group_index))
            .and_then(Option::take)
    }
}

#[cfg(test)]
pub(super) struct FinishedLayout<C> {
    pub(super) groups: Vec<FusionBlockMatrixGroup<C>>,
}

/// Pairs each destination coupled-sector group with the operand groups of the
/// same coupled sector, giving every shared GEMM index one tree order.
///
/// Why not trust each operand's own order: the GEMM multiplies rows and
/// columns by position, and an expert tiling may stack the trees of a
/// coupled sector in any order, so `lhs` columns and `rhs` rows (and the
/// destination rows and columns against the operand ones) could otherwise name
/// different trees at one position (#1517). Aligned orders are left
/// untouched, so an operand whose storage already matches keeps its direct
/// GEMM; a re-based one is packed through its group matrix.
fn pair_coupled_groups<C>(
    mut lhs: FusionBlockMatrixLayout,
    mut rhs: FusionBlockMatrixLayout,
    dst: FusionBlockMatrixLayout,
    mut finish_lhs: impl FnMut(
        FusionBlockMatrixGroupBuilder,
    ) -> Result<FusionBlockMatrixGroup<C>, OperationError>,
    mut finish_rhs: impl FnMut(
        FusionBlockMatrixGroupBuilder,
    ) -> Result<FusionBlockMatrixGroup<C>, OperationError>,
    mut finish_dst: impl FnMut(
        FusionBlockMatrixGroupBuilder,
    ) -> Result<FusionBlockMatrixGroup<C>, OperationError>,
) -> Result<(Vec<FusionBlockContractGroupPlan<C>>, HashSet<usize>), OperationError>
where
    C: DenseBlockScalar,
{
    let mut groups = Vec::new();
    let mut active_dst_blocks = HashSet::<usize>::new();
    for dst_group in dst.groups.into_iter().flatten() {
        let Some(mut lhs_group) = lhs.take_group(dst_group.coupled) else {
            continue;
        };
        let Some(mut rhs_group) = rhs.take_group(dst_group.coupled) else {
            continue;
        };
        adopt_tree_offsets(&mut lhs_group.row_offsets, &dst_group.row_offsets)?;
        adopt_tree_offsets(&mut rhs_group.row_offsets, &lhs_group.col_offsets)?;
        adopt_tree_offsets(&mut rhs_group.col_offsets, &dst_group.col_offsets)?;
        for block_index in &dst_group.blocks {
            debug_assert!(
                !active_dst_blocks.contains(block_index),
                "core fusion-block dst subblock must be scattered exactly once"
            );
        }
        active_dst_blocks.extend(dst_group.blocks.iter().copied());
        groups.push(FusionBlockContractGroupPlan::new(
            finish_lhs(lhs_group)?,
            finish_rhs(rhs_group)?,
            finish_dst(dst_group)?,
        )?);
    }
    Ok((groups, active_dst_blocks))
}

/// Re-bases `offsets` onto the tree positions of `reference`, the other
/// matrix that shares this GEMM index. Empty trees occupy no position and are
/// ignored; any other tree must appear on both sides with one dimension.
fn adopt_tree_offsets(
    offsets: &mut CoupledMatrixSide<FusionTreeKey, TreeMatrixOffset>,
    reference: &CoupledMatrixSide<FusionTreeKey, TreeMatrixOffset>,
) -> Result<(), OperationError> {
    let nonempty = |side: &CoupledMatrixSide<FusionTreeKey, TreeMatrixOffset>| {
        side.iter().filter(|(_, placed)| placed.dim != 0).count()
    };
    let mut aligned = true;
    for (tree, placed) in offsets.iter().filter(|(_, placed)| placed.dim != 0) {
        let Some(target) = reference.get(tree) else {
            return Err(OperationError::StructureMismatch {
                tensor: "fusion contraction index",
            });
        };
        if target.dim != placed.dim {
            return Err(OperationError::ShapeMismatch {
                dst: vec![target.dim],
                src: vec![placed.dim],
            });
        }
        aligned &= target.offset == placed.offset;
    }
    if nonempty(offsets) != nonempty(reference) {
        return Err(OperationError::StructureMismatch {
            tensor: "fusion contraction index",
        });
    }
    if !aligned {
        for (tree, placed) in offsets.iter_mut() {
            placed.offset = reference.get(tree).map_or(0, |target| target.offset);
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) struct FusionBlockMatrixGroupBuilder {
    coupled: SectorId,
    // Core's placement rule owns each tree's (offset, dim); the contraction
    // takes the maps by value and re-bases them in place (#1517).
    row_offsets: CoupledMatrixSide<FusionTreeKey, TreeMatrixOffset>,
    col_offsets: CoupledMatrixSide<FusionTreeKey, TreeMatrixOffset>,
    tree_pairs: HashSet<(FusionTreeKey, FusionTreeKey)>,
    blocks: Vec<usize>,
    logical_blocks: Option<Vec<usize>>,
    occupied_elements: usize,
}

impl FusionBlockMatrixGroupBuilder {
    fn new(coupled: SectorId) -> Self {
        Self {
            coupled,
            row_offsets: CoupledMatrixSide::default(),
            col_offsets: CoupledMatrixSide::default(),
            tree_pairs: HashSet::new(),
            blocks: Vec::new(),
            logical_blocks: None,
            occupied_elements: 0,
        }
    }

    fn add_tree_pair(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        block_index: usize,
    ) -> Result<(), OperationError> {
        self.add_tree_pair_entry(row_tree, row_dim, col_tree, col_dim, block_index)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_tree_pair_mapped(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        logical_block_index: usize,
        storage_block_index: usize,
    ) -> Result<(), OperationError> {
        self.add_tree_pair_entry(row_tree, row_dim, col_tree, col_dim, storage_block_index)?;
        self.logical_blocks
            .get_or_insert_with(Vec::new)
            .push(logical_block_index);
        Ok(())
    }

    fn add_tree_pair_entry(
        &mut self,
        row_tree: FusionTreeKey,
        row_dim: usize,
        col_tree: FusionTreeKey,
        col_dim: usize,
        storage_block_index: usize,
    ) -> Result<(), OperationError> {
        if !self.tree_pairs.insert((row_tree.clone(), col_tree.clone())) {
            return Err(OperationError::StructureMismatch { tensor: "fusion" });
        }
        let overflow = || OperationError::ElementCountOverflow;
        let (row, new_row) =
            self.row_offsets
                .place(row_tree, || Ok(row_dim), overflow, TreeMatrixOffset::from)?;
        if !new_row && row.dim != row_dim {
            return Err(OperationError::ShapeMismatch {
                dst: vec![row.dim],
                src: vec![row_dim],
            });
        }
        let (col, new_col) =
            self.col_offsets
                .place(col_tree, || Ok(col_dim), overflow, TreeMatrixOffset::from)?;
        if !new_col && col.dim != col_dim {
            return Err(OperationError::ShapeMismatch {
                dst: vec![col.dim],
                src: vec![col_dim],
            });
        }
        let block_elements = row_dim
            .checked_mul(col_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.occupied_elements = self
            .occupied_elements
            .checked_add(block_elements)
            .ok_or(OperationError::ElementCountOverflow)?;
        self.blocks.push(storage_block_index);
        Ok(())
    }

    pub(super) fn finish<R>(
        self,
        _rule: &R,
        space: &DynamicFusionMapSpace,
    ) -> Result<FusionBlockMatrixGroup<R::Scalar>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: DenseBlockScalar,
    {
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        let block_indices = self.blocks;
        for &block_index in &block_indices {
            let block = space.structure().block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..space.nout()])?);
            let domain_strides = column_major_strides_usize(&block.shape()[space.nout()..])?;
            for stride in domain_strides {
                let matrix_stride = stride
                    .checked_mul(self.row_offsets.extent())
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = col
                .offset
                .checked_mul(self.row_offsets.extent())
                .and_then(|offset| offset.checked_add(row.offset))
                .ok_or(OperationError::ElementCountOverflow)?;
            let matrix_offset = offset_to_isize(matrix_offset)?;
            // Coefficient-free by contract (TensorKit mul! parity): fermionic
            // supertrace twists are applied during rhs materialization on the
            // dynamic route, never inside the GEMM plan.
            let coefficient = R::Scalar::one();
            subblocks.push(FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: block.shape().to_vec(),
                    strides: strides_to_isize(block.strides())?,
                    offset: offset_to_isize(block.offset())?,
                },
                matrix_offset,
                matrix_strides,
                coefficient,
            });
        }
        let matrix_elements = self
            .row_offsets
            .extent()
            .checked_mul(self.col_offsets.extent())
            .ok_or(OperationError::ElementCountOverflow)?;
        let covers_matrix = self.occupied_elements == matrix_elements;
        let direct_offset = direct_group_matrix_offset_generic(&subblocks, covers_matrix);
        Ok(FusionBlockMatrixGroup {
            coupled: self.coupled,
            rows: self.row_offsets.extent(),
            cols: self.col_offsets.extent(),
            needs_clear: !covers_matrix,
            direct_offset,
            block_indices,
            subblocks,
        })
    }

    #[cfg(test)]
    pub(super) fn finish_operand<R>(
        self,
        _rule: &R,
        source: &FusionOperandLayout<'_>,
        op: MatrixOp,
    ) -> Result<FusionBlockMatrixGroup<R::Scalar>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
        R::Scalar: DenseBlockScalar,
    {
        self.finish_operand_generic(source, op)
    }

    /// The operand sibling of [`Self::finish_generic`]: a lazy adjoint's
    /// subblocks addressed in its parent's storage, laid out for `op`.
    pub(super) fn finish_operand_generic<C>(
        self,
        source: &FusionOperandLayout<'_>,
        op: MatrixOp,
    ) -> Result<FusionBlockMatrixGroup<C>, OperationError>
    where
        C: DenseBlockScalar,
    {
        let storage = source.storage_space();
        let physical_rows = match op {
            MatrixOp::Identity => self.row_offsets.extent(),
            MatrixOp::Transpose | MatrixOp::Adjoint => self.col_offsets.extent(),
        };
        let logical_blocks =
            self.logical_blocks
                .as_deref()
                .ok_or(OperationError::StructureMismatch {
                    tensor: "oriented fusion group",
                })?;
        if logical_blocks.len() != self.blocks.len() {
            return Err(OperationError::StructureMismatch {
                tensor: "oriented fusion group",
            });
        }
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        for (&logical_index, &storage_index) in logical_blocks.iter().zip(&self.blocks) {
            let key = source.logical_key(logical_index)?;
            let block = storage.structure().block(storage_index)?;
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let split = storage.nout();
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..split])?);
            for stride in column_major_strides_usize(&block.shape()[split..])? {
                let matrix_stride = stride
                    .checked_mul(physical_rows)
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = match op {
                MatrixOp::Identity => col
                    .offset
                    .checked_mul(self.row_offsets.extent())
                    .and_then(|offset| offset.checked_add(row.offset)),
                MatrixOp::Transpose | MatrixOp::Adjoint => row
                    .offset
                    .checked_mul(self.col_offsets.extent())
                    .and_then(|offset| offset.checked_add(col.offset)),
            }
            .ok_or(OperationError::ElementCountOverflow)?;
            subblocks.push(FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: block.shape().to_vec(),
                    strides: strides_to_isize(block.strides())?,
                    offset: offset_to_isize(block.offset())?,
                },
                matrix_offset: offset_to_isize(matrix_offset)?,
                matrix_strides,
                coefficient: C::one(),
            });
        }
        let matrix_elements = self
            .row_offsets
            .extent()
            .checked_mul(self.col_offsets.extent())
            .ok_or(OperationError::ElementCountOverflow)?;
        let covers_matrix = self.occupied_elements == matrix_elements;
        let direct_offset = direct_group_matrix_offset_generic(&subblocks, covers_matrix);
        Ok(FusionBlockMatrixGroup {
            coupled: self.coupled,
            rows: self.row_offsets.extent(),
            cols: self.col_offsets.extent(),
            needs_clear: !covers_matrix,
            direct_offset,
            block_indices: self.blocks,
            subblocks,
        })
    }

    /// Generic-fusion sibling of [`Self::finish`]: byte-for-byte
    /// the same block/matrix layout, with the coefficient fixed to `1.0`.
    /// SU(N) is bosonic, so there is no supertrace twist — exactly the value
    /// `R::Scalar::one()` returns on the mult-free path. Takes no rule (the
    /// layout math is symmetry-agnostic once blocks are grouped by coupled
    /// sector).
    pub(super) fn finish_generic<C>(
        self,
        structure: &Arc<symtn_core::BlockStructure>,
        nout: usize,
    ) -> Result<FusionBlockMatrixGroup<C>, OperationError>
    where
        C: DenseBlockScalar,
    {
        let mut subblocks = Vec::with_capacity(self.blocks.len());
        let block_indices = self.blocks;
        for &block_index in &block_indices {
            let block = structure.block(block_index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "fusion",
                    index: block_index,
                });
            };
            let row = self
                .row_offsets
                .get(key.codomain_tree())
                .expect("row tree offset collected before finish");
            let col = self
                .col_offsets
                .get(key.domain_tree())
                .expect("column tree offset collected before finish");
            let mut matrix_strides = Vec::<isize>::with_capacity(block.shape().len());
            matrix_strides.extend(column_major_strides_isize(&block.shape()[..nout])?);
            let domain_strides = column_major_strides_usize(&block.shape()[nout..])?;
            for stride in domain_strides {
                let matrix_stride = stride
                    .checked_mul(self.row_offsets.extent())
                    .ok_or(OperationError::ElementCountOverflow)?;
                matrix_strides.push(isize::try_from(matrix_stride).map_err(|_| {
                    OperationError::StrideOverflow {
                        value: matrix_stride,
                    }
                })?);
            }
            let matrix_offset = col
                .offset
                .checked_mul(self.row_offsets.extent())
                .and_then(|offset| offset.checked_add(row.offset))
                .ok_or(OperationError::ElementCountOverflow)?;
            let matrix_offset = offset_to_isize(matrix_offset)?;
            // The core form glues lhs.domain to rhs.codomain without crossing
            // legs, so no braid, twist, or supertrace sign enters for any
            // braiding style: the coefficient is 1 (TensorKit `mul!`). Any
            // leg crossing belongs to a staged source or output transform.
            let coefficient = C::one();
            subblocks.push(FusionSubblockMatrixLayout {
                block: FusionStridedBlockLayout {
                    shape: block.shape().to_vec(),
                    strides: strides_to_isize(block.strides())?,
                    offset: offset_to_isize(block.offset())?,
                },
                matrix_offset,
                matrix_strides,
                coefficient,
            });
        }
        let matrix_elements = self
            .row_offsets
            .extent()
            .checked_mul(self.col_offsets.extent())
            .ok_or(OperationError::ElementCountOverflow)?;
        let covers_matrix = self.occupied_elements == matrix_elements;
        let direct_offset = direct_group_matrix_offset_generic(&subblocks, covers_matrix);
        Ok(FusionBlockMatrixGroup {
            coupled: self.coupled,
            rows: self.row_offsets.extent(),
            cols: self.col_offsets.extent(),
            needs_clear: !covers_matrix,
            direct_offset,
            block_indices,
            subblocks,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TreeMatrixOffset {
    offset: usize,
    dim: usize,
}

impl From<CoupledTreePlacement> for TreeMatrixOffset {
    fn from(placed: CoupledTreePlacement) -> Self {
        Self {
            offset: placed.offset,
            dim: placed.dim,
        }
    }
}

fn coupled_sector(tree: &FusionTreeKey) -> SectorId {
    tree.coupled()
}
