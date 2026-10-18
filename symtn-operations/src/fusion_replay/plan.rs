use super::*;

impl FusionBlockContractPlan<f64> {
    pub fn from_parts(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan<f64>>,
    ) -> Result<Self, OperationError> {
        Self::from_parts_generic(
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts_with_ops(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan<f64>>,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<Self, OperationError> {
        Self::from_parts_with_ops_generic(
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
            lhs_op,
            rhs_op,
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_canonical_coupled_regions_with_ops(
        dst_structure: &Arc<BlockStructure>,
        dst_nout: usize,
        lhs_storage_structure: &Arc<BlockStructure>,
        lhs_storage_nout: usize,
        rhs_storage_structure: &Arc<BlockStructure>,
        rhs_storage_nout: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<Option<Self>, OperationError> {
        Self::try_from_canonical_coupled_regions_with_ops_generic(
            dst_structure,
            dst_nout,
            lhs_storage_structure,
            lhs_storage_nout,
            rhs_storage_structure,
            rhs_storage_nout,
            lhs_op,
            rhs_op,
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_canonical_coupled_regions_with_ops_and_alpha(
        dst_structure: &Arc<BlockStructure>,
        dst_nout: usize,
        lhs_storage_structure: &Arc<BlockStructure>,
        lhs_storage_nout: usize,
        rhs_storage_structure: &Arc<BlockStructure>,
        rhs_storage_nout: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha_for_coupled: impl FnMut(SectorId) -> Result<f64, OperationError>,
    ) -> Result<Option<Self>, OperationError> {
        Self::try_from_canonical_coupled_regions_with_ops_and_alpha_generic(
            dst_structure,
            dst_nout,
            lhs_storage_structure,
            lhs_storage_nout,
            rhs_storage_structure,
            rhs_storage_nout,
            lhs_op,
            rhs_op,
            alpha_for_coupled,
        )
    }
}

impl<C> FusionBlockContractPlan<C>
where
    C: Copy + PartialEq + One,
{
    /// True when every group reads and writes coupled-sector matrices directly
    /// in storage.
    pub fn is_fully_direct(&self) -> bool {
        self.irregular.is_empty()
    }

    /// The destination blocks no GEMM or scatter job writes. A replay into a
    /// retained (non-zeroed) destination must initialise exactly these; a
    /// fresh zero destination needs nothing (see
    /// [`ContractDestinationInit::Zeroed`]).
    pub fn inactive_destination_regions(&self) -> &[FusionScaleBlockLayout] {
        &self.inactive_dst_scale_blocks
    }

    /// Assembles a compiled plan; called by the symmetric compile layer.
    /// Direct groups form one backend batch, while each irregular group owns a
    /// fixed group-local pack/scatter descriptor in coupled-sector order.
    /// Overlapping direct destination ranges are rejected because backends may
    /// run the direct batch concurrently.
    pub fn from_parts_generic(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan<C>>,
    ) -> Result<Self, OperationError> {
        Self::from_parts_with_ops_generic(
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
            MatrixOp::Identity,
            MatrixOp::Identity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts_with_ops_generic(
        dst_structure: Arc<BlockStructure>,
        lhs_structure: Arc<BlockStructure>,
        rhs_structure: Arc<BlockStructure>,
        inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
        groups: Vec<FusionBlockContractGroupPlan<C>>,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<Self, OperationError> {
        validate_compiled_plan_layouts(
            &dst_structure,
            &lhs_structure,
            &rhs_structure,
            &inactive_dst_scale_blocks,
            &groups,
            lhs_op,
            rhs_op,
        )?;
        let (direct_batch, irregular, max_irregular_scratch_len) =
            compile_group_execution(&groups)?;
        let direct_batch_alpha = vec![C::one(); direct_batch.len()];
        let direct_batch_runs = strided_batch_runs(&direct_batch);
        Ok(Self {
            dst_structure,
            lhs_structure,
            rhs_structure,
            inactive_dst_scale_blocks,
            groups,
            direct_batch,
            direct_batch_alpha,
            direct_batch_runs,
            irregular,
            max_irregular_scratch_len,
            lhs_op,
            rhs_op,
        })
    }

    /// Compiles canonical coupled-sector structures without synthesizing
    /// per-tree fusion groups or subblock descriptors.
    ///
    /// Callers must first validate category identity and core composition.
    /// `Some` certifies only an exact canonical structural replay; `None`
    /// requests the general symmetry-aware compiler.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_canonical_coupled_regions_with_ops_generic(
        dst_structure: &Arc<BlockStructure>,
        dst_nout: usize,
        lhs_storage_structure: &Arc<BlockStructure>,
        lhs_storage_nout: usize,
        rhs_storage_structure: &Arc<BlockStructure>,
        rhs_storage_nout: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<Option<Self>, OperationError> {
        Self::try_from_canonical_coupled_regions_impl(
            dst_structure,
            dst_nout,
            lhs_storage_structure,
            lhs_storage_nout,
            rhs_storage_structure,
            rhs_storage_nout,
            lhs_op,
            rhs_op,
            |_| Ok(C::one()),
        )
    }

    /// Canonical storage plan with one already-verified coefficient per
    /// coupled-sector GEMM job.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_canonical_coupled_regions_with_ops_and_alpha_generic(
        dst_structure: &Arc<BlockStructure>,
        dst_nout: usize,
        lhs_storage_structure: &Arc<BlockStructure>,
        lhs_storage_nout: usize,
        rhs_storage_structure: &Arc<BlockStructure>,
        rhs_storage_nout: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha_for_coupled: impl FnMut(SectorId) -> Result<C, OperationError>,
    ) -> Result<Option<Self>, OperationError> {
        Self::try_from_canonical_coupled_regions_impl(
            dst_structure,
            dst_nout,
            lhs_storage_structure,
            lhs_storage_nout,
            rhs_storage_structure,
            rhs_storage_nout,
            lhs_op,
            rhs_op,
            alpha_for_coupled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_from_canonical_coupled_regions_impl(
        dst_structure: &Arc<BlockStructure>,
        dst_nout: usize,
        lhs_storage_structure: &Arc<BlockStructure>,
        lhs_storage_nout: usize,
        rhs_storage_structure: &Arc<BlockStructure>,
        rhs_storage_nout: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        mut alpha_for_coupled: impl FnMut(SectorId) -> Result<C, OperationError>,
    ) -> Result<Option<Self>, OperationError> {
        let Some(dst_regions) = sorted_coupled_regions(dst_structure, dst_nout)? else {
            return Ok(None);
        };
        let Some(lhs_regions) = sorted_coupled_regions(lhs_storage_structure, lhs_storage_nout)?
        else {
            return Ok(None);
        };
        let Some(rhs_regions) = sorted_coupled_regions(rhs_storage_structure, rhs_storage_nout)?
        else {
            return Ok(None);
        };

        // Equal content ids are one structure content, whose coupled regions
        // at one `nout` are one list: the same coupled sector then has the
        // same tree extents on both sides, so the comparison is skipped.
        let same_regions = |structure: &Arc<BlockStructure>, nout: usize, op: MatrixOp| {
            op == MatrixOp::Identity
                && nout == dst_nout
                && structure.content_id() == dst_structure.content_id()
        };
        let dst_rows_are_lhs_rows = same_regions(lhs_storage_structure, lhs_storage_nout, lhs_op);
        let dst_cols_are_rhs_cols = same_regions(rhs_storage_structure, rhs_storage_nout, rhs_op);
        let mut direct_batch = Vec::with_capacity(dst_regions.len());
        let mut inactive_dst_scale_blocks = Vec::<FusionScaleBlockLayout>::new();
        let mut previous_inactive = false;
        let mut lhs_index = 0usize;
        let mut rhs_index = 0usize;
        for dst in dst_regions.iter() {
            let follows_inactive = std::mem::replace(&mut previous_inactive, true);
            while lhs_regions
                .get(lhs_index)
                .is_some_and(|region| region.coupled() < dst.coupled())
            {
                lhs_index += 1;
            }
            while rhs_regions
                .get(rhs_index)
                .is_some_and(|region| region.coupled() < dst.coupled())
            {
                rhs_index += 1;
            }
            let lhs = lhs_regions
                .get(lhs_index)
                .map(|region| OrientedCoupledRegion::new(region, lhs_op))
                .filter(|region| region.coupled() == dst.coupled());
            let rhs = rhs_regions
                .get(rhs_index)
                .map(|region| OrientedCoupledRegion::new(region, rhs_op))
                .filter(|region| region.coupled() == dst.coupled());
            let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
                // Why merge: regions tile storage in order, so a run of
                // inactive sectors is one contiguous range, and scaling it
                // once equals scaling each sector; a layout per sector cost
                // two heap vectors and one kernel call each.
                match inactive_dst_scale_blocks.last_mut() {
                    Some(last) if follows_inactive => last.block.shape[0] += dst.range().len(),
                    _ => inactive_dst_scale_blocks.push(contiguous_scale_layout(dst.range())?),
                }
                continue;
            };
            previous_inactive = false;
            if lhs.cols() != rhs.rows()
                || dst.rows() != lhs.rows()
                || dst.cols() != rhs.cols()
                || lhs.col_trees() != rhs.row_trees()
                || (!dst_rows_are_lhs_rows && dst.row_trees() != lhs.row_trees())
                || (!dst_cols_are_rhs_cols && dst.col_trees() != rhs.col_trees())
            {
                return Ok(None);
            }
            direct_batch.push((
                Rank2GemmBatchJob {
                    dst_offset: dst.range().start,
                    lhs_offset: lhs.storage_range().start,
                    rhs_offset: rhs.storage_range().start,
                    rows: lhs.rows(),
                    contracted: lhs.cols(),
                    cols: rhs.cols(),
                },
                alpha_for_coupled(rhs.coupled())?,
            ));
        }
        validate_direct_plan_layouts(
            dst_structure,
            lhs_storage_structure,
            rhs_storage_structure,
            &inactive_dst_scale_blocks,
            &direct_batch,
        )?;
        let (direct_batch, direct_batch_alpha, direct_batch_runs) =
            if direct_batch.iter().all(|&(_, alpha)| alpha == C::one()) {
                let (jobs, alphas): (Vec<_>, Vec<_>) = direct_batch.into_iter().unzip();
                let runs = strided_batch_runs(&jobs);
                (jobs, alphas, runs)
            } else {
                group_scaled_direct_batch(&direct_batch)
            };
        Ok(Some(Self {
            dst_structure: Arc::clone(dst_structure),
            lhs_structure: Arc::clone(lhs_storage_structure),
            rhs_structure: Arc::clone(rhs_storage_structure),
            inactive_dst_scale_blocks,
            groups: Vec::new(),
            direct_batch,
            direct_batch_alpha,
            direct_batch_runs,
            irregular: Vec::new(),
            max_irregular_scratch_len: 0,
            lhs_op,
            rhs_op,
        }))
    }
}

pub fn direct_group_matrix_offset_generic<C>(
    subblocks: &[FusionSubblockMatrixLayout<C>],
    covers_matrix: bool,
) -> Option<usize>
where
    C: Copy + PartialEq + One,
{
    if !covers_matrix {
        return None;
    }
    let mut base: Option<isize> = None;
    for subblock in subblocks {
        if subblock.coefficient != C::one() {
            return None;
        }
        let strides_match = subblock
            .block
            .shape
            .iter()
            .zip(subblock.block.strides.iter().zip(&subblock.matrix_strides))
            .all(|(&dim, (&stride, &matrix_stride))| dim <= 1 || stride == matrix_stride);
        if !strides_match {
            return None;
        }
        let offset = subblock.block.offset - subblock.matrix_offset;
        if offset < 0 {
            return None;
        }
        match base {
            None => base = Some(offset),
            Some(existing) if existing != offset => return None,
            Some(_) => {}
        }
    }
    base.and_then(|offset| usize::try_from(offset).ok())
}

pub fn fusion_scale_block_layouts_excluding(
    structure: &BlockStructure,
    excluded_blocks: &HashSet<usize>,
) -> Result<Vec<FusionScaleBlockLayout>, OperationError> {
    let mut layouts = Vec::with_capacity(structure.block_count());
    for block_index in 0..structure.block_count() {
        if excluded_blocks.contains(&block_index) {
            continue;
        }
        let block = structure.block(block_index)?;
        layouts.push(FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: block.shape().to_vec(),
                strides: strides_to_isize(block.strides())?,
                offset: offset_to_isize(block.offset())?,
            },
        });
    }
    Ok(layouts)
}

fn group_scratch_layout<C>(
    class: FusionGroupExecutionClass,
    group: &FusionBlockContractGroupPlan<C>,
) -> Result<FusionGroupScratchLayout, OperationError> {
    let lhs_len = if class.packs_lhs() {
        direct_matrix_len(group.lhs.rows, group.lhs.cols)?
    } else {
        0
    };
    let rhs_len = if class.packs_rhs() {
        direct_matrix_len(group.rhs.rows, group.rhs.cols)?
    } else {
        0
    };
    let dst_len = if class.scatters_dst() {
        direct_matrix_len(group.dst.rows, group.dst.cols)?
    } else {
        0
    };
    let rhs_offset = lhs_len;
    let dst_offset = rhs_offset
        .checked_add(rhs_len)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let total_len = dst_offset
        .checked_add(dst_len)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    Ok(FusionGroupScratchLayout {
        lhs_len,
        rhs_offset,
        rhs_len,
        dst_offset,
        dst_len,
        total_len,
    })
}

fn sorted_coupled_regions(
    structure: &Arc<BlockStructure>,
    nout: usize,
) -> Result<Option<Arc<[CoupledSectorRegion]>>, OperationError> {
    let Some(regions) = structure
        .coupled_sector_regions(nout)
        .map_err(OperationError::from_core_preserving_context)?
    else {
        return Ok(None);
    };
    if regions
        .windows(2)
        .any(|pair| pair[0].coupled() >= pair[1].coupled())
    {
        return Ok(None);
    }
    Ok(Some(regions))
}

#[derive(Clone, Copy)]
struct OrientedCoupledRegion<'a> {
    storage: &'a CoupledSectorRegion,
    op: MatrixOp,
}

impl<'a> OrientedCoupledRegion<'a> {
    fn new(storage: &'a CoupledSectorRegion, op: MatrixOp) -> Self {
        Self { storage, op }
    }

    fn coupled(self) -> symtn_core::SectorId {
        self.storage.coupled()
    }

    fn rows(self) -> usize {
        match self.op {
            MatrixOp::Identity => self.storage.rows(),
            MatrixOp::Transpose | MatrixOp::Adjoint => self.storage.cols(),
        }
    }

    fn cols(self) -> usize {
        match self.op {
            MatrixOp::Identity => self.storage.cols(),
            MatrixOp::Transpose | MatrixOp::Adjoint => self.storage.rows(),
        }
    }

    fn row_trees(self) -> &'a [symtn_core::CoupledTreeExtent] {
        match self.op {
            MatrixOp::Identity => self.storage.row_trees(),
            MatrixOp::Transpose | MatrixOp::Adjoint => self.storage.col_trees(),
        }
    }

    fn col_trees(self) -> &'a [symtn_core::CoupledTreeExtent] {
        match self.op {
            MatrixOp::Identity => self.storage.col_trees(),
            MatrixOp::Transpose | MatrixOp::Adjoint => self.storage.row_trees(),
        }
    }

    fn storage_range(self) -> std::ops::Range<usize> {
        self.storage.range()
    }
}

fn contiguous_scale_layout(
    range: std::ops::Range<usize>,
) -> Result<FusionScaleBlockLayout, OperationError> {
    Ok(FusionScaleBlockLayout {
        block: FusionStridedBlockLayout {
            shape: vec![range.len()],
            strides: vec![1],
            offset: offset_to_isize(range.start)?,
        },
    })
}

pub(super) fn compile_group_execution<C>(
    groups: &[FusionBlockContractGroupPlan<C>],
) -> Result<CompiledGroupExecution, OperationError> {
    let mut direct_batch = Vec::new();
    let mut direct_destinations = Vec::new();
    let mut irregular = Vec::with_capacity(groups.len());
    let mut max_scratch_len = 0;
    for (group_index, group) in groups.iter().enumerate() {
        let class = FusionGroupExecutionClass::compile(group);
        if let Some(dst_offset) = group.dst.direct_offset {
            direct_destinations.push(Rank2GemmBatchJob {
                dst_offset,
                lhs_offset: 0,
                rhs_offset: 0,
                rows: group.dst.rows,
                contracted: group.lhs.cols,
                cols: group.dst.cols,
            });
        }
        if class.is_direct() {
            let (Some(lhs_offset), Some(rhs_offset), Some(dst_offset)) = (
                group.lhs.direct_offset,
                group.rhs.direct_offset,
                group.dst.direct_offset,
            ) else {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "direct fusion contraction class requires direct matrix offsets",
                });
            };
            direct_batch.push(Rank2GemmBatchJob {
                dst_offset,
                lhs_offset,
                rhs_offset,
                rows: group.lhs.rows,
                contracted: group.lhs.cols,
                cols: group.rhs.cols,
            });
            continue;
        }
        let scratch = group_scratch_layout(class, group)?;
        max_scratch_len = max_scratch_len.max(scratch.total_len);
        irregular.push(FusionIrregularGroupExecution {
            group_index,
            class,
            job: Rank2GemmBatchJob {
                dst_offset: group.dst.direct_offset.unwrap_or(0),
                lhs_offset: group.lhs.direct_offset.unwrap_or(0),
                rhs_offset: group.rhs.direct_offset.unwrap_or(0),
                rows: group.lhs.rows,
                contracted: group.lhs.cols,
                cols: group.rhs.cols,
            },
            scratch,
        });
    }
    validate_disjoint_direct_destinations(&direct_destinations)?;
    direct_batch.sort_by_key(|job| {
        (
            job.rows,
            job.contracted,
            job.cols,
            job.dst_offset,
            job.lhs_offset,
            job.rhs_offset,
        )
    });

    Ok((direct_batch, irregular, max_scratch_len))
}

fn validate_disjoint_direct_destinations(jobs: &[Rank2GemmBatchJob]) -> Result<(), OperationError> {
    let mut dst_ranges: Vec<(usize, usize)> = jobs
        .iter()
        .filter_map(|job| match direct_matrix_len(job.rows, job.cols) {
            Ok(0) => None,
            Ok(len) => Some(Ok((job.dst_offset, len))),
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<_, _>>()?;
    dst_ranges.sort_unstable();
    for pair in dst_ranges.windows(2) {
        let (base, len) = pair[0];
        let end = base
            .checked_add(len)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        if end > pair[1].0 {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "core contraction groups must write disjoint destination ranges",
            });
        }
    }
    Ok(())
}

/// A scaled (twisted) plan's direct jobs grouped by coefficient, in
/// first-appearance order and stably within a group, with each group's run
/// partition: one Host batch per coefficient. O(n·k) for n jobs and k
/// distinct coefficients (a fermionic θ has two).
fn group_scaled_direct_batch<C>(
    pairs: &[(Rank2GemmBatchJob, C)],
) -> (Vec<Rank2GemmBatchJob>, Vec<C>, Vec<usize>)
where
    C: Copy + PartialEq,
{
    let mut coefficients: Vec<C> = Vec::new();
    for &(_, alpha) in pairs {
        if !coefficients.contains(&alpha) {
            coefficients.push(alpha);
        }
    }
    let mut jobs = Vec::with_capacity(pairs.len());
    let mut alphas = Vec::with_capacity(pairs.len());
    let mut runs = Vec::new();
    let mut group_runs = Vec::new();
    for &coefficient in &coefficients {
        let start = jobs.len();
        for &(job, alpha) in pairs {
            if alpha == coefficient {
                jobs.push(job);
                alphas.push(alpha);
            }
        }
        strided_batch_runs_into(&jobs[start..], &mut group_runs);
        runs.extend_from_slice(&group_runs);
    }
    (jobs, alphas, runs)
}

#[cfg(test)]
mod scaled_group_tests {
    use super::*;

    fn job(offset: usize, stride: usize) -> Rank2GemmBatchJob {
        Rank2GemmBatchJob {
            dst_offset: offset * stride,
            lhs_offset: offset * stride,
            rhs_offset: offset * stride,
            rows: 2,
            contracted: 2,
            cols: 2,
        }
    }

    /// Interleaved ±1 coefficients group stably, positive first (first
    /// appearance), and each group's runs are its own partition: a strided
    /// run never spans the two groups.
    #[test]
    fn interleaved_coefficients_group_stably_with_per_group_runs() {
        let pairs: Vec<_> = (0..6)
            .map(|i| (job(i, 4), if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let (jobs, alphas, runs) = group_scaled_direct_batch(&pairs);
        let order: Vec<_> = jobs.iter().map(|job| job.dst_offset / 4).collect();
        assert_eq!(order, [0, 2, 4, 1, 3, 5]);
        assert_eq!(alphas, [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
        // Each group is constant-stride (8 apart): one run per group.
        assert_eq!(runs, [3, 3]);
    }

    /// Geometry: J0 alone in group A, then J1 and J2 in group
    /// B with a stride unlike J0→J1. Splitting a whole-list partition at the
    /// group boundary gives [1, 1, 1]; per-group runs give [1, 2].
    #[test]
    fn runs_are_partitioned_per_group_not_split_from_the_whole_list() {
        let j0 = job(0, 1);
        let j1 = job(5, 1);
        let j2 = job(7, 1);
        let (jobs, _, runs) = group_scaled_direct_batch(&[(j0, 1.0), (j1, -1.0), (j2, -1.0)]);
        assert_eq!(jobs, [j0, j1, j2]);
        assert_eq!(runs, [1, 2]);
        assert_eq!(strided_batch_runs(&jobs), [2, 1]);
    }
}
