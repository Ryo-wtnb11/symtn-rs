use super::*;

/// One coupled sector's factor pair: `left` is `left_rows x kept` (leading
/// columns of a column-major matrix), `right` is `kept x cols` (leading rows
/// of a column-major matrix with leading dimension `right_leading`).
pub(super) struct FactorPair<D> {
    pub(super) sector: SectorId,
    pub(super) kept: usize,
    pub(super) left: Vec<D>,
    pub(super) left_rows: usize,
    pub(super) right: Vec<D>,
    pub(super) right_leading: usize,
}

pub(super) struct SectorRank {
    pub(super) sector: SectorId,
    pub(super) kept: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum FactorSide {
    Left,
    Right,
}

#[derive(Clone, Copy)]
pub(super) enum FactorPlacement {
    Direct,
    Adjoint,
}

pub(super) fn build_bound_factor_space<R>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    new_leg: SectorLeg,
    side: FactorSide,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let bond = FusionProductSpace::new([new_leg]);
    let hom = match side {
        FactorSide::Left => FusionTreeHomSpace::new(homspace.codomain().clone(), bond),
        FactorSide::Right => FusionTreeHomSpace::new(bond, homspace.domain().clone()),
    };
    let authority = MfAuthority(authority);
    let staged = authority.stage(hom)?;
    authority.commit(staged, |_| Ok(()))
}

/// Builds the `(codomain <- W, W <- domain)` factor pair shared by SVD and
/// the orthogonal factorizations, in the coupled-sector matrix layout.
pub(super) fn build_left_right_bound_pair<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[SectorMatricization<D>],
    pairs: &mut [FactorPair<D>],
) -> Result<DynamicFactorPair<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let dimensions = pairs
        .iter()
        .map(|pair| (pair.sector, pair.kept))
        .collect::<BTreeMap<_, _>>();
    Ok((
        build_bound_factor(
            authority,
            homspace,
            matricizations,
            pairs,
            &dimensions,
            FactorSide::Left,
        )?,
        build_bound_factor(
            authority,
            homspace,
            matricizations,
            pairs,
            &dimensions,
            FactorSide::Right,
        )?,
    ))
}

pub(super) fn build_left_bound_factor<R, D, M>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
    M: SectorGeometry,
{
    let dimensions = pairs
        .iter()
        .map(|pair| (pair.sector, pair.kept))
        .collect::<BTreeMap<_, _>>();
    build_bound_factor(
        authority,
        homspace,
        matricizations,
        pairs,
        &dimensions,
        FactorSide::Left,
    )
}

pub(super) fn build_bound_factor<R, D, M>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
    M: SectorGeometry,
{
    build_bound_factor_with_placement(
        authority,
        homspace,
        matricizations,
        pairs,
        dimensions,
        side,
        FactorPlacement::Direct,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_bound_factor_with_placement<R, D, M>(
    authority: &BoundDynamicFusionMapSpace<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    placement: FactorPlacement,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
    M: SectorGeometry,
{
    publish_one_sided_factor(
        &MfAuthority(authority),
        homspace,
        matricizations,
        pairs,
        dimensions,
        side,
        placement,
    )
}

/// Publishes one factor `codomain <- W` (left) or `W <- domain` (right) with
/// bond `W` sized by `dimensions`: the per-sector factor columns of `pairs`
/// where the source has the sector, identity completion where it does not.
#[allow(clippy::too_many_arguments)]
pub(super) fn publish_one_sided_factor<A, R, D, M>(
    authority: &A,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: &mut [FactorPair<D>],
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
    placement: FactorPlacement,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    D: FactorScalar,
    M: SectorGeometry,
{
    let bond = FusionProductSpace::new([SectorLeg::new(
        dimensions
            .iter()
            .map(|(&sector, &dimension)| (sector, dimension)),
        false,
    )]);
    let output_hom = match side {
        FactorSide::Left => FusionTreeHomSpace::new(homspace.codomain().clone(), bond),
        FactorSide::Right => FusionTreeHomSpace::new(bond, homspace.domain().clone()),
    };
    let source_trees = match (side, placement) {
        (FactorSide::Left, FactorPlacement::Direct)
        | (FactorSide::Right, FactorPlacement::Adjoint) => FactorSide::Left,
        (FactorSide::Right, FactorPlacement::Direct)
        | (FactorSide::Left, FactorPlacement::Adjoint) => FactorSide::Right,
    };
    let staged = authority.stage(output_hom)?;
    let mut routes = matricizations
        .iter()
        .map(|matrix| (matrix.sector(), None))
        .collect::<FxHashMap<SectorId, Option<&FactorPair<D>>>>();
    let mut validated_placements = None;
    let space = authority.commit(staged, |structure| {
        let placements = PlacementIndex::new(matricizations, &[source_trees]);
        for block in structure.blocks() {
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            let sector = coupled_of(tree);
            if routes.contains_key(&sector) {
                placements.placement(sector, source_trees, tree)?;
            }
        }
        validated_placements = Some(placements);
        Ok(())
    })?;
    let required_len = space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    #[cfg(test)]
    crate::probe::hit(match side {
        FactorSide::Left => crate::probe::Event::LeftFactorBufferBuilds,
        FactorSide::Right => crate::probe::Event::RightFactorBufferBuilds,
    });
    for pair in pairs.iter() {
        let route = routes
            .get_mut(&pair.sector)
            .ok_or(OperationError::SpaceMismatch {
                message: "factor sector absent from the source tensor",
            })?;
        *route = Some(pair);
    }
    let (nout, nin) = match side {
        FactorSide::Left => (space.space().nout(), 1),
        FactorSide::Right => (1, space.space().nin()),
    };
    if let Some(identities) = one_sided_factor_output_plan(
        space.space().structure(),
        matricizations,
        pairs,
        dimensions,
        required_len,
        side,
        source_trees,
    ) {
        let data = take_one_sided_factors(pairs, &identities, required_len, side);
        return Ok(BoundDynFactor::from_bound(space, data, nout, nin)?);
    }
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::OneSidedFallbackPublications);
    let mut data = vec![D::zero(); required_len];
    let mut missing_offsets = FxHashMap::<SectorId, usize>::default();
    let placements = validated_placements
        .unwrap_or_else(|| PlacementIndex::new(matricizations, &[source_trees]));
    let structure = space.space().structure();
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let (sector, matrix_axis) = match side {
            FactorSide::Left => (coupled_of(key.codomain_tree()), block.shape().len() - 1),
            FactorSide::Right => (coupled_of(key.domain_tree()), 0),
        };
        if let Some(&route) = routes.get(&sector) {
            let pair = route.ok_or(OperationError::SpaceMismatch {
                message: "factor rank absent for a populated source sector",
            })?;
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            let side_offset = placements.placement(sector, source_trees, tree)?.0;
            let (factor, factor_rows) = match side {
                FactorSide::Left => (pair.left.as_slice(), pair.left_rows),
                FactorSide::Right => (pair.right.as_slice(), pair.right_leading),
            };
            scatter_matrix_block(
                &mut data,
                block.shape(),
                block.strides(),
                block.offset(),
                matrix_axis,
                side,
                factor,
                factor_rows,
                side_offset,
            );
            continue;
        }
        let dimension = *dimensions
            .get(&sector)
            .ok_or(OperationError::SpaceMismatch {
                message: "factor sector absent from the source tensor",
            })?;
        let side_offset = missing_offsets.entry(sector).or_default();
        let extent = block
            .shape()
            .iter()
            .enumerate()
            .filter(|&(axis, _)| axis != matrix_axis)
            .try_fold(1usize, |acc, (_, &value)| acc.checked_mul(value))
            .ok_or(OperationError::ElementCountOverflow)?;
        let side_end = side_offset
            .checked_add(extent)
            .ok_or(OperationError::ElementCountOverflow)?;
        if side_end > dimension {
            return Err(OperationError::ElementCountMismatch {
                expected: dimension,
                actual: side_end,
            }
            .into());
        }
        scatter_identity_matrix_block(
            &mut data,
            block.shape(),
            block.strides(),
            block.offset(),
            matrix_axis,
            dimension,
            *side_offset,
            extent,
        )?;
        *side_offset = side_end;
    }
    Ok(BoundDynFactor::from_bound(space, data, nout, nin)?)
}

pub(super) fn scatter_left_sector_blocks<D, M>(
    left_space: &DynamicFusionMapSpace,
    left_data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let left_structure = Arc::clone(left_space.structure());
    for block_index in groups.blocks(matrix.sector()) {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::LeftVisits);
        let block = left_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of(key.codomain_tree()), matrix.sector());
        let (row_offset, _) =
            index.placement(matrix.sector(), FactorSide::Left, key.codomain_tree())?;
        scatter_matrix_block(
            left_data,
            block.shape(),
            block.strides(),
            block.offset(),
            block.shape().len() - 1,
            FactorSide::Left,
            factor,
            factor_rows,
            row_offset,
        );
    }
    Ok(())
}

pub(super) fn scatter_right_sector_blocks<D, M>(
    right_space: &DynamicFusionMapSpace,
    right_data: &mut [D],
    matrix: &M,
    index: &PlacementIndex<'_>,
    groups: &SectorBlockGroups,
    factor: &[D],
    factor_rows: usize,
) -> Result<(), OperationError>
where
    D: FactorScalar,
    M: SectorGeometry,
{
    let right_structure = Arc::clone(right_space.structure());
    for block_index in groups.blocks(matrix.sector()) {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::RightVisits);
        let block = right_structure
            .block(block_index)
            .map_err(OperationError::from_core_preserving_context)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        debug_assert_eq!(coupled_of(key.domain_tree()), matrix.sector());
        let (col_offset, _) =
            index.placement(matrix.sector(), FactorSide::Right, key.domain_tree())?;
        scatter_matrix_block(
            right_data,
            block.shape(),
            block.strides(),
            block.offset(),
            0,
            FactorSide::Right,
            factor,
            factor_rows,
            col_offset,
        );
    }
    Ok(())
}

pub(super) fn reorder_columns_in_place<D: Copy>(
    vectors: &mut [D],
    n: usize,
    order: &[usize],
    visited: &mut [bool],
    scratch: &mut [D],
) {
    visited[..n].fill(false);
    for start in 0..n {
        if visited[start] {
            continue;
        }
        scratch[..n].copy_from_slice(&vectors[start * n..(start + 1) * n]);
        let mut destination = start;
        loop {
            visited[destination] = true;
            let source = order[destination];
            if source == start {
                vectors[destination * n..(destination + 1) * n].copy_from_slice(&scratch[..n]);
                break;
            }
            for row in 0..n {
                vectors[destination * n + row] = vectors[source * n + row];
            }
            destination = source;
        }
    }
}

#[cfg(test)]
pub(crate) fn reorder_columns_in_place_for_test<D: Copy>(
    vectors: &mut [D],
    n: usize,
    order: &[usize],
    visited: &mut [bool],
    column_scratch: &mut [D],
) {
    reorder_columns_in_place(vectors, n, order, visited, column_scratch);
}

/// Call-local first-match index of source trees by `(sector, side, key)`,
/// built once per publication call over every matricization.
///
/// Why not scan `tree(side, i)` per output block: the references reach a
/// block in O(1) (TensorKit's hashed sector dictionary, QSpace's cumulative
/// offsets), while the scan cost O(T_c·K) per block. Why one table for the
/// whole call rather than one per matricization and side: at G=16 with one
/// or two trees per sector the per-table allocation dominated the lookup
/// saving (+72 allocation calls per paired publication). For a handful of
/// trees the scan's early exit can still beat SipHash plus this single
/// allocation; that constant is disclosed, not dispatched on. Why not key
/// by matricization instead of sector: `sector_matricizations*` produce one
/// matricization per coupled sector and region admission rejects duplicate
/// sectors, so the sector already identifies the matricization.
pub(super) struct PlacementIndex<'a> {
    pub(super) by_tree: FxHashMap<(SectorId, FactorSide, &'a FusionTreeKey), (usize, &'a [usize])>,
}

impl<'a> PlacementIndex<'a> {
    pub(super) fn new<M: SectorGeometry>(matricizations: &'a [M], sides: &[FactorSide]) -> Self {
        debug_assert_eq!(
            matricizations
                .iter()
                .map(SectorGeometry::sector)
                .collect::<FxHashSet<_>>()
                .len(),
            matricizations.len(),
            "placement index requires one matricization per coupled sector"
        );
        let capacity = matricizations
            .iter()
            .map(|matrix| {
                sides
                    .iter()
                    .map(|&side| matrix.tree_count(side))
                    .sum::<usize>()
            })
            .sum();
        let mut by_tree = FxHashMap::with_capacity_and_hasher(capacity, Default::default());
        for matrix in matricizations {
            let sector = matrix.sector();
            for &side in sides {
                let count = matrix.tree_count(side);
                for index in 0..count {
                    if let Some(extent) = matrix.tree(side, index) {
                        by_tree
                            .entry((sector, side, extent.tree))
                            .or_insert((extent.offset, extent.shape));
                    }
                }
                #[cfg(test)]
                {
                    crate::probe::hit(crate::probe::Event::PlacementIndexedSides);
                    crate::probe::add(crate::probe::Event::PlacementIndexedTrees, count);
                }
            }
        }
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::PlacementIndexBuilds);
        Self { by_tree }
    }

    pub(super) fn placement(
        &self,
        sector: SectorId,
        side: FactorSide,
        tree: &FusionTreeKey,
    ) -> Result<(usize, &'a [usize]), OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::PlacementLookups);
        self.by_tree
            .get(&(sector, side, tree))
            .copied()
            .ok_or(OperationError::SpaceMismatch {
                message: match side {
                    FactorSide::Left => "factor codomain tree absent from the source matricization",
                    FactorSide::Right => "factor domain tree absent from the source matricization",
                },
            })
    }
}

/// Output block indices of one factor side grouped by the coupled sector of
/// that side's tree, in structure order within a sector. Built once per
/// publication call next to [`PlacementIndex`] and dropped at return, so a
/// paired publication over `G_s` matricizations visits `B + F` output blocks
/// per side (one grouping pass plus the scattered blocks) instead of
/// `G_s * B`; the cost is one `16 * B`-byte allocation per side per call,
/// paid also for `G_s = 1`.
///
/// Why not one structure-major pass routing each output block to its
/// matricization: the compact SVD and EIGH consumers keep each sector's dense
/// factor in a workspace reused across sectors, so scattering has to happen
/// per matricization, and a structure-major pass would also change which
/// missing-tree error fires first when two sectors are defective. Why not a
/// symtn-core accessor: nothing exposes the coupled grouping
/// (`sorted_indices` is uncoupled-major and the coupled ranges known at
/// layout time are dropped at `BlockStructure`).
pub(super) struct SectorBlockGroups {
    /// `(coupled sector, block index)` sorted by sector; the stable sort keeps
    /// ascending structure order within a sector.
    pub(super) entries: Vec<(SectorId, usize)>,
}

impl SectorBlockGroups {
    pub(super) fn new(
        structure: &BlockStructure,
        side: FactorSide,
    ) -> Result<Self, OperationError> {
        let mut entries = Vec::with_capacity(structure.block_count());
        for index in 0..structure.block_count() {
            let block = structure
                .block(index)
                .map_err(OperationError::from_core_preserving_context)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let tree = match side {
                FactorSide::Left => key.codomain_tree(),
                FactorSide::Right => key.domain_tree(),
            };
            entries.push((coupled_of(tree), index));
        }
        entries.sort_by_key(|&(sector, _)| sector);
        #[cfg(test)]
        {
            use crate::probe::Event;
            let (grouped, built) = match side {
                FactorSide::Left => (Event::LeftGrouped, Event::LeftGroupsBuilt),
                FactorSide::Right => (Event::RightGrouped, Event::RightGroupsBuilt),
            };
            crate::probe::add(grouped, structure.block_count());
            crate::probe::hit(built);
        }
        Ok(Self { entries })
    }

    pub(super) fn blocks(&self, sector: SectorId) -> impl Iterator<Item = usize> + '_ {
        let start = self.entries.partition_point(|&(s, _)| s < sector);
        self.entries[start..]
            .iter()
            .take_while(move |&&(s, _)| s == sector)
            .map(|&(_, index)| index)
    }
}
