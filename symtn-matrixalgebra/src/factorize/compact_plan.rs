use super::*;

/// One source coupled-sector region's route to its two compact factor
/// regions. A zero-rank region has neither factor region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactFactorRoute {
    pub(super) source_region: usize,
    pub(super) left_region: Option<usize>,
    pub(super) right_region: Option<usize>,
    pub(super) sector: SectorId,
    pub(super) rank: usize,
}

impl CompactFactorRoute {
    /// Index of the source region in [`CompactFactorPlan::source_regions`].
    pub fn source_region(&self) -> usize {
        self.source_region
    }

    /// Index in [`CompactFactorPlan::left_regions`]; `None` at rank zero.
    pub fn left_region(&self) -> Option<usize> {
        self.left_region
    }

    /// Index in [`CompactFactorPlan::right_regions`]; `None` at rank zero.
    pub fn right_region(&self) -> Option<usize> {
        self.right_region
    }

    pub fn sector(&self) -> SectorId {
        self.sector
    }

    /// The compact bond dimension `min(rows, cols)` of this sector.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

/// Per-call routing from the source coupled-sector regions to the two factor
/// regions: the one placement-neutral authority for the compact bond, the
/// factor spaces, the sector routes and their tree layouts, shared by Host
/// and CUDA. Owned by the calling factorization and dropped with it.
///
/// Reference map (TensorKit `initialize_output` + `foreachblock`, QSpace
/// `EigenSymmetric` / `blockSVD`), what was ported and the Rust differences:
/// `docs/design.md#factorization`.
///
/// Why not `Arc` the plan or its source half: nothing shares it beyond the
/// one call that builds it, so the wrappers were two heap allocations per
/// call with no owner to serve.
#[derive(Debug)]
pub struct CompactFactorPlan {
    pub(super) source_layout: ValidatedDynamicFusionLayout,
    pub(super) source_regions: Arc<[CoupledSectorRegion]>,
    pub(super) left_layout: ValidatedDynamicFusionLayout,
    pub(super) right_layout: ValidatedDynamicFusionLayout,
    pub(super) left_regions: Arc<[CoupledSectorRegion]>,
    pub(super) right_regions: Arc<[CoupledSectorRegion]>,
    pub(super) routes: Vec<CompactFactorRoute>,
}

impl CompactFactorPlan {
    pub fn source_regions(&self) -> &Arc<[CoupledSectorRegion]> {
        &self.source_regions
    }

    pub fn left_regions(&self) -> &Arc<[CoupledSectorRegion]> {
        &self.left_regions
    }

    pub fn right_regions(&self) -> &[CoupledSectorRegion] {
        &self.right_regions
    }

    /// One route per source region, in source order.
    pub fn routes(&self) -> &[CompactFactorRoute] {
        &self.routes
    }

    /// The nonzero-rank routes with their `(left, right)` factor regions: a
    /// zero-rank sector has nothing to decompose or publish.
    pub fn executed_routes(&self) -> impl Iterator<Item = (&CompactFactorRoute, usize, usize)> {
        self.routes
            .iter()
            .filter_map(|route| Some((route, route.left_region?, route.right_region?)))
    }

    /// The left factor space `codomain <- W`, bound to `input`'s provider.
    pub fn left_space<R>(
        &self,
        input: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        input.rebind_validated(&self.left_layout)
    }

    /// The right factor space `W <- domain`, bound to `input`'s provider.
    pub fn right_space<R>(
        &self,
        input: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        input.rebind_validated(&self.right_layout)
    }

    /// The spectrum factor space `W <- W` of the compact bond, by the one
    /// spectrum-bond rule of [`spectrum_bond`].
    pub fn bond_space<R>(
        &self,
        input: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        leg_bond_space::<MultiplicityFreeAdmissionMode, R>(
            input,
            compact_bond_leg(&self.source_regions),
        )
    }

    /// Whether `route`'s left factor region lists the source's codomain
    /// trees with the source's extents and offsets, so the source rows map
    /// to the factor rows positionally. `false` at rank zero.
    pub fn left_preserves_trees(&self, route: &CompactFactorRoute) -> bool {
        route.left_region.is_some_and(|index| {
            self.left_regions[index].row_trees()
                == self.source_regions[route.source_region].row_trees()
        })
    }

    /// The domain-tree counterpart of [`Self::left_preserves_trees`].
    pub fn right_preserves_trees(&self, route: &CompactFactorRoute) -> bool {
        route.right_region.is_some_and(|index| {
            self.right_regions[index].col_trees()
                == self.source_regions[route.source_region].col_trees()
        })
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub enum CheckedGenericFactorPlanError<E> {
    Provider(E),
    Operation(OperationError),
}

impl<E> From<CheckedGenericStructureError<E>> for CheckedGenericFactorPlanError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Provider(error) => Self::Provider(error),
            CheckedGenericStructureError::Core(error) => {
                Self::Operation(OperationError::from_core_preserving_context(error))
            }
        }
    }
}

impl<E> From<OperationError> for CheckedGenericFactorPlanError<E> {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

impl<E> From<DenseError> for CheckedGenericFactorPlanError<E> {
    fn from(error: DenseError) -> Self {
        Self::Operation(OperationError::Dense(error))
    }
}

impl<E: fmt::Display> fmt::Display for CheckedGenericFactorPlanError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(_) => f.write_str("fusion provider failed"),
            Self::Operation(_) => f.write_str("tensor operation failed"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericFactorPlanError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Operation(error) => Some(error),
        }
    }
}

/// The Host direct-region plan: `None` unless `input` is a coupled-sector
/// matrix layout whose factors keep the source's tree order.
pub(super) fn compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    build_compact_factor_plan(input, input.validated_layout())
}

pub(super) fn build_compact_factor_plan<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    source_layout: ValidatedDynamicFusionLayout,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let space = input.space();
    let Some(regions) = checked_sector_regions(space.structure(), space.nout())? else {
        return Ok(None);
    };
    let plan = route_compact_factors(input, source_layout, regions)?;
    // The direct path publishes each dense factor region positionally, so it
    // is sound only when the fresh factor lists the source's trees in the
    // source's order. Why not scatter here: a reordered tiling takes the
    // packed path, whose `PlacementIndex` scatter already maps by tree
    // identity, so one authority owns non-canonical publication.
    if !plan.routes.iter().all(|route| {
        route.rank == 0 || (plan.left_preserves_trees(route) && plan.right_preserves_trees(route))
    }) {
        return Ok(None);
    }
    Ok(Some(plan))
}

/// The compact factor plan of `input` over its coupled-sector regions,
/// whatever the factors' tree order; each route then reports whether its
/// factor regions keep the source's trees
/// ([`CompactFactorPlan::left_preserves_trees`]).
///
/// The regions are cached per structure, so a caller that admitted them
/// first (a device op does, before its placement checks) pays no second
/// compilation here.
pub fn compact_factor_routes<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<CompactFactorPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let space = input.space();
    let regions = checked_sector_regions(space.structure(), space.nout())?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "compact factor source is not a coupled-sector matrix layout",
        },
    )?;
    route_compact_factors(input, input.validated_layout(), regions)
}

fn route_compact_factors<R>(
    input: &BoundDynamicFusionMapSpace<R>,
    source_layout: ValidatedDynamicFusionLayout,
    regions: Arc<[CoupledSectorRegion]>,
) -> Result<CompactFactorPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let space = input.space();
    let bond = compact_bond_leg(&regions);
    let u_space =
        build_bound_factor_space(input, space.homspace(), bond.clone(), FactorSide::Left)?;
    let vh_space = build_bound_factor_space(input, space.homspace(), bond, FactorSide::Right)?;
    let left_regions = checked_sector_regions(u_space.space().structure(), u_space.space().nout())?
        .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
            message: "compact left factor is not a coupled-sector matrix layout",
        })?;
    let right_regions =
        checked_sector_regions(vh_space.space().structure(), vh_space.space().nout())?.ok_or_else(
            || OperationError::UnsupportedTensorContractScope {
                message: "compact right factor is not a coupled-sector matrix layout",
            },
        )?;
    let routes = compile_compact_factor_routes(&regions, &left_regions, &right_regions)?;
    Ok(CompactFactorPlan {
        source_layout,
        source_regions: regions,
        left_layout: u_space.validated_layout(),
        right_layout: vh_space.validated_layout(),
        left_regions,
        right_regions,
        routes,
    })
}

/// The bond leg `W` shared by both compact factors: one sector per source
/// region with degeneracy `min(rows, cols)`.
pub(super) fn compact_bond_leg(regions: &[CoupledSectorRegion]) -> SectorLeg {
    SectorLeg::new(
        regions
            .iter()
            .map(|region| (region_sector(region), region.rows().min(region.cols()))),
        false,
    )
}

/// Whether a compact diagonal's factor with the nondual bond `bond()` lives
/// on the input space itself: the input is a nondual one-leg bond `V <- V`
/// and `bond()` is `V`. TensorKit sizes the factor bond as `fuse(V)`, which is
/// `V` for a nondual `V`; then the factor HomSpace is the admitted one and
/// needs no new space. `bond` runs only for such an input, so a dual input,
/// which keeps the provider-built `fuse(V)`, pays nothing for the check.
pub(super) fn factor_bond_is_input_bond(
    space: &DynamicFusionMapSpace,
    bond: impl FnOnce() -> SectorLeg,
) -> bool {
    let homspace = space.homspace();
    space.nout() == 1
        && space.nin() == 1
        && homspace.codomain() == homspace.domain()
        && !homspace.codomain().legs()[0].is_dual()
        && homspace.codomain().legs()[0] == bond()
}

/// A factor on the admitted input space, one dense matrix per coupled-sector
/// region in `blocks`, written where the region lies.
pub(super) fn factor_on_input_space<R, D>(
    authority: &BoundDynamicFusionMapSpace<R>,
    regions: &[CoupledSectorRegion],
    blocks: impl IntoIterator<Item = Vec<D>>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    D: FactorScalar,
{
    let space = authority.space();
    let len = space.required_len()?;
    let mut data = Vec::with_capacity(len);
    let mut placed = Vec::new();
    for (region, block) in regions.iter().zip(blocks) {
        let range = region.range();
        if block.len() != range.len() {
            return Err(OperationError::ElementCountMismatch {
                expected: range.len(),
                actual: block.len(),
            });
        }
        if placed.is_empty() && range.start == data.len() {
            data.extend(block);
        } else {
            // Why not always write in place: regions tile the payload in order
            // for every layout the admission accepts, so a gap is the rare case.
            placed.push((range, block));
        }
    }
    if !placed.is_empty() || data.len() != len {
        data.resize(len, D::zero());
        for (range, block) in placed {
            data[range].copy_from_slice(&block);
        }
    }
    BoundDynFactor::from_bound(authority.clone(), data, space.nout(), space.nin())
}

pub(super) fn compile_compact_factor_routes(
    source_regions: &[CoupledSectorRegion],
    left_regions: &[CoupledSectorRegion],
    right_regions: &[CoupledSectorRegion],
) -> Result<Vec<CompactFactorRoute>, OperationError> {
    let left_by_sector = SectorRegionIndex::new(left_regions)?;
    let right_by_sector = SectorRegionIndex::new(right_regions)?;
    let mut routes = Vec::with_capacity(source_regions.len());
    // Why not per-region `used` tables: the sector -> region index is
    // injective and each nonzero route consumes a distinct index per side, so
    // "an unused nonzero region exists" is exactly "used < nonzero regions".
    let mut used_left = 0usize;
    let mut used_right = 0usize;
    for (source_region, region) in source_regions.iter().enumerate() {
        let sector = region_sector(region);
        let rank = region.rows().min(region.cols());
        let (left_region, right_region) = if rank == 0 {
            (None, None)
        } else {
            let left_region = sector_region_index_of(&left_by_sector, sector, "left")?;
            let right_region = sector_region_index_of(&right_by_sector, sector, "right")?;
            validate_factor_region(&left_regions[left_region], region.rows(), rank, "left")?;
            validate_factor_region(&right_regions[right_region], rank, region.cols(), "right")?;
            used_left += 1;
            used_right += 1;
            (Some(left_region), Some(right_region))
        };
        routes.push(CompactFactorRoute {
            source_region,
            left_region,
            right_region,
            sector,
            rank,
        });
    }
    validate_no_unused_factor_regions(left_regions, used_left, "left")?;
    validate_no_unused_factor_regions(right_regions, used_right, "right")?;
    Ok(routes)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_for_test<R>(
    input: &BoundDynamicFusionMapSpace<R>,
) -> Result<Option<CompactFactorPlan>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    compact_factor_plan(input)
}

#[cfg(test)]
#[expect(
    clippy::type_complexity,
    reason = "the test seam returns source, left, and right region tables in documented order"
)]
pub(crate) fn compact_factor_plan_regions_for_test(
    plan: &CompactFactorPlan,
) -> (
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
    Arc<[CoupledSectorRegion]>,
) {
    (
        Arc::clone(&plan.source_regions),
        Arc::clone(&plan.left_regions),
        Arc::clone(&plan.right_regions),
    )
}

#[cfg(test)]
pub(crate) fn validate_compact_factor_routes_for_test(
    source: &[CoupledSectorRegion],
    u: &[CoupledSectorRegion],
    vh: &[CoupledSectorRegion],
) -> Result<Vec<CompactFactorRoute>, OperationError> {
    compile_compact_factor_routes(source, u, vh)
}

#[cfg(test)]
pub(crate) fn compact_factor_plan_routes_for_test(
    plan: &CompactFactorPlan,
) -> &[CompactFactorRoute] {
    &plan.routes
}

#[cfg(test)]
impl CompactFactorRoute {
    pub(crate) fn factor_regions_for_test(&self) -> (usize, Option<usize>, Option<usize>) {
        (self.source_region, self.left_region, self.right_region)
    }
}

pub(super) fn region_sector(region: &CoupledSectorRegion) -> SectorId {
    region.coupled()
}

pub(super) fn sector_region_index_map(
    regions: &[CoupledSectorRegion],
) -> Result<FxHashMap<SectorId, usize>, OperationError> {
    let mut by_sector = FxHashMap::with_capacity_and_hasher(regions.len(), Default::default());
    for (index, region) in regions.iter().enumerate() {
        if by_sector.insert(region_sector(region), index).is_some() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "coupled-sector region description contains a duplicate sector",
            });
        }
    }
    Ok(by_sector)
}

/// Proves that a compact diagonal spectrum exactly covers aligned square
/// source regions and returns the sole sector lookup used by its caller.
/// Region labels are unique because callers pass `checked_sector_regions` output.
pub(super) fn aligned_diagonal_spectrum_by_sector<'a, D>(
    regions: &[CoupledSectorRegion],
    spectrum: &'a [SectorSpectrum<D>],
) -> Result<FxHashMap<SectorId, &'a SectorSpectrum<D>>, &'static str> {
    if spectrum.len() != regions.len() {
        return Err("compact diagonal spectrum sector count differs from the bond");
    }
    let mut by_sector = FxHashMap::with_capacity_and_hasher(spectrum.len(), Default::default());
    for entry in spectrum {
        if by_sector.insert(entry.sector, entry).is_some() {
            return Err("compact diagonal spectrum lists a bond sector twice");
        }
    }
    for region in regions {
        let entry = by_sector
            .get(&region.coupled())
            .ok_or("compact diagonal spectrum is missing a bond sector")?;
        if !region.has_aligned_diagonal() || region.rows() != region.cols() {
            return Err("compact diagonal spectrum bond region is not an aligned square diagonal");
        }
        if entry.values.len() != region.rows() {
            return Err("compact diagonal spectrum length differs from the sector degeneracy");
        }
    }
    Ok(by_sector)
}

/// Sector -> region index lookup over one factor's region table.
///
/// Canonical factor layouts list their regions strictly sorted by coupled
/// sector (a structural invariant of the derived layout, checked here in
/// O(G)), so a binary search serves without building a map. Expert region
/// tables that are not sorted keep the hash-map path, which also reports the
/// duplicate-sector error.
pub(super) enum SectorRegionIndex<'a> {
    Sorted(&'a [CoupledSectorRegion]),
    Map(FxHashMap<SectorId, usize>),
}

impl<'a> SectorRegionIndex<'a> {
    pub(super) fn new(regions: &'a [CoupledSectorRegion]) -> Result<Self, OperationError> {
        if regions
            .windows(2)
            .all(|pair| region_sector(&pair[0]) < region_sector(&pair[1]))
        {
            Ok(Self::Sorted(regions))
        } else {
            sector_region_index_map(regions).map(Self::Map)
        }
    }

    pub(super) fn get(&self, sector: SectorId) -> Option<usize> {
        match self {
            Self::Sorted(regions) => regions.binary_search_by_key(&sector, region_sector).ok(),
            Self::Map(map) => map.get(&sector).copied(),
        }
    }
}

pub(super) fn sector_region_index_of(
    regions: &SectorRegionIndex<'_>,
    sector: SectorId,
    side: &'static str,
) -> Result<usize, OperationError> {
    regions
        .get(sector)
        .ok_or_else(|| OperationError::SpaceMismatch {
            message: match side {
                "left" => "compact left factor is missing a nonzero-rank sector",
                _ => "compact right factor is missing a nonzero-rank sector",
            },
        })
}

pub(super) fn validate_factor_region(
    region: &CoupledSectorRegion,
    rows: usize,
    cols: usize,
    side: &'static str,
) -> Result<(), OperationError> {
    if region.rows() != rows || region.cols() != cols {
        return Err(OperationError::SpaceMismatch {
            message: match side {
                "left" => "compact left sector region has an unexpected shape",
                _ => "compact right sector region has an unexpected shape",
            },
        });
    }
    Ok(())
}

pub(super) fn validate_no_unused_factor_regions(
    regions: &[CoupledSectorRegion],
    used: usize,
    side: &'static str,
) -> Result<(), OperationError> {
    let nonzero = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    if used < nonzero {
        return Err(OperationError::SpaceMismatch {
            message: match side {
                "left" => "compact left factor contains an unused nonzero sector",
                _ => "compact right factor contains an unused nonzero sector",
            },
        });
    }
    Ok(())
}
