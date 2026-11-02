use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CoupledFusionTrees {
    pub(crate) coupled: SectorId,
    pub(crate) trees: Vec<FusionTreeKey>,
}

/// The trees of one enumeration, grouped by coupled sector while they are
/// pushed into a single arena: O(1) arena allocations per enumeration instead
/// of one or more per tree.
#[derive(Default)]
pub(super) struct CoupledTreeRows {
    pub(super) builder: TreeArenaBuilder,
    groups: Vec<(SectorId, usize)>,
    group_of_row: Vec<usize>,
    index: FxHashMap<SectorId, usize>,
}

impl CoupledTreeRows {
    /// The group of `coupled`, created on first use even if it stays empty.
    pub(super) fn group(&mut self, coupled: SectorId) -> usize {
        *self.index.entry(coupled).or_insert_with(|| {
            self.groups.push((coupled, 0));
            self.groups.len() - 1
        })
    }

    /// Assign every row pushed since `first_row` to `group`.
    pub(super) fn assign(&mut self, group: usize, first_row: usize) {
        let added = self.builder.len() - first_row;
        self.group_of_row.extend(std::iter::repeat_n(group, added));
        self.groups[group].1 += added;
    }

    pub(super) fn finish(self) -> Vec<CoupledFusionTrees> {
        let arena = self.builder.freeze();
        let mut grouped = self
            .groups
            .iter()
            .map(|&(coupled, count)| CoupledFusionTrees {
                coupled,
                trees: Vec::with_capacity(count),
            })
            .collect::<Vec<_>>();
        for (row, &group) in self.group_of_row.iter().enumerate() {
            grouped[group].trees.push(arena.key(row));
        }
        grouped
    }
}

pub(crate) fn validate_multiplicity_free_execution_style<R>(rule: &R) -> Result<(), CoreError>
where
    R: FusionRule,
{
    if rule.fusion_style().is_multiplicity_free() {
        Ok(())
    } else {
        Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: rule.fusion_style(),
        })
    }
}

pub(crate) fn try_visit_selected_leg_tuples<E, F>(
    legs: &[SectorLeg],
    remaining: usize,
    current: &mut [FusionTreeLeg],
    emit: &mut F,
) -> Result<(), E>
where
    F: FnMut(&[FusionTreeLeg]) -> Result<(), E>,
{
    if remaining == 0 {
        return emit(current);
    }

    let index = remaining - 1;
    for &sector in legs[index].sectors() {
        current[index] = FusionTreeLeg::new(sector, legs[index].is_dual());
        try_visit_selected_leg_tuples(legs, remaining - 1, current, emit)?;
    }
    Ok(())
}

/// The fusion trees of every coupled sector of `space`, ascending by coupled
/// sector. Within a sector the trees follow TensorKit's `fusiontrees(P, c)`
/// (`_fusiontree_iterate`): last uncoupled sector slowest, then the inner
/// line feeding its vertex in `coupled ⊗ dual(last)` order, then the vertex
/// label, then the front of the tree; uncoupled sectors iterate by
/// `SectorId`. Trees of one uncoupled tuple therefore need not be adjacent.
pub(crate) fn fusion_trees_by_coupled_for_space<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Vec<CoupledFusionTrees>
where
    R: MultiplicityFreeFusionRule,
{
    fusion_trees_by_coupled_for_space_within(rule, space, None)
}

/// [`fusion_trees_by_coupled_for_space`] for the coupled sectors in the
/// sorted `within` only, when given: a HomSpace's trees exist only for the
/// coupled sectors both sides reach (TensorKit's `blocksectors(W)` drives
/// `fusiontrees(W)` the same way), so the other side's need not be built.
pub(crate) fn fusion_trees_by_coupled_for_space_within<R>(
    rule: &R,
    space: &FusionProductSpace,
    within: Option<&[SectorId]>,
) -> Vec<CoupledFusionTrees>
where
    R: MultiplicityFreeFusionRule,
{
    let steps = SpaceFusionSteps::try_new(
        space,
        within,
        || rule.vacuum(),
        |sector| Ok::<_, std::convert::Infallible>(rule.dual(sector)),
        |front, last| Ok(rule.fusion_channels(front, last)),
        |front, last, coupled| Ok(rule.nsymbol(front, last, coupled)),
    );
    match steps {
        Ok(steps) => steps.collect(),
        Err(never) => match never {},
    }
}

/// Checked sibling of [`fusion_trees_by_coupled_for_space`] for external
/// multiplicity-free providers.
///
/// It enumerates the identical [`CoupledFusionTrees`] set and canonical order
/// using only the fallible [`CheckedFusionAlgebra`] primitives, returning the
/// exact [`FusionAlgebraError`] instead of relying on the infallible hot-path
/// methods, which may panic or overflow on an unrepresentable sector produced
/// by an external provider. Both run the one [`SpaceFusionSteps`] walk, so
/// keys and order match byte-for-byte on any provider whose checked methods
/// agree with their infallible counterparts.
pub(crate) fn try_fusion_trees_by_coupled_for_space_checked<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Result<Vec<CoupledFusionTrees>, FusionAlgebraError>
where
    R: CheckedFusionAlgebra,
{
    let steps = SpaceFusionSteps::try_new(
        space,
        None,
        || rule.vacuum(),
        |sector| rule.try_dual_sector(sector),
        |front, last| rule.try_fusion_channels(front, last),
        |front, last, coupled| rule.try_nsymbol(front, last, coupled),
    )?;
    Ok(steps.collect())
}

/// One fusion vertex of a product space: `front ⊗ last → coupled` with
/// `labels` vertex labels, where `front` is a sector reachable by fusing the
/// legs before `last`'s leg.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FusionStep {
    coupled: SectorId,
    last: SectorId,
    front: SectorId,
    labels: usize,
}

/// The fusion trees of a product space for the selected coupled sectors,
/// stored as the vertices of each leg (TensorKit's `fusiontrees(P, c)` for
/// each selected `c`).
///
/// Built by one forward fold over the legs' sector sets, like TensorKit's
/// `blocksectors`; it records each live vertex instead of only the reached
/// sectors. A backward pass from the selected coupled sectors then orders
/// the inner lines of the vertices those sectors reach, so the walk visits
/// no dead branch and no unselected sector is fused with a dual leg.
///
/// With `V` recorded vertices, `B` inverse-channel candidates of the
/// `coupled ⊗ dual(last)` products of reached vertex groups, `T` emitted
/// trees and `N` legs: the build sorts in O(V log V), filters the inverse
/// candidates in O(B log V) and the walk costs O(T · N · log V), plus the
/// provider's forward `front ⊗ last` and per-candidate N-symbol queries
/// and its `B`-candidate inverse queries. None of these scales with the
/// number of uncoupled tuples reaching a coupled sector.
struct SpaceFusionSteps<'a> {
    space: &'a FusionProductSpace,
    /// `levels[k - 1]`: the vertices fusing leg `k`, sorted by
    /// `(coupled, last)`, then in walk order of `front`. Above the first
    /// level, only vertices reached from a selected coupled sector.
    levels: Vec<Vec<FusionStep>>,
    /// Selected coupled sectors with at least one tree, ascending.
    coupled: Vec<SectorId>,
}

impl<'a> SpaceFusionSteps<'a> {
    /// `within`, when given, is sorted and restricts the coupled sectors.
    fn try_new<E>(
        space: &'a FusionProductSpace,
        within: Option<&[SectorId]>,
        vacuum: impl FnOnce() -> SectorId,
        mut dual: impl FnMut(SectorId) -> Result<SectorId, E>,
        mut channels: impl FnMut(SectorId, SectorId) -> Result<SectorVec, E>,
        mut nsymbol: impl FnMut(SectorId, SectorId, SectorId) -> Result<usize, E>,
    ) -> Result<Self, E> {
        let legs = space.legs();
        // A leg without sectors empties the product (unlike rank zero, the
        // unit with one vacuum tree), and an empty selection keeps no tree:
        // neither may query a prefix product, which a bounded provider can
        // fail to represent.
        if legs.iter().any(|leg| leg.sectors().is_empty()) || within.is_some_and(<[_]>::is_empty) {
            return Ok(Self {
                space,
                levels: Vec::new(),
                coupled: Vec::new(),
            });
        }
        let mut reached = match legs.first() {
            None => vec![vacuum()],
            Some(leg) => leg.sectors().to_vec(),
        };
        let mut levels = Vec::with_capacity(legs.len().saturating_sub(1));
        for leg in legs.iter().skip(1) {
            let mut forward = Vec::new();
            for &front in &reached {
                for &last in leg.sectors() {
                    for coupled in channels(front, last)? {
                        let labels = nsymbol(front, last, coupled)?;
                        if labels > 0 {
                            forward.push(FusionStep {
                                coupled,
                                last,
                                front,
                                labels,
                            });
                        }
                    }
                }
            }
            forward.sort_unstable_by_key(|step| (step.coupled, step.last, step.front));
            forward.dedup();
            reached = forward.iter().map(|step| step.coupled).collect();
            reached.dedup();
            levels.push(forward);
        }
        if let Some(within) = within {
            reached.retain(|coupled| within.binary_search(coupled).is_ok());
        }
        // Order inner lines from the selected sectors down: a vertex group
        // `(coupled, last)` asks for `coupled ⊗ dual(last)` only if the walk
        // will enter `coupled`. That product can leave a bounded provider's
        // domain even when the forward fold did not, so it must not be
        // requested for sectors outside the selection.
        let mut entered = reached.clone();
        for forward in levels.iter_mut().skip(1).rev() {
            // The front is an inner line, which iterates as
            // `coupled ⊗ dual(last)` (TensorKit's `vertexiterN`), possibly
            // not in id order; that list also decides which fronts
            // (and how many times) the backward walk visits. The first
            // level keeps leg order (TensorKit's two-leg `_fusiontree_iterate`).
            let mut level = Vec::with_capacity(forward.len());
            let mut fronts = Vec::new();
            for run in forward.chunk_by(|a, b| (a.coupled, a.last) == (b.coupled, b.last)) {
                let (coupled, last) = (run[0].coupled, run[0].last);
                if entered.binary_search(&coupled).is_err() {
                    continue;
                }
                for front in channels(coupled, dual(last)?)? {
                    if let Ok(at) = run.binary_search_by_key(&front, |step| step.front) {
                        level.push(run[at]);
                        fronts.push(front);
                    }
                }
            }
            fronts.sort_unstable();
            fronts.dedup();
            entered = fronts;
            *forward = level;
        }
        Ok(Self {
            space,
            levels,
            coupled: reached,
        })
    }

    /// The trees of each selected coupled sector, one group per sector in
    /// ascending order, in one shared arena.
    fn collect(&self) -> Vec<CoupledFusionTrees> {
        // Trees of one tuple are not adjacent in this order, so equal
        // external runs are shared by content.
        let mut builder = TreeArenaBuilder::sharing();
        let mut walk = TreeWalk::new(self.space);
        let mut ranges = Vec::with_capacity(self.coupled.len());
        for &sector in &self.coupled {
            let first = builder.len();
            self.push_trees(sector, &mut walk, &mut builder);
            ranges.push((sector, first..builder.len()));
        }
        let arena = builder.freeze();
        ranges
            .into_iter()
            .map(|(coupled, rows)| CoupledFusionTrees {
                coupled,
                trees: rows.map(|row| arena.key(row)).collect(),
            })
            .collect()
    }

    fn push_trees(&self, coupled: SectorId, walk: &mut TreeWalk, builder: &mut TreeArenaBuilder) {
        match self.space.len() {
            0 => walk.push(coupled, builder),
            legs => self.visit(legs - 1, coupled, coupled, walk, builder),
        }
    }
    /// TensorKit's `_fusiontree_iterate` (fusiontrees/iterator.jl): at leg
    /// `level`, the uncoupled sector varies slowest, then the inner line
    /// feeding its vertex, then the vertex label, then the front of the tree
    /// recursively. Uncoupled sectors ascend by `SectorId`; inner lines
    /// keep `coupled ⊗ dual(last)` channel order.
    fn visit(
        &self,
        level: usize,
        sector: SectorId,
        coupled: SectorId,
        walk: &mut TreeWalk,
        builder: &mut TreeArenaBuilder,
    ) {
        if level == 0 {
            if self.space.legs()[0]
                .sectors()
                .binary_search(&sector)
                .is_ok()
            {
                walk.uncoupled[0] = sector;
                walk.push(coupled, builder);
            }
            return;
        }
        let steps = &self.levels[level - 1];
        let start = steps.partition_point(|step| step.coupled < sector);
        for step in steps[start..]
            .iter()
            .take_while(|step| step.coupled == sector)
        {
            walk.uncoupled[level] = step.last;
            if level >= 2 {
                walk.lines[level - 2] = step.front;
            }
            for label in 1..=step.labels {
                walk.vertices[level - 1] =
                    MultiplicityIndex::new(label).expect("vertex labels start at one");
                self.visit(level - 1, step.front, coupled, walk, builder);
            }
        }
    }
}

/// The tree under construction during a [`SpaceFusionSteps`] walk.
struct TreeWalk {
    uncoupled: Vec<SectorId>,
    is_dual: Vec<bool>,
    /// Inner lines, innermost first: `lines[j]` fuses legs `0..=j + 1`.
    lines: Vec<SectorId>,
    vertices: Vec<MultiplicityIndex>,
}

impl TreeWalk {
    fn new(space: &FusionProductSpace) -> Self {
        let legs = space.len();
        Self {
            uncoupled: vec![SectorId::new(0); legs],
            is_dual: space.legs().iter().map(SectorLeg::is_dual).collect(),
            lines: vec![SectorId::new(0); legs.saturating_sub(2)],
            vertices: vec![MultiplicityIndex::ONE; legs.saturating_sub(1)],
        }
    }

    fn push(&self, coupled: SectorId, builder: &mut TreeArenaBuilder) {
        builder.push_parts(
            self.uncoupled.iter().copied(),
            coupled,
            self.is_dual.iter().copied(),
            self.lines.iter().copied(),
            self.vertices.iter().copied(),
        );
    }
}

pub(crate) fn fusion_trees_by_coupled_for_selected_space<R>(
    rule: &R,
    space: &FusionProductSpace,
    selected: &[SectorId],
) -> Result<Vec<CoupledFusionTrees>, CoreError>
where
    R: MultiplicityFreeFusionRule,
{
    if selected.len() != space.len() {
        return Err(CoreError::DimensionMismatch {
            expected: space.len(),
            actual: selected.len(),
        });
    }
    for (&sector, leg) in selected.iter().zip(space.legs()) {
        // Sectors are stored sorted (SortedVectorDict invariant); binary-search
        // to stay consistent with `SectorLeg::degeneracy`.
        if leg.sectors().binary_search(&sector).is_err() {
            return Err(CoreError::InvalidSector { sector });
        }
    }

    let legs = selected
        .iter()
        .zip(space.legs())
        .map(|(&sector, leg)| FusionTreeLeg::new(sector, leg.is_dual()))
        .collect::<Vec<_>>();
    let uncoupled: Vec<SectorId> = legs.iter().map(|leg| leg.sector()).collect();
    let is_dual: Vec<bool> = legs.iter().map(|leg| leg.is_dual()).collect();
    let mut rows = CoupledTreeRows::default();
    let mut externals_row = None;
    for coupled in reachable_coupled_sectors(rule, &uncoupled) {
        let first = rows.builder.len();
        push_fusion_trees_for_coupled_sharing(
            rule,
            &mut rows.builder,
            &uncoupled,
            &is_dual,
            &uncoupled,
            coupled,
            &mut externals_row,
        );
        if rows.builder.len() > first {
            let group = rows.group(coupled);
            rows.assign(group, first);
        }
    }
    let mut grouped = rows.finish();
    grouped.sort_by_key(|group| group.coupled);
    Ok(grouped)
}

/// Generic-fusion (outer-multiplicity) sibling of
/// [`fusion_trees_by_coupled_for_space`]: emits multiplicity-aware fusion-tree
/// keys (one per vertex-label combination) through the same
/// [`SpaceFusionSteps`] walk. `R: FusionRule` (not
/// `MultiplicityFreeFusionRule`) so outer-multiplicity rules can drive the
/// Space layer.
///
/// Before the walk, the clean classification folds every uncoupled tuple:
/// `P = ∏ (sectors of each leg)` calls of the provider's
/// `coupled_sector_fold`, each costing that provider's per-tuple fold. The
/// walk itself, restricted to the clean sectors, has the
/// [`SpaceFusionSteps`] cost.
///
/// Escape semantics (Option A, refute/b3b-verify): the coupled candidates of
/// each leg tuple are classified by [`FusionRule::coupled_sector_fold`]. Trees
/// are enumerated for CLEAN sectors only (their tree set is exactly the
/// provider's complete set); tainted / escaped / poisoned candidates are reported in the
/// returned aggregate so the caller can refuse construction with an `Err` —
/// block dimensions are either exactly right or an error, never silently
/// truncated. A sector clean in one tuple but tainted in another is tainted
/// overall (its block would mix complete and incomplete tree sets).
pub(crate) fn fusion_trees_by_coupled_for_space_generic<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> (Vec<CoupledFusionTrees>, CoupledSectorFold)
where
    R: FusionRule,
{
    let checked = InfallibleGeneric::new(rule);
    match fusion_trees_by_coupled_for_space_generic_checked(&checked, space) {
        Ok(result) => result,
        Err(CheckedGenericStructureError::Provider(never)) => match never {},
        Err(CheckedGenericStructureError::Core(error)) => {
            // The infallible API only reports the bounded-table failure after the
            // aggregate is built. All other core failures are impossible for
            // a structurally valid infallible FusionRule walk.
            panic!("infallible Generic structural walk failed unexpectedly: {error}")
        }
    }
}

pub(crate) fn fusion_trees_by_coupled_for_space_generic_checked<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Result<(Vec<CoupledFusionTrees>, CoupledSectorFold), CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
{
    // The clean classification is per tuple (a sector clean in one tuple but
    // tainted in another is tainted overall), so it still folds every tuple;
    // the trees of the clean sectors then come from the one space walk.
    let mut aggregate = CoupledSectorFoldBuilder::default();
    let mut uncoupled = Vec::with_capacity(space.len());
    space.try_visit_selected_leg_tuples(&mut |tuple| {
        uncoupled.clear();
        uncoupled.extend(tuple.iter().map(|leg| leg.sector()));
        let fold = rule
            .try_coupled_sector_fold(&uncoupled)
            .map_err(CheckedGenericStructureError::Provider)?;
        aggregate.absorb(fold);
        Ok::<(), CheckedGenericStructureError<R::Error>>(())
    })?;
    // Tainted-anywhere beats clean-somewhere, and an unknown split anywhere
    // demotes every candidate; both live in the builder's seal.
    let aggregate = aggregate.seal();
    // `fusion_channels_in_table`: a clean sector has no tree through a
    // frontier inner line, so skipping frontier channels drops only branches
    // that never reach a walked sector.
    let steps = SpaceFusionSteps::try_new(
        space,
        Some(aggregate.clean()),
        || rule.vacuum(),
        |sector| {
            rule.try_dual(sector)
                .map_err(CheckedGenericStructureError::Provider)
        },
        |front, last| {
            rule.try_fusion_channels_in_table(front, last)
                .map_err(CheckedGenericStructureError::Provider)
        },
        |front, last, coupled| {
            rule.try_nsymbol(front, last, coupled)
                .map_err(CheckedGenericStructureError::Provider)
        },
    )?;
    let grouped = steps.collect();
    Ok((grouped, aggregate))
}

/// Shared codomain×domain merge on equal coupled sectors (the generic sibling
/// of the loop in `fusion_tree_keys_uncached`).
#[cfg(test)]
pub(crate) fn merge_generic_tree_groups(
    codomain: &[CoupledFusionTrees],
    domain: &[CoupledFusionTrees],
) -> Vec<FusionTreePairKey> {
    let mut keys = Vec::new();
    let mut codomain_index = 0usize;
    let mut domain_index = 0usize;
    while codomain_index < codomain.len() && domain_index < domain.len() {
        match codomain[codomain_index]
            .coupled
            .cmp(&domain[domain_index].coupled)
        {
            std::cmp::Ordering::Less => codomain_index += 1,
            std::cmp::Ordering::Greater => domain_index += 1,
            std::cmp::Ordering::Equal => {
                for domain_tree in &domain[domain_index].trees {
                    for codomain_tree in &codomain[codomain_index].trees {
                        keys.push(FusionTreePairKey::pair(
                            codomain_tree.clone(),
                            domain_tree.clone(),
                        ));
                    }
                }
                codomain_index += 1;
                domain_index += 1;
            }
        }
    }
    keys
}

/// Human-readable summary of a non-clean coupled fold, for the construction
/// `Err` (names the escaping sectors — never silently dropped).
pub(crate) fn fusion_fold_error_message(side: &str, fold: &CoupledSectorFold) -> String {
    let mut parts = Vec::new();
    if !fold.out_of_table().is_empty() {
        parts.push(format!(
            "out-of-table coupled candidates on the {side} side: {}",
            fold.out_of_table().join(", ")
        ));
    }
    if !fold.tainted().is_empty() {
        parts.push(format!(
            "sectors requiring out-of-table intermediates on the {side} side: {:?}",
            fold.tainted()
        ));
    }
    if fold.is_unknown() {
        parts.push(format!(
            "the {side}-side fold left the one-hop frontier shell (conservative)"
        ));
    }
    format!(
        "bounded Generic provider cannot represent this space exactly ({}); block \
         dimensions are either exact or an error, never truncated. Use \
         fusion_tree_keys_generic_for_coupled for provably-clean sectors, or \
         extend the provider catalog.",
        parts.join("; ")
    )
}

/// Coupled sectors reachable by fusing all legs — TensorKit's `blocksectors`.
/// Computed once per leg tuple (not per enumeration node): the forward fold
/// `⊗` over the legs with dedup. Used only to drive the per-coupled grouping;
/// the tree enumeration itself does not consult it (see below).
pub(crate) fn reachable_coupled_sectors<R>(rule: &R, effective: &[SectorId]) -> Vec<SectorId>
where
    R: MultiplicityFreeFusionRule,
{
    let mut acc: Vec<SectorId> = match effective.first() {
        None => vec![rule.vacuum()],
        Some(&first) => vec![first],
    };
    for &last in effective.iter().skip(1) {
        acc = acc
            .iter()
            .flat_map(|&front| rule.fusion_channels(front, last))
            .collect();
        acc.sort_unstable();
        acc.dedup();
    }
    acc.sort_unstable();
    acc.dedup();
    acc
}

/// Enumerate the fusion trees of `uncoupled` (with `is_dual`) into `coupled`,
/// ported from TensorKit's `_fusiontree_iterate` (fusiontrees/iterator.jl).
/// It walks the inner lines *backward* from `coupled`: peel the last leg `b`,
/// let the adjacent inner line `a` range over `coupled ⊗ dual(b)`, recurse on
/// the front legs fusing to `a`, and prune dead branches by the recursion
/// yielding nothing — no forward `possible_coupled` reachability set, matching
/// TensorKit. Like TensorKit's *lazy* iterator it never materializes an
/// intermediate tree list per recursion level: a single `visit` walk pushes
/// each completed key straight into `out`, threading one reused inner-line
/// stack. Multiplicity-free, so every vertex is the trivial label.
pub(crate) fn collect_fusion_trees_for_coupled<R>(
    rule: &R,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) -> Vec<FusionTreeKey>
where
    R: MultiplicityFreeFusionRule,
{
    let mut builder = TreeArenaBuilder::default();
    push_fusion_trees_for_coupled(rule, &mut builder, uncoupled, is_dual, effective, coupled);
    let count = builder.len();
    let arena = builder.freeze();
    (0..count).map(|row| arena.key(row)).collect()
}

/// Push every tree of `uncoupled` into `coupled` (see
/// [`collect_fusion_trees_for_coupled`]) as rows of `builder`.
pub(super) fn push_fusion_trees_for_coupled<R>(
    rule: &R,
    builder: &mut TreeArenaBuilder,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
) where
    R: MultiplicityFreeFusionRule,
{
    push_fusion_trees_for_coupled_sharing(
        rule, builder, uncoupled, is_dual, effective, coupled, &mut None,
    );
}

/// [`push_fusion_trees_for_coupled`] for one external tuple across calls:
/// `externals_row` is the first pushed row of this tuple, whose externals
/// every later tree of the tuple shares (as the base shared one externals
/// `Arc` per tuple).
pub(super) fn push_fusion_trees_for_coupled_sharing<R>(
    rule: &R,
    builder: &mut TreeArenaBuilder,
    uncoupled: &[SectorId],
    is_dual: &[bool],
    effective: &[SectorId],
    coupled: SectorId,
    externals_row: &mut Option<usize>,
) where
    R: MultiplicityFreeFusionRule,
{
    // `inner_rev` accumulates the inner lines outermost-first as the walk
    // descends; the stored key wants innermost-first, so emit reverses it.
    let mut inner_rev: Vec<SectorId> = Vec::new();
    let vertices = uncoupled.len().saturating_sub(1);
    visit_fusion_trees(rule, effective, coupled, &mut inner_rev, &mut |inner_rev| {
        let inner = inner_rev.iter().rev().copied();
        match *externals_row {
            Some(row) => {
                builder.push_lines_with_externals_of(row, coupled, inner);
            }
            None => {
                *externals_row = Some(builder.push_parts(
                    uncoupled.iter().copied(),
                    coupled,
                    is_dual.iter().copied(),
                    inner,
                    std::iter::repeat_n(MultiplicityIndex::ONE, vertices),
                ));
            }
        }
    });
}

fn visit_fusion_trees<R, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    emit: &mut F,
) where
    R: MultiplicityFreeFusionRule,
    F: FnMut(&[SectorId]),
{
    visit_fusion_trees_where(rule, effective, coupled, inner_rev, &|_, _| true, emit);
}

pub(super) fn visit_fusion_trees_where<R, P, F>(
    rule: &R,
    effective: &[SectorId],
    coupled: SectorId,
    inner_rev: &mut Vec<SectorId>,
    prefix_allowed: &P,
    emit: &mut F,
) where
    R: MultiplicityFreeFusionRule,
    P: Fn(usize, SectorId) -> bool,
    F: FnMut(&[SectorId]),
{
    if !prefix_allowed(effective.len(), coupled) {
        return;
    }
    match effective.len() {
        0 => {
            if coupled == rule.vacuum() {
                emit(inner_rev);
            }
        }
        1 => {
            if effective[0] == coupled {
                emit(inner_rev);
            }
        }
        2 => {
            if rule.nsymbol(effective[0], effective[1], coupled) != 0 {
                emit(inner_rev);
            }
        }
        _ => {
            let last = effective[effective.len() - 1];
            let front_effective = &effective[..effective.len() - 1];
            // Inner line `a` ranges over `coupled ⊗ dual(last)` (TensorKit's
            // `vertexiterN = coupled ⊗ dual(b)`); `Nsymbol(a, last, coupled)` is
            // the last vertex. No forward-reachability filter — dead `a` simply
            // emit nothing from the recursion.
            for front_coupled in rule.fusion_channels(coupled, rule.dual(last)) {
                if rule.nsymbol(front_coupled, last, coupled) == 0 {
                    continue;
                }
                inner_rev.push(front_coupled);
                visit_fusion_trees_where(
                    rule,
                    front_effective,
                    front_coupled,
                    inner_rev,
                    prefix_allowed,
                    emit,
                );
                inner_rev.pop();
            }
        }
    }
}
