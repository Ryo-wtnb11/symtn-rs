use super::*;

pub(crate) enum FusionTermAccumulator<K, S> {
    Empty,
    Singleton(K, S),
    Map {
        order: Vec<K>,
        coefficients: FxHashMap<K, S>,
    },
}

impl<K, S> FusionTermAccumulator<K, S>
where
    K: Clone + Eq + Hash,
    S: Clone + Add<Output = S>,
{
    pub(crate) fn new() -> Self {
        Self::Empty
    }

    pub(crate) fn push(&mut self, key: K, coefficient: S) {
        match self {
            Self::Empty => {
                *self = Self::Singleton(key, coefficient);
            }
            Self::Singleton(existing_key, existing) if existing_key == &key => {
                *existing = existing.clone() + coefficient;
            }
            Self::Singleton(_, _) => {
                let previous = std::mem::replace(self, Self::Empty);
                let Self::Singleton(existing_key, existing_coefficient) = previous else {
                    unreachable!("matched singleton state");
                };
                let mut order = Vec::with_capacity(2);
                let mut coefficients = FxHashMap::default();
                Self::push_map_term(
                    &mut order,
                    &mut coefficients,
                    existing_key,
                    existing_coefficient,
                );
                Self::push_map_term(&mut order, &mut coefficients, key, coefficient);
                *self = Self::Map {
                    order,
                    coefficients,
                };
            }
            Self::Map {
                order,
                coefficients,
            } => {
                Self::push_map_term(order, coefficients, key, coefficient);
            }
        }
    }

    fn push_map_term(
        order: &mut Vec<K>,
        coefficients: &mut FxHashMap<K, S>,
        key: K,
        coefficient: S,
    ) {
        match coefficients.entry(key) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                *existing = existing.clone() + coefficient;
            }
            Entry::Vacant(entry) => {
                order.push(entry.key().clone());
                entry.insert(coefficient);
            }
        }
    }

    pub(crate) fn into_vec(self) -> Vec<(K, S)> {
        match self {
            Self::Empty => Vec::new(),
            Self::Singleton(key, coefficient) => vec![(key, coefficient)],
            Self::Map {
                order,
                mut coefficients,
            } => {
                let mut terms = Vec::with_capacity(order.len());
                for key in order {
                    let coefficient = coefficients
                        .remove(&key)
                        .expect("accumulator order only contains inserted keys");
                    terms.push((key, coefficient));
                }
                terms
            }
        }
    }
}

/// Batched analog of [`compose_terms`]: apply `transform` to every
/// tree-pair of a whole block at once, threading a coefficient *matrix* (a
/// sparse column per original source) instead of re-running the per-source
/// term list. `columns[i]` maps `src index -> coefficient` for `basis[i]`.
///
/// This is the TensorKit 0.17 `artin_braid`/`fsbraid` batching: the elementary
/// step (bend / Artin braid) is walked ONCE for the block and its coefficients
/// are spread across all source columns, so intermediate allocation is
/// O(steps) rather than O(steps × sources) — the term-list style symtn used
/// (equivalent to TensorKit ≤0.16's per-tree `FusionTreeDict`) allocated a
/// fresh accumulator and cloned keys per source per step.
/// Dense coefficient matrix (TK's `Matrix{E}`): rows are destination basis
/// trees, columns the original sources. Stored row-major in ONE flat
/// allocation that grows amortized as rows are added, instead of a
/// `Vec<Vec<_>>` that heap-allocs a fresh column per destination tree (the
/// batched braid over a whole block adds hundreds of thousands of rows across
/// its bend/braid steps, so the per-row allocation dominated the cold
/// recoupling build).
pub(crate) struct DenseColumns<S> {
    pub(super) data: Vec<Option<S>>,
    pub(crate) num_src: usize,
    pub(crate) num_rows: usize,
}

/// Ordered block-linear result produced by one categorical block transform.
///
/// This is an internal crate-boundary vocabulary for lowering reduced blocks.
/// Destination keys are unique and ordered by source-major first appearance;
/// columns retain the caller's source order. The order of the destinations a
/// single source first reaches is deterministic for identical inputs but
/// otherwise unspecified (it is neither HomSpace order nor stable across
/// algorithm changes), so consumers must resolve destinations by key, never by
/// position. Structurally absent entries remain distinct from present zero
/// coefficients, and no destination row is wholly absent.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedBlockLinearMap<K, S> {
    pub(super) destinations: Vec<K>,
    pub(super) source_count: usize,
    pub(super) storage: OrderedBlockLinearStorage<S>,
}

/// Structural storage for an [`OrderedBlockLinearMap`].
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedBlockLinearStorage<S> {
    /// Exactly one structurally present destination for every source column.
    SingletonColumns {
        destination_rows: Vec<usize>,
        coefficients: Vec<S>,
    },
    /// Row-major destination-by-source coefficients.
    DenseDstSrc(Vec<Option<S>>),
}

#[doc(hidden)]
impl<K, S> OrderedBlockLinearMap<K, S> {
    #[inline]
    pub fn destinations(&self) -> &[K] {
        &self.destinations
    }

    #[inline]
    pub fn source_count(&self) -> usize {
        self.source_count
    }

    #[inline]
    pub fn storage(&self) -> &OrderedBlockLinearStorage<S> {
        &self.storage
    }

    #[inline]
    pub fn into_parts(self) -> (Vec<K>, usize, OrderedBlockLinearStorage<S>) {
        (self.destinations, self.source_count, self.storage)
    }
}

/// One block transform's structurally present entries per source column
/// (TensorKit's `U[dst, src]` of `fsbraid`, sparse, column-major): the trace
/// lowering's permutation input, and, over the lowered (open) destinations,
/// the trace value the composed-coefficient cache retains.
///
/// `column(n)` lists source `n`'s present `(destination index, coefficient)`
/// pairs. A present zero is an entry; a structurally absent coefficient is
/// none, so the two stay distinct. Every destination has at least one entry.
/// The storage is exact-length boxed slices: the value is immutable, and its
/// retained bytes are its lengths.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct BlockSourceColumns<K, S> {
    destinations: Box<[K]>,
    column_start: Box<[usize]>,
    entries: Box<[(usize, S)]>,
}

#[doc(hidden)]
impl<K, S> BlockSourceColumns<K, S> {
    /// `column_start` holds `source_count + 1` ascending offsets into
    /// `entries`, starting at 0; every entry's row indexes `destinations`.
    pub fn new(
        destinations: Box<[K]>,
        column_start: Box<[usize]>,
        entries: Box<[(usize, S)]>,
    ) -> Self {
        debug_assert_eq!(column_start.first(), Some(&0));
        debug_assert_eq!(column_start.last(), Some(&entries.len()));
        debug_assert!(entries.iter().all(|(row, _)| *row < destinations.len()));
        Self {
            destinations,
            column_start,
            entries,
        }
    }

    /// The empty block: no source, no destination.
    pub(crate) fn empty() -> Self {
        Self::new(Box::default(), Box::new([0]), Box::default())
    }

    /// Every source maps to the destination of its own position with
    /// coefficient one.
    pub(crate) fn identity(destinations: Box<[K]>) -> Self
    where
        S: CategoricalScalar,
    {
        let source_count = destinations.len();
        Self::new(
            destinations,
            (0..=source_count).collect(),
            (0..source_count).map(|row| (row, S::one())).collect(),
        )
    }

    #[inline]
    pub fn destinations(&self) -> &[K] {
        &self.destinations
    }

    /// Replace every destination in place, in order; a cache re-homes its
    /// keys into the entry's own arena this way (single-arena ownership, #2192).
    pub fn rehome_destinations(&mut self, mut rehome: impl FnMut(&K) -> K) {
        for destination in self.destinations.iter_mut() {
            *destination = rehome(destination);
        }
    }

    #[inline]
    pub fn source_count(&self) -> usize {
        self.column_start.len() - 1
    }

    /// Source `source`'s present entries.
    #[inline]
    pub fn column(&self, source: usize) -> &[(usize, S)] {
        &self.entries[self.column_start[source]..self.column_start[source + 1]]
    }

    /// Present entries over all sources.
    #[inline]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl<S> DenseColumns<S> {
    pub(crate) fn with_capacity(num_src: usize, rows_hint: usize) -> Self {
        Self {
            data: Vec::with_capacity(rows_hint.saturating_mul(num_src)),
            num_src,
            num_rows: 0,
        }
    }

    /// Append a new all-empty row, returning its index.
    pub(crate) fn push_empty_row(&mut self) -> usize {
        let row = self.num_rows;
        self.data
            .resize_with(self.data.len() + self.num_src, || None);
        self.num_rows += 1;
        row
    }

    #[inline]
    pub(crate) fn row(&self, row: usize) -> &[Option<S>] {
        let start = row * self.num_src;
        &self.data[start..start + self.num_src]
    }

    #[inline]
    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [Option<S>] {
        let start = row * self.num_src;
        &mut self.data[start..start + self.num_src]
    }
}

/// Compose a column-batched block with one move: every basis row's move
/// terms are spread across all source columns (`dst += step · source`). The
/// dense block composer for multiplicity-free keys and compact locals; the
/// Generic keyed block uses the sparse [`compose_generic_block`].
pub(crate) fn compose_block_terms<K, S, E, F, I>(
    basis: &[K],
    columns: &DenseColumns<S>,
    mut transform: F,
) -> Result<(Vec<K>, DenseColumns<S>), E>
where
    K: Eq + Hash,
    S: Clone + Add<Output = S> + Mul<Output = S>,
    F: FnMut(&K) -> Result<I, E>,
    I: IntoIterator<Item = (K, S)>,
{
    let num_src = columns.num_src;
    // Dedup destination tree-pairs to dense rows. The key is *moved* into the
    // map (no per-destination clone — this dedup is the hottest FusionTreeKey
    // clone/eq/hash site on the cold recoupling path); `next_basis` is rebuilt
    // from the map by row index afterwards. Rows are assigned in first-
    // appearance order, so the rebuilt `next_basis` order — and therefore every
    // coefficient — is bit-for-bit identical to pushing the key eagerly.
    let mut index: FxHashMap<K, usize> = FxHashMap::default();
    let mut next_columns: DenseColumns<S> = DenseColumns::with_capacity(num_src, basis.len());
    for (source_row, source_key) in basis.iter().enumerate() {
        for (dst_key, step_coefficient) in transform(source_key)? {
            let row = match index.get(&dst_key) {
                Some(&row) => row,
                None => {
                    let row = next_columns.push_empty_row();
                    index.insert(dst_key, row);
                    row
                }
            };
            // dst_column[src] += step_coefficient * source_column[src] for each
            // source that reaches this basis tree. Source and destination live
            // in different matrices, so the borrows don't overlap.
            let source_column = columns.row(source_row);
            let dst_column = next_columns.row_mut(row);
            for (src, source_coefficient) in source_column.iter().enumerate() {
                let Some(source_coefficient) = source_coefficient else {
                    continue;
                };
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                dst_column[src] = Some(match dst_column[src].take() {
                    Some(existing) => existing + contribution,
                    None => contribution,
                });
            }
        }
    }
    // Rebuild the basis in row order (= first-appearance order). Rows are dense
    // `0..index.len()`, so place each moved key at its row index.
    let mut slots: Vec<Option<K>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let next_basis: Vec<K> = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();
    Ok((next_basis, next_columns))
}

pub(super) struct CompactMultiplicityFreeTreeBasis {
    pub(super) frame: MultiplicityFreeTreeFrame,
    pub(super) locals: Vec<MultiplicityFreeTreeLocal>,
}

impl CompactMultiplicityFreeTreeBasis {
    pub(super) fn from_group<R>(
        group: ValidatedFusionTreeBlockGroup<'_, R>,
    ) -> Result<Self, CoreError> {
        let src_keys = group.src_keys;
        let (frame, first_local) =
            MultiplicityFreeTreeFrame::split(group.projection.tree_at(0).ok_or(
                CoreError::MalformedFusionTree {
                    message: "compact block basis requires at least one source",
                },
            )?);
        let mut locals = Vec::with_capacity(src_keys.len());
        locals.push(first_local);
        for (index, source) in src_keys.iter().enumerate().skip(1) {
            if !frame.matches_tree(source) {
                return Err(CoreError::MalformedFusionTree {
                    message: "fusion-tree keys must share one group",
                });
            }
            locals.push(MultiplicityFreeTreeLocal::from_proven(
                group
                    .projection
                    .tree_at(index)
                    .expect("validated projection covers every source"),
            ));
        }
        Ok(Self { frame, locals })
    }
}

pub(crate) struct CompactMultiplicityFreeTreePairBasis {
    pub(super) frame: MultiplicityFreeTreePairFrame,
    pub(super) locals: Vec<MultiplicityFreeTreePairLocal>,
}

pub(super) struct CompactMultiplicityFreeTreePairBlock<S> {
    pub(super) basis: CompactMultiplicityFreeTreePairBasis,
    pub(super) columns: DenseColumns<S>,
    pub(super) records_dimensions: bool,
}

pub(super) type CompactMultiplicityFreeTreePairRows<S> = Vec<Vec<(FusionTreePairKey, S)>>;

impl CompactMultiplicityFreeTreePairBasis {
    pub(crate) fn from_group<R>(
        group: ValidatedTreePairBlockGroup<'_, R>,
    ) -> Result<Self, CoreError> {
        let (frame, first_local) =
            MultiplicityFreeTreePairFrame::split(group.projection.pair_at(0).ok_or(
                CoreError::MalformedFusionTree {
                    message: "compact block basis requires at least one source",
                },
            )?);
        let mut locals = Vec::with_capacity(group.source_len);
        locals.push(first_local);
        for index in 1..group.source_len {
            let source = group
                .projection
                .pair_at(index)
                .expect("validated projection covers every source");
            if !frame.matches_tree_pair_ref(source) {
                return Err(CoreError::MalformedFusionTree {
                    message: TREE_PAIR_BLOCK_GROUP_ERROR,
                });
            }
            locals.push(MultiplicityFreeTreePairLocal::from_proven(source));
        }
        Ok(Self { frame, locals })
    }
}

/// Sparse source columns of a Generic keyed block: TensorKit's coefficient
/// matrix `U` of `fsbraid`/`fstranspose`, stored column-major and sparse.
/// `entries[column_start[n]..column_start[n + 1]]` are source `n`'s present
/// `(basis row, coefficient)` pairs. Structural absence is a missing entry, so
/// a present zero stays distinct from an absent coefficient.
///
/// Why not the dense `DenseColumns` of the multiplicity-free compact drivers:
/// spreading a move over `B_j × N` slots costs `O(K_j·N)` per step even when no
/// intermediate tree is shared, an `N`-fold scale regression against
/// per-source composition when sharing is low. Sparse columns cost
/// `Σ_x c_j(x)·k_j(x)` multiplies, exactly the per-source composition count.
struct GenericSourceColumns<S> {
    column_start: Vec<usize>,
    entries: Vec<(usize, S)>,
}

impl<S> GenericSourceColumns<S> {
    fn identity(source_count: usize) -> Self
    where
        S: CategoricalScalar,
    {
        Self {
            column_start: (0..=source_count).collect(),
            entries: (0..source_count).map(|row| (row, S::one())).collect(),
        }
    }

    fn source_count(&self) -> usize {
        self.column_start.len() - 1
    }

    fn column(&self, source: usize) -> &[(usize, S)] {
        &self.entries[self.column_start[source]..self.column_start[source + 1]]
    }
}

type GenericBlockState<S> = (Vec<FusionTreePairKey>, GenericSourceColumns<S>);

/// Elementary-step counters of the Generic block composer, for measurements
/// (`testing` feature only).
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenericBlockStepCounts {
    /// Whole-basis elementary moves applied (`S`, summed over groups).
    pub steps: u64,
    /// Move invocations, one per distinct basis row (`Σ_j B_j`).
    pub moves: u64,
    /// Move invocations a per-source composition of the same reachable
    /// terms performs (`Σ_j Σ_x c_j(x)`).
    pub per_source_moves: u64,
    /// Coefficient spread multiplies (`Σ_j Σ_x c_j(x)·k_j(x)`).
    pub spread_multiplies: u64,
}

/// This thread's composer step counts since the last call.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub fn take_generic_block_step_counts() -> GenericBlockStepCounts {
    crate::probe::take_steps()
}

/// Apply one move to the whole current basis (TensorKit `U = U_tmp * U` on a
/// `FusionTreeBlock`): the move runs once per distinct basis row, in row
/// order, before any coefficient is spread. That step-major visit order is the
/// approved checked first-error order (#1962). Destination rows are numbered
/// by first appearance over `(row, move output)`, as in
/// [`compose_block_terms`]; each source column then accumulates
/// `step · source` over its own entries in their stored order.
fn compose_generic_block<S, E, F, I>(
    basis: &[FusionTreePairKey],
    columns: &GenericSourceColumns<S>,
    mut transform: F,
) -> Result<GenericBlockState<S>, E>
where
    S: Clone + Add<Output = S> + Mul<Output = S>,
    F: FnMut(&FusionTreePairKey) -> Result<I, E>,
    I: IntoIterator<Item = (FusionTreePairKey, S)>,
{
    let mut index: FxHashMap<FusionTreePairKey, usize> = FxHashMap::default();
    let mut move_start = Vec::with_capacity(basis.len() + 1);
    let mut move_terms = Vec::with_capacity(basis.len());
    move_start.push(0);
    for key in basis {
        for (destination, coefficient) in transform(key)? {
            let next_row = index.len();
            let row = *index.entry(destination).or_insert(next_row);
            move_terms.push((row, coefficient));
        }
        move_start.push(move_terms.len());
    }
    let mut slots: Vec<Option<FusionTreePairKey>> = (0..index.len()).map(|_| None).collect();
    for (key, row) in index {
        slots[row] = Some(key);
    }
    let next_basis: Vec<FusionTreePairKey> = slots
        .into_iter()
        .map(|key| key.expect("dense rows 0..len are all filled"))
        .collect();

    // `position[row]` is the entry of `row` in the column being built; a
    // position below the column's first entry belongs to an earlier column.
    let mut position = vec![usize::MAX; next_basis.len()];
    let mut column_start = Vec::with_capacity(columns.column_start.len());
    let mut entries: Vec<(usize, S)> = Vec::with_capacity(columns.entries.len());
    column_start.push(0);
    for source in 0..columns.source_count() {
        let begin = entries.len();
        for (row, source_coefficient) in columns.column(source) {
            for (next_row, step_coefficient) in &move_terms[move_start[*row]..move_start[*row + 1]]
            {
                let contribution = step_coefficient.clone() * source_coefficient.clone();
                let at = position[*next_row];
                if (begin..entries.len()).contains(&at) {
                    let existing = &mut entries[at].1;
                    *existing = existing.clone() + contribution;
                } else {
                    position[*next_row] = entries.len();
                    entries.push((*next_row, contribution));
                }
            }
        }
        column_start.push(entries.len());
    }
    #[cfg(any(test, feature = "testing"))]
    crate::probe::add_step(
        basis.len() as u64,
        columns.entries.len() as u64,
        columns
            .entries
            .iter()
            .map(|(row, _)| (move_start[row + 1] - move_start[*row]) as u64)
            .sum::<u64>(),
    );
    Ok((
        next_basis,
        GenericSourceColumns {
            column_start,
            entries,
        },
    ))
}

/// Order a Generic block result source-major by first appearance (the
/// [`OrderedBlockLinearMap`] contract). Within one source, destinations follow
/// that source's column order. Cost is `O(nnz + B)` plus the `D × N` dense
/// matrix of a non-singleton result, which the group spec stores anyway.
fn order_generic_block<S>(
    (basis, columns): GenericBlockState<S>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    let source_count = columns.source_count();
    let mut ordered_row = vec![usize::MAX; basis.len()];
    let mut ordered_basis_rows = Vec::with_capacity(basis.len());
    for (row, _) in &columns.entries {
        if ordered_row[*row] == usize::MAX {
            ordered_row[*row] = ordered_basis_rows.len();
            ordered_basis_rows.push(*row);
        }
    }
    let mut slots = basis.into_iter().map(Some).collect::<Vec<_>>();
    let destinations = ordered_basis_rows
        .iter()
        .map(|&row| {
            slots[row]
                .take()
                .expect("ordered block rows contain each basis row once")
        })
        .collect::<Vec<_>>();
    let is_singleton = columns
        .column_start
        .windows(2)
        .all(|range| range[1] - range[0] == 1);
    let GenericSourceColumns {
        column_start,
        entries,
    } = columns;
    let storage = if is_singleton {
        let (destination_rows, coefficients) = entries
            .into_iter()
            .map(|(row, coefficient)| (ordered_row[row], coefficient))
            .unzip();
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        }
    } else {
        let mut dense: Vec<Option<S>> = (0..destinations.len().saturating_mul(source_count))
            .map(|_| None)
            .collect();
        let mut source = 0;
        for (entry, (row, coefficient)) in entries.into_iter().enumerate() {
            while entry >= column_start[source + 1] {
                source += 1;
            }
            dense[ordered_row[row] * source_count + source] = Some(coefficient);
        }
        OrderedBlockLinearStorage::DenseDstSrc(dense)
    };
    OrderedBlockLinearMap {
        destinations,
        source_count,
        storage,
    }
}

/// The identity transform of a block: every source maps to itself with
/// coefficient one, without seeding or spreading a coefficient matrix.
fn identity_generic_block<S: CategoricalScalar>(
    src_keys: Vec<FusionTreePairKey>,
) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    let source_count = src_keys.len();
    OrderedBlockLinearMap {
        destinations: src_keys,
        source_count,
        storage: OrderedBlockLinearStorage::SingletonColumns {
            destination_rows: (0..source_count).collect(),
            coefficients: (0..source_count).map(|_| S::one()).collect(),
        },
    }
}

/// A whole-block Generic transform before it is put in a consumer's form:
/// the transform builders' [`OrderedBlockLinearMap`] or the trace lowering's
/// [`BlockSourceColumns`].
enum GenericBlockOutcome<S> {
    /// Every source maps to itself with coefficient one (no move ran).
    Identity(Vec<FusionTreePairKey>),
    Composed(GenericBlockState<S>),
}

impl<S: CategoricalScalar> GenericBlockOutcome<S> {
    fn into_ordered(self) -> OrderedBlockLinearMap<FusionTreePairKey, S> {
        match self {
            Self::Identity(src_keys) => identity_generic_block(src_keys),
            Self::Composed(state) => order_generic_block(state),
        }
    }

    /// Moves the composer's sparse columns out, with destinations numbered
    /// and each column ordered as [`order_generic_block`] does
    /// (source-major first appearance). Why keep that order rather than the
    /// composer's insertion order: trace lowering visits a source's rows in
    /// it, so it fixes the order of lowering provider queries (and first
    /// errors) and the accumulation order of terms. Cost `O(nnz log c + D)`,
    /// `c` the widest column, on a build only.
    fn into_source_columns(self) -> BlockSourceColumns<FusionTreePairKey, S> {
        let (basis, columns) = match self {
            Self::Identity(src_keys) => {
                return BlockSourceColumns::identity(src_keys.into_boxed_slice())
            }
            Self::Composed(state) => state,
        };
        let GenericSourceColumns {
            column_start,
            mut entries,
        } = columns;
        let mut ordered_row = vec![usize::MAX; basis.len()];
        let mut ordered_basis_rows = Vec::with_capacity(basis.len());
        for (row, _) in &mut entries {
            if ordered_row[*row] == usize::MAX {
                ordered_row[*row] = ordered_basis_rows.len();
                ordered_basis_rows.push(*row);
            }
            *row = ordered_row[*row];
        }
        // A column holds each row once (the composer accumulates repeats).
        for range in column_start.windows(2) {
            entries[range[0]..range[1]].sort_unstable_by_key(|(row, _)| *row);
        }
        let mut slots = basis.into_iter().map(Some).collect::<Vec<_>>();
        let destinations = ordered_basis_rows
            .into_iter()
            .map(|row| {
                slots[row]
                    .take()
                    .expect("ordered block rows contain each basis row once")
            })
            .collect();
        BlockSourceColumns::new(
            destinations,
            column_start.into_boxed_slice(),
            entries.into_boxed_slice(),
        )
    }
}

/// The one Generic keyed-block driver of the shared block schedule, over the
/// fallible symbol access (a checked provider, or the test-only infallible
/// adapter). Each hook is one or two whole-basis moves of the unchanged
/// `GenericK` kernels, as TensorKit applies `artin_braid`, `bendleft`,
/// `bendright`, `foldright` and `foldleft` to a whole `FusionTreeBlock`.
struct GenericTreePairBlockDriver<'a, C> {
    rule: &'a C,
}

impl<C> BlockDriver for GenericTreePairBlockDriver<'_, C>
where
    C: GenericRigidAccess,
{
    type State = GenericBlockState<C::Scalar>;
    type Error = CheckedGenericSymbolError<C::Error>;
    type BraidSchedule<'s> = &'s [PreparedArtinStep];

    fn bend(
        &mut self,
        (basis, columns): Self::State,
        bend: Bend,
    ) -> Result<Self::State, Self::Error> {
        let rule = self.rule;
        compose_generic_block(&basis, &columns, |key| match bend {
            Bend::Left => generic_bendleft_tree_pair_result(rule, key),
            Bend::Right => generic_bendright_tree_pair_result(rule, key),
        })
    }

    /// TensorKit `fsbraid` (`braiding_manipulations.jl:302-339`): one
    /// whole-block `artin_braid` per adjacent swap, `U = U_tmp * U`.
    fn braid_codomain(
        &mut self,
        mut state: Self::State,
        steps: &[PreparedArtinStep],
    ) -> Result<Self::State, Self::Error> {
        let rule = self.rule;
        for step in steps {
            state = compose_generic_block(&state.0, &state.1, |key| {
                generic_artin_braid_at_with_inverse_checked(
                    rule,
                    key.codomain_tree(),
                    step.index,
                    step.inverse,
                )
                .map(|terms| with_domain(key, terms))
            })?;
        }
        Ok(state)
    }

    /// TensorKit `cycleclockwise`/`cycleanticlockwise` on a `FusionTreeBlock`
    /// (`duality_manipulations.jl:401-440`): fold then bend, or bend then fold
    /// when the side the fold would take from is empty. Every row of a block
    /// shares one split, so the first row decides for the whole basis.
    fn cycle(
        &mut self,
        state: Self::State,
        direction: PreparedCycleDirection,
    ) -> Result<Self::State, Self::Error> {
        let rule = self.rule;
        let fold = |key: &FusionTreePairKey| match direction {
            PreparedCycleDirection::Clockwise => generic_foldright_tree_pair_result(rule, key),
            PreparedCycleDirection::Anticlockwise => left_move_by_swap(key, |swapped| {
                generic_foldright_tree_pair_result(rule, swapped)
            }),
        };
        let bend = |key: &FusionTreePairKey| match direction {
            PreparedCycleDirection::Clockwise => generic_bendleft_tree_pair_result(rule, key),
            PreparedCycleDirection::Anticlockwise => generic_bendright_tree_pair_result(rule, key),
        };
        let bend_first = state.0.first().is_some_and(|key| match direction {
            PreparedCycleDirection::Clockwise => key.codomain_tree().uncoupled().is_empty(),
            PreparedCycleDirection::Anticlockwise => key.domain_tree().uncoupled().is_empty(),
        });
        let (basis, columns) = state;
        if bend_first {
            let (basis, columns) = compose_generic_block(&basis, &columns, bend)?;
            compose_generic_block(&basis, &columns, fold)
        } else {
            let (basis, columns) = compose_generic_block(&basis, &columns, fold)?;
            compose_generic_block(&basis, &columns, bend)
        }
    }
}

fn empty_generic_block<S>() -> OrderedBlockLinearMap<FusionTreePairKey, S> {
    OrderedBlockLinearMap {
        destinations: Vec::new(),
        source_count: 0,
        storage: OrderedBlockLinearStorage::SingletonColumns {
            destination_rows: Vec::new(),
            coefficients: Vec::new(),
        },
    }
}

fn validate_generic_block_split(
    src_keys: &[FusionTreePairKey],
    codomain_rank: usize,
    domain_rank: usize,
) -> Result<(), CoreError> {
    for source in src_keys {
        if source.codomain_tree().uncoupled().len() != codomain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: codomain_rank,
                actual: source.codomain_tree().uncoupled().len(),
            });
        }
        if source.domain_tree().uncoupled().len() != domain_rank {
            return Err(CoreError::DimensionMismatch {
                expected: domain_rank,
                actual: source.domain_tree().uncoupled().len(),
            });
        }
    }
    Ok(())
}

/// Whole-block Generic braid: TensorKit `fsbraid` on one `FusionTreeBlock`
/// (`braiding_manipulations.jl:302-339`) as selected per source block by
/// `GenericTreeTransformer` (`treetransformers.jl:53-114`), TensorKit
/// `cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91`. Repartition into the codomain,
/// apply each Artin step to the whole basis, repartition to the target split.
///
/// QSpace `d2d3d7da` has no ordered Artin/tree-block schedule; the borrowed
/// technique is its grouped structural reuse (`QSpace::Permute`,
/// `CRef::Permute`/`SortDegQ`, `X3Map::contractDegQ`): every elementary move
/// runs once per distinct intermediate tree of the group and is shared by all
/// source columns.
///
/// Rust deviations: provider failures are typed and short-circuit at the
/// first failing row of the first failing step; TensorKit keeps the full dense
/// block `|D_j| × N`, here only the reachable rows with sparse source columns.
///
/// `src_keys` must be admitted for `rule`; the callers own that proof.
fn generic_braid_block_result<C>(
    rule: &C,
    src_keys: Vec<FusionTreePairKey>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    generic_braid_block_outcome(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )
    .map(GenericBlockOutcome::into_ordered)
}

fn generic_braid_block_outcome<C>(
    rule: &C,
    src_keys: Vec<FusionTreePairKey>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<GenericBlockOutcome<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    let Some(first) = src_keys.first() else {
        return Ok(GenericBlockOutcome::Identity(src_keys));
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    if codomain_levels.len() != codomain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: codomain_rank,
            actual: codomain_levels.len(),
        }
        .into());
    }
    if domain_levels.len() != domain_rank {
        return Err(CoreError::DimensionMismatch {
            expected: domain_rank,
            actual: domain_levels.len(),
        }
        .into());
    }
    validate_generic_block_split(&src_keys, codomain_rank, domain_rank)?;
    let GenericBraidSchedule {
        permutation,
        identity,
        steps,
    } = generic_braid_schedule(
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )?;
    if identity {
        return Ok(GenericBlockOutcome::Identity(src_keys));
    }
    let columns = GenericSourceColumns::identity(src_keys.len());
    let state = block_braid(
        &mut GenericTreePairBlockDriver { rule },
        (src_keys, columns),
        codomain_rank,
        permutation.len(),
        codomain_permutation.len(),
        &steps,
    )?;
    Ok(GenericBlockOutcome::Composed(state))
}

fn generic_permute_block_result<C>(
    rule: &C,
    src_keys: Vec<FusionTreePairKey>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    generic_permute_block_outcome(rule, src_keys, codomain_permutation, domain_permutation)
        .map(GenericBlockOutcome::into_ordered)
}

fn generic_permute_block_outcome<C>(
    rule: &C,
    src_keys: Vec<FusionTreePairKey>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<GenericBlockOutcome<C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        }
        .into());
    }
    let Some(first) = src_keys.first() else {
        return Ok(GenericBlockOutcome::Identity(src_keys));
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    generic_braid_block_outcome(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
        &codomain_levels,
        &domain_levels,
    )
}

/// Whole-block Generic planar transpose: TensorKit `fstranspose`
/// (`duality_manipulations.jl:540-566`, `cfaa073e`) on one `FusionTreeBlock`:
/// repartition, then whole-block `cycleanticlockwise`/`cycleclockwise` moves.
/// Same QSpace correspondence, Rust deviations and admission precondition as
/// [`generic_braid_block_result`].
fn generic_transpose_block_result<C>(
    rule: &C,
    src_keys: Vec<FusionTreePairKey>,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: GenericRigidAccess,
{
    let Some(first) = src_keys.first() else {
        return Ok(empty_generic_block());
    };
    let codomain_rank = first.codomain_tree().uncoupled().len();
    let domain_rank = first.domain_tree().uncoupled().len();
    let permutation = linearize_tree_pair_permutation(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    )?;
    if !is_cyclic_permutation(&permutation) {
        return Err(CoreError::InvalidPermutation {
            permutation,
            rank: codomain_rank + domain_rank,
        }
        .into());
    }
    validate_generic_block_split(&src_keys, codomain_rank, domain_rank)?;
    let position = permutation.iter().position(|&axis| axis == 0);
    let identity = tree_pair_axis_map_is_identity(
        codomain_permutation,
        domain_permutation,
        codomain_rank,
        domain_rank,
    );
    let Some(position) = position.filter(|_| !identity) else {
        return Ok(identity_generic_block(src_keys));
    };
    let columns = GenericSourceColumns::identity(src_keys.len());
    let state = block_transpose(
        &mut GenericTreePairBlockDriver { rule },
        (src_keys, columns),
        codomain_rank,
        codomain_permutation.len(),
        transpose_cycles(position, codomain_rank + domain_rank),
    )?;
    Ok(order_generic_block(state))
}

/// Proof that every fusion-tree block key of one exact [`BlockStructure`] is
/// admitted by one checked Generic provider. It lets the whole-block checked
/// composers skip per-source re-validation without letting an unadmitted
/// vertex label reach an F/R block lookup: the proof can only be built by
/// [`Self::try_new`], which validates every key.
///
/// Why not make this crate-private: `symtn-tensors` builds it in its checked
/// plan preflight and consumes it across the crate boundary.
#[doc(hidden)]
pub struct CheckedGenericAdmittedFusionTreeBlockStructure<'provider, 'structure, P> {
    provider: &'provider P,
    structure: &'structure BlockStructure,
}

#[doc(hidden)]
impl<'provider, 'structure, P>
    CheckedGenericAdmittedFusionTreeBlockStructure<'provider, 'structure, P>
where
    P: CheckedGenericFusion,
{
    /// Validate every block key with `provider`, in block order. The first
    /// failure is returned; nothing is retained.
    pub fn try_new(
        provider: &'provider P,
        structure: &'structure BlockStructure,
    ) -> Result<Self, CheckedGenericStructureError<P::Error>> {
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                return Err(CoreError::ExpectedFusionTreePairKey {
                    actual: block.key().kind(),
                }
                .into());
            };
            validate_generic_fusion_tree_pair_checked(provider, key)?;
        }
        Ok(Self {
            provider,
            structure,
        })
    }

    /// [`Self::try_new`] for a caller that already read `identity` from
    /// `provider`: a success is recorded on `structure`, and a later call
    /// with an equal identity makes no provider query (#2233). TensorKit
    /// (`cfaa073e`) likewise enumerates a HomSpace's trees once
    /// (`src/spaces/structure.jl:sectorstructure`) and never re-validates
    /// them per transform (`add_transform!`). Why not inside `try_new`: an
    /// identity read there would add a provider query to every admission.
    pub fn try_new_once(
        provider: &'provider P,
        identity: &RuleIdentity,
        structure: &'structure BlockStructure,
    ) -> Result<Self, CheckedGenericStructureError<P::Error>> {
        if structure.is_checked_generic_admitted(identity) {
            return Ok(Self {
                provider,
                structure,
            });
        }
        let admitted = Self::try_new(provider, structure)?;
        structure.record_checked_generic_admitted(identity.clone());
        Ok(admitted)
    }

    #[inline]
    pub fn structure(&self) -> &'structure BlockStructure {
        self.structure
    }

    fn admitted_keys(&self, block_indices: &[usize]) -> Result<Vec<FusionTreePairKey>, CoreError> {
        block_indices
            .iter()
            .map(|&index| match self.structure.block(index)?.key() {
                BlockKey::FusionTree(key) => Ok(key.clone()),
                key => Err(CoreError::ExpectedFusionTreePairKey { actual: key.kind() }),
            })
            .collect()
    }
}

#[doc(hidden)]
impl<P> CheckedGenericAdmittedFusionTreeBlockStructure<'_, '_, P>
where
    P: CheckedGenericRigidSymbols,
{
    /// Whole-block checked Generic braid of the admitted blocks
    /// `block_indices`, which must share one fusion-tree group.
    pub fn generic_braid_ordered_for_block_indices(
        &self,
        block_indices: &[usize],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
        codomain_levels: &[usize],
        domain_levels: &[usize],
    ) -> Result<
        OrderedBlockLinearMap<FusionTreePairKey, P::Scalar>,
        CheckedGenericSymbolError<P::Error>,
    > {
        generic_braid_block_result(
            self.provider,
            self.admitted_keys(block_indices)?,
            codomain_permutation,
            domain_permutation,
            codomain_levels,
            domain_levels,
        )
    }

    /// Whole-block checked Generic permutation (symmetric braiding only).
    pub fn generic_permute_ordered_for_block_indices(
        &self,
        block_indices: &[usize],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<
        OrderedBlockLinearMap<FusionTreePairKey, P::Scalar>,
        CheckedGenericSymbolError<P::Error>,
    > {
        generic_permute_block_result(
            self.provider,
            self.admitted_keys(block_indices)?,
            codomain_permutation,
            domain_permutation,
        )
    }

    /// Whole-block checked Generic planar transpose.
    pub fn generic_transpose_ordered_for_block_indices(
        &self,
        block_indices: &[usize],
        codomain_permutation: &[usize],
        domain_permutation: &[usize],
    ) -> Result<
        OrderedBlockLinearMap<FusionTreePairKey, P::Scalar>,
        CheckedGenericSymbolError<P::Error>,
    > {
        generic_transpose_block_result(
            self.provider,
            self.admitted_keys(block_indices)?,
            codomain_permutation,
            domain_permutation,
        )
    }
}

/// Whole-block checked Generic permutation of one fusion-tree group of
/// `structure`, read in `orientation`: the checked twin of
/// [`multiplicity_free_permute_tree_pair_block_indexed`] for trace lowering.
///
/// Each oriented source key of `src_indices` is validated once, in order,
/// then the group is recoupled step-major as [`generic_permute_block_result`]
/// and returned as its present per-source columns
/// (TensorKit `cfaa073e` `tensoroperations.jl:_trace_permute!`: one
/// permutation matrix per fusion block). Why not
/// [`CheckedGenericAdmittedFusionTreeBlockStructure`]: it admits the whole
/// stored structure up front, while trace admits one group at a time (so a
/// later group's error cannot overtake an earlier source's lowering error)
/// and permutes the adjoint-oriented keys of a lazy adjoint source.
#[doc(hidden)]
pub fn generic_permute_tree_pair_block_indexed_checked<C>(
    provider: &C,
    structure: &BlockStructure,
    src_indices: &[usize],
    orientation: FusionTreePairOrientation,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<BlockSourceColumns<FusionTreePairKey, C::Scalar>, CheckedGenericSymbolError<C::Error>>
where
    C: CheckedGenericRigidSymbols,
{
    let mut src_keys = Vec::with_capacity(src_indices.len());
    for &index in src_indices {
        let block = structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(CoreError::ExpectedFusionTreePairKey {
                actual: block.key().kind(),
            }
            .into());
        };
        let key = match orientation {
            FusionTreePairOrientation::Direct => key.clone(),
            FusionTreePairOrientation::Adjoint => {
                FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
            }
        };
        validate_generic_fusion_tree_pair_checked(provider, &key)
            .map_err(map_checked_generic_structure_error)?;
        src_keys.push(key);
    }
    generic_permute_block_outcome(provider, src_keys, codomain_permutation, domain_permutation)
        .map(GenericBlockOutcome::into_source_columns)
}

/// Validate infallible-rule keys as the test-only block entries' seed did.
#[cfg(any(test, feature = "testing"))]
fn admit_generic_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
) -> Result<Vec<FusionTreePairKey>, CoreError>
where
    R: GenericFusionSymbols,
    R::Scalar: CategoricalScalar,
{
    if rule.fusion_style() != FusionStyleKind::Generic {
        return Err(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: rule.fusion_style(),
        });
    }
    src_keys
        .iter()
        .map(|source| validate_fusion_tree_pair_for_rule(rule, source).map(|_| source.clone()))
        .collect()
}

#[doc(hidden)]
#[cfg(any(test, feature = "testing"))]
pub fn generic_braid_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let src_keys = admit_generic_tree_pair_block(rule, src_keys)?;
    generic_braid_block_result(
        &InfallibleGenericFR(rule),
        src_keys,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )
    .map_err(map_infallible_generic_symbol_error)
}

#[doc(hidden)]
#[cfg(any(test, feature = "testing"))]
pub fn generic_permute_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    if !rule.braiding_style().is_symmetric() {
        return Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: rule.braiding_style(),
        });
    }
    let src_keys = admit_generic_tree_pair_block(rule, src_keys)?;
    generic_permute_block_result(
        &InfallibleGenericFR(rule),
        src_keys,
        codomain_permutation,
        domain_permutation,
    )
    .map_err(map_infallible_generic_symbol_error)
}

#[doc(hidden)]
#[cfg(any(test, feature = "testing"))]
pub fn generic_transpose_tree_pair_block_ordered<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<OrderedBlockLinearMap<FusionTreePairKey, R::Scalar>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    let src_keys = admit_generic_tree_pair_block(rule, src_keys)?;
    generic_transpose_block_result(
        &InfallibleGenericFR(rule),
        src_keys,
        codomain_permutation,
        domain_permutation,
    )
    .map_err(map_infallible_generic_symbol_error)
}
