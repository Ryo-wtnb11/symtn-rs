use super::*;

use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use symtn_core::{
    ErasedStructureCacheControl, StructureCache, StructureCacheEquivalent, StructureCacheInfo,
    StructureCacheKind,
};
use tcontract_exec::WorkModel;

use crate::block_kernel::driver::OpKind;
use crate::block_kernel::gemm::ENTRY_OVERHEAD_BYTES;

/// The working tables of [`allocate`], kept for reuse: the grant keys, the
/// LPT order and heaps, the equal-share columns, the reservation's tables
/// and lanes, and per-lane loads and counts.
#[derive(Debug, Default)]
pub(super) struct PlaceScratch {
    keys: Vec<GrantKey>,
    order: Vec<usize>,
    load: Vec<Reverse<(Duration, usize, usize)>>,
    narrow: Vec<Reverse<(Duration, usize, usize)>>,
    /// Per lane: the baseline's modeled load, then the compaction's counts.
    lane_load: Vec<Duration>,
    count: Vec<usize>,
    columns: Columns,
    moldable: Moldable,
}

impl PlaceScratch {
    pub(super) fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        words(self.order.capacity() + self.count.capacity())
            .saturating_add(self.keys.capacity().saturating_mul(size_of::<GrantKey>()))
            .saturating_add(
                self.load
                    .capacity()
                    .saturating_add(self.narrow.capacity())
                    .saturating_mul(size_of::<Reverse<(Duration, usize, usize)>>()),
            )
            .saturating_add(
                self.lane_load
                    .capacity()
                    .saturating_mul(size_of::<Duration>()),
            )
            .saturating_add(self.columns.retained_bytes())
            .saturating_add(self.moldable.retained_bytes())
    }
}

pub(super) fn words(n: usize) -> usize {
    n.saturating_mul(std::mem::size_of::<usize>())
}

/// Records one comparison sort of the placement's working tables.
fn sorted() {
    #[cfg(test)]
    crate::probe::add(crate::probe::Event::PlacementSort, 1);
}

/// A job's modeled time in a lane where its provider grants it `width`
/// slots: its serial time `serial` when the grant is one slot, otherwise
/// `serial / width` plus the token's dispatch cost at that width, in integer
/// nanoseconds.
pub(crate) fn item_time(exec: &Exec<'_>, serial: Duration, width: usize) -> Duration {
    match NonZeroUsize::new(width).filter(|w| w.get() > 1) {
        None => serial,
        Some(w) => (serial / u32::try_from(w.get()).unwrap_or(u32::MAX))
            .saturating_add(exec.token_ref().dispatch_cost(w)),
    }
}

/// Whether an allocation of jobs with serial times `times` and mean work
/// `mean_work` on `exec` can search: Auto, at least two jobs, a batch
/// decision of at least two lanes and at least two slots (the lane cap and
/// slot count [`allocate`] derives). Otherwise it returns the baseline,
/// whatever the grant.
pub(crate) fn searchable(exec: &Exec<'_>, times: &[Duration], mean_work: f64) -> bool {
    exec.batch_policy().strategy == BatchStrategy::Auto
        && exec.budget() >= 2
        && times.len() >= 2
        && exec
            .batch_decision(&BatchQuery::new(times.len(), mean_work))
            .lanes
            >= 2
        && exec
            .token_ref()
            .width_for(
                times
                    .iter()
                    .fold(Duration::ZERO, |sum, &t| sum.saturating_add(t)),
                1,
            )
            .get()
            >= 2
}

/// Lane allocation of jobs with serial times `times` (canonical order) and
/// mean weighted work `mean_work` (multiply-adds; it feeds the
/// `BatchQuery`, while `times` feed `width_for`, so neither is converted
/// twice) into `tables`, as the module documentation describes.
///
/// `grant(j, times[j], w)` is the width job `j`'s provider takes in a lane of
/// `w` reserved slots, `key(j)` its other inputs (see [`GrantKey`]) and
/// `family` the op kind whose grant function `grant` is; Auto scores the
/// candidates with it, and takes a searched placement from `cache` when one
/// was computed from the same inputs (see [`LaneAllocationKey`]), offering
/// each new search for admission. Without a grant, and under a forced
/// strategy, the baseline placement is returned and no candidate is scored;
/// `key` is then not called, and the cache is not read.
///
/// Why a lookup and a publish rather than the cache's single-flight build:
/// that build takes an owned key, which a hit would have to stage. Racing
/// misses of one key each search, and the losers count as duplicate builds.
pub(crate) fn allocate(
    exec: &Exec<'_>,
    times: &[Duration],
    (family, key): (OpKind, impl Fn(usize) -> GrantKey),
    mean_work: f64,
    model: Option<impl JobModel>,
    cache: &LaneAllocationCache,
    tables: &mut LaneTables,
) {
    // Captured before any input is read, so a reset during the search
    // keeps its result out of the cache.
    let epoch = symtn_core::core_reset_epoch();
    let LaneTables {
        placement, scratch, ..
    } = tables;
    placement.clear();
    // The keys are staged only where a search can run, so a serial context
    // reads no grant input.
    let mut keys = std::mem::take(&mut scratch.keys);
    keys.clear();
    if model.is_some() && searchable(exec, times, mean_work) {
        keys.extend((0..times.len()).map(key));
    }
    // Staged keys mean a search: `searchable` admits only lane counts of two
    // or more, so the search branch of `place` runs.
    let query = (!keys.is_empty()).then_some(LaneAllocationQuery {
        family,
        exec,
        times,
        keys: &keys,
        mean_work,
    });
    if let Some(entry) = query.as_ref().and_then(|query| cache.get(query)) {
        placement.restore(&entry);
    } else {
        place(exec, times, &keys, mean_work, model, placement, scratch);
        if let Some(query) = query {
            let bytes = charge(exec.budget(), times.len(), placement.lanes());
            if bytes > cache.info().max_entry_bytes() {
                cache.reject();
            } else {
                cache.publish(&query.to_key(), Arc::new(placement.entry()), bytes, epoch);
            }
        }
    }
    scratch.keys = keys;
}

fn place(
    exec: &Exec<'_>,
    times: &[Duration],
    keys: &[GrantKey],
    mean_work: f64,
    model: Option<impl JobModel>,
    placement: &mut Placement,
    scratch: &mut PlaceScratch,
) {
    let jobs = times.len();
    let decision = exec.batch_decision(&BatchQuery::new(jobs, mean_work));
    if jobs == 0 {
        return;
    }
    let (slots, lanes) = if exec.batch_policy().strategy == BatchStrategy::Auto {
        let total = times
            .iter()
            .fold(Duration::ZERO, |sum, &t| sum.saturating_add(t));
        let slots = exec.token_ref().width_for(total, 1).get();
        (slots, slots.min(jobs).min(decision.lanes))
    } else {
        (
            decision.lanes * decision.item_threads,
            decision.lanes.min(jobs),
        )
    };
    let lanes = lanes.max(1);
    if lanes == 1 {
        placement.sequential(slots, jobs);
        return;
    }

    let mut order = std::mem::take(&mut scratch.order);
    order.clear();
    order.extend(0..jobs);
    order.sort_unstable_by_key(|&j| (Reverse(times[j]), j));
    sorted();
    // LPT, ties to the lowest lane. The job count breaks load ties first so
    // zero-time estimates still leave no reserved lane empty.
    let mut load = std::mem::take(&mut scratch.load);
    load.clear();
    load.extend((0..lanes).map(|k| Reverse((Duration::ZERO, 0, k))));
    let mut load = BinaryHeap::from(load);
    placement.lane_of.resize(jobs, 0);
    for &j in &order {
        if let Some(Reverse((sum, count, k))) = load.pop() {
            placement.lane_of[j] = k;
            load.push(Reverse((sum.saturating_add(times[j]), count + 1, k)));
        }
    }
    let mut load = load.into_vec();
    placement
        .shares
        .extend((0..lanes).map(|k| slots / lanes + usize::from(k < slots % lanes)));
    if let (BatchStrategy::Auto, Some(model)) = (exec.batch_policy().strategy, model) {
        let grant = |j: usize, width: usize| model.grant(j, times[j], width);
        let item_of = |j: usize, grant: Grant| model.item(exec, j, times[j], grant);
        let fill = |j: usize, width: usize| item_of(j, grant(j, width));
        let mut columns = std::mem::take(&mut scratch.columns);
        let mut narrow = std::mem::take(&mut scratch.narrow);
        columns.begin(times, keys, |j| grows(grant(j, slots), [1; STAGES]));
        columns.prepare((slots, lanes), &fill);
        let lane_load = &mut scratch.lane_load;
        lane_load.clear();
        lane_load.resize(lanes, Duration::ZERO);
        for (j, &k) in placement.lane_of.iter().enumerate() {
            lane_load[k] = lane_load[k].saturating_add(columns.get(j, placement.shares[k]));
        }
        let baseline = lane_load.iter().copied().max().unwrap_or(Duration::ZERO);
        let mut best = (baseline, None);
        // Ties keep the baseline, then the larger lane count. Shares only
        // widen as the count falls, so each width's column is filled once.
        for count in (1..=lanes).rev() {
            columns.prepare((slots, count), &fill);
            // A candidate that cannot score strictly lower is not placed:
            // the choice is the one the full sweep makes.
            if columns.lower_bound((slots, count)) >= best.0.as_nanos() {
                continue;
            }
            let item = |j: usize, width: usize| columns.get(j, width);
            let score = equal_shares(slots, count, &order, &item, &mut load, &mut narrow, None);
            if score < best.0 {
                best = (score, Some(count));
            }
        }
        let mut moldable = std::mem::take(&mut scratch.moldable);
        let score = moldable.place((slots, lanes), &order, times, &columns, (&item_of, &grant));
        // Ties keep the baseline and the equal shares.
        if score.is_some_and(|score| score < best.0) {
            best.1 = None;
            std::mem::swap(&mut placement.shares, &mut moldable.shares);
            std::mem::swap(&mut placement.lane_of, &mut moldable.lane_of);
        }
        scratch.moldable = moldable;
        if let Some(count) = best.1 {
            columns.prepare((slots, count), &fill);
            let item = |j: usize, width: usize| columns.get(j, width);
            equal_shares(
                slots,
                count,
                &order,
                &item,
                &mut load,
                &mut narrow,
                Some(placement),
            );
        }
        scratch.columns = columns;
        scratch.narrow = narrow;
    }
    scratch.order = order;
    scratch.load = load;
    compact(placement, &mut scratch.count);
    debug_assert!(
        placement.shares.iter().all(|&share| share > 0)
            && placement.shares.iter().sum::<usize>() <= slots
    );
}

/// Everything a searched placement is a function of, as the
/// [`StructureCacheKind::LaneAllocations`] cache keys it.
///
/// Why the key is complete: a searched placement is a function of the
/// context's budget B, its work model (the strategy is Auto, the only one
/// that searches), the mean work (the batch decision), the token's dispatch
/// cost at every width `2..=B` (the slot count, the batch decision, every
/// grant and every item time read only those), whether the token carries a
/// team (the Faer GEMM and the factorization grants), the jobs' serial times
/// in canonical order (the slot count, the order, the placements and the
/// scores), their grant keys (every other input of a job's stage widths and
/// item times) and the op kind, which fixes the model: every GEMM block op
/// grants through `gemm::grant` (keys: the plan's algorithm and the work,
/// last word zero); a single-stage factorization through `factor::grant`
/// (keys: the item's total and payback work, last word zero), and a
/// composite factorization job per stage, its factorization through
/// `factor::grant` and each product through `gemm::grant` on its plan (keys:
/// its extents and a nonzero tag of its op, its scalar and whether its plans
/// resolved, which fix its stages' work, geometries and plans), so equal keys
/// of different kinds or models never share a placement. Floats are keyed by
/// their bits, so equality is reflexive and a differently signed zero only
/// searches again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LaneAllocationKey {
    family: OpKind,
    budget: usize,
    model: [u64; 5],
    mean_work: u64,
    team: bool,
    costs: Box<[Duration]>,
    times: Box<[Duration]>,
    keys: Box<[GrantKey]>,
}

/// One call's [`LaneAllocationKey`], borrowed: a lookup reads the dispatch
/// costs from the token instead of staging them, so a hit allocates nothing.
struct LaneAllocationQuery<'a, 'e> {
    family: OpKind,
    exec: &'a Exec<'e>,
    times: &'a [Duration],
    keys: &'a [GrantKey],
    mean_work: f64,
}

/// The work model's fields, by bits.
fn model_bits(model: &WorkModel) -> [u64; 5] {
    [
        model.item_overhead_ns.to_bits(),
        model.muladds_per_ns.to_bits(),
        model.min_work_per_lane_ns.to_bits(),
        model.min_items as u64,
        model.min_items_per_lane as u64,
    ]
}

/// The token's dispatch cost at `width`.
fn dispatch_cost(exec: &Exec<'_>, width: NonZeroUsize) -> Duration {
    #[cfg(test)]
    crate::probe::add(crate::probe::Event::PlacementCostRead, 1);
    exec.token_ref().dispatch_cost(width)
}

/// The token's dispatch cost at every width `2..=budget`.
fn dispatch_costs<'a>(exec: &'a Exec<'_>) -> impl Iterator<Item = Duration> + 'a {
    (2..=exec.budget())
        .filter_map(NonZeroUsize::new)
        .map(|width| dispatch_cost(exec, width))
}

/// The dispatch costs a key's hash covers: those at widths 2 and B, the
/// ends of its `2..=B` row.
///
/// Why not the whole row: equality compares every width, so a row that
/// differs only inside still misses, and a lookup reads the token's row once
/// (to compare) instead of twice. A complete key that needs no read per
/// width needs a value of the token that fixes its row, which threadteam
/// does not expose.
fn cost_ends(exec: &Exec<'_>) -> [Option<Duration>; 2] {
    let end = |width| NonZeroUsize::new(width).map(|width| dispatch_cost(exec, width));
    match exec.budget() {
        budget if budget >= 2 => [end(2), end(budget)],
        _ => [None; 2],
    }
}

/// The one hash of a key and of its borrowed query, field by field.
#[allow(clippy::too_many_arguments)]
fn hash_key<H: Hasher>(
    state: &mut H,
    family: OpKind,
    budget: usize,
    model: [u64; 5],
    mean_work: u64,
    team: bool,
    cost_ends: [Option<Duration>; 2],
    times: &[Duration],
    keys: &[GrantKey],
) {
    (family, budget, model, mean_work, team, cost_ends).hash(state);
    times.hash(state);
    keys.hash(state);
}

impl Hash for LaneAllocationKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_key(
            state,
            self.family,
            self.budget,
            self.model,
            self.mean_work,
            self.team,
            [self.costs.first().copied(), self.costs.last().copied()],
            &self.times,
            &self.keys,
        );
    }
}

impl LaneAllocationQuery<'_, '_> {
    fn model(&self) -> [u64; 5] {
        model_bits(&self.exec.batch_policy().model)
    }

    fn team(&self) -> bool {
        self.exec.token_ref().carries_team()
    }

    fn to_key(&self) -> LaneAllocationKey {
        LaneAllocationKey {
            family: self.family,
            budget: self.exec.budget(),
            model: self.model(),
            mean_work: self.mean_work.to_bits(),
            team: self.team(),
            costs: dispatch_costs(self.exec).collect(),
            times: self.times.into(),
            keys: self.keys.into(),
        }
    }
}

impl Hash for LaneAllocationQuery<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_key(
            state,
            self.family,
            self.exec.budget(),
            self.model(),
            self.mean_work.to_bits(),
            self.team(),
            cost_ends(self.exec),
            self.times,
            self.keys,
        );
    }
}

impl StructureCacheEquivalent<LaneAllocationKey> for LaneAllocationQuery<'_, '_> {
    fn equivalent(&self, key: &LaneAllocationKey) -> bool {
        key.family == self.family
            && key.budget == self.exec.budget()
            && key.model == self.model()
            && key.mean_work == self.mean_work.to_bits()
            && key.team == self.team()
            && *key.times == *self.times
            && *key.keys == *self.keys
            && key.costs.len() + 1 == key.budget
            && key.costs.iter().copied().eq(dispatch_costs(self.exec))
    }
}

/// A searched placement, immutable: each lane's share and the end of its
/// jobs in `jobs`, whose jobs ascend within each lane.
#[derive(Debug)]
pub(crate) struct LaneAllocationEntry {
    lanes: Box<[(usize, usize)]>,
    jobs: Box<[usize]>,
}

pub(crate) type LaneAllocationCache = StructureCache<LaneAllocationKey, LaneAllocationEntry>;

/// The bytes charged for one cached placement of `jobs` jobs on `lanes`
/// lanes at budget `budget`: the inline key and entry, the entry overhead,
/// and their boxed slices (`B - 1` costs, J times, keys and job indices, one
/// pair per lane). Saturates. A function of the lengths alone, so an
/// oversized entry is refused before its owned key is staged.
fn charge(budget: usize, jobs: usize, lanes: usize) -> u64 {
    use std::mem::size_of;
    [
        size_of::<LaneAllocationKey>(),
        size_of::<LaneAllocationEntry>(),
        ENTRY_OVERHEAD_BYTES,
        budget
            .saturating_sub(1)
            .saturating_mul(size_of::<Duration>()),
        jobs.saturating_mul(size_of::<Duration>()),
        jobs.saturating_mul(size_of::<GrantKey>()),
        lanes.saturating_mul(size_of::<(usize, usize)>()),
        jobs.saturating_mul(size_of::<usize>()),
    ]
    .into_iter()
    .fold(0, |total: u64, bytes| total.saturating_add(bytes as u64))
}

struct LaneAllocationsControl;

impl ErasedStructureCacheControl for LaneAllocationsControl {
    fn info(&self) -> StructureCacheInfo {
        lane_allocations().info()
    }

    fn clear(&self) {
        ErasedStructureCacheControl::clear(lane_allocations())
    }

    fn set_byte_budget(&self, bytes: u64) {
        ErasedStructureCacheControl::set_byte_budget(lane_allocations(), bytes)
    }
}

/// The process-global lane-allocation cache, registered with `symtn-core`'s
/// structure-cache registry on first use.
pub(crate) fn lane_allocations() -> &'static LaneAllocationCache {
    static CACHE: OnceLock<LaneAllocationCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Refused only if another crate claimed the slot first. Then
        // `symtn::cache::clear` cannot reach this cache, so it admits
        // nothing: every batch searches per call, with identical results.
        let budget = symtn_core::register_structure_cache(
            StructureCacheKind::LaneAllocations,
            &LaneAllocationsControl,
        )
        .unwrap_or(0);
        StructureCache::new(StructureCacheKind::LaneAllocations, budget)
    })
}

impl Placement {
    /// The placement `entry` holds, in these tables' capacity.
    fn restore(&mut self, entry: &LaneAllocationEntry) {
        self.clear();
        self.lane_starts.push(0);
        for &(share, end) in entry.lanes.iter() {
            self.shares.push(share);
            self.lane_starts.push(end);
        }
        self.lane_jobs.extend_from_slice(&entry.jobs);
        self.lane_of.resize(entry.jobs.len(), 0);
        for lane in 0..self.shares.len() {
            for i in self.lane_starts[lane]..self.lane_starts[lane + 1] {
                self.lane_of[self.lane_jobs[i]] = lane;
            }
        }
    }

    /// This placement as a cache entry.
    fn entry(&self) -> LaneAllocationEntry {
        LaneAllocationEntry {
            lanes: self
                .shares
                .iter()
                .zip(&self.lane_starts[1..])
                .map(|(&share, &end)| (share, end))
                .collect(),
            jobs: self.lane_jobs.as_slice().into(),
        }
    }
}

/// Every job's modeled item time at two lane widths, so a width the
/// candidates share is granted once per job, not once per candidate. Jobs
/// whose provider stays serial at the full slot count are queried once and
/// never again.
#[derive(Debug, Default)]
struct Columns {
    /// The width of each column; zero (no lane has it) when unfilled.
    width: [usize; 2],
    items: [Vec<Duration>; 2],
    /// The jobs the provider widens at the full slot count, ascending.
    widened: Vec<usize>,
    /// Each job's representative: the lowest job with its serial time and
    /// grant key, whose grants it shares at every width.
    rep: Vec<usize>,
    /// Jobs ordered by serial time, grant key and index, to find `rep`.
    by_key: Vec<usize>,
    /// Whether each job is widened at the full slot count.
    widens: Vec<bool>,
    /// The sum and the largest of the other jobs' serial times, in ns.
    serial_total: u128,
    serial_largest: u128,
}

impl Columns {
    /// Starts an allocation of jobs with serial times `times` and grant keys
    /// `keys`: both columns hold every job's serial time, each job's
    /// representative is the lowest job with its serial time and key (whose
    /// grants it shares at every width, so only representatives are
    /// queried), and `widens(j)` (the provider's grant at the full slot
    /// count S is above one), asked of representatives only, splits the jobs.
    ///
    /// Why a job serial at S is serial at every share `w <= S`: each step of
    /// a grant is a released width rule that starts at width one and moves
    /// only to a width `k` in `2..=b` that both pays and strictly lowers its
    /// objective, where the payoff and the objective at `k` depend on the
    /// token's dispatch cost at `k` alone, not on the ceiling `b`
    /// (threadteam `width_for_by`, reached through `Exec::width_for_within`
    /// and `batch_decision_within`). A ceiling below `b` scans a prefix of
    /// the same candidates, so if none qualifies under `b` none qualifies
    /// under it either; and when the minimum under `b` sits at `q <= b'`
    /// it is also the minimum under `b'`, so a step's result never grows
    /// as its ceiling falls. The factorization grant is one such step. The
    /// GEMM grant's second step runs within the first step's result, which
    /// is thus at most its value at S: a Faer plan's one-item decision is
    /// its inner width rule or one thread (`faer_auto`, one item, no lanes),
    /// and a Packed plan's is its narrowed width (`packed_auto`, one item,
    /// no lanes), both width rules within that smaller ceiling. Hence a
    /// grant of one at S is one at every share, and such a job's item time is
    /// its serial time at every candidate width. A composite job's stages are
    /// each one such grant, so a job whose every stage is one at S is serial
    /// at every share, its item time the sum of its stages' serial times,
    /// which is its serial time.
    fn begin(&mut self, times: &[Duration], keys: &[GrantKey], widens: impl Fn(usize) -> bool) {
        self.width = [0; 2];
        for column in &mut self.items {
            column.clear();
            column.extend_from_slice(times);
        }
        self.by_key.clear();
        self.by_key.extend(0..times.len());
        self.by_key
            .sort_unstable_by_key(|&j| (times[j], keys[j], j));
        sorted();
        self.rep.clear();
        self.rep.resize(times.len(), 0);
        let mut first = None;
        for &j in &self.by_key {
            let start = match first {
                Some(f) if times[f] == times[j] && keys[f] == keys[j] => f,
                _ => j,
            };
            first = Some(start);
            self.rep[j] = start;
        }
        self.widened.clear();
        self.widens.clear();
        (self.serial_total, self.serial_largest) = (0, 0);
        for (j, t) in times.iter().enumerate() {
            let rep = self.rep[j];
            let wide = if rep == j {
                widens(j)
            } else {
                self.widens[rep]
            };
            self.widens.push(wide);
            if wide {
                self.widened.push(j);
            } else {
                self.serial_total = self.serial_total.saturating_add(t.as_nanos());
                self.serial_largest = self.serial_largest.max(t.as_nanos());
            }
        }
    }

    /// Keeps the columns of the widths of `count` equal shares of `slots`
    /// (`count <= slots`) and fills a missing one's widened jobs with
    /// `fill(job, width)`; the other jobs keep their serial time.
    fn prepare(
        &mut self,
        (slots, count): (usize, usize),
        fill: &impl Fn(usize, usize) -> Duration,
    ) {
        let narrow = slots / count;
        let widths = [narrow, narrow + 1];
        let widths = &widths[..1 + usize::from(slots % count != 0)];
        for &width in widths {
            if self.width.contains(&width) {
                continue;
            }
            let slot = usize::from(widths.contains(&self.width[0]));
            self.width[slot] = width;
            for &j in &self.widened {
                let rep = self.rep[j];
                self.items[slot][j] = if rep == j {
                    fill(j, width)
                } else {
                    self.items[slot][rep]
                };
            }
        }
    }

    /// A lower bound, in nanoseconds, on the score of the equal-share
    /// placement over `count` lanes of `slots` whose widths' columns are
    /// prepared: `max(max_j m_j, ceil(sum_j m_j / count))`, where `m_j` is
    /// job `j`'s smaller item time over the two share widths.
    ///
    /// Proof. Every job runs on exactly one lane, whose width is one of the
    /// two shares, so its item time there is at least `m_j` (the minimum,
    /// because the item time need not fall as the width grows). A lane's load
    /// is the sum of its jobs' item times, so the largest load is at least
    /// any one job's `m_j`, and at least the mean load, which is at least
    /// `sum_j m_j / count`; a load is whole nanoseconds, hence the ceiling.
    /// The placement saturates its loads at `Duration::MAX`, so its score is
    /// at least `min(bound, Duration::MAX)`; any score it is compared with
    /// is at most `Duration::MAX`, so `bound >= score` still proves the
    /// candidate cannot score strictly lower.
    /// The serial jobs' part is their precomputed sum and maximum, so the
    /// bound costs one step per widened job.
    fn lower_bound(&self, (slots, count): (usize, usize)) -> u128 {
        let narrow = slots / count;
        let wide = narrow + usize::from(slots % count != 0);
        let (mut largest, mut total) = (self.serial_largest, self.serial_total);
        for &job in &self.widened {
            let m = self.get(job, narrow).min(self.get(job, wide)).as_nanos();
            largest = largest.max(m);
            total = total.saturating_add(m);
        }
        largest.max(total.div_ceil(count as u128))
    }

    fn get(&self, job: usize, width: usize) -> Duration {
        debug_assert!(self.width.contains(&width));
        self.items[usize::from(self.width[0] != width)][job]
    }

    fn retained_bytes(&self) -> usize {
        self.items
            .iter()
            .map(|column| column.capacity())
            .fold(0, usize::saturating_add)
            .saturating_mul(std::mem::size_of::<Duration>())
            .saturating_add(
                (self.widened.capacity())
                    .saturating_add(self.rep.capacity())
                    .saturating_add(self.by_key.capacity())
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
            .saturating_add(self.widens.capacity())
    }
}

/// Whether some stage of `wider` takes more slots than in `narrower`.
fn grows(wider: Grant, narrower: Grant) -> bool {
    wider.iter().zip(&narrower).any(|(a, b)| a > b)
}

/// The large-job reservations: their target search tables (shared by both
/// layouts), and each layout's lanes (each job's lane and each lane's
/// share), kept for reuse.
#[derive(Debug, Default)]
struct Moldable {
    /// Row `i` holds the item time of the `i`-th largest job at every width
    /// `1..=S`, for the jobs above the mean load only.
    table: Vec<Duration>,
    targets: Vec<Duration>,
    /// `tail[r]`: the serial time of the jobs from the `r`-th largest on,
    /// in ns.
    tail: Vec<u128>,
    /// Each reserved lane's load and job count.
    wide: Vec<(Duration, usize)>,
    /// The width-one lanes, by `(load, jobs, lane)`.
    thin: Vec<Reverse<(Duration, usize, usize)>>,
    shares: Vec<usize>,
    lane_of: Vec<usize>,
    /// The other layout's lanes, swapped into `shares` and `lane_of` when it
    /// wins.
    other_shares: Vec<usize>,
    other_lane_of: Vec<usize>,
    grants: Grants,
}

/// The grants the reservations queried, each at most once per
/// (representative, width), in dense tables, so their capacities are their
/// whole allocation. A cell whose first stage is zero is not yet queried (no
/// width rule grants zero slots; such a grant would only be queried again).
#[derive(Debug, Default)]
struct Grants {
    /// S, the length of a `heavy` row.
    slots: usize,
    /// Each representative's row: below `heavy_rows` a row of `heavy`,
    /// otherwise of `tail`; `usize::MAX` for the jobs never queried.
    row: Vec<usize>,
    heavy_rows: usize,
    tail_rows: usize,
    /// The jobs above the mean load: every width `1..=S`, row-major.
    heavy: Vec<Grant>,
    /// Each width's column of `tail`, or `usize::MAX`.
    column: Vec<usize>,
    /// The other widened jobs, column-major over the widths they meet.
    tail: Vec<Grant>,
}

impl Grants {
    /// Gives a row to each widened representative: a `heavy` row to those
    /// of the `heavy` jobs, a `tail` row to the others.
    fn begin(&mut self, slots: usize, heavy: &[usize], columns: &Columns) {
        let jobs = columns.rep.len();
        self.slots = slots;
        self.row.clear();
        self.row.resize(jobs, usize::MAX);
        self.heavy_rows = 0;
        for &j in heavy {
            let r = columns.rep[j];
            if columns.widens[j] && self.row[r] == usize::MAX {
                self.row[r] = self.heavy_rows;
                self.heavy_rows += 1;
            }
        }
        self.tail_rows = 0;
        for j in 0..jobs {
            if columns.widens[j] && columns.rep[j] == j && self.row[j] == usize::MAX {
                self.row[j] = self.heavy_rows + self.tail_rows;
                self.tail_rows += 1;
            }
        }
        self.heavy.clear();
        self.heavy.resize(self.heavy_rows * slots, [0; STAGES]);
        self.column.clear();
        self.column.resize(slots + 1, usize::MAX);
        self.tail.clear();
    }

    /// Job `j`'s stage widths at `width <= S`: one slot per stage for a job
    /// whose providers keep every stage serial at S (see [`Columns::begin`]),
    /// otherwise its representative's, queried with `query(representative,
    /// width)` on first use.
    fn get(
        &mut self,
        columns: &Columns,
        j: usize,
        width: usize,
        query: &impl Fn(usize, usize) -> Grant,
    ) -> Grant {
        if !columns.widens[j] {
            return [1; STAGES];
        }
        let r = columns.rep[j];
        let row = self.row[r];
        let cell = if row < self.heavy_rows {
            &mut self.heavy[row * self.slots + width - 1]
        } else {
            if self.column[width] == usize::MAX {
                self.column[width] = self.tail.len() / self.tail_rows;
                self.tail
                    .resize(self.tail.len() + self.tail_rows, [0; STAGES]);
            }
            &mut self.tail[self.column[width] * self.tail_rows + row - self.heavy_rows]
        };
        if cell[0] == 0 {
            *cell = query(r, width);
        }
        *cell
    }

    fn retained_bytes(&self) -> usize {
        words(self.row.capacity() + self.column.capacity()).saturating_add(
            self.heavy
                .capacity()
                .saturating_add(self.tail.capacity())
                .saturating_mul(std::mem::size_of::<Grant>()),
        )
    }
}

/// `n` nanoseconds as a duration, saturating.
fn from_nanos(n: u128) -> Duration {
    Duration::new(
        u64::try_from(n / 1_000_000_000).unwrap_or(u64::MAX),
        (n % 1_000_000_000) as u32,
    )
}

/// What both reservation layouts read: the slot count and lane cap, the
/// order, the serial times and the item-time table of the jobs above the
/// mean load.
struct Reservation<'a> {
    slots: usize,
    cap: usize,
    order: &'a [usize],
    times: &'a [Duration],
    table: &'a [Duration],
    targets: &'a [Duration],
    tail: &'a [u128],
    large: usize,
}

impl Reservation<'_> {
    fn row(&self, i: usize) -> &[Duration] {
        &self.table[i * self.slots..(i + 1) * self.slots]
    }

    /// The least width at which the `i`-th largest job meets `target`
    /// (every target is at least each large job's least item time).
    fn need(&self, i: usize, target: Duration) -> usize {
        1 + self
            .row(i)
            .iter()
            .position(|&t| t <= target)
            .unwrap_or(self.slots - 1)
    }

    /// The large jobs above `target`, a prefix of the order.
    fn reserved(&self, target: Duration) -> usize {
        self.order[..self.large]
            .iter()
            .take_while(|&&j| self.times[j] > target)
            .count()
    }

    /// Whether `target` passes: the reserved jobs' least widths fit in S,
    /// and, for the exclusive layout, the other jobs' serial time fits on
    /// the one-slot lanes left, `tail <= (S - used) * target`.
    ///
    /// Why the least passing target is found by bisection: as the target
    /// rises each least width can only fall, and each job that leaves the
    /// reservation frees at least one slot while adding at most the new
    /// target to the tail, so a passing target keeps passing.
    fn passes(&self, target: Duration, exclusive: bool) -> bool {
        let reserved = self.reserved(target);
        let used = (0..reserved)
            .map(|i| self.need(i, target))
            .fold(0usize, usize::saturating_add);
        if used > self.slots {
            return false;
        }
        if !exclusive {
            return true;
        }
        let free = (self.slots - used) as u128;
        (reserved == self.order.len() || free >= 1)
            && self.tail[reserved] <= free.saturating_mul(target.as_nanos())
    }
}

impl Moldable {
    /// Places the jobs, `order` largest first, on lanes of uneven widths
    /// over `slots` with two reservations, and returns the better one's
    /// modeled makespan (ties keep the shared layout), its lanes in
    /// `self.shares` and `self.lane_of`, including the empty one-slot lanes,
    /// which the caller's compaction drops; or no score when both put jobs on
    /// more than `cap` lanes (the batch decision's lane count, which a
    /// candidate must not exceed). `query(r, w)` is representative job `r`'s
    /// stage widths (see `columns`) on a lane of `w` slots, queried at most
    /// once per pair, and `item_of(j, q)` job `j`'s modeled time under stage
    /// widths `q`.
    ///
    /// 1. The target's lower bound is `LB = max(ceil(sum t / S), max_j
    ///    min_w item(j, w))`. Only jobs above the mean load can raise the
    ///    second term, and fewer than S jobs are above it.
    /// 2. For a target `T >= LB`, each job with `t_j > T` needs the least
    ///    width `p_j(T)` with `item(j, p_j(T)) <= T`; the others take one
    ///    slot. The least `T` among `LB` and the item times at least `LB`
    ///    that [passes](Reservation::passes) is found by bisection (the
    ///    largest job's serial time, if above `LB`, passes with no
    ///    reservation).
    /// 3. Each job above the target gets a lane of its width, the remaining
    ///    slots become one-slot lanes, and every other job, largest first,
    ///    goes to the lane minimising `(load + item, jobs, lane)`: any lane
    ///    in the shared layout, a one-slot lane in the exclusive one, whose
    ///    target also had to fit the tail on those lanes.
    /// 4. Empty one-slot lanes are dropped, and their slots widen the
    ///    reserved lanes, largest job first, while some stage of that job's
    ///    grant still grows. Any slot left stays idle.
    ///
    /// The score is the completed placement's, not the target: a passing
    /// target does not prove that the tail fits under it.
    ///
    /// Why the exclusive layout: the shared one's target counts only the
    /// reserved widths, so a target that reserves every slot forces the
    /// tail onto the reserved lanes, behind the largest jobs; the exclusive
    /// layout is the one whose tail fits beside them.
    fn place(
        &mut self,
        (slots, cap): (usize, usize),
        order: &[usize],
        times: &[Duration],
        columns: &Columns,
        (item_of, query): (
            &impl Fn(usize, Grant) -> Duration,
            &impl Fn(usize, usize) -> Grant,
        ),
    ) -> Option<Duration> {
        let Self {
            table,
            targets,
            tail,
            wide,
            thin,
            shares,
            lane_of,
            other_shares,
            other_lane_of,
            grants,
        } = self;
        let grant = |g: &mut Grants, j: usize, w: usize| g.get(columns, j, w, query);
        let item = |g: &mut Grants, j: usize, w: usize| item_of(j, grant(g, j, w));
        tail.clear();
        tail.resize(order.len() + 1, 0);
        for (i, &j) in order.iter().enumerate().rev() {
            tail[i] = tail[i + 1].saturating_add(times[j].as_nanos());
        }
        let total = tail[0];
        let mean = total.div_ceil(slots as u128);
        let heavy = order
            .iter()
            .take_while(|&&j| times[j].as_nanos() > mean)
            .count();
        grants.begin(slots, &order[..heavy], columns);
        table.clear();
        for &j in &order[..heavy] {
            for width in 1..=slots {
                table.push(item(grants, j, width));
            }
        }
        let rows = &*table;
        let row = |i: usize| &rows[i * slots..(i + 1) * slots];
        let bound = (0..heavy)
            .filter_map(|i| row(i).iter().min())
            .fold(from_nanos(mean), |bound, &least| bound.max(least));
        let large = order[..heavy]
            .iter()
            .take_while(|&&j| times[j] > bound)
            .count();
        targets.clear();
        targets.push(bound);
        for i in 0..large {
            targets.extend(row(i).iter().filter(|&&t| t >= bound));
        }
        targets.sort_unstable();
        sorted();
        targets.dedup();
        let reservation = Reservation {
            slots,
            cap,
            order,
            times,
            table: rows,
            targets,
            tail,
            large,
        };
        let shared = layout(
            &reservation,
            false,
            (wide, thin),
            (shares, lane_of),
            grants,
            &item,
            &grant,
        );
        let exclusive = layout(
            &reservation,
            true,
            (wide, thin),
            (other_shares, other_lane_of),
            grants,
            &item,
            &grant,
        );
        // Ties keep the shared layout.
        match (shared, exclusive) {
            (shared, Some(x)) if shared.is_none_or(|shared| x < shared) => {
                std::mem::swap(shares, other_shares);
                std::mem::swap(lane_of, other_lane_of);
                Some(x)
            }
            (shared, _) => shared,
        }
    }

    fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        self.table
            .capacity()
            .saturating_add(self.targets.capacity())
            .saturating_mul(size_of::<Duration>())
            .saturating_add(self.tail.capacity().saturating_mul(size_of::<u128>()))
            .saturating_add(
                self.wide
                    .capacity()
                    .saturating_mul(size_of::<(Duration, usize)>()),
            )
            .saturating_add(
                self.thin
                    .capacity()
                    .saturating_mul(size_of::<Reverse<(Duration, usize, usize)>>()),
            )
            .saturating_add(self.grants.retained_bytes())
            .saturating_add(words(
                self.shares.capacity()
                    + self.lane_of.capacity()
                    + self.other_shares.capacity()
                    + self.other_lane_of.capacity(),
            ))
    }
}

/// One-slot lanes by `(load, jobs, lane)`, as a heap's storage.
type ThinLanes = Vec<Reverse<(Duration, usize, usize)>>;

/// One reservation layout (steps 2 to 4 of [`Moldable::place`]) into
/// `shares` and `lane_of`, with `wide` and `thin` as scratch; its score, or
/// none when more than the cap of its lanes hold jobs.
#[allow(clippy::too_many_arguments)]
fn layout(
    r: &Reservation<'_>,
    exclusive: bool,
    (wide, thin): (&mut Vec<(Duration, usize)>, &mut ThinLanes),
    (shares, lane_of): (&mut Vec<usize>, &mut Vec<usize>),
    grants: &mut Grants,
    item: &impl Fn(&mut Grants, usize, usize) -> Duration,
    grant: &impl Fn(&mut Grants, usize, usize) -> Grant,
) -> Option<Duration> {
    let (slots, order) = (r.slots, r.order);
    let target = r.targets[r.targets.partition_point(|&t| !r.passes(t, exclusive))];
    let reserved = r.reserved(target);
    let used: usize = (0..reserved).map(|i| r.need(i, target)).sum();

    let count = reserved + (slots - used);
    shares.clear();
    lane_of.clear();
    lane_of.resize(r.times.len(), 0);
    wide.clear();
    for (i, &j) in order.iter().take(reserved).enumerate() {
        let width = r.need(i, target);
        shares.push(width);
        lane_of[j] = i;
        wide.push((r.row(i)[width - 1], 1));
    }
    let mut heap = BinaryHeap::from(std::mem::take(thin));
    heap.clear();
    for k in reserved..count {
        shares.push(1);
        heap.push(Reverse((Duration::ZERO, 0, k)));
    }
    for &j in &order[reserved..] {
        let mut best = heap
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(item(grants, j, 1)), jobs, k));
        if !exclusive {
            for (k, &(load, jobs)) in wide.iter().enumerate() {
                let key = (load.saturating_add(item(grants, j, shares[k])), jobs, k);
                best = Some(best.map_or(key, |best| best.min(key)));
            }
        }
        let Some((load, jobs, k)) = best else {
            unreachable!("a target passes with a lane for every job")
        };
        if k < reserved {
            wide[k] = (load, jobs + 1);
        } else {
            heap.pop();
            heap.push(Reverse((load, jobs + 1, k)));
        }
        lane_of[j] = k;
    }
    let mut idle = heap
        .iter()
        .filter(|Reverse((_, jobs, _))| *jobs == 0)
        .count();
    let score = heap
        .iter()
        .map(|&Reverse((load, _, _))| load)
        .max()
        .unwrap_or(Duration::ZERO);
    *thin = heap.into_vec();
    if count - idle > r.cap {
        return None;
    }
    for (k, &j) in order.iter().take(reserved).enumerate() {
        while idle > 0 && grows(grant(grants, j, shares[k] + 1), grant(grants, j, shares[k])) {
            shares[k] += 1;
            idle -= 1;
        }
        wide[k].0 = Duration::ZERO;
    }
    // Each reserved lane's load at its final width, in one job pass.
    for (j, &k) in lane_of.iter().enumerate() {
        if k < reserved {
            wide[k].0 = wide[k].0.saturating_add(item(grants, j, shares[k]));
        }
    }
    Some(wide.iter().fold(score, |score, &(load, _)| score.max(load)))
}

/// The modeled makespan of the equal-share placement over `count` lanes of
/// `slots`: lane `k` reserves `slots / count + [k < slots % count]`, the jobs
/// in `order` seed one lane each (the largest on the widest), and every later
/// job goes to the lane minimising `(load + item, jobs, lane)`. With `record`,
/// the placement is written into its shares and each job's lane.
///
/// Lanes of one width share each job's item time, so each width class keeps
/// one heap ordered by `(load, jobs, lane)` and a job compares the two tops.
fn equal_shares(
    slots: usize,
    count: usize,
    order: &[usize],
    item: &impl Fn(usize, usize) -> Duration,
    wide_heap: &mut Vec<Reverse<(Duration, usize, usize)>>,
    narrow_heap: &mut Vec<Reverse<(Duration, usize, usize)>>,
    mut record: Option<&mut Placement>,
) -> Duration {
    let (narrow, wide_lanes) = (slots / count, slots % count);
    let width = |k: usize| narrow + usize::from(k < wide_lanes);
    let mut wide = BinaryHeap::from(std::mem::take(wide_heap));
    let mut thin = BinaryHeap::from(std::mem::take(narrow_heap));
    wide.clear();
    thin.clear();
    if let Some(placement) = record.as_deref_mut() {
        placement.shares.clear();
        placement.shares.extend((0..count).map(width));
    }
    for (k, &j) in order.iter().take(count).enumerate() {
        let entry = Reverse((item(j, width(k)), 1, k));
        if k < wide_lanes {
            wide.push(entry);
        } else {
            thin.push(entry);
        }
        if let Some(placement) = record.as_deref_mut() {
            placement.lane_of[j] = k;
        }
    }
    for &j in order.iter().skip(count) {
        let wide_key = wide
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(item(j, narrow + 1)), jobs, k));
        let thin_key = thin
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(item(j, narrow)), jobs, k));
        let (heap, key) = match (wide_key, thin_key) {
            (Some(a), Some(b)) if a <= b => (&mut wide, a),
            (Some(a), None) => (&mut wide, a),
            (_, Some(b)) => (&mut thin, b),
            (None, None) => unreachable!("count >= 1 seeds a lane"),
        };
        heap.pop();
        let (load, jobs, k) = key;
        heap.push(Reverse((load, jobs + 1, k)));
        if let Some(placement) = record.as_deref_mut() {
            placement.lane_of[j] = k;
        }
    }
    let score = wide
        .iter()
        .chain(thin.iter())
        .map(|&Reverse((load, _, _))| load)
        .max()
        .unwrap_or(Duration::ZERO);
    *wide_heap = wide.into_vec();
    *narrow_heap = thin.into_vec();
    score
}

/// The width of a flat phase of `runnable` independent parts on `exec`, with
/// `estimate` the serial time of the whole phase: under `Auto` the budget,
/// at most `runnable`, when `estimate` pays for one section of the whole
/// budget, and one otherwise; under a forced policy its decision for
/// `runnable` items, `lanes * item_threads` slots, at most the budget and at
/// most `runnable` (`Sequential`: one). Model-free: no estimate per part,
/// search or cache.
///
/// Why a forced width is capped by the parts too: the width sizes per-worker
/// scratch, and a worker with no part has none to run, while the parts
/// still run on the caller's whole token.
pub fn flat_width(exec: &Exec<'_>, estimate: Duration, runnable: usize) -> usize {
    let budget = exec.budget();
    let runnable = runnable.max(1);
    match exec.batch_policy().strategy {
        BatchStrategy::Auto => match NonZeroUsize::new(budget) {
            Some(width) if runnable > 1 && exec.token_ref().pays(estimate, 1, width) => {
                budget.min(runnable)
            }
            _ => 1,
        },
        _ => {
            let decision = exec.batch_decision(&BatchQuery::new(runnable, 0.0));
            decision
                .lanes
                .saturating_mul(decision.item_threads)
                .clamp(1, budget.max(1))
                .min(runnable)
        }
    }
}

/// The job count of the left half of a split of `items >= 2` jobs over
/// `threads >= 2` logical threads: the left threads' proportional share,
/// rounded up, keeping at least one job on each side.
pub(super) fn parallel_split(items: usize, threads: usize) -> usize {
    let left_threads = threads / 2;
    items
        .saturating_mul(left_threads)
        .div_ceil(threads)
        .clamp(1, items - 1)
}

/// Drops the empty lanes of `placement.shares`, renumbering the others in
/// order, and builds each lane's ascending job list from
/// `placement.lane_of`, with `count` as scratch.
fn compact(placement: &mut Placement, count: &mut Vec<usize>) {
    let jobs = placement.lane_of.len();
    count.clear();
    count.resize(placement.shares.len(), 0);
    for &l in &placement.lane_of {
        count[l] += 1;
    }
    // `count` becomes each kept lane's new index.
    let mut kept = 0;
    placement.lane_starts.clear();
    for (l, size) in count.iter_mut().enumerate() {
        if *size > 0 {
            placement.shares[kept] = placement.shares[l];
            placement.lane_starts.push(*size);
            *size = kept;
            kept += 1;
        }
    }
    placement.shares.truncate(kept);
    // Lane sizes to starts, then a source-order fill.
    let mut start = 0;
    for size in placement.lane_starts.iter_mut() {
        (*size, start) = (start, start + *size);
    }
    placement.lane_starts.push(start);
    placement.lane_jobs.clear();
    placement.lane_jobs.resize(jobs, 0);
    for j in 0..jobs {
        let l = count[placement.lane_of[j]];
        placement.lane_of[j] = l;
        placement.lane_jobs[placement.lane_starts[l]] = j;
        placement.lane_starts[l] += 1;
    }
    // The fill advanced each start to the next lane's.
    for l in (1..kept).rev() {
        placement.lane_starts[l] = placement.lane_starts[l - 1];
    }
    placement.lane_starts[0] = 0;
}
