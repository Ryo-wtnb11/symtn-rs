use super::*;

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
/// `w` reserved slots and `key(j)` its other inputs (see [`GrantKey`]);
/// Auto scores the candidates with it. Without a grant, and under a forced
/// strategy, the baseline placement is returned and no candidate is scored;
/// `key` is then not called.
pub(crate) fn allocate(
    exec: &Exec<'_>,
    times: &[Duration],
    key: impl Fn(usize) -> GrantKey,
    mean_work: f64,
    grant: Option<impl Fn(usize, Duration, usize) -> usize>,
    tables: &mut LaneTables,
) {
    let LaneTables {
        placement, scratch, ..
    } = tables;
    placement.clear();
    // The keys are staged only where a search can run, so a serial context
    // reads no grant input.
    let mut keys = std::mem::take(&mut scratch.keys);
    keys.clear();
    if grant.is_some() && searchable(exec, times, mean_work) {
        keys.extend((0..times.len()).map(key));
    }
    place(exec, times, &keys, mean_work, grant, placement, scratch);
    scratch.keys = keys;
}

fn place(
    exec: &Exec<'_>,
    times: &[Duration],
    keys: &[GrantKey],
    mean_work: f64,
    grant: Option<impl Fn(usize, Duration, usize) -> usize>,
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
    if let (BatchStrategy::Auto, Some(grant)) = (exec.batch_policy().strategy, grant) {
        let fill = |j: usize, width: usize| item_time(exec, times[j], grant(j, times[j], width));
        let mut columns = std::mem::take(&mut scratch.columns);
        let mut narrow = std::mem::take(&mut scratch.narrow);
        columns.begin(times, keys, |j| grant(j, times[j], slots) > 1);
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
        let provider = |r: usize, width: usize| grant(r, times[r], width);
        let item_of = |serial: Duration, width: usize| item_time(exec, serial, width);
        let score = moldable.place(
            (slots, lanes),
            &order,
            times,
            &columns,
            (&item_of, &provider),
        );
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
    /// its serial time at every candidate width.
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

/// Candidate M, large-job reservation: its target search tables and its
/// lanes (each job's lane and each lane's share), kept for reuse.
#[derive(Debug, Default)]
struct Moldable {
    /// Row `i` holds the item time of the `i`-th largest job at every width
    /// `1..=S`, for the jobs above the mean load only.
    table: Vec<Duration>,
    targets: Vec<Duration>,
    /// Each reserved lane's load and job count.
    wide: Vec<(Duration, usize)>,
    /// The width-one lanes, by `(load, jobs, lane)`.
    thin: Vec<Reverse<(Duration, usize, usize)>>,
    shares: Vec<usize>,
    lane_of: Vec<usize>,
    grants: Grants,
}

/// The grants the reservation queried, each at most once per
/// (representative, width), in dense tables, so their capacities are their
/// whole allocation. A zero cell is not yet queried (no width rule grants
/// zero slots; such a grant would only be queried again).
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
    heavy: Vec<usize>,
    /// Each width's column of `tail`, or `usize::MAX`.
    column: Vec<usize>,
    /// The other widened jobs, column-major over the widths they meet.
    tail: Vec<usize>,
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
        self.heavy.resize(self.heavy_rows * slots, 0);
        self.column.clear();
        self.column.resize(slots + 1, usize::MAX);
        self.tail.clear();
    }

    /// Job `j`'s grant at `width <= S`: one slot for a job the provider keeps
    /// serial at S (see [`Columns::begin`]), otherwise its representative's
    /// grant, queried with `query(representative, width)` on first use.
    fn get(
        &mut self,
        columns: &Columns,
        j: usize,
        width: usize,
        query: &impl Fn(usize, usize) -> usize,
    ) -> usize {
        if !columns.widens[j] {
            return 1;
        }
        let r = columns.rep[j];
        let row = self.row[r];
        let cell = if row < self.heavy_rows {
            &mut self.heavy[row * self.slots + width - 1]
        } else {
            if self.column[width] == usize::MAX {
                self.column[width] = self.tail.len() / self.tail_rows;
                self.tail.resize(self.tail.len() + self.tail_rows, 0);
            }
            &mut self.tail[self.column[width] * self.tail_rows + row - self.heavy_rows]
        };
        if *cell == 0 {
            *cell = query(r, width);
        }
        *cell
    }

    fn retained_bytes(&self) -> usize {
        [&self.row, &self.heavy, &self.column, &self.tail]
            .into_iter()
            .map(Vec::capacity)
            .fold(0, usize::saturating_add)
            .saturating_mul(std::mem::size_of::<usize>())
    }
}

/// `n` nanoseconds as a duration, saturating.
fn from_nanos(n: u128) -> Duration {
    Duration::new(
        u64::try_from(n / 1_000_000_000).unwrap_or(u64::MAX),
        (n % 1_000_000_000) as u32,
    )
}

impl Moldable {
    /// Places the jobs, `order` largest first, on lanes of uneven widths
    /// over `slots`, and returns the lanes with the modeled makespan, or no
    /// score when more than `cap` lanes hold jobs (the batch decision's lane
    /// count, which the candidate must not exceed). `query(r, w)` is the
    /// provider's width for representative job `r` (see `columns`) on a lane
    /// of `w` slots, queried at most once per pair, and `item_of(t, q)` the
    /// modeled time of a job of serial time `t` granted `q` slots. The lanes
    /// are written to `self.shares` and `self.lane_of`, including the empty
    /// one-slot lanes, which the caller's compaction drops.
    ///
    /// 1. The target's lower bound is `LB = max(ceil(sum t / S), max_j
    ///    min_w item(j, w))`. Only jobs above the mean load can raise the
    ///    second term, and fewer than S jobs are above it.
    /// 2. For a target `T >= LB`, each job with `t_j > T` needs the least
    ///    width `p_j(T)` with `item(j, p_j(T)) <= T`; the others take one
    ///    slot. `T` passes when the needed widths sum to at most S, which is
    ///    monotone in `T`, so a binary search over `LB` and the item times
    ///    at least `LB` finds the least passing target (the largest job's
    ///    serial time, if above `LB`, passes with no reservation).
    /// 3. Each job above the target gets a lane of its width, the remaining
    ///    slots become one-slot lanes, and every other job, largest first,
    ///    goes to the lane minimising `(load + item, jobs, lane)`.
    /// 4. Empty one-slot lanes are dropped, and their slots widen the
    ///    reserved lanes, largest job first, while that job's grant still
    ///    grows. Any slot left stays idle.
    ///
    /// The score is the completed placement's, not the target: a passing
    /// target does not prove that the tail fits under it.
    #[allow(clippy::too_many_arguments)]
    fn place(
        &mut self,
        (slots, cap): (usize, usize),
        order: &[usize],
        times: &[Duration],
        columns: &Columns,
        (item_of, query): (
            &impl Fn(Duration, usize) -> Duration,
            &impl Fn(usize, usize) -> usize,
        ),
    ) -> Option<Duration> {
        let Self {
            table,
            targets,
            wide,
            thin,
            shares,
            lane_of,
            grants,
        } = self;
        let grant = |g: &mut Grants, j: usize, w: usize| g.get(columns, j, w, query);
        let item = |g: &mut Grants, j: usize, w: usize| item_of(times[j], grant(g, j, w));
        let total = times
            .iter()
            .fold(0u128, |sum, t| sum.saturating_add(t.as_nanos()));
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
        let table = &*table;
        let row = |i: usize| &table[i * slots..(i + 1) * slots];
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
        // Every target is at least each large job's least item time.
        let need = |i: usize, target: Duration| {
            1 + row(i)
                .iter()
                .position(|&t| t <= target)
                .unwrap_or(slots - 1)
        };
        let reserved = |target: Duration| {
            order[..large]
                .iter()
                .take_while(|&&j| times[j] > target)
                .count()
        };
        let passes = |target: Duration| {
            (0..reserved(target))
                .map(|i| need(i, target))
                .fold(0usize, usize::saturating_add)
                <= slots
        };
        let target = targets[targets.partition_point(|&t| !passes(t))];
        let reserved = reserved(target);
        let used: usize = (0..reserved).map(|i| need(i, target)).sum();

        let count = reserved + (slots - used);
        shares.clear();
        lane_of.clear();
        lane_of.resize(times.len(), 0);
        wide.clear();
        for (i, &j) in order.iter().take(reserved).enumerate() {
            let width = need(i, target);
            shares.push(width);
            lane_of[j] = i;
            wide.push((row(i)[width - 1], 1));
        }
        let mut heap = BinaryHeap::from(std::mem::take(thin));
        heap.clear();
        for k in reserved..count {
            shares.push(1);
            heap.push(Reverse((Duration::ZERO, 0, k)));
        }
        for &j in &order[reserved..] {
            let mut best = heap.peek().map(|&Reverse((load, jobs, k))| {
                (load.saturating_add(item(grants, j, 1)), jobs, k)
            });
            for (k, &(load, jobs)) in wide.iter().enumerate() {
                let key = (load.saturating_add(item(grants, j, shares[k])), jobs, k);
                best = Some(best.map_or(key, |best| best.min(key)));
            }
            let Some((load, jobs, k)) = best else {
                unreachable!("a target passes with at least one lane")
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
        if count - idle > cap {
            return None;
        }
        for (k, &j) in order.iter().take(reserved).enumerate() {
            while idle > 0 && grant(grants, j, shares[k] + 1) > grant(grants, j, shares[k]) {
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

    fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        self.table
            .capacity()
            .saturating_add(self.targets.capacity())
            .saturating_mul(size_of::<Duration>())
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
            .saturating_add(words(self.shares.capacity() + self.lane_of.capacity()))
    }
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
