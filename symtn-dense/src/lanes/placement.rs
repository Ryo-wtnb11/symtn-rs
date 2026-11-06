use super::*;

#[derive(Debug, Default)]
pub(super) struct PlaceScratch {
    /// `D(k)` in ns at index k, for `2 <= k <= B`.
    costs: Vec<u128>,
    /// Every stage's serial estimate in ns, sections, kind and scored width
    /// (one for a stage that does not widen), flat in job order.
    stage_ns: Vec<u128>,
    sections: Vec<u32>,
    kinds: Vec<StageKind>,
    scored: Vec<usize>,
    /// Per job: E_j, its widest scored stage w_j and its reserved allotment.
    job_ns: Vec<u128>,
    job_width: Vec<usize>,
    allot: Vec<usize>,
    /// The jobs by descending estimate, then ascending index.
    order: Vec<usize>,
    /// The width of each distinct widening estimate, and this call's keys.
    ///
    /// Why the map is emptied key by key rather than cleared: `clear` resets
    /// every bucket of the retained table, a cost of the largest earlier call
    /// rather than of this one. Removing this call's keys costs O(G_w), plus
    /// hashbrown's in-place rehash of its deleted markers, at most once per
    /// table capacity of insertions (amortized O(1) per key).
    memo: HashMap<Estimate, usize>,
    memo_keys: Vec<Estimate>,
    /// The memo's full capacity, the charge's basis: the largest capacity it
    /// reported, sampled after every insertion. Why not `memo.capacity()`
    /// itself: a removal may leave a deleted marker, which lowers the
    /// reported capacity (by a hash-dependent amount) without freeing a
    /// bucket; a growth rehashes them away, so the sample right after it is
    /// the new table's full capacity.
    memo_capacity: usize,
    /// Per lane: accumulated time and job count.
    tau: Vec<u128>,
    count: Vec<usize>,
    /// Lanes by `(load, jobs, lane)`, one entry per lane.
    pub(super) heap: BinaryHeap<Reverse<(u128, usize, usize)>>,
    /// Per job, its time on an equal-share candidate's narrow and wide
    /// shares.
    at: Vec<[u128; 2]>,
    /// The index of each distinct lane share (by share, `usize::MAX` if
    /// none), the distinct shares and a queued job's time on each.
    share_slot: Vec<usize>,
    distinct: Vec<usize>,
    distinct_time: Vec<u128>,
    /// Every job's time summary, `(w, sum E, sum of sections)` per distinct
    /// scored stage width, job `j`'s from `bucket_starts[j]`; `bucket_of`
    /// indexes one job's buckets by width while they are built
    /// (`usize::MAX` if none).
    bucket_of: Vec<usize>,
    bucket_starts: Vec<usize>,
    buckets: Vec<(usize, u128, u128)>,
    /// The barrier walk's slowest movement per round.
    round_max: Vec<u128>,
    /// The baseline's and the one-lane candidate's scores, when scored.
    #[cfg(test)]
    pub(super) scores: (Option<u128>, Option<u128>),
}

impl PlaceScratch {
    pub(super) fn retained_bytes(&self) -> usize {
        let wide = std::mem::size_of::<u128>();
        words(
            self.scored.capacity()
                + self.job_width.capacity()
                + self.allot.capacity()
                + self.order.capacity()
                + self.count.capacity()
                + self.share_slot.capacity()
                + self.distinct.capacity()
                + self.bucket_of.capacity()
                + self.bucket_starts.capacity(),
        )
        .saturating_add(wide.saturating_mul(
            self.costs.capacity()
                + self.stage_ns.capacity()
                + self.job_ns.capacity()
                + self.tau.capacity()
                + self.distinct_time.capacity()
                + self.round_max.capacity()
                + 2 * self.at.capacity(),
        ))
        .saturating_add(
            self.kinds
                .capacity()
                .saturating_mul(std::mem::size_of::<StageKind>()),
        )
        .saturating_add(
            self.sections
                .capacity()
                .saturating_mul(std::mem::size_of::<u32>()),
        )
        .saturating_add(memo_bytes(self.memo_capacity))
        .saturating_add(
            self.memo_keys
                .capacity()
                .saturating_mul(std::mem::size_of::<Estimate>()),
        )
        .saturating_add(self.heap.capacity().saturating_mul(std::mem::size_of::<(
            u128,
            usize,
            usize,
        )>()))
        .saturating_add(self.buckets.capacity().saturating_mul(std::mem::size_of::<(
            usize,
            u128,
            u128,
        )>()))
    }

    /// `D(x)` for a section of `x` participants; zero below two.
    fn dispatch(&self, x: usize) -> u128 {
        if x < 2 {
            0
        } else {
            self.costs[x]
        }
    }

    /// `t_k(x)` of stage `k`.
    fn stage_time(&self, k: usize, x: usize) -> u128 {
        let e = self.stage_ns[k];
        if x < 2 {
            e
        } else {
            (e / x as u128)
                .saturating_add(u128::from(self.sections[k]).saturating_mul(self.costs[x]))
        }
    }

    /// Builds every job's time summary from the scored stage widths.
    fn summarize(&mut self, starts: &[usize], budget: usize) {
        self.bucket_of.clear();
        self.bucket_of.resize(budget + 1, usize::MAX);
        self.buckets.clear();
        self.bucket_starts.clear();
        for j in 0..starts.len() - 1 {
            let first = self.buckets.len();
            self.bucket_starts.push(first);
            for k in starts[j]..starts[j + 1] {
                let (w, e, sections) = (self.scored[k], self.stage_ns[k], self.sections[k]);
                let slot = match self.bucket_of[w] {
                    usize::MAX => {
                        self.bucket_of[w] = self.buckets.len();
                        self.buckets.push((w, 0, 0));
                        self.buckets.len() - 1
                    }
                    slot => slot,
                };
                let bucket = &mut self.buckets[slot];
                bucket.1 = bucket.1.saturating_add(e);
                bucket.2 += u128::from(sections);
            }
            for &(w, _, _) in &self.buckets[first..] {
                self.bucket_of[w] = usize::MAX;
            }
        }
        self.bucket_starts.push(self.buckets.len());
    }

    /// Fills `at` with every job's time on `narrow` and `narrow + 1` slots,
    /// an equal-share candidate's two shares.
    fn share_times(&mut self, count: usize, narrow: usize) {
        self.at.clear();
        for j in 0..count {
            let at = [self.job_time(j, narrow), self.job_time(j, narrow + 1)];
            self.at.push(at);
        }
    }

    /// Job `j`'s time on a share of `a` slots.
    fn job_time(&self, j: usize, a: usize) -> u128 {
        self.buckets[self.bucket_starts[j]..self.bucket_starts[j + 1]]
            .iter()
            .fold(0u128, |sum, &(w, e, n)| {
                let x = a.min(w);
                let t = if x < 2 {
                    e
                } else {
                    (e / x as u128).saturating_add(n.saturating_mul(self.costs[x]))
                };
                sum.saturating_add(t)
            })
    }
}

/// Upper bound of the width memo's retained bytes at `capacity` items:
/// hashbrown keeps at most `2 capacity` buckets (load factor 7/8, or one
/// spare below eight buckets), each a slot plus one control byte, and one
/// trailing control group of at most 16 bytes.
fn memo_bytes(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let slot = std::mem::size_of::<(Estimate, usize)>() + 1;
    capacity.saturating_mul(2 * slot).saturating_add(16)
}

pub(super) fn words(n: usize) -> usize {
    n.saturating_mul(std::mem::size_of::<usize>())
}

fn ns(duration: Duration) -> u128 {
    duration.as_nanos()
}

fn duration(ns: u128) -> Duration {
    Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX))
}

/// The lanes one `Auto` candidate builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Candidate {
    /// This many lanes of equal shares, filled largest job first.
    Equal(usize),
    /// Dedicated lanes for wide jobs plus one-slot lanes.
    Reserved,
}

/// Places `jobs` on `exec`'s budget for the runner of `form`, into
/// `tables` (see the module documentation).
pub fn place(exec: &Exec<'_>, jobs: &dyn Jobs, form: Form, tables: &mut LaneTables) {
    let LaneTables {
        placement, scratch, ..
    } = tables;
    placement.clear();
    #[cfg(test)]
    {
        scratch.scores = (None, None);
    }
    let count = jobs.count();
    if count == 0 {
        return;
    }
    if exec.batch_policy().strategy != BatchStrategy::Auto {
        return place_forced(exec, jobs, placement, scratch);
    }
    let budget = exec.budget();
    if budget == 1 {
        return placement.sequential(1, count);
    }
    let token = exec.token_ref();

    // Widths.
    scratch.costs.clear();
    scratch.costs.resize(budget + 1, 0);
    let mut cheapest = Duration::MAX;
    for k in 2..=budget {
        let cost = NonZeroUsize::new(k).map_or(Duration::MAX, |k| token.dispatch_cost(k));
        scratch.costs[k] = ns(cost);
        cheapest = cheapest.min(cost);
    }
    scratch.stage_ns.clear();
    scratch.sections.clear();
    scratch.kinds.clear();
    scratch.scored.clear();
    scratch.job_ns.clear();
    scratch.job_width.clear();
    // Empty after every completed call (`memo_keys`); this clear only costs
    // the table after an unwound one.
    scratch.memo.clear();
    scratch.memo_keys.clear();
    for j in 0..count {
        placement.stage_starts.push(placement.widths.len());
        let (mut e_j, mut w_j) = (0u128, 1usize);
        for k in 0..jobs.stages(j) {
            let estimate = jobs.estimate(j, k);
            let largest = estimate
                .payback
                .map_or(estimate.serial, |p| p.max(estimate.serial));
            let sections = estimate.sections.max(1);
            // A stage that does not widen is scored at one slot and offered
            // its lane's whole share, which the provider's own rule divides.
            let (offered, scored) = if !estimate.widen {
                (budget, 1)
            } else if !threadteam::pays_at(largest, sections, cheapest) {
                // No width pays below `PAYS` sections of the cheapest dispatch.
                (1, 1)
            } else {
                let w = *scratch.memo.entry(estimate).or_insert_with(|| {
                    scratch.memo_keys.push(estimate);
                    let w = token.width_for(estimate.serial, sections);
                    estimate
                        .payback
                        .map_or(w, |p| w.min(token.width_for(p, sections)))
                        .get()
                });
                scratch.memo_capacity = scratch.memo_capacity.max(scratch.memo.capacity());
                (w, w)
            };
            let e = ns(estimate.serial);
            placement.widths.push(offered);
            scratch.scored.push(scored);
            scratch.stage_ns.push(e);
            scratch.sections.push(sections);
            scratch.kinds.push(jobs.kind(j, k));
            e_j = e_j.saturating_add(e);
            w_j = w_j.max(scored);
        }
        scratch.job_ns.push(e_j);
        scratch.job_width.push(w_j);
    }
    placement.stage_starts.push(placement.widths.len());
    for key in scratch.memo_keys.drain(..) {
        scratch.memo.remove(&key);
    }

    // Admission: the whole call's estimate decides the slots it uses.
    let total = scratch
        .job_ns
        .iter()
        .fold(0u128, |sum, &e| sum.saturating_add(e));
    let slots = token.width_for(duration(total), 1).get();
    let baseline = slots.min(count);
    if baseline <= 1 {
        return placement.sequential(slots, count);
    }
    scratch.summarize(&placement.stage_starts, budget);
    scratch.order.clear();
    scratch.order.extend(0..count);
    let job_ns = &scratch.job_ns;
    scratch
        .order
        .sort_unstable_by_key(|&j| (Reverse(job_ns[j]), j));

    // Candidates: the baseline's equal shares, every smaller lane count, then
    // the reservation; a later one replaces the best only by scoring strictly
    // lower, so ties keep the baseline, then the larger lane count.
    let fork_pays = |lanes: usize| {
        lanes < 2 || NonZeroUsize::new(lanes).is_some_and(|l| token.pays(duration(total), 1, l))
    };
    let mut best: Option<(u128, Candidate)> = None;
    let mut built = None;
    for lanes in (1..=baseline).rev() {
        if !fork_pays(lanes) {
            continue;
        }
        scratch.share_times(count, slots / lanes);
        if let (Form::Native, Some((score, _))) = (form, best) {
            if equal_bound(scratch, slots, lanes) >= score {
                continue;
            }
        }
        equal_lanes(placement, scratch, count, slots, lanes);
        let score = lanes_score(form, placement, scratch);
        #[cfg(test)]
        {
            if lanes == baseline {
                scratch.scores.0 = Some(score);
            }
            if lanes == 1 {
                scratch.scores.1 = Some(score);
            }
        }
        built = Some(Candidate::Equal(lanes));
        if best.is_none_or(|(b, _)| score < b) {
            best = Some((score, Candidate::Equal(lanes)));
        }
    }
    if reserved_lanes(placement, scratch, count, slots, baseline) {
        built = Some(Candidate::Reserved);
        let score = lanes_score(form, placement, scratch);
        if fork_pays(placement.lanes()) && best.is_none_or(|(b, _)| score < b) {
            best = Some((score, Candidate::Reserved));
        }
    }
    // One lane always pays, so a candidate was scored.
    let chosen = best.map_or(Candidate::Equal(1), |(_, c)| c);
    if built != Some(chosen) {
        match chosen {
            Candidate::Equal(lanes) => {
                scratch.share_times(count, slots / lanes);
                equal_lanes(placement, scratch, count, slots, lanes);
            }
            Candidate::Reserved => {
                reserved_lanes(placement, scratch, count, slots, baseline);
            }
        }
    }
}

/// The score of the lanes in `placement` under `form`'s runner.
fn lanes_score(form: Form, placement: &Placement, scratch: &mut PlaceScratch) -> u128 {
    match form {
        Form::Native => scratch
            .tau
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(scratch.dispatch(placement.lanes())),
        Form::Barrier => barrier_time(placement, scratch),
    }
}

/// A lower bound of the native score of `lanes` equal shares from `at`:
/// every job takes at least the smaller of its two share times, so the
/// slowest lane takes at least the largest of those and at least their mean.
fn equal_bound(scratch: &PlaceScratch, slots: usize, lanes: usize) -> u128 {
    let uneven = !slots.is_multiple_of(lanes);
    let (largest, sum) = scratch
        .at
        .iter()
        .map(|&[narrow, wide]| if uneven { narrow.min(wide) } else { narrow })
        .fold((0u128, 0u128), |(largest, sum), m| {
            (largest.max(m), sum.saturating_add(m))
        });
    largest
        .max(sum.div_ceil(lanes as u128))
        .saturating_add(scratch.dispatch(lanes))
}

/// `lanes` lanes of equal shares of `slots` (the remainder one slot each to
/// the lowest lanes), filled by the longest-processing-time rule: jobs by
/// descending estimate, then index, each to the lane of least modeled load,
/// then fewest jobs, then lowest index. Job times come from `scratch.at`.
fn equal_lanes(
    placement: &mut Placement,
    scratch: &mut PlaceScratch,
    count: usize,
    slots: usize,
    lanes: usize,
) {
    placement.clear_lanes();
    let (narrow, extra) = (slots / lanes, slots % lanes);
    placement
        .shares
        .extend((0..lanes).map(|l| narrow + usize::from(l < extra)));
    placement.lane_of.resize(count, usize::MAX);
    scratch.tau.clear();
    scratch.tau.resize(lanes, 0);
    scratch.heap.clear();
    scratch.heap.extend((0..lanes).map(|l| Reverse((0, 0, l))));
    for i in 0..count {
        let j = scratch.order[i];
        // `lanes >= 1` seeds the heap, which keeps one entry per lane.
        let Some(Reverse((load, jobs, l))) = scratch.heap.pop() else {
            unreachable!("one heap entry per lane")
        };
        let t = load.saturating_add(scratch.at[j][usize::from(l < extra)]);
        scratch.tau[l] = t;
        scratch.heap.push(Reverse((t, jobs + 1, l)));
        placement.lane_of[j] = l;
    }
    compact(placement, scratch, count);
}

/// The reservation candidate, an intentional heuristic rather than main's
/// moldable search: each job is allotted `a_j = min(w_j, floor(slots E_j /
/// E_tot))`; a job with `a_j >= 2` gets a lane of its own and the rest of
/// the slots become one-slot lanes, at most `cap` lanes in all; every other
/// job goes in source order to the lane on which it would finish first
/// (ties to the lowest lane). False, with nothing built, when the dedicated
/// lanes alone exceed `cap`.
fn reserved_lanes(
    placement: &mut Placement,
    scratch: &mut PlaceScratch,
    count: usize,
    slots: usize,
    cap: usize,
) -> bool {
    let total = scratch
        .job_ns
        .iter()
        .fold(0u128, |sum, &e| sum.saturating_add(e));
    scratch.allot.clear();
    scratch.allot.extend((0..count).map(|j| {
        let proportional = (slots as u128).saturating_mul(scratch.job_ns[j]) / total.max(1);
        scratch.job_width[j].min(usize::try_from(proportional).unwrap_or(usize::MAX))
    }));
    let dedicated = scratch.allot.iter().filter(|&&a| a >= 2).count();
    if dedicated > cap {
        return false;
    }
    placement.clear_lanes();
    placement.lane_of.resize(count, usize::MAX);
    scratch.tau.clear();
    let mut reserved = 0;
    for j in 0..count {
        let a = scratch.allot[j];
        if a >= 2 {
            placement.lane_of[j] = placement.shares.len();
            placement.shares.push(a);
            scratch.tau.push(scratch.job_time(j, a));
            reserved += a;
        }
    }
    for _ in 0..(slots - reserved).min(cap - dedicated) {
        placement.shares.push(1);
        scratch.tau.push(0);
    }
    let lanes = placement.shares.len();
    scratch.share_slot.clear();
    scratch.share_slot.resize(slots + 1, usize::MAX);
    scratch.distinct.clear();
    for &share in &placement.shares {
        if scratch.share_slot[share] == usize::MAX {
            scratch.share_slot[share] = scratch.distinct.len();
            scratch.distinct.push(share);
        }
    }
    // The heap holds one entry per lane. A wide job's scan changes a lane's
    // time without updating it; the next narrow job then rebuilds it from
    // `tau`, O(L) the wide job's lane scan already paid.
    scratch.heap.clear();
    let mut stale = true;
    for j in 0..count {
        if scratch.allot[j] >= 2 {
            continue;
        }
        let lane = if scratch.job_width[j] == 1 {
            // A narrow job takes the same time on every lane.
            if stale {
                scratch.heap.clear();
                scratch
                    .heap
                    .extend((0..lanes).map(|l| Reverse((scratch.tau[l], 0, l))));
                stale = false;
            }
            let lane = least_lane(&mut scratch.heap);
            scratch.tau[lane] = scratch.tau[lane].saturating_add(scratch.job_ns[j]);
            scratch.heap.push(Reverse((scratch.tau[lane], 0, lane)));
            lane
        } else {
            scratch.distinct_time.clear();
            for d in 0..scratch.distinct.len() {
                let t = scratch.job_time(j, scratch.distinct[d]);
                scratch.distinct_time.push(t);
            }
            let finish = |l: usize| {
                let slot = scratch.share_slot[placement.shares[l]];
                scratch.tau[l].saturating_add(scratch.distinct_time[slot])
            };
            let lane =
                (1..lanes).fold(0, |best, l| if finish(l) < finish(best) { l } else { best });
            let t = finish(lane);
            scratch.tau[lane] = t;
            stale = true;
            lane
        };
        placement.lane_of[j] = lane;
    }
    compact(placement, scratch, count);
    true
}

/// A forced policy's placement: its batch decision, authoritative.
fn place_forced(
    exec: &Exec<'_>,
    jobs: &dyn Jobs,
    placement: &mut Placement,
    scratch: &mut PlaceScratch,
) {
    let count = jobs.count();
    let decision = exec.batch_decision(&BatchQuery::new(count, 0.0));
    let share = decision.item_threads.max(1);
    if decision.lanes <= 1 {
        return placement.sequential(share, count);
    }
    placement.shares.resize(decision.lanes, share);
    placement.lane_of.resize(count, usize::MAX);
    scratch.tau.clear();
    scratch.tau.resize(decision.lanes, 0);
    scratch.heap.clear();
    scratch
        .heap
        .extend((0..decision.lanes).map(|l| Reverse((0, 0, l))));
    for j in 0..count {
        let e_j = (0..jobs.stages(j)).fold(0u128, |sum, k| {
            sum.saturating_add(ns(jobs.estimate(j, k).serial))
        });
        let lane = least_lane(&mut scratch.heap);
        scratch.tau[lane] = scratch.tau[lane].saturating_add(e_j);
        placement.lane_of[j] = lane;
        scratch.heap.push(Reverse((scratch.tau[lane], 0, lane)));
    }
    compact(placement, scratch, count);
}

/// Places each of `count` jobs alone on a lane of one slot, reading no
/// estimate: the fan-out of a caller whose partition into jobs is already
/// its placement. Every stage then runs at one slot.
pub fn place_each(count: usize, tables: &mut LaneTables) {
    let placement = &mut tables.placement;
    placement.clear();
    placement.shares.resize(count, 1);
    placement.lane_starts.extend(0..=count);
    placement.lane_jobs.extend(0..count);
    placement.lane_of.extend(0..count);
}

/// The lane of least accumulated time, ties to the lowest, popped from
/// `heap`, which holds one current entry per lane (with a zero job count);
/// the caller pushes the lane back with its new time.
fn least_lane(heap: &mut BinaryHeap<Reverse<(u128, usize, usize)>>) -> usize {
    // A placement has at least one lane, so the heap is never empty here.
    heap.pop().map_or(0, |Reverse((_, _, l))| l)
}

/// Drops the empty lanes of `placement.shares` (with their `tau`) and builds
/// each remaining lane's ascending job list from `placement.lane_of`.
fn compact(placement: &mut Placement, scratch: &mut PlaceScratch, count: usize) {
    let lanes = placement.shares.len();
    scratch.count.clear();
    scratch.count.resize(lanes, 0);
    for &l in &placement.lane_of {
        scratch.count[l] += 1;
    }
    // `count` becomes each kept lane's new index.
    let mut kept = 0;
    for l in 0..lanes {
        if scratch.count[l] > 0 {
            placement.shares[kept] = placement.shares[l];
            scratch.tau[kept] = scratch.tau[l];
            placement.lane_starts.push(scratch.count[l]);
            scratch.count[l] = kept;
            kept += 1;
        }
    }
    placement.shares.truncate(kept);
    scratch.tau.truncate(kept);
    // Lane sizes to starts, then a source-order fill.
    let mut start = 0;
    for size in placement.lane_starts.iter_mut() {
        (*size, start) = (start, start + *size);
    }
    placement.lane_starts.push(start);
    placement.lane_jobs.clear();
    placement.lane_jobs.resize(count, 0);
    for j in 0..count {
        let l = scratch.count[placement.lane_of[j]];
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

/// The barrier form's lanes score: per round the slowest lane's movement,
/// every provider stage serialized at its whole-engine width, and one outer
/// fork per round the runner enters (none for one lane).
fn barrier_time(placement: &Placement, scratch: &mut PlaceScratch) -> u128 {
    scratch.round_max.clear();
    let (mut serialized, mut rounds) = (0u128, 0usize);
    for l in 0..placement.lanes() {
        let share = placement.shares[l];
        // `trailing`: a movement stage follows the lane's last provider stage
        // (or the lane has none), so the runner enters one more round for it.
        let (mut round, mut movement, mut trailing) = (0usize, 0u128, false);
        for &j in placement.jobs(l) {
            for k in placement.stage_starts[j]..placement.stage_starts[j + 1] {
                let w = scratch.scored[k];
                match scratch.kinds[k] {
                    StageKind::Movement => {
                        movement = movement.saturating_add(scratch.stage_time(k, share.min(w)));
                        trailing = true;
                    }
                    StageKind::Provider => {
                        raise(&mut scratch.round_max, round, movement);
                        serialized = serialized.saturating_add(scratch.stage_time(k, w));
                        (round, movement, trailing) = (round + 1, 0, false);
                    }
                }
            }
        }
        raise(&mut scratch.round_max, round, movement);
        rounds = rounds.max(round + usize::from(trailing));
    }
    let forks = (rounds as u128).saturating_mul(scratch.dispatch(placement.lanes()));
    scratch
        .round_max
        .iter()
        .fold(serialized, |sum, &t| sum.saturating_add(t))
        .saturating_add(forks)
}

fn raise(round_max: &mut Vec<u128>, round: usize, time: u128) {
    if round_max.len() <= round {
        round_max.resize(round + 1, 0);
    }
    round_max[round] = round_max[round].max(time);
}
