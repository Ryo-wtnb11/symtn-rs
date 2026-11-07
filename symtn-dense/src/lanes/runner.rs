use super::*;

/// The stage bodies of a placed batch.
pub trait Work: Jobs + Sync {
    /// One job's exclusive state.
    type Job: Send;
    /// One lane's state (its initialized source and destination buffers),
    /// kept for the whole call and lent to every stage the lane runs.
    type Lane: Send;
    type Error: Send;

    /// Runs stage `stage` of `job` on `exec`, whose budget is the stage's
    /// selected width (`Placement::selected`) under [`Placed`] and the job's
    /// own token under the other assignments: a movement stage
    /// dispatches at most that many participants and checks, under `Auto`,
    /// that its estimate pays for its final fork on `exec`'s token (running
    /// inline otherwise), and a provider stage hands `exec` to the provider
    /// as its whole offer. The barrier form never calls it for a provider
    /// stage.
    fn run(
        &self,
        job: usize,
        stage: usize,
        state: &mut Self::Job,
        lane: &mut Self::Lane,
        exec: Exec<'_>,
    ) -> Result<(), Self::Error>;
}

/// The engine side of the barrier form: the caller owns the engine and
/// enters it once per round, and submits each pending provider stage on the
/// engine's whole offer between rounds.
pub trait Barrier<W: Work + ?Sized> {
    /// A refused engine entry.
    type Refused;

    /// Enters the engine and calls `round` once with its whole context,
    /// returning `round`'s refusal of the lane split as the caller's own.
    fn region(
        &mut self,
        round: &mut dyn FnMut(Exec<'_>) -> Result<(), threadteam::Error>,
    ) -> Result<(), Self::Refused>;

    /// Submits provider stage `stage` of `job` over `lane`'s buffers, on the
    /// whole engine.
    fn submit(
        &mut self,
        job: usize,
        stage: usize,
        state: &mut W::Job,
        lane: &mut W::Lane,
    ) -> Result<(), W::Error>;
}

/// The job references and per-lane parts of earlier barrier runs, kept empty
/// between calls so a warm run reuses their capacity.
#[derive(Debug, Default)]
pub(super) struct RunScratch {
    refs: Vec<Vec<ParkedRef>>,
    parts: Vec<Mutex<Part<'static, 'static, (), ()>>>,
}

type ParkedRef = (usize, &'static mut ());

impl RunScratch {
    pub(super) fn retained_bytes(&self) -> usize {
        let inner = self.refs.iter().fold(0usize, |sum, refs| {
            sum.saturating_add(
                refs.capacity()
                    .saturating_mul(std::mem::size_of::<ParkedRef>()),
            )
        });
        inner
            .saturating_add(
                self.refs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Vec<ParkedRef>>()),
            )
            .saturating_add(
                self.parts
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Mutex<Part<'static, 'static, (), ()>>>()),
            )
    }

    /// Lends each lane its jobs' states, ascending, its lane state and a
    /// cursor at its first stage.
    fn lend<'a, 'b, S, L>(
        &mut self,
        placement: &Placement,
        states: &'b mut [S],
        lanes: &'a mut [L],
    ) -> Vec<Mutex<Part<'a, 'b, S, L>>> {
        let count = placement.lanes();
        // `run_barrier` checked the supply before lending.
        debug_assert_eq!(states.len(), placement.lane_of.len());
        debug_assert!(lanes.len() >= count);
        let mut refs: Vec<Vec<(usize, &'b mut S)>> = std::mem::take(&mut self.refs)
            .into_iter()
            .map(recycle)
            .collect();
        if refs.len() < count {
            refs.resize_with(count, Vec::new);
        }
        // Each lane's vector reserved exactly, so a cold call grows each once.
        for (l, lane) in refs.iter_mut().take(count).enumerate() {
            lane.reserve(placement.jobs(l).len());
        }
        for (j, state) in states.iter_mut().enumerate() {
            refs[placement.lane_of[j]].push((j, state));
        }
        let mut parts: Vec<Mutex<Part<'a, 'b, S, L>>> = recycle(std::mem::take(&mut self.parts));
        parts.extend(
            refs.drain(..count)
                .zip(lanes.iter_mut())
                .map(|(jobs, lane)| {
                    Mutex::new(Part {
                        jobs,
                        lane,
                        pos: 0,
                        stage: 0,
                    })
                }),
        );
        self.refs = refs.into_iter().map(recycle).collect();
        parts
    }

    /// Returns each lane's reference vector to its lane position, ahead of
    /// the unused tail, so the next call's lane `l` reuses lane `l`'s
    /// capacity.
    fn park<S, L>(&mut self, mut parts: Vec<Mutex<Part<'_, '_, S, L>>>) {
        self.refs.splice(
            0..0,
            parts.drain(..).map(|part| {
                recycle(
                    part.into_inner()
                        .unwrap_or_else(PoisonError::into_inner)
                        .jobs,
                )
            }),
        );
        self.parts = recycle(parts);
    }
}

/// `vec`'s allocation, emptied, for elements of another lifetime or type of
/// the same layout.
///
/// Why not a transmute: this crate forbids unsafe code. std collects a mapped
/// `vec::IntoIter` in place when the layouts match; that is an optimization,
/// not a documented guarantee, so the warm allocation tests pin it (a
/// regression would reallocate, never misbehave).
pub(crate) fn recycle<A, B>(mut vec: Vec<A>) -> Vec<B> {
    vec.clear();
    vec.into_iter().map(|_| unreachable!("emptied")).collect()
}

/// One lane's jobs, its state and its cursor: the next stage `stage` of the
/// job at `pos`.
#[derive(Debug)]
struct Part<'a, 'b, S, L> {
    jobs: Vec<(usize, &'b mut S)>,
    lane: &'a mut L,
    pos: usize,
    stage: usize,
}

impl<S, L> Part<'_, '_, S, L> {
    /// Moves the cursor past finished jobs; the job index and stage it then
    /// points at, or `None` when the lane is done.
    fn next(&mut self, jobs: &dyn Jobs) -> Option<(usize, usize)> {
        while let Some(&(j, _)) = self.jobs.get(self.pos) {
            if self.stage < jobs.stages(j) {
                return Some((j, self.stage));
            }
            (self.pos, self.stage) = (self.pos + 1, 0);
        }
        None
    }

    /// Records the current job's outcome: the next stage, or its failure,
    /// which ends that job only.
    fn step<E>(&mut self, outcome: Result<(), E>, failure: &Mutex<Option<(usize, E)>>) {
        match outcome {
            Ok(()) => self.stage += 1,
            Err(error) => {
                let job = self.jobs[self.pos].0;
                let mut first = failure.lock().unwrap_or_else(PoisonError::into_inner);
                if first.as_ref().is_none_or(|&(j, _)| job < j) {
                    *first = Some((job, error));
                }
                (self.pos, self.stage) = (self.pos + 1, 0);
            }
        }
    }

    /// Runs the lane's stages on `exec` until it is done or at a provider
    /// stage.
    fn advance<W: Work<Job = S, Lane = L>>(
        &mut self,
        work: &W,
        placement: &Placement,
        share: usize,
        exec: &mut Exec<'_>,
        failure: &Mutex<Option<(usize, W::Error)>>,
    ) {
        while let Some((j, k)) = self.next(work) {
            if work.kind(j, k) == StageKind::Provider {
                return;
            }
            let width =
                NonZeroUsize::new(placement.selected(j, k, share)).unwrap_or(NonZeroUsize::MIN);
            let state = &mut *self.jobs[self.pos].1;
            let outcome = work.run(j, k, state, self.lane, exec.reborrow().limit(width));
            self.step(outcome, failure);
        }
    }
}

/// A run's job outcome: every job succeeded, or the lowest-indexed failing
/// job and its error.
pub type Outcome<E> = Result<(), (usize, E)>;

/// How [`run_native`] lends a batch's jobs to the slots of its context:
/// [`Split`], [`Each`] or [`Placed`]. Each assignment states what it needs of
/// the work's state.
pub trait Assignment<W: Work>: sealed::Assign<W> {}

impl<W: Work, A: sealed::Assign<W>> Assignment<W> for A {}

mod sealed {
    use super::*;

    pub trait Assign<W: Work> {
        fn run(
            self,
            exec: Exec<'_>,
            work: &W,
            states: &mut [W::Job],
            lanes: &mut [W::Lane],
        ) -> Result<Outcome<W::Error>, Refusal<threadteam::Error>>;
    }
}

/// The jobs split in two, recursively, over `threads` logical threads: the
/// left half takes `parallel_split(J, threads)` jobs and `threads / 2`
/// threads, the right half the rest, and the two halves run as the children
/// of a two-child fork of the token, which halves it. A part of one thread
/// or one job is a leaf, which runs its jobs in order on its own token,
/// every stage on that token.
///
/// A part of `n` threads borrows `n` entries of the caller's lanes when they
/// cover it; otherwise the left half keeps the part's lanes and the right
/// half makes its own with `lane`, as does a call given no lanes.
///
/// A leaf stops at its first failing job; its sibling subtree still runs,
/// and the left subtree's failure is returned first, so the returned job is
/// the lowest that failed while every job before it in its leaf succeeded.
///
/// Why the job, lane and error states must be `Sync`: the two halves and
/// their failures pass through `RwLock` slots, which do not allocate; the
/// `Mutex` that would pass `Send`-only state boxes an OS lock on its first
/// lock on some targets (Apple's pthread mutex), an allocation at every
/// split.
#[derive(Clone, Copy, Debug)]
pub struct Split<F> {
    pub threads: usize,
    pub lane: F,
}

/// Each job alone on one slot, contiguous: the context must hold one slot
/// per job, and job `j` borrows lane state `j`. Every job runs, and the
/// lowest failing one is returned.
///
/// Why one flat fork over `Mutex` slots rather than [`Split`]'s recursion:
/// it lends `Send`-only job, lane and error state, which an allocation-free
/// `RwLock` cannot share. The call allocates its slot vector, and on some
/// targets (Apple's pthread mutex) each slot's first lock.
#[derive(Clone, Copy, Debug)]
pub struct Each;

/// The lanes of `tables`' placement, each one `fork_sized` child of exactly
/// its share; lane `l` borrows lane state `l`, and every stage runs on its
/// selected width (`Placement::selected`) of its lane's share. Every job
/// runs to its end or its own first failing stage, and the lowest failing
/// job is returned.
#[derive(Debug)]
pub struct Placed<'a> {
    pub tables: &'a mut LaneTables,
}

/// Runs every job of `assignment` on `exec` with exclusive access to its own
/// `states` entry, and returns the lowest-indexed job's failure, as each
/// [`Assignment`] describes.
///
/// A panic unwinds after the forks join. The outer error, before any stage
/// ran, is a [`Refusal`]: `states` not holding one entry per job (the
/// placement's under [`Placed`], `work`'s count otherwise), `lanes` holding
/// fewer entries than the placement has lanes ([`Placed`]) or the work has
/// jobs ([`Each`]), or threadteam's refusal of the lane shares ([`Placed`])
/// or of one slot per job ([`Each`]).
pub fn run_native<W: Work, A: Assignment<W>>(
    exec: Exec<'_>,
    work: &W,
    assignment: A,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, Refusal<threadteam::Error>> {
    sealed::Assign::run(assignment, exec, work, states, lanes)
}

/// Why a run did not start, or, for [`run_barrier`], why its host stopped it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal<R> {
    /// The context's refusal: threadteam's, of lane shares or slots, before
    /// any stage; or the barrier host's refused entry.
    Context(R),
    /// `states` holds `given` entries for `jobs` jobs.
    States { jobs: usize, given: usize },
    /// `lanes` holds `given` entries where `needed` are lent.
    Lanes { needed: usize, given: usize },
}

impl<R: std::fmt::Display> std::fmt::Display for Refusal<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Context(refused) => refused.fmt(f),
            Self::States { jobs, given } => {
                write!(f, "{given} job states were given for {jobs} jobs")
            }
            Self::Lanes { needed, given } => {
                write!(f, "{given} lane states were given where {needed} are lent")
            }
        }
    }
}

impl<R: std::error::Error + 'static> std::error::Error for Refusal<R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Context(refused) => Some(refused),
            _ => None,
        }
    }
}

/// One state per job and at least `needed` lane states, or the refusal.
fn supply<R>(jobs: usize, states: usize, needed: usize, lanes: usize) -> Result<(), Refusal<R>> {
    if states != jobs {
        return Err(Refusal::States {
            jobs,
            given: states,
        });
    }
    if lanes < needed {
        return Err(Refusal::Lanes {
            needed,
            given: lanes,
        });
    }
    Ok(())
}

impl<W: Work> sealed::Assign<W> for Placed<'_> {
    fn run(
        self,
        exec: Exec<'_>,
        work: &W,
        states: &mut [W::Job],
        lanes: &mut [W::Lane],
    ) -> Result<Outcome<W::Error>, Refusal<threadteam::Error>> {
        let placement = &self.tables.placement;
        supply(
            placement.lane_of.len(),
            states.len(),
            placement.lanes(),
            lanes.len(),
        )?;
        run_placed(exec, work, self.tables, states, lanes).map_err(Refusal::Context)
    }
}

impl<W: Work> sealed::Assign<W> for Each {
    fn run(
        self,
        exec: Exec<'_>,
        work: &W,
        states: &mut [W::Job],
        lanes: &mut [W::Lane],
    ) -> Result<Outcome<W::Error>, Refusal<threadteam::Error>> {
        let jobs = work.count();
        supply(jobs, states.len(), jobs, lanes.len())?;
        let budget = exec.budget();
        let Some(width) = NonZeroUsize::new(jobs) else {
            return Ok(Ok(()));
        };
        if jobs > budget {
            return Err(Refusal::Context(threadteam::Error::SharesExceedBudget {
                budget,
                sum: jobs,
            }));
        }
        let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
        // Each slot: the job's state and lane, then its failure.
        let slots: Vec<_> = states
            .iter_mut()
            .zip(lanes.iter_mut())
            .map(|(state, lane)| Mutex::new((Some((state, lane)), None)))
            .collect();
        // A token of exactly `jobs` slots gives each of its `jobs` children
        // one.
        exec.limit(width).into_token().fork(jobs, |job, token| {
            let mut slot = slots[job].lock().unwrap_or_else(PoisonError::into_inner);
            if let Some((state, lane)) = slot.0.take() {
                let mut exec = Exec::from_token(token, workspace, policy);
                slot.1 = run_job(work, job, state, lane, &mut exec);
            }
        });
        let failure = slots
            .into_iter()
            .find_map(|slot| slot.into_inner().unwrap_or_else(PoisonError::into_inner).1);
        Ok(failure.map_or(Ok(()), Err))
    }
}

impl<W: Work, F> sealed::Assign<W> for Split<F>
where
    W::Job: Sync,
    W::Lane: Sync,
    W::Error: Sync,
    F: Fn() -> W::Lane + Sync,
{
    fn run(
        self,
        exec: Exec<'_>,
        work: &W,
        states: &mut [W::Job],
        lanes: &mut [W::Lane],
    ) -> Result<Outcome<W::Error>, Refusal<threadteam::Error>> {
        supply(work.count(), states.len(), 0, lanes.len())?;
        if states.is_empty() {
            return Ok(Ok(()));
        }
        let mut own = None;
        let lanes = match lanes {
            [] => std::slice::from_mut(own.insert((self.lane)())),
            lanes => lanes,
        };
        let failure = split(work, &self.lane, self.threads, 0, states, lanes, exec);
        Ok(failure.map_or(Ok(()), Err))
    }
}

/// Runs every stage of `job` in order on `exec`, stopping at its first
/// failure.
fn run_job<W: Work>(
    work: &W,
    job: usize,
    state: &mut W::Job,
    lane: &mut W::Lane,
    exec: &mut Exec<'_>,
) -> Option<(usize, W::Error)> {
    (0..work.stages(job)).find_map(|stage| {
        work.run(job, stage, state, lane, exec.reborrow())
            .err()
            .map(|error| (job, error))
    })
}

/// [`Split`] over `states`, whose first job is `first`, with at least one
/// lane state.
fn split<W, F>(
    work: &W,
    fresh: &F,
    threads: usize,
    first: usize,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
    mut exec: Exec<'_>,
) -> Option<(usize, W::Error)>
where
    W: Work,
    W::Job: Sync,
    W::Lane: Sync,
    W::Error: Sync,
    F: Fn() -> W::Lane + Sync,
{
    if threads <= 1 || states.len() <= 1 {
        let lane = &mut lanes[0];
        return states
            .iter_mut()
            .enumerate()
            .find_map(|(i, state)| run_job(work, first + i, state, lane, &mut exec));
    }
    let middle = parallel_split(states.len(), threads);
    let (left_states, right_states) = states.split_at_mut(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_lanes, mut right_lanes) = if lanes.len() >= threads {
        let (left, right) = lanes.split_at_mut(left_threads);
        (left, Lanes::Shared(right))
    } else {
        (lanes, Lanes::Own(fresh()))
    };
    let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
    join(
        exec.into_token(),
        |token| {
            let exec = Exec::from_token(token, workspace, policy);
            split(
                work,
                fresh,
                left_threads,
                first,
                left_states,
                left_lanes,
                exec,
            )
        },
        |token| {
            let exec = Exec::from_token(token, workspace, policy);
            let lanes = right_lanes.get();
            let right = first + middle;
            split(work, fresh, right_threads, right, right_states, lanes, exec)
        },
    )
}

/// The lane states of a split's right half: the back of the caller's when
/// they cover every thread, its own otherwise.
enum Lanes<'a, L> {
    Shared(&'a mut [L]),
    Own(L),
}

impl<L> Lanes<'_, L> {
    fn get(&mut self) -> &mut [L] {
        match self {
            Self::Shared(lanes) => lanes,
            Self::Own(lane) => std::slice::from_mut(lane),
        }
    }
}

/// Runs `left` and `right` as the two children of one fork of `token`, each
/// on its half of the token, and returns the left failure, else the right.
///
/// Why `RwLock` slots, for the closures and their failures alike: `fork`
/// shares one `Fn` with both children, and each side is a `FnOnce` that owns
/// its half of the split buffers; a std `Mutex` boxes an OS lock on its
/// first lock on some targets, while `RwLock` does not allocate.
fn join<A, B, E>(token: threadteam::Exec<'_>, left: A, right: B) -> Option<(usize, E)>
where
    A: FnOnce(threadteam::Exec<'_>) -> Option<(usize, E)> + Send + Sync,
    B: FnOnce(threadteam::Exec<'_>) -> Option<(usize, E)> + Send + Sync,
    E: Send + Sync,
{
    let (left, right) = (RwLock::new(Some(left)), RwLock::new(Some(right)));
    let failures = [RwLock::new(None), RwLock::new(None)];
    token.fork(2, |side, child| {
        let failure = if side == 0 {
            take(&left).and_then(|run| run(child))
        } else {
            take(&right).and_then(|run| run(child))
        };
        *failures[side]
            .write()
            .unwrap_or_else(PoisonError::into_inner) = failure;
    });
    let [left, right] = failures.map(|f| f.into_inner().unwrap_or_else(PoisonError::into_inner));
    left.or(right)
}

/// The value of `slot`, emptied; `None` only if it was already taken.
fn take<T>(slot: &RwLock<Option<T>>) -> Option<T> {
    slot.write().unwrap_or_else(PoisonError::into_inner).take()
}

/// One lane of [`Placed`]: its range of the flat job arena, the
/// count filled so far, and its lane state.
#[derive(Debug)]
struct Lent<'r, 'a, 'b, S, L> {
    jobs: &'r mut [Option<(usize, &'b mut S)>],
    filled: usize,
    lane: &'a mut L,
}

/// The flat job arena and the per-lane parts of [`Placed`], kept empty
/// between calls so a warm run reuses their capacity.
#[derive(Debug, Default)]
pub(super) struct Arena {
    slots: Vec<Option<ParkedRef>>,
    parts: Vec<Mutex<Lent<'static, 'static, 'static, (), ()>>>,
}

impl Arena {
    pub(super) fn retained_bytes(&self) -> usize {
        self.slots
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<ParkedRef>>())
            .saturating_add(self.parts.capacity().saturating_mul(std::mem::size_of::<
                Mutex<Lent<'static, 'static, 'static, (), ()>>,
            >()))
    }
}

/// [`Placed`], on a supply already checked: zero and one lane run without
/// the arena; more lanes fill one
/// flat arena of J slots in lane order and fork once.
///
/// Why each lane part sits behind a `Mutex`: it lends `Send`-only job and
/// lane state, which an allocation-free `RwLock` cannot share. A fresh
/// `Mutex` allocates on its first lock on some targets (Apple's pthread
/// mutex), so a multi-lane call pays one such allocation per lane, as main's
/// runner does, and a failing call one more for its shared failure.
fn run_placed<W: Work>(
    exec: Exec<'_>,
    work: &W,
    tables: &mut LaneTables,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, threadteam::Error> {
    let LaneTables {
        placement, arena, ..
    } = tables;
    let count = placement.lanes();
    if count == 0 {
        return Ok(Ok(()));
    }
    if count == 1 {
        let share = placement.shares[0];
        // `limit` clamps; refuse an oversized share as `fork_sized` would.
        if share > exec.budget() {
            return Err(threadteam::Error::SharesExceedBudget {
                budget: exec.budget(),
                sum: share,
            });
        }
        let width = NonZeroUsize::new(share).ok_or(threadteam::Error::ZeroBudget)?;
        let mut exec = exec.limit(width);
        let jobs = states.iter_mut().enumerate();
        let failure = run_lane(work, placement, share, &mut exec, jobs, &mut lanes[0]);
        return Ok(failure.map_or(Ok(()), Err));
    }
    let mut slots: Vec<Option<(usize, &mut W::Job)>> = recycle(std::mem::take(&mut arena.slots));
    slots.resize_with(states.len(), || None);
    let mut parts: Vec<Mutex<Lent<'_, '_, '_, W::Job, W::Lane>>> =
        recycle(std::mem::take(&mut arena.parts));
    parts.reserve_exact(count);
    let mut rest = &mut slots[..];
    for (l, lane) in lanes.iter_mut().take(count).enumerate() {
        let (jobs, tail) = std::mem::take(&mut rest).split_at_mut(placement.jobs(l).len());
        parts.push(Mutex::new(Lent {
            jobs,
            filled: 0,
            lane,
        }));
        rest = tail;
    }
    #[cfg(test)]
    crate::probe::add(crate::probe::Event::RunnerLanes, parts.len());
    for (j, state) in states.iter_mut().enumerate() {
        let part = parts[placement.lane_of[j]]
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        part.jobs[part.filled] = Some((j, state));
        part.filled += 1;
    }
    let failure = Mutex::new(None);
    let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
    let forked = exec.into_token().fork_sized(&placement.shares, |l, token| {
        let mut part = parts[l].lock().unwrap_or_else(PoisonError::into_inner);
        let Lent { jobs, lane, .. } = &mut *part;
        let mut exec = Exec::from_token(token, workspace, policy);
        let jobs = jobs.iter_mut().filter_map(Option::take);
        let share = placement.shares[l];
        if let Some((job, error)) = run_lane(work, placement, share, &mut exec, jobs, lane) {
            let mut first = failure.lock().unwrap_or_else(PoisonError::into_inner);
            if first.as_ref().is_none_or(|&(j, _)| job < j) {
                *first = Some((job, error));
            }
        }
    });
    arena.parts = recycle(parts);
    arena.slots = recycle(slots);
    forked?;
    let failure = failure.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(failure.map_or(Ok(()), Err))
}

/// Runs one lane's `jobs`, ascending, each stage on its selected width of
/// `share`; a job stops at its own first failing stage and the lane goes on.
/// Returns the lane's first, so lowest, failure.
fn run_lane<'s, W: Work>(
    work: &W,
    placement: &Placement,
    share: usize,
    exec: &mut Exec<'_>,
    jobs: impl Iterator<Item = (usize, &'s mut W::Job)>,
    lane: &mut W::Lane,
) -> Option<(usize, W::Error)>
where
    W::Job: 's,
{
    let mut first = None;
    for (j, state) in jobs {
        for k in 0..work.stages(j) {
            let width =
                NonZeroUsize::new(placement.selected(j, k, share)).unwrap_or(NonZeroUsize::MIN);
            if let Err(error) = work.run(j, k, state, lane, exec.reborrow().limit(width)) {
                first.get_or_insert((j, error));
                break;
            }
        }
    }
    first
}

/// [`run_native`] in rounds, for an executor whose provider stages need the
/// engine itself: each round is one `host.region`, in which every lane runs
/// its stages up to its next provider stage; then, in lane order, each
/// pending provider stage is submitted alone through `host.submit` on the
/// whole engine over its lane's own state. The call takes, maximized over
/// lanes, each lane's provider stages plus one when movement follows the
/// last of them; no round is entered once every lane is done.
///
/// Every lane keeps its share and its `lanes` entry in every round. Failures
/// are job-local exactly as in [`Placed`], so the reported job does not
/// depend on the lane count; a failed submission is never retried. The outer
/// error is `host`'s refused entry, which stops the call
/// ([`Refusal::Context`]), or, before any stage, a supply [`Refusal`] as
/// [`run_native`]'s over [`Placed`].
pub fn run_barrier<W: Work, H: Barrier<W>>(
    host: &mut H,
    work: &W,
    tables: &mut LaneTables,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, Refusal<H::Refused>> {
    let LaneTables { placement, run, .. } = tables;
    supply(
        placement.lane_of.len(),
        states.len(),
        placement.lanes(),
        lanes.len(),
    )?;
    if placement.lanes() == 0 {
        return Ok(Ok(()));
    }
    let failure = Mutex::new(None);
    let mut parts = run.lend(placement, states, lanes);
    let outcome = loop {
        let pending = parts.iter_mut().any(|part| {
            part.get_mut()
                .unwrap_or_else(PoisonError::into_inner)
                .next(work)
                .is_some()
        });
        if !pending {
            break Ok(());
        }
        let entered = host.region(&mut |exec| round(exec, work, placement, &mut parts, &failure));
        if let Err(refused) = entered {
            break Err(refused);
        }
        for part in parts.iter_mut() {
            let part = part.get_mut().unwrap_or_else(PoisonError::into_inner);
            if let Some((j, k)) = part.next(work) {
                let state = &mut *part.jobs[part.pos].1;
                let submitted = host.submit(j, k, state, part.lane);
                part.step(submitted, &failure);
            }
        }
    };
    run.park(parts);
    outcome.map_err(Refusal::Context)?;
    let failure = failure.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(failure.map_or(Ok(()), Err))
}

/// Runs every lane's stages on `exec` up to its next provider stage: one
/// `fork_sized` child of exactly its share per lane.
///
/// Why one lane runs without the fork and its lock: the fork would run it
/// inline anyway, and a platform whose std mutex is boxed lazily (Apple's
/// pthread mutex) allocates on a fresh mutex's first lock.
fn round<W: Work>(
    exec: Exec<'_>,
    work: &W,
    placement: &Placement,
    parts: &mut [Mutex<Part<'_, '_, W::Job, W::Lane>>],
    failure: &Mutex<Option<(usize, W::Error)>>,
) -> Result<(), threadteam::Error> {
    let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
    if let [part] = parts {
        let share = placement.shares[0];
        // `limit` clamps; refuse an oversized share as `fork_sized` would.
        if share > exec.budget() {
            return Err(threadteam::Error::SharesExceedBudget {
                budget: exec.budget(),
                sum: share,
            });
        }
        let width = NonZeroUsize::new(share).ok_or(threadteam::Error::ZeroBudget)?;
        let mut exec = Exec::from_token(exec.into_token().limit(width), workspace, policy);
        let part = part.get_mut().unwrap_or_else(PoisonError::into_inner);
        part.advance(work, placement, share, &mut exec, failure);
        return Ok(());
    }
    let parts = &*parts;
    exec.into_token().fork_sized(&placement.shares, |l, token| {
        let mut part = parts[l].lock().unwrap_or_else(PoisonError::into_inner);
        let mut exec = Exec::from_token(token, workspace, policy);
        part.advance(work, placement, placement.shares[l], &mut exec, failure);
    })
}
