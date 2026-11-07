use super::*;

/// The job bodies of a batch. Job indices are dense and stable for one
/// placement and its run.
pub trait Work: Sync {
    /// One job's exclusive state.
    type Job: Send;
    /// One lane's state (its initialized source and destination buffers),
    /// kept for the whole call and lent to every job the lane runs.
    type Lane: Send;
    type Error: Send;

    /// The number of jobs J.
    fn count(&self) -> usize;

    /// Runs `job` on `exec`, whose budget is its lane's share under
    /// `Placed` and the job's own token under the other assignments: a
    /// movement body dispatches at most that many participants and checks,
    /// under `Auto`, that its estimate pays for its final fork on `exec`'s
    /// token (running inline otherwise), and a provider body hands `exec` to
    /// the provider as its whole offer.
    fn run(
        &self,
        job: usize,
        state: &mut Self::Job,
        lane: &mut Self::Lane,
        exec: Exec<'_>,
    ) -> Result<(), Self::Error>;
}

type ParkedRef = (usize, &'static mut ());

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

/// A run's job outcome: every job succeeded, or the lowest-indexed failing
/// job and its error.
pub type Outcome<E> = Result<(), (usize, E)>;

/// How [`run_native`] lends a batch's jobs to the slots of its context:
/// [`Split`], [`Each`] or the block kernels' `Placed`. Each assignment states what it needs of
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
/// or one job is a leaf, which runs its jobs in order on its own token.
///
/// The call lends its one lane state (`lanes[0]`) to the root part. At each
/// fork `split` divides the part's lane: it shrinks it to the left half's
/// and returns the right half's, so a lane can carry the part's own slice of
/// a destination. A failing `split` stops its part before either half runs,
/// reported as the part's first job.
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
    pub split: F,
}

/// Where a [`Split`] part forks: `job` is the right half's first job, and
/// `left` of the part's `threads` logical threads go to the left half.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    pub job: usize,
    pub left: usize,
    pub threads: usize,
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
/// its share; lane `l` borrows lane state `l`, and every job runs on its
/// lane's whole share. Every job runs, and the lowest failing job is
/// returned.
#[derive(Debug)]
pub(crate) struct Placed<'a> {
    pub(crate) tables: &'a mut LaneTables,
}

/// Runs every job of `assignment` on `exec` with exclusive access to its own
/// `states` entry, and returns the lowest-indexed job's failure, as each
/// [`Assignment`] describes.
///
/// A panic unwinds after the forks join. The outer error, before any job
/// ran, is a [`Refusal`]: `states` not holding one entry per job (the
/// placement's under `Placed`, `work`'s count otherwise), `lanes` holding
/// fewer entries than the placement has lanes (`Placed`) or the work has
/// jobs ([`Each`]), or threadteam's refusal of the lane shares (`Placed`)
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

/// Why a run did not start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal<R> {
    /// The context's refusal: threadteam's, of lane shares or slots, before
    /// any job.
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
    F: Fn(&mut W::Lane, Cut) -> Result<W::Lane, W::Error> + Sync,
{
    fn run(
        self,
        exec: Exec<'_>,
        work: &W,
        states: &mut [W::Job],
        lanes: &mut [W::Lane],
    ) -> Result<Outcome<W::Error>, Refusal<threadteam::Error>> {
        let jobs = work.count();
        supply(jobs, states.len(), usize::from(jobs > 0), lanes.len())?;
        if states.is_empty() {
            return Ok(Ok(()));
        }
        let failure = split(
            work,
            &self.split,
            self.threads,
            0,
            states,
            &mut lanes[0],
            exec,
        );
        Ok(failure.map_or(Ok(()), Err))
    }
}

/// Runs `job` on `exec`; its failure, if any, with its index.
fn run_job<W: Work>(
    work: &W,
    job: usize,
    state: &mut W::Job,
    lane: &mut W::Lane,
    exec: &mut Exec<'_>,
) -> Option<(usize, W::Error)> {
    work.run(job, state, lane, exec.reborrow())
        .err()
        .map(|error| (job, error))
}

/// [`Split`] over `states`, whose first job is `first`, on `lane`.
fn split<W, F>(
    work: &W,
    divide: &F,
    threads: usize,
    first: usize,
    states: &mut [W::Job],
    lane: &mut W::Lane,
    mut exec: Exec<'_>,
) -> Option<(usize, W::Error)>
where
    W: Work,
    W::Job: Sync,
    W::Lane: Sync,
    W::Error: Sync,
    F: Fn(&mut W::Lane, Cut) -> Result<W::Lane, W::Error> + Sync,
{
    if threads <= 1 || states.len() <= 1 {
        return states
            .iter_mut()
            .enumerate()
            .find_map(|(i, state)| run_job(work, first + i, state, lane, &mut exec));
    }
    let middle = parallel_split(states.len(), threads);
    let (left_states, right_states) = states.split_at_mut(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let right = first + middle;
    let cut = Cut {
        job: right,
        left: left_threads,
        threads,
    };
    let mut right_lane = match divide(lane, cut) {
        Ok(right_lane) => right_lane,
        Err(error) => return Some((first, error)),
    };
    let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
    join(
        exec.into_token(),
        |token| {
            let exec = Exec::from_token(token, workspace, policy);
            split(work, divide, left_threads, first, left_states, lane, exec)
        },
        |token| {
            let exec = Exec::from_token(token, workspace, policy);
            let lane = &mut right_lane;
            split(work, divide, right_threads, right, right_states, lane, exec)
        },
    )
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
        let failure = run_lane(work, &mut exec, jobs, &mut lanes[0]);
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
        if let Some((job, error)) = run_lane(work, &mut exec, jobs, lane) {
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

/// Runs one lane's `jobs`, ascending, each on the lane's token; a failing
/// job does not stop the lane. Returns the lane's first, so lowest, failure.
fn run_lane<'s, W: Work>(
    work: &W,
    exec: &mut Exec<'_>,
    jobs: impl Iterator<Item = (usize, &'s mut W::Job)>,
    lane: &mut W::Lane,
) -> Option<(usize, W::Error)>
where
    W::Job: 's,
{
    let mut first = None;
    for (j, state) in jobs {
        if let Some(failure) = run_job(work, j, state, lane, exec) {
            first.get_or_insert(failure);
        }
    }
    first
}
