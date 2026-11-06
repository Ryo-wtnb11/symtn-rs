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
    /// selected width (`Placement::selected`): a movement stage dispatches
    /// at most that many participants and checks, under `Auto`, that its
    /// estimate pays for its final fork on `exec`'s token (running inline
    /// otherwise), and a provider stage hands `exec` to
    /// the provider as its whole offer. The barrier form never calls it for a
    /// provider stage.
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

/// The job references and per-lane parts of earlier runs, kept empty
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
        assert_eq!(states.len(), placement.lane_of.len(), "one state per job");
        assert!(lanes.len() >= count, "one lane state per lane");
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

    /// Runs the lane's stages on `exec` until it is done or, with `cut`, at
    /// a provider stage.
    fn advance<W: Work<Job = S, Lane = L>>(
        &mut self,
        work: &W,
        placement: &Placement,
        share: usize,
        exec: &mut Exec<'_>,
        cut: bool,
        failure: &Mutex<Option<(usize, W::Error)>>,
    ) {
        while let Some((j, k)) = self.next(work) {
            if cut && work.kind(j, k) == StageKind::Provider {
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

/// Runs every job of `tables`' placement on `exec` with exclusive access to
/// its own `states` entry, every stage inside its lane, and returns the
/// lowest-indexed job's failure.
///
/// Lane `l` is one `fork_sized` child of exactly its share, with `exec`'s
/// workspace and policy, and borrows `lanes[l]` for the call. A job stops at
/// its own first failing stage; every other job runs to its end. A panic
/// unwinds after the lanes join. The outer error is threadteam's refusal of
/// the lane shares, before any stage ran: unreachable for a placement made on
/// `exec`'s budget.
///
/// # Panics
///
/// When `states` does not hold one entry per placed job, or `lanes` fewer
/// entries than the placement has lanes.
pub fn run_native<W: Work>(
    exec: Exec<'_>,
    work: &W,
    tables: &mut LaneTables,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, threadteam::Error> {
    let LaneTables { placement, run, .. } = tables;
    if placement.lanes() == 0 {
        return Ok(Ok(()));
    }
    let failure = Mutex::new(None);
    let mut parts = run.lend(placement, states, lanes);
    let forked = round(exec, work, placement, &mut parts, false, &failure);
    run.park(parts);
    forked?;
    let failure = failure.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(failure.map_or(Ok(()), Err))
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
/// are job-local exactly as in [`run_native`], so the reported job does not
/// depend on the lane count; a failed submission is never retried. The outer
/// error is `host`'s refused entry, which stops the call.
///
/// # Panics
///
/// As [`run_native`].
pub fn run_barrier<W: Work, H: Barrier<W>>(
    host: &mut H,
    work: &W,
    tables: &mut LaneTables,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, H::Refused> {
    let LaneTables { placement, run, .. } = tables;
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
        let entered =
            host.region(&mut |exec| round(exec, work, placement, &mut parts, true, &failure));
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
    outcome?;
    let failure = failure.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(failure.map_or(Ok(()), Err))
}

/// Runs every lane's stages on `exec` (up to its next provider stage with
/// `cut`): one `fork_sized` child of exactly its share per lane.
///
/// Why one lane runs without the fork and its lock: the fork would run it
/// inline anyway, and a platform whose std mutex is boxed lazily (Apple's
/// pthread mutex) allocates on a fresh mutex's first lock.
fn round<W: Work>(
    exec: Exec<'_>,
    work: &W,
    placement: &Placement,
    parts: &mut [Mutex<Part<'_, '_, W::Job, W::Lane>>],
    cut: bool,
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
        part.advance(work, placement, share, &mut exec, cut, failure);
        return Ok(());
    }
    let parts = &*parts;
    exec.into_token().fork_sized(&placement.shares, |l, token| {
        let mut part = parts[l].lock().unwrap_or_else(PoisonError::into_inner);
        let mut exec = Exec::from_token(token, workspace, policy);
        part.advance(
            work,
            placement,
            placement.shares[l],
            &mut exec,
            cut,
            failure,
        );
    })
}
