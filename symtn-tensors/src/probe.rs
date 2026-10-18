//! Test probes of symtn-tensors (#2221): one thread-local ledger of
//! [`Event`] counts, read as a window through [`record`] or consumed
//! through [`take`], plus the armed hooks that run inside production code.
//!
//! Every count is per thread, so concurrent tests do not move each other's.
//! Process-global statistics (`StructureCache` hits and misses, the reset
//! epoch), the cache test lock and public profiling types are not probes
//! and stay with their owners. Fixtures that count calls into a test's own
//! callback live in the test that supplies the callback.

use std::cell::RefCell;

use crate::OperationError;

/// One observed occurrence in tensor execution. Each fact has one event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// A lazy adjoint's block structure was built.
    AdjointViewBuild,
    /// A fusion-aware trace compiled its axis plan.
    TraceAxisPlanDerivation,
    /// A fusion-aware trace derived its selected HomSpace.
    TraceSelectedHomspaceDerivation,
    /// A source layout was matched to the core layout by HomSpace id.
    SourceLayoutHomspaceIdComparison,
    /// The core contraction preflight compiled.
    CoreContractPreflight,
    /// A core destination HomSpace was derived and compared.
    CoreHomspaceDerivation,
    /// A zero-copy contraction candidate walk ran.
    CandidateWalk,
    /// A contraction axis plan was compiled, at any call site.
    ContractAxisPlanCompile,
    /// A contraction candidate was scored.
    CandidateScore,
    /// The fusion contraction preflight ran for candidate selection.
    ContractPreflight,
    /// A candidate's transformed layout was probed (one per operand side).
    CandidateLayoutProbe,
    /// A fusion operand projection was prepared.
    FusionOperandProjectionPrepare,
    /// A final result layout was built.
    FinalResultLayoutBuild,
    /// A layout was built on the caller-supplied shapes path.
    CallerShapePathBuild,
    /// A scratch structure was built.
    ScratchStructureBuild,
    /// A permuted or contracted HomSpace was built by a layout capability.
    DerivedHomspaceBuild,
    /// A fusion block matrix layout answered a group lookup.
    FusionLayoutLookup,
    /// A fusion block matrix layout was compiled.
    FusionLayoutCompile,
    /// A multiplicity-free tree-transform capability was validated.
    MultiplicityFreeCapabilityValidation,
    /// An ordered tree-pair group was lowered to specs.
    OrderedTreePairLowering,
    /// A tree-pair group was assembled by the per-row callback assembler.
    RowCallbackTreePairAssembly,
    /// A tree-pair operation was prepared.
    TreePairOperationPreparation,
    /// An oriented tree-pair plan was compiled.
    OrientedTreePairCompile,
    /// A completed-transformer lookup was answered by a resident entry.
    CompletedHit,
    /// A completed-transformer miss ran a build.
    CompletedBuild,
    /// A completed transformer was offered for publication.
    CompletedPublication,
    /// A coefficient-group lookup was answered by a resident entry.
    GroupHit,
    /// A coefficient-group lookup found none.
    GroupMiss,
    /// A built coefficient group was offered for publication.
    GroupPublication,
    /// A trace-column lookup was answered by a resident entry.
    TraceHit,
    /// A trace-column lookup found none.
    TraceMiss,
    /// A built trace column was offered for publication.
    TracePublication,
}

const EVENTS: usize = Event::TracePublication as usize + 1;

/// Runs once, inside the next raw trace execution on this thread.
pub(crate) type TraceRawExecutionHook = Box<dyn FnOnce() -> Result<(), OperationError>>;

struct Probe {
    counts: [usize; EVENTS],
    /// What [`take`] has already returned, per event.
    taken: [usize; EVENTS],
    /// Source blocks of the trace transforms, in invocation order.
    trace_sources: Vec<usize>,
    before_publish: Option<Box<dyn FnOnce()>>,
    trace_raw_execution: Option<TraceRawExecutionHook>,
}

std::thread_local! {
    static PROBE: RefCell<Probe> = const {
        RefCell::new(Probe {
            counts: [0; EVENTS],
            taken: [0; EVENTS],
            trace_sources: Vec::new(),
            before_publish: None,
            trace_raw_execution: None,
        })
    };
}

/// Records one `event`.
pub(crate) fn hit(event: Event) {
    PROBE.with(|probe| probe.borrow_mut().counts[event as usize] += 1);
}

/// What this thread recorded since [`record`] returned this guard.
pub(crate) struct Recording {
    counts: [usize; EVENTS],
}

/// Starts a recording.
pub(crate) fn record() -> Recording {
    Recording {
        counts: PROBE.with(|probe| probe.borrow().counts),
    }
}

impl Recording {
    /// Occurrences of `event` since the recording started.
    pub(crate) fn get(&self, event: Event) -> usize {
        PROBE.with(|probe| probe.borrow().counts[event as usize]) - self.counts[event as usize]
    }
}

/// Occurrences of `event` on this thread since the last `take` of it.
pub(crate) fn take(event: Event) -> usize {
    PROBE.with(|probe| {
        let probe = &mut *probe.borrow_mut();
        let seen = probe.counts[event as usize] - probe.taken[event as usize];
        probe.taken[event as usize] = probe.counts[event as usize];
        seen
    })
}

/// This thread's completed-transformer activity since the last take.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CompletedActivity {
    /// Lookups answered by a resident entry.
    pub(crate) hits: usize,
    /// Misses that ran a build.
    pub(crate) builds: usize,
    /// Builds offered for publication (publishable keys only).
    pub(crate) publications: usize,
}

pub(crate) fn take_completed_transformer_activity() -> CompletedActivity {
    CompletedActivity {
        hits: take(Event::CompletedHit),
        builds: take(Event::CompletedBuild),
        publications: take(Event::CompletedPublication),
    }
}

/// This thread's composed-coefficient or trace-column activity since the
/// last take.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CoefficientGroupActivity {
    /// Group lookups answered by a resident entry.
    pub(crate) hits: usize,
    /// Group lookups that found none (each is built unless an earlier
    /// group's build fails).
    pub(crate) misses: usize,
    /// Built groups offered for publication.
    pub(crate) publications: usize,
}

pub(crate) fn take_coefficient_group_activity() -> CoefficientGroupActivity {
    CoefficientGroupActivity {
        hits: take(Event::GroupHit),
        misses: take(Event::GroupMiss),
        publications: take(Event::GroupPublication),
    }
}

/// Trace-column activity; transform groups are counted apart by
/// [`take_coefficient_group_activity`].
pub(crate) fn take_trace_column_activity() -> CoefficientGroupActivity {
    CoefficientGroupActivity {
        hits: take(Event::TraceHit),
        misses: take(Event::TraceMiss),
        publications: take(Event::TracePublication),
    }
}

/// Notes that a trace transform ran for source block `source`.
pub(crate) fn push_trace_source(source: usize) {
    PROBE.with(|probe| probe.borrow_mut().trace_sources.push(source));
}

/// Forgets the trace transform sources noted so far.
pub(crate) fn reset_trace_sources() {
    PROBE.with(|probe| probe.borrow_mut().trace_sources.clear());
}

/// The trace transform sources noted since the last reset or take.
pub(crate) fn take_trace_sources() -> Vec<usize> {
    PROBE.with(|probe| std::mem::take(&mut probe.borrow_mut().trace_sources))
}

/// Arms `hook` to run on this thread right before its next
/// completed-transformer publication, after the build: a test reaches the
/// straddling-clear window there.
pub(crate) fn set_before_publish_hook(hook: Box<dyn FnOnce()>) {
    PROBE.with(|probe| probe.borrow_mut().before_publish = Some(hook));
}

/// Takes the armed before-publication hook.
pub(crate) fn take_before_publish_hook() -> Option<Box<dyn FnOnce()>> {
    PROBE.with(|probe| probe.borrow_mut().before_publish.take())
}

/// Arms `hook` to run inside this thread's next raw trace execution.
pub(crate) fn set_trace_raw_execution_hook(hook: TraceRawExecutionHook) {
    PROBE.with(|probe| probe.borrow_mut().trace_raw_execution = Some(hook));
}

/// Takes the armed raw-trace-execution hook.
pub(crate) fn take_trace_raw_execution_hook() -> Option<TraceRawExecutionHook> {
    PROBE.with(|probe| probe.borrow_mut().trace_raw_execution.take())
}
