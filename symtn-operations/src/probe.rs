//! Test probes of symtn-operations (#2221): one thread-local ledger of
//! [`Event`] counts, consumed through [`take`], plus the armed write
//! bitmaps of the owned writers.
//!
//! The ledger exists under the `testing` feature because symtn-tensors tests
//! read the two public takers; the bitmaps are this crate's own tests.
//! Process-global observation (`host_pool`'s pool observations) is not a
//! probe and stays with its owner.

use std::cell::RefCell;

/// One observed occurrence in dense-operation execution. Each fact has one
/// event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Event {
    /// A checked block layout was filled (one pass per block per operand set).
    LayoutPass,
    /// A checked update walked one strided span.
    SpanWalk,
    /// An output element was zero-filled before an owned block overwrite
    /// because the layout was not proved to tile its storage.
    OwnedPrefillElement,
    /// A movement phase evaluated the width model.
    MovementModelEvaluation,
    /// A replay built its movement decisions (a decision-memo miss).
    MovementDecisionBuild,
}

const EVENTS: usize = Event::MovementDecisionBuild as usize + 1;

struct Probe {
    counts: [usize; EVENTS],
    /// What [`take`] has already returned, per event.
    taken: [usize; EVENTS],
    #[cfg(test)]
    trace_writes: Option<Vec<usize>>,
    #[cfg(test)]
    cat_writes: Option<Vec<usize>>,
}

std::thread_local! {
    static PROBE: RefCell<Probe> = const {
        RefCell::new(Probe {
            counts: [0; EVENTS],
            taken: [0; EVENTS],
            #[cfg(test)]
            trace_writes: None,
            #[cfg(test)]
            cat_writes: None,
        })
    };
}

/// Records `count` occurrences of `event`.
pub(crate) fn add(event: Event, count: usize) {
    PROBE.with(|probe| probe.borrow_mut().counts[event as usize] += count);
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

/// Layout passes and span walks of the checked block updates.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CheckedBlockPasses {
    pub layout_passes: usize,
    pub span_walks: usize,
}

/// Returns and resets this thread's [`CheckedBlockPasses`].
#[doc(hidden)]
pub fn take_checked_block_passes() -> CheckedBlockPasses {
    CheckedBlockPasses {
        layout_passes: take(Event::LayoutPass),
        span_walks: take(Event::SpanWalk),
    }
}

/// Returns and resets the number of output elements this thread zero-filled
/// before an owned block overwrite because the layout was not proved to tile
/// its storage.
#[doc(hidden)]
pub fn take_owned_block_prefills() -> usize {
    take(Event::OwnedPrefillElement)
}

/// Returns and resets this thread's movement width model evaluation count.
#[doc(hidden)]
pub fn take_movement_model_evaluations() -> usize {
    take(Event::MovementModelEvaluation)
}

/// Returns and resets this thread's count of movement decision builds (memo
/// misses), counted whether or not the movement model was calibrated.
#[doc(hidden)]
pub fn take_movement_decision_builds() -> usize {
    take(Event::MovementDecisionBuild)
}

/// Which owned writer a write bitmap observes.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum Writer {
    Trace,
    Cat,
}

#[cfg(test)]
fn slot(probe: &mut Probe, writer: Writer) -> &mut Option<Vec<usize>> {
    match writer {
        Writer::Trace => &mut probe.trace_writes,
        Writer::Cat => &mut probe.cat_writes,
    }
}

/// Arms a zeroed write counter of `len` destination elements for `writer`.
#[cfg(test)]
pub(crate) fn arm_writes(writer: Writer, len: usize) {
    PROBE.with(|probe| *slot(&mut probe.borrow_mut(), writer) = Some(vec![0; len]));
}

/// Whether a write counter is armed for `writer`.
#[cfg(test)]
pub(crate) fn writes_armed(writer: Writer) -> bool {
    PROBE.with(|probe| slot(&mut probe.borrow_mut(), writer).is_some())
}

/// Counts one write to destination element `index`, if armed.
#[cfg(test)]
pub(crate) fn observe_write(writer: Writer, index: usize) {
    PROBE.with(|probe| {
        if let Some(writes) = slot(&mut probe.borrow_mut(), writer).as_mut() {
            writes[index] += 1;
        }
    });
}

/// Disarms the counter of `writer` and returns the writes it saw.
#[cfg(test)]
pub(crate) fn take_writes(writer: Writer) -> Vec<usize> {
    PROBE.with(|probe| slot(&mut probe.borrow_mut(), writer).take().unwrap())
}
