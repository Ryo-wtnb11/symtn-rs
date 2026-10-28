//! Test probes of symtn-core (#2221): one thread-local ledger of [`Event`]
//! counts, read through [`record`], and the one reset-hook slot.
//!
//! Process-global statistics (`StructureCache` hits and misses, the reset
//! epoch) are not probes and stay with their owners. The composer step
//! counts are the one measurement a sibling crate reads, so they stay
//! available under the `testing` feature.

#[cfg(any(test, feature = "testing"))]
pub(crate) use steps::*;

#[cfg(any(test, feature = "testing"))]
mod steps {
    use crate::GenericBlockStepCounts;

    std::thread_local! {
        static STEPS: std::cell::Cell<GenericBlockStepCounts> =
            const { std::cell::Cell::new(GenericBlockStepCounts {
                steps: 0,
                moves: 0,
                per_source_moves: 0,
                spread_multiplies: 0,
            }) };
    }

    /// Adds one composed step to this thread's counts.
    pub(crate) fn add_step(moves: u64, per_source_moves: u64, spread_multiplies: u64) {
        STEPS.with(|cell| {
            let mut next = cell.get();
            next.steps += 1;
            next.moves += moves;
            next.per_source_moves += per_source_moves;
            next.spread_multiplies += spread_multiplies;
            cell.set(next);
        });
    }

    /// This thread's counts since the last call.
    pub(crate) fn take_steps() -> GenericBlockStepCounts {
        STEPS.with(|cell| cell.replace(GenericBlockStepCounts::default()))
    }
}

#[cfg(test)]
pub(crate) use ledger::*;

#[cfg(test)]
mod ledger {
    use std::cell::RefCell;

    /// One observed occurrence in core execution. Each fact has one event.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Event {
        /// A hom-space descriptor was materialized into an owned hom-space.
        DescriptorMaterialization,
        /// A storage-injectivity check ran its exact element-level scan.
        ExactStorageFallback,
        /// A borrowed unique braid answered a position query from raw axes.
        UniqueBorrowedPositionQuery,
        /// A permuted-destination memo answered a lookup.
        PermutedMemoHit,
        /// A permuted-destination memo lookup read the overflow map.
        PermutedMemoMapRead,
        /// A permuted-destination memo publication was charged.
        PermutedMemoPublication,
        /// A complete-HomSpace structure was offered for admission.
        CompleteHomSpaceMiss,
        /// A key was looked up in the sector-structure cache.
        SectorStructureLookup,
        /// A fusion-tree layout was built cold and published.
        LayoutBuild,
        /// A fusion-tree layout was published.
        LayoutAdmission,
        /// A coupled grid was inserted by reconstruction.
        CoupledGridReconstructionInsert,
        /// A coupled grid derived one side.
        CoupledGridSideDerivation,
        /// A compact block was scattered to keyed terms.
        CompactBlockScatter,
        /// Destination rows of those compact blocks.
        CompactDestinationRows,
        /// Source columns of those compact blocks.
        CompactSourceColumns,
        /// Coefficient slots of those compact blocks.
        CompactCoefficientSlots,
        /// Coefficient bytes of those compact blocks.
        CompactCoefficientBytes,
    }

    const EVENTS: usize = Event::CompactCoefficientBytes as usize + 1;

    struct Probe {
        counts: [usize; EVENTS],
        mid_reset_hook: Option<Box<dyn FnOnce()>>,
    }

    std::thread_local! {
        static PROBE: RefCell<Probe> = const {
            RefCell::new(Probe { counts: [0; EVENTS], mid_reset_hook: None })
        };
    }

    /// Records one `event`.
    pub(crate) fn hit(event: Event) {
        add(event, 1);
    }

    /// Records `count` occurrences of `event`.
    pub(crate) fn add(event: Event, count: usize) {
        PROBE.with(|probe| probe.borrow_mut().counts[event as usize] += count);
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

    /// Runs `hook` once, in the middle of the next structure-cache reset on
    /// this thread.
    pub(crate) fn set_mid_reset_hook(hook: Box<dyn FnOnce()>) {
        PROBE.with(|probe| probe.borrow_mut().mid_reset_hook = Some(hook));
    }

    /// Takes the armed mid-reset hook.
    pub(crate) fn take_mid_reset_hook() -> Option<Box<dyn FnOnce()>> {
        PROBE.with(|probe| probe.borrow_mut().mid_reset_hook.take())
    }
}
