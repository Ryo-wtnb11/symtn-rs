//! Test probes of symtn-matrixalgebra (#2221): one thread-local ledger of
//! [`Event`] counts, read through [`record`], a log of buffer-identity
//! observations, and the two route-policy flags tests force.
//!
//! The ledger is the only counter store. The `*Probe` structs below are
//! read-only views assembled from it, so a test names the facts it asserts
//! as fields while each fact is still counted once.

use std::cell::RefCell;

/// Declares [`Event`] and, for each view, the struct a [`Recording`] fills
/// from its events. `singles` are events read directly through
/// [`Recording::get`].
macro_rules! events {
    (
        singles { $($(#[$single_meta:meta])* $single:ident),* $(,)? }
        $(
            $(#[$meta:meta])*
            $view:ident => $method:ident { $($(#[$fmeta:meta])* $field:ident: $event:ident),* $(,)? }
        )*
    ) => {
        /// One observed occurrence in factorization execution. Each fact has
        /// one event.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum Event {
            $($($(#[$fmeta])* $event,)*)*
            $($(#[$single_meta])* $single,)*
        }

        const ALL: &[Event] = &[$($(Event::$event,)*)* $(Event::$single,)*];

        $(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
            pub(crate) struct $view {
                $($(#[$fmeta])* pub $field: usize,)*
            }

            impl Recording {
                pub(crate) fn $method(&self) -> $view {
                    $view { $($field: self.get(Event::$event),)* }
                }
            }
        )*
    };
}

events! {
    singles {
        /// Left factor buffers built by multiplicity-free publication.
        LeftFactorBufferBuilds,
        /// Right factor buffers built by multiplicity-free publication.
        RightFactorBufferBuilds,
        /// Calls of `MfAuthority::stage`, one per factor space built. Cache
        /// hits are shown by structure identity across repeated calls.
        MfFactorSpaceStage,
        /// Sectors whose eigenvalues the order authority had to permute.
        EigOrderPermutation,
    }
    /// Owned-output publications of compact SVD.
    CompactSvdCopyProbe => compact_svd_copy {
        owned_output_publications: SvdOwnedOutputPublications,
        owned_output_owner_reused: SvdOwnedOutputOwnerReused,
    }
    /// Diagonal bond constructions.
    DiagonalBondBuildProbe => diagonal_bond_build {
        calls: DiagonalBondBuilds,
        values: DiagonalBondValues,
    }
    /// The placement index of multiplicity-free pair publication.
    PlacementIndexProbe => placement_index {
        /// Index tables allocated (one per owner call).
        index_builds: PlacementIndexBuilds,
        /// `(matricization, side)` pairs inserted into a table.
        indexed_sides: PlacementIndexedSides,
        indexed_trees: PlacementIndexedTrees,
        lookups: PlacementLookups,
    }
    /// Block visits of the paired scatter helpers.
    ScatterVisitProbe => scatter_visit {
        /// Output blocks scanned by the grouping pass, per side (`B`).
        left_grouped: LeftGrouped,
        right_grouped: RightGrouped,
        /// Groupings built, per side (one per publication call).
        left_groups_built: LeftGroupsBuilt,
        right_groups_built: RightGroupsBuilt,
        /// Output blocks iterated by the paired scatter helpers, per side (`F`).
        left_visits: LeftVisits,
        right_visits: RightVisits,
    }
    /// Publication of the generic checked factor pair.
    GenericPairPublicationProbe => generic_pair_publication {
        ordered_key_validation_events: PairOrderedKeyValidations,
        fallback_row_lookups: PairFallbackRowLookups,
        fallback_col_lookups: PairFallbackColLookups,
        left_owner_reused: PairLeftOwnerReused,
        right_owner_reused: PairRightOwnerReused,
        left_appended_elements: PairLeftAppended,
        right_appended_elements: PairRightAppended,
        left_scattered_elements: PairLeftScatteredElements,
        right_scattered_elements: PairRightScatteredElements,
        left_scatter_calls: PairLeftScatterCalls,
        right_scatter_calls: PairRightScatterCalls,
        canonical_publications: PairCanonicalPublications,
        fallback_publications: PairFallbackPublications,
    }
    /// Publication of one side of a factor pair.
    OneSidedPublicationProbe => one_sided_publication {
        canonical_publications: OneSidedCanonicalPublications,
        fallback_publications: OneSidedFallbackPublications,
        owner_reused: OneSidedOwnerReused,
        /// Elements written by canonical publication beyond the moved first
        /// owner: appended factors plus in-place identity blocks.
        appended_elements: OneSidedAppendedElements,
        /// Heap bytes held by publication plans (identity segment lists).
        plan_bytes: OneSidedPlanBytes,
    }
}

const EVENTS: usize = ALL.len();

/// A buffer-identity observation: where a buffer lived when the code under
/// test handed it on, so tests can prove it was moved rather than copied.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Observation {
    CheckedCompactInput(CheckedCompactInputObservation),
    CheckedEighPairs(Vec<usize>),
    CheckedSvdStage(usize, usize),
    StagedDestination(usize, [Option<usize>; 3]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckedCompactOperation {
    Qr,
    Svd,
    Lq,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckedCompactInputObservation {
    pub operation: CheckedCompactOperation,
    pub input_pointer: usize,
    pub matrix_pointer: usize,
    pub elements: usize,
}

struct Probe {
    counts: [usize; EVENTS],
    log: Vec<Observation>,
}

impl Default for Probe {
    fn default() -> Self {
        Self {
            counts: [0; EVENTS],
            log: Vec::new(),
        }
    }
}

std::thread_local! {
    static PROBE: RefCell<Probe> = RefCell::new(Probe::default());
}

/// The probe state of the calling thread. A scope body may run on a backend
/// worker thread while the test reads on its own; moving the state in and out
/// makes the probes observe the call whichever thread runs it.
pub(crate) struct Carried(Probe);

impl Carried {
    pub(crate) fn take() -> Self {
        Self(PROBE.with(|probe| std::mem::take(&mut *probe.borrow_mut())))
    }

    pub(crate) fn put(self) {
        PROBE.with(|probe| *probe.borrow_mut() = self.0);
    }
}

/// Records one `event`.
pub(crate) fn hit(event: Event) {
    add(event, 1);
}

/// Records `count` occurrences of `event`.
pub(crate) fn add(event: Event, count: usize) {
    PROBE.with(|probe| probe.borrow_mut().counts[event as usize] += count);
}

fn observe(observation: Observation) {
    PROBE.with(|probe| probe.borrow_mut().log.push(observation));
}

/// What this thread recorded since [`record`] returned this guard.
pub(crate) struct Recording {
    counts: [usize; EVENTS],
    log_start: usize,
}

/// Starts a recording.
pub(crate) fn record() -> Recording {
    PROBE.with(|probe| {
        let probe = probe.borrow();
        Recording {
            counts: probe.counts,
            log_start: probe.log.len(),
        }
    })
}

impl Recording {
    /// Occurrences of `event` since the recording started.
    pub(crate) fn get(&self, event: Event) -> usize {
        PROBE.with(|probe| probe.borrow().counts[event as usize]) - self.counts[event as usize]
    }

    fn observed<T>(&self, pick: impl Fn(&Observation) -> Option<T>) -> Vec<T> {
        PROBE.with(|probe| {
            probe.borrow().log[self.log_start..]
                .iter()
                .filter_map(pick)
                .collect()
        })
    }

    pub(crate) fn checked_compact_inputs(&self) -> Vec<CheckedCompactInputObservation> {
        self.observed(|o| match o {
            Observation::CheckedCompactInput(input) => Some(*input),
            _ => None,
        })
    }

    /// The pair pointers of the last checked Hermitian publication.
    pub(crate) fn checked_eigh_pairs(&self) -> Vec<usize> {
        self.observed(|o| match o {
            Observation::CheckedEighPairs(pointers) => Some(pointers.clone()),
            _ => None,
        })
        .pop()
        .unwrap_or_default()
    }

    pub(crate) fn checked_svd_stages(&self) -> Vec<(usize, usize)> {
        self.observed(|o| match o {
            Observation::CheckedSvdStage(u, vt) => Some((*u, *vt)),
            _ => None,
        })
    }

    /// Every destination the staging boundary handed to a sector, with its
    /// sector index and each output's payload address.
    pub(crate) fn staged_destinations(&self) -> Vec<(usize, [Option<usize>; 3])> {
        self.observed(|o| match o {
            Observation::StagedDestination(index, outputs) => Some((*index, *outputs)),
            _ => None,
        })
    }

    /// Factor buffers built as `(left, right)`.
    pub(crate) fn factor_buffer_builds(&self) -> (usize, usize) {
        (
            self.get(Event::LeftFactorBufferBuilds),
            self.get(Event::RightFactorBufferBuilds),
        )
    }
}

/// Records an owned-output publication of SVD, and whether the first
/// owner's allocation was kept.
pub(crate) fn svd_owned_output(owner_reused: bool) {
    hit(Event::SvdOwnedOutputPublications);
    add(Event::SvdOwnedOutputOwnerReused, usize::from(owner_reused));
}

pub(crate) fn checked_compact_input<D>(
    operation: CheckedCompactOperation,
    input: &[D],
    matrix: &[D],
) {
    observe(Observation::CheckedCompactInput(
        CheckedCompactInputObservation {
            operation,
            input_pointer: input.as_ptr() as usize,
            matrix_pointer: matrix.as_ptr() as usize,
            elements: matrix.len(),
        },
    ));
}

pub(crate) fn checked_eigh_pairs(pointers: Vec<usize>) {
    observe(Observation::CheckedEighPairs(pointers));
}

pub(crate) fn checked_svd_stage<D>(u: &[D], vt: &[D]) {
    observe(Observation::CheckedSvdStage(
        u.as_ptr() as usize,
        vt.as_ptr() as usize,
    ));
}

/// Records the payload address of each output of a destination handed to
/// sector `index` by the staging boundary.
pub(crate) fn staged_destination(index: usize, dest: &mut symtn_dense::FactorDest<'_>) {
    fn address(write: &symtn_dense::DenseWrite<'_>) -> usize {
        use symtn_dense::DenseWrite as W;
        match write {
            W::F32(view) => view.data()[view.offset()..].as_ptr() as usize,
            W::F64(view) => view.data()[view.offset()..].as_ptr() as usize,
            W::C32(view) => view.data()[view.offset()..].as_ptr() as usize,
            W::C64(view) => view.data()[view.offset()..].as_ptr() as usize,
            W::I32(view) => view.data()[view.offset()..].as_ptr() as usize,
            W::I64(view) => view.data()[view.offset()..].as_ptr() as usize,
            W::Bool(view) => view.data()[view.offset()..].as_ptr() as usize,
        }
    }
    observe(Observation::StagedDestination(
        index,
        dest.outputs_mut()
            .each_ref()
            .map(|output| output.as_ref().map(address)),
    ));
}
