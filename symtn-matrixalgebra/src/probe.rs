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
        /// A values-only matricization fell back to the packed path.
        ValuesMatricizationFallback,
        /// Bytes allocated by `sector_matricizations{,_generic}`.
        InputPackBytes,
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
    /// Copies of compact SVD: input packs, output scatters, owned outputs.
    CompactSvdCopyProbe => compact_svd_copy {
        input_pack_calls: SvdInputPackCalls,
        input_pack_bytes: SvdInputPackBytes,
        output_scatter_calls: SvdOutputScatterCalls,
        output_scatter_bytes: SvdOutputScatterBytes,
        owned_output_publications: SvdOwnedOutputPublications,
        owned_output_owner_reused: SvdOwnedOutputOwnerReused,
    }
    /// Copies of compact QR.
    CompactQrCopyProbe => compact_qr_copy {
        input_pack_calls: QrInputPackCalls,
        input_pack_bytes: QrInputPackBytes,
        output_scatter_calls: QrOutputScatterCalls,
        output_scatter_bytes: QrOutputScatterBytes,
        owned_output_publications: QrOwnedOutputPublications,
        owned_output_owner_reused: QrOwnedOutputOwnerReused,
    }
    /// Copies of compact LQ.
    CompactLqCopyProbe => compact_lq_copy {
        input_pack_calls: LqInputPackCalls,
        input_pack_bytes: LqInputPackBytes,
        output_scatter_calls: LqOutputScatterCalls,
        output_scatter_bytes: LqOutputScatterBytes,
        scratch_buffer_count: LqScratchBuffers,
        scratch_capacity_bytes: LqScratchCapacityBytes,
        adjoint_scratch_fill_calls: LqAdjointFillCalls,
        adjoint_scratch_fill_bytes: LqAdjointFillBytes,
        final_adjoint_copy_calls: LqFinalAdjointCopyCalls,
        final_adjoint_copy_bytes: LqFinalAdjointCopyBytes,
        output_prefill_bytes: LqOutputPrefillBytes,
    }
    /// Copies of Hermitian eigendecomposition.
    EighCopyProbe => eigh_copy {
        input_pack_calls: EighInputPackCalls,
        input_pack_bytes: EighInputPackBytes,
        output_scatter_calls: EighOutputScatterCalls,
        output_scatter_bytes: EighOutputScatterBytes,
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
    EighOwnedVector(usize),
    CheckedEighPairs(Vec<usize>),
    CheckedSvdStage(usize, usize),
    MfSvdFallback(usize, usize),
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
    pub adjoint_pointer: Option<usize>,
    pub elements: usize,
}

/// The compact factorization whose copies an event belongs to.
#[derive(Clone, Copy)]
pub(crate) enum CopyOp {
    Svd,
    Qr,
    Lq,
    Eigh,
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

    pub(crate) fn eigh_owned_vectors(&self) -> Vec<usize> {
        self.observed(|o| match o {
            Observation::EighOwnedVector(pointer) => Some(*pointer),
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

    pub(crate) fn mf_svd_fallbacks(&self) -> Vec<(usize, usize)> {
        self.observed(|o| match o {
            Observation::MfSvdFallback(u, vt) => Some((*u, *vt)),
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

fn bytes<D>(elements: usize) -> usize {
    elements * std::mem::size_of::<D>()
}

/// Records the bytes of the packed matricizations of `sector_matricizations`.
pub(crate) fn input_pack_bytes(sizes: impl Iterator<Item = usize>) {
    add(Event::InputPackBytes, sizes.sum());
}

/// Records the packed input matricizations of one compact `op`; `sizes` are
/// their byte sizes.
pub(crate) fn input_pack(op: CopyOp, sizes: impl ExactSizeIterator<Item = usize>) {
    let (calls, size) = match op {
        CopyOp::Svd => (Event::SvdInputPackCalls, Event::SvdInputPackBytes),
        CopyOp::Qr => (Event::QrInputPackCalls, Event::QrInputPackBytes),
        CopyOp::Lq => (Event::LqInputPackCalls, Event::LqInputPackBytes),
        CopyOp::Eigh => (Event::EighInputPackCalls, Event::EighInputPackBytes),
    };
    add(calls, sizes.len());
    add(size, sizes.sum());
}

/// Records `calls` output scatters of `op` writing `elements` in total.
pub(crate) fn output_scatter<D>(op: CopyOp, calls: usize, elements: usize) {
    let (call, size) = match op {
        CopyOp::Svd => (Event::SvdOutputScatterCalls, Event::SvdOutputScatterBytes),
        CopyOp::Qr => (Event::QrOutputScatterCalls, Event::QrOutputScatterBytes),
        CopyOp::Lq => (Event::LqOutputScatterCalls, Event::LqOutputScatterBytes),
        CopyOp::Eigh => (Event::EighOutputScatterCalls, Event::EighOutputScatterBytes),
    };
    add(call, calls);
    add(size, bytes::<D>(elements));
}

/// Records an owned-output publication of QR or SVD, and whether the first
/// owner's allocation was kept.
pub(crate) fn owned_output(op: CopyOp, owner_reused: bool) {
    let (publications, reused) = match op {
        CopyOp::Svd => (
            Event::SvdOwnedOutputPublications,
            Event::SvdOwnedOutputOwnerReused,
        ),
        CopyOp::Qr => (
            Event::QrOwnedOutputPublications,
            Event::QrOwnedOutputOwnerReused,
        ),
        CopyOp::Lq | CopyOp::Eigh => unreachable!("only QR and SVD publish owned outputs"),
    };
    hit(publications);
    add(reused, usize::from(owner_reused));
}

pub(crate) fn lq_scratch<D>(elements: usize) {
    hit(Event::LqScratchBuffers);
    add(Event::LqScratchCapacityBytes, bytes::<D>(elements));
}

pub(crate) fn lq_adjoint_fill<D>(elements: usize) {
    hit(Event::LqAdjointFillCalls);
    add(Event::LqAdjointFillBytes, bytes::<D>(elements));
}

pub(crate) fn lq_final_adjoint_copy<D>(elements: usize) {
    hit(Event::LqFinalAdjointCopyCalls);
    add(Event::LqFinalAdjointCopyBytes, bytes::<D>(elements));
}

pub(crate) fn lq_output_prefill<D>(elements: usize) {
    add(Event::LqOutputPrefillBytes, bytes::<D>(elements));
}

pub(crate) fn checked_compact_input<D>(
    operation: CheckedCompactOperation,
    input: &[D],
    matrix: &[D],
    adjoint: Option<&[D]>,
) {
    observe(Observation::CheckedCompactInput(
        CheckedCompactInputObservation {
            operation,
            input_pointer: input.as_ptr() as usize,
            matrix_pointer: matrix.as_ptr() as usize,
            adjoint_pointer: adjoint.map(|data| data.as_ptr() as usize),
            elements: matrix.len(),
        },
    ));
}

pub(crate) fn eigh_owned_vector<D>(vectors: &[D]) {
    observe(Observation::EighOwnedVector(vectors.as_ptr() as usize));
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

pub(crate) fn mf_svd_fallback<D>(u: &[D], vt: &[D]) {
    observe(Observation::MfSvdFallback(
        u.as_ptr() as usize,
        vt.as_ptr() as usize,
    ));
}
