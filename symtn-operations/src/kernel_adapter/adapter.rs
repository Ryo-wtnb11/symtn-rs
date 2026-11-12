use super::movement::{
    span, with_payload_map, with_transform_map, Accumulate, Axpby, Init, Update, Write,
};
use super::*;
use strided_kernel::UpdatePlan;

pub(crate) mod sealed {
    use super::*;
    use strided_kernel::execution;

    /// The smallest paying element count of one token shape, found at most
    /// once per call and token shape on one thread.
    ///
    /// Why: Strided's `parallel_width` scans every width up to the budget on each
    /// call, so asking it for every small block of a serial replay costs
    /// O(blocks x budget) and makes wide engines slower on small tensors. A
    /// width pays only if `estimate >= PAYS * dispatch_cost(width)`, which is
    /// monotone in the cost, so no width pays exactly when the cheapest one does
    /// not: one `pays` query against that width rejects a block in O(1), and only
    /// blocks it admits pay for the full width search.
    ///
    /// Why lazy: the O(budget + log bound) scan runs only when an overwrite of
    /// a role with a retained plan reaches admission, so a call of planless or
    /// accumulating blocks pays nothing. A gate serves the blocks of one
    /// thread, because a token's dispatch cost depends on the calling
    /// thread: one per call for the calling thread's unsplit lists, one per
    /// forked leaf otherwise. A leaf of a replay split shares its call's
    /// [`GateKeys`], so the leaves of one call search each cost key at most
    /// once between them.
    ///
    /// Why the memo is a plain field behind `&mut self` rather than a `Cell`:
    /// a forked leaf's gate travels inside its lane through the replay
    /// split's `Sync` slots before the leaf runs; exclusive access gives the
    /// same single-owner memo.
    pub struct LeafGate<'k> {
        /// The largest element count any planned role this gate admits has.
        bound: usize,
        /// The last scan, keyed by the token shape it was taken on.
        memo: Option<GateScan>,
        /// The call's prepared cost keys, shared by its split leaves.
        keys: Option<&'k GateKeys>,
        /// Whether this gate belongs to a watched call.
        #[cfg(test)]
        watched: bool,
    }

    #[derive(Clone, Copy)]
    struct GateScan {
        budget: core::num::NonZeroUsize,
        team: bool,
        min_elements: Option<usize>,
    }

    impl<'k> LeafGate<'k> {
        /// A gate for roles of at most `bound` elements.
        pub(crate) fn new(bound: usize) -> Self {
            Self {
                bound,
                memo: None,
                keys: None,
                #[cfg(test)]
                watched: gate_counts::watching(),
            }
        }

        /// A gate for roles of at most `keys`' bound that takes a team
        /// token's scan from `keys` when `keys` prepared its budget, running
        /// the scan there if no gate of the call has yet.
        pub(crate) fn shared(keys: &'k GateKeys) -> Self {
            Self {
                bound: keys.bound,
                memo: None,
                keys: Some(keys),
                #[cfg(test)]
                watched: keys.watched,
            }
        }

        /// The cheapest width of `exec` above one and its dispatch cost.
        fn cheapest(
            exec: &crate::host_pool::Exec<'_>,
        ) -> Option<(core::num::NonZeroUsize, core::time::Duration)> {
            let mut cheapest = None;
            let mut best = core::time::Duration::MAX;
            for width in 2..=exec.budget().get() {
                let Some(width) = core::num::NonZeroUsize::new(width) else {
                    continue;
                };
                let cost = exec.dispatch_cost(width);
                if cost < best {
                    best = cost;
                    cheapest = Some((width, cost));
                }
            }
            cheapest
        }

        /// The smallest element count up to `bound` whose copy estimate pays
        /// at the token's cheapest width, or `None` when none does. A copy
        /// estimate is monotone in the element count, so a binary search over
        /// `pays` at that one width finds it in O(log bound).
        fn scan(&self, exec: &crate::host_pool::Exec<'_>) -> Option<usize> {
            let bound = self.bound;
            #[cfg(test)]
            gate_counts::add(self.watched, |counts| counts.scans += 1);
            let (cheapest, _) = Self::cheapest(exec)?;
            let class = execution::KernelClass::Copy;
            let pays = |elements: usize| {
                #[cfg(test)]
                gate_counts::add(self.watched, |counts| counts.pays += 1);
                exec.pays(class.estimate(elements), 1, cheapest)
            };
            if !pays(bound) {
                return None;
            }
            if pays(0) {
                return Some(0);
            }
            let (mut below, mut at) = (0usize, bound);
            while at - below > 1 {
                let middle = below + (at - below) / 2;
                if pays(middle) {
                    at = middle;
                } else {
                    below = middle;
                }
            }
            Some(at)
        }

        /// Exactly `parallel_width(exec, Copy, elements) > 1` for
        /// `elements <= bound`: a block below the paying size is rejected
        /// without a width search.
        ///
        /// Why the memo is keyed by budget and team authority: one gate
        /// serves every unsplit block list of a call on the calling thread,
        /// whose tokens all come from one engine, so those two fields are
        /// what can still differ between its tokens; the calling thread fixes
        /// the remaining dispatch kind. A forked leaf gets a gate of its own.
        #[inline]
        pub(crate) fn widens(
            &mut self,
            exec: &crate::host_pool::Exec<'_>,
            elements: usize,
        ) -> bool {
            if exec.is_serial() {
                return false;
            }
            let (budget, team) = (exec.budget(), exec.carries_team());
            let min_elements = match self.memo {
                Some(scan) if scan.budget == budget && scan.team == team => scan.min_elements,
                _ => {
                    let shared = self
                        .keys
                        .filter(|_| team)
                        .and_then(|keys| keys.slot(budget.get()));
                    let min_elements = match shared {
                        Some((cost, scan)) => {
                            debug_assert_eq!(
                                Self::cheapest(exec).map(|(_, cost)| cost),
                                Some(*cost),
                                "a prepared gate key differs from its token's"
                            );
                            *scan.get_or_init(|| self.scan(exec))
                        }
                        None => self.scan(exec),
                    };
                    self.memo = Some(GateScan {
                        budget,
                        team,
                        min_elements,
                    });
                    min_elements
                }
            };
            match min_elements {
                Some(min) if elements >= min => {
                    #[cfg(test)]
                    gate_counts::add(self.watched, |counts| counts.widths += 1);
                    execution::parallel_width(exec, execution::KernelClass::Copy, elements).get()
                        > 1
                }
                _ => false,
            }
        }
    }

    /// The gate cost keys of one replay call, prepared by its caller before
    /// any split: for each budget a split leaf of the call's root token can
    /// get, the dispatch cost of its cheapest width, and one lazily filled
    /// scan per distinct cost.
    ///
    /// Why the caller can prepare them: a team token's dispatch cost is read
    /// from its team's cost row whatever thread runs it, and a fork, a
    /// reborrow and a limit all keep the team authority (threadteam
    /// `Exec::dispatch_cost`, `fork`, `limit`), so a leaf of budget `b` has
    /// the cheapest cost the caller reads from its own token limited to `b`.
    /// A scan depends only on the bound and that cost, so leaves with equal
    /// costs share it; `OnceLock` runs it at most once and only when a leaf
    /// asks, so a call whose leaves admit nothing searches nothing.
    #[derive(Debug, Default)]
    pub(crate) struct GateKeys {
        bound: usize,
        /// Each prepared budget and the index of its cost's slot.
        budgets: Vec<(usize, usize)>,
        slots: Vec<(core::time::Duration, std::sync::OnceLock<Option<usize>>)>,
        /// Whether the call that prepared the keys is watched.
        #[cfg(test)]
        watched: bool,
    }

    /// Scratch only: a clone starts empty.
    impl Clone for GateKeys {
        fn clone(&self) -> Self {
            Self::default()
        }
    }

    impl GateKeys {
        /// Prepares the keys of `exec`'s halving budgets for gates of
        /// `bound`, or none when no role is planned (`bound` zero) or
        /// `exec` carries no team.
        pub(crate) fn prepare(&mut self, exec: &mut crate::host_pool::Exec<'_>, bound: usize) {
            self.bound = bound;
            self.budgets.clear();
            self.slots.clear();
            if bound == 0 || exec.is_serial() || !exec.carries_team() {
                return;
            }
            // Split halves a token as threadteam's even fork does, so the
            // budgets at depth d are the floor and ceil of `root / 2^d`.
            let root = exec.budget().get();
            let mut divisor = 1usize;
            while root.div_ceil(divisor) >= 2 {
                for budget in [root / divisor, root.div_ceil(divisor)] {
                    if budget < 2 || self.budgets.iter().any(|&(b, _)| b == budget) {
                        continue;
                    }
                    let limited = exec
                        .reborrow()
                        .limit(core::num::NonZeroUsize::new(budget).expect("budget >= 2"));
                    let Some((_, cost)) = LeafGate::cheapest(&limited) else {
                        continue;
                    };
                    let slot = match self.slots.iter().position(|&(c, _)| c == cost) {
                        Some(slot) => slot,
                        None => {
                            self.slots.push((cost, std::sync::OnceLock::new()));
                            self.slots.len() - 1
                        }
                    };
                    self.budgets.push((budget, slot));
                }
                let Some(next) = divisor.checked_mul(2) else {
                    break;
                };
                divisor = next;
            }
        }

        /// Marks the keys as `gate`'s call's, for the gates shared from them.
        #[cfg(test)]
        pub(crate) fn watch_as(&mut self, gate: &LeafGate<'_>) {
            self.watched = gate.watched;
        }

        /// The cost and scan slot of a prepared `budget`.
        fn slot(
            &self,
            budget: usize,
        ) -> Option<&(core::time::Duration, std::sync::OnceLock<Option<usize>>)> {
            let &(_, slot) = self.budgets.iter().find(|&&(b, _)| b == budget)?;
            self.slots.get(slot)
        }

        /// Bytes of the tables' capacity.
        pub(crate) fn retained_bytes(&self) -> usize {
            self.budgets
                .capacity()
                .saturating_mul(core::mem::size_of::<(usize, usize)>())
                .saturating_add(self.slots.capacity().saturating_mul(core::mem::size_of::<(
                    core::time::Duration,
                    std::sync::OnceLock<Option<usize>>,
                )>()))
        }
    }

    /// Unnameable outside this crate, so its methods are crate-internal.
    ///
    /// Compiled replay calls these for an overwrite whose role's plan its
    /// leaf gate admitted: an adapter that replays plans runs it across
    /// `exec`; the default returns `None`, leaving the block to the public
    /// primitive's serial walk.
    pub trait Sealed {
        /// A Single block's overwrite `dst = scale * op(src)` by the role's
        /// admitted `plan`. Replay never plans an accumulate: the plan's
        /// `alpha * f(x) + beta * y` multiplies `f(x)` by a payload-typed one,
        /// which is not the exact `y + f(x)` of `transform_strided_baked`.
        ///
        /// Why a hook rather than a branch in `transform_strided_baked`: a
        /// threaded structural scale shares its coefficient across workers,
        /// which needs `C: Sync`, and that public method also serves fusion
        /// pack and scatter, whose coefficient type carries no such bound and
        /// whose layouts carry no plan.
        #[allow(clippy::too_many_arguments)]
        fn planned_transform<T, C>(
            &mut self,
            _exec: crate::host_pool::Exec<'_>,
            _plan: &UpdatePlan,
            _role: BakedFusedLayout<'_>,
            _dst_data: &mut [T],
            _src_data: &[T],
            _dst_offset: isize,
            _src_offset: isize,
            _source_conjugate: bool,
            _scale: TransformScale<T, C>,
        ) -> Option<Result<(), OperationError>>
        where
            C: Copy + Sync,
            T: Copy
                + Send
                + Sync
                + Add<T, Output = T>
                + Mul<T, Output = T>
                + PartialEq
                + Zero
                + One
                + ConjugateValue
                + RecouplingCoefficientAction<C>,
        {
            None
        }

        /// A recoupling scatter column's overwrite `dst = alpha * packed` by
        /// the role's admitted `plan`, with the identity and zero arms of
        /// `copy_scale_strided_baked`.
        #[allow(clippy::too_many_arguments)]
        fn planned_scatter<T>(
            &mut self,
            _exec: crate::host_pool::Exec<'_>,
            _plan: &UpdatePlan,
            _role: BakedFusedLayout<'_>,
            _dst_data: &mut [T],
            _packed: &[T],
            _dst_offset: isize,
            _packed_offset: isize,
            _alpha: T,
        ) -> Option<Result<(), OperationError>>
        where
            T: Copy
                + Send
                + Sync
                + Add<T, Output = T>
                + Mul<T, Output = T>
                + PartialEq
                + Zero
                + One
                + ConjugateValue,
        {
            None
        }
    }

    /// Test observation of the gates of one watched call, on every thread:
    /// a forked leaf's gate runs on a worker.
    ///
    /// Why a flag carried by the gate rather than a key matched at the count:
    /// a size key such as the gate's bound is shared by any concurrent test
    /// whose planned roles reach the same size, and a thread-local key misses
    /// the forked leaves that run on pool workers. A gate built on the
    /// watching thread is marked watched, and the call's shared gate keys
    /// take that mark from the call's gate, so every gate of the watched call
    /// counts wherever it runs and no gate of another call does.
    #[cfg(test)]
    pub(crate) mod gate_counts {
        use std::cell::Cell;
        use std::sync::{Mutex, MutexGuard, PoisonError};

        /// What the watched gates did.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub(crate) struct Counts {
            /// Dispatch-cost scans.
            pub scans: usize,
            /// `pays` queries of those scans.
            pub pays: usize,
            /// Width searches of admitted roles.
            pub widths: usize,
        }

        static COUNTS: Mutex<Counts> = Mutex::new(Counts {
            scans: 0,
            pays: 0,
            widths: 0,
        });
        static WATCHERS: Mutex<()> = Mutex::new(());

        std::thread_local! {
            static WATCHING: Cell<bool> = const { Cell::new(false) };
        }

        /// Whether a gate built on this thread belongs to a watched call.
        pub(super) fn watching() -> bool {
            WATCHING.with(Cell::get)
        }

        pub(super) fn add(watched: bool, count: impl FnOnce(&mut Counts)) {
            if watched {
                count(&mut COUNTS.lock().unwrap_or_else(PoisonError::into_inner));
            }
        }

        /// Counts the gates built on this thread until the watch drops; one
        /// watch at a time.
        pub(crate) struct Watch {
            _one_at_a_time: MutexGuard<'static, ()>,
        }

        pub(crate) fn watch() -> Watch {
            let guard = WATCHERS.lock().unwrap_or_else(PoisonError::into_inner);
            *COUNTS.lock().unwrap_or_else(PoisonError::into_inner) = Counts::default();
            WATCHING.with(|watching| watching.set(true));
            Watch {
                _one_at_a_time: guard,
            }
        }

        impl Watch {
            /// The counts so far, reset to zero.
            pub(crate) fn take(&self) -> Counts {
                std::mem::take(&mut *COUNTS.lock().unwrap_or_else(PoisonError::into_inner))
            }
        }

        impl Drop for Watch {
            fn drop(&mut self) {
                WATCHING.with(|watching| watching.set(false));
            }
        }
    }
}

/// Backend-neutral low-level kernel adapter for host-slice replay.
///
/// Replay drivers (tree-transform pack/recoupling/scatter, fusion-block
/// pack/scatter/scale) call these primitives instead of concrete kernel
/// functions, so the low-level execution backend (scalar loops, strided-rs,
/// BLAS, future C++ kernels) is replaceable behind one boundary.
///
/// The data contract is host slices. Device replay needs a separate
/// storage-aware adapter; device storage must not be hidden behind this trait.
///
/// View contract: a caller may pass any strided views that address the same
/// set of (destination, source) element pairs as the blocks they stand for,
/// in any traversal order. Compiled
/// tree-transform replay passes its compiled normalized roles as `shape` and
/// strides — extent-1 axes dropped, axes reordered, contiguous runs fused, a
/// zero extent as `[0]` and rank 0 as `[1]` — so their rank is at least 1 and
/// may differ from the tensor rank. Inactive-destination `scale_strided` and
/// zero fills pass the one-sided normalized view of the destination block.
/// Every primitive is elementwise without reduction, so an adapter that
/// iterates the views it receives produces the same values.
///
/// Token contract: `exec` is the caller's share of the engine token. Every
/// primitive here is one serial walk; compiled replay widens an overwrite of
/// a Single block or scatter column onto the role's retained Strided plan
/// before calling them, when Strided's own copy cost rule admits it.
///
/// Bounds contract: the `*_baked` methods validate only ranks. The caller
/// guarantees that every offset reachable through the views (or through
/// `baked`) lies inside `dst_data` and `src_data`; replay drivers derive them
/// from validated block descriptors. Callers holding unproven layouts use
/// [`StridedHostKernelAdapter::tensoradd_strided_checked`].
///
/// # Panics
///
/// The `*_baked` methods panic on bounds-checked slice indexing, possibly
/// after partial writes, when the bounds contract is violated; a violation
/// is never undefined behavior.
///
/// Sealed: the `testing` feature adds a method, so an implementor outside this
/// crate could not compile in both feature configurations.
pub trait HostKernelAdapter<T>: sealed::Sealed {
    /// `dst = alpha * op(src) + beta * dst` over strided views, where `op` is
    /// conjugation when `source_conjugate` is set (tensoradd / single-block
    /// tree replay primitive).
    ///
    /// `baked` is an optional prebaked fused layout (issue #232): a pure
    /// function of the (block, role) stride pair, so it is dtype-independent
    /// and correctness-neutral versus recomputation; an adapter that does not
    /// fuse may ignore it. `index` is caller-owned traversal scratch (compiled
    /// replay keeps it in its execution workspace); `None` lets the adapter
    /// use its own.
    ///
    /// # Panics
    ///
    /// On a bounds-contract violation (see the trait docs).
    #[allow(clippy::too_many_arguments)]
    fn add_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>;

    /// `dst = alpha * src + beta * dst` over strided views without
    /// conjugation (scatter primitive). `baked` and `index` as in
    /// [`add_strided_baked`](Self::add_strided_baked).
    ///
    /// # Panics
    ///
    /// On a bounds-contract violation (see the trait docs).
    #[allow(clippy::too_many_arguments)]
    fn axpby_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>;

    /// `dst = alpha * op(src)` over strided views (pack primitive). `baked`
    /// and `index` as in [`add_strided_baked`](Self::add_strided_baked).
    ///
    /// # Panics
    ///
    /// On a bounds-contract violation (see the trait docs).
    #[allow(clippy::too_many_arguments)]
    fn copy_scale_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>;

    /// `dst = scale * op(src) + beta * dst` over one tree-transform block,
    /// with the structural coefficient still in its own type.
    ///
    /// Why this exists next to [`add_strided_baked`](Self::add_strided_baked):
    /// those take the scale already promoted to the payload type `T`, which
    /// turns a real coefficient into a complex multiply. `beta` is `None` for
    /// an overwrite (the destination is never read) and `Some` for an
    /// accumulate. The default promotes and forwards, so adapters that cannot
    /// exploit a real coefficient keep their current behavior.
    ///
    /// # Panics
    ///
    /// On a bounds-contract violation (see the trait docs).
    #[allow(clippy::too_many_arguments)]
    fn transform_strided_baked<C>(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        scale: TransformScale<T, C>,
        beta: Option<T>,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C> + One + PartialEq,
    {
        let alpha = scale.into_data();
        match beta {
            Some(beta) => self.add_strided_baked(
                exec,
                zero_strides,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                source_conjugate,
                alpha,
                beta,
                baked,
                index,
            ),
            None => self.copy_scale_strided_baked(
                exec,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                source_conjugate,
                alpha,
                baked,
                index,
            ),
        }
    }

    /// `dst = beta * dst` over a strided block (inactive-block scale
    /// primitive).
    fn scale_strided(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError>;

    /// `destination[:, dst] = Σ_src coefficient[dst, src] * source[:, src]`
    /// over packed tree columns.
    ///
    /// TensorKit's dense-vector GenericTreeTransformer uses `U[dst, src]` and
    /// computes `buffer_dst = buffer_src * transpose(U)` after packing source
    /// trees as columns. This is the BLAS/GEMM replacement point for the
    /// recoupling matrix application.
    #[cfg(any(test, feature = "testing"))]
    #[allow(clippy::too_many_arguments)]
    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>;
}

/// Default host kernel adapter backed by the strided-rs style raw kernels.
///
/// Production recoupling is a grouped dense-executor GEMM that does not go
/// through this adapter; the scalar-loop `recoupling_src_times_u_transpose`
/// exists only under `cfg(test)` / the `testing` feature as the reference
/// oracle.
///
/// Why not `Copy` or externally constructible: direct/unbaked calls retain
/// mutable normalization and traversal scratch. Compiled replay supplies its
/// traversal indices from the execution workspace; clones preserve
/// configuration but never share either scratch source.
#[derive(Debug, Default)]
pub struct StridedHostKernelAdapter {
    scratch: StridedKernelScratch,
}

impl Clone for StridedHostKernelAdapter {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl StridedHostKernelAdapter {
    /// `dst = alpha * op(src) + beta * dst` over one block whose layout the
    /// caller has not proved in bounds (degeneracy restriction and scatter).
    ///
    /// Both reachable extents are checked once. A destination whose axes are
    /// separated then runs Strided's raw kernels or the same fused span walk
    /// as tree-transform replay, with no per-element offset check; any other
    /// destination runs a column-major loop over the original axes with
    /// checked offsets, so its overwrite and accumulation order stay
    /// defined. The element arithmetic is the scalar kernels' action, so
    /// `alpha = 1, beta = 0` stays a bit-exact copy rather than `1 * src`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensoradd_strided_checked<T>(
        &mut self,
        mut exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + Send
            + Sync,
    {
        validate_raw_strided_bounds(dst_data.len(), shape, dst_strides, dst_offset)?;
        validate_raw_strided_bounds(src_data.len(), shape, src_strides, src_offset)?;
        #[cfg(any(test, feature = "testing"))]
        {
            crate::probe::add(crate::probe::Event::LayoutPass, 1);
            crate::probe::add(crate::probe::Event::SpanWalk, 1);
        }
        let action = raw_strided_action(alpha, beta);
        let StridedKernelScratch { layout, index } = &mut self.scratch;
        combine_fused_layout(
            exec.reborrow(),
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            action,
            layout,
            index,
        )
    }

    /// [`span`] under `update` over `baked`, or over `shape` normalized into
    /// this adapter's scratch, with the caller's traversal `index` or this
    /// adapter's own. Each caller passes a closure of its own macro arm, so
    /// every instance has one call site (see the movement module).
    #[allow(clippy::too_many_arguments)]
    fn walk<T, F>(
        &mut self,
        index: Option<&mut [usize]>,
        baked: Option<BakedFusedLayout<'_>>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        update: Update<T>,
        map: F,
    ) -> Result<(), OperationError>
    where
        T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
        F: Fn(T) -> T,
    {
        let StridedKernelScratch {
            layout: scratch,
            index: own_index,
        } = &mut self.scratch;
        let layout = match baked {
            Some(baked) => baked,
            None => {
                normalize_fused_layout(shape, dst_strides, src_strides, scratch)?;
                BakedFusedLayout::from_compiled_normalized_slices(
                    scratch.dims(),
                    scratch.dst_strides(),
                    scratch.src_strides(),
                    None,
                )
            }
        };
        let rank = layout.dims().len();
        let index = match index {
            Some(index) => {
                let Some(index) = index.get_mut(..rank) else {
                    return Err(OperationError::InvalidArgument {
                        message: "fused traversal scratch is shorter than the normalized rank",
                    });
                };
                index
            }
            None => {
                own_index.resize(rank, 0);
                own_index.as_mut_slice()
            }
        };
        let (dst, src) = (dst_data, src_data);
        match update {
            Update::Overwrite => {
                span::<Init, _, T>(dst, src, layout, dst_offset, src_offset, index, &Write(map));
            }
            Update::Accumulate => span::<Init, _, T>(
                dst,
                src,
                layout,
                dst_offset,
                src_offset,
                index,
                &Accumulate(map),
            ),
            Update::Axpby(beta) => span::<Init, _, T>(
                dst,
                src,
                layout,
                dst_offset,
                src_offset,
                index,
                &Axpby { beta, map },
            ),
        }
        Ok(())
    }
}

impl sealed::Sealed for StridedHostKernelAdapter {
    #[allow(clippy::too_many_arguments)]
    fn planned_transform<T, C>(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        plan: &UpdatePlan,
        role: BakedFusedLayout<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        scale: TransformScale<T, C>,
    ) -> Option<Result<(), OperationError>>
    where
        C: Copy + Sync,
        T: Copy
            + Send
            + Sync
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + RecouplingCoefficientAction<C>,
    {
        Some(with_transform_map!(T, scale, source_conjugate, |map| {
            planned_overwrite(
                plan, role, dst_data, src_data, dst_offset, src_offset, exec, map,
            )
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn planned_scatter<T>(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        plan: &UpdatePlan,
        role: BakedFusedLayout<'_>,
        dst_data: &mut [T],
        packed: &[T],
        dst_offset: isize,
        packed_offset: isize,
        alpha: T,
    ) -> Option<Result<(), OperationError>>
    where
        T: Copy
            + Send
            + Sync
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue,
    {
        Some(with_payload_map!(T, alpha, false, |map| {
            planned_overwrite(
                plan,
                role,
                dst_data,
                packed,
                dst_offset,
                packed_offset,
                exec,
                map,
            )
        }))
    }
}

impl<T> HostKernelAdapter<T> for StridedHostKernelAdapter
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    fn add_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        validate_strided_ranks(shape, dst_strides, src_strides)?;
        if beta.is_zero() || beta.is_one() {
            let update = if beta.is_zero() {
                Update::Overwrite
            } else {
                Update::Accumulate
            };
            with_payload_map!(T, alpha, source_conjugate, |map| self.walk(
                index,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                update,
                map,
            ))?;
            zero_strides.clear();
            return Ok(());
        }
        tensoradd_raw_strided_kernel_trusted(
            exec,
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
        )
    }

    fn axpby_strided_baked(
        &mut self,
        exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: T,
        beta: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        validate_strided_ranks(shape, dst_strides, src_strides)?;
        if beta.is_zero() || beta.is_one() {
            let update = if beta.is_zero() {
                Update::Overwrite
            } else {
                Update::Accumulate
            };
            return with_payload_map!(T, alpha, false, |map| self.walk(
                index,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                update,
                map,
            ));
        }
        if baked.is_some() {
            // Why `scale_value` rather than `with_payload_map!`: this is the
            // raw combine loop's `Axpby` arithmetic, which scales even at
            // `alpha = 1`, so a compiled call and an unbaked one agree.
            return self.walk(
                index,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                Update::Axpby(beta),
                move |value: T| crate::scalar::scale_value(value, alpha),
            );
        }
        axpby_raw_strided_kernel_trusted(
            exec,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        )
    }

    fn copy_scale_strided_baked(
        &mut self,
        _exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        validate_strided_ranks(shape, dst_strides, src_strides)?;
        with_payload_map!(T, alpha, source_conjugate, |map| self.walk(
            index,
            baked,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            Update::Overwrite,
            map,
        ))
    }

    /// The structural coefficient stays in its own type; see
    /// `with_transform_map!`.
    #[allow(clippy::too_many_arguments)]
    fn transform_strided_baked<C>(
        &mut self,
        _exec: crate::host_pool::Exec<'_>,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [T],
        src_data: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        scale: TransformScale<T, C>,
        beta: Option<T>,
        baked: Option<BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C> + One + PartialEq,
    {
        // Why no closure around the body: it would be a separate function
        // holding the whole walk, called once per role.
        if let Err(error) = validate_strided_ranks(shape, dst_strides, src_strides) {
            zero_strides.clear();
            return Err(error);
        }
        let update = match beta {
            None => Update::Overwrite,
            Some(beta) if beta.is_zero() => Update::Overwrite,
            Some(beta) if beta.is_one() => Update::Accumulate,
            Some(beta) => Update::Axpby(beta),
        };
        let result = with_transform_map!(T, scale, source_conjugate, |map| self.walk(
            index,
            baked,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            update,
            map,
        ));
        zero_strides.clear();
        result
    }

    fn scale_strided(
        &mut self,
        _exec: crate::host_pool::Exec<'_>,
        dst_data: &mut [T],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: T,
    ) -> Result<(), OperationError> {
        scale_raw_strided_kernel_trusted(dst_data, shape, dst_strides, dst_offset, beta)
    }

    #[cfg(any(test, feature = "testing"))]
    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        _exec: crate::host_pool::Exec<'_>,
        destination: &mut [T],
        source: &[T],
        recoupling_coefficients_dst_src: &[C],
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        T: RecouplingCoefficientAction<C>,
    {
        validate_recoupling_lens(
            destination.len(),
            source.len(),
            recoupling_coefficients_dst_src.len(),
            element_count,
            src_count,
            dst_count,
        )?;
        for dst_index in 0..dst_count {
            let dst_column_start = dst_index * element_count;
            let coefficient_row_start = dst_index * src_count;
            for element in 0..element_count {
                let mut sum = T::zero();
                for src_index in 0..src_count {
                    let coeff = recoupling_coefficients_dst_src[coefficient_row_start + src_index];
                    let src_value = source[element + src_index * element_count];
                    sum = sum + src_value.scale_by_coefficient(coeff);
                }
                destination[dst_column_start + element] = sum;
            }
        }
        Ok(())
    }
}

/// Shared dimension validation for recoupling matrix application.
///
/// All adapter implementations should validate against the same packed-column
/// layout before touching data.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn validate_recoupling_lens(
    destination_len: usize,
    source_len: usize,
    coefficient_len: usize,
    element_count: usize,
    src_count: usize,
    dst_count: usize,
) -> Result<(), OperationError> {
    let expected_source_len = element_count
        .checked_mul(src_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let expected_destination_len = element_count
        .checked_mul(dst_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let coefficient_count = src_count
        .checked_mul(dst_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;

    if source_len != expected_source_len {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_source_len,
            actual: source_len,
        });
    }
    if destination_len != expected_destination_len {
        return Err(OperationError::ElementCountMismatch {
            expected: expected_destination_len,
            actual: destination_len,
        });
    }
    if coefficient_len < coefficient_count {
        return Err(OperationError::CoefficientCountMismatch {
            expected: coefficient_count,
            actual: coefficient_len,
        });
    }
    Ok(())
}
