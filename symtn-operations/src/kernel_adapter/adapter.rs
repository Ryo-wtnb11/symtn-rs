use super::*;
use strided_kernel::UpdatePlan;

pub(crate) mod sealed {
    use super::*;
    use strided_kernel::execution;

    /// The smallest element counts that widen and that fill a token shape,
    /// found once for the first shape a gate is asked about (a call's whole
    /// token, which it keeps) and once for each change of a second,
    /// replaceable shape (a narrowed lane).
    ///
    /// Why: Strided's `parallel_width` scans every width up to the budget on
    /// each call, so asking it for every block (or every component a threaded
    /// replay classifies) costs O(blocks x budget) and makes wide engines
    /// slower on small tensors. Inside the token's exact domain (see
    /// [`LeafGate::exact_domain`]) both answers are monotone in the element
    /// count, so one threshold each answers every query there in O(1) once
    /// the shape is scanned; a count outside it asks Strided directly.
    ///
    /// Why lazy: the scan runs on the first query of a token shape: a planned
    /// overwrite reaching leaf admission, a threaded replay classifying its
    /// components, or a threaded chunk scatter deciding whether its moved
    /// elements pay a second worker (the last also for accumulating or
    /// planless replays). A serial replay of planless or accumulating blocks
    /// asks nothing. A gate lives on the thread that runs its blocks, because
    /// a token's dispatch cost depends on the calling thread: one per call
    /// for the calling thread's unsplit lists, one per forked leaf otherwise.
    pub struct LeafGate {
        /// The largest element count any planned role this gate admits has.
        bound: usize,
        /// Scans keyed by the token shape they were taken on: the first
        /// shape asked (the call's whole token) keeps slot 0; any later shape
        /// (a lane narrowed to its component's width) takes slot 1, so
        /// narrowing never evicts the whole token's scan.
        memo: [core::cell::Cell<Option<GateScan>>; 2],
    }

    /// One token shape's thresholds: plain `Copy` data, so a scan taken on
    /// the calling thread can seed a gate on a worker ([`LeafGate::seeded`]).
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct GateScan {
        budget: core::num::NonZeroUsize,
        team: bool,
        /// The smallest count that widens past one worker.
        min_elements: Option<usize>,
        /// The smallest count that widens to the whole budget.
        min_full: Option<usize>,
        /// The largest count of the token's exact domain, if any.
        exact: Option<usize>,
    }

    impl LeafGate {
        /// A gate for roles of at most `bound` elements.
        pub(crate) fn new(bound: usize) -> Self {
            Self {
                bound,
                memo: [core::cell::Cell::new(None), core::cell::Cell::new(None)],
            }
        }

        /// A gate for roles of at most `bound` elements that already holds
        /// `scan`, taken by a gate of the same `bound` ([`Self::scan_for`]).
        /// It answers a query on `scan`'s token shape without scanning; on
        /// any other shape (another budget or team authority) the memo check
        /// misses and it scans as a fresh gate would, so a seed changes the
        /// cost of an answer, never the answer.
        ///
        /// Why a seed rather than a shared gate: a gate is `!Sync`, while the
        /// scan is plain data a worker can copy.
        pub(crate) fn seeded(bound: usize, scan: GateScan) -> Self {
            Self {
                bound,
                memo: [
                    core::cell::Cell::new(Some(scan)),
                    core::cell::Cell::new(None),
                ],
            }
        }

        /// The scan of `exec`'s token shape: this gate's memo of it, or a
        /// scan taken now (and kept). `exec` is not serial.
        pub(crate) fn scan_for(&self, exec: &crate::host_pool::Exec<'_>) -> GateScan {
            self.thresholds(exec)
        }

        /// Whether `scan` was taken on `exec`'s token shape, so a gate seeded
        /// with it answers `exec` from its memo.
        pub(crate) fn scan_matches(scan: &GateScan, exec: &crate::host_pool::Exec<'_>) -> bool {
            scan.budget == exec.budget() && scan.team == exec.carries_team()
        }

        /// The smallest element count up to `bound` (inside the token's exact
        /// domain) at which Strided's `parallel_width` for a copy exceeds
        /// one, or `None` when `bound` does not.
        ///
        /// Why a bisection is exact: in real arithmetic, the copy estimate is monotone in the
        /// element count, and `parallel_width > 1` holds exactly when some
        /// width `k >= 2` both pays (`estimate >= PAYS * D(k)`) and beats one
        /// worker (`D(k) + estimate / k < estimate`); each condition only
        /// gets easier as the estimate grows, so their union is monotone.
        /// Free dispatch (`D = 0`) still needs a nonzero estimate to beat one
        /// worker, so an empty or zero-cost copy never widens. A width pays
        /// only if the cheapest dispatch cost pays, so the first count that
        /// pays there bounds the search from below in O(budget + log bound),
        /// and the bisection above it asks Strided O(log bound) times.
        fn scan_widen(exec: &crate::host_pool::Exec<'_>, bound: usize) -> Option<usize> {
            #[cfg(test)]
            GATE_SCANS.with(|scans| scans.set(scans.get() + 1));
            let mut cheapest = None;
            let mut best = core::time::Duration::MAX;
            for width in 2..=exec.budget().get() {
                let Some(width) = core::num::NonZeroUsize::new(width) else {
                    continue;
                };
                let cost = exec.dispatch_cost(width);
                if cost < best {
                    best = cost;
                    cheapest = Some(width);
                }
            }
            let cheapest = cheapest?;
            let class = execution::KernelClass::Copy;
            let pays = |elements: usize| exec.pays(class.estimate(elements), 1, cheapest);
            let widens = |elements: usize| {
                #[cfg(test)]
                WIDTH_QUERIES.with(|queries| queries.set(queries.get() + 1));
                execution::parallel_width(exec, class, elements).get() > 1
            };
            if !pays(bound) || !widens(bound) {
                return None;
            }
            // The first paying count: no smaller count widens.
            let first_paying = if pays(0) {
                0
            } else {
                let (mut below, mut at) = (0usize, bound);
                while at - below > 1 {
                    let middle = below + (at - below) / 2;
                    if pays(middle) {
                        at = middle;
                    } else {
                        below = middle;
                    }
                }
                at
            };
            if widens(first_paying) {
                return Some(first_paying);
            }
            let (mut below, mut at) = (first_paying, bound);
            while at - below > 1 {
                let middle = below + (at - below) / 2;
                if widens(middle) {
                    at = middle;
                } else {
                    below = middle;
                }
            }
            Some(at)
        }

        /// The smallest element count up to `bound` (inside the token's exact
        /// domain) at which Strided's `parallel_width` for a copy is the whole
        /// budget, at or above the widening count `from`; `None` when `bound`
        /// does not saturate.
        ///
        /// Why a bisection is exact: in real arithmetic, `parallel_width` picks the width `k`
        /// minimising `D(k) + estimate / k` among the widths that pay. As the
        /// estimate grows, a wider width's objective falls faster than a
        /// narrower one's (slope `1 / k`), so once the whole budget `B` beats
        /// every paying narrower width it keeps beating it; a width that only
        /// starts paying later has `D(k) > D(B)` (paying is monotone in the
        /// cost, and `B` already pays), so its objective
        /// `D(k) + estimate / k` exceeds `D(B) + estimate / B` for good.
        /// `parallel_width == B` is therefore monotone in the element count;
        /// [`Self::exact_domain`] carries this over to threadteam's rounded
        /// comparison.
        fn scan_full(
            exec: &crate::host_pool::Exec<'_>,
            bound: usize,
            from: usize,
        ) -> Option<usize> {
            let budget = exec.budget().get();
            let full = |elements: usize| {
                #[cfg(test)]
                WIDTH_QUERIES.with(|queries| queries.set(queries.get() + 1));
                execution::parallel_width(exec, execution::KernelClass::Copy, elements).get()
                    == budget
            };
            if !full(bound) {
                return None;
            }
            if full(from) {
                return Some(from);
            }
            let (mut below, mut at) = (from, bound);
            while at - below > 1 {
                let middle = below + (at - below) / 2;
                if full(middle) {
                    at = middle;
                } else {
                    below = middle;
                }
            }
            Some(at)
        }

        /// The largest element count whose width question the thresholds
        /// answer exactly on this token: `None` when there is none.
        ///
        /// Why a domain: threadteam compares the objectives
        /// `D(k) + estimate / k` as rounded `f64`s, and the monotonicity
        /// arguments of [`Self::scan_widen`] and [`Self::scan_full`] hold for
        /// real arithmetic. For integer nanoseconds `E` and `D(k)`, two
        /// different real objectives of widths `j, k <= B` differ by at least
        /// `1 / B^2` (both are integers plus multiples of `1 / k`, `1 / j`),
        /// and width one's objective `E` differs from another by at least
        /// `1 / B`. Each rounded objective is within `2^-52` of its real
        /// value relatively (one division, one addition, both operands exact
        /// below `2^53`). So while every objective stays below `2^51 / B^2`,
        /// that is `E + max D(k) < 2^51 / B^2`, a rounded comparison agrees
        /// with the real one except at a real tie, where either answer keeps
        /// the predicate a single step: the predicates are monotone there.
        /// Beyond it rounding can break a tie the other way at a larger
        /// estimate, so larger counts ask Strided. The copy estimate is
        /// monotone in the element count, so the domain is a count prefix,
        /// found by bisection on the estimate alone.
        fn exact_domain(exec: &crate::host_pool::Exec<'_>) -> Option<usize> {
            let budget = exec.budget().get() as u128;
            let max_cost = (2..=exec.budget().get())
                .filter_map(core::num::NonZeroUsize::new)
                .map(|width| exec.dispatch_cost(width))
                .filter(|&cost| cost != core::time::Duration::MAX)
                .map(|cost| cost.as_nanos())
                .max()
                .unwrap_or(0);
            let limit = ((1u128 << 51) / (budget * budget)).checked_sub(max_cost + 1)?;
            let estimate =
                |elements: usize| execution::KernelClass::Copy.estimate(elements).as_nanos();
            if estimate(0) > limit {
                return None;
            }
            if estimate(usize::MAX) <= limit {
                return Some(usize::MAX);
            }
            let (mut inside, mut outside) = (0usize, usize::MAX);
            while outside - inside > 1 {
                let middle = inside + (outside - inside) / 2;
                if estimate(middle) <= limit {
                    inside = middle;
                } else {
                    outside = middle;
                }
            }
            Some(inside)
        }

        /// The token shape's widening and saturating counts and exact domain,
        /// scanned once per shape.
        ///
        /// Why the memo is keyed by budget and team authority: one gate
        /// serves every unsplit block list of a call on the calling thread,
        /// whose tokens all come from one engine, so those two fields are
        /// what can still differ between its tokens; the calling thread fixes
        /// the remaining dispatch kind. The gate is not `Sync`, so a forked
        /// worker builds its own.
        fn thresholds(&self, exec: &crate::host_pool::Exec<'_>) -> GateScan {
            let (budget, team) = (exec.budget(), exec.carries_team());
            let hit = |slot: &core::cell::Cell<Option<GateScan>>| {
                slot.get()
                    .filter(|scan| scan.budget == budget && scan.team == team)
            };
            match hit(&self.memo[0]).or_else(|| hit(&self.memo[1])) {
                Some(scan) => scan,
                None => {
                    let exact = Self::exact_domain(exec);
                    let bound = exact.map(|exact| exact.min(self.bound));
                    let min_elements = bound.and_then(|bound| Self::scan_widen(exec, bound));
                    let min_full = bound
                        .zip(min_elements)
                        .and_then(|(bound, from)| Self::scan_full(exec, bound, from));
                    let scan = GateScan {
                        budget,
                        team,
                        min_elements,
                        min_full,
                        exact,
                    };
                    let slot = usize::from(self.memo[0].get().is_some());
                    self.memo[slot].set(Some(scan));
                    scan
                }
            }
        }

        /// `elements` against a threshold scanned up to `min(bound, exact)`:
        /// `Some` answer where the threshold decides it exactly (below the
        /// scanned range by the threshold, above it by monotonicity inside
        /// the exact domain), `None` where Strided must be asked.
        fn decide(
            &self,
            scan: GateScan,
            threshold: Option<usize>,
            elements: usize,
        ) -> Option<bool> {
            let exact = scan.exact?;
            if elements > exact {
                return None;
            }
            match threshold {
                Some(min) => Some(elements >= min),
                None if elements <= self.bound => Some(false),
                None => None,
            }
        }

        /// The gate's O(1) answer to `parallel_width > 1` and to
        /// `parallel_width == budget` where its thresholds decide them, each
        /// `None` where only Strided can answer.
        pub(crate) fn decided(
            &self,
            exec: &crate::host_pool::Exec<'_>,
            elements: usize,
        ) -> (Option<bool>, Option<bool>) {
            if exec.is_serial() {
                return (Some(false), Some(true));
            }
            let scan = self.thresholds(exec);
            (
                self.decide(scan, scan.min_elements, elements),
                self.decide(scan, scan.min_full, elements),
            )
        }

        /// The largest count of the token's exact domain, where Strided's
        /// copy width is monotone in the count (see
        /// [`Self::exact_domain`]); `None` when the domain is empty.
        pub(crate) fn exact_end(&self, exec: &crate::host_pool::Exec<'_>) -> Option<usize> {
            if exec.is_serial() {
                return Some(usize::MAX);
            }
            self.thresholds(exec).exact
        }

        /// Strided's own width for a copy of `elements` on `exec`: one full
        /// width search (O(budget)), counted for tests.
        pub(crate) fn width(exec: &crate::host_pool::Exec<'_>, elements: usize) -> usize {
            #[cfg(test)]
            WIDTH_QUERIES.with(|queries| queries.set(queries.get() + 1));
            execution::parallel_width(exec, execution::KernelClass::Copy, elements).get()
        }

        /// Exactly `parallel_width(exec, Copy, elements) > 1`: O(1) inside
        /// the token's exact domain once its threshold is known, Strided's
        /// own answer outside it.
        #[inline]
        pub(crate) fn widens(&self, exec: &crate::host_pool::Exec<'_>, elements: usize) -> bool {
            if exec.is_serial() {
                return false;
            }
            let scan = self.thresholds(exec);
            self.decide(scan, scan.min_elements, elements)
                .unwrap_or_else(|| Self::width(exec, elements) > 1)
        }

        /// Exactly `parallel_width(exec, Copy, elements) == budget`, whether
        /// a copy of `elements` uses the whole token (always on a serial
        /// one): O(1) inside the token's exact domain once its threshold is
        /// known, Strided's own answer outside it.
        #[inline]
        pub(crate) fn saturates(&self, exec: &crate::host_pool::Exec<'_>, elements: usize) -> bool {
            if exec.is_serial() {
                return true;
            }
            let scan = self.thresholds(exec);
            self.decide(scan, scan.min_full, elements)
                .unwrap_or_else(|| Self::width(exec, elements) == exec.budget().get())
        }
    }

    #[cfg(test)]
    thread_local! {
        /// Dispatch-cost scans this thread's gates ran (test observation).
        pub(crate) static GATE_SCANS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
        /// Strided width searches this thread's gates ran (test observation).
        pub(crate) static WIDTH_QUERIES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
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
                + One,
        {
            None
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
    /// Both reachable extents are checked once, then the block runs the same
    /// fused span walk as tree-transform replay, with no per-element offset
    /// check. The element arithmetic is the scalar kernels' action, so
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
        if raw_rank_within_strided_limit_routed(
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
        )? {
            return Ok(());
        }
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

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn fused_pair_baked_dispatch<T, Apply, ElementOp>(
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
        apply: Apply,
        op: ElementOp,
    ) -> Result<(), OperationError>
    where
        T: Copy,
        Apply: Fn(&mut T, T),
        ElementOp: Fn(T) -> T,
    {
        match index {
            Some(index) => fused_pair_baked_with_index(
                &mut self.scratch.layout,
                baked,
                index,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                apply,
                op,
            ),
            None => fused_pair_baked(
                &mut self.scratch,
                baked,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                apply,
                op,
            ),
        }
    }

    /// The structural-coefficient element op, chosen once per block.
    ///
    /// The three source forms are TensorKit's: no scale at all for `One()`, a
    /// componentwise `payload * sectorscalar` for a real coefficient, and a
    /// payload-type multiply only when `α` has already been folded in.
    #[allow(clippy::too_many_arguments)]
    fn transform_strided_baked_impl<T, C>(
        &mut self,
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
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + RecouplingCoefficientAction<C>
            + Send
            + Sync,
    {
        validate_strided_ranks(shape, dst_strides, src_strides)?;
        macro_rules! run {
            ($op:expr) => {{
                let op = $op;
                match beta {
                    None => self.fused_pair_baked_dispatch(
                        index,
                        baked,
                        dst_data,
                        src_data,
                        shape,
                        dst_strides,
                        src_strides,
                        dst_offset,
                        src_offset,
                        |dst: &mut T, value| *dst = value,
                        op,
                    ),
                    Some(beta) if beta.is_zero() => self.fused_pair_baked_dispatch(
                        index,
                        baked,
                        dst_data,
                        src_data,
                        shape,
                        dst_strides,
                        src_strides,
                        dst_offset,
                        src_offset,
                        |dst: &mut T, value| *dst = value,
                        op,
                    ),
                    Some(beta) if beta.is_one() => self.fused_pair_baked_dispatch(
                        index,
                        baked,
                        dst_data,
                        src_data,
                        shape,
                        dst_strides,
                        src_strides,
                        dst_offset,
                        src_offset,
                        |dst: &mut T, value| *dst = *dst + value,
                        op,
                    ),
                    Some(beta) => self.fused_pair_baked_dispatch(
                        index,
                        baked,
                        dst_data,
                        src_data,
                        shape,
                        dst_strides,
                        src_strides,
                        dst_offset,
                        src_offset,
                        move |dst: &mut T, value| *dst = beta * *dst + value,
                        op,
                    ),
                }
            }};
        }
        if scale.is_identity() {
            return run!(move |value: T| value.maybe_conj(source_conjugate));
        }
        // A zero `α * coeff` is VectorInterface's `zero(x) * α`, whatever
        // the source holds; see `TransformScale::scale`.
        if scale.is_zero() {
            let zero = scale.apply(T::zero());
            return run!(move |_: T| zero);
        }
        match scale {
            TransformScale::Structural(coefficient) => run!(move |value: T| value
                .maybe_conj(source_conjugate)
                .scale_by_coefficient(coefficient)),
            TransformScale::Data(alpha) => {
                run!(move |value: T| alpha * value.maybe_conj(source_conjugate))
            }
        }
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
        macro_rules! run {
            ($op:expr) => {
                planned_overwrite(
                    plan, role, dst_data, src_data, dst_offset, src_offset, exec, $op,
                )
            };
        }
        // The element ops of `transform_strided_baked_impl`, arm for arm.
        Some(if scale.is_identity() {
            run!(move |value: T| value.maybe_conj(source_conjugate))
        } else if scale.is_zero() {
            let zero = scale.apply(T::zero());
            run!(move |_: T| zero)
        } else {
            match scale {
                TransformScale::Structural(coefficient) => {
                    let coefficient = &coefficient;
                    run!(move |value: T| value
                        .maybe_conj(source_conjugate)
                        .scale_by_coefficient(*coefficient))
                }
                TransformScale::Data(alpha) => {
                    run!(move |value: T| alpha * value.maybe_conj(source_conjugate))
                }
            }
        })
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
        T: Copy + Send + Sync + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One,
    {
        macro_rules! run {
            ($op:expr) => {
                planned_overwrite(
                    plan,
                    role,
                    dst_data,
                    packed,
                    dst_offset,
                    packed_offset,
                    exec,
                    $op,
                )
            };
        }
        // The element ops of `copy_scale_strided_baked`, arm for arm.
        Some(if alpha.is_one() {
            run!(|value: T| value)
        } else if alpha.is_zero() {
            let zero = T::zero() * alpha;
            run!(move |_: T| zero)
        } else {
            run!(move |value: T| alpha * value)
        })
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
            let assign = beta.is_zero();
            macro_rules! run {
                ($op:expr) => {
                    self.fused_pair_baked_dispatch(
                        index,
                        baked,
                        dst_data,
                        src_data,
                        shape,
                        dst_strides,
                        src_strides,
                        dst_offset,
                        src_offset,
                        move |dst: &mut T, value| {
                            if assign {
                                *dst = value;
                            } else {
                                *dst = *dst + value;
                            }
                        },
                        $op,
                    )
                };
            }
            // Why the identity and zero arms: see `copy_scale_strided_baked`.
            if alpha.is_one() {
                run!(move |value: T| value.maybe_conj(source_conjugate))?;
            } else if alpha.is_zero() {
                let zero = T::zero() * alpha;
                run!(move |_: T| zero)?;
            } else {
                run!(move |value: T| alpha * value.maybe_conj(source_conjugate))?;
            }
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
            let assign = beta.is_zero();
            macro_rules! run {
                ($op:expr) => {
                    self.fused_pair_baked_dispatch(
                        index,
                        baked,
                        dst_data,
                        src_data,
                        shape,
                        dst_strides,
                        src_strides,
                        dst_offset,
                        src_offset,
                        move |dst: &mut T, value| {
                            if assign {
                                *dst = value;
                            } else {
                                *dst = *dst + value;
                            }
                        },
                        $op,
                    )
                };
            }
            // Why the identity and zero arms: see `copy_scale_strided_baked`.
            if alpha.is_one() {
                return run!(|value: T| value);
            }
            if alpha.is_zero() {
                let zero = T::zero() * alpha;
                return run!(move |_: T| zero);
            }
            return run!(move |value: T| alpha * value);
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
        macro_rules! run {
            ($op:expr) => {
                self.fused_pair_baked_dispatch(
                    index,
                    baked,
                    dst_data,
                    src_data,
                    shape,
                    dst_strides,
                    src_strides,
                    dst_offset,
                    src_offset,
                    |dst: &mut T, value| *dst = value,
                    $op,
                )
            };
        }
        // Why the identity arm: `1 * (inf + 0i)` is `inf + NaN i`, so a pack or
        // scatter at alpha = 1 must copy rather than multiply. Why the zero
        // arm: VectorInterface's `scale(x, 0) = zero(x) * 0`, never
        // `0 * inf = NaN`, so a zero scale does not read the source.
        if alpha.is_one() {
            return run!(move |value: T| value.maybe_conj(source_conjugate));
        }
        if alpha.is_zero() {
            let zero = T::zero() * alpha;
            return run!(move |_: T| zero);
        }
        run!(move |value: T| alpha * value.maybe_conj(source_conjugate))
    }

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
        let result = self.transform_strided_baked_impl(
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            scale,
            beta,
            baked,
            index,
        );
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
