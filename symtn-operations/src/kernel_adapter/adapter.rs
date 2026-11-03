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
    /// accumulating blocks pays nothing. A gate lives on the thread that runs
    /// its blocks, because a token's dispatch cost depends on the calling
    /// thread: one per call for the calling thread's unsplit lists, one per
    /// forked leaf otherwise.
    pub struct LeafGate {
        /// The largest element count any planned role this gate admits has.
        bound: usize,
        /// The last scan, keyed by the token shape it was taken on.
        memo: core::cell::Cell<Option<GateScan>>,
    }

    #[derive(Clone, Copy)]
    struct GateScan {
        budget: core::num::NonZeroUsize,
        team: bool,
        min_elements: Option<usize>,
    }

    impl LeafGate {
        /// A gate for roles of at most `bound` elements.
        pub(crate) fn new(bound: usize) -> Self {
            Self {
                bound,
                memo: core::cell::Cell::new(None),
            }
        }

        /// The smallest element count up to `bound` whose copy estimate pays
        /// at the token's cheapest width, or `None` when none does. A copy
        /// estimate is monotone in the element count, so a binary search over
        /// `pays` at that one width finds it in O(log bound).
        fn scan(exec: &crate::host_pool::Exec<'_>, bound: usize) -> Option<usize> {
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
        /// the remaining dispatch kind. The gate is not `Sync`, so a forked
        /// worker builds its own.
        #[inline]
        pub(crate) fn widens(&self, exec: &crate::host_pool::Exec<'_>, elements: usize) -> bool {
            if exec.is_serial() {
                return false;
            }
            let (budget, team) = (exec.budget(), exec.carries_team());
            let min_elements = match self.memo.get() {
                Some(scan) if scan.budget == budget && scan.team == team => scan.min_elements,
                _ => {
                    let min_elements = Self::scan(exec, self.bound);
                    self.memo.set(Some(GateScan {
                        budget,
                        team,
                        min_elements,
                    }));
                    min_elements
                }
            };
            match min_elements {
                Some(min) if elements >= min => {
                    execution::parallel_width(exec, execution::KernelClass::Copy, elements).get()
                        > 1
                }
                _ => false,
            }
        }
    }

    #[cfg(test)]
    thread_local! {
        /// Dispatch-cost scans this thread's gates ran (test observation).
        pub(crate) static GATE_SCANS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
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
