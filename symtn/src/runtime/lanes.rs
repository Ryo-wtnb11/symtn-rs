use super::*;

pub(crate) type CoefficientCtx<D, Key, C> = TensorContractFusionExecutionContext<
    D,
    Key,
    DenseTreeTransformOperations,
    DenseTreeTransformOperations,
    C,
>;
pub type Ctx<D, Key> = CoefficientCtx<D, Key, f64>;

mod coefficient_lane_private {
    pub trait Sealed<C> {}

    impl<T: crate::typed::ScalarOps> Sealed<f64> for T {}
    impl Sealed<num_complex::Complex64> for num_complex::Complex64 {}
}

/// Selects one of the three payload/coefficient pairs owned by a runtime.
///
/// This stays private to the crate: provider scalar compatibility is a static
/// execution detail, not another public context type.
pub(crate) trait MultiplicityFreeCoefficientLane<C: symtn_tensors::DenseBlockScalar>:
    ScalarOps + symtn_tensors::RecouplingCoefficientAction<C> + coefficient_lane_private::Sealed<C>
{
    fn lane(
        context: &mut TensorExecutionContext,
    ) -> Result<&mut CoefficientCtx<Self, RuleIdentity, C>, Error>;
}
/// The supported payload/coefficient execution contexts for one cache-key
/// namespace. Existing operations dispatch on the stored dtype to the
/// real-coefficient lanes.
///
/// The double-precision lanes are built with the runtime; the single-precision
/// ones are built on first use. Every admitted payload dtype needs its own
/// dense executors and contract workspace, so building all four eagerly would
/// charge every runtime for dtypes the program never names. `lane_config`
/// carries what [`Ctxs::with_config`] was given, plus the settings applied
/// afterwards, so a lane created later is configured exactly like an eager one.
pub struct Ctxs<Key: Clone + Eq + Hash + Send + Sync + 'static> {
    pub(crate) f64: Ctx<f64, Key>,
    pub(crate) c64: Ctx<Complex64, Key>,
    pub(crate) f32: Option<Box<Ctx<f32, Key>>>,
    pub(crate) c32: Option<Box<Ctx<Complex32, Key>>>,
    lane_config: LaneConfig,
}

/// What a deferred [`Ctxs`] lane needs to be built exactly like an eager one.
#[derive(Clone)]
struct LaneConfig {
    /// `None` for a `Default` [`Ctxs`], whose lanes own private env-driven
    /// pools rather than a runtime's shared CPU context.
    shared: Option<symtn_dense::SharedCpuContext>,
    recoupling_threads: Option<NonZeroUsize>,
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Default for Ctxs<Key> {
    fn default() -> Self {
        Self {
            f64: Ctx::default(),
            c64: Ctx::default(),
            f32: None,
            c32: None,
            lane_config: LaneConfig {
                shared: None,
                recoupling_threads: None,
            },
        }
    }
}

/// Builds one contraction/recoupling backend on the runtime's shared CPU
/// context.
pub(super) fn make_transform_ops(
    ctx: &symtn_dense::SharedCpuContext,
) -> Result<DenseTreeTransformOperations, Error> {
    Ok(DenseTreeTransformOperations::new(
        symtn_dense::DefaultDenseExecutor::with_shared_context(ctx),
    ))
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Ctxs<Key> {
    /// Builds the per-scalar contexts on the runtime's shared CPU context.
    pub(crate) fn with_config(ctx: &symtn_dense::SharedCpuContext) -> Result<Self, Error> {
        Ok(Self {
            f64:
                Ctx::with_parts(
                    symtn_tensors::TreeTransformExecutionContext::new(make_transform_ops(ctx)?),
                    make_transform_ops(ctx)?,
                    <DenseTreeTransformOperations as symtn_tensors::TensorContractBackend<
                        f64,
                        f64,
                    >>::Workspace::default(),
                ),
            c64: Ctx::with_parts(
                symtn_tensors::TreeTransformExecutionContext::new(make_transform_ops(ctx)?),
                make_transform_ops(ctx)?,
                <DenseTreeTransformOperations as symtn_tensors::TensorContractBackend<
                    Complex64,
                    f64,
                >>::Workspace::default(),
            ),
            f32: None,
            c32: None,
            lane_config: LaneConfig {
                shared: Some(ctx.clone()),
                recoupling_threads: None,
            },
        })
    }

    /// Builds one deferred lane the way [`Self::with_config`] built the eager
    /// ones, then replays the settings applied since.
    fn make_lane<D: ScalarOps>(config: &LaneConfig) -> Result<Box<Ctx<D, Key>>, Error> {
        let mut lane =
            match &config.shared {
                Some(ctx) => Ctx::with_parts(
                    symtn_tensors::TreeTransformExecutionContext::new(make_transform_ops(ctx)?),
                    make_transform_ops(ctx)?,
                    <DenseTreeTransformOperations as symtn_tensors::TensorContractBackend<
                        D,
                        f64,
                    >>::Workspace::default(),
                ),
                None => Ctx::default(),
            };
        if let Some(threads) = config.recoupling_threads {
            lane.tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
        Ok(Box::new(lane))
    }

    pub(crate) fn f32_lane(&mut self) -> Result<&mut Ctx<f32, Key>, Error> {
        let lane = match self.f32.take() {
            Some(lane) => lane,
            None => Self::make_lane(&self.lane_config)?,
        };
        Ok(self.f32.insert(lane))
    }

    pub(crate) fn c32_lane(&mut self) -> Result<&mut Ctx<Complex32, Key>, Error> {
        let lane = match self.c32.take() {
            Some(lane) => lane,
            None => Self::make_lane(&self.lane_config)?,
        };
        Ok(self.c32.insert(lane))
    }

    pub(crate) fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
        self.lane_config.recoupling_threads = Some(threads);
        self.f64
            .tree_context_mut()
            .backend_mut()
            .set_recoupling_threads(threads);
        self.c64
            .tree_context_mut()
            .backend_mut()
            .set_recoupling_threads(threads);
        if let Some(lane) = self.f32.as_mut() {
            lane.tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
        if let Some(lane) = self.c32.as_mut() {
            lane.tree_context_mut()
                .backend_mut()
                .set_recoupling_threads(threads);
        }
    }

    #[cfg(test)]
    pub(crate) fn recoupling_threads_are(&mut self, expected: usize) -> bool {
        self.f64
            .tree_context_mut()
            .backend_mut()
            .recoupling_threads()
            .get()
            == expected
            && self
                .c64
                .tree_context_mut()
                .backend_mut()
                .recoupling_threads()
                .get()
                == expected
            && self.f32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .recoupling_threads()
                    .get()
                    == expected
            })
            && self.c32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .recoupling_threads()
                    .get()
                    == expected
            })
    }

    #[cfg(test)]
    pub(crate) fn shares_cpu_context(&mut self, shared: &symtn_dense::SharedCpuContext) -> bool {
        self.f64
            .tree_context_mut()
            .backend_mut()
            .dense()
            .shares_cpu_context(shared)
            && self
                .f64
                .contract_backend()
                .dense()
                .shares_cpu_context(shared)
            && self
                .c64
                .tree_context_mut()
                .backend_mut()
                .dense()
                .shares_cpu_context(shared)
            && self
                .c64
                .contract_backend()
                .dense()
                .shares_cpu_context(shared)
            && self.f32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .dense()
                    .shares_cpu_context(shared)
                    && lane.contract_backend().dense().shares_cpu_context(shared)
            })
            && self.c32.as_mut().is_none_or(|lane| {
                lane.tree_context_mut()
                    .backend_mut()
                    .dense()
                    .shares_cpu_context(shared)
                    && lane.contract_backend().dense().shares_cpu_context(shared)
            })
    }

    /// Builds both deferred lanes, so a test can assert that a lane created
    /// after the runtime carries the same configuration as an eager one.
    #[cfg(test)]
    pub(crate) fn force_single_precision_lanes(&mut self) -> Result<(), Error> {
        self.f32_lane()?;
        self.c32_lane()?;
        Ok(())
    }
}

fn make_complex_multiplicity_free_ctx(
    ctx: &symtn_dense::SharedCpuContext,
) -> Result<CoefficientCtx<Complex64, RuleIdentity, Complex64>, Error> {
    Ok(CoefficientCtx::with_parts(
        symtn_tensors::TreeTransformExecutionContext::new(make_transform_ops(ctx)?),
        make_transform_ops(ctx)?,
        <DenseTreeTransformOperations as symtn_tensors::TensorContractBackend<
            Complex64,
            Complex64,
        >>::Workspace::default(),
    ))
}

macro_rules! rule_lanes {
    ($callback:ident) => {
        $callback! {
            mf: symtn_core::RuleIdentity,
            generic: symtn_core::RuleIdentity,
        }
    };
}

macro_rules! define_tensor_execution_context {
    ($( $field:ident: $key:ty ),+ $(,)?) => {
        /// Runtime-owned host tensor execution state.
        pub(crate) struct TensorExecutionContext {
            $(pub(crate) $field: Ctxs<$key>,)+
            pub(super) mf_c64_coeff_c64: CoefficientCtx<Complex64, RuleIdentity, Complex64>,
            #[cfg(all(test, feature = "cgc-gen"))]
            generic_lane_uses: usize,
        }

        impl TensorExecutionContext {
            // Why not retain a Runtime here: pooled contexts live inside that
            // Runtime, so the back-reference would form an Arc cycle.
            pub(crate) fn for_config(config: &RuntimeExecutionConfig) -> Result<Self, Error> {
                let mut context = Self {
                    $($field: Ctxs::with_config(&config.shared_ctx)?,)+
                    mf_c64_coeff_c64: make_complex_multiplicity_free_ctx(&config.shared_ctx)?,
                    #[cfg(all(test, feature = "cgc-gen"))]
                    generic_lane_uses: 0,
                };
                if let Some(threads) = config.recoupling_threads {
                    context.set_recoupling_threads(threads);
                }
                Ok(context)
            }

            pub(super) fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
                $(self.$field.set_recoupling_threads(threads);)+
                self.mf_c64_coeff_c64
                    .tree_context_mut()
                    .backend_mut()
                    .set_recoupling_threads(threads);
            }

            /// Returns the multiplicity-free lane matching scalar `D`.
            ///
            /// Only this lane is exposed to the typed facade; Generic-fusion
            /// execution remains behind its provider-specific boundary.
            pub(crate) fn multiplicity_free_lane<D: ScalarOps>(
                &mut self,
            ) -> Result<&mut Ctx<D, symtn_core::RuleIdentity>, Error> {
                D::ctx_of(&mut self.mf)
            }

            /// Returns the statically compatible complex-payload,
            /// complex-coefficient multiplicity-free lane.
            #[allow(
                dead_code,
                reason = "runtime foundation for the #592 typed-facade follow-up"
            )]
            pub(crate) fn complex_multiplicity_free_lane(
                &mut self,
            ) -> &mut CoefficientCtx<Complex64, RuleIdentity, Complex64> {
                &mut self.mf_c64_coeff_c64
            }

            pub(crate) fn generic_lane<D: ScalarOps>(
                &mut self,
            ) -> Result<&mut Ctx<D, symtn_core::RuleIdentity>, Error> {
                #[cfg(all(test, feature = "cgc-gen"))]
                {
                    self.generic_lane_uses += 1;
                }
                D::ctx_of(&mut self.generic)
            }

            #[cfg(all(test, feature = "cgc-gen"))]
            pub(crate) fn generic_lane_uses(&self) -> usize {
                self.generic_lane_uses
            }

            #[cfg(test)]
            pub(crate) fn recoupling_threads_are(&mut self, expected: usize) -> bool {
                true $(&& self.$field.recoupling_threads_are(expected))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .recoupling_threads()
                        .get()
                        == expected
            }

            #[cfg(test)]
            pub(crate) fn shares_cpu_context(
                &mut self,
                shared: &symtn_dense::SharedCpuContext,
            ) -> bool {
                true $(&& self.$field.shares_cpu_context(shared))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .dense()
                        .shares_cpu_context(shared)
                    && self
                        .mf_c64_coeff_c64
                        .contract_backend()
                        .dense()
                        .shares_cpu_context(shared)
            }

        }
    };
}

impl<D: ScalarOps> MultiplicityFreeCoefficientLane<f64> for D {
    fn lane(
        context: &mut TensorExecutionContext,
    ) -> Result<&mut CoefficientCtx<Self, RuleIdentity, f64>, Error> {
        Self::ctx_of(&mut context.mf)
    }
}

impl MultiplicityFreeCoefficientLane<Complex64> for Complex64 {
    fn lane(
        context: &mut TensorExecutionContext,
    ) -> Result<&mut CoefficientCtx<Self, RuleIdentity, Complex64>, Error> {
        Ok(&mut context.mf_c64_coeff_c64)
    }
}

rule_lanes!(define_tensor_execution_context);

macro_rules! define_runtime_state {
    ($( $field:ident: $key:ty ),+ $(,)?) => {
        /// Expert-layer execution contexts for the multiplicity-free and
        /// Generic-fusion namespaces, plus the rule-independent dense executor.
        ///
        /// CPU state only: a CUDA device context lives behind its own
        /// device lease on `RuntimeInner`, so a device operation never holds
        /// this lock.
        pub(crate) struct RuntimeState {
            $(pub(crate) $field: Ctxs<$key>,)+
            pub(super) mf_c64_coeff_c64: CoefficientCtx<Complex64, RuleIdentity, Complex64>,
            pub(crate) dense: Box<dyn symtn_dense::DenseExecutor + Send>,
        }

        impl RuntimeState {
            // Why not use `Ctxs::default()`: each default context creates a
            // private env-driven pool instead of sharing the runtime CPU pool.
            pub(super) fn with_config(
                dense: Box<dyn symtn_dense::DenseExecutor + Send>,
                ctx: &symtn_dense::SharedCpuContext,
            ) -> Result<Self, Error> {
                Ok(Self {
                    $($field: Ctxs::with_config(ctx)?,)+
                    mf_c64_coeff_c64: make_complex_multiplicity_free_ctx(ctx)?,
                    dense,
                })
            }

            pub(super) fn set_recoupling_threads(&mut self, threads: NonZeroUsize) {
                $(self.$field.set_recoupling_threads(threads);)+
                self.mf_c64_coeff_c64
                    .tree_context_mut()
                    .backend_mut()
                    .set_recoupling_threads(threads);
            }

            #[cfg(test)]
            pub(crate) fn recoupling_threads_are(&mut self, expected: usize) -> bool {
                true $(&& self.$field.recoupling_threads_are(expected))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .recoupling_threads()
                        .get()
                        == expected
            }

            #[cfg(test)]
            pub(crate) fn shares_cpu_context(
                &mut self,
                shared: &symtn_dense::SharedCpuContext,
            ) -> bool {
                true $(&& self.$field.shares_cpu_context(shared))+
                    && self
                        .mf_c64_coeff_c64
                        .tree_context_mut()
                        .backend_mut()
                        .dense()
                        .shares_cpu_context(shared)
                    && self
                        .mf_c64_coeff_c64
                        .contract_backend()
                        .dense()
                        .shares_cpu_context(shared)
            }
        }
    };
}

rule_lanes!(define_runtime_state);
