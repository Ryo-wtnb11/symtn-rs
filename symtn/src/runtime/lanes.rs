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

/// Selects one of the three payload/coefficient pairs of an execution context.
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
/// real-coefficient lanes. A lane is built on first use: an operation runs on
/// one payload dtype, so building every lane would charge each call for lanes
/// it never names. A lane holds descriptors and scratch only; every operation
/// runs on the caller's engine.
///
/// The first lane a context names is stored inline, so a standalone
/// operation, which names one, allocates no lane. Later lanes of other dtypes
/// are boxed, so moving the context copies at most one lane per namespace
/// (#228).
pub struct Ctxs<Key: Clone + Eq + Hash + Send + Sync + 'static> {
    first: Option<FirstLane<Key>>,
    f64: Option<Box<Ctx<f64, Key>>>,
    c64: Option<Box<Ctx<Complex64, Key>>>,
    f32: Option<Box<Ctx<f32, Key>>>,
    c32: Option<Box<Ctx<Complex32, Key>>>,
}

enum FirstLane<Key: Clone + Eq + Hash + Send + Sync + 'static> {
    F64(Ctx<f64, Key>),
    C64(Ctx<Complex64, Key>),
    F32(Ctx<f32, Key>),
    C32(Ctx<Complex32, Key>),
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Default for Ctxs<Key> {
    fn default() -> Self {
        Self {
            first: None,
            f64: None,
            c64: None,
            f32: None,
            c32: None,
        }
    }
}

fn built<T: Default>(lane: &mut Option<Box<T>>) -> &mut T {
    lane.get_or_insert_with(Box::default)
}

macro_rules! lane_accessor {
    ($name:ident, $field:ident, $variant:ident, $ty:ty) => {
        pub(crate) fn $name(&mut self) -> &mut Ctx<$ty, Key> {
            let first = self
                .first
                .get_or_insert_with(|| FirstLane::$variant(Ctx::default()));
            match first {
                FirstLane::$variant(lane) => lane,
                _ => built(&mut self.$field),
            }
        }
    };
}

impl<Key: Clone + Eq + Hash + Send + Sync + 'static> Ctxs<Key> {
    lane_accessor!(f64_lane, f64, F64, f64);
    lane_accessor!(c64_lane, c64, C64, Complex64);
    lane_accessor!(f32_lane, f32, F32, f32);
    lane_accessor!(c32_lane, c32, C32, Complex32);
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
        /// Host tensor execution scratch of one operation or prepared workspace.
        pub(crate) struct TensorExecutionContext {
            $(pub(crate) $field: Ctxs<$key>,)+
            mf_c64_coeff_c64: Option<Box<CoefficientCtx<Complex64, RuleIdentity, Complex64>>>,
        }

        impl TensorExecutionContext {
            pub(crate) fn new() -> Self {
                Self {
                    $($field: Ctxs::default(),)+
                    mf_c64_coeff_c64: None,
                }
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
                built(&mut self.mf_c64_coeff_c64)
            }

            pub(crate) fn generic_lane<D: ScalarOps>(
                &mut self,
            ) -> Result<&mut Ctx<D, symtn_core::RuleIdentity>, Error> {
                D::ctx_of(&mut self.generic)
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
        Ok(built(&mut context.mf_c64_coeff_c64))
    }
}

rule_lanes!(define_tensor_execution_context);
