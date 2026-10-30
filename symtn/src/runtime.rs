//! User-layer runtime: owns shared execution and cache state so everyday
//! tensor code never passes explicit contexts around.

use std::any::Any;
use std::hash::Hash;
use std::sync::{Arc, Mutex, Weak};

use num_complex::{Complex32, Complex64};
use symtn_core::{HomSpaceId, RuleIdentity};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, DenseTreeTransformOperations, TensorContractFusionExecutionContext,
    TreeTransformOperation,
};

use crate::error::Error;
use crate::typed::ScalarOps;

#[cfg(feature = "cuda")]
mod cuda;
mod lanes;
mod leases;
#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(crate) use self::cuda::*;
pub(crate) use self::lanes::*;
pub use self::lanes::{Ctx, Ctxs};
pub(crate) use self::leases::*;

/// Type-keyed storage for state a downstream crate keeps on a [`Runtime`].
///
/// Each value lives under its own type, so a lookup returns the value of the
/// requested type or nothing. Why not one `Option<Box<dyn Any>>`: a value of
/// another type in that single slot left its owner a choice between a panic
/// and silently dropping someone else's state.
#[doc(hidden)]
#[derive(Default)]
pub struct ExtensionSlot {
    values: Vec<Box<dyn Any + Send>>,
}

impl ExtensionSlot {
    /// The stored value of type `T`, if one was inserted.
    pub fn get_mut<T: Any + Send>(&mut self) -> Option<&mut T> {
        self.values
            .iter_mut()
            .find_map(|value| value.as_mut().downcast_mut::<T>())
    }

    /// The stored value of type `T`, inserting `init()` first if there is none.
    pub fn get_or_insert_with<T: Any + Send>(&mut self, init: impl FnOnce() -> T) -> &mut T {
        let index = match self
            .values
            .iter()
            .position(|value| value.as_ref().is::<T>())
        {
            Some(index) => index,
            None => {
                self.values.push(Box::new(init()));
                self.values.len() - 1
            }
        };
        self.values[index]
            .as_mut()
            .downcast_mut::<T>()
            .expect("the value at `index` was found or inserted as a `T`")
    }
}

struct RuntimeInner {
    /// Contraction-plan cache behind its own mutex, separate from `state` but
    /// still synchronized while cache work is in progress.
    extensions: Mutex<ExtensionSlot>,
}

/// Execution runtime for the user-layer [`crate::typed::TensorMap`] API.
///
/// A `Runtime` is built once via [`Runtime::builder`] and then carried
/// implicitly by every tensor created from it; operations reuse the
/// Runtime-owned completed tree-transform store without explicit context
/// arguments. Context-local operation caches stay disabled; cloning a
/// `Runtime` clones a shared handle, not the state.
///
/// Concurrency: a standalone tensor operation builds its execution scratch for
/// the call and runs on the caller's engine; the Runtime holds no execution
/// state for it. Shared structure stores and the plan cache keep their own
/// short lookup and publication locks.
///
/// CUDA: a Runtime is not tied to a device. A device operation runs on the
/// caller's CUDA engine, which owns its stream and allocation domain; tensors
/// of one Runtime may live on several engines, and an operation refuses
/// operands of an engine other than the one it is given.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{Z2FusionRule, Z2Irrep};
/// use symtn::typed::{GradedSpace, Runtime, TensorMap};
///
/// let engine = &mut symtn::typed::CpuEngine::serial();
/// let rt = Runtime::builder().build()?;
/// let v = GradedSpace::try_new(
///     Arc::new(Z2FusionRule),
///     [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
/// )?;
/// let a: TensorMap<_, f64> = TensorMap::zeros(engine, &rt, [&v], [&v])?;
/// assert_eq!(a.norm(engine, 2.0)?, 0.0);
/// # Ok::<(), symtn::typed::Error>(())
/// ```
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

/// Non-owning identity for internal state parked outside an active execution.
///
/// Equality and hashing use the Runtime allocation's address. The held `Weak`
/// keeps that allocation reserved, so a later Runtime can never reuse the
/// address while this identity is live.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RuntimeIdentity {
    inner: Weak<RuntimeInner>,
}

impl PartialEq for RuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for RuntimeIdentity {}

impl std::hash::Hash for RuntimeIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.as_ptr().hash(state);
    }
}

impl RuntimeIdentity {
    #[doc(hidden)]
    pub fn matches(&self, runtime: &Runtime) -> bool {
        self.inner
            .upgrade()
            .is_some_and(|inner| Arc::ptr_eq(&inner, &runtime.inner))
    }

    #[doc(hidden)]
    pub fn is_alive(&self) -> bool {
        self.inner.strong_count() != 0
    }
}

impl Runtime {
    /// Starts building a runtime. The default runtime uses the CPU backend;
    /// feature-gated device options such as CUDA are attached through the
    /// builder so tensor construction keeps the same shape.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub(crate) fn same_runtime(&self, other: &Runtime) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn identity(&self) -> RuntimeIdentity {
        RuntimeIdentity {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// The resolved transformer of a typed Host `*_into` request whose spaces
    /// were proved to match once before (the exact-layout memo on the
    /// process-global entry of coefficient type `C`); `None` takes the full
    /// path.
    pub(crate) fn exact_layout_tree_pair_hit_of<R, C: 'static + Send + Sync>(
        rule: &RuleIdentity,
        operation: symtn_tensors::TreeTransformOperationView<'_>,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> Option<symtn_tensors::TreeTransformStructure<C>> {
        let (source_homspace, source_layout) = bound_layout_identity(source);
        let (destination_homspace, destination_layout) = bound_layout_identity(destination);
        symtn_tensors::exact_layout_tree_pair_hit::<C>(
            rule,
            operation,
            destination.space().structure(),
            source.space().structure(),
            (&source_homspace, source_layout),
            (&destination_homspace, destination_layout),
        )
    }

    pub(crate) fn admit_exact_tree_pair_layout_of<R, C: 'static + Send + Sync>(
        rule: &RuleIdentity,
        operation: &TreeTransformOperation,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> bool {
        let (source_homspace, source_layout) = bound_layout_identity(source);
        let (destination_homspace, destination_layout) = bound_layout_identity(destination);
        symtn_tensors::admit_exact_tree_pair_layout::<C>(
            rule,
            operation,
            destination.space().structure(),
            source.space().structure(),
            (&source_homspace, source_layout),
            (&destination_homspace, destination_layout),
        )
    }

    /// The dense executor of one factorization: descriptors and job scratch
    /// owned by this invocation, dropped when it returns. It owns no threads;
    /// the operation runs it on the caller's engine.
    pub(crate) fn lease_dense(&self) -> DenseLease {
        #[cfg(test)]
        {
            crate::typed::probe::hit(crate::typed::probe::Event::DenseLease);
            if let Some(executor) = crate::typed::probe::take_injected_dense() {
                return DenseLease {
                    executor: Some(executor),
                    injected: true,
                };
            }
        }
        DenseLease {
            executor: Some(Box::new(symtn_dense::DefaultDenseExecutor::new())),
            #[cfg(test)]
            injected: false,
        }
    }

    /// Locked access to the type-erased downstream extension slot (the
    /// contraction-plan cache and its configuration: the types live in
    /// `symtn-network`, which depends on this crate and claims and downcasts
    /// its entry on first use). Behind its own mutex rather than the coarse
    /// `state` one (#155), so the network hot path never contends with
    /// standalone ops. Expert seam for `symtn-network`; the mutex is held for
    /// the duration of `f`, so do not run tensor operations inside it.
    pub(crate) fn with_extension_slot<R>(&self, f: impl FnOnce(&mut ExtensionSlot) -> R) -> R {
        f(&mut self
            .inner
            .extensions
            .lock()
            .expect("symtn extension slot poisoned"))
    }
}

fn bound_layout_identity<R>(space: &BoundDynamicFusionMapSpace<R>) -> (HomSpaceId, [usize; 3]) {
    let space = space.space();
    (
        space.homspace().id(),
        [
            space.structure().content_id(),
            space.homspace().codomain().len(),
            space.homspace().domain().len(),
        ],
    )
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

/// Builder for [`Runtime`]; see [`Runtime::builder`].
#[derive(Debug, Default)]
pub struct RuntimeBuilder {}

impl RuntimeBuilder {
    /// Finishes the build.
    pub fn build(self) -> Result<Runtime, Error> {
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                extensions: Mutex::new(ExtensionSlot::default()),
            }),
        })
    }
}
