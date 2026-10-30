//! Sealed storage execution traits: one typed operation body per family,
//! with each storage supplying only its kernels (#1756).

use super::*;

pub(super) mod sealed {
    pub trait Sealed {}

    /// The proof that a family kernel is called from inside this crate.
    ///
    /// Why not sealing alone: sealing stops external impls, not external
    /// calls through `<S as XExec<R, D>>::method`, and a kernel skips the
    /// admission its typed body owns (`PreparedExec::lease_members` would
    /// hand out the device lease). The field is private and [`EXEC`] is
    /// `pub(crate)`, so no value of this type exists outside the crate.
    #[derive(Clone, Copy)]
    pub struct ExecToken(());

    /// The only [`ExecToken`].
    pub(crate) const EXEC: ExecToken = ExecToken(());
}

impl<D> sealed::Sealed for Vec<D> {}

/// One storage's placement and capability set for rule `R`, payload `D`.
///
/// The family execution traits (e.g. [`ReduceExec`]) are its subtraits. Each
/// CUDA impl restates the device capability bounds for its family; a rule or
/// payload outside them has no device method at all. The dispatch pattern is
/// described in the "Operation dispatch" section of the `symtn::typed` module
/// documentation.
///
/// Every method of the family takes an `ExecToken` that only this crate can
/// construct, so external code can name a kernel but not call it:
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{PreparedExec, Runtime};
///
/// fn call(_runtime: &Runtime) {
///     let _kernel = <Vec<f64> as PreparedExec<U1FusionRule, f64>>::lease_members;
/// }
/// ```
///
/// ```compile_fail,E0061
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{PreparedExec, Runtime};
///
/// fn call(engine: &mut dyn symtn::typed::Engine, _runtime: &Runtime) {
///     let _lease = <Vec<f64> as PreparedExec<U1FusionRule, f64>>::lease_members(engine);
/// }
/// ```
#[doc(hidden)]
pub trait TypedStorage<R, D>: TensorStorage<D> + sealed::Sealed + Sized + 'static {
    /// The storage noun in shared error messages ("host", "CUDA").
    const NOUN: &'static str;

    /// The compact-representation bridge: `Some(t)` on Host, `None` on CUDA.
    ///
    /// Only compact arms may branch on it; CUDA tensors never hold compact
    /// data, since `to_cuda` densifies and every device kernel outputs dense.
    fn as_host(_: ExecToken, t: &TensorMap<R, D, Self>) -> Option<&TensorMap<R, D>>;

    /// [`Self::as_host`] for a compact arm's existing destination.
    fn as_host_mut(_: ExecToken, t: &mut TensorMap<R, D, Self>) -> Option<&mut TensorMap<R, D>>;

    /// The inverse of [`Self::as_host`] for a compact arm's result. Reached
    /// only after `as_host` returned `Some`, so the CUDA impl's error is
    /// unreachable.
    fn from_host(_: ExecToken, t: TensorMap<R, D>) -> Result<TensorMap<R, D, Self>, Error>;
}

impl<R, D: 'static> TypedStorage<R, D> for Vec<D> {
    const NOUN: &'static str = "host";

    fn as_host(_: ExecToken, t: &TensorMap<R, D, Self>) -> Option<&TensorMap<R, D>> {
        Some(t)
    }

    fn as_host_mut(_: ExecToken, t: &mut TensorMap<R, D, Self>) -> Option<&mut TensorMap<R, D>> {
        Some(t)
    }

    fn from_host(_: ExecToken, t: TensorMap<R, D>) -> Result<TensorMap<R, D, Self>, Error> {
        Ok(t)
    }
}
