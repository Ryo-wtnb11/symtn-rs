use super::*;

/// Scalar payloads supported by [`TensorMap`], base capability.
///
/// Admits the payload-dtype-independent half of the typed API: construction
/// and inspection, [`TensorMap::adjoint`], `scale`/`axpby`, the
/// reductions (`norm`, `inner`, `tr`),
/// contraction/`compose`/`otimes`/`cat`, the structural transforms
/// (`permute`, `braid`, `transpose`, `repartition`, `twist`, `flip`),
/// `restrict_leg`/`embed_leg`/`diagview`, trace, and
/// network execution.
///
/// Factorizations need [`FactorizationScalar`]; matrix functions, inverses,
/// solves and the general eigendecomposition need [`AdvancedLinalgScalar`].
///
/// This trait is sealed; the supported scalar types are `f64`,
/// [`num_complex::Complex64`], `f32` and [`num_complex::Complex32`].
/// Admission is split across markers rather than carried by one trait so that
/// a payload dtype joins one family at a time, with its own review and its own
/// tolerance evidence (<https://github.com/Ryo-wtnb11/symtn-rs/issues/1065>);
/// single precision has since joined every host family.
///
/// # Precision
///
/// The structural coefficients of a symmetry (F/R symbols, quantum
/// dimensions, fermionic signs) stay `f64` for every payload dtype and act on
/// a single-precision payload through the existing coefficient actions, so an
/// `f32` tensor carries `f32` data recoupled by exactly-computed `f64`
/// algebraic numbers. This follows TensorKit, whose
/// `promote_permute(Float32, I)` stays `Float32` for a real sector scalar
/// type (`indexmanipulations.jl`). A provider with *complex* structural
/// coefficients and a single-precision payload remains a compile-time
/// boundary.
///
/// [`TensorMap::norm`] returns `f64` for every payload dtype and accumulate in `f64`;
/// [`TensorMap::inner`] and [`TensorMap::tr`] return the
/// payload type, as TensorKit's do, but also sum in double precision.
///
/// Tolerances are the caller's: the predicates that take a `tol` have no
/// default, and a tolerance chosen for `f64` (`1e-12` in the examples of this
/// crate) rejects everything at `f32`. Scale it by the payload's own epsilon
/// — `f32::EPSILON` is about `1.2e-7` — as TensorKit's `rtoldefault` does.
///
/// Execution state is per payload dtype, but the structural caches are not:
/// completed transformers and composed coefficients are process-global
/// ([`crate::cache`]), so single- and double-precision activity is reported
/// and cleared together.
///
/// # Annotate the payload dtype
///
/// With two real payloads, a float literal does not select one on its own:
/// inference waits for the end-of-function `f64` fallback, which arrives too
/// late for method resolution, so a payload dtype taken *solely* from float
/// literals, whose result then has a method called on it, fails with `E0689 ambiguous numeric type {float}`. It affects
/// [`TensorMap::diagonal`], [`TensorMap::from_subblock_fn`] closures returning
/// literals, and `scale`/`axpby` coefficients on an un-annotated
/// `zeros`/`isomorphism`/`rand_with_seed`:
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let engine = &mut symtn::typed::CpuEngine::serial();
/// let runtime = Runtime::builder().build()?;
/// let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
/// // Annotated, so the literals have a type to be.
/// let t: TensorMap<U1FusionRule, f64> = TensorMap::diagonal(engine,
///     &runtime,
///     &leg,
///     [SectorSpectrum { sector: U1Irrep::new(0), values: vec![2.0, 3.0] }],
/// )?;
/// assert!((t.tr(engine)?.abs() - 5.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
///
/// Passing such a result to an `f64` parameter, comparing it, or never
/// pinning it still compiles, because the fallback does arrive. Only a method
/// call on it does not.
///
/// # What single precision does *not* reach
///
/// Each rejection below is paired with the same body at `f64`, which is what
/// shows the `compile_fail` is the payload dtype and not an unrelated mistake.
///
/// Factorizations are *not* in this list: single precision reaches
/// [`FactorizationScalar`] under
/// <https://github.com/Ryo-wtnb11/symtn-rs/issues/1324>, so `svd_compact` and its
/// siblings compile at `f32` and [`num_complex::Complex32`]. Neither is
/// persistence: the wire format has `f32` and `Complex32` tags since
/// <https://github.com/Ryo-wtnb11/symtn-rs/issues/1325> (see
/// [`TypedPersistenceCodec`]).
///
/// Nor is the advanced family: matrix functions, `inv`/`pinv`/`solve` and the
/// general eigendecomposition reach both single-precision payloads under
/// <https://github.com/Ryo-wtnb11/symtn-rs/issues/1459> (see
/// [`AdvancedLinalgScalar`]). What stays closed is a provider with complex
/// structural coefficients, described above.
///
/// A device payload is *not* one of the closed gates: every dtype of this
/// marker uploads (<https://github.com/Ryo-wtnb11/symtn-rs/issues/1336>) and, since
/// <https://github.com/Ryo-wtnb11/symtn-rs/issues/1341>, factorizes on device.
#[cfg_attr(
    feature = "cuda",
    doc = "```
use symtn::sector::U1FusionRule;
use symtn::typed::TensorMap;

fn single_precision_upload(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, f32>) {
    let _ = tensor.to_cuda(engine);
}

fn double_precision_upload(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, f64>) {
    let _ = tensor.to_cuda(engine);
}
```"
)]
#[allow(private_bounds)]
pub trait TensorScalar: ScalarOps {}

impl TensorScalar for f64 {}
impl TensorScalar for num_complex::Complex64 {}
impl TensorScalar for f32 {}
impl TensorScalar for num_complex::Complex32 {}

/// Scalar payloads admitted to the factorization family.
///
/// Adds, on top of [`TensorScalar`]: QR/LQ (compact and full), SVD (compact,
/// full, values), Hermitian eigendecomposition (full, values), left/right
/// null spaces, and left/right polar.
/// [`GradedSpace::find_truncated`] carries no payload and stays on
/// [`TensorScalar`].
///
/// Sealed through [`TensorScalar`]: implemented for `f64`,
/// [`num_complex::Complex64`], `f32` and [`num_complex::Complex32`]. Matrix
/// functions, `inv`/`pinv`/`solve` and the general eigendecomposition are on
/// [`AdvancedLinalgScalar`].
///
/// # Single-precision tolerances
///
/// Every factorization here is backward stable in the payload's own working
/// precision, so a result computed at `f32`/[`num_complex::Complex32`] agrees
/// with the `f64`/[`num_complex::Complex64`] factorization of the *same*
/// (exactly widened) input only to `eps(real(D))`, not to `f64::EPSILON`:
/// `f32::EPSILON` is `1.19e-7`, about `9e8` times coarser. Concretely,
///
/// * every `tol` is the caller's and has no default. A residual test such as
///   `t.axpby(1, &t.adjoint()?, -1)?.norm(2.0)? <= tol * t.norm(2.0)?.max(1.0)`
///   (TensorKit `ishermitian`) sees a residual that an `f64` payload keeps
///   near `1e-16` sit near `1e-7` in `f32`, so a `tol` chosen for `f64` —
///   `1e-12` in most of the doctests here — rejects a perfectly good
///   single-precision result.
///   Scale it with the payload: MatrixAlgebraKit's `defaulttol`
///   (`src/common/defaults.jl`) is `eps(real(T))^(2/3)`, which is `3.7e-11` at
///   `f64` and `2.4e-5` at `f32`; TensorKit's `isapprox` default is
///   `sqrt(eps(real(T)))`, `1.5e-8` and `3.5e-4`;
/// * the truncation policies that take tolerances
///   ([`Truncation::relative_cutoff`], [`Truncation::relative_error`]) are the same
///   story: a cutoff below the payload's own noise floor keeps noise. The
///   spectra themselves, the truncation `error` and every norm this crate
///   returns are `f64` at *every* payload dtype (they are widened, not
///   recomputed), so the `f64` type of a tolerance says nothing about the
///   precision of the values it is compared against;
/// * **the kept/discarded sets near a tie may legitimately differ from the
///   double-precision run of the same physics.** The singular values and
///   Hermitian eigenvalues of a single-precision block carry a relative error
///   of order `eps(f32)` times the condition number, so two values separated
///   by less than that are not ordered reliably, and a magnitude-driven
///   policy ([`Truncation::rank`], [`Truncation::relative_cutoff`],
///   [`Truncation::relative_error`]) can act on a different order.
///
///   What holds at every payload dtype is the policy's **postcondition
///   against the spectrum that run actually computed**: a kept value is at or
///   above the threshold of that run, and the reported `error` is the weighted
///   2-norm of what that run discarded, within its budget. What does *not*
///   carry across dtypes is a comparison of the two runs' outcomes, and how
///   far it fails depends on the policy:
///
///   * [`Truncation::rank`] at a tie swaps two interchangeable states, so the
///     kept count is the budget either way and the discarded weight agrees to
///     the accuracy of the values themselves;
///   * [`Truncation::relative_cutoff`] and [`Truncation::relative_error`] have
///     a *boundary*, not a tie: a value within noise of the threshold, or a
///     tail whose cumulative weight is within noise of the budget, is kept by
///     one run and dropped by the other. Then the kept count differs by one
///     state and the discarded weight differs by that whole state's weight —
///     not by `eps * cond`.
///
///   Separate the spectrum by more than `eps(real(D)) * cond`, and keep the
///   budget away from a cumulative-weight boundary by the same margin, if the
///   outcome has to be reproducible across dtypes.
///
/// # Source compatibility
///
/// The `E0689` note on [`TensorScalar`] applies here too, one level further
/// in: a payload dtype taken *solely* from float literals is not pinned to
/// `f64` by calling a factorization on it, because four payload types
/// implement it. Inference waits for the end-of-function fallback, which is too late for a method call
/// on the result — `t.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)?.d.diagview(engine)?[0].values[0].abs()` on an
/// un-annotated `from_subblock_fn` tensor needs the payload dtype written
/// down. The same holds for [`GradedSpace::find_truncated`], whose spectrum
/// type is `SpectrumMagnitude` and has four implementors.
///
/// A caller generic over the base marker cannot reach a factorization:
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{TensorMap, TensorScalar};
///
/// fn base_only<D: TensorScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.svd_compact(engine, &[0], &[1]);
/// }
/// ```
///
/// The same body compiles once the caller asks for this marker, which is what
/// shows the rejection above is the bound and not an unrelated mistake:
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing<D: FactorizationScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.svd_compact(engine, &[0], &[1]);
/// }
/// ```
///
/// The same pair for `qr_compact`:
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{TensorMap, TensorScalar};
///
/// fn base_only<D: TensorScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.qr_compact(engine, &[0], &[1]);
/// }
/// ```
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing<D: FactorizationScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.qr_compact(engine, &[0], &[1]);
/// }
/// ```
///
/// This marker does not *by itself* carry the device factorizations:
/// those are `CudaFactorizationPayload`, which every payload of this marker
/// implements (<https://github.com/Ryo-wtnb11/symtn-rs/issues/1341>), device QR
/// included (<https://github.com/Ryo-wtnb11/symtn-rs/issues/1271>). The examples below
/// only compile under the `cuda` feature, so a host-only CI run does not
/// exercise them.
#[cfg_attr(
    feature = "cuda",
    doc = "```
use symtn::sector::U1FusionRule;
use symtn::typed::{CudaStorage, TensorMap};

fn f32_device_svd(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, f32, CudaStorage<f32>>) {
    let _ = tensor.svd_compact(engine, &[0], &[1]);
}

fn f64_device_svd(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>) {
    let _ = tensor.svd_compact(engine, &[0], &[1]);
}
```

```
use num_complex::Complex32;
use symtn::sector::U1FusionRule;
use symtn::typed::{CudaStorage, TensorMap};

fn c32_device_qr(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, Complex32, CudaStorage<Complex32>>) {
    let _ = tensor.qr_compact(engine, &[0], &[1]);
}
```"
)]
pub trait FactorizationScalar: TensorScalar {}

impl FactorizationScalar for f64 {}
impl FactorizationScalar for num_complex::Complex64 {}
impl FactorizationScalar for f32 {}
impl FactorizationScalar for num_complex::Complex32 {}

/// Scalar payloads admitted to the advanced linear-algebra family.
///
/// Adds, on top of [`FactorizationScalar`]: the matrix functions
/// ([`TensorMap::exp`]), [`TensorMap::inv`], [`TensorMap::pinv`],
/// [`TensorMap::solve`], and the general (non-Hermitian)
/// eigendecomposition (`eig_full`, `eig_vals`).
///
/// These are the operations whose accuracy depends on conditioning rather than
/// on one backend call, so they are the last family a new payload dtype joins.
///
/// Sealed through [`TensorScalar`]: implemented for `f64`,
/// [`num_complex::Complex64`], `f32` and [`num_complex::Complex32`] (single
/// precision since <https://github.com/Ryo-wtnb11/symtn-rs/issues/1459>).
///
/// # Single precision
///
/// Every operation runs in the payload's own precision; nothing is silently
/// widened. The general eigendecomposition returns its factors in
/// `D::Eig` — [`num_complex::Complex32`] for an `f32` or
/// [`num_complex::Complex32`] payload — on both the multiplicity-free and the
/// Checked-Generic dispatch, while the reported eigenvalue list (`eig_vals`)
/// stays `Complex64` at every payload dtype, as every spectrum and norm of
/// this crate does.
///
/// A single-precision result agrees with the double-precision result of the
/// same input only to `eps(f32)` times the conditioning of the operation:
/// `inv`, `solve`, `pinv` and negative powers carry the condition number of
/// the divisor, and eigenvalues that of the eigenbasis. `pinv`'s `rcond`
/// is the caller's; one below `eps(f32)` keeps noise. Unlike TensorKit, whose
/// ordinary test matrix skips `\` at `Float32`/`ComplexF32`, `solve` is
/// admitted: it is one LU solve per coupled block, backward stable at any
/// working precision, and tested against the widened oracle.
///
/// Neither the base marker nor [`FactorizationScalar`] reaches a matrix
/// function:
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{TensorMap, TensorScalar};
///
/// fn base_only<D: TensorScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.exp(engine, &[0], &[1]);
/// }
/// ```
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing_only<D: FactorizationScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.exp(engine, &[0], &[1]);
/// }
/// ```
///
/// This marker does:
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{AdvancedLinalgScalar, TensorMap};
///
/// fn advanced<D: AdvancedLinalgScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.exp(engine, &[0], &[1]);
///     let _ = tensor.inv(engine, &[0], &[1]);
///     let _ = tensor.solve(engine, &[0], &[1], tensor, &[0], &[1]);
/// }
/// ```
///
/// [`TensorMap::inv`] on its own, so the pair above cannot pass on `exp` alone:
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{FactorizationScalar, TensorMap};
///
/// fn factorizing_only<D: FactorizationScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.inv(engine, &[0], &[1]);
/// }
/// ```
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{AdvancedLinalgScalar, TensorMap};
///
/// fn advanced<D: AdvancedLinalgScalar>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>) {
///     let _ = tensor.inv(engine, &[0], &[1]);
/// }
/// ```
///
/// And the general eigendecomposition, whose `D::Eig: TensorScalar` bound is
/// held constant across the pair so that only the marker differs:
///
/// ```compile_fail
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{FactorizationScalar, TensorMap, TensorScalar};
///
/// fn factorizing_only<D>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>)
/// where
///     D: FactorizationScalar,
///     D::Eig: TensorScalar,
/// {
///     let _ = tensor.eig_full(engine, &[0], &[1]);
/// }
/// ```
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{AdvancedLinalgScalar, TensorMap, TensorScalar};
///
/// fn advanced<D>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D>)
/// where
///     D: AdvancedLinalgScalar,
///     D::Eig: TensorScalar,
/// {
///     let _ = tensor.eig_full(engine, &[0], &[1]);
/// }
/// ```
pub trait AdvancedLinalgScalar: FactorizationScalar {}

impl AdvancedLinalgScalar for f64 {}
impl AdvancedLinalgScalar for num_complex::Complex64 {}
impl AdvancedLinalgScalar for f32 {}
impl AdvancedLinalgScalar for num_complex::Complex32 {}

/// Scalar payloads a [`CudaStorage`] device buffer can own.
///
/// This bundles the typed payload trait [`TensorScalar`] (whose `ScalarOps`
/// half selects the matching multiplicity-free execution context) with the
/// device dtype [`crate::typed::CudaScalar`]. All four payload dtypes of the
/// base family are admitted: `f64`, [`num_complex::Complex64`], `f32` and
/// [`num_complex::Complex32`].
///
/// This marker carries the *base* device family only — transfer, arithmetic,
/// the reductions, contraction/`compose` and the structural transforms. The
/// device factorizations, QR included, sit behind [`CudaFactorizationPayload`]
/// on top of it.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub trait CudaPayload: TensorScalar + symtn_dense::CudaScalar {}

#[cfg(feature = "cuda")]
impl CudaPayload for f64 {}
#[cfg(feature = "cuda")]
impl CudaPayload for num_complex::Complex64 {}
#[cfg(feature = "cuda")]
impl CudaPayload for f32 {}
#[cfg(feature = "cuda")]
impl CudaPayload for num_complex::Complex32 {}

/// Device payloads admitted to the *device* factorization family
/// (`svd_compact`, `eigh_full`, `qr_compact`).
///
/// The marker exists so that admitting a dtype to the device *base* family and
/// to the *host* factorization family cannot, together, open the device
/// factorizations by accident: those two admissions say nothing about the
/// device SVD gauge, the device Hermitian admission rule or the host-side
/// truncation composition at that dtype. All four payloads are admitted here
/// because <https://github.com/Ryo-wtnb11/symtn-rs/issues/1341> supplied that
/// evidence for `f32` and [`num_complex::Complex32`]; device `qr_compact`
/// covers all four
/// (<https://github.com/Ryo-wtnb11/symtn-rs/issues/1271>).
///
/// Sealed through [`CudaPayload`].
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub trait CudaFactorizationPayload: CudaPayload + FactorizationScalar {}

#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for f64 {}
#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for num_complex::Complex64 {}
#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for f32 {}
#[cfg(feature = "cuda")]
impl CudaFactorizationPayload for num_complex::Complex32 {}

/// Internal scalar operations shared by typed tensor execution.
/// The reduction accumulator is [`symtn_tensors::WideScalar::Wide`], which
/// owns that decision for every crate: `Self` for the double-precision pair
/// (so those reductions are unchanged, down to the emitted arithmetic) and the
/// double-precision scalar of the same field for a single-precision payload.
/// `Wide: FactorScalar` because the typed reductions form conjugated products
/// in the accumulator and widen it to `Complex64` at the end.
///
/// TensorKit reaches comparable accuracy differently — `LinearAlgebra.norm`
/// scales each block — so accumulating wide is a deliberate symtn choice,
/// recorded in <https://github.com/Ryo-wtnb11/symtn-rs/issues/1315>.
pub(crate) trait ScalarOps:
    FactorScalar
    + symtn_tensors::RecouplingCoefficientAction<f64>
    + symtn_tensors::ZeroBytes
    + symtn_tensors::WideScalar<Wide: FactorScalar>
{
    /// Whether `FactorScalar::adjoint` is the identity (a real dtype).
    const CONJUGATION_IS_IDENTITY: bool;

    /// Returns the execution lane for this payload dtype, building it if the
    /// execution context has not needed it yet. Building a lane cannot fail
    /// today; the `Result` is the facade's error path for lane selection.
    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> Result<&mut Ctx<Self, Key>, Error>;
    fn rand_unit(state: &mut u64) -> Self;
    fn abs_value(self) -> f64;
    fn exp_value(self) -> Self;
    fn recip_value(self) -> Self;
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols
        + PhysicalFusionBasis<Scalar = <R as MultiplicityFreeFusionSymbols>::Scalar>,
    D: TensorScalar
        + symtn_tensors::RecouplingCoefficientAction<<R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    /// Expands the reduced coupled-layout payload into an owned physical-basis
    /// tensor on the Host.
    ///
    /// The returned axes and entries follow [`PhysicalDense`]. The provider's
    /// carrier basis and fusion coefficients determine the embedding; no
    /// symmetry-specific dispatch occurs in this method.
    ///
    /// # Cost
    ///
    /// A lazy adjoint or a compact diagonal is first densified into an
    /// operation-local reduced buffer, released on return: one copy of the
    /// reduced payload, never larger than the physical output. TensorKit's
    /// `convert(Array, t)` pays the same copy through `t[f₁, f₂]`.
    ///
    /// # Index order matches TensorKit
    ///
    /// Each axis is ordered as TensorKit's `convert(Array, t)` orders it: a
    /// dual space `V'` is listed in `V`'s order, so bending a leg with
    /// [`Self::permute`] or [`Self::repartition`] leaves its index order
    /// unchanged. For the built-in U(1) and SU(2) providers, whose braiding is
    /// bosonic, the expansion of a permuted tensor is exactly the matching
    /// axis permutation of this array.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let rt = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [-1, 0, 1].map(|q| (U1Irrep::new(q), 1)))?;
    /// let id = TensorMap::<U1FusionRule, f64>::isomorphism(engine, &rt, [&v], [&v])?;
    ///
    /// // `V <- V`.
    /// let matrix = id.to_physical_dense(engine).expect("U(1) has a physical basis");
    /// assert_eq!(matrix.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    ///
    /// // `V ⊗ V' <- ()`: the same entries.
    /// let bent = id.permute(engine, &[0, 1], &[])?;
    /// let vector = bent.to_physical_dense(engine).expect("U(1) has a physical basis");
    /// assert_eq!(vector.data, matrix.data);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// Providers opt in at compile time by implementing
    /// [`PhysicalFusionBasis`]:
    ///
    /// ```compile_fail
    /// use symtn::sector::Z2FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn unsupported(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<Z2FusionRule, f64>) {
    ///     let _ = tensor.to_physical_dense(engine);
    /// }
    /// ```
    pub fn to_physical_dense(
        &self,
        engine: &mut dyn Engine,
    ) -> Result<PhysicalDense<D>, PhysicalDenseError<<R as PhysicalFusionBasis>::Error>> {
        symtn_operations::host_pool::region(
            engine,
            symtn_operations::host_pool::HostPoolSite::Strided,
            |_| (),
        )?;
        // Why an operation-local copy rather than `Unsupported`: the physical
        // output is never smaller than the reduced payload, so the copy at
        // most doubles the bytes this conversion already writes, and
        // `PhysicalDenseError` has no `Unsupported` variant to refuse with.
        let adjoint;
        let payload = match &self.repr {
            TypedTensorRepr::Owned(body) => body.materialized_dense_data(),
            TypedTensorRepr::Adjoint(view) => {
                #[cfg(test)]
                probe::hit(probe::Event::AdjointMaterialization);
                adjoint = symtn_tensors::materialize_adjoint_data_dyn(
                    view.parent.space.space(),
                    view.logical_space.space(),
                    view.parent_data(),
                )?;
                std::borrow::Cow::Borrowed(adjoint.as_slice())
            }
        };
        let source = BoundDynamicTensorRef::try_new(self.logical_space(), &payload)?;
        let (shape, data) = expand_physical_host(source)?;
        Ok(PhysicalDense { shape, data })
    }

    /// Projects physical-basis data into this tensor's exact reduced schema.
    ///
    /// `self` supplies the runtime, provider allocation, fusion-tree layout,
    /// offsets, and strides. Its numeric payload is not read or modified. All
    /// validation and provider queries finish before the returned tensor is
    /// constructed.
    pub fn project_physical_dense(
        &self,
        engine: &mut dyn Engine,
        physical: &PhysicalDense<D>,
    ) -> Result<Self, PhysicalDenseError<<R as PhysicalFusionBasis>::Error>> {
        symtn_operations::host_pool::region(
            engine,
            symtn_operations::host_pool::HostPoolSite::Strided,
            |_| (),
        )?;
        let data = project_physical_host(
            self.logical_space(),
            physical.shape.as_slice(),
            physical.data.as_slice(),
        )?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(self.logical_space().clone(), data)),
        })
    }
}
