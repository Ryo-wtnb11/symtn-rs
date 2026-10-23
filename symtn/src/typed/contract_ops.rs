use super::*;

impl<R, D, S> TensorMap<R, D, S> {
    /// The storage operand a contraction stages and plans with: an owned
    /// tensor direct, a lazy adjoint its parent read storage-conjugate.
    pub(super) fn fusion_operand(&self) -> symtn_tensors::FusionOperand<'_> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => {
                symtn_tensors::FusionOperand::direct(body.space.space())
            }
            TypedTensorRepr::Adjoint(view) => {
                symtn_tensors::FusionOperand::adjoint(view.parent.space.space())
            }
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    pub(super) fn fusion_operand_and_data(
        &self,
    ) -> (symtn_tensors::FusionOperand<'_>, std::borrow::Cow<'_, [D]>) {
        match &self.repr {
            TypedTensorRepr::Owned(body) => (
                symtn_tensors::FusionOperand::direct(body.space.space()),
                body.materialized_dense_data(),
            ),
            TypedTensorRepr::Adjoint(view) => (
                symtn_tensors::FusionOperand::adjoint(view.parent.space.space()),
                std::borrow::Cow::Borrowed(view.parent_data()),
            ),
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorProductDispatch<R, D>,
    D: TensorScalar,
{
    /// Tensor product in one category, ordered as
    /// `codomain(self), codomain(other); domain(self), domain(other)`.
    ///
    /// The two codomain trees and the two domain trees are merged
    /// independently with F moves. No legs cross and no R symbol is needed,
    /// including for a `NoBraiding` provider.
    ///
    /// Equal provider identities are sufficient; the two tensors may own
    /// different `Arc` allocations. The output always retains `self`'s exact
    /// provider allocation. A compact diagonal operand is densified into an
    /// operation-local buffer first; the output is dense either way.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`] is reported before provider work. Checked
    /// Generic providers preserve algebra and malformed-F failures in
    /// [`GenericTensorError::TensorProduct`].
    ///
    /// ```compile_fail
    /// use symtn::sector::TableRule;
    /// use symtn::typed::TensorMap;
    /// fn unavailable(tensor: &TensorMap<TableRule, f64>) {
    ///     let _ = tensor.otimes(tensor);
    /// }
    /// ```
    ///
    /// An adjoint view `other` (`t.adjoint_view()`) returns
    /// [`Error::Unsupported`]: this operation would copy it. Pass
    /// `&t.adjoint()?.materialize()?` instead.
    pub fn otimes<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D>>,
    ) -> Result<Self, FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        other.refuse_borrowed_view("otimes")?;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        let lhs_owned = self.materialized_tensor_uncached()?;
        let rhs_owned = other.materialized_tensor_uncached()?;
        let (Some(lhs_body), Some(rhs_body)) = (lhs_owned.owned_body(), rhs_owned.owned_body())
        else {
            return Err(internal_layout_error("uncached materialization is owned").into());
        };
        let (space, data) = <R::Mode as TypedTensorProductDispatch<R, D>>::product(
            &lhs_body.space,
            lhs_body.materialized_dense_data().as_ref(),
            &rhs_body.space,
            rhs_body.materialized_dense_data().as_ref(),
        )?;
        Ok(self.published(space, data))
    }
}

/// The leg roles of a pairwise contraction (TensorOperations
/// `tensorcontract!`'s `pA[2]`, `pB[1]` and `pAB`): which legs are contracted,
/// and how the open legs are ordered and split into the result's codomain and
/// domain.
///
/// Open legs are numbered `0..open_rank`, the open legs of the left operand in
/// ascending axis order first, then those of the right operand.
/// `codomain ++ domain` must be a permutation of `0..open_rank`.
///
/// The result is defined as the contraction that puts every open leg of the
/// left operand in the codomain and every open leg of the right one in the
/// domain, followed by [`TensorMap::permute`] onto `(codomain, domain)`; the
/// operation performs it without that separate permute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContractSpec<'a> {
    /// Contracted axes of the left operand, paired in order with `rhs`.
    pub lhs: &'a [usize],
    /// Contracted axes of the right operand.
    pub rhs: &'a [usize],
    /// Open legs forming the result's codomain, in order.
    pub codomain: &'a [usize],
    /// Open legs forming the result's domain, in order.
    pub domain: &'a [usize],
}

impl ContractSpec<'_> {
    /// `codomain ++ domain`, the single output order the engine takes with
    /// [`Self::codomain`]'s length as the split.
    pub(super) fn output_axes(&self) -> SmallVec<[usize; 8]> {
        self.codomain.iter().chain(self.domain).copied().collect()
    }
}

/// The staged request of `spec` between `lhs` and `rhs`, whose output order
/// is `output_axes` (`spec.output_axes()`) and whose conjugation flags are
/// the operands': the ordinary contraction, or with `planar` the planar one
/// over those admitted orders.
pub(super) fn contract_request<'a>(
    lhs: symtn_tensors::FusionOperand<'_>,
    rhs: symtn_tensors::FusionOperand<'_>,
    spec: &ContractSpec<'a>,
    output_axes: &'a [usize],
    planar: Option<&'a symtn_tensors::PlanarContractOrders>,
) -> symtn_tensors::ContractRequest<'a> {
    let axes = symtn_tensors::TensorContractSpec::new_with_conjugation(
        spec.lhs,
        spec.rhs,
        OutputAxisOrder::from_axes(output_axes),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    let codomain_rank = spec.codomain.len();
    match planar {
        None => symtn_tensors::ContractRequest::Contract {
            axes,
            codomain_rank,
        },
        Some(orders) => symtn_tensors::ContractRequest::Planar {
            axes,
            codomain_rank,
            orders,
        },
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
    S: ContractExec<R, D>,
{
    /// Contracts `spec.lhs` of `self` with `spec.rhs` of `other` (pairwise, in
    /// list order) and returns the open legs as `spec.codomain ← spec.domain`
    /// (TensorKit `tensorcontract!` with `pAB = (codomain, domain)`).
    ///
    /// The result equals the contraction whose codomain is every open axis of
    /// `self` and whose domain is every open axis of `other`, followed by
    /// [`Self::permute`] onto `(spec.codomain, spec.domain)` — see
    /// [`ContractSpec`]. A leg moved across the split is dualized exactly as
    /// that permute dualizes it.
    ///
    /// **Cost.** No permute pass follows the contraction. As in TensorKit's
    /// `blas_contract!`, the GEMMs write the result directly when its layout
    /// allows, and otherwise write a Runtime-pooled temporary that one tree
    /// transform moves into the result, inside this call — never slower than
    /// the contraction followed by the explicit permute.
    ///
    /// **Braiding scope**: ordinary contraction is available only for
    /// symmetric braiding (Bosonic, Fermionic), as TensorKit `blas_contract!`.
    /// An anyonic or unbraided (`NoBraiding`) provider is rejected even when
    /// these axes are the canonical, crossing-free ones: general axes are
    /// defined by braiding legs into place, and the call carries no planar
    /// embedding or braid direction. The crossing-free case is exactly
    /// [`Self::compose`], which every braiding style admits; general planar
    /// contraction is [`Self::planar_contract`] (TensorKit `@planar`).
    /// Behaviour change (#1372):
    /// a canonical `NoBraiding` contraction was accepted before.
    ///
    /// For fermionic symmetric braiding this **twists**
    /// dual contracted legs with the fermionic supertrace twist — unlike
    /// composition (TensorKit `A * B` / `mul!`), which never does. Bosonic
    /// rules are unaffected; fermionic rules can differ by signs.
    /// [`Self::compose`] is the other semantics, and its documentation states
    /// the exact relation between the two.
    ///
    /// # Compact fast paths
    ///
    /// Contracting **one** leg against a factor in compact diagonal storage —
    /// an `s` from [`Self::svd_compact`], a `d` from [`Self::eigh_full`] — is
    /// a per-leg bond scaling and is run as one: the other operand's contracted
    /// leg is multiplied by the spectrum in place of a GEMM, and the result is
    /// laid out with a single [`Self::permute`]. Mathematically it is the same
    /// tensor the dense route computes, so this is a cost question only, and any
    /// pattern that does not fit falls through to the dense route rather than
    /// being refused.
    ///
    /// **Complexity.** In `docs/complexity_parity_policy.md`'s parameters — `d`
    /// the per-sector bond degeneracy, `n` the other operand's *open*-leg size,
    /// so its blocks hold `d·n` entries — the dense route materializes the
    /// spectrum as a `Σ_c d_c²` block-diagonal buffer and multiplies it in, at
    /// O(d²) storage and O(d²·n) work. The scaling route touches each of those
    /// `d·n` entries once, at O(d) storage and O(d·n) work, which is the order
    /// that policy's row requires. `D · D` multiplies the two spectra
    /// elementwise and stays compact, at O(d).
    ///
    /// **TensorKit correspondence.** This is what TensorKit's
    /// `DiagonalTensorMap` gets from its type: `block(D, c)` is a `Diagonal`, so
    /// LinearAlgebra dispatches the multiplication to `lmul!`/`rmul!` scaling
    /// (`diagonal.jl`), with no braiding or recoupling of its own.
    ///
    /// **Which patterns.** Exactly the two geometries that are a composition on
    /// the contracted leg, in either order — the contracted leg of the compact
    /// operand is its bond, and the other operand's is a leg on the side that
    /// faces it (`t`'s domain against `D`'s codomain, or `D`'s domain against a
    /// codomain leg of `t`, at any position). A `D · D` product in any output
    /// order is the product spectrum followed by one [`Self::permute`]: a
    /// swapped result stays compact (TensorKit's `DiagonalTensorMap`), and a
    /// result with both legs on one side is dense. A leg on the far side or
    /// more than one contracted leg takes the dense route: they are not proved
    /// geometries. A supertrace twist on a dual contracted leg of `other`
    /// would also decline, and cannot currently arise — see
    /// `diagonal_contract_pattern`.
    ///
    /// Both admission modes take these arms. The reordering permute is the
    /// mode's own tree transform, so in checked Generic mode a provider
    /// failure there is reported as [`GenericTensorError::Plan`] from that
    /// permute staging, in place of the contraction planning the dense route
    /// would have run. One outcome is storage-dependent, because the arm
    /// answers what TensorKit computes where the dense route is unsupported:
    /// a checked Generic fermionic provider (dense checked contraction
    /// requires Bosonic braiding). A symmetric rule with complex symbols
    /// takes both routes, like a real-symbol one.
    ///
    /// The result is bound to `self`'s provider allocation, the same
    /// left-authority rule [`Self::zeros`] uses for its first leg: the two
    /// operands must agree on
    /// [`crate::sector::FusionRule::rule_identity`], which makes the choice of
    /// allocation immaterial to the algebra.
    ///
    /// # CUDA storage
    ///
    /// Device tensors run this same body: the destination, the route, the
    /// orientation and axis-order candidate, the source and output transform
    /// structures, the operand borrow decisions and the core plan all come
    /// from the one planner, with the device executor's capabilities (fully
    /// direct cores only). The device only replays that route — each
    /// non-identity source transform into device scratch, the coupled-sector
    /// GEMMs, the output transform — which is TensorKit's `blas_contract!`
    /// and QSpace's `contract` dataflow. A contraction whose canonical core
    /// resolves from the operands alone takes no Host context lease.
    ///
    /// **Cost.** A warm call uploads nothing but the output initialisation
    /// (#740): exactly one H2D of `required_len * size_of::<D>()` bytes, from
    /// one host `Vec` of that size; its only device allocation is the
    /// returned output; it downloads nothing. The first use of a transform
    /// structure uploads its coefficient payload once (as the device
    /// [`Self::permute`] does), and a new high-water mark of the operand or
    /// core-destination scratch costs one zero upload of the new size per
    /// buffer; both then stay resident on the Runtime (see
    /// `Runtime::cuda_contract_scratch_bytes`) until the
    /// `Runtime::clear_cuda_caches`. Per
    /// call the device submits one region move per Single block / pack /
    /// scatter column, one GEMM per recoupling job and per coupled-sector
    /// job, one zero fill per inactive layout of each transform in overwrite
    /// mode, and one zero fill per core-destination block no GEMM writes (the
    /// Host clears its whole core destination instead). The warm contract
    /// holds while this Runtime's Host transform store admits the structures
    /// (see [`Self::permute`]).
    ///
    /// **Numerics.** Device and Host agree to dtype tolerance, never bitwise:
    /// the GEMM summation order is cuTENSOR's, a Single-block move rounds as
    /// `alpha * (c * x)`, and over a non-canonical expert tiling the Host
    /// packs around its core where the device runs the transform route. Exact
    /// zeros may differ in sign (Host `-0.0` where the device writes `+0.0`).
    ///
    /// **Fermionic twist.** The canonical form with a twist uniform per
    /// coupled sector folds it into the GEMM alpha; every other case folds
    /// `θ_b` into the descriptor alpha of the source-transform move writing
    /// core-right block `b`, as the Host folds it into the same move.
    ///
    /// [`Error::PlacementMismatch`] for an operand on another device is
    /// reported last, under the device lease, before any device work.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes.
    /// - [`Error::Operation`] with
    ///   [`crate::typed::OperationError::UnsupportedTensorContractScope`] for
    ///   non-symmetric (anyonic or `NoBraiding`) providers, whatever the axes.
    /// - [`Error::Operation`] with
    ///   [`crate::typed::OperationError::InvalidAxisSet`] (`tensor` `"lhs"` /
    ///   `"rhs"`) for a malformed contracted-axis list, and with
    ///   [`crate::typed::OperationError::InvalidPermutation`] for a
    ///   `codomain ++ domain` that is not a permutation of the open axes.
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] for
    ///   mismatched contracted legs, or operands whose
    ///   providers report different rule identities. Those all come back from
    ///   the expert layer, which owns the rules; re-checking them here would
    ///   be a second copy free to drift.
    ///   Checked Generic providers preserve provider and replay failures in
    ///   [`GenericTensorError::Plan`] and currently accept direct-owned inputs.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&runtime, [&v], [&v], 8)?;
    /// let id = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    ///
    /// // Contracting the domain leg with the identity's codomain leg is a
    /// // no-op on the payload; `[0] ← [1]` keeps the open axes in place.
    /// let spec = ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0], domain: &[1] };
    /// let out = t.contract(&id, &spec)?;
    /// assert_eq!(out.dense_data()?, t.dense_data()?);
    ///
    /// // Both open legs in the codomain: the same as permuting afterwards.
    /// let spec = ContractSpec { lhs: &[1], rhs: &[0], codomain: &[0, 1], domain: &[] };
    /// assert_eq!(out.permute(&[0, 1], &[])?.dense_data()?, t.contract(&id, &spec)?.dense_data()?);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn contract<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, S>>,
        spec: &ContractSpec<'_>,
    ) -> Result<Self, FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        // The one check the expert layer cannot make: it never sees the two
        // runtimes, and mixing execution state across them is a trust-boundary
        // violation rather than an algebra error. Scalar type and placement
        // need no arm here — `D` and `S` are type parameters.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        reject_non_symmetric_contraction(<R::Mode as TypedTensorModeDispatch<R>>::braiding_style(
            self.provider(),
        ))
        .map_err(Error::from)?;
        if let (Some(lhs), Some(rhs)) = (S::as_host(EXEC, self), S::as_host(EXEC, other)) {
            if lhs.spectrum().is_some() || rhs.spectrum().is_some() {
                if let Some(compact) = lhs.try_contract_diagonal(
                    rhs,
                    spec.lhs,
                    spec.rhs,
                    &spec.output_axes(),
                    spec.codomain.len(),
                )? {
                    return Ok(S::from_host(EXEC, compact)?);
                }
            }
        }
        let (space, data) =
            <R::Mode as TypedTensorContractDispatch<R, D>>::contract(self, other, spec, None)?;
        Ok(self.published(space, data))
    }

    /// `destination = alpha * self.contract(other, spec) + beta * destination`,
    /// TensorKit `tensorcontract!(C, A, pA, false, B, pB, false, pAB, α, β)`,
    /// preserving the destination's provider, space, body and dense host
    /// allocation. `destination` must have the space of that result, including
    /// its codomain/domain split. Every admission mode and operand
    /// representation [`Self::contract`] takes is taken.
    ///
    /// `alpha` and `beta` ride the epilogue of whatever writes each element:
    /// the core GEMMs (TensorKit's `mul!(C, A, B, α, β)`) or an output
    /// transform's add (`tensoradd!(C, Cnew, pAB, false, α, β)`). `beta == 0`
    /// never reads `destination` (NaN does not survive; unreached blocks
    /// become `+0`), and `alpha == 0` never reads the operands. Nothing clears
    /// the destination first.
    ///
    /// A coupled sector no GEMM reaches: on the core route, where the GEMMs
    /// write the destination directly, it becomes `beta * destination`
    /// (TensorKit's `rmul!(C, β)`), so `beta == 1` leaves it bit for bit. On a
    /// route with an output transform (the one-call route's C transform, or
    /// copyC below) the transform writes every element, so an unreached one
    /// becomes `alpha * (+0) + beta * destination`: IEEE addition turns a
    /// `-0.0` into `+0.0` even for `beta == 1`, and a non-finite `alpha`
    /// gives NaN — as TensorKit's `tensoradd!`, and as the eager
    /// [`Self::contract`] followed by [`Self::axpby`].
    ///
    /// **Route.** The same memcost choice as [`Self::contract`]: when a
    /// zero-copy candidate plus one output permute is cheaper, the product is
    /// written with its own output order into Runtime-pooled scratch and one
    /// tree transform adds it into `destination` with `alpha`/`beta`
    /// (TensorKit `blas_contract!`'s `copyC`); no operand is rebuilt.
    ///
    /// **Compact operand.** [`Self::contract`]'s compact patterns write into
    /// `destination` without densifying (TensorKit `mul!` scaling a
    /// `Diagonal` block): in the scaled operand's own order, one pass
    /// `beta * destination + alpha * scaled`; otherwise one operation-local
    /// buffer of the scaled operand and one [`Self::permute_into`]; `D · D`
    /// as the product spectrum read by [`Self::permute_into`], in any output
    /// order. Any other pattern densifies the compact operand into an
    /// operation-local buffer, where TensorKit copies it too.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; then
    /// [`crate::typed::OperationError::UnsupportedTensorContractScope`] for
    /// non-symmetric (anyonic or `NoBraiding`) providers, as for
    /// [`Self::contract`]; [`Error::RuleMismatch`];
    /// [`Error::InvalidArgument`] for a destination that is not owned dense
    /// storage of its placement or aliases an operand (a lazy adjoint's
    /// parent included); then the contraction's own errors, as for
    /// [`Self::contract`]; `SpaceMismatch` for a destination of the wrong
    /// space or layout; [`Error::InvalidArgument`] for one of the wrong
    /// length; [`Error::DestinationShared`] when `destination` shares its
    /// storage with a clone; on CUDA storage, [`Error::PlacementMismatch`]
    /// last. Runtime-context leasing counts as validation.
    /// Checked Generic reports these in [`GenericTensorError::Facade`] and
    /// staging failures in [`GenericTensorError::Plan`], as for
    /// [`Self::contract`].
    ///
    /// # CUDA storage
    ///
    /// The same body and route as [`Self::contract`]. On the core route the
    /// coupled sectors no GEMM reaches become `beta * destination` through a
    /// zero-source region move — zeros for `beta = 0`, which never reads the
    /// destination, and nothing for `beta = 1`. No destination reset or
    /// separate `beta` pass over written blocks; a warm call transfers
    /// nothing and allocates nothing on the device; scratch and coefficient
    /// payloads are as for [`Self::contract`].
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn contract_into<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, S>>,
        spec: &ContractSpec<'_>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime)
            || !self.runtime.same_runtime(&destination.runtime)
        {
            return Err(Error::RuntimeMismatch.into());
        }
        reject_non_symmetric_contraction(<R::Mode as TypedTensorModeDispatch<R>>::braiding_style(
            self.provider(),
        ))
        .map_err(Error::from)?;
        // The held admitted identities: provider-free, and the provider's
        // own by #2046.
        let identity = self.logical_space().space().admission().rule_identity();
        if identity != other.logical_space().space().admission().rule_identity()
            || identity
                != destination
                    .logical_space()
                    .space()
                    .admission()
                    .rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        // Why the storage bodies: a lazy-adjoint operand aliases its parent.
        destination_precheck(destination, &self.storage_body().data, S::NOUN)?;
        destination_precheck(destination, &other.storage_body().data, S::NOUN)?;
        if let (Some(lhs), Some(rhs), Some(dst)) = (
            S::as_host(EXEC, self),
            S::as_host(EXEC, other),
            S::as_host_mut(EXEC, destination),
        ) {
            if (lhs.spectrum().is_some() || rhs.spectrum().is_some())
                && lhs.try_contract_diagonal_into(rhs, spec, dst, alpha, beta)?
            {
                return Ok(());
            }
        }
        <R::Mode as TypedTensorContractDispatch<R, D>>::contract_into(
            self,
            other,
            spec,
            None,
            destination,
            alpha,
            beta,
        )
    }

    /// The admitted orders of a planar `spec` against `other`, after the
    /// checks `planar_contract` shares with `contract`: placement (Host
    /// only) and the planarity of `spec`.
    fn planar_orders(
        &self,
        other: &Self,
        spec: &ContractSpec<'_>,
    ) -> Result<symtn_tensors::PlanarContractOrders, Error> {
        if S::as_host(EXEC, self).is_none() || S::as_host(EXEC, other).is_none() {
            return Err(Error::UnsupportedOnDevice(
                "planar_contract has no device route; transfer with to_host()".to_owned(),
            ));
        }
        let shape = |tensor: &Self| {
            let space = tensor.logical_space().space();
            (space.nout(), space.rank())
        };
        Ok(symtn_tensors::PlanarContractOrders::validate(
            shape(self),
            shape(other),
            spec.lhs,
            spec.rhs,
            &spec.output_axes(),
            spec.codomain.len(),
        )?)
    }

    fn is_bosonic(&self) -> bool {
        <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider())
            == symtn_core::BraidingStyleKind::Bosonic
    }

    /// Planar contraction, TensorKit `@planar` for one pairwise contraction
    /// (`planarcontract!` plus its output `planaradd!`): contracts
    /// `spec.lhs` of `self` with `spec.rhs` of `other` without crossing any
    /// legs, and returns the open legs as `spec.codomain ← spec.domain`, in
    /// [`Self::contract`]'s leg roles ([`ContractSpec`]). Every braiding
    /// style is admitted, `Anyonic` and `NoBraiding` included: no R symbol is
    /// used.
    ///
    /// **Planarity.** A tensor's cyclic boundary is its codomain axes in
    /// order followed by its domain axes reversed, as for
    /// [`Self::transpose`]. `spec` is planar when `spec.lhs` is one arc of
    /// `self`'s boundary and `spec.rhs` one arc of `other`'s, the pairing
    /// walks the two arcs in opposite directions, and `spec.codomain`
    /// followed by `spec.domain` reversed is a rotation of the boundary the
    /// two open arcs leave (`self`'s open legs from just after its arc, then
    /// `other`'s). The list order of `spec.lhs`/`spec.rhs` is free; only the
    /// pairs matter. Like TensorKit's `@planar`, an output rotation is
    /// admitted and fused into the call; `planarcontract!` alone asserts the
    /// unrotated split.
    ///
    /// **Semantics by braiding style.** The result is `transpose(self)` onto
    /// `(open | contracted)`, `transpose(other)` onto `(contracted | open)`,
    /// [`Self::compose`], then one [`Self::transpose`] onto the requested
    /// legs. For a Bosonic rule this is exactly [`Self::contract`] (a cyclic
    /// transpose is a braid there), and the call runs `contract`'s planner
    /// after the planarity check. For a **Fermionic** rule it is the
    /// composition semantics, not the contraction: no supertrace twist is
    /// applied, so it equals `compose` whenever `spec` is the composition
    /// geometry and differs from `contract` by a sign on every dual
    /// contracted leg carrying an odd sector (see [`Self::compose`]).
    /// Planarity is checked for every style, Bosonic included (stricter than
    /// `planarcontract!`, which skips it for Bosonic, and equal to `@planar`).
    ///
    /// **Cost.** The owned output is the only allocation; each operand
    /// transpose writes Runtime-pooled scratch and is skipped when the
    /// operand already has its planar core layout (TensorKit `A′ = A`), and
    /// the output transpose is skipped when the requested order is the
    /// core's. The work is TensorKit's `transpose!` ×2 + `mul!` +
    /// `planaradd!`: the transposes' recoupling terms (F, pivotal and
    /// Frobenius–Schur coefficients only) plus `Σ_c m_c n_c k_c` GEMM work.
    ///
    /// Host storage only; checked Generic providers are admitted for Bosonic
    /// braiding (the ordinary contraction).
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes.
    /// - [`Error::UnsupportedOnDevice`] for CUDA storage.
    /// - [`Error::Operation`] with the malformed-spec errors of
    ///   [`Self::contract`]; then
    ///   [`crate::typed::OperationError::InvalidAxisSet`] (`"lhs"` / `"rhs"`)
    ///   when a contracted set is not one boundary arc or the pairing does
    ///   not reverse orientation, and
    ///   [`crate::typed::OperationError::InvalidPermutation`] when the output
    ///   is not a rotation of the open boundary. All are reported before any
    ///   provider work.
    /// - Then the errors of [`Self::contract`] for the contracted legs and
    ///   rule identities; a non-Bosonic checked Generic provider is
    ///   [`crate::typed::OperationError::UnsupportedTensorContractScope`].
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use num_complex::Complex64;
    /// use symtn::sector::{TableRule, TableSector};
    /// use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let fib = Arc::new(TableRule::fibonacci());
    /// let v = GradedSpace::try_new(Arc::clone(&fib), [(TableSector::new(2), 1)])?;
    /// let b = GradedSpace::try_new(fib, [(TableSector::new(1), 2), (TableSector::new(2), 3)])?;
    /// // Golden-chain two-site H_eff: θ′[l s1 s2; r] := H[s1 s2; t1 t2] θ[l t1 t2; r].
    /// let h: TensorMap<_, Complex64> = TensorMap::rand_with_seed(&runtime, [&v, &v], [&v, &v], 1)?;
    /// let theta: TensorMap<_, Complex64> =
    ///     TensorMap::rand_with_seed(&runtime, [&b, &v, &v], [&b], 2)?;
    /// // Open legs: s1 = 0, s2 = 1 (of h), l = 2, r = 3 (of θ).
    /// let spec = ContractSpec { lhs: &[2, 3], rhs: &[1, 2], codomain: &[2, 0, 1], domain: &[3] };
    /// let out = h.planar_contract(&theta, &spec)?;
    /// assert_eq!(out.codomain_rank(), 3);
    ///
    /// // The same as transposing θ by hand and composing, then transposing.
    /// let by_hand = h
    ///     .compose(&theta.transpose(&[1, 2], &[0, 3])?)?
    ///     .transpose(&[2, 0, 1], &[3])?;
    /// let residual = out.axpby(Complex64::ONE, &by_hand, -Complex64::ONE)?.norm(2.0)?;
    /// assert!(residual < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn planar_contract<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, S>>,
        spec: &ContractSpec<'_>,
    ) -> Result<Self, FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        let orders = self.planar_orders(other, spec)?;
        if self.is_bosonic() {
            return self.contract(other, spec);
        }
        let (space, data) = <R::Mode as TypedTensorContractDispatch<R, D>>::contract(
            self,
            other,
            spec,
            Some(&orders),
        )?;
        Ok(self.published(space, data))
    }

    /// `destination = alpha * self.planar_contract(other, spec) + beta *
    /// destination`, TensorKit `planarcontract!(C, A, pA, B, pB, pAB, α, β)`
    /// (with `@planar`'s output rotation fused in), preserving the
    /// destination's provider, space, body and dense host allocation.
    /// `destination` must have the space of that result. It takes every
    /// operand [`Self::planar_contract`] takes and allocates nothing owned;
    /// scratch is Runtime-pooled.
    ///
    /// `alpha` and `beta` ride the epilogue of whatever writes each element
    /// (the core GEMMs, as `mul!(C, A′, B′, α, β)`, or the output transpose),
    /// with [`Self::contract_into`]'s rules: `beta == 0` never reads
    /// `destination`, `alpha == 0` never reads the operands. A Bosonic rule
    /// runs [`Self::contract_into`] after the planarity check.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; [`Error::RuleMismatch`];
    /// [`Error::InvalidArgument`] for a destination that is not owned dense
    /// storage or aliases an operand; [`Error::UnsupportedOnDevice`] for CUDA
    /// storage; then the planarity errors of [`Self::planar_contract`]; then the
    /// remaining errors of [`Self::contract_into`] (`SpaceMismatch`, length,
    /// [`Error::DestinationShared`]).
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write.
    pub fn planar_contract_into<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, S>>,
        spec: &ContractSpec<'_>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        if !self.runtime.same_runtime(&other.runtime)
            || !self.runtime.same_runtime(&destination.runtime)
        {
            return Err(Error::RuntimeMismatch.into());
        }
        let identity = self.logical_space().space().admission().rule_identity();
        if identity != other.logical_space().space().admission().rule_identity()
            || identity
                != destination
                    .logical_space()
                    .space()
                    .admission()
                    .rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        destination_precheck(destination, &self.storage_body().data, S::NOUN)?;
        destination_precheck(destination, &other.storage_body().data, S::NOUN)?;
        let orders = self.planar_orders(other, spec)?;
        if self.is_bosonic() {
            return self.contract_into(other, spec, destination, alpha, beta);
        }
        <R::Mode as TypedTensorContractDispatch<R, D>>::contract_into(
            self,
            other,
            spec,
            Some(&orders),
            destination,
            alpha,
            beta,
        )
    }
}

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// TensorKit `deligneproduct`: embeds `self` as `(a, 𝟙)` and `other` as
    /// `(𝟙, b)` in the supplied ordered product category, then combines the
    /// embedded tensors with the F-only [`Self::otimes`] route.
    ///
    /// This operation is typed-only and keeps the payload type `D` unchanged.
    /// The caller supplies the exact [`ProductFusionRule`], including its
    /// component providers and codec; both component [`crate::sector::RuleIdentity`] values
    /// must match the operands, and the codec participates in the product
    /// identity. [`CanonicalUnitFusionRule`] is required for both components
    /// because symtn stores no separate unitor data. Factor order and nested
    /// association are preserved exactly rather than reassociated or swapped.
    ///
    /// Validation reports [`Error::RuntimeMismatch`] before component
    /// [`Error::RuleMismatch`]. Both vacuum embeddings, codec decodes, and
    /// source/target fusion-tree bijections are prepared before either
    /// embedded `TensorMap` or layout is published. After that transaction
    /// succeeds, the operation builds the two embedded tensors by copying
    /// their dense data (materializing a compact operand when necessary).
    ///
    /// An adjoint view `other` (`t.adjoint_view()`) returns
    /// [`Error::Unsupported`]: this operation would copy it. Pass
    /// `&t.adjoint()?.materialize()?` instead.
    pub fn deligne_product<'a, R2, C>(
        &self,
        other: impl Into<TensorRef<'a, R2, D>>,
        product: Arc<ProductFusionRule<R, R2, C>>,
    ) -> Result<TensorMap<ProductFusionRule<R, R2, C>, D>, Error>
    where
        R: TypedSectorAdmission<
                Sector = <R as SectorCodec>::Sector,
                Error = FusionAlgebraError,
                Mode = MultiplicityFreeAdmissionMode,
            > + CanonicalUnitFusionRule,
        R2: TypedSectorAdmission<
                Sector = <R2 as SectorCodec>::Sector,
                Error = FusionAlgebraError,
                Mode = MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + CanonicalUnitFusionRule,
        C: ProductSectorCodec + Sync + 'static,
    {
        let other = other.into().operand()?;
        let other = &*other;
        other.refuse_borrowed_view("deligne_product")?;
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        if product.left_rule().rule_identity() != self.logical_space().provider().rule_identity()
            || product.right_rule().rule_identity()
                != other.logical_space().provider().rule_identity()
        {
            return Err(Error::RuleMismatch);
        }
        let left_vacuum = product
            .left_rule()
            .decode_sector(product.left_rule().vacuum())?;
        let right_vacuum = product
            .right_rule()
            .decode_sector(product.right_rule().vacuum())?;
        let left = prepare_product_operand(
            self,
            Arc::clone(&product),
            |sector| ProductSector::new(sector, right_vacuum.clone()),
            |sector| sector.left().clone(),
        )?;
        let right = prepare_product_operand(
            other,
            product,
            |sector| ProductSector::new(left_vacuum.clone(), sector),
            |sector| sector.right().clone(),
        )?;
        let left = left.commit()?;
        let right = right.commit()?;
        left.otimes(&right)
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
    S: ContractExec<R, D>,
{
    /// Categorical composition of two tensor maps, TensorKit `A * B` / `mul!`:
    /// `self`'s whole domain is contracted against `other`'s whole codomain,
    /// leaving `self.codomain() <- other.domain()`.
    ///
    /// **Fermionic semantics**: unlike [`Self::contract`] (TensorKit
    /// `tensorcontract!` / `@tensor`), composition never twists dual
    /// contracted legs — there is no supertrace here. Bosonic rules cannot
    /// tell the two apart; a fermionic one differs by a sign on every dual
    /// contracted leg carrying an odd sector, so the exact relation is
    /// `self.compose(other) == self.contract(twist(other, other's dual
    /// codomain legs), ..)`. Reach for `compose` when you mean operator
    /// multiplication of tensor maps, and for `contract` when you mean
    /// index-notation contraction.
    ///
    /// **Anyonic and unbraided semantics**: composition remains
    /// coupled-sector block multiplication for every braiding style,
    /// `NoBraiding` included, as TensorKit `mul!` checks none. The fixed
    /// domain/codomain boundary supplies the whole geometry, so no legs are
    /// exchanged and no R symbol is used; this is why `compose` is admitted
    /// where [`Self::contract`] rejects the same canonical axes.
    ///
    /// The axes are not arguments, deliberately: composition is defined by the
    /// codomain/domain split itself, and TensorKit's `*` takes none.
    ///
    /// The result is bound to `self`'s provider allocation — the same
    /// left-authority rule as [`Self::contract`] and [`Self::zeros`] — with one
    /// exemption: the `D * t` compact arm below returns `t`'s own space and
    /// runtime handle, because that space *is* the destination and rebuilding
    /// it under the left allocation would be a copy for nothing. The two
    /// allocations must already agree on
    /// [`crate::sector::FusionRule::rule_identity`] for the composition to be
    /// legal at all, so the choice is immaterial to the algebra.
    ///
    /// # Compact fast paths
    ///
    /// When either operand carries compact diagonal storage — an `s` from
    /// [`Self::svd_compact`], a `d` from [`Self::eigh_full`] — and the
    /// destination is representable, this takes TensorKit's
    /// `DiagonalTensorMap` route instead of a GEMM: `t * D` and `D * t` scale
    /// one bond axis per block (`rmul!` / `lmul!`), and `D * D` multiplies the
    /// two spectra elementwise and stays compact. Verified twist-free against
    /// TK's `diagonal.jl`: `block(D, c)` is a `Diagonal`, so LinearAlgebra
    /// dispatches to scaling, with no braiding or recoupling. The result is the
    /// same tensor the dense route computes, so this is a cost question only,
    /// and any operand or destination that does not fit falls through to the
    /// dense path rather than being refused. That path densifies a compact
    /// operand into an operation-local buffer first. Both admission modes and
    /// both coefficient lanes take these arms.
    ///
    /// # Errors
    ///
    /// - [`Error::RuntimeMismatch`] when the operands belong to different
    ///   runtimes, as for [`Self::contract`].
    /// - [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`]
    ///   when the two are not composable — mismatched ranks, legs that are not
    ///   mutually dual, or providers reporting different rule identities.
    ///   Those come back from the expert layer, which owns the rules.
    ///   Checked Generic failures use [`GenericTensorError::Plan`].
    ///
    /// On CUDA storage the composition plans for the device executor, which
    /// runs only fully-direct cores, so a non-canonical expert tiling the
    /// Host packs is [`crate::typed::OperationError::UnsupportedTensorContractScope`]
    /// there; [`Error::PlacementMismatch`] comes last, as for
    /// [`Self::contract`].
    ///
    /// ```compile_fail
    /// use symtn::sector::TableRule;
    /// use symtn::typed::TensorMap;
    /// fn unavailable(tensor: &TensorMap<TableRule, f64>) {
    ///     let _ = tensor.compose(tensor);
    /// }
    /// ```
    #[doc(alias = "mul")]
    pub fn compose<'a>(
        &self,
        other: impl Into<TensorRef<'a, R, D, S>>,
    ) -> Result<Self, FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        // Runtime first, exactly as `contract`: crossing runtimes is a
        // trust-boundary violation rather than an algebra error, and the
        // expert layer never sees the two runtimes.
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        if let (Some(lhs), Some(rhs)) = (S::as_host(EXEC, self), S::as_host(EXEC, other)) {
            if lhs.spectrum().is_some() || rhs.spectrum().is_some() {
                if let Some(compact) = lhs.compose_compact(rhs)? {
                    return Ok(S::from_host(EXEC, compact)?);
                }
            }
        }
        let (space, data) = <R::Mode as TypedTensorContractDispatch<R, D>>::compose(self, other)?;
        Ok(self.published(space, data))
    }
}

type RouteReplay<D, C> = fn(
    &mut crate::runtime::TensorExecutionContext,
    &symtn_tensors::StorageContractResolution<C>,
    (&Arc<BlockStructure>, &mut [D]),
    (&Arc<BlockStructure>, &[D]),
    (&Arc<BlockStructure>, &[D]),
    D,
    symtn_tensors::ContractDestinationInit<D>,
) -> Result<(), TransformExecError>;

fn host_route_replay<L, D, C>(
    context: &mut crate::runtime::TensorExecutionContext,
    resolution: &symtn_tensors::StorageContractResolution<C>,
    dst: (&Arc<BlockStructure>, &mut [D]),
    lhs: (&Arc<BlockStructure>, &[D]),
    rhs: (&Arc<BlockStructure>, &[D]),
    alpha: D,
    init: symtn_tensors::ContractDestinationInit<D>,
) -> Result<(), TransformExecError>
where
    L: HostTreeLane<D, C>,
    D: symtn_tensors::DenseRecouplingScalar
        + RecouplingCoefficientAction<C>
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    L::contract_lane(context)?
        .execute_contract_resolution_host(resolution, dst, lhs, rhs, alpha, init)
        .map_err(TransformExecError::Replay)
}

/// One planned contraction route, handed to the storage kernel by value: the
/// pooled context lease it was planned under (`None` when the lock-free core
/// answered and no lease was taken), the route, and the staged destination
/// and stored operand block structures it maps between.
///
/// Host replays under that lease, taking one only when planning took none,
/// so a Host contraction takes one lease; CUDA releases it before taking the
/// device lease. The Host replay of the staging mode's lane is bound here,
/// where that lane's scalar bounds are proved.
///
/// Internal and unstable despite being public for [`ContractExec`].
#[doc(hidden)]
pub struct ContractReplay<'a, D, C> {
    pub(super) runtime: &'a Runtime,
    pub(super) lease: Option<crate::runtime::ContextLease<'a>>,
    execute: RouteReplay<D, C>,
    pub(super) resolution: &'a symtn_tensors::StorageContractResolution<C>,
    pub(super) destination: &'a Arc<BlockStructure>,
    pub(super) destination_len: usize,
    pub(super) lhs: &'a Arc<BlockStructure>,
    pub(super) rhs: &'a Arc<BlockStructure>,
}

impl<'a, D, C> ContractReplay<'a, D, C>
where
    D: symtn_tensors::DenseRecouplingScalar
        + RecouplingCoefficientAction<C>
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    /// The replay of `resolution` on lane `L` into `destination`
    /// (`destination_len` elements) from the stored `operands`.
    pub(super) fn new<L: HostTreeLane<D, C>>(
        runtime: &'a Runtime,
        lease: Option<crate::runtime::ContextLease<'a>>,
        resolution: &'a symtn_tensors::StorageContractResolution<C>,
        (destination, destination_len): (&'a DynamicFusionMapSpace, usize),
        (lhs, rhs): (
            symtn_tensors::FusionOperand<'a>,
            symtn_tensors::FusionOperand<'a>,
        ),
    ) -> Self {
        Self {
            runtime,
            lease,
            execute: host_route_replay::<L, D, C>,
            resolution,
            destination: destination.structure(),
            destination_len,
            lhs: lhs.storage_space().structure(),
            rhs: rhs.storage_space().structure(),
        }
    }
}

/// The storage kernels of the one contraction and composition body:
/// admission, staging, planning, the destination checks and the commit are
/// shared; a kernel only replays the planned route on its storage.
#[doc(hidden)]
pub trait ContractExec<R, D>: TypedStorage<R, D> {
    /// The planner capabilities of this storage's route executor: Host's
    /// eager executor packs an irregular core; the device runs only
    /// fully-direct cores.
    type Caps: symtn_tensors::ExecCaps;

    /// `contract(lhs, rhs)` into a fresh zeroed payload of `replay`'s
    /// destination, reading each operand's stored payload (a lazy adjoint's
    /// parent; Host densifies a compact operand).
    fn contract_owned<C: symtn_tensors::DenseBlockScalar>(
        _: ExecToken,
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D, Self>,
        rhs: &TensorMap<R, D, Self>,
    ) -> Result<Self, TransformExecError>
    where
        D: RecouplingCoefficientAction<C>;

    /// `destination = alpha · contract(lhs, rhs) + beta · destination` into a
    /// destination that passed every admission check.
    fn contract_into<C: symtn_tensors::DenseBlockScalar>(
        _: ExecToken,
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D, Self>,
        rhs: &TensorMap<R, D, Self>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>
    where
        D: RecouplingCoefficientAction<C>;
}

/// The storage kernels of the prepared batches ([`ContractPlan`],
/// [`ComposePlan`]): one plan body, with each storage supplying its lease,
/// output payload, member replay and retained-bytes accounting. Kept apart
/// from [`ContractExec`] so the eager execution traits carry no
/// prepared-batch lifecycle.
#[doc(hidden)]
pub trait PreparedExec<R, D>: TypedStorage<R, D> {
    /// What a prepared stack replay holds from its output allocation through
    /// its replay: the device lease on CUDA; nothing on Host, whose replay
    /// takes its context lease itself.
    type MembersLease<'r>;

    /// Reserves this storage's resources for a new [`ContractWorkspace`].
    fn reserve_members(
        _: ExecToken,
        plan: &ContractPlan<R, D, Self>,
        workspace: &mut ContractWorkspace<R, D, Self>,
    ) -> Result<(), Error>;

    /// Takes the lease of one prepared replay, after every no-write check.
    fn lease_members(_: ExecToken, runtime: &Runtime) -> Result<Self::MembersLease<'_>, Error>;

    /// A zeroed payload of `members` members of `member_len` elements each
    /// (`total` in all).
    fn zeroed_members(
        _: ExecToken,
        lease: &Self::MembersLease<'_>,
        total: usize,
        member_len: usize,
        members: usize,
    ) -> Result<Self, Error>;

    /// Replays `plan` on every member of the checked operands (each its
    /// stored block structure and payload) into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn contract_members(
        _: ExecToken,
        plan: &ContractPlan<R, D, Self>,
        workspace: &mut ContractWorkspace<R, D, Self>,
        lease: &mut Self::MembersLease<'_>,
        dst: &mut Self,
        lhs: (&Arc<BlockStructure>, &Self),
        rhs: (&Arc<BlockStructure>, &Self),
        members: usize,
        init: symtn_tensors::ContractDestinationInit<D>,
    ) -> Result<(), Error>;

    /// The workspace's retained device scratch (none on Host).
    fn retained_scratch_bytes(_: ExecToken, workspace: &ContractWorkspace<R, D, Self>) -> usize;

    /// The retained bytes of one workspace-owned output payload of `len`
    /// elements.
    fn retained_payload_bytes(_: ExecToken, payload: &Self, len: usize) -> usize;
}

impl<R, D: TensorScalar> PreparedExec<R, D> for Vec<D> {
    type MembersLease<'r> = ();

    fn reserve_members(
        _: ExecToken,
        _plan: &ContractPlan<R, D>,
        _workspace: &mut ContractWorkspace<R, D>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn lease_members(_: ExecToken, _runtime: &Runtime) -> Result<(), Error> {
        Ok(())
    }

    fn zeroed_members(
        _: ExecToken,
        _lease: &(),
        total: usize,
        _member_len: usize,
        _members: usize,
    ) -> Result<Self, Error> {
        Ok(symtn_tensors::zeroed_payload(total))
    }

    fn contract_members(
        _: ExecToken,
        plan: &ContractPlan<R, D>,
        workspace: &mut ContractWorkspace<R, D>,
        _lease: &mut (),
        dst: &mut Self,
        lhs: (&Arc<BlockStructure>, &Self),
        rhs: (&Arc<BlockStructure>, &Self),
        members: usize,
        init: symtn_tensors::ContractDestinationInit<D>,
    ) -> Result<(), Error> {
        let mut lease = plan.runtime.lease_context()?;
        lease
            .context()
            .multiplicity_free_lane::<D>()?
            .execute_storage_contract_members_host(
                &plan.resolution,
                (plan.space.space().structure(), dst),
                (lhs.0, lhs.1),
                (rhs.0, rhs.1),
                &mut workspace.members,
                members,
                init,
            )?;
        Ok(())
    }

    fn retained_scratch_bytes(_: ExecToken, _workspace: &ContractWorkspace<R, D>) -> usize {
        0
    }

    fn retained_payload_bytes(_: ExecToken, payload: &Self, _len: usize) -> usize {
        payload.capacity().saturating_mul(std::mem::size_of::<D>())
    }
}

impl<R, D: TensorScalar> ContractExec<R, D> for Vec<D> {
    type Caps = symtn_tensors::HostEagerExecutor;

    fn contract_owned<C: symtn_tensors::DenseBlockScalar>(
        _: ExecToken,
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
    ) -> Result<Self, TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let mut data = symtn_tensors::zeroed_payload(replay.destination_len);
        host_contract(
            replay,
            lhs,
            rhs,
            &mut data,
            D::from_real(1.0),
            symtn_tensors::ContractDestinationInit::Zeroed,
        )?;
        Ok(data)
    }

    fn contract_into<C: symtn_tensors::DenseBlockScalar>(
        _: ExecToken,
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        host_contract(
            replay,
            lhs,
            rhs,
            destination,
            alpha,
            symtn_tensors::ContractDestinationInit::Axpby(beta),
        )
    }
}

fn host_contract<R, D: TensorScalar, C>(
    replay: ContractReplay<'_, D, C>,
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    destination: &mut [D],
    alpha: D,
    init: symtn_tensors::ContractDestinationInit<D>,
) -> Result<(), TransformExecError> {
    let ContractReplay {
        runtime,
        lease,
        execute,
        resolution,
        destination: destination_structure,
        lhs: lhs_structure,
        rhs: rhs_structure,
        ..
    } = replay;
    let mut lease = match lease {
        Some(lease) => lease,
        None => runtime.lease_context()?,
    };
    let (_, lhs_data) = lhs.fusion_operand_and_data();
    let (_, rhs_data) = rhs.fusion_operand_and_data();
    execute(
        lease.context(),
        resolution,
        (destination_structure, destination),
        (lhs_structure, &lhs_data),
        (rhs_structure, &rhs_data),
        alpha,
        init,
    )
}
