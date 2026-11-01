use super::*;

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTwistDispatch<R, D>,
    D: TensorScalar,
    S: TwistExec<R, D>,
{
    /// TensorKit `twist(t, inds; inv)` (and its in-place `twist!`): multiplies
    /// each fusion-tree block by the product over `legs` (flat leg indices,
    /// codomain first) of that leg's ribbon-twist eigenvalue, or by its
    /// inverse for [`Direction::Inverse`]. Repeated legs contribute one twist
    /// factor per occurrence, matching TensorKit’s product over `inds`.
    ///
    /// A bosonic provider, a NoBraiding provider restricted to unit legs, or
    /// selected sectors whose staged twists are all one returns a body-sharing
    /// clone, matching TensorKit's `copy = false` behavior. Otherwise the
    /// operation publishes one fresh scaled payload on the exact admitted
    /// space and provider allocation. A lazy adjoint redirects through its
    /// parent with the inverse operation. A compact diagonal stays compact in
    /// both modes (TensorKit `DiagonalTensorMap`): its spectrum is scaled by
    /// `θ(c)^|legs|` per sector without materializing the dense blocks.
    ///
    /// Reference: TensorKit `twist!`
    /// (`src/tensors/indexmanipulations.jl:62-77` @`cfaa073`), which tests
    /// `has_shared_twist` and then `scale!(t[f₁,f₂], θ)` per fusion-tree
    /// block. QSpace has no ribbon-twist path at all — a concrete absence, not
    /// an unexamined one — so the block-scaling decomposition follows
    /// TensorKit and only the storage kernel is Rust/device specific.
    ///
    /// # CUDA storage
    ///
    /// θ is built on the Host from the block structure alone, before the
    /// device lease, and reaches the kernel as the contraction descriptor's
    /// own scale: for every provider the device admits (`Scalar = f64`) a
    /// ribbon twist is a real sign, never zero, so the descriptor cannot let
    /// CUDA skip a source read: NaN and real infinities propagate as on Host.
    /// An infinite *complex* entry instead becomes NaN in both components,
    /// because the device always multiplies where Host copies a factor-1 block
    /// (the #1301 deviation, disclosed by `cuda_region_axpby`); `f64` and every
    /// finite payload are exact. No θ table is uploaded. An eager call keeps
    /// no device state between calls, so every call transfers the #740 output
    /// initialisation (one H2D of the output bytes) and the region's
    /// one-element `1` template (one H2D of one element): two H2D, two device
    /// allocations, no download, and one strided move per non-empty block. Residual: a whole-buffer bitwise device copy followed
    /// by in-place per-block scaling would submit fewer kernels, but Tenferro
    /// has no in-place strided scale (`cuda_region_axpby` cannot alias its
    /// source and destination), so it is not expressible.
    ///
    /// # Errors
    ///
    /// An out-of-range leg is rejected before the empty-list short circuit.
    /// Non-unit NoBraiding legs are invalid. Checked-Generic pivotal failures
    /// retain their typed provider error, and no result is published until all
    /// selected twist values have been staged successfully.
    pub fn twist(
        &self,
        engine: &mut dyn Engine,
        legs: &[usize],
        direction: Direction,
    ) -> Result<Self, FacadeError<R>> {
        let inverse = direction.is_inverse();
        let rank = self.rank();
        let name = if inverse { "inverse twist" } else { "twist" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            ))
            .into());
        }
        if legs.is_empty() {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        let provider = self.provider();
        let braiding = <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(provider);
        if braiding == symtn_core::BraidingStyleKind::NoBraiding {
            // A unit leg twists by one; any other has no twist to apply.
            let homspace = self.logical_space().space().homspace();
            let vacuum = <R::Mode as TypedSpaceModeDispatch<R>>::vacuum(provider);
            let nout = homspace.codomain().len();
            for &leg in legs {
                let sectors = if leg < nout {
                    homspace.codomain().legs()[leg].sectors()
                } else {
                    homspace.domain().legs()[leg - nout].sectors()
                };
                if sectors.iter().any(|&sector| sector != vacuum) {
                    return Err(Error::InvalidArgument(format!(
                        "{name} leg {leg} carries non-unit sectors but the fusion rule has no braiding"
                    ))
                    .into());
                }
            }
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        if braiding.is_bosonic() {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            // Why not call `adjoint()`: the original view already owns the
            // exact admitted logical space, and re-deriving it would query
            // the provider after all fallible twist values had been staged.
            return Ok(self
                .parent_of(view)
                .twist_owned(engine, &axes, !inverse)?
                .lazy_adjoint_on(&view.logical_space)?);
        }
        self.twist_owned(engine, legs, inverse)
    }

    /// The twist of an owned tensor on its own space: unless every twist value
    /// is one, one scaled copy of the compact spectrum or the dense payload.
    fn twist_owned(
        &self,
        engine: &mut dyn Engine,
        legs: &[usize],
        inverse: bool,
    ) -> Result<Self, FacadeError<R>> {
        let nout = self.codomain_rank();
        let space = self.logical_space();
        let Some(theta) = <R::Mode as TypedTensorTwistDispatch<R, D>>::twist_values(
            space.provider(),
            space.space().structure(),
            nout,
            legs,
        )?
        else {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        };
        if S::as_host(EXEC, self).is_some_and(|t| t.spectrum().is_some()) {
            admit_host_engine(engine)?;
        }
        if let Some(compact) = S::as_host(EXEC, self)
            .and_then(|t| compact_arms::twist_spectrum(t, legs, inverse, &theta))
        {
            return Ok(S::from_host(EXEC, compact)?);
        }
        let name = if inverse { "inverse twist" } else { "twist" };
        Ok(S::scale_blocks(
            engine,
            EXEC,
            self,
            &twist_block_factor(&theta, nout, legs, inverse),
            name,
        )?)
    }
}

/// The one per-block twist coefficient every storage applies: the product of
/// θ over `legs` of the block's uncoupled sectors (conjugated for the
/// inverse), and one for a block without a fusion tree. `theta` is the mode's
/// staged [`TypedTensorTwistDispatch::twist_values`], so evaluating this
/// never fails on the provider.
pub(super) fn twist_block_factor<'a, C: CategoricalScalar + Copy>(
    theta: &'a impl Fn(SectorId) -> C,
    nout: usize,
    legs: &'a [usize],
    inverse: bool,
) -> impl Fn(&BlockKey) -> C + 'a {
    move |key| match key {
        BlockKey::FusionTree(key) => twist_factor_with_inverse(
            legs.iter()
                .map(|&leg| theta(uncoupled_sector_of_leg(key, nout, leg)))
                .fold(C::one(), |product, factor| product * factor),
            inverse,
        ),
        _ => C::one(),
    }
}

/// The twist kernel for one storage (#1756): [`TensorMap::twist`] owns
/// admission, the representation arms and the θ authority, and calls this
/// once, last, with the per-block factor.
#[doc(hidden)]
pub trait TwistExec<R, D>: TypedStorage<R, D> {
    /// A fresh owned dense payload on owned `t`'s space with every block
    /// scaled by `factor_of(block key)`; `operation` names the call in a
    /// capability error.
    fn scale_blocks<C: TwistCoefficient<D>>(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
        factor_of: &dyn Fn(&BlockKey) -> C,
        operation: &'static str,
    ) -> Result<TensorMap<R, D, Self>, Error>;
}

impl<R, D: TensorScalar> TwistExec<R, D> for Vec<D> {
    fn scale_blocks<C: TwistCoefficient<D>>(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D>,
        factor_of: &dyn Fn(&BlockKey) -> C,
        _: &'static str,
    ) -> Result<TensorMap<R, D>, Error> {
        admit_host_engine(engine)?;
        let space = t.logical_space();
        let mut data = t
            .owned_body()
            .ok_or_else(|| internal_layout_error("twist input is owned"))?
            .materialized_dense_data()
            .as_ref()
            .to_vec();
        scale_blocks_impl(space.space(), &mut data, factor_of)?;
        Ok(t.published(space.clone(), data))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorFlipDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `flip(t, I; inv)`:
    /// return a tensor isomorphic to `self` where the duality flag of each
    /// leg in `legs` (distinct flat indices, codomain first) is toggled,
    /// `space(t', i) = flip(space(t, i))`. The stored sectors and the block
    /// layout are unchanged; each fusion-tree block picks up the
    /// Z-isomorphism phase of TensorKit's fusion-tree `flip`
    /// per flipped leg with uncoupled sector `a` and pre-flip duality `d`
    /// (χ = Frobenius–Schur phase, θ = ribbon twist, both in the provider's
    /// scalar, complex for an anyonic table): codomain leg → `d ? χ·θ : 1`;
    /// domain leg → `d ? χ̄ : θ`.
    ///
    /// Like TensorKit's, this `flip` is *not* an involution: flipping the
    /// same leg in two calls returns to the original spaces but scales each block
    /// by `χ_a·θ_a` (codomain) or `θ_a·χ̄_a` (domain), e.g. by θ = −1 on
    /// fermionic legs; so `flip⁴ = id` only when every `θ_a² = 1`, and for
    /// an anyon it is `θ_a²`. [`Direction::Inverse`] applies the inverse
    /// isomorphism, and the lazy-adjoint redirect relies on it being the
    /// conjugate one (a unitary category's χ and θ are phases).
    ///
    /// One scaled copy of the dense payload into a fresh body, O(len); a
    /// compact spectrum factor materializes first (the flipped space is no
    /// longer a bond space, so the result cannot stay compact). The same
    /// facade narrowings as [`Self::twist`] apply: a lazy dense adjoint
    /// redirects through the parent with the inverse categorical map without
    /// materializing; there is no device arm, and checked Generic follows its
    /// provider-mode dispatch.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a leg is out of range or repeated.
    /// The full out-of-range check precedes the duplicate check; both precede
    /// provider queries and the empty-list short-circuit. Empty `legs` returns
    /// an identical clone.
    /// Otherwise [`Error::Operation`] /
    /// [`Error::Core`] from the layout derivation of the toggled hom space.
    /// Checked-Generic admission and pivotal failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn flip(
        &self,
        engine: &mut dyn Engine,
        legs: &[usize],
        direction: Direction,
    ) -> Result<Self, FacadeError<R>> {
        admit_host_engine(engine)?;
        let inverse = direction.is_inverse();
        let rank = self.rank();
        let name = if inverse { "inverse flip" } else { "flip" };
        if let Some(&leg) = legs.iter().find(|&&leg| leg >= rank) {
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} out of range for rank {rank}"
            ))
            .into());
        }
        if let Some((_, &leg)) = legs
            .iter()
            .enumerate()
            .find(|&(index, leg)| legs[..index].contains(leg))
        {
            return Err(
                Error::InvalidArgument(format!("{name} leg {leg} listed more than once")).into(),
            );
        }
        if legs.is_empty() {
            return Ok(self.clone());
        }
        if <R::Mode as TypedTensorModeDispatch<R>>::braiding_style(self.provider())
            == symtn_core::BraidingStyleKind::NoBraiding
        {
            let leg = legs[0];
            return Err(Error::InvalidArgument(format!(
                "{name} leg {leg} needs the twist and Frobenius-Schur coefficients but the fusion rule has no braiding"
            ))
            .into());
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let logical_space = self.flip_destination(legs)?.0;
            let parent = Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            };
            let axes = logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            );
            let flipped_parent = parent.flip_owned(&axes, !inverse)?;
            let TypedTensorRepr::Owned(parent) = &flipped_parent.repr else {
                return Err(internal_layout_error("a parent flip stays owned").into());
            };
            return Ok(Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                    Arc::clone(parent),
                    logical_space,
                ))),
            });
        }
        self.flip_owned(legs, inverse)
    }

    /// The flip of an owned tensor: one scaled copy of its dense payload on
    /// the toggled space.
    fn flip_owned(&self, legs: &[usize], inverse: bool) -> Result<Self, FacadeError<R>> {
        let nout = self.codomain_rank();
        let (space, occurrences) = self.flip_destination(legs)?;
        let data = {
            let values = <R::Mode as TypedTensorFlipDispatch<R, D>>::pivotal_values(
                self.provider(),
                space.space().structure(),
                nout,
                &occurrences,
            )?;
            let mut data = self
                .owned_body()
                .ok_or_else(|| internal_layout_error("flip input is owned"))?
                .materialized_dense_data()
                .as_ref()
                .to_vec();
            scale_blocks_impl(space.space(), &mut data, &|key| match key {
                BlockKey::FusionTree(key) => {
                    flip_block_factor(&values, key, nout, &occurrences, inverse)
                }
                _ => {
                    <<R::Mode as TypedTensorFlipDispatch<R, D>>::Coeff as CategoricalScalar>::one()
                }
            })?;
            data
        };
        Ok(self.published(space, data))
    }

    /// The admitted duality-toggled space of a flip of `legs`, with each
    /// distinct leg's stored duality. The stored layout must not change.
    #[expect(
        clippy::type_complexity,
        reason = "the flip destination carries its occurrence metadata"
    )]
    fn flip_destination(
        &self,
        legs: &[usize],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<(usize, bool)>), FacadeError<R>> {
        let (homspace, occurrences) =
            flip_toggled_homspace(self.logical_space().space().homspace(), legs);
        let space =
            <R::Mode as TypedTensorFlipDispatch<R, D>>::root(self.logical_space(), homspace)?;
        check_flip_layout_identity(
            self.logical_space().space().structure(),
            space.space().structure(),
        )?;
        Ok((space, occurrences))
    }
}
