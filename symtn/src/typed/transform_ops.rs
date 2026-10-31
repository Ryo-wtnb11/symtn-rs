use super::*;

mod multiplicity_free;
mod replay;
mod unit_legs;
pub(super) use self::multiplicity_free::*;
pub use self::replay::*;
pub(super) use self::unit_legs::*;

/// The exact-layout probe of a repartition: the planar axes of
/// `with_planar_axes`'s `Repartition` arm, laid out on the stack so a warm
/// typed `repartition_into` allocates nothing. Ranks above the inline bound
/// take the full path (no probe).
pub(super) fn repartition_probe(
    source_codomain_rank: usize,
    source_rank: usize,
    num_codomain: usize,
    probe: &mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>),
) {
    const INLINE_RANK: usize = 64;
    if source_rank > INLINE_RANK || num_codomain > source_rank {
        return;
    }
    let mut axes = [0usize; INLINE_RANK];
    for (slot, axis) in axes
        .iter_mut()
        .zip((0..source_codomain_rank).chain((source_codomain_rank..source_rank).rev()))
    {
        *slot = axis;
    }
    axes[num_codomain..source_rank].reverse();
    probe(tree_operation_view(
        TreeTransformOperationKind::Transpose,
        &axes[..num_codomain],
        &axes[num_codomain..source_rank],
    ));
}

/// The borrowed exact-layout probe of a permutation request without levels.
pub(super) fn tree_operation_view<'a>(
    kind: TreeTransformOperationKind,
    codomain_axes: &'a [usize],
    domain_axes: &'a [usize],
) -> symtn_tensors::TreeTransformOperationView<'a> {
    symtn_tensors::TreeTransformOperationView::new(kind, codomain_axes, domain_axes, &[], &[])
}

/// The braid operation of `braid`/`braid_into`, with the facade's one own
/// check: one level per source axis.
pub(super) fn braid_operation(
    rank: usize,
    codomain_rank: usize,
    codomain_axes: &[usize],
    domain_axes: &[usize],
    levels: &[usize],
) -> Result<TreeTransformOperation, Error> {
    if levels.len() != rank {
        return Err(Error::InvalidArgument(format!(
            "braid levels must list one level per source axis \
             (expected {rank}, got {})",
            levels.len()
        )));
    }
    Ok(TreeTransformOperation::braid(
        codomain_axes.iter().copied(),
        domain_axes.iter().copied(),
        levels[..codomain_rank].iter().copied(),
        levels[codomain_rank..].iter().copied(),
    ))
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D>,
    D: TensorScalar,
    S: TransformExec<R, D>,
{
    /// `destination = alpha * self.permute(codomain_axes, domain_axes) +
    /// beta * destination`, TensorKit `permute!(tdst, tsrc, p, α, β)`, without
    /// replacing the destination's provider, space, body or host allocation.
    ///
    /// `beta` rides the add that writes each destination block, so it is
    /// applied once per element and never as a separate pass. `beta == 0`
    /// (IEEE, `-0.0` too) never reads `destination`: a NaN there does not
    /// survive, and the blocks the transform does not reach become `+0`.
    /// `beta == 1` leaves those blocks untouched. `alpha == 0` does not read
    /// the source (VectorInterface's `scale(x, 0) = zero(x)`).
    ///
    /// A lazy-adjoint source is read through its parent's storage, and a
    /// compact diagonal source writes its transformed diagonal whenever the
    /// transform maps each bond block to one block of the same shape with one
    /// coefficient (otherwise it is densified into an operation-local buffer
    /// first); neither is materialized as a result.
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`], [`Error::RuleMismatch`], then
    /// [`Error::InvalidArgument`] for a destination that is not owned dense
    /// host storage, or one that aliases the source (a lazy adjoint's
    /// parent included); then the operation's own errors, as for [`Self::permute`];
    /// `SpaceMismatch` for a destination whose space or block layout does not
    /// match the result, [`Error::InvalidArgument`] for one whose length does
    /// not, and [`Error::DestinationShared`] when `destination` shares its
    /// storage with a clone. Checked Generic reports these in
    /// [`GenericTensorError::Facade`] and staging failures in
    /// [`GenericTensorError::Plan`], as for [`Self::permute`]. On CUDA the
    /// storage noun names the placement, and a payload of another device or
    /// engine is [`Error::PlacementMismatch`], checked first under the device
    /// lease.
    ///
    /// # CUDA storage
    ///
    /// The admission, staging and exact-layout memo are the ones above; the
    /// staging's context lease is released before the device lease, and
    /// nothing under the device lease leases again.
    ///
    /// `beta` rides the `dot_general` epilogue of the move or scatter that
    /// writes each destination layout, once per element; the layouts no move
    /// writes become `beta * destination` through a zero-source move (zeros
    /// for `beta = 0`, which never reads the destination; nothing for
    /// `beta = 1`).
    ///
    /// No payload crosses the host boundary and nothing is downloaded. Unlike
    /// the returning [`Self::permute`] it pays no output initialisation. The
    /// device state of the replay — the structure's coefficient payload, the
    /// pack/scatter workspace, the one-element `1` and the zero template — is
    /// created by each call on the caller's engine, as for every eager
    /// operation: there is no workspace to keep it in. `alpha == 0` writes
    /// `beta * destination` over every written layout through that zero
    /// template, sized to the largest written layout.
    ///
    /// Numerics are [`Self::permute`]'s, with the caller scale applied as the
    /// executor's rule: a Single block rounds as `alpha * (c * x)` where the
    /// Host folds the scales and rounds as `(alpha * c) * x`; Multi blocks
    /// agree in order with the Host (`alpha * (U x)`). `alpha == 0` — `-0.0`
    /// included, by IEEE comparison — does not read the source, as on Host;
    /// only the sign of an exact zero may differ.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn permute_into(
        &self,
        engine: &mut dyn Engine,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        self.tree_transform_into(
            engine,
            destination,
            alpha,
            beta,
            || {
                Ok(TreeTransformOperation::permute(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
            |probe| {
                probe(tree_operation_view(
                    TreeTransformOperationKind::Permute,
                    codomain_axes,
                    domain_axes,
                ))
            },
        )
    }

    /// `destination = alpha * self.braid(codomain_axes, domain_axes, levels) +
    /// beta * destination`, TensorKit `braid!(tdst, tsrc, p, levels, α, β)`.
    /// `levels` must list one level per source axis, as for
    /// [`Self::braid`]; otherwise destination rules and errors are
    /// [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    #[allow(clippy::too_many_arguments)]
    pub fn braid_into(
        &self,
        engine: &mut dyn Engine,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        let operation = braid_operation(
            self.rank(),
            self.codomain_rank(),
            codomain_axes,
            domain_axes,
            levels,
        )?;
        let probe_operation = operation.clone();
        self.tree_transform_into(
            engine,
            destination,
            alpha,
            beta,
            || Ok(operation),
            |probe| {
                probe(symtn_tensors::TreeTransformOperationView::of(
                    &probe_operation,
                ))
            },
        )
    }

    /// `destination = alpha * self.transpose(codomain_axes, domain_axes) +
    /// beta * destination`, TensorKit `transpose!(tdst, tsrc, p, α, β)`.
    /// Destination rules and errors are [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn transpose_into(
        &self,
        engine: &mut dyn Engine,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        self.tree_transform_into(
            engine,
            destination,
            alpha,
            beta,
            || {
                with_planar_axes(
                    self.codomain_rank(),
                    self.rank(),
                    PlanarRequestKind::Explicit {
                        codomain_axes,
                        domain_axes,
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |probe| {
                probe(tree_operation_view(
                    TreeTransformOperationKind::Transpose,
                    codomain_axes,
                    domain_axes,
                ))
            },
        )
    }

    /// `destination = alpha * self.repartition(destination.codomain_rank()) +
    /// beta * destination`, TensorKit `repartition!(tdst, tsrc, α, β)`: the
    /// target split is the destination's own. Destination rules and errors
    /// are [`Self::permute_into`]'s.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    pub fn repartition_into(
        &self,
        engine: &mut dyn Engine,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        let source_codomain_rank = self.codomain_rank();
        let source_rank = self.rank();
        let destination_codomain_rank = destination.codomain_rank();
        let destination_rank = destination.rank();
        self.tree_transform_into(engine,
            destination,
            alpha,
            beta,
            || {
                if destination_rank != source_rank {
                    return Err(Error::InvalidArgument(format!(
                        "repartition destination rank {destination_rank} does not match source rank {source_rank}"
                    )));
                }
                with_planar_axes(
                    source_codomain_rank,
                    source_rank,
                    PlanarRequestKind::Repartition {
                        num_codomain: destination_codomain_rank,
                    },
                    |codomain_axes, domain_axes| {
                        Ok(TreeTransformOperation::transpose(
                            codomain_axes.iter().copied(),
                            domain_axes.iter().copied(),
                        ))
                    },
                )
            },
            |probe| {
                if destination_rank == source_rank {
                    repartition_probe(
                        source_codomain_rank,
                        source_rank,
                        destination_codomain_rank,
                        probe,
                    );
                }
            },
        )
    }

    /// The one admission boundary of every typed beta-accumulating tree
    /// transform; the mode's [`TypedTensorTransformDispatch::transform_into`]
    /// stages, checks the destination's space and has the storage write.
    pub(super) fn tree_transform_into(
        &self,
        engine: &mut dyn Engine,
        destination: &mut Self,
        alpha: D,
        beta: D,
        operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
        exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
    ) -> Result<(), FacadeError<R>> {
        if !self.runtime.same_runtime(&destination.runtime) {
            return Err(Error::RuntimeMismatch.into());
        }
        // The held admitted identities: provider-free, and the provider's
        // own by #2046.
        if self.logical_space().space().admission().rule_identity()
            != destination
                .logical_space()
                .space()
                .admission()
                .rule_identity()
        {
            return Err(Error::RuleMismatch.into());
        }
        // Why the storage body: a lazy-adjoint source aliases its parent.
        destination_precheck(destination, &self.storage_body().data, S::NOUN)?;
        <R::Mode as TypedTensorTransformDispatch<R, D>>::transform_into(
            engine,
            self,
            destination,
            alpha,
            beta,
            operation,
            exact_layout_probe,
        )
    }

    /// TensorKit `permute`: re-arranges legs with symmetric braiding.
    ///
    /// `codomain_axes` and `domain_axes` list source axis numbers (`0..rank`,
    /// codomain axes first) for the new codomain and domain.
    ///
    /// # Compact storage
    ///
    /// An exact identity returns a compact diagonal factor unchanged. A
    /// rank-(1,1) leg swap through [`Self::permute`] or [`Self::transpose`]
    /// keeps it compact. An admitted rank-(1,1) [`Self::braid`] reads its
    /// spectrum directly and publishes a dense result. Other non-identity
    /// `permute`/`transpose`/`repartition` cases and unadmitted braids
    /// materialize the compact source into an operation-local buffer and
    /// publish a dense `Σ_c k_c²` result.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`] when the axis lists
    /// are malformed (out of range, repeated, or not a partition of `0..rank`),
    /// as at every axis-taking entry point; [`Error::Operation`] /
    /// [`Error::Core`] / [`Error::FusionAlgebra`] when the provider cannot
    /// support the braiding the requested motion needs. The expert layer's own typed errors are the
    /// contract here: re-validating the axes at this layer would be a second
    /// copy of a rule that already exists one call down, free to drift.
    /// Checked Generic providers report an axis misuse in
    /// [`GenericTensorError::Facade`] with the same [`Error`], and other
    /// failures in [`GenericTensorError::Plan`] with the concrete provider error
    /// preserved as its source. On CUDA a payload of another device or engine
    /// is [`Error::PlacementMismatch`], before any device write.
    ///
    /// # CUDA storage
    ///
    /// Everything categorical is the Host's, unchanged: the same operation,
    /// rule validation, staged destination, `RuleIdentity`-keyed transform
    /// cache and completed transformer. Only the replay differs, so a device
    /// result is the Host result up to dtype tolerance. The staging's context
    /// lease is released before the device lease is taken, and nothing under
    /// the device lease leases again: one device lease per replay.
    ///
    /// A warm call uploads no structure data and downloads nothing. Its only
    /// transfer is the output initialisation (#740): exactly one H2D of
    /// `required_len * size_of::<D>()` bytes, from one host `Vec` of that
    /// size, and its only device allocation is the returned output. The first
    /// call for a given structure additionally uploads that structure's
    /// coefficient payload once and may grow the pack/scatter workspace once.
    ///
    /// On device the replay runs in overwrite mode, so it also zero-fills
    /// every *inactive* destination layout on the device — redundant work over
    /// an output that was just uploaded as zeros, costing one submission and
    /// `Σ(inactive layout elements)` device writes per call. It is kept
    /// because it is what makes the executor's destination mode independent of
    /// what the destination held, which `*_into` with `beta = 0` relies on;
    /// removing it here would need a second replay mode for no transfer saving.
    ///
    /// That warm contract holds only while this Runtime's Host transform store
    /// admits the structure: with a tree-transform cache byte budget of zero,
    /// or for a structure whose entry exceeds that budget, the store hands
    /// back a fresh allocation per call and the device therefore re-uploads
    /// its coefficients per call.
    ///
    /// Numerics are the device executor's: the block set written is the
    /// Host's exactly; coefficient-`1` `f64` moves are bitwise, and everything
    /// else agrees to dtype tolerance, with one rounding position moved for
    /// Single blocks (`alpha * (c * x)` where the Host folds the scales and
    /// rounds as `(alpha * c) * x`) and with the recoupling GEMM's summation
    /// order that of cuTENSOR rather than the Host kernel.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    /// )?;
    /// let w = GradedSpace::try_new(
    ///     Arc::new(U1FusionRule),
    ///     [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    /// )?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&w], 7)?;
    /// assert_eq!(t.leg_dims()?, [3, 2]);
    ///
    /// let swapped = t.permute(engine, &[1], &[0])?;
    /// assert_eq!(swapped.leg_dims()?, [2, 3]);
    /// // A bosonic two-leg swap is an involution: swapping back restores the
    /// // payload exactly.
    /// assert_eq!(swapped.permute(engine, &[1], &[0])?.dense_data()?, t.dense_data()?);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn permute(
        &self,
        engine: &mut dyn Engine,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        self.tree_transform(
            engine,
            TreeTransformOperation::permute(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
            ),
        )
    }

    /// Runs `op` on the matrix view `permute(self, rows, cols)`: the leg roles
    /// of every factorization and matrix function.
    ///
    /// The current split borrows `self`, so it costs no transform and no
    /// clone. Any other split is one [`Self::permute`], whose typed errors
    /// (malformed axes, non-symmetric braiding) are returned before `op` runs.
    pub(super) fn with_leg_roles<T>(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
        op: impl FnOnce(&Self, &mut dyn Engine) -> Result<T, FacadeError<R>>,
    ) -> Result<T, FacadeError<R>> {
        if self.axes_are_identity(rows, cols) {
            return op(self, engine);
        }
        let permuted = self.permute(engine, rows, cols)?;
        op(&permuted, engine)
    }

    /// TensorKit `braid`: re-arranges legs with an explicit braid, one level
    /// per source axis.
    ///
    /// `codomain_axes` and `domain_axes` name source axes exactly as for
    /// [`Self::permute`]. `levels` is per source *strand*, one entry for every
    /// axis in `0..rank` — codomain axes first — and it is split by the
    /// **source** codomain rank, so entry `i` always describes source axis `i`
    /// regardless of where that axis ends up. The levels decide which strand
    /// crosses above at each transposition; for a symmetric (bosonic) braiding
    /// they cannot change the result, and this is then [`Self::permute`].
    ///
    /// A compact rank-(1,1) diagonal is read directly when the compiled braid
    /// has one term per source block, its coefficient remains finite and
    /// nonzero in the payload dtype, and destination coverage is complete;
    /// otherwise it uses the ordinary dense replay.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `levels` does not have one entry per
    /// source axis — the one check this facade makes itself, because the axis
    /// lists and the levels are validated by different layers and a
    /// mis-lengthed `levels` would otherwise be split silently.
    ///
    /// Otherwise, as for [`Self::permute`]: [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`] for malformed axis
    /// lists, and [`Error::Operation`] / [`Error::Core`] /
    /// [`Error::FusionAlgebra`] from the expert layer for a provider that
    /// cannot support the requested braiding; this layer does not re-validate
    /// axes. Checked Generic reports an axis misuse in
    /// [`GenericTensorError::Facade`] and other failures in
    /// [`GenericTensorError::Plan`].
    pub fn braid(
        &self,
        engine: &mut dyn Engine,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        levels: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        // Keep the established axis/level pre-check and diagnostic.
        let rank = self.rank();
        if levels.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "braid levels must list one level per source axis \
                 (expected {rank}, got {})",
                levels.len()
            ))
            .into());
        }
        if self.axes_are_identity(codomain_axes, domain_axes) {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        let nout = self.codomain_rank();
        self.tree_transform(
            engine,
            TreeTransformOperation::braid(
                codomain_axes.iter().copied(),
                domain_axes.iter().copied(),
                levels[..nout].iter().copied(),
                levels[nout..].iter().copied(),
            ),
        )
    }

    /// TensorKit `repartition(t, N₁, N₂)`: moves the planar boundary so the
    /// codomain holds `num_codomain` legs and the domain holds the rest.
    ///
    /// The planar order — codomain followed by reversed domain — is preserved;
    /// legs that cross the boundary are bent, and so arrive with their dual
    /// flag flipped and their sectors dualized, without any braid being
    /// introduced. The identity repartition returns its input unchanged; a
    /// compact diagonal whose boundary moves is densified into an
    /// operation-local buffer first.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `num_codomain` exceeds the rank, and
    /// otherwise, as for [`Self::permute`], [`Error::Operation`] /
    /// [`Error::Core`] / [`Error::FusionAlgebra`] from the expert layer, which
    /// owns the validation this facade passes through; an axis misuse is
    /// [`Error::Operation`] with
    /// [`crate::typed::OperationError::InvalidPermutation`]. Checked Generic
    /// reports an axis misuse in [`GenericTensorError::Facade`] and other
    /// failures in [`GenericTensorError::Plan`].
    pub fn repartition(
        &self,
        engine: &mut dyn Engine,
        num_codomain: usize,
    ) -> Result<Self, FacadeError<R>> {
        if num_codomain == self.codomain_rank() {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        self.planar(engine, PlanarRequestKind::Repartition { num_codomain })
    }

    /// TensorKit `transpose(t, (p₁, p₂))`: the planar transpose with an
    /// explicit cyclic axis map.
    ///
    /// `codomain_axes` and `domain_axes` are flat source axis numbers
    /// (`0..rank`, codomain axes first), exactly as for [`Self::permute`], but
    /// together they must describe one **cyclic rotation** of the planar source
    /// order (codomain axes followed by the domain axes reversed).
    ///
    /// Planar means it **never braids**: legs are bent across the boundary, and
    /// bending conjugates them, so legs that cross it carry flipped dual flags.
    /// Spelling this as a [`Self::permute`] of the same axis order would be
    /// wrong for any provider whose braiding is not symmetric — the two agree
    /// only up to the R-symbols a permute inserts and this does not.
    ///
    /// TensorKit's argument-free `transpose(t)` is the full rotation
    /// `codomain_axes = (nout..rank).rev()`, `domain_axes = (0..nout).rev()`,
    /// which carries every codomain leg across the boundary and every domain
    /// leg back; it is its own inverse. There is no argument-free overload:
    /// the axes are the operation's leg roles and are always stated.
    /// [`Self::repartition`] is the other planar parameterization, by the
    /// target codomain rank.
    ///
    /// # Errors
    ///
    /// [`Error::Operation`] / [`Error::Core`] / [`Error::FusionAlgebra`] when
    /// the axis lists are malformed or are not a cyclic rotation of the planar
    /// order — a re-arrangement that would need a braid is refused rather than
    /// silently braided. As everywhere in this
    /// facade the expert layer owns that validation; it is not repeated here.
    pub fn transpose(
        &self,
        engine: &mut dyn Engine,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        if self.axes_are_identity(codomain_axes, domain_axes) {
            self.admit_shortcut_engine(engine)?;
            return Ok(self.clone());
        }
        self.planar(
            engine,
            PlanarRequestKind::Explicit {
                codomain_axes,
                domain_axes,
            },
        )
    }

    /// Shared body of the three planar operations: derive the planar axis
    /// order, let the expert layer check it, and run it as a transpose.
    ///
    /// The shared axis derivation defines what "planar" means for each request
    /// kind; duplicating it here would allow the definitions to drift.
    fn planar(
        &self,
        engine: &mut dyn Engine,
        kind: PlanarRequestKind<'_>,
    ) -> Result<Self, FacadeError<R>> {
        let operation = with_planar_axes(
            self.codomain_rank(),
            self.rank(),
            kind,
            |codomain_axes, domain_axes| {
                // Why `transpose` and not `permute` even when the axes happen
                // to be a plain permutation: domain trees run opposite to the
                // planar boundary, so flattening them into a permute would
                // braid a different leg across it.
                Ok(TreeTransformOperation::transpose(
                    codomain_axes.iter().copied(),
                    domain_axes.iter().copied(),
                ))
            },
        )
        .map_err(FacadeError::<R>::from)?;
        self.tree_transform(engine, operation)
    }

    /// The one body of every admitted permutation, braid and planar
    /// transpose: the compact rank-(1,1) arm and the mode's lazy-adjoint arm,
    /// else one dense transform published as an owned tensor.
    pub(super) fn tree_transform(
        &self,
        engine: &mut dyn Engine,
        operation: TreeTransformOperation,
    ) -> Result<Self, FacadeError<R>> {
        if let Some(host) = S::as_host(EXEC, self) {
            if let Some(compact) =
                compact_arms::transform_rank_one_diagonal(engine, host, &operation)?
            {
                return Ok(S::from_host(EXEC, compact)?);
            }
        }
        if let TypedTensorRepr::Adjoint(_) = &self.repr {
            if let Some(lazy) =
                <R::Mode as TypedTensorTransformDispatch<R, D>>::try_lazy_adjoint_transform(
                    engine, self, &operation,
                )?
            {
                return Ok(lazy);
            }
        }
        let (space, data) =
            <R::Mode as TypedTensorTransformDispatch<R, D>>::transform(engine, self, operation)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}

/// [`map_spectrum`]'s cross-dtype sibling: the same sector-and-length
/// preserving value map, for the conversions whose output dtype differs from
/// the input's — which is exactly why the two cannot share one signature.
pub(super) fn map_spectrum_dtype<A: Copy, B>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<A>],
    value_of: impl Fn(A) -> B,
) -> Vec<symtn_matrixalgebra::SectorSpectrum<B>> {
    spectrum
        .iter()
        .map(|entry| symtn_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| value_of(value)).collect(),
        })
        .collect()
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedAdjointSpace<R>,
    D: TensorScalar,
    S: TypedStorage<R, D>,
{
    /// Returns the adjoint `self^H`, swapping codomain and domain and
    /// conjugate-transposing every coupled-sector block.
    ///
    /// Dense storage becomes a lazy parent-backed view: its logical space is
    /// available immediately, and only [`Self::materialize`] builds the
    /// whole logical payload. Applying `adjoint` twice
    /// returns the original owned parent. Compact diagonal storage instead
    /// performs an owned `O(sum_c k_c)` conjugation and stays compact, as
    /// TensorKit `adjoint(::DiagonalTensorMap)` does.
    /// Every result keeps the source's exact provider `Arc`, and a lazy view
    /// shares the parent's storage, so on CUDA it reads the same device
    /// allocation without any transfer.
    ///
    /// If layout construction or checked pivotal data fails, that layout or
    /// provider error is returned before a view is created. For real payloads
    /// this is a transpose; complex payloads are conjugated as well.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 2)?;
    /// assert_eq!(t.adjoint(engine)?.adjoint(engine)?.dense_data()?, t.dense_data()?);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn adjoint(&self, engine: &mut dyn Engine) -> Result<Self, FacadeError<R>> {
        // The lazy view moves no payload; the engine is the uniform call
        // shape of a tensor operation.
        let _ = engine;
        self.lazy_adjoint()
    }

    /// [`Self::adjoint`] without an engine: a lazy view or a compact
    /// conjugate, never a payload pass.
    pub(super) fn lazy_adjoint(&self) -> Result<Self, FacadeError<R>> {
        // Why not a lazy view of a compact diagonal: densifying its unstored
        // zeros through conjugation would publish them as `0-0i`.
        if let Some(adjoint) = self.compact_adjoint() {
            return Ok(adjoint);
        }
        Ok(match &self.repr {
            TypedTensorRepr::Owned(parent) => {
                let logical_space =
                    <R::Mode as TypedAdjointSpace<R>>::adjoint_space(&parent.space)?;
                debug_assert!(Arc::ptr_eq(
                    parent.space.provider_arc(),
                    logical_space.provider_arc()
                ));
                Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                        Arc::clone(parent),
                        logical_space,
                    ))),
                }
            }
            TypedTensorRepr::Adjoint(view) => Self {
                runtime: self.runtime.clone(),
                repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
            },
        })
    }

    /// Borrowed adjoint view: the operand [`Self::adjoint`] would give,
    /// passed without building it. See [`TensorRef`].
    pub fn adjoint_view(&self) -> TensorRef<'_, R, D, S> {
        TensorRef {
            base: self,
            adjoint: Some(|tensor| {
                tensor
                    .lazy_adjoint()
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::facade_error_into_error)
            }),
        }
    }
}
