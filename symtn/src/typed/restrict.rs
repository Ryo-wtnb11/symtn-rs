use super::*;

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    /// Checks that this tensor can have leg `axis` exchanged for `expected`.
    fn require_selected_leg(
        &self,
        axis: usize,
        expected: &GradedSpace<R>,
        operation: &str,
    ) -> Result<(), FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorModeDispatch<R>,
    {
        if self.network_has_compact_payload() {
            return Err(Error::InvalidArgument(format!(
                "{operation} requires a dense Host payload"
            ))
            .into());
        }
        require_selected_leg_of(self.logical_space(), axis, expected, operation)
    }

    /// This tensor's hom-space with leg `axis` replaced, built into a root
    /// layout of the same provider.
    fn root_with_replaced_leg(
        &self,
        axis: usize,
        replacement: &SectorLeg,
    ) -> Result<BoundDynamicFusionMapSpace<R>, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        space_with_replaced_legs(self.logical_space(), |candidate| {
            (candidate == axis).then_some(replacement)
        })
    }

    /// Restricts each listed leg to the subspace its selection names.
    ///
    /// `legs` is a set of `(axis, selection)` pairs on distinct axes; a
    /// single-leg restriction is `&[(axis, &selection)]`. Each pair is
    /// composition with the inclusion isometry `ι_σ` of [`LegSelection`] on
    /// that leg — `ι_σ^† ∘ t` for a codomain leg, `t ∘ ι_σ` for a domain leg.
    /// Isometries on distinct legs commute, so the result equals any order of
    /// one-pair restrictions, bit for bit. The tensor stays invariant, every
    /// leg keeps its dual flag, and every surviving fusion tree keeps its
    /// inner lines and vertices: `ι_σ` is the identity on irreps, so no
    /// structural coefficient, braid or fermionic sign enters. Selecting a
    /// single degeneracy index of sector `q` leaves a leg equal to the
    /// one-dimensional space of `q`, so the charge stays explicit on the leg.
    ///
    /// A compact diagonal `bond <- bond` map (the `s` of an SVD, the `d` of
    /// an `eigh`) stays compact when both legs get the same selection,
    /// `&[(0, &sel), (1, &sel)]`: the result is again an endomorphism of the
    /// kept subspace (TensorKit `truncate_diagonal!`). This is the third of
    /// the three calls that turn a compact factorization plus a
    /// [`GradedSpace::find_truncated`] decision into a truncated one.
    ///
    /// [`Self::embed_leg`] is the adjoint: `restrict_leg` after `embed_leg` is
    /// the identity, `embed_leg` after `restrict_leg` is the orthogonal
    /// projector onto the subspace.
    ///
    /// # Cost
    ///
    /// One strided copy per destination block and piece for all `k` legs at
    /// once, where a block's pieces are the product of the kept-position runs
    /// of its sectors (one for a contiguous selection, at most two per leg for
    /// a selection of the largest `|λ|` of an ascending spectrum, at most the
    /// kept count in general). `O(selected payload)` data movement, no matrix
    /// multiplication and no recoupling. One payload allocation plus
    /// `O(rank + blocks + rank · pieces)` structural work, independent of the
    /// degeneracy dimensions. (TensorKit pays a
    /// contraction with an explicit isometry tensor for an arbitrary leg;
    /// symtn addresses the degeneracy axes inside each reduced block
    /// directly.) A compact input copies only the `O(sum_c k'_c)` kept
    /// values; the discarded ones are never touched.
    ///
    /// Defined for Host payloads: device storage has no such method, so an
    /// unsupported placement is a compile-time absence rather than a runtime
    /// error. A lazy adjoint is read in place.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `legs` is empty, an axis is out of
    /// range or appears twice, or an axis is not the leg its selection was
    /// built from; [`Error::RuleMismatch`] when a selection belongs to a
    /// different rule; [`Error::Unsupported`] with
    /// [`Alternative::Materialize`] for a compact diagonal payload with any
    /// set other than both legs and one selection (call
    /// [`Self::materialize`] first for a dense result). Nothing is allocated
    /// before every check has passed.
    pub fn restrict_leg(
        &self,
        engine: &mut dyn Engine,
        legs: &[(usize, &LegSelection<R>)],
    ) -> Result<Self, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        // Host-local work: the engine is the uniform call shape.
        let _ = engine;
        require_restriction_set(self.logical_space(), legs)?;
        let compact = self
            .spectrum()
            .map(|spectrum| match legs {
                [(0, first), (1, second)] | [(1, second), (0, first)]
                    if first.entries == second.entries =>
                {
                    Ok((spectrum, *first))
                }
                // A compact diagonal stays compact only with one selection on
                // both legs; any other restriction needs the dense payload.
                _ => Err(FacadeError::<R>::from(Error::Unsupported {
                    operation: "restrict_leg",
                    alternative: Alternative::Materialize,
                })),
            })
            .transpose()?;
        let destination = restricted_space(self.logical_space(), legs)?;
        if let Some((spectrum, selection)) = compact {
            let mut kept = Vec::with_capacity(selection.entries.len());
            for (sector, runs) in &selection.entries {
                // Both lists are in canonical `SectorId` order, so this is a
                // lookup, not a scan. A violated order can only make the
                // search miss, which is the typed error below — never a match
                // on the wrong sector, because the key is compared.
                let source = spectrum
                    .binary_search_by_key(sector, |entry| entry.sector)
                    .ok()
                    .map(|index| &spectrum[index].values)
                    .filter(|values| runs.last().is_some_and(|run| run.end <= values.len()))
                    .ok_or_else(|| {
                        FacadeError::<R>::from(Error::InvalidArgument(format!(
                            "restrict_leg: the compact payload has no positions {runs:?} for \
                             sector {sector:?}"
                        )))
                    })?;
                kept.push(symtn_matrixalgebra::SectorSpectrum {
                    sector: *sector,
                    values: runs
                        .iter()
                        .flat_map(|run| &source[run.clone()])
                        .copied()
                        .collect(),
                });
            }
            return Ok(self.with_spectrum_on(destination, kept));
        }
        let (source, source_data) = self.fusion_operand_and_data();
        let data = symtn_tensors::oriented_fusion_restrict_owned(
            destination.space().structure(),
            source,
            &source_data,
            &restriction_runs(self.rank(), legs),
        )
        .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(destination, data)),
        })
    }

    /// Embeds leg `axis` back into the parent leg of `selection`, zero-padding
    /// the degeneracy indices the selection does not name.
    ///
    /// This is the adjoint of [`Self::restrict_leg`]: composition with `ι_σ`
    /// on a codomain leg and with `ι_σ^†` on a domain leg. The receiver's leg
    /// `axis` must equal [`LegSelection::subspace`]; the result carries
    /// [`LegSelection::parent`] on that axis. No target space is passed
    /// separately — the selection already owns both legs, so they cannot
    /// disagree.
    ///
    /// # Cost
    ///
    /// `O(source payload)` data movement into an allocator-zeroed output of
    /// `O(destination payload)`, no matrix multiplication and no recoupling:
    /// one strided add per source block and piece of the kept-position runs,
    /// as [`Self::restrict_leg`] reads them.
    ///
    /// Defined for Host payloads only, as [`Self::restrict_leg`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the payload is a compact diagonal one,
    /// `axis` is out of range, or `axis` is not [`LegSelection::subspace`];
    /// [`Error::RuleMismatch`] when the selection belongs to a different rule.
    /// Every block is preflighted before the first destination element is
    /// written.
    pub fn embed_leg(
        &self,
        engine: &mut dyn Engine,
        axis: usize,
        selection: &LegSelection<R>,
    ) -> Result<Self, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        self.require_selected_leg(axis, selection.subspace(), "embed_leg")?;
        let destination = self.root_with_replaced_leg(axis, selection.parent().leg())?;
        let len = destination
            .space()
            .required_len()
            .map_err(Error::from)
            .map_err(FacadeError::<R>::from)?;
        let mut data = symtn_tensors::zeroed_payload::<D>(len);
        let mut runs: Vec<symtn_tensors::SectorRunTable<'_>> = vec![None; self.rank()];
        runs[axis] = Some(selection.entries.as_slice());
        let (source, source_data) = self.fusion_operand_and_data();
        symtn_operations::host_pool::region(
            engine,
            symtn_operations::host_pool::HostPoolSite::Strided,
            |exec| {
                symtn_tensors::fusion_scatter_add_assign(
                    exec,
                    destination.space().structure(),
                    &mut data,
                    self.logical_space().space().structure(),
                    source,
                    &source_data,
                    &runs,
                )
            },
        )?
        .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(destination, data)),
        })
    }
}
