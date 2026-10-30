use super::*;

/// A symmetry-graded vector space bound to one fusion-rule provider.
///
/// Each sector has a degeneracy, and the space owns this complete map even when
/// a tensor uses only some of its fusion trees. A dual space stores dual sector
/// labels and reports `true` from [`Self::is_dual`].
///
/// If a provider query fails, the returned error keeps the provider error as
/// its source. No partial result is returned.
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{Error, GradedSpace};
///
/// # fn main() -> Result<(), Error> {
/// let v = GradedSpace::try_new(
///     Arc::new(U1FusionRule),
///     [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
/// )?;
/// let w = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(-1), 1)])?;
///
/// let fused = v.fuse(&w)?;
/// assert_eq!(fused.degeneracy(&U1Irrep::new(-1))?, 2);
/// assert_eq!(fused.degeneracy(&U1Irrep::new(0))?, 1);
/// assert_eq!(v.try_dual()?.degeneracy(&U1Irrep::new(-1))?, 1);
/// # Ok(())
/// # }
/// ```
pub struct GradedSpace<R> {
    pub(super) provider: Arc<R>,
    pub(super) leg: SectorLeg,
}

// Why hand-written instead of derived: the derives would demand `R: Clone` and
// `R: Debug`, neither of which a provider needs to satisfy — the provider is
// shared behind an `Arc` and its labels, not the rule itself, are what a
// diagnostic wants to show.
impl<R> Clone for GradedSpace<R> {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
            leg: self.leg.clone(),
        }
    }
}

impl<R> PartialEq for GradedSpace<R>
where
    R: TypedSectorAdmission,
{
    fn eq(&self, other: &Self) -> bool {
        TypedSectorAdmission::typed_rule_identity(self.provider())
            == TypedSectorAdmission::typed_rule_identity(other.provider())
            && self.leg == other.leg
    }
}

impl<R> Eq for GradedSpace<R> where R: TypedSectorAdmission {}

impl<R> core::fmt::Debug for GradedSpace<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GradedSpace")
            .field("leg", &self.leg)
            .finish_non_exhaustive()
    }
}

impl<R> GradedSpace<R>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    /// Creates a nondual space from `(sector label, degeneracy)` pairs.
    ///
    /// Input order does not affect the result: sectors are stored in the
    /// provider's [`crate::sector::SectorId`] order, and zero-degeneracy entries
    /// are omitted. Call [`Self::try_dual`] to construct the dual space.
    ///
    /// The space retains `provider`. Clone one [`Arc`] into several
    /// independently constructed spaces to share one allocation (and its
    /// performance cache); spaces derived from an existing space
    /// ([`Self::try_dual`], [`Self::fuse`], [`Self::unitspace`]) already share
    /// its provider.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{Error, GradedSpace};
    ///
    /// let rule = Arc::new(U1FusionRule);
    /// let physical = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)])?;
    /// let bond = GradedSpace::try_new(rule, [(U1Irrep::new(1), 3)])?;
    /// assert_eq!(physical.fuse(&bond)?.dim()?, 6.0);
    /// let alone = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)])?;
    /// assert_eq!(alone.dim()?, 1.0);
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Complexity
    ///
    /// `O(k log k)` for `k` input pairs.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate labels or two labels with the same
    /// encoded id.
    pub fn try_new<Pairs>(provider: Arc<R>, pairs: Pairs) -> Result<Self, FacadeError<R>>
    where
        Pairs: IntoIterator<Item = (R::Sector, usize)>,
    {
        let pairs: Vec<(R::Sector, usize)> = pairs.into_iter().collect();
        // Why duplicate detection precedes `SectorLeg::try_new`: the leg only
        // ever sees encoded ids, so its own duplicate error can only name a
        // `SectorId`. Diagnosing here lets the caller see the label it wrote,
        // and separates a caller duplicate from a provider whose codec aliases
        // two labels onto one id — the leg cannot tell those apart at all.
        let mut sorted: Vec<&R::Sector> = pairs.iter().map(|(label, _)| label).collect();
        sorted.sort_unstable();
        if let Some(window) = sorted.windows(2).find(|window| window[0] == window[1]) {
            return Err(Error::InvalidArgument(format!(
                "sector label {:?} is declared more than once",
                window[0]
            ))
            .into());
        }

        let mut encoded = Vec::with_capacity(pairs.len());
        for (label, degeneracy) in &pairs {
            encoded.push((
                TypedSectorAdmission::try_encode_label(provider.as_ref(), label)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                label,
                *degeneracy,
            ));
        }
        let mut by_id: Vec<_> = encoded.iter().collect();
        by_id.sort_unstable_by_key(|(id, _, _)| *id);
        if let Some(window) = by_id.windows(2).find(|window| window[0].0 == window[1].0) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: labels {:?} and {:?} both encode to {:?}",
                window[0].1, window[1].1, window[0].0
            ))
            .into());
        }

        let leg = SectorLeg::try_new(
            encoded.iter().map(|(id, _, degeneracy)| (*id, *degeneracy)),
            false,
        )
        .map_err(|error| FacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self { provider, leg })
    }

    /// Returns sector labels parallel to [`Self::degeneracies`] in stored id
    /// order, or the provider's decode error.
    pub fn sectors(&self) -> Result<Vec<R::Sector>, FacadeError<R>> {
        self.leg
            .sectors()
            .iter()
            .map(|&id| {
                TypedSectorAdmission::try_decode_label(self.provider.as_ref(), id)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
            })
            .collect()
    }

    /// Returns a sector's degeneracy, or zero when the provider can encode the
    /// label but the space does not contain it.
    pub fn degeneracy(&self, sector: &R::Sector) -> Result<usize, FacadeError<R>> {
        let id = TypedSectorAdmission::try_encode_label(self.provider.as_ref(), sector)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        Ok(self.leg.degeneracy(id).unwrap_or(0))
    }

    /// Returns the dual space with every sector replaced by its provider dual,
    /// the same degeneracies, and the dual flag reversed.
    ///
    /// For a valid provider, applying this operation twice returns the original
    /// space. It takes `O(k log k)` for `k` stored sectors and retains the same
    /// provider instance.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when two sectors have the same dual.
    pub fn try_dual(&self) -> Result<Self, FacadeError<R>> {
        let sectors = self
            .leg
            .sectors()
            .iter()
            .copied()
            .zip(self.leg.degeneracies().iter().copied())
            .map(|(sector, degeneracy)| {
                TypedSectorAdmission::try_dual_id(self.provider.as_ref(), sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
                    .map(|dual| (dual, degeneracy))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut duals: Vec<_> = sectors.iter().map(|(sector, _)| *sector).collect();
        duals.sort_unstable();
        if let Some(duplicate) = duals.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidArgument(format!(
                "dual map is not injective: sector {:?} appears multiple times",
                duplicate[0]
            ))
            .into());
        }
        let leg = SectorLeg::try_new(sectors, !self.leg.is_dual())
            .map_err(|error| FacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self {
            provider: Arc::clone(&self.provider),
            leg,
        })
    }

    /// This leg read as a fixed truncation target: TensorKit `truncspace(V)`.
    ///
    /// The resulting [`Truncation::space`] policy keeps exactly this leg's
    /// degeneracy for every coupled sector it carries, and drops any sector it
    /// does not — TensorKit reads the same `dim(V, c)`, which is zero for an
    /// absent sector. A request longer than the spectrum offers is clamped to
    /// it. The dual flag is ignored: a truncation target names sector content,
    /// not orientation.
    ///
    /// The profile records this leg's provider identity, so handing it to a
    /// factorization of a tensor built on a different rule returns an error
    /// rather than silently truncating to nothing.
    ///
    /// # Complexity
    ///
    /// `O(k log k)` in the number of sectors (one `BTreeMap` build); no
    /// spectrum or payload is touched.
    pub fn truncspace(&self) -> TruncationSpace {
        TruncationSpace::new(
            TypedSectorAdmission::typed_rule_identity(self.provider.as_ref()),
            self.leg
                .sectors()
                .iter()
                .copied()
                .zip(self.leg.degeneracies().iter().copied()),
        )
    }
}

impl<R> GradedSpace<R>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
{
    /// Returns the quantum-dimension-weighted total dimension
    /// `sum_c degeneracy(c) * quantum_dimension(c)` without integer rounding.
    ///
    /// # Errors
    ///
    /// A checked provider's dimension error, or, for a multiplicity-free
    /// provider whose `Complex64` quantum dimension has a nonzero imaginary
    /// part, [`Error::Operation`](crate::typed::Error::Operation) wrapping
    /// [`OperationError::InvalidArgument`](crate::typed::OperationError::InvalidArgument).
    pub fn dim(&self) -> Result<f64, FacadeError<R>> {
        let mut total = 0.0;
        for (&sector, &degeneracy) in self.leg.sectors().iter().zip(self.leg.degeneracies()) {
            total += degeneracy as f64
                * <R::Mode as TypedSpaceModeDispatch<R>>::dim(self.provider(), sector)?;
        }
        Ok(total)
    }

    /// Returns the non-dual unit space: the provider's vacuum sector with
    /// degeneracy one and the same provider instance as this space.
    pub fn unitspace(&self) -> Result<Self, FacadeError<R>> {
        let vacuum = <R::Mode as TypedSpaceModeDispatch<R>>::vacuum(self.provider());
        TypedSectorAdmission::try_decode_label(self.provider(), vacuum)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        let leg = SectorLeg::try_new([(vacuum, 1)], false)
            .map_err(|error| FacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self {
            provider: Arc::clone(self.provider_arc()),
            leg,
        })
    }

    /// Fuses the two spaces and returns a non-dual space using the same
    /// provider instance as this space.
    ///
    /// If `d_a` and `d_b` are input degeneracies, the result has
    /// `d_c = sum_(a,b) d_a d_b N_ab^c`, including fusion multiplicities.
    /// The provider identities must match; this is checked before any fusion
    /// query.
    ///
    /// Returns [`Error::RuleMismatch`] when the provider identities differ, or
    /// [`Error::InvalidArgument`] if a degeneracy overflows.
    pub fn fuse(&self, other: &Self) -> Result<Self, FacadeError<R>> {
        self.require_same_identity(other)?;
        let mut fused = std::collections::BTreeMap::<SectorId, usize>::new();
        for (left, left_deg) in self.leg.iter() {
            for (right, right_deg) in other.leg.iter() {
                let pair_deg = left_deg.checked_mul(right_deg).ok_or_else(|| {
                    FacadeError::<R>::from(Error::InvalidArgument(
                        "fuse: degeneracy multiplication overflow".into(),
                    ))
                })?;
                for coupled in <R::Mode as TypedSpaceModeDispatch<R>>::fusion_channels(
                    self.provider(),
                    left,
                    right,
                )? {
                    let multiplicity = <R::Mode as TypedSpaceModeDispatch<R>>::nsymbol(
                        self.provider(),
                        left,
                        right,
                        coupled,
                    )?;
                    let contribution = pair_deg.checked_mul(multiplicity).ok_or_else(|| {
                        FacadeError::<R>::from(Error::InvalidArgument(
                            "fuse: degeneracy multiplication overflow".into(),
                        ))
                    })?;
                    let entry = fused.entry(coupled).or_insert(0);
                    *entry = entry.checked_add(contribution).ok_or_else(|| {
                        FacadeError::<R>::from(Error::InvalidArgument(format!(
                            "fuse: degeneracy overflow for sector {coupled:?}"
                        )))
                    })?;
                }
            }
        }
        fused.retain(|_, degeneracy| *degeneracy != 0);
        for &sector in fused.keys() {
            TypedSectorAdmission::try_decode_label(self.provider(), sector)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        }
        let leg = SectorLeg::try_new(fused, false)
            .map_err(|error| FacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self {
            provider: Arc::clone(self.provider_arc()),
            leg,
        })
    }

    /// Returns the direct sum, adding degeneracies of matching sectors and
    /// retaining the same provider instance and dual flag as this space.
    ///
    /// Returns [`Error::RuleMismatch`] when the provider identities differ, or
    /// `SpaceMismatch` when the dual flags differ, or
    /// [`Error::InvalidArgument`] when a degeneracy overflows.
    pub fn oplus(&self, other: &Self) -> Result<Self, FacadeError<R>> {
        self.require_same_identity(other)?;
        let leg = oplus_sector_legs(&self.leg, &other.leg).map_err(FacadeError::<R>::from)?;
        Ok(Self {
            provider: Arc::clone(self.provider_arc()),
            leg,
        })
    }

    fn require_same_identity(&self, other: &Self) -> Result<(), FacadeError<R>> {
        if TypedSectorAdmission::typed_rule_identity(self.provider())
            != TypedSectorAdmission::typed_rule_identity(other.provider())
        {
            return Err(FacadeError::<R>::from(Error::RuleMismatch));
        }
        Ok(())
    }
}

impl<R> GradedSpace<R>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTruncationDispatch<R>,
{
    /// Pads a spectrum to this whole leg with zeros: each sector of the leg
    /// gets the magnitudes `spectra` gives it, followed by zeros up to its
    /// degeneracy, and a sector `spectra` omits is all zeros.
    ///
    /// This is TensorKit's `extended_S` in
    /// `MAK.truncate(::typeof(left_null!), (U, S), strategy)`
    /// (`src/factorizations/truncation.jl`): the diagonal of a full SVD's
    /// rectangular `s` ([`TensorMap::diagview`]) covers only the leading
    /// `min(m_c, n_c)` states of `u`'s bond `W_out`, and the remaining
    /// `m_c - n_c` states, as well as every sector absent from the domain, are
    /// the rectangular zeros. The result is a complete spectrum for
    /// [`Self::find_truncated`] on that bond.
    ///
    /// Why a separate step instead of a lenient [`Self::find_truncated`]: the
    /// selection keeps its complete-spectrum check, so a short or missing
    /// spectrum is padded only where the caller says the zeros are
    /// structural.
    ///
    /// The rank-revealing null space (TensorKit
    /// `left_null(t; alg = :svd, trunc)`) is this padding composed with a
    /// keep-below selection, as [`TensorMap::left_null`] shows.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a sector is repeated, absent from this
    /// leg, or carries more values than its degeneracy. Provider
    /// label-encoding failures are returned unchanged.
    ///
    /// # Complexity
    ///
    /// `O(K + G log G)` for `K` states in `G` sectors of this leg.
    pub fn extend_spectrum<V>(
        &self,
        spectra: &[SectorSpectrum<R::Sector, V>],
    ) -> Result<Vec<SectorSpectrum<R::Sector, f64>>, FacadeError<R>>
    where
        V: SpectrumMagnitude,
    {
        let mut given = std::collections::BTreeMap::new();
        for entry in spectra {
            let sector = TypedSectorAdmission::try_encode_label(self.provider(), &entry.sector)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
            let degeneracy = self.leg.degeneracy(sector).ok_or_else(|| {
                FacadeError::<R>::from(Error::InvalidArgument(format!(
                    "sector {:?} is absent from the extended leg",
                    entry.sector
                )))
            })?;
            if entry.values.len() > degeneracy {
                return Err(Error::InvalidArgument(format!(
                    "spectrum for sector {:?} has {} values, more than the leg degeneracy {degeneracy}",
                    entry.sector,
                    entry.values.len()
                ))
                .into());
            }
            if given.insert(sector, &entry.values).is_some() {
                return Err(Error::InvalidArgument(format!(
                    "sector {:?} carries more than one spectrum",
                    entry.sector
                ))
                .into());
            }
        }
        self.leg
            .sectors()
            .iter()
            .zip(self.sectors()?)
            .zip(self.leg.degeneracies())
            .map(|((sector, label), &degeneracy)| {
                let mut values: Vec<f64> = given.get(sector).map_or_else(Vec::new, |values| {
                    values.iter().map(|value| value.magnitude()).collect()
                });
                values.resize(degeneracy, 0.0);
                Ok(SectorSpectrum {
                    sector: label,
                    values,
                })
            })
            .collect()
    }

    /// Decides which states of this leg a [`Truncation`] keeps, and how much
    /// weight it discards.
    ///
    /// This is TensorKit's `findtruncated` over a `SectorVector` plus
    /// `truncation_error` in one call: `spectra` are per-sector values in the
    /// order a symtn `*_compact` / `*_full` factorization publishes them, and
    /// the result names the surviving bond states by their stored positions. A truncated factorization is
    /// this decision composed with an untruncated one: apply it with
    /// [`TensorMap::restrict_leg`]: on the bond leg of each isometric factor,
    /// and on both legs of the spectrum factor, which keeps a compact one
    /// compact. The same
    /// recipe truncates the `d` and `v` of `eigh_full` and `eig_full`.
    ///
    /// ```
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, Svd, TensorMap, Truncation};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(std::sync::Arc::new(U1FusionRule), [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)])?;
    /// let t: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v, &v], [&v], 6)?;
    ///
    /// let Svd { u, s, vh } = t.svd_compact(engine, &[0, 1], &[2])?;
    /// let found = s.domain()[0].find_truncated(&s.diagview(engine)?, &Truncation::rank(2))?;
    /// let u = u.restrict_leg(engine, &[(u.codomain_rank(), &found.selection)])?;
    /// let s = s.restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])?;
    /// let vh = vh.restrict_leg(engine, &[(0, &found.selection)])?;
    ///
    /// // The best rank-2 approximation, and the weight it discards.
    /// let approximation = u.compose(engine, &s)?.compose(engine, &vh)?;
    /// let residual = t.axpby(engine, 1.0, &approximation, -1.0)?.norm(engine, 2.0)?;
    /// assert!((residual - found.error).abs() < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// # Naming deviation
    ///
    /// MatrixAlgebraKit's `findtruncated` is a free function over a
    /// `SectorVector`, which carries the sector structure with the values.
    /// symtn's `&[SectorSpectrum]` carries neither the fusion rule (needed for
    /// the quantum-dimension weight and for rejecting a foreign
    /// [`TruncationSpace`]) nor the degeneracies (needed to validate the input
    /// and to build the selection's parent), so the leg is the receiver: it is
    /// the Rust equivalent of the `SectorVector`'s structure.
    ///
    /// # Input contract
    ///
    /// `spectra` must name every sector of this leg exactly once, each with
    /// `values.len()` equal to that sector's degeneracy. Input order does not
    /// matter: the decision is taken in TensorKit's sector order (the
    /// provider's `sector_order_key`), so at an exact cross-sector tie the
    /// kept sector is the one TensorKit keeps.
    ///
    /// Values are selected by magnitude (`|v|`), so signed `eigh` eigenvalues
    /// and complex `eig` eigenvalues can be passed as published, in any order
    /// within a sector; non-finite values are rejected. The kept set is
    /// TensorKit's: ties in magnitude go to the earlier sector in TensorKit's
    /// order, then to the earlier position, which [`Truncation::Rank`] keeps
    /// first and [`Truncation::DiscardWeight`] discards first. Kept states
    /// stay in stored order (see [`Truncation::Space`] for the one policy
    /// where TensorKit reorders them).
    ///
    /// # Complexity
    ///
    /// Spectrum-sized only, never payload-sized, for `K = sum_c n_c` values
    /// in `G` sectors: `O(K)` to copy and validate; for [`Truncation::Rank`],
    /// [`Truncation::RankSmallest`], [`Truncation::DiscardWeight`] and
    /// [`Truncation::Space`] each sector's positions in magnitude order,
    /// `O(n_c)` when that sector is monotone in magnitude in either
    /// direction, ties included (an SVD spectrum, padded or not, or a
    /// spectrum stored by ascending magnitude), and `O(n_c log n_c)`
    /// otherwise (signed `eigh` eigenvalues stored ascending are two
    /// monotone runs in magnitude; the stable sort merges them), then
    /// `O(G + k log G)` to merge `k` kept (`Rank`, `RankSmallest`) or
    /// discarded (`DiscardWeight`) values; `O(K)` for the threshold policies
    /// ([`Truncation::Tolerance`], [`Truncation::ToleranceInf`],
    /// [`Truncation::ToleranceBelow`]: one norm and one pass) and for the
    /// error; and `O(K + G log G)` to build the selection.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a sector is missing, repeated, unknown
    /// to this leg, or carries a spectrum of the wrong length, and for a
    /// malformed policy or an invalid spectrum; [`Error::RuleMismatch`]
    /// (through the operation error) for a
    /// [`TruncationSpace`] built against another rule. Provider label-encoding
    /// failures are returned unchanged.
    pub fn find_truncated<V>(
        &self,
        spectra: &[SectorSpectrum<R::Sector, V>],
        truncation: &Truncation,
    ) -> Result<TruncatedSelection<R>, FacadeError<R>>
    where
        V: SpectrumMagnitude,
    {
        let expected = self.leg.sectors().len();
        if spectra.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "find_truncated needs one spectrum per leg sector, got {} for {expected}",
                spectra.len()
            ))
            .into());
        }
        let mut encoded = Vec::with_capacity(spectra.len());
        for entry in spectra {
            let sector = TypedSectorAdmission::try_encode_label(self.provider(), &entry.sector)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
            let degeneracy = self.leg.degeneracy(sector).ok_or_else(|| {
                FacadeError::<R>::from(Error::InvalidArgument(format!(
                    "sector {:?} is absent from the truncated leg",
                    entry.sector
                )))
            })?;
            if entry.values.len() != degeneracy {
                return Err(Error::InvalidArgument(format!(
                    "spectrum for sector {:?} has {} values, expected the leg degeneracy {degeneracy}",
                    entry.sector,
                    entry.values.len()
                ))
                .into());
            }
            encoded.push(symtn_matrixalgebra::SectorSpectrum {
                sector,
                values: entry.values.clone(),
            });
        }
        encoded.sort_unstable_by_key(|entry| entry.sector);
        if let Some(pair) = encoded
            .windows(2)
            .find(|pair| pair[0].sector == pair[1].sector)
        {
            return Err(Error::InvalidArgument(format!(
                "sector {:?} carries more than one spectrum",
                pair[0].sector
            ))
            .into());
        }
        let decision = <R::Mode as TypedTruncationDispatch<R>>::decide_bond_truncation(
            self.provider(),
            &encoded,
            truncation,
        )?;
        // An all-discarded sector is omitted; a decision that discards
        // everything is the ordinary bond leg with no sectors.
        let selection = LegSelection::from_runs(
            self,
            encoded
                .iter()
                .zip(&decision.kept)
                .filter_map(|(entry, mask)| {
                    let runs = kept_runs(mask);
                    (!runs.is_empty()).then_some((entry.sector, runs))
                })
                .collect(),
        )?;
        Ok(TruncatedSelection {
            selection,
            error: decision.error,
        })
    }
}

/// The canonical maximal runs of the `true` positions of `mask`, in `O(n)`.
fn kept_runs(mask: &[bool]) -> symtn_tensors::SelectedRuns {
    let mut runs = symtn_tensors::SelectedRuns::new();
    for (position, &kept) in mask.iter().enumerate() {
        if !kept {
            continue;
        }
        match runs.last_mut() {
            Some(run) if run.end == position => run.end += 1,
            _ => runs.push(position..position + 1),
        }
    }
    runs
}

impl<R> GradedSpace<R> {
    /// Returns per-sector degeneracies parallel to [`Self::sectors`].
    #[inline]
    pub fn degeneracies(&self) -> &[usize] {
        self.leg.degeneracies()
    }

    /// Returns whether this is the dual space.
    #[inline]
    pub fn is_dual(&self) -> bool {
        self.leg.is_dual()
    }

    /// Returns the provider bound to this space.
    #[inline]
    pub fn provider(&self) -> &R {
        self.provider.as_ref()
    }

    // Bound-free so the crate-internal accessors stay usable wherever the leg
    // travels, independently of what the caller has to certify.
    pub(crate) fn leg(&self) -> &SectorLeg {
        &self.leg
    }

    /// Raw logical leg snapshot used by typed network replay admission.
    pub(crate) fn network_sector_leg(&self) -> &SectorLeg {
        &self.leg
    }

    pub(crate) fn provider_arc(&self) -> &Arc<R> {
        &self.provider
    }
}

/// A subspace of one graded leg, given as a set of kept degeneracy positions
/// per sector.
///
/// For a leg `V = ⊕_c ℂ^{n_c} ⊗ R_c`, a selection
/// `σ = { c ↦ P_c }` with `P_c = {p_0 < … < p_{k_c − 1}} ⊆ [0, n_c)` nonempty
/// names the subspace `W = ⊕_{c ∈ σ} ℂ^{k_c} ⊗ R_c` and the inclusion
/// isometry `ι_σ : W → V`, which is the identity on every irrep and the
/// order-preserving coordinate inclusion `e_j ↦ e_{p_j}` on the degeneracy
/// spaces. Positions therefore never cut into a multiplet: for a
/// non-Abelian sector they select whole copies of `R_c`, exactly as
/// TensorKit's `dim(V, c)` counts degeneracy and its truncation keeps or
/// drops whole multiplets.
///
/// Sectors are named in the leg's **own** labels, as [`GradedSpace::sectors`]
/// reports them. For a dual leg those are already the dualised labels; a
/// caller never dualises a label itself.
///
/// A selection is validated once, against its parent leg, and then reused:
/// [`TensorMap::restrict_leg`] requires the tensor's leg to equal
/// [`Self::parent`], [`TensorMap::embed_leg`] requires it to equal
/// [`Self::subspace`].
///
/// # TensorKit correspondence
///
/// TensorKit has no per-leg selection primitive. Its production route for a
/// bond is `truncate_domain!`/`truncate_codomain!`/`truncate_diagonal!`
/// (`src/factorizations/truncation.jl`), a per-sector `view(b, :, I)` with
/// `I` a Bool mask, an ascending index vector or a range; for an arbitrary
/// leg it is a contraction with `isometry(W ← V)`, which realises only
/// leading-index selections. A `LegSelection` is the set such an `I` names,
/// stored as sorted maximal runs of positions per sector rather than as a
/// mask or index list: the same set, and the kernels copy each run as one
/// strided piece. A selection carries no order of its own, so an `I` that
/// would permute columns has no counterpart (see [`Self::try_new`]).
pub struct LegSelection<R> {
    parent: GradedSpace<R>,
    subspace: GradedSpace<R>,
    // Sorted by `SectorId`, parallel to `subspace`'s stored sectors: the
    // kernels look the runs up by the id they read from a block's own key,
    // and borrow this table instead of building one per call and axis.
    // Runs are canonical (maximal, sorted, nonempty), so equal tables are
    // equal sets.
    pub(super) entries: Vec<(SectorId, symtn_tensors::SelectedRuns)>,
}

// Why hand-written: both fields clone through an `Arc`, exactly as
// `GradedSpace` does, so the derive's `R: Clone` bound would be a provider
// requirement that nothing here actually needs.
impl<R> Clone for LegSelection<R> {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            subspace: self.subspace.clone(),
            entries: self.entries.clone(),
        }
    }
}

impl<R> core::fmt::Debug for LegSelection<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("LegSelection")
            .field("parent", &self.parent)
            .field("entries", &self.entries)
            .finish()
    }
}

impl<R> LegSelection<R>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    /// Validates `pairs` against `parent` and records the resulting subspace.
    ///
    /// Each pair names a sector and its kept degeneracy positions, strictly
    /// increasing: a range `a..b` (a contiguous selection) or any list such
    /// as `[0, 3]` or a `Vec<usize>`. Input order of the pairs does not
    /// matter; entries are stored in the provider's
    /// [`crate::sector::SectorId`] order, as the leg itself stores them.
    ///
    /// Stricter than TensorKit: `view(b, :, I)` accepts an unsorted or
    /// repeated `I` and then permutes or duplicates columns, which is not a
    /// subspace; such positions are an error here.
    ///
    /// # Complexity
    ///
    /// `O(s log s + K)` for `s` pairs naming `K` positions in total; no
    /// payload is touched.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `pairs` is empty, a label is
    /// repeated, two labels encode to the same id, a sector names no
    /// position, its positions are not strictly increasing, a sector is
    /// absent from `parent`, or a position is not below that sector's
    /// degeneracy.
    pub fn try_new<Pairs, P>(parent: &GradedSpace<R>, pairs: Pairs) -> Result<Self, FacadeError<R>>
    where
        Pairs: IntoIterator<Item = (R::Sector, P)>,
        P: IntoIterator<Item = usize>,
    {
        let pairs: Vec<(R::Sector, P)> = pairs.into_iter().collect();
        if pairs.is_empty() {
            return Err(Error::InvalidArgument(
                "leg selection must name at least one sector".to_string(),
            )
            .into());
        }
        // Diagnose duplicates on the label, not on the encoded id: the leg can
        // only ever name a `SectorId`, which would not tell a caller's repeat
        // apart from a codec that aliases two labels onto one id.
        let mut sorted: Vec<&R::Sector> = pairs.iter().map(|(label, _)| label).collect();
        sorted.sort_unstable();
        if let Some(window) = sorted.windows(2).find(|window| window[0] == window[1]) {
            return Err(Error::InvalidArgument(format!(
                "sector label {:?} is selected more than once",
                window[0]
            ))
            .into());
        }

        let mut entries = Vec::with_capacity(pairs.len());
        for (label, positions) in pairs {
            let id = TypedSectorAdmission::try_encode_label(parent.provider(), &label)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
            let degeneracy = parent.leg().degeneracy(id).ok_or_else(|| {
                FacadeError::<R>::from(Error::InvalidArgument(format!(
                    "sector {label:?} is absent from the selected leg"
                )))
            })?;
            let mut runs = symtn_tensors::SelectedRuns::new();
            for position in positions {
                if position >= degeneracy {
                    return Err(Error::InvalidArgument(format!(
                        "leg selection position {position} for sector {label:?} exceeds its \
                         degeneracy {degeneracy}"
                    ))
                    .into());
                }
                match runs.last_mut() {
                    Some(run) if position == run.end => run.end += 1,
                    Some(run) if position < run.end => {
                        return Err(Error::InvalidArgument(format!(
                            "leg selection positions for sector {label:?} must be strictly \
                             increasing, got {position} after {}",
                            run.end - 1
                        ))
                        .into());
                    }
                    _ => runs.push(position..position + 1),
                }
            }
            if runs.is_empty() {
                return Err(Error::InvalidArgument(format!(
                    "leg selection for sector {label:?} must name at least one position"
                ))
                .into());
            }
            entries.push((id, runs));
        }
        entries.sort_unstable_by_key(|(id, _)| *id);
        if let Some(window) = entries.windows(2).find(|window| window[0].0 == window[1].0) {
            return Err(Error::InvalidArgument(format!(
                "SectorCodec law violation: two selected labels both encode to {:?}",
                window[0].0
            ))
            .into());
        }
        Self::from_runs(parent, entries)
    }

    /// Records a selection whose `entries` are already valid for `parent`:
    /// ascending by [`SectorId`], each sector of `parent`, each with
    /// canonical runs inside its degeneracy, which [`Self::try_new`] and
    /// [`GradedSpace::find_truncated`] have established.
    ///
    /// Why `entries` may be empty here but not in `try_new`: a decision that
    /// discards everything leaves no sector to name, and that outcome is an
    /// ordinary bond leg with no sectors. An empty list from a *caller* is
    /// almost always a bug, so `try_new` keeps rejecting it.
    fn from_runs(
        parent: &GradedSpace<R>,
        entries: Vec<(SectorId, symtn_tensors::SelectedRuns)>,
    ) -> Result<Self, FacadeError<R>> {
        // The subspace is built on the parent's own dual flag and stored ids:
        // going through `GradedSpace::try_new` + `try_dual` would dualise the
        // labels and name a different leg.
        let leg = SectorLeg::try_new(
            entries
                .iter()
                .map(|(id, runs)| (*id, runs.iter().map(ExactSizeIterator::len).sum())),
            parent.is_dual(),
        )
        .map_err(|error| FacadeError::<R>::from(Error::InvalidArgument(error.to_string())))?;
        Ok(Self {
            parent: parent.clone(),
            subspace: GradedSpace {
                provider: Arc::clone(parent.provider_arc()),
                leg,
            },
            entries,
        })
    }

    /// The kept degeneracy positions of `sector`, ascending: empty for a
    /// sector of the parent leg that is not selected.
    ///
    /// This is the index set TensorKit's `findtruncated` returns as `ind`,
    /// in stored order, for mapping the kept states back to the parent leg
    /// (for example, which eigenvalues survived a truncation).
    ///
    /// # Complexity
    ///
    /// `O(k + log s)` for `k` kept positions of `sector` and `s` selected
    /// sectors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `sector` is absent from the parent
    /// leg; provider label-encoding failures are returned unchanged.
    pub fn positions(&self, sector: &R::Sector) -> Result<Vec<usize>, FacadeError<R>> {
        let id = TypedSectorAdmission::try_encode_label(self.parent.provider(), sector)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
        if self.parent.leg().degeneracy(id).is_none() {
            return Err(Error::InvalidArgument(format!(
                "sector {sector:?} is absent from the selection's parent leg"
            ))
            .into());
        }
        Ok(self
            .entries
            .binary_search_by_key(&id, |(candidate, _)| *candidate)
            .map(|index| self.entries[index].1.iter().cloned().flatten().collect())
            .unwrap_or_default())
    }
}

/// Checks that leg `axis` of `space` can be exchanged for `expected`: the
/// axis exists, the rules agree, and the leg is `expected`. Shared by the
/// eager and the stacked restriction, so both reject with the same errors.
pub(super) fn require_selected_leg_of<R>(
    space: &BoundDynamicFusionMapSpace<R>,
    axis: usize,
    expected: &GradedSpace<R>,
    operation: &str,
) -> Result<(), FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let homspace = space.space().homspace();
    let codomain_rank = homspace.codomain().len();
    let rank = codomain_rank + homspace.domain().len();
    if axis >= rank {
        return Err(Error::InvalidArgument(format!(
            "{operation}: axis {axis} is out of range for rank {rank}"
        ))
        .into());
    }
    if TypedSectorAdmission::typed_rule_identity(space.provider())
        != TypedSectorAdmission::typed_rule_identity(expected.provider())
    {
        return Err(Error::RuleMismatch.into());
    }
    let leg = if axis < codomain_rank {
        &homspace.codomain().legs()[axis]
    } else {
        &homspace.domain().legs()[axis - codomain_rank]
    };
    if leg != expected.leg() {
        return Err(Error::InvalidArgument(format!(
            "{operation}: axis {axis} is not the leg this selection was built from"
        ))
        .into());
    }
    Ok(())
}

/// `space`'s hom-space with leg `axis` replaced by `replacement(axis)` where
/// that is `Some`, built into a root layout of the same provider.
pub(super) fn space_with_replaced_legs<'a, R>(
    space: &BoundDynamicFusionMapSpace<R>,
    replacement: impl Fn(usize) -> Option<&'a SectorLeg>,
) -> Result<BoundDynamicFusionMapSpace<R>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
{
    let homspace = space.space().homspace();
    let codomain_rank = homspace.codomain().len();
    let replaced = |product: &FusionProductSpace, base: usize| {
        FusionProductSpace::new(
            product
                .legs()
                .iter()
                .enumerate()
                .map(|(offset, leg)| replacement(base + offset).unwrap_or(leg).clone())
                .collect::<Vec<_>>(),
        )
    };
    let homspace = FusionTreeHomSpace::new(
        replaced(homspace.codomain(), 0),
        replaced(homspace.domain(), codomain_rank),
    );
    <R::Mode as TypedTensorRootDispatch<R>>::build_root(Arc::clone(space.provider_arc()), homspace)
}

/// The kernel's per-axis run tables for a restriction set on a tensor of
/// rank `rank`, borrowed from the selections.
pub(super) fn restriction_runs<'a, R>(
    rank: usize,
    legs: &[(usize, &'a LegSelection<R>)],
) -> Vec<symtn_tensors::SectorRunTable<'a>> {
    let mut runs = vec![None; rank];
    for &(axis, selection) in legs {
        runs[axis] = Some(selection.entries.as_slice());
    }
    runs
}

/// Checks a `restrict_leg` set against `space`. Shared by the eager and the
/// stacked restriction, so both reject with the same errors; each pair is
/// checked exactly as a single-axis restriction is.
pub(super) fn require_restriction_set<R>(
    space: &BoundDynamicFusionMapSpace<R>,
    legs: &[(usize, &LegSelection<R>)],
) -> Result<(), FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
{
    // Why not accept an empty set as the identity: it would be a second
    // spelling of `materialize`.
    if legs.is_empty() {
        return Err(Error::InvalidArgument(
            "restrict_leg needs at least one (axis, selection) pair".to_string(),
        )
        .into());
    }
    for (index, &(axis, selection)) in legs.iter().enumerate() {
        require_selected_leg_of(space, axis, selection.parent(), "restrict_leg")?;
        if legs[..index].iter().any(|&(seen, _)| seen == axis) {
            return Err(
                Error::InvalidArgument(format!("restrict_leg: axis {axis} appears twice")).into(),
            );
        }
    }
    Ok(())
}

/// `space` with every leg of a checked `restrict_leg` set replaced by its
/// selection's subspace.
pub(super) fn restricted_space<R>(
    space: &BoundDynamicFusionMapSpace<R>,
    legs: &[(usize, &LegSelection<R>)],
) -> Result<BoundDynamicFusionMapSpace<R>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
{
    space_with_replaced_legs(space, |axis| {
        legs.iter()
            .find(|&&(candidate, _)| candidate == axis)
            .map(|&(_, selection)| selection.subspace().leg())
    })
}

/// The outcome of [`GradedSpace::find_truncated`]: what survives, and the norm
/// of what does not.
///
/// MatrixAlgebraKit returns the same pair as `ind` from `findtruncated` plus
/// `truncation_error!`; symtn names the index set a [`LegSelection`] so that
/// [`TensorMap::restrict_leg`] can apply it to the three factors of one
/// decomposition.
pub struct TruncatedSelection<R> {
    /// The kept bond subspace. It is empty exactly when the policy discarded
    /// every state, as `Rank(0)` does.
    pub selection: LegSelection<R>,
    /// MatrixAlgebraKit `truncation_error`: `sqrt(sum_c dim(c) sum_discarded v^2)`.
    pub error: f64,
}

// Hand-written for the reason `LegSelection`'s are: the derives would demand
// `R: Clone`/`R: Debug`, and the provider lives behind an `Arc`.
impl<R> Clone for TruncatedSelection<R> {
    fn clone(&self) -> Self {
        Self {
            selection: self.selection.clone(),
            error: self.error,
        }
    }
}

impl<R> core::fmt::Debug for TruncatedSelection<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("TruncatedSelection")
            .field("selection", &self.selection)
            .field("error", &self.error)
            .finish()
    }
}

impl<R> LegSelection<R> {
    /// The leg this selection was validated against.
    #[inline]
    pub fn parent(&self) -> &GradedSpace<R> {
        &self.parent
    }

    /// The selected subspace: the selected sectors with degeneracy equal to
    /// their number of kept positions, the parent's dual flag, and the
    /// parent's provider.
    #[inline]
    pub fn subspace(&self) -> &GradedSpace<R> {
        &self.subspace
    }

    /// Whether this selection is the whole parent leg, so that restricting
    /// with it would only copy.
    ///
    /// True exactly when every parent sector keeps all of its degeneracy
    /// positions — including the degenerate case of a parent with no sectors,
    /// where the only selection is also the whole leg.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.subspace.leg == self.parent.leg
    }
}
