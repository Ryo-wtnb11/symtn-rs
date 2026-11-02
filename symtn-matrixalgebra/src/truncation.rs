//! Spectrum truncation policies for the fusion-tensor factorizations.
//!
//! Design (informed by MatrixAlgebraKit's `TruncationStrategy`, but
//! intentionally narrower): every policy here is a magnitude rule over
//! per-sector spectra of non-negative magnitudes in the factorization's
//! stored order, which need not be descending (`eigh` publishes ascending
//! signed eigenvalues, `eig` an arbitrary order). A decision is TensorKit's
//! `findtruncated` kept set: a per-sector mask over the stored positions,
//! computed host-side as a pure scalar computation. Arbitrary filters and
//! signed eigenvalue windows are not policies here; `by = abs, rev = true` is
//! fixed.
//!
//! All budgets are weighted by the coupled sector's quantum dimension: one
//! kept value of an SU(2) spin-j sector consumes `2j + 1` of a rank budget
//! and contributes `(2j + 1) * value^2` to the 2-norm.
//!
//! An exact tie between sectors goes to the sector TensorKit's `isless`
//! puts first: [`Truncation::Rank`] keeps it first and
//! [`Truncation::DiscardWeight`] discards it first, as TensorKit's stable
//! `sortperm` over its sorted `SectorVector` does. The order comes from the
//! provider's `sector_order_key`; `docs/sector_id_compatibility.md` records it
//! for every built-in provider.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;

use symtn_core::{RuleIdentity, SectorId, SectorOrderKey};

/// A fixed per-sector rank, TensorKit's `TruncationSpace`
/// (`src/factorizations/truncation.jl:261-269`).
///
/// TensorKit reads the target rank of coupled sector `c` as
/// `dim(strategy.space, c)` — the *reduced* (per-sector degeneracy) dimension
/// of a target space — and then applies a plain `truncrank` inside that block:
/// the `dim(strategy.space, c)` largest magnitudes of the sector.
///
/// Build one from a space via `GradedSpace::truncspace` rather than by hand:
/// the sector keys are the engine's opaque
/// [`SectorId`]s, so the space that produced them is the only honest source,
/// and the [`RuleIdentity`] recorded alongside is what lets the factorization
/// reject a profile built against a different fusion rule.
#[derive(Clone, Debug, PartialEq)]
pub struct TruncationSpace {
    rule: RuleIdentity,
    ranks: BTreeMap<SectorId, usize>,
}

impl TruncationSpace {
    /// Builds a profile from a rule identity and its `(sector, rank)` pairs.
    ///
    /// Intended for the facade adapters, which take both from one space. A
    /// sector missing from `ranks` is truncated away entirely (rank zero),
    /// matching TensorKit: `dim(V, c)` of an absent sector is zero.
    pub fn new(rule: RuleIdentity, ranks: impl IntoIterator<Item = (SectorId, usize)>) -> Self {
        Self {
            rule,
            ranks: ranks.into_iter().collect(),
        }
    }

    /// The fusion rule this profile's sector ids belong to.
    pub fn rule(&self) -> &RuleIdentity {
        &self.rule
    }

    /// The requested rank of a coupled sector; `0` when the sector is absent.
    pub fn rank(&self, sector: SectorId) -> usize {
        self.ranks.get(&sector).copied().unwrap_or(0)
    }
}

/// Truncation policy over per-sector spectra of magnitudes in stored order.
///
/// Ordering policies ([`Truncation::Rank`], [`Truncation::DiscardWeight`],
/// [`Truncation::Space`]) follow TensorKit's flat stable `sortperm` over its
/// `SectorVector`: the keep order is `|v|` descending, then sector order, then
/// position ascending; the discard order is `|v|` ascending, then sector
/// order, then position ascending. So at an exact tie within a sector,
/// `Rank` keeps the earlier position and `DiscardWeight` discards the earlier
/// position.
///
/// # Tolerances and the payload's precision
///
/// Every tolerance here is `f64` and every spectrum reaching a decision is
/// `f64`, at *every* payload dtype: [`crate::FactorScalar::real_spectrum`]
/// widens a single-precision spectrum rather than recomputing it. The `f64`
/// type therefore says nothing about how accurate the values are. A spectrum
/// produced by an `f32`/`Complex32` factorization carries a relative error of
/// order `f32::EPSILON` times the block's condition number, so a cutoff chosen
/// at `f64` scale keeps that noise, and two values closer together than that
/// noise are not ordered reliably.
///
/// Every policy here keeps its postcondition against the spectrum it was
/// handed — a kept value is at or above that run's threshold, the reported
/// error is the weighted 2-norm of what that run discarded — at every payload
/// dtype. Comparing two *runs* is what single precision can break, and by how
/// much depends on the policy: [`Truncation::Rank`] at a tie swaps two
/// interchangeable states, so the kept count and the discarded weight survive;
/// [`Truncation::Tolerance`] and [`Truncation::DiscardWeight`] have a
/// boundary, and a value within noise of the threshold, or a tail whose
/// cumulative weight is within noise of the budget, is kept by one run and
/// dropped by the other — a whole state's difference in both the count and the
/// weight. MatrixAlgebraKit scales its own default with the element type
/// (`src/common/defaults.jl` `defaulttol(x) = eps(real(float(one(eltype(x)))))^(2/3)`);
/// symtn has no defaults, so the scaling is the caller's.
#[derive(Clone, Debug, PartialEq)]
pub enum Truncation {
    /// Keep everything.
    Full,
    /// Keep the largest values while the quantum-dimension-weighted total
    /// dimension stays at or below the bound.
    Rank(usize),
    /// Discard values below `max(atol, rtol * norm)`, where `norm` is the
    /// weighted 2-norm of the full spectrum.
    #[non_exhaustive]
    Tolerance { atol: f64, rtol: f64 },
    /// Discard values below `max(atol, rtol * normInf)`, where `normInf` is the
    /// unweighted maximum value. This matches TensorKit `trunctol(..., p=Inf)`.
    #[non_exhaustive]
    ToleranceInf { atol: f64, rtol: f64 },
    /// Keep the values at or below `max(atol, rtol * norm)`, `norm` the
    /// weighted 2-norm of the full spectrum: MatrixAlgebraKit
    /// `trunctol(; atol, rtol, keep_below = true)`, the complement side of
    /// [`Truncation::Tolerance`] with the threshold itself kept.
    #[non_exhaustive]
    ToleranceBelow { atol: f64, rtol: f64 },
    /// Discard the smallest values while the weighted 2-norm of everything
    /// discarded stays at or below `rtol * norm`.
    #[non_exhaustive]
    DiscardWeight { rtol: f64 },
    /// Keep the smallest values while the quantum-dimension-weighted total
    /// dimension stays at or below the bound: MatrixAlgebraKit
    /// `truncrank(rank; rev = false)`, [`Truncation::Rank`] in the discard
    /// order.
    RankSmallest(usize),
    /// Keep the requested number of largest magnitudes in every coupled
    /// sector, clamped to what the spectrum offers. TensorKit
    /// `TruncationSpace`.
    ///
    /// Deviation: the kept states stay in stored order. TensorKit returns
    /// `sortperm(d; by=abs, rev=true)[1:k]` here and so also permutes the kept
    /// columns into magnitude order; a mask names the same set without an
    /// order. TensorKit is itself not uniform here: its `truncrank`, and
    /// `truncspace & truncrank`, return a mask in stored order. For a
    /// descending (SVD) spectrum both are the leading `k` states.
    Space(TruncationSpace),
    /// Keep a value only if every component keeps it: the per-sector
    /// intersection of the kept sets.
    All(Vec<Truncation>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TruncationError {
    InvalidPolicy {
        message: &'static str,
    },
    InvalidSpectrum {
        message: &'static str,
    },
    /// A [`Truncation::Space`] profile was built against a different fusion
    /// rule than the tensor being factorized, so its [`SectorId`] keys name
    /// different sectors than the spectra do.
    RuleMismatch,
}

impl fmt::Display for TruncationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy { message } => write!(f, "invalid truncation policy: {message}"),
            Self::InvalidSpectrum { message } => {
                write!(f, "invalid truncation spectrum: {message}")
            }
            Self::RuleMismatch => write!(
                f,
                "truncation space profile was built for a different fusion rule"
            ),
        }
    }
}

impl std::error::Error for TruncationError {}

impl From<TruncationError> for symtn_tensors::OperationError {
    fn from(error: TruncationError) -> Self {
        Self::InvalidArgument {
            // Policy and spectrum details already have the required static lifetime.
            message: match error {
                TruncationError::RuleMismatch => {
                    "truncation space profile was built for a different fusion rule"
                }
                TruncationError::InvalidPolicy { message }
                | TruncationError::InvalidSpectrum { message } => message,
            },
        }
    }
}

impl Truncation {
    /// Keep at most `rank` weighted dimensions.
    pub fn rank(rank: usize) -> Self {
        Self::Rank(rank)
    }

    /// Discard values below the absolute cutoff.
    pub fn absolute_cutoff(atol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            atol,
            "tolerance absolute cutoff must be finite and non-negative",
        )?;
        Ok(Self::Tolerance { atol, rtol: 0.0 })
    }

    /// Discard values below `rtol` times the weighted 2-norm.
    pub fn relative_cutoff(rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            rtol,
            "tolerance relative cutoff must be finite and non-negative",
        )?;
        Ok(Self::Tolerance { atol: 0.0, rtol })
    }

    /// Keep values at or below `max(atol, rtol * norm)`, `norm` the weighted
    /// 2-norm (MatrixAlgebraKit `trunctol(; atol, rtol, keep_below = true)`).
    ///
    /// Why one constructor for both tolerances, unlike
    /// [`Self::absolute_cutoff`]/[`Self::relative_cutoff`]: kept-below sets
    /// combine by union under `max`, and [`Self::and`] intersects.
    ///
    /// With [`Self::rank_smallest`] this is MatrixAlgebraKit's
    /// `null_truncation_strategy(; atol, rtol, maxnullity)`
    /// (`src/interface/truncation.jl`), the selection of a rank-revealing
    /// null space over a full SVD's spectrum padded with its rectangular
    /// zeros (TensorKit `src/factorizations/truncation.jl`
    /// `MAK.truncate(::typeof(left_null!), ...)`). MatrixAlgebraKit's default
    /// for that use is `rtol = eps(real(T))^(2/3)` (`defaulttol`).
    pub fn below(atol: f64, rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            atol,
            "keep-below absolute cutoff must be finite and non-negative",
        )?;
        validate_nonnegative_finite(
            rtol,
            "keep-below relative cutoff must be finite and non-negative",
        )?;
        Ok(Self::ToleranceBelow { atol, rtol })
    }

    /// Keep at most `rank` weighted dimensions of the smallest values
    /// (MatrixAlgebraKit `truncrank(rank; rev = false)`).
    pub fn rank_smallest(rank: usize) -> Self {
        Self::RankSmallest(rank)
    }

    /// Discard values below `rtol` times the largest value.
    pub fn relative_inf_cutoff(rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            rtol,
            "infinity-norm relative cutoff must be finite and non-negative",
        )?;
        Ok(Self::ToleranceInf { atol: 0.0, rtol })
    }

    /// Bound the relative truncation error (weighted 2-norm of the discarded
    /// tail) by `rtol`: `error <= rtol * norm` up to a rounding slack of
    /// `(n + 5) * f64::EPSILON` relative to the budget (`n` the total number
    /// of values).
    ///
    /// A state whose discard meets the budget exactly, up to that rounding,
    /// is discarded (TensorKit `SectorVector` `TruncationByError`, which
    /// breaks on `> budget`; MatrixAlgebraKit's strict `>=` would keep it).
    /// `rtol = 0` therefore discards exactly the zero values and nothing
    /// else.
    pub fn relative_error(rtol: f64) -> Result<Self, TruncationError> {
        validate_nonnegative_finite(
            rtol,
            "discard-weight tolerance must be finite and non-negative",
        )?;
        Ok(Self::DiscardWeight { rtol })
    }

    /// Keep the per-sector ranks `profile` names (TensorKit `truncspace`).
    pub fn space(profile: TruncationSpace) -> Self {
        Self::Space(profile)
    }

    /// Intersects two policies (both must keep a value).
    pub fn and(self, other: Truncation) -> Self {
        match (self, other) {
            (Truncation::Full, other) => other,
            (this, Truncation::Full) => this,
            (Truncation::All(mut components), Truncation::All(others)) => {
                components.extend(others);
                Truncation::All(components)
            }
            (Truncation::All(mut components), other) => {
                components.push(other);
                Truncation::All(components)
            }
            (this, Truncation::All(mut components)) => {
                components.insert(0, this);
                Truncation::All(components)
            }
            (this, other) => Truncation::All(vec![this, other]),
        }
    }
}

/// One coupled sector's spectrum offered to the selection: its identity, its
/// quantum dimension and its non-negative magnitudes, in the order the
/// factorization stored them.
///
/// `sector` is only read by [`Truncation::Space`], the one policy whose
/// decision is per-sector rather than magnitude-driven; every other policy
/// stays identity-blind.
#[derive(Clone, Copy, Debug)]
pub struct WeightedSpectrum<'a> {
    pub sector: SectorId,
    pub weight: f64,
    pub values: &'a [f64],
}

/// The outcome of a truncation decision: per sector, a mask over its stored
/// values (`kept[i][j]`: whether `spectra[i].values[j]` is kept, TensorKit's
/// `SectorVector{Bool}`), and the weighted 2-norm of everything discarded.
///
/// One `Vec<bool>` per sector rather than a flat mask plus offsets: the
/// caller turns each sector's mask into that sector's kept positions, and
/// the per-sector shape makes the pairing with `spectra` the indexing itself.
#[derive(Clone, Debug, PartialEq)]
pub struct TruncationDecision {
    pub kept: Vec<Vec<bool>>,
    pub error: f64,
}

/// Selects the kept positions per sector for `truncation` over `spectra`.
///
/// Host-side scalar computation by design: spectra are small compared to the
/// tensors, so the decision never needs to touch device data.
///
/// `rule` is the fusion rule the caller's [`WeightedSpectrum::sector`] ids
/// belong to. It is checked against every [`Truncation::Space`] profile in
/// `truncation` *before* any selection runs, so a profile built from another
/// rule's space is rejected rather than silently reading its sector ids as
/// this rule's — which would truncate to rank zero at random.
///
/// Why the rule is a parameter rather than a check each factorization makes
/// for itself: this is the single seam every truncated factorization already
/// routes through, so putting the guard here is the one place a future caller
/// cannot forget it.
///
/// # Errors
///
/// [`TruncationError::RuleMismatch`] for a foreign profile,
/// [`TruncationError::InvalidPolicy`] for a policy with a non-finite or
/// negative tolerance, [`TruncationError::InvalidSpectrum`] for spectra that
/// are not finite and non-negative, or that repeat a sector.
///
/// # Complexity
///
/// For `K` values in `G` sectors: `O(K)` to validate and for the error, plus
/// per sector its positions in keep or discard order for the ordering
/// policies — `O(n_c)` when the sector is monotone in magnitude in either
/// direction, ties included (any SVD spectrum), `O(n_c log n_c)` otherwise
/// — then `O(G + k log G)` to merge `k` kept (`Rank`, `RankSmallest`) or
/// discarded (`DiscardWeight`) candidates. The threshold policies
/// (`Tolerance`, `ToleranceInf`, `ToleranceBelow`) are one norm and one pass,
/// `O(K)`.
///
/// The decision does not depend on the order of `spectra`: it is taken in
/// ascending `order_key(sector)` order and `kept` is reported in the
/// caller's order. That order breaks exact cross-sector ties: the earlier
/// sector is kept first by [`Truncation::Rank`] and discarded first by
/// [`Truncation::DiscardWeight`], as TensorKit's stable `sortperm` over its
/// `SectorVector` does. Callers pass the rule's
/// [`FusionRule::sector_order_key`](symtn_core::FusionRule::sector_order_key),
/// so a tie keeps the sector TensorKit keeps.
pub fn select_truncation(
    spectra: &[WeightedSpectrum<'_>],
    truncation: &Truncation,
    rule: &RuleIdentity,
    order_key: impl Fn(SectorId) -> SectorOrderKey,
) -> Result<TruncationDecision, TruncationError> {
    validate_rule(truncation, rule)?;
    validate_truncation(truncation)?;
    validate_spectra(spectra)?;
    // Cross-sector ties go to the earlier slice position and the norms sum in
    // slice order, so the slice is put in TensorKit's sector order first.
    // Why sort instead of rejecting: the id-ordered typed facade
    // legitimately feeds other orders, and the O(G log G) permutation
    // of G slice headers, one key per sector, is spectrum-free work. The
    // `SectorId` after the key only makes the order total for a provider
    // whose keys break the distinctness contract.
    let key = |spectrum: &WeightedSpectrum<'_>| (order_key(spectrum.sector), spectrum.sector);
    if spectra.windows(2).all(|pair| key(&pair[0]) < key(&pair[1])) {
        return Ok(decide(spectra, truncation));
    }
    let mut order: Vec<usize> = (0..spectra.len()).collect();
    order.sort_by_cached_key(|&index| key(&spectra[index]));
    if order
        .windows(2)
        .any(|pair| spectra[pair[0]].sector == spectra[pair[1]].sector)
    {
        return Err(TruncationError::InvalidSpectrum {
            message: "each sector may carry only one spectrum",
        });
    }
    let sorted: Vec<WeightedSpectrum<'_>> = order.iter().map(|&index| spectra[index]).collect();
    let ascending = decide(&sorted, truncation);
    let mut kept = vec![Vec::new(); spectra.len()];
    for (&index, mask) in order.iter().zip(ascending.kept) {
        kept[index] = mask;
    }
    Ok(TruncationDecision {
        kept,
        error: ascending.error,
    })
}

fn decide(spectra: &[WeightedSpectrum<'_>], truncation: &Truncation) -> TruncationDecision {
    let kept = kept_masks(spectra, truncation);
    let error = discarded_norm(spectra, &kept);
    TruncationDecision { kept, error }
}

/// Rejects every [`Truncation::Space`] profile in `truncation` that was built
/// against a rule other than `rule`. Recurses through [`Truncation::All`]
/// because [`Truncation::and`] can bury a profile inside a composite.
fn validate_rule(truncation: &Truncation, rule: &RuleIdentity) -> Result<(), TruncationError> {
    match truncation {
        Truncation::Space(profile) => {
            if profile.rule == *rule {
                Ok(())
            } else {
                Err(TruncationError::RuleMismatch)
            }
        }
        Truncation::All(components) => components
            .iter()
            .try_for_each(|component| validate_rule(component, rule)),
        _ => Ok(()),
    }
}

fn validate_nonnegative_finite(value: f64, message: &'static str) -> Result<(), TruncationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TruncationError::InvalidPolicy { message })
    }
}

fn validate_truncation(truncation: &Truncation) -> Result<(), TruncationError> {
    match truncation {
        // A `Space` profile carries only `usize` ranks and a rule identity;
        // the rule is checked by `validate_rule` and there is no numeric
        // domain left to reject here.
        Truncation::Full
        | Truncation::Rank(_)
        | Truncation::RankSmallest(_)
        | Truncation::Space(_) => Ok(()),
        Truncation::ToleranceBelow { atol, rtol } => {
            validate_nonnegative_finite(
                *atol,
                "keep-below absolute cutoff must be finite and non-negative",
            )?;
            validate_nonnegative_finite(
                *rtol,
                "keep-below relative cutoff must be finite and non-negative",
            )
        }
        Truncation::Tolerance { atol, rtol } => {
            validate_nonnegative_finite(
                *atol,
                "tolerance absolute cutoff must be finite and non-negative",
            )?;
            validate_nonnegative_finite(
                *rtol,
                "tolerance relative cutoff must be finite and non-negative",
            )
        }
        Truncation::ToleranceInf { atol, rtol } => {
            validate_nonnegative_finite(
                *atol,
                "infinity-norm absolute cutoff must be finite and non-negative",
            )?;
            validate_nonnegative_finite(
                *rtol,
                "infinity-norm relative cutoff must be finite and non-negative",
            )
        }
        Truncation::DiscardWeight { rtol } => validate_nonnegative_finite(
            *rtol,
            "discard-weight tolerance must be finite and non-negative",
        ),
        Truncation::All(components) => {
            for component in components {
                validate_truncation(component)?;
            }
            Ok(())
        }
    }
}

fn validate_spectra(spectra: &[WeightedSpectrum<'_>]) -> Result<(), TruncationError> {
    for spectrum in spectra {
        if !spectrum.weight.is_finite() || spectrum.weight <= 0.0 {
            return Err(TruncationError::InvalidSpectrum {
                message: "sector weight must be finite and positive",
            });
        }
        for &value in spectrum.values {
            if !value.is_finite() || value < 0.0 {
                return Err(TruncationError::InvalidSpectrum {
                    message: "spectrum values must be finite and non-negative",
                });
            }
        }
    }
    Ok(())
}

fn kept_masks(spectra: &[WeightedSpectrum<'_>], truncation: &Truncation) -> Vec<Vec<bool>> {
    match truncation {
        Truncation::Full => uniform_masks(spectra, true),
        // TensorKit `findtruncated(::SectorVector, ::TruncationByOrder)`
        // (truncation.jl:171-203): walk the flat keep order and mark until the
        // running `dim(c)` sum overflows. Here each sector's keep order is
        // merged by a heap whose head ties go to the lower sector, which is
        // the same flat order.
        Truncation::Rank(rank) => rank_masks(spectra, *rank, KEEP),
        // The same walk in TensorKit's `rev = false` order.
        Truncation::RankSmallest(rank) => rank_masks(spectra, *rank, DISCARD),
        // MatrixAlgebraKit `findtruncated(::AbstractVector, ::TruncationByValue)`
        // is `findall(>=(atol) ∘ abs, d)`: a filter, whatever the order.
        Truncation::Tolerance { atol, rtol } => {
            threshold_masks(spectra, atol.max(rtol * full_norm(spectra)))
        }
        Truncation::ToleranceInf { atol, rtol } => {
            threshold_masks(spectra, atol.max(rtol * full_norm_inf(spectra)))
        }
        // MatrixAlgebraKit's `keep_below` filter is `≤(atol)`.
        Truncation::ToleranceBelow { atol, rtol } => {
            let threshold = atol.max(rtol * full_norm(spectra));
            spectra
                .iter()
                .map(|spectrum| {
                    spectrum
                        .values
                        .iter()
                        .map(|&value| value <= threshold)
                        .collect()
                })
                .collect()
        }
        // TensorKit `findtruncated(::SectorVector, ::TruncationByError)`
        // (truncation.jl:227-256): ascending raw magnitude in the flat stable
        // `sortperm` (equal values: lower sector, then lower position, go
        // first), weight only in the squared error, stop at the first
        // candidate that overflows. The budget is `(rtol * norm)^2` as in
        // MatrixAlgebraKit `_truncerr_impl`, not TensorKit's
        // `rtol^p * norm(values, p)` (truncation.jl:230), so that
        // `relative_error(rtol)` means `error <= rtol * norm` literally.
        //
        // Each sector's discard order is merged by a call-local min-heap on
        // (value asc, sector asc): O(G + D log G) selection after the
        // per-sector sorts, with O(G) heap workspace instead of rescanning G
        // sectors per discard (O(D * G)).
        Truncation::DiscardWeight { rtol } => {
            let norm = full_norm(spectra);
            let bound = rtol * norm;
            let budget = bound * bound;
            let values: usize = spectra.iter().map(|spectrum| spectrum.values.len()).sum();
            let slack = 1.0 + (values + 5) as f64 * f64::EPSILON;
            // In range, the historical unscaled comparison. Out of range
            // (#1440), `BudgetUnits` compares in exact power-of-two units
            // instead: unscaled, `budget` or `limit` is `Inf` near 1e154
            // (`Inf > Inf` never stops the scan, so everything is discarded),
            // and `budget` is `0` or `rtol * norm` rounds to the subnormal
            // grid near 1e-162 (a value whose square underflows, or one within
            // that rounding of the bound, is discarded). An infinite `bound`
            // needs `rtol > 1`, so discarding everything is the right answer
            // and stays unscaled.
            let in_range = bound.is_infinite()
                || (budget >= UNSCALED_POWER_SUM_MIN && (budget * slack).is_finite());
            let units = (!in_range).then(|| BudgetUnits::new(*rtol, norm));
            let budget = units.map_or(budget, |units| units.budget);
            // Slack for the rounding of the two quantities compared, both of
            // order `budget` (`u = eps / 2`, `n` values, Higham gamma bounds
            // in this exact evaluation order):
            // - `discarded`: each term `(w * v) * v` rounds twice, then at
            //   most `n - 1` sequential additions: `(n + 1)u`.
            // - `norm^2`: `v * v`, the per-sector sum, `weight *` and the sum
            //   over sectors: `(n + 1)u`. `budget = (rtol * sqrt(.))^2`
            //   squares the sqrt and the `rtol *` roundings (`4u`) and rounds
            //   once more: `(n + 6)u`.
            // - `limit = budget * (1 + k eps)`: the factor is exact, the
            //   product rounds once: `u`.
            // Total `(2n + 8)u = (n + 4) eps` to first order; `n + 5` leaves
            // one eps for the second-order terms.
            //
            // Relative to `budget`, so a power-of-two rescaling cannot move a
            // decision; any other rescaling can move one only for a tail
            // whose exact weight lies within about this slack of the budget.
            // An absolute slack such as `1e-15` would swamp tiny spectra and
            // vanish for large ones. `f64::EPSILON` because every quantity here is
            // `f64` at every payload dtype (see the type-level docs).
            //
            // MatrixAlgebraKit `_truncerr_impl` uses no slack and a strict
            // `>=` break; TensorKit's `SectorVector` `TruncationByError`
            // discards while the running error is `<= budget`, which this
            // keeps, so a budget met exactly up to rounding discards that
            // state.
            let limit = budget * slack;
            let orders = ordered_positions(spectra, DISCARD);
            let mut kept = uniform_masks(spectra, true);
            let mut tails: BinaryHeap<TailCandidate> = orders
                .iter()
                .enumerate()
                .filter_map(|(sector, order)| {
                    order.first().map(|&position| TailCandidate {
                        value: spectra[sector].values[position],
                        sector,
                        cursor: 0,
                    })
                })
                .collect();
            let mut discarded = 0.0;
            while let Some(TailCandidate {
                value,
                sector,
                cursor,
            }) = tails.pop()
            {
                let weight = spectra[sector].weight;
                let next = discarded
                    + match units {
                        None => weight * value * value,
                        Some(units) => {
                            let value = units.scaled(value);
                            weight * value * value
                        }
                    };
                if next > limit {
                    break;
                }
                discarded = next;
                kept[sector][orders[sector][cursor]] = false;
                if let Some(&position) = orders[sector].get(cursor + 1) {
                    tails.push(TailCandidate {
                        value: spectra[sector].values[position],
                        sector,
                        cursor: cursor + 1,
                    });
                }
            }
            kept
        }
        // TensorKit `findtruncated(values, ::TruncationSpace)`
        // (truncation.jl:261-265): a plain `truncrank(dim(space, c))` inside
        // each block, i.e. the first `dim(space, c)` entries of the block's
        // keep order. Absent sector -> rank zero (TK's `dim(V, c)` is zero
        // there); clamped to what the spectrum actually offers, since asking
        // for more than exists is not an error there either. TensorKit returns
        // those positions in magnitude order and so also permutes the kept
        // columns; a mask cannot, and keeps them in stored order (see
        // `Truncation::Space`).
        Truncation::Space(profile) => spectra
            .iter()
            .map(|spectrum| {
                let mut mask = vec![false; spectrum.values.len()];
                let rank = profile.rank(spectrum.sector);
                for &position in sector_order(spectrum.values, KEEP).iter().take(rank) {
                    mask[position] = true;
                }
                mask
            })
            .collect(),
        // MatrixAlgebraKit `_ind_intersect`: per-sector set intersection.
        Truncation::All(components) => {
            let mut kept = uniform_masks(spectra, true);
            for component in components {
                for (mask, other) in kept.iter_mut().zip(kept_masks(spectra, component)) {
                    for (slot, keep) in mask.iter_mut().zip(other) {
                        *slot &= keep;
                    }
                }
            }
            kept
        }
    }
}

/// TensorKit `findtruncated(::SectorVector, ::TruncationByOrder)`
/// (truncation.jl:171-203): walk the flat order (`KEEP`: `rev = true`,
/// `DISCARD`: `rev = false`) and mark until the running `dim(c)` sum
/// overflows. Each sector's order is merged by a heap whose head ties go to
/// the lower sector, which is the same flat order.
fn rank_masks(spectra: &[WeightedSpectrum<'_>], rank: usize, direction: bool) -> Vec<Vec<bool>> {
    let orders = ordered_positions(spectra, direction);
    // The heap pops the largest key, so the discard order negates it.
    let key = |value: f64| if direction == KEEP { value } else { -value };
    let mut kept = uniform_masks(spectra, false);
    let mut heap: BinaryHeap<DescendingCandidate> = orders
        .iter()
        .enumerate()
        .filter_map(|(sector, order)| {
            order.first().map(|&position| DescendingCandidate {
                value: key(spectra[sector].values[position]),
                sector,
                cursor: 0,
            })
        })
        .collect();
    let mut used = 0.0;
    let budget = rank as f64;
    while let Some(DescendingCandidate { sector, cursor, .. }) = heap.pop() {
        let weight = spectra[sector].weight;
        // TensorKit `totaldim > howmany && break` compares the running
        // `dim(c)` sum with no slack. The weights are exact `dim(c)` (#1871),
        // so a tolerance here would only admit a state TensorKit rejects.
        if used + weight > budget {
            break;
        }
        used += weight;
        kept[sector][orders[sector][cursor]] = true;
        if let Some(&position) = orders[sector].get(cursor + 1) {
            heap.push(DescendingCandidate {
                value: key(spectra[sector].values[position]),
                sector,
                cursor: cursor + 1,
            });
        }
    }
    kept
}

fn uniform_masks(spectra: &[WeightedSpectrum<'_>], keep: bool) -> Vec<Vec<bool>> {
    spectra
        .iter()
        .map(|spectrum| vec![keep; spectrum.values.len()])
        .collect()
}

fn threshold_masks(spectra: &[WeightedSpectrum<'_>], threshold: f64) -> Vec<Vec<bool>> {
    spectra
        .iter()
        .map(|spectrum| {
            spectrum
                .values
                .iter()
                .map(|&value| value >= threshold)
                .collect()
        })
        .collect()
}

/// TensorKit's keep order (`by = abs, rev = true`).
const KEEP: bool = true;
/// TensorKit's discard order (`by = abs, rev = false`).
const DISCARD: bool = false;

/// The positions of one sector's `values` in TensorKit's flat stable
/// `sortperm` order restricted to that sector: by magnitude (descending for
/// [`KEEP`], ascending for [`DISCARD`]), equal magnitudes by ascending
/// position.
///
/// `O(n)` for every monotone sector, ties included, in either direction, and
/// `O(n log n)` otherwise:
/// - monotone in the requested direction (a descending SVD spectrum under
///   `KEEP`): the stable sort sees one run, `n - 1` comparisons;
/// - monotone the other way (ascending magnitudes under `KEEP`, a
///   descending spectrum under `DISCARD`): the positions reversed, with each group
///   of equal values reversed back so ties stay position-ascending. Why not
///   leave this to the sort: it reverses only *strictly* descending runs, so
///   a run with exact ties (degenerate eigenvalues, `[5, 5, 4, 4]`) would
///   cost `O(n log n)`.
///
/// `partial_cmp` / `==` rather than `total_cmp`, so `-0.0` ties `0.0` as
/// under `by = abs`; NaN never reaches here (`validate_spectra`).
fn sector_order(values: &[f64], keep: bool) -> Vec<usize> {
    let reversed = values.windows(2).all(|pair| {
        if keep {
            pair[0] <= pair[1]
        } else {
            pair[0] >= pair[1]
        }
    });
    if reversed {
        let mut order: Vec<usize> = (0..values.len()).rev().collect();
        let mut start = 0;
        while start < order.len() {
            let value = values[order[start]];
            let end = start
                + order[start..]
                    .iter()
                    .take_while(|&&position| values[position] == value)
                    .count();
            order[start..end].reverse();
            start = end;
        }
        return order;
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let ascending = values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal);
        if keep {
            ascending.reverse()
        } else {
            ascending
        }
    });
    order
}

fn ordered_positions(spectra: &[WeightedSpectrum<'_>], keep: bool) -> Vec<Vec<usize>> {
    spectra
        .iter()
        .map(|spectrum| sector_order(spectrum.values, keep))
        .collect()
}

/// The out-of-range `DiscardWeight` comparison in units of
/// `2^e = floor2(rtol) * floor2(norm) * floor2(ρ ν)`, where `floor2` is
/// [`power_of_two_floor`], `ρ = rtol / floor2(rtol)` and
/// `ν = norm / floor2(norm)`, both in `[1, 2)` and exact.
///
/// `rtol * norm = ρ ν 2^e / floor2(ρ ν)`, so the budget in these units is
/// `(ρ ν / floor2(ρ ν))^2` in `[1, 4)`, formed with the single rounding of
/// `ρ ν` that the in-range `rtol * norm` also has. The bound itself is never
/// formed, so it cannot round on the subnormal grid, and `e` is carried as an
/// integer because `2^e` can lie outside the `f64` range. Each value is
/// rescaled by `2^-e` in one direction only ([`scale_by_power_of_two`]), so
/// the result is exact whenever it is normal: the decision is the in-range
/// decision for `spectrum / 2^e`. A term that overflows exceeds the budget
/// and stops the scan; a rescaled value that is not normal is below
/// `2^-1022` of the bound, and its square below `2^-2044` of the budget.
///
/// Why one integer exponent rather than dividing by the three factors in
/// turn: whichever factor goes first, a mixed-direction chain (a large
/// `floor2(norm)` with a subnormal `floor2(rtol)`) can underflow or overflow
/// an intermediate quotient whose final value is in range.
#[derive(Clone, Copy)]
struct BudgetUnits {
    exponent: i32,
    budget: f64,
}

impl BudgetUnits {
    fn new(rtol: f64, norm: f64) -> Self {
        if rtol == 0.0 || norm == 0.0 {
            // A zero budget discards exactly the zero values: in units of
            // the smallest subnormal every positive value is at least `1`.
            return Self {
                exponent: power_of_two_exponent(power_of_two_floor(0.0)),
                budget: 0.0,
            };
        }
        let norm_scale = power_of_two_floor(norm);
        let rtol_scale = power_of_two_floor(rtol);
        let product = (rtol / rtol_scale) * (norm / norm_scale);
        let product_scale = power_of_two_floor(product);
        let bound = product / product_scale;
        Self {
            exponent: power_of_two_exponent(norm_scale)
                + power_of_two_exponent(rtol_scale)
                + power_of_two_exponent(product_scale),
            budget: bound * bound,
        }
    }

    fn scaled(self, value: f64) -> f64 {
        scale_by_power_of_two(value, -self.exponent)
    }
}

/// `e` for a positive power of two `2^e`, normal or subnormal.
fn power_of_two_exponent(power: f64) -> i32 {
    let bits = power.to_bits();
    match (bits >> 52) as i32 {
        0 => 63 - bits.leading_zeros() as i32 - 1074,
        biased => biased - 1023,
    }
}

/// `value * 2^exponent`, exact whenever the result is normal.
///
/// The factors all move the same way, so an intermediate overflows only if
/// the result does, and an intermediate becomes subnormal (the only place a
/// step rounds) only if the result is subnormal too.
fn scale_by_power_of_two(mut value: f64, mut exponent: i32) -> f64 {
    const MAX_EXPONENT: i32 = f64::MAX_EXP - 1;
    const MIN_EXPONENT: i32 = f64::MIN_EXP - 1;
    let power = |exponent: i32| f64::from_bits(((exponent + MAX_EXPONENT) as u64) << 52);
    while exponent > MAX_EXPONENT {
        value *= power(MAX_EXPONENT);
        exponent -= MAX_EXPONENT;
    }
    while exponent < MIN_EXPONENT {
        value *= power(MIN_EXPONENT);
        exponent -= MIN_EXPONENT;
    }
    value * power(exponent)
}

/// A sector's next discard candidate for `DiscardWeight`: `cursor` indexes
/// that sector's discard order. `Ord` is inverted so the max-heap pops the
/// smallest value, lowest sector first; the heap holds one candidate per
/// sector, so `cursor` never needs to break a tie. Not
/// `Reverse<DescendingCandidate>`: that would break cross-sector ties toward
/// the highest sector. `partial_cmp` rather than `total_cmp` keeps
/// `-0.0 == 0.0`; NaN never reaches here (`validate_spectra`).
#[derive(Clone, Copy, Debug)]
struct TailCandidate {
    value: f64,
    sector: usize,
    cursor: usize,
}

impl PartialEq for TailCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.sector == other.sector
    }
}

impl Eq for TailCandidate {}

impl PartialOrd for TailCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TailCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .partial_cmp(&self.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.sector.cmp(&self.sector))
    }
}

/// A sector's next candidate for `Rank` / `RankSmallest`: `cursor` indexes
/// that sector's keep (discard) order and `value` is the heap key, the
/// magnitude (its negation). The max-heap pops the largest key, lowest
/// sector first — TensorKit's `sortperm(parent(values); rev)` over its own
/// sector order once `select_truncation` has put the slice in that order.
#[derive(Clone, Copy, Debug)]
struct DescendingCandidate {
    value: f64,
    sector: usize,
    cursor: usize,
}

impl PartialEq for DescendingCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.sector == other.sector && self.cursor == other.cursor
    }
}

impl Eq for DescendingCandidate {}

impl PartialOrd for DescendingCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DescendingCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.sector.cmp(&self.sector))
            .then_with(|| other.cursor.cmp(&self.cursor))
    }
}

/// An unscaled power sum at or above this is accurate to rounding: any term
/// that underflowed is below `f64::MIN_POSITIVE`, under `EPSILON` relative to
/// the sum.
const UNSCALED_POWER_SUM_MIN: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// A finite-`p` norm from its unscaled weighted power sum `sum`, rescaled when
/// `sum` overflowed or underflowed.
///
/// Julia's `generic_norm2` / `generic_normp` (LinearAlgebra `generic.jl:468`,
/// `:498`) divide every entry by `maxabs = normInf(x)` when the unscaled sum
/// would leave the range, return `maxabs` itself when it is zero, infinite or
/// NaN, and never rescale for `p <= 1`. This runs that scaled branch only
/// after the unscaled sum proved out of range, so an in-range norm keeps one
/// pass and no division per entry, and an out-of-range one pays two more.
/// A single-pass running scale (LAPACK `dnrm2`) would charge every call a
/// division and a comparison per entry.
///
/// `scale` returns Julia's `maxabs`, or any positive scale within a factor of
/// two below it (the truncation norms pass its power-of-two floor, so the
/// scaled pass divides exactly).
///
/// Callers sum over all coupled sectors with one global scale. TensorKit's
/// non-UniqueFusion `_norm` instead adds `dim(c) * norm(b, p)^p` unscaled, so
/// it returns `Inf` or `0` once one weighted block power leaves the range even
/// though the norm itself is representable; symtn returns the representable
/// value.
#[doc(hidden)]
pub fn rescaled_power_norm<E>(
    sum: f64,
    p: f64,
    scale: impl FnOnce() -> f64,
    scaled_sum: impl FnOnce(f64) -> Result<f64, E>,
) -> Result<f64, E> {
    let root = |sum: f64| {
        if p == 2.0 {
            sum.sqrt()
        } else {
            sum.powf(p.recip())
        }
    };
    if p <= 1.0 || (sum.is_finite() && sum >= UNSCALED_POWER_SUM_MIN) {
        return Ok(root(sum));
    }
    let scale = scale();
    if scale == 0.0 || !scale.is_finite() {
        return Ok(scale);
    }
    Ok(scale * root(scaled_sum(scale)?))
}

/// The largest power of two at or below a finite `value > 0`; the smallest
/// subnormal, `2^-1074`, for `value == 0`.
///
/// Dividing by it is exact for every value that stays normal, so a quantity
/// computed on `spectrum / scale` is the in-range computation on an exactly
/// rescaled spectrum. Why the smallest subnormal for zero: a zero budget must
/// discard exactly the zero values, and `v / 2^-1074 >= 1` for every `v > 0`,
/// where `v * v` itself underflows to zero below about `1.5e-162`.
fn power_of_two_floor(value: f64) -> f64 {
    const EXPONENT: u64 = 0x7ff0_0000_0000_0000;
    let bits = value.to_bits();
    if bits & EXPONENT != 0 {
        f64::from_bits(bits & EXPONENT)
    } else if bits != 0 {
        f64::from_bits(1 << (63 - bits.leading_zeros()))
    } else {
        f64::from_bits(1)
    }
}

/// `sqrt(Σ weight · Σ value²)` over the values `(sector, position)` for
/// which `summed` holds, finite whenever the result is representable (#1440).
///
/// The in-range pass is the historical unscaled sum, in stored order within a
/// sector. Out of range, the scale is the power-of-two floor of the largest
/// value summed — a maximum over every summed value, since a sector's values
/// may come in any order — so the result is exactly
/// `scale * (in-range norm of spectrum / scale)`.
fn weighted_norm(spectra: &[WeightedSpectrum<'_>], summed: impl Fn(usize, usize) -> bool) -> f64 {
    let max = || {
        let mut max = 0.0_f64;
        for (sector, spectrum) in spectra.iter().enumerate() {
            for (position, &value) in spectrum.values.iter().enumerate() {
                if summed(sector, position) {
                    max = max.max(value);
                }
            }
        }
        max
    };
    rescaled_power_norm::<std::convert::Infallible>(
        weighted_square_sum(spectra, &summed, |value| value * value),
        2.0,
        || {
            let max = max();
            if max == 0.0 {
                0.0
            } else {
                power_of_two_floor(max)
            }
        },
        |scale| {
            Ok(weighted_square_sum(spectra, &summed, |value| {
                (value / scale) * (value / scale)
            }))
        },
    )
    .unwrap_or_else(|never| match never {})
}

fn weighted_square_sum(
    spectra: &[WeightedSpectrum<'_>],
    summed: &impl Fn(usize, usize) -> bool,
    square: impl Fn(f64) -> f64,
) -> f64 {
    spectra
        .iter()
        .enumerate()
        .map(|(sector, spectrum)| {
            spectrum.weight
                * spectrum
                    .values
                    .iter()
                    .enumerate()
                    .filter(|&(position, _)| summed(sector, position))
                    .map(|(_, &value)| square(value))
                    .sum::<f64>()
        })
        // Not `sum()`: its f64 identity is `-0.0`, so a decision that
        // discards nothing would report `sqrt(-0.0) = -0.0`. The norm is a
        // magnitude a caller prints and compares, and `-0` is not one.
        // Folding from `+0.0` normalizes it without changing any other total,
        // because `0.0 + x == x` for every `x` that is not `-0.0`.
        .fold(0.0_f64, |total, sector| total + sector)
}

fn full_norm(spectra: &[WeightedSpectrum<'_>]) -> f64 {
    weighted_norm(spectra, |_, _| true)
}

fn full_norm_inf(spectra: &[WeightedSpectrum<'_>]) -> f64 {
    spectra
        .iter()
        .flat_map(|spectrum| spectrum.values.iter().copied())
        .fold(0.0, f64::max)
}

fn discarded_norm(spectra: &[WeightedSpectrum<'_>], kept: &[Vec<bool>]) -> f64 {
    weighted_norm(spectra, |sector, position| !kept[sector][position])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule the test spectra's sector ids belong to. `select_truncation`
    /// takes one; any stable identity works for the magnitude-driven
    /// policies, which never look at a sector.
    struct TestRule;

    fn rule() -> RuleIdentity {
        RuleIdentity::of_type::<TestRule>()
    }

    fn other_rule() -> RuleIdentity {
        struct OtherRule;
        RuleIdentity::of_type::<OtherRule>()
    }

    /// Sector ids are the entry positions, so a profile keyed by position is
    /// the same thing a space-derived profile would be.
    fn spectra<'a>(entries: &'a [(f64, Vec<f64>)]) -> Vec<WeightedSpectrum<'a>> {
        entries
            .iter()
            .enumerate()
            .map(|(index, (weight, values))| WeightedSpectrum {
                sector: SectorId::new(index),
                weight: *weight,
                values,
            })
            .collect()
    }

    fn select(
        spectra: &[WeightedSpectrum<'_>],
        truncation: &Truncation,
    ) -> Result<TruncationDecision, TruncationError> {
        select_truncation(spectra, truncation, &rule(), |sector| {
            SectorOrderKey::position(sector.id() as u64)
        })
    }

    /// The kept count per sector, asserting that every mask is a prefix:
    /// what TensorKit keeps on a descending spectrum without within-sector
    /// ties.
    #[track_caller]
    fn counts(kept: &[Vec<bool>]) -> Vec<usize> {
        kept.iter()
            .map(|mask| {
                let count = mask.iter().take_while(|&&keep| keep).count();
                assert!(
                    mask[count..].iter().all(|&keep| !keep),
                    "{mask:?} is not a prefix"
                );
                count
            })
            .collect()
    }

    /// A per-sector mask keeping exactly `positions`.
    fn mask(len: usize, positions: &[usize]) -> Vec<bool> {
        (0..len)
            .map(|position| positions.contains(&position))
            .collect()
    }

    fn profile(pairs: [(usize, usize); 2]) -> TruncationSpace {
        TruncationSpace::new(
            rule(),
            pairs.map(|(sector, rank)| (SectorId::new(sector), rank)),
        )
    }

    #[test]
    fn a_decision_that_discards_nothing_reports_positive_zero() {
        // Rust's f64 `Sum` identity is `-0.0`, so an empty discarded tail would
        // report `sqrt(-0.0) = -0.0`. The magnitude a caller prints and compares
        // bitwise against another path's zero must be `+0.0`.
        let entries = [(1.0, vec![2.0, 1.0]), (3.0, vec![4.0, 0.5])];
        let spectra = spectra(&entries);
        for truncation in [Truncation::Full, Truncation::rank(usize::MAX)] {
            let decision = select(&spectra, &truncation).unwrap();
            assert_eq!(counts(&decision.kept), vec![2, 2]);
            assert_eq!(
                decision.error.to_bits(),
                0.0_f64.to_bits(),
                "{truncation:?} discarded nothing and must report +0.0, got {}",
                decision.error
            );
        }
    }

    #[test]
    fn rank_budget_compares_the_dimension_sum_without_slack() {
        // What: TensorKit `totaldim > howmany && break` (truncation.jl:198):
        // a state whose weight overflows the budget by any amount is not
        // kept, and one that meets it exactly is.
        let over = 1.0 + 1e-13;
        let entries = [(over, vec![2.0]), (1.0, vec![1.0])];
        let decision = select(&spectra(&entries), &Truncation::rank(1)).unwrap();
        assert_eq!(counts(&decision.kept), vec![0, 0]);
        let entries = [(1.0, vec![2.0]), (1.0, vec![1.0])];
        let decision = select(&spectra(&entries), &Truncation::rank(1)).unwrap();
        assert_eq!(counts(&decision.kept), vec![1, 0]);
    }

    #[test]
    fn rank_budget_is_quantum_dimension_weighted() {
        let entries = [(1.0, vec![5.0, 1.0]), (3.0, vec![4.0, 0.5])];
        let spectra = spectra(&entries);
        // Budget 4: keep 5.0 (weight 1) and 4.0 (weight 3) exactly.
        let decision = select(&spectra, &Truncation::rank(4)).unwrap();
        assert_eq!(counts(&decision.kept), vec![1, 1]);
        // Budget 5: the next candidate (1.0, weight 1) fits.
        let decision = select(&spectra, &Truncation::rank(5)).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 1]);
        // Budget 6: 0.5 has weight 3 and does not fit.
        let decision = select(&spectra, &Truncation::rank(6)).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 1]);
    }

    #[test]
    fn rank_ties_keep_parent_storage_order() {
        let entries = [(1.0, vec![2.0, 1.0]), (1.0, vec![2.0, 1.0])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::rank(1)).unwrap();
        assert_eq!(counts(&decision.kept), vec![1, 0]);

        let decision = select(&spectra, &Truncation::rank(3)).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 1]);
    }

    #[test]
    fn tolerance_thresholds_against_weighted_norm() {
        let entries = [(1.0, vec![4.0, 3.0, 0.1])];
        let spectra = spectra(&entries);
        let truncation = Truncation::absolute_cutoff(1.0).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
        assert_eq!(counts(&decision.kept), vec![2]);
        assert!((decision.error - 0.1).abs() < 1e-12);

        // norm = 5.001..., rtol 0.5 => threshold ~2.5: keeps 4 and 3.
        let truncation = Truncation::relative_cutoff(0.5).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
        assert_eq!(counts(&decision.kept), vec![2]);
    }

    #[test]
    fn tolerance_inf_thresholds_against_unweighted_max() {
        let entries = [(3.0, vec![4.0, 3.0, 0.1]), (1.0, vec![2.5])];
        let spectra = spectra(&entries);
        let truncation = Truncation::relative_inf_cutoff(0.7).unwrap();
        let decision = select(&spectra, &truncation).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 0]);
    }

    #[test]
    fn and_composition_takes_the_stricter_prefix() {
        let entries = [(1.0, vec![4.0, 3.0, 2.0, 1.0])];
        let spectra = spectra(&entries);
        let combined = Truncation::rank(3).and(Truncation::absolute_cutoff(2.5).unwrap());
        let decision = select(&spectra, &combined).unwrap();
        assert_eq!(counts(&decision.kept), vec![2]);

        let combined = Truncation::rank(1).and(Truncation::absolute_cutoff(0.5).unwrap());
        let decision = select(&spectra, &combined).unwrap();
        assert_eq!(counts(&decision.kept), vec![1]);
    }

    #[test]
    fn full_keeps_everything_with_zero_error() {
        let entries = [(1.0, vec![2.0, 1.0]), (2.0, vec![1.5])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::Full).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 1]);
        assert_eq!(decision.error, 0.0);
    }

    #[test]
    fn operation_error_preserves_truncation_reasons() {
        for (error, reason) in [
            (
                TruncationError::InvalidPolicy {
                    message: "tolerance must be finite",
                },
                "tolerance",
            ),
            (
                TruncationError::InvalidSpectrum {
                    message: "spectrum values must be finite and non-negative",
                },
                "spectrum values",
            ),
            (TruncationError::RuleMismatch, "different fusion rule"),
        ] {
            assert!(
                matches!(symtn_tensors::OperationError::from(error), symtn_tensors::OperationError::InvalidArgument { message } if message.contains(reason))
            );
        }
    }

    #[test]
    fn non_finite_spectrum_returns_typed_error_for_every_policy() {
        let entries = [(1.0, vec![3.0, f64::NAN, 1.0])];
        let spectra = spectra(&entries);
        let policies = [
            Truncation::rank(1),
            Truncation::absolute_cutoff(1.0).unwrap(),
            Truncation::relative_inf_cutoff(0.5).unwrap(),
            Truncation::relative_error(0.1).unwrap(),
            Truncation::rank(2).and(Truncation::absolute_cutoff(0.5).unwrap()),
        ];

        for policy in policies {
            assert!(matches!(
                select(&spectra, &policy),
                Err(TruncationError::InvalidSpectrum { .. })
            ));
        }
    }

    #[test]
    fn invalid_policy_returns_typed_error() {
        let policies = [
            Truncation::absolute_cutoff(f64::NAN),
            Truncation::relative_cutoff(f64::INFINITY),
            Truncation::relative_inf_cutoff(-1.0),
            Truncation::relative_error(f64::NAN),
            Truncation::below(-1.0, 0.0),
            Truncation::below(0.0, f64::NAN),
        ];

        for policy in policies {
            assert!(matches!(policy, Err(TruncationError::InvalidPolicy { .. })));
        }

        let entries = [(1.0, vec![3.0, 2.0, 1.0])];
        let spectra = spectra(&entries);
        let unchecked = Truncation::rank(3).and(Truncation::Tolerance {
            atol: 0.0,
            rtol: f64::NAN,
        });
        assert!(matches!(
            select(&spectra, &unchecked),
            Err(TruncationError::InvalidPolicy { .. })
        ));
    }

    #[test]
    fn space_profile_keeps_exactly_the_requested_prefix_counts() {
        // What: the counts come from the profile alone. The magnitudes are
        // arranged so that no magnitude-driven policy would produce `[1, 3]` —
        // sector 0 holds the three largest values — so a decision that leaked
        // into `Rank` / `Tolerance` behaviour cannot pass.
        let entries = [(1.0, vec![9.0, 8.0, 7.0]), (3.0, vec![2.0, 1.0, 0.5])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::space(profile([(0, 1), (1, 3)]))).unwrap();
        assert_eq!(counts(&decision.kept), vec![1, 3]);
        // The reported error is still the weighted 2-norm of the discarded tail.
        let expected = (8.0f64 * 8.0 + 7.0 * 7.0).sqrt();
        assert!((decision.error - expected).abs() < 1e-12);
    }

    #[test]
    fn space_profile_treats_absent_sectors_as_rank_zero_and_clamps_the_rest() {
        // What: TensorKit reads `dim(space, c)`, which is zero for a sector the
        // target space does not carry — so an absent key drops that sector
        // entirely. A key asking for more than the spectrum has is clamped
        // rather than rejected: the prefix simply cannot be longer.
        let entries = [(1.0, vec![5.0, 4.0]), (2.0, vec![3.0])];
        let spectra = spectra(&entries);
        let sparse = TruncationSpace::new(rule(), [(SectorId::new(0), 9)]);
        let decision = select(&spectra, &Truncation::space(sparse)).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 0]);
    }

    #[test]
    fn space_profile_composes_as_a_prefix_rule() {
        // What: `and` still takes the per-sector minimum, so a profile can be
        // intersected with a magnitude rule without leaving prefix-land.
        let entries = [(1.0, vec![4.0, 3.0, 0.1]), (1.0, vec![2.0, 1.0])];
        let spectra = spectra(&entries);
        let combined = Truncation::space(profile([(0, 3), (1, 1)]))
            .and(Truncation::absolute_cutoff(1.0).unwrap());
        let decision = select(&spectra, &combined).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 1]);
    }

    #[test]
    fn a_profile_from_another_rule_is_a_typed_error_before_any_selection() {
        // What: a foreign rule's sector ids name different sectors, so reading
        // them as this rule's would silently zero the spectrum out instead of
        // failing. Checked through `and` too, since a profile can be buried
        // inside a composite.
        let entries = [(1.0, vec![5.0, 4.0]), (2.0, vec![3.0])];
        let spectra = spectra(&entries);
        let foreign = TruncationSpace::new(other_rule(), [(SectorId::new(0), 1)]);

        assert_eq!(
            select(&spectra, &Truncation::space(foreign.clone())),
            Err(TruncationError::RuleMismatch)
        );
        assert_eq!(
            select(
                &spectra,
                &Truncation::rank(2).and(Truncation::space(foreign)),
            ),
            Err(TruncationError::RuleMismatch)
        );
    }

    #[test]
    fn a_shuffled_feed_decides_exactly_as_the_ascending_one() {
        // What: the tie among the three 1.0 tails (and the 2.0 heads) is
        // broken by ascending sector, so every permutation of the feed must
        // report the same per-sector counts and the same error bits. The
        // spectra are small integers, so every partial sum of squares is exact
        // and the error bits do not depend on the summation order.
        // Ascending oracle by hand: rank(4) keeps 3.0, 2.0, 2.0 and the first
        // 1.0 (sector 0); rank(5) adds sector 1's 1.0. Sector 2 holds the
        // largest value, so no permutation coincides with the ascending one
        // by accident of magnitude.
        let entries = [
            (1.0, vec![2.0, 1.0]),
            (1.0, vec![2.0, 1.0]),
            (1.0, vec![3.0, 1.0]),
        ];
        let ascending = spectra(&entries);
        let policies = [
            (Truncation::rank(4), vec![2, 1, 1]),
            (Truncation::rank(5), vec![2, 2, 1]),
            // norm^2 = 20, budget 0.09 * 20 = 1.8: one 1.0 tail (sector 0).
            (Truncation::relative_error(0.3).unwrap(), vec![1, 2, 2]),
            (Truncation::relative_inf_cutoff(0.5).unwrap(), vec![1, 1, 1]),
            (Truncation::Full, vec![2, 2, 2]),
        ];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for (policy, expected) in policies {
            let reference = select(&ascending, &policy).unwrap();
            assert_eq!(counts(&reference.kept), expected, "{policy:?}");
            for permutation in permutations {
                let feed: Vec<_> = permutation.iter().map(|&i| ascending[i]).collect();
                let decision = select(&feed, &policy).unwrap();
                let back: Vec<Vec<bool>> = permutation
                    .iter()
                    .map(|&i| reference.kept[i].clone())
                    .collect();
                assert_eq!(decision.kept, back, "{policy:?} {permutation:?}");
                assert_eq!(decision.error.to_bits(), reference.error.to_bits());
            }
        }

        let mut duplicated = ascending.clone();
        duplicated[0].sector = duplicated[2].sector;
        assert!(matches!(
            select(&duplicated, &Truncation::Full),
            Err(TruncationError::InvalidSpectrum { message }) if message.contains("one spectrum")
        ));
    }

    #[test]
    fn stored_order_spectra_select_the_tensorkit_kept_set() {
        // What: values arrive in the factorization's stored order and the
        // kept set is TensorKit's mask over those positions. Expected masks
        // from TensorKit `findtruncated(::SectorVector, ...)` at cfaa073e
        // (one Trivial-like block), by hand:
        // - eigh ascending `[-3, -1, 0.5, 2]` (magnitudes `[3, 1, 0.5, 2]`):
        //   `sortperm(rev)` = `[1, 4, 2, 3]`; `truncrank(2)` keeps {0, 3};
        //   `truncerror`: norm^2 = 14.25, rtol 0.3 -> budget 1.2825 discards
        //   0.5 (0.25) and 1 (1.25 total), then 2 overflows -> {0, 3};
        //   `trunctol(atol = 1.5)` is `findall` -> {0, 3}; `truncspace(1)`
        //   -> {0}.
        // - eig arbitrary `[1, 3, 0.5, 2, 2.5]`: `truncrank(3)` keeps
        //   3, 2.5, 2 -> {1, 3, 4} (two runs).
        let entries = [(1.0, vec![3.0, 1.0, 0.5, 2.0])];
        let eigh = spectra(&entries);
        let both = mask(4, &[0, 3]);
        for policy in [
            Truncation::rank(2),
            Truncation::relative_error(0.3).unwrap(),
            Truncation::absolute_cutoff(1.5).unwrap(),
            Truncation::rank(3).and(Truncation::absolute_cutoff(1.5).unwrap()),
        ] {
            let decision = select(&eigh, &policy).unwrap();
            assert_eq!(decision.kept, std::slice::from_ref(&both), "{policy:?}");
            assert!(
                (decision.error - 1.25f64.sqrt()).abs() < 1e-15,
                "{policy:?}"
            );
        }
        let one = TruncationSpace::new(rule(), [(SectorId::new(0), 1)]);
        let decision = select(&eigh, &Truncation::space(one)).unwrap();
        assert_eq!(decision.kept, [mask(4, &[0])]);

        let entries = [(1.0, vec![1.0, 3.0, 0.5, 2.0, 2.5])];
        let decision = select(&spectra(&entries), &Truncation::rank(3)).unwrap();
        assert_eq!(decision.kept, [mask(5, &[1, 3, 4])]);
        assert!((decision.error - 1.25f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn rank_ties_within_a_sector_keep_the_earlier_position() {
        // TensorKit `sortperm(rev = true)` is stable: of `[1, 2, 1, 2]` the
        // order is `[2, 4, 1, 3]`, so `truncrank(3)` keeps {0, 1, 3}.
        let entries = [(1.0, vec![1.0, 2.0, 1.0, 2.0])];
        let decision = select(&spectra(&entries), &Truncation::rank(3)).unwrap();
        assert_eq!(decision.kept, [mask(4, &[0, 1, 3])]);
    }

    #[test]
    fn signed_zeros_tie_in_both_orders() {
        // `by = abs` makes -0.0 and 0.0 equal, so position decides; NaN is
        // rejected before any order is built (`validate_spectra`).
        let values = [0.0, -0.0, 0.0, 1.0];
        assert_eq!(sector_order(&values, KEEP), [3, 0, 1, 2]);
        assert_eq!(sector_order(&values, DISCARD), [0, 1, 2, 3]);
        let values = [-0.0, 0.0];
        assert_eq!(sector_order(&values, KEEP), [0, 1]);
        assert_eq!(sector_order(&values, DISCARD), [0, 1]);
    }

    #[test]
    fn monotone_orders_equal_the_full_key_sort_with_ties() {
        // What: the `O(n)` reversal for a sector monotone the other way gives
        // exactly the full-key order `(value, position asc)`, on tie-heavy
        // spectra in both directions, plus unordered ones (the sort path).
        let full_key = |values: &[f64], keep: bool| {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by(|&a, &b| {
                let by_value = if keep {
                    values[b].partial_cmp(&values[a])
                } else {
                    values[a].partial_cmp(&values[b])
                };
                by_value.unwrap().then(a.cmp(&b))
            });
            order
        };
        let descending = [
            vec![],
            vec![1.0],
            vec![5.0, 5.0, 4.0, 4.0, 4.0, 1.0, 0.0, -0.0, 0.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 2.0, 2.0, 1.0, 1.0],
        ];
        let mut cases = 0;
        for values in descending {
            let mut ascending = values.clone();
            ascending.reverse();
            let mut shuffled = values.clone();
            shuffled.rotate_left(values.len() / 2);
            for values in [&values, &ascending, &shuffled] {
                for keep in [KEEP, DISCARD] {
                    assert_eq!(
                        sector_order(values, keep),
                        full_key(values, keep),
                        "{values:?} keep {keep}"
                    );
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 5 * 3 * 2);
    }

    #[test]
    fn an_ascending_out_of_range_spectrum_scales_by_its_maximum() {
        // What (with stored order): `1e200^2` overflows, so the norm is
        // rescaled. Scaled by the first value (`1.0`) the sum stays `Inf`, the
        // threshold is `Inf` and everything was discarded; scaled by the
        // maximum the norm is `1e200` and only the `1.0` goes.
        let entries = [(1.0, vec![1.0, 1e200])];
        let ascending = spectra(&entries);
        let norm = full_norm(&ascending);
        assert!((norm - 1e200).abs() <= 1e200 * f64::EPSILON, "{norm:e}");
        for policy in [
            Truncation::relative_cutoff(0.5).unwrap(),
            Truncation::relative_error(0.5).unwrap(),
        ] {
            let decision = select(&ascending, &policy).unwrap();
            assert_eq!(decision.kept, [mask(2, &[1])], "{policy:?}");
            assert_eq!(decision.error, 1.0, "{policy:?}");
        }
        // The discarded norm takes its scale from what it sums too: here the
        // `1e200` is discarded and the `1.0` kept.
        let error = discarded_norm(&ascending, &[mask(2, &[0])]);
        assert!((error - 1e200).abs() <= 1e200 * f64::EPSILON, "{error:e}");
    }

    /// `sqrt(Σ weight · Σ value²)` over the selected values, summed in units
    /// of their maximum: the oracle's own norm, not `weighted_norm`.
    fn oracle_norm(
        spectra: &[WeightedSpectrum<'_>],
        selected: impl Fn(usize, usize) -> bool,
    ) -> f64 {
        let picked = || {
            spectra
                .iter()
                .enumerate()
                .flat_map(move |(sector, spectrum)| {
                    spectrum
                        .values
                        .iter()
                        .enumerate()
                        .map(move |(position, &value)| (sector, position, value))
                })
        };
        let max = picked()
            .filter(|&(s, p, _)| selected(s, p))
            .map(|(_, _, value)| value)
            .fold(0.0, f64::max);
        if max == 0.0 {
            return 0.0;
        }
        let sum: f64 = picked()
            .filter(|&(s, p, _)| selected(s, p))
            .map(|(s, _, value)| spectra[s].weight * (value / max) * (value / max))
            .sum();
        max * sum.sqrt()
    }

    /// Independent oracle for the ordering policies, TensorKit's
    /// `findtruncated(::SectorVector, ...)` restated: flatten every value as
    /// `(value, sector, position)`, sort with the flat order spelled out as a
    /// full key (so it does not lean on sort stability), and walk it with
    /// the documented stop rule; `Space` sorts each sector the same way and
    /// takes its rank. Shares no code with the per-sector sorts or the heap
    /// merge. Sector ids are entry positions, and so is the sector order.
    fn flat_oracle(spectra: &[WeightedSpectrum<'_>], truncation: &Truncation) -> Vec<Vec<bool>> {
        let flat = |sector_filter: Option<usize>| -> Vec<(f64, usize, usize)> {
            spectra
                .iter()
                .enumerate()
                .filter(|&(sector, _)| sector_filter.is_none_or(|only| only == sector))
                .flat_map(|(sector, spectrum)| {
                    spectrum
                        .values
                        .iter()
                        .enumerate()
                        .map(move |(position, &value)| (value, sector, position))
                })
                .collect()
        };
        let keep_key = |a: &(f64, usize, usize), b: &(f64, usize, usize)| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        };
        match truncation {
            // MatrixAlgebraKit `truncrank(rank; rev = false)`: the flat
            // stable ascending order, then the same `dim(c)` walk.
            Truncation::RankSmallest(rank) => {
                let mut kept: Vec<Vec<bool>> = spectra
                    .iter()
                    .map(|spectrum| vec![false; spectrum.values.len()])
                    .collect();
                let mut order = flat(None);
                order.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap()
                        .then(a.1.cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                });
                let mut total = 0.0;
                for (_, sector, position) in order {
                    total += spectra[sector].weight;
                    if total > *rank as f64 {
                        break;
                    }
                    kept[sector][position] = true;
                }
                kept
            }
            Truncation::ToleranceBelow { atol, rtol } => {
                let threshold = atol.max(rtol * oracle_norm(spectra, |_, _| true));
                spectra
                    .iter()
                    .map(|spectrum| spectrum.values.iter().map(|&v| v <= threshold).collect())
                    .collect()
            }
            Truncation::All(components) => {
                let mut kept: Vec<Vec<bool>> = spectra
                    .iter()
                    .map(|spectrum| vec![true; spectrum.values.len()])
                    .collect();
                for component in components {
                    for (mask, other) in kept.iter_mut().zip(flat_oracle(spectra, component)) {
                        for (slot, keep) in mask.iter_mut().zip(other) {
                            *slot &= keep;
                        }
                    }
                }
                kept
            }
            Truncation::Rank(rank) => {
                let mut kept: Vec<Vec<bool>> = spectra
                    .iter()
                    .map(|spectrum| vec![false; spectrum.values.len()])
                    .collect();
                let mut order = flat(None);
                order.sort_by(keep_key);
                let mut total = 0.0;
                for (_, sector, position) in order {
                    total += spectra[sector].weight;
                    if total > *rank as f64 {
                        break;
                    }
                    kept[sector][position] = true;
                }
                kept
            }
            Truncation::DiscardWeight { rtol } => {
                let norm = oracle_norm(spectra, |_, _| true);
                let budget = (rtol * norm) * (rtol * norm);
                let count: usize = spectra.iter().map(|spectrum| spectrum.values.len()).sum();
                let limit = budget * (1.0 + (count + 5) as f64 * f64::EPSILON);
                let mut kept: Vec<Vec<bool>> = spectra
                    .iter()
                    .map(|spectrum| vec![true; spectrum.values.len()])
                    .collect();
                let mut order = flat(None);
                order.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap()
                        .then(a.1.cmp(&b.1))
                        .then(a.2.cmp(&b.2))
                });
                let mut total = 0.0;
                for (value, sector, position) in order {
                    total += spectra[sector].weight * value * value;
                    if total > limit {
                        break;
                    }
                    kept[sector][position] = false;
                }
                kept
            }
            Truncation::Space(profile) => spectra
                .iter()
                .enumerate()
                .map(|(sector, spectrum)| {
                    let mut order = flat(Some(sector));
                    order.sort_by(keep_key);
                    let rank = profile.rank(spectrum.sector);
                    let mut mask = vec![false; spectrum.values.len()];
                    for &(_, _, position) in order.iter().take(rank) {
                        mask[position] = true;
                    }
                    mask
                })
                .collect(),
            other => unreachable!("no flat oracle for {other:?}"),
        }
    }

    /// The pinned fixture (non-dyadic tenths, so partial sums round) plus an
    /// SU(2)-weighted one; `(1/90)^(1/2)` puts the budget on the two smallest
    /// tails up to rounding.
    fn scale_fixtures() -> Vec<Vec<(f64, Vec<f64>)>> {
        vec![
            vec![
                (1.0, vec![0.9, 0.3, 0.1]),
                (1.0, vec![0.7, 0.2, 0.1]),
                (1.0, vec![0.5, 0.3, 0.1]),
            ],
            vec![
                (1.0, vec![0.8, 0.4, 0.1]),
                (2.0, vec![0.6, 0.3]),
                (3.0, vec![0.2, 0.1]),
            ],
        ]
    }

    const SCALE_RTOLS: [f64; 5] = [0.05, 0.1, 0.2, 0.3, 0.105_409_255_338_945_98];

    fn scaled(entries: &[(f64, Vec<f64>)], scale: impl Fn(f64) -> f64) -> Vec<(f64, Vec<f64>)> {
        entries
            .iter()
            .map(|(weight, values)| (*weight, values.iter().map(|&v| scale(v)).collect()))
            .collect()
    }

    #[test]
    fn discard_weight_decision_is_invariant_under_uniform_rescaling() {
        // What: spectrum * s scales the budget by s^2 with it, so the kept
        // sets must not move. Powers of two scale exactly (f64 and f32
        // spectra alike), so the error must scale bit for bit; powers of ten
        // round every value and must still keep the same sets.
        assert_eq!(SCALE_RTOLS[4], (1.0f64 / 90.0).sqrt());
        let mut cases = 0;
        for entries in scale_fixtures() {
            let unit = spectra(&entries);
            for rtol in SCALE_RTOLS {
                let policy = Truncation::relative_error(rtol).unwrap();
                let reference = select(&unit, &policy).unwrap();
                for exponent in [-60, -30, -10, 0, 10, 30, 60] {
                    let s = 2f64.powi(exponent);
                    let f64_entries = scaled(&entries, |v| v * s);
                    let f32_entries = scaled(&entries, |v| f64::from(v as f32 * s as f32));
                    let f32_unit_entries = scaled(&entries, |v| f64::from(v as f32));
                    let f32_reference = select(&spectra(&f32_unit_entries), &policy).unwrap();
                    for (scaled_entries, expected) in
                        [(&f64_entries, &reference), (&f32_entries, &f32_reference)]
                    {
                        let decision = select(&spectra(scaled_entries), &policy).unwrap();
                        assert_eq!(decision.kept, expected.kept, "rtol {rtol} s 2^{exponent}");
                        assert_eq!(decision.error.to_bits(), (expected.error * s).to_bits());
                        cases += 1;
                    }
                }
                for exponent in -6..=6 {
                    let s = 10f64.powi(exponent);
                    let decision = select(&spectra(&scaled(&entries, |v| v * s)), &policy).unwrap();
                    assert_eq!(decision.kept, reference.kept, "rtol {rtol} s 1e{exponent}");
                    assert!((decision.error - reference.error * s).abs() <= 1e-14 * s);
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 2 * SCALE_RTOLS.len() * (7 * 2 + 13));
    }

    #[test]
    fn discard_weight_discards_a_state_that_meets_the_budget_exactly() {
        // What: TensorKit `SectorVector` `TruncationByError` semantics — the
        // running error may equal the budget. norm = 2, rtol 0.5 -> budget
        // exactly 1: one 1.0 goes, the second would make 2 > 1. Of the four
        // equal values the earliest goes (stable ascending `sortperm`).
        // (MatrixAlgebraKit `_truncerr_impl` breaks at `>= budget` and would
        // keep all four.)
        let entries = [(1.0, vec![1.0, 1.0, 1.0, 1.0])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::relative_error(0.5).unwrap()).unwrap();
        assert_eq!(decision.kept, [mask(4, &[1, 2, 3])]);
        assert_eq!(decision.error, 1.0);
    }

    #[test]
    fn ordering_policies_match_the_flatten_and_sort_oracle() {
        // What: on a fixed grid of spectra fed both in generation order
        // (arbitrary, as `eig` stores them) and sorted descending (as SVD
        // does), with repeated values inside and across sectors, every
        // ordering policy keeps exactly the oracle's mask.
        const WEIGHTS: [f64; 6] = [1.0, 2.0, 3.0, 0.5, 2.5, 4.0];
        const VALUES: [f64; 7] = [3.0, 2.0, 1.0, 1.0, 0.5, 0.25, 0.0];
        const RTOLS: [f64; 6] = [0.0, 0.05, 0.3, 0.7, 1.0, 1.5];
        const RANKS: [usize; 5] = [0, 1, 3, 6, 40];
        // Tiny deterministic LCG; the grid is fixed, not random.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = |bound: usize| -> usize {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize) % bound
        };
        let mut cases = 0;
        for sectors in [1usize, 2, 3, 5, 8] {
            for _ in 0..40 {
                let stored: Vec<(f64, Vec<f64>)> = (0..sectors)
                    .map(|_| {
                        let len = next(7);
                        let values: Vec<f64> = (0..len).map(|_| VALUES[next(7)]).collect();
                        (WEIGHTS[next(6)], values)
                    })
                    .collect();
                let profile = TruncationSpace::new(
                    rule(),
                    (0..sectors).map(|sector| (SectorId::new(sector), next(5))),
                );
                let mut descending = stored.clone();
                for (_, values) in &mut descending {
                    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
                }
                for entries in [&stored, &descending] {
                    let weighted = spectra(entries);
                    let policies = RTOLS
                        .iter()
                        .map(|&rtol| Truncation::relative_error(rtol).unwrap())
                        .chain(RANKS.iter().map(|&rank| Truncation::rank(rank)))
                        .chain([Truncation::space(profile.clone())])
                        // MatrixAlgebraKit `null_truncation_strategy` parts
                        // and their intersection.
                        .chain(
                            RTOLS
                                .iter()
                                .map(|&rtol| Truncation::below(0.25, rtol).unwrap()),
                        )
                        .chain(RANKS.iter().map(|&rank| Truncation::rank_smallest(rank)))
                        .chain(RANKS.iter().map(|&rank| {
                            Truncation::below(0.0, 0.3)
                                .unwrap()
                                .and(Truncation::rank_smallest(rank))
                        }));
                    for policy in policies {
                        let kept = flat_oracle(&weighted, &policy);
                        let decision = select(&weighted, &policy).unwrap();
                        assert_eq!(decision.kept, kept, "entries {entries:?} {policy:?}");
                        let error = oracle_norm(&weighted, |s, p| !kept[s][p]);
                        assert!((decision.error - error).abs() <= 1e-12);
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 5 * 40 * 2 * (2 * RTOLS.len() + 3 * RANKS.len() + 1));
    }

    #[test]
    fn tail_candidate_orders_signed_zeros_equal_then_lower_sector_first() {
        // `partial_cmp` treats -0.0 == 0.0, so the tie falls to the sector;
        // the heap is a max-heap, so Greater means "pops first".
        let lower = TailCandidate {
            value: -0.0,
            sector: 0,
            cursor: 0,
        };
        let higher = TailCandidate {
            value: 0.0,
            sector: 1,
            cursor: 0,
        };
        assert_eq!(lower.cmp(&higher), Ordering::Greater);
        let lower = TailCandidate {
            value: -0.0,
            sector: 1,
            cursor: 0,
        };
        let higher = TailCandidate {
            value: 0.0,
            sector: 0,
            cursor: 0,
        };
        assert_eq!(lower.cmp(&higher), Ordering::Less);
    }

    #[test]
    fn discard_weight_cross_sector_ties_go_to_the_lower_sector() {
        // norm^2 = 21; rtol 0.31 -> budget 2.018 admits exactly two 1.0^2.
        // Sectors 0 and 1 lose their tails; sector 2 keeps its 1.0. The
        // inverted tie rule (highest sector first) would give [2, 1, 0].
        let entries = [
            (1.0, vec![3.0, 1.0]),
            (1.0, vec![3.0, 1.0]),
            (1.0, vec![1.0]),
        ];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::relative_error(0.31).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![1, 1, 1]);
        assert!((decision.error - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn discard_weight_within_sector_ties_discard_the_earlier_position() {
        // Behaviour change: TensorKit's `TruncationByError` walks the
        // ascending stable `sortperm`, so of equal values the earlier
        // position goes first, even on a descending SVD spectrum. TensorKit
        // `findtruncated` at cfaa073e gives these masks.
        // norm^2 = 14; rtol 0.6 -> budget 5.04 admits two of the three 2*1^2.
        let entries = [(2.0, vec![2.0, 1.0, 1.0, 1.0])];
        let decision = select(
            &spectra(&entries),
            &Truncation::relative_error(0.6).unwrap(),
        )
        .unwrap();
        assert_eq!(decision.kept, [mask(4, &[0, 3])]);
        assert!((decision.error - 2.0).abs() < 1e-12);
        // `[3, 1, 1]`, norm^2 = 11; rtol 0.35 -> budget 1.3475 admits one 1.
        let entries = [(1.0, vec![3.0, 1.0, 1.0])];
        let decision = select(
            &spectra(&entries),
            &Truncation::relative_error(0.35).unwrap(),
        )
        .unwrap();
        assert_eq!(decision.kept, [mask(3, &[0, 2])]);
        assert_eq!(decision.error, 1.0);
    }

    #[test]
    fn discard_weight_stops_at_the_first_failure() {
        // Tails ascending: 1.0 (w=4, cost 4) then 1.5 (w=1, cost 2.25).
        // norm^2 = 131.25; rtol 0.16 -> budget 3.36. The first candidate
        // fails, so nothing is discarded even though 2.25 alone would fit.
        let entries = [(4.0, vec![5.0, 1.0]), (1.0, vec![5.0, 1.5])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::relative_error(0.16).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 2]);
        assert_eq!(decision.error, 0.0);
    }

    #[test]
    fn discard_weight_zero_values_cost_nothing() {
        let entries = [(1.0, vec![1.0, 0.0, -0.0]), (3.0, vec![0.0])];
        let mixed = spectra(&entries);
        let decision = select(&mixed, &Truncation::relative_error(0.0).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![1, 0]);
        assert_eq!(decision.error, 0.0);

        let entries = [(1.0, vec![0.0, 0.0]), (2.0, vec![-0.0])];
        let all_zero = spectra(&entries);
        let decision = select(&all_zero, &Truncation::relative_error(0.0).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![0, 0]);
        assert_eq!(decision.error, 0.0);
    }

    #[test]
    fn discard_weight_skips_empty_sectors() {
        // norm^2 = 8.75; rtol 0.4 -> budget 1.4 admits both 0.5 tails
        // (costs 0.5 and 0.25) but not 2.0 (cost 8).
        let entries = [
            (1.0, vec![]),
            (2.0, vec![2.0, 0.5]),
            (1.0, vec![]),
            (1.0, vec![0.5]),
        ];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::relative_error(0.4).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![0, 1, 0, 0]);
        assert!((decision.error - 0.75f64.sqrt()).abs() < 1e-12);

        let decision = select(&[], &Truncation::relative_error(0.5).unwrap()).unwrap();
        assert!(decision.kept.is_empty());
        assert_eq!(decision.error, 0.0);
    }

    #[test]
    fn discard_weight_no_all_and_partial_discard() {
        // norm^2 = 9 + 4 + 1 + 2 * 6.25 = 26.5.
        let entries = [(1.0, vec![3.0, 2.0, 1.0]), (2.0, vec![2.5])];
        let spectra = spectra(&entries);
        let decision = select(&spectra, &Truncation::relative_error(0.0).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![3, 1]);
        let decision = select(&spectra, &Truncation::relative_error(1.0).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![0, 0]);
        // budget 2.385: 1.0 fits, then 2.0 (cost 4) fails.
        let decision = select(&spectra, &Truncation::relative_error(0.3).unwrap()).unwrap();
        assert_eq!(counts(&decision.kept), vec![2, 1]);
        assert!((decision.error - 1.0).abs() < 1e-12);
    }

    #[test]
    fn discard_weight_composes_as_per_sector_minimum() {
        let entries = [(1.0, vec![4.0, 3.0, 2.0, 1.0]), (2.0, vec![3.5, 0.5])];
        let spectra = spectra(&entries);
        let rank = Truncation::rank(3);
        let error = Truncation::relative_error(0.3).unwrap();
        let by_rank = select(&spectra, &rank).unwrap().kept;
        let by_error = select(&spectra, &error).unwrap().kept;
        assert_ne!(by_rank, by_error, "fixture must make both components bind");
        let expected: Vec<Vec<bool>> = by_rank
            .iter()
            .zip(&by_error)
            .map(|(a, b)| a.iter().zip(b).map(|(a, b)| a & b).collect())
            .collect();

        for combined in [
            rank.clone().and(error.clone()),
            error.clone().and(rank.clone()),
        ] {
            let decision = select(&spectra, &combined).unwrap();
            assert_eq!(decision.kept, expected);
            assert_eq!(decision.error, discarded_norm(&spectra, &decision.kept));
        }
    }

    #[test]
    fn out_of_range_powers_of_two_scale_norms_and_decisions_exactly() {
        // What: at `2^±700` and beyond every square leaves the `f64`
        // range, yet the norm is representable. Scaling by a power of two is
        // exact, so the kept counts must equal the unit spectrum's and the
        // error must scale bit for bit, for every norm-driven policy.
        let mut cases = 0;
        for entries in scale_fixtures() {
            let unit = spectra(&entries);
            for rtol in SCALE_RTOLS {
                for policy in [
                    Truncation::relative_error(rtol).unwrap(),
                    Truncation::relative_cutoff(rtol).unwrap(),
                    Truncation::rank(3),
                ] {
                    let reference = select(&unit, &policy).unwrap();
                    for exponent in [-1000, -700, -540, 540, 700, 1000] {
                        let s = 2f64.powi(exponent);
                        let decision =
                            select(&spectra(&scaled(&entries, |v| v * s)), &policy).unwrap();
                        assert_eq!(decision.kept, reference.kept, "{policy:?} s 2^{exponent}");
                        assert_eq!(
                            decision.error.to_bits(),
                            (reference.error * s).to_bits(),
                            "{policy:?} s 2^{exponent}"
                        );
                        assert_eq!(
                            full_norm(&spectra(&scaled(&entries, |v| v * s))).to_bits(),
                            (full_norm(&unit) * s).to_bits()
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 2 * SCALE_RTOLS.len() * 3 * 6);
    }

    #[test]
    fn a_zero_budget_discards_exactly_the_zero_values_even_when_squares_underflow() {
        // What: `relative_error(0)` keeps every positive value (its type-level
        // contract). `1e-170^2` underflows to zero, so an unscaled budget of
        // zero would discard it; the error is zero, not `1e-170`.
        let entries = [(1.0, vec![1e-170, 0.0]), (2.0, vec![3e-200])];
        let decision = select(
            &spectra(&entries),
            &Truncation::relative_error(0.0).unwrap(),
        )
        .unwrap();
        assert_eq!(counts(&decision.kept), [1, 1]);
        assert_eq!(decision.error.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn a_budget_whose_square_underflows_is_compared_in_scaled_units() {
        // What: norm 1, rtol 1e-170 -> budget `1e-340`, below the `f64`
        // range, as is every tail square. Ascending, `8e-171 <= 1e-170` is
        // discarded; with `9e-171` the tail norm is `sqrt(64 + 81) 1e-171
        // ≈ 1.2e-170`, over the bound, so it is kept. Unscaled, the budget
        // and both squares are zero and both would be discarded.
        let entries = [(1.0, vec![1.0, 9e-171, 8e-171])];
        let decision = select(
            &spectra(&entries),
            &Truncation::relative_error(1e-170).unwrap(),
        )
        .unwrap();
        assert_eq!(counts(&decision.kept), [2]);
        // Relative, as `error / 1e-171`: the absolute tolerance would accept
        // zero here. One value reaches the error.
        crate::test_numerics::numerics::assert_close("error", decision.error / 1e-171, 8.0, 1);
    }

    #[test]
    fn a_budget_whose_slack_overflows_is_compared_in_scaled_units() {
        // What: `rtol * norm = 1.3407807929942596e154` puts the budget in
        // `(MAX / (1 + (n + 5) eps), MAX]`: finite, but `limit` overflows, and
        // `next > Inf` never stops the scan. Exact oracle (Python
        // `fractions.Fraction` over these f64 values): with
        // `B = rtol^2 * Σ w v^2`, the weighted `2^1000` tail is at most `B`
        // (`B / 2^1000 ≈ 1.68e7`, `≈ 8.4e6` with weight 2), and the whole
        // spectrum is not (`rtol < 1`). So one value is discarded and the error
        // is `sqrt(w) 2^500`.
        let rtol = 1.3407807929942596e154 * 2f64.powi(-530);
        for weight in [1.0, 2.0] {
            let entries = [(1.0, vec![2f64.powi(530)]), (weight, vec![2f64.powi(500)])];
            let unit = spectra(&entries);
            let bound = rtol * full_norm(&unit);
            let n = 2.0;
            assert!(bound * bound <= f64::MAX);
            assert!((bound * bound * (1.0 + (n + 5.0) * f64::EPSILON)).is_infinite());
            let decision = select(&unit, &Truncation::relative_error(rtol).unwrap()).unwrap();
            assert_eq!(counts(&decision.kept), [1, 0], "weight {weight}");
            // Exact: one dyadic term, `sqrt(w)` rounded once.
            assert_eq!(decision.error, weight.sqrt() * 2f64.powi(500));
        }
        // The same window with both values in one sector.
        let entries = [(1.0, vec![2f64.powi(530), 2f64.powi(500)])];
        let decision = select(
            &spectra(&entries),
            &Truncation::relative_error(rtol).unwrap(),
        )
        .unwrap();
        assert_eq!(counts(&decision.kept), [1]);
        assert_eq!(decision.error, 2f64.powi(500));
    }

    #[test]
    fn a_subnormal_bound_is_not_rounded_before_the_comparison() {
        // What: `rtol * norm` with `rtol = 3 * 2^-1074`, norm `≈ 1.9` is
        // `≈ 5.7 * 2^-1074`, which rounds to `6 * 2^-1074` on the subnormal
        // grid. Exact oracle (Python `fractions.Fraction`): the tail
        // `6 * 2^-1074` squared over `rtol^2 * norm^2` is `1 / 0.9025 > 1`,
        // so it must be kept and the error is zero.
        let tiny = f64::from_bits(1);
        let entries = [(1.0, vec![1.9, 6.0 * tiny])];
        let decision = select(
            &spectra(&entries),
            &Truncation::relative_error(3.0 * tiny).unwrap(),
        )
        .unwrap();
        assert_eq!(counts(&decision.kept), [2]);
        assert_eq!(decision.error.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn a_large_norm_with_a_subnormal_rtol_keeps_the_rescaling_exact() {
        // What: `floor2(norm) = 1024` and `floor2(rtol)` subnormal. Dividing
        // by the norm's scale first put the tail on the subnormal grid and
        // discarded it. Exact oracle (Python `fractions.Fraction` over these
        // f64 values): `v / (rtol * norm)` is `1.4900853664633535` for A and
        // `1.0189416273584906` for B, both above 1, so both keep everything
        // and report a zero error.
        let tiny = f64::from_bits(1);
        for (first, tail, rtol) in [
            (1.0001 * 1024.0, 1526.0 * tiny, tiny),
            ((5.3 / 3.0) * 1024.0, 5530.0 * tiny, 3.0 * tiny),
        ] {
            let entries = [(1.0, vec![first, tail])];
            let decision = select(
                &spectra(&entries),
                &Truncation::relative_error(rtol).unwrap(),
            )
            .unwrap();
            assert_eq!(counts(&decision.kept), [2], "tail {tail:e} rtol {rtol:e}");
            assert_eq!(decision.error.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn power_of_two_rescaling_is_exact_across_the_whole_exponent_range() {
        let tiny = f64::from_bits(1);
        for (power, exponent) in [
            (tiny, -1074),
            (f64::MIN_POSITIVE, -1022),
            (1.0, 0),
            (2f64.powi(1023), 1023),
        ] {
            assert_eq!(power_of_two_exponent(power), exponent);
        }
        assert_eq!(
            scale_by_power_of_two(1.5, -2148).to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(scale_by_power_of_two(tiny, 2097), 2f64.powi(1023));
        assert_eq!(scale_by_power_of_two(2f64.powi(1023), -2097), tiny);
        assert_eq!(
            scale_by_power_of_two(3.0 * tiny, 2000),
            3.0 * 2f64.powi(926)
        );
        assert!(scale_by_power_of_two(1.0, 2048).is_infinite());
    }
}
