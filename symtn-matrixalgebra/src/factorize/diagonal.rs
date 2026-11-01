//! Compact-diagonal input, factorized directly as MatrixAlgebraKit's
//! `DiagonalAlgorithm` does (`src/implementations/{svd,qr,lq,eig,eigh}.jl`,
//! v0.6.8, reached through TensorKit `src/factorizations/diagonal.jl`).
//!
//! Every spectrum takes the direct route. A nonfinite value is refused by the
//! shared finite-input stage, as symtn's dense routes refuse it (#1986). Why
//! no dense fallback: `DiagonalAlgorithm` has none.
//!
//! Every family admits its input through [`diagonal_bond`] and then reads one
//! value map per diagonal entry `a`:
//!
//! | Value map | Families | Where |
//! |---|---|---|
//! | magnitude `\|a\|` for `S`, phase for `Vh` | SVD | `compact_diagonal_svd_sector`, `svd_vals_diagonal` |
//! | none (square, so the shape-based null space is empty) | null | `null_diagonal` |
//! | phase and magnitude | QR/LQ, polar | [`diagonal_phase_magnitude_spectra`] |
//! | real part, after the Hermiticity check | eigh | `hermitian_diagonal_bond` |
//! | complex value, after the eigenvalue check | eig | `validate_diagonal_eigenvalues`, `eig_vals_diagonal` |
//!
//! The elementwise matrix functions of a compact diagonal (TensorKit
//! `inv`/`pinv`/`exp`/`\\` on `DiagonalTensorMap`, `src/tensors/diagonal.jl`)
//! admit it through [`admit_compact_diagonal`], the structural half of
//! [`diagonal_bond`]: they map nonfinite values through (#1986 covers
//! factorizations only), and the pseudo-inverse refuses them in its own
//! cutoff ([`pinv_diagonal_spectrum`]).

use std::ops::Deref;

use super::*;

/// The input bond of a compact diagonal, in a layout whose coupled-sector
/// regions are aligned one-tree diagonals, with the spectrum keyed by sector.
pub(super) struct DiagonalBond<'a, R, D> {
    space: BondSpace<'a, R>,
    pub(super) regions: Arc<[CoupledSectorRegion]>,
    pub(super) by_sector: FxHashMap<SectorId, &'a SectorSpectrum<D>>,
}

enum BondSpace<'a, R> {
    Input(&'a BoundDynamicFusionMapSpace<R>),
    Normalized(BoundDynamicFusionMapSpace<R>),
}

impl<R, D> DiagonalBond<'_, R, D> {
    /// The bond space the regions describe: the input itself, or its
    /// canonical layout when the input's layout was normalized.
    pub(super) fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        match &self.space {
            BondSpace::Input(space) => space,
            BondSpace::Normalized(space) => space,
        }
    }

    /// The spectrum entry of `region`, in region order.
    pub(super) fn entry(&self, region: &CoupledSectorRegion) -> &SectorSpectrum<D> {
        self.by_sector[&region.coupled()]
    }
}

impl<R, D> Deref for DiagonalBond<'_, R, D> {
    type Target = [CoupledSectorRegion];

    fn deref(&self) -> &[CoupledSectorRegion] {
        &self.regions
    }
}

pub(super) fn aligned_one_tree_regions(regions: &[CoupledSectorRegion]) -> bool {
    regions.iter().all(|region| {
        region.has_aligned_diagonal()
            && region.row_trees().len() == 1
            && region.col_trees().len() == 1
    })
}

/// Admits a compact diagonal `spectrum` on `space` for a factorization of
/// `family`: [`admit_diagonal_bond`], then the shared finite-input stage
/// ([`require_finite_factor_input`]).
pub(super) fn diagonal_bond<'a, A, R, D>(
    authority: &A,
    space: &'a BoundDynamicFusionMapSpace<R>,
    spectrum: &'a [SectorSpectrum<D>],
    family: FactorFamily,
) -> Result<DiagonalBond<'a, R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = admit_diagonal_bond(authority, space, spectrum)?;
    require_finite_factor_input(
        spectrum
            .iter()
            .flat_map(|entry| entry.values.iter().copied()),
        family,
    )?;
    Ok(bond)
}

/// Admits a compact diagonal `spectrum` on `space` for an elementwise
/// matrix function in fusion mode `M` (inv, pinv, exp, solve): the same
/// structural admission as [`admit_diagonal_bond`] (a one-leg `V <- V` whose
/// coupled sectors and degeneracies the spectrum covers), with no value
/// check. The result keeps `space`, as TensorKit's `inv`/`pinv`/`exp` keep
/// `d.domain`; compact storage does not depend on its layout.
///
/// Why not [`admit_diagonal_bond`] itself: an elementwise map reads no
/// coupled-sector region, so a sector-sorted spectrum is matched against the
/// bond's blocks directly, with no per-sector map. A bound one-leg space has
/// one block per coupled
/// sector, so a count match plus a match per block covers every entry
/// exactly once.
///
/// An unsorted spectrum takes the region admission: compact QR/LQ/polar
/// keep the input space and emit their phase and magnitude spectra in region
/// order ([`diagonal_phase_magnitude_spectra`]).
#[doc(hidden)]
pub fn admit_compact_diagonal<M, R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<(), M::Error>
where
    M: FactorMode<R>,
{
    let raw = space.space();
    require_one_leg_endomorphism(raw)?;
    if !spectrum
        .windows(2)
        .all(|pair| pair[0].sector < pair[1].sector)
    {
        return admit_diagonal_bond(&M::authority(space), space, spectrum).map(drop);
    }
    let structure = raw.structure();
    if structure.block_count() != spectrum.len() {
        return Err(spectrum_mismatch(
            "compact diagonal spectrum sector count differs from the bond",
        )
        .into());
    }
    for index in 0..structure.block_count() {
        let block = structure
            .block(index)
            .map_err(OperationError::from_core_preserving_context)?;
        let entry = block.key().as_fusion_tree_pair().and_then(|pair| {
            spectrum
                .binary_search_by_key(&pair.codomain_tree().coupled(), |entry| entry.sector)
                .ok()
        });
        let Some(entry) = entry.map(|index| &spectrum[index]) else {
            return Err(
                spectrum_mismatch("compact diagonal spectrum is missing a bond sector").into(),
            );
        };
        let length = entry.values.len();
        if block.shape() != [length, length] {
            return Err(spectrum_mismatch(
                "compact diagonal spectrum length differs from the sector degeneracy",
            )
            .into());
        }
    }
    Ok(())
}

fn require_one_leg_endomorphism(space: &DynamicFusionMapSpace) -> Result<(), OperationError> {
    let homspace = space.homspace();
    if space.nout() != 1
        || space.nin() != 1
        || homspace.codomain().legs() != homspace.domain().legs()
    {
        return Err(OperationError::SpaceMismatch {
            message: "compact diagonal input must be a one-leg endomorphism V <- V",
        });
    }
    Ok(())
}

fn spectrum_mismatch(message: &'static str) -> OperationError {
    OperationError::InvalidArgument { message }
}

/// The structural admission of a compact diagonal `spectrum` on `space`.
///
/// # Errors
///
/// - `space` is not a one-leg endomorphism `V <- V`, or `spectrum` does not
///   cover its coupled sectors with the bond degeneracies. Diagonal storage
///   exists only in that form, so either is misuse; MAK asserts
///   `m == n && isdiag(A)` and checks the output sizes at the same stage.
/// - The coupled-sector region error of an inconsistent structure.
///
/// A layout whose regions are not aligned one-tree diagonals (an expert
/// layout) is a representation TensorKit has no counterpart for; it is
/// re-derived in canonical layout through `authority`. The spectrum does not
/// depend on layout, so this is representation conversion, not a solver
/// fallback.
fn admit_diagonal_bond<'a, A, R, D>(
    authority: &A,
    space: &'a BoundDynamicFusionMapSpace<R>,
    spectrum: &'a [SectorSpectrum<D>],
) -> Result<DiagonalBond<'a, R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
{
    let raw = space.space();
    let homspace = raw.homspace();
    require_one_leg_endomorphism(raw)?;
    let (space, regions) = match checked_sector_regions(raw.structure(), 1)? {
        Some(regions) if aligned_one_tree_regions(&regions) => (BondSpace::Input(space), regions),
        _ => {
            let staged = authority.stage(homspace.clone())?;
            let canonical = authority.commit(staged, |_| Ok(()))?;
            let regions = checked_sector_regions(canonical.space().structure(), 1)?
                .filter(|regions| aligned_one_tree_regions(regions))
                .ok_or(OperationError::UnsupportedTensorContractScope {
                    message: "canonical bond layout has no aligned coupled-sector diagonal",
                })?;
            (BondSpace::Normalized(canonical), regions)
        }
    };
    let by_sector =
        aligned_diagonal_spectrum_by_sector(&regions, spectrum).map_err(spectrum_mismatch)?;
    Ok(DiagonalBond {
        space,
        regions,
        by_sector,
    })
}

/// Phase and magnitude of one finite diagonal value, with unit phase at
/// zero. Scaling before normalizing keeps complex subnormals on the unit
/// circle; the magnitude overflows to `Inf` past the dtype's range, as MAK's
/// `abs` does. Why not MAK's `s / abs(s)` for the phase: the two agree to
/// within rounding, and the scaled form is symtn's established result.
pub(super) fn diagonal_phase_magnitude<D: FactorScalar>(value: D) -> (D, D) {
    let value = value.widen_complex();
    let scale = value.re.abs().max(value.im.abs());
    let (phase, magnitude) = if scale == 0.0 {
        (Complex64::new(1.0, 0.0), 0.0)
    } else {
        let normalized = value / scale;
        let norm = normalized.norm();
        (normalized / norm, scale * norm)
    };
    (D::from_complex64(phase), D::from_real(magnitude))
}

/// Phase and magnitude spectra of a compact diagonal, `(sign_safe(a), abs(a))`
/// per entry, both on the input bond `V <- V`, its dual orientation included
/// (TensorKit keeps `W = V`, `diagonal.jl` `initialize_output(qr_full!, …)`).
/// They are MAK `_diagonal_qr!` with `positive = true` (`q`, `r`; full and
/// compact coincide, LQ exchanges the factors) and the polar of a diagonal
/// (`PolarViaSVD` over `svd_compact!(::DiagonalAlgorithm)`: `W`, `P`, with
/// `Wh = W`).
#[expect(clippy::type_complexity)]
pub(super) fn diagonal_phase_magnitude_spectra<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    family: FactorFamily,
) -> Result<(Vec<SectorSpectrum<D>>, Vec<SectorSpectrum<D>>), A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, family)?;
    let mut phase = Vec::with_capacity(bond.len());
    let mut magnitude = Vec::with_capacity(bond.len());
    for region in bond.iter() {
        let (phases, magnitudes) = bond
            .entry(region)
            .values
            .iter()
            .map(|&value| diagonal_phase_magnitude(value))
            .unzip();
        phase.push(SectorSpectrum {
            sector: region.coupled(),
            values: phases,
        });
        magnitude.push(SectorSpectrum {
            sector: region.coupled(),
            values: magnitudes,
        });
    }
    Ok((phase, magnitude))
}
