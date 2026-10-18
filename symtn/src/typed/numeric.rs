use super::*;

impl ScalarOps for f64 {
    const CONJUGATION_IS_IDENTITY: bool = true;

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> Result<&mut Ctx<Self, Key>, Error> {
        Ok(&mut ctxs.f64)
    }

    fn rand_unit(state: &mut u64) -> Self {
        random_unit(state)
    }

    fn abs_value(self) -> f64 {
        self.abs()
    }

    fn exp_value(self) -> Self {
        self.exp()
    }

    fn recip_value(self) -> Self {
        1.0 / self
    }
}

/// Literal port of Julia 1.11.6 `Base.robust_cinv`
/// (`base/complex.jl:508`), the Baudin–Smith reciprocal kernel
/// [`julia_complex64_reciprocal`] falls back to once `c`/`d` are out of the
/// fast path's safe-square range. Returns `(p, q)` such that `1/(c+di) ≈
/// p + qi` once the caller's scale factor is reapplied.
#[inline]
pub(super) fn robust_cinv(c: f64, d: f64) -> (f64, f64) {
    let r = d / c;
    let z = d.mul_add(r, c);
    let p = 1.0 / z;
    let q = -r / z;
    (p, q)
}

/// Literal port of Julia 1.11.6 `Base.inv(w::ComplexF64)`
/// (`base/complex.jl:479`), symtn's TensorKit/Julia semantics reference for
/// `Complex64`'s compact `inv`/`pinv` reciprocal (#1463: the naive `1/z` this
/// replaces squares both components and so overflows above `|z| ~ 2^512` and
/// underflows to zero below `|z| ~ 2^-511`, although `1/z` is finite and
/// representable across that whole range). `Complex32`'s reciprocal follows
/// the companion `inv(z::Complex{T}) where T<:Union{Float16,Float32}`
/// (`base/complex.jl:473`) instead, in [`julia_complex32_reciprocal_wide`].
///
/// This is a direct translation, not an independently designed algorithm:
/// every branch, constant, and operation order (`muladd` → `mul_add`,
/// `copysign`/`flipsign` → `f64::copysign`) matches the Julia source so the
/// two stay bit-for-bit comparable — assuming `muladd` fuses on the host
/// running Julia, as it does on the aarch64 machine the literal bit
/// patterns in `symtn/tests/scaled_complex64_reciprocal.rs` were checked on;
/// Rust's `f64::mul_add` always fuses (IEEE 754 semantics, software fallback
/// where the hardware does not), so a Julia host where `muladd` does *not*
/// fuse would no longer match bitwise. See the `file:line` citations on each
/// helper and `symtn/tests/scaled_complex64_reciprocal.rs`, which checks this
/// port at each branch boundary against values checked with that same Julia
/// version, plus the scaling, conjugation, and `z * inv(z) = 1` identities.
/// TensorKit's semantics — matching Julia's own `ComplexF64` arithmetic, not
/// an independently chosen accuracy target — take precedence over a tighter
/// (and costlier) bound such as correct rounding: the fast path below is
/// exact whenever it applies, and the scaled/`robust_cinv` fallback is exact
/// in range without claiming last-bit accuracy outside it.
///
/// # Domain
///
/// `w` must be nonzero; the caller (`ScalarOps::recip_value`, reached only
/// through the compact `inv`/`pinv` arms, which already reject an exact-zero
/// entry) never calls this with `w == 0 + 0i`. Every other input — finite,
/// infinite, or `NaN` in either component — reaches this function and gets
/// exactly Julia's `inv(::ComplexF64)` result for it: an infinite component
/// takes the signed-zero branch below (not `NaN`, unlike the naive `1/z`
/// this replaces), and a `NaN` component propagates through whichever branch
/// its finite counterpart's magnitude selects, exactly as it does in Julia.
// Why not Tenferro: elementwise complex division/reciprocal is dense-backend
// arithmetic, not symtn's to own long-term. This port is interim until
// tensor4all/tenferro-rs#1922 lands a scale-robust primitive (symtn#1512
// tracks it); `symtn/tests/scaled_complex64_reciprocal.rs` stays the
// acceptance test when that happens.
#[inline]
pub(super) fn julia_complex64_reciprocal(w: Complex64) -> Complex64 {
    let (c, d) = (w.re, w.im);
    let (absc, absd) = (c.abs(), d.abs());
    let (cd, dc) = if absc > absd {
        (absc, absd)
    } else {
        (absd, absc)
    };
    // no overflow from abs2
    if (f64::MIN_POSITIVE / 2.0).sqrt() <= cd && cd <= (f64::MAX / 2.0).sqrt() {
        let denom = cd.mul_add(cd, dc * dc);
        // `conj(w) / denom`, i.e. `Complex(re/denom, im/denom)` per Julia's
        // `/(z::Complex, x::Real)` (`base/complex.jl:355`).
        return Complex64::new(c / denom, -d / denom);
    }
    if c.is_infinite() || d.is_infinite() {
        // `complex(copysign(0.0, c), flipsign(-0.0, d))`; `flipsign(-0.0, d)
        // == copysign(0.0, -d)` for every `d` (`flipsign` negates its first
        // argument exactly when its second argument's sign bit is set).
        return Complex64::new(0.0_f64.copysign(c), 0.0_f64.copysign(-d));
    }

    let eps = f64::EPSILON;
    let bs = 2.0 / (eps * eps);

    let mut c = c;
    let mut d = d;
    let mut s = 1.0;
    if cd >= f64::MAX / 2.0 {
        c *= 0.5;
        d *= 0.5;
        s = 0.5; // scale down c, d
    } else if cd <= 2.0 * f64::MIN_POSITIVE / eps {
        c *= bs;
        d *= bs;
        s = bs; // scale up c, d
    }

    let (p, q) = if absd <= absc {
        robust_cinv(c, d)
    } else {
        let (qq, pp) = robust_cinv(-d, -c);
        (pp, qq)
    };
    Complex64::new(p * s, q * s)
}

/// Literal port of Julia 1.11.6
/// `inv(z::Complex{T}) where T<:Union{Float16,Float32}` (`base/complex.jl:473`),
/// used by `Complex32::recip_value` after widening to `Complex64` (matching
/// Julia's `widen(z)`), narrowing at the end (matching Julia's `oftype`).
/// `c`/`d` here are that widened pair: a finite `f32` widens to a magnitude
/// far inside `f64`'s range, so the `c*c + d*d` this computes never
/// overflows or underflows the way the pre-#1463 naive `1/z` it replaces
/// did — like that code, it never needs the scaled fallback
/// [`julia_complex64_reciprocal`] uses; it shares the naming and citation
/// style for one reciprocal family, not because it shares failure modes. An
/// infinite or `NaN` `c`/`d` (from an infinite or `NaN` `f32` input, which
/// widening preserves) still reaches this function and is handled below,
/// exactly as Julia's own function handles it.
// Interim per the "Why not Tenferro" note above `julia_complex64_reciprocal`
// (symtn#1512).
#[inline]
pub(super) fn julia_complex32_reciprocal_wide(c: f64, d: f64) -> (f64, f64) {
    if c.is_infinite() || d.is_infinite() {
        return (0.0_f64.copysign(c), 0.0_f64.copysign(-d));
    }
    let mag = 1.0 / c.mul_add(c, d * d);
    (c * mag, -d * mag)
}

impl ScalarOps for num_complex::Complex64 {
    const CONJUGATION_IS_IDENTITY: bool = false;

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> Result<&mut Ctx<Self, Key>, Error> {
        Ok(&mut ctxs.c64)
    }

    fn rand_unit(state: &mut u64) -> Self {
        Self::new(random_unit(state), random_unit(state))
    }

    fn abs_value(self) -> f64 {
        self.norm()
    }

    fn exp_value(self) -> Self {
        self.exp()
    }

    fn recip_value(self) -> Self {
        // The compact `inv`/`pinv` callers only ever reach a nonzero `self`
        // (see `ScalarOps::recip_value`'s doc); the literal Julia port
        // handles a finite, an infinite, or a `NaN` component itself,
        // exactly as `Base.inv(::ComplexF64)` does.
        julia_complex64_reciprocal(self)
    }
}

impl ScalarOps for f32 {
    const CONJUGATION_IS_IDENTITY: bool = true;

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> Result<&mut Ctx<Self, Key>, Error> {
        ctxs.f32_lane()
    }

    fn rand_unit(state: &mut u64) -> Self {
        random_unit_f32(state)
    }

    fn abs_value(self) -> f64 {
        f64::from(self.abs())
    }

    fn exp_value(self) -> Self {
        self.exp()
    }

    fn recip_value(self) -> Self {
        1.0 / self
    }
}

impl ScalarOps for num_complex::Complex32 {
    const CONJUGATION_IS_IDENTITY: bool = false;

    fn ctx_of<Key: Clone + Eq + Hash + Send + Sync + 'static>(
        ctxs: &mut Ctxs<Key>,
    ) -> Result<&mut Ctx<Self, Key>, Error> {
        ctxs.c32_lane()
    }

    fn rand_unit(state: &mut u64) -> Self {
        Self::new(random_unit_f32(state), random_unit_f32(state))
    }

    fn abs_value(self) -> f64 {
        // Via `Complex64` rather than `Complex32::norm`: the f32 hypot can
        // overflow or flush to zero for components that the f64 return type
        // represents exactly.
        symtn_tensors::WideScalar::widen(self).norm()
    }

    fn exp_value(self) -> Self {
        self.exp()
    }

    fn recip_value(self) -> Self {
        // Literal port of Julia's `inv(z::Complex{Float32})`
        // (`base/complex.jl:473`, see `julia_complex32_reciprocal_wide`):
        // widen, invert, narrow. Every `f32` squared modulus is a normal
        // `f64`, so this widened computation never overflows/underflows the
        // way `Complex64::recip_value`'s own naive form did pre-#1463 — it
        // shares the naming and citation style for one reciprocal family,
        // not because it needs the fix.
        let wide = symtn_tensors::WideScalar::widen(self);
        let (re, im) = julia_complex32_reciprocal_wide(wide.re, wide.im);
        Self::new(re as f32, im as f32)
    }
}

pub(super) fn random_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64) / ((1_u64 << 52) as f64) - 1.0
}

/// The `f32` twin of [`random_unit`], drawn in the payload type.
///
/// Why not `random_unit(state) as f32`: the 53-bit draw in `[-1, 1)` rounds to
/// exactly `1.0` whenever it exceeds `1 - 2^-25`, so the half-open range the
/// name promises would not survive narrowing. Taking 24 bits — the `f32`
/// significand — and dividing by `2^23` keeps every draw exactly representable
/// and the upper bound open, mirroring the 53-bit/`2^52` double draw above.
pub(super) fn random_unit_f32(state: &mut u64) -> f32 {
    ((splitmix64(state) >> 40) as f32) / ((1_u32 << 23) as f32) - 1.0
}

pub(super) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Enumerates the coupled-sector matrix regions of a coupled-layout block
/// structure and verifies that every fusion-tree block addresses exactly the
/// packed column-major sector matrix. The structural constructors and the
/// device paths rely on these offsets, so any other layout is an explicit
/// error, never silent misaddressing.
pub(crate) fn sector_regions(
    structure: &BlockStructure,
    nout: usize,
) -> Result<Arc<[CoupledSectorRegion]>, Error> {
    structure
        .coupled_sector_regions(nout)?
        .ok_or_else(|| internal_layout_error("non-packed coupled-sector layout"))
}

/// Quantum-dimension-weighted region inner product `sum_c dim(c) * <a_c, b_c>`
/// over the packed coupled-sector regions, first argument conjugated.
///
/// `weight_of` supplies `dim(c)` and is fallible for the same reason as
/// [`weighted_trace`]'s: the checked Generic mode queries its provider in
/// place, once per region, instead of materializing a per-call weight map.
pub(crate) fn coupled_region_inner<D, W, E>(
    structure: &BlockStructure,
    nout: usize,
    a: &[D],
    b: &[D],
    mut weight_of: W,
) -> Result<Complex64, E>
where
    D: ScalarOps,
    W: FnMut(SectorId) -> Result<f64, E>,
    E: From<Error>,
{
    let regions = sector_regions(structure, nout)?;
    let required_len = structure.required_len().map_err(Error::from)?;
    if a.len() != required_len || b.len() != required_len {
        return Err(internal_layout_error(
            "coupled-sector regions do not cover the scalar buffers",
        )
        .into());
    }

    let mut total = Complex64::new(0.0, 0.0);
    for region in regions.iter() {
        let range = region.range();
        let lhs = a.get(range.clone()).ok_or_else(|| {
            internal_layout_error("coupled-sector region exceeds the left scalar buffer")
        })?;
        let rhs = b.get(range).ok_or_else(|| {
            internal_layout_error("coupled-sector region exceeds the right scalar buffer")
        })?;
        // Accumulated in `D::Wide` (see `symtn_tensors::WideScalar`): the identity for
        // the double-precision payloads, double precision for the single ones.
        let mut partial = D::Wide::from_real(0.0);
        for (&ai, &bi) in lhs.iter().zip(rhs) {
            partial = partial + FactorScalar::adjoint(ai.widen()) * bi.widen();
        }
        total += partial.widen_complex() * weight_of(region.coupled())?;
    }
    Ok(total)
}

/// `dim(c)` of a multiplicity-free rule through its coefficient algebra
/// (#1855), the one place the facade reads a multiplicity-free quantum
/// dimension.
pub(crate) fn multiplicity_free_dim<R>(rule: &R, sector: SectorId) -> Result<f64, Error>
where
    R: MultiplicityFreeRigidSymbols,
    symtn_core::MultiplicityFreeAdmissionMode: symtn_tensors::RigidCoefficientAlgebra<R>
        + symtn_tensors::CoefficientAlgebra<R, Error = symtn_tensors::OperationError>,
{
    <symtn_core::MultiplicityFreeAdmissionMode as symtn_tensors::RigidCoefficientAlgebra<R>>::dim(
        rule, sector,
    )
    .map_err(Error::from)
}

/// Quantum-dimension-weighted block trace of an endomorphism:
/// `sum_c dim(c) * tr(b_c)`, matching TensorKit's `tr` (`linalg.jl`, the
/// native `sum_c dim(c) * tr(block)`). Only fusion-tree blocks whose codomain
/// and domain trees coincide sit on the coupled-block diagonal; every other
/// block is off-diagonal in `b_c` and contributes nothing. Within a diagonal
/// block the trace pairs codomain degeneracy axis `i` with domain axis
/// `nout + i` (equal degeneracies, since the spaces match). Real tensors give
/// an exactly-real result. Fermionic twists belong to `trace_pairs` / tensor
/// contractions and are not part of this matrix trace.
///
/// `weight_of` supplies `dim(c)`; it is fallible so the checked Generic mode can
/// query its provider directly (TensorKit `linalg.jl::tr` is one definition
/// for every sector type, and so is this).
pub(crate) fn weighted_trace<D, W, E>(
    structure: &BlockStructure,
    nout: usize,
    data: &[D],
    mut weight_of: W,
) -> Result<Complex64, E>
where
    D: ScalarOps,
    W: FnMut(SectorId) -> Result<f64, E>,
    E: From<Error>,
{
    if let Ok(Some(regions)) = structure.coupled_sector_regions(nout) {
        if regions
            .iter()
            .all(CoupledSectorRegion::has_aligned_diagonal)
        {
            let mut total = Complex64::new(0.0, 0.0);
            for region in regions.iter() {
                let weight = weight_of(region.coupled())?;
                let step = region.rows() + 1;
                for (tree_index, (row, col)) in region
                    .row_trees()
                    .iter()
                    .zip(region.col_trees())
                    .enumerate()
                {
                    let mut partial = D::Wide::from_real(0.0);
                    let start = region.range().start + row.offset() + region.rows() * col.offset();
                    let row_end = region
                        .row_trees()
                        .get(tree_index + 1)
                        .map_or(region.rows(), CoupledTreeExtent::offset);
                    for diagonal in 0..row_end - row.offset() {
                        partial = partial + data[start + diagonal * step].widen();
                    }
                    total += partial.widen_complex() * weight;
                }
            }
            return Ok(total);
        }
    }
    let mut total = Complex64::new(0.0, 0.0);
    for index in 0..structure.block_count() {
        let block = structure.block(index).map_err(Error::from)?;
        let key = match block.key() {
            BlockKey::FusionTree(key) => key,
            _ => {
                return Err(
                    Error::InvalidArgument("tr() requires fusion-tree blocks".to_string()).into(),
                )
            }
        };
        // Off the coupled-block diagonal (codomain tree != domain tree): not on
        // the matrix diagonal of b_c, so it drops out of tr(b_c).
        if key.codomain_tree() != key.domain_tree() {
            continue;
        }
        let weight = weight_of(key.codomain_tree().coupled())?;
        let shape = block.shape();
        let strides = block.strides();
        let offset = block.offset();
        // Walk only the codomain degeneracy multi-index and index both halves
        // diagonally (axis i and axis nout+i share the index) — the degeneracy
        // trace of this coupled sub-block.
        let count: usize = shape[..nout].iter().product();
        let mut partial = D::Wide::from_real(0.0);
        for linear in 0..count {
            let mut remainder = linear;
            let mut position = offset;
            for axis in 0..nout {
                let coordinate = remainder % shape[axis];
                remainder /= shape[axis];
                position += coordinate * (strides[axis] + strides[nout + axis]);
            }
            partial = partial + data[position].widen();
        }
        total += partial.widen_complex() * weight;
    }
    Ok(total)
}

/// TensorKit `_norm`'s domain check on `p` (`linalg.jl:271-274`): only finite
/// positive `p` and `+Inf` name a norm; TensorKit throws `ArgumentError` for
/// anything else, which here is [`Error::InvalidArgument`].
///
/// `p <= 0.0` is false for NaN, so NaN needs its own test — without it a NaN
/// `p` would sail through and return NaN instead of being rejected.
pub(crate) fn validate_norm_p(p: f64) -> Result<(), Error> {
    if p.is_nan() || p <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "norm requires a finite p > 0 or p = inf, got {p}"
        )));
    }
    Ok(())
}

/// Surfaces a pinv seam's argument rejection (a non-finite singular value)
/// as [`Error::InvalidArgument`], the variant the facade already uses for a
/// bad `rcond` and the compact arm uses for the same non-finite condition, so
/// the storage form does not pick the public error.
pub(super) fn pinv_seam_error(error: symtn_tensors::OperationError) -> Error {
    match error {
        symtn_tensors::OperationError::InvalidArgument { message } => {
            Error::InvalidArgument(message.to_string())
        }
        other => other.into(),
    }
}

/// Julia's `max` for the `norm(t, Inf)` reduction: NaN in either argument wins.
///
/// `f64::max` implements IEEE `maxNum`, which *discards* a NaN operand, so a
/// payload poisoned by a single NaN would fold to a finite magnitude and hide
/// the overflow that produced it. TensorKit reduces with `mapreduce(max, …)`
/// (`src/tensors/linalg.jl:_norm`), and Julia's `max` propagates NaN.
pub(super) fn max_propagating_nan(accumulator: f64, magnitude: f64) -> f64 {
    if accumulator.is_nan() || magnitude.is_nan() {
        f64::NAN
    } else if magnitude > accumulator {
        magnitude
    } else {
        accumulator
    }
}

/// `Σ_c dim(c) · Σ_{x ∈ block_c} term(x)` over the coupled-sector regions of
/// one dense payload: the weighted sum under TensorKit `_norm(blocks(t), p, 0)`
/// (`linalg.jl:261-270`) for finite `p > 0`.
///
/// Why no `UniqueFusion` whole-buffer fast path like `dense_weighted_inner`'s (`reduction_ops.rs`): that
/// one exists to reach a single BLAS-shaped dot, and an entrywise `|x|^p`
/// reduction has no such kernel to reach. The region walk is a partition of the
/// same buffer, so the element count is identical either way.
pub(crate) fn coupled_region_weighted_sum<D, W, E>(
    structure: &BlockStructure,
    nout: usize,
    data: &[D],
    mut weight_of: W,
    term: impl Fn(D) -> f64,
) -> Result<f64, E>
where
    D: ScalarOps,
    W: FnMut(SectorId) -> Result<f64, E>,
    E: From<Error>,
{
    let regions = sector_regions(structure, nout)?;
    if data.len() != structure.required_len().map_err(Error::from)? {
        return Err(
            internal_layout_error("coupled-sector regions do not cover the scalar buffer").into(),
        );
    }

    let mut total = 0.0;
    for region in regions.iter() {
        let block = data.get(region.range()).ok_or_else(|| {
            internal_layout_error("coupled-sector region exceeds the scalar buffer")
        })?;
        let partial: f64 = block.iter().map(|&value| term(value)).sum();
        total += weight_of(region.coupled())? * partial;
    }
    Ok(total)
}

/// The largest entry magnitude, NaN-propagating like Julia's `normInf`.
pub(super) fn max_abs<D: ScalarOps>(values: impl IntoIterator<Item = D>) -> f64 {
    // Why `widen_complex().norm()` rather than an f64/c64 match: the widening
    // is exact and `Complex64::new(x, 0.0).norm()` is exactly `|x|`, so one
    // expression covers every payload dtype.
    values
        .into_iter()
        .map(|value| value.widen_complex().norm())
        .fold(0.0, max_propagating_nan)
}

/// `|x / max|^p` of one entry, in `f64`.
///
/// For `p == 2` the components are divided and squared separately, like
/// BLAS `dznrm2`: `hypot` first would add a square root per entry, and
/// `re² + im²` before dividing is the overflow being avoided.
pub(super) fn scaled_power<D: ScalarOps>(value: D, max: f64, p: f64) -> f64 {
    let value = value.widen_complex();
    if p == 2.0 {
        let (re, im) = (value.re / max, value.im / max);
        re * re + im * im
    } else {
        (value.norm() / max).powf(p)
    }
}

pub(crate) fn absorb_mapped<D, S>(
    destination_structure: &BlockStructure,
    destination: &mut [D],
    source_structure: &BlockStructure,
    source: &[S],
    mut map: impl FnMut(S) -> Result<D, Error>,
) -> Result<(), Error>
where
    S: Copy,
{
    let destination_sector = destination_structure.sector_structure();
    let source_sector = source_structure.sector_structure();
    let mut destination_position = 0usize;
    let mut source_position = 0usize;
    while destination_position < destination_sector.sorted_indices().len()
        && source_position < source_sector.sorted_indices().len()
    {
        let destination_index = destination_sector.sorted_indices()[destination_position];
        let source_index = source_sector.sorted_indices()[source_position];
        let destination_block = destination_structure.block(destination_index)?;
        let source_block = source_structure.block(source_index)?;
        match destination_block.key().cmp(source_block.key()) {
            std::cmp::Ordering::Less => destination_position += 1,
            std::cmp::Ordering::Greater => source_position += 1,
            std::cmp::Ordering::Equal => {
                if !matches!(destination_block.key(), BlockKey::FusionTree(_)) {
                    return Err(internal_layout_error(
                        "absorb reached a non-fusion-tree block key",
                    ));
                }
                // Why not construct BlockViews or a minimum-shape vector per
                // block: the validated layouts already provide every borrowed
                // stride, and absorb needs only this operation-local walk.
                copy_absorb_prefix(
                    destination,
                    destination_block.shape(),
                    destination_block.strides(),
                    destination_block.offset(),
                    source,
                    source_block.shape(),
                    source_block.strides(),
                    source_block.offset(),
                    &mut map,
                )?;
                destination_position += 1;
                source_position += 1;
            }
        }
    }
    Ok(())
}

pub(crate) fn absorb_compact_source<D: TensorScalar>(
    destination_structure: &BlockStructure,
    destination: &mut [D],
    source_structure: &BlockStructure,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
) -> Result<(), Error> {
    let by_sector: std::collections::HashMap<_, _> =
        spectrum.iter().map(|entry| (entry.sector, entry)).collect();
    if by_sector.len() != spectrum.len() {
        return Err(internal_layout_error(
            "compact absorb has duplicate spectrum sectors",
        ));
    }
    let destination_sector = destination_structure.sector_structure();
    let source_sector = source_structure.sector_structure();
    let mut destination_position = 0;
    let mut source_position = 0;
    while destination_position < destination_sector.sorted_indices().len()
        && source_position < source_sector.sorted_indices().len()
    {
        let destination_index = destination_sector.sorted_indices()[destination_position];
        let source_index = source_sector.sorted_indices()[source_position];
        let destination_block = destination_structure.block(destination_index)?;
        let source_block = source_structure.block(source_index)?;
        match destination_block.key().cmp(source_block.key()) {
            std::cmp::Ordering::Less => destination_position += 1,
            std::cmp::Ordering::Greater => source_position += 1,
            std::cmp::Ordering::Equal => {
                let pair = source_block.key().as_fusion_tree_pair().ok_or_else(|| {
                    internal_layout_error("absorb reached a non-fusion-tree block key")
                })?;
                let [source_rows, source_cols] = source_block.shape() else {
                    return Err(internal_layout_error(
                        "compact absorb source is not a matrix",
                    ));
                };
                let [destination_rows, destination_cols] = destination_block.shape() else {
                    return Err(internal_layout_error(
                        "compact absorb destination is not a matrix",
                    ));
                };
                let entry = by_sector
                    .get(&pair.codomain_tree().coupled())
                    .ok_or_else(|| {
                        internal_layout_error("compact absorb spectrum sector is absent")
                    })?;
                if source_rows != source_cols || entry.values.len() != *source_rows {
                    return Err(internal_layout_error(
                        "compact absorb spectrum shape disagrees",
                    ));
                }
                let prefix_rows = (*source_rows).min(*destination_rows);
                let prefix_cols = (*source_cols).min(*destination_cols);
                if destination_block.strides()[0] == 1
                    && destination_block.strides()[1] >= *destination_rows
                {
                    for column in 0..prefix_cols {
                        let start = column
                            .checked_mul(destination_block.strides()[1])
                            .and_then(|column_offset| {
                                destination_block.offset().checked_add(column_offset)
                            })
                            .ok_or_else(|| {
                                internal_layout_error("compact absorb destination offset overflow")
                            })?;
                        let end = start.checked_add(prefix_rows).ok_or_else(|| {
                            internal_layout_error("compact absorb destination offset overflow")
                        })?;
                        let cells = destination.get_mut(start..end).ok_or_else(|| {
                            internal_layout_error("compact absorb destination offset out of bounds")
                        })?;
                        cells.fill(D::from_real(0.0));
                        if column < prefix_rows {
                            cells[column] = entry.values[column];
                        }
                    }
                } else {
                    for column in 0..prefix_cols {
                        for row in 0..prefix_rows {
                            let offset = row
                                .checked_mul(destination_block.strides()[0])
                                .and_then(|row_offset| {
                                    destination_block.offset().checked_add(row_offset)
                                })
                                .and_then(|offset| {
                                    column
                                        .checked_mul(destination_block.strides()[1])
                                        .and_then(|column_offset| offset.checked_add(column_offset))
                                })
                                .ok_or_else(|| {
                                    internal_layout_error(
                                        "compact absorb destination offset overflow",
                                    )
                                })?;
                            *destination.get_mut(offset).ok_or_else(|| {
                                internal_layout_error(
                                    "compact absorb destination offset out of bounds",
                                )
                            })? = if row == column {
                                entry.values[row]
                            } else {
                                D::from_real(0.0)
                            };
                        }
                    }
                }
                destination_position += 1;
                source_position += 1;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_absorb_prefix<D, S>(
    destination: &mut [D],
    destination_shape: &[usize],
    destination_strides: &[usize],
    destination_offset: usize,
    source: &[S],
    source_shape: &[usize],
    source_strides: &[usize],
    source_offset: usize,
    map: &mut impl FnMut(S) -> Result<D, Error>,
) -> Result<(), Error>
where
    S: Copy,
{
    if destination_shape.len() != source_shape.len()
        || destination_shape.len() != destination_strides.len()
        || source_shape.len() != source_strides.len()
    {
        return Err(internal_layout_error("absorb block ranks differ"));
    }

    #[allow(clippy::too_many_arguments)]
    fn walk<D, S>(
        remaining_axes: usize,
        destination: &mut [D],
        destination_shape: &[usize],
        destination_strides: &[usize],
        destination_offset: usize,
        source: &[S],
        source_shape: &[usize],
        source_strides: &[usize],
        source_offset: usize,
        map: &mut impl FnMut(S) -> Result<D, Error>,
    ) -> Result<(), Error>
    where
        S: Copy,
    {
        if remaining_axes == 0 {
            let value = source
                .get(source_offset)
                .copied()
                .ok_or_else(|| internal_layout_error("absorb source offset out of bounds"))?;
            let value = map(value)?;
            *destination.get_mut(destination_offset).ok_or_else(|| {
                internal_layout_error("absorb destination offset out of bounds")
            })? = value;
            return Ok(());
        }

        let axis = remaining_axes - 1;
        let extent = destination_shape[axis].min(source_shape[axis]);
        if remaining_axes == 1 {
            if destination_strides[axis] == 1 && source_strides[axis] == 1 {
                let destination_end = destination_offset
                    .checked_add(extent)
                    .ok_or_else(|| internal_layout_error("absorb destination offset overflow"))?;
                let source_end = source_offset
                    .checked_add(extent)
                    .ok_or_else(|| internal_layout_error("absorb source offset overflow"))?;
                let destination = destination
                    .get_mut(destination_offset..destination_end)
                    .ok_or_else(|| {
                        internal_layout_error("absorb destination offset out of bounds")
                    })?;
                let source = source
                    .get(source_offset..source_end)
                    .ok_or_else(|| internal_layout_error("absorb source offset out of bounds"))?;
                for (destination, &source) in destination.iter_mut().zip(source) {
                    *destination = map(source)?;
                }
                return Ok(());
            }
            for coordinate in 0..extent {
                let destination_index = coordinate
                    .checked_mul(destination_strides[axis])
                    .and_then(|offset| destination_offset.checked_add(offset))
                    .ok_or_else(|| internal_layout_error("absorb destination offset overflow"))?;
                let source_index = coordinate
                    .checked_mul(source_strides[axis])
                    .and_then(|offset| source_offset.checked_add(offset))
                    .ok_or_else(|| internal_layout_error("absorb source offset overflow"))?;
                let value = source
                    .get(source_index)
                    .copied()
                    .ok_or_else(|| internal_layout_error("absorb source offset out of bounds"))?;
                *destination.get_mut(destination_index).ok_or_else(|| {
                    internal_layout_error("absorb destination offset out of bounds")
                })? = map(value)?;
            }
            return Ok(());
        }
        for coordinate in 0..extent {
            let destination_offset = coordinate
                .checked_mul(destination_strides[axis])
                .and_then(|offset| destination_offset.checked_add(offset))
                .ok_or_else(|| internal_layout_error("absorb destination offset overflow"))?;
            let source_offset = coordinate
                .checked_mul(source_strides[axis])
                .and_then(|offset| source_offset.checked_add(offset))
                .ok_or_else(|| internal_layout_error("absorb source offset overflow"))?;
            walk(
                axis,
                destination,
                destination_shape,
                destination_strides,
                destination_offset,
                source,
                source_shape,
                source_strides,
                source_offset,
                map,
            )?;
        }
        Ok(())
    }

    walk(
        destination_shape.len(),
        destination,
        destination_shape,
        destination_strides,
        destination_offset,
        source,
        source_shape,
        source_strides,
        source_offset,
        map,
    )
}
