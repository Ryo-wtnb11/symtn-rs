use super::*;

/// Scalar contract for the factorization layer: dense-executor I/O plus the
/// adjoint and real-embedding used by the factor builders. Implemented for
/// the double-precision real and complex scalars.
pub trait FactorScalar: DenseRecouplingScalar {
    /// Output scalar of the general (non-Hermitian) eigendecomposition.
    type Eig: FactorScalar;
    /// Real scalar used by singular-value/eigenvalue outputs.
    type Real: DenseRecouplingScalar + Into<f64>;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError>;
    /// Consuming implementations may transfer backend-owned host storage.
    /// The default implementation copies from the borrowed slice.
    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        Self::dense_slice(&tensor).map(ToOwned::to_owned)
    }
    /// Converts a sector payload into the dense executor's owned full-SVD
    /// input. Custom scalar implementations retain the payload for the
    /// compatibility route.
    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Err(input)
    }
    /// Real spectrum output (singular values, Hermitian eigenvalues) widened
    /// to `f64` for the host-side truncation policies.
    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError>;
    /// [`Self::real_spectrum`] into `out` (cleared first), reusing its
    /// capacity. Custom scalar implementations keep the allocating route.
    fn real_spectrum_into(tensor: &DenseTensor, out: &mut Vec<f64>) -> Result<(), DenseError> {
        out.clear();
        out.extend(Self::real_spectrum(tensor)?);
        Ok(())
    }
    fn from_real(value: f64) -> Self;
    /// Widens to `Complex64` (general eigenvalue bookkeeping).
    fn widen_complex(self) -> Complex64;
    /// Narrows from `Complex64` (lossy for the single-precision scalars).
    fn from_complex64(value: Complex64) -> Self;
    fn adjoint(self) -> Self;
    /// LAPACK's `xLAMCH('P')` — `eps * base`, the spacing of one — in the
    /// *real component* type of `Self`.
    fn epsilon() -> f64;
    /// LAPACK's `xLAMCH('S')` — the safe minimum, the smallest positive normal
    /// number — in the *real component* type of `Self`.
    ///
    /// Paired with [`FactorScalar::epsilon`] this reproduces the machine bounds
    /// the reference `gebal` derives (`sgebal.f:330` for the single-precision
    /// blocks, `dgebal.f:330` for the double), which have to come from the
    /// component type: the double bounds let the radix loop of a single
    /// precision block reach a factor around `2^970`, and `from_real` of that
    /// is `inf` in `f32`.
    fn safe_minimum() -> f64;
}

impl FactorScalar for f32 {
    type Eig = num_complex::Complex32;
    type Real = f32;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_f32_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_f32_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::F32(input))
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor
            .as_f32_slice()?
            .iter()
            .map(|&value| value as f64)
            .collect())
    }

    fn real_spectrum_into(tensor: &DenseTensor, out: &mut Vec<f64>) -> Result<(), DenseError> {
        let values = tensor.as_f32_slice()?;
        out.clear();
        out.extend(values.iter().map(|&value| value as f64));
        Ok(())
    }

    fn from_real(value: f64) -> Self {
        value as f32
    }

    fn widen_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }

    fn from_complex64(value: Complex64) -> Self {
        value.re as f32
    }

    fn adjoint(self) -> Self {
        self
    }

    fn epsilon() -> f64 {
        f32::EPSILON as f64
    }

    fn safe_minimum() -> f64 {
        f32::MIN_POSITIVE as f64
    }
}

impl FactorScalar for f64 {
    type Eig = Complex64;
    type Real = f64;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_f64_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_f64_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::F64(input))
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor.as_f64_slice()?.to_vec())
    }

    fn real_spectrum_into(tensor: &DenseTensor, out: &mut Vec<f64>) -> Result<(), DenseError> {
        let values = tensor.as_f64_slice()?;
        out.clear();
        out.extend_from_slice(values);
        Ok(())
    }

    fn from_real(value: f64) -> Self {
        value
    }

    fn widen_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }

    fn from_complex64(value: Complex64) -> Self {
        value.re
    }

    fn adjoint(self) -> Self {
        self
    }

    fn epsilon() -> f64 {
        f64::EPSILON
    }

    fn safe_minimum() -> f64 {
        f64::MIN_POSITIVE
    }
}

impl FactorScalar for num_complex::Complex32 {
    type Eig = num_complex::Complex32;
    type Real = f32;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_c32_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_c32_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::C32(input))
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor
            .as_f32_slice()?
            .iter()
            .map(|&value| value as f64)
            .collect())
    }

    fn real_spectrum_into(tensor: &DenseTensor, out: &mut Vec<f64>) -> Result<(), DenseError> {
        let values = tensor.as_f32_slice()?;
        out.clear();
        out.extend(values.iter().map(|&value| value as f64));
        Ok(())
    }

    fn from_real(value: f64) -> Self {
        num_complex::Complex32::new(value as f32, 0.0)
    }

    fn widen_complex(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }

    fn from_complex64(value: Complex64) -> Self {
        num_complex::Complex32::new(value.re as f32, value.im as f32)
    }

    fn adjoint(self) -> Self {
        self.conj()
    }

    fn epsilon() -> f64 {
        f32::EPSILON as f64
    }

    fn safe_minimum() -> f64 {
        f32::MIN_POSITIVE as f64
    }
}

impl FactorScalar for Complex64 {
    type Eig = Complex64;
    type Real = f64;

    fn dense_slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
        tensor.as_c64_slice()
    }

    fn dense_into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
        tensor.into_c64_vec()
    }

    fn dense_into_owned(input: Vec<Self>) -> Result<DenseOwned, Vec<Self>> {
        Ok(DenseOwned::C64(input))
    }

    fn real_spectrum(tensor: &DenseTensor) -> Result<Vec<f64>, DenseError> {
        Ok(tensor.as_f64_slice()?.to_vec())
    }

    fn real_spectrum_into(tensor: &DenseTensor, out: &mut Vec<f64>) -> Result<(), DenseError> {
        let values = tensor.as_f64_slice()?;
        out.clear();
        out.extend_from_slice(values);
        Ok(())
    }

    fn from_real(value: f64) -> Self {
        Complex64::new(value, 0.0)
    }

    fn widen_complex(self) -> Complex64 {
        self
    }

    fn from_complex64(value: Complex64) -> Self {
        value
    }

    fn adjoint(self) -> Self {
        self.conj()
    }

    fn epsilon() -> f64 {
        f64::EPSILON
    }

    fn safe_minimum() -> f64 {
        f64::MIN_POSITIVE
    }
}

/// Magnitude used by the truncation selection over a spectrum.
pub trait SpectrumMagnitude: Copy {
    fn magnitude(self) -> f64;
}

impl SpectrumMagnitude for f64 {
    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl SpectrumMagnitude for Complex64 {
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

// The single-precision magnitudes widen *before* the absolute value or the
// hypotenuse, so a `Complex32` whose components straddle the `f32` range still
// reports a finite magnitude and the truncation policies compare the same
// `f64` quantities they compare for a double-precision payload.
impl SpectrumMagnitude for f32 {
    fn magnitude(self) -> f64 {
        f64::from(self).abs()
    }
}

impl SpectrumMagnitude for num_complex::Complex32 {
    fn magnitude(self) -> f64 {
        Complex64::new(f64::from(self.re), f64::from(self.im)).norm()
    }
}

/// One coupled sector's factorization spectrum: singular values (`f64`,
/// descending), Hermitian eigenvalues (signed `f64`, ascending), or general
/// eigenvalues (`Complex64`, ascending lexicographic `(re, im)`).
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSpectrum<V = f64> {
    pub sector: SectorId,
    pub values: Vec<V>,
}

/// The factorization a finite-input check guards; it names the operation in
/// the error.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorFamily {
    Svd,
    Qr,
    Lq,
    Polar,
    Null,
    Eig,
    Eigh,
}

/// The one finite-input stage of every factorization (#1986): a NaN or an
/// infinite component is refused with the same typed error in both modes and
/// on dense and diagonal storage, after the route's structural admission
/// (bond, endomorphism, stacking, polar shape) and before any numerical work.
///
/// Why stricter than TensorKit and MatrixAlgebraKit, which have no uniform
/// finite-input stage: there, dense eig/eigh refuse it through
/// backend-specific errors (LAPACK `chkfinite`, or the Hermiticity check),
/// dense SVD fails in LAPACK only for NaN, and QR/LQ/polar/null and the
/// diagonal routes propagate it. One typed error costs an `O(n)` scan,
/// negligible next to the `O(n³)` factorization.
pub(crate) fn require_finite_factor_input<D: FactorScalar>(
    values: impl IntoIterator<Item = D>,
    family: FactorFamily,
) -> Result<(), OperationError> {
    let finite = values.into_iter().all(|value| {
        let value = value.widen_complex();
        value.re.is_finite() && value.im.is_finite()
    });
    if finite {
        return Ok(());
    }
    Err(nonfinite_factor_input(family))
}

/// The one error of [`require_finite_factor_input`], for a storage that
/// decides finiteness itself (a CUDA payload scans on its device).
#[doc(hidden)]
pub fn nonfinite_factor_input(family: FactorFamily) -> OperationError {
    OperationError::InvalidArgument {
        message: match family {
            FactorFamily::Svd => "svd input components must be finite",
            FactorFamily::Qr => "qr input components must be finite",
            FactorFamily::Lq => "lq input components must be finite",
            FactorFamily::Polar => "polar input components must be finite",
            FactorFamily::Null => "null input components must be finite",
            FactorFamily::Eig => "eig input components must be finite",
            FactorFamily::Eigh => "eigh input components must be finite",
        },
    }
}
