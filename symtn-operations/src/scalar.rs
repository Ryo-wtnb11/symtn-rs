use core::ops::{Add, Mul};

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use symtn_dense::{DenseRead, DenseScalar, DenseView, DenseViewMut, DenseWrite};

pub trait ConjugateValue: Copy + strided_kernel::ElementOpApply {
    fn maybe_conj(self, conjugate: bool) -> Self;
}

macro_rules! impl_real_conjugate_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ConjugateValue for $ty {
                #[inline]
                fn maybe_conj(self, _conjugate: bool) -> Self {
                    self
                }
            }
        )+
    };
}

impl_real_conjugate_value!(f32, f64, i32, i64);

impl ConjugateValue for Complex32 {
    #[inline]
    fn maybe_conj(self, conjugate: bool) -> Self {
        if conjugate {
            self.conj()
        } else {
            self
        }
    }
}

impl ConjugateValue for Complex64 {
    #[inline]
    fn maybe_conj(self, conjugate: bool) -> Self {
        if conjugate {
            self.conj()
        } else {
            self
        }
    }
}

pub trait RealStructuralCoefficient: Copy {}

impl RealStructuralCoefficient for f32 {}
impl RealStructuralCoefficient for f64 {}

pub trait TreeTransformScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + Zero
    + One
    + ConjugateValue
    + Send
    + Sync
{
}

impl<T> TreeTransformScalar for T where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync
{
}

/// Action of a categorical recoupling coefficient on tensor storage data.
///
/// TensorKit allows, for example, real SU(2) coefficients to act on complex
/// tensor blocks. Rust needs that conversion boundary to be explicit.
pub trait RecouplingCoefficientAction<C>: Copy {
    fn scale_by_coefficient(self, coefficient: C) -> Self;
    fn coefficient_as_data(coefficient: C) -> Self;
}

macro_rules! impl_same_recoupling_coefficient_action {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl RecouplingCoefficientAction<$ty> for $ty {
                #[inline]
                fn scale_by_coefficient(self, coefficient: $ty) -> Self {
                    self * coefficient
                }

                #[inline]
                fn coefficient_as_data(coefficient: $ty) -> Self {
                    coefficient
                }
            }
        )+
    };
}

impl_same_recoupling_coefficient_action!(f32, f64, i32, i64, Complex32, Complex64);

impl RecouplingCoefficientAction<f64> for f32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient as f32
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        coefficient as f32
    }
}

impl RecouplingCoefficientAction<f32> for f64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * f64::from(coefficient)
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        f64::from(coefficient)
    }
}

impl RecouplingCoefficientAction<f32> for Complex32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * coefficient
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        Self::new(coefficient, 0.0)
    }
}

impl RecouplingCoefficientAction<f64> for Complex32 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient as f32
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        Self::new(coefficient as f32, 0.0)
    }
}

impl RecouplingCoefficientAction<f32> for Complex64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f32) -> Self {
        self * f64::from(coefficient)
    }

    #[inline]
    fn coefficient_as_data(coefficient: f32) -> Self {
        Self::new(f64::from(coefficient), 0.0)
    }
}

impl RecouplingCoefficientAction<f64> for Complex64 {
    #[inline]
    fn scale_by_coefficient(self, coefficient: f64) -> Self {
        self * coefficient
    }

    #[inline]
    fn coefficient_as_data(coefficient: f64) -> Self {
        Self::new(coefficient, 0.0)
    }
}

/// How one Host transform block is scaled: TensorKit's `α * coeff`.
///
/// Why not a plain `D`: collapsing a real structural coefficient into a
/// complex payload type turns a componentwise scale into a full complex
/// multiply, so `(1+0i) * (inf+0i)` becomes `inf + NaN i` and the sign of a
/// `-0` component is lost. TensorKit keeps the coefficient in
/// `sectorscalartype(I)` (a `Float64` for U(1), SU(2) and fZ2×U(1)) all the way
/// into `stridedtensoradd!`, where `ComplexF64 * Float64` is componentwise.
/// `Structural` carries the same distinction in the type, and is selected when
/// the caller's `α` is the multiplicative identity — TensorKit's `One()`.
#[derive(Clone, Copy, Debug)]
pub enum TransformScale<D, C> {
    /// `α == 1`: the structural coefficient acts on the payload directly.
    Structural(C),
    /// `α != 1`: the coefficient is folded into the payload type, as before.
    Data(D),
}

impl<D, C> TransformScale<D, C>
where
    D: RecouplingCoefficientAction<C> + One + PartialEq,
    C: Copy,
{
    #[inline]
    pub fn new(alpha: D, coefficient: C) -> Self {
        if alpha.is_one() {
            Self::Structural(coefficient)
        } else {
            Self::Data(alpha.scale_by_coefficient(coefficient))
        }
    }

    /// The coefficient promoted to the payload type, for adapters that cannot
    /// keep it real.
    #[inline]
    pub fn into_data(self) -> D {
        match self {
            Self::Structural(coefficient) => D::coefficient_as_data(coefficient),
            Self::Data(scale) => scale,
        }
    }

    /// Whether this scale is exactly one, so the transform must not multiply.
    #[inline]
    pub fn is_identity(self) -> bool {
        match self {
            Self::Structural(coefficient) => D::coefficient_as_data(coefficient).is_one(),
            Self::Data(scale) => scale.is_one(),
        }
    }

    /// Whether this scale is zero, so that it contributes VectorInterface's
    /// `zero(x) * α` whatever the source holds.
    #[inline]
    pub fn is_zero(self) -> bool
    where
        D: Zero,
    {
        match self {
            Self::Structural(coefficient) => D::coefficient_as_data(coefficient).is_zero(),
            Self::Data(scale) => scale.is_zero(),
        }
    }

    /// `scale * value`, with a `Structural` coefficient acting in its own type
    /// (componentwise for a real coefficient on a complex payload).
    #[inline]
    pub fn apply(self, value: D) -> D
    where
        D: Mul<D, Output = D>,
    {
        match self {
            Self::Structural(coefficient) => value.scale_by_coefficient(coefficient),
            Self::Data(scale) => scale * value,
        }
    }

    /// VectorInterface's `scale(x, α) = (iszero(α) ? zero(x) : x) * α` with
    /// this scale as `α`, and no multiply at exactly one (TensorKit's `One()`).
    /// Callers that pick one element op per block test
    /// [`is_identity`](Self::is_identity) and [`is_zero`](Self::is_zero)
    /// themselves.
    #[inline]
    pub fn scale(self, value: D) -> D
    where
        D: Mul<D, Output = D> + Zero,
    {
        if self.is_identity() {
            value
        } else if self.is_zero() {
            self.apply(D::zero())
        } else {
            self.apply(value)
        }
    }
}

/// VectorInterface's `scale(x, α) = (iszero(α) ? zero(x) : x) * α` for a
/// payload-typed `α`: a zero scale gives an exact zero rather than
/// `0 * inf = NaN`. An exact one still multiplies here; the callers that
/// stand for TensorKit's `One()` keep their own identity arm.
#[doc(hidden)]
#[inline]
pub fn scale_value<T>(value: T, alpha: T) -> T
where
    T: Copy + Mul<T, Output = T> + Zero,
{
    if alpha.is_zero() {
        T::zero() * alpha
    } else {
        alpha * value
    }
}

#[doc(hidden)]
pub trait DenseBlockScalar:
    Copy
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + PartialEq
    + Zero
    + One
    + ConjugateValue
    + Send
    + Sync
    + 'static
{
    fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    /// Dtype-erased carrier for accumulate-form GEMM parameters.
    fn dense_scalar(self) -> DenseScalar;
}

#[doc(hidden)]
pub trait DenseRecouplingScalar: DenseBlockScalar + RecouplingCoefficientAction<Self> {}

impl<T> DenseRecouplingScalar for T where T: DenseBlockScalar + RecouplingCoefficientAction<Self> {}

macro_rules! impl_dense_block_scalar {
    ($ty:ty, $read_variant:ident, $write_variant:ident) => {
        impl DenseBlockScalar for $ty {
            fn dense_read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$read_variant(view)
            }

            fn dense_write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$write_variant(view)
            }

            fn dense_scalar(self) -> DenseScalar {
                DenseScalar::$read_variant(self)
            }
        }
    };
}

impl_dense_block_scalar!(f32, F32, F32);
impl_dense_block_scalar!(f64, F64, F64);
impl_dense_block_scalar!(Complex32, C32, C32);
impl_dense_block_scalar!(Complex64, C64, C64);

/// Pairs a payload scalar with the double-precision member of its own field.
///
/// Reductions that sum a whole block, coupled region or spectrum accumulate in
/// [`WideScalar::Wide`] rather than in the payload type. For `f64` and
/// `Complex64` that is the payload type itself and [`WideScalar::widen`] is the
/// identity, so those reductions sum exactly the values they summed before,
/// with the same emitted arithmetic. For `f32` and `Complex32` it is double
/// precision: a naive single-precision sum of `n` products loses up to
/// `n * 6e-8` relative (about `sqrt(n) * 6e-8` in practice) and saturates to
/// infinity near `1.8e19`, while the result is widened to `Complex64`
/// immediately afterwards in every caller — so accumulating narrow buys
/// nothing and costs accuracy and range.
///
/// The `Wide` type carries `RecouplingCoefficientAction<f64>` so a
/// quantum-dimension weight can be applied to the accumulator without first
/// narrowing it to the payload type.
#[doc(hidden)]
pub trait WideScalar: DenseBlockScalar + RecouplingCoefficientAction<f64> {
    /// The double-precision scalar of the same field: real for a real payload,
    /// complex for a complex one.
    type Wide: DenseBlockScalar + RecouplingCoefficientAction<f64> + WideScalar<Wide = Self::Wide>;

    /// Exact widening. The identity when `Wide = Self`.
    fn widen(self) -> Self::Wide;

    /// Narrows a finished accumulator back to the payload type, rounding to
    /// nearest. The identity when `Wide = Self`.
    fn narrow(wide: Self::Wide) -> Self;
}

impl WideScalar for f64 {
    type Wide = Self;

    #[inline]
    fn widen(self) -> Self::Wide {
        self
    }

    #[inline]
    fn narrow(wide: Self::Wide) -> Self {
        wide
    }
}

impl WideScalar for Complex64 {
    type Wide = Self;

    #[inline]
    fn widen(self) -> Self::Wide {
        self
    }

    #[inline]
    fn narrow(wide: Self::Wide) -> Self {
        wide
    }
}

impl WideScalar for f32 {
    type Wide = f64;

    #[inline]
    fn widen(self) -> Self::Wide {
        f64::from(self)
    }

    #[inline]
    fn narrow(wide: Self::Wide) -> Self {
        wide as f32
    }
}

impl WideScalar for Complex32 {
    type Wide = Complex64;

    #[inline]
    fn widen(self) -> Self::Wide {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }

    #[inline]
    fn narrow(wide: Self::Wide) -> Self {
        Self::new(wide.re as f32, wide.im as f32)
    }
}
