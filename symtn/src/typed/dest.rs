//! The destination argument of the `_into` operations.

use super::TensorScalar;

/// Where an `_into` operation writes, and how it combines with what is
/// there: tenferro's `Dest` grammar, with coefficients of the tensor's own
/// dtype `D` so a complex coefficient cannot reach a real tensor.
///
/// `Overwrite` and `Accumulate` are `Scaled` with `(alpha, beta)` equal to
/// `(1, 0)` and `(1, 1)`; each operation documents how it treats `beta = 0`.
///
/// ```
/// use std::sync::Arc;
///
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{Dest, GradedSpace, Runtime, TensorMap};
///
/// let engine = &mut symtn::typed::CpuEngine::serial();
/// let runtime = Runtime::new();
/// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0).unwrap(), 2)])?;
/// let x: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 1)?;
/// let mut y = x.zeros_like(engine)?;
/// x.axpby_into(engine, Dest::overwrite(&mut y))?;
/// x.axpby_into(engine, Dest::scaled(&mut y, 2.0, 0.5))?;
/// assert_eq!(y.dense_data()?, x.scale(engine, 2.5)?.dense_data()?);
/// # Ok::<(), symtn::typed::Error>(())
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum Dest<'a, T, D> {
    /// `out = op(...)`.
    Overwrite(&'a mut T),
    /// `out = op(...) + out`.
    Accumulate(&'a mut T),
    /// `out = alpha * op(...) + beta * out`.
    Scaled {
        /// The destination.
        out: &'a mut T,
        /// Scale applied to the operation result.
        alpha: D,
        /// Scale applied to the previous contents of `out`.
        beta: D,
    },
}

impl<'a, T, D> Dest<'a, T, D> {
    /// Build a [`Dest::Overwrite`] destination.
    pub fn overwrite(out: &'a mut T) -> Self {
        Self::Overwrite(out)
    }

    /// Build a [`Dest::Accumulate`] destination.
    pub fn accumulate(out: &'a mut T) -> Self {
        Self::Accumulate(out)
    }

    /// Build a [`Dest::Scaled`] destination, `out = alpha * op + beta * out`.
    pub fn scaled(out: &'a mut T, alpha: D, beta: D) -> Self {
        Self::Scaled { out, alpha, beta }
    }
}

impl<'a, T, D: TensorScalar> Dest<'a, T, D> {
    /// The destination and its `(alpha, beta)`.
    pub(crate) fn into_parts(self) -> (&'a mut T, D, D) {
        match self {
            Self::Overwrite(out) => (out, D::one(), D::zero()),
            Self::Accumulate(out) => (out, D::one(), D::one()),
            Self::Scaled { out, alpha, beta } => (out, alpha, beta),
        }
    }
}
