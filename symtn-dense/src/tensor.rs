use num_complex::{Complex32, Complex64};

use crate::{DenseBackend, DenseDType, DenseError};

#[cfg(feature = "tenferro")]
use crate::tenferro_adapter::tenferro_error;

#[derive(Debug)]
pub struct DenseTensor {
    backend: DenseBackend,
    // Checked once in `from_tenferro`, so `dtype()` stays infallible although
    // Tenferro's `DType::External` has no `DenseDType`.
    dtype: DenseDType,
    inner: DenseTensorInner,
}

#[derive(Debug)]
enum DenseTensorInner {
    #[cfg(feature = "tenferro")]
    #[allow(dead_code)]
    Tenferro(tenferro_tensor::Tensor),
    #[cfg(not(feature = "tenferro"))]
    #[allow(dead_code)]
    Empty(std::convert::Infallible),
}

impl DenseTensor {
    #[inline]
    pub fn backend(&self) -> DenseBackend {
        self.backend
    }

    pub fn dtype(&self) -> DenseDType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor.shape(),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    /// symtn's own dtype check, ahead of the Tenferro accessor, so asking a
    /// tensor for the wrong scalar type is [`DenseError::DTypeMismatch`] and
    /// [`DenseError::Backend`] stays reserved for Tenferro's own failures.
    fn require_dtype(&self, expected: DenseDType, op: &'static str) -> Result<(), DenseError> {
        if self.dtype == expected {
            Ok(())
        } else {
            Err(DenseError::DTypeMismatch {
                op,
                expected,
                actual: self.dtype,
            })
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32], DenseError> {
        self.require_dtype(DenseDType::F32, "DenseTensor::as_f32_slice")?;
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<f32>()
                .map_err(|err| tenferro_error("DenseTensor::as_f32_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_f64_slice(&self) -> Result<&[f64], DenseError> {
        self.require_dtype(DenseDType::F64, "DenseTensor::as_f64_slice")?;
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<f64>()
                .map_err(|err| tenferro_error("DenseTensor::as_f64_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_c32_slice(&self) -> Result<&[Complex32], DenseError> {
        self.require_dtype(DenseDType::C32, "DenseTensor::as_c32_slice")?;
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<Complex32>()
                .map_err(|err| tenferro_error("DenseTensor::as_c32_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    pub fn as_c64_slice(&self) -> Result<&[Complex64], DenseError> {
        self.require_dtype(DenseDType::C64, "DenseTensor::as_c64_slice")?;
        match &self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => tensor
                .as_slice::<Complex64>()
                .map_err(|err| tenferro_error("DenseTensor::as_c64_slice", err)),
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match *inner {},
        }
    }

    /// Consume a host-backed tensor and transfer its `f64` buffer.
    pub fn into_f64_vec(self) -> Result<Vec<f64>, DenseError> {
        self.require_dtype(DenseDType::F64, "DenseTensor::into_f64_vec")?;
        match self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => {
                <f64 as tenferro_tensor::TensorScalar>::into_typed(tensor)
                    .map_err(|failure| failure.into_parts().1)
                    .and_then(|typed| {
                        tenferro_tensor::TypedTensor::<f64>::into_host_vec(typed)
                            .map_err(|failure| failure.into_parts().1)
                    })
                    .map_err(|err| tenferro_error("DenseTensor::into_f64_vec", err))
            }
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match inner {},
        }
    }

    /// Consume a host-backed tensor and transfer its `f32` buffer.
    pub fn into_f32_vec(self) -> Result<Vec<f32>, DenseError> {
        self.require_dtype(DenseDType::F32, "DenseTensor::into_f32_vec")?;
        match self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => {
                <f32 as tenferro_tensor::TensorScalar>::into_typed(tensor)
                    .map_err(|failure| failure.into_parts().1)
                    .and_then(|typed| {
                        tenferro_tensor::TypedTensor::<f32>::into_host_vec(typed)
                            .map_err(|failure| failure.into_parts().1)
                    })
                    .map_err(|err| tenferro_error("DenseTensor::into_f32_vec", err))
            }
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match inner {},
        }
    }

    /// Consume a host-backed tensor and transfer its `Complex32` buffer.
    pub fn into_c32_vec(self) -> Result<Vec<Complex32>, DenseError> {
        self.require_dtype(DenseDType::C32, "DenseTensor::into_c32_vec")?;
        match self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => {
                <Complex32 as tenferro_tensor::TensorScalar>::into_typed(tensor)
                    .map_err(|failure| failure.into_parts().1)
                    .and_then(|typed| {
                        tenferro_tensor::TypedTensor::<Complex32>::into_host_vec(typed)
                            .map_err(|failure| failure.into_parts().1)
                    })
                    .map_err(|err| tenferro_error("DenseTensor::into_c32_vec", err))
            }
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match inner {},
        }
    }

    /// Consume a host-backed tensor and transfer its `Complex64` buffer.
    pub fn into_c64_vec(self) -> Result<Vec<Complex64>, DenseError> {
        self.require_dtype(DenseDType::C64, "DenseTensor::into_c64_vec")?;
        match self.inner {
            #[cfg(feature = "tenferro")]
            DenseTensorInner::Tenferro(tensor) => {
                <Complex64 as tenferro_tensor::TensorScalar>::into_typed(tensor)
                    .map_err(|failure| failure.into_parts().1)
                    .and_then(|typed| {
                        tenferro_tensor::TypedTensor::<Complex64>::into_host_vec(typed)
                            .map_err(|failure| failure.into_parts().1)
                    })
                    .map_err(|err| tenferro_error("DenseTensor::into_c64_vec", err))
            }
            #[cfg(not(feature = "tenferro"))]
            DenseTensorInner::Empty(inner) => match inner {},
        }
    }

    #[cfg(feature = "tenferro")]
    #[cfg(not(feature = "provider-inject"))]
    pub(crate) fn from_tenferro(tensor: tenferro_tensor::Tensor) -> Result<Self, DenseError> {
        Ok(Self {
            backend: DenseBackend::Tenferro,
            dtype: dense_dtype_from_tenferro(tensor.dtype())?,
            inner: DenseTensorInner::Tenferro(tensor),
        })
    }
}

#[cfg(feature = "tenferro")]
pub(crate) fn dense_dtype_from_tenferro(
    dtype: tenferro_tensor::DType,
) -> Result<DenseDType, DenseError> {
    Ok(match dtype {
        tenferro_tensor::DType::F32 => DenseDType::F32,
        tenferro_tensor::DType::F64 => DenseDType::F64,
        tenferro_tensor::DType::I32 => DenseDType::I32,
        tenferro_tensor::DType::I64 => DenseDType::I64,
        tenferro_tensor::DType::Bool => DenseDType::Bool,
        tenferro_tensor::DType::C32 => DenseDType::C32,
        tenferro_tensor::DType::C64 => DenseDType::C64,
        tenferro_tensor::DType::External(_) => {
            return Err(DenseError::Unsupported {
                op: "tenferro_dtype",
                message: "externally defined Tenferro scalar types have no symtn dense dtype"
                    .to_string(),
            });
        }
    })
}

#[cfg(all(test, feature = "tenferro"))]
mod tests {
    use super::*;

    /// A real `DType::External` tensor needs `tenferro_tensor_core::ErasedHostTensor`,
    /// which `tenferro-tensor` does not re-export and symtn does not depend on, so
    /// the boundary is pinned on the mapping every constructor routes through.
    #[test]
    fn external_tenferro_dtype_is_a_typed_unsupported_error() {
        let err = dense_dtype_from_tenferro(tenferro_tensor::DType::External(
            std::any::TypeId::of::<u8>(),
        ))
        .unwrap_err();
        assert!(
            matches!(
                err,
                DenseError::Unsupported {
                    op: "tenferro_dtype",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn native_tenferro_dtypes_map_one_to_one() {
        use tenferro_tensor::DType;
        for (tenferro, dense) in [
            (DType::F32, DenseDType::F32),
            (DType::F64, DenseDType::F64),
            (DType::I32, DenseDType::I32),
            (DType::I64, DenseDType::I64),
            (DType::Bool, DenseDType::Bool),
            (DType::C32, DenseDType::C32),
            (DType::C64, DenseDType::C64),
        ] {
            assert_eq!(dense_dtype_from_tenferro(tenferro).unwrap(), dense);
        }
    }
}
