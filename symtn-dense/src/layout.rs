use crate::DenseError;

pub(crate) fn validate_dense_layout(
    len: usize,
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> Result<(), DenseError> {
    if shape.len() != strides.len() {
        return Err(DenseError::RankMismatch {
            shape: shape.len(),
            strides: strides.len(),
        });
    }
    if shape.contains(&0) {
        return if offset <= len {
            Ok(())
        } else {
            Err(DenseError::OutOfBounds)
        };
    }
    if offset >= len {
        return Err(DenseError::OutOfBounds);
    }
    let max_delta = max_offset_delta(shape, strides)?;
    let Some(last) = offset.checked_add(max_delta) else {
        return Err(DenseError::OffsetOverflow { value: offset });
    };
    if last < len {
        Ok(())
    } else {
        Err(DenseError::OutOfBounds)
    }
}

fn max_offset_delta(shape: &[usize], strides: &[usize]) -> Result<usize, DenseError> {
    shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |acc, (&dim, &stride)| {
            let steps = dim.saturating_sub(1);
            let Some(delta) = steps.checked_mul(stride) else {
                return Err(DenseError::StrideOverflow { value: stride });
            };
            match acc.checked_add(delta) {
                Some(total) => Ok(total),
                None => Err(DenseError::ElementCountOverflow),
            }
        })
}

#[cfg(feature = "tenferro")]
pub(crate) fn strides_to_isize(
    strides: &[usize],
) -> Result<tenferro_tensor::StrideVec, DenseError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect()
}

#[cfg(all(test, feature = "tenferro"))]
mod tests {
    use super::*;

    #[test]
    fn stride_conversion_uses_backend_carrier_for_arbitrary_rank() {
        let empty = strides_to_isize(&[]).unwrap();
        assert!(empty.is_empty());

        let short = strides_to_isize(&[1, 2, 4]).unwrap();
        assert_eq!(short.as_slice(), &[1, 2, 4]);

        let long = strides_to_isize(&[1, 2, 4, 8, 16, 32, 64, 128, 256]).unwrap();
        assert_eq!(long.as_slice(), &[1, 2, 4, 8, 16, 32, 64, 128, 256]);
    }

    #[test]
    fn stride_conversion_preserves_overflow_error() {
        assert_eq!(
            strides_to_isize(&[usize::MAX]).unwrap_err(),
            DenseError::StrideOverflow { value: usize::MAX }
        );
    }
}
