use symtn_core::Placement;

use crate::{host_scratch::HostScratchBuffer, ReportsPlacement};

use super::route_host::HostRouteScratch;

/// Host scratch workspace of the eager contraction routes: the owned source
/// copies and the core destination of a `DynamicTree` replay.
///
/// The route artifact is the single structure authority of each buffer, so
/// the buffers carry no space tag. Device lowering uses a separate device
/// scratch workspace.
#[derive(Clone, Debug)]
pub(crate) struct HostDynamicFusionScratchWorkspace<T> {
    lhs: HostScratchBuffer<T>,
    rhs: HostScratchBuffer<T>,
    dst: HostScratchBuffer<T>,
}

pub(crate) type DynamicFusionScratchWorkspace<T> = HostDynamicFusionScratchWorkspace<T>;

impl<T> Default for HostDynamicFusionScratchWorkspace<T> {
    fn default() -> Self {
        Self {
            lhs: HostScratchBuffer::default(),
            rhs: HostScratchBuffer::default(),
            dst: HostScratchBuffer::default(),
        }
    }
}

impl<T> HostDynamicFusionScratchWorkspace<T> {
    /// The eager route scratch: no core slot, so a core destination's
    /// inactive blocks are zero-filled on every call.
    pub(super) fn route_scratch<C>(&mut self) -> HostRouteScratch<'_, T, C> {
        HostRouteScratch {
            lhs: &mut self.lhs,
            rhs: &mut self.rhs,
            core_dst: &mut self.dst,
            core: None,
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        [&self.lhs, &self.rhs, &self.dst]
            .into_iter()
            .fold(0usize, |bytes, buffer| {
                bytes.saturating_add(buffer.capacity().saturating_mul(std::mem::size_of::<T>()))
            })
    }

    #[cfg(test)]
    pub(crate) fn dst_data_mut(&mut self) -> Option<&mut [T]> {
        (!self.dst.is_empty()).then(|| self.dst.as_mut_slice())
    }
}

impl<T> ReportsPlacement for HostDynamicFusionScratchWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

#[cfg(test)]
mod tests {
    use super::super::route_host::prepare;
    use super::*;

    #[test]
    fn route_scratch_reuses_a_long_enough_prefix_without_writing_it() {
        // What: a warm request up to the initialized length keeps the same
        // allocation and its contents (the next replay overwrites them); a
        // shorter request does not shrink the buffer.
        let mut buffer = HostScratchBuffer::default();
        prepare(&mut buffer, 3);
        assert_eq!(buffer.capacity(), 3);
        buffer.as_mut_slice().copy_from_slice(&[1.0, f64::NAN, 3.0]);
        let data = buffer.as_slice().as_ptr();

        for len in [3, 2] {
            prepare(&mut buffer, len);
            assert_eq!(buffer.as_slice().as_ptr(), data);
            assert_eq!(buffer.len(), 3);
            assert_eq!(buffer.as_slice()[0], 1.0);
            assert!(buffer.as_slice()[1].is_nan());
        }
    }

    #[test]
    fn route_scratch_growth_is_judged_by_initialized_length_and_is_fresh_zeros() {
        // What: a short initialized length with spare capacity (as
        // `restore_copy_c_scratch` may hand in) still grows to the request,
        // and growth yields exact-length zeros for real and complex scalars
        // instead of copying the old contents.
        let mut buffer = HostScratchBuffer::filled(8, 7.0_f64);
        buffer.resize_filled(2, 0.0);
        assert!(buffer.capacity() >= 4);

        prepare(&mut buffer, 4);

        assert_eq!(buffer.as_slice(), &[0.0; 4]);
        assert_eq!(buffer.capacity(), 4);

        let mut complex = HostScratchBuffer::filled(2, num_complex::Complex64::new(1.0, -1.0));
        prepare(&mut complex, 5);
        assert_eq!(
            complex.as_slice(),
            &[num_complex::Complex64::new(0.0, 0.0); 5]
        );
    }

    #[test]
    fn dynamic_fusion_scratch_retained_bytes_use_capacity() {
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        prepare(&mut workspace.lhs, 3);
        prepare(&mut workspace.rhs, 5);
        prepare(&mut workspace.dst, 7);

        assert_eq!(
            workspace.retained_bytes(),
            (3 + 5 + 7) * std::mem::size_of::<f64>()
        );
    }
}
