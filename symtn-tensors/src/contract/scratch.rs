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

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
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
    fn route_scratch_cold_is_exact_and_reuse_keeps_initialized_contents() {
        // What: cold scratch has exact capacity; same-length reuse neither
        // clears nor reallocates (the next replay overwrites it).
        let mut buffer = HostScratchBuffer::default();
        prepare(&mut buffer, 3);
        assert_eq!(buffer.capacity(), 3);
        buffer.as_mut_slice().copy_from_slice(&[1.0, f64::NAN, 3.0]);

        prepare(&mut buffer, 3);

        assert_eq!(buffer.capacity(), 3);
        assert_eq!(buffer.as_slice()[0], 1.0);
        assert!(buffer.as_slice()[1].is_nan());
    }

    #[test]
    fn route_scratch_growth_initializes_only_the_new_tail() {
        let mut buffer = HostScratchBuffer::default();
        prepare(&mut buffer, 3);
        buffer.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0]);

        prepare(&mut buffer, 5);

        assert_eq!(buffer.as_slice(), &[1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn dynamic_fusion_scratch_retained_bytes_use_capacity_and_clear_releases_it() {
        let mut workspace = HostDynamicFusionScratchWorkspace::<f64>::default();
        prepare(&mut workspace.lhs, 3);
        prepare(&mut workspace.rhs, 5);
        prepare(&mut workspace.dst, 7);

        assert_eq!(
            workspace.retained_bytes(),
            (3 + 5 + 7) * std::mem::size_of::<f64>()
        );
        workspace.clear();
        assert_eq!(workspace.retained_bytes(), 0);
    }
}
