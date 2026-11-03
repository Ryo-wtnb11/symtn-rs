use symtn_core::SimilarStorage;

/// Crate-internal same-placement scratch allocator.
///
/// This is the operations-layer adapter around `SimilarStorage`; it allocates
/// storage from an existing storage reference but does not expose host slices or
/// choose an execution backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamePlacementScratchAllocator;

impl SamePlacementScratchAllocator {
    #[inline]
    pub fn filled_like<T, S>(&self, storage: &S, len: usize, value: T) -> S::Similar
    where
        T: Clone,
        S: SimilarStorage<T>,
    {
        storage.similar_filled(len, value)
    }
}

/// Source/destination scratch shape for tree-transform replay.
///
/// The cached tree-transform structure remains placement-neutral; these slots
/// name the storage-local scratch buffers needed to replay source packs into
/// destination packs. Current host replay instantiates both slots with
/// `HostScratchBuffer<T>`.
#[derive(Clone, Debug)]
pub struct TreeTransformScratchBuffers<Source, Destination> {
    source: Source,
    destination: Destination,
}

impl<Source, Destination> Default for TreeTransformScratchBuffers<Source, Destination>
where
    Source: Default,
    Destination: Default,
{
    fn default() -> Self {
        Self {
            source: Source::default(),
            destination: Destination::default(),
        }
    }
}

impl<Source, Destination> TreeTransformScratchBuffers<Source, Destination> {
    #[inline]
    pub fn from_parts(source: Source, destination: Destination) -> Self {
        Self {
            source,
            destination,
        }
    }

    #[inline]
    pub fn source(&self) -> &Source {
        &self.source
    }

    #[inline]
    pub fn source_mut(&mut self) -> &mut Source {
        &mut self.source
    }

    #[inline]
    pub fn destination(&self) -> &Destination {
        &self.destination
    }

    #[inline]
    pub fn destination_mut(&mut self) -> &mut Destination {
        &mut self.destination
    }

    #[inline]
    pub fn source_and_destination_mut(&mut self) -> (&Source, &mut Destination) {
        (&self.source, &mut self.destination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtn_core::{Placement, TensorStorage};

    #[derive(Debug)]
    struct PlacementOnlyStorage<T> {
        len: usize,
        _marker: std::marker::PhantomData<T>,
    }

    #[derive(Debug, PartialEq)]
    struct PlacementOnlyScratch<T> {
        data: Vec<T>,
    }

    impl<T> PlacementOnlyStorage<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<T> TensorStorage<T> for PlacementOnlyStorage<T> {
        fn len(&self) -> usize {
            self.len
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    impl<T: Clone> SimilarStorage<T> for PlacementOnlyStorage<T> {
        type Similar = PlacementOnlyScratch<T>;

        fn similar_filled(&self, len: usize, value: T) -> Self::Similar
        where
            T: Clone,
        {
            PlacementOnlyScratch {
                data: vec![value; len],
            }
        }
    }

    impl<T> TensorStorage<T> for PlacementOnlyScratch<T> {
        fn len(&self) -> usize {
            self.data.len()
        }

        fn placement(&self) -> Placement {
            Placement::Host
        }
    }

    #[test]
    fn same_placement_allocator_uses_similar_storage_type_without_host_slices() {
        let storage = PlacementOnlyStorage::<i32>::new(2);
        let allocator = SamePlacementScratchAllocator;

        let scratch = allocator.filled_like(&storage, 3, 7);

        assert_eq!(scratch.data, vec![7, 7, 7]);
        assert_eq!(scratch.len(), 3);
        assert_eq!(scratch.placement(), storage.placement());
    }

    #[test]
    fn tree_transform_scratch_buffers_keep_source_and_destination_slots() {
        let buffers = TreeTransformScratchBuffers::from_parts(
            PlacementOnlyScratch { data: vec![1, 2] },
            PlacementOnlyScratch {
                data: vec![3, 4, 5],
            },
        );

        assert_eq!(buffers.source().data, vec![1, 2]);
        assert_eq!(buffers.destination().data, vec![3, 4, 5]);
        assert_eq!(buffers.source().len(), 2);
        assert_eq!(buffers.destination().len(), 3);
    }
}
