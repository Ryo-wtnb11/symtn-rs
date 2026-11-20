use super::*;

/// Inline storage for one-based outer-multiplicity vertex labels.
pub(crate) type MultiplicityVec = SmallVec<[MultiplicityIndex; 8]>;
/// Inline storage for `usize` metadata (dims, strides, indices, permutations).
pub type DimVec = SmallVec<[usize; 8]>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Placement {
    Host,
    /// Storage resident on a CUDA device, identified by its ordinal.
    Cuda(usize),
}

pub trait TensorStorage<T> {
    fn len(&self) -> usize;
    fn placement(&self) -> Placement;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Storage-backed scratch allocation matching the source storage placement.
///
/// This is the TensorKit `similar(data, len)` boundary in symtn-owned terms.
/// Implementations should allocate temporary storage with the same placement as
/// `self`; host storage returns host buffers, and future device storage should
/// return device-resident buffers.
pub trait SimilarStorage<T>: TensorStorage<T> {
    type Similar: TensorStorage<T>;

    fn similar_filled(&self, len: usize, value: T) -> Self::Similar
    where
        T: Clone;
}

pub trait HostReadableStorage<T>: TensorStorage<T> {
    fn as_slice(&self) -> &[T];
}

pub trait HostWritableStorage<T>: HostReadableStorage<T> {
    fn as_mut_slice(&mut self) -> &mut [T];
}

impl<T> TensorStorage<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> SimilarStorage<T> for Vec<T> {
    type Similar = Vec<T>;

    #[inline]
    fn similar_filled(&self, len: usize, value: T) -> Self::Similar
    where
        T: Clone,
    {
        vec![value; len]
    }
}

impl<T> HostReadableStorage<T> for Vec<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> HostWritableStorage<T> for Vec<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}
