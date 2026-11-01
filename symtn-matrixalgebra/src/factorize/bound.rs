use super::*;

// ---------------------------------------------------------------------------
// Dynamic-rank representation.
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Test-only rank-erased operand: a complete dynamic fusion space plus its
/// flat data in the coupled-sector matrix layout. The rank is a runtime
/// property of the space, as on every dynamic entry these tests exercise.
#[derive(Clone, Debug)]
pub(crate) struct TestTensor<D> {
    space: Arc<DynamicFusionMapSpace>,
    dims: Vec<usize>,
    data: Vec<D>,
}

#[cfg(test)]
impl<D> TestTensor<D> {
    pub(crate) fn from_dyn(data: Vec<D>, space: DynamicFusionMapSpace) -> Result<Self, CoreError> {
        let required = space.structure().required_len()?;
        if data.len() != required {
            return Err(CoreError::DimensionMismatch {
                expected: required,
                actual: data.len(),
            });
        }
        // SectorLeg is the complete axis-space authority, including sectors
        // with no participating tree.
        let dims = space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(space.homspace().domain().legs())
            .map(|leg| {
                leg.degeneracies().iter().try_fold(0usize, |total, &dim| {
                    total
                        .checked_add(dim)
                        .ok_or(CoreError::ElementCountOverflow)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            space: Arc::new(space),
            dims,
            data,
        })
    }

    pub(crate) fn from_vec_with_fusion_space(
        data: Vec<D>,
        space: DynamicFusionMapSpace,
    ) -> Result<Self, CoreError> {
        Self::from_dyn(data, space)
    }

    pub(crate) fn from_block_fn_with_fusion_space<F>(
        space: DynamicFusionMapSpace,
        background: D,
        fill: F,
    ) -> Result<Self, CoreError>
    where
        D: Clone,
        F: FnMut(&BlockKey, &[usize]) -> D,
    {
        let len = space.required_len()?;
        let mut tensor = Self::from_dyn(vec![background; len], space)?;
        tensor.fill_block_elements(fill)?;
        Ok(tensor)
    }

    pub(crate) fn fusion_space(&self) -> &DynamicFusionMapSpace {
        &self.space
    }

    pub(crate) fn structure(&self) -> &Arc<BlockStructure> {
        self.space.structure()
    }

    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub(crate) fn data(&self) -> &[D] {
        &self.data
    }

    pub(crate) fn data_mut(&mut self) -> &mut [D] {
        &mut self.data
    }

    /// Fills every block element with `fill(key, indices)`, independent of
    /// the storage layout.
    pub(crate) fn fill_block_elements<F>(&mut self, mut fill: F) -> Result<(), CoreError>
    where
        F: FnMut(&BlockKey, &[usize]) -> D,
    {
        let structure = Arc::clone(self.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index)?;
            let shape = block.shape();
            let strides = block.strides();
            let count: usize = shape.iter().product();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = block.offset()
                    + indices
                        .iter()
                        .zip(strides)
                        .map(|(&index, &stride)| index * stride)
                        .sum::<usize>();
                self.data[position] = fill(block.key(), &indices);
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
/// Test operand plus the sole provider authority accepted by
/// provider-sensitive factorization and matrix-function APIs.
pub(crate) struct BoundTensorMapRef<'a, R, D> {
    pub(super) space: &'a BoundDynamicFusionMapSpace<R>,
    pub(super) tensor: &'a TestTensor<D>,
}

#[cfg(test)]
/// Owned test operand that retains the provider authority for its fusion
/// space.
#[derive(Clone, Debug)]
pub(crate) struct BoundTensorMap<R, D> {
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    pub(super) tensor: TestTensor<D>,
}

#[cfg(test)]
impl<R, D> BoundTensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols,
{
    pub fn try_new(provider: Arc<R>, tensor: TestTensor<D>) -> Result<Self, OperationError> {
        let space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
            tensor.fusion_space().clone(),
            provider,
        )?;
        BoundDynamicTensorRef::try_new(&space, tensor.data())?;
        Ok(Self { space, tensor })
    }

    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        &self.space
    }

    pub fn tensor(&self) -> &TestTensor<D> {
        &self.tensor
    }

    pub fn data(&self) -> &[D] {
        self.tensor.data()
    }

    pub fn as_ref(&self) -> BoundTensorMapRef<'_, R, D> {
        BoundTensorMapRef {
            space: &self.space,
            tensor: &self.tensor,
        }
    }

    pub fn into_parts(self) -> (BoundDynamicFusionMapSpace<R>, TestTensor<D>) {
        (self.space, self.tensor)
    }
}

#[cfg(test)]
impl<R, D> std::ops::Deref for BoundTensorMap<R, D> {
    type Target = TestTensor<D>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

#[cfg(test)]
impl<'a, R, D> BoundTensorMapRef<'a, R, D>
where
    R: MultiplicityFreeRigidSymbols,
{
    pub fn space(&self) -> &'a BoundDynamicFusionMapSpace<R> {
        self.space
    }

    pub fn tensor(&self) -> &'a TestTensor<D> {
        self.tensor
    }

    pub(crate) fn dynamic(&self) -> BoundDynamicTensorRef<'_, R, D> {
        // Why not expose an unchecked constructor: `BoundTensorMap::try_new`
        // establishes this invariant once, while external dynamic inputs must
        // continue through `BoundDynamicTensorRef::try_new`.
        BoundDynamicTensorRef::try_new(self.space, self.tensor.data())
            .expect("BoundTensorMap preserves its validated dynamic layout")
    }
}

/// Owned dynamic factor that retains the provider used to create its complete
/// fusion space.
pub struct BoundDynFactor<R, D> {
    pub(super) space: BoundDynamicFusionMapSpace<R>,
    pub(super) data: Vec<D>,
}

pub(super) type DynamicFactorPair<R, D> = (BoundDynFactor<R, D>, BoundDynFactor<R, D>);

impl<R, D> fmt::Debug for BoundDynFactor<R, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundDynFactor")
            .field("space", &self.space)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl<R, D> Clone for BoundDynFactor<R, D>
where
    D: Clone,
{
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            data: self.data.clone(),
        }
    }
}

impl<R, D> BoundDynFactor<R, D> {
    pub(crate) fn from_bound(
        space: BoundDynamicFusionMapSpace<R>,
        data: Vec<D>,
        expected_nout: usize,
        expected_nin: usize,
    ) -> Result<Self, OperationError> {
        if space.space().nout() != expected_nout || space.space().nin() != expected_nin {
            return Err(OperationError::RankMismatch {
                expected: expected_nout + expected_nin,
                actual: space.space().rank(),
            });
        }
        let expected = space
            .space()
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        if data.len() != expected {
            return Err(OperationError::from_core_preserving_context(
                CoreError::DimensionMismatch {
                    expected,
                    actual: data.len(),
                },
            ));
        }
        Ok(Self { space, data })
    }

    pub fn space(&self) -> &BoundDynamicFusionMapSpace<R> {
        &self.space
    }

    pub fn data(&self) -> &[D] {
        &self.data
    }

    #[cfg(test)]
    pub(crate) fn data_mut(&mut self) -> &mut [D] {
        &mut self.data
    }

    pub fn into_parts(self) -> (BoundDynamicFusionMapSpace<R>, Vec<D>) {
        (self.space, self.data)
    }
}

#[cfg(test)]
pub(crate) fn adjoint_bound_factor<R, D>(
    factor: &BoundDynFactor<R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let (space, data) = symtn_tensors::adjoint_bound_dyn(factor.space(), factor.data())?;
    let nout = space.space().nout();
    let nin = space.space().nin();
    BoundDynFactor::from_bound(space, data, nout, nin)
}

#[cfg(test)]
pub(crate) fn typed_from_bound_factor<R, D>(
    factor: BoundDynFactor<R, D>,
) -> Result<BoundTensorMap<R, D>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let (space, data) = factor.into_parts();
    let tensor = TestTensor::from_dyn(data, space.space().clone())
        .map_err(OperationError::from_core_preserving_context)?;
    Ok(BoundTensorMap { space, tensor })
}
